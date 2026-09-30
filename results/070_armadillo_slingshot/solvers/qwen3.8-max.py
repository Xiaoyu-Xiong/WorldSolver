"""Physically-based solver for the Armadillo slingshot scene.

Volumetric corotated-linear-elastic tetrahedral FEM integrated with an
explicit symplectic-Euler scheme at small substeps.  The two palm grasp
patches are Dirichlet-fixed; a temporary kinematic pull is applied to the
lower-limb patch through stiff penalty springs whose target ramps upward
from ``target_start`` to ``target_peak`` and is fully removed at release.
After release the stored elastic energy snaps the body back and it
overshoots its earlier position while the grasp points remain held.

All required raw trajectory fields (mass, velocity, deformation gradients,
per-domain momentum budgets) are exported per sample frame.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_ROOT = Path(__file__).resolve().parents[1]
_SOLID = _ROOT / "assets" / "armadillo_solid.npz"


class BenchmarkSolver:
    # Tuned numerical material (config material_reference is metadata only).
    E = 8.0e4          # Pa
    NU = 0.35          # -
    RHO = 250.0        # kg/m^3
    DT_SUB = 0.0008    # s, explicit substep (CFL-safe for this mesh/material)
    K_SPRING = 2.0e4   # N/m per pulled node
    DAMP = 0.30        # global velocity damping, 1/s
    VMAX = 120.0       # m/s safety clamp

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        init = scene.initial_state()
        self.X0 = np.array(init.position, dtype=np.float64)
        self.n = len(self.X0)
        self.x = self.X0.copy()
        self.v = np.zeros_like(self.X0)
        self.t = 0.0
        self.nsteps = 0

        data = np.load(_SOLID)
        tets = np.array(data["tetrahedra"], dtype=np.int64)

        # Attributes from the frozen scene definition.
        self.fixed = np.asarray(init.attributes["fixed_grasp"], dtype=np.float64) > 0.5
        self.pulled = np.asarray(init.attributes["pulled_patch"], dtype=np.float64) > 0.5
        self.fixed_attr = np.asarray(init.attributes["fixed_grasp"], dtype=np.float64)
        self.pulled_attr = np.asarray(init.attributes["pulled_patch"], dtype=np.float64)

        # Orient tets positively and precompute material operators.
        p0 = tets[:, 0]
        Dm = np.stack((self.X0[tets[:, 1]] - self.X0[p0],
                       self.X0[tets[:, 2]] - self.X0[p0],
                       self.X0[tets[:, 3]] - self.X0[p0]), axis=2)
        det = np.linalg.det(Dm)
        bad = det <= 0.0
        if np.any(bad):
            tets[bad] = tets[bad][:, [0, 2, 1, 3]]
            Dm[bad] = Dm[bad][:, :, [1, 0, 2]]
            det = np.linalg.det(Dm)
        self.tets = tets
        self.V0 = det / 6.0
        self.Bm = np.linalg.inv(Dm)
        self.BmT = np.transpose(self.Bm, (0, 2, 1))
        self.ne = len(tets)

        # Lumped mass.
        m = np.zeros(self.n)
        contrib = self.RHO * self.V0 * 0.25
        for j in range(4):
            np.add.at(m, tets[:, j], contrib)
        m = np.maximum(m, 1e-6)
        self.m = m
        self.inv_m = 1.0 / m
        self.M_total = float(m.sum())

        # Elastic constants.
        self.mu = self.E / (2.0 * (1.0 + self.NU))
        self.lam = self.E * self.NU / ((1.0 + self.NU) * (1.0 - 2.0 * self.NU))

        # Loading / events.
        g = np.asarray(cfg["gravity"], dtype=np.float64)
        self.g = g
        bc = cfg["boundary_conditions"]
        tp = bc["temporary_pull"]
        tl = cfg["event_timeline"]
        self.z_start = float(tp["target_start"][2])
        self.z_peak = float(tp["target_peak"][2])
        self.pull_t0 = float(tp["active_interval_s"][0])
        self.pull_peak_t = float(tl["pull_peak_s"])
        self.release_t = float(tp["active_interval_s"][1])
        self.zc0 = float(self.X0[self.pulled, 2].mean())

        idx = np.flatnonzero(self.pulled)
        self.pull_idx = idx
        self.k_node = np.full(len(idx), self.K_SPRING)
        self.c_node = 1.2 * np.sqrt(self.k_node * m[idx])
        self.pull_targets0 = self.X0[idx].copy()

        # Momentum-budget domains: row 0 whole body, row 1 pulled patch.
        self.G = 2
        self.W = np.zeros((self.G, self.n))
        self.W[0, :] = 1.0
        self.W[1, self.pulled] = 1.0
        self.impulse = np.zeros((self.G, 3))
        self.inflow = np.zeros((self.G, 3))
        self.outflow = np.zeros((self.G, 3))

        # Frozen positive momentum scale (not the initial net momentum).
        span = float(self.X0[:, 2].max() - self.X0[:, 2].min())
        v_ref = np.sqrt(2.0 * np.linalg.norm(g) * max(span, 1e-3))
        self.reference_momentum = self.M_total * v_ref

        self._R_prev = np.tile(np.eye(3), (self.ne, 1, 1))

    # ------------------------------------------------------------- mechanics
    def _deformation(self):
        t = self.tets
        p0 = t[:, 0]
        Ds = np.stack((self.x[t[:, 1]] - self.x[p0],
                       self.x[t[:, 2]] - self.x[p0],
                       self.x[t[:, 3]] - self.x[p0]), axis=2)
        F = Ds @ self.Bm
        return F

    def _corotated(self, F):
        """Corotated PK1 stress and internal nodal forces; returns (f_int, F)."""
        try:
            U, _, Vh = np.linalg.svd(F)
            d = np.sign(np.linalg.det(U @ Vh))
            d[d == 0.0] = 1.0
            D = np.zeros_like(F)
            D[:, 0, 0] = 1.0
            D[:, 1, 1] = 1.0
            D[:, 2, 2] = d
            R = U @ D @ Vh
            self._R_prev = R
        except np.linalg.LinAlgError:
            R = self._R_prev
        J = np.linalg.det(F)
        Jc = np.clip(J, 0.15, None)
        # adj(F): rows are cross products of the other two rows of F.
        r0, r1, r2 = F[:, 0, :], F[:, 1, :], F[:, 2, :]
        adj = np.stack((np.cross(r1, r2), np.cross(r2, r0), np.cross(r0, r1)), axis=1)
        # P = 2 mu (F - R) + lam (J - 1) J F^{-T},  J F^{-T} = adj(F)^T
        P = 2.0 * self.mu * (F - R) + (self.lam * (Jc - 1.0))[:, None, None] * np.transpose(adj, (0, 2, 1))
        H = self.V0[:, None, None] * (P @ self.BmT)
        f_int = np.zeros((self.n, 3))
        t = self.tets
        np.add.at(f_int, t[:, 1], -H[:, :, 0])
        np.add.at(f_int, t[:, 2], -H[:, :, 1])
        np.add.at(f_int, t[:, 3], -H[:, :, 2])
        np.add.at(f_int, t[:, 0], H[:, :, 0] + H[:, :, 1] + H[:, :, 2])
        return f_int, F

    def _pull_shift(self, time: float) -> float:
        if time >= self.release_t:
            return 0.0
        span = max(self.pull_peak_t - self.pull_t0, 1e-9)
        s = min(max((time - self.pull_t0) / span, 0.0), 1.0)
        z_t = self.z_start + (self.z_peak - self.z_start) * s
        return z_t - self.zc0

    def _spring_force(self, time: float) -> np.ndarray:
        f = np.zeros((self.n, 3))
        if time >= self.release_t or len(self.pull_idx) == 0:
            return f
        s = min(time / 0.12, 1.0) ** 2
        dz = self._pull_shift(time)
        tgt = self.pull_targets0
        idx = self.pull_idx
        d = tgt - self.x[idx]
        d[:, 2] += dz
        f[idx] = s * (self.k_node[:, None] * d - self.c_node[:, None] * self.v[idx])
        return f

    def _external_terms(self, time: float, f_int: np.ndarray):
        """Return per-node (gravity, pull, reaction) external force terms."""
        f_grav = self.m[:, None] * self.g[None, :]
        f_spr = self._spring_force(time)
        f_tot = f_int + f_grav + f_spr
        reac = np.zeros_like(f_tot)
        reac[self.fixed] = -f_tot[self.fixed]
        return f_grav, f_spr, reac

    # --------------------------------------------------------------- stepping
    def _substep(self, dt: float) -> None:
        time = self.t
        f_int, _ = self._corotated(self._deformation())
        f_grav, f_spr, reac = self._external_terms(time, f_int)

        f_tot = f_int + f_grav + f_spr
        f_tot[self.fixed] = 0.0

        self.v += (f_tot * self.inv_m[:, None]) * dt
        self.v[self.fixed] = 0.0
        self.v *= 1.0 / (1.0 + self.DAMP * dt)
        sp = np.linalg.norm(self.v, axis=1)
        over = sp > self.VMAX
        if np.any(over):
            self.v[over] *= (self.VMAX / sp[over])[:, None]
        self.x += self.v * dt
        self.x[self.fixed] = self.X0[self.fixed]

        # Cumulative external impulse per budget domain (trapezoid-free, force
        # evaluated at substep start, consistent with symplectic Euler).
        terms = (f_grav, f_spr, reac)
        for f_term in terms:
            self.impulse += dt * (self.W @ f_term)

        self.t = time + dt
        self.nsteps += 1

    # ------------------------------------------------------------------ frame
    def _make_frame(self, time: float) -> StateFrame:
        f_int, F = self._corotated(self._deformation())
        f_grav, f_spr, reac = self._external_terms(time, f_int)

        budget_force = np.zeros((self.G, 3, 3))  # [G, K, 3]
        for k, term in enumerate((f_grav, f_spr, reac)):
            budget_force[:, k, :] = self.W @ term
        f_ext_node = f_grav + f_spr + reac

        frame = StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            mass=self.m.copy(),
            attributes={
                "fixed_grasp": self.fixed_attr.copy(),
                "pulled_patch": self.pulled_attr.copy(),
                "mass_quantity": self.m.copy(),
            },
            fields={
                "element_deformation_gradient": F,
                "external_force": f_ext_node,
                "budget_domain_weights": self.W.copy(),
                "budget_external_force": budget_force,
                "budget_cumulative_external_impulse": self.impulse.copy(),
                "budget_cumulative_inflow_momentum": self.inflow.copy(),
                "budget_cumulative_outflow_momentum": self.outflow.copy(),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "reference_momentum": self.reference_momentum,
                "density_kg_m3": self.RHO,
                "youngs_modulus_pa": self.E,
                "poisson_ratio": self.NU,
                "total_mass": self.M_total,
            },
        )
        frame.validate()
        return frame

    # ----------------------------------------------------------------- public
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        guard = 0
        while self.t < time - 1e-12 and guard < 2_000_000:
            dt = min(self.DT_SUB, time - self.t)
            self._substep(dt)
            guard += 1
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        ke = 0.5 * float(np.sum(self.m[:, None] * self.v ** 2))
        return {
            "substeps": self.nsteps,
            "time": self.t,
            "kinetic_energy_J": ke,
            "total_mass_kg": self.M_total,
            "model": "corotated linear tetrahedral FEM, explicit substepping",
        }
