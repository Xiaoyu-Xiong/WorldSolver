"""Physically-based solver for the pneumatic cushion inflation scene.

Quasi-static nonlinear finite-element solution of a nearly-incompressible
hyperelastic (compressible neo-Hookean) solid wall sealed around an air
chamber.  A prescribed uniform internal pressure acts as a follower
traction on the inner chamber surface and is ramped from 0 to the target
value; equilibrium is found with Newton-Raphson and sparse direct solves.
Soft anchoring springs remove rigid-body modes without restraining the
face inflation (net spring force is reported as an external budget term).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        mat = cfg["material_properties"]
        E = float(mat["young_modulus_pa"])
        nu = float(mat["poisson_ratio"])
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.rho = float(mat["density_kg_m3"])

        load = cfg["loading"]
        self.p_target = float(load["target_internal_pressure_pa"])
        self.p_initial = float(load["initial_internal_pressure_pa"])
        self.p_ext = float(load["external_pressure_pa"])
        self.ramp_start = float(load["ramp_start_s"])
        self.ramp_end = float(load["ramp_end_s"])
        self.hold_end = float(load.get("hold_end_s", cfg["duration"]))

        asset = Path(__file__).resolve().parents[1] / "assets" / "cushion_solid.npz"
        with np.load(asset) as data:
            self.X0 = np.asarray(data["vertices"], dtype=np.float64).copy()
            self.tets = np.asarray(data["tetrahedra"], dtype=np.int64).copy()
            self.inner_faces = np.asarray(data["inner_chamber_faces"], dtype=np.int64).copy()
            self.nodal_volume = np.asarray(data["nodal_volume"], dtype=np.float64).copy()
            self.wall_region = np.asarray(data["wall_region"], dtype=np.int32).copy()

        self.n_nodes = len(self.X0)
        self.n_tets = len(self.tets)
        self.mass = self.rho * self.nodal_volume

        # Reference tet geometry: shape-function gradients G [Ne,4,3] and volumes.
        p0 = self.X0[self.tets[:, 0]]
        E0 = self.X0[self.tets[:, 1:]] - p0[:, None, :]          # rows = edges
        det = np.linalg.det(E0)
        self.V0 = np.abs(det) / 6.0
        Minv = np.linalg.inv(E0)                                  # [Ne,3,3]
        self.G = np.empty((self.n_tets, 4, 3))
        self.G[:, 1:, :] = np.transpose(Minv, (0, 2, 1))          # G_m = column m of E0^-1
        self.G[:, 0, :] = -self.G[:, 1:, :].sum(axis=1)
        # DOF index table for assembly (frozen element order).
        self.dofs12 = (self.tets[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(self.n_tets, 12)
        self.rows = np.broadcast_to(self.dofs12[:, :, None], (self.n_tets, 12, 12)).ravel()
        self.cols = np.broadcast_to(self.dofs12[:, None, :], (self.n_tets, 12, 12)).ravel()
        self.eye3 = np.eye(3)

        # Orient inner chamber face normals to point out of the cavity
        # (away from the chamber center at the origin): pressure pushes the
        # wall away from the cavity interior.
        fa, fb, fc = (self.X0[self.inner_faces[:, k]] for k in range(3))
        A0 = 0.5 * np.cross(fb - fa, fc - fa)
        cen = (fa + fb + fc) / 3.0
        self.face_sign = np.sign(np.einsum("ij,ij->i", A0, cen))
        self.face_sign[self.face_sign == 0.0] = 1.0

        # Soft rigid-body-mode regularization (does not restrain bulging).
        self.k_spring = 1.0  # N/m per node
        self.spring_rows = np.arange(3 * self.n_nodes)

        # Reference force scale: target pressure over the broadest face area.
        dims = np.asarray(cfg["geometry"]["outer_dimensions_m"], dtype=float)
        self.reference_force = float(self.p_target * dims[0] * dims[1])

        # State.
        self.x = self.X0.copy()
        self.p_cur = self.p_initial
        self.prev_x = self.X0.copy()
        self.prev_t = 0.0
        self.time = 0.0
        self.n_inc = 1
        self._total_iters = 0
        self._last_converged = True

    # ------------------------------------------------------------------
    def _pressure(self, t: float) -> float:
        if self.ramp_end <= self.ramp_start:
            return self.p_target if t >= self.ramp_end else self.p_initial
        s = (t - self.ramp_start) / (self.ramp_end - self.ramp_start)
        s = min(max(s, 0.0), 1.0)
        return self.p_initial + s * (self.p_target - self.p_initial)

    # ------------------------------------------------------------------
    def _pressure_force(self, x, p_net):
        f = np.zeros((self.n_nodes, 3))
        if p_net == 0.0 or len(self.inner_faces) == 0:
            return f
        a = x[self.inner_faces[:, 0]]
        b = x[self.inner_faces[:, 1]]
        c = x[self.inner_faces[:, 2]]
        av = 0.5 * np.cross(b - a, c - a) * self.face_sign[:, None]
        contrib = (p_net / 3.0) * av
        for k in range(3):
            np.add.at(f, self.inner_faces[:, k], contrib)
        return f

    def _deformation(self, x):
        p0 = x[self.tets[:, 0]]
        e = x[self.tets[:, 1:]] - p0[:, None, :]
        # F = sum_a x_a outer G_a  (exact for the linear tet basis)
        F = np.einsum("nak,nal->nkl", x[self.tets], self.G)
        return F

    def _internal_force(self, x, F):
        J = np.linalg.det(F)
        J = np.maximum(J, 1e-10)
        lnJ = np.log(J)
        FinvT = np.linalg.inv(F).transpose(0, 2, 1)
        P = self.mu * (F - FinvT) + (self.lam * lnJ)[:, None, None] * FinvT
        contrib = self.V0[:, None, None] * np.einsum("nil,nal->nai", P, self.G)
        f = np.zeros((self.n_nodes, 3))
        np.add.at(f, self.tets, contrib)
        return f, P, FinvT, lnJ

    def _assemble_K(self, F, FinvT, lnJ):
        B = FinvT
        Ct = (self.mu * np.einsum("ij,lm->iljm", self.eye3, self.eye3)[None]
              + self.lam * np.einsum("nil,njm->niljm", B, B)
              - ((self.lam * lnJ - self.mu)[:, None, None, None, None]
                 * np.einsum("nml,nji->niljm", B, B)))
        K_local = self.V0[:, None, None, None, None] * np.einsum(
            "niljm,nal,nbm->nabij", Ct, self.G, self.G)
        vals = K_local.ravel()
        rows = np.concatenate([self.rows, self.spring_rows])
        cols = np.concatenate([self.cols, self.spring_rows])
        v2 = np.concatenate([vals, np.full(3 * self.n_nodes, self.k_spring)])
        K = sparse.coo_matrix((v2, (rows, cols)),
                              shape=(3 * self.n_nodes, 3 * self.n_nodes)).tocsc()
        return K

    def _newton(self, x, p_net, max_it=40, tol=1e-7):
        fext_scale = 1.0
        for it in range(max_it):
            F = self._deformation(x)
            fint, P, FinvT, lnJ = self._internal_force(x, F)
            fpress = self._pressure_force(x, p_net)
            fspr = -self.k_spring * (x - self.X0)
            R = fpress + fspr - fint
            rmax = np.max(np.abs(R))
            fext_scale = max(1.0, np.max(np.abs(fpress)) if p_net != 0.0 else 1.0)
            if rmax < tol * fext_scale:
                self._total_iters += it
                return x, True
            K = self._assemble_K(F, FinvT, lnJ)
            try:
                dx = splu(K).solve(R.ravel())
            except Exception:
                self._total_iters += it
                return x, False
            if not np.all(np.isfinite(dx)):
                self._total_iters += it
                return x, False
            step = np.max(np.abs(dx))
            if step > 0.2:  # simple line-search safeguard
                dx *= 0.2 / step
            x = x + dx.reshape(-1, 3)
        self._total_iters += max_it
        F = self._deformation(x)
        fint, _, _, _ = self._internal_force(x, F)
        R = self._pressure_force(x, p_net) - self.k_spring * (x - self.X0) - fint
        return x, bool(np.max(np.abs(R)) < 1e-4 * fext_scale)

    def _solve_to_pressure(self, p_to):
        p_from = self.p_cur
        x_start = self.x.copy()
        for attempt in range(3):
            n = max(1, self.n_inc * (2 ** attempt))
            ok = True
            xt = x_start.copy()
            for i in range(1, n + 1):
                p_i = p_from + (p_to - p_from) * i / n
                xt, conv = self._newton(xt, p_i - self.p_ext)
                if not conv or not np.all(np.isfinite(xt)):
                    ok = False
                    break
            if ok:
                self.n_inc = n
                self._last_converged = True
                self.x = xt
                self.p_cur = p_to
                return
        self._last_converged = False
        self.x = xt if np.all(np.isfinite(xt)) else x_start
        self.p_cur = p_to

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        time = float(time)
        dt = time - self.prev_t
        p_to = self._pressure(time)
        if abs(p_to - self.p_cur) > 0.0 or time <= self.prev_t:
            self._solve_to_pressure(p_to)
        else:
            self._solve_to_pressure(p_to)
        if dt > 0.0:
            v = (self.x - self.prev_x) / dt
        else:
            v = np.zeros_like(self.x)
        self.prev_x = self.x.copy()
        self.prev_t = time
        self.time = time
        return self._build_frame(time, v)

    def _build_frame(self, time, v):
        F = self._deformation(self.x)
        p_net = self.p_cur - self.p_ext
        fpress = self._pressure_force(self.x, p_net)
        fspr = -self.k_spring * (self.x - self.X0)
        budget = np.zeros((1, 2, 3))
        budget[0, 0] = fpress.sum(axis=0)   # prescribed internal-pressure traction
        budget[0, 1] = fspr.sum(axis=0)     # rigid-body-mode constraint springs
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=np.asarray(v, dtype=np.float64),
            mass=self.mass.copy(),
            attributes={
                "wall_region": self.wall_region.copy(),
                "mass_quantity": self.mass[:, None].copy(),
            },
            fields={
                "budget_external_force": budget,
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": F,
            },
            scalars={
                "reference_force": float(self.reference_force),
                "internal_pressure_pa": float(self.p_cur),
                "external_pressure_pa": float(self.p_ext),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        disp = np.max(np.linalg.norm(self.x - self.X0, axis=1)) if hasattr(self, "x") else 0.0
        return {
            "pressure_pa": getattr(self, "p_cur", 0.0),
            "max_displacement_m": float(disp),
            "newton_iterations": int(getattr(self, "_total_iters", 0)),
            "last_step_converged": bool(getattr(self, "_last_converged", True)),
        }
