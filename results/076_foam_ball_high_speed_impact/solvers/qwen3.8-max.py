"""Physically-based solver for the high-speed foam ball impact scene.

Volumetric co-rotational linear-elastic FEM on a Delaunay tetrahedral mesh of
the ball samples, integrated explicitly (semi-implicit Euler) with a penalty
contact model against the fixed plate surface at x = plate_x.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------ setup ------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        frame0 = scene.initial_state()
        meta = scene.metadata()
        cfg = getattr(scene, "config", None) or {}

        E = float(meta.get("youngs_modulus", 1.0e7))
        nu = float(meta.get("poisson_ratio", 0.45))
        rho = float(meta.get("density", 1150.0))

        grav = [0.0, 0.0, -9.81]
        plane_x = 0.0
        if isinstance(cfg, dict):
            ic = cfg.get("initial_conditions") or {}
            if ic.get("gravity") is not None:
                grav = list(ic["gravity"])
            bc = cfg.get("boundary_conditions") or {}
            plane_x = float(bc.get("plate_contact_surface_x", 0.0))
        self.gravity = np.asarray(grav, dtype=np.float64)
        self.plane_x = plane_x

        self.x = np.array(frame0.position, dtype=np.float64)
        self.v = np.array(frame0.velocity, dtype=np.float64)
        self.rest = np.array(frame0.attributes["rest_position"], dtype=np.float64)
        n = len(self.x)

        scene_mass = np.asarray(frame0.mass, dtype=np.float64)
        self.total_mass = float(scene_mass.sum())
        v0 = self.v[0]
        self.reference_momentum = float(self.total_mass * np.linalg.norm(v0))
        if not np.isfinite(self.reference_momentum) or self.reference_momentum <= 0:
            self.reference_momentum = 1.0

        # ---- tet mesh from rest positions ----
        tets = Delaunay(self.rest).simplices
        Dm, V0 = self._elem_matrices(self.rest, tets)
        keep = V0 > 0.02 * np.median(V0)
        tets = tets[keep]
        Dm, V0 = self._elem_matrices(self.rest, tets)
        self.tets = tets
        self.V0 = V0
        self.Bm = np.linalg.inv(Dm)  # (Ne,3,3)
        self.ne = len(tets)

        # flattened node indices for fast force scatter
        self.nodes_flat = tets.ravel()

        # ---- lumped masses (scaled to the declared total mass) ----
        m = np.zeros(n)
        np.add.at(m, self.nodes_flat, np.repeat(rho * V0 / 4.0, 4))
        m *= self.total_mass / m.sum()
        self.m = m
        self.inv_m = 1.0 / m

        # ---- material ----
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # ---- stable explicit time step ----
        c_wave = np.sqrt((self.lam + 2.0 * self.mu) / rho)
        p = self.rest[tets]  # (Ne,4,3)
        edges = np.concatenate([
            p[:, 1:] - p[:, :1],
            (p[:, 2] - p[:, 1])[:, None],
            (p[:, 3] - p[:, 1])[:, None],
            (p[:, 3] - p[:, 2])[:, None],
        ], axis=1)  # (Ne,6,3)
        h_min = float(np.linalg.norm(edges, axis=2).min())
        dt_elastic = 0.35 * h_min / c_wave

        self.kp = 2.0e6                     # normal penalty stiffness [N/m]
        self.cp = 0.4 * np.sqrt(self.kp * m)  # per-node contact damping
        self.mu_friction = 0.3
        dt_contact = 0.8 * 2.0 * np.sqrt(float(m.min()) / self.kp)
        self.dt = float(min(dt_elastic, dt_contact))

        self.drag = 10.0  # very light global damping (1/s)

        self.t = 0.0
        self.cum_impulse = np.zeros(3)
        self.weights = np.ones((1, n))
        self.zeros_g3 = np.zeros((1, 3))
        self.zeros_k = np.zeros(1)

        # prime external-force bookkeeping at t = 0
        self._ext_terms = np.zeros((3, 3))
        self._accumulate_forces(0.0)

    @staticmethod
    def _elem_matrices(pos, tets):
        p0 = pos[tets[:, 0]]
        Dm = np.stack([pos[tets[:, i + 1]] - p0 for i in range(3)], axis=-1)
        V0 = np.abs(np.linalg.det(Dm)) / 6.0
        return Dm, V0

    # ------------------------------ physics ------------------------------
    def _deformation_gradients(self):
        x = self.x
        d0 = x[self.tets[:, 0]]
        Ds = np.stack([x[self.tets[:, i + 1]] - d0 for i in range(3)], axis=-1)
        return Ds @ self.Bm

    def _accumulate_forces(self, dt_sub: float) -> np.ndarray:
        """Total nodal force (internal + gravity + contact) as [N,3]."""
        F = self._deformation_gradients()
        U, _S, Vt = np.linalg.svd(F)
        R = U @ Vt
        d = np.linalg.det(R)
        sgn = np.where(d < 0.0, -1.0, 1.0)
        U[:, :, 2] *= sgn[:, None]
        R = U @ Vt
        RtF = np.matmul(np.transpose(R, (0, 2, 1)), F)
        eps = 0.5 * (RtF + np.transpose(RtF, (0, 2, 1)))
        eps[:, 0, 0] -= 1.0
        eps[:, 1, 1] -= 1.0
        eps[:, 2, 2] -= 1.0
        tr = eps[:, 0, 0] + eps[:, 1, 1] + eps[:, 2, 2]
        sig = 2.0 * self.mu * eps
        sig[:, 0, 0] += self.lam * tr
        sig[:, 1, 1] += self.lam * tr
        sig[:, 2, 2] += self.lam * tr
        P = np.matmul(R, sig)  # first Piola-Kirchhoff
        H = np.matmul(self.V0[:, None, None] * P, np.transpose(self.Bm, (0, 2, 1)))

        ftet = np.empty((self.ne, 4, 3))
        ftet[:, 1:, :] = -np.transpose(H, (0, 2, 1))
        ftet[:, 0, :] = H.sum(axis=1)

        f = np.zeros((len(self.x), 3))
        flat = ftet.reshape(-1, 3)
        for k in range(3):
            f[:, k] = np.bincount(self.nodes_flat, weights=flat[:, k],
                                  minlength=len(self.x))

        # gravity
        f_grav_total = self.m.sum() * self.gravity
        f += self.m[:, None] * self.gravity[None, :]

        # penalty contact with the fixed plate plane (normal +x)
        pen = self.plane_x - self.x[:, 0]
        idx = np.flatnonzero(pen > 0.0)
        f_n_total = np.zeros(3)
        f_t_total = np.zeros(3)
        if len(idx):
            fn = self.kp[idx] * pen[idx] if np.ndim(self.kp) else self.kp * pen[idx]
            vn = self.v[idx, 0]
            cn = self.cp[idx]
            fn = fn - cn * np.minimum(vn, 0.0)
            f[idx, 0] += fn
            f_n_total[0] = fn.sum()
            # Coulomb friction against tangential slip
            vt = self.v[idx, 1:]
            ft = -cn[:, None] * vt
            mag = np.linalg.norm(ft, axis=1)
            cap = self.mu_friction * np.maximum(fn, 0.0)
            over = mag > cap
            if over.any():
                scale = np.zeros_like(mag)
                good = ~over
                scale[good] = 1.0
                scale[over] = cap[over] / np.maximum(mag[over], 1e-30)
                ft *= scale[:, None]
            f[idx, 1] += ft[:, 0]
            f[idx, 2] += ft[:, 1]
            f_t_total[1] = ft[:, 0].sum()
            f_t_total[2] = ft[:, 1].sum()

        # external-force budget terms: [gravity, contact normal, contact friction]
        self._ext_terms = np.stack([f_grav_total, f_n_total, f_t_total])
        self.cum_impulse += (f_grav_total + f_n_total + f_t_total) * dt_sub
        return f

    def _step(self, dt_sub: float) -> None:
        f = self._accumulate_forces(dt_sub)
        self.v += (f * self.inv_m[:, None]) * dt_sub
        self.v *= (1.0 - self.drag * dt_sub)
        self.x += self.v * dt_sub

    # ------------------------------ output ------------------------------
    def _frame(self, time: float) -> StateFrame:
        F = self._deformation_gradients()
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            attributes={
                "rest_position": self.rest.copy(),
                "mass_quantity": self.m.copy(),
            },
            fields={
                "element_deformation_gradient": F,
                "budget_domain_weights": self.weights.copy(),
                "budget_external_force": self._ext_terms[None, :, :].copy(),
                "budget_cumulative_external_impulse": self.cum_impulse[None, :].copy(),
                "budget_cumulative_inflow_momentum": self.zeros_g3.copy(),
                "budget_cumulative_outflow_momentum": self.zeros_g3.copy(),
                "cumulative_mass_source": self.zeros_k.copy(),
            },
            scalars={
                "reference_momentum": self.reference_momentum,
                "total_mass": self.total_mass,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        while self.t < target - 1e-12 and guard < 5_000_000:
            dt_sub = min(self.dt, target - self.t)
            self._step(dt_sub)
            self.t += dt_sub
            guard += 1
        return self._frame(max(self.t, target) if self.t >= target - 1e-9 else self.t)

    def diagnostics(self) -> dict:
        return {
            "time": self.t,
            "dt": self.dt,
            "n_nodes": int(len(self.m)),
            "n_elements": int(self.ne),
            "reference_momentum": self.reference_momentum,
        }
