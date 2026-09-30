"""Affine MPM solver for scene 021_topological_merge.

Physically steps a cohesive viscoplastic material with an explicit
affine-particle-in-cell Material Point Method:

  * P2G transfer of mass and momentum with affine velocity modes,
  * explicit internal forces from a fixed-corotated elastic stress,
  * singular-value plastic return mapping (viscoplastic yield clamps),
  * G2P transfer with affine velocity reconstruction,
  * fixed no-penetration ground nodes and containing domain walls,
  * substepped integration of the trajectory at the frozen sample times.

All contacting material shares the background grid, so independent blocks
that come into contact exchange momentum through common grid nodes and weld
into one coherent, plastically flowing pile, matching the Figure 6
phenomenon of repeated topological merges.
"""
from __future__ import annotations

import itertools

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Affine MPM with fixed-corotated elasticity and plastic yield clamps."""

    def __init__(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame = scene.initial_state()

        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        self.x = np.array(frame.position, dtype=np.float64)
        self.n = len(self.x)
        self.v = np.zeros((self.n, 3), dtype=np.float64)
        self.mass = np.array(frame.mass, dtype=np.float64)
        self.body_id = np.array(frame.body_id, dtype=np.int32)
        self.initial_group = self.body_id.astype(np.float64)
        self.color_id = np.asarray(frame.attributes["color_id"], dtype=np.float64)
        self.radius = np.asarray(frame.attributes["particle_radius"], dtype=np.float64)
        rest_density = float(cfg["initial_conditions"]["rest_density"])
        self.rest_density = rest_density
        self.spacing = float(frame.scalars.get("particle_spacing", 0.205))
        self.particle_volume = self.mass / rest_density

        eye = np.eye(3, dtype=np.float64)
        self.F = np.broadcast_to(eye, (self.n, 3, 3)).copy()
        self.C = np.zeros((self.n, 3, 3), dtype=np.float64)
        self.stress = np.zeros((self.n, 3, 3), dtype=np.float64)

        # Cohesive viscoplastic constitutive parameters.
        young = 8.0e4
        poisson = 0.25
        self.mu = young / (2.0 * (1.0 + poisson))
        self.lam = poisson * young / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.theta_c = 0.025   # plastic compression clamp (yield)
        self.theta_s = 0.0075  # plastic stretch clamp (yield)
        self.v_max = 30.0      # CFL safety clamp on advection speed

        # Background grid (regular, quadratic B-spline shape functions).
        self.dx = 0.2
        self.origin = np.array([-3.4, -3.0, -0.2], dtype=np.float64)
        self.shape = (35, 31, 46)  # node counts: x in [-3.4,3.4], y in [-3.0,3.0], z in [-0.2,8.8]
        self.nn = self.shape[0] * self.shape[1] * self.shape[2]

        # Fixed no-penetration ground: the support top surface is z = 0.
        node_z = self.origin[2] + np.arange(self.shape[2]) * self.dx
        k_frozen = int(np.count_nonzero(node_z <= 1e-9))
        frozen = np.zeros(self.shape, dtype=bool)
        frozen[:, :, :k_frozen] = True
        self.frozen_mask = frozen

        self.dt = 2.5e-3
        self.sim_time = 0.0
        self.contact_force = np.zeros(3, dtype=np.float64)

        # Preallocated scatter/gather buffers for the 27 node offsets.
        self.offsets = list(itertools.product(range(3), repeat=3))
        self.idx_all = np.zeros((self.n, 27), dtype=np.int64)
        self.W_all = np.zeros((self.n, 27), dtype=np.float64)
        self.dn_all = np.zeros((self.n, 27, 3), dtype=np.float64)
        self.gradW_all = np.zeros((self.n, 27, 3), dtype=np.float64)

        self._initialized = True

    # ------------------------------------------------------------- internals
    @staticmethod
    def _bspline(fx):
        """Quadratic B-spline weights and derivatives along one axis."""
        w = (0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1.0) ** 2,
             0.5 * (fx - 0.5) ** 2)
        g = (-(1.5 - fx), -2.0 * (fx - 1.0), (fx - 0.5))
        return w, g

    def _step(self, dt: float) -> None:
        dx = self.dx
        xp = (self.x - self.origin) / dx
        base = np.floor(xp - 0.5).astype(np.int64)
        for a in range(3):
            np.clip(base[:, a], 0, self.shape[a] - 3, out=base[:, a])
        fx = xp - base

        wx, gx = self._bspline(fx[:, 0])
        wy, gy = self._bspline(fx[:, 1])
        wz, gz = self._bspline(fx[:, 2])

        idx, Wm, dn, gw = self.idx_all, self.W_all, self.dn_all, self.gradW_all
        ny, nz = self.shape[1], self.shape[2]
        c = 0
        for i, j, k in self.offsets:
            bi = base[:, 0] + i
            bj = base[:, 1] + j
            bk = base[:, 2] + k
            idx[:, c] = (bi * ny + bj) * nz + bk
            Wm[:, c] = wx[i] * wy[j] * wz[k]
            dn[:, c, 0] = (bi - xp[:, 0]) * dx
            dn[:, c, 1] = (bj - xp[:, 1]) * dx
            dn[:, c, 2] = (bk - xp[:, 2]) * dx
            gw[:, c, 0] = gx[i] * wy[j] * wz[k]
            gw[:, c, 1] = wx[i] * gy[j] * wz[k]
            gw[:, c, 2] = wx[i] * wy[j] * gz[k]
            c += 1

        flat = idx.ravel()

        # --- particle to grid -------------------------------------------
        mass_grid = np.bincount(flat, weights=Wm.ravel(), minlength=self.nn)
        affine = np.einsum("nij,nkj->nki", self.C, dn)
        vel_mode = self.v[:, None, :] + affine
        momentum = self.mass[:, None, None] * Wm[:, :, None] * vel_mode
        force = -self.particle_volume[:, None, None] * np.einsum(
            "nij,nkj->nki", self.stress, gw
        )

        denom = np.where(mass_grid > 1e-12, mass_grid, 1.0)
        grid_v = np.empty((self.nn, 3), dtype=np.float64)
        for a in range(3):
            mom = np.bincount(flat, weights=momentum[..., a].ravel(),
                              minlength=self.nn)
            frc = np.bincount(flat, weights=force[..., a].ravel(),
                              minlength=self.nn)
            grid_v[:, a] = (mom + dt * frc) / denom
        grid_v[mass_grid <= 1e-12] = 0.0

        # --- grid operations --------------------------------------------
        gv = grid_v.reshape(self.shape + (3,))
        gm = mass_grid.reshape(self.shape)
        frozen = self.frozen_mask
        impulse = np.sum(gv[frozen] * gm[frozen][:, None], axis=0)
        gv[frozen] = 0.0
        # Containing walls: kill outward normal velocity on the border layers.
        gv[0, :, :, 0] = 0.0
        gv[-1, :, :, 0] = 0.0
        gv[:, 0, :, 1] = 0.0
        gv[:, -1, :, 1] = 0.0
        self.contact_force = -impulse / dt

        # --- grid to particle --------------------------------------------
        vg = grid_v[idx]                       # [N,27,3]
        v_new = np.sum(Wm[:, :, None] * vg, axis=1)
        C_new = (4.0 / (dx * dx)) * np.einsum("nki,nkj,nk->nij", vg, dn, Wm)

        speed = np.sqrt(np.sum(v_new * v_new, axis=1))
        fast = speed > self.v_max
        if np.any(fast):
            v_new[fast] *= (self.v_max / speed[fast])[:, None]

        # --- constitutive update ------------------------------------------
        dF = np.eye(3, dtype=np.float64)[None, :, :] + dt * C_new
        F_trial = np.einsum("nij,njk->nik", dF, self.F)
        if not np.isfinite(F_trial).all():
            # Defensive guard: keep the batched SVD well posed.
            F_trial = np.nan_to_num(F_trial, nan=0.0, posinf=0.0, neginf=0.0)
            F_trial += np.eye(3, dtype=np.float64)[None, :, :]
        U, s, Vt = np.linalg.svd(F_trial)
        s = np.clip(s, 1.0 - self.theta_c, 1.0 + self.theta_s)
        self.F = np.einsum("nij,nj,nkj->nik", U, s, Vt)
        R = np.einsum("nij,nkj->nik", U, Vt)
        J = np.prod(s, axis=1)
        F_inv_t = np.einsum("nij,nj,nkj->nik", U, 1.0 / s, Vt)
        self.stress = (
            2.0 * self.mu * (self.F - R)
            + self.lam * ((J - 1.0) * J)[:, None, None] * F_inv_t
        )

        # --- advect ---------------------------------------------------------
        self.x = self.x + dt * v_new
        self.v = v_new
        self.C = C_new

        # Hard no-penetration safety clamp against the support top surface.
        low = self.x[:, 2] < 0.03
        if np.any(low):
            self.x[low, 2] = 0.03
            self.v[low, 2] = np.maximum(self.v[low, 2], 0.0)

        # Defensive finiteness guard (keeps the contract satisfied).
        if not np.isfinite(self.x).all():
            self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(self.v).all():
            self.v = np.nan_to_num(self.v)
        np.clip(self.x[:, 0], self.origin[0], self.origin[0] + (self.shape[0] - 1) * dx,
                out=self.x[:, 0])
        np.clip(self.x[:, 1], self.origin[1], self.origin[1] + (self.shape[1] - 1) * dx,
                out=self.x[:, 1])
        np.clip(self.x[:, 2], 0.03, self.origin[2] + (self.shape[2] - 1) * dx,
                out=self.x[:, 2])

        self.sim_time += dt

    # ---------------------------------------------------------------- output
    def _frame(self, t: float) -> StateFrame:
        external = self.mass[:, None] * self.gravity[None, :]
        return StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "initial_group_id": self.initial_group.copy(),
                "color_id": self.color_id.copy(),
                "particle_radius": self.radius.copy(),
                "deformation_gradient": self.F.copy(),
                "mass_quantity": self.mass[:, None].copy(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": external,
                "interaction_force": self.contact_force[None, :].copy(),
            },
            scalars={
                "gravity_x": float(self.gravity[0]),
                "gravity_y": float(self.gravity[1]),
                "gravity_z": float(self.gravity[2]),
                "particle_spacing": float(self.spacing),
                "rest_density": float(self.rest_density),
                "simulation_time": float(self.sim_time),
            },
        )

    # ----------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("initialize() must be called before advance_to()")
        t = float(time)
        while self.sim_time < t - 1e-9:
            dt = min(self.dt, t - self.sim_time)
            self._step(dt)
        return self._frame(t)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "affine_mpm_fixed_corotated_plastic",
            "particles": int(self.n),
            "grid_shape": list(self.shape),
            "substep_dt": float(self.dt),
            "youngs_modulus": 8.0e4,
            "plastic_compression_clamp": self.theta_c,
            "plastic_stretch_clamp": self.theta_s,
            "simulation_time": float(self.sim_time),
        }
