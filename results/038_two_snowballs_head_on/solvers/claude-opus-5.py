"""Explicit MLS-MPM elasto-plastic snow solver (no external physics engine).

Physical model
--------------
Continuum snow is discretised with material points carrying mass, velocity,
elastic deformation gradient ``F``, plastic volume ratio ``Jp`` and an affine
velocity matrix ``C`` (APIC / MLS-MPM).  Each substep performs

1. elastic predictor ``F <- (I + dt C) F``;
2. snow plasticity: SVD of ``F`` with singular values clamped to
   ``[1-theta_c, 1+theta_s]`` (Stomakhin et al. 2013), the rejected part
   accumulated into ``Jp`` and exponential hardening of the Lame moduli;
3. fixed-corotated Kirchhoff stress ``tau = 2 mu (F-R) F^T + lambda J (J-1) I``
   built in closed form from the SVD;
4. particle-to-grid transfer of mass, momentum and the MLS stress divergence
   with quadratic B-spline weights on a fixed Cartesian background grid;
5. grid velocity update (gravity), Coulomb-friction floor contact and
   container walls;
6. grid-to-particle transfer (velocity + affine matrix) and symplectic-Euler
   advection with particle-level floor clamping.

Momentum transfer is conservative (the quadratic B-spline first moment vanishes
exactly), so the total momentum changes only through gravity and boundary
contact, both of which are reported in ``fields.external_force``.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # --- snow constitutive parameters (Stomakhin et al. 2013, Section 5) ---
    POISSON = 0.2
    YOUNGS_REFERENCE = 1.4e5      # Pa at the paper's reference snow density
    DENSITY_REFERENCE = 400.0     # kg/m^3 reference density for modulus scaling
    THETA_C = 2.5e-2              # critical compression
    THETA_S = 7.5e-3              # critical stretch
    HARDENING = 10.0              # xi
    HARDEN_MIN = 0.1
    HARDEN_MAX = 5.0

    # --- discretisation / contact ---
    DX = 0.125                    # background grid spacing [m]
    DT_TARGET = 4.0e-4            # target substep [s] (CFL limited)
    FLOOR_FRICTION = 0.45         # Coulomb coefficient on the packed-snow floor
    PARTICLE_TANGENT_DAMP = 0.95  # mild tangential loss for clamped particles
    WALL_PAD = 2                  # grid cells treated as domain wall
    SPEED_CAP = 60.0              # numerical safety ceiling [m/s]
    MASS_EPS = 1.0e-12

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = scene.config
        self.config = config

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=np.float64)
        self.vel = np.array(frame0.velocity, dtype=np.float64)
        self.n = int(self.pos.shape[0])
        if frame0.mass is None:
            raise ValueError("scene initial state must provide per-particle mass")
        self.mass = np.array(frame0.mass, dtype=np.float64)
        self.body_id = np.array(frame0.body_id, dtype=np.int64)
        self.initial_component_id = self.body_id.copy()

        attributes0 = dict(frame0.attributes)
        reference_volume = attributes0.get("reference_volume")
        if reference_volume is None:
            reference_volume = self.mass / float(config["snowballs"]["nominal_density"])
        self.reference_volume = np.array(reference_volume, dtype=np.float64)
        self.base_scalars = {str(k): float(v) for k, v in frame0.scalars.items()}

        self.gravity = np.asarray(config["gravity"], dtype=np.float64)
        self.floor_z = float(config["floor"]["top_z"])
        snow = config["snowballs"]
        self.spacing = float(snow["particle_spacing"])

        # Material point volume: the sampling lattice tiles space with one
        # particle per spacing^3 cell, so that is the consistent reference
        # volume for the stress divergence.  The resulting effective density
        # scales the Young modulus so the dimensionless stiffness
        # E / (rho v^2) matches the reference snow configuration.
        self.vol0 = self.spacing ** 3
        self.density_eff = float(self.mass.mean()) / self.vol0
        youngs = self.YOUNGS_REFERENCE * self.density_eff / self.DENSITY_REFERENCE
        self.youngs = float(youngs)
        self.mu0 = youngs / (2.0 * (1.0 + self.POISSON))
        self.lam0 = youngs * self.POISSON / ((1.0 + self.POISSON) * (1.0 - 2.0 * self.POISSON))

        # per-particle state
        self.F = np.tile(np.eye(3, dtype=np.float64), (self.n, 1, 1))
        self.C = np.zeros((self.n, 3, 3), dtype=np.float64)
        self.Jp = np.ones(self.n, dtype=np.float64)

        self._setup_grid()
        self._setup_time()

        self.impulse = np.zeros((self.n, 3), dtype=np.float64)
        self.last_sample_time = 0.0
        self.step_count = 0
        self.sim_time = 0.0
        self.max_speed = float(np.linalg.norm(self.vel, axis=1).max())
        self.repair_count = 0
        self.cap_count = 0

    def _setup_grid(self) -> None:
        dx = float(self.DX)
        self.dx = dx
        self.inv_dx = 1.0 / dx
        # Symmetric about x = 0 and y = 0 so a head-on impact stays symmetric;
        # z = floor_z lands exactly on grid layer index WALL_PAD.
        half = np.asarray(self.config["floor"]["half_extents"], dtype=np.float64)
        lo = np.array(
            [-float(half[0]), -min(float(half[1]), 2.5), self.floor_z - self.WALL_PAD * dx],
            dtype=np.float64,
        )
        hi = np.array(
            [float(half[0]), min(float(half[1]), 2.5), self.floor_z + 2.0], dtype=np.float64
        )
        dims = np.rint((hi - lo) / dx).astype(np.int64) + 1
        self.origin = lo
        self.dims = (int(dims[0]), int(dims[1]), int(dims[2]))
        self.n_nodes = int(dims[0] * dims[1] * dims[2])
        self.floor_k = int(round((self.floor_z - lo[2]) / dx))
        # Admissible particle band: the quadratic stencil must stay in-grid.
        # Chosen symmetrically about the domain centre so a head-on impact
        # cannot be biased by the clamp itself.
        self.lo_clamp = lo + 1.5 * dx
        self.hi_clamp = lo + (dims.astype(np.float64) - 2.5) * dx
        self.offsets = np.array(
            [[i, j, k] for i in range(3) for j in range(3) for k in range(3)],
            dtype=np.int64,
        )
        np.clip(self.pos, self.lo_clamp, self.hi_clamp, out=self.pos)

    def _setup_time(self) -> None:
        sample_dt = float(self.config["sample_dt"])
        substeps = max(1, int(math.ceil(sample_dt / self.DT_TARGET)))
        self.substeps_per_sample = substeps
        self.dt = sample_dt / substeps
        self.sample_dt = sample_dt
        wave = math.sqrt(
            (self.lam0 + 2.0 * self.mu0) * self.HARDEN_MAX / max(self.density_eff, 1e-9)
        )
        self.cfl_dt = 0.4 * self.dx / max(wave + 10.0, 1e-9)

    # ------------------------------------------------------- constitutive law
    def _sanitize_F(self) -> None:
        """Keep the elastic deformation gradient finite and non-degenerate."""
        bad = ~np.isfinite(self.F).all(axis=(1, 2))
        if np.any(bad):
            self.F[bad] = np.eye(3)
            self.C[bad] = 0.0
            self.repair_count += int(bad.sum())
        np.clip(self.F, -10.0, 10.0, out=self.F)

    def _plastic_return_and_stress(self):
        """Snow plasticity + fixed-corotated Kirchhoff stress from the SVD."""
        self._sanitize_F()
        try:
            U, sigma, Vt = np.linalg.svd(self.F)
        except np.linalg.LinAlgError:
            self.F[:] = np.eye(3)
            self.C[:] = 0.0
            self.repair_count += self.n
            U, sigma, Vt = np.linalg.svd(self.F)

        sigma = np.clip(sigma, 1.0e-4, 1.0e4)
        sigma_clamped = np.clip(sigma, 1.0 - self.THETA_C, 1.0 + self.THETA_S)

        # plastic volume ratio: the rejected singular-value part
        det_all = np.prod(sigma, axis=1)
        det_elastic = np.prod(sigma_clamped, axis=1)
        self.Jp = np.clip(self.Jp * det_all / np.maximum(det_elastic, 1.0e-8), 0.05, 20.0)

        # rebuild the purely elastic deformation gradient F = U S V^T
        self.F = (U * sigma_clamped[:, None, :]) @ Vt

        # hardening: compaction (Jp<1) stiffens the snow
        harden = np.exp(self.HARDENING * (1.0 - self.Jp))
        np.clip(harden, self.HARDEN_MIN, self.HARDEN_MAX, out=harden)
        mu = self.mu0 * harden
        lam = self.lam0 * harden

        # R = U V^T (polar rotation), tau = 2 mu (F - R) F^T + lam J (J-1) I
        R = U @ Vt
        Fm = self.F
        tau = 2.0 * mu[:, None, None] * ((Fm - R) @ np.transpose(Fm, (0, 2, 1)))
        press = lam * det_elastic * (det_elastic - 1.0)
        idx = np.arange(3)
        tau[:, idx, idx] += press[:, None]
        return tau

    # ------------------------------------------------------------- transfers
    def _weights(self, pos):
        """Quadratic B-spline base index and per-axis weights."""
        fx = (pos - self.origin[None, :]) * self.inv_dx
        base = np.floor(fx - 0.5).astype(np.int64)
        np.clip(base[:, 0], 0, self.dims[0] - 3, out=base[:, 0])
        np.clip(base[:, 1], 0, self.dims[1] - 3, out=base[:, 1])
        np.clip(base[:, 2], 0, self.dims[2] - 3, out=base[:, 2])
        d = fx - base.astype(np.float64)  # in [0.5, 1.5]
        w = np.empty((pos.shape[0], 3, 3), dtype=np.float64)  # [N, node, axis]
        w[:, 0, :] = 0.5 * (1.5 - d) ** 2
        w[:, 1, :] = 0.75 - (d - 1.0) ** 2
        w[:, 2, :] = 0.5 * (d - 0.5) ** 2
        return base, w, d

    def _stencil(self, base, w, d):
        """Flat node indices, scalar weights and node-particle offsets."""
        nx, ny, nz = self.dims
        off = self.offsets  # [27,3]
        node = base[:, None, :] + off[None, :, :]
        flat = (node[:, :, 0] * ny + node[:, :, 1]) * nz + node[:, :, 2]
        wt = (
            w[:, off[:, 0], 0]
            * w[:, off[:, 1], 1]
            * w[:, off[:, 2], 2]
        )  # [N,27]
        dpos = (off.astype(np.float64)[None, :, :] - d[:, None, :]) * self.dx
        return flat, wt, dpos

    # ------------------------------------------------------------- time step
    def _substep(self) -> None:
        dt = self.dt
        nx, ny, nz = self.dims
        n_nodes = self.n_nodes

        tau = self._plastic_return_and_stress()

        # MLS-MPM affine momentum matrix: stress divergence + APIC term
        scale = -dt * self.vol0 * 4.0 * self.inv_dx * self.inv_dx
        affine = scale * tau + self.mass[:, None, None] * self.C

        base, w, d = self._weights(self.pos)
        flat, wt, dpos = self._stencil(base, w, d)
        flat_r = flat.ravel()

        node_mom = wt[:, :, None] * (
            self.mass[:, None, None] * self.vel[:, None, :]
            + np.matmul(dpos, np.transpose(affine, (0, 2, 1)))
        )
        grid_m = np.bincount(
            flat_r, weights=(wt * self.mass[:, None]).ravel(), minlength=n_nodes
        )
        grid_mv = np.empty((n_nodes, 3), dtype=np.float64)
        for c in range(3):
            grid_mv[:, c] = np.bincount(
                flat_r, weights=node_mom[:, :, c].ravel(), minlength=n_nodes
            )

        active = grid_m > self.MASS_EPS
        gv = np.zeros((n_nodes, 3), dtype=np.float64)
        np.divide(grid_mv, grid_m[:, None], out=gv, where=active[:, None])
        gv[active] += dt * self.gravity[None, :]

        gv_free = gv.copy()
        self._apply_grid_boundaries(gv, nx, ny, nz)
        gdv = gv - gv_free  # velocity change produced by boundary contact

        gvn = gv[flat]          # [N,27,3]
        gdvn = gdv[flat]
        new_v = np.einsum("nk,nkc->nc", wt, gvn, optimize=True)
        dv_bc = np.einsum("nk,nkc->nc", wt, gdvn, optimize=True)
        self.C = (4.0 * self.inv_dx * self.inv_dx) * np.matmul(
            np.transpose(wt[:, :, None] * gvn, (0, 2, 1)), dpos
        )

        v_reference = new_v.copy()
        self.vel = new_v
        self.pos = self.pos + dt * self.vel
        self._apply_particle_boundaries()

        self.impulse += self.mass[:, None] * (dv_bc + (self.vel - v_reference))

        # elastic predictor for the next substep
        eye = np.eye(3, dtype=np.float64)[None, :, :]
        self.F = np.matmul(eye + dt * self.C, self.F)

        speed = np.linalg.norm(self.vel, axis=1)
        self.max_speed = max(self.max_speed, float(speed.max()))
        self.step_count += 1

    def _apply_grid_boundaries(self, gv, nx, ny, nz) -> None:
        g3 = gv.reshape(nx, ny, nz, 3)
        k = self.floor_k
        if k > 0:
            g3[:, :, :k, :] = 0.0
        # floor layer: no penetration + Coulomb friction on the tangent plane
        vz = g3[:, :, k, 2].copy()
        contact = vz < 0.0
        normal_impulse = np.where(contact, -vz, 0.0)
        vt = g3[:, :, k, :2].copy()
        vt_norm = np.sqrt(vt[..., 0] ** 2 + vt[..., 1] ** 2)
        friction_scale = np.where(
            (vt_norm > 1.0e-12) & contact,
            np.maximum(0.0, 1.0 - self.FLOOR_FRICTION * normal_impulse
                       / np.maximum(vt_norm, 1.0e-12)),
            1.0,
        )
        g3[:, :, k, 0] = vt[..., 0] * friction_scale
        g3[:, :, k, 1] = vt[..., 1] * friction_scale
        g3[:, :, k, 2] = np.where(contact, 0.0, vz)

        pad = self.WALL_PAD
        g3[:pad, :, :, 0] = np.maximum(g3[:pad, :, :, 0], 0.0)
        g3[-pad:, :, :, 0] = np.minimum(g3[-pad:, :, :, 0], 0.0)
        g3[:, :pad, :, 1] = np.maximum(g3[:, :pad, :, 1], 0.0)
        g3[:, -pad:, :, 1] = np.minimum(g3[:, -pad:, :, 1], 0.0)
        g3[:, :, -pad:, 2] = np.minimum(g3[:, :, -pad:, 2], 0.0)

    def _apply_particle_boundaries(self) -> None:
        below = self.pos[:, 2] < self.floor_z
        if np.any(below):
            self.pos[below, 2] = self.floor_z
            vz = self.vel[below, 2]
            sticking = vz < 0.0
            rows = np.nonzero(below)[0]
            hit = rows[sticking]
            if hit.size:
                self.vel[hit, 2] = 0.0
                self.vel[hit, 0] *= self.PARTICLE_TANGENT_DAMP
                self.vel[hit, 1] *= self.PARTICLE_TANGENT_DAMP
        # keep every particle inside the background grid stencil band
        outside = np.any(
            (self.pos < self.lo_clamp[None, :]) | (self.pos > self.hi_clamp[None, :]), axis=1
        )
        if np.any(outside):
            np.clip(self.pos, self.lo_clamp, self.hi_clamp, out=self.pos)
            self.vel[outside] = 0.0
        # numerical safety ceiling; should never engage in a stable run
        speed = np.linalg.norm(self.vel, axis=1)
        hot = speed > self.SPEED_CAP
        if np.any(hot):
            self.vel[hot] *= (self.SPEED_CAP / speed[hot])[:, None]
            self.cap_count += int(hot.sum())
        if not np.isfinite(self.pos).all() or not np.isfinite(self.vel).all():
            bad = ~(np.isfinite(self.pos).all(axis=1) & np.isfinite(self.vel).all(axis=1))
            self.pos[bad] = np.clip(
                np.nan_to_num(self.pos[bad], nan=0.0, posinf=0.0, neginf=0.0),
                self.lo_clamp,
                self.hi_clamp,
            )
            self.vel[bad] = 0.0
            self.repair_count += int(bad.sum())

    # ---------------------------------------------------------------- sampling
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < 0.0:
            raise ValueError("advance_to requires a finite non-negative time")
        target_steps = int(round(target / self.dt))
        while self.step_count < target_steps:
            self._substep()
        self.sim_time = self.step_count * self.dt
        return self._make_frame(target)

    def _make_frame(self, time: float) -> StateFrame:
        interval = float(time) - self.last_sample_time
        gravity_force = self.mass[:, None] * self.gravity[None, :]
        if interval > 1.0e-12:
            contact_force = self.impulse / interval
        else:
            contact_force = np.zeros_like(self.impulse)
        external_force = gravity_force + contact_force
        external_force = np.nan_to_num(
            external_force, nan=0.0, posinf=0.0, neginf=0.0
        )
        self.impulse = np.zeros((self.n, 3), dtype=np.float64)
        self.last_sample_time = float(time)

        momentum = self.mass[:, None] * self.vel
        kinetic = 0.5 * float(np.sum(self.mass * np.sum(self.vel * self.vel, axis=1)))
        scalars = dict(self.base_scalars)
        scalars.update(
            {
                "total_mass": float(self.mass.sum()),
                "kinetic_energy": kinetic,
                "total_momentum_x": float(momentum[:, 0].sum()),
                "total_momentum_y": float(momentum[:, 1].sum()),
                "total_momentum_z": float(momentum[:, 2].sum()),
                "substep_dt": float(self.dt),
                "effective_density": float(self.density_eff),
                "youngs_modulus": float(self.youngs),
            }
        )

        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(self.pos, dtype=np.float64),
            velocity=np.ascontiguousarray(self.vel, dtype=np.float64),
            mass=np.ascontiguousarray(self.mass, dtype=np.float64),
            body_id=np.ascontiguousarray(self.body_id, dtype=np.int64),
            attributes={
                "mass_quantity": np.ascontiguousarray(self.mass, dtype=np.float64),
                "reference_volume": np.ascontiguousarray(
                    self.reference_volume, dtype=np.float64
                ),
                "deformation_gradient": np.ascontiguousarray(self.F, dtype=np.float64),
                "plastic_volume_ratio": np.ascontiguousarray(self.Jp, dtype=np.float64),
                "initial_component_id": np.ascontiguousarray(
                    self.initial_component_id, dtype=np.int64
                ),
            },
            fields={
                "external_force": np.ascontiguousarray(external_force, dtype=np.float64),
                "floor_signed_distance": np.ascontiguousarray(
                    self.pos[:, 2] - self.floor_z, dtype=np.float64
                ),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            },
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self) -> dict[str, object]:
        momentum = self.mass[:, None] * self.vel
        return {
            "solver": "self_implemented_mls_mpm_snow",
            "particles": int(self.n),
            "grid_dims": [int(v) for v in self.dims],
            "grid_spacing": float(self.dx),
            "substep_dt": float(self.dt),
            "substeps_per_sample": int(self.substeps_per_sample),
            "cfl_reference_dt": float(self.cfl_dt),
            "total_substeps": int(self.step_count),
            "simulated_time": float(self.sim_time),
            "effective_density": float(self.density_eff),
            "youngs_modulus": float(self.youngs),
            "lame_mu": float(self.mu0),
            "lame_lambda": float(self.lam0),
            "total_mass": float(self.mass.sum()),
            "final_momentum": [float(v) for v in momentum.sum(axis=0)],
            "max_speed": float(self.max_speed),
            "speed_cap_events": int(self.cap_count),
            "state_repairs": int(self.repair_count),
        }

