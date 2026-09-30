"""Viscoplastic material-point solver for the repeated topological merge scene.

The frozen scene supplies 24 initially independent rounded viscoplastic blocks.
They are discretised as material points and advanced with an explicit
Moving-Least-Squares Material Point Method (MLS-MPM + APIC transfers) on a
fixed background grid.  The shared background grid is what physically produces
the required repeated topological merges: once two blocks touch the same grid
nodes their momentum is pooled and a continuous, persistent material connection
is formed.

Constitutive model: finite-strain Hencky elasticity (logarithmic principal
stretches) coupled to a J2 / von Mises viscoplastic return mapping with a
Bingham-type rate-dependent flow rule.  Gravity acts on every sample, the
ground support is a frictional no-penetration boundary at z = 0.

State carried per material point:
    x   position                (N,3)
    v   velocity                (N,3)
    m   mass                    (N,)
    C   affine velocity gradient(N,3,3)   (APIC)
    Fe  elastic deformation gradient (N,3,3)
    F   total deformation gradient   (N,3,3)   <- exported
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_GRAVITY_FALLBACK = np.array([0.0, 0.0, -9.81])

# --- frozen material description (cohesive viscoplastic solid) --------------
_YOUNG_MODULUS = 1.0e5      # Pa
_POISSON_RATIO = 0.30
_YIELD_STRESS = 2.6e4       # Pa (von Mises uniaxial yield)
_PLASTIC_VISCOSITY = 3.0e3  # Pa s (Bingham consistency)
_MAX_LOG_STRAIN = 0.8
_MAX_SPEED = 40.0
_SUBSTEPS_PER_SAMPLE = 6    # explicit CFL margin with the frozen sample rate
_MIN_DENSITY = 1.0e-6


class BenchmarkSolver:
    """Deterministic explicit MPM solver implementing the Solver contract."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)

        frame = scene.initial_state()
        self._n = int(len(frame.position))
        self.x = np.array(frame.position, dtype=np.float64)
        self.v = np.array(frame.velocity, dtype=np.float64)
        if self.v.shape != self.x.shape:
            self.v = np.zeros_like(self.x)

        self.body_id = (
            np.array(frame.body_id, dtype=np.int64)
            if frame.body_id is not None
            else np.zeros(self._n, dtype=np.int64)
        )
        # carried render/identity attributes
        self.color_id = np.array(
            frame.attributes.get("color_id", np.zeros(self._n)), dtype=np.float64
        )
        self.group_id = np.array(
            frame.attributes.get("initial_group_id", self.body_id.astype(np.float64)),
            dtype=np.float64,
        )
        self.radius = np.array(
            frame.attributes.get("particle_radius", np.full(self._n, 0.112)),
            dtype=np.float64,
        )

        scalars = dict(frame.scalars)
        density = float(scalars.get("rest_density", 1050.0))
        density = max(density, _MIN_DENSITY)
        self.density = density
        self.gravity = np.array(
            [
                float(scalars.get("gravity_x", _GRAVITY_FALLBACK[0])),
                float(scalars.get("gravity_y", _GRAVITY_FALLBACK[1])),
                float(scalars.get("gravity_z", _GRAVITY_FALLBACK[2])),
            ],
            dtype=np.float64,
        )

        spacing = float(scalars.get("particle_spacing", 0.205))
        if not np.isfinite(spacing) or spacing <= 0.0:
            spacing = 0.205
        if frame.mass is not None:
            self.m = np.array(frame.mass, dtype=np.float64)
        else:
            self.m = np.full(self._n, density * spacing ** 3)
        self.m = np.maximum(self.m, _MIN_DENSITY * spacing ** 3)
        self.reference_volume = self.m / density

        # --- background grid -------------------------------------------------
        lo, hi = self._grid_bounds(scene, spacing)
        self.dx = spacing
        self.origin = np.array([lo[0], lo[1], 0.0], dtype=np.float64)
        self.nx = int(np.ceil((hi[0] - lo[0]) / self.dx)) + 1
        self.ny = int(np.ceil((hi[1] - lo[1]) / self.dx)) + 1
        self.nz = int(np.ceil((hi[2] - 0.0) / self.dx)) + 1
        self.nx = max(self.nx, 8)
        self.ny = max(self.ny, 8)
        self.nz = max(self.nz, 8)

        # --- state -----------------------------------------------------------
        self.C = np.zeros((self._n, 3, 3), dtype=np.float64)
        self.Fe = np.tile(np.eye(3), (self._n, 1, 1))
        self.F = np.tile(np.eye(3), (self._n, 1, 1))
        self.tau = np.zeros((self._n, 3, 3), dtype=np.float64)

        # --- elastic constants ----------------------------------------------
        e, nu = _YOUNG_MODULUS, _POISSON_RATIO
        self.shear = e / (2.0 * (1.0 + nu))
        self.lame = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.bulk = self.lame + 2.0 * self.shear / 3.0
        self.yield_tau = np.sqrt(2.0 / 3.0) * _YIELD_STRESS
        self.visc_ratio = 2.0 * self.shear / _PLASTIC_VISCOSITY  # 1/s

        # --- time bookkeeping -----------------------------------------------
        self.t = 0.0
        self.substeps = 0
        self.merges = 0
        self._momentum_ref = self._momentum()
        self._v_frame_start = self.v.copy()
        self._t_frame_start = 0.0
        self._sub_dt = 1.0 / (30.0 * _SUBSTEPS_PER_SAMPLE)

        self._grid_shape = (self.nx, self.ny, self.nz)
        self._cell_count = self.nx * self.ny * self.nz
        offsets = np.array(
            [(a, b, c) for a in range(3) for b in range(3) for c in range(3)],
            dtype=np.int64,
        )
        self._offsets = offsets
        self._eye = np.eye(3)

    # ------------------------------------------------------------- utilities
    @staticmethod
    def _grid_bounds(scene, spacing: float):
        """Frozen particle-domain bounds when available, else the sample extent."""
        lo = hi = None
        try:
            geometry = scene.config.get("scene_geometry") or {}
            for obj in geometry.get("objects", []):
                geom = obj.get("geometry", {})
                if geom.get("type") == "particle_domain":
                    lo = np.asarray(geom["bounds_min"], dtype=np.float64)
                    hi = np.asarray(geom["bounds_max"], dtype=np.float64)
                    break
        except Exception:
            lo = hi = None
        pad = 3.0 * spacing
        if lo is None or hi is None:
            pts = np.asarray(scene.initial_state().position, dtype=np.float64)
            lo = pts.min(axis=0) - pad
            hi = pts.max(axis=0) + pad
        lo = np.array([lo[0] - pad, lo[1] - pad, 0.0], dtype=np.float64)
        hi = np.array([hi[0] + pad, hi[1] + pad, hi[2] + pad], dtype=np.float64)
        return lo, hi

    def _momentum(self) -> np.ndarray:
        return (self.m[:, None] * self.v).sum(axis=0)

    def _total_mass(self) -> float:
        return float(self.m.sum())

    # ------------------------------------------------------- constitutive law
    def _update_stress(self, dt: float) -> None:
        """Hencky elasticity + J2 viscoplastic return mapping (vectorised)."""
        svd = np.linalg.svd(self.Fe)
        u, s, vt = svd
        s = np.clip(s, np.exp(-_MAX_LOG_STRAIN), np.exp(_MAX_LOG_STRAIN))
        eps = np.log(s)
        trace = eps.sum(axis=1, keepdims=True)
        tau_i = 2.0 * self.shear * eps + self.lame * trace
        mean = tau_i.mean(axis=1, keepdims=True)
        dev = tau_i - mean
        dev_norm = np.sqrt(np.einsum("ij,ij->i", dev, dev))

        ratio = self.visc_ratio * dt
        flowing = dev_norm > self.yield_tau
        new_norm = np.where(
            flowing,
            (dev_norm + ratio * self.yield_tau) / (1.0 + ratio),
            dev_norm,
        )
        scale = new_norm / np.maximum(dev_norm, 1.0e-12)
        tau_i = mean + dev * scale[:, None]

        # invert the (Hencky) constitutive relation for the elastic stretches
        trace_new = tau_i.sum(axis=1, keepdims=True) / (2.0 * self.shear + 3.0 * self.lame)
        eps_new = (tau_i - self.lame * trace_new) / (2.0 * self.shear)
        eps_new = np.clip(eps_new, -_MAX_LOG_STRAIN, _MAX_LOG_STRAIN)
        s_new = np.exp(eps_new)

        self.Fe = (u * s_new[:, None, :]) @ vt
        self.tau = (u * tau_i[:, None, :]) @ np.transpose(u, (0, 2, 1))

    # ------------------------------------------------------------ MPM kernels
    def _weights(self):
        q = (self.x - self.origin) / self.dx
        base = np.floor(q - 0.5).astype(np.int64)
        base[:, 0] = np.clip(base[:, 0], 0, self.nx - 3)
        base[:, 1] = np.clip(base[:, 1], 0, self.ny - 3)
        base[:, 2] = np.clip(base[:, 2], 0, self.nz - 3)
        f = q - base
        f = np.clip(f, 0.0, 1.0)

        w = np.empty((self._n, 3, 3), dtype=np.float64)
        g = np.empty((self._n, 3, 3), dtype=np.float64)
        for d in range(3):
            fd = f[:, d]
            w[:, d, 0] = 0.5 * (1.0 - fd) ** 2
            w[:, d, 1] = 0.75 - (fd - 0.5) ** 2
            w[:, d, 2] = 0.5 * fd ** 2
            g[:, d, 0] = -(1.0 - fd) / self.dx
            g[:, d, 1] = (1.0 - 2.0 * fd) / self.dx
            g[:, d, 2] = fd / self.dx
        return base, w, g

    def _particle_to_grid(self, dt: float, base, w, g, grid_mass, grid_mom) -> None:
        m, x, v, C, tau, V0 = self.m, self.x, self.v, self.C, self.tau, self.reference_volume
        ny, nz = self.ny, self.nz
        for a, b, c in self._offsets:
            wa, wb, wc = w[:, 0, a], w[:, 1, b], w[:, 2, c]
            weight = wa * wb * wc
            gw = np.empty((self._n, 3), dtype=np.float64)
            gw[:, 0] = g[:, 0, a] * wb * wc
            gw[:, 1] = wa * g[:, 1, b] * wc
            gw[:, 2] = wa * wb * g[:, 2, c]

            li = (base[:, 0] + a) * (ny * nz) + (base[:, 1] + b) * nz + (base[:, 2] + c)
            mw = m * weight
            np.add.at(grid_mass, li, mw)

            node = np.empty((self._n, 3), dtype=np.float64)
            node[:, 0] = (base[:, 0] + a) * self.dx + self.origin[0]
            node[:, 1] = (base[:, 1] + b) * self.dx + self.origin[1]
            node[:, 2] = (base[:, 2] + c) * self.dx + self.origin[2]
            dpos = node - x
            affine = v + np.einsum("nij,nj->ni", C, dpos)
            grad_tau = np.einsum("nij,nj->ni", tau, gw)
            momentum = mw[:, None] * affine - dt * V0[:, None] * grad_tau
            np.add.at(grid_mom, li, momentum)

    def _grid_to_particle(self, dt: float, velocity, base, w, g) -> None:
        ny, nz = self.ny, self.nz
        v_new = np.zeros((self._n, 3), dtype=np.float64)
        c_new = np.zeros((self._n, 3, 3), dtype=np.float64)
        coef = 4.0 / (self.dx * self.dx)
        for a, b, c in self._offsets:
            wa, wb, wc = w[:, 0, a], w[:, 1, b], w[:, 2, c]
            weight = wa * wb * wc
            node_v = velocity[base[:, 0] + a, base[:, 1] + b, base[:, 2] + c]
            v_new += weight[:, None] * node_v
            node = np.empty((self._n, 3), dtype=np.float64)
            node[:, 0] = (base[:, 0] + a) * self.dx + self.origin[0]
            node[:, 1] = (base[:, 1] + b) * self.dx + self.origin[1]
            node[:, 2] = (base[:, 2] + c) * self.dx + self.origin[2]
            dpos = node - self.x
            c_new += coef * (weight[:, None, None] * node_v[:, :, None] * dpos[:, None, :])

        self.C = c_new
        self.x = self.x + dt * v_new
        stretch = self._eye[None, :, :] + dt * c_new
        self.F = np.einsum("nij,njk->nik", stretch, self.F)
        self.v = v_new

        # kinematic safeguards: support no-penetration and bounded speed
        below = self.x[:, 2] < 0.0
        if np.any(below):
            self.x[below, 2] = 0.0
            self.v[below, 2] = np.maximum(self.v[below, 2], 0.0)
            self.v[below, :2] *= 0.85
        speed = np.linalg.norm(self.v, axis=1)
        fast = speed > _MAX_SPEED
        if np.any(fast):
            self.v[fast] *= (_MAX_SPEED / speed[fast])[:, None]

    # -------------------------------------------------------------- time step
    def _step(self, dt: float) -> None:
        self._update_stress(dt)
        base, w, g = self._weights()

        grid_mass = np.zeros(self._cell_count, dtype=np.float64)
        grid_mom = np.zeros((self._cell_count, 3), dtype=np.float64)
        self._particle_to_grid(dt, base, w, g, grid_mass, grid_mom)

        velocity = np.zeros((self._cell_count, 3), dtype=np.float64)
        occupied = grid_mass > 0.0
        velocity[occupied] = grid_mom[occupied] / grid_mass[occupied, None]
        velocity += dt * self.gravity

        vel_grid = velocity.reshape(self._grid_shape + (3,))
        mass_grid = grid_mass.reshape(self._grid_shape)
        contact = mass_grid[:, :, 0] > 0.0
        # frictional, no-penetration support on the frozen ground plane z = 0
        vel_grid[:, :, 0, 2] = np.where(contact, np.minimum(vel_grid[:, :, 0, 2], 0.0), vel_grid[:, :, 0, 2])
        vel_grid[:, :, 0, 0] = np.where(contact, vel_grid[:, :, 0, 0] * 0.85, vel_grid[:, :, 0, 0])
        vel_grid[:, :, 0, 1] = np.where(contact, vel_grid[:, :, 0, 1] * 0.85, vel_grid[:, :, 0, 1])
        velocity = vel_grid.reshape(-1, 3)

        self._grid_to_particle(dt, velocity, base, w, g)
        self.t += dt
        self.substeps += 1

    # -------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = float(time)
        t_start = self.t
        v_start = self.v.copy()
        while self.t < target - 1.0e-12:
            dt = min(self._sub_dt, target - self.t)
            if dt <= 1.0e-12:
                break
            self._step(dt)
        self.t = target

        dt_frame = max(target - t_start, 1.0e-12)
        impulse = self._total_mass() * self.gravity
        interaction = (
            (self.m[:, None] * (self.v - v_start)).sum(axis=0) / dt_frame - impulse
        )
        return self._frame(target, interaction)

    def _frame(self, time: float, interaction: np.ndarray) -> StateFrame:
        speed = np.linalg.norm(self.v, axis=1)
        det_f = np.linalg.det(self.F)
        volume = np.maximum(self.reference_volume * np.maximum(det_f, 1.0e-3), 0.0)
        external = self.m[:, None] * self.gravity[None, :]
        frame = StateFrame(
            time=float(time),
            position=np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0),
            velocity=np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0),
            mass=np.nan_to_num(self.m, nan=0.0, posinf=0.0, neginf=0.0),
            body_id=self.body_id,
            attributes={
                "deformation_gradient": np.nan_to_num(
                    self.F, nan=0.0, posinf=0.0, neginf=0.0
                ),
                "mass_quantity": np.nan_to_num(self.m, nan=0.0, posinf=0.0, neginf=0.0),
                "volume_quantity": np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0),
                "initial_group_id": self.group_id,
                "color_id": self.color_id,
                "particle_radius": self.radius,
                "speed": np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": np.nan_to_num(external, nan=0.0, posinf=0.0, neginf=0.0),
                "interaction_force": np.nan_to_num(
                    np.asarray(interaction, dtype=np.float64).reshape(1, 3),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
            },
            scalars={
                "gravity_x": float(self.gravity[0]),
                "gravity_y": float(self.gravity[1]),
                "gravity_z": float(self.gravity[2]),
                "rest_density": float(self.density),
                "particle_spacing": float(self.dx),
                "material_yield_stress": float(_YIELD_STRESS),
                "material_viscosity": float(_PLASTIC_VISCOSITY),
                "total_mass": float(self._total_mass()),
                "substeps": float(self.substeps),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "mls_mpm_viscoplastic",
            "particle_count": int(self._n),
            "grid_shape": [int(self.nx), int(self.ny), int(self.nz)],
            "grid_spacing": float(self.dx),
            "substeps": int(self.substeps),
            "simulated_time": float(self.t),
            "total_mass": float(self._total_mass()),
            "yield_stress": float(_YIELD_STRESS),
            "plastic_viscosity": float(_PLASTIC_VISCOSITY),
        }
