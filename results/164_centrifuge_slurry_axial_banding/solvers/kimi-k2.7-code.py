"""Self-contained Lagrangian slurry solver for centrifuge axial banding."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Lagrangian slurry solver for a horizontal rotating centrifuge drum.

    The model treats the slurry as a set of Lagrangian particles immersed in an
    implicit carrier fluid. Particle-particle repulsion is modelled through a
    grid-based pressure that depends on local volume fraction, while the
    rotating drum wall supplies a kinematic tangential velocity. Two particle
    species of different size separate radially under an effective percolation
    force, and axial banding is seeded by a phase-opposed perturbation along
    the drum axis.
    """

    def __init__(self) -> None:
        # mutable state
        self.position: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        self.mass: np.ndarray | None = None
        self.body_id: np.ndarray | None = None
        self.species: np.ndarray | None = None
        self.radius: np.ndarray | None = None
        self.mass_quantity: np.ndarray | None = None
        self.time = 0.0
        self._scene: Any | None = None
        self._active_mask: np.ndarray | None = None

        # physical parameters (overridden in initialize)
        self.R = 0.92
        self.L = 4.8
        self.center_z = 1.05
        self.omega = 5.5
        self.rho_p = 1450.0
        self.species_count = 2
        self.g = np.array([0.0, 0.0, -9.81], dtype=float)

        # diagnostic / grid
        self.field_shape = (40, 18, 18)
        self.grid_spacing = 0.12
        self.characteristic_length = 1.84
        self.contact_length_scale = 0.12

        # numerical parameters
        self.substeps = 10
        self.dt = 0.04 / self.substeps
        self.phi_crit = 0.55
        self.Kp = 3e4
        self.wall_k = 3e4
        self.wall_c = 30.0
        self.wall_mu = 0.3
        self.drag_coeff = 12.0
        self.seg_strength = 25.0
        self.band_strength = 0.4
        self.conc_strength = 0.02
        self.vmax = 18.0

        # grid coordinates (set in initialize)
        self.xmin = self.ymin = self.zmin = 0.0
        self.xmax = self.ymax = self.zmax = 0.0
        self.dx = self.dy = self.dz = 1.0
        self.Vcell = 1.0

    def initialize(self, scene: Any, seed: int) -> None:
        self._scene = scene

        frame = scene.initial_state()
        # Copy arrays so the solver owns mutable state.
        self.position = np.array(frame.position, dtype=float)
        self.velocity = np.array(frame.velocity, dtype=float)
        self.mass = np.array(frame.mass, dtype=float)
        self.body_id = np.array(frame.body_id, dtype=np.int32)
        self.species = np.array(frame.attributes["species_id"], dtype=np.int32)
        self.radius = np.array(frame.attributes["particle_radius"], dtype=float)
        self.mass_quantity = np.array(frame.attributes["mass_quantity"], dtype=float)
        self.time = float(frame.time)

        # Pull parameters from the scene configuration when available.
        cfg = getattr(scene, "config", {})
        if "centrifuge" in cfg:
            c = cfg["centrifuge"]
            self.R = float(c.get("radius", self.R))
            self.L = float(c.get("length", self.L))
            self.center_z = float(c.get("center_height", self.center_z))
            self.omega = float(c.get("angular_velocity_rad_s", self.omega))
            self.rho_p = float(c.get("particle_density", self.rho_p))
            self.species_count = int(c.get("species_count", self.species_count))
        if "gravity" in cfg:
            self.g = np.array(cfg["gravity"], dtype=float)

        # Diagnostic grid shape from the initial velocity field.
        if "velocity_field" in frame.fields:
            vf = np.asarray(frame.fields["velocity_field"])
            if vf.ndim == 4:
                self.field_shape = tuple(vf.shape[:3])

        # Use the physical cylinder bounding box for the diagnostic grid.
        self.xmin, self.xmax = -self.L / 2.0, self.L / 2.0
        self.ymin, self.ymax = -self.R, self.R
        self.zmin, self.zmax = self.center_z - self.R, self.center_z + self.R

        nx, ny, nz = self.field_shape
        self.dx = (self.xmax - self.xmin) / max(1, nx - 1)
        self.dy = (self.ymax - self.ymin) / max(1, ny - 1)
        self.dz = (self.zmax - self.zmin) / max(1, nz - 1)
        self.Vcell = self.dx * self.dy * self.dz

        self.characteristic_length = 2.0 * self.R
        self.grid_spacing = 0.12
        self.contact_length_scale = self.grid_spacing

        self._active_mask = np.ones(self.field_shape, dtype=bool)

    # --------------------------------------------------------------------- #
    # Grid helpers
    # --------------------------------------------------------------------- #
    def _to_cell(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ix = ((pos[:, 0] - self.xmin) / self.dx).astype(np.int64)
        iy = ((pos[:, 1] - self.ymin) / self.dy).astype(np.int64)
        iz = ((pos[:, 2] - self.zmin) / self.dz).astype(np.int64)
        nx, ny, nz = self.field_shape
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        iz = np.clip(iz, 0, nz - 1)
        return ix, iy, iz

    def _deposit_scalar(self, pos: np.ndarray, scalar: np.ndarray) -> np.ndarray:
        ix, iy, iz = self._to_cell(pos)
        grid = np.zeros(self.field_shape, dtype=float)
        np.add.at(grid, (ix, iy, iz), scalar)
        return grid

    def _pressure_force(self, pos: np.ndarray, r: np.ndarray) -> np.ndarray:
        Vp = (4.0 / 3.0) * math.pi * r ** 3
        vol_grid = self._deposit_scalar(pos, Vp)
        phi = vol_grid / self.Vcell
        # Smooth the volume fraction to obtain a stable pressure gradient.
        phi_smooth = uniform_filter(phi, size=3, mode="nearest")
        dphi = phi_smooth - self.phi_crit
        P = np.where(dphi > 0.0, self.Kp * dphi * dphi, 0.0)
        gradP = np.gradient(P, self.dx, self.dy, self.dz)
        ix, iy, iz = self._to_cell(pos)
        gradP_p = np.stack([g[ix, iy, iz] for g in gradP], axis=-1)
        return -Vp[:, None] * gradP_p

    def _concentration_force(self, pos: np.ndarray) -> np.ndarray:
        ix, iy, iz = self._to_cell(pos)
        force = np.zeros_like(pos)
        for k in range(self.species_count):
            mask = self.species == k
            if not np.any(mask):
                continue
            mass_k = np.where(mask, self.mass, 0.0)
            grid_k = self._deposit_scalar(pos, mass_k)
            grid_k = uniform_filter(grid_k, size=3, mode="nearest")
            grad = np.gradient(grid_k, self.dx, self.dy, self.dz)[0]
            grid_k_p = grad[ix[mask], iy[mask], iz[mask]]
            denom = grid_k[ix[mask], iy[mask], iz[mask]] + 1e-12
            force[mask, 0] += self.conc_strength * grid_k_p / denom
        return force

    def _velocity_field(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        ix, iy, iz = self._to_cell(pos)
        grid = np.zeros(self.field_shape + (3,), dtype=float)
        count = np.zeros(self.field_shape, dtype=float)
        for d in range(3):
            np.add.at(grid[..., d], (ix, iy, iz), vel[:, d])
        np.add.at(count, (ix, iy, iz), 1.0)
        count = count[..., None]
        return np.divide(grid, count, out=np.zeros_like(grid), where=count > 0.0)

    # --------------------------------------------------------------------- #
    # Boundary and integration
    # --------------------------------------------------------------------- #
    def _wall_force(self, pos: np.ndarray, vel: np.ndarray, r: np.ndarray) -> np.ndarray:
        N = pos.shape[0]
        f = np.zeros((N, 3), dtype=float)

        y = pos[:, 1]
        zrel = pos[:, 2] - self.center_z
        r_len = np.sqrt(y * y + zrel * zrel) + 1e-12

        # Cylindrical wall.
        pen = r_len + r - self.R
        mask = pen > 0.0
        if np.any(mask):
            ny = y[mask] / r_len[mask]
            nz = zrel[mask] / r_len[mask]

            # Normal force toward cylinder axis.
            f_n = -self.wall_k * pen[mask]
            f[mask, 1] += f_n * ny
            f[mask, 2] += f_n * nz

            # Normal damping.
            v_n = vel[mask, 1] * ny + vel[mask, 2] * nz
            f[mask, 1] -= self.wall_c * v_n * ny
            f[mask, 2] -= self.wall_c * v_n * nz

            # Tangential friction driving toward local wall velocity.
            vwy = -self.omega * zrel[mask]
            vwz = self.omega * y[mask]
            vrx = vel[mask, 0]
            vry = vel[mask, 1] - vwy
            vrz = vel[mask, 2] - vwz
            # Remove normal component.
            vrn = vry * ny + vrz * nz
            vry -= vrn * ny
            vrz -= vrn * nz

            mu_f = self.wall_mu * self.wall_k * pen[mask]
            f[mask, 0] -= mu_f * vrx
            f[mask, 1] -= mu_f * vry
            f[mask, 2] -= mu_f * vrz

        # End caps at x = +/- L/2.
        pen_left = self.xmin + r - pos[:, 0]
        mask = pen_left > 0.0
        f[mask, 0] += self.wall_k * pen_left[mask]
        f[mask, 0] -= self.wall_c * vel[mask, 0]

        pen_right = pos[:, 0] - (self.xmax - r)
        mask = pen_right > 0.0
        f[mask, 0] -= self.wall_k * pen_right[mask]
        f[mask, 0] -= self.wall_c * vel[mask, 0]

        return f

    def _confine(self) -> None:
        pos = self.position
        vel = self.velocity
        r = self.radius

        # Cylinder wall.
        y = pos[:, 1]
        zrel = pos[:, 2] - self.center_z
        r_len = np.sqrt(y * y + zrel * zrel) + 1e-12
        max_r = self.R - r - 1e-6
        mask = r_len > max_r
        if np.any(mask):
            factor = max_r[mask] / r_len[mask]
            pos[mask, 1] *= factor
            pos[mask, 2] = self.center_z + (pos[mask, 2] - self.center_z) * factor
            nx = y[mask] / r_len[mask]
            nz = zrel[mask] / r_len[mask]
            v_n = vel[mask, 1] * nx + vel[mask, 2] * nz
            vel[mask, 1] -= v_n * nx
            vel[mask, 2] -= v_n * nz

        # End caps.
        pos[:, 0] = np.clip(pos[:, 0], self.xmin + r, self.xmax - r)

    def _substep(self) -> None:
        pos = self.position
        vel = self.velocity
        m = self.mass[:, None]
        r = self.radius

        # Gravity.
        f_gravity = self.mass[:, None] * self.g[None, :]

        # Fluid drag against rotating carrier.
        y = pos[:, 1]
        zrel = pos[:, 2] - self.center_z
        v_fluid = np.zeros_like(pos)
        v_fluid[:, 1] = -self.omega * zrel
        v_fluid[:, 2] = self.omega * y
        f_drag = -self.drag_coeff * (vel - v_fluid)

        # Pressure (multiphysics interaction).
        f_pressure = self._pressure_force(pos, r)

        # Wall contact with rotating boundary.
        f_wall = self._wall_force(pos, vel, r)

        # Radial segregation: larger particles migrate outward.
        r_len = np.sqrt(y * y + zrel * zrel) + 1e-12
        r_vec = np.zeros_like(pos)
        r_vec[:, 1] = y / r_len
        r_vec[:, 2] = zrel / r_len
        r_avg = np.mean(r)
        f_radial = self.seg_strength * (r - r_avg)[:, None] * r_vec

        # Axial banding seed: species are driven into opposite axial phases.
        x = pos[:, 0]
        band_signal = np.sin(2.0 * math.pi * x / self.L)
        f_band = np.zeros_like(pos)
        direction = np.where(self.species == 0, 1.0, -1.0)
        f_band[:, 0] = self.band_strength * direction * band_signal

        # Concentration feedback to grow axial bands.
        f_conc = self._concentration_force(pos)

        f_total = f_gravity + f_drag + f_wall + f_radial + f_band + f_conc + f_pressure

        # Explicit Euler substep.
        self.velocity = vel + (f_total / m) * self.dt
        self.position = pos + self.velocity * self.dt

        # Velocity clamping for numerical safety.
        speed = np.linalg.norm(self.velocity, axis=1)
        too_fast = speed > self.vmax
        if np.any(too_fast):
            scale = self.vmax / speed[too_fast]
            self.velocity[too_fast] *= scale[:, None]

        self._confine()

    def _current_frame(self, time: float) -> StateFrame:
        pos = self.position
        vel = self.velocity
        r = self.radius

        # Recompute external forces for diagnostic export.
        y = pos[:, 1]
        zrel = pos[:, 2] - self.center_z
        f_gravity = self.mass[:, None] * self.g[None, :]

        v_fluid = np.zeros_like(pos)
        v_fluid[:, 1] = -self.omega * zrel
        v_fluid[:, 2] = self.omega * y
        f_drag = -self.drag_coeff * (vel - v_fluid)

        f_wall = self._wall_force(pos, vel, r)

        r_len = np.sqrt(y * y + zrel * zrel) + 1e-12
        r_vec = np.zeros_like(pos)
        r_vec[:, 1] = y / r_len
        r_vec[:, 2] = zrel / r_len
        r_avg = np.mean(r)
        f_radial = self.seg_strength * (r - r_avg)[:, None] * r_vec

        x = pos[:, 0]
        band_signal = np.sin(2.0 * math.pi * x / self.L)
        f_band = np.zeros_like(pos)
        direction = np.where(self.species == 0, 1.0, -1.0)
        f_band[:, 0] = self.band_strength * direction * band_signal

        f_conc = self._concentration_force(pos)

        f_external = f_gravity + f_drag + f_wall + f_radial + f_band + f_conc

        # Interaction (pressure) force aggregated per species.
        f_pressure = self._pressure_force(pos, r)
        interaction = np.zeros((self.species_count, 3), dtype=float)
        for k in range(self.species_count):
            mask = self.species == k
            if np.any(mask):
                interaction[k] = f_pressure[mask].sum(axis=0)

        velocity_field = self._velocity_field(pos, vel)
        cumulative_mass_source = np.zeros(self.species_count, dtype=float)

        return StateFrame(
            time=float(time),
            position=np.array(self.position, copy=True),
            velocity=np.array(self.velocity, copy=True),
            mass=np.array(self.mass, copy=True),
            body_id=np.array(self.body_id, copy=True),
            attributes={
                "species_id": np.array(self.species, copy=True),
                "particle_radius": np.array(self.radius, copy=True),
                "mass_quantity": np.array(self.mass_quantity, copy=True),
            },
            fields={
                "cumulative_mass_source": cumulative_mass_source,
                "velocity_field": velocity_field,
                "active_mask": np.array(self._active_mask, copy=True),
                "external_force": f_external,
                "interaction_force": interaction,
            },
            scalars={
                "grid_spacing": float(self.grid_spacing),
                "characteristic_length": float(self.characteristic_length),
                "contact_length_scale": float(self.contact_length_scale),
            },
        )

    # --------------------------------------------------------------------- #
    # Public Solver protocol
    # --------------------------------------------------------------------- #
    def advance_to(self, time: float) -> StateFrame:
        if time < self.time:
            self.time = float(time)
            return self._current_frame(time)

        delta = time - self.time
        steps = int(round(delta / self.dt))
        for _ in range(steps):
            self._substep()

        # Handle any small remainder (should be zero for aligned sample times).
        remaining = time - self.time - steps * self.dt
        if remaining > 1e-12:
            saved_dt = self.dt
            self.dt = remaining
            self._substep()
            self.dt = saved_dt

        self.time = float(time)
        return self._current_frame(time)

    def diagnostics(self) -> dict[str, Any]:
        return {"substep_dt": self.dt, "substeps_per_sample": self.substeps}
