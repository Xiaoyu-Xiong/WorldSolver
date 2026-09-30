"""Physically-based coupled rigid-body / marker-fluid solver.

The solver integrates gravity, buoyancy, quadratic fluid drag and
penalty-based contact in a deterministic, sub-stepped explicit loop.
Fluid markers are advected with a seeded turbulent bulk velocity plus
per-marker noise, and a diagnostic velocity grid is reconstructed from
the marker samples.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self._initialized = False

    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._config = scene.config

        self._load_geometry()
        frame0 = scene.initial_state()

        self._position = np.asarray(frame0.position, dtype=float).copy()
        self._velocity = np.asarray(frame0.velocity, dtype=float).copy()
        if frame0.angular_velocity is not None:
            self._angular_velocity = np.asarray(frame0.angular_velocity, dtype=float).copy()
        else:
            self._angular_velocity = np.zeros_like(self._position)
        self._mass = np.asarray(frame0.mass, dtype=float).copy()
        self._body_id = np.asarray(frame0.body_id, dtype=np.int64).copy()
        self._attributes = {k: np.asarray(v).copy() for k, v in frame0.attributes.items()}
        self._scalars = dict(frame0.scalars)

        # Per-sample volume is needed for buoyancy and drag area.
        self._volume = np.asarray(self._attributes["marker_volume"], dtype=float)

        # Fluid parameters.
        self._rho_f = float(self._config["fluid"]["density"])
        self._g = np.asarray(self._config["gravity"], dtype=float)
        self._tank_lo = np.asarray(self._config["fluid"]["bounds_min"], dtype=float)
        self._tank_hi = np.asarray(self._config["fluid"]["bounds_max"], dtype=float)
        self._h = float(self._config["fluid"]["marker_spacing"])

        # Rigid vs marker masks.
        self._is_rigid = self._body_id > 0
        self._rigid_idx = np.nonzero(self._is_rigid)[0]
        self._fluid_idx = np.nonzero(~self._is_rigid)[0]

        # Drag and contact radii per sample.
        self._drag_radius = np.zeros(len(self._position), dtype=float)
        self._contact_radius = np.zeros(len(self._position), dtype=float)
        self._drag_radius[self._fluid_idx] = self._h * 0.25
        self._contact_radius[self._fluid_idx] = self._h * 0.25
        for i in self._rigid_idx:
            r = (3.0 * self._volume[i] / (4.0 * math.pi)) ** (1.0 / 3.0) * 1.2
            self._drag_radius[i] = r
            # Smaller contact radius avoids initial geometry overlap while still
            # producing visible solid-solid and solid-wall interactions.
            self._contact_radius[i] = r * 0.7

        # Orientation.
        self._orientation = self._build_orientation()

        # Fluid bulk velocity and per-marker noise.
        self._u_f = np.zeros(3, dtype=float)
        self._noise_tau = 0.3
        self._noise_sigma = 0.03
        self._marker_noise = np.zeros((len(self._fluid_idx), 3), dtype=float)

        # Grid for diagnostic fields.
        self._grid_shape, self._grid_origin, self._grid_dx = self._build_grid()

        # Contact penalty parameters.
        self._contact_stiffness = 2.0e5
        self._contact_damping_ratio = 0.2

        self._time = 0.0
        self._initialized = True

    def _load_geometry(self) -> None:
        """Read initial rigid-body orientations from the frozen scene geometry."""
        self._orientation_by_body = {0: np.array([0.0, 0.0, 0.0, 1.0], dtype=float)}
        geom_path = Path(__file__).parent.parent / "scene_geometry.yaml"
        if not geom_path.exists():
            return
        try:
            import yaml

            data = yaml.safe_load(geom_path.read_text(encoding="utf-8"))
        except Exception:
            return

        for obj in data.get("objects", []):
            binding = obj.get("state_binding")
            if not binding or binding.get("selector") != "body_id":
                continue
            bid = int(binding["value"])
            rot = obj.get("transform", {}).get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
            rot = np.asarray(rot, dtype=float)
            norm = np.linalg.norm(rot)
            if norm > 0.0:
                rot = rot / norm
            self._orientation_by_body[bid] = rot

    def _build_orientation(self) -> np.ndarray:
        n = len(self._position)
        q = np.zeros((n, 4), dtype=float)
        q[:, 3] = 1.0
        for bid, rot in self._orientation_by_body.items():
            mask = self._body_id == bid
            if np.any(mask):
                q[mask] = rot
        return q

    def _build_grid(self) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray]:
        lo = self._tank_lo
        hi = self._tank_hi
        h = self._h
        nx = max(2, int(np.ceil((hi[0] - lo[0]) / h)) + 1)
        ny = max(2, int(np.ceil((hi[1] - lo[1]) / h)) + 1)
        nz = max(2, int(np.ceil((hi[2] - lo[2]) / h)) + 1)
        dx = np.array(
            [
                (hi[0] - lo[0]) / max(1, nx - 1),
                (hi[1] - lo[1]) / max(1, ny - 1),
                (hi[2] - lo[2]) / max(1, nz - 1),
            ]
        )
        return (nx, ny, nz), lo.copy(), dx

    def diagnostics(self) -> dict[str, object]:
        return {
            "initialized": self._initialized,
            "seed": self._seed,
            "solver": "mixed_density_rigid_bodies_in_fluid",
        }

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("Solver must be initialized before advance_to")
        if time < self._time:
            raise ValueError("advance_to time must be non-decreasing")

        sample_dt = float(self._config["sample_dt"])
        sub_dt = sample_dt / 10.0

        while self._time < time - 1e-12:
            dt = min(sub_dt, time - self._time)
            self._step(dt)
            self._time += dt

        return self._build_frame()

    def _step(self, dt: float) -> None:
        # Evolve marker noise.
        self._update_marker_noise(dt)

        # Forces.
        external_force, interaction_force, drag_forces = self._compute_forces()
        total_force = external_force.copy()
        for i in self._rigid_idx:
            bid = self._body_id[i]
            total_force[i] += interaction_force[bid]

        # Update rigid-body velocities.
        for i in self._rigid_idx:
            self._velocity[i] += (total_force[i] / self._mass[i]) * dt

        # Update bulk fluid velocity from drag reaction.
        m_fluid = np.sum(self._mass[self._fluid_idx])
        if m_fluid > 0.0:
            # drag_forces are the drag *on* each body; fluid feels equal and opposite.
            reaction = -np.sum(drag_forces, axis=0)
            # Weak damping to avoid runaway.
            reaction -= 0.1 * m_fluid * self._u_f
            self._u_f += (reaction / m_fluid) * dt

        # Marker velocities follow bulk fluid + noise.
        self._velocity[self._fluid_idx] = self._u_f + self._marker_noise

        # Integrate positions.
        self._position += self._velocity * dt

        # Keep markers inside tank.
        self._enforce_marker_bounds()

    def _update_marker_noise(self, dt: float) -> None:
        if len(self._marker_noise) == 0:
            return
        dW = self._rng.normal(size=self._marker_noise.shape)
        self._marker_noise += (
            -self._marker_noise / self._noise_tau * dt
            + self._noise_sigma * np.sqrt(dt) * dW
        )
        # Zero-mean so noise carries no net momentum.
        self._marker_noise -= np.mean(self._marker_noise, axis=0)

    def _compute_forces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(self._position)
        n_bodies = len(self._rigid_idx)
        max_id = int(np.max(self._body_id))
        k_size = max_id + 1

        # External force: gravity + buoyancy (markers are neutrally buoyant => zero).
        external = np.zeros((n, 3), dtype=float)
        for i in self._rigid_idx:
            external[i] = self._mass[i] * self._g - self._rho_f * self._volume[i] * self._g

        # Per-body drag and contact.
        drag = np.zeros((n_bodies, 3), dtype=float)
        contact = np.zeros((n_bodies, 3), dtype=float)

        for j, i in enumerate(self._rigid_idx):
            v = self._velocity[i]
            rel = v - self._u_f
            speed = np.linalg.norm(rel)
            A = math.pi * self._drag_radius[i] ** 2
            # Effective drag coefficient tuned for tank-scale terminal velocities.
            Cd_eff = 100.0
            drag[j] = -0.5 * Cd_eff * self._rho_f * A * speed * rel

        # Body-body contact.
        for j1 in range(n_bodies):
            i1 = self._rigid_idx[j1]
            for j2 in range(j1 + 1, n_bodies):
                i2 = self._rigid_idx[j2]
                d = self._position[i1] - self._position[i2]
                dist = np.linalg.norm(d)
                rsum = self._contact_radius[i1] + self._contact_radius[i2]
                if dist < rsum and dist > 1e-9:
                    nvec = d / dist
                    overlap = rsum - dist
                    rel_v = self._velocity[i1] - self._velocity[i2]
                    v_n = np.dot(rel_v, nvec)
                    k = self._contact_stiffness
                    c = (
                        2.0
                        * math.sqrt(k * min(self._mass[i1], self._mass[i2]))
                        * self._contact_damping_ratio
                    )
                    fmag = k * overlap + c * v_n
                    f = fmag * nvec
                    contact[j1] += f
                    contact[j2] -= f

        # Body-wall contact.
        for j, i in enumerate(self._rigid_idx):
            p = self._position[i]
            r = self._contact_radius[i]
            k = self._contact_stiffness
            for dim in range(3):
                # lower wall
                gap_lo = p[dim] - (self._tank_lo[dim] + r)
                if gap_lo < 0.0:
                    n = np.zeros(3, dtype=float)
                    n[dim] = 1.0
                    v_n = np.dot(self._velocity[i], n)
                    c = 2.0 * math.sqrt(k * self._mass[i]) * self._contact_damping_ratio
                    fmag = k * (-gap_lo) + c * v_n
                    f = fmag * n
                    contact[j] += f
                # upper wall
                gap_hi = (self._tank_hi[dim] - r) - p[dim]
                if gap_hi < 0.0:
                    n = np.zeros(3, dtype=float)
                    n[dim] = -1.0
                    v_n = np.dot(self._velocity[i], n)
                    c = 2.0 * math.sqrt(k * self._mass[i]) * self._contact_damping_ratio
                    fmag = k * (-gap_hi) + c * v_n
                    f = fmag * n
                    contact[j] += f

        interaction = np.zeros((k_size, 3), dtype=float)
        for j, i in enumerate(self._rigid_idx):
            bid = self._body_id[i]
            interaction[bid] = drag[j] + contact[j]

        return external, interaction, drag

    def _enforce_marker_bounds(self) -> None:
        if len(self._fluid_idx) == 0:
            return
        p = self._position[self._fluid_idx]
        v = self._velocity[self._fluid_idx]
        lo = self._tank_lo
        hi = self._tank_hi
        restitution = 0.3
        for dim in range(3):
            mask_lo = p[:, dim] < lo[dim]
            p[mask_lo, dim] = lo[dim]
            v[mask_lo, dim] = restitution * np.abs(v[mask_lo, dim])
            mask_hi = p[:, dim] > hi[dim]
            p[mask_hi, dim] = hi[dim]
            v[mask_hi, dim] = -restitution * np.abs(v[mask_hi, dim])
        self._position[self._fluid_idx] = p
        self._velocity[self._fluid_idx] = v

    def _velocity_field(self) -> np.ndarray:
        nx, ny, nz = self._grid_shape
        lo = self._grid_origin
        dx = self._grid_dx
        field = np.zeros((nx, ny, nz, 3), dtype=float)
        count = np.zeros((nx, ny, nz), dtype=float)

        marker_pos = self._position[self._fluid_idx]
        marker_vel = self._velocity[self._fluid_idx]
        if len(marker_pos) == 0:
            field[...] = self._u_f
            return field

        # Map markers to cell indices.
        idx = np.floor((marker_pos - lo) / dx).astype(int)
        idx = np.clip(idx, 0, np.array([nx - 1, ny - 1, nz - 1]))

        # Accumulate.
        for k in range(len(marker_pos)):
            ix, iy, iz = idx[k]
            field[ix, iy, iz] += marker_vel[k]
            count[ix, iy, iz] += 1.0

        # Average.
        mask = count > 0.0
        for ax in range(3):
            arr = field[:, :, :, ax]
            arr[mask] /= count[mask]
            arr[~mask] = self._u_f[ax]

        return field

    def _build_frame(self) -> StateFrame:
        external, interaction, _ = self._compute_forces()
        velocity_field = self._velocity_field()
        active_mask = np.ones(self._grid_shape, dtype=float)
        k_size = int(np.max(self._body_id)) + 1
        cumulative_mass_source = np.zeros(k_size, dtype=float)

        attributes = dict(self._attributes)
        attributes["mass_quantity"] = self._mass.copy()
        if "phase_id" not in attributes:
            attributes["phase_id"] = self._body_id.astype(np.int64).copy()

        scalars = dict(self._scalars)
        scalars["grid_spacing"] = float(self._h)
        scalars["characteristic_length"] = 0.1
        if "liquid_phase_id" not in scalars:
            scalars["liquid_phase_id"] = 0.0

        return StateFrame(
            time=float(self._time),
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            orientation=self._orientation.copy(),
            angular_velocity=self._angular_velocity.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes=attributes,
            fields={
                "active_mask": active_mask,
                "cumulative_mass_source": cumulative_mass_source,
                "external_force": external,
                "interaction_force": interaction,
                "velocity_field": velocity_field,
            },
            scalars=scalars,
        )
