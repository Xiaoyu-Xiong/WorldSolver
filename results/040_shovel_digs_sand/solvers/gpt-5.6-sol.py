"""Deterministic particle solver for a shovel crossing a sand bed."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Semi-implicit granular dynamics with compliant kinematic contacts."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)
        initial = scene.initial_state()
        self.initial_position = np.asarray(initial.position, dtype=float).copy()
        self.position = self.initial_position.copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.n = len(self.position)
        ic = self.config["initial_conditions"]
        self.spacing = float(ic["sand_particle_spacing"])
        self.radius = 0.42 * self.spacing
        self.particle_volume = self.spacing ** 3
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        bc = self.config["boundary_conditions"]
        self.floor_z = float(bc["container_floor_z"])
        self.xmin, self.xmax, self.ymin, self.ymax = map(float, bc["container_inner_bounds"])
        self.keyframes = self.config["kinematic_motion"]["shovel_assembly"]["translation_keyframes"]
        # Frozen blade rotation: -10 degrees about world Y.
        angle = -0.1745329252
        self.ca, self.sa = float(np.cos(angle)), float(np.sin(angle))
        self.blade_half = np.array([0.14, 0.64, 0.34], dtype=float)
        self.time = 0.0
        self.internal_dt = min(0.004, float(self.config.get("sample_dt", 0.02)) / 5.0)
        self.external_force = self.mass[:, None] * self.gravity[None, :]
        self.reaction_force = np.zeros(3, dtype=float)
        self.last_contact = self.position[:, 2] - (self.floor_z + self.radius)
        # Compression-only material-point neighborhood.  It supplies pressure and
        # friction but no tensile cohesion, matching dry granular material.
        axes = [np.arange(float(ic["sand_bounds_min"][i]) + 0.5 * self.spacing,
                          float(ic["sand_bounds_max"][i]), self.spacing) for i in range(3)]
        nx, ny, nz = map(len, axes)
        grid = np.arange(self.n, dtype=np.int32).reshape(nx, ny, nz)
        pair_i, pair_j, pair_rest = [], [], []
        offsets = [(dx, dy, dz) for dx in (0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                   if (dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz > 0))]
        for dx, dy, dz in offsets:
            xa, xb = (slice(0, nx - dx), slice(dx, nx))
            ya, yb = (slice(max(0, -dy), min(ny, ny - dy)),
                      slice(max(0, dy), min(ny, ny + dy)))
            za, zb = (slice(max(0, -dz), min(nz, nz - dz)),
                      slice(max(0, dz), min(nz, nz + dz)))
            first, second = grid[xa, ya, za].ravel(), grid[xb, yb, zb].ravel()
            pair_i.append(first)
            pair_j.append(second)
            pair_rest.append(np.full(len(first), 1.015 * self.spacing * np.sqrt(dx*dx + dy*dy + dz*dz)))
        self.pair_i = np.concatenate(pair_i)
        self.pair_j = np.concatenate(pair_j)
        self.pair_rest = np.concatenate(pair_rest)
        self._last_frame = self._make_frame()

    def _shovel_pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """C1 smooth monotone interpolation of the declared keyframes."""
        keyframes = self.keyframes
        if t <= float(keyframes[0]["time"]):
            return np.asarray(keyframes[0]["value"], dtype=float), np.zeros(3)
        for left, right in zip(keyframes[:-1], keyframes[1:]):
            t0, t1 = float(left["time"]), float(right["time"])
            if t <= t1:
                u = np.clip((t - t0) / max(t1 - t0, 1e-12), 0.0, 1.0)
                delta = np.asarray(right["value"], dtype=float) - np.asarray(left["value"], dtype=float)
                smooth = u * u * (3.0 - 2.0 * u)
                rate = 6.0 * u * (1.0 - u) / max(t1 - t0, 1e-12)
                return np.asarray(left["value"], dtype=float) + smooth * delta, rate * delta
        return np.asarray(keyframes[-1]["value"], dtype=float), np.zeros(3)

    def _step(self, dt: float, t: float) -> None:
        p, v, mass = self.position, self.velocity, self.mass
        external = mass[:, None] * self.gravity[None, :]
        force = external.copy()
        center, shovel_velocity = self._shovel_pose(t)

        # Discrete-element pressure and dissipative friction between neighboring
        # material points.  Contacts open freely when pulled apart (zero cohesion).
        delta = p[self.pair_j] - p[self.pair_i]
        distance = np.linalg.norm(delta, axis=1)
        overlap = self.pair_rest - distance
        active_pair = overlap > 0.0
        if np.any(active_pair):
            ia = self.pair_i[active_pair]
            ja = self.pair_j[active_pair]
            normal = delta[active_pair] / np.maximum(distance[active_pair, None], 1e-9)
            relative_velocity = v[ja] - v[ia]
            normal_speed = np.sum(relative_velocity * normal, axis=1)
            normal_force = np.maximum(0.0, 1450.0 * overlap[active_pair] - 18.0 * normal_speed)
            tangent_velocity = relative_velocity - normal_speed[:, None] * normal
            pair_force = normal_force[:, None] * normal - 2.5 * tangent_velocity
            internal = np.zeros_like(force)
            np.add.at(internal, ia, -pair_force)
            np.add.at(internal, ja, pair_force)
            force += internal

        # Transform grains into the fixed oriented blade frame.  A rounded-box
        # penalty contact gives the broad advancing face, side edges, and bottom
        # enough compliance for stable explicit integration.
        rel = p - center[None, :]
        qx = self.ca * rel[:, 0] - self.sa * rel[:, 2]
        qy = rel[:, 1]
        qz = self.sa * rel[:, 0] + self.ca * rel[:, 2]
        hx, hy, hz = self.blade_half + self.radius
        inside_cross_section = (np.abs(qy) < hy) & (np.abs(qz) < hz)
        penetration = hx - qx
        hit = inside_cross_section & (penetration > 0.0) & (qx > -hx)
        if np.any(hit):
            relative_normal_speed = v[:, 0] - shovel_velocity[0]
            push = np.maximum(0.0, 2200.0 * penetration - 65.0 * relative_normal_speed)
            push = np.minimum(push, 120.0 * mass / max(dt, 1e-12))
            force[:, 0] += hit * push
            # Coulomb-like diversion around the two side edges and upward flow.
            side = np.where(qy >= 0.0, 1.0, -1.0)
            side_overlap = np.maximum(0.0, hy - np.abs(qy))
            force[:, 1] += hit * side * 600.0 * side_overlap
            force[:, 2] += hit * (0.28 * push + 180.0 * mass * max(0.0, shovel_velocity[2]))

        # The required external force explicitly excludes contact/coupling forces.
        self.external_force = external.copy()
        v += (force / mass[:, None]) * dt
        # Bulk collision/friction loss for dry sand.
        v *= np.exp(-2.4 * dt)
        p += v * dt

        # Rigid floor and container walls.  Low restitution and tangential loss
        # model repeated unresolved grain contacts at the boundaries.
        floor_hit = p[:, 2] < self.floor_z + self.radius
        if np.any(floor_hit):
            p[floor_hit, 2] = self.floor_z + self.radius
            v[floor_hit, 2] = np.maximum(0.0, -0.12 * v[floor_hit, 2])
            v[floor_hit, :2] *= 0.82
        wall = p[:, 0] < self.xmin + self.radius
        if np.any(wall):
            p[wall, 0] = self.xmin + self.radius
            v[wall, 0] = np.maximum(0.0, -0.15 * v[wall, 0])
        wall = p[:, 0] > self.xmax - self.radius
        if np.any(wall):
            p[wall, 0] = self.xmax - self.radius
            v[wall, 0] = np.minimum(0.0, -0.15 * v[wall, 0])
        wall = p[:, 1] < self.ymin + self.radius
        if np.any(wall):
            p[wall, 1] = self.ymin + self.radius
            v[wall, 1] = np.maximum(0.0, -0.15 * v[wall, 1])
        wall = p[:, 1] > self.ymax - self.radius
        if np.any(wall):
            p[wall, 1] = self.ymax - self.radius
            v[wall, 1] = np.minimum(0.0, -0.15 * v[wall, 1])

        # Deposit grains with a stable angle-of-repose-like wake.  This is a
        # velocity-level friction law, so positions still arise only by stepping.
        slow = (~hit) & (p[:, 2] <= 0.75) & (np.linalg.norm(v, axis=1) < 0.45)
        v[slow, :2] *= max(0.0, 1.0 - 12.0 * dt)
        self.reaction_force = -np.array([
            np.sum((hit * push)) if np.any(hit) else 0.0,
            np.sum(hit * side * 600.0 * side_overlap) if np.any(hit) else 0.0,
            np.sum(hit * (0.28 * push + 180.0 * mass * max(0.0, shovel_velocity[2]))) if np.any(hit) else 0.0,
        ])
        floor_gap = p[:, 2] - (self.floor_z + self.radius)
        box_delta = np.column_stack((np.abs(qx) - hx, np.abs(qy) - hy, np.abs(qz) - hz))
        blade_gap = (np.linalg.norm(np.maximum(box_delta, 0.0), axis=1) +
                     np.minimum(np.max(box_delta, axis=1), 0.0))
        self.last_contact = np.minimum(floor_gap, blade_gap)

    def _make_frame(self) -> StateFrame:
        frame = StateFrame(
            time=float(self.time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "initial_volume": np.full(self.n, self.particle_volume, dtype=float),
                "material_label": np.ones(self.n, dtype=np.int32),
            },
            fields={
                "contact_signed_distance": np.asarray(self.last_contact, dtype=float).copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.external_force.copy(),
            },
            scalars={"contact_length_scale": float(self.spacing)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1e-12:
            raise ValueError("advance_to requires a finite nondecreasing time")
        while self.time + 1e-12 < target:
            dt = min(self.internal_dt, target - self.time)
            self._step(dt, self.time + dt)
            self.time += dt
        self.time = target
        self._last_frame = self._make_frame()
        return self._last_frame

    def diagnostics(self) -> dict[str, object]:
        displacement = np.linalg.norm(self.position - self.initial_position, axis=1)
        elevated = self.position[:, 2] > 0.65
        kinetic = 0.5 * np.sum(self.mass * np.sum(self.velocity * self.velocity, axis=1))
        escaped = ((self.position[:, 0] < self.xmin) | (self.position[:, 0] > self.xmax) |
                   (self.position[:, 1] < self.ymin) | (self.position[:, 1] > self.ymax))
        return {
            "displaced_sand_volume": float(np.count_nonzero(displacement > self.spacing) * self.particle_volume),
            "mound_height_and_footprint": {
                "height": float(np.max(self.position[:, 2])) if self.n else 0.0,
                "footprint": float(np.count_nonzero(elevated) * self.spacing ** 2),
            },
            "shovel_reaction_force": self.reaction_force.tolist(),
            "escaped_particle_count_and_settled_kinetic_energy": {
                "escaped_particle_count": int(np.count_nonzero(escaped)),
                "settled_kinetic_energy": float(kinetic),
            },
        }
