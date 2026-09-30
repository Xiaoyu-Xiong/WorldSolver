"""Solver-neutral initial liquid and prescribed rigid boundary, never fluid evolution."""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkScene:
    def __init__(self, config: dict):
        self.config = config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"],
            "seed": int(self.config["seed"]),
            "phenomenon": "rotating_blade_stirring",
            "spatial_dimension": 3,
            "fluid_body_id": 0,
            "blade_body_id": int(self.config["blade"]["body_id"]),
            "reference": "Power Particles, Figure 5, PDF page 6",
            "parameter_provenance": "benchmark choices, not paper dimensions",
        }

    def blade_kinematics(self, time: float) -> dict:
        """Analytic boundary condition: integral of a smoothstep speed ramp.

        Negative rotation about +z is clockwise when seen from above. This
        helper describes the actuator only; the solver must compute liquid motion.
        """
        if not np.isfinite(time) or time < 0:
            raise ValueError("time must be finite and nonnegative")
        blade = self.config["blade"]
        ramp = float(blade["ramp_duration"])
        speed = float(blade["angular_speed"])
        if ramp <= 0:
            raise ValueError("ramp_duration must be positive")
        s = min(time / ramp, 1.0)
        omega = speed * (3 * s**2 - 2 * s**3)
        integral = ramp * (s**3 - 0.5 * s**4) + max(time - ramp, 0.0)
        angle = float(blade["initial_angle"]) + speed * integral
        c, s = np.cos(angle), np.sin(angle)
        return {
            "position": np.asarray(blade["center"], dtype=float),
            "orientation": np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]),
            "velocity": np.zeros(3),
            "angular_velocity": np.array([0.0, 0.0, omega]),
            "rotation_matrix": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
            "angle": angle,
        }

    def blade_signed_distance(self, points: np.ndarray, time: float) -> np.ndarray:
        """Exact oriented-box distance, positive outside the blade."""
        pose = self.blade_kinematics(time)
        local = (np.asarray(points) - pose["position"]) @ pose["rotation_matrix"]
        q = np.abs(local) - np.asarray(self.config["blade"]["half_extents"])
        return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)

    def blade_surface_samples(self, time: float, spacing: float = 0.05) -> dict:
        """All six solid faces; the solver selects CURRENT wetted patches.

        Target velocities are analytic boundary data, NOT measured fluid velocities.
        Returned normals point outward from the blade into the surrounding medium.
        """
        if spacing <= 0:
            raise ValueError("spacing must be positive")
        half = np.asarray(self.config["blade"]["half_extents"], dtype=float)
        positions, normals = [], []
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            count = np.maximum(np.ceil(2 * half[other] / spacing).astype(int), 1)
            coords = [
                -half[a] + (np.arange(n) + 0.5) * (2 * half[a] / n)
                for a, n in zip(other, count, strict=True)
            ]
            uv = np.stack(np.meshgrid(*coords, indexing="ij"), axis=-1).reshape(-1, 2)
            for sign in (-1, 1):
                p = np.zeros((len(uv), 3))
                p[:, other] = uv
                p[:, axis] = sign * half[axis]
                n = np.zeros_like(p)
                n[:, axis] = sign
                positions.append(p)
                normals.append(n)
        pose = self.blade_kinematics(time)
        relative = np.concatenate(positions) @ pose["rotation_matrix"].T
        return {
            "position": relative + pose["position"],
            "normal": np.concatenate(normals) @ pose["rotation_matrix"].T,
            "target_velocity": np.cross(pose["angular_velocity"], relative),
        }

    def initial_tank_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """Wetted tank patches at t=0 only; normals point into the liquid."""
        half = float(self.config["tank"]["interior_half_width"])
        depth = float(self.config["liquid"]["depth"])
        h = float(self.config["domain"]["grid_spacing"])
        xy = np.arange(-half + h / 2, half, h)
        z = np.arange(h / 2, depth, h)
        floor = np.stack(np.meshgrid(xy, xy, indexing="ij"), axis=-1).reshape(-1, 2)
        positions = [np.column_stack((floor, np.zeros(len(floor))))]
        normals = [np.tile([0.0, 0.0, 1.0], (len(floor), 1))]
        side = np.stack(np.meshgrid(xy, z, indexing="ij"), axis=-1).reshape(-1, 2)
        for axis in (0, 1):
            for sign in (-1, 1):
                p = np.zeros((len(side), 3))
                p[:, axis] = sign * half
                p[:, 1 - axis] = side[:, 0]
                p[:, 2] = side[:, 1]
                n = np.zeros_like(p)
                n[:, axis] = -sign
                positions.append(p)
                normals.append(n)
        return np.concatenate(positions), np.concatenate(normals)

    def initial_state(self) -> StateFrame:
        liquid, domain = self.config["liquid"], self.config["domain"]
        half = float(self.config["tank"]["interior_half_width"])
        depth, spacing = float(liquid["depth"]), float(liquid["particle_spacing"])
        xy = np.arange(-half + spacing / 2, half, spacing)
        z = np.arange(spacing / 2, depth, spacing)
        points = np.stack(np.meshgrid(xy, xy, z, indexing="ij"), axis=-1).reshape(-1, 3)
        # Keep representative samples half a cell away from the solid blade.
        points = points[self.blade_signed_distance(points, 0.0) >= spacing / 2 - 1e-12]
        pose = self.blade_kinematics(0.0)
        blade_half = np.asarray(self.config["blade"]["half_extents"])
        submerged = np.clip(depth - (pose["position"][2] - blade_half[2]), 0, 2 * blade_half[2])
        fluid_volume = (2 * half) ** 2 * depth - 4 * blade_half[0] * blade_half[1] * submerged
        # Quadrature weights sum to the analytic liquid volume excluding the blade.
        volume = fluid_volume / len(points)
        count = len(points) + 1
        position = np.vstack((points, pose["position"]))
        body_id = np.zeros(count, dtype=np.int64)
        body_id[-1] = int(self.config["blade"]["body_id"])
        orientation = np.tile([0.0, 0.0, 0.0, 1.0], (count, 1))
        orientation[-1] = pose["orientation"]
        mass_quantity = np.r_[np.full(len(points), volume * float(liquid["density"])), 0.0]

        h = float(domain["grid_spacing"])
        axes = [
            float(lo) + (np.arange(n) + 0.5) * h
            for lo, n in zip(domain["bounds_min"], domain["reference_grid"], strict=True)
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        phi = grid[..., 2] - depth
        solid_distance = self.blade_signed_distance(grid, 0.0)
        wet = (phi < 0) & (solid_distance > 0)
        # Do not evaluate divergence on stencils crossing air or solid boundaries.
        active = (
            (phi < -h)
            & (grid[..., 2] > h)
            & (np.abs(grid[..., 0]) < half - h)
            & (np.abs(grid[..., 1]) < half - h)
            & (solid_distance > h)
        )
        wall_position, wall_normal = self.initial_tank_samples()
        blade_samples = self.blade_surface_samples(0.0)
        wetted = blade_samples["position"][:, 2] < depth
        blade_position = blade_samples["position"][wetted]
        frame = StateFrame(
            time=0.0,
            position=position,
            velocity=np.zeros_like(position),
            orientation=orientation,
            angular_velocity=np.zeros_like(position),
            # Kinematic actuator has no simulated mass; liquid-only balance uses mass_quantity.
            mass=mass_quantity.copy(),
            body_id=body_id,
            attributes={
                "mass_quantity": mass_quantity,
                "particle_volume": np.r_[np.full(len(points), volume), 0.0],
                "particle_radius": np.r_[np.full(len(points), spacing / 2), 0.0],
            },
            fields={
                "velocity_field": np.zeros((*phi.shape, 3)),
                "active_mask": active.astype(float),
                # Liquid-air level set extends THROUGH solids; not a solid SDF.
                "liquid_level_set": phi,
                "density_field": wet.astype(float) * float(liquid["density"]),
                "cumulative_mass_source": np.zeros(1),
                "boundary_position": wall_position,
                "boundary_normal": wall_normal,
                "boundary_velocity": np.zeros_like(wall_position),
                "boundary_target_velocity": np.zeros_like(wall_position),
                "blade_boundary_position": blade_position,
                "blade_boundary_normal": blade_samples["normal"][wetted],
                "blade_boundary_velocity": np.zeros_like(blade_position),
                "blade_boundary_target_velocity": blade_samples["target_velocity"][wetted],
                "grid_origin": np.asarray(domain["bounds_min"], dtype=float),
            },
            scalars={
                "grid_spacing": h,
                "characteristic_length": float(2 * blade_half[0]),
                "reference_velocity": float(
                    abs(self.config["blade"]["angular_speed"]) * blade_half[0]
                ),
                "initial_liquid_volume": float(fluid_volume),
            },
        )
        frame.validate()
        return frame

    def sample_times(self) -> list[float]:
        count = int(round(float(self.config["duration"]) / float(self.config["sample_dt"])))
        return np.linspace(0.0, float(self.config["duration"]), count + 1).tolist()
