"""Deterministic mass-spring solver for the knitted patch benchmark.

The sheet is represented by lumped vertex masses joined by orthotropic yarn,
shear, and bending springs. A semi-implicit Euler integrator advances the
system with small internal time steps. The left wale is fixed while the
right wale follows a smooth extension/partial-unloading cycle.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physically step a coherent, compliant knit-like shell."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # The model is deliberately deterministic.
        self.config = scene.config
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.rest_position = self.position.copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.zeros(len(self.position), dtype=np.int32)
        self.nx, self.ny = map(int, self.config["cloth"]["grid_shape"])
        if self.nx * self.ny != len(self.position):
            raise ValueError("cloth grid_shape does not match the initial state")

        self.gravity = np.asarray(self.config["gravity"], dtype=np.float64)
        self.thickness = float(self.config["cloth"]["thickness"])
        self.current_time = 0.0
        self.internal_dt = 1.0 / 720.0
        self.step_count = 0

        grid = np.arange(self.nx * self.ny, dtype=np.int32).reshape(self.ny, self.nx)
        self.left = grid[:, 0].copy()
        self.right = grid[:, -1].copy()
        self.pinned = np.zeros(len(self.position), dtype=bool)
        self.pinned[self.left] = True
        self.pinned[self.right] = True

        boundary = np.zeros((self.ny, self.nx), dtype=np.int32)
        boundary[[0, -1], :] = 1
        boundary[:, [0, -1]] = 1
        self.boundary = boundary.ravel()

        # Spring families mimic the easy initial rotation and subsequent yarn
        # locking of a knit. Compression stiffness is reduced in _forces so
        # the sheet can wrinkle instead of behaving like a solid membrane.
        pairs: list[tuple[int, int]] = []
        stiffness: list[float] = []
        damping: list[float] = []
        locking: list[float] = []

        def add_family(a, b, k: float, c: float, lock: float) -> None:
            aa = np.asarray(a, dtype=np.int32).ravel()
            bb = np.asarray(b, dtype=np.int32).ravel()
            pairs.extend(zip(aa.tolist(), bb.tolist()))
            stiffness.extend([k] * len(aa))
            damping.extend([c] * len(aa))
            locking.extend([lock] * len(aa))

        # Course and wale springs are deliberately orthotropic.
        add_family(grid[:, :-1], grid[:, 1:], 10.0, 0.0040, 1.11)
        add_family(grid[:-1, :], grid[1:, :], 15.0, 0.0045, 1.08)
        # Both cell diagonals provide shear coupling and natural transverse
        # contraction when the courses are pulled apart.
        add_family(grid[:-1, :-1], grid[1:, 1:], 4.2, 0.0025, 1.16)
        add_family(grid[1:, :-1], grid[:-1, 1:], 4.2, 0.0025, 1.16)
        # Two-hop springs are a simple discrete bending/curvature resistance.
        add_family(grid[:, :-2], grid[:, 2:], 0.75, 0.0015, 1.30)
        add_family(grid[:-2, :], grid[2:, :], 0.95, 0.0015, 1.30)

        pair_array = np.asarray(pairs, dtype=np.int32)
        self.spring_i = pair_array[:, 0]
        self.spring_j = pair_array[:, 1]
        delta = self.rest_position[self.spring_j] - self.rest_position[self.spring_i]
        self.spring_rest = np.linalg.norm(delta, axis=1)
        self.spring_k = np.asarray(stiffness, dtype=np.float64)
        self.spring_c = np.asarray(damping, dtype=np.float64)
        self.spring_lock = np.asarray(locking, dtype=np.float64)

        # A tiny alternating rest-length bias represents yarn crimp. It does
        # not prescribe motion: it acts only through the same spring law and
        # seeds stitch-scale wrinkles under load.
        parity = ((self.spring_i % self.nx) + (self.spring_i // self.nx)) & 1
        bias = np.where(parity == 0, 0.9975, 1.0025)
        self.spring_rest *= bias

        self._last_external = self.mass[:, None] * self.gravity[None, :]

    @staticmethod
    def _smoothstep5(u: float) -> tuple[float, float, float]:
        """Quintic position, first derivative, and second derivative."""
        u = float(np.clip(u, 0.0, 1.0))
        s = u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
        ds = 30.0 * u * u * (1.0 - u) * (1.0 - u)
        dds = 60.0 * u * (1.0 - u) * (1.0 - 2.0 * u)
        return s, ds, dds

    def _grip_motion(self, time: float) -> tuple[float, float, float]:
        """Right-grip displacement, velocity, and acceleration along x."""
        if time <= 0.35:
            return 0.0, 0.0, 0.0
        if time < 4.25:
            span = 3.90
            s, ds, dds = self._smoothstep5((time - 0.35) / span)
            return 0.90 * s, 0.90 * ds / span, 0.90 * dds / (span * span)
        if time <= 5.25:
            return 0.90, 0.0, 0.0
        if time < 6.50:
            span = 1.25
            s, ds, dds = self._smoothstep5((time - 5.25) / span)
            return 0.90 - 0.25 * s, -0.25 * ds / span, -0.25 * dds / (span * span)
        return 0.65, 0.0, 0.0

    @staticmethod
    def _box_distance_and_normal(points, center, half_extent):
        local = points - center[None, :]
        q = np.abs(local) - half_extent[None, :]
        positive = np.maximum(q, 0.0)
        outside_length = np.linalg.norm(positive, axis=1)
        signed = outside_length + np.minimum(np.max(q, axis=1), 0.0)

        normal = np.zeros_like(points)
        outside = outside_length > 1.0e-12
        normal[outside] = (
            positive[outside] / outside_length[outside, None]
        ) * np.sign(local[outside])
        inside = ~outside
        if np.any(inside):
            axes = np.argmax(q[inside], axis=1)
            rows = np.flatnonzero(inside)
            signs = np.sign(local[rows, axes])
            signs[signs == 0.0] = 1.0
            normal[rows, axes] = signs
        return signed, normal

    def _forces(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        force = self.mass[:, None] * self.gravity[None, :]

        delta = self.position[self.spring_j] - self.position[self.spring_i]
        length = np.linalg.norm(delta, axis=1)
        length = np.maximum(length, 1.0e-12)
        direction = delta / length[:, None]
        strain = length - self.spring_rest
        ratio = length / self.spring_rest

        # Loops are soft while rotating, then become rapidly stiffer as yarn
        # segments straighten. Cubic locking is smooth and conservative.
        compression_scale = np.where(strain >= 0.0, 1.0, 0.18)
        extension = np.maximum(ratio - self.spring_lock, 0.0)
        elastic = self.spring_k * compression_scale * strain
        elastic += 120.0 * self.spring_rest * extension * extension * extension
        relative_v = self.velocity[self.spring_j] - self.velocity[self.spring_i]
        viscous = self.spring_c * np.einsum("ij,ij->i", relative_v, direction)
        spring_force = (elastic + viscous)[:, None] * direction
        np.add.at(force, self.spring_i, spring_force)
        np.add.at(force, self.spring_j, -spring_force)

        # Mild air drag removes unresolved high-frequency vibration.
        force -= 0.0011 * self.velocity

        # Ground contact at z=0, including shell half-thickness.
        radius = 0.5 * self.thickness
        gap = self.position[:, 2] - radius
        active = (gap < 0.0) & (~self.pinned)
        if np.any(active):
            normal_speed = self.velocity[active, 2]
            force[active, 2] += -900.0 * gap[active] - 0.12 * np.minimum(normal_speed, 0.0)

        # Collision with the two kinematic grip boxes. Attached edge masses
        # are excluded because their placement inside the jaws is intentional.
        displacement, grip_speed, _ = self._grip_motion(time)
        centers = (
            np.array([-1.5, 0.0, 2.15]),
            np.array([1.5 + displacement, 0.0, 2.15]),
        )
        half = np.array([0.12, 1.25, 0.12])
        for center, obstacle_vx in ((centers[0], 0.0), (centers[1], grip_speed)):
            distance, normal = self._box_distance_and_normal(self.position, center, half)
            active = (distance < radius) & (~self.pinned)
            if np.any(active):
                relative = self.velocity[active].copy()
                relative[:, 0] -= obstacle_vx
                vn = np.einsum("ij,ij->i", relative, normal[active])
                magnitude = 700.0 * (radius - distance[active]) - 0.10 * np.minimum(vn, 0.0)
                force[active] += magnitude[:, None] * normal[active]

        # Report gravity and the force supplied by the jaws. Spring/contact
        # pairs remain internal and are not included for free samples.
        external = self.mass[:, None] * self.gravity[None, :]
        _, _, grip_accel = self._grip_motion(time)
        desired_accel = np.zeros((len(self.position), 3), dtype=np.float64)
        desired_accel[self.right, 0] = grip_accel
        actuator = self.mass[:, None] * desired_accel - force
        external[self.pinned] += actuator[self.pinned]
        return force, external

    def _apply_grips(self, time: float) -> None:
        displacement, speed, _ = self._grip_motion(time)
        self.position[self.left] = self.rest_position[self.left]
        self.velocity[self.left] = 0.0
        self.position[self.right] = self.rest_position[self.right]
        self.position[self.right, 0] += displacement
        self.velocity[self.right] = 0.0
        self.velocity[self.right, 0] = speed

    def _substep(self, dt: float) -> None:
        self._apply_grips(self.current_time)
        force, _ = self._forces(self.current_time)
        free = ~self.pinned
        self.velocity[free] += dt * force[free] / self.mass[free, None]
        self.position[free] += dt * self.velocity[free]

        # Position-level nonpenetration complements the compliant contact law.
        floor = 0.5 * self.thickness
        hit = free & (self.position[:, 2] < floor)
        if np.any(hit):
            self.position[hit, 2] = floor
            self.velocity[hit, 2] = np.maximum(self.velocity[hit, 2], 0.0) * 0.08
            self.velocity[hit, :2] *= 0.82

        self.current_time += dt
        self._apply_grips(self.current_time)
        self.step_count += 1

    def _deformation_gradient(self) -> np.ndarray:
        p = self.position.reshape(self.ny, self.nx, 3)
        r = self.rest_position.reshape(self.ny, self.nx, 3)
        du = np.empty_like(p)
        dv = np.empty_like(p)

        du[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (
            r[:, 2:, 0] - r[:, :-2, 0]
        )[:, :, None]
        du[:, 0] = (p[:, 1] - p[:, 0]) / (r[:, 1, 0] - r[:, 0, 0])[:, None]
        du[:, -1] = (p[:, -1] - p[:, -2]) / (r[:, -1, 0] - r[:, -2, 0])[:, None]

        rest_dy = r[2:, :, 1] - r[:-2, :, 1]
        dv[1:-1] = (p[2:] - p[:-2]) / rest_dy[:, :, None]
        dv[0] = (p[1] - p[0]) / (r[1, :, 1] - r[0, :, 1])[:, None]
        dv[-1] = (p[-1] - p[-2]) / (r[-1, :, 1] - r[-2, :, 1])[:, None]

        u_length = np.maximum(np.linalg.norm(du, axis=2), 1.0e-12)
        shear = np.einsum("ijk,ijk->ij", du, dv) / u_length
        v_normal = np.sqrt(
            np.maximum(np.einsum("ijk,ijk->ij", dv, dv) - shear * shear, 0.0)
        )
        result = np.zeros((self.ny, self.nx, 2, 2), dtype=np.float64)
        result[:, :, 0, 0] = u_length
        result[:, :, 0, 1] = shear
        result[:, :, 1, 1] = v_normal
        return result.reshape(-1, 2, 2)

    def _contact_distances(self, time: float) -> np.ndarray:
        radius = 0.5 * self.thickness
        ground = self.position[:, 2] - radius
        displacement, _, _ = self._grip_motion(time)
        half = np.array([0.12, 1.25, 0.12])
        left, _ = self._box_distance_and_normal(
            self.position, np.array([-1.5, 0.0, 2.15]), half
        )
        right, _ = self._box_distance_and_normal(
            self.position, np.array([1.5 + displacement, 0.0, 2.15]), half
        )
        return np.minimum(ground, np.minimum(left, right) - radius)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.current_time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing times")
        while self.current_time < target - 1.0e-12:
            self._substep(min(self.internal_dt, target - self.current_time))
        # Snap away accumulated floating-point time error at sample boundaries.
        self.current_time = target
        self._apply_grips(target)
        _, self._last_external = self._forces(target)

        frame = StateFrame(
            time=target,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": self._deformation_gradient(),
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
            },
            fields={
                "contact_signed_distance": self._contact_distances(target),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": self._last_external.copy(),
            },
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "semi_implicit_euler_mass_spring",
            "internal_steps": int(self.step_count),
            "spring_count": int(len(self.spring_i)),
            "final_time": float(self.current_time),
        }
