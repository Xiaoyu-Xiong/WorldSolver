"""Deterministic rigid-body solver for the frictional loop scene."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Fixed-step rolling sphere/vertical-loop integrator."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        config = scene.config
        initial = scene.initial_state()
        self.time = 0.0
        self.position = np.asarray(initial.position[0], dtype=float).copy()
        self.velocity = np.asarray(initial.velocity[0], dtype=float).copy()
        self.mass = float(initial.mass[0]) if initial.mass is not None else 1.0
        geometry = config.get("scene_geometry", {}).get("objects", [])
        loop = next((obj for obj in geometry if obj.get("id") == "loop_track"), None)
        ball = next((obj for obj in geometry if obj.get("id") == "ball"), None)
        translation = loop.get("transform", {}).get("translation", [0.0, 0.0, 1.2]) if loop else [0.0, 0.0, 1.2]
        self.center = np.asarray(translation, dtype=float)
        self.ball_radius = float(ball.get("geometry", {}).get("radius", 0.24) if ball else 0.24)
        radial = self.position[[0, 2]] - self.center[[0, 2]]
        self.rail_radius = float(np.linalg.norm(radial))
        self.theta = math.atan2(radial[0], -radial[1])
        self.theta_velocity = float(np.dot(self.velocity, self._tangent(self.theta)) / self.rail_radius)
        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.g = max(0.0, -float(self.gravity[2]))
        self.friction = max(0.0, float(config.get("friction_coefficient", 0.18)))
        sample_dt = float(config.get("sample_dt", 0.02))
        self.step_size = min(0.001, sample_dt / 20.0)
        self.effective_mass_ratio = 1.4  # solid sphere translation + spin
        self.restitution = 0.10
        self.mode = "contact"
        self.spin_angle = 0.0
        self.angular_velocity = np.array([0.0, -self.rail_radius * self.theta_velocity / self.ball_radius, 0.0])
        self.detachments = 0
        self.impacts = 0
        self.first_detachment_time = None
        self.minimum_normal_force = float("inf")

    @staticmethod
    def _tangent(theta: float) -> np.ndarray:
        return np.array([math.cos(theta), 0.0, math.sin(theta)])

    @staticmethod
    def _outward(theta: float) -> np.ndarray:
        return np.array([math.sin(theta), 0.0, -math.cos(theta)])

    def _normal_force(self) -> float:
        speed = self.rail_radius * self.theta_velocity
        return self.mass * (speed * speed / self.rail_radius + self.g * math.cos(self.theta))

    def _release(self) -> None:
        self.mode = "flight"
        self.detachments += 1
        if self.first_detachment_time is None:
            self.first_detachment_time = self.time

    def _step_contact(self, dt: float) -> None:
        normal = self._normal_force()
        self.minimum_normal_force = min(self.minimum_normal_force, normal)
        if normal <= 0.0:
            self._release()
            self._step_flight(dt)
            return
        speed = self.rail_radius * self.theta_velocity
        drive = -self.g * math.sin(self.theta)
        resistance = self.friction * self.g
        if abs(speed) < 1e-8:
            if abs(drive) <= resistance:
                acceleration = 0.0
                self.theta_velocity = 0.0
            else:
                direction = math.copysign(1.0, drive)
                acceleration = (drive - resistance * direction) / self.effective_mass_ratio
        else:
            direction = math.copysign(1.0, speed)
            acceleration = (drive - resistance * direction) / self.effective_mass_ratio
        old = self.theta_velocity
        self.theta_velocity = old + acceleration * dt / self.rail_radius
        if old * self.theta_velocity < 0.0 and abs(drive) <= resistance:
            self.theta_velocity = 0.0
        self.theta += self.theta_velocity * dt
        tangent = self._tangent(self.theta)
        self.position = self.center + self.rail_radius * self._outward(self.theta)
        self.position[1] = self.center[1]
        self.velocity = self.rail_radius * self.theta_velocity * tangent
        self.angular_velocity[:] = (0.0, -self.rail_radius * self.theta_velocity / self.ball_radius, 0.0)
        self.spin_angle += self.angular_velocity[1] * dt
        updated = self._normal_force()
        self.minimum_normal_force = min(self.minimum_normal_force, updated)
        if updated <= 0.0:
            self._release()

    def _step_flight(self, dt: float) -> None:
        self.velocity += self.gravity * dt
        self.position += self.velocity * dt
        self.spin_angle += self.angular_velocity[1] * dt
        floor_z = self.ball_radius
        if self.position[2] < floor_z:
            self.position[2] = floor_z
            if self.velocity[2] < 0.0:
                impact_speed = -self.velocity[2]
                self.velocity[2] = self.restitution * impact_speed
                horizontal = self.velocity[:2]
                damping = min(1.0, self.friction * (1.0 + self.restitution))
                self.velocity[:2] = horizontal * (1.0 - damping)
                self.impacts += 1
                if self.velocity[2] < 0.05 and float(np.linalg.norm(self.velocity[:2])) < 0.05:
                    self.velocity[:] = 0.0
        relative = self.position - self.center
        radius = math.hypot(float(relative[0]), float(relative[2]))
        if radius <= self.rail_radius:
            return
        outward = np.array([relative[0] / radius, 0.0, relative[2] / radius])
        self.position = self.center + self.rail_radius * outward
        self.position[1] = self.center[1]
        outward_speed = float(np.dot(self.velocity, outward))
        if outward_speed <= 0.0:
            return
        self.velocity -= (1.0 + self.restitution) * outward_speed * outward
        theta = math.atan2(float(outward[0]), -float(outward[2]))
        tangent = self._tangent(theta)
        tangent_speed = float(np.dot(self.velocity, tangent))
        friction_delta = min(abs(tangent_speed), self.friction * (1.0 + self.restitution) * outward_speed)
        if tangent_speed != 0.0:
            self.velocity -= math.copysign(friction_delta, tangent_speed) * tangent
        self.impacts += 1
        post = float(np.dot(self.velocity, tangent))
        feasible = self.mass * (post * post / self.rail_radius + self.g * math.cos(theta))
        if self.restitution * outward_speed < 0.20 and feasible > 0.0:
            self.mode = "contact"
            self.theta = theta
            self.theta_velocity = post / self.rail_radius
            self.velocity = post * tangent
            self.angular_velocity[:] = (0.0, -post / self.ball_radius, 0.0)

    def _step(self, dt: float) -> None:
        if self.mode == "contact":
            self._step_contact(dt)
        else:
            self._step_flight(dt)
        self.time += dt

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1e-10:
            raise ValueError("advance_to times must be finite and nondecreasing")
        while self.time + 1e-12 < target:
            self._step(min(self.step_size, target - self.time))
        self.time = target
        relative = self.position - self.center
        radius = math.hypot(float(relative[0]), float(relative[2]))
        orientation = np.array([[0.0, math.sin(self.spin_angle / 2.0), 0.0, math.cos(self.spin_angle / 2.0)]])
        fields = {
            "constraint_target": np.array([self.rail_radius]),
            "constraint_value": np.array([radius]),
            "contact_signed_distance": np.array([self.rail_radius - radius]),
            "external_force": np.array([self.mass * self.gravity]),
        }
        frame = StateFrame(
            time=target,
            position=self.position.reshape(1, 3).copy(),
            velocity=self.velocity.reshape(1, 3).copy(),
            orientation=orientation,
            angular_velocity=self.angular_velocity.reshape(1, 3).copy(),
            mass=np.array([self.mass]), body_id=np.array([0], dtype=int),
            attributes={"rolling": np.array([1.0 if self.mode == "contact" else 0.0])},
            fields=fields,
            scalars={"constraint_scale": self.rail_radius, "contact_length_scale": self.ball_radius},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        minimum = self.minimum_normal_force if math.isfinite(self.minimum_normal_force) else 0.0
        return {"method": "fixed_step_rolling_sphere_with_contact_impulses", "integration_step": self.step_size,
                "detachments": self.detachments, "first_detachment_time": self.first_detachment_time,
                "rail_impacts": self.impacts, "minimum_normal_force": minimum}
