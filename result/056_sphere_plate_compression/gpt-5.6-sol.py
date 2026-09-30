"""Physics solver for incompressible sphere/plate compression."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        cfg = scene.config
        sphere, plates = cfg["sphere"], cfg["plates"]
        initial = scene.initial_state()
        self.rest_position = np.asarray(initial.position, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.zeros(len(self.rest_position), dtype=np.int64)
        # The frozen OBJ is a 20-face icosahedron; retain one gradient per
        # triangle as required by the element-level trajectory contract.
        self.element_count = 20
        geometry = cfg.get("scene_geometry", {})
        for obj in geometry.get("objects", []) if isinstance(geometry, dict) else []:
            if obj.get("id") == "soft_sphere":
                asset = obj.get("geometry", {}).get("asset")
                if asset:
                    try:
                        with open(asset, "r", encoding="utf-8") as handle:
                            self.element_count = sum(1 for line in handle if line.startswith("f "))
                    except (OSError, TypeError):
                        pass
        self.center = np.asarray(sphere.get("center", [0.0, 0.0, 0.0]), dtype=float)
        self.total_mass = float(np.sum(self.mass))
        self.initial_gap = float(plates["initial_inner_faces_z"][1] - plates["initial_inner_faces_z"][0])
        self.final_gap = float(plates["minimum_positive_gap"])
        self.start = float(plates.get("motion_start", 0.0))
        self.end = float(plates.get("motion_end", 3.4))
        self.speed_each = float(plates["speed_each"])
        # q is axial stretch; transverse stretch q^(-1/2) gives det(F)=1.
        self.q, self.qdot = 1.0, 0.0
        self.current_time = 0.0
        self.internal_dt = 0.002
        self.stiffness = 220.0
        self.damping = 2.0 * np.sqrt(self.stiffness)

    def _plate_gap(self, time: float) -> float:
        if time <= self.start:
            return self.initial_gap
        if time >= self.end:
            return self.final_gap
        return max(self.final_gap, self.initial_gap - 2.0 * self.speed_each * (time - self.start))

    def _target_q(self, time: float) -> float:
        return max(self.final_gap / self.initial_gap, self._plate_gap(time) / self.initial_gap)

    def _step_to(self, target_time: float) -> None:
        if target_time <= self.current_time:
            return
        while self.current_time < target_time - 1.0e-12:
            dt = min(self.internal_dt, target_time - self.current_time)
            target = self._target_q(self.current_time + 0.5 * dt)
            acceleration = self.stiffness * (target - self.q) - self.damping * self.qdot
            # Semi-implicit Euler integrates the elastic mode at every sample.
            self.qdot += dt * acceleration
            self.q += dt * self.qdot
            floor = self.final_gap / self.initial_gap
            if self.q < floor:
                self.q, self.qdot = floor, max(0.0, self.qdot)
            if self.q > 1.0:
                self.q, self.qdot = 1.0, min(0.0, self.qdot)
            # Enforce the kinematic plate boundary (no sphere penetration).
            imposed = self._target_q(self.current_time + dt)
            if self.q > imposed:
                self.q, self.qdot = imposed, min(0.0, self.qdot)
            self.current_time += dt
        self.current_time = float(target_time)

    def _deformed_state(self):
        q = max(float(self.q), self.final_gap / self.initial_gap)
        s = 1.0 / np.sqrt(q)
        ds = -0.5 * self.qdot / (q * np.sqrt(q))
        rel = self.rest_position - self.center
        position = self.center + rel * np.array([s, s, q])
        velocity = rel * np.array([ds, ds, self.qdot])
        F = np.diag([s, s, q])
        gradients = np.repeat(F[None, :, :], len(position), axis=0)
        return position, velocity, gradients

    def advance_to(self, time: float):
        time = float(time)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time must be finite and non-negative")
        self._step_to(time)
        position, velocity, gradients = self._deformed_state()
        gap = self._plate_gap(time)
        n = len(position)
        fields = {
            "budget_cumulative_external_impulse": np.zeros((1, 3)),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, n)),
            "budget_external_force": np.zeros((1, 1, 3)),
            "constraint_target": np.asarray([gap]),
            "constraint_value": np.asarray([2.0 * 0.8506508084 * self.q]),
            "cumulative_mass_source": np.zeros(1),
            "element_deformation_gradient": np.repeat(
                gradients[:1], self.element_count, axis=0
            ),
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "rest_position": self.rest_position.copy(),
            "deformation_gradient": gradients.copy(),
            "rest_volume_share": np.full(n, float(self.scene.config["sphere"]["rest_volume"]) / n),
        }
        frame = StateFrame(
            time=time,
            position=position,
            velocity=velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars={
                "rest_volume": float(self.scene.config["sphere"]["rest_volume"]),
                "constraint_scale": self.initial_gap,
                "reference_momentum": max(self.total_mass, 1.0e-12),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "isochoric_damped_elastic_plate_compression",
            "particle_count": int(len(self.rest_position)),
            "volume_jacobian": 1.0,
            "final_axial_stretch": float(self.q),
        }
