"""Deterministic particle solver for the ground-impact oobleck scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        frame = scene.initial_state()
        self._position = np.asarray(frame.position, dtype=np.float64).copy()
        self._velocity = np.asarray(frame.velocity, dtype=np.float64).copy()
        self._mass = np.asarray(frame.mass, dtype=np.float64).copy()
        self._body_id = np.asarray(frame.body_id, dtype=np.int32).copy()
        self._initial_position = self._position.copy()
        self._time = 0.0
        cfg = scene.config
        initial = cfg["initial_conditions"]
        self._gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self._floor_z = float(cfg["boundary_conditions"]["ground"]["plane_z"])
        self._spacing = float(initial.get("particle_spacing", 0.045))
        self._particle_radius = 0.5 * self._spacing
        self._body_indices = {bid: np.flatnonzero(self._body_id == bid) for bid in (1, 2)}
        self._rest_offsets = np.zeros_like(self._position)
        for bid, idx in self._body_indices.items():
            self._rest_offsets[idx] = self._initial_position[idx] - self._initial_position[idx].mean(axis=0)
        self._contact_events = 0
        self._max_penetration = 0.0

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self._time - 1e-10:
            raise ValueError("advance_to requires finite nondecreasing time")
        remaining = target - self._time
        while remaining > 1e-12:
            dt = min(0.004, remaining)
            self._substep(dt)
            self._time += dt
            remaining -= dt
        return self._make_frame(target)

    def _substep(self, dt: float) -> None:
        pos, vel = self._position, self._velocity
        force = self._mass[:, None] * self._gravity[None, :]
        for bid, idx in self._body_indices.items():
            if idx.size == 0:
                continue
            centroid = np.average(pos[idx], axis=0, weights=self._mass[idx])
            vmean = np.average(vel[idx], axis=0, weights=self._mass[idx])
            rel, relv = pos[idx] - centroid, vel[idx] - vmean
            shear = np.linalg.norm(relv, axis=1)
            if bid == 2:
                k, damping = 520.0 * (1.0 + 1.35 * np.mean(shear)), 15.0 + 4.0 * np.mean(shear)
            else:
                k, damping = 255.0, 24.0
            force[idx] += self._mass[idx, None] * (k * (self._rest_offsets[idx] - rel) - damping * relv)
            gap = pos[idx, 2] - (self._floor_z + self._particle_radius)
            contact = gap < 0.018
            if np.any(contact):
                radial = rel[:, :2]
                radial = radial / np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-8)
                pressure = (10.0 if bid == 1 else 18.0) * (1.0 + np.maximum(0.0, -vel[idx, 2]))
                force[idx[contact], :2] += self._mass[idx[contact], None] * pressure * radial[contact]
                rate = (1.35 if bid == 1 else 0.72) * dt
                self._rest_offsets[idx[contact]] += rate * (rel[contact] - self._rest_offsets[idx[contact]])
        vel += dt * force / self._mass[:, None]
        pos += dt * vel
        floor = self._floor_z + self._particle_radius
        penetrating = pos[:, 2] < floor
        if np.any(penetrating):
            depth = floor - pos[penetrating, 2]
            self._max_penetration = max(self._max_penetration, float(np.max(depth)))
            self._contact_events += int(np.count_nonzero(penetrating))
            pos[penetrating, 2] = floor
            ids = self._body_id[penetrating]
            restitution = np.where(ids == 2, 0.62, 0.28)
            incoming = vel[penetrating, 2] < 0.0
            vz = vel[penetrating, 2]
            vz[incoming] *= -restitution[incoming]
            vel[penetrating, 2] = vz
            vel[penetrating, :2] *= 0.90
        speed = np.linalg.norm(vel, axis=1)
        too_fast = speed > 35.0
        if np.any(too_fast):
            vel[too_fast] *= (35.0 / speed[too_fast])[:, None]

    def _make_frame(self, time: float) -> StateFrame:
        n = len(self._position)
        deformation = np.tile(np.eye(3, dtype=np.float64), (n, 1, 1))
        for bid, idx in self._body_indices.items():
            if idx.size:
                rest, cur = self._rest_offsets[idx], self._position[idx] - self._position[idx].mean(axis=0)
                cov = rest.T @ rest
                F = (cur.T @ rest) @ np.linalg.pinv(cov + 1e-8 * np.eye(3))
                if np.isfinite(F).all():
                    deformation[idx] = np.clip(F, -5.0, 5.0)
        external = self._mass[:, None] * self._gravity[None, :]
        frame = StateFrame(
            time=time, position=self._position.copy(), velocity=self._velocity.copy(),
            mass=self._mass.copy(), body_id=self._body_id.copy(),
            attributes={"particle_radius": np.full(n, self._particle_radius), "rest_position": self._initial_position.copy(),
                        "material_label": self._body_id.copy(), "mass_quantity": self._mass.copy(),
                        "deformation_gradient": deformation},
            fields={"contact_signed_distance": self._position[:, 2] - (self._floor_z + self._particle_radius),
                    "cumulative_mass_source": np.zeros(1), "external_force": external},
            scalars={"contact_length_scale": self._particle_radius, "total_mass": float(np.sum(self._mass))},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "particle_kelvin_voigt_oobleck", "particle_count": int(len(self._position)),
                "contact_events": int(self._contact_events), "max_penetration": float(self._max_penetration)}
