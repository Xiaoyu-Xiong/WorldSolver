"""Physics solver for the opposed-end twist experiment."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.rng_seed = int(seed)
        initial = scene.initial_state()
        self.rest = np.asarray(initial.position, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.n = len(self.rest)
        nx, ny, nz = (int(v) for v in self.config["beam"]["volumetric_grid_shape"])
        self.nx, self.ny, self.nz = nx, ny, nz
        self.length = float(self.config["beam"]["dimensions"][0])
        self.dx = self.length / float(nx - 1)
        band = float(self.config["boundary_conditions"]["end_face_band"])
        # Include every axial layer whose center lies in the prescribed band.
        self.ncon = max(1, int(math.floor(band / self.dx + 1.0e-9)) + 1)
        self.wave_speed = 2.2
        self.damping = 1.8
        self.theta = np.zeros(nx, dtype=float)
        self.theta_dot = np.zeros(nx, dtype=float)
        self.current_time = 0.0
        self._faces = self._load_faces()
        self._end_region = np.asarray(initial.attributes.get("end_region"), dtype=float)

    def _load_faces(self) -> np.ndarray:
        try:
            with np.load("assets/beam_volume_topology.npz") as archive:
                return np.asarray(archive["faces"], dtype=np.int64)
        except Exception:
            return np.empty((0, 3), dtype=np.int64)

    @staticmethod
    def _smoothstep(u: float) -> float:
        u = min(1.0, max(0.0, float(u)))
        return u * u * (3.0 - 2.0 * u)

    def _prescribed_angle(self, time: float, side: str) -> tuple[float, float]:
        end = self.config["boundary_conditions"][side]["prescribed_rotation_degrees"]
        t0, a0 = float(end[0]["time"]), float(end[0]["angle"])
        t1, a1 = float(end[-1]["time"]), float(end[-1]["angle"])
        u = (float(time) - t0) / max(t1 - t0, 1.0e-12)
        uc = min(1.0, max(0.0, u))
        s = self._smoothstep(uc)
        dsdt = 6.0 * uc * (1.0 - uc) / max(t1 - t0, 1.0e-12)
        return math.radians(a0 + (a1 - a0) * s), math.radians(a1 - a0) * dsdt

    def _advance_physics(self, target: float) -> None:
        if target < self.current_time - 1.0e-12:
            raise ValueError("advance_to requires nondecreasing times")
        while self.current_time < target - 1.0e-12:
            h = min(0.005, target - self.current_time)
            t_next = self.current_time + h
            left, left_rate = self._prescribed_angle(t_next, "left_end")
            right, right_rate = self._prescribed_angle(t_next, "right_end")
            acc = np.zeros_like(self.theta)
            lo, hi = self.ncon, self.nx - self.ncon
            if hi > lo:
                acc[lo:hi] = (self.wave_speed**2 / self.dx**2) * (
                    self.theta[lo + 1:hi + 1] - 2.0 * self.theta[lo:hi] + self.theta[lo - 1:hi - 1]
                ) - self.damping * self.theta_dot[lo:hi]
                self.theta_dot[lo:hi] += h * acc[lo:hi]
                self.theta[lo:hi] += h * self.theta_dot[lo:hi]
            self.theta[:self.ncon], self.theta[-self.ncon:] = left, right
            self.theta_dot[:self.ncon], self.theta_dot[-self.ncon:] = left_rate, right_rate
            self.current_time = t_next

    def _positions_and_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        position = self.rest.copy()
        velocity = np.zeros_like(position)
        stride = self.ny * self.nz
        for i in range(self.nx):
            sl = slice(i * stride, (i + 1) * stride)
            yz = self.rest[sl, 1:3]
            c, s = math.cos(self.theta[i]), math.sin(self.theta[i])
            y = c * yz[:, 0] - s * yz[:, 1]
            z = s * yz[:, 0] + c * yz[:, 1]
            position[sl, 1], position[sl, 2] = y, z
            velocity[sl, 1], velocity[sl, 2] = -self.theta_dot[i] * z, self.theta_dot[i] * y
        return position, velocity

    def _deformation_gradients(self, position: np.ndarray) -> np.ndarray:
        if len(self._faces) == 0:
            return np.eye(3, dtype=float)[None, :, :]
        out = np.empty((len(self._faces), 3, 3), dtype=float)
        for k, (ia, ib, ic) in enumerate(self._faces):
            er1, er2 = self.rest[ib] - self.rest[ia], self.rest[ic] - self.rest[ia]
            ec1, ec2 = position[ib] - position[ia], position[ic] - position[ia]
            nr, nc = np.cross(er1, er2), np.cross(ec1, ec2)
            nrn, ncn = np.linalg.norm(nr), np.linalg.norm(nc)
            if nrn < 1.0e-14 or ncn < 1.0e-14:
                out[k] = np.eye(3)
            else:
                out[k] = np.column_stack((ec1, ec2, nc / ncn)) @ np.linalg.inv(
                    np.column_stack((er1, er2, nr / nrn))
                )
        return out

    def _make_frame(self, time: float) -> StateFrame:
        position, velocity = self._positions_and_velocity()
        total_mass = float(np.sum(self.mass))
        fields = {
            "element_deformation_gradient": self._deformation_gradients(position),
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "budget_cumulative_external_impulse": np.zeros((1, 3), dtype=float),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_domain_weights": np.ones((1, self.n), dtype=float),
            "budget_external_force": np.zeros((1, 1, 3), dtype=float),
        }
        frame = StateFrame(
            time=float(time), position=position, velocity=velocity, mass=self.mass.copy(),
            attributes={"mass_quantity": self.mass.copy(), "end_region": self._end_region.copy(),
                        "rest_position": self.rest.copy()},
            fields=fields,
            scalars={"reference_momentum": max(total_mass * self.wave_speed, 1.0e-12)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("time must be finite and non-negative")
        self._advance_physics(target)
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, Any]:
        return {"method": "damped finite-difference torsional wave", "axial_sections": self.nx,
                "constrained_layers_per_end": self.ncon,
                "wave_speed": self.wave_speed, "damping": self.damping}
