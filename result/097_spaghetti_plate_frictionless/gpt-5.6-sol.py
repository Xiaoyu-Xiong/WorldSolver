"""Deterministic, self-contained elastic-rod solver for the spaghetti scene.

The discretisation is a position based (SHAKE-like) integrator. Each authored
polyline is an inextensible chain, with a compliant straightness constraint for
bending resistance. Plate and strand contacts are unilateral and frictionless.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.config = scene.config
        self.x = np.asarray(initial.position, dtype=float).copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=int).copy()
        self.base_attributes = {
            key: np.asarray(value).copy() for key, value in initial.attributes.items()
        }
        self.n = len(self.x)
        self.count = int(self.config["strand_count"])
        self.nv = int(self.config["strand"]["vertex_count"])
        self.radius = float(self.config["strand"]["radius"])
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.time = 0.0
        self.step_size = 0.002
        self.constraint_iterations = 5

        curves = self.x.reshape(self.count, self.nv, 3)
        self.rest_length = np.linalg.norm(curves[:, 1:] - curves[:, :-1], axis=2)
        rng = np.random.default_rng(int(seed))
        self.bend_weight = 0.018 * rng.uniform(0.85, 1.15, self.count)

        self.total_mass = float(np.sum(self.mass))
        self.gravity_force = self.total_mass * self.gravity
        self.contact_impulse = np.zeros(3, dtype=float)
        self.last_plate_normal = np.tile(np.array([0.0, 0.0, 1.0]), (self.n, 1))
        self.last_plate_force = np.zeros(self.n, dtype=float)
        self._symmetry_broken = False
        self._steps = 0

    @staticmethod
    def _profile(radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Height and radial slope of the authored piecewise-linear bowl."""
        rp = np.array([0.10, 0.88, 1.05, 1.14])
        zp = np.array([0.075, 0.080, 0.135, 0.220])
        height = np.interp(radius, rp, zp)
        slope = np.zeros_like(radius)
        for lo, hi in zip(range(3), range(1, 4)):
            mask = (radius >= rp[lo]) & (radius < rp[hi])
            slope[mask] = (zp[hi] - zp[lo]) / (rp[hi] - rp[lo])
        slope[radius >= rp[-1]] = (zp[-1] - zp[-2]) / (rp[-1] - rp[-2])
        return height, slope

    def _edge_constraints(self) -> None:
        q = self.x.reshape(self.count, self.nv, 3)
        d = q[:, 1:] - q[:, :-1]
        length = np.linalg.norm(d, axis=2)
        safe = np.maximum(length, 1.0e-12)
        correction = 0.5 * ((length - self.rest_length) / safe)[..., None] * d
        q[:, :-1] += correction
        q[:, 1:] -= correction

    def _bending_constraints(self) -> None:
        q = self.x.reshape(self.count, self.nv, 3)
        curvature = q[:, :-2] - 2.0 * q[:, 1:-1] + q[:, 2:]
        w = self.bend_weight[:, None, None]
        q[:, :-2] -= w * curvature
        q[:, 1:-1] += 2.0 * w * curvature
        q[:, 2:] -= w * curvature

    def _strand_contacts(self) -> None:
        cutoff = 2.0 * self.radius
        pairs = cKDTree(self.x).query_pairs(cutoff, output_type="ndarray")
        if pairs.size == 0:
            return
        i = pairs[:, 0]
        j = pairs[:, 1]
        keep = (self.body_id[i] != self.body_id[j]) | (np.abs(i - j) > 2)
        i, j = i[keep], j[keep]
        if len(i) == 0:
            return
        delta = self.x[j] - self.x[i]
        distance = np.linalg.norm(delta, axis=1)
        normal = np.zeros_like(delta)
        good = distance > 1.0e-12
        normal[good] = delta[good] / distance[good, None]
        if np.any(~good):
            angle = (i[~good] * 0.754877666 + j[~good] * 0.569840296) % 1.0
            angle *= 2.0 * np.pi
            normal[~good, 0] = np.cos(angle)
            normal[~good, 1] = np.sin(angle)
        amount = 0.5 * np.maximum(cutoff - distance, 0.0)
        corr = amount[:, None] * normal
        np.add.at(self.x, i, -corr)
        np.add.at(self.x, j, corr)

    def _plate_contacts(self) -> tuple[np.ndarray, np.ndarray]:
        xy = self.x[:, :2]
        radial = np.linalg.norm(xy, axis=1)
        height, slope = self._profile(radial)
        radial_dir = np.zeros_like(xy)
        nonzero = radial > 1.0e-12
        radial_dir[nonzero] = xy[nonzero] / radial[nonzero, None]
        normal = np.column_stack((-slope[:, None] * radial_dir, np.ones(self.n)))
        normal /= np.linalg.norm(normal, axis=1)[:, None]
        gap = (self.x[:, 2] - height) / np.sqrt(1.0 + slope * slope)
        penetration = np.maximum(self.radius - gap, 0.0)
        active = penetration > 0.0
        self.x[active] += penetration[active, None] * normal[active]

        wall_radius = 1.14 - self.radius
        wall = (radial > wall_radius) & (self.x[:, 2] < 0.22 + self.radius)
        if np.any(wall):
            inward = -radial_dir[wall]
            excess = radial[wall] - wall_radius
            self.x[wall, :2] += excess[:, None] * inward
            wall_indices = np.flatnonzero(wall)
            normal[wall_indices] = np.column_stack(
                (inward, np.zeros(len(wall_indices)))
            )
            penetration[wall_indices] += excess
            active[wall_indices] = True
        return active, normal

    def _substep(self, dt: float) -> None:
        old_x = self.x.copy()
        ballistic_v = self.v + self.gravity * dt
        self.x += ballistic_v * dt

        # Resolve the measure-zero perfect stacking symmetry with a sub-micron,
        # deterministic round-off perturbation; the authored t=0 frame is exact.
        if not self._symmetry_broken:
            phase = np.arange(self.count, dtype=float) * 2.399963229728653
            offset = 2.0e-7 * np.column_stack((np.cos(phase), np.sin(phase)))
            offset -= np.mean(offset, axis=0)
            self.x.reshape(self.count, self.nv, 3)[:, :, :2] += offset[:, None, :]
            self._symmetry_broken = True

        plate_active = np.zeros(self.n, dtype=bool)
        plate_normal = self.last_plate_normal
        for _ in range(self.constraint_iterations):
            self._edge_constraints()
            self._bending_constraints()
            self._strand_contacts()
            active, normals = self._plate_contacts()
            plate_active |= active
            plate_normal = normals

        new_v = (self.x - old_x) / dt
        correction_impulse = np.sum(
            self.mass[:, None] * (new_v - ballistic_v), axis=0
        )
        self.contact_impulse += correction_impulse
        self.v = new_v

        velocity_change = self.v - ballistic_v
        projected = np.maximum(np.sum(velocity_change * plate_normal, axis=1), 0.0)
        self.last_plate_force = self.mass * projected / dt
        self.last_plate_force[~plate_active] = 0.0
        self.last_plate_normal = plate_normal
        self._steps += 1

    def _frame(self) -> StateFrame:
        attributes = dict(self.base_attributes)
        attributes["mass_quantity"] = self.mass.copy()
        zero_budget = np.zeros((1, 3), dtype=float)
        fields = {
            "budget_cumulative_external_impulse": self.contact_impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": zero_budget.copy(),
            "budget_cumulative_outflow_momentum": zero_budget.copy(),
            "budget_domain_weights": np.ones((1, self.n), dtype=float),
            "budget_external_force": self.gravity_force.reshape(1, 1, 3).copy(),
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "friction_active_mask": (self.last_plate_force > 0.0).astype(float),
            "friction_dynamic_coefficient": np.zeros(self.n, dtype=float),
            "friction_normal": self.last_plate_normal.copy(),
            "friction_normal_force": self.last_plate_force.copy(),
            "friction_relative_velocity": self.v.copy(),
            "friction_static_coefficient": np.zeros(self.n, dtype=float),
            "friction_tangential_force": np.zeros((self.n, 3), dtype=float),
        }
        speed_scale = float(np.sqrt(np.linalg.norm(self.gravity) * 1.3))
        scalars = {
            "friction_stick_speed_tolerance": 1.0e-4,
            "reference_force": max(np.linalg.norm(self.gravity_force), 1.0e-9),
            "reference_momentum": max(self.total_mass * speed_scale, 1.0e-9),
            "reference_velocity": max(speed_scale, 1.0e-9),
            "strand_strand_static_friction": 0.0,
            "strand_strand_kinetic_friction": 0.0,
            "strand_plate_static_friction": 0.0,
            "strand_plate_kinetic_friction": 0.0,
        }
        frame = StateFrame(
            time=float(self.time), position=self.x.copy(), velocity=self.v.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes=attributes, fields=fields, scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-12:
            raise ValueError("advance_to requires finite nondecreasing times")
        while self.time < target - 1.0e-12:
            dt = min(self.step_size, target - self.time)
            self._substep(dt)
            self.time += dt
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position_based_elastic_rods",
            "substeps": self._steps,
            "time": float(self.time),
            "finite": bool(np.isfinite(self.x).all() and np.isfinite(self.v).all()),
        }
