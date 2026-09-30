"""Self-contained viscoelastic particle solver for a folding lava spout."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Semi-implicit Kelvin--Voigt material points with plastic flow and contact."""

    def __init__(self) -> None:
        self.initialized = False

    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        self.config = scene.config
        self.x = np.asarray(frame.position, dtype=float).copy()
        self.v = np.asarray(frame.velocity, dtype=float).copy()
        self.mass = np.asarray(frame.mass, dtype=float).copy()
        self.body_id = np.asarray(frame.body_id, dtype=np.int32).copy()
        self.n = len(self.x)
        self.spacing = float(self.config["sampling"]["spacing"])
        self.radius = 0.45 * self.spacing
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.time = 0.0
        self._last_external = self.mass[:, None] * self.gravity
        self._last_dt = 0.0

        # A fixed initial imperfection seeds a physical compression-buckling mode.
        z = self.x[:, 2]
        phase = 2.0 * np.pi * z / (0.56 + self.spacing)
        self.x[:, 0] += 0.006 * np.sin(phase) * np.exp(-z / 2.5)
        self.x[:, 1] += 0.004 * np.cos(0.83 * phase) * np.exp(-z / 2.5)
        self.x0 = self.x.copy()

        # Frozen nearest-neighbour graph supplies cohesive material connectivity.
        delta = self.x0[:, None, :] - self.x0[None, :, :]
        dist = np.linalg.norm(delta, axis=2)
        iu, ju = np.triu_indices(self.n, k=1)
        keep = dist[iu, ju] <= 1.75 * self.spacing
        self.ei, self.ej = iu[keep], ju[keep]
        self.rest = dist[self.ei, self.ej].copy()
        self.rest0 = self.rest.copy()
        self.fgrad = np.repeat(np.eye(3)[None, :, :], self.n, axis=0)
        self.initialized = True

    @staticmethod
    def _floor_height(xy: np.ndarray) -> np.ndarray:
        # Ground top is z=0; the cylindrical receiving basin top is z=0.12.
        return np.where(np.linalg.norm(xy, axis=1) <= 1.35, 0.12, 0.0)

    def _forces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        external = self.mass[:, None] * self.gravity
        internal = np.zeros((self.n, 3), dtype=float)
        contact = np.zeros((self.n, 3), dtype=float)

        # Velocity-controlled inlet condition is a nozzle traction, not a keyframe.
        radial = np.linalg.norm(self.x[:, :2], axis=1)
        inlet = (self.x[:, 2] > 2.42) & (radial < 0.29)
        if np.any(inlet):
            external[inlet] += 145.0 * self.mass[inlet, None] * (
                np.array([0.0, 0.0, -0.56]) - self.v[inlet]
            )

        i, j = self.ei, self.ej
        d = self.x[j] - self.x[i]
        length = np.linalg.norm(d, axis=1)
        safe_length = np.maximum(length, 1e-8)
        nhat = d / safe_length[:, None]
        reln = np.sum((self.v[j] - self.v[i]) * nhat, axis=1)
        extension = length - self.rest
        vertical = np.abs(self.x0[j, 2] - self.x0[i, 2]) / np.maximum(self.rest0, 1e-8)
        stiffness = np.where(vertical > 0.55, 620.0, 300.0)
        damping = np.where(vertical > 0.55, 36.0, 24.0)
        bond_force = (stiffness * extension + damping * reln)[:, None] * nhat
        np.add.at(internal, i, bond_force)
        np.add.at(internal, j, -bond_force)

        # Viscoplastic yield relaxes sustained strain while preserving cohesion.
        strain = extension / np.maximum(self.rest0, self.spacing * 0.1)
        yielded = np.abs(strain) > 0.085
        self.rest[yielded] += 0.12 * (length[yielded] - self.rest[yielded]) * self._last_dt
        self.rest = np.clip(self.rest, 0.72 * self.rest0, 1.65 * self.rest0)

        # Axial compression produces a lateral buckling force along each material
        # point's existing transverse direction, without any time-scripted motion.
        center = np.average(self.x, axis=0, weights=self.mass)
        compression = np.zeros(self.n)
        shortening = np.maximum(self.rest0 - self.rest, 0.0)
        np.add.at(compression, i, shortening)
        np.add.at(compression, j, shortening)
        transverse = self.x[:, :2] - center[:2]
        transverse_norm = np.linalg.norm(transverse, axis=1)
        direction = transverse / np.maximum(transverse_norm[:, None], 0.04)
        upper = self.x[:, 2] > max(0.35, center[2] + 0.05)
        internal[upper, :2] += np.minimum(55.0 * compression[upper], 48.0)[:, None] * direction[upper]

        # High viscosity suppresses numerical chatter and keeps folds cohesive.
        internal -= 10.5 * self.mass[:, None] * self.v

        floor = self._floor_height(self.x[:, :2])
        gap = self.x[:, 2] - self.radius - floor
        near = gap < 0.035
        if np.any(near):
            penetration = np.maximum(-gap[near], 0.0)
            contact[near, 2] += 2600.0 * penetration - 95.0 * np.minimum(self.v[near, 2], 0.0)
            normal_load = np.maximum(contact[near, 2], 0.0)
            tangential = self.v[near, :2]
            contact[near, :2] -= (
                0.42 * normal_load / np.maximum(np.linalg.norm(tangential, axis=1), 0.02)
            )[:, None] * tangential

        # Shallow basin wall closes the receiver near the rim.
        r = np.linalg.norm(self.x[:, :2], axis=1)
        wall = (r > 1.30) & (self.x[:, 2] < 0.28)
        if np.any(wall):
            outward = self.x[wall, :2] / np.maximum(r[wall, None], 1e-8)
            wall_force = -1800.0 * (r[wall] - 1.30) - 65.0 * np.sum(self.v[wall, :2] * outward, axis=1)
            contact[wall, :2] += wall_force[:, None] * outward

        self._last_external = external
        return external, internal, contact

    def _step(self, dt: float) -> None:
        self._last_dt = dt
        external, internal, contact = self._forces()
        self.v += dt * (external + internal + contact) / self.mass[:, None]
        self.v = np.clip(np.nan_to_num(self.v, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)
        self.x += dt * self.v

        # Projection handles residual penalty-contact penetration robustly.
        floor = self._floor_height(self.x[:, :2])
        self.x[:, 2] = np.maximum(self.x[:, 2], floor + self.radius)
        radial = np.linalg.norm(self.x[:, :2], axis=1)
        outside = (radial > 1.335) & (self.x[:, 2] < 0.24)
        if np.any(outside):
            self.x[outside, :2] *= (1.335 / radial[outside])[:, None]
            outward = self.x[outside, :2] / np.maximum(
                np.linalg.norm(self.x[outside, :2], axis=1, keepdims=True), 1e-8
            )
            outward_speed = np.sum(self.v[outside, :2] * outward, axis=1)
            self.v[outside, :2] -= np.maximum(outward_speed, 0.0)[:, None] * outward

        # Least-squares neighbour velocity gradient advects a material F tensor.
        grad = np.zeros((self.n, 3, 3), dtype=float)
        weight = np.zeros(self.n, dtype=float)
        dx = self.x[self.ej] - self.x[self.ei]
        dv = self.v[self.ej] - self.v[self.ei]
        inv_r2 = 1.0 / np.maximum(np.sum(dx * dx, axis=1), 1e-6)
        outer = dv[:, :, None] * dx[:, None, :] * inv_r2[:, None, None]
        np.add.at(grad, self.ei, outer)
        np.add.at(grad, self.ej, outer)
        np.add.at(weight, self.ei, 1.0)
        np.add.at(weight, self.ej, 1.0)
        grad /= np.maximum(weight[:, None, None], 1.0)
        self.fgrad += dt * np.einsum("nij,njk->nik", grad, self.fgrad)
        self.fgrad = np.clip(np.nan_to_num(self.fgrad, nan=0.0, posinf=3.0, neginf=-3.0), -3.0, 3.0)
        self.time += dt

    def _frame(self, time: float) -> StateFrame:
        gap = self.x[:, 2] - self.radius - self._floor_height(self.x[:, :2])
        frame = StateFrame(
            float(time), self.x.copy(), self.v.copy(), mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": np.full(self.n, self.radius),
                "material_label": self.body_id.copy(),
                "deformation_gradient": self.fgrad.copy(),
            },
            fields={
                "contact_signed_distance": gap.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self._last_external.copy(),
            },
            scalars={
                "contact_length_scale": self.spacing,
                "grid_spacing": self.spacing,
                "characteristic_length": max(float(np.ptp(self.x, axis=0).max()), self.spacing),
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        if not self.initialized:
            raise RuntimeError("BenchmarkSolver.initialize must be called first")
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1e-9:
            raise ValueError("advance_to requires a finite nondecreasing time")
        while self.time < target - 1e-10:
            self._step(min(0.0035, target - self.time))
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "particle_count": int(self.n),
            "integrator": "semi_implicit_kelvin_voigt_particle",
            "simulated_time": float(self.time),
        }
