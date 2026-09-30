"""Deterministic thin-cloth solver for the compressed wrinkle-band scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit mass-spring cloth with moving grips and contact response."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = scene.config
        cloth = cfg["cloth"]
        self.nx, self.ny = map(int, cloth["grid_shape"])
        self.thickness = float(cloth["thickness"])
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        self.time = 0.0
        self.max_step = 1.0 / 600.0
        initial = scene.initial_state()
        self.rest_position = np.asarray(initial.position, dtype=np.float64).copy()
        self.position = self.rest_position.copy()
        self.velocity = np.zeros_like(self.position)
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.inv_mass = 1.0 / self.mass
        self.count = len(self.position)

        # A small deterministic imperfection seeds physical buckling modes.
        rng = np.random.default_rng(int(seed))
        grid = self.position.reshape(self.ny, self.nx, 3)
        u = np.linspace(0.0, 1.0, self.nx)[None, :]
        v = np.linspace(0.0, 1.0, self.ny)[:, None]
        window = np.sin(np.pi * u) * np.sin(np.pi * v)
        coherent = np.sin(7.0 * np.pi * u + 0.22 * np.sin(2.0 * np.pi * v))
        noise = rng.normal(0.0, 1.0, (self.ny, self.nx))
        grid[:, :, 2] += window * (1.2e-3 * coherent + 1.5e-4 * noise)

        self.left = np.arange(0, self.count, self.nx, dtype=np.int64)
        self.right = self.left + self.nx - 1
        self.pinned = np.zeros(self.count, dtype=bool)
        self.pinned[self.left] = True
        self.pinned[self.right] = True
        self.boundary_vertex = self.pinned.astype(np.int32)
        self.grip_start, self.grip_travel = 1.5, 0.30
        self.load_duration = 3.25
        self.grip_half = np.array([0.12, 1.25, 0.12], dtype=np.float64)
        self.ground_height = 0.0

        edge_i, edge_j, rest, stiffness, damping = [], [], [], [], []

        def add_link(a: int, b: int, k: float, c: float) -> None:
            edge_i.append(a); edge_j.append(b)
            rest.append(float(np.linalg.norm(self.rest_position[b] - self.rest_position[a])))
            stiffness.append(k); damping.append(c)

        for iy in range(self.ny):
            for ix in range(self.nx):
                p = iy * self.nx + ix
                if ix + 1 < self.nx: add_link(p, p + 1, 105.0, 0.035)
                if iy + 1 < self.ny: add_link(p, p + self.nx, 105.0, 0.035)
                if ix + 1 < self.nx and iy + 1 < self.ny:
                    add_link(p, p + self.nx + 1, 38.0, 0.018)
                    add_link(p + 1, p + self.nx, 38.0, 0.018)
                if ix + 2 < self.nx: add_link(p, p + 2, 1.8, 0.006)
                if iy + 2 < self.ny: add_link(p, p + 2 * self.nx, 1.8, 0.006)
        self.edge_i = np.asarray(edge_i, dtype=np.int64)
        self.edge_j = np.asarray(edge_j, dtype=np.int64)
        self.rest_length = np.asarray(rest, dtype=np.float64)
        self.stiffness = np.asarray(stiffness, dtype=np.float64)
        self.link_damping = np.asarray(damping, dtype=np.float64)
        self.external_force = self.mass[:, None] * self.gravity[None, :]
        self._set_pins(0.0)

    def _grip_motion(self, time: float) -> tuple[float, float]:
        q = float(np.clip(time / self.load_duration, 0.0, 1.0))
        smooth = q * q * (3.0 - 2.0 * q)
        speed = self.grip_travel * 6.0 * q * (1.0 - q) / self.load_duration if 0.0 < q < 1.0 else 0.0
        return self.grip_travel * smooth, speed

    def _set_pins(self, time: float) -> None:
        displacement, speed = self._grip_motion(time)
        self.position[self.left] = self.rest_position[self.left]
        self.position[self.right] = self.rest_position[self.right]
        self.position[self.left, 0] += displacement
        self.position[self.right, 0] -= displacement
        self.velocity[self.left] = 0.0; self.velocity[self.right] = 0.0
        self.velocity[self.left, 0] = speed; self.velocity[self.right, 0] = -speed

    def _spring_force(self) -> np.ndarray:
        delta = self.position[self.edge_j] - self.position[self.edge_i]
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length, 1.0e-12)[:, None]
        relative_speed = np.sum((self.velocity[self.edge_j] - self.velocity[self.edge_i]) * direction, axis=1)
        magnitude = self.stiffness * (length - self.rest_length) + self.link_damping * relative_speed
        pair_force = magnitude[:, None] * direction
        force = np.zeros_like(self.position)
        np.add.at(force, self.edge_i, pair_force); np.add.at(force, self.edge_j, -pair_force)
        return force

    @staticmethod
    def _box_sdf(points: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
        q = np.abs(points - center[None, :]) - half[None, :]
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)

    def _resolve_box(self, center: np.ndarray, box_velocity: np.ndarray) -> None:
        half = self.grip_half + 0.5 * self.thickness
        local = self.position - center[None, :]
        inside = np.all(np.abs(local) < half[None, :], axis=1) & ~self.pinned
        indices = np.flatnonzero(inside)
        if not len(indices): return
        penetration = half[None, :] - np.abs(local[indices])
        axes = np.argmin(penetration, axis=1)
        signs = np.sign(local[indices, axes]); signs[signs == 0.0] = 1.0
        for particle, axis, sign in zip(indices, axes, signs):
            self.position[particle, axis] = center[axis] + sign * half[axis]
            relative = self.velocity[particle, axis] - box_velocity[axis]
            if relative * sign < 0.0: self.velocity[particle, axis] = box_velocity[axis] - 0.08 * relative
            self.velocity[particle, np.arange(3) != axis] *= 0.985

    def _resolve_contacts(self, time: float) -> None:
        floor = self.ground_height + 0.5 * self.thickness
        hit = (self.position[:, 2] < floor) & ~self.pinned
        self.position[hit, 2] = floor
        downward = hit & (self.velocity[:, 2] < 0.0)
        self.velocity[downward, 2] *= -0.08; self.velocity[hit, :2] *= 0.96
        displacement, speed = self._grip_motion(time)
        self._resolve_box(np.array([-self.grip_start + displacement, 0.0, 2.15]), np.array([speed, 0.0, 0.0]))
        self._resolve_box(np.array([self.grip_start - displacement, 0.0, 2.15]), np.array([-speed, 0.0, 0.0]))

    def _step(self, dt: float) -> None:
        self._set_pins(self.time)
        force = self._spring_force() + self.external_force - 0.0025 * self.velocity
        free = ~self.pinned
        self.velocity[free] += dt * force[free] * self.inv_mass[free, None]
        self.position[free] += dt * self.velocity[free]
        self.time += dt
        self._set_pins(self.time); self._resolve_contacts(self.time); self._set_pins(self.time)

    def _deformation_gradient(self) -> np.ndarray:
        p = self.position.reshape(self.ny, self.nx, 3)
        dx = float(self.rest_position[1, 0] - self.rest_position[0, 0])
        dy = float(self.rest_position[self.nx, 1] - self.rest_position[0, 1])
        tx = np.empty_like(p); ty = np.empty_like(p)
        tx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * dx); tx[:, 0] = (p[:, 1] - p[:, 0]) / dx; tx[:, -1] = (p[:, -1] - p[:, -2]) / dx
        ty[1:-1] = (p[2:] - p[:-2]) / (2.0 * dy); ty[0] = (p[1] - p[0]) / dy; ty[-1] = (p[-1] - p[-2]) / dy
        tx, ty = tx.reshape(-1, 3), ty.reshape(-1, 3)
        metric = np.empty((self.count, 2, 2)); metric[:, 0, 0] = np.einsum("ij,ij->i", tx, tx); metric[:, 0, 1] = np.einsum("ij,ij->i", tx, ty); metric[:, 1, 0] = metric[:, 0, 1]; metric[:, 1, 1] = np.einsum("ij,ij->i", ty, ty)
        vals, vecs = np.linalg.eigh(metric); vals = np.sqrt(np.maximum(vals, 1.0e-12))
        return np.einsum("nij,nj,nkj->nik", vecs, vals, vecs)

    def _contact_distances(self) -> np.ndarray:
        displacement, _ = self._grip_motion(self.time)
        left = np.array([-self.grip_start + displacement, 0.0, 2.15]); right = np.array([self.grip_start - displacement, 0.0, 2.15])
        radius = 0.5 * self.thickness
        return np.concatenate((self.position[:, 2] - self.ground_height - radius, self._box_sdf(self.position, left, self.grip_half) - radius, self._box_sdf(self.position, right, self.grip_half) - radius))

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1.0e-10: raise ValueError("advance_to times must be nondecreasing")
        while self.time + 1.0e-12 < target: self._step(min(self.max_step, target - self.time))
        self.time = target; self._set_pins(self.time)
        frame = StateFrame(target, self.position.copy(), self.velocity.copy(), mass=self.mass.copy(), body_id=np.zeros(self.count, dtype=np.int32), attributes={"mass_quantity": self.mass.copy(), "boundary_vertex": self.boundary_vertex.copy(), "deformation_gradient": self._deformation_gradient()}, fields={"contact_signed_distance": self._contact_distances(), "cumulative_mass_source": np.zeros(1), "external_force": self.external_force.copy()}, scalars={"contact_length_scale": self.thickness})
        frame.validate(); return frame

    def diagnostics(self) -> dict[str, object]:
        return {"method": "substepped mass-spring cloth", "internal_step": self.max_step, "spring_count": int(len(self.edge_i)), "simulated_time": float(self.time)}
