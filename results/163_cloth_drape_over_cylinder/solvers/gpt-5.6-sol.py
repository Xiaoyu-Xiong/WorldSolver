"""Deterministic cloth solver for the cylinder-drape scene.

The cloth is a lumped-mass triangular sheet. Stretch, shear, and bending are
represented by springs and advanced with small semi-implicit Euler steps.
Rigid contact is handled after each step by projection and a friction impulse.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physically step a rectangular cloth over the frozen rigid geometry."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        cfg = scene.config
        initial = scene.initial_state()

        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.inv_mass = 1.0 / self.mass
        self.body_id = np.zeros(len(self.mass), dtype=np.int32)
        self.time = 0.0

        self.nx, self.ny = map(int, cfg["cloth"]["grid_shape"])
        self.sx, self.sy = map(float, cfg["cloth"]["size"])
        self.thickness = float(cfg["cloth"]["thickness"])
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        self.external_force = self.mass[:, None] * self.gravity[None, :]

        # The frozen cylinder's local z axis is rotated onto world y.
        self.cylinder_center = np.array([0.0, 0.0, 0.7], dtype=np.float64)
        self.cylinder_radius = 0.52
        self.cylinder_half_length = 1.4
        self.contact_radius = self.cylinder_radius + 0.5 * self.thickness
        self.ground_height = 0.5 * self.thickness

        self.max_step = 1.0 / 600.0
        self.linear_drag = 0.18
        self.friction = 0.24
        self.restitution = 0.015
        self.steps = 0
        self.contact_count = 0
        self._build_springs()

    def _build_springs(self) -> None:
        """Build springs following the regular triangulated sheet topology."""
        nx, ny = self.nx, self.ny
        edge_i: list[int] = []
        edge_j: list[int] = []
        stiffness: list[float] = []
        damping: list[float] = []

        def add(a: int, b: int, k: float, c: float) -> None:
            edge_i.append(a)
            edge_j.append(b)
            stiffness.append(k)
            damping.append(c)

        for j in range(ny):
            for i in range(nx - 1):
                add(j * nx + i, j * nx + i + 1, 52.0, 0.030)
        for j in range(ny - 1):
            for i in range(nx):
                add(j * nx + i, (j + 1) * nx + i, 52.0, 0.030)

        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                add(a, a + nx + 1, 24.0, 0.020)
                add(a + 1, a + nx, 24.0, 0.020)

        for j in range(ny):
            for i in range(nx - 2):
                add(j * nx + i, j * nx + i + 2, 1.4, 0.008)
        for j in range(ny - 2):
            for i in range(nx):
                add(j * nx + i, (j + 2) * nx + i, 1.4, 0.008)

        self.edge_i = np.asarray(edge_i, dtype=np.int32)
        self.edge_j = np.asarray(edge_j, dtype=np.int32)
        self.spring_k = np.asarray(stiffness, dtype=np.float64)
        self.spring_c = np.asarray(damping, dtype=np.float64)
        delta = self.position[self.edge_j] - self.position[self.edge_i]
        self.rest_length = np.linalg.norm(delta, axis=1)

    def _spring_forces(self) -> np.ndarray:
        delta = self.position[self.edge_j] - self.position[self.edge_i]
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length, 1.0e-12)[:, None]
        relative_speed = np.sum(
            (self.velocity[self.edge_j] - self.velocity[self.edge_i]) * direction,
            axis=1,
        )
        magnitude = self.spring_k * (length - self.rest_length)
        magnitude += self.spring_c * relative_speed
        pair_force = magnitude[:, None] * direction

        force = self.external_force.copy()
        np.add.at(force, self.edge_i, pair_force)
        np.add.at(force, self.edge_j, -pair_force)
        return force

    def _resolve_contacts(self) -> None:
        """Project penetrations against the finite cylinder and ground."""
        p = self.position
        v = self.velocity
        rel_x = p[:, 0] - self.cylinder_center[0]
        rel_z = p[:, 2] - self.cylinder_center[2]
        radial = np.sqrt(rel_x * rel_x + rel_z * rel_z)
        inside_span = np.abs(p[:, 1] - self.cylinder_center[1]) <= self.cylinder_half_length
        hit = inside_span & (radial < self.contact_radius)
        indices = np.flatnonzero(hit)
        if indices.size:
            rr = np.maximum(radial[indices], 1.0e-12)
            normal = np.column_stack((rel_x[indices] / rr,
                                      np.zeros(indices.size),
                                      rel_z[indices] / rr))
            penetration = self.contact_radius - radial[indices]
            p[indices] += penetration[:, None] * normal

            vn = np.sum(v[indices] * normal, axis=1)
            approaching = vn < 0.0
            if np.any(approaching):
                ids = indices[approaching]
                normals = normal[approaching]
                old_vn = vn[approaching]
                v[ids] -= ((1.0 + self.restitution) * old_vn)[:, None] * normals
                tangent = v[ids] - np.sum(v[ids] * normals, axis=1)[:, None] * normals
                tangent_speed = np.linalg.norm(tangent, axis=1)
                impulse_limit = self.friction * (1.0 + self.restitution) * (-old_vn)
                scale = np.maximum(0.0, 1.0 - impulse_limit / np.maximum(tangent_speed, 1.0e-12))
                v[ids] -= (1.0 - scale)[:, None] * tangent
            self.contact_count += int(indices.size)

        ground_hit = p[:, 2] < self.ground_height
        indices = np.flatnonzero(ground_hit)
        if indices.size:
            p[indices, 2] = self.ground_height
            downward = v[indices, 2] < 0.0
            ids = indices[downward]
            if ids.size:
                impact = -v[ids, 2].copy()
                v[ids, 2] *= -self.restitution
                planar_speed = np.linalg.norm(v[ids, :2], axis=1)
                reduction = self.friction * (1.0 + self.restitution) * impact
                scale = np.maximum(0.0, 1.0 - reduction / np.maximum(planar_speed, 1.0e-12))
                v[ids, :2] *= scale[:, None]
            self.contact_count += int(indices.size)

    def _step(self, dt: float) -> None:
        force = self._spring_forces()
        self.velocity += dt * force * self.inv_mass[:, None]
        self.velocity *= np.exp(-self.linear_drag * dt)
        self.position += dt * self.velocity
        self._resolve_contacts()
        self.steps += 1

    def _contact_signed_distance(self) -> np.ndarray:
        q_radial = np.sqrt(
            (self.position[:, 0] - self.cylinder_center[0]) ** 2
            + (self.position[:, 2] - self.cylinder_center[2]) ** 2
        ) - self.cylinder_radius
        q_axial = np.abs(self.position[:, 1] - self.cylinder_center[1]) - self.cylinder_half_length
        outside = np.sqrt(np.maximum(q_radial, 0.0) ** 2 + np.maximum(q_axial, 0.0) ** 2)
        cylinder_sdf = outside + np.minimum(np.maximum(q_radial, q_axial), 0.0)
        return cylinder_sdf - 0.5 * self.thickness

    def _deformation_gradient(self) -> np.ndarray:
        """Return the in-material-plane stretch tensor at every vertex."""
        grid = self.position.reshape(self.ny, self.nx, 3)
        d_x = np.empty_like(grid)
        d_y = np.empty_like(grid)
        hx = self.sx / (self.nx - 1)
        hy = self.sy / (self.ny - 1)
        d_x[:, 1:-1] = (grid[:, 2:] - grid[:, :-2]) / (2.0 * hx)
        d_x[:, 0] = (grid[:, 1] - grid[:, 0]) / hx
        d_x[:, -1] = (grid[:, -1] - grid[:, -2]) / hx
        d_y[1:-1] = (grid[2:] - grid[:-2]) / (2.0 * hy)
        d_y[0] = (grid[1] - grid[0]) / hy
        d_y[-1] = (grid[-1] - grid[-2]) / hy

        c00 = np.sum(d_x * d_x, axis=2).ravel()
        c01 = np.sum(d_x * d_y, axis=2).ravel()
        c11 = np.sum(d_y * d_y, axis=2).ravel()
        root_det = np.sqrt(np.maximum(c00 * c11 - c01 * c01, 0.0))
        denom = np.sqrt(np.maximum(c00 + c11 + 2.0 * root_det, 1.0e-18))
        deformation = np.empty((len(self.mass), 2, 2), dtype=np.float64)
        deformation[:, 0, 0] = (c00 + root_det) / denom
        deformation[:, 0, 1] = c01 / denom
        deformation[:, 1, 0] = c01 / denom
        deformation[:, 1, 1] = (c11 + root_det) / denom
        return deformation

    def _frame(self, time: float) -> StateFrame:
        boundary = np.zeros((self.ny, self.nx), dtype=np.int32)
        boundary[[0, -1], :] = 1
        boundary[:, [0, -1]] = 1
        frame = StateFrame(
            float(time), self.position.copy(), self.velocity.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": boundary.ravel(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self._contact_signed_distance(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": self.external_force.copy(),
            },
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-12:
            raise ValueError("advance_to requires finite, nondecreasing times")
        while self.time < target - 1.0e-12:
            dt = min(self.max_step, target - self.time)
            self._step(dt)
            self.time += dt
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "semi_implicit_euler_mass_spring",
            "internal_steps": self.steps,
            "resolved_contacts": self.contact_count,
            "particle_count": len(self.mass),
            "spring_count": len(self.edge_i),
        }
