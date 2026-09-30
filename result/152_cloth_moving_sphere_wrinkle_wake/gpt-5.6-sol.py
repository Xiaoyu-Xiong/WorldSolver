"""Deterministic mass-spring cloth solver for the moving-sphere scene."""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.nx, self.ny = (int(v) for v in cloth["grid_shape"])
        self.n = self.nx * self.ny
        self.sx, self.sy = (float(v) for v in cloth["size"])
        self.center = np.asarray(cloth["center"], dtype=float)
        self.thickness = float(cloth.get("thickness", 0.015))
        self.radius = 0.62
        self.mass_value = float(cloth["total_mass"]) / self.n
        self.mass = np.full(self.n, self.mass_value, dtype=float)
        self.inv_mass = 1.0 / self.mass

        gx = np.linspace(-self.sx * 0.5, self.sx * 0.5, self.nx)
        gy = np.linspace(-self.sy * 0.5, self.sy * 0.5, self.ny)
        xx, yy = np.meshgrid(gx, gy, indexing="xy")
        self.rest_xy = np.column_stack((xx.ravel(), yy.ravel()))
        self.rest = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(self.n))) + self.center
        # A shallow initial sag gives the supported sheet a gently draped state.
        sag = 0.045 * np.maximum(
            0.0,
            1.0 - 0.55 * (xx.ravel() / (self.sx * 0.5)) ** 2
            - 0.45 * (yy.ravel() / (self.sy * 0.5)) ** 2,
        )
        self.position = self.rest.copy()
        self.position[:, 2] -= sag
        self.velocity = np.zeros((self.n, 3), dtype=float)
        self.boundary = (
            (np.isclose(xx, gx[0])) | (np.isclose(xx, gx[-1]))
            | (np.isclose(yy, gy[0])) | (np.isclose(yy, gy[-1]))
        ).ravel().astype(np.int32)

        # Structural, shear, and two-cell bending springs.
        edges_i: list[int] = []
        edges_j: list[int] = []
        edge_k: list[float] = []
        edge_rest: list[float] = []

        def add_edge(i: int, j: int, stiffness: float) -> None:
            edges_i.append(i)
            edges_j.append(j)
            edge_k.append(stiffness)
            edge_rest.append(float(np.linalg.norm(self.rest[j] - self.rest[i])))

        for iy in range(self.ny):
            for ix in range(self.nx):
                i = iy * self.nx + ix
                if ix + 1 < self.nx:
                    add_edge(i, i + 1, 38.0)
                if iy + 1 < self.ny:
                    add_edge(i, i + self.nx, 38.0)
                if ix + 1 < self.nx and iy + 1 < self.ny:
                    add_edge(i, i + self.nx + 1, 24.0)
                if ix > 0 and iy + 1 < self.ny:
                    add_edge(i, i + self.nx - 1, 24.0)
                if ix + 2 < self.nx:
                    add_edge(i, i + 2, 7.0)
                if iy + 2 < self.ny:
                    add_edge(i, i + 2 * self.nx, 7.0)
        self.edge_i = np.asarray(edges_i, dtype=np.int32)
        self.edge_j = np.asarray(edges_j, dtype=np.int32)
        self.edge_k = np.asarray(edge_k, dtype=float)
        self.edge_rest = np.asarray(edge_rest, dtype=float)
        self.edge_damping = 0.035
        self.anchor_k = 26.0
        self.anchor_damping = 0.35
        self.dt_internal = 0.0015
        self.time = 0.0
        self.last_contact = np.zeros(self.n, dtype=float)
        self._resolve_contacts(0.0)

    def _sphere_state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Prescribed smooth left-to-right sphere traverse."""
        start, finish = 0.20, 6.35
        q = np.clip((float(t) - start) / (finish - start), 0.0, 1.0)
        s = q * q * (3.0 - 2.0 * q)
        ds = 6.0 * q * (1.0 - q) / (finish - start) if 0.0 < q < 1.0 else 0.0
        x = -1.12 + 2.24 * s
        vx = 2.24 * ds
        y = 0.06 * np.sin(0.75 * float(t))
        vy = 0.045 * np.cos(0.75 * float(t))
        return np.array([x, y, 0.68]), np.array([vx, vy, 0.0])

    def _forces(self, t: float) -> np.ndarray:
        force = self.mass[:, None] * self.gravity[None, :]
        delta = self.position[self.edge_j] - self.position[self.edge_i]
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length, 1.0e-8)[:, None]
        relative = self.velocity[self.edge_j] - self.velocity[self.edge_i]
        extension_rate = np.sum(relative * direction, axis=1)
        pair = (
            self.edge_k * (length - self.edge_rest)
            + self.edge_damping * extension_rate
        )[:, None] * direction
        np.add.at(force, self.edge_i, pair)
        np.add.at(force, self.edge_j, -pair)

        edge = self.boundary.astype(bool)
        force[edge] += self.anchor_k * (self.rest[edge] - self.position[edge])
        force[edge] -= self.anchor_damping * self.velocity[edge]

        sphere, sphere_velocity = self._sphere_state(t)
        delta = self.position - sphere[None, :]
        length = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(length, 1.0e-8)[:, None]
        normal[length < 1.0e-8] = np.array([0.0, 0.0, 1.0])
        gap = length - (self.radius + 0.5 * self.thickness)
        approaching = np.sum(
            (self.velocity - sphere_velocity[None, :]) * normal, axis=1
        )
        magnitude = 520.0 * np.maximum(-gap, 0.0) - 0.45 * np.minimum(approaching, 0.0)
        force += magnitude[:, None] * normal

        ground_penetration = np.maximum(0.025 - self.position[:, 2], 0.0)
        force[:, 2] += (
            700.0 * ground_penetration
            - 6.0 * np.minimum(self.velocity[:, 2], 0.0)
        )
        return force

    def _resolve_contacts(self, t: float) -> None:
        sphere, sphere_velocity = self._sphere_state(t)
        delta = self.position - sphere[None, :]
        length = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(length, 1.0e-8)[:, None]
        normal[length < 1.0e-8] = np.array([0.0, 0.0, 1.0])
        clearance = self.radius + 0.5 * self.thickness + 1.0e-5
        hit = length < clearance
        if np.any(hit):
            self.position[hit] = sphere[None, :] + normal[hit] * clearance
            relative = self.velocity[hit] - sphere_velocity[None, :]
            normal_speed = np.sum(relative * normal[hit], axis=1)
            relative -= np.minimum(normal_speed, 0.0)[:, None] * normal[hit]
            relative *= 0.84
            self.velocity[hit] = sphere_velocity[None, :] + relative

        self.position[:, 2] = np.maximum(self.position[:, 2], 0.025)
        ground = self.position[:, 2] <= 0.025 + 1.0e-7
        self.velocity[ground & (self.velocity[:, 2] < 0.0), 2] = 0.0
        self.last_contact = np.linalg.norm(self.position - sphere[None, :], axis=1) - self.radius

    def _step(self, dt: float, t: float) -> None:
        force = self._forces(t)
        self.velocity += dt * force * self.inv_mass[:, None]
        self.velocity *= np.exp(-0.32 * dt)
        self.position += dt * self.velocity
        self._resolve_contacts(t + dt)

    def _deformation_gradient(self) -> np.ndarray:
        """Vertex-centered in-plane surface deformation gradients."""
        p = self.position
        result = np.zeros((self.n, 2, 2), dtype=float)
        for iy in range(self.ny):
            for ix in range(self.nx):
                i = iy * self.nx + ix
                il = iy * self.nx + max(ix - 1, 0)
                ir = iy * self.nx + min(ix + 1, self.nx - 1)
                ib = max(iy - 1, 0) * self.nx + ix
                it = min(iy + 1, self.ny - 1) * self.nx + ix
                dx = (p[ir] - p[il]) / max(
                    self.rest_xy[ir, 0] - self.rest_xy[il, 0], 1.0e-8
                )
                dy = (p[it] - p[ib]) / max(
                    self.rest_xy[it, 1] - self.rest_xy[ib, 1], 1.0e-8
                )
                result[i] = np.array([[dx[0], dx[1]], [dy[0], dy[1]]])
        return result

    def advance_to(self, time: float):
        target = max(float(time), self.time)
        while self.time < target - 1.0e-12:
            dt = min(self.dt_internal, target - self.time)
            self._step(dt, self.time)
            self.time += dt
        self._resolve_contacts(target)
        frame = StateFrame(
            time=target,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "external_force": (
                    self.mass[:, None] * self.gravity[None, :]
                ).copy(),
                "contact_signed_distance": self.last_contact.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "semi_implicit_mass_spring",
            "cloth_vertices": int(self.n),
            "internal_timestep": float(self.dt_internal),
        }
