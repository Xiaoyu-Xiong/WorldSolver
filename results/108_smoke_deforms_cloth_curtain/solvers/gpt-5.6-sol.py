"""Deterministic two-way smoke and cloth solver for the curtain scene."""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Eulerian gas coupled to a lumped-mass spring shell."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.config = scene.config
        initial = scene.initial_state()
        self.time = 0.0
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.rest_position = self.position.copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.base_attributes = {
            key: np.asarray(value).copy() for key, value in initial.attributes.items()
        }
        self.smoke = np.asarray(initial.fields["smoke_density"], dtype=np.float32).copy()
        self.fluid_velocity = np.asarray(
            initial.fields["fluid_velocity"], dtype=np.float32
        ).copy()

        domain = self.config["domain"]
        self.grid_shape = tuple(int(v) for v in domain["grid_shape"])
        self.lower = np.asarray(domain["bounds_min"], dtype=np.float64)
        self.upper = np.asarray(domain["bounds_max"], dtype=np.float64)
        self.spacing = (self.upper - self.lower) / np.asarray(self.grid_shape)
        axes = [
            self.lower[d] + (np.arange(self.grid_shape[d]) + 0.5) * self.spacing[d]
            for d in range(3)
        ]
        self.grid_x = axes[0].astype(np.float32)
        self.grid_y = axes[1].astype(np.float32)
        self.grid_z = axes[2].astype(np.float32)
        self.yz_y, self.yz_z = np.meshgrid(self.grid_y, self.grid_z, indexing="ij")
        jet = self.config["jet"]
        radial2 = ((self.yz_y - float(jet["center"][1])) ** 2 +
                   (self.yz_z - float(jet["center"][2])) ** 2)
        self.jet_cross_section = radial2 <= float(jet["radius"]) ** 2
        self.jet_x_mask = self.grid_x <= float(jet["center"][0]) + 0.18

        self.faces = np.asarray(scene.metadata()["topology"]["faces"], dtype=np.int32)
        self.pinned = np.asarray(self.config["cloth"]["pinned_vertices"], dtype=np.int32)
        self._make_shell_discretization()
        self.gravity = np.asarray(
            self.config["initial_conditions"]["gravity"], dtype=np.float64
        )
        self.gravity_force = self.mass.sum() * self.gravity
        self.last_support_force = np.zeros(3, dtype=np.float64)
        self.last_fluid_force = np.zeros_like(self.position)
        self.cumulative_external_impulse = np.zeros(3, dtype=np.float64)
        self.steps = 0
        self.max_cloth_speed = 0.0
        self.max_fluid_speed = 0.0

    def _make_shell_discretization(self) -> None:
        nx, nz = (int(v) for v in self.config["cloth"]["grid_shape"])
        links: list[tuple[int, int, float, float]] = []

        def add(i: int, j: int, stiffness: float, damping: float) -> None:
            links.append((i, j, stiffness, damping))

        # Stretch, shear, and weak two-edge bending springs.
        for row in range(nz):
            for col in range(nx):
                i = row * nx + col
                if col + 1 < nx:
                    add(i, i + 1, 105.0, 0.42)
                if row + 1 < nz:
                    add(i, i + nx, 105.0, 0.42)
                if col + 1 < nx and row + 1 < nz:
                    add(i, i + nx + 1, 34.0, 0.24)
                    add(i + 1, i + nx, 34.0, 0.24)
                if col + 2 < nx:
                    add(i, i + 2, 7.0, 0.10)
                if row + 2 < nz:
                    add(i, i + 2 * nx, 7.0, 0.10)
        self.links = []
        for i, j, stiffness, damping in links:
            rest = float(np.linalg.norm(self.rest_position[j] - self.rest_position[i]))
            self.links.append((i, j, rest, stiffness, damping))
        self.vertex_area = np.zeros(len(self.position), dtype=np.float64)
        for tri in self.faces:
            a, b, c = self.rest_position[tri]
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            self.vertex_area[tri] += area / 3.0

    @staticmethod
    def _shifted(q: np.ndarray, axis: int, forward: bool) -> np.ndarray:
        out = np.empty_like(q)
        dst = [slice(None)] * q.ndim
        src = [slice(None)] * q.ndim
        if forward:
            dst[axis], src[axis] = slice(None, -1), slice(1, None)
            out[tuple(dst)] = q[tuple(src)]
            dst[axis], src[axis] = slice(-1, None), slice(-1, None)
        else:
            dst[axis], src[axis] = slice(1, None), slice(None, -1)
            out[tuple(dst)] = q[tuple(src)]
            dst[axis], src[axis] = slice(None, 1), slice(None, 1)
        out[tuple(dst)] = q[tuple(src)]
        return out

    def _advect(self, q: np.ndarray, advector: np.ndarray, dt: float) -> np.ndarray:
        """Dimensionally split donor-cell advection with open boundaries."""
        result = q
        for axis in range(3):
            c = np.clip(advector[..., axis] * (dt / self.spacing[axis]), -0.92, 0.92)
            qm = self._shifted(result, axis, False)
            qp = self._shifted(result, axis, True)
            result = (result - np.maximum(c, 0.0) * (result - qm)
                      - np.minimum(c, 0.0) * (qp - result))
        return result

    def _inject_jet(self, t: float) -> None:
        jet = self.config["jet"]
        if t < float(jet["start_time"]) or t > float(jet["active_until"]):
            return
        ramp = min(1.0, max(0.0, (t - float(jet["start_time"])) /
                            max(float(jet["ramp_duration"]), 1.0e-9)))
        source = self.jet_x_mask[:, None, None] & self.jet_cross_section[None, :, :]
        target = np.asarray(jet["axis"], dtype=np.float32) * (
            float(jet["speed"]) * ramp
        )
        self.fluid_velocity[source] = target
        self.smoke[source] = np.float32(float(jet["smoke_density"]) * ramp)

    def _curtain_surface_on_yz(self):
        """Rasterize the moving triangular shell as a thin grid boundary."""
        yy, zz = self.yz_y, self.yz_z
        covered = np.zeros(yy.shape, dtype=bool)
        surface_x = np.zeros(yy.shape, dtype=np.float32)
        surface_v = np.zeros(yy.shape + (3,), dtype=np.float32)
        for tri in self.faces:
            p = self.position[tri]
            y0, z0 = p[0, 1], p[0, 2]
            y1, z1 = p[1, 1], p[1, 2]
            y2, z2 = p[2, 1], p[2, 2]
            denom = (z1 - z2) * (y0 - y2) + (y2 - y1) * (z0 - z2)
            if abs(denom) < 1.0e-10:
                continue
            w0 = ((z1 - z2) * (yy - y2) + (y2 - y1) * (zz - z2)) / denom
            w1 = ((z2 - z0) * (yy - y2) + (y0 - y2) * (zz - z2)) / denom
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1.0e-5) & (w1 >= -1.0e-5) & (w2 >= -1.0e-5)
            surface_x[inside] = (w0[inside] * p[0, 0] + w1[inside] * p[1, 0]
                                 + w2[inside] * p[2, 0])
            for d in range(3):
                surface_v[..., d][inside] = (
                    w0[inside] * self.velocity[tri[0], d]
                    + w1[inside] * self.velocity[tri[1], d]
                    + w2[inside] * self.velocity[tri[2], d]
                )
            covered |= inside
        return covered, surface_x, surface_v

    def _apply_curtain_boundary(self, dt: float) -> None:
        covered, surface_x, surface_v = self._curtain_surface_on_yz()
        if not np.any(covered):
            return
        distance = self.grid_x[:, None, None] - surface_x[None, :, :]
        influence = np.exp(-0.5 * (distance / 0.105) ** 2).astype(np.float32)
        influence *= covered[None, :, :]
        penalty = 1.0 - np.exp(-24.0 * dt * influence)
        relative_x = self.fluid_velocity[..., 0] - surface_v[None, ..., 0]
        removed = penalty * relative_x
        self.fluid_velocity[..., 0] -= removed

        # Pressure from stopped normal flow turns the jet around free edges.
        center_y = 0.5 * (float(self.position[:, 1].min()) +
                          float(self.position[:, 1].max()))
        center_z = 0.5 * (float(self.position[:, 2].min()) +
                          float(self.position[:, 2].max()))
        out_y = (self.yz_y - center_y) / 1.15
        out_z = (self.yz_z - center_z) / 1.30
        length = np.sqrt(out_y * out_y + out_z * out_z + 0.035)
        out_y, out_z = out_y / length, out_z / length
        self.fluid_velocity[..., 1] += 0.58 * removed * out_y[None, :, :]
        self.fluid_velocity[..., 2] += 0.58 * removed * out_z[None, :, :]
        tangential = 0.10 * penalty
        self.fluid_velocity[..., 1] += tangential * (
            surface_v[None, ..., 1] - self.fluid_velocity[..., 1])
        self.fluid_velocity[..., 2] += tangential * (
            surface_v[None, ..., 2] - self.fluid_velocity[..., 2])

    def _project_velocity(self, dt: float) -> None:
        """Three Jacobi sweeps of a constant-density pressure projection."""
        u = self.fluid_velocity
        h = float(self.spacing[0])
        divergence = np.zeros(self.grid_shape, dtype=np.float32)
        divergence[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1, 0] - u[:-2, 1:-1, 1:-1, 0])
            + (u[1:-1, 2:, 1:-1, 1] - u[1:-1, :-2, 1:-1, 1])
            + (u[1:-1, 1:-1, 2:, 2] - u[1:-1, 1:-1, :-2, 2])
        ) / (2.0 * h)
        rhs_scale = (h * h) / max(dt, 1.0e-8)
        pressure = np.zeros(self.grid_shape, dtype=np.float32)
        for _ in range(3):
            nxt = np.zeros_like(pressure)
            nxt[1:-1, 1:-1, 1:-1] = (
                pressure[2:, 1:-1, 1:-1] + pressure[:-2, 1:-1, 1:-1]
                + pressure[1:-1, 2:, 1:-1] + pressure[1:-1, :-2, 1:-1]
                + pressure[1:-1, 1:-1, 2:] + pressure[1:-1, 1:-1, :-2]
                - rhs_scale * divergence[1:-1, 1:-1, 1:-1]
            ) / 6.0
            pressure = nxt
        scale = dt / (2.0 * h)
        u[1:-1, 1:-1, 1:-1, 0] -= scale * (
            pressure[2:, 1:-1, 1:-1] - pressure[:-2, 1:-1, 1:-1])
        u[1:-1, 1:-1, 1:-1, 1] -= scale * (
            pressure[1:-1, 2:, 1:-1] - pressure[1:-1, :-2, 1:-1])
        u[1:-1, 1:-1, 1:-1, 2] -= scale * (
            pressure[1:-1, 1:-1, 2:] - pressure[1:-1, 1:-1, :-2])

    def _step_fluid(self, dt: float, t_end: float) -> None:
        advector = self.fluid_velocity.copy()
        self.smoke = self._advect(self.smoke, advector, dt).astype(np.float32)
        for component in range(3):
            self.fluid_velocity[..., component] = self._advect(
                self.fluid_velocity[..., component], advector, dt
            ).astype(np.float32)
        self.fluid_velocity *= np.float32(math.exp(-0.018 * dt))
        self._inject_jet(t_end)
        self._apply_curtain_boundary(dt)
        self._project_velocity(dt)
        self._inject_jet(t_end)
        np.clip(self.smoke, 0.0, 1.0, out=self.smoke)
        np.clip(self.fluid_velocity, -6.0, 6.0, out=self.fluid_velocity)

    def _sample_grid_velocity(self, points: np.ndarray) -> np.ndarray:
        coordinates = (points - self.lower[None, :]) / self.spacing[None, :] - 0.5
        base = np.floor(coordinates).astype(np.int64)
        fraction = coordinates - base
        last_base = np.asarray(self.grid_shape, dtype=np.int64) - 2
        base = np.minimum(np.maximum(base, 0), last_base)
        fraction = np.minimum(np.maximum(fraction, 0.0), 1.0)
        value = np.zeros((len(points), 3), dtype=np.float64)
        for dx in (0, 1):
            wx = (1.0 - fraction[:, 0]) if dx == 0 else fraction[:, 0]
            for dy in (0, 1):
                wy = (1.0 - fraction[:, 1]) if dy == 0 else fraction[:, 1]
                for dz in (0, 1):
                    wz = (1.0 - fraction[:, 2]) if dz == 0 else fraction[:, 2]
                    value += (wx * wy * wz)[:, None] * self.fluid_velocity[
                        base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz
                    ]
        return value

    def _vertex_normals(self) -> np.ndarray:
        normals = np.zeros_like(self.position)
        for tri in self.faces:
            a, b, c = self.position[tri]
            normals[tri] += np.cross(b - a, c - a)
        lengths = np.linalg.norm(normals, axis=1)
        good = lengths > 1.0e-12
        normals[good] /= lengths[good, None]
        normals[~good] = np.array([1.0, 0.0, 0.0])
        return normals

    def _aerodynamic_forces(self) -> np.ndarray:
        gas_velocity = self._sample_grid_velocity(self.position)
        normals = self._vertex_normals()
        relative = gas_velocity - self.velocity
        normal_speed = np.einsum("ij,ij->i", relative, normals)
        rho_air = 1.18
        pressure = (0.5 * rho_air * self.vertex_area * normal_speed
                    * np.abs(normal_speed))[:, None] * normals
        tangent = relative - normal_speed[:, None] * normals
        tangent_speed = np.linalg.norm(tangent, axis=1)
        viscous = (0.055 * rho_air * self.vertex_area * tangent_speed)[:, None] * tangent
        force = pressure + viscous
        magnitude = np.linalg.norm(force, axis=1)
        excessive = magnitude > 2.5
        force[excessive] *= (2.5 / magnitude[excessive])[:, None]
        return force

    def _shell_forces(self, aerodynamic: np.ndarray) -> np.ndarray:
        force = self.mass[:, None] * self.gravity[None, :] + aerodynamic
        for i, j, rest, stiffness, damping in self.links:
            delta = self.position[j] - self.position[i]
            length = float(np.linalg.norm(delta))
            if length < 1.0e-10:
                continue
            direction = delta / length
            relative_speed = float(np.dot(self.velocity[j] - self.velocity[i], direction))
            pair_force = (stiffness * (length - rest) + damping * relative_speed) * direction
            force[i] += pair_force
            force[j] -= pair_force
        force -= 0.025 * self.velocity
        return force

    def _step_cloth(self, dt: float) -> np.ndarray:
        substeps = max(1, int(math.ceil(dt / 0.003)))
        h = dt / substeps
        support_impulse = np.zeros(3, dtype=np.float64)
        aerodynamic = np.zeros_like(self.position)
        for _ in range(substeps):
            aerodynamic = self._aerodynamic_forces()
            force = self._shell_forces(aerodynamic)
            trial_velocity = self.velocity + h * force / self.mass[:, None]
            support_impulse -= np.sum(
                self.mass[self.pinned, None] * trial_velocity[self.pinned], axis=0
            )
            self.velocity = trial_velocity
            self.velocity[self.pinned] = 0.0
            speed = np.linalg.norm(self.velocity, axis=1)
            fast = speed > 12.0
            self.velocity[fast] *= (12.0 / speed[fast])[:, None]
            self.position += h * self.velocity
            self.position[self.pinned] = self.rest_position[self.pinned]
            contact = self.position[:, 2] < 0.012
            if np.any(contact):
                self.position[contact, 2] = 0.012
                downward = contact & (self.velocity[:, 2] < 0.0)
                self.velocity[downward, 2] *= -0.08
                self.velocity[contact, :2] *= 0.82
        self.last_fluid_force = aerodynamic
        return support_impulse / max(dt, 1.0e-12)

    def _integrate(self, target_time: float) -> None:
        if target_time < self.time - 1.0e-10:
            raise ValueError("advance_to requires nondecreasing sample times")
        while self.time + 1.0e-10 < target_time:
            dt = min(0.02, target_time - self.time)
            end_time = self.time + dt
            self._step_fluid(dt, end_time)
            support = self._step_cloth(dt)
            self.last_support_force = support
            self.cumulative_external_impulse += (self.gravity_force + support) * dt
            self.time = end_time
            self.steps += 1
            self.max_cloth_speed = max(
                self.max_cloth_speed, float(np.max(np.linalg.norm(self.velocity, axis=1)))
            )
            self.max_fluid_speed = max(
                self.max_fluid_speed,
                float(np.max(np.linalg.norm(self.fluid_velocity, axis=-1))),
            )

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("sample time must be finite")
        self._integrate(target)
        normals = self._vertex_normals()
        external_terms = np.zeros((1, 2, 3), dtype=np.float64)
        external_terms[0, 0] = self.gravity_force
        external_terms[0, 1] = self.last_support_force
        attributes = {key: value.copy() for key, value in self.base_attributes.items()}
        attributes["mass_quantity"] = self.mass.copy()
        fields = {
            "smoke_density": self.smoke.copy(),
            "fluid_velocity": self.fluid_velocity.copy(),
            "boundary_normal": normals,
            "boundary_target_velocity": self.velocity.copy(),
            "boundary_velocity": self.velocity.copy(),
            "budget_cumulative_external_impulse": self.cumulative_external_impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_domain_weights": np.ones((1, len(self.mass)), dtype=np.float64),
            "budget_external_force": external_terms,
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "interaction_force": self.last_fluid_force.copy(),
        }
        total_mass = float(self.mass.sum())
        jet_speed = float(self.config["jet"]["speed"])
        scalars = {
            "jet_start_time": float(self.config["jet"]["start_time"]),
            "jet_speed": jet_speed,
            "reference_momentum": max(total_mass * jet_speed, 1.0e-6),
            "reference_velocity": max(jet_speed, 1.0e-6),
        }
        return StateFrame(
            time=target,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "projected Eulerian gas with immersed-boundary spring shell",
            "steps": int(self.steps),
            "time": float(self.time),
            "max_cloth_speed": float(self.max_cloth_speed),
            "max_fluid_speed": float(self.max_fluid_speed),
            "finite": bool(
                np.all(np.isfinite(self.position))
                and np.all(np.isfinite(self.velocity))
                and np.all(np.isfinite(self.smoke))
                and np.all(np.isfinite(self.fluid_velocity))
            ),
        }
