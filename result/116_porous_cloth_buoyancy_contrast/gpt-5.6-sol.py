"""Deterministic porous-cloth / incompressible-liquid solver.

The liquid is advanced on the regular grid represented by the water samples and
the samples themselves are advected as passive Lagrangian markers. Each fabric
is a mass-spring shell. Hydrostatic pressure and finite Darcy drag couple the
two discretizations, with the reaction deposited on the liquid grid.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Self-contained two-way fluid/fabric time stepper."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.config = scene.config
        initial = scene.initial_state()
        self.time = 0.0
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.base_attributes = {
            key: np.asarray(value).copy() for key, value in initial.attributes.items()
        }
        self.n_water = int(np.count_nonzero(self.body_id == 0))
        self.water_position = self.position[:self.n_water].copy()
        self.cloth_position = self.position[self.n_water:].copy()
        self.cloth_velocity = self.velocity[self.n_water:].copy()
        self.cloth_mass = self.mass[self.n_water:].copy()
        self.cloth_body = self.body_id[self.n_water:].copy()

        wp = self.water_position
        self.axes = tuple(np.unique(wp[:, axis]) for axis in range(3))
        self.grid_shape = tuple(len(axis) for axis in self.axes)
        self.spacing = float(np.median(np.diff(self.axes[0])))
        self.grid_origin = np.asarray([axis[0] for axis in self.axes])
        self.fluid_velocity = self.velocity[:self.n_water].reshape(
            self.grid_shape + (3,)
        ).copy()
        self.water_particle_velocity = self.velocity[:self.n_water].copy()
        self.water_cell_mass = float(self.mass[0])

        cloth_cfg = self.config["cloth"]
        self.nx, self.ny = map(int, cloth_cfg["grid_shape"])
        self.nodes_per_cloth = self.nx * self.ny
        self.faces_local = np.asarray(
            scene.metadata()["cloth_topology"]["faces"], dtype=np.int32
        )
        self._make_spring_lattice()
        densities = np.asarray([
            float(cloth_cfg["regimes"][name]["density"])
            for name in ("light", "neutral", "heavy")
        ])
        self.cloth_density = densities[self.cloth_body - 1]
        self.cloth_volume = self.cloth_mass / self.cloth_density
        self.gravity = np.asarray(self.config["gravity"], dtype=np.float64)
        self.rho_water = float(self.config["water"]["density"])
        self.surface_z = float(self.config["water"]["free_surface_z"])
        self.porosity = float(cloth_cfg["porosity"])
        self.permeability = float(cloth_cfg["permeability_m2"])
        self.thickness = float(cloth_cfg["thickness"])

        n = len(self.mass)
        self.domain_weights = np.zeros((4, n), dtype=np.float64)
        self.mass_quantity = np.zeros((n, 4), dtype=np.float64)
        for material in range(4):
            mask = self.body_id == material
            self.domain_weights[material, mask] = 1.0
            self.mass_quantity[mask, material] = self.mass[mask]
        self.initial_momentum = self._domain_momentum()
        self.interaction_impulse = np.zeros((4, 3), dtype=np.float64)
        self.cumulative_external_impulse = np.zeros((4, 3), dtype=np.float64)
        self.last_interaction = self._interaction_forces_only()
        self.last_external = np.zeros((4, 3), dtype=np.float64)
        for material in (1, 2, 3):
            self.last_external[material] = (
                self.cloth_mass[self.cloth_body == material].sum() * self.gravity
            )
        cloth_total_mass = float(self.cloth_mass.sum())
        depth = max(self.surface_z, 0.1)
        self.reference_momentum = max(
            cloth_total_mass * np.sqrt(np.linalg.norm(self.gravity) * depth), 1.0
        )
        self.step_count = 0

    def _make_spring_lattice(self) -> None:
        pair_i, pair_j, stiffness, damping = [], [], [], []

        def add(a: int, b: int, k: float, c: float) -> None:
            pair_i.append(a)
            pair_j.append(b)
            stiffness.append(k)
            damping.append(c)

        for body in range(3):
            off = body * self.nodes_per_cloth
            for iy in range(self.ny):
                for ix in range(self.nx):
                    a = off + iy * self.nx + ix
                    if ix + 1 < self.nx:
                        add(a, a + 1, 34.0, 0.10)
                    if iy + 1 < self.ny:
                        add(a, a + self.nx, 34.0, 0.10)
                    if ix + 1 < self.nx and iy + 1 < self.ny:
                        add(a, a + self.nx + 1, 12.0, 0.055)
                        add(a + 1, a + self.nx, 12.0, 0.055)
                    if ix + 2 < self.nx:
                        add(a, a + 2, 1.8, 0.025)
                    if iy + 2 < self.ny:
                        add(a, a + 2 * self.nx, 1.8, 0.025)
        self.spring_i = np.asarray(pair_i, dtype=np.int32)
        self.spring_j = np.asarray(pair_j, dtype=np.int32)
        self.spring_k = np.asarray(stiffness, dtype=np.float64)
        self.spring_c = np.asarray(damping, dtype=np.float64)
        delta = self.cloth_position[self.spring_j] - self.cloth_position[self.spring_i]
        self.spring_rest = np.linalg.norm(delta, axis=1)

    def _spring_forces(self) -> np.ndarray:
        i, j = self.spring_i, self.spring_j
        delta = self.cloth_position[j] - self.cloth_position[i]
        length = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(length[:, None], 1.0e-12)
        relative_speed = np.sum(
            (self.cloth_velocity[j] - self.cloth_velocity[i]) * direction, axis=1
        )
        magnitude = self.spring_k * (length - self.spring_rest) + self.spring_c * relative_speed
        edge_force = magnitude[:, None] * direction
        force = np.zeros_like(self.cloth_position)
        np.add.at(force, i, edge_force)
        np.add.at(force, j, -edge_force)
        return force

    def _cloth_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        normal_sum = np.zeros_like(self.cloth_position)
        area = np.zeros(len(self.cloth_position), dtype=np.float64)
        for body in range(3):
            faces = self.faces_local + body * self.nodes_per_cloth
            a = self.cloth_position[faces[:, 0]]
            b = self.cloth_position[faces[:, 1]]
            c = self.cloth_position[faces[:, 2]]
            cross = np.cross(b - a, c - a)
            triangle_area = 0.5 * np.linalg.norm(cross, axis=1)
            for corner in range(3):
                np.add.at(normal_sum, faces[:, corner], cross)
                np.add.at(area, faces[:, corner], triangle_area / 3.0)
        norm = np.linalg.norm(normal_sum, axis=1)
        normals = normal_sum / np.maximum(norm[:, None], 1.0e-12)
        return normals, np.maximum(area, 1.0e-7)

    def _grid_coordinates(self, points: np.ndarray):
        q = (points - self.grid_origin) / self.spacing
        upper = np.asarray(self.grid_shape, dtype=np.float64) - 1.000001
        q = np.clip(q, 0.0, upper)
        base = np.floor(q).astype(np.int32)
        base = np.minimum(base, np.asarray(self.grid_shape, dtype=np.int32) - 2)
        return base, q - base

    def _interpolate_grid(self, points: np.ndarray) -> np.ndarray:
        base, frac = self._grid_coordinates(points)
        result = np.zeros((len(points), 3), dtype=np.float64)
        for ox in (0, 1):
            wx = frac[:, 0] if ox else 1.0 - frac[:, 0]
            for oy in (0, 1):
                wy = frac[:, 1] if oy else 1.0 - frac[:, 1]
                for oz in (0, 1):
                    wz = frac[:, 2] if oz else 1.0 - frac[:, 2]
                    idx = base + np.asarray([ox, oy, oz], dtype=np.int32)
                    result += (wx * wy * wz)[:, None] * self.fluid_velocity[
                        idx[:, 0], idx[:, 1], idx[:, 2]
                    ]
        return result

    def _scatter_grid_force(self, points: np.ndarray, forces: np.ndarray) -> np.ndarray:
        base, frac = self._grid_coordinates(points)
        flat = np.zeros((int(np.prod(self.grid_shape)), 3), dtype=np.float64)
        ny, nz = self.grid_shape[1], self.grid_shape[2]
        for ox in (0, 1):
            wx = frac[:, 0] if ox else 1.0 - frac[:, 0]
            for oy in (0, 1):
                wy = frac[:, 1] if oy else 1.0 - frac[:, 1]
                for oz in (0, 1):
                    wz = frac[:, 2] if oz else 1.0 - frac[:, 2]
                    idx = base + np.asarray([ox, oy, oz], dtype=np.int32)
                    linear = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]
                    np.add.at(flat, linear, (wx * wy * wz)[:, None] * forces)
        return flat.reshape(self.grid_shape + (3,))

    def _hydrodynamic_forces(self) -> np.ndarray:
        normals, area = self._cloth_geometry()
        relative = self._interpolate_grid(self.cloth_position) - self.cloth_velocity
        normal_speed = np.sum(relative * normals, axis=1)
        relative_normal = normal_speed[:, None] * normals
        relative_tangent = relative - relative_normal
        transition = max(0.045, 5.0 * self.thickness)
        submerged = np.clip(
            (self.surface_z + 0.5 * transition - self.cloth_position[:, 2]) / transition,
            0.0,
            1.0,
        )
        viscosity = 1.0e-3
        darcy = (
            viscosity * self.thickness * (1.0 - self.porosity)
            / max(self.permeability, 1.0e-16)
        )
        normal_linear = darcy * area
        tangent_linear = 0.055 * normal_linear
        quadratic = 0.20 * self.rho_water * (1.0 - self.porosity) * area
        drag = (
            normal_linear[:, None] * relative_normal
            + tangent_linear[:, None] * relative_tangent
            + (quadratic * np.abs(normal_speed))[:, None] * relative_normal
        ) * submerged[:, None]
        buoyancy = (
            -self.rho_water * self.cloth_volume[:, None] * self.gravity[None, :]
            * submerged[:, None]
        )
        return buoyancy + drag

    def _interaction_forces_only(self) -> np.ndarray:
        hydro = self._hydrodynamic_forces()
        interaction = np.zeros((4, 3), dtype=np.float64)
        for material in (1, 2, 3):
            interaction[material] = hydro[self.cloth_body == material].sum(axis=0)
        interaction[0] = -interaction[1:].sum(axis=0)
        return interaction

    @staticmethod
    def _laplacian(field: np.ndarray, h: float) -> np.ndarray:
        padded = np.pad(field, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="edge")
        neighbors = (
            padded[2:, 1:-1, 1:-1] + padded[:-2, 1:-1, 1:-1]
            + padded[1:-1, 2:, 1:-1] + padded[1:-1, :-2, 1:-1]
            + padded[1:-1, 1:-1, 2:] + padded[1:-1, 1:-1, :-2]
        )
        return (neighbors - 6.0 * field) / (h * h)

    def _project_fluid(self) -> None:
        u, h = self.fluid_velocity, self.spacing
        u[0, :, :, 0] = u[-1, :, :, 0] = 0.0
        u[:, 0, :, 1] = u[:, -1, :, 1] = 0.0
        u[:, :, 0, 2] = 0.0
        div = np.zeros(self.grid_shape, dtype=np.float64)
        div[1:-1] += (u[2:, :, :, 0] - u[:-2, :, :, 0]) / (2.0 * h)
        div[:, 1:-1] += (u[:, 2:, :, 1] - u[:, :-2, :, 1]) / (2.0 * h)
        div[:, :, 1:-1] += (u[:, :, 2:, 2] - u[:, :, :-2, 2]) / (2.0 * h)
        div[:, :, -1] += (u[:, :, -1, 2] - u[:, :, -2, 2]) / h
        potential = np.zeros(self.grid_shape, dtype=np.float64)
        for _ in range(14):
            pad = np.pad(potential, 1, mode="edge")
            neighbor_sum = (
                pad[2:, 1:-1, 1:-1] + pad[:-2, 1:-1, 1:-1]
                + pad[1:-1, 2:, 1:-1] + pad[1:-1, :-2, 1:-1]
                + pad[1:-1, 1:-1, 2:] + pad[1:-1, 1:-1, :-2]
            )
            potential = (neighbor_sum - h * h * div) / 6.0
            # Atmospheric Dirichlet pressure at the configured open surface.
            potential[:, :, -1] = 0.0
        gradient = np.zeros_like(u)
        gradient[1:-1, :, :, 0] = (potential[2:] - potential[:-2]) / (2.0 * h)
        gradient[:, 1:-1, :, 1] = (potential[:, 2:] - potential[:, :-2]) / (2.0 * h)
        gradient[:, :, 1:-1, 2] = (
            potential[:, :, 2:] - potential[:, :, :-2]
        ) / (2.0 * h)
        gradient[:, :, -1, 2] = -potential[:, :, -2] / h
        self.fluid_velocity -= gradient
        self.fluid_velocity[0, :, :, 0] = self.fluid_velocity[-1, :, :, 0] = 0.0
        self.fluid_velocity[:, 0, :, 1] = self.fluid_velocity[:, -1, :, 1] = 0.0
        self.fluid_velocity[:, :, 0, 2] = 0.0

    def _domain_momentum(self) -> np.ndarray:
        momentum = np.zeros((4, 3), dtype=np.float64)
        momentum[0] = np.sum(
            self.mass[:self.n_water, None] * self.water_particle_velocity, axis=0
        )
        for material in (1, 2, 3):
            mask = self.cloth_body == material
            momentum[material] = np.sum(
                self.cloth_mass[mask, None] * self.cloth_velocity[mask], axis=0
            )
        return momentum

    def _substep(self, dt: float) -> None:
        momentum_before = self._domain_momentum()
        hydro = self._hydrodynamic_forces()
        interaction = np.zeros((4, 3), dtype=np.float64)
        for material in (1, 2, 3):
            interaction[material] = hydro[self.cloth_body == material].sum(axis=0)
        interaction[0] = -interaction[1:].sum(axis=0)
        cloth_force = self.cloth_mass[:, None] * self.gravity[None, :]
        cloth_force += self._spring_forces() + hydro
        self.cloth_velocity += dt * cloth_force / self.cloth_mass[:, None]
        self.cloth_position += dt * self.cloth_velocity

        tank_min = np.asarray(self.config["tank"]["interior_bounds_min"], dtype=float)
        tank_max = np.asarray(self.config["tank"]["interior_bounds_max"], dtype=float)
        clearance = 0.5 * self.thickness
        for axis in (0, 1):
            low, high = tank_min[axis] + clearance, tank_max[axis] - clearance
            below = self.cloth_position[:, axis] < low
            above = self.cloth_position[:, axis] > high
            self.cloth_position[below, axis] = low
            self.cloth_position[above, axis] = high
            self.cloth_velocity[below & (self.cloth_velocity[:, axis] < 0), axis] = 0.0
            self.cloth_velocity[above & (self.cloth_velocity[:, axis] > 0), axis] = 0.0
        floor = tank_min[2] + clearance
        hit_floor = self.cloth_position[:, 2] < floor
        self.cloth_position[hit_floor, 2] = floor
        self.cloth_velocity[hit_floor & (self.cloth_velocity[:, 2] < 0), 2] = 0.0
        self.cloth_velocity[hit_floor, :2] *= max(0.0, 1.0 - 5.0 * dt)

        grid_force = self._scatter_grid_force(self.cloth_position, -hydro)
        self.fluid_velocity += dt * grid_force / self.water_cell_mass
        self.fluid_velocity += (
            dt * 0.0025 * self._laplacian(self.fluid_velocity, self.spacing)
        )
        self._project_fluid()
        marker_velocity = self._interpolate_grid(self.water_position)
        self.water_position += dt * marker_velocity
        water_low = np.asarray(self.config["water"]["initial_bounds_min"], dtype=float)
        water_high = np.asarray(self.config["water"]["initial_bounds_max"], dtype=float)
        margin = 0.08 * self.spacing
        self.water_position = np.clip(
            self.water_position, water_low + margin, water_high - margin
        )
        self.water_particle_velocity = self._interpolate_grid(self.water_position)

        momentum_after = self._domain_momentum()
        self.interaction_impulse += dt * interaction
        self.cumulative_external_impulse = (
            momentum_after - self.initial_momentum - self.interaction_impulse
        )
        self.last_interaction = interaction
        self.last_external = (momentum_after - momentum_before) / dt - interaction
        self.time += dt
        self.step_count += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, monotonically increasing time")
        while self.time < target - 1.0e-12:
            self._substep(min(0.005, target - self.time))
        self.time = target
        position = np.vstack((self.water_position, self.cloth_position))
        velocity = np.vstack((self.water_particle_velocity, self.cloth_velocity))
        external_terms = np.zeros((4, 4, 3), dtype=np.float64)
        for material in range(4):
            external_terms[material, material] = self.last_external[material]
        attributes = {
            key: value.copy() for key, value in self.base_attributes.items()
        }
        attributes["mass_quantity"] = self.mass_quantity.copy()
        frame = StateFrame(
            time=target,
            position=position.copy(),
            velocity=velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields={
                "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((4, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((4, 3)),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": external_terms,
                "cumulative_mass_source": np.zeros(4),
                "interaction_force": self.last_interaction.copy(),
            },
            scalars={
                "water_density": self.rho_water,
                "reference_momentum": self.reference_momentum,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "projected-grid liquid with porous mass-spring shells",
            "substeps": int(self.step_count),
            "simulation_time": float(self.time),
            "grid_shape": tuple(int(v) for v in self.grid_shape),
        }
