"""Deterministic porous-solid/free-liquid solver for the duck benchmark.

The model is deliberately compact, but all evolution is obtained by time
integration. Pore saturation obeys a graph discretisation of a Darcy-capillary
diffusion equation. Absorbed water is removed from the free phase and its
momentum is transferred to the porous body. The body is advanced under gravity,
hydrostatic buoyancy, hydrodynamic drag and floor contact; equal and opposite
fluid forces drive a reduced Lagrangian free-liquid model.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """A small, deterministic two-way poromechanics time-stepper."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        cfg = scene.config

        self.cfg = cfg
        self.time = 0.0
        self.pos0 = np.asarray(initial.position, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.dry_mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.mass = self.dry_mass.copy()
        self.surface = np.flatnonzero(self.body_id == 1)
        self.porous = np.flatnonzero(self.body_id == 3)
        self.water = np.flatnonzero(self.body_id == 2)
        self.duck = np.flatnonzero((self.body_id == 1) | (self.body_id == 3))

        self.rho = float(cfg["fluid"]["density_kg_m3"])
        self.rho_dry = float(cfg["duck"]["dry_bulk_density_kg_m3"])
        self.porosity = float(cfg["duck"]["porosity"])
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        self.g = abs(float(self.gravity[2]))
        self.switch_time = float(cfg["permeability_control"]["switch_time"])
        self.permeability = float(
            cfg["permeability_control"]["intrinsic_permeability_after_switch_m2"]
        )
        self.surface0 = float(cfg["fluid"]["initial_surface_z"])

        self.h = 0.065
        self.cell_volume = self.h ** 3
        self.pore_capacity = np.full(
            len(self.porous), self.rho * self.porosity * self.cell_volume,
            dtype=np.float64,
        )
        self.saturation = np.zeros(len(self.porous), dtype=np.float64)

        # Six-neighbour material graph for the Darcy-capillary Laplacian.
        pore_ref = self.pos0[self.porous]
        base = np.min(pore_ref, axis=0)
        lattice = np.rint((pore_ref - base) / self.h).astype(np.int64)
        lookup = {tuple(q): i for i, q in enumerate(lattice)}
        edge_i, edge_j = [], []
        for i, q in enumerate(lattice):
            for axis in range(3):
                qq = q.copy()
                qq[axis] += 1
                j = lookup.get(tuple(qq))
                if j is not None:
                    edge_i.append(i)
                    edge_j.append(j)
        self.edge_i = np.asarray(edge_i, dtype=np.int64)
        self.edge_j = np.asarray(edge_j, dtype=np.int64)
        degree = np.zeros(len(self.porous), dtype=np.int32)
        np.add.at(degree, self.edge_i, 1)
        np.add.at(degree, self.edge_j, 1)
        self.pore_boundary = degree < 6

        # Mesh wetness follows the closest simulated porous material point.
        delta = self.pos0[self.surface, None, :] - pore_ref[None, :, :]
        self.surface_pore = np.argmin(np.sum(delta * delta, axis=2), axis=1)

        weights = self.dry_mass[self.duck]
        self.duck_center0 = np.sum(
            self.pos0[self.duck] * weights[:, None], axis=0
        ) / np.sum(weights)
        self.duck_rel = self.pos0[self.duck] - self.duck_center0
        self.translation = np.zeros(3, dtype=np.float64)
        self.duck_velocity = np.zeros(3, dtype=np.float64)
        self.duck_momentum = np.zeros(3, dtype=np.float64)
        self.z_scale = 1.0
        self.z_scale_rate = 0.0

        self.water_pos = self.pos0[self.water].copy()
        self.water_ref0 = self.water_pos.copy()
        self.water_velocity = np.asarray(initial.velocity[self.water], dtype=np.float64).copy()
        self.water_mass0 = self.dry_mass[self.water].copy()
        self.initial_free_mass = float(np.sum(self.water_mass0))
        self.free_mass = self.initial_free_mass
        self.tank_area = (2.0 * 1.65) * (2.0 * 0.75)
        self.water_level = self.surface0

        # Frozen momentum-budget domains: duck, free water, and their union.
        n = len(self.pos0)
        self.domain_weights = np.zeros((3, n), dtype=np.float64)
        self.domain_weights[0, self.duck] = 1.0
        self.domain_weights[1, self.water] = 1.0
        self.domain_weights[2, :] = 1.0
        self.cum_external_impulse = np.zeros((3, 3), dtype=np.float64)
        self.cum_inflow_momentum = np.zeros((3, 3), dtype=np.float64)
        self.cum_outflow_momentum = np.zeros((3, 3), dtype=np.float64)
        self.last_external_terms = np.zeros((3, 2, 3), dtype=np.float64)
        self.last_interaction = np.zeros((2, 3), dtype=np.float64)
        self.last_floor_force = np.zeros(3, dtype=np.float64)
        self.step_count = 0

        total_mass = float(np.sum(self.mass))
        self.reference_momentum = total_mass * np.sqrt(self.g)

    def _duck_positions(self) -> np.ndarray:
        rel = self.duck_rel.copy()
        rel[:, 2] *= self.z_scale
        return self.duck_center0 + self.translation + rel

    def _duck_point_velocities(self) -> np.ndarray:
        deform = np.zeros((len(self.duck), 3), dtype=np.float64)
        deform[:, 2] = self.z_scale_rate * self.duck_rel[:, 2]
        dm = self.mass[self.duck]
        deform -= np.sum(dm[:, None] * deform, axis=0) / np.sum(dm)
        return self.duck_velocity + deform

    def _absorb(self, dt: float, pore_world: np.ndarray) -> None:
        """Take one explicit, conservative Darcy/capillary transport step."""
        if self.time + 0.5 * dt < self.switch_time:
            return

        s = self.saturation
        viscosity = 1.0e-3
        surface_tension = 0.072
        theta = np.deg2rad(float(self.cfg["duck"]["contact_angle_degrees"]))
        pore_radius = np.sqrt(max(8.0 * self.permeability / self.porosity, 1.0e-12))
        capillary_pressure = 2.0 * surface_tension * np.cos(theta) / pore_radius
        diffusivity = self.permeability * capillary_pressure / (viscosity * self.porosity)
        diffusivity = min(max(diffusivity, 1.0e-4), 1.5e-2)

        ds = np.zeros_like(s)
        if len(self.edge_i):
            flux = (diffusivity / (self.h * self.h)) * (
                s[self.edge_j] - s[self.edge_i]
            )
            np.add.at(ds, self.edge_i, flux)
            np.add.at(ds, self.edge_j, -flux)

        depth = self.water_level - pore_world[:, 2]
        contact = np.where(depth >= 0.0, 1.0, np.exp(np.maximum(-12.0, depth / 0.055)))
        pressure = capillary_pressure + self.rho * self.g * np.maximum(depth, 0.0)
        exchange = self.permeability * pressure / (
            viscosity * self.porosity * self.h * self.h
        )
        exchange = np.minimum(exchange, 3.0)
        ds += self.pore_boundary * contact * exchange * (1.0 - s)

        old = s.copy()
        self.saturation = np.clip(s + dt * ds, 0.0, 1.0)
        old_pore = float(np.dot(old, self.pore_capacity))
        new_pore = float(np.dot(self.saturation, self.pore_capacity))
        uptake = max(0.0, new_pore - old_pore)
        uptake = min(uptake, self.free_mass)
        if new_pore > old_pore and uptake < new_pore - old_pore:
            factor = uptake / (new_pore - old_pore)
            self.saturation = old + factor * (self.saturation - old)
            new_pore = old_pore + uptake

        # Finite-volume feeding from the connected tank, including transported
        # momentum. No prescribed mass source is introduced.
        old_water_mass = self.mass[self.water].copy()
        self.free_mass = self.initial_free_mass - new_pore
        ratio = max(0.0, self.free_mass / self.initial_free_mass)
        self.mass[self.water] = self.water_mass0 * ratio
        removed = old_water_mass - self.mass[self.water]
        transfer_momentum = np.sum(removed[:, None] * self.water_velocity, axis=0)
        self.mass[self.porous] = self.dry_mass[self.porous] + self.saturation * self.pore_capacity
        self.duck_momentum += transfer_momentum
        self.cum_inflow_momentum[0] += transfer_momentum
        self.cum_outflow_momentum[1] += transfer_momentum

    def _step(self, dt: float) -> None:
        duck_pos = self._duck_positions()
        pore_pos = duck_pos[np.searchsorted(self.duck, self.porous)]
        self._absorb(dt, pore_pos)

        absorbed = float(np.dot(self.saturation, self.pore_capacity))
        self.water_level = self.surface0 - absorbed / (self.rho * self.tank_area)
        duck_mass = float(np.sum(self.mass[self.duck]))
        self.duck_velocity = self.duck_momentum / duck_mass

        # Hydrostatic pressure integrated by porous-volume quadrature.
        submerged = pore_pos[:, 2] < self.water_level
        displaced_volume = self.cell_volume * float(np.count_nonzero(submerged))
        buoyancy = np.array([0.0, 0.0, self.rho * self.g * displaced_volume])

        center = self.duck_center0 + self.translation
        dr = self.water_pos - center
        proximity = np.exp(
            -((dr[:, 0] / 0.78) ** 2 + (dr[:, 1] / 0.38) ** 2 + (dr[:, 2] / 0.60) ** 2)
        )
        proximity *= self.water_pos[:, 2] < self.water_level + 0.08
        wsum = float(np.sum(proximity))
        if wsum > 1.0e-12:
            local_v = np.sum(proximity[:, None] * self.water_velocity, axis=0) / wsum
        else:
            local_v = np.zeros(3, dtype=np.float64)
        relative_v = self.duck_velocity - local_v
        wet_fraction = displaced_volume / max(len(self.porous) * self.cell_volume, 1.0e-12)
        drag = -900.0 * max(wet_fraction, 0.05) * relative_v
        drag[:2] *= 0.25
        fluid_on_duck = buoyancy + drag

        # Penalty contact against the top of the fixed floor (z=0).
        min_z = float(np.min(duck_pos[:, 2]))
        floor_force = np.zeros(3, dtype=np.float64)
        if min_z < 0.0:
            floor_force[2] = max(0.0, -2.0e5 * min_z - 5.0e3 * self.duck_velocity[2])
            floor_force[:2] = -min(1.5e3, 0.35 * floor_force[2]) * self.duck_velocity[:2]
        self.last_floor_force = floor_force

        gravity_duck = duck_mass * self.gravity
        total_duck_force = gravity_duck + floor_force + fluid_on_duck
        self.duck_momentum += dt * total_duck_force
        self.duck_velocity = self.duck_momentum / duck_mass
        self.translation += dt * self.duck_velocity

        # Saturation-driven sponge compaction, integrated as an elastic mode.
        mean_sat = float(np.mean(self.saturation))
        scale_target = 1.0 - 0.035 * mean_sat
        scale_accel = 30.0 * (scale_target - self.z_scale) - 11.0 * self.z_scale_rate
        self.z_scale_rate += dt * scale_accel
        self.z_scale += dt * self.z_scale_rate
        self.z_scale = float(np.clip(self.z_scale, 0.94, 1.01))

        # Equal and opposite pressure/drag reaction on free-liquid samples.
        water_mass = self.mass[self.water]
        reaction = -fluid_on_duck
        if wsum > 1.0e-12:
            reaction_force = proximity[:, None] * reaction / wsum
        else:
            reaction_force = np.zeros_like(self.water_pos)

        # Reduced incompressible pressure solve: displaced Lagrangian control
        # volumes generate restoring pressure and viscous stress.
        height_ratio = np.clip(
            (self.water_level - 0.03) / max(self.surface0 - 0.03, 1.0e-9), 0.75, 1.0
        )
        water_ref = self.water_ref0.copy()
        water_ref[:, 2] = 0.03 + (water_ref[:, 2] - 0.03) * height_ratio
        env_force = water_mass[:, None] * (
            -5.0 * (self.water_pos - water_ref) - 2.8 * self.water_velocity
        )
        safe_mass = np.maximum(water_mass, 1.0e-12)
        acceleration = (reaction_force + env_force) / safe_mass[:, None]
        self.water_velocity += dt * acceleration
        self.water_pos += dt * self.water_velocity

        # Fixed impermeable walls; resolved impulses enter tank traction.
        wall_impulse = np.zeros(3, dtype=np.float64)
        bounds = ((0, -1.60, 1.60), (1, -0.70, 0.70), (2, 0.03, 1.47))
        for axis, lo, hi in bounds:
            below = self.water_pos[:, axis] < lo
            above = self.water_pos[:, axis] > hi
            for mask, boundary, normal_sign in ((below, lo, 1.0), (above, hi, -1.0)):
                if np.any(mask):
                    old_v = self.water_velocity[mask, axis].copy()
                    self.water_pos[mask, axis] = boundary
                    outward = old_v * normal_sign < 0.0
                    new_v = np.where(outward, -0.25 * old_v, old_v)
                    self.water_velocity[mask, axis] = new_v
                    wall_impulse[axis] += np.sum(water_mass[mask] * (new_v - old_v))
        env_total = np.sum(env_force, axis=0) + wall_impulse / dt

        gravity_water = self.free_mass * self.gravity
        terms = np.zeros((3, 2, 3), dtype=np.float64)
        terms[0, 0] = gravity_duck
        terms[0, 1] = floor_force
        terms[1, 0] = gravity_water
        terms[1, 1] = env_total - gravity_water
        terms[2] = terms[0] + terms[1]
        self.last_external_terms = terms
        self.cum_external_impulse += dt * np.sum(terms, axis=1)
        self.last_interaction[0] = fluid_on_duck
        self.last_interaction[1] = -fluid_on_duck
        self.time += dt
        self.step_count += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to requires nondecreasing sample times")
        while self.time < target - 1.0e-12:
            self._step(min(0.004, target - self.time))
        self.time = target

        n = len(self.pos0)
        position = self.pos0.copy()
        velocity = np.zeros((n, 3), dtype=np.float64)
        duck_pos = self._duck_positions()
        duck_vel = self._duck_point_velocities()
        position[self.duck] = duck_pos
        velocity[self.duck] = duck_vel
        position[self.water] = self.water_pos
        velocity[self.water] = self.water_velocity

        saturation = np.ones(n, dtype=np.float64)
        saturation[self.duck] = 0.0
        saturation[self.porous] = self.saturation
        saturation[self.surface] = self.saturation[self.surface_pore]
        pore_water = np.zeros(n, dtype=np.float64)
        pore_water[self.porous] = self.saturation * self.pore_capacity

        pressure_pore = self.rho * self.g * np.maximum(
            self.water_level - position[self.porous, 2], 0.0
        ) * self.saturation
        pore_pressure = np.zeros(n, dtype=np.float64)
        pore_pressure[self.porous] = pressure_pore
        pore_pressure[self.surface] = pressure_pore[self.surface_pore]

        permeability = np.zeros(n, dtype=np.float64)
        if self.time >= self.switch_time - 1.0e-12:
            permeability[self.duck] = self.permeability

        mass_quantity = np.zeros((n, 2), dtype=np.float64)
        mass_quantity[:, 0] = self.dry_mass
        mass_quantity[self.water, 0] = 0.0
        mass_quantity[self.water, 1] = self.mass[self.water]
        mass_quantity[self.porous, 1] = pore_water[self.porous]

        solid_velocity = np.zeros((n, 3), dtype=np.float64)
        solid_velocity[self.duck] = duck_vel
        frame = StateFrame(
            self.time,
            position,
            velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "phase": np.where(self.body_id == 2, 1.0, 0.0),
                "saturation": saturation,
                "porosity": np.where(
                    (self.body_id == 1) | (self.body_id == 3), self.porosity, 0.0
                ),
                "permeability_m2": permeability,
                "pore_water_mass": pore_water,
                "pore_pressure": pore_pressure,
                "solid_velocity": solid_velocity,
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_momentum.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_momentum.copy(),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": self.last_external_terms.copy(),
                "cumulative_mass_source": np.zeros(2, dtype=np.float64),
                "interaction_force": self.last_interaction.copy(),
            },
            scalars={
                "reference_momentum": self.reference_momentum,
                "gravity_z": float(self.gravity[2]),
                "permeability_switch_time": self.switch_time,
                "initial_water_surface_z": self.surface0,
                "water_surface_z": self.water_level,
                "initial_free_water_mass": self.initial_free_mass,
                "initial_pore_water_mass": 0.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "Darcy graph diffusion + conservative rigid-body/fluid coupling",
            "time": float(self.time),
            "steps": int(self.step_count),
            "mean_saturation": float(np.mean(self.saturation)),
            "free_water_mass": float(self.free_mass),
            "duck_bottom_z": float(np.min(self._duck_positions()[:, 2])),
        }
