"""Self-contained coupled particle solver for the sand-dam breach scene."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Weakly-compressible water coupled to wet-weakened cohesive grains."""

    SAND = 1
    WATER = 2

    def initialize(self, scene, seed: int) -> None:
        initial = scene.initial_state()
        self.time = 0.0
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.attributes = {k: np.asarray(v).copy() for k, v in initial.attributes.items()}
        self.saturation = np.asarray(self.attributes["sand_saturation"], dtype=np.float64).copy()
        self.gravity = np.asarray(scene.config.get("gravity", (0.0, 0.0, -9.81)), dtype=np.float64)
        self.sand = self.body_id == self.SAND
        self.water = self.body_id == self.WATER
        self.sand_indices = np.flatnonzero(self.sand)
        self.n = len(self.position)
        self.rng = np.random.default_rng(int(seed))

        # Initial proximity bonds give the intact skeleton axial and shear
        # stiffness.  Seeded small disorder represents natural sand variability.
        sp = self.position[self.sand_indices]
        local = cKDTree(sp).query_pairs(0.126, output_type="ndarray")
        if len(local):
            self.bond_i = self.sand_indices[local[:, 0]]
            self.bond_j = self.sand_indices[local[:, 1]]
            d = self.position[self.bond_j] - self.position[self.bond_i]
            self.bond_rest = np.linalg.norm(d, axis=1)
            self.bond_toughness = self.rng.uniform(0.88, 1.12, len(local))
        else:
            self.bond_i = np.empty(0, dtype=np.int64)
            self.bond_j = np.empty(0, dtype=np.int64)
            self.bond_rest = np.empty(0)
            self.bond_toughness = np.empty(0)
        self.bond_damage = np.zeros(len(self.bond_i))

        self.domain_weights = np.vstack((np.ones(self.n), self.sand.astype(float),
                                         self.water.astype(float)))
        self.cumulative_external_impulse = np.zeros((3, 3))
        self.cumulative_inflow = np.zeros((3, 3))
        self.cumulative_outflow = np.zeros((3, 3))
        self.cumulative_mass_source = np.zeros(2)
        self.last_boundary_force = np.zeros((3, 3))
        self.last_interaction_force = np.zeros((2, 3))
        total_mass = float(self.mass.sum())
        self.reference_momentum = max(
            1.0, total_mass * np.sqrt(float(np.linalg.norm(self.gravity)) * 1.75)
        )
        self.step_count = 0

    @staticmethod
    def _scatter(force, i, j, pair_force) -> None:
        """Scatter an antisymmetric force; pair_force is the force on i."""
        if len(i):
            np.add.at(force, i, pair_force)
            np.add.at(force, j, -pair_force)

    def _cohesion(self, force, dt: float) -> None:
        active = self.bond_damage < 1.0
        if not np.any(active):
            return
        ids = np.flatnonzero(active)
        i, j = self.bond_i[ids], self.bond_j[ids]
        delta = self.position[j] - self.position[i]
        distance = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(distance[:, None], 1e-10)
        strain = (distance - self.bond_rest[ids]) / self.bond_rest[ids]
        wet = 0.5 * (self.saturation[i] + self.saturation[j])
        limit = (0.018 + 0.065 * (1.0 - wet)) * self.bond_toughness[ids]
        overload = np.maximum(strain / np.maximum(limit, 1e-5) - 1.0, 0.0)
        wet_fatigue = np.maximum(strain - 0.005, 0.0) * np.maximum(wet - 0.45, 0.0)
        self.bond_damage[ids] = np.minimum(
            1.0, self.bond_damage[ids] + dt * (7.0 * overload + 14.0 * wet_fatigue)
        )
        ids = ids[self.bond_damage[ids] < 1.0]
        if not len(ids):
            return
        i, j = self.bond_i[ids], self.bond_j[ids]
        delta = self.position[j] - self.position[i]
        distance = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(distance[:, None], 1e-10)
        extension = distance - self.bond_rest[ids]
        wet = 0.5 * (self.saturation[i] + self.saturation[j])
        vn = np.einsum("ij,ij->i", self.velocity[j] - self.velocity[i], normal)
        stiffness = 1550.0 * (1.0 - 0.84 * wet) * (1.0 - self.bond_damage[ids])
        self._scatter(force, i, j, (stiffness * extension + 10.0 * vn)[:, None] * normal)

    def _neighbors(self, force, dt: float) -> np.ndarray:
        pairs = cKDTree(self.position).query_pairs(0.145, output_type="ndarray")
        interaction = np.zeros(3)
        if not len(pairs):
            return interaction
        i, j = pairs[:, 0], pairs[:, 1]
        delta = self.position[j] - self.position[i]
        distance = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(distance[:, None], 1e-10)
        rel = self.velocity[j] - self.velocity[i]
        vn = np.einsum("ij,ij->i", rel, normal)
        mi, mj = self.body_id[i], self.body_id[j]

        # Water pressure and viscosity, both momentum conserving.
        mask = (mi == self.WATER) & (mj == self.WATER)
        if np.any(mask):
            rr, nn = distance[mask], normal[mask]
            overlap = np.maximum(0.108 - rr, 0.0)
            pf = (-1350.0 * overlap + 12.0 * np.minimum(vn[mask], 0.0))[:, None] * nn
            pf += 2.2 * rel[mask] * np.maximum(1.0 - rr / 0.145, 0.0)[:, None]
            self._scatter(force, i[mask], j[mask], pf)

        # Granular exclusion persists when cohesive bonds have failed.
        mask = (mi == self.SAND) & (mj == self.SAND) & (distance < 0.083)
        if np.any(mask):
            pf = (-2600.0 * (0.083 - distance[mask])
                  + 18.0 * np.minimum(vn[mask], 0.0))[:, None] * normal[mask]
            self._scatter(force, i[mask], j[mask], pf)

        # Reorient mixed pairs so forces and wetting are expressed sand-first.
        mixed = mi != mj
        if np.any(mixed):
            pi, pj = i[mixed], j[mixed]
            first_sand = self.body_id[pi] == self.SAND
            si = np.where(first_sand, pi, pj)
            wi = np.where(first_sand, pj, pi)
            dsw = self.position[wi] - self.position[si]
            rr = np.linalg.norm(dsw, axis=1)
            nn = dsw / np.maximum(rr[:, None], 1e-10)
            relsw = self.velocity[wi] - self.velocity[si]
            vnsw = np.einsum("ij,ij->i", relsw, nn)
            weight = np.maximum(1.0 - rr / 0.145, 0.0)
            overlap = np.maximum(0.094 - rr, 0.0)
            # Equal/opposite interphase pressure and Darcy-like momentum drag.
            on_sand = (-1900.0 * overlap
                       + 18.0 * np.minimum(vnsw, 0.0))[:, None] * nn
            on_sand += (7.0 * weight)[:, None] * relsw
            np.add.at(force, si, on_sand)
            np.add.at(force, wi, -on_sand)
            interaction = on_sand.sum(axis=0)
            # Pore filling depends solely on actual water/grain proximity.
            np.add.at(self.saturation, si,
                      dt * 1.35 * (1.0 - self.saturation[si]) * weight)

        np.clip(self.saturation, 0.0, 1.0, out=self.saturation)
        self.saturation[self.water] = 1.0
        return interaction

    def _diffuse_saturation(self, dt: float) -> None:
        active = self.bond_damage < 1.0
        i, j = self.bond_i[active], self.bond_j[active]
        if not len(i):
            return
        flux = 0.13 * (self.saturation[j] - self.saturation[i])
        change = np.zeros(self.n)
        np.add.at(change, i, flux)
        np.add.at(change, j, -flux)
        self.saturation[self.sand] += dt * change[self.sand]
        np.clip(self.saturation, 0.0, 1.0, out=self.saturation)

    def _boundaries(self, velocity_before) -> np.ndarray:
        """Project against the inner faces of the frozen flume boxes."""
        radii = np.where(self.sand, 0.040, 0.047)
        # (axis, plane, inward sense). Side walls remain impermeable supports.
        limits = ((0, -3.93, 1), (0, 4.43, -1),
                  (1, -0.95, 1), (1, 0.95, -1), (2, 0.0, 1))
        for axis, plane, sense in limits:
            bound = plane + sense * radii
            if sense > 0:
                hit = self.position[:, axis] < bound
                outward = self.velocity[:, axis] < 0.0
            else:
                hit = self.position[:, axis] > bound
                outward = self.velocity[:, axis] > 0.0
            active = hit & outward
            self.position[hit, axis] = bound[hit]
            self.velocity[active, axis] *= -0.025
            if axis == 2 and np.any(active):
                friction = np.where(self.sand[active], 0.90, 0.985)
                self.velocity[active, 0] *= friction
                self.velocity[active, 1] *= friction
        particle_impulse = self.mass[:, None] * (self.velocity - velocity_before)
        result = np.zeros((3, 3))
        result[1] = particle_impulse[self.sand].sum(axis=0)
        result[2] = particle_impulse[self.water].sum(axis=0)
        result[0] = result[1] + result[2]
        return result

    def _gravity_force(self) -> np.ndarray:
        result = np.zeros((3, 3))
        result[1] = self.mass[self.sand].sum() * self.gravity
        result[2] = self.mass[self.water].sum() * self.gravity
        result[0] = result[1] + result[2]
        return result

    def _step(self, dt: float) -> None:
        force = self.mass[:, None] * self.gravity
        self._cohesion(force, dt)
        interaction = self._neighbors(force, dt)
        self._diffuse_saturation(dt)
        # Symplectic Euler is stable and dissipative for contact dynamics.
        self.velocity += dt * force / self.mass[:, None]
        velocity_before_boundary = self.velocity.copy()
        self.position += dt * self.velocity
        boundary_impulse = self._boundaries(velocity_before_boundary)
        gravity_force = self._gravity_force()
        self.last_boundary_force = boundary_impulse / dt
        self.cumulative_external_impulse += gravity_force * dt + boundary_impulse
        self.last_interaction_force[0] = interaction
        self.last_interaction_force[1] = -interaction
        self.time += dt
        self.step_count += 1

    def _frame(self) -> StateFrame:
        attrs = {k: np.asarray(v).copy() for k, v in self.attributes.items()}
        attrs["sand_saturation"] = self.saturation.copy()
        attrs["mass_quantity"] = self.mass.copy()
        external_force = np.stack((self._gravity_force(), self.last_boundary_force), axis=1)
        fields = {
            "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cumulative_inflow.copy(),
            "budget_cumulative_outflow_momentum": self.cumulative_outflow.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": external_force,
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "interaction_force": self.last_interaction_force.copy(),
        }
        scalars = {
            "reference_momentum": float(self.reference_momentum),
            "initial_sand_mass": float(self.mass[self.sand].sum()),
            "initial_water_mass": float(self.mass[self.water].sum()),
        }
        frame = StateFrame(float(self.time), self.position.copy(), self.velocity.copy(),
                           mass=self.mass.copy(), body_id=self.body_id.copy(),
                           attributes=attrs, fields=fields, scalars=scalars)
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1e-10:
            raise ValueError("advance_to times must be nondecreasing")
        while self.time < target - 1e-12:
            self._step(min(0.005, target - self.time))
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "coupled_particle_fluid_and_wet_cohesive_grains",
            "steps": int(self.step_count),
            "particle_count": int(self.n),
            "intact_bond_fraction": float(np.mean(self.bond_damage < 1.0))
            if len(self.bond_damage) else 0.0,
            "maximum_sand_saturation": float(np.max(self.saturation[self.sand])),
            "sand_mass_retained": float(self.mass[self.sand].sum()),
        }
