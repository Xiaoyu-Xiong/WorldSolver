"""Deterministic two-way liquid/deformable-body solver for the toy drop.

The shell is a small Galerkin finite-element model: its mesh is represented by
three uniform strain modes and rigid translation. Liquid samples are
Lagrangian particles with short-range pressure and viscosity. Penalty tractions
at the moving cavity wall are applied with equal and opposite generalized force
to the shell, so neither side of the coupling is kinematic.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit, fixed-parameter multiphysics time integrator."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # The discretization is deterministic and has no random forcing.
        initial = scene.initial_state()

        self._time = 0.0
        self._initial_position = np.asarray(initial.position, dtype=float).copy()
        self._body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self._mass = np.asarray(initial.mass, dtype=float).copy()
        self._n = len(self._mass)

        self._shell_indices = np.flatnonzero(self._body_id == 1)
        self._liquid_indices = np.flatnonzero(self._body_id == 2)
        self._red_indices = np.flatnonzero(self._body_id == 3)
        self._yellow_indices = np.flatnonzero(self._body_id == 4)
        self._marker_indices = np.concatenate((self._red_indices,
                                                self._yellow_indices))

        # This is the cavity coordinate origin used by scene.py.
        self._origin = np.array((0.0, 0.0, 1.36), dtype=float)
        shell0 = self._initial_position[self._shell_indices]
        self._shell_reference = shell0 - self._origin
        self._shell_sample_mass = self._mass[self._shell_indices]
        self._shell_mass = float(np.sum(self._shell_sample_mass))

        # Mesh point x = x0 + translation + (scale - 1) * reference.
        self._translation = np.zeros(3, dtype=float)
        self._translation_velocity = np.asarray(
            initial.velocity[self._shell_indices[0]], dtype=float).copy()
        self._scale = np.ones(3, dtype=float)
        self._scale_velocity = np.zeros(3, dtype=float)

        # Consistent mass matrices retain translation/strain coupling because
        # the cavity origin need not coincide with the shell mass centroid.
        self._mode_mass = []
        for axis in range(3):
            r = self._shell_reference[:, axis]
            first = float(np.dot(self._shell_sample_mass, r))
            second = float(np.dot(self._shell_sample_mass, r * r))
            self._mode_mass.append(np.array(
                ((self._shell_mass, first), (first, second)), dtype=float))

        self._liquid_position = self._initial_position[
            self._liquid_indices].copy()
        self._liquid_velocity = np.asarray(
            initial.velocity[self._liquid_indices], dtype=float).copy()
        self._liquid_mass = self._mass[self._liquid_indices].copy()
        self._fluid_mass = float(np.sum(self._liquid_mass))

        # Fixed local-neighborhood candidates. The pressure branch is
        # compression-only (no tensile springs), as required for a liquid.
        delta = (self._liquid_position[:, None, :] -
                 self._liquid_position[None, :, :])
        distance2 = np.einsum("ijk,ijk->ij", delta, delta)
        edge_i, edge_j = np.where(np.triu(
            (distance2 > 1.0e-12) & (distance2 < 0.171 ** 2), 1))
        self._edge_i = edge_i.astype(np.intp)
        self._edge_j = edge_j.astype(np.intp)

        # Markers begin coincident with fluid material samples. Retaining that
        # material label is exact passive Lagrangian advection.
        marker0 = self._initial_position[self._marker_indices]
        md = marker0[:, None, :] - self._liquid_position[None, :, :]
        self._marker_source = np.argmin(
            np.einsum("ijk,ijk->ij", md, md), axis=1)

        self._gravity = np.array((0.0, 0.0, -9.81), dtype=float)
        self._cumulative_impulse = np.zeros((3, 3), dtype=float)
        self._last_external_terms = np.zeros((3, 3, 3), dtype=float)
        self._last_interaction = np.zeros(3, dtype=float)
        self._step_count = 0

        # Frozen budgets: shell, liquid (including passive labels), and union.
        self._budget_weights = np.zeros((3, self._n), dtype=float)
        self._budget_weights[0, self._body_id == 1] = 1.0
        self._budget_weights[1, np.isin(self._body_id, (2, 3, 4))] = 1.0
        self._budget_weights[2, :] = 1.0

        self._reference_velocity = float(np.sqrt(9.81))
        self._reference_momentum = float(
            (self._shell_mass + self._fluid_mass) * self._reference_velocity)

    def _shell_state(self) -> tuple[np.ndarray, np.ndarray]:
        position = (self._origin + self._translation +
                    self._shell_reference * self._scale)
        velocity = (self._translation_velocity +
                    self._shell_reference * self._scale_velocity)
        return position, velocity

    def _fluid_internal_force(self) -> np.ndarray:
        """Pairwise compressive pressure and symmetric viscous traction."""
        force = np.zeros_like(self._liquid_position)
        i = self._edge_i
        j = self._edge_j
        d = self._liquid_position[j] - self._liquid_position[i]
        distance = np.sqrt(np.einsum("ij,ij->i", d, d) + 1.0e-18)
        active = distance < 0.122
        if not np.any(active):
            return force

        ia = i[active]
        ja = j[active]
        da = d[active]
        ra = distance[active]
        direction = da / ra[:, None]
        overlap = 0.122 - ra

        # Barotropic particle pressure: compression only, never tension.
        pressure = 5.0 * overlap
        relative_velocity = self._liquid_velocity[ja] - self._liquid_velocity[ia]
        viscosity = 0.006 * relative_velocity
        pair_force = -pressure[:, None] * direction + viscosity
        np.add.at(force, ia, pair_force)
        np.add.at(force, ja, -pair_force)
        return force

    def _wall_force(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return liquid traction and equal/opposite shell reactions."""
        center = self._origin + self._translation
        relative = self._liquid_position - center
        axes = np.array((0.80, 0.49, 0.56), dtype=float) * self._scale
        normalized = relative / axes
        radius = np.sqrt(np.einsum("ij,ij->i", normalized, normalized) +
                         1.0e-18)
        active = radius > 0.965

        liquid_force = np.zeros_like(self._liquid_position)
        shell_reaction = np.zeros(3, dtype=float)
        shape_reaction = np.zeros(3, dtype=float)
        if not np.any(active):
            return liquid_force, shell_reaction, shape_reaction

        rel = relative[active]
        rad = radius[active]
        raw_normal = rel / (axes * axes)
        normal = raw_normal / np.linalg.norm(raw_normal, axis=1)[:, None]
        # First-order signed physical distance from the implicit ellipsoid.
        penetration = (rad - 0.965) / np.linalg.norm(
            raw_normal / rad[:, None], axis=1)
        wall_velocity = (self._translation_velocity +
                         rel * (self._scale_velocity / self._scale))
        relative_velocity = self._liquid_velocity[active] - wall_velocity
        outgoing_speed = np.einsum("ij,ij->i", relative_velocity, normal)
        magnitude = (320.0 * penetration +
                     0.65 * np.maximum(outgoing_speed, 0.0))
        contact_force = -magnitude[:, None] * normal
        liquid_force[active] = contact_force

        reaction_each = -contact_force
        shell_reaction = np.sum(reaction_each, axis=0)
        contact_reference = rel / self._scale
        shape_reaction = np.sum(reaction_each * contact_reference, axis=0)
        return liquid_force, shell_reaction, shape_reaction

    def _ground_force(self, shell_position: np.ndarray,
                      shell_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        depth = np.maximum(-shell_position[:, 2], 0.0)
        active = depth > 0.0
        point_force = np.zeros_like(shell_position)
        if np.any(active):
            normal_force = (3500.0 * depth[active] -
                            14.0 * shell_velocity[active, 2])
            point_force[active, 2] = np.maximum(normal_force, 0.0)
        resultant = np.sum(point_force, axis=0)
        generalized = np.sum(point_force * self._shell_reference, axis=0)
        return resultant, generalized

    def _evaluate_forces(self):
        shell_position, shell_velocity = self._shell_state()
        fluid_force = self._fluid_internal_force()
        fluid_force += self._liquid_mass[:, None] * self._gravity

        wall_on_fluid, wall_on_shell, wall_shape = self._wall_force()
        fluid_force += wall_on_fluid
        floor_force, floor_shape = self._ground_force(shell_position,
                                                       shell_velocity)

        shell_gravity = self._shell_mass * self._gravity
        gravity_shape = np.array([
            np.dot(self._shell_sample_mass * self._gravity[a],
                   self._shell_reference[:, a]) for a in range(3)
        ])

        # Hyperelastic independent strain plus a near-incompressible volume
        # energy and Rayleigh damping.
        strain = self._scale - 1.0
        jacobian = float(np.prod(self._scale))
        elastic_shape = -(650.0 * strain + 1700.0 * strain ** 3)
        elastic_shape -= 900.0 * (jacobian - 1.0) * (
            jacobian / self._scale)
        elastic_shape -= 11.0 * self._scale_velocity

        translation_force = shell_gravity + floor_force + wall_on_shell
        shape_force = (gravity_shape + floor_shape + wall_shape +
                       elastic_shape)

        # Force terms are gravity, floor traction, and coupling traction.
        terms = np.zeros((3, 3, 3), dtype=float)
        terms[0, 0] = shell_gravity
        terms[0, 1] = floor_force
        terms[0, 2] = wall_on_shell
        terms[1, 0] = self._fluid_mass * self._gravity
        terms[1, 2] = -wall_on_shell
        terms[2, 0] = (self._shell_mass + self._fluid_mass) * self._gravity
        terms[2, 1] = floor_force
        return fluid_force, translation_force, shape_force, terms, wall_on_shell

    def _substep(self, dt: float) -> None:
        (fluid_force, translation_force, shape_force,
         terms, interaction) = self._evaluate_forces()

        self._liquid_velocity += (dt * fluid_force /
                                  self._liquid_mass[:, None])
        self._liquid_position += dt * self._liquid_velocity

        translation_acceleration = np.empty(3, dtype=float)
        scale_acceleration = np.empty(3, dtype=float)
        for axis in range(3):
            acceleration = np.linalg.solve(
                self._mode_mass[axis],
                np.array((translation_force[axis], shape_force[axis])))
            translation_acceleration[axis] = acceleration[0]
            scale_acceleration[axis] = acceleration[1]

        self._translation_velocity += dt * translation_acceleration
        self._scale_velocity += dt * scale_acceleration
        self._translation += dt * self._translation_velocity
        self._scale += dt * self._scale_velocity

        self._cumulative_impulse += dt * np.sum(terms, axis=1)
        self._last_external_terms = terms
        self._last_interaction = interaction
        self._time += dt
        self._step_count += 1

    def _boundary_fields(self, shell_velocity: np.ndarray):
        reference = self._shell_reference
        # Inverse-transpose transform of approximate undeformed normals.
        raw = reference / np.array((1.0, 0.64 ** 2, 0.82 ** 2))
        radial_zero = np.linalg.norm(raw, axis=1) < 1.0e-12
        raw[radial_zero, 2] = np.where(reference[radial_zero, 2] >= 0.0,
                                       1.0, -1.0)
        raw /= self._scale
        normal = raw / np.linalg.norm(raw, axis=1)[:, None]
        # One additional boundary sample represents the stationary floor.
        normal = np.vstack((normal, np.array((0.0, 0.0, 1.0))))
        boundary_velocity = np.vstack((shell_velocity, np.zeros((1, 3))))
        return normal, boundary_velocity

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self._time - 1.0e-12:
            raise ValueError("advance_to times must be nondecreasing")

        # Resolve the explicit penalty frequencies with a fixed maximum step.
        maximum_dt = 0.0005
        while self._time < target - 1.0e-12:
            self._substep(min(maximum_dt, target - self._time))

        # Evaluate instantaneous fields at exactly the requested sample state.
        (_, _, _, current_terms,
         current_interaction) = self._evaluate_forces()
        self._last_external_terms = current_terms
        self._last_interaction = current_interaction

        shell_position, shell_velocity = self._shell_state()
        position = self._initial_position.copy()
        velocity = np.zeros_like(position)
        position[self._shell_indices] = shell_position
        velocity[self._shell_indices] = shell_velocity
        position[self._liquid_indices] = self._liquid_position
        velocity[self._liquid_indices] = self._liquid_velocity
        position[self._marker_indices] = self._liquid_position[self._marker_source]
        velocity[self._marker_indices] = self._liquid_velocity[self._marker_source]

        boundary_normal, boundary_velocity = self._boundary_fields(shell_velocity)
        fields = {
            "boundary_normal": boundary_normal,
            "boundary_target_velocity": boundary_velocity.copy(),
            "boundary_velocity": boundary_velocity,
            "budget_cumulative_external_impulse": self._cumulative_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((3, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((3, 3), dtype=float),
            "budget_domain_weights": self._budget_weights.copy(),
            "budget_external_force": current_terms,
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "interaction_force": np.vstack((current_interaction,
                                             -current_interaction)),
        }
        attributes = {
            "passive_marker": np.isin(self._body_id, (3, 4)).astype(float),
            "mass_quantity": self._mass.copy(),
        }
        scalars = {
            "gravity_z": -9.81,
            "cavity_is_closed": 1.0,
            "reference_momentum": self._reference_momentum,
            "reference_velocity": self._reference_velocity,
        }
        return StateFrame(target, position, velocity, mass=self._mass.copy(),
                          body_id=self._body_id.copy(), attributes=attributes,
                          fields=fields, scalars=scalars)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "Galerkin hyperelastic shell + Lagrangian particle fluid",
            "time": float(self._time),
            "substeps": int(self._step_count),
            "two_way_coupled": True,
            "closed_cavity": True,
        }
