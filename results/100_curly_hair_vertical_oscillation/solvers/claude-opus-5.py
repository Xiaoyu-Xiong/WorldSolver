"""Discrete elastic-rod solver for the vertically driven curly hair clump.

Model
-----
The single physical guide strand is a polygonal centreline of 65 nodes (64
edges) advanced as a discrete elastic rod:

* Inextensibility is a hard constraint; every edge keeps its rest arclength, so
  the material length measured here is centreline arclength, not root-to-tip
  span.  Constraints are enforced with a mass-weighted Lagrange-multiplier
  projection whose multiplier system is tridiagonal and solved directly.
* The natural curly rest shape (circular helix, radius 0.035 m, pitch 0.07 m
  per turn) is stored as the rest second-neighbour distance (discrete turning
  angle, i.e. curvature) and the rest third-neighbour distance (discrete
  dihedral, i.e. torsion).  Deviations are penalised by conservative pairwise
  potentials whose stiffnesses are calibrated to the flexural rigidity EI and
  torsional rigidity GJ of the fibre cross-section.  Both act along the line
  joining their two nodes, so linear momentum is conserved exactly.
* Dissipation uses exactly the two mechanisms named in the frozen config:
  Kelvin-Voigt internal damping (gamma) on the same generalised strains as the
  elastic response, and linear ambient drag (nu) per unit length on nodal
  velocities.  Gravity acts on the lumped nodal masses.
* The root node is kinematic: it holds still until start_time and then follows
  the prescribed vertical sinusoid.  The root orientation is frozen, so the
  first edge is clamped rigidly to the root, supplying the clamped-tangent
  boundary condition.  The tip is free and never prescribed.

Integration is symplectic (semi-implicit) Euler with a fixed substep taken from
an explicit stability bound on the elastic and viscous terms, followed by
constraint projection.  The scheme is deterministic and uses no randomness.
The exported momentum budget is assembled from the same discrete forces and
constraint impulses the integrator applies; the root clamp reaction follows
from the momentum balance of the prescribed nodes.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from worldsolver.contracts import StateFrame

YOUNG_MODULUS = 4.0e9  # Pa, keratin fibre tensile modulus
POISSON_RATIO = 0.48  # near-incompressible cortex
STABILITY_SAFETY = 0.4
MAX_SUBSTEPS = 4000
MIN_SUBSTEPS = 100
PROJECTION_PASSES = 2
PROJECTION_TOLERANCE = 1.0e-12
MIN_BEND_SINE = 0.05


def _rotate_about_axis(vector, axis, angle):
    """Rodrigues rotation of ``vector`` about unit ``axis`` by ``angle``."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return (
        vector * cos_a
        + np.cross(axis, vector) * sin_a
        + axis * (np.dot(axis, vector) * (1.0 - cos_a))
    )


class BenchmarkSolver:
    """Elastic-rod solver honouring the frozen curly-hair configuration."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = dict(scene.config)
        self.config = config
        guide = config["physical_guide"]
        drive = config["root_motion"]

        frame = scene.initial_state()
        self.position = np.array(frame.position, dtype=float)
        self.velocity = np.array(frame.velocity, dtype=float)
        self.node_count = int(self.position.shape[0])
        self.mass = np.array(frame.mass, dtype=float)

        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.sample_dt = float(config["sample_dt"])
        self.duration = float(config["duration"])

        # cross-section and material moduli
        self.material_length = float(guide["material_length"])
        fibre_radius = float(guide["radius"])
        density = float(guide["density"])
        area = np.pi * fibre_radius**2
        self.line_density = density * area
        second_moment = 0.25 * np.pi * fibre_radius**4
        polar_moment = 2.0 * second_moment
        shear_modulus = YOUNG_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
        self.bending_rigidity = YOUNG_MODULUS * second_moment  # EI
        self.torsion_rigidity = shear_modulus * polar_moment  # GJ
        self.internal_gamma = float(guide["internal_damping_gamma"])
        self.ambient_nu = float(guide["ambient_drag_nu"])

        # frozen natural-helix rest geometry
        rest = self.position.copy()
        self.rest_position = rest
        edges = np.diff(rest, axis=0)
        self.rest_edge_length = np.linalg.norm(edges, axis=1)
        self.rest_total_length = float(self.rest_edge_length.sum())
        self.rest_d13 = np.linalg.norm(rest[2:] - rest[:-2], axis=1)
        self.rest_d14 = np.linalg.norm(rest[3:] - rest[:-3], axis=1)
        self.edge_count = int(self.rest_edge_length.size)

        self._build_stiffness(rest, edges)
        self._build_boundary(drive, rest)
        self._build_constraints()
        self._choose_timestep()
        self._init_budget()

        self.step_index = 0
        self.time = 0.0
        self.max_constraint_error = 0.0
        self.root_force = self._probe_root_force()

    def _build_stiffness(self, rest, edges) -> None:
        """Calibrate the neighbour potentials to EI and GJ of the fibre."""
        node_count = self.node_count
        lengths = self.rest_edge_length
        tangents = edges / lengths[:, None]

        # discrete turning angle at interior nodes
        cos_theta = np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0)
        theta0 = np.arccos(cos_theta)
        sin_theta = np.maximum(np.sin(theta0), MIN_BEND_SINE)
        ds_bend = 0.5 * (lengths[:-1] + lengths[1:])

        # d13^2 = a^2 + b^2 - 2ab cos(theta)  =>  d(d13)/dtheta = ab sin(theta)/d13
        d_dtheta = lengths[:-1] * lengths[1:] * sin_theta / np.maximum(self.rest_d13, 1e-12)
        self.k_bend = (self.bending_rigidity / ds_bend) / np.maximum(d_dtheta**2, 1e-18)

        # third-neighbour potential calibrated to GJ through the numerical
        # sensitivity of d14 to the discrete dihedral angle
        ds_twist = lengths[1:-1]
        sensitivity = np.zeros(self.rest_d14.size)
        delta_angle = 1.0e-4
        for index in range(self.rest_d14.size):
            p0 = rest[index]
            p1 = rest[index + 1]
            p2 = rest[index + 2]
            p3 = rest[index + 3]
            axis = p2 - p1
            norm = np.linalg.norm(axis)
            if norm < 1e-14:
                continue
            axis = axis / norm
            rotated = p2 + _rotate_about_axis(p3 - p2, axis, delta_angle)
            sensitivity[index] = abs(
                np.linalg.norm(rotated - p0) - np.linalg.norm(p3 - p0)
            ) / delta_angle
        k_twist = np.where(
            sensitivity > 1e-9,
            (self.torsion_rigidity / ds_twist) / np.maximum(sensitivity**2, 1e-18),
            0.0,
        )
        # keep the torsional surrogate from dominating the explicit stability limit
        bend_scale = float(np.max(self.k_bend)) if self.k_bend.size else 0.0
        self.k_twist = np.minimum(k_twist, bend_scale)

        # Kelvin-Voigt internal damping on the same generalised strains
        gamma = self.internal_gamma
        self.c_bend = self.k_bend * (gamma / max(self.bending_rigidity, 1e-30))
        self.c_twist = self.k_twist * (gamma / max(self.torsion_rigidity, 1e-30))

        # lumped ambient drag coefficient per node (nu per unit length)
        lumped_length = np.zeros(node_count)
        np.add.at(lumped_length, np.arange(node_count - 1), 0.5 * lengths)
        np.add.at(lumped_length, np.arange(1, node_count), 0.5 * lengths)
        self.lumped_length = lumped_length
        self.drag_coefficient = self.ambient_nu * lumped_length

    def _build_boundary(self, drive, rest) -> None:
        axis = np.asarray(drive["axis"], dtype=float)
        norm = np.linalg.norm(axis)
        self.drive_axis = axis / norm if norm > 0.0 else np.array([0.0, 0.0, 1.0])
        self.root_rest = np.asarray(drive["rest_position"], dtype=float)
        self.drive_start = float(drive["start_time"])
        self.drive_amplitude = float(drive["amplitude"])
        self.drive_omega = 2.0 * np.pi * float(drive["frequency"])
        self.drive_phase = float(drive.get("phase", 0.0))
        self.root_orientation_fixed = bool(drive.get("root_orientation_fixed", True))

        # clamped nodes: the root, plus the first edge when the root frame is
        # frozen (clamped-tangent boundary condition)
        clamped = [0, 1] if self.root_orientation_fixed else [0]
        self.clamped_indices = np.asarray(clamped, dtype=int)
        self.clamped_offsets = rest[self.clamped_indices] - rest[0]
        free_mask = np.ones(self.node_count, dtype=bool)
        free_mask[self.clamped_indices] = False
        self.free_indices = np.flatnonzero(free_mask)
        self.free_mask = free_mask

    def _build_constraints(self) -> None:
        inverse_mass = np.zeros(self.node_count)
        positive = self.free_mask & (self.mass > 0.0)
        inverse_mass[positive] = 1.0 / self.mass[positive]
        self.inverse_mass = inverse_mass

        # only edges with at least one movable endpoint carry a multiplier
        movable = inverse_mass > 0.0
        active = movable[:-1] | movable[1:]
        first = int(np.flatnonzero(active)[0])
        self.constraint_start = first
        self.constraint_count = self.edge_count - first
        self.constraint_rest = self.rest_edge_length[first:].copy()
        self._band = np.zeros((3, self.constraint_count))

    def _choose_timestep(self) -> None:
        """Explicit stability bound from the stiffest and most viscous node."""
        stiffness = np.zeros(self.node_count)
        damping = np.zeros(self.node_count)
        for offset, k_array, c_array in (
            (2, self.k_bend, self.c_bend),
            (3, self.k_twist, self.c_twist),
        ):
            if k_array.size == 0:
                continue
            low = np.arange(k_array.size)
            high = low + offset
            np.add.at(stiffness, low, k_array)
            np.add.at(stiffness, high, k_array)
            np.add.at(damping, low, c_array)
            np.add.at(damping, high, c_array)
        damping = damping + self.drag_coefficient

        mass = np.where(self.mass > 0.0, self.mass, np.inf)
        # Gershgorin bound for pairwise potentials: lambda_max <= 2*sum(k)
        omega_max = np.sqrt(2.0 * float(np.max(2.0 * stiffness / mass)))
        dt_elastic = 2.0 / max(omega_max, 1e-30)
        dt_viscous = 2.0 / max(float(np.max(damping / mass)), 1e-30)
        dt_target = STABILITY_SAFETY * min(dt_elastic, dt_viscous)

        substeps = int(np.ceil(self.sample_dt / max(dt_target, 1e-12)))
        self.substeps = int(min(max(substeps, MIN_SUBSTEPS), MAX_SUBSTEPS))
        self.dt = self.sample_dt / self.substeps
        self.dt_elastic_limit = float(dt_elastic)
        self.dt_viscous_limit = float(dt_viscous)

    def _init_budget(self) -> None:
        self.cumulative_impulse = np.zeros(3)
        self.cumulative_inflow = np.zeros(3)
        self.cumulative_outflow = np.zeros(3)
        self.cumulative_mass_source = np.zeros(1)
        self.total_mass = float(self.mass.sum())
        drive_speed = self.drive_amplitude * self.drive_omega
        fall_speed = float(
            np.sqrt(max(float(np.linalg.norm(self.gravity)) * self.material_length, 1e-12))
        )
        self.reference_momentum = float(
            max(self.total_mass * max(drive_speed, fall_speed), 1e-12)
        )
        self.constraint_scale = float(np.mean(self.rest_edge_length))

    # ------------------------------------------------------------------
    # kinematic root
    # ------------------------------------------------------------------
    def _root_state(self, time):
        position = self.root_rest.copy()
        velocity = np.zeros(3)
        acceleration = np.zeros(3)
        if time >= self.drive_start:
            phase = self.drive_omega * (time - self.drive_start) + self.drive_phase
            amplitude = self.drive_amplitude
            position = position + self.drive_axis * (amplitude * np.sin(phase))
            velocity = self.drive_axis * (amplitude * self.drive_omega * np.cos(phase))
            acceleration = self.drive_axis * (
                -amplitude * self.drive_omega**2 * np.sin(phase)
            )
        return position, velocity, acceleration

    # ------------------------------------------------------------------
    # forces
    # ------------------------------------------------------------------
    def _internal_forces(self, position, velocity):
        """Elastic and Kelvin-Voigt forces relative to the helical rest shape."""
        force = np.zeros_like(position)

        # second-neighbour (curvature) potential
        delta = position[2:] - position[:-2]
        distance = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(distance, 1e-14)[:, None]
        rate = np.sum((velocity[2:] - velocity[:-2]) * direction, axis=1)
        magnitude = self.k_bend * (distance - self.rest_d13) + self.c_bend * rate
        impulse = magnitude[:, None] * direction
        force[:-2] += impulse
        force[2:] -= impulse

        # third-neighbour (torsion) potential
        if self.rest_d14.size:
            delta = position[3:] - position[:-3]
            distance = np.linalg.norm(delta, axis=1)
            direction = delta / np.maximum(distance, 1e-14)[:, None]
            rate = np.sum((velocity[3:] - velocity[:-3]) * direction, axis=1)
            magnitude = self.k_twist * (distance - self.rest_d14) + self.c_twist * rate
            impulse = magnitude[:, None] * direction
            force[:-3] += impulse
            force[3:] -= impulse
        return force

    def _external_forces(self, velocity):
        gravity = self.mass[:, None] * self.gravity[None, :]
        drag = -self.drag_coefficient[:, None] * velocity
        return gravity, drag

    # ------------------------------------------------------------------
    # inextensibility projection
    # ------------------------------------------------------------------
    def _project(self, position):
        start = self.constraint_start
        inverse_mass = self.inverse_mass
        band = self._band
        low = inverse_mass[start:-1]
        high = inverse_mass[start + 1 :]
        diagonal_base = low + high
        error = 0.0
        for _ in range(PROJECTION_PASSES):
            delta = position[start + 1 :] - position[start:-1]
            length = np.linalg.norm(delta, axis=1)
            direction = delta / np.maximum(length, 1e-14)[:, None]
            residual = length - self.constraint_rest
            error = float(np.max(np.abs(residual)))
            if error < PROJECTION_TOLERANCE:
                break
            coupling = -high[:-1] * np.sum(direction[:-1] * direction[1:], axis=1)
            band[0, 0] = 0.0
            band[0, 1:] = coupling
            band[1, :] = diagonal_base
            band[2, :-1] = coupling
            band[2, -1] = 0.0
            multiplier = solve_banded(
                (1, 1), band, residual, overwrite_ab=False, check_finite=False
            )
            shift = multiplier[:, None] * direction
            position[start:-1] += low[:, None] * shift
            position[start + 1 :] -= high[:, None] * shift
        return error

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _step(self) -> None:
        dt = self.dt
        position = self.position
        velocity = self.velocity
        free = self.free_indices
        clamped = self.clamped_indices

        internal = self._internal_forces(position, velocity)
        gravity, drag = self._external_forces(velocity)
        total = internal + gravity + drag

        previous_position = position
        previous_velocity = velocity

        predicted = velocity.copy()
        predicted[free] += (dt / self.mass[free, None]) * total[free]

        candidate = position + dt * predicted
        root_position, _, _ = self._root_state((self.step_index + 1) * dt)
        candidate[clamped] = root_position + self.clamped_offsets

        error = self._project(candidate)
        if error > self.max_constraint_error:
            self.max_constraint_error = error

        new_velocity = (candidate - previous_position) / dt

        # Momentum bookkeeping: internal elastic/damping pairs and constraint
        # impulses between free nodes cancel, so the only external actions on
        # the strand are gravity, ambient drag and the kinematic root clamp.
        momentum_change = (self.mass[:, None] * (new_velocity - previous_velocity)).sum(
            axis=0
        )
        body_impulse = dt * (gravity + drag).sum(axis=0)
        root_impulse = momentum_change - body_impulse

        self.position = candidate
        self.velocity = new_velocity
        self.cumulative_impulse = self.cumulative_impulse + momentum_change
        self.root_force = root_impulse / dt
        self.step_index += 1
        self.time = self.step_index * dt

    def _probe_root_force(self):
        """One throw-away step exposing the clamp reaction at t = 0."""
        saved = (
            self.position.copy(),
            self.velocity.copy(),
            self.cumulative_impulse.copy(),
            self.step_index,
            self.time,
            self.max_constraint_error,
        )
        self.root_force = np.zeros(3)
        self._step()
        force = np.asarray(self.root_force, dtype=float).copy()
        (
            self.position,
            self.velocity,
            self.cumulative_impulse,
            self.step_index,
            self.time,
            self.max_constraint_error,
        ) = saved
        return force

    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.dt))
        while self.step_index < target:
            self._step()
        return self._frame(float(time))

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        position = self.position
        velocity = self.velocity
        gravity, drag = self._external_forces(velocity)
        root_force = np.asarray(self.root_force, dtype=float)

        edge_length = np.linalg.norm(np.diff(position, axis=0), axis=1)
        node_index = np.arange(self.node_count)

        # single global domain: the physical guide strand
        domain_weights = np.ones((1, self.node_count))
        external_force = np.stack(
            (gravity.sum(axis=0), drag.sum(axis=0), root_force)
        )[None, :, :]

        # per-node external force: body terms everywhere, clamp reaction at the root
        nodal_external = gravity + drag
        nodal_external[0] = nodal_external[0] + root_force

        attributes = {
            "node_index": node_index.astype(float),
            "is_root": (node_index == 0).astype(float),
            "is_free_tip": (node_index == self.node_count - 1).astype(float),
            "kinematic_root": (~self.free_mask).astype(float),
            "mass_quantity": self.mass.copy(),
            "lumped_length": self.lumped_length.copy(),
        }
        fields = {
            "constraint_value": edge_length,
            "constraint_target": self.rest_edge_length.copy(),
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "budget_domain_weights": domain_weights,
            "budget_external_force": external_force,
            "budget_cumulative_external_impulse": self.cumulative_impulse.copy()[None, :],
            "budget_cumulative_inflow_momentum": self.cumulative_inflow.copy()[None, :],
            "budget_cumulative_outflow_momentum": self.cumulative_outflow.copy()[None, :],
            "external_force": nodal_external,
        }
        scalars = {
            "constraint_scale": self.constraint_scale,
            "reference_momentum": self.reference_momentum,
            "material_length": float(edge_length.sum()),
            "rest_material_length": self.rest_total_length,
            "internal_damping_gamma": self.internal_gamma,
            "ambient_drag_nu": self.ambient_nu,
            "bending_rigidity": self.bending_rigidity,
            "torsion_rigidity": self.torsion_rigidity,
            "substep_dt": self.dt,
            "root_height": float(position[0, 2]),
            "tip_height": float(position[-1, 2]),
        }
        frame = StateFrame(
            time=float(time),
            position=position.copy(),
            velocity=velocity.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        momentum = (self.mass[:, None] * self.velocity).sum(axis=0)
        residual = float(
            np.linalg.norm(momentum - self.cumulative_impulse) / self.reference_momentum
        )
        edge_length = np.linalg.norm(np.diff(self.position, axis=0), axis=1)
        return {
            "substeps_per_sample": int(self.substeps),
            "substep_dt": float(self.dt),
            "dt_elastic_limit": float(self.dt_elastic_limit),
            "dt_viscous_limit": float(self.dt_viscous_limit),
            "steps_taken": int(self.step_index),
            "max_constraint_error_m": float(self.max_constraint_error),
            "final_material_length_m": float(edge_length.sum()),
            "rest_material_length_m": float(self.rest_total_length),
            "momentum_budget_residual": residual,
            "bending_rigidity_Nm2": float(self.bending_rigidity),
            "torsion_rigidity_Nm2": float(self.torsion_rigidity),
            "max_bend_stiffness_N_per_m": float(np.max(self.k_bend)),
            "max_twist_stiffness_N_per_m": float(np.max(self.k_twist)),
        }
