"""Deterministic discrete elastic-rod solver for a pre-twisted closed ring.

The centreline is represented by periodic lumped masses.  Its mechanical
energy contains axial stretch, isotropic bending, and a reduced Michell-mode
part of the twist energy.  The latter loses positive definiteness when the
stored twist exceeds the configured critical twist, which lets the supplied
small out-of-plane perturbation grow dynamically.  A quartic term represents
the leading geometric nonlinearity and bounds the post-buckling branch.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Periodic, momentum-conserving elastic-rod time stepper."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # The model is deterministic; the scaffold supplies the seed mode.
        self.scene = scene
        self.config = scene.config
        rod = self.config["rod"]
        initial = self.config["initial_conditions"]

        source = scene.initial_state()
        self.x = np.asarray(source.position, dtype=float).copy()
        self.v = np.asarray(source.velocity, dtype=float).copy()
        self.mass = np.asarray(source.mass, dtype=float).copy()
        self.n = len(self.x)
        self.time = 0.0

        self.radius = float(rod["radius"])
        self.rod_radius = float(rod["cross_section_radius"])
        self.rest_length = float(rod["length"]) / self.n
        self.total_mass = float(np.sum(self.mass))
        self.phase = 2.0 * np.pi * np.arange(self.n, dtype=float) / self.n
        self.mode = int(initial["out_of_plane_mode"])
        self.mode_cos = np.cos(self.mode * self.phase)
        self.mode_sin = np.sin(self.mode * self.phase)

        # SI-valued reduced rod parameters.  Alpha and beta retain their
        # configured roles as bending and twisting stiffness multipliers.
        alpha = float(rod["bending_stiffness_alpha"])
        beta = float(rod["twisting_stiffness_beta"])
        self.bending_stiffness = 0.35 * alpha
        self.stretch_stiffness = 1.2e4 * max(alpha, 0.1)
        self.stretch_damping = 8.0
        self.internal_damping_rate = 0.55

        length = float(rod["length"])
        twist_density = float(initial["stored_total_twist_rad"]) / length
        critical_density = float(initial["critical_twist_rad"]) / length
        twist_ratio_squared = (twist_density / critical_density) ** 2
        # Match the reduced twist destabilization to this discretization's
        # mode-m bending eigenvalue.  Thus total modal stiffness is exactly
        # neutral at the configured critical twist, independent of resolution.
        laplacian_eigenvalue = 4.0 * np.sin(np.pi * self.mode / self.n) ** 2
        discrete_modal_bending = (
            self.bending_stiffness
            * laplacian_eigenvalue**2
            / self.rest_length**3
        )
        self.michell_stiffness = (
            0.5 * self.n * beta * discrete_modal_bending * twist_ratio_squared
        )
        self.postbuckling_amplitude = 0.42 * self.radius
        self.twist_density = twist_density
        self.material_angle = int(initial["material_frame_winding_number"]) * self.phase

        # Explicit symplectic Euler remains stable below the fastest discrete
        # bending/stretch mode at this spatial resolution.
        self.max_step = min(1.0e-3, float(self.config["sample_dt"]) / 10.0)

        # Non-neighbour pairs for circular-tube self contact.  Adjacent and
        # next-adjacent samples belong to the same local tube neighbourhood.
        ii, jj = np.triu_indices(self.n, k=1)
        separation = np.minimum(jj - ii, self.n - (jj - ii))
        keep = separation >= 3
        self.contact_i = ii[keep]
        self.contact_j = jj[keep]
        self.contact_distance = 2.02 * self.rod_radius
        self.contact_stiffness = 4.0e3
        self.contact_damping = 12.0

        self.step_count = 0
        self.maximum_height = float(np.max(self.x[:, 2]) - np.min(self.x[:, 2]))

    def _rigid_velocity(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Best fitting translation plus rotation of the current samples."""
        center = np.sum(self.mass[:, None] * x, axis=0) / self.total_mass
        v_center = np.sum(self.mass[:, None] * v, axis=0) / self.total_mass
        r = x - center
        relative_v = v - v_center
        inertia = np.zeros((3, 3), dtype=float)
        eye = np.eye(3)
        for k in range(self.n):
            inertia += self.mass[k] * (
                np.dot(r[k], r[k]) * eye - np.outer(r[k], r[k])
            )
        angular_momentum = np.sum(
            np.cross(r, self.mass[:, None] * relative_v), axis=0
        )
        omega = np.linalg.solve(inertia + 1.0e-12 * eye, angular_momentum)
        return v_center + np.cross(omega[None, :], r)

    def _remove_rigid_components(self, force: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Project internal forces to exactly zero resultant and zero torque."""
        center = np.sum(self.mass[:, None] * x, axis=0) / self.total_mass
        r = x - center
        force = force - self.mass[:, None] * np.sum(force, axis=0) / self.total_mass

        eye = np.eye(3)
        inertia = np.zeros((3, 3), dtype=float)
        for k in range(self.n):
            inertia += self.mass[k] * (
                np.dot(r[k], r[k]) * eye - np.outer(r[k], r[k])
            )
        torque = np.sum(np.cross(r, force), axis=0)
        angular_acceleration = np.linalg.solve(inertia + 1.0e-12 * eye, torque)
        force -= self.mass[:, None] * np.cross(angular_acceleration[None, :], r)
        return force

    def _forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        force = np.zeros_like(x)

        # Periodic axial Hooke/Kelvin forces enforce near-inextensibility.
        edge = np.roll(x, -1, axis=0) - x
        edge_length = np.linalg.norm(edge, axis=1)
        tangent = edge / np.maximum(edge_length[:, None], 1.0e-12)
        relative_velocity = np.roll(v, -1, axis=0) - v
        extension_rate = np.sum(relative_velocity * tangent, axis=1)
        edge_force = (
            self.stretch_stiffness * (edge_length - self.rest_length)
            + self.stretch_damping * extension_rate
        )[:, None] * tangent
        force += edge_force - np.roll(edge_force, 1, axis=0)

        # Gradient of B/(2 ds^3) sum |x[i+1]-2x[i]+x[i-1]|^2.
        laplacian = np.roll(x, -1, axis=0) - 2.0 * x + np.roll(x, 1, axis=0)
        biharmonic = (
            np.roll(laplacian, -1, axis=0)
            - 2.0 * laplacian
            + np.roll(laplacian, 1, axis=0)
        )
        force -= self.bending_stiffness * biharmonic / self.rest_length**3

        # Reduced twist/writhe coupling.  Above the Michell threshold the
        # quadratic coefficient is negative; the quartic term is the physical
        # leading-order post-buckling stabilization.
        z = x[:, 2] - np.sum(self.mass * x[:, 2]) / self.total_mass
        amplitude = np.array(
            [
                2.0 * np.mean(z * self.mode_cos),
                2.0 * np.mean(z * self.mode_sin),
            ]
        )
        amplitude_squared = float(np.dot(amplitude, amplitude))
        modal_factor = self.michell_stiffness * (
            1.0 - amplitude_squared / self.postbuckling_amplitude**2
        )
        modal_shape = amplitude[0] * self.mode_cos + amplitude[1] * self.mode_sin
        force[:, 2] += (2.0 / self.n) * modal_factor * modal_shape

        # Internal Rayleigh damping acts only on deformation velocity, leaving
        # exact rigid translation and rotation untouched.
        rigid_velocity = self._rigid_velocity(x, v)
        force -= (
            self.internal_damping_rate
            * self.mass[:, None]
            * (v - rigid_velocity)
        )

        # Penalty contact between nonlocal tube samples.
        delta = x[self.contact_j] - x[self.contact_i]
        distance = np.linalg.norm(delta, axis=1)
        active = distance < self.contact_distance
        if np.any(active):
            ia = self.contact_i[active]
            ja = self.contact_j[active]
            normal = delta[active] / np.maximum(distance[active, None], 1.0e-12)
            normal_speed = np.sum((v[ja] - v[ia]) * normal, axis=1)
            magnitude = (
                self.contact_stiffness * (self.contact_distance - distance[active])
                - self.contact_damping * np.minimum(normal_speed, 0.0)
            )
            pair_force = magnitude[:, None] * normal
            np.add.at(force, ia, -pair_force)
            np.add.at(force, ja, pair_force)

        # All actions above are internal to an unsupported closed ring.
        return self._remove_rigid_components(force, x)

    def _step(self, dt: float) -> None:
        force = self._forces(self.x, self.v)
        self.v += dt * force / self.mass[:, None]
        self.x += dt * self.v
        self.time += dt
        self.step_count += 1

        if not np.isfinite(self.x).all() or not np.isfinite(self.v).all():
            raise FloatingPointError("elastic-ring integration became non-finite")
        height = float(np.max(self.x[:, 2]) - np.min(self.x[:, 2]))
        self.maximum_height = max(self.maximum_height, height)

    def _angular_velocity(self) -> np.ndarray:
        chord = np.roll(self.x, -1, axis=0) - np.roll(self.x, 1, axis=0)
        tangent = chord / np.maximum(np.linalg.norm(chord, axis=1)[:, None], 1.0e-12)
        chord_rate = np.roll(self.v, -1, axis=0) - np.roll(self.v, 1, axis=0)
        tangent_rate = (
            chord_rate
            - tangent * np.sum(tangent * chord_rate, axis=1)[:, None]
        ) / np.maximum(np.linalg.norm(chord, axis=1)[:, None], 1.0e-12)
        return np.cross(tangent, tangent_rate)

    def _frame(self, requested_time: float) -> StateFrame:
        edge = np.roll(self.x, -1, axis=0) - self.x
        edge_length = np.linalg.norm(edge, axis=1)
        zeros_g3 = np.zeros((1, 3), dtype=float)
        zeros_gk3 = np.zeros((1, 1, 3), dtype=float)
        inertia_tensor = np.zeros((self.n, 3, 3), dtype=float)

        attributes = {
            "arc_length": self.rest_length * np.arange(self.n, dtype=float),
            "closed_curve_index": np.arange(self.n, dtype=float),
            "natural_curvature": np.zeros(self.n, dtype=float),
            "natural_twist": np.zeros(self.n, dtype=float),
            "material_frame_angle": self.material_angle.copy(),
            "stored_twist_density": np.full(self.n, self.twist_density, dtype=float),
            "inertia_tensor": inertia_tensor,
            "mass_quantity": self.mass.copy(),
        }
        fields = {
            "budget_cumulative_external_angular_impulse": zeros_g3.copy(),
            "budget_cumulative_external_impulse": zeros_g3.copy(),
            "budget_cumulative_inflow_angular_momentum": zeros_g3.copy(),
            "budget_cumulative_inflow_momentum": zeros_g3.copy(),
            "budget_cumulative_outflow_angular_momentum": zeros_g3.copy(),
            "budget_cumulative_outflow_momentum": zeros_g3.copy(),
            "budget_domain_weights": np.ones((1, self.n), dtype=float),
            "budget_external_couple": zeros_gk3.copy(),
            "budget_external_force": zeros_gk3.copy(),
            "budget_force_position": zeros_gk3.copy(),
            "constraint_value": edge_length.copy(),
            "constraint_target": np.full(self.n, self.rest_length, dtype=float),
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }
        speed_scale = np.sqrt(
            self.bending_stiffness / max(self.total_mass * self.radius, 1.0e-12)
        )
        scalars = {
            "rod_radius": self.rod_radius,
            "stored_total_twist_rad": self.twist_density * self.n * self.rest_length,
            "material_frame_winding_number": float(
                self.config["initial_conditions"]["material_frame_winding_number"]
            ),
            "closed_ring": 1.0,
            "constraint_scale": self.rest_length,
            "reference_momentum": max(self.total_mass * speed_scale, 1.0e-9),
            "reference_angular_momentum": max(
                self.total_mass * self.radius * speed_scale, 1.0e-9
            ),
        }
        frame = StateFrame(
            time=float(requested_time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            angular_velocity=self._angular_velocity(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be finite and nondecreasing")
        while self.time < target - 1.0e-12:
            dt = min(self.max_step, target - self.time)
            self._step(dt)
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        edge_length = np.linalg.norm(np.roll(self.x, -1, axis=0) - self.x, axis=1)
        return {
            "model": "periodic_discrete_elastic_rod_with_michell_twist_energy",
            "integration_steps": int(self.step_count),
            "maximum_peak_to_peak_height": float(self.maximum_height),
            "maximum_relative_edge_error": float(
                np.max(np.abs(edge_length - self.rest_length)) / self.rest_length
            ),
        }
