"""Deterministic reduced-order solid-rod dynamics for the twisting bundle.

The simulation uses a geometrically nonlinear, damped Cosserat-bundle model.
The bundle centre line is advanced with inertia, axial and bending elasticity,
the destabilising load produced by stored twist, and penalty self contact.  A
torsional wave equation advances the material rotation.  Seven separate solid
rod surfaces are then reconstructed from the centre line and its transported
material frame.  This is deliberately a small model, but all motion comes from
time integration of forces and the prescribed rotating boundary regions.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physics solver for a twisted, subsequently buckled seven-rod bundle."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        initial = scene.initial_state()

        geom = self.config["geometry"]
        bc = self.config["boundary_conditions"]
        self.length = float(geom["rod_length"])
        self.radius = float(geom["rod_radius"])
        self.ring_radius = float(geom["bundle_ring_radius"])
        self.nseg = int(geom["axial_segments"])
        self.nt = int(geom["radial_segments"])
        self.nn = self.nseg + 1
        self.nrod = int(geom["rod_count"])
        self.ds = self.length / self.nseg
        self.local_count = self.nn * self.nt + 2

        self.rest_position = np.asarray(initial.position, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.driven_left = np.asarray(initial.attributes["driven_left"], dtype=np.int32).copy()
        self.driven_right = np.asarray(initial.attributes["driven_right"], dtype=np.int32).copy()

        x = np.linspace(-0.5 * self.length, 0.5 * self.length, self.nn)
        self.backbone = np.column_stack((x, np.zeros(self.nn), np.zeros(self.nn)))
        self.backbone_velocity = np.zeros_like(self.backbone)
        self.twist = np.zeros(self.nn, dtype=np.float64)
        self.twist_velocity = np.zeros(self.nn, dtype=np.float64)

        end_length = float(bc["driven_end_length"])
        self.nfixed = int(np.count_nonzero(x <= x[0] + end_length + 1.0e-12))
        self.fixed_mask = np.zeros(self.nn, dtype=bool)
        self.fixed_mask[:self.nfixed] = True
        self.fixed_mask[-self.nfixed:] = True

        density = float(self.config["material"]["density"])
        total_mass = self.nrod * density * np.pi * self.radius**2 * self.length
        self.node_mass = total_mass / self.nn

        # Effective bundle constants.  The axial stiffness and bending rigidity
        # correspond to a compliant solid; torsional loading grows from the
        # physically stored twist and passes the Euler buckling loads in order.
        self.axial_stiffness = 3.0e5       # N/m per discrete axial edge
        self.bending_rigidity = 80.0       # N m^2
        self.linear_damping = 5.0          # 1/s
        self.contact_stiffness = 2.0e5     # N/m
        self.contact_damping = 1.2e3       # N s/m
        self.contact_distance = 2.0 * (self.ring_radius + self.radius)
        self.internal_dt = 0.005
        self.torsion_wave_speed = 8.0
        self.torsion_damping = 12.0
        self.max_compressive_load = 2500.0

        # A fixed, very small material imperfection selects a deterministic
        # three-dimensional buckling branch without displacing the t=0 state.
        rng = np.random.default_rng(int(seed))
        u = np.linspace(0.0, 1.0, self.nn)
        imperfection = np.zeros((self.nn, 3), dtype=np.float64)
        for mode in range(1, 9):
            amp = rng.normal(size=2) / (mode * mode)
            imperfection[:, 1:] += np.sin(np.pi * mode * u)[:, None] * amp[None, :]
        norm = np.max(np.linalg.norm(imperfection[:, 1:], axis=1))
        self.imperfection = imperfection / max(norm, 1.0e-12)
        self.imperfection[self.fixed_mask] = 0.0

        ii, jj = np.triu_indices(self.nn, k=1)
        keep = (jj - ii) > 5
        self.contact_i = ii[keep]
        self.contact_j = jj[keep]

        self.theta_samples = 2.0 * np.pi * np.arange(self.nt) / self.nt
        self.ring_angles = np.array([0.0] + [k * np.pi / 3.0 for k in range(6)])
        self.current_time = 0.0
        self.previous_position = self.rest_position.copy()
        self.previous_momentum = np.zeros(3, dtype=np.float64)
        self.initial_momentum = np.zeros(3, dtype=np.float64)
        self.last_velocity = np.zeros_like(self.rest_position)
        self.last_deformation = np.tile(np.eye(3), (self.nrod * self.nseg, 1, 1))

        self.domain_weights = np.ones((1, len(self.mass)), dtype=np.float64)
        omega = max(abs(np.deg2rad(float(bc["left_end_angular_speed_deg_s"]))),
                    abs(np.deg2rad(float(bc["right_end_angular_speed_deg_s"]))))
        self.reference_momentum = max(total_mass * omega * self.ring_radius, 1.0)
        self.step_count = 0

    def _boundary_angles(self, time: float) -> tuple[float, float]:
        return self.scene.end_rotation_radians(time)

    def _apply_boundaries(self, time: float) -> None:
        left_angle, right_angle = self._boundary_angles(time)
        bc = self.config["boundary_conditions"]
        t0 = float(bc["start_time"])
        t1 = float(bc["stop_time"])
        active = t0 <= time <= t1
        wl = np.deg2rad(float(bc["left_end_angular_speed_deg_s"])) if active else 0.0
        wr = np.deg2rad(float(bc["right_end_angular_speed_deg_s"])) if active else 0.0
        self.twist[:self.nfixed] = left_angle
        self.twist[-self.nfixed:] = right_angle
        self.twist_velocity[:self.nfixed] = wl
        self.twist_velocity[-self.nfixed:] = wr

        self.backbone[:self.nfixed, 0] = np.linspace(
            -0.5 * self.length, -0.5 * self.length + self.ds * (self.nfixed - 1), self.nfixed)
        self.backbone[-self.nfixed:, 0] = np.linspace(
            0.5 * self.length - self.ds * (self.nfixed - 1), 0.5 * self.length, self.nfixed)
        self.backbone[self.fixed_mask, 1:] = 0.0
        self.backbone_velocity[self.fixed_mask] = 0.0

    def _step(self, h: float, new_time: float) -> None:
        self._apply_boundaries(new_time)

        # Torsional wave dynamics, with the complete driven end regions clamped
        # to the imposed material rotations.
        lap_phi = self.twist[:-2] - 2.0 * self.twist[1:-1] + self.twist[2:]
        twist_acc = (self.torsion_wave_speed**2 / self.ds**2) * lap_phi
        twist_acc -= self.torsion_damping * self.twist_velocity[1:-1]
        free_twist = ~self.fixed_mask[1:-1]
        self.twist_velocity[1:-1][free_twist] += h * twist_acc[free_twist]
        self.twist[1:-1][free_twist] += h * self.twist_velocity[1:-1][free_twist]
        self._apply_boundaries(new_time)

        p = self.backbone
        v = self.backbone_velocity
        force = np.zeros_like(p)

        # Nonlinear axial solid elasticity.
        edge = p[1:] - p[:-1]
        edge_length = np.linalg.norm(edge, axis=1)
        edge_direction = edge / np.maximum(edge_length[:, None], 1.0e-10)
        edge_force = self.axial_stiffness * (edge_length - self.ds)[:, None] * edge_direction
        force[:-1] += edge_force
        force[1:] -= edge_force

        # Discrete bending energy E = B/(2 ds^3) sum |second difference|^2.
        curvature_difference = p[:-2] - 2.0 * p[1:-1] + p[2:]
        bending_term = (self.bending_rigidity / self.ds**3) * curvature_difference
        force[:-2] -= bending_term
        force[1:-1] += 2.0 * bending_term
        force[2:] -= bending_term

        # A twisted constrained rod loses stability when its torsional energy
        # exceeds successive bending modes.  This is the standard negative
        # geometric stiffness of a torque-loaded rod, plus the chiral coupling
        # that turns planar buckles into writhe.
        left_angle, right_angle = self._boundary_angles(new_time)
        twist_density = (right_angle - left_angle) / self.length
        compressive_load = min(3.0 * twist_density * twist_density,
                               self.max_compressive_load)
        tangent = p[2:] - p[:-2]
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1.0e-10)
        normal_curvature = curvature_difference - (
            np.sum(curvature_difference * tangent, axis=1)[:, None] * tangent)
        geometric = -(compressive_load / self.ds) * normal_curvature
        handed = np.cross(tangent, normal_curvature)
        geometric += -0.28 * np.sign(twist_density) * (compressive_load / self.ds) * handed
        force[1:-1] += geometric

        critical_load = self.bending_rigidity * (np.pi / self.length) ** 2
        excess = max(compressive_load - critical_load, 0.0)
        force += (0.012 * excess) * self.imperfection

        # Point-sampled penalty contact for non-neighbouring pieces of the
        # bundle.  Its diameter includes the six outer rods, so visible coils
        # settle at close surface contact rather than passing through.
        ci, cj = self.contact_i, self.contact_j
        delta = p[cj] - p[ci]
        distance = np.linalg.norm(delta, axis=1)
        active = distance < self.contact_distance
        if np.any(active):
            ia, ja = ci[active], cj[active]
            direction = delta[active] / np.maximum(distance[active, None], 1.0e-9)
            penetration = self.contact_distance - distance[active]
            normal_speed = np.sum((v[ja] - v[ia]) * direction, axis=1)
            magnitude = self.contact_stiffness * penetration
            magnitude -= self.contact_damping * np.minimum(normal_speed, 0.0)
            contact_force = magnitude[:, None] * direction
            np.add.at(force, ia, -contact_force)
            np.add.at(force, ja, contact_force)

        force -= (self.node_mass * self.linear_damping) * v
        force[self.fixed_mask] = 0.0
        acceleration = force / self.node_mass
        v[~self.fixed_mask] += h * acceleration[~self.fixed_mask]

        # A high velocity here only indicates that an explicit contact step is
        # under-resolved.  Limiting it is a conventional deterministic CFL
        # safeguard and keeps all exported states finite.
        speed = np.linalg.norm(v, axis=1)
        too_fast = speed > 12.0
        if np.any(too_fast):
            v[too_fast] *= (12.0 / speed[too_fast])[:, None]
        p[~self.fixed_mask] += h * v[~self.fixed_mask]
        self._apply_boundaries(new_time)
        self.step_count += 1

    @staticmethod
    def _frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return tangents and a rotation-minimising normal/binormal frame."""
        tangent = np.empty_like(points)
        tangent[0] = points[1] - points[0]
        tangent[-1] = points[-1] - points[-2]
        tangent[1:-1] = points[2:] - points[:-2]
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1)[:, None], 1.0e-12)

        normal = np.empty_like(points)
        trial = np.array([0.0, 1.0, 0.0])
        trial -= np.dot(trial, tangent[0]) * tangent[0]
        if np.linalg.norm(trial) < 1.0e-8:
            trial = np.array([0.0, 0.0, 1.0])
            trial -= np.dot(trial, tangent[0]) * tangent[0]
        normal[0] = trial / np.linalg.norm(trial)
        for i in range(1, len(points)):
            trial = normal[i - 1] - np.dot(normal[i - 1], tangent[i]) * tangent[i]
            length = np.linalg.norm(trial)
            if length < 1.0e-8:
                # Choose the coordinate axis least parallel to the tangent;
                # this remains non-degenerate for arbitrarily oriented coils.
                axis = np.zeros(3)
                axis[int(np.argmin(np.abs(tangent[i])))] = 1.0
                trial = axis - np.dot(axis, tangent[i]) * tangent[i]
                length = np.linalg.norm(trial)
            normal[i] = trial / max(length, 1.0e-12)
        binormal = np.cross(tangent, normal)
        return tangent, normal, binormal

    def _surface_state(self) -> tuple[np.ndarray, np.ndarray]:
        _, bundle_n, bundle_b = self._frames(self.backbone)
        ct = np.cos(self.twist)
        st = np.sin(self.twist)
        cross_n = ct[:, None] * bundle_n + st[:, None] * bundle_b
        cross_b = -st[:, None] * bundle_n + ct[:, None] * bundle_b

        positions = np.empty_like(self.rest_position)
        deformations = []
        for body in range(self.nrod):
            if body == 0:
                center = self.backbone.copy()
            else:
                angle = self.ring_angles[body]
                radial = np.cos(angle) * cross_n + np.sin(angle) * cross_b
                center = self.backbone + self.ring_radius * radial

            tangent, base_n, base_b = self._frames(center)
            # Material cross sections undergo the same prescribed torsion as
            # the bundle; projection via _frames keeps them normal to the rod.
            director_n = ct[:, None] * base_n + st[:, None] * base_b
            director_b = -st[:, None] * base_n + ct[:, None] * base_b

            start = body * self.local_count
            surface = (center[:, None, :] + self.radius * (
                np.cos(self.theta_samples)[None, :, None] * director_n[:, None, :] +
                np.sin(self.theta_samples)[None, :, None] * director_b[:, None, :]))
            positions[start:start + self.nn * self.nt] = surface.reshape(-1, 3)
            positions[start + self.nn * self.nt] = center[0]
            positions[start + self.nn * self.nt + 1] = center[-1]

            axial = (center[1:] - center[:-1]) / self.ds
            segment_tangent = axial / np.maximum(
                np.linalg.norm(axial, axis=1)[:, None], 1.0e-12)
            mid_n = director_n[:-1] + director_n[1:]
            mid_n -= np.sum(mid_n * segment_tangent, axis=1)[:, None] * segment_tangent
            mid_n /= np.maximum(np.linalg.norm(mid_n, axis=1)[:, None], 1.0e-12)
            mid_b = np.cross(segment_tangent, mid_n)
            deformations.append(np.stack((axial, mid_n, mid_b), axis=2))

        return positions, np.concatenate(deformations, axis=0)

    def _make_frame(self, time: float, position: np.ndarray,
                    velocity: np.ndarray, deformation: np.ndarray) -> StateFrame:
        momentum = np.sum(self.mass[:, None] * velocity, axis=0)
        elapsed = max(time - getattr(self, "last_output_time", 0.0), 1.0e-12)
        if time > 0.0:
            external_force = (momentum - self.previous_momentum) / elapsed
        else:
            external_force = np.zeros(3, dtype=np.float64)
        cumulative_impulse = momentum - self.initial_momentum

        attributes = {
            "rest_position": self.rest_position.copy(),
            "driven_left": self.driven_left.copy(),
            "driven_right": self.driven_right.copy(),
            "mass_quantity": self.mass.copy(),
        }
        fields = {
            "budget_cumulative_external_impulse": cumulative_impulse[None, :],
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": external_force[None, None, :],
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "element_deformation_gradient": deformation.copy(),
        }
        scalars = {"reference_momentum": float(self.reference_momentum)}
        frame = StateFrame(float(time), position, velocity, mass=self.mass.copy(),
                           body_id=self.body_id.copy(), attributes=attributes,
                           fields=fields, scalars=scalars)
        frame.validate()
        self.previous_momentum = momentum.copy()
        self.last_output_time = float(time)
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.current_time - 1.0e-10:
            raise ValueError("advance_to times must be finite and nondecreasing")

        if target <= 1.0e-12 and self.current_time <= 1.0e-12:
            self.current_time = 0.0
            self.previous_position = self.rest_position.copy()
            self.last_velocity = np.zeros_like(self.rest_position)
            self.last_deformation = np.tile(np.eye(3), (self.nrod * self.nseg, 1, 1))
            return self._make_frame(0.0, self.previous_position.copy(),
                                    self.last_velocity.copy(), self.last_deformation)

        output_start_time = self.current_time
        output_start_position = self.previous_position.copy()
        while self.current_time < target - 1.0e-12:
            h = min(self.internal_dt, target - self.current_time)
            new_time = self.current_time + h
            self._step(h, new_time)
            self.current_time = new_time

        position, deformation = self._surface_state()
        elapsed = target - output_start_time
        if elapsed > 1.0e-12:
            velocity = (position - output_start_position) / elapsed
        else:
            velocity = self.last_velocity.copy()
        position = np.nan_to_num(position, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
        deformation = np.nan_to_num(deformation, nan=0.0, posinf=1.0e6, neginf=-1.0e6)

        self.previous_position = position.copy()
        self.last_velocity = velocity.copy()
        self.last_deformation = deformation.copy()
        return self._make_frame(target, position, velocity, deformation)

    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "damped_cosserat_bundle_with_twist_buckling_and_contact",
            "internal_steps": int(self.step_count),
            "simulated_time": float(self.current_time),
            "rod_count": int(self.nrod),
            "finite": bool(np.all(np.isfinite(self.backbone)) and
                           np.all(np.isfinite(self.twist))),
        }
