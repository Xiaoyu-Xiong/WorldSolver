"""Self-contained discrete elastic rod solver for the V-shaped ribbon.

The centerline is advanced with lumped masses, anisotropic bending forces and
SHAKE-style inextensibility projections. Material twist is a second-order
torsional rod discretization integrated by backward Euler. The naturally
sharp bend is represented by a strain-stiffening torsional hinge: at small
loads it is the compliant part of the rod and localizes rotation on the driven
arm, while its tangent stiffness increases enough for later twist to pass onto
the fixed arm.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length > 1.0e-14:
        return vector / length
    return fallback.copy()


def _rotate_about_axis(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation, used for material directors and natural curvature."""
    c = math.cos(angle)
    s = math.sin(angle)
    return c * vector + s * np.cross(axis, vector) + (1.0 - c) * axis * np.dot(axis, vector)


class BenchmarkSolver:
    """Damped dynamic relaxation of an inextensible anisotropic elastic rod."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.config = scene.config
        initial = scene.initial_state()

        self.x = np.asarray(initial.position, dtype=float).copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.reference = self.x.copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.n = len(self.x)
        self.apex = int(self.config["rod"]["apex_vertex"])
        self.rest_lengths = np.linalg.norm(np.diff(self.reference, axis=0), axis=1)
        self.rest_second_difference = (
            self.reference[:-2] - 2.0 * self.reference[1:-1] + self.reference[2:]
        )
        self.arc_length = np.r_[0.0, np.cumsum(self.rest_lengths)]

        rod = self.config["rod"]
        width = float(rod["cross_section_width"])
        thickness = float(rod["cross_section_thickness"])
        density = float(rod["density"])
        area = width * thickness

        # The two second moments retain the rectangular-section anisotropy.
        self.young_modulus = 2.0e5
        self.shear_modulus = 7.0e4
        i_weak = width * thickness**3 / 12.0
        i_strong = thickness * width**3 / 12.0
        mean_ds = float(np.mean(self.rest_lengths))
        self.bend_weak = self.young_modulus * i_weak / mean_ds**3
        self.bend_strong = self.young_modulus * i_strong / mean_ds**3
        self.linear_drag = 0.06

        # Saint-Venant torsion constant for a thin rectangular strip.
        torsion_constant = width * thickness**3 * (
            1.0 / 3.0 - 0.21 * (thickness / width) * (1.0 - thickness**4 / (12.0 * width**4))
        )
        normal_edge_stiffness = self.shear_modulus * torsion_constant / mean_ds
        self.normal_edge_twist_stiffness = normal_edge_stiffness
        self.twist_stiffness = np.full(self.n - 1, normal_edge_stiffness, dtype=float)
        # The corner's nonlinear tangent stiffness models flattening of the
        # bent anisotropic section under increasing rotation.
        self.apex_hinge_stiffness = max(2.5e-4, normal_edge_stiffness * 0.002)
        self.apex_hinge_limit = normal_edge_stiffness * 0.65
        self.twist_inertia = self.mass * (width * width + thickness * thickness) / 12.0
        self.twist_drag = np.full(self.n, 2.0e-4, dtype=float)
        self.theta = np.zeros(self.n, dtype=float)
        self.omega = np.zeros(self.n, dtype=float)

        bc = self.config["boundary_conditions"]
        self.driven_position = np.asarray(bc["driven_end"]["position"], dtype=float)
        self.fixed_position = np.asarray(bc["fixed_end"]["position"], dtype=float)
        schedule = np.asarray(bc["driven_end"]["rotation_schedule"], dtype=float)
        self.schedule_times = schedule[:, 0]
        self.schedule_angles = schedule[:, 1]

        self.current_time = 0.0
        self.internal_steps = 0
        self.last_external_force = np.zeros(3, dtype=float)
        self.last_boundary_couples = np.zeros((2, 3), dtype=float)
        self.area = area
        self.density = density
        self.width = width
        self.thickness = thickness

    def _drive(self, time: float) -> tuple[float, float]:
        angle = float(np.interp(time, self.schedule_times, self.schedule_angles))
        if time <= self.schedule_times[0] or time >= self.schedule_times[-1]:
            return angle, 0.0
        interval = int(np.searchsorted(self.schedule_times, time, side="right") - 1)
        dt = self.schedule_times[interval + 1] - self.schedule_times[interval]
        rate = (self.schedule_angles[interval + 1] - self.schedule_angles[interval]) / dt
        return angle, float(rate)

    def _edge_twist_stiffness(self) -> np.ndarray:
        stiffness = self.twist_stiffness.copy()
        jump = abs(float(self.theta[self.apex] - self.theta[self.apex + 1]))
        nonlinear = self.apex_hinge_stiffness * (1.0 + 10.0 * (jump / 3.0) ** 2)
        stiffness[self.apex] = min(nonlinear, self.apex_hinge_limit)
        return stiffness

    @staticmethod
    def _solve_tridiagonal(
        lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray
    ) -> np.ndarray:
        """Thomas solve for the positive definite implicit torsion system."""
        d = diagonal.copy()
        b = rhs.copy()
        u = upper.copy()
        for i in range(1, len(d)):
            factor = lower[i - 1] / d[i - 1]
            d[i] -= factor * u[i - 1]
            b[i] -= factor * b[i - 1]
        result = np.empty_like(b)
        result[-1] = b[-1] / d[-1]
        for i in range(len(d) - 2, -1, -1):
            result[i] = (b[i] - u[i] * result[i + 1]) / d[i]
        return result

    def _step_twist(self, dt: float, new_time: float) -> None:
        old_theta = self.theta.copy()
        stiffness = self._edge_twist_stiffness()
        drive_angle, drive_rate = self._drive(new_time)

        inertia = self.twist_inertia[1:-1]
        drag = self.twist_drag[1:-1]
        inertial_coefficient = inertia / dt**2 + drag / dt
        diagonal = inertial_coefficient + stiffness[:-1] + stiffness[1:]
        lower = -stiffness[1:-1].copy()
        upper = -stiffness[1:-1].copy()
        rhs = inertial_coefficient * self.theta[1:-1] + inertia * self.omega[1:-1] / dt
        rhs[0] += stiffness[0] * drive_angle
        new_internal = self._solve_tridiagonal(lower, diagonal, upper, rhs)

        self.theta[0] = drive_angle
        self.theta[1:-1] = new_internal
        self.theta[-1] = 0.0
        self.omega[1:-1] = (self.theta[1:-1] - old_theta[1:-1]) / dt
        self.omega[0] = drive_rate
        self.omega[-1] = 0.0

    def _bending_force(self) -> np.ndarray:
        force = np.zeros_like(self.x)
        reference_normal = np.array([0.0, 1.0, 0.0])
        fallback_tangent = np.array([1.0, 0.0, 0.0])
        for local, i in enumerate(range(1, self.n - 1)):
            tangent = _unit(self.x[i + 1] - self.x[i - 1], fallback_tangent)
            director_one = _rotate_about_axis(reference_normal, tangent, float(self.theta[i]))
            director_one -= tangent * np.dot(tangent, director_one)
            director_one = _unit(director_one, reference_normal)
            director_two = _unit(np.cross(tangent, director_one), np.array([0.0, 0.0, 1.0]))

            natural = _rotate_about_axis(
                self.rest_second_difference[local], tangent, float(self.theta[i])
            )
            residual = self.x[i - 1] - 2.0 * self.x[i] + self.x[i + 1] - natural
            weighted = (
                self.bend_weak * np.dot(residual, director_one) * director_one
                + self.bend_strong * np.dot(residual, director_two) * director_two
                + self.bend_weak * np.dot(residual, tangent) * tangent
            )
            force[i - 1] -= weighted
            force[i] += 2.0 * weighted
            force[i + 1] -= weighted
        return force

    def _project_lengths(self) -> None:
        inverse_mass = 1.0 / self.mass
        inverse_mass[0] = 0.0
        inverse_mass[-1] = 0.0
        for iteration in range(10):
            self.x[0] = self.driven_position
            self.x[-1] = self.fixed_position
            edge_order = (
                range(self.n - 1)
                if iteration % 2 == 0
                else range(self.n - 2, -1, -1)
            )
            for edge in edge_order:
                delta = self.x[edge + 1] - self.x[edge]
                length = float(np.linalg.norm(delta))
                if length <= 1.0e-14:
                    continue
                error = length - self.rest_lengths[edge]
                w_left = inverse_mass[edge]
                w_right = inverse_mass[edge + 1]
                total = w_left + w_right
                if total == 0.0:
                    continue
                correction = (error / length) * delta
                self.x[edge] += (w_left / total) * correction
                self.x[edge + 1] -= (w_right / total) * correction
        self.x[0] = self.driven_position
        self.x[-1] = self.fixed_position

    def _step_centerline(self, dt: float) -> None:
        old_x = self.x.copy()
        old_momentum = np.sum(self.mass[:, None] * self.v, axis=0)
        elastic_force = self._bending_force()
        acceleration = (elastic_force - self.linear_drag * self.v) / self.mass[:, None]
        acceleration[0] = 0.0
        acceleration[-1] = 0.0
        self.v += dt * acceleration
        self.x += dt * self.v
        self._project_lengths()
        self.v = (self.x - old_x) / dt
        self.v[0] = 0.0
        self.v[-1] = 0.0
        new_momentum = np.sum(self.mass[:, None] * self.v, axis=0)
        self.last_external_force = (new_momentum - old_momentum) / dt

    def _update_boundary_couples(self) -> None:
        stiffness = self._edge_twist_stiffness()
        driven_tangent = _unit(self.x[1] - self.x[0], np.array([1.0, 0.0, 0.0]))
        fixed_tangent = _unit(self.x[-1] - self.x[-2], np.array([1.0, 0.0, 0.0]))
        driven_scalar = stiffness[0] * (self.theta[0] - self.theta[1])
        fixed_scalar = stiffness[-1] * (self.theta[-1] - self.theta[-2])
        self.last_boundary_couples[0] = driven_scalar * driven_tangent
        self.last_boundary_couples[1] = fixed_scalar * fixed_tangent

    def _orientation(self) -> tuple[np.ndarray, np.ndarray]:
        quaternion = np.empty((self.n, 4), dtype=float)
        angular_velocity = np.empty((self.n, 3), dtype=float)
        fallback = np.array([1.0, 0.0, 0.0])
        for i in range(self.n):
            if i == 0:
                tangent = _unit(self.x[1] - self.x[0], fallback)
            elif i == self.n - 1:
                tangent = _unit(self.x[-1] - self.x[-2], fallback)
            else:
                tangent = _unit(self.x[i + 1] - self.x[i - 1], fallback)
            half_angle = 0.5 * float(self.theta[i])
            quaternion[i, :3] = tangent * math.sin(half_angle)
            quaternion[i, 3] = math.cos(half_angle)
            angular_velocity[i] = self.omega[i] * tangent
        quaternion[-1] = np.array([0.0, 0.0, 0.0, 1.0])
        angular_velocity[-1] = 0.0
        return quaternion, angular_velocity

    def _frame(self, time: float) -> StateFrame:
        orientation, angular_velocity = self._orientation()
        drive_angle, _ = self._drive(time)
        target_orientation = orientation[[0, -1]].copy()
        axis = _unit(self.x[1] - self.x[0], np.array([1.0, 0.0, 0.0]))
        target_orientation[0, :3] = axis * math.sin(0.5 * drive_angle)
        target_orientation[0, 3] = math.cos(0.5 * drive_angle)
        target_orientation[1] = np.array([0.0, 0.0, 0.0, 1.0])

        boundary_forces = np.empty((1, 2, 3), dtype=float)
        boundary_forces[0, 0] = 0.5 * self.last_external_force
        boundary_forces[0, 1] = 0.5 * self.last_external_force
        force_positions = np.array([[self.x[0], self.x[-1]]], dtype=float)
        boundary_couples = self.last_boundary_couples[None, :, :].copy()

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            attributes={
                "arc_length": self.arc_length.copy(),
                "material_twist": self.theta.copy(),
                "cross_section_width": np.full(self.n, self.width),
                "cross_section_thickness": np.full(self.n, self.thickness),
                "is_apex": (np.arange(self.n) == self.apex).astype(float),
                "is_driven_end": (np.arange(self.n) == 0).astype(float),
                "is_fixed_end": (np.arange(self.n) == self.n - 1).astype(float),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_couple": boundary_couples,
                "budget_external_force": boundary_forces,
                "budget_force_position": force_positions,
                "constraint_value": orientation[[0, -1]].copy(),
                "constraint_target": target_orientation,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "driven_end_angle": drive_angle,
                "constraint_scale": 1.0,
                "reference_force": max(self.young_modulus * self.area, 1.0e-12),
                "reference_torque": max(self.normal_edge_twist_stiffness, 1.0e-12),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self.current_time - 1.0e-12:
            raise ValueError("advance_to requires finite, nondecreasing times")
        while self.current_time < target - 1.0e-12:
            dt = min(0.001, target - self.current_time)
            new_time = self.current_time + dt
            self._step_twist(dt, new_time)
            self._step_centerline(dt)
            self.current_time = new_time
            self.internal_steps += 1
        self.current_time = target
        self._update_boundary_couples()
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "implicit torsion plus constrained anisotropic discrete rod",
            "internal_steps": int(self.internal_steps),
            "maximum_twist_radians": float(np.max(np.abs(self.theta))),
            "maximum_speed": float(np.max(np.linalg.norm(self.v, axis=1))),
        }
