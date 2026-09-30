"""Deterministic pressure-actuated solid/beam solver for the soft finger.

The finger is discretised into geometrically exact planar solid sections.  The
section curvature degrees of freedom are advanced with an implicit Euler step
for the damped angular momentum balance.  Closed-chamber pressure has zero net
force, but, because the compliant chamber wall is offset from the passive
underside, it performs work through a bending moment.  Each rendered material
point is transported by the resulting finite-bending map.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Reduced finite-section model of a three-dimensional silicone solid."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # The model contains no stochastic terms.
        self.scene = scene
        initial = scene.initial_state()
        self.rest = np.asarray(initial.position, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.fixed = self.rest[:, 0] <= float(
            scene.config["initial_conditions"]["base_support_region_x"][1]
        )

        self.base_x = float(scene.config["initial_conditions"]["base_support_region_x"][1])
        self.end_x = float(np.max(self.rest[:, 0]))
        self.length = self.end_x - self.base_x
        # The axis follows the middle of the passive, strain-limiting wall.
        low_z = float(np.min(self.rest[:, 2]))
        self.axis_z = low_z + 0.007

        asset = Path(__file__).resolve().parents[1] / "assets" / "pneumatic_finger.obj"
        self.faces = self._read_faces(asset)
        self.face_centres = np.asarray(
            [np.mean(self.rest[face], axis=0) for face in self.faces], dtype=float
        )
        self.chambers = self._find_chambers()

        # Twelve finite sections resolve the six chambers and their compliant gaps.
        self.section_count = 12
        self.ds = self.length / self.section_count
        self.q = np.zeros(self.section_count, dtype=float)  # angle change per section
        self.qd = np.zeros_like(self.q)

        # Passive-wall silicone bending stiffness EI.  The chosen modulus is in
        # the usual range for a soft platinum-cure silicone.
        young_modulus = 3.0e5
        passive_width = max(float(np.ptp(self.rest[:, 1])) * (0.040 / 0.048), 0.040)
        passive_thickness = 0.014
        second_moment = passive_width * passive_thickness**3 / 12.0
        self.bending_stiffness = young_modulus * second_moment
        self.K = np.eye(self.section_count) * (self.bending_stiffness / self.ds)

        # Use the physical material point inertia in the generalized mass matrix.
        J0 = self._position_jacobian(self.q)
        flat_J = J0.reshape((-1, self.section_count))
        coordinate_mass = np.repeat(self.mass, 3)
        self.M = flat_J.T @ (coordinate_mass[:, None] * flat_J)
        flexible_mass = float(np.sum(self.mass[~self.fixed]))
        section_mass = flexible_mass / self.section_count
        rotary_inertia = section_mass * (0.040**2 + passive_thickness**2) / 12.0
        self.M += np.eye(self.section_count) * max(rotary_inertia, 1.0e-9)

        # Rayleigh damping damps the short transient while retaining the static
        # pressure/elasticity equilibrium during the prescribed hold.
        self.C = 5.0 * self.M + 0.0025 * self.K

        pressure_cfg = scene.config["pressure"]
        self.target_pressure = float(pressure_cfg["chamber_pressure_pa"])
        self.ramp_start = float(pressure_cfg["ramp_start_s"])
        self.ramp_end = float(pressure_cfg["ramp_end_s"])
        self.hold_until = float(pressure_cfg["hold_until_s"])

        chamber_width = float(np.median([c[3] - c[2] for c in self.chambers]))
        chamber_height = float(np.median([c[5] - c[4] for c in self.chambers]))
        # Moment per unit pressure from the asymmetric chamber/passive-wall
        # cross-section.  The geometric factor accounts for the nearly cancelling
        # tractions on the enclosed chamber and their offset compliance.
        self.pressure_moment_coefficient = chamber_width * chamber_height**2 / 48.0
        self.active_fraction = self._section_chamber_fractions()

        pressure_area = chamber_width * chamber_height
        self.reference_force = max(self.target_pressure * pressure_area, 1.0e-12)
        self.current_time = 0.0
        self.internal_steps = 0
        self.max_tip_displacement = 0.0
        self.max_curvature = 0.0
        self.tip_mask = self.rest[:, 0] >= self.end_x - 1.0e-9

    @staticmethod
    def _read_faces(asset: Path) -> list[np.ndarray]:
        faces: list[np.ndarray] = []
        for line in asset.read_text(encoding="utf-8").splitlines():
            if line.startswith("f "):
                indices = [int(item.split("/")[0]) - 1 for item in line.split()[1:]]
                faces.append(np.asarray(indices, dtype=int))
        return faces

    def _find_chambers(self) -> list[tuple[float, float, float, float, float, float]]:
        """Infer chamber component bounds from the frozen disconnected-box mesh."""
        adjacency: list[set[int]] = [set() for _ in range(len(self.rest))]
        for face in self.faces:
            for vertex in face:
                adjacency[int(vertex)].update(int(v) for v in face if v != vertex)
        unseen = set(range(len(self.rest)))
        bounds = []
        while unseen:
            seed = unseen.pop()
            component = {seed}
            stack = [seed]
            while stack:
                vertex = stack.pop()
                new = adjacency[vertex].intersection(unseen)
                unseen.difference_update(new)
                component.update(new)
                stack.extend(new)
            points = self.rest[np.asarray(sorted(component), dtype=int)]
            lo = np.min(points, axis=0)
            hi = np.max(points, axis=0)
            # Chamber lobes are tall components; this excludes the underside and nose.
            if hi[2] - lo[2] >= 0.030 and hi[2] >= 0.045:
                bounds.append((lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
        bounds.sort(key=lambda item: item[0])
        if not bounds:
            raise ValueError("soft-finger mesh contains no enclosed chamber components")
        return bounds

    def _section_chamber_fractions(self) -> np.ndarray:
        fractions = np.zeros(self.section_count, dtype=float)
        for j in range(self.section_count):
            x0 = self.base_x + j * self.ds
            x1 = x0 + self.ds
            overlap = 0.0
            for chamber in self.chambers:
                overlap += max(0.0, min(x1, chamber[1]) - max(x0, chamber[0]))
            fractions[j] = min(overlap / self.ds, 1.0)
        return fractions

    @staticmethod
    def _sinc_half(angle):
        """Return sin(angle/2)/(angle/2), continuously at zero."""
        return np.sinc(np.asarray(angle) / (2.0 * np.pi))

    def _section_frames(self, q: np.ndarray):
        origins = np.empty((self.section_count + 1, 2), dtype=float)
        angles = np.empty(self.section_count + 1, dtype=float)
        origins[0] = (self.base_x, self.axis_z)
        angles[0] = 0.0
        for j in range(self.section_count):
            half_angle = angles[j] + 0.5 * q[j]
            chord = self.ds * float(self._sinc_half(q[j]))
            origins[j + 1] = origins[j] + chord * np.array(
                [np.cos(half_angle), np.sin(half_angle)]
            )
            angles[j + 1] = angles[j] + q[j]
        return origins, angles

    def _positions(self, q: np.ndarray) -> np.ndarray:
        result = self.rest.copy()
        origins, angles = self._section_frames(q)
        for index, material_point in enumerate(self.rest):
            if self.fixed[index]:
                continue
            s = float(np.clip(material_point[0] - self.base_x, 0.0, self.length))
            section = min(int(s / self.ds), self.section_count - 1)
            u = s - section * self.ds
            fraction = u / self.ds
            turn = fraction * q[section]
            phi = angles[section] + turn
            chord = u * float(self._sinc_half(turn))
            mid_angle = angles[section] + 0.5 * turn
            centre = origins[section] + chord * np.array(
                [np.cos(mid_angle), np.sin(mid_angle)]
            )
            z_offset = material_point[2] - self.axis_z
            result[index, 0] = centre[0] - np.sin(phi) * z_offset
            result[index, 1] = material_point[1]
            result[index, 2] = centre[1] + np.cos(phi) * z_offset
        return result

    def _position_jacobian(self, q: np.ndarray) -> np.ndarray:
        jacobian = np.empty((len(self.rest), 3, self.section_count), dtype=float)
        epsilon = 2.0e-6
        for j in range(self.section_count):
            offset = np.zeros(self.section_count, dtype=float)
            offset[j] = epsilon
            jacobian[:, :, j] = (
                self._positions(q + offset) - self._positions(q - offset)
            ) / (2.0 * epsilon)
        return jacobian

    def _pressure(self, time: float) -> float:
        if time <= self.ramp_start:
            return 0.0
        if time < self.ramp_end:
            return self.target_pressure * (time - self.ramp_start) / (
                self.ramp_end - self.ramp_start
            )
        if time <= self.hold_until:
            return self.target_pressure
        return self.target_pressure

    def _pressure_torque(self, time: float) -> np.ndarray:
        # Negative curvature lengthens the chambered (+z) side and curls the tip down.
        return (
            -self._pressure(time)
            * self.pressure_moment_coefficient
            * self.active_fraction
        )

    def _step(self, step_size: float) -> None:
        next_time = self.current_time + step_size
        torque = self._pressure_torque(next_time)
        # Backward Euler for M*qdd + C*qd + K*q = pressure torque.
        system = self.M + step_size * self.C + step_size**2 * self.K
        rhs = self.M @ self.qd + step_size * (torque - self.K @ self.q)
        next_qd = np.linalg.solve(system, rhs)
        self.q = self.q + step_size * next_qd
        self.qd = next_qd
        self.current_time = next_time
        self.internal_steps += 1

    def _kinematics(self, acceleration: np.ndarray):
        position = self._positions(self.q)
        velocity_dt = 1.0e-5
        if np.any(self.qd):
            velocity = (
                self._positions(self.q + velocity_dt * self.qd)
                - self._positions(self.q - velocity_dt * self.qd)
            ) / (2.0 * velocity_dt)
            convective_dt = 2.0e-4
            convective = (
                self._positions(self.q + convective_dt * self.qd)
                - 2.0 * position
                + self._positions(self.q - convective_dt * self.qd)
            ) / convective_dt**2
        else:
            velocity = np.zeros_like(position)
            convective = np.zeros_like(position)

        if np.any(acceleration):
            derivative_scale = 1.0e-5
            tangential = (
                self._positions(self.q + derivative_scale * acceleration)
                - self._positions(self.q - derivative_scale * acceleration)
            ) / (2.0 * derivative_scale)
        else:
            tangential = np.zeros_like(position)
        return position, velocity, tangential + convective

    def _deformation_gradients(self) -> np.ndarray:
        origins, angles = self._section_frames(self.q)
        del origins
        gradients = np.repeat(np.eye(3)[None, :, :], len(self.face_centres), axis=0)
        for element, centre in enumerate(self.face_centres):
            if centre[0] <= self.base_x:
                continue
            s = float(np.clip(centre[0] - self.base_x, 0.0, self.length))
            section = min(int(s / self.ds), self.section_count - 1)
            u = s - section * self.ds
            phi = angles[section] + self.q[section] * u / self.ds
            curvature = self.q[section] / self.ds
            axial_stretch = 1.0 - curvature * (centre[2] - self.axis_z)
            cosine = np.cos(phi)
            sine = np.sin(phi)
            gradients[element] = np.array(
                [
                    [axial_stretch * cosine, 0.0, -sine],
                    [0.0, 1.0, 0.0],
                    [axial_stretch * sine, 0.0, cosine],
                ]
            )
        return gradients

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.current_time - 1.0e-12:
            raise ValueError("advance_to requires finite, nondecreasing times")
        while self.current_time < target - 1.0e-12:
            self._step(min(0.001, target - self.current_time))
        # Remove insignificant accumulated time roundoff and report the requested time.
        self.current_time = target

        torque = self._pressure_torque(target)
        generalized_acceleration = np.linalg.solve(
            self.M, torque - self.C @ self.qd - self.K @ self.q
        )
        position, velocity, point_acceleration = self._kinematics(generalized_acceleration)
        net_external_force = np.sum(self.mass[:, None] * point_acceleration, axis=0)

        # Pressure on each closed chamber has zero resultant.  The second term is
        # the support resultant required by the constrained base/actuation couple.
        budget_force = np.zeros((1, 2, 3), dtype=float)
        budget_force[0, 1] = net_external_force

        tip_displacement = float(
            np.max(np.linalg.norm(position[self.tip_mask] - self.rest[self.tip_mask], axis=1))
        )
        self.max_tip_displacement = max(self.max_tip_displacement, tip_displacement)
        self.max_curvature = max(
            self.max_curvature, float(np.max(np.abs(self.q / self.ds)))
        )

        frame = StateFrame(
            time=target,
            position=position,
            velocity=velocity,
            mass=self.mass.copy(),
            body_id=np.zeros(len(self.rest), dtype=int),
            attributes={
                "fixed_base": self.fixed.astype(float),
                "rest_position": self.rest.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_force": budget_force,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "element_deformation_gradient": self._deformation_gradients(),
            },
            scalars={
                "chamber_pressure_pa": self._pressure(target),
                "reference_force": self.reference_force,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "implicit_damped_geometrically_exact_finite_sections",
            "section_count": int(self.section_count),
            "internal_steps": int(self.internal_steps),
            "max_tip_displacement_m": float(self.max_tip_displacement),
            "max_abs_curvature_per_m": float(self.max_curvature),
            "final_pressure_pa": float(self._pressure(self.current_time)),
        }
