"""Physically based solver for pitch-controlled cable winding.

The cable is represented by a finite-radius extensible discrete rod. Stretch
and Timoshenko bending are advanced with compliant position constraints
(XPBD), while circular-cross-section twist is advanced with an implicit
torsional spring solve. Contact is unilateral and the tangential impulse is
limited by the configured generalized power-friction law.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic discrete elastic-rod solver."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        cfg = scene.config
        cable = cfg["cable"]
        contact = cfg["contact"]

        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=np.float64).copy()
        self.v = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.n = len(self.x)
        self.inv_mass = 1.0 / self.mass
        self.inv_mass[-1] = 0.0

        self.length = float(cable["length"])
        self.rest_length = self.length / (self.n - 1)
        self.radius = float(cable["radius"])
        self.young = float(cable["young_modulus_pa"])
        self.shear = float(cable["shear_modulus_pa"])
        self.poisson = float(cable["poisson_ratio"])
        self.area = math.pi * self.radius**2
        self.second_moment = math.pi * self.radius**4 / 4.0
        self.polar_moment = 2.0 * self.second_moment
        self.EA = self.young * self.area
        self.EI = self.young * self.second_moment
        self.GJ = self.shear * self.polar_moment

        cyl = cfg["cylinder"]
        self.cylinder_center = np.asarray(cyl["center"], dtype=np.float64)
        self.cylinder_radius = float(cyl["radius"])
        self.cylinder_half_height = 0.5 * float(cyl["height"])
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        self.kappa = float(contact["kappa"])
        self.friction_exponent = float(contact["exponent_N"])

        shear_correction = 0.9
        self.stretch_compliance = self.rest_length / self.EA
        self.bend_compliance = (
            self.rest_length**3 / self.EI
            + self.rest_length / (shear_correction * self.shear * self.area)
        )

        self.twist = np.zeros(self.n, dtype=np.float64)
        self.twist_rate = np.zeros(self.n, dtype=np.float64)
        chunk = np.full(self.n, self.rest_length)
        chunk[[0, -1]] *= 0.5
        self.spin_inertia = float(cable["density"]) * self.polar_moment * chunk

        pair_i, pair_j = np.triu_indices(self.n, k=3)
        self.self_i = pair_i
        self.self_j = pair_j

        self.time = 0.0
        self.max_step = 0.001
        self.constraint_iterations = 6
        self.step_count = 0
        self._last_average_force = np.sum(self.mass) * self.gravity
        self._last_average_torque = np.zeros(3)

        self._initial_momentum = self._linear_momentum()
        self._initial_angular_momentum = self._angular_momentum()
        total_mass = float(np.sum(self.mass))
        speed_scale = math.sqrt(float(np.linalg.norm(self.gravity)) * self.length)
        self.reference_momentum = max(total_mass * speed_scale, 1.0e-12)
        self.reference_angular_momentum = max(
            self.reference_momentum * self.length, 1.0e-12
        )

    def _grip_pose(self, time: float) -> tuple[np.ndarray, float]:
        """Return exact prescribed grip position and material twist angle."""
        cfg = self.scene.config
        winding = cfg["winding"]
        t = float(np.clip(time, 0.0, float(cfg["duration"])))
        ta = float(winding["approach_end_time"])
        tw = float(winding["winding_end_time"])
        angle0 = float(winding["start_angle_rad"])

        if t < ta:
            s = t / ta
            u = s * s * (3.0 - 2.0 * s)
            initial = np.asarray(
                cfg["initial_conditions"]["driven_end"], dtype=np.float64
            )
            radius = float(winding["centerline_orbit_radius"])
            start = np.array(
                [radius * math.cos(angle0), radius * math.sin(angle0),
                 float(winding["start_height"])], dtype=np.float64
            )
            return (1.0 - u) * initial + u * start, angle0

        s = float(np.clip((t - ta) / (tw - ta), 0.0, 1.0))
        u = s * s * (3.0 - 2.0 * s)
        angle = angle0 + 2.0 * math.pi * float(winding["turns"]) * u
        radius = float(winding["centerline_orbit_radius"])
        z0 = float(winding["start_height"])
        z1 = float(winding["end_height"])
        position = np.array(
            [radius * math.cos(angle), radius * math.sin(angle),
             (1.0 - u) * z0 + u * z1], dtype=np.float64
        )
        return position, angle

    def _cylinder_sdf_normals(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signed distance and outward normal of the finite closed cylinder."""
        rel = points - self.cylinder_center
        rho = np.linalg.norm(rel[:, :2], axis=1)
        radial = rho - self.cylinder_radius
        axial = np.abs(rel[:, 2]) - self.cylinder_half_height
        rp = np.maximum(radial, 0.0)
        ap = np.maximum(axial, 0.0)
        outside = np.sqrt(rp * rp + ap * ap)
        sdf = outside + np.minimum(np.maximum(radial, axial), 0.0)

        normal = np.zeros_like(points)
        safe_rho = np.maximum(rho, 1.0e-12)
        er = np.zeros_like(points)
        er[:, 0] = rel[:, 0] / safe_rho
        er[:, 1] = rel[:, 1] / safe_rho
        ez = np.zeros_like(points)
        ez[:, 2] = np.where(rel[:, 2] >= 0.0, 1.0, -1.0)

        corner = (radial > 0.0) & (axial > 0.0)
        if np.any(corner):
            denom = np.maximum(outside[corner], 1.0e-12)
            normal[corner] = (
                er[corner] * (rp[corner] / denom)[:, None]
                + ez[corner] * (ap[corner] / denom)[:, None]
            )
        side = (~corner) & (radial >= axial)
        cap = (~corner) & (~side)
        normal[side] = er[side]
        normal[cap] = ez[cap]
        return sdf, normal

    def _project_environment(self) -> None:
        movable = self.inv_mass > 0.0
        sdf, normal = self._cylinder_sdf_normals(self.x)
        penetration = self.radius - sdf
        hit = movable & (penetration > 0.0)
        self.x[hit] += penetration[hit, None] * normal[hit]

        floor_penetration = self.radius - self.x[:, 2]
        hit_floor = movable & (floor_penetration > 0.0)
        self.x[hit_floor, 2] += floor_penetration[hit_floor]

    def _project_self_contact(self) -> None:
        i = self.self_i
        j = self.self_j
        delta = self.x[j] - self.x[i]
        distance = np.linalg.norm(delta, axis=1)
        hit = distance < (2.0 * self.radius)
        if not np.any(hit):
            return
        ih = i[hit]
        jh = j[hit]
        d = delta[hit]
        dist = distance[hit]
        normal = d / np.maximum(dist[:, None], 1.0e-12)
        coincident = dist < 1.0e-12
        if np.any(coincident):
            normal[coincident] = np.array([1.0, 0.0, 0.0])
        wi = self.inv_mass[ih]
        wj = self.inv_mass[jh]
        total_w = wi + wj
        depth = 2.0 * self.radius - dist
        correction = 0.8 * depth[:, None] * normal
        accum = np.zeros_like(self.x)
        counts = np.zeros(self.n, dtype=np.float64)
        np.add.at(accum, ih, -correction * (wi / total_w)[:, None])
        np.add.at(accum, jh, correction * (wj / total_w)[:, None])
        np.add.at(counts, ih, 1.0)
        np.add.at(counts, jh, 1.0)
        active = counts > 0.0
        self.x[active] += accum[active] / counts[active, None]

    def _solve_rod_constraints(self, h: float) -> None:
        stretch_lambda = np.zeros(self.n - 1, dtype=np.float64)
        bend_lambda = np.zeros((self.n - 2, 3), dtype=np.float64)
        alpha_s = self.stretch_compliance / (h * h)
        alpha_b = self.bend_compliance / (h * h)

        for _ in range(self.constraint_iterations):
            for parity in (0, 1):
                edge = np.arange(parity, self.n - 1, 2)
                delta = self.x[edge + 1] - self.x[edge]
                length = np.linalg.norm(delta, axis=1)
                direction = delta / np.maximum(length[:, None], 1.0e-12)
                w0 = self.inv_mass[edge]
                w1 = self.inv_mass[edge + 1]
                dl = (
                    -(length - self.rest_length) - alpha_s * stretch_lambda[edge]
                ) / (w0 + w1 + alpha_s)
                stretch_lambda[edge] += dl
                self.x[edge] -= (w0 * dl)[:, None] * direction
                self.x[edge + 1] += (w1 * dl)[:, None] * direction

            for color in (0, 1, 2):
                bend = np.arange(color, self.n - 2, 3)
                curvature_numerator = (
                    self.x[bend] - 2.0 * self.x[bend + 1] + self.x[bend + 2]
                )
                w0 = self.inv_mass[bend]
                w1 = self.inv_mass[bend + 1]
                w2 = self.inv_mass[bend + 2]
                denominator = w0 + 4.0 * w1 + w2 + alpha_b
                dl = (
                    -curvature_numerator - alpha_b * bend_lambda[bend]
                ) / denominator[:, None]
                bend_lambda[bend] += dl
                self.x[bend] += w0[:, None] * dl
                self.x[bend + 1] -= 2.0 * w1[:, None] * dl
                self.x[bend + 2] += w2[:, None] * dl

            self._project_environment()
            self._project_self_contact()

    @staticmethod
    def _solve_tridiagonal(
        lower: np.ndarray, diagonal: np.ndarray, upper: np.ndarray, rhs: np.ndarray
    ) -> np.ndarray:
        n = len(diagonal)
        c = upper.copy()
        d = rhs.copy()
        b = diagonal.copy()
        for index in range(1, n):
            factor = lower[index - 1] / b[index - 1]
            b[index] -= factor * c[index - 1]
            d[index] -= factor * d[index - 1]
        result = np.empty(n, dtype=np.float64)
        result[-1] = d[-1] / b[-1]
        for index in range(n - 2, -1, -1):
            result[index] = (d[index] - c[index] * result[index + 1]) / b[index]
        return result

    def _advance_twist(self, h: float, target_twist: float) -> None:
        free_n = self.n - 1
        theta_base = self.twist.copy()
        theta_base[-1] = target_twist
        laplacian = np.empty(free_n, dtype=np.float64)
        laplacian[0] = theta_base[1] - theta_base[0]
        laplacian[1:] = (
            theta_base[: free_n - 1]
            - 2.0 * theta_base[1:free_n]
            + theta_base[2:free_n + 1]
        )

        torsion_k = self.GJ / self.rest_length
        inertia = self.spin_inertia[:free_n]
        rotational_damping = 0.12
        diagonal = inertia * (1.0 + h * rotational_damping)
        diagonal[0] += h * h * torsion_k
        diagonal[1:] += 2.0 * h * h * torsion_k
        off_diagonal = np.full(free_n - 1, -h * h * torsion_k)
        rhs = inertia * self.twist_rate[:free_n] + h * torsion_k * laplacian
        new_rate = self._solve_tridiagonal(
            off_diagonal, diagonal, off_diagonal, rhs
        )
        self.twist[:free_n] += h * new_rate
        self.twist_rate[:free_n] = new_rate
        self.twist_rate[-1] = (target_twist - self.twist[-1]) / h
        self.twist[-1] = target_twist

    def _apply_contact_friction(self, unconstrained_velocity: np.ndarray) -> None:
        """Apply ||J_t|| <= kappa |J_n|**N at fixed contacts."""
        sdf, normal = self._cylinder_sdf_normals(self.x)
        hit = (sdf <= self.radius + 2.0e-6) & (self.inv_mass > 0.0)
        if np.any(hit):
            ids = np.flatnonzero(hit)
            nrm = normal[ids]
            velocity = self.v[ids]
            free_velocity = unconstrained_velocity[ids]
            normal_speed = np.sum(velocity * nrm, axis=1)
            inward = normal_speed < 0.0
            velocity[inward] -= normal_speed[inward, None] * nrm[inward]
            tangent_velocity = velocity - (
                np.sum(velocity * nrm, axis=1)[:, None] * nrm
            )
            tangent_speed = np.linalg.norm(tangent_velocity, axis=1)
            normal_delta_v = np.abs(
                np.sum((velocity - free_velocity) * nrm, axis=1)
            )
            normal_impulse = self.mass[ids] * normal_delta_v
            bound = self.kappa * np.power(
                normal_impulse + 1.0e-16, self.friction_exponent
            )
            needed = self.mass[ids] * tangent_speed
            fraction = np.minimum(1.0, bound / np.maximum(needed, 1.0e-16))
            velocity -= fraction[:, None] * tangent_velocity
            self.v[ids] = velocity

        floor_hit = (self.x[:, 2] <= self.radius + 2.0e-6) & (
            self.inv_mass > 0.0
        )
        if np.any(floor_hit):
            ids = np.flatnonzero(floor_hit)
            velocity = self.v[ids]
            free_velocity = unconstrained_velocity[ids]
            velocity[:, 2] = np.maximum(velocity[:, 2], 0.0)
            tangent_velocity = velocity.copy()
            tangent_velocity[:, 2] = 0.0
            tangent_speed = np.linalg.norm(tangent_velocity, axis=1)
            normal_impulse = self.mass[ids] * np.abs(
                velocity[:, 2] - free_velocity[:, 2]
            )
            bound = self.kappa * np.power(
                normal_impulse + 1.0e-16, self.friction_exponent
            )
            needed = self.mass[ids] * tangent_speed
            fraction = np.minimum(1.0, bound / np.maximum(needed, 1.0e-16))
            velocity -= fraction[:, None] * tangent_velocity
            self.v[ids] = velocity

    def _substep(self, h: float) -> None:
        old_x = self.x.copy()
        old_momentum = self._linear_momentum()
        old_angular = self._angular_momentum()

        self.v += h * self.gravity
        self.v *= math.exp(-0.025 * h)
        unconstrained_velocity = self.v.copy()
        self.x += h * self.v

        next_time = self.time + h
        grip_position, grip_twist = self._grip_pose(next_time)
        self.x[-1] = grip_position
        self._solve_rod_constraints(h)
        self.x[-1] = grip_position

        self.v = (self.x - old_x) / h
        self._apply_contact_friction(unconstrained_velocity)
        self.v[-1] = (grip_position - old_x[-1]) / h
        self._advance_twist(h, grip_twist)

        self.time = next_time
        new_momentum = self._linear_momentum()
        new_angular = self._angular_momentum()
        self._last_average_force = (new_momentum - old_momentum) / h
        self._last_average_torque = (new_angular - old_angular) / h
        self.step_count += 1

    def _tangents(self) -> np.ndarray:
        tangent = np.empty_like(self.x)
        tangent[0] = self.x[1] - self.x[0]
        tangent[-1] = self.x[-1] - self.x[-2]
        tangent[1:-1] = self.x[2:] - self.x[:-2]
        tangent /= np.maximum(
            np.linalg.norm(tangent, axis=1)[:, None], 1.0e-12
        )
        return tangent

    def _inertia_tensors(self) -> np.ndarray:
        tangent = self._tangents()
        chunk = np.full(self.n, self.rest_length)
        chunk[[0, -1]] *= 0.5
        axial = 0.5 * self.mass * self.radius**2
        transverse = self.mass * (chunk**2 / 12.0 + self.radius**2 / 4.0)
        identity = np.eye(3)[None, :, :]
        outer = tangent[:, :, None] * tangent[:, None, :]
        return transverse[:, None, None] * identity + (
            axial - transverse
        )[:, None, None] * outer

    def _linear_momentum(self) -> np.ndarray:
        return np.sum(self.mass[:, None] * self.v, axis=0)

    def _angular_momentum(self) -> np.ndarray:
        orbital = np.sum(np.cross(self.x, self.mass[:, None] * self.v), axis=0)
        tangent = self._tangents()
        spin = np.sum(
            (0.5 * self.mass * self.radius**2 * self.twist_rate)[:, None]
            * tangent,
            axis=0,
        )
        return orbital + spin

    def _frame(self, requested_time: float) -> StateFrame:
        tangent = self._tangents()
        angular_velocity = self.twist_rate[:, None] * tangent
        inertia = self._inertia_tensors()
        momentum = self._linear_momentum()
        angular_momentum = self._angular_momentum()
        cumulative_impulse = momentum - self._initial_momentum
        cumulative_angular_impulse = (
            angular_momentum - self._initial_angular_momentum
        )

        total_mass = float(np.sum(self.mass))
        center_of_mass = np.sum(self.mass[:, None] * self.x, axis=0) / total_mass
        gravity_force = total_mass * self.gravity
        gravity_torque = np.cross(center_of_mass, gravity_force)
        reaction_force = self._last_average_force - gravity_force
        reaction_couple = self._last_average_torque - gravity_torque

        zero_budget = np.zeros((1, 3), dtype=np.float64)
        fields = {
            "budget_cumulative_external_angular_impulse":
                cumulative_angular_impulse[None, :],
            "budget_cumulative_external_impulse": cumulative_impulse[None, :],
            "budget_cumulative_inflow_angular_momentum": zero_budget.copy(),
            "budget_cumulative_inflow_momentum": zero_budget.copy(),
            "budget_cumulative_outflow_angular_momentum": zero_budget.copy(),
            "budget_cumulative_outflow_momentum": zero_budget.copy(),
            "budget_domain_weights": np.ones((1, self.n), dtype=np.float64),
            "budget_external_force": np.array(
                [[gravity_force, reaction_force]], dtype=np.float64
            ),
            "budget_force_position": np.array(
                [[center_of_mass, np.zeros(3)]], dtype=np.float64
            ),
            "budget_external_couple": np.array(
                [[np.zeros(3), reaction_couple]], dtype=np.float64
            ),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
        }
        frame = StateFrame(
            time=float(requested_time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "inertia_tensor": inertia,
                "mass_quantity": self.mass.copy(),
                "arc_fraction": np.linspace(0.0, 1.0, self.n),
                "rest_segment_length": np.full(self.n, self.rest_length),
                "cable_radius": np.full(self.n, self.radius),
                "driven_endpoint": (np.arange(self.n) == self.n - 1).astype(float),
                "free_endpoint": (np.arange(self.n) == 0).astype(float),
            },
            fields=fields,
            scalars={
                "reference_momentum": self.reference_momentum,
                "reference_angular_momentum": self.reference_angular_momentum,
                "young_modulus_pa": self.young,
                "poisson_ratio": self.poisson,
                "generalized_friction_kappa": self.kappa,
                "generalized_friction_exponent": self.friction_exponent,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-12:
            raise ValueError("advance_to times must be finite and nondecreasing")
        remaining = target - self.time
        if remaining > 1.0e-12:
            steps = max(1, int(math.ceil(remaining / self.max_step)))
            h = remaining / steps
            for _ in range(steps):
                self._substep(h)
            self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        segment_lengths = np.linalg.norm(np.diff(self.x, axis=0), axis=1)
        return {
            "method": "XPBD extensible discrete rod with implicit torsion",
            "substeps": int(self.step_count),
            "constraint_iterations": int(self.constraint_iterations),
            "minimum_segment_length": float(np.min(segment_lengths)),
            "maximum_segment_length": float(np.max(segment_lengths)),
            "finite": bool(np.isfinite(self.x).all() and np.isfinite(self.v).all()),
        }
