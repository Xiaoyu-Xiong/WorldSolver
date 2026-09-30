"""Solver implementation for the armadillo roller stick-slip benchmark.

This solver integrates a low-poly tetrahedral elastic solid with high-friction
kinematic rollers using an explicit corotated hyperelastic FEM model, a
penalty contact model, and Coulomb friction.  It is implemented with NumPy
and SciPy only (no genesis/gs).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import Delaunay
from worldsolver.contracts import StateFrame


Array = NDArray[np.floating]


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.config = scene.config
        frame0 = scene.initial_state()

        # Per-sample state
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int64).copy()
        self.rest_position = np.asarray(frame0.attributes.get("rest_position", self.position), dtype=float).copy()
        self.n = self.position.shape[0]

        # Simulation parameters from config
        bc = self.config.get("boundary_conditions", {})
        self.roller_radius: float = 0.22
        self.roller_centers = np.array(
            [
                [-0.20, 0.0, 0.30],  # upper left
                [0.30, 0.0, 0.30],   # upper right
                [-0.20, 0.0, -0.30], # lower left
                [0.30, 0.0, -0.30],  # lower right
            ],
            dtype=float,
        )
        # Effective angular velocities chosen so that the gap surfaces move in
        # the intended positive-x transport direction.
        self.roller_omega = np.array([-5.0, -5.0, 5.0, 5.0], dtype=float)
        self.roller_start = float(bc.get("start_time", 0.35))
        self.roller_ramp = float(bc.get("ramp_duration", 0.25))
        self.friction_coeff = float(bc.get("contact_friction_coefficient", 0.5))
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.intended_direction = bc.get("intended_transport_direction", "positive_x")
        if self.intended_direction == "negative_x":
            self.roller_omega *= -1.0

        # Material model
        E = 5.0e5  # Pa, moderate stiffness
        nu = 0.45
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Explicit time-stepping sub-step size
        self.substep: float = 1.0e-4

        # Contact penalty stiffness
        self.k_pen = 2.0e5
        self.k_damp_contact = 2.0e3
        self.vel_damping = 1.0
        self.stick_tol = 0.01  # m/s

        # Tetrahedral mesh from reference positions
        self.tets = self._build_tets(self.rest_position)
        self.n_tets = self.tets.shape[0]
        self.Dm_inv: NDArray[np.floating] = np.zeros((self.n_tets, 3, 3))
        self.tet_volume0: NDArray[np.floating] = np.zeros(self.n_tets)
        for e, nodes in enumerate(self.tets):
            p0, p1, p2, p3 = self.rest_position[nodes]
            Dm = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
            vol = np.linalg.det(Dm) / 6.0
            if abs(vol) < 1e-12:
                # Degenerate tetrahedron; mark with zero volume/inverse.
                self.Dm_inv[e] = np.eye(3)
                self.tet_volume0[e] = 0.0
                continue
            if vol < 0.0:
                # Flip to obtain a positive reference volume
                nodes[1], nodes[2] = nodes[2], nodes[1]
                p1, p2 = p2, p1
                Dm = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
                vol = -vol
            self.Dm_inv[e] = np.linalg.inv(Dm)
            self.tet_volume0[e] = vol

        # Budget bookkeeping
        self.G = 2  # global domain, armadillo domain
        self.K_force = 2  # gravity, contact
        self.K_mass = 1
        self.cumulative_external_impulse = np.zeros((self.G, 3), dtype=float)
        self.cumulative_inflow = np.zeros((self.G, 3), dtype=float)
        self.cumulative_outflow = np.zeros((self.G, 3), dtype=float)
        self.cumulative_mass_source = np.zeros(self.K_mass, dtype=float)
        self.time: float = 0.0
        self._last_time: float = 0.0

    def diagnostics(self) -> dict[str, object]:
        return {
            "nodes": self.n,
            "tets": self.n_tets,
            "substep": self.substep,
        }

    def advance_to(self, time: float) -> StateFrame:
        dt = time - self.time
        if dt < 0.0:
            raise ValueError("advance_to called with a backward time")

        if dt > 0.0:
            n_sub = max(1, int(np.ceil(dt / self.substep)))
            h = dt / n_sub
            for _ in range(n_sub):
                self._step(h)
            # Do not let rounding errors drift the requested time
            self.time = float(time)

        return self._build_frame()

    def _build_tets(self, positions: NDArray[np.floating]) -> NDArray[np.integer]:
        # Delaunay tetrahedralisation of the node cloud.
        delaunay = Delaunay(positions)
        tets = np.asarray(delaunay.simplices, dtype=np.int64)
        # Ensure each tet has a non-zero volume under the current pose.
        keep = []
        for nodes in tets:
            p0, p1, p2, p3 = positions[nodes]
            vol = np.linalg.det(np.column_stack([p1 - p0, p2 - p0, p3 - p0])) / 6.0
            if abs(vol) > 1e-12:
                keep.append(nodes)
        if len(keep) == 0:
            # Fallback: simple tet connecting the first four nodes.
            keep = [np.arange(4, dtype=np.int64)]
        return np.asarray(keep, dtype=np.int64)

    def _ramp_factor(self, t: float) -> float:
        if t < self.roller_start:
            return 0.0
        x = (t - self.roller_start) / self.roller_ramp
        if x >= 1.0:
            return 1.0
        return x * x

    def _roller_surface_velocity(self, point: NDArray[np.floating], t: float) -> NDArray[np.floating]:
        ramp = self._ramp_factor(t)
        v = np.zeros(3, dtype=float)
        for center, omega0 in zip(self.roller_centers, self.roller_omega):
            r = point - center
            omega = omega0 * ramp
            # angular velocity about y
            v += np.array([omega * r[2], 0.0, -omega * r[0]], dtype=float)
        return v

    def _elastic_forces(self, positions: NDArray[np.floating]) -> NDArray[np.floating]:
        forces = np.zeros_like(positions)
        for e, nodes in enumerate(self.tets):
            vol0 = self.tet_volume0[e]
            if vol0 <= 0.0:
                continue
            Dm_inv = self.Dm_inv[e]
            p0, p1, p2, p3 = positions[nodes]
            Ds = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
            F = Ds @ Dm_inv
            # Neo-Hookean first Piola stress with stability for small J.
            J = np.linalg.det(F)
            if abs(J) < 1e-8:
                # Treat near-singular element as rigid for this step.
                continue
            F_inv_T = np.linalg.inv(F).T
            P = self.mu * (F - F_inv_T) + self.lam * np.log(max(J, 1e-8)) * F_inv_T
            # Nodal forces: f_i = -V0 * P * grad_Ni  (i=1,2,3), f_0 = -(f1+f2+f3)
            H = P @ Dm_inv.T
            f1 = -vol0 * H[:, 0]
            f2 = -vol0 * H[:, 1]
            f3 = -vol0 * H[:, 2]
            f0 = -(f1 + f2 + f3)
            forces[nodes[0]] += f0
            forces[nodes[1]] += f1
            forces[nodes[2]] += f2
            forces[nodes[3]] += f3
        return forces

    def _contact_forces(self, positions: NDArray[np.floating], velocities: NDArray[np.floating], t: float,
                        ) -> tuple[NDArray[np.floating], list[dict]]:
        forces = np.zeros_like(positions)
        contacts: list[dict] = []
        ramp = self._ramp_factor(t)
        for i in range(self.n):
            p = positions[i]
            v = velocities[i]
            for center, omega0 in zip(self.roller_centers, self.roller_omega):
                dx = p[0] - center[0]
                dz = p[2] - center[2]
                dist = np.hypot(dx, dz)
                if dist < 1e-9:
                    continue
                penetration = self.roller_radius - dist
                if penetration <= 0.0:
                    continue
                n = np.array([dx / dist, 0.0, dz / dist], dtype=float)
                r = p - center
                omega = omega0 * ramp
                v_surf = np.array([omega * r[2], 0.0, -omega * r[0]], dtype=float)
                v_rel = v - v_surf
                v_n = np.dot(v_rel, n)
                v_t = v_rel - v_n * n

                f_n_mag = self.k_pen * penetration - self.k_damp_contact * v_n
                if f_n_mag < 0.0:
                    f_n_mag = 0.0
                f_n = f_n_mag * n

                # Coulomb friction
                mu = self.friction_coeff
                v_t_mag = np.linalg.norm(v_t)
                if v_t_mag < self.stick_tol:
                    f_t = -mu * f_n_mag * v_t / max(self.stick_tol, 1e-8)
                else:
                    f_t = -mu * f_n_mag * v_t / v_t_mag

                forces[i] += f_n + f_t

                contact = {
                    "normal": n,
                    "normal_force": f_n_mag,
                    "relative_velocity": v_rel,
                    "tangential_force": f_t,
                }
                contacts.append(contact)
        return forces, contacts

    def _step(self, h: float) -> None:
        t = self.time
        # Elastic forces
        f_elastic = self._elastic_forces(self.position)
        # Gravity
        f_gravity = self.mass[:, None] * self.gravity
        # Contact + friction
        f_contact, _ = self._contact_forces(self.position, self.velocity, t)
        # Velocity damping (artificial viscous)
        f_damp = -self.vel_damping * self.velocity

        f_total = f_elastic + f_gravity + f_contact + f_damp
        a = f_total / self.mass[:, None]
        self.velocity += a * h
        self.position += self.velocity * h

        # Budget update (gravity + contact are the external impulses sources)
        ext_total = f_gravity + f_contact
        domain_weights = self._domain_weights()  # shape (G, N)
        # total external force per domain
        ext_per_domain = domain_weights @ ext_total  # (G, 3)
        self.cumulative_external_impulse += ext_per_domain * h
        # No mass flow -> inflow/outflow remain zero.

        self.time = t + h

    def _domain_weights(self) -> NDArray[np.floating]:
        w = np.zeros((self.G, self.n), dtype=float)
        # Global domain
        w[0, :] = 1.0 / self.n
        # Armadillo domain
        mask = self.body_id == 1
        if np.any(mask):
            w[1, mask] = 1.0
        return w

    def _budget_external_force(self, f_gravity: NDArray[np.floating],
                                f_contact: NDArray[np.floating]) -> NDArray[np.floating]:
        # shape (G, K_force, 3)
        w = self._domain_weights()  # (G, N)
        budget = np.zeros((self.G, self.K_force, 3), dtype=float)
        budget[:, 0, :] = w @ f_gravity
        budget[:, 1, :] = w @ f_contact
        return budget

    def _element_deformation_gradient(self) -> NDArray[np.floating]:
        F = np.zeros((self.n_tets, 3, 3), dtype=float)
        for e, nodes in enumerate(self.tets):
            vol0 = self.tet_volume0[e]
            if vol0 <= 0.0:
                F[e] = np.eye(3)
                continue
            p0, p1, p2, p3 = self.position[nodes]
            Ds = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
            F[e] = Ds @ self.Dm_inv[e]
        return F

    def _build_frame(self) -> StateFrame:
        # Recompute current external forces for the budget field export.
        f_elastic = self._elastic_forces(self.position)
        f_gravity = self.mass[:, None] * self.gravity
        f_contact, contacts = self._contact_forces(self.position, self.velocity, self.time)
        budget_force = self._budget_external_force(f_gravity, f_contact)

        # Friction contact arrays
        if contacts:
            C = len(contacts)
            normal = np.stack([c["normal"] for c in contacts])
            normal_force = np.array([c["normal_force"] for c in contacts])
            rel_vel = np.stack([c["relative_velocity"] for c in contacts])
            tangential_force = np.stack([c["tangential_force"] for c in contacts])
            active_mask = np.ones(C, dtype=float)
            static_coeff = np.full(C, self.friction_coeff, dtype=float)
            dynamic_coeff = np.full(C, self.friction_coeff, dtype=float)
        else:
            C = 0
            normal = np.zeros((C, 3), dtype=float)
            normal_force = np.zeros(C, dtype=float)
            rel_vel = np.zeros((C, 3), dtype=float)
            tangential_force = np.zeros((C, 3), dtype=float)
            active_mask = np.zeros(C, dtype=float)
            static_coeff = np.zeros(C, dtype=float)
            dynamic_coeff = np.zeros(C, dtype=float)

        # Reference scales
        total_mass = self.mass.sum()
        surface_speed = self.roller_radius * abs(self.roller_omega[0])
        reference_force = max(1.0, self.k_pen * self.roller_radius * 0.1)
        reference_momentum = max(1.0, total_mass * surface_speed)
        reference_velocity = max(0.1, surface_speed)

        # attributes
        mass_quantity = self.mass.copy()

        fields = {
            "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cumulative_inflow.copy(),
            "budget_cumulative_outflow_momentum": self.cumulative_outflow.copy(),
            "budget_domain_weights": self._domain_weights().copy(),
            "budget_external_force": budget_force.copy(),
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "element_deformation_gradient": self._element_deformation_gradient(),
            "friction_active_mask": active_mask,
            "friction_dynamic_coefficient": dynamic_coeff,
            "friction_normal": normal,
            "friction_normal_force": normal_force,
            "friction_relative_velocity": rel_vel,
            "friction_static_coefficient": static_coeff,
            "friction_tangential_force": tangential_force,
        }

        scalars = {
            "friction_stick_speed_tolerance": self.stick_tol,
            "reference_force": reference_force,
            "reference_momentum": reference_momentum,
            "reference_velocity": reference_velocity,
        }

        frame = StateFrame(
            time=self.time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "rest_position": self.rest_position.copy(),
                "mass_quantity": mass_quantity,
            },
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame
