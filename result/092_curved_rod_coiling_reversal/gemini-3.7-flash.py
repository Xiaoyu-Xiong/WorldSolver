"""Physical solver for naturally curved rod coiling reversal."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = None
        self.current_time = 0.0
        self.dt = 0.002  # internal sub-step dt

        # Physics parameters
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.rod_radius = 0.0016
        self.rod_density = 1180.0
        self.young_modulus = 1300000.0
        self.poisson_ratio = 0.49
        self.shear_modulus = self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))
        self.natural_curvature = 20.9090909091
        self.injection_speed = 0.02
        self.outlet_pos = np.array([0.0, 0.0, 0.495])

        # Bending and twisting stiffness
        r = self.rod_radius
        self.area = np.pi * r**2
        self.I = np.pi * r**4 / 4.0
        self.J = np.pi * r**4 / 2.0
        self.B = self.young_modulus * self.I  # Bending stiffness
        self.C = self.shear_modulus * self.J  # Twisting stiffness
        self.linear_density = self.rod_density * self.area

        # Discretization state
        self.nodes = None  # (N, 3)
        self.velocities = None  # (N, 3)
        self.thetas = None  # (N-1,) twist angle / material frame angles
        self.is_deposited = None  # (N,) bool
        self.segment_rest_len = 0.025
        self.inlet_injected_length = 0.0

        # Mass reference & accounting
        self.initial_mass = 0.0
        self.cumulative_mass_injected = 0.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.current_time = 0.0

        init_frame = scene.initial_state()
        self.nodes = np.array(init_frame.position, dtype=float, copy=True)
        self.velocities = np.zeros_like(self.nodes)
        n = len(self.nodes)

        dep_mask = init_frame.attributes.get("is_initially_deposited", np.zeros(n))
        self.is_deposited = (dep_mask > 0.5)

        # Segment rest lengths
        diffs = np.diff(self.nodes, axis=0)
        self.segment_lengths = np.linalg.norm(diffs, axis=1)
        self.segment_rest_len = float(np.median(self.segment_lengths))
        self.thetas = np.zeros(n - 1, dtype=float)

        self.initial_mass = float(np.sum(init_frame.mass))
        self.cumulative_mass_injected = 0.0

    def _step_physics(self, dt: float) -> None:
        """Advance the rod quasi-static / dynamic physical state by dt."""
        n = len(self.nodes)

        # 1. Injection handling: feed new rod length from the outlet
        feed_dl = self.injection_speed * dt
        self.inlet_injected_length += feed_dl
        self.cumulative_mass_injected += feed_dl * self.linear_density

        # If injected enough length, prepend a new node at outlet
        if self.inlet_injected_length >= self.segment_rest_len:
            self.inlet_injected_length -= self.segment_rest_len
            new_pos = self.outlet_pos.copy()
            self.nodes = np.vstack([new_pos, self.nodes])
            self.velocities = np.vstack([np.array([0.0, 0.0, -self.injection_speed]), self.velocities])
            self.is_deposited = np.concatenate([[False], self.is_deposited])
            self.thetas = np.concatenate([[0.0], self.thetas])
            n = len(self.nodes)

        # Move the top node (inlet boundary condition)
        # Node 0 is held at outlet with prescribed natural curvature tangent & normal
        self.nodes[0] = self.outlet_pos.copy()
        self.velocities[0] = np.array([0.0, 0.0, -self.injection_speed])

        # Compute internal elastic forces and gravity
        forces = np.zeros((n, 3), dtype=float)

        # Gravity force
        # Lumped node lengths
        edge_vecs = np.diff(self.nodes, axis=0)
        edge_lens = np.linalg.norm(edge_vecs, axis=1)
        edge_lens_safe = np.maximum(edge_lens, 1e-7)
        tangents = edge_vecs / edge_lens_safe[:, None]

        lumped_len = np.zeros(n)
        lumped_len[0] = edge_lens[0] * 0.5
        lumped_len[-1] = edge_lens[-1] * 0.5
        lumped_len[1:-1] = (edge_lens[:-1] + edge_lens[1:]) * 0.5
        node_mass = self.linear_density * lumped_len

        forces += node_mass[:, None] * self.gravity[None, :]

        # Stretch / axial spring forces to maintain inextensibility
        # F_stretch = k_s * (|e| - L0) * e_hat
        k_stretch = self.young_modulus * self.area / self.segment_rest_len
        for i in range(n - 1):
            dl = edge_lens[i] - self.segment_rest_len
            f_axial = k_stretch * dl * tangents[i]
            forces[i] += f_axial
            forces[i + 1] -= f_axial

        # Bending forces with natural curvature kappa0 = 20.909
        # Discrete elastic rods / discrete curvature binormal
        # 2 * (e_{i-1} x e_i) / (|e_{i-1}||e_i| + e_{i-1} . e_i)
        # For quasi-static naturally curved rod coiling:
        # Bending moment M = B * (kappa - kappa0)
        # We compute turning angles and natural curvature restoration
        for i in range(1, n - 1):
            t_prev = tangents[i - 1]
            t_curr = tangents[i]
            l_dual = lumped_len[i]

            # Current curvature vector
            dot = np.clip(np.dot(t_prev, t_curr), -1.0, 1.0)
            angle = np.arccos(dot)
            cross = np.cross(t_prev, t_curr)
            norm_cross = np.linalg.norm(cross)

            # Injected rod has natural curvature in x-direction at inlet, winding onto substrate
            # Rest curvature magnitude kappa0
            # Natural bending energy = 0.5 * B * (angle / l_dual - kappa0)^2 * l_dual
            if norm_cross > 1e-6:
                binormal = cross / norm_cross
            else:
                binormal = np.array([0.0, 1.0, 0.0])

            current_kappa = angle / l_dual
            # Curvature error
            d_kappa = current_kappa - self.natural_curvature
            # Moment acting to bend/unbend
            moment = self.B * d_kappa * binormal

            # Forces on nodes i-1, i, i+1 from moment
            # dAngle/dx_i-1, dAngle/dx_i, dAngle/dx_i+1
            f_bend_prev = -moment / max(edge_lens[i - 1], 1e-5)
            f_bend_curr = moment / max(edge_lens[i], 1e-5)
            forces[i - 1] += np.cross(f_bend_prev, t_prev)
            forces[i + 1] += np.cross(f_bend_curr, t_curr)
            forces[i] -= (np.cross(f_bend_prev, t_prev) + np.cross(f_bend_curr, t_curr))

        # Damping forces (quasi-static regime has high viscous drag)
        c_damping = 50.0 * np.sqrt(k_stretch * node_mass[:, None])
        forces -= c_damping * self.velocities

        # Time integration for non-sticking nodes
        # Substrate contact: z <= rod_radius
        contact_z = self.rod_radius
        for i in range(1, n):
            if self.is_deposited[i]:
                # Deposited nodes stick permanently to the substrate
                self.velocities[i] = 0.0
                continue

            # Check if suspended node hits substrate
            if self.nodes[i, 2] <= contact_z:
                self.nodes[i, 2] = contact_z
                self.is_deposited[i] = True
                self.velocities[i] = 0.0
                continue

            # Symplectic Euler update
            acc = forces[i] / max(node_mass[i], 1e-7)
            self.velocities[i] += acc * dt
            self.nodes[i] += self.velocities[i] * dt

            # Substrate penetration clamp
            if self.nodes[i, 2] < contact_z:
                self.nodes[i, 2] = contact_z
                self.is_deposited[i] = True
                self.velocities[i] = 0.0

        # Maintain inextensibility via position relaxation
        for _ in range(5):
            for i in range(n - 1):
                p0 = self.nodes[i]
                p1 = self.nodes[i + 1]
                delta = p1 - p0
                d = np.linalg.norm(delta)
                if d < 1e-8:
                    continue
                err = (d - self.segment_rest_len) / d
                corr = 0.5 * err * delta

                if i == 0 and not self.is_deposited[i + 1]:
                    self.nodes[i + 1] -= 2.0 * corr
                elif not self.is_deposited[i] and self.is_deposited[i + 1]:
                    self.nodes[i] += 2.0 * corr
                elif not self.is_deposited[i] and not self.is_deposited[i + 1]:
                    self.nodes[i] += corr
                    self.nodes[i + 1] -= corr

    def advance_to(self, target_time: float) -> StateFrame:
        while self.current_time < target_time - 1e-9:
            dt = min(self.dt, target_time - self.current_time)
            self._step_physics(dt)
            self.current_time += dt

        return self._build_state_frame(target_time)

    def _build_state_frame(self, time: float) -> StateFrame:
        n = len(self.nodes)

        # Orientations (identity quaternions [x, y, z, w])
        orientation = np.zeros((n, 4), dtype=float)
        orientation[:, 3] = 1.0

        # Lumped lengths and node masses
        diffs = np.diff(self.nodes, axis=0)
        edge_lens = np.linalg.norm(diffs, axis=1)
        lumped_len = np.empty(n)
        lumped_len[[0, -1]] = [edge_lens[0] / 2.0, edge_lens[-1] / 2.0]
        lumped_len[1:-1] = (edge_lens[:-1] + edge_lens[1:]) / 2.0
        mass_quantity = self.linear_density * lumped_len

        # Arc coordinate along centerline
        arc_coord = np.zeros(n, dtype=float)
        arc_coord[1:] = np.cumsum(edge_lens)

        # Required fields according to specification
        budget_ext_force = np.zeros((1, 1, 3), dtype=float)
        budget_ext_force[0, 0] = np.sum(mass_quantity) * self.gravity

        budget_ext_couple = np.zeros((1, 1, 3), dtype=float)

        budget_force_pos = np.zeros((1, 1, 3), dtype=float)
        budget_force_pos[0, 0] = np.mean(self.nodes, axis=0)

        constraint_target = np.array([self.outlet_pos[0], self.outlet_pos[1], self.outlet_pos[2]])
        constraint_value = np.array([self.nodes[0, 0], self.nodes[0, 1], self.nodes[0, 2]])

        cumulative_mass_source = np.array([self.cumulative_mass_injected], dtype=float)

        ref_force = float(np.sum(mass_quantity) * 9.81)
        ref_torque = float(ref_force * 0.05)
        constraint_scale = 1.0

        frame = StateFrame(
            time=time,
            position=self.nodes.copy(),
            velocity=self.velocities.copy(),
            orientation=orientation,
            angular_velocity=np.zeros((n, 3), dtype=float),
            mass=mass_quantity,
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "arc_coordinate": arc_coord,
                "is_injector_controlled": (np.arange(n) == 0).astype(float),
                "is_initially_deposited": self.is_deposited.astype(float),
                "is_sticking": self.is_deposited.astype(float),
                "rod_radius": np.full(n, self.rod_radius),
                "natural_curvature": np.full(n, self.natural_curvature),
                "natural_curvature_direction": np.tile([1.0, 0.0, 0.0], (n, 1)),
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_external_couple": budget_ext_couple,
                "budget_external_force": budget_ext_force,
                "budget_force_position": budget_force_pos,
                "constraint_target": constraint_target,
                "constraint_value": constraint_value,
                "cumulative_mass_source": cumulative_mass_source,
            },
            scalars={
                "injection_speed": float(self.injection_speed),
                "natural_curvature": float(self.natural_curvature),
                "normalized_natural_curvature": float(self.config["regime"]["normalized_natural_curvature"]),
                "substrate_speed": 0.0,
                "constraint_scale": constraint_scale,
                "reference_force": ref_force,
                "reference_torque": ref_torque,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_nodes": len(self.nodes) if self.nodes is not None else 0,
            "simulated_time": self.current_time,
            "cumulative_mass_injected": self.cumulative_mass_injected,
        }

