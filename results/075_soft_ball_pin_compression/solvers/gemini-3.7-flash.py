"""Physics-based FEM / Non-linear Elasticity Solver with Pin-Array Contact for Soft Ball Compression."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Load ball volume topology & rest vertices
        root = Path(scene._root)
        ball_data = json.loads((root / "assets" / "soft_ball_volume.json").read_text())
        self.rest_x = np.asarray(ball_data["vertices"], dtype=float)
        self.tets = np.asarray(ball_data["tetrahedra"], dtype=int)
        self.tris = np.asarray(ball_data.get("surface_triangles", ball_data.get("faces", [])), dtype=int)
        self.n_nodes = len(self.rest_x)
        self.n_tets = len(self.tets)

        # Load pin arrays
        lower_data = json.loads((root / "assets" / "lower_pins.json").read_text())
        upper_data = json.loads((root / "assets" / "upper_pins.json").read_text())

        self.lower_pin_verts = np.asarray(lower_data["vertices"], dtype=float)
        self.lower_pin_edges = np.asarray(lower_data["edges"], dtype=int)
        self.upper_pin_verts_rest = np.asarray(upper_data["vertices"], dtype=float)
        self.upper_pin_edges = np.asarray(upper_data["edges"], dtype=int)

        self.pin_radius = 0.009  # from scene_geometry.yaml

        # Physical constants
        self.density = float(self.config["initial_conditions"]["ball_density"])  # 980 kg/m^3
        self.gravity = np.asarray(self.config["gravity"], dtype=float)  # [0, 0, -9.81]

        # Elastic material properties (Neo-Hookean / St. Venant-Kirchhoff)
        # Soft rubber: Young's modulus E ~ 8e4 Pa, Poisson's ratio nu ~ 0.45
        self.youngs_modulus = 9.0e4
        self.poissons_ratio = 0.42
        self.mu = self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))
        self.lmbda = (self.youngs_modulus * self.poissons_ratio) / (
            (1.0 + self.poissons_ratio) * (1.0 - 2.0 * self.poissons_ratio)
        )

        # Precompute tet rest shapes (Dm_inv) and reference volumes
        self.Dm_inv = np.zeros((self.n_tets, 3, 3), dtype=float)
        self.rest_vol = np.zeros(self.n_tets, dtype=float)
        self.mass = np.zeros(self.n_nodes, dtype=float)

        for e in range(self.n_tets):
            i0, i1, i2, i3 = self.tets[e]
            X0 = self.rest_x[i0]
            X1 = self.rest_x[i1]
            X2 = self.rest_x[i2]
            X3 = self.rest_x[i3]
            Dm = np.column_stack([X1 - X0, X2 - X0, X3 - X0])
            detDm = np.linalg.det(Dm)
            vol = abs(detDm) / 6.0
            self.rest_vol[e] = vol
            self.Dm_inv[e] = np.linalg.inv(Dm)
            m_node = self.density * vol / 4.0
            self.mass[self.tets[e]] += m_node

        self.total_mass = np.sum(self.mass)

        # Initial conditions
        init_v = np.asarray(self.config["initial_conditions"]["ball_velocity"], dtype=float)
        self.x = self.rest_x.copy()
        self.v = np.tile(init_v, (self.n_nodes, 1))

        # Kinematic keyframes for upper pins
        bc_upper = self.config["boundary_conditions"]["upper_pin_array"]
        self.upper_axis = np.asarray(bc_upper["axis"], dtype=float)
        self.upper_keyframes = [(k["time"], k["displacement"]) for k in bc_upper["keyframes"]]

        # State tracking
        self.current_time = 0.0
        self.sub_dt = 1e-4  # fine substep for contact stability
        self.budget_cumulative_external_impulse = np.zeros((1, 3), dtype=float)
        self.budget_cumulative_inflow_momentum = np.zeros((1, 3), dtype=float)
        self.budget_cumulative_outflow_momentum = np.zeros((1, 3), dtype=float)
        self.cumulative_mass_source = np.zeros(1, dtype=float)
        self.reference_momentum = max(self.total_mass * np.linalg.norm(self.gravity) * 0.5, 1.0)

        # Damping parameter
        self.damping_coeff = 25.0

    def get_upper_pin_displacement(self, t: float) -> float:
        keyframes = self.upper_keyframes
        if t <= keyframes[0][0]:
            return keyframes[0][1]
        for i in range(len(keyframes) - 1):
            t0, d0 = keyframes[i]
            t1, d1 = keyframes[i + 1]
            if t0 <= t <= t1:
                alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return d0 + alpha * (d1 - d0)
        return keyframes[-1][1]

    def compute_internal_forces_and_F(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        forces = np.zeros((self.n_nodes, 3), dtype=float)
        F_list = np.zeros((self.n_tets, 3, 3), dtype=float)

        for e in range(self.n_tets):
            i0, i1, i2, i3 = self.tets[e]
            x0 = x[i0]
            x1 = x[i1]
            x2 = x[i2]
            x3 = x[i3]

            Ds = np.column_stack([x1 - x0, x2 - x0, x3 - x0])
            Dm_inv = self.Dm_inv[e]
            F = Ds @ Dm_inv
            F_list[e] = F

            # St. Venant-Kirchhoff / Neo-Hookean strain & first Piola-Kirchhoff stress P
            # Green strain: E = 0.5 * (F^T F - I)
            C = F.T @ F
            I3 = np.eye(3)
            E = 0.5 * (C - I3)
            # Second Piola-Kirchhoff stress S = 2*mu*E + lambda*tr(E)*I
            S = 2.0 * self.mu * E + self.lmbda * np.trace(E) * I3
            # First Piola-Kirchhoff stress P = F * S
            P = F @ S

            vol = self.rest_vol[e]
            # Forces on nodes 1, 2, 3: H = -vol * P * Dm_inv^T
            H = -vol * (P @ Dm_inv.T)
            f1 = H[:, 0]
            f2 = H[:, 1]
            f3 = H[:, 2]
            f0 = -(f1 + f2 + f3)

            forces[i0] += f0
            forces[i1] += f1
            forces[i2] += f2
            forces[i3] += f3

        return forces, F_list

    def step_physics(self, dt: float, current_t: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance physics by one substep dt using symplectic Euler with IPC-style barrier contact."""
        # 1. Internal elastic forces & deformation gradients
        f_int, F_list = self.compute_internal_forces_and_F(self.x)

        # 2. Gravity forces
        f_grav = self.mass[:, None] * self.gravity[None, :]

        # 3. Contact forces with pin arrays
        f_contact = np.zeros_like(self.x)

        # Pin geometry at current time
        upper_disp = self.get_upper_pin_displacement(current_t)
        upper_pin_verts = self.upper_pin_verts_rest + self.upper_axis[None, :] * upper_disp

        contact_radius = self.pin_radius + 0.015  # effective collision margin
        k_contact = 5.0e5  # penalty contact stiffness
        c_contact = 500.0  # contact damping

        # Check contact between ball nodes and lower pins (and lower base)
        for i in range(self.n_nodes):
            pos = self.x[i]
            vel = self.v[i]

            # Lower pin array contacts
            for edge in self.lower_pin_edges:
                p0 = self.lower_pin_verts[edge[0]]
                p1 = self.lower_pin_verts[edge[1]]
                seg = p1 - p0
                seg_len2 = np.dot(seg, seg)
                if seg_len2 > 1e-12:
                    t_proj = np.clip(np.dot(pos - p0, seg) / seg_len2, 0.0, 1.0)
                    closest = p0 + t_proj * seg
                else:
                    closest = p0
                diff = pos - closest
                dist = np.linalg.norm(diff)
                if dist < contact_radius and dist > 1e-9:
                    normal = diff / dist
                    overlap = contact_radius - dist
                    v_rel = np.dot(vel, normal)
                    f_n = (k_contact * overlap - c_contact * min(0.0, v_rel)) * normal
                    f_contact[i] += f_n

            # Upper pin array contacts
            for edge in self.upper_pin_edges:
                p0 = upper_pin_verts[edge[0]]
                p1 = upper_pin_verts[edge[1]]
                seg = p1 - p0
                seg_len2 = np.dot(seg, seg)
                if seg_len2 > 1e-12:
                    t_proj = np.clip(np.dot(pos - p0, seg) / seg_len2, 0.0, 1.0)
                    closest = p0 + t_proj * seg
                else:
                    closest = p0
                diff = pos - closest
                dist = np.linalg.norm(diff)
                if dist < contact_radius and dist > 1e-9:
                    normal = diff / dist
                    overlap = contact_radius - dist
                    # pin moving velocity
                    # approximate velocity from keyframes if moving
                    v_rel = np.dot(vel, normal)
                    f_n = (k_contact * overlap - c_contact * min(0.0, v_rel)) * normal
                    f_contact[i] += f_n

        # 4. Total forces and integration
        # Damping
        f_damp = -self.damping_coeff * self.mass[:, None] * self.v
        f_total = f_int + f_grav + f_contact + f_damp

        # Acceleration
        a = f_total / self.mass[:, None]

        # Symplectic Euler update
        self.v += a * dt
        self.x += self.v * dt

        # External forces for momentum budget: gravity + contact
        f_ext = f_grav + f_contact
        return f_ext, F_list

    def advance_to(self, time: float) -> StateFrame:
        if time < self.current_time:
            pass  # return current state if already advanced
        else:
            time_to_advance = time - self.current_time
            if time_to_advance > 1e-9:
                n_steps = max(1, int(np.ceil(time_to_advance / self.sub_dt)))
                dt = time_to_advance / n_steps
                for _ in range(n_steps):
                    f_ext, _ = self.step_physics(dt, self.current_time)
                    # Accumulate external impulse
                    total_f_ext = np.sum(f_ext, axis=0)  # [3]
                    self.budget_cumulative_external_impulse[0] += total_f_ext * dt
                    self.current_time += dt

        self.current_time = time

        # Compute current deformation gradients for export
        _, F_list = self.compute_internal_forces_and_F(self.x)

        # Budget external forces at the current snapshot
        f_grav = self.mass[:, None] * self.gravity[None, :]
        upper_disp = self.get_upper_pin_displacement(self.current_time)
        upper_pin_verts = self.upper_pin_verts_rest + self.upper_axis[None, :] * upper_disp
        contact_radius = self.pin_radius + 0.015
        k_contact = 5.0e5
        c_contact = 500.0
        f_contact = np.zeros_like(self.x)

        for i in range(self.n_nodes):
            pos = self.x[i]
            vel = self.v[i]
            for edge in self.lower_pin_edges:
                p0 = self.lower_pin_verts[edge[0]]
                p1 = self.lower_pin_verts[edge[1]]
                seg = p1 - p0
                seg_len2 = np.dot(seg, seg)
                if seg_len2 > 1e-12:
                    t_proj = np.clip(np.dot(pos - p0, seg) / seg_len2, 0.0, 1.0)
                    closest = p0 + t_proj * seg
                else:
                    closest = p0
                diff = pos - closest
                dist = np.linalg.norm(diff)
                if dist < contact_radius and dist > 1e-9:
                    normal = diff / dist
                    overlap = contact_radius - dist
                    v_rel = np.dot(vel, normal)
                    f_contact[i] += (k_contact * overlap - c_contact * min(0.0, v_rel)) * normal

            for edge in self.upper_pin_edges:
                p0 = upper_pin_verts[edge[0]]
                p1 = upper_pin_verts[edge[1]]
                seg = p1 - p0
                seg_len2 = np.dot(seg, seg)
                if seg_len2 > 1e-12:
                    t_proj = np.clip(np.dot(pos - p0, seg) / seg_len2, 0.0, 1.0)
                    closest = p0 + t_proj * seg
                else:
                    closest = p0
                diff = pos - closest
                dist = np.linalg.norm(diff)
                if dist < contact_radius and dist > 1e-9:
                    normal = diff / dist
                    overlap = contact_radius - dist
                    v_rel = np.dot(vel, normal)
                    f_contact[i] += (k_contact * overlap - c_contact * min(0.0, v_rel)) * normal

        # Build budget_external_force shape [G, K, 3] = [1, 1, 3]
        total_external_force = np.sum(f_grav + f_contact, axis=0, keepdims=True)  # [1, 3]
        budget_external_force = total_external_force[None, :, :]  # [1, 1, 3]

        # Domain weights: shape [G, N] = [1, N]
        budget_domain_weights = np.ones((1, self.n_nodes), dtype=float)

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "rest_position": self.rest_x.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.budget_cumulative_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.budget_cumulative_inflow_momentum.copy(),
                "budget_cumulative_outflow_momentum": self.budget_cumulative_outflow_momentum.copy(),
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "element_deformation_gradient": F_list.copy(),
            },
            scalars={
                "density": float(self.density),
                "reference_momentum": float(self.reference_momentum),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_nodes": self.n_nodes,
            "n_tets": self.n_tets,
            "total_mass": float(self.total_mass),
        }

