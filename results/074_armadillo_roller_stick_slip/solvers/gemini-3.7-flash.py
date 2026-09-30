"""BenchmarkSolver implementation for 074_armadillo_roller_stick_slip.

Simulates the volumetric elastic Armadillo passing through kinematic rollers with friction,
exhibiting stick-slip instability.
"""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.config = {}
        self.time = 0.0

        # State arrays
        self.x = np.zeros((0, 3), dtype=np.float64)
        self.v = np.zeros((0, 3), dtype=np.float64)
        self.mass = np.zeros(0, dtype=np.float64)
        self.x_rest = np.zeros((0, 3), dtype=np.float64)
        self.body_id = np.zeros(0, dtype=np.int64)

        # Geometry / Elements
        self.num_nodes = 0
        self.num_elements = 0
        self.tetrahedra = np.zeros((0, 4), dtype=np.int64)
        self.inv_Dm = np.zeros((0, 3, 3), dtype=np.float64)
        self.tet_volumes = np.zeros(0, dtype=np.float64)

        # Springs for volumetric elasticity
        self.spring_edges = np.zeros((0, 2), dtype=np.int64)
        self.spring_rest_lens = np.zeros(0, dtype=np.float64)
        self.spring_k = 0.0

        # Rollers configuration
        # 4 rollers: upper-left, upper-right, lower-left, lower-right
        # Cylinders with axis along Y (from y=-0.36 to y=+0.36, radius=0.22)
        # Centers in XZ:
        # UL: (-0.2, 0.3), UR: (0.3, 0.3)
        # LL: (-0.2, -0.3), LR: (0.3, -0.3)
        self.rollers = [
            {"id": "upper_left", "pos": np.array([-0.2, 0.0, 0.3]), "radius": 0.22, "half_h": 0.36, "sign": 1},  # upper -> omega > 0 (turns CW around Y -> top goes -X, bottom/gap goes +X)
            {"id": "upper_right", "pos": np.array([0.3, 0.0, 0.3]), "radius": 0.22, "half_h": 0.36, "sign": 1},
            {"id": "lower_left", "pos": np.array([-0.2, 0.0, -0.3]), "radius": 0.22, "half_h": 0.36, "sign": -1}, # lower -> omega < 0 (turns CCW around Y -> top/gap goes +X, bottom goes -X)
            {"id": "lower_right", "pos": np.array([0.3, 0.0, -0.3]), "radius": 0.22, "half_h": 0.36, "sign": -1},
        ]

        # Kinematic parameters
        self.omega_upper = 5.0
        self.omega_lower = -5.0
        self.start_time = 0.35
        self.ramp_duration = 0.25
        self.mu_friction = 0.5
        self.gravity = np.array([0.0, 0.0, -9.81])

        # Cumulative tracking
        self.cum_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self.last_ext_force = np.zeros(3, dtype=np.float64)

        # Contact state cache for trajectory output
        self.contact_normal_forces = np.zeros(0, dtype=np.float64)
        self.contact_tangential_forces = np.zeros((0, 3), dtype=np.float64)
        self.contact_normals = np.zeros((0, 3), dtype=np.float64)
        self.contact_rel_vels = np.zeros((0, 3), dtype=np.float64)
        self.contact_active_mask = np.zeros(0, dtype=np.int32)

        # Numerical parameters
        self.sub_dt = 0.0005  # sub-stepping time step for stability

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        # Load initial state from scene
        init_frame = scene.initial_state()
        self.time = float(init_frame.time)
        self.x = np.array(init_frame.position, dtype=np.float64)
        self.v = np.array(init_frame.velocity, dtype=np.float64)
        self.mass = np.array(init_frame.mass, dtype=np.float64)
        self.x_rest = np.array(init_frame.attributes.get("rest_position", self.x.copy()), dtype=np.float64)
        self.body_id = np.array(init_frame.body_id, dtype=np.int64)
        self.num_nodes = len(self.x)

        # Configure roller motion parameters
        bc = self.config.get("boundary_conditions", {})
        self.omega_upper = float(bc.get("upper_angular_velocity_rad_s", 5.0))
        self.omega_lower = float(bc.get("lower_angular_velocity_rad_s", -5.0))
        self.start_time = float(bc.get("start_time", 0.35))
        self.ramp_duration = float(bc.get("ramp_duration", 0.25))
        self.mu_friction = float(bc.get("contact_friction_coefficient", 0.5))
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Build tetrahedral mesh / elastic elements from vertices
        self._build_elastic_elements()

        # Reset tracking
        self.cum_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self.last_ext_force = np.zeros(3, dtype=np.float64)
        self._evaluate_contacts(self.time)

    def _build_elastic_elements(self) -> None:
        """Construct a volumetric tetrahedral representation and spring network."""
        n = self.num_nodes

        # Form Delaunay tetrahedralization or synthetic tetrahedra from vertices
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(self.x_rest)
            tets = tri.simplices
        except Exception:
            tets = []
            for i in range(n - 3):
                tets.append([i, i+1, i+2, i+3])
            tets = np.array(tets, dtype=np.int64)

        self.tetrahedra = np.array(tets, dtype=np.int64)
        self.num_elements = len(self.tetrahedra)

        # Compute reference shape matrices inv(D_m) and element volumes
        self.inv_Dm = np.zeros((self.num_elements, 3, 3), dtype=np.float64)
        self.tet_volumes = np.zeros(self.num_elements, dtype=np.float64)

        for e_idx, tet in enumerate(self.tetrahedra):
            p0 = self.x_rest[tet[0]]
            p1 = self.x_rest[tet[1]]
            p2 = self.x_rest[tet[2]]
            p3 = self.x_rest[tet[3]]
            Dm = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
            det = np.linalg.det(Dm)
            if abs(det) > 1e-12:
                self.inv_Dm[e_idx] = np.linalg.inv(Dm)
                self.tet_volumes[e_idx] = abs(det) / 6.0
            else:
                self.inv_Dm[e_idx] = np.eye(3)
                self.tet_volumes[e_idx] = 1e-6

        # Build comprehensive spring network (all Delaunay edges + cross struts)
        edges = set()
        for tet in self.tetrahedra:
            for i in range(4):
                for j in range(i + 1, 4):
                    i1, i2 = min(tet[i], tet[j]), max(tet[i], tet[j])
                    edges.add((i1, i2))

        # Add all pairwise connections within rest-distance threshold for solid elasticity
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.x_rest[i] - self.x_rest[j])
                if dist < 0.35:  # connect nearby nodes
                    edges.add((i, j))

        edge_list = list(edges)
        self.spring_edges = np.array(edge_list, dtype=np.int64)
        self.spring_rest_lens = np.linalg.norm(self.x_rest[self.spring_edges[:, 0]] - self.x_rest[self.spring_edges[:, 1]], axis=1)

        # Young's modulus E ~ 5e5 Pa -> spring stiffness
        # Total mass 2.4 kg, node mass ~ 0.1 kg
        self.spring_k = 15000.0  # N/m per spring

    def _get_roller_omega(self, t: float, sign: int) -> float:
        """Compute roller angular velocity with smooth ramp."""
        target_omega = self.omega_upper if sign > 0 else self.omega_lower
        if t < self.start_time:
            return 0.0
        elif t < self.start_time + self.ramp_duration:
            alpha = (t - self.start_time) / self.ramp_duration
            # Smooth step
            s = alpha * alpha * (3.0 - 2.0 * alpha)
            return target_omega * s
        else:
            return target_omega

    def _compute_deformation_gradients(self) -> np.ndarray:
        """Compute deformation gradient F = D_s * inv(D_m) for each tetrahedron."""
        F_all = np.zeros((self.num_elements, 3, 3), dtype=np.float64)
        for e_idx, tet in enumerate(self.tetrahedra):
            p0 = self.x[tet[0]]
            p1 = self.x[tet[1]]
            p2 = self.x[tet[2]]
            p3 = self.x[tet[3]]
            Ds = np.column_stack([p1 - p0, p2 - p0, p3 - p0])
            F = Ds @ self.inv_Dm[e_idx]
            F_all[e_idx] = F
        return F_all

    def _evaluate_contacts(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute contact kinematics, normal forces, and friction forces for all nodes."""
        n = self.num_nodes
        num_rollers = len(self.rollers)
        total_contacts = n * num_rollers

        normal_forces = np.zeros(total_contacts, dtype=np.float64)
        tangential_forces = np.zeros((total_contacts, 3), dtype=np.float64)
        normals = np.zeros((total_contacts, 3), dtype=np.float64)
        rel_vels = np.zeros((total_contacts, 3), dtype=np.float64)
        active_mask = np.zeros(total_contacts, dtype=np.int32)
        node_forces = np.zeros((n, 3), dtype=np.float64)

        c_idx = 0
        k_contact = 80000.0  # Contact normal spring stiffness N/m
        c_contact = 500.0    # Contact damping

        for r_info in self.rollers:
            r_pos = r_info["pos"]
            radius = r_info["radius"]
            half_h = r_info["half_h"]
            sign = r_info["sign"]
            omega = self._get_roller_omega(t, sign)

            for i in range(n):
                px, py, pz = self.x[i]
                vx, vy, vz = self.v[i]

                # Roller cylinder: axis along Y passing through (r_pos[0], y, r_pos[2])
                dx = px - r_pos[0]
                dz = pz - r_pos[2]
                dist_2d = math.hypot(dx, dz)

                # Check bounding cylinder in Y
                if abs(py - r_pos[1]) <= half_h + 0.1 and dist_2d > 1e-6:
                    nx = dx / dist_2d
                    nz = dz / dist_2d
                    normal = np.array([nx, 0.0, nz])
                    penetration = radius - dist_2d

                    # Roller surface velocity at contact point
                    # Center is (r_pos[0], py, r_pos[2]). Vector from center to surface is (nx*radius, 0, nz*radius)
                    # Angular velocity is omega around Y axis: omega_vec = [0, omega, 0]
                    # v_surf = omega_vec x r_vec = [omega * r_z, 0, -omega * r_x] = [omega * (nz * radius), 0, -omega * (nx * radius)]
                    v_surf_x = omega * (nz * radius)
                    v_surf_z = -omega * (nx * radius)
                    v_surf = np.array([v_surf_x, 0.0, v_surf_z])

                    v_node = np.array([vx, vy, vz])
                    v_rel = v_node - v_surf  # Acting relative to roller

                    normals[c_idx] = normal
                    rel_vels[c_idx] = v_rel

                    if penetration > 0.0:
                        active_mask[c_idx] = 1

                        # Normal spring-damper force on node (pushing outwards along normal)
                        v_n = np.dot(v_node, normal)
                        fn_mag = k_contact * penetration - c_contact * v_n
                        fn_mag = max(0.0, fn_mag)
                        normal_forces[c_idx] = fn_mag
                        f_normal = fn_mag * normal

                        # Tangential velocity
                        v_rel_n = np.dot(v_rel, normal)
                        v_rel_t = v_rel - v_rel_n * normal
                        v_rel_t_norm = np.linalg.norm(v_rel_t)

                        # Friction force (Coulomb with regularized stick-slip)
                        # When v_rel_t is small, dynamic friction resists motion
                        max_friction = self.mu_friction * fn_mag
                        v_tol = 0.01  # friction stick speed tolerance
                        if v_rel_t_norm > 1e-6:
                            # Regularized Coulomb friction opposing relative motion
                            reg = math.tanh(v_rel_t_norm / v_tol)
                            f_tang = -reg * max_friction * (v_rel_t / v_rel_t_norm)
                        else:
                            f_tang = np.zeros(3)

                        tangential_forces[c_idx] = f_tang
                        node_forces[i] += f_normal + f_tang
                    else:
                        active_mask[c_idx] = 0
                else:
                    normals[c_idx] = np.array([0.0, 0.0, 1.0])
                    rel_vels[c_idx] = self.v[i]
                    active_mask[c_idx] = 0

                c_idx += 1

        self.contact_normal_forces = normal_forces
        self.contact_tangential_forces = tangential_forces
        self.contact_normals = normals
        self.contact_rel_vels = rel_vels
        self.contact_active_mask = active_mask

        return node_forces

    def _step(self, dt: float) -> None:
        """Perform sub-step integration with internal elasticity and contact forces."""
        t = self.time
        n = self.num_nodes

        # 1. Internal elastic forces from spring network
        f_internal = np.zeros((n, 3), dtype=np.float64)

        p1 = self.x[self.spring_edges[:, 0]]
        p2 = self.x[self.spring_edges[:, 1]]
        disp = p2 - p1
        curr_lens = np.linalg.norm(disp, axis=1)
        valid = curr_lens > 1e-8

        dirs = np.zeros_like(disp)
        dirs[valid] = disp[valid] / curr_lens[valid, None]

        strain = curr_lens - self.spring_rest_lens
        # Elastic damping
        v1 = self.v[self.spring_edges[:, 0]]
        v2 = self.v[self.spring_edges[:, 1]]
        v_rel = v2 - v1
        v_rel_proj = np.sum(v_rel * dirs, axis=1)
        damping_c = 15.0

        f_mag = self.spring_k * strain + damping_c * v_rel_proj
        f_spring = dirs * f_mag[:, None]

        np.add.at(f_internal, self.spring_edges[:, 0], f_spring)
        np.add.at(f_internal, self.spring_edges[:, 1], -f_spring)

        # 2. Shape matching / volumetric preservation force
        # Center of mass and deformation restoration
        com = np.mean(self.x, axis=0)
        com_rest = np.mean(self.x_rest, axis=0)
        q = self.x_rest - com_rest
        p = self.x - com

        # Approximate rotation using polar decomposition of covariance matrix
        A = q.T @ p
        U, _, Vt = np.linalg.svd(A)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        target_pos = com + q @ R.T
        f_vol = 500.0 * (target_pos - self.x)
        f_internal += f_vol

        # 3. External contact forces
        f_contact = self._evaluate_contacts(t)

        # 4. Gravity force
        f_gravity = self.mass[:, None] * self.gravity[None, :]

        # Total force on nodes
        f_total = f_internal + f_contact + f_gravity

        # Net external force on the armadillo body (contact + gravity)
        ext_force_total = np.sum(f_contact + f_gravity, axis=0)
        self.last_ext_force = ext_force_total
        self.cum_external_impulse[0] += ext_force_total * dt

        # Integration (Symplectic Euler)
        inv_mass = 1.0 / self.mass
        acc = f_total * inv_mass[:, None]

        # Small artificial numerical damping for stability
        self.v = (self.v + acc * dt) * math.exp(-0.1 * dt)
        self.x = self.x + self.v * dt
        self.time += dt

    def advance_to(self, time: float) -> StateFrame:
        # Advance simulation in sub-steps until target time
        target_t = float(time)
        while self.time < target_t - 1e-8:
            dt_step = min(self.sub_dt, target_t - self.time)
            self._step(dt_step)

        # Re-evaluate contact states at target time
        self._evaluate_contacts(self.time)

        # Deformation gradients
        def_grads = self._compute_deformation_gradients()

        # Total mass
        total_mass = float(np.sum(self.mass))

        # Build required fields dictionary
        num_contacts = len(self.contact_normal_forces)

        fields = {
            "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),  # [1, 3]
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),  # [1, 3]
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64), # [1, 3]
            "budget_domain_weights": np.ones((1, self.num_nodes), dtype=np.float64), # [1, N]
            "budget_external_force": self.last_ext_force.reshape(1, 1, 3).copy(),     # [1, 1, 3]
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),                 # [1]
            "element_deformation_gradient": def_grads,                               # [Ne, 3, 3]
            "friction_active_mask": self.contact_active_mask.copy(),                  # [C]
            "friction_dynamic_coefficient": np.full(num_contacts, self.mu_friction, dtype=np.float64), # [C]
            "friction_normal": self.contact_normals.copy(),                          # [C, 3]
            "friction_normal_force": self.contact_normal_forces.copy(),              # [C]
            "friction_relative_velocity": self.contact_rel_vels.copy(),              # [C, 3]
            "friction_static_coefficient": np.full(num_contacts, self.mu_friction, dtype=np.float64),  # [C]
            "friction_tangential_force": self.contact_tangential_forces.copy(),      # [C, 3]
        }

        attributes = {
            "rest_position": self.x_rest.copy(),
            "mass_quantity": self.mass.copy(),
        }

        scalars = {
            "friction_stick_speed_tolerance": 0.01,
            "reference_force": 25.0,
            "reference_momentum": 2.4 * 0.5,
            "reference_velocity": 0.5,
            "roller_friction": self.mu_friction,
            "reference_youngs_modulus_pa": 5.0e5,
        }

        frame = StateFrame(
            time=self.time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "num_nodes": self.num_nodes,
            "num_elements": self.num_elements,
            "contacts_count": len(self.contact_normal_forces),
        }

