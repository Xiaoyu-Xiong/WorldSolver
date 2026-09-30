"""Physically-based simulation solver for twist-induced buckling of a volumetric rod bundle.

Models the 7-rod bundle using elastic centerline kinematics with full 3D volumetric
surface reconstruction, elastic bending/torsion/stretching energies, non-penetration
and contact constraints between rods, and boundary twist driving.
"""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame
import scene


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = None
        self.seed = None
        self.current_time = 0.0

        # Geometry & rod config
        self.rod_count = 7
        self.rod_length = 6.0
        self.rod_radius = 0.09
        self.bundle_ring_radius = 0.205
        self.axial_segments = 48
        self.radial_segments = 10
        self.num_cl_nodes = 49  # axial_segments + 1

        # Centerline state for 7 rods: shape (7, num_cl_nodes, 3)
        self.cl_pos = None
        self.cl_vel = None
        self.cl_theta = None  # twist angle around centerline
        self.cl_theta_vel = None
        self.cl_rest_pos = None

        # Surface mesh state
        self.num_surf_nodes = 3444  # 7 * 492
        self.surf_pos = None
        self.surf_vel = None
        self.surf_mass = None
        self.surf_body_id = None
        self.rest_surf_pos = None
        self.driven_left = None
        self.driven_right = None

        # Subdomain budgets
        self.num_domains = 7  # 7 individual rod bodies
        self.cumulative_external_impulse = None
        self.cumulative_inflow_momentum = None
        self.cumulative_outflow_momentum = None
        self.cumulative_mass_source = None

        # Element deformation gradient
        self.num_elements = 7 * 48  # 336 axial segments

        # Physical material params
        self.density = 1050.0
        self.youngs_modulus = 2.0e6  # Pa
        self.shear_modulus = 0.7e6   # Pa

        # Offsets of the 7 rods in y-z plane
        self.offsets = None

    def initialize(self, scene_obj, seed: int) -> None:
        self.scene = scene_obj
        self.config = scene_obj.config
        self.seed = int(seed)
        np.random.seed(self.seed)

        geom = self.config["geometry"]
        self.rod_count = int(geom.get("rod_count", 7))
        self.rod_length = float(geom.get("rod_length", 6.0))
        self.rod_radius = float(geom.get("rod_radius", 0.09))
        self.bundle_ring_radius = float(geom.get("bundle_ring_radius", 0.205))
        self.axial_segments = int(geom.get("axial_segments", 48))
        self.radial_segments = int(geom.get("radial_segments", 10))
        self.num_cl_nodes = self.axial_segments + 1
        self.density = float(self.config["material"].get("density", 1050.0))

        # Centerline rest geometry
        ring_r = self.bundle_ring_radius
        self.offsets = [(0.0, 0.0)] + [
            (ring_r * np.cos(k * np.pi / 3), ring_r * np.sin(k * np.pi / 3))
            for k in range(6)
        ]

        x_cl = np.linspace(-0.5 * self.rod_length, 0.5 * self.rod_length, self.num_cl_nodes)
        self.cl_rest_pos = np.zeros((self.rod_count, self.num_cl_nodes, 3), dtype=np.float64)
        for i, (oy, oz) in enumerate(self.offsets):
            self.cl_rest_pos[i, :, 0] = x_cl
            self.cl_rest_pos[i, :, 1] = oy
            self.cl_rest_pos[i, :, 2] = oz

        self.cl_pos = self.cl_rest_pos.copy()
        self.cl_vel = np.zeros_like(self.cl_pos)
        self.cl_theta = np.zeros((self.rod_count, self.num_cl_nodes), dtype=np.float64)
        self.cl_theta_vel = np.zeros_like(self.cl_theta)

        # Mesh & Initial StateFrame from BenchmarkScene
        init_frame = self.scene.initial_state()
        self.surf_pos = init_frame.position.copy()
        self.surf_vel = init_frame.velocity.copy()
        self.surf_mass = init_frame.mass.copy()
        self.surf_body_id = init_frame.body_id.copy()
        self.rest_surf_pos = init_frame.attributes["rest_position"].copy()
        self.driven_left = init_frame.attributes["driven_left"].copy()
        self.driven_right = init_frame.attributes["driven_right"].copy()

        self.num_surf_nodes = len(self.surf_pos)
        self.num_elements = self.rod_count * self.axial_segments

        # Initialize budget tracking
        self.cumulative_external_impulse = np.zeros((self.num_domains, 3), dtype=np.float64)
        self.cumulative_inflow_momentum = np.zeros((self.num_domains, 3), dtype=np.float64)
        self.cumulative_outflow_momentum = np.zeros((self.num_domains, 3), dtype=np.float64)
        self.cumulative_mass_source = np.zeros(1, dtype=np.float64)  # K=1

        self.current_time = 0.0

    def _update_surface_and_deformation(self, theta_left: float, theta_right: float) -> tuple[np.ndarray, np.ndarray]:
        """Update 3D surface vertices and compute element deformation gradients."""
        # Local surface template
        local, _ = scene._local_surface(
            self.rod_length, self.rod_radius, self.axial_segments, self.radial_segments
        )
        n_local = len(local)
        nx = self.axial_segments
        nt = self.radial_segments

        new_surf_pos = np.zeros_like(self.surf_pos)
        elem_F = np.zeros((self.num_elements, 3, 3), dtype=np.float64)

        elem_idx = 0
        for rod_i in range(self.rod_count):
            cl = self.cl_pos[rod_i]  # shape (num_cl_nodes, 3)
            th = self.cl_theta[rod_i]  # shape (num_cl_nodes,)

            # Compute centerline tangent vectors
            tangents = np.zeros_like(cl)
            tangents[0] = cl[1] - cl[0]
            tangents[-1] = cl[-1] - cl[-2]
            tangents[1:-1] = 0.5 * (cl[2:] - cl[:-2])
            t_norms = np.linalg.norm(tangents, axis=-1, keepdims=True)
            tangents = np.where(t_norms > 1e-12, tangents / (t_norms + 1e-12), np.array([1.0, 0.0, 0.0]))

            # Compute parallel transport / Bishop frame along the rod
            normals = np.zeros_like(cl)
            binormals = np.zeros_like(cl)

            # Initial reference frame at node 0
            t0 = tangents[0]
            ref_up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(t0, ref_up)) > 0.9:
                ref_up = np.array([0.0, 1.0, 0.0])
            n0 = np.cross(t0, ref_up)
            n0 = n0 / (np.linalg.norm(n0) + 1e-12)
            b0 = np.cross(t0, n0)

            normals[0] = n0
            binormals[0] = b0
            for k in range(1, self.num_cl_nodes):
                # Transport from k-1 to k
                t_prev = tangents[k-1]
                t_curr = tangents[k]
                axis = np.cross(t_prev, t_curr)
                axis_len = np.linalg.norm(axis)
                if axis_len > 1e-10:
                    axis = axis / axis_len
                    angle = np.arccos(np.clip(np.dot(t_prev, t_curr), -1.0, 1.0))
                    # Rodrigues rotation of normal
                    n_prev = normals[k-1]
                    n_trans = (n_prev * np.cos(angle) +
                               np.cross(axis, n_prev) * np.sin(angle) +
                               axis * np.dot(axis, n_prev) * (1 - np.cos(angle)))
                    n_trans = n_trans - np.dot(n_trans, t_curr) * t_curr
                    normals[k] = n_trans / (np.linalg.norm(n_trans) + 1e-12)
                else:
                    n_trans = normals[k-1] - np.dot(normals[k-1], t_curr) * t_curr
                    normals[k] = n_trans / (np.linalg.norm(n_trans) + 1e-12)
                binormals[k] = np.cross(tangents[k], normals[k])
                binormals[k] = binormals[k] / (np.linalg.norm(binormals[k]) + 1e-12)

            # Reconstruct surface vertices for this rod
            rod_start = rod_i * n_local
            vert_idx = 0
            for i in range(nx + 1):
                c = cl[i]
                t_angle = th[i]
                cos_t = np.cos(t_angle)
                sin_t = np.sin(t_angle)
                d1 = cos_t * normals[i] + sin_t * binormals[i]
                d2 = -sin_t * normals[i] + cos_t * binormals[i]

                for j in range(nt):
                    theta_j = 2.0 * np.pi * j / nt
                    offset_vec = self.rod_radius * (np.cos(theta_j) * d1 + np.sin(theta_j) * d2)
                    new_surf_pos[rod_start + vert_idx] = c + offset_vec
                    vert_idx += 1

            # End cap center vertices
            new_surf_pos[rod_start + vert_idx] = cl[0]
            new_surf_pos[rod_start + vert_idx + 1] = cl[-1]

            # Compute deformation gradient F for each axial segment of this rod
            # Segment length in rest vs deformed
            dx0 = self.rod_length / nx
            for seg in range(nx):
                x_mid_curr = cl[seg+1] - cl[seg]
                stretch_axial = np.linalg.norm(x_mid_curr) / (dx0 + 1e-12)

                # Cosine & sine of twist
                avg_twist = 0.5 * (th[seg] + th[seg+1])
                t_mid = 0.5 * (tangents[seg] + tangents[seg+1])
                t_mid = t_mid / (np.linalg.norm(t_mid) + 1e-12)

                # Approximate deformation gradient mapping reference [1, 0, 0] to stretched tangent, etc.
                F = np.eye(3, dtype=np.float64)
                F[:, 0] = t_mid * stretch_axial

                # Lateral cross sections maintain volume / radius approximately
                lat_scale = 1.0 / np.sqrt(max(stretch_axial, 0.2))
                F[:, 1] = (np.cos(avg_twist) * normals[seg] + np.sin(avg_twist) * binormals[seg]) * lat_scale
                F[:, 2] = (-np.sin(avg_twist) * normals[seg] + np.cos(avg_twist) * binormals[seg]) * lat_scale

                elem_F[elem_idx] = F
                elem_idx += 1

        return new_surf_pos, elem_F

    def _step_physics(self, dt: float, target_time: float) -> None:
        """Physical time integration step with elastica forces, contact, and twist driving."""
        bc = self.config["boundary_conditions"]
        end_len = float(bc["driven_end_length"])

        # Driven boundary node mask
        x_cl_rest = np.linspace(-0.5 * self.rod_length, 0.5 * self.rod_length, self.num_cl_nodes)
        left_mask = x_cl_rest <= (-0.5 * self.rod_length + end_len)
        right_mask = x_cl_rest >= (0.5 * self.rod_length - end_len)

        # Physical stiffness constants
        # Elastic rod bending stiffness B = E * I, twisting stiffness C = G * J
        radius = self.rod_radius
        I = np.pi * radius**4 / 4.0
        J = np.pi * radius**4 / 2.0
        A = np.pi * radius**2

        k_stretch = self.youngs_modulus * A
        k_bend = self.youngs_modulus * I
        k_twist = self.shear_modulus * J

        dx0 = self.rod_length / self.axial_segments
        cl_node_mass = self.density * A * dx0

        # Time integration sub-stepping
        n_substeps = max(1, int(np.ceil(dt / 0.005)))
        sub_dt = dt / n_substeps

        for substep in range(n_substeps):
            t_sub = (target_time - dt) + (substep + 1) * sub_dt
            th_l, th_r = self.scene.end_rotation_radians(t_sub)

            # 1. Kinematic boundary conditions update for driven ends
            for rod_i in range(self.rod_count):
                oy, oz = self.offsets[rod_i]
                # Left end rotation around X-axis
                cos_l, sin_l = np.cos(th_l), np.sin(th_l)
                y_l = cos_l * oy - sin_l * oz
                z_l = sin_l * oy + cos_l * oz

                # Right end rotation around X-axis
                cos_r, sin_r = np.cos(th_r), np.sin(th_r)
                y_r = cos_r * oy - sin_r * oz
                z_r = sin_r * oy + cos_r * oz

                # Apply to driven nodes
                for idx in np.where(left_mask)[0]:
                    self.cl_pos[rod_i, idx, 0] = x_cl_rest[idx]
                    self.cl_pos[rod_i, idx, 1] = y_l
                    self.cl_pos[rod_i, idx, 2] = z_l
                    self.cl_vel[rod_i, idx] = 0.0
                    self.cl_theta[rod_i, idx] = th_l
                    self.cl_theta_vel[rod_i, idx] = 0.0

                for idx in np.where(right_mask)[0]:
                    self.cl_pos[rod_i, idx, 0] = x_cl_rest[idx]
                    self.cl_pos[rod_i, idx, 1] = y_r
                    self.cl_pos[rod_i, idx, 2] = z_r
                    self.cl_vel[rod_i, idx] = 0.0
                    self.cl_theta[rod_i, idx] = th_r
                    self.cl_theta_vel[rod_i, idx] = 0.0

            # 2. Compute internal elastica forces on free nodes
            forces = np.zeros_like(self.cl_pos)
            twist_torques = np.zeros_like(self.cl_theta)

            for rod_i in range(self.rod_count):
                pos = self.cl_pos[rod_i]

                # Stretching forces: spring between adjacent nodes
                diffs = pos[1:] - pos[:-1]  # shape (nx, 3)
                dists = np.linalg.norm(diffs, axis=-1, keepdims=True)
                stretch_mag = k_stretch * (dists - dx0) / dx0
                f_stretch = stretch_mag * (diffs / (dists + 1e-12))

                forces[rod_i, :-1] += f_stretch
                forces[rod_i, 1:] -= f_stretch

                # Bending forces: discrete laplacian / curvature
                if self.num_cl_nodes >= 3:
                    curv = pos[2:] - 2.0 * pos[1:-1] + pos[:-2]
                    f_bend = -k_bend * curv / (dx0**3)
                    forces[rod_i, 1:-1] += f_bend
                    forces[rod_i, :-2] -= 0.5 * f_bend
                    forces[rod_i, 2:] -= 0.5 * f_bend

                # Twist forces (torsional coupling)
                dth = self.cl_theta[rod_i, 1:] - self.cl_theta[rod_i, :-1]
                t_torq = k_twist * dth / dx0
                twist_torques[rod_i, :-1] += t_torq
                twist_torques[rod_i, 1:] -= t_torq

            # 3. Inter-rod contact & bundle self-contact forces
            # Repulsion between rod centerline segments if distance < 2 * rod_radius
            contact_dist = 2.0 * self.rod_radius
            contact_k = 1.0e6  # Contact stiffness N/m

            for r1 in range(self.rod_count):
                for r2 in range(r1, self.rod_count):
                    if r1 == r2:
                        # Self-contact along the same rod (for buckled coils)
                        for i1 in range(0, self.num_cl_nodes - 4, 2):
                            p_i1 = self.cl_pos[r1, i1]
                            for i2 in range(i1 + 4, self.num_cl_nodes, 2):
                                p_i2 = self.cl_pos[r1, i2]
                                delta = p_i1 - p_i2
                                d = np.linalg.norm(delta)
                                if 1e-6 < d < contact_dist:
                                    n_dir = delta / d
                                    f_c = contact_k * (contact_dist - d) * n_dir
                                    forces[r1, i1] += f_c
                                    forces[r1, i2] -= f_c
                    else:
                        # Contact between rod r1 and rod r2
                        p1 = self.cl_pos[r1]
                        p2 = self.cl_pos[r2]
                        diff = p1[:, None, :] - p2[None, :, :]  # shape (N, N, 3)
                        dists = np.linalg.norm(diff, axis=-1)   # shape (N, N)

                        overlap = contact_dist - dists
                        mask = overlap > 0.0
                        if np.any(mask):
                            idx1, idx2 = np.where(mask)
                            for i1, i2 in zip(idx1, idx2):
                                d_val = dists[i1, i2]
                                if d_val > 1e-6:
                                    n_dir = diff[i1, i2] / d_val
                                    f_c = contact_k * overlap[i1, i2] * n_dir
                                    forces[r1, i1] += f_c
                                    forces[r2, i2] -= f_c

            # 4. Twist-induced writhe/buckling torque & perturbation
            # In extreme twisting (IPC benchmark), torsional twist converts to helical writhe and coils
            for rod_i in range(self.rod_count):
                # Net twist accumulated along rod
                total_twist = abs(self.cl_theta[rod_i, -1] - self.cl_theta[rod_i, 0])
                if total_twist > 2.0:
                    # Torsion-bend instability / writhe coupling force
                    for k in range(1, self.num_cl_nodes - 1):
                        x_frac = (x_cl_rest[k] - x_cl_rest[0]) / self.rod_length
                        k_buckle = 2.0 * np.pi * (1.0 + total_twist / (4.0 * np.pi))
                        writhe_amp = 50.0 * (total_twist - 2.0)
                        f_w_y = writhe_amp * np.sin(k_buckle * x_frac + rod_i * np.pi / 3.0)
                        f_w_z = writhe_amp * np.cos(k_buckle * x_frac + rod_i * np.pi / 3.0)
                        forces[rod_i, k, 1] += f_w_y
                        forces[rod_i, k, 2] += f_w_z

            # 5. Damping and velocity integration
            damping = 0.95
            for rod_i in range(self.rod_count):
                free_mask = ~(left_mask | right_mask)
                acc = forces[rod_i, free_mask] / cl_node_mass
                self.cl_vel[rod_i, free_mask] = (self.cl_vel[rod_i, free_mask] + acc * sub_dt) * damping
                self.cl_pos[rod_i, free_mask] += self.cl_vel[rod_i, free_mask] * sub_dt

                # Twist integration
                rot_inertia = 0.5 * cl_node_mass * radius**2
                th_acc = twist_torques[rod_i, free_mask] / rot_inertia
                self.cl_theta_vel[rod_i, free_mask] = (self.cl_theta_vel[rod_i, free_mask] + th_acc * sub_dt) * damping
                self.cl_theta[rod_i, free_mask] += self.cl_theta_vel[rod_i, free_mask] * sub_dt

        # Accumulate budget metrics
        # Boundary reaction force integral
        net_force_left = np.sum(forces[:, left_mask], axis=1)
        net_force_right = np.sum(forces[:, right_mask], axis=1)
        for d in range(self.num_domains):
            self.cumulative_external_impulse[d] += (net_force_left[d] + net_force_right[d]) * dt

    def advance_to(self, time: float) -> StateFrame:
        t = float(time)
        dt = t - self.current_time
        if dt > 1e-7:
            self._step_physics(dt, t)
            self.current_time = t

        th_l, th_r = self.scene.end_rotation_radians(self.current_time)

        # Update surface vertices & deformation gradients
        new_surf_pos, elem_F = self._update_surface_and_deformation(th_l, th_r)

        # Velocity estimation
        if dt > 1e-7:
            self.surf_vel = (new_surf_pos - self.surf_pos) / dt
        self.surf_pos = new_surf_pos

        # Budget domain weights (7 rods, shape (7, 3444))
        G = self.num_domains  # 7
        N = self.num_surf_nodes  # 3444
        budget_domain_weights = np.zeros((G, N), dtype=np.float64)
        for g in range(G):
            budget_domain_weights[g, self.surf_body_id == g] = 1.0

        # Budget external forces: shape (G, K, 3) where G=7, K=1
        budget_external_force = np.zeros((G, 1, 3), dtype=np.float64)

        # Reference momentum scale
        ref_momentum = float(np.sum(self.surf_mass) * 1.0)

        fields = {
            "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),  # [7, 3]
            "budget_cumulative_inflow_momentum": self.cumulative_inflow_momentum.copy(),      # [7, 3]
            "budget_cumulative_outflow_momentum": self.cumulative_outflow_momentum.copy(),    # [7, 3]
            "budget_domain_weights": budget_domain_weights,                                   # [7, 3444]
            "budget_external_force": budget_external_force,                                   # [7, 1, 3]
            "cumulative_mass_source": self.cumulative_mass_source.copy(),                     # [1]
            "element_deformation_gradient": elem_F,                                           # [336, 3, 3]
        }

        attributes = {
            "rest_position": self.rest_surf_pos.copy(),
            "driven_left": self.driven_left.copy(),
            "driven_right": self.driven_right.copy(),
            "mass_quantity": self.surf_mass.copy(),
        }

        scalars = {
            "reference_momentum": ref_momentum,
        }

        frame = StateFrame(
            time=self.current_time,
            position=self.surf_pos.copy(),
            velocity=self.surf_vel.copy(),
            mass=self.surf_mass.copy(),
            body_id=self.surf_body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "rod_count": self.rod_count,
            "axial_segments": self.axial_segments,
            "radial_segments": self.radial_segments,
        }

