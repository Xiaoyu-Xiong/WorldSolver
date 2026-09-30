"""Physics solver for two-way coupled elastic bag lifting and water sloshing."""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.config = None
        self.current_time = 0.0

        # State arrays
        self.bag_pos = None  # (N_bag, 3)
        self.bag_vel = None  # (N_bag, 3)
        self.bag_mass = None  # (N_bag,)

        self.water_pos = None  # (N_water, 3)
        self.water_vel = None  # (N_water, 3)
        self.water_mass = None  # (N_water,)

        # Bag mesh topology
        self.bag_faces = None  # (M, 3)
        self.rim_indices = None  # (R,)
        self.is_rim = None  # (N_bag,) bool
        self.edges = None  # (E, 2)
        self.edge_rest_len = None  # (E,)
        self.bending_edges = None  # pairs of opposite vertices across edges
        self.bend_rest_len = None

        # Lift trajectory parameters
        self.lift_start = 0.2
        self.lift_duration = 0.75
        self.lift_start_center = np.array([0.0, 0.0, 1.1])
        self.lift_end_center = np.array([0.0, 0.0, 2.58])
        self.rim_initial_rel = None

        # Simulation boundaries
        self.tank_min = np.array([-1.8, -1.5, 0.0])
        self.tank_max = np.array([1.8, 1.5, 3.2])
        self.gravity = np.array([0.0, 0.0, -9.81])

        # Material & physical properties
        self.rest_density = 1000.0
        self.particle_spacing = 0.1
        self.water_particle_radius = 0.055
        self.bag_particle_radius = 0.008

        # Numerical integration
        self.sub_dt = 0.0025  # 8 substeps per 0.02s sample_dt

        # Budget and diagnostic tracking
        self.cumulative_external_impulse = np.zeros((2, 3), dtype=np.float64)
        self.cumulative_inflow_momentum = np.zeros((2, 3), dtype=np.float64)
        self.cumulative_outflow_momentum = np.zeros((2, 3), dtype=np.float64)
        self.cumulative_mass_source = np.zeros(2, dtype=np.float64)
        self.last_interaction_force = np.zeros((2, 3), dtype=np.float64)
        self.last_budget_external_force = np.zeros((2, 1, 3), dtype=np.float64)

        # Boundary samples
        self._init_boundary_samples()

    def _init_boundary_samples(self) -> None:
        """Initialize representative boundary sample points and normals for the tank."""
        b_pts = []
        b_nrm = []
        # Tank floor (z=0, normal=[0,0,1])
        gx = np.linspace(-1.7, 1.7, 18)
        gy = np.linspace(-1.4, 1.4, 15)
        for x in gx:
            for y in gy:
                b_pts.append([x, y, 0.0])
                b_nrm.append([0.0, 0.0, 1.0])
        # Tank walls
        gz = np.linspace(0.1, 3.1, 16)
        for z in gz:
            for y in gy:
                b_pts.append([-1.8, y, z])
                b_nrm.append([1.0, 0.0, 0.0])
                b_pts.append([1.8, y, z])
                b_nrm.append([-1.0, 0.0, 0.0])
            for x in gx:
                b_pts.append([x, -1.5, z])
                b_nrm.append([0.0, 1.0, 0.0])
                b_pts.append([x, 1.5, z])
                b_nrm.append([0.0, -1.0, 0.0])

        self.boundary_positions = np.array(b_pts, dtype=np.float64)
        self.boundary_normals = np.array(b_nrm, dtype=np.float64)
        self.boundary_velocity = np.zeros_like(self.boundary_positions)
        self.boundary_target_velocity = np.zeros_like(self.boundary_positions)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)

        pos = np.asarray(init_frame.position, dtype=np.float64)
        vel = np.asarray(init_frame.velocity, dtype=np.float64)
        mass = np.asarray(init_frame.mass, dtype=np.float64)
        body_id = np.asarray(init_frame.body_id, dtype=np.int32)

        bag_mask = body_id == 0
        water_mask = body_id == 1

        self.bag_pos = pos[bag_mask].copy()
        self.bag_vel = vel[bag_mask].copy()
        self.bag_mass = mass[bag_mask].copy()

        self.water_pos = pos[water_mask].copy()
        self.water_vel = vel[water_mask].copy()
        self.water_mass = mass[water_mask].copy()

        self.n_bag = len(self.bag_pos)
        self.n_water = len(self.water_pos)

        # Bag topology & rim
        self.bag_faces = np.asarray(scene._bag_faces, dtype=np.int32)
        self.rim_indices = np.asarray(scene._rim_indices, dtype=np.int32)
        self.is_rim = np.zeros(self.n_bag, dtype=bool)
        self.is_rim[self.rim_indices] = True

        # Rim initial offset relative to center
        self.rim_initial_rel = self.bag_pos[self.rim_indices] - self.lift_start_center

        # Build cloth edge springs and bending constraints
        edge_set = set()
        adj_faces = {}
        for face in self.bag_faces:
            for i in range(3):
                u, v = face[i], face[(i + 1) % 3]
                edge = (min(u, v), max(u, v))
                edge_set.add(edge)
                adj_faces.setdefault(edge, []).append(face[(i + 2) % 3])

        self.edges = np.array(list(edge_set), dtype=np.int32)
        p0 = self.bag_pos[self.edges[:, 0]]
        p1 = self.bag_pos[self.edges[:, 1]]
        self.edge_rest_len = np.linalg.norm(p0 - p1, axis=1)

        # Bending springs between unshared vertices of adjacent triangles
        bend_list = []
        for edge, opp_verts in adj_faces.items():
            if len(opp_verts) == 2:
                bend_list.append((opp_verts[0], opp_verts[1]))
        if bend_list:
            self.bending_edges = np.array(bend_list, dtype=np.int32)
            bp0 = self.bag_pos[self.bending_edges[:, 0]]
            bp1 = self.bag_pos[self.bending_edges[:, 1]]
            self.bend_rest_len = np.linalg.norm(bp0 - bp1, axis=1)
        else:
            self.bending_edges = np.zeros((0, 2), dtype=np.int32)
            self.bend_rest_len = np.zeros(0, dtype=np.float64)

        # Read config specifics
        lift_cfg = self.config.get("boundary_conditions", {}).get("rim_lift", {})
        self.lift_start = float(lift_cfg.get("start_time", 0.2))
        self.lift_duration = float(lift_cfg.get("rise_duration", 0.75))
        self.lift_start_center = np.array(lift_cfg.get("start_center", [0.0, 0.0, 1.1]), dtype=np.float64)
        self.lift_end_center = np.array(lift_cfg.get("end_center", [0.0, 0.0, 2.58]), dtype=np.float64)

        pool_cfg = self.config.get("initial_conditions", {}).get("pool", {})
        self.rest_density = float(pool_cfg.get("rest_density", 1000.0))
        self.particle_spacing = float(pool_cfg.get("particle_spacing", 0.1))
        self.water_particle_radius = 0.55 * self.particle_spacing

        grav = self.config.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.array(grav, dtype=np.float64)

    def _rim_trajectory(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute prescribed rim position and velocity at time t."""
        if t <= self.lift_start:
            pos = self.lift_start_center + self.rim_initial_rel
            vel = np.zeros_like(pos)
        elif t >= self.lift_start + self.lift_duration:
            pos = self.lift_end_center + self.rim_initial_rel
            vel = np.zeros_like(pos)
        else:
            # Smooth cubic polynomial (smoothstep / S-curve)
            s = (t - self.lift_start) / self.lift_duration
            s_poly = 3.0 * s**2 - 2.0 * s**3
            s_poly_dot = (6.0 * s - 6.0 * s**2) / self.lift_duration
            disp = self.lift_end_center - self.lift_start_center
            center_pos = self.lift_start_center + s_poly * disp
            center_vel = s_poly_dot * disp
            pos = center_pos + self.rim_initial_rel
            vel = np.tile(center_vel, (len(self.rim_indices), 1))
        return pos, vel

    def _step_physics(self, dt: float) -> None:
        """Physical time step using XPBD / SPH-based coupled dynamics."""
        # 1. Update prescribed rim vertices
        rim_pos, rim_vel = self._rim_trajectory(self.current_time + dt)
        self.bag_vel[self.rim_indices] = rim_vel
        self.bag_pos[self.rim_indices] = rim_pos

        # 2. External forces: Gravity & Aerodynamic / Hydrodynamic drag
        non_rim = ~self.is_rim
        self.bag_vel[non_rim] += self.gravity * dt
        self.water_vel += self.gravity * dt

        # Bag drag
        self.bag_vel[non_rim] *= math.exp(-0.15 * dt)
        self.water_vel *= math.exp(-0.02 * dt)

        # 3. Predict positions
        pred_bag_pos = self.bag_pos.copy()
        pred_bag_pos[non_rim] += self.bag_vel[non_rim] * dt

        pred_water_pos = self.water_pos.copy()
        pred_water_pos += self.water_vel * dt

        # 4. Cloth stretch & bend relaxation (XPBD iterations)
        stretch_compliance = 1e-4 / dt**2
        bend_compliance = 1e-2 / dt**2

        for _ in range(8):
            # Stretch constraints
            p0 = pred_bag_pos[self.edges[:, 0]]
            p1 = pred_bag_pos[self.edges[:, 1]]
            diff = p0 - p1
            dist = np.linalg.norm(diff, axis=1, keepdims=True)
            dist = np.maximum(dist, 1e-7)
            dir_vec = diff / dist
            c = dist.squeeze(-1) - self.edge_rest_len

            w0 = np.where(self.is_rim[self.edges[:, 0]], 0.0, 1.0 / self.bag_mass[self.edges[:, 0]])
            w1 = np.where(self.is_rim[self.edges[:, 1]], 0.0, 1.0 / self.bag_mass[self.edges[:, 1]])
            w_sum = w0 + w1 + stretch_compliance
            valid = w_sum > 1e-12
            delta_lambda = np.zeros_like(c)
            delta_lambda[valid] = -c[valid] / w_sum[valid]

            corr = dir_vec * delta_lambda[:, None]
            # Accumulate corrections safely
            np.add.at(pred_bag_pos, self.edges[:, 0], (w0[:, None] * corr))
            np.add.at(pred_bag_pos, self.edges[:, 1], (-w1[:, None] * corr))
            # Enforce rim boundary position exactly
            pred_bag_pos[self.rim_indices] = rim_pos

            # Bending constraints
            if len(self.bending_edges) > 0:
                bp0 = pred_bag_pos[self.bending_edges[:, 0]]
                bp1 = pred_bag_pos[self.bending_edges[:, 1]]
                bdiff = bp0 - bp1
                bdist = np.linalg.norm(bdiff, axis=1, keepdims=True)
                bdist = np.maximum(bdist, 1e-7)
                bdir = bdiff / bdist
                bc = bdist.squeeze(-1) - self.bend_rest_len

                bw0 = np.where(self.is_rim[self.bending_edges[:, 0]], 0.0, 1.0 / self.bag_mass[self.bending_edges[:, 0]])
                bw1 = np.where(self.is_rim[self.bending_edges[:, 1]], 0.0, 1.0 / self.bag_mass[self.bending_edges[:, 1]])
                bw_sum = bw0 + bw1 + bend_compliance
                bvalid = bw_sum > 1e-12
                b_delta = np.zeros_like(bc)
                b_delta[bvalid] = -bc[bvalid] / bw_sum[bvalid]

                bcorr = bdir * b_delta[:, None]
                np.add.at(pred_bag_pos, self.bending_edges[:, 0], (bw0[:, None] * bcorr))
                np.add.at(pred_bag_pos, self.bending_edges[:, 1], (-bw1[:, None] * bcorr))
                pred_bag_pos[self.rim_indices] = rim_pos

        # 5. Fluid-fluid collision / incompressibility (Position-Based Fluids relaxation)
        h_fluid = self.particle_spacing * 1.05
        # Fast spatial hashing / KD-tree for water neighbor search
        tree_water = cKDTree(pred_water_pos)
        pairs = tree_water.query_pairs(h_fluid, output_type="ndarray")
        if len(pairs) > 0:
            i = pairs[:, 0]
            j = pairs[:, 1]
            p_i = pred_water_pos[i]
            p_j = pred_water_pos[j]
            dp = p_i - p_j
            d = np.linalg.norm(dp, axis=1, keepdims=True)
            d = np.maximum(d, 1e-6)
            overlap = h_fluid - d.squeeze(-1)
            valid_p = overlap > 0
            if np.any(valid_p):
                i_v = i[valid_p]
                j_v = j[valid_p]
                n_ij = dp[valid_p] / d[valid_p]
                f_pbf = 0.5 * overlap[valid_p, None] * n_ij
                np.add.at(pred_water_pos, i_v, f_pbf * 0.4)
                np.add.at(pred_water_pos, j_v, -f_pbf * 0.4)

        # 6. Two-way Bag-Fluid Coupling & Impermeability
        # Pairwise repulsion and traction between water particles and bag vertices / surface
        bag_tree = cKDTree(pred_bag_pos)
        coupling_radius = self.water_particle_radius + 0.04
        water_bag_pairs = bag_tree.query_ball_point(pred_water_pos, r=coupling_radius)

        total_coupl_force_bag = np.zeros(3, dtype=np.float64)
        total_coupl_force_water = np.zeros(3, dtype=np.float64)

        for w_idx, b_indices in enumerate(water_bag_pairs):
            if len(b_indices) == 0:
                continue
            w_p = pred_water_pos[w_idx]
            b_pts = pred_bag_pos[b_indices]
            diff = w_p - b_pts
            dist = np.linalg.norm(diff, axis=1)
            dist_safe = np.maximum(dist, 1e-6)
            overlap = coupling_radius - dist
            pos_overlap = overlap > 0
            if np.any(pos_overlap):
                idxs = np.array(b_indices)[pos_overlap]
                ov = overlap[pos_overlap]
                n_dir = diff[pos_overlap] / dist_safe[pos_overlap, None]
                # Force magnitude proportional to penetration and relative velocity
                stiffness = 25000.0  # N/m equivalent repulsion
                damping = 80.0
                rel_v = self.water_vel[w_idx] - self.bag_vel[idxs]
                v_normal = np.sum(rel_v * n_dir, axis=1)

                f_mag = stiffness * ov - damping * v_normal
                f_mag = np.maximum(f_mag, 0.0)

                f_vec = f_mag[:, None] * n_dir  # Force on water from bag
                # Update positions and velocities
                f_total_w = np.sum(f_vec, axis=0)
                pred_water_pos[w_idx] += (f_total_w / self.water_mass[w_idx]) * (dt**2)
                self.water_vel[w_idx] += (f_total_w / self.water_mass[w_idx]) * dt
                total_coupl_force_water += f_total_w

                # Equal and opposite reaction on bag
                for k, b_id in enumerate(idxs):
                    if not self.is_rim[b_id]:
                        f_on_bag = -f_vec[k]
                        pred_bag_pos[b_id] += (f_on_bag / self.bag_mass[b_id]) * (dt**2)
                        self.bag_vel[b_id] += (f_on_bag / self.bag_mass[b_id]) * dt
                        total_coupl_force_bag += f_on_bag

        # 7. Boundary collisions (Tank floor & walls)
        # Bag boundary collision
        pred_bag_pos[non_rim, 0] = np.clip(pred_bag_pos[non_rim, 0], self.tank_min[0] + 0.02, self.tank_max[0] - 0.02)
        pred_bag_pos[non_rim, 1] = np.clip(pred_bag_pos[non_rim, 1], self.tank_min[1] + 0.02, self.tank_max[1] - 0.02)
        pred_bag_pos[non_rim, 2] = np.clip(pred_bag_pos[non_rim, 2], self.tank_min[2] + 0.02, self.tank_max[2] - 0.02)

        # Water boundary collision with restitution and friction
        r_w = self.water_particle_radius
        for dim in range(3):
            # Min wall
            hit_min = pred_water_pos[:, dim] < (self.tank_min[dim] + r_w)
            pred_water_pos[hit_min, dim] = self.tank_min[dim] + r_w
            self.water_vel[hit_min, dim] = np.maximum(0.0, -0.2 * self.water_vel[hit_min, dim])

            # Max wall (open top for z)
            if dim < 2:
                hit_max = pred_water_pos[:, dim] > (self.tank_max[dim] - r_w)
                pred_water_pos[hit_max, dim] = self.tank_max[dim] - r_w
                self.water_vel[hit_max, dim] = np.minimum(0.0, -0.2 * self.water_vel[hit_max, dim])

        # 8. Update velocities and finalize positions
        self.bag_vel[non_rim] = (pred_bag_pos[non_rim] - self.bag_pos[non_rim]) / dt
        self.bag_pos = pred_bag_pos

        self.water_vel = (pred_water_pos - self.water_pos) / dt
        self.water_pos = pred_water_pos

        # Track external impulse & interaction forces
        ext_f_bag = np.sum(self.bag_mass[:, None] * self.gravity, axis=0)
        ext_f_water = np.sum(self.water_mass[:, None] * self.gravity, axis=0)

        self.cumulative_external_impulse[0] += ext_f_bag * dt
        self.cumulative_external_impulse[1] += ext_f_water * dt

        self.last_budget_external_force[0, 0] = ext_f_bag
        self.last_budget_external_force[1, 0] = ext_f_water

        self.last_interaction_force[0] = total_coupl_force_bag
        self.last_interaction_force[1] = total_coupl_force_water

        self.current_time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward until target_time
        while self.current_time < target_time - 1e-8:
            dt = min(self.sub_dt, target_time - self.current_time)
            self._step_physics(dt)

        # Pack combined arrays
        combined_pos = np.vstack((self.bag_pos, self.water_pos))
        combined_vel = np.vstack((self.bag_vel, self.water_vel))
        combined_mass = np.r_[self.bag_mass, self.water_mass]
        combined_body_id = np.r_[
            np.zeros(self.n_bag, dtype=np.int32),
            np.ones(self.n_water, dtype=np.int32),
        ]

        # Domain budget weights (2 domains: bag=0, water=1)
        budget_domain_weights = np.zeros((2, len(combined_pos)), dtype=np.float64)
        budget_domain_weights[0, :self.n_bag] = 1.0
        budget_domain_weights[1, self.n_bag:] = 1.0

        controlled = np.zeros(len(combined_pos), dtype=np.float64)
        controlled[self.rim_indices] = 1.0

        particle_radius = np.r_[
            np.full(self.n_bag, self.bag_particle_radius, dtype=np.float64),
            np.full(self.n_water, self.water_particle_radius, dtype=np.float64),
        ]

        ref_velocity = 2.5
        ref_momentum = float(np.sum(combined_mass) * ref_velocity)

        frame = StateFrame(
            time=float(target_time),
            position=combined_pos,
            velocity=combined_vel,
            mass=combined_mass,
            body_id=combined_body_id,
            attributes={
                "rim_controlled": controlled,
                "material_phase": combined_body_id.astype(np.float64),
                "particle_radius": particle_radius,
                "mass_quantity": combined_mass.copy(),
            },
            fields={
                "boundary_normal": self.boundary_normals.copy(),
                "boundary_target_velocity": self.boundary_target_velocity.copy(),
                "boundary_velocity": self.boundary_velocity.copy(),
                "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cumulative_inflow_momentum.copy(),
                "budget_cumulative_outflow_momentum": self.cumulative_outflow_momentum.copy(),
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": self.last_budget_external_force.copy(),
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "interaction_force": self.last_interaction_force.copy(),
            },
            scalars={
                "water_rest_density": float(self.rest_density),
                "bag_total_mass": float(np.sum(self.bag_mass)),
                "bag_thickness": float(self.config.get("initial_conditions", {}).get("bag", {}).get("thickness", 0.0015)),
                "initial_free_surface_z": float(self.config.get("initial_conditions", {}).get("pool", {}).get("free_surface_z", 1.25)),
                "reference_velocity": ref_velocity,
                "reference_momentum": ref_momentum,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "n_bag_vertices": self.n_bag if self.bag_pos is not None else 0,
            "n_water_particles": self.n_water if self.water_pos is not None else 0,
        }

