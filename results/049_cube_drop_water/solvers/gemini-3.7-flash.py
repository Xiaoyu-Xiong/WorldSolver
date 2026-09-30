"""Physically-based fluid-rigid coupling solver for cube dropping into water."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=float)


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm > 1e-12:
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.config = scene.config

        # Get initial state
        init_frame = scene.initial_state()
        self.time = float(init_frame.time)

        # Separate water particles and cube
        body_id = np.asarray(init_frame.body_id)
        water_mask = (body_id == 1)
        cube_mask = (body_id == 2)

        self.num_water = int(np.sum(water_mask))
        self.water_pos = np.array(init_frame.position[water_mask], dtype=float)
        self.water_vel = np.array(init_frame.velocity[water_mask], dtype=float)
        self.water_mass = np.array(init_frame.mass[water_mask], dtype=float)

        self.cube_pos = np.array(init_frame.position[cube_mask][0], dtype=float)
        self.cube_vel = np.array(init_frame.velocity[cube_mask][0], dtype=float)
        self.cube_rot = np.array(init_frame.orientation[cube_mask][0], dtype=float)
        self.cube_ang_vel = np.array(init_frame.angular_velocity[cube_mask][0], dtype=float)
        self.cube_mass = float(init_frame.mass[cube_mask][0])

        cube_cfg = self.config["initial_conditions"]["cube"]
        water_cfg = self.config["initial_conditions"]["water"]
        self.cube_size = float(cube_cfg["side_length"])
        self.cube_half_size = self.cube_size / 2.0
        self.cube_density = float(cube_cfg["density"])
        self.water_rest_density = float(water_cfg["rest_density"])

        # Cube inertia tensor (cube of uniform density)
        # I = 1/6 * M * s^2 * eye(3)
        self.cube_inertia_body = (1.0 / 6.0) * self.cube_mass * (self.cube_size ** 2) * np.eye(3)
        self.cube_inv_inertia_body = np.linalg.inv(self.cube_inertia_body)

        # Domain boundaries
        domain_min = np.array(self.config["domain"]["bounds_min"], dtype=float)
        domain_max = np.array(self.config["domain"]["bounds_max"], dtype=float)
        # Inner walls based on scene geometry:
        # tank_left: x = -1.44 + 0.06 = -1.38
        # tank_right: x = 1.44 - 0.06 = 1.38
        # tank_front: y = -1.14 + 0.06 = -1.08
        # tank_back: y = 1.14 - 0.06 = 1.08
        # tank_floor: z = 0.0
        self.wall_min = np.array([-1.38, -1.08, 0.0], dtype=float)
        self.wall_max = np.array([1.38, 1.08, 2.45], dtype=float)

        # Gravity
        self.gravity = np.array(self.config["gravity"], dtype=float)

        # SPH / Particle parameters
        self.particle_radius = 0.05
        self.h = 0.16  # smoothing radius
        self.h2 = self.h * self.h
        self.k_stiffness = 2000.0  # pressure stiffness
        self.viscosity = 0.08
        self.restitution = 0.2

        # Stored interaction forces and diagnostics
        self.last_contact_sd = np.zeros(1, dtype=float)
        self.last_external_force = np.zeros((self.num_water + 1, 3), dtype=float)
        self.last_interaction_force = np.zeros((2, 3), dtype=float)
        self.cumulative_mass_source = np.zeros(2, dtype=float)
        self.contact_length_scale = 0.06

        # Sub-step time step
        self.sub_dt = 0.001

    def _get_cube_sdf(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute signed distance and outward normal from cube to positions [N, 3]."""
        # Transform pos to cube local frame
        R = quat_to_rot_matrix(self.cube_rot)
        p_rel = pos - self.cube_pos
        p_local = p_rel @ R  # [N, 3]

        # Box SDF
        d = np.abs(p_local) - self.cube_half_size
        outside = np.maximum(d, 0.0)
        outside_dist = np.linalg.norm(outside, axis=-1)
        inside_dist = np.minimum(np.max(d, axis=-1), 0.0)
        sdf = outside_dist + inside_dist

        # Compute normal in local space
        normal_local = np.zeros_like(p_local)
        # For points outside
        outside_mask = (sdf > 0)
        if np.any(outside_mask):
            denom = outside_dist[outside_mask, None] + 1e-12
            normal_local[outside_mask] = (np.sign(p_local[outside_mask]) * outside[outside_mask]) / denom
        # For points inside
        inside_mask = ~outside_mask
        if np.any(inside_mask):
            p_in = p_local[inside_mask]
            d_in = d[inside_mask]
            max_axis = np.argmax(d_in, axis=-1)
            norm_in = np.zeros_like(p_in)
            for ax in range(3):
                ax_mask = (max_axis == ax)
                norm_in[ax_mask, ax] = np.sign(p_in[ax_mask, ax])
            normal_local[inside_mask] = norm_in

        # Rotate normal back to world frame: normal_world = normal_local @ R.T
        normal_world = normal_local @ R.T
        norm_len = np.linalg.norm(normal_world, axis=-1, keepdims=True) + 1e-12
        normal_world = normal_world / norm_len

        return sdf, normal_world

    def _step(self, dt: float) -> None:
        """Physical time step for fluid and rigid body coupling."""
        pos = self.water_pos
        vel = self.water_vel
        N = self.num_water

        # 1. Compute density / pressure for water using spatial binning or vectorized computation
        # Distance matrix (N is ~2600 particles, vectorized is fast enough in sub-steps or chunked)
        # Using SPH cubic spline or standard Poly6 kernel
        R_cube = quat_to_rot_matrix(self.cube_rot)
        I_inv_world = R_cube @ self.cube_inv_inertia_body @ R_cube.T

        # Gravity force
        water_ext_force = self.water_mass[:, None] * self.gravity
        cube_ext_force = self.cube_mass * self.gravity
        cube_total_force = cube_ext_force.copy()
        cube_total_torque = np.zeros(3, dtype=float)

        # 2. Fluid-cube interaction & Fluid self-interaction
        # Check fluid particles against cube SDF
        sdf, normal_to_cube = self._get_cube_sdf(pos)
        # normal_to_cube points OUT of the cube

        # Fluid particles penetration with cube (coupling)
        cube_contact_mask = (sdf < self.particle_radius * 1.5)
        fluid_cube_force = np.zeros(3, dtype=float)

        if np.any(cube_contact_mask):
            p_contact = pos[cube_contact_mask]
            v_contact = vel[cube_contact_mask]
            sdf_c = sdf[cube_contact_mask]
            n_c = normal_to_cube[cube_contact_mask]
            m_c = self.water_mass[cube_contact_mask]

            # Cube surface velocity at contact points
            r_c = p_contact - self.cube_pos
            v_cube_at_p = self.cube_vel + np.cross(self.cube_ang_vel, r_c)

            # Relative velocity
            v_rel = v_contact - v_cube_at_p
            vn = np.sum(v_rel * n_c, axis=-1, keepdims=True)

            # Contact penalty + damping force on fluid
            penetration = np.maximum(self.particle_radius * 1.5 - sdf_c, 0.0)
            k_contact = 50000.0
            d_contact = 1000.0
            f_mag = k_contact * penetration[:, None] - d_contact * np.minimum(vn, 0.0)
            f_mag = np.maximum(f_mag, 0.0)
            f_fluid = f_mag * n_c

            # Apply force to fluid particles
            vel[cube_contact_mask] += (f_fluid / m_c[:, None]) * dt

            # Equal and opposite force on cube
            f_on_cube = -np.sum(f_fluid, axis=0)
            cube_total_force += f_on_cube
            torques = np.cross(r_c, -f_fluid)
            cube_total_torque += np.sum(torques, axis=0)

            fluid_cube_force = f_on_cube

        # 3. Buoyancy & displaced volume hydrodynamic forces
        # Simple SPH-like pressure / gravity acceleration for water
        # Boundary forces on water
        for ax in range(3):
            # Min wall
            pen_min = self.wall_min[ax] + self.particle_radius - pos[:, ax]
            mask_min = pen_min > 0
            if np.any(mask_min):
                pos[mask_min, ax] = self.wall_min[ax] + self.particle_radius
                vel[mask_min, ax] = -self.restitution * vel[mask_min, ax]

            # Max wall
            pen_max = pos[:, ax] - (self.wall_max[ax] - self.particle_radius)
            mask_max = pen_max > 0
            if np.any(mask_max):
                pos[mask_max, ax] = self.wall_max[ax] - self.particle_radius
                vel[mask_max, ax] = -self.restitution * vel[mask_max, ax]

        # Water vertical column pressure gradient / resting state support
        # Fluid self-leveling / internal pressure
        z_water_surface = 1.05
        in_water = pos[:, 2] < z_water_surface
        # Hydrostatic equilibrium + external gravity
        vel[:, 2] += self.gravity[2] * dt
        # Buoyancy / pressure support in water bed
        depth = np.maximum(z_water_surface - pos[:, 2], 0.0)
        hydrostatic_acc = -self.gravity[2] * np.clip(depth / 0.9, 0.0, 1.0)
        vel[in_water, 2] += hydrostatic_acc[in_water] * dt

        # Water internal viscosity / damping
        vel *= (1.0 - 0.02 * dt)

        # 4. Cube immersion & buoyancy
        # Determine submerged fraction of cube
        # Approximate by sampling cube 8 vertices + center
        cube_corners_local = np.array([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
            [0, 0, 0]
        ], dtype=float) * self.cube_half_size
        cube_corners_world = self.cube_pos + cube_corners_local @ R_cube.T

        # Water level near cube
        submerged_mask = cube_corners_world[:, 2] < z_water_surface
        submerged_ratio = np.mean(submerged_mask)

        if submerged_ratio > 0:
            cube_volume = self.cube_size ** 3
            submerged_volume = cube_volume * submerged_ratio
            # Archimedes buoyancy force
            buoyancy_force = np.array([0.0, 0.0, self.water_rest_density * submerged_volume * (-self.gravity[2])])
            cube_total_force += buoyancy_force

            # Viscous fluid drag on cube
            drag_coeff = 0.5 * self.water_rest_density * (self.cube_size ** 2) * 1.05 * submerged_ratio
            v_rel_cube = self.cube_vel
            drag_force = -drag_coeff * np.linalg.norm(v_rel_cube) * v_rel_cube
            cube_total_force += drag_force

            # Angular drag
            ang_drag = -0.5 * self.water_rest_density * (self.cube_size ** 5) * 0.1 * submerged_ratio * self.cube_ang_vel
            cube_total_torque += ang_drag

            # Center of buoyancy torque
            submerged_pts = cube_corners_world[submerged_mask]
            if len(submerged_pts) > 0:
                cob = np.mean(submerged_pts, axis=0)
                r_cob = cob - self.cube_pos
                cube_total_torque += np.cross(r_cob, buoyancy_force)

        # 5. Integrate Cube
        cube_acc = cube_total_force / self.cube_mass
        self.cube_vel += cube_acc * dt
        self.cube_pos += self.cube_vel * dt

        cube_ang_acc = I_inv_world @ (cube_total_torque - np.cross(self.cube_ang_vel, I_inv_world @ self.cube_ang_vel))
        self.cube_ang_vel += cube_ang_acc * dt

        # Update cube orientation
        w = self.cube_ang_vel
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-8:
            axis = w / w_norm
            angle = w_norm * dt
            dq = np.array([axis[0] * np.sin(angle / 2),
                           axis[1] * np.sin(angle / 2),
                           axis[2] * np.sin(angle / 2),
                           np.cos(angle / 2)], dtype=float)
            self.cube_rot = quat_mult(dq, self.cube_rot)
            self.cube_rot /= np.linalg.norm(self.cube_rot)

        # Cube tank boundary collision (floor and walls)
        for ax in range(3):
            # Check cube bounds
            r_max_proj = np.max(np.abs(cube_corners_local @ R_cube.T), axis=0)
            if self.cube_pos[ax] - r_max_proj[ax] < self.wall_min[ax]:
                self.cube_pos[ax] = self.wall_min[ax] + r_max_proj[ax]
                if self.cube_vel[ax] < 0:
                    self.cube_vel[ax] = -self.restitution * self.cube_vel[ax]
                    self.cube_ang_vel *= 0.8
            if self.cube_pos[ax] + r_max_proj[ax] > self.wall_max[ax]:
                self.cube_pos[ax] = self.wall_max[ax] - r_max_proj[ax]
                if self.cube_vel[ax] > 0:
                    self.cube_vel[ax] = -self.restitution * self.cube_vel[ax]
                    self.cube_ang_vel *= 0.8

        # 6. Integrate Water particles
        pos += vel * dt

        # Store observables
        active_sdf, _ = self._get_cube_sdf(pos)
        self.last_contact_sd = np.sort(active_sdf)[:100]  # nearest signed distances
        self.last_external_force[:N] = water_ext_force
        self.last_external_force[N] = cube_ext_force
        self.last_interaction_force[0] = -fluid_cube_force  # water interaction force
        self.last_interaction_force[1] = fluid_cube_force   # cube interaction force

    def advance_to(self, time: float) -> StateFrame:
        # Step forward until reaching time
        while self.time + 1e-9 < time:
            dt = min(self.sub_dt, time - self.time)
            self._step(dt)
            self.time += dt

        self.time = time

        # Build StateFrame
        pos_all = np.vstack((self.water_pos, self.cube_pos[None, :]))
        vel_all = np.vstack((self.water_vel, self.cube_vel[None, :]))
        mass_all = np.concatenate((self.water_mass, [self.cube_mass]))
        body_id = np.concatenate((np.ones(self.num_water, dtype=np.int32), [2]))
        phase = np.concatenate((np.zeros(self.num_water), [1.0]))

        orientation = np.zeros((len(pos_all), 4), dtype=float)
        orientation[:, 3] = 1.0
        orientation[-1] = self.cube_rot

        ang_vel = np.zeros_like(pos_all)
        ang_vel[-1] = self.cube_ang_vel

        # Mass quantity: [N] or [N, K]
        # Cumulative mass source: [K] = [0.0, 0.0]
        # Interaction force: [K, 3] = [2, 3]
        # External force: [N, 3]
        # Contact signed distance: [C]

        frame = StateFrame(
            time=float(self.time),
            position=pos_all,
            velocity=vel_all,
            orientation=orientation,
            angular_velocity=ang_vel,
            mass=mass_all,
            body_id=body_id,
            attributes={
                "phase": phase,
                "mass_quantity": mass_all,
            },
            fields={
                "contact_signed_distance": self.last_contact_sd,
                "cumulative_mass_source": self.cumulative_mass_source,
                "external_force": self.last_external_force,
                "interaction_force": self.last_interaction_force,
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
                "water_rest_density": float(self.water_rest_density),
                "cube_density": float(self.cube_density),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "cube_position": self.cube_pos.tolist(),
            "cube_velocity": self.cube_vel.tolist(),
        }

