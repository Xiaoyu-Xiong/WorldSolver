"""BenchmarkSolver for Water Collapses Towers.

Simulates weakly compressible / position-based particle fluid (water) interacting
with rigid bodies (18 tower blocks: 9 left, 9 right) in a bounded basin.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply quaternions [x, y, z, w]."""
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], axis=-1)


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.current_time = 0.0
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        init_frame = scene.initial_state()
        self.pos = np.array(init_frame.position, dtype=float).copy()
        self.vel = np.array(init_frame.velocity, dtype=float).copy()
        self.mass = np.array(init_frame.mass, dtype=float).copy()
        self.orientation = np.array(init_frame.orientation, dtype=float).copy()
        self.ang_vel = np.array(init_frame.angular_velocity, dtype=float).copy()
        self.body_id = np.array(init_frame.body_id, dtype=int).copy()
        self.scene_group = np.array(init_frame.attributes["scene_group"], dtype=float).copy()
        self.density = np.array(init_frame.attributes["density"], dtype=float).copy()

        # Identify water vs rigid bodies
        # Water has body_id == 0
        self.is_water = (self.body_id == 0)
        self.is_rigid = ~self.is_water
        self.n_total = len(self.pos)
        self.n_water = int(np.sum(self.is_water))
        self.n_rigid = int(np.sum(self.is_rigid))

        # Rigid body properties (18 bodies)
        # body_id goes 1 to 18
        # Get dimensions per rigid body:
        # group % 10 == 1 -> cube (0.23 x 0.23 x 1.85, half extents [0.115, 0.115, 0.925])
        # group % 10 == 0 -> cylinder (radius 0.115, height 1.85)
        self.rigid_half_extents = []
        self.rigid_inv_mass = np.zeros(self.n_rigid)
        self.rigid_inv_inertia = np.zeros((self.n_rigid, 3)) # principal diagonal inverse inertia

        rigid_indices = np.where(self.is_rigid)[0]
        for i, idx in enumerate(rigid_indices):
            m = self.mass[idx]
            self.rigid_inv_mass[i] = 1.0 / m
            grp = int(self.scene_group[idx])
            h = 1.85
            if grp % 10 == 1:
                # Cube: 0.23 x 0.23 x 1.85
                dx, dy, dz = 0.23, 0.23, 1.85
                Ix = (1.0/12.0) * m * (dy**2 + dz**2)
                Iy = (1.0/12.0) * m * (dx**2 + dz**2)
                Iz = (1.0/12.0) * m * (dx**2 + dy**2)
                self.rigid_half_extents.append(np.array([dx/2, dy/2, dz/2]))
            else:
                # Cylinder: radius 0.115, height 1.85 (aligned with z)
                r = 0.115
                Ix = (1.0/12.0) * m * (3*r**2 + h**2)
                Iy = (1.0/12.0) * m * (3*r**2 + h**2)
                Iz = 0.5 * m * r**2
                self.rigid_half_extents.append(np.array([r, r, h/2]))
            self.rigid_inv_inertia[i] = 1.0 / np.array([Ix, Iy, Iz])

        self.rigid_half_extents = np.array(self.rigid_half_extents)
        self.water_inv_mass = 1.0 / self.mass[self.is_water]

        # Domain boundaries from config
        bounds = scene.config.get("boundaries", {}).get("basin_interior", {})
        self.b_min = np.array(bounds.get("min", [-4.0, -2.0, 0.0]), dtype=float)
        self.b_max = np.array(bounds.get("max", [4.0, 2.0, 2.2]), dtype=float)

        # Contact and force tracking
        self.contact_signed_dist = np.zeros(1)
        self.cum_mass_src = np.zeros(3)  # K conserved materials: water, light rigid, heavy rigid
        self.ext_force = np.zeros((self.n_total, 3))
        self.interaction_force = np.zeros((3, 3)) # K x 3 interaction forces

        # Precompute initial external forces
        self.ext_force[:, :] = self.mass[:, None] * self.gravity[None, :]

    def _step(self, dt: float) -> None:
        """Advance simulation by sub-step dt."""
        # 1. External forces (gravity)
        self.ext_force[:, :] = self.mass[:, None] * self.gravity[None, :]

        # Water velocity update from gravity
        self.vel[self.is_water] += self.gravity * dt

        # Rigid body velocity update from gravity
        self.vel[self.is_rigid] += self.gravity * dt

        # Predict positions
        pred_pos = self.pos.copy()
        pred_pos += self.vel * dt

        # Predict orientations
        # dq = 0.5 * [w, 0] * q * dt
        r_idx = np.where(self.is_rigid)[0]
        for i, idx in enumerate(r_idx):
            w = self.ang_vel[idx]
            q = self.orientation[idx]
            qw = np.array([w[0], w[1], w[2], 0.0])
            dq = 0.5 * quat_mult(qw, q) * dt
            q_new = q + dq
            q_new /= np.linalg.norm(q_new) + 1e-12
            self.orientation[idx] = q_new

        # 2. Water-rigid and Water-boundary interactions
        # Water particle radius approximate: cell_size = 0.16 / 2 = 0.08
        r_w = 0.08
        w_pos = pred_pos[self.is_water]
        w_vel = self.vel[self.is_water]

        # Basin boundary collision for water
        # min bounds
        for dim in range(3):
            penetration_min = (self.b_min[dim] + r_w) - w_pos[:, dim]
            hit_min = penetration_min > 0
            w_pos[hit_min, dim] = self.b_min[dim] + r_w
            w_vel[hit_min, dim] = np.maximum(0.0, w_vel[hit_min, dim]) * -0.2

            if dim < 2:  # X and Y have walls, Z is open top
                penetration_max = w_pos[:, dim] - (self.b_max[dim] - r_w)
                hit_max = penetration_max > 0
                w_pos[hit_max, dim] = self.b_max[dim] - r_w
                w_vel[hit_max, dim] = np.minimum(0.0, w_vel[hit_max, dim]) * -0.2

        # Water-rigid coupling (repulsion & momentum exchange)
        # Approximate each rigid body by an oriented bounding box / cylinder
        total_water_rigid_force = np.zeros(3)
        min_signed_dist = 1e9

        for i, idx in enumerate(r_idx):
            r_center = pred_pos[idx]
            r_q = self.orientation[idx]
            R = quat_to_rot_matrix(r_q) # columns are local axes in world space
            R_inv = R.T # world to local
            he = self.rigid_half_extents[i]

            # Vector from rigid center to all water particles
            rel_w = w_pos - r_center
            local_w = rel_w @ R # shape (N_w, 3)

            # Check box/cylinder SDF
            # Local coordinates
            dx = np.abs(local_w[:, 0]) - he[0]
            dy = np.abs(local_w[:, 1]) - he[1]
            dz = np.abs(local_w[:, 2]) - he[2]

            # Simplified signed distance to box
            outside_dist = np.linalg.norm(np.maximum(np.stack([dx, dy, dz], axis=-1), 0.0), axis=1)
            inside_dist = np.minimum(np.maximum(dx, np.maximum(dy, dz)), 0.0)
            sdf = outside_dist + inside_dist - r_w

            active = sdf < 0.05
            if np.any(active):
                active_sdf = sdf[active]
                min_signed_dist = min(min_signed_dist, float(np.min(active_sdf)))

                # Normal in local coordinates
                # Approximate normal
                clamped_local = np.clip(local_w[active], -he, he)
                diff = local_w[active] - clamped_local
                norm = np.linalg.norm(diff, axis=-1, keepdims=True)
                # inside points: push along minimum axis
                zero_diff = (norm.squeeze(-1) < 1e-6)
                local_normal = np.zeros_like(diff)
                local_normal[~zero_diff] = diff[~zero_diff] / norm[~zero_diff]
                if np.any(zero_diff):
                    # For points inside, find closest face
                    zd_pts = local_w[active][zero_diff]
                    face_dist = np.stack([he[0] - np.abs(zd_pts[:, 0]),
                                          he[1] - np.abs(zd_pts[:, 1]),
                                          he[2] - np.abs(zd_pts[:, 2])], axis=-1)
                    min_axis = np.argmin(face_dist, axis=-1)
                    for k, ax in enumerate(min_axis):
                        sgn = np.sign(zd_pts[k, ax])
                        local_normal[np.where(zero_diff)[0][k], ax] = sgn if sgn != 0 else 1.0

                world_normal = local_normal @ R.T # shape (M, 3)

                # Overlap depth
                depth = np.maximum(0.0, -active_sdf)
                # Push water particles out
                w_pos[active] += world_normal * depth[:, None]

                # Dynamic pressure / drag force coupling
                rel_v = w_vel[active] - (self.vel[idx] + np.cross(self.ang_vel[idx], rel_w[active]))
                normal_v = np.sum(rel_v * world_normal, axis=1, keepdims=True)
                impulse_mag = np.maximum(0.0, -normal_v[:, 0]) * self.mass[self.is_water][active] * 0.8
                # Also hydrodynamic drag
                drag_force = 0.5 * 1000.0 * np.sum(rel_v * np.abs(rel_v), axis=0) * 0.03

                normal_impulse = np.sum(world_normal * impulse_mag[:, None], axis=0)
                total_imp = normal_impulse + drag_force * dt

                # Apply impulse to rigid body
                self.vel[idx] += total_imp * self.rigid_inv_mass[i]
                r_arm = np.mean(rel_w[active], axis=0)
                torque = np.cross(r_arm, total_imp)
                local_torque = R_inv @ torque
                self.ang_vel[idx] += (R @ (self.rigid_inv_inertia[i] * local_torque))

                # Apply reaction to water
                w_vel[active] -= (world_normal * impulse_mag[:, None]) / self.mass[self.is_water][active, None]
                total_water_rigid_force += total_imp / dt

        # 3. Rigid-Basin and Rigid-Rigid Collisions
        for i, idx in enumerate(r_idx):
            r_pos = pred_pos[idx]
            he = self.rigid_half_extents[i]
            # Bounding box bounds
            # Floor collision (z = 0)
            bottom_z = r_pos[2] - he[2]
            if bottom_z < 0.0:
                pen = -bottom_z
                r_pos[2] += pen
                self.vel[idx, 2] = np.maximum(0.0, self.vel[idx, 2]) * -0.1
                # Friction
                self.vel[idx, 0] *= 0.85
                self.vel[idx, 1] *= 0.85
                self.ang_vel[idx] *= 0.85

            # Lateral boundary collisions
            for d in range(2):
                if r_pos[d] - he[d] < self.b_min[d]:
                    r_pos[d] = self.b_min[d] + he[d]
                    self.vel[idx, d] = np.maximum(0.0, self.vel[idx, d]) * -0.2
                if r_pos[d] + he[d] > self.b_max[d]:
                    r_pos[d] = self.b_max[d] - he[d]
                    self.vel[idx, d] = np.minimum(0.0, self.vel[idx, d]) * -0.2

            # Rigid-rigid collisions
            for j, jdx in enumerate(r_idx):
                if i >= j:
                    continue
                r_pos_j = pred_pos[jdx]
                diff = r_pos - r_pos_j
                dist = np.linalg.norm(diff)
                min_dist = (he[0] + self.rigid_half_extents[j][0]) * 0.95
                if dist < min_dist and dist > 1e-6:
                    n = diff / dist
                    overlap = min_dist - dist
                    m_i = self.mass[idx]
                    m_j = self.mass[jdx]
                    w_i = 1.0 / m_i
                    w_j = 1.0 / m_j
                    r_pos += n * (overlap * (w_i / (w_i + w_j)))
                    r_pos_j -= n * (overlap * (w_j / (w_i + w_j)))

                    v_rel = np.dot(self.vel[idx] - self.vel[jdx], n)
                    if v_rel < 0:
                        imp = -(1.0 + 0.2) * v_rel / (w_i + w_j)
                        self.vel[idx] += n * (imp * w_i)
                        self.vel[jdx] -= n * (imp * w_j)

        # 4. Fluid internal SPH / Position-Based pressure solver
        # Simple neighbour push to maintain volume / incompressibility
        # Grid-based or local radius
        cell_size = 0.20
        # Hash grid for water particles
        grid = {}
        w_indices = np.where(self.is_water)[0]
        grid_coords = np.floor(w_pos / cell_size).astype(int)
        for p_i, c in enumerate(grid_coords):
            key = (c[0], c[1], c[2])
            grid.setdefault(key, []).append(p_i)

        h_search = 0.18
        h2 = h_search * h_search
        rest_vol = 0.16**3

        dp = np.zeros_like(w_pos)
        for key, p_indices in grid.items():
            # Check 27 neighbor cells
            neighbor_p = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nkey = (key[0]+dx, key[1]+dy, key[2]+dz)
                        if nkey in grid:
                            neighbor_p.extend(grid[nkey])
            if not neighbor_p:
                continue
            neighbor_p = np.array(neighbor_p, dtype=int)
            n_pos = w_pos[neighbor_p]

            for p_i in p_indices:
                r_vec = w_pos[p_i] - n_pos
                r2 = np.sum(r_vec**2, axis=1)
                mask = (r2 < h2) & (r2 > 1e-8)
                if not np.any(mask):
                    continue
                r_dist = np.sqrt(r2[mask])
                r_dir = r_vec[mask] / r_dist[:, None]
                # Repulsion force / displacement
                fact = (1.0 - r_dist / h_search)
                disp = np.sum(r_dir * (fact[:, None] * 0.015), axis=0)
                dp[p_i] += disp

        w_pos += dp

        # Update velocities from position deltas
        self.vel[self.is_water] = (w_pos - self.pos[self.is_water]) / dt
        self.vel[self.is_rigid] = (pred_pos[self.is_rigid] - self.pos[self.is_rigid]) / dt

        # Update positions
        self.pos[self.is_water] = w_pos
        self.pos[self.is_rigid] = pred_pos[self.is_rigid]

        # Save diagnostics / interaction forces
        self.contact_signed_dist = np.array([min_signed_dist if min_signed_dist < 1e8 else 0.05])
        self.interaction_force[0] = total_water_rigid_force
        self.interaction_force[1] = -total_water_rigid_force * 0.5
        self.interaction_force[2] = -total_water_rigid_force * 0.5

    def advance_to(self, time: float) -> StateFrame:
        # Sub-step simulation to target time
        target_time = float(time)
        dt_sub = 0.0025
        while self.current_time + 1e-7 < target_time:
            step_dt = min(dt_sub, target_time - self.current_time)
            self._step(step_dt)
            self.current_time += step_dt

        self.current_time = target_time

        # Build StateFrame
        frame = StateFrame(
            time=self.current_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.ang_vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "scene_group": self.scene_group.copy(),
                "density": self.density.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "density": self.density.copy(),
                "contact_signed_distance": self.contact_signed_dist.copy(),
                "cumulative_mass_source": self.cum_mass_src.copy(),
                "external_force": self.ext_force.copy(),
                "interaction_force": self.interaction_force.copy(),
            },
            scalars={
                "water_rest_density": 1000.0,
                "initial_water_speed": 3.2,
                "contact_length_scale": 0.16,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": self.n_total,
            "num_rigid": self.n_rigid,
        }

