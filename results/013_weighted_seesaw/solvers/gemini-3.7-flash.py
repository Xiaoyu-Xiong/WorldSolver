"""Physics-based rigid body solver for the weighted seesaw scene.

Physical phenomenon:
- A seesaw board pivoted at the origin (x=0, y=0, z=0.7)
- A heavier block on the right (x=1.05) and a lighter block on the left (x=-1.0)
- Gravity tilts the seesaw toward the heavier block side
- The heavier block slides off the tilted board and falls to the ground
- Once the heavy block falls off, the lighter block tips the seesaw back toward the left
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def quat_from_angle_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    """Return quaternion [x, y, z, w] for rotation around normalized axis."""
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle * 0.5)
    c = np.cos(angle * 0.5)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c], dtype=float)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=float)


class BenchmarkSolver:
    def __init__(self):
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.masses = np.array([4.0, 3.0, 1.0], dtype=float)
        self.body_ids = np.array([0, 1, 2], dtype=int)

        # Dimensions (half-extents)
        self.board_half = np.array([1.8, 0.28, 0.07], dtype=float)
        self.heavy_half = np.array([0.28, 0.28, 0.28], dtype=float)
        self.light_half = np.array([0.22, 0.22, 0.22], dtype=float)

        # Board pivot point (top of pivot cylinder)
        # Pivot cylinder radius 0.22, height 0.7, translation [0, 0, 0.35] -> top center is [0, 0, 0.70]
        # Board translation is [0, 0, 0.78], bottom of board is 0.78 - 0.07 = 0.71 (resting on pivot)
        self.pivot_pos = np.array([0.0, 0.0, 0.70], dtype=float)
        self.pivot_board_offset = np.array([0.0, 0.0, -0.07], dtype=float) # pivot relative to board center

        # Board inertia around y-axis (pitch rotation)
        # I_yy for box = m/12 * (w^2 + h^2) where w=2*hx=3.6, h=2*hz=0.14
        m_board = self.masses[0]
        self.I_board_y = m_board / 12.0 * ((2 * self.board_half[0])**2 + (2 * self.board_half[2])**2)

        # Simulation substepping
        self.sub_dt = 0.001
        self.current_time = 0.0

        # State variables
        # Board: angle theta (pitch around Y axis, positive = right side down)
        self.theta = 0.0
        self.omega = 0.0

        # Blocks: 3D position and velocity
        self.block_pos = [
            np.array([1.05, 0.0, 1.13], dtype=float),  # Heavy block
            np.array([-1.0, 0.0, 1.07], dtype=float),  # Light block
        ]
        self.block_vel = [
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
        ]
        self.block_quat = [
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        ]
        self.block_angvel = [
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
        ]

        # Contact tracking for fields
        self.last_contact_sd = 0.0
        self.last_constraint_val = 0.0
        self.last_constraint_target = 0.0

    def initialize(self, scene, seed: int) -> None:
        init_state = scene.initial_state()
        self.masses = np.array(init_state.mass, dtype=float)
        self.body_ids = np.array(init_state.body_id, dtype=int)

        self.theta = 0.0
        self.omega = 0.0

        self.block_pos = [
            np.array(init_state.position[1], dtype=float),
            np.array(init_state.position[2], dtype=float),
        ]
        self.block_vel = [
            np.array(init_state.velocity[1], dtype=float),
            np.array(init_state.velocity[2], dtype=float),
        ]
        self.block_quat = [
            np.array(init_state.orientation[1], dtype=float),
            np.array(init_state.orientation[2], dtype=float),
        ]
        self.block_angvel = [
            np.array(init_state.angular_velocity[1], dtype=float),
            np.array(init_state.angular_velocity[2], dtype=float),
        ]
        self.current_time = 0.0

    def _step(self, dt: float) -> None:
        """Advance physics simulation by sub-step dt."""
        g = 9.81
        m_b = self.masses[0]
        m_h = self.masses[1]
        m_l = self.masses[2]

        # Board center of mass position based on angle theta (rotation around Y axis: [0, 1, 0])
        # Center of mass is pivot_pos - R * pivot_board_offset
        # Rotation around Y by theta:
        # x' = x*cos(theta) + z*sin(theta)
        # z' = -x*sin(theta) + z*cos(theta)
        # Board center:
        # Since pivot_board_offset = [0, 0, -0.07], in board frame pivot is at (0, 0, -0.07).
        # In world frame, board_center = pivot_pos - R(theta) * [0, 0, -0.07]
        # = [0, 0, 0.70] - [-0.07*sin(theta), 0, -0.07*cos(theta)]
        # = [0.07*sin(theta), 0, 0.70 + 0.07*cos(theta)]
        board_cm = self.pivot_pos + np.array([np.sin(self.theta)*0.07, 0.0, np.cos(self.theta)*0.07])

        # Max tilt angle before board hits the ground
        # Board half-length = 1.8, half-thickness = 0.07
        # Board tip height: z_tip = board_cm[2] - 1.8*sin(|theta|) - 0.07*cos(|theta|)
        # Ground is at z = 0
        # Tip hits ground when 0.70 - 1.8*sin(theta_max) = 0 -> theta_max ~ arcsin(0.7/1.8) ~ 0.40 rad (~23 deg)
        max_theta = np.arcsin(0.68 / 1.8)

        # Board coordinate frame axes in world:
        # u_board (along board length, +x): [cos(theta), 0, -sin(theta)]
        # n_board (normal to board, +z):     [sin(theta), 0, cos(theta)]
        u_b = np.array([np.cos(self.theta), 0.0, -np.sin(self.theta)])
        n_b = np.array([np.sin(self.theta), 0.0, np.cos(self.theta)])

        # Top surface of board has offset +0.07 in n_b direction from board_cm:
        # r_top_center = board_cm + 0.07 * n_b

        total_torque = 0.0
        effective_I = self.I_board_y

        min_signed_dist = 100.0

        # Process each block
        blocks_data = [
            (0, m_h, self.heavy_half, 0.3),  # (idx, mass, half_extents, friction_coeff)
            (1, m_l, self.light_half, 0.5),
        ]

        for idx, m_k, half_ext, mu in blocks_data:
            pos = self.block_pos[idx]
            vel = self.block_vel[idx]
            hz = half_ext[2]
            hx = half_ext[0]

            # Distance along board length (s) and normal to board (d)
            rel_pos = pos - board_cm
            s_proj = np.dot(rel_pos, u_b)
            d_proj = np.dot(rel_pos, n_b)

            # Board surface normal gap
            # Top surface is at d = +self.board_half[2] (0.07)
            # Bottom of block is at d - hz
            gap_board = (d_proj - hz) - self.board_half[2]

            # Ground collision for block
            gap_ground = pos[2] - hz

            on_board = False
            # Check if block is within board length bounds and contacting top surface
            if np.abs(s_proj) <= (self.board_half[0] + hx * 0.5) and gap_board < 0.05 and pos[2] > 0.3:
                on_board = True
                min_signed_dist = min(min_signed_dist, gap_board)

                # Rigid contact / resting constraint on board
                # Target normal distance: d_proj = self.board_half[2] + hz
                target_d = self.board_half[2] + hz

                # If penetrating or close, resolve normal position & velocity
                if gap_board < 0.0:
                    d_proj = target_d
                    pos = board_cm + s_proj * u_b + d_proj * n_b
                    self.block_pos[idx] = pos

                # Normal contact force from board on block:
                # Blocks experience gravity + centrifugal/board acceleration
                # Component of gravity normal to board: g_n = g * cos(theta)
                f_normal = m_k * g * np.cos(self.theta)

                # Gravity along board tangent: g_t = -g * sin(theta)
                # Tangential acceleration from gravity along board:
                a_tangent_grav = -g * np.sin(self.theta)

                # Board angular velocity tangential acceleration: omega * s
                v_board_tangent = 0.0 # pure rotation doesn't move tangent at center, but normal vel is -s * omega

                # Relative tangential velocity of block on board
                v_tangent = np.dot(vel, u_b)

                # Friction force along board tangent
                # Static friction threshold: mu * f_normal / m_k = mu * g * cos(theta)
                # Gravity pulling down slope: g * |sin(theta)|
                # Slipping occurs when |sin(theta)| > mu * cos(theta) => tan(|theta|) > mu
                friction_limit = mu * f_normal

                if np.abs(v_tangent) < 1e-3 and np.abs(m_k * a_tangent_grav) <= friction_limit:
                    # Static friction holds
                    v_tangent = 0.0
                    a_tangent = 0.0
                else:
                    # Dynamic sliding friction opposing motion (or opposing gravity if starting to slide)
                    slide_dir = np.sign(v_tangent) if np.abs(v_tangent) > 1e-3 else -np.sign(np.sin(self.theta))
                    friction_force = -slide_dir * friction_limit
                    a_tangent = a_tangent_grav + friction_force / m_k

                # Update tangential velocity and position
                v_tangent += a_tangent * dt
                s_proj += v_tangent * dt

                # Board velocity at contact point
                # Point velocity = omega x r = [0, omega, 0] x (s_proj * u_b + d_proj * n_b)
                # Board normal velocity at s_proj is -s_proj * omega * u_b_perp = s_proj * omega (in normal/vertical direction)
                vel = v_tangent * u_b + (-s_proj * self.omega) * n_b

                # Reconstruct world position
                pos = board_cm + s_proj * u_b + d_proj * n_b

                self.block_pos[idx] = pos
                self.block_vel[idx] = vel

                # Match block orientation to board pitch
                self.block_quat[idx] = quat_from_angle_axis(self.theta, np.array([0.0, 1.0, 0.0]))
                self.block_angvel[idx] = np.array([0.0, self.omega, 0.0])

                # Torque exerted by this block on the seesaw board around Y axis:
                # Contact force on board is -F_contact = - (f_normal * n_b + friction * u_b)
                # Torque = s_proj * f_normal (positive s_proj gives positive torque, tilting right side down)
                torque_k = s_proj * (m_k * g * np.cos(self.theta)) - self.board_half[2] * (m_k * a_tangent_grav)
                total_torque += torque_k
                effective_I += m_k * (s_proj**2)

            else:
                # Free fall in air or contact with ground
                if pos[2] - hz <= 0.0:
                    # Contact with ground
                    min_signed_dist = min(min_signed_dist, pos[2] - hz)
                    pos[2] = hz
                    vel[2] = -0.1 * vel[2] if vel[2] < -0.1 else 0.0 # slight bounce/damping
                    # Ground friction
                    vel[0] *= max(0.0, 1.0 - 5.0 * dt)
                    vel[1] *= max(0.0, 1.0 - 5.0 * dt)
                    self.block_angvel[idx] *= max(0.0, 1.0 - 5.0 * dt)
                else:
                    # Gravity in air
                    vel += self.gravity * dt
                    min_signed_dist = min(min_signed_dist, pos[2] - hz)

                pos += vel * dt
                self.block_pos[idx] = pos
                self.block_vel[idx] = vel

        # Board angular acceleration: alpha = total_torque / effective_I - damping
        damping = 0.8 * self.omega
        alpha = (total_torque - damping) / max(effective_I, 0.1)

        self.omega += alpha * dt
        self.theta += self.omega * dt

        # Enforce board ground contact limit (stops at max_theta)
        if self.theta > max_theta:
            self.theta = max_theta
            if self.omega > 0:
                self.omega = -0.1 * self.omega # restitution
        elif self.theta < -max_theta:
            self.theta = -max_theta
            if self.omega < 0:
                self.omega = -0.1 * self.omega

        self.last_contact_sd = min_signed_dist
        self.last_constraint_val = self.theta
        self.last_constraint_target = 0.0

    def advance_to(self, time: float) -> StateFrame:
        # Step forward to requested time
        while self.current_time + self.sub_dt * 0.5 < time:
            step_size = min(self.sub_dt, time - self.current_time)
            self._step(step_size)
            self.current_time += step_size

        # Board state
        board_quat = quat_from_angle_axis(self.theta, np.array([0.0, 1.0, 0.0]))
        board_cm = self.pivot_pos + np.array([np.sin(self.theta)*0.07, 0.0, np.cos(self.theta)*0.07])
        board_vel = np.array([np.cos(self.theta)*0.07*self.omega, 0.0, -np.sin(self.theta)*0.07*self.omega])
        board_angvel = np.array([0.0, self.omega, 0.0])

        positions = np.array([
            board_cm,
            self.block_pos[0],
            self.block_pos[1],
        ], dtype=float)

        velocities = np.array([
            board_vel,
            self.block_vel[0],
            self.block_vel[1],
        ], dtype=float)

        orientations = np.array([
            board_quat,
            self.block_quat[0],
            self.block_quat[1],
        ], dtype=float)

        ang_velocities = np.array([
            board_angvel,
            self.block_angvel[0],
            self.block_angvel[1],
        ], dtype=float)

        # External force = gravity for all dynamic bodies
        ext_forces = np.array([
            self.masses[0] * self.gravity,
            self.masses[1] * self.gravity,
            self.masses[2] * self.gravity,
        ], dtype=float)

        # Constraint values (e.g. pivot joint / angle constraint)
        constraint_val = np.array([self.last_constraint_val], dtype=float)
        constraint_tgt = np.array([self.last_constraint_target], dtype=float)
        contact_sd = np.array([self.last_contact_sd], dtype=float)

        # Construct StateFrame
        frame = StateFrame(
            time=float(time),
            position=positions,
            velocity=velocities,
            orientation=orientations,
            angular_velocity=ang_velocities,
            mass=self.masses.copy(),
            body_id=self.body_ids.copy(),
            fields={
                "constraint_target": constraint_tgt,
                "constraint_value": constraint_val,
                "contact_signed_distance": contact_sd,
                "external_force": ext_forces,
            },
            scalars={
                "constraint_scale": 1.0,
                "contact_length_scale": 0.07,
            }
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "theta": float(self.theta),
            "omega": float(self.omega),
        }

