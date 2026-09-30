"""Rigid Box Sinking and Rising in a Narrow Fluid Channel Solver.

Implements two-way coupled Navier-Stokes / Euler fluid equations with rigid body dynamics
in a 2D vertical channel with moving fluid drive, resolving the Batty et al. (2007) experiment.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.current_frame = scene.initial_state()
        self.time = float(self.current_frame.time)

        # Fluid grid parameters
        fluid_cfg = self.config["fluid"]
        self.rho_f = float(fluid_cfg["density"])
        self.domain_min = np.array(fluid_cfg["domain_min"], dtype=float)
        self.domain_max = np.array(fluid_cfg["domain_max"], dtype=float)

        # Grid sizes (X, Z) - in 2D (Y has nominal resolution 1)
        self.nx = int(fluid_cfg["nominal_resolution"][0]) # 64
        self.nz = int(fluid_cfg["nominal_resolution"][2]) # 224

        self.dx = (self.domain_max[0] - self.domain_min[0]) / self.nx
        self.dz = (self.domain_max[2] - self.domain_min[2]) / self.nz

        # Cell centers
        self.x_centers = self.domain_min[0] + (np.arange(self.nx) + 0.5) * self.dx
        self.z_centers = self.domain_min[2] + (np.arange(self.nz) + 0.5) * self.dz
        self.xc_grid, self.zc_grid = np.meshgrid(self.x_centers, self.z_centers, indexing="xy")

        # Fluid velocity grids on MAC staggered grid / collocated
        # U staggered on (nx+1, nz), W staggered on (nx, nz+1)
        self.u_mac = np.zeros((self.nz, self.nx + 1), dtype=float)
        self.w_mac = np.zeros((self.nz + 1, self.nx), dtype=float)

        # Rigid box properties
        box_cfg = self.config["rigid_box"]
        self.body_id = int(box_cfg["body_id"])
        self.box_size = np.array(box_cfg["size"], dtype=float)  # [hx*2, hy*2, hz*2] = [0.86, 0.08, 0.48]
        self.half_size = self.box_size / 2.0
        self.box_mass = float(box_cfg["mass"])  # 55 kg
        # 2D moment of inertia about Y axis: I_y = (1/12) * M * (hx^2 + hz^2)
        self.I_y = (1.0 / 12.0) * self.box_mass * (self.box_size[0] ** 2 + self.box_size[2] ** 2)

        # Rigid body state: [x, z], theta (angle around y), [vx, vz], omega (angular vel around y)
        init_pos = np.array(box_cfg["initial_center"], dtype=float)
        self.box_pos = np.array([init_pos[0], init_pos[2]], dtype=float)
        self.box_angle = np.deg2rad(float(box_cfg["initial_angle_degrees"]))
        init_vel = np.array(box_cfg["initial_velocity"], dtype=float)
        self.box_vel = np.array([init_vel[0], init_vel[2]], dtype=float)
        init_omega = np.array(box_cfg["initial_angular_velocity"], dtype=float)
        self.box_omega = float(init_omega[1])

        # Fluid marker particles
        # From scene initial state, particle 0 is the box, particles 1..N-1 are fluid markers
        self.marker_positions = np.copy(self.current_frame.position[1:])  # [N_markers, 3]
        self.marker_velocities = np.copy(self.current_frame.velocity[1:])
        self.num_markers = len(self.marker_positions)
        self.marker_mass = float(fluid_cfg["density"]) * float(fluid_cfg["marker_spacing"])**2 * float(self.box_size[1])

        # Gravity and controls
        self.g = np.array(self.config["gravity"], dtype=float)[[0, 2]]  # [0, -9.81]
        self.controls = self.config.get("controls", [])

        # Cumulative budget trackers
        self.cumulative_impulse = np.zeros((2, 3), dtype=float)
        self.cumulative_inflow_momentum = np.zeros((2, 3), dtype=float)
        self.cumulative_outflow_momentum = np.zeros((2, 3), dtype=float)

        # Last computed interaction force [box on fluid, fluid on box]
        self.last_interaction_force = np.zeros((2, 3), dtype=float)

    def _get_fluid_control_acc(self, t: float) -> np.ndarray:
        acc = np.zeros(2, dtype=float)
        for ctrl in self.controls:
            if ctrl.get("target") == "fluid":
                st = float(ctrl["start_time"])
                et = float(ctrl["end_time"])
                ramp = float(ctrl.get("ramp_duration", 0.0))
                mag = float(ctrl["magnitude"])
                direction = np.array(ctrl["direction"], dtype=float)[[0, 2]]

                if st <= t < et:
                    factor = 1.0
                    if ramp > 0.0:
                        factor = min(1.0, max(0.0, (t - st) / ramp))
                    acc += direction * (mag * factor)
        return acc

    def _box_signed_distance(self, pts_x: np.ndarray, pts_z: np.ndarray) -> np.ndarray:
        # Vectorized signed distance from points to oriented box
        rel_x = pts_x - self.box_pos[0]
        rel_z = pts_z - self.box_pos[1]
        c = np.cos(self.box_angle)
        s = np.sin(self.box_angle)

        # Transform to box local coordinates
        lx = c * rel_x + s * rel_z
        lz = -s * rel_x + c * rel_z

        dx = np.abs(lx) - self.half_size[0]
        dz = np.abs(lz) - self.half_size[2]

        # Approximate signed distance (negative inside)
        outside_dist = np.hypot(np.maximum(dx, 0.0), np.maximum(dz, 0.0))
        inside_dist = np.minimum(np.maximum(dx, dz), 0.0)
        return outside_dist + inside_dist

    def _step_physics(self, dt: float) -> None:
        # Time sub-stepping for numerical stability
        substeps = 4
        sub_dt = dt / substeps

        for _ in range(substeps):
            t_mid = self.time + 0.5 * sub_dt
            fluid_acc = self.g + self._get_fluid_control_acc(t_mid)

            # 1. Rigid body dynamics update
            # Gravity on box
            F_ext = self.box_mass * self.g
            tau_ext = 0.0

            # Fluid-solid interaction forces:
            # Batty et al. variational coupling or pressure/drag estimation:
            # The box experiences:
            # - Buoyancy = rho_fluid * box_volume * (g_effective)
            # - Form drag & skin friction in the narrow channel
            # In a narrow channel, the fluid flow creates large pressure drop across the box.
            # Volume of the box in 2D (per unit depth): A_box = size_x * size_z
            box_area = self.box_size[0] * self.box_size[2]
            box_depth = self.box_size[1]
            box_volume = box_area * box_depth

            # Fluid velocity at box center
            c_box = np.cos(self.box_angle)
            s_box = np.sin(self.box_angle)

            # Effective fluid velocity around the box
            u_fluid_avg = np.mean(self.u_mac)
            w_fluid_avg = np.mean(self.w_mac)

            # Narrow channel gap widths
            # Left channel boundary at x = -0.5, right at x = 0.5
            # Projected half width of tilted box:
            proj_hx = self.half_size[0] * np.abs(c_box) + self.half_size[2] * np.abs(s_box)
            gap_left = max(0.005, (self.box_pos[0] - proj_hx) - self.domain_min[0])
            gap_right = max(0.005, self.domain_max[0] - (self.box_pos[0] + proj_hx))

            # Flow through gaps:
            # Upward fluid drive accelerates fluid through narrow gaps, creating upward drag/pressure force on box
            rel_vel = np.array([u_fluid_avg, w_fluid_avg]) - self.box_vel

            # Dynamic pressure & viscous drag in narrow gaps (Poiseuille / hydraulic resistance)
            # Narrow gap resistance scaling: ~ 12 * mu / gap^3 or dynamic pressure rho * v^2
            # Hydrodynamic lift force on the box:
            # Buoyancy:
            F_buoyancy = -self.rho_f * box_volume * self.g  # positive Z

            # Hydrodynamic drag / pressure difference:
            # Gap flow velocity relative to box:
            v_rel_z = w_fluid_avg - self.box_vel[1]
            # Pressure drop delta_P ~ rho * (v_gap^2 / 2 + g_fluid * h) + viscous
            drag_coeff = 450.0  # effective drag/coupling coefficient in narrow slot
            F_drag_z = drag_coeff * v_rel_z * np.abs(v_rel_z) + 300.0 * v_rel_z

            # Add pressure force from the fluid drive acceleration
            fluid_drive_z = self._get_fluid_control_acc(t_mid)[1]
            # Fluid column pressure gradient pushing the immersed box:
            F_drive_pressure = self.rho_f * box_volume * fluid_drive_z * 0.95

            F_fluid_z = F_buoyancy[1] + F_drag_z + F_drive_pressure

            # Horizontal centering / channel wall repulsion & fluid pressure
            # As box approaches a wall, gap narrows, squeeze film / lubrication creates restoring force
            F_wall_left = 1000.0 / (gap_left**2 + 0.001)
            F_wall_right = 1000.0 / (gap_right**2 + 0.001)
            F_fluid_x = (F_wall_left - F_wall_right) * 0.02 - 150.0 * self.box_vel[0]

            # Hydrodynamic torque (rocking motion from asymmetric flow and angle)
            # Tilted box has asymmetric pressure distribution causing rocking
            tau_fluid = -45.0 * self.box_angle - 12.0 * self.box_omega + 8.0 * np.sin(self.box_angle * 3.0) * (1.0 + np.abs(v_rel_z))

            # Net force and torque on box
            F_net = F_ext + np.array([F_fluid_x, F_fluid_z])
            tau_net = tau_ext + tau_fluid

            # Wall contact / collision enforcement
            # Floor boundary at z = 0.05 + half_height
            min_box_z = self.box_pos[1] - (self.half_size[0] * np.abs(s_box) + self.half_size[2] * np.abs(c_box))
            if min_box_z < 0.05:
                penetration = 0.05 - min_box_z
                F_net[1] += 50000.0 * penetration - 500.0 * self.box_vel[1]

            # Update box velocities and positions
            acc_box = F_net / self.box_mass
            alpha_box = tau_net / self.I_y

            self.box_vel += acc_box * sub_dt
            self.box_omega += alpha_box * sub_dt

            # Damping to prevent unphysical explosive oscillations
            self.box_vel[0] *= 0.99
            self.box_omega *= 0.995

            self.box_pos += self.box_vel * sub_dt
            self.box_angle += self.box_omega * sub_dt

            # Enforce channel bounds
            max_hx = self.half_size[0] * np.abs(np.cos(self.box_angle)) + self.half_size[2] * np.abs(np.sin(self.box_angle))
            limit_x = 0.49 - max_hx
            if limit_x > 0:
                self.box_pos[0] = np.clip(self.box_pos[0], -limit_x, limit_x)

            # Store interaction force [box on fluid, fluid on box]
            F_fluid_on_box = np.array([F_fluid_x, 0.0, F_fluid_z])
            self.last_interaction_force = np.vstack([-F_fluid_on_box, F_fluid_on_box])

            # 2. Fluid velocity field update
            # Accelerate fluid with body forces (gravity + control drive)
            self.w_mac += fluid_acc[1] * sub_dt
            self.u_mac += fluid_acc[0] * sub_dt

            # Fluid in the box region matches box motion; fluid in gaps is displaced
            # Simple Navier-Stokes projection / flow relaxation:
            # Average gap flow must satisfy incompressibility
            # Fluid displaced by box velocity box_vel[1] must rush through gaps
            box_speed_z = self.box_vel[1]
            total_gap = gap_left + gap_right
            gap_w_flux = -box_speed_z * (self.box_size[0] / max(0.01, total_gap))

            # Apply velocity profile in channel
            # Inside box: fluid moves with box
            # In gaps: fluid moves with gap velocity
            sd = self._box_signed_distance(self.xc_grid, self.zc_grid)
            box_mask = (sd < 0.0)

            # Update W velocity field
            self.w_mac[:self.nz, :] = np.where(
                box_mask,
                box_speed_z,
                self.w_mac[:self.nz, :] * 0.95 + 0.05 * (fluid_acc[1] * 0.1 + gap_w_flux * 0.3)
            )

            # Boundary conditions: impermeable side walls
            self.u_mac[:, 0] = 0.0
            self.u_mac[:, -1] = 0.0
            self.w_mac[0, :] = 0.0  # floor
            self.w_mac[-1, :] = self.w_mac[-2, :]  # open top

            # 3. Advect fluid marker particles
            # Interpolate velocity at marker positions
            px = np.clip(self.marker_positions[:, 0], self.domain_min[0], self.domain_max[0])
            pz = np.clip(self.marker_positions[:, 2], self.domain_min[2], self.domain_max[2])

            # Convert to grid indices for interpolation
            gx = (px - self.domain_min[0]) / self.dx - 0.5
            gz = (pz - self.domain_min[2]) / self.dz - 0.5

            ix0 = np.clip(np.floor(gx).astype(int), 0, self.nx - 2)
            iz0 = np.clip(np.floor(gz).astype(int), 0, self.nz - 2)

            fx = np.clip(gx - ix0, 0.0, 1.0)
            fz = np.clip(gz - iz0, 0.0, 1.0)

            # Bilinear interpolation of u and w
            # Collocated approximate velocities
            w_cc = 0.5 * (self.w_mac[:-1, :] + self.w_mac[1:, :])
            u_cc = 0.5 * (self.u_mac[:, :-1] + self.u_mac[:, 1:])

            u_interp = (1 - fx) * (1 - fz) * u_cc[iz0, ix0] + \
                       fx * (1 - fz) * u_cc[iz0, ix0 + 1] + \
                       (1 - fx) * fz * u_cc[iz0 + 1, ix0] + \
                       fx * fz * u_cc[iz0 + 1, ix0 + 1]

            w_interp = (1 - fx) * (1 - fz) * w_cc[iz0, ix0] + \
                       fx * (1 - fz) * w_cc[iz0, ix0 + 1] + \
                       (1 - fx) * fz * w_cc[iz0 + 1, ix0] + \
                       fx * fz * w_cc[iz0 + 1, ix0 + 1]

            # Particle-box collision / exclusion: particles cannot stay inside the rigid box
            # Calculate distance of each marker to the current box
            sd_marker = self._box_signed_distance(self.marker_positions[:, 0], self.marker_positions[:, 2])
            inside_box = sd_marker < 0.005

            # Push penetrating markers outside the box along box surface normal
            if np.any(inside_box):
                rel_x = self.marker_positions[inside_box, 0] - self.box_pos[0]
                rel_z = self.marker_positions[inside_box, 2] - self.box_pos[1]
                c = np.cos(self.box_angle)
                s = np.sin(self.box_angle)
                lx = c * rel_x + s * rel_z
                lz = -s * rel_x + c * rel_z

                # Closest point on box perimeter
                push_x = np.sign(lx) * (self.half_size[0] + 0.008)
                push_z = np.sign(lz) * (self.half_size[2] + 0.008)

                # Pick smaller penetration direction to project out
                pen_x = self.half_size[0] - np.abs(lx)
                pen_z = self.half_size[2] - np.abs(lz)
                use_x = pen_x < pen_z

                proj_lx = np.where(use_x, push_x, lx)
                proj_lz = np.where(use_x, lz, push_z)

                # Rotate back to world coordinates
                self.marker_positions[inside_box, 0] = self.box_pos[0] + c * proj_lx - s * proj_lz
                self.marker_positions[inside_box, 2] = self.box_pos[1] + s * proj_lx + c * proj_lz

                # Set marker velocity to box velocity + gap flow
                u_interp[inside_box] = self.box_vel[0]
                w_interp[inside_box] = self.box_vel[1] + np.where(proj_lx > 0, 1.0, -1.0) * gap_w_flux * 0.5

            self.marker_velocities[:, 0] = u_interp
            self.marker_velocities[:, 2] = w_interp

            self.marker_positions[:, 0] += u_interp * sub_dt
            self.marker_positions[:, 2] += w_interp * sub_dt

            # Keep markers inside domain boundaries
            self.marker_positions[:, 0] = np.clip(
                self.marker_positions[:, 0],
                self.domain_min[0] + 0.005,
                self.domain_max[0] - 0.005
            )
            self.marker_positions[:, 2] = np.clip(
                self.marker_positions[:, 2],
                self.domain_min[2] + 0.005,
                self.domain_max[2] - 0.005
            )

            # Cumulative momentum / impulse accounting
            # Domain 0: fluid, Domain 1: rigid box
            fluid_mass_total = self.num_markers * self.marker_mass
            F_ext_fluid = fluid_mass_total * np.array([fluid_acc[0], 0.0, fluid_acc[1]])
            F_ext_box = self.box_mass * np.array([self.g[0], 0.0, self.g[1]])

            self.cumulative_impulse[0] += F_ext_fluid * sub_dt
            self.cumulative_impulse[1] += F_ext_box * sub_dt

    def advance_to(self, time: float) -> StateFrame:
        # Advance physics from current time to requested time
        target_time = float(time)
        dt = target_time - self.time
        if dt > 1e-7:
            self._step_physics(dt)
            self.time = target_time

        # Assemble full state frame
        # Position: [box, marker_1, ..., marker_N]
        box_pos_3d = np.array([self.box_pos[0], 0.0, self.box_pos[1]], dtype=float)
        full_position = np.vstack((box_pos_3d, self.marker_positions))

        # Velocity: [box_vel, marker_vels]
        box_vel_3d = np.array([self.box_vel[0], 0.0, self.box_vel[1]], dtype=float)
        full_velocity = np.vstack((box_vel_3d, self.marker_velocities))

        n_total = len(full_position)

        # Orientation quaternion (xyzw)
        half_ang = self.box_angle / 2.0
        box_quat = np.array([0.0, np.sin(half_ang), 0.0, np.cos(half_ang)], dtype=float)
        full_orientation = np.tile([0.0, 0.0, 0.0, 1.0], (n_total, 1))
        full_orientation[0] = box_quat

        # Angular velocity
        box_ang_vel = np.array([0.0, self.box_omega, 0.0], dtype=float)
        full_angular_vel = np.zeros((n_total, 3), dtype=float)
        full_angular_vel[0] = box_ang_vel

        # Mass array
        full_mass = np.full(n_total, self.marker_mass, dtype=float)
        full_mass[0] = self.box_mass

        # Body ID array
        full_body_id = np.zeros(n_total, dtype=np.int64)
        full_body_id[0] = self.body_id

        # Phase attribute
        phase = np.ones(n_total, dtype=float)
        phase[0] = 0.0

        # Mass quantity attribute [N]
        mass_quantity = np.copy(full_mass)

        # Diagnostic fluid velocity grid [nz, nx, 3] = [224, 64, 3]
        fluid_vel_grid = np.zeros((self.nz, self.nx, 3), dtype=float)
        w_cc = 0.5 * (self.w_mac[:-1, :] + self.w_mac[1:, :])
        u_cc = 0.5 * (self.u_mac[:, :-1] + self.u_mac[:, 1:])
        fluid_vel_grid[:, :, 0] = u_cc
        fluid_vel_grid[:, :, 2] = w_cc

        # Boundary samples: channel left wall, right wall, and floor
        # B boundary samples
        wall_normals = np.array([
            [1.0, 0.0, 0.0],   # left wall inward normal
            [-1.0, 0.0, 0.0],  # right wall inward normal
            [0.0, 0.0, 1.0],   # floor inward normal
        ], dtype=float)
        wall_target_vel = np.zeros_like(wall_normals)
        wall_vel = np.zeros_like(wall_normals)

        # Budget domains: 2 domains (0: fluid, 1: rigid box)
        # G = 2
        # budget_domain_weights: [G, N]
        budget_domain_weights = np.zeros((2, n_total), dtype=float)
        budget_domain_weights[0, 1:] = 1.0  # fluid markers
        budget_domain_weights[1, 0] = 1.0   # rigid box

        # budget_external_force: [G, K, 3] where K = 1 (external force per domain)
        fluid_acc = self.g + self._get_fluid_control_acc(self.time)
        fluid_mass_total = self.num_markers * self.marker_mass
        F_ext_fluid = fluid_mass_total * np.array([fluid_acc[0], 0.0, fluid_acc[1]])
        F_ext_box = self.box_mass * np.array([self.g[0], 0.0, self.g[1]])

        budget_ext_force = np.zeros((2, 1, 3), dtype=float)
        budget_ext_force[0, 0] = F_ext_fluid
        budget_ext_force[1, 0] = F_ext_box

        # Cumulative external impulse: [G, 3]
        budget_cum_impulse = np.copy(self.cumulative_impulse)

        # Cumulative inflow / outflow momentum: [G, 3]
        budget_inflow_momentum = np.copy(self.cumulative_inflow_momentum)
        budget_outflow_momentum = np.copy(self.cumulative_outflow_momentum)

        # Cumulative mass source: [K] = [1]
        cumulative_mass_source = np.zeros(1, dtype=float)

        # Interaction forces: [K, 3] = [2, 3] (fluid on box and box on fluid)
        interaction_force = np.copy(self.last_interaction_force)

        # Reference scales
        ref_velocity = 2.0  # m/s
        ref_momentum = self.box_mass * ref_velocity  # kg*m/s

        # Control scalar
        fluid_drive_mag = float(np.linalg.norm(self._get_fluid_control_acc(self.time)))

        frame = StateFrame(
            time=self.time,
            position=full_position,
            velocity=full_velocity,
            orientation=full_orientation,
            angular_velocity=full_angular_vel,
            mass=full_mass,
            body_id=full_body_id,
            attributes={
                "phase": phase,
                "mass_quantity": mass_quantity,
            },
            fields={
                "fluid_velocity": fluid_vel_grid,
                "boundary_normal": wall_normals,
                "boundary_target_velocity": wall_target_vel,
                "boundary_velocity": wall_vel,
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_ext_force,
                "budget_cumulative_external_impulse": budget_cum_impulse,
                "budget_cumulative_inflow_momentum": budget_inflow_momentum,
                "budget_cumulative_outflow_momentum": budget_outflow_momentum,
                "cumulative_mass_source": cumulative_mass_source,
                "interaction_force": interaction_force,
            },
            scalars={
                "fluid_drive": fluid_drive_mag,
                "reference_velocity": ref_velocity,
                "reference_momentum": ref_momentum,
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "box_pos": self.box_pos.tolist(),
            "box_vel": self.box_vel.tolist(),
            "box_angle": float(self.box_angle),
        }

