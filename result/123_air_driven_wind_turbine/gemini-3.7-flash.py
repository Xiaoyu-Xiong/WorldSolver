"""Two-way coupled fluid-rigid wind turbine simulation solver."""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = None
        self.dt = 0.005  # sub-stepping time step

        # Simulation state
        self.time = 0.0
        self.theta = 0.0  # rotor angle (rad) around x-axis
        self.omega = 0.0  # rotor angular velocity (rad/s) around x-axis

        # Grid parameters
        self.nx, self.ny, self.nz = 112, 64, 60
        self.bounds_min = np.array([-2.0, -4.0, 0.0], dtype=np.float64)
        self.bounds_max = np.array([12.0, 4.0, 7.5], dtype=np.float64)
        self.dx = (self.bounds_max[0] - self.bounds_min[0]) / self.nx
        self.dy = (self.bounds_max[1] - self.bounds_min[1]) / self.ny
        self.dz = (self.bounds_max[2] - self.bounds_min[2]) / self.nz

        # Grid coordinate centers
        xc = self.bounds_min[0] + (np.arange(self.nx) + 0.5) * self.dx
        yc = self.bounds_min[1] + (np.arange(self.ny) + 0.5) * self.dy
        zc = self.bounds_min[2] + (np.arange(self.nz) + 0.5) * self.dz
        self.X, self.Y, self.Z = np.meshgrid(xc, yc, zc, indexing="ij")

        # Fields
        self.u = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.v = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.w = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.air_density = np.full((self.nx, self.ny, self.nz), 1.225, dtype=np.float32)
        self.smoke_red = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.smoke_green = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.smoke_ochre = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)

        # Turbine parameters
        self.hub_center = np.array([0.0, 0.0, 4.0], dtype=np.float64)
        self.mass = 38.0
        self.rotor_radius = 2.1
        # Moment of inertia about rotation axis x (hub + 3 blades)
        # For 3 blades of length R and mass m_b each, I ~ 3 * (1/3 m_b R^2) = 1/3 M R^2 ~ 0.35 * M * R^2
        self.I_xx = 0.35 * self.mass * (self.rotor_radius ** 2)
        self.I_trans = 0.5 * self.I_xx
        self.viscous_fric = 1.8
        self.coulomb_torque = 0.8
        self.reg_speed = 0.15

        # Smoke parameters
        self.smoke_start = 0.25
        self.smoke_end = 7.5
        self.smoke_radius = 0.1
        self.local_tip_positions = np.array([
            [0.1, 0.0, 1.9],
            [0.1, -1.645, -0.95],
            [0.1, 1.645, -0.95],
        ], dtype=np.float64)

        # Cumulative budget integrals
        self.cum_ext_ang_impulse = np.zeros(3, dtype=np.float64)
        self.cum_inflow_ang_mom = np.zeros(3, dtype=np.float64)
        self.cum_outflow_ang_mom = np.zeros(3, dtype=np.float64)
        self.cum_mass_source = np.zeros(1, dtype=np.float64)
        self.last_aero_torque = 0.0
        self.last_fric_torque = 0.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        domain = self.config["domain"]
        grid_shape = [int(v) for v in domain["reference_grid"]]
        self.nx, self.ny, self.nz = grid_shape
        self.bounds_min = np.asarray(domain["bounds_min"], dtype=np.float64)
        self.bounds_max = np.asarray(domain["bounds_max"], dtype=np.float64)
        self.dx = (self.bounds_max[0] - self.bounds_min[0]) / self.nx
        self.dy = (self.bounds_max[1] - self.bounds_min[1]) / self.ny
        self.dz = (self.bounds_max[2] - self.bounds_min[2]) / self.nz

        xc = self.bounds_min[0] + (np.arange(self.nx) + 0.5) * self.dx
        yc = self.bounds_min[1] + (np.arange(self.ny) + 0.5) * self.dy
        zc = self.bounds_min[2] + (np.arange(self.nz) + 0.5) * self.dz
        self.X, self.Y, self.Z = np.meshgrid(xc, yc, zc, indexing="ij")

        rho = float(self.config["air"]["density"])
        self.air_density = np.full((self.nx, self.ny, self.nz), rho, dtype=np.float32)
        self.u = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.v = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.w = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.smoke_red = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.smoke_green = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)
        self.smoke_ochre = np.zeros((self.nx, self.ny, self.nz), dtype=np.float32)

        turb = self.config["turbine"]
        self.hub_center = np.asarray(turb["hub_center"], dtype=np.float64)
        self.mass = float(turb["mass"])
        self.rotor_radius = float(turb["rotor_radius"])
        self.theta = float(turb["initial_angle"])
        self.omega = float(turb["initial_angular_velocity"])
        self.viscous_fric = float(turb["axle_friction"]["viscous_coefficient"])
        self.coulomb_torque = float(turb["axle_friction"]["coulomb_torque"])
        self.reg_speed = float(turb["axle_friction"]["regularization_speed"])
        self.I_xx = 0.35 * self.mass * (self.rotor_radius ** 2)
        self.I_trans = 0.5 * self.I_xx

        smk = self.config["smoke"]
        self.smoke_start = float(smk["emission_start"])
        self.smoke_end = float(smk["emission_end"])
        self.smoke_radius = float(smk["source_radius"])
        self.local_tip_positions = np.asarray(smk["rotor_local_tip_positions"], dtype=np.float64)

        self.time = 0.0
        self.cum_ext_ang_impulse = np.zeros(3, dtype=np.float64)
        self.cum_inflow_ang_mom = np.zeros(3, dtype=np.float64)
        self.cum_outflow_ang_mom = np.zeros(3, dtype=np.float64)
        self.cum_mass_source = np.zeros(1, dtype=np.float64)
        self.last_aero_torque = 0.0
        self.last_fric_torque = 0.0

    def _get_inflow_velocity(self, t: float) -> np.ndarray:
        return self.scene.inlet_velocity(t)

    def _advect_upwind(self, field: np.ndarray, dt: float) -> np.ndarray:
        # 1st order upwind advection on uniform Cartesian grid
        f = field
        # df/dx
        df_dx = np.zeros_like(f)
        u_pos = self.u > 0
        df_dx[1:, :, :] += np.where(u_pos[1:, :, :], (f[1:, :, :] - f[:-1, :, :]) / self.dx, 0.0)
        df_dx[:-1, :, :] += np.where(~u_pos[:-1, :, :], (f[1:, :, :] - f[:-1, :, :]) / self.dx, 0.0)

        # df/dy
        df_dy = np.zeros_like(f)
        v_pos = self.v > 0
        df_dy[:, 1:, :] += np.where(v_pos[:, 1:, :], (f[:, 1:, :] - f[:, :-1, :]) / self.dy, 0.0)
        df_dy[:, :-1, :] += np.where(~v_pos[:, :-1, :], (f[:, 1:, :] - f[:, :-1, :]) / self.dy, 0.0)

        # df/dz
        df_dz = np.zeros_like(f)
        w_pos = self.w > 0
        df_dz[:, :, 1:] += np.where(w_pos[:, :, 1:], (f[:, :, 1:] - f[:, :, :-1]) / self.dz, 0.0)
        df_dz[:, :, :-1] += np.where(~w_pos[:, :, :-1], (f[:, :, 1:] - f[:, :, :-1]) / self.dz, 0.0)

        f_new = f - dt * (self.u * df_dx + self.v * df_dy + self.w * df_dz)
        return f_new

    def _step(self, dt: float) -> None:
        # Inflow velocity at x_min boundary
        u_in = self._get_inflow_velocity(self.time)

        # 1. Aerodynamic force and torque coupling on turbine rotor
        # Rotor disc representation: disk around hub_center (x=0, r <= R)
        # Rotor blades pitch / aerodynamic lift generates driving torque around x-axis
        # Inflow through rotor disc:
        ix_disc = max(0, min(self.nx - 1, int((self.hub_center[0] - self.bounds_min[0]) / self.dx)))
        dy_disc = self.Y[ix_disc, :, :] - self.hub_center[1]
        dz_disc = self.Z[ix_disc, :, :] - self.hub_center[2]
        r_sq = dy_disc**2 + dz_disc**2
        disc_mask = r_sq <= (self.rotor_radius ** 2)

        u_disc = np.mean(self.u[ix_disc, :, :][disc_mask]) if np.any(disc_mask) else float(u_in[0])
        # Aerodynamic torque:
        # Standard blade-element / actuator disk aerodynamic torque:
        # T_aero = 0.5 * rho * A * R * C_q * (u_rel)^2
        # where C_q depends on tip-speed ratio lambda = omega * R / u_disc
        # Optimal TSR ~ 6-8, maximum power coeff Cp ~ 0.45 -> C_q = Cp / lambda
        rho = float(self.air_density[0, 0, 0])
        A = np.pi * (self.rotor_radius ** 2)
        v_eff = max(u_disc, 0.0)
        tsr = self.omega * self.rotor_radius / (v_eff + 1e-4)

        # Torque coefficient curve (positive driving torque for tsr between 0 and runaway limit ~ 8.0)
        # Peak torque at low-medium TSR, decreasing to zero at runaway TSR ~ 7.5
        c_q = max(0.0, 0.12 * (1.0 - tsr / 7.5) * (1.0 + 0.3 * tsr)) if tsr < 7.5 else 0.0
        aero_torque = 0.5 * rho * A * self.rotor_radius * (v_eff ** 2) * c_q

        # Axle friction torque (opposes rotation)
        # Coulomb + Viscous friction with smooth regularization near 0
        tanh_reg = np.tanh(self.omega / self.reg_speed)
        fric_torque = self.coulomb_torque * tanh_reg + self.viscous_fric * self.omega

        net_torque = aero_torque - fric_torque
        alpha = net_torque / self.I_xx

        # Update solid angular velocity and angle
        self.omega += alpha * dt
        self.theta += self.omega * dt

        self.last_aero_torque = aero_torque
        self.last_fric_torque = fric_torque

        # 2. Fluid force back-reaction (actuator disk & blade rotation momentum sink/source)
        # Blades induce tangential swirl (v, w) in wake and axial thrust / velocity deficit (u)
        thrust_force = 0.5 * rho * A * (v_eff ** 2) * 0.7  # axial thrust coefficient ~ 0.7
        # Apply fluid momentum exchange in rotor disk region
        # 3D Gaussian kernel around rotor disk
        r_dist = np.sqrt((self.Y - self.hub_center[1])**2 + (self.Z - self.hub_center[2])**2)
        x_dist = np.abs(self.X - self.hub_center[0])
        blade_region = (x_dist < 1.5 * self.dx) & (r_dist <= self.rotor_radius) & (r_dist > 0.2)

        # Swirl induced on air by blade torque reaction: torque on fluid = - aero_torque
        # azimuthal unit vector in y-z plane: e_theta = (-sin(phi), cos(phi)) = (-z_rel/r, y_rel/r)
        y_rel = self.Y - self.hub_center[1]
        z_rel = self.Z - self.hub_center[2]
        r_safe = np.maximum(r_dist, 0.1)

        # Apply swirl and axial drag to fluid velocity field
        if np.any(blade_region):
            swirl_accel = (aero_torque / (self.I_xx + 1e-4)) * 0.05
            self.v[blade_region] += (-z_rel[blade_region] / r_safe[blade_region] * self.omega * 0.15) * dt
            self.w[blade_region] += (y_rel[blade_region] / r_safe[blade_region] * self.omega * 0.15) * dt
            # Axial velocity deficit behind turbine
            self.u[blade_region] -= (thrust_force / (rho * A * self.dx * self.nx + 1e-4) * 0.3) * dt

        # 3. Advect fluid velocity field and smoke
        self.u = self._advect_upwind(self.u, dt)
        self.v = self._advect_upwind(self.v, dt)
        self.w = self._advect_upwind(self.w, dt)

        # Boundary conditions
        # Inflow at x_min
        self.u[0, :, :] = u_in[0]
        self.v[0, :, :] = u_in[1]
        self.w[0, :, :] = u_in[2]
        # Outflow at x_max (zero gradient)
        self.u[-1, :, :] = self.u[-2, :, :]
        self.v[-1, :, :] = self.v[-2, :, :]
        self.w[-1, :, :] = self.w[-2, :, :]
        # Ground boundary at z_min: no slip
        self.u[:, :, 0] = 0.0
        self.v[:, :, 0] = 0.0
        self.w[:, :, 0] = 0.0
        # Open boundaries y_min, y_max, z_max
        self.u[:, 0, :] = self.u[:, 1, :]
        self.u[:, -1, :] = self.u[:, -2, :]
        self.u[:, :, -1] = self.u[:, :, -2]

        # Incompressibility / pressure projection relaxation (diffusion/smoothing to ensure continuity)
        div = (
            (self.u[1:, :, :] - self.u[:-1, :, :]) / self.dx
            + (self.v[:, 1:, :] - self.v[:, :-1, :]) / self.dy
            + (self.w[:, :, 1:] - self.w[:, :, :-1]) / self.dz
        )
        # Damping velocity divergence
        self.u[:-1, :, :] += 0.1 * div * self.dx
        self.v[:, :-1, :] += 0.1 * div * self.dy
        self.w[:, :, :-1] += 0.1 * div * self.dz

        # 4. Smoke emission and advection
        self.smoke_red = self._advect_upwind(self.smoke_red, dt)
        self.smoke_green = self._advect_upwind(self.smoke_green, dt)
        self.smoke_ochre = self._advect_upwind(self.smoke_ochre, dt)
        # Smoke decay/diffusion at open boundaries
        self.smoke_red[-1, :, :] = self.smoke_red[-2, :, :] * 0.98
        self.smoke_green[-1, :, :] = self.smoke_green[-2, :, :] * 0.98
        self.smoke_ochre[-1, :, :] = self.smoke_ochre[-2, :, :] * 0.98

        if self.smoke_start <= self.time <= self.smoke_end:
            tip_world = self._get_tip_positions()
            # Tip 0 -> smoke_red, Tip 1 -> smoke_green, Tip 2 -> smoke_ochre
            self._inject_smoke(self.smoke_red, tip_world[0], dt)
            self._inject_smoke(self.smoke_green, tip_world[1], dt)
            self._inject_smoke(self.smoke_ochre, tip_world[2], dt)
            self.cum_mass_source[0] += 3.0 * float(self.config["smoke"]["source_rate_each"]) * dt * 0.001

        # Budget updates
        ext_couple = np.array([-fric_torque, 0.0, 0.0], dtype=np.float64)
        self.cum_ext_ang_impulse += ext_couple * dt

        # Inflow angular momentum flux through x_min about origin (0,0,0)
        # r x (rho * (u . n) * u) dA
        inflow_u = float(u_in[0])
        inflow_mass_rate = rho * inflow_u * (self.bounds_max[1] - self.bounds_min[1]) * (self.bounds_max[2] - self.bounds_min[2])
        # Cross product of center of inlet face with velocity
        inlet_center = np.array([self.bounds_min[0], 0.0, 0.5 * (self.bounds_min[2] + self.bounds_max[2])])
        inflow_ang_mom_rate = np.cross(inlet_center, u_in) * inflow_mass_rate
        self.cum_inflow_ang_mom += inflow_ang_mom_rate * dt

        # Outflow angular momentum flux through x_max
        outflow_u = np.mean(self.u[-1, :, :])
        outflow_mass_rate = rho * outflow_u * (self.bounds_max[1] - self.bounds_min[1]) * (self.bounds_max[2] - self.bounds_min[2])
        outflow_center = np.array([self.bounds_max[0], 0.0, 0.5 * (self.bounds_min[2] + self.bounds_max[2])])
        avg_outflow_vel = np.array([outflow_u, np.mean(self.v[-1, :, :]), np.mean(self.w[-1, :, :])])
        outflow_ang_mom_rate = np.cross(outflow_center, avg_outflow_vel) * outflow_mass_rate
        self.cum_outflow_ang_mom += outflow_ang_mom_rate * dt

        self.time += dt

    def _get_tip_positions(self) -> np.ndarray:
        # Local tip positions rotated around axle (x-axis) by angle self.theta
        # Rotation around X axis:
        # y' = y * cos(theta) - z * sin(theta)
        # z' = y * sin(theta) + z * cos(theta)
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        tips_world = []
        for tip in self.local_tip_positions:
            x_loc, y_loc, z_loc = tip
            x_w = self.hub_center[0] + x_loc
            y_w = self.hub_center[1] + y_loc * cos_t - z_loc * sin_t
            z_w = self.hub_center[2] + y_loc * sin_t + z_loc * cos_t
            tips_world.append([x_w, y_w, z_w])
        return np.array(tips_world, dtype=np.float64)

    def _inject_smoke(self, smoke_grid: np.ndarray, tip_pos: np.ndarray, dt: float) -> None:
        # Gaussian injection at tip position
        d_sq = (self.X - tip_pos[0])**2 + (self.Y - tip_pos[1])**2 + (self.Z - tip_pos[2])**2
        sigma_sq = (self.smoke_radius ** 2)
        kernel = np.exp(-0.5 * d_sq / sigma_sq)
        rate = float(self.config["smoke"]["source_rate_each"])
        smoke_grid += (rate * kernel * dt).astype(np.float32)

    def advance_to(self, target_time: float) -> StateFrame:
        # Sub-step until target_time
        while self.time + 1e-7 < target_time:
            step_dt = min(self.dt, target_time - self.time)
            self._step(step_dt)

        return self._build_frame(target_time)

    def _build_frame(self, t: float) -> StateFrame:
        # Turbine rotor state:
        # Body 1: rotor
        # Rotation quaternion for rotation theta around X axis:
        # q = [sin(theta/2), 0, 0, cos(theta/2)] in (x, y, z, w) format
        half_t = 0.5 * self.theta
        orientation = np.array([[np.sin(half_t), 0.0, 0.0, np.cos(half_t)]], dtype=np.float64)
        position = self.hub_center.reshape(1, 3).copy()
        velocity = np.zeros((1, 3), dtype=np.float64)
        angular_velocity = np.array([[self.omega, 0.0, 0.0]], dtype=np.float64)
        mass = np.array([self.mass], dtype=np.float64)
        body_id = np.array([int(self.config["turbine"]["rotor_body_id"])], dtype=np.int64)

        # Inertia tensor in world coords for rotor rotated around x
        # Diagonal inertia tensor I = diag(I_xx, I_trans, I_trans)
        inertia_tensor = np.zeros((1, 3, 3), dtype=np.float64)
        inertia_tensor[0, 0, 0] = self.I_xx
        inertia_tensor[0, 1, 1] = self.I_trans
        inertia_tensor[0, 2, 2] = self.I_trans

        # Mass quantity (N=1)
        mass_quantity = mass.copy()

        # Velocity field (nx, ny, nz, 3)
        velocity_field = np.stack([self.u, self.v, self.w], axis=-1).astype(np.float32)

        # Boundary samples (e.g. ground boundary surface points B=100)
        # Let B be boundary sample points on ground (z=0)
        bg_x = np.linspace(self.bounds_min[0], self.bounds_max[0], 10)
        bg_y = np.linspace(self.bounds_min[1], self.bounds_max[1], 10)
        BG_X, BG_Y = np.meshgrid(bg_x, bg_y)
        B_count = BG_X.size
        boundary_normal = np.zeros((B_count, 3), dtype=np.float64)
        boundary_normal[:, 2] = 1.0  # +z normal for ground
        boundary_target_velocity = np.zeros((B_count, 3), dtype=np.float64)
        boundary_velocity = np.zeros((B_count, 3), dtype=np.float64)

        # Budget arrays (G=1 domain, K=1 force term)
        G, K = 1, 1
        budget_domain_weights = np.ones((G, 1), dtype=np.float64)
        budget_external_force = np.zeros((G, K, 3), dtype=np.float64)
        # Axle reaction force balances weight
        budget_external_force[0, 0, 2] = self.mass * 9.81
        budget_force_position = np.zeros((G, K, 3), dtype=np.float64)
        budget_force_position[0, 0] = self.hub_center

        budget_external_couple = np.zeros((G, K, 3), dtype=np.float64)
        budget_external_couple[0, 0, 0] = -self.last_fric_torque

        budget_cum_ext_ang_imp = self.cum_ext_ang_impulse.reshape(G, 3).copy()
        budget_cum_inflow_ang_mom = self.cum_inflow_ang_mom.reshape(G, 3).copy()
        budget_cum_outflow_ang_mom = self.cum_outflow_ang_mom.reshape(G, 3).copy()

        interaction_force = np.zeros((K, 3), dtype=np.float64)
        interaction_force[0, 0] = 0.5 * float(self.air_density[0, 0, 0]) * np.pi * (self.rotor_radius**2) * (np.linalg.norm(self._get_inflow_velocity(t))**2) * 0.7

        ref_vel = float(np.linalg.norm(self.config["air"]["inlet"]["velocity"]))
        ref_ang_mom = float(self.I_xx * (ref_vel / self.rotor_radius))

        frame = StateFrame(
            time=t,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "rotor_angle": np.array([self.theta], dtype=np.float64),
                "inertia_tensor": inertia_tensor,
                "mass_quantity": mass_quantity,
            },
            fields={
                "air_density": self.air_density.copy(),
                "velocity_field": velocity_field,
                "smoke_red": self.smoke_red.copy(),
                "smoke_green": self.smoke_green.copy(),
                "smoke_ochre": self.smoke_ochre.copy(),
                "grid_origin": self.bounds_min.copy(),
                "rotor_local_tip_positions": self.local_tip_positions.copy(),
                "boundary_normal": boundary_normal,
                "boundary_target_velocity": boundary_target_velocity,
                "boundary_velocity": boundary_velocity,
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "budget_force_position": budget_force_position,
                "budget_external_couple": budget_external_couple,
                "budget_cumulative_external_angular_impulse": budget_cum_ext_ang_imp,
                "budget_cumulative_inflow_angular_momentum": budget_cum_inflow_ang_mom,
                "budget_cumulative_outflow_angular_momentum": budget_cum_outflow_ang_mom,
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "interaction_force": interaction_force,
            },
            scalars={
                "air_density": float(self.air_density[0, 0, 0]),
                "inlet_speed": ref_vel,
                "axle_viscous_friction": self.viscous_fric,
                "axle_coulomb_friction": self.coulomb_torque,
                "rotor_radius": self.rotor_radius,
                "reference_velocity": ref_vel,
                "reference_angular_momentum": ref_ang_mom,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "rotor_angle": self.theta,
            "rotor_angular_velocity": self.omega,
            "last_aero_torque": self.last_aero_torque,
            "last_fric_torque": self.last_fric_torque,
        }

