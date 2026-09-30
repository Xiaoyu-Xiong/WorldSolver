"""Physically-based Drucker-Prager / granular MPM solver for sand ball impact."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.current_time = 0.0
        self.frame_idx = 0

        # State arrays
        self.positions = None
        self.velocities = None
        self.masses = None
        self.body_ids = None
        self.deformation_gradients = None
        self.F_e = None  # Elastic part of deformation gradient
        self.F_p = None  # Plastic part

        # Rigid ball state
        self.ball_idx = None
        self.ball_radius = 0.54
        self.ball_mass = None

        # Physical constants
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.dt = 1e-4  # Simulation substep dt

        # Sand material parameters (Drucker-Prager)
        self.E = 1.0e6      # Young's modulus (Pa)
        self.nu = 0.25      # Poisson's ratio
        self.mu_0 = self.E / (2.0 * (1.0 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.friction_angle = 35.0 * np.pi / 180.0  # ~35 deg for sand
        self.sin_phi = np.sin(self.friction_angle)
        self.alpha_dp = np.sqrt(2.0 / 3.0) * (2.0 * self.sin_phi) / (3.0 - self.sin_phi)

        # MPM Grid parameters
        self.dx = 0.08
        self.grid_origin = np.array([-3.2, -1.6, -0.2])
        self.grid_size = np.array([80, 40, 52], dtype=int)
        self.inv_dx = 1.0 / self.dx

        # Trajectory metadata and buffers
        self.spacing = 0.12
        self.last_contact_sds = np.array([0.0])

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.positions = init_frame.position.copy()
        self.velocities = init_frame.velocity.copy()
        self.masses = init_frame.mass.copy()
        self.body_ids = init_frame.body_id.copy()

        self.num_particles = len(self.positions)
        self.sand_mask = (self.body_ids == 0)
        self.ball_idx = np.where(self.body_ids == 1)[0][0]

        self.ball_mass = self.masses[self.ball_idx]
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]))
        self.spacing = float(scene.config.get("sampling", {}).get("spacing", 0.12))

        # Deformation gradients
        self.F_e = np.repeat(np.eye(3)[None], self.num_particles, axis=0)
        self.F_p = np.repeat(np.eye(3)[None], self.num_particles, axis=0)
        self.deformation_gradients = np.repeat(np.eye(3)[None], self.num_particles, axis=0)
        self.particle_volumes = self.masses / float(scene.config.get("sampling", {}).get("density", 1350.0))

        # Substep dt based on sound speed and CFL
        c = np.sqrt(self.E / 1350.0)  # ~27 m/s
        self.dt = min(0.001, 0.4 * self.dx / c)

        self.current_time = 0.0
        self.last_contact_sds = np.array([0.0])

    def _project_drucker_prager(self, F_trial: np.ndarray) -> np.ndarray:
        """Project elastic deformation gradient onto Drucker-Prager yield surface (Stomakhin/Klár et al.)."""
        U, sig, Vt = np.linalg.svd(F_trial)
        # Avoid log of zero or negative
        sig = np.clip(sig, 1e-5, 1e5)
        epsilon = np.log(sig)
        trace_eps = np.sum(epsilon)
        eps_dev = epsilon - (trace_eps / 3.0)
        norm_dev = np.linalg.norm(eps_dev)

        if trace_eps >= 0:
            # Complete tension yielding: all stresses zero -> elastic strain zero
            epsilon_new = np.zeros(3)
        elif norm_dev <= 1e-12:
            epsilon_new = epsilon
        else:
            # Yield condition: ||eps_dev|| + (3*lambda + 2*mu)/(2*mu) * trace_eps * alpha <= 0
            # For Hencky elasticity:
            gamma = norm_dev + (3.0 * self.lambda_0 + 2.0 * self.mu_0) / (2.0 * self.mu_0) * trace_eps * self.alpha_dp
            if gamma > 0:
                epsilon_new = epsilon - (gamma / norm_dev) * eps_dev
            else:
                epsilon_new = epsilon

        sig_new = np.exp(epsilon_new)
        F_e_new = U @ np.diag(sig_new) @ Vt
        return F_e_new

    def _step_substep(self, dt: float) -> None:
        # Ball position and velocity
        ball_pos = self.positions[self.ball_idx]
        ball_vel = self.velocities[self.ball_idx]
        ball_r = self.ball_radius

        sand_indices = np.where(self.sand_mask)[0]
        sand_pos = self.positions[sand_indices]
        sand_vel = self.velocities[sand_indices]
        sand_mass = self.masses[sand_indices]
        n_sand = len(sand_indices)

        # 1. Update Sand MPM Grid
        gx, gy, gz = self.grid_size
        grid_mass = np.zeros((gx, gy, gz), dtype=np.float64)
        grid_mom = np.zeros((gx, gy, gz, 3), dtype=np.float64)

        # Particle stress calculation and B-spline weights
        stresses = np.zeros((n_sand, 3, 3), dtype=np.float64)

        # Compute Kirchhoff stress for sand particles from F_e
        for i_local, p in enumerate(sand_indices):
            Fe = self.F_e[p]
            U, sig, Vt = np.linalg.svd(Fe)
            sig = np.clip(sig, 1e-5, 1e5)
            eps = np.log(sig)
            # Second Piola-Kirchhoff or Kirchhoff stress tau
            # tau = 2 * mu * eps + lambda * tr(eps) * I
            tau_diag = 2.0 * self.mu_0 * eps + self.lambda_0 * np.sum(eps)
            tau = U @ np.diag(tau_diag) @ Vt
            # J = det(Fe)
            J = np.linalg.det(Fe)
            if J > 1e-5:
                cauchy_stress = tau / J
            else:
                cauchy_stress = tau
            stresses[i_local] = cauchy_stress * self.particle_volumes[p]

        # P2G: Particle to Grid Transfer with APIC / PIC
        p_base = ((sand_pos - self.grid_origin) * self.inv_dx - 0.5).astype(int)
        p_base = np.clip(p_base, 0, [gx - 3, gy - 3, gz - 3])
        fx = (sand_pos - self.grid_origin) * self.inv_dx - p_base

        # Quadratic B-spline weights
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw = [fx - 1.5, -2.0 * (fx - 1.0), fx - 0.5]

        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    weight = w[ix][:, 0] * w[iy][:, 1] * w[iz][:, 2]
                    # Gradient of weight
                    dweight = np.stack([
                        dw[ix][:, 0] * w[iy][:, 1] * w[iz][:, 2] * self.inv_dx,
                        w[ix][:, 0] * dw[iy][:, 1] * w[iz][:, 2] * self.inv_dx,
                        w[ix][:, 0] * w[iy][:, 1] * dw[iz][:, 2] * self.inv_dx
                    ], axis=-1)

                    node_x = p_base[:, 0] + ix
                    node_y = p_base[:, 1] + iy
                    node_z = p_base[:, 2] + iz

                    # Momentum and mass contribution
                    m_contrib = weight * sand_mass
                    mom_contrib = m_contrib[:, None] * sand_vel

                    # Stress divergence force: - volume * stress * dweight
                    # stress dot dweight
                    force_contrib = -np.einsum('nij,nj->ni', stresses, dweight) * dt

                    np.add.at(grid_mass, (node_x, node_y, node_z), m_contrib)
                    np.add.at(grid_mom, (node_x, node_y, node_z), mom_contrib + force_contrib)

        # 2. Grid operations: gravity and boundary conditions
        active_nodes = grid_mass > 1e-12
        grid_vel = np.zeros_like(grid_mom)
        grid_vel[active_nodes] = grid_mom[active_nodes] / grid_mass[active_nodes, None]
        grid_vel[active_nodes] += self.gravity * dt

        # Grid ground collision (ground at z = 0.0)
        # z=0 plane corresponds to grid index:
        z_zero_idx = int(np.round((0.0 - self.grid_origin[2]) * self.inv_dx))
        for iz in range(0, max(1, z_zero_idx + 1)):
            grid_vel[:, :, iz, 2] = np.maximum(0.0, grid_vel[:, :, iz, 2])
            # Friction on ground
            grid_vel[:, :, iz, 0] *= 0.5
            grid_vel[:, :, iz, 1] *= 0.5

        # Grid boundaries
        grid_vel[0:2, :, :, :] = 0.0
        grid_vel[gx-2:gx, :, :, :] = 0.0
        grid_vel[:, 0:2, :, :] = 0.0
        grid_vel[:, gy-2:gy, :, :] = 0.0
        grid_vel[:, :, 0:2, :] = 0.0

        # 3. G2P: Grid to Particle Transfer & Deformation Gradient Update
        new_sand_vel = np.zeros_like(sand_vel)
        grad_v = np.zeros((n_sand, 3, 3), dtype=np.float64)

        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    weight = w[ix][:, 0] * w[iy][:, 1] * w[iz][:, 2]
                    dweight = np.stack([
                        dw[ix][:, 0] * w[iy][:, 1] * w[iz][:, 2] * self.inv_dx,
                        w[ix][:, 0] * dw[iy][:, 1] * w[iz][:, 2] * self.inv_dx,
                        w[ix][:, 0] * w[iy][:, 1] * dw[iz][:, 2] * self.inv_dx
                    ], axis=-1)

                    node_x = p_base[:, 0] + ix
                    node_y = p_base[:, 1] + iy
                    node_z = p_base[:, 2] + iz

                    g_v = grid_vel[node_x, node_y, node_z]
                    new_sand_vel += weight[:, None] * g_v
                    grad_v += np.einsum('ni,nj->nij', g_v, dweight)

        # Update sand particle deformation gradients
        sand_pos += new_sand_vel * dt
        sand_vel = new_sand_vel

        # F_e update: F_trial = (I + dt * grad_v) @ F_e
        for i_local, p in enumerate(sand_indices):
            F_trial = (np.eye(3) + dt * grad_v[i_local]) @ self.F_e[p]
            # Drucker-Prager plasticity return mapping
            self.F_e[p] = self._project_drucker_prager(F_trial)
            self.deformation_gradients[p] = self.F_e[p]

        # 4. Rigid ball update & sand-ball interaction
        # Ball gravity
        ball_vel += self.gravity * dt
        ball_pos += ball_vel * dt

        # Ground collision for ball
        if ball_pos[2] - ball_r < 0.0:
            ball_pos[2] = ball_r
            if ball_vel[2] < 0:
                ball_vel[2] = 0.0
            ball_vel[0] *= 0.5
            ball_vel[1] *= 0.5

        # Ball-sand contact / collision resolution
        rel_pos = sand_pos - ball_pos
        dist = np.linalg.norm(rel_pos, axis=1)
        penetration = ball_r - dist
        contacting = penetration > 0

        if np.any(contacting):
            contact_indices = np.where(contacting)[0]
            normals = rel_pos[contact_indices] / np.maximum(dist[contact_indices, None], 1e-8)
            pen_depths = penetration[contact_indices]

            # Position correction for sand particles
            sand_pos[contact_indices] += normals * pen_depths[:, None]

            # Relative velocity
            v_rel = sand_vel[contact_indices] - ball_vel
            vn = np.sum(v_rel * normals, axis=1)

            # Approaching velocities
            approaching = vn < 0
            if np.any(approaching):
                app_contact = contact_indices[approaching]
                app_normals = normals[approaching]
                app_vn = vn[approaching]

                # Sand particle masses and ball mass coupling
                m_p = sand_mass[app_contact]
                m_b = self.ball_mass

                # Impulse for sand particles (with inelastic collision restitution e=0)
                # J = -(1 + e) * vn / (1/m_p + 1/m_b)
                # Tangential friction
                impulse_mag = -app_vn / (1.0 / m_p + 1.0 / m_b)
                sand_vel[app_contact] += app_normals * (impulse_mag / m_p)[:, None]

                # Reaction impulse on ball
                total_ball_impulse = -np.sum(app_normals * impulse_mag[:, None], axis=0)
                ball_vel += total_ball_impulse / m_b

                # Additional repulsive/frictional force on ball during deep penetration
                sand_vel[app_contact] += -0.1 * v_rel[approaching]

        # Store contact signed distance metrics
        self.last_contact_sds = np.minimum(dist - ball_r, sand_pos[:, 2])

        # Boundary clamp for sand positions to avoid flying to infinity
        sand_pos[:, 0] = np.clip(sand_pos[:, 0], -3.0, 3.0)
        sand_pos[:, 1] = np.clip(sand_pos[:, 1], -1.2, 1.2)
        sand_pos[:, 2] = np.maximum(sand_pos[:, 2], 0.0)

        # Update state
        self.positions[sand_indices] = sand_pos
        self.velocities[sand_indices] = sand_vel
        self.positions[self.ball_idx] = ball_pos
        self.velocities[self.ball_idx] = ball_vel

    def advance_to(self, time: float) -> StateFrame:
        if time < self.current_time:
            time = self.current_time

        target_time = float(time)
        while self.current_time + 1e-8 < target_time:
            dt = min(self.dt, target_time - self.current_time)
            if dt < 1e-9:
                break
            self._step_substep(dt)
            self.current_time += dt

        self.current_time = target_time

        # Construct StateFrame
        external_force = np.zeros_like(self.positions)
        external_force[:] = self.masses[:, None] * self.gravity

        contact_sd = self.last_contact_sds if len(self.last_contact_sds) > 0 else np.array([0.0])
        cum_mass = np.zeros(len(self.positions), dtype=np.float64)

        frame = StateFrame(
            self.current_time,
            self.positions.copy(),
            self.velocities.copy(),
            mass=self.masses.copy(),
            body_id=self.body_ids.copy(),
            attributes={
                "mass_quantity": self.masses.copy(),
                "particle_radius": np.full(len(self.positions), 0.45 * self.spacing),
                "material_label": self.body_ids.copy(),
                "deformation_gradient": self.deformation_gradients.copy(),
            },
            fields={
                "contact_signed_distance": contact_sd.astype(np.float64),
                "cumulative_mass_source": cum_mass,
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": float(self.spacing),
                "grid_spacing": float(self.spacing),
                "characteristic_length": float(max(float(np.ptp(self.positions, axis=0).max()), self.spacing)),
                "reference_velocity": 1.0,
            }
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": len(self.positions) if self.positions is not None else 0
        }

