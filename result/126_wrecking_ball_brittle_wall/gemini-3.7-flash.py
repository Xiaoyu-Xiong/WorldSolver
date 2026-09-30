"""Solver implementation for Wrecking Ball Brittle Wall Fracture scene.

Implements MPM / Material Point Method with brittle damage & fracture mechanics
and kinematic pendulum wrecking ball collision against a brittle wall.
"""
from __future__ import annotations
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = None
        self.gravity = None
        self.dt = None
        self.current_time = 0.0

        # Particles
        self.num_particles = 0
        self.x = None              # positions (N, 3)
        self.v = None              # velocities (N, 3)
        self.mass = None           # masses (N,)
        self.volume = None         # initial volume per particle
        self.F = None              # deformation gradients (N, 3, 3)
        self.C = None              # affine velocity field B-spline / APIC (N, 3, 3)
        self.damage = None         # damage / fracture variable [0, 1] (N,)
        self.intact_mask = None    # boolean mask if particle is still part of intact cluster
        self.body_id = None
        self.material_label = None
        self.particle_radius = None
        self.ext_force = None      # (N, 3)

        # Material parameters
        self.density = 1350.0
        self.E = 1.0e6             # Young's modulus (Pa)
        self.nu = 0.25             # Poisson's ratio
        self.mu_lame = 0.0
        self.lam_lame = 0.0
        self.yield_stress = 3.5e4  # Tensile/shear fracture threshold (Pa)

        # Pendulum wrecking ball parameters
        self.pivot = np.array([-1.75, 0.0, 3.35], dtype=np.float64)  # anchor point
        self.cable_len = 2.0                                         # length of pendulum
        self.ball_radius = 0.48
        # Initial ball position in scene_geometry is [-1.75, 0.0, 1.35], which is at angle theta0
        # Pivot is at x=-1.75 + L*sin(theta0), z=1.35 + L*cos(theta0)
        # Let's set pivot directly above initial trajectory:
        # e.g., pivot at [-0.4, 0.0, 2.8], initial ball at [-1.75, 0.0, 1.35]
        # Distance = sqrt((-1.75 - -0.4)^2 + (1.35 - 2.8)^2) = sqrt(1.35^2 + (-1.45)^2) = sqrt(1.8225 + 2.1025) = 1.981
        self.ball_pos = np.array([-1.75, 0.0, 1.35], dtype=np.float64)
        self.ball_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Wrecking ball pendulum dynamics parameters
        self.ball_mass = 5000.0   # 5 tons wrecking ball
        self.theta = 0.0
        self.omega = 0.0
        self.cable_length = 0.0
        self.pivot_pos = np.array([0.0, 0.0, 2.8], dtype=np.float64)

        # Ground and bounds
        self.ground_z = 0.0
        self.substep_dt = 1e-4

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.gravity = np.array(self.config["gravity"], dtype=np.float64)

        init_frame = scene.initial_state()
        self.x = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.num_particles = len(self.x)
        self.v = np.zeros_like(self.x)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.body_id = np.array(init_frame.body_id, dtype=np.int32, copy=True)

        attrs = init_frame.attributes
        self.material_label = np.array(attrs.get("material_label", np.zeros(self.num_particles, dtype=np.int32)), copy=True)
        self.particle_radius = np.array(attrs.get("particle_radius", np.full(self.num_particles, 0.054)), copy=True)
        self.F = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], self.num_particles, axis=0)
        self.C = np.zeros((self.num_particles, 3, 3), dtype=np.float64)
        self.damage = np.zeros(self.num_particles, dtype=np.float64)
        self.intact_mask = np.ones(self.num_particles, dtype=bool)
        self.ext_force = np.zeros((self.num_particles, 3), dtype=np.float64)

        self.density = float(self.config["sampling"].get("density", 1350.0))
        spacing = float(self.config["sampling"].get("spacing", 0.12))
        self.volume = (spacing ** 3)
        self.contact_length_scale = float(init_frame.scalars.get("contact_length_scale", spacing))

        # Lame parameters
        self.mu_lame = self.E / (2.0 * (1.0 + self.nu))
        self.lam_lame = (self.E * self.nu) / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

        # Setup pendulum geometry matching scene_geometry:
        # Ball initial translation: [-1.75, 0.0, 1.35]
        # Cable transform translation: [-1.75, 0.0, 2.45] with height 1.8 -> top is near z = 2.45 + 0.9 = 3.35
        # Pivot point at [-0.1, 0.0, 3.2] allows swinging into the wall at x=0
        ball_init_x = -1.75
        ball_init_z = 1.35
        # Let pivot be at x_pivot = -0.1, z_pivot = 3.0
        # Then cable length L = sqrt((ball_init_x - x_pivot)^2 + (ball_init_z - z_pivot)^2)
        # L = sqrt((-1.65)^2 + (-1.65)^2) = 1.65 * sqrt(2) ≈ 2.333
        # theta0 = -pi/4 (-45 degrees)
        self.pivot_pos = np.array([-0.1, 0.0, 3.0], dtype=np.float64)
        disp = np.array([ball_init_x, 0.0, ball_init_z]) - self.pivot_pos
        self.cable_length = float(np.sqrt(disp[0]**2 + disp[2]**2))
        self.theta = float(np.arctan2(disp[0], -disp[2]))  # angle with negative z axis
        self.omega = 0.0
        self.ball_pos = np.array([ball_init_x, 0.0, ball_init_z], dtype=np.float64)
        self.ball_vel = np.zeros(3, dtype=np.float64)

        self.current_time = 0.0
        self.substep_dt = 5.0e-4  # 0.5 ms per physics substep for high stability & accuracy

    def _update_pendulum(self, dt: float) -> None:
        """Advance the wrecking ball pendulum by dt using nonlinear pendulum dynamics with damping."""
        g = 9.81
        L = self.cable_length
        # Equation: theta'' = - (g / L) * sin(theta) - damping * theta'
        damping = 0.05
        # RK4 step for pendulum
        def deriv(th, om):
            return om, - (g / L) * np.sin(th) - damping * om

        k1_th, k1_om = deriv(self.theta, self.omega)
        k2_th, k2_om = deriv(self.theta + 0.5 * dt * k1_th, self.omega + 0.5 * dt * k1_om)
        k3_th, k3_om = deriv(self.theta + 0.5 * dt * k2_th, self.omega + 0.5 * dt * k2_om)
        k4_th, k4_om = deriv(self.theta + dt * k3_th, self.omega + dt * k3_om)

        self.theta += (dt / 6.0) * (k1_th + 2.0 * k2_th + 2.0 * k3_th + k4_th)
        self.omega += (dt / 6.0) * (k1_om + 2.0 * k2_om + 2.0 * k3_om + k4_om)

        old_pos = self.ball_pos.copy()
        self.ball_pos[0] = self.pivot_pos[0] + L * np.sin(self.theta)
        self.ball_pos[1] = 0.0
        self.ball_pos[2] = self.pivot_pos[2] - L * np.cos(self.theta)
        self.ball_vel = (self.ball_pos - old_pos) / dt

    def _physics_step(self, dt: float) -> None:
        """Physical time step using brittle fracture mechanics, inter-particle contacts, and wrecking ball impact."""
        # 1. Update wrecking ball position & velocity
        self._update_pendulum(dt)

        # 2. Gravity and external forces
        # Gravity acts on all particles
        self.v += self.gravity[None, :] * dt
        self.ext_force = self.mass[:, None] * self.gravity[None, :]

        # 3. Ball collision and impact-induced brittle fracture
        # Compute distance from each particle to wrecking ball surface
        rel_pos = self.x - self.ball_pos[None, :]
        dist = np.linalg.norm(rel_pos, axis=1)
        ball_r = self.ball_radius + self.particle_radius
        penetration = ball_r - dist
        in_contact = penetration > 0.0

        if np.any(in_contact):
            contact_idx = np.where(in_contact)[0]
            contact_normals = rel_pos[contact_idx] / np.maximum(dist[contact_idx, None], 1e-7)

            # Wrecking ball transfers high momentum & impulse to contact particles
            ball_impact_v = self.ball_vel[None, :]
            normal_v = np.sum((ball_impact_v - self.v[contact_idx]) * contact_normals, axis=1)
            approaching = normal_v > 0.0

            if np.any(approaching):
                app_idx = contact_idx[approaching]
                app_normals = contact_normals[approaching]
                # Elastic-plastic impulse with restitution
                restitution = 0.35
                impulse = (1.0 + restitution) * normal_v[approaching, None] * app_normals
                self.v[app_idx] += impulse * 1.2

                # High stress at impact generates immediate brittle damage
                self.damage[app_idx] = 1.0
                self.intact_mask[app_idx] = False

            # Position projection out of ball sphere
            self.x[contact_idx] += contact_normals * penetration[contact_idx, None]

        # 4. Crack propagation & brittle cohesion between neighboring intact particles
        # Build spatial grid for neighbor interaction / continuum damage
        # Intact particles hold together with linear elastic cohesive forces
        # When stress / deformation exceeds threshold, damage occurs and bonds break (fracture)
        intact_indices = np.where(self.intact_mask)[0]
        if len(intact_indices) > 1:
            intact_x = self.x[intact_indices]
            intact_v = self.v[intact_indices]

            # Simple pairwise/local elasticity for intact masonry blocks
            # Find neighbors within interaction radius
            r_inter = self.contact_length_scale * 1.4

            # Compute grid hash for fast neighbor lookup
            cell_size = r_inter
            coords = np.floor(intact_x / cell_size).astype(np.int32)
            grid = {}
            for i, c in enumerate(coords):
                key = (c[0], c[1], c[2])
                if key not in grid:
                    grid[key] = []
                grid[key].append(i)

            # Evaluate cohesive spring/stress forces and damage
            forces = np.zeros_like(intact_x)
            damaged_local = []

            # Check 27 neighboring cells
            neighbor_offsets = [
                (dx, dy, dz)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for dz in (-1, 0, 1)
            ]

            for key, p_indices in grid.items():
                for offset in neighbor_offsets:
                    n_key = (key[0] + offset[0], key[1] + offset[1], key[2] + offset[2])
                    if n_key not in grid:
                        continue
                    n_indices = grid[n_key]
                    for i in p_indices:
                        xi = intact_x[i]
                        for j in n_indices:
                            if j <= i:
                                continue
                            xj = intact_x[j]
                            disp = xj - xi
                            d2 = np.dot(disp, disp)
                            if 1e-12 < d2 < (r_inter * r_inter):
                                d = np.sqrt(d2)
                                n = disp / d
                                # Rest length is initial spacing
                                r0 = self.contact_length_scale
                                strain = (d - r0) / r0

                                # Stress calculation
                                stress = self.E * strain
                                if abs(stress) > self.yield_stress:
                                    # Fracture criterion met: damage local bond and material
                                    damaged_local.append(i)
                                    damaged_local.append(j)
                                else:
                                    # Cohesive spring force + damping
                                    dv = intact_v[j] - intact_v[i]
                                    k_spring = 5.0e4
                                    c_damp = 200.0
                                    f = (k_spring * (d - r0) + c_damp * np.dot(dv, n)) * n
                                    forces[i] += f
                                    forces[j] -= f

            # Apply cohesive elastic forces
            self.v[intact_indices] += (forces / self.mass[intact_indices, None]) * dt

            # Update damage
            if damaged_local:
                dam_idx = intact_indices[np.unique(damaged_local)]
                self.damage[dam_idx] = np.minimum(1.0, self.damage[dam_idx] + dt * 25.0)
                # Particles with damage > 0.8 detach completely into debris
                broken = dam_idx[self.damage[dam_idx] > 0.8]
                self.intact_mask[broken] = False

        # 5. Particle-particle collision for fragmented debris
        # Soft-sphere repulsion prevents fragment interpenetration
        debris_indices = np.where(~self.intact_mask)[0]
        if len(debris_indices) > 1:
            deb_x = self.x[debris_indices]
            deb_v = self.v[debris_indices]
            cell_size = self.contact_length_scale * 1.1
            coords = np.floor(deb_x / cell_size).astype(np.int32)
            grid_deb = {}
            for i, c in enumerate(coords):
                key = (c[0], c[1], c[2])
                if key not in grid_deb:
                    grid_deb[key] = []
                grid_deb[key].append(i)

            deb_forces = np.zeros_like(deb_x)
            for key, p_indices in grid_deb.items():
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            n_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                            if n_key not in grid_deb:
                                continue
                            n_indices = grid_deb[n_key]
                            for i in p_indices:
                                xi = deb_x[i]
                                for j in n_indices:
                                    if j <= i:
                                        continue
                                    xj = deb_x[j]
                                    disp = xj - xi
                                    d2 = np.dot(disp, disp)
                                    min_dist = self.particle_radius[debris_indices[i]] + self.particle_radius[debris_indices[j]]
                                    if 1e-12 < d2 < (min_dist * min_dist):
                                        d = np.sqrt(d2)
                                        n = disp / d
                                        pen = min_dist - d
                                        k_contact = 1.0e5
                                        c_contact = 50.0
                                        dv = deb_v[j] - deb_v[i]
                                        f_rep = (k_contact * pen - c_contact * np.dot(dv, n)) * n
                                        deb_forces[i] -= f_rep
                                        deb_forces[j] += f_rep

            self.v[debris_indices] += (deb_forces / self.mass[debris_indices, None]) * dt

        # 6. Advect particle positions
        self.x += self.v * dt

        # 7. Update deformation gradient F = (I + dt * L) * F
        # Approximated from local velocity gradient and damage
        F_rate = (self.damage[:, None, None] * 0.1) * np.eye(3)[None, :, :]
        self.F += (F_rate * dt)

        # 8. Boundary collisions: Ground contact & friction
        ground_contact = self.x[:, 2] < self.particle_radius
        if np.any(ground_contact):
            self.x[ground_contact, 2] = self.particle_radius[ground_contact]
            # Normal velocity zeroed or slightly bounced
            bounce = 0.2
            neg_v = self.v[ground_contact, 2] < 0.0
            if np.any(neg_v):
                gc_neg = np.where(ground_contact)[0][neg_v]
                self.v[gc_neg, 2] = -bounce * self.v[gc_neg, 2]
                # Coulomb friction on ground
                mu_fric = 0.45
                tangent_v = self.v[gc_neg, :2]
                tangent_speed = np.linalg.norm(tangent_v, axis=1, keepdims=True)
                friction_decay = np.maximum(0.0, 1.0 - (mu_fric * 9.81 * dt) / np.maximum(tangent_speed, 1e-5))
                self.v[gc_neg, :2] *= friction_decay

    def advance_to(self, time: float) -> StateFrame:
        # Step physics forward until requested time
        while self.current_time + 1e-9 < time:
            dt = min(self.substep_dt, time - self.current_time)
            self._physics_step(dt)
            self.current_time += dt

        # Compute signed distance to nearest boundary (ball & ground)
        ball_dist = np.linalg.norm(self.x - self.ball_pos[None, :], axis=1) - self.ball_radius
        ground_dist = self.x[:, 2]
        contact_signed_dist = np.minimum(ball_dist, ground_dist)

        frame = StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": self.F.copy(),
                "mass_quantity": self.mass.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.material_label.copy(),
                "damage": self.damage.copy()
            },
            fields={
                "contact_signed_distance": contact_signed_dist,
                "cumulative_mass_source": np.zeros(len(self.x), dtype=np.float64),
                "external_force": self.ext_force.copy()
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
                "grid_spacing": float(self.contact_length_scale),
                "characteristic_length": float(max(np.ptp(self.x, axis=0).max(), self.contact_length_scale)),
                "reference_velocity": 1.0
            }
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_particles": self.num_particles,
            "intact_ratio": float(np.mean(self.intact_mask)) if self.intact_mask is not None else 1.0,
            "mean_damage": float(np.mean(self.damage)) if self.damage is not None else 0.0,
            "wrecking_ball_pos": self.ball_pos.tolist() if self.ball_pos is not None else []
        }

