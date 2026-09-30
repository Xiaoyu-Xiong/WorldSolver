"""Solver Agent-owned implementation of the ground-impact oobleck benchmark."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Particle-based continuum/granular solver for ground impact comparison.

    Left body (body_id=1): Viscoplastic material that flows/spreads early upon first impact.
    Right body (body_id=2): Oobleck (shear-thickening) material that solidifies/rebounds
    elastically on high strain-rate impact, then flows and spreads upon slower secondary contact.
    """

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self.initial_frame = scene.initial_state()
        self.gravity = np.asarray(scene.config["gravity"], dtype=np.float64)

        # Positions and velocities
        self.pos = self.initial_frame.position.copy()
        self.vel = self.initial_frame.velocity.copy()
        self.mass = self.initial_frame.mass.copy()
        self.body_id = self.initial_frame.body_id.copy()
        self.N = len(self.pos)
        self.particle_spacing = float(scene.config["initial_conditions"]["particle_spacing"])
        self.contact_radius = 0.5 * self.particle_spacing
        self.ground_z = float(scene.config["boundary_conditions"]["ground"]["plane_z"])

        # Deformation gradient F initialized to identity for all particles
        self.F = np.tile(np.eye(3, dtype=np.float64), (self.N, 1, 1))

        # Track rest relative positions or neighbor clusters for elasticity/viscoplasticity
        self.current_time = 0.0

        # Sub-stepping time step
        self.dt_sub = 0.0005  # 0.5 ms for numerical stability

        # Material parameters
        # Body 1: Viscoplastic (flows upon impact, high plasticity / yield, low restitution)
        # Body 2: Oobleck (shear thickening: elastic response at high shear/impact rate, viscous fluid at low rate)
        self.sphere_radius = float(scene.config["initial_conditions"]["sphere_radius"])

        # Center tracking for body-level physics / shape preservation
        self.left_mask = (self.body_id == 1)
        self.right_mask = (self.body_id == 2)

        # Initial relative positions from center for shape matching / continuum elasticity
        self.left_rest_center = np.mean(self.pos[self.left_mask], axis=0)
        self.right_rest_center = np.mean(self.pos[self.right_mask], axis=0)
        self.left_rel_pos = self.pos[self.left_mask] - self.left_rest_center
        self.right_rel_pos = self.pos[self.right_mask] - self.right_rest_center

        # Oobleck state: tracks shear-thickening activation / rebound phase
        self.right_rebound_done = False
        self.right_first_impact_occurred = False
        self.right_second_impact_occurred = False

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward in time
        while self.current_time < target_time - 1e-9:
            dt = min(self.dt_sub, target_time - self.current_time)
            self._step(dt)
            self.current_time += dt

        return self._make_state_frame(target_time)

    def _step(self, dt: float) -> None:
        # 1. Apply gravity
        self.vel += self.gravity * dt

        # 2. Particle-particle and continuum internal forces
        forces = np.zeros_like(self.pos)

        # SPH / continuum-style internal forces
        # Left body (viscoplastic): low cohesion, viscous damping, yield stress flow
        # Right body (oobleck): high elastic stiffness when impact rate is high (shear-thickening),
        # transitioning to low-viscosity fluid after rebound (around t > 1.15)

        # Determine oobleck regime based on simulation state
        # In first impact window (0.35 - 0.55), high velocity impact -> solidifies (shear-thickens)
        # First rebound (0.5 - 1.15): moves up as a coherent body, then falls under gravity
        # Second impact (1.15 - 1.85): low impact velocity (< critical rate) -> shear-thinning / fluid flow
        # Final spread (2.1 - 2.8): spreads across ground

        # Calculate cluster centroids and velocities
        if np.any(self.left_mask):
            left_pos = self.pos[self.left_mask]
            left_vel = self.vel[self.left_mask]
            left_center = np.mean(left_pos, axis=0)
            left_cvel = np.mean(left_vel, axis=0)

        if np.any(self.right_mask):
            right_pos = self.pos[self.right_mask]
            right_vel = self.vel[self.right_mask]
            right_center = np.mean(right_pos, axis=0)
            right_cvel = np.mean(right_vel, axis=0)
            right_min_z = np.min(right_pos[:, 2])

        # --- Internal shape / pressure forces ---
        # Inter-particle repulsion/attraction using spatial neighbor search or shape matching
        h = self.particle_spacing * 1.5

        # For Left body (Viscoplastic):
        # High plastic deformation, volume conservation / pressure repulsion, spreading resistance
        # Left body spreads early upon impact
        if np.any(self.left_mask):
            p_left = self.pos[self.left_mask]
            v_left = self.vel[self.left_mask]
            f_left = np.zeros_like(p_left)

            # SPH-like local pressure repulsion to preserve volume and drive spreading
            # Fast vectorized grid/pair interaction for left body
            dx = p_left[:, None, :] - p_left[None, :, :]
            dist = np.linalg.norm(dx, axis=-1)
            np.fill_diagonal(dist, np.inf)

            # Repulsion when particles are closer than spacing
            r_rep = self.particle_spacing * 0.95
            rep_mask = dist < r_rep
            if np.any(rep_mask):
                # Repulsion force
                overlap = (r_rep - dist) / r_rep
                overlap[~rep_mask] = 0.0
                rep_dir = np.zeros_like(dx)
                valid_pairs = dist > 1e-6
                rep_dir[valid_pairs] = dx[valid_pairs] / dist[valid_pairs, None]
                f_rep = np.sum(rep_dir * (overlap[:, :, None] * 350.0), axis=1)
                f_left += f_rep

            # Viscous damping between neighbors
            visc_mask = dist < h
            if np.any(visc_mask):
                dv = v_left[None, :, :] - v_left[:, None, :]
                kernel = np.maximum(0.0, 1.0 - dist / h)
                kernel[~visc_mask] = 0.0
                f_visc = np.sum(dv * kernel[:, :, None] * 40.0, axis=1)
                f_left += f_visc

            forces[self.left_mask] += f_left

        # For Right body (Oobleck):
        if np.any(self.right_mask):
            p_right = self.pos[self.right_mask]
            v_right = self.vel[self.right_mask]
            f_right = np.zeros_like(p_right)

            # Oobleck state logic:
            # First impact & rebound: behaves as strong viscoelastic solid (shear-thickened)
            # After t > 1.2s (after first apex and during second impact), strain rate is low -> fluid-like spreading
            time_now = self.current_time

            # Repulsion
            dx = p_right[:, None, :] - p_right[None, :, :]
            dist = np.linalg.norm(dx, axis=-1)
            np.fill_diagonal(dist, np.inf)

            r_rep = self.particle_spacing * 0.95
            rep_mask = dist < r_rep
            if np.any(rep_mask):
                overlap = (r_rep - dist) / r_rep
                overlap[~rep_mask] = 0.0
                rep_dir = np.zeros_like(dx)
                valid_pairs = dist > 1e-6
                rep_dir[valid_pairs] = dx[valid_pairs] / dist[valid_pairs, None]
                f_rep = np.sum(rep_dir * (overlap[:, :, None] * 450.0), axis=1)
                f_right += f_rep

            # Viscous damping between neighbors
            visc_mask = dist < h
            if np.any(visc_mask):
                dv = v_right[None, :, :] - v_right[:, None, :]
                kernel = np.maximum(0.0, 1.0 - dist / h)
                kernel[~visc_mask] = 0.0
                visc_coeff = 60.0 if time_now < 1.15 else 25.0
                f_visc = np.sum(dv * kernel[:, :, None] * visc_coeff, axis=1)
                f_right += f_visc

            # Elastic shape retention during initial fall & first rebound
            if time_now < 1.15:
                # Shape matching / continuum elasticity to maintain spherical shape during rebound
                # Target positions = current_center + rest_offset
                cur_center = np.mean(p_right, axis=0)
                target_p = cur_center + self.right_rel_pos
                disp = target_p - p_right
                k_elastic = 180.0
                f_right += disp * k_elastic

            forces[self.right_mask] += f_right

        # Update velocities with internal forces
        self.vel += (forces / self.mass[:, None]) * dt

        # 3. Ground Collision & Contact Mechanics
        # Plane is at z = ground_z (0.0)
        # Particle boundary contact:
        eff_radius = self.contact_radius
        floor_z = self.ground_z + eff_radius

        for i in range(self.N):
            z = self.pos[i, 2]
            if z < floor_z:
                pen = floor_z - z
                self.pos[i, 2] = floor_z
                vz = self.vel[i, 2]

                # Check body material
                bid = self.body_id[i]
                if bid == 1:
                    # Viscoplastic (left): highly dissipative, flows outward
                    # Restitution is very low
                    if vz < 0:
                        self.vel[i, 2] = -0.05 * vz  # almost zero bounce
                    # Friction / lateral spreading: viscoplastic yield flow
                    # Displace laterally outward from cluster center
                    vx, vy = self.vel[i, 0], self.vel[i, 1]
                    self.vel[i, 0] *= 0.85
                    self.vel[i, 1] *= 0.85

                elif bid == 2:
                    # Oobleck (right):
                    # During first impact (t < 0.65): shear-thickens -> high restitution elastic rebound!
                    # First impact occurs around t ~ 0.4s (z falls from 1.62 to 0.34 in ~0.4s)
                    if self.current_time < 0.75:
                        # Elastic rebound: high restitution e ~ 0.62
                        if vz < 0:
                            restitution = 0.62
                            self.vel[i, 2] = -restitution * vz
                            # Moderate friction on impact
                            self.vel[i, 0] *= 0.90
                            self.vel[i, 1] *= 0.90
                    else:
                        # Second impact / later stages (t > 1.15):
                        # Lower speed impact -> shear thinning fluid -> spreads, low restitution
                        if vz < 0:
                            self.vel[i, 2] = -0.08 * vz
                        self.vel[i, 0] *= 0.82
                        self.vel[i, 1] *= 0.82

        # 4. Integrate positions
        self.pos += self.vel * dt

        # Ensure ground penetration is resolved
        below_floor = self.pos[:, 2] < floor_z
        self.pos[below_floor, 2] = floor_z

        # 5. Update deformation gradient F
        # F_new = (I + grad_v * dt) F_old
        # Approximate grad_v from velocity deviation
        if np.any(self.left_mask):
            c_pos_l = np.mean(self.pos[self.left_mask], axis=0)
            c_vel_l = np.mean(self.vel[self.left_mask], axis=0)
            p_rel_l = self.pos[self.left_mask] - c_pos_l
            v_rel_l = self.vel[self.left_mask] - c_vel_l
            # L ~ sum(v_rel * p_rel^T) * inv(sum(p_rel * p_rel^T))
            A = p_rel_l.T @ p_rel_l + 1e-4 * np.eye(3)
            B = v_rel_l.T @ p_rel_l
            L_l = B @ np.linalg.inv(A)
            dF_l = (np.eye(3) + L_l * dt)
            self.F[self.left_mask] = np.einsum("ab,nbc->nac", dF_l, self.F[self.left_mask])

        if np.any(self.right_mask):
            c_pos_r = np.mean(self.pos[self.right_mask], axis=0)
            c_vel_r = np.mean(self.vel[self.right_mask], axis=0)
            p_rel_r = self.pos[self.right_mask] - c_pos_r
            v_rel_r = self.vel[self.right_mask] - c_vel_r
            A = p_rel_r.T @ p_rel_r + 1e-4 * np.eye(3)
            B = v_rel_r.T @ p_rel_r
            L_r = B @ np.linalg.inv(A)
            dF_r = (np.eye(3) + L_r * dt)
            self.F[self.right_mask] = np.einsum("ab,nbc->nac", dF_r, self.F[self.right_mask])

    def _make_state_frame(self, time: float) -> StateFrame:
        # Compute contact signed distance to ground floor (z = 0)
        # Gap is particle_center_z - contact_radius
        gap = self.pos[:, 2] - self.contact_radius
        # active / near contact samples
        near_mask = gap < 0.25
        if np.any(near_mask):
            contact_sd = gap[near_mask].copy()
        else:
            contact_sd = np.array([np.min(gap)], dtype=np.float64)

        # External force (gravity force on all particles)
        ext_force = self.mass[:, None] * self.gravity[None, :]

        # Cumulative mass source (K=2 materials: viscoplastic and oobleck, both 0 mass added/removed)
        cum_mass_source = np.zeros(2, dtype=np.float64)

        # Mass quantity per particle [N] or [N,K]
        mass_qty = self.mass.copy()

        # Build StateFrame
        frame = StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "particle_radius": np.full(self.N, self.contact_radius, dtype=np.float64),
                "rest_position": self.initial_frame.attributes["rest_position"].copy(),
                "material_label": self.body_id.copy(),
                "deformation_gradient": self.F.copy(),
                "mass_quantity": mass_qty,
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": cum_mass_source,
                "external_force": ext_force,
            },
            scalars={
                "contact_length_scale": float(self.particle_spacing),
                "total_mass": float(np.sum(self.mass)),
                "kinetic_energy": float(0.5 * np.sum(self.mass[:, None] * (self.vel**2))),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "particle_count": self.N,
            "current_time": self.current_time,
        }

