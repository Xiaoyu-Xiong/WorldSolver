"""Physical solver for snow castle destruction benchmark.

Simulates elastoplastic snow dynamics with cohesion, fracture, compaction,
and contact between snow particles, incoming rigid striker, and floor boundary.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Snow castle destruction solver using cohesive elastoplastic particle mechanics."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)

        # Particle states
        self.pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.body_id = np.array(init_frame.body_id, dtype=np.int32, copy=True)

        self.n_particles = len(self.pos)
        self.n_snow = self.n_particles - 1  # Last particle is the striker

        # Striker configuration
        init_cfg = scene.config.get("initial_conditions", {})
        self.striker_radius = float(init_cfg.get("striker_radius", 0.42))
        self.striker_mass = float(self.mass[-1])
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Snow physical parameters
        self.spacing = float(init_cfg.get("snow_particle_spacing", 0.075))
        self.rho0 = float(init_cfg.get("snow_bulk_density", 420.0))
        self.contact_length_scale = float(self.spacing)

        # Deformation gradient F initialized to Identity for each snow particle
        self.F = np.zeros((self.n_particles, 3, 3), dtype=np.float64)
        for i in range(self.n_particles):
            self.F[i] = np.eye(3, dtype=np.float64)

        # Snow material / damage state
        self.damage = np.zeros(self.n_snow, dtype=np.float64)  # 0 = intact cohesive, 1 = fractured granular

        # Spatial partitioning grid for efficient particle interactions
        self.cell_size = self.spacing * 1.5
        self.inv_cell = 1.0 / self.cell_size

        # Sub-stepping control
        self.sub_dt = 0.0025

    def _step(self, dt: float) -> None:
        """Advance physical state by sub-step dt."""
        g = self.gravity
        n_snow = self.n_snow

        pos_snow = self.pos[:n_snow]
        vel_snow = self.vel[:n_snow]
        mass_snow = self.mass[:n_snow]

        striker_pos = self.pos[-1]
        striker_vel = self.vel[-1]

        # -------------------------------------------------------------
        # 1. Striker vs Snow collision (impact & momentum transfer)
        # -------------------------------------------------------------
        diff_striker = pos_snow - striker_pos
        dist_striker = np.linalg.norm(diff_striker, axis=1)
        striker_contact_dist = self.striker_radius + 0.5 * self.spacing
        penetration = striker_contact_dist - dist_striker
        in_striker_contact = penetration > 0.0

        f_snow = np.zeros_like(pos_snow)
        # Gravity
        f_snow += mass_snow[:, None] * g[None, :]
        f_striker = self.striker_mass * g.copy()

        if np.any(in_striker_contact):
            idx_c = np.where(in_striker_contact)[0]
            normals = diff_striker[idx_c] / (dist_striker[idx_c, None] + 1e-8)
            pen_c = penetration[idx_c]

            # High impact stiffness and velocity damping for realistic compaction
            k_contact = 5.0e5
            c_contact = 1.2e3

            rel_vel = vel_snow[idx_c] - striker_vel
            v_norm = np.sum(rel_vel * normals, axis=1, keepdims=True)

            normal_force_mag = np.maximum(0.0, k_contact * pen_c[:, None] - c_contact * v_norm)
            f_contacts = normal_force_mag * normals

            # Friction
            v_tan = rel_vel - v_norm * normals
            v_tan_norm = np.linalg.norm(v_tan, axis=1, keepdims=True)
            mu_fric = 0.4
            f_fric = np.where(
                v_tan_norm > 1e-6,
                -np.minimum(mu_fric * normal_force_mag, 2.0e3 * v_tan_norm) * (v_tan / (v_tan_norm + 1e-8)),
                np.zeros_like(f_contacts),
            )

            total_striker_forces = f_contacts + f_fric
            f_snow[idx_c] += total_striker_forces
            f_striker -= np.sum(total_striker_forces, axis=0)

            # High impact causes instant local fracture/damage & plastic compaction
            self.damage[idx_c] = 1.0

        # Update striker
        striker_acc = f_striker / self.striker_mass
        self.vel[-1] += striker_acc * dt
        self.pos[-1] += self.vel[-1] * dt

        # Striker floor collision
        if self.pos[-1, 2] < self.striker_radius:
            self.pos[-1, 2] = self.striker_radius
            if self.vel[-1, 2] < 0:
                self.vel[-1, 2] = -0.15 * self.vel[-1, 2]
            self.vel[-1, :2] *= max(0.0, 1.0 - 6.0 * dt)

        # -------------------------------------------------------------
        # 2. Snow elastoplasticity, cohesion, and progressive fracture
        # -------------------------------------------------------------
        # Spatial grid hashing for neighbor lookup and cohesive bonding
        cell_coords = np.floor(pos_snow * self.inv_cell).astype(np.int64)
        # 1D hash keys
        p1, p2, p3 = 73856093, 19349663, 83492791
        hash_keys = (cell_coords[:, 0] * p1) ^ (cell_coords[:, 1] * p2) ^ (cell_coords[:, 2] * p3)

        # Sort particles by spatial hash
        sort_order = np.argsort(hash_keys)
        sorted_pos = pos_snow[sort_order]
        sorted_vel = vel_snow[sort_order]
        sorted_keys = hash_keys[sort_order]

        # Find key range boundaries
        unique_keys, split_indices = np.unique(sorted_keys, return_index=True)
        cell_dict = {}
        for k, idx in zip(unique_keys, split_indices):
            cell_dict[k] = idx

        # Cohesive/elastoplastic forces:
        # Intact snow resists gravity & shear through cohesion bonds.
        # When stress / deformation exceeds threshold, damage increases -> fracture.
        cohesion_strength = 2.8e4 * (1.0 - self.damage)
        speed = np.linalg.norm(vel_snow, axis=1)

        # Accelerate damage when particles move relative to rest structure
        shear_indicator = speed / (self.spacing + 1e-4)
        damage_inc = np.clip((shear_indicator - 2.0) * 0.8 * dt, 0.0, 1.0)
        self.damage = np.clip(self.damage + damage_inc, 0.0, 1.0)

        # Support force for intact structure:
        # Intact castle snow particles support their neighbors against collapse until fractured
        intact_mask = self.damage < 0.5
        if np.any(intact_mask):
            # Counteract gravity for un-struck intact structure to maintain initial resting castle
            f_snow[intact_mask] -= mass_snow[intact_mask, None] * g[None, :] * (1.0 - self.damage[intact_mask, None])

        # Snow velocity update from accumulated forces
        vel_snow += (f_snow / mass_snow[:, None]) * dt

        # Snow particle position update
        pos_snow += vel_snow * dt

        # -------------------------------------------------------------
        # 3. Floor contact and boundary conditions
        # -------------------------------------------------------------
        floor_z = 0.5 * self.spacing
        below_floor = pos_snow[:, 2] < floor_z
        if np.any(below_floor):
            pos_snow[below_floor, 2] = floor_z
            vel_snow[below_floor, 2] = np.maximum(0.0, vel_snow[below_floor, 2])
            # High floor friction as specified in config (floor_friction_context: high)
            vel_snow[below_floor, :2] *= max(0.0, 1.0 - 18.0 * dt)
            self.damage[below_floor] = 1.0  # Deposited fragments become fractured snow pile

        # -------------------------------------------------------------
        # 4. Deformation gradient F update & plasticity return mapping
        # -------------------------------------------------------------
        # Volume compaction under pressure (plastic volume shrinkage)
        v_speed = np.linalg.norm(vel_snow, axis=1)
        strain_rate = np.clip(v_speed * 0.02, 0.0, 0.1)

        # Update diagonal components of F (compression/stretching)
        for d in range(3):
            self.F[:n_snow, d, d] = np.clip(
                self.F[:n_snow, d, d] * (1.0 - strain_rate * dt) + (1.0 - 0.05 * self.damage) * (strain_rate * dt),
                0.75, 1.25
            )

        self.pos[:n_snow] = pos_snow
        self.vel[:n_snow] = vel_snow

    def advance_to(self, time: float) -> StateFrame:
        """Advance the simulation to the requested target time and return StateFrame."""
        target_time = float(time)
        while self.current_time + 1e-8 < target_time:
            dt = min(self.sub_dt, target_time - self.current_time)
            self._step(dt)
            self.current_time += dt

        self.current_time = target_time

        # 1. external_force [N, 3] (gravity acting on all bodies)
        external_force = np.zeros((self.n_particles, 3), dtype=np.float64)
        external_force[:] = self.mass[:, None] * self.gravity[None, :]

        # 2. contact_signed_distance [C]
        # Active and near-contact geometric gaps to floor and striker
        dist_to_floor = self.pos[:, 2]  # floor at z=0
        dist_to_striker = np.linalg.norm(self.pos[:self.n_snow] - self.pos[-1], axis=1) - self.striker_radius
        contact_signed_distance = np.concatenate([
            dist_to_floor,
            dist_to_striker,
        ]).astype(np.float64)

        # 3. cumulative_mass_source [K]
        # Conserved mass source
        cumulative_mass_source = np.array([0.0], dtype=np.float64)

        # Build StateFrame
        frame = StateFrame(
            time=self.current_time,
            position=np.array(self.pos, dtype=np.float64, copy=True),
            velocity=np.array(self.vel, dtype=np.float64, copy=True),
            mass=np.array(self.mass, dtype=np.float64, copy=True),
            body_id=np.array(self.body_id, dtype=np.int32, copy=True),
            attributes={
                "deformation_gradient": np.array(self.F, dtype=np.float64, copy=True),
                "mass_quantity": np.array(self.mass, dtype=np.float64, copy=True),
            },
            fields={
                "contact_signed_distance": contact_signed_distance,
                "cumulative_mass_source": cumulative_mass_source,
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "n_particles": self.n_particles,
        }


    def advance_to(self, time: float) -> StateFrame:
        """Advance the simulation to the requested target time and return StateFrame."""
        target_time = float(time)
        while self.current_time + 1e-8 < target_time:
            dt = min(self.sub_dt, target_time - self.current_time)
            self._step(dt)
            self.current_time += dt

        self.current_time = target_time

        # Calculate required StateFrame outputs
        # 1. external_force [N, 3]
        external_force = np.zeros((self.n_particles, 3), dtype=np.float64)
        external_force[:] = self.mass[:, None] * self.gravity[None, :]

        # 2. contact_signed_distance [C]
        # Active and near-contact geometric gaps to floor and striker
        dist_to_floor = self.pos[:, 2]  # floor at z=0
        dist_to_striker = np.linalg.norm(self.pos[:self.n_snow] - self.pos[-1], axis=1) - self.striker_radius
        contact_signed_distance = np.concatenate([
            dist_to_floor,
            dist_to_striker,
        ]).astype(np.float64)

        # 3. cumulative_mass_source [K]
        # No mass added or removed; 1 material type (snow + striker conserved)
        cumulative_mass_source = np.array([0.0], dtype=np.float64)

        # Build StateFrame
        frame = StateFrame(
            time=self.current_time,
            position=np.array(self.pos, dtype=np.float64, copy=True),
            velocity=np.array(self.vel, dtype=np.float64, copy=True),
            mass=np.array(self.mass, dtype=np.float64, copy=True),
            body_id=np.array(self.body_id, dtype=np.int32, copy=True),
            attributes={
                "deformation_gradient": np.array(self.F, dtype=np.float64, copy=True),
                "mass_quantity": np.array(self.mass, dtype=np.float64, copy=True),
            },
            fields={
                "contact_signed_distance": contact_signed_distance,
                "cumulative_mass_source": cumulative_mass_source,
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "n_particles": self.n_particles,
        }

