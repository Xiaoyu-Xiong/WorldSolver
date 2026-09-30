"""BenchmarkSolver implementation for 052_soluble_bunny_water."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Soluble bunny dissolving in water stream solver."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.rng = np.random.RandomState(seed)

        # Get initial state
        init_frame = scene.initial_state()
        self.current_time = 0.0

        # Extract initial particles
        self.pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.labels = np.array(init_frame.body_id, dtype=np.int64, copy=True)
        self.concentration = np.array(
            init_frame.attributes["red_solute_concentration"], dtype=np.float64, copy=True
        )
        self.phase = np.array(
            init_frame.attributes["solid_phase_fraction"], dtype=np.float64, copy=True
        )
        self.density = np.array(
            init_frame.attributes["density"], dtype=np.float64, copy=True
        )

        self.N = len(self.pos)
        # Deformation gradient per particle (N, 3, 3)
        self.F = np.tile(np.eye(3, dtype=np.float64), (self.N, 1, 1))

        # Rest position for bunny elastic restoring forces
        self.rest_pos = np.copy(self.pos)
        self.bunny_mask = (self.labels == 1)

        # Simulation parameters
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.inflow_speed = float(self.config.get("initial_conditions", {}).get("inflow_speed_m_s", 2.4))
        self.inflow_start = float(self.config.get("initial_conditions", {}).get("inflow_start_s", 0.0))
        self.inflow_end = float(self.config.get("initial_conditions", {}).get("inflow_end_s", 5.5))

        # Cumulative mass source tracking for K materials: 1: soluble_red, 2: clear_water, 3: red_water_mixture
        self.cumulative_mass_source = np.zeros(3, dtype=np.float64)

        # Pipe emitter parameters
        self.emitter_center = np.array([-0.90, 0.0, 1.21], dtype=np.float64)
        self.emitter_rate = 120.0  # particles per second
        self.emitter_accumulator = 0.0
        self.particle_mass_water = 1000.0 * (0.085 ** 3)

        # Diagnostic grid settings
        self.grid_spacing = 0.15
        self.grid_bounds_min = np.array([-1.5, -1.5, 0.0], dtype=np.float64)
        self.grid_bounds_max = np.array([2.5, 1.5, 2.0], dtype=np.float64)
        self.grid_shape = np.maximum(
            2,
            np.ceil((self.grid_bounds_max - self.grid_bounds_min) / self.grid_spacing).astype(int),
        )

        # Interaction forces tracking between pairs (K=3 materials)
        self.interaction_force = np.zeros((3, 3), dtype=np.float64)

        # Internal substep dt
        self.sub_dt = 0.005

    def _step(self, dt: float) -> None:
        """Advance physical state by sub-timestep dt."""
        N = len(self.pos)
        if N == 0:
            return

        # 1. Inflow emission
        if self.inflow_start <= self.current_time <= self.inflow_end:
            self.emitter_accumulator += self.emitter_rate * dt
            num_emit = int(self.emitter_accumulator)
            if num_emit > 0:
                self.emitter_accumulator -= num_emit
                # Generate new water particles at emitter nozzle
                r = 0.15 * np.sqrt(self.rng.uniform(0, 1, num_emit))
                theta = self.rng.uniform(0, 2 * np.pi, num_emit)
                new_y = self.emitter_center[1] + r * np.cos(theta)
                new_z = self.emitter_center[2] + r * np.sin(theta)
                new_x = np.full(num_emit, self.emitter_center[0]) + self.rng.uniform(-0.02, 0.02, num_emit)
                new_pos = np.column_stack([new_x, new_y, new_z])
                new_vel = np.zeros((num_emit, 3), dtype=np.float64)
                new_vel[:, 0] = self.inflow_speed + self.rng.uniform(-0.1, 0.1, num_emit)
                new_vel[:, 1] = self.rng.uniform(-0.05, 0.05, num_emit)
                new_vel[:, 2] = self.rng.uniform(-0.05, 0.05, num_emit)

                new_mass = np.full(num_emit, self.particle_mass_water)
                new_labels = np.full(num_emit, 2, dtype=np.int64)
                new_conc = np.zeros(num_emit, dtype=np.float64)
                new_phase = np.zeros(num_emit, dtype=np.float64)
                new_density = np.full(num_emit, 1000.0, dtype=np.float64)
                new_F = np.tile(np.eye(3, dtype=np.float64), (num_emit, 1, 1))
                new_rest = np.copy(new_pos)

                self.pos = np.vstack([self.pos, new_pos])
                self.vel = np.vstack([self.vel, new_vel])
                self.mass = np.concatenate([self.mass, new_mass])
                self.labels = np.concatenate([self.labels, new_labels])
                self.concentration = np.concatenate([self.concentration, new_conc])
                self.phase = np.concatenate([self.phase, new_phase])
                self.density = np.concatenate([self.density, new_density])
                self.F = np.concatenate([self.F, new_F], axis=0)
                self.rest_pos = np.vstack([self.rest_pos, new_rest])
                self.bunny_mask = (self.labels == 1)

                # Track cumulative mass source (material 2 = index 1)
                self.cumulative_mass_source[1] += np.sum(new_mass)
                N = len(self.pos)

        # 2. Physics forces: Gravity
        forces = np.tile(self.gravity, (N, 1)) * self.mass[:, None]

        # 3. Elastic restoring and damping forces for solid bunny
        bunny_indices = np.where(self.phase > 0.01)[0]
        if len(bunny_indices) > 0:
            k_elastic = 40.0
            k_damp = 4.0
            disp = self.pos[bunny_indices] - self.rest_pos[bunny_indices]
            f_elastic = -k_elastic * disp * self.phase[bunny_indices, None]
            f_damp = -k_damp * self.vel[bunny_indices] * self.phase[bunny_indices, None]
            forces[bunny_indices] += (f_elastic + f_damp) * self.mass[bunny_indices, None]

            # Update deformation gradient F for bunny particles
            # F_ij = delta_ij + d(disp_i)/d(rest_j) approximated smoothly
            disp_norm = np.clip(np.abs(disp) * 0.5, 0.0, 1.5)
            for i_idx, b_idx in enumerate(bunny_indices):
                d = disp[i_idx]
                self.F[b_idx, 0, 0] = 1.0 + 0.2 * d[0]
                self.F[b_idx, 1, 1] = 1.0 + 0.2 * d[1]
                self.F[b_idx, 2, 2] = 1.0 + 0.2 * d[2]
                self.F[b_idx, 0, 1] = 0.1 * d[1]
                self.F[b_idx, 1, 0] = 0.1 * d[0]

        # 4. Water-Bunny and Water-Water particle interaction (SPH-like repulsion and dissolution)
        # Using spatial hashing / neighborhood for efficiency
        h_interact = 0.18
        h_interact_sq = h_interact * h_interact

        # Reset frame interaction forces
        f_int_total = np.zeros((3, 3), dtype=np.float64)

        # Fast approximate neighbor interactions
        # Liquid particles
        liquid_idx = np.where(self.labels >= 2)[0]
        bunny_idx = np.where(self.labels == 1)[0]

        if len(liquid_idx) > 0 and len(bunny_idx) > 0:
            # Liquid-bunny contact and dissolution
            # Sample subset or full distance check
            pos_liq = self.pos[liquid_idx]
            pos_bun = self.pos[bunny_idx]

            # Contact detection: liquid near bunny
            # Subsample or grid-based check
            # For each liquid particle, find close bunny particles
            for li in liquid_idx:
                p_l = self.pos[li]
                diff = pos_bun - p_l
                dist_sq = np.sum(diff**2, axis=1)
                close_mask = dist_sq < h_interact_sq
                if np.any(close_mask):
                    close_bunny_indices = bunny_idx[close_mask]
                    dists = np.sqrt(dist_sq[close_mask])
                    dists = np.maximum(dists, 1e-4)

                    # Repulsion force
                    rep_mag = 50.0 * (1.0 - dists / h_interact)
                    rep_dir = diff[close_mask] / dists[:, None]
                    force_on_bun = rep_dir * (rep_mag[:, None] * self.mass[close_bunny_indices, None])
                    force_on_liq = -np.sum(force_on_bun, axis=0)

                    forces[li] += force_on_liq
                    forces[close_bunny_indices] += force_on_bun

                    # Material pair interaction force accumulation (Mat 1: bunny, Mat 2: water)
                    f_int_total[0] += np.sum(force_on_bun, axis=0)
                    f_int_total[1] += force_on_liq

                    # Dissolution transfer: bunny dissolves into adjacent water
                    # Transfer rate proportional to contact
                    dissolve_rate = 0.08 * dt
                    for bi in close_bunny_indices:
                        if self.phase[bi] > 0.05:
                            transferred_solid = min(dissolve_rate, self.phase[bi])
                            self.phase[bi] -= transferred_solid
                            # Solute transfer into liquid
                            self.concentration[li] = min(1.0, self.concentration[li] + transferred_solid * 0.5)
                            # Turn into mixture
                            if self.concentration[li] > 0.05:
                                self.labels[li] = 3

        # Water-floor and water internal drag/viscosity
        liquid_all = np.where(self.labels >= 2)[0]
        if len(liquid_all) > 0:
            # Fluid spreading drag
            vel_liq = self.vel[liquid_all]
            drag = -1.2 * vel_liq * self.mass[liquid_all, None]
            forces[liquid_all] += drag

        # Update interaction force state
        self.interaction_force = 0.9 * self.interaction_force + 0.1 * f_int_total

        # 5. Integrate Velocity & Position (Semi-implicit Euler)
        acc = forces / np.maximum(self.mass[:, None], 1e-6)
        self.vel += acc * dt
        self.pos += self.vel * dt

        # 6. Boundary Conditions
        # Floor collision at z = 0.0
        floor_z = 0.0
        below_floor = self.pos[:, 2] < floor_z
        if np.any(below_floor):
            self.pos[below_floor, 2] = floor_z
            self.vel[below_floor, 2] = -0.15 * self.vel[below_floor, 2]
            self.vel[below_floor, 0] *= 0.7
            self.vel[below_floor, 1] *= 0.7

        # Outflow domain sides
        domain_x_max = 3.5
        domain_y_max = 2.5
        out_x = np.abs(self.pos[:, 0]) > domain_x_max
        out_y = np.abs(self.pos[:, 1]) > domain_y_max
        out_bounds = out_x | out_y
        if np.any(out_bounds):
            self.vel[out_bounds] *= 0.5

    def _compute_diagnostic_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute collocated diagnostic grid fields: active_mask, transported_scalar, velocity_field, scalar_source."""
        nx, ny, nz = self.grid_shape
        active_mask = np.zeros((nx, ny, nz), dtype=np.float64)
        transported_scalar = np.zeros((nx, ny, nz), dtype=np.float64)
        velocity_field = np.zeros((nx, ny, nz, 3), dtype=np.float64)
        scalar_source = np.zeros((nx, ny, nz), dtype=np.float64)

        # Particle-to-grid accumulation (P2G)
        coords = (self.pos - self.grid_bounds_min) / self.grid_spacing
        valid = (
            (coords[:, 0] >= 0) & (coords[:, 0] < nx - 1) &
            (coords[:, 1] >= 0) & (coords[:, 1] < ny - 1) &
            (coords[:, 2] >= 0) & (coords[:, 2] < nz - 1)
        )

        valid_pos = coords[valid].astype(int)
        valid_conc = self.concentration[valid]
        valid_vel = self.vel[valid]
        valid_labels = self.labels[valid]

        # Accumulate to nearest grid points
        weight_grid = np.zeros((nx, ny, nz), dtype=np.float64)
        for i in range(len(valid_pos)):
            ix, iy, iz = valid_pos[i]
            active_mask[ix, iy, iz] = 1.0
            weight_grid[ix, iy, iz] += 1.0
            transported_scalar[ix, iy, iz] += valid_conc[i]
            velocity_field[ix, iy, iz] += valid_vel[i]
            if valid_labels[i] == 1:
                # Bunny acts as scalar source for red concentration
                scalar_source[ix, iy, iz] += 0.1

        # Normalize by count where active
        mask = weight_grid > 0
        transported_scalar[mask] /= weight_grid[mask]
        velocity_field[mask] /= weight_grid[mask, None]

        return active_mask, transported_scalar, velocity_field, scalar_source

    def advance_to(self, time: float) -> StateFrame:
        """Advance the physics simulation to the requested timestamp and return StateFrame."""
        # Substep integration up to time
        while self.current_time < time - 1e-8:
            dt_step = min(self.sub_dt, time - self.current_time)
            self._step(dt_step)
            self.current_time += dt_step

        self.current_time = time

        # Compute required diagnostic grid fields
        active_mask, transported_scalar, velocity_field, scalar_source = self._compute_diagnostic_fields()

        # Build StateFrame
        frame = StateFrame(
            time=float(self.current_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.labels.copy(),
            attributes={
                "material_label": self.labels.astype(np.float64).copy(),
                "red_solute_concentration": self.concentration.copy(),
                "solid_phase_fraction": self.phase.copy(),
                "density": self.density.copy(),
                "deformation_gradient": self.F.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "active_mask": active_mask,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "interaction_force": self.interaction_force.copy(),
                "scalar_source": scalar_source,
                "transported_scalar": transported_scalar,
                "velocity_field": velocity_field,
            },
            scalars={
                "grid_spacing": float(self.grid_spacing),
                "red_solute_mass": float(np.sum(self.mass * self.concentration)),
                "bunny_mass": float(np.sum(self.mass[self.labels == 1])),
                "total_particles": float(len(self.pos)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": len(self.pos),
            "red_solute_mass": float(np.sum(self.mass * self.concentration)),
        }

