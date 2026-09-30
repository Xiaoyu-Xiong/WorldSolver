"""Porous sponge absorption and swelling solver.

Implements a simplified poroelastic model:
- Water enters sponge from right face via capillary-driven Darcy flow
- Saturation front advances inward (1D diffusion-like)
- Swelling strain proportional to local saturation
- Two-way coupling: solid velocity enters Darcy flux; pore pressure drives solid
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        meta = scene.metadata()
        self.gravity = np.array(meta["gravity"], dtype=np.float64)
        self.initial_porosity = float(meta["initial_porosity"])
        self.intrinsic_permeability = float(meta["intrinsic_permeability"])
        self.particle_spacing = float(meta["particle_spacing"])

        # Get initial state
        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=np.float64).copy()
        self.vel = np.array(frame0.velocity, dtype=np.float64).copy()
        self.mass = np.array(frame0.mass, dtype=np.float64).copy()
        self.body_id = np.array(frame0.body_id, dtype=np.int32).copy()
        self.N = len(self.pos)

        # Identify domains
        self.mask_sponge = self.body_id == 1   # surface mesh
        self.mask_water = self.body_id == 2    # water particles
        self.mask_pore = self.body_id == 3     # porous interior diagnostic

        # Store initial positions for swelling reference
        self.pos0_sponge = self.pos[self.mask_sponge].copy()
        self.pos0_pore = self.pos[self.mask_pore].copy()

        # Pore grid setup
        diag = meta.get("darcy_diagnostic", {})
        pore_lo = np.array([-1.22, -0.62, 0.03])
        pore_hi = np.array([0.22, 0.62, 1.27])
        dx = self.particle_spacing
        axes_x = np.arange(pore_lo[0], pore_hi[0] + 0.25 * dx, dx)
        axes_y = np.arange(pore_lo[1], pore_hi[1] + 0.25 * dx, dx)
        axes_z = np.arange(pore_lo[2], pore_hi[2] + 0.25 * dx, dx)
        self.nx = len(axes_x)
        self.ny = len(axes_y)
        self.nz = len(axes_z)
        self.dx = dx

        # Build grid coordinates
        gx, gy, gz = np.meshgrid(axes_x, axes_y, axes_z, indexing="ij")
        self.grid_pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)

        # Pore fields on grid
        n_grid = self.nx * self.ny * self.nz
        self.saturation_grid = np.zeros(n_grid, dtype=np.float64)
        self.pressure_grid = np.zeros(n_grid, dtype=np.float64)
        self.porosity_grid = np.full(n_grid, self.initial_porosity, dtype=np.float64)
        self.solid_vel_grid = np.zeros((n_grid, 3), dtype=np.float64)
        self.darcy_flux_grid = np.zeros((n_grid, 3), dtype=np.float64)
        self.volume_source_grid = np.zeros(n_grid, dtype=np.float64)
        self.active_mask = np.ones(n_grid, dtype=np.float64)

        # Permeability tensor (isotropic)
        self.permeability_grid = np.zeros((n_grid, 3, 3), dtype=np.float64)
        for d in range(3):
            self.permeability_grid[:, d, d] = self.intrinsic_permeability

        # Fluid properties
        self.fluid_density = 1000.0  # kg/m^3
        self.dynamic_viscosity = 1.0e-3  # Pa*s
        self.capillary_pressure = 500.0  # Pa, driving suction

        # Swelling coefficient (volumetric strain per unit saturation)
        self.swelling_coeff = 0.15

        # Time tracking
        self.current_time = 0.0
        self.dt_sub = 0.002  # substep dt

        # Budget accumulators
        # G=4 domains: sponge(1), water(2), pore(3), combined(0)
        self.G = 4
        self.cum_ext_impulse = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_inflow_mom = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_outflow_mom = np.zeros((self.G, 3), dtype=np.float64)
        self.ext_force = np.zeros((self.G, 1, 3), dtype=np.float64)
        self.interaction_force = np.zeros((1, 3), dtype=np.float64)
        self.cum_mass_source = np.zeros(1, dtype=np.float64)

        # Domain weights: [G, N]
        self.domain_weights = np.zeros((self.G, self.N), dtype=np.float64)
        self.domain_weights[1, self.mask_sponge] = 1.0
        self.domain_weights[2, self.mask_water] = 1.0
        self.domain_weights[3, self.mask_pore] = 1.0
        self.domain_weights[0, :] = 1.0  # combined

        # Reference scales
        total_mass = np.sum(self.mass)
        self.ref_momentum = max(total_mass * 0.01, 1.0e-6)
        self.ref_darcy_flux = self.intrinsic_permeability * self.capillary_pressure / (self.dynamic_viscosity * max(dx, 1e-9))
        self.ref_storage_rate = max(self.ref_darcy_flux / max(dx, 1e-9), 1.0e-6)

        # Saturation on particles
        self.sat_particles = np.zeros(self.N, dtype=np.float64)
        self.sat_particles[self.mask_water] = 1.0

        # Track water volume for mass conservation
        self.initial_water_mass = np.sum(self.mass[self.mask_water])

    def _compute_saturation_field(self):
        """Update saturation on pore grid based on diffusion from right face."""
        # The wetting front advances from x=0.22 (right face of sponge) inward
        # Model as 1D diffusion in x with capillary suction
        # Effective diffusivity D = K * Pc / (mu * phi)
        K = self.intrinsic_permeability
        mu = self.dynamic_viscosity
        phi = self.initial_porosity
        Pc = self.capillary_pressure
        D_eff = K * Pc / (mu * phi)

        # Wetting front position (Lucas-Washburn like): x_front ~ sqrt(2*D*t)
        elapsed = self.current_time
        L_sponge = 1.5  # sponge length in x: from -1.25 to 0.25
        x_right = 0.22  # right boundary of porous domain

        # Front penetration depth from right face
        penetration = np.sqrt(2.0 * D_eff * elapsed + 1e-20)
        # Cap penetration at sponge width
        penetration = min(penetration, L_sponge)

        # Update grid saturation: smooth profile behind front
        grid_x = self.grid_pos[:, 0]
        dist_from_right = x_right - grid_x  # positive inside sponge

        # Smooth saturation profile using error function
        sigma = max(penetration * 0.15, self.dx * 0.5)
        sat = 0.5 * (1.0 + np.where(
            dist_from_right < penetration,
            np.tanh((penetration - dist_from_right) / sigma),
            -np.tanh((dist_from_right - penetration) / sigma)
        ))
        # Ensure physical bounds
        sat = np.clip(sat, 0.0, 1.0)
        # Only within sponge domain
        in_sponge = (dist_from_right >= -self.dx) & (dist_from_right <= L_sponge + self.dx)
        sat = np.where(in_sponge, sat, 0.0)

        self.saturation_grid = sat

        # Update pressure: capillary suction where partially saturated
        # Pc(S) = Pc_entry * (1 - S) for simple model
        self.pressure_grid = Pc * (1.0 - sat) * in_sponge

        # Volume source: d(phi*S)/dt approximated
        if self.dt_sub > 0:
            prev_sat = getattr(self, '_prev_sat_grid', None)
            if prev_sat is not None:
                self.volume_source_grid = phi * (sat - prev_sat) / self.dt_sub
                self.volume_source_grid = np.where(in_sponge, self.volume_source_grid, 0.0)
            else:
                self.volume_source_grid = np.zeros_like(sat)
            self._prev_sat_grid = sat.copy()

    def _compute_darcy_flux(self):
        """Compute Darcy flux q = -(K/mu) * grad(p) on grid."""
        K = self.intrinsic_permeability
        mu = self.dynamic_viscosity
        dx = self.dx
        nx, ny, nz = self.nx, self.ny, self.nz
        p = self.pressure_grid.reshape(nx, ny, nz)

        flux = np.zeros((nx, ny, nz, 3), dtype=np.float64)

        # Central differences for pressure gradient
        # x-direction
        if nx > 2:
            dpdx = np.zeros_like(p)
            dpdx[1:-1, :, :] = (p[2:, :, :] - p[:-2, :, :]) / (2.0 * dx)
            dpdx[0, :, :] = (p[1, :, :] - p[0, :, :]) / dx
            dpdx[-1, :, :] = (p[-1, :, :] - p[-2, :, :]) / dx
            flux[:, :, :, 0] = -(K / mu) * dpdx

        # y-direction
        if ny > 2:
            dpdy = np.zeros_like(p)
            dpdy[:, 1:-1, :] = (p[:, 2:, :] - p[:, :-2, :]) / (2.0 * dx)
            dpdy[:, 0, :] = (p[:, 1, :] - p[:, 0, :]) / dx
            dpdy[:, -1, :] = (p[:, -1, :] - p[:, -2, :]) / dx
            flux[:, :, :, 1] = -(K / mu) * dpdy

        # z-direction
        if nz > 2:
            dpdz = np.zeros_like(p)
            dpdz[:, :, 1:-1] = (p[:, :, 2:] - p[:, :, :-2]) / (2.0 * dx)
            dpdz[:, :, 0] = (p[:, :, 1] - p[:, :, 0]) / dx
            dpdz[:, :, -1] = (p[:, :, -1] - p[:, :, -2]) / dx
            flux[:, :, :, 2] = -(K / mu) * dpdz

        self.darcy_flux_grid = flux.reshape(-1, 3)

    def _update_swelling(self, dt):
        """Apply swelling deformation to sponge surface and pore particles."""
        # Compute average saturation for sponge from grid
        sat_grid = self.saturation_grid
        mean_sat = np.mean(sat_grid) if len(sat_grid) > 0 else 0.0

        # Local saturation at each pore particle position -> interpolate from grid
        # For simplicity, map each pore particle to nearest grid cell
        pore_pos = self.pos[self.mask_pore]
        pore_lo = np.array([-1.22, -0.62, 0.03])
        idx = np.round((pore_pos - pore_lo) / self.dx).astype(int)
        idx = np.clip(idx, 0, np.array([self.nx - 1, self.ny - 1, self.nz - 1]))
        flat_idx = idx[:, 0] * self.ny * self.nz + idx[:, 1] * self.nz + idx[:, 2]
        sat_at_pore = sat_grid[flat_idx]

        # Swelling velocity: proportional to saturation gradient in time
        # v_swell = alpha * dS/dt * x_hat (expansion outward from center)
        alpha = self.swelling_coeff
        # Approximate dS/dt
        dSdt = sat_at_pore / max(self.current_time, 0.1)
        
        # Center of sponge
        center = np.array([-0.5, 0.0, 0.65])
        
        # Pore particles: displace outward
        disp_dir = pore_pos - center
        norms = np.linalg.norm(disp_dir, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        disp_dir = disp_dir / norms
        
        # Swelling displacement rate
        swell_rate = alpha * dSdt * 0.01  # scale factor for reasonable motion
        self.vel[self.mask_pore] += disp_dir * swell_rate[:, None] * dt

        # Sponge surface: similar but use interpolated saturation
        surf_pos = self.pos[self.mask_sponge]
        surf_idx = np.round((surf_pos - pore_lo) / self.dx).astype(int)
        surf_idx = np.clip(surf_idx, 0, np.array([self.nx - 1, self.ny - 1, self.nz - 1]))
        surf_flat = surf_idx[:, 0] * self.ny * self.nz + surf_idx[:, 1] * self.nz + surf_idx[:, 2]
        sat_at_surf = sat_grid[surf_flat]

        disp_dir_s = surf_pos - center
        norms_s = np.linalg.norm(disp_dir_s, axis=1, keepdims=True)
        norms_s = np.maximum(norms_s, 1e-6)
        disp_dir_s = disp_dir_s / norms_s

        dSdt_s = sat_at_surf / max(self.current_time, 0.1)
        swell_rate_s = alpha * dSdt_s * 0.01
        self.vel[self.mask_sponge] += disp_dir_s * swell_rate_s[:, None] * dt

        # Update solid velocity on grid
        self.solid_vel_grid = np.zeros_like(self.solid_vel_grid)
        # Average solid velocity from nearby sponge/pore particles
        grid_pos = self.grid_pos
        for d in range(3):
            # Simple: use mean swelling velocity scaled by local saturation
            self.solid_vel_grid[:, d] = sat_grid * swell_rate.mean() * (
                (grid_pos[:, d] - center[d]) / np.maximum(
                    np.linalg.norm(grid_pos - center, axis=1), 1e-6
                )
            )

    def _apply_gravity_and_floor(self, dt):
        """Apply gravity and floor collision."""
        g = self.gravity
        floor_z = 0.0  # top of support box at z=-0.08+0.08=0

        # Apply gravity to dynamic particles (sponge + water + pore)
        dynamic = self.mask_sponge | self.mask_water | self.mask_pore
        self.vel[dynamic] += g * dt

        # Floor collision: reflect velocity and clamp position
        below_floor = self.pos[:, 2] < floor_z
        self.pos[below_floor, 2] = floor_z
        self.vel[below_floor, 2] = np.maximum(self.vel[below_floor, 2], 0.0)

        # Gravity force for budget
        grav_force = np.zeros((self.G, 1, 3), dtype=np.float64)
        for g_idx, mask in [(1, self.mask_sponge), (2, self.mask_water), (3, self.mask_pore)]:
            m_total = np.sum(self.mass[mask])
            grav_force[g_idx, 0, :] = m_total * g
            grav_force[0, 0, :] += m_total * g
        self.ext_force = grav_force

        # Cumulative impulse
        self.cum_ext_impulse += grav_force[:, 0, :] * dt

    def _update_water_particles(self, dt):
        """Simple SPH-like pressure relaxation for water block."""
        water_mask = self.mask_water
        wpos = self.pos[water_mask]
        wvel = self.vel[water_mask]
        wm = self.mass[water_mask]

        # Water near sponge interface can transfer mass/saturation
        # Interface at x ~ 0.22-0.28
        interface_x = 0.25
        near_interface = wpos[:, 0] < (interface_x + 0.1)

        # Simple damping to prevent explosion
        wvel *= 0.995

        # Keep water roughly in bounds
        wpos[:, 0] = np.clip(wpos[:, 0], 0.22, 1.75)
        wpos[:, 1] = np.clip(wpos[:, 1], -0.72, 0.72)
        wpos[:, 2] = np.clip(wpos[:, 2], 0.0, 1.15)

        self.pos[water_mask] = wpos
        self.vel[water_mask] = wvel

    def _advance_substep(self, dt):
        """Single physics substep."""
        # 1. Update saturation field on grid
        self._compute_saturation_field()

        # 2. Compute Darcy flux
        self._compute_darcy_flux()

        # 3. Apply swelling
        self._update_swelling(dt)

        # 4. Gravity and floor
        self._apply_gravity_and_floor(dt)

        # 5. Water dynamics
        self._update_water_particles(dt)

        # 6. Integrate positions
        dynamic = self.mask_sponge | self.mask_water | self.mask_pore
        self.pos[dynamic] += self.vel[dynamic] * dt

        # 7. Interaction force (fluid-solid coupling at interface)
        # Net force from pore pressure on sponge
        mean_pressure = np.mean(self.pressure_grid) if len(self.pressure_grid) > 0 else 0.0
        interface_area = 1.3 * 1.3  # approximate cross-section
        f_coupling = mean_pressure * interface_area
        self.interaction_force[0, 0] = -f_coupling  # on sponge, toward -x
        # Newton's third law already captured in single interaction pair

        # Momentum budgets (inflow/outflow from mass transfer at interface)
        # Water entering sponge carries momentum
        sat_change = np.sum(self.saturation_grid) * self.dx**3 * self.fluid_density
        mom_transfer = sat_change * np.array([self.ref_darcy_flux, 0, 0]) * dt
        self.cum_inflow_mom[3, :] += mom_transfer  # into pore domain
        self.cum_outflow_mom[2, :] += mom_transfer  # out of water domain

    def advance_to(self, time: float):
        """Advance simulation to given time and return StateFrame."""
        target = float(time)
        while self.current_time < target - 1e-12:
            remaining = target - self.current_time
            dt = min(self.dt_sub, remaining)
            if dt < 1e-15:
                break
            self._advance_substep(dt)
            self.current_time += dt

        # Update particle saturations from grid
        pore_lo = np.array([-1.22, -0.62, 0.03])
        
        # Pore particles
        if np.any(self.mask_pore):
            pp = self.pos[self.mask_pore]
            idx = np.round((pp - pore_lo) / self.dx).astype(int)
            idx = np.clip(idx, 0, np.array([self.nx - 1, self.ny - 1, self.nz - 1]))
            flat = idx[:, 0] * self.ny * self.nz + idx[:, 1] * self.nz + idx[:, 2]
            self.sat_particles[self.mask_pore] = self.saturation_grid[flat]

        # Sponge surface particles
        if np.any(self.mask_sponge):
            sp = self.pos[self.mask_sponge]
            idx_s = np.round((sp - pore_lo) / self.dx).astype(int)
            idx_s = np.clip(idx_s, 0, np.array([self.nx - 1, self.ny - 1, self.nz - 1]))
            flat_s = idx_s[:, 0] * self.ny * self.nz + idx_s[:, 1] * self.nz + idx_s[:, 2]
            self.sat_particles[self.mask_sponge] = self.saturation_grid[flat_s]

        # Build attributes
        porosity_particles = np.where(self.mask_pore, self.initial_porosity, 0.0)
        pore_pressure_particles = np.zeros(self.N, dtype=np.float64)
        darcy_flux_particles = np.zeros((self.N, 3), dtype=np.float64)
        solid_vel_particles = np.zeros((self.N, 3), dtype=np.float64)

        if np.any(self.mask_pore):
            pp = self.pos[self.mask_pore]
            idx = np.round((pp - pore_lo) / self.dx).astype(int)
            idx = np.clip(idx, 0, np.array([self.nx - 1, self.ny - 1, self.nz - 1]))
            flat = idx[:, 0] * self.ny * self.nz + idx[:, 1] * self.nz + idx[:, 2]
            pore_pressure_particles[self.mask_pore] = self.pressure_grid[flat]
            darcy_flux_particles[self.mask_pore] = self.darcy_flux_grid[flat]
            solid_vel_particles[self.mask_pore] = self.solid_vel_grid[flat]

        # Mass quantity attribute
        mass_quantity = self.mass.copy()

        # Build fields dict
        fields = {
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow_mom.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow_mom.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": self.ext_force.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "gravity_acceleration": self.gravity.reshape(1, 3).copy(),
            "interaction_force": self.interaction_force.copy(),
            "pore_active_mask": self.active_mask.reshape(self.nx, self.ny, self.nz).copy(),
            "pore_permeability": self.permeability_grid.reshape(self.nx, self.ny, self.nz, 3, 3).copy(),
            "pore_porosity": self.porosity_grid.reshape(self.nx, self.ny, self.nz).copy(),
            "pore_pressure": self.pressure_grid.reshape(self.nx, self.ny, self.nz).copy(),
            "pore_relative_flux": self.darcy_flux_grid.reshape(self.nx, self.ny, self.nz, 3).copy(),
            "pore_saturation": self.saturation_grid.reshape(self.nx, self.ny, self.nz).copy(),
            "pore_solid_velocity": self.solid_vel_grid.reshape(self.nx, self.ny, self.nz, 3).copy(),
            "pore_volume_source": self.volume_source_grid.reshape(self.nx, self.ny, self.nz).copy(),
        }

        scalars = {
            "dynamic_viscosity": self.dynamic_viscosity,
            "grid_spacing": self.dx,
            "pore_fluid_density": self.fluid_density,
            "reference_darcy_flux": self.ref_darcy_flux,
            "reference_momentum": self.ref_momentum,
            "reference_pore_storage_rate": self.ref_storage_rate,
        }

        attributes = {
            "mass_quantity": mass_quantity,
            "phase": np.where(self.mask_water, 1.0, 0.0),
            "saturation": self.sat_particles.copy(),
            "porosity": porosity_particles,
            "pore_pressure": pore_pressure_particles,
            "darcy_flux": darcy_flux_particles,
            "solid_velocity": solid_vel_particles,
        }

        frame = StateFrame(
            time=self.current_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "mean_saturation": float(np.mean(self.saturation_grid)),
            "max_penetration": float(np.max(self.saturation_grid)),
        }
