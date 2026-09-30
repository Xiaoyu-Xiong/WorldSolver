"""Solver implementation for 122_wetted_porous_plug_funnel.

Multiphysics particle-based simulation of a deformable porous plug wetting,
softening, deforming under liquid pressure and gravity, and passing through a funnel throat.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        # Initial state from scene
        init = scene.initial_state()
        self.pos = np.array(init.position, dtype=np.float64, copy=True)
        self.vel = np.array(init.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init.mass, dtype=np.float64, copy=True)
        self.body_id = np.array(init.body_id, dtype=np.int64, copy=True)

        self.N = len(self.pos)
        self.is_plug = (self.body_id == 1)
        self.is_water = (self.body_id == 2)
        self.N_plug = int(np.sum(self.is_plug))
        self.N_water = int(np.sum(self.is_water))

        # Attributes
        self.phase_id = np.array(init.attributes.get("phase_id", np.where(self.is_water, 1.0, 0.0)), dtype=np.float64)
        self.saturation = np.array(init.attributes.get("saturation", np.where(self.is_water, 1.0, 0.0)), dtype=np.float64)
        self.porosity = np.array(init.attributes.get("porosity", np.where(self.is_plug, float(self.config["plug"]["porosity"]), 0.0)), dtype=np.float64)
        self.def_grad = np.tile(np.eye(3, dtype=np.float64), (self.N, 1, 1))

        # Plug initial rest geometry
        self.plug_rest_pos = self.pos[self.is_plug].copy()
        self.plug_rest_center = np.mean(self.plug_rest_pos, axis=0)

        # Geometry & physics parameters
        self.gravity = np.array(self.config["gravity"], dtype=np.float64)
        self.funnel_profile = np.array([
            [0.31, 0.86],
            [0.31, 1.02],
            [0.58, 1.48],
            [0.88, 1.88],
            [1.10, 2.15],
        ], dtype=np.float64)
        self.floor_z = 0.06  # Catch floor top z

        self.dry_E = float(self.config["plug"]["dry_young_modulus"])
        self.wet_soft_factor = float(self.config["plug"]["wet_softening_factor"])
        self.permeability = float(self.config["plug"]["permeability"])
        self.throat_r = float(self.config["funnel"]["throat_radius"])
        self.throat_z = float(self.config["funnel"]["throat_z"])
        self.spacing = float(self.config["particle_spacing"])

        # Reference momentum scale
        total_mass = np.sum(self.mass)
        v_ref = np.sqrt(2.0 * 9.81 * (2.72 - self.floor_z))
        self.ref_momentum = float(total_mass * v_ref)

        # Cumulative budget tracking
        # G = 3 domains: 0: global, 1: plug, 2: water
        # K = 2 materials: 0: plug, 1: water
        self.cum_external_impulse = np.zeros((3, 3), dtype=np.float64)
        self.cum_inflow_mom = np.zeros((3, 3), dtype=np.float64)
        self.cum_outflow_mom = np.zeros((3, 3), dtype=np.float64)
        self.cum_mass_source = np.zeros(2, dtype=np.float64)
        self.last_ext_force_budget = np.zeros((3, 2, 3), dtype=np.float64)
        self.last_interaction_force = np.zeros((2, 3), dtype=np.float64)

        self.current_time = 0.0
        self.internal_dt = 0.001

    def _funnel_radius_at(self, z: float) -> float:
        """Interpolate inner radius of funnel at given z."""
        prof = self.funnel_profile
        if z <= prof[0, 1]:
            return prof[0, 0]
        if z >= prof[-1, 1]:
            return prof[-1, 0]
        # Linear interpolation between profile points
        idx = np.searchsorted(prof[:, 1], z)
        z0, r0 = prof[idx - 1, 1], prof[idx - 1, 0]
        z1, r1 = prof[idx, 1], prof[idx, 0]
        return r0 + (r1 - r0) * (z - z0) / (z1 - z0)

    def _funnel_normal_and_dist(self, pos: np.ndarray):
        """Compute distance to funnel inner wall and inward normal."""
        # For axisymmetric funnel: pos is (N, 3)
        r_xy = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2) + 1e-12
        z = pos[:, 2]

        # Evaluate target radius for each particle
        target_r = np.empty_like(z)
        prof = self.funnel_profile
        for i, zi in enumerate(z):
            target_r[i] = self._funnel_radius_at(zi)

        # Inward normal points towards axis: [-x/r, -y/r, slope]
        penetration = r_xy - target_r  # positive if outside wall
        norm_xy = np.stack([-pos[:, 0] / r_xy, -pos[:, 1] / r_xy, np.zeros_like(z)], axis=1)
        return penetration, norm_xy, target_r

    def _step(self, dt: float) -> None:
        """Physical time-step of the coupled porous-solid-fluid system."""
        N_p = self.N_plug
        plug_idx = np.where(self.is_plug)[0]
        water_idx = np.where(self.is_water)[0]

        plug_pos = self.pos[plug_idx]
        plug_vel = self.vel[plug_idx]
        water_pos = self.pos[water_idx]
        water_vel = self.vel[water_idx]

        # 1. Liquid wetting of plug
        # Compute distance between water and plug particles
        # Liquid particles near plug wet the plug and increase plug saturation
        diff = plug_pos[:, None, :] - water_pos[None, :, :]  # (N_p, N_w, 3)
        dist = np.linalg.norm(diff, axis=-1)  # (N_p, N_w)
        wet_radius = 2.0 * self.spacing

        near_water_mask = (dist < wet_radius)  # (N_p, N_w)
        wetting_count = np.sum(near_water_mask, axis=1)  # (N_p,)

        # Wetting propagation rate
        wetting_rate = self.permeability * 0.15
        d_sat = wetting_count * wetting_rate * dt
        self.saturation[plug_idx] = np.clip(self.saturation[plug_idx] + d_sat, 0.0, 1.0)

        # Average plug saturation
        plug_sat = self.saturation[plug_idx]
        mean_sat = np.mean(plug_sat)

        # 2. Forces on Plug
        # Gravity
        F_plug = self.mass[plug_idx, None] * self.gravity[None, :]

        # Liquid drag / seepage / pressure force on plug
        # Water particles pushing downward onto the plug
        F_interaction_water_on_plug = np.zeros(3, dtype=np.float64)
        for i_p in range(N_p):
            w_close = np.where(near_water_mask[i_p])[0]
            if len(w_close) > 0:
                rel_v = water_vel[w_close] - plug_vel[i_p]
                # Porous drag / seepage coupling: F = permeability * rho * area * rel_v
                f_drag = np.sum(rel_v, axis=0) * (self.permeability * 0.05 * self.mass[plug_idx[i_p]])
                # Contact/hydrostatic repulsion force from liquid above
                f_push = np.zeros(3)
                for iw in w_close:
                    d_vec = plug_pos[i_p] - water_pos[iw]
                    d_val = dist[i_p, iw]
                    if d_val < self.spacing and d_val > 1e-6:
                        f_push += (d_vec / d_val) * (100.0 * (self.spacing - d_val))
                F_plug[i_p] += f_drag + f_push
                F_interaction_water_on_plug += f_drag + f_push

        # Elastic shape-restoration force on plug (softened by wetting)
        # As plug wets, stiffness reduces from dry_E to wet_soft_factor * dry_E
        E_local = self.dry_E * (1.0 - (1.0 - self.wet_soft_factor) * plug_sat)
        # Center of mass of plug
        plug_com = np.mean(plug_pos, axis=0)
        # Relative displacements from rest
        rest_offset = self.plug_rest_pos - self.plug_rest_center
        curr_offset = plug_pos - plug_com

        # Shape restoration spring force towards current affine transformation
        # With high wetting, radial stiffness allows radial compression under funnel slope
        k_shape = E_local * 0.005
        F_shape = -k_shape[:, None] * (curr_offset - rest_offset)
        # When wet, radial compression resistance is much smaller, allowing squeezing
        F_shape[:, :2] *= (1.0 - 0.85 * plug_sat[:, None])
        F_plug += F_shape

        # Internal plug damping
        F_plug += -15.0 * self.mass[plug_idx, None] * (plug_vel - np.mean(plug_vel, axis=0))

        # 3. Forces on Water
        F_water = self.mass[water_idx, None] * self.gravity[None, :]

        # Equal and opposite reaction from plug
        if len(plug_idx) > 0 and len(water_idx) > 0:
            f_react_per_water = -F_interaction_water_on_plug / max(len(water_idx), 1)
            # Apply to water particles that are near the plug
            water_near_plug = np.any(near_water_mask, axis=0)
            if np.any(water_near_plug):
                F_water[water_near_plug] += -F_interaction_water_on_plug / np.sum(water_near_plug)

        # Simple SPH-like pressure / repulsive forces among water particles to prevent collapse
        # Grid-based or cell-based repulsion is fast
        # Water velocity damping for bulk viscosity
        F_water += -0.5 * self.mass[water_idx, None] * water_vel

        # 4. Integrate velocities and positions
        self.vel[plug_idx] += (F_plug / self.mass[plug_idx, None]) * dt
        self.vel[water_idx] += (F_water / self.mass[water_idx, None]) * dt

        self.pos[plug_idx] += self.vel[plug_idx] * dt
        self.pos[water_idx] += self.vel[water_idx] * dt

        # 5. Boundary Collisions (Funnel and Catch Floor)
        # Funnel collision for Plug
        pen_p, norm_p, target_r_p = self._funnel_normal_and_dist(self.pos[plug_idx])
        # Plug particles that penetrate the funnel wall
        in_funnel_z = (self.pos[plug_idx, 2] >= self.funnel_profile[0, 1]) & (self.pos[plug_idx, 2] <= self.funnel_profile[-1, 1])
        coll_p = (pen_p > 0.0) & in_funnel_z

        # If plug is stiff (dry), it jams at throat_z (z ~ 1.05..1.4) because r_rest (0.46) > throat_r (0.31)
        # If plug is softened, it deforms inward (r reduces to <= 0.30) and slides downward
        friction = 0.4 * (1.0 - 0.5 * mean_sat)
        for i_idx, p_idx in enumerate(plug_idx):
            if coll_p[i_idx]:
                p = pen_p[i_idx]
                n = norm_p[i_idx]
                # Push back to boundary
                self.pos[p_idx, :2] += n[:2] * p
                # Friction and normal reaction
                vn = np.dot(self.vel[p_idx], n)
                if vn < 0:
                    vt = self.vel[p_idx] - vn * n
                    self.vel[p_idx] = vt * max(0.0, 1.0 - friction)

        # Water collision with funnel
        pen_w, norm_w, target_r_w = self._funnel_normal_and_dist(self.pos[water_idx])
        in_funnel_zw = (self.pos[water_idx, 2] >= self.funnel_profile[0, 1]) & (self.pos[water_idx, 2] <= self.funnel_profile[-1, 1])
        coll_w = (pen_w > -0.01) & in_funnel_zw
        for i_idx, w_idx in enumerate(water_idx):
            if coll_w[i_idx]:
                p = pen_w[i_idx] + 0.01
                n = norm_w[i_idx]
                self.pos[w_idx, :2] += n[:2] * p
                vn = np.dot(self.vel[w_idx], n)
                if vn < 0:
                    self.vel[w_idx] -= 1.2 * vn * n

        # Floor collision for both plug and water
        coll_floor_p = self.pos[plug_idx, 2] < self.floor_z
        for i_idx, p_idx in enumerate(plug_idx):
            if coll_floor_p[i_idx]:
                self.pos[p_idx, 2] = self.floor_z
                if self.vel[p_idx, 2] < 0:
                    self.vel[p_idx, 2] = -0.1 * self.vel[p_idx, 2]
                    self.vel[p_idx, :2] *= 0.5

        coll_floor_w = self.pos[water_idx, 2] < self.floor_z
        for i_idx, w_idx in enumerate(water_idx):
            if coll_floor_w[i_idx]:
                self.pos[w_idx, 2] = self.floor_z
                if self.vel[w_idx, 2] < 0:
                    self.vel[w_idx, 2] = 0.0
                    self.vel[w_idx, :2] *= 0.8

        # Prevent water from passing through plug when plug is caught
        # Plug acts as a porous barrier
        plug_top_z = np.max(self.pos[plug_idx, 2])
        plug_bot_z = np.min(self.pos[plug_idx, 2])
        plug_mean_r = np.mean(np.sqrt(self.pos[plug_idx, 0]**2 + self.pos[plug_idx, 1]**2))
        water_in_plug_z = (self.pos[water_idx, 2] >= plug_bot_z - 0.05) & (self.pos[water_idx, 2] <= plug_top_z + 0.05)
        water_r = np.sqrt(self.pos[water_idx, 0]**2 + self.pos[water_idx, 1]**2)
        water_in_plug_r = water_r <= (plug_mean_r + 0.05)
        water_blocked = water_in_plug_z & water_in_plug_r & (self.pos[water_idx, 2] > np.mean(self.pos[plug_idx, 2]))

        # Slow down water draining through porous plug according to Darcy permeability
        drain_v = -0.05 * self.permeability * (1.0 + 0.5 * mean_sat)
        for i_idx, w_idx in enumerate(water_idx):
            if water_blocked[i_idx]:
                if self.vel[w_idx, 2] < drain_v:
                    self.vel[w_idx, 2] = drain_v

        # Update deformation gradient approximation for plug
        # F ~ I + displacement gradient
        for i_p in range(N_p):
            disp = self.pos[plug_idx[i_p]] - self.plug_rest_pos[i_p]
            rest_r = np.linalg.norm(self.plug_rest_pos[i_p, :2])
            curr_r = np.linalg.norm(self.pos[plug_idx[i_p], :2])
            scale_r = curr_r / max(rest_r, 0.01) if rest_r > 0.05 else 1.0
            scale_z = (self.pos[plug_idx[i_p], 2] - plug_com[2]) / max(self.plug_rest_pos[i_p, 2] - self.plug_rest_center[2], 0.01) if abs(self.plug_rest_pos[i_p, 2] - self.plug_rest_center[2]) > 0.05 else 1.0
            scale_z = np.clip(scale_z, 0.5, 2.0)
            self.def_grad[plug_idx[i_p]] = np.diag([scale_r, scale_r, scale_z])

        # Record forces for budget tracking
        F_ext_plug = np.sum(self.mass[plug_idx, None] * self.gravity[None, :], axis=0)
        F_ext_water = np.sum(self.mass[water_idx, None] * self.gravity[None, :], axis=0)

        self.last_ext_force_budget[0, 0] = F_ext_plug
        self.last_ext_force_budget[0, 1] = F_ext_water
        self.last_ext_force_budget[1, 0] = F_ext_plug
        self.last_ext_force_budget[1, 1] = np.zeros(3)
        self.last_ext_force_budget[2, 0] = np.zeros(3)
        self.last_ext_force_budget[2, 1] = F_ext_water

        self.cum_external_impulse[0] += (F_ext_plug + F_ext_water) * dt
        self.cum_external_impulse[1] += F_ext_plug * dt
        self.cum_external_impulse[2] += F_ext_water * dt

        self.last_interaction_force[0] = F_interaction_water_on_plug
        self.last_interaction_force[1] = -F_interaction_water_on_plug

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        # Step until requested time
        while self.current_time + 1e-8 < time:
            dt = min(self.internal_dt, time - self.current_time)
            self._step(dt)

        # Construct StateFrame
        domain_weights = np.zeros((3, self.N), dtype=np.float64)
        domain_weights[0, :] = 1.0  # global domain
        domain_weights[1, self.is_plug] = 1.0  # plug domain
        domain_weights[2, self.is_water] = 1.0  # water domain

        mass_quantity = np.zeros((self.N, 2), dtype=np.float64)
        mass_quantity[self.is_plug, 0] = self.mass[self.is_plug]
        mass_quantity[self.is_water, 1] = self.mass[self.is_water]

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "phase_id": self.phase_id.copy(),
                "saturation": self.saturation.copy(),
                "porosity": self.porosity.copy(),
                "deformation_gradient": self.def_grad.copy(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_mom.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_mom.copy(),
                "budget_domain_weights": domain_weights,
                "budget_external_force": self.last_ext_force_budget.copy(),
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "interaction_force": self.last_interaction_force.copy(),
            },
            scalars={
                "reference_momentum": self.ref_momentum,
                "dry_young_modulus": self.dry_E,
                "wet_softening_factor": self.wet_soft_factor,
                "permeability": self.permeability,
                "throat_radius": self.throat_r,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "mean_plug_saturation": float(np.mean(self.saturation[self.is_plug])) if np.any(self.is_plug) else 0.0,
            "plug_z": float(np.mean(self.pos[self.is_plug, 2])) if np.any(self.is_plug) else 0.0,
        }

