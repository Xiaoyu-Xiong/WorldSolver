"""Particle-based egg-frying solver.

A lightweight, physically-based solver for the 051_egg_frying scene.  It models
the egg white and yolk as viscous, repulsive particles that fall under gravity,
collide with a hot pan, spread radially, and undergo a temperature-driven
cooking front that converts liquid egg into solid egg from the pan upward.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver for the egg-frying scene."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = scene.config

        frame0 = scene.initial_state()
        self.position = frame0.position.copy()
        self.velocity = frame0.velocity.copy()
        self.mass = frame0.mass.copy()
        self.body_id = frame0.body_id.copy()

        attrs = frame0.attributes
        self.material_concentration = attrs["material_concentration"].copy()
        self.material_label = attrs["material_label"].copy()
        self.liquid_phase_fraction = attrs["liquid_phase_fraction"].copy()
        self.solid_phase_fraction = attrs["solid_phase_fraction"].copy()
        self.temperature = attrs["temperature_k"].copy()
        self.pan_contact_distance = attrs["pan_contact_distance"].copy()

        self.t = 0.0
        self.sample_dt = float(self.config["sample_dt"])

        ic = self.config["initial_conditions"]
        bc = self.config["boundary_conditions"]
        self.pan_T = float(ic["pan_temperature_k"])
        self.ambient_T = float(ic["ambient_temperature_k"])
        self.gravity_vec = np.array(ic["gravity"], dtype=float)
        self.pan_z = float(bc["pan_contact_height_m"])
        self.pan_radius = float(bc["pan_no_escape_radius_m"])

        # Thermal model parameters.
        self.cooking_T = 340.0
        self.cooked_T = 360.0
        self.cooking_rate = 0.35
        self.pan_heat_coeff = 2.0  # 1/s effective heating rate at the pan surface.
        self.ambient_coeff = 0.05  # 1/s cooling to ambient.
        self.thermal_diffusivity = 2.5e-6  # m^2/s, enhanced for visual front.
        self.thermal_cutoff = 0.015  # m neighbor cutoff for conduction.

        # Mechanical parameters.
        self.repulsion_stiffness = 900.0
        self.repulsion_range = 0.012
        self.viscosity = 0.12
        self.friction = 0.35
        self.restitution = 0.0

        # Diagnostic grid (covers the egg and pan region).
        self.grid_origin = np.array([-0.21, -0.21, 0.0], dtype=float)
        self.grid_dx = 0.013125
        self.grid_nx = 32
        self.grid_ny = 32
        self.grid_nz = 8
        self.contact_scale = 0.007

        self.rng = np.random.default_rng(seed)
        self.cumulative_mass_source = np.zeros(2, dtype=float)
        self.interaction_force = np.zeros((2, 3), dtype=float)

    def _resolve_pan_collision(self, dt: float) -> None:
        # Floor at the hot pan surface.
        pen = self.pan_z - self.position[:, 2]
        contact = pen > 0.0
        if np.any(contact):
            self.position[contact, 2] = self.pan_z + 1e-6
            v_n = self.velocity[contact, 2]
            self.velocity[contact, 2] = np.where(v_n < 0.0, 0.0, v_n)
            self.velocity[contact, :2] *= 1.0 - self.friction * dt

        # Pan wall confinement for the shallow egg pool.
        r_xy = np.linalg.norm(self.position[:, :2], axis=1)
        low = self.position[:, 2] < self.pan_z + 0.045
        beyond = (r_xy > self.pan_radius) & low
        if np.any(beyond):
            scale = self.pan_radius / r_xy[beyond]
            self.position[beyond, :2] *= scale[:, None]
            pos_r = self.position[beyond, :2]
            v = self.velocity[beyond, :2]
            r_norm = np.linalg.norm(pos_r, axis=1, keepdims=True)
            r_norm = np.where(r_norm == 0.0, 1.0, r_norm)
            normal = pos_r / r_norm
            v_dot = np.sum(v * normal, axis=1, keepdims=True)
            outward = v_dot[:, 0] > 0.0
            if np.any(outward):
                self.velocity[beyond, :2][outward] -= (
                    v_dot[outward] * normal[outward]
                ).reshape(-1, 2)

    def _update_temperature(self, dt: float) -> None:
        n = len(self.position)
        dT = np.zeros(n, dtype=float)

        # Strong heating from the hot pan for particles touching/near it.
        dist_to_pan = self.position[:, 2] - self.pan_z
        near_pan = dist_to_pan < 0.004
        dT[near_pan] += (
            self.pan_heat_coeff
            * (self.pan_T - self.temperature[near_pan])
            * dt
        )

        # Weak ambient cooling.
        dT += self.ambient_coeff * (self.ambient_T - self.temperature) * dt

        # Particle-particle thermal conduction.
        if n > 1:
            h = self.thermal_cutoff
            tree = cKDTree(self.position)
            pairs = tree.query_ball_point(self.position, r=h)
            for i in range(n):
                js = pairs[i]
                if not js:
                    continue
                js = np.array(js, dtype=int)
                js = js[js != i]
                if len(js) == 0:
                    continue
                avg = self.temperature[js].mean()
                dT[i] += self.thermal_diffusivity * (avg - self.temperature[i]) * dt / (h * h)

        self.temperature += dT
        self.temperature = np.clip(self.temperature, 200.0, 500.0)

    def _update_phase(self, dt: float) -> None:
        raw = 1.0 - self.solid_phase_fraction
        temp_factor = np.clip(
            (self.temperature - self.cooking_T) / (self.cooked_T - self.cooking_T),
            0.0,
            1.0,
        )
        d_solid = self.cooking_rate * temp_factor * raw * dt
        self.solid_phase_fraction = np.clip(
            self.solid_phase_fraction + d_solid, 0.0, 1.0
        )
        self.liquid_phase_fraction = 1.0 - self.solid_phase_fraction

    def _step(self, dt: float) -> None:
        n = len(self.position)
        forces = np.zeros_like(self.position)

        # Gravity.
        forces += self.mass[:, None] * self.gravity_vec[None, :]

        # Pairwise repulsion for volume conservation and spreading.
        if n > 1:
            h = self.repulsion_range
            tree = cKDTree(self.position)
            pairs = tree.query_ball_point(self.position, r=2.0 * h)
            for i in range(n):
                js = pairs[i]
                if not js:
                    continue
                js = np.array(js, dtype=int)
                js = js[js > i]
                if len(js) == 0:
                    continue
                dx = self.position[js] - self.position[i]
                dist = np.linalg.norm(dx, axis=1)
                mask = (dist > 0.0) & (dist < 2.0 * h)
                if not np.any(mask):
                    continue
                dx = dx[mask]
                dist = dist[mask]
                js = js[mask]
                mag = self.repulsion_stiffness * (1.0 - dist / (2.0 * h)) ** 2
                f = (mag / dist)[:, None] * (-dx)
                forces[i] += f.sum(axis=0)
                forces[js] -= f

        # Viscous damping.
        forces -= self.viscosity * self.velocity / dt

        # Semi-implicit Euler integration.
        self.velocity += (forces / self.mass[:, None]) * dt
        self.position += self.velocity * dt

        # Enforce pan contact and confinement.
        self._resolve_pan_collision(dt)

        # Thermal and phase updates.
        self._update_temperature(dt)
        self._update_phase(dt)

        # Diagnostic contact distance.
        self.pan_contact_distance = self.position[:, 2] - self.pan_z

        # Net interaction force per material body (white / yolk) from the pan.
        self.interaction_force = np.zeros((2, 3), dtype=float)
        contact_mask = self.pan_contact_distance < 0.001
        for k in range(2):
            mask = self.body_id == (k + 1)
            contacting = mask & contact_mask
            if np.any(contacting):
                # Vertical support force balancing weight of contacting particles.
                self.interaction_force[k, 2] = np.sum(self.mass[contacting]) * 9.81

    def _grid_scalar(self, particle_values: np.ndarray):
        grid = np.zeros((self.grid_nx, self.grid_ny, self.grid_nz), dtype=float)
        weight = np.zeros_like(grid)
        dx = self.grid_dx
        origin = self.grid_origin
        for i in range(len(self.position)):
            idx = ((self.position[i] - origin) / dx).astype(int)
            for ix in range(max(0, idx[0] - 1), min(self.grid_nx, idx[0] + 3)):
                for iy in range(max(0, idx[1] - 1), min(self.grid_ny, idx[1] + 3)):
                    for iz in range(max(0, idx[2] - 1), min(self.grid_nz, idx[2] + 3)):
                        cell_pos = origin + (np.array([ix, iy, iz]) + 0.5) * dx
                        r = np.linalg.norm(self.position[i] - cell_pos)
                        if r > 3.0 * dx:
                            continue
                        w = np.exp(-(r * r) / (2.0 * dx * dx))
                        grid[ix, iy, iz] += w * particle_values[i]
                        weight[ix, iy, iz] += w
        mask = weight > 0.0
        grid[mask] /= weight[mask]
        return grid, mask

    def _grid_vector(self, particle_values: np.ndarray):
        grid = np.zeros((self.grid_nx, self.grid_ny, self.grid_nz, 3), dtype=float)
        weight = np.zeros((self.grid_nx, self.grid_ny, self.grid_nz), dtype=float)
        dx = self.grid_dx
        origin = self.grid_origin
        for i in range(len(self.position)):
            idx = ((self.position[i] - origin) / dx).astype(int)
            for ix in range(max(0, idx[0] - 1), min(self.grid_nx, idx[0] + 3)):
                for iy in range(max(0, idx[1] - 1), min(self.grid_ny, idx[1] + 3)):
                    for iz in range(max(0, idx[2] - 1), min(self.grid_nz, idx[2] + 3)):
                        cell_pos = origin + (np.array([ix, iy, iz]) + 0.5) * dx
                        r = np.linalg.norm(self.position[i] - cell_pos)
                        if r > 3.0 * dx:
                            continue
                        w = np.exp(-(r * r) / (2.0 * dx * dx))
                        grid[ix, iy, iz] += w * particle_values[i]
                        weight[ix, iy, iz] += w
        mask = weight > 0.0
        grid[mask] /= weight[mask, None]
        return grid, weight

    def _make_frame(self, time: float) -> StateFrame:
        temp_grid, active_mask = self._grid_scalar(self.temperature)

        # Scalar source: pan heating term on the grid.
        source = np.zeros_like(temp_grid)
        for iz in range(self.grid_nz):
            z_cell = self.grid_origin[2] + (iz + 0.5) * self.grid_dx
            if z_cell < self.pan_z + 0.005:
                source[:, :, iz] = self.pan_heat_coeff * (self.pan_T - temp_grid[:, :, iz])

        vel_grid, _ = self._grid_vector(self.velocity)

        # Signed distance to the pan surface for each particle.
        contact_sd = self.position[:, 2] - self.pan_z

        fields = {
            "active_mask": active_mask.astype(float),
            "contact_signed_distance": contact_sd,
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "interaction_force": self.interaction_force.copy(),
            "scalar_source": source,
            "transported_scalar": temp_grid,
            "velocity_field": vel_grid,
        }

        attributes = {
            "material_concentration": self.material_concentration,
            "material_label": self.material_label,
            "liquid_phase_fraction": self.liquid_phase_fraction,
            "solid_phase_fraction": self.solid_phase_fraction,
            "temperature_k": self.temperature,
            "pan_contact_distance": self.pan_contact_distance,
            "mass_quantity": self.mass.copy(),
        }

        scalars = {
            "contact_length_scale": float(self.contact_scale),
            "grid_spacing": float(self.grid_dx),
        }

        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        if time < self.t:
            return self._make_frame(time)
        dt = time - self.t
        # Fixed small substep for stability.
        n_sub = max(1, int(np.ceil(dt / 0.001)))
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            self._step(sub_dt)
        self.t = time
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "egg_frying_particle_solver",
            "num_particles": len(self.position),
            "final_temperature_min": float(np.min(self.temperature)),
            "final_temperature_max": float(np.max(self.temperature)),
            "max_solid_fraction": float(np.max(self.solid_phase_fraction)),
            "mean_solid_fraction": float(np.mean(self.solid_phase_fraction)),
        }
