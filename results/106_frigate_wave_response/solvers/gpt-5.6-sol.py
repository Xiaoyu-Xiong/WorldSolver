"""Deterministic shallow-water/rigid-body solver for the frigate scene.

The water is advanced on a staggered finite-difference grid. The hull is an
impermeable plan-view obstacle, while heave and pitch are advanced from
strip-integrated water pressure. Hull motion feeds back into continuity as the
rate of displaced volume. Render markers follow the stepped velocity field.
"""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        self.cfg = scene.config
        initial = scene.initial_state()
        fluid = self.cfg["fluid"]
        wave = self.cfg["incoming_waves"]
        ship = self.cfg["frigate"]
        self.rho = float(fluid["density"])
        self.g = abs(float(self.cfg["gravity"][2]))
        self.h0 = float(fluid["still_water_level"])
        self.spacing = float(fluid["marker_spacing"])
        self.lo = np.asarray(fluid["domain_min"], dtype=np.float64)
        self.hi = np.asarray(fluid["domain_max"], dtype=np.float64)
        self.amplitude = float(wave["amplitude"])
        self.wavelength = float(wave["wavelength"])
        self.period = float(wave["period"])
        self.phase = float(wave["phase_radians"])
        self.k_wave = 2.0 * math.pi / self.wavelength
        self.omega = 2.0 * math.pi / self.period
        self.wave_c = self.wavelength / self.period
        self.wave_g = self.wave_c * self.wave_c / self.h0
        self.base_u = float(fluid["initial_velocity"][0])

        self.xc = np.arange(self.lo[0] + 0.5 * self.spacing,
                            self.hi[0], self.spacing)
        self.yc = np.arange(self.lo[1] + 0.5 * self.spacing,
                            self.hi[1], self.spacing)
        self.nx, self.ny = len(self.xc), len(self.yc)
        xx, yy = np.meshgrid(self.xc, self.yc, indexing="ij")
        self.xx = xx
        self.eta = self.h0 + self.amplitude * np.sin(
            self.k_wave * xx + self.phase)

        # Staggered velocities start as a right-going progressive wave.
        self.uf = np.full((self.nx + 1, self.ny), self.base_u,
                          dtype=np.float64)
        eta_u = np.empty_like(self.uf)
        eta_u[1:-1] = 0.5 * (self.eta[:-1] + self.eta[1:])
        eta_u[0] = self.h0 + self.amplitude * math.sin(
            self.k_wave * self.lo[0] + self.phase)
        eta_u[-1] = self.eta[-1]
        self.uf += (self.wave_c / self.h0) * (eta_u - self.h0)
        self.vf = np.zeros((self.nx, self.ny + 1), dtype=np.float64)

        # Mesh dimensions define an impermeable hull footprint. Closed faces
        # make the stepped water flow around its bow and sides.
        taper = np.clip(1.0 - np.abs(xx) / 4.35, 0.0, 1.0) ** 0.35
        half_beam = 0.82 * taper
        self.solid = (np.abs(xx) < 4.35) & (np.abs(yy) < half_beam)
        self.open_u = np.ones_like(self.uf, dtype=bool)
        self.open_u[1:-1] = ~(self.solid[:-1] | self.solid[1:])
        self.open_v = np.ones_like(self.vf, dtype=bool)
        self.open_v[:, 0] = False
        self.open_v[:, -1] = False
        self.open_v[:, 1:-1] = ~(self.solid[:, :-1] | self.solid[:, 1:])
        self.uf[~self.open_u] = 0.0
        self.vf[~self.open_v] = 0.0

        adjacent = np.zeros_like(self.solid)
        adjacent[:-1] |= self.solid[1:]
        adjacent[1:] |= self.solid[:-1]
        adjacent[:, :-1] |= self.solid[:, 1:]
        adjacent[:, 1:] |= self.solid[:, :-1]
        ring = adjacent & ~self.solid
        ring_weight = taper * ring
        waterplane_area = float(np.sum(2.0 * half_beam[:, 0]) * self.spacing)
        denom = float(np.sum(ring_weight) * self.spacing ** 2)
        self.displacement_pattern = ring_weight * waterplane_area / max(denom, 1e-12)

        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id).copy()
        self.ship_index = int(np.flatnonzero(
            self.body_id == int(ship["body_id"]))[0])
        self.fluid_indices = np.flatnonzero(
            self.body_id != int(ship["body_id"]))
        self.phase_attr = np.asarray(initial.attributes["phase"]).copy()
        self.visual_attr = np.asarray(initial.attributes["ship_visual"]).copy()
        fp = self.position[self.fluid_indices]
        initial_surface = self.h0 + self.amplitude * np.sin(
            self.k_wave * fp[:, 0] + self.phase)
        self.sigma = np.clip(fp[:, 2] / np.maximum(initial_surface, 0.2),
                             0.0, 1.0)

        self.ship_mass = float(ship["mass"])
        self.ship_z0 = float(ship["initial_center"][2])
        self.ship_z = self.ship_z0
        self.ship_vz = float(ship["initial_velocity"][2])
        self.pitch = 0.0
        self.pitch_rate = float(ship["initial_angular_velocity"][1])
        self.added_heave_mass = 0.75 * self.ship_mass
        self.pitch_inertia = self.ship_mass * 8.55 ** 2 / 12.0
        self.added_pitch_inertia = 0.55 * self.pitch_inertia
        self.heave_damping = 1.05e4
        self.pitch_damping = 8.0e4
        self.station_x = np.linspace(-4.2, 4.2, 43)
        self.station_dx = float(self.station_x[1] - self.station_x[0])
        self.station_beam = 1.64 * np.clip(
            1.0 - np.abs(self.station_x) / 4.35, 0.0, 1.0) ** 0.35

        self.time = 0.0
        self.last_hydro_force = np.array(
            [0.0, 0.0, self.ship_mass * self.g])
        self.boundary_normals = np.asarray(
            [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0],
             [0.0, 1.0, 0.0]], dtype=np.float64)
        self.boundary_velocity = np.zeros((3, 3), dtype=np.float64)
        self.cumulative_in = np.zeros(3, dtype=np.float64)
        self.cumulative_out = np.zeros(3, dtype=np.float64)
        self.cumulative_external = np.zeros((3, 3), dtype=np.float64)
        self.last_interval_in = np.zeros(3, dtype=np.float64)
        self.last_interval_out = np.zeros(3, dtype=np.float64)
        self.last_frame_time = 0.0
        self.last_ship_momentum = (self.mass[self.ship_index]
                                   * self.velocity[self.ship_index])
        self.last_fluid_momentum = np.sum(
            self.mass[self.fluid_indices, None]
            * self.velocity[self.fluid_indices], axis=0)
        self.last_external_terms = self._initial_external_terms()
        total_mass = float(np.sum(self.mass))
        self.reference_velocity = max(
            0.1, abs(self.base_u) + math.sqrt(self.g * self.amplitude))
        self.reference_momentum = total_mass * self.reference_velocity
        self.steps = 0

    def _initial_external_terms(self) -> np.ndarray:
        terms = np.zeros((3, 2, 3), dtype=np.float64)
        fluid_mass = float(np.sum(self.mass[self.fluid_indices]))
        ship_gravity = np.array([0.0, 0.0, -self.ship_mass * self.g])
        fluid_gravity = np.array([0.0, 0.0, -fluid_mass * self.g])
        terms[1, 0] = ship_gravity
        terms[2, 0] = fluid_gravity
        terms[2, 1] = -(ship_gravity + fluid_gravity)
        terms[0] = terms[1] + terms[2]
        return terms

    def _sample_eta(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        gx = (x - self.xc[0]) / self.spacing
        gy = (y - self.yc[0]) / self.spacing
        i0 = np.floor(gx).astype(np.int64)
        j0 = np.floor(gy).astype(np.int64)
        fx = gx - i0
        fy = gy - j0
        i0 = np.clip(i0, 0, self.nx - 2)
        j0 = np.clip(j0, 0, self.ny - 2)
        return ((1.0 - fx) * (1.0 - fy) * self.eta[i0, j0]
                + fx * (1.0 - fy) * self.eta[i0 + 1, j0]
                + (1.0 - fx) * fy * self.eta[i0, j0 + 1]
                + fx * fy * self.eta[i0 + 1, j0 + 1])

    def _cell_velocity(self, x: np.ndarray, y: np.ndarray):
        ii = np.clip(np.rint((x - self.xc[0]) / self.spacing).astype(np.int64),
                     0, self.nx - 1)
        jj = np.clip(np.rint((y - self.yc[0]) / self.spacing).astype(np.int64),
                     0, self.ny - 1)
        uc = 0.5 * (self.uf[:-1] + self.uf[1:])
        vc = 0.5 * (self.vf[:, :-1] + self.vf[:, 1:])
        return uc[ii, jj], vc[ii, jj]

    def _pressure_load(self):
        beam_half = 0.5 * self.station_beam
        eta_port = self._sample_eta(
            self.station_x, beam_half + self.spacing)
        eta_starboard = self._sample_eta(
            self.station_x, -beam_half - self.spacing)
        local_surface = 0.5 * (eta_port + eta_starboard)
        heave = self.ship_z - self.ship_z0
        immersion = (local_surface - self.h0 - heave
                     + self.pitch * self.station_x)
        strip_force = (self.rho * self.g * self.station_beam
                       * self.station_dx * immersion)
        force = float(np.sum(strip_force)
                      - self.heave_damping * self.ship_vz)
        torque = float(-np.sum(self.station_x * strip_force)
                       - self.pitch_damping * self.pitch_rate)
        return force, torque

    def _advance_fluid(self, dt: float) -> None:
        self.uf[1:-1] -= (dt * self.wave_g
                           * (self.eta[1:] - self.eta[:-1]) / self.spacing)
        self.vf[:, 1:-1] -= (dt * self.wave_g
                              * (self.eta[:, 1:] - self.eta[:, :-1])
                              / self.spacing)
        damping = math.exp(-0.018 * dt)
        self.uf[1:-1] = (self.base_u
                         + (self.uf[1:-1] - self.base_u) * damping)
        self.vf[:, 1:-1] *= damping
        self.uf[~self.open_u] = 0.0
        self.vf[~self.open_v] = 0.0
        eta_left = self.h0 + self.amplitude * math.sin(
            self.k_wave * self.lo[0] - self.omega * (self.time + dt)
            + self.phase)
        self.uf[0, :] = (self.base_u + self.wave_c / self.h0
                         * (eta_left - self.h0))
        self.uf[-1, :] = self.uf[-2, :]
        divergence = ((self.uf[1:] - self.uf[:-1]) / self.spacing
                      + (self.vf[:, 1:] - self.vf[:, :-1]) / self.spacing)
        source = self.displacement_pattern * (
            -self.ship_vz + self.pitch_rate * self.xx)
        self.eta += dt * (-self.h0 * divergence + 0.72 * source)
        self.eta[0, :] = eta_left
        self.eta[-1, :] = self.eta[-2, :]
        self.eta[self.solid] = self.h0
        np.clip(self.eta, 0.35, 2.85, out=self.eta)

    def _advance_ship(self, dt: float) -> None:
        force, torque = self._pressure_load()
        az = force / (self.ship_mass + self.added_heave_mass)
        pitch_accel = torque / (
            self.pitch_inertia + self.added_pitch_inertia)
        self.ship_vz += dt * az
        self.ship_z += dt * self.ship_vz
        self.pitch_rate += dt * pitch_accel
        self.pitch += dt * self.pitch_rate
        self.last_hydro_force[:] = (0.0, 0.0,
                                    self.ship_mass * (self.g + az))

    def _advance_markers(self, dt: float) -> None:
        inds = self.fluid_indices
        old = self.position[inds].copy()
        x, y = old[:, 0], old[:, 1]
        vx, vy = self._cell_velocity(x, y)
        new_x = x + dt * vx
        new_y = y + dt * vy
        width = self.hi[0] - self.lo[0]
        crossed_out = new_x >= self.hi[0]
        if np.any(crossed_out):
            count = np.count_nonzero(crossed_out)
            momenta = self.mass[inds[crossed_out], None] * np.column_stack(
                (vx[crossed_out], vy[crossed_out], np.zeros(count)))
            amount = np.sum(momenta, axis=0)
            self.cumulative_out += amount
            self.cumulative_in += amount
            self.last_interval_out += amount
            self.last_interval_in += amount
        new_x = self.lo[0] + np.mod(new_x - self.lo[0], width)
        over = new_y > self.hi[1]
        under = new_y < self.lo[1]
        new_y[over] = 2.0 * self.hi[1] - new_y[over]
        new_y[under] = 2.0 * self.lo[1] - new_y[under]
        vy[over | under] *= -1.0
        surface = self._sample_eta(new_x, new_y)
        new_z = self.sigma * np.maximum(surface, 0.2)

        dx = new_x
        dz = new_z - self.ship_z
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        local_x = cp * dx - sp * dz
        local_z = sp * dx + cp * dz
        hull_half = 0.82 * np.clip(
            1.0 - np.abs(local_x) / 4.35, 0.0, 1.0) ** 0.35
        hull_bottom = -1.10 + 0.16 * np.abs(local_x) / 4.35
        inside = ((np.abs(local_x) < 4.35)
                  & (np.abs(new_y) < hull_half)
                  & (local_z > hull_bottom) & (local_z < 0.38))
        if np.any(inside):
            side = np.where(new_y[inside] >= 0.0, 1.0, -1.0)
            new_y[inside] = side * (hull_half[inside] + 0.025)
            vy[inside] = side * np.maximum(
                np.abs(vy[inside]), 0.15 * np.abs(vx[inside]))

        self.position[inds, 0] = new_x
        self.position[inds, 1] = new_y
        self.position[inds, 2] = new_z
        self.velocity[inds, 0] = vx
        self.velocity[inds, 1] = vy
        self.velocity[inds, 2] = (new_z - old[:, 2]) / dt
        si = self.ship_index
        self.position[si] = (0.0, 0.0, self.ship_z)
        self.velocity[si] = (0.0, 0.0, self.ship_vz)

    def _substep(self, dt: float) -> None:
        self._advance_fluid(dt)
        self._advance_ship(dt)
        self._advance_markers(dt)
        self.time += dt
        self.steps += 1

    def _budget_terms(self, interval: float) -> np.ndarray:
        if interval <= 0.0:
            return self.last_external_terms.copy()
        ship_p = self.mass[self.ship_index] * self.velocity[self.ship_index]
        fluid_p = np.sum(self.mass[self.fluid_indices, None]
                         * self.velocity[self.fluid_indices], axis=0)
        ship_dpdt = (ship_p - self.last_ship_momentum) / interval
        fluid_dpdt = (fluid_p - self.last_fluid_momentum) / interval
        flux_rate = (self.last_interval_in - self.last_interval_out) / interval
        gravity_ship = np.array([0.0, 0.0, -self.ship_mass * self.g])
        fluid_mass = float(np.sum(self.mass[self.fluid_indices]))
        gravity_fluid = np.array([0.0, 0.0, -fluid_mass * self.g])
        hydro = ship_dpdt - gravity_ship
        self.last_hydro_force = hydro
        fluid_external = fluid_dpdt - flux_rate + hydro
        terms = np.zeros((3, 2, 3), dtype=np.float64)
        terms[1, 0] = gravity_ship
        terms[2, 0] = gravity_fluid
        terms[2, 1] = fluid_external - gravity_fluid
        terms[0] = terms[1] + terms[2]
        self.cumulative_external += np.sum(terms, axis=1) * interval
        self.last_ship_momentum = ship_p
        self.last_fluid_momentum = fluid_p
        self.last_external_terms = terms
        return terms

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-10:
            raise ValueError("advance_to times must be nondecreasing")
        self.last_interval_in[:] = 0.0
        self.last_interval_out[:] = 0.0
        while self.time < target - 1e-12:
            self._substep(min(0.01, target - self.time))
        self.time = target
        interval = target - self.last_frame_time
        external_terms = self._budget_terms(interval)
        self.last_frame_time = target

        n = len(self.mass)
        orientation = np.zeros((n, 4), dtype=np.float32)
        orientation[:, 3] = 1.0
        half_pitch = 0.5 * self.pitch
        orientation[self.ship_index] = (
            0.0, math.sin(half_pitch), 0.0, math.cos(half_pitch))
        angular_velocity = np.zeros((n, 3), dtype=np.float32)
        angular_velocity[self.ship_index, 1] = self.pitch_rate
        weights = np.zeros((3, n), dtype=np.float32)
        weights[0, :] = 1.0
        weights[1, self.ship_index] = 1.0
        weights[2, self.fluid_indices] = 1.0
        cumulative_in = np.zeros((3, 3), dtype=np.float64)
        cumulative_out = np.zeros((3, 3), dtype=np.float64)
        cumulative_in[0] = cumulative_in[2] = self.cumulative_in
        cumulative_out[0] = cumulative_out[2] = self.cumulative_out
        interaction = np.vstack((self.last_hydro_force,
                                 -self.last_hydro_force))

        return StateFrame(
            time=target,
            position=self.position.astype(np.float32, copy=True),
            velocity=self.velocity.astype(np.float32, copy=True),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass,
            body_id=self.body_id,
            attributes={"phase": self.phase_attr,
                        "ship_visual": self.visual_attr,
                        "mass_quantity": self.mass},
            fields={
                "boundary_normal": self.boundary_normals,
                "boundary_target_velocity": self.boundary_velocity,
                "boundary_velocity": self.boundary_velocity,
                "budget_cumulative_external_impulse": self.cumulative_external.copy(),
                "budget_cumulative_inflow_momentum": cumulative_in,
                "budget_cumulative_outflow_momentum": cumulative_out,
                "budget_domain_weights": weights,
                "budget_external_force": external_terms,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "interaction_force": interaction,
            },
            scalars={"still_water_level": self.h0,
                     "reference_momentum": self.reference_momentum,
                     "reference_velocity": self.reference_velocity},
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "staggered_shallow_water_rigid_strip_coupling",
            "substeps": self.steps,
            "fluid_grid": [self.nx, self.ny],
            "heave": self.ship_z - self.ship_z0,
            "pitch_radians": self.pitch,
            "two_way_coupling": True,
        }
