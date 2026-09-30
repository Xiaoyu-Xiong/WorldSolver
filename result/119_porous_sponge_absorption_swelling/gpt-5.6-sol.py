"""Self-contained poro-elasto-capillary solver for the sponge benchmark.

The porous phase is advanced with a cell-centred finite-volume Darcy model.
The solid is a damped, lumped poroelastic discretisation whose eigenstrain is
set by local liquid content, and the exterior water is a compressible
Lagrangian lattice. Liquid entering a pore is removed from the finite exterior
phase, coupling swelling, permeability, and the reservoir.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic reduced-order physical solver using only NumPy."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self.time = 0.0
        self.hmax = 0.005
        self.dx = 0.06
        self.g = np.array((0.0, 0.0, -9.81), dtype=np.float64)

        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        p0 = np.asarray(initial.position, dtype=np.float64)
        self.ns = int(np.count_nonzero(self.body_id == 1))
        self.nw = int(np.count_nonzero(self.body_id == 2))
        self.npore = int(np.count_nonzero(self.body_id == 3))
        self.surface_ref = p0[:self.ns].copy()
        self.water_ref = p0[self.ns:self.ns + self.nw].copy()
        self.pore_ref = p0[-self.npore:].copy()
        self.surface_pos = self.surface_ref.copy()
        self.water_pos = self.water_ref.copy()
        self.pore_pos = self.pore_ref.copy()
        self.surface_vel = np.zeros_like(self.surface_pos)
        self.water_vel = np.zeros_like(self.water_pos)
        self.pore_vel = np.zeros_like(self.pore_pos)

        self.pore_shape = (25, 21, 21)
        self.water_shape = (25, 25, 19)
        if int(np.prod(self.pore_shape)) != self.npore:
            raise ValueError("unexpected porous diagnostic grid")
        if int(np.prod(self.water_shape)) != self.nw:
            raise ValueError("unexpected exterior-water grid")

        m0 = np.asarray(initial.mass, dtype=np.float64)
        self.surface_mass = m0[:self.ns].copy()
        self.water_total = float(np.sum(m0[self.ns:self.ns + self.nw]))
        self.water_initial = self.water_total
        self.pore_dry_mass = m0[-self.npore:].copy()

        self.rho = 1000.0
        self.mu = 0.05
        self.phi0 = 0.90
        self.k0 = 3.0e-8
        self.capillary_pressure = 6000.0
        self.swelling = 0.32
        self.cell_volume = self.dx ** 3
        self.saturation = np.zeros(self.pore_shape, dtype=np.float64)
        self.porosity = np.full(self.pore_shape, self.phi0, dtype=np.float64)
        self.pressure = np.full(self.pore_shape, -self.capillary_pressure,
                                dtype=np.float64)
        self.permeability = np.full(self.pore_shape, self.k0 * 0.02,
                                    dtype=np.float64)
        self.flux = np.zeros(self.pore_shape + (3,), dtype=np.float64)
        self.storage_rate = np.zeros(self.pore_shape, dtype=np.float64)

        # Momentum budgets: combined system, sponge, and exterior water.
        n = len(self.body_id)
        self.domain_weights = np.zeros((3, n), dtype=np.float32)
        self.domain_weights[0, :] = 1.0
        self.domain_weights[1, self.body_id != 2] = 1.0
        self.domain_weights[2, self.body_id == 2] = 1.0
        self.external_force = np.zeros((3, 3, 3), dtype=np.float64)
        self.external_impulse = np.zeros((3, 3), dtype=np.float64)
        self.inflow_momentum = np.zeros((3, 3), dtype=np.float64)
        self.outflow_momentum = np.zeros((3, 3), dtype=np.float64)
        self.interaction = np.zeros((2, 3), dtype=np.float64)
        self.absorbed_mass = 0.0
        self.steps = 0
        self._update_pore_diagnostics()

    def _phi_from_saturation(self, s: np.ndarray) -> np.ndarray:
        # Incompressible skeleton grains: phi = 1 - (1-phi0)/J.
        jacobian = (1.0 + self.swelling * s) ** 3
        return 1.0 - (1.0 - self.phi0) / jacobian

    def _update_pore_diagnostics(self) -> None:
        s = self.saturation
        self.porosity = self._phi_from_saturation(s)
        # Regularised relative permeability permits dry-front imbibition.
        rel = 0.02 + 0.98 * s ** 3
        porosity_factor = ((self.porosity / self.phi0) ** 3
                           * ((1.0 - self.phi0) /
                              np.maximum(1.0 - self.porosity, 1.0e-6)) ** 2)
        self.permeability = np.minimum(self.k0 * rel * porosity_factor,
                                       4.0 * self.k0)
        self.pressure = -self.capillary_pressure * (1.0 - s)

    def _advance_pores(self, h: float) -> float:
        """Take one conservative finite-volume Darcy step."""
        old_storage = self.porosity * self.saturation
        p, k, d = self.pressure, self.permeability, self.dx
        nx, ny, nz = self.pore_shape
        fx = np.zeros((nx + 1, ny, nz), dtype=np.float64)
        fy = np.zeros((nx, ny + 1, nz), dtype=np.float64)
        fz = np.zeros((nx, ny, nz + 1), dtype=np.float64)

        kf = 2.0 * k[:-1] * k[1:] / np.maximum(k[:-1] + k[1:], 1.0e-30)
        fx[1:nx] = -(kf / self.mu) * ((p[1:] - p[:-1]) / d)
        kf = 2.0 * k[:, :-1] * k[:, 1:] / np.maximum(
            k[:, :-1] + k[:, 1:], 1.0e-30)
        fy[:, 1:ny] = -(kf / self.mu) * ((p[:, 1:] - p[:, :-1]) / d)
        kf = 2.0 * k[:, :, :-1] * k[:, :, 1:] / np.maximum(
            k[:, :, :-1] + k[:, :, 1:], 1.0e-30)
        fz[:, :, 1:nz] = -(kf / self.mu) * (
            (p[:, :, 1:] - p[:, :, :-1]) / d - self.rho * self.g[2])

        # Hydrostatic finite reservoir contacts the sponge's right face.
        footprint = (1.75 - 0.28) * (0.72 - (-0.72))
        water_height = min(1.15, self.water_total / (self.rho * footprint))
        z = self.pore_ref.reshape(self.pore_shape + (3,))[0, 0, :, 2]
        wet_face = z <= water_height
        pext = self.rho * 9.81 * np.maximum(water_height - z, 0.0)
        right_flux = -(k[-1] / self.mu) * (pext[None, :] - p[-1]) / (0.5 * d)
        right_flux = np.minimum(right_flux, 0.0)
        fx[nx] = np.where(wet_face[None, :], right_flux, 0.0)

        proposed_volume = max(0.0, -float(np.sum(fx[nx])) * d * d * h)
        available_volume = max(0.0, self.water_total / self.rho)
        if proposed_volume > available_volume and proposed_volume > 0.0:
            fx[nx] *= available_volume / proposed_volume

        div = ((fx[1:] - fx[:-1]) + (fy[:, 1:] - fy[:, :-1])
               + (fz[:, :, 1:] - fz[:, :, :-1])) / d
        new_storage = np.clip(old_storage - h * div, 0.0, 0.999999)

        # Invert phi(S)*S by fixed-point iteration.
        snew = np.clip(new_storage / self.phi0, 0.0, 1.0)
        for _ in range(4):
            snew = np.clip(new_storage / self._phi_from_saturation(snew), 0.0, 1.0)
        actual_storage = self._phi_from_saturation(snew) * snew
        absorbed = self.rho * self.cell_volume * float(
            np.sum(actual_storage - old_storage))
        absorbed = min(max(absorbed, 0.0), self.water_total)
        self.water_total -= absorbed
        self.absorbed_mass += absorbed
        self.storage_rate = (actual_storage - old_storage) / h
        self.saturation = snew
        self._update_pore_diagnostics()

        self.flux[..., 0] = 0.5 * (fx[:-1] + fx[1:])
        self.flux[..., 1] = 0.5 * (fy[:, :-1] + fy[:, 1:])
        self.flux[..., 2] = 0.5 * (fz[:, :, :-1] + fz[:, :, 1:])
        return absorbed

    def _sample_grid(self, points: np.ndarray, field: np.ndarray) -> np.ndarray:
        origin = np.array((-1.22, -0.62, 0.03))
        ijk = np.rint((points - origin) / self.dx).astype(np.int64)
        ijk[:, 0] = np.clip(ijk[:, 0], 0, self.pore_shape[0] - 1)
        ijk[:, 1] = np.clip(ijk[:, 1], 0, self.pore_shape[1] - 1)
        ijk[:, 2] = np.clip(ijk[:, 2], 0, self.pore_shape[2] - 1)
        return field[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    def _advance_solid(self, h: float, reaction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pore_s = self.saturation.reshape(-1)
        surf_s = self._sample_grid(self.surface_ref, self.saturation)
        center = np.array((-0.50, 0.0, 0.0))
        pore_target = center + (1.0 + self.swelling * pore_s)[:, None] * (
            self.pore_ref - center)
        surf_target = center + (1.0 + self.swelling * surf_s)[:, None] * (
            self.surface_ref - center)

        pos = np.concatenate((self.surface_pos, self.pore_pos))
        vel = np.concatenate((self.surface_vel, self.pore_vel))
        target = np.concatenate((surf_target, pore_target))
        pore_liquid = self.rho * self.cell_volume * (
            self.porosity.reshape(-1) * pore_s)
        masses = np.concatenate((self.surface_mass,
                                 self.pore_dry_mass + pore_liquid))
        total_mass = float(np.sum(masses))
        vcom = np.sum(masses[:, None] * vel, axis=0) / total_mass
        accel = 36.0 * (target - pos) - 7.0 * (vel - vcom)
        # Elastic/eigenstress forces are internal and may not accelerate COM.
        accel -= np.sum(masses[:, None] * accel, axis=0)[None, :] / total_mass
        accel += self.g + reaction[None, :] / total_mass

        support_accel = np.zeros_like(accel)
        penetration = np.maximum(-pos[:, 2], 0.0)
        supported = (pos[:, 2] <= 1.0e-10) & (vel[:, 2] <= 0.0)
        support_accel[:, 2] = (
            9000.0 * penetration
            + 90.0 * np.maximum(-vel[:, 2], 0.0) * (penetration > 0.0)
            + np.maximum(-accel[:, 2], 0.0) * supported)
        contact = pos[:, 2] < 0.012
        support_accel[contact, :2] = -8.0 * vel[contact, :2]
        accel += support_accel
        vel += h * accel
        pos += h * vel
        below = pos[:, 2] < 0.0
        pos[below, 2] = 0.0
        vel[below & (vel[:, 2] < 0.0), 2] = 0.0

        self.surface_pos, self.pore_pos = pos[:self.ns], pos[self.ns:]
        self.surface_vel, self.pore_vel = vel[:self.ns], vel[self.ns:]
        support_force = np.sum(masses[:, None] * support_accel, axis=0)
        return masses, support_force

    def _water_lattice_acceleration(self) -> np.ndarray:
        pos = self.water_pos.reshape(self.water_shape + (3,))
        ref = self.water_ref.reshape(self.water_shape + (3,))
        accel = np.zeros_like(pos)
        stiffness = 1500.0
        for axis in range(3):
            lower = [slice(None)] * 4
            upper = [slice(None)] * 4
            lower[axis], upper[axis] = slice(0, -1), slice(1, None)
            lo, hi = tuple(lower), tuple(upper)
            extension = (pos[hi] - pos[lo]) - (ref[hi] - ref[lo])
            force = stiffness * extension
            accel[lo] += force
            accel[hi] -= force
        return accel.reshape(-1, 3)

    def _advance_water(self, h: float) -> tuple[np.ndarray, np.ndarray]:
        mass_each = self.water_total / self.nw
        accel = self._water_lattice_acceleration()
        damping_accel = -1.2 * self.water_vel
        accel += self.g + damping_accel

        support_accel = np.zeros_like(accel)
        penetration = np.maximum(0.025 - self.water_pos[:, 2], 0.0)
        supported = ((self.water_pos[:, 2] <= 0.025 + 1.0e-10)
                     & (self.water_vel[:, 2] <= 0.0))
        support_accel[:, 2] = (
            7000.0 * penetration
            + 80.0 * np.maximum(-self.water_vel[:, 2], 0.0)
            * (penetration > 0.0)
            + np.maximum(-accel[:, 2], 0.0) * supported)
        accel += support_accel

        # Collision with the evolving sponge provides mechanical two-way coupling.
        face = float(np.max(self.surface_pos[:, 0]))
        ymin, ymax = (float(np.min(self.surface_pos[:, 1])),
                      float(np.max(self.surface_pos[:, 1])))
        zmax = float(np.max(self.surface_pos[:, 2]))
        in_front = ((self.water_pos[:, 1] >= ymin)
                    & (self.water_pos[:, 1] <= ymax)
                    & (self.water_pos[:, 2] <= zmax + 0.03)
                    & (self.water_pos[:, 0] > face - 0.12))
        pen = np.maximum(face + 0.03 - self.water_pos[:, 0], 0.0)
        collision_ax = np.where(
            in_front & (pen > 0.0),
            np.maximum(4500.0 * pen - 45.0 * self.water_vel[:, 0], 0.0), 0.0)
        accel[:, 0] += collision_ax
        fluid_contact_force = np.array((mass_each * float(np.sum(collision_ax)),
                                        0.0, 0.0))

        self.water_vel += h * accel
        self.water_pos += h * self.water_vel
        below = self.water_pos[:, 2] < 0.025
        self.water_pos[below, 2] = 0.025
        self.water_vel[below & (self.water_vel[:, 2] < 0.0), 2] = 0.0
        support_force = mass_each * np.sum(support_accel, axis=0)
        damping_force = mass_each * np.sum(damping_accel, axis=0)
        return fluid_contact_force, np.stack((support_force, damping_force))

    def _step(self, h: float) -> None:
        water_velocity_before = np.mean(self.water_vel, axis=0)
        absorbed = self._advance_pores(h)
        contact_on_water, water_other_forces = self._advance_water(h)
        reaction_on_solid = -contact_on_water
        solid_masses, solid_support = self._advance_solid(h, reaction_on_solid)
        water_mass, solid_mass = self.water_total, float(np.sum(solid_masses))

        self.interaction[0], self.interaction[1] = reaction_on_solid, contact_on_water
        self.external_force.fill(0.0)
        # Force terms K are gravity, floor support, and physical damping.
        self.external_force[1, 0] = solid_mass * self.g
        self.external_force[1, 1] = solid_support
        self.external_force[2, 0] = water_mass * self.g
        self.external_force[2, 1] = water_other_forces[0]
        self.external_force[2, 2] = water_other_forces[1]
        self.external_force[0] = self.external_force[1] + self.external_force[2]
        self.external_impulse += h * np.sum(self.external_force, axis=1)

        transfer_momentum = absorbed * water_velocity_before
        self.inflow_momentum[1] += transfer_momentum
        self.outflow_momentum[2] += transfer_momentum
        self.time += h
        self.steps += 1

    def _make_frame(self) -> StateFrame:
        pore_s = self.saturation.reshape(-1)
        surf_s = self._sample_grid(self.surface_ref, self.saturation)
        surf_phi = self._sample_grid(self.surface_ref, self.porosity)
        surf_p = self._sample_grid(self.surface_ref, self.pressure)
        surf_q = self._sample_grid(self.surface_ref, self.flux)
        pore_liquid = self.rho * self.cell_volume * self.porosity.reshape(-1) * pore_s
        water_mass = np.full(self.nw, self.water_total / self.nw)
        mass = np.concatenate((self.surface_mass, water_mass,
                               self.pore_dry_mass + pore_liquid))
        position = np.concatenate((self.surface_pos, self.water_pos, self.pore_pos))
        velocity = np.concatenate((self.surface_vel, self.water_vel, self.pore_vel))
        n = len(mass)

        saturation = np.concatenate((surf_s, np.ones(self.nw), pore_s))
        porosity = np.concatenate((surf_phi, np.zeros(self.nw),
                                   self.porosity.reshape(-1)))
        pore_pressure = np.concatenate((surf_p, np.zeros(self.nw),
                                        self.pressure.reshape(-1)))
        darcy_flux = np.concatenate((surf_q, np.zeros((self.nw, 3)),
                                     self.flux.reshape(-1, 3)))
        solid_velocity = np.zeros((n, 3), dtype=np.float64)
        solid_velocity[:self.ns] = self.surface_vel
        solid_velocity[-self.npore:] = self.pore_vel

        permeability_tensor = np.zeros(self.pore_shape + (3, 3), dtype=np.float32)
        kval = self.permeability.astype(np.float32)
        permeability_tensor[..., 0, 0] = kval
        permeability_tensor[..., 1, 1] = kval
        permeability_tensor[..., 2, 2] = kval
        fields = {
            "budget_cumulative_external_impulse": self.external_impulse.astype(np.float32),
            "budget_cumulative_inflow_momentum": self.inflow_momentum.astype(np.float32),
            "budget_cumulative_outflow_momentum": self.outflow_momentum.astype(np.float32),
            "budget_domain_weights": self.domain_weights,
            "budget_external_force": self.external_force.astype(np.float32),
            "cumulative_mass_source": np.zeros(2, dtype=np.float32),
            "gravity_acceleration": self.g.reshape(1, 3).astype(np.float32),
            "interaction_force": self.interaction.astype(np.float32),
            "pore_active_mask": np.ones(self.pore_shape, dtype=np.float32),
            "pore_permeability": permeability_tensor,
            "pore_porosity": self.porosity.astype(np.float32),
            "pore_pressure": self.pressure.astype(np.float32),
            "pore_relative_flux": self.flux.astype(np.float32),
            "pore_saturation": self.saturation.astype(np.float32),
            "pore_solid_velocity": self.pore_vel.reshape(
                self.pore_shape + (3,)).astype(np.float32),
            "pore_volume_source": np.zeros(self.pore_shape, dtype=np.float32),
        }
        attributes = {
            "phase": np.concatenate((np.zeros(self.ns), np.ones(self.nw),
                                     np.zeros(self.npore))).astype(np.float32),
            "saturation": saturation.astype(np.float32),
            "porosity": porosity.astype(np.float32),
            "pore_pressure": pore_pressure.astype(np.float32),
            "darcy_flux": darcy_flux.astype(np.float32),
            "solid_velocity": solid_velocity.astype(np.float32),
            "mass_quantity": mass.astype(np.float32),
        }
        scalars = {
            "gravity_z": -9.81,
            "fluid_volume_initial": self.water_initial / self.rho,
            "dynamic_viscosity": self.mu,
            "grid_spacing": self.dx,
            "pore_fluid_density": self.rho,
            "reference_darcy_flux": 0.02,
            "reference_momentum": max(self.water_initial * 0.2, 1.0),
            "reference_pore_storage_rate": 0.1,
        }
        frame = StateFrame(time=float(self.time),
                           position=position.astype(np.float32),
                           velocity=velocity.astype(np.float32),
                           mass=mass.astype(np.float32),
                           body_id=self.body_id,
                           attributes=attributes,
                           fields=fields,
                           scalars=scalars)
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing times")
        while self.time < target - 1.0e-12:
            self._step(min(self.hmax, target - self.time))
        self.time = target
        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "finite_volume_darcy_lumped_poroelastic",
            "internal_steps": self.steps,
            "absorbed_mass_kg": float(self.absorbed_mass),
            "remaining_water_mass_kg": float(self.water_total),
            "maximum_saturation": float(np.max(self.saturation)),
        }
