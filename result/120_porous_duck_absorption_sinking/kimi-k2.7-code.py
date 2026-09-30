"""Reduced-order coupled solver for the porous-duck absorption experiment.

The duck is treated as a vertically translating rigid porous body.  Free water
is represented by the same particle set used by the scaffold, but the water
surface height and the particle vertical positions are evolved from mass
conservation.  The duck is coupled to the water through hydrostatic buoyancy,
drag, and Darcy-like absorption once permeability is enabled at the prescribed
switch time.

This implementation does not import or use Genesis/gs in any way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame
from worldsolver.config import load_yaml


class BenchmarkSolver:
    """Simple physically-based solver for the porous duck scene."""

    def __init__(self) -> None:
        self._initialized = False

    # --------------------------------------------------------------------- #
    # Public protocol
    # --------------------------------------------------------------------- #
    def initialize(self, scene, seed: int) -> None:
        self._seed = seed
        self._scene = scene

        # Load authoritative parameters from the frozen scaffold.
        config = load_yaml(Path("config.yaml"))
        duck_cfg = config["duck"]
        fluid_cfg = config["fluid"]
        perm_cfg = config["permeability_control"]

        self.gravity = np.asarray(config["gravity"], dtype=np.float64)
        self.rho_fluid = float(fluid_cfg["density_kg_m3"])
        self.rho_dry = float(duck_cfg["dry_bulk_density_kg_m3"])
        self.porosity = float(duck_cfg["porosity"])
        self.S0 = float(duck_cfg["initial_saturation"])
        self.h0 = float(fluid_cfg["initial_surface_z"])
        self.switch_time = float(perm_cfg["switch_time"])
        self.k_perm = float(perm_cfg["intrinsic_permeability_after_switch_m2"])

        # Initial state from the frozen scene.
        frame0 = scene.initial_state()
        self.t = 0.0
        self.z = 0.0          # vertical duck displacement (m)
        self.v = 0.0          # vertical duck velocity (m/s)
        self.S = float(self.S0)

        # Core arrays.
        self.pos0 = np.asarray(frame0.position, dtype=np.float64)
        self.vel = np.asarray(frame0.velocity, dtype=np.float64)
        self.mass = np.asarray(frame0.mass, dtype=np.float64)
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32)

        # Attribute copies / defaults.
        self.attr = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in frame0.attributes.items()
        }
        for name, default in (
            ("phase", 0.0),
            ("saturation", 0.0),
            ("porosity", 0.0),
            ("permeability_m2", 0.0),
            ("pore_water_mass", 0.0),
            ("pore_pressure", 0.0),
            ("solid_velocity", 0.0),
        ):
            if name not in self.attr:
                self.attr[name] = np.full_like(self.mass, default)

        # Domain indices.
        self.solid_idx = np.where((self.body_id == 1) | (self.body_id == 3))[0]
        self.water_idx = np.where(self.body_id == 2)[0]

        # Per-sample dry mass and solid volume.
        self.dry_mass = np.zeros_like(self.mass)
        self.dry_mass[self.solid_idx] = self.mass[self.solid_idx]
        self.solid_volume = np.zeros_like(self.mass)
        self.solid_volume[self.solid_idx] = self.dry_mass[self.solid_idx] / self.rho_dry

        # Water volume (constant cell volume).
        if len(self.water_idx):
            self.cell_volume = float(self.mass[self.water_idx[0]] / self.rho_fluid)
        else:
            self.cell_volume = 0.065 ** 3

        # Duck/pore bookkeeping.
        self.V_duck = float(self.solid_volume[self.solid_idx].sum())
        self.V_pore_max = self.porosity * self.V_duck
        self.V_free0 = float(len(self.water_idx) * self.cell_volume)
        self.V_sub0 = self._submerged_volume(self.z, self.h0)
        self.dry_total = float(self.dry_mass[self.solid_idx].sum())

        # Cached min duck z for floor contact.
        self._min_z0 = float(self.pos0[self.solid_idx, 2].min()) if len(self.solid_idx) else 0.0

        # Budget bookkeeping.
        self.G = 3  # [free water, duck, tank]
        self.K = 2  # [dry material, water]
        self.cum_ext_impulse = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_in = np.zeros((self.G, 3), dtype=np.float64)
        self.cum_out = np.zeros((self.G, 3), dtype=np.float64)

        # Reference momentum scale (frozen positive scalar).
        char_speed = 1.0
        self.reference_momentum = max(1.0, self.dry_total * char_speed)

        # Store the initial frame (time = 0) for the first call.
        self._frame0 = self._build_frame(0.0)
        self._initialized = True

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("Solver has not been initialized")
        if time < self.t - 1e-12:
            raise ValueError("advance_to must be called with non-decreasing time")

        if time <= 0.0:
            return self._frame0

        target = float(time)
        # Fixed sub-step; choose a value that divides the sample interval cleanly.
        dt = 0.001
        while self.t < target:
            step = min(dt, target - self.t)
            self._substep(step)
            self.t += step

        return self._build_frame(self.t)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "reduced_order_porous_duck_1d_coupled",
            "substeps_per_sample": 20,
            "duck_total_volume": self.V_duck,
            "duck_dry_mass": self.dry_total,
            "porosity": self.porosity,
        }

    # --------------------------------------------------------------------- #
    # Physics helpers
    # --------------------------------------------------------------------- #
    def _submerged_volume(self, z_duck: float, h: float) -> float:
        """Submerged volume of the duck for a given duck offset and water height."""
        if self.V_duck <= 0.0:
            return 0.0
        solid_z = self.pos0[self.solid_idx, 2] + z_duck
        vol = self.solid_volume[self.solid_idx]
        return float(np.where(solid_z < h, vol, 0.0).sum())

    def _wetted_volume(self, z_duck: float, h: float) -> float:
        """Pore volume currently in contact with free water."""
        return self._submerged_volume(z_duck, h)

    def _pore_water_mass(self) -> float:
        return self.rho_fluid * self.V_pore_max * self.S

    def _update_h(self, z_duck: float) -> float:
        """Water surface height from volume conservation.

        free_volume = V_free0 - V_pore + (V_sub - V_sub0)
        """
        V_pore = self.V_pore_max * self.S
        # First guess at the current water surface.
        h = self.h0
        for _ in range(3):
            Vsub = self._submerged_volume(z_duck, h)
            free_vol = self.V_free0 - V_pore + (Vsub - self.V_sub0)
            ratio = free_vol / self.V_free0 if self.V_free0 > 0 else 1.0
            ratio = max(ratio, 0.05)
            h = self.h0 * ratio
        return h

    def _substep(self, dt: float) -> None:
        # Current water surface.
        h = self._update_h(self.z)
        V_sub = self._submerged_volume(self.z, h)
        V_wet = V_sub

        # Absorption of free water into pores after the permeability switch.
        dSdt = 0.0
        if self.t >= self.switch_time and self.S < 1.0 and self.V_pore_max > 0.0:
            mu = 1e-3          # dynamic viscosity of water (Pa.s)
            L = 0.1            # characteristic duck length (m)
            pressure_scale = self.rho_fluid * (-self.gravity[2]) * L
            rate_coeff = (self.k_perm / mu) * (pressure_scale / L)
            # Wetted fraction of pore space drives saturation.
            wet_fraction = V_wet / self.V_pore_max
            dSdt = rate_coeff * wet_fraction * (1.0 - self.S)
            dSdt = min(dSdt, (1.0 - self.S) / max(dt, 1e-12))
        dSdt = max(dSdt, 0.0)

        # Current duck mass (dry + absorbed water).
        M_pore = self._pore_water_mass()
        M = self.dry_total + M_pore

        # Forces in the vertical (z) direction.
        g_z = self.gravity[2]            # negative
        F_grav = M * g_z
        F_buoy = self.rho_fluid * (-g_z) * V_sub

        # Quadratic drag on the submerged duck volume.
        A_sub = max(V_sub, 0.0) ** (2.0 / 3.0)
        F_drag = -0.5 * self.rho_fluid * A_sub * abs(self.v) * self.v

        # Predict velocity/position.
        a = (F_grav + F_buoy + F_drag) / max(M, 1e-12)
        v_new = self.v + a * dt
        z_new = self.z + v_new * dt

        # Floor contact (tank floor top at z = 0).
        z_floor_offset = -self._min_z0
        contact_impulse = np.zeros(3, dtype=np.float64)
        if z_new < z_floor_offset:
            v_before = v_new
            z_new = z_floor_offset
            v_new = 0.0
            # Record upward impulse for the duck budget.
            impulse_mag = M * (-v_before) if v_before < 0 else 0.0
            contact_impulse = np.array([0.0, 0.0, impulse_mag], dtype=np.float64)

        # Update state.
        self.z = z_new
        self.v = v_new
        old_S = self.S
        self.S = min(1.0, self.S + dSdt * dt)
        actual_dS = self.S - old_S

        # Update water surface and particle positions.
        h = self._update_h(self.z)
        ratio = h / self.h0 if self.h0 > 0 else 1.0
        # Water particle vertical positions scale with surface height.
        old_water_z = self.pos0[self.water_idx, 2].copy()
        new_water_z = old_water_z * ratio
        self.vel[self.water_idx, 0] = 0.0
        self.vel[self.water_idx, 1] = 0.0
        self.vel[self.water_idx, 2] = (new_water_z - old_water_z) / dt
        # Positions are rebuilt on demand in _build_frame.

        # ---- Budget accounting ----
        # Free water volume and per-particle water mass.
        V_pore = self.V_pore_max * self.S
        free_vol = self.V_free0 - V_pore + (self._submerged_volume(self.z, h) - self.V_sub0)
        free_vol = max(free_vol, 0.0)
        n_water = len(self.water_idx)
        m_w = self.rho_fluid * free_vol / max(n_water, 1)

        # External gravity on each material per domain.
        # Domain 0: free water; Domain 1: duck; Domain 2: tank (no particles).
        # Material 0: dry solid; Material 1: water (free + pore).
        dry_m = self.dry_mass
        water_m = np.zeros_like(self.mass)
        # Pore water mass on solid particles.
        pore_per_sample = self.S * self.porosity * self.solid_volume * self.rho_fluid
        water_m[self.solid_idx] = pore_per_sample[self.solid_idx]
        water_m[self.water_idx] = m_w

        # Gravity forces per sample, per material.
        F_grav_dry = dry_m[:, None] * self.gravity[None, :]
        F_grav_wat = water_m[:, None] * self.gravity[None, :]

        # Domain weights.
        weights = np.zeros((self.G, len(self.mass)), dtype=np.float64)
        if n_water:
            weights[0, self.water_idx] = 1.0
        if len(self.solid_idx):
            weights[1, self.solid_idx] = 1.0
        # Tank domain remains zero (no particles).

        # External force per domain and material.
        ext_force = np.zeros((self.G, self.K, 3), dtype=np.float64)
        ext_force[:, 0, :] = weights @ F_grav_dry
        ext_force[:, 1, :] = weights @ F_grav_wat

        # Floor contact is external to the duck (domain 1), applied to dry material.
        if contact_impulse[2] > 0.0:
            ext_force[1, 0, :] += contact_impulse / dt

        # Cumulative external impulse.
        self.cum_ext_impulse += ext_force.sum(axis=1) * dt

        # Mass transfer momentum bookkeeping.
        if actual_dS > 0.0 and n_water:
            dm_dt = self.rho_fluid * self.V_pore_max * (actual_dS / dt)
            v_water_avg = float(self.vel[self.water_idx, 2].mean())
            momentum_vec = np.array([0.0, 0.0, dm_dt * v_water_avg], dtype=np.float64)
            # Water loses this momentum (outflow), duck gains it (inflow).
            self.cum_out[0, :] += momentum_vec
            self.cum_in[1, :] += momentum_vec

        # ---- Update persistent arrays ----
        self.mass[self.solid_idx] = self.dry_mass[self.solid_idx] + pore_per_sample[self.solid_idx]
        self.mass[self.water_idx] = m_w
        self.attr["saturation"][self.solid_idx] = self.S
        self.attr["saturation"][self.water_idx] = 1.0
        self.attr["pore_water_mass"][self.solid_idx] = pore_per_sample[self.solid_idx]
        self.attr["permeability_m2"][self.solid_idx] = 0.0 if self.t < self.switch_time else self.k_perm
        self.attr["solid_velocity"][self.solid_idx] = 0.0
        self.attr["solid_velocity"][self.solid_idx, 2] = self.v

    # --------------------------------------------------------------------- #
    # Frame construction
    # --------------------------------------------------------------------- #
    def _build_frame(self, time: float) -> StateFrame:
        # Build current positions from the base configuration plus duck offset.
        positions = self.pos0.copy()
        if len(self.solid_idx):
            positions[self.solid_idx, 2] += self.z

        # Velocities.
        velocities = self.vel.copy()
        if len(self.solid_idx):
            velocities[self.solid_idx] = 0.0
            velocities[self.solid_idx, 2] = self.v

        # Water surface ratio for current positions.
        h = self._update_h(self.z)
        ratio = h / self.h0 if self.h0 > 0 else 1.0
        if len(self.water_idx):
            old_z = self.pos0[self.water_idx, 2]
            positions[self.water_idx, 2] = old_z * ratio
            # Recompute water velocity from the stored derivative.
            velocities[self.water_idx, 0] = 0.0
            velocities[self.water_idx, 1] = 0.0
            # Keep the per-particle vertical velocity set in _substep.
            velocities[self.water_idx, 2] = self.vel[self.water_idx, 2]

        # Mass quantity [N, K].
        water_m = np.zeros_like(self.mass)
        pore_per_sample = self.S * self.porosity * self.solid_volume * self.rho_fluid
        water_m[self.solid_idx] = pore_per_sample[self.solid_idx]
        water_m[self.water_idx] = self.mass[self.water_idx]
        mass_quantity = np.stack([self.dry_mass, water_m], axis=1)
        self.mass = mass_quantity.sum(axis=1)

        # Interaction force [K, 3]: solid<->fluid pressure/drag pair.
        V_sub = self._submerged_volume(self.z, h)
        g = -self.gravity[2]
        F_buoy = self.rho_fluid * g * V_sub
        A_sub = max(V_sub, 0.0) ** (2.0 / 3.0)
        F_drag = -0.5 * self.rho_fluid * A_sub * abs(self.v) * self.v
        F_on_solid = np.array([0.0, 0.0, F_buoy + F_drag], dtype=np.float64)
        interaction_force = np.stack([F_on_solid, -F_on_solid], axis=0)

        # Budget external force snapshot at this frame.
        dry_m = self.dry_mass
        water_m_frame = np.zeros_like(self.mass)
        water_m_frame[self.solid_idx] = pore_per_sample[self.solid_idx]
        water_m_frame[self.water_idx] = self.mass[self.water_idx]
        F_grav_dry = dry_m[:, None] * self.gravity[None, :]
        F_grav_wat = water_m_frame[:, None] * self.gravity[None, :]
        weights = np.zeros((self.G, len(self.mass)), dtype=np.float64)
        if len(self.water_idx):
            weights[0, self.water_idx] = 1.0
        if len(self.solid_idx):
            weights[1, self.solid_idx] = 1.0
        budget_external_force = np.zeros((self.G, self.K, 3), dtype=np.float64)
        budget_external_force[:, 0, :] = weights @ F_grav_dry
        budget_external_force[:, 1, :] = weights @ F_grav_wat

        fields = {
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_in.copy(),
            "budget_cumulative_outflow_momentum": self.cum_out.copy(),
            "budget_domain_weights": weights.copy(),
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": np.zeros(self.K, dtype=np.float64),
            "interaction_force": interaction_force,
        }

        scalars = {
            "reference_momentum": float(self.reference_momentum),
        }

        attributes = dict(self.attr)
        attributes["mass_quantity"] = mass_quantity

        frame = StateFrame(
            time=float(time),
            position=positions,
            velocity=velocities,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame
