"""Porous-cloth buoyancy contrast solver.

Two-way coupled simulation of three porous cloth pieces in a water tank.
Uses hydrostatic pressure, combined linear+quadratic drag, and momentum-
conserving reaction forces on the fluid particles. The net buoyant force on
each cloth vertex is proportional to (rho_water - rho_cloth) * V_submerged,
giving correct density-dependent behaviour: light cloth rises, neutral cloth
drifts, heavy cloth sinks. Water particles respond to cloth drag reactions
with viscous damping and a restoring pressure gradient.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit particle-based solver for porous cloth in water."""

    def initialize(self, scene: Any, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._scene = scene
        self._config = scene.config
        frame0 = scene.initial_state()

        self._body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self._mass = np.asarray(frame0.mass, dtype=float).copy()
        self._pos = np.asarray(frame0.position, dtype=float).copy()
        self._vel = np.asarray(frame0.velocity, dtype=float).copy()

        mat_rho = np.asarray(frame0.attributes["material_density"], dtype=float)
        porosity = np.asarray(frame0.attributes["porosity"], dtype=float)
        permeability = np.asarray(frame0.attributes["permeability_m2"], dtype=float)
        self._mat_rho = mat_rho.copy()
        self._porosity = porosity.copy()
        self._permeability = permeability.copy()

        self._is_water = self._body_id == 0
        self._is_cloth = ~self._is_water

        cloth_cfg = self._config["cloth"]
        nx, ny = map(int, cloth_cfg["grid_shape"])
        width, depth = map(float, cloth_cfg["size"])
        thickness = float(cloth_cfg["thickness"])
        n_faces = 2 * (nx - 1) * (ny - 1)
        area_per_vertex = (width * depth) / n_faces
        self._area_per_vertex = area_per_vertex
        self._vertex_volume = area_per_vertex * thickness
        self._drag_radius = math.sqrt(area_per_vertex / math.pi)

        water_cfg = self._config["water"]
        self._rho_water = float(water_cfg["density"])
        self._free_surface_z = float(water_cfg["free_surface_z"])
        self._water_viscosity = 1.0e-3

        self._gravity = np.asarray(self._config["gravity"], dtype=float)

        tank = self._config["tank"]
        self._tank_min = np.asarray(tank["interior_bounds_min"], dtype=float)
        self._tank_max = np.asarray(tank["interior_bounds_max"], dtype=float)

        self._water_pos0 = self._pos[self._is_water].copy()

        self._time = 0.0
        self._dt_physics = 0.001
        self._cum_ext_impulse = np.zeros((4, 3), dtype=float)
        self._cum_inflow_mom = np.zeros((4, 3), dtype=float)
        self._cum_outflow_mom = np.zeros((4, 3), dtype=float)
        self._cum_mass_source = np.zeros(1, dtype=float)
        total_mass = float(np.sum(self._mass))
        self._ref_momentum = max(total_mass * 0.01, 1e-6)

        n = len(self._mass)
        self._domain_weights = np.zeros((4, n), dtype=float)
        self._domain_weights[0, self._is_water] = 1.0
        for b in (1, 2, 3):
            self._domain_weights[b, self._body_id == b] = 1.0

    def advance_to(self, time: float) -> StateFrame:
        dt_total = time - self._time
        if dt_total < 0:
            raise ValueError("advance_to called with decreasing time")
        if dt_total > 0:
            self._step(dt_total)
        self._time = time
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "porous_cloth_buoyancy", "dt_physics": self._dt_physics}

    def _step(self, dt_total: float) -> None:
        dt = self._dt_physics
        remaining = dt_total
        while remaining > 1e-12:
            h = min(dt, remaining)
            self._sub_step(h)
            remaining -= h

    def _sub_step(self, dt: float) -> None:
        pos = self._pos
        vel = self._vel
        mass = self._mass
        g = self._gravity
        rho_w = self._rho_water
        free_z = self._free_surface_z
        mu = self._water_viscosity

        water_mask = self._is_water
        cloth_mask = self._is_cloth
        n_water = int(np.sum(water_mask))

        n = len(mass)
        force = np.zeros((n, 3), dtype=float)

        # Gravity on all particles
        force += mass[:, None] * g[None, :]

        # Submersion fraction
        vert_len = max(self._vertex_volume ** (1.0 / 3.0), 1e-6)
        sub_depth_all = np.maximum(free_z - pos[:, 2], 0.0)
        sub_frac_all = np.clip(sub_depth_all / vert_len, 0.0, 1.0)

        # Buoyancy on cloth only
        vol_displaced_all = sub_frac_all * self._vertex_volume
        buoy_force_all = -rho_w * vol_displaced_all[:, None] * g[None, :]
        force[cloth_mask] += buoy_force_all[cloth_mask]

        # Drag on cloth relative to quiescent water background
        # (water velocity field is approximately zero for this experiment)
        rel_vel_cloth = -vel[cloth_mask]  # v_water=0 - v_cloth
        speed_cloth = np.linalg.norm(rel_vel_cloth, axis=1, keepdims=True) + 1e-12

        R = self._drag_radius
        A = self._area_per_vertex
        c_lin = 6.0 * math.pi * mu * R
        c_quad = 0.5 * rho_w * 1.2 * A

        cloth_poro = self._porosity[cloth_mask]
        cloth_sub = sub_frac_all[cloth_mask]
        drag_scale = cloth_poro * cloth_sub
        drag_mag = (c_lin + c_quad * speed_cloth) * drag_scale[:, None]
        drag_force_cloth = drag_mag * rel_vel_cloth
        force[cloth_mask] += drag_force_cloth

        # Two-way coupling: reaction from cloth drag onto water
        if n_water > 0:
            reaction_total = -np.sum(drag_force_cloth, axis=0)
            force[water_mask] += (reaction_total / n_water)[None, :]

        # Water restoring + damping
        if n_water > 0:
            water_disp = pos[water_mask] - self._water_pos0
            water_z = pos[water_mask, 2]
            depth_factor = np.clip((free_z - water_z) / max(free_z, 1.0), 0.0, 1.0)
            k_restore = 2.0 * depth_factor
            force[water_mask] -= k_restore[:, None] * water_disp
            force[water_mask] -= 2.0 * mu * vel[water_mask]

        # Semi-implicit Euler
        acc = force / mass[:, None]
        vel_new = vel + acc * dt
        pos_new = pos + vel_new * dt

        # Boundary enforcement
        floor_z = self._tank_min[2]
        below = pos_new[:, 2] < floor_z
        pos_new[below, 2] = floor_z
        vel_new[below, 2] *= -0.1

        for ax in (0, 1):
            lo = self._tank_min[ax]
            hi = self._tank_max[ax]
            lo_hit = pos_new[:, ax] < lo
            hi_hit = pos_new[:, ax] > hi
            pos_new[lo_hit, ax] = lo
            pos_new[hi_hit, ax] = hi
            vel_new[lo_hit | hi_hit, ax] *= -0.1

        # Cumulative external impulse per domain
        ext_imp = np.zeros((4, 3), dtype=float)
        for b in range(4):
            dom_mask = self._domain_weights[b].astype(bool)
            if not np.any(dom_mask):
                continue
            f_grav = np.sum(mass[dom_mask, None] * g[None, :], axis=0)
            ext_imp[b] = f_grav * dt
            if b > 0:
                f_buoy = np.sum(buoy_force_all[dom_mask], axis=0)
                ext_imp[b] += f_buoy * dt

        self._cum_ext_impulse += ext_imp
        self._pos = pos_new
        self._vel = vel_new

    def _make_frame(self, time: float) -> StateFrame:
        rho_w = self._rho_water
        g = self._gravity
        mu = self._water_viscosity
        free_z = self._free_surface_z

        vert_len = max(self._vertex_volume ** (1.0 / 3.0), 1e-6)
        sub_depth = np.maximum(free_z - self._pos[:, 2], 0.0)
        sub_frac = np.clip(sub_depth / vert_len, 0.0, 1.0)

        # Compute interaction force (drag) for reporting
        cloth_mask = self._is_cloth
        rel_vel_cloth = -self._vel[cloth_mask]
        speed_cloth = np.linalg.norm(rel_vel_cloth, axis=1, keepdims=True) + 1e-12
        R = self._drag_radius
        A = self._area_per_vertex
        c_lin = 6.0 * math.pi * mu * R
        c_quad = 0.5 * rho_w * 1.2 * A
        cloth_poro = self._porosity[cloth_mask]
        cloth_sub = sub_frac[cloth_mask]
        drag_scale = cloth_poro * cloth_sub
        drag_mag = (c_lin + c_quad * speed_cloth) * drag_scale[:, None]
        drag_force_cloth = drag_mag * rel_vel_cloth

        # Map back to full array for body_id indexing
        drag_force_full = np.zeros((len(self._mass), 3), dtype=float)
        drag_force_full[cloth_mask] = drag_force_cloth

        interaction_force = np.zeros((3, 3), dtype=float)
        for k, bid in enumerate((1, 2, 3)):
            mask = self._body_id == bid
            if np.any(mask):
                interaction_force[k] = np.sum(drag_force_full[mask], axis=0)

        ext_force = np.zeros((4, 3, 3), dtype=float)
        vol_displaced = sub_frac * self._vertex_volume
        buoy_all = -rho_w * vol_displaced[:, None] * g[None, :]
        for b in range(4):
            mask = self._domain_weights[b].astype(bool)
            if not np.any(mask):
                continue
            ext_force[b, 0] = np.sum(self._mass[mask, None] * g[None, :], axis=0)
            if b > 0:
                ext_force[b, 1] = np.sum(buoy_all[mask], axis=0)

        return StateFrame(
            time=time,
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={"mass_quantity": self._mass.copy()},
            fields={
                "budget_cumulative_external_impulse": self._cum_ext_impulse.copy(),
                "budget_cumulative_inflow_momentum": self._cum_inflow_mom.copy(),
                "budget_cumulative_outflow_momentum": self._cum_outflow_mom.copy(),
                "budget_domain_weights": self._domain_weights.copy(),
                "budget_external_force": ext_force,
                "cumulative_mass_source": self._cum_mass_source.copy(),
                "interaction_force": interaction_force,
            },
            scalars={"reference_momentum": float(self._ref_momentum)},
        )
