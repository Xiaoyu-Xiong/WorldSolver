"""Simplified coupled sand--water particle solver for the sand-dam breach scene.

The solver uses a particle-based two-phase model:

* Water is evolved with weakly-compressible SPH pressure plus a small
  artificial viscosity.
* Sand is evolved as a cohesive granular material: Hookean repulsion for
  overlaps, velocity damping, and a wetness-dependent cohesive attraction.
* Sand and water exchange momentum through SPH pressure and a drag term.
* Sand saturation increases when in contact with water, which reduces the
  cohesive strength and allows the dam to fail.
* Boundary walls are modelled with penalty forces and position clamping.
* A small set of supply-water particles are continuously recycled from the
  supply chute to keep water reaching the dam.

The implementation is fully deterministic and uses only NumPy/SciPy.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # --- time stepping ---------------------------------------------------
        self.dt = 0.001
        self.current_time = 0.0

        # --- initial state ---------------------------------------------------
        frame0 = scene.initial_state()
        self.pos = np.asarray(frame0.position, dtype=float).copy()
        self.vel = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id).copy()
        self.attributes = {
            k: np.asarray(v).copy() for k, v in (frame0.attributes or {}).items()
        }
        self.volume = np.asarray(self.attributes["particle_volume"], dtype=float).copy()
        self.porosity = np.asarray(self.attributes["porosity"], dtype=float).copy()
        self.saturation = np.asarray(self.attributes["sand_saturation"], dtype=float).copy()
        self.material_id = np.asarray(self.attributes["material_id"], dtype=int).copy()

        # phase: 0 = sand, 1 = water
        self.SAND_ID = 1
        self.WATER_ID = 2
        self.phase = np.where(self.body_id == self.WATER_ID, 1, 0)
        # keep water fully saturated diagnostically
        self.saturation[self.phase == 1] = 1.0

        # identify supply particles: water particles that start in the chute
        is_water = self.phase == 1
        self.is_supply = is_water & (self.pos[:, 2] > 1.4)
        self.supply_age = np.zeros(len(self.pos), dtype=float)

        self.g = np.asarray([0.0, 0.0, -9.81], dtype=float)

        # --- physical parameters ---------------------------------------------
        self.h = 0.10
        self.cutoff = 2.0 * self.h
        self.k_neighbors = 80

        self.rho0_water = 1000.0
        self.rho0_sand = 1650.0
        self.B_water = 1e5
        self.gamma_eos = 2.0

        self.r_eq = 0.085
        self.k_sand = 1e5
        self.c_damp = 20.0

        self.h_coh = 0.18
        self.c0 = 2e4
        self.cohesion_area = self.h * self.h

        self.gamma_drag = 5.0
        self.eta_visc = 0.1
        self.alpha_sat = 5.0

        # boundary penalty
        self.k_wall = 1e5
        self.c_wall = 60.0

        # kernel normalisation for the 3D cubic spline
        self.kernel_norm = 1.0 / (np.pi * self.h ** 3)

        # --- budget / trajectory fields --------------------------------------
        self.n = len(self.pos)
        self.n_domains = 3  # sand, water, all
        self.n_terms = 3  # gravity, boundary, source
        self.n_materials = 2  # sand, water

        self.domain_weights = np.zeros((self.n_domains, self.n), dtype=float)
        sand_mask = self.phase == 0
        water_mask = self.phase == 1
        self.domain_weights[0, sand_mask] = 1.0
        self.domain_weights[1, water_mask] = 1.0
        self.domain_weights[2, :] = 1.0

        self.cumulative_external_impulse = np.zeros((self.n_domains, 3), dtype=float)
        self.cumulative_inflow_momentum = np.zeros((self.n_domains, 3), dtype=float)
        self.cumulative_outflow_momentum = np.zeros((self.n_domains, 3), dtype=float)
        self.cumulative_mass_source = np.zeros(self.n_materials, dtype=float)
        self.budget_external_force = np.zeros((self.n_domains, self.n_terms, 3), dtype=float)
        self.interaction_force = np.zeros((self.n_materials, 3), dtype=float)

        total_mass = float(self.mass.sum())
        # characteristic momentum scale: total mass * sqrt(g * dam_height)
        self.reference_momentum = total_mass * np.sqrt(9.81 * 1.75)

        # material mass quantity [N, 2]
        self.mass_quantity = np.zeros((self.n, self.n_materials), dtype=float)
        self.mass_quantity[sand_mask, 0] = self.mass[sand_mask]
        self.mass_quantity[water_mask, 1] = self.mass[water_mask]

        self._frame0_time = float(frame0.time)

    # -----------------------------------------------------------------------
    def _kernel(self, r: np.ndarray):
        """3D cubic spline kernel and its radial derivative."""
        q = r / self.h
        f = np.zeros_like(q)
        fp = np.zeros_like(q)
        m1 = q < 1.0
        f[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
        fp[m1] = -3.0 * q[m1] + 2.25 * q[m1] ** 2
        m2 = (q >= 1.0) & (q < 2.0)
        f[m2] = 0.25 * (2.0 - q[m2]) ** 3
        fp[m2] = -0.75 * (2.0 - q[m2]) ** 2
        W = self.kernel_norm * f
        dWdr = self.kernel_norm / self.h * fp
        return W, dWdr

    def _boundary_forces(self) -> np.ndarray:
        """Penalty forces from the flume walls and floor."""
        Fb = np.zeros_like(self.pos)
        # floor (z = 0)
        pen = -self.pos[:, 2]
        mask = pen > 0.0
        Fb[mask, 2] += self.k_wall * pen[mask] - self.c_wall * self.vel[mask, 2]
        # left reservoir wall (x = -4.07)
        pen = -4.07 - self.pos[:, 0]
        mask = pen > 0.0
        Fb[mask, 0] += self.k_wall * pen[mask] - self.c_wall * self.vel[mask, 0]
        # right retaining wall (x = 4.43)
        pen = self.pos[:, 0] - 4.43
        mask = pen > 0.0
        Fb[mask, 0] -= self.k_wall * pen[mask] + self.c_wall * self.vel[mask, 0]
        # back side wall (y = 0.95)
        pen = self.pos[:, 1] - 0.95
        mask = pen > 0.0
        Fb[mask, 1] -= self.k_wall * pen[mask] + self.c_wall * self.vel[mask, 1]
        # front side wall (y = -0.95)
        pen = -0.95 - self.pos[:, 1]
        mask = pen > 0.0
        Fb[mask, 1] += self.k_wall * pen[mask] - self.c_wall * self.vel[mask, 1]
        return Fb

    def _clamp_positions(self) -> None:
        self.pos[:, 2] = np.maximum(self.pos[:, 2], 0.0)
        self.pos[:, 0] = np.clip(self.pos[:, 0], -4.07 + 1e-6, 4.43 - 1e-6)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -0.95 + 1e-6, 0.95 - 1e-6)

    def _reset_supply(self) -> None:
        """Recycle supply particles back to the chute."""
        if not np.any(self.is_supply):
            return
        # particles that have travelled through the domain or hit the floor
        mask = self.is_supply & ((self.pos[:, 0] > 2.0) | (self.pos[:, 2] < 0.3))
        if not np.any(mask):
            return
        idx = np.nonzero(mask)[0]
        n_reset = idx.size
        # local chute coordinates: u along local x, w along local z, v along y
        u = self.rng.uniform(-0.55, 0.55, size=n_reset)
        v = self.rng.uniform(-0.65, 0.65, size=n_reset)
        w = self.rng.uniform(-0.06, 0.06, size=n_reset)
        sin_t = -0.1305262
        cos_t = 0.9914449
        # center of chute
        cx, cy, cz = -3.05, 0.0, 2.55
        x_rel = u * cos_t + w * sin_t
        z_rel = -u * sin_t + w * cos_t
        new_pos = np.column_stack([cx + x_rel, cy + v, cz + z_rel])
        # prescribed inflow velocity, slightly along +x and downward
        new_vel = np.tile(np.array([0.6, 0.0, -2.5], dtype=float), (n_reset, 1))
        # add small random perturbation
        new_vel += self.rng.normal(0.0, 0.15, size=(n_reset, 3))
        new_vel[:, 2] = np.minimum(new_vel[:, 2], -1.0)

        # momentum accounting for the source term
        old_mom = self.mass[idx, None] * self.vel[idx]
        new_mom = self.mass[idx, None] * new_vel
        delta = new_mom - old_mom
        self.cumulative_inflow_momentum[1] += delta.sum(axis=0)
        self.cumulative_inflow_momentum[2] += delta.sum(axis=0)

        # outflow budget: momentum of particles being reset
        outflow = np.sum(old_mom, axis=0)
        self.cumulative_outflow_momentum[1] += outflow
        self.cumulative_outflow_momentum[2] += outflow

        self.pos[idx] = new_pos
        self.vel[idx] = new_vel

    def _step(self, dt: float) -> None:
        n = self.n
        tree = cKDTree(self.pos)
        dist, idx = tree.query(self.pos, k=self.k_neighbors, distance_upper_bound=self.cutoff)

        # valid neighbour pairs (exclude self and beyond cutoff)
        valid = (dist < self.cutoff) & (dist > 1e-12)
        row = np.arange(n)[:, None]
        i_full = np.broadcast_to(row, (n, self.k_neighbors))[valid]
        j_full = idx[valid]
        r_full = dist[valid]

        # kernel values for all ordered pairs
        W_full, dWdr_full = self._kernel(r_full)

        # --- density (water only) -------------------------------------------
        phase_i = self.phase[i_full]
        phase_j = self.phase[j_full]
        mass_j = self.mass[j_full]
        ww_pair = (phase_i == 1) & (phase_j == 1)
        rho = np.zeros(n, dtype=float)
        np.add.at(rho, i_full[ww_pair], mass_j[ww_pair] * W_full[ww_pair])

        # pressure (water only)
        P = np.zeros(n, dtype=float)
        water_mask = self.phase == 1
        rho_w = rho[water_mask]
        ratio = rho_w / self.rho0_water
        P_w = self.B_water * np.maximum(ratio ** 2 - 1.0, 0.0)
        P[water_mask] = P_w

        # --- unique pair force pass ------------------------------------------
        unique = i_full < j_full
        iu = i_full[unique]
        ju = j_full[unique]
        ru = r_full[unique]
        Wu = W_full[unique]
        dWu = dWdr_full[unique]

        r_vec = self.pos[ju] - self.pos[iu]
        n_vec = r_vec / ru[:, None]
        v_rel = self.vel[iu] - self.vel[ju]
        m_i = self.mass[iu]
        m_j = self.mass[ju]
        rho_i = np.maximum(rho[iu], 1e-6)
        rho_j = np.maximum(rho[ju], 1e-6)
        phase_iu = self.phase[iu]
        phase_ju = self.phase[ju]

        # start with SPH pressure term (zero for sand-sand)
        P_i = P[iu]
        P_j = P[ju]
        term = m_i * m_j * (P_i / (rho_i ** 2) + P_j / (rho_j ** 2))
        f_on_i = term[:, None] * dWu[:, None] * n_vec

        # sand-sand repulsion, damping and cohesion
        ss = (phase_iu == 0) & (phase_ju == 0)
        if np.any(ss):
            mask_rep = ss & (ru < self.r_eq)
            # repulsive spring
            f_rep = -self.k_sand * (self.r_eq - ru)[:, None] * n_vec
            # normal damping
            v_n = np.sum(v_rel * n_vec, axis=1)
            f_damp = -self.c_damp * v_n[:, None] * n_vec
            f_on_i += (f_rep + f_damp) * mask_rep[:, None]

            mask_coh = ss & (ru < self.h_coh)
            sat_i = self.saturation[iu]
            sat_j = self.saturation[ju]
            c_eff = self.c0 * (1.0 - sat_i) * (1.0 - sat_j) * self.cohesion_area
            f_coh = (c_eff * (self.h_coh - ru) / self.h_coh)[:, None] * n_vec
            f_on_i += f_coh * mask_coh[:, None]

        # sand-water drag
        sw = (phase_iu != phase_ju)
        if np.any(sw):
            f_drag = -self.gamma_drag * v_rel * Wu[:, None]
            f_on_i += f_drag * sw[:, None]

        # water-water artificial viscosity
        ww = (phase_iu == 1) & (phase_ju == 1)
        if np.any(ww):
            f_visc = -self.eta_visc * v_rel * Wu[:, None]
            f_on_i += f_visc * ww[:, None]

        # accumulate pairwise forces onto particles
        F = np.zeros((n, 3), dtype=float)
        np.add.at(F, iu, f_on_i)
        np.add.at(F, ju, -f_on_i)

        # --- interaction-force accounting -------------------------------------
        F_sand_from_water = np.zeros(3, dtype=float)
        F_water_from_sand = np.zeros(3, dtype=float)
        if np.any(sw):
            sign_sand = np.where(phase_iu == 0, 1.0, -1.0)
            sign_water = np.where(phase_iu == 1, 1.0, -1.0)
            F_sand_from_water = np.sum(f_on_i[sw] * sign_sand[sw, None], axis=0)
            F_water_from_sand = np.sum(f_on_i[sw] * sign_water[sw, None], axis=0)
        self.interaction_force[0] = F_sand_from_water
        self.interaction_force[1] = F_water_from_sand

        # --- saturation update -----------------------------------------------
        sat_delta = np.zeros(n, dtype=float)
        if np.any(sw):
            i_sand_j_water = sw & (phase_iu == 0)
            i_water_j_sand = sw & (phase_iu == 1)
            vol_j = self.volume[ju]
            vol_i = self.volume[iu]
            np.add.at(sat_delta, iu[i_sand_j_water],
                      self.alpha_sat * vol_j[i_sand_j_water] * Wu[i_sand_j_water])
            np.add.at(sat_delta, ju[i_water_j_sand],
                      self.alpha_sat * vol_i[i_water_j_sand] * Wu[i_water_j_sand])

        # --- external forces (gravity + boundaries) ---------------------------
        F_gravity = self.mass[:, None] * self.g
        F_boundary = self._boundary_forces()
        F += F_gravity + F_boundary

        # update budget fields
        # term 0 = gravity, term 1 = boundary, term 2 = source (zero here)
        self.budget_external_force[:, 0, :] = self.domain_weights @ F_gravity
        self.budget_external_force[:, 1, :] = self.domain_weights @ F_boundary
        self.budget_external_force[:, 2, :] = 0.0
        self.cumulative_external_impulse += (
            self.budget_external_force[:, 0, :] + self.budget_external_force[:, 1, :]
        ) * dt

        # --- integrate -------------------------------------------------------
        self.vel = self.vel + dt * F / self.mass[:, None]
        self.pos = self.pos + dt * self.vel
        self._clamp_positions()

        # saturation only evolves for sand
        self.saturation += dt * sat_delta
        self.saturation = np.clip(self.saturation, 0.0, 1.0)
        self.saturation[water_mask] = 1.0

        # --- recycle supply water --------------------------------------------
        self._reset_supply()

        self.current_time += dt

    # -----------------------------------------------------------------------
    def _build_frame(self, time: float) -> StateFrame:
        # ensure arrays are C-contiguous finite copies
        pos = np.asarray(self.pos, dtype=float)
        vel = np.asarray(self.vel, dtype=float)
        mass = np.asarray(self.mass, dtype=float)
        body_id = np.asarray(self.body_id)
        saturation = np.asarray(self.saturation, dtype=float)
        porosity = np.asarray(self.porosity, dtype=float)
        material_id = np.asarray(self.material_id)
        mass_quantity = np.asarray(self.mass_quantity, dtype=float)

        fields = {
            "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cumulative_inflow_momentum.copy(),
            "budget_cumulative_outflow_momentum": self.cumulative_outflow_momentum.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": self.budget_external_force.copy(),
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "interaction_force": self.interaction_force.copy(),
        }
        scalars = {"reference_momentum": float(self.reference_momentum)}

        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=body_id,
            attributes={
                "material_id": material_id,
                "sand_saturation": saturation,
                "porosity": porosity,
                "particle_volume": self.volume.copy(),
                "mass_quantity": mass_quantity,
            },
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        while self.current_time < time - 0.5 * self.dt:
            dt = min(self.dt, time - self.current_time)
            self._step(dt)
        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": self.dt,
            "particles": self.n,
            "reference_momentum": float(self.reference_momentum),
        }
