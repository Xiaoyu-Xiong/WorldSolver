"""Wettable sand + water coupled particle solver.

Physical model (implemented directly with NumPy/SciPy, no external engine):

* Sand pile: soft-sphere DEM granular medium.  Grains interact through a
  normal spring-dashpot, a Coulomb-limited tangential friction and a
  wetness dependent capillary cohesion.  Sand never changes phase and never
  dissolves, so the solid mass of every grain is invariant.
* Water: coarse-grained dissipative-particle liquid.  Water particles carry
  the real water mass and interact through a soft conservative pair force
  (short range repulsion, longer range cohesion / surface tension), a
  dissipative drag and gravity.  Gravity drives the fall, the impact and the
  run-off; the cohesion keeps the pouring stream and the run-off puddle
  coherent while the repulsion keeps the liquid nearly incompressible under
  its own weight.
* Coupling: penalty contact between water and grains (equal and opposite,
  momentum conserving) and a mass-conserving capillary wetting transfer of
  water mass from the liquid into the contacting grains.  Absorbed water is
  further redistributed through grain contacts (capillary wicking), so a
  wetting front penetrates the pile starting exactly at the stream impact
  points and the pile surface deforms under the impact and the loss of
  friction of the saturated grains.
* Boundaries: the frozen tray floor and the three low tray walls are
  enforced as rigid, frictional non-penetration constraints.

Each material keeps its own mass: ``attributes.mass_quantity`` reports the
solid sand mass and the water mass (absorbed water plus free liquid) per
sample and the column sums are conserved, so ``cumulative_mass_source`` is
identically zero (no prescribed external source is introduced).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# --------------------------------------------------------------------------
# integration / pair search
# --------------------------------------------------------------------------
DT = 1.0e-3
RMAX = 0.10
GRAVITY_Z = -9.81

# --------------------------------------------------------------------------
# sand grains (soft-sphere DEM)
# --------------------------------------------------------------------------
R_S = 3.73e-2                 # grain contact radius (pile lattice pitch 0.075)
RC_SS = 2.0 * R_S             # grain/grain contact distance
KN = 60.0                     # normal stiffness                [N/m]
CN = 0.09                     # normal dashpot                  [N s/m]
CT = 5.0e-2                   # tangential dashpot              [N s/m]
MU_DRY = 0.55                 # dry Coulomb friction coefficient
MU_WET_DROP = 0.30            # friction lost at full saturation
K_COH = 3.5e-3                # capillary cohesion force scale  [N]
COH_DRY = 1.0                 # dry cohesion factor
COH_WET = 2.2                 # additional cohesion at saturation
R_COH = RC_SS + 1.5e-2        # cohesion interaction range      [m]
R_WICK = RC_SS + 1.2e-2       # capillary wicking range         [m]
K_WICK = 1.2                  # wicking rate                    [1/s]

# --------------------------------------------------------------------------
# water (coarse-grained dissipative particle liquid)
# --------------------------------------------------------------------------
R_W = 3.0e-2                  # water radius against grains      [m]
D0_W = 4.5e-2                 # zero-force water/water separation[m]
RC_WW = 9.5e-2                # water/water interaction range    [m]
K_WW = 8.0                    # water/water stiffness            [N/m]
K_CWW = 1.0e-3                # water/water cohesion             [N/m]
C_WW = 1.0e-2                 # water/water dissipation          [N s/m]
CT_WW = 5.0e-3                # water/water tangential drag      [N s/m]

# --------------------------------------------------------------------------
# sand/water coupling
# --------------------------------------------------------------------------
RC_SW = R_S + R_W             # water/grain contact distance
K_SW = 60.0                   # water/grain penalty stiffness    [N/m]
C_SW = 5.0e-2                 # water/grain dashpot              [N s/m]
CT_SW = 2.0e-2                # water/grain tangential drag      [N s/m]
MU_SW = 0.35                  # water/grain tangential limit

# --------------------------------------------------------------------------
# capillary wetting
# --------------------------------------------------------------------------
R_WET = 8.0e-2                # liquid/grain wetting range       [m]
WET_RAMP_LEN = 3.0e-2         # wetting rate ramp length         [m]
K_WET = 4.0e-2                # saturation per second at contact [1/s]
W_MAX = 0.05                  # saturated wettable sand concentration

# --------------------------------------------------------------------------
# frozen tray geometry (scene_geometry.yaml)
# --------------------------------------------------------------------------
WALL_LEFT_X = -1.47
WALL_RIGHT_X = 1.47
WALL_BACK_Y = 1.07
WALL_TOP_Z = 0.28
WALL_SPAN = 1.20
MU_WALL = 0.35

# --------------------------------------------------------------------------
# diagnostic grid
# --------------------------------------------------------------------------
GS = 0.155
GRID_MIN = np.array([-1.55, -1.16, 0.0])
GRID_MAX = np.array([1.55, 1.16, 2.48])

CONTACT_LENGTH_SCALE = float(RC_SS)


class BenchmarkSolver:
    """Coupled wettable-sand / water particle solver."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        pos = np.asarray(frame.position, dtype=np.float64)
        vel = np.asarray(frame.velocity, dtype=np.float64)
        body = np.asarray(frame.body_id, dtype=np.int64)
        mass = np.asarray(frame.mass, dtype=np.float64)

        self.n = int(pos.shape[0])
        self.pos = np.array(pos, copy=True)
        self.vel = np.array(vel, copy=True)
        self.body_id = body.astype(np.int32)
        self.is_sand = body == 1
        self.is_water = body == 2

        self.m_solid = np.where(self.is_sand, mass, 0.0)
        self.m_water = np.where(self.is_water, mass, 0.0)
        self.m_water0 = np.where(self.is_water, mass, 1.0)
        self.m_abs = np.zeros(self.n, dtype=np.float64)
        self.wetness = np.where(self.is_sand, 0.0, 1.0)
        self.radius = np.where(self.is_sand, R_S, R_W)
        self.render_radius = np.where(self.is_sand, 3.6e-2, 5.0e-2)

        # diagnostic grid -------------------------------------------------
        n_xyz = np.maximum(np.floor((GRID_MAX - GRID_MIN) / GS).astype(np.int64), 2)
        self.grid_n = n_xyz
        self.grid_center = [
            GRID_MIN[k] + (np.arange(n_xyz[k], dtype=np.float64) + 0.5) * GS
            for k in range(3)
        ]
        self.phi = np.zeros(tuple(n_xyz), dtype=np.float64)
        self._wet_accum = np.zeros(int(n_xyz.prod()), dtype=np.float64)
        self._source_rate = np.zeros(tuple(n_xyz), dtype=np.float64)
        self._vel_grid = np.zeros(tuple(n_xyz) + (3,), dtype=np.float64)
        self._active_mask = np.ones(tuple(n_xyz), dtype=np.float64)
        self._interval = 0.0

        # bookkeeping -----------------------------------------------------
        self._pp_force = np.zeros((self.n, 3), dtype=np.float64)
        self._wall_force = np.zeros((self.n, 3), dtype=np.float64)
        self._pi = np.zeros(0, dtype=np.int64)
        self._pj = np.zeros(0, dtype=np.int64)
        self._pd = np.zeros(0, dtype=np.float64)
        self._cumulative_source = np.zeros(2, dtype=np.float64)

        self.sim_time = 0.0
        self.steps = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target < self.sim_time:
            target = self.sim_time
        guard = 0
        while self.sim_time < target - 1e-12 and guard < 500000:
            h = min(DT, target - self.sim_time)
            self._step(h)
            self.sim_time += h
            self._interval += h
            self.steps += 1
            guard += 1
        self.sim_time = target
        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": int(self.steps),
            "time": float(self.sim_time),
            "sand_mass": float(self.m_solid.sum()),
            "free_water_mass": float(self.m_water.sum()),
            "absorbed_water_mass": float(self.m_abs.sum()),
            "max_wetness": (
                float(self.wetness[self.is_sand].max()) if np.any(self.is_sand) else 0.0
            ),
            "grid_shape": [int(v) for v in self.grid_n],
        }

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _particle_mass(self) -> np.ndarray:
        return self.m_solid + self.m_abs + self.m_water

    def _cell_index(self, pos: np.ndarray) -> np.ndarray:
        ijk = np.floor((pos - GRID_MIN) / GS).astype(np.int64)
        np.clip(ijk, 0, self.grid_n - 1, out=ijk)
        return ijk

    def _flat_index(self, ijk: np.ndarray) -> np.ndarray:
        ny, nz = int(self.grid_n[1]), int(self.grid_n[2])
        return (ijk[:, 0] * ny + ijk[:, 1]) * nz + ijk[:, 2]

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _step(self, h: float) -> None:
        self._pair_forces()
        mass = np.maximum(self._particle_mass(), 1.0e-12)
        acc = self._pp_force / mass[:, None]
        acc[:, 2] += GRAVITY_Z
        self.vel += acc * h
        self.pos += h * self.vel
        self._wetting(h)
        self._resolve_walls(h)

    # ------------------------------------------------------------------
    # paired interaction forces
    # ------------------------------------------------------------------
    def _pair_forces(self) -> None:
        pos, vel = self.pos, self.vel
        f = np.zeros((self.n, 3), dtype=np.float64)
        pairs = cKDTree(pos).query_pairs(RMAX, output_type="ndarray")
        if pairs.size == 0:
            self._pp_force = f
            self._pi = np.zeros(0, dtype=np.int64)
            self._pj = np.zeros(0, dtype=np.int64)
            self._pd = np.zeros(0, dtype=np.float64)
            return
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        i = pairs[:, 0]
        j = pairs[:, 1]
        dv = pos[j] - pos[i]
        d = np.linalg.norm(dv, axis=1)
        keep = d > 1e-12
        i, j = i[keep], j[keep]
        dv = dv[keep]
        d = d[keep]
        nvec = dv / d[:, None]
        vrel = vel[j] - vel[i]
        vn = np.einsum("ij,ij->i", vrel, nvec)
        vt = vrel - vn[:, None] * nvec
        si = self.is_sand[i]
        sj = self.is_sand[j]
        wet = self.wetness

        # ---- grain / grain ---------------------------------------------
        sel = si & sj
        if np.any(sel):
            ii, jj = i[sel], j[sel]
            dd, nn = d[sel], nvec[sel]
            vvn, vvt = vn[sel], vt[sel]
            overlap = np.minimum(RC_SS - dd, 1.0e-2)
            fnormal = np.maximum(
                np.where(overlap > 0.0, KN * overlap, 0.0) - CN * vvn, 0.0
            )
            wbar = np.clip(0.5 * (wet[ii] + wet[jj]) / W_MAX, 0.0, 1.0)
            ramp = np.clip((R_COH - dd) / (R_COH - RC_SS), 0.0, 1.0)
            cohesion = K_COH * (COH_DRY + COH_WET * wbar) * ramp
            mu = MU_DRY - MU_WET_DROP * wbar
            ft = -CT * vvt
            lim = mu * fnormal
            ftn = np.linalg.norm(ft, axis=1)
            scale = np.where(ftn > lim, lim / np.maximum(ftn, 1e-30), 1.0)
            ft = ft * scale[:, None]
            fi = (fnormal - cohesion)[:, None] * nn + ft
            np.add.at(f, ii, fi)
            np.add.at(f, jj, -fi)

        # ---- grain / water ---------------------------------------------
        sel = si != sj
        if np.any(sel):
            ii, jj = i[sel], j[sel]
            nn = nvec[sel]
            vvn, vvt = vn[sel], vt[sel]
            overlap = np.minimum(RC_SW - d[sel], 1.5e-2)
            fnormal = np.maximum(K_SW * np.maximum(overlap, 0.0) - C_SW * vvn, 0.0)
            ft = -CT_SW * vvt
            lim = MU_SW * fnormal
            ftn = np.linalg.norm(ft, axis=1)
            scale = np.where(ftn > lim, lim / np.maximum(ftn, 1e-30), 1.0)
            ft = ft * scale[:, None]
            fi = fnormal[:, None] * nn + ft
            np.add.at(f, ii, fi)
            np.add.at(f, jj, -fi)

        # ---- water / water ---------------------------------------------
        sel = (~si) & (~sj)
        if np.any(sel):
            ii, jj = i[sel], j[sel]
            dd, nn = d[sel], nvec[sel]
            vvn, vvt = vn[sel], vt[sel]
            rep = K_WW * np.maximum(D0_W - dd, 0.0)
            beyond = np.clip((dd - D0_W) / (RC_WW - D0_W), 0.0, 1.0)
            coh = K_CWW * (1.0 - beyond)
            fnormal = rep - coh - C_WW * vvn
            ft = -CT_WW * vvt
            fi = fnormal[:, None] * nn + ft
            np.add.at(f, ii, fi)
            np.add.at(f, jj, -fi)

        self._pp_force = f
        self._pi, self._pj, self._pd = i, j, d

    # ------------------------------------------------------------------
    # capillary wetting: absorption and wicking
    # ------------------------------------------------------------------
    def _wetting(self, h: float) -> None:
        if self._pi.size == 0:
            return
        i, j, d = self._pi, self._pj, self._pd
        si = self.is_sand[i]
        sj = self.is_sand[j]

        # ---- liquid -> grain absorption ---------------------------------
        sel = (si != sj) & (d < R_WET)
        if np.any(sel):
            ii, jj = i[sel], j[sel]
            dd = d[sel]
            ii_sand = si[sel]
            s_idx = np.where(ii_sand, ii, jj)
            w_idx = np.where(ii_sand, jj, ii)
            ramp = np.clip((R_WET - dd) / WET_RAMP_LEN, 0.0, 1.0)
            demand = K_WET * h * self.m_solid[s_idx] * ramp
            cap = np.maximum(W_MAX * self.m_solid[s_idx] - self.m_abs[s_idx], 0.0)
            tot = np.zeros(self.n, dtype=np.float64)
            np.add.at(tot, s_idx, demand)
            tot_p = tot[s_idx]
            scale_p = np.where(
                tot_p > 0.0, np.minimum(1.0, cap / np.maximum(tot_p, 1e-30)), 0.0
            )
            d1 = demand * scale_p
            need = np.zeros(self.n, dtype=np.float64)
            np.add.at(need, w_idx, d1)
            avail = np.maximum(self.m_water - 0.3 * self.m_water0, 0.0)
            need_p = need[w_idx]
            scale_w = np.where(
                need_p > 0.0, np.minimum(1.0, avail / np.maximum(need_p, 1e-30)), 0.0
            )
            d2 = d1 * scale_w
            np.add.at(self.m_abs, s_idx, d2)
            np.add.at(self.m_water, w_idx, -d2)
            flat = self._flat_index(self._cell_index(self.pos[s_idx]))
            np.add.at(
                self._wet_accum, flat, d2 / np.maximum(self.m_solid[s_idx], 1e-30)
            )

        # ---- capillary wicking through grain contacts -------------------
        sel = si & sj & (d < R_WICK)
        if np.any(sel):
            ii, jj = i[sel], j[sel]
            dd = d[sel]
            ramp = np.clip((R_WICK - dd) / (R_WICK - RC_SS), 0.0, 1.0)
            coef = np.minimum(K_WICK * h * ramp, 0.25)
            flux = coef * (self.m_abs[ii] - self.m_abs[jj])
            room_i = np.maximum(W_MAX * self.m_solid[ii] - self.m_abs[ii], 0.0)
            room_j = np.maximum(W_MAX * self.m_solid[jj] - self.m_abs[jj], 0.0)
            gain_j = flux > 0.0
            room_gain = np.where(gain_j, room_j, room_i)
            gain_idx = np.where(gain_j, jj, ii)
            tot = np.zeros(self.n, dtype=np.float64)
            np.add.at(tot, gain_idx, np.abs(flux))
            tot_g = tot[gain_idx]
            scale_g = np.where(
                tot_g > 0.0, np.minimum(1.0, room_gain / np.maximum(tot_g, 1e-30)), 0.0
            )
            flux = flux * scale_g
            np.add.at(self.m_abs, ii, -flux)
            np.add.at(self.m_abs, jj, flux)

        if np.any(self.is_sand):
            sat = self.m_abs[self.is_sand] / np.maximum(self.m_solid[self.is_sand], 1e-30)
            self.wetness[self.is_sand] = np.clip(sat, 0.0, W_MAX)

    # ------------------------------------------------------------------
    # rigid tray boundaries (non-penetration + Coulomb friction)
    # ------------------------------------------------------------------
    def _resolve_walls(self, h: float) -> None:
        pos = self.pos
        vel = self.vel
        r = self.radius
        mass = self._particle_mass()
        self._wall_force[:] = 0.0

        def impulse(mask_idx, normal, pen):
            if mask_idx.size == 0:
                return
            pos[mask_idx] += pen[:, None] * normal[None, :]
            vn = vel[mask_idx] @ normal
            vmag = np.linalg.norm(vel[mask_idx], axis=1)
            vt_mag = np.sqrt(np.maximum(vmag * vmag - vn * vn, 0.0))
            inb = vn < 0.0
            if not np.any(inb):
                return
            dvn = np.where(inb, -vn, 0.0)
            fr = np.where(
                vt_mag > 1e-12,
                np.minimum(1.0, MU_WALL * dvn / np.maximum(vt_mag, 1e-12)),
                0.0,
            )
            tangential = vel[mask_idx] - vn[:, None] * normal[None, :]
            vel[mask_idx] -= fr[:, None] * tangential
            vel[mask_idx] -= np.where(inb, vn, 0.0)[:, None] * normal[None, :]
            self._wall_force[mask_idx] += (
                (mass[mask_idx] * dvn / h)[:, None] * normal[None, :]
            )

        # floor (top face z = 0)
        pen = r - pos[:, 2]
        impulse(np.flatnonzero(pen > 0.0), np.array([0.0, 0.0, 1.0]), pen[pen > 0.0])

        # left wall inner face x = -1.47
        pen = (WALL_LEFT_X + r) - pos[:, 0]
        keep = (pen > 0.0) & (np.abs(pos[:, 1]) < WALL_SPAN) & (pos[:, 2] < WALL_TOP_Z + r)
        impulse(np.flatnonzero(keep), np.array([1.0, 0.0, 0.0]), pen[keep])

        # right wall inner face x = +1.47
        pen = pos[:, 0] - (WALL_RIGHT_X - r)
        keep = (pen > 0.0) & (np.abs(pos[:, 1]) < WALL_SPAN) & (pos[:, 2] < WALL_TOP_Z + r)
        impulse(np.flatnonzero(keep), np.array([-1.0, 0.0, 0.0]), pen[keep])

        # back wall inner face y = +1.07
        pen = pos[:, 1] - (WALL_BACK_Y - r)
        keep = (pen > 0.0) & (np.abs(pos[:, 0]) < WALL_SPAN) & (pos[:, 2] < WALL_TOP_Z + r)
        impulse(np.flatnonzero(keep), np.array([0.0, -1.0, 0.0]), pen[keep])

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def _update_fields_for_sample(self, dt_interval: float) -> None:
        pos = self.pos
        flat = self._flat_index(self._cell_index(pos))
        ncells = int(self.grid_n.prod())
        mass = self._particle_mass()
        num = np.zeros((ncells, 3), dtype=np.float64)
        den = np.zeros(ncells, dtype=np.float64)
        for k in range(3):
            np.add.at(num[:, k], flat, mass * self.vel[:, k])
        np.add.at(den, flat, mass)
        vgrid = np.zeros((ncells, 3), dtype=np.float64)
        nz = den > 0.0
        vgrid[nz] = num[nz] / den[nz, None]
        self._vel_grid = vgrid.reshape(tuple(self.grid_n) + (3,))

        cx = self.grid_center[0][:, None, None]
        cy = self.grid_center[1][None, :, None]
        cz = self.grid_center[2][None, None, :]

        # transported scalar: semi-Lagrangian advection + diffusion + source
        if dt_interval > 0.0:
            u = self._vel_grid
            coords = [
                (cx - dt_interval * u[..., 0] - GRID_MIN[0]) / GS - 0.5,
                (cy - dt_interval * u[..., 1] - GRID_MIN[1]) / GS - 0.5,
                (cz - dt_interval * u[..., 2] - GRID_MIN[2]) / GS - 0.5,
            ]
            adv = map_coordinates(self.phi, coords, order=1, mode="nearest")
            padded = np.pad(adv, 1, mode="edge")
            lap = (
                padded[:-2, 1:-1, 1:-1]
                + padded[2:, 1:-1, 1:-1]
                + padded[1:-1, :-2, 1:-1]
                + padded[1:-1, 2:, 1:-1]
                + padded[1:-1, 1:-1, :-2]
                + padded[1:-1, 1:-1, 2:]
                - 6.0 * adv
            )
            alpha = 1.0e-4
            self._source_rate = self._wet_accum.reshape(tuple(self.grid_n)) / dt_interval
            self.phi = np.clip(
                adv + alpha * dt_interval * lap / (GS * GS)
                + self._source_rate * dt_interval,
                0.0,
                W_MAX,
            )
        self._wet_accum[:] = 0.0

        # active mask: coupled tray / pouring volume
        active = (
            (np.abs(cx) <= 1.50)
            & (cy <= 1.10)
            & (cy >= -1.10)
            & (cz <= 2.40)
            & (cz >= -0.02)
        )
        self._active_mask = active.astype(np.float64)

    def _contact_samples(self) -> np.ndarray:
        pairs = cKDTree(self.pos).query_pairs(RMAX, output_type="ndarray")
        if pairs.size == 0:
            return np.zeros(0, dtype=np.float64)
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.linalg.norm(self.pos[j] - self.pos[i], axis=1)
        si = self.is_sand[i]
        sj = self.is_sand[j]
        both_s = si & sj
        cross = si != sj
        cutoff = np.where(both_s, RC_SS, np.where(cross, RC_SW, RC_WW))
        sel = d < cutoff
        if not np.any(sel):
            return np.zeros(0, dtype=np.float64)
        return d[sel] - cutoff[sel]

    # ------------------------------------------------------------------
    # frame assembly
    # ------------------------------------------------------------------
    def _make_frame(self) -> StateFrame:
        self._update_fields_for_sample(self._interval)

        n = self.n
        mq = np.zeros((n, 2), dtype=np.float64)
        mq[:, 0] = self.m_solid
        mq[:, 1] = self.m_abs + self.m_water

        sat = np.zeros(n, dtype=np.float64)
        if np.any(self.is_sand):
            sat[self.is_sand] = np.clip(
                self.m_abs[self.is_sand] / np.maximum(self.m_solid[self.is_sand], 1e-30),
                0.0,
                W_MAX,
            )
        water_conc = np.where(self.is_sand, sat / W_MAX, 1.0)

        contact = self._pp_force + self._wall_force
        f_sand = (
            contact[self.is_sand].sum(axis=0) if np.any(self.is_sand) else np.zeros(3)
        )
        f_water = (
            contact[self.is_water].sum(axis=0) if np.any(self.is_water) else np.zeros(3)
        )

        attributes = {
            "material_id": self.body_id.astype(np.float64),
            "solid_phase_fraction": self.is_sand.astype(np.float64),
            "water_concentration": water_conc,
            "wetness": self.wetness.copy(),
            "particle_radius": self.render_radius.copy(),
            "mass_quantity": mq,
        }
        fields = {
            "active_mask": self._active_mask,
            "contact_signed_distance": self._contact_samples(),
            "cumulative_mass_source": self._cumulative_source.copy(),
            "interaction_force": np.vstack((f_sand, f_water)),
            "scalar_source": self._source_rate,
            "transported_scalar": self.phi,
            "velocity_field": self._vel_grid,
        }
        scalars = {
            "contact_length_scale": float(CONTACT_LENGTH_SCALE),
            "grid_spacing": float(GS),
        }
        self._interval = 0.0
        return StateFrame(
            float(self.sim_time),
            self.pos.copy(),
            self.vel.copy(),
            mass=self._particle_mass().copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
