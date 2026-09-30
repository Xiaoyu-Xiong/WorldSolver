"""Wettable sand / pouring water solver.

Physical model
--------------
Unified particle method (no external physics engine):

* Sand granular phase: DEM (spring-dashpot) normal contacts with a
  Coulomb-clamped tangential friction force.  Dry sand uses a high friction
  coefficient so the initial conical pile rests; wet sand loses friction
  (lubrication) and gains capillary cohesion, so wetted regions of the pile
  deform and clump.
* Water phase: spring-dashpot repulsion, short-range cohesion (surface
  tension) and tangential viscosity.  Water is poured continuously from the
  two spouts (prescribed mass source) onto the initially dry pile, runs off
  along the pile flanks and is retained by the tray walls.
* Coupling: wetting only, by concentration transfer ("phase evolution
  disabled for sand").  Sand particles carry a wetness scalar in [0, 0.05]
  (0 dry, 0.05 saturated per the paper assumption).  Wetness grows where
  water particles are in contact (the wetting front starts at the stream
  impact points) and diffuses between contacting sand particles with a
  downward percolation bias.  No sand/water mass is exchanged, so both
  material masses are exactly conserved.

Integration: explicit (semi-implicit Euler) with a fixed substep dt = 1e-3 s
(20 substeps per 0.02 s sample) and velocity clamping for robustness; the
pour is emitted on a deterministic lattice.  All neighbor queries use the
scipy cKDTree cell search.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------- constants
G_Z = -9.81
W_MAX = 0.05                     # saturated wetness (paper assumption)

R_SAND, R_WATER = 0.0375, 0.0325  # particle radii [m]
M_SAND, M_WATER = 0.00063, 0.000275  # particle masses [kg] (frozen scaffold)

# DEM parameters (sand-sand / water-water / sand-water)
K_SS, CN_SS, CT_SS = 300.0, 0.35, 1.00
K_WW, CN_WW, CT_WW = 150.0, 0.18, 0.08
K_SW, CN_SW, CT_SW = 180.0, 0.20, 0.30
MU_SS_DRY, MU_SS_WET = 0.85, 0.50   # dry pile rests; wet sand lubricated
MU_WW, MU_SW = 0.15, 0.08
COH_WW, COH_RANGE_WW = 1.5e-3, 0.075  # water cohesion [N], range [m]
COH_WET_SAND = 1.2e-3                # capillary cohesion at saturation [N]

WET_RADIUS = 0.10                 # water-sand wetting interaction radius [m]
WET_RATE = 0.014                  # wetness growth rate per contact [1/s]
DIFF_D = 5.0                      # sand-sand wetness diffusion [1/s]
DIFF_DOWN_BIAS = 1.5              # enhanced downward percolation

WALL_K, WALL_CN, WALL_CT = 250.0, 0.20, 0.20
MU_WALL_SAND, MU_WALL_WATER = 0.70, 0.05

V_CLAMP = 6.5                     # velocity safety clamp [m/s]
DT = 1.0e-3                       # physics substep [s]
PAIR_R = 0.10                     # neighbor search radius [m]

# prescribed pouring (spout emitters): water is emitted on the same 0.065 m
# particle lattice the frozen initial streams use, at the stream speed, so the
# pour is a clean continuation of the initial slugs (no artificial overlap).
EMIT_START, EMIT_DT, EMIT_END = 0.07, 0.0351, 4.0
EMIT_PER_SPOUT = 8
SPOUT_X, SPOUT_Y, SPOUT_Z = 0.42, 0.08, 1.99
POUR_VZ, POUR_VX = -1.85, 0.38
_LATTICE = 0.065

# tray containment (inner faces of the frozen box walls; open front is
# virtually closed at the mirrored back-wall position so runoff stays in tray)
X_FACE, Y_BACK_FACE, Y_FRONT_FACE = 1.47, 1.07, 1.07

# diagnostic grid
H_GRID = 0.1
GX0, GX1, NX = -1.55, 1.55, 32
GY0, GY1, NY = -1.15, 1.15, 24
GZ0, GZ1, NZ = 0.0, 1.20, 13

CONTACT_LEN_SCALE = 0.075         # frozen mean sand element size [m]
MAX_CONTACT_SAMPLES = 4000


class BenchmarkSolver:
    """Particle solver for the wettable-sand / pouring-water scene."""

    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        n0 = len(frame.position)
        n_events = int(np.floor((EMIT_END - EMIT_START) / EMIT_DT)) + 1
        cap = n0 + n_events * EMIT_PER_SPOUT * 2 + 32

        self.pos = np.zeros((cap, 3))
        self.vel = np.zeros((cap, 3))
        self.mass = np.zeros(cap)
        self.body = np.zeros(cap, dtype=np.int32)
        self.rad = np.zeros(cap)
        self.w = np.zeros(cap)          # sand wetness in [0, W_MAX]
        self.wsrc = np.zeros(cap)       # wetness source accumulator

        self.pos[:n0] = frame.position
        self.vel[:n0] = frame.velocity
        self.mass[:n0] = frame.mass
        self.body[:n0] = frame.body_id.astype(np.int32)
        sand0 = self.body[:n0] == 1
        self.rad[:n0] = np.where(sand0, R_SAND, R_WATER)

        self.n = n0
        self.t = 0.0
        self.rng = np.random.default_rng(int(seed))
        self.emit_index = 0
        self.next_emit = EMIT_START
        self.emitted_mass = 0.0
        self.int_accum = np.zeros((2, 3))   # sand<-water, water<-sand
        self.int_count = 0
        self.last_frame_time = 0.0
        self.initial_sand_mass = float(self.mass[sand0].sum())
        self.initial_water_mass = float(self.mass[~sand0].sum())
        self._frames = 0

    # ------------------------------------------------------------ emission
    _EMIT_OFFSETS = tuple(
        (dx, dy, dz)
        for dz in (0.0, _LATTICE)
        for dy in (-_LATTICE * 0.5, _LATTICE * 0.5)
        for dx in (-_LATTICE * 0.5, _LATTICE * 0.5)
    )

    def _emit_due(self) -> None:
        while self.next_emit <= self.t + DT + 1e-9 and self.next_emit <= EMIT_END + 1e-9:
            for side in (-1.0, 1.0):
                for dx, dy, dz in self._EMIT_OFFSETS:
                    if self.n >= len(self.body):
                        return
                    self.pos[self.n] = (side * (SPOUT_X + dx), SPOUT_Y + dy, SPOUT_Z + dz)
                    self.vel[self.n] = (-side * POUR_VX, 0.0, POUR_VZ)
                    self.mass[self.n] = M_WATER
                    self.body[self.n] = 2
                    self.rad[self.n] = R_WATER
                    self.w[self.n] = 0.0
                    self.wsrc[self.n] = 0.0
                    self.n += 1
                    self.emitted_mass += M_WATER
            self.emit_index += 1
            self.next_emit = EMIT_START + self.emit_index * EMIT_DT

    # -------------------------------------------------- pairwise interaction
    def _pair_forces(self, n: int) -> np.ndarray:
        pos, vel = self.pos[:n], self.vel[:n]
        f = np.zeros((n, 3))
        tree = cKDTree(pos)
        pairs = tree.query_pairs(PAIR_R, output_type="ndarray")
        if len(pairs) == 0:
            return f
        i, j = pairs[:, 0], pairs[:, 1]
        dvec = pos[j] - pos[i]
        d = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
        ok = d > 1e-9
        if not ok.all():
            i, j, dvec, d = i[ok], j[ok], dvec[ok], d[ok]
        if len(i) == 0:
            return f
        nvec = dvec / d[:, None]
        vrel = vel[j] - vel[i]
        vn = np.einsum("ij,ij->i", vrel, nvec)
        vt = vrel - vn[:, None] * nvec

        bi, bj = self.body[i] == 1, self.body[j] == 1
        ss = bi & bj
        ww = (~bi) & (~bj)
        sw = bi ^ bj

        R = self.rad[i] + self.rad[j]
        delta = R - d
        contact = delta > 0.0

        k = np.where(ss, K_SS, np.where(ww, K_WW, K_SW))
        cn = np.where(ss, CN_SS, np.where(ww, CN_WW, CN_SW))
        ct = np.where(ss, CT_SS, np.where(ww, CT_WW, CT_SW))
        w_i, w_j = self.w[i], self.w[j]
        wet_frac = np.minimum(w_i, w_j) / W_MAX
        mu = np.where(ss, MU_SS_DRY - (MU_SS_DRY - MU_SS_WET) * wet_frac,
                      np.where(ww, MU_WW, MU_SW))

        # normal spring-dashpot (repulsion only)
        Fn = np.where(contact, k * np.maximum(delta, 0.0) - cn * vn, 0.0)
        np.maximum(Fn, 0.0, out=Fn)

        # tangential viscous friction clamped by Coulomb limit
        ftv = ct[:, None] * vt
        ftn = np.sqrt(np.einsum("ij,ij->i", ftv, ftv))
        cap = mu * Fn
        scale = np.where(ftn > 1e-12, np.minimum(1.0, cap / np.maximum(ftn, 1e-12)), 0.0)
        ftv = ftv * scale[:, None]

        Fnv = Fn[:, None] * nvec
        f_i = -Fnv + ftv
        f_j = Fnv - ftv

        # water cohesion (surface tension) just outside contact
        coh = np.zeros_like(d)
        cmask = ww & (~contact) & (d < COH_RANGE_WW)
        if cmask.any():
            span = np.maximum(COH_RANGE_WW - R[cmask], 1e-6)
            coh[cmask] = COH_WW * np.maximum(1.0 - (d[cmask] - R[cmask]) / span, 0.0)
            cohv = coh[:, None] * nvec
            f_i = f_i + cohv
            f_j = f_j - cohv

        # capillary cohesion between wet sand grains in contact
        cohw = np.where(ss & contact, COH_WET_SAND * wet_frac, 0.0)
        cohv2 = cohw[:, None] * nvec
        f_i = f_i + cohv2
        f_j = f_j - cohv2

        for c in range(3):
            f[:, c] += np.bincount(i, weights=f_i[:, c], minlength=n)
            f[:, c] += np.bincount(j, weights=f_j[:, c], minlength=n)

        # ---- wetting: concentration transfer from contacting water to sand
        if sw.any():
            wet_pairs = sw & (d < WET_RADIUS)
            if wet_pairs.any():
                sand_member = np.where(bi, i, j)
                counts = np.bincount(sand_member[wet_pairs], minlength=n)
                grow = WET_RATE * (1.0 - self.w[:n] / W_MAX) * counts
                grow = np.minimum(grow, (W_MAX - self.w[:n]) / DT)
                self.w[:n] += grow * DT
                self.wsrc[:n] += grow * DT

        # ---- wetness diffusion between contacting sand grains (percolation)
        if ss.any():
            i_s, j_s = i[ss], j[ss]
            wi, wj = self.w[i_s], self.w[j_s]
            flow = DIFF_D * (wi - wj) * DT
            bias = 1.0 + DIFF_DOWN_BIAS * np.clip(-nvec[ss, 2], 0.0, 1.0)
            ex = flow * bias
            self.w[:n] += np.bincount(i_s, weights=-ex, minlength=n)
            self.w[:n] += np.bincount(j_s, weights=ex, minlength=n)
            np.clip(self.w[:n], 0.0, W_MAX, out=self.w[:n])

        # ---- cross-material interaction force accumulation (paired)
        if sw.any():
            fi_sw, fj_sw = f_i[sw], f_j[sw]
            sand_is_i = bi[sw][:, None]
            f_sand = np.where(sand_is_i, fi_sw, fj_sw)
            f_water = np.where(sand_is_i, fj_sw, fi_sw)
            self.int_accum[0] += f_sand.sum(axis=0)
            self.int_accum[1] += f_water.sum(axis=0)
            self.int_count += 1
        return f

    # ---------------------------------------------------------- wall forces
    def _wall_forces(self, n: int) -> np.ndarray:
        pos, vel = self.pos[:n], self.vel[:n]
        f = np.zeros((n, 3))
        rad = self.rad[:n]
        is_sand = self.body[:n] == 1
        mu = np.where(is_sand, MU_WALL_SAND, MU_WALL_WATER)
        ct = np.where(is_sand, WALL_CT, WALL_CT * 0.25)
        # (axis, outward normal sign, face coordinate)
        halfspaces = (
            (2, 1.0, 0.0),         # floor (top face at z = 0)
            (0, 1.0, -X_FACE),     # left wall inner face
            (0, -1.0, X_FACE),     # right wall inner face
            (1, -1.0, Y_BACK_FACE),   # back wall inner face
            (1, 1.0, -Y_FRONT_FACE),  # front containment (keeps runoff in tray)
        )
        for axis, s, face in halfspaces:
            sd = s * pos[:, axis] - face
            pen = rad - sd
            m = pen > 0.0
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            vn = s * vel[idx, axis]
            Fn = np.maximum(WALL_K * pen[idx] - WALL_CN * vn, 0.0)
            f[idx, axis] += s * Fn
            vt = vel[idx].copy()
            vt[:, axis] = 0.0
            fvec = -ct[idx][:, None] * vt
            fmag = np.sqrt(np.einsum("ij,ij->i", fvec, fvec))
            lim = mu[idx] * Fn
            sc = np.where(fmag > 1e-12, np.minimum(1.0, lim / np.maximum(fmag, 1e-12)), 0.0)
            f[idx] += fvec * sc[:, None]
        return f

    # -------------------------------------------------------------- substep
    def _substep(self) -> None:
        self._emit_due()
        n = self.n
        f = self._pair_forces(n)
        f += self._wall_forces(n)
        acc = f / self.mass[:n, None]
        acc[:, 2] += G_Z
        vel = self.vel[:n] + acc * DT
        speed = np.sqrt(np.einsum("ij,ij->i", vel, vel))
        vel *= np.minimum(1.0, V_CLAMP / np.maximum(speed, 1e-12))[:, None]
        self.vel[:n] = vel
        self.pos[:n] += vel * DT
        # hard safety containment (should never trigger)
        np.clip(self.pos[:n, 0], -1.60, 1.60, out=self.pos[:n, 0])
        np.clip(self.pos[:n, 1], -1.25, 1.25, out=self.pos[:n, 1])
        np.clip(self.pos[:n, 2], -0.05, 2.60, out=self.pos[:n, 2])
        self.t += DT

    # ------------------------------------------------------------- assembly
    def _grid_indices(self, pos):
        ix = np.clip(((pos[:, 0] - GX0) / H_GRID).astype(np.int64), 0, NX - 1)
        iy = np.clip(((pos[:, 1] - GY0) / H_GRID).astype(np.int64), 0, NY - 1)
        iz = np.clip(((pos[:, 2] - GZ0) / H_GRID).astype(np.int64), 0, NZ - 1)
        return ix, iy, iz

    def _contact_distances(self, pos, n):
        gaps = []
        tree = cKDTree(pos)
        pairs = tree.query_pairs(0.09, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            d = np.sqrt(np.einsum("ij,ij->i", pos[j] - pos[i], pos[j] - pos[i]))
            g = d - (self.rad[i] + self.rad[j])
            gaps.append(g[g < 0.02])
        rad = self.rad[:n]
        halfspaces = (
            (2, 1.0, 0.0), (0, 1.0, -X_FACE), (0, -1.0, X_FACE),
            (1, -1.0, Y_BACK_FACE), (1, 1.0, -Y_FRONT_FACE),
        )
        for axis, s, face in halfspaces:
            sd = s * pos[:, axis] - face
            g = sd - rad
            g = g[g < 0.02]
            if g.size:
                gaps.append(g)
        out = np.concatenate(gaps) if gaps else np.zeros(0)
        out = out[np.isfinite(out)]
        if out.size > MAX_CONTACT_SAMPLES:
            stride = int(np.ceil(out.size / MAX_CONTACT_SAMPLES))
            out = out[::stride]
        if out.size == 0:
            out = np.zeros(1)
        return out

    def _assemble(self, time: float):
        n = self.n
        pos = self.pos[:n].copy()
        vel = self.vel[:n].copy()
        mass = self.mass[:n].copy()
        body = self.body[:n].copy()
        w = self.w[:n].copy()
        sand = body == 1
        water = body == 2

        ix, iy, iz = self._grid_indices(pos)
        cell = (ix, iy, iz)
        cnt = np.zeros((NX, NY, NZ))
        np.add.at(cnt, cell, 1.0)
        vel_sum = np.zeros((NX, NY, NZ, 3))
        np.add.at(vel_sum, cell, vel)
        velocity_field = vel_sum / np.maximum(cnt, 1.0)[..., None]

        sand_cnt = np.zeros((NX, NY, NZ))
        np.add.at(sand_cnt, (ix[sand], iy[sand], iz[sand]), 1.0)
        active_mask = (sand_cnt > 0.0).astype(np.float64)
        w_sum = np.zeros((NX, NY, NZ))
        np.add.at(w_sum, (ix[sand], iy[sand], iz[sand]), w[sand])
        transported_scalar = w_sum / np.maximum(sand_cnt, 1.0)

        src_sum = np.zeros((NX, NY, NZ))
        np.add.at(src_sum, (ix[sand], iy[sand], iz[sand]), self.wsrc[:n][sand])
        elapsed = max(time - self.last_frame_time, 1e-9)
        scalar_source = src_sum / elapsed
        self.wsrc[:n] = 0.0

        contact_sd = self._contact_distances(pos, n)

        if self.int_count > 0:
            interaction_force = self.int_accum / float(self.int_count)
        else:
            interaction_force = np.zeros((2, 3))
        self.int_accum[:] = 0.0
        self.int_count = 0

        mass_quantity = np.stack([np.where(sand, mass, 0.0),
                                  np.where(water, mass, 0.0)], axis=1)

        attributes = {
            "material_id": body.astype(np.int32),
            "solid_phase_fraction": sand.astype(np.float64),
            "water_concentration": np.where(sand, w / W_MAX, 1.0),
            "wetness": np.where(sand, w, 1.0),
            "mass_quantity": mass_quantity,
        }
        fields = {
            "velocity_field": velocity_field,
            "active_mask": active_mask,
            "transported_scalar": transported_scalar,
            "scalar_source": scalar_source,
            "contact_signed_distance": contact_sd,
            "interaction_force": interaction_force,
            "cumulative_mass_source": np.array([0.0, self.emitted_mass]),
        }
        scalars = {
            "contact_length_scale": CONTACT_LEN_SCALE,
            "grid_spacing": H_GRID,
            "sand_mass": float(mass[sand].sum()),
            "water_mass": float(mass[water].sum()),
            "poured_water_mass": float(self.emitted_mass),
            "mean_sand_wetness": float(w[sand].mean()) if sand.any() else 0.0,
            "max_sand_wetness": float(w[sand].max()) if sand.any() else 0.0,
            "wet_sand_fraction": float((w[sand] > 1e-6).mean()) if sand.any() else 0.0,
        }
        self.last_frame_time = float(time)
        self._frames += 1

        from worldsolver.contracts import StateFrame
        frame = StateFrame(
            float(time), pos, vel, mass=mass, body_id=body.astype(np.int32),
            attributes=attributes, fields=fields, scalars=scalars,
        )
        frame.validate()
        return frame

    # --------------------------------------------------------------- public
    def advance_to(self, time: float):
        target = float(time)
        if target > self.t:
            steps = int(round((target - self.t) / DT))
            for _ in range(max(steps, 0)):
                self._substep()
            self.t = target  # snap to the exact requested sample time
        return self._assemble(self.t)

    def diagnostics(self) -> dict[str, object]:
        n = self.n
        sand = self.body[:n] == 1
        water = self.body[:n] == 2
        return {
            "particles": int(n),
            "sand_particles": int(sand.sum()),
            "water_particles": int(water.sum()),
            "sand_mass_kg": float(self.mass[:n][sand].sum()),
            "water_mass_kg": float(self.mass[:n][water].sum()),
            "poured_water_mass_kg": float(self.emitted_mass),
            "mean_sand_wetness": float(self.w[:n][sand].mean()),
            "max_sand_wetness": float(self.w[:n][sand].max()),
            "frames_emitted": int(self._frames),
            "final_time": float(self.t),
        }
