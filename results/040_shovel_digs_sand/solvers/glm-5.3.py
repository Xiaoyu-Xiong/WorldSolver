"""Granular solver for 040_shovel_digs_sand.

Position-based dynamics (PBD) granular material with:
- sphere-sphere contacts at rest spacing (compression driven),
- Coulomb friction on particle-particle, floor/wall and shovel contacts,
- kinematic shovel blade (rotated box) driven by smooth-monotone (PCHIP)
  keyframe translation,
- vectorized cell-list pair search.

The physics is stepped with a fixed substep; states are exported at the
scene's requested sample times as StateFrame with all required fields.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

from worldsolver.contracts import StateFrame


GRAVITY = np.array([0.0, 0.0, -9.81])
SUB_DT = 0.002          # physical substep
SOLVER_ITERS = 2        # contact iterations per substep
PP_CORR = 0.85          # Jacobi relaxation for particle-particle corrections
MU_PP = 0.55            # sand internal friction coefficient
MU_WALL = 0.45          # sand-wall friction
MU_TOOL = 0.5           # sand-shovel friction
DAMPING = 0.02          # mild global velocity damping per second

# Frozen shovel blade geometry (scene_geometry.yaml).
BLADE_HE = np.array([0.14, 0.64, 0.34])
BLADE_QUAT = np.array([0.0, -0.0872, 0.0, 0.9962])  # xyzw
BLADE_T0 = np.array([-2.05, 0.0, 0.38])

# Frozen container (config.yaml boundary_conditions).
X_MIN, X_MAX = -2.35, 2.35
Y_MIN, Y_MAX = -1.5, 1.5
FLOOR_Z = 0.0

KEY_TIMES = np.array([0.0, 0.5, 3.2, 4.0])
KEY_POS = np.array([
    [-2.05, 0.0, 0.38],
    [-2.05, 0.0, 0.38],
    [1.35, 0.0, 0.46],
    [1.7, 0.0, 0.86],
])

_KEY_STRIDE = 1 << 21  # cell hashing stride


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class BenchmarkSolver:
    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        self.x = np.array(frame.position, dtype=float)
        self.v = np.zeros_like(self.x)
        self.n = len(self.x)
        self.mass = np.array(frame.mass, dtype=float)
        self.body_id = np.ones(self.n, dtype=np.int32)
        self.initial_volume = float(getattr(frame.scalars, "get", dict().get)("initial_sand_volume", 0.0)) \
            if isinstance(frame.scalars, dict) else 0.0
        if not self.initial_volume:
            self.initial_volume = float(self.n * (0.075 ** 3))

        meta = scene.metadata() if hasattr(scene, "metadata") else {}
        ic = getattr(scene, "config", {}).get("initial_conditions", {}) if hasattr(scene, "config") else {}
        self.spacing = float(ic.get("sand_particle_spacing", 0.075))
        self.d0 = self.spacing          # rest contact distance
        self.radius = 0.5 * self.spacing
        self.g = np.asarray(meta.get("gravity", GRAVITY), dtype=float)

        # shovel interpolation (smooth monotone)
        if _HAVE_SCIPY:
            self._pos_spline = PchipInterpolator(KEY_TIMES, KEY_POS, axis=0, extrapolate=True)
            self._vel_spline = self._pos_spline.derivative()
        else:
            self._pos_spline = None

        self.R_blade = _quat_to_matrix(BLADE_QUAT)
        self.R_blade_T = self.R_blade.T

        self.t = 0.0
        self._x0 = self.x.copy()
        self._z0_max = float(self.x[:, 2].max()) if self.n else 0.0
        self._frame_cache: dict[float, StateFrame] = {}
        self._contact_sd = np.zeros(0)
        self._escaped_ever = 0

        # cell grid geometry (covers container interior with margin)
        self.cell = self.d0
        self.grid_min = np.array([X_MIN - 3 * self.cell, Y_MIN - 3 * self.cell,
                                  FLOOR_Z - 3 * self.cell])
        self._pair_offsets = self._half_neighborhood()

        self._ready = True

    # ------------------------------------------------------------- kinematics
    def _shovel_state(self, t: float):
        if self._pos_spline is not None:
            c = np.asarray(self._pos_spline(t), dtype=float)
            vel = np.asarray(self._vel_spline(t), dtype=float)
        else:
            c = np.array([np.interp(t, KEY_TIMES, KEY_POS[:, k]) for k in range(3)])
            eps = 1e-4
            cp = np.array([np.interp(t + eps, KEY_TIMES, KEY_POS[:, k]) for k in range(3)])
            cm = np.array([np.interp(t - eps, KEY_TIMES, KEY_POS[:, k]) for k in range(3)])
            vel = (cp - cm) / (2 * eps)
        return c, vel

    # -------------------------------------------------------------- neighbors
    @staticmethod
    def _half_neighborhood():
        offs = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if (di, dj, dk) > (0, 0, 0):
                        offs.append((di, dj, dk))
        # include the zero offset (same-cell pairs) handled with i<j filter
        offs.append((0, 0, 0))
        keys = [((di * _KEY_STRIDE) + dj) * _KEY_STRIDE + dk for di, dj, dk in offs]
        same_cell = np.array([o == (0, 0, 0) for o in offs])
        return np.array(offs), np.array(keys), same_cell

    def _cell_keys(self, x: np.ndarray):
        ijk = np.floor((x - self.grid_min) / self.cell).astype(np.int64)
        keys = (ijk[:, 0] * _KEY_STRIDE + ijk[:, 1]) * _KEY_STRIDE + ijk[:, 2]
        return ijk, keys

    def _find_pairs(self, x: np.ndarray):
        """Vectorized cell-list pair search. Returns (i, j) index arrays with i<j."""
        _, keys = self._cell_keys(x)
        order = np.argsort(keys, kind="stable")
        skeys = keys[order]
        # unique cells
        uniq_start = np.empty(0)
        first = np.ones(len(skeys), dtype=bool)
        first[1:] = skeys[1:] != skeys[:-1]
        ukeys = skeys[first]
        starts = np.flatnonzero(first)
        counts = np.diff(np.append(starts, len(skeys)))

        offs, off_keys, same_cell = self._pair_offsets
        ii_list, jj_list = [], []
        for o in range(len(offs)):
            target = ukeys + off_keys[o]
            loc = np.searchsorted(ukeys, target)
            loc_clipped = np.clip(loc, 0, len(ukeys) - 1)
            hit = ukeys[loc_clipped] == target
            if not np.any(hit):
                continue
            sa = starts[hit]
            ca = counts[hit]
            sb = starts[loc_clipped[hit]]
            cb = counts[loc_clipped[hit]]
            m = ca * cb
            total = int(m.sum())
            if total == 0:
                continue
            # within-block flat index
            block_start = np.repeat(np.cumsum(m) - m, m)
            within = np.arange(total) - block_start
            cb_rep = np.repeat(cb, m)
            a_off = within // cb_rep
            b_off = within % cb_rep
            i = np.repeat(sa, m) + a_off
            j = np.repeat(sb, m) + b_off
            if same_cell[o]:
                keep = i < j
                i, j = i[keep], j[keep]
            if len(i):
                ii_list.append(order[i])
                jj_list.append(order[j])
        if ii_list:
            return np.concatenate(ii_list), np.concatenate(jj_list)
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    # ------------------------------------------------------------- constraints
    def _apply_particle_contacts(self, x: np.ndarray, xp: np.ndarray,
                                 sd_out: list) -> None:
        i, j = self._find_pairs(x)
        if len(i) == 0:
            return
        d = x[j] - x[i]
        dist2 = np.einsum("ij,ij->i", d, d)
        good = dist2 < self.d0 * self.d0
        good &= dist2 > 1e-12
        if not np.any(good):
            return
        i, j, d = i[good], j[good], d[good]
        dist = np.sqrt(dist2[good])
        n = d / dist[:, None]
        gap = dist - self.d0
        sd_out.append(gap)
        w = np.repeat(PP_CORR / 2.0, len(i))
        corr = (gap * w)[:, None] * n

        # Coulomb friction on tangential relative displacement
        drel = (x[i] - xp[i]) - (x[j] - xp[j])
        tang = drel - np.einsum("ij,ij->i", drel, n)[:, None] * n
        tnorm = np.sqrt(np.einsum("ij,ij->i", tang, tang))
        lim = MU_PP * np.abs(gap) * PP_CORR
        scale = np.ones_like(tnorm)
        act = tnorm > 1e-12
        scale[act] = np.minimum(1.0, lim[act] / tnorm[act])
        fric = -0.5 * (tang * scale[:, None])

        move_i = corr + fric
        np.add.at(x, i, move_i)
        np.add.at(x, j, -move_i)

    def _apply_container(self, x: np.ndarray, xp: np.ndarray,
                         sd_out: list) -> None:
        r = self.radius
        mu = MU_WALL
        # floor
        pen = r - (x[:, 2] - FLOOR_Z)
        act = pen > 0
        if np.any(act):
            idx = np.flatnonzero(act)
            sd_out.append(pen[idx] * -1.0 + 0.0)  # negative => penetration
            x[idx, 2] += pen[idx]
            # friction against static floor
            tang = x[idx, :2] - xp[idx, :2]
            tn = np.sqrt(np.einsum("ij,ij->i", tang, tang))
            lim = mu * pen[idx]
            sc = np.ones_like(tn)
            m = tn > 1e-12
            sc[m] = np.minimum(1.0, lim[m] / tn[m])
            x[idx, :2] -= tang * sc[:, None]
        # walls
        for axis, lo, hi in ((0, X_MIN, X_MAX), (1, Y_MIN, Y_MAX)):
            pen_hi = x[:, axis] + r - hi
            act = pen_hi > 0
            if np.any(act):
                idx = np.flatnonzero(act)
                x[idx, axis] -= pen_hi[idx]
                sd_out.append(-pen_hi[idx])
            pen_lo = lo - (x[:, axis] - r)
            act = pen_lo > 0
            if np.any(act):
                idx = np.flatnonzero(act)
                x[idx, axis] += pen_lo[idx]
                sd_out.append(-pen_lo[idx])

    def _apply_shovel(self, x: np.ndarray, xp: np.ndarray, t: float,
                      sd_out: list) -> None:
        center, vel = self._shovel_state(t)
        pl = (x - center) @ self.R_blade_T
        cl = np.clip(pl, -BLADE_HE, BLADE_HE)
        diff = pl - cl
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        # signed distance to box surface (negative inside)
        q = np.abs(pl) - BLADE_HE
        outside = np.sqrt(np.sum(np.maximum(q, 0.0) ** 2, axis=1)) + np.minimum(
            np.max(q, axis=1), 0.0)
        pen = self.radius - outside
        act = pen > 0
        if not np.any(act):
            return
        idx = np.flatnonzero(act)
        sd_out.append(-pen[idx])
        # push particle to surface + radius along local normal
        nl = diff[idx]
        zero = np.einsum("ij,ij->i", nl, nl) < 1e-16
        if np.any(zero):
            zidx = idx[zero]
            qz = np.abs(pl[zidx]) - BLADE_HE
            ax = np.argmin(np.abs(qz), axis=1)
            nn = np.zeros((len(zidx), 3))
            nn[np.arange(len(zidx)), ax] = np.sign(pl[zidx][np.arange(len(zidx)), ax])
            nl[zero] = nn
        nl /= np.sqrt(np.einsum("ij,ij->i", nl, nl))[:, None] + 1e-16
        target_local = cl[idx] + nl * self.radius
        target_world = center + target_local @ self.R_blade_T
        x[idx] = target_world
        # friction against moving blade (relative displacement this substep)
        center_next, _ = self._shovel_state(t + SUB_DT)
        blade_disp = center_next - center
        drel = (x[idx] - xp[idx]) - blade_disp[None, :]
        n_world = nl @ self.R_blade_T
        tang = drel - np.einsum("ij,ij->i", drel, n_world)[:, None] * n_world
        tn = np.sqrt(np.einsum("ij,ij->i", tang, tang))
        lim = MU_TOOL * pen[idx]
        sc = np.ones_like(tn)
        m = tn > 1e-12
        sc[m] = np.minimum(1.0, lim[m] / tn[m])
        x[idx] -= tang * sc[:, None]

    # ------------------------------------------------------------------- step
    def _substep(self) -> None:
        h = SUB_DT
        t = self.t
        xp = self.x.copy()
        self.v += self.g * h
        self.v *= max(0.0, 1.0 - DAMPING * h)
        self.x += self.v * h

        sd: list = []
        for _ in range(SOLVER_ITERS):
            self._apply_shovel(self.x, xp, t, sd)
            self._apply_container(self.x, xp, sd)
            self._apply_particle_contacts(self.x, xp, sd)
        if sd:
            self._contact_sd = np.concatenate(sd)
        self.v = (self.x - xp) / h
        # kill residual interpenetration velocity spikes
        vmax = 20.0
        sp = np.sqrt(np.einsum("ij,ij->i", self.v, self.v))
        fast = sp > vmax
        if np.any(fast):
            self.v[fast] *= (vmax / sp[fast])[:, None]
        self.t += h

    # -------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = float(time)
        key = round(target, 6)
        if key in self._frame_cache:
            return self._frame_cache[key]
        guard = 0
        while self.t < target - 1e-9:
            self._substep()
            guard += 1
            if guard > 100000:
                break
        frame = self._make_frame(target)
        self._frame_cache[key] = frame
        return frame

    def _make_frame(self, t: float) -> StateFrame:
        n = self.n
        ext = np.repeat((self.mass[:, None] * self.g[None, :]), 1, axis=0)
        sd = self._contact_sd if len(self._contact_sd) else np.zeros(1)
        escaped = int(np.sum((self.x[:, 2] < -0.05) |
                             (np.abs(self.x[:, 0]) > 3.2) |
                             (np.abs(self.x[:, 1]) > 2.2)))
        ke = float(0.5 * np.sum(self.mass * np.einsum("ij,ij->i", self.v, self.v)))
        frame = StateFrame(
            time=t,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "contact_signed_distance": np.asarray(sd, dtype=float),
                "cumulative_mass_source": np.zeros(1),
                "external_force": ext,
            },
            scalars={
                "contact_length_scale": float(self.spacing),
                "displaced_sand_volume": self._displaced_volume(),
                "mound_height": self._mound_height(),
                "shovel_reaction_force": self._shovel_force(),
                "escaped_particle_count": float(escaped),
                "settled_kinetic_energy": ke,
                "substep_dt": SUB_DT,
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------ diagnostics
    def _displaced_volume(self) -> float:
        moved = np.linalg.norm(self.x - self._x0, axis=1) > 1.5 * self.spacing
        return float(np.count_nonzero(moved) * self.spacing ** 3)

    def _mound_height(self) -> float:
        return float(self.x[:, 2].max() - self._z0_max) if self.n else 0.0

    def _shovel_force(self) -> float:
        center, vel = self._shovel_state(self.t)
        pl = (self.x - center) @ self.R_blade_T
        q = np.abs(pl) - BLADE_HE
        outside = np.sqrt(np.sum(np.maximum(q, 0.0) ** 2, axis=1)) + np.minimum(
            np.max(q, axis=1), 0.0)
        contact = outside < self.radius
        return float(np.count_nonzero(contact)) * self.mass.mean() * 9.81 * 0.25

    def diagnostics(self) -> dict[str, object]:
        return {
            "displaced_sand_volume": self._displaced_volume(),
            "mound_height": self._mound_height(),
            "shovel_reaction_force": self._shovel_force(),
            "escaped_particle_count": int(np.sum(
                (self.x[:, 2] < -0.05) | (np.abs(self.x[:, 0]) > 3.2) |
                (np.abs(self.x[:, 1]) > 2.2))),
            "settled_kinetic_energy": float(
                0.5 * np.sum(self.mass * np.einsum("ij,ij->i", self.v, self.v))),
            "final_time": float(self.t),
            "num_particles": int(self.n),
        }
