"""Granular PBD solver for the shovel-digs-sand scene.

Physically-based position-based-dynamics (PBD) granular solver:
  * semi-implicit Euler advection under gravity,
  * pairwise non-penetration constraints at rest spacing (uniform-grid
    acceleration, binned force/correction accumulation),
  * exact signed-distance projection against the kinematic shovel blade
    (oriented box SDF) and shaft (capsule SDF) driven by the config
    keyframes with smooth monotone (smoothstep) interpolation,
  * container floor/wall constraints,
  * velocity-level Coulomb-like friction for particle pairs, tool surface,
    and floor so the sand attains a stable angle of repose and is dragged
    and displaced by the advancing blade.

All required trajectory fields are exported per frame.
"""
from __future__ import annotations

import numpy as np

try:  # StateFrame comes from the platform contracts package.
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - fallback for odd import contexts
    StateFrame = None

HALF_OFFSETS = (
    (1, -1, -1), (1, -1, 0), (1, -1, 1),
    (1, 0, -1), (1, 0, 0), (1, 0, 1),
    (1, 1, -1), (1, 1, 0), (1, 1, 1),
    (0, 1, -1), (0, 1, 0), (0, 1, 1),
    (0, 0, 1),
)


def _quat_to_R(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class BenchmarkSolver:
    # ---- tunables -------------------------------------------------------
    SUB_DT = 0.005          # target substep
    ITERS = 2               # projection iterations per substep
    RELAX = 0.85            # pair constraint relaxation
    MU_PAIR = 0.25          # tangential pair friction fraction per substep
    MU_TOOL = 0.7           # tool-surface tangential coupling
    R_TOOL = 0.025          # clearance radius around tool surfaces
    DRAG = 0.999            # global velocity damping per substep
    V_MAX = 20.0            # speed clamp

    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)
        frame0 = scene.initial_state()
        self.N = int(len(frame0.position))
        self.p = np.ascontiguousarray(frame0.position, dtype=float)
        self.v = np.ascontiguousarray(frame0.velocity, dtype=float)
        self.m = np.ascontiguousarray(frame0.mass, dtype=float)
        self.body_id = np.ascontiguousarray(frame0.body_id, dtype=np.int32)
        self.t = 0.0

        ic = cfg["initial_conditions"]
        self.spacing = float(ic["sand_particle_spacing"])
        self.dmin = self.spacing
        lo = np.asarray(ic["sand_bounds_min"], float)
        hi = np.asarray(ic["sand_bounds_max"], float)
        self.bed_top = float(hi[2])
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), float)

        bc = cfg["boundary_conditions"]
        self.floor_z = float(bc["container_floor_z"])
        ib = [float(x) for x in bc["container_inner_bounds"]]
        self.xlo, self.xhi, self.ylo, self.yhi = ib

        # ---- shovel geometry (from frozen scene geometry, with fallbacks)
        geo = cfg.get("scene_geometry") or {}
        objs = {o.get("id"): o for o in geo.get("objects", [])} if isinstance(geo, dict) else {}
        blade = objs.get("shovel_blade", {}) or {}
        shaft = objs.get("shovel_shaft", {}) or {}
        bg = blade.get("geometry", {}) or {}
        bt = blade.get("transform", {}) or {}
        self.blade_h = np.asarray(bg.get("half_extents", [0.14, 0.64, 0.34]), float)
        self.blade_R = _quat_to_R(bt.get("rotation_xyzw", [0.0, -0.0872, 0.0, 0.9962]))
        self.blade_c0 = np.asarray(bt.get("translation", [-2.05, 0.0, 0.38]), float)
        sg = shaft.get("geometry", {}) or {}
        st = shaft.get("transform", {}) or {}
        self.shaft_r = float(sg.get("radius", 0.065))
        self.shaft_L = 0.5 * float(sg.get("height", 2.0))
        self.shaft_R = _quat_to_R(st.get("rotation_xyzw", [0.0, -0.3606, 0.0, 0.9327]))
        self.shaft_c0 = np.asarray(st.get("translation", [-2.76, 0.0, 1.25]), float)
        self.shaft_off = self.shaft_c0 - self.blade_c0
        self.shaft_axis = self.shaft_R @ np.array([0.0, 0.0, 1.0])

        km = (cfg.get("kinematic_motion", {}) or {}).get("shovel_assembly", {}) or {}
        kfs = km.get("translation_keyframes") or [
            {"time": 0.0, "value": [-2.05, 0.0, 0.38]},
            {"time": 0.5, "value": [-2.05, 0.0, 0.38]},
            {"time": 3.2, "value": [1.35, 0.0, 0.46]},
            {"time": 4.0, "value": [1.7, 0.0, 0.86]},
        ]
        self.kt = np.array([float(k["time"]) for k in kfs])
        self.kv = np.array([np.asarray(k["value"], float) for k in kfs])

        # ---- background grid for neighbor search
        self.cs = self.dmin
        self.ox, self.oy, self.oz = -2.7, -1.8, -0.25
        self.GX = int(np.ceil((2.7 - self.ox) / self.cs))
        self.GY = int(np.ceil((1.8 - self.oy) / self.cs))
        self.GZ = int(np.ceil((2.45 - self.oz) / self.cs))

        # diagnostics accumulators
        self.tool_impulse = 0.0
        self.tool_contact_time = 0.0
        self.peak_tool_force = 0.0
        self._last_frame = None

    # ---- kinematics ------------------------------------------------------
    def shovel_center(self, t: float) -> np.ndarray:
        kt, kv = self.kt, self.kv
        if t <= kt[0]:
            return kv[0].copy()
        if t >= kt[-1]:
            return kv[-1].copy()
        i = int(np.searchsorted(kt, t, side="right") - 1)
        i = min(max(i, 0), len(kt) - 2)
        t0, t1 = kt[i], kt[i + 1]
        s = (t - t0) / (t1 - t0)
        s = s * s * (3.0 - 2.0 * s)  # smoothstep: monotone and C1
        return kv[i] + s * (kv[i + 1] - kv[i])

    def shovel_center_dot(self, t: float) -> np.ndarray:
        h = 1e-3
        return (self.shovel_center(t + h) - self.shovel_center(t - h)) / (2.0 * h)

    # ---- SDF helpers -----------------------------------------------------
    def _box_sdf(self, p, c, R, h):
        loc = (p - c) @ R
        q = np.abs(loc) - h
        pos = np.maximum(q, 0.0)
        outside = np.sqrt(np.sum(pos * pos, axis=1))
        inside = np.minimum(np.maximum(q[:, 0], np.maximum(q[:, 1], q[:, 2])), 0.0)
        sdf = outside + inside
        nrm = np.zeros_like(loc)
        ext = outside > 1e-12
        nrm[ext] = pos[ext] * np.sign(loc[ext])
        nrm[ext] /= np.maximum(np.linalg.norm(nrm[ext], axis=1, keepdims=True), 1e-12)
        if np.any(~ext):
            idx = np.nonzero(~ext)[0]
            axis = np.argmax(q[idx], axis=1)
            nl = np.zeros((len(idx), 3))
            nl[np.arange(len(idx)), axis] = np.sign(loc[idx, np.arange(3)[axis]] + 1e-15) + 1e-12
            nl[nl == 0.0] = 1.0
            nrm[idx] = nl @ R.T
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        return sdf, nrm

    def _capsule_sdf(self, p, c, axis, L, r):
        d = p - c
        t = np.clip(d @ axis, -L, L)
        closest = c + t[:, None] * axis[None, :]
        diff = p - closest
        dist = np.linalg.norm(diff, axis=1)
        near0 = dist < 1e-9
        nrm = np.where(near0[:, None], np.array([0.0, 0.0, 1.0]), diff / np.maximum(dist, 1e-12)[:, None])
        return dist - r, nrm

    def _tool_sdf(self, p, c_b):
        sdf_b, n_b = self._box_sdf(p, c_b, self.blade_R, self.blade_h)
        sdf_s, n_s = self._capsule_sdf(p, c_b + self.shaft_off, self.shaft_axis,
                                       self.shaft_L, self.shaft_r)
        use_s = sdf_s < sdf_b
        sdf = np.where(use_s, sdf_s, sdf_b)
        nrm = np.where(use_s[:, None], n_s, n_b)
        return sdf, nrm

    # ---- neighbor pairs --------------------------------------------------
    def _cell_ids(self, p):
        ix = np.floor((p[:, 0] - self.ox) / self.cs).astype(np.int64)
        iy = np.floor((p[:, 1] - self.oy) / self.cs).astype(np.int64)
        iz = np.floor((p[:, 2] - self.oz) / self.cs).astype(np.int64)
        np.clip(ix, 0, self.GX - 1, out=ix)
        np.clip(iy, 0, self.GY - 1, out=iy)
        np.clip(iz, 0, self.GZ - 1, out=iz)
        return ix + iy * self.GX + iz * (self.GX * self.GY)

    def _expand_blocks(self, startsA, cA, startsB, cB):
        tot = cA * cB
        if tot.sum() == 0:
            return (np.empty(0, np.int64),) * 2
        base = np.repeat(np.cumsum(tot) - tot, tot)
        r = np.arange(int(tot.sum())) - base
        cBr = np.repeat(cB, tot)
        ia = np.repeat(startsA, tot) + r // cBr
        ib = np.repeat(startsB, tot) + r % cBr
        return ia, ib

    def _build_pairs(self, p):
        cid = self._cell_ids(p)
        order = np.argsort(cid, kind="stable")
        sc = cid[order]
        uniq, starts, counts = np.unique(sc, return_index=True, return_counts=True)
        a_list = []
        b_list = []
        n_sorted = len(sc)
        kmax = min(int(counts.max()), 8)
        for k in range(1, kmax + 1):
            idx = np.arange(n_sorted - k)
            sel = sc[idx] == sc[idx + k]
            if sel.any():
                a_list.append(order[idx[sel]])
                b_list.append(order[idx[sel] + k])
        ix = uniq % self.GX
        iy = (uniq // self.GX) % self.GY
        iz = uniq // (self.GX * self.GY)
        ncell = len(uniq)
        for dx, dy, dz in HALF_OFFSETS:
            tx, ty, tz = ix + dx, iy + dy, iz + dz
            ok = (tx >= 0) & (tx < self.GX) & (ty >= 0) & (ty < self.GY) & (tz >= 0) & (tz < self.GZ)
            if not ok.any():
                continue
            tcid = np.where(ok, tx + ty * self.GX + tz * (self.GX * self.GY), -1)
            pos = np.searchsorted(uniq, tcid)
            pos_c = np.minimum(pos, ncell - 1)
            found = ok & (uniq[pos_c] == tcid)
            if not found.any():
                continue
            cellsA = np.nonzero(found)[0]
            cellsB = pos_c[found]
            ia, ib = self._expand_blocks(starts[cellsA], counts[cellsA],
                                         starts[cellsB], counts[cellsB])
            a_list.append(order[ia])
            b_list.append(order[ib])
        if not a_list:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        a = np.concatenate(a_list)
        b = np.concatenate(b_list)
        # distance prefilter
        d = p[a] - p[b]
        d2 = np.einsum("ij,ij->i", d, d)
        cut = (self.dmin * 1.05) ** 2
        keep = d2 < cut
        return a[keep], b[keep]

    def _accumulate(self, out, ai, bi, w):
        # out[ai] += w ; out[bi] -= w  (equal-mass symmetric application)
        for ax in range(3):
            out[:, ax] += (np.bincount(ai, weights=w[:, ax], minlength=self.N)
                           - np.bincount(bi, weights=w[:, ax], minlength=self.N))

    def _pair_project(self, p, a, b):
        if len(a) == 0:
            return
        d = p[a] - p[b]
        dist = np.sqrt(np.maximum(np.einsum("ij,ij->i", d, d), 1e-14))
        ov = self.dmin - dist
        sel = ov > 0.0
        if not sel.any():
            return
        ai, bi = a[sel], b[sel]
        n = d[sel] / dist[sel, None]
        c = (0.5 * self.RELAX * ov[sel])[:, None] * n
        self._accumulate(p, ai, bi, c)

    def _pair_friction(self, p, v, a, b):
        if len(a) == 0:
            return
        d = p[a] - p[b]
        dist = np.sqrt(np.maximum(np.einsum("ij,ij->i", d, d), 1e-14))
        sel = dist < self.dmin
        if not sel.any():
            return
        ai, bi = a[sel], b[sel]
        n = d[sel] / dist[sel, None]
        vrel = v[ai] - v[bi]
        vn = np.sum(vrel * n, axis=1, keepdims=True)
        vt = vrel - vn * n
        dv = (0.5 * self.MU_PAIR) * vt
        self._accumulate(v, ai, bi, -dv)

    # ---- simulation ------------------------------------------------------
    def _substep(self, t_new: float, dt: float):
        p, v = self.p, self.v
        v += self.gravity * dt
        p_prev = p.copy()
        p += v * dt

        a, b = self._build_pairs(p)

        c_b = self.shovel_center(t_new)
        v_tool = self.shovel_center_dot(t_new)

        for _ in range(self.ITERS):
            self._pair_project(p, a, b)
            sdf, nrm = self._tool_sdf(p, c_b)
            sel = sdf < self.R_TOOL
            if sel.any():
                p[sel] += ((self.R_TOOL - sdf[sel])[:, None]) * nrm[sel]
            np.maximum(p[:, 2], self.floor_z, out=p[:, 2])
            np.clip(p[:, 0], self.xlo, self.xhi, out=p[:, 0])
            np.clip(p[:, 1], self.ylo, self.yhi, out=p[:, 1])
            np.minimum(p[:, 2], 2.35, out=p[:, 2])

        v[:] = (p - p_prev) / dt

        # friction: pairs
        self._pair_friction(p, v, a, b)

        # friction / drag: tool surface (kinematic coupling)
        sdf, nrm = self._tool_sdf(p, c_b)
        sel = sdf < self.R_TOOL + 0.02
        if sel.any():
            rel = v[sel] - v_tool[None, :]
            n = nrm[sel]
            vn = np.sum(rel * n, axis=1, keepdims=True)
            vt = rel - vn * n
            dv = self.MU_TOOL * vt
            v_before = v[sel].copy()
            v[sel] -= dv
            imp = float(np.sum(self.m[sel, None] * np.abs(v[sel] - v_before)))
            self.tool_impulse += imp
            self.tool_contact_time += dt
            f_inst = imp / dt if dt > 0 else 0.0
            if f_inst > self.peak_tool_force:
                self.peak_tool_force = f_inst

        # floor friction
        on_floor = p[:, 2] < self.floor_z + 0.015
        if on_floor.any():
            v[on_floor, 0] *= 0.86
            v[on_floor, 1] *= 0.86

        v *= self.DRAG
        speed = np.linalg.norm(v, axis=1)
        over = speed > self.V_MAX
        if over.any():
            v[over] *= (self.V_MAX / speed[over])[:, None]
        bad = ~np.isfinite(v).all(axis=1)
        if bad.any():
            v[bad] = 0.0
        bad = ~np.isfinite(p).all(axis=1)
        if bad.any():
            p[bad] = p_prev[bad]
            v[bad] = 0.0

    def advance_to(self, time: float):
        t = float(time)
        span = t - self.t
        if span > 1e-12:
            n = max(1, int(np.ceil(span / self.SUB_DT - 1e-9)))
            dt = span / n
            for s in range(n):
                self._substep(self.t + dt * (s + 1), dt)
            self.t = t
        return self._frame(t)

    # ---- output ----------------------------------------------------------
    def _frame(self, t: float):
        p, v = self.p, self.v
        c_b = self.shovel_center(t)
        sdf, _ = self._tool_sdf(p, c_b)
        gaps = []
        near = sdf < self.R_TOOL + 0.05
        if near.any():
            gaps.append(sdf[near])
        zgap = p[:, 2] - self.floor_z
        sel = zgap < 0.08
        if sel.any():
            gaps.append(zgap[sel])
        for arr, lo, hi in ((p[:, 0], self.xlo, self.xhi), (p[:, 1], self.ylo, self.yhi)):
            g1 = arr - lo
            s1 = g1 < 0.08
            if s1.any():
                gaps.append(g1[s1])
            g2 = hi - arr
            s2 = g2 < 0.08
            if s2.any():
                gaps.append(g2[s2])
        csd = np.concatenate(gaps) if gaps else np.array([0.1])

        ext_force = self.m[:, None] * self.gravity[None, :]
        ke = float(0.5 * np.sum(self.m * np.einsum("ij,ij->i", v, v)))

        attributes = {
            "mass_quantity": self.m.copy(),
            "material_label": np.ones(self.N, dtype=np.int32),
        }
        fields = {
            "contact_signed_distance": csd.astype(float),
            "cumulative_mass_source": np.zeros(1),
            "external_force": ext_force.astype(float),
        }
        scalars = {
            "contact_length_scale": float(self.spacing),
            "kinetic_energy": ke,
            "particle_count": float(self.N),
        }
        frame = StateFrame(
            time=t,
            position=p.copy(),
            velocity=v.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        self._last_frame = frame
        return frame

    def diagnostics(self) -> dict:
        p, v = self.p, self.v
        # heightfield-based mound / displaced-volume metrics
        cs = 0.15
        xs = np.arange(self.xlo, self.xhi, cs)
        ys = np.arange(self.ylo, self.yhi, cs)
        ix = np.clip(((p[:, 0] - self.xlo) / cs).astype(int), 0, len(xs) - 1)
        iy = np.clip(((p[:, 1] - self.ylo) / cs).astype(int), 0, len(ys) - 1)
        flat = iy * len(xs) + ix
        hmax = np.zeros(len(xs) * len(ys))
        np.maximum.at(hmax, flat, p[:, 2])
        hmax = hmax.reshape(len(ys), len(xs))
        base = self.bed_top
        displaced = float(np.sum(np.maximum(hmax - base, 0.0)) * cs * cs)
        mound_mask = hmax > base + 0.02
        mound_height = float(hmax.max())
        footprint = float(mound_mask.sum()) * cs * cs
        escaped = int(np.sum(
            (p[:, 0] < self.xlo - 1e-6) | (p[:, 0] > self.xhi + 1e-6)
            | (p[:, 1] < self.ylo - 1e-6) | (p[:, 1] > self.yhi + 1e-6)
            | (p[:, 2] < self.floor_z - 1e-6)
        ))
        ke = float(0.5 * np.sum(self.m * np.einsum("ij,ij->i", v, v)))
        mean_force = (self.tool_impulse / self.tool_contact_time
                      if self.tool_contact_time > 1e-9 else 0.0)
        return {
            "displaced_sand_volume": displaced,
            "mound_height": mound_height,
            "mound_footprint_area": footprint,
            "shovel_reaction_force_mean": float(mean_force),
            "shovel_reaction_force_peak": float(self.peak_tool_force),
            "shovel_impulse_total": float(self.tool_impulse),
            "escaped_particle_count": escaped,
            "settled_kinetic_energy": ke,
            "particle_count": int(self.N),
        }
