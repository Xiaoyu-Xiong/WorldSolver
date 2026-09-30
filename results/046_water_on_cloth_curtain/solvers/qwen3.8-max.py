"""Two-way coupled water jet / cloth curtain solver.

Water: weakly-compressible SPH particles (Tait EOS, spiky pressure gradient,
simple Laplacian viscosity), recycled through a prescribed nozzle inflow so the
stream is continuous while total water mass is conserved.
Cloth: position-based-dynamics mass-spring sheet (structural + shear + bending
distance constraints), top row pinned.
Coupling: impulse-based two-way water<->cloth contact (water loads the cloth;
moving cloth redirects the water), plus container walls and catch-tank floor.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


# --------------------------------------------------------------------------- #
# uniform spatial hash
# --------------------------------------------------------------------------- #
class _Grid:
    def __init__(self, lo, hi, cell):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.cell = float(cell)
        dims = np.ceil((np.asarray(hi, dtype=np.float64) - self.lo) / self.cell)
        self.dims = np.maximum(dims.astype(np.int64), 1)
        self.nx, self.ny, self.nz = (int(v) for v in self.dims)

    def _cell_of(self, pos):
        idx = np.floor((pos - self.lo) / self.cell).astype(np.int64)
        idx[:, 0] = np.clip(idx[:, 0], 0, self.nx - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, self.ny - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, self.nz - 1)
        return idx

    def build(self, pos):
        idx = self._cell_of(pos)
        cellid = (idx[:, 0] * self.ny + idx[:, 1]) * self.nz + idx[:, 2]
        order = np.argsort(cellid, kind="stable")
        self.order = order
        self.sorted = cellid[order]
        self.pos = pos
        return self

    def query_pairs(self, qpos):
        """Candidate pairs (i in qpos, j in self.pos) from the 27-cell stencil."""
        qidx = self._cell_of(qpos)
        Is, Js = [], []
        cum_cache = None
        for dxo in (-1, 0, 1):
            for dyo in (-1, 0, 1):
                for dzo in (-1, 0, 1):
                    nb = qidx + np.array((dxo, dyo, dzo), dtype=np.int64)
                    nb[:, 0] = np.clip(nb[:, 0], 0, self.nx - 1)
                    nb[:, 1] = np.clip(nb[:, 1], 0, self.ny - 1)
                    nb[:, 2] = np.clip(nb[:, 2], 0, self.nz - 1)
                    q = (nb[:, 0] * self.ny + nb[:, 1]) * self.nz + nb[:, 2]
                    lo_ = np.searchsorted(self.sorted, q, "left")
                    hi_ = np.searchsorted(self.sorted, q, "right")
                    cnt = hi_ - lo_
                    sel = cnt > 0
                    if not sel.any():
                        continue
                    cs = cnt[sel]
                    i_src = np.repeat(np.nonzero(sel)[0], cs)
                    starts = np.repeat(lo_[sel], cs)
                    if cum_cache is None:
                        pass
                    ar = np.arange(i_src.size) - np.repeat(
                        np.cumsum(cs) - cs, cs)
                    Js.append(self.order[starts + ar])
                    Is.append(i_src)
        if not Is:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        return np.concatenate(Is), np.concatenate(Js)


def _scatter3(idx, vals, n):
    out = np.zeros((n, 3))
    for c in range(3):
        out[:, c] = np.bincount(idx, weights=vals[:, c], minlength=n)
    return out


def _scatter1(idx, vals, n):
    return np.bincount(idx, weights=vals, minlength=n)


# --------------------------------------------------------------------------- #
class BenchmarkSolver:
    DT = 0.0025
    H = 0.14                 # SPH smoothing radius
    R_CLOTH = 0.085          # water-cloth contact radius
    RHO0 = 1000.0
    CS = 10.0                # artificial sound speed
    GAMMA = 7.0
    MU = 0.06                # dynamic viscosity
    NOZZLE_C = np.array((-1.235, 0.0, 1.72))
    NOZZLE_R = 0.185
    JET_V = 3.2
    JET_RADIUS = 0.20
    MASS_FLUX = 1000.0 * np.pi * 0.20 ** 2 * 3.2   # ~402 kg/s prescribed inflow
    W_LO = np.array((-1.55, -1.35, 0.012))
    W_HI = np.array((2.05, 1.35, 2.72))
    C_LO = np.array((-1.50, -1.34, 0.02))
    C_HI = np.array((2.05, 1.34, 2.75))

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed) + 4601)
        frame0 = scene.initial_state()
        pos = np.array(frame0.position, dtype=np.float64)
        body = np.asarray(frame0.body_id)
        self.Nc = int(np.sum(body == 0))
        self.Nw = int(np.sum(body == 1))
        self.cloth_x = np.array(pos[: self.Nc])
        self.cloth_v = np.array(frame0.velocity[: self.Nc])
        self.water_x = np.array(pos[self.Nc:])
        self.water_v = np.array(frame0.velocity[self.Nc:])
        self.m_cloth = np.array(frame0.mass[: self.Nc])
        self.m_water = np.array(frame0.mass[self.Nc:])
        self.pinned = np.asarray(frame0.attributes["pinned"][: self.Nc]) > 0.5
        self.pin_x = self.cloth_x[self.pinned].copy()
        self.pin_idx = np.nonzero(self.pinned)[0]
        self._cloth_x0 = self.cloth_x.copy()
        g = np.asarray(scene.config.get("gravity", (0.0, 0.0, -9.81)), dtype=np.float64)
        self.g = g

        # ---- cloth constraint sets from the 31 x 41 grid (index = iz*31+iy)
        ny, nz = 31, 41
        a_s, b_s = [], []
        for iz in range(nz):
            for iy in range(ny):
                i = iz * ny + iy
                if iy + 1 < ny:
                    a_s.append(i); b_s.append(i + 1)
                if iz + 1 < nz:
                    a_s.append(i); b_s.append(i + ny)
        a_d, b_d = [], []
        for iz in range(nz - 1):
            for iy in range(ny - 1):
                i = iz * ny + iy
                a_d.append(i); b_d.append(i + ny + 1)
                a_d.append(i + 1); b_d.append(i + ny)
        a_b, b_b = [], []
        for iz in range(nz):
            for iy in range(ny):
                i = iz * ny + iy
                if iy + 2 < ny:
                    a_b.append(i); b_b.append(i + 2)
                if iz + 2 < nz:
                    a_b.append(i); b_b.append(i + 2 * ny)
        sets = []
        for (aa, bb, k) in ((a_s, b_s, 1.0), (a_d, b_d, 1.0), (a_b, b_b, 0.35)):
            aa = np.asarray(aa); bb = np.asarray(bb)
            rest = np.linalg.norm(self.cloth_x[bb] - self.cloth_x[aa], axis=1)
            sets.append((aa, bb, rest, k))
        self.c_sets = sets
        self.w_cloth = np.where(self.pinned, 0.0, 1.0 / self.m_cloth)
        self.cloth_imp = np.zeros((self.Nc, 3))

        # ---- SPH constants
        h = self.H
        self.h2 = h * h
        self.POLY6 = 315.0 / (64.0 * np.pi * h ** 9)
        self.SPIKY = 45.0 / (np.pi * h ** 6)
        self.B = self.RHO0 * self.CS ** 2 / self.GAMMA

        self.grid_w = _Grid((-1.75, -1.55, -0.05), (2.30, 1.55, 2.90), h)
        self.grid_c = _Grid((-1.75, -1.55, -0.05), (2.30, 1.55, 2.90), h)

        self.t = 0.0
        self.step = 0
        self.interval_imp = np.zeros(3)     # water->cloth impulse accumulator
        self.interval_time = 0.0
        self.cum_recycled = 0.0
        self._last = self._make_frame(0.0)

    # ------------------------------------------------------------- one step
    def _simulate_step(self, dt):
        H = self.H
        h2 = self.h2
        Nw = self.Nw
        xw, vw, mw = self.water_x, self.water_v, self.m_water

        # ---------------- SPH density / pressure
        self.grid_w.build(xw)
        i, j = self.grid_w.query_pairs(xw)
        dx = xw[i] - xw[j]
        r2 = np.einsum("ij,ij->i", dx, dx)
        m = r2 < h2
        i, j, dx, r2 = i[m], j[m], dx[m], r2[m]
        r = np.sqrt(np.maximum(r2, 1e-12))
        w = self.POLY6 * (h2 - r2) ** 3
        rho = _scatter1(i, mw[j] * w, Nw)
        rho = np.maximum(rho, 300.0)
        pr = self.B * ((rho / self.RHO0) ** self.GAMMA - 1.0)
        pr = np.maximum(pr, 0.0)

        # ---------------- pressure + viscosity accelerations
        near = r > 1e-6
        ii, jj, dxx, rr = i[near], j[near], dx[near], r[near]
        coeff = self.SPIKY * (H - rr) ** 2 / rr
        pterm = mw[jj] * (pr[ii] / rho[ii] ** 2 + pr[jj] / rho[jj] ** 2) * coeff
        acc = _scatter3(ii, pterm[:, None] * dxx, Nw)
        visc = (self.MU / rho[ii]) * mw[jj] / rho[jj] * self.SPIKY * (H - rr)
        dv = vw[jj] - vw[ii]
        acc += _scatter3(ii, visc[:, None] * dv, Nw)
        acc += self.g

        vw += acc * dt
        np.clip(vw[:, 0], -25.0, 25.0, out=vw[:, 0])
        np.clip(vw[:, 1], -25.0, 25.0, out=vw[:, 1])
        np.clip(vw[:, 2], -25.0, 25.0, out=vw[:, 2])

        # ---------------- two-way water/cloth impulses
        self.cloth_imp[:] = 0.0
        self.grid_c.build(self.cloth_x)
        ci, cj = self.grid_c.query_pairs(xw)
        if ci.size:
            d = xw[ci] - self.cloth_x[cj]
            dist = np.sqrt(np.maximum(np.einsum("ij,ij->i", d, d), 1e-12))
            act = dist < self.R_CLOTH
            ci, cj, d, dist = ci[act], cj[act], d[act], dist[act]
            if ci.size:
                n = d / dist[:, None]
                vrel = vw[ci] - self.cloth_v[cj]
                vn = np.einsum("ij,ij->i", vrel, n)
                ap = vn < 0.0
                if ap.any():
                    cia, cja, na, vna = ci[ap], cj[ap], n[ap], vn[ap]
                    inv = 1.0 / mw[cia] + 1.0 / self.m_cloth[cja]
                    jn = -(1.0 + 0.0) * vna / inv          # normal impulse
                    # Coulomb-ish tangential damping
                    vt = vrel[ap] - vna[:, None] * na
                    vt_n = np.linalg.norm(vt, axis=1)
                    jt = np.minimum(0.15 * jn, vt_n / inv)
                    safe = vt_n > 1e-9
                    tt = np.zeros_like(vt)
                    tt[safe] = vt[safe] / vt_n[safe, None]
                    dw = (jn[:, None] * na + jt[:, None] * tt) / mw[cia, None]
                    vw[cia] += dw
                    imp = -(jn[:, None] * na + jt[:, None] * tt)
                    np.add.at(self.cloth_imp, cja, imp)
                # positional de-overlap of water only
                pen = np.maximum(self.R_CLOTH - dist, 0.0)
                xw[ci] += (0.7 * pen)[:, None] * n
        self.interval_imp += self.cloth_imp.sum(axis=0)
        self.interval_time += dt

        # ---------------- integrate water + walls
        xw += vw * dt
        lo, hi = self.W_LO, self.W_HI
        for c in range(3):
            below = xw[:, c] < lo[c]
            if below.any():
                xw[below, c] = lo[c]
                vcol = vw[below, c]
                vw[below, c] = np.where(vcol < 0.0, -0.2 * vcol, vcol)
                for t_ in range(3):
                    if t_ != c:
                        vw[below, t_] *= 0.995
            above = xw[:, c] > hi[c]
            if above.any():
                xw[above, c] = hi[c]
                vcol = vw[above, c]
                vw[above, c] = np.where(vcol > 0.0, -0.2 * vcol, vcol)

        # ---------------- prescribed nozzle inflow via recycling (mass-conserving)
        budget = self.MASS_FLUX * dt
        speed = np.linalg.norm(vw, axis=1)
        elig = np.nonzero(
            (xw[:, 2] < 0.34) & (xw[:, 2] > 0.02) & (speed < 0.7)
            & (xw[:, 0] > -1.30) & (np.abs(xw[:, 1]) < 1.30)
        )[0]
        if elig.size:
            pick = self.rng.permutation(elig)
            msel = mw[pick]
            cum = np.cumsum(msel)
            k = int(np.searchsorted(cum, budget) + 1)
            k = min(k, pick.size)
            sel = pick[:k]
            if sel.size:
                rad = self.NOZZLE_R * np.sqrt(self.rng.random(sel.size))
                ang = self.rng.random(sel.size) * 2.0 * np.pi
                xw[sel, 0] = self.NOZZLE_C[0]
                xw[sel, 1] = self.NOZZLE_C[1] + rad * np.cos(ang)
                xw[sel, 2] = self.NOZZLE_C[2] + rad * np.sin(ang)
                vw[sel] = np.array((self.JET_V, 0.0, 0.0))
                self.cum_recycled += float(mw[sel].sum())

        # ---------------- cloth PBD step
        xc, vc = self.cloth_x, self.cloth_v
        vc += self.g * dt
        vc += self.cloth_imp / self.m_cloth[:, None]
        sp = np.linalg.norm(vc, axis=1)
        over = sp > 20.0
        if over.any():
            vc[over] *= (20.0 / sp[over])[:, None]
        vc *= 0.9995
        p = xc + vc * dt
        p[self.pin_idx] = self.pin_x
        for _ in range(6):
            for aa, bb, rest, k in self.c_sets:
                d = p[bb] - p[aa]
                dist = np.sqrt(np.maximum(np.einsum("ij,ij->i", d, d), 1e-14))
                corr = (k * (dist - rest) / dist)[:, None] * d
                wa = self.w_cloth[aa]
                wb = self.w_cloth[bb]
                wsum = np.maximum(wa + wb, 1e-12)
                np.add.at(p, aa, (wa / wsum)[:, None] * corr)
                np.add.at(p, bb, -(wb / wsum)[:, None] * corr)
            p[self.pin_idx] = self.pin_x
            np.clip(p[:, 0], self.C_LO[0], self.C_HI[0], out=p[:, 0])
            np.clip(p[:, 1], self.C_LO[1], self.C_HI[1], out=p[:, 1])
            np.clip(p[:, 2], self.C_LO[2], self.C_HI[2], out=p[:, 2])
        vc[:] = (p - xc) / dt
        xc[:] = p
        vc[self.pin_idx] = 0.0

        self.t += dt
        self.step += 1

    # ---------------------------------------------------------------- frame
    def _make_frame(self, time):
        Nc, Nw = self.Nc, self.Nw
        N = Nc + Nw
        pos = np.vstack((self.cloth_x, self.water_x))
        vel = np.vstack((self.cloth_v, self.water_v))
        mass = np.concatenate((self.m_cloth, self.m_water))
        body = np.concatenate(
            (np.zeros(Nc, dtype=np.int32), np.ones(Nw, dtype=np.int32)))
        pinned = np.zeros(N)
        pinned[:Nc] = self.pinned.astype(np.float64)

        mass_quantity = np.zeros((N, 2))
        mass_quantity[:Nc, 0] = self.m_cloth
        mass_quantity[Nc:, 1] = self.m_water

        ext = np.zeros((N, 3))
        ext += mass[:, None] * self.g

        # interaction force averaged over the last sample interval
        if self.interval_time > 0.0:
            F = self.interval_imp / self.interval_time
        else:
            F = np.zeros(3)
        inter = np.vstack((F, -F))
        self.interval_imp = np.zeros(3)
        self.interval_time = 0.0

        # signed contact gaps for water samples near the cloth
        self.grid_c.build(self.cloth_x)
        ci, cj = self.grid_c.query_pairs(self.water_x)
        margin = self.R_CLOTH + 0.05
        if ci.size:
            dd = np.linalg.norm(self.water_x[ci] - self.cloth_x[cj], axis=1)
            sel = dd < margin
            gaps = dd[sel] - self.R_CLOTH
            if gaps.size > 40000:
                gaps = gaps[:: max(1, gaps.size // 40000)]
        else:
            gaps = np.zeros(0)
        if gaps.size == 0:
            gaps = np.array((margin,))

        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=body,
            attributes={
                "pinned": pinned,
                "material_phase": body.astype(np.float64),
                "mass_quantity": mass_quantity,
            },
            fields={
                "external_force": ext,
                "interaction_force": inter,
                "contact_signed_distance": gaps,
                "cumulative_mass_source": np.zeros(2),
            },
            scalars={
                "water_density": self.RHO0,
                "cloth_total_mass": float(self.m_cloth.sum()),
                "cloth_thickness": 0.0015,
                "nozzle_radius": self.JET_RADIUS,
                "nominal_inflow_speed": self.JET_V,
                "contact_length_scale": 0.0613,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = int(round(float(time) / self.DT))
        guard = 0
        while self.step < target and guard < 4 * (target - self.step + 2):
            self._simulate_step(self.DT)
            guard += 1
        self._last = self._make_frame(time)
        return self._last

    # ---------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict:
        down = np.sum(self.water_x[:, 0] > 0.30)
        disp = np.linalg.norm(self.cloth_x - self._cloth_x0, axis=1) \
            if hasattr(self, "_cloth_x0") else np.zeros(self.Nc)
        return {
            "leakage_across_cloth_particles": int(down),
            "cloth_max_displacement": float(np.max(disp)) if self.Nc else 0.0,
            "water_total_mass": float(self.m_water.sum()),
            "cumulative_recycled_mass": float(self.cum_recycled),
            "steps": int(self.step),
        }
