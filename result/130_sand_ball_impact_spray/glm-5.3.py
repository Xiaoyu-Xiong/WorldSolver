"""Granular impact solver for scene ``130_sand_ball_impact_spray``.

Physics
-------
The sand bed is integrated as a dissipative granular medium with an explicit
symplectic-Euler discrete-element scheme:

* grain-grain contacts: linear spring-dashpot normal force with Coulomb
  friction (tangential force capped by mu * N);
* grain-ball contacts: the same contact law against the rigid spherical
  impactor, whose own translational dynamics emerge from the contact forces
  (equal and opposite) plus gravity;
* static boundaries: rigid container floor (top face of the ground box at
  z = 0) and the frozen particle-domain side walls, with normal damping and
  Coulomb friction;
* gravity from the frozen configuration.

Per-sample deformation gradients of the granular medium are recovered as the
least-squares best-fit affine map between the initial and the current
neighbourhood displacement of each grain, evaluated on the frozen sample
grid.  The scheme is deterministic and produces finite states at every
requested sample time.
"""
from __future__ import annotations

import itertools

import numpy as np

from worldsolver.contracts import StateFrame

# Half-space of cell offsets (lexicographically non-negative) so every
# unordered neighbouring-cell pair is scanned exactly once.
_NEIGHBOR_OFFSETS = tuple(
    o for o in itertools.product((-1, 0, 1), repeat=3) if o >= (0, 0, 0)
)


class BenchmarkSolver:
    """Explicit DEM solver: rigid ball impact into a dense sand bed."""

    SUBSTEPS_PER_SAMPLE = 160
    NEIGHBOR_RADIUS = 0.30  # neighbourhood used for deformation-gradient fit
    NEAR_CONTACT_BAND = 0.15
    WALL_X = 3.0
    WALL_Y = 1.2
    Z_TOP = 3.8

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame0 = scene.initial_state()

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.dt = float(cfg["sample_dt"]) / self.SUBSTEPS_PER_SAMPLE
        spacing = float(cfg["sampling"]["spacing"])
        density = float(cfg["sampling"]["density"])

        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.body = np.array(frame0.body_id, dtype=np.int32)
        self.X0 = self.pos.copy()
        self.n = self.pos.shape[0]
        self.steps = 0
        self.time = 0.0

        is_sand = self.body == 0
        is_ball = self.body == 1
        self.idx_sand = np.nonzero(is_sand)[0]
        self.idx_ball = np.nonzero(is_ball)[0]
        self._ball_index = int(self.idx_ball[0]) if len(self.idx_ball) else -1

        # Rigid ball radius from its frozen mass and the medium density.
        m_ball = float(self.mass[self.idx_ball[0]])
        self.r_ball = float((3.0 * m_ball / (4.0 * np.pi * density)) ** (1.0 / 3.0))
        pr = np.asarray(frame0.attributes["particle_radius"], dtype=float)
        self.r_sand = float(pr[self.idx_sand[0]])
        self.radius = np.where(is_ball, self.r_ball, self.r_sand).astype(float)

        # Contact parameters (linear spring-dashpot + Coulomb friction).
        m_sand = float(self.mass[self.idx_sand].min())
        self.k = 3.0e5  # grain-grain / grain-ball normal stiffness [N/m]
        self.cn = 2.0 * 0.35 * np.sqrt(self.k * 0.5 * m_sand)
        self.ct = 1.5 * self.cn
        self.k_ball_floor = 1.0e6
        self.c_ball_floor = 2.0 * 0.6 * np.sqrt(self.k_ball_floor * m_ball)
        self.kf = np.where(is_ball, self.k_ball_floor, self.k)
        self.cf = np.where(is_ball, self.c_ball_floor, self.cn)
        self.mu_ss = 0.55   # sand-sand friction
        self.mu_bs = 0.45   # ball-sand friction
        self.mu_wall = 0.50  # boundary friction
        self.drag = 0.05    # aerodynamic velocity damping [1/s]
        self.vmax = 20.0    # numerical safety clamp [m/s]

        self.attr_radius = pr.copy()
        self.scalars0 = dict(frame0.scalars)
        self.scalars0["contact_length_scale"] = float(spacing)

        # Uniform hash grids for broad-phase contact search.
        origin = np.array([-3.3, -1.5, -0.3])
        extent = np.array([6.6, 3.0, 5.3])
        self.ss_contact = 2.0 * self.r_sand
        self.ss_cell = max(0.12, self.ss_contact)
        self.ss_origin = origin
        self.ss_dims = tuple(int(np.ceil(e / self.ss_cell)) for e in extent)
        self.fd_cell = float(self.NEIGHBOR_RADIUS)
        self.fd_origin = origin
        self.fd_dims = tuple(int(np.ceil(e / self.fd_cell)) for e in extent)

    # ------------------------------------------------------------------
    # broad phase: candidate pairs in a uniform hash grid
    # ------------------------------------------------------------------
    @staticmethod
    def _pairs(p, origin, dims, cell):
        nx, ny, nz = dims
        inv = 1.0 / cell
        ix = np.clip(((p[:, 0] - origin[0]) * inv).astype(np.int64), 0, nx - 1)
        iy = np.clip(((p[:, 1] - origin[1]) * inv).astype(np.int64), 0, ny - 1)
        iz = np.clip(((p[:, 2] - origin[2]) * inv).astype(np.int64), 0, nz - 1)
        cid = (ix * ny + iy) * nz + iz
        ncell = nx * ny * nz
        order = np.argsort(cid, kind="stable")
        counts = np.bincount(cid, minlength=ncell)
        starts = np.zeros(ncell + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        n = p.shape[0]
        ar = np.arange(n)

        out_i = []
        out_j = []
        for dx, dy, dz in _NEIGHBOR_OFFSETS:
            if dx or dy or dz:
                nix = ix + dx
                niy = iy + dy
                niz = iz + dz
                valid = (
                    (nix >= 0) & (nix < nx) & (niy >= 0) & (niy < ny)
                    & (niz >= 0) & (niz < nz)
                )
                if not valid.any():
                    continue
                nb = (nix * ny + niy) * nz + niz
                nbv = nb[valid]
                cnt = counts[nbv]
                tot = int(cnt.sum())
                if tot == 0:
                    continue
                vidx = ar[valid]
                ii = np.repeat(vidx, cnt)
                base = np.repeat(starts[nbv], cnt)
                csum = np.cumsum(cnt) - cnt
                offs = np.arange(tot) - np.repeat(csum, cnt)
                jj = order[base + offs]
                out_i.append(ii)
                out_j.append(jj)
            else:
                cnt = counts[cid]
                tot = int(cnt.sum())
                if tot == 0:
                    continue
                ii = np.repeat(ar, cnt)
                base = np.repeat(starts[cid], cnt)
                csum = np.cumsum(cnt) - cnt
                offs = np.arange(tot) - np.repeat(csum, cnt)
                jj = order[base + offs]
                keep = ii < jj
                out_i.append(ii[keep])
                out_j.append(jj[keep])
        if not out_i:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        return np.concatenate(out_i), np.concatenate(out_j)

    # ------------------------------------------------------------------
    # one explicit substep
    # ------------------------------------------------------------------
    def _step(self):
        pos, vel, mass = self.pos, self.vel, self.mass
        dt = self.dt
        f = mass[:, None] * self.gravity[None, :]

        s = self.idx_sand
        ps = pos[s]
        vs = vel[s]
        floc = np.zeros((s.shape[0], 3))

        # ---- grain-grain contacts ------------------------------------
        i, j = self._pairs(ps, self.ss_origin, self.ss_dims, self.ss_cell)
        if i.size:
            d = ps[j] - ps[i]
            dist2 = np.einsum("ij,ij->i", d, d)
            act = (dist2 < self.ss_contact ** 2) & (dist2 > 1e-16)
            if act.any():
                i = i[act]
                j = j[act]
                dist = np.sqrt(dist2[act])
                n = d[act] / dist[:, None]
                vrel = vs[j] - vs[i]
                vn = np.einsum("ij,ij->i", vrel, n)
                fmag = self.k * (self.ss_contact - dist) - self.cn * vn
                np.maximum(fmag, 0.0, out=fmag)
                vt = vrel - vn[:, None] * n
                vtm = np.linalg.norm(vt, axis=1)
                ftm = np.minimum(self.mu_ss * fmag, self.ct * vtm)
                sc = np.where(vtm > 1e-9, ftm / np.maximum(vtm, 1e-12), 0.0)
                fpair = fmag[:, None] * n - sc[:, None] * vt  # force on j
                np.add.at(floc, j, fpair)
                np.add.at(floc, i, -fpair)

        # ---- grain-ball contacts -------------------------------------
        if self._ball_index >= 0:
            b = self._ball_index
            bp = pos[b]
            bv = vel[b]
            d = bp[None, :] - ps
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            rsb = self.r_ball + self.r_sand
            act = (dist < rsb) & (dist > 1e-12)
            if act.any():
                nn = d[act] / dist[act][:, None]
                vrel = bv[None, :] - vs[act]
                vn = np.einsum("ij,ij->i", vrel, nn)
                fmag = self.k * (rsb - dist[act]) - self.cn * vn
                np.maximum(fmag, 0.0, out=fmag)
                vt = vrel - vn[:, None] * nn
                vtm = np.linalg.norm(vt, axis=1)
                ftm = np.minimum(self.mu_bs * fmag, self.ct * vtm)
                sc = np.where(vtm > 1e-9, ftm / np.maximum(vtm, 1e-12), 0.0)
                fb = fmag[:, None] * nn - sc[:, None] * vt  # force on ball
                floc[act] -= fb
                f[b] += fb.sum(axis=0)

        f[s] += floc

        # ---- container floor (top face of the ground box, z = 0) -----
        r = self.radius
        gap = r - pos[:, 2]
        act = gap > 0.0
        if act.any():
            fz = self.kf[act] * gap[act] - self.cf[act] * vel[act, 2]
            np.maximum(fz, 0.0, out=fz)
            f[act, 2] += fz
            tv = vel[act, :2]
            vtm = np.linalg.norm(tv, axis=1)
            ftm = np.minimum(self.mu_wall * fz, self.ct * vtm)
            sc = np.where(vtm > 1e-9, ftm / np.maximum(vtm, 1e-12), 0.0)
            f[act, :2] -= sc[:, None] * tv

        # ---- domain side walls ---------------------------------------
        for axis, lim in ((0, self.WALL_X), (1, self.WALL_Y)):
            xa = pos[:, axis]
            sgn = np.sign(xa)
            sgn[sgn == 0] = 1.0
            gapw = (lim - r) - np.abs(xa)
            act = gapw < 0.0
            if act.any():
                idx = np.nonzero(act)[0]
                pen = -gapw[idx]
                vn = sgn[idx] * vel[idx, axis]
                fm = self.k * pen - self.cn * vn
                np.maximum(fm, 0.0, out=fm)
                f[idx, axis] += sgn[idx] * fm
                others = [a for a in (0, 1, 2) if a != axis]
                oa = np.asarray(others)
                tv = vel[idx][:, others]
                vtm = np.linalg.norm(tv, axis=1)
                ftm = np.minimum(self.mu_wall * fm, self.ct * vtm)
                sc = np.where(vtm > 1e-9, ftm / np.maximum(vtm, 1e-12), 0.0)
                f[idx[:, None], oa[None, :]] -= sc[:, None] * tv

        # ---- soft ceiling of the declared particle domain ------------
        gapc = (self.Z_TOP - r) - pos[:, 2]
        act = gapc < 0.0
        if act.any():
            idx = np.nonzero(act)[0]
            pen = -gapc[idx]
            vn = -vel[idx, 2]
            fm = self.k * pen - self.cn * vn
            np.maximum(fm, 0.0, out=fm)
            f[idx, 2] -= fm
            tv = vel[idx, :2]
            vtm = np.linalg.norm(tv, axis=1)
            ftm = np.minimum(self.mu_wall * fm, self.ct * vtm)
            sc = np.where(vtm > 1e-9, ftm / np.maximum(vtm, 1e-12), 0.0)
            f[idx, :2] -= sc[:, None] * tv

        # ---- integrate (symplectic Euler) ----------------------------
        acc = f / mass[:, None]
        vel += acc * dt
        vel[s] *= 1.0 - self.drag * dt
        sp = np.sqrt(np.einsum("ij,ij->i", vel, vel))
        over = sp > self.vmax
        if over.any():
            vel[over] *= (self.vmax / sp[over])[:, None]
        pos += vel * dt
        self.steps += 1
        self.time = self.steps * self.dt

    # ------------------------------------------------------------------
    # per-sample deformation gradient (best-fit affine neighbourhood map)
    # ------------------------------------------------------------------
    def _deformation_gradient(self):
        F = np.tile(np.eye(3), (self.n, 1, 1))
        s = self.idx_sand
        if s.size == 0:
            return F
        ps = self.pos[s]
        Xs = self.X0[s]
        i, j = self._pairs(ps, self.fd_origin, self.fd_dims, self.fd_cell)
        if i.size:
            d = ps[j] - ps[i]
            dist2 = np.einsum("ij,ij->i", d, d)
            act = (dist2 < self.NEIGHBOR_RADIUS ** 2) & (dist2 > 1e-16)
            if act.any():
                i = i[act]
                j = j[act]
                d = d[act]
                dist = np.sqrt(dist2[act])
                dX = Xs[j] - Xs[i]
                w = 1.0 / (0.02 + dist)
                ns = s.shape[0]
                A = np.zeros((ns, 3, 3))
                B = np.zeros((ns, 3, 3))
                for a in range(3):
                    for b in range(3):
                        ca = w * dX[:, a] * dX[:, b]
                        np.add.at(A[:, a, b], i, ca)
                        np.add.at(A[:, a, b], j, ca)
                        cb = w * d[:, a] * dX[:, b]
                        np.add.at(B[:, a, b], i, cb)
                        np.add.at(B[:, a, b], j, cb)
                cnt = np.zeros(ns, dtype=np.int64)
                np.add.at(cnt, i, 1)
                np.add.at(cnt, j, 1)
                ok = cnt >= 4
                if ok.any():
                    eye = np.eye(3)
                    Ai = A[ok] + 0.05 * eye[None, :, :]
                    Bi = B[ok]
                    Fi = np.linalg.solve(Ai, np.swapaxes(Bi, 1, 2))
                    Fi = np.swapaxes(Fi, 1, 2)
                    np.clip(Fi, -10.0, 10.0, out=Fi)
                    F[s[ok]] = np.nan_to_num(Fi, nan=0.0, posinf=0.0, neginf=0.0)
        return F

    # ------------------------------------------------------------------
    # signed geometric gap to the nearest obstacle per sample
    # ------------------------------------------------------------------
    def _contact_gaps(self):
        gap = np.full(self.n, 1.0e3)
        s = self.idx_sand
        ps = self.pos[s]
        rs = self.r_sand
        g_floor = ps[:, 2] - rs
        if self._ball_index >= 0:
            dball = np.linalg.norm(ps - self.pos[self._ball_index][None, :], axis=1)
            g_ball = dball - (self.r_ball + rs)
        else:
            g_ball = np.full(s.shape[0], 1.0e3)
        g_wall = np.minimum(
            (self.WALL_X - rs) - np.abs(ps[:, 0]),
            (self.WALL_Y - rs) - np.abs(ps[:, 1]),
        )
        g_ceil = (self.Z_TOP - rs) - ps[:, 2]
        gap[s] = np.minimum(np.minimum(g_floor, g_ball), np.minimum(g_wall, g_ceil))
        if self._ball_index >= 0:
            b = self._ball_index
            bp = self.pos[b]
            rb = self.r_ball
            gb = min(
                float(bp[2] - rb),
                float((self.WALL_X - rb) - abs(bp[0])),
                float((self.WALL_Y - rb) - abs(bp[1])),
                float((self.Z_TOP - rb) - bp[2]),
            )
            gap[b] = gb
        return gap

    # ------------------------------------------------------------------
    # sampled state frame
    # ------------------------------------------------------------------
    def _frame(self, t):
        F = self._deformation_gradient()
        gap = self._contact_gaps()
        sel = np.nonzero(gap < self.NEAR_CONTACT_BAND)[0]
        csd = gap[sel]
        if csd.size == 0:
            csd = np.array([float(gap.min())])
        ext = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(t),
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": self.attr_radius.copy(),
                "material_label": self.body.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1),
                "external_force": ext,
                "contact_signed_distance": csd,
            },
            scalars=dict(self.scalars0),
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    # solver contract
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target_steps = int(round(float(time) / self.dt))
        if target_steps < self.steps:
            target_steps = self.steps
        while self.steps < target_steps:
            self._step()
        return self._frame(float(time))

    def diagnostics(self) -> dict:
        out = {
            "method": "dem_spring_dashpot_coulomb",
            "dt": float(self.dt),
            "steps": int(self.steps),
            "time": float(self.time),
            "num_sand": int(self.idx_sand.size),
        }
        if self._ball_index >= 0:
            out["ball_height"] = float(self.pos[self._ball_index, 2])
        out["max_speed"] = float(np.max(np.linalg.norm(self.vel, axis=1)))
        return out
