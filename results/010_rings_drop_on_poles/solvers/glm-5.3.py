"""Rigid-body solver: nonconvex rings dropped onto a vertical pole array.

Physics: impulse-based contact dynamics with Coulomb friction, substepped
semi-implicit (symplectic Euler) integration, quaternion orientation
dynamics, and a sampled-sphere representation of the frozen nonconvex
axisymmetric ring profile.  Statics: ground box top plane and nine vertical
pole cylinders.  All motion is produced by the time-stepping procedure.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is expected to exist
    cKDTree = None

try:
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover
    StateFrame = object

# ----- frozen scene constants (scene_geometry.yaml) -----
GROUND_TOP_Z = 0.0
POLE_R = 0.075
POLE_TOP = 2.6
POLE_XY = np.array([[x, y] for x in (-0.9, 0.0, 0.9) for y in (-0.9, 0.0, 0.9)])

WALL_T = 0.06            # ring wall thickness (contact length scale)
RS = 0.5 * WALL_T        # contact sphere radius
# Cross-section midline (r, z) of the axisymmetric ring wall, local frame.
PROFILE_RZ = np.array([
    [0.21, 0.00], [0.31, 0.02], [0.37, 0.08],
    [0.37, 0.18], [0.31, 0.24], [0.21, 0.26],
])
Z_CENTER = 0.13          # local COM height (by symmetry)
N_ANG = 16               # angular samples around the ring

G = 9.81
N_SUB = 8                # substeps per sample_dt
MARGIN = 0.02            # contact detection margin
SLOP = 0.004
BETA = 0.25              # Baumgarte penetration recovery factor
MAX_BIAS = 2.0           # cap on separation bias velocity (m/s)
N_ITER = 18              # contact solver iterations (relaxed Jacobi)
RELAX = 0.75
MU = 0.45                # Coulomb friction coefficient
CAP_PER_PAIR = 8         # max sphere contacts kept per body pair
LIN_DAMP = 0.08          # 1/s
ANG_DAMP = 0.25          # 1/s
VMAX = 25.0
WMAX = 80.0


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """xyzw quaternions to rotation matrices, shape (N,3,3)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    R = np.empty((n, 3, 3))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return R


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product for xyzw quaternions, shape (N,4)."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out = np.empty_like(a)
    out[:, 3] = aw * bw - ax * bx - ay * by - az * bz
    out[:, 0] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 1] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 2] = aw * bz + ax * by - ay * bx + az * bw
    return out


def _axis_angle_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    s = math.sin(0.5 * angle)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * angle)])


class BenchmarkSolver:
    """Impulse-based rigid-body contact solver for the rings-on-poles scene."""

    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", {}) or {}
        sample_dt = float(cfg.get("sample_dt", 0.02))
        self.sample_dt = sample_dt
        self.dt = sample_dt / N_SUB
        self.t = 0.0
        self.step_index = 0

        frame = scene.initial_state()
        self.pos = np.array(frame.position, dtype=float).copy()
        self.vel = np.array(frame.velocity, dtype=float).copy()
        n = self.pos.shape[0]
        self.N = n
        if frame.orientation is not None:
            self.quat = np.array(frame.orientation, dtype=float).copy()
        else:
            self.quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
        self.quat /= np.linalg.norm(self.quat, axis=1, keepdims=True)
        if frame.angular_velocity is not None:
            self.omega = np.array(frame.angular_velocity, dtype=float).copy()
        else:
            self.omega = np.zeros((n, 3))
        if frame.mass is not None:
            self.mass = np.array(frame.mass, dtype=float).copy()
        else:
            self.mass = np.ones(n)
        self.body_id = np.arange(n, dtype=np.int64)

        # Deterministic small perturbations (break perfect symmetries).
        rng = np.random.default_rng(int(seed))
        for i in range(n):
            axis = rng.normal(size=3)
            na = np.linalg.norm(axis)
            if na < 1e-9:
                axis = np.array([0.0, 0.0, 1.0])
            else:
                axis = axis / na
            dq = _axis_angle_quat(axis, rng.uniform(0.0, 0.2))
            qi = _quat_mul(dq[None, :], self.quat[i][None, :])[0]
            self.quat[i] = qi / np.linalg.norm(qi)
            self.omega[i] = rng.normal(scale=0.25, size=3)

        # Sampled-sphere template of the nonconvex ring (local, COM at origin).
        ang = np.arange(N_ANG) * (2.0 * math.pi / N_ANG)
        ca, sa = np.cos(ang), np.sin(ang)
        pts = []
        for r, z in PROFILE_RZ:
            for k in range(N_ANG):
                pts.append((r * ca[k], r * sa[k], z - Z_CENTER))
        self.template = np.asarray(pts, dtype=float)   # (K,3)
        self.K = self.template.shape[0]

        # Body-frame unit-mass inertia of the sample cloud.
        p = self.template
        I_unit = np.zeros((3, 3))
        ms = 1.0 / self.K
        for k in range(self.K):
            pk = p[k]
            I_unit += ms * (np.dot(pk, pk) * np.eye(3) - np.outer(pk, pk))
        I_unit[np.diag_indices(3)] += 1e-6
        self.I_unit = I_unit

        # Constant external (gravity) force per body.
        self.ext_force = np.zeros((n, 3))
        self.ext_force[:, 2] = -G * self.mass

        self.Iinv_world = np.zeros((n, 3, 3))
        self._report_gap = np.zeros(0)
        self._report_force = np.zeros((0, 3))

        # Initial contact report (no impulses applied yet).
        R = _quat_to_mat(self.quat)
        pts0 = np.einsum("nij,kj->nki", R, self.template) + self.pos[:, None, :]
        con = self._detect(pts0)
        if con is not None:
            self._report_gap = con["gap"].copy()
            self._report_force = np.zeros((len(con["gap"]), 3))

    # ------------------------------------------------------------------ contacts
    def _detect(self, pts: np.ndarray):
        """Detect contacts. Returns dict with cA, cB, n, gap, point arrays."""
        n_bodies, K = self.N, self.K
        flat = pts.reshape(-1, 3)
        lists = {"cA": [], "cB": [], "n": [], "gap": [], "point": []}

        # --- ring vs ring (sphere-sphere) ---
        radius = 2.0 * RS + MARGIN
        pairs = self._sphere_pairs(flat, radius, K)
        if pairs is not None and len(pairs[0]):
            ia, ib = pairs
            ra_id = ia // K
            rb_id = ib // K
            m = ra_id != rb_id
            ia, ib, ra_id, rb_id = ia[m], ib[m], ra_id[m], rb_id[m]
            if len(ia):
                swap = ra_id > rb_id
                ia2 = np.where(swap, ib, ia)
                ib2 = np.where(swap, ia, ib)
                lo = np.minimum(ra_id, rb_id)
                hi = np.maximum(ra_id, rb_id)
                ra_id, rb_id = lo, hi
                d = flat[ib2] - flat[ia2]
                dist = np.linalg.norm(d, axis=1)
                good = dist > 1e-9
                gap = dist - 2.0 * RS
                m = good & (gap < MARGIN)
                if m.any():
                    ia2, ib2 = ia2[m], ib2[m]
                    ra_id, rb_id = ra_id[m], rb_id[m]
                    d, dist, gap = d[m], dist[m], gap[m]
                    nrm = d / dist[:, None]
                    # limit contacts per body pair
                    key = ra_id * n_bodies + rb_id
                    order = np.lexsort((gap, key))
                    ks = key[order]
                    if len(ks):
                        new_grp = np.empty(len(ks), dtype=bool)
                        new_grp[0] = True
                        new_grp[1:] = ks[1:] != ks[:-1]
                        gid = np.cumsum(new_grp) - 1
                        first = np.flatnonzero(new_grp)
                        pos = np.arange(len(ks)) - first[gid]
                        sel = order[pos < CAP_PER_PAIR]
                        ia2, ib2 = ia2[sel], ib2[sel]
                        ra_id, rb_id = ra_id[sel], rb_id[sel]
                        nrm, gap = nrm[sel], gap[sel]
                    point = flat[ia2] + RS * nrm
                    lists["cA"].append(ra_id)
                    lists["cB"].append(rb_id)
                    lists["n"].append(nrm)
                    lists["gap"].append(gap)
                    lists["point"].append(point)

        # --- ring vs pole (capped vertical cylinders) ---
        reach = 0.40 + POLE_R + RS + MARGIN
        for px, py in POLE_XY:
            ring_mask = (
                np.hypot(self.pos[:, 0] - px, self.pos[:, 1] - py) < reach
            )
            if not ring_mask.any():
                continue
            ring_idx = np.flatnonzero(ring_mask)
            sub = pts[ring_mask]                       # (M,K,3)
            dx = sub[..., 0] - px
            dy = sub[..., 1] - py
            dr = np.hypot(dx, dy)
            ux = dx / np.maximum(dr, 1e-9)
            uy = dy / np.maximum(dr, 1e-9)
            z = sub[..., 2]
            inside = (z > GROUND_TOP_Z) & (z < POLE_TOP)
            cz = np.clip(z, GROUND_TOP_Z, POLE_TOP)
            radial = np.where(inside | (dr >= POLE_R), POLE_R, 0.0)
            cpx = px + radial * ux
            cpy = py + radial * uy
            vx = sub[..., 0] - cpx
            vy = sub[..., 1] - cpy
            vz = sub[..., 2] - cz
            dist = np.sqrt(vx * vx + vy * vy + vz * vz)
            gap = dist - RS
            m = (gap < MARGIN) & (dist > 1e-9)
            if m.any():
                rows, cols = np.nonzero(m)
                centers = sub[rows, cols]
                dvec = np.stack([-vx[rows, cols], -vy[rows, cols], -vz[rows, cols]], axis=1)
                nrm = dvec / dist[rows, cols][:, None]
                gap_p = gap[rows, cols]
                point = centers + RS * nrm
                lists["cA"].append(ring_idx[rows])
                lists["cB"].append(np.full(len(rows), n_bodies, dtype=np.int64))
                lists["n"].append(nrm)
                lists["gap"].append(gap_p)
                lists["point"].append(point)

        # --- ring vs ground (top plane) ---
        m = pts[..., 2] < (RS + MARGIN)
        if m.any():
            rows, cols = np.nonzero(m)
            centers = pts[rows, cols]
            gap_g = centers[:, 2] - RS
            nrm = np.zeros((len(rows), 3))
            nrm[:, 2] = -1.0
            point = centers + RS * nrm
            lists["cA"].append(rows.astype(np.int64))
            lists["cB"].append(np.full(len(rows), n_bodies, dtype=np.int64))
            lists["n"].append(nrm)
            lists["gap"].append(gap_g)
            lists["point"].append(point)

        if not lists["cA"]:
            return None
        return {
            "cA": np.concatenate(lists["cA"]),
            "cB": np.concatenate(lists["cB"]),
            "n": np.concatenate(lists["n"]),
            "gap": np.concatenate(lists["gap"]),
            "point": np.concatenate(lists["point"]),
        }

    def _sphere_pairs(self, flat, radius, K):
        """Pairs of sphere indices within radius, as (ia, ib) arrays."""
        if cKDTree is not None:
            tree = cKDTree(flat)
            pairs = tree.query_pairs(radius, output_type="ndarray")
            if len(pairs) == 0:
                return np.zeros(0, np.int64), np.zeros(0, np.int64)
            return pairs[:, 0], pairs[:, 1]
        # Fallback: ring-center broadphase + per-pair numpy distances.
        n = self.N
        centers = flat.reshape(n, K, 3).mean(axis=1)
        ia_all, ib_all = [], []
        lim = 2.0 * 0.40 + radius
        for i in range(n):
            d = np.linalg.norm(centers[i + 1:] - centers[i], axis=1)
            for j_rel in np.flatnonzero(d < lim):
                j = i + 1 + j_rel
                a = flat[i * K:(i + 1) * K]
                b = flat[j * K:(j + 1) * K]
                diff = a[:, None, :] - b[None, :, :]
                dist = np.sqrt((diff * diff).sum(-1))
                rr, cc = np.nonzero(dist < radius)
                if len(rr):
                    ia_all.append(rr + i * K)
                    ib_all.append(cc + j * K)
        if not ia_all:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        return np.concatenate(ia_all), np.concatenate(ib_all)

    # ------------------------------------------------------------------ solver
    @staticmethod
    def _denom(invm_a, invm_b, Ii_a, Ii_b, ra, rb, d):
        rax = np.cross(ra, d)
        aa = np.einsum("nij,nj->ni", Ii_a, rax)
        t_a = np.einsum("ni,ni->n", np.cross(aa, ra), d)
        rbx = np.cross(rb, d)
        ab = np.einsum("nij,nj->ni", Ii_b, rbx)
        t_b = np.einsum("ni,ni->n", np.cross(ab, rb), d)
        return invm_a + invm_b + t_a + t_b

    def _solve(self, con, dt):
        n = self.N
        p = n + 1
        cA = con["cA"].astype(np.int64)
        cB = con["cB"].astype(np.int64)
        nrm = con["n"]
        gap = con["gap"]
        point = con["point"]
        C = len(cA)
        if C == 0:
            return

        invm_pad = np.zeros(p)
        invm_pad[:n] = 1.0 / self.mass
        Iinv_pad = np.zeros((p, 3, 3))
        Iinv_pad[:n] = self.Iinv_world

        invm_a = invm_pad[cA]
        invm_b = invm_pad[cB]
        Ii_a = Iinv_pad[cA]
        Ii_b = Iinv_pad[cB]

        ra = point - self.pos[cA]
        pos_pad = np.vstack([self.pos, np.zeros((1, 3))])
        rb = point - pos_pad[cB]

        kn = np.maximum(self._denom(invm_a, invm_b, Ii_a, Ii_b, ra, rb, nrm), 1e-6)
        a_vec = np.where(
            np.abs(nrm[:, 2:3]) < 0.9,
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        t1 = np.cross(nrm, a_vec)
        t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
        t2 = np.cross(nrm, t1)
        kt1 = np.maximum(self._denom(invm_a, invm_b, Ii_a, Ii_b, ra, rb, t1), 1e-6)
        kt2 = np.maximum(self._denom(invm_a, invm_b, Ii_a, Ii_b, ra, rb, t2), 1e-6)

        bias = np.minimum(
            BETA * np.maximum(-gap - SLOP, 0.0) / dt, MAX_BIAS
        )

        vp = np.zeros((p, 3))
        vp[:n] = self.vel
        wp = np.zeros((p, 3))
        wp[:n] = self.omega

        lam_n = np.zeros(C)
        lam_1 = np.zeros(C)
        lam_2 = np.zeros(C)

        def scatter(idx, dv, out):
            for k in range(3):
                out[:, k] += np.bincount(idx, weights=dv[:, k], minlength=p)

        for _ in range(N_ITER):
            v_a = vp[cA]
            w_a = wp[cA]
            v_b = vp[cB]
            w_b = wp[cB]
            vrel = (v_b + np.cross(w_b, rb)) - (v_a + np.cross(w_a, ra))
            vn = np.einsum("nc,nc->n", vrel, nrm)
            dl = RELAX * (bias - vn) / kn
            new_n = np.maximum(lam_n + dl, 0.0)
            dl = new_n - lam_n
            lam_n = new_n

            vt1 = np.einsum("nc,nc->n", vrel, t1)
            vt2 = np.einsum("nc,nc->n", vrel, t2)
            d1 = -RELAX * vt1 / kt1
            d2 = -RELAX * vt2 / kt2
            n1 = lam_1 + d1
            n2 = lam_2 + d2
            mag = np.hypot(n1, n2)
            cap = MU * lam_n
            over = mag > cap
            scale = np.where(over & (mag > 1e-12), cap / np.maximum(mag, 1e-12), 1.0)
            n1 = n1 * scale
            n2 = n2 * scale
            d1 = n1 - lam_1
            d2 = n2 - lam_2
            lam_1 = n1
            lam_2 = n2

            imp_b = (
                dl[:, None] * nrm + d1[:, None] * t1 + d2[:, None] * t2
            )
            imp_a = -imp_b

            imp_sum = np.zeros((p, 3))
            scatter(cA, imp_a, imp_sum)
            scatter(cB, imp_b, imp_sum)
            vp += invm_pad[:, None] * imp_sum

            tor_a = np.cross(ra, imp_a)
            tor_b = np.cross(rb, imp_b)
            dom_a = np.einsum("nij,nj->ni", Ii_a, tor_a)
            dom_b = np.einsum("nij,nj->ni", Ii_b, tor_b)
            dw = np.zeros((p, 3))
            scatter(cA, dom_a, dw)
            scatter(cB, dom_b, dw)
            wp += dw

        self.vel = vp[:n]
        self.omega = wp[:n]

        # Settling assist: extra damping for slow bodies in contact.
        touch = np.zeros(n, dtype=bool)
        touch[cA[cA < n]] = True
        sp = np.linalg.norm(self.vel, axis=1)
        sw = np.linalg.norm(self.omega, axis=1)
        slow = touch & (sp < 0.05) & (sw < 0.25)
        if slow.any():
            self.vel[slow] *= 0.6
            self.omega[slow] *= 0.6

        # Mild direct positional de-penetration (translation only).
        pen = np.maximum(-(gap + SLOP), 0.0)
        corr = np.minimum(0.4 * pen, 0.008)
        if (corr > 0).any():
            tot = invm_a + invm_b
            share_a = np.where(tot > 0, invm_a / np.maximum(tot, 1e-12), 0.0)
            dxa = -nrm * (corr * share_a)[:, None]
            dxb = nrm * (corr * (1.0 - share_a))[:, None]
            dsum = np.zeros((p, 3))
            scatter(cA, dxa, dsum)
            dyn_b = cB < n
            scatter(cB[dyn_b], dxb[dyn_b], dsum)
            self.pos += dsum[:n]

        imp_total = (
            lam_n[:, None] * nrm + lam_1[:, None] * t1 + lam_2[:, None] * t2
        )
        self._report_gap = gap.copy()
        self._report_force = imp_total / dt

    # ------------------------------------------------------------------ stepping
    def _step(self):
        dt = self.dt
        n = self.N
        self.vel[:, 2] -= G * dt
        if LIN_DAMP > 0:
            self.vel *= math.exp(-LIN_DAMP * dt)
        if ANG_DAMP > 0:
            self.omega *= math.exp(-ANG_DAMP * dt)

        R = _quat_to_mat(self.quat)
        pts = np.einsum("nij,kj->nki", R, self.template) + self.pos[:, None, :]
        Iw = np.einsum("nij,njk,nlk->nil", R, self.I_unit, R)
        Iw /= self.mass[:, None, None]
        self.Iinv_world = np.linalg.inv(Iw)

        con = self._detect(pts)
        if con is not None and len(con["gap"]):
            self._solve(con, dt)

        # Integrate pose.
        self.pos += self.vel * dt
        wq = np.concatenate([self.omega, np.zeros((n, 1))], axis=1)
        dq = 0.5 * _quat_mul(wq, self.quat)
        self.quat = self.quat + dq * dt
        nq = np.linalg.norm(self.quat, axis=1, keepdims=True)
        bad = (nq < 1e-8)[:, 0]
        if bad.any():
            self.quat[bad] = np.array([0.0, 0.0, 0.0, 1.0])
            nq = np.linalg.norm(self.quat, axis=1, keepdims=True)
        self.quat /= nq

        # Safety clamps and finiteness guards.
        sp = np.linalg.norm(self.vel, axis=1)
        hi = sp > VMAX
        if hi.any():
            self.vel[hi] *= (VMAX / sp[hi])[:, None]
        sw = np.linalg.norm(self.omega, axis=1)
        hi = sw > WMAX
        if hi.any():
            self.omega[hi] *= (WMAX / sw[hi])[:, None]
        np.nan_to_num(self.pos, copy=False)
        np.nan_to_num(self.vel, copy=False)
        np.nan_to_num(self.omega, copy=False)

        self.step_index += 1
        self.t = self.step_index * dt

    # ------------------------------------------------------------------ API
    def advance_to(self, time: float):
        target = float(time)
        target_steps = int(round(target / self.dt))
        if target_steps < self.step_index:
            target_steps = self.step_index
        while self.step_index < target_steps:
            self._step()
        return self._frame(target)

    def _frame(self, t: float):
        gap = np.asarray(self._report_gap, dtype=float)
        force = np.asarray(self._report_force, dtype=float)
        if force.size == 0:
            force = np.zeros((0, 3))
        return StateFrame(
            time=float(t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.quat.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": gap,
                "external_force": self.ext_force.copy(),
                "interaction_force": force,
            },
            scalars={"contact_length_scale": float(WALL_T)},
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "num_bodies": int(self.N),
            "num_contacts": int(len(self._report_gap)),
            "substep_dt": float(self.dt),
        }
