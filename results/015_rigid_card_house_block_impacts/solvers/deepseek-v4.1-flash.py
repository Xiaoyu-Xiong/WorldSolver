"""Deterministic rigid-body solver for the card-house impact scene.

Physics: full 3-D rigid bodies (semi-implicit Euler) with an impulse-based
projected Gauss-Seidel contact solver, Coulomb friction (box friction against
the accumulated normal impulse), speculative contacts, Baumgarte positional
bias and restitution.  Contact manifolds come from sampling the surface of
each box against the other box (closest-point / minimum-axis penetration),
which stays well defined for the thin card plates used by this scene.

No external physics engine is used: only NumPy plus the frozen contract.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# --------------------------------------------------------------------------
# frozen scene description (mirrors scene_geometry.yaml / config.yaml)
# --------------------------------------------------------------------------
CARD_H = (0.45, 0.18, 0.015)
BLOCK_H = (0.16, 0.16, 0.16)
CENTER_H = (0.19, 0.19, 0.19)
FLOOR_H = (3.2, 2.2, 0.05)
FLOOR_CENTER = (0.0, 0.0, -0.05)

SUB_DT = 0.0025          # fixed physics substep (divides sample_dt exactly)
MAX_ITER = 20            # projected Gauss-Seidel sweeps per substep
BETA = 0.08              # Baumgarte positional feedback
SLOP = 0.004             # allowed penetration (m)
MAX_CORR = 0.12          # cap on positional-bias separation speed (m/s)
MARGIN = 0.02            # speculative contact window (m)
CLUSTER = 0.004          # contact point clustering (m)
MAX_PER_PAIR = 10
REST_THRESHOLD = 0.35    # minimum approach speed for restitution (m/s)
MAX_SPEED = 12.0
MAX_SPIN = 60.0
MAX_REPORT_PEN = 0.04


def _quat_to_mat(q):
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def _tangents(nx, ny, nz):
    ax, ay, az = abs(nx), abs(ny), abs(nz)
    if ax <= ay and ax <= az:
        hx, hy, hz = 1.0, 0.0, 0.0
    elif ay <= az:
        hx, hy, hz = 0.0, 1.0, 0.0
    else:
        hx, hy, hz = 0.0, 0.0, 1.0
    t1x = ny * hz - nz * hy
    t1y = nz * hx - nx * hz
    t1z = nx * hy - ny * hx
    m = (t1x * t1x + t1y * t1y + t1z * t1z) ** 0.5
    t1x, t1y, t1z = t1x / m, t1y / m, t1z / m
    t2x = ny * t1z - nz * t1y
    t2y = nz * t1x - nx * t1z
    t2z = nx * t1y - ny * t1x
    return (t1x, t1y, t1z), (t2x, t2y, t2z)


def _angular_term(I, rx, ry, rz, dx, dy, dz):
    """d . ((Iinv (r x d)) x r) for a symmetric inverse inertia I (flat 9)."""
    cx = ry * dz - rz * dy
    cy = rz * dx - rx * dz
    cz = rx * dy - ry * dx
    ax = I[0] * cx + I[1] * cy + I[2] * cz
    ay = I[3] * cx + I[4] * cy + I[5] * cz
    az = I[6] * cx + I[7] * cy + I[8] * cz
    lx = ay * rz - az * ry
    ly = az * rx - ax * rz
    lz = ax * ry - ay * rx
    return dx * lx + dy * ly + dz * lz


def _relvel(vA, wA, rA, vB, wB, rB, dx, dy, dz):
    """d . (v_rel at contact)."""
    ax, ay, az = vA
    wx, wy, wz = wA
    rx, ry, rz = rA
    vx = ax + wy * rz - wz * ry
    vy = ay + wz * rx - wx * rz
    vz = az + wx * ry - wy * rx
    bx, by, bz = vB
    wx, wy, wz = wB
    rx, ry, rz = rB
    bx = bx + wy * rz - wz * ry
    by = by + wz * rx - wx * rz
    bz = bz + wx * ry - wy * rx
    return (vx - bx) * dx + (vy - by) * dy + (vz - bz) * dz


def _apply(vel, omg, invm, Iinv, i, j, rA, rB, px, py, pz):
    """Apply +p at rA on body i and -p at rB on body j."""
    rx, ry, rz = rA
    lx = ry * pz - rz * py
    ly = rz * px - rx * pz
    lz = rx * py - ry * px
    I = Iinv[i]
    w = omg[i]
    w[0] += I[0] * lx + I[1] * ly + I[2] * lz
    w[1] += I[3] * lx + I[4] * ly + I[5] * lz
    w[2] += I[6] * lx + I[7] * ly + I[8] * lz
    mi = invm[i]
    v = vel[i]
    v[0] += mi * px
    v[1] += mi * py
    v[2] += mi * pz
    rx, ry, rz = rB
    lx = ry * pz - rz * py
    ly = rz * px - rx * pz
    lz = rx * py - ry * px
    I = Iinv[j]
    w = omg[j]
    w[0] -= I[0] * lx + I[1] * ly + I[2] * lz
    w[1] -= I[3] * lx + I[4] * ly + I[5] * lz
    w[2] -= I[6] * lx + I[7] * ly + I[8] * lz
    mj = invm[j]
    v = vel[j]
    v[0] -= mj * px
    v[1] -= mj * py
    v[2] -= mj * pz


class BenchmarkSolver:
    """Impulse-based frictional rigid-body solver for the frozen card house."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = getattr(scene, "config", None) or {}
        frame = scene.initial_state()

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        cp = cfg.get("contact_parameters", {}) or {}
        self.mu = float(cp.get("friction_coefficient", 0.8))
        self.restitution = float(cp.get("restitution", 0.1))

        pos = np.asarray(frame.position, dtype=float).copy()
        vel = np.asarray(frame.velocity, dtype=float).copy()
        quat = np.asarray(frame.orientation, dtype=float).copy()
        omg = np.asarray(frame.angular_velocity, dtype=float).copy()
        mass = np.asarray(frame.mass, dtype=float).copy()
        bid = np.asarray(frame.body_id, dtype=int)

        n_out = len(mass)
        self.n_out = n_out
        nb = n_out + 1                      # + static floor
        self.nb = nb

        half = np.zeros((nb, 3), dtype=float)
        half[:n_out] = np.array([CARD_H] * 10 + [BLOCK_H, BLOCK_H, CENTER_H], dtype=float)
        half[nb - 1] = np.array(FLOOR_H, dtype=float)

        # honour the frozen geometry file when present
        geo = (cfg.get("scene_geometry") or {}).get("objects") or []
        for obj in geo:
            bind = obj.get("state_binding") or {}
            g = obj.get("geometry") or {}
            he = g.get("half_extents")
            if he is None:
                continue
            if bind.get("selector") == "body_id" and g.get("type") == "box":
                half[int(bind["value"])] = np.asarray(he, dtype=float)
            elif obj.get("id") == "floor" and g.get("type") == "box":
                half[nb - 1] = np.asarray(he, dtype=float)
        self.half = half

        release = np.zeros(nb, dtype=float)
        is_block = np.zeros(nb, dtype=bool)
        attrs = getattr(frame, "attributes", None) or {}
        if "release_time" in attrs:
            rel = np.asarray(attrs["release_time"], dtype=float)
            release[:n_out] = rel
        else:
            rel = cfg.get("initial_conditions", {}).get(
                "impact_block_release_times", [1.5, 5.0, 8.5]
            )
            release[n_out - 3: n_out] = np.asarray(rel, dtype=float)[:3]
        if "is_impact_block" in attrs:
            is_block[:n_out] = np.asarray(attrs["is_impact_block"], dtype=float) > 0.5
        else:
            is_block[n_out - 3: n_out] = True
        self.release_time = release
        self.is_block = is_block

        m = np.ones(nb, dtype=float)
        m[:n_out] = mass
        self.mass = m
        ibody_inv = np.zeros((nb, 3), dtype=float)
        for b in range(n_out):
            hx, hy, hz = half[b]
            mx = m[b]
            if mx <= 0.0:
                continue
            ibody_inv[b, 0] = 3.0 / (mx * (hy * hy + hz * hz))
            ibody_inv[b, 1] = 3.0 / (mx * (hx * hx + hz * hz))
            ibody_inv[b, 2] = 3.0 / (mx * (hx * hx + hy * hy))
        self.ibody_inv = ibody_inv

        # full body list: scene bodies then the static floor box
        self.pos = np.zeros((nb, 3), dtype=float)
        self.pos[:n_out] = pos
        self.pos[nb - 1] = np.array(FLOOR_CENTER, dtype=float)
        self.vel = np.zeros((nb, 3), dtype=float)
        self.vel[:n_out] = vel
        self.quat = np.zeros((nb, 4), dtype=float)
        self.quat[:n_out] = quat
        self.quat[nb - 1] = np.array([0.0, 0.0, 0.0, 1.0])
        self.omg = np.zeros((nb, 3), dtype=float)
        self.omg[:n_out] = omg
        self.body_id = bid

        self.held = np.zeros(nb, dtype=bool)
        self.held[:n_out] = is_block[:n_out]
        self.inv_mass = np.zeros(nb, dtype=float)
        for b in range(n_out):
            if not self.held[b]:
                self.inv_mass[b] = 1.0 / m[b]

        # surface sample points (all combinations of face/edge/centre samples)
        pts = []
        for sx in (-1.0, 0.0, 1.0):
            for sy in (-1.0, 0.0, 1.0):
                for sz in (-1.0, 0.0, 1.0):
                    if sx == 0.0 and sy == 0.0 and sz == 0.0:
                        continue
                    pts.append((sx, sy, sz))
        self.sample_signs = np.asarray(pts, dtype=float)
        self.local_samples = self.sample_signs[None, :, :] * half[:, None, :]
        self.ns = len(pts)

        self.R = np.array([_quat_to_mat(q) for q in self.quat])
        self._refresh_kinematics()

        self.step_index = 0
        self.t = 0.0
        self._last_contacts = []
        self._warm = {}
        self.max_pen_seen = 0.0
        self.max_contacts = 0
        self.stats = {"steps": 0, "max_penetration": 0.0, "max_contacts": 0}

    # ------------------------------------------------------------- helpers
    def _refresh_kinematics(self):
        half = self.half
        self.R = np.array([_quat_to_mat(q) for q in self.quat])
        self.samples = self.pos[:, None, :] + np.einsum(
            "bsj,bij->bsi", self.local_samples, self.R
        )
        self.aabb_min = self.samples.min(axis=1)
        self.aabb_max = self.samples.max(axis=1)
        Iinv = np.zeros((self.nb, 9), dtype=float)
        for b in range(self.nb):
            if self.inv_mass[b] <= 0.0:
                continue
            R = self.R[b]
            d = self.ibody_inv[b]
            M = R @ np.diag(d) @ R.T
            Iinv[b] = M.reshape(9)
        self.Iinv = Iinv

    def _update_releases(self):
        for b in range(self.n_out):
            if self.held[b] and self.t >= self.release_time[b] - 1e-12:
                self.held[b] = False
                self.inv_mass[b] = 1.0 / self.mass[b]

    # ------------------------------------------------------ contact search
    def _gather(self, owner, other, bucket):
        R = self.R[other]
        dloc = (self.samples[owner] - self.pos[other]) @ R
        h = self.half[other]
        cl = np.clip(dloc, -h, h)
        dv = dloc - cl
        dist = np.sqrt(np.einsum("ij,ij->i", dv, dv))
        inside = dist < 1e-9
        pen = h[None, :] - np.abs(dloc)
        k = np.argmin(pen, axis=1)
        rows = np.arange(len(dloc))
        pk = pen[rows, k]
        sgn = np.sign(dloc[rows, k])
        sgn = np.where(sgn == 0.0, 1.0, sgn)
        qloc = cl.copy()
        qloc[rows, k] = sgn * h[k]
        nloc = np.zeros_like(dloc)
        nloc[rows, k] = sgn
        safe = np.maximum(dist, 1e-12)
        outn = dv / safe[:, None]
        signed = np.where(inside, -pk, dist)
        nvec = np.where(inside[:, None], nloc, outn) @ R.T
        pvec = self.pos[other] + qloc @ R.T
        sel = np.nonzero(signed <= MARGIN)[0]
        for idx in sel:
            key = (
                round(float(pvec[idx, 0]) / CLUSTER),
                round(float(pvec[idx, 1]) / CLUSTER),
                round(float(pvec[idx, 2]) / CLUSTER),
            )
            s = max(float(signed[idx]), -MAX_REPORT_PEN)
            cur = bucket.get(key)
            if cur is None or s < cur[0]:
                bucket[key] = (s, nvec[idx].copy(), pvec[idx].copy(), owner, other)

    def _detect(self, dt):
        nb = self.nb
        buckets = []
        for i in range(nb):
            for j in range(i + 1, nb):
                if self.inv_mass[i] <= 0.0 and self.inv_mass[j] <= 0.0:
                    continue
                if np.any(self.aabb_min[i] > self.aabb_max[j] + MARGIN):
                    continue
                if np.any(self.aabb_min[j] > self.aabb_max[i] + MARGIN):
                    continue
                bucket = {}
                self._gather(i, j, bucket)
                self._gather(j, i, bucket)
                if bucket:
                    buckets.append(bucket)

        contacts = []
        warm_new = {}
        for bucket in buckets:
            items = sorted(bucket.values(), key=lambda e: e[0])
            for s, nvec, pvec, owner, other in items[:MAX_PER_PAIR]:
                n = np.asarray(nvec, dtype=float)
                nn = float(np.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]))
                if nn <= 1e-9:
                    continue
                n = n / nn
                t1, t2 = _tangents(n[0], n[1], n[2])
                rA = pvec - self.pos[owner]
                rB = pvec - self.pos[other]
                iA = int(owner)
                iB = int(other)
                Ii = self.Iinv[iA]
                Ij = self.Iinv[iB]
                m11 = self.inv_mass[iA] + self.inv_mass[iB]
                kn = m11 + _angular_term(Ii, rA[0], rA[1], rA[2], n[0], n[1], n[2])
                kn += _angular_term(Ij, rB[0], rB[1], rB[2], n[0], n[1], n[2])
                if kn <= 1e-12:
                    continue
                kt1 = m11 + _angular_term(Ii, rA[0], rA[1], rA[2], t1[0], t1[1], t1[2])
                kt1 += _angular_term(Ij, rB[0], rB[1], rB[2], t1[0], t1[1], t1[2])
                kt2 = m11 + _angular_term(Ii, rA[0], rA[1], rA[2], t2[0], t2[1], t2[2])
                kt2 += _angular_term(Ij, rB[0], rB[1], rB[2], t2[0], t2[1], t2[2])
                if s < 0.0:
                    exc = max(-s - SLOP, 0.0)
                    va = min(BETA * exc / dt, MAX_CORR)
                else:
                    va = -s / dt
                key = (
                    iA,
                    iB,
                    round(float(pvec[0]) / CLUSTER),
                    round(float(pvec[1]) / CLUSTER),
                    round(float(pvec[2]) / CLUSTER),
                )
                prev = self._warm.get(key)
                if prev is None:
                    lam_n = lam_t1 = lam_t2 = 0.0
                else:
                    lam_n, lam_t1, lam_t2 = prev
                contacts.append(
                    [
                        iA,
                        iB,
                        float(n[0]),
                        float(n[1]),
                        float(n[2]),
                        float(t1[0]),
                        float(t1[1]),
                        float(t1[2]),
                        float(t2[0]),
                        float(t2[1]),
                        float(t2[2]),
                        float(rA[0]),
                        float(rA[1]),
                        float(rA[2]),
                        float(rB[0]),
                        float(rB[1]),
                        float(rB[2]),
                        0.0,
                        1.0 / kn,
                        1.0 / kt1,
                        1.0 / kt2,
                        float(max(lam_n, 0.0)),
                        float(lam_t1),
                        float(lam_t2),
                        float(pvec[0]),
                        float(pvec[1]),
                        float(pvec[2]),
                        float(s),
                        va,
                        key,
                    ]
                )
                warm_new[key] = contacts[-1]
        self._warm_next = warm_new
        return contacts

    # ---------------------------------------------------------- solve step
    def _solve(self, contacts, dt):
        vel = self.vel.tolist()
        omg = self.omg.tolist()
        invm = self.inv_mass.tolist()
        Iinv = [list(row) for row in self.Iinv]
        mu = self.mu
        e = self.restitution

        # restitution targets from the pre-solve approach speed
        for c in contacts:
            iA, iB = c[0], c[1]
            vn0 = _relvel(
                vel[iA], omg[iA], c[11:14],
                vel[iB], omg[iB], c[14:17],
                c[2], c[3], c[4],
            )
            va = c[28]
            if vn0 < -REST_THRESHOLD:
                bounce = -e * vn0
                if bounce > va:
                    va = bounce
            c[17] = va

        # warm start
        for c in contacts:
            ln, lt1, lt2 = c[21], c[22], c[23]
            if ln == 0.0 and lt1 == 0.0 and lt2 == 0.0:
                continue
            px = ln * c[2] + lt1 * c[5] + lt2 * c[8]
            py = ln * c[3] + lt1 * c[6] + lt2 * c[9]
            pz = ln * c[4] + lt1 * c[7] + lt2 * c[10]
            _apply(vel, omg, invm, Iinv, c[0], c[1], c[11:14], c[14:17], px, py, pz)

        for _ in range(MAX_ITER):
            for c in contacts:
                iA, iB = c[0], c[1]
                vA = vel[iA]
                wA = omg[iA]
                vB = vel[iB]
                wB = omg[iB]
                rA = c[11:14]
                rB = c[14:17]
                nx, ny, nz = c[2], c[3], c[4]
                vn = _relvel(vA, wA, rA, vB, wB, rB, nx, ny, nz)
                dlam = (c[17] - vn) * c[18]
                new_n = c[21] + dlam
                if new_n < 0.0:
                    new_n = 0.0
                dlam = new_n - c[21]
                c[21] = new_n
                if dlam != 0.0:
                    _apply(
                        vel, omg, invm, Iinv, iA, iB, rA, rB,
                        dlam * nx, dlam * ny, dlam * nz,
                    )
                if c[21] <= 0.0:
                    continue
                maxf = mu * c[21]
                t1x, t1y, t1z = c[5], c[6], c[7]
                vt = _relvel(vA, wA, rA, vB, wB, rB, t1x, t1y, t1z)
                dl = -vt * c[19]
                new_t = c[22] + dl
                if new_t > maxf:
                    new_t = maxf
                elif new_t < -maxf:
                    new_t = -maxf
                dl = new_t - c[22]
                c[22] = new_t
                if dl != 0.0:
                    _apply(
                        vel, omg, invm, Iinv, iA, iB, rA, rB,
                        dl * t1x, dl * t1y, dl * t1z,
                    )
                t2x, t2y, t2z = c[8], c[9], c[10]
                vt = _relvel(vA, wA, rA, vB, wB, rB, t2x, t2y, t2z)
                dl = -vt * c[20]
                new_t = c[23] + dl
                if new_t > maxf:
                    new_t = maxf
                elif new_t < -maxf:
                    new_t = -maxf
                dl = new_t - c[23]
                c[23] = new_t
                if dl != 0.0:
                    _apply(
                        vel, omg, invm, Iinv, iA, iB, rA, rB,
                        dl * t2x, dl * t2y, dl * t2z,
                    )

        v = np.asarray(vel, dtype=float)
        w = np.asarray(omg, dtype=float)
        np.clip(v, -MAX_SPEED, MAX_SPEED, out=v)
        np.clip(w, -MAX_SPIN, MAX_SPIN, out=w)
        self.vel = v
        self.omg = w

    # ------------------------------------------------------------- stepping
    def _step(self, dt):
        self._update_releases()
        self._refresh_kinematics()
        for b in range(self.nb):
            if self.inv_mass[b] > 0.0:
                self.vel[b] += dt * self.gravity
        contacts = self._detect(dt)
        self._solve(contacts, dt)

        # integration of positions / orientations
        self.pos += dt * self.vel
        w = self.omg
        q = self.quat
        qw = q[:, 3]
        qx = q[:, 0]
        qy = q[:, 1]
        qz = q[:, 2]
        hx = 0.5 * dt
        dqx = hx * (w[:, 0] * qw + w[:, 1] * qz - w[:, 2] * qy)
        dqy = hx * (w[:, 1] * qw + w[:, 2] * qx - w[:, 0] * qz)
        dqz = hx * (w[:, 2] * qw + w[:, 0] * qy - w[:, 1] * qx)
        dqw = hx * (-w[:, 0] * qx - w[:, 1] * qy - w[:, 2] * qz)
        q = q + np.stack([dqx, dqy, dqz, dqw], axis=1)
        norm = np.sqrt(np.einsum("ij,ij->i", q, q))
        norm = np.where(norm < 1e-12, 1.0, norm)
        self.quat = q / norm[:, None]

        self._last_contacts = contacts
        self._warm = dict(self._warm_next)
        self.step_index += 1
        self.t = self.step_index * SUB_DT
        pen = 0.0
        for c in contacts:
            if c[27] < -pen:
                pen = -c[27]
        if pen > self.stats["max_penetration"]:
            self.stats["max_penetration"] = float(pen)
        if len(contacts) > self.stats["max_contacts"]:
            self.stats["max_contacts"] = int(len(contacts))
        self.stats["steps"] = int(self.step_index)

    # -------------------------------------------------------------- public
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        while (self.step_index + 1) * SUB_DT <= target + 1e-9:
            self._step(SUB_DT)
            guard += 1
            if guard > 100000:
                break
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        n = self.n_out
        contacts = self._last_contacts
        dt = SUB_DT
        k = len(contacts)
        if k:
            dist = np.empty(k, dtype=float)
            f = np.empty((k, 3), dtype=float)
            for idx, c in enumerate(contacts):
                dist[idx] = c[27]
                px = c[21] * c[2] + c[22] * c[5] + c[23] * c[8]
                py = c[21] * c[3] + c[22] * c[6] + c[23] * c[9]
                pz = c[21] * c[4] + c[22] * c[7] + c[23] * c[10]
                f[idx, 0] = px / dt
                f[idx, 1] = py / dt
                f[idx, 2] = pz / dt
            interaction = np.concatenate([f, -f], axis=0)
        else:
            dist = np.zeros(0, dtype=float)
            interaction = np.zeros((0, 3), dtype=float)

        ext = np.zeros((n, 3), dtype=float)
        for b in range(n):
            if self.inv_mass[b] > 0.0:
                ext[b] = self.mass[b] * self.gravity

        frame = StateFrame(
            time=float(time),
            position=self.pos[:n].copy(),
            velocity=self.vel[:n].copy(),
            orientation=self.quat[:n].copy(),
            angular_velocity=self.omg[:n].copy(),
            mass=self.mass[:n].copy(),
            body_id=self.body_id,
            attributes={
                "release_time": self.release_time[:n].copy(),
                "is_impact_block": self.is_block[:n].astype(float),
            },
            fields={
                "contact_signed_distance": dist,
                "external_force": ext,
                "interaction_force": interaction,
            },
            scalars={
                "contact_length_scale": float(2.0 * min(CARD_H)),
                "friction_coefficient": float(self.mu),
                "restitution": float(self.restitution),
                "contact_count": float(k),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "scene_id": "015_rigid_card_house_block_impacts",
            "integrator": "semi-implicit Euler + projected Gauss-Seidel impulses",
            "substep": SUB_DT,
            "iterations": MAX_ITER,
            "friction_coefficient": self.mu,
            "restitution": self.restitution,
            "seed": self.seed,
            "steps": int(self.step_index),
            "sim_time": float(self.t),
            "max_contacts": int(self.stats["max_contacts"]),
            "max_penetration": float(self.stats["max_penetration"]),
        }
