"""Rigid-body impulse solver for the circular domino chain reaction.

Physics: 24 upright box dominoes on a static floor box.  The trigger domino
starts tilted and slightly sunk, so gravity topples it into its neighbours and
a toppling front propagates around the circle.

Method: semi-implicit Euler with fixed sub-steps.  Each sub-step
  1. applies gravity,
  2. detects contacts (OBB-OBB SAT with face clipping / edge-edge closest
     points, plus domino-corner vs floor-plane contacts),
  3. resolves them with accumulated sequential impulses (normal impulse with
     restitution + Baumgarte positional bias, Coulomb friction in two tangent
     directions),
  4. integrates positions and quaternions.

Exported per frame: body ids, masses, positions, velocities, orientations,
angular velocities, external (gravity) forces, per-pair interaction forces
(accumulated contact impulse / sample interval) and signed contact distances.
"""
from __future__ import annotations

import math
import os

import numpy as np

from worldsolver.contracts import StateFrame


def _qmul(a, b):
    """Batched Hamilton product of xyzw quaternion arrays (n,4)."""
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out = np.empty_like(a)
    out[:, 0] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 1] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 2] = aw * bz + ax * by - ay * bx + az * bw
    out[:, 3] = aw * bw - ax * bx - ay * by - az * bz
    return out


class BenchmarkSolver:
    NEAR = 0.06          # signed-gap reporting margin (m)
    FLOOR_MARGIN = 0.002  # speculative floor contact margin (m)
    BETA = 0.15          # Baumgarte factor
    SLOP = 0.002         # allowed penetration (m)
    MAX_BIAS = 1.0       # max Baumgarte/restitution bias velocity (m/s)
    ITERS = 8
    REST_THRESH = 0.6    # approach speed above which restitution applies

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        ct = cfg.get("contact", {}) or {}
        self.mu = float(ct.get("friction", 0.55))
        self.restitution = float(ct.get("restitution", 0.12))
        self.cls = float(ct.get("length_scale", 0.02))

        st = scene.initial_state()
        self.ids = np.asarray(st.body_id).astype(np.int32)
        self.n = int(len(self.ids))
        self.c = np.array(st.position, dtype=float)
        q = np.array(st.orientation, dtype=float)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        self.q = q
        self.v = np.zeros((self.n, 3))
        self.w = np.zeros((self.n, 3))
        self.m = np.array(st.mass, dtype=float)
        self.inv_m = 1.0 / self.m

        # geometry (half extents per body, static floor box)
        he = np.tile(np.array([0.07, 0.2, 0.55]), (self.n, 1))
        floor_c = np.array([0.0, 0.0, -0.05])
        floor_h = np.array([3.2, 2.2, 0.05])
        try:
            import yaml
            path = os.path.join(os.getcwd(), "scene_geometry.yaml")
            with open(path, "r") as fh:
                geo = yaml.safe_load(fh)
            for obj in geo.get("objects", []):
                geom = obj.get("geometry", {}) or {}
                if geom.get("type") != "box":
                    continue
                hvec = np.asarray(geom.get("half_extents", [0.07, 0.2, 0.55]), dtype=float)
                tr = obj.get("transform", {}) or {}
                tl = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float)
                if obj.get("motion") == "static" or obj.get("id") == "floor":
                    floor_h = hvec
                    floor_c = tl
                else:
                    sb = obj.get("state_binding") or {}
                    if sb.get("selector") == "body_id":
                        bid = int(sb.get("value"))
                        if 0 <= bid < self.n:
                            he[bid] = hvec
        except Exception:
            pass
        self.he = he
        self.floor_c = floor_c
        self.floor_h = floor_h
        self.floor_top = float(floor_c[2] + floor_h[2])
        hx, hy, hz = he[:, 0], he[:, 1], he[:, 2]
        self.Iloc_inv = np.stack([
            3.0 / (self.m * (hy * hy + hz * hz)),
            3.0 / (self.m * (hx * hx + hz * hz)),
            3.0 / (self.m * (hx * hx + hy * hy)),
        ], axis=1)
        self.brad = np.linalg.norm(he, axis=1)
        signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                          for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
        self.local_corners = signs[None, :, :] * he[:, None, :]

        self.h = 0.002
        self.step_count = 0
        self.last_t = 0.0
        self.pairJ = {}
        self.last_gaps = np.zeros(0)
        self.Rs = self._quat_to_R(self.q)
        self.contacts_count = 0

    @staticmethod
    def _quat_to_R(q):
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = np.empty((q.shape[0], 3, 3))
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    # ------------------------------------------------------------- simulation
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        target = int(round(time / self.h))
        if target < self.step_count:
            target = self.step_count
        self.pairJ = {}
        dt_int = max(time - self.last_t, 1e-9)
        if target == self.step_count:
            # no motion (first frame): still report contact geometry at t
            _, gaps = self._detect()
            self.last_gaps = np.asarray(gaps, dtype=float)
        while self.step_count < target:
            self._substep()
            self.step_count += 1
        self.last_t = time
        return self._make_frame(time, dt_int)

    def _substep(self):
        h = self.h
        self.v += h * self.gravity[None, :]
        self.Rs = self._quat_to_R(self.q)
        invI = np.einsum('nij,nj,nkj->nik', self.Rs, self.Iloc_inv, self.Rs)
        contacts, gaps = self._detect()
        self.last_gaps = np.asarray(gaps, dtype=float) if gaps else np.zeros(0)
        self.contacts_count = len(contacts)
        self._solve(contacts, invI, h)
        damp = 1.0 / (1.0 + 0.05 * h)
        self.v *= damp
        self.w *= damp
        sp = np.linalg.norm(self.v, axis=1)
        bad = sp > 20.0
        if np.any(bad):
            self.v[bad] *= (20.0 / sp[bad])[:, None]
        spw = np.linalg.norm(self.w, axis=1)
        bad = spw > 60.0
        if np.any(bad):
            self.w[bad] *= (60.0 / spw[bad])[:, None]
        self.c += self.v * h
        wq = np.concatenate([self.w, np.zeros((self.n, 1))], axis=1)
        self.q += 0.5 * h * _qmul(wq, self.q)
        self.q /= np.linalg.norm(self.q, axis=1, keepdims=True)

    # -------------------------------------------------------------- detection
    def _detect(self):
        n = self.n
        cs, Rs = self.c, self.Rs
        contacts = []   # (a, b, normal(i->j), point, depth); a==n means floor
        gaps = []
        ez = np.array([0.0, 0.0, 1.0])
        # --- domino corners vs floor plane
        corners = cs[:, None, :] + np.einsum('nij,nkj->nki', Rs, self.local_corners)
        zc = corners[:, :, 2] - self.floor_top
        for i in range(n):
            below = np.nonzero(zc[i] < self.FLOOR_MARGIN)[0]
            for k in below:
                p = corners[i, k]
                if (abs(p[0] - self.floor_c[0]) <= self.floor_h[0]
                        and abs(p[1] - self.floor_c[1]) <= self.floor_h[1]):
                    contacts.append((n, i, ez, p.copy(), -float(zc[i, k])))
            if below.size == 0:
                g = float(zc[i].min())
                if g < self.NEAR:
                    gaps.append(g)
        # --- domino pairs (OBB SAT)
        D = np.linalg.norm(cs[:, None, :] - cs[None, :, :], axis=-1)
        idx = np.arange(n)
        mask = (D < (self.brad[:, None] + self.brad[None, :] + self.NEAR)) & (idx[:, None] < idx[None, :])
        for i, j in zip(*np.nonzero(mask)):
            self._pair_contacts(int(i), int(j), contacts, gaps)
        return contacts, gaps

    def _pair_contacts(self, i, j, contacts, gaps):
        Ra, Rb = self.Rs[i], self.Rs[j]
        ha, hb = self.he[i], self.he[j]
        ci, cj = self.c[i], self.c[j]
        T = cj - ci
        Rm = Ra.T @ Rb
        t = Ra.T @ T
        absR = np.abs(Rm)
        best_gap = -1e18
        best = None
        # face axes of A
        for ax in range(3):
            gap = abs(t[ax]) - ha[ax] - float(absR[ax, :] @ hb)
            if gap > best_gap:
                best_gap, best = gap, ('A', ax, 0)
            if gap > self.NEAR:
                return
        # face axes of B
        for ax in range(3):
            tb = float(Rm[:, ax] @ t)
            gap = abs(tb) - hb[ax] - float(absR[:, ax] @ ha)
            if gap > best_gap:
                best_gap, best = gap, ('B', 0, ax)
            if gap > self.NEAR:
                return
        # cross axes
        for ii in range(3):
            i1, i2 = (ii + 1) % 3, (ii + 2) % 3
            for jj in range(3):
                j1, j2 = (jj + 1) % 3, (jj + 2) % 3
                lenL2 = 1.0 - Rm[ii, jj] * Rm[ii, jj]
                if lenL2 < 1e-8:
                    continue
                lenL = math.sqrt(lenL2)
                d = t[i2] * Rm[i1, jj] - t[i1] * Rm[i2, jj]
                rA = ha[i1] * absR[i2, jj] + ha[i2] * absR[i1, jj]
                rB = hb[j1] * absR[ii, j2] + hb[j2] * absR[ii, j1]
                gap = (abs(d) - rA - rB) / lenL
                if gap > best_gap:
                    best_gap, best = gap, ('E', ii, jj)
                if gap > self.NEAR:
                    return
        if best is None:
            return
        if best_gap > 0.0:
            gaps.append(float(best_gap))
            return
        depth_total = -best_gap
        added = 0
        kind, ax_i, ax_j = best
        if kind == 'A':
            s = 1.0 if t[ax_i] >= 0.0 else -1.0
            nref = Ra[:, ax_i] * s
            pts = self._face_clip(i, j, ax_i, nref)
            for p, sep in pts:
                contacts.append((i, j, nref, p, sep))
                gaps.append(-sep)
                added += 1
        elif kind == 'B':
            tb = Rm[:, ax_j] @ t
            s = 1.0 if tb >= 0.0 else -1.0
            nref = Rb[:, ax_j] * s          # outward from B toward A
            pts = self._face_clip(j, i, ax_j, nref)
            nAB = -nref
            for p, sep in pts:
                contacts.append((i, j, nAB, p, sep))
                gaps.append(-sep)
                added += 1
        else:
            dirA = Ra[:, ax_i]
            dirB = Rb[:, ax_j]
            i1, i2 = (ax_i + 1) % 3, (ax_i + 2) % 3
            s1 = 1.0 if t[i1] >= 0.0 else -1.0
            s2 = 1.0 if t[i2] >= 0.0 else -1.0
            pa = ci + Ra[:, i1] * (ha[i1] * s1) + Ra[:, i2] * (ha[i2] * s2)
            j1, j2 = (ax_j + 1) % 3, (ax_j + 2) % 3
            tB = Rb.T @ (ci - cj)
            u1 = 1.0 if tB[j1] >= 0.0 else -1.0
            u2 = 1.0 if tB[j2] >= 0.0 else -1.0
            pb = cj + Rb[:, j1] * (hb[j1] * u1) + Rb[:, j2] * (hb[j2] * u2)
            cA, cB = self._seg_closest(pa - dirA * ha[ax_i], pa + dirA * ha[ax_i],
                                       pb - dirB * hb[ax_j], pb + dirB * hb[ax_j])
            nvec = np.cross(dirA, dirB)
            ln = float(np.linalg.norm(nvec))
            if ln > 1e-9:
                nvec = nvec / ln
                if float(nvec @ T) < 0.0:
                    nvec = -nvec
                mid = 0.5 * (cA + cB)
                contacts.append((i, j, nvec, mid, depth_total))
                gaps.append(-depth_total)
                added += 1
        if added == 0:
            # fallback single contact along centre direction
            dist = float(np.linalg.norm(T))
            nvec = T / dist if dist > 1e-9 else np.array([0.0, 0.0, 1.0])
            p = 0.5 * (ci + cj)
            contacts.append((i, j, nvec, p, depth_total))
            gaps.append(-depth_total)

    def _face_clip(self, ref, inc, ax, nref):
        """Clip the incident box face against the reference face; return
        [(point, depth)] with depth >= ~0 for penetrating points."""
        Rr, cr, hr = self.Rs[ref], self.c[ref], self.he[ref]
        Ri, ci, hi = self.Rs[inc], self.c[inc], self.he[inc]
        dots = Ri.T @ nref
        jj = int(np.argmax(np.abs(dots)))
        sg = -1.0 if dots[jj] >= 0.0 else 1.0
        u = Ri[:, jj] * sg                       # incident outward face normal
        k1, k2 = (jj + 1) % 3, (jj + 2) % 3
        v1 = Ri[:, k1] * hi[k1]
        v2 = Ri[:, k2] * hi[k2]
        fc = ci + u * hi[jj]
        poly = [fc - v1 - v2, fc - v1 + v2, fc + v1 + v2, fc + v1 - v2]
        m1, m2 = (ax + 1) % 3, (ax + 2) % 3
        for am_ax in (m1, m2):
            am = Rr[:, am_ax]
            hm = float(hr[am_ax])
            cm = float(cr @ am)
            poly = self._clip_poly(poly, am, cm + hm)
            if not poly:
                return []
            poly = self._clip_poly(poly, -am, -(cm - hm))
            if not poly:
                return []
        off = float(cr @ nref) + float(hr[ax])
        out = []
        for p in poly:
            sep = float(p @ nref) - off
            if sep <= 1e-4:
                out.append((p.copy(), max(-sep, 0.0)))
        return out

    @staticmethod
    def _clip_poly(poly, nrm, d):
        out = []
        cnt = len(poly)
        for a in range(cnt):
            p0 = poly[a]
            p1 = poly[(a + 1) % cnt]
            d0 = float(p0 @ nrm) - d
            d1 = float(p1 @ nrm) - d
            if d0 <= 0.0:
                out.append(p0)
            if d0 * d1 < 0.0:
                tt = d0 / (d0 - d1)
                out.append(p0 + tt * (p1 - p0))
        return out

    @staticmethod
    def _seg_closest(p1, q1, p2, q2):
        d1 = q1 - p1
        d2 = q2 - p2
        r = p1 - p2
        a = float(d1 @ d1)
        e = float(d2 @ d2)
        f = float(d2 @ r)
        if a <= 1e-12 and e <= 1e-12:
            return p1.copy(), p2.copy()
        if a <= 1e-12:
            s = 0.0
            tpar = min(max(f / e, 0.0), 1.0)
        else:
            cc = float(d1 @ r)
            if e <= 1e-12:
                tpar = 0.0
                s = min(max(-cc / a, 0.0), 1.0)
            else:
                b = float(d1 @ d2)
                den = a * e - b * b
                s = (b * f - cc * e) / den if den > 1e-12 else 0.0
                s = min(max(s, 0.0), 1.0)
                tpar = (b * s + f) / e
                if tpar < 0.0:
                    tpar = 0.0
                    s = min(max(-cc / a, 0.0), 1.0)
                elif tpar > 1.0:
                    tpar = 1.0
                    s = min(max((b - cc) / a, 0.0), 1.0)
        return p1 + d1 * s, p2 + d2 * tpar

    # ------------------------------------------------------------ impulse solve
    def _solve(self, contacts, invI, h):
        if not contacts:
            return
        n = self.n
        prep = []
        zero3 = np.zeros(3)
        zeroI = np.zeros((3, 3))
        for (a, b, nrm, p, depth) in contacts:
            dyn_a = a < n
            ra = (p - self.c[a]) if dyn_a else zero3
            rb = p - self.c[b]
            ima = float(self.inv_m[a]) if dyn_a else 0.0
            imb = float(self.inv_m[b])
            Ia = invI[a] if dyn_a else zeroI
            Ib = invI[b]
            ran = np.cross(ra, nrm)
            rbn = np.cross(rb, nrm)
            kn = ima + imb
            if dyn_a:
                kn += float(nrm @ np.cross(Ia @ ran, ra))
            kn += float(nrm @ np.cross(Ib @ rbn, rb))
            if kn <= 1e-12:
                continue
            axis = np.array([1.0, 0.0, 0.0]) if abs(nrm[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
            t1 = np.cross(axis, nrm)
            t1 /= np.linalg.norm(t1)
            t2 = np.cross(nrm, t1)
            def effmass(tvec, rax_a, rax_b):
                kk = ima + imb
                if dyn_a:
                    kk += float(tvec @ np.cross(Ia @ rax_a, ra))
                kk += float(tvec @ np.cross(Ib @ rax_b, rb))
                return kk
            rat1 = np.cross(ra, t1); rbt1 = np.cross(rb, t1)
            rat2 = np.cross(ra, t2); rbt2 = np.cross(rb, t2)
            kt1 = effmass(t1, rat1, rbt1)
            kt2 = effmass(t2, rat2, rbt2)
            kt1_inv = 1.0 / kt1 if kt1 > 1e-12 else 0.0
            kt2_inv = 1.0 / kt2 if kt2 > 1e-12 else 0.0
            vr = self.v[b] + np.cross(self.w[b], rb)
            if dyn_a:
                vr = vr - (self.v[a] + np.cross(self.w[a], ra))
            vn0 = float(vr @ nrm)
            bias = min(self.BETA / h * max(depth - self.SLOP, 0.0), self.MAX_BIAS)
            if vn0 < -self.REST_THRESH:
                bias = min(bias + -self.restitution * vn0, 3.0)
            prep.append([a, b, dyn_a, ima, imb, Ia, Ib, ra, rb, nrm, t1, t2,
                         1.0 / kn, kt1_inv, kt2_inv, bias, 0.0, 0.0, 0.0,
                         np.zeros(3)])
        v, w = self.v, self.w
        for _ in range(self.ITERS):
            for pc in prep:
                (a, b, dyn_a, ima, imb, Ia, Ib, ra, rb, nrm, t1, t2,
                 kn_inv, kt1_inv, kt2_inv, bias, lam, lam1, lam2, Jsum) = pc
                # normal
                vr = v[b] + np.cross(w[b], rb)
                if dyn_a:
                    vr = vr - (v[a] + np.cross(w[a], ra))
                vn = float(vr @ nrm)
                dlam = (bias - vn) * kn_inv
                new = lam + dlam
                if new < 0.0:
                    new = 0.0
                dlam = new - lam
                lam = new
                if dlam != 0.0:
                    P = dlam * nrm
                    if dyn_a:
                        v[a] -= P * ima
                        w[a] -= Ia @ np.cross(ra, P)
                    v[b] += P * imb
                    w[b] += Ib @ np.cross(rb, P)
                    Jsum += P
                # friction 1
                vr = v[b] + np.cross(w[b], rb)
                if dyn_a:
                    vr = vr - (v[a] + np.cross(w[a], ra))
                vt = float(vr @ t1)
                dl = -vt * kt1_inv
                mx = self.mu * lam
                new1 = min(max(lam1 + dl, -mx), mx)
                dl = new1 - lam1
                lam1 = new1
                if dl != 0.0:
                    P = dl * t1
                    if dyn_a:
                        v[a] -= P * ima
                        w[a] -= Ia @ np.cross(ra, P)
                    v[b] += P * imb
                    w[b] += Ib @ np.cross(rb, P)
                    Jsum += P
                # friction 2
                vr = v[b] + np.cross(w[b], rb)
                if dyn_a:
                    vr = vr - (v[a] + np.cross(w[a], ra))
                vt = float(vr @ t2)
                dl = -vt * kt2_inv
                mx = self.mu * lam
                new2 = min(max(lam2 + dl, -mx), mx)
                dl = new2 - lam2
                lam2 = new2
                if dl != 0.0:
                    P = dl * t2
                    if dyn_a:
                        v[a] -= P * ima
                        w[a] -= Ia @ np.cross(ra, P)
                    v[b] += P * imb
                    w[b] += Ib @ np.cross(rb, P)
                    Jsum += P
                pc[16] = lam
                pc[17] = lam1
                pc[18] = lam2
        for pc in prep:
            a, b = pc[0], pc[1]
            Jsum = pc[19]
            key = (a, b)
            if key in self.pairJ:
                self.pairJ[key] += Jsum
            else:
                self.pairJ[key] = Jsum.copy()

    # ------------------------------------------------------------------ output
    def _make_frame(self, time, dt_int):
        n = self.n
        ext = self.m[:, None] * self.gravity[None, :]
        if self.pairJ:
            keys = sorted(self.pairJ.keys())
            inter = np.stack([self.pairJ[k] for k in keys]) / dt_int
        else:
            inter = np.zeros((0, 3))
        gaps = self.last_gaps
        if gaps is None or gaps.size == 0:
            gaps = np.array([self.NEAR])
        frame = StateFrame(
            time,
            self.c.copy(),
            self.v.copy(),
            orientation=self.q.copy(),
            angular_velocity=self.w.copy(),
            mass=self.m.copy(),
            body_id=self.ids.copy(),
            attributes={},
            fields={
                "contact_signed_distance": np.asarray(gaps, dtype=float),
                "external_force": np.asarray(ext, dtype=float),
                "interaction_force": np.asarray(inter, dtype=float),
            },
            scalars={"contact_length_scale": self.cls},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "substeps": int(self.step_count),
            "last_contact_count": int(self.contacts_count),
            "last_gap_samples": int(len(self.last_gaps)),
        }
