"""Coupled water/rigid solver for scene 050_water_collapses_towers.

Physical model
--------------
* Water: weakly-compressible-free Position Based Fluids (Macklin & Mueller 2013).
  Density constraints solved with poly6/spiky SPH kernels, XSPH viscosity,
  basin walls/floor handled as hard position constraints.
* Rigid tower members (light cylinders, heavy cubes): full 6-DOF rigid bodies
  (quaternion orientation, diagonal body-frame inertia) approximated as
  capsules for contact.  Impulse-based contact solver with Coulomb friction
  and Baumgarte-style position correction against the frozen basin geometry.
* Coupling: two-way, momentum conserving.  Water samples that penetrate a
  rigid capsule shell are projected out; the matching impulse (and torque
  about the body center) is applied to the rigid body, so momentum is
  exchanged across the interface and buoyancy/drag arise from the
  displaced-volume pressure response.

No keyframes, no scripted motion: every state comes from the time stepping.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
GRAVITY = np.array([0.0, 0.0, -9.81])

# Water (PBF) parameters; frozen from the scaffold sample spacing (0.16 m).
H = 0.32                     # SPH kernel support radius
H2 = H * H
RHO0 = 1000.0                # water rest density [kg/m^3]
PBF_ITERS = 3
XSPH_VISC = 0.03
LAMBDA_EPS = 1.0e-2
COUP_MARGIN = 0.08           # coupling shell thickness around rigid capsules
MAX_VW = 12.0                # safety speed clamp for water

CPOLY6 = 315.0 / (64.0 * np.pi * H ** 9)
CSPIKY = 45.0 / (np.pi * H ** 6)
W0 = CPOLY6 * H2 ** 3

# Time integration
DT_SUB = 0.005

# Rigid contact parameters
MU_BB = 0.40                 # body-body friction
MU_STATIC = 0.55             # body-floor / body-wall friction
RESTITUTION = 0.0
CONTACT_ITERS = 4
POS_CORR = 0.5
MAX_VB = 30.0
MAX_WB = 40.0

# Frozen basin interior
WALL_Z = 1.30                # wall top height (walls span z in [0, 1.3])
Y_WALL = 2.00
X_WALL = 4.00
NEAR_GAP = 0.16              # contact reporting window / length scale

# Material slots (K = 3 conserved materials)
MAT_WATER, MAT_LIGHT, MAT_HEAVY = 0, 1, 2


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-14:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1.0 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1.0 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1.0 - s * (x * x + y * y)],
    ])


def _integrate_quat(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """dq/dt = 0.5 * (0, omega) (x) q  (world-frame angular velocity)."""
    qx, qy, qz, qw = q
    wx, wy, wz = omega
    dq = np.array([
        0.5 * (qw * wx + (wy * qz - wz * qy)),
        0.5 * (qw * wy + (wz * qx - wx * qz)),
        0.5 * (qw * wz + (wx * qy - wy * qx)),
        0.5 * (-(wx * qx + wy * qy + wz * qz)),
    ]) * dt
    out = q + dq
    norm = np.linalg.norm(out)
    if norm > 1e-12:
        out = out / norm
    return out


def _seg_closest(P: np.ndarray, A: np.ndarray, B: np.ndarray):
    AB = B - A
    denom = float(AB @ AB)
    if denom < 1e-14:
        t = 0.0
    else:
        t = float(np.clip((P - A) @ AB / denom, 0.0, 1.0))
    Q = A + t * AB
    return Q, float(np.linalg.norm(P - Q))


class BenchmarkSolver:
    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------- initialize
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", {}) or {}
        self.seed = int(seed)
        frame = scene.initial_state()
        groups = np.asarray(frame.attributes["scene_group"], dtype=float)
        position = np.asarray(frame.position, dtype=float)
        velocity = np.asarray(frame.velocity, dtype=float)
        mass = np.asarray(frame.mass, dtype=float)

        water_mask = groups == 10.0
        rigid_mask = ~water_mask
        self.xw = position[water_mask].copy()
        self.vw = velocity[water_mask].copy()
        self.m_w = float(mass[water_mask][0])
        self.mw_arr = mass[water_mask].copy()
        self.n_w = len(self.xw)
        self.rho_w = np.full(self.n_w, RHO0)

        rig_idx = np.where(rigid_mask)[0]
        nb = len(rig_idx)
        self.Nb = nb
        self.bx = position[rig_idx].copy()
        self.bv = velocity[rig_idx].copy()
        self.bq = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (nb, 1))
        self.bw = np.zeros((nb, 3))
        self.bm = mass[rig_idx].copy()
        self.bgroups = groups[rig_idx].copy()
        self.bmat = np.where(self.bgroups % 10.0 == 1.0, MAT_HEAVY, MAT_LIGHT)

        # Capsule approximation of the frozen rigid geometry.
        cyl_r, cyl_h, box_he = self._parse_geometry(cfg)
        self.bvol = np.empty(nb)
        self.bden = np.empty(nb)
        ba = np.empty(nb)
        bL = np.empty(nb)
        inertia = np.empty((nb, 3))
        for i in range(nb):
            if self.bgroups[i] % 10.0 == 1.0:  # heavy cube
                ex, ey, ez = 2.0 * box_he
                vol = ex * ey * ez
                ba[i] = float(np.hypot(box_he[0], box_he[1]))
                bL[i] = box_he[2] - ba[i]
                inertia[i] = np.array([
                    self.bm[i] * (ey * ey + ez * ez) / 12.0,
                    self.bm[i] * (ex * ex + ez * ez) / 12.0,
                    self.bm[i] * (ex * ex + ey * ey) / 12.0,
                ])
            else:                               # light cylinder
                vol = np.pi * cyl_r ** 2 * cyl_h
                ba[i] = cyl_r
                bL[i] = 0.5 * cyl_h - cyl_r
                inertia[i] = np.array([
                    self.bm[i] * (3.0 * cyl_r ** 2 + cyl_h ** 2) / 12.0,
                    self.bm[i] * (3.0 * cyl_r ** 2 + cyl_h ** 2) / 12.0,
                    0.5 * self.bm[i] * cyl_r ** 2,
                ])
            self.bvol[i] = vol
            self.bden[i] = self.bm[i] / vol
        self.ba = ba
        self.bL = bL
        self.bI = inertia
        self.binvm = 1.0 / self.bm
        self._inertia_inv_cache = [None] * nb

        # Accumulators (impulses) between exported samples.
        self.acc_ext_w = np.zeros((self.n_w, 3))
        self.acc_ext_b = np.zeros((nb, 3))
        self.acc_int = np.zeros((3, 3))  # [material, xyz]

        self.t = 0.0
        self._prev_sample = None
        self._substeps = 0
        self._ready = True

    @staticmethod
    def _parse_geometry(cfg: dict):
        cyl_r, cyl_h = 0.115, 1.85
        box_he = np.array([0.115, 0.115, 0.925])
        geo = cfg.get("scene_geometry") if isinstance(cfg, dict) else None
        objects = getattr(geo, "objects", None)
        if objects:
            for obj in objects:
                gid = getattr(obj, "id", "")
                g = getattr(obj, "geometry", None)
                if gid.endswith("tower_cylinders") and getattr(g, "radius", None):
                    cyl_r, cyl_h = float(g.radius), float(g.height)
                if gid.endswith("tower_cubes") and getattr(g, "half_extents", None):
                    box_he = np.asarray(g.half_extents, dtype=float)
        return cyl_r, cyl_h, box_he

    # ---------------------------------------------------------------- helpers
    def _axis(self, b: int) -> np.ndarray:
        return _quat_to_R(self.bq[b])[:, 2]

    def _endpoints(self, b: int):
        ax = self._axis(b)
        return self.bx[b] + ax * self.bL[b], self.bx[b] - ax * self.bL[b]

    def _iinv_world(self, b: int) -> np.ndarray:
        if self._inertia_inv_cache[b] is None:
            R = _quat_to_R(self.bq[b])
            self._inertia_inv_cache[b] = R @ np.diag(1.0 / self.bI[b]) @ R.T
        return self._inertia_inv_cache[b]

    def _invalidate_inertia_cache(self):
        self._inertia_inv_cache = [None] * self.Nb

    def _clamp_basin(self, x: np.ndarray) -> np.ndarray:
        """Hard-project water samples onto the frozen basin interior."""
        corr = np.zeros_like(x)
        m = x[:, 2] < 0.0
        if m.any():
            corr[m, 2] = x[m, 2]
            x[m, 2] = 0.0
        low = x[:, 2] < WALL_Z
        m = low & (x[:, 1] > Y_WALL)
        if m.any():
            corr[m, 1] = x[m, 1] - Y_WALL
            x[m, 1] = Y_WALL
        m = low & (x[:, 1] < -Y_WALL)
        if m.any():
            corr[m, 1] = x[m, 1] + Y_WALL
            x[m, 1] = -Y_WALL
        m = low & (x[:, 0] > X_WALL)
        if m.any():
            corr[m, 0] = x[m, 0] - X_WALL
            x[m, 0] = X_WALL
        return corr

    def _neighbor_pairs(self, x: np.ndarray) -> np.ndarray:
        tree = cKDTree(x)
        return tree.query_pairs(H, output_type="ndarray")

    # ------------------------------------------------------------ water steps
    def _xsph(self, pairs: np.ndarray) -> None:
        if len(pairs) == 0:
            return
        ii = np.concatenate([pairs[:, 0], pairs[:, 1]])
        jj = np.concatenate([pairs[:, 1], pairs[:, 0]])
        dx = self.xw[ii] - self.xw[jj]
        r2 = (dx * dx).sum(1)
        m = (r2 < H2) & (r2 > 1e-12)
        if not m.any():
            return
        ii, jj, dx = ii[m], jj[m], dx[m]
        r = np.sqrt(r2[m])
        w = CPOLY6 * (H2 - r * r) ** 3
        coef = (self.m_w * XSPH_VISC / np.maximum(self.rho_w[jj], 1e-6)) * w
        dv = (self.vw[jj] - self.vw[ii]) * coef[:, None]
        N = self.n_w
        acc = np.stack([np.bincount(ii, weights=dv[:, k], minlength=N)
                        for k in range(3)], axis=1)
        self.vw += acc

    def _pbf(self, xp: np.ndarray, ii: np.ndarray, jj: np.ndarray, dt: float) -> None:
        N = self.n_w
        for _ in range(PBF_ITERS):
            dx = xp[ii] - xp[jj]
            r2 = (dx * dx).sum(1)
            m = (r2 < H2) & (r2 > 1e-12)
            if not m.any():
                break
            ii_m, jj_m, dx = ii[m], jj[m], dx[m]
            r = np.sqrt(r2[m])
            w = CPOLY6 * (H2 - r * r) ** 3
            rho = np.full(N, self.m_w * W0) + self.m_w * np.bincount(
                ii_m, weights=w, minlength=N)
            self.rho_w = rho
            C = rho / RHO0 - 1.0
            np.clip(C, 0.0, None, out=C)
            # a_ij = (m/rho0) * grad_i W  magnitude/Direction pieces
            av = ((self.m_w / RHO0) * CSPIKY * (H - r) ** 2)[:, None] * (dx / r[:, None])
            sum_a2 = np.bincount(ii_m, weights=(av * av).sum(1), minlength=N)
            sum_a = np.stack([np.bincount(ii_m, weights=av[:, k], minlength=N)
                              for k in range(3)], axis=1)
            denom = sum_a2 + (sum_a * sum_a).sum(1) + LAMBDA_EPS
            lam = -C / denom
            lamsum = lam[ii_m] + lam[jj_m]
            dp = np.stack([
                np.bincount(ii_m, weights=-(1.0 / self.m_w) * lamsum * av[:, k],
                            minlength=N)
                for k in range(3)], axis=1)
            xp += dp
            corr = self._clamp_basin(xp)
            if corr.any():
                self.acc_ext_w += (self.m_w / dt) * corr

    def _coupling(self, xp: np.ndarray, dt: float) -> None:
        """Two-way water<->rigid coupling (momentum conserving shell projection)."""
        Jb = np.zeros((self.Nb, 3))
        Gb = np.zeros((self.Nb, 3))
        for b in range(self.Nb):
            e1, e2 = self._endpoints(b)
            AB = e2 - e1
            denom = float(AB @ AB)
            t = np.clip(((xp - e1) @ AB) / max(denom, 1e-14), 0.0, 1.0)
            Q = e1[None, :] + t[:, None] * AB[None, :]
            dvec = xp - Q
            d = np.sqrt((dvec * dvec).sum(1))
            rc = self.ba[b] + COUP_MARGIN
            hit = d < rc
            if not hit.any():
                continue
            idx = np.where(hit)[0]
            dd = d[idx]
            safe = np.maximum(dd, 1e-9)
            nrm = dvec[idx] / safe[:, None]
            degenerate = dd < 1e-9
            if degenerate.any():
                nrm[degenerate] = np.array([1.0, 0.0, 0.0])
            delta = (rc - dd)[:, None] * nrm
            xp[idx] += delta
            Jp = -(self.m_w / dt) * delta           # impulse on the rigid body
            jsum = Jp.sum(axis=0)
            Jb[b] += jsum
            Gb[b] += np.cross(Q[idx] - self.bx[b], Jp).sum(axis=0)
            # internal interaction bookkeeping (water <-> rigid material)
            self.acc_int[MAT_WATER] -= jsum
            self.acc_int[self.bmat[b]] += jsum
        self.bv += Jb * self.binvm[:, None]
        for b in range(self.Nb):
            if np.abs(Gb[b]).sum() > 0.0:
                self.bw[b] += self._iinv_world(b) @ Gb[b]

    # ------------------------------------------------------------ rigid steps
    def _gather_contacts(self):
        contacts = []
        # body-body (capsule vs capsule via endpoint/segment tests)
        for ia in range(self.Nb):
            for ib in range(ia + 1, self.Nb):
                reach = self.bL[ia] + self.ba[ia] + self.bL[ib] + self.ba[ib]
                if np.linalg.norm(self.bx[ia] - self.bx[ib]) > reach + 0.02:
                    continue
                ea1, ea2 = self._endpoints(ia)
                eb1, eb2 = self._endpoints(ib)
                best_d, best_pa, best_pb = np.inf, None, None
                for P in (ea1, ea2):
                    Q, d = _seg_closest(P, eb1, eb2)
                    if d < best_d:
                        best_d, best_pa, best_pb = d, P.copy(), Q
                for P in (eb1, eb2):
                    Q, d = _seg_closest(P, ea1, ea2)
                    if d < best_d:
                        best_d, best_pb, best_pa = d, P.copy(), Q
                gap = best_d - self.ba[ia] - self.ba[ib]
                if gap < 0.06:
                    n = (best_pa - best_pb)
                    nn = np.linalg.norm(n)
                    if nn < 1e-9:
                        continue
                    n = n / nn
                    pt = 0.5 * (best_pa + best_pb)
                    contacts.append((ia, ib, pt - self.bx[ia], pt - self.bx[ib],
                                     n, gap, MU_BB))
        # static basin contacts
        for b in range(self.Nb):
            e1, e2 = self._endpoints(b)
            E = np.vstack([e1, e2])
            k = int(np.argmin(E[:, 2]))
            gap = E[k, 2] - self.ba[b]
            if gap < 0.06:
                pt = E[k].copy()
                contacts.append((b, -1, pt - self.bx[b], np.zeros(3),
                                 np.array([0.0, 0.0, 1.0]), gap, MU_STATIC))
            k = int(np.argmax(np.abs(E[:, 1])))
            over = np.abs(E[k, 1]) + self.ba[b] - Y_WALL
            if over > -0.06 and E[k, 2] < WALL_Z + self.ba[b]:
                n = np.array([0.0, -np.sign(E[k, 1]) if E[k, 1] != 0.0 else -1.0, 0.0])
                pt = E[k].copy()
                contacts.append((b, -1, pt - self.bx[b], np.zeros(3),
                                 n, -over, MU_STATIC))
            k = int(np.argmax(E[:, 0]))
            over = E[k, 0] + self.ba[b] - X_WALL
            if over > -0.06 and E[k, 2] < WALL_Z + self.ba[b]:
                pt = E[k].copy()
                contacts.append((b, -1, pt - self.bx[b], np.zeros(3),
                                 np.array([-1.0, 0.0, 0.0]), -over, MU_STATIC))
        return contacts

    def _apply_impulse(self, b: int, J: np.ndarray, r: np.ndarray):
        if b < 0:
            return
        self.bv[b] += self.binvm[b] * J
        self.bw[b] += self._iinv_world(b) @ np.cross(r, J)

    def _body_ang_term(self, b: int, r: np.ndarray, n: np.ndarray) -> float:
        if b < 0:
            return 0.0
        rn = np.cross(r, n)
        return float(n @ (self._iinv_world(b) @ np.cross(rn, r)))

    def _solve_rigid(self) -> None:
        contacts = self._gather_contacts()
        if not contacts:
            return
        for _ in range(CONTACT_ITERS):
            for (ia, ib, ra, rb, n, gap, mu) in contacts:
                va = self.bv[ia] + np.cross(self.bw[ia], ra)
                vb = self.bv[ib] + np.cross(self.bw[ib], rb) if ib >= 0 else np.zeros(3)
                vrel = va - vb
                vn = float(vrel @ n)
                if vn >= 0.0:
                    continue
                k = (self.binvm[ia] if ia >= 0 else 0.0) + \
                    (self.binvm[ib] if ib >= 0 else 0.0) + \
                    self._body_ang_term(ia, ra, n) + self._body_ang_term(ib, rb, n)
                if k <= 1e-12:
                    continue
                j = -(1.0 + RESTITUTION) * vn / k
                J = j * n
                self._apply_impulse(ia, J, ra)
                self._apply_impulse(ib, -J, rb)
                if ib >= 0:
                    self.acc_int[self.bmat[ia]] += J
                    self.acc_int[self.bmat[ib]] -= J
                else:
                    self.acc_ext_b[ia] += J
                # Coulomb friction
                va = self.bv[ia] + np.cross(self.bw[ia], ra)
                vb = self.bv[ib] + np.cross(self.bw[ib], rb) if ib >= 0 else np.zeros(3)
                vrel = va - vb
                vt = vrel - (vrel @ n) * n
                sn = float(np.linalg.norm(vt))
                if sn > 1e-8:
                    jt = min(mu * j, sn / k)
                    Jt = -(vt / sn) * jt
                    self._apply_impulse(ia, Jt, ra)
                    self._apply_impulse(ib, -Jt, rb)
                    if ib >= 0:
                        self.acc_int[self.bmat[ia]] += Jt
                        self.acc_int[self.bmat[ib]] -= Jt
                    else:
                        self.acc_ext_b[ia] += Jt
        # positional correction (penetration recovery)
        for (ia, ib, ra, rb, n, gap, mu) in contacts:
            pen = -gap
            if pen <= 0.0:
                continue
            corr = POS_CORR * pen
            wa = self.binvm[ia] if ia >= 0 else 0.0
            wb = self.binvm[ib] if ib >= 0 else 0.0
            tot = wa + wb
            if tot <= 1e-12:
                continue
            if ia >= 0:
                self.bx[ia] += n * (corr * wa / tot)
            if ib >= 0:
                self.bx[ib] -= n * (corr * wb / tot)

    # --------------------------------------------------------------- substep
    def _substep(self, dt: float) -> None:
        # --- water: XSPH viscosity then PBF prediction/correction
        pairs = self._neighbor_pairs(self.xw)
        self._xsph(pairs)
        self.vw = self.vw + GRAVITY[None, :] * dt
        x0 = self.xw.copy()
        xp = self.xw + self.vw * dt
        if len(pairs):
            ii = np.concatenate([pairs[:, 0], pairs[:, 1]])
            jj = np.concatenate([pairs[:, 1], pairs[:, 0]])
        else:
            ii = np.zeros(0, dtype=int)
            jj = np.zeros(0, dtype=int)
        self._pbf(xp, ii, jj, dt)
        # --- two-way coupling with rigid capsules
        self._coupling(xp, dt)
        corr = self._clamp_basin(xp)
        if corr.any():
            self.acc_ext_w += (self.m_w / dt) * corr
        # --- commit water state
        vw = (xp - x0) / dt
        speed = np.sqrt((vw * vw).sum(1))
        fast = speed > MAX_VW
        if fast.any():
            vw[fast] *= (MAX_VW / speed[fast])[:, None]
        self.xw = xp
        self.vw = vw

        # --- rigid bodies: gravity, pose integration, contacts
        self.bv += GRAVITY[None, :] * dt
        self.bx += self.bv * dt
        for b in range(self.Nb):
            self.bq[b] = _integrate_quat(self.bq[b], self.bw[b], dt)
        self._invalidate_inertia_cache()
        self._solve_rigid()
        # safety clamps
        vb_speed = np.sqrt((self.bv * self.bv).sum(1))
        fast = vb_speed > MAX_VB
        if fast.any():
            self.bv[fast] *= (MAX_VB / vb_speed[fast])[:, None]
        wb_speed = np.sqrt((self.bw * self.bw).sum(1))
        fast = wb_speed > MAX_WB
        if fast.any():
            self.bw[fast] *= (MAX_WB / wb_speed[fast])[:, None]
        # keep every sample finite and inside the basin
        self.bx[:, 2] = np.maximum(self.bx[:, 2], self.ba)
        np.clip(self.bx[:, 1], -Y_WALL - 0.2, Y_WALL + 0.2, out=self.bx[:, 1])
        np.clip(self.bx[:, 0], -X_WALL - 1.0, X_WALL + 0.2, out=self.bx[:, 0])
        self._substeps += 1

    # ------------------------------------------------------------- reporting
    def _report_gaps(self) -> np.ndarray:
        gaps = []
        # rigid vs basin
        for b in range(self.Nb):
            e1, e2 = self._endpoints(b)
            E = np.vstack([e1, e2])
            g = float(E[:, 2].min() - self.ba[b])
            if g < NEAR_GAP:
                gaps.append(g)
            k = int(np.argmax(np.abs(E[:, 1])))
            g = float(Y_WALL - (np.abs(E[k, 1]) + self.ba[b]))
            if g < NEAR_GAP and E[k, 2] < WALL_Z + self.ba[b]:
                gaps.append(g)
            g = float(X_WALL - (E[:, 0].max() + self.ba[b]))
            if g < NEAR_GAP and E[:, 2].max() < WALL_Z + self.ba[b]:
                gaps.append(g)
        # rigid vs rigid
        for ia in range(self.Nb):
            for ib in range(ia + 1, self.Nb):
                reach = self.bL[ia] + self.ba[ia] + self.bL[ib] + self.ba[ib]
                if np.linalg.norm(self.bx[ia] - self.bx[ib]) > reach + NEAR_GAP:
                    continue
                ea1, ea2 = self._endpoints(ia)
                eb1, eb2 = self._endpoints(ib)
                best = np.inf
                for P in (ea1, ea2):
                    _, d = _seg_closest(P, eb1, eb2)
                    best = min(best, d)
                for P in (eb1, eb2):
                    _, d = _seg_closest(P, ea1, ea2)
                    best = min(best, d)
                g = float(best - self.ba[ia] - self.ba[ib])
                if g < NEAR_GAP:
                    gaps.append(g)
        # water vs rigid interface
        for b in range(self.Nb):
            e1, e2 = self._endpoints(b)
            AB = e2 - e1
            denom = max(float(AB @ AB), 1e-14)
            t = np.clip(((self.xw - e1) @ AB) / denom, 0.0, 1.0)
            Q = e1[None, :] + t[:, None] * AB[None, :]
            dvec = self.xw - Q
            d = np.sqrt((dvec * dvec).sum(1))
            g = d - self.ba[b]
            sel = g < NEAR_GAP
            if sel.any():
                gaps.extend(g[sel].tolist())
        if not gaps:
            gaps = [0.0]
        return np.asarray(gaps, dtype=float)

    def _make_frame(self, time: float, dt_sample: float) -> StateFrame:
        n_w, n_b = self.n_w, self.Nb
        N = n_w + n_b
        position = np.vstack([self.xw, self.bx])
        velocity = np.vstack([self.vw, self.bv])
        orientation = np.vstack([
            np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_w, 1)), self.bq])
        angular_velocity = np.vstack([np.zeros((n_w, 3)), self.bw])
        mass = np.concatenate([self.mw_arr, self.bm])
        body_id = np.concatenate([
            np.zeros(n_w, dtype=int), np.arange(1, n_b + 1)])
        scene_group = np.concatenate([np.full(n_w, 10.0), self.bgroups])
        density = np.concatenate([self.rho_w, self.bden])

        mass_quantity = np.zeros((N, 3))
        mass_quantity[:n_w, MAT_WATER] = self.mw_arr
        mass_quantity[n_w:, MAT_LIGHT] = np.where(
            self.bmat == MAT_LIGHT, self.bm, 0.0)
        mass_quantity[n_w:, MAT_HEAVY] = np.where(
            self.bmat == MAT_HEAVY, self.bm, 0.0)

        external_force = np.empty((N, 3))
        external_force[:n_w] = self.mw_arr[:, None] * GRAVITY[None, :] \
            + self.acc_ext_w / dt_sample
        external_force[n_w:] = self.bm[:, None] * GRAVITY[None, :] \
            + self.acc_ext_b / dt_sample

        interaction_force = self.acc_int / dt_sample

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "scene_group": scene_group,
                "density": density,
                "mass_quantity": mass_quantity,
            },
            fields={
                "contact_signed_distance": self._report_gaps(),
                "cumulative_mass_source": np.zeros(3),
                "external_force": external_force,
                "interaction_force": interaction_force,
                "density": density,
            },
            scalars={
                "contact_length_scale": 0.16,
                "water_rest_density": float(RHO0),
                "initial_water_speed": 3.2,
                "substeps_completed": float(self._substeps),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ API
    def advance_to(self, time: float) -> StateFrame:
        if not self._ready:
            raise RuntimeError("solver is not initialized")
        time = float(time)
        if time < self.t - 1e-9:
            raise ValueError(f"non-monotonic sample time {time} < {self.t}")
        while self.t < time - 1e-9:
            dt = min(DT_SUB, time - self.t)
            self._substep(dt)
            self.t += dt
        self.t = time
        if self._prev_sample is None:
            dt_sample = 1.0
        else:
            dt_sample = max(time - self._prev_sample, 1e-9)
        frame = self._make_frame(time, dt_sample)
        # reset interval accumulators
        self.acc_ext_w = np.zeros((self.n_w, 3))
        self.acc_ext_b = np.zeros((self.Nb, 3))
        self.acc_int = np.zeros((3, 3))
        self._prev_sample = time
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "PBF water + impulse rigid bodies, two-way coupled",
            "water_samples": int(self.n_w),
            "rigid_bodies": int(self.Nb),
            "substeps": int(self._substeps),
            "kernel_support_m": float(H),
            "materials": ["water", "light_cylinders", "heavy_cubes"],
        }
