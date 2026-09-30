"""Rigid-body solver: a driven bolt spins through a nut.

Physics model
-------------
Full 6-DOF rigid body dynamics (semi-implicit / symplectic Euler with
sub-stepping) for the bolt and nut, with:

* gravity on every body,
* a prescribed-velocity motor torque driving the bolt about its own axis,
* helical thread contact between bolt and nut, derived from the frozen
  thread geometry (pitch, handedness, tooth/groove phases, backlash) that
  is measured from the frozen mesh assets at initialization,
* regularized Coulomb friction at every contact point,
* floor contacts for the nut and (as a safeguard) the bolt tip.

The thread interface is modelled as a set of discrete contact points
distributed around the engaged thread turns.  The screw invariant

    Phi = (t_b - t_n) - h*(p/2pi)*(psi_b - psi_n) + (c_b - c_n)

measures the axial offset of the bolt tooth from the nut groove centre
(modulo the pitch).  Flank contact acts along the true helical flank
normal  n = s*(h*sin(lam)*t - cos(lam)*z)  (lead angle lam = p/(2 pi r_c)),
so axial force and torque are correctly coupled: rotating the bolt makes
it advance axially, exactly the screw mechanism of the paper's Figure 4.

No keyframes, no precomputed motion: every state is produced by the
time-stepping procedure below.
"""
from __future__ import annotations

import math
from pathlib import Path

from worldsolver.contracts import StateFrame

import numpy as np

TWO_PI = 2.0 * math.pi
EZ = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------- Quaternion
def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _q_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n <= 0.0:
        return np.eye(3)
    s = 2.0 / n
    x, y, z, w = x * s, y * s, z * s, w * s
    return np.array([
        [1.0 - (y * y + z * z), x * y - w * z, x * z + w * y],
        [x * y + w * z, 1.0 - (x * x + z * z), y * z - w * x],
        [x * z - w * y, y * z + w * x, 1.0 - (x * x + y * y)],
    ])


def _wrap(x: float, p: float) -> float:
    return (x + 0.5 * p) % p - 0.5 * p


# ---------------------------------------------------------------- Mesh input
def _load_vertices(path: Path) -> np.ndarray:
    verts = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"no vertices in {path}")
    return np.asarray(verts, dtype=float)


def _find_asset(name: str) -> Path:
    here = Path(__file__).resolve()
    cands = [
        here.parent.parent / "assets" / name,
        here.parent / "assets" / name,
        Path("assets") / name,
        Path(name),
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(name)


# ------------------------------------------------- Thread geometry extraction
def _fit_helix(verts: np.ndarray, rlo: float, rhi: float, zlo: float, zhi: float):
    """Best (handedness, pitch, invariant centre) for a helical feature."""
    r = np.hypot(verts[:, 0], verts[:, 1])
    z = verts[:, 2]
    m = (r > rlo) & (r < rhi) & (z > zlo) & (z < zhi)
    if m.sum() < 8:
        m = (r > rlo) & (r < rhi)
    th = np.arctan2(verts[m, 1], verts[m, 0])
    zz = z[m]
    best = (-1.0, 1.0, 0.2)
    for h in (1.0, -1.0):
        for p in np.arange(0.08, 0.60, 0.002):
            iv = zz - h * (p / TWO_PI) * th
            conc = abs(np.mean(np.exp(1j * TWO_PI * iv / p)))
            if conc > best[0]:
                best = (conc, h, float(p))
    conc, h, p0 = best
    for p in np.arange(max(0.02, p0 - 0.004), p0 + 0.0041, 0.0002):
        iv = zz - h * (p / TWO_PI) * th
        c = abs(np.mean(np.exp(1j * TWO_PI * iv / p)))
        if c > conc:
            conc, p0 = c, float(p)
    iv = zz - h * (p0 / TWO_PI) * th
    ph = np.exp(1j * TWO_PI * iv / p0)
    centre = math.atan2(float(np.mean(np.imag(ph))), float(np.mean(np.real(ph))))
    c0 = centre / TWO_PI * p0
    return h, p0, c0, conc, m


def _halfwidth_profile(verts, mask, centre, p, h, rlo, rhi, nbins=14):
    """Axial half-width of a helical feature as a function of radius."""
    r = np.hypot(verts[:, 0], verts[:, 1])
    th = np.arctan2(verts[:, 1], verts[:, 0])
    iv = verts[:, 2] - h * (p / TWO_PI) * th
    off = np.abs(_wrap_vec(iv - centre, p))[mask]
    rr = r[mask]
    prof_r, prof_w = [], []
    for b in range(nbins):
        lo = rlo + (rhi - rlo) * b / nbins
        hi = rlo + (rhi - rlo) * (b + 1) / nbins
        sel = (rr >= lo) & (rr < hi)
        if sel.sum() >= 2:
            prof_r.append(0.5 * (lo + hi))
            prof_w.append(float(off[sel].max()))
    return np.asarray(prof_r), np.asarray(prof_w)


def _wrap_vec(x: np.ndarray, p: float) -> np.ndarray:
    return (x + 0.5 * p) % p - 0.5 * p


# ---------------------------------------------------------------- Body state
class _Body:
    def __init__(self, mass, com_local, inertia_local, pos, quat):
        self.m = float(mass)
        self.com_local = np.asarray(com_local, float)
        self.I_local = np.asarray(inertia_local, float)
        self.x = np.asarray(pos, float).copy()        # body origin (mesh frame)
        self.q = np.asarray(quat, float).copy()       # xyzw
        self.v = np.zeros(3)                          # COM velocity
        self.w = np.zeros(3)                          # angular velocity

    @property
    def R(self):
        return _q_to_R(self.q)

    def com_world(self):
        return self.x + self.R @ self.com_local

    def origin_velocity(self):
        return self.v - np.cross(self.w, self.R @ self.com_local)

    def world_inertia(self):
        R = self.R
        return R @ self.I_local @ R.T


def _point_mass_inertia(verts: np.ndarray, mass: float):
    """COM and inertia tensor from an equal-mass vertex cloud."""
    com = verts.mean(axis=0)
    r = verts - com
    n = len(verts)
    mp = mass / n
    I = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                I[i, j] = mp * float(np.sum(r[:, 0] ** 2 + r[:, 1] ** 2 + r[:, 2] ** 2)
                                     - np.sum(r[:, i] ** 2))
            else:
                I[i, j] = -mp * float(np.sum(r[:, i] * r[:, j]))
    # regularize (avoid singular inertia for near-degenerate clouds)
    I += np.eye(3) * (1e-6 * max(1.0, abs(I).max()))
    return com, 0.5 * (I + I.T)


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


class BenchmarkSolver:
    # --- numerical parameters (penalty contact, semi-implicit Euler) ---
    DT = 2.5e-4
    KF = 1.0e6              # N/m  thread flank contact stiffness
    CD = 800.0              # Ns/m normal damping
    FCAP = 2000.0           # N    normal force cap
    KR = 2.0e5              # radial thread contact stiffness
    CR = 300.0
    FCAP_R = 800.0
    KT = 3000.0             # tilt (thread jam) stiffness  [N m/rad]
    CT = 60.0
    FCAP_T = 150.0
    KF_FLOOR = 2.0e6
    CD_FLOOR = 2000.0
    FCAP_FLOOR = 8000.0
    OMEGA_REF = -3.0        # rad/s  motor target (CW -> descent, RH thread)
    KM_MOTOR = 25.0
    TAUMAX = 8.0
    VREG = 0.02             # friction regularization speed (thread)
    VREG_FLOOR = 0.01
    N_ANG = 12              # circumferential thread contact samples
    N_FLOOR = 8
    N_TIPPTS = 6
    FLOOR_Z = 0.0           # top surface of the frozen floor box

    def initialize(self, scene, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self._rng = rng  # determinism hook (not required for the dynamics)
        cfg = scene.config
        self.gravity = np.asarray(cfg["gravity"], float)
        self.friction = float(cfg["contact"]["friction"])
        self.length_scale = float(cfg["contact"]["length_scale"])
        bodies = cfg["bodies"]

        init = scene.initial_state()
        pos0 = np.asarray(init.position, float)
        quat0 = np.asarray(init.orientation, float)

        bv = _load_vertices(_find_asset("threaded_bolt.obj"))
        nv = _load_vertices(_find_asset("threaded_nut.obj"))
        self._bolt_verts = bv
        self._nut_verts = nv

        # ---------------- bolt thread geometry ----------------
        r_bv = np.hypot(bv[:, 0], bv[:, 1])
        thr = bv[(bv[:, 2] > -0.6) & (bv[:, 2] < 0.6)]
        r_thr = np.hypot(thr[:, 0], thr[:, 1])
        r_root = float(np.percentile(r_thr, 5))
        r_crest_b = float(np.percentile(r_thr, 99.5))
        h_b, p_b, c_b, conc_b, m_b = _fit_helix(
            bv, r_root + 0.6 * (r_crest_b - r_root), r_crest_b + 1e-6, -0.6, 0.55)
        crest_z = bv[m_b, 2]
        self.zb_lo = float(crest_z.min() - 0.1 * p_b)
        self.zb_hi = float(crest_z.max() + 0.1 * p_b)

        # ---------------- nut thread geometry ----------------
        # (pitch/handedness must match the bolt; only the phase is fitted)
        r_n = np.hypot(nv[:, 0], nv[:, 1])
        z_nv = nv[:, 2]
        nut_thr = nv[(np.abs(z_nv) < 0.85 * np.percentile(np.abs(z_nv), 90))
                     & (r_n < 0.6 * r_n.max())]
        r_nt = np.hypot(nut_thr[:, 0], nut_thr[:, 1])
        r_hole = float(np.percentile(r_nt, 1))
        r_back = float(np.percentile(r_nt, 99))
        tip_mask = (r_n > r_hole - 1e-9) & (r_n < r_hole + 0.25 * (r_back - r_hole))
        th_n = np.arctan2(nv[tip_mask, 1], nv[tip_mask, 0])
        iv_n = nv[tip_mask, 2] - h_b * (p_b / TWO_PI) * th_n
        ph_n = np.exp(1j * TWO_PI * iv_n / p_b)
        c_tooth = math.atan2(float(np.mean(np.imag(ph_n))),
                             float(np.mean(np.real(ph_n)))) / TWO_PI * p_b
        self.zn_lo = float(nv[tip_mask, 2].min())
        self.zn_hi = float(nv[tip_mask, 2].max())

        # use the bolt fit as the reference thread definition
        self.h = h_b
        self.p = p_b
        self.c_b = c_b
        self.c_n = (c_tooth + 0.5 * self.p) % self.p  # nut groove centre

        # lead angle at the mean contact radius
        self.r_c = 0.5 * (r_crest_b + r_hole)
        self.tan_lam = self.p / (TWO_PI * self.r_c)
        self.lam = math.atan(self.tan_lam)
        self.cos_lam = math.cos(self.lam)
        self.sin_lam = math.sin(self.lam)
        self.r_crest_b = r_crest_b
        self.r_hole = r_hole
        self.r_back = r_back

        # backlash from radial half-width profiles of tooth / groove
        c_groove = self.c_n
        br, bw = _halfwidth_profile(
            bv, m_b, self.c_b, self.p, self.h,
            r_root + 0.25 * (r_crest_b - r_root), r_crest_b + 1e-6)
        nut_wall_mask = ((r_n > r_hole - 1e-9) & (r_n < r_back + 1e-6)
                         & (z_nv > self.zn_lo - 1e-6) & (z_nv < self.zn_hi + 1e-6))
        nr, nw = _halfwidth_profile(
            nv, nut_wall_mask, c_groove, self.p, self.h, r_hole, r_back + 1e-6)
        b_half = 0.03
        if len(br) and len(nr):
            rlo = max(br.min(), nr.min()) + 1e-4
            rhi = min(br.max(), nr.max()) - 1e-4
            if rhi > rlo:
                rgrid = np.linspace(rlo, rhi, 30)
                bw_i = np.interp(rgrid, br, bw)
                nw_i = np.interp(rgrid, nr, nw)
                b_half = float(np.clip(np.min(nw_i - bw_i), 0.008, 0.06))
        self.b_half = b_half
        self.c_play = float(np.clip(0.8 * (self.r_back - self.r_crest_b), 0.004, 0.02))

        # ---------------- rigid bodies ----------------
        com_b, I_b = _point_mass_inertia(bv, float(bodies[0]["mass"]))
        com_n, I_n = _point_mass_inertia(nv, float(bodies[1]["mass"]))
        self.bolt = _Body(bodies[0]["mass"], com_b, I_b, pos0[0], quat0[0])
        self.nut = _Body(bodies[1]["mass"], com_n, I_n, pos0[1], quat0[1])
        self.mass = np.asarray([self.bolt.m, self.nut.m], float)

        # bolt tip face (flat disc at the lowest z of the bolt mesh)
        self.tip_z = float(bv[:, 2].min())
        self.tip_r = float(np.percentile(np.hypot(
            bv[bv[:, 2] < self.tip_z + 1e-6, 0], bv[bv[:, 2] < self.tip_z + 1e-6, 1]), 90))
        # nut bottom face
        self.nut_bot = float(nv[:, 2].min())
        self.nut_bot_r = 0.68 * float(np.percentile(np.hypot(
            nv[nv[:, 2] < self.nut_bot + 1e-6, 0], nv[nv[:, 2] < self.nut_bot + 1e-6, 1]), 90))

        self.t = 0.0
        self._steps = 0
        self._last = None
        self._diag = {"engaged": False, "min_gap": float("inf")}

    # ---------------------------------------------------------- contact model
    def _contacts(self, collect: bool):
        """Compute all contact forces. Returns (Fb, Tb, Fn, Tn, forces, gaps)."""
        bolt, nut = self.bolt, self.nut
        Rb, Rn = bolt.R, nut.R
        cb, cn = bolt.com_world(), nut.com_world()
        zb = Rb @ EZ  # bolt axis (world)
        zn = Rn @ EZ  # nut axis (world)
        vb, vn = bolt.v, nut.v
        wb, wn = bolt.w, nut.w
        vob, von = bolt.origin_velocity(), nut.origin_velocity()

        Fb = np.zeros(3); Tb = np.zeros(3)
        Fn = np.zeros(3); Tn = np.zeros(3)
        forces, gaps = [], []

        def apply(P, f_b, f_n=None):
            nonlocal Fb, Tb, Fn, Tn
            Fb = Fb + f_b
            Tb = Tb + np.cross(P - cb, f_b)
            fn = -f_b if f_n is None else f_n
            Fn = Fn + fn
            Tn = Tn + np.cross(P - cn, fn)

        mu = self.friction
        p, h = self.p, self.h
        cl, sl = self.cos_lam, self.sin_lam

        # ---------- screw invariant ----------
        Rrel = Rn.T @ Rb
        yaw = math.atan2(Rrel[1, 0], Rrel[0, 0])
        tb_n = Rn.T @ (bolt.x - nut.x)
        Phi = tb_n[2] - h * (p / TWO_PI) * yaw + (self.c_b - self.c_n)
        dw = _wrap(Phi, p)
        sw = 1.0 if dw >= 0.0 else -1.0
        adw = abs(dw)

        vz_rel = float((vob - von) @ zn)
        wz_rel = float((wb - wn) @ zn)
        Phi_dot = vz_rel - h * (p / TWO_PI) * wz_rel

        # axial position of the bolt thread region in the nut frame
        zb_lo_n = tb_n[2] + self.zb_lo
        zb_hi_n = tb_n[2] + self.zb_hi
        ov = min(zb_hi_n, self.zn_hi) - max(zb_lo_n, self.zn_lo)
        d_entry = zb_lo_n - self.zn_hi

        # ---------- thread flank contacts ----------
        nlev = 0
        if ov > 1e-4:
            nlev = int(_clamp(round(ov / p) + 1, 1, 3))
            zl = np.linspace(max(zb_lo_n, self.zn_lo), min(zb_hi_n, self.zn_hi),
                             nlev + 2)[1:-1]
            g_flank = (self.b_half - adw) * cl
            gdot = -sw * cl * Phi_dot
            delta = max(0.0, -g_flank)
            ddot = -gdot
            N = 0.0
            if delta > 0.0 or ddot > 0.0:
                N = _clamp(self.KF * delta + self.CD * ddot, 0.0, self.FCAP)
            Npt = N / (nlev * self.N_ANG)
            ang = np.arange(self.N_ANG) * (TWO_PI / self.N_ANG)
            for zj in zl:
                for th_i in ang:
                    pnl = np.array([self.r_c * math.cos(th_i),
                                    self.r_c * math.sin(th_i), zj])
                    P = nut.x + Rn @ pnl
                    nn = Rn @ (np.array([-math.sin(th_i), math.cos(th_i), 0.0]) * (sw * h * sl)
                               - np.array([0.0, 0.0, sw * cl]))
                    f = Npt * nn
                    # regularized Coulomb friction
                    vp = (vb + np.cross(wb, P - cb)) - (vn + np.cross(wn, P - cn))
                    vt = vp - (vp @ nn) * nn
                    f = f - mu * Npt * vt / math.sqrt(vt @ vt + self.VREG ** 2)
                    apply(P, f)
                    if collect:
                        forces.append(f)
                        gaps.append(g_flank)
            self._diag["engaged"] = N > 0.0

        # ---------- thread entry (tip / plateau) contact ----------
        ride = max(0.0, adw - self.b_half) * cl
        g_tip = d_entry - ride
        if ov <= 1e-4 and d_entry > -0.5 * p:
            ridedot = sw * cl * Phi_dot if adw > self.b_half else 0.0
            gdot_t = vz_rel - ridedot
            delta = max(0.0, -g_tip)
            ddot_t = -gdot_t
            N = 0.0
            if delta > 0.0 or ddot_t > 0.0:
                N = _clamp(self.KF * delta + self.CD * ddot_t, 0.0, self.FCAP)
            if N > 0.0:
                # axial support on the bolt axis point of the thread bottom
                P = bolt.x + Rb @ np.array([0.0, 0.0, self.zb_lo])
                f = N * zn
                apply(P, f)
                # plateau sliding friction: pure torque about the nut axis
                tq = mu * N * self.r_c
                sdir = (wb - wn) @ zn
                if abs(sdir) > 1e-9:
                    Tb = Tb - tq * math.copysign(1.0, sdir) * zn
                    Tn = Tn + tq * math.copysign(1.0, sdir) * zn
                if collect:
                    forces.append(f)
                    gaps.append(g_tip)

        # ---------- radial thread contacts (centering / tilt) ----------
        if ov > 1e-4:
            zb_n_dir = Rn.T @ zb
            for zj in (-0.10, 0.10):
                zj = _clamp(zj, self.zn_lo + 0.02, self.zn_hi - 0.02)
                if abs(zb_n_dir[2]) < 1e-6:
                    continue
                s = (zj - tb_n[2]) / zb_n_dir[2]
                axis_pt_n = tb_n + s * zb_n_dir
                ell = axis_pt_n[:2]
                el = float(np.linalg.norm(ell))
                if el > 1e-9:
                    gap_r = self.c_play - el
                    if gap_r < 0.02:
                        P = nut.x + Rn @ np.array([axis_pt_n[0], axis_pt_n[1], zj])
                        nrm = Rn @ np.array([ell[0] / el, ell[1] / el, 0.0])
                        vp = (vb + np.cross(wb, P - cb)) - (vn + np.cross(wn, P - cn))
                        vrad = float(vp @ nrm)
                        delta = max(0.0, -gap_r)
                        N = 0.0
                        if delta > 0.0 or vrad < 0.0:
                            N = _clamp(self.KR * delta - self.CR * min(vrad, 0.0),
                                       0.0, self.FCAP_R)
                        if N > 0.0:
                            f = -N * nrm  # pushes the bolt back toward the axis
                            apply(P, f)
                        if collect:
                            forces.append(f)
                            gaps.append(gap_r)
                else:
                    if collect:
                        forces.append(np.zeros(3))
                        gaps.append(self.c_play)

        # ---------- thread jam (relative tilt) contact ----------
        mis = zb - zn
        mi = float(np.linalg.norm(mis))
        gap_t = 0.03 - mi
        if gap_t < 0.01:
            if mi > 1e-9:
                nrm = mis / mi
                wrel = wb - wn
                vtilt = float(np.cross(wrel, zb) @ nrm)
                delta = max(0.0, -gap_t)
                N = 0.0
                if delta > 0.0 or vtilt < 0.0:
                    N = _clamp(self.KT * delta - self.CT * min(vtilt, 0.0),
                               0.0, self.FCAP_T)
                if N > 0.0:
                    # restoring torque about the axis perpendicular to the tilt
                    tq = -N * np.cross(zn, nrm)
                    Tb = Tb + tq
                    Tn = Tn - tq
                else:
                    tq = np.zeros(3)
                if collect:
                    forces.append(tq)
                    gaps.append(gap_t)

        # ---------- floor contacts ----------
        for (body, is_nut) in ((self.nut, True), (self.bolt, False)):
            R = body.R
            if is_nut:
                pts = [np.array([self.nut_bot_r * math.cos(a),
                                 self.nut_bot_r * math.sin(a), self.nut_bot])
                       for a in np.arange(self.N_FLOOR) * (TWO_PI / self.N_FLOOR)]
            else:
                pts = [np.array([self.tip_r * math.cos(a),
                                 self.tip_r * math.sin(a), self.tip_z])
                       for a in np.arange(self.N_TIPPTS) * (TWO_PI / self.N_TIPPTS)]
            cw = body.com_world()
            for pl in pts:
                P = body.x + R @ pl
                gap = P[2] - self.FLOOR_Z
                if gap > 0.05:
                    continue
                nn = EZ
                vp = body.v + np.cross(body.w, P - cw)
                delta = max(0.0, -gap)
                vz = float(vp @ nn)
                N = 0.0
                if delta > 0.0 or vz < 0.0:
                    N = _clamp(self.KF_FLOOR * delta - self.CD_FLOOR * min(vz, 0.0),
                               0.0, self.FCAP_FLOOR)
                if N > 0.0:
                    f = N * nn
                    vt = vp - (vp @ nn) * nn
                    vt[2] = 0.0
                    f = f - mu * N * vt / math.sqrt(vt @ vt + self.VREG_FLOOR ** 2)
                    if is_nut:
                        apply(P, np.zeros(3), f)  # force on the nut only
                    else:
                        apply(P, f, np.zeros(3))  # force on the bolt only
                    if collect:
                        forces.append(f)
                        gaps.append(gap)

        return Fb, Tb, Fn, Tn, forces, gaps

    # ------------------------------------------------------------- dynamics
    def _step(self, dt: float) -> None:
        bolt, nut = self.bolt, self.nut
        Fb, Tb, Fn, Tn, _, _ = self._contacts(False)

        # gravity
        Fb = Fb + bolt.m * self.gravity
        Fn = Fn + nut.m * self.gravity

        # motor: velocity-controlled torque about the bolt's own axis
        ax = bolt.R @ EZ
        w_ax = float(bolt.w @ ax)
        tq = _clamp(self.KM_MOTOR * (self.OMEGA_REF - w_ax), -self.TAUMAX, self.TAUMAX)
        Tb = Tb + tq * ax

        # semi-implicit Euler on the COM velocities
        bolt.v = bolt.v + (Fb / bolt.m) * dt
        nut.v = nut.v + (Fn / nut.m) * dt

        for body, tau in ((bolt, Tb), (nut, Tn)):
            Iw = body.world_inertia()
            L = Iw @ body.w
            wdot = np.linalg.solve(Iw, tau - np.cross(body.w, L))
            body.w = body.w + wdot * dt
            body.x = body.x + body.v * dt
            dq = _qmul(np.array([body.w[0], body.w[1], body.w[2], 0.0]), body.q)
            body.q = body.q + 0.5 * dt * dq
            body.q = body.q / np.linalg.norm(body.q)

        self._steps += 1

    # --------------------------------------------------------------- frames
    def advance_to(self, time: float):
        t = float(time)
        if not (self.t < t - 1e-9):
            if self._last is not None and abs(self._last[0] - t) < 1e-12:
                return self._last[1]
            if self.t > t + 1e-9:
                # requested time in the past: return the current state
                return self._frame(t)
        while self.t < t - 1e-12:
            dt = min(self.DT, t - self.t)
            self._step(dt)
            self.t = self.t + dt
        frame = self._frame(t)
        self._last = (t, frame)
        return frame

    def _frame(self, t: float):
        bolt, nut = self.bolt, self.nut
        Fb, Tb, Fn, Tn, forces, gaps = self._contacts(True)
        Fb = Fb + bolt.m * self.gravity
        Fn = Fn + nut.m * self.gravity

        pos = np.vstack([bolt.x, nut.x])
        vel = np.vstack([bolt.origin_velocity(), nut.origin_velocity()])
        ori = np.vstack([bolt.q, nut.q])
        omg = np.vstack([bolt.w, nut.w])

        if len(forces) == 0:
            forces = [np.zeros(3)]
            gaps = [0.0]
        inter = np.asarray(forces, float).reshape(-1, 3)
        cg = np.asarray(gaps, float).reshape(-1)
        ext = np.vstack([bolt.m * self.gravity, nut.m * self.gravity])

        frame = StateFrame(
            time=float(t),
            position=pos,
            velocity=vel,
            orientation=ori,
            angular_velocity=omg,
            mass=self.mass.copy(),
            body_id=np.asarray([0, 1], dtype=np.int32),
            attributes={"release_time": np.asarray([0.0, 0.0])},
            fields={
                "external_force": ext,
                "interaction_force": inter,
                "contact_signed_distance": cg,
            },
            scalars={
                "contact_length_scale": float(self.length_scale),
                "characteristic_length": float(self.p),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": self._steps,
            "time": self.t,
            "pitch": self.p,
            "handedness": self.h,
            "backlash": 2.0 * self.b_half,
            "engaged": self._diag.get("engaged", False),
        }
