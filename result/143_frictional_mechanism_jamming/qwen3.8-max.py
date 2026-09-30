"""Physically-based rigid-body solver for the frictional mechanism jamming scene.

Eight L-shaped links (each a union of two boxes: a flat slab plus an upright
prong at its +x end) form an interlocking chain confined between two static
guide rails.  Gravity settles the assembly, and a prescribed driver force
pushes the left-most link along +x, loading the chain.  Penalty (spring-damper)
contacts with regularized Coulomb friction propagate forces through the
prong/slab joints until friction locks the mechanism (jamming), following the
qualitative behaviour of Kaufman et al., "Fast Frictional Dynamics for Rigid
Bodies" (Figs. 1(d), 4).

Numerics: semi-implicit Euler at 1 ms substeps, box feature-point contacts,
broadphase AABB pruning, stiffness ramped during the first second so that the
small frozen assembly interferences relax smoothly instead of exploding.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
DT = 1.0e-3               # physics substep
SKIN = 0.003              # collision shrink per box face (numerical clearance)
K_MAX = 4.0e4             # penalty stiffness (N/m) at full ramp
ZETA = 0.9                # contact damping ratio (near-inelastic, e~0.12)
F_CAP = 4.0e3             # per-feature contact force cap (N)
MARGIN = 0.012            # near-contact margin for feature generation
FRIC_EPS = 2.0e-3         # friction regularization slip scale (m/s)
LIN_DAMP = 0.25           # weak global linear damping (1/s)
ANG_DAMP = 1.5            # weak global angular damping (1/s)
V_MAX = 15.0
W_MAX = 40.0
DRIVE_BODY = 0            # driven link (link_0)
DRIVE_FORCE = 30.0        # N, +x, ramped over 1 s (below slide threshold -> jams)
DRIVE_RAMP = 1.0
K_RAMP_TIME = 1.0         # stiffness ramp duration

G_vec = None  # set in initialize


def quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def box_features(half: np.ndarray) -> np.ndarray:
    """14 feature points (8 corners + 6 face centers) of a box in its frame."""
    hx, hy, hz = half
    corners = np.array([[sx * hx, sy * hy, sz * hz]
                        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = np.array([
        [hx, 0, 0], [-hx, 0, 0], [0, hy, 0], [0, -hy, 0], [0, 0, hz], [0, 0, -hz]])
    return np.vstack([corners, faces])


class BenchmarkSolver:
    # ------------------------------------------------------------ lifecycle
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        bodies = cfg["bodies"]
        self.seed = int(seed)
        self.mu = float(cfg["contact"]["friction"])
        self.length_scale = float(cfg["contact"]["length_scale"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        nb = len(bodies)
        self.nb = nb
        self.mass = np.asarray([float(b["mass"]) for b in bodies])
        pos0 = np.asarray([b["position"] for b in bodies], dtype=float)
        quat0 = np.asarray([b.get("orientation", [0., 0., 0., 1.]) for b in bodies],
                           dtype=float)
        quat0 /= np.linalg.norm(quat0, axis=1, keepdims=True)

        # ---- L-link shape (mesh local frame), split into two mass parts:
        # slab: x[-0.24,0.24] y[+-0.16] z[-0.16,0]; prong: x[0.08,0.24] z[0,0.30]
        v_slab = 0.48 * 0.32 * 0.16
        v_prong = 0.16 * 0.32 * 0.30
        v_union = 0.48 * 0.32 * 0.16 + 0.16 * 0.32 * 0.46 - 0.16 * 0.32 * 0.16
        m_slab = self.mass[0] * v_slab / v_union
        m_prong = self.mass[0] * v_prong / v_union
        c_slab = np.array([0.0, 0.0, -0.08])
        c_prong = np.array([0.16, 0.0, 0.15])
        h_slab = np.array([0.24, 0.16, 0.08])
        h_prong = np.array([0.08, 0.16, 0.15])

        m_tot = m_slab + m_prong
        c_com = (m_slab * c_slab + m_prong * c_prong) / m_tot  # COM in mesh frame
        I_o = np.zeros((3, 3))
        for mp, cp, hp in ((m_slab, c_slab, h_slab), (m_prong, c_prong, h_prong)):
            Ib = mp / 12.0 * np.diag([hp[1] ** 2 + hp[2] ** 2,
                                      hp[0] ** 2 + hp[2] ** 2,
                                      hp[0] ** 2 + hp[1] ** 2])
            I_o += Ib + mp * (np.dot(cp, cp) * np.eye(3) - np.outer(cp, cp))
        I_com = I_o - m_tot * (np.dot(c_com, c_com) * np.eye(3) - np.outer(c_com, c_com))
        self.c_com = c_com
        self.I_inv_body = np.linalg.inv(I_com)

        # ---- dynamic collision boxes in COM-local frame (skin-shrunk)
        s = SKIN
        self.box_c_local = np.array([c_slab - c_com,
                                     np.array([0.16, 0.0, 0.07]) - c_com])  # (2,3)
        self.box_h_local = np.array([h_slab - s,
                                     np.array([0.08, 0.16, 0.23]) - s])      # (2,3)
        self.box_feat_local = [box_features(h) for h in self.box_h_local]    # 2 x (14,3)

        # ---- state (COM based)
        R0 = np.stack([quat_to_R(q) for q in quat0])
        self.q = quat0.copy()
        self.p_com = pos0 + np.einsum("nij,j->ni", R0, c_com)
        self.v = np.zeros((nb, 3))
        self.w = np.zeros((nb, 3))
        self.inv_m = 1.0 / self.mass
        self.R = R0

        # ---- static boxes: floor, guide_lower, guide_upper
        self.stat_c = np.array([[0.0, 0.0, -0.05], [0.0, 0.0, 0.12], [0.0, 0.0, 1.02]])
        self.stat_h = np.array([[3.2, 2.2, 0.05], [2.25, 0.28, 0.08], [2.25, 0.28, 0.08]])
        self.stat_feat = [box_features(h) for h in self.stat_h]
        ns = len(self.stat_c)
        # global box table: [0..2*nb) dynamic (body i boxes 2i,2i+1), then static
        self.nbox = 2 * nb + ns
        self.box_owner = np.array([i // 2 for i in range(2 * nb)] + [-1 - j for j in range(ns)])
        # candidate pairs (at least one dynamic, different bodies)
        pairs = []
        for a in range(self.nbox):
            for b in range(a + 1, self.nbox):
                oa, ob = self.box_owner[a], self.box_owner[b]
                if oa == ob:
                    continue
                if oa < 0 and ob < 0:
                    continue
                pairs.append((a, b))
        self.pairs = pairs
        self.h_arr = np.vstack([np.tile(self.box_h_local, (nb, 1)), self.stat_h])
        self.feat_arr = ([self.box_feat_local[i % 2] for i in range(2 * nb)]
                         + self.stat_feat)

        self.step = 0
        self.t = 0.0
        self.time = 0.0
        self._max_pen = 0.0
        self._max_speed = 0.0

    # ------------------------------------------------------------ helpers
    def _refresh(self):
        self.R = np.stack([quat_to_R(q) for q in self.q])

    def _box_world(self):
        """World centers, rotations, half extents, AABBs for all boxes."""
        nb = self.nb
        cw = np.empty((self.nbox, 3))
        Rw = np.empty((self.nbox, 3, 3))
        for i in range(nb):
            Ri = self.R[i]
            for j in range(2):
                k = 2 * i + j
                cw[k] = self.p_com[i] + Ri @ self.box_c_local[j]
                Rw[k] = Ri
        cw[2 * nb:] = self.stat_c
        Rw[2 * nb:] = np.eye(3)
        ext = np.einsum("kij,kj->ki", np.abs(Rw), self.h_arr)
        return cw, Rw, cw - ext - MARGIN, cw + ext + MARGIN

    def _drive(self, t):
        return DRIVE_FORCE * min(1.0, max(0.0, t / DRIVE_RAMP))

    def _stiffness(self, t):
        s = min(1.0, t / K_RAMP_TIME)
        s = s * s * (3.0 - 2.0 * s)  # smoothstep
        return K_MAX * max(s, 1.0e-3)

    # ------------------------------------------------------------ contacts
    def _contacts(self, k, collect=False, wide=0.0):
        """Evaluate penalty contacts. Returns (forces, torques, pairs_info)."""
        nb = self.nb
        F = np.zeros((nb, 3))
        T = np.zeros((nb, 3))
        cw, Rw, amin, amax = self._box_world()
        if wide > 0.0:
            amin = amin - wide
            amax = amax + wide
        info = {} if collect else None
        mu = self.mu
        for (a, b) in self.pairs:
            if not (np.all(amin[a] <= amax[b]) and np.all(amin[b] <= amax[a])):
                continue
            pair_f = np.zeros(3)
            pair_sd = np.inf
            if collect:
                primary = min(o for o in (self.box_owner[a], self.box_owner[b])
                              if o >= 0)
            for (A, B) in ((a, b), (b, a)):
                oa, ob = self.box_owner[A], self.box_owner[B]
                dynA = oa >= 0
                if not dynA:
                    # feature sources are always dynamic boxes; static boxes are
                    # only probed as targets (their features are far away).
                    continue
                RB = Rw[B]
                cB = cw[B]
                hB = self.h_arr[B]
                pts = self.feat_arr[A]
                pw = self.p_com[oa] + pts @ self.R[oa].T
                inv_mA = self.inv_m[oa]
                local = (pw - cB) @ RB          # (P,3) in B frame
                q = np.abs(local) - hB          # >0 outside per axis
                inside = np.all(q < 0.0, axis=1)
                idx = np.where(np.all(q < MARGIN, axis=1))[0] if collect \
                    else np.where(inside)[0]
                if idx.size == 0:
                    continue
                if collect:
                    # signed gap: negative inside (= -penetration on the
                    # min-overlap axis), positive L-inf separation outside
                    pair_sd = min(pair_sd, float(q[idx].max(axis=1).min()))
                for m_ in idx:
                    if not inside[m_]:
                        continue
                    ax = int(np.argmax(q[m_]))   # min-penetration axis
                    sgn = np.sign(local[m_, ax]) or 1.0
                    n = RB[:, ax] * sgn          # outward from B, toward A point
                    depth = -q[m_, ax]           # >0 penetration
                    if depth > self._max_pen:
                        self._max_pen = depth
                    pc = pw[m_] - 0.5 * depth * n
                    if dynA:
                        rA_c = pc - self.p_com[oa]
                        vAc = self.v[oa] + np.cross(self.w[oa], rA_c)
                    else:
                        rA_c = None
                        vAc = np.zeros(3)
                    if ob >= 0:
                        rB_c = pc - self.p_com[ob]
                        vBc = self.v[ob] + np.cross(self.w[ob], rB_c)
                        inv_mB = self.inv_m[ob]
                        IBinv = self.R[ob] @ self.I_inv_body @ self.R[ob].T
                    else:
                        rB_c = None
                        vBc = np.zeros(3)
                        inv_mB = 0.0
                        IBinv = None
                    vrel = vAc - vBc
                    vn = float(vrel @ n)
                    m_eff = 1.0 / (inv_mA + inv_mB) if dynA else np.inf
                    if not dynA:
                        m_eff = 1.0 / inv_mB if inv_mB > 0 else 1.0
                    c_d = 2.0 * ZETA * np.sqrt(k * m_eff)
                    fn = k * depth - c_d * vn
                    if fn <= 0.0:
                        continue
                    if fn > F_CAP:
                        fn = F_CAP
                    vt = vrel - vn * n
                    speed = float(np.linalg.norm(vt))
                    ff = -mu * fn * vt / (speed + FRIC_EPS)
                    ft_max = mu * fn
                    nrm = float(np.linalg.norm(ff))
                    if nrm > ft_max:
                        ff *= ft_max / nrm
                    fA = fn * n + ff             # force on body owning A
                    if dynA:
                        F[oa] += fA
                        T[oa] += np.cross(rA_c, fA)
                    if ob >= 0:
                        F[ob] -= fA
                        T[ob] -= np.cross(rB_c, fA)
                    if collect:
                        pair_f = pair_f + (fA if oa == primary else -fA)
            if collect and (np.linalg.norm(pair_f) > 1e-9 or pair_sd < self.length_scale):
                info[(a, b)] = (pair_f, pair_sd)
        return F, T, info

    # ------------------------------------------------------------ stepping
    def _substep(self):
        t = self.t
        k = self._stiffness(t)
        F, T, _ = self._contacts(k)
        fd = self._drive(t)
        Ftot = F + self.mass[:, None] * self.gravity[None, :]
        Ftot[DRIVE_BODY, 0] += fd
        self.v += Ftot * self.inv_m[:, None] * DT
        for i in range(self.nb):
            Iw_inv = self.R[i] @ self.I_inv_body @ self.R[i].T
            self.w[i] += Iw_inv @ T[i] * DT
        self.v *= 1.0 / (1.0 + LIN_DAMP * DT)
        self.w *= 1.0 / (1.0 + ANG_DAMP * DT)
        np.clip(self.v, -V_MAX, V_MAX, out=self.v)
        wnorm = np.linalg.norm(self.w, axis=1, keepdims=True)
        over = wnorm > W_MAX
        if np.any(over):
            self.w[over[:, 0]] *= (W_MAX / wnorm[over])
        self.p_com += self.v * DT
        for i in range(self.nb):
            wx, wy, wz = self.w[i]
            dq = 0.5 * DT * np.array([
                wx * self.q[i, 3] + wy * self.q[i, 2] - wz * self.q[i, 1],
                -wx * self.q[i, 2] + wy * self.q[i, 3] + wz * self.q[i, 0],
                wx * self.q[i, 1] - wy * self.q[i, 0] + wz * self.q[i, 3],
                -wx * self.q[i, 0] - wy * self.q[i, 1] - wz * self.q[i, 2],
            ])
            self.q[i] += dq
            self.q[i] /= np.linalg.norm(self.q[i])
        self._refresh()
        sp = float(np.abs(self.v).max())
        if sp > self._max_speed:
            self._max_speed = sp
        self.step += 1
        self.t = self.step * DT

    # ------------------------------------------------------------ sampling
    def _frame(self, time: float) -> StateFrame:
        # origin (mesh-frame) pose from COM state
        Rw_c = np.einsum("nij,j->ni", self.R, self.c_com)
        pos = self.p_com - Rw_c
        vel = self.v - np.cross(self.w, Rw_c)
        fd = self._drive(time)
        ext = np.tile(self.mass[:, None] * self.gravity[None, :], (1, 1))
        ext[DRIVE_BODY, 0] += fd
        k = self._stiffness(time)
        _, _, info = self._contacts(k, collect=True, wide=0.02)
        # pair aggregation to body-level interaction forces
        body_pair_f = {}
        sds = []
        for (a, b), (f, sd) in info.items():
            oa, ob = self.box_owner[a], self.box_owner[b]
            if oa >= 0 and ob >= 0:
                ia, ib = (oa, ob) if oa <= ob else (ob, oa)
            elif oa >= 0:
                ia, ib = oa, ob
            else:
                ia, ib = ob, oa
            # f is the contact force acting on body ia from partner ib
            key = (ia, ib)
            body_pair_f[key] = body_pair_f.get(key, np.zeros(3)) + f
            if np.isfinite(sd) and sd < self.length_scale:
                sds.append(sd)
        if body_pair_f:
            inter = np.stack(list(body_pair_f.values()))
        else:
            inter = np.zeros((0, 3))
        if sds:
            csd = np.asarray(sorted(sds), dtype=float)
        else:
            csd = np.array([0.5 * self.length_scale])
        frame = StateFrame(
            float(time),
            pos,
            vel,
            orientation=self.q.copy(),
            angular_velocity=self.w.copy(),
            mass=self.mass.copy(),
            body_id=np.arange(self.nb, dtype=np.int32),
            attributes={"release_time": np.zeros(self.nb)},
            fields={
                "external_force": ext,
                "interaction_force": inter,
                "contact_signed_distance": csd,
            },
            scalars={"contact_length_scale": float(self.length_scale)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------ API
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(time / DT))
        guard = 0
        while self.step < target:
            self._substep()
            guard += 1
            if guard > 4_000_000:
                break
        self.time = time
        return self._frame(time)

    def diagnostics(self) -> dict:
        return {
            "solver": "penalty_rigid_body_L_link_chain",
            "substeps": int(self.step),
            "max_penetration_m": float(self._max_pen),
            "max_speed_m_s": float(self._max_speed),
            "drive_force_N": DRIVE_FORCE,
        }
