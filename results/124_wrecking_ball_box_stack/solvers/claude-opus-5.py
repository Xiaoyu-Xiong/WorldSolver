"""Rigid-body solver for scene 124_wrecking_ball_box_stack.

Self-contained 3D rigid-body dynamics:

* semi-implicit (symplectic) Euler time stepping at a fixed small dt
* signed-distance contact generation (box vertex / face, sphere-box,
  capsule-box, half-space floor) with a speculative margin
* velocity-level contact resolution by projected Gauss-Seidel over a
  graph colouring of the contact graph, with warm starting, Baumgarte
  penetration bias, Newton restitution and a Coulomb friction cone
* the suspending chain is modelled as exact 3-DoF ball-and-socket joints
  solved as one dense saddle-point block each iteration, so the links stay
  connected regardless of the 8 kg : 0.18 kg mass ratio

No animation tables: every state comes from integrating the equations of
motion under gravity plus contact and joint constraint forces.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_I3 = np.eye(3)


# ----------------------------------------------------------------- utilities
def _quats_to_mats(q: np.ndarray) -> np.ndarray:
    """Batch xyzw quaternion -> rotation matrix (local -> world)."""
    q = np.asarray(q, dtype=float)
    nrm = np.linalg.norm(q, axis=1)
    nrm = np.where(nrm > 1e-15, nrm, 1.0)
    q = q / nrm[:, None]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.empty((q.shape[0], 3, 3), dtype=float)
    m[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    m[:, 0, 1] = 2.0 * (xy - wz)
    m[:, 0, 2] = 2.0 * (xz + wy)
    m[:, 1, 0] = 2.0 * (xy + wz)
    m[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    m[:, 1, 2] = 2.0 * (yz - wx)
    m[:, 2, 0] = 2.0 * (xz - wy)
    m[:, 2, 1] = 2.0 * (yz + wx)
    m[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return m


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Batch xyzw quaternion product a * b."""
    av, aw = a[:, :3], a[:, 3:4]
    bv, bw = b[:, :3], b[:, 3:4]
    v = aw * bv + bw * av + np.cross(av, bv)
    w = aw * bw - np.sum(av * bv, axis=1, keepdims=True)
    return np.concatenate([v, w], axis=1)


def _integrate_quat(q: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
    """Exponential-map orientation update for world-frame angular velocity."""
    ang = np.linalg.norm(w, axis=1) * dt
    axis = np.zeros_like(w)
    big = ang > 1e-12
    if np.any(big):
        axis[big] = w[big] / (np.linalg.norm(w[big], axis=1)[:, None])
    half = 0.5 * ang
    dq = np.concatenate([axis * np.sin(half)[:, None], np.cos(half)[:, None]], axis=1)
    out = _quat_mul(dq, q)
    nrm = np.linalg.norm(out, axis=1)
    nrm = np.where(nrm > 1e-15, nrm, 1.0)
    return out / nrm[:, None]


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def _tangent_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.zeros_like(n)
    use_z = np.abs(n[:, 2]) < 0.9
    helper[use_z, 2] = 1.0
    helper[~use_z, 0] = 1.0
    t1 = np.cross(n, helper)
    ln = np.linalg.norm(t1, axis=1)
    ln = np.where(ln > 1e-12, ln, 1.0)
    t1 = t1 / ln[:, None]
    t2 = np.cross(n, t1)
    return t1, t2


# ---------------------------------------------------------------- containers
class _Contacts:
    """Flat contact arrays for one time step."""

    __slots__ = ("ia", "ib", "n", "cp", "sep", "cid")

    def __init__(self, ia, ib, n, cp, sep, cid):
        self.ia = ia
        self.ib = ib
        self.n = n
        self.cp = cp
        self.sep = sep
        self.cid = cid


# ------------------------------------------------------------------- solver
class BenchmarkSolver:
    # frozen geometry (scene_geometry.yaml)
    BOX_H = np.array([0.17, 0.38, 0.17])
    BALL_R = 0.42
    CAP_R = 0.07
    CAP_HALF = 0.125          # half of the capsule cylinder height
    FLOOR_TOP = 0.0           # floor box: half_extents z=0.05 at z=-0.05
    CHAIN_PITCH_HALF = 0.16   # half of the 0.32 m link spacing -> joint anchors
    BALL_ANCHOR_Z = 0.44      # ball local attachment point on the shaft

    # numerics
    DT = 1.0 / 500.0
    N_ITER = 15
    MARGIN = 0.03             # speculative contact band
    SLOP = 0.0015
    BETA = 0.2
    MAX_PUSH = 2.0
    REST_VEL_THRESH = 0.6
    JOINT_BETA = 0.3
    LIN_DAMP = 0.04
    ANG_DAMP = 0.10
    MAX_VERTS_PER_PAIR = 4
    SWING_SPEED = 2.5         # ball speed at the low point of the arc, m/s

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.cfg = cfg
        rng = np.random.default_rng(int(seed))
        self._rng = rng

        frame0 = scene.initial_state()
        self.n = int(frame0.position.shape[0])
        n = self.n
        self.S = n  # index of the immovable pseudo-body (floor / world)

        self.p = np.array(frame0.position, dtype=float)
        self.q = np.array(frame0.orientation, dtype=float) if frame0.orientation is not None \
            else np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
        self.v = np.zeros((n, 3), dtype=float)
        self.w = np.zeros((n, 3), dtype=float)
        self.body_id = np.arange(n, dtype=np.int32)
        self.mass_dyn = np.array(frame0.mass, dtype=float)
        self.release_time = np.asarray(
            frame0.attributes.get("release_time", np.zeros(n)), dtype=float)

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        contact_cfg = cfg.get("contact", {})
        self.mu = float(contact_cfg.get("friction", 0.5))
        self.restitution = float(contact_cfg.get("restitution", 0.1))
        self.length_scale = float(contact_cfg.get("contact_length_scale",
                                                  contact_cfg.get("length_scale", 0.02)))
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.duration = float(cfg.get("duration", 10.0))

        # ---- classify bodies from the frozen config ids
        ids = [str(b["id"]) for b in cfg["bodies"]]
        self.ids = ids
        self.kind = np.zeros(n, dtype=np.int8)  # 0 box, 1 sphere(ball), 2 capsule
        self.box_idx, self.cap_idx = [], []
        self.ball_idx = -1
        for i, name in enumerate(ids):
            if name.startswith("box"):
                self.kind[i] = 0
                self.box_idx.append(i)
            elif name.startswith("chain"):
                self.kind[i] = 2
                self.cap_idx.append(i)
            else:
                self.kind[i] = 1
                self.ball_idx = i
        self.box_idx = np.asarray(self.box_idx, dtype=np.int64)
        self.cap_idx = np.asarray(self.cap_idx, dtype=np.int64)
        # chain links ordered bottom-up (chain_0 nearest the ball)
        order = np.argsort([float(self.p[i, 2]) for i in self.cap_idx])
        self.chain_order = self.cap_idx[order]

        # ---- mass / inertia
        self.invm = np.zeros(n + 1, dtype=float)
        self.mass = np.zeros(n + 1, dtype=float)
        self.mass[:n] = self.mass_dyn
        self.invm[:n] = 1.0 / np.maximum(self.mass_dyn, 1e-12)
        self.Ibody = np.zeros((n + 1, 3, 3), dtype=float)
        self.Ibody_inv = np.zeros((n + 1, 3, 3), dtype=float)
        h = self.BOX_H
        for i in range(n):
            m = float(self.mass_dyn[i])
            if self.kind[i] == 0:
                d = m / 3.0 * np.array([h[1] ** 2 + h[2] ** 2,
                                        h[0] ** 2 + h[2] ** 2,
                                        h[0] ** 2 + h[1] ** 2])
            elif self.kind[i] == 1:
                s = 0.4 * m * self.BALL_R ** 2
                d = np.array([s, s, s])
            else:
                L = 2.0 * self.CAP_HALF + 2.0 * self.CAP_R
                r = self.CAP_R
                it = m * (3.0 * r * r + L * L) / 12.0
                ia = 0.5 * m * r * r
                d = np.array([it, it, ia])
            self.Ibody[i] = np.diag(d)
            self.Ibody_inv[i] = np.diag(1.0 / np.maximum(d, 1e-15))

        # bounding radii for broad phase
        self.brad = np.zeros(n, dtype=float)
        self.brad[self.kind == 0] = float(np.linalg.norm(h))
        if self.ball_idx >= 0:
            self.brad[self.ball_idx] = self.BALL_R
        self.brad[self.kind == 2] = self.CAP_HALF + self.CAP_R

        # box local vertices
        sg = np.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                       for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)], dtype=float)
        self.vloc = sg * h[None, :]

        # ---- joints: ball <-> chain_0 <-> ... <-> chain_5 <-> world anchor
        ch = self.chain_order
        self.joint_bodies = [self.ball_idx] + [int(c) for c in ch]
        joints = []
        a_up = np.array([0.0, 0.0, self.CHAIN_PITCH_HALF])
        a_dn = np.array([0.0, 0.0, -self.CHAIN_PITCH_HALF])
        joints.append((0, 1, np.array([0.0, 0.0, self.BALL_ANCHOR_Z]), a_dn))
        for k in range(len(ch) - 1):
            joints.append((1 + k, 2 + k, a_up.copy(), a_dn.copy()))
        top = int(ch[-1])
        self.anchor_world = self.p[top] + a_up
        joints.append((len(ch), -1, a_up.copy(), self.anchor_world.copy()))
        self.joints = joints
        self.n_joint = len(joints)
        self.jb = np.asarray(self.joint_bodies, dtype=np.int64)
        self.n_jb = len(self.jb)

        # ---- collision filtering: jointed / chain-internal pairs are skipped
        self.allow = np.ones((n, n), dtype=bool)
        np.fill_diagonal(self.allow, False)
        chain_and_ball = [int(self.ball_idx)] + [int(c) for c in ch]
        for a in chain_and_ball:
            for b in chain_and_ball:
                self.allow[a, b] = False

        # ---- initial condition: the suspended assembly is on its pendulum arc.
        # Rigid rotation about the fixed overhead pivot (y axis, scene plane x-z)
        # gives the ball the speed of a body released from a displaced,
        # taut-chain configuration.  It swings out, returns and strikes the stack.
        lever = float(self.anchor_world[2] - self.p[self.ball_idx, 2])
        omega_y = self.SWING_SPEED / max(lever, 1e-6)
        omega = np.array([0.0, omega_y, 0.0])
        for b in chain_and_ball:
            r = self.p[b] - self.anchor_world
            self.v[b] = np.cross(omega, r)
            self.w[b] = omega

        self.t = 0.0
        self.R = _quats_to_mats(self.q)
        self._warm: dict[int, tuple[float, float, float]] = {}
        self._last_contact_force = np.zeros((0, 3), dtype=float)
        self._last_joint_force = np.zeros((self.n_joint, 3), dtype=float)
        self._n_steps = 0
        self._max_pen = 0.0

    # -------------------------------------------------------- contact search
    def _box_world_verts(self) -> np.ndarray:
        bi = self.box_idx
        return self.p[bi][:, None, :] + np.einsum('bij,kj->bki', self.R[bi], self.vloc)

    def _sphere_vs_boxes(self, centers, radii, ia, jb, cid):
        """Contacts between spheres (body ia, world centre) and boxes jb."""
        if len(ia) == 0:
            return None
        RB = self.R[jb]
        rel = centers - self.p[jb]
        qq = np.einsum('pab,pa->pb', RB, rel)
        h = self.BOX_H
        cl = np.clip(qq, -h, h)
        d = qq - cl
        dist = np.linalg.norm(d, axis=1)
        outside = dist > 1e-9
        nl = np.zeros_like(d)
        nl[outside] = d[outside] / dist[outside, None]
        sep = dist - radii
        inside = ~outside
        if np.any(inside):
            qi = qq[inside]
            pen = h[None, :] - np.abs(qi)
            k = np.argmin(pen, axis=1)
            rows = np.arange(k.size)
            sgn = np.sign(qi[rows, k])
            sgn[sgn == 0.0] = 1.0
            tmp = np.zeros((k.size, 3))
            tmp[rows, k] = sgn
            nl[inside] = tmp
            sep[inside] = -pen[rows, k] - radii[inside]
        keep = sep < self.MARGIN
        if not np.any(keep):
            return None
        nw = np.einsum('pab,pb->pa', RB[keep], nl[keep])
        cpw = self.p[jb[keep]] + np.einsum('pab,pb->pa', RB[keep], cl[keep])
        return _Contacts(ia[keep], jb[keep], nw, cpw, sep[keep], cid[keep])

    def _gen_contacts(self) -> _Contacts:
        p, R = self.p, self.R
        S = self.S
        margin = self.MARGIN
        parts: list[_Contacts] = []
        bi_all = self.box_idx
        nb = bi_all.size
        Vw = self._box_world_verts()

        # --- box vertices vs floor half-space (z >= 0)
        z = Vw[:, :, 2]
        m = z < margin
        if np.any(m):
            r_, c_ = np.nonzero(m)
            cp = Vw[r_, c_].copy()
            cp[:, 2] = self.FLOOR_TOP
            ia = bi_all[r_]
            nrm = np.zeros((ia.size, 3)); nrm[:, 2] = 1.0
            cid = 1_000_000 + ia.astype(np.int64) * 8 + c_
            parts.append(_Contacts(ia, np.full(ia.size, S, dtype=np.int64),
                                   nrm, cp, z[r_, c_] - self.FLOOR_TOP, cid))

        # --- ball vs floor
        b = self.ball_idx
        if b >= 0:
            sep = float(p[b, 2] - self.BALL_R - self.FLOOR_TOP)
            if sep < margin:
                cp = np.array([[p[b, 0], p[b, 1], self.FLOOR_TOP]])
                parts.append(_Contacts(np.array([b], dtype=np.int64),
                                       np.array([S], dtype=np.int64),
                                       np.array([[0.0, 0.0, 1.0]]), cp,
                                       np.array([sep]),
                                       np.array([1_900_000 + b], dtype=np.int64)))

        # --- capsule end caps vs floor
        if self.cap_idx.size:
            ci = self.cap_idx
            axis = self.R[ci][:, :, 2] * self.CAP_HALF
            ends = np.concatenate([p[ci] + axis, p[ci] - axis], axis=0)
            owner = np.concatenate([ci, ci])
            tag = np.concatenate([np.zeros(ci.size, dtype=np.int64),
                                  np.ones(ci.size, dtype=np.int64)])
            sep = ends[:, 2] - self.CAP_R - self.FLOOR_TOP
            keep = sep < margin
            if np.any(keep):
                cp = ends[keep].copy()
                cp[:, 2] = self.FLOOR_TOP
                nrm = np.zeros((int(keep.sum()), 3)); nrm[:, 2] = 1.0
                cid = 1_950_000 + owner[keep] * 4 + tag[keep]
                parts.append(_Contacts(owner[keep],
                                       np.full(int(keep.sum()), S, dtype=np.int64),
                                       nrm, cp, sep[keep], cid))

        # --- broad phase between distinct bodies (bounding spheres)
        d = p[:, None, :] - p[None, :, :]
        dist2 = np.sum(d * d, axis=2)
        thr = self.brad[:, None] + self.brad[None, :] + margin
        cand = (dist2 < thr * thr) & self.allow
        cand = np.triu(cand, 1)
        ai, bj = np.nonzero(cand)

        if ai.size:
            ka, kb = self.kind[ai], self.kind[bj]
            # box-box: both vertex orderings
            bb = (ka == 0) & (kb == 0)
            if np.any(bb):
                A = np.concatenate([ai[bb], bj[bb]])
                B = np.concatenate([bj[bb], ai[bb]])
                c = self._box_box(A, B, Vw)
                if c is not None:
                    parts.append(c)
            # sphere(ball) vs box
            sb = ((ka == 1) & (kb == 0)) | ((ka == 0) & (kb == 1))
            if np.any(sb):
                sa = np.where(ka[sb] == 1, ai[sb], bj[sb])
                sbx = np.where(ka[sb] == 1, bj[sb], ai[sb])
                cid = 3_000_000 + sa * 64 + sbx
                c = self._sphere_vs_boxes(p[sa], np.full(sa.size, self.BALL_R),
                                          sa, sbx, cid)
                if c is not None:
                    parts.append(c)
            # capsule vs box: three sample spheres along the segment
            cb = ((ka == 2) & (kb == 0)) | ((ka == 0) & (kb == 2))
            if np.any(cb):
                ca = np.where(ka[cb] == 2, ai[cb], bj[cb])
                cbx = np.where(ka[cb] == 2, bj[cb], ai[cb])
                axis = self.R[ca][:, :, 2] * self.CAP_HALF
                for s_i, s in enumerate((-1.0, 0.0, 1.0)):
                    centers = p[ca] + s * axis
                    cid = 4_000_000 + (ca * 64 + cbx) * 4 + s_i
                    c = self._sphere_vs_boxes(centers,
                                              np.full(ca.size, self.CAP_R),
                                              ca, cbx, cid)
                    if c is not None:
                        parts.append(c)

        if not parts:
            e0 = np.zeros(0, dtype=np.int64)
            return _Contacts(e0, e0, np.zeros((0, 3)), np.zeros((0, 3)),
                             np.zeros(0), e0)
        return _Contacts(
            np.concatenate([c.ia for c in parts]).astype(np.int64),
            np.concatenate([c.ib for c in parts]).astype(np.int64),
            np.concatenate([c.n for c in parts]),
            np.concatenate([c.cp for c in parts]),
            np.concatenate([c.sep for c in parts]),
            np.concatenate([c.cid for c in parts]).astype(np.int64))

    def _box_box(self, A, B, Vw):
        """Vertices of box A against the faces of box B (signed distance)."""
        pos = np.searchsorted(self.box_idx, A)
        VA = Vw[pos]                                    # (P,8,3)
        RB = self.R[B]                                  # (P,3,3)
        rel = VA - self.p[B][:, None, :]
        qq = np.einsum('pab,pka->pkb', RB, rel)         # local coords in B
        pen = self.BOX_H[None, None, :] - np.abs(qq)
        depth = pen.min(axis=2)
        ax = pen.argmin(axis=2)
        sep_all = -depth
        k = min(self.MAX_VERTS_PER_PAIR, sep_all.shape[1])
        sel = np.argsort(sep_all, axis=1, kind='stable')[:, :k]
        P = sel.shape[0]
        rows = np.repeat(np.arange(P), k)
        cols = sel.ravel()
        sep = sep_all[rows, cols]
        keep = sep < self.MARGIN
        if not np.any(keep):
            return None
        rows, cols, sep = rows[keep], cols[keep], sep[keep]
        axk = ax[rows, cols]
        sgn = np.sign(qq[rows, cols, axk])
        sgn[sgn == 0.0] = 1.0
        nrm = RB[rows, :, axk] * sgn[:, None]
        cp = VA[rows, cols]
        ia = A[rows]
        ib = B[rows]
        cid = 2_000_000 + (ia * 8 + cols) * 64 + ib
        return _Contacts(ia, ib, nrm, cp, sep, cid.astype(np.int64))

    # ------------------------------------------------------------- colouring
    def _colour(self, ia, ib):
        S = self.S
        used: dict[int, int] = {}
        cols = np.empty(ia.size, dtype=np.int64)
        al = ia.tolist()
        bl = ib.tolist()
        for k in range(len(al)):
            a = al[k]; b = bl[k]
            m = used.get(a, 0)
            if b != S:
                m |= used.get(b, 0)
            c = 0
            while (m >> c) & 1:
                c += 1
            cols[k] = c
            used[a] = used.get(a, 0) | (1 << c)
            if b != S:
                used[b] = used.get(b, 0) | (1 << c)
        order = np.argsort(cols, kind='stable')
        cs = cols[order]
        bounds = np.searchsorted(cs, np.arange(cs[-1] + 2)) if cs.size else [0]
        groups = []
        for c in range(int(cs[-1]) + 1) if cs.size else []:
            g = order[bounds[c]:bounds[c + 1]]
            if g.size:
                groups.append(g)
        return groups

    # ------------------------------------------------------------ joint block
    def _build_joint_block(self, dt):
        nb = self.n_jb
        nd = 6 * nb
        nr = 3 * self.n_joint
        J = np.zeros((nr, nd))
        C = np.zeros(nr)
        for k, (la, lb, aa, ab) in enumerate(self.joints):
            ga = self.jb[la]
            ra = self.R[ga] @ aa
            pa = self.p[ga] + ra
            r0 = 3 * k
            J[r0:r0 + 3, 6 * la:6 * la + 3] = _I3
            J[r0:r0 + 3, 6 * la + 3:6 * la + 6] = -_skew(ra)
            if lb >= 0:
                gb = self.jb[lb]
                rb = self.R[gb] @ ab
                pb = self.p[gb] + rb
                J[r0:r0 + 3, 6 * lb:6 * lb + 3] = -_I3
                J[r0:r0 + 3, 6 * lb + 3:6 * lb + 6] = _skew(rb)
            else:
                pb = ab
            C[r0:r0 + 3] = pa - pb
        Minv = np.zeros((nd, nd))
        for l, g in enumerate(self.jb):
            Minv[6 * l:6 * l + 3, 6 * l:6 * l + 3] = self.invm[g] * _I3
            Minv[6 * l + 3:6 * l + 6, 6 * l + 3:6 * l + 6] = \
                self.R[g] @ self.Ibody_inv[g] @ self.R[g].T
        A = J @ Minv @ J.T + 1e-9 * np.eye(nr)
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            Ainv = np.linalg.pinv(A)
        self._J = J
        self._C = C
        self._Minv = Minv
        self._MJt = Minv @ J.T
        self._Ainv = Ainv
        self._jbias = (self.JOINT_BETA / dt) * C

    def _solve_joints(self, lam_acc):
        u = np.empty(6 * self.n_jb)
        for l, g in enumerate(self.jb):
            u[6 * l:6 * l + 3] = self.v[g]
            u[6 * l + 3:6 * l + 6] = self.w[g]
        rhs = -(self._J @ u + self._jbias)
        lam = self._Ainv @ rhs
        if not np.all(np.isfinite(lam)):
            return lam_acc
        du = self._MJt @ lam
        for l, g in enumerate(self.jb):
            self.v[g] += du[6 * l:6 * l + 3]
            self.w[g] += du[6 * l + 3:6 * l + 6]
        return lam_acc + lam

    # ------------------------------------------------------------------ step
    def _step(self, dt):
        n = self.n
        self.R = _quats_to_mats(self.q)
        # world inverse inertia
        Iinv = np.zeros((n + 1, 3, 3))
        Rd = self.R
        Iinv[:n] = np.einsum('bij,bjk,blk->bil', Rd, self.Ibody_inv[:n], Rd)

        # --- unconstrained velocity update (gravity only external force)
        self.v += dt * self.gravity[None, :]

        # --- contacts
        ct = self._gen_contacts()
        C = ct.sep.size
        lam_n = np.zeros(C); lam_t1 = np.zeros(C); lam_t2 = np.zeros(C)
        if C:
            warm = self._warm
            if warm:
                cl = ct.cid.tolist()
                for k in range(C):
                    got = warm.get(cl[k])
                    if got is not None:
                        lam_n[k] = got[0]; lam_t1[k] = got[1]; lam_t2[k] = got[2]

            ia, ib, nrm = ct.ia, ct.ib, ct.n
            ra = ct.cp - self.p[np.minimum(ia, n - 1)]
            rb = np.where((ib < n)[:, None], ct.cp - self.p[np.minimum(ib, n - 1)], 0.0)
            t1, t2 = _tangent_basis(nrm)

            Ia = Iinv[ia]; Ib = Iinv[ib]
            invma = self.invm[ia]; invmb = self.invm[ib]

            def keff(dirv):
                ua = np.cross(ra, dirv)
                ub = np.cross(rb, dirv)
                k = invma + invmb
                k = k + np.einsum('pi,pij,pj->p', ua, Ia, ua)
                k = k + np.einsum('pi,pij,pj->p', ub, Ib, ub)
                return np.maximum(k, 1e-10)

            kn = keff(nrm); kt1 = keff(t1); kt2 = keff(t2)

            # pre-solve normal approach speed for Newton restitution
            ibc = np.minimum(ib, n - 1)
            dynb = (ib < n)[:, None]
            vb0 = np.where(dynb, self.v[ibc] + np.cross(self.w[ibc], rb), 0.0)
            vrel0 = (self.v[ia] + np.cross(self.w[ia], ra)) - vb0
            vn0 = np.einsum('pi,pi->p', vrel0, nrm)
            pen = -ct.sep
            bias = np.where(
                pen > self.SLOP,
                np.minimum(self.BETA * (pen - self.SLOP) / dt, self.MAX_PUSH),
                np.where(pen < 0.0, pen / dt, 0.0))
            rest = np.where(vn0 < -self.REST_VEL_THRESH, -self.restitution * vn0, 0.0)
            vtarget = bias + rest
            self._max_pen = max(self._max_pen, float(np.max(pen)) if C else 0.0)

            # apply warm-start impulses
            if warm and np.any(lam_n != 0.0):
                Pw = lam_n[:, None] * nrm + lam_t1[:, None] * t1 + lam_t2[:, None] * t2
                self._scatter(ia, ib, ra, rb, Pw, Iinv)

            groups = self._colour(ia, ib)
        else:
            groups = []

        # --- iterate: contact PGS interleaved with an exact joint block solve
        self._build_joint_block(dt)
        lam_j = np.zeros(3 * self.n_joint)
        mu = self.mu
        for _ in range(self.N_ITER):
            for g in groups:
                iag = ia[g]; ibg = ib[g]
                rag = ra[g]; rbg = rb[g]
                ng = nrm[g]; t1g = t1[g]; t2g = t2[g]
                stat = (ibg >= n)[:, None]
                vb = np.where(stat, 0.0, self.v[np.minimum(ibg, n - 1)] +
                              np.cross(self.w[np.minimum(ibg, n - 1)], rbg))
                vrel = (self.v[iag] + np.cross(self.w[iag], rag)) - vb
                vn = np.einsum('pi,pi->p', vrel, ng)
                dl = -(vn - vtarget[g]) / kn[g]
                new_n = np.maximum(lam_n[g] + dl, 0.0)
                dn = new_n - lam_n[g]
                lam_n[g] = new_n
                vt1 = np.einsum('pi,pi->p', vrel, t1g)
                vt2 = np.einsum('pi,pi->p', vrel, t2g)
                l1 = lam_t1[g] - vt1 / kt1[g]
                l2 = lam_t2[g] - vt2 / kt2[g]
                mag = np.sqrt(l1 * l1 + l2 * l2)
                lim = mu * new_n
                scale = np.where(mag > lim, lim / np.maximum(mag, 1e-12), 1.0)
                l1 = l1 * scale; l2 = l2 * scale
                d1 = l1 - lam_t1[g]; d2 = l2 - lam_t2[g]
                lam_t1[g] = l1; lam_t2[g] = l2
                P = dn[:, None] * ng + d1[:, None] * t1g + d2[:, None] * t2g
                self._scatter(iag, ibg, rag, rbg, P, Iinv)
            lam_j = self._solve_joints(lam_j)

        # --- integrate
        if not np.all(np.isfinite(self.v)):
            self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(self.w)):
            self.w = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.v, -60.0, 60.0, out=self.v)
        np.clip(self.w, -80.0, 80.0, out=self.w)
        self.v *= (1.0 - self.LIN_DAMP * dt)
        self.w *= (1.0 - self.ANG_DAMP * dt)
        self.p += dt * self.v
        self.q = _integrate_quat(self.q, self.w, dt)
        self.R = _quats_to_mats(self.q)
        self.t += dt
        self._n_steps += 1

        # --- record constraint forces of this step
        if C:
            self._warm = {int(k): (float(a), float(b), float(c))
                          for k, a, b, c in zip(ct.cid.tolist(), lam_n, lam_t1, lam_t2)}
            self._last_contact_force = (lam_n[:, None] * nrm +
                                        lam_t1[:, None] * t1 +
                                        lam_t2[:, None] * t2) / dt
        else:
            self._warm = {}
            self._last_contact_force = np.zeros((0, 3))
        self._last_joint_force = lam_j.reshape(self.n_joint, 3) / dt

    def _scatter(self, ia, ib, ra, rb, P, Iinv):
        n = self.n
        self.v[ia] += self.invm[ia][:, None] * P
        self.w[ia] += np.einsum('pij,pj->pi', Iinv[ia], np.cross(ra, P))
        act = ib < n
        if np.any(act):
            ibv = ib[act]
            self.v[ibv] -= self.invm[ibv][:, None] * P[act]
            self.w[ibv] -= np.einsum('pij,pj->pi', Iinv[ibv],
                                     np.cross(rb[act], P[act]))

    # -------------------------------------------------------------- sampling
    def advance_to(self, time: float):
        target = float(time)
        dt = self.DT
        guard = 0
        max_steps = int(round(self.duration / dt)) + 64
        while self.t < target - 1e-9 and guard < max_steps:
            h = min(dt, target - self.t)
            if h < 1e-12:
                break
            self._step(h)
            guard += 1

        n = self.n
        ct = self._gen_contacts()
        sd = np.asarray(ct.sep, dtype=float)
        if sd.size == 0:
            sd = np.zeros(1, dtype=float)
        sd = np.nan_to_num(sd, nan=0.0, posinf=0.0, neginf=0.0)

        ext = self.mass_dyn[:, None] * self.gravity[None, :]
        inter = np.concatenate([
            np.nan_to_num(self._last_joint_force, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(self._last_contact_force, nan=0.0, posinf=0.0, neginf=0.0),
        ], axis=0)
        if inter.shape[0] == 0:
            inter = np.zeros((1, 3), dtype=float)

        pos = np.nan_to_num(self.p, nan=0.0, posinf=0.0, neginf=0.0)
        vel = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        ang = np.nan_to_num(self.w, nan=0.0, posinf=0.0, neginf=0.0)
        quat = np.nan_to_num(self.q, nan=0.0, posinf=0.0, neginf=0.0)
        bad = np.linalg.norm(quat, axis=1) < 1e-8
        if np.any(bad):
            quat[bad] = np.array([0.0, 0.0, 0.0, 1.0])
        quat = quat / np.linalg.norm(quat, axis=1)[:, None]

        frame = StateFrame(
            time=max(0.0, target),
            position=pos.copy(),
            velocity=vel.copy(),
            orientation=quat.copy(),
            angular_velocity=ang.copy(),
            mass=self.mass_dyn.copy(),
            body_id=self.body_id.copy(),
            attributes={"release_time": self.release_time.copy()},
            fields={
                "contact_signed_distance": sd,
                "external_force": ext,
                "interaction_force": inter,
            },
            scalars={"contact_length_scale": float(self.length_scale)},
        )
        frame.validate()
        return frame

    # ----------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        return {
            "integrator": "semi-implicit Euler + projected Gauss-Seidel contacts",
            "dt": float(self.DT),
            "solver_iterations": int(self.N_ITER),
            "steps": int(self._n_steps),
            "n_bodies": int(self.n),
            "n_joints": int(self.n_joint),
            "friction": float(self.mu),
            "restitution": float(self.restitution),
            "contact_margin": float(self.MARGIN),
            "max_penetration": float(self._max_pen),
            "sim_time": float(self.t),
        }
