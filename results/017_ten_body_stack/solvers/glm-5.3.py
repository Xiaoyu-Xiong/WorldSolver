"""Impulse-based rigid-body solver for scene 017_ten_body_stack.

Physics: ten dynamic boxes settle under gravity onto a static ground box.
Collision detection uses a separating-axis test (SAT) between oriented
boxes with vertex-in-box contact manifolds.  Contacts are resolved with a
projected-impulse (accumulated, relaxed Jacobi / sequential-impulse style)
velocity solver featuring Coulomb friction, Baumgarte stabilization and a
position-level projection pass.  Time integration is semi-implicit
(symplectic) Euler with substeps far below the sample interval, giving
deterministic, finite states at every requested sample time.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

DT = 5.0e-4              # physics substep (s); 40 substeps per 0.02 s sample
DENSITY = 1000.0         # kg / m^3
MU = 0.7                 # Coulomb friction coefficient
ITERATIONS = 50          # Gauss-Seidel impulse sweeps per substep
PSEUDO_ITERATIONS = 12   # split-impulse (pseudo velocity) sweeps
PSEUDO_BETA = 0.4        # split-impulse de-penetration factor
PSEUDO_CAP = 4.0         # cap on pseudo de-penetration velocity (m/s)
SLOP = 1.0e-5            # penetration slop kept so resting contacts persist
DAMPING = 0.05           # linear/angular velocity damping (1/s)
MARGIN = 4.0e-3          # broadphase margin (m)
NEAR = 5.0e-2            # near-contact reporting threshold (m)
CONTACT_LENGTH_SCALE = 0.2  # frozen contact normalization length (box height, m)


# ----------------------------------------------------------------------------
# quaternion helpers (xyzw convention)
# ----------------------------------------------------------------------------
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array((
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ))


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array((
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    ))


def _box_vertices(c: np.ndarray, R: np.ndarray, e: np.ndarray) -> np.ndarray:
    signs = np.array([[sx, sy, sz] for sx in (-1.0, 1.0)
                      for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)])
    return c + (signs * e) @ R.T


def _sat(cA: np.ndarray, RA: np.ndarray, eA: np.ndarray,
         cB: np.ndarray, RB: np.ndarray, eB: np.ndarray):
    """Separating-axis test between two OBBs.

    Returns (sep, n, axis_type).  If the boxes overlap, sep is the signed
    distance along the minimum-translation axis (sep <= 0; negative means
    penetration with depth = -sep) and n is the unit normal pointing from
    A to B.  If they are separated, sep is the smallest positive axis
    separation (the face-to-face gap for face contacts) with n pointing
    from A to B along that axis.  axis_type is 0 for an A face axis, 1 for
    a B face axis and 2 for an edge cross axis.
    """
    d = cB - cA
    axes = [(RA[:, i].copy(), 0) for i in range(3)]
    axes += [(RB[:, j].copy(), 1) for j in range(3)]
    for i in range(3):
        for j in range(3):
            axes.append((np.cross(RA[:, i], RB[:, j]), 2))

    best_pen = -np.inf       # maximum axis separation (overlap case)
    pen_n = np.zeros(3)
    pen_typ = 0
    best_gap = np.inf        # minimum positive separation (gap case)
    gap_n = np.zeros(3)
    gap_typ = 0

    for L, typ in axes:
        ln = float(np.sqrt(L @ L))
        if ln < 1.0e-9:
            continue
        Ln = L / ln
        ra = float(np.abs(RA.T @ Ln) @ eA)
        rb = float(np.abs(RB.T @ Ln) @ eB)
        sep = abs(float(d @ Ln)) - (ra + rb)
        s = 1.0 if float(d @ Ln) >= 0.0 else -1.0
        if sep > best_pen:
            best_pen = sep
            pen_n = Ln * s
            pen_typ = typ
        if 0.0 < sep < best_gap:
            best_gap = sep
            gap_n = Ln * s
            gap_typ = typ

    if best_pen > 0.0:  # a separating axis exists: report the closest gap
        return best_gap, gap_n, gap_typ
    return best_pen, pen_n, pen_typ


def _exit_depth(cR: np.ndarray, RR: np.ndarray, e: np.ndarray,
                p_world: np.ndarray, dir_world: np.ndarray) -> float:
    """Distance from an interior point to the box surface along dir_world."""
    p = RR.T @ (p_world - cR)
    dv = RR.T @ dir_world
    t = 1.0
    for i in range(3):
        di = float(dv[i])
        if di > 1.0e-12:
            ti = (float(e[i]) - float(p[i])) / di
        elif di < -1.0e-12:
            ti = (-float(e[i]) - float(p[i])) / di
        else:
            continue
        if ti < t:
            t = ti
    return t


def _manifold(cA, RA, eA, cB, RB, eB, n, axis_type, sep):
    """Contact points for a penetrating box pair.

    Vertices of one box inside the other become contact points; the
    penetration depth is the exit distance along the SAT normal.  The
    incident vertex set is chosen from the SAT axis type, with a fallback
    to a midpoint contact for edge-edge penetrations where no vertex of
    either box is inside the other.
    """
    ptsB = []   # vertices of B inside A
    depB = []
    for v in _box_vertices(cB, RB, eB):
        p = RA.T @ (v - cA)
        if np.all(np.abs(p) <= eA):
            ptsB.append(v)
            depB.append(_exit_depth(cA, RA, eA, v, n))
    ptsA = []   # vertices of A inside B
    depA = []
    for v in _box_vertices(cA, RA, eA):
        p = RB.T @ (v - cB)
        if np.all(np.abs(p) <= eB):
            ptsA.append(v)
            depA.append(_exit_depth(cB, RB, eB, v, -n))

    if axis_type == 0:
        if ptsB:
            return ptsB, depB
        return ptsA, depA
    if axis_type == 1:
        if ptsA:
            return ptsA, depA
        return ptsB, depB
    # edge cross axis: combine both vertex sets
    if ptsA or ptsB:
        return ptsA + ptsB, depA + depB
    if sep < 0.0:
        # edge-edge penetration without interior vertices: single
        # midpoint contact prevents tunneling
        return [0.5 * (cA + cB)], [-sep]
    return [], []


class BenchmarkSolver:
    """Ten-body rigid stack settling under gravity (impulse-based dynamics)."""

    def __init__(self) -> None:
        self._ready = False
        self._step_count = 0
        self._last_pair_force: dict[tuple[int, int], np.ndarray] = {}

    # -- setup ---------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        np.random.seed(int(seed))  # deterministic (no stochastic terms used)
        cfg = getattr(scene, "config", {}) or {}
        self.g = np.array(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        geometry = (cfg.get("scene_geometry") or {}).get("objects", [])
        half_by_id: dict[int, np.ndarray] = {}
        ground_center = np.zeros(3)
        ground_half = np.array([4.0, 4.0, 0.05])
        for obj in geometry:
            geom = obj.get("geometry") or {}
            he = np.array(geom.get("half_extents", [0.45, 0.45, 0.1]), dtype=float)
            if obj.get("motion") == "static":
                tr = (obj.get("transform") or {}).get("translation", [0.0, 0.0, 0.0])
                ground_center = np.array(tr, dtype=float)
                ground_half = he
            else:
                binding = obj.get("state_binding") or {}
                bid = binding.get("value")
                if bid is not None:
                    half_by_id[int(bid)] = he

        init = scene.initial_state()
        pos = np.array(init.position, dtype=float)
        vel = np.array(init.velocity, dtype=float)
        if init.orientation is not None:
            quat = np.array(init.orientation, dtype=float)
        else:
            quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(pos), 1))
        if init.angular_velocity is not None:
            omega = np.array(init.angular_velocity, dtype=float)
        else:
            omega = np.zeros((len(pos), 3))
        body_ids = np.array(init.body_id, dtype=int)

        n = len(pos)
        self.n = n
        self.total = n + 1  # last slot is the static ground box
        self.body_ids = body_ids

        half = np.zeros((self.total, 3))
        mass = np.zeros(n)
        inertia = np.zeros((self.total, 3))
        for i, bid in enumerate(body_ids):
            e = half_by_id.get(int(bid), np.array([0.45, 0.45, 0.1]))
            half[i] = e
            m = 8.0 * float(np.prod(e)) * DENSITY
            mass[i] = m
            fx, fy, fz = 2.0 * e
            inertia[i] = m / 12.0 * np.array([fy * fy + fz * fz,
                                              fx * fx + fz * fz,
                                              fx * fx + fy * fy])
        half[n] = ground_half
        self.half = half
        self.radius = np.linalg.norm(half, axis=1)
        self.mass = mass

        self.x = np.vstack((pos, ground_center[None, :]))
        self.v = np.vstack((vel, np.zeros((1, 3))))
        self.q = np.vstack((quat, np.array([[0.0, 0.0, 0.0, 1.0]])))
        self.w = np.vstack((omega, np.zeros((1, 3))))
        self.invm = np.zeros(self.total)
        self.invm[:n] = 1.0 / mass
        self.ibody = inertia
        self.ibody_inv = np.zeros((self.total, 3))
        for i in range(n):
            self.ibody_inv[i] = 1.0 / np.maximum(inertia[i], 1.0e-12)

        self.R = np.zeros((self.total, 3, 3))
        self.invI = np.zeros((self.total, 3, 3))
        self.Iw = np.zeros((self.total, 3, 3))
        self._update_rot()
        self._ready = True

    def _update_rot(self) -> None:
        for i in range(self.total):
            R = _quat_to_mat(self.q[i])
            self.R[i] = R
            self.invI[i] = R @ np.diag(self.ibody_inv[i]) @ R.T
            self.Iw[i] = R @ np.diag(self.ibody[i]) @ R.T

    # -- time stepping ---------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if not self._ready:
            raise RuntimeError("solver not initialized")
        target = int(round(float(time) / DT))
        while self._step_count < target:
            self._step(DT)
            self._step_count += 1
        return self._frame(float(time))

    def _step(self, dt: float) -> None:
        n = self.n
        # external forces: gravity (+ small damping for settling)
        self.v[:n] += self.g * dt
        self.v[:n] *= max(0.0, 1.0 - DAMPING * dt)
        self.w[:n] *= max(0.0, 1.0 - DAMPING * dt)
        # gyroscopic torque
        for i in range(n):
            L = self.Iw[i] @ self.w[i]
            self.w[i] += self.invI[i] @ (-np.cross(self.w[i], L)) * dt

        pv = None
        pw = None
        contacts = self._detect()
        if contacts:
            data = self._solve(contacts, dt)
            if data is not None:
                pv, pw = self._solve_pseudo(data, dt)

        # position integration (semi-implicit); pseudo velocities from the
        # split-impulse pass de-penetrate without touching real velocities
        if pv is None:
            self.x[:n] += self.v[:n] * dt
            w_int = self.w[:n]
        else:
            self.x[:n] += (self.v[:n] + pv[:n]) * dt
            w_int = self.w[:n] + pw[:n]
        wq = np.zeros(4)
        for i in range(n):
            wq[0:3] = w_int[i]
            dq = _quat_mul(wq, self.q[i])
            qn = self.q[i] + 0.5 * dt * dq
            self.q[i] = qn / np.linalg.norm(qn)
        self._update_rot()

    # -- collision detection ---------------------------------------------------
    def _detect(self):
        contacts = []
        total = self.total
        for i in range(total):
            for j in range(i + 1, total):
                if self.invm[i] == 0.0 and self.invm[j] == 0.0:
                    continue
                d = self.x[j] - self.x[i]
                reach = self.radius[i] + self.radius[j] + MARGIN
                if float(d @ d) > reach * reach:
                    continue
                sep, nrm, typ = _sat(self.x[i], self.R[i], self.half[i],
                                     self.x[j], self.R[j], self.half[j])
                if sep > 0.0:
                    continue  # separated: no contact constraint
                pts, depths = _manifold(self.x[i], self.R[i], self.half[i],
                                        self.x[j], self.R[j], self.half[j],
                                        nrm, typ, sep)
                if not pts:
                    continue
                contacts.append({
                    "ia": i, "ib": j, "n": nrm,
                    "pts": np.array(pts), "depths": np.array(depths),
                })
        return contacts

    # -- contact impulse solver --------------------------------------------------
    def _prepare(self, contacts, dt: float):
        """Flatten contact manifolds and precompute constraint terms."""
        ia_l, ib_l, ra_l, rb_l, n_l, dep_l, pair_l = [], [], [], [], [], [], []
        for k, c in enumerate(contacts):
            for p, dep in zip(c["pts"], c["depths"]):
                ia_l.append(c["ia"])
                ib_l.append(c["ib"])
                ra_l.append(p - self.x[c["ia"]])
                rb_l.append(p - self.x[c["ib"]])
                n_l.append(c["n"])
                dep_l.append(float(dep))
                pair_l.append(k)
        if not ia_l:
            return None

        ia = np.array(ia_l, dtype=int)
        ib = np.array(ib_l, dtype=int)
        ra = np.array(ra_l, dtype=float)
        rb = np.array(rb_l, dtype=float)
        n = np.array(n_l, dtype=float)
        depth = np.array(dep_l, dtype=float)
        C = len(ia)

        invIa = self.invI[ia]
        invIb = self.invI[ib]
        invma = self.invm[ia]
        invmb = self.invm[ib]

        def effective(dirs: np.ndarray) -> np.ndarray:
            raX = np.cross(ra, dirs)
            rbX = np.cross(rb, dirs)
            ka = np.einsum("ij,ij->i", raX, np.einsum("ijk,ik->ij", invIa, raX))
            kb = np.einsum("ij,ij->i", rbX, np.einsum("ijk,ik->ij", invIb, rbX))
            return invma + invmb + ka + kb

        kn = np.maximum(effective(n), 1.0e-12)
        ref = np.tile(np.array([0.0, 0.0, 1.0]), (C, 1))
        flip = np.abs(np.einsum("ij,ij->i", n, ref)) > 0.9
        ref[flip] = np.array([1.0, 0.0, 0.0])
        t1 = np.cross(n, ref)
        t1 /= np.maximum(np.linalg.norm(t1, axis=1), 1.0e-12)[:, None]
        t2 = np.cross(n, t1)
        k1 = np.maximum(effective(t1), 1.0e-12)
        k2 = np.maximum(effective(t2), 1.0e-12)

        return {
            "C": C, "contacts": contacts, "dt": dt,
            "ia": ia_l, "ib": ib_l, "pair": pair_l,
            "ra": ra.tolist(), "rb": rb.tolist(),
            "n": n.tolist(), "t1": t1.tolist(), "t2": t2.tolist(),
            "kn": kn.tolist(), "k1": k1.tolist(), "k2": k2.tolist(),
            "depth": depth.tolist(),
            "invm": self.invm.tolist(), "invI": self.invI.tolist(),
        }

    def _gs_pass(self, data, v, w, bias, iterations, friction):
        """Sequential (Gauss-Seidel) projected-impulse sweep.

        Normal impulses are clamped non-negative (non-penetrating,
        perfectly inelastic contacts); friction is bounded by the Coulomb
        cone.  Impulses are applied immediately so simultaneous contact
        points on one body cannot overshoot.
        """
        ia = data["ia"]
        ib = data["ib"]
        RA_ = data["ra"]
        RB_ = data["rb"]
        NN = data["n"]
        T1 = data["t1"]
        T2 = data["t2"]
        KN = data["kn"]
        K1 = data["k1"]
        K2 = data["k2"]
        BI = bias
        invm = data["invm"]
        invI = data["invI"]
        C = data["C"]
        lam_n = [0.0] * C
        lam_1 = [0.0] * C
        lam_2 = [0.0] * C

        for _ in range(iterations):
            for c in range(C):
                i = ia[c]
                j = ib[c]
                ra_ = RA_[c]
                rb_ = RB_[c]
                nc = NN[c]
                vi = v[i]
                vj = v[j]
                wi = w[i]
                wj = w[j]
                vrx = (vj[0] + wj[1] * rb_[2] - wj[2] * rb_[1]) - (vi[0] + wi[1] * ra_[2] - wi[2] * ra_[1])
                vry = (vj[1] + wj[2] * rb_[0] - wj[0] * rb_[2]) - (vi[1] + wi[2] * ra_[0] - wi[0] * ra_[2])
                vrz = (vj[2] + wj[0] * rb_[1] - wj[1] * rb_[0]) - (vi[2] + wi[0] * ra_[1] - wi[1] * ra_[0])
                vn = vrx * nc[0] + vry * nc[1] + vrz * nc[2]
                dl = -(vn + BI[c]) / KN[c]
                ln = lam_n[c] + dl
                if ln < 0.0:
                    ln = 0.0
                dl = ln - lam_n[c]
                lam_n[c] = ln
                if dl != 0.0:
                    self._impulse(v, w, i, j, invm, invI, ra_, rb_,
                                  dl * nc[0], dl * nc[1], dl * nc[2])
                    if friction:
                        vi = v[i]
                        vj = v[j]
                        wi = w[i]
                        wj = w[j]
                if not friction:
                    continue
                # Coulomb friction along the two tangent axes
                vrx = (vj[0] + wj[1] * rb_[2] - wj[2] * rb_[1]) - (vi[0] + wi[1] * ra_[2] - wi[2] * ra_[1])
                vry = (vj[1] + wj[2] * rb_[0] - wj[0] * rb_[2]) - (vi[1] + wi[2] * ra_[0] - wi[0] * ra_[2])
                vrz = (vj[2] + wj[0] * rb_[1] - wj[1] * rb_[0]) - (vi[2] + wi[0] * ra_[1] - wi[1] * ra_[0])
                tc = T1[c]
                vt = vrx * tc[0] + vry * tc[1] + vrz * tc[2]
                d = -vt / K1[c]
                lim = MU * lam_n[c]
                lt = lam_1[c] + d
                if lt > lim:
                    lt = lim
                elif lt < -lim:
                    lt = -lim
                d1 = lt - lam_1[c]
                lam_1[c] = lt
                tc = T2[c]
                vt = vrx * tc[0] + vry * tc[1] + vrz * tc[2]
                d = -vt / K2[c]
                lt = lam_2[c] + d
                if lt > lim:
                    lt = lim
                elif lt < -lim:
                    lt = -lim
                d2 = lt - lam_2[c]
                lam_2[c] = lt
                if d1 != 0.0 or d2 != 0.0:
                    u1 = T1[c]
                    u2 = T2[c]
                    self._impulse(v, w, i, j, invm, invI, ra_, rb_,
                                  d1 * u1[0] + d2 * u2[0],
                                  d1 * u1[1] + d2 * u2[1],
                                  d1 * u1[2] + d2 * u2[2])
        return lam_n, lam_1, lam_2

    def _solve(self, contacts, dt: float):
        """Real-velocity contact solve (perfectly inelastic, no bias)."""
        data = self._prepare(contacts, dt)
        if data is None:
            return None
        v = self.v.tolist()
        w = self.w.tolist()
        lam_n, lam_1, lam_2 = self._gs_pass(
            data, v, w, [0.0] * data["C"], ITERATIONS, True)
        self.v[:] = np.array(v, dtype=float)
        self.w[:] = np.array(w, dtype=float)

        # total impulse transmitted over the substep, aggregated per pair
        C = data["C"]
        imp = [[0.0, 0.0, 0.0] for _ in range(len(contacts))]
        NN = data["n"]
        T1 = data["t1"]
        T2 = data["t2"]
        pair = data["pair"]
        for c in range(C):
            acc = imp[pair[c]]
            nc = NN[c]
            u1 = T1[c]
            u2 = T2[c]
            acc[0] += lam_n[c] * nc[0] + lam_1[c] * u1[0] + lam_2[c] * u2[0]
            acc[1] += lam_n[c] * nc[1] + lam_1[c] * u1[1] + lam_2[c] * u2[1]
            acc[2] += lam_n[c] * nc[2] + lam_1[c] * u1[2] + lam_2[c] * u2[2]

        forces: dict[tuple[int, int], np.ndarray] = {}
        for k, con in enumerate(contacts):
            key = (con["ia"], con["ib"])
            f = np.array(imp[k], dtype=float) / dt
            if con["ib"] >= self.n:  # static ground partner: report the
                f = -f               # force acting on the dynamic body
            if key in forces:
                forces[key] = forces[key] + f
            else:
                forces[key] = f
        self._last_pair_force = forces
        return data

    def _solve_pseudo(self, data, dt: float):
        """Split-impulse de-penetration pass.

        Solves the same normal constraints against pseudo velocities that
        integrate positions only, so penetration is removed without
        injecting bounce into the real velocity state.
        """
        n = self.n
        v = [[0.0, 0.0, 0.0] for _ in range(self.total)]
        w = [[0.0, 0.0, 0.0] for _ in range(self.total)]
        bias = []
        for dep in data["depth"]:
            b = PSEUDO_BETA / dt * max(dep - SLOP, 0.0)
            bias.append(min(b, PSEUDO_CAP))
        self._gs_pass(data, v, w, bias, PSEUDO_ITERATIONS, False)
        pv = np.zeros((self.total, 3))
        pw = np.zeros((self.total, 3))
        for i in range(n):
            pv[i] = v[i]
            pw[i] = w[i]
        return pv, pw

    @staticmethod
    def _impulse(v, w, i, j, invm, invI, ra, rb, Px, Py, Pz):
        """Apply impulse (Px,Py,Pz) to body j and its reaction to body i."""
        mj = invm[j]
        if mj != 0.0:
            v[j][0] += Px * mj
            v[j][1] += Py * mj
            v[j][2] += Pz * mj
            tx = rb[1] * Pz - rb[2] * Py
            ty = rb[2] * Px - rb[0] * Pz
            tz = rb[0] * Py - rb[1] * Px
            Ij = invI[j]
            w[j][0] += Ij[0][0] * tx + Ij[0][1] * ty + Ij[0][2] * tz
            w[j][1] += Ij[1][0] * tx + Ij[1][1] * ty + Ij[1][2] * tz
            w[j][2] += Ij[2][0] * tx + Ij[2][1] * ty + Ij[2][2] * tz
        mi = invm[i]
        if mi != 0.0:
            v[i][0] -= Px * mi
            v[i][1] -= Py * mi
            v[i][2] -= Pz * mi
            tx = ra[1] * Pz - ra[2] * Py
            ty = ra[2] * Px - ra[0] * Pz
            tz = ra[0] * Py - ra[1] * Px
            Ii = invI[i]
            w[i][0] -= Ii[0][0] * tx + Ii[0][1] * ty + Ii[0][2] * tz
            w[i][1] -= Ii[1][0] * tx + Ii[1][1] * ty + Ii[1][2] * tz
            w[i][2] -= Ii[2][0] * tx + Ii[2][1] * ty + Ii[2][2] * tz

    # -- frame export --------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        n = self.n
        seps: list[float] = []
        pairs: list[tuple[int, int]] = []
        for i in range(self.total):
            for j in range(i + 1, self.total):
                if self.invm[i] == 0.0 and self.invm[j] == 0.0:
                    continue
                d = self.x[j] - self.x[i]
                reach = self.radius[i] + self.radius[j] + NEAR
                if float(d @ d) > reach * reach:
                    continue
                sep, _, _ = _sat(self.x[i], self.R[i], self.half[i],
                                 self.x[j], self.R[j], self.half[j])
                if sep < NEAR:
                    seps.append(sep)
                    pairs.append((i, j))

        if seps:
            csd = np.array(seps, dtype=float)
        else:
            csd = np.zeros(1)

        if pairs:
            inter = np.array([self._last_pair_force.get(key, np.zeros(3))
                              for key in pairs], dtype=float)
        else:
            inter = np.zeros((1, 3))

        external = self.mass[:, None] * self.g[None, :]

        return StateFrame(
            time=time,
            position=self.x[:n].copy(),
            velocity=self.v[:n].copy(),
            orientation=(self.q[:n] / np.linalg.norm(
                self.q[:n], axis=1)[:, None]).copy(),
            angular_velocity=self.w[:n].copy(),
            mass=self.mass.copy(),
            body_id=self.body_ids.copy(),
            fields={
                "contact_signed_distance": csd,
                "external_force": external,
                "interaction_force": inter,
            },
            scalars={"contact_length_scale": CONTACT_LENGTH_SCALE},
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "impulse-based rigid body dynamics (SAT + Coulomb friction)",
            "substep_dt": DT,
            "substeps": self._step_count,
            "iterations": ITERATIONS,
            "friction": MU,
            "bodies": int(self.n),
        }
