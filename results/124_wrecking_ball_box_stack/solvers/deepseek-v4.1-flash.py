"""Self-contained rigid-body solver for the wrecking-ball / box-stack scene.

Physical model
--------------
* Semi-implicit (symplectic) Euler time integration with a fixed micro step of
  4 ms; every requested sample time lands exactly on a step boundary.
* Contacts between the frozen convex primitives (boxes, the spherical wrecking
  ball, the capsule chain links and the rigid floor slab) are generated with a
  full separating-axis test plus reference/incident face clipping for box-box
  manifolds.  Contacts are solved with a projected Gauss-Seidel sequential
  impulse solver including Coulomb friction, restitution and Baumgarte
  penetration stabilisation.
* The wrecking ball and the six interlinked chain links form one compound rigid
  body (the links are rigidly interlinked, which is the rigid limit of the taut
  chain).  It is suspended from the fixed overhead anchor by a ball-and-socket
  constraint solved as a 3x3 impulse system plus a position projection, so the
  chain can never separate or stretch.
* The ball starts at the frozen configuration with pendulum motion about the
  anchor, i.e. the state of a ball just released from a raised, displaced
  position with the chain taut; from there the motion evolves under gravity and
  contacts alone.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-12


def _norm(v):
    return float(np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def _cross(a, b):
    return np.array(
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]],
        dtype=float,
    )


def _skew(v):
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]], dtype=float
    )


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


def _quat_norm(q):
    n = float(np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def _quat_mat(q):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def _quat_integrate(q, w, dt):
    wl = _norm(w)
    if wl < 1e-12:
        return _quat_norm(q)
    ang = wl * dt
    s = np.sin(0.5 * ang) / wl
    dq = np.array([w[0] * s, w[1] * s, w[2] * s, np.cos(0.5 * ang)], dtype=float)
    return _quat_norm(_quat_mul(dq, q))


# --------------------------------------------------------------------------
# convex primitive collision (support-free, explicit geometry)
# --------------------------------------------------------------------------
def _clip_polygon(poly, n, d):
    """Sutherland-Hodgman: keep the part of the polygon with dot(n, p) <= d."""
    out = []
    m = len(poly)
    for i in range(m):
        a = poly[i]
        b = poly[(i + 1) % m]
        da = float(n @ a) - d
        db = float(n @ b) - d
        if da <= 0.0:
            out.append(a)
        if (da < 0.0) != (db < 0.0):
            t = da / (da - db)
            out.append(a + t * (b - a))
    return out


def _box_box(ca, ra, ha, cb, rb, hb, margin):
    """Manifold between two oriented boxes; normal points from A to B."""
    t = cb - ca
    ax_a = (ra[:, 0], ra[:, 1], ra[:, 2])
    ax_b = (rb[:, 0], rb[:, 1], rb[:, 2])
    face = None
    for i in range(3):
        ax = ax_a[i]
        rax = sum(ha[k] * abs(float(ax @ ax_a[k])) for k in range(3))
        rbx = sum(hb[k] * abs(float(ax @ ax_b[k])) for k in range(3))
        d = float(ax @ t)
        ov = rax + rbx - abs(d)
        if ov <= 0.0:
            return []
        if face is None or ov < face[0]:
            face = (ov, ax * (1.0 if d >= 0.0 else -1.0), "a", i)
    for j in range(3):
        ax = ax_b[j]
        rax = sum(ha[k] * abs(float(ax @ ax_a[k])) for k in range(3))
        rbx = sum(hb[k] * abs(float(ax @ ax_b[k])) for k in range(3))
        d = float(ax @ t)
        ov = rax + rbx - abs(d)
        if ov <= 0.0:
            return []
        if ov < face[0]:
            face = (ov, ax * (1.0 if d >= 0.0 else -1.0), "b", j)
    cross = None
    for i in range(3):
        for j in range(3):
            ax = _cross(ax_a[i], ax_b[j])
            ln = _norm(ax)
            if ln < 1e-6:
                continue
            ax = ax / ln
            rax = sum(ha[k] * abs(float(ax @ ax_a[k])) for k in range(3))
            rbx = sum(hb[k] * abs(float(ax @ ax_b[k])) for k in range(3))
            d = float(ax @ t)
            ov = rax + rbx - abs(d)
            if ov <= 0.0:
                return []
            if cross is None or ov < cross[0]:
                cross = (ov, ax * (1.0 if d >= 0.0 else -1.0), i, j)
    if cross is not None and cross[0] < face[0] - 1e-6:
        ov, n_ab, _i, _j = cross
        pa = ca.copy()
        pb = cb.copy()
        for k in range(3):
            pa = pa + ha[k] * np.sign(float(ax_a[k] @ n_ab)) * ax_a[k]
            pb = pb - hb[k] * np.sign(float(ax_b[k] @ n_ab)) * ax_b[k]
        return [(0.5 * (pa + pb), n_ab, ov)]
    ov, n_ab, which, idx = face
    if which == "a":
        ref_c, ref_r, ref_h, ref_i = ca, ra, ha, idx
        ref_n = n_ab
        inc_c, inc_r, inc_h = cb, rb, hb
    else:
        ref_c, ref_r, ref_h, ref_i = cb, rb, hb, idx
        ref_n = -n_ab
        inc_c, inc_r, inc_h = ca, ra, ha
    face_center = ref_c + ref_h[ref_i] * ref_n
    others = [k for k in range(3) if k != ref_i]
    u = ref_r[:, others[0]]
    v = ref_r[:, others[1]]
    hu = ref_h[others[0]]
    hv = ref_h[others[1]]
    dots = [float(inc_r[:, k] @ ref_n) for k in range(3)]
    kk = int(np.argmin(dots))
    sgn = 1.0 if dots[kk] <= 0.0 else -1.0
    inc_normal = inc_r[:, kk] * sgn
    j1, j2 = [k for k in range(3) if k != kk]
    fc = inc_c + inc_h[kk] * inc_normal
    e1 = inc_r[:, j1] * inc_h[j1]
    e2 = inc_r[:, j2] * inc_h[j2]
    quad = [fc + e1 + e2, fc + e1 - e2, fc - e1 - e2, fc - e1 + e2]
    for n_p, d_p in ((-u, hu), (u, hu), (-v, hv), (v, hv)):
        quad = _clip_polygon(quad, n_p, float(n_p @ ref_c) + d_p)
        if len(quad) < 3:
            return []
    pts = []
    for p in quad:
        sep = float(ref_n @ (p - face_center))
        if sep > margin:
            continue
        dup = False
        for q in pts:
            if _norm(p - q[0]) < 1e-5:
                dup = True
                break
        if not dup:
            pts.append((p, n_ab, -sep))
    return pts


def _box_sphere(cb, rb, hb, cs, radius, margin):
    """Normal points from the box towards the sphere."""
    local = rb.T @ (cs - cb)
    cl = np.clip(local, -hb, hb)
    closest = cb + rb @ cl
    delta = cs - closest
    dist = _norm(delta)
    if dist > radius + margin:
        return []
    if dist > 1e-9:
        n = delta / dist
        return [(closest, n, radius - dist)]
    diffs = [hb[k] - abs(local[k]) for k in range(3)]
    k = int(np.argmin(diffs))
    s = 1.0 if local[k] >= 0.0 else -1.0
    n = rb[:, k] * s
    return [(cs - n * radius, n, radius + diffs[k])]


def _box_capsule(cb, rb, hb, cc, rc, hl, margin):
    """Sample the capsule axis and treat each sample as a sphere."""
    out = []
    axis = rc[:, 2]
    for f in (-1.0, -0.5, 0.0, 0.5, 1.0):
        cs = cc + axis * (hl * f)
        out.extend(_box_sphere(cb, rb, hb, cs, rc, margin))
    return out


def _sphere_sphere(ca, ra, cb, rb, margin):
    delta = cb - ca
    dist = _norm(delta)
    if dist > ra + rb + margin:
        return []
    if dist > 1e-9:
        n = delta / dist
    else:
        n = np.array([0.0, 0.0, 1.0])
    return [(ca + n * ra, n, ra + rb - dist)]


def _prim_contacts(k1, c1, r1, p1, k2, c2, r2, p2, margin):
    """Contacts between two world-frame primitives; normal from 1 to 2."""
    if k1 == "box" and k2 == "box":
        return _box_box(c1, r1, p1[0], c2, r2, p2[0], margin)
    if k1 == "box" and k2 == "sphere":
        return _box_sphere(c1, r1, p1[0], c2, p2[0], margin)
    if k1 == "sphere" and k2 == "box":
        return [(pt, -n, d) for (pt, n, d) in _box_sphere(c2, r2, p2[0], c1, p1[0], margin)]
    if k1 == "box" and k2 == "capsule":
        return _box_capsule(c1, r1, p1[0], c2, r2, p2[0], p2[1], margin)
    if k1 == "capsule" and k2 == "box":
        return [
            (pt, -n, d)
            for (pt, n, d) in _box_capsule(c2, r2, p2[0], c1, r1, p1[0], p1[1], margin)
        ]
    if k1 == "sphere" and k2 == "sphere":
        return _sphere_sphere(c1, p1[0], c2, p2[0], margin)
    return []


class _Body:
    __slots__ = (
        "kind",
        "params",
        "prims",
        "mass",
        "inv_mass",
        "i_local",
        "iinv_local",
        "pos",
        "vel",
        "quat",
        "rot",
        "omega",
        "iinv",
        "static",
    )

    def __init__(self, kind, params, mass, pos, quat=None, prims=None):
        self.kind = kind
        self.params = params
        self.prims = prims
        self.mass = float(mass)
        self.inv_mass = 0.0 if mass <= 0.0 else 1.0 / float(mass)
        self.static = mass <= 0.0
        self.pos = np.asarray(pos, dtype=float).copy()
        self.vel = np.zeros(3)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0]) if quat is None else _quat_norm(np.asarray(quat, dtype=float))
        self.rot = _quat_mat(self.quat)
        self.omega = np.zeros(3)
        self.i_local = np.zeros((3, 3))
        self.iinv_local = np.zeros((3, 3))
        self.iinv = np.zeros((3, 3))

    def set_local_inertia(self, mat):
        self.i_local = np.asarray(mat, dtype=float)
        if self.static:
            self.iinv_local = np.zeros((3, 3))
        else:
            try:
                self.iinv_local = np.linalg.inv(self.i_local)
            except np.linalg.LinAlgError:
                self.iinv_local = np.zeros((3, 3))
        self.update_world()

    def update_world(self):
        if self.static:
            self.iinv = np.zeros((3, 3))
        else:
            self.iinv = self.rot @ self.iinv_local @ self.rot.T


def _prims_world(body):
    if body.prims is None:
        return [(body.kind, body.pos, body.rot, body.params)]
    out = []
    for kind, off, params in body.prims:
        out.append((kind, body.pos + body.rot @ off, body.rot, params))
    return out


def _prim_aabb(kind, c, rot, params):
    if kind == "box":
        ext = np.abs(rot) @ params[0]
    elif kind == "sphere":
        ext = np.full(3, params[0])
    else:
        ext = np.full(3, params[0]) + params[1] * np.abs(rot[:, 2])
    return c - ext, c + ext


def _box_inertia(mass, h):
    return np.diag(
        [
            mass / 3.0 * (h[1] ** 2 + h[2] ** 2),
            mass / 3.0 * (h[0] ** 2 + h[2] ** 2),
            mass / 3.0 * (h[0] ** 2 + h[1] ** 2),
        ]
    )


def _sphere_inertia(mass, radius):
    i = 0.4 * mass * radius * radius
    return np.diag([i, i, i])


def _capsule_inertia(mass, radius, half_len):
    iz = 0.5 * mass * radius * radius
    ix = mass * (0.25 * radius * radius + (1.0 / 3.0) * half_len * half_len)
    return np.diag([ix, ix, iz])


class _Contact:
    __slots__ = ("a", "b", "p", "n", "depth", "ra", "rb", "kn", "n_imp", "t_imp", "target", "imp")

    def __init__(self, a, b, p, n, depth):
        self.a = a
        self.b = b
        self.p = p
        self.n = n
        self.depth = float(depth)
        self.ra = np.zeros(3)
        self.rb = np.zeros(3)
        self.kn = 1.0
        self.n_imp = 0.0
        self.t_imp = 0.0
        self.target = 0.0
        self.imp = np.zeros(3)


class BenchmarkSolver:
    """Rigid-body wrecking-ball simulation (self-contained, deterministic)."""

    def __init__(self):
        self.ready = False

    # -- setup ------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        contact = cfg.get("contact", {})
        self.mu = float(contact.get("friction", 0.55))
        self.rest = float(contact.get("restitution", 0.12))
        self.length_scale = float(contact.get("length_scale", 0.02))
        self.dt = 0.004
        self.iters = 10
        self.margin = 0.02
        self.beta = 0.2
        self.slop = 0.004
        self.swing_rate = 1.0
        self.steps = 0
        self.sim_time = 0.0

        bodies_cfg = cfg["bodies"]
        ids = [str(b["id"]) for b in bodies_cfg]
        self.n_rows = len(bodies_cfg)
        self.mass_cfg = np.asarray([float(b["mass"]) for b in bodies_cfg], dtype=float)
        pos_cfg = np.asarray([b["position"] for b in bodies_cfg], dtype=float)
        self.body_id = np.arange(self.n_rows, dtype=np.int32)
        self.mass = self.mass_cfg.copy()

        geo = cfg.get("scene_geometry") or {}
        gmap = {obj.get("id"): obj for obj in geo.get("objects", [])}

        # ---- rigid floor ------------------------------------------------
        floor_spec = gmap.get("floor", {})
        f_geo = floor_spec.get("geometry", {}) or {}
        f_half = np.asarray(f_geo.get("half_extents", [3.2, 2.2, 0.05]), dtype=float)
        f_pos = np.asarray(
            (floor_spec.get("transform", {}) or {}).get("translation", [0.0, 0.0, -0.05]),
            dtype=float,
        )
        self.floor = _Body("box", (f_half,), 0.0, f_pos)
        self.floor.set_local_inertia(np.zeros((3, 3)))

        # ---- boxes ------------------------------------------------------
        self.box_bodies = []
        box_rows = {}
        for i, bid in enumerate(ids):
            if not bid.startswith("box_"):
                continue
            spec = gmap.get(bid, {})
            g = spec.get("geometry", {}) or {}
            half = np.asarray(g.get("half_extents", [0.17, 0.38, 0.17]), dtype=float)
            quat = np.asarray(
                (spec.get("transform", {}) or {}).get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]),
                dtype=float,
            )
            body = _Body("box", (half,), self.mass_cfg[i], pos_cfg[i], quat)
            body.set_local_inertia(_box_inertia(self.mass_cfg[i], half))
            self.box_bodies.append(body)
            box_rows[i] = body

        # ---- wrecking ball + interlinked chain (one compound pendulum) ---
        ball_idx = ids.index("wrecking_ball")
        chain_idx = [i for i, bid in enumerate(ids) if bid.startswith("chain_")]
        self.ball_row = ball_idx
        self.chain_rows = chain_idx
        ball_mass = float(self.mass_cfg[ball_idx])
        ball_pos = pos_cfg[ball_idx]
        link_mass = float(self.mass_cfg[chain_idx[0]])
        link_pos = pos_cfg[chain_idx]
        c_spec = gmap.get(ids[chain_idx[0]], {})
        c_geo = c_spec.get("geometry", {}) or {}
        cap_r = float(c_geo.get("radius", 0.07))
        cap_h = float(c_geo.get("height", 0.25))
        cap_hl = 0.5 * cap_h
        ball_r = 0.43
        total = ball_mass + link_mass * len(chain_idx)
        com = (ball_mass * ball_pos + link_mass * link_pos.sum(axis=0)) / total
        self.ball_off = ball_pos - com
        self.link_off = [p - com for p in link_pos]
        prims = [("sphere", self.ball_off.copy(), (ball_r,))]
        for off in self.link_off:
            prims.append(("capsule", off.copy(), (cap_r, cap_hl)))
        inertia = np.zeros((3, 3))
        eye = np.eye(3)
        for m, d, mat in (
            (ball_mass, self.ball_off, _sphere_inertia(ball_mass, ball_r)),
            *[(link_mass, off, _capsule_inertia(link_mass, cap_r, cap_hl)) for off in self.link_off],
        ):
            inertia = inertia + mat + m * ((d @ d) * eye - np.outer(d, d))
        self.pendulum = _Body("compound", None, total, com, prims=prims)
        self.pendulum.set_local_inertia(inertia)
        pivot_local = self.link_off[-1] + np.array([0.0, 0.0, cap_hl + cap_r])
        self.pivot_local = pivot_local
        self.anchor = self.pendulum.pos + self.pivot_local.copy()
        pv = self.anchor - self.pendulum.pos
        omega0 = np.array([0.0, -self.swing_rate, 0.0])
        self.pendulum.omega = omega0
        self.pendulum.vel = _cross(omega0, -pv)

        # ---- bookkeeping ------------------------------------------------
        self.pendulum.update_world()
        self.phys = self.box_bodies + [self.pendulum, self.floor]
        self.dyn = [b for b in self.phys if not b.static]
        chain_map = {row: k for k, row in enumerate(chain_idx)}
        self.rows = []
        for i in range(self.n_rows):
            if i == ball_idx:
                self.rows.append(("ball", None))
            elif i in chain_map:
                self.rows.append(("link", chain_map[i]))
            elif i in box_rows:
                self.rows.append(("body", box_rows[i]))
            else:
                self.rows.append(("rest", np.zeros(3)))
        self._last_contacts = []
        self.ready = True

    # -- collision detection ---------------------------------------------
    def _collide(self):
        bodies = self.phys
        n = len(bodies)
        prims = [_prims_world(b) for b in bodies]
        los = np.empty((n, 3))
        his = np.empty((n, 3))
        for i in range(n):
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            for kind, c, rot, params in prims[i]:
                plo, phi = _prim_aabb(kind, c, rot, params)
                lo = np.minimum(lo, plo)
                hi = np.maximum(hi, phi)
            los[i] = lo
            his[i] = hi
        margin = self.margin
        contacts = []
        for i in range(n):
            bi = bodies[i]
            for j in range(i + 1, n):
                bj = bodies[j]
                if bi.static and bj.static:
                    continue
                if np.any(los[i] > his[j]) or np.any(los[j] > his[i]):
                    continue
                pi = prims[i]
                pj = prims[j]
                for kind1, c1, r1, par1 in pi:
                    for kind2, c2, r2, par2 in pj:
                        for pt, nrm, depth in _prim_contacts(
                            kind1, c1, r1, par1, kind2, c2, r2, par2, margin
                        ):
                            contacts.append(_Contact(bi, bj, pt, nrm, depth))
        return contacts

    def _prepare(self, contacts, dt):
        inv_dt = 1.0 / dt
        for c in contacts:
            a = c.a
            b = c.b
            c.ra = c.p - a.pos
            c.rb = c.p - b.pos
            n = c.n
            kn = a.inv_mass + b.inv_mass
            if not a.static:
                kn += float(n @ _cross(a.iinv @ _cross(c.ra, n), c.ra))
            if not b.static:
                kn += float(n @ _cross(b.iinv @ _cross(c.rb, n), c.rb))
            c.kn = kn if kn > 1e-12 else 1e-12
            va = a.vel + _cross(a.omega, c.ra)
            vb = b.vel + _cross(b.omega, c.rb)
            vn0 = float((vb - va) @ n)
            if c.depth < 0.0:
                c.target = c.depth * inv_dt
            else:
                bias = self.beta * inv_dt * max(0.0, c.depth - self.slop)
                rest = -self.rest * vn0 if vn0 < -1.0 else 0.0
                c.target = bias if bias > rest else rest

    def _solve_contact(self, c):
        a = c.a
        b = c.b
        ra = c.ra
        rb = c.rb
        n = c.n
        va = a.vel + _cross(a.omega, ra)
        vb = b.vel + _cross(b.omega, rb)
        vn = float((vb - va) @ n)
        dl = (c.target - vn) / c.kn
        new = c.n_imp + dl
        if new < 0.0:
            new = 0.0
        dl = new - c.n_imp
        c.n_imp = new
        if dl != 0.0:
            imp = dl * n
            if not a.static:
                a.vel -= a.inv_mass * imp
                a.omega -= a.iinv @ _cross(ra, imp)
            if not b.static:
                b.vel += b.inv_mass * imp
                b.omega += b.iinv @ _cross(rb, imp)
            c.imp += imp
        # Coulomb friction
        va = a.vel + _cross(a.omega, ra)
        vb = b.vel + _cross(b.omega, rb)
        vrel = vb - va
        vt = vrel - float(vrel @ n) * n
        vtl = _norm(vt)
        if vtl > 1e-7:
            t = vt / vtl
            kt = a.inv_mass + b.inv_mass
            if not a.static:
                kt += float(t @ _cross(a.iinv @ _cross(ra, t), ra))
            if not b.static:
                kt += float(t @ _cross(b.iinv @ _cross(rb, t), rb))
            if kt < 1e-12:
                return
            dl = -vtl / kt
            lim = self.mu * c.n_imp
            new = c.t_imp + dl
            if new > lim:
                new = lim
            elif new < -lim:
                new = -lim
            dl = new - c.t_imp
            c.t_imp = new
            if dl != 0.0:
                imp = dl * t
                if not a.static:
                    a.vel -= a.inv_mass * imp
                    a.omega -= a.iinv @ _cross(ra, imp)
                if not b.static:
                    b.vel += b.inv_mass * imp
                    b.omega += b.iinv @ _cross(rb, imp)
                c.imp += imp

    def _solve_pivot(self):
        p = self.pendulum
        r = p.rot @ self.pivot_local
        vrel = p.vel + _cross(p.omega, r)
        k = p.inv_mass * np.eye(3) + _skew(r).T @ p.iinv @ _skew(r)
        k = k + 1e-9 * np.eye(3)
        lam = np.linalg.solve(k, -vrel)
        p.vel += p.inv_mass * lam
        p.omega += p.iinv @ _cross(r, lam)

    def _project_pivot(self):
        p = self.pendulum
        r = p.rot @ self.pivot_local
        err = self.anchor - (p.pos + r)
        p.pos += 0.8 * err

    def _step(self, dt):
        for b in self.dyn:
            b.vel = b.vel + self.gravity * dt
            b.update_world()
        contacts = self._collide()
        self._last_contacts = contacts
        self._prepare(contacts, dt)
        for _ in range(self.iters):
            for c in contacts:
                self._solve_contact(c)
            self._solve_pivot()
        vmax = 25.0
        for b in self.dyn:
            sp = _norm(b.vel)
            if sp > vmax:
                b.vel = b.vel * (vmax / sp)
            sp = _norm(b.omega)
            if sp > vmax:
                b.omega = b.omega * (vmax / sp)
            b.pos = b.pos + b.vel * dt
            b.quat = _quat_integrate(b.quat, b.omega, dt)
            b.rot = _quat_mat(b.quat)
            b.update_world()
        self._project_pivot()
        self.pendulum.rot = _quat_mat(self.pendulum.quat)
        self.pendulum.update_world()

    # -- sampling ---------------------------------------------------------
    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        while self.steps < target:
            self._step(self.dt)
            self.steps += 1
        self.sim_time = float(time)
        return self._frame(float(time))

    def diagnostics(self):
        if not self._last_contacts:
            return {"steps": self.steps}
        depths = [c.depth for c in self._last_contacts]
        return {
            "steps": self.steps,
            "contacts": len(self._last_contacts),
            "max_depth": float(max(depths)),
            "min_depth": float(min(depths)),
        }

    def _frame(self, time):
        n = self.n_rows
        pos = np.zeros((n, 3))
        vel = np.zeros((n, 3))
        quat = np.zeros((n, 4))
        om = np.zeros((n, 3))
        p = self.pendulum
        for i, (kind, ref) in enumerate(self.rows):
            if kind == "body":
                b = ref
                pos[i] = b.pos
                vel[i] = b.vel
                quat[i] = b.quat
                om[i] = b.omega
            elif kind == "ball":
                r = p.rot @ self.ball_off
                pos[i] = p.pos + r
                vel[i] = p.vel + _cross(p.omega, r)
                quat[i] = p.quat
                om[i] = p.omega
            elif kind == "link":
                r = p.rot @ self.link_off[ref]
                pos[i] = p.pos + r
                vel[i] = p.vel + _cross(p.omega, r)
                quat[i] = p.quat
                om[i] = p.omega
            else:
                pos[i] = ref
                quat[i] = np.array([0.0, 0.0, 0.0, 1.0])
        contacts = self._last_contacts
        if not contacts:
            contacts = self._collide()
        if contacts:
            sdist = np.asarray([-c.depth for c in contacts], dtype=float)
            iforce = np.asarray([c.imp / self.dt for c in contacts], dtype=float)
        else:
            sdist = np.asarray([self.margin], dtype=float)
            iforce = np.zeros((1, 3))
        eforce = self.mass_cfg[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(time),
            pos,
            vel,
            orientation=quat,
            angular_velocity=om,
            mass=self.mass_cfg.copy(),
            body_id=self.body_id.copy(),
            scalars={"contact_length_scale": self.length_scale},
            fields={
                "contact_signed_distance": sdist,
                "external_force": eforce,
                "interaction_force": iforce,
            },
        )
        frame.validate()
        return frame
