"""Frictional mechanism jamming: impulse-based rigid-body contact dynamics.

The frozen scene is an articulated mechanism of eight interlocked L-shaped
links (a base plate plus an upright post) confined in a channel between two
guide rails.  The solver advances the assembly with

* semi-implicit (symplectic) Euler time stepping at 1 ms substeps,
* separating-axis box-box collision with face-clipping contact manifolds
  (each link is the rigid union of two boxes; the guides and floor are
  static boxes),
* a projected Gauss-Seidel contact solver with a normal/friction split
  (Staggered-Projection style) using Coulomb friction cones, and
* split-impulse (pseudo-velocity) position correction so penetration is
  removed without injecting energy.

A ramped external drive loads the leftmost link in +x.  Contact forces
propagate link-to-link through the interlocked chain while friction
progressively resists relative motion, and the mechanism settles into a
visibly jammed, interlock-locked configuration free of persistent overlap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# quaternion helpers (x, y, z, w ordering)
# ---------------------------------------------------------------------------
def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quat_integrate(q: np.ndarray, w: np.ndarray, h: float) -> np.ndarray:
    dq = _quat_multiply(np.array([w[0], w[1], w[2], 0.0]), q)
    return _quat_normalize(q + 0.5 * h * dq)


# ---------------------------------------------------------------------------
# link geometry (L-shaped rigid link = base plate box + post box)
# ---------------------------------------------------------------------------
_FALLBACK_BOXES = (
    ((0.0, 0.0, -0.08), (0.24, 0.16, 0.08)),  # base plate
    ((0.16, 0.0, 0.07), (0.08, 0.16, 0.23)),  # upright post
)


def _load_link_boxes(asset: str | None):
    """Derive the two axis-aligned boxes of the L-link from its OBJ asset.

    Returns ``(collision_boxes, mass_boxes)`` where ``mass_boxes`` are
    non-overlapping (the post is clipped above the base) for correct mass
    properties.
    """
    candidates = []
    if asset:
        candidates.append(Path(asset))
        candidates.append(Path(__file__).resolve().parents[1] / asset)
    verts: list[tuple[float, float, float]] = []
    for cand in candidates:
        try:
            with open(cand, "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("v "):
                        parts = line.split()
                        verts.append(
                            (float(parts[1]), float(parts[2]), float(parts[3]))
                        )
            break
        except OSError:
            continue
    if len(verts) < 8:
        base_c, base_e = _FALLBACK_BOXES[0]
        post_c, post_e = _FALLBACK_BOXES[1]
        split_z = 0.0
    else:
        v = np.asarray(verts, dtype=float)
        zs = np.unique(np.round(v[:, 2], 6))
        zmin, z2, ztop = float(zs[0]), float(zs[-2]), float(zs[-1])
        y0, y1 = float(v[:, 1].min()), float(v[:, 1].max())
        top = v[v[:, 2] >= ztop - 1e-9]
        mid = v[np.abs(v[:, 2] - z2) <= 1e-9]
        px0, px1 = float(top[:, 0].min()), float(top[:, 0].max())
        bx0, bx1 = float(mid[:, 0].min()), float(mid[:, 0].max())
        base_c = (0.5 * (bx0 + bx1), 0.5 * (y0 + y1), 0.5 * (zmin + z2))
        base_e = (0.5 * (bx1 - bx0), 0.5 * (y1 - y0), 0.5 * (z2 - zmin))
        post_c = (0.5 * (px0 + px1), 0.5 * (y0 + y1), 0.5 * (zmin + ztop))
        post_e = (0.5 * (px1 - px0), 0.5 * (y1 - y0), 0.5 * (ztop - zmin))
        split_z = z2
    # non-overlapping decomposition for mass properties: the post is
    # clipped to the portion above the base plate (z >= split_z)
    post_top = post_c[2] + post_e[2]
    mass_post_c = (post_c[0], post_c[1], 0.5 * (split_z + post_top))
    mass_post_e = (post_e[0], post_e[1], 0.5 * (post_top - split_z))
    collision_boxes = [
        (np.asarray(base_c, dtype=float), np.asarray(base_e, dtype=float)),
        (np.asarray(post_c, dtype=float), np.asarray(post_e, dtype=float)),
    ]
    mass_boxes = [
        (np.asarray(base_c, dtype=float), np.asarray(base_e, dtype=float)),
        (np.asarray(mass_post_c, dtype=float), np.asarray(mass_post_e, dtype=float)),
    ]
    return collision_boxes, mass_boxes


def _composite_mass_properties(mass_boxes, total_mass: float):
    volumes = [8.0 * float(np.prod(he)) for _, he in mass_boxes]
    total_volume = sum(volumes)
    density = total_mass / max(total_volume, 1e-12)
    masses = [density * v for v in volumes]
    com = np.zeros(3)
    for m, (c, _) in zip(masses, mass_boxes):
        com += m * c
    com /= total_mass
    inertia = np.zeros((3, 3))
    for m, (c, he) in zip(masses, mass_boxes):
        hx, hy, hz = (float(e) for e in he)
        diag = np.diag(
            [
                (m / 3.0) * (hy * hy + hz * hz),
                (m / 3.0) * (hx * hx + hz * hz),
                (m / 3.0) * (hx * hx + hy * hy),
            ]
        )
        d = c - com
        inertia += diag + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return com, inertia


# ---------------------------------------------------------------------------
# box-box collision (SAT with reference-face clipping)
# ---------------------------------------------------------------------------
def _clip_plane(poly, normal: np.ndarray, limit: float):
    kept = []
    count = len(poly)
    if count == 0:
        return kept
    for idx in range(count):
        p = poly[idx]
        q = poly[(idx + 1) % count]
        dp = float(normal @ p) - limit
        dq = float(normal @ q) - limit
        if dp <= 0.0:
            kept.append(p)
        if (dp < 0.0 < dq) or (dq < 0.0 < dp):
            t = dp / (dp - dq)
            kept.append(p + t * (q - p))
    return kept


def _clip_face_points(ref_c, ref_R, ref_e, ref_i, ref_n, inc_c, inc_R, inc_e, tol):
    inc_axes = (inc_R[:, 0], inc_R[:, 1], inc_R[:, 2])
    dots = [float(inc_axes[k] @ ref_n) for k in range(3)]
    # the incident face is the one whose outward normal (either +u_k or -u_k)
    # is most anti-parallel to the reference normal
    j = int(np.argmax([abs(v) for v in dots]))
    m = inc_axes[j] if dots[j] < 0.0 else -inc_axes[j]
    face_center = inc_c + m * inc_e[j]
    others = [k for k in range(3) if k != j]
    u1, u2 = inc_axes[others[0]], inc_axes[others[1]]
    h1, h2 = inc_e[others[0]], inc_e[others[1]]
    # proper quad loop (counter-clockwise around the incident face normal)
    poly = [
        face_center + h1 * u1 + h2 * u2,
        face_center + h1 * u1 - h2 * u2,
        face_center - h1 * u1 - h2 * u2,
        face_center - h1 * u1 + h2 * u2,
    ]
    for k in range(3):
        if k == ref_i:
            continue
        t = ref_R[:, k]
        center_proj = float(ref_c @ t)
        ext = ref_e[k]
        poly = _clip_plane(poly, t, center_proj + ext)
        if not poly:
            return []
        poly = _clip_plane(poly, -t, -(center_proj - ext))
        if not poly:
            return []
    ref_face_center = ref_c + ref_n * ref_e[ref_i]
    points = []
    for p in poly:
        sep = float((p - ref_face_center) @ ref_n)
        if sep <= tol:
            points.append((p, -sep))
    return points


def _collide_boxes(cA, RA, eA, cB, RB, eB, margin):
    """SAT collision between two oriented boxes.

    Returns ``(normal, points)`` with the unit normal pointing from box A to
    box B and ``points`` a list of ``(world_point, depth)`` (positive depth =
    penetration), or ``None`` when the boxes are separated by more than
    ``margin``.
    """
    a = (RA[:, 0], RA[:, 1], RA[:, 2])
    b = (RB[:, 0], RB[:, 1], RB[:, 2])
    d = cB - cA
    best = np.inf
    kind = -1  # 0: face of A, 1: face of B, 2: edge-edge
    axis_i = -1
    axis_j = -1
    sign = 1.0
    for i in range(3):
        u = a[i]
        dist = float(d @ u)
        s = 1.0 if dist >= 0.0 else -1.0
        rb = (
            eB[0] * abs(float(b[0] @ u))
            + eB[1] * abs(float(b[1] @ u))
            + eB[2] * abs(float(b[2] @ u))
        )
        o = eA[i] + rb - abs(dist)
        if o < -margin:
            return None
        if o < best:
            best = o
            kind = 0
            axis_i = i
            sign = s
    for j in range(3):
        u = b[j]
        dist = float(d @ u)
        s = 1.0 if dist >= 0.0 else -1.0
        ra = (
            eA[0] * abs(float(a[0] @ u))
            + eA[1] * abs(float(a[1] @ u))
            + eA[2] * abs(float(a[2] @ u))
        )
        o = eB[j] + ra - abs(dist)
        if o < -margin:
            return None
        if o < best:
            best = o
            kind = 1
            axis_j = j
            sign = s
    edge_normal = None
    for i in range(3):
        for j in range(3):
            L = np.cross(a[i], b[j])
            ln = float(np.linalg.norm(L))
            if ln < 1e-6:
                continue
            L = L / ln
            dist = float(d @ L)
            s = 1.0 if dist >= 0.0 else -1.0
            ra = (
                eA[0] * abs(float(a[0] @ L))
                + eA[1] * abs(float(a[1] @ L))
                + eA[2] * abs(float(a[2] @ L))
            )
            rb = (
                eB[0] * abs(float(b[0] @ L))
                + eB[1] * abs(float(b[1] @ L))
                + eB[2] * abs(float(b[2] @ L))
            )
            o = ra + rb - abs(dist)
            if o < -margin:
                return None
            # small bias so near-degenerate edge axes lose ties to face axes
            if o + 1e-4 < best:
                best = o
                kind = 2
                axis_i = i
                axis_j = j
                edge_normal = s * L
    if kind == 0:
        n = sign * a[axis_i]
        points = _clip_face_points(cA, RA, eA, axis_i, n, cB, RB, eB, margin)
    elif kind == 1:
        n = sign * b[axis_j]  # A -> B
        ref_n = -n  # outward normal of B's reference face (points toward A)
        points = _clip_face_points(cB, RB, eB, axis_j, ref_n, cA, RA, eA, margin)
    else:
        n = edge_normal
        pA = cA.copy()
        for k in range(3):
            if k != axis_i:
                pA = pA + a[k] * (eA[k] * (1.0 if float(a[k] @ n) >= 0.0 else -1.0))
        pB = cB.copy()
        for k in range(3):
            if k != axis_j:
                pB = pB + b[k] * (eB[k] * (1.0 if float(b[k] @ n) <= 0.0 else -1.0))
        u = a[axis_i]
        v = b[axis_j]
        r = pB - pA
        bq = float(u @ v)
        dq = float(u @ r)
        eq = float(v @ r)
        den = 1.0 - bq * bq
        if abs(den) < 1e-9:
            s_param = 0.0
            t_param = eq
        else:
            s_param = (bq * eq - dq) / den
            t_param = (eq - bq * dq) / den
        point = 0.5 * (pA + s_param * u + pB + t_param * v)
        points = [(point, best)]
    if not points:
        points = [(0.5 * (cA + cB), best)]
    return n, points


# ---------------------------------------------------------------------------
# rigid bodies and contacts
# ---------------------------------------------------------------------------
class _Body:
    __slots__ = (
        "name",
        "dynamic",
        "mass",
        "inv_mass",
        "pos",
        "quat",
        "vel",
        "w",
        "pvel",
        "pw",
        "R",
        "com_local",
        "com",
        "i_local",
        "inv_i_local",
        "i_world",
        "inv_i_world",
        "boxes",
        "aabb_lo",
        "aabb_hi",
        "driven",
    )

    def __init__(self, name, dynamic, pos, quat, boxes, mass=None, com_local=None,
                 inertia=None):
        self.name = name
        self.dynamic = dynamic
        self.pos = np.asarray(pos, dtype=float).copy()
        self.quat = _quat_normalize(np.asarray(quat, dtype=float))
        self.boxes = boxes
        self.R = _quat_to_matrix(self.quat)
        self.vel = np.zeros(3)
        self.w = np.zeros(3)
        self.pvel = np.zeros(3)
        self.pw = np.zeros(3)
        self.driven = False
        if dynamic:
            self.mass = float(mass)
            self.inv_mass = 1.0 / self.mass
            self.com_local = com_local
            self.i_local = inertia
            self.inv_i_local = np.linalg.inv(inertia)
        else:
            self.mass = 0.0
            self.inv_mass = 0.0
            self.com_local = np.zeros(3)
            self.i_local = np.zeros((3, 3))
            self.inv_i_local = np.zeros((3, 3))
        self.com = self.pos + self.R @ self.com_local
        self.i_world = self.R @ self.i_local @ self.R.T
        self.inv_i_world = self.R @ self.inv_i_local @ self.R.T
        self.aabb_lo = np.zeros(3)
        self.aabb_hi = np.zeros(3)


class _Contact:
    __slots__ = (
        "ba", "bb", "ra", "rb", "n", "t1", "t2",
        "kn", "kt1", "kt2", "pen", "bounce",
        "acc_n", "acc_t1", "acc_t2", "acc_pn",
    )

    def __init__(self, ba: _Body, bb: _Body, point, normal, pen):
        self.ba = ba
        self.bb = bb
        self.ra = np.asarray(point, dtype=float) - ba.com
        self.rb = np.asarray(point, dtype=float) - bb.com
        self.n = normal
        helper = np.zeros(3)
        helper[int(np.argmin(np.abs(normal)))] = 1.0
        t1 = np.cross(normal, helper)
        t1 /= max(float(np.linalg.norm(t1)), 1e-12)
        self.t1 = t1
        self.t2 = np.cross(normal, t1)
        self.pen = float(pen)
        self.bounce = 0.0
        self.acc_n = 0.0
        self.acc_t1 = 0.0
        self.acc_t2 = 0.0
        self.acc_pn = 0.0
        rn_a = np.cross(self.ra, normal)
        rn_b = np.cross(self.rb, normal)
        self.kn = (
            ba.inv_mass
            + bb.inv_mass
            + float((ba.inv_i_world @ rn_a) @ rn_a)
            + float((bb.inv_i_world @ rn_b) @ rn_b)
        )
        rt1_a = np.cross(self.ra, self.t1)
        rt1_b = np.cross(self.rb, self.t1)
        self.kt1 = (
            ba.inv_mass
            + bb.inv_mass
            + float((ba.inv_i_world @ rt1_a) @ rt1_a)
            + float((bb.inv_i_world @ rt1_b) @ rt1_b)
        )
        rt2_a = np.cross(self.ra, self.t2)
        rt2_b = np.cross(self.rb, self.t2)
        self.kt2 = (
            ba.inv_mass
            + bb.inv_mass
            + float((ba.inv_i_world @ rt2_a) @ rt2_a)
            + float((bb.inv_i_world @ rt2_b) @ rt2_b)
        )
        for value in (self.kn, self.kt1, self.kt2):
            if not np.isfinite(value) or value <= 1e-9:
                raise ValueError("degenerate contact effective mass")

    def total_impulse(self):
        return self.acc_n * self.n + self.acc_t1 * self.t1 + self.acc_t2 * self.t2


class _MechanismSim:
    """Impulse-based frictional rigid-body simulation of the mechanism."""

    def __init__(self, gravity, friction, restitution, length_scale,
                 substep=0.001, vel_iters=14, pos_iters=6):
        self.gravity = np.asarray(gravity, dtype=float)
        self.mu = float(friction)
        self.restitution = float(restitution)
        self.length_scale = float(length_scale)
        self.substep = float(substep)
        self.vel_iters = int(vel_iters)
        self.pos_iters = int(pos_iters)
        self.rest_thresh = 0.15
        self.margin = 0.008
        self.beta = 0.2
        self.slop = 0.002
        self.max_correction = 0.6
        self.lin_damp = 0.02
        self.ang_damp = 0.10
        # driver: ramped horizontal load on the leftmost link
        self.drive_magnitude = 28.0
        self.drive_start = 0.4
        self.drive_ramp = 1.2
        self.bodies: list[_Body] = []
        self.contacts: list[_Contact] = []
        self.last_gaps = np.zeros(0)
        self.last_forces = np.zeros((0, 3))
        self.steps = 0

    # -- setup ------------------------------------------------------------
    def add_body(self, body: _Body) -> None:
        self.bodies.append(body)

    def drive_force(self, t: float) -> float:
        if t <= self.drive_start:
            return 0.0
        ramp = (t - self.drive_start) / max(self.drive_ramp, 1e-9)
        return self.drive_magnitude * min(max(ramp, 0.0), 1.0)

    # -- collision detection ----------------------------------------------
    def _update_derived(self) -> None:
        for body in self.bodies:
            if body.dynamic:
                body.R = _quat_to_matrix(body.quat)
                body.com = body.pos + body.R @ body.com_local
                body.i_world = body.R @ body.i_local @ body.R.T
                body.inv_i_world = body.R @ body.inv_i_local @ body.R.T
        for body in self.bodies:
            lo = np.full(3, np.inf)
            hi = np.full(3, -np.inf)
            abs_R = np.abs(body.R)
            for center_local, half in body.boxes:
                center = body.pos + body.R @ center_local
                ext = abs_R @ half
                lo = np.minimum(lo, center - ext)
                hi = np.maximum(hi, center + ext)
            body.aabb_lo = lo
            body.aabb_hi = hi

    def _detect(self) -> None:
        self._update_derived()
        contacts: list[_Contact] = []
        bodies = self.bodies
        count = len(bodies)
        margin = self.margin
        pad = margin + 0.004
        for i in range(count):
            bi = bodies[i]
            for j in range(i + 1, count):
                bj = bodies[j]
                if not bi.dynamic and not bj.dynamic:
                    continue
                if np.any(bi.aabb_hi + pad < bj.aabb_lo) or np.any(
                    bj.aabb_hi + pad < bi.aabb_lo
                ):
                    continue
                for ci, ei in bi.boxes:
                    center_a = bi.pos + bi.R @ ci
                    for cj, ej in bj.boxes:
                        center_b = bj.pos + bj.R @ cj
                        result = _collide_boxes(
                            center_a, bi.R, ei, center_b, bj.R, ej, margin
                        )
                        if result is None:
                            continue
                        normal, points = result
                        for point, depth in points:
                            contacts.append(_Contact(bi, bj, point, normal, depth))
        self.contacts = contacts

    def refresh_contacts(self) -> None:
        """Detect contacts without solving (used for the t = 0 frame)."""
        self._detect()
        self.last_gaps = np.array([-c.pen for c in self.contacts], dtype=float)
        self.last_forces = np.zeros((len(self.contacts), 3))

    # -- solving ----------------------------------------------------------
    def _apply_impulse(self, c: _Contact, impulse, pseudo: bool):
        ba, bb = c.ba, c.bb
        if pseudo:
            if ba.dynamic:
                ba.pvel -= ba.inv_mass * impulse
                ba.pw -= ba.inv_i_world @ np.cross(c.ra, impulse)
            if bb.dynamic:
                bb.pvel += bb.inv_mass * impulse
                bb.pw += bb.inv_i_world @ np.cross(c.rb, impulse)
        else:
            if ba.dynamic:
                ba.vel -= ba.inv_mass * impulse
                ba.w -= ba.inv_i_world @ np.cross(c.ra, impulse)
            if bb.dynamic:
                bb.vel += bb.inv_mass * impulse
                bb.w += bb.inv_i_world @ np.cross(c.rb, impulse)

    def _relative_velocity(self, c: _Contact, pseudo: bool):
        ba, bb = c.ba, c.bb
        if pseudo:
            va = ba.pvel + np.cross(ba.pw, c.ra)
            vb = bb.pvel + np.cross(bb.pw, c.rb)
        else:
            va = ba.vel + np.cross(ba.w, c.ra)
            vb = bb.vel + np.cross(bb.w, c.rb)
        return vb - va

    def _solve_velocity(self) -> None:
        mu = self.mu
        for _ in range(self.vel_iters):
            for c in self.contacts:
                rel = self._relative_velocity(c, False)
                vn = float(rel @ c.n)
                target = vn + c.bounce
                if target < 0.0:
                    j = -target / c.kn
                    new_acc = c.acc_n + j
                    if new_acc < 0.0:
                        new_acc = 0.0
                    j = new_acc - c.acc_n
                    c.acc_n = new_acc
                    if j != 0.0:
                        self._apply_impulse(c, j * c.n, False)
                limit = mu * c.acc_n
                rel = self._relative_velocity(c, False)
                vt1 = float(rel @ c.t1)
                jt = -vt1 / c.kt1
                new_t = c.acc_t1 + jt
                if new_t > limit:
                    new_t = limit
                elif new_t < -limit:
                    new_t = -limit
                jt = new_t - c.acc_t1
                c.acc_t1 = new_t
                if jt != 0.0:
                    self._apply_impulse(c, jt * c.t1, False)
                vt2 = float(rel @ c.t2)
                jt = -vt2 / c.kt2
                new_t = c.acc_t2 + jt
                if new_t > limit:
                    new_t = limit
                elif new_t < -limit:
                    new_t = -limit
                jt = new_t - c.acc_t2
                c.acc_t2 = new_t
                if jt != 0.0:
                    self._apply_impulse(c, jt * c.t2, False)

    def _solve_position(self, h: float) -> None:
        beta, slop, max_corr = self.beta, self.slop, self.max_correction
        for c in self.contacts:
            c.acc_pn = 0.0
        for _ in range(self.pos_iters):
            for c in self.contacts:
                pen = c.pen - slop
                bias = 0.0
                if pen > 0.0:
                    bias = beta * pen / h
                    if bias > max_corr:
                        bias = max_corr
                rel = self._relative_velocity(c, True)
                vpn = float(rel @ c.n)
                j = (bias - vpn) / c.kn
                new_acc = c.acc_pn + j
                if new_acc < 0.0:
                    new_acc = 0.0
                j = new_acc - c.acc_pn
                c.acc_pn = new_acc
                if j != 0.0:
                    self._apply_impulse(c, j * c.n, True)

    def _integrate(self, h: float) -> None:
        for body in self.bodies:
            if not body.dynamic:
                body.pvel = np.zeros(3)
                body.pw = np.zeros(3)
                continue
            r = body.com - body.pos
            v_origin = body.vel - np.cross(body.w, r)
            p_origin = body.pvel - np.cross(body.pw, r)
            body.pos = body.pos + h * (v_origin + p_origin)
            body.quat = _quat_integrate(body.quat, body.w + body.pw, h)
            body.vel *= max(0.0, 1.0 - self.lin_damp * h)
            body.w *= max(0.0, 1.0 - self.ang_damp * h)
            body.pvel = np.zeros(3)
            body.pw = np.zeros(3)

    def step(self, h: float, t: float) -> None:
        bodies = self.bodies
        drive = self.drive_force(t)
        for body in bodies:
            if not body.dynamic:
                continue
            force = self.gravity * body.mass
            if body.driven:
                force = force + np.array([drive, 0.0, 0.0])
            body.vel += h * body.inv_mass * force
            angular_momentum = body.i_world @ body.w
            body.w += h * (body.inv_i_world @ (-np.cross(body.w, angular_momentum)))
        self._detect()
        if self.contacts:
            for c in self.contacts:
                vn0 = float(self._relative_velocity(c, False) @ c.n)
                if vn0 < -self.rest_thresh:
                    c.bounce = -self.restitution * vn0
                else:
                    c.bounce = 0.0
            self._solve_velocity()
            self._solve_position(h)
        self._integrate(h)
        self.last_gaps = np.array([-c.pen for c in self.contacts], dtype=float)
        if self.contacts:
            scale = 1.0 / h
            self.last_forces = np.array(
                [c.total_impulse() * scale for c in self.contacts], dtype=float
            )
        else:
            self.last_forces = np.zeros((0, 3))
        self.steps += 1


# ---------------------------------------------------------------------------
# benchmark solver
# ---------------------------------------------------------------------------
class BenchmarkSolver:
    """Solver for scene 143 (frictional mechanism jamming)."""

    def __init__(self):
        self._sim: _MechanismSim | None = None
        self._t = 0.0
        self._substep = 0.001
        self._length_scale = 0.02
        self._gravity = np.array([0.0, 0.0, -9.81])
        self._masses = None
        self._emitted = 0

    # -- setup ------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", None)
        if not isinstance(config, dict):
            config = {}
        gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        contact = config.get("contact", {}) or {}
        friction = float(contact.get("friction", 0.55))
        restitution = float(contact.get("restitution", 0.12))
        length_scale = float(contact.get("length_scale", 0.02))

        initial = scene.initial_state()
        positions = np.asarray(initial.position, dtype=float)
        orientations = np.asarray(initial.orientation, dtype=float)
        masses = np.asarray(initial.mass, dtype=float)

        # link mesh geometry (shared by all dynamic links)
        asset = None
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []) or []:
            if obj.get("motion") == "dynamic":
                asset = obj.get("geometry", {}).get("asset")
                break
        collision_boxes, mass_boxes = _load_link_boxes(asset)

        sim = _MechanismSim(
            gravity, friction, restitution, length_scale, substep=self._substep
        )
        link_bodies = []
        for index, spec in enumerate(config.get("bodies", [])):
            com, inertia = _composite_mass_properties(mass_boxes, float(masses[index]))
            body = _Body(
                spec.get("id", f"link_{index}"),
                True,
                positions[index],
                orientations[index],
                collision_boxes,
                mass=float(masses[index]),
                com_local=com,
                inertia=inertia,
            )
            if index == 0:
                body.driven = True
            sim.add_body(body)
            link_bodies.append(body)
        # confining static geometry from the frozen scene geometry
        for obj in geometry.get("objects", []) or []:
            if obj.get("motion") != "static":
                continue
            geom = obj.get("geometry", {})
            if geom.get("type") != "box":
                continue
            half = np.asarray(geom.get("half_extents", [1.0, 1.0, 1.0]), dtype=float)
            transform = obj.get("transform", {}) or {}
            translation = np.asarray(
                transform.get("translation", [0.0, 0.0, 0.0]), dtype=float
            )
            rotation = np.asarray(
                transform.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float
            )
            sim.add_body(
                _Body(
                    obj.get("id", "static"),
                    False,
                    translation,
                    rotation,
                    [(np.zeros(3), half)],
                )
            )
        self._sim = sim
        self._t = 0.0
        self._gravity = gravity
        self._length_scale = length_scale
        self._masses = masses.copy()
        self._link_count = len(link_bodies)
        self._seed = int(seed)
        sim.refresh_contacts()

    # -- stepping ---------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if self._sim is None:
            raise RuntimeError("initialize() must be called before advance_to()")
        if target < self._t - 1e-12:
            target = self._t
        while self._t < target - 1e-12:
            h = min(self._sim.substep, target - self._t)
            self._sim.step(h, self._t + 0.5 * h)
            self._t += h
            if target - self._t < 1e-12:
                self._t = target
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        sim = self._sim
        bodies = sim.bodies[: self._link_count]
        n = len(bodies)
        position = np.zeros((n, 3))
        velocity = np.zeros((n, 3))
        orientation = np.zeros((n, 4))
        angular = np.zeros((n, 3))
        external = np.zeros((n, 3))
        drive = sim.drive_force(time)
        for i, body in enumerate(bodies):
            # origin->COM offset from the current orientation (body.com may be
            # one substep stale relative to body.pos)
            r = _quat_to_matrix(body.quat) @ body.com_local
            position[i] = body.pos
            velocity[i] = body.vel - np.cross(body.w, r)
            orientation[i] = _quat_normalize(body.quat)
            angular[i] = body.w
            external[i] = sim.gravity * body.mass
            if body.driven:
                external[i] += np.array([drive, 0.0, 0.0])
        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular,
            mass=np.asarray(self._masses, dtype=float),
            body_id=np.arange(n, dtype=np.int32),
            attributes={"release_time": np.zeros(n)},
            fields={
                "contact_signed_distance": np.asarray(sim.last_gaps, dtype=float),
                "external_force": external,
                "interaction_force": np.asarray(sim.last_forces, dtype=float),
            },
            scalars={"contact_length_scale": float(self._length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        sim = self._sim
        if sim is None:
            return {"success": False, "error": "not initialized"}
        gaps = np.asarray(sim.last_gaps, dtype=float)
        return {
            "success": True,
            "method": "impulse-based frictional rigid-body dynamics "
            "(SAT box manifolds, projected Gauss-Seidel, split impulses)",
            "substeps": int(sim.steps),
            "final_time": float(self._t),
            "contact_count": int(len(sim.contacts)),
            "min_contact_gap": float(gaps.min()) if gaps.size else 0.0,
            "max_speed": float(
                max(
                    (float(np.linalg.norm(b.vel)) for b in sim.bodies if b.dynamic),
                    default=0.0,
                )
            ),
        }
