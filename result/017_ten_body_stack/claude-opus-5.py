"""Ten-body rigid box stack: self-contained rigid-body contact dynamics.

Physical model
--------------
Ten rigid boxes (0.9 x 0.9 x 0.2 m) are released with small gaps above a
static ground box.  Each body carries full 6-DOF rigid dynamics:

  * semi-implicit (symplectic) Euler integration of linear and angular state,
  * face-axis SAT collision detection between oriented boxes with
    Sutherland-Hodgman face clipping to build contact manifolds,
  * velocity-level sequential-impulse (projected Gauss-Seidel) contact solver
    with speculative contacts, Baumgarte penetration recovery, zero
    restitution and Coulomb friction,
  * warm starting of normal impulses so load transfers through the whole
    stack within a single time step.

Nothing about the motion is scripted: the settling, the successive load
transfer and the oscillation decay all emerge from the contact solve.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

SUBSTEPS = 16          # substeps per sample interval (dt = sample_dt / SUBSTEPS)
ITERATIONS = 24        # projected Gauss-Seidel sweeps per substep
CONTACT_MARGIN = 0.02  # speculative-contact detection band [m]
PENETRATION_SLOP = 2.0e-4
BAUMGARTE = 0.30
MAX_RECOVERY = 0.8     # cap on penetration recovery speed [m/s]
FRICTION = 0.6
MAX_POINTS = 4         # contact points kept per manifold
MAX_SPEED = 60.0
MAX_OMEGA = 60.0


def _quat_matrix(q) -> np.ndarray:
    """Rotation matrix of an xyzw quaternion (columns are body axes)."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _clip_half(poly, axis: int, sign: float, limit: float):
    """Keep the part of a 2D polygon with sign * p[axis] <= limit."""
    out = []
    count = len(poly)
    for i in range(count):
        a = poly[i]
        b = poly[(i + 1) % count]
        da = sign * a[axis] - limit
        db = sign * b[axis] - limit
        if da <= 0.0:
            out.append(a)
        if (da > 0.0) != (db > 0.0):
            t = da / (da - db)
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def _clip_rect(poly, hu: float, hv: float):
    poly = _clip_half(poly, 0, 1.0, hu)
    if not poly:
        return poly
    poly = _clip_half(poly, 0, -1.0, hu)
    if not poly:
        return poly
    poly = _clip_half(poly, 1, 1.0, hv)
    if not poly:
        return poly
    return _clip_half(poly, 1, -1.0, hv)

def _box_separation(ca, ra, ha, cb, rb, hb):
    """Face-axis SAT.  Returns (separation, normal, reference_is_a).

    ``separation`` is the largest gap over the six face axes (negative means
    penetration along that axis); ``normal`` points from A towards B.
    """
    best = -1.0e30
    best_axis = np.array([0.0, 0.0, 1.0])
    best_is_a = True
    d = cb - ca
    for is_a in (True, False):
        rot = ra if is_a else rb
        hh = ha if is_a else hb
        oth_rot = rb if is_a else ra
        oth_h = hb if is_a else ha
        for k in range(3):
            axis = rot[:, k]
            proj = (
                abs(float(axis @ oth_rot[:, 0])) * oth_h[0]
                + abs(float(axis @ oth_rot[:, 1])) * oth_h[1]
                + abs(float(axis @ oth_rot[:, 2])) * oth_h[2]
            )
            along = float(axis @ d)
            dist = abs(along) - (hh[k] + proj)
            if dist > best:
                best = dist
                sgn = 1.0 if along >= 0.0 else -1.0
                best_axis = axis * sgn if is_a else -(axis * sgn)
                best_is_a = is_a
    return best, best_axis, best_is_a


def _manifold(ca, ra, ha, cb, rb, hb, margin):
    """Contact manifold between two oriented boxes.

    Returns ``(normal, points, separations)`` with the normal pointing from A
    to B, or ``None`` when the boxes are farther apart than ``margin``.
    """
    sep, normal, ref_is_a = _box_separation(ca, ra, ha, cb, rb, hb)
    if sep > margin:
        return None

    if ref_is_a:
        ref_c, ref_r, ref_h = ca, ra, ha
        inc_c, inc_r, inc_h = cb, rb, hb
        ref_axis = normal
    else:
        ref_c, ref_r, ref_h = cb, rb, hb
        inc_c, inc_r, inc_h = ca, ra, ha
        ref_axis = -normal

    k_ref = int(np.argmax(np.abs(ref_r.T @ ref_axis)))
    sgn = 1.0 if float(ref_r[:, k_ref] @ ref_axis) >= 0.0 else -1.0
    n_ref = ref_r[:, k_ref] * sgn
    face_c = ref_c + n_ref * float(ref_h[k_ref])
    iu, iv = (k_ref + 1) % 3, (k_ref + 2) % 3
    u = ref_r[:, iu]
    v = ref_r[:, iv]
    hu, hv = float(ref_h[iu]), float(ref_h[iv])

    dots = inc_r.T @ n_ref
    k_inc = int(np.argmax(np.abs(dots)))
    isg = -1.0 if float(dots[k_inc]) >= 0.0 else 1.0
    n_inc = inc_r[:, k_inc] * isg
    inc_face_c = inc_c + n_inc * float(inc_h[k_inc])
    ju, jv = (k_inc + 1) % 3, (k_inc + 2) % 3
    eu = inc_r[:, ju] * float(inc_h[ju])
    ev = inc_r[:, jv] * float(inc_h[jv])
    quad = (
        inc_face_c + eu + ev,
        inc_face_c + eu - ev,
        inc_face_c - eu - ev,
        inc_face_c - eu + ev,
    )

    poly2 = []
    for p in quad:
        rel = p - face_c
        poly2.append((float(rel @ u), float(rel @ v)))
    clipped = _clip_rect(poly2, hu, hv)
    if not clipped:
        return None

    denom = float(n_inc @ n_ref)
    points = []
    seps = []
    for (a, b) in clipped:
        base = face_c + u * a + v * b
        rel = base - inc_face_c
        t = -float(rel @ n_inc) / denom if abs(denom) > 1e-9 else 0.0
        p_inc = base + n_ref * t
        gap = float((p_inc - face_c) @ n_ref)
        if gap > margin:
            continue
        points.append(0.5 * (base + p_inc))
        seps.append(gap)
    if not points:
        return None
    if len(points) > MAX_POINTS:
        order = np.argsort(np.asarray(seps))[:MAX_POINTS]
        points = [points[i] for i in order]
        seps = [seps[i] for i in order]
    return normal, points, seps


class BenchmarkSolver:
    """Rigid-body contact solver for the ten-body stack."""

    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", {}) or {})
        frame0 = scene.initial_state()

        self.gravity = np.asarray(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        ).reshape(3)
        self.sample_dt = float(config.get("sample_dt", 0.02))
        self.dt = self.sample_dt / SUBSTEPS

        geometry = config.get("scene_geometry") or {}
        objects = list(geometry.get("objects", []))
        by_body: dict[int, dict] = {}
        ground = None
        for item in objects:
            binding = item.get("state_binding") or {}
            if binding.get("selector") == "body_id":
                by_body[int(binding["value"])] = item
            elif item.get("semantic") == "boundary" or item.get("motion") == "static":
                if ground is None:
                    ground = item

        self.body_id = np.asarray(frame0.body_id, dtype=np.int64).copy()
        self.n = int(self.body_id.size)
        self.mass_arr = (
            np.asarray(frame0.mass, dtype=float).copy()
            if frame0.mass is not None
            else np.ones(self.n)
        )

        # index 0 is the static ground, dynamic bodies follow in body_id order
        half = [np.array([4.0, 4.0, 0.05])]
        centers = [np.array([0.0, 0.0, -0.05])]
        if ground is not None:
            he = ground.get("geometry", {}).get("half_extents")
            if he is not None:
                half[0] = np.asarray(he, dtype=float).reshape(3)
            tr = (ground.get("transform") or {}).get("translation")
            if tr is not None:
                centers[0] = np.asarray(tr, dtype=float).reshape(3)

        pos0 = np.asarray(frame0.position, dtype=float).reshape(self.n, 3)
        default_half = np.array([0.45, 0.45, 0.1])
        for i in range(self.n):
            item = by_body.get(int(self.body_id[i]), {})
            he = item.get("geometry", {}).get("half_extents")
            half.append(
                np.asarray(he, dtype=float).reshape(3) if he is not None else default_half.copy()
            )
            centers.append(pos0[i].copy())

        self.nb = self.n + 1
        self.half = half
        self.pos = np.asarray(centers, dtype=float)
        self.quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.nb, 1))
        if frame0.orientation is not None:
            q0 = np.asarray(frame0.orientation, dtype=float).reshape(self.n, 4)
            for i in range(self.n):
                nrm = float(np.linalg.norm(q0[i]))
                if nrm > 1e-9:
                    self.quat[i + 1] = q0[i] / nrm

        self.vel = np.zeros((self.nb, 3))
        self.omega = np.zeros((self.nb, 3))
        if frame0.velocity is not None:
            self.vel[1:] = np.asarray(frame0.velocity, dtype=float).reshape(self.n, 3)
        if frame0.angular_velocity is not None:
            self.omega[1:] = np.asarray(frame0.angular_velocity, dtype=float).reshape(self.n, 3)

        self.inv_mass = np.zeros(self.nb)
        self.inv_inertia_body = np.zeros((self.nb, 3))
        for i in range(1, self.nb):
            m = float(self.mass_arr[i - 1])
            h = self.half[i]
            if m <= 0.0:
                continue
            self.inv_mass[i] = 1.0 / m
            ex, ey, ez = 2.0 * h[0], 2.0 * h[1], 2.0 * h[2]
            ix = m * (ey * ey + ez * ez) / 12.0
            iy = m * (ex * ex + ez * ez) / 12.0
            iz = m * (ex * ex + ey * ey) / 12.0
            self.inv_inertia_body[i] = [1.0 / ix, 1.0 / iy, 1.0 / iz]

        # canonical load path: ground->body_01 then body_k->body_{k+1}
        self.interfaces = [(0, 1)] + [(i, i + 1) for i in range(1, self.n)]
        self.interface_index = {pair: k for k, pair in enumerate(self.interfaces)}
        self.n_interfaces = len(self.interfaces)

        # AABB broad-phase candidates: all box pairs (ground + bodies)
        self.all_pairs = [
            (a, b) for a in range(self.nb) for b in range(a + 1, self.nb)
            if not (a == 0 and b == 0)
        ]

        self.contact_length_scale = float(2.0 * default_half[2])
        if self.n > 0:
            self.contact_length_scale = float(2.0 * self.half[1][2])

        self.warm: dict[tuple[int, int], list[float]] = {}
        self.step_index = 0
        self.time = 0.0
        self.max_penetration = 0.0
        self.total_steps = 0
        self.interface_impulse = np.zeros((self.n_interfaces, 3))
        self.window_time = 0.0

    # ------------------------------------------------------------------
    # collision detection
    # ------------------------------------------------------------------
    def _rotations(self):
        return [_quat_matrix(self.quat[i]) for i in range(self.nb)]

    def _world_inv_inertia(self, rots):
        out = []
        for i in range(self.nb):
            d = self.inv_inertia_body[i]
            if d[0] == 0.0 and d[1] == 0.0 and d[2] == 0.0:
                out.append(np.zeros((3, 3)))
            else:
                r = rots[i]
                out.append((r * d) @ r.T)
        return out

    def _collect_contacts(self, rots):
        """Build the contact list for the current configuration."""
        contacts = []
        for (a, b) in self.all_pairs:
            if self.inv_mass[a] == 0.0 and self.inv_mass[b] == 0.0:
                continue
            delta = self.pos[b] - self.pos[a]
            ea = np.abs(rots[a]) @ self.half[a]
            eb = np.abs(rots[b]) @ self.half[b]
            if np.any(np.abs(delta) > ea + eb + CONTACT_MARGIN):
                continue
            result = _manifold(
                self.pos[a], rots[a], self.half[a],
                self.pos[b], rots[b], self.half[b],
                CONTACT_MARGIN,
            )
            if result is None:
                continue
            normal, points, seps = result
            contacts.append((a, b, normal, points, seps))
        # bottom-up ordering propagates the load through the stack per sweep
        contacts.sort(key=lambda c: min(float(p[2]) for p in c[3]))
        return contacts

    def _prepare(self, contacts, inv_i, dt):
        """Flatten contacts into solver rows with precomputed constraint data.

        Row layout::

            0 a          1 b          2 ra         3 rb
            4 n          5 t1         6 t2
            7 inv_kn     8 inv_k1     9 inv_k2
            10 target_n  11 pn        12 p1        13 p2
            14 key       15 idx       16 sep
            17 an 18 bn  19 a1 20 b1  21 a2 22 b2   (angular response vectors)
        """
        rows = []
        up = np.array([0.0, 0.0, 1.0])
        ex = np.array([1.0, 0.0, 0.0])
        for (a, b, normal, points, seps) in contacts:
            n = normal
            t1 = np.cross(n, up if abs(float(n[2])) < 0.9 else ex)
            nt = float(np.linalg.norm(t1))
            t1 = t1 / nt if nt > 1e-12 else ex.copy()
            t2 = np.cross(n, t1)
            key = (a, b)
            cached = self.warm.get(key)
            iia, iib = inv_i[a], inv_i[b]
            ima, imb = float(self.inv_mass[a]), float(self.inv_mass[b])
            for idx, point in enumerate(points):
                ra = point - self.pos[a]
                rb = point - self.pos[b]
                sep = float(seps[idx])
                data = []
                ok = True
                for di, d in enumerate((n, t1, t2)):
                    ca = np.cross(ra, d)
                    cb = np.cross(rb, d)
                    wa = iia @ ca
                    wb = iib @ cb
                    k = ima + imb + float(ca @ wa) + float(cb @ wb)
                    data.append(
                        (
                            (float(wa[0]), float(wa[1]), float(wa[2])),
                            (float(wb[0]), float(wb[1]), float(wb[2])),
                            1.0 / k if k > 1e-12 else 0.0,
                        )
                    )
                    if di == 0 and k <= 1e-12:
                        ok = False
                if not ok:
                    continue
                if sep < 0.0:
                    target = min(
                        MAX_RECOVERY,
                        BAUMGARTE * max(0.0, -sep - PENETRATION_SLOP) / dt,
                    )
                else:
                    # speculative contact: permit closing exactly the open gap
                    target = -sep / dt
                warm_n = 0.0
                if cached is not None and idx < len(cached) and sep <= 0.0:
                    warm_n = float(cached[idx])
                rows.append(
                    [int(a), int(b),
                     (float(ra[0]), float(ra[1]), float(ra[2])),
                     (float(rb[0]), float(rb[1]), float(rb[2])),
                     (float(n[0]), float(n[1]), float(n[2])),
                     (float(t1[0]), float(t1[1]), float(t1[2])),
                     (float(t2[0]), float(t2[1]), float(t2[2])),
                     data[0][2], data[1][2], data[2][2],
                     float(target), warm_n, 0.0, 0.0, key, idx, sep,
                     data[0][0], data[0][1],
                     data[1][0], data[1][1],
                     data[2][0], data[2][1]]
                )
        return rows

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        vel, om, pos = self.vel, self.omega, self.pos
        im = self.inv_mass
        dynamic = im > 0.0

        # gravity (external force / mass)
        vel[dynamic] += self.gravity * dt

        rots = self._rotations()
        inv_i = self._world_inv_inertia(rots)
        contacts = self._collect_contacts(rots)
        rows = self._prepare(contacts, inv_i, dt)

        # Scalar working state: the Gauss-Seidel sweeps touch a few million
        # rows, so plain Python floats beat per-row numpy temporaries.
        vx = [float(v) for v in vel[:, 0]]
        vy = [float(v) for v in vel[:, 1]]
        vz = [float(v) for v in vel[:, 2]]
        wx = [float(v) for v in om[:, 0]]
        wy = [float(v) for v in om[:, 1]]
        wz = [float(v) for v in om[:, 2]]
        ims = [float(v) for v in im]

        # warm start
        for row in rows:
            pn = row[11]
            if pn == 0.0:
                continue
            a = row[0]
            b = row[1]
            nx, ny, nz = row[4]
            an = row[17]
            bn = row[18]
            ia = ims[a] * pn
            ib = ims[b] * pn
            vx[a] -= ia * nx
            vy[a] -= ia * ny
            vz[a] -= ia * nz
            vx[b] += ib * nx
            vy[b] += ib * ny
            vz[b] += ib * nz
            wx[a] -= pn * an[0]
            wy[a] -= pn * an[1]
            wz[a] -= pn * an[2]
            wx[b] += pn * bn[0]
            wy[b] += pn * bn[1]
            wz[b] += pn * bn[2]

        for _ in range(ITERATIONS):
            for row in rows:
                a = row[0]
                b = row[1]
                rax, ray, raz = row[2]
                rbx, rby, rbz = row[3]
                ima = ims[a]
                imb = ims[b]

                wax, way, waz = wx[a], wy[a], wz[a]
                wbx, wby, wbz = wx[b], wy[b], wz[b]
                # relative velocity of the contact point on B w.r.t. A
                rvx = (vx[b] + wby * rbz - wbz * rby) - (vx[a] + way * raz - waz * ray)
                rvy = (vy[b] + wbz * rbx - wbx * rbz) - (vy[a] + waz * rax - wax * raz)
                rvz = (vz[b] + wbx * rby - wby * rbx) - (vz[a] + wax * ray - way * rax)

                nx, ny, nz = row[4]
                vn = rvx * nx + rvy * ny + rvz * nz
                pn_old = row[11]
                pn_new = pn_old + (row[10] - vn) * row[7]
                if pn_new < 0.0:
                    pn_new = 0.0
                row[11] = pn_new
                d_pn = pn_new - pn_old
                if d_pn != 0.0:
                    an = row[17]
                    bn = row[18]
                    ia = ima * d_pn
                    ib = imb * d_pn
                    vx[a] -= ia * nx
                    vy[a] -= ia * ny
                    vz[a] -= ia * nz
                    vx[b] += ib * nx
                    vy[b] += ib * ny
                    vz[b] += ib * nz
                    wx[a] -= d_pn * an[0]
                    wy[a] -= d_pn * an[1]
                    wz[a] -= d_pn * an[2]
                    wx[b] += d_pn * bn[0]
                    wy[b] += d_pn * bn[1]
                    wz[b] += d_pn * bn[2]

                limit = FRICTION * pn_new
                if limit <= 0.0:
                    row[12] = 0.0
                    row[13] = 0.0
                    continue
                for slot, di, wa_i, wb_i, k_i in ((12, 5, 19, 20, 8), (13, 6, 21, 22, 9)):
                    inv_k = row[k_i]
                    if inv_k == 0.0:
                        continue
                    tx, ty, tz = row[di]
                    wax, way, waz = wx[a], wy[a], wz[a]
                    wbx, wby, wbz = wx[b], wy[b], wz[b]
                    rvx = (vx[b] + wby * rbz - wbz * rby) - (vx[a] + way * raz - waz * ray)
                    rvy = (vy[b] + wbz * rbx - wbx * rbz) - (vy[a] + waz * rax - wax * raz)
                    rvz = (vz[b] + wbx * rby - wby * rbx) - (vz[a] + wax * ray - way * rax)
                    vt = rvx * tx + rvy * ty + rvz * tz
                    p_old = row[slot]
                    p_new = p_old - vt * inv_k
                    if p_new > limit:
                        p_new = limit
                    elif p_new < -limit:
                        p_new = -limit
                    row[slot] = p_new
                    d_pt = p_new - p_old
                    if d_pt != 0.0:
                        wa = row[wa_i]
                        wb = row[wb_i]
                        ia = ima * d_pt
                        ib = imb * d_pt
                        vx[a] -= ia * tx
                        vy[a] -= ia * ty
                        vz[a] -= ia * tz
                        vx[b] += ib * tx
                        vy[b] += ib * ty
                        vz[b] += ib * tz
                        wx[a] -= d_pt * wa[0]
                        wy[a] -= d_pt * wa[1]
                        wz[a] -= d_pt * wa[2]
                        wx[b] += d_pt * wb[0]
                        wy[b] += d_pt * wb[1]
                        wz[b] += d_pt * wb[2]

        vel[:, 0] = vx
        vel[:, 1] = vy
        vel[:, 2] = vz
        om[:, 0] = wx
        om[:, 1] = wy
        om[:, 2] = wz
        # static bodies never move
        vel[~dynamic] = 0.0
        om[~dynamic] = 0.0

        # cache normal impulses for warm starting and accumulate contact forces
        warm: dict[tuple[int, int], list[float]] = {}
        for row in rows:
            key, idx = row[14], row[15]
            slot = warm.setdefault(key, [])
            while len(slot) <= idx:
                slot.append(0.0)
            slot[idx] = row[11]
            k = self.interface_index.get(key)
            if k is not None:
                n = row[4]
                t1 = row[5]
                t2 = row[6]
                pn, p1, p2 = row[11], row[12], row[13]
                self.interface_impulse[k, 0] += n[0] * pn + t1[0] * p1 + t2[0] * p2
                self.interface_impulse[k, 1] += n[1] * pn + t1[1] * p1 + t2[1] * p2
                self.interface_impulse[k, 2] += n[2] * pn + t1[2] * p1 + t2[2] * p2
            if row[16] < 0.0:
                pen = -row[16]
                if pen > self.max_penetration:
                    self.max_penetration = pen
        self.warm = warm

        # velocity safety clamp (keeps the integration finite under any event)
        np.clip(vel, -MAX_SPEED, MAX_SPEED, out=vel)
        np.clip(om, -MAX_OMEGA, MAX_OMEGA, out=om)
        if not np.isfinite(vel).all():
            np.nan_to_num(vel, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(om).all():
            np.nan_to_num(om, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # integrate positions and orientations
        pos[dynamic] += vel[dynamic] * dt
        # q_dot = 0.5 * omega_quat * q  (xyzw convention)
        q = self.quat
        ox, oy, oz = om[:, 0], om[:, 1], om[:, 2]
        qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        dqx = 0.5 * (ox * qw + oy * qz - oz * qy)
        dqy = 0.5 * (-ox * qz + oy * qw + oz * qx)
        dqz = 0.5 * (ox * qy - oy * qx + oz * qw)
        dqw = 0.5 * (-ox * qx - oy * qy - oz * qz)
        new_q = np.column_stack((qx + dqx * dt, qy + dqy * dt, qz + dqz * dt, qw + dqw * dt))
        norms = np.linalg.norm(new_q, axis=1)
        norms[norms < 1e-12] = 1.0
        new_q = new_q / norms[:, None]
        q[dynamic] = new_q[dynamic]
        self.total_steps += 1

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def _interface_gaps(self) -> np.ndarray:
        """Signed geometric gap at each canonical stack interface."""
        rots = self._rotations()
        gaps = np.empty(self.n_interfaces)
        for k, (a, b) in enumerate(self.interfaces):
            sep, _n, _ref = _box_separation(
                self.pos[a], rots[a], self.half[a],
                self.pos[b], rots[b], self.half[b],
            )
            gaps[k] = sep
        return gaps

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        target_step = int(round(target / self.dt))
        if target_step < self.step_index:
            target_step = self.step_index

        self.interface_impulse[:] = 0.0
        steps = target_step - self.step_index
        for _ in range(steps):
            self._substep(self.dt)
            self.step_index += 1
        self.time = self.step_index * self.dt

        window = steps * self.dt
        if window > 0.0:
            interaction = self.interface_impulse / window
        else:
            interaction = np.zeros((self.n_interfaces, 3))

        gaps = self._interface_gaps()
        external = self.mass_arr[:, None] * self.gravity[None, :]

        position = np.array(self.pos[1:], dtype=float, copy=True)
        velocity = np.array(self.vel[1:], dtype=float, copy=True)
        orientation = np.array(self.quat[1:], dtype=float, copy=True)
        angular = np.array(self.omega[1:], dtype=float, copy=True)
        for arr in (position, velocity, orientation, angular):
            if not np.isfinite(arr).all():
                np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        min_gap = float(np.min(gaps))
        if -min_gap > self.max_penetration:
            self.max_penetration = -min_gap
        self.last_min_gap = min_gap
        self.last_top_height = float(position[-1, 2]) if self.n else 0.0
        self.last_speed = float(np.max(np.abs(velocity))) if self.n else 0.0

        frame = StateFrame(
            time=target,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular,
            mass=np.array(self.mass_arr, dtype=float, copy=True),
            body_id=np.array(self.body_id, dtype=np.int64, copy=True),
            attributes={},
            fields={
                "contact_signed_distance": np.ascontiguousarray(gaps, dtype=float),
                "external_force": np.ascontiguousarray(external, dtype=float),
                "interaction_force": np.ascontiguousarray(interaction, dtype=float),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        return frame

    def diagnostics(self) -> dict:
        return {
            "solver": "self_rigid_sequential_impulse",
            "substeps_per_sample": int(SUBSTEPS),
            "iterations_per_substep": int(ITERATIONS),
            "dt": float(self.dt),
            "total_steps": int(self.total_steps),
            "max_penetration_m": float(self.max_penetration),
            "final_min_gap_m": float(getattr(self, "last_min_gap", 0.0)),
            "final_top_height_m": float(getattr(self, "last_top_height", 0.0)),
            "final_max_speed_mps": float(getattr(self, "last_speed", 0.0)),
            "friction": float(FRICTION),
            "contact_length_scale_m": float(self.contact_length_scale),
        }

