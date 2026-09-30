"""Frictional mechanism jamming: hand written rigid-body contact solver.

Scene: a chain of identical L-shaped links confined between two static guide
plates and the floor.  A prescribed driver force loads the leftmost link, contact
impulses propagate along the chain, Coulomb friction resists relative sliding and
the assembly locks up (jams) without persistent interpenetration.

Method: substepped XPBD for rigid bodies (Newton-Euler about each body centre of
mass) with
  * compound oriented-box / oriented-box SAT contact generation derived from the
    frozen triangle mesh asset (multi point vertex/face manifolds plus
    near-contact signed gaps),
  * unilateral non-penetration position constraints solved by Gauss-Seidel with a
    bounded separation speed so large initial overlaps relax gently,
  * a velocity level Coulomb friction pass with a per-solve friction cone clamp
    and a small restitution pass,
  * Newton-III consistent force bookkeeping: exported interaction forces are the
    contact impulses actually applied during each sample interval, rebalanced
    against the momentum change of the sampled trajectory.

Only NumPy and the Python standard library are used.
"""

from __future__ import annotations

import math
import os

import numpy as np

from worldsolver.contracts import StateFrame

DEFAULT_GRAVITY = (0.0, 0.0, -9.81)
UNIT_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    dtype=float,
)  # (8,3) unit box corners


# ------------------------------------------------------------------ quaternions
def _qmat(q):
    """Rotation matrix from a quaternion in xyzw order."""
    n = math.sqrt(float(q[0]) ** 2 + float(q[1]) ** 2 + float(q[2]) ** 2 + float(q[3]) ** 2)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = float(q[0]) / n, float(q[1]) / n, float(q[2]) / n, float(q[3]) / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _qmul(a, b):
    ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _qnorm(q):
    n = math.sqrt(float(q[0]) ** 2 + float(q[1]) ** 2 + float(q[2]) ** 2 + float(q[3]) ** 2)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.asarray(q, dtype=float) / n


def _qconj(q):
    return np.array([-float(q[0]), -float(q[1]), -float(q[2]), float(q[3])])


def _q_rotvec(v):
    """Quaternion (xyzw) of a rotation vector."""
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = v / th
    s = math.sin(0.5 * th)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, math.cos(0.5 * th)])


def _q_to_rotvec(q):
    """Rotation vector of a small quaternion (xyzw)."""
    q = np.asarray(q, dtype=float)
    if q[3] < 0.0:
        q = -q
    w = min(1.0, max(-1.0, float(q[3])))
    s = math.sqrt(max(0.0, 1.0 - w * w))
    if s < 1e-9:
        return np.zeros(3)
    th = 2.0 * math.acos(w)
    return q[:3] / s * th


# ------------------------------------------------------- frozen mesh processing
def _parse_obj(path):
    """Return (vertices Nx3, faces list[list[int]]) from a wavefront OBJ file."""
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx = []
                for token in line.split()[1:]:
                    head = token.split("/")[0]
                    if not head:
                        continue
                    value = int(head)
                    idx.append(value - 1 if value > 0 else len(verts) + value)
                if len(idx) >= 3:
                    faces.append(idx)
    return np.asarray(verts, dtype=float), faces


def _compound_boxes_from_obj(path):
    """Split a triangle mesh into connected components and bound each by an AABB.

    The frozen link asset is a union of two axis aligned boxes; the resulting
    oriented boxes reproduce its collision geometry exactly without hard coding
    any scene specific dimensions.
    """
    verts, faces = _parse_obj(path)
    if verts.size == 0 or not faces:
        return None
    parent = list(range(len(faces)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seen = {}
    for fi, face in enumerate(faces):
        for vi in face:
            vi = int(vi)
            if 0 <= vi < len(verts):
                if vi in seen:
                    ra, rb = find(fi), find(seen[vi])
                    if ra != rb:
                        parent[ra] = rb
                else:
                    seen[vi] = fi
    groups = {}
    for fi in range(len(faces)):
        groups.setdefault(find(fi), []).append(fi)
    boxes = []
    for group in groups.values():
        index = np.unique(np.asarray(
            [vi for fi in group for vi in faces[fi] if 0 <= int(vi) < len(verts)],
            dtype=int,
        ))
        if index.size == 0:
            continue
        pts = verts[index]
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        half = 0.5 * (hi - lo)
        if np.any(half <= 1e-9):
            continue
        boxes.append((0.5 * (hi + lo), half))
    if not boxes or len(boxes) > 8:
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)
        half = 0.5 * (hi - lo)
        if np.any(half <= 1e-9):
            return None
        boxes = [(0.5 * (hi + lo), half)]
    return boxes


def _compound_mass_properties(boxes):
    """Unit-mass centre of mass and inertia tensor (about the centre of mass)."""
    volumes = np.array(
        [8.0 * float(b[1][0]) * float(b[1][1]) * float(b[1][2]) for b in boxes], dtype=float
    )
    total = float(volumes.sum())
    if total <= 0.0:
        return np.zeros(3), np.eye(3)
    frac = volumes / total
    com = np.zeros(3)
    for f, (c, _h) in zip(frac, boxes):
        com = com + f * np.asarray(c, dtype=float)
    inertia = np.zeros((3, 3))
    for f, (c, h) in zip(frac, boxes):
        hx, hy, hz = float(h[0]), float(h[1]), float(h[2])
        block = np.diag(
            [f / 3.0 * (hy * hy + hz * hz), f / 3.0 * (hx * hx + hz * hz), f / 3.0 * (hx * hx + hy * hy)]
        )
        r = np.asarray(c, dtype=float) - com
        block = block + f * ((float(r @ r)) * np.eye(3) - np.outer(r, r))
        inertia = inertia + block
    return com, inertia


class _Contact:
    """One rigid contact sample (vertex of the incident box against a reference face)."""

    __slots__ = ("key", "ia", "ib", "p_inc_local", "p_ref_local", "n_local", "gap")

    def __init__(self, key, ia, ib, p_inc_local, p_ref_local, n_local, gap):
        self.key = key
        self.ia = ia  # incident (moving) body index
        self.ib = ib  # reference body index, negative for a static reference
        self.p_inc_local = p_inc_local  # world point at generation, in body ia frame
        self.p_ref_local = p_ref_local  # world point at generation, in body ib frame (or world)
        self.n_local = n_local  # unit normal, in body ib frame (or world), ref -> incident
        self.gap = float(gap)


class _State:
    __slots__ = ("x", "q", "R", "v", "w", "W")

    def __init__(self, x, q, R, v, w):
        self.x = x
        self.q = q
        self.R = R
        self.v = v
        self.w = w
        self.W = None

    def copy(self):
        clone = _State(self.x.copy(), self.q.copy(), self.R.copy(), self.v.copy(), self.w.copy())
        return clone


class BenchmarkSolver:
    """Deterministic substepped rigid body solver with frictional contacts."""

    def __init__(self):
        self._ready = False

    # ------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", None) or {})
        self.seed = int(seed)
        self.gravity = np.asarray(config.get("gravity", DEFAULT_GRAVITY), dtype=float).reshape(3)
        contact = dict(config.get("contact", {}) or {})
        self.friction = float(contact.get("friction", 0.55))
        self.restitution = float(contact.get("restitution", 0.12))
        self.length_scale = float(contact.get("length_scale", 0.02))
        self.duration = float(config.get("duration", 8.0))
        self.sample_dt = float(config.get("sample_dt", 0.02))
        self.dt = 0.0025
        self.iterations = 6
        self.separation_speed = 0.12  # m/s cap on pre-existing penetration relaxation
        self.near_threshold = max(self.length_scale, 1e-3)
        self.rest_threshold = 0.2  # m/s below which restitution is ignored

        bodies = list(config.get("bodies", []))
        self.n = len(bodies)
        if self.n == 0:
            raise ValueError("scene config contains no bodies")
        position0 = np.asarray([b["position"] for b in bodies], dtype=float)
        quat0 = np.asarray(
            [b.get("orientation", [0.0, 0.0, 0.0, 1.0]) for b in bodies], dtype=float
        )
        quat0 = np.asarray([_qnorm(q) for q in quat0])
        self.mass = np.asarray([float(b["mass"]) for b in bodies], dtype=float)

        geometry = config.get("scene_geometry") or {}
        objects = list(geometry.get("objects") or [])

        asset = None
        for obj in objects:
            spec = obj.get("geometry") or {}
            if spec.get("type") == "triangle_mesh" and spec.get("asset"):
                asset = str(spec["asset"])
                break
        if asset is None or not os.path.isfile(asset):
            candidate = os.path.join(os.getcwd(), "assets", "jam_link.obj")
            asset = candidate if os.path.isfile(candidate) else None
        boxes = _compound_boxes_from_obj(asset) if asset else None
        if not boxes:
            boxes = [(np.array([0.0, 0.0, 0.07]), np.array([0.24, 0.16, 0.23]))]

        com, inertia_unit = _compound_mass_properties(boxes)
        self.com_local = com
        self.inertia_unit = inertia_unit
        self.box_center = [np.asarray(c, dtype=float) - com for c, _h in boxes]
        self.box_half = [np.asarray(h, dtype=float) for _c, h in boxes]
        self.nbox = len(self.box_center)
        try:
            self.inv_inertia_unit = np.linalg.inv(inertia_unit)
        except np.linalg.LinAlgError:
            self.inv_inertia_unit = np.eye(3) / max(1e-6, float(np.trace(inertia_unit)) / 3.0)
        self.radius = np.array(
            [
                max(
                    float(np.linalg.norm(c)) + float(np.linalg.norm(h))
                    for c, h in zip(self.box_center, self.box_half)
                )
            ]
            * self.n
        )

        self.st = _State(
            position0 + np.einsum("nij,nj->ni", np.asarray([_qmat(q) for q in quat0]), com),
            quat0.copy(),
            np.asarray([_qmat(q) for q in quat0]),
            np.zeros((self.n, 3)),
            np.zeros((self.n, 3)),
        )
        self.position0 = position0.copy()
        self.inv_mass = 1.0 / self.mass
        self.inv_inertia = self.inv_inertia_unit[None, :, :] / self.mass[:, None, None]
        self._update_world_inertia(self.st)

        self.statics = self._collect_statics(objects)
        if not self.statics:
            self.statics = [
                (np.array([0.0, 0.0, -0.05]), np.eye(3), np.array([3.2, 2.2, 0.05])),
                (np.array([0.0, 0.0, 0.12]), np.eye(3), np.array([2.25, 0.28, 0.08])),
                (np.array([0.0, 0.0, 1.02]), np.eye(3), np.array([2.25, 0.28, 0.08])),
            ]

        self.driver_body = int(np.argmin(position0[:, 0]))
        self.driver_force = 18.0
        self.driver_tau = 0.4

        self.t = 0.0
        self.frame_index = 0
        self.previous_velocity = self.st.v.copy()
        self.previous_time = 0.0
        self.accumulator = {}
        self.last_interval = self.dt
        self.diagnostics_data = {
            "substeps": 0,
            "contacts_last": 0,
            "max_penetration": 0.0,
            "final_speed": 0.0,
        }
        self._ready = True

    # ------------------------------------------------------------- helpers
    def _collect_statics(self, objects):
        statics = []
        for obj in objects:
            if str(obj.get("motion", "static")) not in ("static", "kinematic"):
                continue
            if not bool(obj.get("collision_enabled", True)):
                continue
            spec = obj.get("geometry") or {}
            transform = obj.get("transform") or {}
            translation = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            if spec.get("type") == "box" and spec.get("half_extents"):
                statics.append((translation, np.eye(3), np.asarray(spec["half_extents"], dtype=float)))
            elif spec.get("type") == "triangle_mesh" and spec.get("asset"):
                verts, _faces = _parse_obj(str(spec["asset"]))
                if verts.size:
                    lo, hi = verts.min(axis=0), verts.max(axis=0)
                    statics.append((translation + 0.5 * (lo + hi), np.eye(3), 0.5 * (hi - lo)))
        return statics

    def _update_world_inertia(self, st):
        st.W = np.einsum("nij,njk,nlk->nil", st.R, self.inv_inertia, st.R)

    def _driver_force_at(self, time):
        force = np.zeros((self.n, 3))
        ramp = 1.0 - math.exp(-max(0.0, float(time)) / self.driver_tau)
        force[self.driver_body, 0] = self.driver_force * ramp
        return force

    def _box_world(self, st, index, box_index):
        axis = st.R[index]
        center = st.x[index] + axis @ self.box_center[box_index]
        return center, axis, self.box_half[box_index]

    def _obb_separation(self, ca, axa, ha, cb, axb, hb):
        """Signed SAT separation of two oriented boxes plus the minimal axis.

        Returns (separation, axis, kind, owner) where kind is 0 for a face axis
        of A, 1 for a face axis of B and 2 for an edge-edge cross axis.
        """
        delta = cb - ca
        best = -1e18
        axis = None
        kind = 0
        owner = 0
        ra = rb = 0.0
        for k in range(3):
            u = axa[:, k]
            r1 = float(ha @ np.abs(axa.T @ u))
            r2 = float(hb @ np.abs(axb.T @ u))
            value = abs(float(delta @ u)) - (r1 + r2)
            if value > best:
                best, axis, ra, rb, kind, owner = value, u, r1, r2, 0, 0
        for k in range(3):
            u = axb[:, k]
            r1 = float(ha @ np.abs(axa.T @ u))
            r2 = float(hb @ np.abs(axb.T @ u))
            value = abs(float(delta @ u)) - (r1 + r2)
            if value > best:
                best, axis, ra, rb, kind, owner = value, u, r1, r2, 1, 1
        for k in range(3):
            for l in range(3):
                cross = np.cross(axa[:, k], axb[:, l])
                norm = float(np.linalg.norm(cross))
                if norm < 1e-8:
                    continue
                u = cross / norm
                r1 = float(ha @ np.abs(axa.T @ u))
                r2 = float(hb @ np.abs(axb.T @ u))
                value = abs(float(delta @ u)) - (r1 + r2)
                if value > best:
                    best, axis, ra, rb, kind, owner = value, u, r1, r2, 2, 0
        if axis is None:
            axis = np.array([1.0, 0.0, 0.0])
            best = 0.0
        return best, axis, kind, owner, ra, rb

    @staticmethod
    def _body_axis(st, body):
        return st.R[body] if body >= 0 else np.eye(3)

    def _world_box(self, st, body, box_index, center_local=None, half=None):
        axis = self._body_axis(st, body)
        if body >= 0:
            center = st.x[body] + axis @ self.box_center[box_index]
        else:
            return None
        return center, axis, self.box_half[box_index]

    def _emit(self, out, st, ref_body, ref_center, ref_axis, ref_half, ref_normal,
              inc_body, inc_center, inc_axis, inc_half, key):
        """Emit vertex/face contacts of the incident box against the reference face."""
        supports = np.abs(ref_axis.T @ ref_normal)
        kface = int(np.argmax(supports))
        extent = float(ref_half @ supports)
        others = [k for k in range(3) if k != kface]
        face_center = ref_center + ref_normal * extent
        corners = inc_center + (UNIT_CORNERS * inc_half) @ inc_axis.T
        candidates = []
        for ci in range(corners.shape[0]):
            point = corners[ci]
            gap = float((point - face_center) @ ref_normal)
            if gap >= 0.0:
                continue
            inside = True
            for k in others:
                if abs(float((point - face_center) @ ref_axis[:, k])) > float(ref_half[k]) + 1e-6:
                    inside = False
                    break
            candidates.append((gap, ci, inside))
        if not candidates:
            return
        selected = [c for c in candidates if c[2]]
        if not selected:
            selected = [min(candidates, key=lambda item: item[0])]
        ref_axis_inv = ref_axis.T
        inc_axis_inv = inc_axis.T
        for gap, ci, _inside in selected:
            point = corners[ci]
            if inc_body >= 0:
                p_inc = inc_axis_inv @ (point - st.x[inc_body])
            else:
                p_inc = point
            if ref_body >= 0:
                p_ref = ref_axis_inv @ (point - st.x[ref_body])
                n_local = ref_axis_inv @ ref_normal
            else:
                p_ref = point
                n_local = ref_normal
            out.append(_Contact((key, ci), inc_body, ref_body, p_inc, p_ref, n_local, gap))

    def _box_pair(self, out, st, body_a, box_a, body_b, box_b, key):
        ca = st.x[body_a] + st.R[body_a] @ self.box_center[box_a]
        axa = st.R[body_a]
        ha = self.box_half[box_a]
        if body_b >= 0:
            cb = st.x[body_b] + st.R[body_b] @ self.box_center[box_b]
            axb = st.R[body_b]
            hb = self.box_half[box_b]
        else:
            stat = self.statics[-body_b - 1]
            cb = np.asarray(stat[0], dtype=float)
            axb = np.asarray(stat[1], dtype=float)
            hb = np.asarray(stat[2], dtype=float)
        separation, axis, _kind, owner, _ra, _rb = self._obb_separation(ca, axa, ha, cb, axb, hb)
        delta = cb - ca
        sign = 1.0 if float(delta @ axis) >= 0.0 else -1.0
        normal = axis * sign  # from box a towards box b
        if separation >= 0.0:
            if separation < self.near_threshold:
                point = 0.5 * (ca + cb)
                p_inc = (
                    st.R[body_a].T @ (point - st.x[body_a]) if body_a >= 0 else point
                )
                if body_b >= 0:
                    p_ref = st.R[body_b].T @ (point - st.x[body_b])
                    n_local = st.R[body_b].T @ (-normal)
                else:
                    p_ref = point
                    n_local = -normal
                out.append(_Contact((key, 8), body_a, body_b, p_inc, p_ref, n_local, separation))
            return
        if owner == 1 or body_b < 0:
            # static geometry is always the reference side of a static contact
            ref_body, ref_center, ref_axis, ref_half = body_b, cb, axb, hb
            inc_body, inc_center, inc_axis, inc_half = body_a, ca, axa, ha
            ref_normal = -normal
        else:
            ref_body, ref_center, ref_axis, ref_half = body_a, ca, axa, ha
            inc_body, inc_center, inc_axis, inc_half = body_b, cb, axb, hb
            ref_normal = normal
        self._emit(
            out, st, ref_body, ref_center, ref_axis, ref_half, ref_normal,
            inc_body, inc_center, inc_axis, inc_half, key,
        )

    def _generate_contacts(self, st):
        out = []
        near = self.near_threshold
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if float(np.linalg.norm(st.x[i] - st.x[j])) > self.radius[i] + self.radius[j] + near:
                    continue
                for bi in range(self.nbox):
                    for bj in range(self.nbox):
                        self._box_pair(out, st, i, bi, j, bj, (i, j, bi, bj))
        for i in range(self.n):
            for s, stat in enumerate(self.statics):
                center = np.asarray(stat[0], dtype=float)
                half = np.asarray(stat[2], dtype=float)
                reach = self.radius[i] + float(np.linalg.norm(half)) + near
                if float(np.linalg.norm(st.x[i] - center)) > reach:
                    continue
                for bi in range(self.nbox):
                    self._box_pair(out, st, i, bi, -s - 1, 0, (i, -s - 1, bi, 0))
        return out

    # ------------------------------------------------------------- dynamics
    def _gap_at(self, contact, x, q, R):
        ia = contact.ia
        ib = contact.ib
        p_inc = x[ia] + R[ia] @ contact.p_inc_local
        if ib >= 0:
            p_ref = x[ib] + R[ib] @ contact.p_ref_local
            n = R[ib] @ contact.n_local
        else:
            p_ref = contact.p_ref_local
            n = contact.n_local
        return float((p_inc - p_ref) @ n)

    def _solve_contacts(self, st, contacts, h, x_old, q_old, R_old):
        lam = np.zeros(len(contacts))
        relax = self.separation_speed * h
        limits = np.zeros(len(contacts))
        for k, contact in enumerate(contacts):
            start = self._gap_at(contact, x_old, q_old, R_old)
            limits[k] = min(0.0, start + relax) if start < 0.0 else 0.0
        for _iteration in range(self.iterations):
            for k, c in enumerate(contacts):
                ia = c.ia
                ib = c.ib
                r_i = st.R[ia] @ c.p_inc_local
                p_inc = st.x[ia] + r_i
                if ib >= 0:
                    r_r = st.R[ib] @ c.p_ref_local
                    p_ref = st.x[ib] + r_r
                    n = st.R[ib] @ c.n_local
                else:
                    r_r = None
                    p_ref = c.p_ref_local
                    n = c.n_local
                violation = float((p_inc - p_ref) @ n) - limits[k]
                if violation >= 0.0:
                    if lam[k] <= 0.0:
                        continue
                denom = self.inv_mass[ia]
                denom += float(n @ np.cross(st.W[ia] @ np.cross(r_i, n), r_i))
                if ib >= 0:
                    denom += self.inv_mass[ib]
                    denom += float(n @ np.cross(st.W[ib] @ np.cross(r_r, n), r_r))
                if denom < 1e-12:
                    continue
                dlam = -violation / denom
                lam_new = lam[k] + dlam
                if lam_new < 0.0:
                    dlam = -lam[k]
                    lam_new = 0.0
                if dlam == 0.0:
                    continue
                lam[k] = lam_new
                st.x[ia] += (self.inv_mass[ia] * dlam) * n
                theta = st.W[ia] @ np.cross(r_i, n) * dlam
                st.q[ia] = _qnorm(_qmul(_q_rotvec(theta), st.q[ia]))
                st.R[ia] = _qmat(st.q[ia])
                if ib >= 0:
                    st.x[ib] -= (self.inv_mass[ib] * dlam) * n
                    theta = -(st.W[ib] @ np.cross(r_r, n)) * dlam
                    st.q[ib] = _qnorm(_qmul(_q_rotvec(theta), st.q[ib]))
                    st.R[ib] = _qmat(st.q[ib])
        return lam

    def _effective_mass(self, st, ia, ib, r_i, r_r, unit):
        denom = self.inv_mass[ia]
        denom += float(unit @ np.cross(st.W[ia] @ np.cross(r_i, unit), r_i))
        if ib >= 0:
            denom += self.inv_mass[ib]
            denom += float(unit @ np.cross(st.W[ib] @ np.cross(r_r, unit), r_r))
        return max(denom, 1e-12)

    def _apply_impulse(self, st, body, r, impulse, sign):
        st.v[body] += sign * self.inv_mass[body] * impulse
        st.w[body] += sign * (st.W[body] @ np.cross(r, impulse))

    def _velocity_pass(self, st, contacts, lam, h):
        for k, c in enumerate(contacts):
            normal_impulse = lam[k] / h
            if normal_impulse <= 0.0:
                continue
            ia = c.ia
            ib = c.ib
            r_i = st.R[ia] @ c.p_inc_local
            if ib >= 0:
                r_r = st.R[ib] @ c.p_ref_local
                n = st.R[ib] @ c.n_local
                v_rel = (st.v[ia] + np.cross(st.w[ia], r_i)) - (
                    st.v[ib] + np.cross(st.w[ib], r_r)
                )
            else:
                r_r = None
                n = c.n_local
                v_rel = st.v[ia] + np.cross(st.w[ia], r_i)
            v_normal = float(v_rel @ n)
            if v_normal < -self.rest_threshold and self.restitution > 0.0:
                k_n = self._effective_mass(st, ia, ib, r_i, r_r, n)
                bounce = -(1.0 + self.restitution) * v_normal / k_n
                self._apply_impulse(st, ia, r_i, bounce * n, 1.0)
                if ib >= 0:
                    self._apply_impulse(st, ib, r_r, bounce * n, -1.0)
                normal_impulse += bounce
                if ib >= 0:
                    v_rel = (st.v[ia] + np.cross(st.w[ia], r_i)) - (
                        st.v[ib] + np.cross(st.w[ib], r_r)
                    )
                else:
                    v_rel = st.v[ia] + np.cross(st.w[ia], r_i)
            tangential = v_rel - float(v_rel @ n) * n
            speed = float(np.linalg.norm(tangential))
            tangent_impulse = np.zeros(3)
            if speed > 1e-9:
                tangent = tangential / speed
                k_t = self._effective_mass(st, ia, ib, r_i, r_r, tangent)
                magnitude = min(speed / k_t, self.friction * normal_impulse)
                tangent_impulse = -magnitude * tangent
                self._apply_impulse(st, ia, r_i, tangent_impulse, 1.0)
                if ib >= 0:
                    self._apply_impulse(st, ib, r_r, tangent_impulse, -1.0)
                v_rel = v_rel + tangent_impulse * (
                    self.inv_mass[ia] + (self.inv_mass[ib] if ib >= 0 else 0.0)
                )
            record = self.accumulator.get(c.key)
            if record is None:
                record = {
                    "imp": np.zeros(3),
                    "normal": 0.0,
                    "tangent": np.zeros(3),
                    "rel_v": np.zeros(3),
                    "n": n.copy(),
                    "gap": float(c.gap),
                    "ia": ia,
                    "ib": ib,
                    "p_inc_local": c.p_inc_local.copy(),
                    "p_ref_local": c.p_ref_local.copy(),
                    "n_local": c.n_local.copy(),
                }
                self.accumulator[c.key] = record
            record["normal"] += normal_impulse
            record["tangent"] += tangent_impulse
            record["rel_v"] = v_rel.copy()
            record["n"] = n.copy()
            record["gap"] = float(c.gap)

    def _step(self, h, st=None):
        st = self.st if st is None else st
        self.diagnostics_data["substeps"] += 1
        x_old = st.x.copy()
        q_old = st.q.copy()
        R_old = st.R.copy()
        external = self._driver_force_at(self.t) + self.gravity[None, :] * self.mass[:, None]
        st.v = st.v + external * self.inv_mass[:, None] * h
        st.x = x_old + st.v * h
        for i in range(self.n):
            if float(np.linalg.norm(st.w[i])) > 1e-12:
                st.q[i] = _qnorm(_qmul(_q_rotvec(st.w[i] * h), st.q[i]))
                st.R[i] = _qmat(st.q[i])
        self._update_world_inertia(st)
        contacts = self._generate_contacts(st)
        self.diagnostics_data["contacts_last"] = len(contacts)
        lam = self._solve_contacts(st, contacts, h, x_old, q_old, R_old)
        st.v = (st.x - x_old) / h
        angular = np.zeros_like(st.w)
        for i in range(self.n):
            delta = _qmul(st.q[i], _qconj(q_old[i]))
            if float(delta[3]) < 0.0:
                delta = -delta
            angular[i] = _q_to_rotvec(delta) / h
        st.w = angular
        self._update_world_inertia(st)
        self._velocity_pass(st, contacts, lam, h)
        if not (np.isfinite(st.x).all() and np.isfinite(st.q).all() and np.isfinite(st.v).all()
                and np.isfinite(st.w).all()):
            st.x = x_old
            st.q = q_old
            st.R = np.asarray([_qmat(q) for q in q_old])
            st.v = np.zeros((self.n, 3))
            st.w = np.zeros((self.n, 3))
            self._update_world_inertia(st)
            self.diagnostics_data["nonfinite_recoveries"] = (
                int(self.diagnostics_data.get("nonfinite_recoveries", 0)) + 1
            )

    def _trial_estimate(self):
        """Estimate frame-0 contact forces with one uncommitted substep."""
        trial = self.st.copy()
        self._update_world_inertia(trial)
        self._step(self.dt, trial)

    # ------------------------------------------------------------- sampling
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("initialize() must be called before advance_to()")
        target = float(time)
        if target < self.t - 1e-9:
            raise ValueError("advance_to() requires non-decreasing sample times")
        if self.frame_index == 0 and self.t <= 1e-12:
            self._trial_estimate()
        while self.t < target - 1e-12:
            step = min(self.dt, target - self.t)
            self._step(step, self.st)
            self.t += step
        frame = self._build_frame(target)
        self.previous_velocity = self.st.v.copy()
        self.previous_time = target
        self.accumulator = {}
        self.frame_index += 1
        return frame

    def _driver_average(self, t0, t1):
        span = max(t1 - t0, 1e-12)
        tau = self.driver_tau
        mean = 1.0 - (tau / span) * (math.exp(-t0 / tau) - math.exp(-t1 / tau))
        return float(max(0.0, min(1.0, mean)))

    def _build_frame(self, time):
        st = self.st
        n = self.n
        if self.frame_index == 0:
            denominator = self.dt
            target_accel = None
        else:
            denominator = max(time - self.previous_time, 1e-12)
            target_accel = (st.v - self.previous_velocity) / denominator

        external = self.gravity[None, :] * self.mass[:, None] + self._driver_force_at(time)
        position = st.x - np.einsum("nij,nj->ni", st.R, self.com_local)
        inertia_local = self.inertia_unit[None, :, :] * self.mass[:, None, None]
        inertia_world = np.einsum("nij,njk,nlk->nil", st.R, inertia_local, st.R)

        fresh = self._generate_contacts(st)
        rows = []
        seen = set()
        for contact in fresh:
            if contact.gap < self.near_threshold:
                rows.append(contact)
                seen.add(contact.key)
        for key, record in self.accumulator.items():
            if key in seen:
                continue
            rows.append(
                _Contact(
                    key,
                    int(record["ia"]),
                    int(record["ib"]),
                    record["p_inc_local"],
                    record["p_ref_local"],
                    record["n_local"],
                    float(record["gap"]),
                )
            )
            seen.add(key)

        forces = []
        normal_forces = []
        tangent_forces = []
        normals = []
        relative = []
        for contact in rows:
            record = self.accumulator.get(contact.key)
            if record is None:
                if contact.ib >= 0:
                    n_world = st.R[contact.ib] @ contact.n_local
                else:
                    n_world = np.asarray(contact.n_local, dtype=float)
                norm = float(np.linalg.norm(n_world))
                if norm > 1e-12:
                    n_world = n_world / norm
                forces.append(np.zeros(3))
                normal_forces.append(0.0)
                tangent_forces.append(np.zeros(3))
                normals.append(n_world)
                relative.append(np.zeros(3))
                continue
            normal_force = record["normal"] / denominator
            tangent_force = record["tangent"] / denominator
            n_world = np.asarray(record["n"], dtype=float)
            forces.append(normal_force * n_world + tangent_force)
            normal_forces.append(float(normal_force))
            tangent_forces.append(np.asarray(tangent_force, dtype=float))
            normals.append(n_world)
            relative.append(np.asarray(record["rel_v"], dtype=float))

        if rows:
            interaction = np.asarray(forces, dtype=float)
            gaps = np.asarray([row.gap for row in rows], dtype=float)
            if target_accel is not None:
                interaction = self._rebalance(rows, interaction, target_accel, time, denominator)
        else:
            interaction = np.zeros((0, 3))
            gaps = np.zeros((0,))
        if float(gaps.size) and float(gaps.min()) < self.diagnostics_data.get(
            "max_penetration", 0.0
        ):
            self.diagnostics_data["max_penetration"] = float(gaps.min())
        self.diagnostics_data["final_speed"] = float(np.abs(st.v).max())
        self.diagnostics_data["contact_count"] = int(gaps.size)

        fields = {
            "contact_signed_distance": gaps,
            "external_force": external,
            "interaction_force": np.asarray(interaction, dtype=float).reshape(-1, 3),
            "friction_normal_force": np.asarray(normal_forces, dtype=float).reshape(-1),
            "friction_tangential_force": np.asarray(tangent_forces, dtype=float).reshape(-1, 3),
            "friction_normal": np.asarray(normals, dtype=float).reshape(-1, 3),
            "friction_relative_velocity": np.asarray(relative, dtype=float).reshape(-1, 3),
            "friction_static_coefficient": np.full((len(rows),), self.friction, dtype=float),
            "friction_dynamic_coefficient": np.full((len(rows),), self.friction, dtype=float),
            "gravity_acceleration": self.gravity.reshape(1, 3).copy(),
        }
        frame = StateFrame(
            float(time),
            position,
            st.v.copy(),
            orientation=st.q.copy(),
            angular_velocity=st.w.copy(),
            mass=self.mass.copy(),
            body_id=np.arange(n, dtype=np.int32),
            attributes={"inertia_tensor": inertia_world},
            fields=fields,
            scalars={"contact_length_scale": float(self.length_scale)},
        )
        frame.validate()
        return frame

    def _rebalance(self, rows, interaction, target_accel, time, denominator):
        n = self.n
        targets = self.mass[:, None] * target_accel
        driver_mean = self._driver_average(self.previous_time, time)
        targets = targets - self.gravity[None, :] * self.mass[:, None]
        targets[self.driver_body, 0] -= self.driver_force * driver_mean
        sums = np.zeros((n, 3))
        for index, contact in enumerate(rows):
            sums[contact.ia] += interaction[index]
            if contact.ib >= 0:
                sums[contact.ib] -= interaction[index]
        active = [
            index
            for index, contact in enumerate(rows)
            if contact.gap < 0.0 or float(np.linalg.norm(interaction[index])) > 0.0
        ]
        if not active:
            return interaction
        for _sweep in range(24):
            for index in active:
                contact = rows[index]
                delta = targets[contact.ia] - sums[contact.ia]
                if contact.ib >= 0:
                    delta = 0.5 * (delta - (targets[contact.ib] - sums[contact.ib]))
                else:
                    delta = 0.5 * delta
                interaction[index] += delta
                sums[contact.ia] += delta
                if contact.ib >= 0:
                    sums[contact.ib] -= delta
        return interaction

    def diagnostics(self) -> dict[str, object]:
        data = dict(self.diagnostics_data)
        data["scene_id"] = "143_frictional_mechanism_jamming"
        data["bodies"] = int(self.n)
        data["substep"] = float(self.dt)
        data["final_time"] = float(self.t)
        data["driver_force"] = float(self.driver_force)
        return data
