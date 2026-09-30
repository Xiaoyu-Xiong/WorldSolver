"""Self-contained impulse-based 3D rigid-body solver for 124_wrecking_ball_box_stack.

Physics: semi-implicit Euler time stepping with a sequential-impulse (projected
Gauss-Seidel) velocity solver for box/sphere/capsule contacts (Coulomb friction,
restitution) and spherical (ball-and-socket) chain joints. Position drift and
penetration are removed with a split-impulse pseudo-velocity pass. No external
physics engine is used; only NumPy.
"""
from __future__ import annotations

import math
import os

import numpy as np

from worldsolver.contracts import StateFrame

H = 1.0e-3                     # substep (s)
GRAVITY = (0.0, 0.0, -9.81)    # filled from config at init
MU = 0.55
RESTITUTION = 0.12
LENGTH_SCALE = 0.02
SLOP = 1.0e-3
BETA_POS = 0.35
BETA_JOINT = 0.5
VEL_ITERS = 12
POS_ITERS = 5
GEN_MARGIN = 2.0e-3
REPORT_MARGIN = 0.05
RELEASE_ANGLE = 0.7853981633974483   # 45 deg, displaces ball away from the stack
DAMP_LIN = 0.02
DAMP_ANG = 0.05
V_MAX = 30.0
REST_THRESH = 0.3

# pair type codes
SHAPE_CODE = {
    ("box", "box"): 0,
    ("sphere", "box"): 1, ("box", "sphere"): 2,
    ("capsule", "box"): 3, ("box", "capsule"): 4,
    ("capsule", "capsule"): 5,
    ("sphere", "capsule"): 6, ("capsule", "sphere"): 7,
    ("sphere", "sphere"): 8,
}


def quat_to_mat(q):
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ])


def quat_mul(a, b):
    ax, ay, az, aw = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx, by, bz, bw = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_step(q, wx, wy, wz):
    """Integrate unit quaternion by world angular velocity over H."""
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    f = 0.5 * H
    dqx = f * (wx * qw + wy * qz - wz * qy)
    dqy = f * (wy * qw + wz * qx - wx * qz)
    dqz = f * (wz * qw + wx * qy - wy * qx)
    dqw = f * (-wx * qx - wy * qy - wz * qz)
    nx, ny, nz, nw = qx + dqx, qy + dqy, qz + dqz, qw + dqw
    n = math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.array([nx / n, ny / n, nz / n, nw / n])


def kterm(i9, rx, ry, rz, tx, ty, tz):
    """v^T I v with v = r x t; i9 is row-major 3x3 world inverse inertia."""
    cx = ry * tz - rz * ty
    cy = rz * tx - rx * tz
    cz = rx * ty - ry * tx
    return (i9[0] * cx * cx + i9[4] * cy * cy + i9[8] * cz * cz
            + 2.0 * i9[1] * cx * cy + 2.0 * i9[2] * cx * cz + 2.0 * i9[5] * cy * cz)


def rel_vel(A, B, rax, ray, raz, rbx, rby, rbz):
    Av, Aw, Bv, Bw = A.v, A.w, B.v, B.w
    rx = Bv[0] + (Bw[1] * rbz - Bw[2] * rby) - Av[0] - (Aw[1] * raz - Aw[2] * ray)
    ry = Bv[1] + (Bw[2] * rbx - Bw[0] * rbz) - Av[1] - (Aw[2] * rax - Aw[0] * raz)
    rz = Bv[2] + (Bw[0] * rby - Bw[1] * rbx) - Av[2] - (Aw[0] * ray - Aw[1] * rax)
    return rx, ry, rz


def apply_impulse(A, B, rax, ray, raz, rbx, rby, rbz, ix, iy, iz):
    ima = A.ima
    if ima != 0.0:
        Av, Aw = A.v, A.w
        Av[0] -= ima * ix
        Av[1] -= ima * iy
        Av[2] -= ima * iz
        cx = ray * iz - raz * iy
        cy = raz * ix - rax * iz
        cz = rax * iy - ray * ix
        I = A.iinv9
        Aw[0] -= I[0] * cx + I[1] * cy + I[2] * cz
        Aw[1] -= I[3] * cx + I[4] * cy + I[5] * cz
        Aw[2] -= I[6] * cx + I[7] * cy + I[8] * cz
    imb = B.ima
    if imb != 0.0:
        Bv, Bw = B.v, B.w
        Bv[0] += imb * ix
        Bv[1] += imb * iy
        Bv[2] += imb * iz
        cx = rby * iz - rbz * iy
        cy = rbz * ix - rbx * iz
        cz = rbx * iy - rby * ix
        I = B.iinv9
        Bw[0] += I[0] * cx + I[1] * cy + I[2] * cz
        Bw[1] += I[3] * cx + I[4] * cy + I[5] * cz
        Bw[2] += I[6] * cx + I[7] * cy + I[8] * cz


def solve_normal(A, B, rax, ray, raz, rbx, rby, rbz, nx, ny, nz, kn, target, lam_old):
    rx, ry, rz = rel_vel(A, B, rax, ray, raz, rbx, rby, rbz)
    vn = nx * rx + ny * ry + nz * rz
    dl = (target - vn) / kn
    lam = lam_old + dl
    if lam < 0.0:
        lam = 0.0
    dl = lam - lam_old
    if dl != 0.0:
        apply_impulse(A, B, rax, ray, raz, rbx, rby, rbz, dl * nx, dl * ny, dl * nz)
    return lam


def solve_friction(A, B, rax, ray, raz, rbx, rby, rbz, tx, ty, tz, kt, lam_old, maxf):
    rx, ry, rz = rel_vel(A, B, rax, ray, raz, rbx, rby, rbz)
    vt = tx * rx + ty * ry + tz * rz
    dl = -vt / kt
    lam = lam_old + dl
    if lam > maxf:
        lam = maxf
    elif lam < -maxf:
        lam = -maxf
    dl = lam - lam_old
    if dl != 0.0:
        apply_impulse(A, B, rax, ray, raz, rbx, rby, rbz, dl * tx, dl * ty, dl * tz)
    return lam


def solve_joint(A, B, rax, ray, raz, rbx, rby, rbz, m, errx, erry, errz, use_pseudo):
    """Spherical joint block solve. m = row-major inverse effective mass."""
    if use_pseudo:
        Av, Aw, Bv, Bw = A.u, A.pu, B.u, B.pu
    else:
        Av, Aw, Bv, Bw = A.v, A.w, B.v, B.w
    rx = Bv[0] + (Bw[1] * rbz - Bw[2] * rby) - Av[0] - (Aw[1] * raz - Aw[2] * ray)
    ry = Bv[1] + (Bw[2] * rbx - Bw[0] * rbz) - Av[1] - (Aw[2] * rax - Aw[0] * raz)
    rz = Bv[2] + (Bw[0] * rby - Bw[1] * rbx) - Av[2] - (Aw[0] * ray - Aw[1] * rax)
    # rhs = -(vrel + beta * err / H) ; err = 0 for the pure velocity pass
    sx = -(rx + errx)
    sy = -(ry + erry)
    sz = -(rz + errz)
    lx = m[0] * sx + m[1] * sy + m[2] * sz
    ly = m[3] * sx + m[4] * sy + m[5] * sz
    lz = m[6] * sx + m[7] * sy + m[8] * sz
    apply_impulse(A, B, rax, ray, raz, rbx, rby, rbz, lx, ly, lz)
    return lx, ly, lz


class Body:
    __slots__ = ("idx", "shape", "mass", "ima", "inv_ib", "pos", "q", "v", "w",
                 "u", "pu", "R", "iinv_w", "iinv9", "he", "radius", "seg_half",
                 "corners", "dyn")

    def __init__(self, idx, shape, mass, pos, q, he=None, radius=0.0, seg_half=0.0,
                 dyn=True):
        self.idx = idx
        self.shape = shape
        self.mass = float(mass)
        self.dyn = bool(dyn)
        self.ima = (1.0 / self.mass) if (dyn and self.mass > 0.0) else 0.0
        self.pos = np.asarray(pos, dtype=float).copy()
        self.q = np.asarray(q, dtype=float).copy()
        self.v = [0.0, 0.0, 0.0]
        self.w = [0.0, 0.0, 0.0]
        self.u = [0.0, 0.0, 0.0]
        self.pu = [0.0, 0.0, 0.0]
        self.radius = float(radius)
        self.seg_half = float(seg_half)
        self.he = np.asarray(he, dtype=float) if he is not None else None
        if shape == "box" and self.he is not None:
            s = np.array([[1., 1., 1.], [1., 1., -1.], [1., -1., 1.], [1., -1., -1.],
                          [-1., 1., 1.], [-1., 1., -1.], [-1., -1., 1.], [-1., -1., -1.]])
            self.corners = s * self.he
        else:
            self.corners = None
        if self.ima != 0.0:
            if shape == "box":
                hx, hy, hz = float(self.he[0]), float(self.he[1]), float(self.he[2])
                ix = self.mass / 3.0 * (hy * hy + hz * hz)
                iy = self.mass / 3.0 * (hx * hx + hz * hz)
                iz = self.mass / 3.0 * (hx * hx + hy * hy)
            elif shape == "sphere":
                iv = 0.4 * self.mass * self.radius * self.radius
                ix = iy = iz = iv
            else:  # capsule: cylinder approximation, axis = local z
                r = self.radius
                l2 = (2.0 * self.seg_half) ** 2
                ip = self.mass * (3.0 * r * r + l2) / 12.0
                ix = iy = ip
                iz = 0.5 * self.mass * r * r
            self.inv_ib = np.array([1.0 / ix, 1.0 / iy, 1.0 / iz])
        else:
            self.inv_ib = np.zeros(3)
        self.R = np.eye(3)
        self.iinv_w = np.zeros((3, 3))
        self.iinv9 = (0.0,) * 9

    def refresh(self):
        self.R = quat_to_mat(self.q)
        self.iinv_w = (self.R * self.inv_ib) @ self.R.T
        self.iinv9 = tuple(self.iinv_w.ravel())


# ---------------------------------------------------------------------------
# narrowphase geometry helpers (module level, numpy based)
# ---------------------------------------------------------------------------

def sphere_box_geom(center, radius, bx):
    """Sphere (center, radius) vs oriented box bx.

    Returns (point, nout, sd): contact point on box surface, unit normal
    pointing from the box toward the sphere, and signed distance
    (negative = penetration).
    """
    loc = bx.R.T @ (center - bx.pos)
    he = bx.he
    cl = np.clip(loc, -he, he)
    a = np.abs(loc)
    if a[0] < he[0] and a[1] < he[1] and a[2] < he[2]:
        sl = he - a
        k = int(np.argmin(sl))
        sgn = 1.0 if loc[k] >= 0.0 else -1.0
        nout = bx.R[:, k] * sgn
        return center + nout * float(sl[k]), nout, -float(sl[k]) - radius
    d = loc - cl
    dist = float(math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))
    if dist > 1e-12:
        nout = bx.R @ (d / dist)
    else:
        nout = bx.R[:, 2].copy()
    return bx.pos + bx.R @ cl, nout, dist - radius


def gen_sphere_box(s, bx, ia, ib, flip, out):
    point, nout, sd = sphere_box_geom(s.pos, s.radius, bx)
    if sd < GEN_MARGIN:
        if flip:  # ia is the sphere -> normal from sphere to box
            nx, ny, nz = -nout[0], -nout[1], -nout[2]
        else:
            nx, ny, nz = nout[0], nout[1], nout[2]
        out.append((ia, ib, float(point[0]), float(point[1]), float(point[2]),
                    float(nx), float(ny), float(nz), -sd))


def gen_capsule_box(c, bx, ia, ib, flip, out):
    ax = c.R[:, 2] * c.seg_half
    for t in (-1.0, -0.5, 0.0, 0.5, 1.0):
        center = c.pos + ax * t
        point, nout, sd = sphere_box_geom(center, c.radius, bx)
        if sd < GEN_MARGIN:
            if flip:  # ia is the capsule
                nx, ny, nz = -nout[0], -nout[1], -nout[2]
            else:
                nx, ny, nz = nout[0], nout[1], nout[2]
            out.append((ia, ib, float(point[0]), float(point[1]), float(point[2]),
                        float(nx), float(ny), float(nz), -sd))


def gen_box_box(a, b, ia, ib, out):
    # a's vertices inside b: push a out along nearest b face -> n(a->b) = -face
    cw = a.corners @ a.R.T + a.pos
    loc = (cw - b.pos) @ b.R
    sl = b.he - np.abs(loc)
    ins = np.all(sl > 0.0, axis=1)
    hits = []
    for r in np.nonzero(ins)[0]:
        rr = sl[r]
        k = int(np.argmin(rr))
        sgn = 1.0 if loc[r, k] >= 0.0 else -1.0
        col = b.R[:, k] * sgn
        hits.append((float(rr[k]), cw[r, 0], cw[r, 1], cw[r, 2],
                     -col[0], -col[1], -col[2]))
    hits.sort(key=lambda d: -d[0])
    for d in hits[:4]:
        out.append((ia, ib, float(d[1]), float(d[2]), float(d[3]),
                    float(d[4]), float(d[5]), float(d[6]), d[0]))
    # b's vertices inside a: n(a->b) = +face of a
    cw2 = b.corners @ b.R.T + b.pos
    loc2 = (cw2 - a.pos) @ a.R
    sl2 = a.he - np.abs(loc2)
    ins2 = np.all(sl2 > 0.0, axis=1)
    hits2 = []
    for r in np.nonzero(ins2)[0]:
        rr = sl2[r]
        k = int(np.argmin(rr))
        sgn = 1.0 if loc2[r, k] >= 0.0 else -1.0
        col = a.R[:, k] * sgn
        hits2.append((float(rr[k]), cw2[r, 0], cw2[r, 1], cw2[r, 2],
                      col[0], col[1], col[2]))
    hits2.sort(key=lambda d: -d[0])
    for d in hits2[:4]:
        out.append((ia, ib, float(d[1]), float(d[2]), float(d[3]),
                    float(d[4]), float(d[5]), float(d[6]), d[0]))


def seg_seg_closest(p0, p1, q0, q1):
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    aa = float(d1 @ d1)
    ee = float(d2 @ d2)
    ff = float(d2 @ r)
    if aa <= 1e-14 and ee <= 1e-14:
        return p0, q0
    if aa <= 1e-14:
        t = min(1.0, max(0.0, ff / ee))
        return p0, q0 + t * d2
    c = float(d1 @ r)
    if ee <= 1e-14:
        s = min(1.0, max(0.0, -c / aa))
        return p0 + s * d1, q0
    bb = float(d1 @ d2)
    denom = aa * ee - bb * bb
    if denom > 1e-14 * aa * ee:
        s = min(1.0, max(0.0, (bb * ff - c * ee) / denom))
    else:
        s = 0.0
    t = (bb * s + ff) / ee
    if t < 0.0:
        t = 0.0
        s = min(1.0, max(0.0, -c / aa))
    elif t > 1.0:
        t = 1.0
        s = min(1.0, max(0.0, (bb - c) / aa))
    return p0 + s * d1, q0 + t * d2


def gen_capsule_capsule(a, b, ia, ib, out):
    ax = a.R[:, 2] * a.seg_half
    bx = b.R[:, 2] * b.seg_half
    c1, c2 = seg_seg_closest(a.pos - ax, a.pos + ax, b.pos - bx, b.pos + bx)
    dv = c2 - c1
    dist = float(math.sqrt(dv @ dv))
    sd = dist - (a.radius + b.radius)
    if sd < GEN_MARGIN:
        if dist > 1e-9:
            nx, ny, nz = dv[0] / dist, dv[1] / dist, dv[2] / dist
        else:
            nx, ny, nz = 1.0, 0.0, 0.0
        point = 0.5 * (c1 + c2)
        out.append((ia, ib, float(point[0]), float(point[1]), float(point[2]),
                    float(nx), float(ny), float(nz), -sd))


def gen_sphere_capsule(s, c, ia, ib, flip, out):
    ax = c.R[:, 2] * c.seg_half
    p0 = c.pos - ax
    d = (c.pos + ax) - p0
    dd = float(d @ d)
    if dd > 1e-14:
        t = float((s.pos - p0) @ d) / dd
        t = min(1.0, max(0.0, t))
    else:
        t = 0.0
    cp = p0 + t * d
    dv = cp - s.pos
    dist = float(math.sqrt(dv @ dv))
    sd = dist - (s.radius + c.radius)
    if sd < GEN_MARGIN:
        if dist > 1e-9:
            ux, uy, uz = dv[0] / dist, dv[1] / dist, dv[2] / dist
        else:
            ux, uy, uz = 0.0, 0.0, 1.0
        if flip:  # ia is the sphere: normal from sphere toward capsule
            nx, ny, nz = ux, uy, uz
        else:
            nx, ny, nz = -ux, -uy, -uz
        out.append((ia, ib, float(cp[0]), float(cp[1]), float(cp[2]),
                    float(nx), float(ny), float(nz), -sd))


# ---------------------------------------------------------------------------
# signed-distance helpers for trajectory reporting
# ---------------------------------------------------------------------------

def sd_box_box(a, b):
    best = float("inf")
    for (src, dst) in ((a, b), (b, a)):
        cw = src.corners @ src.R.T + src.pos
        loc = (cw - dst.pos) @ dst.R
        cl = np.clip(loc, -dst.he, dst.he)
        sl = dst.he - np.abs(loc)
        ins = np.all(sl > 0.0, axis=1)
        if np.any(ins):
            mins = np.min(sl[ins], axis=1)
            best = min(best, -float(np.min(mins)))
        else:
            d = loc - cl
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            best = min(best, float(np.min(dist)))
    return best


def sd_capsule_box(c, bx):
    ax = c.R[:, 2] * c.seg_half
    best = float("inf")
    for t in (-1.0, -0.5, 0.0, 0.5, 1.0):
        center = c.pos + ax * t
        _, _, sd = sphere_box_geom(center, c.radius, bx)
        best = min(best, sd)
    return best


def sd_capsule_capsule(a, b):
    ax = a.R[:, 2] * a.seg_half
    bx = b.R[:, 2] * b.seg_half
    c1, c2 = seg_seg_closest(a.pos - ax, a.pos + ax, b.pos - bx, b.pos + bx)
    dv = c2 - c1
    return float(math.sqrt(dv @ dv)) - (a.radius + b.radius)


def sd_sphere_capsule(s, c):
    ax = c.R[:, 2] * c.seg_half
    p0 = c.pos - ax
    d = (c.pos + ax) - p0
    dd = float(d @ d)
    if dd > 1e-14:
        t = min(1.0, max(0.0, float((s.pos - p0) @ d) / dd))
    else:
        t = 0.0
    dv = (p0 + t * d) - s.pos
    return float(math.sqrt(dv @ dv)) - (s.radius + c.radius)


class BenchmarkSolver:
    """Sequential-impulse rigid-body solver for the wrecking-ball scene."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        init = scene.initial_state()
        pos0 = np.asarray(init.position, dtype=float)
        quat0 = np.asarray(init.orientation, dtype=float)
        mass0 = np.asarray(init.mass, dtype=float)
        n = len(mass0)

        g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.grav = (float(g[0]), float(g[1]), float(g[2]))
        contact = cfg.get("contact", {}) or {}
        self.mu = float(contact.get("friction", MU))
        self.rest = float(contact.get("restitution", RESTITUTION))
        self.length_scale = float(contact.get("length_scale", LENGTH_SCALE))

        attr = dict(init.attributes or {})
        self.release_time = np.asarray(attr.get("release_time", np.zeros(n)),
                                        dtype=float)

        # ---- geometry lookup -------------------------------------------------
        geom = cfg.get("scene_geometry") or {}
        objects = geom.get("objects", []) if isinstance(geom, dict) else []
        by_body = {}
        floor_obj = None
        for obj in objects:
            bind = obj.get("state_binding")
            if bind and bind.get("selector") == "body_id":
                by_body[int(bind["value"])] = obj
            elif obj.get("id") == "floor":
                floor_obj = obj

        bodies = []
        for i in range(n):
            obj = by_body.get(i)
            gtype = None
            he = None
            radius = 0.0
            seg_half = 0.0
            if obj is not None:
                geo = obj.get("geometry", {}) or {}
                gtype = geo.get("type")
                if gtype == "box":
                    he = [float(v) for v in geo.get("half_extents", [0.1, 0.1, 0.1])]
                elif gtype == "capsule":
                    radius = float(geo.get("radius", 0.05))
                    seg_half = float(geo.get("height", 0.2))
                elif gtype == "triangle_mesh":
                    radius = self._mesh_radius(geo.get("asset"))
                    gtype = "sphere"
            else:  # fallback from config ids
                bid = cfg["bodies"][i]["id"]
                if bid.startswith("box_"):
                    gtype = "box"
                    he = [0.17, 0.38, 0.17]
                elif bid.startswith("chain_"):
                    gtype = "capsule"
                    radius = 0.07
                    seg_half = 0.25
                else:
                    gtype = "sphere"
                    radius = 0.42
            bodies.append(Body(i, gtype, mass0[i], pos0[i], quat0[i],
                               he=he, radius=radius, seg_half=seg_half, dyn=True))

        # ---- static floor ----------------------------------------------------
        floor_he = [3.2, 2.2, 0.05]
        floor_pos = np.array([0.0, 0.0, -0.05])
        if floor_obj is not None:
            fgeo = floor_obj.get("geometry", {}) or {}
            floor_he = [float(v) for v in fgeo.get("half_extents", floor_he)]
            ftr = (floor_obj.get("transform", {}) or {}).get("translation", None)
            if ftr is not None:
                floor_pos = np.asarray(ftr, dtype=float)
        floor = Body(len(bodies), "box", 0.0, floor_pos, [0.0, 0.0, 0.0, 1.0],
                     he=floor_he, dyn=False)
        bodies.append(floor)
        self.floor_idx = floor.idx

        # ---- pendulum assembly: ball + chain links ---------------------------
        chains = [b for b in bodies[:n] if b.shape == "capsule"]
        balls = [b for b in bodies[:n] if b.shape == "sphere"]
        chains.sort(key=lambda b: float(b.pos[2]))
        tip = (chains[0].seg_half + chains[0].radius) if chains else 0.32

        top = chains[-1]
        anchor = top.pos + np.array([0.0, 0.0, tip])

        # rigidly rotate ball + chain about the anchor, displacing the ball to
        # the -x side (away from the box stack) with the chain taut, at rest.
        pend = chains + balls
        ang = RELEASE_ANGLE
        rrot = np.array([[math.cos(ang), 0.0, math.sin(ang)],
                         [0.0, 1.0, 0.0],
                         [-math.sin(ang), 0.0, math.cos(ang)]])
        qrot = np.array([0.0, math.sin(0.5 * ang), 0.0, math.cos(0.5 * ang)])
        for b in pend:
            b.pos = anchor + rrot @ (b.pos - anchor)
            b.q = quat_mul(qrot, b.q)

        # ---- joints (spherical) ----------------------------------------------
        world = Body(len(bodies), "point", 0.0, anchor, [0.0, 0.0, 0.0, 1.0],
                     dyn=False)
        bodies.append(world)
        self.world_idx = world.idx

        joints = []
        if chains:
            joints.append((top.idx, world.idx, (0.0, 0.0, tip), (0.0, 0.0, 0.0)))
            for k in range(len(chains) - 1):
                lower = chains[k]
                upper = chains[k + 1]
                joints.append((upper.idx, lower.idx, (0.0, 0.0, -tip), (0.0, 0.0, tip)))
            if balls:
                ball = balls[0]
                low = chains[0]
                baz = float(low.pos[2]) - tip - float(ball.pos[2])
                joints.append((low.idx, ball.idx, (0.0, 0.0, -tip), (0.0, 0.0, baz)))
        self.joints = joints

        # ---- broadphase pair table --------------------------------------------
        skip = set()
        ci = [b.idx for b in chains]
        for k in range(len(ci) - 1):
            skip.add((ci[k], ci[k + 1]))
        if balls and chains:
            skip.add((min(balls[0].idx, ci[0]), max(balls[0].idx, ci[0])))
        pi = []
        pj = []
        pt = []
        nb = len(bodies)
        for i in range(nb):
            for j in range(i + 1, nb):
                if i == self.world_idx or j == self.world_idx:
                    continue
                bi = bodies[i]
                bj = bodies[j]
                if (not bi.dyn) and (not bj.dyn):
                    continue
                if (i, j) in skip:
                    continue
                code = SHAPE_CODE.get((bi.shape, bj.shape))
                if code is None:
                    continue
                pi.append(i)
                pj.append(j)
                pt.append(code)
        self._pi = np.asarray(pi, dtype=np.int32)
        self._pj = np.asarray(pj, dtype=np.int32)
        self._pt = pt
        self._pairs = list(zip(pi, pj))

        self.bodies = bodies
        self.n_dyn = n
        self.gravity_vec = np.array(self.grav, dtype=float)
        self.step = 0
        self.time = 0.0
        self._last_rows = []
        self._last_lams = []
        self._last_jlam = []
        self._last_jdata = []

    @staticmethod
    def _mesh_radius(asset_path):
        """Bounding-cylinder radius of a triangle mesh asset (sphere approx)."""
        try:
            path = asset_path
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            if not os.path.isfile(path):
                return 0.42
            r = 0.0
            with open(path, "r", errors="ignore") as fh:
                for line in fh:
                    if line.startswith("v "):
                        parts = line.split()
                        x = float(parts[1])
                        y = float(parts[2])
                        r = max(r, math.sqrt(x * x + y * y))
            return r if r > 1e-6 else 0.42
        except Exception:
            return 0.42

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def _aabbs(self):
        bodies = self.bodies
        nb = len(bodies)
        lo = np.empty((nb, 3))
        hi = np.empty((nb, 3))
        for b in bodies:
            i = b.idx
            if b.shape == "box":
                e = np.abs(b.R) @ b.he
                lo[i] = b.pos - e
                hi[i] = b.pos + e
            elif b.shape == "sphere":
                lo[i] = b.pos - b.radius
                hi[i] = b.pos + b.radius
            elif b.shape == "capsule":
                ax = b.R[:, 2] * b.seg_half
                p1 = b.pos + ax
                p2 = b.pos - ax
                lo[i] = np.minimum(p1, p2) - b.radius
                hi[i] = np.maximum(p1, p2) + b.radius
            else:
                lo[i] = b.pos
                hi[i] = b.pos
        return lo, hi

    def _detect(self, margin):
        bodies = self.bodies
        lo, hi = self._aabbs()
        PI, PJ = self._pi, self._pj
        m = np.all(lo[PI] <= hi[PJ] + margin, axis=1) & \
            np.all(lo[PJ] <= hi[PI] + margin, axis=1)
        out = []
        for k in np.nonzero(m)[0]:
            i = int(PI[k])
            j = int(PJ[k])
            code = self._pt[k]
            a = bodies[i]
            b = bodies[j]
            if code == 0:
                gen_box_box(a, b, i, j, out)
            elif code == 1:
                gen_sphere_box(a, b, i, j, True, out)
            elif code == 2:
                gen_sphere_box(b, a, j, i, False, out)
            elif code == 3:
                gen_capsule_box(a, b, i, j, True, out)
            elif code == 4:
                gen_capsule_box(b, a, j, i, False, out)
            elif code == 5:
                gen_capsule_capsule(a, b, i, j, out)
            elif code == 6:
                gen_sphere_capsule(a, b, i, j, True, out)
            elif code == 7:
                gen_sphere_capsule(b, a, j, i, False, out)
            else:  # sphere-sphere
                dv = b.pos - a.pos
                dist = float(math.sqrt(dv @ dv))
                sd = dist - (a.radius + b.radius)
                if sd < margin:
                    if dist > 1e-9:
                        nx, ny, nz = dv / dist
                    else:
                        nx, ny, nz = 1.0, 0.0, 0.0
                    point = 0.5 * (a.pos + b.pos)
                    out.append((i, j, float(point[0]), float(point[1]),
                                float(point[2]), float(nx), float(ny),
                                float(nz), -sd))
        return out

    def _build_rows(self, contacts):
        bodies = self.bodies
        rows = []
        for (ia, ib, px, py, pz, nx, ny, nz, depth) in contacts:
            A = bodies[ia]
            B = bodies[ib]
            rax = px - float(A.pos[0])
            ray = py - float(A.pos[1])
            raz = pz - float(A.pos[2])
            rbx = px - float(B.pos[0])
            rby = py - float(B.pos[1])
            rbz = pz - float(B.pos[2])
            nl = math.sqrt(nx * nx + ny * ny + nz * nz)
            if nl < 1e-12:
                continue
            nx /= nl
            ny /= nl
            nz /= nl
            if abs(nz) < 0.9:
                # t1 = n x (0,0,1)
                t1x, t1y, t1z = ny, -nx, 0.0
            else:
                # t1 = n x (1,0,0)
                t1x, t1y, t1z = 0.0, nz, -ny
            tl = math.sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
            if tl < 1e-12:
                continue
            t1x /= tl
            t1y /= tl
            t1z /= tl
            t2x = ny * t1z - nz * t1y
            t2y = nz * t1x - nx * t1z
            t2z = nx * t1y - ny * t1x
            ia9 = A.iinv9
            ib9 = B.iinv9
            kn = A.ima + B.ima + kterm(ia9, rax, ray, raz, nx, ny, nz) \
                + kterm(ib9, rbx, rby, rbz, nx, ny, nz)
            if kn <= 1e-12:
                continue
            ku = A.ima + B.ima + kterm(ia9, rax, ray, raz, t1x, t1y, t1z) \
                + kterm(ib9, rbx, rby, rbz, t1x, t1y, t1z)
            kv = A.ima + B.ima + kterm(ia9, rax, ray, raz, t2x, t2y, t2z) \
                + kterm(ib9, rbx, rby, rbz, t2x, t2y, t2z)
            if ku <= 1e-12:
                ku = 1e-12
            if kv <= 1e-12:
                kv = 1e-12
            rx, ry, rz = rel_vel(A, B, rax, ray, raz, rbx, rby, rbz)
            vn0 = nx * rx + ny * ry + nz * rz
            if vn0 < -REST_THRESH:
                bounce = -self.rest * vn0
            else:
                bounce = 0.0
            pb = BETA_POS * max(depth - SLOP, 0.0) / H
            rows.append((ia, ib, rax, ray, raz, rbx, rby, rbz,
                         nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z,
                         kn, ku, kv, bounce, pb))
        return rows

    def _joint_data(self):
        bodies = self.bodies
        jdata = []
        eye3 = np.eye(3)
        for (ia, ib, la, lb) in self.joints:
            A = bodies[ia]
            B = bodies[ib]
            ra = A.R @ np.asarray(la, dtype=float)
            rb = B.R @ np.asarray(lb, dtype=float)
            M = (A.ima + B.ima) * eye3
            if A.ima != 0.0:
                M = M - _skew_np(ra) @ A.iinv_w @ _skew_np(ra)
            if B.ima != 0.0:
                M = M - _skew_np(rb) @ B.iinv_w @ _skew_np(rb)
            try:
                Minv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                Minv = np.zeros((3, 3))
            jdata.append((ia, ib,
                          float(ra[0]), float(ra[1]), float(ra[2]),
                          float(rb[0]), float(rb[1]), float(rb[2]),
                          tuple(Minv.ravel())))
        return jdata

    def _substep(self):
        bodies = self.bodies
        gx, gy, gz = self.grav
        fl = 1.0 - DAMP_LIN * H
        fa = 1.0 - DAMP_ANG * H
        for b in bodies:
            if b.dyn:
                b.refresh()
                v = b.v
                w = b.w
                v[0] += gx * H
                v[1] += gy * H
                v[2] += gz * H
                v[0] *= fl
                v[1] *= fl
                v[2] *= fl
                w[0] *= fa
                w[1] *= fa
                w[2] *= fa
                b.u[0] = b.u[1] = b.u[2] = 0.0
                b.pu[0] = b.pu[1] = b.pu[2] = 0.0
            else:
                b.refresh()

        contacts = self._detect(GEN_MARGIN)
        rows = self._build_rows(contacts)
        lams = [[0.0, 0.0, 0.0] for _ in rows]
        plams = [[0.0] for _ in rows]
        jdata = self._joint_data()
        jlam = [[0.0, 0.0, 0.0] for _ in jdata]
        mu = self.mu

        # ---- velocity pass (contacts + joints) ----
        for _ in range(VEL_ITERS):
            for k in range(len(rows)):
                (ia, ib, rax, ray, raz, rbx, rby, rbz,
                 nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z,
                 kn, ku, kv, bounce, pb) = rows[k]
                A = bodies[ia]
                B = bodies[ib]
                lk = lams[k]
                lk[0] = solve_normal(A, B, rax, ray, raz, rbx, rby, rbz,
                                     nx, ny, nz, kn, bounce, lk[0])
                maxf = mu * lk[0]
                lk[1] = solve_friction(A, B, rax, ray, raz, rbx, rby, rbz,
                                       t1x, t1y, t1z, ku, lk[1], maxf)
                lk[2] = solve_friction(A, B, rax, ray, raz, rbx, rby, rbz,
                                       t2x, t2y, t2z, kv, lk[2], maxf)
            for k in range(len(jdata)):
                (ia, ib, rax, ray, raz, rbx, rby, rbz, m9) = jdata[k]
                lx, ly, lz = solve_joint(bodies[ia], bodies[ib],
                                         rax, ray, raz, rbx, rby, rbz, m9,
                                         0.0, 0.0, 0.0, False)
                jl = jlam[k]
                jl[0] += lx
                jl[1] += ly
                jl[2] += lz

        # ---- pseudo-velocity (position stabilization) pass ----
        for _ in range(POS_ITERS):
            for k in range(len(rows)):
                (ia, ib, rax, ray, raz, rbx, rby, rbz,
                 nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z,
                 kn, ku, kv, bounce, pb) = rows[k]
                if pb <= 0.0:
                    continue
                plams[k][0] = solve_normal(bodies[ia], bodies[ib],
                                           rax, ray, raz, rbx, rby, rbz,
                                           nx, ny, nz, kn, pb, plams[k][0])
            for k in range(len(jdata)):
                (ia, ib, rax, ray, raz, rbx, rby, rbz, m9) = jdata[k]
                A = bodies[ia]
                B = bodies[ib]
                ex = (float(B.pos[0]) + rbx) - (float(A.pos[0]) + rax)
                ey = (float(B.pos[1]) + rby) - (float(A.pos[1]) + ray)
                ez = (float(B.pos[2]) + rbz) - (float(A.pos[2]) + raz)
                solve_joint(A, B, rax, ray, raz, rbx, rby, rbz, m9,
                            BETA_JOINT * ex / H, BETA_JOINT * ey / H,
                            BETA_JOINT * ez / H, True)

        # ---- integrate ----
        vmax2 = V_MAX * V_MAX
        for b in bodies:
            if not b.dyn:
                continue
            v = b.v
            w = b.w
            u = b.u
            pu = b.pu
            vx = v[0] + u[0]
            vy = v[1] + u[1]
            vz = v[2] + u[2]
            wx = w[0] + pu[0]
            wy = w[1] + pu[1]
            wz = w[2] + pu[2]
            s2 = vx * vx + vy * vy + vz * vz
            if s2 != s2 or s2 > vmax2:  # NaN or runaway
                v[0] = v[1] = v[2] = 0.0
                vx = vy = vz = 0.0
            s2w = wx * wx + wy * wy + wz * wz
            if s2w != s2w or s2w > vmax2:
                w[0] = w[1] = w[2] = 0.0
                wx = wy = wz = 0.0
            b.pos = b.pos + np.array([vx * H, vy * H, vz * H])
            b.q = quat_step(b.q, wx, wy, wz)

        self._last_rows = rows
        self._last_lams = lams
        self._last_jlam = jlam
        self._last_jdata = jdata
        self.step += 1
        self.time = self.step * H

    # ------------------------------------------------------------------
    # trajectory output
    # ------------------------------------------------------------------
    def _signed_distances(self):
        bodies = self.bodies
        lo, hi = self._aabbs()
        PI, PJ = self._pi, self._pj
        m = np.all(lo[PI] <= hi[PJ] + REPORT_MARGIN, axis=1) & \
            np.all(lo[PJ] <= hi[PI] + REPORT_MARGIN, axis=1)
        sds = []
        for k in np.nonzero(m)[0]:
            i = int(PI[k])
            j = int(PJ[k])
            code = self._pt[k]
            a = bodies[i]
            b = bodies[j]
            if code == 0:
                sd = sd_box_box(a, b)
            elif code == 1 or code == 2:
                sd = sphere_box_geom(a.pos if code == 1 else b.pos,
                                     a.radius if code == 1 else b.radius,
                                     b if code == 1 else a)[2]
            elif code == 3 or code == 4:
                sd = sd_capsule_box(a if code == 3 else b,
                                    b if code == 3 else a)
            elif code == 5:
                sd = sd_capsule_capsule(a, b)
            elif code == 6 or code == 7:
                sd = sd_sphere_capsule(a if code == 6 else b,
                                       b if code == 6 else a)
            else:
                dv = b.pos - a.pos
                sd = float(math.sqrt(dv @ dv)) - (a.radius + b.radius)
            if sd < REPORT_MARGIN:
                sds.append(float(sd))
        if not sds:
            return np.zeros(0, dtype=float)
        return np.asarray(sds, dtype=float)

    def _interaction_forces(self):
        out = []
        rows = self._last_rows
        lams = self._last_lams
        for k in range(len(rows)):
            (ia, ib, rax, ray, raz, rbx, rby, rbz,
             nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z,
             kn, ku, kv, bounce, pb) = rows[k]
            lk = lams[k]
            fx = (lk[0] * nx + lk[1] * t1x + lk[2] * t2x) / H
            fy = (lk[0] * ny + lk[1] * t1y + lk[2] * t2y) / H
            fz = (lk[0] * nz + lk[1] * t1z + lk[2] * t2z) / H
            out.append((fx, fy, fz))
        for jl in self._last_jlam:
            out.append((jl[0] / H, jl[1] / H, jl[2] / H))
        if not out:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(out, dtype=float)

    def _frame(self, t):
        bodies = self.bodies[: self.n_dyn]
        n = self.n_dyn
        position = np.array([b.pos for b in bodies], dtype=float).reshape(n, 3)
        velocity = np.array([b.v for b in bodies], dtype=float).reshape(n, 3)
        orientation = np.array([b.q for b in bodies], dtype=float).reshape(n, 4)
        angular = np.array([b.w for b in bodies], dtype=float).reshape(n, 3)
        mass = np.array([b.mass for b in bodies], dtype=float)
        body_id = np.arange(n, dtype=np.int32)
        external = np.outer(mass, self.gravity_vec)
        frame = StateFrame(
            time=float(t),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular,
            mass=mass,
            body_id=body_id,
            attributes={"release_time": self.release_time},
            fields={
                "contact_signed_distance": self._signed_distances(),
                "external_force": external,
                "interaction_force": self._interaction_forces(),
            },
            scalars={"contact_length_scale": self.length_scale},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = int(round(float(time) / H))
        while self.step < target:
            self._substep()
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "sequential-impulse rigid-body dynamics (split impulse)",
            "substep": H,
            "substeps": int(self.step),
            "dynamic_bodies": int(self.n_dyn),
            "joints": len(self.joints),
            "contacts_last": len(self._last_rows),
        }


def _skew_np(v):
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
