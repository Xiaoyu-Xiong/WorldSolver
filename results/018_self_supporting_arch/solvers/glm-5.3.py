"""Rigid-body frictional contact solver for the self-supporting parabolic arch.

Physical model
--------------
Each voussoir is a full 3D rigid body (6 DOF) whose convex hull geometry is
loaded from the frozen OBJ assets.  Mass properties (volume, centroid, inertia
tensor) are computed exactly from the triangle mesh with uniform density, then
scaled so the total mass matches the frozen scene masses.

Time integration
----------------
Semi-implicit (symplectic) Euler with a fixed substep ``h``.  Each substep:

1. gravity and gyroscopic torques are applied,
2. contacts are detected between convex hulls (vertex-in-hull tests against
   the mating face planes) and against the static ground plane at z = 0,
3. the contact impulses are solved with a projected Gauss-Seidel (sequential
   impulse) iteration including a Coulomb friction cone (mu = 0.6) and a
   Baumgarte penetration bias,
4. positions and quaternions are integrated.

No keyframes, no scripted motion: the arch stands (or collapses) purely as a
result of the frictional contact dynamics.
"""
from __future__ import annotations

import math
import os

import numpy as np

try:
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - fallback shim if contracts unavailable
    from dataclasses import dataclass, field
    from typing import Any, Mapping

    @dataclass
    class StateFrame:  # type: ignore
        time: float
        position: Any
        velocity: Any
        orientation: Any = None
        angular_velocity: Any = None
        mass: Any = None
        body_id: Any = None
        attributes: Mapping[str, Any] = field(default_factory=dict)
        fields: Mapping[str, Any] = field(default_factory=dict)
        scalars: Mapping[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------
# mesh utilities
# ----------------------------------------------------------------------

def _load_obj(path):
    verts: list[list[float]] = []
    faces: list[tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, dtype=float), faces


def _mesh_mass_properties(verts, faces):
    """Exact volume, centroid and inertia tensor (unit density) of a mesh."""
    volume = 0.0
    first_moment = np.zeros(3)
    second_moment = np.zeros((3, 3))
    origin = np.zeros(3)
    for a, b, c in faces:
        p0, p1, p2 = verts[a], verts[b], verts[c]
        vol = float(np.dot(p0, np.cross(p1, p2))) / 6.0
        if vol == 0.0:
            continue
        volume += vol
        tet = (p0, p1, p2, origin)
        # first moment:  int x dV = vol/4 * sum_a v_a
        first_moment += vol * 0.25 * (p0 + p1 + p2 + origin)
        # second moment (exact simplex quadrature):
        #   int x x^T dV = vol/20 * [ (sum_a v_a)(sum_b v_b)^T + sum_a v_a v_a^T ]
        s = p0 + p1 + p2 + origin
        acc = np.outer(s, s)
        for va in tet:
            acc += np.outer(va, va)
        second_moment += (vol / 20.0) * acc
    if volume == 0.0:
        raise ValueError("degenerate mesh volume")
    if volume < 0.0:
        # inward face winding: every signed contribution is negated
        volume = -volume
        first_moment = -first_moment
        second_moment = -second_moment
    centroid = first_moment / volume
    centered_second = second_moment - volume * np.outer(centroid, centroid)
    trace = float(np.trace(centered_second))
    inertia = trace * np.eye(3) - centered_second  # unit density
    return volume, centroid, inertia


def _face_planes(verts, faces, tol=1e-9):
    """Outward face planes (n, d) with n . x = d for points on the face."""
    planes = []
    seen = set()
    for a, b, c in faces:
        p0, p1, p2 = verts[a], verts[b], verts[c]
        n = np.cross(p1 - p0, p2 - p0)
        ln = float(np.linalg.norm(n))
        if ln < 1e-12:
            continue
        n = n / ln
        d = float(np.dot(n, p0))
        dists = verts @ n - d
        if dists.max() > tol and dists.min() < -tol:
            continue  # not a hull face plane
        if dists.max() > tol:
            n, d = -n, -d
        key = (round(n[0], 4), round(n[1], 4), round(n[2], 4), round(d, 4))
        if key in seen:
            continue
        seen.add(key)
        planes.append((n, d))
    if not planes:
        raise ValueError("no face planes found")
    return planes


# ----------------------------------------------------------------------
# quaternion helpers (xyzw)
# ----------------------------------------------------------------------

def _quat_to_R(q):
    x, y, z, w = q
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


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + bw * ax + (ay * bz - az * by),
            aw * by + bw * ay + (az * bx - ax * bz),
            aw * bz + bw * az + (ax * by - ay * bx),
            aw * bw - (ax * bx + ay * by + az * bz),
        ]
    )


# ----------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------

class BenchmarkSolver:
    """Impulse-based rigid-body solver for the 15-voussoir parabolic arch."""

    SUBSTEP_H = 0.001
    PGS_ITERATIONS = 25
    BAUMGARTE_BETA = 0.2
    SLOP = 1.0e-4
    DETECT_TOL = 3.0e-3
    REPORT_TOL = 5.0e-3
    BIAS_CAP = 1.0

    def __init__(self):
        self._ready = False
        self._n = 0
        self._steps = 0

    # -- setup ---------------------------------------------------------

    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        self.seed = int(seed)

        grav = cfg.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.asarray(grav, dtype=float).reshape(3)

        friction = cfg.get("friction", {})
        if isinstance(friction, dict):
            mu = friction.get("coefficient", 0.6)
        else:
            mu = 0.6
        self.mu = float(mu)

        frame0 = scene.initial_state()
        n = int(len(frame0.position))
        self._n = n

        asset_dir = self._find_asset_dir()
        meshes = []
        for i in range(n):
            path = os.path.join(asset_dir, "voussoir_%02d.obj" % i)
            verts, faces = _load_obj(path)
            meshes.append((verts, faces))

        # body state arrays; index n is the static ground body
        nb = n + 1
        self.pos = np.zeros((nb, 3))
        self.vel = np.zeros((nb, 3))
        self.quat = np.zeros((nb, 4))
        self.omega = np.zeros((nb, 3))
        self.mass = np.zeros(nb)
        self.inv_mass = np.zeros(nb)
        self.I_body = np.zeros((nb, 3, 3))
        self.inv_I_body = np.zeros((nb, 3, 3))
        self.verts_local = [None] * n
        self.face_normals = [None] * n
        self.face_offsets = [None] * n
        self.bound_radius = np.zeros(n)

        self.pos[:n] = np.asarray(frame0.position, dtype=float)
        self.vel[:n] = np.asarray(frame0.velocity, dtype=float)
        self.quat[:n, :] = [0.0, 0.0, 0.0, 1.0]
        q0 = getattr(frame0, "orientation", None)
        if q0 is not None:
            q0 = np.asarray(q0, dtype=float)
            if q0.shape == (n, 4):
                self.quat[:n] = q0
        w0 = getattr(frame0, "angular_velocity", None)
        if w0 is not None:
            w0 = np.asarray(w0, dtype=float)
            if w0.shape == (n, 3):
                self.omega[:n] = w0
        self.mass[:n] = np.asarray(frame0.mass, dtype=float)
        self.inv_mass[:n] = 1.0 / self.mass[:n]
        self.body_id = np.arange(n, dtype=int)

        for i in range(n):
            verts, faces = meshes[i]
            volume, centroid, inertia_unit = _mesh_mass_properties(verts, faces)
            density = self.mass[i] / volume
            self.I_body[i] = density * inertia_unit
            self.inv_I_body[i] = np.linalg.inv(self.I_body[i])
            local = verts - centroid  # COM-centred local frame
            self.verts_local[i] = local
            # the scene translation places the asset origin; the rigid body
            # position is the true centre of mass, so the assembled world
            # geometry (translation + raw verts) is reproduced exactly.
            self.pos[i] += centroid
            planes = _face_planes(local, faces)
            self.face_normals[i] = np.array([p[0] for p in planes])
            self.face_offsets[i] = np.array([p[1] for p in planes])
            self.bound_radius[i] = float(
                np.max(np.linalg.norm(local, axis=1))
            ) + 1.0e-6

        # ground body (index n): static, zero inverse mass/inertia
        self.pos[n] = 0.0
        self.quat[n] = [0.0, 0.0, 0.0, 1.0]

        self.pos0 = self.pos[:n].copy()  # frozen initial configuration

        # frozen normalization scales
        span = float(self.pos0[:, 0].max() - self.pos0[:, 0].min())
        if not (span > 0.0 and np.isfinite(span)):
            span = 1.0
        self.constraint_scale = span
        thickness = 0.0
        for i in range(n):
            local = self.verts_local[i]
            thickness += float(local[:, 2].max() - local[:, 2].min())
        thickness /= max(n, 1)
        if not (thickness > 0.0 and np.isfinite(thickness)):
            thickness = 0.5
        self.contact_length_scale = thickness

        self.external_force = self.mass[:n, None] * self.gravity[None, :]

        self._steps = 0
        self._max_pen = 0.0
        self._contacts_last = 0
        self._ready = True

    @staticmethod
    def _find_asset_dir():
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(here, os.pardir, "assets"),
            os.path.join(os.getcwd(), "assets"),
            here,
        ]
        for cand in candidates:
            if os.path.isfile(os.path.join(cand, "voussoir_00.obj")):
                return os.path.abspath(cand)
        raise FileNotFoundError("voussoir asset directory not found")

    # -- contact detection ---------------------------------------------

    def _world_frames(self):
        n = self._n
        R = np.zeros((n, 3, 3))
        for i in range(n):
            R[i] = _quat_to_R(self.quat[i])
        verts_world = np.einsum("nij,nkj->nki", R, self.verts_local) + self.pos[
            :n, None, :
        ]
        return R, verts_world

    def _detect(self, R, verts_world, tol):
        n = self._n
        ground = n
        contacts = []  # (i_static_side, j_moving_side, point, normal, gap)

        # ground plane z = 0
        gz = np.array([0.0, 0.0, 1.0])
        for i in range(n):
            for k in range(verts_world.shape[1]):
                z = float(verts_world[i, k, 2])
                if z <= tol:
                    contacts.append(
                        (ground, i, verts_world[i, k].copy(), gz.copy(), z)
                    )

        # body-body pairs (broadphase on bounding spheres)
        for i in range(n):
            for j in range(i + 1, n):
                diff = self.pos[j] - self.pos[i]
                rsum = self.bound_radius[i] + self.bound_radius[j] + tol
                if float(diff @ diff) > rsum * rsum:
                    continue
                seen_points = set()
                self._vertex_face(i, j, verts_world[i], R[j], tol, contacts, seen_points)
                self._vertex_face(j, i, verts_world[j], R[i], tol, contacts, seen_points)
        return contacts

    def _vertex_face(self, src, dst, src_verts_world, R_dst, tol, contacts, seen):
        """Test src vertices against dst hull; contact normal points dst->src."""
        rel = src_verts_world - self.pos[dst]
        local = rel @ R_dst  # R^T p
        normals = self.face_normals[dst]
        offsets = self.face_offsets[dst]
        s = local @ normals.T - offsets  # signed distance per face
        for k in range(s.shape[0]):
            row = s[k]
            kbest = int(np.argmax(row))
            gap = float(row[kbest])
            if gap > tol:
                continue
            point = src_verts_world[k]
            key = (round(float(point[0]), 4), round(float(point[1]), 4),
                   round(float(point[2]), 4))
            if key in seen:
                continue
            seen.add(key)
            normal_world = R_dst @ normals[kbest]
            # normal points from dst toward src
            contacts.append((dst, src, point.copy(), normal_world, gap))

    # -- dynamics ------------------------------------------------------

    def _substep(self):
        n = self._n
        h = self.SUBSTEP_H
        mu = self.mu

        R = np.zeros((n, 3, 3))
        inv_I_world = np.zeros((n + 1, 3, 3))
        for i in range(n):
            R[i] = _quat_to_R(self.quat[i])
            inv_I_world[i] = R[i] @ self.inv_I_body[i] @ R[i].T

        # gravity + gyroscopic torque
        self.vel[:n] += h * self.gravity[None, :]
        for i in range(n):
            Iw = R[i] @ self.I_body[i] @ R[i].T
            angular_momentum = Iw @ self.omega[i]
            gyro = -np.cross(self.omega[i], angular_momentum)
            self.omega[i] += h * (inv_I_world[i] @ gyro)

        R_full, verts_world = self._world_frames()
        contacts = self._detect(R_full, verts_world, self.DETECT_TOL)
        self._contacts_last = len(contacts)

        if contacts:
            self._solve_contacts(contacts, inv_I_world)

        # integrate
        self.pos[:n] += h * self.vel[:n]
        for i in range(n):
            wq = np.array(
                [self.omega[i, 0], self.omega[i, 1], self.omega[i, 2], 0.0]
            )
            dq = (0.5 * h) * _quat_mul(wq, self.quat[i])
            q = self.quat[i] + dq
            nq = float(np.linalg.norm(q))
            if nq < 1e-12:
                q = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                q = q / nq
            self.quat[i] = q

        # safety clamps (keep state finite and bounded)
        if not np.isfinite(self.pos[:n]).all() or not np.isfinite(self.vel[:n]).all():
            raise RuntimeError("non-finite rigid body state")
        np.clip(self.vel[:n], -50.0, 50.0, out=self.vel[:n])
        np.clip(self.omega[:n], -100.0, 100.0, out=self.omega[:n])

    def _solve_contacts(self, contacts, inv_I_world):
        n = self._n
        ground = n
        h = self.SUBSTEP_H
        mu = self.mu
        beta = self.BAUMGARTE_BETA
        slop = self.SLOP
        bias_cap = self.BIAS_CAP

        m = len(contacts)
        bi = np.array([c[0] for c in contacts], dtype=int)
        bj = np.array([c[1] for c in contacts], dtype=int)
        points = np.array([c[2] for c in contacts], dtype=float)
        normals = np.array([c[3] for c in contacts], dtype=float)
        gaps = np.array([c[4] for c in contacts], dtype=float)

        ri = points - self.pos[bi]
        rj = points - self.pos[bj]
        imi = self.inv_mass[bi]
        imj = self.inv_mass[bj]
        Ii = inv_I_world[bi]  # (m,3,3)
        Ij = inv_I_world[bj]

        def effective_mass(dirs):
            # k = 1/mi + 1/mj + d . ((Ii (ri x d)) x ri + (Ij (rj x d)) x rj)
            ai = np.cross(ri, dirs)
            aj = np.cross(rj, dirs)
            wi = np.einsum("mij,mj->mi", Ii, ai)
            wj = np.einsum("mij,mj->mi", Ij, aj)
            ti = np.cross(wi, ri)
            tj = np.cross(wj, rj)
            return imi + imj + np.einsum("mi,mi->m", dirs, ti + tj)

        kn = effective_mass(normals)
        kn = np.where(kn > 1e-12, kn, 1e12)

        # tangent frame
        helper = np.zeros_like(normals)
        helper[:, 2] = 1.0
        helper[(np.abs(normals[:, 2]) > 0.9), 0] = 1.0
        helper[(np.abs(normals[:, 2]) > 0.9), 2] = 0.0
        t1 = np.cross(normals, helper)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
        t2 = np.cross(normals, t1)
        k1 = effective_mass(t1)
        k1 = np.where(k1 > 1e-12, k1, 1e12)
        k2 = effective_mass(t2)
        k2 = np.where(k2 > 1e-12, k2, 1e12)

        pen = np.maximum(0.0, -gaps - slop)
        bias = np.minimum(beta / h * pen, bias_cap)
        self._max_pen = max(self._max_pen, float(pen.max()) if m else 0.0)

        # flatten body state to python lists for a fast PGS inner loop
        vx = self.vel[:, 0].tolist()
        vy = self.vel[:, 1].tolist()
        vz = self.vel[:, 2].tolist()
        wx = self.omega[:, 0].tolist()
        wy = self.omega[:, 1].tolist()
        wz = self.omega[:, 2].tolist()
        im_list = self.inv_mass.tolist()
        II = []
        for b in range(n + 1):
            M = inv_I_world[b] if b < n else np.zeros((3, 3))
            II.append((M[0, 0], M[0, 1], M[0, 2],
                       M[1, 0], M[1, 1], M[1, 2],
                       M[2, 0], M[2, 1], M[2, 2]))

        rix = ri[:, 0].tolist(); riy = ri[:, 1].tolist(); riz = ri[:, 2].tolist()
        rjx = rj[:, 0].tolist(); rjy = rj[:, 1].tolist(); rjz = rj[:, 2].tolist()
        nx = normals[:, 0].tolist(); ny = normals[:, 1].tolist(); nz = normals[:, 2].tolist()
        t1x = t1[:, 0].tolist(); t1y = t1[:, 1].tolist(); t1z = t1[:, 2].tolist()
        t2x = t2[:, 0].tolist(); t2y = t2[:, 1].tolist(); t2z = t2[:, 2].tolist()
        kn_l = kn.tolist(); k1_l = k1.tolist(); k2_l = k2.tolist()
        bias_l = bias.tolist()
        bi_l = bi.tolist(); bj_l = bj.tolist()

        # contact record layout (mutable list):
        # [ i, j, rix, riy, riz, rjx, rjy, rjz, nx, ny, nz,
        #   t1x, t1y, t1z, t2x, t2y, t2z, kn, k1, k2, bias, lam ]
        C = []
        for c in range(m):
            C.append([bi_l[c], bj_l[c], rix[c], riy[c], riz[c],
                      rjx[c], rjy[c], rjz[c],
                      nx[c], ny[c], nz[c],
                      t1x[c], t1y[c], t1z[c], t2x[c], t2y[c], t2z[c],
                      kn_l[c], k1_l[c], k2_l[c], bias_l[c], 0.0])

        for _ in range(self.PGS_ITERATIONS):
            for rec in C:
                i = rec[0]; j = rec[1]
                rix_ = rec[2]; riy_ = rec[3]; riz_ = rec[4]
                rjx_ = rec[5]; rjy_ = rec[6]; rjz_ = rec[7]
                nx_ = rec[8]; ny_ = rec[9]; nz_ = rec[10]
                t1x_ = rec[11]; t1y_ = rec[12]; t1z_ = rec[13]
                t2x_ = rec[14]; t2y_ = rec[15]; t2z_ = rec[16]
                kn_ = rec[17]; k1_ = rec[18]; k2_ = rec[19]
                bias_ = rec[20]
                lam = rec[21]

                imi_ = im_list[i]
                imj_ = im_list[j]
                Ii_ = II[i]
                Ij_ = II[j]

                # point velocities
                vix = vx[i] + (wy[i] * riz_ - wz[i] * riy_)
                viy = vy[i] + (wz[i] * rix_ - wx[i] * riz_)
                viz = vz[i] + (wx[i] * riy_ - wy[i] * rix_)
                vjx = vx[j] + (wy[j] * rjz_ - wz[j] * rjy_)
                vjy = vy[j] + (wz[j] * rjx_ - wx[j] * rjz_)
                vjz = vz[j] + (wx[j] * rjy_ - wy[j] * rjx_)

                dvx = vjx - vix
                dvy = vjy - viy
                dvz = vjz - viz

                # normal impulse
                vn = dvx * nx_ + dvy * ny_ + dvz * nz_
                dl = (-vn + bias_) / kn_
                new_lam = lam + dl
                if new_lam < 0.0:
                    new_lam = 0.0
                dl = new_lam - lam
                rec[21] = new_lam
                if dl != 0.0:
                    px = dl * nx_; py = dl * ny_; pz = dl * nz_
                    vx[i] -= px * imi_; vy[i] -= py * imi_; vz[i] -= pz * imi_
                    vx[j] += px * imj_; vy[j] += py * imj_; vz[j] += pz * imj_
                    # torque = r x P ; w -= Iinv * torque
                    tix = riy_ * pz - riz_ * py
                    tiy = riz_ * px - rix_ * pz
                    tiz = rix_ * py - riy_ * px
                    wx[i] -= Ii_[0] * tix + Ii_[1] * tiy + Ii_[2] * tiz
                    wy[i] -= Ii_[3] * tix + Ii_[4] * tiy + Ii_[5] * tiz
                    wz[i] -= Ii_[6] * tix + Ii_[7] * tiy + Ii_[8] * tiz
                    tjx = rjy_ * pz - rjz_ * py
                    tjy = rjz_ * px - rjx_ * pz
                    tjz = rjx_ * py - rjy_ * px
                    wx[j] += Ij_[0] * tjx + Ij_[1] * tjy + Ij_[2] * tjz
                    wy[j] += Ij_[3] * tjx + Ij_[4] * tjy + Ij_[5] * tjz
                    wz[j] += Ij_[6] * tjx + Ij_[7] * tjy + Ij_[8] * tjz

                # friction impulse (Coulomb cone around the normal impulse)
                if new_lam > 0.0:
                    vix = vx[i] + (wy[i] * riz_ - wz[i] * riy_)
                    viy = vy[i] + (wz[i] * rix_ - wx[i] * riz_)
                    viz = vz[i] + (wx[i] * riy_ - wy[i] * rix_)
                    vjx = vx[j] + (wy[j] * rjz_ - wz[j] * rjy_)
                    vjy = vy[j] + (wz[j] * rjx_ - wx[j] * rjz_)
                    vjz = vz[j] + (wx[j] * rjy_ - wy[j] * rjx_)
                    dvx = vjx - vix
                    dvy = vjy - viy
                    dvz = vjz - viz
                    vt1 = dvx * t1x_ + dvy * t1y_ + dvz * t1z_
                    vt2 = dvx * t2x_ + dvy * t2y_ + dvz * t2z_
                    jt1 = -vt1 / k1_
                    jt2 = -vt2 / k2_
                    max_f = mu * new_lam
                    mag2 = jt1 * jt1 + jt2 * jt2
                    if mag2 > max_f * max_f and mag2 > 0.0:
                        sc = max_f / math.sqrt(mag2)
                        jt1 *= sc
                        jt2 *= sc
                    px = jt1 * t1x_ + jt2 * t2x_
                    py = jt1 * t1y_ + jt2 * t2y_
                    pz = jt1 * t1z_ + jt2 * t2z_
                    if px != 0.0 or py != 0.0 or pz != 0.0:
                        vx[i] -= px * imi_; vy[i] -= py * imi_; vz[i] -= pz * imi_
                        vx[j] += px * imj_; vy[j] += py * imj_; vz[j] += pz * imj_
                        tix = riy_ * pz - riz_ * py
                        tiy = riz_ * px - rix_ * pz
                        tiz = rix_ * py - riy_ * px
                        wx[i] -= Ii_[0] * tix + Ii_[1] * tiy + Ii_[2] * tiz
                        wy[i] -= Ii_[3] * tix + Ii_[4] * tiy + Ii_[5] * tiz
                        wz[i] -= Ii_[6] * tix + Ii_[7] * tiy + Ii_[8] * tiz
                        tjx = rjy_ * pz - rjz_ * py
                        tjy = rjz_ * px - rjx_ * pz
                        tjz = rjx_ * py - rjy_ * px
                        wx[j] += Ij_[0] * tjx + Ij_[1] * tjy + Ij_[2] * tjz
                        wy[j] += Ij_[3] * tjx + Ij_[4] * tjy + Ij_[5] * tjz
                        wz[j] += Ij_[6] * tjx + Ij_[7] * tjy + Ij_[8] * tjz

        # write back (ground row n was never modified: inverse mass/inertia 0)
        self.vel[:, 0] = vx
        self.vel[:, 1] = vy
        self.vel[:, 2] = vz
        self.omega[:, 0] = wx
        self.omega[:, 1] = wy
        self.omega[:, 2] = wz

    # -- API -----------------------------------------------------------

    def advance_to(self, time: float) -> StateFrame:
        if not self._ready:
            raise RuntimeError("solver not initialized")
        t = float(time)
        if not np.isfinite(t) or t < 0.0:
            raise ValueError("advance_to requires a finite non-negative time")
        target = int(round(t / self.SUBSTEP_H))
        while self._steps < target:
            self._substep()
            self._steps += 1

        n = self._n
        R, verts_world = self._world_frames()
        contacts = self._detect(R, verts_world, self.REPORT_TOL)
        gaps = np.array([c[4] for c in contacts], dtype=float)
        if gaps.size == 0:
            gaps = np.zeros(0, dtype=float)

        frame = StateFrame(
            time=t,
            position=self.pos[:n].copy(),
            velocity=self.vel[:n].copy(),
            orientation=self.quat[:n].copy(),
            angular_velocity=self.omega[:n].copy(),
            mass=self.mass[:n].copy(),
            body_id=self.body_id.copy(),
            fields={
                "constraint_value": self.pos[:n].copy(),
                "constraint_target": self.pos0.copy(),
                "contact_signed_distance": gaps,
                "external_force": self.external_force.copy(),
            },
            scalars={
                "constraint_scale": float(self.constraint_scale),
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        drift = 0.0
        if self._ready and self._n:
            drift = float(
                np.max(np.linalg.norm(self.pos[: self._n] - self.pos0, axis=1))
            )
        return {
            "num_bodies": int(self._n),
            "substep": float(self.SUBSTEP_H),
            "pgs_iterations": int(self.PGS_ITERATIONS),
            "friction_coefficient": float(self.mu),
            "simulated_substeps": int(self._steps),
            "simulated_time": float(self._steps * self.SUBSTEP_H),
            "contacts_last_substep": int(self._contacts_last),
            "max_penetration": float(self._max_pen),
            "max_block_drift": drift,
        }
