"""Impulse-based rigid-body solver for the circular domino chain-reaction scene.

Genuine physical simulation (no keyframes, no scripted motion, no external
engines):

  * semi-implicit (symplectic) Euler time stepping of rigid-body motion,
  * unilateral frictional contacts between boxes and between boxes and the
    static floor, generated from the frozen box geometry (sample points inside
    the neighbouring box with nearest-face normals),
  * sequential-impulse (projected Gauss-Seidel) contact resolution with Coulomb
    friction and restitution,
  * a separate position-projection pass (split-impulse style) that removes
    penetration without injecting kinetic energy.

All ordering is fixed, therefore the trajectory is deterministic.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_GRAVITY = np.array([0.0, 0.0, -9.81])

_H = 0.0025             # fixed inner time step [s]
_ITERS = 6              # Gauss-Seidel sweeps per step
_SLOP = 5.0e-4          # allowed resting penetration [m]
_BAUMGARTE = 0.25       # fraction of penetration removed per step
_MAX_CORR = 4.0e-3      # cap on positional correction per step [m]
_MARGIN = 0.02          # near-contact activation distance [m]
_REST_THRESHOLD = 1.0   # approach speed above which restitution applies [m/s]
_MAX_CONTACTS_PAIR = 8
_MAX_VEL = 40.0
_MAX_OMEGA = 150.0


def _quat_to_mat(q):
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
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


def _mat_to_quat(r):
    tr = float(r[0, 0] + r[1, 1] + r[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _surface_samples(half):
    """Fixed deterministic surface sample set of a box (corners, face centers,
    edge midpoints) used for contact point generation."""
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append((sx * half[0], sy * half[1], sz * half[2]))
    for a in range(3):
        for s in (-1.0, 1.0):
            p = [0.0, 0.0, 0.0]
            p[a] = s * half[a]
            pts.append(tuple(p))
    for a in range(3):
        for b in range(a + 1, 3):
            for sa in (-1.0, 1.0):
                for sb in (-1.0, 1.0):
                    p = [0.0, 0.0, 0.0]
                    p[a] = sa * half[a]
                    p[b] = sb * half[b]
                    pts.append(tuple(p))
    out = np.array(pts, dtype=float)
    return np.unique(np.round(out, 12), axis=0)


class BenchmarkSolver:
    """Rigid-body solver implementing the Solver contract."""

    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", {}) or {}
        contact = config.get("contact", {}) or {}
        bodies = config.get("bodies", []) or []

        state = scene.initial_state()
        self.pos = np.array(state.position, dtype=float)
        self.ori = np.array(state.orientation, dtype=float)
        for k in range(len(self.ori)):
            n = np.linalg.norm(self.ori[k])
            self.ori[k] = self.ori[k] / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
        self.vel = np.array(state.velocity, dtype=float)
        if state.angular_velocity is not None:
            self.omega = np.array(state.angular_velocity, dtype=float)
        else:
            self.omega = np.zeros_like(self.pos)
        self.mass = np.array(state.mass, dtype=float)
        self.body_id = np.array(state.body_id, dtype=np.int32)
        self.n = int(len(self.pos))
        self.inv_mass = np.where(self.mass > 0.0, 1.0 / np.maximum(self.mass, 1e-12), 0.0)

        geom = config.get("scene_geometry", {}) or {}
        half = None
        for item in geom.get("objects", []) or []:
            if item.get("id") == "domino_0":
                he = item.get("geometry", {}).get("half_extents")
                if he is not None:
                    half = np.array(he, dtype=float)
        if half is None:
            half = np.array([0.07, 0.2, 0.55], dtype=float)
        self.half = half
        self.samples = _surface_samples(self.half)
        self.n_samples = int(len(self.samples))

        # Body-frame inertia of a solid box.
        hx, hy, hz = float(self.half[0]), float(self.half[1]), float(self.half[2])
        diag = np.array(
            [
                (hy * hy + hz * hz) / 3.0,
                (hx * hx + hz * hz) / 3.0,
                (hx * hx + hy * hy) / 3.0,
            ]
        )
        self.inv_inertia_body = 1.0 / (self.mass[:, None] * diag[None, :])

        self.friction = float(contact.get("friction", 0.55))
        self.restitution = float(contact.get("restitution", 0.12))
        self.length_scale = float(contact.get("length_scale", 0.02))

        grav = config.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.asarray(grav, dtype=float).reshape(3)

        self.radius = float(np.linalg.norm(self.half))
        self.R = np.stack([_quat_to_mat(self.ori[i]) for i in range(self.n)], axis=0)
        self.inv_inertia_world = np.zeros((self.n, 3, 3))
        self.rot_samples = np.zeros((self.n, self.n_samples, 3))
        self._update_rotations()

        self.time = 0.0
        self.steps = 0
        self.h = _H
        self.contact_gap = np.zeros(0, dtype=float)
        self.impulse = np.zeros((self.n, 3), dtype=float)
        self.last_contacts = 0
        self.max_penetration = 0.0
        self.max_speed = 0.0
        self._vn_after = np.zeros(0)
        self.time = 0.0

    # ------------------------------------------------------------------
    def _update_rotations(self):
        self.rot_samples = np.einsum("nij,nj->ni", self.R, self.samples)
        # world inverse inertia: R diag(invI) R^T
        self.inv_inertia_world = np.einsum(
            "nij,nj,nkj->nik", self.R, self.inv_inertia_body, self.R
        )

    # ------------------------------------------------------------------
    def _detect(self):
        """Generate the deterministic contact list for the current positions.

        Each contact is (body_i, body_j or -1 for the static floor, point,
        unit normal acting on body_i, signed gap). Negative gap = penetration.
        """
        pos = self.pos
        R = self.R
        pts = self.rot_samples + pos[:, None, :]
        half = self.half
        ci, cj, cp, cn, cg = [], [], [], [], []

        # --- static floor (top surface z = 0, extent from the frozen floor box)
        floor_x, floor_y = 3.2, 2.2
        for i in range(self.n):
            z = pts[i, :, 2]
            on_floor = (
                (np.abs(pts[i, :, 0]) <= floor_x)
                & (np.abs(pts[i, :, 1]) <= floor_y)
                & (z < _MARGIN)
            )
            idx = np.nonzero(on_floor)[0]
            if idx.size == 0:
                continue
            order = idx[np.argsort(z[idx], kind="stable")][:4]
            for k in order:
                ci.append(i)
                cj.append(-1)
                cp.append(pts[i, k].copy())
                cn.append(np.array([0.0, 0.0, 1.0]))
                cg.append(float(z[k]))

        # --- box / box contacts (broad phase by bounding spheres)
        reach = 2.0 * self.radius + _MARGIN
        for i in range(self.n):
            for j in range(i + 1, self.n):
                d = pos[j] - pos[i]
                if float(d @ d) > reach * reach:
                    continue
                found = []
                # samples of i inside j
                L = (pts[i] - pos[j]) @ R[j]
                rem = half[None, :] - np.abs(L)
                inside = np.all(rem >= 0.0, axis=1)
                for k in np.nonzero(inside)[0]:
                    rk = rem[k]
                    a = int(np.argmin(rk))
                    sgn = 1.0 if L[k, a] >= 0.0 else -1.0
                    nrm = sgn * R[j][:, a]
                    found.append((-float(rk[a]), pts[i, k].copy(), nrm))
                # samples of j inside i
                L = (pts[j] - pos[i]) @ R[i]
                rem = half[None, :] - np.abs(L)
                inside = np.all(rem >= 0.0, axis=1)
                for k in np.nonzero(inside)[0]:
                    rk = rem[k]
                    a = int(np.argmin(rk))
                    sgn = 1.0 if L[k, a] >= 0.0 else -1.0
                    nrm = -sgn * R[i][:, a]
                    found.append((-float(rk[a]), pts[j, k].copy(), nrm))
                # near contacts (separated but inside the activation margin)
                if len(found) < 4:
                    for (pa, other_R, other_c, flip) in (
                        (pts[i], R[j], pos[j], 1.0),
                        (pts[j], R[i], pos[i], -1.0),
                    ):
                        L = (pa - other_c) @ other_R
                        rem = half[None, :] - np.abs(L)
                        outside = ~np.all(rem >= 0.0, axis=1)
                        out = np.sqrt(np.sum(np.maximum(-rem, 0.0) ** 2, axis=1))
                        near = (out < _MARGIN) & outside
                        for k in np.nonzero(near)[0]:
                            dvec = np.maximum(-rem[k], 0.0)
                            a = int(np.argmax(dvec))
                            sgn = 1.0 if L[k, a] >= 0.0 else -1.0
                            nrm = flip * sgn * other_R[:, a]
                            found.append((float(out[k]), pa[k].copy(), nrm))
                if not found:
                    continue
                found.sort(key=lambda item: item[0])
                for gap, point, nrm in found[:_MAX_CONTACTS_PAIR]:
                    ci.append(i)
                    cj.append(j)
                    cp.append(point)
                    cn.append(nrm)
                    cg.append(gap)

        if not ci:
            empty = np.zeros((0, 3))
            return (
                np.zeros(0, dtype=int),
                np.zeros(0, dtype=int),
                empty,
                empty,
                np.zeros(0),
            )
        return (
            np.asarray(ci, dtype=int),
            np.asarray(cj, dtype=int),
            np.asarray(cp, dtype=float),
            np.asarray(cn, dtype=float),
            np.asarray(cg, dtype=float),
        )

    # ------------------------------------------------------------------
    def _solve(self, ci, cj, cp, cn, cg, h):
        """Sequential-impulse contact resolution with Coulomb friction."""
        im = self.inv_mass
        invIw = self.inv_inertia_world
        vel = self.vel
        om = self.omega
        imp = self.impulse
        imp[:] = 0.0
        k_count = len(ci)
        has_j = cj >= 0
        cj_safe = np.where(has_j, cj, 0)
        ri = cp - self.pos[ci]
        rj = cp - self.pos[cj_safe]

        rxi = np.cross(ri, cn)
        kn = im[ci] + np.einsum("kj,kji,ki->k", rxi, invIw[ci], rxi)
        if np.any(has_j):
            rxj = np.cross(rj, cn)
            kn = kn + np.where(
                has_j,
                im[cj_safe] + np.einsum("kj,kji,ki->k", rxj, invIw[cj_safe], rxj),
                0.0,
            )
        kn = np.maximum(kn, 1e-9)

        # approach velocity before the solve (for restitution)
        vn0 = np.zeros(k_count)
        for k in range(k_count):
            i = int(ci[k])
            vi = vel[i] + np.cross(om[i], ri[k])
            if has_j[k]:
                j = int(cj[k])
                vj = vel[j] + np.cross(om[j], rj[k])
                vn0[k] = float((vi - vj) @ cn[k])
            else:
                vn0[k] = float(vi @ cn[k])

        Pn = np.zeros(k_count)
        mu = self.friction
        e_global = self.restitution
        for _ in range(_ITERS):
            for k in range(k_count):
                i = int(ci[k])
                j = int(cj[k]) if has_j[k] else -1
                nrm = cn[k]
                rik = ri[k]
                vi = vel[i] + np.cross(om[i], rik)
                if j >= 0:
                    rjk = rj[k]
                    vrel = vi - (vel[j] + np.cross(om[j], rjk))
                else:
                    rjk = None
                    vrel = vi
                vn = float(vrel @ nrm)
                e = e_global if vn0[k] < -_REST_THRESHOLD else 0.0
                lam = -(1.0 + e) * vn / kn[k]
                new = Pn[k] + lam
                if new < 0.0:
                    new = 0.0
                dlam = new - Pn[k]
                Pn[k] = new
                if dlam != 0.0:
                    impulse_vec = dlam * nrm
                    vel[i] += impulse_vec * im[i]
                    om[i] += invIw[i] @ np.cross(rik, impulse_vec)
                    imp[i] += impulse_vec
                    if j >= 0:
                        vel[j] -= impulse_vec * im[j]
                        om[j] -= invIw[j] @ np.cross(rjk, impulse_vec)
                        imp[j] -= impulse_vec

                # tangential (Coulomb friction) impulse
                vi = vel[i] + np.cross(om[i], rik)
                if j >= 0:
                    vrel = vi - (vel[j] + np.cross(om[j], rjk))
                else:
                    vrel = vi
                vn2 = float(vrel @ nrm)
                vt = vrel - vn2 * nrm
                vtn = float(np.linalg.norm(vt))
                if vtn > 1e-9 and Pn[k] > 0.0:
                    tdir = vt / vtn
                    rti = np.cross(rik, tdir)
                    kt = im[i] + float(rti @ (invIw[i] @ rti))
                    if j >= 0:
                        rtj = np.cross(rjk, tdir)
                        kt += im[j] + float(rtj @ (invIw[j] @ rtj))
                    lam_t = -vtn / max(kt, 1e-9)
                    limit = mu * Pn[k]
                    if lam_t < -limit:
                        lam_t = -limit
                    elif lam_t > limit:
                        lam_t = limit
                    if lam_t != 0.0:
                        fvec = lam_t * tdir
                        vel[i] += fvec * im[i]
                        om[i] += invIw[i] @ np.cross(rik, fvec)
                        imp[i] += fvec
                        if j >= 0:
                            vel[j] -= fvec * im[j]
                            om[j] -= invIw[j] @ np.cross(rjk, fvec)
                            imp[j] -= fvec

        # relative normal velocity after the solve (used by position projection)
        vna = np.zeros(k_count)
        for k in range(k_count):
            i = int(ci[k])
            vi = vel[i] + np.cross(om[i], ri[k])
            if has_j[k]:
                j = int(cj[k])
                vna[k] = float((vi - (vel[j] + np.cross(om[j], rj[k]))) @ cn[k])
            else:
                vna[k] = float(vi @ cn[k])
        self._vn_after = vna
        self._contacts_cache = (ci, cj, cp, cn, cg)

    # ------------------------------------------------------------------
    def _project(self, ci, cj, cp, cn, cg, h):
        """Split-impulse style position projection: removes penetration without
        changing any velocity (so the momentum budget stays consistent)."""
        im = self.inv_mass
        w = im.copy()
        beta = 0.2
        max_corr = 3.0e-3
        for _ in range(2):
            for k in range(len(ci)):
                vna = float(self._vn_after[k]) if k < len(self._vn_after) else 0.0
                gap_now = cg[k] + vna * h
                pen = -gap_now
                if pen <= _SLOP:
                    continue
                corr = beta * (pen - _SLOP)
                if corr > max_corr:
                    corr = max_corr
                i = int(ci[k])
                j = int(cj[k])
                wi = w[i]
                wj = w[j] if j >= 0 else 0.0
                tot = wi + wj
                if tot <= 0.0:
                    continue
                shift = cn[k] * corr
                self.pos[i] += shift * (wi / tot)
                if j >= 0:
                    self.pos[j] -= shift * (wj / tot)

    # ------------------------------------------------------------------
    def _step(self, h):
        ci, cj, cp, cn, cg = self._detect()
        self.last_contacts = int(len(ci))
        self.contact_gap = np.asarray(cg, dtype=float)

        # gravity (the only external force)
        self.vel += self.gravity[None, :] * h

        if len(ci):
            self._solve(ci, cj, cp, cn, cg, h)
            self._project(ci, cj, cp, cn, cg, h)

        self.pos += self.vel * h

        for i in range(self.n):
            w = self.omega[i]
            q = self.ori[i]
            dq = _quat_mul(np.array([w[0], w[1], w[2], 0.0]), q)
            q = q + 0.5 * dq * h
            nrm = float(np.linalg.norm(q))
            if nrm < 1e-12:
                q = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                q = q / nrm
            self.ori[i] = q

        self.R = np.stack([_quat_to_mat(self.ori[i]) for i in range(self.n)], axis=0)
        self._update_rotations()

        if not (np.isfinite(self.pos).all() and np.isfinite(self.vel).all()):
            self.vel[:] = 0.0
            self.omega[:] = 0.0
            self.pos[:] = np.nan_to_num(self.pos, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(self.omega).all():
            self.omega[:] = 0.0

        speeds = np.linalg.norm(self.vel, axis=1)
        fast = speeds > _MAX_VEL
        if np.any(fast):
            self.vel[fast] *= (_MAX_VEL / speeds[fast])[:, None]
        spins = np.linalg.norm(self.omega, axis=1)
        wild = spins > _MAX_OMEGA
        if np.any(wild):
            self.omega[wild] *= (_MAX_OMEGA / spins[wild])[:, None]

        self.max_speed = max(self.max_speed, float(np.max(np.linalg.norm(self.vel, axis=1))))
        if self.contact_gap.size:
            self.max_penetration = max(self.max_penetration, float(-np.min(self.contact_gap)))

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target):
            target = self.time
        target = max(target, 0.0)
        n_steps = int(round(target / self.h))
        guard = 0
        while self.steps < n_steps and guard < 5_000_000:
            self._step(self.h)
            self.steps += 1
            guard += 1
        self.time = self.steps * self.h
        return self._frame(target)

    # ------------------------------------------------------------------
    def _frame(self, time: float):
        ci, cj, cp, cn, cg = self._detect()
        gaps = np.asarray(cg, dtype=float).reshape(-1)
        force = self.impulse / self.h if self.steps > 0 else np.zeros((self.n, 3))
        external = self.mass[:, None] * self.gravity[None, :]
        return StateFrame(
            float(time),
            self.pos.copy(),
            self.vel.copy(),
            orientation=self.ori.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "contact_signed_distance": gaps,
                "external_force": external,
                "interaction_force": force.copy(),
            },
            scalars={"contact_length_scale": self.length_scale},
        )

    # ------------------------------------------------------------------
    def diagnostics(self):
        return {
            "solver": "impulse_rigid_body",
            "steps": int(self.steps),
            "time_step": float(self.h),
            "contacts_last_frame": int(self.last_contacts),
            "max_penetration_m": float(self.max_penetration),
            "max_speed_mps": float(self.max_speed),
            "friction": float(self.friction),
            "restitution": float(self.restitution),
        }
