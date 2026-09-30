"""Physically-based cloth solver: compressed cloth wrinkle band.

Explicit dynamics on a triangular cloth mesh (mass-spring network with
structural / shear / bending edges), semi-implicit (symplectic) Euler
substeps, kinematic grips clamping the two opposite sheet boundaries and
moving inward to compress the sheet, gravity, and penalty-free projection
contacts against the frozen collision boxes.  Wrinkles nucleate through
Euler buckling of the compressed sheet (material slack + weak bending
stiffness) seeded by a deterministic small perturbation.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _box_sdf(points: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    q = np.abs(points - center) - half
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


class BenchmarkSolver:
    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        cloth = cfg["cloth"]
        self.nx, self.ny = (int(v) for v in cloth["grid_shape"])
        self.thickness = float(cloth["thickness"])

        frame = scene.initial_state()
        p0 = np.array(frame.position, dtype=float)
        self.N = len(p0)
        self.mass = np.array(frame.mass, dtype=float)
        self.body_id = np.zeros(self.N, dtype=np.int32)
        self.minv = 1.0 / np.maximum(self.mass, 1e-12)

        # material (reference) coordinates: the flat sheet
        self.X = p0[:, :2].copy()

        # deterministic tiny perturbation to nucleate buckling wrinkles
        rng = np.random.default_rng(int(seed) * 1009 + 17)
        self.p = p0.copy()
        self.p[:, 2] += rng.normal(0.0, 1.5e-3, self.N)
        self.v = np.zeros((self.N, 3))

        self.t = 0.0
        self._steps = 0

        # ---- grid topology -------------------------------------------------
        nx, ny = self.nx, self.ny
        self.dx = float(cloth["size"][0]) / max(nx - 1, 1)
        self.dy = float(cloth["size"][1]) / max(ny - 1, 1)

        # boundary (clamped) columns: left x-min and right x-max
        xmin = self.X[:, 0].min()
        xmax = self.X[:, 0].max()
        self.clamped = (self.X[:, 0] < xmin + 1e-9) | (self.X[:, 0] > xmax - 1e-9)
        self.free = ~self.clamped
        self.clamp_y0 = self.p[self.clamped, 1].copy()
        self.clamp_z0 = 2.15  # grip center height (frozen geometry)
        self.grip_x0 = np.where(self.X[self.clamped, 0] < 0.0, -1.5, 1.5)

        # triangles
        tris = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                b = a + 1
                c = a + nx
                d = c + 1
                tris.append((a, b, d))
                tris.append((a, d, c))
        self.tri = np.asarray(tris, dtype=np.int64)
        # material edge matrices and inverses for deformation gradients
        A = self.X[self.tri[:, 0]]
        B = self.X[self.tri[:, 1]]
        C = self.X[self.tri[:, 2]]
        Dm = np.stack((B - A, C - A), axis=-1)  # [T,2,2]
        det = Dm[:, 0, 0] * Dm[:, 1, 1] - Dm[:, 0, 1] * Dm[:, 1, 0]
        det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        Dminv = np.empty_like(Dm)
        Dminv[:, 0, 0] = Dm[:, 1, 1] / det
        Dminv[:, 0, 1] = -Dm[:, 0, 1] / det
        Dminv[:, 1, 0] = -Dm[:, 1, 0] / det
        Dminv[:, 1, 1] = Dm[:, 0, 0] / det
        self.Dminv = Dminv

        # ---- spring network -------------------------------------------------
        slack = 1.02  # slight material slack (sheet slightly larger than span)
        k_st, kd_st = 600.0, 0.06
        k_sh, kd_sh = 180.0, 0.02
        k_bd, kd_bd = 5.0, 0.004

        si, sj, rest, ks, kds = [], [], [], [], []

        def add(i, j, r, k, kd):
            si.append(i); sj.append(j); rest.append(r); ks.append(k); kds.append(kd)

        for j in range(ny):
            for i in range(nx - 1):
                add(j * nx + i, j * nx + i + 1, self.dx * slack, k_st, kd_st)
        for j in range(ny - 1):
            for i in range(nx):
                add(j * nx + i, (j + 1) * nx + i, self.dy * slack, k_st, kd_st)
        diag = float(np.hypot(self.dx, self.dy)) * slack
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                b = a + 1
                c = a + nx
                d = c + 1
                add(a, d, diag, k_sh, kd_sh)
                add(b, c, diag, k_sh, kd_sh)
        for j in range(ny):
            for i in range(nx - 2):
                add(j * nx + i, j * nx + i + 2, 2.0 * self.dx * slack, k_bd, kd_bd)
        for j in range(ny - 2):
            for i in range(nx):
                add(j * nx + i, (j + 2) * nx + i, 2.0 * self.dy * slack, k_bd, kd_bd)

        self.si = np.asarray(si, dtype=np.int64)
        self.sj = np.asarray(sj, dtype=np.int64)
        self.srest = np.asarray(rest, dtype=float)
        self.sk = np.asarray(ks, dtype=float)
        self.skd = np.asarray(kds, dtype=float)

        # external (body) force per sample: gravity only
        self.f_ext = self.mass[:, None] * self.gravity[None, :]

        # frozen collision boxes
        self.ground_c = np.array([0.0, 0.0, -0.05])
        self.ground_h = np.array([2.4, 2.0, 0.05])
        self.grip_h = np.array([0.12, 1.25, 0.12])
        self.grip_z = 2.15
        self.grip_x_start = 1.5

        # grip kinematics: move inward between t0 and t1
        self.dmax = 0.5
        self.t0, self.t1 = 0.4, 3.4

        self.h = 4.0e-4
        self.drag = 1.0
        self.contact_margin = 0.06

        self._ready = True

    # ------------------------------------------------------- grip kinematics
    def _grip(self, t: float):
        span = self.t1 - self.t0
        u = (t - self.t0) / span
        uc = min(max(u, 0.0), 1.0)
        s = uc * uc * (3.0 - 2.0 * uc)
        disp = self.dmax * s
        if u <= 0.0 or u >= 1.0:
            vel = 0.0
        else:
            vel = self.dmax * 6.0 * uc * (1.0 - uc) / span
        return disp, vel

    # ------------------------------------------------------------ one substep
    def _step(self, h: float) -> None:
        t_new = self.t + h
        p, v = self.p, self.v

        # spring forces (structural + shear + bending) with edge damping
        d = p[self.sj] - p[self.si]
        L = np.linalg.norm(d, axis=1)
        L = np.maximum(L, 1e-9)
        direction = d / L[:, None]
        rel = np.sum((v[self.sj] - v[self.si]) * direction, axis=1)
        mag = self.sk * (L - self.srest) + self.skd * rel
        fv = mag[:, None] * direction
        force = self.f_ext.copy()
        np.add.at(force, self.si, fv)
        np.add.at(force, self.sj, -fv)

        # semi-implicit Euler with linear drag
        v += h * (force * self.minv[:, None])
        v /= (1.0 + h * self.drag)
        p += h * v

        # clamped grip boundaries (kinematic prescription)
        disp, gvel = self._grip(t_new)
        idx = np.where(self.clamped)[0]
        side = np.where(self.grip_x0 < 0.0, -1.0, 1.0)
        p[idx, 0] = side * (self.grip_x_start - disp)
        p[idx, 1] = self.clamp_y0
        p[idx, 2] = self.clamp_z0
        v[idx, 0] = side * gvel
        v[idx, 1] = 0.0
        v[idx, 2] = 0.0

        # contacts (projection, penalty-free) for free samples
        self._collide_box(self.ground_c, self.ground_h + np.array([0.0, 0.0, 0.5 * self.thickness]))
        d2, _ = self._grip(t_new)
        for s in (-1.0, 1.0):
            c = np.array([s * (self.grip_x_start - d2), 0.0, self.grip_z])
            self._collide_box(c, self.grip_h + 0.5 * self.thickness)

        self.t = t_new
        self._steps += 1

    def _collide_box(self, center: np.ndarray, half: np.ndarray) -> None:
        q = np.abs(self.p - center) - half
        inside = (q < 0.0).all(axis=1) & self.free
        if not np.any(inside):
            return
        idx = np.where(inside)[0]
        qi = q[idx]
        axis = np.argmax(qi, axis=1)  # closest face = least penetration
        for a in range(3):
            sel = axis == a
            if not np.any(sel):
                continue
            k = idx[sel]
            sgn = np.sign(self.p[k, a] - center[a])
            sgn[sgn == 0.0] = 1.0
            self.p[k, a] = center[a] + sgn * half[a]
            vn = self.v[k, a] * sgn
            self.v[k, a] = np.where(vn < 0.0, 0.0, self.v[k, a])
            # simple tangential friction while in contact
            others = [b for b in range(3) if b != a]
            self.v[k, others[0]] *= (1.0 - 4.0 * self.h)
            self.v[k, others[1]] *= (1.0 - 4.0 * self.h)

    # ------------------------------------------------------ per-sample output
    def _deformation_gradients(self) -> np.ndarray:
        p = self.p
        a = p[self.tri[:, 0]]
        b = p[self.tri[:, 1]]
        c = p[self.tri[:, 2]]
        Ds = np.stack((b - a, c - a), axis=-1)  # [T,3,2]
        F3 = Ds @ self.Dminv  # [T,3,2] world deformation gradient
        ntri = np.cross(b - a, c - a)
        area = 0.5 * np.linalg.norm(ntri, axis=1)
        area = np.maximum(area, 1e-14)

        Fv = np.zeros((self.N, 3, 2))
        nv = np.zeros((self.N, 3))
        wv = np.zeros(self.N)
        for k in range(3):
            np.add.at(Fv, self.tri[:, k], F3 * area[:, None, None])
            np.add.at(nv, self.tri[:, k], ntri * area[:, None])
            np.add.at(wv, self.tri[:, k], area)
        wv = np.maximum(wv, 1e-14)
        Fbar = Fv / wv[:, None, None]
        n = nv / np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-14)

        # local tangent frame per sample
        e1 = Fbar[:, :, 0].copy()
        e1 = e1 - np.sum(e1 * n, axis=1, keepdims=True) * n
        e1n = np.linalg.norm(e1, axis=1)
        bad = e1n < 1e-9
        if np.any(bad):
            fallback = np.zeros((int(np.sum(bad)), 3))
            fallback[:, 0] = 1.0
            e1[bad] = fallback - np.sum(fallback * n[bad], axis=1, keepdims=True) * n[bad]
            e1n[bad] = np.linalg.norm(e1[bad], axis=1)
        e1 = e1 / np.maximum(e1n, 1e-14)[:, None]
        e2 = np.cross(n, e1)

        F2 = np.empty((self.N, 2, 2))
        F2[:, 0, 0] = np.sum(e1 * Fbar[:, :, 0], axis=1)
        F2[:, 0, 1] = np.sum(e1 * Fbar[:, :, 1], axis=1)
        F2[:, 1, 0] = np.sum(e2 * Fbar[:, :, 0], axis=1)
        F2[:, 1, 1] = np.sum(e2 * Fbar[:, :, 1], axis=1)
        F2[~np.isfinite(F2).all(axis=(1, 2))] = np.eye(2)
        return F2

    def _contact_distances(self) -> np.ndarray:
        gap = _box_sdf(self.p, self.ground_c, self.ground_h)
        disp, _ = self._grip(self.t)
        for s in (-1.0, 1.0):
            c = np.array([s * (self.grip_x_start - disp), 0.0, self.grip_z])
            gap = np.minimum(gap, _box_sdf(self.p, c, self.grip_h))
        return gap

    def _make_frame(self, t: float) -> StateFrame:
        gap = self._contact_distances()
        active = gap <= self.contact_margin
        contact_dist = gap[active].copy()
        frame = StateFrame(
            float(t),
            self.p.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradients(),
            },
            fields={
                "contact_signed_distance": contact_dist,
                "cumulative_mass_source": np.zeros(self.N),
                "external_force": self.f_ext.copy(),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ----------------------------------------------------------------- public
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("initialize() must be called before advance_to()")
        target = float(time)
        if target < self.t - 1e-12:
            return self._make_frame(self.t)
        while target - self.t > 1e-9:
            hh = min(self.h, target - self.t)
            self._step(hh)
        self.t = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "substeps": int(self._steps),
            "samples": int(self.N),
            "grip_displacement": float(self.dmax * (lambda u: u * u * (3 - 2 * u))(
                min(max((self.t - self.t0) / (self.t1 - self.t0), 0.0), 1.0))),
            "min_contact_gap": float(np.min(self._contact_distances())) if self._ready else 0.0,
        }
