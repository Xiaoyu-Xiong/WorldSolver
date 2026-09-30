"""Physics-based thin-sheet solver: cloth wrinkle wake from a moving sphere.

Method
------
The cloth is discretised as a triangular mass-spring membrane (17x17 grid) with
structural, shear and weak bending springs, integrated with a symplectic
(semi-implicit) Euler scheme on a fixed sub-step dt.  Gravity is the only
external body force.  The rigid sphere is a kinematic collider following a
prescribed smooth-ramped translation;  contacts are resolved with a
position-level non-penetration projection (normal impulse + tangential friction),
which keeps the sheet non-penetrating at every exported sample.  A short damped
settle phase produces the initial gently draped configuration.

Everything is plain NumPy: no external simulation engine is used.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None)
        if not isinstance(cfg, dict):
            cfg = {}
        cloth = cfg.get("cloth") or {}
        nx, ny = (int(v) for v in cloth.get("grid_shape", (17, 17)))
        sx, sy = (float(v) for v in cloth.get("size", (3.0, 2.5)))
        center = np.asarray(cloth.get("center", (0.0, 0.0, 1.22)), dtype=float)
        self.thickness = float(cloth.get("thickness", 0.015))
        total_mass = float(cloth.get("total_mass", 0.32))
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        duration = float(cfg.get("duration", 7.0))

        self.nx, self.ny = nx, ny
        n = nx * ny
        self.n = n
        self.duration = duration

        x = np.linspace(-0.5 * sx, 0.5 * sx, nx)
        y = np.linspace(-0.5 * sy, 0.5 * sy, ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        self.ref2d = np.column_stack((xx.ravel(), yy.ravel()))
        self.hx = sx / (nx - 1)
        self.hy = sy / (ny - 1)
        self.rest_pos = np.column_stack(
            (self.ref2d[:, 0], self.ref2d[:, 1], np.zeros(n))
        ) + center

        self.mass = np.full(n, total_mass / n)
        self.inv_mass = 1.0 / self.mass

        # ---- rigid colliders (frozen scene geometry) -----------------------
        self.sphere_r = 0.62
        self.sphere_c0 = np.asarray([0.0, 0.0, 0.68], dtype=float)
        self.sphere_v0 = 0.12  # prescribed translation speed along +x [m/s]
        self.ramp = 0.4        # smooth ramp time for the prescribed motion [s]
        self.r_eff = self.sphere_r + 0.5 * self.thickness
        self.ground_top = 0.0  # top face of the frozen ground box
        self.ground_off = 0.5 * self.thickness

        # ---- time stepping -------------------------------------------------
        self.substeps = 250
        self.dt = sample_dt / self.substeps
        self.steps = 0

        # ---- springs -------------------------------------------------------
        idx = np.arange(n).reshape(ny, nx)
        self.idx = idx
        pairs, rest, stiff = [], [], []

        def add(a, b, length, k):
            pairs.append((a.ravel(), b.ravel()))
            rest.append(np.full(a.size, float(length)))
            stiff.append(np.full(a.size, float(k)))

        add(idx[:, :-1], idx[:, 1:], self.hx, 4000.0)                # structural x
        add(idx[:-1, :], idx[1:, :], self.hy, 4000.0)                # structural y
        diag = float(np.hypot(self.hx, self.hy))
        add(idx[:-1, :-1], idx[1:, 1:], diag, 2000.0)                # shear
        add(idx[:-1, 1:], idx[1:, :-1], diag, 2000.0)                # shear
        if nx >= 3:
            add(idx[:, :-2], idx[:, 2:], 2.0 * self.hx, 1.0)         # bending x
        if ny >= 3:
            add(idx[:-2, :], idx[2:, :], 2.0 * self.hy, 1.0)         # bending y

        self.I = np.concatenate([p[0] for p in pairs])
        self.J = np.concatenate([p[1] for p in pairs])
        self.L0 = np.concatenate(rest)
        self.K = np.concatenate(stiff)
        # viscous spring dashpots (light, for numerical robustness)
        zeta = 0.15
        self.Cd = 2.0 * zeta * np.sqrt(self.K * 0.5 * self.mass[0])

        # ---- pinned samples (corner and edge-midpoint drape support) -------
        pins = [idx[0, 0], idx[0, -1], idx[-1, 0], idx[-1, -1],
                idx[0, nx // 2], idx[-1, nx // 2],
                idx[ny // 2, 0], idx[ny // 2, -1]]
        self.pinned = np.unique(np.asarray(pins, dtype=np.int64))

        # ---- reference triangles for the deformation gradient --------------
        qi, qj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="xy")
        q = (qj * nx + qi).reshape(-1)
        tri = np.concatenate(
            [np.stack([q, q + 1, q + 1 + nx], axis=1),
             np.stack([q, q + 1 + nx, q + nx], axis=1)], axis=0
        ).astype(np.int64)
        self.tri = tri
        a2 = self.ref2d[tri[:, 0]]
        b2 = self.ref2d[tri[:, 1]]
        c2 = self.ref2d[tri[:, 2]]
        Dm = np.stack([b2 - a2, c2 - a2], axis=-1)  # [T,2,2], columns = edges
        self.Dm_inv = np.linalg.inv(Dm)

        # ---- initial state: flat sheet pushed out of the sphere ------------
        self.pos = self.rest_pos.copy()
        self.vel = np.zeros((n, 3))
        self._project_static(self._sphere_center(0.0))

        # ---- settle into a gentle drape ------------------------------------
        self.mu = 0.6
        settle_steps = int(round(1.5 / self.dt))
        for k in range(settle_steps):
            self.steps += 1
            self._step(3.0, 0.0)  # sphere held at start pose while draping
        self.vel[:] = 0.0
        self.steps = 0

        # diagnostics bookkeeping
        self.min_gap = float(np.min(self._signed_distances(self._sphere_center(0.0))))
        self.max_speed = 0.0
        self.applied_external = self.mass[:, None] * self.gravity[None, :]

    # --------------------------------------------------------------- helpers
    def _sphere_center(self, t: float) -> np.ndarray:
        v0, T = self.sphere_v0, self.ramp
        if t <= 0.0:
            s = 0.0
        elif t < T:
            u = t / T
            s = v0 * T * (u ** 3 - 0.5 * u ** 4)
        else:
            s = v0 * (0.5 * T) + v0 * (t - T)
        return self.sphere_c0 + np.asarray([s, 0.0, 0.0])

    def _signed_distances(self, center: np.ndarray) -> np.ndarray:
        rel = self.pos - center[None, :]
        dist = np.sqrt((rel * rel).sum(1))
        sphere_gap = dist - self.r_eff
        ground_gap = self.pos[:, 2] - self.ground_top - self.ground_off
        return np.minimum(sphere_gap, ground_gap)

    def _project_static(self, center: np.ndarray) -> None:
        rel = self.pos - center[None, :]
        dist = np.sqrt((rel * rel).sum(1))
        pen = dist < self.r_eff
        if np.any(pen):
            safe = np.maximum(dist[pen], 1e-12)
            self.pos[pen] = center[None, :] + rel[pen] * (self.r_eff / safe)[:, None]
        below = self.pos[:, 2] < self.ground_top + self.ground_off
        if np.any(below):
            self.pos[below, 2] = self.ground_top + self.ground_off

    # ------------------------------------------------------------ time step
    def _step(self, damping: float, t: float) -> None:
        p, v = self.pos, self.vel
        d = p[self.J] - p[self.I]
        L = np.sqrt((d * d).sum(1))
        u = d / np.maximum(L, 1e-12)[:, None]
        fs = ((self.K * (L - self.L0))[:, None]) * u
        vn = ((v[self.J] - v[self.I]) * u).sum(1)
        fs += (self.Cd * vn)[:, None] * u

        acc = np.empty_like(p)
        for c in range(3):
            w = fs[:, c]
            acc[:, c] = (np.bincount(self.I, weights=w, minlength=self.n)
                         - np.bincount(self.J, weights=w, minlength=self.n))
        acc *= self.inv_mass[:, None]
        acc += self.gravity[None, :]
        acc[self.pinned] = 0.0

        v += acc * self.dt
        v *= np.exp(-damping * self.dt)
        v[self.pinned] = 0.0
        p += v * self.dt

        self._resolve_contact(t)
        p[self.pinned] = self.rest_pos[self.pinned]
        v[self.pinned] = 0.0

    def _resolve_contact(self, t: float) -> None:
        center = self._sphere_center(t)
        rel = self.pos - center[None, :]
        dist = np.sqrt((rel * rel).sum(1))
        nhat = rel / np.maximum(dist, 1e-12)[:, None]
        active = dist < self.r_eff
        if np.any(active):
            n = nhat[active]
            self.pos[active] = center[None, :] + n * (self.r_eff + 1e-12)
            va = self.vel[active]
            vnormal = (va * n).sum(1)
            va = va - np.minimum(vnormal, 0.0)[:, None] * n
            vt = va - (va * n).sum(1)[:, None] * n
            self.vel[active] = va - min(self.mu * 8.0 * self.dt, 1.0) * vt
        below = self.pos[:, 2] < self.ground_top + self.ground_off
        if np.any(below):
            self.pos[below, 2] = self.ground_top + self.ground_off
            self.vel[below, 2] = np.maximum(self.vel[below, 2], 0.0)

    # --------------------------------------------------------------- output
    def _deformation_gradient(self) -> np.ndarray:
        p = self.pos
        tri = self.tri
        a = p[tri[:, 0]]
        b = p[tri[:, 1]]
        c = p[tri[:, 2]]
        e1 = b - a
        e2 = c - a
        nvec = np.cross(e1, e2)
        nlen = np.sqrt((nvec * nvec).sum(1))
        t1 = e1 / np.maximum(np.sqrt((e1 * e1).sum(1)), 1e-12)[:, None]
        nhat = nvec / np.maximum(nlen, 1e-12)[:, None]
        t2 = np.cross(nhat, t1)
        Ds = np.empty((len(tri), 2, 2))
        Ds[:, 0, 0] = (e1 * t1).sum(1)
        Ds[:, 0, 1] = (e2 * t1).sum(1)
        Ds[:, 1, 0] = (e1 * t2).sum(1)
        Ds[:, 1, 1] = (e2 * t2).sum(1)
        F = Ds @ self.Dm_inv
        Fv = np.zeros((self.n, 2, 2))
        cnt = np.zeros(self.n)
        for k in range(3):
            np.add.at(Fv, tri[:, k], F)
            np.add.at(cnt, tri[:, k], 1.0)
        good = cnt > 0
        Fv[good] /= cnt[good][:, None, None]
        Fv[~good] = np.eye(2)
        return Fv

    def _frame(self, time: float) -> StateFrame:
        center = self._sphere_center(time)
        gaps = self._signed_distances(center)
        self.min_gap = min(self.min_gap, float(gaps.min()))
        speed = np.sqrt((self.vel * self.vel).sum(1))
        self.max_speed = max(self.max_speed, float(speed.max()))
        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
                "boundary_vertex": np.isin(
                    np.arange(self.n), self.pinned).astype(np.int32),
            },
            fields={
                "contact_signed_distance": gaps,
                "external_force": self.applied_external.copy(),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- driver
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.dt))
        if target < self.steps:
            target = self.steps
        while self.steps < target:
            self.steps += 1
            self._step(0.45, self.steps * self.dt)
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "mass_spring_semi_implicit",
            "internal_dt": self.dt,
            "substeps_per_sample": self.substeps,
            "steps_completed": self.steps,
            "vertex_count": self.n,
            "spring_count": int(self.I.size),
            "min_contact_gap": float(self.min_gap),
            "max_speed": float(self.max_speed),
            "total_mass": float(self.mass.sum()),
            "pinned_samples": int(self.pinned.size),
        }
