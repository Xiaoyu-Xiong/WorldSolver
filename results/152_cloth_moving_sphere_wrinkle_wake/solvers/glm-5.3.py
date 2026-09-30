"""Cloth wrinkle wake from a moving sphere (task 152).

Physically-based XPBD cloth solver (no Genesis):
  - 17x17 vertex grid sheet, boundary edges pinned.
  - Distance constraints for stretch (structural), shear and bending,
    solved with substepped, compliance-based dynamics (XPBD).
  - Kinematic sphere prescribed to sweep along +x beneath the sheet.
  - Non-penetrating node-vs-sphere / node-vs-ground contact with
    normal projection and Coulomb-like tangential friction.
  - Deterministic, finite state at every requested sample time.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


class BenchmarkSolver:
    """XPBD mass-spring cloth driven by a moving rigid sphere."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.scene = scene
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)

        cloth = cfg["cloth"]
        nx, ny = (int(v) for v in cloth["grid_shape"])
        self.nx, self.ny = nx, ny
        sx, sy = (float(v) for v in cloth["size"])
        center = np.asarray(cloth["center"], dtype=float)
        self.thickness = float(cloth["thickness"])
        self.half_t = 0.5 * self.thickness
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        # --- initial cloth state (mirrors the frozen scaffold) -------------
        frame0 = scene.initial_state()
        x = np.array(frame0.position, dtype=float)
        self.n = len(x)
        self.mass = np.array(frame0.mass, dtype=float)
        self.inv_mass = 1.0 / self.mass
        boundary = np.asarray(frame0.attributes.get(
            "boundary_vertex", np.zeros(self.n)), dtype=bool)
        # Pin the sheet edges: taut sheet that the sphere presses against.
        self.pinned = boundary.copy()
        self.inv_mass[self.pinned] = 0.0
        self.body_id = np.zeros(self.n, dtype=np.int32)

        self.pos = x.copy()
        self.vel = np.zeros_like(self.pos)
        self.t = 0.0

        dx = sx / (nx - 1)
        dy = sy / (ny - 1)
        self.dx, self.dy = dx, dy

        # --- constraints ---------------------------------------------------
        def vid(i, j):
            return i * nx + j

        struct_a, struct_b, shear_a, shear_b = [], [], [], []
        bend_a, bend_b = [], []
        for i in range(ny):
            for j in range(nx):
                if j + 1 < nx:
                    struct_a.append(vid(i, j)); struct_b.append(vid(i, j + 1))
                if i + 1 < ny:
                    struct_a.append(vid(i, j)); struct_b.append(vid(i + 1, j))
                if j + 2 < nx:
                    bend_a.append(vid(i, j)); bend_b.append(vid(i, j + 2))
                if i + 2 < ny:
                    bend_a.append(vid(i, j)); bend_b.append(vid(i + 2, j))
                if i + 1 < ny and j + 1 < nx:
                    shear_a.append(vid(i, j)); shear_b.append(vid(i + 1, j + 1))
                    shear_a.append(vid(i, j + 1)); shear_b.append(vid(i + 1, j))
        self.groups = []
        for (a, b, comp) in ((struct_a, struct_b, 1.0e-7),
                             (shear_a, shear_b, 1.0e-5),
                             (bend_a, bend_b, 1.0e-3)):
            ia = np.asarray(a, dtype=np.int64)
            ib = np.asarray(b, dtype=np.int64)
            rest = np.linalg.norm(x[ia] - x[ib], axis=1)
            wa = self.inv_mass[ia]
            wb = self.inv_mass[ib]
            self.groups.append({
                "ia": ia, "ib": ib, "rest": rest, "alpha": float(comp),
                "wa": wa, "wb": wb, "wsum": wa + wb,
            })

        # --- obstacle / ground ----------------------------------------------
        self.sphere_r = 0.62
        self.sphere_c0 = np.array([0.0, 0.0, 0.68])
        self.ground_z = 0.0           # top face of the static ground box
        # prescribed sweep: hold, smoothstep along +x, hold
        self.sweep_t0, self.sweep_t1 = 0.8, 5.6
        self.sweep_amp = 0.85

        # --- numerics --------------------------------------------------------
        self.n_sub = 20                # substeps per sample interval
        self.n_iter = 3                # constraint iterations per substep
        self.relax = 0.8               # Jacobi relaxation
        self.drag = 0.4                # global velocity damping [1/s]
        self.mu_sphere = 0.05          # per-substep tangential friction factor
        self.mu_ground = 0.5
        self.contact_query = 0.12      # near-contact reporting band [m]
        self._steps = 0

    # ------------------------------------------------------------------ world
    def sphere_center(self, t: float) -> np.ndarray:
        u = (t - self.sweep_t0) / (self.sweep_t1 - self.sweep_t0)
        c = self.sphere_c0.copy()
        c[0] = self.sphere_c0[0] + self.sweep_amp * _smoothstep(u)
        return c

    def sphere_velocity(self, t: float) -> np.ndarray:
        u = (t - self.sweep_t0) / (self.sweep_t1 - self.sweep_t0)
        v = np.zeros(3)
        if 0.0 < u < 1.0:
            s_prime = 6.0 * u * (1.0 - u)
            v[0] = (self.sweep_amp / (self.sweep_t1 - self.sweep_t0)) * s_prime
        return v

    # ------------------------------------------------------------------ step
    def _substep(self, h: float, c_now: np.ndarray, v_sph: np.ndarray) -> None:
        free = ~self.pinned
        # explicit inertial guess
        self.vel[free] += h * self.gravity
        # global damping
        damp = 1.0 / (1.0 + self.drag * h)
        self.vel[free] *= damp

        prev = self.pos.copy()
        self.pos[free] += h * self.vel[free]

        # XPBD distance-constraint solve (vectorized Jacobi with relaxation)
        for g in self.groups:
            g["lam"] = np.zeros(len(g["ia"]))
        for _ in range(self.n_iter):
            for g in self.groups:
                ia, ib = g["ia"], g["ib"]
                d = self.pos[ia] - self.pos[ib]
                l = np.linalg.norm(d, axis=1)
                l = np.maximum(l, 1e-12)
                nrm = d / l[:, None]
                C = l - g["rest"]
                alpha_t = g["alpha"] / (h * h)
                denom = g["wsum"] + alpha_t
                dlam = (-C - alpha_t * g["lam"]) / np.maximum(denom, 1e-12)
                dlam *= self.relax
                corr_a = dlam[:, None] * g["wa"][:, None] * nrm
                corr_b = -dlam[:, None] * g["wb"][:, None] * nrm
                dx = np.zeros_like(self.pos)
                np.add.at(dx, ia, corr_a)
                np.add.at(dx, ib, corr_b)
                self.pos += dx
                g["lam"] += dlam

        # --- contact: rigid sphere (kinematic) -----------------------------
        r_eff = self.sphere_r + self.half_t
        rel = self.pos - c_now[None, :]
        dist = np.linalg.norm(rel, axis=1)
        dist = np.maximum(dist, 1e-12)
        pen = r_eff - dist
        hit = pen > 0.0
        if np.any(hit):
            nvec = rel / dist[:, None]
            self.pos[hit] += nvec[hit] * pen[hit, None]

        # --- contact: ground plane (top of static box) ----------------------
        pen_g = (self.ground_z + self.half_t) - self.pos[:, 2]
        hit_g = pen_g > 0.0
        if np.any(hit_g):
            self.pos[hit_g, 2] += pen_g[hit_g]

        # velocity from positions (free vertices)
        self.vel[free] = (self.pos[free] - prev[free]) / h

        # contact velocity response: remove approach + tangential friction
        if np.any(hit):
            nvec = rel / np.maximum(dist, 1e-12)[:, None]
            vrel = self.vel[hit] - v_sph[None, :]
            vn = np.sum(vrel * nvec[hit], axis=1)
            appr = vn < 0.0
            idx = np.where(hit)[0]
            # cancel approaching normal velocity relative to the sphere
            rows = idx[appr]
            if len(rows):
                self.vel[rows] -= nvec[rows] * vn[appr, None]
            # Coulomb-like tangential friction toward sphere surface
            vrel2 = self.vel[hit] - v_sph[None, :]
            vt = vrel2 - nvec[hit] * np.sum(vrel2 * nvec[hit], axis=1)[:, None]
            self.vel[hit] -= np.minimum(1.0, self.mu_sphere) * vt

        if np.any(hit_g):
            vg = self.vel[hit_g]
            vz = vg[:, 2]
            below = vz < 0.0
            if np.any(below):
                vg[below, 2] = 0.0
            vg[:, :2] *= max(0.0, 1.0 - self.mu_ground)
            self.vel[hit_g] = vg

        self.vel[self.pinned] = 0.0
        self._steps += 1

    # ----------------------------------------------------------------- frames
    def _deformation_gradient(self) -> np.ndarray:
        p = self.pos.reshape(self.ny, self.nx, 3)
        tu = np.gradient(p, self.dx, axis=1)   # tangent along material x
        tv = np.gradient(p, self.dy, axis=0)   # tangent along material y
        F = np.empty((self.ny, self.nx, 2, 2))
        F[:, :, 0, 0] = tu[:, :, 0]
        F[:, :, 0, 1] = tv[:, :, 0]
        F[:, :, 1, 0] = tu[:, :, 1]
        F[:, :, 1, 1] = tv[:, :, 1]
        return F.reshape(self.n, 2, 2)

    def _contact_signed_distance(self, c_now: np.ndarray) -> np.ndarray:
        r_eff = self.sphere_r + self.half_t
        sd_sphere = np.linalg.norm(self.pos - c_now[None, :], axis=1) - r_eff
        sd_ground = self.pos[:, 2] - (self.ground_z + self.half_t)
        sd = np.minimum(sd_sphere, sd_ground)
        return sd[sd < self.contact_query]

    def _frame(self, t: float) -> StateFrame:
        c_now = self.sphere_center(t)
        ext_force = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(t),
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.pinned.astype(np.int32),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self._contact_signed_distance(c_now),
                "cumulative_mass_source": np.zeros(self.n),
                "external_force": ext_force,
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- api
    def advance_to(self, time: float):
        target = max(0.0, float(time))
        if target <= self.t + 1e-12:
            return self._frame(self.t if target < self.t else target)
        while self.t < target - 1e-12:
            h = min(1.0 / (self.n_sub * 30.0), target - self.t)
            h = max(h, 1e-9)
            c_now = self.sphere_center(self.t + 0.5 * h)
            v_sph = self.sphere_velocity(self.t + 0.5 * h)
            self._substep(h, c_now, v_sph)
            self.t += h
        self.t = target
        return self._frame(self.t)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "substeps": int(self._steps),
            "pinned": int(np.count_nonzero(self.pinned)),
            "max_speed": float(np.max(np.linalg.norm(self.vel, axis=1))),
        }
