"""Sand ball-impact solver: explicit MLS-MPM with Drucker-Prager elastoplasticity.

Physics
-------
* Sand is a continuum sampled by the frozen particle cloud from ``scene.py``.
  Each particle carries mass, initial volume, elastic deformation gradient and an
  APIC affine velocity matrix.  Momentum is transferred particle -> grid -> particle
  with quadratic B-splines (MLS-MPM, Hu et al. 2018).
* Constitutive model: St.Venant-Kirchhoff elasticity in Hencky (log) strain with the
  Drucker-Prager return mapping of Klar et al. 2016 ("Drucker-Prager Elastoplasticity
  for Sand Animation"), including the cone-tip projection for tensile states.  This
  gives the frictional, shear-band / spray behaviour of dry sand.
* The impact ball is a rigid body integrated with symplectic Euler.  Coupling is done
  on the background grid: node velocities inside the sphere are projected onto the
  rigid surface with Coulomb friction and the reaction impulse is applied back on the
  ball, so momentum exchange is two-way.  A particle-level push-out keeps samples
  outside the analytic sphere.
* The container floor (frozen ground box, top face z = 0) is a separating frictional
  boundary on the grid, plus a hard, non-penetrating constraint for the ball.

Everything is a fixed-step deterministic integration: no keyframes, no scripted motion.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(scene.config)
        sampling = cfg["sampling"]

        self.spacing = float(sampling["spacing"])
        self.density = float(sampling["density"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        frame0 = scene.initial_state()
        self.x0 = np.asarray(frame0.position, dtype=float).copy()
        self.mass_all = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.n_all = len(self.x0)

        attrs0 = dict(frame0.attributes)
        self.particle_radius = np.asarray(
            attrs0.get("particle_radius", np.full(self.n_all, 0.45 * self.spacing)), dtype=float
        ).copy()
        self.material_label = np.asarray(
            attrs0.get("material_label", self.body_id), dtype=np.int32
        ).copy()
        self.scalars0 = {k: float(v) for k, v in dict(frame0.scalars).items()}

        # ---- object split: body_id 0 = granular sand, body_id 1 = rigid ball
        self.sand_idx = np.flatnonzero(self.body_id == 0)
        ball_idx = np.flatnonzero(self.body_id == 1)
        self.ball_i = int(ball_idx[0]) if len(ball_idx) else -1

        self.xp = self.x0[self.sand_idx].copy()
        self.vp = np.zeros_like(self.xp)
        self.mp = self.mass_all[self.sand_idx].copy()
        self.vol0 = self.mp / self.density
        self.n_p = len(self.xp)
        eye = np.eye(3)
        self.Fp = np.repeat(eye[None], self.n_p, axis=0).copy()
        self.Cp = np.zeros((self.n_p, 3, 3))

        # ---- rigid ball from frozen geometry
        self.ball_radius = 0.54
        geom = cfg.get("scene_geometry") or {}
        for obj in geom.get("objects", []):
            if obj.get("id") == "ball":
                r = obj.get("geometry", {}).get("radius")
                if r:
                    self.ball_radius = float(r)
        self.ball_x = (
            self.x0[self.ball_i].copy() if self.ball_i >= 0 else np.array([0.0, 0.0, 1.85])
        )
        self.ball_v = np.zeros(3)
        self.ball_m = float(self.mass_all[self.ball_i]) if self.ball_i >= 0 else 1.0

        # ---- floor (ground box top face)
        self.floor_z = 0.0
        for obj in geom.get("objects", []):
            if obj.get("id") == "ground":
                he = obj.get("geometry", {}).get("half_extents")
                tr = obj.get("transform", {}).get("translation", [0.0, 0.0, 0.0])
                if he:
                    self.floor_z = float(tr[2]) + float(he[2])

        # ---- material parameters (sand regime of the source paper)
        E, nu = 3.537e5, 0.3
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        friction_angle = np.deg2rad(35.0)
        s = np.sin(friction_angle)
        self.alpha = np.sqrt(2.0 / 3.0) * 2.0 * s / (3.0 - s)
        self.mu_floor = 0.55
        self.mu_ball = 0.35

        # ---- background grid
        self.dx = 0.2
        self.inv_dx = 1.0 / self.dx
        self.origin = np.array([-3.2, -1.4, self.floor_z - 0.4])
        upper = np.array([3.2, 1.4, self.floor_z + 3.2])
        counts = np.ceil((upper - self.origin) * self.inv_dx).astype(int) + 1
        self.nx, self.ny, self.nz = (int(c) for c in np.maximum(counts, 5))
        self.n_cells = self.nx * self.ny * self.nz
        gi, gj, gk = np.meshgrid(
            np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing="ij"
        )
        self.node_pos = (
            np.column_stack((gi.ravel(), gj.ravel(), gk.ravel())).astype(float) * self.dx
            + self.origin
        )
        self.lo = self.origin + 1.51 * self.dx
        self.hi = self.origin + (np.array([self.nx, self.ny, self.nz]) - 2.51) * self.dx

        off = np.stack(np.meshgrid(np.arange(3), np.arange(3), np.arange(3), indexing="ij"), -1)
        self.off = off.reshape(-1, 3).astype(np.int64)   # (27,3)
        self.offf = self.off.astype(float)

        # ---- time stepping
        self.sample_dt = float(cfg["sample_dt"])
        self.n_sub = 48
        self.dt = self.sample_dt / self.n_sub
        self.time = 0.0
        self.steps = 0
        self.max_speed_seen = 0.0

    # ------------------------------------------------------------ plasticity
    def _stress_and_plasticity(self):
        """Drucker-Prager return mapping; returns Kirchhoff stress tau (P,3,3)."""
        F = self.Fp
        U, sig, Vt = np.linalg.svd(F)
        sig = np.clip(sig, 1e-6, 1e6)
        eps = np.log(sig)
        tr = eps.sum(axis=1, keepdims=True)
        eps_hat = eps - tr / 3.0
        norm = np.linalg.norm(eps_hat, axis=1, keepdims=True)
        coef = (3.0 * self.lam + 2.0 * self.mu) / (2.0 * self.mu)
        dgamma = norm + coef * tr * self.alpha

        eps_flow = eps - dgamma * eps_hat / np.maximum(norm, 1e-12)
        eps_new = np.where(dgamma <= 0.0, eps, eps_flow)
        eps_new = np.where(tr > 0.0, 0.0, eps_new)          # cone tip: full release
        eps_new = np.clip(eps_new, -3.0, 3.0)

        sig_new = np.exp(eps_new)
        self.Fp = np.einsum("pij,pj,pjk->pik", U, sig_new, Vt)

        tr_new = eps_new.sum(axis=1, keepdims=True)
        tau_diag = 2.0 * self.mu * eps_new + self.lam * tr_new
        tau = np.einsum("pij,pj,pkj->pik", U, tau_diag, U)
        return tau

    # ------------------------------------------------------------ one substep
    def _substep(self):
        dt, dx, inv_dx = self.dt, self.dx, self.inv_dx

        tau = self._stress_and_plasticity()
        affine = self.mp[:, None, None] * self.Cp - (
            (4.0 * dt * inv_dx * inv_dx) * self.vol0[:, None, None] * tau
        )

        xg = (self.xp - self.origin) * inv_dx
        base = np.floor(xg - 0.5).astype(np.int64)
        base[:, 0] = np.clip(base[:, 0], 0, self.nx - 3)
        base[:, 1] = np.clip(base[:, 1], 0, self.ny - 3)
        base[:, 2] = np.clip(base[:, 2], 0, self.nz - 3)
        fx = xg - base                                        # (P,3)

        w = np.empty((self.n_p, 3, 3))
        w[:, 0] = 0.5 * (1.5 - fx) ** 2
        w[:, 1] = 0.75 - (fx - 1.0) ** 2
        w[:, 2] = 0.5 * (fx - 0.5) ** 2
        wt = w[:, self.off[:, 0], 0] * w[:, self.off[:, 1], 1] * w[:, self.off[:, 2], 2]  # (P,27)

        nodes = base[:, None, :] + self.off[None, :, :]        # (P,27,3)
        flat = ((nodes[:, :, 0] * self.ny + nodes[:, :, 1]) * self.nz + nodes[:, :, 2]).ravel()
        dpos = (self.offf[None, :, :] - fx[:, None, :]) * dx   # (P,27,3)

        contrib = (self.mp[:, None] * wt)[:, :, None] * self.vp[:, None, :]
        contrib = contrib + wt[:, :, None] * np.einsum("pab,pnb->pna", affine, dpos)

        gm = np.bincount(flat, weights=(wt * self.mp[:, None]).ravel(), minlength=self.n_cells)
        gv = np.empty((self.n_cells, 3))
        for d in range(3):
            gv[:, d] = np.bincount(
                flat, weights=np.ascontiguousarray(contrib[:, :, d]).ravel(), minlength=self.n_cells
            )

        active = gm > 1e-12
        gv[active] /= gm[active, None]
        gv[~active] = 0.0
        gv[active] += dt * self.gravity

        # --------- rigid ball: predictor, then grid coupling with reaction impulse
        ball_v = self.ball_v + dt * self.gravity
        rel = self.node_pos - self.ball_x
        dist = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        hit = active & (dist < self.ball_radius + 0.6 * dx)
        if np.any(hit):
            n = rel[hit] / np.maximum(dist[hit, None], 1e-12)
            vrel = gv[hit] - ball_v[None, :]
            vn = np.einsum("ij,ij->i", vrel, n)
            approach = vn < 0.0
            if np.any(approach):
                idx = np.flatnonzero(hit)[approach]
                nn = n[approach]
                vr = vrel[approach]
                vnn = vn[approach]
                vt = vr - vnn[:, None] * nn
                vt_norm = np.linalg.norm(vt, axis=1)
                scale = np.maximum(
                    0.0, 1.0 - self.mu_ball * np.abs(vnn) / np.maximum(vt_norm, 1e-9)
                )
                v_new = ball_v[None, :] + vt * scale[:, None]
                impulse = (gm[idx, None] * (v_new - gv[idx])).sum(axis=0)
                gv[idx] = v_new
                ball_v = ball_v - impulse / self.ball_m

        # --------- floor and side-wall boundary conditions on the grid
        znode = self.node_pos[:, 2]
        floor = active & (znode <= self.floor_z + 1e-9)
        if np.any(floor):
            v = gv[floor]
            down = v[:, 2] < 0.0
            if np.any(down):
                sub = v[down]
                vn = -sub[:, 2]
                vt = sub[:, :2]
                vt_norm = np.linalg.norm(vt, axis=1)
                scale = np.maximum(0.0, 1.0 - self.mu_floor * vn / np.maximum(vt_norm, 1e-9))
                sub[:, :2] = vt * scale[:, None]
                sub[:, 2] = 0.0
                v[down] = sub
            gv[floor] = v
        # slip side walls / open top guard
        xn, yn = self.node_pos[:, 0], self.node_pos[:, 1]
        wall_lo_x = xn <= self.origin[0] + 1.01 * dx
        wall_hi_x = xn >= self.origin[0] + (self.nx - 2.01) * dx
        wall_lo_y = yn <= self.origin[1] + 1.01 * dx
        wall_hi_y = yn >= self.origin[1] + (self.ny - 2.01) * dx
        gv[wall_lo_x, 0] = np.maximum(gv[wall_lo_x, 0], 0.0)
        gv[wall_hi_x, 0] = np.minimum(gv[wall_hi_x, 0], 0.0)
        gv[wall_lo_y, 1] = np.maximum(gv[wall_lo_y, 1], 0.0)
        gv[wall_hi_y, 1] = np.minimum(gv[wall_hi_y, 1], 0.0)

        # --------- grid to particle
        gvp = gv[flat].reshape(self.n_p, 27, 3)
        self.vp = np.einsum("pn,pna->pa", wt, gvp)
        self.Cp = 4.0 * inv_dx * inv_dx * np.einsum("pn,pna,pnb->pab", wt, gvp, dpos)
        self.Fp = np.einsum("pab,pbc->pac", np.eye(3)[None] + dt * self.Cp, self.Fp)

        speed = np.linalg.norm(self.vp, axis=1)
        big = speed > 60.0
        if np.any(big):
            self.vp[big] *= (60.0 / speed[big])[:, None]
        self.max_speed_seen = max(self.max_speed_seen, float(speed.max()) if self.n_p else 0.0)

        self.xp = self.xp + dt * self.vp

        # --------- particle-level rigid/floor corrections (keep samples admissible)
        rp = self.particle_radius[self.sand_idx]
        d = self.xp - self.ball_x
        dn = np.sqrt(np.einsum("ij,ij->i", d, d))
        pen = (self.ball_radius + 0.35 * rp) - dn
        inside = pen > 0.0
        if np.any(inside):
            nrm = d[inside] / np.maximum(dn[inside, None], 1e-12)
            self.xp[inside] += nrm * pen[inside, None]
            vrel = self.vp[inside] - ball_v[None, :]
            vn = np.einsum("ij,ij->i", vrel, nrm)
            neg = vn < 0.0
            if np.any(neg):
                sel = np.flatnonzero(inside)[neg]
                corr = -vn[neg][:, None] * nrm[neg]
                dpm = (self.mp[sel, None] * corr).sum(axis=0)
                self.vp[sel] += corr
                ball_v = ball_v - dpm / self.ball_m

        below = self.xp[:, 2] < self.floor_z + 0.3 * rp
        if np.any(below):
            self.xp[below, 2] = self.floor_z + 0.3 * rp[below]
            self.vp[below, 2] = np.maximum(self.vp[below, 2], 0.0)
        np.clip(self.xp, self.lo, self.hi, out=self.xp)

        # --------- advance the rigid ball, resolve floor contact
        self.ball_v = ball_v
        self.ball_x = self.ball_x + dt * self.ball_v
        gap = (self.ball_x[2] - self.ball_radius) - self.floor_z
        if gap < 0.0:
            self.ball_x[2] = self.floor_z + self.ball_radius
            if self.ball_v[2] < 0.0:
                self.ball_v[2] = 0.0
            self.ball_v[0] *= 0.9
            self.ball_v[1] *= 0.9

        self.steps += 1

    # ---------------------------------------------------------------- frames
    def _frame(self, time: float) -> StateFrame:
        n = self.n_all
        pos = np.empty((n, 3))
        vel = np.zeros((n, 3))
        pos[self.sand_idx] = self.xp
        vel[self.sand_idx] = self.vp
        if self.ball_i >= 0:
            pos[self.ball_i] = self.ball_x
            vel[self.ball_i] = self.ball_v

        F = np.repeat(np.eye(3)[None], n, axis=0)
        F[self.sand_idx] = self.Fp

        ext_force = self.mass_all[:, None] * self.gravity[None, :]

        rp = self.particle_radius[self.sand_idx]
        d = self.xp - self.ball_x
        dn = np.sqrt(np.einsum("ij,ij->i", d, d))
        sd_ball = dn - self.ball_radius - rp
        sd_floor = (self.xp[:, 2] - self.floor_z) - rp
        sd = np.minimum(sd_ball, sd_floor)
        band = 2.0 * self.spacing
        near = sd < band
        ball_floor_gap = (self.ball_x[2] - self.ball_radius) - self.floor_z
        contact_sd = np.concatenate((sd[near], np.array([ball_floor_gap])))

        frame = StateFrame(
            float(time),
            np.ascontiguousarray(pos),
            np.ascontiguousarray(vel),
            orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1)),
            angular_velocity=np.zeros((n, 3)),
            mass=self.mass_all.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass_all.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.material_label.copy(),
                "deformation_gradient": F,
            },
            fields={
                "contact_signed_distance": np.ascontiguousarray(contact_sd, dtype=float),
                "cumulative_mass_source": np.zeros(1),
                "external_force": np.ascontiguousarray(ext_force),
            },
            scalars={
                "contact_length_scale": float(self.scalars0.get("contact_length_scale", self.spacing)),
                "grid_spacing": float(self.scalars0.get("grid_spacing", self.spacing)),
                "characteristic_length": float(
                    self.scalars0.get("characteristic_length", max(self.spacing, 1.0))
                ),
                "reference_velocity": float(self.scalars0.get("reference_velocity", 1.0)),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 1e-12:
            self._substep()
            self.time = self.steps * self.dt
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MLS-MPM + Drucker-Prager (Hencky elasticity), rigid ball grid coupling",
            "substeps": int(self.steps),
            "dt": float(self.dt),
            "grid": [int(self.nx), int(self.ny), int(self.nz)],
            "grid_dx": float(self.dx),
            "sand_particles": int(self.n_p),
            "ball_mass": float(self.ball_m),
            "ball_final_z": float(self.ball_x[2]),
            "max_particle_speed": float(self.max_speed_seen),
        }
