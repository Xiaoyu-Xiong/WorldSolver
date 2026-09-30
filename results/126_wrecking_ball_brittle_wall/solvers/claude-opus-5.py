"""Brittle-fracture solver: wrecking ball versus a bonded-particle masonry panel.

Physical model (implemented from scratch with NumPy only)
--------------------------------------------------------
*  The dynamic subject is the destructible central chunk of the masonry wall,
   discretised as a lattice of mass points (the frozen sampling cloud).
*  Cohesion is provided by a bond-based peridynamic network: every pair of
   samples closer than ``1.8 * spacing`` in the reference configuration carries
   a linear axial spring with viscous damping.  A bond is *irreversibly*
   removed when its stretch exceeds a randomised tensile limit or falls below a
   compressive crushing limit.  Broken bonds are the newly created fracture
   surfaces, so cracks initiate at the impact zone and propagate/branch through
   the lattice.
*  Samples that touch the surrounding static masonry (bottom / top / lateral
   layers) additionally carry brittle anchor springs to the frozen wall; they
   snap once their displacement exceeds a brittle threshold, which is what
   releases fragments for gravity-driven fall.
*  The wrecking ball is a rigid pendulum (single rotational degree of freedom
   about the cable attachment point taken from the frozen cable geometry).  It
   starts at rest, displaced to one side, and is released under gravity.  Ball
   and lattice are coupled two-way through penalty contact.
*  Fragments collide with the rigid floor (Coulomb friction), with the static
   wall slab (activated once a sample has left the slab) and with each other.

Time stepping is an explicit symplectic (semi-implicit Euler / leapfrog)
scheme with a fixed substep chosen well below the elastic and contact
stability limits, so the trajectory is deterministic and finite everywhere.
"""
from __future__ import annotations

import numpy as np

try:  # StateFrame is platform owned; the fallback keeps the module importable.
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover
    StateFrame = None


class BenchmarkSolver:
    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = dict(getattr(scene, "config", {}) or {})
        rng = np.random.default_rng(int(seed))

        frame0 = scene.initial_state()
        self.x0 = np.asarray(frame0.position, dtype=float).copy()
        self.x = self.x0.copy()
        self.n = len(self.x)
        self.v = np.zeros((self.n, 3), dtype=float)
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = (
            np.asarray(frame0.body_id, dtype=np.int32).copy()
            if frame0.body_id is not None
            else np.zeros(self.n, dtype=np.int32)
        )
        attrs0 = dict(frame0.attributes)
        sampling = self.config.get("sampling", {})
        self.spacing = float(sampling.get("spacing", 0.12))
        self.density = float(sampling.get("density", 1350.0))
        self.radius = np.asarray(
            attrs0.get("particle_radius", np.full(self.n, 0.45 * self.spacing)), dtype=float
        ).copy()
        self.material_label = np.asarray(
            attrs0.get("material_label", np.zeros(self.n)), dtype=np.int32
        ).copy()
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        self._read_geometry()

        # ------------------------------------------------ bond network
        h = self.spacing
        cutoff = 1.8 * h
        d = self.x0[None, :, :] - self.x0[:, None, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        iu, ju = np.triu_indices(self.n, k=1)
        keep = dist[iu, ju] <= cutoff
        self.bi = iu[keep]
        self.bj = ju[keep]
        self.b_r0 = dist[self.bi, self.bj]
        # Softer long (diagonal) bonds -> nearly isotropic effective stiffness.
        youngs = 1.2e7  # Pa, soft brittle masonry regime
        k_axial = youngs * h / 3.0
        self.b_k = k_axial * (h / self.b_r0)
        m_red = 0.5 * float(np.mean(self.mass))
        self.b_c = 2.0 * 0.06 * np.sqrt(self.b_k * m_red)
        # Randomised brittle strength (heterogeneous masonry -> branching cracks)
        u = rng.random(len(self.bi))
        self.b_eps_t = 0.018 * (0.7 + 0.6 * u)
        self.b_eps_c = -0.22 * (0.8 + 0.4 * rng.random(len(self.bi)))
        self.b_active = np.ones(len(self.bi), dtype=bool)
        self.bond_count0 = int(self.b_active.sum())

        # ------------------------------------------------ brittle anchors
        eps = 0.25 * h
        lo = self.x0.min(axis=0)
        hi = self.x0.max(axis=0)
        anchored = (
            (self.x0[:, 2] <= lo[2] + eps)
            | (self.x0[:, 2] >= hi[2] - eps)
            | (self.x0[:, 1] <= lo[1] + eps)
            | (self.x0[:, 1] >= hi[1] - eps)
        )
        self.anchor_idx = np.nonzero(anchored)[0]
        self.anchor_x0 = self.x0[self.anchor_idx].copy()
        self.anchor_k = 8.0e5
        self.anchor_c = 2.0 * 0.3 * np.sqrt(self.anchor_k * float(np.mean(self.mass)))
        self.anchor_break = 0.008 * (0.8 + 0.4 * rng.random(len(self.anchor_idx)))
        self.anchor_alive = np.ones(len(self.anchor_idx), dtype=bool)
        self.anchor_count0 = int(self.anchor_alive.sum())

        # ------------------------------------------------ contact model
        self.k_ground = 4.0e6
        self.k_ball = 5.0e6
        self.k_pp = 1.2e6
        m_ref = float(np.mean(self.mass))
        self.c_ground = 2.0 * 0.35 * np.sqrt(self.k_ground * m_ref)
        self.c_ball = 2.0 * 0.3 * np.sqrt(self.k_ball * m_ref)
        self.c_pp = 2.0 * 0.25 * np.sqrt(self.k_pp * 0.5 * m_ref)
        self.mu_ground = 0.55
        self.mu_ball = 0.4
        self.drag = 0.12  # mild air drag, 1/s
        self.v_cap = 60.0

        # ------------------------------------------------ pendulum state
        # Release angle from the downward vertical; the ball starts at rest and
        # gravity swings it into the wall face.
        self.pivot = np.array(
            [self.ball_x0[0] + (self.pivot_z - self.ball_x0[2]) * np.tan(np.deg2rad(35.0)),
             self.ball_x0[1],
             self.pivot_z],
            dtype=float,
        )
        rvec = self.ball_x0 - self.pivot
        self.cable_len = float(np.linalg.norm(rvec))
        self.theta = float(np.arctan2(rvec[0], -rvec[2]))  # negative: on -x side
        self.theta_dot = 0.0
        g = float(abs(self.gravity[2])) if abs(self.gravity[2]) > 1e-9 else 9.81
        self.g_mag = g
        self.ball_mass = 7800.0 * (4.0 / 3.0) * np.pi * self.ball_radius ** 3
        self.ball_inertia = self.ball_mass * (self.cable_len ** 2 + 0.4 * self.ball_radius ** 2)
        self.cable_damp = 0.02 * self.ball_inertia

        # Collision against the static wall slab is enabled per sample only once
        # that sample is fully clear of the slab surface.  Samples of the
        # destructible chunk start flush with (or slightly inside) the frozen
        # masonry, so activating them immediately would inject a spurious
        # penalty impulse at t=0 and eject the front column.
        self.slab_clear = self.radius + 0.01
        self.slab_active = self._slab_sdf(self.x) > self.slab_clear

        self.dt_max = 1.5e-4
        self.time = 0.0
        self.substeps = 0
        self.max_speed = 0.0
        self._F = np.repeat(np.eye(3)[None], self.n, axis=0)

    def _read_geometry(self) -> None:
        geom = self.config.get("scene_geometry", {}) or {}
        objects = geom.get("objects", []) or []
        self.ground_top = 0.0
        self.ball_x0 = np.array([-1.75, 0.0, 1.35])
        self.ball_radius = 0.48
        self.pivot_z = 3.35
        slab_lo, slab_hi = [], []
        for item in objects:
            oid = str(item.get("id", ""))
            g = item.get("geometry", {}) or {}
            t = np.asarray((item.get("transform", {}) or {}).get("translation", [0, 0, 0]), float)
            if oid == "ground" and g.get("type") == "box":
                he = np.asarray(g["half_extents"], float)
                self.ground_top = float(t[2] + he[2])
            elif oid.startswith("wall_row") and g.get("type") == "box":
                he = np.asarray(g["half_extents"], float)
                slab_lo.append(t - he)
                slab_hi.append(t + he)
            elif oid == "wrecking_ball":
                self.ball_x0 = t.astype(float)
                self.ball_radius = float(g.get("radius", 0.48))
            elif oid == "cable":
                self.pivot_z = float(t[2] + 0.5 * float(g.get("height", 1.8)))
        if slab_lo:
            self.slab_lo = np.min(np.asarray(slab_lo), axis=0)
            self.slab_hi = np.max(np.asarray(slab_hi), axis=0)
        else:
            self.slab_lo = np.array([-0.18, -0.96, 0.0])
            self.slab_hi = np.array([0.18, 0.96, 1.44])

    # ------------------------------------------------------------ geometry
    def _slab_sdf(self, x):
        """Signed distance to the static masonry slab (positive outside)."""
        c = 0.5 * (self.slab_lo + self.slab_hi)
        he = 0.5 * (self.slab_hi - self.slab_lo)
        q = np.abs(x - c) - he
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def _ball_center(self, theta=None):
        th = self.theta if theta is None else theta
        return self.pivot + self.cable_len * np.array([np.sin(th), 0.0, -np.cos(th)])

    def _ball_velocity(self):
        return self.cable_len * self.theta_dot * np.array(
            [np.cos(self.theta), 0.0, np.sin(self.theta)]
        )

    # ------------------------------------------------------------- physics
    def _forces(self):
        """Internal + contact forces on the samples and the reaction torque."""
        f = self.mass[:, None] * self.gravity[None, :]

        # --- bonds (brittle, irreversible breakage) ---
        act = self.b_active
        if np.any(act):
            i = self.bi[act]
            j = self.bj[act]
            d = self.x[j] - self.x[i]
            r = np.sqrt(np.sum(d * d, axis=1))
            r = np.maximum(r, 1e-12)
            nvec = d / r[:, None]
            r0 = self.b_r0[act]
            stretch = (r - r0) / r0
            vrel = np.sum((self.v[j] - self.v[i]) * nvec, axis=1)
            fmag = self.b_k[act] * (r - r0) + self.b_c[act] * vrel
            fvec = fmag[:, None] * nvec
            np.add.at(f, i, fvec)
            np.add.at(f, j, -fvec)
            broken = (stretch > self.b_eps_t[act]) | (stretch < self.b_eps_c[act])
            if np.any(broken):
                idx = np.nonzero(act)[0][broken]
                self.b_active[idx] = False

        # --- brittle anchors to the surrounding static masonry ---
        if len(self.anchor_idx) and np.any(self.anchor_alive):
            alive = self.anchor_alive
            ai = self.anchor_idx[alive]
            disp = self.x[ai] - self.anchor_x0[alive]
            fa = -self.anchor_k * disp - self.anchor_c * self.v[ai]
            np.add.at(f, ai, fa)
            snap = np.linalg.norm(disp, axis=1) > self.anchor_break[alive]
            if np.any(snap):
                self.anchor_alive[np.nonzero(alive)[0][snap]] = False

        # --- rigid floor contact with Coulomb friction ---
        gap = (self.x[:, 2] - self.radius) - self.ground_top
        hit = gap < 0.0
        if np.any(hit):
            pen = -gap[hit]
            vn = self.v[hit, 2]
            fn = self.k_ground * pen - self.c_ground * vn
            fn = np.maximum(fn, 0.0)
            f[hit, 2] += fn
            vt = self.v[hit][:, :2]
            sp = np.linalg.norm(vt, axis=1)
            scale = np.where(sp > 1e-6, np.minimum(self.mu_ground * fn / np.maximum(sp, 1e-6) * 1.0, 1e6), 0.0)
            ft = -vt * scale[:, None]
            lim = self.mu_ground * fn
            ftn = np.linalg.norm(ft, axis=1)
            over = ftn > lim
            if np.any(over):
                ft[over] = ft[over] * (lim[over] / np.maximum(ftn[over], 1e-12))[:, None]
            hit_idx = np.nonzero(hit)[0]
            np.add.at(f, hit_idx, np.column_stack((ft[:, 0], ft[:, 1], np.zeros(len(hit_idx)))))

        # --- static masonry slab contact (only once a sample has left it) ---
        sdf = self._slab_sdf(self.x)
        self.slab_active |= sdf > self.slab_clear
        pen_mask = self.slab_active & (sdf < self.radius)
        if np.any(pen_mask):
            xi = self.x[pen_mask]
            c = 0.5 * (self.slab_lo + self.slab_hi)
            nvec = xi - c
            he = 0.5 * (self.slab_hi - self.slab_lo)
            q = np.abs(nvec) - he
            axis = np.argmax(q, axis=1)
            nrm = np.zeros_like(xi)
            nrm[np.arange(len(xi)), axis] = np.sign(nvec[np.arange(len(xi)), axis] + 1e-12)
            pen = self.radius[pen_mask] - sdf[pen_mask]
            vn = np.sum(self.v[pen_mask] * nrm, axis=1)
            fn = np.maximum(self.k_ground * pen - self.c_ground * vn, 0.0)
            np.add.at(f, np.nonzero(pen_mask)[0], fn[:, None] * nrm)

        # --- sample/sample repulsion (fragment piling) ---
        d = self.x[None, :, :] - self.x[:, None, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        rsum = self.radius[:, None] + self.radius[None, :]
        pm = dist < rsum
        if np.any(pm):
            ii, jj = np.nonzero(np.triu(pm, 1))
            dv = self.x[jj] - self.x[ii]
            rr = np.maximum(np.linalg.norm(dv, axis=1), 1e-12)
            nvec = dv / rr[:, None]
            pen = (self.radius[ii] + self.radius[jj]) - rr
            vrel = np.sum((self.v[jj] - self.v[ii]) * nvec, axis=1)
            fmag = np.maximum(self.k_pp * pen - self.c_pp * vrel, 0.0)
            fvec = -fmag[:, None] * nvec
            np.add.at(f, ii, fvec)
            np.add.at(f, jj, -fvec)

        # --- wrecking ball contact (two-way coupled with the pendulum) ---
        center = self._ball_center()
        vball = self._ball_velocity()
        dc = self.x - center[None, :]
        rc = np.maximum(np.linalg.norm(dc, axis=1), 1e-12)
        bgap = rc - (self.ball_radius + self.radius)
        bm = bgap < 0.0
        torque = 0.0
        if np.any(bm):
            nvec = dc[bm] / rc[bm][:, None]
            pen = -bgap[bm]
            vrel = self.v[bm] - vball[None, :]
            vn = np.sum(vrel * nvec, axis=1)
            fn = np.maximum(self.k_ball * pen - self.c_ball * vn, 0.0)
            vt = vrel - vn[:, None] * nvec
            sp = np.linalg.norm(vt, axis=1)
            ft = np.where(sp[:, None] > 1e-6, -vt / np.maximum(sp, 1e-6)[:, None] * (self.mu_ball * fn)[:, None], 0.0)
            fvec = fn[:, None] * nvec + ft
            np.add.at(f, np.nonzero(bm)[0], fvec)
            reaction = -np.sum(fvec, axis=0)
            arm = center - self.pivot
            torque = float(arm[2] * reaction[0] - arm[0] * reaction[2])

        # --- mild air drag (numerical hygiene, physically small) ---
        f -= self.drag * self.mass[:, None] * self.v
        return f, torque, bgap, sdf, gap

    def _step(self, dt: float) -> None:
        f, torque, _, _, _ = self._forces()
        self.v += dt * (f / self.mass[:, None])
        sp = np.linalg.norm(self.v, axis=1)
        over = sp > self.v_cap
        if np.any(over):
            self.v[over] *= (self.v_cap / sp[over])[:, None]
        self.x += dt * self.v
        # pendulum: I*theta_dd = -m g L sin(theta) - c*theta_dot + torque
        acc = (
            -self.ball_mass * self.g_mag * self.cable_len * np.sin(self.theta)
            - self.cable_damp * self.theta_dot
            + torque
        ) / self.ball_inertia
        self.theta_dot += dt * acc
        self.theta += dt * self.theta_dot
        self.substeps += 1
        self.max_speed = max(self.max_speed, float(sp.max()) if self.n else 0.0)

    # ------------------------------------------------------ deformation
    def _deformation_gradient(self) -> np.ndarray:
        act = self.b_active
        i = np.concatenate([self.bi[act], self.bj[act]])
        j = np.concatenate([self.bj[act], self.bi[act]])
        F = np.repeat(np.eye(3)[None], self.n, axis=0)
        if len(i) == 0:
            return F
        X = self.x0[j] - self.x0[i]
        xcur = self.x[j] - self.x[i]
        w = 1.0 / np.maximum(np.linalg.norm(X, axis=1), 1e-12)
        A = np.zeros((self.n, 3, 3))
        B = np.zeros((self.n, 3, 3))
        np.add.at(A, i, w[:, None, None] * (X[:, :, None] * X[:, None, :]))
        np.add.at(B, i, w[:, None, None] * (xcur[:, :, None] * X[:, None, :]))
        cnt = np.zeros(self.n)
        np.add.at(cnt, i, 1.0)
        lam = 1e-6 * self.spacing ** 2
        A += lam * np.eye(3)[None]
        ok = cnt >= 4
        if np.any(ok):
            try:
                F[ok] = B[ok] @ np.linalg.inv(A[ok])
            except np.linalg.LinAlgError:
                pass
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(F, -10.0, 10.0)

    # ----------------------------------------------------------- stepping
    def advance_to(self, time: float):
        target = float(time)
        if target > self.time + 1e-12:
            span = target - self.time
            n = int(np.ceil(span / self.dt_max))
            dt = span / n
            for _ in range(n):
                self._step(dt)
            self.time = target
        return self._frame(target)

    def _gaps(self):
        """Pure signed-gap query (no state mutation, no breakage evaluation)."""
        ggap = (self.x[:, 2] - self.radius) - self.ground_top
        center = self._ball_center()
        rc = np.maximum(np.linalg.norm(self.x - center[None, :], axis=1), 1e-12)
        bgap = rc - (self.ball_radius + self.radius)
        slab_sdf = self._slab_sdf(self.x)
        return bgap, slab_sdf, ggap

    def _frame(self, t: float):
        bgap, slab_sdf, ggap = self._gaps()
        contact_gap = np.minimum(ggap, bgap)
        slab_gap = np.where(self.slab_active, slab_sdf - self.radius, np.inf)
        contact_gap = np.minimum(contact_gap, slab_gap)
        contact_gap = np.nan_to_num(
            np.clip(contact_gap, -10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0
        )
        self._F = self._deformation_gradient()
        ext = self.mass[:, None] * self.gravity[None, :]

        frame = StateFrame(
            time=float(t),
            position=np.ascontiguousarray(self.x.copy()),
            velocity=np.ascontiguousarray(self.v.copy()),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": self.radius.copy(),
                "material_label": self.material_label.copy(),
                "deformation_gradient": self._F.copy(),
            },
            fields={
                "external_force": np.ascontiguousarray(ext),
                "contact_signed_distance": contact_gap,
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "contact_length_scale": float(self.spacing),
                "grid_spacing": float(self.spacing),
                "characteristic_length": float(
                    max(float(np.ptp(self.x0, axis=0).max()), self.spacing)
                ),
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    # -------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "bonded-particle peridynamic brittle fracture + rigid pendulum",
            "substeps": int(self.substeps),
            "substep_dt": float(self.dt_max),
            "samples": int(self.n),
            "bonds_initial": int(self.bond_count0),
            "bonds_remaining": int(self.b_active.sum()),
            "anchors_initial": int(self.anchor_count0),
            "anchors_remaining": int(self.anchor_alive.sum()),
            "max_sample_speed": float(self.max_speed),
            "cable_length": float(self.cable_len),
            "ball_mass": float(self.ball_mass),
            "final_theta": float(self.theta),
        }
