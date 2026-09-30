"""Wrecking-ball impact on a brittle masonry wall.

Physical model
--------------
* The subject samples form a bonded-particle lattice representing the brittle
  masonry core of the wall.  Every initial neighbour pair within
  ``1.8 * spacing`` carries a linear elastic central spring with light viscous
  damping.  A bond fails permanently (brittle rupture) once its engineering
  strain magnitude exceeds the failure threshold, which creates the newly
  formed crack surfaces.
* Particles whose bonds have broken interact through unilateral, frictional
  point contacts so that separated fragments collide, pile up and come to
  rest on the floor instead of interpenetrating.
* The floor is a penalty contact with Coulomb friction.
* The heavy wrecking ball is a rigid pendulum: a point mass on a massless
  inextensible cable anchored above the wall.  The ball is released from rest
  at its frozen initial pose, swings down under gravity and strikes the wall.
  Contact forces from the masonry feed back onto the pendulum as a generalized
  force, so the ball slows while ploughing through the wall and keeps
  oscillating afterwards.
* Time integration is explicit symplectic (semi-implicit) Euler with a fixed
  substep chosen well below the elastic CFL limit of the stiffest spring, and
  a velocity clamp guards against numerical blow-up.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # --- numerical / material parameters -----------------------------------
    SUBSTEPS_PER_SECOND = 6000.0
    BOND_RC_FACTOR = 1.8          # neighbour bond cutoff in units of spacing
    K_BOND = 1.0e6                # N/m   bond spring stiffness
    C_BOND = 60.0                 # N s/m bond damping
    BREAK_STRAIN = 0.035          # brittle failure strain (tension/compression)
    K_CONTACT = 2.0e5             # N/m   fragment-fragment contact
    C_CONTACT = 120.0
    MU_CONTACT = 0.45
    K_GROUND = 3.0e5              # N/m   floor contact
    C_GROUND = 200.0
    MU_GROUND = 0.6
    K_BALL = 6.0e5                # N/m   ball contact
    C_BALL = 300.0
    MU_BALL = 0.30
    CABLE_DAMPING = 4.0           # N m s  pendulum cable damping
    BALL_DENSITY = 7800.0         # kg/m^3 solid steel ball
    VMAX = 25.0                   # m/s safety clamp
    NEAR_CONTACT_BAND = 0.30      # m  band for reported contact gaps

    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        cfg = getattr(scene, "config", None) or {}
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), float)
        sampling = cfg.get("sampling", {})
        self.spacing = float(sampling.get("spacing", 0.12))
        density = float(sampling.get("density", 1350.0))

        self.pos = np.array(frame0.position, float)
        self.vel = np.zeros_like(self.pos)
        self.n = len(self.pos)
        self.mass = np.array(frame0.mass, float)
        if self.mass is None or self.mass.shape != (self.n,):
            self.mass = np.full(self.n, density * self.spacing ** 3)
        self.body_id = np.array(frame0.body_id, np.int32) if frame0.body_id is not None \
            else np.zeros(self.n, np.int32)
        self.mass_quantity = self.mass.copy()
        self.particle_radius = np.asarray(
            frame0.attributes.get(
                "particle_radius", np.full(self.n, 0.45 * self.spacing)
            ), float
        ).copy()
        self.material_label = np.asarray(
            frame0.attributes.get("material_label", self.body_id)
        ).copy()
        self.init_scalars = dict(frame0.scalars)

        self.h = 1.0 / float(self.SUBSTEPS_PER_SECOND)
        self.t = 0.0
        self.ext_impulse = np.zeros((self.n, 3))
        self.ext_ref_time = 0.0
        self.bonds_broken = 0
        self.steps_taken = 0

        self._setup_geometry(cfg)
        self._setup_lattice()

    # ------------------------------------------------------------------ setup
    def _setup_geometry(self, cfg: dict) -> None:
        objects = (cfg.get("scene_geometry") or {}).get("objects", [])
        boxes = []
        ball_center = None
        ball_radius = 0.48
        cable = None
        wall_centers = []
        for obj in objects:
            if obj.get("id") == "subject":
                continue
            geom = obj.get("geometry", {})
            trans = obj.get("transform", {})
            center = np.asarray(trans.get("translation", (0.0, 0.0, 0.0)), float)
            gtype = geom.get("type")
            if gtype == "box":
                half = np.asarray(geom["half_extents"], float)
                boxes.append((center, half, str(obj.get("material_id", ""))))
                if obj.get("material_id") == "masonry":
                    wall_centers.append(center)
            elif gtype == "sphere" and obj.get("motion") == "kinematic":
                ball_center = center
                ball_radius = float(geom.get("radius", 0.48))
            elif gtype == "capsule":
                cable = (center, float(geom.get("radius", 0.05)),
                         float(geom.get("height", 1.8)))
        self.boxes = boxes
        self.ball_radius = ball_radius

        # floor plane: top surface of the ground box
        floor_z = 0.0
        for center, half, _ in boxes:
            if center[2] <= 0.0 and half[2] < half[0]:
                floor_z = center[2] + half[2]
        self.floor_z = floor_z

        # pendulum anchor: directly above the wall centre, at the top of the
        # suspension cable.  The ball hangs from the cable and is released
        # from its frozen (displaced) initial pose.
        anchor_z = 3.4
        if cable is not None:
            c, r, height = cable
            anchor_z = c[2] + 0.5 * height + r
        if wall_centers:
            anchor_xy = np.mean(np.asarray(wall_centers)[:, :2], axis=0)
        else:
            anchor_xy = np.zeros(2)
        self.pivot = np.array([anchor_xy[0], anchor_xy[1], anchor_z])
        if ball_center is None:
            ball_center = np.array([-1.75, 0.0, 1.35])
        self.cable_length = float(np.linalg.norm(ball_center - self.pivot))
        rel = ball_center - self.pivot
        # theta measured from the downward vertical, swing plane y = anchor y
        self.theta = float(np.arctan2(rel[0], -rel[2]))
        self.omega = 0.0
        self.ball_mass = self.BALL_DENSITY * (4.0 / 3.0) * np.pi * ball_radius ** 3
        self.ball_center0 = ball_center.copy()
        self._ball_state(self.theta)

    def _setup_lattice(self) -> None:
        pos = self.pos
        n = self.n
        rc_bond = self.BOND_RC_FACTOR * self.spacing
        iu, ju = np.triu_indices(n, k=1)
        d0 = np.linalg.norm(pos[ju] - pos[iu], axis=1)
        bond_mask = d0 <= rc_bond
        self.bi = iu[bond_mask].copy()
        self.bj = ju[bond_mask].copy()
        self.bL0 = d0[bond_mask].copy()
        self.bref = pos[self.bj] - pos[self.bi]           # material-space offsets
        self.bactive = np.ones(len(self.bi), bool)
        self.broken_count = 0

        # map pair -> bond index for contact exclusion of intact bonds
        bond_of_pair = np.full((n, n), -1, np.int64)
        bond_of_pair[self.bi, self.bj] = np.arange(len(self.bi))
        self.pui = iu
        self.pju = ju
        self.pbond = bond_of_pair[iu, ju]
        self.contact_radius = 0.5 * self.spacing

    # ------------------------------------------------------------------ ball
    def _ball_state(self, theta: float) -> None:
        L = self.cable_length
        c = self.pivot + np.array([L * np.sin(theta), 0.0, -L * np.cos(theta)])
        v = np.array([L * np.cos(theta), 0.0, L * np.sin(theta)]) * self.omega
        self.ball_pos = c
        self.ball_vel = v

    # -------------------------------------------------------------- stepping
    def _step(self) -> None:
        h = self.h
        pos, vel = self.pos, self.vel
        f = np.zeros_like(pos)
        f_ext = np.repeat(self.gravity[None, :], self.n, axis=0) * self.mass[:, None]
        inv_m = 1.0 / self.mass

        # --- brittle bonds -------------------------------------------------
        if np.any(self.bactive):
            dvec = pos[self.bj] - pos[self.bi]
            d = np.linalg.norm(dvec, axis=1)
            d = np.maximum(d, 1e-12)
            strain = (d - self.bL0) / self.bL0
            newly_broken = self.bactive & (np.abs(strain) > self.BREAK_STRAIN)
            if np.any(newly_broken):
                self.bactive &= ~newly_broken
                self.broken_count += int(np.count_nonzero(newly_broken))
            act = self.bactive
            nh = dvec / d[:, None]
            vn = np.einsum("ij,ij->i", vel[self.bj] - vel[self.bi], nh)
            fs = self.K_BOND * (d - self.bL0) + self.C_BOND * vn
            fs = fs * act
            fb = fs[:, None] * nh
            np.add.at(f, self.bi[act], fb[act])
            np.add.at(f, self.bj[act], -fb[act])

        # --- fragment-fragment unilateral frictional contact ---------------
        dv = pos[self.pju] - pos[self.pui]
        dd = np.linalg.norm(dv, axis=1)
        dd = np.maximum(dd, 1e-12)
        overlap = 2.0 * self.contact_radius - dd
        intact = (self.pbond >= 0) & self.bactive[np.maximum(self.pbond, 0)]
        cand = (overlap > 0.0) & ~intact
        if np.any(cand):
            nh = dv / dd[:, None]
            relv = vel[self.pju] - vel[self.pui]
            vn = np.einsum("ij,ij->i", relv, nh)
            fn = np.maximum(0.0, self.K_CONTACT * overlap - self.C_CONTACT * np.minimum(vn, 0.0))
            fn = fn * cand
            vt = relv - vn[:, None] * nh
            vt_norm = np.linalg.norm(vt, axis=1)
            vt_mag = np.maximum(vt_norm, 1e-6)
            cap = self.MU_CONTACT * fn
            ft_mag = np.minimum(cap, self.C_CONTACT * vt_norm)
            ft = (ft_mag / vt_mag)[:, None] * vt * cand[:, None]
            fp = fn[:, None] * nh
            np.add.at(f, self.pui[cand], (-fp + ft)[cand])
            np.add.at(f, self.pju[cand], (fp - ft)[cand])

        # --- wrecking ball contact (external) ------------------------------
        rvec = pos - self.ball_pos
        rdist = np.linalg.norm(rvec, axis=1)
        rdist = np.maximum(rdist, 1e-9)
        bover = (self.ball_radius + self.contact_radius) - rdist
        bmask = bover > 0.0
        if np.any(bmask):
            nh = rvec / rdist[:, None]
            relv = vel - self.ball_vel
            vn = np.einsum("ij,ij->i", relv, nh)
            fn = np.maximum(0.0, self.K_BALL * bover - self.C_BALL * np.minimum(vn, 0.0))
            vt = relv - vn[:, None] * nh
            vt_norm = np.linalg.norm(vt, axis=1)
            vt_mag = np.maximum(vt_norm, 1e-6)
            ft_mag = np.minimum(self.MU_BALL * fn, 80.0 * vt_norm)
            ft = (ft_mag / vt_mag)[:, None] * vt
            fb = fn[:, None] * nh - ft
            f[bmask] += fb[bmask]
            f_ext[bmask] += fb[bmask]
            self.ball_reaction = -np.sum(fb[bmask], axis=0)
        else:
            self.ball_reaction = np.zeros(3)

        # --- floor contact (external) --------------------------------------
        gover = self.contact_radius - (pos[:, 2] - self.floor_z)
        gmask = gover > 0.0
        if np.any(gmask):
            vz = vel[gmask, 2]
            fn = np.maximum(0.0, self.K_GROUND * gover[gmask] - self.C_GROUND * np.minimum(vz, 0.0))
            fg = np.zeros((int(np.count_nonzero(gmask)), 3))
            fg[:, 2] = fn
            vxy = vel[gmask, :2]
            vxy_norm = np.linalg.norm(vxy, axis=1)
            vxy_mag = np.maximum(vxy_norm, 1e-6)
            ft_mag = np.minimum(self.MU_GROUND * fn, 200.0 * vxy_norm)
            fg[:, :2] = -(ft_mag / vxy_mag)[:, None] * vxy
            f[gmask] += fg
            f_ext[gmask] += fg

        # --- integrate particles (semi-implicit Euler) ----------------------
        vel = vel + h * f * inv_m[:, None]
        speed = np.linalg.norm(vel, axis=1)
        fast = speed > self.VMAX
        if np.any(fast):
            vel[fast] *= (self.VMAX / speed[fast])[:, None]
        pos = pos + h * vel
        self.pos, self.vel = pos, vel

        # --- integrate the pendulum (ball) ----------------------------------
        L = self.cable_length
        tangent = np.array([L * np.cos(self.theta), 0.0, L * np.sin(self.theta)])
        Q = float(np.dot(self.ball_reaction, tangent))
        torque = -self.ball_mass * (-self.gravity[2]) * L * np.sin(self.theta) \
            - self.CABLE_DAMPING * self.omega + Q
        alpha = torque / (self.ball_mass * L * L)
        self.omega += h * alpha
        self.theta += h * self.omega
        self._ball_state(self.theta)

        self.ext_impulse += f_ext * h
        self.steps_taken += 1

    # ------------------------------------------------------------- sampling
    def _deformation_gradients(self) -> np.ndarray:
        eye = np.eye(3)
        F = np.repeat(eye[None, :, :], self.n, axis=0)
        pos = self.pos
        # per-particle bond adjacency (material-space bond offsets bref)
        if not hasattr(self, "_adj_lists"):
            adj = [[] for _ in range(self.n)]
            for k in range(len(self.bi)):
                adj[self.bi[k]].append((k, +1))
                adj[self.bj[k]].append((k, -1))
            self._adj_lists = adj
        for i in range(self.n):
            rows = self._adj_lists[i]
            if len(rows) < 4:
                continue
            xi = []
            dx = []
            for k, sign in rows:
                if not self.bactive[k]:
                    continue
                xi.append(sign * self.bref[k])
                dx.append(sign * (pos[self.bj[k]] - pos[self.bi[k]]))
            if len(xi) < 4:
                continue
            X = np.asarray(xi)          # material-space offsets
            U = np.asarray(dx)          # current offsets
            A = X.T @ X
            B = U.T @ X
            try:
                Fi = np.linalg.solve(A.T, B.T).T
                if np.all(np.isfinite(Fi)):
                    F[i] = Fi
            except np.linalg.LinAlgError:
                continue
        return F

    def _contact_gaps(self) -> np.ndarray:
        gaps = []
        pos = self.pos
        for center, half, _ in self.boxes:
            q = np.abs(pos - center) - half
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(np.max(q, axis=1), 0.0)
            gaps.append(outside + inside)
        # current (kinematic) ball position
        gaps.append(np.linalg.norm(pos - self.ball_pos, axis=1) - self.ball_radius)
        sdf = np.min(np.asarray(gaps), axis=0)
        near = sdf < self.NEAR_CONTACT_BAND
        return sdf[near]

    def _frame(self, t: float) -> StateFrame:
        dt_ext = max(t - self.ext_ref_time, 1e-9)
        external_force = self.ext_impulse / dt_ext
        if t <= 0.0:
            external_force = np.repeat(
                (self.mass[:, None] * self.gravity[None, :]), 1, axis=0
            )
        gaps = self._contact_gaps()
        scalars = dict(self.init_scalars)
        scalars["contact_length_scale"] = float(self.spacing)
        frame = StateFrame(
            time=float(t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass_quantity.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.material_label.copy(),
                "deformation_gradient": self._deformation_gradients(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1),
                "external_force": external_force.reshape(self.n, 3),
                "contact_signed_distance": np.asarray(gaps, float),
            },
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float):
        target = float(time)
        if target < self.t - 1e-9:
            raise ValueError("advance_to called with a decreasing time")
        n_steps = int(np.floor((target - self.t) / self.h + 1e-6))
        for _ in range(n_steps):
            self._step()
        self.t = target
        frame = self._frame(target)
        self.ext_impulse[:] = 0.0
        self.ext_ref_time = target
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": self.steps_taken,
            "bonds_initial": int(len(self.bi)),
            "bonds_broken": int(self.broken_count),
            "ball_theta": float(self.theta),
            "ball_omega": float(self.omega),
            "cable_length": float(self.cable_length),
            "ball_mass": float(self.ball_mass),
        }
