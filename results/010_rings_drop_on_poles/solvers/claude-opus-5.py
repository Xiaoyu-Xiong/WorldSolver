"""Rigid-body solver: nonconvex rings dropping onto a vertical pole array.

Physical model
--------------
Each ring of the frozen ``axisymmetric_container`` collection is a rigid body
whose collision geometry is a closed *necklace* of overlapping spheres placed on
the ring tube centre circle.  That proxy is genuinely nonconvex: another ring or
a pole passes freely through the hole, but nothing crosses the tube because
neighbouring beads overlap.  Threading, hooking and interlocking therefore
emerge from the contact law instead of being scripted.

Contacts (ring/ring, ring/ground, ring/pole) use a linear penalty
spring-dashpot normal law with regularised Coulomb friction.  Bodies advance
with a fixed-step semi-implicit (symplectic) Euler scheme; rotations are
integrated in the body frame with the exact torus inertia tensor and quaternion
renormalisation.  Everything is deterministic: fixed step size, fixed operation
order, no wall-clock or random adaptation.

Exported trajectory conventions
-------------------------------
``fields.external_force``    gravity + air drag + reactions from the *static*
                             boundaries (receiving ground box, poles).  Internal
                             ring/ring pairs are excluded, so the global balance
                             d(sum m v)/dt = sum F_ext holds.
``fields.interaction_force`` one row per near/active internal ring-ring contact
                             sample: the force applied to the first body of the
                             pair (the second body receives its negative).
``fields.contact_signed_distance`` signed gaps of every active and near-contact
                             sample (ring/ground, ring/pole, ring/ring).
"""
from __future__ import annotations

import time as _time

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

GRAVITY = 9.81


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices for a stack of xyzw quaternions."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.empty((q.shape[0], 3, 3), dtype=float)
    m[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    m[:, 0, 1] = 2.0 * (xy - wz)
    m[:, 0, 2] = 2.0 * (xz + wy)
    m[:, 1, 0] = 2.0 * (xy + wz)
    m[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    m[:, 1, 2] = 2.0 * (yz - wx)
    m[:, 2, 0] = 2.0 * (xz - wy)
    m[:, 2, 1] = 2.0 * (yz + wx)
    m[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return m


def _scatter(target: np.ndarray, idx: np.ndarray, values: np.ndarray, n: int) -> None:
    """Accumulate per-sample vectors into per-body rows."""
    if idx.size == 0:
        return
    for c in range(3):
        target[:, c] += np.bincount(idx, weights=values[:, c], minlength=n)


class BenchmarkSolver:
    """Penalty-contact rigid-body solver for the rings/poles benchmark."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.duration = float(cfg.get("duration", 8.0))

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=float).reshape(-1, 3).copy()
        self.n = int(self.pos.shape[0])
        vel = getattr(frame0, "velocity", None)
        self.vel = (
            np.array(vel, dtype=float).reshape(-1, 3).copy()
            if vel is not None
            else np.zeros((self.n, 3))
        )
        quat = getattr(frame0, "orientation", None)
        if quat is None:
            self.quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n, 1))
        else:
            self.quat = np.array(quat, dtype=float).reshape(self.n, 4).copy()
        self._normalize_quat()
        omega = getattr(frame0, "angular_velocity", None)
        self.omega = (
            np.array(omega, dtype=float).reshape(-1, 3).copy()
            if omega is not None
            else np.zeros((self.n, 3))
        )
        mass = getattr(frame0, "mass", None)
        self.mass = (
            np.array(mass, dtype=float).reshape(-1).copy()
            if mass is not None
            else np.ones(self.n)
        )
        self.mass = np.where(np.isfinite(self.mass) & (self.mass > 1e-9), self.mass, 1.0)
        self.body_id = np.arange(self.n, dtype=np.int64)

        self._parse_geometry(cfg.get("scene_geometry"))
        self._build_proxy()
        self._contact_parameters()

        # Fixed integration step; an integer number of substeps per sample.
        target = 1.25e-4
        self.substeps = max(1, int(round(self.sample_dt / target)))
        self.dt = self.sample_dt / self.substeps
        self.time = 0.0
        self.step_index = 0
        self._tree_pairs = np.empty((0, 2), dtype=np.int64)
        self._pairs_age = 10 ** 9
        self._max_penetration = 0.0
        self._steps_done = 0
        self._sim_seconds = 0.0
        self._frames_emitted = 0

    def _normalize_quat(self) -> None:
        norm = np.linalg.norm(self.quat, axis=1)
        norm = np.where(norm < 1e-12, 1.0, norm)
        self.quat = self.quat / norm[:, None]

    def _parse_geometry(self, geom) -> None:
        """Read the frozen ground box, pole array and ring profile."""
        self.ground_z = 0.0
        self.ground_half = np.array([3.2, 3.2])
        self.ground_center = np.zeros(2)
        poles: list[tuple[float, float, float, float, float]] = []
        profile = None
        wall = 0.06
        objects = []
        if isinstance(geom, dict):
            objects = geom.get("objects", []) or []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            g = obj.get("geometry", {}) or {}
            tr = (obj.get("transform", {}) or {}).get("translation", [0.0, 0.0, 0.0])
            tr = [float(v) for v in list(tr) + [0.0, 0.0, 0.0]][:3]
            gtype = g.get("type")
            semantic = obj.get("semantic")
            motion = obj.get("motion")
            enabled = bool(obj.get("collision_enabled", True))
            if gtype == "box" and semantic == "boundary":
                he = [float(v) for v in g.get("half_extents", [3.2, 3.2, 0.08])]
                self.ground_z = tr[2] + he[2]
                self.ground_half = np.array([he[0], he[1]])
                self.ground_center = np.array([tr[0], tr[1]])
            elif gtype == "cylinder" and motion in {"static", "kinematic"} and enabled:
                r = float(g.get("radius", 0.075))
                h = float(g.get("height", 2.6))
                poles.append((tr[0], tr[1], tr[2] - 0.5 * h, tr[2] + 0.5 * h, r))
            elif gtype == "axisymmetric_container" and motion == "dynamic":
                prof = g.get("inner_profile")
                if prof:
                    profile = np.asarray(prof, dtype=float)
                wall = float(g.get("wall_thickness", wall) or wall)
        if not poles:
            poles = [
                (x, y, 0.0, 2.6, 0.075)
                for x in (-0.9, 0.0, 0.9)
                for y in (-0.9, 0.0, 0.9)
            ]
        poles_arr = np.asarray(poles, dtype=float)
        self.pole_xy = poles_arr[:, 0:2].copy()
        self.pole_z0 = poles_arr[:, 2].copy()
        self.pole_z1 = poles_arr[:, 3].copy()
        self.pole_r = poles_arr[:, 4].copy()
        self.wall_thickness = max(float(wall), 1e-3)
        if profile is None:
            profile = np.array(
                [[0.18, 0.0], [0.28, 0.02], [0.34, 0.08],
                 [0.34, 0.18], [0.28, 0.24], [0.18, 0.26]]
            )
        self.profile = profile

    def _build_proxy(self) -> None:
        """Size the nonconvex bead-necklace proxy of one ring."""
        prof = self.profile
        r_equator = float(prof[:, 0].max())
        z_span = float(prof[:, 1].max() - prof[:, 1].min())
        self.bead_r = 0.5 * self.wall_thickness
        # Mid-surface radius of the revolved shell.
        r_geom = r_equator + 0.5 * self.wall_thickness

        # The frozen initial layout fixes an upper bound on the proxy radius:
        # coplanar neighbours must not start interpenetrating.
        d = self.pos[:, None, :] - self.pos[None, :, :]
        planar = np.linalg.norm(d[:, :, 0:2], axis=2)
        coplanar = np.abs(d[:, :, 2]) <= max(z_span, 2.0 * self.bead_r)
        np.fill_diagonal(coplanar, False)
        if np.any(coplanar):
            d_min = float(planar[coplanar].min())
        else:
            d_min = 2.0 * r_geom + 2.0 * self.bead_r
        r_layout = 0.5 * max(d_min - 2.0 * self.bead_r, 4.0 * self.bead_r)

        # Threading requirement: the hole must clear the thickest pole.
        r_thread = float(self.pole_r.max()) + 2.0 * self.bead_r
        self.ring_R = float(max(min(r_geom, r_layout), r_thread))

        # Bead count so that the surface gap between neighbouring beads stays
        # below one bead diameter; nothing can then tunnel through the tube.
        ratio = float(np.clip(1.5 * self.bead_r / self.ring_R, 1e-3, 0.99))
        k = int(np.ceil(np.pi / np.arcsin(ratio)))
        self.k_beads = int(np.clip(k, 12, 28))

        ang = 2.0 * np.pi * np.arange(self.k_beads) / self.k_beads
        self.offsets = np.stack(
            [self.ring_R * np.cos(ang), self.ring_R * np.sin(ang), np.zeros(self.k_beads)],
            axis=1,
        )
        self.bead_body = np.repeat(np.arange(self.n, dtype=np.int64), self.k_beads)
        self.n_beads = self.n * self.k_beads
        self.bead_mass = self.mass / float(self.k_beads)

        # Exact solid-torus inertia (body frame, symmetry axis = z).
        rr, tt = self.ring_R, self.bead_r
        self.inertia = np.stack(
            [
                self.mass * (0.5 * rr * rr + 0.625 * tt * tt),
                self.mass * (0.5 * rr * rr + 0.625 * tt * tt),
                self.mass * (rr * rr + 0.75 * tt * tt),
            ],
            axis=1,
        )
        self.inv_inertia = 1.0 / np.maximum(self.inertia, 1e-12)

    def _contact_parameters(self) -> None:
        self.kn = 6.0e5              # normal penalty stiffness [N/m]
        self.cn = 5.0e2              # normal dashpot [N s/m]
        self.ct = 2.0e2              # tangential viscous regularisation [N s/m]
        self.mu_ring = 0.45
        self.mu_ground = 0.50
        self.mu_pole = 0.35
        self.f_max = 4.0e4           # safety clamp on a single normal force [N]
        self.lin_drag = 0.12         # air drag [N s/m]
        self.ang_drag = 0.06         # rotational drag [N m s/rad]
        self.v_max = 40.0
        self.w_max = 80.0
        self.report_margin = 0.2 * self.wall_thickness   # near-contact band
        self.skin = 0.05                                 # Verlet-list skin
        self.pair_every = 3
        self.pair_radius = 2.0 * self.bead_r + self.skin
        self.contact_length_scale = float(self.wall_thickness)

    # -------------------------------------------------------------- kinematics
    def _bead_state(self):
        """World bead arms, centres and velocities of every ring necklace."""
        rot = _quat_to_matrix(self.quat)
        arms = np.einsum("nij,kj->nki", rot, self.offsets)
        centers = self.pos[:, None, :] + arms
        bvel = self.vel[:, None, :] + np.cross(
            np.broadcast_to(self.omega[:, None, :], arms.shape), arms
        )
        return rot, arms.reshape(-1, 3), centers.reshape(-1, 3), bvel.reshape(-1, 3)

    def _contact_force(self, normal, relvel, depth, mu):
        """Penalty normal spring-dashpot plus regularised Coulomb friction."""
        vn = np.einsum("ij,ij->i", relvel, normal)
        fn = np.where(depth > 0.0, np.clip(self.kn * depth - self.cn * vn, 0.0, self.f_max), 0.0)
        vt = relvel - vn[:, None] * normal
        vt_mag = np.linalg.norm(vt, axis=1)
        ft = np.minimum(self.ct * vt_mag, mu * fn)
        scale = np.where(vt_mag > 1e-9, ft / np.maximum(vt_mag, 1e-12), 0.0)
        return fn[:, None] * normal - scale[:, None] * vt

    # ----------------------------------------------------------------- forces
    def _forces(self, rot, arms, centers, bvel, report):
        n = self.n
        ext = np.zeros((n, 3))       # gravity + drag + static-boundary reactions
        fint = np.zeros((n, 3))      # internal ring-ring contacts
        torque = np.zeros((n, 3))
        ext[:, 2] -= self.mass * GRAVITY
        ext -= self.lin_drag * self.vel
        torque -= self.ang_drag * self.omega

        thresh = self.report_margin if report else 0.0
        gaps: list[np.ndarray] = []
        pair_forces: list[np.ndarray] = []

        # --- receiving ground box (top face) -------------------------------
        inside = (
            np.abs(centers[:, 0] - self.ground_center[0]) <= self.ground_half[0]
        ) & (np.abs(centers[:, 1] - self.ground_center[1]) <= self.ground_half[1])
        gap_g = centers[:, 2] - self.bead_r - self.ground_z
        sel = np.flatnonzero(inside & (gap_g < thresh))
        if sel.size:
            normal = np.zeros((sel.size, 3))
            normal[:, 2] = 1.0
            f = self._contact_force(normal, bvel[sel], -gap_g[sel], self.mu_ground)
            bi = self.bead_body[sel]
            _scatter(ext, bi, f, n)
            _scatter(torque, bi, np.cross(arms[sel] - self.bead_r * normal, f), n)
            if report:
                gaps.append(gap_g[sel])

        # --- vertical pole array (capsule about each pole axis) -------------
        for p in range(self.pole_xy.shape[0]):
            dxy = centers[:, 0:2] - self.pole_xy[p]
            zc = np.clip(centers[:, 2], self.pole_z0[p], self.pole_z1[p])
            dz = centers[:, 2] - zc
            dist = np.sqrt(dxy[:, 0] ** 2 + dxy[:, 1] ** 2 + dz * dz)
            gap_p = dist - self.pole_r[p] - self.bead_r
            sel = np.flatnonzero(gap_p < thresh)
            if not sel.size:
                continue
            d_sel = np.maximum(dist[sel], 1e-9)
            normal = np.empty((sel.size, 3))
            normal[:, 0] = dxy[sel, 0] / d_sel
            normal[:, 1] = dxy[sel, 1] / d_sel
            normal[:, 2] = dz[sel] / d_sel
            degenerate = dist[sel] < 1e-9
            if np.any(degenerate):
                normal[degenerate] = np.array([1.0, 0.0, 0.0])
            f = self._contact_force(normal, bvel[sel], -gap_p[sel], self.mu_pole)
            bi = self.bead_body[sel]
            _scatter(ext, bi, f, n)
            _scatter(torque, bi, np.cross(arms[sel] - self.bead_r * normal, f), n)
            if report:
                gaps.append(gap_p[sel])

        # --- ring/ring nonconvex contacts ----------------------------------
        pairs = self._tree_pairs
        if pairs.shape[0]:
            ia, ib = pairs[:, 0], pairs[:, 1]
            dvec = centers[ia] - centers[ib]
            dist = np.linalg.norm(dvec, axis=1)
            gap_r = dist - 2.0 * self.bead_r
            sel = np.flatnonzero(gap_r < thresh)
            if sel.size:
                sa, sb = ia[sel], ib[sel]
                d_sel = np.maximum(dist[sel], 1e-9)
                normal = dvec[sel] / d_sel[:, None]
                relvel = bvel[sa] - bvel[sb]
                f = self._contact_force(normal, relvel, -gap_r[sel], self.mu_ring)
                bodies_a = self.bead_body[sa]
                bodies_b = self.bead_body[sb]
                _scatter(fint, bodies_a, f, n)
                _scatter(fint, bodies_b, -f, n)
                _scatter(torque, bodies_a, np.cross(arms[sa] - self.bead_r * normal, f), n)
                _scatter(torque, bodies_b, np.cross(arms[sb] + self.bead_r * normal, -f), n)
                if report:
                    gaps.append(gap_r[sel])
                    pair_forces.append(f)

        if not report:
            return ext, fint, torque, None, None
        gap_all = (
            np.concatenate(gaps) if gaps else np.array([float(np.min(gap_g))])
        )
        pair_all = (
            np.concatenate(pair_forces) if pair_forces else np.zeros((1, 3))
        )
        return ext, fint, torque, gap_all, pair_all

    # ------------------------------------------------------- broad phase / step
    def _refresh_pairs(self, centers: np.ndarray) -> None:
        tree = cKDTree(centers)
        pairs = tree.query_pairs(self.pair_radius, output_type="ndarray")
        if pairs.size:
            pairs = np.asarray(pairs, dtype=np.int64)
            keep = self.bead_body[pairs[:, 0]] != self.bead_body[pairs[:, 1]]
            pairs = np.ascontiguousarray(pairs[keep])
        else:
            pairs = np.empty((0, 2), dtype=np.int64)
        self._tree_pairs = pairs
        self._pairs_age = 0

    def _step(self, report: bool):
        """One fixed semi-implicit Euler step; returns report data if asked."""
        rot, arms, centers, bvel = self._bead_state()
        if self._pairs_age >= self.pair_every:
            self._refresh_pairs(centers)
        else:
            self._pairs_age += 1
        ext, fint, torque, gaps, pair_forces = self._forces(rot, arms, centers, bvel, report)

        dt = self.dt
        force = ext + fint
        self.vel = self.vel + dt * (force / self.mass[:, None])
        speed = np.linalg.norm(self.vel, axis=1)
        over = speed > self.v_max
        if np.any(over):
            self.vel[over] *= (self.v_max / speed[over])[:, None]

        # Body-frame Euler equation with the diagonal torus inertia tensor.
        tau_b = np.einsum("nji,nj->ni", rot, torque)
        w_b = np.einsum("nji,nj->ni", rot, self.omega)
        gyro = np.cross(w_b, self.inertia * w_b)
        alpha_b = (tau_b - gyro) * self.inv_inertia
        self.omega = self.omega + dt * np.einsum("nij,nj->ni", rot, alpha_b)
        spin = np.linalg.norm(self.omega, axis=1)
        over = spin > self.w_max
        if np.any(over):
            self.omega[over] *= (self.w_max / spin[over])[:, None]

        self.pos = self.pos + dt * self.vel
        v = self.quat[:, 0:3]
        w = self.quat[:, 3]
        dq = np.empty_like(self.quat)
        dq[:, 0:3] = 0.5 * (w[:, None] * self.omega + np.cross(self.omega, v))
        dq[:, 3] = -0.5 * np.einsum("ij,ij->i", self.omega, v)
        self.quat = self.quat + dt * dq
        self._normalize_quat()

        self.pos = np.nan_to_num(self.pos, nan=0.0, posinf=0.0, neginf=0.0)
        self.vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
        self.omega = np.nan_to_num(self.omega, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.pos[:, 2], self.ground_z - 5.0, 60.0, out=self.pos[:, 2])
        self.step_index += 1
        self._steps_done += 1
        return ext, fint, gaps, pair_forces

    # ------------------------------------------------------------- public API
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        started = _time.perf_counter()
        while self.step_index * self.dt < target - 0.5 * self.dt:
            self._step(False)
        self.time = target
        # Report contacts and external loads at the emitted state itself.
        rot, arms, centers, bvel = self._bead_state()
        if self._pairs_age >= self.pair_every or self._tree_pairs.shape[0] == 0:
            self._refresh_pairs(centers)
        ext, _fint, _tau, gaps, pair_forces = self._forces(rot, arms, centers, bvel, True)
        self._sim_seconds += _time.perf_counter() - started
        self._frames_emitted += 1
        if gaps.size:
            self._max_penetration = max(self._max_penetration, float(-np.min(gaps)))
        return self._make_frame(target, ext, gaps, pair_forces)

    def _make_frame(self, t, ext, gaps, pair_forces) -> StateFrame:
        ext = np.nan_to_num(np.asarray(ext, dtype=float).reshape(self.n, 3))
        gaps = np.nan_to_num(np.asarray(gaps, dtype=float).reshape(-1))
        pair_forces = np.nan_to_num(np.asarray(pair_forces, dtype=float).reshape(-1, 3))
        if gaps.size == 0:
            gaps = np.zeros(1)
        if pair_forces.shape[0] == 0:
            pair_forces = np.zeros((1, 3))
        self._last_contacts = int(gaps.size)
        return StateFrame(
            time=float(t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.quat.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "external_force": ext,
                "interaction_force": pair_forces,
                "contact_signed_distance": gaps,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )

    def diagnostics(self) -> dict[str, object]:
        w_b = np.einsum("nji,nj->ni", _quat_to_matrix(self.quat), self.omega)
        ke = float(
            0.5 * np.sum(self.mass * np.sum(self.vel ** 2, axis=1))
            + 0.5 * np.sum(self.inertia * w_b ** 2)
        )
        top = float(np.max(self.pos[:, 2]) + self.bead_r) if self.n else 0.0
        return {
            "method": "penalty-contact rigid bodies, bead-necklace nonconvex rings",
            "bodies": int(self.n),
            "beads_per_ring": int(self.k_beads),
            "ring_radius": float(self.ring_R),
            "bead_radius": float(self.bead_r),
            "dt": float(self.dt),
            "substeps_per_sample": int(self.substeps),
            "steps": int(self._steps_done),
            "frames": int(self._frames_emitted),
            "runtime_seconds": float(self._sim_seconds),
            "runtime_per_frame": float(
                self._sim_seconds / max(self._frames_emitted, 1)
            ),
            "max_penetration": float(self._max_penetration),
            "contact_length_scale": float(self.contact_length_scale),
            "final_kinetic_energy": ke,
            "settled_top_height": top,
            "last_contact_samples": int(getattr(self, "_last_contacts", 0)),
        }
