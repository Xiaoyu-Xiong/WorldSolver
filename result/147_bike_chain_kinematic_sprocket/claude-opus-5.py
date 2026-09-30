"""Bicycle chain on sprockets.

Planar (x-z) constrained rigid-body solver written from scratch on NumPy.

Model
-----
* 20 rigid chain links, each a planar rigid body (x, z, theta) with a pin at
  each end (local +-a along its long axis).  Adjacent links share a pin, the
  loop is closed (link 19 -> link 0).  The pin joints are enforced as hard
  bilateral constraints with sequential impulses (Gauss-Seidel over two
  disjoint red/black joint batches) plus a position projection pass, so the
  chain neither stretches nor separates.
* 2 sprockets, each a single rotational degree of freedom about a fixed axle
  (the frozen axle_left / axle_right boundaries), inertia of a disc.  The
  toothed boundary is the frozen sprocket profile: 14 teeth, root radius
  0.54 m, tip radius 0.68 m, tips centred at 9.643 deg + k * 25.714 deg,
  represented as an implicit surface phi(r, th) = r - R_profile(th) so that
  flank normals (and therefore tangential drive) come out of the geometry.
* Chain/sprocket interaction: the shared pin of each joint carries a roller.
  Roller-vs-profile contact uses a compliant normal law (stiffness + damping
  tuned to the frozen restitution) and regularised Coulomb friction with the
  frozen friction coefficient.  Reaction torques are applied to the sprockets,
  so the follower is driven purely by contact.
* The right sprocket receives a prescribed drive torque (smooth ramp) minus
  bearing viscosity; the left sprocket is a free follower with bearing
  viscosity only.

Time stepping is a fixed-step semi-implicit (symplectic) Euler scheme with
constraint impulses, sub-stepped an integer number of times per output sample,
which makes the trajectory deterministic and finite.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------- geometry ---
N_TEETH = 14
R_TIP = 0.68
R_ROOT = 0.54
TOOTH_DELTA = 2.0 * math.pi / N_TEETH          # angular pitch of the teeth
TOOTH_PHASE = math.radians(9.642857)           # first tooth centre (mesh)
TIP_HALF = TOOTH_DELTA / 8.0                   # flat tip half width
FLANK_SPAN = TOOTH_DELTA / 2.0 - TIP_HALF      # tip -> valley angular span

LINK_LEN = 0.51        # chain_link.obj extent along its long axis
LINK_HEIGHT = 0.31     # chain_link.obj extent across the link
ROLLER_R = 0.05        # pin / roller radius (tube radius of the link mesh)
RIDE_CLEARANCE = 5.0e-4  # taut-loop clearance over the tooth tips
FLOOR_TOP = 0.0        # frozen floor top surface


def _tooth_profile(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sprocket radial profile R(theta) and dR/dtheta in the body frame."""
    psi = theta - TOOTH_PHASE
    phi = np.mod(psi + 0.5 * TOOTH_DELTA, TOOTH_DELTA) - 0.5 * TOOTH_DELTA
    aphi = np.abs(phi)
    u = np.clip((aphi - TIP_HALF) / FLANK_SPAN, 0.0, 1.0)
    shape = 0.5 * (1.0 + np.cos(math.pi * u))
    radius = R_ROOT + (R_TIP - R_ROOT) * shape
    dshape = -0.5 * math.pi / FLANK_SPAN * np.sin(math.pi * u)
    dshape = np.where(aphi > TIP_HALF, dshape, 0.0) * np.sign(phi)
    return radius, (R_TIP - R_ROOT) * dshape


def _perp(v: np.ndarray) -> np.ndarray:
    """CCW perpendicular in the (x, z) plane."""
    return np.stack([-v[:, 1], v[:, 0]], axis=1)


def _cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


class BenchmarkSolver:
    # ------------------------------------------------------------- set-up ---
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        frame0 = scene.initial_state()

        self.cfg = cfg
        self.seed = int(seed)
        self.gravity = float(np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]))[2])
        contact = dict(cfg.get("contact", {}) or {})
        self.mu = float(contact.get("friction", 0.55))
        self.restitution = float(contact.get("restitution", 0.12))
        self.length_scale = float(contact.get("length_scale", 0.02))

        pos0 = np.asarray(frame0.position, dtype=float)
        self.mass_all = np.asarray(frame0.mass, dtype=float).copy()
        self.n_bodies = pos0.shape[0]
        self.body_id = np.arange(self.n_bodies, dtype=np.int32)

        bodies = cfg.get("bodies", [])
        ids = [str(b.get("id", "")) for b in bodies]
        try:
            i_left = ids.index("sprocket_left")
            i_right = ids.index("sprocket_right")
        except ValueError:
            i_left, i_right = self.n_bodies - 2, self.n_bodies - 1
        self.i_sp = (i_left, i_right)
        self.link_idx = np.asarray(
            [i for i in range(self.n_bodies) if i not in self.i_sp], dtype=int
        )
        self.n_links = int(self.link_idx.size)

        # sprocket centres in the (x, z) plane (fixed axles)
        self.sc = np.stack(
            [pos0[i_left][[0, 2]], pos0[i_right][[0, 2]]], axis=0
        ).astype(float)

        # ------------------------------------------------------ properties ---
        self.m = float(np.mean(self.mass_all[self.link_idx]))
        self.m_link = self.mass_all[self.link_idx].copy()
        self.inv_m = 1.0 / self.m_link
        self.I_link = self.m_link * (LINK_LEN ** 2 + LINK_HEIGHT ** 2) / 12.0
        self.inv_I = 1.0 / self.I_link
        m_sp = self.mass_all[[i_left, i_right]].astype(float)
        r_sp = 0.5 * (R_TIP + R_ROOT)
        self.I_sp = 0.5 * m_sp * r_sp ** 2

        # Closed-loop kinematics fix the link pitch: the taut loop wraps both
        # sprockets on the tooth-tip envelope, so its length is the racetrack
        # perimeter and one link spans perimeter / n_links.
        self.center_dist = float(np.linalg.norm(self.sc[1] - self.sc[0]))
        self.r_roll = ROLLER_R
        self.r_wrap = R_TIP + self.r_roll + RIDE_CLEARANCE
        self.pitch = (2.0 * self.center_dist + 2.0 * math.pi * self.r_wrap) / self.n_links
        self.a = 0.5 * self.pitch

        # --------------------------------------------------- contact model ---
        self.k_c = 1.5e4
        e = min(max(self.restitution, 1e-3), 0.95)
        le = math.log(e)
        zeta = -le / math.sqrt(math.pi ** 2 + le ** 2)
        self.c_c = 2.0 * zeta * math.sqrt(self.k_c * self.m)
        self.c_t = 300.0
        self.fn_max = 800.0
        self.drag = 0.02
        self.ang_drag = 1.0e-3
        self.c_j = 2.0e-3

        # ------------------------------------------------------------ drive ---
        self.tau_drive = 5.0
        self.b_drive = 2.0
        self.b_follow = 0.06
        self.ramp_time = 0.6

        # --------------------------------------------------- discretisation ---
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.dt = 2.5e-4
        self.sub = max(1, int(round(self.sample_dt / self.dt)))
        self.dt = self.sample_dt / self.sub
        self.n_vel_iter = 8
        self.n_pos_iter = 3
        self.beta = 0.25
        self.relax = 0.7

        # ------------------------------------------------------ joint index ---
        self.j_i1 = np.arange(self.n_links, dtype=int)
        self.j_i2 = (self.j_i1 + 1) % self.n_links
        self.batches = [
            np.arange(0, self.n_links, 2, dtype=int),
            np.arange(1, self.n_links, 2, dtype=int),
        ]

        # ------------------------------------------------------ seeding ------
        self._seed_chain()

        self.theta = np.zeros(2)
        self.omega = np.zeros(2)

        self.t = 0.0
        self.step_count = 0
        self.gaps = np.zeros((self.n_links, 2))
        self.cforce = np.zeros((self.n_links, 2, 2))
        self.joint_force = np.zeros((self.n_links, 2))
        self._max_pen = 0.0
        self._blowups = 0

        # measure contacts once so frame 0 carries physical fields
        self._contact_and_forces()

    # ------------------------------------------------------------ seeding ---
    def _seed_chain(self) -> None:
        n = self.n_links
        r_seed = self.r_wrap
        self.r_seed = r_seed
        length = n * self.pitch
        s_pin = (np.arange(n + 1, dtype=float) - 0.5) * (length / n)
        pins = self._path_point(s_pin, r_seed, length)
        pins[-1] = pins[0]

        d = pins[1:] - pins[:-1]
        self.pos = 0.5 * (pins[:-1] + pins[1:])
        self.ang = np.arctan2(d[:, 1], d[:, 0])
        self.vel = np.zeros((n, 2))
        self.omg = np.zeros(n)

        # make the closed loop exactly consistent with the rigid link length
        for _ in range(300):
            self._project_positions(0.8)

    def _path_point(self, s: np.ndarray, radius: float, length: float) -> np.ndarray:
        """Race-track path: right arc, top run, left arc, bottom run."""
        s = np.mod(np.asarray(s, dtype=float), length)
        c_left, c_right = self.sc[0], self.sc[1]
        span = self.center_dist
        q1 = 0.5 * math.pi * radius
        q2 = q1 + span
        q3 = q2 + math.pi * radius
        q4 = q3 + span
        out = np.zeros((s.size, 2))

        m = s < q1
        ang = s[m] / radius
        out[m] = c_right + radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)

        m = (s >= q1) & (s < q2)
        u = s[m] - q1
        out[m] = np.stack([c_right[0] - u, np.full(u.shape, c_right[1] + radius)], axis=1)

        m = (s >= q2) & (s < q3)
        ang = 0.5 * math.pi + (s[m] - q2) / radius
        out[m] = c_left + radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)

        m = (s >= q3) & (s < q4)
        u = s[m] - q3
        out[m] = np.stack([c_left[0] + u, np.full(u.shape, c_left[1] - radius)], axis=1)

        m = s >= q4
        ang = 1.5 * math.pi + (s[m] - q4) / radius
        out[m] = c_right + radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        return out

    # ------------------------------------------------------- joint solver ---
    def _joint_geometry(self, batch):
        i1 = self.j_i1[batch]
        i2 = self.j_i2[batch]
        c1, s1 = np.cos(self.ang[i1]), np.sin(self.ang[i1])
        c2, s2 = np.cos(self.ang[i2]), np.sin(self.ang[i2])
        r1 = self.a * np.stack([c1, s1], axis=1)
        r2 = -self.a * np.stack([c2, s2], axis=1)
        err = (self.pos[i2] + r2) - (self.pos[i1] + r1)
        return i1, i2, r1, r2, err

    def _joint_matrix(self, i1, i2, r1, r2):
        p1 = _perp(r1)
        p2 = _perp(r2)
        im1 = self.inv_m[i1]
        im2 = self.inv_m[i2]
        ii1 = self.inv_I[i1]
        ii2 = self.inv_I[i2]
        k00 = im1 + im2 + ii1 * p1[:, 0] ** 2 + ii2 * p2[:, 0] ** 2
        k11 = im1 + im2 + ii1 * p1[:, 1] ** 2 + ii2 * p2[:, 1] ** 2
        k01 = ii1 * p1[:, 0] * p1[:, 1] + ii2 * p2[:, 0] * p2[:, 1]
        det = k00 * k11 - k01 * k01
        det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        return k00, k11, k01, det

    @staticmethod
    def _solve2(k00, k11, k01, det, b):
        lx = (k11 * b[:, 0] - k01 * b[:, 1]) / det
        lz = (k00 * b[:, 1] - k01 * b[:, 0]) / det
        return np.stack([lx, lz], axis=1)

    def _solve_velocity_constraints(self, dt: float) -> None:
        self.joint_force[:] = 0.0
        for _ in range(self.n_vel_iter):
            for batch in self.batches:
                i1, i2, r1, r2, err = self._joint_geometry(batch)
                k00, k11, k01, det = self._joint_matrix(i1, i2, r1, r2)
                v1 = self.vel[i1] + self.omg[i1, None] * _perp(r1)
                v2 = self.vel[i2] + self.omg[i2, None] * _perp(r2)
                rhs = -(v2 - v1) - (self.beta / dt) * err
                lam = self._solve2(k00, k11, k01, det, rhs)
                self.vel[i2] += lam * self.inv_m[i2, None]
                self.omg[i2] += _cross2(r2, lam) * self.inv_I[i2]
                self.vel[i1] -= lam * self.inv_m[i1, None]
                self.omg[i1] -= _cross2(r1, lam) * self.inv_I[i1]
                self.joint_force[batch] += lam / dt

    def _project_positions(self, relax: float) -> None:
        for batch in self.batches:
            i1, i2, r1, r2, err = self._joint_geometry(batch)
            k00, k11, k01, det = self._joint_matrix(i1, i2, r1, r2)
            lam = self._solve2(k00, k11, k01, det, -relax * err)
            self.pos[i2] += lam * self.inv_m[i2, None]
            self.ang[i2] += _cross2(r2, lam) * self.inv_I[i2]
            self.pos[i1] -= lam * self.inv_m[i1, None]
            self.ang[i1] -= _cross2(r1, lam) * self.inv_I[i1]

    # ------------------------------------------------------------ forces ----
    def _contact_and_forces(self):
        i1, i2 = self.j_i1, self.j_i2
        c1, s1 = np.cos(self.ang), np.sin(self.ang)
        r1 = self.a * np.stack([c1[i1], s1[i1]], axis=1)
        r2 = -self.a * np.stack([c1[i2], s1[i2]], axis=1)
        p1 = self.pos[i1] + r1
        p2 = self.pos[i2] + r2
        x_roll = 0.5 * (p1 + p2)
        v1 = self.vel[i1] + self.omg[i1, None] * _perp(r1)
        v2 = self.vel[i2] + self.omg[i2, None] * _perp(r2)
        v_roll = 0.5 * (v1 + v2)

        force = np.zeros((self.n_links, 2))
        torque = np.zeros(self.n_links)
        tau_sp = np.zeros(2)

        for k in range(2):
            d = x_roll - self.sc[k]
            r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            th = np.arctan2(d[:, 1], d[:, 0]) - self.theta[k]
            r_prof, dr_prof = _tooth_profile(th)
            gap = r - r_prof - self.r_roll
            self.gaps[:, k] = gap

            rhat = d / r[:, None]
            that = _perp(rhat)
            nvec = rhat - (dr_prof / r)[:, None] * that
            nvec /= np.maximum(np.linalg.norm(nvec, axis=1), 1e-12)[:, None]
            tvec = _perp(nvec)

            v_surf = self.omega[k] * _perp(d)
            v_rel = v_roll - v_surf
            vn = np.sum(v_rel * nvec, axis=1)
            vt = np.sum(v_rel * tvec, axis=1)

            active = gap < 0.0
            fn = np.where(active, -self.k_c * gap - self.c_c * vn, 0.0)
            fn = np.clip(fn, 0.0, self.fn_max)
            limit = self.mu * fn
            ft = -np.clip(self.c_t * vt, -limit, limit)
            f = fn[:, None] * nvec + ft[:, None] * tvec
            self.cforce[:, k, :] = f

            half = 0.5 * f
            force[i1] += half
            torque[i1] += _cross2(r1, half)
            force[i2] += half
            torque[i2] += _cross2(r2, half)
            tau_sp[k] -= np.sum(_cross2(d, f))

            pen = float(np.min(gap))
            if pen < self._max_pen:
                self._max_pen = pen

        # gravity and drag on the links
        force[:, 1] += self.m_link * self.gravity
        force -= self.drag * self.vel
        torque -= self.ang_drag * self.omg

        # articulation (joint) damping
        dw = self.omg[i2] - self.omg[i1]
        tj = -self.c_j * dw
        torque[i2] += tj
        torque[i1] -= tj

        # floor contact (frozen boundary at z = 0)
        z_bot = self.pos[:, 1] - 0.5 * LINK_HEIGHT
        pen = np.minimum(z_bot - FLOOR_TOP, 0.0)
        fz = np.where(pen < 0.0, -self.k_c * pen - self.c_c * self.vel[:, 1], 0.0)
        force[:, 1] += np.clip(fz, 0.0, self.fn_max)

        return force, torque, tau_sp

    def _drive_torque(self, t: float) -> np.ndarray:
        u = min(max(t / self.ramp_time, 0.0), 1.0)
        ramp = u * u * (3.0 - 2.0 * u)
        tau = np.zeros(2)
        tau[1] += self.tau_drive * ramp - self.b_drive * self.omega[1]
        tau[0] += -self.b_follow * self.omega[0]
        return tau

    # -------------------------------------------------------------- step ----
    def _step(self) -> None:
        dt = self.dt
        force, torque, tau_sp = self._contact_and_forces()
        tau_sp = tau_sp + self._drive_torque(self.t)

        self.vel += (force * self.inv_m[:, None]) * dt
        self.omg += (torque * self.inv_I) * dt
        self.omega += (tau_sp / self.I_sp) * dt

        self._solve_velocity_constraints(dt)

        np.clip(self.vel, -50.0, 50.0, out=self.vel)
        np.clip(self.omg, -200.0, 200.0, out=self.omg)
        np.clip(self.omega, -60.0, 60.0, out=self.omega)

        self.pos += self.vel * dt
        self.ang += self.omg * dt
        self.theta += self.omega * dt

        for _ in range(self.n_pos_iter):
            self._project_positions(self.relax)

        if not (np.isfinite(self.pos).all() and np.isfinite(self.vel).all()):
            self._blowups += 1
            self.pos = np.nan_to_num(self.pos, nan=0.0, posinf=0.0, neginf=0.0)
            self.vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
            self.ang = np.nan_to_num(self.ang, nan=0.0, posinf=0.0, neginf=0.0)
            self.omg = np.nan_to_num(self.omg, nan=0.0, posinf=0.0, neginf=0.0)

        self.step_count += 1
        self.t = self.step_count * dt

    # ------------------------------------------------------------ output ----
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.dt))
        while self.step_count < target:
            self._step()
        self._contact_and_forces()
        return self._make_frame(float(time))

    def _quat_y(self, angle):
        angle = np.asarray(angle, dtype=float)
        half = 0.5 * angle
        z = np.zeros_like(half)
        return np.stack([z, -np.sin(half), z, np.cos(half)], axis=-1)

    def _make_frame(self, time: float) -> StateFrame:
        n = self.n_bodies
        position = np.zeros((n, 3))
        velocity = np.zeros((n, 3))
        orientation = np.zeros((n, 4))
        omega3 = np.zeros((n, 3))

        li = self.link_idx
        position[li, 0] = self.pos[:, 0]
        position[li, 2] = self.pos[:, 1]
        velocity[li, 0] = self.vel[:, 0]
        velocity[li, 2] = self.vel[:, 1]
        orientation[li] = self._quat_y(self.ang)
        omega3[li, 1] = -self.omg

        for k, bi in enumerate(self.i_sp):
            position[bi, 0] = self.sc[k, 0]
            position[bi, 2] = self.sc[k, 1]
            orientation[bi] = self._quat_y(self.theta[k])
            omega3[bi, 1] = -self.omega[k]

        external = np.zeros((n, 3))
        external[:, 2] = self.mass_all * self.gravity

        n_contacts = self.n_links * 2
        inter = np.zeros((self.n_links + n_contacts, 3))
        inter[: self.n_links, 0] = self.joint_force[:, 0]
        inter[: self.n_links, 2] = self.joint_force[:, 1]
        cf = self.cforce.reshape(-1, 2)
        inter[self.n_links :, 0] = cf[:, 0]
        inter[self.n_links :, 2] = cf[:, 1]

        gaps = self.gaps.reshape(-1).copy()

        frame = StateFrame(
            time=max(float(time), 0.0),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=omega3,
            mass=self.mass_all.copy(),
            body_id=self.body_id.copy(),
            attributes={},
            fields={
                "external_force": external,
                "interaction_force": inter,
                "contact_signed_distance": gaps,
            },
            scalars={"contact_length_scale": float(self.length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": float(self.dt),
            "substeps_per_sample": int(self.sub),
            "steps": int(self.step_count),
            "pitch": float(self.pitch),
            "seed_wrap_radius": float(self.r_seed),
            "max_penetration": float(self._max_pen),
            "sprocket_omega": [float(self.omega[0]), float(self.omega[1])],
            "nonfinite_events": int(self._blowups),
        }
