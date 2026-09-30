"""Self-contained rigid-body solver for the spolling coin decay scene.

Physical model
--------------
One rigid cylinder (the coin) over a frozen static box floor, integrated with a
semi-implicit (symplectic) Euler scheme in world coordinates:

    translation   m dv/dt = m g + F_air + sum_i f_i
    rotation      dL/dt   = sum_i (r_i x f_i) + tau_air
                  omega   = R I_body^-1 R^T L
    orientation   q       = exp(omega dt / 2) * q     (exact exponential map)

The only cylinder features that can touch a plane are its two rim circles, so
contact is sampled at ``n_theta`` points on each rim.  Every penetrating sample
carries a compliant unilateral normal force

    f_n = K delta - C ramp(delta) v_n ,   f_n >= 0

with the damping constant derived from the frozen restitution
(C = 2 zeta sqrt(K m), zeta from log e), and a regularized Coulomb friction
force

    f_t = -mu f_n v_t / max(|v_t|, v_eps) .

The aggregate normal stiffness/damping is held fixed and split across the
currently active samples, which keeps the explicit step stable whether the coin
touches at a single rim point or on a whole face, and keeps penetration far
below the frozen contact length scale.

That single law reproduces impact, slipping and rolling.  The coin is released
tilted with the classic rolling-disk (Euler disk) precession
omega = Omega (z_hat - cos(alpha) a_hat) plus an extra axial spin, i.e. it both
spins and rolls ("spolls").  Slip friction and weak aerodynamic drag drain
energy, so the tilt angle shrinks while the steady-precession relation
I_t Omega^2 sin(alpha) = m g r drives the contact point around ever faster and
tighter until the coin settles flat at rest.

Reported forces are the exact averages of the forces the integrator applied
over the following sample interval, so the exported external and interaction
forces reproduce the sampled momentum balance.  ``interaction_force`` rows and
``contact_signed_distance`` entries share one fixed sample ordering.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n <= 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
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


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    av, aw = a[:3], float(a[3])
    bv, bw = b[:3], float(b[3])
    v = aw * bv + bw * av + np.cross(av, bv)
    w = aw * bw - float(np.dot(av, bv))
    return np.array([v[0], v[1], v[2], w])


def _quat_from_omega(omega: np.ndarray, dt: float) -> np.ndarray:
    norm = float(np.linalg.norm(omega))
    theta = norm * dt
    if theta < 1e-14:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = omega / norm
    s = np.sin(0.5 * theta)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(0.5 * theta)])


class BenchmarkSolver:
    """Explicit rigid-body integrator with compliant frictional plane contact."""

    dt = 2.0e-5              # integration step [s]
    n_theta = 48             # contact samples per rim circle
    k_total = 2.0e6          # aggregate normal stiffness [N/m]
    damp_ramp = 2.0e-4       # penetration over which contact damping ramps in [m]
    v_eps = 0.02             # friction regularization velocity [m/s]
    c_linear_drag = 0.04     # translational air drag [N s/m]
    c_angular_drag = 0.004   # rotational air drag [N m s/rad]

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})

        frame0 = scene.initial_state()
        self.sample_dt = float(config.get("sample_dt", 0.02))
        self.duration = float(config.get("duration", 8.0))
        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        contact = dict(config.get("contact", {}) or {})
        self.mu = float(contact.get("friction", 0.55))
        self.restitution = float(np.clip(float(contact.get("restitution", 0.12)), 1e-4, 0.999))
        self.length_scale = float(contact.get("length_scale", 0.02))

        # ---- frozen geometry -------------------------------------------------
        radius, height, floor_top = 0.48, 0.08, 0.0
        floor_hx, floor_hy = 3.2, 2.2
        geometry = config.get("scene_geometry") or {}
        for item in geometry.get("objects", []) or []:
            geo = dict(item.get("geometry", {}) or {})
            transform = dict(item.get("transform", {}) or {})
            if item.get("id") == "coin" and geo.get("type") == "cylinder":
                radius = float(geo.get("radius", radius))
                height = float(geo.get("height", height))
            elif item.get("id") == "floor" and geo.get("type") == "box":
                he = np.asarray(geo.get("half_extents", [floor_hx, floor_hy, 0.05]), float)
                tr = np.asarray(transform.get("translation", [0.0, 0.0, -0.05]), float)
                floor_hx, floor_hy = float(he[0]), float(he[1])
                floor_top = float(tr[2] + he[2])
        self.radius = radius
        self.height = height
        self.half_height = 0.5 * height
        self.floor_top = floor_top
        self.floor_hx = floor_hx
        self.floor_hy = floor_hy

        # ---- rigid inertia (solid cylinder, symmetry axis = body z) ----------
        self.mass = float(np.asarray(frame0.mass, float)[0])
        i_axial = 0.5 * self.mass * radius * radius
        i_trans = self.mass * (3.0 * radius * radius + height * height) / 12.0
        self.i_body_inv_diag = np.array([1.0 / i_trans, 1.0 / i_trans, 1.0 / i_axial])
        self.i_trans = i_trans

        # ---- fixed contact samples on both rim circles (body frame) ---------
        phi = 2.0 * np.pi * np.arange(self.n_theta) / self.n_theta
        ring = np.stack([radius * np.cos(phi), radius * np.sin(phi)], axis=1)
        lower = np.concatenate([ring, np.full((self.n_theta, 1), -self.half_height)], axis=1)
        upper = np.concatenate([ring, np.full((self.n_theta, 1), +self.half_height)], axis=1)
        self.samples = np.concatenate([lower, upper], axis=0)
        self.n_samples = int(self.samples.shape[0])

        # contact damping from the frozen restitution
        log_e = float(np.log(self.restitution))
        zeta = -log_e / np.sqrt(np.pi * np.pi + log_e * log_e)
        self.c_total = 2.0 * zeta * np.sqrt(self.k_total * self.mass)

        # ---- initial state ---------------------------------------------------
        self.x = np.asarray(frame0.position, float)[0].copy()
        self.q = _quat_normalize(np.asarray(frame0.orientation, float)[0].copy())
        self.v = np.zeros(3)

        rot = _quat_to_matrix(self.q)
        axis = rot @ np.array([0.0, 0.0, 1.0])
        up = np.array([0.0, 0.0, 1.0])
        cos_a = float(np.clip(axis[2], -1.0, 1.0))
        sin_a = float(np.sqrt(max(1.0 - cos_a * cos_a, 1.0e-4)))
        g_mag = abs(float(self.gravity[2])) if abs(float(self.gravity[2])) > 1.0e-9 else 9.81
        # steady rolling-disk precession rate: I_t Omega^2 sin(alpha) = m g r
        omega_prec = 1.15 * np.sqrt(self.mass * g_mag * radius / (i_trans * sin_a))
        spin_axial = 3.0  # extra axial spin -> contact slips (spinning + rolling)
        omega = omega_prec * (up - cos_a * axis) + spin_axial * axis
        i_world = rot @ np.diag(1.0 / self.i_body_inv_diag) @ rot.T
        self.L = i_world @ omega
        self.omega = omega.copy()

        self.window = max(int(round(self.sample_dt / self.dt)), 1)
        self.step_index = 0
        self._max_penetration = 0.0
        self._min_gap = float("inf")
        self._max_speed = 0.0

    # -------------------------------------------------------------- mechanics
    def _gaps(self, offsets: np.ndarray) -> np.ndarray:
        return offsets[:, 2] + self.x[2] - self.floor_top

    def _contact_forces(self, offsets: np.ndarray, gap: np.ndarray):
        """Per-sample world contact force on the coin and the resulting torque."""
        forces = np.zeros((self.n_samples, 3))
        px = offsets[:, 0] + self.x[0]
        py = offsets[:, 1] + self.x[1]
        active = (gap < 0.0) & (np.abs(px) <= self.floor_hx) & (np.abs(py) <= self.floor_hy)
        n_active = int(np.count_nonzero(active))
        if n_active == 0:
            return forces, np.zeros(3)

        idx = np.flatnonzero(active)
        delta = -gap[idx]
        off = offsets[idx]
        vp = self.v + np.cross(np.broadcast_to(self.omega, off.shape), off)
        vn = vp[:, 2]

        stiffness = self.k_total / n_active
        damping = self.c_total / n_active
        ramp = np.minimum(1.0, delta / self.damp_ramp)
        fn = np.maximum(stiffness * delta - damping * ramp * vn, 0.0)

        vtx, vty = vp[:, 0], vp[:, 1]
        speed = np.sqrt(vtx * vtx + vty * vty)
        scale = -self.mu * fn / np.maximum(speed, self.v_eps)
        f = np.empty((idx.size, 3))
        f[:, 0] = scale * vtx
        f[:, 1] = scale * vty
        f[:, 2] = fn
        forces[idx] = f
        return forces, np.cross(off, f).sum(axis=0)

    def _step(self, accum_external, accum_contact):
        rot = _quat_to_matrix(self.q)
        self.omega = rot @ (self.i_body_inv_diag * (rot.T @ self.L))

        offsets = self.samples @ rot.T
        gap = self._gaps(offsets)
        forces, torque = self._contact_forces(offsets, gap)

        external = self.mass * self.gravity - self.c_linear_drag * self.v
        torque = torque - self.c_angular_drag * self.omega

        if accum_external is not None:
            accum_external += external
            accum_contact += forces

        self.v = self.v + ((external + forces.sum(axis=0)) / self.mass) * self.dt
        self.L = self.L + torque * self.dt
        self.omega = rot @ (self.i_body_inv_diag * (rot.T @ self.L))
        self.x = self.x + self.v * self.dt
        self.q = _quat_normalize(_quat_mul(_quat_from_omega(self.omega, self.dt), self.q))
        self.step_index += 1

        min_gap = float(gap.min())
        if min_gap < self._min_gap:
            self._min_gap = min_gap
        if -min_gap > self._max_penetration:
            self._max_penetration = -min_gap
        speed = float(np.linalg.norm(self.v))
        if speed > self._max_speed:
            self._max_speed = speed

    # -------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        n_steps = int(round((target - self.step_index * self.dt) / self.dt))
        for _ in range(max(n_steps, 0)):
            self._step(None, None)

        # state exactly at the requested sample time
        rot = _quat_to_matrix(self.q)
        self.omega = rot @ (self.i_body_inv_diag * (rot.T @ self.L))
        position = self.x.reshape(1, 3).copy()
        velocity = self.v.reshape(1, 3).copy()
        orientation = self.q.reshape(1, 4).copy()
        angular_velocity = self.omega.reshape(1, 3).copy()
        signed_distance = self._gaps(self.samples @ rot.T).copy()

        # forces averaged over the following sample interval; the same steps
        # also advance the simulation to the next sample time.
        accum_external = np.zeros(3)
        accum_contact = np.zeros((self.n_samples, 3))
        for _ in range(self.window):
            self._step(accum_external, accum_contact)

        frame = StateFrame(
            time=target,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=np.array([self.mass], dtype=float),
            body_id=np.array([0], dtype=np.int32),
            attributes={"release_time": np.zeros(1)},
            fields={
                "external_force": (accum_external / self.window).reshape(1, 3),
                "interaction_force": accum_contact / self.window,
                "contact_signed_distance": signed_distance,
            },
            scalars={"contact_length_scale": self.length_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        rot = _quat_to_matrix(self.q)
        axis = rot @ np.array([0.0, 0.0, 1.0])
        tilt = float(np.degrees(np.arccos(float(np.clip(abs(axis[2]), -1.0, 1.0)))))
        return {
            "method": "semi-implicit rigid body + compliant frictional rim contact",
            "dt": float(self.dt),
            "steps": int(self.step_index),
            "contact_samples": int(self.n_samples),
            "aggregate_normal_stiffness": float(self.k_total),
            "final_tilt_of_disk_from_horizontal_deg": tilt,
            "max_penetration_m": float(self._max_penetration),
            "max_penetration_over_length_scale": float(self._max_penetration / self.length_scale),
            "final_linear_speed": float(np.linalg.norm(self.v)),
            "final_angular_speed": float(np.linalg.norm(self.omega)),
        }
