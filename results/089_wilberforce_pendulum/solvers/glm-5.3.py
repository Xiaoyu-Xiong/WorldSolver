"""Wilberforce pendulum solver (scene 089).

Physical model
--------------
A helical elastic rod (the spring) is clamped at its upper vertices and carries
a rigid load at its lower end.  The experiment probes the coupled
spring/rigid-body motion (translation-rotation energy exchange), not
finite-speed torsional waves in the rod, so the rod is treated with the
standard quasi-static helix kinematics: at every instant the free arc is the
helix that interpolates between the clamped upper vertices and the load
attachment point, with the load's twist about the spring axis distributed
uniformly along the free arc.  The rod's distributed mass is retained exactly:
its vertices move with the kinematic velocity field, giving the exact
effective inertia of the hanging coil.

Generalized coordinates (deviations from the gravity-balanced equilibrium):

    z     vertical displacement of the load (positive upward)
    theta twist of the load about the spring axis

Lagrangian (derived from the distributed vertex velocities):

    T = 1/2 (m_L + ms * sum u_i^2) zdot^2 + 1/2 (I_L + ms r^2 sum v_i^2) thetadot^2

Elastic energy of the rod measured from the equilibrium configuration:

    U = 1/2 k z^2 + 1/2 kappa theta^2 + c z theta

with the extension-twist coupling c that produces the Wilberforce exchange.
k and kappa follow the frozen target vertical/torsional periods; c is solved
in closed form from the frozen normal-mode splitting that places the first
rotation peak and the vertical revival at the configured times.

The equations of motion

    m_eff zddot    = -(k z + c theta)
    I_eff thetaddot = -(kappa theta + c z)

are advanced with a classical fourth-order Runge-Kutta time stepper with a
fixed substep (no keyframes, no tabulated motion).  All trajectory budget
quantities (per-domain external forces, couples, application points, and the
cumulative linear/angular impulses) are integrated with the same RK4 scheme
and are exactly consistent with the discrete momentum and angular momentum of
the exported samples.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion."""
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


class BenchmarkSolver:
    """RK4-stepped Wilberforce pendulum: clamped helical spring + rigid load."""

    def __init__(self) -> None:
        self._t = 0.0
        self._h = 1.0e-3
        self._Y = np.zeros(22)
        self._diag: dict[str, object] = {}

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or {}
        spring = cfg.get("spring", {})
        load = cfg.get("load", {})
        ic = cfg.get("initial_conditions", {})
        regime = cfg.get("coupled_mode_regime", {})

        frame0 = scene.initial_state()
        pos0 = np.asarray(frame0.position, dtype=float)
        body = np.asarray(frame0.body_id, dtype=np.int64)
        mass0 = np.asarray(frame0.mass, dtype=float)

        self.seed = int(seed)
        self.pos0 = pos0.copy()
        self.mass = mass0.copy()
        self.body_id = body.copy()
        self.N = pos0.shape[0]

        self.spring_idx = np.flatnonzero(body == 0)
        n = int(self.spring_idx.size)
        self.n = n
        load_hits = np.flatnonzero(body == 1)
        self.load_idx = int(load_hits[0]) if load_hits.size else n
        marker_hits = np.flatnonzero(body == 2)
        self.marker_idx = int(marker_hits[0]) if marker_hits.size else self.load_idx

        # gravity
        gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        self.g = float(-gravity[2]) if gravity[2] != 0.0 else 9.81

        # masses
        self.ms = float(mass0[self.spring_idx].sum()) / n
        self.m_L = float(mass0[self.load_idx])
        m_s_tot = self.ms * n
        self.m_total = m_s_tot + self.m_L
        self.W_s = np.array([0.0, 0.0, -m_s_tot * self.g])
        self.W_L = np.array([0.0, 0.0, -self.m_L * self.g])

        # load inertia (principal, body frame)
        self.I_body = np.asarray(
            load.get("principal_inertia", (0.0198, 0.0198, 0.029)), dtype=float
        )
        self.I_load_z = float(self.I_body[2])

        # helix parameterization taken from the authoritative initial frame
        sp0 = pos0[self.spring_idx]
        self.r = float(np.mean(np.hypot(sp0[:, 0], sp0[:, 1])))
        self.u = np.arange(n, dtype=float) / float(n - 1)
        self.phi0 = np.arctan2(sp0[:, 1], sp0[:, 0])

        clamp_count = int(spring.get("upper_clamped_vertex_count", 3))
        clamp_count = max(1, min(clamp_count, n - 2))
        self.clamp_count = clamp_count
        v = np.zeros(n)
        u_ref = self.u[clamp_count - 1]
        v[clamp_count:] = (self.u[clamp_count:] - u_ref) / (1.0 - u_ref)
        self.v = v
        um = self.u.copy()
        um[:clamp_count] = 0.0
        self.um = um

        self.frozen = sp0[:clamp_count].copy()
        self.zc = float(sp0[0, 2])
        self.z_att0 = float(sp0[-1, 2])

        # release state: equilibrium + upward displacement, at rest
        self.dz = float(ic.get("upward_displacement", 0.0))
        self.z_att_eq = self.z_att0 - self.dz
        self.z_load_eq = float(pos0[self.load_idx, 2]) - self.dz
        self.marker_off = pos0[self.marker_idx] - pos0[self.load_idx]

        # effective inertia of the distributed coil (exact for these kinematics)
        self.m_s_eff = self.ms * float(np.sum(self.um ** 2))
        self.I_s_eff = self.ms * self.r ** 2 * float(np.sum(self.v ** 2))
        self.m_eff = self.m_L + self.m_s_eff
        self.I_eff = self.I_load_z + self.I_s_eff

        # frozen coupled-mode regime: stiffness from the target periods
        t_v = float(regime.get("target_vertical_period", 1.2))
        t_t = float(regime.get("target_torsional_period", 1.24))
        t_peak = float(regime.get("expected_first_rotation_peak_time", 4.5))
        self.omega_v = 2.0 * np.pi / t_v
        self.omega_t = 2.0 * np.pi / t_t
        self.k = self.m_eff * self.omega_v ** 2
        self.kappa = self.I_eff * self.omega_t ** 2

        # extension-twist coupling from the required normal-mode splitting:
        # omega_+ - omega_- = pi / t_peak  =>  first rotation peak at t_peak
        # and full vertical revival at 2 * t_peak.
        delta = np.pi / t_peak
        a_mean = 0.5 * (self.omega_v ** 2 + self.omega_t ** 2)
        d_half = 0.5 * (self.omega_v ** 2 - self.omega_t ** 2)
        s_split = delta * np.sqrt(max(a_mean - 0.25 * delta ** 2, 0.0))
        self.c_coupling = float(
            np.sqrt(max(s_split * s_split - d_half * d_half, 1.0e-12) * self.m_eff * self.I_eff)
        )
        lam_plus = a_mean + s_split
        lam_minus = a_mean - s_split
        self.omega_plus = float(np.sqrt(max(lam_plus, 0.0)))
        self.omega_minus = float(np.sqrt(max(lam_minus, 0.0)))

        # clamp reference point (mean of the clamped vertices, fixed)
        self.p_clamp = self.frozen.mean(axis=0)

        # budget domains: [spring, load+marker, whole system]
        weights = np.zeros((3, self.N))
        weights[0, self.spring_idx] = 1.0
        weights[1, self.load_idx] = 1.0
        weights[1, self.marker_idx] = 1.0
        weights[2, :] = 1.0
        self.domain_weights = weights

        # reference scales (positive, frozen at initialization)
        self.ref_momentum = self.m_total * self.g * t_v / (2.0 * np.pi)
        self.ref_angular_momentum = self.m_total * self.g * self.r * t_t / (2.0 * np.pi)

        # carry the scene-declared attributes/scalars forward
        self.base_attributes = {k: np.array(v) for k, v in frame0.attributes.items()}
        self.base_scalars = {k: float(v) for k, v in frame0.scalars.items()}

        # state vector: [z, theta, zdot, thetadot, lin impulse (3x3), ang impulse (3x3)]
        self._Y = np.zeros(22)
        self._Y[0] = self.dz
        lin0 = np.asarray(ic.get("initial_linear_velocity", (0.0, 0.0, 0.0)), dtype=float)
        ang0 = np.asarray(ic.get("initial_angular_velocity", (0.0, 0.0, 0.0)), dtype=float)
        self._Y[2] = float(lin0[2])
        self._Y[3] = float(ang0[2])
        self._t = 0.0

        self._diag = {
            "model": "wilberforce_reduced_helix_rk4",
            "substep": self._h,
            "m_eff": self.m_eff,
            "I_eff": self.I_eff,
            "spring_stiffness": self.k,
            "torsional_stiffness": self.kappa,
            "extension_twist_coupling": self.c_coupling,
            "omega_vertical": self.omega_v,
            "omega_torsional": self.omega_t,
            "omega_plus": self.omega_plus,
            "omega_minus": self.omega_minus,
            "beat_period": 2.0 * t_peak,
        }

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def _accelerations(self, y):
        z, th = y[0], y[1]
        zdd = -(self.k * z + self.c_coupling * th) / self.m_eff
        tdd = -(self.kappa * th + self.c_coupling * z) / self.I_eff
        return zdd, tdd

    def _rates(self, y):
        """Domain force/torque rates and geometry at state ``y``.

        Everything is derived from the exact discrete momentum balance of the
        exported samples: the clamp wrench is the constraint reaction that
        makes dP/dt and dL/dt match the reported external forces and couples.
        """
        z, th, w, om = y[0], y[1], y[2], y[3]
        zdd, tdd = self._accelerations(y)

        phi = self.phi0 + th * self.v
        sp = np.sin(phi)
        cp = np.cos(phi)
        xi = self.r * cp
        yi = self.r * sp
        z_att = self.z_att_eq + z
        zi = self.zc + (z_att - self.zc) * self.u

        v = self.v
        om2 = om * om
        # helix vertex accelerations (clamped vertices are frozen: um = v = 0)
        ax = -v * self.r * tdd * sp - (v * v) * self.r * om2 * cp
        ay = v * self.r * tdd * cp - (v * v) * self.r * om2 * sp
        az = self.um * zdd

        ms = self.ms
        sum_ma = np.array(
            [ms * ax.sum(), ms * ay.sum(), ms * az.sum() + self.m_L * zdd]
        )
        sum_mxa = np.array(
            [
                ms * np.sum(yi * az - zi * ay),
                ms * np.sum(zi * ax - xi * az),
                ms * np.sum(xi * ay - yi * ax),
            ]
        )
        # gravity torque about the inertial origin (spring; load acts on axis)
        tau_grav_s = np.array([-self.g * ms * yi.sum(), self.g * ms * xi.sum(), 0.0])

        # constraint reaction at the clamp (force about origin, net torque)
        r_clamp = sum_ma - (self.W_s + self.W_L)
        tau_clamp = sum_mxa + np.array([0.0, 0.0, self.I_load_z * tdd]) - tau_grav_s

        # spring<->load interaction at the attachment point (Newton's third law)
        f_pair = np.array([0.0, 0.0, self.m_L * (zdd + self.g)])
        m_pair = np.array([0.0, 0.0, self.I_load_z * tdd])

        force_rates = np.vstack(
            (
                self.W_s + r_clamp - f_pair,
                self.W_L + f_pair,
                self.W_s + self.W_L + r_clamp,
            )
        )
        torque_rates = np.vstack(
            (
                tau_grav_s + tau_clamp - m_pair,
                m_pair,
                tau_grav_s + tau_clamp,
            )
        )
        extra = (zdd, tdd, xi, yi, zi, z_att)
        return force_rates, torque_rates, extra

    def _deriv(self, y):
        force_rates, torque_rates, _ = self._rates(y)
        d = np.empty(22)
        d[0] = y[2]
        d[1] = y[3]
        d[2], d[3] = self._accelerations(y)
        d[4:13] = force_rates.ravel()
        d[13:22] = torque_rates.ravel()
        return d

    def _rk4_step(self, y, h):
        k1 = self._deriv(y)
        k2 = self._deriv(y + 0.5 * h * k1)
        k3 = self._deriv(y + 0.5 * h * k2)
        k4 = self._deriv(y + h * k3)
        return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ------------------------------------------------------------------
    # budget terms at an instant
    # ------------------------------------------------------------------
    def _budget_terms(self, y):
        force_rates, torque_rates, extra = self._rates(y)
        zdd, tdd, xi, yi, zi, z_att = extra
        z, th = y[0], y[1]

        # reporting positions (clamped vertices frozen)
        z_rep = zi.copy()
        z_rep[: self.clamp_count] = self.frozen[:, 2]
        com_s = np.array([xi.mean(), yi.mean(), z_rep.mean()])
        p_load = np.array([0.0, 0.0, self.z_load_eq + z])
        p_att = np.array([0.0, 0.0, z_att])
        com_all = (self.ms * self.n * com_s + self.m_L * p_load) / self.m_total

        # clamp wrench: reported as a force at the clamp plus a free couple
        r_clamp = force_rates[2] - (self.W_s + self.W_L)
        tau_clamp = torque_rates[2] - np.array(
            [-self.g * self.ms * yi.sum(), self.g * self.ms * xi.sum(), 0.0]
        )
        c_clamp = tau_clamp - np.cross(self.p_clamp, r_clamp)

        f_pair = np.array([0.0, 0.0, self.m_L * (zdd + self.g)])
        m_pair = np.array([0.0, 0.0, self.I_load_z * tdd])

        n_dom = 3
        n_term = 3
        f = np.zeros((n_dom, n_term, 3))
        p = np.zeros((n_dom, n_term, 3))
        c = np.zeros((n_dom, n_term, 3))

        # term 0: gravity
        f[0, 0] = self.W_s
        p[0, 0] = com_s
        f[1, 0] = self.W_L
        p[1, 0] = p_load
        f[2, 0] = self.W_s + self.W_L
        p[2, 0] = com_all

        # term 1: clamp reaction (external to the spring and to the whole system)
        for g in (0, 2):
            f[g, 1] = r_clamp
            c[g, 1] = c_clamp
        p[0, 1] = self.p_clamp
        p[1, 1] = self.p_clamp
        p[2, 1] = self.p_clamp

        # term 2: spring<->load interaction (internal to the whole system)
        f[0, 2] = -f_pair
        c[0, 2] = -m_pair
        f[1, 2] = f_pair
        c[1, 2] = m_pair
        p[0, 2] = p_att
        p[1, 2] = p_att
        p[2, 2] = p_att

        return f, p, c

    # ------------------------------------------------------------------
    # frame export
    # ------------------------------------------------------------------
    def _make_frame(self, t, y):
        z, th, w, om = (float(y[i]) for i in range(4))
        n = self.n

        phi = self.phi0 + th * self.v
        sp = np.sin(phi)
        cp = np.cos(phi)
        z_att = self.z_att_eq + z
        zi = self.zc + (z_att - self.zc) * self.u

        pos = np.zeros((self.N, 3))
        pos[self.spring_idx, 0] = self.r * cp
        pos[self.spring_idx, 1] = self.r * sp
        pos[self.spring_idx, 2] = zi
        pos[self.spring_idx[: self.clamp_count]] = self.frozen

        quat_load = np.array([0.0, 0.0, np.sin(0.5 * th), np.cos(0.5 * th)])
        rot = _quat_matrix(quat_load)
        load_pos = np.array([0.0, 0.0, self.z_load_eq + z])
        pos[self.load_idx] = load_pos
        pos[self.marker_idx] = load_pos + rot @ self.marker_off

        vel = np.zeros((self.N, 3))
        vel[self.spring_idx, 0] = -self.v * self.r * om * sp
        vel[self.spring_idx, 1] = self.v * self.r * om * cp
        vel[self.spring_idx, 2] = self.um * w
        omega_vec = np.array([0.0, 0.0, om])
        vel[self.load_idx] = np.array([0.0, 0.0, w])
        vel[self.marker_idx] = np.array([0.0, 0.0, w]) + np.cross(
            omega_vec, rot @ self.marker_off
        )

        angw = np.zeros((self.N, 3))
        angw[self.spring_idx, 2] = self.v * om
        angw[self.load_idx] = omega_vec
        angw[self.marker_idx] = omega_vec

        quat = np.zeros((self.N, 4))
        quat[:, 3] = 1.0
        half = 0.5 * th * self.v
        quat[self.spring_idx, 2] = np.sin(half)
        quat[self.spring_idx, 3] = np.cos(half)
        quat[self.load_idx] = quat_load
        quat[self.marker_idx] = quat_load

        inertia = np.zeros((self.N, 3, 3))
        inertia[self.load_idx] = rot @ np.diag(self.I_body) @ rot.T

        attributes = dict(self.base_attributes)
        attributes["inertia_tensor"] = inertia
        attributes["mass_quantity"] = self.mass.copy()

        f_terms, p_terms, c_terms = self._budget_terms(y)

        fields = {
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": f_terms,
            "budget_force_position": p_terms,
            "budget_external_couple": c_terms,
            "budget_cumulative_external_impulse": y[4:13].reshape(3, 3).copy(),
            "budget_cumulative_external_angular_impulse": y[13:22].reshape(3, 3).copy(),
            "budget_cumulative_inflow_momentum": np.zeros((3, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((3, 3)),
            "budget_cumulative_inflow_angular_momentum": np.zeros((3, 3)),
            "budget_cumulative_outflow_angular_momentum": np.zeros((3, 3)),
            "cumulative_mass_source": np.zeros(1),
        }

        scalars = dict(self.base_scalars)
        scalars["reference_momentum"] = float(self.ref_momentum)
        scalars["reference_angular_momentum"] = float(self.ref_angular_momentum)

        return StateFrame(
            float(t),
            pos,
            vel,
            orientation=quat,
            angular_velocity=angw,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    # ------------------------------------------------------------------
    # solver contract
    # ------------------------------------------------------------------
    def advance_to(self, time):
        t = float(time)
        if t > self._t:
            n_full = int(np.floor((t - self._t) / self._h + 1.0e-9))
            for _ in range(n_full):
                self._Y = self._rk4_step(self._Y, self._h)
                self._t += self._h
            remainder = t - self._t
            if remainder > 1.0e-9:
                self._Y = self._rk4_step(self._Y, remainder)
                self._t = t
        return self._make_frame(t, self._Y)

    def diagnostics(self):
        return dict(self._diag)
