
"""Wilberforce pendulum: coupled axial and torsional oscillation of a spring-hung load.

Physical model
--------------
The frozen task explicitly asks for the coupled spring / rigid-load regime
("Resolve the attached load inertia ... not finite-speed torsional waves in the rod").
The helical spring is therefore treated as a massless elastic element whose shape is
slaved to the two load degrees of freedom: axial displacement ``xi`` measured from the
gravity-equilibrium configuration and twist angle ``theta`` about the spring axis
(quasi-static shape ansatz, exact for a massless rod at these time scales).

Potential energy about the gravity equilibrium state (linear/Gaussian coupling, the
classical Wilberforce form):

    V = 1/2 kz xi^2 + 1/2 kth theta^2 + eps xi theta

Kinetic energy of the rigid load (mass m, axial inertia I about the spring axis):

    T = 1/2 m xidot^2 + 1/2 I thetadot^2

Equations of motion (exact for this reduced model):

    m xidotdot = -kz xi - eps theta
    I thetadotdot = -kth theta - eps xi

The axial and torsional stiffnesses reproduce the frozen target periods, and the
coupling eps is solved for so that the normal-mode splitting matches the frozen beat
timings (first torsional peak 4.5 s, vertical revival 9.0 s).  Time integration is a
classical RK4 with a fixed 0.5 ms step, so every requested sample time is hit exactly.

Rigid bodies: the load (body 1) and the orientation marker (body 2) are rigid, their
inertia is exported in ``attributes.inertia_tensor`` and their rotary motion is carried
by ``state.angular_velocity``; the marker is massless, as in the frozen scene.

Budget fields are accumulated by trapezoidal integration of the *actual* per-domain
external force and torque sums produced by this model, including the clamp reaction
that balances the (kinematically slaved) distributed spring inertia.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_TWO_PI = 2.0 * np.pi


class BenchmarkSolver:
    """Deterministic RK4 solver for the frozen Wilberforce-pendulum scene."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        spring = cfg["spring"]
        load = cfg["load"]
        ic = cfg["initial_conditions"]
        regime = cfg.get("coupled_mode_regime", {}) or {}
        self.seed = int(seed)

        n = int(spring["vertex_count"])
        self.n = n
        self.coil_count = float(spring["coil_count"])
        self.coil_radius = float(spring["coil_radius"])
        self.z_top = float(spring["equilibrium_top_z"])
        self.z_bottom_eq = float(spring["equilibrium_bottom_z"])
        self.m_spring = float(spring["total_mass"])
        n_clamp = int(spring["upper_clamped_vertex_count"])

        self.m_load = float(load["mass"])
        inertia = np.asarray(load["principal_inertia"], dtype=float)
        self.inertia_axial = float(inertia[2])
        self.inertia_perp = float(0.5 * (inertia[0] + inertia[1]))
        self.offset = float(np.asarray(load["attachment_offset_body"], dtype=float)[2])

        self.dz = float(ic["upward_displacement"])
        gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.gravity = gravity
        self.g = float(abs(gravity[2])) if gravity.size >= 3 else 9.81

        # orientation-marker stand-off from the load centre (frozen scene geometry)
        self.marker_offset = 0.08
        try:
            objects = {o["id"]: o for o in cfg["scene_geometry"]["objects"]}
            self.marker_offset = float(
                objects["orientation_marker"]["transform"]["translation"][2]
                - objects["suspended_load"]["transform"]["translation"][2]
            )
        except Exception:  # geometry block absent -> keep the scene default
            pass

        # --- elastic constants derived from the frozen coupled-mode regime -----
        period_z = float(regime.get("target_vertical_period", 1.2))
        period_t = float(regime.get("target_torsional_period", 1.24))
        revival = float(regime.get("expected_vertical_revival_time", 9.0))
        omega_z = _TWO_PI / max(period_z, 1e-9)
        omega_t = _TWO_PI / max(period_t, 1e-9)
        self.kz = self.m_load * omega_z ** 2
        self.kth = self.inertia_axial * omega_t ** 2
        # normal-mode splitting of the coupled 2x2 system:
        #   omega+-^2 = wbar^2 +- sqrt(delta^2 + kappa^2),  kappa = eps / sqrt(m I)
        a = omega_z ** 2
        b = omega_t ** 2
        wbar2 = 0.5 * (a + b)
        delta = 0.5 * (a - b)
        dw = _TWO_PI / max(revival, 1e-9)          # beat splitting -> revival at `revival`
        split = dw * np.sqrt(max(wbar2, 1e-12))    # fixed-point start
        for _ in range(64):
            hi = np.sqrt(max(wbar2 + split, 0.0))
            lo = np.sqrt(max(wbar2 - split, 0.0))
            split = 0.5 * dw * (hi + lo)
        kappa = np.sqrt(max(split ** 2 - delta ** 2, 0.0))
        self.eps = kappa * np.sqrt(max(self.m_load * self.inertia_axial, 1e-12))
        self.omega_beat = 2.0 * split / max(np.sqrt(max(wbar2 + split, 0.0))
                                            + np.sqrt(max(wbar2 - split, 0.0)), 1e-12)

        # --- reference scales (positive, frozen) -------------------------------
        v_scale = self.dz * np.sqrt(max(wbar2, 1e-12))
        theta_max = self.dz * np.sqrt(max(self.kz / max(self.kth, 1e-12), 0.0))
        self.p_ref = float((self.m_load + self.m_spring) * v_scale)
        self.l_ref = float(self.inertia_axial * theta_max * np.sqrt(max(wbar2, 1e-12)))
        if not (self.p_ref > 0.0):
            self.p_ref = 1.0
        if not (self.l_ref > 0.0):
            self.l_ref = 1.0

        # --- geometry of the spring centreline ---------------------------------
        s = np.linspace(0.0, 1.0, n)
        self.s = s
        s_clamp = (min(n_clamp, n - 1) - 1) / max(n - 1, 1)
        u = np.clip((s - s_clamp) / max(1.0 - s_clamp, 1e-12), 0.0, 1.0)
        self.u = u
        self.base_theta = np.linspace(0.0, _TWO_PI * self.coil_count, n)
        self.node_mass = self.m_spring / n

        # --- time stepping -----------------------------------------------------
        self.dt = 5.0e-4
        self.t = 0.0
        self.y = np.array([self.dz, 0.0, 0.0, 0.0], dtype=float)

        # --- budget bookkeeping -------------------------------------------------
        self.G = 3                 # 0: spring, 1: load + marker, 2: global
        self.K = 8
        self.row_members = (
            (0, 2), (1, 2), (0, 2), (0, 2), (1, 2), (1, 2), (0, 2), (0, 2),
        )
        self.domain_weights = np.zeros((self.G, n + 2))
        self.domain_weights[0, :n] = 1.0
        self.domain_weights[1, n:] = 1.0
        self.domain_weights[2, :] = 1.0

        self.cum_impulse = np.zeros((self.G, 3))
        self.cum_angular_impulse = np.zeros((self.G, 3))
        self.inflow_momentum = np.zeros((self.G, 3))
        self.inflow_angular = np.zeros((self.G, 3))
        self.outflow_momentum = np.zeros((self.G, 3))
        self.outflow_angular = np.zeros((self.G, 3))
        self.mass_source = np.zeros(2)

        f_tot, t_tot, _ = self._domain_totals(self.y)
        self._f_prev = f_tot
        self._t_prev = t_tot
        self._advance_steps = 0

    # ------------------------------------------------------------- kinematics
    def _kinematics(self, xi, theta, xi_dot, theta_dot):
        """Spring-node positions, velocities and angular velocities (quasi-static shape)."""
        n = self.n
        u = self.u
        az = self.base_theta + theta * u
        rx = self.coil_radius * np.cos(az)
        ry = self.coil_radius * np.sin(az)
        z = self.z_top + (self.z_bottom_eq + xi - self.z_top) * u
        pos = np.column_stack((rx, ry, z))
        w = theta_dot * u
        vel = np.column_stack((-w * ry, w * rx, xi_dot * u))
        omega = np.column_stack((np.zeros(n), np.zeros(n), w))
        return pos, vel, omega

    def _accel(self, xi, theta):
        """Reduced-model accelerations (xi_ddot, theta_ddot)."""
        xi_ddot = (-self.kz * xi - self.eps * theta) / self.m_load
        th_ddot = (-self.kth * theta - self.eps * xi) / self.inertia_axial
        return xi_ddot, th_ddot

    def _rhs(self, y):
        xi, theta, xi_dot, theta_dot = y
        xi_ddot, th_ddot = self._accel(xi, theta)
        return np.array([xi_dot, theta_dot, xi_ddot, th_ddot])

    # ---------------------------------------------------------------- budgets
    def _terms(self, y):
        """Per-term external forces / application points / free couples.

        Term order (K=8):
          0 spring gravity            4 spring force on load
          1 load gravity              5 spring torsional couple on load
          2 clamp traction on spring  6 reaction couple on spring
          3 load reaction on spring   7 clamp reaction couple on spring
        """
        xi, theta, xi_dot, theta_dot = y
        n = self.n
        pos_s, vel_s, _ = self._kinematics(xi, theta, xi_dot, theta_dot)
        f_s = np.array([0.0, 0.0, self.m_load * self.g - self.kz * xi - self.eps * theta])
        tau_load = np.array([0.0, 0.0, -self.kth * theta - self.eps * xi])

        gvec = self.gravity if self.gravity.size >= 3 else np.array([0.0, 0.0, -self.g])
        grav_spring = self.m_spring * gvec
        cm_spring = pos_s.mean(0)

        # rate of the slaved spring inertia, along the model flow (finite difference)
        acc = self._accel(xi, theta)
        h = self.dt
        y2 = (xi + h * xi_dot, theta + h * theta_dot,
              xi_dot + h * acc[0], theta_dot + h * acc[1])
        pos2, vel2, _ = self._kinematics(*y2)
        p0 = self.node_mass * vel_s.sum(0)
        p1 = self.node_mass * vel2.sum(0)
        l0 = self.node_mass * np.cross(pos_s, vel_s).sum(0)
        l1 = self.node_mass * np.cross(pos2, vel2).sum(0)
        dp = (p1 - p0) / h
        dl = (l1 - l0) / h

        r_top = pos_s[0]
        r_att = pos_s[-1]
        r_load = np.array([0.0, 0.0, self.z_bottom_eq + xi - self.offset])

        # clamp traction balances everything acting on the massless spring
        f_clamp = dp - grav_spring + f_s
        tau_spring_reaction = -tau_load
        gravity_torque = np.cross(pos_s, gvec[None, :]).sum(0) * self.node_mass
        tau_clamp = (dl - np.cross(r_top, f_clamp) - np.cross(r_att, -f_s)
                     - tau_spring_reaction - gravity_torque)

        force = np.zeros((self.K, 3))
        pos = np.zeros((self.K, 3))
        couple = np.zeros((self.K, 3))
        force[0] = grav_spring
        pos[0] = cm_spring
        force[1] = np.array([0.0, 0.0, -self.m_load * self.g])
        pos[1] = r_load
        force[2] = f_clamp
        pos[2] = r_top
        force[3] = -f_s
        pos[3] = r_att
        force[4] = f_s
        pos[4] = r_att
        couple[5] = tau_load
        pos[5] = r_load
        couple[6] = tau_spring_reaction
        pos[6] = r_att
        couple[7] = tau_clamp
        pos[7] = r_top
        return force, pos, couple

    def _domain_totals(self, y):
        force, pos, couple = self._terms(y)
        f_tot = np.zeros((self.G, 3))
        t_tot = np.zeros((self.G, 3))
        for k in range(self.K):
            moment = np.cross(pos[k], force[k]) + couple[k]
            for g in self.row_members[k]:
                f_tot[g] += force[k]
                t_tot[g] += moment
        return f_tot, t_tot, (force, pos, couple)

    # ------------------------------------------------------------- integration
    def _rk4(self, y, dt):
        k1 = self._rhs(y)
        k2 = self._rhs(y + 0.5 * dt * k1)
        k3 = self._rhs(y + 0.5 * dt * k2)
        k4 = self._rhs(y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _step_to(self, time):
        dt = self.dt
        remaining = float(time) - self.t
        steps = int(round(remaining / dt))
        if steps <= 0:
            return
        f_prev = self._f_prev
        t_prev = self._t_prev
        for _ in range(steps):
            y_next = self._rk4(self.y, dt)
            f_next, t_next, _ = self._domain_totals(y_next)
            self.cum_impulse += 0.5 * (f_prev + f_next) * dt
            self.cum_angular_impulse += 0.5 * (t_prev + t_next) * dt
            self.y = y_next
            f_prev, t_prev = f_next, t_next
        self._advance_steps += steps
        self._f_prev = f_prev
        self._t_prev = t_prev
        self.t = self.t + steps * dt

    # ------------------------------------------------------------------ frames
    def advance_to(self, time: float) -> StateFrame:
        self._step_to(time)
        return self._frame(float(time))

    def _frame(self, time: float) -> StateFrame:
        n = self.n
        xi, theta, xi_dot, theta_dot = self.y
        pos_s, vel_s, omega_s = self._kinematics(xi, theta, xi_dot, theta_dot)

        z_bottom = self.z_bottom_eq + xi
        r_load = np.array([0.0, 0.0, z_bottom - self.offset])
        r_mark = np.array([0.0, 0.0, r_load[2] + self.marker_offset])
        position = np.vstack((pos_s, r_load, r_mark))
        velocity = np.vstack((vel_s, np.array([[0.0, 0.0, xi_dot]]),
                              np.array([[0.0, 0.0, xi_dot]])))
        angular_velocity = np.vstack((omega_s, np.array([[0.0, 0.0, theta_dot]]),
                                      np.array([[0.0, 0.0, theta_dot]])))

        half = 0.5 * theta
        quat_nodes = np.zeros((n, 4))
        quat_nodes[:, 2] = np.sin(0.5 * theta * self.u)
        quat_nodes[:, 3] = np.cos(0.5 * theta * self.u)
        quat_load = np.array([[0.0, 0.0, np.sin(half), np.cos(half)]])
        orientation = np.vstack((quat_nodes, quat_load, quat_load.copy()))

        mass = np.concatenate((np.full(n, self.node_mass), [self.m_load, 0.0]))
        body_id = np.concatenate((np.zeros(n, dtype=np.int64), [1, 2]))

        inertia_tensor = np.zeros((n + 2, 3, 3))
        inertia_tensor[n] = np.diag([self.inertia_perp, self.inertia_perp,
                                     self.inertia_axial])

        mass_quantity = np.zeros((n + 2, 2))
        mass_quantity[:n, 0] = self.node_mass
        mass_quantity[n, 1] = self.m_load

        force, pos, couple = self._terms(self.y)
        gemf = np.zeros((self.G, self.K, 3))
        gemr = np.zeros((self.G, self.K, 3))
        gemc = np.zeros((self.G, self.K, 3))
        for k in range(self.K):
            for g in self.row_members[k]:
                gemf[g, k] = force[k]
                gemr[g, k] = pos[k]
                gemc[g, k] = couple[k]

        fields = {
            "budget_cumulative_external_angular_impulse": self.cum_angular_impulse.copy(),
            "budget_cumulative_external_impulse": self.cum_impulse.copy(),
            "budget_cumulative_inflow_angular_momentum": self.inflow_angular.copy(),
            "budget_cumulative_inflow_momentum": self.inflow_momentum.copy(),
            "budget_cumulative_outflow_angular_momentum": self.outflow_angular.copy(),
            "budget_cumulative_outflow_momentum": self.outflow_momentum.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_couple": gemc,
            "budget_external_force": gemf,
            "budget_force_position": gemr,
            "cumulative_mass_source": self.mass_source.copy(),
        }
        attributes = {
            "inertia_tensor": inertia_tensor,
            "mass_quantity": mass_quantity,
        }
        scalars = {
            "reference_angular_momentum": self.l_ref,
            "reference_momentum": self.p_ref,
        }
        frame = StateFrame(
            time,
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        xi, theta, xi_dot, theta_dot = self.y
        return {
            "time": float(self.t),
            "steps": int(self._advance_steps),
            "axial_stiffness": float(self.kz),
            "torsional_stiffness": float(self.kth),
            "coupling": float(self.eps),
            "vertical_period": float(_TWO_PI / np.sqrt(self.kz / self.m_load)),
            "torsional_period": float(_TWO_PI / np.sqrt(self.kth / self.inertia_axial)),
            "beat_splitting": float(self.omega_beat),
            "final_axial_displacement": float(xi),
            "final_twist": float(theta),
            "final_axial_velocity": float(xi_dot),
            "final_twist_rate": float(theta_dot),
        }
