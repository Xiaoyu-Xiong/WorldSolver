"""Deterministic physics solver for the piston-crank tight-contact mechanism.

The mechanism is a planar slider-crank embedded in the frozen scene:

* the crank is a rigid rotor pinned to ``crank_axle`` at ``(-0.8, 0, 1)`` and turns
  about the world z axis (the axle cylinder is rotated 90 deg about x, so its axis
  is z); the crank pin sits at radius ``r`` from the pivot,
* the connecting rod is a rigid link of length ``L`` between the crank pin and the
  wrist pin, which coincides with the centre of the piston body,
* the piston is a prismatic slider: it translates along x inside the two rigid
  guide plates whose inner faces are 0.12 m apart vertically.  Gravity acts along
  -z, so the piston is pressed against ``guide_lower`` and stays in tight contact
  there, while the guides carry the normal load and Coulomb friction acts along x.

Because the loop closure is holonomic the in-plane motion has a single degree of
freedom.  We integrate the Lagrange equation of that coordinate

    M(theta) * theta'' + 0.5 * M'(theta) * theta'^2 = tau_motor + Q_friction

with a classical 4th-order Runge-Kutta scheme at a fixed substep, where ``M`` is
the configuration-dependent generalised inertia of crank + rod + piston, the motor
is a clamped proportional speed controller tracking a smooth spin-up ramp from rest
("prescribed motor"), and the generalised friction force is the Coulomb force at
the guide projected onto the crank coordinate.  The out-of-plane (z) motion of the
piston is integrated separately with a rigid, restitution-based contact projection
against ``guide_lower``.  Every reported quantity (body states, external forces,
interaction force pairs, signed contacts) is derived from that same physical
integration, so the exported state is dynamically consistent.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

_TWO_PI = 2.0 * math.pi


def _quat_z(angle: float) -> np.ndarray:
    """Quaternion (xyzw) for a rotation of ``angle`` about the world z axis."""
    return np.array([0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)])


def _rot_z(angle: float) -> np.ndarray:
    ca, sa = math.cos(angle), math.sin(angle)
    return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])


class BenchmarkSolver:
    """Rigid slider-crank with tight guide contact, integrated by time-stepping."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", {}) or {})
        gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        if gravity.shape != (3,) or not np.isfinite(gravity).all():
            gravity = np.array([0.0, 0.0, -9.81])
        self.gravity = gravity
        self.g_z = float(gravity[2])

        contact = dict(config.get("contact", {}) or {})
        self.mu = float(contact.get("friction", 0.55))
        self.restitution = float(contact.get("restitution", 0.12))
        self.length_scale = float(contact.get("length_scale", 0.02))

        bodies = list(config.get("bodies", []) or [])
        mass = [float(b.get("mass", 1.0)) for b in bodies]
        while len(mass) < 3:
            mass.append(1.0)
        self.mass = np.asarray(mass[:3], dtype=float)
        self.body_id = np.arange(3, dtype=np.int32)

        # ---- frozen mechanism geometry (scene_geometry.yaml) -----------------
        self.pivot = np.array([-0.8, 0.0, 1.0])
        self.crank_radius = 0.35        # crank pin throw (crank disc radius 0.357)
        self.rod_length = 1.6           # connecting rod joint-to-joint length
        piston_half_z = 0.31            # piston.obj half extent along z
        guide_lower_top = 0.63          # guide_lower centre 0.55 + half extent 0.08
        guide_upper_bottom = 1.37       # guide_upper centre 1.45 - half extent 0.08
        self.piston_half_z = piston_half_z
        self.z_rest = guide_lower_top + piston_half_z          # 0.94
        self.guide_upper_bottom = guide_upper_bottom
        self.z_start = 0.5 * (guide_lower_top + guide_upper_bottom)  # aligned centre

        # rigid-body inertia (about the z axis, planar motion)
        self.inertia_crank = 0.15
        self.inertia_rod = 0.22
        self.piston_inertia_diag = np.array([0.1457, 0.1411, 0.1587])
        self.crank_inertia_diag = np.array([0.10, 0.10, self.inertia_crank])

        # ---- motor: prescribed ramp from rest, speed-controlled --------------
        self.omega_target = _TWO_PI          # 1 revolution per second
        self.ramp_time = 0.4
        self.motor_gain = 40.0
        self.motor_torque_max = 80.0

        # ---- integration controls -------------------------------------------
        self.dt = 2.5e-4
        self.friction_eps = 3.0e-3

        # ---- primary state ---------------------------------------------------
        self.t = 0.0
        self.theta = 0.0        # crank angle, 0 => piston at top dead centre
        self.theta_dot = 0.0
        self.z = self.z_start
        self.z_dot = 0.0
        self.in_contact = False
        self._normal_accum = 0.0
        self._normal_time = 0.0
        self.steps = 0

    # -------------------------------------------------------------- kinematics
    def _kinematics(self, theta: float):
        """Positions and theta-derivatives of the loop closure at crank angle."""
        r, L = self.crank_radius, self.rod_length
        s = r * math.sin(theta)          # crank pin y
        c = r * math.cos(theta)          # crank pin x offset from pivot
        D = math.sqrt(max(L * L - s * s, 1e-12))
        x_c = self.pivot[0] + c + D      # wrist pin x (piston centre)
        dx = -s - (s * c) / D            # d x_c / d theta
        ddx = -c - ((c * c - s * s) / D + (s * s * c * c) / (D * D * D))
        return x_c, dx, ddx, s, c, D

    def _generalized_inertia(self, theta: float):
        _, dx, _, s, c, D = self._kinematics(theta)
        dgx = 0.5 * (dx - s)             # d(rod centre)/d theta
        dgy = 0.5 * c
        k = c / D                        # d(rod angle)/d theta
        m_rod, m_pis = self.mass[1], self.mass[2]
        return (
            self.inertia_crank
            + self.inertia_rod * k * k
            + m_rod * (dgx * dgx + dgy * dgy)
            + m_pis * dx * dx
        )

    def _generalized_inertia_prime(self, theta: float):
        _, dx, ddx, s, c, D = self._kinematics(theta)
        dgx, dgy = 0.5 * (dx - s), 0.5 * c
        dgxp, dgyp = 0.5 * (ddx - c), -0.5 * s
        k = c / D
        k_prime = -s * (self.rod_length ** 2 - self.crank_radius ** 2) / (D * D * D)
        m_rod, m_pis = self.mass[1], self.mass[2]
        return (
            self.inertia_rod * 2.0 * k * k_prime
            + m_rod * 2.0 * (dgx * dgxp + dgy * dgyp)
            + m_pis * 2.0 * dx * ddx
        )

    # ------------------------------------------------------------ motor/friction
    def _target_speed(self, t: float) -> float:
        if t >= self.ramp_time:
            return self.omega_target
        u = t / self.ramp_time
        return self.omega_target * u * u * (3.0 - 2.0 * u)

    def _motor(self, t: float, theta_dot: float) -> float:
        tau = self.motor_gain * (self._target_speed(t) - theta_dot)
        return max(-self.motor_torque_max, min(self.motor_torque_max, tau))

    def _friction_x(self, theta_dot: float, dx: float, normal: float) -> float:
        """Coulomb friction on the piston along x, opposing the sliding velocity."""
        if normal <= 0.0:
            return 0.0
        v = dx * theta_dot
        return -self.mu * normal * math.tanh(v / self.friction_eps)

    # ---------------------------------------------------------------- dynamics
    def _deriv(self, t: float, theta: float, theta_dot: float):
        _, dx, _, _, _, _ = self._kinematics(theta)
        normal = self.mass[2] * abs(self.g_z) if self.in_contact else 0.0
        friction = self._friction_x(theta_dot, dx, normal)
        torque = self._motor(t, theta_dot) + friction * dx
        m_eff = self._generalized_inertia(theta)
        m_prime = self._generalized_inertia_prime(theta)
        return theta_dot, (torque - 0.5 * m_prime * theta_dot * theta_dot) / m_eff

    def _step(self, dt: float) -> None:
        th, thd, t = self.theta, self.theta_dot, self.t
        k1 = self._deriv(t, th, thd)
        k2 = self._deriv(t + 0.5 * dt, th + 0.5 * dt * k1[0], thd + 0.5 * dt * k1[1])
        k3 = self._deriv(t + 0.5 * dt, th + 0.5 * dt * k2[0], thd + 0.5 * dt * k2[1])
        k4 = self._deriv(t + dt, th + dt * k3[0], thd + dt * k3[1])
        self.theta = th + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        self.theta_dot = thd + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0

        # --- vertical piston motion with rigid guide contact (restitution) ---
        weight = self.mass[2] * abs(self.g_z)
        normal_force = 0.0
        self.z_dot -= abs(self.g_z) * dt
        self.z += self.z_dot * dt
        if self.z <= self.z_rest:
            self.z = self.z_rest
            if self.z_dot < 0.0:
                before = self.z_dot
                self.z_dot = -self.restitution * self.z_dot
                normal_force += self.mass[2] * (before - self.z_dot) / dt
            normal_force += weight
            self.in_contact = True
        else:
            self.in_contact = False
        self._normal_accum += normal_force * dt
        self._normal_time += dt
        self.t = t + dt
        self.steps += 1

    # ------------------------------------------------------------------- frame
    def _frame(self, normal: float, time: float) -> StateFrame:
        th, thd = self.theta, self.theta_dot
        x_c, dx, ddx, s, c, D = self._kinematics(th)
        m_crank, m_rod, m_pis = self.mass

        # accelerations come from the same equation of motion that was integrated
        thdd = self._deriv(self.t, th, thd)[1]

        bx, by = self.pivot[0] + c, s
        gx, gy = 0.5 * (bx + x_c), 0.5 * by
        phi = math.atan2(-s, D)
        phi_dot = (-c / D) * thd

        v_bx, v_by = -s * thd, c * thd
        v_cx = dx * thd
        v_gx, v_gy = 0.5 * (v_bx + v_cx), 0.5 * v_by

        a_bx = -s * thdd - c * thd * thd
        a_by = c * thdd - s * thd * thd
        a_cx = ddx * thd * thd + dx * thdd
        a_gx, a_gy = 0.5 * (a_bx + a_cx), 0.5 * a_by

        position = np.array(
            [
                [self.pivot[0], self.pivot[1], self.pivot[2]],
                [gx, gy, self.pivot[2]],
                [x_c, 0.0, self.z],
            ]
        )
        velocity = np.array(
            [
                [0.0, 0.0, 0.0],
                [v_gx, v_gy, 0.0],
                [v_cx, 0.0, self.z_dot],
            ]
        )
        orientation = np.array([_quat_z(th), _quat_z(phi), [0.0, 0.0, 0.0, 1.0]])
        angular_velocity = np.array([[0.0, 0.0, thd], [0.0, 0.0, phi_dot], [0.0, 0.0, 0.0]])

        # ---- external (gravity) forces, per body ----------------------------
        external_force = self.mass[:, None] * self.gravity[None, :]

        # ---- internal paired interaction forces ------------------------------
        a_pz = 0.0 if self.in_contact else self.g_z
        friction = self._friction_x(thd, dx, normal)
        contact_piston = np.array([friction, 0.0, normal])
        f_wrist = (
            np.array([m_pis * a_cx, 0.0, m_pis * a_pz])
            - m_pis * self.gravity
            - contact_piston
        )
        f_pin = (
            np.array([m_rod * a_gx, m_rod * a_gy, 0.0])
            - m_rod * self.gravity
            + f_wrist
        )
        upper_normal = 0.0
        upper_piston = np.array([0.0, 0.0, upper_normal])
        interaction_force = np.vstack(
            [
                f_pin,
                -f_pin,
                f_wrist,
                -f_wrist,
                contact_piston,
                -contact_piston,
                upper_piston,
                -upper_piston,
            ]
        )

        # ---- signed contact distances (positive = gap) -----------------------
        gap_lower = self.z - self.piston_half_z - (self.z_rest - self.piston_half_z)
        gap_upper = self.guide_upper_bottom - (self.z + self.piston_half_z)
        contact_signed_distance = np.array([gap_lower, gap_upper])

        # ---- rigid-body inertia tensors in world coordinates -----------------
        inertia = np.zeros((3, 3, 3))
        inertia[0] = _rot_z(th) @ np.diag(self.crank_inertia_diag) @ _rot_z(th).T
        inertia[1] = _rot_z(phi) @ np.diag([0.02, 0.02, self.inertia_rod]) @ _rot_z(phi).T
        inertia[2] = np.diag(self.piston_inertia_diag)

        frame = StateFrame(
            time,
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "inertia_tensor": inertia,
                "release_time": np.zeros(3),
            },
            fields={
                "external_force": external_force,
                "interaction_force": interaction_force,
                "contact_signed_distance": contact_signed_distance,
                "gravity_acceleration": self.gravity.reshape(1, 3),
            },
            scalars={
                "contact_length_scale": self.length_scale,
                "crank_angle": th,
                "motor_torque": self._motor(self.t, thd),
            },
        )
        return frame

    # ------------------------------------------------------------------ runner
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t < target - 1e-12:
            dt = min(self.dt, target - self.t)
            self._step(dt)
            guard += 1
            if guard > 100_000_000:
                break
        normal = self._normal_accum / self._normal_time if self._normal_time > 0.0 else 0.0
        self._normal_accum = 0.0
        self._normal_time = 0.0
        frame = self._frame(normal, target)
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "planar slider-crank with tight guide contact",
            "integrator": "RK4",
            "substep": self.dt,
            "steps": int(self.steps),
            "crank_radius": self.crank_radius,
            "rod_length": self.rod_length,
            "gravity": self.gravity.tolist(),
            "friction": self.mu,
            "restitution": self.restitution,
            "contact_length_scale": self.length_scale,
            "final_time": float(self.t),
            "final_crank_angle": float(self.theta),
            "final_piston_position": [float(self._kinematics(self.theta)[0]), 0.0, float(self.z)],
        }
