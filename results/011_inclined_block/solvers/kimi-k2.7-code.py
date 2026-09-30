"""Physics solver for the inclined block scene.

The block is modelled as a rigid body constrained to slide without tumbling on a
fixed inclined plane.  The plane normal and down-slope direction are derived from
the frozen initial orientation.  Contact is enforced by keeping the centre of
mass at the correct normal distance above the plane; tangential motion follows
the 1-D Coulomb-friction sliding law and is advanced with semi-implicit Euler.

Required StateFrame fields are populated from the dynamic state:
  * state.position / velocity / orientation / angular_velocity / mass / body_id
  * fields.external_force
  * fields.contact_signed_distance
  * fields.constraint_value / fields.constraint_target
  * scalars.constraint_scale / scalars.contact_length_scale
"""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        # Scene and bookkeeping
        self.scene = None
        self.seed = 0
        self._time = 0.0

        # Dynamic state
        self.p = np.zeros(3, dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.omega = np.zeros(3, dtype=float)

        # Physical parameters
        self.mass = 2.0
        self.gravity = 9.8
        self.mu = 0.3
        self.dt_sub = 1e-4

        # Geometry derived from the frozen orientation
        self.n = np.array([0.0, 0.0, 1.0], dtype=float)
        self.t_dir = np.array([0.0, 1.0, 0.0], dtype=float)
        self.d_com = 0.38  # normal distance from plane origin to COM

        # Gravity projections along / into the plane
        self.g_vec = np.array([0.0, 0.0, -self.gravity], dtype=float)
        self.g_n = 0.0
        self.g_t = 0.0
        self.acc_net = 0.0

        # Initial on-plane reference point for analytical target
        self.p_start = np.zeros(3, dtype=float)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self._time = 0.0

        init = scene.initial_state()
        self.p = np.asarray(init.position, dtype=float).reshape(3).copy()
        self.v = np.asarray(init.velocity, dtype=float).reshape(3).copy()
        if init.orientation is not None and np.asarray(init.orientation).size >= 4:
            self.q = np.asarray(init.orientation, dtype=float).reshape(4).copy()
        else:
            self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.omega = np.zeros(3, dtype=float)
        if init.mass is not None:
            self.mass = float(np.asarray(init.mass).reshape(-1)[0])

        # Plane normal from quaternion xyzw: rotate local +z axis.
        x, y, z, w = self.q
        n = np.array(
            [
                2.0 * (x * z + y * w),
                2.0 * (y * z - x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dtype=float,
        )
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            n = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            n /= n_norm
        self.n = n

        # Down-slope unit vector = projection of gravity onto the plane.
        g_parallel = self.g_vec - np.dot(self.g_vec, self.n) * self.n
        g_parallel_norm = np.linalg.norm(g_parallel)
        if g_parallel_norm < 1e-12:
            self.t_dir = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            self.t_dir = g_parallel / g_parallel_norm

        # The block must sit on the plane top surface. Plane half-thickness in
        # normal direction is 0.08; block half-extent along its local z is 0.3.
        self.d_com = 0.08 + 0.3
        self.p = self.p - (np.dot(self.p, self.n) - self.d_com) * self.n
        self.p_start = self.p.copy()

        # Remove any normal / off-slope velocity components and keep only the
        # tangential component.
        v_t = np.dot(self.v, self.t_dir)
        self.v = v_t * self.t_dir

        # Precompute gravity components.
        self.g_n = -np.dot(self.g_vec, self.n)
        self.g_t = np.dot(self.g_vec, self.t_dir)

        # Net acceleration magnitude for the analytical sliding target.  When the
        # block starts from rest and the slope overcomes static friction, the
        # tangential acceleration is g_t - mu * g_n.
        self.acc_net = max(0.0, self.g_t - self.mu * self.g_n)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self._time:
            # Time must be monotonic; return the current state at the requested time.
            return self._make_frame(target)

        while self._time < target:
            dt = min(self.dt_sub, target - self._time)
            self._step(dt)
            self._time += dt

        return self._make_frame(target)

    def _step(self, dt: float) -> None:
        v_t = float(np.dot(self.v, self.t_dir))

        # Coulomb friction on the inclined plane.
        if abs(v_t) > 1e-12:
            a_t = self.g_t - self.mu * self.g_n * np.sign(v_t)
        else:
            if abs(self.g_t) <= self.mu * self.g_n:
                a_t = 0.0
            else:
                a_t = self.g_t - self.mu * self.g_n * np.sign(self.g_t)

        # Semi-implicit Euler step for the tangential coordinate.
        v_t_new = v_t + a_t * dt
        # Damped rest: remove tiny spurious oscillations when net acceleration is zero.
        if a_t == 0.0:
            v_t_new = 0.0

        self.v = v_t_new * self.t_dir
        self.p = self.p + self.v * dt

        # Enforce the contact constraint: COM stays at fixed normal distance.
        self.p = self.p - (np.dot(self.p, self.n) - self.d_com) * self.n

    def _make_frame(self, time: float) -> StateFrame:
        # Signed distance from the block bottom to the plane top surface.
        signed_dist = float(np.dot(self.p, self.n) - self.d_com)

        # Analytical sliding target from rest on the plane: s = 0.5 * a * t^2.
        t_eff = max(0.0, time)
        s_target = 0.5 * self.acc_net * t_eff * t_eff
        p_target = self.p_start + s_target * self.t_dir

        position = np.array([self.p.copy()], dtype=float)
        velocity = np.array([self.v.copy()], dtype=float)
        orientation = np.array([self.q.copy()], dtype=float)
        angular_velocity = np.array([self.omega.copy()], dtype=float)
        mass = np.array([self.mass], dtype=float)
        body_id = np.array([1], dtype=int)

        fields = {
            "constraint_target": np.array([p_target.copy()], dtype=float),
            "constraint_value": np.array([self.p.copy()], dtype=float),
            "contact_signed_distance": np.array([signed_dist], dtype=float),
            "external_force": np.array([self.mass * self.g_vec.copy()], dtype=float),
        }
        scalars = {
            "constraint_scale": 1.0,
            "contact_length_scale": 0.1,
        }

        return StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "sub_step": self.dt_sub,
            "mass": self.mass,
            "friction_coefficient": self.mu,
        }
