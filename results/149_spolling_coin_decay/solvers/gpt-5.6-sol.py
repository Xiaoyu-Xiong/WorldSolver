"""Deterministic rigid-body solver for the spinning coin scene."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


def _rotation(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=float)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0., -v[2], v[1]], [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]], dtype=float)


def _quat_step(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    speed = float(np.linalg.norm(omega))
    if speed * dt < 1.e-10:
        dq = np.r_[0.5 * omega * dt, 1.]
    else:
        half = 0.5 * speed * dt
        dq = np.r_[omega * (math.sin(half) / speed), math.cos(half)]
    av, aw, bv, bw = dq[:3], dq[3], q[:3], q[3]
    result = np.r_[aw*bv + bw*av + np.cross(av, bv), aw*bw - np.dot(av, bv)]
    return result / np.linalg.norm(result)


class BenchmarkSolver:
    """Semi-implicit six degree-of-freedom integration of a coin."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        config, body = scene.config, scene.config["bodies"][0]
        self.mass = float(body["mass"])
        self.inv_mass = 1. / self.mass
        self.radius, self.half_height = .48, .04
        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.mu = float(config["contact"]["friction"])
        self.restitution = float(config["contact"]["restitution"])
        self.length_scale = float(config["contact"]["length_scale"])
        self.position = np.asarray(body["position"], dtype=float).copy()
        self.velocity = np.zeros(3, dtype=float)
        self.orientation = np.asarray(body.get("orientation", [0., 0., 0., 1.]),
                                      dtype=float).copy()
        self.orientation /= np.linalg.norm(self.orientation)
        axis = _rotation(self.orientation)[:, 2]
        self.angular_velocity = 22. * axis + np.array([0., 0., 4.])
        i_t = self.mass * (3.*self.radius**2 + (2.*self.half_height)**2) / 12.
        i_z = .5 * self.mass * self.radius**2
        self.inertia_body = np.diag([i_t, i_t, i_z])
        self.inv_inertia_body = np.diag([1./i_t, 1./i_t, 1./i_z])
        self.time, self.internal_dt, self.step_count = 0., .001, 0
        self.last_contact_force = np.zeros(3, dtype=float)

    def _support(self):
        axis = _rotation(self.orientation)[:, 2]
        horizontal = np.array([-axis[0]*axis[2], -axis[1]*axis[2],
                               1.-axis[2]*axis[2]])
        hn = float(np.linalg.norm(horizontal))
        if hn < 1.e-10:
            relative = (-1. if axis[2] >= 0. else 1.) * self.half_height * axis
        else:
            side = -self.radius * horizontal / hn
            relative = side + (-1. if axis[2] >= 0. else 1.) * self.half_height * axis
        return relative, float(self.position[2] + relative[2])

    def _inertia_world(self):
        rot = _rotation(self.orientation)
        return rot @ self.inertia_body @ rot.T, rot @ self.inv_inertia_body @ rot.T

    def _apply_impulse(self, impulse, arm, inv_inertia):
        self.velocity += impulse * self.inv_mass
        self.angular_velocity += inv_inertia @ np.cross(arm, impulse)

    def _step(self, dt: float) -> None:
        inertia, inv_inertia = self._inertia_world()
        angular_momentum = inertia @ self.angular_velocity
        self.angular_velocity += inv_inertia @ (-np.cross(self.angular_velocity,
                                                          angular_momentum)) * dt
        self.velocity += self.gravity * dt
        arm, gap = self._support()
        contact_force = np.zeros(3, dtype=float)
        point_velocity = self.velocity + np.cross(self.angular_velocity, arm)
        if gap <= .0015 and (point_velocity[2] < 0. or gap < 0.):
            k = self.inv_mass*np.eye(3) - _skew(arm) @ inv_inertia @ _skew(arm)
            normal_mass = max(float(k[2, 2]), 1.e-12)
            closing = float(point_velocity[2])
            restitution = self.restitution if closing < -.35 else 0.
            correction = min(1., .18*max(-gap, 0.)/dt)
            normal_impulse = max(0., (-(1.+restitution)*closing + correction) / normal_mass)
            normal = np.array([0., 0., normal_impulse])
            self._apply_impulse(normal, arm, inv_inertia)
            point_velocity = self.velocity + np.cross(self.angular_velocity, arm)
            try:
                tangent_xy = np.linalg.solve(k[:2, :2], -point_velocity[:2])
            except np.linalg.LinAlgError:
                tangent_xy = -point_velocity[:2] / max(float(np.trace(k[:2, :2]))*.5, 1.e-12)
            limit, norm = self.mu*normal_impulse, float(np.linalg.norm(tangent_xy))
            if norm > limit and norm > 0.: tangent_xy *= limit/norm
            self._apply_impulse(np.array([tangent_xy[0], tangent_xy[1], 0.]), arm, inv_inertia)
            normal_force = normal_impulse / dt
            omega = self.angular_velocity.copy()
            resistance = np.array([.010, .010, .004]) * normal_force
            angular_impulse = np.zeros(3)
            for j in range(3):
                if abs(omega[j]) > 1.e-10:
                    available = abs((inertia @ omega)[j])
                    angular_impulse[j] = -math.copysign(min(resistance[j]*dt, available), omega[j])
            self.angular_velocity += inv_inertia @ angular_impulse
            contact_force = (normal + np.array([tangent_xy[0], tangent_xy[1], 0.])) / dt
        self.position += self.velocity * dt
        self.orientation = _quat_step(self.orientation, self.angular_velocity, dt)
        _, new_gap = self._support()
        if new_gap < 0.: self.position[2] -= new_gap
        arm_after, final_gap = self._support()
        point_speed = np.linalg.norm(self.velocity + np.cross(self.angular_velocity, arm_after))
        if final_gap < 2.e-4 and point_speed < .018 and np.linalg.norm(self.angular_velocity) < .12:
            self.velocity[:] = 0.; self.angular_velocity[:] = 0.
        self.last_contact_force = contact_force
        self.time += dt
        self.step_count += 1

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.e-10: raise ValueError("advance_to times must be nondecreasing")
        while self.time < target - 1.e-12:
            self._step(min(self.internal_dt, target-self.time))
        self.time = target
        _, gap = self._support()
        interaction_pair = np.vstack((self.last_contact_force,
                                      -self.last_contact_force))
        frame = StateFrame(target, self.position.reshape(1, 3).copy(),
                           self.velocity.reshape(1, 3).copy(),
                           orientation=self.orientation.reshape(1, 4).copy(),
                           angular_velocity=self.angular_velocity.reshape(1, 3).copy(),
                           mass=np.array([self.mass]), body_id=np.array([0], dtype=np.int32),
                           fields={"contact_signed_distance": np.array([gap]),
                                   "external_force": (self.mass*self.gravity).reshape(1, 3),
                                   "interaction_force": interaction_pair},
                           scalars={"contact_length_scale": self.length_scale})
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"integrator": "semi_implicit_newton_euler",
                "internal_steps": self.step_count, "simulated_time": self.time}
