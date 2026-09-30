"""Reduced-order physical solver for the elastic torus pile.

Each torus is represented by a deformable volumetric lump with translational
degrees of freedom and three principal stretches. Penalty contact, gravity,
and damped elastic dynamics are integrated with a fixed semi-implicit Euler
step. Original torus vertices are reconstructed from these physical states.
"""
from __future__ import annotations

import math
import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.rng_seed = int(seed)
        initial = scene.initial_state()
        self.rest = np.asarray(initial.position, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.n = len(self.rest)
        self.nb = int(self.body_id.max()) + 1
        self.body_mass = np.array([self.mass[self.body_id == b].sum() for b in range(self.nb)])
        self.centers = np.array([self.rest[self.body_id == b].mean(axis=0) for b in range(self.nb)])
        self.center0 = self.centers.copy()
        self.vel = np.zeros((self.nb, 3), dtype=float)
        self.stretch = np.ones((self.nb, 3), dtype=float)
        self.stretch_rate = np.zeros((self.nb, 3), dtype=float)
        self.gravity = np.array(scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.orient = np.asarray(initial.orientation, dtype=float).copy()
        self.rot = np.array([self._quat_to_mat(self.orient[np.flatnonzero(self.body_id == b)[0]])
                             for b in range(self.nb)])
        self.local = np.empty_like(self.rest)
        for b in range(self.nb):
            ix = self.body_id == b
            self.local[ix] = (self.rest[ix] - self.center0[b]) @ self.rot[b]
        self.current_time = 0.0
        self.cum_impulse = np.zeros((self.nb, 3), dtype=float)
        self.last_ext = self.body_mass[:, None] * self.gravity[None, :]
        self._last_frame = self._make_frame(0.0, self.last_ext)

    @staticmethod
    def _quat_to_mat(q: np.ndarray) -> np.ndarray:
        x, y, z, w = np.asarray(q, dtype=float)
        n = math.sqrt(x*x + y*y + z*z + w*w)
        if n < 1e-12:
            return np.eye(3)
        x, y, z, w = x/n, y/n, z/n, w/n
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

    @staticmethod
    def _post_centers() -> np.ndarray:
        a = 2.35
        return np.array([[a*math.cos(2*math.pi*j/12), a*math.sin(2*math.pi*j/12)]
                         for j in range(12)], dtype=float)

    def _step(self, dt: float) -> None:
        nb = self.nb
        ext_force = self.body_mass[:, None] * self.gravity[None, :]
        ext_force -= 1.2 * self.body_mass[:, None] * self.vel
        force = ext_force.copy()
        boundary = np.zeros_like(force)
        pair_force = np.zeros_like(force)
        contact_load = np.zeros(nb, dtype=float)
        radii = 0.54 * np.maximum(self.stretch[:, 0], self.stretch[:, 1])
        half_z = np.empty(nb, dtype=float)
        for b in range(nb):
            deformed = (self.local[self.body_id == b] * self.stretch[b]) @ self.rot[b].T
            half_z[b] = np.max(np.abs(deformed[:, 2]))
        # floor top is z=0
        bottom = self.centers[:, 2] - half_z
        pen = np.maximum(0.0, -bottom)
        fz = 1800.0 * pen - 35.0 * np.minimum(self.vel[:, 2], 0.0)
        boundary[:, 2] += fz
        contact_load += np.abs(fz)
        # radial penalty against each surrounding post
        posts = self._post_centers()
        for b in range(nb):
            dxy = self.centers[b, :2][None, :] - posts
            dist = np.linalg.norm(dxy, axis=1)
            j = int(np.argmin(dist))
            penetration = radii[b] + 0.13 - dist[j]
            if penetration > 0.0 and dist[j] > 1e-9:
                normal = np.array([dxy[j, 0], dxy[j, 1], 0.0]) / dist[j]
                rel = float(np.dot(self.vel[b], normal))
                mag = 900.0 * penetration - 20.0 * min(rel, 0.0)
                boundary[b] += mag * normal
                contact_load[b] += abs(mag)
        # Pairwise anisotropic contact volumes.  The smaller vertical extent
        # avoids false contacts between the initially separated flat rings.
        for i in range(nb):
            for j in range(i + 1, nb):
                d = self.centers[i] - self.centers[j]
                radial_sum = radii[i] + radii[j]
                vertical_sum = half_z[i] + half_z[j]
                scaled = np.array([d[0] / radial_sum, d[1] / radial_sum,
                                   d[2] / max(vertical_sum, 1e-6)])
                distance = float(np.linalg.norm(scaled))
                if distance < 1.0:
                    grad = np.array([d[0] / radial_sum**2, d[1] / radial_sum**2,
                                     d[2] / max(vertical_sum, 1e-6)**2])
                    gn = float(np.linalg.norm(grad))
                    normal = grad / gn if gn > 1e-9 else np.array([1.0, 0.0, 0.0])
                    penetration = (1.0 - distance) * min(radial_sum, vertical_sum)
                    rel = float(np.dot(self.vel[i] - self.vel[j], normal))
                    mag = 700.0 * penetration - 14.0 * min(rel, 0.0)
                    f = mag * normal
                    force[i] += f
                    force[j] -= f
                    pair_force[i] += f
                    pair_force[j] -= f
                    contact_load[i] += abs(mag)
                    contact_load[j] += abs(mag)
        # incompressible principal-stretch dynamics (sx*sy*sz ~= 1)
        load_ratio = contact_load / np.maximum(self.body_mass * 9.81, 1e-6)
        target_z = np.clip(1.0 - 0.24 * load_ratio, 0.58, 1.0)
        target = np.column_stack((1.0 / np.sqrt(target_z), 1.0 / np.sqrt(target_z), target_z))
        self.stretch_rate += dt * (42.0 * (target - self.stretch) - 9.0 * self.stretch_rate)
        self.stretch += dt * self.stretch_rate
        self.stretch = np.clip(self.stretch, [0.55, 0.55, 0.5], [1.8, 1.8, 1.4])
        # Project exactly onto det(F)=1 and project stretch velocity tangent to
        # that constraint, a reduced incompressible-solid update.
        self.stretch /= np.cbrt(np.prod(self.stretch, axis=1))[:, None]
        trace_rate = np.sum(self.stretch_rate / self.stretch, axis=1) / 3.0
        self.stretch_rate -= self.stretch * trace_rate[:, None]
        # semi-implicit translation update
        acc = (force + boundary) / self.body_mass[:, None]
        self.vel += dt * acc
        self.centers += dt * self.vel
        below = self.centers[:, 2] - half_z
        correction = np.maximum(0.0, -below)
        self.centers[:, 2] += correction
        self.vel[:, 2] = np.where(correction > 0.0, np.maximum(self.vel[:, 2], 0.0), self.vel[:, 2])
        self.last_ext = ext_force + boundary + pair_force
        self.cum_impulse += self.last_ext * dt

    def _make_frame(self, time: float, ext_force: np.ndarray) -> StateFrame:
        pos = np.empty_like(self.rest)
        vel = np.empty_like(self.rest)
        F = np.empty((self.n, 3, 3), dtype=float)
        for b in range(self.nb):
            ix = self.body_id == b
            r = self.rot[b]
            pos[ix] = self.centers[b] + (self.local[ix] * self.stretch[b]) @ r.T
            vel[ix] = self.vel[b] + (self.local[ix] * self.stretch_rate[b]) @ r.T
            F[ix] = r @ np.diag(self.stretch[b]) @ r.T
        weights = np.zeros((self.nb, self.n), dtype=float)
        for b in range(self.nb):
            weights[b, self.body_id == b] = 1.0
        volume = np.prod(self.stretch, axis=1)
        frame = StateFrame(
            time=float(time), position=pos, velocity=vel, orientation=self.orient,
            angular_velocity=np.zeros((self.n, 3)), mass=self.mass, body_id=self.body_id,
            attributes={"rest_position": self.rest.copy(), "mass_quantity": self.mass.copy()},
            fields={
                "budget_cumulative_external_impulse": self.cum_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((self.nb, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((self.nb, 3)),
                "budget_domain_weights": weights,
                "budget_external_force": ext_force[:, None, :],
                "constraint_target": np.ones((self.nb, 1)),
                "constraint_value": volume[:, None],
                "cumulative_mass_source": np.zeros((1,)),
                "element_deformation_gradient": F,
            },
            scalars={"constraint_scale": 1.0, "reference_momentum": 1.0},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.current_time - 1e-12:
            raise ValueError("advance_to time must be nondecreasing")
        while self.current_time + 1e-12 < target:
            h = min(0.005, target - self.current_time)
            self._step(h)
            self.current_time += h
        self._last_frame = self._make_frame(target, self.last_ext)
        return self._last_frame

    def diagnostics(self) -> dict[str, object]:
        return {"integrator": "semi_implicit_euler_penalty_contact", "bodies": int(self.nb),
                "vertices": int(self.n), "time": float(self.current_time)}
