"""Kinematic bicycle-chain rigid-body solver.

Implements a planar 2.5-D penalty rigid-body simulation for a closed chain
of 20 links driven by two rotating sprockets.  Sprocket axles are held fixed,
links are modelled as small rigid disks connected by stiff springs, and the
sprocket-link coupling uses a viscous tangential drive plus a unilateral
radial repulsion to prevent tooth penetration.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = seed
        self.config = getattr(scene, "config", {})
        self.meta = scene.metadata()

        self.gravity = np.asarray(
            self.meta.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        )
        self.contact_length_scale = float(
            self.config.get("contact", {}).get("length_scale", 0.02)
        )

        frame0 = scene.initial_state()
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        if frame0.orientation is not None:
            self.orientation = np.asarray(frame0.orientation, dtype=float).copy()
        else:
            self.orientation = np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (len(self.position), 1)
            )
        if frame0.angular_velocity is not None:
            self.angular_velocity = np.asarray(frame0.angular_velocity, dtype=float).copy()
        else:
            self.angular_velocity = np.zeros_like(self.position)
        self.mass = np.asarray(frame0.mass, dtype=float)
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32)
        self.N = len(self.position)

        # body groups
        self.chain = np.arange(20, dtype=np.int32)
        self.sprockets = np.array([20, 21], dtype=np.int32)
        self.sprocket_center = self.position[self.sprockets].copy()
        self.sprocket_angle = np.zeros(2, dtype=float)
        self.sprocket_omega = np.zeros(2, dtype=float)

        # geometry / physical parameters
        self.R_pitch = 0.50  # sprocket pitch radius (m)
        self.r_link = 0.08   # equivalent link disk radius (m)
        self.r_sprocket = self.R_pitch + self.r_link
        self.I_sprocket = 0.50  # moment of inertia about axle
        self.omega_target = 2.0   # driven sprocket target angular speed (rad/s)
        self.drive_kp = 5.0
        self.drive_kd = 0.5

        # link-link rest lengths from the frozen initial configuration
        p = self.position[self.chain]
        diffs = np.roll(p, -1, axis=0) - p
        self.rest_length = np.linalg.norm(diffs, axis=1)

        # spring/damper coefficients
        self.k_link = 5.0e4
        self.c_link = 50.0
        self.k_radial = 2.0e4
        self.c_radial = 20.0
        self.k_tan = 2.0e3
        self.sprocket_damp = 0.1

        self.sample_dt = float(self.config.get("sample_dt", 0.02))
        self.n_substeps = max(1, int(round(self.sample_dt / 1.0e-4)))

        self.current_time = 0.0
        self._last_frame = self._make_frame(0.0)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _quat_y(self, angle: float) -> np.ndarray:
        return np.array([0.0, np.sin(angle * 0.5), 0.0, np.cos(angle * 0.5)], dtype=float)

    def _normal_xz(self, vec: np.ndarray) -> np.ndarray:
        n = np.array([vec[0], 0.0, vec[2]], dtype=float)
        d = np.linalg.norm(n)
        if d < 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return n / d

    def _make_frame(self, t: float) -> StateFrame:
        # orient chain links along their local tangent
        orient = self.orientation.copy()
        p = self.position[self.chain]
        for i in range(20):
            prev = (i - 1) % 20
            nxt = (i + 1) % 20
            v = p[nxt] - p[prev]
            angle = np.arctan2(v[2], v[0])
            orient[self.chain[i]] = self._quat_y(angle)

        # sprocket orientations from integrated angle
        for k, idx in enumerate(self.sprockets):
            orient[idx] = self._quat_y(self.sprocket_angle[k])

        # angular velocity samples
        ang = self.angular_velocity.copy()
        for k, idx in enumerate(self.sprockets):
            ang[idx, 1] = self.sprocket_omega[k]

        ext, inter, dist = self._compute_fields()

        frame = StateFrame(
            time=float(t),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=orient,
            angular_velocity=ang,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"release_time": np.zeros(self.N, dtype=float)},
            fields={
                "external_force": ext,
                "interaction_force": inter,
                "contact_signed_distance": dist,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        return frame

    def _compute_fields(self):
        """Return (external_force[N,3], interaction_force[K,3], signed_distance[C])."""
        ext = np.zeros((self.N, 3), dtype=float)
        for i in range(self.N):
            ext[i] = self.mass[i] * self.gravity

        n_pairs = 20 + 40  # link-link + link-sprocket
        inter = np.zeros((n_pairs, 3), dtype=float)
        dists = np.zeros(n_pairs, dtype=float)

        # ---- link-link pairs ----
        for i in range(20):
            j = (i + 1) % 20
            idx_i, idx_j = self.chain[i], self.chain[j]
            r_ij = self.position[idx_j] - self.position[idx_i]
            d = np.linalg.norm(r_ij)
            if d < 1e-12:
                d = 1e-12
            n = r_ij / d
            # spring + damping on first body
            f = self.k_link * (d - self.rest_length[i]) * n
            v_rel = self.velocity[idx_j] - self.velocity[idx_i]
            f += self.c_link * np.dot(v_rel, n) * n
            inter[i] = f
            dists[i] = d - 2.0 * self.r_link

        # ---- link-sprocket pairs ----
        pair_idx = 20
        for s in range(2):
            center = self.sprocket_center[s]
            for i in range(20):
                idx = self.chain[i]
                r_vec = self.position[idx] - center
                n = self._normal_xz(r_vec)
                d = np.linalg.norm(np.array([r_vec[0], 0.0, r_vec[2]]))

                f = np.zeros(3, dtype=float)
                # radial repulsion (unilateral contact)
                if d < self.r_sprocket:
                    penetration = self.r_sprocket - d
                    f_rad_mag = self.k_radial * penetration
                    v = self.velocity[idx]  # sprocket translation is zero
                    v_n = np.dot(v, n)
                    f += (f_rad_mag - self.c_radial * v_n) * n
                    dists[pair_idx] = d - self.r_sprocket
                else:
                    dists[pair_idx] = d - self.r_sprocket

                # tangential viscous drive
                t = np.array([-n[2], 0.0, n[0]], dtype=float)
                v_link_tan = np.dot(self.velocity[idx], t)
                v_surf = self.sprocket_omega[s] * self.R_pitch
                if d < self.r_sprocket + 0.15:
                    f += self.k_tan * (v_surf - v_link_tan) * t

                inter[pair_idx] = f
                pair_idx += 1

        return ext, inter, dists

    def _apply_forces_and_step(self, dt: float) -> None:
        ext, inter, _ = self._compute_fields()
        del ext

        # link-link
        for i in range(20):
            j = (i + 1) % 20
            idx_i, idx_j = self.chain[i], self.chain[j]
            f = inter[i]
            self.velocity[idx_i] += (f / self.mass[idx_i]) * dt
            self.velocity[idx_j] -= (f / self.mass[idx_j]) * dt

        # link-sprocket: update link velocities and sprocket angular velocities
        pair_idx = 20
        torques = np.zeros(2, dtype=float)
        for s in range(2):
            for i in range(20):
                idx = self.chain[i]
                f = inter[pair_idx]
                # link linear acceleration
                self.velocity[idx] += (f / self.mass[idx]) * dt
                # torque on sprocket from reaction force -f at pitch radius
                r = self.R_pitch * self._normal_xz(self.position[idx] - self.sprocket_center[s])
                torques[s] += -r[2] * f[0] + r[0] * f[2]
                pair_idx += 1

        # integrate sprocket angular velocities
        for s in range(2):
            self.sprocket_omega[s] += (torques[s] / self.I_sprocket) * dt

        # prescribed drive on the right-hand sprocket (local s=1)
        s = 1
        drive_torque = (
            self.drive_kp * (self.omega_target - self.sprocket_omega[s])
            - self.drive_kd * self.sprocket_omega[s]
        )
        self.sprocket_omega[s] += (drive_torque / self.I_sprocket) * dt

        # light angular damping on both sprockets
        self.sprocket_omega -= self.sprocket_damp * self.sprocket_omega * dt

        # integrate positions / orientations
        self.position[self.chain] += self.velocity[self.chain] * dt
        self.sprocket_angle += self.sprocket_omega * dt
        for k, idx in enumerate(self.sprockets):
            self.orientation[idx] = self._quat_y(self.sprocket_angle[k])

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time <= self.current_time:
            return self._last_frame

        dt_total = time - self.current_time
        dt = dt_total / self.n_substeps
        for _ in range(self.n_substeps):
            self._apply_forces_and_step(dt)

        self.current_time = time
        self._last_frame = self._make_frame(time)
        return self._last_frame

    def diagnostics(self) -> dict[str, object]:
        return {"seed": self.seed, "substeps": self.n_substeps}
