"""Discrete elastic rod solver for the telephone-cord perversion scene.

The rod is modeled as a chain of inextensible edges with bending and twisting
elasticity derived from a perverted natural rest shape: two mirror-image
helices of opposite handedness joined by a localized perversion.  Endpoint
positions and orientations are prescribed according to the quasi-static
schedule in the frozen scaffold.  The dynamics are integrated with a
damped, semi-implicit time step and a position-based inextensibility
projection.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        rod_cfg = self.config["rod"]
        self.n = int(rod_cfg["vertex_count"])
        self.total_length = float(rod_cfg["total_length"])
        self.rod_mass = float(rod_cfg["mass"])
        self.width = float(rod_cfg["cross_section"]["width"])
        self.thickness = float(rod_cfg["cross_section"]["thickness"])
        self.helix_radius = float(rod_cfg["natural_shape"]["radius"])
        self.pitch = float(rod_cfg["natural_shape"]["pitch_per_turn"])
        self.gravity_vec = np.array(
            self.config.get("gravity", [0.0, 0.0, 0.0]), dtype=float
        )

        frame0 = scene.initial_state()
        self.x = np.asarray(frame0.position, dtype=float).copy()
        self.v = np.asarray(frame0.velocity, dtype=float).copy()
        if frame0.orientation is not None:
            self.q = np.asarray(frame0.orientation, dtype=float).copy()
        else:
            self.q = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n, 1))
        if frame0.angular_velocity is not None:
            self.omega = np.asarray(frame0.angular_velocity, dtype=float).copy()
        else:
            self.omega = np.zeros_like(self.x)
        if frame0.mass is not None:
            self.mass = np.asarray(frame0.mass, dtype=float).copy()
        else:
            self.mass = np.full(self.n, self.rod_mass / self.n)
        self.attributes = {
            k: np.asarray(v, dtype=float).copy()
            for k, v in frame0.attributes.items()
        }

        # Perverted natural rest shape: two opposite-handed helices meeting at a
        # perversion.  Encode the handedness through the material-curvature sign.
        s = np.linspace(0.0, 1.0, self.n)
        # Smooth sign transition over a few vertices at the centre.
        raw_sign = np.where(s < 0.5, -1.0, 1.0)
        transition = np.clip((s - 0.5) * self.n / 3.0, -1.0, 1.0)
        self.helix_sign = raw_sign * (0.5 + 0.5 * np.sin(0.5 * np.pi * transition))

        # Helix curvature of the source spring.
        h = self.pitch / (2.0 * np.pi)
        denom = self.helix_radius ** 2 + h ** 2
        kappa = self.helix_radius / denom

        # Edge rest length (uniform because the rest shape is length preserving).
        self.rest_edge_len = self.total_length / (self.n - 1)

        # Target curvature binormal per vertex, in material coordinates.
        # Material y-axis is the principal bending direction of the strip.
        self.rest_K = np.zeros((self.n, 3))
        self.rest_K[:, 1] = kappa * self.rest_edge_len * self.helix_sign

        # Deterministic, tiny transverse seed consistent with the perverted
        # rest curvature.  This breaks the symmetry of the initially flat rod
        # so that the two halves buckle into opposite-handed helices.
        seed_amp = 0.012
        self.x[:, 1] += seed_amp * self.helix_sign * np.sin(np.pi * s)

        # Time stepping and material parameters.
        self.t = 0.0
        self.k_bend = 0.2
        self.k_twist = 5.0
        self.c_damp = 30.0
        self.c_twist_damp = 5.0
        self.dt_internal = 2.0e-4
        self.pbd_iters = 5

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _curvature_binormal(a, b):
        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        D = la * lb + np.dot(a, b)
        if D < 1e-12:
            return np.zeros(3)
        return 2.0 * np.cross(a, b) / D

    @staticmethod
    def _curvature_jacobians(a, b):
        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        D = la * lb + np.dot(a, b)
        if D < 1e-12:
            Z = np.zeros((3, 3))
            return Z, Z
        u = np.cross(a, b)
        gradD_a = lb / la * a + b
        gradD_b = la / lb * b + a
        sx_b = np.array(
            [[0.0, -b[2], b[1]], [b[2], 0.0, -b[0]], [-b[1], b[0], 0.0]]
        )
        sx_a = np.array(
            [[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]]
        )
        J_a = -2.0 / D * sx_b - 2.0 / D ** 2 * np.outer(u, gradD_a)
        J_b = 2.0 / D * sx_a - 2.0 / D ** 2 * np.outer(u, gradD_b)
        return J_a, J_b

    @staticmethod
    def _rotate_vector(v, q):
        qv = q[:3]
        qw = q[3]
        return v + 2.0 * np.cross(qv, np.cross(qv, v) + qw * v)

    @staticmethod
    def _quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ]
        )

    @staticmethod
    def _quat_conjugate(q):
        return np.array([-q[0], -q[1], -q[2], q[3]])

    # ------------------------------------------------------------- time stepping
    def _endpoint_targets(self, t):
        return self.scene.endpoint_targets(t)

    def _step(self, dt, t):
        targets = self._endpoint_targets(t)
        left_pos = np.array(targets["left"]["position"], dtype=float)
        right_pos = np.array(targets["right"]["position"], dtype=float)

        f = np.zeros_like(self.x)
        # gravity
        f += self.gravity_vec[None, :] * self.mass[:, None]

        # Bending forces from discrete curvature binormal energy.
        for i in range(1, self.n - 1):
            a = self.x[i] - self.x[i - 1]
            b = self.x[i + 1] - self.x[i]
            K = self._curvature_binormal(a, b)
            target_K = self._rotate_vector(self.rest_K[i], self.q[i])
            dK = K - target_K
            J_a, J_b = self._curvature_jacobians(a, b)
            Fa = -self.k_bend * (J_a.T @ dK)
            Fb = -self.k_bend * (J_b.T @ dK)
            f[i - 1] += Fa
            f[i] += -(Fa + Fb)
            f[i + 1] += Fb

        # Semi-implicit Euler with linear damping.
        self.v += dt * (f / self.mass[:, None] - self.c_damp * self.v)
        self.x += dt * self.v

        # Inextensibility projection (position-based dynamics).
        for _ in range(self.pbd_iters):
            e = self.x[1:] - self.x[:-1]
            len_e = np.linalg.norm(e, axis=1)
            dir_e = e / len_e[:, None]
            delta = (len_e - self.rest_edge_len)[:, None] * dir_e
            self.x[:-1] -= 0.5 * delta
            self.x[1:] += 0.5 * delta

        # Enforce kinematic endpoint constraints.
        self.x[0] = left_pos
        self.x[-1] = right_pos
        self.v[0] = 0.0
        self.v[-1] = 0.0

        # Orientation dynamics: damped spring to the identity material frame.
        # The perversion is encoded in the rest curvature, not in a global twist.
        axis = -self.q[:, :3] * np.sign(self.q[:, 3:4])
        self.omega += dt * (self.k_twist * axis - self.c_twist_damp * self.omega)
        self.omega[0] = 0.0
        self.omega[-1] = 0.0

        for i in range(self.n):
            wq = np.array(
                [self.omega[i, 0], self.omega[i, 1], self.omega[i, 2], 0.0]
            )
            self.q[i] += dt * 0.5 * self._quat_multiply(wq, self.q[i])
            norm = np.linalg.norm(self.q[i])
            if norm > 1e-12:
                self.q[i] /= norm
        self.q[0] = np.array([0.0, 0.0, 0.0, 1.0])
        self.q[-1] = np.array([0.0, 0.0, 0.0, 1.0])

    def advance_to(self, time: float):
        if time <= self.t:
            return self._make_frame(time)
        n_sub = max(1, int(np.ceil((time - self.t) / self.dt_internal)))
        dt = (time - self.t) / n_sub
        for k in range(n_sub):
            self._step(dt, self.t + (k + 0.5) * dt)
        self.t = time
        return self._make_frame(time)

    def _make_frame(self, time):
        targets = self.scene.endpoint_targets(self.t)
        left_pos = np.array(targets["left"]["position"], dtype=float)
        right_pos = np.array(targets["right"]["position"], dtype=float)

        constraint_target = np.vstack([left_pos, right_pos])
        constraint_value = np.vstack([self.x[0], self.x[-1]])

        G, K = 1, 1
        budget_external_force = np.zeros((G, K, 3), dtype=float)
        budget_external_couple = np.zeros((G, K, 3), dtype=float)
        budget_force_position = np.zeros((G, K, 3), dtype=float)

        total_gravity = np.sum(
            self.gravity_vec[None, :] * self.mass[:, None], axis=0
        )
        budget_external_force[0, 0] = total_gravity
        budget_force_position[0, 0] = np.mean(self.x, axis=0)

        fields = {
            "budget_external_force": budget_external_force,
            "budget_external_couple": budget_external_couple,
            "budget_force_position": budget_force_position,
            "constraint_target": constraint_target,
            "constraint_value": constraint_value,
            "cumulative_mass_source": np.zeros(K, dtype=float),
        }

        scalars = {
            "constraint_scale": 1.0,
            "reference_force": 1.0,
            "reference_torque": 1.0,
        }

        attributes = dict(self.attributes)
        attributes["mass_quantity"] = self.mass.copy()

        return StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=self.q.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "discrete_elastic_rod", "seed": self.seed}
