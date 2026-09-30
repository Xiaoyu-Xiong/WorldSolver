from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Penalty-based rigid contact solver for the billiard impulse chain.

    Six identical spheres rest on a horizontal felt table.  An incoming sphere
    strikes the near end of the line; repulsive contact forces transmit the
    impulse through the chain and launch the far sphere.  Gravity, rolling
    resistance, and penalty contact are integrated with a stable explicit
    sub-stepping scheme.
    """

    def __init__(self) -> None:
        self.scene = None
        self.seed = 0
        self.t = 0.0

        self.pos = None
        self.vel = None
        self.mass = None
        self.body_id = None

        # Frozen physical parameters
        self.radius = 0.2
        self.g = np.array([0.0, 0.0, -9.81], dtype=float)
        self.rolling_friction = 0.015
        self.restitution = 0.95
        self.contact_stiffness = 1.0e5
        self.contact_damping = 1.0e2
        self.substeps = 200
        self.contact_margin = 0.05

        self._diag: dict = {}

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)

        state = scene.initial_state()

        self.t = float(state.time)
        self.pos = np.asarray(state.position, dtype=float).copy()
        self.vel = np.asarray(state.velocity, dtype=float).copy()
        self.mass = np.asarray(state.mass, dtype=float).copy()
        self.body_id = np.asarray(state.body_id, dtype=int).copy()

        if self.mass.ndim == 0 or self.mass.shape[0] != self.pos.shape[0]:
            self.mass = np.full(
                self.pos.shape[0],
                float(self.mass) if np.isscalar(self.mass) else 1.0,
                dtype=float,
            )

        self._diag = {
            "solver": "BenchmarkSolver",
            "seed": self.seed,
            "bodies": int(self.pos.shape[0]),
        }

    def advance_to(self, time: float) -> StateFrame:
        dt = float(time) - self.t
        if dt > 0.0:
            n = max(1, self.substeps)
            h = dt / n
            for _ in range(n):
                self._step(h)
            self.t = float(time)
        return self._make_frame(self.t)

    def diagnostics(self) -> dict:
        return self._diag.copy()

    # ------------------------------------------------------------------
    # Internal physics
    # ------------------------------------------------------------------
    def _external_force(self) -> np.ndarray:
        """Gravity plus rolling resistance; excludes contact forces."""
        n = self.pos.shape[0]
        F = np.zeros((n, 3), dtype=float)
        F += self.mass[:, None] * self.g

        for i in range(n):
            v_xy = self.vel[i, :2]
            speed = float(np.linalg.norm(v_xy))
            if speed > 1.0e-8:
                decel = self.rolling_friction * 9.81
                F[i, :2] -= self.mass[i] * decel * (v_xy / speed)

        return F

    def _contact_forces(self) -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:
        """Compute contact forces, pairs, signed distances, and interaction force array."""
        n = self.pos.shape[0]
        F = np.zeros((n, 3), dtype=float)
        pairs: list[tuple[int, int]] = []
        sds: list[float] = []
        force_list: list[np.ndarray] = []

        # Sphere-sphere contacts
        for i in range(n):
            for j in range(i + 1, n):
                dvec = self.pos[j] - self.pos[i]
                dist = float(np.linalg.norm(dvec))
                if dist < 1.0e-12:
                    continue
                nvec = dvec / dist
                sd = dist - 2.0 * self.radius

                if sd < self.contact_margin:
                    sds.append(sd)
                    pairs.append((i, j))

                if sd < 0.0:
                    v_rel = self.vel[j] - self.vel[i]
                    v_n = float(np.dot(v_rel, nvec))
                    penetration = -sd
                    f_mag = self.contact_stiffness * penetration - self.contact_damping * v_n
                    f = f_mag * nvec

                    F[i] -= f
                    F[j] += f
                    force_list.append(f.copy())

        # Sphere-ground contacts
        ground_n = np.array([0.0, 0.0, 1.0], dtype=float)
        for i in range(n):
            sd = float(self.pos[i, 2]) - self.radius
            if sd < self.contact_margin:
                sds.append(sd)
                pairs.append((i, -1))

            if sd < 0.0:
                v_n = float(self.vel[i, 2])
                penetration = -sd
                f_mag = self.contact_stiffness * penetration - self.contact_damping * v_n
                f = f_mag * ground_n

                F[i] += f
                force_list.append(f.copy())

        signed = np.array(sds, dtype=float) if sds else np.zeros((0,), dtype=float)
        force_arr = (
            np.array(force_list, dtype=float)
            if force_list
            else np.zeros((0, 3), dtype=float)
        )

        return F, pairs, signed, force_arr

    def _step(self, h: float) -> None:
        F_ext = self._external_force()
        F_int, _, _, _ = self._contact_forces()
        F_total = F_ext + F_int

        self.vel += h * F_total / self.mass[:, None]
        self.pos += h * self.vel

        # Keep the spheres on the felt plane and apply restitution for any
        # small numerical bounce.
        for i in range(self.pos.shape[0]):
            if self.pos[i, 2] < self.radius:
                self.pos[i, 2] = self.radius
                if self.vel[i, 2] < 0.0:
                    self.vel[i, 2] *= -self.restitution

    def _rolling_angular_velocity(self) -> np.ndarray:
        """Kinematic rolling-without-slipping angular velocity."""
        n = self.pos.shape[0]
        omega = np.zeros((n, 3), dtype=float)
        for i in range(n):
            v_xy = self.vel[i, :2]
            omega[i, 0] = -v_xy[1] / self.radius
            omega[i, 1] = v_xy[0] / self.radius
        return omega

    def _make_frame(self, time: float) -> StateFrame:
        F_ext = self._external_force()
        _, _, signed, interaction = self._contact_forces()
        omega = self._rolling_angular_velocity()
        n = self.pos.shape[0]

        return StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (n, 1)),
            angular_velocity=omega,
            fields={
                "external_force": F_ext,
                "interaction_force": interaction,
                "contact_signed_distance": signed,
            },
            scalars={
                "contact_length_scale": self.radius,
            },
        )
