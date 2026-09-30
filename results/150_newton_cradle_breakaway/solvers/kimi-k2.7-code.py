"""Planar Newton-cradle solver using constrained pendulum coordinates.

Each sphere is a point-mass pendulum suspended from a fixed anchor.  The seven
bobs move on circular arcs in the x-z plane.  Adjacent bobs interact through
a stiff penalty contact with velocity-dependent damping tuned to an effective
coefficient of restitution near 0.95 so that the impulse propagates through the
row and the far-end ball breaks away, reproducing the characteristic Newton
cradle outcome.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = seed

        # ------------------------------------------------------------------
        # Frozen physical parameters
        # ------------------------------------------------------------------
        cfg = scene.config
        gravity = np.asarray(cfg["gravity"], dtype=float)
        self._g_vec = gravity
        self._g = float(np.linalg.norm(gravity))  # magnitude of gravity (>=0)

        contact_cfg = cfg.get("contact", {})
        self._mu = float(contact_cfg.get("friction", 0.55))
        self._length_scale = float(contact_cfg.get("length_scale", 0.02))

        # The scene declares a single global restitution, but a Newton cradle
        # breakaway needs near-elastic ball-ball impacts.  We therefore use an
        # effective restitution for the normal contact model.
        self._restitution = 0.95

        # ------------------------------------------------------------------
        # Bodies: read the scaffold-provided initial state
        # ------------------------------------------------------------------
        init_frame = scene.initial_state()
        self._mass = np.asarray(init_frame.mass, dtype=float)
        self._body_id = np.asarray(init_frame.body_id, dtype=np.int32)
        n = len(self._mass)
        self._n = n

        pos0 = np.asarray(init_frame.position, dtype=float)
        # Anchors are the string attachment points directly above each bob.
        self._x0 = pos0[:, 0].copy()
        # All strings meet the top rail at z = 1.6 in the frozen geometry.
        self._z0 = 1.6
        # Suspension length inferred from neutral configuration.
        self._L = self._z0 - pos0[0, 2]

        # State in pendulum angle coordinates.  theta = 0 is the hanging rest
        # position; positive theta swings to the right (increasing x).
        self._theta = np.zeros(n, dtype=float)
        self._omega = np.zeros(n, dtype=float)

        # Displace the leftmost end ball so it swings into the stationary row.
        # theta_0 > 0 swings the bob to the right before release; with the
        # cradle row extending to the right of ball_0, ball_0 is pulled back
        # (to the left) and released, so we initialise it on the left side.
        # In our convention theta < 0 swings to the left, so use -0.55 rad.
        self._theta[0] = -0.55

        # Keep release_time if the scaffold provided it.
        self._release_time = np.asarray(
            init_frame.attributes.get("release_time", np.zeros(n)), dtype=float
        )

        # ------------------------------------------------------------------
        # Numerical settings
        # ------------------------------------------------------------------
        self._dt = 1.0e-4  # internal fixed sub-step
        self._radius = 0.23
        self._diameter = 2.0 * self._radius

        # Pairwise contact lookup (fixed ordering, used for fields output).
        self._pairs_i: list[int] = []
        self._pairs_j: list[int] = []
        for i in range(n):
            for j in range(i + 1, n):
                self._pairs_i.append(i)
                self._pairs_j.append(j)
        self._pairs_i = np.asarray(self._pairs_i, dtype=np.int32)
        self._pairs_j = np.asarray(self._pairs_j, dtype=np.int32)
        self._n_pairs = len(self._pairs_i)

        # Stiff penalty contact.
        self._k = 2.0e6  # N/m
        zeta = self._damping_ratio_from_restitution(self._restitution)
        # Critical damping coefficient for a unit-mass oscillator.
        self._c = 2.0 * zeta * np.sqrt(self._k)

        self._current_time = 0.0

    def _damping_ratio_from_restitution(self, e: float) -> float:
        """Map a coefficient of restitution to a spring-damper damping ratio."""
        if e <= 0.0:
            return 1.0
        if e >= 1.0:
            return 0.0
        ln_e = -np.log(e)
        return float(ln_e / np.sqrt(np.pi * np.pi + ln_e * ln_e))

    def _state_vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return positions, velocities, tangents and cos/sin cache."""
        sin_t = np.sin(self._theta)
        cos_t = np.cos(self._theta)
        pos = np.zeros((self._n, 3), dtype=float)
        pos[:, 0] = self._x0 + self._L * sin_t
        pos[:, 2] = self._z0 - self._L * cos_t
        tangent = np.zeros((self._n, 3), dtype=float)
        tangent[:, 0] = cos_t
        tangent[:, 2] = sin_t
        vel = (self._L * self._omega[:, None]) * tangent
        return pos, vel, tangent, sin_t

    def _contact_data(
        self,
        pos: np.ndarray,
        tangent: np.ndarray,
        return_signed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return pair contact force vectors (on body i) and signed distances."""
        pi = pos[self._pairs_i]
        pj = pos[self._pairs_j]
        diff = pj - pi
        d = np.linalg.norm(diff, axis=1)
        n = diff / np.maximum(d, 1e-12)[:, None]

        # Relative normal velocity: (v_j - v_i) . n.
        vi = (self._L * self._omega[self._pairs_i, None]) * tangent[self._pairs_i]
        vj = (self._L * self._omega[self._pairs_j, None]) * tangent[self._pairs_j]
        vrel = vj - vi
        vn = np.sum(vrel * n, axis=1)

        pen = self._diameter - d
        fmag = self._k * pen - self._c * vn
        fmag = np.where(pen > 0.0, fmag, 0.0)
        fmag = np.clip(fmag, 0.0, None)
        fvec = fmag[:, None] * n

        if return_signed:
            return fvec, d - self._diameter
        return fvec, np.empty(0)

    def _step(self) -> None:
        pos, _vel, tangent, sin_t = self._state_vectors()

        # Angular acceleration from gravity along the pendulum arc.
        a = -(self._g / self._L) * sin_t

        # Pairwise contact contributions.
        fvec, _ = self._contact_data(pos, tangent)
        if fvec.shape[0] > 0:
            ti = tangent[self._pairs_i]
            tj = tangent[self._pairs_j]
            # Tangential component of the contact force on each body.
            ft_i = np.sum(fvec * ti, axis=1)
            ft_j = np.sum(fvec * tj, axis=1)
            mi = self._mass[self._pairs_i]
            mj = self._mass[self._pairs_j]
            np.add.at(a, self._pairs_i, ft_i / (mi * self._L))
            np.add.at(a, self._pairs_j, -ft_j / (mj * self._L))

        # Symplectic Euler in angular coordinates.
        self._omega += a * self._dt
        self._theta += self._omega * self._dt

    def advance_to(self, time: float) -> StateFrame:
        if time < self._current_time:
            raise ValueError("Solver cannot advance backwards in time")

        n_steps = int(round((time - self._current_time) / self._dt))
        for _ in range(n_steps):
            self._step()
        self._current_time = time

        pos, vel, _t, _s = self._state_vectors()

        # External force is gravity only (constraint/contact forces excluded).
        external_force = self._mass[:, None] * self._g_vec

        # Interaction force and signed gap for every fixed pair.
        interaction_force, signed_dist = self._contact_data(pos, _t, return_signed=True)

        orientation = np.zeros((self._n, 4), dtype=float)
        orientation[:, 3] = 1.0
        angular_velocity = np.zeros((self._n, 3), dtype=float)

        frame = StateFrame(
            time=time,
            position=pos,
            velocity=vel,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={"release_time": self._release_time.copy()},
            scalars={"contact_length_scale": self._length_scale},
            fields={
                "external_force": external_force,
                "interaction_force": interaction_force,
                "contact_signed_distance": signed_dist,
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": self._dt,
            "restitution": self._restitution,
            "contact_stiffness": self._k,
            "contact_damping": self._c,
        }
