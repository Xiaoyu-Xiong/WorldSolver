"""Baseline penalty-based rigid contact solver for the rings-over-poles scene.

The dynamic object ``ring_collection`` is represented by 24 rigid-body samples.
Each sample is evolved with a stable, explicit time-stepping scheme under gravity
and penalty contact with the ground, the nine poles, and the other rings.
"""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import Scene, StateFrame


class BenchmarkSolver:
    """Penalty contact solver for nonconvex rings dropped onto vertical poles."""

    # Physics constants
    GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)

    # Effective contact radius of a ring sample (m).  Chosen to keep the
    # staggered initial configuration free of large overlaps while still being
    # large enough to interact with the poles and with one another.
    RING_RADIUS = 0.15

    # Contact stiffness and damping for the penalty springs.
    CONTACT_STIFFNESS = 8.0e4
    CONTACT_DAMPING = 1.2e3

    # Small amount of global velocity damping so the pile settles.
    GLOBAL_DAMPING = 0.8

    # Internal substep size for stable explicit integration.
    SUBSTEP = 0.002

    # Geometry helpers
    GROUND_Z = 0.0
    POLE_HEIGHT = 2.6

    def __init__(self) -> None:
        self._initialized = False
        self._time = 0.0
        self._positions: np.ndarray | None = None
        self._velocities: np.ndarray | None = None
        self._masses: np.ndarray | None = None
        self._body_id: np.ndarray | None = None
        self._orientation: np.ndarray | None = None
        self._angular_velocity: np.ndarray | None = None
        self._poles: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Solver protocol
    # ------------------------------------------------------------------
    def initialize(self, scene: Scene, seed: int) -> None:
        del seed  # deterministic after the scaffold's initial state

        initial = scene.initial_state()
        if initial.position is None or initial.velocity is None:
            raise ValueError("Scene initial state missing position/velocity")

        self._positions = np.asarray(initial.position, dtype=float).copy()
        self._velocities = np.asarray(initial.velocity, dtype=float).copy()
        n = self._positions.shape[0]

        if self._velocities.shape != self._positions.shape:
            raise ValueError("Initial velocity shape does not match position shape")

        if initial.mass is not None:
            self._masses = np.asarray(initial.mass, dtype=float).copy()
        else:
            self._masses = np.ones(n, dtype=float)

        if initial.body_id is not None:
            self._body_id = np.asarray(initial.body_id, dtype=int).copy()
        else:
            self._body_id = np.arange(n, dtype=int)

        if initial.orientation is not None:
            self._orientation = np.asarray(initial.orientation, dtype=float).copy()
        else:
            self._orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=float), (n, 1))

        if initial.angular_velocity is not None:
            self._angular_velocity = np.asarray(initial.angular_velocity, dtype=float).copy()
        else:
            self._angular_velocity = np.zeros((n, 3), dtype=float)

        self._time = float(initial.time)
        self._build_poles()
        self._initialized = True

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("Solver has not been initialized")
        if time < self._time - 1e-12:
            raise ValueError("Requested time is earlier than current simulation time")

        target = float(time)
        while self._time < target - 1e-12:
            dt = min(self.SUBSTEP, target - self._time)
            self._step(dt)
            self._time += dt

        signed_distances, interaction_forces = self._contact_data()

        return StateFrame(
            time=self._time,
            position=self._positions.copy(),
            velocity=self._velocities.copy(),
            orientation=self._orientation.copy(),
            angular_velocity=self._angular_velocity.copy(),
            mass=self._masses.copy(),
            body_id=self._body_id.copy(),
            scalars={"contact_length_scale": 0.06},
            fields={
                "external_force": self._external_force(),
                "interaction_force": interaction_forces,
                "contact_signed_distance": signed_distances,
            },
        )

    def diagnostics(self) -> dict:
        return {
            "solver": "BenchmarkSolver",
            "substep": self.SUBSTEP,
            "sample_count": len(self._positions) if self._positions is not None else 0,
        }

    # ------------------------------------------------------------------
    # Scene geometry
    # ------------------------------------------------------------------
    def _build_poles(self) -> None:
        """The nine poles form a regular 3x3 grid at z=0..2.6."""
        coords = np.array(
            [
                [-0.9, -0.9],
                [0.0, -0.9],
                [0.9, -0.9],
                [-0.9, 0.0],
                [0.0, 0.0],
                [0.9, 0.0],
                [-0.9, 0.9],
                [0.0, 0.9],
                [0.9, 0.9],
            ],
            dtype=float,
        )
        # pole format: [x, y, radius]
        self._poles = np.concatenate(
            [coords, np.full((coords.shape[0], 1), 0.075, dtype=float)],
            axis=1,
        )

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        pos = self._positions
        vel = self._velocities
        n = pos.shape[0]

        # Gravity is the only external body force.
        forces = self._masses[:, None] * self.GRAVITY[None, :]

        # Accumulate contact forces.
        forces = self._ring_ground_forces(pos, vel, forces)
        forces = self._ring_pole_forces(pos, vel, forces)
        forces = self._ring_ring_forces(pos, vel, forces)

        # Global velocity damping to help the pile settle.
        forces -= self.GLOBAL_DAMPING * vel

        # Semi-implicit (symplectic) Euler integration.
        accel = forces / self._masses[:, None]
        self._velocities = vel + dt * accel
        self._positions = pos + dt * self._velocities

    # ------------------------------------------------------------------
    # Contact force helpers
    # ------------------------------------------------------------------
    def _penalty_force(self, penetration: float, v_rel_n: float) -> float:
        """Return scalar repulsive force magnitude from overlap and relative speed."""
        if penetration <= 0.0:
            return 0.0
        f = self.CONTACT_STIFFNESS * penetration - self.CONTACT_DAMPING * v_rel_n
        return max(f, 0.0)

    def _ring_ground_forces(self, pos: np.ndarray, vel: np.ndarray, forces: np.ndarray) -> np.ndarray:
        sd = pos[:, 2] - self.RING_RADIUS
        penetrating = sd < 0.0
        if not np.any(penetrating):
            return forces
        penetration = -sd[penetrating]
        v_rel_n = vel[penetrating, 2]
        f_mag = self._penalty_force_vec(penetration, v_rel_n)
        forces[penetrating, 2] += f_mag
        return forces

    def _ring_pole_forces(self, pos: np.ndarray, vel: np.ndarray, forces: np.ndarray) -> np.ndarray:
        for px, py, pr in self._poles:
            dx = pos[:, 0] - px
            dy = pos[:, 1] - py
            horiz = np.hypot(dx, dy)
            # Guard against samples exactly on the pole axis.
            horiz_safe = np.where(horiz < 1e-9, 1.0, horiz)
            sd = horiz - (pr + self.RING_RADIUS)

            # Only side contacts within the pole's vertical extent are considered.
            in_height = (pos[:, 2] >= 0.0) & (pos[:, 2] <= self.POLE_HEIGHT)
            penetrating = in_height & (sd < 0.0)
            if not np.any(penetrating):
                continue

            penetration = -sd[penetrating]
            nx = dx[penetrating] / horiz_safe[penetrating]
            ny = dy[penetrating] / horiz_safe[penetrating]
            normal = np.column_stack([nx, ny, np.zeros_like(nx)])
            v_rel_n = np.einsum("ij,ij->i", vel[penetrating], normal)
            f_mag = self._penalty_force_vec(penetration, v_rel_n)
            forces[penetrating] += f_mag[:, None] * normal
        return forces

    def _ring_ring_forces(self, pos: np.ndarray, vel: np.ndarray, forces: np.ndarray) -> np.ndarray:
        n = pos.shape[0]
        # Pairwise differences.
        diff = pos[:, None, :] - pos[None, :, :]  # (n, n, 3)
        d2 = np.sum(diff * diff, axis=2)
        d = np.sqrt(d2)
        d_safe = np.where(d < 1e-9, 1.0, d)

        threshold = 2.0 * self.RING_RADIUS
        # Strict upper triangle to avoid double counting.
        mask = (d < threshold) & (np.triu(np.ones((n, n), dtype=bool), k=1))
        i, j = np.where(mask)
        if i.size == 0:
            return forces

        normal = diff[i, j] / d_safe[i, j][:, None]
        penetration = threshold - d[i, j]
        v_rel = vel[i] - vel[j]
        v_rel_n = np.einsum("ij,ij->i", v_rel, normal)
        f_mag = self._penalty_force_vec(penetration, v_rel_n)
        f_vec = f_mag[:, None] * normal

        # Apply equal and opposite forces.
        forces[i] += f_vec
        forces[j] -= f_vec
        return forces

    def _penalty_force_vec(self, penetration: np.ndarray, v_rel_n: np.ndarray) -> np.ndarray:
        """Vectorized scalar repulsive force magnitude."""
        f = self.CONTACT_STIFFNESS * penetration - self.CONTACT_DAMPING * v_rel_n
        return np.clip(f, 0.0, None)

    # ------------------------------------------------------------------
    # Trajectory field extraction
    # ------------------------------------------------------------------
    def _external_force(self) -> np.ndarray:
        """Gravity force on every dynamic sample."""
        return self._masses[:, None] * self.GRAVITY[None, :]

    def _contact_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Return signed distances and interaction force for all active contacts."""
        pos = self._positions
        vel = self._velocities
        sd_list = []
        force_list = []

        # Ground contacts.
        sd_ground = pos[:, 2] - self.RING_RADIUS
        ground_active = sd_ground < 0.0
        if np.any(ground_active):
            penetration = -sd_ground[ground_active]
            v_rel_n = vel[ground_active, 2]
            f_mag = self._penalty_force_vec(penetration, v_rel_n)
            for i, sd_i, f in zip(np.where(ground_active)[0], sd_ground[ground_active], f_mag):
                if f <= 0.0:
                    continue
                sd_list.append(float(sd_i))
                force_list.append([0.0, 0.0, float(f)])

        # Pole contacts.
        for px, py, pr in self._poles:
            dx = pos[:, 0] - px
            dy = pos[:, 1] - py
            horiz = np.hypot(dx, dy)
            horiz_safe = np.where(horiz < 1e-9, 1.0, horiz)
            sd = horiz - (pr + self.RING_RADIUS)
            in_height = (pos[:, 2] >= 0.0) & (pos[:, 2] <= self.POLE_HEIGHT)
            active = in_height & (sd < 0.0)
            if not np.any(active):
                continue
            idx = np.where(active)[0]
            nx = dx[active] / horiz_safe[active]
            ny = dy[active] / horiz_safe[active]
            normal = np.column_stack([nx, ny, np.zeros_like(nx)])
            penetration = -sd[active]
            v_rel_n = np.einsum("ij,ij->i", vel[active], normal)
            f_mag = self._penalty_force_vec(penetration, v_rel_n)
            for i, sd_i, f, nvec in zip(idx, sd[active], f_mag, normal):
                if f <= 0.0:
                    continue
                sd_list.append(float(sd_i))
                force_list.append((f * nvec).tolist())

        # Ring-ring contacts.
        n = pos.shape[0]
        diff = pos[:, None, :] - pos[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        d = np.sqrt(d2)
        d_safe = np.where(d < 1e-9, 1.0, d)
        threshold = 2.0 * self.RING_RADIUS
        mask = (d < threshold) & (np.triu(np.ones((n, n), dtype=bool), k=1))
        i, j = np.where(mask)
        if i.size > 0:
            normal = diff[i, j] / d_safe[i, j][:, None]
            penetration = threshold - d[i, j]
            v_rel = vel[i] - vel[j]
            v_rel_n = np.einsum("ij,ij->i", v_rel, normal)
            f_mag = self._penalty_force_vec(penetration, v_rel_n)
            for a, b, sd_i, f, nvec in zip(i, j, -penetration, f_mag, normal):
                if f <= 0.0:
                    continue
                sd_list.append(float(sd_i))
                # Report the force acting on body a.
                force_list.append((f * nvec).tolist())

        if not sd_list:
            return np.zeros((0,), dtype=float), np.zeros((0, 3), dtype=float)
        return np.asarray(sd_list, dtype=float), np.asarray(force_list, dtype=float)
