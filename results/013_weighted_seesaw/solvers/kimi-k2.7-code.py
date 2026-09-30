"""Solver Agent-owned implementation for the weighted seesaw scene.

Implements a 2D rigid-body dynamics simulation in the X-Z plane (symmetric about Y)
using an explicit penalty-contact time stepper.  The seesaw board is modelled as
a rigid body pinned to the cylindrical pivot (centred at z = 0.70 m) so that it
rotates about the pivot.  The two blocks are full 2D rigid bodies that interact
with the board, the ground, and the pivot through spring-damper contact forces
with Coulomb friction.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = seed

        # Physics constants
        self.g = 9.81
        # Contact parameters (SI units)
        self.k_contact = 5.0e4          # contact stiffness [N/m]
        self.c_damp = 5.0e2             # normal damping [N/(m/s)]
        self.mu = 0.30                  # dynamic friction coefficient

        # Time stepping
        self.dt = 5.0e-4                # internal step [s]
        self.current_time = 0.0

        # Initial state from the scene contract (3 bodies: board, heavy, light)
        frame = scene.initial_state()
        self.mass = np.asarray(frame.mass, dtype=float).copy()
        self.body_id = np.asarray(frame.body_id, dtype=int).copy()

        # 2D half extents in the X-Z simulation plane for each body
        self.half = np.array(
            [
                [1.80, 0.07],   # board
                [0.28, 0.28],   # heavy block
                [0.22, 0.22],   # light block
            ],
            dtype=float,
        )

        # Moments of inertia about the out-of-plane (Y) axis.
        # For the board the pivot is the fixed centre of rotation, so use the
        # parallel-axis theorem.  For the blocks use the centre-of-mass axis.
        self.I = np.zeros(3, dtype=float)
        for i in range(1, 3):
            self.I[i] = self.mass[i] / 3.0 * (self.half[i, 0] ** 2 + self.half[i, 1] ** 2)
        # Board I about its own CM, then add m*d^2 with d = distance to pivot.
        self.I[0] = self.mass[0] / 3.0 * (self.half[0, 0] ** 2 + self.half[0, 1] ** 2)
        self.I[0] += self.mass[0] * self.r_board ** 2

        # The board pivots about the top of the static cylinder at (0, 0.70 m).
        self.pivot = np.array([0.0, 0.70])
        self.r_board = 0.08              # distance from pivot to board centre

        # Board state: angle theta (positive => right end goes down) and angular
        # velocity omega, both in the clockwise convention.
        self.theta = np.zeros(3, dtype=float)
        self.omega = np.zeros(3, dtype=float)

        # Block positions and velocities in the X-Z plane.
        self.pos = np.zeros((3, 2), dtype=float)
        self.vel = np.zeros((3, 2), dtype=float)

        # Map the 3-D initial state into the 2-D simulation plane.
        position = np.asarray(frame.position, dtype=float)
        velocity = np.asarray(frame.velocity, dtype=float)
        self.pos[:, 0] = position[:, 0]
        self.pos[:, 1] = position[:, 2]
        self.vel[:, 0] = velocity[:, 0]
        self.vel[:, 1] = velocity[:, 2]

        # The board position is kinematically determined by theta, so keep its
        # stored 2-D position equal to the derived centre for output convenience.
        self.pos[0] = self._board_centre(0.0)

    # --------------------------------------------------------------------- #
    # Geometry helpers
    # --------------------------------------------------------------------- #
    def _board_centre(self, theta: float) -> np.ndarray:
        """Return board centre in world X-Z for a given board angle."""
        return self.pivot + self.r_board * np.array([np.sin(theta), np.cos(theta)])

    @staticmethod
    def _axes(theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Return local X and Z unit vectors in world X-Z for a clockwise angle."""
        c = np.cos(theta)
        s = np.sin(theta)
        ux = np.array([c, -s])
        uz = np.array([s, c])
        return ux, uz

    @staticmethod
    def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
        """Scalar z-component of the cross product in the X-Z plane."""
        return a[0] * b[1] - a[1] * b[0]

    def _point_velocity(self, body: int, p: np.ndarray) -> np.ndarray:
        """Velocity of point p attached to body ``body``."""
        if body == 0:
            # Board rotates about the pivot.
            return self.omega[0] * np.array(
                [p[1] - self.pivot[1], -(p[0] - self.pivot[0])]
            )
        v = self.vel[body].copy()
        omega = self.omega[body]
        if abs(omega) > 0.0:
            r = p - self.pos[body]
            v += omega * np.array([r[1], -r[0]])
        return v

    # --------------------------------------------------------------------- #
    # Contact detection (signed distance + normal + point)
    # --------------------------------------------------------------------- #
    def _rect_rect(
        self,
        c1: np.ndarray,
        h1: np.ndarray,
        ux1: np.ndarray,
        uz1: np.ndarray,
        c2: np.ndarray,
        h2: np.ndarray,
        ux2: np.ndarray,
        uz2: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Signed distance, normal (body2 -> body1), and contact point for two rectangles."""
        axes = [ux1, uz1, ux2, uz2]
        best_gap = -1e20
        best_axis = axes[0]

        for a in axes:
            # Projection radii onto axis a (taking absolute dot products).
            r1 = h1[0] * abs(np.dot(ux1, a)) + h1[1] * abs(np.dot(uz1, a))
            r2 = h2[0] * abs(np.dot(ux2, a)) + h2[1] * abs(np.dot(uz2, a))
            d = np.dot(c2 - c1, a)

            # Gap when a points from body1 toward body2.
            gap = d - r1 - r2
            if gap > best_gap:
                best_gap = gap
                best_axis = a

            # Gap when a points from body1 away from body2.
            gap2 = -d - r1 - r2
            if gap2 > best_gap:
                best_gap = gap2
                best_axis = a

        # Choose normal direction so it points from body2 toward body1.
        n = best_axis
        if np.dot(c1 - c2, n) < 0.0:
            n = -n

        # Approximate contact point as midpoint of the two centres.
        p = 0.5 * (c1 + c2)
        return float(best_gap), n, p

    def _rect_circle(
        self,
        c_rect: np.ndarray,
        h: np.ndarray,
        ux: np.ndarray,
        uz: np.ndarray,
        c_circ: np.ndarray,
        radius: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Signed distance, normal (circle -> rectangle), and contact point."""
        d = c_circ - c_rect
        x_l = np.dot(d, ux)
        z_l = np.dot(d, uz)
        cx = max(-h[0], min(h[0], x_l))
        cz = max(-h[1], min(h[1], z_l))
        closest = c_rect + cx * ux + cz * uz
        delta = c_circ - closest
        dist = float(np.linalg.norm(delta))
        if dist < 1e-12:
            # Degenerate: circle centre inside rectangle.
            n = np.array([0.0, 1.0])
            signed = -radius
        else:
            n = delta / dist
            signed = dist - radius
        # n currently points rectangle -> circle.  We need circle -> rectangle.
        n = -n
        return signed, n, closest

    def _rect_plane(
        self,
        c: np.ndarray,
        h: np.ndarray,
        ux: np.ndarray,
        uz: np.ndarray,
        plane_z: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Signed distance of rectangle above horizontal plane z = plane_z."""
        # Minimum z over the four vertices.
        min_z = c[1]
        for sx, sz in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            v = c + sx * h[0] * ux + sz * h[1] * uz
            min_z = min(min_z, v[1])
        signed = min_z - plane_z
        n = np.array([0.0, 1.0])
        # Contact point: average of the lowest vertices.
        low_vertices = []
        for sx, sz in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            v = c + sx * h[0] * ux + sz * h[1] * uz
            if abs(v[1] - min_z) < 1e-9:
                low_vertices.append(v)
        p = np.mean(low_vertices, axis=0) if low_vertices else c.copy()
        return signed, n, p

    # --------------------------------------------------------------------- #
    # Time stepper
    # --------------------------------------------------------------------- #
    def _contacts(self) -> list[tuple[int, int, float, np.ndarray, np.ndarray]]:
        """Return all candidate contact pairs as (i, j, signed_dist, normal, point)."""
        c0 = self._board_centre(self.theta[0])
        ux0, uz0 = self._axes(self.theta[0])
        ux1, uz1 = self._axes(self.theta[1])
        ux2, uz2 = self._axes(self.theta[2])

        contacts = []

        # Board / blocks vs ground (z=0).
        sd, n, p = self._rect_plane(c0, self.half[0], ux0, uz0, 0.0)
        contacts.append((0, -1, sd, n, p))
        sd, n, p = self._rect_plane(self.pos[1], self.half[1], ux1, uz1, 0.0)
        contacts.append((1, -1, sd, n, p))
        sd, n, p = self._rect_plane(self.pos[2], self.half[2], ux2, uz2, 0.0)
        contacts.append((2, -1, sd, n, p))

        # Board vs blocks.
        sd, n, p = self._rect_rect(
            c0, self.half[0], ux0, uz0,
            self.pos[1], self.half[1], ux1, uz1,
        )
        contacts.append((0, 1, sd, n, p))
        sd, n, p = self._rect_rect(
            c0, self.half[0], ux0, uz0,
            self.pos[2], self.half[2], ux2, uz2,
        )
        contacts.append((0, 2, sd, n, p))

        # Heavy vs light block.
        sd, n, p = self._rect_rect(
            self.pos[1], self.half[1], ux1, uz1,
            self.pos[2], self.half[2], ux2, uz2,
        )
        contacts.append((1, 2, sd, n, p))

        # Blocks vs pivot (cylinder, modelled as a circle in X-Z of radius 0.22 m).
        sd, n, p = self._rect_circle(
            self.pos[1], self.half[1], ux1, uz1, self.pivot, 0.22
        )
        contacts.append((1, -2, sd, n, p))
        sd, n, p = self._rect_circle(
            self.pos[2], self.half[2], ux2, uz2, self.pivot, 0.22
        )
        contacts.append((2, -2, sd, n, p))

        return contacts

    def _step(self, h: float) -> None:
        """Advance one internal sub-step with explicit symplectic Euler."""
        c0 = self._board_centre(self.theta[0])
        ux0, uz0 = self._axes(self.theta[0])

        # Accumulated forces and torques.
        F = np.zeros((3, 2), dtype=float)
        tau = np.zeros(3, dtype=float)

        # Gravity on each body.
        for i in range(3):
            F[i, 1] -= self.mass[i] * self.g

        # Gravity torque on the board about the pivot.
        tau[0] += self.mass[0] * self.g * c0[0]

        contacts = self._contacts()
        for i, j, signed, n, p in contacts:
            if signed >= 0.0:
                continue
            depth = -signed

            vi = self._point_velocity(i, p)
            if j >= 0:
                vj = self._point_velocity(j, p)
            else:
                vj = np.zeros(2, dtype=float)
            vrel = vi - vj
            vn = float(np.dot(vrel, n))

            # Spring-damper normal force (magnitude, positive means repulsion).
            f_mag = self.k_contact * depth - self.c_damp * vn
            if f_mag <= 0.0:
                continue

            Fn_vec = f_mag * n

            # Coulomb friction in the tangent plane.
            vt = vrel - vn * n
            vtmag = float(np.linalg.norm(vt))
            if vtmag > 1e-9:
                Ft_vec = -self.mu * f_mag * vt / vtmag
            else:
                Ft_vec = np.zeros(2, dtype=float)

            Ftotal = Fn_vec + Ft_vec

            # Apply to dynamic bodies.
            F[i] += Ftotal
            if j >= 0:
                F[j] -= Ftotal

            # Torques.
            if i == 0:
                tau[0] += -self._cross2d(p - self.pivot, Ftotal)
            else:
                tau[i] += -self._cross2d(p - self.pos[i], Ftotal)

            if j == 0:
                tau[0] += self._cross2d(p - self.pivot, Ftotal)
            elif j >= 0:
                tau[j] += self._cross2d(p - self.pos[j], Ftotal)

        # Semi-implicit Euler update.
        self.vel += h * F / self.mass[:, None]
        self.omega += h * tau / self.I

        self.pos += h * self.vel
        self.theta += h * self.omega

        # Keep the stored board position consistent with the integrated angle.
        self.pos[0] = self._board_centre(self.theta[0])

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.current_time < target - 1e-12:
            h = min(self.dt, target - self.current_time)
            self._step(h)
            self.current_time += h
        return self._build_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {}

    # --------------------------------------------------------------------- #
    # Output formatting
    # --------------------------------------------------------------------- #
    def _build_frame(self, t: float) -> StateFrame:
        c0 = self._board_centre(self.theta[0])

        positions = np.zeros((3, 3), dtype=float)
        velocities = np.zeros((3, 3), dtype=float)

        # Board.
        positions[0] = [c0[0], 0.0, c0[1]]
        v_board = self.omega[0] * np.array([c0[1] - self.pivot[1], -c0[0]])
        velocities[0] = [v_board[0], 0.0, v_board[1]]

        # Blocks.
        positions[1] = [self.pos[1, 0], 0.0, self.pos[1, 1]]
        velocities[1] = [self.vel[1, 0], 0.0, self.vel[1, 1]]
        positions[2] = [self.pos[2, 0], 0.0, self.pos[2, 1]]
        velocities[2] = [self.vel[2, 0], 0.0, self.vel[2, 1]]

        # Orientation: clockwise rotation in X-Z is a negative rotation about Y.
        orientation = np.zeros((3, 4), dtype=float)
        orientation[:, 1] = -np.sin(self.theta / 2.0)
        orientation[:, 3] = np.cos(self.theta / 2.0)

        # Angular velocity: (0, -omega, 0).
        angular_velocity = np.zeros((3, 3), dtype=float)
        angular_velocity[:, 1] = -self.omega

        # External force: gravity only.
        external_force = np.zeros((3, 3), dtype=float)
        external_force[:, 2] = -self.mass * self.g

        # Constraint: board angle relative to the rest (horizontal) angle.
        constraint_value = np.array([float(self.theta[0])])
        constraint_target = np.array([0.0])

        # Signed distances for all candidate contact pairs.
        contacts = self._contacts()
        contact_signed_distance = np.array([c[2] for c in contacts], dtype=float)

        fields = {
            "external_force": external_force,
            "contact_signed_distance": contact_signed_distance,
            "constraint_value": constraint_value,
            "constraint_target": constraint_target,
        }
        scalars = {
            "constraint_scale": 1.0,
            "contact_length_scale": 0.1,
        }

        return StateFrame(
            time=float(t),
            position=positions,
            velocity=velocities,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields=fields,
            scalars=scalars,
        )
