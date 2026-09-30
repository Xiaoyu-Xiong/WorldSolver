"""Simple mass-spring cloth solver for the knit-patch uniaxial stretch scene.

The solver implements explicit time stepping with stretching/shear/bending
springs, gravity, obstacle contact penalties, and kinematic grip constraints.
It exports the required raw trajectory fields for the WorldSolver platform.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physically-based cloth solver for a uniaxially stretched knit patch."""

    def __init__(self):
        self.scene = None
        self.seed = 0
        self.t = 0.0

        # state
        self.position: NDArray[np.float64] | None = None
        self.velocity: NDArray[np.float64] | None = None
        self.mass: NDArray[np.float64] | None = None
        self.body_id: NDArray[np.int32] | None = None
        self.thickness = 0.015
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # geometry / topology
        self.nx = 17
        self.ny = 17
        self.uv: NDArray[np.float64] | None = None
        self.rest_pos: NDArray[np.float64] | None = None
        self.left_grip_center = np.array([-1.5, 0.0, 2.15], dtype=np.float64)
        self.right_grip_center = np.array([1.5, 0.0, 2.15], dtype=np.float64)

        # edges: (i, j, rest_length)
        self.edge_i: NDArray[np.int64] | None = None
        self.edge_j: NDArray[np.int64] | None = None
        self.edge_l0: NDArray[np.float64] | None = None
        self.bend_i: NDArray[np.int64] | None = None
        self.bend_j: NDArray[np.int64] | None = None
        self.bend_l0: NDArray[np.float64] | None = None
        self.neighbors: list[list[int]] | None = None

        # masks
        self.left_grip_mask: NDArray[np.bool_] | None = None
        self.right_grip_mask: NDArray[np.bool_] | None = None
        self.free_mask: NDArray[np.bool_] | None = None

        # material parameters
        self.k_stretch = 10.0
        self.k_shear = 8.0
        self.k_bend = 2.0
        self.k_contact = 500.0
        self.damping = 3.0
        self.dt_max = 1e-4

        # motion profile for the moving grip
        self.t_stretch = 4.0
        self.max_displacement = 0.7

        # cached diagnostics
        self._diag = {"substeps": 0}

    # ------------------------------------------------------------------
    # Solver protocol
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        np.random.seed(self.seed)

        frame = scene.initial_state()
        self.t = float(frame.time)
        self.position = np.asarray(frame.position, dtype=np.float64).copy()
        self.velocity = np.asarray(frame.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(frame.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(frame.body_id, dtype=np.int32).copy()
        self.rest_pos = self.position.copy()

        cfg = scene.config
        cloth_cfg = cfg.get("cloth", {})
        self.nx, self.ny = map(int, cloth_cfg.get("grid_shape", [17, 17]))
        self.thickness = float(cloth_cfg.get("thickness", 0.015))

        gravity = cfg.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.asarray(gravity, dtype=np.float64)

        # material coordinates are the initial planar (x, y) positions
        self.uv = self.position[:, :2].copy()

        self._build_topology()
        self._build_masks()

        # initialize optional trajectory fields with sensible values
        F = self._compute_deformation_gradient(self.position)
        ext = self._external_forces(self.position, self.t)
        contact = self._contact_signed_distances(self.position)
        self.attributes = {
            "mass_quantity": self.mass.copy(),
            "boundary_vertex": np.asarray(
                frame.attributes.get("boundary_vertex", np.zeros(len(self.position), dtype=np.int32))
            ),
            "deformation_gradient": F,
        }
        self.fields = {
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "external_force": ext,
            "contact_signed_distance": contact,
        }
        self.scalars = {
            "contact_length_scale": self.thickness,
        }

    def advance_to(self, time: float) -> StateFrame:
        if self.position is None:
            raise RuntimeError("Solver not initialized")

        dt_total = time - self.t
        if dt_total < 0:
            raise ValueError("advance_to called with a time earlier than current time")

        n_sub = max(1, int(np.ceil(dt_total / self.dt_max)))
        h = dt_total / n_sub

        for _ in range(n_sub):
            self._step(h)
            self._diag["substeps"] += 1

        self.t = time

        # export frame with required fields
        F = self._compute_deformation_gradient(self.position)
        ext = self._external_forces(self.position, self.t)
        contact = self._contact_signed_distances(self.position)

        frame = StateFrame(
            time=time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.attributes["boundary_vertex"].copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": ext,
                "contact_signed_distance": contact,
            },
            scalars={
                "contact_length_scale": self.thickness,
            },
        )
        return frame

    def diagnostics(self) -> dict:
        return self._diag.copy()

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    def _step(self, h: float) -> None:
        pos = self.position
        vel = self.velocity
        n = pos.shape[0]

        # external forces (gravity + contact)
        f_ext = self._external_forces(pos, self.t)
        force = f_ext.copy()

        # stretch/shear spring forces
        if self.edge_i is not None:
            d = pos[self.edge_j] - pos[self.edge_i]
            dist = np.linalg.norm(d, axis=1)
            dist = np.where(dist < 1e-12, 1e-12, dist)
            strain = (dist - self.edge_l0) / self.edge_l0
            mag = self.k_stretch * strain
            f = mag[:, None] * (d / dist[:, None])
            np.add.at(force, self.edge_i, f)
            np.add.at(force, self.edge_j, -f)

        # bending springs (two-ring neighbours)
        if self.bend_i is not None:
            d = pos[self.bend_j] - pos[self.bend_i]
            dist = np.linalg.norm(d, axis=1)
            dist = np.where(dist < 1e-12, 1e-12, dist)
            strain = (dist - self.bend_l0) / self.bend_l0
            mag = self.k_bend * strain
            f = mag[:, None] * (d / dist[:, None])
            np.add.at(force, self.bend_i, f)
            np.add.at(force, self.bend_j, -f)

        # integrate
        acc = force / self.mass[:, None]
        vel_new = vel + acc * h
        vel_new *= 1.0 - self.damping * h
        pos_new = pos + vel_new * h

        # enforce grip constraints (kinematic)
        disp = self._grip_displacement(self.t + h)
        vel_right = np.array([self._grip_velocity(self.t + h), 0.0, 0.0], dtype=np.float64)

        pos_new[self.left_grip_mask] = self.rest_pos[self.left_grip_mask]
        pos_new[self.right_grip_mask] = self.rest_pos[self.right_grip_mask] + np.array([disp, 0.0, 0.0])
        vel_new[self.left_grip_mask] = 0.0
        vel_new[self.right_grip_mask] = vel_right

        self.position = pos_new
        self.velocity = vel_new
        self.t += h

    # ------------------------------------------------------------------
    # Geometry / topology
    # ------------------------------------------------------------------
    def _build_topology(self) -> None:
        nx, ny = self.nx, self.ny

        def idx(i, j):
            return i * ny + j

        stretch_i, stretch_j, stretch_l0 = [], [], []
        bend_i, bend_j, bend_l0 = [], [], []

        # horizontal and vertical edges + diagonal shear edges
        for i in range(nx):
            for j in range(ny):
                v = idx(i, j)
                # horizontal
                if i + 1 < nx:
                    w = idx(i + 1, j)
                    stretch_i.append(v)
                    stretch_j.append(w)
                    stretch_l0.append(self._rest_length(v, w))
                # vertical
                if j + 1 < ny:
                    w = idx(i, j + 1)
                    stretch_i.append(v)
                    stretch_j.append(w)
                    stretch_l0.append(self._rest_length(v, w))
                # diagonals for shear
                if i + 1 < nx and j + 1 < ny:
                    w1 = idx(i + 1, j + 1)
                    stretch_i.append(v)
                    stretch_j.append(w1)
                    stretch_l0.append(self._rest_length(v, w1))
                    w2 = idx(i + 1, j)
                    w3 = idx(i, j + 1)
                    stretch_i.append(w2)
                    stretch_j.append(w3)
                    stretch_l0.append(self._rest_length(w2, w3))

        # bending springs: second neighbours
        for i in range(nx):
            for j in range(ny):
                v = idx(i, j)
                if i + 2 < nx:
                    w = idx(i + 2, j)
                    bend_i.append(v)
                    bend_j.append(w)
                    bend_l0.append(self._rest_length(v, w))
                if j + 2 < ny:
                    w = idx(i, j + 2)
                    bend_i.append(v)
                    bend_j.append(w)
                    bend_l0.append(self._rest_length(v, w))

        self.edge_i = np.asarray(stretch_i, dtype=np.int64)
        self.edge_j = np.asarray(stretch_j, dtype=np.int64)
        self.edge_l0 = np.asarray(stretch_l0, dtype=np.float64)
        self.bend_i = np.asarray(bend_i, dtype=np.int64)
        self.bend_j = np.asarray(bend_j, dtype=np.int64)
        self.bend_l0 = np.asarray(bend_l0, dtype=np.float64)

        # one-ring neighbours for deformation-gradient computation
        n = nx * ny
        self.neighbors = [[] for _ in range(n)]
        for a, b in zip(self.edge_i, self.edge_j):
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)

    def _build_masks(self) -> None:
        x = self.rest_pos[:, 0]
        # select the outermost grid columns as the gripped edges
        self.left_grip_mask = np.abs(x - (-1.5)) < 1e-6
        self.right_grip_mask = np.abs(x - 1.5) < 1e-6
        self.free_mask = ~(self.left_grip_mask | self.right_grip_mask)

    def _rest_length(self, a: int, b: int) -> float:
        return float(np.linalg.norm(self.rest_pos[a] - self.rest_pos[b]))

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------
    def _grip_displacement(self, t: float) -> float:
        t = float(t)
        if t >= self.t_stretch:
            return self.max_displacement
        return self.max_displacement * 0.5 * (1.0 - np.cos(np.pi * t / self.t_stretch))

    def _grip_velocity(self, t: float) -> float:
        t = float(t)
        if t >= self.t_stretch:
            return 0.0
        return (
            self.max_displacement
            * 0.5
            * np.sin(np.pi * t / self.t_stretch)
            * (np.pi / self.t_stretch)
        )

    def _external_forces(self, pos: NDArray[np.float64], t: float) -> NDArray[np.float64]:
        force = self.mass[:, None] * self.gravity[None, :]

        # ground contact
        ground_d = self._box_signed_distance(pos, np.array([0.0, 0.0, -0.05]), np.array([2.4, 2.0, 0.05]))
        ground_pen = np.maximum(-ground_d, 0.0)
        force[:, 2] += self.k_contact * ground_pen * 0.5 * self.thickness

        # left grip (fixed)
        left_d = self._box_signed_distance(pos, self.left_grip_center, np.array([0.12, 1.25, 0.12]))
        force = self._add_contact_force(force, pos, self.left_grip_center, left_d)

        # right grip (moving)
        right_center = self.right_grip_center.copy()
        right_center[0] += self._grip_displacement(t)
        right_d = self._box_signed_distance(pos, right_center, np.array([0.12, 1.25, 0.12]))
        force = self._add_contact_force(force, pos, right_center, right_d)

        return force

    def _add_contact_force(
        self,
        force: NDArray[np.float64],
        pos: NDArray[np.float64],
        center: NDArray[np.float64],
        signed_dist: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        pen = np.maximum(-signed_dist, 0.0)
        normal = pos - center[None, :]
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        norm = np.where(norm < 1e-12, 1.0, norm)
        normal = normal / norm
        force += self.k_contact * pen[:, None] * normal
        return force

    def _contact_signed_distances(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        ground_d = self._box_signed_distance(pos, np.array([0.0, 0.0, -0.05]), np.array([2.4, 2.0, 0.05]))
        left_d = self._box_signed_distance(pos, self.left_grip_center, np.array([0.12, 1.25, 0.12]))
        right_center = self.right_grip_center.copy()
        right_center[0] += self._grip_displacement(self.t)
        right_d = self._box_signed_distance(pos, right_center, np.array([0.12, 1.25, 0.12]))
        return np.concatenate([ground_d, left_d, right_d])

    @staticmethod
    def _box_signed_distance(
        p: NDArray[np.float64], c: NDArray[np.float64], h: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Signed distance from points p to an axis-aligned box with centre c and half-extents h."""
        delta = np.abs(p - c[None, :]) - h[None, :]
        outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
        inside = np.min(delta, axis=1)
        return outside + inside

    def _compute_deformation_gradient(self, pos: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute a per-vertex 2x2 deformation gradient in material coordinates."""
        n = pos.shape[0]
        F = np.zeros((n, 2, 2), dtype=np.float64)
        for i in range(n):
            neigh = self.neighbors[i]
            if len(neigh) < 2:
                F[i] = np.eye(2)
                continue
            dU = self.uv[neigh] - self.uv[i]  # (m, 2)
            dX = pos[neigh, :2] - pos[i, :2]  # (m, 2)
            # least-squares fit dX = F @ dU  =>  F = dX * pinv(dU)
            F_i = dX.T @ np.linalg.pinv(dU.T)
            F[i] = F_i
        return F
