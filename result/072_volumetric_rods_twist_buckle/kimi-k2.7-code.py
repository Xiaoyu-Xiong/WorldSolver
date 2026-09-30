"""Explicit elastic rod-bundle solver for the twist-buckling scene.

The seven volumetric rods are modelled as mass-spring networks built from the
surface mesh.  End regions are driven by rigid rotations about the x-axis.
Contact between rods (and self contact) is handled by a penalty repulsion with
pairwise damping.  Time integration is a damped symplectic Euler substepping
scheme, so the solver actually steps physics forward in time and returns a
finite state at every requested sample time.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self._scene: object | None = None
        self._seed = 0
        self._rest: np.ndarray | None = None
        self._pos: np.ndarray | None = None
        self._vel: np.ndarray | None = None
        self._mass: np.ndarray | None = None
        self._body_id: np.ndarray | None = None
        self._driven_left: np.ndarray | None = None
        self._driven_right: np.ndarray | None = None
        self._time = 0.0

        # Mesh topology (per rod)
        self._faces: list[tuple[int, int, int]] = []
        self._edges: np.ndarray | None = None
        self._edge_rest: np.ndarray | None = None
        self._edge_stiff: np.ndarray | None = None

        # Element (triangle) bookkeeping
        self._triangles: np.ndarray | None = None
        self._n_elements = 0

        # Per-vertex offsets for the 7 rods
        self._offsets: list[int] = []
        self._n_local = 0

        # Budget bookkeeping
        self._cum_ext_impulse: np.ndarray | None = None
        self._cum_in: np.ndarray | None = None
        self._cum_out: np.ndarray | None = None
        self._domain_weights: np.ndarray | None = None

        # Physical parameters
        self._young = 2.0e5
        self._contact_stiff = 8.0e4
        self._contact_damp = 2.0e2
        self._damping = 25.0
        self._dt = 0.001
        self._gravity = np.array([0.0, 0.0, -0.1])

        # Derived geometry
        self._rod_radius = 0.0
        self._reference_momentum = 0.0

    def initialize(self, scene: object, seed: int) -> None:
        self._scene = scene
        self._seed = seed

        frame0 = scene.initial_state()
        self._rest = np.asarray(frame0.position, dtype=float)
        self._pos = self._rest.copy()
        self._vel = np.asarray(frame0.velocity, dtype=float)
        self._mass = np.asarray(frame0.mass, dtype=float)
        self._body_id = np.asarray(frame0.body_id, dtype=np.int32)
        self._driven_left = np.asarray(frame0.attributes["driven_left"], dtype=bool)
        self._driven_right = np.asarray(frame0.attributes["driven_right"], dtype=bool)
        self._time = 0.0

        g = scene.config["geometry"]
        self._rod_radius = float(g["rod_radius"])

        self._build_topology(int(g["axial_segments"]), int(g["radial_segments"]))

        n_bodies = int(np.max(self._body_id)) + 1
        self._cum_ext_impulse = np.zeros((n_bodies, 3), dtype=float)
        self._cum_in = np.zeros((n_bodies, 3), dtype=float)
        self._cum_out = np.zeros((n_bodies, 3), dtype=float)

        # Domain weights: one row per rod, one-hot membership.
        n = len(self._pos)
        self._domain_weights = np.zeros((n_bodies, n), dtype=float)
        for b in range(n_bodies):
            mask = self._body_id == b
            self._domain_weights[b, mask] = 1.0

        # Reference momentum scale.
        omega = np.deg2rad(72.0)  # rad/s
        ring_r = float(g["bundle_ring_radius"])
        total_mass = float(self._mass.sum())
        self._reference_momentum = total_mass * ring_r * omega

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "explicit_mass_spring_with_penalty_contact",
            "substep_dt": self._dt,
            "sample_dt": self._scene.config["sample_dt"],
            "bodies": int(self._domain_weights.shape[0]),
            "samples": len(self._pos) if self._pos is not None else 0,
            "elements": self._n_elements,
        }

    def _build_topology(self, nx: int, nt: int) -> None:
        n_local = (nx + 1) * nt + 2
        self._n_local = n_local
        faces: list[tuple[int, int, int]] = []
        for i in range(nx):
            for j in range(nt):
                a = i * nt + j
                b = i * nt + (j + 1) % nt
                c = (i + 1) * nt + j
                d = (i + 1) * nt + (j + 1) % nt
                faces.append((a, c, d))
                faces.append((a, d, b))
        left = n_local - 2
        right = n_local - 1
        base = nx * nt
        for j in range(nt):
            faces.append((left, (j + 1) % nt, j))
            faces.append((right, base + j, base + (j + 1) % nt))

        edge_set: set[tuple[int, int]] = set()
        for a, b, c in faces:
            edge_set.add((min(a, b), max(a, b)))
            edge_set.add((min(a, c), max(a, c)))
            edge_set.add((min(b, c), max(b, c)))

        # Cross-links through the rod cross-section for bulk rigidity.
        for i in range(nx + 1):
            for j in range(nt // 2):
                a = i * nt + j
                b = i * nt + (j + nt // 2) % nt
                edge_set.add((min(a, b), max(a, b)))

        self._faces = faces
        n_rods = int(np.max(self._body_id)) + 1
        self._offsets = [b * n_local for b in range(n_rods)]

        edges_local = np.asarray(sorted(edge_set), dtype=np.int32)
        # Broadcast to all rods.
        edges = []
        for off in self._offsets:
            edges.append(edges_local + off)
        edges_arr = np.vstack(edges)
        self._edges = edges_arr

        rest = self._rest
        dX = rest[edges_arr[:, 1]] - rest[edges_arr[:, 0]]
        self._edge_rest = np.linalg.norm(dX, axis=1)

        # Per-edge stiffness: E * A / l0 with cross-sectional area A = pi r^2.
        area = np.pi * self._rod_radius ** 2
        self._edge_stiff = self._young * area / self._edge_rest

        # Element list per rod.
        tri_local = np.asarray(faces, dtype=np.int32)
        tri_all = []
        for off in self._offsets:
            tri_all.append(tri_local + off)
        self._triangles = np.vstack(tri_all)
        self._n_elements = len(self._triangles)

    def _rotation_x(self, theta: float) -> np.ndarray:
        c = np.cos(theta)
        s = np.sin(theta)
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])

    def _drive_target(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return target position/velocity for left and right driven vertices at time t."""
        theta_l, theta_r = self._scene.end_rotation_radians(t)
        omega_l = np.deg2rad(self._scene.config["boundary_conditions"]["left_end_angular_speed_deg_s"])
        omega_r = np.deg2rad(self._scene.config["boundary_conditions"]["right_end_angular_speed_deg_s"])

        R_l = self._rotation_x(theta_l)
        R_r = self._rotation_x(theta_r)

        rest = self._rest
        pos_l = (R_l @ rest[self._driven_left].T).T
        pos_r = (R_r @ rest[self._driven_right].T).T

        # Velocity = omega e_x × x.
        vel_l = np.zeros_like(pos_l)
        vel_l[:, 1] = -omega_l * pos_l[:, 2]
        vel_l[:, 2] = omega_l * pos_l[:, 1]
        vel_r = np.zeros_like(pos_r)
        vel_r[:, 1] = -omega_r * pos_r[:, 2]
        vel_r[:, 2] = omega_r * pos_r[:, 1]

        return pos_l, vel_l, pos_r, vel_r

    def _internal_forces(self) -> np.ndarray:
        pos = self._pos
        i = self._edges[:, 0]
        j = self._edges[:, 1]
        dx = pos[j] - pos[i]
        dist = np.linalg.norm(dx, axis=1, keepdims=True)
        # Avoid zero division.
        safe = np.maximum(dist, 1e-12)
        strain = (dist - self._edge_rest[:, None]) / self._edge_rest[:, None]
        force_mag = self._edge_stiff[:, None] * (dist - self._edge_rest[:, None])
        f = force_mag * dx / safe
        forces = np.zeros_like(pos)
        np.add.at(forces, i, f)
        np.add.at(forces, j, -f)
        return forces

    def _contact_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute penalty contact forces and per-body net contact force."""
        pos = self._pos
        tree = cKDTree(pos)
        pairs = tree.query_pairs(self._rod_radius * 2.0, output_type="ndarray")

        forces = np.zeros_like(pos)
        if len(pairs) == 0:
            body_contact = np.zeros((self._domain_weights.shape[0], 3), dtype=float)
            return forces, body_contact

        i = pairs[:, 0]
        j = pairs[:, 1]
        dx = pos[j] - pos[i]
        dist = np.linalg.norm(dx, axis=1)
        mask = dist > 1e-12
        i = i[mask]
        j = j[mask]
        dx = dx[mask]
        dist = dist[mask]

        # Self-contact filter: within the same rod only keep pairs whose rest
        # separation is large (looped contact), not adjacent mesh vertices.
        same_body = self._body_id[i] == self._body_id[j]
        rest_sep = np.linalg.norm(self._rest[i] - self._rest[j], axis=1)
        keep = (~same_body) | (rest_sep > 2.5 * self._rod_radius)
        i = i[keep]
        j = j[keep]
        dx = dx[keep]
        dist = dist[keep]

        if len(i) == 0:
            body_contact = np.zeros((self._domain_weights.shape[0], 3), dtype=float)
            return forces, body_contact

        n = dx / dist[:, None]
        overlap = (self._rod_radius * 2.0 - dist)[:, None]
        rel_v = self._vel[j] - self._vel[i]
        normal_v = np.sum(rel_v * n, axis=1)[:, None]
        f_mag = self._contact_stiff * overlap - self._contact_damp * normal_v
        f_mag = np.clip(f_mag, 0.0, None)
        f = f_mag * n

        np.add.at(forces, i, -f)
        np.add.at(forces, j, f)

        # Per-body net contact force.
        n_bodies = self._domain_weights.shape[0]
        body_contact = np.zeros((n_bodies, 3), dtype=float)
        for b in range(n_bodies):
            mask_b = self._body_id[i] == b
            if np.any(mask_b):
                body_contact[b] -= f[mask_b].sum(axis=0)
            mask_b = self._body_id[j] == b
            if np.any(mask_b):
                body_contact[b] += f[mask_b].sum(axis=0)
        return forces, body_contact

    def _element_deformation_gradient(self) -> np.ndarray:
        tri = self._triangles
        X = self._rest
        x = self._pos
        X0 = X[tri[:, 0]]
        X1 = X[tri[:, 1]]
        X2 = X[tri[:, 2]]
        x0 = x[tri[:, 0]]
        x1 = x[tri[:, 1]]
        x2 = x[tri[:, 2]]

        dX1 = X1 - X0
        dX2 = X2 - X0
        dx1 = x1 - x0
        dx2 = x2 - x0

        nX = np.cross(dX1, dX2)
        nx = np.cross(dx1, dx2)
        len_nX = np.linalg.norm(nX, axis=1)
        len_nx = np.linalg.norm(nx, axis=1)
        nX = nX / np.maximum(len_nX, 1e-12)[:, None]
        nx = nx / np.maximum(len_nx, 1e-12)[:, None]

        A = np.stack([dX1, dX2, nX], axis=2)  # (Ne,3,3)
        B = np.stack([dx1, dx2, nx], axis=2)
        A_inv = np.linalg.inv(A)
        F = B @ A_inv
        return F

    def _substep(self) -> None:
        pos = self._pos
        vel = self._vel
        mass = self._mass
        h = self._dt

        # External and internal forces.
        f_int = self._internal_forces()
        f_contact, body_contact = self._contact_forces()

        # Gravity (very light, just keeps the bundle from drifting vertically).
        f_grav = mass[:, None] * self._gravity

        # Global velocity damping.
        f_damp = -self._damping * vel

        f_total = f_int + f_contact + f_grav + f_damp
        acc = f_total / mass[:, None]

        # Free velocity update.
        vel_free = vel + h * acc

        # Drive target for the boundary regions.
        pos_l, vel_l, pos_r, vel_r = self._drive_target(self._time + h)

        # Constraint impulse = m*(v_target - v_free) for driven samples.
        left_idx = np.nonzero(self._driven_left)[0]
        right_idx = np.nonzero(self._driven_right)[0]
        impulse_left = mass[left_idx, None] * (vel_l - vel_free[left_idx])
        impulse_right = mass[right_idx, None] * (vel_r - vel_free[right_idx])

        f_constraint = np.zeros_like(f_total)
        f_constraint[left_idx] = impulse_left / h
        f_constraint[right_idx] = impulse_right / h

        # Apply projection.
        vel_new = vel_free.copy()
        pos_new = pos + h * vel_new
        vel_new[left_idx] = vel_l
        vel_new[right_idx] = vel_r
        pos_new[left_idx] = pos_l
        pos_new[right_idx] = pos_r

        # Budget update: external force on each body = contact + boundary constraint.
        n_bodies = self._domain_weights.shape[0]
        ext_force = body_contact.copy()
        for b in range(n_bodies):
            mask = self._body_id == b
            ext_force[b] += f_constraint[mask].sum(axis=0)

        self._cum_ext_impulse += h * ext_force

        self._pos = pos_new
        self._vel = vel_new
        self._time += h

    def _make_frame(self) -> StateFrame:
        # Deformation gradient for each surface triangle.
        F = self._element_deformation_gradient()

        n_bodies = self._domain_weights.shape[0]

        fields = {
            "element_deformation_gradient": F,
            "budget_cumulative_external_impulse": self._cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": self._cum_in.copy(),
            "budget_cumulative_outflow_momentum": self._cum_out.copy(),
            "budget_domain_weights": self._domain_weights.copy(),
            "budget_external_force": np.zeros((n_bodies, 1, 3), dtype=float),
            "cumulative_mass_source": np.zeros((1,), dtype=float),
        }

        frame = StateFrame(
            time=self._time,
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={
                "mass_quantity": self._mass.copy(),
                "rest_position": self._rest.copy(),
                "driven_left": self._driven_left.astype(np.int32),
                "driven_right": self._driven_right.astype(np.int32),
            },
            fields=fields,
            scalars={"reference_momentum": float(self._reference_momentum)},
        )
        return frame

    def advance_to(self, time: float) -> StateFrame:
        if time < self._time:
            raise ValueError("advance_to must not move backwards in time")
        target = float(time)
        # Substep forward until the requested sample time.
        while self._time + self._dt / 2.0 < target:
            self._substep()
        # Snap to the exact sample time with a final smaller substep.
        if target > self._time:
            saved_dt = self._dt
            self._dt = target - self._time
            self._substep()
            self._dt = saved_dt

        frame = self._make_frame()
        frame.validate()
        return frame
