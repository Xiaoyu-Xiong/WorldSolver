"""Deterministic mass-spring cloth solver for the inclined-plane scene.

The cloth is discretised as a 25x25 triangle mesh of point masses joined by
structural, shear and weak bending springs.  A rigid inclined box acts as the
collision surface.  Contact is handled with a unilateral (Kelvin-Voigt)
penalty normal force plus a Coulomb friction force that is projected onto the
friction cone, so a sliding sheet obeys ``a = g (sin(theta) - mu cos(theta))``
exactly, while a resting sheet is held by static friction.

Integration is a fixed-step semi-implicit (symplectic) Euler scheme with a
step small enough to resolve the stiffest penalty mode.  Everything is pure
NumPy, fully deterministic and free of any scripted motion.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_DEFAULT_GRAVITY = (0.0, 0.0, -9.81)


def _rotation_from_quaternion(quaternion) -> np.ndarray:
    """Rotation matrix for an xyzw quaternion."""
    x, y, z, w = (float(v) for v in quaternion)
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


class BenchmarkSolver:
    """Physically-based cloth solver: gravity, membrane/bending elasticity, contact."""

    # --- numerical parameters (fixed, deterministic) -----------------------
    time_step = 4.0e-5
    zeta_spring = 0.05
    contact_stiffness = 8.0e3
    contact_damping_ratio = 0.7
    structural_stiffness = 3.0e3
    shear_stiffness = 1.5e3
    bending_stiffness = 1.0e-1
    max_spring_force = 20.0
    max_contact_force = 50.0
    max_speed = 50.0
    smoothing_speed = 1.0e-6

    # ----------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        config = getattr(scene, "config", {}) or {}
        metadata = scene.metadata()
        frame0 = scene.initial_state()

        self.X = np.array(frame0.position, dtype=float)
        self.V = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float).reshape(-1)
        self.N = int(self.X.shape[0])
        self.gravity = np.asarray(config.get("gravity", _DEFAULT_GRAVITY), dtype=float).reshape(3)

        self.faces = np.asarray(metadata["topology"]["faces"], dtype=np.int64)
        grid_shape = metadata["topology"].get("grid_shape", (25, 25))
        self.nx, self.ny = int(grid_shape[0]), int(grid_shape[1])

        friction = config.get("friction", {}) or {}
        self.friction_coefficient = float(friction.get("coefficient", 0.2))

        contact_cfg = config.get("contact", {}) or {}
        self.thickness = float(contact_cfg.get("thickness", 0.004))

        # Reference (material) coordinates come from the scene's uv attribute.
        uv = frame0.attributes.get("uv")
        if uv is None:
            u = np.tile(np.arange(self.nx) / max(self.nx - 1, 1), self.ny)
            v = np.repeat(np.arange(self.ny) / max(self.ny - 1, 1), self.nx)
            uv = np.stack([u, v], axis=1)
        uv = np.asarray(uv, dtype=float)
        cloth_cfg = config.get("cloth", {}) or {}
        side = float(cloth_cfg.get("side_length", 0.0))
        if side <= 0.0:
            span_x = float(np.linalg.norm(self.X[self.nx - 1] - self.X[0])) if self.nx > 1 else 0.0
            idx_y = (self.ny - 1) * self.nx
            span_y = float(np.linalg.norm(self.X[idx_y] - self.X[0])) if self.ny > 1 else 0.0
            side = max(span_x, span_y, 1e-6)
        self.side_length = side
        self.reference = uv * side
        self.uv = uv.copy()

        self._build_plane(config)
        self._build_springs()
        self._build_reference_elements()

        # Per node mass contribution (a single conserved material).
        self.mass_quantity = self.mass.copy()
        self.cumulative_mass_source = np.zeros(1, dtype=float)
        self.contact_length_scale = float(np.mean(np.linalg.norm(
            self.reference[self.faces[:, 1]] - self.reference[self.faces[:, 0]], axis=1)))
        self.constraint_scale = float(self.contact_length_scale)
        self._stiffness_scale = float(np.mean(
            self.mass[self.faces], axis=1).mean()) if self.N else 1.0

        self.time = 0.0
        self.steps_taken = 0
        self.contact_time = None
        self._force_cache = None

    # ------------------------------------------------------------------
    def _build_plane(self, config) -> None:
        """Frozen rigid box transform / half extents -> contact plane data."""
        geometry = config.get("scene_geometry") or {}
        plane = None
        for item in geometry.get("objects", []):
            if item.get("id") == "inclined_plane":
                plane = item
                break

        if plane is not None:
            half = np.asarray(plane["geometry"]["half_extents"], dtype=float)
            transform = plane.get("transform", {})
            rotation = _rotation_from_quaternion(transform.get("rotation_xyzw", (0.0, 0.0, 0.0, 1.0)))
            self.plane_center = np.asarray(transform.get("translation", (0.0, 0.0, 0.0)), dtype=float)
        else:  # pragma: no cover - config fallback
            incline = config.get("incline", {}) or {}
            angle = np.deg2rad(float(incline.get("angle_degrees", 25.0)))
            rotation = _rotation_from_quaternion(
                (0.0, -np.sin(angle / 2.0), 0.0, np.cos(angle / 2.0))
            )
            self.plane_center = np.asarray(incline.get("plane_translation", (0.0, 0.0, 0.0)), dtype=float)
            half = np.array([1.8, 1.15, float(incline.get("plane_half_thickness", 0.04))])
        self.plane_rotation = rotation
        self.plane_half_extents = half
        self.plane_axis_u = rotation @ np.array([1.0, 0.0, 0.0])
        self.plane_axis_v = rotation @ np.array([0.0, 1.0, 0.0])
        normal = rotation @ np.array([0.0, 0.0, 1.0])
        if normal[2] < 0.0:  # ensure outward (upper) face normal
            normal = -normal
        self.plane_normal = normal
        self.offset_along_normal = float(half[2]) + 0.5 * self.thickness
        # Collision is only active over the box's upper face footprint.
        self.contact_footprint = np.array([half[0], half[1]])

    # ------------------------------------------------------------------
    def _build_springs(self) -> None:
        nx, ny = self.nx, self.ny
        pairs_i: list[int] = []
        pairs_j: list[int] = []
        kinds: list[int] = []

        def add(i: int, j: int, kind: int) -> None:
            pairs_i.append(i)
            pairs_j.append(j)
            kinds.append(kind)

        for j in range(ny):  # structural, u direction
            for i in range(nx - 1):
                add(j * nx + i, j * nx + i + 1, 0)
        for j in range(ny - 1):  # structural, v direction
            for i in range(nx):
                add(j * nx + i, (j + 1) * nx + i, 0)
        for j in range(ny - 1):  # shear diagonals
            for i in range(nx - 1):
                a = j * nx + i
                b = a + 1
                c = a + nx
                d = a + nx + 1
                add(a, d, 1)
                add(b, c, 1)
        for j in range(ny):  # bending, u direction
            for i in range(nx - 2):
                add(j * nx + i, j * nx + i + 2, 2)
        for j in range(ny - 2):  # bending, v direction
            for i in range(nx):
                add(j * nx + i, (j + 2) * nx + i, 2)

        self.spring_i = np.asarray(pairs_i, dtype=np.int64)
        self.spring_j = np.asarray(pairs_j, dtype=np.int64)
        kinds = np.asarray(kinds, dtype=np.int64)

        rest = np.linalg.norm(self.reference[self.spring_j] - self.reference[self.spring_i], axis=1)
        stiffness = np.empty_like(rest)
        stiffness[kinds == 0] = self.structural_stiffness
        stiffness[kinds == 1] = self.shear_stiffness
        stiffness[kinds == 2] = self.bending_stiffness
        self.spring_rest = np.maximum(rest, 1e-9)
        self.spring_k = stiffness
        reduced_mass = 0.5 * float(np.mean(self.mass))
        self.spring_c = 2.0 * self.zeta_spring * np.sqrt(stiffness * reduced_mass)

    # ------------------------------------------------------------------
    def _build_reference_elements(self) -> None:
        faces = self.faces
        ref = self.reference[faces]  # [Ne,3,2]
        edge_b = ref[:, 1] - ref[:, 0]
        edge_c = ref[:, 2] - ref[:, 0]
        matrix = np.stack([edge_b, edge_c], axis=2)  # columns
        determinant = matrix[:, 0, 0] * matrix[:, 1, 1] - matrix[:, 0, 1] * matrix[:, 1, 0]
        safe = np.where(np.abs(determinant) < 1e-14, 1e-14, determinant)
        inverse = np.empty_like(matrix)
        inverse[:, 0, 0] = matrix[:, 1, 1] / safe
        inverse[:, 0, 1] = -matrix[:, 0, 1] / safe
        inverse[:, 1, 0] = -matrix[:, 1, 0] / safe
        inverse[:, 1, 1] = matrix[:, 0, 0] / safe
        self.element_reference_inverse = inverse
        self.vertex_valence = np.maximum(
            np.bincount(faces.reshape(-1), minlength=self.N), 1
        ).astype(float)
        self.contact_damping = (
            2.0 * self.contact_damping_ratio * np.sqrt(self.contact_stiffness * float(np.mean(self.mass)))
        )

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def _spring_forces(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        i, j = self.spring_i, self.spring_j
        delta = X[j] - X[i]
        length = np.sqrt(np.einsum("ij,ij->i", delta, delta)) + 1e-12
        unit = delta / length[:, None]
        stretch = length - self.spring_rest
        relative = V[j] - V[i]
        rate = np.einsum("ij,ij->i", relative, unit)
        magnitude = self.spring_k * stretch + self.spring_c * rate
        magnitude = np.clip(magnitude, -self.max_spring_force, self.max_spring_force)
        force = magnitude[:, None] * unit
        out = np.empty_like(X)
        for axis in range(3):
            out[:, axis] = np.bincount(i, weights=force[:, axis], minlength=self.N) - np.bincount(
                j, weights=force[:, axis], minlength=self.N
            )
        return out

    def _signed_gap(self, X: np.ndarray) -> np.ndarray:
        relative = X - self.plane_center
        along_normal = relative @ self.plane_normal - self.offset_along_normal
        along_u = np.abs(relative @ self.plane_axis_u)
        along_v = np.abs(relative @ self.plane_axis_v)
        inside = (along_u <= self.contact_footprint[0]) & (along_v <= self.contact_footprint[1])
        return np.where(inside, along_normal, 1.0e3)

    def _contact_forces(self, X: np.ndarray, V: np.ndarray, non_contact: np.ndarray):
        gap = self._signed_gap(X)
        penetration = np.maximum(-gap, 0.0)
        normal_speed = V @ self.plane_normal
        normal_force = self.contact_stiffness * penetration - self.contact_damping * normal_speed
        active = penetration > 0.0
        normal_force = np.where(active, np.maximum(normal_force, 0.0), 0.0)
        normal_force = np.minimum(normal_force, self.max_contact_force)

        # Tangential component of the non-contact (gravity + elastic) forces.
        normal_component = non_contact @ self.plane_normal
        tangential = non_contact - normal_component[:, None] * self.plane_normal
        tangential_speed = V - normal_speed[:, None] * self.plane_normal
        speed = np.sqrt(np.einsum("ij,ij->i", tangential_speed, tangential_speed))

        cone = self.friction_coefficient * normal_force
        tangential_magnitude = np.sqrt(np.einsum("ij,ij->i", tangential, tangential))
        stick = active & (tangential_magnitude <= cone)
        slide = active & ~stick

        friction = np.zeros_like(X)
        if np.any(stick):
            friction[stick] = -tangential[stick]
        if np.any(slide):
            direction = tangential_speed[slide] / np.maximum(speed[slide], self.smoothing_speed)[:, None]
            friction[slide] = -cone[slide, None] * direction
        return gap, active, normal_force, friction

    def _step(self, dt: float) -> None:
        X, V = self.X, self.V
        force = self._spring_forces(X, V)
        force += self.mass[:, None] * self.gravity[None, :]
        gap, active, normal_force, friction = self._contact_forces(X, V, force)
        force += normal_force[:, None] * self.plane_normal[None, :]
        force += friction
        acceleration = force / self.mass[:, None]
        V = V + acceleration * dt
        np.clip(V, -self.max_speed, self.max_speed, out=V)
        self.X = X + V * dt
        if not np.isfinite(self.X).all() or not np.isfinite(V).all():  # pragma: no cover
            self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
            V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
        self.V = V
        gap = self._signed_gap(self.X)
        if self.contact_time is None and float(np.min(gap)) <= 0.0:
            self.contact_time = self.time + dt
        self.steps_taken += 1

    # ------------------------------------------------------------------
    def _deformation_gradient(self, X: np.ndarray) -> np.ndarray:
        faces = self.faces
        points = X[faces]
        edge_b = points[:, 1] - points[:, 0]
        edge_c = points[:, 2] - points[:, 0]
        normal = np.cross(edge_b, edge_c)
        normal_length = np.sqrt(np.einsum("ij,ij->i", normal, normal)) + 1e-14
        normal_unit = normal / normal_length[:, None]
        edge_b_length = np.sqrt(np.einsum("ij,ij->i", edge_b, edge_b)) + 1e-14
        axis_1 = edge_b / edge_b_length[:, None]
        axis_2 = np.cross(normal_unit, axis_1)
        q_b = np.stack(
            [np.einsum("ij,ij->i", edge_b, axis_1), np.einsum("ij,ij->i", edge_b, axis_2)], axis=1
        )
        q_c = np.stack(
            [np.einsum("ij,ij->i", edge_c, axis_1), np.einsum("ij,ij->i", edge_c, axis_2)], axis=1
        )
        current = np.stack([q_b, q_c], axis=2)
        element_gradient = current @ self.element_reference_inverse

        gradient = np.zeros((self.N, 2, 2), dtype=float)
        for corner in range(3):
            for row in range(2):
                for column in range(2):
                    gradient[:, row, column] += np.bincount(
                        faces[:, corner],
                        weights=element_gradient[:, row, column],
                        minlength=self.N,
                    )
        return gradient / self.vertex_valence[:, None, None]

    # ------------------------------------------------------------------
    def _build_frame(self, time: float) -> StateFrame:
        force = self._spring_forces(self.X, self.V)
        force += self.mass[:, None] * self.gravity[None, :]
        gap, active, normal_force, friction = self._contact_forces(self.X, self.V, force)
        penetration = np.maximum(-gap, 0.0)
        gap_report = np.where(gap >= 1.0e2, 1.0e2, gap)
        normal_broadcast = np.broadcast_to(self.plane_normal, (self.N, 3)).copy()
        active_mask = active.astype(float)
        zero = np.zeros(self.N, dtype=float)

        frame = StateFrame(
            float(time),
            self.X.copy(),
            self.V.copy(),
            mass=self.mass.copy(),
            attributes={
                "uv": self.uv.copy(),
                "cloth_vertex": np.ones(self.N, dtype=float),
                "mass_quantity": self.mass_quantity.copy(),
                "deformation_gradient": self._deformation_gradient(self.X),
            },
            fields={
                "constraint_value": penetration,
                "constraint_target": zero.copy(),
                "contact_signed_distance": gap_report,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "friction_active_mask": active_mask,
                "friction_normal": normal_broadcast,
                "friction_normal_force": normal_force,
                "friction_relative_velocity": self.V.copy(),
                "friction_tangential_force": friction,
                "friction_dynamic_coefficient": np.full(self.N, self.friction_coefficient),
                "friction_static_coefficient": np.full(self.N, self.friction_coefficient),
                "gravity_acceleration": self.gravity.reshape(1, 3).copy(),
            },
            scalars={
                "friction_coefficient": float(self.friction_coefficient),
                "constraint_scale": float(self.constraint_scale),
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-12:
            raise ValueError("advance_to requires monotonically increasing times")
        while self.time < target - 1e-12:
            step = min(self.time_step, target - self.time)
            self._step(step)
            self.time += step
        self.time = target
        return self._build_frame(target)

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "system": "mass_spring_cloth_penalty_contact",
            "nodes": int(self.N),
            "elements": int(len(self.faces)),
            "internal_time_step": float(self.time_step),
            "steps": int(self.steps_taken),
            "first_contact_time": None if self.contact_time is None else float(self.contact_time),
            "friction_coefficient": float(self.friction_coefficient),
            "contact_thickness": float(self.thickness),
            "final_time": float(self.time),
        }
