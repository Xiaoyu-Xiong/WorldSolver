"""Solver Agent-owned implementation for the thumb-index pinch scene.

This solver simulates the hand flesh as a soft, nearly-incompressible elastic
solid using an explicit time-stepping mass-spring/shell model with a
volume-preserving pressure term.  The thumb and index bones act as kinematic
collision obstacles; the thumb follows the prescribed drive from the frozen
configuration.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# Small geometric helpers
# ---------------------------------------------------------------------------


def _normalize_rows(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, eps)


def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion (xyzw) to a 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quaternion_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions (xyzw)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _quaternion_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    half = angle_rad / 2.0
    s = math.sin(half)
    c = math.cos(half)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])


# ---------------------------------------------------------------------------


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = getattr(scene, "config", {})

        # ------------------------------------------------------------------
        # Load the frozen initial state supplied by the scene.
        # ------------------------------------------------------------------
        init_frame = scene.initial_state()
        self.position: np.ndarray = np.asarray(init_frame.position, dtype=float).copy()
        self.velocity: np.ndarray = np.asarray(init_frame.velocity, dtype=float).copy()
        self.mass: np.ndarray = np.asarray(init_frame.mass, dtype=float).copy()
        self.rest_position: np.ndarray = np.asarray(
            init_frame.attributes["rest_position"], dtype=float
        ).copy()
        self.fixed_mask: np.ndarray = np.asarray(
            init_frame.attributes["fixed_mask"], dtype=int
        ).copy()
        self.n_vertices = self.position.shape[0]

        # ------------------------------------------------------------------
        # Read the triangle topology.  The .obj file only contains vertices
        # and faces and is guaranteed to be watertight.
        # ------------------------------------------------------------------
        asset_path = Path("assets") / "hand_flesh.obj"
        faces: list[list[int]] = []
        with asset_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("f "):
                    parts = line.split()
                    faces.append([int(p.split("/")[0]) for p in parts[1:4]])
        self.faces = np.asarray(faces, dtype=int) - 1  # zero-based
        self.n_faces = self.faces.shape[0]

        # ------------------------------------------------------------------
        # Build unique edge list for edge springs.
        # ------------------------------------------------------------------
        edge_set: set[tuple[int, int]] = set()
        edges_list: list[list[int]] = []
        for f in self.faces:
            for u, v_ in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                key = (min(u, v_), max(u, v_))
                if key not in edge_set:
                    edge_set.add(key)
                    edges_list.append([key[0], key[1]])
        self.edges = np.asarray(edges_list, dtype=int)

        # ------------------------------------------------------------------
        # Precompute rest-shape quantities.
        # ------------------------------------------------------------------
        X = self.rest_position
        f0 = self.faces[:, 0]
        f1 = self.faces[:, 1]
        f2 = self.faces[:, 2]

        e1 = X[f1] - X[f0]
        e2 = X[f2] - X[f0]
        n0_vec = np.cross(e1, e2)
        self.rest_area = 0.5 * np.linalg.norm(n0_vec, axis=1)
        self.rest_volume = self._signed_volume(X)

        # Pseudo-tet deformation gradient basis: [e1, e2, normal].
        n0 = _normalize_rows(n0_vec)
        Dm = np.stack([e1, e2, n0], axis=2)  # (F,3,3)
        Dm_det = np.linalg.det(Dm)
        min_det = Dm_det.min()
        if min_det == 0.0 or not np.isfinite(min_det):
            raise ValueError("Rest deformation gradient basis is singular")
        self.Dm_inv = np.linalg.inv(Dm)

        # Edge rest lengths.
        p0 = X[self.edges[:, 0]]
        p1 = X[self.edges[:, 1]]
        self.rest_edge_len = np.linalg.norm(p1 - p0, axis=1)

        # ------------------------------------------------------------------
        # Material and discretisation parameters.
        # ------------------------------------------------------------------
        self.thickness = 0.005  # m
        self.mu = 1000.0  # Pa
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        # Edge spring stiffness = in-plane elastic modulus * thickness.
        self.k_edge = self.mu * self.thickness  # N/m
        # Area spring stiffness = modulus * thickness / rest_area.
        self.k_area = self.mu * self.thickness / np.maximum(self.rest_area, 1e-12)

        # Pressure-like volume stiffness.
        self.k_volume = 5.0e5  # Pa

        # Velocity damping (per unit mass) to remove high-frequency oscillations.
        self.damping = 8.0  # 1/s

        # Contact penalty with the driven bones.
        self.k_contact = 8.0e4  # N/m
        self.contact_max_force = 2.0e4  # N
        self.contact_margin = 0.002  # m

        # Time-stepping.
        self.sample_dt = float(self.config.get("sample_dt", 0.02))
        self.substeps = 10
        self.sub_dt = self.sample_dt / self.substeps

        # Bone transforms.
        self.thumb_drive = self.config.get("thumb_drive", {})
        self._thumb_start = float(self.thumb_drive.get("start_time", 0.3))
        self._thumb_end = float(self.thumb_drive.get("end_time", 1.65))
        self._thumb_hold = float(self.thumb_drive.get("hold_until", 2.4))
        self._thumb_axis = np.asarray(self.thumb_drive.get("rotation_axis", [0.0, 0.0, 1.0]), dtype=float)
        self._thumb_angle = math.radians(float(self.thumb_drive.get("rotation_degrees", 34.0)))
        self._thumb_translation = np.asarray(self.thumb_drive.get("translation", [0.018, 0.052, 0.004]), dtype=float)

        # Scene geometry capsule data (from scene_geometry.yaml).
        self.index_center = np.array([0.105, 0.030, 0.012])
        self.index_quat = np.array([0.0, 0.7071068, 0.0, 0.7071068])
        self.index_radius = 0.010
        self.index_half_height = 0.205 / 2.0

        self.thumb_center0 = np.array([0.025, -0.074, -0.002])
        self.thumb_quat0 = np.array([0.0, 0.7071068, 0.0, 0.7071068])
        self.thumb_radius = 0.012
        self.thumb_half_height = 0.105 / 2.0

        # Cached index bone segment (it is static).
        self._index_p1, self._index_p2 = self._capsule_segment(
            self.index_center, self.index_quat, self.index_half_height
        )

        # Simulation time and stored frame cache.
        self.t = 0.0
        self._frame_cache: dict[float, StateFrame] = {}

    # -----------------------------------------------------------------------
    # Scene helpers
    # -----------------------------------------------------------------------

    def _signed_volume(self, x: np.ndarray) -> float:
        """Signed volume of the closed triangle mesh."""
        i = self.faces[:, 0]
        j = self.faces[:, 1]
        k = self.faces[:, 2]
        v = np.einsum("ij,ij->i", x[i], np.cross(x[j], x[k])) / 6.0
        return float(v.sum())

    def _capsule_segment(self, center: np.ndarray, quat: np.ndarray, half_height: float):
        R = _quaternion_to_matrix(quat)
        axis = R[:, 2]  # default capsule axis is z
        p1 = center - half_height * axis
        p2 = center + half_height * axis
        return p1, p2

    def _thumb_pose(self, t: float):
        """Return thumb capsule center and quaternion at time t."""
        # Easing parameter.
        if t <= self._thumb_start:
            alpha = 0.0
        elif t >= self._thumb_end:
            alpha = 1.0
        else:
            s = (t - self._thumb_start) / (self._thumb_end - self._thumb_start)
            alpha = s * s * (3.0 - 2.0 * s)  # smoothstep

        center = self.thumb_center0 + alpha * self._thumb_translation
        quat = _quaternion_mul(
            _quaternion_from_axis_angle(self._thumb_axis, alpha * self._thumb_angle),
            self.thumb_quat0,
        )
        return center, quat

    # -----------------------------------------------------------------------
    # Force and collision routines
    # -----------------------------------------------------------------------

    def _elastic_forces(self, x: np.ndarray) -> np.ndarray:
        """Edge and area elastic forces."""
        f = np.zeros_like(x)

        # Edge springs.
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        dp = x[j] - x[i]
        l = np.linalg.norm(dp, axis=1, keepdims=True)
        l = np.maximum(l, 1e-12)
        dir_ = dp / l
        stretch = l[:, 0] - self.rest_edge_len
        f_edge = self.k_edge * stretch[:, None] * dir_
        np.add.at(f, i, f_edge)
        np.add.at(f, j, -f_edge)

        # Area springs for in-plane stiffness.
        i = self.faces[:, 0]
        j = self.faces[:, 1]
        k = self.faces[:, 2]
        xi, xj, xk = x[i], x[j], x[k]
        d1 = xj - xi
        d2 = xk - xi
        n_vec = np.cross(d1, d2)
        area = 0.5 * np.linalg.norm(n_vec, axis=1)
        n_unit = _normalize_rows(n_vec)

        # grad_i A = 0.5 * n_unit x (xk - xj)
        grad_i = 0.5 * np.cross(n_unit, xk - xj)
        grad_j = 0.5 * np.cross(n_unit, xi - xk)
        grad_k = 0.5 * np.cross(n_unit, xj - xi)

        delta = (area - self.rest_area)[:, None]
        f_i = -self.k_area[:, None] * delta * grad_i
        f_j = -self.k_area[:, None] * delta * grad_j
        f_k = -self.k_area[:, None] * delta * grad_k

        np.add.at(f, i, f_i)
        np.add.at(f, j, f_j)
        np.add.at(f, k, f_k)

        return f

    def _volume_pressure_forces(self, x: np.ndarray) -> np.ndarray:
        """Pressure forces derived from the current enclosed volume."""
        i = self.faces[:, 0]
        j = self.faces[:, 1]
        k = self.faces[:, 2]
        v_tri = np.einsum("ij,ij->i", x[i], np.cross(x[j], x[k])) / 6.0
        V = v_tri.sum()
        p = self.k_volume * (V - self.rest_volume) / self.rest_volume

        grad_i = np.cross(x[j], x[k]) / 6.0
        grad_j = np.cross(x[k], x[i]) / 6.0
        grad_k = np.cross(x[i], x[j]) / 6.0

        f = np.zeros_like(x)
        np.add.at(f, i, -p * grad_i)
        np.add.at(f, j, -p * grad_j)
        np.add.at(f, k, -p * grad_k)
        return f

    def _capsule_contact(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        radius: float,
        x: np.ndarray,
        v: np.ndarray,
    ):
        """Compute contact force on the flesh vertices from a capsule.

        Returns
        -------
        force : (N,3) array of contact forces
        v_out : (N,3) array of velocities after contact damping/projection
        x_out : (N,3) array of positions after positional projection
        """
        axis = p2 - p1
        inv_len2 = 1.0 / np.dot(axis, axis)

        t = np.clip(np.einsum("ij,j->i", x - p1, axis) * inv_len2, 0.0, 1.0)
        closest = p1 + t[:, None] * axis
        diff = x - closest
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-12)
        n = diff / dist

        sd = dist[:, 0] - radius
        mask = sd < self.contact_margin
        force = np.zeros_like(x)
        if not np.any(mask):
            return force, v.copy(), x.copy()

        depth = self.contact_margin - sd[mask]
        mag = self.k_contact * depth
        mag = np.minimum(mag, self.contact_max_force)
        force[mask] = mag[:, None] * n[mask]

        # Damping: remove inward normal velocity for penetrating vertices.
        v_n = np.einsum("ij,ij->i", v, n)
        inward = v_n < 0.0
        damp_mask = mask & inward
        if np.any(damp_mask):
            v_new = v.copy()
            v_new[damp_mask] -= (1.0 + 0.0) * v_n[damp_mask][:, None] * n[damp_mask]
        else:
            v_new = v.copy()

        # Positional projection to the capsule surface for deep contacts.
        inside = sd < 0.0
        if np.any(inside):
            x_new = x.copy()
            x_new[inside] = closest[inside] + radius * n[inside]
        else:
            x_new = x.copy()

        return force, v_new, x_new

    def _collision_forces(self, t: float, x: np.ndarray, v: np.ndarray):
        """Aggregate contact forces from the driven thumb and the static index bone."""
        # Thumb.
        thumb_center, thumb_quat = self._thumb_pose(t)
        thumb_p1, thumb_p2 = self._capsule_segment(
            thumb_center, thumb_quat, self.thumb_half_height
        )
        f_thumb, v_new, x_new = self._capsule_contact(thumb_p1, thumb_p2, self.thumb_radius, x, v)

        # Index bone (static).
        f_index, v_new, x_new = self._capsule_contact(
            self._index_p1, self._index_p2, self.index_radius, x_new, v_new
        )

        force = f_thumb + f_index
        return force, v_new, x_new

    # -----------------------------------------------------------------------
    # Time stepping
    # -----------------------------------------------------------------------

    def _substep(self, t: float) -> None:
        x = self.position
        v = self.velocity

        f = np.zeros_like(x)

        # Elastic forces (internal).
        f += self._elastic_forces(x)
        # Volume pressure (internal, but drives the outward web bulge).
        f += self._volume_pressure_forces(x)
        # Body damping.
        f -= self.damping * self.mass[:, None] * v

        # Contact forces and kinematic projection.
        f_contact, v_projected, x_projected = self._collision_forces(t, x, v)
        f += f_contact

        # Symplectic Euler integration on the projected state.
        a = f / self.mass[:, None]
        # Fixed nodes have zero acceleration.
        a[self.fixed_mask == 1] = 0.0

        v_new = v_projected + a * self.sub_dt
        x_new = x_projected + v_new * self.sub_dt

        # Enforce boundary conditions.
        fixed = self.fixed_mask == 1
        x_new[fixed] = self.rest_position[fixed]
        v_new[fixed] = 0.0

        self.position = x_new
        self.velocity = v_new

    # -----------------------------------------------------------------------
    # Frame export
    # -----------------------------------------------------------------------

    def _compute_deformation_gradient(self) -> np.ndarray:
        """Per-triangle deformation gradient F = D_s * D_m^{-1}."""
        i = self.faces[:, 0]
        j = self.faces[:, 1]
        k = self.faces[:, 2]
        x = self.position
        d1 = x[j] - x[i]
        d2 = x[k] - x[i]
        n = _normalize_rows(np.cross(d1, d2))
        Ds = np.stack([d1, d2, n], axis=2)
        F = np.einsum("nij,njk->nik", Ds, self.Dm_inv)
        return F

    def _make_frame(self, time: float) -> StateFrame:
        x = self.position
        v = self.velocity

        # Current contact and damping forces for the budget field.
        thumb_center, thumb_quat = self._thumb_pose(time)
        thumb_p1, thumb_p2 = self._capsule_segment(
            thumb_center, thumb_quat, self.thumb_half_height
        )
        f_thumb, _, _ = self._capsule_contact(thumb_p1, thumb_p2, self.thumb_radius, x, v)
        f_index, _, _ = self._capsule_contact(self._index_p1, self._index_p2, self.index_radius, x, v)
        f_contact = f_thumb + f_index
        f_damping = -self.damping * self.mass[:, None] * v

        total_contact = f_contact.sum(axis=0)
        total_damping = f_damping.sum(axis=0)
        budget = np.stack([total_contact, total_damping], axis=0)[None, :, :]  # (1,2,3)

        F = self._compute_deformation_gradient()

        frame = StateFrame(
            time=float(time),
            position=x.copy(),
            velocity=v.copy(),
            mass=self.mass.copy(),
            attributes={
                "rest_position": self.rest_position.copy(),
                "fixed_mask": self.fixed_mask.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "element_deformation_gradient": F.copy(),
                "budget_external_force": budget,
                "cumulative_mass_source": np.zeros(2, dtype=float),
            },
            scalars={
                "reference_force": 1.0,
            },
        )
        return frame

    # -----------------------------------------------------------------------
    # Public solver API
    # -----------------------------------------------------------------------

    def advance_to(self, time: float) -> StateFrame:
        if time in self._frame_cache:
            return self._frame_cache[time]

        # Ensure we never step backwards.
        if time < self.t:
            # Return the closest cached frame (initially t=0 frame is not cached).
            return self._make_frame(time)

        # Integrate with fixed substeps until the requested sample time.
        while self.t < time:
            step_end = min(self.t + self.sub_dt, time)
            self._substep(0.5 * (self.t + step_end))  # midpoint evaluation
            self.t = step_end

        frame = self._make_frame(time)
        self._frame_cache[time] = frame
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "mass_spring_membrane_volume",
            "vertices": self.n_vertices,
            "faces": self.n_faces,
            "sub_dt": self.sub_dt,
            "rest_volume": float(self.rest_volume),
        }
