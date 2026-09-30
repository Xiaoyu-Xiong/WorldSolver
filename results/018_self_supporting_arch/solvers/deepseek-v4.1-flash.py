"""Rigid-body solver for the self-supporting parabolic arch (scene 018).

The scene is a chain of 15 rigid voussoir blocks (convex prisms) resting on a
static frictional ground plane at z=0.  The solver implements a physical
time-stepping scheme from scratch with NumPy:

  * rigid-body state (position, quaternion, linear/angular velocity),
  * exact mass/inertia from the frozen triangle meshes,
  * vertex-versus-face contact detection between all body pairs (both
    directions) plus vertex-versus-ground contacts,
  * a projected-Gauss-Seidel sequential-impulse contact solver with Coulomb
    friction (mu = 0.6), Baumgarte penetration stabilisation and warm
    starting,
  * semi-implicit Euler integration with quaternion normalisation.

The physical phenomenon reproduced is the arch carrying its own weight in
compression through the mating frictional joints down to the two springing
blocks on the ground.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# --------------------------------------------------------------------------
# frozen numerical parameters
# --------------------------------------------------------------------------
_DEFAULT_GRAVITY = np.array([0.0, 0.0, -9.81])
_DEFAULT_MU = 0.6
_SUBSTEPS_PER_SAMPLE = 20          # dt_sub = sample_dt / 20 = 1 ms
_PGS_ITERS = 80
_BAUMGARTE = 0.2
_SLOP = 1.0e-3
_CONTACT_MARGIN = 3.0e-3
_MAX_PENETRATION = 0.05
_MAX_BIAS_VELOCITY = 0.2
_DAMPING = 0.2
_MAX_LIN_VELOCITY = 25.0
_MAX_ANG_VELOCITY = 25.0
_IN_POLYGON_TOL = 2.5e-3
_PAIR_MARGIN = 0.02


# --------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------
def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f":
            indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
            for k in range(1, len(indices) - 1):
                triangles.append([indices[0], indices[k], indices[k + 1]])
    return np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=int)


def _mesh_properties(
    vertices: np.ndarray, triangles: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume, centroid and unit-density inertia tensor about the centroid."""
    simplex_second_moment = (np.eye(3) + np.ones((3, 3))) / 120.0
    volume = 0.0
    first = np.zeros(3)
    second = np.zeros((3, 3))
    for tri in triangles:
        a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        det = float(np.dot(a, np.cross(b, c)))
        tet_volume = det / 6.0
        mat = np.stack([a, b, c], axis=1)
        volume += tet_volume
        first += tet_volume * (a + b + c) / 4.0
        second += det * (mat @ simplex_second_moment @ mat.T)
    sign = 1.0 if volume >= 0.0 else -1.0
    volume *= sign
    first *= sign
    second *= sign
    if volume <= 0.0:
        raise ValueError("degenerate mesh")
    centroid = first / volume
    centered = second - volume * np.outer(centroid, centroid)
    inertia = np.trace(centered) * np.eye(3) - centered
    return float(volume), centroid, inertia


def _mesh_faces(
    vertices: np.ndarray, triangles: np.ndarray, center: np.ndarray
) -> list[list[int]]:
    """Group coplanar triangles into outward-oriented polygonal faces."""
    groups: list[tuple[np.ndarray, float, list[int]]] = []
    for tri in triangles:
        p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1.0e-12:
            continue
        normal = normal / norm
        offset = float(np.dot(normal, p0))
        placed = False
        for index, (gn, go, tris) in enumerate(groups):
            same = np.allclose(normal, gn, atol=1.0e-6) and abs(offset - go) < 1.0e-6
            flipped = np.allclose(normal, -gn, atol=1.0e-6) and abs(offset + go) < 1.0e-6
            if same or flipped:
                tris.append(int(tri[0]))
                tris.append(int(tri[1]))
                tris.append(int(tri[2]))
                placed = True
                break
        if not placed:
            groups.append((normal, offset, [int(tri[0]), int(tri[1]), int(tri[2])]))
    faces: list[list[int]] = []
    for normal, _offset, flat in groups:
        unique = sorted(set(flat))
        points = vertices[unique]
        face_center = points.mean(axis=0)
        oriented = np.asarray(normal, dtype=float)
        if np.dot(face_center - center, oriented) < 0.0:
            oriented = -oriented
        axis = points[0] - face_center
        axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
        side = np.cross(oriented, axis)
        angles = np.arctan2((points - face_center) @ side, (points - face_center) @ axis)
        order = np.argsort(angles)
        faces.append([unique[int(o)] for o in order])
    return faces


def _quat_to_mat(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty((q.shape[0], 3, 3), dtype=float)
    out[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    out[:, 0, 1] = 2.0 * (x * y - z * w)
    out[:, 0, 2] = 2.0 * (x * z + y * w)
    out[:, 1, 0] = 2.0 * (x * y + z * w)
    out[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    out[:, 1, 2] = 2.0 * (y * z - x * w)
    out[:, 2, 0] = 2.0 * (x * z - y * w)
    out[:, 2, 1] = 2.0 * (y * z + x * w)
    out[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return out


def _integrate_quaternion(quaternion: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    qw, qx, qy, qz = quaternion[:, 3], quaternion[:, 0], quaternion[:, 1], quaternion[:, 2]
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    half = 0.5 * dt
    d0 = half * (wx * qw + wy * qz - wz * qy)
    d1 = half * (-wx * qz + wy * qw + wz * qx)
    d2 = half * (wx * qy - wy * qx + wz * qw)
    d3 = half * (-wx * qx - wy * qy - wz * qz)
    updated = quaternion + np.stack([d0, d1, d2, d3], axis=1)
    norm = np.linalg.norm(updated, axis=1, keepdims=True)
    return updated / np.maximum(norm, 1.0e-12)


def _skew(vector: np.ndarray) -> np.ndarray:
    out = np.zeros((vector.shape[0], 3, 3), dtype=float)
    out[:, 0, 1] = -vector[:, 2]
    out[:, 0, 2] = vector[:, 1]
    out[:, 1, 0] = vector[:, 2]
    out[:, 1, 2] = -vector[:, 0]
    out[:, 2, 0] = -vector[:, 1]
    out[:, 2, 1] = vector[:, 0]
    return out


def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.zeros_like(normal)
    smallest = np.argmin(np.abs(normal), axis=1)
    axis[np.arange(normal.shape[0]), smallest] = 1.0
    t1 = np.cross(normal, axis)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1.0e-12)
    t2 = np.cross(normal, t1)
    return t1, t2


def _scatter_add(target: np.ndarray, index: np.ndarray, values: np.ndarray) -> None:
    size = target.shape[0]
    for component in range(3):
        target[:, component] += np.bincount(
            index, weights=values[:, component], minlength=size
        )


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
class BenchmarkSolver:
    """Physical time-stepping solver for the self-supporting arch."""

    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        config = getattr(scene, "config", None) or {}
        gravity = np.asarray(config.get("gravity", _DEFAULT_GRAVITY), dtype=float)
        self._gravity = gravity if gravity.shape == (3,) else _DEFAULT_GRAVITY.copy()
        friction = config.get("friction", {})
        self._mu = float(friction.get("coefficient", _DEFAULT_MU)) if isinstance(
            friction, dict
        ) else float(_DEFAULT_MU)
        sample_dt = float(config.get("sample_dt", 0.02))
        self._sample_dt = sample_dt if sample_dt > 0.0 else 0.02
        self._substeps_per_sample = _SUBSTEPS_PER_SAMPLE
        self._dt = self._sample_dt / float(self._substeps_per_sample)

        initial = scene.initial_state()
        position = np.asarray(initial.position, dtype=float)
        self._n_bodies = int(position.shape[0])
        self._x = position.copy()
        self._v = np.asarray(initial.velocity, dtype=float).copy()
        orientation = initial.orientation
        if orientation is None:
            orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self._n_bodies, 1))
        self._q = np.asarray(orientation, dtype=float).copy()
        self._q /= np.maximum(np.linalg.norm(self._q, axis=1, keepdims=True), 1.0e-12)
        angular = initial.angular_velocity
        self._w = (
            np.asarray(angular, dtype=float).copy()
            if angular is not None
            else np.zeros((self._n_bodies, 3), dtype=float)
        )
        self._mass = np.asarray(initial.mass, dtype=float).copy()
        self._body_id = np.asarray(initial.body_id, dtype=int).copy()

        self._load_meshes(config)
        # The scene transform places the mesh origin; the dynamics integrate the
        # centre of mass, which is offset from that origin by R @ com_local.
        self._x = self._x + np.einsum(
            "nij,nj->ni", _quat_to_mat(self._q), self._com_local
        )
        self._build_body_arrays()
        self._build_contact_features()
        self._warm: dict[tuple, np.ndarray] = {}
        self._step_count = 0
        self._max_penetration = 0.0
        self._min_gap = float("inf")

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _asset_map(self, config: dict) -> dict[int, str]:
        assets: dict[int, str] = {}
        geometry = config.get("scene_geometry")
        if isinstance(geometry, dict):
            for item in geometry.get("objects", []):
                binding = item.get("state_binding") or {}
                if binding.get("selector") == "body_id" and item.get("motion") == "dynamic":
                    asset = (item.get("geometry") or {}).get("asset")
                    if asset:
                        assets[int(binding["value"])] = str(asset)
        return assets

    def _load_meshes(self, config: dict) -> None:
        base = Path(__file__).resolve().parent.parent
        assets = self._asset_map(config)
        vertices: list[np.ndarray] = []
        face_vertices: list[np.ndarray] = []
        centers_of_mass: list[np.ndarray] = []
        for body in range(self._n_bodies):
            asset = assets.get(body, f"assets/voussoir_{body:02d}.obj")
            path = Path(asset)
            if not path.is_absolute():
                path = base / asset
            if not path.is_file():
                path = base / f"assets/voussoir_{body:02d}.obj"
            verts, tris = _load_obj(path)
            volume, centroid, inertia = _mesh_properties(verts, tris)
            faces = _mesh_faces(verts, tris, centroid)
            if len(faces) != 6 or any(len(f) != 4 for f in faces):
                raise ValueError(f"unexpected face structure in {path}")
            # Work in body coordinates measured from the centre of mass so the
            # dynamics use the true COM; positions reported to the platform are
            # converted back to the frozen mesh origin in _snapshot.
            vertices.append(verts - centroid)
            face_vertices.append(np.asarray(faces, dtype=int))
            centers_of_mass.append(centroid)
            density = self._mass[body] / volume
            body_inertia = density * inertia
            body_inertia += 1.0e-9 * np.eye(3) * max(float(np.trace(body_inertia)), 1.0)
            self._inertia_body = getattr(self, "_inertia_body", [])
            self._inertia_body.append(body_inertia)
        self._verts_local = np.asarray(vertices, dtype=float)
        self._faces_idx = np.asarray(face_vertices, dtype=int)
        self._com_local = np.asarray(centers_of_mass, dtype=float)
        self._inertia_body = np.asarray(self._inertia_body, dtype=float)
        self._inv_inertia_body = np.linalg.inv(self._inertia_body)

    def _build_body_arrays(self) -> None:
        n = self._n_bodies
        local = self._verts_local
        idx = self._faces_idx
        body_axis = np.arange(n)[:, None, None]
        face_points = local[body_axis, idx]
        centers = face_points.mean(axis=2)
        edges = []
        for edge in range(4):
            first = local[np.arange(n)[:, None], idx[:, :, edge], :]
            second = local[np.arange(n)[:, None], idx[:, :, (edge + 1) % 4], :]
            edges.append((first, second))
        normals = np.cross(
            face_points[:, :, 1] - face_points[:, :, 0],
            face_points[:, :, 2] - face_points[:, :, 0],
        )
        normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1.0e-12)
        mesh_center = local.mean(axis=1)
        outward = np.einsum(
            "nfc,nc->nf", centers - mesh_center[:, None, :], normals
        ) < 0.0
        normals[outward] *= -1.0
        self._face_normals = normals
        self._face_centers = centers
        edge_normals = np.empty((n, 6, 4, 3), dtype=float)
        edge_points = np.empty((n, 6, 4, 3), dtype=float)
        for edge, (first, second) in enumerate(edges):
            direction = second - first
            candidate = np.cross(direction, normals)
            candidate /= np.maximum(
                np.linalg.norm(candidate, axis=2, keepdims=True), 1.0e-12
            )
            edge_normals[:, :, edge, :] = candidate
            edge_points[:, :, edge, :] = first
        self._edge_normals = edge_normals
        self._edge_points = edge_points
        self._radius = np.max(np.linalg.norm(local, axis=2), axis=1)
        thickness = float(np.mean(local[:, :, 1].max(axis=1) - local[:, :, 1].min(axis=1)))
        characteristic = float(
            np.mean(np.max(local, axis=1) - np.min(local, axis=1))
        )
        self._contact_length_scale = max(thickness, 1.0e-6)
        self._constraint_scale = max(characteristic, 1.0e-6)
        self._inv_mass_full = np.concatenate(
            [1.0 / np.maximum(self._mass, 1.0e-12), np.zeros(1)]
        )
        inertia_full = np.concatenate(
            [self._inv_inertia_body, np.zeros((1, 3, 3))], axis=0
        )
        self._inv_inertia_full_local = inertia_full

    def _build_contact_features(self) -> None:
        """Frozen per-frame contact samples: 4 corners per joint + ground vertices."""
        n = self._n_bodies
        rotation = _quat_to_mat(self._q)
        points = np.einsum("nij,nkj->nki", rotation, self._verts_local) + self._x[:, None, :]
        normals = np.einsum("nij,nfj->nfi", rotation, self._face_normals)
        centers = np.einsum("nij,nfj->nfi", rotation, self._face_centers) + self._x[:, None, :]
        edge_normals = np.einsum("nij,nfkj->nfki", rotation, self._edge_normals)
        edge_points = (
            np.einsum("nij,nfkj->nfki", rotation, self._edge_points)
            + self._x[:, None, None, :]
        )
        joint_features: list[tuple[int, int, int, int]] = []
        seen: set[tuple] = set()
        for lower in range(n - 1):
            upper = lower + 1
            for face in range(6):
                plane_normal = normals[lower, face]
                offset = float(np.dot(centers[lower, face], plane_normal))
                for vertex in range(points.shape[1]):
                    gap = float(np.dot(points[upper, vertex], plane_normal) - offset)
                    if abs(gap) > 1.0e-5:
                        continue
                    inside = True
                    for edge in range(4):
                        distance = float(
                            np.dot(
                                points[upper, vertex] - edge_points[lower, face, edge],
                                edge_normals[lower, face, edge],
                            )
                        )
                        if distance > _IN_POLYGON_TOL:
                            inside = False
                            break
                    if not inside:
                        continue
                    key = (
                        upper,
                        vertex,
                        round(float(points[upper, vertex, 0]), 4),
                        round(float(points[upper, vertex, 1]), 4),
                        round(float(points[upper, vertex, 2]), 4),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    joint_features.append((upper, vertex, lower, face))
        self._joint_features = np.asarray(joint_features, dtype=int).reshape(-1, 4)
        springing = [0, n - 1]
        ground = [
            (body, vertex)
            for body in springing
            for vertex in range(points.shape[1])
            if points[body, vertex, 2] < 1.0e-3
        ]
        self._ground_features = np.asarray(ground, dtype=int).reshape(-1, 2)

    # ------------------------------------------------------------------
    # contact detection and solve
    # ------------------------------------------------------------------
    def _world_geometry(self):
        rotation = _quat_to_mat(self._q)
        points = np.einsum("nij,nkj->nki", rotation, self._verts_local) + self._x[:, None, :]
        normals = np.einsum("nij,nfj->nfi", rotation, self._face_normals)
        centers = np.einsum("nij,nfj->nfi", rotation, self._face_centers) + self._x[:, None, :]
        edge_normals = np.einsum("nij,nfkj->nfki", rotation, self._edge_normals)
        edge_points = (
            np.einsum("nij,nfkj->nfki", rotation, self._edge_points)
            + self._x[:, None, None, :]
        )
        return points, normals, centers, edge_normals, edge_points

    def _detect_contacts(self):
        n = self._n_bodies
        points, normals, centers, edge_normals, edge_points = self._world_geometry()
        separation = np.linalg.norm(self._x[:, None, :] - self._x[None, :, :], axis=-1)
        reach = self._radius[:, None] + self._radius[None, :] + _PAIR_MARGIN
        mask = (separation < reach) & (~np.eye(n, dtype=bool))
        pairs = np.argwhere(mask)
        contact_v = np.zeros(0, dtype=int)
        contact_k = np.zeros(0, dtype=int)
        contact_f = np.zeros(0, dtype=int)
        contact_face = np.zeros(0, dtype=int)
        contact_p = np.zeros((0, 3))
        contact_n = np.zeros((0, 3))
        depth = np.zeros(0)
        if pairs.shape[0]:
            lower = pairs[:, 0]
            upper = pairs[:, 1]
            vertex_points = points[lower]
            face_normals = normals[upper]
            face_centers = centers[upper]
            face_edge_normals = edge_normals[upper]
            face_edge_points = edge_points[upper]
            face_offset = np.einsum("pfc,pfc->pf", face_centers, face_normals)
            edge_offset = np.einsum("pfec,pfec->pfe", face_edge_points, face_edge_normals)
            gap = (
                np.einsum("pkc,pfc->pkf", vertex_points, face_normals)
                - face_offset[:, None, :]
            )
            inside = (
                np.einsum("pkc,pfec->pkfe", vertex_points, face_edge_normals)
                - edge_offset[:, :, None, :]
            ).max(axis=-1)
            active = (gap < _CONTACT_MARGIN) & (gap > -_MAX_PENETRATION) & (
                inside <= _IN_POLYGON_TOL
            )
            pair_index, vertex_index, face_index = np.nonzero(active)
            contact_v = lower[pair_index]
            contact_k = vertex_index
            contact_f = upper[pair_index]
            contact_face = face_index
            contact_p = points[contact_v, contact_k]
            contact_n = normals[contact_f, contact_face]
            depth = -gap[pair_index, vertex_index, face_index]
        ground_depth = -points[:, :, 2]
        ground_hit = np.nonzero(ground_depth < _CONTACT_MARGIN)
        ground_body, ground_vertex = ground_hit[0], ground_hit[1]
        if ground_body.size:
            contact_v = np.concatenate([contact_v, ground_body])
            contact_k = np.concatenate([contact_k, ground_vertex])
            contact_f = np.concatenate([contact_f, np.full(ground_body.size, -1, dtype=int)])
            contact_face = np.concatenate(
                [contact_face, np.full(ground_body.size, -1, dtype=int)]
            )
            contact_p = np.concatenate([contact_p, points[ground_body, ground_vertex]], axis=0)
            contact_n = np.concatenate(
                [contact_n, np.tile(np.array([0.0, 0.0, 1.0]), (ground_body.size, 1))],
                axis=0,
            )
            depth = np.concatenate([depth, ground_depth[ground_body, ground_vertex]])
        return self._prune_contacts(
            contact_v, contact_k, contact_f, contact_face, contact_p, contact_n, depth
        )

    @staticmethod
    def _prune_contacts(contact_v, contact_k, contact_f, contact_face, contact_p, contact_n, depth):
        """Remove duplicated constraints (the meshes share exact corner points)."""
        order = np.lexsort(
            (
                contact_face,
                contact_f,
                contact_k,
                np.round(contact_p[:, 2], 4),
                np.round(contact_p[:, 1], 4),
                np.round(contact_p[:, 0], 4),
                contact_v,
            )
        )
        buckets: dict[tuple, list[np.ndarray]] = {}
        keep: list[int] = []
        for position in order:
            index = int(position)
            key = (
                int(contact_v[index]),
                int(contact_f[index]),
                round(float(contact_p[index, 0]), 4),
                round(float(contact_p[index, 1]), 4),
                round(float(contact_p[index, 2]), 4),
            )
            stored = buckets.get(key)
            normal = contact_n[index]
            if stored is None:
                buckets[key] = [normal]
                keep.append(index)
                continue
            duplicate = False
            for other in stored:
                if abs(float(np.dot(other, normal))) > 0.9:
                    duplicate = True
                    break
            if not duplicate:
                stored.append(normal)
                keep.append(index)
        selection = np.asarray(keep, dtype=int)
        return (
            contact_v[selection],
            contact_k[selection],
            contact_f[selection],
            contact_face[selection],
            contact_p[selection],
            contact_n[selection],
            depth[selection],
        )

    def _solve_contacts(self, contact_v, contact_k, contact_f, contact_face, contact_p, contact_n, depth):
        count = contact_v.shape[0]
        self._warm_next = {}
        if count == 0:
            self._warm = self._warm_next
            return
        static_body = self._n_bodies
        body_a = contact_v
        body_b = np.where(contact_f < 0, static_body, contact_f)
        x_full = np.concatenate([self._x, np.zeros((1, 3))], axis=0)
        v_full = np.concatenate([self._v, np.zeros((1, 3))], axis=0)
        w_full = np.concatenate([self._w, np.zeros((1, 3))], axis=0)
        inv_mass = self._inv_mass_full
        rotation = _quat_to_mat(self._q)
        inverse_inertia_world = np.einsum(
            "nij,njk,nlk->nil", rotation, self._inv_inertia_body, rotation
        )
        inv_inertia = np.concatenate(
            [inverse_inertia_world, np.zeros((1, 3, 3))], axis=0
        )
        r_a = contact_p - x_full[body_a]
        r_b = contact_p - x_full[body_b]
        skew_a = _skew(r_a)
        skew_b = _skew(r_b)
        inertia_a = inv_inertia[body_a]
        inertia_b = inv_inertia[body_b]
        mass_matrix = (inv_mass[body_a] + inv_mass[body_b])[:, None, None] * np.eye(3)
        mass_matrix -= np.einsum("cai,cab,cbj->cij", skew_a, inertia_a, skew_a)
        mass_matrix -= np.einsum("cai,cab,cbj->cij", skew_b, inertia_b, skew_b)
        normal = contact_n
        tangent_1, tangent_2 = _tangent_basis(normal)
        k_normal = np.maximum(
            np.einsum("ci,cij,cj->c", normal, mass_matrix, normal), 1.0e-12
        )
        k_t1 = np.maximum(np.einsum("ci,cij,cj->c", tangent_1, mass_matrix, tangent_1), 1.0e-12)
        k_t2 = np.maximum(np.einsum("ci,cij,cj->c", tangent_2, mass_matrix, tangent_2), 1.0e-12)
        bias = np.clip(
            _BAUMGARTE / self._dt * np.maximum(depth - _SLOP, 0.0), 0.0, _MAX_BIAS_VELOCITY
        )
        keys = [
            (
                int(body_a[i]),
                int(contact_k[i]),
                int(contact_f[i]),
                int(contact_face[i]),
            )
            for i in range(count)
        ]
        lam = np.zeros((count, 3), dtype=float)
        for index, key in enumerate(keys):
            previous = self._warm.get(key)
            if previous is not None:
                lam[index] = previous
        impulse = (
            lam[:, 0, None] * normal
            + lam[:, 1, None] * tangent_1
            + lam[:, 2, None] * tangent_2
        )
        self._apply_impulse(
            v_full, w_full, inv_mass, inv_inertia, body_a, body_b, r_a, r_b, impulse
        )
        for _ in range(_PGS_ITERS):
            relative = self._relative_velocity(v_full, w_full, body_a, body_b, r_a, r_b)
            normal_speed = np.einsum("ci,ci->c", normal, relative)
            delta = (bias - normal_speed) / k_normal
            updated = np.maximum(0.0, lam[:, 0] + delta)
            applied = updated - lam[:, 0]
            lam[:, 0] = updated
            self._apply_impulse(
                v_full, w_full, inv_mass, inv_inertia, body_a, body_b, r_a, r_b,
                applied[:, None] * normal,
            )
            relative = self._relative_velocity(v_full, w_full, body_a, body_b, r_a, r_b)
            slip_1 = np.einsum("ci,ci->c", tangent_1, relative)
            slip_2 = np.einsum("ci,ci->c", tangent_2, relative)
            trial_1 = lam[:, 1] - slip_1 / k_t1
            trial_2 = lam[:, 2] - slip_2 / k_t2
            limit = self._mu * lam[:, 0]
            magnitude = np.sqrt(trial_1 * trial_1 + trial_2 * trial_2)
            scale = np.where(magnitude > limit, limit / np.maximum(magnitude, 1e-12), 1.0)
            target_1 = trial_1 * scale
            target_2 = trial_2 * scale
            applied_1 = target_1 - lam[:, 1]
            applied_2 = target_2 - lam[:, 2]
            lam[:, 1] = target_1
            lam[:, 2] = target_2
            self._apply_impulse(
                v_full, w_full, inv_mass, inv_inertia, body_a, body_b, r_a, r_b,
                applied_1[:, None] * tangent_1 + applied_2[:, None] * tangent_2,
            )
        self._v = v_full[: self._n_bodies]
        self._w = w_full[: self._n_bodies]
        for index, key in enumerate(keys):
            self._warm_next[key] = lam[index].copy()
        self._warm = self._warm_next
        self._max_penetration = max(self._max_penetration, float(depth.max()))

    @staticmethod
    def _relative_velocity(v, w, body_a, body_b, r_a, r_b):
        return (
            v[body_a]
            + np.cross(w[body_a], r_a)
            - v[body_b]
            - np.cross(w[body_b], r_b)
        )

    @staticmethod
    def _apply_impulse(v, w, inv_mass, inv_inertia, body_a, body_b, r_a, r_b, impulse):
        index = np.concatenate([body_a, body_b])
        sign = np.concatenate(
            [np.ones(body_a.shape[0]), -np.ones(body_b.shape[0])]
        )
        radius = np.concatenate([r_a, r_b], axis=0)
        applied = sign[:, None] * impulse
        angular = np.cross(radius, applied)
        torque = np.einsum("cij,cj->ci", inv_inertia[index], angular)
        _scatter_add(v, index, inv_mass[index, None] * applied)
        _scatter_add(w, index, torque)

    # ------------------------------------------------------------------
    # stepping and sampling
    # ------------------------------------------------------------------
    def _substep(self) -> None:
        dt = self._dt
        self._v += self._gravity[None, :] * dt
        decay = max(0.0, 1.0 - _DAMPING * dt)
        self._v *= decay
        self._w *= decay
        contacts = self._detect_contacts()
        self._solve_contacts(*contacts)
        self._v = np.clip(self._v, -_MAX_LIN_VELOCITY, _MAX_LIN_VELOCITY)
        self._w = np.clip(self._w, -_MAX_ANG_VELOCITY, _MAX_ANG_VELOCITY)
        if not np.isfinite(self._v).all():
            self._v = np.zeros_like(self._v)
        if not np.isfinite(self._w).all():
            self._w = np.zeros_like(self._w)
        self._x = self._x + self._v * dt
        self._q = _integrate_quaternion(self._q, self._w, dt)
        if not np.isfinite(self._x).all():
            self._x = np.nan_to_num(self._x, nan=0.0, posinf=0.0, neginf=0.0)
        self._step_count += 1

    def advance_to(self, time: float):
        target_time = float(time)
        sample_index = int(round(target_time / self._sample_dt))
        target_steps = sample_index * self._substeps_per_sample
        if target_steps > self._step_count:
            steps = min(target_steps - self._step_count, 2000000)
            for _ in range(steps):
                self._substep()
        return self._snapshot(target_time)

    def _contact_gaps(self) -> np.ndarray:
        points, normals, centers, _edge_normals, _edge_points = self._world_geometry()
        gaps: list[np.ndarray] = []
        if self._joint_features.size:
            upper = self._joint_features[:, 0]
            vertex = self._joint_features[:, 1]
            lower = self._joint_features[:, 2]
            face = self._joint_features[:, 3]
            offsets = np.einsum("nfc,nfc->nf", centers, normals)
            value = (
                np.einsum("nc,nc->n", points[upper, vertex], normals[lower, face])
                - offsets[lower, face]
            )
            gaps.append(value)
        if self._ground_features.size:
            body = self._ground_features[:, 0]
            vertex = self._ground_features[:, 1]
            gaps.append(points[body, vertex, 2])
        if not gaps:
            return np.zeros(0, dtype=float)
        return np.concatenate(gaps).astype(float)

    def _snapshot(self, time: float) -> StateFrame:
        gaps = self._contact_gaps()
        finite = np.isfinite(gaps)
        if not finite.all():
            gaps = np.nan_to_num(gaps, nan=0.0, posinf=0.0, neginf=0.0)
        self._min_gap = float(gaps.min()) if gaps.size else 0.0
        fields = {
            "constraint_target": np.zeros_like(gaps),
            "constraint_value": gaps.copy(),
            "contact_signed_distance": gaps.copy(),
            "external_force": self._mass[:, None] * self._gravity[None, :],
        }
        scalars = {
            "constraint_scale": float(self._constraint_scale),
            "contact_length_scale": float(self._contact_length_scale),
        }
        origin = self._x - np.einsum(
            "nij,nj->ni", _quat_to_mat(self._q), self._com_local
        )
        frame = StateFrame(
            time=float(time),
            position=origin,
            velocity=self._v.copy(),
            orientation=self._q.copy(),
            angular_velocity=self._w.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": int(self._step_count),
            "substep_dt": float(self._dt),
            "friction_coefficient": float(self._mu),
            "max_penetration": float(self._max_penetration),
            "min_contact_gap": float(self._min_gap),
            "num_contact_features": int(
                self._joint_features.shape[0] + self._ground_features.shape[0]
            ),
        }
