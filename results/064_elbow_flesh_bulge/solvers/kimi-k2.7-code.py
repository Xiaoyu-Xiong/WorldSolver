"""Physics solver for 064_elbow_flesh_bulge.

Implements a stable explicit elastodynamic simulation of a soft arm flexing
around prescribed bone kinematics.  The flesh mesh is treated as a nearly
incompressible elastic solid using edge-pair tetrahedral elements built from
the frozen triangle surface.  Bone attachment is enforced by kinematically
driving vertices near the bone capsules, and the forearm bone follows the
smoothstep prescribed-hinge rotation from the task.

The solver is deterministic and advances with a fixed sub-step until the
requested sample time, then returns a ``worldsolver.contracts.StateFrame``
containing all required raw trajectory fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _smoothstep(t: float) -> float:
    """Return 3t^2 - 2t^3 clamped to [0, 1]."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


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


def _capsule_sdf(p: np.ndarray, center: np.ndarray, axis: np.ndarray,
                 height: float, radius: float) -> float:
    """Unsigned distance from point p to a capsule.  axis is a unit vector."""
    local = p - center
    along = np.dot(local, axis)
    half_h = height * 0.5
    if along < -half_h:
        closest = center - axis * half_h
    elif along > half_h:
        closest = center + axis * half_h
    else:
        closest = center + axis * along
    return np.linalg.norm(p - closest) - radius


def _capsule_nearest(p: np.ndarray, center: np.ndarray, axis: np.ndarray,
                     height: float, radius: float) -> np.ndarray:
    """Nearest point on the capsule surface to p."""
    local = p - center
    along = np.dot(local, axis)
    half_h = height * 0.5
    if along < -half_h:
        closest = center - axis * half_h
    elif along > half_h:
        closest = center + axis * half_h
    else:
        closest = center + axis * along
    vec = p - closest
    dist = np.linalg.norm(vec)
    if dist < 1e-12:
        perp = np.array([1.0, 0.0, 0.0])
        if abs(axis[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        perp = perp - axis * np.dot(axis, perp)
        perp /= np.linalg.norm(perp) + 1e-12
        return closest + perp * radius
    return closest + (vec / dist) * radius


# ---------------------------------------------------------------------------
# Mesh / geometry helpers
# ---------------------------------------------------------------------------

def _load_obj(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(x) for x in line.split()[1:4]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int) - 1


def _opposite_edges(faces: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
    """For each directed edge (i,j) return the list of (face, k) using it."""
    edge_map: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for fi, f in enumerate(faces):
        for a in range(3):
            i = f[a]
            j = f[(a + 1) % 3]
            key = (i, j)
            if key not in edge_map:
                edge_map[key] = []
            edge_map[key].append(np.array([fi, a]))
    return edge_map


def _build_edge_list(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return unique undirected edges and per-edge opposite vertex pairs."""
    edge_set: set = set()
    for f in faces:
        for a in range(3):
            i = f[a]
            j = f[(a + 1) % 3]
            if i > j:
                i, j = j, i
            edge_set.add((i, j))
    edges = np.asarray(sorted(edge_set), dtype=int)
    edge_face_map = _opposite_edges(faces)
    opp = []
    for i, j in edges:
        o1 = None
        o2 = None
        for (fi, a) in edge_face_map.get((i, j), []):
            f = faces[fi]
            o1 = f[(a + 2) % 3]
            break
        for (fi, a) in edge_face_map.get((j, i), []):
            f = faces[fi]
            o2 = f[(a + 2) % 3]
            break
        if o1 is None or o2 is None:
            o2 = o1 if o1 is not None else i
        opp.append([o1, o2])
    return edges, np.asarray(opp, dtype=int)


def _tet_volume(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                p3: np.ndarray) -> float:
    return abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        self._scene = scene
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        config = scene.config
        self._duration = float(config["duration"])
        self._sample_dt = float(config["sample_dt"])
        self._density = float(config["initial_conditions"]["flesh_density_kg_m3"])

        # Load rest-surface mesh
        mesh_path = Path(__file__).parent.parent / "assets" / "arm_flesh.obj"
        self._rest_pos, self._faces = _load_obj(mesh_path)
        self._n = self._rest_pos.shape[0]
        self._rest_pos = self._rest_pos.astype(float)

        # Build edge/tet topology from the surface mesh
        self._edges, self._edge_opps = _build_edge_list(self._faces)

        # Reference volumes for each edge-based tet element
        self._ref_volume = np.zeros(len(self._edges))
        for ei, (i, j) in enumerate(self._edges):
            k, l = self._edge_opps[ei]
            p0 = self._rest_pos[i]
            p1 = self._rest_pos[j]
            p2 = self._rest_pos[k]
            p3 = self._rest_pos[l] if l != i and l != j and l != k else p2
            self._ref_volume[ei] = max(_tet_volume(p0, p1, p2, p3), 1e-12)

        # Per-node mass inherited from the scene contract
        init_frame = scene.initial_state()
        self._mass = np.asarray(init_frame.mass, dtype=float).copy()

        # Bone kinematics
        forearm_cfg = config["boundary_conditions"]["forearm_bone"]
        self._hinge_center = np.array(forearm_cfg["hinge_center"], dtype=float)
        self._hinge_axis = np.array(forearm_cfg["hinge_axis"], dtype=float)
        self._hinge_axis /= np.linalg.norm(self._hinge_axis) + 1e-12
        self._hinge_start = float(forearm_cfg["start_time"])
        self._hinge_end = float(forearm_cfg["end_time"])
        self._start_angle = float(forearm_cfg["start_angle_degrees"])
        self._end_angle = float(forearm_cfg["end_angle_degrees"])

        # Bone capsule definitions (world, rest)
        self._humerus_center = np.array([-0.36, 0.0, 1.05])
        self._humerus_axis = np.array([0.0, 0.70710678, 0.0, 0.70710678])
        self._humerus_radius = 0.05
        self._humerus_height = 0.62
        self._forearm_center = np.array([0.38, 0.0, 1.05])
        self._forearm_axis_q = np.array([0.0, 0.70710678, 0.0, 0.70710678])
        self._forearm_radius = 0.04
        self._forearm_height = 0.66

        # Initial state
        self._pos = self._rest_pos.copy()
        self._vel = np.zeros_like(self._pos)
        self._time = 0.0

        # Deformation gradient per edge element
        self._F = np.tile(np.eye(3, dtype=float)[None, ...], (len(self._edges), 1, 1))

        # Trajectory field dimensions
        self._g_domain_count = 1
        self._k_materials = 1

        # Elastic material parameters (nearly incompressible soft tissue)
        self._mu = 6.0e3
        self._lam = 3.0e3

        # Diagnostics
        self._diag: Dict[str, Any] = {"substeps": 0}

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    def _forearm_angle(self, t: float) -> float:
        if t <= self._hinge_start:
            return self._start_angle
        if t >= self._hinge_end:
            return self._end_angle
        tau = (t - self._hinge_start) / (self._hinge_end - self._hinge_start)
        s = _smoothstep(tau)
        return self._start_angle + s * (self._end_angle - self._start_angle)

    def _forearm_rotation_matrix(self, t: float) -> np.ndarray:
        angle = np.deg2rad(self._forearm_angle(t))
        c, s = np.cos(angle), np.sin(angle)
        axis = self._hinge_axis
        K = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        return np.eye(3) + s * K + (1.0 - c) * (K @ K)

    def _forearm_transform(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        R = self._forearm_rotation_matrix(t)
        c0 = self._hinge_center
        c_init = self._forearm_center
        center = c0 + R @ (c_init - c0)
        M = _quaternion_to_matrix(self._forearm_axis_q)
        new_M = R @ M
        return center, new_M

    # ------------------------------------------------------------------
    # Elastic forces
    # ------------------------------------------------------------------
    def _elastic_force(self) -> np.ndarray:
        """Return nodal elastic force for a stable Neo-Hookean-like solid."""
        force = np.zeros_like(self._pos)
        mu = self._mu
        lam = self._lam

        for ei, (i, j) in enumerate(self._edges):
            k, l = self._edge_opps[ei]
            p0 = self._pos[i]
            p1 = self._pos[j]
            p2 = self._pos[k]
            # For boundary edges, create a tiny offset to avoid degeneracy
            p3 = self._pos[l] if l != i and l != j and l != k else p2 + np.array([1e-6, 0.0, 0.0])

            Dm = np.stack(
                [
                    self._rest_pos[j] - self._rest_pos[i],
                    self._rest_pos[k] - self._rest_pos[i],
                    self._rest_pos[l] - self._rest_pos[i],
                ],
                axis=1,
            )
            if abs(np.linalg.det(Dm)) < 1e-12:
                continue
            Dm_inv = np.linalg.inv(Dm)
            Ds = np.stack([p1 - p0, p2 - p0, p3 - p0], axis=1)
            F = Ds @ Dm_inv

            # Guard against singular or inverted configurations
            detF = np.linalg.det(F)
            if abs(detF) < 1e-12:
                F = np.eye(3, dtype=float)
                detF = 1.0
            self._F[ei] = F

            J = max(detF, 0.1)
            F_inv_T = np.linalg.inv(F).T
            logJ = np.log(J)
            # Stable Neo-Hookean first Piola stress
            P = mu * (F - F_inv_T) + lam * logJ * F_inv_T
            vol = self._ref_volume[ei]
            H = -vol * P @ Dm_inv.T
            f1 = H[:, 0]
            f2 = H[:, 1]
            f3 = H[:, 2]
            f0 = -(f1 + f2 + f3)
            force[i] += f0
            force[j] += f1
            force[k] += f2
            force[l] += f3
        return force

    # ------------------------------------------------------------------
    # Attachment / collision projection
    # ------------------------------------------------------------------
    def _bone_fields(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (humerus_center, humerus_axis, forearm_center, forearm_axis)."""
        hum_axis = _quaternion_to_matrix(self._humerus_axis)
        fore_center, fore_R = self._forearm_transform(t)
        return self._humerus_center, hum_axis[:, 1], fore_center, fore_R[:, 1]

    def _classify_attachment(self, p: np.ndarray, t: float) -> np.ndarray:
        """Return array of -1/0/1 for humerus/free/forearm attachment."""
        h_center, h_axis, f_center, f_axis = self._bone_fields(t)
        labels = np.zeros(self._n, dtype=int)
        for i in range(self._n):
            d_h = _capsule_sdf(p[i], h_center, h_axis, self._humerus_height, self._humerus_radius)
            d_f = _capsule_sdf(p[i], f_center, f_axis, self._forearm_height, self._forearm_radius)
            # Forearm threshold a bit larger to capture the flesh layer
            if d_f < 0.05:
                labels[i] = 1
            elif d_h < 0.05:
                labels[i] = -1
        return labels

    def _project_to_bone(self, p: np.ndarray, t: float) -> np.ndarray:
        """Project any vertex lying inside a bone to the bone surface."""
        h_center, h_axis, f_center, f_axis = self._bone_fields(t)
        out = p.copy()
        for i in range(self._n):
            d_h = _capsule_sdf(p[i], h_center, h_axis, self._humerus_height, self._humerus_radius)
            d_f = _capsule_sdf(p[i], f_center, f_axis, self._forearm_height, self._forearm_radius)
            if d_h < 0.0:
                out[i] = _capsule_nearest(p[i], h_center, h_axis, self._humerus_height, self._humerus_radius)
            elif d_f < 0.0:
                out[i] = _capsule_nearest(p[i], f_center, f_axis, self._forearm_height, self._forearm_radius)
        return out

    def _drive_attachments(self, t: float) -> None:
        """Kinematically drive vertices near bones."""
        labels = self._classify_attachment(self._pos, t)
        # Forearm rigid-body transform: rotate rest positions about the hinge
        R = self._forearm_rotation_matrix(t)
        for i in range(self._n):
            if labels[i] == 1:
                # forearm-attached: follow the rotating forearm about the hinge
                p0 = self._rest_pos[i] - self._hinge_center
                self._pos[i] = self._hinge_center + R @ p0
                self._vel[i] = 0.0
            elif labels[i] == -1:
                # humerus-attached: fixed
                self._pos[i] = self._rest_pos[i]
                self._vel[i] = 0.0

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------
    def _step(self, dt: float, t: float) -> None:
        f_elastic = self._elastic_force()
        gravity = np.array([0.0, 0.0, -9.81])
        f_ext = self._mass[:, None] * gravity
        # Light Rayleigh-style damping to stabilise explicit integration
        f_damp = -0.5 * self._vel
        f_total = f_elastic + f_ext + f_damp

        self._vel += dt * f_total / self._mass[:, None]
        self._pos += dt * self._vel

        # Enforce bone collision and attachment
        self._pos = self._project_to_bone(self._pos, t + dt)
        self._drive_attachments(t + dt)

        self._time = t + dt

    def _advance_once(self, dt: float) -> None:
        # Fixed internal physics step for stability
        sub_dt = 2.5e-4
        n_sub = max(1, int(np.ceil(dt / sub_dt)))
        actual_sub_dt = dt / n_sub
        for _ in range(n_sub):
            self._step(actual_sub_dt, self._time)
        self._diag["substeps"] += n_sub

    def advance_to(self, time: float) -> StateFrame:
        dt = self._sample_dt
        while self._time + dt / 2.0 < time:
            self._advance_once(dt)
        if time > self._time:
            self._advance_once(time - self._time)
        self._time = time

        # Assemble required trajectory fields
        element_F = self._F.copy()

        f_gravity = (self._mass[:, None] * np.array([0.0, 0.0, -9.81])).sum(axis=0)
        budget_external_force = np.zeros(
            (self._g_domain_count, self._k_materials, 3), dtype=float
        )
        budget_external_force[0, 0, :] = f_gravity

        cumulative_mass_source = np.zeros(self._k_materials, dtype=float)
        reference_force = 1.0e3  # N

        frame = StateFrame(
            time=float(self._time),
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            mass=self._mass.copy(),
            attributes={
                "mass_quantity": self._mass.copy(),
                "rest_position": self._rest_pos.copy(),
            },
            fields={
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": cumulative_mass_source,
                "element_deformation_gradient": element_F,
            },
            scalars={
                "reference_force": float(reference_force),
            },
        )
        return frame

    def diagnostics(self) -> Dict[str, Any]:
        return self._diag.copy()
