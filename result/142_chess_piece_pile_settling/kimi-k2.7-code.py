"""Physically-based rigid body solver for the chess piece pile scene.

The solver uses an explicit time-stepping scheme with a penalty-based
frictional contact model.  Each chess piece is represented by a cloud of
spherical surface samples, so non-convex interlocking is approximated while
still performing genuine 3-D rigid body dynamics.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree

    _HAS_CKD_TREE = True
except Exception:  # pragma: no cover
    _HAS_CKD_TREE = False

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers
# ---------------------------------------------------------------------------

def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from a unit quaternion [x, y, z, w]."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)],
            [2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)],
            [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def _quat_mult(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Quaternion product p * q (both xyzw)."""
    px, py, pz, pw = p
    qx, qy, qz, qw = q
    return np.array(
        [
            pw * qx + px * qw + py * qz - pz * qy,
            pw * qy - px * qz + py * qw + pz * qx,
            pw * qz + px * qy - py * qx + pz * qw,
            pw * qw - px * qx - py * qy - pz * qz,
        ]
    )


# ---------------------------------------------------------------------------
# OBJ loader
# ---------------------------------------------------------------------------

def _parse_obj(path: str):
    """Parse an OBJ file, returning (vertices, faces)."""
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx: List[int] = []
                for token in parts[1:]:
                    # strip texture/normal indices
                    idx.append(int(token.split("/")[0]) - 1)
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


# ---------------------------------------------------------------------------
# Rigid body representation
# ---------------------------------------------------------------------------

class _RigidBody:
    __slots__ = (
        "mass",
        "inv_mass",
        "i_local",
        "inv_i_local",
        "com_local",
        "samples",
        "radius",
        "bounding_radius",
        "x",
        "q",
        "v",
        "w",
        "R",
    )

    def __init__(
        self,
        mass: float,
        i_local: np.ndarray,
        com_local: np.ndarray,
        samples: np.ndarray,
        radius: float,
    ) -> None:
        self.mass = float(mass)
        self.inv_mass = 1.0 / mass
        self.i_local = i_local
        self.inv_i_local = np.linalg.inv(i_local)
        self.com_local = com_local
        self.samples = samples  # (S, 3), relative to COM
        self.radius = radius
        self.bounding_radius = float(np.max(np.linalg.norm(samples, axis=1))) + radius

        self.x = np.zeros(3, dtype=float)
        self.q = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.w = np.zeros(3, dtype=float)
        self.R = np.eye(3, dtype=float)

    def update_rotation_matrix(self) -> None:
        self.R = _quat_to_mat(self.q)

    def world_samples(self) -> np.ndarray:
        return self.x[:, None] + self.R @ self.samples.T  # (3, S)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed

        init = scene.initial_state()
        self.N = int(init.position.shape[0])
        self.mass = np.asarray(init.mass, dtype=float)

        cfg = scene.config
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        contact_cfg = cfg["contact"]
        self.friction = float(contact_cfg["friction"])
        self.restitution = float(contact_cfg["restitution"])
        self.length_scale = float(contact_cfg["length_scale"])

        # Explicit sub-step used inside each requested sample interval.
        self.dt = 0.001
        self.damping = 0.02

        # Map body_id to the asset declared in scene_geometry.yaml.
        asset_for_body = _load_assets_for_bodies(Path("scene_geometry.yaml"))

        bodies: List[_RigidBody] = []
        for i in range(self.N):
            asset_path = asset_for_body[i]
            i_local, com_local, samples, radius = _build_collision_data(
                asset_path, self.mass[i], seed + i
            )
            body = _RigidBody(
                mass=self.mass[i],
                i_local=i_local,
                com_local=com_local,
                samples=samples,
                radius=radius,
            )
            # StateFrame.position is the mesh origin, while the solver tracks the
            # centre of mass.  Output position = x - R * com_local.
            body.x = np.asarray(init.position[i], dtype=float) + com_local
            body.q = np.asarray(init.orientation[i], dtype=float)
            body.q /= np.linalg.norm(body.q)
            body.v = np.asarray(init.velocity[i], dtype=float)
            body.w = np.asarray(init.angular_velocity[i], dtype=float)
            body.update_rotation_matrix()
            bodies.append(body)

        self.bodies = bodies
        self.time = 0.0

    def advance_to(self, time: float) -> StateFrame:
        n_steps = max(0, int(round((time - self.time) / self.dt)))
        for _ in range(n_steps):
            self._step()
        # Guard against floating-point drift in the requested sample time.
        if self.time < time - 1e-12:
            self._step()
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps": int(round(self.time / self.dt)),
            "num_bodies": self.N,
        }

    # -----------------------------------------------------------------------
    # Physics stepping
    # -----------------------------------------------------------------------

    def _step(self) -> None:
        forces = [np.zeros(3, dtype=float) for _ in self.bodies]
        torques = [np.zeros(3, dtype=float) for _ in self.bodies]

        # Gravity, plus a tiny amount of global velocity damping to aid settling.
        for i, b in enumerate(self.bodies):
            forces[i] += b.mass * self.gravity - self.damping * b.v * b.mass
            torques[i] -= self.damping * b.w * np.min(np.diag(b.i_local))

        contacts = self._detect_contacts()
        self._apply_contacts(contacts, forces, torques)

        # Semi-implicit Euler integration.
        for b, f, t in zip(self.bodies, forces, torques):
            b.v += self.dt * f * b.inv_mass
            b.x += self.dt * b.v

            inv_i_world = b.R @ b.inv_i_local @ b.R.T
            i_world = np.linalg.inv(inv_i_world)
            alpha = inv_i_world @ (t - np.cross(b.w, i_world @ b.w))
            b.w += self.dt * alpha

            omega_q = np.array([b.w[0], b.w[1], b.w[2], 0.0], dtype=float)
            qdot = 0.5 * _quat_mult(omega_q, b.q)
            b.q += self.dt * qdot
            b.q /= np.linalg.norm(b.q)
            b.update_rotation_matrix()

        self.time += self.dt

    # -----------------------------------------------------------------------
    # Contact detection and response
    # -----------------------------------------------------------------------

    def _detect_contacts(self) -> List[dict]:
        contacts: List[dict] = []

        # Floor at z = 0, normal pointing upward.
        for i, b in enumerate(self.bodies):
            pts = b.world_samples().T  # (S, 3)
            signed = pts[:, 2] - b.radius
            for idx in np.nonzero(signed < 0.0)[0]:
                contacts.append({
                    "type": "floor",
                    "i": i,
                    "j": -1,
                    "normal": np.array([0.0, 0.0, 1.0], dtype=float),
                    "signed_distance": float(signed[idx]),
                    "point": pts[idx] - np.array([0.0, 0.0, b.radius], dtype=float),
                })

        # Pairwise body-body contacts.
        for i in range(self.N):
            bi = self.bodies[i]
            pi = bi.world_samples().T
            for j in range(i + 1, self.N):
                bj = self.bodies[j]
                if np.linalg.norm(bi.x - bj.x) > bi.bounding_radius + bj.bounding_radius:
                    continue
                pj = bj.world_samples().T
                contacts.extend(_body_body_contacts(bi, i, bj, j, pi, pj))

        return contacts

    def _apply_contacts(
        self,
        contacts: List[dict],
        forces: List[np.ndarray],
        torques: List[np.ndarray],
    ) -> None:
        k = 2.0e3  # penalty stiffness [N/m]
        c = 30.0   # penalty damping [N/(m/s)]
        mu = self.friction

        for con in contacts:
            i = con["i"]
            j = con["j"]
            n = con["normal"]
            d = con["signed_distance"]
            p = con["point"]

            bi = self.bodies[i]
            r_i = p - bi.x
            v_i = bi.v + np.cross(bi.w, r_i)
            if j >= 0:
                bj = self.bodies[j]
                r_j = p - bj.x
                v_j = bj.v + np.cross(bj.w, r_j)
            else:
                bj = None
                r_j = np.zeros(3, dtype=float)
                v_j = np.zeros(3, dtype=float)

            v_rel = v_i - v_j
            v_n = float(np.dot(v_rel, n))
            if d >= 0.0 and v_n >= 0.0:
                continue

            fn_mag = max(0.0, -k * d - c * v_n)
            if fn_mag <= 0.0:
                continue

            # Extra energy loss for fast impacts.
            if v_n < -0.01:
                fn_mag *= (1.0 + self.restitution)

            f_n = n * fn_mag

            v_t = v_rel - n * v_n
            vt_norm = np.linalg.norm(v_t)
            if vt_norm > 1e-4:
                f_t = -v_t / vt_norm * (mu * fn_mag)
            else:
                f_t = np.zeros(3, dtype=float)

            f_total = f_n + f_t

            forces[i] += f_total
            torques[i] += np.cross(r_i, f_total)
            if bj is not None:
                forces[j] -= f_total
                torques[j] -= np.cross(r_j, f_total)

    # -----------------------------------------------------------------------
    # Frame export
    # -----------------------------------------------------------------------

    def _make_frame(self, t: float) -> StateFrame:
        positions = np.empty((self.N, 3), dtype=float)
        velocities = np.empty((self.N, 3), dtype=float)
        orientations = np.empty((self.N, 4), dtype=float)
        angular_velocities = np.empty((self.N, 3), dtype=float)

        for i, b in enumerate(self.bodies):
            orientations[i] = b.q
            R = b.R
            positions[i] = b.x - R @ b.com_local
            velocities[i] = b.v
            angular_velocities[i] = b.w

        external_force = (self.mass[:, None] * self.gravity[None, :]).copy()

        contacts = self._detect_contacts()
        signed_distances = [c["signed_distance"] for c in contacts]

        interaction_force = self._interaction_forces(contacts)

        frame = StateFrame(
            time=float(t),
            position=positions,
            velocity=velocities,
            orientation=orientations,
            angular_velocity=angular_velocities,
            mass=self.mass,
            body_id=np.arange(self.N, dtype=np.int32),
            attributes={},
            fields={
                "external_force": external_force,
                "interaction_force": interaction_force,
                "contact_signed_distance": np.asarray(signed_distances, dtype=float)
                if signed_distances
                else np.zeros((0,), dtype=float),
            },
            scalars={"contact_length_scale": self.length_scale},
        )
        frame.validate()
        return frame

    def _interaction_forces(self, contacts: List[dict]) -> np.ndarray:
        """Return per-pair internal contact forces (body-body only)."""
        k = 2.0e3
        c = 30.0
        mu = self.friction
        pair_forces: dict[Tuple[int, int], np.ndarray] = {}

        for con in contacts:
            if con["type"] != "body":
                continue
            i = con["i"]
            j = con["j"]
            n = con["normal"]
            d = con["signed_distance"]
            p = con["point"]

            bi = self.bodies[i]
            bj = self.bodies[j]
            r_i = p - bi.x
            r_j = p - bj.x
            v_i = bi.v + np.cross(bi.w, r_i)
            v_j = bj.v + np.cross(bj.w, r_j)
            v_rel = v_i - v_j
            v_n = float(np.dot(v_rel, n))
            if d >= 0.0 and v_n >= 0.0:
                continue

            fn_mag = max(0.0, -k * d - c * v_n)
            if fn_mag <= 0.0:
                continue
            if v_n < -0.01:
                fn_mag *= (1.0 + self.restitution)
            f_n = n * fn_mag

            v_t = v_rel - n * v_n
            vt_norm = np.linalg.norm(v_t)
            if vt_norm > 1e-4:
                f_t = -v_t / vt_norm * (mu * fn_mag)
            else:
                f_t = np.zeros(3, dtype=float)

            f_vec = f_n + f_t
            key = (i, j)
            if key not in pair_forces:
                pair_forces[key] = np.zeros(3, dtype=float)
            pair_forces[key] += f_vec

        if pair_forces:
            return np.vstack(list(pair_forces.values()))
        return np.zeros((0, 3), dtype=float)


# ---------------------------------------------------------------------------
# Geometry / collision data helpers
# ---------------------------------------------------------------------------

def _load_assets_for_bodies(geometry_path: Path) -> dict[int, str]:
    """Return a mapping body_id -> asset path from scene_geometry.yaml."""
    text = geometry_path.read_text(encoding="utf-8")
    # Fallback if yaml is unavailable.
    try:
        import yaml

        data = yaml.safe_load(text)
        mapping: dict[int, str] = {}
        for obj in data.get("objects", []):
            binding = obj.get("state_binding")
            if not binding or binding.get("selector") != "body_id":
                continue
            bid = int(binding["value"])
            asset = obj["geometry"]["asset"]
            mapping[bid] = str(asset)
        return mapping
    except Exception:
        return _parse_geometry_yaml(text)


def _parse_geometry_yaml(text: str) -> dict[int, str]:
    """Minimal YAML parser for the body_id -> asset mapping."""
    mapping: dict[int, str] = {}
    current_bid: int | None = None
    current_asset: str | None = None
    in_binding = False
    in_geometry = False

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- id:"):
            if current_bid is not None and current_asset is not None:
                mapping[current_bid] = current_asset
            current_bid = None
            current_asset = None
            in_binding = False
            in_geometry = False
        elif stripped.startswith("state_binding:"):
            in_binding = True
            in_geometry = False
        elif stripped.startswith("geometry:"):
            in_binding = False
            in_geometry = True
        elif stripped.startswith("value:") and in_binding:
            current_bid = int(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("asset:") and in_geometry:
            current_asset = stripped.split(":", 1)[1].strip().strip('"\'')

    if current_bid is not None and current_asset is not None:
        mapping[current_bid] = current_asset
    return mapping


def _build_collision_data(
    asset_path: str, mass: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (inertia_about_com, com_local, surface_samples_relative_to_com, sphere_radius)."""
    verts, faces = _parse_obj(asset_path)
    if verts.size == 0:
        inertia = np.eye(3, dtype=float) * 1e-3
        return inertia, np.zeros(3, dtype=float), np.zeros((1, 3), dtype=float), 0.02

    # Centre of mass via signed tetrahedra volumes from the origin.
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1, v2)
    vols = np.einsum("ij,ij->i", v0, cross) / 6.0
    total_vol = np.sum(vols)
    if abs(total_vol) < 1e-12:
        com = np.mean(verts, axis=0)
        total_vol = 1.0
    else:
        com = np.sum(vols[:, None] * (v0 + v1 + v2) / 4.0, axis=0) / total_vol

    # Inertia by sampling inside tetrahedra (com, face).
    rng = np.random.default_rng(abs(int(seed)) % (2 ** 31))
    samples_per_tet = 8
    all_pts: List[np.ndarray] = []
    all_weights: List[float] = []

    for f_idx in range(faces.shape[0]):
        a = com
        b = verts[faces[f_idx, 0]]
        c_ = verts[faces[f_idx, 1]]
        d = verts[faces[f_idx, 2]]
        mat = np.array([b - a, c_ - a, d - a]).T
        vol = np.linalg.det(mat) / 6.0
        if vol <= 0.0:
            continue
        u = rng.random((samples_per_tet, 3))
        s = np.cbrt(u[:, 0])
        t = np.cbrt(u[:, 1])
        r1 = 1.0 - s
        r2 = s * (1.0 - t)
        r3 = s * t * (1.0 - u[:, 2])
        r4 = s * t * u[:, 2]
        pts = (
            r1[:, None] * a
            + r2[:, None] * b
            + r3[:, None] * c_
            + r4[:, None] * d
        )
        all_pts.append(pts)
        all_weights.extend([vol / samples_per_tet] * samples_per_tet)

    if all_pts:
        pts = np.vstack(all_pts)
        weights = np.asarray(all_weights, dtype=float)
        weights /= weights.sum()
        com2 = np.sum(weights[:, None] * pts, axis=0)
        rel = pts - com2
        m_i = weights * mass
        rr = rel[:, None, :] * rel[:, :, None]
        r_sq = np.sum(rel ** 2, axis=1)
        inertia = np.sum(
            m_i[:, None, None]
            * (r_sq[:, None, None] * np.eye(3)[None, :, :] - rr),
            axis=0,
        )
        inertia = np.diag(np.diag(inertia)) + 0.001 * np.eye(3)
    else:
        com2 = com.copy()
        inertia = np.eye(3, dtype=float) * 1e-3

    # Surface samples used for collision detection.
    max_samples = 120
    if len(verts) <= max_samples:
        surface = verts - com2
    else:
        idx = np.round(np.linspace(0, len(verts) - 1, max_samples)).astype(int)
        surface = verts[idx] - com2

    radius = 0.018
    return inertia, com2, surface, radius


def _body_body_contacts(
    bi: _RigidBody,
    i: int,
    bj: _RigidBody,
    j: int,
    pi: np.ndarray,
    pj: np.ndarray,
) -> List[dict]:
    """Detect sphere-cloud contacts between two rigid bodies."""
    contacts: List[dict] = []
    r_sum = bi.radius + bj.radius

    if _HAS_CKD_TREE:
        tree_i = cKDTree(pi)
        tree_j = cKDTree(pj)
        pairs = tree_i.sparse_distance_matrix(tree_j, r_sum, output_type="ndarray")
        if pairs.size == 0:
            return contacts
        di = pairs["i"]
        dj = pairs["j"]
        dists = pairs["v"]
        valid = dists > 1e-12
        di = di[valid]
        dj = dj[valid]
        dists = dists[valid]
        normals = (pi[di] - pj[dj]) / dists[:, None]
        signed = dists - r_sum
        points = pi[di] - normals * bi.radius
    else:
        diff = pj[None, :, :] - pi[:, None, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = (dists > 1e-12) & (dists < r_sum)
        if not np.any(mask):
            return contacts
        di, dj = np.nonzero(mask)
        dists = dists[di, dj]
        normals = -diff[di, dj] / dists[:, None]
        signed = dists - r_sum
        points = pi[di] - normals * bi.radius

    for a in range(len(di)):
        contacts.append({
            "type": "body",
            "i": i,
            "j": j,
            "normal": normals[a],
            "signed_distance": float(signed[a]),
            "point": points[a],
        })
    return contacts
