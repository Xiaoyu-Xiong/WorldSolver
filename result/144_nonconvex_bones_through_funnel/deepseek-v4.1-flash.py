"""Physically based rigid-body solver: nonconvex bone-shaped bodies through a funnel.

The solver implements its own rigid-body model with a real time-stepping scheme:

* every bone mesh is approximated by a compact union of spheres derived directly
  from the frozen ``assets/bone.obj`` mesh (medial sphere cover dilated by a small
  inflation length), so the nonconvex dumbbell shape, its multi-point contacts and
  its jamming behaviour are preserved;
* bone/bone contacts are sphere/sphere, bone/static contacts are sphere/triangle
  against the actual frozen ``assets/funnel.obj`` mesh and sphere/box against the
  frozen floor box;
* contacts are resolved with a sequential-impulse (projected Gauss-Seidel) solver
  with warm starting, Coulomb friction, restitution and Baumgarte position bias,
  integrated with semi-implicit Euler substeps.

All quantities are deterministic and finite; no animation, keyframes or
precomputed trajectories are used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

try:  # optional acceleration used once, for the initial sphere cover
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


# --------------------------------------------------------------------------- #
# quaternion helpers (xyzw order, matching the frozen scene contract)
# --------------------------------------------------------------------------- #
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(np.isfinite(n) & (n > 1e-12), n, 1.0)
    out = q / n
    return np.where(np.isfinite(out), out, np.array([0.0, 0.0, 0.0, 1.0]))


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    s = 2.0 / max(x * x + y * y + z * z + w * w, 1e-12)
    xx, yy, zz = s * x * x, s * y * y, s * z * z
    xy, xz, yz = s * x * y, s * x * z, s * y * z
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    return np.array(
        [
            [1.0 - yy - zz, xy - wz, xz + wy],
            [xy + wz, 1.0 - xx - zz, yz - wx],
            [xz - wy, yz + wx, 1.0 - xx - yy],
        ]
    )


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return _quat_normalize(np.array([x, y, z, w]))


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# mesh utilities
# --------------------------------------------------------------------------- #
def _load_obj(path: Path):
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line[0] == "#":
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            idx = []
            for token in parts[1:]:
                value = int(token.split("/")[0])
                idx.append(value - 1 if value > 0 else len(verts) + value)
            for k in range(1, len(idx) - 1):
                faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def _face_normals(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(length, 1e-12)


def _mesh_inertia(v: np.ndarray, f: np.ndarray, mass: float):
    """Volume, centre of mass and inertia tensor of a closed triangle mesh."""
    p = v[f]  # (M, 3, 3)
    vol = np.einsum("mi,mi->m", p[:, 0], np.cross(p[:, 1], p[:, 2])) / 6.0
    total = float(vol.sum())
    if abs(total) < 1e-12:
        total = 1e-12
    s = p.sum(axis=1)  # (M, 3)
    first = (vol[:, None] * s / 4.0).sum(axis=0)
    com = first / total
    m2 = (vol / 20.0)[:, None, None] * (
        np.einsum("mi,mj->mij", s, s) + np.einsum("mki,mkj->mij", p, p)
    )
    m2 = m2.sum(axis=0)
    rho = mass / total
    inertia_o = rho * (np.trace(m2) * np.eye(3) - m2)
    inertia = inertia_o - mass * (
        float(com @ com) * np.eye(3) - np.outer(com, com)
    )
    return total, com, inertia, rho


def _nearest(samples: np.ndarray, queries: np.ndarray):
    if cKDTree is not None:
        tree = cKDTree(samples)
        dist, idx = tree.query(queries, k=1)
        return np.asarray(dist, dtype=float), np.asarray(idx, dtype=int)
    dist = np.empty(len(queries))
    idx = np.empty(len(queries), dtype=int)
    for start in range(0, len(queries), 256):  # pragma: no cover
        block = queries[start : start + 256]
        d2 = ((block[:, None, :] - samples[None, :, :]) ** 2).sum(axis=2)
        idx[start : start + 256] = d2.argmin(axis=1)
        dist[start : start + 256] = np.sqrt(d2.min(axis=1))
    return dist, idx


def _point_triangle_distances(points: np.ndarray, v0, e1, e2) -> np.ndarray:
    """Exact distance from every point to every triangle, computed in chunks."""
    out = np.empty((len(points), len(v0)))
    for start in range(0, len(points), 192):
        block = points[start : start + 192]
        repeat = len(block)
        p = np.repeat(block, len(v0), axis=0)
        a = np.tile(v0, (repeat, 1))
        b = np.tile(e1, (repeat, 1))
        c = np.tile(e2, (repeat, 1))
        cp = _closest_points_on_triangles(p, a, b, c)
        out[start : start + 192] = np.linalg.norm(p - cp, axis=1).reshape(repeat, len(v0))
    return out


def _inside_mesh(points: np.ndarray, v0, e1, e2, direction) -> np.ndarray:
    """Exact ray-parity inside test along a generic direction."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    counts = np.zeros(len(points), dtype=np.int64)
    for start in range(0, len(points), 128):
        block = points[start : start + 128]
        repeat = len(block)
        o = np.repeat(block, len(v0), axis=0)
        a = np.tile(v0, (repeat, 1))
        b = np.tile(e1, (repeat, 1))
        c = np.tile(e2, (repeat, 1))
        dvec = np.broadcast_to(d, o.shape)
        tvec = o - a
        pvec = np.cross(dvec, c)
        det = np.einsum("ij,ij->i", b, pvec)
        ok = np.abs(det) > 1e-15
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        qvec = np.cross(tvec, b)
        v = np.einsum("ij,j->i", qvec, d) * inv
        t = np.einsum("ij,ij->i", c, qvec) * inv
        hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-9)
        counts[start : start + 128] = hit.reshape(repeat, len(v0)).sum(axis=1)
    return (counts % 2) == 1


def _sphere_cover(
    v: np.ndarray, f: np.ndarray, spacing: float, inflate: float, max_spheres: int
):
    """Compact sphere union that follows a closed nonconvex mesh.

    Candidate spheres are the medial spheres of a uniform interior grid: a node
    inside the mesh becomes a sphere whose radius is its distance to the surface
    plus ``inflate``, so the union never leaves the mesh dilated by ``inflate``.
    Candidates are selected by a greedy cover of densely sampled surface points,
    which keeps the union small while tracking every convex and concave feature.
    """
    fn = _face_normals(v, f)
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    edges = np.concatenate(
        [
            (v[f[:, 0]] + v[f[:, 1]]) / 2.0,
            (v[f[:, 1]] + v[f[:, 2]]) / 2.0,
            (v[f[:, 2]] + v[f[:, 0]]) / 2.0,
        ],
        axis=0,
    )
    samples = np.concatenate([v, v[f].mean(axis=1), edges], axis=0)
    snormals = np.concatenate([vn, fn, np.repeat(fn, 3, axis=0)], axis=0)

    lo = v.min(axis=0)
    hi = v.max(axis=0)
    axes = []
    for d in range(3):
        axes.append(np.arange(lo[d] + 0.5 * spacing, hi[d] + 1e-9, spacing))
        if axes[-1].size == 0:
            axes[-1] = np.array([0.5 * (lo[d] + hi[d])])
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    # candidate medial spheres: interior grid nodes plus inward offsets of the
    # surface samples (which reach into thin limbs the grid may step over)
    candidates = np.concatenate([grid, samples - snormals * (2.5 * inflate)], axis=0)
    tri_v0 = v[f[:, 0]]
    tri_e1 = v[f[:, 1]] - tri_v0
    tri_e2 = v[f[:, 2]] - tri_v0
    inside = _inside_mesh(candidates, tri_v0, tri_e1, tri_e2, (0.3121, 0.5317, 0.7871))
    nodes = candidates[inside]
    if len(nodes) == 0:
        nodes = v.mean(axis=0, keepdims=True)
    # exact distance to the mesh keeps every sphere inside mesh + inflate
    radii = _point_triangle_distances(nodes, tri_v0, tri_e1, tri_e2).min(axis=1) + inflate

    reach = np.empty((len(nodes), len(samples)), dtype=bool)
    for start in range(0, len(samples), 512):
        block = samples[start : start + 512]
        d2 = ((nodes[:, None, :] - block[None, :, :]) ** 2).sum(axis=2)
        reach[:, start : start + 512] = d2 <= (radii[:, None] ** 2)

    uncovered = np.ones(len(samples), dtype=bool)
    chosen: list[int] = []
    for _ in range(max_spheres):
        count = (reach & uncovered[None, :]).sum(axis=1)
        best = int(np.argmax(count))
        if count[best] == 0:
            break
        chosen.append(best)
        uncovered &= ~reach[best]
        if not uncovered.any():
            break

    centres = [nodes[i] for i in chosen]
    radii_out = [float(radii[i]) for i in chosen]
    leftover = np.nonzero(uncovered)[0]
    room = max_spheres - len(centres)
    if room > 0 and leftover.size:
        stride = max(1, int(np.ceil(len(leftover) / room)))
        extra = samples[leftover[::stride][:room]] - snormals[leftover[::stride][:room]] * (
            0.5 * inflate
        )
        extra_dist = _point_triangle_distances(extra, tri_v0, tri_e1, tri_e2).min(axis=1)
        for k in range(len(extra)):
            centres.append(extra[k])
            radii_out.append(float(extra_dist[k]) + inflate)

    return np.asarray(centres, dtype=float), np.asarray(radii_out, dtype=float)


def _closest_points_on_triangles(p: np.ndarray, v0: np.ndarray, e1: np.ndarray, e2: np.ndarray):
    """Vectorised closest point on triangles (Ericson, Real-Time Collision Detection)."""
    d = v0 - p
    a = np.einsum("ij,ij->i", e1, e1)
    b = np.einsum("ij,ij->i", e1, e2)
    c = np.einsum("ij,ij->i", e2, e2)
    d1 = np.einsum("ij,ij->i", e1, d)
    d2 = np.einsum("ij,ij->i", e2, d)
    denom = a * c - b * b
    safe = np.abs(denom) > 1e-16
    inv = np.where(safe, 1.0 / np.where(safe, denom, 1.0), 0.0)
    s = (b * d2 - c * d1) * inv
    t = (b * d1 - a * d2) * inv

    sa = np.maximum(a, 1e-16)
    sc = np.maximum(c, 1e-16)
    den_bc = np.maximum(a - 2.0 * b + c, 1e-16)

    # candidate regions
    s_ac = np.zeros_like(s)
    t_ac = np.clip(-d2 / sc, 0.0, 1.0)
    s_ab = np.clip(-d1 / sa, 0.0, 1.0)
    t_ab = np.zeros_like(t)
    s_bc = np.clip((c + d1 - d2) / den_bc, 0.0, 1.0)
    t_bc = 1.0 - s_bc

    vertex_a = np.zeros_like(s)
    t_vertex_a = np.zeros_like(s)

    outside = (s + t) > 1.0
    neg_s = s < 0.0
    neg_t = t < 0.0

    s_out = np.where(
        outside,
        s_bc,
        np.where(neg_s, np.where(neg_t, vertex_a, s_ac), np.where(neg_t, s_ab, s)),
    )
    t_out = np.where(
        outside,
        t_bc,
        np.where(neg_s, np.where(neg_t, t_vertex_a, t_ac), np.where(neg_t, t_ab, t)),
    )
    cp = v0 + s_out[:, None] * e1 + t_out[:, None] * e2
    return cp


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
class BenchmarkSolver:
    """Deterministic rigid-body solver with nonconvex mesh contact geometry."""

    # time stepping / contact parameters (physical, not tuned to a script)
    dt = 0.002
    iterations = 20
    narrow = 0.012  # contact detection band (m)
    near_gap = 0.030  # exported "near contact" band (m)
    slop = 0.0022  # allowed overlap before positional bias (m)
    beta = 0.22  # Baumgarte factor
    max_bias = 0.6  # cap on bias speed (m/s)
    rest_threshold = 0.35  # minimum impact speed for restitution
    proj = 0.2  # position projection fraction
    max_proj = 0.004  # max projection per step (m)
    warm_n = 0.9
    warm_t = 0.8
    max_speed = 25.0
    max_spin = 60.0

    def __init__(self) -> None:
        self.initialised = False
        self.nb = 0
        self.ns = 0
        self.time = 0.0
        self.warm: dict[int, np.ndarray] = {}
        self.diagnostics_values: dict[str, object] = {}

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.seed = int(seed)
        gravity = cfg.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        contact_cfg = cfg.get("contact", {}) or {}
        self.friction = float(contact_cfg.get("friction", 0.55))
        self.restitution = float(contact_cfg.get("restitution", 0.12))
        self.length_scale = float(contact_cfg.get("length_scale", 0.02))
        # the sphere cover is the mesh dilated by at most ``inflate``: contacts
        # are then never missed, only detected slightly early (bounded error)
        self.inflate = 1.1 * self.length_scale

        bodies = list(cfg.get("bodies", []))
        self.nb = len(bodies)
        if self.nb == 0:
            raise ValueError("scene defines no dynamic bodies")
        self.body_id = np.arange(self.nb, dtype=np.int32)
        self.mass = np.asarray([float(b.get("mass", 1.0)) for b in bodies], dtype=float)
        self.mass = np.maximum(self.mass, 1e-9)
        self.inv_mass = 1.0 / self.mass

        # ------------------------------------------------------- static scene
        geometry = cfg.get("scene_geometry") or {}
        objects = geometry.get("objects", []) if isinstance(geometry, dict) else []
        self.floor_c = None
        self.floor_h = None
        tri_v0 = []
        tri_e1 = []
        tri_e2 = []
        tri_n = []
        for obj in objects:
            spec = obj.get("geometry") or {}
            kind = spec.get("type")
            translation = (obj.get("transform") or {}).get("translation", [0.0, 0.0, 0.0])
            if obj.get("motion") != "static":
                continue
            if kind == "box":
                self.floor_c = np.asarray(translation, dtype=float)
                self.floor_h = np.asarray(spec.get("half_extents", [1.0, 1.0, 0.1]), dtype=float)
            elif kind == "triangle_mesh" and spec.get("asset"):
                verts, faces = _load_obj(self._asset_path(spec["asset"]))
                verts = self._apply_transform(verts, obj.get("transform") or {}, spec)
                v0 = verts[faces[:, 0]]
                e1 = verts[faces[:, 1]] - v0
                e2 = verts[faces[:, 2]] - v0
                normals = np.cross(e1, e2)
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
                tri_v0.append(v0)
                tri_e1.append(e1)
                tri_e2.append(e2)
                tri_n.append(normals)
        if tri_v0:
            self.tri_v0 = np.concatenate(tri_v0, axis=0)
            self.tri_e1 = np.concatenate(tri_e1, axis=0)
            self.tri_e2 = np.concatenate(tri_e2, axis=0)
            self.tri_n = np.concatenate(tri_n, axis=0)
            self.tri_c = self.tri_v0 + (self.tri_e1 + self.tri_e2) / 3.0
            self.tri_r = np.maximum(
                np.linalg.norm(self.tri_e1, axis=1), np.linalg.norm(self.tri_e2, axis=1)
            )
            self.tri_r = np.maximum(self.tri_r, np.linalg.norm(self.tri_e1 - self.tri_e2, axis=1))
            corners = np.concatenate(
                [self.tri_v0, self.tri_v0 + self.tri_e1, self.tri_v0 + self.tri_e2,
                 self.tri_v0 + self.tri_e1 + self.tri_e2],
                axis=0,
            )
            self.tri_lo = corners.min(axis=0)
            self.tri_hi = corners.max(axis=0)
        else:
            self.tri_v0 = None

        # ----------------------------------------------------------- own mesh
        asset = self._bone_asset(objects, bodies)
        verts, faces = _load_obj(self._asset_path(asset))
        volume, com, inertia, _rho = _mesh_inertia(verts, faces, float(self.mass[0]))
        self.bone_V = verts - com
        self.bone_F = faces
        self.inertia = inertia
        self.inv_inertia = np.linalg.inv(inertia + 1e-12 * np.eye(3))
        self.proxy_c, self.proxy_r = _sphere_cover(
            self.bone_V, faces, spacing=0.035, inflate=self.inflate, max_spheres=132
        )
        self.ns = len(self.proxy_c)
        self.bound_r = float(np.max(np.linalg.norm(self.proxy_c, axis=1) + self.proxy_r)) + 1e-6

        # ------------------------------------------------------ initial state
        self.pos, self.quat = self._initial_layout()
        self.rot = np.stack([_quat_to_mat(q) for q in self.quat], axis=0)
        self.vel = np.zeros((self.nb, 3))
        self.omega = np.zeros((self.nb, 3))
        self.invI_world = np.stack(
            [self.rot[i] @ self.inv_inertia @ self.rot[i].T for i in range(self.nb)], axis=0
        )
        self.time = 0.0
        self.step_index = 0
        self.warm.clear()
        self.initialised = True
        self.diagnostics_values = {
            "bodies": self.nb,
            "spheres_per_body": self.ns,
            "funnel_triangles": 0 if self.tri_v0 is None else int(len(self.tri_v0)),
            "dt": self.dt,
        }

    def _apply_transform(self, verts, transform, spec):
        scale = np.asarray(transform.get("scale", [1.0, 1.0, 1.0]), dtype=float)
        rotation = np.asarray(transform.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float)
        translation = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), dtype=float)
        matrix = _quat_to_mat(_quat_normalize(rotation))
        return (verts * scale) @ matrix.T + translation

    def _asset_path(self, asset: str) -> Path:
        here = Path(__file__).resolve().parent
        candidates = []
        candidate = Path(asset)
        if candidate.is_absolute():
            candidates.append(candidate)
        for base in (Path.cwd(), here.parent, here, Path.cwd().parent, here.parent.parent):
            candidates.append(base / asset)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"scene asset not found: {asset}")

    @staticmethod
    def _bone_asset(objects, bodies):
        for obj in objects:
            spec = obj.get("geometry") or {}
            if spec.get("type") == "triangle_mesh" and obj.get("motion") != "static":
                return spec.get("asset")
        for obj in objects:
            spec = obj.get("geometry") or {}
            if spec.get("type") == "triangle_mesh" and str(obj.get("id", "")).startswith("bone"):
                return spec.get("asset")
        raise FileNotFoundError("no bone mesh declared in the scene geometry")

    # ------------------------------------------------------- initial packing
    def _initial_layout(self):
        rng = np.random.default_rng(self.seed * 7919 + 104729)
        base = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        pos = np.zeros((self.nb, 3))
        quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.nb, 1))
        placed = []
        for k in range(self.nb):
            layer = k // 6
            column = k % 6
            x0 = -1.25 + 0.5 * column
            y0 = 0.30 if ((column + layer) % 2 == 0) else -0.30
            z0 = 2.00 + 0.62 * layer
            chosen = None
            for _attempt in range(90):
                p = np.array(
                    [
                        x0 + rng.uniform(-0.02, 0.02),
                        y0 + rng.uniform(-0.06, 0.06),
                        z0 + rng.uniform(-0.015, 0.015),
                    ]
                )
                roll = math.radians(rng.uniform(-22.0, 22.0))
                tilt = math.radians(rng.uniform(-7.0, 7.0))
                yaw = math.radians(rng.uniform(-6.0, 6.0))
                flip = int(rng.integers(0, 2))
                rot = (
                    _rot_z(yaw)
                    @ _rot_x(tilt)
                    @ base
                    @ _rot_z(roll)
                    @ _rot_x(math.pi * flip)
                )
                q = _mat_to_quat(rot)
                centres, radii = self._proxy_world(rot, p)
                lowest = float((self.bone_V @ rot.T + p)[:, 2].min())
                if lowest < 1.55:
                    continue
                if not self._clear_bodies(centres, radii, placed):
                    continue
                if not self._clear_static(centres, radii):
                    continue
                chosen = (p, q, centres, radii)
                break
            if chosen is None:
                rot = base
                q = _mat_to_quat(rot)
                p = np.array([x0, y0, z0])
                centres, radii = self._proxy_world(rot, p)
                chosen = (p, q, centres, radii)
            pos[k] = chosen[0]
            quat[k] = chosen[1]
            centres, radii = chosen[2], chosen[3]
            placed.append(
                (
                    centres,
                    radii,
                    (centres - radii[:, None]).min(axis=0),
                    (centres + radii[:, None]).max(axis=0),
                )
            )
        return pos, quat

    def _proxy_world(self, rot, p):
        return self.proxy_c @ rot.T + p, self.proxy_r

    def _clear_bodies(self, centres, radii, placed):
        low = (centres - radii[:, None]).min(axis=0)
        high = (centres + radii[:, None]).max(axis=0)
        for other_c, other_r, other_lo, other_hi in placed:
            if np.any(low > other_hi) or np.any(other_lo > high):
                continue
            d = centres[:, None, :] - other_c[None, :, :]
            dist = np.sqrt(np.maximum(np.einsum("ijk,ijk->ij", d, d), 0.0))
            if np.any(dist - (radii[:, None] + other_r[None, :]) < 0.008):
                return False
        return True

    def _clear_static(self, centres, radii):
        if self.floor_c is not None:
            rel = centres - self.floor_c[None, :]
            q = np.abs(rel) - self.floor_h[None, :]
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            sdf = outside + np.minimum(q.max(axis=1), 0.0)
            if np.any(sdf - radii < 0.008):
                return False
        if self.tri_v0 is not None:
            box = np.maximum(
                np.maximum(self.tri_lo[None, :] - centres, centres - self.tri_hi[None, :]), 0.0
            )
            close = np.nonzero(np.linalg.norm(box, axis=1) < radii + 0.02)[0]
            for i in close:
                d = self.tri_c - centres[i]
                dist = np.linalg.norm(d, axis=1)
                cand = np.nonzero(dist < self.tri_r + radii[i] + 0.05)[0]
                if cand.size == 0:
                    continue
                gap = self._sphere_triangles_gap(
                    np.repeat(centres[i][None, :], cand.size, axis=0),
                    np.full(cand.size, radii[i]),
                    cand,
                )
                if np.any(gap < 0.008):
                    return False
        return True

    def _sphere_triangles_gap(self, centres, radii, cand):
        v0 = self.tri_v0[cand]
        e1 = self.tri_e1[cand]
        e2 = self.tri_e2[cand]
        normals = self.tri_n[cand]
        cp = _closest_points_on_triangles(centres, v0, e1, e2)
        diff = centres - cp
        dist = np.linalg.norm(diff, axis=1)
        sign = np.sign(np.einsum("ij,ij->i", diff, normals))
        sign = np.where(sign == 0.0, 1.0, sign)
        return sign * dist - radii

    # ---------------------------------------------------------- contact phase
    def _sphere_world(self):
        return np.einsum("nij,sj->nsi", self.rot, self.proxy_c) + self.pos[:, None, :]

    def _contact_list(self):
        nb = self.nb
        ns = self.ns
        cw = self._sphere_world()
        radii = self.proxy_r
        low = (cw - radii[None, :, None]).min(axis=1)
        high = (cw + radii[None, :, None]).max(axis=1)
        ia: list[np.ndarray] = []
        ib: list[np.ndarray] = []
        nrm: list[np.ndarray] = []
        pnt: list[np.ndarray] = []
        gap: list[np.ndarray] = []
        key: list[np.ndarray] = []

        for i in range(nb):
            for j in range(i + 1, nb):
                if np.any(low[i] > high[j]) or np.any(low[j] > high[i]):
                    continue
                if np.linalg.norm(self.pos[i] - self.pos[j]) > 2.0 * self.bound_r:
                    continue
                ca = cw[i]
                cb = cw[j]
                a2 = np.einsum("ij,ij->i", ca, ca)
                b2 = np.einsum("ij,ij->i", cb, cb)
                d2 = a2[:, None] + b2[None, :] - 2.0 * (ca @ cb.T)
                np.maximum(d2, 0.0, out=d2)
                thr = radii[:, None] + radii[None, :] + self.narrow
                mask = d2 < thr * thr
                if not mask.any():
                    continue
                ii, jj = np.nonzero(mask)
                delta = ca[ii] - cb[jj]
                ds = np.sqrt(np.einsum("ij,ij->i", delta, delta))
                nn = delta / np.maximum(ds, 1e-9)[:, None]
                bad = ds < 1e-9
                if bad.any():
                    nn[bad] = np.array([0.0, 0.0, 1.0])
                pp = 0.5 * (ca[ii] + cb[jj])
                ia.append(np.full(len(ii), i))
                ib.append(np.full(len(ii), j))
                nrm.append(nn)
                pnt.append(pp)
                gap.append(ds - (radii[ii] + radii[jj]))
                key.append((i * nb + j) * ns * ns + ii * ns + jj)

        if self.floor_c is not None:
            rel = cw - self.floor_c[None, None, :]
            q = np.abs(rel) - self.floor_h[None, None, :]
            outside = np.sqrt(
                np.maximum(np.einsum("ijk,ijk->ij", np.maximum(q, 0.0), np.maximum(q, 0.0)), 0.0)
            )
            sdf = outside + np.minimum(q.max(axis=-1), 0.0)
            gap_all = sdf - radii[None, :]
            mask = gap_all < self.narrow
            if mask.any():
                bi, si = np.nonzero(mask)
                cc = cw[bi, si]
                relb = rel[bi, si]
                qb = q[bi, si]
                outb = outside[bi, si]
                closest = self.floor_c[None, :] + np.clip(relb, -self.floor_h, self.floor_h)
                nn = np.zeros_like(cc)
                good = outb > 1e-9
                if good.any():
                    nn[good] = (cc[good] - closest[good]) / outb[good, None]
                bad = ~good
                if bad.any():
                    depth = self.floor_h[None, :] - np.abs(relb[bad])
                    axis = np.argmin(depth, axis=1)
                    sign = np.sign(relb[bad, axis])
                    sign = np.where(sign == 0.0, 1.0, sign)
                    nn[bad] = 0.0
                    nn[bad, axis] = sign
                    closest[bad] = cc[bad] - nn[bad] * depth[
                        np.arange(len(axis)), axis
                    ][:, None]
                pp = closest
                ia.append(bi)
                ib.append(np.full(len(bi), -1))
                nrm.append(nn)
                pnt.append(pp)
                gap.append(gap_all[bi, si])
                key.append(-(1 + bi * self.ns + si))

        if self.tri_v0 is not None:
            for i in range(nb):
                dc = self.tri_c - self.pos[i]
                dist_c = np.linalg.norm(dc, axis=1)
                cand = np.nonzero(dist_c < self.tri_r + self.bound_r + self.narrow)[0]
                if cand.size == 0:
                    continue
                c = cw[i]
                # plane-distance prefilter: the gap of a sphere to a triangle is
                # never smaller than its distance to the triangle plane.
                diff = c[:, None, :] - self.tri_v0[None, cand, :]
                plane = np.abs(np.einsum("smk,mk->sm", diff, self.tri_n[cand]))
                est = plane - radii[:, None]
                si, ti = np.nonzero(est < self.narrow)
                if si.size == 0:
                    continue
                tri = cand[ti]
                cc = c[si]
                g = self._sphere_triangles_gap(cc, radii[si], tri)
                keep = g < self.narrow
                if not keep.any():
                    continue
                si = si[keep]
                tri = tri[keep]
                cc = cc[keep]
                gg = g[keep]
                v0 = self.tri_v0[tri]
                e1 = self.tri_e1[tri]
                e2 = self.tri_e2[tri]
                normals = self.tri_n[tri]
                cp = _closest_points_on_triangles(cc, v0, e1, e2)
                diff = cc - cp
                dist = np.linalg.norm(diff, axis=1)
                dot = np.einsum("ij,ij->i", diff, normals)
                nn = normals.copy()
                out = dot > 0.0
                ok = out & (dist > 1e-9)
                if ok.any():
                    nn[ok] = diff[ok] / dist[ok, None]
                ia.append(np.full(len(si), i))
                ib.append(np.full(len(si), -2 - tri))
                nrm.append(nn)
                pnt.append(cp)
                gap.append(gg)
                key.append(
                    -(1 + self.nb * self.ns)
                    - (i * self.ns * len(self.tri_v0) + si * len(self.tri_v0) + tri)
                )

        if not ia:
            return None
        ia_arr = np.concatenate(ia).astype(np.int64)
        ib_arr = np.concatenate(ib).astype(np.int64)
        n_arr = np.concatenate(nrm, axis=0)
        p_arr = np.concatenate(pnt, axis=0)
        gap_arr = np.concatenate(gap)
        key_arr = np.concatenate(key).astype(np.int64)
        inv = np.linalg.norm(n_arr, axis=1)
        n_arr = n_arr / np.maximum(inv, 1e-12)[:, None]
        ra = p_arr - self.pos[ia_arr]
        rb = np.zeros_like(ra)
        dyn = ib_arr >= 0
        if dyn.any():
            rb[dyn] = p_arr[dyn] - self.pos[ib_arr[dyn]]
        pen = np.maximum(-gap_arr, 0.0)
        return {
            "ia": ia_arr,
            "ib": ib_arr,
            "n": n_arr,
            "p": p_arr,
            "ra": ra,
            "rb": rb,
            "gap": gap_arr,
            "pen": pen,
            "key": key_arr,
        }

    # ---------------------------------------------------------------- solving
    def _accum(self, idx, weights):
        out = np.zeros((self.nb, 3))
        for k in range(3):
            out[:, k] = np.bincount(idx, weights=weights[:, k], minlength=self.nb)
        return out

    def _apply_side(self, idx, impulses, offsets):
        if len(idx) == 0:
            return
        self.vel += self._accum(idx, impulses * self.inv_mass[idx][:, None])
        tau = np.cross(offsets, impulses)
        spin = np.einsum("mij,mj->mi", self.invI_world[idx], tau)
        self.omega += self._accum(idx, spin)

    def _eff_mass(self, direction, offset, idx):
        t = np.cross(offset, direction)
        w = np.einsum("mij,mj->mi", self.invI_world[idx], t)
        return self.inv_mass[idx] + np.einsum("mi,mi->m", np.cross(w, offset), direction)

    def _point_velocity(self, idx, offset):
        return self.vel[idx] + np.cross(self.omega[idx], offset)

    def _solve(self, con, dt, agg):
        ia = con["ia"]
        ib = con["ib"]
        n = con["n"]
        ra = con["ra"]
        rb = con["rb"]
        key = con["key"]
        pen = con["pen"]
        count = len(ia)
        dyn = ib >= 0
        idx_all = np.concatenate([ia, ib[dyn]])
        off_all = np.concatenate([ra, rb[dyn]], axis=0)

        jn = np.zeros(count)
        jt = np.zeros((count, 3))
        for k in range(count):
            prev = self.warm.get(int(key[k]))
            if prev is not None and int(prev[4]) == self.step_index - 1:
                jn[k] = max(self.warm_n * float(prev[0]), 0.0)
                jt[k] = self.warm_t * prev[1:4]

        kn = self._eff_mass(n, ra, ia).copy()
        if dyn.any():
            kn[dyn] += self._eff_mass(n[dyn], rb[dyn], ib[dyn])
        kn = np.maximum(kn, 1e-9)

        va = self._point_velocity(ia, ra)
        vb = np.zeros_like(va)
        if dyn.any():
            vb[dyn] = self._point_velocity(ib[dyn], rb[dyn])
        vn0 = np.einsum("ij,ij->i", va - vb, n)

        target = np.where(vn0 < -self.rest_threshold, -self.restitution * vn0, 0.0)
        bias = self.beta * np.maximum(pen - self.slop, 0.0) / dt
        target = np.maximum(target, np.minimum(bias, self.max_bias))

        mu = self.friction
        # warm start
        impulse = n * jn[:, None] + jt
        if np.any(jn > 0.0) or np.any(np.abs(jt) > 0.0):
            self._apply_side(
                idx_all,
                np.concatenate([impulse, -impulse[dyn]], axis=0),
                off_all,
            )

        for _ in range(self.iterations):
            va = self._point_velocity(ia, ra)
            vb = np.zeros_like(va)
            if dyn.any():
                vb[dyn] = self._point_velocity(ib[dyn], rb[dyn])
            vrel = va - vb
            vn = np.einsum("ij,ij->i", vrel, n)
            delta = (target - vn) / kn
            new_jn = np.maximum(jn + delta, 0.0)
            delta = new_jn - jn
            jn = new_jn
            impulse = n * delta[:, None]
            self._apply_side(
                idx_all,
                np.concatenate([impulse, -impulse[dyn]], axis=0),
                off_all,
            )

            va = self._point_velocity(ia, ra)
            vb = np.zeros_like(va)
            if dyn.any():
                vb[dyn] = self._point_velocity(ib[dyn], rb[dyn])
            vrel = va - vb
            vn = np.einsum("ij,ij->i", vrel, n)
            vt = vrel - n * vn[:, None]
            speed = np.linalg.norm(vt, axis=1)
            active = speed > 1e-7
            if not active.any():
                continue
            tangent = vt / np.maximum(speed, 1e-12)[:, None]
            kt = self._eff_mass(tangent, ra, ia).copy()
            if dyn.any():
                kt[dyn] += self._eff_mass(tangent[dyn], rb[dyn], ib[dyn])
            kt = np.maximum(kt, 1e-9)
            delta = np.zeros_like(jt)
            step = np.where(active, -speed / kt, 0.0)
            delta = tangent * step[:, None]
            trial = jt + delta
            limit = mu * jn
            mag = np.linalg.norm(trial, axis=1)
            scale = np.where(mag > limit, limit / np.maximum(mag, 1e-12), 1.0)
            new_jt = trial * scale[:, None] * np.where(active, 1.0, 0.0)
            new_jt = np.where(active[:, None], new_jt, jt)
            delta = new_jt - jt
            jt = new_jt
            self._apply_side(
                idx_all,
                np.concatenate([delta, -delta[dyn]], axis=0),
                off_all,
            )

        for k in range(count):
            self.warm[int(key[k])] = np.array(
                [jn[k], jt[k, 0], jt[k, 1], jt[k, 2], float(self.step_index)]
            )

        impulse = n * jn[:, None] + jt
        for k in range(count):
            rec = agg.get(int(key[k]))
            if rec is None:
                agg[int(key[k])] = [impulse[k].copy(), int(ia[k]), int(ib[k])]
            else:
                rec[0] += impulse[k]

    # -------------------------------------------------------------- stepper
    def _update_inertia(self):
        self.invI_world = np.stack(
            [self.rot[i] @ self.inv_inertia @ self.rot[i].T for i in range(self.nb)], axis=0
        )

    def _step(self, agg):
        self.step_index += 1
        self.vel += self.gravity * self.dt
        self._update_inertia()
        con = self._contact_list()
        if con is not None:
            self._solve(con, self.dt, agg)
        self.pos += self.vel * self.dt
        w = self.omega
        zero = np.zeros((self.nb, 1))
        wq = np.concatenate([w, zero], axis=1)
        self.quat = _quat_normalize(self.quat + 0.5 * _quat_mul(wq, self.quat) * self.dt)
        self.rot = np.stack([_quat_to_mat(q) for q in self.quat], axis=0)
        if con is not None:
            self._project(con)
        self._clamp()

    def _project(self, con):
        pen = con["pen"]
        mask = pen > self.slop
        if not mask.any():
            return
        ia = con["ia"][mask]
        ib = con["ib"][mask]
        n = con["n"][mask]
        corr = np.minimum(self.proj * (pen[mask] - self.slop), self.max_proj)
        dyn = ib >= 0
        wa = self.inv_mass[ia]
        push = n * corr[:, None]
        self.pos += self._accum(ia, push * wa[:, None] * np.where(dyn, 0.5, 1.0)[:, None])
        if dyn.any():
            wb = self.inv_mass[ib[dyn]]
            self.pos += self._accum(ib[dyn], -push[dyn] * wb[:, None] * 0.5)

    def _clamp(self):
        np.nan_to_num(self.pos, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.vel, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.omega, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(self.quat).all():
            self.quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.nb, 1))
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        scale = np.where(speed > self.max_speed, self.max_speed / np.maximum(speed, 1e-9), 1.0)
        self.vel *= scale
        spin = np.linalg.norm(self.omega, axis=1, keepdims=True)
        scale = np.where(spin > self.max_spin, self.max_spin / np.maximum(spin, 1e-9), 1.0)
        self.omega *= scale

    # -------------------------------------------------------------- contract
    def _frame(self, time: float, agg, substeps: int) -> StateFrame:
        con = self._contact_list()
        if con is None:
            gaps = np.zeros(1)
        else:
            gap = con["gap"]
            near = gap[gap < self.near_gap]
            gaps = near if near.size else np.zeros(1)

        rows = []
        duration = max(substeps * self.dt, 1e-9)
        for key in sorted(agg):
            impulse, a, b = agg[key]
            force = impulse / duration
            rows.append(force)
            if b >= 0:
                rows.append(-force)
        interaction = np.asarray(rows, dtype=float) if rows else np.zeros((1, 3))

        external = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(time),
            self.pos.copy(),
            self.vel.copy(),
            orientation=self.quat.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"release_time": np.zeros(self.nb)},
            fields={
                "external_force": external,
                "interaction_force": interaction,
                "contact_signed_distance": np.asarray(gaps, dtype=float),
            },
            scalars={"contact_length_scale": float(self.length_scale)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        if not self.initialised:
            raise RuntimeError("initialize() must be called before advance_to()")
        target = float(time)
        steps = int(round((target - self.time) / self.dt))
        if steps < 0:
            steps = 0
        agg: dict[int, list] = {}
        for _ in range(steps):
            self._step(agg)
        self.time = target
        return self._frame(target, agg, steps)

    def diagnostics(self) -> dict[str, object]:
        values = dict(self.diagnostics_values)
        values.update(
            {
                "solver": "rigid_body_sphere_cover_sequential_impulse",
                "time": float(self.time),
                "speed_max": float(np.linalg.norm(self.vel, axis=1).max())
                if self.nb
                else 0.0,
                "contacts_tracked": int(len(self.warm)),
            }
        )
        return values
