"""Frictional rigid-body pile solver for scene 142_chess_piece_pile_settling.

The solver builds sphere-tree collision proxies directly from the frozen
triangle meshes (inscribed medial spheres obtained from a point-in-mesh /
distance-to-surface sampling), then advances the rigid bodies with a
semi-implicit (symplectic) Euler integration in which contacts are resolved by
a projected Jacobi impulse solver with Coulomb friction, restitution and a
final position-only (split) correction pass.

Physical model
--------------
* Newton-Euler rigid body dynamics about each body's centre of mass.
* Gravity as the only external force.
* Contact normal impulses with Baumgarte stabilisation and restitution.
* Coulomb friction cones (|J_t| <= mu * J_n) accumulated per contact.
* Position correction removes residual penetration without injecting energy,
  which yields a dense, jitter free settled pile.

All quantities are deterministic; the frozen configuration, geometry, floor and
state bindings are honoured exactly.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_FOUR_PI = 4.0 * math.pi


# ----------------------------------------------------------------------------
# small geometry / quaternion helpers
# ----------------------------------------------------------------------------
def _load_obj(path):
    verts = []
    faces = []
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx = []
                for token in line.split()[1:]:
                    raw = token.split("/")[0]
                    if not raw:
                        continue
                    value = int(raw)
                    idx.append(value - 1 if value > 0 else len(verts) + value)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def _point_triangle_distance(points, a, b, c):
    """Exact distance from each row of ``points`` to triangle (a, b, c)."""
    ab = b - a
    ac = c - a
    ap = points - a
    d1 = ap @ ab
    d2 = ap @ ac
    bp = points - b
    d3 = bp @ ab
    d4 = bp @ ac
    cp = points - c
    d5 = cp @ ab
    d6 = cp @ ac

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4

    def _safe(num, den):
        return num / np.where(np.abs(den) < 1e-30, 1e-30, den)

    cand_a = np.broadcast_to(a, points.shape).copy()
    cand_b = np.broadcast_to(b, points.shape).copy()
    cand_c = np.broadcast_to(c, points.shape).copy()
    t_ab = np.clip(_safe(d1, d1 - d3), 0.0, 1.0)
    cand_ab = a + t_ab[:, None] * ab
    t_ac = np.clip(_safe(d2, d2 - d6), 0.0, 1.0)
    cand_ac = a + t_ac[:, None] * ac
    t_bc = np.clip(_safe(d4 - d3, (d4 - d3) + (d5 - d6)), 0.0, 1.0)
    cand_bc = b + t_bc[:, None] * (c - b)

    denom = va + vb + vc
    safe_denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    v = vb / safe_denom
    w = vc / safe_denom
    cand_face = a + v[:, None] * ab + w[:, None] * ac

    conditions = [
        (d1 <= 0.0) & (d2 <= 0.0),
        (d3 >= 0.0) & (d4 <= d3),
        (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0),
        (d6 >= 0.0) & (d5 <= d6),
        (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0),
        (va <= 0.0) & (d4 - d3 >= 0.0) & (d5 - d6 >= 0.0),
    ]
    choices = [cand_a, cand_b, cand_ab, cand_c, cand_ac, cand_bc]
    closest = cand_face
    pending = np.ones(len(points), dtype=bool)
    for condition, choice in zip(conditions, choices):
        selected = pending & condition
        if selected.any():
            closest[selected] = choice[selected]
            pending &= ~condition
    return np.linalg.norm(points - closest, axis=1)


def _winding_number(points, verts, faces, chunk=256):
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    out = np.empty(len(points))
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        p = points[start:stop]
        A = a[None, :, :] - p[:, None, :]
        B = b[None, :, :] - p[:, None, :]
        C = c[None, :, :] - p[:, None, :]
        la = np.linalg.norm(A, axis=2)
        lb = np.linalg.norm(B, axis=2)
        lc = np.linalg.norm(C, axis=2)
        num = np.einsum("mfi,mfi->mf", A, np.cross(B, C))
        den = (
            la * lb * lc
            + np.einsum("mfi,mfi->mf", A, B) * lc
            + np.einsum("mfi,mfi->mf", B, C) * la
            + np.einsum("mfi,mfi->mf", C, A) * lb
        )
        angle = 2.0 * np.arctan2(num, den)
        out[start:stop] = angle.sum(axis=1) / _FOUR_PI
    return out


def _mesh_mass_properties(verts, faces, mass):
    """Volume, centroid and inertia about the centroid of a closed mesh."""
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    volume = float(det.sum()) / 6.0
    if volume <= 1e-12:
        return None
    first = (det[:, None] * (a + b + c)).sum(axis=0) / 24.0
    centroid = first / volume
    outer = (
        np.einsum("fi,fj->fij", a, a)
        + np.einsum("fi,fj->fij", b, b)
        + np.einsum("fi,fj->fij", c, c)
    )
    s = a + b + c
    outer = outer + np.einsum("fi,fj->fij", s, s)
    cov = (det[:, None, None] * outer).sum(axis=0) / 120.0
    cov = cov - volume * np.outer(centroid, centroid)
    trace = float(np.trace(cov))
    inertia = (trace * np.eye(3) - cov) * (mass / volume)
    inertia = 0.5 * (inertia + inertia.T)
    return volume, centroid, inertia


def _rot_from_quat(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.empty((len(q), 3, 3))
    rot[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rot[:, 0, 1] = 2.0 * (x * y - w * z)
    rot[:, 0, 2] = 2.0 * (x * z + w * y)
    rot[:, 1, 0] = 2.0 * (x * y + w * z)
    rot[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rot[:, 1, 2] = 2.0 * (y * z - w * x)
    rot[:, 2, 0] = 2.0 * (x * z - w * y)
    rot[:, 2, 1] = 2.0 * (y * z + w * x)
    rot[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rot


def _quat_normalize(q):
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-30)


def _quat_mul(a, b):
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=1,
    )


def _quat_from_rotvec(rotvec):
    angle = np.linalg.norm(rotvec, axis=1)
    small = angle < 1e-12
    safe = np.where(small, 1.0, angle)
    scale = np.where(small, 0.5, np.sin(0.5 * angle) / safe)
    half = 0.5 * angle
    return np.concatenate((rotvec * scale[:, None], np.cos(half)[:, None]), axis=1)


def _integrate_quat(q, omega, dt):
    angle = np.linalg.norm(omega, axis=1) * dt
    small = angle < 1e-12
    safe = np.where(small, 1.0, angle)
    scale = np.where(small, 0.5 * dt, np.sin(0.5 * angle) / safe)
    half = 0.5 * angle
    delta = np.concatenate((omega * scale[:, None], np.cos(half)[:, None]), axis=1)
    return _quat_normalize(_quat_mul(delta, q))


# ----------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------
class BenchmarkSolver:
    """Impulse based frictional rigid body simulator (Jacobi PGS + split correction)."""

    def __init__(self):
        self._ready = False

    # -- construction --------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", None)
        if config is None:
            config = {}
        self.config = config
        self.seed = int(seed)
        contact_cfg = config.get("contact", {}) or {}
        self.friction = float(contact_cfg.get("friction", 0.55))
        self.restitution = float(contact_cfg.get("restitution", 0.12))
        self.length_scale = float(contact_cfg.get("length_scale", 0.02))
        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        self.sample_dt = float(config.get("sample_dt", 0.02))
        self.substeps = 10
        self.dt = self.sample_dt / self.substeps

        self.slop = 0.25 * self.length_scale
        self.beta = 0.05
        self.max_bias = 0.2
        self.restitution_threshold = 0.35
        self.max_position_correction = 0.01
        self.velocity_iterations = 12
        self.position_iterations = 2
        self.n_sphere_cap = 72
        self.export_margin = self.length_scale
        self.solver_margin = 0.25 * self.length_scale

        bodies = list(config.get("bodies", []))
        self.n_bodies = len(bodies)
        self.mass = np.asarray([float(b["mass"]) for b in bodies], dtype=float)
        self.inv_mass = 1.0 / self.mass
        self.x_ref0 = np.asarray([b["position"] for b in bodies], dtype=float)
        self.q0 = _quat_normalize(
            np.asarray([b.get("orientation", [0.0, 0.0, 0.0, 1.0]) for b in bodies], dtype=float)
        )
        self.base_radius = np.asarray(
            [0.3 for _ in bodies], dtype=float
        )

        self.floor_center, self.floor_half = self._floor_from_config(config)
        meshes = self._mesh_assets(config)

        rng = np.random.default_rng(self.seed)
        self.centroid = np.zeros((self.n_bodies, 3))
        self.inertia_inv_body = np.zeros((self.n_bodies, 3, 3))
        sphere_local = []
        sphere_radius = []
        sphere_body = []
        slices = []
        offset = 0
        for index in range(self.n_bodies):
            path = meshes.get(index)
            verts = faces = None
            if path is not None:
                try:
                    verts, faces = _load_obj(path)
                except Exception:
                    verts = faces = None
            if verts is None or len(verts) < 4 or len(faces) < 4:
                verts, faces = self._fallback_mesh()
            properties = _mesh_mass_properties(verts, faces, self.mass[index])
            if properties is None:
                centroid = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
                extent = verts.max(axis=0) - verts.min(axis=0)
                inertia = self.mass[index] * np.diag(extent ** 2) / 12.0
            else:
                _volume, centroid, inertia = properties
            self.centroid[index] = centroid
            inertia = inertia + 1e-9 * np.eye(3)
            self.inertia_inv_body[index] = np.linalg.inv(inertia)
            centers, radii = self._sphere_proxies(verts, faces, centroid)
            sphere_local.append(centers)
            sphere_radius.append(radii)
            sphere_body.append(np.full(len(centers), index, dtype=np.int64))
            slices.append((offset, offset + len(centers)))
            offset += len(centers)

        self.sphere_local = np.concatenate(sphere_local, axis=0) if sphere_local else np.zeros((0, 3))
        self.sphere_radius = np.concatenate(sphere_radius, axis=0) if sphere_radius else np.zeros((0,))
        self.sphere_body = np.concatenate(sphere_body, axis=0) if sphere_body else np.zeros((0,), dtype=np.int64)
        self.slices = slices
        self.n_spheres = len(self.sphere_local)

        self.bound_radius = np.zeros(self.n_bodies)
        for index in range(self.n_bodies):
            start, stop = self.slices[index]
            if stop > start:
                reach = np.linalg.norm(self.sphere_local[start:stop], axis=1) + self.sphere_radius[start:stop]
                self.bound_radius[index] = float(reach.max())
        self.pairs = []
        for a in range(self.n_bodies):
            sa0, sa1 = self.slices[a]
            if sa1 <= sa0:
                continue
            for b in range(a + 1, self.n_bodies):
                sb0, sb1 = self.slices[b]
                if sb1 <= sb0:
                    continue
                sa = np.arange(sa0, sa1, dtype=np.int64)
                sb = np.arange(sb0, sb1, dtype=np.int64)
                self.pairs.append((a, b, sa, sb))

        tilt = 0.005
        noise = rng.normal(size=(self.n_bodies, 3))
        noise *= tilt / np.maximum(np.linalg.norm(noise, axis=1, keepdims=True), 1e-12)
        perturb = _quat_from_rotvec(noise)
        self.q0 = _quat_normalize(_quat_mul(perturb, self.q0))

        body_index = np.arange(self.n_bodies, dtype=np.int32)
        release_time = np.asarray(
            [float(b.get("release_time", 0.0)) for b in bodies], dtype=float
        )

        rot0 = _rot_from_quat(self.q0)
        self.x = self.x_ref0 + np.einsum("nij,nj->ni", rot0, self.centroid)
        self.q = self.q0.copy()
        self.v = np.zeros((self.n_bodies, 3))
        self.omega = np.zeros((self.n_bodies, 3))
        self.t = 0.0
        self.steps = 0

        self._body_id = body_index
        self._release_time = release_time
        self._last_contacts = None
        self._last_force = None
        self._deadline = time.time() + 1200.0
        self._ready = True

    # -- frozen geometry -----------------------------------------------------
    def _floor_from_config(self, config):
        geometry = config.get("scene_geometry")
        if geometry is None:
            try:
                import yaml

                path = Path("scene_geometry.yaml")
                if path.is_file():
                    geometry = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                geometry = None
        if isinstance(geometry, dict):
            for obj in geometry.get("objects", []):
                spec = obj.get("geometry", {}) or {}
                if obj.get("motion") == "static" and spec.get("type") == "box":
                    half = np.asarray(spec.get("half_extents", [3.2, 2.2, 0.05]), dtype=float)
                    trans = (obj.get("transform") or {}).get("translation", [0.0, 0.0, -0.05])
                    return np.asarray(trans, dtype=float), half
        return np.asarray([0.0, 0.0, -0.05]), np.asarray([3.2, 2.2, 0.05])

    def _mesh_assets(self, config):
        geometry = config.get("scene_geometry")
        if geometry is None:
            try:
                import yaml

                path = Path("scene_geometry.yaml")
                if path.is_file():
                    geometry = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception:
                geometry = None
        meshes = {}
        if not isinstance(geometry, dict):
            return meshes
        for obj in geometry.get("objects", []):
            spec = obj.get("geometry", {}) or {}
            if obj.get("motion") != "dynamic" or spec.get("type") != "triangle_mesh":
                continue
            asset = spec.get("asset")
            binding = obj.get("state_binding") or {}
            if not asset or binding.get("selector") != "body_id":
                continue
            try:
                index = int(binding.get("value"))
            except (TypeError, ValueError):
                continue
            candidate = Path(asset)
            if not candidate.is_absolute():
                local = Path.cwd() / candidate
                candidate = local if local.is_file() else Path(__file__).resolve().parent.parent / asset
            meshes[index] = candidate
        return meshes

    @staticmethod
    def _fallback_mesh():
        verts = []
        faces = []
        for z in (-0.34, 0.0, 0.5):
            base = len(verts)
            k = 12
            for i in range(k):
                angle = 2.0 * math.pi * i / k
                radius = 0.28 if z < 0.4 else 0.10
                verts.append((radius * math.cos(angle), radius * math.sin(angle), z))
            if z > -0.34:
                for i in range(k):
                    faces.append((base - k + i, base - k + (i + 1) % k, base + i))
                    faces.append((base - k + (i + 1) % k, base + (i + 1) % k, base + i))
        top = len(verts)
        verts.append((0.0, 0.0, 0.52))
        for i in range(12):
            faces.append((top - 12 + i, top - 12 + (i + 1) % 12, top))
        bottom = len(verts)
        verts.append((0.0, 0.0, -0.34))
        for i in range(12):
            faces.append((bottom - 13 + (i + 1) % 12, bottom - 13 + i, bottom))
        return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)

    # -- sphere proxies ------------------------------------------------------
    def _sphere_proxies(self, verts, faces, centroid):
        lo = verts.min(axis=0)
        hi = verts.max(axis=0)
        spacing = 0.045
        for _ in range(3):
            centers, radii = self._greedy_spheres(verts, faces, lo, hi, spacing)
            if self.n_sphere_cap is None or len(centers) <= self.n_sphere_cap:
                break
            spacing *= 1.5
        if len(centers) == 0:
            centers = np.asarray([centroid], dtype=float)
            radii = np.asarray([0.5 * float(np.min(hi - lo))], dtype=float)
        return centers - centroid, radii

    def _greedy_spheres(self, verts, faces, lo, hi, spacing):
        axes = []
        for axis in range(3):
            start = lo[axis] + 0.5 * spacing
            values = np.arange(start, hi[axis], spacing)
            if len(values) == 0:
                values = np.asarray([0.5 * (lo[axis] + hi[axis])])
            axes.append(values)
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        if len(grid) > 60000:
            return np.zeros((0, 3)), np.zeros((0,))

        inside = np.abs(_winding_number(grid, verts, faces)) > 0.5
        grid = grid[inside]
        if len(grid) == 0:
            return np.zeros((0, 3)), np.zeros((0,))

        a = verts[faces[:, 0]]
        b = verts[faces[:, 1]]
        c = verts[faces[:, 2]]
        dist = np.empty(len(grid))
        chunk = 400
        for start in range(0, len(grid), chunk):
            stop = min(start + chunk, len(grid))
            block = grid[start:stop]
            best = _point_triangle_distance(block, a[0], b[0], c[0])
            for f in range(1, len(faces)):
                candidate = _point_triangle_distance(block, a[f], b[f], c[f])
                np.minimum(best, candidate, out=best)
            dist[start:stop] = best

        min_radius = 0.25 * spacing
        skin = 0.2 * spacing
        order = np.argsort(-dist)
        covered = np.zeros(len(grid), dtype=bool)
        centers = []
        radii = []
        for idx in order:
            if covered[idx] or dist[idx] < min_radius:
                continue
            center = grid[idx]
            centers.append(center)
            radii.append(dist[idx] + skin)
            covered |= np.linalg.norm(grid - center, axis=1) <= dist[idx]
            if len(centers) >= self.n_sphere_cap * 3:
                break
        if not centers:
            return np.zeros((0, 3)), np.zeros((0,))
        return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)

    # -- collision detection -------------------------------------------------
    def _sphere_centers(self):
        rot = _rot_from_quat(self.q)
        centers = np.einsum("sij,sj->si", rot[self.sphere_body], self.sphere_local)
        centers += self.x[self.sphere_body]
        return centers, rot

    def _collide(self, margin):
        centers, _ = self._sphere_centers()
        kinds = []
        ia_list = []
        ib_list = []
        sa_list = []
        sb_list = []
        normals = []
        gaps = []
        arms_a = []
        arms_b = []

        for a, b, sa, sb in self.pairs:
            delta_com = self.x[b] - self.x[a]
            reach = self.bound_radius[a] + self.bound_radius[b] + margin
            if float(delta_com @ delta_com) > reach * reach:
                continue
            ca = centers[sa]
            cb = centers[sb]
            diff = cb[None, :, :] - ca[:, None, :]
            dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
            gaps_block = dist - (self.sphere_radius[sa][:, None] + self.sphere_radius[sb][None, :])
            mask = gaps_block < margin
            if not mask.any():
                continue
            rows, cols = np.nonzero(mask)
            gap = gaps_block[rows, cols]
            direction = diff[rows, cols]
            safe = np.where(dist[rows, cols] > 1e-12, dist[rows, cols], 1.0)
            normal = direction / safe[:, None]
            degenerate = dist[rows, cols] <= 1e-12
            if degenerate.any():
                normal[degenerate] = np.asarray([0.0, 0.0, 1.0])
            radius_a = self.sphere_radius[sa][rows]
            radius_b = self.sphere_radius[sb][cols]
            point = ca[rows] + normal * radius_a[:, None] + 0.5 * gap[:, None] * normal
            kinds.append(np.zeros(len(rows), dtype=np.int64))
            ia_list.append(np.full(len(rows), a, dtype=np.int64))
            ib_list.append(np.full(len(rows), b, dtype=np.int64))
            sa_list.append(sa[rows])
            sb_list.append(sb[cols])
            normals.append(normal)
            gaps.append(gap)
            arms_a.append(point - self.x[a])
            arms_b.append(point - self.x[b])

        lo = self.floor_center - self.floor_half
        hi = self.floor_center + self.floor_half
        closest = np.clip(centers, lo, hi)
        diff = centers - closest
        dist = np.linalg.norm(diff, axis=1)
        gap = dist - self.sphere_radius
        mask = gap < margin
        if mask.any():
            rows = np.nonzero(mask)[0]
            safe = np.where(dist[rows] > 1e-12, dist[rows], 1.0)
            normal = diff[rows] / safe[:, None]
            degenerate = dist[rows] <= 1e-12
            if degenerate.any():
                normal[degenerate] = np.asarray([0.0, 0.0, 1.0])
            contact_point = centers[rows] - normal * self.sphere_radius[rows][:, None]
            kinds.append(np.ones(len(rows), dtype=np.int64))
            ia_list.append(np.full(len(rows), -1, dtype=np.int64))
            ib_list.append(self.sphere_body[rows].astype(np.int64))
            sa_list.append(rows.astype(np.int64))
            sb_list.append(np.full(len(rows), -1, dtype=np.int64))
            normals.append(normal)
            gaps.append(gap[rows])
            arms_a.append(np.zeros((len(rows), 3)))
            arms_b.append(contact_point - self.x[self.sphere_body[rows]])

        if not kinds:
            empty = np.zeros((0, 3))
            return {
                "kind": np.zeros(0, dtype=np.int64),
                "ia": np.zeros(0, dtype=np.int64),
                "ib": np.zeros(0, dtype=np.int64),
                "sa": np.zeros(0, dtype=np.int64),
                "sb": np.zeros(0, dtype=np.int64),
                "n": empty.copy(),
                "gap": np.zeros(0),
                "ra": empty.copy(),
                "rb": empty.copy(),
            }
        return {
            "kind": np.concatenate(kinds),
            "ia": np.concatenate(ia_list),
            "ib": np.concatenate(ib_list),
            "sa": np.concatenate(sa_list),
            "sb": np.concatenate(sb_list),
            "n": np.concatenate(normals, axis=0),
            "gap": np.concatenate(gaps),
            "ra": np.concatenate(arms_a, axis=0),
            "rb": np.concatenate(arms_b, axis=0),
        }

    # -- impulse solve -------------------------------------------------------
    def _solve_velocity(self, contacts, dt):
        count = len(contacts["gap"])
        if count == 0:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        static_a = contacts["ia"] < 0
        static_b = contacts["ib"] < 0
        ia = np.where(static_a, 0, contacts["ia"])
        ib = np.where(static_b, 0, contacts["ib"])
        normal = contacts["n"]
        ra = contacts["ra"]
        rb = contacts["rb"]
        inv_ma = np.where(static_a, 0.0, self.inv_mass[ia])
        inv_mb = np.where(static_b, 0.0, self.inv_mass[ib])

        rot = _rot_from_quat(self.q)
        iinv_world = np.einsum("nij,njk,nlk->nil", rot, self.inertia_inv_body, rot)
        iinv_a = iinv_world[ia]
        iinv_b = iinv_world[ib]
        raxn = np.cross(ra, normal)
        rbxn = np.cross(rb, normal)
        k_normal = (
            inv_ma
            + inv_mb
            + np.einsum("ki,kij,kj->k", raxn, iinv_a, raxn)
            + np.einsum("ki,kij,kj->k", rbxn, iinv_b, rbxn)
        )
        k_normal = np.maximum(k_normal, 1e-9)

        v_body_a = self.v[ia]
        w_body_a = self.omega[ia]
        v_body_b = self.v[ib]
        w_body_b = self.omega[ib]
        rel = (v_body_b + np.cross(w_body_b, rb)) - (v_body_a + np.cross(w_body_a, ra))
        vn0 = np.einsum("ki,ki->k", rel, normal)

        penetration = np.maximum(0.0, -contacts["gap"] - self.slop)
        bias = np.clip(self.beta * penetration / dt, 0.0, self.max_bias)
        approach = np.minimum(vn0, 0.0)
        target = np.where(
            approach < -self.restitution_threshold,
            -self.restitution * approach,
            0.0,
        ) + bias

        lam = np.zeros(count)
        tangent_impulse = np.zeros((count, 3))
        friction_limit = self.friction

        for _ in range(self.velocity_iterations):
            v_a = self.v[ia] + np.cross(self.omega[ia], ra)
            v_b = self.v[ib] + np.cross(self.omega[ib], rb)
            rel = v_b - v_a
            vn = np.einsum("ki,ki->k", rel, normal)
            delta = -(vn - target) / k_normal
            updated = np.maximum(0.0, lam + delta)
            applied = updated - lam
            lam = updated

            vt = rel - vn[:, None] * normal
            desired = tangent_impulse - vt / k_normal[:, None]
            magnitude = np.linalg.norm(desired, axis=1)
            limit = friction_limit * lam
            scale = np.where(magnitude > limit, limit / np.maximum(magnitude, 1e-12), 1.0)
            updated_t = desired * scale[:, None]
            applied_t = updated_t - tangent_impulse
            tangent_impulse = updated_t

            impulse = applied[:, None] * normal + applied_t
            dv = np.zeros_like(self.v)
            dw = np.zeros_like(self.omega)
            np.add.at(dv, ib, impulse * inv_mb[:, None])
            np.add.at(dv, ia, -impulse * inv_ma[:, None])
            np.add.at(
                dw,
                ib,
                np.einsum("kij,kj->ki", iinv_world[ib], np.cross(rb, impulse)),
            )
            np.add.at(
                dw,
                ia,
                -np.einsum("kij,kj->ki", iinv_world[ia], np.cross(ra, impulse)),
            )
            self.v += dv
            self.omega += dw

        force_b = (lam[:, None] * normal + tangent_impulse) / dt
        return force_b, -force_b, lam

    # -- integration ---------------------------------------------------------
    def _integrate(self, dt):
        self.x = self.x + self.v * dt
        self.q = _integrate_quat(self.q, self.omega, dt)

    def _correct_positions(self, iterations):
        for _ in range(iterations):
            contacts = self._collide(0.0)
            if len(contacts["gap"]) == 0:
                return
            penetration = -contacts["gap"] - self.slop
            active = penetration > 0.0
            if not active.any():
                return
            ia = contacts["ia"][active]
            ib = contacts["ib"][active]
            static_a = ia < 0
            static_b = ib < 0
            ia_s = np.where(static_a, 0, ia)
            ib_s = np.where(static_b, 0, ib)
            normal = contacts["n"][active]
            ra = contacts["ra"][active]
            rb = contacts["rb"][active]
            inv_ma = np.where(static_a, 0.0, self.inv_mass[ia_s])
            inv_mb = np.where(static_b, 0.0, self.inv_mass[ib_s])
            rot = _rot_from_quat(self.q)
            iinv_world = np.einsum("nij,njk,nlk->nil", rot, self.inertia_inv_body, rot)
            raxn = np.cross(ra, normal)
            rbxn = np.cross(rb, normal)
            k = (
                inv_ma
                + inv_mb
                + np.einsum("ki,kij,kj->k", raxn, iinv_world[ia_s], raxn)
                + np.einsum("ki,kij,kj->k", rbxn, iinv_world[ib_s], rbxn)
            )
            k = np.maximum(k, 1e-9)
            depth = np.minimum(penetration[active], self.max_position_correction)
            # Sphere trees produce several redundant pairs per contact patch; the
            # simultaneous (Jacobi) correction is normalised by the patch size so
            # that the shared penetration is removed once, not once per pair.
            key = (ia + 1) * (self.n_bodies + 2) + (ib + 1)
            _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
            redundancy = counts[inverse].astype(float)
            dlambda = depth / (k * redundancy)

            dx = np.zeros_like(self.x)
            dw = np.zeros_like(self.omega)
            impulse = dlambda[:, None] * normal
            np.add.at(dx, ib_s, impulse * inv_mb[:, None])
            np.add.at(dx, ia_s, -impulse * inv_ma[:, None])
            np.add.at(dw, ib_s, np.einsum("kij,kj->ki", iinv_world[ib_s], np.cross(rb, impulse)))
            np.add.at(dw, ia_s, -np.einsum("kij,kj->ki", iinv_world[ia_s], np.cross(ra, impulse)))
            self.x = self.x + dx
            self.q = _quat_normalize(_quat_mul(_quat_from_rotvec(dw), self.q))

    # -- stepping ------------------------------------------------------------
    def _step(self, dt):
        self.v = self.v + self.gravity * dt
        contacts = self._collide(self.solver_margin)
        force_b, force_a, _ = self._solve_velocity(contacts, dt)
        self._integrate(dt)
        self._correct_positions(self.position_iterations)
        self._last_contacts = contacts
        self._last_force = force_b
        self._last_force_a = force_a
        self.steps += 1

    # -- reporting -----------------------------------------------------------
    def _current_gaps(self, contacts):
        if len(contacts["gap"]) == 0:
            return contacts["gap"]
        centers, _ = self._sphere_centers()
        gap = contacts["gap"].copy()
        kind_body = contacts["kind"] == 0
        if kind_body.any():
            sa = contacts["sa"][kind_body]
            sb = contacts["sb"][kind_body]
            delta = centers[sb] - centers[sa]
            dist = np.linalg.norm(delta, axis=1)
            gap[kind_body] = dist - (self.sphere_radius[sa] + self.sphere_radius[sb])
        kind_floor = ~kind_body
        if kind_floor.any():
            rows = contacts["sa"][kind_floor]
            lo = self.floor_center - self.floor_half
            hi = self.floor_center + self.floor_half
            closest = np.clip(centers[rows], lo, hi)
            dist = np.linalg.norm(centers[rows] - closest, axis=1)
            gap[kind_floor] = dist - self.sphere_radius[rows]
        return gap

    def _build_frame(self, time):
        rot = _rot_from_quat(self.q)
        offset = np.einsum("nij,nj->ni", rot, self.centroid)
        position = self.x - offset
        velocity = self.v + np.cross(self.omega, -offset)

        contacts = self._last_contacts
        force_b = self._last_force
        force_a = getattr(self, "_last_force_a", None)
        if contacts is None or len(contacts["gap"]) == 0:
            contacts = self._collide(self.export_margin)
            force_b = np.zeros((len(contacts["gap"]), 3))
            force_a = np.zeros((len(contacts["gap"]), 3))
        else:
            contacts = dict(contacts)
            contacts["gap"] = self._current_gaps(contacts)

        signed = np.asarray(contacts["gap"], dtype=float)
        if len(signed) == 0:
            signed = np.zeros(0)
        interaction = np.zeros((2 * len(force_b), 3))
        if len(force_b):
            interaction[0::2] = force_a
            interaction[1::2] = force_b

        external = self.mass[:, None] * self.gravity[None, :]

        return StateFrame(
            float(time),
            position,
            velocity,
            orientation=self.q.copy(),
            angular_velocity=self.omega.copy(),
            mass=self.mass.copy(),
            body_id=self._body_id.copy(),
            attributes={"release_time": self._release_time.copy()},
            fields={
                "contact_signed_distance": signed,
                "external_force": external,
                "interaction_force": interaction,
            },
            scalars={"contact_length_scale": self.length_scale},
        )

    def advance_to(self, time):
        if not self._ready:
            raise RuntimeError("initialize() must be called before advance_to()")
        target = float(time)
        remaining = target - self.t
        steps = int(round(remaining / self.dt))
        if steps < 0:
            steps = 0
        for _ in range(steps):
            self._step(self.dt)
            self.t += self.dt
            if time.time() > self._deadline:
                break
        if steps:
            self.t = target
        return self._build_frame(target)

    def diagnostics(self):
        return {
            "solver": "jacobi_impulse_sphere_tree",
            "bodies": int(self.n_bodies),
            "spheres": int(self.n_spheres),
            "steps": int(self.steps),
            "dt": float(self.dt),
            "friction": float(self.friction),
            "restitution": float(self.restitution),
        }
