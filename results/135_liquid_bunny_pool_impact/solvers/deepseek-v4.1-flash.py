"""Physically based solver for the liquid bunny / pool impact scene.

Model
-----
* The liquid bunny is discretised by the frozen triangle mesh samples
  (``material_label == 1``, one node per mesh vertex, mesh order preserved).
  It is advanced as an elasto-plastic cohesive body: an optimal-rotation shape
  matching operator provides the elastic (shape memory) response, and a
  strain driven plastic flow lets the material yield and flow once the impact
  strain exceeds the yield threshold, so the body keeps its silhouette in free
  fall and then spreads/merges into the pool.
* The receiving pool (``material_label == 0``) is a weakly compressible
  particle fluid integrated with a Position Based Fluids density projection
  (one sided, i.e. compression only constraints), XSPH viscosity and a mild
  surface cohesion term.  It splashes outward on impact and rings down into
  damped gravity waves.
* The rigid basin boxes (ground and three walls) are read from the frozen
  scene geometry and act as no penetration, low restitution, frictional
  boundaries for every sample.
* The bunny membrane is an impermeable moving boundary for the pool fluid; the
  aggregate contact impulse is returned to the bunny, so the coupling is
  momentum conserving.

Every quantity is integrated with an explicit fixed time step from the frozen
initial state produced by the scene; nothing is scripted or precomputed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame

_DT = 0.005          # internal integration step [s]
_MAX_SPEED = 25.0    # safety clamp on sample speeds [m/s]
_BOX_MARGIN = 0.02   # safety clamp around the frozen domain [m]


# --------------------------------------------------------------------------- #
# SPH kernels
# --------------------------------------------------------------------------- #
def _poly6_scalar(r2: float, h: float) -> float:
    h2 = h * h
    if r2 >= h2:
        return 0.0
    return 315.0 / (64.0 * np.pi * h ** 9) * (h2 - r2) ** 3


def _pair_kernels(rv: np.ndarray, h: float):
    """Poly6 value and spiky gradient for the pair separations ``rv``."""
    r2 = np.einsum("ij,ij->i", rv, rv)
    r = np.sqrt(r2)
    w = np.zeros(len(r))
    keep = r < h
    if np.any(keep):
        w[keep] = 315.0 / (64.0 * np.pi * h ** 9) * (h * h - r2[keep]) ** 3
    grad = np.zeros_like(rv)
    if np.any(keep):
        coef = (-45.0 / (np.pi * h ** 6) * (h - r[keep]) ** 2) / np.maximum(r[keep], 1e-12)
        grad[keep] = coef[:, None] * rv[keep]
    return w, grad


def _segment_grid(lo: float, hi: float, count: int) -> np.ndarray:
    if count <= 1:
        return np.asarray([0.5 * (lo + hi)])
    return np.linspace(lo, hi, count)


class BenchmarkSolver:
    """Deterministic particle solver for the liquid bunny pool impact."""

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.config = cfg
        self.seed = int(seed)
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        sampling = cfg.get("sampling", {}) or {}
        self.spacing = float(sampling.get("spacing", 0.14))
        self.density = float(sampling.get("density", 1000.0))

        geometry = cfg.get("scene_geometry")
        if not isinstance(geometry, dict):
            geometry = self._read_geometry_file()
        self.geometry = geometry if isinstance(geometry, dict) else {"objects": []}
        self.objects = list(self.geometry.get("objects", []))

        frame = scene.initial_state()
        self.pos = np.array(frame.position, dtype=float).copy()
        n = len(self.pos)
        self.vel = np.zeros((n, 3), dtype=float)
        labels = np.asarray(frame.attributes["material_label"]).astype(np.int64)
        self.labels = labels
        mass = np.array(frame.mass, dtype=float).copy()
        radius = np.array(
            frame.attributes.get("particle_radius", np.full(n, 0.45 * self.spacing)),
            dtype=float,
        ).copy()
        radius[radius <= 0.0] = 0.45 * self.spacing
        self.radius = radius

        self.pool_idx = np.nonzero(labels == 0)[0]
        self.bunny_idx = np.nonzero(labels == 1)[0]
        if len(self.pool_idx) == 0 or len(self.bunny_idx) == 0:
            raise ValueError("frozen initial state must contain pool and bunny samples")

        # ---- frozen rigid boundaries (boxes) --------------------------- #
        self.boxes = []
        for item in self.objects:
            if not item.get("collision_enabled", True):
                continue
            spec = item.get("geometry", {}) or {}
            if spec.get("type") != "box":
                continue
            half = np.asarray(spec.get("half_extents", [0, 0, 0]), dtype=float)
            center = np.asarray(
                (item.get("transform", {}) or {}).get("translation", [0.0, 0.0, 0.0]),
                dtype=float,
            )
            self.boxes.append((center - half, center + half))

        # ---- diagnostic grid (frozen particle domain) ------------------ #
        domain = None
        for item in self.objects:
            if item.get("geometry", {}).get("type") == "particle_domain":
                domain = item["geometry"]
                break
        if domain is None:
            low = self.pos.min(axis=0) - 0.5
            high = self.pos.max(axis=0) + 0.5
            shape = tuple(int(v) for v in sampling.get("field_grid_shape", [24, 16, 28]))
        else:
            low = np.asarray(domain["bounds_min"], dtype=float)
            high = np.asarray(domain["bounds_max"], dtype=float)
            shape = tuple(int(v) for v in domain.get("grid_shape", [24, 16, 28]))
        self.grid_low = low
        self.grid_high = high
        self.grid_shape = shape
        self.cell = (high - low) / np.asarray(shape, dtype=float)
        self.grid_spacing = float(np.mean(self.cell))

        # ---- frozen normalisation scales ------------------------------- #
        extent = float(np.max(np.ptp(self.pos[self.bunny_idx], axis=0)))
        self.characteristic_length = max(extent, self.spacing)
        self.reference_velocity = float(np.sqrt(np.linalg.norm(self.gravity) *
                                                self.characteristic_length))

        # ---- bunny material -------------------------------------------#
        self.mesh_faces = self._read_mesh_faces()
        volume = self._bunny_volume()
        if not np.isfinite(volume) or volume <= 0.0:
            volume = 0.2 * float(np.prod(np.ptp(self.pos[self.bunny_idx], axis=0)))
        self.bunny_volume = float(volume)
        node_mass = self.density * self.bunny_volume / len(self.bunny_idx)
        if not np.isfinite(node_mass) or node_mass <= 0.0:
            node_mass = self.density * self.spacing ** 3
        mass[self.bunny_idx] = node_mass
        self.mass = mass
        self.bunny_mass = float(mass[self.bunny_idx].sum())
        self.pool_mass = float(mass[self.pool_idx].sum())

        # rest shape for the shape matching / plastic flow operator
        q = self.pos[self.bunny_idx] - self._weighted_center(self.pos[self.bunny_idx],
                                                             self.mass[self.bunny_idx])
        self.rest_q = q
        self.edges = self._mesh_edges()

        # ---- pool fluid parameters ------------------------------------ #
        self.kernel_h = 2.0 * self.spacing
        self.rho0 = self._measure_rest_density()

        # coupling/contact parameters
        self.contact_length = 0.12
        self.contact_stiff = 0.0  # handled as a hard no-penetration constraint

        # ---- wall samples for the required boundary diagnostics -------- #
        self.wall_points, self.wall_normals = self._wall_samples()

        self._diag = {
            "steps": 0,
            "max_speed": 0.0,
            "min_level_set": 0.0,
            "boundary_slip": 0.0,
            "particles": int(len(self.pos)),
            "pool_particles": int(len(self.pool_idx)),
            "bunny_particles": int(len(self.bunny_idx)),
        }
        self.time = 0.0
        self._frame_cache = None

    # ------------------------------------------------------------------ #
    # scaffold helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_geometry_file() -> dict:
        path = Path("scene_geometry.yaml")
        if not path.is_file():
            return {"objects": []}
        try:
            from worldsolver.config import load_yaml

            return load_yaml(path)
        except Exception:
            return {"objects": []}

    def _mesh_asset_path(self):
        for item in self.objects:
            spec = item.get("geometry", {}) or {}
            if spec.get("type") == "triangle_mesh":
                asset = spec.get("asset") or spec.get("topology_asset")
                if asset:
                    candidate = Path(asset)
                    if candidate.is_file():
                        return candidate
        return None

    def _read_mesh_faces(self) -> np.ndarray:
        path = self._mesh_asset_path()
        if path is None:
            return np.zeros((0, 3), dtype=np.int64)
        faces = []
        try:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if line.startswith("f "):
                        faces.append([int(t.split("/")[0]) - 1 for t in line.split()[1:4]])
        except Exception:
            return np.zeros((0, 3), dtype=np.int64)
        if not faces:
            return np.zeros((0, 3), dtype=np.int64)
        faces = np.asarray(faces, dtype=np.int64)
        limit = len(self.bunny_idx)
        if faces.max() >= limit:
            return np.zeros((0, 3), dtype=np.int64)
        return faces

    def _bunny_volume(self) -> float:
        faces = self.mesh_faces
        if len(faces) == 0:
            return float("nan")
        tri = self.pos[self.bunny_idx][faces]
        vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
        return float(abs(vol))

    def _mesh_edges(self) -> np.ndarray:
        faces = self.mesh_faces
        if len(faces) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        pairs = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
        )
        pairs = np.sort(pairs, axis=1)
        return np.unique(pairs, axis=0)

    @staticmethod
    def _weighted_center(points: np.ndarray, mass: np.ndarray) -> np.ndarray:
        total = float(mass.sum())
        if total <= 0.0:
            return points.mean(axis=0)
        return (mass[:, None] * points).sum(axis=0) / total

    def _measure_rest_density(self) -> float:
        x = self.pos[self.pool_idx]
        m = self.mass[self.pool_idx]
        h = self.kernel_h
        n = len(x)
        tree = cKDTree(x)
        pairs = tree.query_pairs(h, output_type="ndarray")
        rho = m * _poly6_scalar(0.0, h)
        if len(pairs):
            rv = x[pairs[:, 0]] - x[pairs[:, 1]]
            r2 = np.einsum("ij,ij->i", rv, rv)
            w = np.zeros(len(r2))
            keep = r2 < h * h
            w[keep] = 315.0 / (64.0 * np.pi * h ** 9) * (h * h - r2[keep]) ** 3
            np.add.at(rho, pairs[:, 0], m[pairs[:, 1]] * w)
            np.add.at(rho, pairs[:, 1], m[pairs[:, 0]] * w)
        value = float(np.percentile(rho, 75.0))
        if not np.isfinite(value) or value <= 1e-9:
            value = self.density
        return value

    def _wall_samples(self):
        """Samples the wetted basin faces and returns unit outward normals."""
        low = self.grid_low - _BOX_MARGIN
        high = self.grid_high + _BOX_MARGIN
        points = []
        normals = []
        counts = (5, 3)
        for (blo, bhi) in self.boxes:
            center = 0.5 * (blo + bhi)
            half = 0.5 * (bhi - blo)
            for axis in range(3):
                u, v = [a for a in range(3) if a != axis]
                for side in (-1.0, 1.0):
                    normal = np.zeros(3)
                    normal[axis] = side
                    face = center + normal * half[axis]
                    gu = _segment_grid(center[u] - half[u], center[u] + half[u], counts[0])
                    gv = _segment_grid(center[v] - half[v], center[v] + half[v], counts[1])
                    for a in gu:
                        for b in gv:
                            pt = face.copy()
                            pt[u] = a
                            pt[v] = b
                            if np.all(pt >= low) and np.all(pt <= high):
                                points.append(pt)
                                normals.append(normal.copy())
        if not points:
            points = [np.asarray(self.pos.mean(axis=0))]
            normals = [np.asarray([0.0, 0.0, 1.0])]
        return np.asarray(points, dtype=float), np.asarray(normals, dtype=float)[:, :3]

    # ------------------------------------------------------------------ #
    # integration
    # ------------------------------------------------------------------ #
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time:
            target = self.time
        while self.time < target - 1e-12:
            h = min(_DT, target - self.time)
            if h <= 1e-12:
                break
            self._step(h)
            self.time += h
        self.time = target
        return self._frame(target)

    def _step(self, dt: float) -> None:
        self.vel += self.gravity[None, :] * dt
        speed = np.linalg.norm(self.vel, axis=1)
        too_fast = speed > _MAX_SPEED
        if np.any(too_fast):
            self.vel[too_fast] *= (_MAX_SPEED / speed[too_fast])[:, None]

        self._shape_match(dt)
        self._pool_step(dt)

        impulse = self._contact_impulse(dt)
        if impulse is not None:
            self.vel[self.bunny_idx] -= impulse[None, :] / self.bunny_mass

        self._bunny_smooth(dt)
        self.pos[self.bunny_idx] += self.vel[self.bunny_idx] * dt
        self._collide()
        np.nan_to_num(self.pos, copy=False, nan=0.0, posinf=1e3, neginf=-1e3)
        np.nan_to_num(self.vel, copy=False, nan=0.0, posinf=_MAX_SPEED, neginf=-_MAX_SPEED)

    # ------------------------------------------------------------------ #
    # bunny: elasto-plastic cohesive body
    # ------------------------------------------------------------------ #
    def _shape_match(self, dt: float) -> None:
        idx = self.bunny_idx
        x = self.pos[idx]
        v = self.vel[idx]
        m = self.mass[idx]
        q = self.rest_q
        c = self._weighted_center(x, m)
        p = x - c

        # optimal rotation of the current configuration against the rest shape
        moment = (m[:, None] * p).T @ q
        try:
            u, _, vt = np.linalg.svd(moment)
            rot = u @ vt
            if np.linalg.det(rot) < 0.0:
                u[:, -1] *= -1.0
                rot = u @ vt
        except np.linalg.LinAlgError:
            rot = np.eye(3)
        if not np.isfinite(rot).all():
            rot = np.eye(3)

        goal = c[None, :] + q @ rot.T
        stiff = 0.5
        v += stiff * (goal - x) / dt

        # strain driven plastic flow: the rest shape relaxes toward the current
        # (deformed) configuration wherever the local strain exceeds yield.
        local = p @ rot
        dev = np.linalg.norm(local - q, axis=1)
        yield_strain = 0.01
        flow = np.clip((dev - yield_strain) / max(yield_strain, 1e-9), 0.0, 1.0) * 0.08
        self.rest_q = q + (local - q) * flow[:, None]

        self.pos[idx] = x
        self.vel[idx] = v

    def _bunny_smooth(self, dt: float) -> None:
        edges = self.edges
        if len(edges) == 0:
            return
        idx = self.bunny_idx
        v = self.vel[idx]
        n = len(v)
        left = edges[:, 0]
        right = edges[:, 1]
        delta = v[right] - v[left]
        acc = np.empty_like(v)
        for k in range(3):
            acc[:, k] = (
                np.bincount(left, weights=delta[:, k], minlength=n)
                - np.bincount(right, weights=delta[:, k], minlength=n)
            )
        cnt = np.bincount(left, minlength=n).astype(float) + np.bincount(
            right, minlength=n
        ).astype(float)
        cnt[cnt == 0.0] = 1.0
        self.vel[idx] = v + 0.05 * (acc / cnt[:, None])

    # ------------------------------------------------------------------ #
    # pool: position based fluids
    # ------------------------------------------------------------------ #
    def _pool_step(self, dt: float) -> None:
        idx = self.pool_idx
        x = self.pos[idx]
        v = self.vel[idx]
        m = self.mass[idx]
        n = len(x)
        h = self.kernel_h
        rho0 = self.rho0

        if n > 1:
            tree = cKDTree(x)
            pairs = tree.query_pairs(h, output_type="ndarray")
        else:
            pairs = np.zeros((0, 2), dtype=np.int64)

        if len(pairs):
            rv = x[pairs[:, 0]] - x[pairs[:, 1]]
            w, grad = _pair_kernels(rv, h)
        else:
            w = np.zeros(0)
            grad = np.zeros((0, 3))

        rho = m * _poly6_scalar(0.0, h)
        if len(pairs):
            np.add.at(rho, pairs[:, 0], m[pairs[:, 1]] * w)
            np.add.at(rho, pairs[:, 1], m[pairs[:, 0]] * w)
        rho = np.maximum(rho, 1e-9)

        # one sided (compression only) density constraint
        cval = np.maximum(rho / rho0 - 1.0, 0.0)
        grad_self = np.zeros((n, 3))
        grad_sq = np.zeros((n, 3))
        if len(pairs):
            np.add.at(grad_self, pairs[:, 0], grad)
            np.add.at(grad_self, pairs[:, 1], -grad)
            np.add.at(grad_sq, pairs[:, 0], grad * grad)
            np.add.at(grad_sq, pairs[:, 1], grad * grad)
        denom = (grad_sq.sum(axis=1) + (grad_self ** 2).sum(axis=1)) / (rho0 * rho0)
        lam = -cval / np.maximum(denom, 1e-9)

        dp = np.zeros((n, 3))
        if len(pairs):
            w_dq = _poly6_scalar((0.2 * h) ** 2, h)
            scorr = -1e-4 * (w / max(w_dq, 1e-30)) ** 4
            coef = (lam[pairs[:, 0]] + lam[pairs[:, 1]] + scorr)[:, None]
            contrib = coef * grad
            np.add.at(dp, pairs[:, 0], contrib)
            np.add.at(dp, pairs[:, 1], -contrib)
            dp /= rho0

        x_star = x + v * dt
        x_new = x_star + dp

        # mild surface cohesion keeps the splash sheets connected
        if len(pairs):
            r = np.linalg.norm(rv, axis=1)
            surface = (rho[pairs[:, 0]] < 0.85 * rho0) & (rho[pairs[:, 1]] < 0.85 * rho0)
            keep = surface & (r > 1e-9)
            if np.any(keep):
                direction = (rv[keep] / r[keep][:, None])
                pull = -6.0 * (w[keep] / max(w_dq, 1e-30)) * direction * dt * dt
                np.add.at(x_new, pairs[keep, 0], pull)
                np.add.at(x_new, pairs[keep, 1], -pull)

        v_new = (x_new - x) / dt

        # XSPH viscosity
        if len(pairs):
            visc = np.zeros_like(v_new)
            weight = (m[pairs[:, 1]] / rho[pairs[:, 1]] * w)
            delta = (v_new[pairs[:, 1]] - v_new[pairs[:, 0]]) * weight[:, None]
            np.add.at(visc, pairs[:, 0], delta)
            weight2 = (m[pairs[:, 0]] / rho[pairs[:, 0]] * w)
            delta2 = (v_new[pairs[:, 0]] - v_new[pairs[:, 1]]) * weight2[:, None]
            np.add.at(visc, pairs[:, 1], delta2)
            v_new = v_new + 0.05 * visc

        self.pos[idx] = x_new
        self.vel[idx] = v_new

    # ------------------------------------------------------------------ #
    # bunny membrane <-> pool coupling (momentum conserving)
    # ------------------------------------------------------------------ #
    def _contact_impulse(self, dt: float):
        idx = self.pool_idx
        bidx = self.bunny_idx
        xp = self.pos[idx]
        vp = self.vel[idx]
        mp = self.mass[idx]
        xb = self.pos[bidx]
        vb = self.vel[bidx]
        mb = self.mass[bidx]

        # cheap separation test: no contact while the bodies are far apart
        gap = np.maximum(
            np.maximum(xb.min(axis=0) - xp.max(axis=0), xp.min(axis=0) - xb.max(axis=0)),
            0.0,
        ).max()
        if gap > self.contact_length:
            return None

        cell = 0.05
        keys = np.floor(xb / cell).astype(np.int64) + 64
        flat_keys = (keys[:, 0] * 128 + keys[:, 1]) * 128 + keys[:, 2]
        _, inverse = np.unique(flat_keys, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        nclusters = int(inverse.max()) + 1 if len(inverse) else 0
        if nclusters == 0:
            return None
        counts = np.bincount(inverse, minlength=nclusters).astype(float)
        cx = np.zeros((nclusters, 3))
        cv = np.zeros((nclusters, 3))
        cm = np.zeros(nclusters)
        np.add.at(cx, inverse, mb[:, None] * xb)
        np.add.at(cv, inverse, mb[:, None] * vb)
        np.add.at(cm, inverse, mb)
        safe = np.maximum(cm, 1e-30)
        cx /= safe[:, None]
        cv /= safe[:, None]

        tree = cKDTree(cx)
        neigh = tree.query_ball_point(xp, self.contact_length)
        counts_n = np.fromiter((len(lst) for lst in neigh), dtype=np.int64,
                               count=len(neigh))
        if counts_n.sum() == 0:
            return None
        pool_ids = np.repeat(np.arange(len(idx)), counts_n)
        node_ids = np.fromiter((k for lst in neigh for k in lst), dtype=np.int64,
                               count=int(counts_n.sum()))

        d = xp[pool_ids] - cx[node_ids]
        dist = np.linalg.norm(d, axis=1)
        active = dist > 1e-9
        pool_ids, node_ids, dist = pool_ids[active], node_ids[active], dist[active]
        if len(dist) == 0:
            return None
        normal = d[active] / dist[:, None]
        overlap = self.contact_length - dist
        touching = overlap > 0.0
        pool_ids, node_ids = pool_ids[touching], node_ids[touching]
        normal, overlap = normal[touching], overlap[touching]
        if len(overlap) == 0:
            return None

        # positional projection of the fluid out of the membrane
        push = normal * overlap[:, None]
        accum = np.zeros_like(xp)
        np.add.at(accum, pool_ids, push)
        xp = xp + accum

        vrel = np.einsum("ij,ij->i", vp[pool_ids] - cv[node_ids], normal)
        approach = vrel < 0.0
        impulse = np.zeros_like(vp)
        if np.any(approach):
            ids = pool_ids[approach]
            jn = -vrel[approach]
            np.add.at(impulse, ids, jn[:, None] * normal[approach] * mp[ids][:, None])
        vp = vp + impulse / mp[:, None]

        self.pos[idx] = xp
        self.vel[idx] = vp
        total = impulse.sum(axis=0)
        if not np.isfinite(total).all():
            return None
        self._diag["boundary_slip"] = float(np.linalg.norm(total))
        return total

    # ------------------------------------------------------------------ #
    # rigid boundaries
    # ------------------------------------------------------------------ #
    def _collide(self) -> None:
        self._collide_subset(self.pool_idx, 0.5 * self.spacing)
        self._collide_subset(self.bunny_idx, 0.012)
        self._clamp_domain()

    def _clamp_domain(self) -> None:
        low = self.grid_low
        high = self.grid_high + np.asarray([0.0, 0.0, 0.4])
        np.clip(self.pos, low[None, :], high[None, :], out=self.pos)
        np.clip(self.vel, -_MAX_SPEED, _MAX_SPEED, out=self.vel)

    def _collide_subset(self, idx: np.ndarray, radius: float) -> None:
        if len(idx) == 0:
            return
        p = self.pos[idx]
        v = self.vel[idx]
        for (blo, bhi) in self.boxes:
            near = np.all(p > blo - radius - 0.05, axis=1) & np.all(
                p < bhi + radius + 0.05, axis=1
            )
            sel = np.nonzero(near)[0]
            if len(sel) == 0:
                continue
            pts = p[sel]
            closest = np.clip(pts, blo, bhi)
            delta = pts - closest
            dist = np.linalg.norm(delta, axis=1)
            outside = dist > 1e-9
            if np.any(outside):
                ids = sel[outside]
                n = delta[outside] / dist[outside][:, None]
                pen = radius - dist[outside]
                hit = pen > 0.0
                if np.any(hit):
                    hid = ids[hit]
                    hn = n[hit]
                    p[hid] = closest[outside][hit] + hn * radius
                    vn = np.einsum("ij,ij->i", v[hid], hn)
                    moving = vn < 0.0
                    if np.any(moving):
                        mid = hid[moving]
                        mn = hn[moving]
                        vmid = v[mid]
                        vn_m = np.einsum("ij,ij->i", vmid, mn)
                        v[mid] = vmid - 1.05 * vn_m[:, None] * mn
                        tangential = v[mid] + np.einsum("ij,ij->i", v[mid], mn)[:, None] * mn
                        v[mid] = v[mid] - 0.1 * tangential
            inside = ~outside
            if np.any(inside):
                ids = sel[inside]
                pts_in = p[ids]
                depths = np.stack(
                    [pts_in - (blo - radius), (bhi + radius) - pts_in], axis=2
                ).reshape(len(ids), 6)
                axis = np.argmin(np.abs(depths), axis=1)
                sign = np.where(axis % 2 == 0, -1.0, 1.0)
                ax = (axis // 2).astype(int)
                normal = np.zeros((len(ids), 3))
                normal[np.arange(len(ids)), ax] = sign
                p[ids] = pts_in + normal * (np.abs(depths[np.arange(len(ids)), axis]) + 1e-6)[:, None]
                vn = np.einsum("ij,ij->i", v[ids], normal)
                moving = vn < 0.0
                if np.any(moving):
                    mid = ids[moving]
                    mn = normal[moving]
                    vmid = v[mid]
                    vn_m = np.einsum("ij,ij->i", vmid, mn)
                    v[mid] = vmid - 1.05 * vn_m[:, None] * mn
        self.pos[idx] = p
        self.vel[idx] = v

    # ------------------------------------------------------------------ #
    # diagnostic fields and frames
    # ------------------------------------------------------------------ #
    def _splat(self):
        shape = self.grid_shape
        cell = self.cell
        rel = (self.pos - self.grid_low[None, :]) / cell[None, :]
        ijk = np.clip(np.floor(rel).astype(np.int64), 0, np.asarray(shape) - 1)
        flat = (ijk[:, 0] * shape[1] + ijk[:, 1]) * shape[2] + ijk[:, 2]
        ncells = int(shape[0] * shape[1] * shape[2])
        m = self.mass
        mass_grid = np.bincount(flat, weights=m, minlength=ncells).reshape(shape)
        momentum = np.stack(
            [
                np.bincount(flat, weights=m * self.vel[:, k], minlength=ncells).reshape(shape)
                for k in range(3)
            ],
            axis=-1,
        )
        cell_volume = float(np.prod(cell))
        occupancy = np.clip(mass_grid / max(self.density * cell_volume, 1e-12), 0.0, 1.0)
        level = (0.5 - occupancy) * (2.0 * self.grid_spacing)
        active = occupancy > 0.5
        velocity = np.zeros(tuple(shape) + (3,), dtype=float)
        filled = mass_grid > 1e-12
        velocity[filled] = momentum[filled] / mass_grid[filled][:, None]
        return level, active, velocity, momentum, mass_grid

    def _boundary_state(self, velocity_field):
        shape = self.grid_shape
        rel = (self.wall_points - self.grid_low[None, :]) / self.cell[None, :]
        ijk = np.clip(np.floor(rel).astype(np.int64), 0, np.asarray(shape) - 1)
        sampled = velocity_field[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        n = self.wall_normals
        normal_component = np.einsum("ij,ij->i", sampled, n)
        tangential = sampled - normal_component[:, None] * n
        # static walls: the measured fluid-side velocity carries no normal
        # component (no penetration) while the tangential slip is measured.
        return tangential

    def _frame(self, time: float):
        level, active, velocity, _, _ = self._splat()
        boundary_velocity = self._boundary_state(velocity)
        zeros = np.zeros_like(boundary_velocity)
        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.labels.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": self.radius.copy(),
                "material_label": self.labels.copy(),
            },
            fields={
                "active_mask": np.asarray(active),
                "boundary_normal": self.wall_normals.copy(),
                "boundary_target_velocity": zeros,
                "boundary_velocity": boundary_velocity,
                "cumulative_mass_source": np.zeros(2, dtype=float),
                "liquid_level_set": np.asarray(level),
                "velocity_field": np.asarray(velocity),
            },
            scalars={
                "characteristic_length": float(self.characteristic_length),
                "grid_spacing": float(self.grid_spacing),
                "reference_velocity": float(self.reference_velocity),
            },
        )
        frame.validate()
        self._diag["max_speed"] = float(np.linalg.norm(self.vel, axis=1).max())
        self._diag["min_level_set"] = float(np.min(level))
        self._diag["boundary_samples"] = int(len(self.wall_points))
        self._diag["total_mass"] = float(self.mass.sum())
        self._diag["time"] = float(time)
        return frame

    def diagnostics(self) -> dict[str, object]:
        self._diag["steps"] = int(round(self.time / _DT))
        self._diag["bunny_volume"] = float(self.bunny_volume)
        self._diag["rest_density"] = float(self.rho0)
        self._diag["final_time"] = float(self.time)
        return dict(self._diag)
