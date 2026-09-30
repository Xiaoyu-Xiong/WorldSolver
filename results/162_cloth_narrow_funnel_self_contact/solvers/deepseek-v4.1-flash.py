"""Physically-based solver for "Cloth through a Narrow Funnel".

Model
-----
The cloth is a thin inextensible triangulated membrane simulated with a
substepped position-based (XPBD) time integrator:

  * every substep the velocity is advanced with the frozen gravity field
    (symplectic Euler prediction ``x <- x + h v``);
  * membrane (structural + shear) and bending distance constraints are
    projected with per-constraint compliance (``alpha``) in a Gauss-Seidel
    sweep, which is the standard position-based discretisation of a thin
    shell's elastic energy;
  * the sheet is kept free of self-intersection by vertex/triangle
    self-contact projection with a cloth-thickness separation band;
  * the rigid funnel (a set of convex wall blocks parsed from the frozen
    ``assets/funnel.obj``) and the ground box are handled with convex
    half-space projections, so no rim penetration or tunnelling occurs;
  * Coulomb-like tangential friction and normal inelasticity are applied at
    every active contact, which reproduces the characteristic folding and
    jamming behaviour of cloth forced through a narrow throat.

No animation, keyframes or precomputed motion tables are used: every frame is
the result of actual time stepping of the physical model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from worldsolver.contracts import StateFrame

_WORKSPACE = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# frozen asset helpers
# --------------------------------------------------------------------------
def _load_obj(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            faces.append([int(token.split("/")[0]) - 1 for token in parts[1:]])
    return np.asarray(vertices, dtype=float), faces


def _connected_blocks(n_vertices: int, faces: list[list[int]]) -> list[np.ndarray]:
    """Split an indexed mesh into vertex-connected components (funnel walls)."""
    parent = list(range(n_vertices))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for face in faces:
        if len(face) < 3:
            continue
        first = face[0]
        for other in face[1:]:
            ra, rb = find(first), find(other)
            if ra != rb:
                parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for index in range(n_vertices):
        groups.setdefault(find(index), []).append(index)
    return [np.asarray(v, dtype=int) for v in groups.values() if len(v) >= 4]


class _Block:
    """One convex rigid piece of the frozen funnel, as outward plane set."""

    __slots__ = ("normals", "offsets")

    def __init__(self, normals: np.ndarray, offsets: np.ndarray) -> None:
        self.normals = normals
        self.offsets = offsets

    def signed_distance(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (gap, outward normal) for points; gap < 0 means penetration."""
        planes = points @ self.normals.T + self.offsets[None, :]
        index = np.argmax(planes, axis=1)
        return planes[np.arange(len(points)), index], self.normals[index]


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
class BenchmarkSolver:
    """XPBD thin-shell cloth interacting with a rigid narrow funnel."""

    #: number of internal substeps per exported sample
    SUBSTEPS_PER_SAMPLE = 16
    #: Gauss-Seidel projection sweeps over the constraint set per substep
    ITERATIONS = 8

    def __init__(self) -> None:
        self._ready = False
        self._steps = 0

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        config = scene.config
        cloth = config["cloth"]
        self.thickness = float(cloth["thickness"])
        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.nx, self.ny = (int(v) for v in cloth["grid_shape"])

        frame = scene.initial_state()
        x = np.asarray(frame.position, dtype=float).copy()
        self.n = int(x.shape[0])
        self.mass = np.asarray(frame.mass, dtype=float).copy()
        self.w = 1.0 / self.mass
        self.rest = x.copy()
        self.rest[:, 2] = 0.0
        self.boundary = np.asarray(
            frame.attributes.get("boundary_vertex", np.zeros(self.n)), dtype=np.int32
        )

        self.x = x
        self.v = np.zeros_like(x)
        self._xprev = np.zeros_like(x)
        self.forbidden: np.ndarray  # set in _build_topology

        self.faces = self._cloth_faces()
        self._build_topology()
        self._build_constraints()

        # contact / material constants
        self.gap_static = 0.5 * self.thickness
        self.self_dist = self.thickness
        self.contact_mask = np.zeros(self.n, dtype=bool)
        self.contact_dir = np.zeros((self.n, 3), dtype=float)

        # time stepping
        sample_dt = float(config.get("sample_dt", 1.0 / 30.0))
        self.h = sample_dt / float(self.SUBSTEPS_PER_SAMPLE)
        self._damp = float(np.exp(-0.6 * self.h))
        self._fric = float(np.exp(-2.5 * self.h))
        self.vmax = 15.0
        self.pos_min = np.array([-8.0, -8.0, -0.25])
        self.pos_max = np.array([8.0, 8.0, 8.0])

        self._mount_funnel()
        self._ready = True

    def _cloth_faces(self) -> np.ndarray:
        asset = _WORKSPACE / "assets" / "cloth_topology.json"
        if asset.is_file():
            try:
                data = json.loads(asset.read_text(encoding="utf-8"))
                faces = np.asarray(data["faces"], dtype=np.int64)
                if faces.ndim == 2 and faces.shape[1] == 3 and faces.max() < self.n:
                    return faces
            except (KeyError, ValueError):
                pass
        nx, ny = self.nx, self.ny
        faces = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                a = iy * nx + ix
                b = a + 1
                c = a + nx
                d = c + 1
                faces.append([a, b, d])
                faces.append([a, d, c])
        return np.asarray(faces, dtype=np.int64)

    def _build_topology(self) -> None:
        """Rest metric of each triangle and the self-contact exclusion mask."""
        tri = self.faces
        self.tri = tri
        e1 = self.rest[tri[:, 1]] - self.rest[tri[:, 0]]
        e2 = self.rest[tri[:, 2]] - self.rest[tri[:, 0]]
        n3 = np.cross(e1, e2)
        a_rest = np.empty((len(tri), 3, 3), dtype=float)
        a_rest[:, :, 0] = e1
        a_rest[:, :, 1] = e2
        a_rest[:, :, 2] = n3
        self.inv_rest = np.linalg.inv(a_rest)

        # conservative self-collision cull radius
        lengths = np.stack(
            [
                np.linalg.norm(self.rest[tri[:, 1]] - self.rest[tri[:, 0]], axis=1),
                np.linalg.norm(self.rest[tri[:, 2]] - self.rest[tri[:, 1]], axis=1),
                np.linalg.norm(self.rest[tri[:, 0]] - self.rest[tri[:, 2]], axis=1),
            ],
            axis=1,
        )
        semiperimeter = 0.5 * lengths.sum(axis=1)
        area = 0.5 * np.linalg.norm(n3, axis=1)
        area = np.maximum(area, 1e-12)
        circum = lengths.prod(axis=1) / (4.0 * area)
        self._cull_r = float(circum.max()) + self.self_dist + 0.02
        self._cull_r2 = self._cull_r * self._cull_r

        ring = np.zeros((self.n, self.n), dtype=bool)
        ring[np.arange(self.n), np.arange(self.n)] = True
        for j in range(3):
            a = tri[:, j]
            b = tri[:, (j + 1) % 3]
            ring[a, b] = True
            ring[b, a] = True
        forbidden = np.zeros((self.n, len(tri)), dtype=bool)
        for t in range(len(tri)):
            forbidden[ring[tri[t]].any(axis=0), t] = True
        self.forbidden = forbidden

    def _build_constraints(self) -> None:
        nx, ny = self.nx, self.ny
        groups: list[tuple[list[int], list[int], list[float], float]] = [
            ([], [], [], 1.0e-9),  # structural: essentially inextensible
            ([], [], [], 5.0e-4),  # shear: stiff weave
            ([], [], [], 0.15),    # bending: compliant, lets the sheet fold
        ]
        structural = groups[0]
        shear = groups[1]
        bending = groups[2]

        def add(group, a, b):
            group[0].append(a)
            group[1].append(b)

        for iy in range(ny):
            for ix in range(nx):
                node = iy * nx + ix
                if ix + 1 < nx:
                    add(structural, node, node + 1)
                if iy + 1 < ny:
                    add(structural, node, node + nx)
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                node = iy * nx + ix
                add(shear, node, node + nx + 1)
                add(shear, node + 1, node + nx)
        for iy in range(ny):
            for ix in range(nx):
                node = iy * nx + ix
                if ix + 2 < nx:
                    add(bending, node, node + 2)
                if iy + 2 < ny:
                    add(bending, node, node + 2 * nx)

        self.groups = []
        for ci, cj, _, alpha in groups:
            i = np.asarray(ci, dtype=np.int64)
            j = np.asarray(cj, dtype=np.int64)
            rest = np.linalg.norm(self.rest[i] - self.rest[j], axis=1)
            self.groups.append(
                {
                    "i": i,
                    "j": j,
                    "rest": rest,
                    "alpha": float(alpha),
                    "lam": np.zeros(len(i)),
                }
            )

    def _mount_funnel(self) -> None:
        """Build the rigid funnel from the frozen asset as convex wall blocks."""
        self.hulls: list[_Block] = []
        asset = _WORKSPACE / "assets" / "funnel.obj"
        if not asset.is_file():
            return
        vertices, faces = _load_obj(asset)
        if len(vertices) == 0 or not faces:
            return
        offset = np.asarray([0.0, 0.0, 0.0])
        for block in _connected_blocks(len(vertices), faces):
            points = vertices[block] + offset[None, :]
            if len(points) < 4:
                continue
            try:
                hull = ConvexHull(points)
            except Exception:
                continue
            normals = np.asarray(hull.equations[:, :3], dtype=float)
            offsets = np.asarray(hull.equations[:, 3], dtype=float)
            norms = np.linalg.norm(normals, axis=1)
            keep = norms > 1e-12
            self.hulls.append(_Block(normals[keep] / norms[keep, None], offsets[keep] / norms[keep]))

    # ------------------------------------------------------------------
    # numerics
    # ------------------------------------------------------------------
    def _solve_group(self, group: dict) -> None:
        x = self.x
        i = group["i"]
        j = group["j"]
        if len(i) == 0:
            return
        delta = x[i] - x[j]
        length = np.linalg.norm(delta, axis=1)
        safe = np.maximum(length, 1e-12)
        direction = delta / safe[:, None]
        value = length - group["rest"]
        alpha_tilde = group["alpha"] / (self.h * self.h)
        denom = self.w[i] + self.w[j] + alpha_tilde
        dlam = (-value - alpha_tilde * group["lam"]) / denom
        group["lam"] += dlam
        update = direction * dlam[:, None]
        np.add.at(x, i, self.w[i, None] * update)
        np.add.at(x, j, -self.w[j, None] * update)

    def _collide_static(self) -> None:
        """Project the sheet out of the rigid funnel walls and the ground."""
        x = self.x
        for hull in self.hulls:
            gap, normal = hull.signed_distance(x)
            push = self.gap_static - gap
            active = push > 0.0
            if np.any(active):
                x[active] += push[active, None] * normal[active]
                self._register_contact(active, normal)

        below = x[:, 2] < self.gap_static
        if np.any(below):
            x[below, 2] = self.gap_static
            self._register_contact(below, np.array([0.0, 0.0, 1.0]))

    def _register_contact(self, mask: np.ndarray, normal: np.ndarray) -> None:
        self.contact_mask |= mask
        self.contact_dir[mask] += normal

    @staticmethod
    def _closest_on_triangle(
        p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Closest point on a triangle (vectorised) plus its barycentric weights."""
        ab = b - a
        ac = c - a
        bc = c - b
        ap = p - a
        bp = p - b
        d1 = np.einsum("ij,ij->i", ab, ap)
        d2 = np.einsum("ij,ij->i", ac, ap)
        d3 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-20)
        d4 = np.einsum("ij,ij->i", ac, ab)
        d5 = np.maximum(np.einsum("ij,ij->i", ac, ac), 1e-20)
        d6 = np.einsum("ij,ij->i", bc, bp)
        den = d3 * d5 - d4 * d4
        den = np.where(np.abs(den) < 1e-20, 1e-20, den)
        v = (d5 * d1 - d4 * d2) / den
        w = (d3 * d2 - d4 * d1) / den
        t = np.clip(d1 / d3, 0.0, 1.0)
        u = np.clip(d2 / d5, 0.0, 1.0)
        s = np.clip(d6 / np.maximum(np.einsum("ij,ij->i", bc, bc), 1e-20), 0.0, 1.0)
        zero = np.zeros_like(v)
        one = np.ones_like(v)

        candidates = [
            (a, np.stack([one, zero, zero], axis=1)),
            (b, np.stack([zero, one, zero], axis=1)),
            (c, np.stack([zero, zero, one], axis=1)),
            (a + t[:, None] * ab, np.stack([one - t, t, zero], axis=1)),
            (a + u[:, None] * ac, np.stack([one - u, zero, u], axis=1)),
            (b + s[:, None] * bc, np.stack([zero, one - s, s], axis=1)),
            (a + v[:, None] * ab + w[:, None] * ac, np.stack([one - v - w, v, w], axis=1)),
        ]
        points = np.stack([item[0] for item in candidates], axis=1)
        weights = np.stack([item[1] for item in candidates], axis=1)
        distance = ((points - p[:, None, :]) ** 2).sum(axis=2)
        best = np.argmin(distance, axis=1)
        rows = np.arange(len(p))
        return points[rows, best], weights[rows, best]

    def _self_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.x
        centroids = x[self.tri].mean(axis=1)
        diff = x[:, None, :] - centroids[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        candidate = (dist2 <= self._cull_r2) & (~self.forbidden)
        vi, ti = np.nonzero(candidate)
        return vi, ti

    def _self_contact(self) -> None:
        vi, ti = self._self_pairs()
        if vi.size == 0:
            return
        tri = self.tri
        for _ in range(2):
            p = self.x[vi]
            a = self.x[tri[ti, 0]]
            b = self.x[tri[ti, 1]]
            c = self.x[tri[ti, 2]]
            point, bary = self._closest_on_triangle(p, a, b, c)
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal, axis=1)
            good = norm > 1e-14
            normal = normal / np.where(good, norm, 1.0)[:, None]
            delta = np.einsum("ij,ij->i", p - point, normal)
            sign = np.where(delta >= 0.0, 1.0, -1.0)
            value = sign * delta - self.self_dist
            active = (value < 0.0) & good
            if not np.any(active):
                return
            idx = np.nonzero(active)[0]
            nk = sign[idx, None] * normal[idx]
            wi = self.w[vi[idx]]
            wj = np.zeros_like(wi)
            for j in range(3):
                wj += bary[idx, j] ** 2 * self.w[tri[ti[idx], j]]
            dlam = -value[idx] / np.maximum(wi + wj, 1e-30)
            np.add.at(self.x, vi[idx], (wi * dlam)[:, None] * nk)
            for j in range(3):
                share = -(dlam * bary[idx, j]) * self.w[tri[ti[idx], j]]
                np.add.at(self.x, tri[ti[idx], j], share[:, None] * nk)
            self._register_contact(vi[idx], nk)
            for j in range(3):
                self._register_contact(tri[ti[idx], j], -nk)

    def _apply_friction(self) -> None:
        mask = self.contact_mask
        if not np.any(mask):
            return
        v = self.v
        direction = self.contact_dir[mask]
        norm = np.linalg.norm(direction, axis=1)
        keep = norm > 1e-12
        if not np.any(keep):
            return
        rows = np.nonzero(mask)[0][keep]
        e = direction[keep] / norm[keep, None]
        velocity = v[rows]
        normal_speed = np.einsum("ij,ij->i", velocity, e)
        tangential = velocity - normal_speed[:, None] * e
        velocity = self._fric * tangential
        approach = np.einsum("ij,ij->i", velocity, e)
        velocity = velocity - np.minimum(approach, 0.0)[:, None] * e
        v[rows] = velocity

    def _substep(self) -> None:
        h = self.h
        x, v = self.x, self.v
        v += h * self.gravity[None, :]
        v *= self._damp
        np.clip(v, -self.vmax, self.vmax, out=v)
        np.copyto(self._xprev, x)
        x += h * v

        for group in self.groups:
            group["lam"][:] = 0.0
        self.contact_mask[:] = False
        self.contact_dir[:] = 0.0

        for iteration in range(self.ITERATIONS):
            for group in self.groups:
                self._solve_group(group)
            if iteration >= self.ITERATIONS - 2:
                self._self_contact()
            self._collide_static()

        v[:] = (x - self._xprev) / h
        np.clip(v, -self.vmax, self.vmax, out=v)
        self._apply_friction()
        np.clip(x, self.pos_min[None, :], self.pos_max[None, :], out=x)
        self._steps += 1

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def _deformation_gradient(self) -> np.ndarray:
        x = self.x
        tri = self.tri
        e1 = x[tri[:, 1]] - x[tri[:, 0]]
        e2 = x[tri[:, 2]] - x[tri[:, 0]]
        current = np.empty((len(tri), 3, 3), dtype=float)
        current[:, :, 0] = e1
        current[:, :, 1] = e2
        current[:, :, 2] = np.cross(e1, e2)
        per_triangle = current @ self.inv_rest
        per_triangle = np.nan_to_num(per_triangle, nan=0.0, posinf=0.0, neginf=0.0)
        per_vertex = np.zeros((self.n, 3, 3), dtype=float)
        count = np.zeros(self.n, dtype=float)
        for j in range(3):
            np.add.at(per_vertex, tri[:, j], per_triangle)
            np.add.at(count, tri[:, j], 1.0)
        count = np.maximum(count, 1.0)
        per_vertex /= count[:, None, None]
        bad = ~np.isfinite(per_vertex).all(axis=(1, 2))
        if np.any(bad):
            per_vertex[bad] = np.eye(3)
        return per_vertex

    def _contact_gaps(self) -> np.ndarray:
        x = self.x
        gaps = np.full(self.n, np.inf)
        for hull in self.hulls:
            gap, _ = hull.signed_distance(x)
            np.minimum(gaps, gap, out=gaps)
        np.minimum(gaps, x[:, 2], out=gaps)
        vi, ti = self._self_pairs()
        if vi.size:
            tri = self.tri
            a = x[tri[ti, 0]]
            b = x[tri[ti, 1]]
            c = x[tri[ti, 2]]
            point, _ = self._closest_on_triangle(x[vi], a, b, c)
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal, axis=1)
            good = norm > 1e-14
            normal = normal / np.where(good, norm, 1.0)[:, None]
            delta = np.einsum("ij,ij->i", x[vi] - point, normal)
            gap = np.abs(delta) - self.self_dist
            np.minimum.at(gaps, vi[good], gap[good])
        return np.asarray(gaps, dtype=float)

    def _frame(self, time: float) -> StateFrame:
        mass = self.mass
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self._contact_gaps(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": mass[:, None] * self.gravity[None, :],
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    # public contract
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if not self._ready:
            raise RuntimeError("initialize() must be called before advance_to()")
        target = int(round(float(time) / self.h))
        while self._steps < target:
            self._substep()
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps": int(self._steps),
            "substep_dt": float(self.h),
            "vertices": int(self.n),
            "triangles": int(len(self.tri)),
            "constraints": int(sum(len(g["i"]) for g in self.groups)),
        }
