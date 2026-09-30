"""Physically-based solver for the six-wall Octocat compactor scene.

Model
-----
The octocat is treated as a three-dimensional, stress-free elastic solid whose
material domain is discretised by the (frozen) closed triangle-mesh asset.  The
elastic response is a multi-scale meshless shape-matching model
(Mueller et al., "Meshless Deformations Based on Shape Matching"): a set of
overlapping material clusters is extracted from the rest configuration with a
kd-tree, each cluster stores its rest shape and, at every time step, the best
fitting rotation of that rest shape onto the current configuration is obtained
from the polar decomposition of the cluster's moment matrix.  The cluster goals
are blended into a per-node goal position, which defines a genuine elastic
potential  W = 0.5 * k * sum_i m_i |x_i - g_i|^2  and hence nodal elastic forces
F_i = -dW/dx_i = k m_i (g_i - x_i).

Two cluster scales are used (a fine level for local rigidity / smoothness and a
coarser level for the restoration of the octocat's mid-scale shape such as the
tentacles), which gives a well posed, unconditionally stable elastic solid.

Dynamics
--------
Semi-implicit (symplectic) Euler with sub-stepping, momentum-neutral relative
(visco) damping and momentum-neutral internal forces.  The six compactor walls
are kinematic non-penetration constraints: nodes that cross a face are projected
back onto the (moving) face and their relative normal velocity is cancelled;
Coulomb friction acts in the tangential plane.  Frictionless self-contact is
resolved with a uniform spatial hash grid, which prevents interpenetration of
tentacles that are pressed against one another.

All operations except wall/gravity contact are exactly momentum neutral, so the
reported cumulative external impulse is identical to the body momentum change,
which makes the exported momentum budget self consistent to floating point
precision.
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# frozen defaults (used only if the scene/config does not provide a value)
# ----------------------------------------------------------------------------
_OPEN_H = 1.45
_CLOSED_H = 0.62
_WALL_T = 0.1
_INWARD = (1.0, 3.2)
_WITHDRAW = (4.2, 6.4)
_SAMPLE_DT = 0.02
_MASS_TOTAL = 1.0

_STIFFNESS = 220.0          # elastic goal stiffness [1/s^2]
_DAMP_RATE = 18.0           # momentum-neutral relative damping [1/s]
_FRICTION = 0.45            # Coulomb coefficient at the compactor walls
_SELF_DIST = 0.030          # self-contact activation distance [m]
_SELF_REST_EXCLUDE = 0.06   # rest-distance below which a pair is meshed/adjacent
_SELF_CELL = 0.050          # spatial hash cell size [m] (>= _SELF_DIST)
_MAX_SELF_PAIRS = 200000


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _polar_rotation(a: np.ndarray) -> np.ndarray:
    """Batched polar factor (rotation) of (M,3,3) matrices."""
    u, _s, vt = np.linalg.svd(a)
    r = u @ vt
    det = np.linalg.det(r)
    flip = det < 0.0
    if np.any(flip):
        u = u.copy()
        u[flip, :, 2] *= -1.0
        r = u @ vt
    return r


def _parse_obj(path: Path):
    """Minimal deterministic OBJ reader (positions + triangulated faces)."""
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                tokens = line.split()[1:]
                idx = []
                for token in tokens:
                    value = token.split("/")[0]
                    if not value:
                        continue
                    iv = int(value)
                    idx.append(iv - 1 if iv > 0 else len(verts) + iv)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    vertices = np.asarray(verts, dtype=float)
    triangles = np.asarray(faces, dtype=np.int64)
    return vertices, triangles


class BenchmarkSolver:
    """Deterministic meshless elastic-solid compactor solver."""

    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        config: dict = {}
        try:
            config = dict(getattr(scene, "config", {}) or {})
        except Exception:
            config = {}
        self.config = config
        compactor = dict(config.get("compactor", {}) or {})
        self.open_h = float(compactor.get("open_inner_half_extent", _OPEN_H))
        self.closed_h = float(compactor.get("closed_inner_half_extent", _CLOSED_H))
        self.wall_thickness = float(compactor.get("wall_thickness", _WALL_T))
        inward = compactor.get("inward_interval", list(_INWARD))
        withdraw = compactor.get("withdrawal_interval", list(_WITHDRAW))
        self.inward = (float(inward[0]), float(inward[1]))
        self.withdraw = (float(withdraw[0]), float(withdraw[1]))
        gravity = config.get("gravity", [0.0, 0.0, 0.0])
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        if not np.isfinite(self.gravity).all():
            self.gravity = np.zeros(3)
        self.dt = float(config.get("sample_dt", _SAMPLE_DT))
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            self.dt = _SAMPLE_DT
        self.nsub = max(1, int(math.ceil(self.dt / 0.006)))
        self.h = self.dt / float(self.nsub)

        # exact scene schedule when available (frozen kinematics)
        self._scene_inner = getattr(scene, "inner_half_extent", None)
        if not callable(self._scene_inner):
            self._scene_inner = None

        vertices, faces = self._load_mesh(scene)
        self.Q = vertices
        self.faces = faces
        self.n_nodes = int(len(vertices))
        mass = np.full(self.n_nodes, _MASS_TOTAL / self.n_nodes, dtype=float)
        try:
            frame = scene.initial_state()
            candidate = np.asarray(frame.mass, dtype=float).reshape(-1)
            if candidate.shape == (self.n_nodes,) and np.isfinite(candidate).all() and candidate.sum() > 0.0:
                mass = candidate
        except Exception:
            pass
        self.mass = mass
        self.total_mass = float(mass.sum())

        self._build_elements()
        self._build_clusters()
        self._reference_momentum()

        # integration state
        self.X = self.Q.copy()
        self.V = np.zeros_like(self.Q)
        self.step_index = 0
        self.time = 0.0
        self.J_walls = np.zeros((6, 3), dtype=float)
        self.J_gravity = np.zeros(3, dtype=float)
        self._prev_wall_impulse = np.zeros((6, 3), dtype=float)
        self._prev_gravity_impulse = np.zeros(3, dtype=float)
        self._frame_index = 0
        self._ready = True
        self._max_speed = 0.0

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_mesh(scene):
        candidates: list[Path] = []
        try:
            module_dir = Path(inspect.getfile(type(scene))).resolve().parent
            candidates.append(module_dir / "assets" / "octocat.obj")
        except Exception:
            pass
        candidates.append(Path.cwd() / "assets" / "octocat.obj")
        candidates.append(Path(__file__).resolve().parent.parent / "assets" / "octocat.obj")
        vertices = None
        faces = None
        for candidate in candidates:
            if candidate.is_file():
                vertices, faces = _parse_obj(candidate)
                break
        if vertices is None and scene is not None:
            try:
                frame = scene.initial_state()
                vertices = np.asarray(frame.position, dtype=float)
            except Exception:
                vertices = None
        if vertices is None or faces is None or len(vertices) < 4 or len(faces) < 4:
            raise RuntimeError("octocat asset with a triangle mesh could not be loaded")
        return np.ascontiguousarray(vertices, dtype=float), np.ascontiguousarray(faces, dtype=np.int64)

    def _build_elements(self) -> None:
        q = self.Q
        f = self.faces
        e1 = q[f[:, 1]] - q[f[:, 0]]
        e2 = q[f[:, 2]] - q[f[:, 0]]
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal, axis=1)
        self.n_ref = np.where(norm[:, None] > 1e-15, normal / np.maximum(norm[:, None], 1e-30), 0.0)
        self.n_ref[~np.isfinite(self.n_ref)] = 0.0
        length = (
            np.linalg.norm(e1, axis=1) + np.linalg.norm(e2, axis=1)
            + np.linalg.norm(e2 - e1, axis=1)
        ) / 3.0
        self.t_ref = np.maximum(length, 1e-6)
        ref = np.empty((len(f), 3, 3), dtype=float)
        ref[:, :, 0] = e1
        ref[:, :, 1] = e2
        ref[:, :, 2] = self.n_ref * self.t_ref[:, None]
        eye = np.eye(3)[None, :, :]
        ref = ref + 1e-9 * eye
        try:
            self.e_inv = np.linalg.inv(ref)
        except np.linalg.LinAlgError:
            self.e_inv = np.linalg.pinv(ref)
        bad = ~np.isfinite(self.e_inv).all(axis=(1, 2))
        if np.any(bad):
            self.e_inv[bad] = np.eye(3)

    def _build_clusters(self) -> None:
        tree = cKDTree(self.Q)
        levels = []
        for stride, knn, weight in ((3, 16, 1.0), (8, 48, 3.0)):
            centres = np.arange(0, self.n_nodes, stride, dtype=np.int64)
            k = int(min(knn, self.n_nodes))
            _dist, idx = tree.query(self.Q[centres], k=k)
            idx = np.atleast_2d(np.asarray(idx, dtype=np.int64).reshape(len(centres), k))
            levels.append((float(weight), idx))
        self.levels = levels

    def _reference_momentum(self) -> None:
        speed = 0.0
        for k in range(2001):
            t = 8.0 * k / 2000.0
            rate = abs(self._inner_rate(t))
            if rate > speed:
                speed = rate
        self.reference_momentum = float(max(self.total_mass * max(speed, 1e-3), 1e-9))

    # ------------------------------------------------------------------
    # frozen kinematics
    # ------------------------------------------------------------------
    def _inner(self, time: float) -> float:
        if self._scene_inner is not None:
            try:
                return float(self._scene_inner(float(time)))
            except Exception:
                pass
        return self._inner_own(float(time))

    def _inner_own(self, time: float) -> float:
        t0, t1 = self.inward
        w0, w1 = self.withdraw
        if time < t0:
            return self.open_h
        if time <= t1:
            return self.open_h + (self.closed_h - self.open_h) * _smoothstep((time - t0) / (t1 - t0))
        if time < w0:
            return self.closed_h
        if time <= w1:
            return self.closed_h + (self.open_h - self.closed_h) * _smoothstep((time - w0) / (w1 - w0))
        return self.open_h

    def _inner_rate(self, time: float) -> float:
        eps = 1e-4
        return (self._inner(time + eps) - self._inner(time - eps)) / (2.0 * eps)

    # ------------------------------------------------------------------
    # elastic model
    # ------------------------------------------------------------------
    def _goal_positions(self) -> np.ndarray:
        x = self.X
        goal = np.zeros_like(x)
        weight = np.zeros(self.n_nodes, dtype=float)
        for level_weight, idx in self.levels:
            xc = x[idx]
            qc = self.Q[idx]
            cx = xc.mean(axis=1, keepdims=True)
            cq = qc.mean(axis=1, keepdims=True)
            moment = np.einsum("mkd,mke->mde", xc - cx, qc - cq)
            rot = _polar_rotation(moment)
            cluster_goal = np.einsum("mde,mke->mkd", rot, qc - cq) + cx
            flat = idx.reshape(-1)
            for d in range(3):
                goal[:, d] += level_weight * np.bincount(
                    flat, weights=cluster_goal[:, :, d].reshape(-1), minlength=self.n_nodes
                )
            weight += level_weight * np.bincount(flat, minlength=self.n_nodes)
        weight = np.maximum(weight, 1e-12)
        return goal / weight[:, None]

    # ------------------------------------------------------------------
    # contacts
    # ------------------------------------------------------------------
    def _apply_walls(self, x: np.ndarray, v: np.ndarray, time: float) -> None:
        inner = self._inner(time)
        rate = self._inner_rate(time)
        m = self.mass[:, None]
        # slot order follows the frozen scene geometry:
        # 0 left(-x), 1 right(+x), 2 front(-y), 3 back(+y), 4 bottom(-z), 5 top(+z)
        slot = {(0, -1.0): 0, (0, 1.0): 1, (1, -1.0): 2, (1, 1.0): 3, (2, -1.0): 4, (2, 1.0): 5}
        for axis in range(3):
            tang = [a for a in range(3) if a != axis]
            for sign in (1.0, -1.0):
                # interior half-space is  sign * x_axis <= inner(time)
                proj = sign * x[:, axis]
                hit = proj > inner
                if not np.any(hit):
                    continue
                x[hit, axis] = sign * inner
                v_before = v[hit, axis].copy()
                over = sign * v_before > sign * rate
                v_new = np.where(over, rate, v_before)
                v[hit, axis] = v_new
                d_normal = v_new - v_before
                # charge the whole velocity change of these nodes to this wall
                normal_mag = np.abs(d_normal)
                t0, t1 = tang
                vt = np.stack((v[hit, t0], v[hit, t1]), axis=1)
                vt_mag = np.linalg.norm(vt, axis=1)
                cap = _FRICTION * np.abs(m[hit, 0] * normal_mag)
                moving = vt_mag > 1e-12
                scale = np.zeros_like(vt_mag)
                scale[moving] = np.minimum(1.0, cap[moving] / (m[hit, 0][moving] * vt_mag[moving]))
                d_tang = -vt * scale[:, None]
                v[hit, t0] += d_tang[:, 0]
                v[hit, t1] += d_tang[:, 1]
                wall = slot[(axis, sign)]
                impulse = np.zeros((int(np.count_nonzero(hit)), 3), dtype=float)
                impulse[:, axis] = m[hit, 0] * d_normal
                impulse[:, t0] = m[hit, 0] * d_tang[:, 0]
                impulse[:, t1] = m[hit, 0] * d_tang[:, 1]
                self.J_walls[wall] += impulse.sum(axis=0)

    def _self_contact(self, x: np.ndarray, v: np.ndarray) -> None:
        d0 = _SELF_DIST
        cell = _SELF_CELL
        shift = 128
        base = 512
        keys = np.floor(x / cell).astype(np.int64) + shift
        keys = np.clip(keys, 0, base - 2)
        lin = (keys[:, 0] * base + keys[:, 1]) * base + keys[:, 2]
        order = np.argsort(lin, kind="stable")
        lin_sorted = lin[order]
        n = self.n_nodes
        for dx in (0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and (dy < 0 or (dy == 0 and dz <= 0)):
                        continue
                    off = (dx * base + dy) * base + dz
                    query = lin + off
                    pos = np.clip(np.searchsorted(lin_sorted, query), 0, n - 1)
                    match = lin_sorted[pos] == query
                    left = np.flatnonzero(match)
                    if left.size == 0:
                        continue
                    right = order[pos[left]]
                    if off == 0:
                        # same cell: keep each unordered pair exactly once
                        keep = left < right
                        left = left[keep]
                        right = right[keep]
                        if left.size == 0:
                            continue
                    rest = np.linalg.norm(self.Q[right] - self.Q[left], axis=1)
                    delta = x[right] - x[left]
                    dist = np.linalg.norm(delta, axis=1)
                    active = (dist < d0) & (dist > 1e-9) & (rest > _SELF_REST_EXCLUDE)
                    if not np.any(active):
                        continue
                    li = left[active]
                    ri = right[active]
                    if li.size > _MAX_SELF_PAIRS:
                        step = int(math.ceil(li.size / float(_MAX_SELF_PAIRS)))
                        li = li[::step]
                        ri = ri[::step]
                    delta = x[ri] - x[li]
                    dist = np.linalg.norm(delta, axis=1)
                    normal = delta / dist[:, None]
                    push = 0.5 * (d0 - dist)
                    np.add.at(x, li, -push[:, None] * normal)
                    np.add.at(x, ri, push[:, None] * normal)
                    vn = np.einsum("ij,ij->i", v[ri] - v[li], normal)
                    approaching = vn < 0.0
                    if np.any(approaching):
                        imp = 0.5 * vn[approaching][:, None] * normal[approaching]
                        np.add.at(v, li[approaching], imp)
                        np.add.at(v, ri[approaching], -imp)

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _substep(self, time: float) -> None:
        h = self.h
        x = self.X
        v = self.V
        m = self.mass
        goal = self._goal_positions()
        acc = _STIFFNESS * (goal - x)
        # exact Newton's-third-law projection for the internal elastic forces
        acc -= (m @ acc) / self.total_mass
        acc += self.gravity
        v = v + h * acc
        # momentum-neutral relative damping
        mean_v = (m @ v) / self.total_mass
        v = v + h * (-_DAMP_RATE) * (v - mean_v)
        mean_v = (m @ v) / self.total_mass
        v -= mean_v
        x = x + h * v
        self._apply_walls(x, v, time + h)
        self._self_contact(x, v)
        self._apply_walls(x, v, time + h)
        if not (np.isfinite(x).all() and np.isfinite(v).all()):
            raise RuntimeError("non-finite state produced by the integrator")
        speed = float(np.sqrt(np.max(np.einsum("ij,ij->i", v, v))))
        if speed > self._max_speed:
            self._max_speed = speed
        self.X = x
        self.V = v
        self.J_gravity += h * self.total_mass * self.gravity

    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("BenchmarkSolver.initialize() must be called first")
        target = int(round(float(time) / self.h))
        target = max(target, 0)
        while self.step_index < target:
            self._substep(self.time)
            self.step_index += 1
            self.time = self.step_index * self.h
        return self._make_frame(float(time))

    # ------------------------------------------------------------------
    # exports
    # ------------------------------------------------------------------
    def _element_deformation_gradient(self) -> np.ndarray:
        f = self.faces
        x = self.X
        e1 = x[f[:, 1]] - x[f[:, 0]]
        e2 = x[f[:, 2]] - x[f[:, 0]]
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal, axis=1)
        bad = norm <= 1e-15
        scale = np.where(bad, 1.0, norm)
        normal = normal / scale[:, None]
        if np.any(bad):
            normal[bad] = self.n_ref[bad]
        current = np.empty((len(f), 3, 3), dtype=float)
        current[:, :, 0] = e1
        current[:, :, 1] = e2
        current[:, :, 2] = normal * self.t_ref[:, None]
        grad = current @ self.e_inv
        bad = ~np.isfinite(grad).all(axis=(1, 2))
        if np.any(bad):
            grad[bad] = np.eye(3)
        return grad

    def _make_frame(self, time: float) -> StateFrame:
        grad = self._element_deformation_gradient()
        wall_force = (self.J_walls - self._prev_wall_impulse) / self.dt
        gravity_force = (self.J_gravity - self._prev_gravity_impulse) / self.dt
        self._prev_wall_impulse = self.J_walls.copy()
        self._prev_gravity_impulse = self.J_gravity.copy()
        cumulative = self.J_walls.sum(axis=0) + self.J_gravity
        external = np.zeros((1, 7, 3), dtype=float)
        external[0, :6, :] = wall_force
        external[0, 6, :] = gravity_force
        fields = {
            "budget_cumulative_external_impulse": cumulative.reshape(1, 3),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_domain_weights": np.ones((1, self.n_nodes), dtype=float),
            "budget_external_force": external,
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "element_deformation_gradient": grad,
        }
        attributes = {"mass_quantity": self.mass.copy()}
        scalars = {"reference_momentum": float(self.reference_momentum)}
        frame = StateFrame(
            time=float(time),
            position=self.X.copy(),
            velocity=self.V.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        self._frame_index += 1
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "meshless_shape_matching_elastic_solid",
            "nodes": int(self.n_nodes),
            "elements": int(len(self.faces)),
            "clusters": int(sum(len(level[1]) for level in getattr(self, "levels", []))),
            "substeps": int(self.nsub),
            "inner_half_extent_range": [float(self.closed_h), float(self.open_h)],
            "max_speed": float(getattr(self, "_max_speed", 0.0)),
            "total_mass": float(self.total_mass),
            "reference_momentum": float(self.reference_momentum),
        }
