"""Knitted legwarmer deformation solver (XPBD knit shell, no external engine).

Physical model
--------------
The knitted sleeve is the frozen 17 x 17 periodic tube (289 lumped particles
carrying the frozen total cloth mass).  Knit mechanics are represented by four
families of extensible constraints projected with an XPBD sweep per substep:

* wale springs (axial, along the leg)    - soft, so the outside of the bend can
  stretch the way knit courses do;
* course springs (hoop, periodic rings)  - stiffer, keeps the tube circumference;
* shear diagonals                        - very soft, lets loops rearrange;
* second-neighbour bending               - very soft, so folds/wrinkles form.

Strain limiting emulates the locking of knit loops at large extension, so the
sheet never unravels.  Contact uses analytic signed distance fields of the
frozen colliders (ground box, thigh/shin capsules, knee sphere, foot capsule)
with Coulomb friction evaluated against the kinematic collider velocity, plus a
vertex-vertex repulsion layer with friction for cloth self-contact (bunching).

The knee is driven kinematically: shin and foot rotate rigidly about the knee
sphere centre following a smooth flexion schedule (settle -> flex -> hold ->
partial extension -> hold).  Elastic cuff rings are soft-attached to the limb
segments they sit on, so the sleeve follows the articulation without slipping.

Time stepping is symplectic-Euler prediction + position projection (XPBD),
stable at the substep used here, and fully deterministic.
"""
from __future__ import annotations

import time as _walltime

import numpy as np

try:  # SciPy is part of the allowed base stack; guard keeps the solver runnable.
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

from worldsolver.contracts import StateFrame

PIVOT = np.array([0.0, 0.0, 1.1])
_MOVING_IDS = ("limb_lower", "foot")
def _smoothstep(u: float) -> tuple[float, float]:
    """C1 ease with value and d/du."""
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u), 6.0 * u * (1.0 - u)


def _knee_schedule(t: float) -> tuple[float, float]:
    """Driven knee flexion angle [rad] and angular rate [rad/s]."""
    settle, flex_end, hold_end, ext_end = 0.7, 3.4, 4.3, 5.9
    peak = np.deg2rad(50.0)
    rest = np.deg2rad(12.0)
    if t <= settle:
        return 0.0, 0.0
    if t <= flex_end:
        span = flex_end - settle
        s, ds = _smoothstep((t - settle) / span)
        return peak * s, peak * ds / span
    if t <= hold_end:
        return peak, 0.0
    if t <= ext_end:
        span = ext_end - hold_end
        s, ds = _smoothstep((t - hold_end) / span)
        return peak + (rest - peak) * s, (rest - peak) * ds / span
    return rest, 0.0


def _rot_y(angle: float) -> np.ndarray:
    """Rotation about +y that lifts the +x axis towards +z (knee flexion)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _scatter(out: np.ndarray, idx: np.ndarray, values: np.ndarray) -> None:
    """Accumulate per-constraint corrections onto particles (Jacobi style)."""
    n = out.shape[0]
    for axis in range(3):
        out[:, axis] += np.bincount(idx, weights=values[:, axis], minlength=n)
def _project_distance(pos, winv, ia, ib, rest, alpha_h2, relax, delta):
    """XPBD distance constraint sweep accumulated into ``delta``."""
    d = pos[ia] - pos[ib]
    length = np.sqrt(np.einsum("ij,ij->i", d, d))
    safe = np.maximum(length, 1e-9)
    wa, wb = winv[ia], winv[ib]
    wsum = wa + wb + alpha_h2
    scale = relax * (length - rest) / (wsum * safe)
    corr = scale[:, None] * d
    _scatter(delta, ia, -(wa[:, None]) * corr)
    _scatter(delta, ib, (wb[:, None]) * corr)


def _project_strain_limit(pos, winv, ia, ib, rest, max_ratio, delta):
    """Knit loops lock beyond a maximum extension; clamp over-stretch only."""
    d = pos[ia] - pos[ib]
    length = np.sqrt(np.einsum("ij,ij->i", d, d))
    limit = rest * max_ratio
    over = length > limit
    if not np.any(over):
        return
    ia, ib = ia[over], ib[over]
    d = d[over]
    length = np.maximum(length[over], 1e-9)
    wa, wb = winv[ia], winv[ib]
    wsum = np.maximum(wa + wb, 1e-12)
    corr = ((length - limit[over]) / (wsum * length))[:, None] * d
    _scatter(delta, ia, -(wa[:, None]) * corr)
    _scatter(delta, ib, (wb[:, None]) * corr)


def _project_attachment(pos, winv, idx, targets, alpha_h2, relax):
    """Soft elastic cuff grip onto the limb segment it rides on."""
    d = pos[idx] - targets
    w = winv[idx]
    factor = relax * w / (w + alpha_h2)
    pos[idx] -= factor[:, None] * d


def _capsule_distance(pos, p0, p1, radius):
    """Signed distance / outward normal / closest surface parameter of a capsule."""
    axis = p1 - p0
    denom = float(axis @ axis)
    if denom < 1e-12:
        t = np.zeros(len(pos))
    else:
        t = np.clip((pos - p0) @ axis / denom, 0.0, 1.0)
    closest = p0[None, :] + t[:, None] * axis[None, :]
    d = pos - closest
    length = np.sqrt(np.einsum("ij,ij->i", d, d))
    safe = np.maximum(length, 1e-9)
    normal = d / safe[:, None]
    degenerate = length < 1e-9
    if np.any(degenerate):
        normal[degenerate] = np.array([0.0, 0.0, 1.0])
    return length - radius, normal, closest
def _build_colliders(config) -> list[dict]:
    """Read the frozen boundary geometry into analytic primitives."""
    colliders: list[dict] = []
    geometry = (config or {}).get("scene_geometry") or {}
    for obj in geometry.get("objects", []) or []:
        if obj.get("semantic") != "boundary" or not obj.get("collision_enabled", True):
            continue
        geo = obj.get("geometry", {}) or {}
        kind = geo.get("type")
        transform = obj.get("transform", {}) or {}
        translation = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), float)
        rot = _quat_matrix(transform.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]))
        moving = obj.get("id") in _MOVING_IDS
        if kind == "capsule":
            half = 0.5 * float(geo.get("height", 0.0))
            local_axis = rot @ np.array([0.0, 0.0, 1.0])
            colliders.append({
                "kind": "capsule",
                "p0": translation - half * local_axis,
                "p1": translation + half * local_axis,
                "radius": float(geo.get("radius", 0.1)),
                "moving": moving,
            })
        elif kind == "sphere":
            colliders.append({
                "kind": "sphere",
                "p0": translation.copy(),
                "p1": translation.copy(),
                "radius": float(geo.get("radius", 0.1)),
                "moving": moving,
            })
        elif kind == "box":
            half_extents = np.asarray(geo.get("half_extents", [1.0, 1.0, 0.05]), float)
            colliders.append({
                "kind": "plane",
                "height": float(translation[2] + half_extents[2]),
                "moving": False,
            })
    if not colliders:  # frozen fallback matching scene_geometry.yaml
        colliders = [
            {"kind": "plane", "height": 0.0, "moving": False},
            {"kind": "capsule", "p0": np.array([-1.27, 0.0, 1.1]),
             "p1": np.array([0.03, 0.0, 1.1]), "radius": 0.34, "moving": False},
            {"kind": "capsule", "p0": np.array([-0.03, 0.0, 1.1]),
             "p1": np.array([1.27, 0.0, 1.1]), "radius": 0.31, "moving": True},
            {"kind": "sphere", "p0": PIVOT.copy(), "p1": PIVOT.copy(),
             "radius": 0.36, "moving": False},
            {"kind": "capsule", "p0": np.array([1.19, -0.1, 0.92]),
             "p1": np.array([1.91, -0.1, 0.92]), "radius": 0.25, "moving": True},
        ]
    return colliders
def _pose_colliders(colliders, theta: float, theta_dot: float) -> list[dict]:
    """Articulate the kinematic shin/foot rigidly about the knee centre."""
    rot = _rot_y(-theta)            # positive theta lifts +x towards +z
    omega = np.array([0.0, -theta_dot, 0.0])
    posed = []
    for col in colliders:
        if col["kind"] == "plane":
            posed.append({"kind": "plane", "height": col["height"],
                          "omega": np.zeros(3)})
            continue
        if col["moving"]:
            p0 = PIVOT + rot @ (col["p0"] - PIVOT)
            p1 = PIVOT + rot @ (col["p1"] - PIVOT)
            w = omega
        else:
            p0, p1, w = col["p0"], col["p1"], np.zeros(3)
        posed.append({"kind": col["kind"], "p0": p0, "p1": p1,
                      "radius": col["radius"], "omega": w})
    return posed


def _collider_gaps(pos, posed, offset: float) -> np.ndarray:
    """Signed geometric gap to the closest boundary (negative = penetration)."""
    gap = np.full(len(pos), 1e3)
    for col in posed:
        if col["kind"] == "plane":
            dist = pos[:, 2] - col["height"]
        else:
            dist, _, _ = _capsule_distance(pos, col["p0"], col["p1"], col["radius"])
        gap = np.minimum(gap, dist - offset)
    return gap


def _resolve_contacts(pos, prev, posed, offset: float, mu: float, h: float) -> float:
    """Non-penetration projection with Coulomb friction against moving skin."""
    worst = 0.0
    for col in posed:
        if col["kind"] == "plane":
            dist = pos[:, 2] - col["height"] - offset
            normal = np.zeros_like(pos)
            normal[:, 2] = 1.0
            surface_velocity = np.zeros_like(pos)
        else:
            raw, normal, closest = _capsule_distance(pos, col["p0"], col["p1"],
                                                     col["radius"])
            dist = raw - offset
            surface_velocity = np.cross(col["omega"], closest - PIVOT)
        hit = dist < 0.0
        if not np.any(hit):
            continue
        depth = -dist[hit]
        worst = max(worst, float(depth.max()))
        n = normal[hit]
        pos[hit] += depth[:, None] * n
        # Friction on the tangential slip relative to the kinematic collider.
        rel = (pos[hit] - prev[hit]) - surface_velocity[hit] * h
        tangential = rel - np.einsum("ij,ij->i", rel, n)[:, None] * n
        norm = np.sqrt(np.einsum("ij,ij->i", tangential, tangential))
        limit = mu * depth
        factor = np.where(norm <= limit, 1.0, limit / np.maximum(norm, 1e-12))
        pos[hit] -= factor[:, None] * tangential
    return worst


def _grid_tangents(pos, na: int, nc: int):
    """Material tangents on the (axial u, circumferential v) knit parametrisation."""
    grid = pos.reshape(na, nc, 3)
    tu = np.empty_like(grid)
    tu[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    tu[0] = grid[1] - grid[0]
    tu[-1] = grid[-1] - grid[-2]
    tv = 0.5 * (np.roll(grid, -1, axis=1) - np.roll(grid, 1, axis=1))
    return tu.reshape(-1, 3), tv.reshape(-1, 3)


def _local_shape_matrix(tu, tv):
    """Express both tangents in a per-vertex orthonormal surface frame -> [N,2,2]."""
    nu = np.maximum(np.linalg.norm(tu, axis=1), 1e-12)
    e1 = tu / nu[:, None]
    normal = np.cross(tu, tv)
    nn = np.maximum(np.linalg.norm(normal, axis=1), 1e-12)
    normal = normal / nn[:, None]
    e2 = np.cross(normal, e1)
    matrix = np.empty((len(tu), 2, 2))
    matrix[:, 0, 0] = np.einsum("ij,ij->i", tu, e1)
    matrix[:, 1, 0] = np.einsum("ij,ij->i", tu, e2)
    matrix[:, 0, 1] = np.einsum("ij,ij->i", tv, e1)
    matrix[:, 1, 1] = np.einsum("ij,ij->i", tv, e2)
    return matrix


class BenchmarkSolver:
    """XPBD knit-shell solver for the driven legwarmer articulation."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = dict(getattr(scene, "config", {}) or {})
        cloth = config.get("cloth", {})
        self.na, self.nc = (int(v) for v in cloth["grid_shape"])  # axial, circumferential
        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.mass = np.array(frame0.mass, dtype=float)
        self.winv = 1.0 / np.maximum(self.mass, 1e-12)
        self.body_id = np.zeros(len(self.pos), dtype=np.int32)
        self.boundary = np.asarray(
            frame0.attributes.get("boundary_vertex", np.zeros(len(self.pos))),
            dtype=np.int32)
        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), float)
        self.thickness = float(cloth.get("thickness", 0.015))
        self.offset = 0.5 * self.thickness
        self.sample_dt = float(config.get("sample_dt", 1.0 / 30.0))
        self.time = 0.0

        # --- numerics ---------------------------------------------------
        self.substeps = 25
        self.iterations = 6
        self.relax = 0.5
        self.h = self.sample_dt / self.substeps
        self.mu_body = 0.55          # yarn / skin friction
        self.mu_self = 0.35          # yarn / yarn friction
        self.drag = 0.02             # aerodynamic drag [N s / m]
        self.damping = 0.999
        self.self_radius = 0.05
        self.pair_radius = 0.075
        self.max_strain = 1.5

        self._build_topology()
        self.colliders = _build_colliders(config)

        # deterministic micro-jitter breaks the perfectly symmetric tube so
        # folds can pick a direction; amplitude far below cloth thickness.
        rng = np.random.default_rng(int(seed))
        self.pos += 2.0e-5 * rng.standard_normal(self.pos.shape)

        # reference (rest) shape operator -> F = I on the pristine tube
        tu0, tv0 = _grid_tangents(self.pos, self.na, self.nc)
        self.rest_inv = np.linalg.inv(_local_shape_matrix(tu0, tv0))

        # remove any residual overlap with the frozen colliders at t = 0
        posed = _pose_colliders(self.colliders, 0.0, 0.0)
        for _ in range(30):
            if _resolve_contacts(self.pos, self.pos.copy(), posed,
                                 self.offset, 0.0, self.h) <= 1e-9:
                break
        # elastic cuff rings ride rigidly with the limb they sit on
        self.cuff_prox = np.arange(self.nc)
        self.cuff_dist = np.arange(len(self.pos) - self.nc, len(self.pos))
        self.cuff_prox_rest = self.pos[self.cuff_prox].copy()
        self.cuff_dist_rest = self.pos[self.cuff_dist].copy()
        self._diag = {"substeps": 0, "max_speed": 0.0, "min_gap": 1.0e3,
                      "max_penetration": 0.0, "wall_seconds": 0.0}
        self._pairs = None
        self._pair_age = 0

    def _build_topology(self) -> None:
        """Wale / course / shear / bending constraint sets on the periodic tube."""
        na, nc = self.na, self.nc
        rows = np.arange(na)
        cols = np.arange(nc)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        index = rr * nc + cc
        self.row_of = rr.reshape(-1)
        self.col_of = cc.reshape(-1)

        def pair(a, b):
            return a.reshape(-1).astype(np.intp), b.reshape(-1).astype(np.intp)

        nxt_c = rr * nc + (cc + 1) % nc              # circumferential neighbour
        nxt_r = index[1:, :]                         # axial neighbour target
        axial = pair(index[:-1, :], nxt_r)
        hoop = pair(index, nxt_c)
        # diagonals of every quad: (r,c)-(r+1,c+1) and (r+1,c)-(r,c+1)
        shear_a = pair(index[:-1, :], (rr[:-1] + 1) * nc + (cc[:-1] + 1) % nc)
        shear_b = pair((rr[:-1] + 1) * nc + cc[:-1],
                       rr[:-1] * nc + (cc[:-1] + 1) % nc)
        bend_a = pair(index[:-2, :], index[2:, :])
        bend_c = pair(index, rr * nc + (cc + 2) % nc)

        # (ia, ib, rest, compliance, apply_strain_limit)
        specs = [
            (axial, 6.0e-3, True),    # wales: soft, knit stretches along the leg
            (hoop, 1.5e-3, True),     # courses: stiffer, holds the tube girth
            (shear_a, 3.0e-2, False),  # loop rearrangement
            (shear_b, 3.0e-2, False),
            (bend_a, 4.0e-1, False),  # very low bending stiffness -> folds
            (bend_c, 4.0e-1, False),
        ]
        self.constraints = []
        h2 = self.h * self.h
        for (ia, ib), alpha, limited in specs:
            rest = np.linalg.norm(self.pos[ia] - self.pos[ib], axis=1)
            self.constraints.append({
                "ia": ia, "ib": ib, "rest": rest,
                "alpha_h2": alpha / h2, "limited": bool(limited),
            })
        self.alpha_attach_h2 = 1.0e-3 / h2

    def _substep(self, t_start: float, h: float) -> None:
        """One symplectic prediction + XPBD projection substep."""
        theta, theta_dot = _knee_schedule(t_start + h)
        posed = _pose_colliders(self.colliders, theta, theta_dot)
        rot = _rot_y(-theta)

        # external forces: gravity + light aerodynamic drag
        force = self.mass[:, None] * self.gravity[None, :] - self.drag * self.vel
        self.vel = (self.vel + h * force * self.winv[:, None]) * self.damping
        prev = self.pos.copy()
        self.pos = self.pos + h * self.vel

        prox_target = self.cuff_prox_rest
        dist_target = PIVOT + (self.cuff_dist_rest - PIVOT) @ rot.T

        delta = np.empty_like(self.pos)
        for _ in range(self.iterations):
            delta[:] = 0.0
            for con in self.constraints:
                _project_distance(self.pos, self.winv, con["ia"], con["ib"],
                                  con["rest"], con["alpha_h2"], self.relax, delta)
            self.pos += delta
            _project_attachment(self.pos, self.winv, self.cuff_prox, prox_target,
                                self.alpha_attach_h2, self.relax)
            _project_attachment(self.pos, self.winv, self.cuff_dist, dist_target,
                                self.alpha_attach_h2, self.relax)

        # knit loops lock at large extension: no unravelling, no infinite stretch
        delta[:] = 0.0
        for con in self.constraints:
            if con["limited"]:
                _project_strain_limit(self.pos, self.winv, con["ia"], con["ib"],
                                      con["rest"], self.max_strain, delta)
        self.pos += delta

        self._resolve_self_contact(prev)
        pen = _resolve_contacts(self.pos, prev, posed, self.offset, self.mu_body, h)

        self.vel = (self.pos - prev) / h
        np.clip(self.vel, -40.0, 40.0, out=self.vel)
        self._diag["substeps"] += 1
        self._diag["max_penetration"] = max(self._diag["max_penetration"], pen)
        self._diag["max_speed"] = max(self._diag["max_speed"],
                                      float(np.abs(self.vel).max()))

    def _resolve_self_contact(self, prev) -> None:
        """Yarn-level self contact: repulsion + tangential friction between folds."""
        if cKDTree is None:
            return
        if self._pairs is None or self._pair_age >= 5:
            tree = cKDTree(self.pos)
            pairs = tree.query_pairs(self.pair_radius, output_type="ndarray")
            if len(pairs):
                ia, ib = pairs[:, 0], pairs[:, 1]
                dr = np.abs(self.row_of[ia] - self.row_of[ib])
                dc = np.abs(self.col_of[ia] - self.col_of[ib])
                dc = np.minimum(dc, self.nc - dc)
                far = (dr + dc) > 2
                pairs = pairs[far]
            self._pairs = pairs
            self._pair_age = 0
        self._pair_age += 1
        pairs = self._pairs
        if pairs is None or len(pairs) == 0:
            return
        ia, ib = pairs[:, 0], pairs[:, 1]
        d = self.pos[ia] - self.pos[ib]
        length = np.sqrt(np.einsum("ij,ij->i", d, d))
        hit = length < self.self_radius
        if not np.any(hit):
            return
        ia, ib = ia[hit], ib[hit]
        d = d[hit]
        length = np.maximum(length[hit], 1e-9)
        n = d / length[:, None]
        depth = self.self_radius - length
        delta = np.zeros_like(self.pos)
        corr = 0.5 * depth[:, None] * n
        rel = (self.pos[ia] - prev[ia]) - (self.pos[ib] - prev[ib])
        tangential = rel - np.einsum("ij,ij->i", rel, n)[:, None] * n
        norm = np.sqrt(np.einsum("ij,ij->i", tangential, tangential))
        limit = self.mu_self * depth
        factor = 0.5 * np.where(norm <= limit, 1.0,
                                limit / np.maximum(norm, 1e-12))
        friction = factor[:, None] * tangential
        _scatter(delta, ia, corr - friction)
        _scatter(delta, ib, -corr + friction)
        self.pos += delta
    def advance_to(self, time: float):
        """Step the knit shell forward to ``time`` and emit the sampled state."""
        target = float(time)
        started = _walltime.perf_counter()
        while self.time < target - 1e-12:
            h = min(self.h, target - self.time)
            if h <= 1e-12:
                break
            self._substep(self.time, h)
            self.time += h
        self.time = max(self.time, target)
        self._diag["wall_seconds"] += _walltime.perf_counter() - started
        return self._frame(target)

    def _frame(self, t: float) -> StateFrame:
        """Assemble the StateFrame with every required raw trajectory field."""
        pos = np.nan_to_num(self.pos, nan=0.0, posinf=1e3, neginf=-1e3)
        vel = np.clip(np.nan_to_num(self.vel, nan=0.0, posinf=40.0, neginf=-40.0),
                      -40.0, 40.0)
        self.pos, self.vel = pos, vel

        theta, theta_dot = _knee_schedule(t)
        posed = _pose_colliders(self.colliders, theta, theta_dot)
        gap = _collider_gaps(pos, posed, self.offset)
        self._diag["min_gap"] = min(self._diag["min_gap"], float(gap.min()))

        external = self.mass[:, None] * self.gravity[None, :] - self.drag * vel

        tu, tv = _grid_tangents(pos, self.na, self.nc)
        shape = _local_shape_matrix(tu, tv)
        F = np.clip(np.matmul(shape, self.rest_inv), -50.0, 50.0)

        frame = StateFrame(
            time=float(t),
            position=pos.copy(),
            velocity=vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": np.ascontiguousarray(F),
            },
            fields={
                "contact_signed_distance": gap.astype(float),
                "external_force": external,
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "contact_length_scale": float(self.thickness),
                "knee_flexion_angle": float(theta),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "XPBD knit shell (wale/course/shear/bend + strain limiting)",
            "substeps_taken": int(self._diag["substeps"]),
            "substep_h": float(self.h),
            "iterations_per_substep": int(self.iterations),
            "particles": int(len(self.pos)),
            "max_speed": float(self._diag["max_speed"]),
            "max_collider_penetration": float(self._diag["max_penetration"]),
            "min_contact_signed_distance": float(self._diag["min_gap"]),
            "solver_wall_seconds": float(self._diag["wall_seconds"]),
            "self_contact": bool(cKDTree is not None),
        }
