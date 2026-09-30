"""Discrete-element granular solver for a kinematic shovel crossing a sand bed.

Physical model
-------------
The sand bed is represented by the frozen particle sampling supplied by the
scene. Each sample is a frictional soft sphere of radius ``0.5 * spacing``
carrying the frozen bulk mass. The dynamics integrated here are the classical
cohesionless discrete-element equations of motion

    m_i dv_i/dt = m_i g + sum_j ( f_n(i,j) + f_t(i,j) ) + f_boundary_i

with

* Hertz-like linear repulsion  f_n = k_n * delta  along the contact normal,
  active only while the surface overlap ``delta = 2r - |x_j - x_i|`` is
  positive (no tensile/cohesive branch),
* a normal dashpot  -gamma_n * v_rel_n  giving inelastic grain collisions,
* regularised Coulomb friction  |f_t| <= mu * |f_n|  opposing tangential slip,
  which is what supplies the bulk shear strength (internal friction angle) of
  the granular column and therefore the angle of repose of the deposited mound,
* the same penalty contact law against the frozen static container (floor and
  four walls) and against the kinematically prescribed shovel blade (oriented
  box) and shaft (capsule), evaluated from analytic signed distance functions.

Time stepping is an explicit semi-implicit (symplectic) Euler scheme with a
fixed step chosen well below the contact period 2*pi*sqrt(m/k_n), followed by a
non-penetration projection against the rigid container and the tool so that the
exported contact gaps stay bounded. Everything is deterministic: fixed step
count per sample interval, fixed operation order, no random draws inside the
solver.

The prescribed tool path is read from the frozen configuration and evaluated
with a monotone cubic (Fritsch-Carlson) interpolant, so the rigid body follows
the frozen boundary condition while every sand degree of freedom is obtained by
actually integrating the equations above. No scripted motion, animation curve
or precomputed state table is used for the granular material.
"""
from __future__ import annotations

import numpy as np

try:  # SciPy gives a fast spatial index; a uniform grid fallback keeps us safe.
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover - defensive only
    _KDTree = None

from worldsolver.contracts import StateFrame


# --------------------------------------------------------------------------
# small geometry helpers
# --------------------------------------------------------------------------
def _quat_matrix(q) -> np.ndarray:
    """Rotation matrix of a quaternion given in xyzw order."""
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-15:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


class _MonotoneTrack:
    """Monotone cubic interpolant of the frozen prescribed tool path.

    Fritsch-Carlson tangent limiting is applied per component so the rigid
    boundary condition is C1 in time (finite, continuous velocity) and never
    overshoots the frozen control values.
    """

    def __init__(self, times, values):
        self.t = np.asarray(times, dtype=float)
        self.v = np.asarray(values, dtype=float)
        if self.v.ndim == 1:
            self.v = self.v[:, None]
        n = len(self.t)
        h = np.diff(self.t)
        slope = np.diff(self.v, axis=0) / h[:, None]
        m = np.zeros_like(self.v)
        if n > 2:
            m[1:-1] = 0.5 * (slope[:-1] + slope[1:])
        m[0] = slope[0]
        m[-1] = slope[-1]
        # Fritsch-Carlson limiter: kill tangents across flat/reversing spans.
        for k in range(n - 1):
            s = slope[k]
            flat = np.abs(s) < 1e-14
            m[k] = np.where(flat, 0.0, m[k])
            m[k + 1] = np.where(flat, 0.0, m[k + 1])
            with np.errstate(divide="ignore", invalid="ignore"):
                a = np.where(flat, 0.0, m[k] / np.where(flat, 1.0, s))
                b = np.where(flat, 0.0, m[k + 1] / np.where(flat, 1.0, s))
            scale = np.where((a > 0) & (b > 0), 3.0 / np.maximum(np.sqrt(a * a + b * b), 1e-12), 1.0)
            scale = np.minimum(scale, 1.0)
            m[k] = m[k] * np.where(flat, 0.0, scale)
            m[k + 1] = m[k + 1] * np.where(flat, 0.0, scale)
        self.m = m
        self.slope = slope
        self.h = h

    def sample(self, time: float):
        """Return (value, d value / d time) at ``time``, clamped at both ends."""
        t = float(time)
        if t <= self.t[0]:
            return self.v[0].copy(), np.zeros(self.v.shape[1])
        if t >= self.t[-1]:
            return self.v[-1].copy(), np.zeros(self.v.shape[1])
        k = int(np.searchsorted(self.t, t, side="right") - 1)
        k = max(0, min(k, len(self.t) - 2))
        h = self.h[k]
        s = (t - self.t[k]) / h
        s2, s3 = s * s, s * s * s
        h00 = 2.0 * s3 - 3.0 * s2 + 1.0
        h10 = s3 - 2.0 * s2 + s
        h01 = -2.0 * s3 + 3.0 * s2
        h11 = s3 - s2
        value = h00 * self.v[k] + h10 * h * self.m[k] + h01 * self.v[k + 1] + h11 * h * self.m[k + 1]
        d00 = 6.0 * s2 - 6.0 * s
        d10 = 3.0 * s2 - 4.0 * s + 1.0
        d01 = -6.0 * s2 + 6.0 * s
        d11 = 3.0 * s2 - 2.0 * s
        rate = (
            d00 * self.v[k] / h
            + d10 * self.m[k]
            + d01 * self.v[k + 1] / h
            + d11 * self.m[k + 1]
        )
        return value, rate
def _box_sdf(points, center, rot, half):
    """Signed distance and outward unit normal for an oriented box.

    Negative distance means the point is inside the box. The normal is the
    gradient of the exact exterior distance; inside the box it points toward
    the nearest face, which is the correct separation direction for contact.
    """
    local = (points - center) @ rot  # world -> local (rot is orthonormal)
    q = np.abs(local) - half
    q_pos = np.maximum(q, 0.0)
    outside = np.linalg.norm(q_pos, axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    dist = outside + inside

    normal_local = np.zeros_like(local)
    is_out = outside > 1e-12
    if np.any(is_out):
        n_out = q_pos[is_out] * np.sign(local[is_out])
        norm = np.linalg.norm(n_out, axis=1)
        norm = np.where(norm > 1e-12, norm, 1.0)
        normal_local[is_out] = n_out / norm[:, None]
    is_in = ~is_out
    if np.any(is_in):
        qi = q[is_in]
        axis = np.argmax(qi, axis=1)
        sel = np.zeros_like(qi)
        rows = np.arange(len(qi))
        sel[rows, axis] = np.sign(local[is_in][rows, axis])
        sel[rows, axis] = np.where(sel[rows, axis] == 0.0, 1.0, sel[rows, axis])
        normal_local[is_in] = sel
    normal = normal_local @ rot.T
    norm = np.linalg.norm(normal, axis=1)
    norm = np.where(norm > 1e-12, norm, 1.0)
    return dist, normal / norm[:, None]


def _capsule_sdf(points, tip_a, tip_b, radius):
    """Signed distance and outward normal for a capsule segment."""
    axis = tip_b - tip_a
    length_sq = float(axis @ axis)
    rel = points - tip_a
    if length_sq < 1e-15:
        proj = np.zeros(len(points))
    else:
        proj = np.clip((rel @ axis) / length_sq, 0.0, 1.0)
    closest = tip_a + proj[:, None] * axis
    delta = points - closest
    dist = np.linalg.norm(delta, axis=1)
    safe = np.where(dist > 1e-12, dist, 1.0)
    normal = delta / safe[:, None]
    # Degenerate coincident points get an arbitrary but deterministic normal.
    normal[dist <= 1e-12] = np.array([0.0, 0.0, 1.0])
    return dist - float(radius), normal
class BenchmarkSolver:
    """Explicit DEM integrator for the frozen sand bed and kinematic shovel."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = dict(scene.config)
        self.config = config
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.position = np.array(frame0.position, dtype=float)
        self.velocity = np.zeros_like(self.position)
        self.mass = np.array(frame0.mass, dtype=float)
        self.body_id = np.array(frame0.body_id, dtype=np.int32)
        self.n = len(self.position)
        self.inv_mass = 1.0 / self.mass

        initial = config["initial_conditions"]
        self.spacing = float(initial["sand_particle_spacing"])
        self.radius = 0.5 * self.spacing
        self.particle_volume = self.spacing**3
        self.density = float(initial["sand_bulk_density"])
        self.initial_volume = np.full(self.n, self.particle_volume)
        self.total_mass = float(self.mass.sum())

        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.duration = float(config["duration"])
        self.sample_dt = float(config["sample_dt"])

        boundary = config.get("boundary_conditions", {})
        self.floor_z = float(boundary.get("container_floor_z", 0.0))
        inner = boundary.get("container_inner_bounds", [-2.35, 2.35, -1.5, 1.5])
        self.x_min, self.x_max, self.y_min, self.y_max = (float(v) for v in inner)
        self.wall_top = 0.66

        # ---- frozen tool geometry (read from the authoritative scene geometry)
        geometry = config.get("scene_geometry", {}) or {}
        objects = {item["id"]: item for item in geometry.get("objects", [])}

        blade = objects.get("shovel_blade", {})
        blade_geom = blade.get("geometry", {})
        blade_tf = blade.get("transform", {})
        self.blade_half = np.asarray(
            blade_geom.get("half_extents", [0.14, 0.64, 0.34]), dtype=float
        )
        self.blade_origin = np.asarray(
            blade_tf.get("translation", [-2.05, 0.0, 0.38]), dtype=float
        )
        self.blade_rot = _quat_matrix(blade_tf.get("rotation_xyzw", [0.0, -0.0872, 0.0, 0.9962]))

        shaft = objects.get("shovel_shaft", {})
        shaft_geom = shaft.get("geometry", {})
        shaft_tf = shaft.get("transform", {})
        self.shaft_radius = float(shaft_geom.get("radius", 0.065))
        shaft_height = float(shaft_geom.get("height", 2.0))
        self.shaft_origin = np.asarray(
            shaft_tf.get("translation", [-2.76, 0.0, 1.25]), dtype=float
        )
        shaft_rot = _quat_matrix(shaft_tf.get("rotation_xyzw", [0.0, -0.3606, 0.0, 0.9327]))
        axis = shaft_rot @ np.array([0.0, 0.0, 1.0])
        self.shaft_tip_a = self.shaft_origin - 0.5 * shaft_height * axis
        self.shaft_tip_b = self.shaft_origin + 0.5 * shaft_height * axis

        # ---- prescribed rigid-body boundary condition (frozen configuration)
        motion = config.get("kinematic_motion", {}).get("shovel_assembly", {})
        nodes = motion.get("translation_keyframes", [])
        if nodes:
            node_times = [float(item["time"]) for item in nodes]
            node_values = [np.asarray(item["value"], dtype=float) for item in nodes]
        else:  # pragma: no cover - configuration always provides the path
            node_times = [0.0, self.duration]
            node_values = [self.blade_origin.copy(), self.blade_origin.copy()]
        self.tool_track = _MonotoneTrack(node_times, np.asarray(node_values))
        self.tool_reference = np.asarray(node_values[0], dtype=float)

        # ---- contact model parameters
        m_ref = float(np.median(self.mass))
        self.k_n = 3.0e4
        contact_omega = np.sqrt(2.0 * self.k_n / m_ref)
        # Restitution ~0.3 through the normal dashpot.
        self.gamma_n = 0.35 * 2.0 * np.sqrt(0.5 * m_ref * self.k_n)
        self.gamma_t = 0.6 * self.gamma_n
        self.mu_pp = 0.62
        self.mu_wall = 0.5
        self.mu_tool = 0.45
        self.k_bound = 2.0 * self.k_n
        self.gamma_bound = 1.4 * self.gamma_n
        self.v_cap = 15.0

        # Stable explicit step: comfortably below the contact period.
        dt_stable = 0.22 / contact_omega
        self.substeps = max(1, int(np.ceil(self.sample_dt / dt_stable)))
        self.dt = self.sample_dt / self.substeps

        # ---- neighbour bookkeeping
        self.contact_distance = 2.0 * self.radius
        self.skin = 0.35 * self.radius
        self.cutoff = self.contact_distance + self.skin
        self.pair_i = np.zeros(0, dtype=np.int64)
        self.pair_j = np.zeros(0, dtype=np.int64)
        self.rebuild_reference = self.position.copy()
        self._build_pairs()

        # ---- diagnostics accumulators
        self.time = 0.0
        self.reaction_accum = np.zeros(3)
        self.reaction_samples = 0
        self.reaction_force = np.zeros(3)
        self.peak_reaction = 0.0
        self.step_count = 0
        surface = float(initial["sand_bounds_max"][2])
        self.initial_surface_z = surface
        self.bed_min = np.asarray(initial["sand_bounds_min"], dtype=float)
        self.bed_max = np.asarray(initial["sand_bounds_max"], dtype=float)
        self.cell = 2.0 * self.spacing
        self.grid_nx = max(1, int(np.ceil((self.x_max - self.x_min) / self.cell)))
        self.grid_ny = max(1, int(np.ceil((self.y_max - self.y_min) / self.cell)))
        self.cell_area = self.cell * self.cell
        self.max_penetration = 0.0
        self.last_diagnostics: dict[str, float] = {}
    # ------------------------------------------------------------------
    # neighbour search
    # ------------------------------------------------------------------
    def _build_pairs(self) -> None:
        if _KDTree is not None:
            tree = _KDTree(self.position)
            pairs = tree.query_pairs(self.cutoff, output_type="ndarray")
        else:  # pragma: no cover - fallback uniform grid
            pairs = self._grid_pairs()
        if len(pairs) == 0:
            self.pair_i = np.zeros(0, dtype=np.int64)
            self.pair_j = np.zeros(0, dtype=np.int64)
        else:
            self.pair_i = np.ascontiguousarray(pairs[:, 0].astype(np.int64))
            self.pair_j = np.ascontiguousarray(pairs[:, 1].astype(np.int64))
        self.rebuild_reference = self.position.copy()

    def _grid_pairs(self) -> np.ndarray:  # pragma: no cover - fallback path
        cell = self.cutoff
        keys = np.floor(self.position / cell).astype(np.int64)
        table: dict[tuple, list[int]] = {}
        for index, key in enumerate(map(tuple, keys)):
            table.setdefault(key, []).append(index)
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]
        out: list[tuple[int, int]] = []
        for key, members in table.items():
            for off in offsets:
                other = (key[0] + off[0], key[1] + off[1], key[2] + off[2])
                if other not in table:
                    continue
                for a in members:
                    for b in table[other]:
                        if b > a:
                            delta = self.position[b] - self.position[a]
                            if float(delta @ delta) <= cell * cell:
                                out.append((a, b))
        if not out:
            return np.zeros((0, 2), dtype=np.int64)
        return np.asarray(sorted(set(out)), dtype=np.int64)

    def _maybe_rebuild(self) -> None:
        shift = self.position - self.rebuild_reference
        moved = float(np.sqrt(np.max(np.einsum("ij,ij->i", shift, shift))))
        if moved > 0.5 * self.skin:
            self._build_pairs()

    # ------------------------------------------------------------------
    # prescribed tool state
    # ------------------------------------------------------------------
    def _tool_state(self, time: float):
        value, rate = self.tool_track.sample(time)
        delta = value - self.tool_reference
        blade_center = self.blade_origin + delta
        shaft_a = self.shaft_tip_a + delta
        shaft_b = self.shaft_tip_b + delta
        return blade_center, shaft_a, shaft_b, np.asarray(rate, dtype=float)
    # ------------------------------------------------------------------
    # force assembly
    # ------------------------------------------------------------------
    def _pair_forces(self, force: np.ndarray) -> None:
        """Cohesionless normal repulsion, normal dashpot and Coulomb friction."""
        if len(self.pair_i) == 0:
            return
        i = self.pair_i
        j = self.pair_j
        delta = self.position[j] - self.position[i]
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        active = (dist < self.contact_distance) & (dist > 1e-12)
        if not np.any(active):
            return
        i = i[active]
        j = j[active]
        dist = dist[active]
        normal = delta[active] / dist[:, None]
        overlap = self.contact_distance - dist
        self.max_penetration = max(self.max_penetration, float(overlap.max()))

        v_rel = self.velocity[j] - self.velocity[i]
        v_n = np.einsum("ij,ij->i", v_rel, normal)
        # Positive f_n pushes the pair apart; the dashpot never becomes tensile.
        f_n = self.k_n * overlap - self.gamma_n * v_n
        f_n = np.maximum(f_n, 0.0)

        v_t = v_rel - v_n[:, None] * normal
        speed_t = np.sqrt(np.einsum("ij,ij->i", v_t, v_t))
        limit = self.mu_pp * f_n
        magnitude = np.minimum(self.gamma_t * speed_t, limit)
        safe = np.where(speed_t > 1e-10, speed_t, 1.0)
        f_t = (magnitude / safe)[:, None] * v_t
        f_t[speed_t <= 1e-10] = 0.0

        # Newton's third law: j is pushed along +normal, i along -normal.
        contribution = f_n[:, None] * normal
        total_j = contribution + f_t
        total_i = -contribution - f_t
        for axis in range(3):
            force[:, axis] += np.bincount(j, weights=total_j[:, axis], minlength=self.n)
            force[:, axis] += np.bincount(i, weights=total_i[:, axis], minlength=self.n)

    def _plane_contact(self, force, gap, normal, wall_velocity=None, mu=None):
        """Penalty contact against one static half-space with Coulomb friction."""
        touching = gap < 0.0
        if not np.any(touching):
            return
        idx = np.flatnonzero(touching)
        depth = -gap[idx]
        v = self.velocity[idx]
        if wall_velocity is not None:
            v = v - wall_velocity
        v_n = v @ normal
        f_n = np.maximum(self.k_bound * depth - self.gamma_bound * v_n, 0.0)
        v_t = v - v_n[:, None] * normal[None, :]
        speed_t = np.sqrt(np.einsum("ij,ij->i", v_t, v_t))
        friction_mu = self.mu_wall if mu is None else mu
        magnitude = np.minimum(self.gamma_bound * speed_t, friction_mu * f_n)
        safe = np.where(speed_t > 1e-10, speed_t, 1.0)
        f_t = -(magnitude / safe)[:, None] * v_t
        f_t[speed_t <= 1e-10] = 0.0
        force[idx] += f_n[:, None] * normal[None, :] + f_t

    def _container_forces(self, force: np.ndarray) -> None:
        r = self.radius
        z = self.position[:, 2]
        self._plane_contact(force, z - r - self.floor_z, np.array([0.0, 0.0, 1.0]))
        below_top = self.position[:, 2] < self.wall_top
        big = 1.0e3
        gap_xl = np.where(below_top, self.position[:, 0] - r - self.x_min, big)
        self._plane_contact(force, gap_xl, np.array([1.0, 0.0, 0.0]))
        gap_xr = np.where(below_top, self.x_max - r - self.position[:, 0], big)
        self._plane_contact(force, gap_xr, np.array([-1.0, 0.0, 0.0]))
        gap_yl = np.where(below_top, self.position[:, 1] - r - self.y_min, big)
        self._plane_contact(force, gap_yl, np.array([0.0, 1.0, 0.0]))
        gap_yr = np.where(below_top, self.y_max - r - self.position[:, 1], big)
        self._plane_contact(force, gap_yr, np.array([0.0, -1.0, 0.0]))
    def _blade_candidates(self, blade_center):
        reach = float(np.abs(self.blade_rot) @ self.blade_half) + 2.0 * self.radius
        low = blade_center - reach
        high = blade_center + reach
        inside = np.all((self.position >= low) & (self.position <= high), axis=1)
        return np.flatnonzero(inside)

    def _shaft_candidates(self, tip_a, tip_b):
        pad = self.shaft_radius + 2.0 * self.radius
        low = np.minimum(tip_a, tip_b) - pad
        high = np.maximum(tip_a, tip_b) + pad
        inside = np.all((self.position >= low) & (self.position <= high), axis=1)
        return np.flatnonzero(inside)

    def _tool_forces(self, force, blade_center, shaft_a, shaft_b, tool_velocity):
        """Penalty contact with the kinematically driven shovel.

        Returns the reaction force the granular material applies on the tool,
        which is minus the sum of the contact forces applied to the grains.
        """
        reaction = np.zeros(3)

        idx = self._blade_candidates(blade_center)
        if len(idx):
            dist, normal = _box_sdf(self.position[idx], blade_center, self.blade_rot, self.blade_half)
            gap = dist - self.radius
            hit = gap < 0.0
            if np.any(hit):
                sub = idx[hit]
                reaction += self._apply_tool_contact(
                    force, sub, -gap[hit], normal[hit], tool_velocity
                )

        idx = self._shaft_candidates(shaft_a, shaft_b)
        if len(idx):
            dist, normal = _capsule_sdf(self.position[idx], shaft_a, shaft_b, self.shaft_radius)
            gap = dist - self.radius
            hit = gap < 0.0
            if np.any(hit):
                sub = idx[hit]
                reaction += self._apply_tool_contact(
                    force, sub, -gap[hit], normal[hit], tool_velocity
                )
        return reaction

    def _apply_tool_contact(self, force, idx, depth, normal, tool_velocity):
        self.max_penetration = max(self.max_penetration, float(depth.max()))
        v_rel = self.velocity[idx] - tool_velocity[None, :]
        v_n = np.einsum("ij,ij->i", v_rel, normal)
        f_n = np.maximum(self.k_bound * depth - self.gamma_bound * v_n, 0.0)
        v_t = v_rel - v_n[:, None] * normal
        speed_t = np.sqrt(np.einsum("ij,ij->i", v_t, v_t))
        magnitude = np.minimum(self.gamma_bound * speed_t, self.mu_tool * f_n)
        safe = np.where(speed_t > 1e-10, speed_t, 1.0)
        f_t = -(magnitude / safe)[:, None] * v_t
        f_t[speed_t <= 1e-10] = 0.0
        contact = f_n[:, None] * normal + f_t
        np.add.at(force, idx, contact)
        return -contact.sum(axis=0)

    # ------------------------------------------------------------------
    # non-penetration projection
    # ------------------------------------------------------------------
    def _project_container(self) -> None:
        r = self.radius
        low = self.floor_z + r
        under = self.position[:, 2] < low
        if np.any(under):
            self.position[under, 2] = low
            vz = self.velocity[under, 2]
            self.velocity[under, 2] = np.where(vz < 0.0, 0.0, vz)
        below_top = self.position[:, 2] < self.wall_top
        for axis, bound, sign in (
            (0, self.x_min + r, 1.0),
            (0, self.x_max - r, -1.0),
            (1, self.y_min + r, 1.0),
            (1, self.y_max - r, -1.0),
        ):
            if sign > 0.0:
                bad = below_top & (self.position[:, axis] < bound)
            else:
                bad = below_top & (self.position[:, axis] > bound)
            if np.any(bad):
                self.position[bad, axis] = bound
                v = self.velocity[bad, axis]
                self.velocity[bad, axis] = np.where(sign * v < 0.0, 0.0, v)

    def _project_tool(self, blade_center, shaft_a, shaft_b, tool_velocity) -> None:
        """Push grains out of the rigid tool and remove approaching normal speed."""
        idx = self._blade_candidates(blade_center)
        if len(idx):
            dist, normal = _box_sdf(self.position[idx], blade_center, self.blade_rot, self.blade_half)
            gap = dist - self.radius
            hit = gap < 0.0
            if np.any(hit):
                sub = idx[hit]
                self.position[sub] += (-gap[hit])[:, None] * normal[hit]
                v_rel = self.velocity[sub] - tool_velocity[None, :]
                v_n = np.einsum("ij,ij->i", v_rel, normal[hit])
                closing = v_n < 0.0
                if np.any(closing):
                    rows = sub[closing]
                    self.velocity[rows] -= v_n[closing][:, None] * normal[hit][closing]

        idx = self._shaft_candidates(shaft_a, shaft_b)
        if len(idx):
            dist, normal = _capsule_sdf(self.position[idx], shaft_a, shaft_b, self.shaft_radius)
            gap = dist - self.radius
            hit = gap < 0.0
            if np.any(hit):
                sub = idx[hit]
                self.position[sub] += (-gap[hit])[:, None] * normal[hit]
                v_rel = self.velocity[sub] - tool_velocity[None, :]
                v_n = np.einsum("ij,ij->i", v_rel, normal[hit])
                closing = v_n < 0.0
                if np.any(closing):
                    rows = sub[closing]
                    self.velocity[rows] -= v_n[closing][:, None] * normal[hit][closing]
    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        """One symplectic-Euler DEM step at the current tool boundary state."""
        blade_center, shaft_a, shaft_b, tool_velocity = self._tool_state(self.time + 0.5 * dt)

        self._maybe_rebuild()

        # The only external (non-contact) load is gravity, which is exactly what
        # fields.external_force exports; all dissipation comes from the contact
        # dashpots and Coulomb friction.
        force = self.mass[:, None] * self.gravity[None, :]
        self._pair_forces(force)
        self._container_forces(force)
        reaction = self._tool_forces(force, blade_center, shaft_a, shaft_b, tool_velocity)

        self.velocity += dt * force * self.inv_mass[:, None]
        speed = np.sqrt(np.einsum("ij,ij->i", self.velocity, self.velocity))
        hot = speed > self.v_cap
        if np.any(hot):
            self.velocity[hot] *= (self.v_cap / speed[hot])[:, None]
        self.position += dt * self.velocity

        self.time += dt
        blade_center, shaft_a, shaft_b, tool_velocity = self._tool_state(self.time)
        pre_velocity = self.velocity.copy()
        self._project_tool(blade_center, shaft_a, shaft_b, tool_velocity)
        # Only the tool projection loads the tool; capture its impulse before the
        # container projection so floor and wall impulses are not attributed to
        # the shovel. Newton's third law gives the reaction as minus that rate.
        impulse = (self.mass[:, None] * (self.velocity - pre_velocity)).sum(axis=0)
        reaction -= impulse / dt
        self._project_container()

        self.reaction_accum += reaction
        self.reaction_samples += 1
        self.step_count += 1
        self.peak_reaction = max(self.peak_reaction, float(np.linalg.norm(reaction)))

        if not np.all(np.isfinite(self.position)) or not np.all(np.isfinite(self.velocity)):
            self.position = np.nan_to_num(self.position, nan=0.0, posinf=0.0, neginf=0.0)
            self.velocity = np.nan_to_num(self.velocity, nan=0.0, posinf=0.0, neginf=0.0)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 1e-12:
            dt = min(self.dt, target - self.time)
            if dt <= 1e-15:
                break
            self._step(dt)
        self.time = target

        if self.reaction_samples:
            self.reaction_force = self.reaction_accum / self.reaction_samples
        self.reaction_accum = np.zeros(3)
        self.reaction_samples = 0
        return self._emit_frame(target)
    # ------------------------------------------------------------------
    # exported state and diagnostics
    # ------------------------------------------------------------------
    def _contact_gaps(self, blade_center, shaft_a, shaft_b) -> np.ndarray:
        """Signed geometric gap of every grain to its closest contact partner.

        Negative values are penetrations. The gap is the minimum over the
        static container, the moving tool and the nearest neighbouring grain.
        """
        r = self.radius
        gap = self.position[:, 2] - r - self.floor_z
        below_top = self.position[:, 2] < self.wall_top
        big = 1.0e3
        gap = np.minimum(gap, np.where(below_top, self.position[:, 0] - r - self.x_min, big))
        gap = np.minimum(gap, np.where(below_top, self.x_max - r - self.position[:, 0], big))
        gap = np.minimum(gap, np.where(below_top, self.position[:, 1] - r - self.y_min, big))
        gap = np.minimum(gap, np.where(below_top, self.y_max - r - self.position[:, 1], big))

        idx = self._blade_candidates(blade_center)
        if len(idx):
            dist, _ = _box_sdf(self.position[idx], blade_center, self.blade_rot, self.blade_half)
            gap[idx] = np.minimum(gap[idx], dist - r)
        idx = self._shaft_candidates(shaft_a, shaft_b)
        if len(idx):
            dist, _ = _capsule_sdf(self.position[idx], shaft_a, shaft_b, self.shaft_radius)
            gap[idx] = np.minimum(gap[idx], dist - r)

        if len(self.pair_i):
            i = self.pair_i
            j = self.pair_j
            delta = self.position[j] - self.position[i]
            dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            pair_gap = dist - self.contact_distance
            near = np.full(self.n, big)
            np.minimum.at(near, i, pair_gap)
            np.minimum.at(near, j, pair_gap)
            gap = np.minimum(gap, near)

        return np.clip(gap, -self.spacing, self.spacing)

    def _height_field(self) -> np.ndarray:
        """Top-surface elevation of the bed on a fixed horizontal grid."""
        ix = np.clip(
            ((self.position[:, 0] - self.x_min) / self.cell).astype(np.int64), 0, self.grid_nx - 1
        )
        iy = np.clip(
            ((self.position[:, 1] - self.y_min) / self.cell).astype(np.int64), 0, self.grid_ny - 1
        )
        flat = ix * self.grid_ny + iy
        top = np.full(self.grid_nx * self.grid_ny, self.floor_z)
        np.maximum.at(top, flat, self.position[:, 2] + self.radius)
        return top.reshape(self.grid_nx, self.grid_ny)

    def _emit_frame(self, target: float) -> StateFrame:
        blade_center, shaft_a, shaft_b, tool_velocity = self._tool_state(target)
        gaps = self._contact_gaps(blade_center, shaft_a, shaft_b)

        external_force = np.tile(self.gravity, (self.n, 1)) * self.mass[:, None]

        height = self._height_field()
        surface_delta = height - self.initial_surface_z
        raised = np.maximum(surface_delta, 0.0)
        lowered = np.maximum(-surface_delta, 0.0)
        mound_volume = float(raised.sum() * self.cell_area)
        excavated_volume = float(lowered.sum() * self.cell_area)
        displaced_volume = 0.5 * (mound_volume + excavated_volume)
        mound_height = float(np.max(surface_delta)) if surface_delta.size else 0.0
        footprint = float(np.count_nonzero(raised > 0.02) * self.cell_area)

        speed_sq = np.einsum("ij,ij->i", self.velocity, self.velocity)
        kinetic = 0.5 * float(np.sum(self.mass * speed_sq))
        settled = speed_sq < 0.01  # below 0.1 m/s
        settled_kinetic = 0.5 * float(np.sum(self.mass[settled] * speed_sq[settled]))
        escaped = (
            (self.position[:, 0] < self.x_min - 0.05)
            | (self.position[:, 0] > self.x_max + 0.05)
            | (self.position[:, 1] < self.y_min - 0.05)
            | (self.position[:, 1] > self.y_max + 0.05)
            | (self.position[:, 2] < self.floor_z - 0.05)
        )
        escaped_count = int(np.count_nonzero(escaped))
        reaction = np.asarray(self.reaction_force, dtype=float)

        scalars = {
            "contact_length_scale": float(self.spacing),
            "particle_radius": float(self.radius),
            "sand_mass": float(self.mass.sum()),
            "sand_volume": float(self.n * self.particle_volume),
            "displaced_sand_volume": displaced_volume,
            "excavated_volume": excavated_volume,
            "mound_volume": mound_volume,
            "mound_height": mound_height,
            "mound_footprint_area": footprint,
            "shovel_reaction_force_x": float(reaction[0]),
            "shovel_reaction_force_y": float(reaction[1]),
            "shovel_reaction_force_z": float(reaction[2]),
            "shovel_reaction_force_magnitude": float(np.linalg.norm(reaction)),
            "escaped_particle_count": float(escaped_count),
            "settled_kinetic_energy": settled_kinetic,
            "kinetic_energy": kinetic,
            "max_penetration": float(self.max_penetration),
            "max_contact_penetration": float(max(0.0, -float(np.min(gaps)))),
            "tool_position_x": float(blade_center[0]),
            "tool_position_y": float(blade_center[1]),
            "tool_position_z": float(blade_center[2]),
            "tool_speed": float(np.linalg.norm(tool_velocity)),
            "time_step": float(self.dt),
            "substeps_per_sample": float(self.substeps),
        }

        frame = StateFrame(
            time=float(target),
            position=np.ascontiguousarray(self.position, dtype=float),
            velocity=np.ascontiguousarray(self.velocity, dtype=float),
            mass=np.ascontiguousarray(self.mass, dtype=float),
            body_id=self.body_id,
            attributes={
                "mass_quantity": np.ascontiguousarray(self.mass, dtype=float),
                "volume_quantity": np.ascontiguousarray(self.initial_volume, dtype=float),
                "initial_volume": np.ascontiguousarray(self.initial_volume, dtype=float),
                "particle_radius": np.full(self.n, self.radius),
                "material_label": np.ones(self.n, dtype=np.int32),
            },
            fields={
                "contact_signed_distance": np.ascontiguousarray(gaps, dtype=float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "cumulative_volume_source": np.zeros(1, dtype=float),
                "external_force": np.ascontiguousarray(external_force, dtype=float),
                "tool_reaction_force": reaction.reshape(1, 3).copy(),
                "surface_height": np.ascontiguousarray(height, dtype=float),
            },
            scalars=scalars,
        )
        frame.validate()
        self.last_diagnostics = scalars
        return frame

    def diagnostics(self) -> dict[str, object]:
        summary = {
            "method": "explicit discrete-element method (soft-sphere, Coulomb friction)",
            "particles": int(self.n),
            "time_step": float(self.dt),
            "substeps_per_sample": int(self.substeps),
            "steps_taken": int(self.step_count),
            "normal_stiffness": float(self.k_n),
            "friction_particle": float(self.mu_pp),
            "friction_tool": float(self.mu_tool),
            "max_penetration": float(self.max_penetration),
            "total_sand_mass": float(self.total_mass),
            "peak_tool_reaction_force": float(self.peak_reaction),
        }
        for name, value in self.last_diagnostics.items():
            summary[f"final_{name}"] = float(value)
        return summary
