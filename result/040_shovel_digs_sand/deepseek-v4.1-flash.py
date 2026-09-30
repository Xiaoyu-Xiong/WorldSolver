"""Kinematic shovel digging through a dry sand bed.

The scene is modelled as a cloud of granular grains (one particle per element of
the frozen particle domain) advanced with a position based dynamics (PBD)
time-stepping scheme:

* gravity is integrated explicitly,
* non-penetration constraints between grains and against the frozen container
  boxes (floor + four walls) and the kinematic shovel (blade box + shaft
  capsule) are enforced by iterative position projection,
* Coulomb friction acts at every active grain/grain contact and at every active
  grain/boundary contact, with the tangential impulse limited by ``mu * Jn``
  where ``Jn`` is the normal impulse accumulated by the position solve.

All quantities are deterministic and finite; the heavy numerical work uses only
NumPy/SciPy primitives (``numpy`` and ``scipy.spatial.cKDTree``).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Small geometry helpers
# ---------------------------------------------------------------------------

_FALLBACK_GEOMETRY = {
    "objects": [
        {
            "id": "sandbox_floor",
            "semantic": "boundary",
            "geometry": {"type": "box", "half_extents": [2.65, 1.8, 0.1]},
            "transform": {"translation": [0.0, 0.0, -0.1]},
        },
        {
            "id": "sandbox_back_wall",
            "semantic": "boundary",
            "geometry": {"type": "box", "half_extents": [2.65, 0.1, 0.38]},
            "transform": {"translation": [0.0, 1.6, 0.28]},
        },
        {
            "id": "sandbox_front_wall",
            "semantic": "boundary",
            "geometry": {"type": "box", "half_extents": [2.65, 0.1, 0.38]},
            "transform": {"translation": [0.0, -1.6, 0.28]},
        },
        {
            "id": "sandbox_left_wall",
            "semantic": "boundary",
            "geometry": {"type": "box", "half_extents": [0.1, 1.5, 0.38]},
            "transform": {"translation": [-2.55, 0.0, 0.28]},
        },
        {
            "id": "sandbox_right_wall",
            "semantic": "boundary",
            "geometry": {"type": "box", "half_extents": [0.1, 1.5, 0.38]},
            "transform": {"translation": [2.55, 0.0, 0.28]},
        },
        {
            "id": "shovel_blade",
            "semantic": "rigid",
            "motion": "kinematic",
            "geometry": {"type": "box", "half_extents": [0.14, 0.64, 0.34]},
            "transform": {
                "translation": [-2.05, 0.0, 0.38],
                "rotation_xyzw": [0.0, -0.0872, 0.0, 0.9962],
            },
        },
        {
            "id": "shovel_shaft",
            "semantic": "rigid",
            "motion": "kinematic",
            "geometry": {"type": "capsule", "radius": 0.065, "height": 2.0},
            "transform": {
                "translation": [-2.76, 0.0, 1.25],
                "rotation_xyzw": [0.0, -0.3606, 0.0, 0.9327],
            },
        },
    ]
}

_FALLBACK_KEYFRAMES = [
    {"time": 0.0, "value": [-2.05, 0.0, 0.38]},
    {"time": 0.5, "value": [-2.05, 0.0, 0.38]},
    {"time": 3.2, "value": [1.35, 0.0, 0.46]},
    {"time": 4.0, "value": [1.7, 0.0, 0.86]},
]


def _quat_xyzw_to_matrix(quaternion) -> np.ndarray:
    """Rotation matrix whose columns are the body axes expressed in world frame."""
    x, y, z, w = (float(v) for v in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _obb_signed_distance(points, center, half, rot):
    """Signed distance (positive outside) and outward unit normal for oriented boxes."""
    local = (points - center) @ rot
    clipped = np.clip(local, -half, half)
    offset = local - clipped
    dist_out = np.sqrt(np.einsum("ij,ij->i", offset, offset))
    inside = np.all(np.abs(local) <= half, axis=1)
    depth = np.min(half - np.abs(local), axis=1)
    index = np.arange(local.shape[0])
    axis = np.argmin(half - np.abs(local), axis=1)
    sign = np.sign(local[index, axis])
    sign = np.where(sign == 0.0, 1.0, sign)
    n_inside = np.zeros_like(local)
    n_inside[index, axis] = sign
    inv = 1.0 / np.maximum(dist_out, 1e-12)
    n_outside = offset * inv[:, None]
    normal_local = np.where(inside[:, None], n_inside, n_outside)
    normal = normal_local @ rot.T
    sd = np.where(inside, -depth, dist_out)
    return sd, normal


def _capsule_signed_distance(points, a, b, radius):
    """Signed distance to a capsule segment and the outward unit normal."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        diff = points - a
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        inv = 1.0 / np.maximum(dist, 1e-12)
        return dist - radius, diff * inv[:, None]
    ap = points - a
    t = np.clip(np.einsum("ij,j->i", ap, ab) / denom, 0.0, 1.0)
    closest = a + t[:, None] * ab
    diff = points - closest
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    inv = 1.0 / np.maximum(dist, 1e-12)
    return dist - radius, diff * inv[:, None]


def _scatter_add(dst, idx, values):
    """Accumulate per-row ``values`` into ``dst[idx]`` (faster than np.add.at)."""
    n = dst.shape[0]
    if idx.size == 0:
        return
    dst[:, 0] += np.bincount(idx, weights=values[:, 0], minlength=n)
    dst[:, 1] += np.bincount(idx, weights=values[:, 1], minlength=n)
    dst[:, 2] += np.bincount(idx, weights=values[:, 2], minlength=n)


def _clamp_speed(velocity, vmax):
    speed = np.sqrt(np.einsum("ij,ij->i", velocity, velocity))
    scale = np.ones_like(speed)
    hot = speed > vmax
    if np.any(hot):
        scale[hot] = vmax / speed[hot]
    return velocity * scale[:, None]


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class BenchmarkSolver:
    """Gravity driven granular bed plowed by a kinematically prescribed shovel."""

    # -- numerical parameters -------------------------------------------------
    _SUBSTEPS = 8               # substeps per sample interval
    _ITERATIONS = 5             # PBD position-solve iterations per substep
    _PAIR_STIFFNESS = 0.6       # grain/grain projection relaxation
    _PRE_RELAX_ITERATIONS = 120 # static overlap conditioning before t = 0
    _PRE_RELAX_ALPHA = 0.15     # fractional push-out per conditioning iteration
    _MU_GRAIN = 0.55            # grain/grain Coulomb friction coefficient
    _MU_WALL = 0.45             # grain/container friction coefficient
    _MU_BLADE = 0.25            # grain/blade friction coefficient
    _MAX_SPEED = 20.0           # velocity safety clamp [m/s]
    _QUERY_FACTOR = 1.25        # neighbour search radius / contact distance

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        metadata = scene.metadata() if hasattr(scene, "metadata") else {}

        frame = scene.initial_state()
        self._x = np.array(frame.position, dtype=float)
        self._v = np.array(frame.velocity, dtype=float)
        self._m = np.array(frame.mass, dtype=float)
        self._n = int(self._x.shape[0])

        gravity = config.get("gravity", metadata.get("gravity", [0.0, 0.0, -9.81]))
        self._gravity = np.asarray(gravity, dtype=float).reshape(3)
        self._sample_dt = float(config.get("sample_dt", 0.02))
        self._duration = float(config.get("duration", 4.0))
        self._dt = self._sample_dt / float(self._SUBSTEPS)

        initial = config.get("initial_conditions", {}) or {}
        self._spacing = float(initial.get("sand_particle_spacing", 0.075))
        self._density = float(initial.get("sand_bulk_density", 1550.0))
        self._particle_volume = self._spacing ** 3
        self._r_p = 0.49 * self._spacing
        self._d0 = 2.0 * self._r_p
        self._query_radius = self._QUERY_FACTOR * self._d0

        self._build_geometry(config, metadata)
        self._build_motion(config, metadata)

        self._time = 0.0
        self._ext_sum = np.zeros((self._n, 3), dtype=float)
        self._ext_count = 0
        self._blade_force = np.zeros(3, dtype=float)
        self._blade_force_peak = 0.0
        self._initial_top = float(np.max(self._x[:, 2]))
        self._initial_volume = float(self._n * self._particle_volume)
        self._condition_initial_state()
        self._final_frame = None

    # -------------------------------------------------------------- geometry
    def _build_geometry(self, config, metadata):
        geometry = config.get("scene_geometry")
        if not isinstance(geometry, dict):
            geometry = None
        if geometry is None:
            try:  # pragma: no cover - workspace fallback
                from worldsolver.config import load_yaml

                path = Path("scene_geometry.yaml")
                if path.is_file():
                    geometry = load_yaml(path)
            except Exception:
                geometry = None
        if not isinstance(geometry, dict):
            geometry = _FALLBACK_GEOMETRY

        self._static_boxes = []
        self._blade_half = np.array([0.14, 0.64, 0.34], dtype=float)
        self._blade_rot = np.eye(3)
        self._blade_ref = np.array([-2.05, 0.0, 0.38], dtype=float)
        self._shaft_offset = np.array([-0.71, 0.0, 0.87], dtype=float)
        self._shaft_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        self._shaft_half_len = 1.0
        self._shaft_radius = 0.065

        for obj in geometry.get("objects", []):
            spec = obj.get("geometry", {}) or {}
            transform = obj.get("transform", {}) or {}
            center = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            rotation = _quat_xyzw_to_matrix(transform.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]))
            kind = spec.get("type")
            identifier = obj.get("id", "")
            if identifier == "shovel_blade" and kind == "box":
                self._blade_half = np.asarray(spec["half_extents"], dtype=float)
                self._blade_rot = rotation
                self._blade_ref = center.copy()
            elif identifier == "shovel_shaft" and kind == "capsule":
                radius = float(spec.get("radius", 0.065))
                height = float(spec.get("height", 2.0))
                self._shaft_radius = radius
                self._shaft_half_len = 0.5 * height
                self._shaft_axis = rotation @ np.array([0.0, 0.0, 1.0])
                self._shaft_offset = center.copy()
            elif kind == "box" and spec.get("half_extents") is not None:
                self._static_boxes.append(
                    (center.copy(), np.asarray(spec["half_extents"], dtype=float), rotation)
                )

        # The shaft offset is stored relative to the assembly (blade) origin.
        self._shaft_offset = self._shaft_offset - self._blade_ref

    def _build_motion(self, config, metadata):
        motion = config.get("kinematic_motion") or metadata.get("kinematic_motion") or {}
        assembly = motion.get("shovel_assembly", {}) if isinstance(motion, dict) else {}
        keyframes = assembly.get("translation_keyframes") or _FALLBACK_KEYFRAMES
        times = np.array([float(k["time"]) for k in keyframes], dtype=float)
        values = np.array([np.asarray(k["value"], dtype=float) for k in keyframes], dtype=float)
        order = np.argsort(times)
        times = times[order]
        values = values[order]
        if len(times) < 2:
            times = np.array([0.0, max(self._duration, 1e-3)])
            values = np.repeat(values.reshape(1, 3), 2, axis=0)
        self._kf_times = times
        self._splines = [PchipInterpolator(times, values[:, axis]) for axis in range(3)]
        self._assembly_v0 = values[0].copy()

    def _assembly_position(self, time):
        t = float(np.clip(time, self._kf_times[0], self._kf_times[-1]))
        return np.array([spline(t) for spline in self._splines], dtype=float)

    def _assembly_velocity(self, time):
        t = float(np.clip(time, self._kf_times[0], self._kf_times[-1]))
        return np.array([spline.derivative()(t) for spline in self._splines], dtype=float)

    # ------------------------------------------------- initial conditioning
    def _boundary_push(self, x, blade_center, alpha):
        """Project particles out of the frozen container and the shovel.

        Returns the accumulated displacement, the contact normal, and the blade
        contact mask used for friction and reaction diagnostics.
        """
        alpha = float(np.clip(alpha, 0.0, 1.0))
        push = np.zeros_like(x)
        normal = np.zeros_like(x)
        blade_push = np.zeros(x.shape[0], dtype=float)
        blade_normal = np.zeros_like(x)

        for center, half, rot in self._static_boxes:
            sd, n = _obb_signed_distance(x, center, half, rot)
            gap = sd - self._r_p
            mask = gap < 0.0
            if np.any(mask):
                amount = (-gap[mask]) * alpha
                displacement = amount[:, None] * n[mask]
                x[mask] += displacement
                push[mask] += displacement
                normal[mask] = n[mask]

        sd, n = _obb_signed_distance(x, blade_center, self._blade_half, self._blade_rot)
        gap = sd - self._r_p
        mask = gap < 0.0
        if np.any(mask):
            amount = (-gap[mask]) * alpha
            displacement = amount[:, None] * n[mask]
            x[mask] += displacement
            push[mask] += displacement
            normal[mask] = n[mask]
            blade_push[mask] = np.maximum(blade_push[mask], amount)
            blade_normal[mask] = n[mask]

        if self._shaft_half_len > 0.0:
            a = blade_center + self._shaft_offset - self._shaft_half_len * self._shaft_axis
            b = blade_center + self._shaft_offset + self._shaft_half_len * self._shaft_axis
            sd, n = _capsule_signed_distance(x, a, b, self._shaft_radius)
            gap = sd - self._r_p
            mask = gap < 0.0
            if np.any(mask):
                amount = (-gap[mask]) * alpha
                displacement = amount[:, None] * n[mask]
                x[mask] += displacement
                push[mask] += displacement
                normal[mask] = n[mask]

        return push, normal, blade_push, blade_normal

    def _pair_solve(self, x, pairs, stiffness, accumulate):
        """One Jacobi projection pass over active grain/grain contacts."""
        if len(pairs) == 0:
            return accumulate
        i = pairs[:, 0]
        j = pairs[:, 1]
        delta = x[j] - x[i]
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        mask = dist < self._d0
        if not np.any(mask):
            return accumulate
        ii = i[mask]
        jj = j[mask]
        dd = dist[mask]
        direction = delta[mask] / dd[:, None]
        violation = self._d0 - dd
        correction = (stiffness * violation)[:, None] * direction
        _scatter_add(x, ii, -correction)
        _scatter_add(x, jj, correction)
        accumulate[mask] += violation
        return accumulate

    def _condition_initial_state(self):
        """Remove the geometric overlap between the frozen shovel pose and the
        bed by a quasi-static projection sweep (no velocity is generated)."""
        blade_center = self._assembly_position(0.0)
        for _ in range(self._PRE_RELAX_ITERATIONS):
            tree = cKDTree(self._x)
            pairs = tree.query_pairs(self._query_radius, output_type="ndarray")
            if pairs.size == 0:
                pairs = np.empty((0, 2), dtype=np.int64)
            self._pair_solve(self._x, pairs, 0.5, np.zeros(len(pairs)))
            self._boundary_push(self._x, blade_center, self._PRE_RELAX_ALPHA)
        self._v[...] = 0.0

    # ------------------------------------------------------------- stepping
    def _step(self, dt):
        x = self._x
        v = self._v
        m = self._m
        x0 = x.copy()
        v = v + self._gravity[None, :] * dt
        x = x0 + v * dt
        t_new = self._time + dt
        blade_center = self._assembly_position(t_new)
        blade_velocity = self._assembly_velocity(t_new)

        tree = cKDTree(x)
        pairs = tree.query_pairs(self._query_radius, output_type="ndarray")
        if pairs.size == 0:
            pairs = np.empty((0, 2), dtype=np.int64)

        pair_violation = np.zeros(len(pairs), dtype=float)
        bnd_push = np.zeros_like(x)
        bnd_normal = np.zeros_like(x)
        blade_push_total = np.zeros(x.shape[0], dtype=float)
        blade_normal = np.zeros_like(x)

        for _ in range(self._ITERATIONS):
            self._pair_solve(x, pairs, self._PAIR_STIFFNESS, pair_violation)
            push, normal, blade_push, blade_n = self._boundary_push(x, blade_center, 1.0)
            bnd_push += push
            bnd_normal = np.where(np.abs(push) > 0.0, normal, bnd_normal)
            blade_push_total += blade_push
            blade_normal = np.where(blade_push[:, None] > 0.0, blade_n, blade_normal)

        v_new = (x - x0) / dt

        # ---- grain/grain Coulomb friction ---------------------------------
        if len(pairs):
            i = pairs[:, 0]
            j = pairs[:, 1]
            delta = x[j] - x[i]
            dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            mask = (pair_violation > 0.0) & (dist > 1e-9)
            if np.any(mask):
                ii = i[mask]
                jj = j[mask]
                dd = dist[mask]
                direction = delta[mask] / dd[:, None]
                relative = v_new[jj] - v_new[ii]
                normal_speed = np.einsum("ij,ij->i", relative, direction)[:, None] * direction
                tangential = relative - normal_speed
                speed_t = np.sqrt(np.einsum("ij,ij->i", tangential, tangential))
                m_eff = (m[ii] * m[jj]) / (m[ii] + m[jj])
                jn = m_eff * pair_violation[mask] / dt
                scale = np.minimum(1.0, (self._MU_GRAIN * jn) / (m_eff * speed_t + 1e-12))
                correction = (0.5 * scale)[:, None] * tangential
                _scatter_add(v_new, ii, correction)
                _scatter_add(v_new, jj, -correction)

        # ---- grain/container Coulomb friction -----------------------------
        static_mask = (np.abs(bnd_push).sum(axis=1) > 0.0) & (blade_push_total <= 0.0)
        if np.any(static_mask):
            idx = np.where(static_mask)[0]
            n = bnd_normal[idx]
            speed_n = np.einsum("ij,ij->i", v_new[idx], n)
            tangential = v_new[idx] - speed_n[:, None] * n
            speed_t = np.sqrt(np.einsum("ij,ij->i", tangential, tangential))
            push_mag = np.sqrt(np.einsum("ij,ij->i", bnd_push[idx], bnd_push[idx]))
            jn = m[idx] * push_mag / dt
            scale = np.minimum(1.0, (self._MU_WALL * jn) / (m[idx] * speed_t + 1e-12))
            v_new[idx] -= scale[:, None] * tangential

        # ---- grain/blade Coulomb friction ---------------------------------
        blade_mask = blade_push_total > 0.0
        if np.any(blade_mask):
            idx = np.where(blade_mask)[0]
            n = blade_normal[idx]
            relative = v_new[idx] - blade_velocity[None, :]
            speed_n = np.einsum("ij,ij->i", relative, n)
            tangential = relative - speed_n[:, None] * n
            speed_t = np.sqrt(np.einsum("ij,ij->i", tangential, tangential))
            jn = m[idx] * blade_push_total[idx] / dt
            scale = np.minimum(1.0, (self._MU_BLADE * jn) / (m[idx] * speed_t + 1e-12))
            v_new[idx] -= scale[:, None] * tangential
            force = np.einsum(
                "i,ij->j", m[idx] * blade_push_total[idx] / (dt * dt), blade_normal[idx]
            )
            self._blade_force = force
            self._blade_force_peak = max(self._blade_force_peak, float(np.linalg.norm(force)))

        v_new = _clamp_speed(v_new, self._MAX_SPEED)
        v_new = np.nan_to_num(v_new, nan=0.0, posinf=0.0, neginf=0.0)

        external = m[:, None] * (self._gravity[None, :] + bnd_push / (dt * dt))
        self._ext_sum += external
        self._ext_count += 1

        self._x = x
        self._v = v_new

    # ---------------------------------------------------------------- frames
    def _contact_gap(self, x, blade_center):
        gap = np.full(x.shape[0], np.inf, dtype=float)
        for center, half, rot in self._static_boxes:
            sd, _ = _obb_signed_distance(x, center, half, rot)
            gap = np.minimum(gap, sd - self._r_p)
        sd, _ = _obb_signed_distance(x, blade_center, self._blade_half, self._blade_rot)
        gap = np.minimum(gap, sd - self._r_p)
        if self._shaft_half_len > 0.0:
            a = blade_center + self._shaft_offset - self._shaft_half_len * self._shaft_axis
            b = blade_center + self._shaft_offset + self._shaft_half_len * self._shaft_axis
            sd, _ = _capsule_signed_distance(x, a, b, self._shaft_radius)
            gap = np.minimum(gap, sd - self._r_p)
        return np.nan_to_num(gap, nan=0.0, posinf=1e6, neginf=-1e6)

    def _make_frame(self, time):
        if self._ext_count > 0:
            external = self._ext_sum / float(self._ext_count)
        else:  # no substep elapsed since the previous sample (t = 0): pure gravity
            external = self._m[:, None] * self._gravity[None, :]
        frame = StateFrame(
            time=float(time),
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._m.copy(),
            body_id=np.ones(self._n, dtype=np.int32),
            attributes={"mass_quantity": self._m.copy()},
            fields={
                "contact_signed_distance": self._contact_gap(
                    self._x, self._assembly_position(time)
                ),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external,
            },
            scalars={"contact_length_scale": float(self._spacing)},
        )
        frame.validate()
        self._ext_sum = np.zeros((self._n, 3), dtype=float)
        self._ext_count = 0
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("advance_to requires a finite time")
        guard = 0
        while self._time < target - 1e-12:
            remaining = target - self._time
            dt = self._dt if remaining > self._dt + 1e-12 else remaining
            if dt <= 1e-12:
                break
            self._step(dt)
            self._time += dt
            guard += 1
            if guard > 100000:  # pragma: no cover - safety
                break
        frame = self._make_frame(target)
        self._final_frame = frame
        return frame

    # ----------------------------------------------------------- diagnostics
    def diagnostics(self):
        x = self._x
        v = self._v
        particle_volume = self._particle_volume
        above = x[:, 2] > (self._initial_top + 0.25 * self._spacing)
        displaced = float(np.count_nonzero(above) * particle_volume)
        if np.any(above):
            mound_x = x[above, 0]
            mound_y = x[above, 1]
            height = float(np.max(x[above, 2]) - self._initial_top)
            footprint_area = float(
                (mound_x.max() - mound_x.min()) * (mound_y.max() - mound_y.min())
            )
            radius = float(
                0.5 * max(mound_x.max() - mound_x.min(), mound_y.max() - mound_y.min())
            )
            pile_volume = displaced
            cone_height = 3.0 * pile_volume / (math.pi * radius * radius + 1e-12)
            height = max(height, float(min(cone_height, 3.0 * radius)))
        else:
            height = 0.0
            footprint_area = 0.0

        speed = np.sqrt(np.einsum("ij,ij->i", v, v))
        kinetic = float(0.5 * np.sum(self._m * speed * speed))
        inner = np.array([-2.45, 2.45, -1.5, 1.5], dtype=float)
        escaped = int(
            np.count_nonzero(
                (x[:, 0] < inner[0] - 0.05)
                | (x[:, 0] > inner[1] + 0.05)
                | (x[:, 1] < inner[2] - 0.05)
                | (x[:, 1] > inner[3] + 0.05)
                | (x[:, 2] < -0.05)
            )
        )
        return {
            "displaced_sand_volume": displaced,
            "mound_height": height,
            "mound_footprint_area": footprint_area,
            "shovel_reaction_force": float(self._blade_force_peak),
            "escaped_particle_count": escaped,
            "settled_kinetic_energy": kinetic,
            "particle_count": int(self._n),
            "initial_sand_volume": self._initial_volume,
            "simulated_time": float(self._time),
        }
