"""Hourglass granular solver.

Physically-based granular simulation using position-based dynamics (PBD):
gravity integration, spherical grain contacts solved as distance constraints,
kinematic (rotating) hourglass glass walls handled as a signed-distance
projection in the container frame, and Coulomb (static/kinetic) friction for
both grain-grain and grain-wall contacts so stable piles form.

The hourglass body is kinematic and is flipped twice (180 deg about y),
exactly as prescribed by the frozen events.  All state is integrated with a
fixed substep, giving deterministic, finite states at every sample time.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# ----------------------------------------------------------------------------
# Physical / numerical constants (frozen from the scene configuration)
# ----------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])
PARTICLE_RADIUS = 0.008
PARTICLE_DENSITY = 1600.0
SLAB_HALF_DEPTH = 0.055

DT_SUB = 1.0 / 600.0          # substep
SOLVE_ITERATIONS = 3
MU_S = 0.7                    # static friction coefficient (sand)
MU_K = 0.45                   # kinetic friction coefficient (sand)
MAX_SPEED = 8.0               # safety clamp

# Hourglass inner profile (radius, local z) from scene_geometry.yaml
_PROFILE = [
    (0.46, 0.02), (0.44, 0.14), (0.32, 0.36), (0.12, 0.60),
    (0.038, 0.72), (0.12, 0.84), (0.32, 1.08), (0.44, 1.30),
    (0.46, 1.42),
]
_PROF_Z = np.array([p[1] for p in _PROFILE], dtype=float)
_PROF_R = np.array([p[0] for p in _PROFILE], dtype=float)
_Z_BOTTOM = _PROF_Z[0]                    # 0.02
_Z_TOP = _PROF_Z[-1]                      # 1.42
_CONTAINER_TZ = 0.08                      # container translation z (world)
_PIVOT_WORLD = np.array([0.0, 0.0, 0.72 + _CONTAINER_TZ])  # world pivot
_PIVOT_LOCAL_Z = 0.72                     # profile symmetry plane

# Flip schedule: (start_time, end_time) each rotating +180 deg about y.
_FLIPS = ((6.0, 6.8), (12.0, 12.8))


def _smoothstep(u: float) -> float:
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return u * u * (3.0 - 2.0 * u)


def hourglass_angle(t: float) -> float:
    """Cumulative rotation angle (radians) of the hourglass body about y."""
    angle = 0.0
    for (t0, t1) in _FLIPS:
        if t >= t1:
            angle += np.pi
        elif t > t0:
            angle += np.pi * _smoothstep((t - t0) / (t1 - t0))
    return angle


class BenchmarkSolver:
    def __init__(self) -> None:
        self._initialized = False

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        self.position = np.array(frame.position, dtype=float)
        self.velocity = np.array(frame.velocity, dtype=float)
        self.mass = np.array(frame.mass, dtype=float)
        n = len(self.position)
        self.n = n
        self.radius = float(np.asarray(frame.attributes["particle_radius"])[0])
        self.diameter = 2.0 * self.radius
        self.time = 0.0
        self._first_substep = True
        self._initial_total_mass = float(self.mass.sum())
        rng = np.random.default_rng(seed)
        self._rng = rng
        self._initialized = True

    # ------------------------------------------------- container transforms
    def _to_local(self, w: np.ndarray, theta: float) -> np.ndarray:
        """World -> container local coordinates (x, y, z_local)."""
        v = w - _PIVOT_WORLD
        c, s = np.cos(theta), np.sin(theta)
        loc = np.empty_like(v)
        loc[:, 0] = c * v[:, 0] - s * v[:, 2]
        loc[:, 1] = v[:, 1]
        loc[:, 2] = s * v[:, 0] + c * v[:, 2]
        loc[:, 2] += _PIVOT_LOCAL_Z
        return loc

    def _to_world(self, loc: np.ndarray, theta: float) -> np.ndarray:
        v = loc.copy()
        v[:, 2] -= _PIVOT_LOCAL_Z
        c, s = np.cos(theta), np.sin(theta)
        w = np.empty_like(v)
        w[:, 0] = c * v[:, 0] + s * v[:, 2]
        w[:, 1] = v[:, 1]
        w[:, 2] = -s * v[:, 0] + c * v[:, 2]
        return w + _PIVOT_WORLD

    def _wall_clamp(self, loc: np.ndarray):
        """Clamp local positions into the glass interior.

        Returns clamped positions, per-particle correction magnitude and the
        axis index (0=x wall, 1=slab wall, 2=top/bottom cap) of the dominant
        correction for friction handling.
        """
        p = loc.copy()
        r = self.radius
        corr = np.zeros(len(p))
        axis = np.zeros(len(p), dtype=np.int64)

        # top / bottom caps
        z_lo = _Z_BOTTOM + r
        z_hi = _Z_TOP - r
        dz_lo = z_lo - p[:, 2]
        dz_hi = p[:, 2] - z_hi
        c_z = np.maximum(dz_lo, dz_hi)      # >0 -> out of range
        mask = c_z > 0.0
        if np.any(mask):
            p[:, 2] = np.clip(p[:, 2], z_lo, z_hi)
            corr[mask] += c_z[mask]
            axis[mask] = 2

        # side wall: |x| <= R(z) - r  (inner profile of the hourglass)
        radius_at = np.interp(p[:, 2], _PROF_Z, _PROF_R)
        lim = np.maximum(radius_at - r, 0.5 * r)
        ax = np.abs(p[:, 0])
        over = ax - lim
        mask = over > 0.0
        if np.any(mask):
            p[:, 0] = np.sign(p[:, 0]) * lim
            corr[mask] += over[mask]
            axis[mask] = 0

        # thin slab front/back walls
        lim_y = max(SLAB_HALF_DEPTH - r, 0.5 * r)
        ay = np.abs(p[:, 1]) - lim_y
        mask = ay > 0.0
        if np.any(mask):
            p[:, 1] = np.sign(p[:, 1]) * lim_y
            corr[mask] += ay[mask]
            axis[mask] = 1
        return p, corr, axis

    # ------------------------------------------------------- one substep
    def _substep(self, dt: float) -> None:
        theta = hourglass_angle(self.time + 0.5 * dt)
        x_prev = self.position.copy()

        # gravity + predict
        v = self.velocity + GRAVITY * dt
        x = self.position + v * dt

        r = self.radius
        d_min = self.diameter

        # neighbor pairs (fixed for this substep)
        tree = cKDTree(x)
        pairs = np.array(sorted(tree.query_pairs(d_min, output_type="ndarray")),
                         dtype=np.int64).reshape(-1, 2)
        has_pairs = len(pairs) > 0

        corr = np.zeros(self.n)
        for _ in range(SOLVE_ITERATIONS):
            # grain-grain contacts
            if has_pairs:
                i, j = pairs[:, 0], pairs[:, 1]
                d = x[j] - x[i]
                dist = np.linalg.norm(d, axis=1)
                bad = dist > 1e-12
                dist_s = np.where(bad, dist, 1.0)
                n = d / dist_s[:, None]
                c = np.where(bad, d_min - dist, 0.0)
                active = c > 0.0
                if np.any(active):
                    ci = np.where(active, c, 0.0)
                    relax = 0.75
                    np.add.at(x, i, -n * (0.5 * relax * ci)[:, None])
                    np.add.at(x, j, n * (0.5 * relax * ci)[:, None])
                    np.add.at(corr, i, 0.5 * ci)
                    np.add.at(corr, j, 0.5 * ci)

            # glass walls (in container frame)
            loc = self._to_local(x, theta)
            loc_c, w_corr, _ = self._wall_clamp(loc)
            if np.any(w_corr > 0.0):
                x = self._to_world(loc_c, theta)
                corr += w_corr

        # -------------------------------------------------- friction pass
        # grain-grain Coulomb friction on relative tangential displacement
        if has_pairs:
            i, j = pairs[:, 0], pairs[:, 1]
            d = x[j] - x[i]
            dist = np.linalg.norm(d, axis=1)
            bad = dist > 1e-12
            dist_s = np.where(bad, dist, 1.0)
            n = d / dist_s[:, None]
            u_rel = (x[i] - x_prev[i]) - (x[j] - x_prev[j])
            u_n = np.sum(u_rel * n, axis=1)
            u_t = u_rel - u_n[:, None] * n
            lt = np.linalg.norm(u_t, axis=1)
            c_f = corr[i] + corr[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                static = lt < MU_S * c_f
                kin_scale = np.maximum(1.0 - MU_K * c_f / np.where(bad & (lt > 1e-12), lt, 1.0), 0.0)
                scale = np.where(static, 0.0, kin_scale)
                scale = np.where(bad, scale, 0.0)
            delta = u_t * (scale - 1.0)[:, None]
            step = 0.5
            np.add.at(x, i, step * delta)
            np.add.at(x, j, -step * delta)

        # grain-wall Coulomb friction (evaluated in the container frame so the
        # rotating glass can drag the sand along with it)
        loc = self._to_local(x, theta)
        loc_prev = self._to_local(x_prev, theta)
        loc_c, w_corr, w_axis = self._wall_clamp(loc)
        contact = w_corr > 1e-12
        if np.any(contact):
            u = loc_c - loc_prev
            idx = np.arange(self.n)
            ncomp = u[idx, w_axis]
            u_t = u.copy()
            u_t[idx, w_axis] = 0.0
            lt = np.linalg.norm(u_t, axis=1)
            static = lt < MU_S * w_corr
            with np.errstate(divide="ignore", invalid="ignore"):
                kin_scale = np.maximum(
                    1.0 - MU_K * w_corr / np.where(lt > 1e-12, lt, 1.0), 0.0)
                scale = np.where(static, 0.0, kin_scale)
            new_u_t = u_t * scale[:, None]
            delta = new_u_t - u_t
            loc_c = loc_c + delta
            x = self._to_world(loc_c, theta)

        # velocity update
        v_new = (x - x_prev) / dt
        # kill the initial-condition relaxation jolt
        if self._first_substep:
            v_new[:] = 0.0
            self._first_substep = False
        speed = np.linalg.norm(v_new, axis=1)
        over = speed > MAX_SPEED
        if np.any(over):
            v_new[over] *= (MAX_SPEED / speed[over])[:, None]

        self.position = x
        self.velocity = v_new
        self.time += dt

    # ------------------------------------------------------- stepping API
    def advance_to(self, time: float):
        time = float(time)
        if not self._initialized:
            raise RuntimeError("initialize() must be called first")
        eps = 1e-9
        guard = 0
        while self.time < time - eps and guard < 100000:
            dt = min(DT_SUB, time - self.time)
            if dt < 1e-12:
                dt = 1e-12
            self._substep(dt)
            guard += 1
        return self._frame(time)

    # -------------------------------------------------------- frame export
    def _contact_signed_distance(self, theta: float) -> np.ndarray:
        """Per-grain signed distance to the nearest active constraint."""
        r = self.radius
        loc = self._to_local(self.position, theta)
        radius_at = np.interp(loc[:, 2], _PROF_Z, _PROF_R)
        gap_x = (radius_at - r) - np.abs(loc[:, 0])
        gap_y = (max(SLAB_HALF_DEPTH - r, 0.5 * r)) - np.abs(loc[:, 1])
        gap_z = np.minimum((loc[:, 2] - (_Z_BOTTOM + r)),
                           ((_Z_TOP - r) - loc[:, 2]))
        gap = np.minimum(np.minimum(gap_x, gap_y), gap_z)

        # include nearest grain-grain contact
        tree = cKDTree(self.position)
        dist, _ = tree.query(self.position, k=2)
        pair_gap = dist[:, 1] - self.diameter
        return np.minimum(gap, pair_gap)

    def _frame(self, time: float):
        theta = hourglass_angle(self.time)
        pos = np.array(self.position, dtype=float)
        vel = np.array(self.velocity, dtype=float)
        mass = np.array(self.mass, dtype=float)
        n = self.n
        frame_attrs = {
            "mass_quantity": mass.copy(),
            "particle_radius": np.full(n, self.radius),
        }
        fields = {
            "contact_signed_distance": self._contact_signed_distance(theta),
            "cumulative_mass_source": np.zeros(1),
            "external_force": mass[:, None] * GRAVITY[None, :],
        }
        scalars = {
            "contact_length_scale": self.diameter,
            "hourglass_rotation_angle": float(hourglass_angle(self.time)),
            "total_mass": float(mass.sum()),
        }
        from worldsolver.contracts import StateFrame
        return StateFrame(
            time=max(float(time), 0.0),
            position=pos,
            velocity=vel,
            mass=mass,
            attributes=frame_attrs,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        if not self._initialized:
            return {}
        speed = float(np.linalg.norm(self.velocity, axis=1).max())
        return {
            "particle_count": int(self.n),
            "particle_radius": self.radius,
            "substep_dt": DT_SUB,
            "initial_total_mass": self._initial_total_mass,
            "current_total_mass": float(self.mass.sum()),
            "hourglass_rotation_deg": float(np.degrees(hourglass_angle(self.time))),
            "max_speed": speed,
            "method": "position-based granular dynamics with Coulomb friction",
        }
