"""Granular pouring-sand solver.

Position-based dynamics (PBD) granular solver implemented with NumPy/SciPy.
Sand grains in the hopper discharge through the neck under gravity, fall as a
stream, impact the floor, and build a pile that relaxes to a stable angle of
repose via Coulomb-limited static/kinetic friction at grain contacts.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

try:  # pragma: no cover - optional import guard
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover
    StateFrame = object


class BenchmarkSolver:
    """PBD granular solver honoring the frozen scene contract."""

    def __init__(self) -> None:
        self.dt = 2.5e-3          # substep size [s]
        self.iters = 3            # constraint iterations per substep
        self.mu = 0.55            # Coulomb friction coefficient (grain & floor)
        self.max_speed = 25.0     # numerical safety clamp [m/s]
        self._last_time = 0.0
        self._last_frame = None

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        meta = scene.metadata()
        self.x = np.array(frame.position, dtype=float)
        self.v = np.array(frame.velocity, dtype=float)
        self.m = np.array(frame.mass, dtype=float)
        self.n = len(self.x)
        self.gravity = np.array(meta["gravity"], dtype=float)
        self.rp = float(meta["nominal_grain_radius"])
        self.spacing = float(meta["particle_spacing"])
        self.dmin = 2.0 * self.rp
        self.cutoff = self.dmin + 0.008
        self.t = 0.0
        self._base_attributes = dict(frame.attributes)
        self._base_scalars = dict(frame.scalars)
        self._last_time = 0.0
        self._last_frame = self._build_frame(0.0)
        self._substeps = 0

    # ------------------------------------------------------------- boundaries
    def _hopper_inner_radius(self, z: np.ndarray) -> np.ndarray:
        zs = np.array([1.42, 1.52, 1.72, 2.28])
        rs = np.array([0.075, 0.075, 0.34, 0.34])
        return np.interp(z, zs, rs)

    def _apply_boundaries(self, x: np.ndarray, xp: np.ndarray) -> None:
        """Project onto floor and hopper walls; apply floor friction."""
        rp = self.rp

        # ---- floor (top surface at z = 0, half extent 1.65 x 1.35)
        z_pre = x[:, 2].copy()
        low = x[:, 2] < rp
        if np.any(low):
            x[low, 2] = rp
            # Coulomb-limited floor friction on tangential displacement
            pen = rp - z_pre[low]
            disp = x[low] - xp[low]
            disp[:, 2] = 0.0
            mag = np.linalg.norm(disp, axis=1)
            scale = np.ones_like(mag)
            move = mag > 1e-12
            scale[move] = np.minimum(1.0, self.mu * pen[move] / mag[move])
            x[low, :2] -= disp[:, :2] * scale[:, None]

        # lateral containment at floor extents
        x[:, 0] = np.clip(x[:, 0], -1.63, 1.63)
        x[:, 1] = np.clip(x[:, 1], -1.33, 1.33)

        # ---- hopper: keep grains inside the (frictionless) inner profile
        z = x[:, 2]
        in_hopper = (z >= 1.42) & (z <= 2.28)
        if np.any(in_hopper):
            idx = np.nonzero(in_hopper)[0]
            r = np.hypot(x[idx, 0], x[idx, 1])
            r_max = self._hopper_inner_radius(x[idx, 2]) - rp
            over = r > r_max
            if np.any(over):
                sub = idx[over]
                rr = np.hypot(x[sub, 0], x[sub, 1])
                rr = np.maximum(rr, 1e-9)
                s = r_max[over] / rr
                x[sub, 0] *= s
                x[sub, 1] *= s

    # ------------------------------------------------------------ integration
    def _substep(self, h: float) -> None:
        x, v, g = self.x, self.v, self.gravity
        xp = x.copy()

        # explicit prediction
        v += g * h
        x += v * h

        # neighbor pair list for this substep
        tree = cKDTree(x)
        pairs = tree.query_pairs(self.cutoff, output_type="ndarray")
        if len(pairs) > 0:
            order = np.lexsort((pairs[:, 1], pairs[:, 0]))
            pairs = pairs[order]
            i = pairs[:, 0]
            j = pairs[:, 1]
            dmin = self.dmin
            for _ in range(self.iters):
                d = x[j] - x[i]
                dist = np.linalg.norm(d, axis=1)
                act = dist < dmin
                if not np.any(act):
                    break
                ia = i[act]
                ja = j[act]
                da = d[act]
                ra = dist[act]
                ra = np.maximum(ra, 1e-9)
                n = da / ra[:, None]
                pen = dmin - ra
                # normal separation (equal masses)
                corr = 0.5 * pen[:, None] * n
                np.add.at(x, ia, corr)
                np.add.at(x, ja, -corr)
                # Coulomb-limited tangential (friction) correction
                rel = (x[ia] - xp[ia]) - (x[ja] - xp[ja])
                rel_n = np.sum(rel * n, axis=1)[:, None] * n
                rel_t = rel - rel_n
                mag = np.linalg.norm(rel_t, axis=1)
                scale = np.ones_like(mag)
                mv = mag > 1e-12
                scale[mv] = np.minimum(1.0, self.mu * pen[mv] / mag[mv])
                ct = 0.5 * rel_t * scale[:, None]
                np.add.at(x, ia, -ct)
                np.add.at(x, ja, ct)

        self._apply_boundaries(x, xp)

        # velocity from total positional change
        v_new = (x - xp) / h
        spd = np.linalg.norm(v_new, axis=1)
        fast = spd > self.max_speed
        if np.any(fast):
            v_new[fast] *= (self.max_speed / spd[fast])[:, None]
        self.v = v_new
        self._substeps += 1

    # ------------------------------------------------------------------ frame
    def _contact_gaps(self) -> np.ndarray:
        tree = cKDTree(self.x)
        pairs = tree.query_pairs(self.cutoff, output_type="ndarray")
        if len(pairs) == 0:
            return np.zeros(0, dtype=float)
        d = self.x[pairs[:, 1]] - self.x[pairs[:, 0]]
        dist = np.linalg.norm(d, axis=1)
        return dist - self.dmin

    def _build_frame(self, time: float) -> "StateFrame":
        n = self.n
        ext = self.m[:, None] * self.gravity[None, :]
        attributes = {}
        for k, val in self._base_attributes.items():
            arr = np.asarray(val, dtype=float).copy()
            if arr.shape[0] == n:
                attributes[k] = arr
        attributes["mass_quantity"] = self.m.copy()
        scalars = {
            "contact_length_scale": float(self.spacing),
            "initial_total_mass": float(np.sum(self.m)),
        }
        for k, val in self._base_scalars.items():
            scalars.setdefault(k, float(val))
        return StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            attributes=attributes,
            fields={
                "contact_signed_distance": self._contact_gaps(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": ext,
            },
            scalars=scalars,
        )

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float):
        target = float(time)
        if target < self.t - 1e-9:
            raise ValueError("advance_to requires non-decreasing times")
        if target <= self.t + 1e-12:
            return self._last_frame if self._last_time == target else self._build_frame(target)
        while self.t < target - 1e-12:
            h = min(self.dt, target - self.t)
            self._substep(h)
            self.t = min(self.t + h, target)
        self._last_time = target
        self._last_frame = self._build_frame(target)
        return self._last_frame

    def diagnostics(self) -> dict:
        return {
            "num_particles": int(self.n),
            "time": float(self.t),
            "substeps": int(getattr(self, "_substeps", 0)),
            "dt": float(self.dt),
            "total_mass": float(np.sum(self.m)),
            "method": "position-based-dynamics granular solver",
        }
