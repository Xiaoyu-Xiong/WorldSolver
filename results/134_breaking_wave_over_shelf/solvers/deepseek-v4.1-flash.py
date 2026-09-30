"""Particle-based free-surface solver for a breaking wave over a submerged shelf.

Physical model
--------------
The liquid is discretised as a set of material particles carrying mass and
volume.  The free surface is therefore represented implicitly and is free to
overturn, entrain voids and splash.  Incompressibility is enforced with a
position-based fluid (PBF) projection (Macklin & Mueller): a density constraint
C_i = rho_i/rho_0 - 1 is solved with a Jacobi (order independent, fully
deterministic) iteration driven by the spiky kernel gradient, supplemented by
the artificial-pressure term that removes particle clustering.  Gravity is
integrated explicitly; the velocity field is recovered from the projected
positions, then smoothed with an XSPH correction.  Rigid geometry (tank floor,
side walls, end wall, submerged shelf and the kinematic dam gate) is handled by
an analytic axis-aligned box projection that guarantees the fluid never leaves
the tank nor enters an obstacle.

Scenario
--------
A coherent water column is held behind the dam gate over deep water.  The gate
is lifted kinematically at t = 0, releasing a surge that shoals over the
submerged shelf, steepens, curls forward and breaks, sending a bore into the
downstream run-out where it reflects from the end wall.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# --------------------------------------------------------------------------
# frozen physical / numerical parameters
# --------------------------------------------------------------------------
_G = np.array([0.0, 0.0, -9.81])
_RHO = 1000.0                     # water density [kg/m^3]
_DX = 0.10                        # rest particle spacing [m]
_MP = _RHO * _DX ** 3             # particle mass [kg]
_HS = 2.0 * _DX                   # SPH smoothing radius [m]
_KNN = 44                         # neighbour list capacity
_ITERS = 4                        # PBF iterations per substep
_DT_SUB = 1.0 / 240.0             # substep [s]
_VMAX = 12.0                      # velocity clamp [m/s]
_DP_MAX = 0.20 * _DX              # per-iteration position correction clamp
_K_AP = 0.05                      # artificial pressure strength
_N_AP = 4.0                       # artificial pressure exponent
_XSPH = 0.03                      # XSPH viscosity coefficient
_GATE_SPEED = 3.0                 # gate lift speed [m/s]
_GATE_LIFT = 1.25                 # gate lift distance [m]
_BOUND_OFFSET = 0.045             # wall stand-off used by the collision projection

# diagnostic (collocated) grid, uniform spacing, shape frozen by the config
_GRID_SHAPE = (24, 16, 28)
_GRID_ORIGIN = np.array([-3.0, -2.0, 0.0])
_GRID_H = 0.25

_W_POLY = 315.0 / (64.0 * np.pi * _HS ** 9)          # poly6 normalisation
_GRAD_SPIKY = 45.0 / (np.pi * _HS ** 6)              # spiky gradient normalisation
_W_DQ = _W_POLY * (_HS ** 2 - (0.2 * _HS) ** 2) ** 3  # artificial pressure reference

# fallback geometry (identical to the frozen scene_geometry.yaml boxes)
_FALLBACK_BOXES = [
    {"id": "ground", "lo": [-3.2, -1.4, -0.10], "hi": [3.2, 1.4, 0.0], "motion": "static"},
    {"id": "tank_side_left", "lo": [-3.2, -1.04, 0.0], "hi": [3.2, -0.92, 1.10], "motion": "static"},
    {"id": "tank_side_right", "lo": [-3.2, 0.92, 0.0], "hi": [3.2, 1.04, 1.10], "motion": "static"},
    {"id": "tank_end_wall", "lo": [3.06, -0.98, 0.0], "hi": [3.18, 0.98, 1.10], "motion": "static"},
    {"id": "submerged_shelf", "lo": [0.25, -0.84, 0.0], "hi": [2.65, 0.84, 0.60], "motion": "static"},
    {"id": "dam_gate", "lo": [-0.49, -0.72, 0.0], "hi": [-0.41, 0.72, 1.10], "motion": "kinematic"},
]


def _boxes_from_geometry(geometry: dict) -> list[dict]:
    """Extract axis-aligned collision boxes from the frozen scene geometry."""
    boxes: list[dict] = []
    for obj in (geometry or {}).get("objects", []) or []:
        spec = obj.get("geometry") or {}
        if spec.get("type") != "box" or not obj.get("collision_enabled", True):
            continue
        half = np.asarray(spec.get("half_extents", [0.0, 0.0, 0.0]), float)
        trans = np.asarray(
            (obj.get("transform") or {}).get("translation", [0.0, 0.0, 0.0]), float
        )
        boxes.append(
            {
                "id": str(obj.get("id", "box")),
                "lo": trans - half,
                "hi": trans + half,
                "motion": str(obj.get("motion", "static")),
            }
        )
    return boxes or [
        {"id": b["id"], "lo": np.asarray(b["lo"], float), "hi": np.asarray(b["hi"], float),
         "motion": b["motion"]}
        for b in _FALLBACK_BOXES
    ]


class BenchmarkSolver:
    """Deterministic position-based fluid over the frozen breaking-wave scene."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = getattr(scene, "config", None) or {}
        self.dt_sub = _DT_SUB
        self.t = 0.0

        # ---- frozen rigid geometry -------------------------------------
        geometry = cfg.get("scene_geometry") or {}
        self.boxes = _boxes_from_geometry(geometry)
        self.gate = next((b for b in self.boxes if b["motion"] == "kinematic"), None)

        # ---- initial material lattice (frozen block bounds, frozen density)
        sampling = cfg.get("sampling") or {}
        blocks = sampling.get("blocks") or [[-2.6, -0.72, 0.1, -0.55, 0.72, 1.15]]
        clouds, labels = [], []
        for k, block in enumerate(blocks):
            lo = np.asarray(block[:3], float)
            hi = np.asarray(block[3:], float)
            axes = [np.arange(lo[i] + 0.5 * _DX, hi[i] - 1e-9, _DX) for i in range(3)]
            if any(len(axis) == 0 for axis in axes):
                continue
            gx, gy, gz = np.meshgrid(*axes, indexing="ij")
            cloud = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
            clouds.append(cloud)
            labels.append(np.full(len(cloud), k, np.int32))
        if not clouds:
            clouds = [np.zeros((1, 3))]
            labels = [np.zeros(1, np.int32)]

        self.x = np.concatenate(clouds).astype(np.float64)
        self.n = len(self.x)
        self.v = np.zeros_like(self.x)
        self.mass = np.full(self.n, _MP, dtype=np.float64)
        self.body_id = np.concatenate(labels)
        self.volume = np.full(self.n, _DX ** 3, dtype=np.float64)

        # ---- rest density measured from the initial sampling -------------
        _, idx, valid = self._neighbours(self.x)
        r = np.linalg.norm(self.x[idx] - self.x[:, None, :], axis=2)
        w = _W_POLY * (_HS ** 2 - r ** 2) ** 3 * valid
        self.rho0 = float(np.percentile(_MP * w.sum(axis=1), 75.0))
        if not np.isfinite(self.rho0) or self.rho0 <= 0.0:
            self.rho0 = _RHO
        self.volume = np.full(self.n, _MP / self.rho0, dtype=np.float64)

        # ---- frozen normalisation scalars --------------------------------
        extent = self.x.max(axis=0) - self.x.min(axis=0)
        depth = float(max(extent[2], _DX))
        self.characteristic_length = float(max(extent.max(), _DX))
        self.reference_velocity = float(np.sqrt(9.81 * depth))
        self.grid_spacing = float(_GRID_H)
        self.contact_length_scale = float(_DX)

        # ---- constant per-sample attributes ------------------------------
        self.deformation_gradient = np.repeat(np.eye(3)[None, :, :], self.n, axis=0)
        self.material_label = self.body_id.astype(np.int32).copy()
        self.particle_radius = np.full(self.n, 0.45 * _DX)
        self.mass_quantity = self.mass.reshape(self.n, 1)

        # ---- rigid-boundary sample set (wall normals + collocated probes) -
        self._build_boundary_samples()

        self.step_count = 0
        self.clipped = 0
        self._last_time = -1.0

    # ------------------------------------------------------------------
    # neighbour queries
    # ------------------------------------------------------------------
    def _neighbours(self, positions: np.ndarray):
        n = len(positions)
        tree = cKDTree(positions)
        dist, idx = tree.query(positions, k=_KNN, distance_upper_bound=_HS, workers=1)
        valid = dist < _HS
        idx = np.where(valid, idx, 0).astype(np.int64)
        valid = valid & (idx < n)
        return dist, idx, valid

    # ------------------------------------------------------------------
    # boundary sample layout
    # ------------------------------------------------------------------
    @staticmethod
    def _line(a: float, b: float, step: float) -> np.ndarray:
        count = max(2, int(np.floor((b - a) / step)) + 1)
        return np.linspace(a, b, count)

    def _build_boundary_samples(self) -> None:
        samples, normals, moving = [], [], []
        boxes = {b["id"]: b for b in self.boxes}

        def add(points: np.ndarray, normal, is_moving: bool) -> None:
            points = np.asarray(points, float).reshape(-1, 3)
            samples.append(points)
            normals.append(np.tile(np.asarray(normal, float), (len(points), 1)))
            moving.append(np.full(len(points), is_moving))

        ground = boxes.get("ground")
        if ground is not None:
            xs, ys = self._line(-3.0, 3.0, 0.5), self._line(-1.0, 1.0, 0.5)
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            add(np.column_stack((gx.ravel(), gy.ravel(),
                                 np.full(gx.size, ground["hi"][2]))), (0, 0, 1), False)
        for name, sgn in (("tank_side_left", 1.0), ("tank_side_right", -1.0)):
            wall = boxes.get(name)
            if wall is None:
                continue
            wall_y = wall["lo"][1] if sgn > 0 else wall["hi"][1]
            xs, zs = self._line(-3.0, 3.0, 0.5), self._line(0.0, 1.0, 0.5)
            gx, gz = np.meshgrid(xs, zs, indexing="ij")
            add(np.column_stack((gx.ravel(), np.full(gx.size, wall_y), gz.ravel())),
                (0, sgn, 0), False)
        end = boxes.get("tank_end_wall")
        if end is not None:
            ys, zs = self._line(-0.9, 0.9, 0.45), self._line(0.0, 1.0, 0.5)
            gy, gz = np.meshgrid(ys, zs, indexing="ij")
            add(np.column_stack((np.full(gy.size, end["lo"][0]), gy.ravel(), gz.ravel())),
                (-1, 0, 0), False)
        shelf = boxes.get("submerged_shelf")
        if shelf is not None:
            xs, ys = self._line(shelf["lo"][0] + 0.05, shelf["hi"][0] - 0.05, 0.5), \
                self._line(shelf["lo"][1] + 0.04, shelf["hi"][1] - 0.04, 0.4)
            gx, gy = np.meshgrid(xs, ys, indexing="ij")
            add(np.column_stack((gx.ravel(), gy.ravel(),
                                 np.full(gx.size, shelf["hi"][2]))), (0, 0, 1), False)
            ys, zs = self._line(shelf["lo"][1], shelf["hi"][1], 0.4), \
                self._line(shelf["lo"][2] + 0.05, shelf["hi"][2] - 0.05, 0.25)
            gy, gz = np.meshgrid(ys, zs, indexing="ij")
            add(np.column_stack((np.full(gy.size, shelf["lo"][0]), gy.ravel(), gz.ravel())),
                (-1, 0, 0), False)
            add(np.column_stack((np.full(gy.size, shelf["hi"][0]), gy.ravel(), gz.ravel())),
                (1, 0, 0), False)
        if self.gate is not None:
            ys, zs = self._line(self.gate["lo"][1] + 0.04, self.gate["hi"][1] - 0.04, 0.35), \
                self._line(self.gate["lo"][2] + 0.10, self.gate["hi"][2] - 0.10, 0.30)
            gy, gz = np.meshgrid(ys, zs, indexing="ij")
            add(np.column_stack((np.full(gy.size, self.gate["lo"][0]), gy.ravel(), gz.ravel())),
                (-1, 0, 0), True)

        self._bs_base = np.concatenate(samples).astype(np.float64)
        self._bs_normal = np.concatenate(normals).astype(np.float64)
        self._bs_moving = np.concatenate(moving).astype(bool)
        self._bs_gate_z = self._bs_base[self._bs_moving, 2].copy()

    def _gate_lift(self, time: float) -> float:
        return float(min(max(time, 0.0) * _GATE_SPEED, _GATE_LIFT))

    def _gate_velocity(self, time: float) -> float:
        return _GATE_SPEED if 0.0 <= time * _GATE_SPEED < _GATE_LIFT else 0.0

    # ------------------------------------------------------------------
    # rigid-boundary projection
    # ------------------------------------------------------------------
    def _resolve_collisions(self, positions: np.ndarray, time: float) -> None:
        """Project particles out of every rigid box (analytic, vectorised)."""
        offset = _BOUND_OFFSET
        for box in self.boxes:
            lo = box["lo"].astype(np.float64).copy()
            hi = box["hi"].astype(np.float64).copy()
            if box["motion"] == "kinematic":
                dz = self._gate_lift(time)
                lo[2] += dz
                hi[2] += dz
            closest = np.clip(positions, lo, hi)
            delta = positions - closest
            dist = np.linalg.norm(delta, axis=1)
            outside = dist > 1e-9
            near = outside & (dist < offset)
            if near.any():
                nrm = delta[near] / dist[near, None]
                positions[near] = closest[near] + nrm * offset
            inside = ~outside
            if inside.any():
                pts = positions[inside]
                gaps = np.stack((pts - lo, hi - pts), axis=1).reshape(len(pts), -1)
                choice = np.argmin(gaps, axis=1)
                comp, side = choice // 2, choice % 2
                pushed = pts.copy()
                for axis in (0, 1, 2):
                    sel = comp == axis
                    if sel.any():
                        target = np.where(side[sel] == 0, lo[axis] - offset, hi[axis] + offset)
                        pushed[sel, axis] = target
                positions[inside] = pushed

    # ------------------------------------------------------------------
    # one physical substep
    # ------------------------------------------------------------------
    def _step(self, dt: float, time: float) -> None:
        x0 = self.x
        v = self.v + dt * _G
        xp = x0 + dt * v

        _, idx, valid = self._neighbours(xp)
        rel = xp[idx] - xp[:, None, :]

        w = None
        for _ in range(_ITERS):
            r = np.linalg.norm(rel, axis=2)
            r_safe = np.where(r > 1e-9, r, 1e-9)
            w = _W_POLY * (_HS ** 2 - r ** 2) ** 3 * valid
            rho = _MP * w.sum(axis=1)
            # grad = grad_{p_i} W(|p_i - p_j|) points away from neighbour j, so for a
            # positive (compressive) constraint the pair is pushed apart.
            grad = (_GRAD_SPIKY * (_HS - r) ** 2 / r_safe)[:, :, None] * rel * valid[:, :, None]
            gsum = grad.sum(axis=1)
            grad_c2 = ((grad * grad).sum(axis=(1, 2)) + (gsum * gsum).sum(axis=1)) / self.rho0 ** 2
            lam = 1.0 / (grad_c2 + 1e-8)
            s_corr = -_K_AP * (np.maximum(w, 0.0) / _W_DQ) ** _N_AP
            coef = lam[:, None] + lam[idx] + s_corr
            correction = (coef[:, :, None] * grad).sum(axis=1) / self.rho0
            mag = np.linalg.norm(correction, axis=1, keepdims=True)
            scale = np.minimum(1.0, _DP_MAX / np.maximum(mag, 1e-12))
            xp = xp + correction * scale
            rel = xp[idx] - xp[:, None, :]

        self._resolve_collisions(xp, time)

        if not np.isfinite(xp).all():
            self.clipped += 1
            self.v = np.where(np.isfinite(self.v), self.v, 0.0)
            return

        v = (xp - x0) / dt

        # XSPH velocity smoothing on the projected configuration
        r = np.linalg.norm(rel, axis=2)
        w = _W_POLY * (_HS ** 2 - r ** 2) ** 3 * valid
        rho = _MP * w.sum(axis=1)
        vol = (_MP / np.maximum(rho, 1e-9))[idx]
        vj = v[idx]
        v = v + _XSPH * (vol[:, :, None] * (vj - v[:, None, :]) * w[:, :, None]).sum(axis=1)

        speed = np.linalg.norm(v, axis=1, keepdims=True)
        over = speed > _VMAX
        if over.any():
            self.clipped += int(over.sum())
            v = np.where(over, v * (_VMAX / np.maximum(speed, 1e-12)), v)

        self.x = xp
        self.v = v
        self.step_count += 1

    # ------------------------------------------------------------------
    # sampling interface
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t + self.dt_sub <= target - 1e-9 and guard < 100000:
            self._step(self.dt_sub, self.t + self.dt_sub)
            self.t += self.dt_sub
            guard += 1
        remainder = target - self.t
        if remainder > 1e-9:
            # close the interval exactly on the requested sample time
            self._step(remainder, target)
        self.t = target
        self._last_time = target
        return self._frame(target)

    # ------------------------------------------------------------------
    # collocated diagnostic grid
    # ------------------------------------------------------------------
    def _grid_fields(self) -> dict:
        shape = _GRID_SHAPE
        h = _GRID_H
        ncell = int(shape[0] * shape[1] * shape[2])
        frac = (self.x - _GRID_ORIGIN) / h - 0.5
        base = np.floor(frac).astype(np.int64)
        f = frac - base
        vol = self.volume
        occ = np.zeros(ncell)
        mom = np.zeros((ncell, 3))
        for dx in (0, 1):
            wx = (f[:, 0] if dx else 1.0 - f[:, 0])
            ix = np.clip(base[:, 0] + dx, 0, shape[0] - 1)
            for dy in (0, 1):
                wy = (f[:, 1] if dy else 1.0 - f[:, 1])
                iy = np.clip(base[:, 1] + dy, 0, shape[1] - 1)
                wxy = wx * wy * vol
                for dz in (0, 1):
                    wz = (f[:, 2] if dz else 1.0 - f[:, 2])
                    iz = np.clip(base[:, 2] + dz, 0, shape[2] - 1)
                    wt = wxy * wz
                    flat = (ix * shape[1] + iy) * shape[2] + iz
                    occ += np.bincount(flat, weights=wt, minlength=ncell)
                    for c in range(3):
                        mom[:, c] += np.bincount(flat, weights=wt * self.v[:, c], minlength=ncell)

        occ = gaussian_filter(occ.reshape(shape), 1.0, mode="nearest")
        mom = np.stack(
            [gaussian_filter(mom[:, c].reshape(shape), 1.0, mode="nearest") for c in range(3)],
            axis=-1,
        )
        alpha = np.clip(occ / h ** 3, 0.0, 1.0)
        velocity = np.where(occ[..., None] > 1e-9, mom / np.maximum(occ, 1e-12)[..., None], 0.0)
        level = np.clip(h * (1.0 - 2.0 * alpha), -h, h)
        return {
            "liquid_level_set": level,
            "velocity_field": velocity,
            "active_mask": (alpha > 0.5).astype(np.float64),
            "density_field": alpha * _RHO,
        }

    def _fluid_velocity_at(self, points: np.ndarray) -> np.ndarray:
        """Mass-weighted SPH interpolation of the fluid velocity at probe points."""
        out = np.zeros_like(points)
        if self.n == 0:
            return out
        tree = cKDTree(self.x)
        for i, neighbours in enumerate(tree.query_ball_point(points, _HS, workers=1)):
            if not neighbours:
                continue
            j = np.asarray(neighbours, dtype=np.int64)
            d = self.x[j] - points[i]
            r = np.linalg.norm(d, axis=1)
            w = _W_POLY * (_HS ** 2 - r ** 2) ** 3
            total = w.sum()
            if total > 0.0:
                out[i] = (w[:, None] * self.v[j]).sum(axis=0) / total
        return out

    def _boundary_fields(self, time: float) -> dict:
        position = self._bs_base.copy()
        if self._bs_moving.any():
            position[self._bs_moving, 2] = self._bs_gate_z + self._gate_lift(time)
        target = np.zeros_like(position)
        if self._bs_moving.any():
            target[self._bs_moving, 2] = self._gate_velocity(time)
        return {
            "boundary_normal": self._bs_normal.copy(),
            "boundary_target_velocity": target,
            "boundary_velocity": self._fluid_velocity_at(position),
        }

    # ------------------------------------------------------------------
    # frame assembly
    # ------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        fields = self._grid_fields()
        fields.update(self._boundary_fields(time))
        fields["cumulative_mass_source"] = np.zeros(1, dtype=np.float64)
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass_quantity.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.material_label.copy(),
                "deformation_gradient": self.deformation_gradient.copy(),
            },
            fields=fields,
            scalars={
                "grid_spacing": self.grid_spacing,
                "characteristic_length": self.characteristic_length,
                "reference_velocity": self.reference_velocity,
                "contact_length_scale": self.contact_length_scale,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        x = self.x
        return {
            "particle_count": int(self.n),
            "rest_density_kg_m3": float(self.rho0),
            "substeps": int(self.step_count),
            "characteristic_length_m": float(self.characteristic_length),
            "reference_velocity_m_s": float(self.reference_velocity),
            "diagnostic_grid_spacing_m": float(self.grid_spacing),
            "final_time_s": float(self.t),
            "total_mass_kg": float(self.mass.sum()),
            "max_speed_m_s": float(np.linalg.norm(self.v, axis=1).max()) if self.n else 0.0,
            "position_bounds_min": [float(value) for value in x.min(axis=0)],
            "position_bounds_max": [float(value) for value in x.max(axis=0)],
            "clamped_samples": int(self.clipped),
        }
