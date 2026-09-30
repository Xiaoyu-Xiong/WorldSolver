"""Dam break through a rigid arch obstacle.

Physically based particle fluid solver (position based incompressibility,
Macklin & Muller style density constraints) with rigid axis-aligned box
obstacles, a kinematic lifting dam gate and a rectangular tank domain.

The tracer particles are the fluid samples; the diagnostic collocated grid
fields (velocity, level set, active mask) are reconstructed from them.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

from worldsolver.contracts import StateFrame

GRAVITY_Z = -9.81
GATE_OPEN_TIME = 0.20
GATE_SPEED = 1.5
MAX_SPEED = 14.0
SOLVER_ITERATIONS = 3
DT_INTERNAL = 1.0 / 300.0
K_CORR = 2.0e-4


class BenchmarkSolver:
    """Incompressible particle fluid with rigid boundary interaction."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config
        sampling = dict(config.get("sampling", {}) or {})
        self.spacing = float(sampling.get("spacing", 0.14))
        self.density = float(sampling.get("density", 1000.0))

        frame0 = scene.initial_state()
        self.positions = np.array(frame0.position, dtype=np.float64)
        self.velocities = np.array(frame0.velocity, dtype=np.float64)
        mass = frame0.mass
        if mass is None:
            mass = np.full(len(self.positions), self.density * self.spacing ** 3)
        self.mass = np.array(mass, dtype=np.float64).reshape(-1)
        body = frame0.body_id
        if body is None:
            body = np.zeros(len(self.positions), dtype=np.int32)
        self.body_id = np.array(body, dtype=np.int32)
        self.n = int(self.positions.shape[0])
        self.particle_radius = np.full(self.n, 0.45 * self.spacing, dtype=np.float64)

        # kernel parameters (cubic spline-free: poly6 smoothing kernel)
        self.h = 2.2 * self.spacing
        self.h2 = self.h * self.h
        self.poly6 = 315.0 / (64.0 * np.pi * self.h ** 9)
        self.grad_coef = -1890.0 / (64.0 * np.pi * self.h ** 9)
        self.w_dq = self.poly6 * (self.h2 - (0.2 * self.h) ** 2) ** 3

        # rest density from the bulk of the frozen initial sample
        rho0 = self._raw_density(self.positions)
        self.rho0 = float(np.percentile(rho0, 95.0))
        if not np.isfinite(self.rho0) or self.rho0 <= 0.0:
            self.rho0 = float(self.mass.mean() / self.spacing ** 3)
        self.specific_volume = float(self.mass.mean() / self.rho0)

        # frozen normalization scales
        span = np.ptp(self.positions, axis=0)
        self.characteristic_length = float(max(span.max(), self.spacing))
        self.reference_velocity = float(np.sqrt(9.81 * max(self.characteristic_length, self.spacing)))

        # ---- rigid geometry from the frozen scene description ----
        self.boxes = []          # static obstacles + floor
        self.gate = None
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []) or []:
            geom = obj.get("geometry", {}) or {}
            if geom.get("type") != "box":
                continue
            half = np.array(geom.get("half_extents", [0.0, 0.0, 0.0]), dtype=np.float64)
            trans = obj.get("transform", {}) or {}
            center = np.array(trans.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
            entry = {
                "id": str(obj.get("id", "")),
                "center": center,
                "half": np.maximum(half, 1e-6),
                "motion": str(obj.get("motion", "static")),
            }
            if entry["motion"] == "kinematic":
                self.gate = entry
            else:
                self.boxes.append(entry)
        self.static_boxes = list(self.boxes)
        if self.gate is not None:
            self.boxes.append(self.gate)

        # tank domain: floor half extents define the rectangular tank walls
        ground_half = np.array([3.2, 1.4, 0.05], dtype=np.float64)
        for entry in self.static_boxes:
            if entry["id"] == "ground":
                ground_half = entry["half"].copy()
        self.tank_min = np.array([-ground_half[0], -ground_half[1], 0.0])
        self.tank_max = np.array([ground_half[0], ground_half[1], 1.0])

        top = 1.58
        for entry in self.boxes:
            top = max(top, float(entry["center"][2] + entry["half"][2]))
        zmax = top + 0.4 * self.spacing * 10.0
        self.grid_low = np.array([self.tank_min[0], self.tank_min[1], 0.0])
        self.grid_high = np.array([self.tank_max[0], self.tank_max[1], zmax])

        shape = sampling.get("field_grid_shape", [24, 16, 28])
        self.grid_shape = tuple(int(v) for v in shape)
        self.axis_spacing = (self.grid_high - self.grid_low) / np.array(self.grid_shape, float)
        self.grid_spacing = float(np.mean(self.axis_spacing))

        # boundary samples from the frozen rigid boxes
        self._build_boundary_samples()

        self.time = 0.0
        self.steps = 0
        self.initialized = True

    # ------------------------------------------------------------------
    # frozen wall sample set
    # ------------------------------------------------------------------
    def _build_boundary_samples(self) -> None:
        normals = []
        targets = []
        centers = []
        gate_flags = []
        for entry in self.static_boxes + ([self.gate] if self.gate is not None else []):
            is_gate = entry is self.gate
            center = entry["center"]
            half = entry["half"]
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    normal = np.zeros(3)
                    normal[axis] = sign
                    face_center = center.copy()
                    face_center[axis] += sign * half[axis]
                    u, v = [a for a in range(3) if a != axis]
                    nu = 3 if self.n > 8 else 2
                    nv = 3
                    for iu in range(nu):
                        for iv in range(nv):
                            point = face_center.copy()
                            point[u] += (iu + 0.5) / nu * 2.0 * half[u] - half[u]
                            point[v] += (iv + 0.5) / nv * 2.0 * half[v] - half[v]
                            normals.append(normal)
                            centers.append(point)
                            targets.append(np.zeros(3))
                            gate_flags.append(is_gate)
        self.boundary_normal = np.array(normals, dtype=np.float64)
        self.boundary_target0 = np.array(targets, dtype=np.float64)
        self.boundary_position = np.array(centers, dtype=np.float64)
        self.boundary_is_gate = np.array(gate_flags, dtype=bool)

    def _gate_offset(self, time: float) -> float:
        if self.gate is None:
            return 0.0
        return float(GATE_SPEED * max(0.0, time - GATE_OPEN_TIME))

    # ------------------------------------------------------------------
    # kernels
    # ------------------------------------------------------------------
    def _raw_density(self, p: np.ndarray) -> np.ndarray:
        w, _ = self._kernel(p)
        return float(self.mass.mean()) * w.sum(axis=1)

    def _kernel(self, p: np.ndarray):
        d = p[:, None, :] - p[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        near = r2 < self.h2
        fac = np.where(near, self.h2 - r2, 0.0)
        w = self.poly6 * fac ** 3
        r = np.sqrt(np.maximum(r2, 1e-24))
        gw = (self.grad_coef * r * fac ** 2 / np.where(near, r, 1.0))[:, :, None] * d
        gw[~near] = 0.0
        return w, gw

    # ------------------------------------------------------------------
    # collision handling
    # ------------------------------------------------------------------
    def _resolve(self, p: np.ndarray, gate_offset: float) -> None:
        if self.gate is not None:
            self._resolve_box(p, self.gate, gate_offset)
        for entry in self.static_boxes:
            self._resolve_box(p, entry, gate_offset)
        # rectangular tank side walls
        margin = 0.25 * self.spacing
        np.clip(p[:, 0], self.grid_low[0] + margin, self.grid_high[0] - margin, out=p[:, 0])
        np.clip(p[:, 1], self.grid_low[1] + margin, self.grid_high[1] - margin, out=p[:, 1])
        np.clip(p[:, 2], 0.0, self.grid_high[2], out=p[:, 2])

    def _resolve_box(self, p: np.ndarray, entry: dict, gate_offset: float) -> None:
        center = entry["center"].copy()
        if entry is self.gate:
            center[2] += gate_offset
        half = entry["half"]
        rel = p - center
        inside = np.all(np.abs(rel) < half, axis=1)
        idx = np.nonzero(inside)[0]
        if idx.size == 0:
            return
        pen = half[None, :] - np.abs(rel[idx])
        axis = np.argmin(pen, axis=1)
        sgn = np.sign(rel[idx, axis])
        sgn[sgn == 0.0] = 1.0
        pad = 1e-3 * self.spacing
        p[idx, axis] = center[axis] + sgn * (half[axis] + pad)

    # ------------------------------------------------------------------
    # one physical time step
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        self.velocities[:, 2] += GRAVITY_Z * dt
        prev = self.positions.copy()
        p = self.positions + dt * self.velocities
        gate_offset = self._gate_offset(self.time + dt)
        self._resolve(p, gate_offset)
        for _ in range(SOLVER_ITERATIONS):
            corr = self._pressure_correction(p)
            p += np.clip(corr, -0.4 * self.spacing, 0.4 * self.spacing)
            self._resolve(p, gate_offset)
        v = (p - prev) / dt
        v = self._xshp(p, v)
        speed = np.linalg.norm(v, axis=1)
        scale = np.where(speed > MAX_SPEED, MAX_SPEED / np.maximum(speed, 1e-12), 1.0)
        v *= scale[:, None]
        bad = ~np.isfinite(v)
        if np.any(bad):
            v[bad] = 0.0
        bad_p = ~np.isfinite(p)
        if np.any(bad_p):
            p[bad_p] = prev[bad_p]
            v[bad_p] = 0.0
        self.positions = p
        self.velocities = v
        self.time += dt
        self.steps += 1

    def _pressure_correction(self, p: np.ndarray) -> np.ndarray:
        w, gw = self._kernel(p)
        rho = float(self.mass.mean()) * w.sum(axis=1)
        c = np.maximum(rho / self.rho0 - 1.0, 0.0)
        gsum = gw.sum(axis=1)
        gsq = np.einsum("ijk,ijk->i", gw, gw)
        denom = (np.einsum("ij,ij->i", gsum, gsum) + gsq) / (self.rho0 ** 2)
        lam = -c / np.maximum(denom, 1e-12)
        lam = np.clip(lam, -5.0e3, 0.0)
        ratio = np.clip(w / max(self.w_dq, 1e-30), 0.0, 1.0)
        scorr = -K_CORR * ratio ** 4
        coef = lam[:, None] + lam[None, :] + scorr
        corr = (gw * coef[:, :, None]).sum(axis=1) / self.rho0
        bad = ~np.isfinite(corr)
        if np.any(bad):
            corr[bad] = 0.0
        return corr

    def _xshp(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        d = p[:, None, :] - p[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        near = r2 < self.h2
        fac = np.where(near, self.h2 - r2, 0.0)
        w = self.poly6 * fac ** 3
        dv = w[:, :, None] * (v[None, :, :] - v[:, None, :])
        return v + 0.002 * self.specific_volume * dv.sum(axis=1)

    # ------------------------------------------------------------------
    # diagnostic collocated grid
    # ------------------------------------------------------------------
    def _cell_index(self, p: np.ndarray) -> np.ndarray:
        idx = np.floor((p - self.grid_low) / self.axis_spacing).astype(np.int64)
        return np.clip(idx, 0, np.array(self.grid_shape, dtype=np.int64) - 1)

    def _grid_fields(self, p: np.ndarray, v: np.ndarray):
        shape = self.grid_shape
        idx = self._cell_index(p)
        flat = idx[:, 0] * shape[1] * shape[2] + idx[:, 1] * shape[2] + idx[:, 2]

        size = int(np.prod(shape))
        counts = np.bincount(flat, minlength=size).astype(np.float64)
        acc = np.zeros((size, 3), dtype=np.float64)
        for c in range(3):
            acc[:, c] = np.bincount(flat, weights=v[:, c], minlength=size)
        cnt_grid = counts.reshape(shape)
        occ = np.zeros(shape, dtype=np.float64)
        np.add.at(occ, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)

        smoothed_cnt = gaussian_filter(cnt_grid, 1.0)
        velocity = np.zeros(shape + (3,), dtype=np.float64)
        for c in range(3):
            num = gaussian_filter(acc[:, c].reshape(shape), 1.0)
            velocity[..., c] = np.where(smoothed_cnt > 1e-6, num / np.maximum(smoothed_cnt, 1e-9), 0.0)

        has = cnt_grid > 0.0
        if not np.all(has):
            _, nearest = distance_transform_edt(~has, return_indices=True)
            velocity = velocity[nearest[0], nearest[1], nearest[2]]
        bad = ~np.isfinite(velocity)
        if np.any(bad):
            velocity[bad] = 0.0

        solid = occ >= 1.0
        solid = binary_dilation(solid, iterations=1)
        solid = gaussian_filter(solid.astype(np.float64), 0.7) > 0.35
        if not np.any(solid):
            solid = occ >= 1.0
        if np.any(solid) and not np.all(solid):
            sampling = tuple(self.axis_spacing.tolist())
            inside = distance_transform_edt(solid, sampling=sampling)
            outside = distance_transform_edt(~solid, sampling=sampling)
            level = outside - inside
        else:
            level = np.full(shape, 1.0)
        level = np.where(np.isfinite(level), level, 0.0)

        active = self._active_mask()
        return velocity, level, active

    def _active_mask(self) -> np.ndarray:
        shape = self.grid_shape
        axes = [self.grid_low[a] + (np.arange(shape[a]) + 0.5) * self.axis_spacing[a]
                for a in range(3)]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        points = np.stack((gx, gy, gz), axis=-1)
        inside = np.ones(shape, dtype=bool)
        for entry in self.static_boxes:
            center = entry["center"]
            half = entry["half"]
            in_box = np.all(np.abs(points - center[None, None, None, :]) < half[None, None, None, :], axis=-1)
            inside &= ~in_box
        return inside

    # ------------------------------------------------------------------
    # frame assembly
    # ------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        p = self.positions
        v = self.velocities
        velocity, level, active = self._grid_fields(p, v)
        targets = self.boundary_target0.copy()
        if self.gate is not None:
            moving = self.boundary_is_gate
            if np.any(moving):
                targets[moving, 2] = GATE_SPEED
        # impermeable no-slip walls: fluid-side wall velocity equals the target
        boundary_velocity = targets

        frame = StateFrame(
            float(time),
            p.copy(),
            v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.body_id.copy(),
                "phase_id": np.zeros(self.n, dtype=np.float64),
            },
            fields={
                "velocity_field": velocity,
                "liquid_level_set": level,
                "active_mask": active.astype(np.float64),
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_velocity": boundary_velocity.copy(),
                "boundary_target_velocity": targets.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            },
            scalars={
                "characteristic_length": self.characteristic_length,
                "grid_spacing": self.grid_spacing,
                "reference_velocity": self.reference_velocity,
                "liquid_phase_id": 0.0,
                "rho0": self.rho0,
                "time_step": DT_INTERNAL,
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time:
            target = self.time
        guard = 0
        while self.time < target - 1e-12:
            dt = min(DT_INTERNAL, target - self.time)
            if dt <= 1e-12:
                break
            self._step(dt)
            guard += 1
            if guard > 2000000:  # pragma: no cover - safety valve
                self.time = target
                break
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        speed = np.linalg.norm(self.velocities, axis=1)
        return {
            "solver": "position_based_particle_fluid",
            "particle_count": int(self.n),
            "steps": int(self.steps),
            "final_time": float(self.time),
            "rest_density": float(self.rho0),
            "max_speed": float(speed.max()) if len(speed) else 0.0,
            "mean_mass": float(self.mass.mean()) if len(self.mass) else 0.0,
        }
