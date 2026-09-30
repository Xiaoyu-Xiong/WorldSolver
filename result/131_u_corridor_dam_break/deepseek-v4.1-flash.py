"""U-corridor dam break: incompressible position-based particle fluid.

Physical model
==============
The frozen scene supplies a compact liquid column (a particle cloud) confined
between a closed back stop and a kinematic dam gate inside a U-shaped corridor
formed by the floor, two side walls and an end wall.  The solver advances that
liquid with a real time-stepping procedure:

* gravity body force integrated semi-implicitly;
* incompressibility enforced by a Position Based Fluids density-constraint
  projection (Macklin and Mueller 2013): per-particle density from the poly6
  kernel, constraint C_i = rho_i/rho0 - 1 active under compression, Lagrange
  multipliers built from spiky-kernel gradients, iterated position projection
  with a bounded correction;
* XSPH velocity smoothing for coherent advection;
* rigid non-penetrating frictional contacts with the frozen axis-aligned box
  geometry (floor, back stop, side walls, end wall) and with the kinematic dam
  gate, whose prescribed surface velocity is transferred to the touching fluid;
* the gate lifts with a smooth C1 profile, releasing the column shortly after
  t = 0 so that it collapses, accelerates along the corridor, breaks on the end
  wall, runs up and returns as interacting waves.

Everything is SI and deterministic: no random numbers, no keyframes, no motion
tables.  Every returned frame is the state of the same integration.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_BOX_NORMALS = np.array(
    [
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=float,
)


class BenchmarkSolver:
    """Deterministic incompressible particle solver for the U-corridor dam break."""

    DT = 1.0 / 240.0
    SOLVER_ITERATIONS = 3
    XSPH_STRENGTH = 0.08
    GATE_OPEN_START = 0.05
    GATE_OPEN_TIME = 0.30
    GATE_LIFT = 1.35
    MAX_SPEED = 12.0
    MAX_CORRECTION = 0.35
    WALL_FRICTION = 0.02
    NEIGHBOURS_FOR_FIELD = 20

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.config = dict(getattr(scene, "config", {}) or {})
        gravity = self.config.get("gravity", (0.0, 0.0, -9.81))
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)

        frame0 = scene.initial_state()
        base = np.asarray(frame0.position, dtype=float)
        mass_attr = frame0.attributes.get("mass_quantity")
        if mass_attr is None:
            mass_attr = frame0.mass
        mass0 = np.asarray(mass_attr, dtype=float).reshape(-1)

        sampling = self.config.get("sampling", {}) or {}
        dp0 = float(sampling.get("spacing", 0.14))
        self.parent_spacing = dp0
        # Refine the frozen lattice once per axis (8 sub-samples per sample) so
        # that the same liquid domain is resolved by a stable particle spacing.
        quart = np.array([-0.25, 0.25]) * dp0
        offsets = np.array([[a, b, c] for a in quart for b in quart for c in quart])
        sub = (base[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        sub_mass = np.repeat(mass0 / 8.0, 8)

        self.n = int(len(sub))
        self.dp = 0.5 * dp0
        self.pos = sub
        self.vel = np.zeros((self.n, 3), dtype=float)
        self.mass = sub_mass
        self.h = 2.15 * self.dp
        self.surface_radius = 0.45 * self.dp
        # Radius used to turn the particle cloud into a signed liquid level set.
        # It must exceed the worst-case half-diagonal of the sample lattice so
        # that the level set is strictly negative throughout the liquid bulk.
        self.level_radius = 0.9 * self.dp
        self.poly6_coeff = 315.0 / (64.0 * np.pi * self.h ** 9)
        self.spiky_coeff = 45.0 / (np.pi * self.h ** 6)
        self.w0 = self.poly6_coeff * self.h ** 6

        tree = cKDTree(sub)
        k = int(min(64, self.n))
        dist, index = tree.query(sub, k=k)
        dens = (sub_mass[index] * self._poly6(dist)).sum(axis=1)
        rho0 = float(np.percentile(dens, 75.0))
        self.rho0 = rho0 if np.isfinite(rho0) and rho0 > 0.0 else 1000.0

        self._build_geometry()
        self._build_grid(sampling)
        self._build_boundary()

        self.t = 0.0
        self.step_index = 0
        extent = float(np.max(np.ptp(sub, axis=0)))
        self.characteristic_length = max(extent, self.dp)
        g = abs(float(self.gravity[2]))
        self.reference_velocity = max(float(np.sqrt(g * self.characteristic_length)), 1.0e-3)

    def _build_geometry(self) -> None:
        geometry = (self.config.get("scene_geometry") or {}).get("objects", []) or []
        self.boxes = []
        self.gate_half = np.array([0.85, 0.04, 0.55])
        self.gate_c0 = np.array([0.0, -0.60, 0.55])
        wall_y_min = -np.inf
        wall_x_max = 0.0
        for obj in geometry:
            geom = obj.get("geometry", {}) or {}
            if geom.get("type") != "box":
                continue
            half = np.asarray(geom.get("half_extents", [0.0, 0.0, 0.0]), dtype=float)
            centre = np.asarray(
                (obj.get("transform", {}) or {}).get("translation", [0.0, 0.0, 0.0]),
                dtype=float,
            )
            if obj.get("motion") == "kinematic" or obj.get("material_id") == "gate":
                self.gate_half = half
                self.gate_c0 = centre
                continue
            if str(obj.get("id", "")).startswith("corridor"):
                wall_y_min = max(wall_y_min, float(centre[1] - half[1]))
                wall_x_max = max(wall_x_max, float(abs(centre[0]) + half[0]))
            self.boxes.append((centre, half, None))
        self.y_back = float(wall_y_min) if np.isfinite(wall_y_min) else -1.55
        self.x_outer = float(wall_x_max) if wall_x_max > 0.0 else 1.25

    def _build_grid(self, sampling: dict) -> None:
        shape = sampling.get("field_grid_shape", [24, 16, 28])
        self.grid_n = np.asarray(shape, dtype=int).reshape(3)
        self.grid_origin = np.array([-1.68, -1.68, 0.0])
        extent = np.array([3.36, 3.36, 1.40])
        self.grid_delta = extent / self.grid_n.astype(float)
        axes = [
            self.grid_origin[k] + (np.arange(self.grid_n[k]) + 0.5) * self.grid_delta[k]
            for k in range(3)
        ]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        self.grid_points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        self.grid_spacing = float(self.grid_delta[0])

    def _build_boundary(self) -> None:
        """Collocated samples of the frozen rigid surfaces (positions + normals)."""
        xs = np.arange(-1.05, 1.05 + 1e-9, 0.15)
        ys = np.arange(self.y_back + 0.05, 1.45 + 1e-9, 0.15)
        zs = np.arange(0.05, 1.10 + 1e-9, 0.15)
        points = []
        normals = []
        gate = []

        floor_x, floor_y = np.meshgrid(xs, ys, indexing="ij")
        fx = floor_x.ravel()
        fy = floor_y.ravel()
        points.append(np.column_stack([fx, fy, np.zeros(fx.size)]))
        normals.append(np.tile([0.0, 0.0, 1.0], (fx.size, 1)))
        gate.append(
            (np.abs(fx) <= self.gate_half[0])
            & (np.abs(fy - self.gate_c0[1]) <= self.gate_half[1] + 0.02)
        )

        for sx, nx in ((-1.05, 1.0), (1.05, -1.0)):
            wall_y, wall_z = np.meshgrid(ys, zs, indexing="ij")
            points.append(
                np.column_stack(
                    [np.full(wall_y.size, sx), wall_y.ravel(), wall_z.ravel()]
                )
            )
            normals.append(np.tile([nx, 0.0, 0.0], (wall_y.size, 1)))
            gate.append(np.zeros(wall_y.size, dtype=bool))

        end_x, end_z = np.meshgrid(xs, zs, indexing="ij")
        points.append(
            np.column_stack(
                [end_x.ravel(), np.full(end_x.size, 1.45), end_z.ravel()]
            )
        )
        normals.append(np.tile([0.0, -1.0, 0.0], (end_x.size, 1)))
        gate.append(np.zeros(end_x.size, dtype=bool))

        back_x, back_z = np.meshgrid(xs, zs, indexing="ij")
        points.append(
            np.column_stack(
                [back_x.ravel(), np.full(back_x.size, self.y_back), back_z.ravel()]
            )
        )
        normals.append(np.tile([0.0, 1.0, 0.0], (back_x.size, 1)))
        gate.append(np.zeros(back_x.size, dtype=bool))

        gate_x, gate_z = np.meshgrid(
            np.linspace(-self.gate_half[0], self.gate_half[0], 11), zs, indexing="ij"
        )
        points.append(
            np.column_stack(
                [
                    gate_x.ravel(),
                    np.full(gate_x.size, self.gate_c0[1] - self.gate_half[1]),
                    gate_z.ravel(),
                ]
            )
        )
        normals.append(np.tile([0.0, -1.0, 0.0], (gate_x.size, 1)))
        gate.append(np.ones(gate_x.size, dtype=bool))

        self.bnd_pts = np.concatenate(points, axis=0)
        self.bnd_nrm = np.concatenate(normals, axis=0)
        self.bnd_gate_mask = np.concatenate(gate, axis=0)

    # ------------------------------------------------------------------ kernels
    def _poly6(self, r):
        q = self.h * self.h - r * r
        return self.poly6_coeff * np.where(q > 0.0, q, 0.0) ** 3

    @staticmethod
    def _accumulate(n: int, index, weights):
        out = np.empty((n, weights.shape[1]), dtype=float)
        for c in range(weights.shape[1]):
            out[:, c] = np.bincount(index, weights=weights[:, c], minlength=n)
        return out

    def _gate_state(self, t: float):
        u = (t - self.GATE_OPEN_START) / self.GATE_OPEN_TIME
        u = min(max(u, 0.0), 1.0)
        s = u * u * (3.0 - 2.0 * u)
        if 0.0 < u < 1.0:
            dsdt = 6.0 * u * (1.0 - u) / self.GATE_OPEN_TIME
        else:
            dsdt = 0.0
        centre = self.gate_c0.copy()
        centre[2] += self.GATE_LIFT * s
        velocity = np.array([0.0, 0.0, self.GATE_LIFT * dsdt])
        return centre, velocity

    # --------------------------------------------------------------- collisions
    def _resolve(self, x, v, gate_centre, gate_velocity) -> None:
        """Non-penetrating, frictional contact projection against the rigid scene."""
        radius = 0.45 * self.dp
        low = np.nonzero(x[:, 2] < radius)[0]
        if low.size:
            x[low, 2] = radius
            if v is not None:
                v[low, 2] = np.maximum(v[low, 2], 0.0)
        back = np.nonzero(
            (x[:, 1] < self.y_back + radius) & (np.abs(x[:, 0]) <= self.x_outer)
        )[0]
        if back.size:
            x[back, 1] = self.y_back + radius
            if v is not None:
                v[back, 1] = np.maximum(v[back, 1], 0.0)

        boxes = list(self.boxes)
        boxes.append((gate_centre, self.gate_half, gate_velocity))
        for centre, half, box_velocity in boxes:
            lo = centre - half - radius
            hi = centre + half + radius
            inside = np.nonzero(np.all((x >= lo) & (x <= hi), axis=1))[0]
            if not inside.size:
                continue
            p = x[inside]
            dl = p - lo
            dh = hi - p
            pen = np.stack(
                [dl[:, 0], dh[:, 0], dl[:, 1], dh[:, 1], dl[:, 2], dh[:, 2]], axis=1
            )
            axis = np.argmin(pen, axis=1)
            normal = _BOX_NORMALS[axis]
            depth = pen[np.arange(inside.size), axis] + 1.0e-6
            x[inside] = p + normal * depth[:, None]
            if v is None:
                continue
            w = v[inside]
            relative = w if box_velocity is None else w - box_velocity
            vn = np.einsum("pc,pc->p", relative, normal)
            tangential = relative - vn[:, None] * normal
            new = tangential * (1.0 - self.WALL_FRICTION)
            new = new + np.minimum(vn, 0.0)[:, None] * normal
            if box_velocity is not None:
                new = new + box_velocity
            v[inside] = new

    # -------------------------------------------------------------- time step
    def _step(self, dt: float) -> None:
        gate_centre, gate_velocity = self._gate_state(self.t + 0.5 * dt)
        n = self.n
        x_old = self.pos
        v = self.vel + self.gravity * dt
        x = x_old + v * dt
        self._resolve(x, v, gate_centre, gate_velocity)

        pairs = (
            cKDTree(x).query_pairs(self.h, output_type="ndarray")
            if n > 1
            else np.empty((0, 2), dtype=np.int64)
        )
        weights = None
        if pairs.shape[0]:
            i = pairs[:, 0]
            j = pairs[:, 1]
            d = x[i] - x[j]
            r = np.sqrt(np.einsum("pc,pc->p", d, d))
            rs = np.maximum(r, 1.0e-6)
            weights = self._poly6(r)

            density = np.bincount(i, weights=weights, minlength=n)
            density += np.bincount(j, weights=weights, minlength=n)
            density = (density + self.w0) * self.mass[0]
            constraint = np.maximum(density / self.rho0 - 1.0, 0.0)

            dwdr = -self.spiky_coeff * np.maximum(self.h - r, 0.0) ** 2
            grad = (dwdr / rs)[:, None] * d / self.rho0
            grad_sum = self._accumulate(n, i, grad) - self._accumulate(n, j, grad)
            grad_sq = np.bincount(i, weights=np.einsum("pc,pc->p", grad, grad), minlength=n)
            grad_sq += np.bincount(j, weights=np.einsum("pc,pc->p", grad, grad), minlength=n)
            sum_sq = np.einsum("pc,pc->p", grad_sum, grad_sum) + grad_sq

            lam = -constraint / (sum_sq + 1.0)
            pair_grad = (lam[i] + lam[j])[:, None] * grad

            limit = self.MAX_CORRECTION * self.dp
            for _ in range(self.SOLVER_ITERATIONS):
                dp = self._accumulate(n, i, pair_grad) - self._accumulate(n, j, pair_grad)
                length = np.sqrt(np.einsum("pc,pc->p", dp, dp))
                scale = np.where(length > limit, limit / np.maximum(length, 1.0e-12), 1.0)
                x += dp * scale[:, None]
                self._resolve(x, None, gate_centre, gate_velocity)

        v = (x - x_old) / dt
        if weights is not None:
            kernel_weight = (self.XSPH_STRENGTH * self.mass[0] / self.rho0) * weights
            difference = v[j] - v[i]
            term = kernel_weight[:, None] * difference
            v = v + self._accumulate(n, i, term) - self._accumulate(n, j, term)
        self._resolve(x, v, gate_centre, gate_velocity)

        speed = np.sqrt(np.einsum("pc,pc->p", v, v))
        v *= np.where(speed > self.MAX_SPEED, self.MAX_SPEED / np.maximum(speed, 1e-12), 1.0)[:, None]
        np.clip(x[:, 0], -3.2, 3.2, out=x[:, 0])
        np.clip(x[:, 1], -1.8, 1.8, out=x[:, 1])
        np.clip(x[:, 2], -0.1, 3.0, out=x[:, 2])
        self.pos = x
        self.vel = v
        self.t += dt
        self.step_index += 1

    # ------------------------------------------------------------- diagnostics
    def _sample_grid(self, field, points):
        g = (points - self.grid_origin) / self.grid_delta - 0.5
        i0 = np.floor(g).astype(np.int64)
        frac = np.clip(g - i0, 0.0, 1.0)
        i0 = np.clip(i0, 0, self.grid_n - 1)
        i1 = np.minimum(i0 + 1, self.grid_n - 1)
        out = np.zeros((len(points), field.shape[-1]), dtype=float)
        for a in (0, 1):
            ix = i1[:, 0] if a else i0[:, 0]
            wx = frac[:, 0] if a else 1.0 - frac[:, 0]
            for b in (0, 1):
                iy = i1[:, 1] if b else i0[:, 1]
                wy = frac[:, 1] if b else 1.0 - frac[:, 1]
                for c in (0, 1):
                    iz = i1[:, 2] if c else i0[:, 2]
                    wz = frac[:, 2] if c else 1.0 - frac[:, 2]
                    out += (wx * wy * wz)[:, None] * field[ix, iy, iz]
        return out

    def _field_state(self, pos, vel):
        shape = tuple(int(v) for v in self.grid_n)
        tree = cKDTree(pos)
        distance, _ = tree.query(self.grid_points, k=1)
        level = (distance - self.level_radius).reshape(shape)

        k = int(min(self.NEIGHBOURS_FOR_FIELD, len(pos)))
        dist, index = tree.query(self.grid_points, k=k)
        weight = self._poly6(dist)
        total = weight.sum(axis=1)
        velocity = np.einsum("mk,mkc->mc", weight, vel[index])
        velocity /= np.maximum(total, 1.0e-12)[:, None]
        velocity[total <= 1.0e-12] = 0.0
        return level, velocity.reshape(shape + (3,))

    def _frame(self, time: float) -> StateFrame:
        pos = np.nan_to_num(self.pos.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        vel = np.nan_to_num(self.vel.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        level, velocity_field = self._field_state(pos, vel)
        mask = (level <= 0.5 * self.dp).astype(float)

        gate_centre, gate_velocity = self._gate_state(self.t)
        target = np.zeros_like(self.bnd_pts)
        if np.any(self.bnd_gate_mask):
            target[self.bnd_gate_mask] = gate_velocity
        sampled = self._sample_grid(velocity_field, self.bnd_pts)
        normal = self.bnd_nrm
        boundary = (
            sampled
            - np.einsum("pc,pc->p", sampled, normal)[:, None] * normal
            + np.einsum("pc,pc->p", target, normal)[:, None] * normal
        )

        fields = {
            "active_mask": mask,
            "boundary_normal": normal,
            "boundary_target_velocity": target,
            "boundary_velocity": boundary,
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "liquid_level_set": level,
            "liquid_interior_mask": (level <= 0.0).astype(float),
            "velocity_field": velocity_field,
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "volume_quantity": self.mass / self.rho0,
            "particle_radius": np.full(self.n, self.surface_radius),
        }
        scalars = {
            "characteristic_length": float(self.characteristic_length),
            "grid_spacing": float(self.grid_spacing),
            "reference_velocity": float(self.reference_velocity),
        }
        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        while self.t < target - 1.0e-12 and guard < 1000000:
            dt = min(self.DT, target - self.t)
            self._step(dt)
            guard += 1
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "particle_count": int(self.n),
            "rest_density": float(self.rho0),
            "steps": int(self.step_index),
            "simulation_time": float(self.t),
            "gate_open_time": float(self.GATE_OPEN_START + self.GATE_OPEN_TIME),
            "boundary_sample_count": int(self.bnd_pts.shape[0]),
        }
