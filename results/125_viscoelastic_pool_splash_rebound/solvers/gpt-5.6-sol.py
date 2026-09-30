"""Deterministic viscoelastic material-point solver for the splash scene."""
from __future__ import annotations

import itertools

import numpy as np

from worldsolver.contracts import StateFrame

class BenchmarkSolver:
    """A compact 3-D MLS-MPM solver with a relaxing elastic constitutive law."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=np.float64).copy()
        self.v = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.F = np.asarray(initial.attributes["deformation_gradient"],
                            dtype=np.float64).copy()
        self.C = np.zeros((len(self.x), 3, 3), dtype=np.float64)

        config = scene.config
        self.gravity = np.asarray(config["gravity"], dtype=np.float64)
        self.spacing = float(config["sampling"]["spacing"])
        self.density = float(config["sampling"]["density"])
        self.volume0 = self.mass / self.density
        self.radius = 0.45 * self.spacing

        # Quadratic B-spline grid enclosing the frozen particle domain.
        self.dx = self.spacing
        self.inv_dx = 1.0 / self.dx
        self.origin = np.array([-3.22, -1.40, -0.14], dtype=np.float64)
        upper = np.array([3.22, 1.40, 3.92], dtype=np.float64)
        self.grid_shape = tuple(
            (np.ceil((upper - self.origin) / self.dx).astype(int) + 1).tolist()
        )
        self.grid_size = int(np.prod(self.grid_shape))
        self.offsets = np.asarray(list(itertools.product(range(3), repeat=3)),
                                  dtype=np.int64)

        # Corotated elasticity and gradual shear relaxation model a material
        # which stretches cohesively and then pulls itself back together.
        young, poisson = 3200.0, 0.34
        self.mu = young / (2.0 * (1.0 + poisson))
        self.lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.relaxation_time = 0.72
        self.max_step = 0.0025
        self.time = 0.0
        self.steps = 0

        # Exact AABBs of all four collidable boxes in scene_geometry.yaml.
        self.boxes = (
            (np.array([-3.2, -1.4, -0.10]), np.array([3.2, 1.4, 0.0])),
            (np.array([-1.45, 0.64, -0.04]), np.array([1.45, 0.80, 0.64])),
            (np.array([-1.53, -0.72, -0.04]), np.array([-1.37, 0.72, 0.64])),
            (np.array([1.37, -0.72, -0.04]), np.array([1.53, 0.72, 0.64])),
        )

    @staticmethod
    def _weights(fx: np.ndarray) -> np.ndarray:
        """Quadratic B-spline weights as [particle, axis, node]."""
        w = np.empty((len(fx), 3, 3), dtype=np.float64)
        w[:, :, 0] = 0.5 * (1.5 - fx) ** 2
        w[:, :, 1] = 0.75 - (fx - 1.0) ** 2
        w[:, :, 2] = 0.5 * (fx - 0.5) ** 2
        return w

    def _stencil(self):
        grid_x = (self.x - self.origin) * self.inv_dx
        base = np.floor(grid_x - 0.5).astype(np.int64)
        fx = grid_x - base
        axis_w = self._weights(fx)
        offsets = self.offsets
        nodes = base[:, None, :] + offsets[None, :, :]
        nodes[:, :, 0] = np.clip(nodes[:, :, 0], 0, self.grid_shape[0] - 1)
        nodes[:, :, 1] = np.clip(nodes[:, :, 1], 0, self.grid_shape[1] - 1)
        nodes[:, :, 2] = np.clip(nodes[:, :, 2], 0, self.grid_shape[2] - 1)
        weight = (axis_w[:, 0, offsets[:, 0]] *
                  axis_w[:, 1, offsets[:, 1]] *
                  axis_w[:, 2, offsets[:, 2]])
        flat = ((nodes[:, :, 0] * self.grid_shape[1] + nodes[:, :, 1]) *
                self.grid_shape[2] + nodes[:, :, 2])
        dpos = (offsets[None, :, :] - fx[:, None, :]) * self.dx
        return flat, weight, dpos

    def _grid_contact(self, velocity: np.ndarray, active: np.ndarray) -> None:
        ny, nz = self.grid_shape[1], self.grid_shape[2]
        ix = active // (ny * nz)
        rem = active - ix * ny * nz
        iy, iz = rem // nz, rem % nz
        p = self.origin + self.dx * np.column_stack((ix, iy, iz))
        gv = velocity[active]

        floor = p[:, 2] <= 0.02
        downward = floor & (gv[:, 2] < 0.0)
        gv[downward, 2] = 0.0
        gv[floor, :2] *= 0.94
        for lo, hi in self.boxes[1:]:
            solid = np.all((p >= lo) & (p <= hi), axis=1)
            gv[solid] = 0.0

        margin = 2
        gv[(ix < margin) & (gv[:, 0] < 0.0), 0] = 0.0
        gv[(ix >= self.grid_shape[0] - margin) & (gv[:, 0] > 0.0), 0] = 0.0
        gv[(iy < margin) & (gv[:, 1] < 0.0), 1] = 0.0
        gv[(iy >= self.grid_shape[1] - margin) & (gv[:, 1] > 0.0), 1] = 0.0
        gv[(iz >= self.grid_shape[2] - margin) & (gv[:, 2] > 0.0), 2] = 0.0
        velocity[active] = gv

    def _particle_contact(self) -> None:
        """Project particle spheres out of the frozen AABBs with friction."""
        r = self.radius
        for lo, hi in self.boxes:
            closest = np.clip(self.x, lo, hi)
            delta = self.x - closest
            dist = np.linalg.norm(delta, axis=1)
            hit = (dist < r) & (dist > 1.0e-12)
            if np.any(hit):
                normal = delta[hit] / dist[hit, None]
                self.x[hit] = closest[hit] + r * normal
                vn = np.sum(self.v[hit] * normal, axis=1)
                incoming = vn < 0.0
                corrected = self.v[hit]
                corrected[incoming] -= 1.08 * vn[incoming, None] * normal[incoming]
                tangent = corrected - np.sum(corrected * normal, axis=1)[:, None] * normal
                corrected -= 0.10 * tangent
                self.v[hit] = corrected

            inside = np.all((self.x >= lo) & (self.x <= hi), axis=1)
            indices = np.flatnonzero(inside)
            if len(indices):
                q = self.x[indices]
                gaps = np.column_stack((q[:, 0] - lo[0], hi[0] - q[:, 0],
                                        q[:, 1] - lo[1], hi[1] - q[:, 1],
                                        q[:, 2] - lo[2], hi[2] - q[:, 2]))
                face = np.argmin(gaps, axis=1)
                normals = np.zeros((len(indices), 3), dtype=np.float64)
                axis = face // 2
                sign = np.where(face % 2 == 0, -1.0, 1.0)
                normals[np.arange(len(indices)), axis] = sign
                boundary = np.where(face % 2 == 0, lo[axis], hi[axis])
                self.x[indices, axis] = boundary + sign * r
                vn = np.sum(self.v[indices] * normals, axis=1)
                incoming = vn < 0.0
                corrected = self.v[indices]
                corrected[incoming] -= 1.08 * vn[incoming, None] * normals[incoming]
                corrected *= 0.96
                self.v[indices] = corrected

    def _substep(self, dt: float) -> None:
        eye = np.eye(3)[None, :, :]

        # Relax shear but retain elastic volume response.
        u, singular, vh = np.linalg.svd(self.F)
        singular = np.clip(singular, 0.48, 2.10)
        geometric_mean = np.cbrt(np.prod(singular, axis=1))[:, None]
        keep = np.exp(-dt / self.relaxation_time)
        singular = geometric_mean + keep * (singular - geometric_mean)
        self.F = (u * singular[:, None, :]) @ vh

        rotation = u @ vh
        determinant = np.clip(np.linalg.det(self.F), 0.35, 2.5)
        finv_t = np.linalg.inv(self.F).transpose(0, 2, 1)
        piola = (2.0 * self.mu * (self.F - rotation) +
                 self.lam * (determinant - 1.0)[:, None, None] *
                 determinant[:, None, None] * finv_t)
        kirchhoff = piola @ self.F.transpose(0, 2, 1)
        stress_affine = (-dt * 4.0 * self.inv_dx ** 2 *
                         self.volume0[:, None, None] * kirchhoff)
        affine = stress_affine + self.mass[:, None, None] * self.C

        flat, weight, dpos = self._stencil()
        momentum = (self.mass[:, None, None] * self.v[:, None, :] +
                    np.einsum("nij,nkj->nki", affine, dpos))
        ids = flat.ravel()
        grid_mass = np.bincount(
            ids, weights=(weight * self.mass[:, None]).ravel(),
            minlength=self.grid_size)
        weighted_momentum = weight[:, :, None] * momentum
        grid_momentum = np.empty((self.grid_size, 3), dtype=np.float64)
        for axis in range(3):
            grid_momentum[:, axis] = np.bincount(
                ids, weights=weighted_momentum[:, :, axis].ravel(),
                minlength=self.grid_size)

        active = np.flatnonzero(grid_mass > 1.0e-14)
        grid_velocity = np.zeros_like(grid_momentum)
        grid_velocity[active] = (grid_momentum[active] /
                                 grid_mass[active, None] + dt * self.gravity)
        self._grid_contact(grid_velocity, active)

        sampled = grid_velocity[flat]
        self.v = np.sum(weight[:, :, None] * sampled, axis=1)
        self.C = 4.0 * self.inv_dx ** 2 * np.einsum(
            "nk,nki,nkj->nij", weight, sampled, dpos)
        self.v *= np.exp(-0.055 * dt)
        speed = np.linalg.norm(self.v, axis=1)
        excessive = speed > 12.0
        self.v[excessive] *= (12.0 / speed[excessive])[:, None]

        self.F = (eye + dt * self.C) @ self.F
        self.x += dt * self.v
        self._particle_contact()
        self.steps += 1

    @staticmethod
    def _box_sdf(points: np.ndarray, lo: np.ndarray,
                 hi: np.ndarray) -> np.ndarray:
        center = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo)
        q = np.abs(points - center) - half
        return (np.linalg.norm(np.maximum(q, 0.0), axis=1) +
                np.minimum(np.max(q, axis=1), 0.0))

    def _frame(self, requested_time: float) -> StateFrame:
        distance = np.full(len(self.x), np.inf, dtype=np.float64)
        for lo, hi in self.boxes:
            distance = np.minimum(distance, self._box_sdf(self.x, lo, hi))
        distance -= self.radius
        external_force = self.mass[:, None] * self.gravity[None, :]
        material_count = max(1, int(self.body_id.max()) + 1)
        frame = StateFrame(
            time=float(requested_time), position=self.x.copy(),
            velocity=self.v.copy(), mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": self.F.copy(),
                "mass_quantity": self.mass.copy(),
                "particle_radius": np.full(len(self.x), self.radius),
                "material_label": self.body_id.copy(),
            },
            fields={
                "contact_signed_distance": distance,
                "cumulative_mass_source": np.zeros(material_count),
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": self.spacing,
                "grid_spacing": self.dx,
                "characteristic_length": 3.0,
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to times must be finite and nondecreasing")
        while self.time < target - 1.0e-12:
            dt = min(self.max_step, target - self.time)
            self._substep(dt)
            self.time += dt
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "3D MLS-MPM with relaxing corotated elasticity",
            "substeps": self.steps,
            "particle_count": len(self.x),
            "time": self.time,
        }
