"""A small, self contained position-based thin-sheet solver.

The implementation deliberately has no dependency on a rigid/cloth simulator.  The
sheet is advanced with semi-implicit Euler followed by iterative inextensibility,
bending, rigid contact, and self-contact projections.  The funnel collision model is
the exact union of the four tapered wall prisms used by ``assets/funnel.obj``.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic PBD cloth solver for the narrow-funnel scene."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        initial = scene.initial_state()
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = np.zeros(len(self.position), dtype=np.int32)
        self.time = 0.0
        self.steps = 0

        self.nx, self.ny = map(int, cfg["cloth"]["grid_shape"])
        self.thickness = float(cfg["cloth"]["thickness"])
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        self.external_force = self.mass[:, None] * self.gravity[None, :]
        self.rest_position = self.position.copy()
        self.rest_dx = float(cfg["cloth"]["size"][0]) / (self.nx - 1)
        self.rest_dy = float(cfg["cloth"]["size"][1]) / (self.ny - 1)

        # A 1/240 s mechanical step gives eight equal steps per output sample.
        self.max_dt = 1.0 / 240.0
        self.solver_iterations = 7
        self.particle_radius = 0.80 * self.thickness
        self.self_separation = 1.25 * self.thickness

        # Constraint families.  Keeping them separate lets a Jacobi solve retain
        # useful stiffness without directional bias.
        grid = np.arange(self.nx * self.ny, dtype=np.int32).reshape(self.ny, self.nx)
        self.constraint_groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        self._add_constraint_group(grid[:, :-1], grid[:, 1:], 0.96)
        self._add_constraint_group(grid[:-1, :], grid[1:, :], 0.96)
        self._add_constraint_group(grid[:-1, :-1], grid[1:, 1:], 0.72)
        self._add_constraint_group(grid[:-1, 1:], grid[1:, :-1], 0.72)
        # Long springs are a simple discrete bending energy: they allow sharp folds
        # while suppressing grid-scale hinge noise.
        self._add_constraint_group(grid[:, :-2], grid[:, 2:], 0.12)
        self._add_constraint_group(grid[:-2, :], grid[2:, :], 0.12)

        # Non-neighbour vertex pairs used for a thickness barrier.  Adjacent and
        # two-ring material neighbours are excluded so ordinary bending is free.
        ids = np.arange(self.nx * self.ny, dtype=np.int32)
        ii, jj = np.triu_indices(len(ids), 1)
        ix, iy = ii % self.nx, ii // self.nx
        jx, jy = jj % self.nx, jj // self.nx
        keep = np.maximum(np.abs(ix - jx), np.abs(iy - jy)) > 2
        self.self_i = ii[keep].astype(np.int32)
        self.self_j = jj[keep].astype(np.int32)

        # Taper data read directly from the frozen funnel mesh dimensions.
        self.funnel_z0 = 0.48
        self.funnel_z1 = 1.72
        height = self.funnel_z1 - self.funnel_z0
        self.slope_x = (1.27 - 0.30) / height
        self.slope_y = (0.87 - 0.20) / height
        self._last_contact_count = 0

    def _add_constraint_group(self, a, b, stiffness: float) -> None:
        i = np.asarray(a, dtype=np.int32).ravel()
        j = np.asarray(b, dtype=np.int32).ravel()
        rest = np.linalg.norm(self.rest_position[i] - self.rest_position[j], axis=1)
        self.constraint_groups.append((i, j, rest, float(stiffness)))

    def _solve_distances(self, p: np.ndarray, group) -> None:
        i, j, rest, stiffness = group
        delta = p[i] - p[j]
        length = np.linalg.norm(delta, axis=1)
        good = length > 1.0e-12
        scale = np.zeros_like(length)
        scale[good] = -0.5 * stiffness * (length[good] - rest[good]) / length[good]
        correction = delta * scale[:, None]

        accum = np.zeros_like(p)
        count = np.zeros(len(p), dtype=np.float64)
        np.add.at(accum, i, correction)
        np.add.at(accum, j, -correction)
        np.add.at(count, i, 1.0)
        np.add.at(count, j, 1.0)
        active = count > 0.0
        p[active] += accum[active] / count[active, None]

    def _solve_self_contact(self, p: np.ndarray) -> None:
        i, j = self.self_i, self.self_j
        delta = p[i] - p[j]
        d2 = np.einsum("ij,ij->i", delta, delta)
        close = d2 < self.self_separation * self.self_separation
        if not np.any(close):
            return
        ci, cj = i[close], j[close]
        vec = delta[close]
        dist = np.sqrt(np.maximum(d2[close], 1.0e-20))
        normal = vec / dist[:, None]

        # Exact coincidence has no geometric normal.  A deterministic index-based
        # direction avoids both NaNs and a biased global ejection direction.
        coincident = dist < 1.0e-8
        if np.any(coincident):
            key = ci[coincident].astype(np.float64) * 12.9898 + cj[coincident] * 78.233
            phase = np.sin(key) * 43758.5453
            angle = 2.0 * math.pi * (phase - np.floor(phase))
            normal[coincident, 0] = np.cos(angle)
            normal[coincident, 1] = np.sin(angle)
            normal[coincident, 2] = 0.25
            normal[coincident] /= np.linalg.norm(normal[coincident], axis=1)[:, None]

        magnitude = 0.5 * np.maximum(self.self_separation - dist, 0.0)
        correction = normal * magnitude[:, None]
        accum = np.zeros_like(p)
        count = np.zeros(len(p), dtype=np.float64)
        np.add.at(accum, ci, correction)
        np.add.at(accum, cj, -correction)
        np.add.at(count, ci, 1.0)
        np.add.at(count, cj, 1.0)
        active = count > 0.0
        p[active] += 1.35 * accum[active] / count[active, None]

    def _project_wall(self, p: np.ndarray, primary: int) -> np.ndarray:
        """Project particles out of one tapered wall prism."""
        z = p[:, 2]
        if primary == 0:
            a, cross = p[:, 0], p[:, 1]
            slope, cross_slope = self.slope_x, self.slope_y
            inner0, outer0, cross_outer0 = 0.30, 0.38, 0.28
            axis, cross_axis = 0, 1
        else:
            a, cross = p[:, 1], p[:, 0]
            slope, cross_slope = self.slope_y, self.slope_x
            inner0, outer0, cross_outer0 = 0.20, 0.28, 0.38
            axis, cross_axis = 1, 0

        sa = np.where(a >= 0.0, 1.0, -1.0)
        sc = np.where(cross >= 0.0, 1.0, -1.0)
        norm = math.sqrt(1.0 + slope * slope)
        cross_norm = math.sqrt(1.0 + cross_slope * cross_slope)
        dz = z - self.funnel_z0
        inner = inner0 + slope * dz
        outer = outer0 + slope * dz
        cross_outer = cross_outer0 + cross_slope * dz

        g = np.column_stack((
            (np.abs(a) - inner) / norm,
            (outer - np.abs(a)) / norm,
            (cross_outer - np.abs(cross)) / cross_norm,
            z - self.funnel_z0,
            self.funnel_z1 - z,
        ))
        inside = np.all(g >= -self.particle_radius, axis=1)
        if not np.any(inside):
            return inside

        ids = np.flatnonzero(inside)
        face = np.argmin(g[inside] + self.particle_radius, axis=1)
        depth = g[ids, face] + self.particle_radius
        gradients = np.zeros((len(ids), 3), dtype=np.float64)
        for f in range(5):
            pick = face == f
            if not np.any(pick):
                continue
            q = ids[pick]
            if f == 0:
                gradients[pick, axis] = sa[q] / norm
                gradients[pick, 2] = -slope / norm
            elif f == 1:
                gradients[pick, axis] = -sa[q] / norm
                gradients[pick, 2] = slope / norm
            elif f == 2:
                gradients[pick, cross_axis] = -sc[q] / cross_norm
                gradients[pick, 2] = cross_slope / cross_norm
            elif f == 3:
                gradients[pick, 2] = 1.0
            else:
                gradients[pick, 2] = -1.0
        p[ids] -= gradients * depth[:, None]
        return inside

    def _project_rigid(self, p: np.ndarray) -> np.ndarray:
        contacted = np.zeros(len(p), dtype=bool)
        ground = p[:, 2] < self.particle_radius
        p[ground, 2] = self.particle_radius
        contacted |= ground
        contacted |= self._project_wall(p, 0)
        contacted |= self._project_wall(p, 1)
        return contacted

    def _step(self, dt: float) -> None:
        old = self.position.copy()
        self.velocity += self.gravity[None, :] * dt
        self.velocity *= math.exp(-0.18 * dt)
        p = self.position + self.velocity * dt

        contacted = np.zeros(len(p), dtype=bool)
        for _ in range(self.solver_iterations):
            for group in self.constraint_groups:
                self._solve_distances(p, group)
            self._solve_self_contact(p)
            self._solve_self_contact(p)
            contacted |= self._project_rigid(p)

        v = (p - old) / dt
        v[contacted] *= 0.994
        speed = np.linalg.norm(v, axis=1)
        too_fast = speed > 8.0
        if np.any(too_fast):
            v[too_fast] *= (8.0 / speed[too_fast])[:, None]
        self.position = p
        self.velocity = v
        self._last_contact_count = int(np.count_nonzero(contacted))
        self.steps += 1

    def _deformation_gradient(self) -> np.ndarray:
        q = self.position.reshape(self.ny, self.nx, 3)
        du = np.empty_like(q)
        dv = np.empty_like(q)
        du[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (2.0 * self.rest_dx)
        du[:, 0] = (q[:, 1] - q[:, 0]) / self.rest_dx
        du[:, -1] = (q[:, -1] - q[:, -2]) / self.rest_dx
        dv[1:-1] = (q[2:] - q[:-2]) / (2.0 * self.rest_dy)
        dv[0] = (q[1] - q[0]) / self.rest_dy
        dv[-1] = (q[-1] - q[-2]) / self.rest_dy
        a, b = du.reshape(-1, 3), dv.reshape(-1, 3)

        f00 = np.maximum(np.linalg.norm(a, axis=1), 1.0e-8)
        f01 = np.einsum("ij,ij->i", a, b) / f00
        f11 = np.sqrt(np.maximum(np.einsum("ij,ij->i", b, b) - f01 * f01, 1.0e-12))
        result = np.zeros((len(a), 2, 2), dtype=np.float64)
        result[:, 0, 0] = f00
        result[:, 0, 1] = f01
        result[:, 1, 1] = f11
        return result

    @staticmethod
    def _prism_signed_distance(p: np.ndarray, primary: int,
                               z0: float, z1: float,
                               slope: float, cross_slope: float,
                               inner0: float, outer0: float,
                               cross_outer0: float) -> np.ndarray:
        a = np.abs(p[:, primary])
        cross = np.abs(p[:, 1 - primary])
        dz = p[:, 2] - z0
        n = math.sqrt(1.0 + slope * slope)
        cn = math.sqrt(1.0 + cross_slope * cross_slope)
        g = np.column_stack((
            (a - (inner0 + slope * dz)) / n,
            ((outer0 + slope * dz) - a) / n,
            ((cross_outer0 + cross_slope * dz) - cross) / cn,
            p[:, 2] - z0,
            z1 - p[:, 2],
        ))
        inside = np.all(g >= 0.0, axis=1)
        sd = np.sqrt(np.sum(np.minimum(g, 0.0) ** 2, axis=1))
        sd[inside] = -np.min(g[inside], axis=1)
        return sd

    def _contact_signed_distance(self) -> np.ndarray:
        p = self.position
        sx = self._prism_signed_distance(
            p, 0, self.funnel_z0, self.funnel_z1,
            self.slope_x, self.slope_y, 0.30, 0.38, 0.28)
        sy = self._prism_signed_distance(
            p, 1, self.funnel_z0, self.funnel_z1,
            self.slope_y, self.slope_x, 0.20, 0.28, 0.38)
        gap = np.minimum.reduce((p[:, 2], sx, sy)) - self.particle_radius

        nearest = np.full(len(p), np.inf, dtype=np.float64)
        delta = p[self.self_i] - p[self.self_j]
        distances = np.linalg.norm(delta, axis=1) - self.self_separation
        np.minimum.at(nearest, self.self_i, distances)
        np.minimum.at(nearest, self.self_j, distances)
        gap = np.minimum(gap, nearest)
        return np.nan_to_num(gap, nan=0.0, posinf=1.0, neginf=-self.thickness)

    def _frame(self, time: float) -> StateFrame:
        boundary = np.zeros(len(self.position), dtype=np.int32)
        q = boundary.reshape(self.ny, self.nx)
        q[0, :] = q[-1, :] = 1
        q[:, 0] = q[:, -1] = 1
        frame = StateFrame(
            float(time), self.position.copy(), self.velocity.copy(),
            mass=self.mass.copy(), body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": boundary,
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self._contact_signed_distance(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": self.external_force.copy(),
            },
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires finite, nondecreasing time")
        while self.time + 1.0e-12 < target:
            dt = min(self.max_dt, target - self.time)
            self._step(dt)
            self.time += dt
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "semi_implicit_euler_pbd_cloth",
            "steps": self.steps,
            "time": self.time,
            "last_rigid_contact_vertices": self._last_contact_count,
            "max_speed": float(np.max(np.linalg.norm(self.velocity, axis=1))),
        }
