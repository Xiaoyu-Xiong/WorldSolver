"""Deterministic, dependency-light solver for the pinned denim scene.

The cloth is advanced with symplectic Euler followed by a position based
projection of warp, weft, bias, and long-range bending constraints. This is
actual time stepping, rather than a prescribed animation.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Position-based thin-sheet solver with fixed upper corners."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.nx, self.ny = map(int, cloth["grid_shape"])
        self.n = self.nx * self.ny
        self.thickness = float(cloth["thickness"])
        self.contact_scale = self.thickness
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)

        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=np.float64).copy()
        self.v = np.zeros_like(self.x)
        self.mass = np.full(
            self.n, float(cloth["total_mass"]) / self.n, dtype=np.float64
        )
        self.inv_mass = 1.0 / self.mass

        # Fixed targets come directly from the frozen initial state.
        self.pinned = np.array(
            [(self.ny - 1) * self.nx, self.ny * self.nx - 1], dtype=np.int64
        )
        self.pin_position = self.x[self.pinned].copy()
        self.inv_mass[self.pinned] = 0.0

        sx, sy = map(float, cloth["size"])
        self.dx = sx / (self.nx - 1)
        self.dy = sy / (self.ny - 1)
        self.time = 0.0
        self.step_count = 0
        self.base_dt = min(1.0 / 240.0, float(cfg["sample_dt"]) / 4.0)

        # Cross-grain ease makes the corner-pinned sheet buckle into folds
        # instead of remaining in a symmetry-locked perfectly planar state.
        self.cross_ease = 1.025
        self._build_constraints()

        uu = np.linspace(0.0, 1.0, self.nx)[None, :]
        vv = np.linspace(0.0, 1.0, self.ny)[:, None]
        self.perturbation = (
            np.sin(np.pi * vv)
            * np.sin(np.pi * uu)
            * (
                0.70 * np.sin(4.0 * np.pi * uu + 0.31)
                + 0.30 * np.sin(7.0 * np.pi * uu - 0.17)
            )
        ).ravel()
        self.perturbation[self.pinned] = 0.0

        self.body_id = np.zeros(self.n, dtype=np.int32)
        self.boundary = self._boundary_flags()
        self._last_external_force = self.mass[:, None] * self.gravity[None, :]

    def _boundary_flags(self) -> np.ndarray:
        out = np.zeros((self.ny, self.nx), dtype=np.int32)
        out[0, :] = out[-1, :] = 1
        out[:, 0] = out[:, -1] = 1
        return out.ravel()

    def _build_constraints(self) -> None:
        """Create colored edge sets; edges in one set share no vertices."""
        grid = np.arange(self.n, dtype=np.int64).reshape(self.ny, self.nx)
        groups = []

        def add_colored(edges, lengths, colors, stiffness):
            edges = np.asarray(edges, dtype=np.int64)
            lengths = np.asarray(lengths, dtype=np.float64)
            colors = np.asarray(colors, dtype=np.int32)
            for color in np.unique(colors):
                use = colors == color
                groups.append(
                    (edges[use, 0], edges[use, 1], lengths[use], stiffness)
                )

        # Warp and weft constraints.
        edges, colors = [], []
        for j in range(self.ny):
            for i in range(self.nx - 1):
                edges.append((grid[j, i], grid[j, i + 1]))
                colors.append(i & 1)
        add_colored(
            edges,
            np.full(len(edges), self.dx * self.cross_ease),
            colors,
            0.92,
        )

        edges, colors = [], []
        for j in range(self.ny - 1):
            for i in range(self.nx):
                edges.append((grid[j, i], grid[j + 1, i]))
                colors.append(j & 1)
        add_colored(edges, np.full(len(edges), self.dy), colors, 0.94)

        # Bias constraints suppress shear while still allowing folds.
        diag_len = math.hypot(self.dx * self.cross_ease, self.dy)
        for direction in (1, -1):
            edges, colors = [], []
            for j in range(self.ny - 1):
                for i in range(self.nx - 1):
                    if direction == 1:
                        pair = (grid[j, i], grid[j + 1, i + 1])
                    else:
                        pair = (grid[j, i + 1], grid[j + 1, i])
                    edges.append(pair)
                    colors.append((i & 1) + 2 * (j & 1))
            add_colored(
                edges, np.full(len(edges), diag_len), colors, 0.38
            )

        # Weak second-neighbour constraints provide discrete bending resistance.
        edges, colors = [], []
        for j in range(self.ny):
            for i in range(self.nx - 2):
                edges.append((grid[j, i], grid[j, i + 2]))
                colors.append(i & 3)
        add_colored(
            edges,
            np.full(len(edges), 2.0 * self.dx * self.cross_ease),
            colors,
            0.055,
        )

        edges, colors = [], []
        for j in range(self.ny - 2):
            for i in range(self.nx):
                edges.append((grid[j, i], grid[j + 2, i]))
                colors.append(j & 3)
        add_colored(
            edges, np.full(len(edges), 2.0 * self.dy), colors, 0.055
        )
        self.constraint_groups = groups

    def _project_group(self, ia, ib, rest, stiffness) -> None:
        d = self.x[ib] - self.x[ia]
        length = np.linalg.norm(d, axis=1)
        safe = np.maximum(length, 1.0e-12)
        wi = self.inv_mass[ia]
        wj = self.inv_mass[ib]
        denom = wi + wj
        active = denom > 0.0
        amount = np.zeros_like(length)
        amount[active] = (
            stiffness
            * (length[active] - rest[active])
            / (safe[active] * denom[active])
        )
        correction = amount[:, None] * d
        self.x[ia] += wi[:, None] * correction
        self.x[ib] -= wj[:, None] * correction

    def _enforce_boundaries(self, previous: np.ndarray) -> None:
        self.x[self.pinned] = self.pin_position

        # Ground box top is z=0; collide the lower cloth surface.
        floor = 0.5 * self.thickness
        hit = self.x[:, 2] < floor
        if np.any(hit):
            self.x[hit, 2] = floor
            self.x[hit, :2] = previous[hit, :2] + 0.72 * (
                self.x[hit, :2] - previous[hit, :2]
            )

    def _step(self, dt: float) -> None:
        previous = self.x.copy()

        # Gravity and a tiny decaying transverse load that selects one branch
        # of the otherwise perfectly symmetric physical buckling mode.
        lateral_accel = (
            0.055 * math.exp(-self.time / 0.22) * self.perturbation
        )
        acceleration = np.repeat(self.gravity[None, :], self.n, axis=0)
        acceleration[:, 1] += lateral_accel
        acceleration[self.pinned] = 0.0
        self._last_external_force = self.mass[:, None] * acceleration

        # Semi-implicit Euler with viscous air/internal damping.
        self.v += dt * acceleration
        self.v *= math.exp(-1.35 * dt)
        self.v[self.pinned] = 0.0
        self.x += dt * self.v

        for _ in range(9):
            for group in self.constraint_groups:
                self._project_group(*group)
            self._enforce_boundaries(previous)

        self.v = (self.x - previous) / dt
        self.v[self.pinned] = 0.0
        contact = self.x[:, 2] <= 0.5 * self.thickness + 1.0e-10
        if np.any(contact):
            self.v[contact, 2] = np.maximum(self.v[contact, 2], 0.0)
            self.v[contact, :2] *= 0.72

        if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()):
            raise FloatingPointError("non-finite cloth state")
        self.time += dt
        self.step_count += 1

    def _deformation_gradient(self) -> np.ndarray:
        """Return per-vertex material-plane right-stretch tensors."""
        p = self.x.reshape(self.ny, self.nx, 3)
        du = np.empty_like(p)
        dv = np.empty_like(p)
        du[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2.0 * self.dx)
        du[:, 0] = (p[:, 1] - p[:, 0]) / self.dx
        du[:, -1] = (p[:, -1] - p[:, -2]) / self.dx
        dv[1:-1] = (p[2:] - p[:-2]) / (2.0 * self.dy)
        dv[0] = (p[1] - p[0]) / self.dy
        dv[-1] = (p[-1] - p[-2]) / self.dy

        # Positive square root of J^T J, objective for an embedded surface.
        g00 = np.einsum("ijk,ijk->ij", du, du)
        g01 = np.einsum("ijk,ijk->ij", du, dv)
        g11 = np.einsum("ijk,ijk->ij", dv, dv)
        determinant = np.maximum(g00 * g11 - g01 * g01, 0.0)
        root_det = np.sqrt(determinant)
        denominator = np.sqrt(
            np.maximum(g00 + g11 + 2.0 * root_det, 1.0e-18)
        )
        deform = np.empty((self.ny, self.nx, 2, 2), dtype=np.float64)
        deform[:, :, 0, 0] = (g00 + root_det) / denominator
        deform[:, :, 0, 1] = g01 / denominator
        deform[:, :, 1, 0] = g01 / denominator
        deform[:, :, 1, 1] = (g11 + root_det) / denominator
        return deform.reshape(self.n, 2, 2)

    def _frame(self, requested_time: float) -> StateFrame:
        external = self._last_external_force.copy()
        if requested_time == 0.0:
            acceleration = np.repeat(self.gravity[None, :], self.n, axis=0)
            acceleration[:, 1] += 0.055 * self.perturbation
            acceleration[self.pinned] = 0.0
            external = self.mass[:, None] * acceleration

        frame = StateFrame(
            float(requested_time),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self.x[:, 2].copy()
                - 0.5 * self.thickness,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": external,
            },
            scalars={"contact_length_scale": self.contact_scale},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1.0e-10:
            raise ValueError("advance_to requires nondecreasing sample times")
        while self.time < target - 1.0e-12:
            self._step(min(self.base_dt, target - self.time))
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position_based_inextensible_cloth",
            "internal_steps": int(self.step_count),
            "pinned_vertices": self.pinned.tolist(),
            "finite": bool(
                np.isfinite(self.x).all() and np.isfinite(self.v).all()
            ),
        }
