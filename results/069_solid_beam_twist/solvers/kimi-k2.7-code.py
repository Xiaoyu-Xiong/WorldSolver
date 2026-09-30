"""Explicit corotated-style tetrahedral FEM solver for the twisted solid beam.

The beam is treated as a three-dimensional elastic solid.  A regular-grid
tetrahedral mesh is built from the rest positions.  Internal forces are
computed with a Saint Venant-Kirchhoff hyperelastic model and the end faces
are driven by kinematic boundary conditions that rotate them in opposite
directions about the beam axis.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic explicit-dynamics elastic solid solver."""

    def __init__(self) -> None:
        self.scene = None
        self.config: dict = {}

        # state
        self.t = 0.0
        self.x: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.mass: np.ndarray | None = None
        self.rest: np.ndarray | None = None
        self.end_region: np.ndarray | None = None
        self.n = 0

        # mesh
        self.elems: np.ndarray | None = None
        self.Dm_inv: np.ndarray | None = None
        self.vol0: np.ndarray | None = None
        self.h_min = 0.0

        # material / time step
        self.E = 5.0e5
        self.nu = 0.35
        self.mu = 0.0
        self.lam = 0.0
        self.damping = 2.0
        self.dt = 2.0e-4
        self.c_wave = 0.0

        # boundary
        self.left_center = np.zeros(3)
        self.right_center = np.zeros(3)
        self.left_mask = np.zeros(0, dtype=bool)
        self.right_mask = np.zeros(0, dtype=bool)
        self.rest_local_left = np.zeros((0, 3))
        self.rest_local_right = np.zeros((0, 3))

        # budget bookkeeping
        self.cum_impulse = np.zeros((2, 3))
        self.sample_impulse = np.zeros((2, 3))
        self.last_sample_dt = 0.0
        self.ref_momentum = 1.0

    # ------------------------------------------------------------------
    # Solver protocol
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = getattr(scene, "config", {})

        frame0 = scene.initial_state()
        self.t = float(frame0.time)
        self.x = np.asarray(frame0.position, dtype=float).copy()
        self.v = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.rest = np.asarray(frame0.attributes["rest_position"], dtype=float).copy()
        self.end_region = np.asarray(frame0.attributes["end_region"], dtype=float).copy()
        self.n = self.x.shape[0]

        # material parameters
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        density = float(self.config.get("beam", {}).get("density", 1000.0))
        self.c_wave = math.sqrt((self.lam + 2.0 * self.mu) / density)

        # build tetrahedral mesh
        self._build_mesh()

        # boundary masks and helpers
        self._init_boundary_data()

        # stable explicit time step
        self.dt = self._choose_time_step()

        # reference momentum scale
        self.ref_momentum = self._reference_momentum()

        # budget state
        self.cum_impulse = np.zeros((2, 3))
        self.sample_impulse = np.zeros((2, 3))
        self.last_sample_dt = 0.0

    def advance_to(self, time: float) -> StateFrame:
        if time < self.t - 1e-12:
            raise ValueError(f"Cannot advance backwards: {self.t} -> {time}")

        if time > self.t:
            self.sample_impulse = np.zeros((2, 3))
            self.last_sample_dt = time - self.t
            nsteps = max(1, int(round((time - self.t) / self.dt)))
            dt_eff = (time - self.t) / nsteps
            for _ in range(nsteps):
                self._step(dt_eff)

        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "elements": self.elems.shape[0] if self.elems is not None else 0,
            "nodes": self.n,
            "dt": self.dt,
            "E": self.E,
            "nu": self.nu,
            "damping": self.damping,
        }

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    def _build_mesh(self) -> None:
        """Build a tetrahedral mesh, preferring the provided topology asset."""
        asset_path = Path("assets/beam_volume_topology.npz")
        if asset_path.is_file():
            try:
                self._load_asset_topology(str(asset_path))
                if self.elems is not None and self.elems.shape[0] > 0:
                    return
            except Exception:
                pass
        self._generate_grid_tets()

    def _load_asset_topology(self, path: str) -> None:
        data = np.load(path)
        for key in ("tets", "elements", "cells", "tetrahedra"):
            if key in data:
                elems = np.asarray(data[key], dtype=int)
                if elems.ndim == 2 and elems.shape[1] == 4 and elems.max() < self.n:
                    self.elems = elems
                    self._compute_tet_reference_data()
                    return

    def _generate_grid_tets(self) -> None:
        beam = self.config.get("beam", {})
        shape = beam.get("volumetric_grid_shape", [self.n, 1, 1])
        nx, ny, nz = (int(v) for v in shape)

        i = np.arange(nx - 1, dtype=int)
        j = np.arange(ny - 1, dtype=int)
        k = np.arange(nz - 1, dtype=int)
        I, J, K = np.meshgrid(i, j, k, indexing="ij")
        # scene ravel order: index = i + nx*j + nx*ny*k
        base = I + nx * J + nx * ny * K

        # 8 corner offsets of a voxel
        offsets = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ]
        corner = []
        for di, dj, dk in offsets:
            corner.append((base + di + nx * dj + nx * ny * dk).ravel())
        corner = np.stack(corner, axis=1)  # [Nv, 8]

        # Kuhn triangulation around the main diagonal 0-7
        tets = np.array(
            [
                [0, 1, 4, 7],
                [0, 1, 5, 7],
                [0, 2, 4, 7],
                [0, 2, 6, 7],
                [0, 3, 5, 7],
                [0, 3, 6, 7],
            ],
            dtype=int,
        )
        self.elems = corner[:, tets].reshape(-1, 4)
        self._compute_tet_reference_data()

    def _compute_tet_reference_data(self) -> None:
        x0 = self.rest[self.elems]
        D = x0[:, 1:3, :].copy()
        D[:, 0, :] = x0[:, 1, :] - x0[:, 0, :]
        D[:, 1, :] = x0[:, 2, :] - x0[:, 0, :]
        D[:, 2, :] = x0[:, 3, :] - x0[:, 0, :]

        det = np.linalg.det(D)
        neg = det < 0.0
        if np.any(neg):
            self.elems = self.elems.copy()
            self.elems[neg, [0, 1]] = self.elems[neg, [1, 0]]
            x0 = self.rest[self.elems]
            D[:, 0, :] = x0[:, 1, :] - x0[:, 0, :]
            D[:, 1, :] = x0[:, 2, :] - x0[:, 0, :]
            D[:, 2, :] = x0[:, 3, :] - x0[:, 0, :]
            det = np.linalg.det(D)

        self.Dm_inv = np.linalg.inv(D)
        self.vol0 = det / 6.0

        # characteristic edge length for stability estimate
        diffs = []
        for a, b in ((0, 1), (0, 2), (0, 3)):
            diffs.append(self.rest[self.elems[:, a]] - self.rest[self.elems[:, b]])
        self.h_min = float(np.min(np.linalg.norm(np.stack(diffs, axis=0), axis=2)))

    # ------------------------------------------------------------------
    # Boundary
    # ------------------------------------------------------------------
    def _init_boundary_data(self) -> None:
        bc = self.config.get("boundary_conditions", {})
        self.left_center = np.asarray(bc["left_end"]["center"], dtype=float)
        self.right_center = np.asarray(bc["right_end"]["center"], dtype=float)
        self.left_mask = self.end_region < -0.5
        self.right_mask = self.end_region > 0.5
        self.rest_local_left = self.rest - self.left_center
        self.rest_local_right = self.rest - self.right_center

    def _target(self, t: float):
        duration = float(self.config.get("duration", 10.0))
        u = t / duration
        if u <= 0.0:
            s = 0.0
            dsdt = 0.0
        elif u >= 1.0:
            s = 1.0
            dsdt = 0.0
        else:
            s = 3.0 * u * u - 2.0 * u * u * u
            dsdt = (6.0 * u * (1.0 - u)) / duration

        angle_left = -math.pi * s
        angle_right = math.pi * s
        omega_left = -math.pi * dsdt
        omega_right = math.pi * dsdt

        target_pos = np.zeros_like(self.rest)
        target_vel = np.zeros_like(self.rest)

        self._apply_end_target(
            target_pos,
            target_vel,
            self.left_mask,
            self.left_center,
            self.rest_local_left,
            angle_left,
            omega_left,
        )
        self._apply_end_target(
            target_pos,
            target_vel,
            self.right_mask,
            self.right_center,
            self.rest_local_right,
            angle_right,
            omega_right,
        )
        return target_pos, target_vel

    @staticmethod
    def _apply_end_target(
        pos: np.ndarray,
        vel: np.ndarray,
        mask: np.ndarray,
        center: np.ndarray,
        rest_local: np.ndarray,
        angle: float,
        omega: float,
    ) -> None:
        if not np.any(mask):
            return
        rl = rest_local[mask]
        c = math.cos(angle)
        s = math.sin(angle)
        tl = rl.copy()
        tl[:, 1] = c * rl[:, 1] - s * rl[:, 2]
        tl[:, 2] = s * rl[:, 1] + c * rl[:, 2]
        pos[mask] = center + tl
        vel[mask] = 0.0
        vel[mask, 1] = -omega * tl[:, 2]
        vel[mask, 2] = omega * tl[:, 1]

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    def _choose_time_step(self) -> float:
        # aim for a small fraction of the acoustic CFL limit
        target = 0.25 * self.h_min / max(self.c_wave, 1.0)
        return min(2.0e-4, target)

    def _step(self, dt: float) -> None:
        f_el = self._elastic_force()
        f = f_el - self.damping * self.v

        t_next = self.t + dt
        target_pos, target_vel = self._target(t_next)

        v_free = self.v + dt * f / self.mass[:, None]

        # impulse required to enforce the kinematic end constraints
        impulse = np.zeros((2, 3))
        for idx, mask in enumerate((self.left_mask, self.right_mask)):
            m_end = self.mass[mask][:, None]
            v_f = v_free[mask]
            v_t = target_vel[mask]
            impulse[idx] = (m_end * (v_t - v_f)).sum(axis=0)

        self.sample_impulse += impulse
        self.cum_impulse += impulse

        # apply constraints
        v_free[self.left_mask] = target_vel[self.left_mask]
        v_free[self.right_mask] = target_vel[self.right_mask]
        self.v = v_free

        self.x = self.x + dt * self.v
        self.x[self.left_mask] = target_pos[self.left_mask]
        self.x[self.right_mask] = target_pos[self.right_mask]
        self.t = t_next

    def _elastic_force(self) -> np.ndarray:
        x_el = self.x[self.elems]  # [Ne, 4, 3]
        D = np.empty((self.elems.shape[0], 3, 3), dtype=float)
        D[:, 0, :] = x_el[:, 1, :] - x_el[:, 0, :]
        D[:, 1, :] = x_el[:, 2, :] - x_el[:, 0, :]
        D[:, 2, :] = x_el[:, 3, :] - x_el[:, 0, :]

        F = D @ self.Dm_inv

        # Saint Venant-Kirchhoff hyperelastic model
        C = np.einsum("nki,nkj->nij", F, F)
        E = 0.5 * (C - np.eye(3, dtype=float)[None, :, :])
        trace_E = np.trace(E, axis1=1, axis2=2)
        S = 2.0 * self.mu * E + self.lam * trace_E[:, None, None] * np.eye(3, dtype=float)[None, :, :]
        P = np.einsum("nij,njk->nik", F, S)

        H = self.vol0[:, None, None] * np.einsum("nij,nkj->nik", P, self.Dm_inv)

        f_local = np.empty((self.elems.shape[0], 4, 3), dtype=float)
        f_local[:, 1, :] = -H[:, 0, :]
        f_local[:, 2, :] = -H[:, 1, :]
        f_local[:, 3, :] = -H[:, 2, :]
        f_local[:, 0, :] = H.sum(axis=1)

        flat_idx = self.elems.ravel()
        flat_force = f_local.reshape(-1, 3)
        f = np.zeros((self.n, 3), dtype=float)
        for d in range(3):
            f[:, d] = np.bincount(flat_idx, weights=flat_force[:, d], minlength=self.n)
        return f

    def _deformation_gradient(self) -> np.ndarray:
        x_el = self.x[self.elems]
        D = np.empty((self.elems.shape[0], 3, 3), dtype=float)
        D[:, 0, :] = x_el[:, 1, :] - x_el[:, 0, :]
        D[:, 1, :] = x_el[:, 2, :] - x_el[:, 0, :]
        D[:, 2, :] = x_el[:, 3, :] - x_el[:, 0, :]
        F = D @ self.Dm_inv
        return F

    # ------------------------------------------------------------------
    # Frame export
    # ------------------------------------------------------------------
    def _reference_momentum(self) -> float:
        beam = self.config.get("beam", {})
        dims = beam.get("dimensions", [4.8, 0.55, 0.55])
        half_width = float(dims[1]) / 2.0
        half_height = float(dims[2]) / 2.0
        r_max = math.sqrt(half_width * half_width + half_height * half_height)
        duration = float(self.config.get("duration", 10.0))
        # maximum angular rate from smoothstep ramp (final angle = pi)
        omega_max = math.pi * 1.5 / duration
        v_max = r_max * omega_max
        return max(float(self.mass.sum() * v_max), 1.0)

    def _smoothstep_scalar(self, t: float) -> float:
        duration = float(self.config.get("duration", 10.0))
        u = t / duration
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        return 3.0 * u * u - 2.0 * u * u * u

    def _make_frame(self) -> StateFrame:
        F = self._deformation_gradient()

        if self.last_sample_dt > 0.0:
            ext = self.sample_impulse / self.last_sample_dt
        else:
            ext = np.zeros((2, 3))
        total_cum = self.cum_impulse.sum(axis=0)

        rel_deg = 360.0 * self._smoothstep_scalar(self.t)

        return StateFrame(
            time=float(self.t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "end_region": self.end_region.copy(),
                "rest_position": self.rest.copy(),
            },
            fields={
                "element_deformation_gradient": F,
                "budget_cumulative_external_impulse": total_cum.reshape(1, 3),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_domain_weights": np.ones((1, self.n), dtype=float),
                "budget_external_force": ext.reshape(1, 2, 3),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "reference_momentum": self.ref_momentum,
                "relative_end_rotation_degrees": float(rel_deg),
            },
        )
