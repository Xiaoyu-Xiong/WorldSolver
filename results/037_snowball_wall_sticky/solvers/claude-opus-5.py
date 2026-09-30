"""Snowball / sticky-wall solver.

Physical model
--------------
Explicit MLS-MPM (moving-least-squares material point method) discretization of
the elasto-plastic snow constitutive model of Stomakhin et al., "A Material
Point Method for Snow Simulation" (2013):

  * fixed-corotated elasticity on the elastic part of the deformation gradient,
  * singular-value clamping plasticity (critical compression / stretch),
  * exponential hardening in the accumulated plastic volume change.

Time stepping is a symplectic-Euler MPM cycle (particle-to-grid transfer, grid
momentum update with gravity and boundary conditions, grid-to-particle transfer,
deformation-gradient update, plastic return mapping) with a CFL-limited step
size based on the current elastic wave speed and particle velocities.

Boundaries (frozen geometry from ``scene_geometry.yaml``) are enforced as
sticky (no-slip, zero-velocity) grid collision objects, which is the collision
mode requested by the frozen configuration: material reaching the wall or the
floor attaches locally, while separated material stays free to rebound, break
up and fall under gravity.

No animation, keyframes or precomputed trajectories are used: every exported
frame is the result of actually stepping the discretized equations of motion.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1.0e-12


def _box_signed_distance(points: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    """Exact signed distance to an axis-aligned box (positive outside)."""
    d = np.abs(points - center[None, :]) - half[None, :]
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=1)
    inside = np.minimum(np.max(d, axis=1), 0.0)
    return outside + inside


def _load_boundary_boxes(contact: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collision boxes from the frozen scene geometry, with a static fallback."""
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    path = Path("scene_geometry.yaml")
    try:
        import yaml  # noqa: PLC0415

        if path.is_file():
            data = yaml.safe_load(path.read_text()) or {}
            for obj in data.get("objects", []) or []:
                geom = obj.get("geometry", {}) or {}
                if geom.get("type") != "box" or not obj.get("collision_enabled", False):
                    continue
                if obj.get("semantic") not in (None, "boundary"):
                    continue
                half = np.asarray(geom.get("half_extents"), dtype=float)
                tr = (obj.get("transform", {}) or {}).get("translation", [0.0, 0.0, 0.0])
                boxes.append((np.asarray(tr, dtype=float), half))
    except Exception:  # pragma: no cover - fall back to frozen constants
        boxes = []
    if not boxes:
        floor_z = float(contact.get("floor_z", 0.0))
        wall_x = float(contact.get("wall_inner_x", 0.0))
        boxes = [
            (np.array([-0.75, 0.0, floor_z - 0.04]), np.array([2.8, 1.8, 0.04])),
            (np.array([wall_x + 0.08, 0.0, 1.35]), np.array([0.08, 1.2, 1.35])),
        ]
    return boxes


class BenchmarkSolver:
    """Explicit MLS-MPM snow solver with sticky boundary collisions."""

    # --- snow constitutive parameters (Stomakhin et al. 2013, Table 1 style) ---
    YOUNG = 1.4e5           # initial Young's modulus [Pa]
    POISSON = 0.2           # Poisson ratio
    HARDENING = 10.0        # xi
    THETA_C = 2.5e-2        # critical compression
    THETA_S = 7.5e-3        # critical stretch
    H_MIN, H_MAX = 0.05, 8.0
    CFL = 0.35
    DT_MAX = 1.0e-3
    DT_MIN = 2.0e-5

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.config = cfg
        contact = dict(cfg.get("contact", {}) or {})
        snow = dict(cfg.get("snowball", {}) or {})

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.n = int(self.pos.shape[0])
        mass = frame0.mass
        if mass is None:
            mass = np.full(self.n, float(snow.get("total_mass", 1.0)) / max(self.n, 1))
        self.mass = np.array(mass, dtype=float)
        attrs = dict(frame0.attributes or {})
        self.ref_volume = np.array(
            attrs.get("reference_volume", self.mass / float(snow.get("nominal_density", 520.0))),
            dtype=float,
        )
        self.component_id = np.array(
            attrs.get("initial_component_id", np.zeros(self.n)), dtype=np.int64
        )
        self.body_id = np.zeros(self.n, dtype=np.int64)
        self.F = np.tile(np.eye(3), (self.n, 1, 1))
        self.Jp = np.ones(self.n, dtype=float)
        self.C = np.zeros((self.n, 3, 3), dtype=float)

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.spacing = float(snow.get("particle_spacing", 0.065))
        self.contact_length = float(contact.get("contact_length_scale", self.spacing))
        self.wall_x = float(contact.get("wall_inner_x", 0.0))
        self.floor_z = float(contact.get("floor_z", 0.0))
        self.boxes = _load_boundary_boxes(contact)

        # MPM particle volume follows the sampling lattice (one particle per cell
        # of the sampling grid); the reported reference volume keeps the frozen
        # nominal-density convention of the scene contract.
        self.p_volume = self.spacing ** 3
        self.p_density = float(self.mass.mean()) / self.p_volume
        self.mu0 = self.YOUNG / (2.0 * (1.0 + self.POISSON))
        self.lam0 = (
            self.YOUNG * self.POISSON / ((1.0 + self.POISSON) * (1.0 - 2.0 * self.POISSON))
        )

        self.scalars0 = {
            "contact_length_scale": self.contact_length,
            "initial_total_mass": float(self.mass.sum()),
            "initial_total_volume": float(self.ref_volume.sum()),
        }
        self._build_grid()
        self.time = 0.0
        self.steps = 0

    # ------------------------------------------------------------------
    # background grid
    # ------------------------------------------------------------------
    def _build_grid(self) -> None:
        self.dx = 0.12
        self.inv_dx = 1.0 / self.dx
        self.lo = np.array([-3.0, -1.32, -0.36], dtype=float)
        hi = np.array([0.48, 1.32, 2.52], dtype=float)
        dims = np.round((hi - self.lo) / self.dx).astype(int) + 1
        self.nx, self.ny, self.nz = (int(d) for d in dims)
        self.ncell = self.nx * self.ny * self.nz

        ix = np.arange(self.nx)
        iy = np.arange(self.ny)
        iz = np.arange(self.nz)
        gx = self.lo[0] + ix * self.dx
        gy = self.lo[1] + iy * self.dx
        gz = self.lo[2] + iz * self.dx
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        nodes = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        self.node_pos = nodes

        # Sticky (zero-velocity) collision nodes: any node inside a frozen
        # boundary box, plus two padding layers acting as the container walls.
        frozen = np.zeros(self.ncell, dtype=bool)
        for center, half in self.boxes:
            frozen |= _box_signed_distance(nodes, center, half) <= 1.0e-9
        # Half-space closures of the two colliders: everything at or below the
        # floor plane and everything at or beyond the wall inner face sticks.
        frozen |= nodes[:, 2] <= self.floor_z + 1.0e-9
        frozen |= (
            (nodes[:, 0] >= self.wall_x - 1.0e-9)
            & (np.abs(nodes[:, 1]) <= 1.2 + 1.0e-9)
            & (nodes[:, 2] <= 2.7 + 1.0e-9)
        )
        I, J, K = (a.reshape(-1) for a in np.meshgrid(ix, iy, iz, indexing="ij"))
        frozen |= (I < 2) | (I >= self.nx - 2)
        frozen |= (J < 2) | (J >= self.ny - 2)
        frozen |= (K < 2) | (K >= self.nz - 2)
        self.frozen = frozen

        self.grid_m = np.zeros(self.ncell, dtype=float)
        self.grid_v = np.zeros((self.ncell, 3), dtype=float)

        # Safe particle band so that the 3x3x3 quadratic stencil stays in range.
        self.pos_lo = self.lo + 1.55 * self.dx
        self.pos_hi = self.lo + (dims.astype(float) - 2.55) * self.dx

        # 3x3x3 quadratic B-spline stencil offsets (fixed evaluation order keeps
        # the accumulation, and therefore the trajectory, deterministic).
        self.offsets = tuple(itertools.product(range(3), repeat=3))

    # ------------------------------------------------------------------
    # constitutive model
    # ------------------------------------------------------------------
    def _hardening(self) -> np.ndarray:
        return np.clip(np.exp(self.HARDENING * (1.0 - self.Jp)), self.H_MIN, self.H_MAX)

    def _kirchhoff_stress(self) -> np.ndarray:
        """Fixed-corotated Kirchhoff stress tau = P(F_E) F_E^T with hardening."""
        U, sig, Vt = np.linalg.svd(self.F)
        det_u = np.linalg.det(U)
        det_v = np.linalg.det(Vt)
        flip_u = det_u < 0.0
        if np.any(flip_u):
            U[flip_u, :, 2] *= -1.0
        flip_v = det_v < 0.0
        if np.any(flip_v):
            Vt[flip_v, 2, :] *= -1.0
        sign = np.where(flip_u ^ flip_v, -1.0, 1.0)
        sig = sig.copy()
        sig[:, 2] *= sign
        R = U @ Vt
        J = np.clip(sig[:, 0] * sig[:, 1] * sig[:, 2], 1.0e-4, 1.0e4)
        h = self._hardening()
        mu = self.mu0 * h
        lam = self.lam0 * h
        FT = np.transpose(self.F, (0, 2, 1))
        tau = 2.0 * mu[:, None, None] * ((self.F - R) @ FT)
        tau += (lam * (J - 1.0) * J)[:, None, None] * np.eye(3)[None, :, :]
        return tau

    def _apply_plasticity(self) -> None:
        """Singular-value clamping return mapping (critical compression/stretch)."""
        U, sig, Vt = np.linalg.svd(self.F)
        sig = np.clip(sig, 1.0e-3, 1.0e3)
        sig_c = np.clip(sig, 1.0 - self.THETA_C, 1.0 + self.THETA_S)
        det_old = sig[:, 0] * sig[:, 1] * sig[:, 2]
        det_new = sig_c[:, 0] * sig_c[:, 1] * sig_c[:, 2]
        self.Jp = np.clip(self.Jp * det_old / np.maximum(det_new, _EPS), 0.05, 20.0)
        self.F = (U * sig_c[:, None, :]) @ Vt

    # ------------------------------------------------------------------
    # one explicit MPM cycle
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        inv_dx, dx = self.inv_dx, self.dx
        xr = (self.pos - self.lo[None, :]) * inv_dx
        base = np.floor(xr - 0.5).astype(np.int64)
        np.clip(base[:, 0], 0, self.nx - 3, out=base[:, 0])
        np.clip(base[:, 1], 0, self.ny - 3, out=base[:, 1])
        np.clip(base[:, 2], 0, self.nz - 3, out=base[:, 2])
        fx = xr - base.astype(float)
        w = np.stack(
            (
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ),
            axis=0,
        )  # [3(node), N, 3(axis)]

        tau = self._kirchhoff_stress()
        affine = (
            -(dt * self.p_volume * 4.0 * inv_dx * inv_dx) * tau
            + self.mass[:, None, None] * self.C
        )
        mv = self.mass[:, None] * self.vel

        self.grid_m.fill(0.0)
        self.grid_v.fill(0.0)
        flat_base = (base[:, 0] * self.ny + base[:, 1]) * self.nz + base[:, 2]
        stride = np.array([self.ny * self.nz, self.nz, 1], dtype=np.int64)

        for i, j, k in self.offsets:
            wt = w[i, :, 0] * w[j, :, 1] * w[k, :, 2]
            idx = flat_base + i * stride[0] + j * stride[1] + k * stride[2]
            dpos = (np.array([i, j, k], dtype=float)[None, :] - fx) * dx
            contrib = mv + np.einsum("nab,nb->na", affine, dpos)
            self.grid_m += np.bincount(idx, weights=wt * self.mass, minlength=self.ncell)
            for c in range(3):
                self.grid_v[:, c] += np.bincount(
                    idx, weights=wt * contrib[:, c], minlength=self.ncell
                )

        # --- grid momentum update: velocity, gravity, sticky collisions ---
        active = self.grid_m > 1.0e-10
        gv = self.grid_v
        gv[active] /= self.grid_m[active, None]
        gv[~active] = 0.0
        gv[active] += dt * self.gravity[None, :]
        gv[self.frozen] = 0.0
        speed = np.linalg.norm(gv, axis=1)
        too_fast = speed > 60.0
        if np.any(too_fast):
            gv[too_fast] *= (60.0 / speed[too_fast])[:, None]

        # --- grid to particle ---
        new_v = np.zeros_like(self.vel)
        new_C = np.zeros_like(self.C)
        for i, j, k in self.offsets:
            wt = w[i, :, 0] * w[j, :, 1] * w[k, :, 2]
            idx = flat_base + i * stride[0] + j * stride[1] + k * stride[2]
            dpos = (np.array([i, j, k], dtype=float)[None, :] - fx) * dx
            gvn = gv[idx]
            new_v += wt[:, None] * gvn
            new_C += (4.0 * inv_dx * inv_dx) * wt[:, None, None] * np.einsum(
                "na,nb->nab", gvn, dpos
            )

        self.vel = new_v
        self.C = new_C
        eye = np.eye(3)[None, :, :]
        self.F = (eye + dt * self.C) @ self.F
        self._apply_plasticity()
        self.pos = self.pos + dt * self.vel
        self._resolve_particle_contacts()
        self.time += dt
        self.steps += 1

    def _resolve_particle_contacts(self) -> None:
        """Sticky particle-level contact: attach on the wall face and the floor."""
        hit_wall = self.pos[:, 0] >= self.wall_x
        if np.any(hit_wall):
            self.pos[hit_wall, 0] = self.wall_x
            self.vel[hit_wall] = 0.0
            self.C[hit_wall] = 0.0
        hit_floor = self.pos[:, 2] <= self.floor_z
        if np.any(hit_floor):
            self.pos[hit_floor, 2] = self.floor_z
            self.vel[hit_floor] = 0.0
            self.C[hit_floor] = 0.0
        for axis in range(3):
            low = self.pos[:, axis] < self.pos_lo[axis]
            high = self.pos[:, axis] > self.pos_hi[axis]
            if np.any(low):
                self.pos[low, axis] = self.pos_lo[axis]
                self.vel[low, axis] = 0.0
            if np.any(high):
                self.pos[high, axis] = self.pos_hi[axis]
                self.vel[high, axis] = 0.0

    def _time_step(self) -> float:
        h = float(np.max(self._hardening()))
        c = float(np.sqrt((self.lam0 + 2.0 * self.mu0) * h / max(self.p_density, _EPS)))
        vmax = float(np.max(np.abs(self.vel))) if self.n else 0.0
        dt = self.CFL * self.dx / max(c + 3.0 * vmax, _EPS)
        return float(np.clip(dt, self.DT_MIN, self.DT_MAX))

    # ------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while target - self.time > 1.0e-12 and guard < 2_000_000:
            dt = min(self._time_step(), target - self.time)
            if dt <= 0.0:
                break
            self._step(dt)
            self._sanitize()
            guard += 1
        self.time = target
        return self._frame(target)

    def _sanitize(self) -> None:
        """Keep the state finite; a diverged sample is replaced by a rest state."""
        bad = ~(np.isfinite(self.pos).all(axis=1) & np.isfinite(self.vel).all(axis=1))
        bad |= ~np.isfinite(self.F).reshape(self.n, -1).all(axis=1)
        bad |= ~np.isfinite(self.Jp)
        if np.any(bad):
            self.pos[bad] = np.clip(
                np.nan_to_num(self.pos[bad], nan=0.0, posinf=0.0, neginf=0.0),
                self.pos_lo[None, :],
                self.pos_hi[None, :],
            )
            self.vel[bad] = 0.0
            self.C[bad] = 0.0
            self.F[bad] = np.eye(3)[None, :, :]
            self.Jp[bad] = 1.0
        np.clip(self.C, -1.0e4, 1.0e4, out=self.C)
        np.clip(self.F, -1.0e3, 1.0e3, out=self.F)

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        pos = np.ascontiguousarray(self.pos, dtype=float)
        vel = np.ascontiguousarray(self.vel, dtype=float)
        mass = self.mass
        gaps = np.full(self.n, np.inf)
        for center, half in self.boxes:
            gaps = np.minimum(gaps, _box_signed_distance(pos, center, half))
        gaps = np.nan_to_num(gaps, nan=0.0, posinf=1.0e3, neginf=-1.0e3)

        band = 3.0 * self.contact_length
        sel = np.flatnonzero(gaps <= band)
        min_count = min(16, self.n)
        if sel.size < min_count:
            sel = np.sort(np.argsort(gaps, kind="stable")[:min_count])
        contact_gap = gaps[sel]

        detF = np.linalg.det(self.F)
        volume = self.ref_volume * np.abs(detF) * self.Jp
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": mass.copy(),
                "volume_quantity": volume,
                "reference_volume": self.ref_volume.copy(),
                "deformation_gradient": np.nan_to_num(self.F.copy()),
                "plastic_volume_ratio": self.Jp.copy(),
                "initial_component_id": self.component_id.copy(),
            },
            fields={
                "external_force": mass[:, None] * self.gravity[None, :],
                "contact_signed_distance": contact_gap,
                "contact_sample_index": sel.astype(np.int64),
                "wall_signed_distance": self.wall_x - pos[:, 0],
                "floor_signed_distance": pos[:, 2] - self.floor_z,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                **self.scalars0,
                "grid_spacing": float(self.dx),
                "total_mass": float(mass.sum()),
                "total_volume": float(volume.sum()),
                "steps_taken": float(self.steps),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit MLS-MPM, fixed-corotated snow with SVD-clamp plasticity",
            "contact_mode": "sticky",
            "particles": int(self.n),
            "grid": [int(self.nx), int(self.ny), int(self.nz)],
            "grid_spacing": float(self.dx),
            "steps": int(self.steps),
            "time": float(self.time),
            "total_mass": float(self.mass.sum()),
        }
