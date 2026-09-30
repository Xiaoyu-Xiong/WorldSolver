"""Multiphase position-based fluid solver for the octagonal two-phase drop.

Physically real time stepping: a predictive-corrective incompressible
particle method (position-based fluids, Macklin & Mueller style) with
per-particle rest densities.  The heavy blue phase (1000 kg/m^3) is
released at t=0, the four-times-lighter pink phase (250 kg/m^3) is held
at its scaffold position until its frozen release time (1.35 s) and then
dropped on top.  Density constraints against each particle's own rest
density give per-phase volume conservation and buoyancy-driven phase
separation; the frozen octagonal basin (floor z=0.08, inradius 1.06066,
rim z=0.62) is an impermeable boundary, the top is open.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class _OctagonalBasin:
    """Impermeable boundary matching assets/octagonal_basin.obj."""

    FLOOR_Z = 0.08
    RIM_Z = 0.62
    INRADIUS = 1.0606601717798213

    def __init__(self, margin: float):
        self.margin = float(margin)
        angles = np.arange(8) * (np.pi / 4.0)
        self.normals = np.stack((np.cos(angles), np.sin(angles)), axis=1)

    def clamp(self, pos: np.ndarray) -> np.ndarray:
        """Project positions back into the (infinite-wall) basin interior."""
        low = self.FLOOR_Z + self.margin
        hit = pos[:, 2] < low
        if hit.any():
            pos[hit, 2] = low
        s = pos[:, :2] @ self.normals.T
        k = np.argmax(s, axis=1)
        smax = s[np.arange(len(s)), k]
        limit = self.INRADIUS - self.margin
        out = smax > limit
        if out.any():
            idx = np.nonzero(out)[0]
            pos[idx, :2] -= self.normals[k[idx]] * (smax[idx] - limit)[:, None]
        return pos


class BenchmarkSolver:
    """Deterministic multiphase PBF solver for scene 003."""

    SUBSTEP = 0.002
    ITERS = 5
    H = 0.13                 # kernel support radius
    PARTICLE_RADIUS = 0.033
    XSPH_ALPHA = 0.10
    CFM = 0.05               # constraint-force-mixing relaxation
    VMAX = 12.0              # m/s safety clamp
    MOVE_CAP_FRAC = 0.08     # per-iteration displacement cap (fraction of H)

    GRID_LO = (-1.15, -1.15, 0.0)
    GRID_HI = (1.15, 1.15, 4.40)
    GRID_SPACING = 0.08
    VELOCITY_RADIUS = 0.16
    SURFACE_OFFSET = 0.045
    CHARACTERISTIC_LENGTH = 1.0606601717798213

    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", {}) or {}
        frame = scene.initial_state()
        self.pos = np.array(frame.position, dtype=float).copy()
        self.vel = np.array(frame.velocity, dtype=float).copy()
        attributes = frame.attributes
        self.phase = np.asarray(attributes["phase_id"], dtype=float)
        self.rho0 = np.asarray(attributes["density"], dtype=float).copy()
        self.release = np.asarray(attributes["release_time"], dtype=float).copy()
        self.pvol = np.asarray(attributes["target_volume"], dtype=float).copy()
        self.mass = self.rho0 * self.pvol
        self.n = len(self.pos)
        self.t = 0.0
        self.substep_count = 0

        gravity = np.asarray(config.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        if gravity.shape != (3,):
            gravity = np.array((0.0, 0.0, -9.81))
        self.gravity = gravity

        self.basin = _OctagonalBasin(self.PARTICLE_RADIUS)

        # Kernel constants (poly6 density, spiky gradient).
        h = self.H
        self._h2 = h * h
        self._poly6 = 315.0 / (64.0 * np.pi * h**9)
        self._spiky = 45.0 / (np.pi * h**6)
        self._w0 = self._poly6 * h**6
        self._move_cap = self.MOVE_CAP_FRAC * h
        # Per-particle CFM epsilon scale evaluated at r = h/2.
        g_half = self._spiky * (0.5 * h) ** 2
        self._eps = self.CFM * (self.mass * g_half / self.rho0) ** 2

        # Frozen diagnostic grid (cell centers).
        lo = np.asarray(self.GRID_LO, dtype=float)
        hi = np.asarray(self.GRID_HI, dtype=float)
        counts = np.maximum(np.ceil((hi - lo) / self.GRID_SPACING).astype(int), 1)
        axes = [lo[a] + (np.arange(counts[a]) + 0.5) * self.GRID_SPACING for a in range(3)]
        mesh = np.meshgrid(*axes, indexing="ij")
        centers = np.stack([m.ravel() for m in mesh], axis=1)
        self.grid_shape = tuple(int(c) for c in counts)
        self.centers = centers
        smax = (centers[:, :2] @ self.basin.normals.T).max(axis=1)
        self.active_mask = (
            (smax <= self.basin.INRADIUS) & (centers[:, 2] >= self.basin.FLOOR_Z)
        )
        self._knn = int(min(6, self.n))

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _pairs(self, points: np.ndarray) -> np.ndarray:
        tree = cKDTree(points)
        try:
            return tree.query_pairs(self.H, output_type="ndarray")
        except TypeError:  # pragma: no cover - older scipy
            return np.asarray(sorted(tree.query_pairs(self.H)), dtype=int).reshape(-1, 2)

    def _substep(self, dt: float) -> None:
        n = self.n
        mass = self.mass
        rho0 = self.rho0
        active = self.t >= self.release - 1e-12
        v = self.vel

        if active.any():
            v[active] += self.gravity * dt
            speed = np.linalg.norm(v, axis=1)
            over = speed > self.VMAX
            if over.any():
                v[over] *= (self.VMAX / speed[over])[:, None]

        x_old = self.pos
        pred = x_old + v * dt
        pairs = self._pairs(pred)

        if len(pairs) == 0:
            self.pos = pred
            return

        pi = pairs[:, 0]
        pj = pairs[:, 1]
        rho = np.empty(n)
        for _ in range(self.ITERS):
            rel = pred[pi] - pred[pj]
            d2 = np.einsum("ij,ij->i", rel, rel)
            np.maximum(d2, 1e-18, out=d2)
            d = np.sqrt(d2)
            inside = d2 < self._h2
            gap = self._h2 - d2
            w = np.where(inside, self._poly6 * gap * gap * gap, 0.0)

            rho[:] = mass * self._w0
            np.add.at(rho, pi, mass[pj] * w)
            np.add.at(rho, pj, mass[pi] * w)
            c_err = rho / rho0 - 1.0

            gm = np.where(inside, self._spiky * (self.H - d) ** 2, 0.0) / d
            gvec = -gm[:, None] * rel

            s_sum = np.zeros((n, 3))
            np.add.at(s_sum, pi, mass[pj, None] * gvec)
            np.add.at(s_sum, pj, -(mass[pi, None] * gvec))
            gv2 = np.einsum("ij,ij->i", gvec, gvec)
            acc = np.zeros(n)
            np.add.at(acc, pi, (mass[pj] / rho0[pi]) ** 2 * gv2)
            np.add.at(acc, pj, (mass[pi] / rho0[pj]) ** 2 * gv2)
            denom = np.einsum("ij,ij->i", s_sum, s_sum) / (rho0 * rho0) + acc + self._eps
            lam = -c_err / denom

            sij = lam[pi] + lam[pj]
            dp = np.zeros((n, 3))
            np.add.at(dp, pi, (sij / rho0[pi])[:, None] * gvec)
            np.add.at(dp, pj, -(sij / rho0[pj])[:, None] * gvec)
            mag = np.linalg.norm(dp, axis=1)
            mx = float(mag.max()) if len(mag) else 0.0
            if mx > self._move_cap:
                dp *= self._move_cap / mx
            dp[~active] = 0.0
            pred += dp
            self.basin.clamp(pred)

        v[:] = (pred - x_old) / dt
        v[~active] = 0.0

        # XSPH viscosity (one pass on the accepted configuration).
        rel = pred[pi] - pred[pj]
        d2 = np.einsum("ij,ij->i", rel, rel)
        np.maximum(d2, 1e-18, out=d2)
        gap = self._h2 - d2
        w = np.where(d2 < self._h2, self._poly6 * gap * gap * gap, 0.0)
        rho_safe = np.maximum(rho, 1e-9)
        dv = np.zeros((n, 3))
        np.add.at(dv, pi, ((v[pj] - v[pi]) * (mass[pj] / rho_safe[pj]) * w)[:, None])
        np.add.at(dv, pj, ((v[pi] - v[pj]) * (mass[pi] / rho_safe[pi]) * w)[:, None])
        if active.any():
            v[active] += self.XSPH_ALPHA * dv[active]

        self.pos = pred
        self.substep_count += 1

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("advance_to requires a finite time")
        while self.t < target - 1e-9:
            dt = min(self.SUBSTEP, target - self.t)
            self._substep(dt)
            self.t += dt
        if target > self.t:
            self.t = target
        self.t = max(self.t, target)
        return self._frame(target)

    # ------------------------------------------------------------------
    # StateFrame export
    # ------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        pos = np.nan_to_num(self.pos, nan=0.0, posinf=2.0, neginf=-2.0)
        pos[:, 2] = np.clip(pos[:, 2], -0.5, 5.0)
        pos[:, :2] = np.clip(pos[:, :2], -1.6, 1.6)
        vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
        speed = np.linalg.norm(vel, axis=1)
        fast = speed > self.VMAX
        if fast.any():
            vel[fast] *= (self.VMAX / speed[fast])[:, None]
        n = self.n

        mass_quantity = np.zeros((n, 2))
        mass_quantity[np.arange(n), self.phase.astype(int)] = self.mass

        tree = cKDTree(pos)
        d1 = tree.query(self.centers, k=1)[0]
        level = (d1 - self.SURFACE_OFFSET).astype(np.float32).reshape(self.grid_shape)
        dk, ik = tree.query(self.centers, k=self._knn)
        wgt = (dk <= self.VELOCITY_RADIUS).astype(np.float64)
        vgrid = (vel[ik] * wgt[..., None]).sum(axis=1)
        cnt = np.maximum(wgt.sum(axis=1), 1.0)
        vgrid = (vgrid / cnt[:, None]).astype(np.float32).reshape(self.grid_shape + (3,))

        attributes = {
            "phase_id": self.phase.copy(),
            "density": self.rho0.copy(),
            "release_time": self.release.copy(),
            "target_volume": self.pvol.copy(),
            "particle_radius": np.full(n, self.PARTICLE_RADIUS),
            "mass_quantity": mass_quantity,
        }
        fields = {
            "active_mask": self.active_mask.astype(np.float32).reshape(self.grid_shape),
            "liquid_level_set": level,
            "velocity_field": vgrid,
            "cumulative_mass_source": np.zeros(2),
        }
        scalars = {
            "characteristic_length": float(self.CHARACTERISTIC_LENGTH),
            "grid_spacing": float(self.GRID_SPACING),
            "gravity_z": float(self.gravity[2]),
            "heavy_phase_volume": float(self.pvol[self.phase < 0.5].sum()),
            "light_phase_volume": float(self.pvol[self.phase > 0.5].sum()),
            "light_release_time": float(self.release.max()),
        }
        return StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=self.mass.copy(),
            body_id=self.phase.astype(np.int32),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "multiphase position-based fluid (predictive-corrective SPH)",
            "particles": int(self.n),
            "kernel_support_radius": float(self.H),
            "substeps": int(self.substep_count),
            "substep_dt": float(self.SUBSTEP),
            "iterations_per_substep": int(self.ITERS),
            "phase_particles": {
                "heavy_blue": int((self.phase < 0.5).sum()),
                "light_pink": int((self.phase > 0.5).sum()),
            },
        }
