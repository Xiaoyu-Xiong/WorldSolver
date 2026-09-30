"""Material Point Method snow solver for scene 037_snowball_wall_sticky.

Recreates the SSCTS13 (Stomakhin et al.) "snowball thrown at a wall with
sticky contact" phenomenon with a self-contained explicit MPM:

* quadratic B-spline grid transfer (APIC affine particle-in-cell),
* fixed-corotational elastoplastic snow with critical-stretch plasticity
  and plastic hardening (mu/lambda scaled by exp(xi*(1-det Fp))),
* sticky contact realized by freezing all degrees of freedom of grid
  nodes inside the impact wall / below the floor (no-slip attachment),
* explicit symplectic grid update with gravity.

No Genesis, no keyframes: the state is produced purely by time stepping.
"""
from __future__ import annotations

import numpy as np

try:  # platform contract types (always present in the runner environment)
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - fallback for isolated syntax checks
    from dataclasses import dataclass, field
    from typing import Mapping

    @dataclass
    class StateFrame:  # type: ignore
        time: float
        position: np.ndarray
        velocity: np.ndarray
        orientation: object = None
        angular_velocity: object = None
        mass: object = None
        body_id: object = None
        attributes: Mapping[str, np.ndarray] = field(default_factory=dict)
        fields: Mapping[str, np.ndarray] = field(default_factory=dict)
        scalars: Mapping[str, float] = field(default_factory=dict)

        def validate(self) -> None:  # minimal stand-in
            pass


# 3x3x3 quadratic B-spline stencil offsets (i, j, k), shape (27, 3).
_OFFSETS = np.array(
    [[i, j, k] for i in range(3) for j in range(3) for k in range(3)], dtype=np.int64
)


class BenchmarkSolver:
    """Explicit MPM snow solver with sticky wall/floor contact."""

    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", {}) or {}
        self.seed = int(seed)
        self.metadata = scene.metadata() if hasattr(scene, "metadata") else {}

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=np.float64)
        self.v = np.array(frame0.velocity, dtype=np.float64)
        self.mass = np.array(frame0.mass, dtype=np.float64)
        n = len(self.x)
        self.n = n
        self.body_id = (
            np.array(frame0.body_id, dtype=np.int64)
            if frame0.body_id is not None
            else np.zeros(n, dtype=np.int64)
        )
        attrs = dict(frame0.attributes)
        nominal_density = float(cfg.get("snowball", {}).get("nominal_density", 520.0))
        self.Vp = (
            np.array(attrs["reference_volume"], dtype=np.float64)
            if "reference_volume" in attrs
            else self.mass / nominal_density
        )
        if "deformation_gradient" in attrs:
            self.F = np.array(attrs["deformation_gradient"], dtype=np.float64)
        else:
            self.F = np.tile(np.eye(3), (n, 1, 1))
        self.Fp = np.tile(np.eye(3), (n, 1, 1))
        self.detFp = np.ones(n, dtype=np.float64)
        self.B = np.zeros((n, 3, 3), dtype=np.float64)  # APIC affine velocity matrix
        self.initial_component_id = (
            np.array(attrs["initial_component_id"], dtype=np.int64)
            if "initial_component_id" in attrs
            else np.zeros(n, dtype=np.int64)
        )

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # ---- contact / frozen geometry -------------------------------------
        contact = cfg.get("contact", {})
        self.wall_x = float(contact.get("wall_inner_x", 0.0))
        self.floor_z = float(contact.get("floor_z", 0.0))
        self.contact_length_scale = float(contact.get("contact_length_scale", 0.065))
        # impact_wall: box centered (0.08, 0, 1.35), half extents (0.08, 1.2, 1.35)
        self.wall_y_half = 1.2
        self.wall_z_top = 2.7

        # ---- snow material (SSCTS13 fixed corotational + hardening) --------
        self.mu0 = 5.0e4
        self.lam0 = 3.5e4
        self.xi = 10.0  # hardening coefficient
        self.hmin, self.hmax = 0.05, 5.0
        self.theta_c = 0.025  # critical compression
        self.theta_s = 0.0075  # critical stretch

        # ---- MPM grid -------------------------------------------------------
        self.dx = 0.1
        xlo, xhi = -2.5, 0.4
        ylo, yhi = -1.0, 1.0
        zlo, zhi = -0.2, 3.0
        self.origin = np.array([xlo, ylo, zlo], dtype=np.float64)
        self.nx = int(round((xhi - xlo) / self.dx)) + 1
        self.ny = int(round((yhi - ylo) / self.dx)) + 1
        self.nz = int(round((zhi - zlo) / self.dx)) + 1
        self.nn = self.nx * self.ny * self.nz
        self.bounds_lo = self.origin.copy()
        self.bounds_hi = self.origin + np.array(
            [self.nx - 1, self.ny - 1, self.nz - 1], dtype=np.float64
        ) * self.dx

        gx = xlo + np.arange(self.nx) * self.dx
        gy = ylo + np.arange(self.ny) * self.dx
        gz = zlo + np.arange(self.nz) * self.dx
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        # sticky contact: freeze every node inside the wall or below the floor
        self.sticky = (GX >= self.wall_x - 1e-9) | (GZ <= self.floor_z + 1e-9)
        self.sticky_flat = self.sticky.reshape(-1)

        self.dt = 4.0e-4
        self.t = 0.0
        self._steps = 0

    # ------------------------------------------------------------------ utils
    def _sanitize(self, prev_x: np.ndarray) -> None:
        if not np.isfinite(self.x).all():
            bad = ~np.isfinite(self.x).all(axis=1)
            self.x[bad] = prev_x[bad]
            self.v[bad] = 0.0
            self.B[bad] = 0.0
            self.F[bad] = np.eye(3)
            self.Fp[bad] = np.eye(3)
            self.detFp[bad] = 1.0
        badv = ~np.isfinite(self.v).all(axis=1)
        if badv.any():
            self.v[badv] = 0.0
            self.B[badv] = 0.0
        badf = ~np.isfinite(self.F).all(axis=(1, 2))
        if badf.any():
            self.F[badf] = np.eye(3)
            self.Fp[badf] = np.eye(3)
            self.detFp[badf] = 1.0

    # ------------------------------------------------------- constitutive law
    def _stress_and_plasticity(self) -> np.ndarray:
        """Apply snow plasticity, advance F/Fp, return per-particle PK1*F^T."""
        U, S, Vt = np.linalg.svd(self.F)  # F = U diag(S) Vt
        S = np.clip(S, 1e-6, None)
        Sc = np.clip(S, 1.0 - self.theta_c, 1.0 + self.theta_s)

        # plastic update: F <- U diag(Sc) Vt,  Fp <- U diag(S/Sc) Vt Fp
        ratio = S / Sc
        self.detFp *= np.prod(ratio, axis=1)
        self.F = U @ (Sc[:, :, None] * Vt)  # U diag(Sc) Vt

        h = np.exp(self.xi * (1.0 - self.detFp))
        h = np.clip(h, self.hmin, self.hmax)
        mu = self.mu0 * h
        lam = self.lam0 * h

        J = np.prod(Sc, axis=1)
        # P F^T = U diag( 2 mu (Sc-1) Sc + lam (J-1) J ) U^T  (diagonal in U basis)
        diag = (
            2.0 * mu[:, None] * (Sc - 1.0) + (lam * (J - 1.0) * J)[:, None] / Sc
        ) * Sc
        stress = U @ (diag[:, :, None] * np.transpose(U, (0, 2, 1)))
        return stress

    # ------------------------------------------------------------- one step
    def _step(self, dt: float) -> None:
        prev_x = self.x.copy()
        stress = self._stress_and_plasticity()

        # ---- particle positions in grid index space ------------------------
        g = (self.x - self.origin) / self.dx  # (P,3)
        base = np.floor(g - 0.5).astype(np.int64)
        base[:, 0] = np.clip(base[:, 0], 0, self.nx - 3)
        base[:, 1] = np.clip(base[:, 1], 0, self.ny - 3)
        base[:, 2] = np.clip(base[:, 2], 0, self.nz - 3)
        t = g - base  # in [0.5, 1.5) nominally

        # d[p, k, a] = t[p, a] - offset_k[a]
        d = t[:, None, :] - _OFFSETS[None, :, :].astype(np.float64)
        a = np.abs(d)
        w_axis = np.where(
            a < 0.5,
            0.75 - a * a,
            np.where(a < 1.5, 0.5 * (1.5 - a) ** 2, 0.0),
        )
        dw_axis = np.where(a < 0.5, -2.0 * d, (a - 1.5) * np.sign(d))

        w = np.prod(w_axis, axis=2)  # (P,27)
        w01 = w_axis[:, :, 0] * w_axis[:, :, 1]
        w02 = w_axis[:, :, 0] * w_axis[:, :, 2]
        w12 = w_axis[:, :, 1] * w_axis[:, :, 2]
        gradw = np.stack(
            [dw_axis[:, :, 0] * w12, dw_axis[:, :, 1] * w02, dw_axis[:, :, 2] * w01],
            axis=2,
        ) / self.dx  # (P,27,3)

        idx = base[:, None, :] + _OFFSETS[None, :, :]  # (P,27,3)
        flat = ((idx[:, :, 0] * self.ny + idx[:, :, 1]) * self.nz + idx[:, :, 2]).reshape(-1)

        delta = idx.astype(np.float64) * self.dx + self.origin[None, None, :] - self.x[:, None, :]

        # ---- P2G: mass, momentum (APIC), internal force ---------------------
        mw = (self.mass[:, None] * w).reshape(-1)
        gmass = np.bincount(flat, weights=mw, minlength=self.nn)

        vel = self.v[:, None, :] + np.einsum("pij,pkj->pki", self.B, delta)
        mom = mw[:, None] * vel.reshape(-1, 3)  # (P*27, 3)
        gmom = np.empty((self.nn, 3), dtype=np.float64)
        for c in range(3):
            gmom[:, c] = np.bincount(flat, weights=mom[:, c], minlength=self.nn)

        fp = -(self.Vp[:, None, None] * np.einsum("pij,pkj->pki", stress, gradw)).reshape(-1, 3)
        gforce = np.empty((self.nn, 3), dtype=np.float64)
        for c in range(3):
            gforce[:, c] = np.bincount(flat, weights=fp[:, c], minlength=self.nn)

        # ---- grid update -----------------------------------------------------
        vgrid = np.zeros((self.nn, 3), dtype=np.float64)
        moving = gmass > 1e-12
        if np.any(moving):
            mi = gmass[moving]
            vgrid[moving] = (
                gmom[moving]
                + dt * gforce[moving]
                + dt * mi[:, None] * self.gravity[None, :]
            ) / mi[:, None]
        # sticky contact: freeze nodes inside the wall / under the floor
        vgrid[self.sticky_flat] = 0.0

        # ---- G2P: APIC transfer ---------------------------------------------
        vg = vgrid[flat].reshape(-1, 27, 3)
        wv = w[:, :, None] * vg
        v_new = wv.sum(axis=1)
        B_new = (4.0 / (self.dx * self.dx)) * np.einsum("pk,pki,pkj->pij", w, vg, delta)

        self.v = v_new
        self.B = B_new
        self.x = self.x + dt * self.v
        self.F = (np.eye(3) + dt * self.B) @ self.F

        # ---- rigid containment (safety net, matches sticky boundaries) ------
        self.x[:, 0] = np.clip(self.x[:, 0], self.bounds_lo[0], self.wall_x - 1e-4)
        self.x[:, 1] = np.clip(self.x[:, 1], self.bounds_lo[1], self.bounds_hi[1])
        self.x[:, 2] = np.clip(self.x[:, 2], self.floor_z + 1e-4, self.bounds_hi[2])
        stuck = (self.x[:, 0] >= self.wall_x - 2e-4) & (self.v[:, 0] > 0.0)
        stuck |= (self.x[:, 2] <= self.floor_z + 2e-4) & (self.v[:, 2] < 0.0)
        if np.any(stuck):
            self.v[stuck] = 0.0
            self.B[stuck] = 0.0

        self._sanitize(prev_x)
        self._steps += 1

    # ------------------------------------------------------------ trajectory
    def _frame(self, t: float):
        x = self.x.copy()
        v = self.v.copy()
        m = self.mass.copy()

        gap_wall = self.wall_x - x[:, 0]
        in_wall = (np.abs(x[:, 1]) <= self.wall_y_half) & (x[:, 2] <= self.wall_z_top)
        gap_wall = np.where(in_wall, gap_wall, np.full_like(gap_wall, 10.0))
        gap_floor = x[:, 2] - self.floor_z
        contact_sd = np.minimum(gap_wall, gap_floor)

        frame = StateFrame(
            time=float(t),
            position=x,
            velocity=v,
            mass=m,
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": m.copy(),
                "reference_volume": self.Vp.copy(),
                "deformation_gradient": self.F.copy(),
                "plastic_determinant": self.detFp.copy(),
                "initial_component_id": self.initial_component_id.copy(),
            },
            fields={
                "external_force": m[:, None] * self.gravity[None, :],
                "contact_signed_distance": contact_sd,
                "wall_signed_distance": gap_wall,
                "floor_signed_distance": gap_floor,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
                "initial_total_mass": float(m.sum()),
                "initial_total_volume": float(self.Vp.sum()),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        t_target = float(time)
        if t_target < self.t:
            t_target = self.t
        while self.t < t_target - 1e-12:
            dt = min(self.dt, t_target - self.t)
            self._step(dt)
            self.t = self.t + dt
        if abs(self.t - t_target) < 1e-12:
            self.t = t_target
        return self._frame(t_target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit_mpm_apic_fixed_corotational_snow",
            "num_particles": int(self.n),
            "grid": [self.nx, self.ny, self.nz],
            "dx": float(self.dx),
            "dt": float(self.dt),
            "time": float(self.t),
            "steps": int(self._steps),
            "total_mass": float(self.mass.sum()),
            "mean_plastic_det": float(np.mean(self.detFp)),
            "contact_mode": "sticky",
        }
