"""Physically-based MPM solver for two snowballs colliding head-on.

Elastoplastic Material Point Method (APIC transfer, quadratic B-splines)
with a fixed-corotated constitutive model, plastic return mapping
(Stomakhin-style snow hardening), and tensile-fracture damage that turns
yielded snow into weak granular debris.  Gravity and a frictional floor
complete the scene physics.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _inv3(A: np.ndarray) -> np.ndarray:
    """Vectorized analytic inverse of a batch of 3x3 matrices."""
    a = A[..., 0, 0]; b = A[..., 0, 1]; c = A[..., 0, 2]
    d = A[..., 1, 0]; e = A[..., 1, 1]; f = A[..., 1, 2]
    g = A[..., 2, 0]; h = A[..., 2, 1]; i = A[..., 2, 2]
    A00 = e * i - f * h
    A01 = c * h - b * i
    A02 = b * f - c * e
    det = a * A00 + d * A01 + g * A02
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    inv_det = 1.0 / det
    out = np.empty_like(A)
    out[..., 0, 0] = A00 * inv_det
    out[..., 0, 1] = A01 * inv_det
    out[..., 0, 2] = A02 * inv_det
    out[..., 1, 0] = (f * g - d * i) * inv_det
    out[..., 1, 1] = (a * i - c * g) * inv_det
    out[..., 1, 2] = (c * d - a * f) * inv_det
    out[..., 2, 0] = (d * h - e * g) * inv_det
    out[..., 2, 1] = (b * g - a * h) * inv_det
    out[..., 2, 2] = (a * e - b * d) * inv_det
    return out


def _det3(A: np.ndarray) -> np.ndarray:
    return (
        A[..., 0, 0] * (A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1])
        - A[..., 0, 1] * (A[..., 1, 0] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 0])
        + A[..., 0, 2] * (A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0])
    )


def _polar_rot(F: np.ndarray) -> np.ndarray:
    """Rotation R from the polar decomposition F = R S (Higham iteration)."""
    R = F.copy()
    for _ in range(3):
        Rit = np.swapaxes(_inv3(R), -1, -2)
        R = 0.5 * (R + Rit)
    return R


class BenchmarkSolver:
    # ---- material constants (packed snow, Stomakhin et al. style) ----
    YOUNG = 1.0e5        # Pa
    POISSON = 0.2
    THETA_C = 0.03       # tensile yield strain
    S_MIN = 0.70         # max allowed compressive singular value drop
    S_MAX_DMG = 1.002    # intact/damaged tensile clamp
    HARDEN_XI = 3.0      # hardening rate
    HARDEN_CAP = 2.0     # max extra stiffness factor

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        snow = cfg["snowballs"]
        frame0 = scene.initial_state()

        self.rng = np.random.default_rng(int(seed))
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.floor_z = float(cfg["floor"]["top_z"])
        fe = cfg["floor"]["half_extents"]
        self.floor_hx = float(fe[0])
        self.floor_hy = float(fe[1])

        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.body_id = np.array(frame0.body_id)
        self.V0 = np.array(frame0.attributes["reference_volume"], dtype=float)
        self.N = len(self.pos)

        self.F = np.tile(np.eye(3), (self.N, 1, 1))
        self.ep = np.zeros(self.N)                 # accumulated plastic strain
        self.damaged = np.zeros(self.N, dtype=bool)  # tensile-fracture damage
        mu0 = self.YOUNG / (2.0 * (1.0 + self.POISSON))
        lam0 = self.YOUNG * self.POISSON / ((1.0 + self.POISSON) * (1.0 - 2.0 * self.POISSON))
        self.mu0 = mu0
        self.lam0 = lam0

        # ---- background grid ----
        self.h = float(snow["particle_spacing"])
        h = self.h
        self.origin = np.array([-6.0, -5.0, -0.375])
        self.nx, self.ny, self.nz = 161, 135, 46
        self.n_nodes = self.nx * self.ny * self.nz

        I, J, K = np.meshgrid(
            np.arange(self.nx), np.arange(self.ny), np.arange(self.nz), indexing="ij"
        )
        flat = (I + self.nx * (J + self.ny * K)).ravel()
        X = self.origin[0] + I.ravel() * h
        Y = self.origin[1] + J.ravel() * h
        Z = self.origin[2] + K.ravel() * h
        on_floor = (
            (Z <= self.floor_z + 1e-9)
            & (np.abs(X) <= self.floor_hx + h)
            & (np.abs(Y) <= self.floor_hy + h)
        )
        self.idx_floor = flat[on_floor]
        self.idx_xlo = flat[I.ravel() <= 0]
        self.idx_xhi = flat[I.ravel() >= self.nx - 1]
        self.idx_ylo = flat[J.ravel() <= 0]
        self.idx_yhi = flat[J.ravel() >= self.ny - 1]
        self.idx_zlo = flat[K.ravel() <= 0]
        self.idx_zhi = flat[K.ravel() >= self.nz - 1]

        # particle position clamp bounds (keep spline support inside grid)
        self.p_lo = self.origin + 0.6 * h
        self.p_hi = self.origin + (np.array([self.nx, self.ny, self.nz]) - 1.4) * h

        self.dt = 1.25e-3
        self.step_count = 0
        self.time = 0.0
        self.scalars0 = dict(frame0.scalars)
        self.combos = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]

    # ------------------------------------------------------------------
    def _step(self) -> None:
        dt = self.dt
        h = self.h
        N = self.N
        pos = self.pos
        vel = self.vel

        # ---------------- stress (fixed corotated, hardened) -------------
        R = _polar_rot(self.F)
        J = _det3(self.F)
        FinvT = np.swapaxes(_inv3(self.F), -1, -2)
        harden = 1.0 + np.minimum(self.HARDEN_XI * self.ep, self.HARDEN_CAP)
        mu = self.mu0 * harden
        lam = self.lam0 * harden
        P = mu[:, None, None] * (self.F - R) + (lam * (J - 1.0) * J)[:, None, None] * FinvT

        # ---------------- P2G ----------------
        xg = (pos - self.origin) / h
        base = np.floor(xg - 0.5).astype(np.int64)
        np.clip(base, 0, np.array([self.nx - 3, self.ny - 3, self.nz - 3]), out=base)
        fx = xg - base
        wa = np.stack(
            [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2],
            axis=-1,
        )  # [N,3axis,3off]
        gw = np.stack(
            [-(1.5 - fx), -2.0 * (fx - 1.0), (fx - 0.5)], axis=-1
        ) / h  # d(w)/d(coord)

        mgrid = np.zeros(self.n_nodes)
        mom = np.zeros((self.n_nodes, 3))
        mv = self.mass[:, None] * (vel + dt * self.gravity[None, :])
        dP = dt * self.V0[:, None, None] * P

        cached = []
        nx, ny = self.nx, self.ny
        for a, b, c in self.combos:
            w = wa[:, 0, a] * wa[:, 1, b] * wa[:, 2, c]
            gWx = gw[:, 0, a] * wa[:, 1, b] * wa[:, 2, c]
            gWy = wa[:, 0, a] * gw[:, 1, b] * wa[:, 2, c]
            gWz = wa[:, 0, a] * wa[:, 1, b] * gw[:, 2, c]
            fl = (base[:, 0] + a) + nx * ((base[:, 1] + b) + ny * (base[:, 2] + c))
            imp = w[:, None] * mv - (
                dP[:, :, 0] * gWx[:, None]
                + dP[:, :, 1] * gWy[:, None]
                + dP[:, :, 2] * gWz[:, None]
            )
            mgrid += np.bincount(fl, weights=w * self.mass, minlength=self.n_nodes)
            for ax in range(3):
                mom[:, ax] += np.bincount(fl, weights=imp[:, ax], minlength=self.n_nodes)
            dx = np.stack(
                [(a - fx[:, 0]) * h, (b - fx[:, 1]) * h, (c - fx[:, 2]) * h], axis=1
            )
            cached.append((fl, w, dx))

        # ---------------- grid update + boundaries ----------------
        vg = np.zeros((self.n_nodes, 3))
        np.divide(mom, mgrid[:, None], out=vg, where=mgrid[:, None] > 1e-12)
        # NB: gravity impulse is already folded into the scattered particle
        # momentum (mv); do not add it again on the grid.
        # frictional floor (finite extents)
        vf = vg[self.idx_floor]
        vf[:, 2] = np.maximum(vf[:, 2], 0.0)
        vf[:, 0] *= 0.92
        vf[:, 1] *= 0.92
        vg[self.idx_floor] = vf
        # domain walls / catcher
        vg[self.idx_xlo, 0] = np.maximum(vg[self.idx_xlo, 0], 0.0)
        vg[self.idx_xhi, 0] = np.minimum(vg[self.idx_xhi, 0], 0.0)
        vg[self.idx_ylo, 1] = np.maximum(vg[self.idx_ylo, 1], 0.0)
        vg[self.idx_yhi, 1] = np.minimum(vg[self.idx_yhi, 1], 0.0)
        vg[self.idx_zlo, 2] = np.maximum(vg[self.idx_zlo, 2], 0.0)
        vg[self.idx_zhi, 2] = np.minimum(vg[self.idx_zhi, 2], 0.0)

        # ---------------- G2P (APIC) ----------------
        vp = np.zeros((N, 3))
        C = np.zeros((N, 3, 3))
        for fl, w, dx in cached:
            vi = vg[fl]
            vp += w[:, None] * vi
            C += w[:, None, None] * (vi[:, :, None] * dx[:, None, :])
        C *= 4.0 / (h * h)
        self.vel = vp
        self.F = self.F + dt * (C @ self.F)
        new_pos = pos + dt * vp

        # keep particles inside the grid
        lo_hit = new_pos < self.p_lo
        hi_hit = new_pos > self.p_hi
        new_pos = np.clip(new_pos, self.p_lo, self.p_hi)
        hit = lo_hit | hi_hit
        self.vel = np.where(hit, 0.0, self.vel)
        self.pos = new_pos

        # ---------------- plastic return mapping + fracture ----------------
        U, s, Vh = np.linalg.svd(self.F)
        self.damaged |= s[:, 0] > 1.0 + self.THETA_C
        hi_lim = np.where(self.damaged, self.S_MAX_DMG, 1.0 + self.THETA_C)
        sn = np.minimum(s, hi_lim[:, None])
        sn = np.maximum(sn, self.S_MIN)
        self.ep += np.abs(s - sn).sum(axis=1)
        self.F = (U * sn[:, None, :]) @ Vh

    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.dt))
        while self.step_count < target:
            self._step()
            self.step_count += 1
        self.time = self.step_count * self.dt

        ke = 0.5 * float(np.sum(self.mass * np.einsum("ij,ij->i", self.vel, self.vel)))
        mom = (self.mass[:, None] * self.vel).sum(axis=0)
        dmg_frac = float(self.damaged.mean()) if self.N else 0.0
        scalars = dict(self.scalars0)
        scalars.update(
            {
                "kinetic_energy": ke,
                "damaged_fraction": dmg_frac,
                "total_mass": float(self.mass.sum()),
                "momentum_x": float(mom[0]),
                "momentum_y": float(mom[1]),
                "momentum_z": float(mom[2]),
                "steps": float(self.step_count),
            }
        )
        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": self.F.copy(),
                "mass_quantity": self.mass.copy(),
                "reference_volume": self.V0.copy(),
                "initial_component_id": self.body_id.copy(),
                "damage": self.damaged.astype(float),
                "plastic_strain": self.ep.copy(),
            },
            fields={
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "floor_signed_distance": self.pos[:, 2] - self.floor_z,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "scheme": "MPM APIC quadratic B-spline, fixed-corotated snow plasticity",
            "steps": self.step_count,
            "dt": self.dt,
            "n_particles": self.N,
            "damaged_fraction": float(self.damaged.mean()) if self.N else 0.0,
        }
