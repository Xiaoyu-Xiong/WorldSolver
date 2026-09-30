"""Weakly-compressible SPH solver for scene 133 (viscosity-contrast splash).

Two matched liquid columns with contrasting dynamic viscosity drop side by
side into two pools separated by a rigid divider inside a rigid basin.

Numerics (implemented from scratch, no external physics engine):

* 3-D cubic-spline SPH with support radius ``2h``.
* Tait equation of state (weak compressibility), pressure clamped at zero
  so free surfaces stay well behaved.
* Momentum-conserving pairwise Laplacian viscosity with a per-material
  kinematic viscosity (low vs high) -> the required viscosity contrast.
* Static boundary particles on the rigid surfaces with dynamic-boundary
  pressure reaction and no-slip viscous friction.
* Explicit symplectic Euler stepping with adaptive, stability-limited dt
  (sound-speed CFL, advective CFL, viscous diffusion limit).
* Hard non-penetration projection against the frozen rigid boxes (floor,
  basin rim, divider) as a safety net so the liquid always respects the
  rigid geometry.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

TAIT_GAMMA = 7.0
SOUND_SPEED = 20.0          # m/s, artificial sound speed
NU_LOW = 1.0e-3             # m^2/s, low-viscosity liquid (water-like)
NU_HIGH = 0.25              # m^2/s, high-viscosity liquid (syrup-like)
DT_MAX = 2.5e-3
V_CLAMP = 20.0

# Frozen rigid geometry (from scene_geometry.yaml).
GROUND_HALF = (3.2, 1.4, 0.05)
GROUND_TOP_Z = 0.0
DIVIDER_HALF = (0.08, 0.9, 0.65)
DIVIDER_CENTER = (0.0, 0.0, 0.65)
DIVIDER_TOP_Z = 1.3
WALL_MARGIN = 0.04


class BenchmarkSolver:
    """Deterministic weakly-compressible SPH liquid solver."""

    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        cfg = scene.config
        self.scene = scene
        self.seed = int(seed)
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        sampling = cfg["sampling"]
        self.spacing = float(sampling["spacing"])
        self.rho0 = float(sampling["density"])
        self.pm = self.rho0 * self.spacing ** 3
        self.h = 1.15 * self.spacing
        self.R = 2.0 * self.h
        self.w0 = 1.0 / (np.pi * self.h ** 3)

        self.pos0 = np.array(frame0.position, dtype=float)
        self.vel0 = np.zeros_like(self.pos0)
        self.label = np.asarray(frame0.attributes["material_label"], dtype=np.int64)
        self.mass = np.asarray(frame0.mass, dtype=float)
        self.nu_of = np.array(
            [NU_LOW if lbl in (0, 2) else NU_HIGH for lbl in range(4)]
        )
        self.nu_max = NU_HIGH

        self.pos = self.pos0.copy()
        self.vel = self.vel0.copy()
        self.N = len(self.pos)
        self.time = 0.0
        self._substeps = 0

        # Kept initial attributes for frame parity.
        self.attr0 = {k: np.array(v) for k, v in frame0.attributes.items()}

        # Frozen scalar scales.
        self.characteristic_length = float(frame0.scalars["characteristic_length"])
        self.reference_velocity = float(
            np.sqrt(2.0 * 9.81 * max(0.1, 1.8 - 0.45))
        )
        self.grid_spacing = self.spacing

        # Rigid clamp limits.
        self.zmin = GROUND_TOP_Z + WALL_MARGIN
        self.xlim = GROUND_HALF[0] - WALL_MARGIN
        self.ylim = GROUND_HALF[1] - WALL_MARGIN
        self.div_h = (
            DIVIDER_HALF[0] + WALL_MARGIN,
            DIVIDER_HALF[1] + WALL_MARGIN,
            DIVIDER_TOP_Z + WALL_MARGIN,
        )

        # Diagnostic grid from the frozen config.
        shape = tuple(int(v) for v in sampling["field_grid_shape"])
        self.gn = shape
        self.glo = np.array([-3.0, -1.2, 0.0], dtype=float)
        self.ghi = np.array([3.0, 1.2, 3.8], dtype=float)
        self.gd = (self.ghi - self.glo) / np.asarray(shape, dtype=float)
        gi = np.indices(shape).reshape(3, -1).T
        self.gcenters = self.glo + (gi + 0.5) * self.gd
        self.gsize = int(np.prod(shape))

        self._build_boundaries()
        self.btree = cKDTree(self.bpos)

    # ------------------------------------------------------------------
    # boundary particles
    # ------------------------------------------------------------------
    def _build_boundaries(self) -> None:
        parts: list[np.ndarray] = []
        norms: list[np.ndarray] = []

        # Ground top face (z = 0), two buried support layers.
        gx = np.linspace(-GROUND_HALF[0], GROUND_HALF[0], 46)
        gy = np.linspace(-GROUND_HALF[1], GROUND_HALF[1], 21)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        for z in (-0.05, -0.19):
            pts = np.column_stack(
                [X.ravel(), Y.ravel(), np.full(X.size, z)]
            )
            parts.append(pts)
            norms.append(np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1)))

        # Divider slab: two particle columns (representing both x faces).
        dy = np.linspace(-0.86, 0.86, 14)
        dz = np.linspace(0.05, 1.25, 10)
        Y, Z = np.meshgrid(dy, dz, indexing="ij")
        for cx in (-0.04, 0.04):
            pts = np.column_stack(
                [np.full(Y.size, cx), Y.ravel(), Z.ravel()]
            )
            parts.append(pts)
            n = np.array([float(np.sign(cx)), 0.0, 0.0])
            norms.append(np.tile(n, (len(pts), 1)))

        # Divider top face (z = 1.26).
        Y, Z = np.meshgrid(dy, [1.26], indexing="ij")
        for cx in (-0.04, 0.04):
            pts = np.column_stack(
                [np.full(Y.size, cx), Y.ravel(), np.full(Y.size, 1.26)]
            )
            parts.append(pts)
            norms.append(np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1)))

        self.bpos = np.concatenate(parts, axis=0)
        self.bnormal = np.concatenate(norms, axis=0)
        self.M = len(self.bpos)

        # Static boundary-boundary density contribution.
        tree = cKDTree(self.bpos)
        rho_bb = np.full(self.M, self.pm * self.w0)
        pairs = tree.query_pairs(self.R, output_type="ndarray")
        if len(pairs):
            a, b = pairs[:, 0], pairs[:, 1]
            w = self._w(np.linalg.norm(self.bpos[a] - self.bpos[b], axis=1))
            np.add.at(rho_bb, a, self.pm * w)
            np.add.at(rho_bb, b, self.pm * w)
        self.rho_bb = rho_bb

    # ------------------------------------------------------------------
    # kernel and equation of state
    # ------------------------------------------------------------------
    def _w(self, r: np.ndarray) -> np.ndarray:
        q = np.asarray(r, dtype=float) / self.h
        out = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        qq = q[m1]
        out[m1] = 1.0 - 1.5 * qq * qq + 0.75 * qq * qq * qq
        qq = 2.0 - q[m2]
        out[m2] = 0.25 * qq * qq * qq
        return out / (np.pi * self.h ** 3)

    def _dwdq(self, r: np.ndarray) -> np.ndarray:
        """dW/dr evaluated at radii r."""
        q = np.asarray(r, dtype=float) / self.h
        out = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        qq = q[m1]
        out[m1] = -3.0 * qq + 2.25 * qq * qq
        qq = 2.0 - q[m2]
        out[m2] = -0.75 * qq * qq
        return out / (np.pi * self.h ** 4)

    def _pressure(self, rho: np.ndarray) -> np.ndarray:
        p = (
            SOUND_SPEED ** 2
            * self.rho0
            / TAIT_GAMMA
            * ((rho / self.rho0) ** TAIT_GAMMA - 1.0)
        )
        return np.maximum(p, 0.0)

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        pos, vel = self.pos, self.vel
        N = self.N

        allpos = np.concatenate([pos, self.bpos], axis=0)
        tree = cKDTree(allpos)
        pairs = tree.query_pairs(self.R, output_type="ndarray")
        pa, pb = pairs[:, 0], pairs[:, 1]
        ff = (pa < N) & (pb < N)
        fa = (pa < N) & (pb >= N)
        af = (pa >= N) & (pb < N)
        ia = np.concatenate([pa[ff], pb[af]])
        ib = np.concatenate([pb[ff], pa[af]])
        fi = np.concatenate([pa[fa], pb[af]])
        bj = np.concatenate([pb[fa] - N, pa[af] - N])

        if ia.size:
            dff = np.linalg.norm(pos[ia] - pos[ib], axis=1)
        else:
            dff = np.zeros(0)
        if fi.size:
            dfb = np.linalg.norm(pos[fi] - self.bpos[bj], axis=1)
        else:
            dfb = np.zeros(0)

        # Densities (fluid + boundary contribution, dynamic boundaries).
        rho = np.full(N, self.pm * self.w0)
        rho_b = self.rho_bb.copy()
        if ia.size:
            wff = self._w(dff)
            np.add.at(rho, ia, self.pm * wff)
            np.add.at(rho, ib, self.pm * wff)
        if fi.size:
            wfb = self._w(dfb)
            np.add.at(rho, fi, self.pm * wfb)
            np.add.at(rho_b, bj, self.pm * wfb)
        np.maximum(rho, 0.5 * self.rho0, out=rho)
        np.maximum(rho_b, 0.5 * self.rho0, out=rho_b)

        p_f = self._pressure(rho)
        p_b = self._pressure(rho_b)

        acc = np.broadcast_to(self.gravity, (N, 3)).copy()

        # Pressure forces (fluid-fluid).
        if ia.size:
            grad = (self._dwdq(dff) / np.maximum(dff, 1e-12))[:, None] * (
                pos[ia] - pos[ib]
            )
            coef = self.pm * (p_f[ia] / rho[ia] ** 2 + p_f[ib] / rho[ib] ** 2)
            np.add.at(acc, ia, -coef[:, None] * grad)
            np.add.at(acc, ib, coef[:, None] * grad)

        # Pressure reaction from rigid boundaries.
        if fi.size:
            gradb = (self._dwdq(dfb) / np.maximum(dfb, 1e-12))[:, None] * (
                pos[fi] - self.bpos[bj]
            )
            coefb = self.pm * (
                p_f[fi] / rho[fi] ** 2 + p_b[bj] / rho_b[bj] ** 2
            )
            np.add.at(acc, fi, -coefb[:, None] * gradb)

        # Laplacian viscosity, momentum conserving, per-material nu.
        if ia.size:
            kappa = -self._dwdq(dff) / np.maximum(dff, 1e-12)
            nua = self.nu_of[self.label[ia]]
            nub = self.nu_of[self.label[ib]]
            nuij = 2.0 * nua * nub / np.maximum(nua + nub, 1e-12)
            rbar = 0.5 * (rho[ia] + rho[ib])
            cv = self.pm * 2.0 * nuij * rbar * kappa / (rho[ia] * rho[ib])
            dv = vel[ib] - vel[ia]
            np.add.at(acc, ia, cv[:, None] * dv)
            np.add.at(acc, ib, -cv[:, None] * dv)

        # No-slip viscous wall friction (static walls, v_wall = 0).
        if fi.size:
            kappab = -self._dwdq(dfb) / np.maximum(dfb, 1e-12)
            cvb = (
                self.pm
                * 2.0
                * self.nu_of[self.label[fi]]
                * kappab
                / rho_b[bj]
            )
            np.add.at(acc, fi, -cvb[:, None] * vel[fi])

        # Symplectic Euler integration.
        vel += dt * acc
        speed = np.linalg.norm(vel, axis=1)
        fast = speed > V_CLAMP
        if np.any(fast):
            vel[fast] *= (V_CLAMP / speed[fast])[:, None]
        pos += dt * vel
        np.nan_to_num(pos, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(vel, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        self._collide()
        self._substeps += 1

    def _collide(self) -> None:
        """Hard non-penetration projection onto the frozen rigid boxes."""
        pos, vel = self.pos, self.vel

        # Ground floor.
        m = pos[:, 2] < self.zmin
        if m.any():
            pos[m, 2] = self.zmin
            vel[m, 2] = np.where(vel[m, 2] < 0.0, 0.0, vel[m, 2])

        # Basin rim (lateral faces of the ground box, extended upward).
        for ax, lim in ((0, self.xlim), (1, self.ylim)):
            v = pos[:, ax]
            m = v > lim
            if m.any():
                pos[m, ax] = lim
                vel[m, ax] = np.where(vel[m, ax] > 0.0, 0.0, vel[m, ax])
            m = v < -lim
            if m.any():
                pos[m, ax] = -lim
                vel[m, ax] = np.where(vel[m, ax] < 0.0, 0.0, vel[m, ax])

        # Divider box (push out along the least-penetration axis).
        hx, hy, hz = self.div_h
        ex = hx - np.abs(pos[:, 0])
        ey = hy - np.abs(pos[:, 1])
        etop = hz - pos[:, 2]
        inside = (ex > 0.0) & (ey > 0.0) & (etop > 0.0)
        if inside.any():
            idx = np.where(inside)[0]
            exits = np.stack([ex[idx], ey[idx], etop[idx]], axis=1)
            axis = np.argmin(exits, axis=1)
            for a, half in ((0, hx), (1, hy), (2, hz)):
                sel = idx[axis == a]
                if sel.size == 0:
                    continue
                if a == 2:
                    pos[sel, 2] = half
                    vel[sel, 2] = np.where(vel[sel, 2] < 0.0, 0.0, vel[sel, 2])
                else:
                    sgn = np.where(np.sign(pos[sel, a]) == 0.0, 1.0, np.sign(pos[sel, a]))
                    pos[sel, a] = sgn * half
                    vel[sel, a] = np.where(
                        sgn * vel[sel, a] < 0.0, 0.0, vel[sel, a]
                    )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        t = float(time)
        if t < self.time - 1e-9:
            self.pos = self.pos0.copy()
            self.vel = self.vel0.copy()
            self.time = 0.0
        while self.time < t - 1e-9:
            remaining = t - self.time
            vmax = max(float(np.max(np.linalg.norm(self.vel, axis=1))), 1e-3)
            dt = min(
                DT_MAX,
                0.35 * self.h / SOUND_SPEED,
                0.3 * self.h / vmax,
                0.08 * self.h * self.h / self.nu_max,
                remaining,
            )
            if dt < 1e-7:
                self.time = t
                break
            self._substep(dt)
            self.time += dt
        self.time = t
        return self._make_frame(t)

    # ------------------------------------------------------------------
    # StateFrame assembly
    # ------------------------------------------------------------------
    def _make_frame(self, t: float) -> StateFrame:
        pos = self.pos.copy()
        vel = self.vel.copy()
        N = self.N

        attrs = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": np.array(self.attr0["particle_radius"]),
            "material_label": self.label.copy(),
            "deformation_gradient": np.array(self.attr0["deformation_gradient"]),
        }

        ftree = cKDTree(pos)

        # Liquid level set (negative inside the liquid).
        d_near, _ = ftree.query(self.gcenters, k=1)
        level = (d_near - 0.6 * self.spacing).reshape(self.gn)

        # Velocity deposit onto the diagnostic grid (SPH scatter).
        wsum = np.zeros(self.gsize)
        wv = np.zeros((self.gsize, 3))
        base = np.floor((pos - self.glo) / self.gd).astype(np.int64)
        offs = np.arange(-3, 4)
        OX, OY, OZ = np.meshgrid(offs, offs, offs, indexing="ij")
        off = np.stack([OX.ravel(), OY.ravel(), OZ.ravel()], axis=1)
        idx = base[:, None, :] + off[None, :, :]
        valid = np.all((idx >= 0) & (idx < np.asarray(self.gn)), axis=2)
        np.clip(idx, 0, np.asarray(self.gn) - 1, out=idx)
        cellc = self.glo + (idx + 0.5) * self.gd
        dist = np.linalg.norm(cellc - pos[:, None, :], axis=2)
        w = self._w(dist) * valid
        flat = (idx[..., 0] * self.gn[1] + idx[..., 1]) * self.gn[2] + idx[..., 2]
        np.add.at(wsum, flat[valid], w[valid])
        contrib = w[..., None] * vel[:, None, :]
        np.add.at(wv, flat[valid], contrib[valid])
        vfield_flat = np.zeros((self.gsize, 3))
        m = wsum > 1e-12
        vfield_flat[m] = wv[m] / wsum[m][:, None]
        vfield = vfield_flat.reshape(tuple(self.gn) + (3,))

        # Wall samples: collocated prescribed and fluid-side velocities.
        darr, iarr = ftree.query(self.bpos, k=8, distance_upper_bound=self.R)
        iarr = np.clip(iarr, 0, N - 1)
        wb = self._w(np.where(np.isfinite(darr), darr, 2.0 * self.h))
        wsum_b = wb.sum(axis=1)
        bvel = np.zeros_like(self.bnormal)
        msk = wsum_b > 1e-12
        bvel[msk] = (
            (wb[msk, :, None] * vel[iarr[msk]]).sum(axis=1)
            / wsum_b[msk, None]
        )

        fields = {
            "active_mask": np.ones(self.gn, dtype=bool),
            "liquid_level_set": level,
            "velocity_field": vfield,
            "boundary_normal": self.bnormal.copy(),
            "boundary_target_velocity": np.zeros_like(self.bnormal),
            "boundary_velocity": bvel,
            "cumulative_mass_source": np.zeros(4),
        }
        scalars = {
            "characteristic_length": self.characteristic_length,
            "grid_spacing": self.grid_spacing,
            "reference_velocity": self.reference_velocity,
            "contact_length_scale": self.spacing,
        }
        frame = StateFrame(
            t,
            pos,
            vel,
            mass=self.mass.copy(),
            body_id=self.label.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "method": "wcsph_laplacian_viscosity",
            "nu_low": NU_LOW,
            "nu_high": NU_HIGH,
            "substeps": self._substeps,
            "particles": int(self.N),
            "boundary_samples": int(self.M),
        }
