"""MLS-MPM solver for the 2D dam-break scene (x-z plane, unit depth in y).

Physically based: weakly-compressible water (equation of state pressure),
gravity, no-slip tank walls enforced on the background grid.  The liquid is
advected with the Affine-Particle-In-Cell / MLS-MPM transfer, so the column
collapses, the front runs across the floor, impacts the right wall, splashes
up, recirculates and settles while keeping a coherent free surface.
"""
from __future__ import annotations

import numpy as np

try:  # optional, only for the diagnostic signed-distance field
    from scipy.ndimage import distance_transform_edt as _edt
except Exception:  # pragma: no cover
    _edt = None

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---- background MPM grid (node centred, x-z plane) ----
    H = 0.05                       # grid spacing [m]
    X_MIN, Z_MIN = -3.05, -0.05    # grid origin
    NX, NZ = 123, 63               # node counts  (x,z in [-3.05,3.05])
    WALL_X = 3.0                   # inner tank faces
    FLOOR_Z = 0.05                 # floor top surface

    G = (0.0, -9.81)               # gravity (x, z)
    RHO0 = 1000.0
    C_SOUND = 15.0
    BULK = RHO0 * C_SOUND ** 2     # equation-of-state bulk modulus
    DT = 0.001                     # substep (CFL: c*dt/h = 0.3)

    # ---- diagnostic grid (cell centred, spans the tank interior) ----
    DG_NX, DG_NZ = 120, 60
    DG_H = 0.05
    DG_X0, DG_Z0 = -3.0, 0.0

    CHAR_LEN = 1.5                 # initial column width
    REF_VEL = float(np.sqrt(9.81 * 2.4))   # sqrt(g*H) of initial column

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        frame0 = scene.initial_state()
        pos = np.asarray(frame0.position, dtype=np.float64)
        vel = np.asarray(frame0.velocity, dtype=np.float64)
        self.mass = np.asarray(frame0.mass, dtype=np.float64).copy()
        if "sample_volume" in frame0.attributes:
            self.vol = np.asarray(frame0.attributes["sample_volume"],
                                  dtype=np.float64).copy()
        else:
            self.vol = self.mass / self.RHO0
        self.np_ = pos.shape[0]
        self.xp = np.ascontiguousarray(pos[:, [0, 2]])
        self.vp = np.ascontiguousarray(vel[:, [0, 2]])
        self.J = np.ones(self.np_, dtype=np.float64)
        self.C = np.zeros((self.np_, 2, 2), dtype=np.float64)
        self.time = 0.0

        gx = self.X_MIN + np.arange(self.NX) * self.H
        gz = self.Z_MIN + np.arange(self.NZ) * self.H
        self.solid = ((gx[:, None] <= -self.WALL_X + 1e-9)
                      | (gx[:, None] >= self.WALL_X - 1e-9)
                      | (gz[None, :] <= self.FLOOR_Z + 1e-9))
        self._init_boundary_samples()

    # ------------------------------------------------------------------
    def _init_boundary_samples(self) -> None:
        pts, nrm = [], []
        xs = np.arange(-3.0, 3.0 + 1e-9, 0.25)
        for x in xs:                                  # floor
            pts.append((x, 0.0, self.FLOOR_Z)); nrm.append((0.0, 0.0, 1.0))
        zs = np.arange(0.15, 3.0, 0.25)
        for z in zs:                                  # left wall
            pts.append((-self.WALL_X, 0.0, z)); nrm.append((1.0, 0.0, 0.0))
        for z in zs:                                  # right wall
            pts.append((self.WALL_X, 0.0, z)); nrm.append((-1.0, 0.0, 0.0))
        self.bnd_pts = np.asarray(pts, dtype=np.float64)
        self.bnd_normal = np.asarray(nrm, dtype=np.float64)
        self.bnd_target_velocity = np.zeros_like(self.bnd_pts)

    # ------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        h, inv_h = self.H, 1.0 / self.H
        m = self.mass
        # ---- particle -> grid ----
        grid_m = np.zeros((self.NX, self.NZ))
        gvx = np.zeros((self.NX, self.NZ))
        gvz = np.zeros((self.NX, self.NZ))

        scaled = (self.xp - np.array([self.X_MIN, self.Z_MIN])) * inv_h - 0.5
        base = np.floor(scaled).astype(np.int64)
        fx = scaled - base
        np.clip(base[:, 0], 0, self.NX - 3, out=base[:, 0])
        np.clip(base[:, 1], 0, self.NZ - 3, out=base[:, 1])
        fx0, fx1 = fx[:, 0], fx[:, 1]
        wx = np.stack((0.5 * (1.0 - fx0) ** 2,
                       0.75 - (fx0 - 0.5) ** 2,
                       0.5 * fx0 ** 2), axis=1)
        wz = np.stack((0.5 * (1.0 - fx1) ** 2,
                       0.75 - (fx1 - 0.5) ** 2,
                       0.5 * fx1 ** 2), axis=1)

        # equation of state pressure force (fluid: only volumetric part)
        S = -dt * 4.0 * inv_h * inv_h * self.vol * self.BULK * (self.J - 1.0)
        C = self.C
        vx, vz = self.vp[:, 0], self.vp[:, 1]

        for i in range(3):
            ni = base[:, 0] + i
            wxi = wx[:, i]
            gx = self.X_MIN + ni * h
            for j in range(3):
                nj = base[:, 1] + j
                w = wxi * wz[:, j]
                dxs = gx - self.xp[:, 0]
                dzs = (self.Z_MIN + nj * h) - self.xp[:, 1]
                mom_x = w * (m * vx
                             + m * (C[:, 0, 0] * dxs + C[:, 0, 1] * dzs)
                             + S * dxs)
                mom_z = w * (m * vz
                             + m * (C[:, 1, 0] * dxs + C[:, 1, 1] * dzs)
                             + S * dzs)
                np.add.at(grid_m, (ni, nj), w * m)
                np.add.at(gvx, (ni, nj), mom_x)
                np.add.at(gvz, (ni, nj), mom_z)

        occ = grid_m > 1e-12
        gvx = np.where(occ, gvx / np.maximum(grid_m, 1e-12), 0.0)
        gvz = np.where(occ, gvz / np.maximum(grid_m, 1e-12), 0.0)
        gvz = np.where(occ, gvz + dt * self.G[1], gvz)
        # no-slip tank walls
        gvx[self.solid] = 0.0
        gvz[self.solid] = 0.0

        # ---- grid -> particle (APIC) ----
        new_vx = np.zeros(self.np_)
        new_vz = np.zeros(self.np_)
        new_C = np.zeros((self.np_, 2, 2))
        k = 4.0 * inv_h * inv_h
        for i in range(3):
            ni = base[:, 0] + i
            wxi = wx[:, i]
            gx = self.X_MIN + ni * h
            for j in range(3):
                nj = base[:, 1] + j
                w = wxi * wz[:, j]
                dxs = gx - self.xp[:, 0]
                dzs = (self.Z_MIN + nj * h) - self.xp[:, 1]
                svx = gvx[ni, nj]
                svz = gvz[ni, nj]
                new_vx += w * svx
                new_vz += w * svz
                new_C[:, 0, 0] += k * w * svx * dxs
                new_C[:, 0, 1] += k * w * svx * dzs
                new_C[:, 1, 0] += k * w * svz * dxs
                new_C[:, 1, 1] += k * w * svz * dzs

        self.vp[:, 0] = new_vx
        self.vp[:, 1] = new_vz
        self.C = new_C
        self.xp[:, 0] += dt * new_vx
        self.xp[:, 1] += dt * new_vz

        det = ((1.0 + dt * new_C[:, 0, 0]) * (1.0 + dt * new_C[:, 1, 1])
               - dt * dt * new_C[:, 0, 1] * new_C[:, 1, 0])
        self.J = np.clip(self.J * det, 0.5, 1.5)

        # particle-level wall clamp (safety, walls already handled on grid)
        x, z = self.xp[:, 0], self.xp[:, 1]
        lo = x < -self.WALL_X
        hi = x > self.WALL_X
        bot = z < self.FLOOR_Z
        top = z > 2.99
        x[lo] = -self.WALL_X; self.vp[lo, 0] = np.maximum(self.vp[lo, 0], 0.0)
        x[hi] = self.WALL_X;  self.vp[hi, 0] = np.minimum(self.vp[hi, 0], 0.0)
        z[bot] = self.FLOOR_Z; self.vp[bot, 1] = np.maximum(self.vp[bot, 1], 0.0)
        z[top] = 2.99;         self.vp[top, 1] = np.minimum(self.vp[top, 1], 0.0)
        self.time += dt

    # ------------------------------------------------------------------
    def _diagnostics_grid(self):
        nx, nz, dh = self.DG_NX, self.DG_NZ, self.DG_H
        ix = np.clip(((self.xp[:, 0] - self.DG_X0) / dh).astype(np.int64), 0, nx - 1)
        iz = np.clip(((self.xp[:, 1] - self.DG_Z0) / dh).astype(np.int64), 0, nz - 1)
        mass_g = np.zeros((nx, nz))
        momx_g = np.zeros((nx, nz))
        momz_g = np.zeros((nx, nz))
        np.add.at(mass_g, (ix, iz), self.mass)
        np.add.at(momx_g, (ix, iz), self.mass * self.vp[:, 0])
        np.add.at(momz_g, (ix, iz), self.mass * self.vp[:, 1])
        rho = mass_g / (dh * dh)               # kg/m^3 (unit depth)
        mask = rho > 0.4 * self.RHO0
        vel = np.zeros((nx, 1, nz, 3))
        occ = mass_g > 1e-9
        vel[:, 0, :, 0][occ] = (momx_g / np.maximum(mass_g, 1e-9))[occ]
        vel[:, 0, :, 2][occ] = (momz_g / np.maximum(mass_g, 1e-9))[occ]
        if _edt is not None:
            ls = _edt(~mask, sampling=dh) - _edt(mask, sampling=dh)
        else:
            ls = np.where(mask, -dh, dh)
        return mask.astype(np.float64)[:, None, :], ls[:, None, :], vel

    # ------------------------------------------------------------------
    def _make_frame(self, t: float) -> StateFrame:
        n = self.np_
        position = np.zeros((n, 3))
        velocity = np.zeros((n, 3))
        position[:, 0] = self.xp[:, 0]
        position[:, 2] = self.xp[:, 1]
        velocity[:, 0] = self.vp[:, 0]
        velocity[:, 2] = self.vp[:, 1]

        mask, level_set, vel_field = self._diagnostics_grid()

        # velocity sampled at the wall boundary points (nearest diag cell)
        bvel = np.zeros_like(self.bnd_pts)
        ix = np.clip(((self.bnd_pts[:, 0] - self.DG_X0) / self.DG_H
                      - 0.5).astype(np.int64), 0, self.DG_NX - 1)
        iz = np.clip(((self.bnd_pts[:, 2] - self.DG_Z0) / self.DG_H
                      - 0.5).astype(np.int64), 0, self.DG_NZ - 1)
        bvel[:, 0] = vel_field[ix, 0, iz, 0]
        bvel[:, 2] = vel_field[ix, 0, iz, 2]

        frame = StateFrame(
            time=float(t),
            position=position,
            velocity=velocity,
            mass=self.mass.copy(),
            attributes={"phase_id": np.zeros(n, dtype=np.int64)},
            fields={
                "boundary_normal": self.bnd_normal.copy(),
                "boundary_target_velocity": self.bnd_target_velocity.copy(),
                "boundary_velocity": bvel,
                "liquid_interior_mask": mask,
                "liquid_level_set": level_set,
                "velocity_field": vel_field,
            },
            scalars={
                "characteristic_length": float(self.CHAR_LEN),
                "grid_spacing": float(self.DG_H),
                "liquid_phase_id": 0.0,
                "reference_velocity": float(self.REF_VEL),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 1e-9:
            dt = min(self.DT, target - self.time)
            self._substep(dt)
        return self._make_frame(target)

    def diagnostics(self) -> dict:
        speed = np.sqrt(self.vp[:, 0] ** 2 + self.vp[:, 1] ** 2)
        return {
            "n_particles": int(self.np_),
            "time": float(self.time),
            "max_particle_speed": float(speed.max()) if self.np_ else 0.0,
            "kinetic_energy": float(0.5 * np.sum(self.mass * speed ** 2)),
            "j_min": float(self.J.min()) if self.np_ else 1.0,
            "j_max": float(self.J.max()) if self.np_ else 1.0,
        }
