"""Vortex-particle smoke explosion: 3D Eulerian buoyant smoke solver.

Physics: incompressible Boussinesq smoke on a staggered-free (collocated)
MAC-style grid with semi-Lagrangian advection, Jacobi pressure projection,
buoyancy from a temperature/density scalar, an initial radial blast forcing,
and no-slip ground plus free-slip outer boundaries.  Tracer particles are
advected by the diagnostic velocity field for the material-point cloud.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.duration = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])
        self.gravity = np.asarray(cfg["gravity"], float)

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, float)
        self.mass = np.array(frame0.mass, float)
        self.body_id = np.array(frame0.body_id)
        self.attributes = {k: np.array(v, float) for k, v in frame0.attributes.items()}
        self.base_scalars = {k: float(v) for k, v in frame0.scalars.items()}

        shape = tuple(cfg["sampling"]["field_grid_shape"])
        self.shape = shape
        lo = np.array([-3.0, -1.2, 0.0])
        hi = np.array([3.0, 1.2, 3.8])
        self.lo, self.hi = lo, hi
        nx, ny, nz = shape
        self.dx = np.array([(hi[i] - lo[i]) / (shape[i] - 1) for i in range(3)])
        x = np.linspace(lo[0], hi[0], nx)[:, None, None]
        y = np.linspace(lo[1], hi[1], ny)[None, :, None]
        z = np.linspace(lo[2], hi[2], nz)[None, None, :]
        self.X, self.Y, self.Z = (np.broadcast_arrays(x, y, z)[i].copy() for i in range(3))

        # scalar state (density/temperature combined; hot buoyant smoke)
        self.s = np.array(frame0.fields["transported_scalar"], float)
        self.active = np.array(frame0.fields["active_mask"], dtype=bool)

        # velocity grid (collocated, [nx,ny,nz,3])
        self.vel = np.zeros(shape + (3,))

        self.t = 0.0
        self._frame0 = frame0

        # explosion parameters
        self.blast_center = np.array([0.0, 0.0, 0.85])
        self.blast_strength = 5.5   # m/s radial impulse scale
        self.blast_radius = 0.55
        self.blast_duration = 0.55  # s of active radial forcing
        self.buoyancy = 2.6         # accel per unit scalar
        self.dissipation = 0.12     # scalar cooling rate 1/s (slow fade)
        self.vel_damping = 0.10     # mild drag 1/s

        # boundary samples (points on the ground surface)
        bx = np.linspace(-3.0, 3.0, 9)
        by = np.linspace(-1.2, 1.2, 5)
        gx, gy = np.meshgrid(bx, by, indexing="ij")
        self.bnd_pos = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
        self.bnd_normal = np.tile(np.array([0.0, 0.0, 1.0]), (self.bnd_pos.shape[0], 1))
        self.bnd_target = np.zeros_like(self.bnd_pos)

        # frozen scalars
        self.scalars = dict(self.base_scalars)
        self.scalars.update({
            "characteristic_length": float(self.base_scalars.get("characteristic_length", 1.0)),
            "grid_spacing": float(self.dx.mean()),
            "reference_velocity": 2.0,
        })

    # ------------------------------------------------------------------
    def _apply_blast(self, dt: float) -> None:
        if self.t >= self.blast_duration:
            return
        cx, cy, cz = self.blast_center
        rx = self.X - cx
        ry = self.Y - cy
        rz = self.Z - cz
        r = np.sqrt(rx * rx + ry * ry + rz * rz) + 1e-9
        w = np.exp(-(r / self.blast_radius) ** 2)
        ramp = 1.0 - self.t / self.blast_duration
        amp = self.blast_strength * ramp * w / r
        self.vel[..., 0] += amp * rx * dt
        self.vel[..., 1] += amp * ry * dt
        self.vel[..., 2] += amp * rz * dt

    def _apply_buoyancy(self, dt: float) -> None:
        # smoke rises: force along -gravity direction proportional to scalar
        g = np.linalg.norm(self.gravity)
        up = -self.gravity / (g + 1e-12)
        self.vel += (self.buoyancy * self.s * dt)[..., None] * up

    def _advect_semi_lagrangian(self, field, dt):
        # trace back
        px = self.X - self.vel[..., 0] * dt
        py = self.Y - self.vel[..., 1] * dt
        pz = self.Z - self.vel[..., 2] * dt
        return self._sample(field, px, py, pz)

    def _sample(self, field, px, py, pz):
        nx, ny, nz = self.shape
        fx = (px - self.lo[0]) / self.dx[0]
        fy = (py - self.lo[1]) / self.dx[1]
        fz = (pz - self.lo[2]) / self.dx[2]
        np.clip(fx, 0, nx - 1.0001, out=fx)
        np.clip(fy, 0, ny - 1.0001, out=fy)
        np.clip(fz, 0, nz - 1.0001, out=fz)
        i0 = fx.astype(int); j0 = fy.astype(int); k0 = fz.astype(int)
        i1 = np.minimum(i0 + 1, nx - 1); j1 = np.minimum(j0 + 1, ny - 1); k1 = np.minimum(k0 + 1, nz - 1)
        tx = fx - i0; ty = fy - j0; tz = fz - k0
        if field.ndim == 3:
            c000 = field[i0, j0, k0]; c100 = field[i1, j0, k0]
            c010 = field[i0, j1, k0]; c110 = field[i1, j1, k0]
            c001 = field[i0, j0, k1]; c101 = field[i1, j0, k1]
            c011 = field[i0, j1, k1]; c111 = field[i1, j1, k1]
        else:
            c000 = field[i0, j0, k0, :]; c100 = field[i1, j0, k0, :]
            c010 = field[i0, j1, k0, :]; c110 = field[i1, j1, k0, :]
            c001 = field[i0, j0, k1, :]; c101 = field[i1, j0, k1, :]
            c011 = field[i0, j1, k1, :]; c111 = field[i1, j1, k1, :]
        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    def _project(self, iters: int = 40) -> None:
        # Jacobi iteration on collocated grid, solving lap(p) = div(u)
        p = np.zeros(self.shape)
        d2 = 1.0 / (self.dx ** 2)
        denom = 2 * d2.sum()

        def shift(a, axis, s):
            return np.roll(a, s, axis=axis)

        div = np.zeros(self.shape)
        for axis in range(3):
            v = self.vel[..., axis]
            div += (shift(v, axis, -1) - shift(v, axis, 1)) / (2 * self.dx[axis])

        for _ in range(iters):
            nb = np.zeros(self.shape)
            for axis, dd in enumerate(d2):
                nb += dd * (shift(p, axis, 1) + shift(p, axis, -1))
            p = (nb - div) / denom
        # apply pressure gradient
        for axis in range(3):
            self.vel[..., axis] -= (shift(p, axis, -1) - shift(p, axis, 1)) / (2 * self.dx[axis])
        self._apply_bc()

    def _apply_bc(self) -> None:
        # ground (z = lo[2] index 0): no-slip
        self.vel[:, :, 0, :] = 0.0
        self.vel[:, :, 1, :] = 0.0
        # outer lateral/top faces: free-slip (zero normal component)
        self.vel[0, :, :, 0] = 0.0
        self.vel[-1, :, :, 0] = 0.0
        self.vel[:, 0, :, 1] = 0.0
        self.vel[:, -1, :, 1] = 0.0
        self.vel[:, :, -1, 2] = 0.0

    def _scalar_source(self, t: float):
        src = np.zeros(self.shape)
        if t < self.blast_duration:
            cx, cy, cz = self.blast_center
            r2 = (self.X - cx) ** 2 + (self.Y - cy) ** 2 + (self.Z - cz) ** 2
            w = np.exp(-r2 / (0.35 ** 2))
            ramp = 1.0 - t / self.blast_duration
            src = 3.0 * ramp * w
        return src

    def _step(self, dt: float) -> None:
        self._apply_blast(dt)
        self._apply_buoyancy(dt)
        # damping
        self.vel *= np.exp(-self.vel_damping * dt)
        self._apply_bc()
        # advect velocity
        self.vel = self._advect_semi_lagrangian(self.vel, dt)
        self._apply_bc()
        # project
        self._project(iters=30)
        # scalar: source + advection + dissipation
        src = self._scalar_source(self.t)
        self.s = self._advect_semi_lagrangian(self.s, dt)
        self.s += src * dt
        self.s *= np.exp(-self.dissipation * dt)
        np.clip(self.s, 0.0, None, out=self.s)
        self.t += dt

    def _advect_particles(self, dt: float) -> None:
        v = self._sample(self.vel, self.pos[:, 0], self.pos[:, 1], self.pos[:, 2])
        self.pos = self.pos + v * dt
        # keep inside domain, stick to ground
        self.pos[:, 0] = np.clip(self.pos[:, 0], self.lo[0], self.hi[0])
        self.pos[:, 1] = np.clip(self.pos[:, 1], self.lo[1], self.hi[1])
        self.pos[:, 2] = np.clip(self.pos[:, 2], self.lo[2], self.hi[2])
        self.pvel = v

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if not hasattr(self, "pvel"):
            self.pvel = np.zeros_like(self.pos)
        # substep for stability
        max_sub = self.sample_dt
        while self.t < target - 1e-12:
            dt = min(max_sub, target - self.t)
            n_sub = max(1, int(np.ceil(dt / 0.0125)))
            h = dt / n_sub
            for _ in range(n_sub):
                self._step(h)
                self._advect_particles(h)
        # boundary velocity diagnostic: sample fluid velocity at wall points
        bnd_vel = self._sample(self.vel, self.bnd_pos[:, 0], self.bnd_pos[:, 1], self.bnd_pos[:, 2])
        bnd_vel[:, 2] = np.minimum(bnd_vel[:, 2], 0.0)  # no penetration
        fields = {
            "active_mask": self.active.copy(),
            "boundary_normal": self.bnd_normal.copy(),
            "boundary_target_velocity": self.bnd_target.copy(),
            "boundary_velocity": bnd_vel,
            "scalar_source": self._scalar_source(min(self.t, self.duration)),
            "transported_scalar": self.s.copy(),
            "velocity_field": self.vel.copy(),
        }
        return StateFrame(
            time=self.t,
            position=self.pos.copy(),
            velocity=self.pvel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={k: v.copy() for k, v in self.attributes.items()},
            fields=fields,
            scalars=dict(self.scalars),
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "eulerian_buoyant_smoke_semi_lagrangian",
            "grid_shape": list(self.shape),
            "blast_strength": self.blast_strength,
            "buoyancy": self.buoyancy,
        }
