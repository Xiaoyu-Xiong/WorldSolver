"""Physically-based buoyant smoke solver (Fedkiw/Stam stable-fluids style).

Implements an incompressible Euler smoke simulation on a collocated grid:
semi-Lagrangian advection, buoyancy forcing (temperature + density),
vorticity confinement (vortex preservation), Jacobi pressure projection
(divergence-free), continuous source injection, density decay and cooling.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---- tunable physical parameters -----------------------------------
    H = 0.12                 # uniform grid spacing (m)
    BUOY = 2.5               # buoyancy acceleration scale
    BETA_T = 1.0             # temperature -> upward buoyancy
    ALPHA_D = 0.2            # density -> mild downward (negative) buoyancy
    COOL = 0.4               # temperature cooling rate (1/s)
    VORT_EPS = 5.0           # vorticity confinement strength
    EMIT_RATE = 3.0          # density/temperature injection rate (1/s)
    VEL_GAIN = 6.0           # momentum injection gain at source
    JACOBI = 50              # pressure projection iterations

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        dom = cfg["domain"]
        self.seed = int(seed)
        self.low = np.asarray(dom["bounds_min"], dtype=np.float64)
        self.high = np.asarray(dom["bounds_max"], dtype=np.float64)
        h = float(self.H)
        self.h = h
        ext = self.high - self.low
        self.nx, self.ny, self.nz = (int(round(ext[i] / h)) for i in range(3))
        shape = (self.nx, self.ny, self.nz)
        self.shape = shape
        self.vol = h ** 3

        # Cell-centre world coordinates.
        axes = [self.low[i] + (np.arange(shape[i]) + 0.5) * h for i in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing="ij")
        self.coords = (X, Y, Z)

        ic = cfg["initial_conditions"]
        src = ic["smoke_source"]
        self.center = np.asarray(src["center"], dtype=np.float64)
        self.src_vel = np.asarray(src["initial_velocity"], dtype=np.float64)
        self.peak_density = float(src["peak_density"])
        self.temp_excess = float(src["temperature_excess"])
        radius = float(src["radius"])
        half_h = float(src["half_height"])

        # Static Gaussian source/emitter envelope.
        radial2 = ((X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2) / radius ** 2
        vert2 = ((Z - self.center[2]) ** 2) / half_h ** 2
        self.g_src = np.exp(-2.5 * (radial2 + vert2))
        self.g_src[self.g_src < 1.0e-4] = 0.0

        # Emitter disk near floor keeps the plume supplied (sustained source).
        disk_radial2 = ((X - self.center[0]) ** 2 + (Y - self.center[1]) ** 2) / radius ** 2
        emit = np.exp(-2.5 * disk_radial2) * np.exp(-2.5 * (Z - 0.06) ** 2 / 0.08 ** 2)
        emit[emit < 1.0e-3] = 0.0
        self.emit_mask = emit

        # Initial density / temperature blob + seeded velocity with perturbation.
        rho0 = self.peak_density * self.g_src.copy()
        self.rho = rho0
        self.temp = self.temp_excess * self.g_src.copy()
        u = np.zeros(shape + (3,), dtype=np.float64)
        env = self.g_src[..., None]
        u += env * self.src_vel
        amp = float(ic.get("perturbation_amplitude", 0.0))
        phase = (self.seed % 360) * np.pi / 180.0
        u[..., 0] += amp * self.g_src * np.sin(2 * np.pi * Y / ext[1] + phase)
        u[..., 1] += amp * self.g_src * np.sin(2 * np.pi * X / ext[0] - phase)
        self.u = u

        self.half_life = float(ic.get("density_half_life", 6.0))
        self.decay = 0.5 ** (1.0 / self.half_life)  # per-second factor base

        # Prescribed scalar source field (density added per second).
        self.scalar_source = self.EMIT_RATE * self.emit_mask

        self.t = 0.0
        self.injected_mass = 0.0
        self.removed_mass = 0.0
        self.char_length = float(ext[2])  # domain height as normalization

    # ---- helpers ---------------------------------------------------------
    def _interp_weights(self, u, dt):
        h = self.h
        xd = self.coords[0] - dt * u[..., 0]
        yd = self.coords[1] - dt * u[..., 1]
        zd = self.coords[2] - dt * u[..., 2]
        ix = np.clip((xd - self.low[0]) / h - 0.5, 0, self.nx - 1)
        iy = np.clip((yd - self.low[1]) / h - 0.5, 0, self.ny - 1)
        iz = np.clip((zd - self.low[2]) / h - 0.5, 0, self.nz - 1)
        x0 = np.floor(ix).astype(np.int32); x1 = np.minimum(x0 + 1, self.nx - 1)
        y0 = np.floor(iy).astype(np.int32); y1 = np.minimum(y0 + 1, self.ny - 1)
        z0 = np.floor(iz).astype(np.int32); z1 = np.minimum(z0 + 1, self.nz - 1)
        tx = ix - x0; ty = iy - y0; tz = iz - z0
        return x0, x1, y0, y1, z0, z1, tx, ty, tz

    def _sample(self, F, w):
        x0, x1, y0, y1, z0, z1, tx, ty, tz = w
        c000 = F[x0, y0, z0]; c100 = F[x1, y0, z0]
        c010 = F[x0, y1, z0]; c110 = F[x1, y1, z0]
        c001 = F[x0, y0, z1]; c101 = F[x1, y0, z1]
        c011 = F[x0, y1, z1]; c111 = F[x1, y1, z1]
        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        return c0 * (1 - tz) + c1 * tz

    def _advect_vector(self, u, dt):
        w = self._interp_weights(u, dt)
        out = np.empty_like(u)
        for c in range(3):
            out[..., c] = self._sample(u[..., c], w)
        return out

    def _apply_vel_bc(self, u):
        # Closed free-slip walls: zero normal component. Open outflow: leave.
        u[0, :, :, 0] = 0.0            # x_min closed
        u[:, 0, :, 1] = 0.0            # y_min closed
        u[:, -1, :, 1] = 0.0           # y_max closed
        u[:, :, 0, 2] = 0.0            # z_min (floor) closed
        # x_max and z_max are open_outflow -> no normal-velocity clamp.

    def _project(self, u, dt):
        h = self.h
        # Divergence via central differences.
        dudx = np.gradient(u[..., 0], h, axis=0)
        dvdy = np.gradient(u[..., 1], h, axis=1)
        dwdz = np.gradient(u[..., 2], h, axis=2)
        div = dudx + dvdy + dwdz
        rhs = div / dt
        p = np.zeros_like(div)
        h2 = h * h
        for _ in range(self.JACOBI):
            pp = np.pad(p, 1, mode="edge")   # Neumann (free-slip) ghosts
            nb = (pp[:-2, 1:-1, 1:-1] + pp[2:, 1:-1, 1:-1] +
                  pp[1:-1, :-2, 1:-1] + pp[1:-1, 2:, 1:-1] +
                  pp[1:-1, 1:-1, :-2] + pp[1:-1, 1:-1, 2:])
            p = (nb - h2 * rhs) / 6.0
            p[-1, :, :] = 0.0    # x_max open -> Dirichlet
            p[:, :, -1] = 0.0    # z_max open -> Dirichlet
        p -= p.mean()
        gx = np.gradient(p, h, axis=0)
        gy = np.gradient(p, h, axis=1)
        gz = np.gradient(p, h, axis=2)
        u[..., 0] -= dt * gx
        u[..., 1] -= dt * gy
        u[..., 2] -= dt * gz
        return u

    def _vorticity_confinement(self, u):
        h = self.h
        ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
        wx = np.gradient(uz, h, axis=1) - np.gradient(uy, h, axis=2)
        wy = np.gradient(ux, h, axis=2) - np.gradient(uz, h, axis=0)
        wz = np.gradient(uy, h, axis=0) - np.gradient(ux, h, axis=1)
        mag = np.sqrt(wx * wx + wy * wy + wz * wz) + 1e-12
        nx_ = np.gradient(mag, h, axis=0)
        ny_ = np.gradient(mag, h, axis=1)
        nz_ = np.gradient(mag, h, axis=2)
        nrm = np.sqrt(nx_ ** 2 + ny_ ** 2 + nz_ ** 2) + 1e-12
        Nx = nx_ / nrm; Ny = ny_ / nrm; Nz = nz_ / nrm
        fx = Ny * wz - Nz * wy
        fy = Nz * wx - Nx * wz
        fz = Nx * wy - Ny * wx
        s = self.VORT_EPS * h
        return np.stack((fx, fy, fz), axis=-1) * s

    # ---- one simulation step --------------------------------------------
    def _step(self, dt):
        # 1. Continuous source injection (mass, heat, momentum).
        add_rho = self.EMIT_RATE * self.emit_mask * dt
        self.rho += add_rho
        self.temp += self.EMIT_RATE * self.emit_mask * dt
        self.injected_mass += float(add_rho.sum()) * self.vol
        m = (self.emit_mask * dt * self.VEL_GAIN)[..., None]
        self.u += m * (self.src_vel - self.u)

        # 2. Forces: buoyancy + vorticity confinement.
        f_buoy = np.zeros_like(self.u)
        f_buoy[..., 2] = self.BUOY * (self.BETA_T * self.temp - self.ALPHA_D * self.rho)
        self.u += dt * f_buoy
        self.u += dt * self._vorticity_confinement(self.u)
        self._apply_vel_bc(self.u)

        # 3. Advection (semi-Lagrangian, unconditionally stable).
        self.u = self._advect_vector(self.u, dt)
        w = self._interp_weights(self.u, dt)
        self.rho = self._sample(self.rho, w)
        self.temp = self._sample(self.temp, w)
        self._apply_vel_bc(self.u)

        # 3b. Open-outflow escape: let scalars leave through x_max / z_max
        # instead of piling up in the clamped semi-Lagrangian boundary cells.
        wz = np.clip(self.u[:, :, -1, 2], 0.0, None) * dt / self.h
        wz = np.clip(wz, 0.0, 1.0)
        self.rho[:, :, -1] *= (1.0 - wz)
        self.temp[:, :, -1] *= (1.0 - wz)
        wx = np.clip(self.u[-1, :, :, 0], 0.0, None) * dt / self.h
        wx = np.clip(wx, 0.0, 1.0)
        self.rho[-1, :, :] *= (1.0 - wx)
        self.temp[-1, :, :] *= (1.0 - wx)

        # 4. Dissipation: density decay + temperature cooling.
        rho_before = self.rho.sum()
        dec = self.decay ** dt
        self.rho *= dec
        self.temp *= np.exp(-self.COOL * dt)
        self.temp[self.temp < 1e-4] = 0.0
        self.removed_mass += float(rho_before - self.rho.sum()) * self.vol * (1.0)

        # 5. Pressure projection -> divergence-free velocity.
        self.u = self._project(self.u, dt)
        self._apply_vel_bc(self.u)

        self.t += dt

    # ---- frame assembly --------------------------------------------------
    def _frame(self, time):
        rho = self.rho
        total = float(rho.sum())
        if total > 1e-9:
            cx = float((rho * self.coords[0]).sum()) / total
            cy = float((rho * self.coords[1]).sum()) / total
            cz = float((rho * self.coords[2]).sum()) / total
            vx = float((rho * self.u[..., 0]).sum()) / total
            vy = float((rho * self.u[..., 1]).sum()) / total
            vz = float((rho * self.u[..., 2]).sum()) / total
        else:
            cx, cy, cz = self.center
            vx = vy = vz = 0.0
        position = np.array([[cx, cy, cz]], dtype=np.float64)
        velocity = np.array([[vx, vy, vz]], dtype=np.float64)
        mass_kg = total * self.vol
        mass = np.array([mass_kg], dtype=np.float64)
        cum_src = np.array([self.injected_mass - self.removed_mass], dtype=np.float64)
        active = np.ones(self.shape, dtype=np.float64)
        fields = {
            "density": rho.copy(),
            "transported_scalar": rho.copy(),
            "velocity_field": self.u.copy(),
            "active_mask": active,
            "scalar_source": self.scalar_source.copy(),
            "cumulative_mass_source": cum_src,
        }
        attributes = {
            "mass_quantity": mass.copy(),
            "density": np.array([self.peak_density], dtype=np.float64),
            "temperature_excess": np.array([self.temp_excess], dtype=np.float64),
        }
        scalars = {
            "characteristic_length": float(self.char_length),
            "grid_spacing": float(self.h),
            "total_mass": mass_kg,
            "time": float(time),
        }
        frame = StateFrame(
            float(time), position, velocity, mass=mass,
            attributes=attributes, fields=fields, scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        time = float(time)
        dt = 0.02
        # Sub-step toward the requested sample time.
        guard = 0
        while self.t < time - 1e-9 and guard < 100000:
            step = min(dt, time - self.t)
            self._step(step)
            guard += 1
        return self._frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "buoyant_smoke_semi_lagrangian",
            "grid_shape": [self.nx, self.ny, self.nz],
            "grid_spacing": float(self.h),
            "time": float(self.t),
            "injected_mass": float(self.injected_mass),
            "removed_mass": float(self.removed_mass),
        }
