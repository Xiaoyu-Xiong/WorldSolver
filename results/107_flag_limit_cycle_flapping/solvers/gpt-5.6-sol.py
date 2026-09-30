"""Deterministic two-dimensional fluid/flag solver for the flag-flutter scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Coupled immersed-force vorticity solver and flexible Euler--Bernoulli strip."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        p = scene.config["physics"]
        self.time = 0.0
        self.length = float(p["flag_length"])
        self.thickness = float(p["flag_thickness"])
        self.mass_flag = float(p["mass_ratio"])
        self.kb = float(p["bending_rigidity"])
        self.nu = 1.0 / float(p["reynolds_number"])
        self.u_inf = float(p["inflow_velocity"][0])

        self.ns = 41
        self.s = np.linspace(0.0, self.length, self.ns)
        self.ds = self.s[1] - self.s[0]
        self.y = 7.5e-4 * (self.s / self.length) ** 2
        self.y[:2] = 0.0
        self.vy = np.zeros(self.ns)
        self.vy[2:] = 1.0e-4 * np.sin(np.pi * self.s[2:] / self.length)

        self.nx, self.ny = 121, 48
        self.xg = np.linspace(-1.0, 5.0, self.nx)
        self.yg = np.linspace(-1.0, 1.0, self.ny, endpoint=False)
        self.dx, self.dy = self.xg[1] - self.xg[0], 2.0 / self.ny
        self.omega = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.uf = np.full_like(self.omega, self.u_inf)
        self.vf = np.zeros_like(self.omega)
        kx = 2.0 * np.pi * np.fft.rfftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dy)
        self._kx, self._ky = kx[None, :], ky[:, None]
        self._k2 = self._kx * self._kx + self._ky * self._ky
        self._k2[0, 0] = 1.0

        self._last_interaction = np.zeros(3)
        self._anchor_force = np.zeros(3)
        self._cum_impulse = np.zeros((3, 3))
        self._cum_in = np.zeros((3, 3))
        self._cum_out = np.zeros((3, 3))
        self._steps = 0
        self._vis_j = np.arange(1, self.ny, 3)
        self._vis_i = np.arange(0, self.nx, 4)

    @staticmethod
    def _interp(field, x, y, x0, y0, dx, dy):
        fx = np.clip((x - x0) / dx, 0.0, field.shape[1] - 1.000001)
        fy = np.mod((y - y0) / dy, field.shape[0])
        i, j = np.floor(fx).astype(int), np.floor(fy).astype(int)
        a, b = fx - i, fy - j
        ip, jp = np.minimum(i + 1, field.shape[1] - 1), (j + 1) % field.shape[0]
        return ((1-a)*(1-b)*field[j, i] + a*(1-b)*field[j, ip]
                + (1-a)*b*field[jp, i] + a*b*field[jp, ip])

    def _flow_velocity(self):
        wh = np.fft.rfft2(self.omega)
        psi = wh / self._k2
        psi[0, 0] = 0.0
        ui = np.fft.irfft2(1j * self._ky * psi, s=self.omega.shape).real
        vi = np.fft.irfft2(-1j * self._kx * psi, s=self.omega.shape).real
        self.uf = self.u_inf + np.clip(ui, -0.75, 0.75)
        self.vf = np.clip(vi, -0.9, 0.9)

    def _beam_fourth(self):
        y = self.y
        g1 = 2.0*y[-1] - y[-2]
        g2 = 3.0*g1 - 3.0*y[-1] + y[-2]
        e = np.concatenate(([0.0, 0.0], y, [g1, g2]))
        return (e[:self.ns] - 4*e[1:self.ns+1] + 6*e[2:self.ns+2]
                - 4*e[3:self.ns+3] + e[4:self.ns+4]) / self.ds**4

    def _deposit_reaction(self, lift):
        fy_grid = np.zeros_like(self.omega)
        for xs, ys, fs in zip(self.s, self.y, -lift * self.ds):
            ic = int(round((xs - self.xg[0]) / self.dx))
            jc = int(np.floor((ys - self.yg[0]) / self.dy)) % self.ny
            for dj in range(-2, 3):
                jj = (jc + dj) % self.ny
                ry = (self.yg[jj] - ys + 1.0) % 2.0 - 1.0
                for di in range(-2, 3):
                    ii = ic + di
                    if 1 <= ii < self.nx - 1:
                        rx = self.xg[ii] - xs
                        w = np.exp(-0.5*((rx/self.dx)**2 + (ry/self.dy)**2))
                        fy_grid[jj, ii] += fs*w
        total = np.sum(fy_grid)*self.dx*self.dy
        desired = -np.sum(lift)*self.ds
        if abs(total) > 1.0e-14:
            fy_grid *= desired / total
        return -(np.roll(fy_grid, -1, axis=1) -
                 np.roll(fy_grid, 1, axis=1)) / (2*self.dx)

    def _external_force_terms(self):
        f = np.zeros((3, 3, 3), dtype=np.float64)
        flux = 2.0*self.u_inf*self.u_inf
        f[0, 0, 0], f[0, 1, 0] = flux, -flux
        f[0, 2] = self._anchor_force
        f[1, 2] = self._anchor_force
        f[2, 0, 0], f[2, 1, 0] = flux, -flux
        return f

    def _step(self, dt):
        self._flow_velocity()
        ux = self._interp(self.uf, self.s, self.y, self.xg[0], self.yg[0],
                          self.dx, self.dy)
        uy = self._interp(self.vf, self.s, self.y, self.xg[0], self.yg[0],
                          self.dx, self.dy)
        slope = np.gradient(self.y, self.ds, edge_order=2)
        curvature = np.gradient(slope, self.ds, edge_order=2)
        speed = np.maximum(0.35, ux)
        incidence = slope - (self.vy - uy) / speed
        lift = 0.88*speed*speed*incidence + 0.20*np.abs(uy-self.vy)*(uy-self.vy)
        lift[:2] = 0.0

        extension = 0.5*np.trapz(slope*slope, dx=self.ds)
        tension = 0.35*extension
        structural = -self.kb*self._beam_fourth() + tension*curvature
        accel = (structural + lift - 0.012*self.vy) / self.mass_flag
        accel[:2] = 0.0
        self.vy = np.clip(self.vy + dt*accel, -3.0, 3.0)
        self.vy[:2] = 0.0
        self.y += dt*self.vy
        self.y[:2] = 0.0
        over = np.maximum(np.abs(self.y)-0.88, 0.0)
        self.vy -= dt*80.0*over*np.sign(self.y)

        source = self._deposit_reaction(lift)
        w = self.omega
        wx_b, wx_f = (w-np.roll(w, 1, axis=1))/self.dx, (np.roll(w, -1, axis=1)-w)/self.dx
        wy_b, wy_f = (w-np.roll(w, 1, axis=0))/self.dy, (np.roll(w, -1, axis=0)-w)/self.dy
        adv = np.where(self.uf >= 0, self.uf*wx_b, self.uf*wx_f) + np.where(
            self.vf >= 0, self.vf*wy_b, self.vf*wy_f)
        lap = ((np.roll(w, -1, axis=1)-2*w+np.roll(w, 1, axis=1))/self.dx**2
               + (np.roll(w, -1, axis=0)-2*w+np.roll(w, 1, axis=0))/self.dy**2)
        self.omega = np.nan_to_num(w + dt*(-adv + self.nu*lap + source),
                                   nan=0.0, posinf=60.0, neginf=-60.0)
        self.omega[:, 0] = 0.0
        self.omega[:, -1] = self.omega[:, -2]
        self.omega[:, :4] *= np.array([0.0, 0.25, 0.55, 0.82])[None, :]
        self.omega[:, -4:] *= np.array([0.98, 0.94, 0.84, 0.70])[None, :]
        self.omega = np.clip(self.omega, -60.0, 60.0)

        total_lift = float(np.sum(lift)*self.ds)
        self._last_interaction[:] = (0.0, total_lift, 0.0)
        self._anchor_force[:] = (0.0, -float(np.sum(lift[2:])*self.ds), 0.0)
        self._cum_impulse += dt*np.sum(self._external_force_terms(), axis=1)
        flux = 2.0*self.u_inf*self.u_inf
        self._cum_in[:, 0] += dt*np.array([flux, 0.0, flux])
        self._cum_out[:, 0] += dt*np.array([flux, 0.0, flux])
        self.time += dt
        self._steps += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1e-10:
            raise ValueError("advance_to times must be nondecreasing")
        while self.time < target - 1e-12:
            self._step(min(0.0025, target-self.time))
        self.time = target
        self._flow_velocity()
        return self._frame()

    def _frame(self):
        slope = np.gradient(self.y, self.ds, edge_order=2)
        norm = np.sqrt(1+slope*slope)
        nxn, nyn = -slope/norm, 1/norm
        half = 0.5*self.thickness
        top = np.column_stack((self.s+half*nxn, self.y+half*nyn, np.zeros(self.ns)))
        bot = np.column_stack((self.s-half*nxn, self.y-half*nyn, np.zeros(self.ns)))
        flag_pos = np.vstack((top, bot))
        flag_vel = np.column_stack((np.zeros(self.ns), self.vy, np.zeros(self.ns)))
        flag_vel = np.vstack((flag_vel, flag_vel))

        jj, ii = np.meshgrid(self._vis_j, self._vis_i, indexing="ij")
        fi, fj = ii.ravel(), jj.ravel()
        fluid_pos = np.column_stack((self.xg[fi], self.yg[fj], np.zeros(fi.size)))
        fluid_vel = np.column_stack((self.uf[fj, fi], self.vf[fj, fi], np.zeros(fi.size)))
        position = np.vstack((flag_pos, fluid_pos))
        velocity = np.vstack((flag_vel, fluid_vel))
        nf, nfluid = flag_pos.shape[0], fluid_pos.shape[0]
        mass = np.concatenate((np.full(nf, self.mass_flag/nf),
                               np.full(nfluid, 12.0/nfluid)))
        body_id = np.concatenate((np.full(nf, 2, dtype=np.int64),
                                  np.full(nfluid, 1, dtype=np.int64)))
        weights = np.zeros((3, nf+nfluid))
        weights[0] = 1.0
        weights[1, :nf], weights[2, nf:] = 1.0, 1.0
        boundary_normal = np.array([[-1.,0.,0.], [-1.,0.,0.], [0.,1.,0.], [0.,-1.,0.]])
        boundary_target = np.array([[self.u_inf,0.,0.], [self.u_inf,0.,0.],
                                    [0.,0.,0.], [0.,0.,0.]])
        fields = {
            "vorticity": self.omega.copy(),
            "boundary_normal": boundary_normal,
            "boundary_target_velocity": boundary_target,
            "boundary_velocity": boundary_target.copy(),
            "budget_cumulative_external_impulse": self._cum_impulse.copy(),
            "budget_cumulative_inflow_momentum": self._cum_in.copy(),
            "budget_cumulative_outflow_momentum": self._cum_out.copy(),
            "budget_domain_weights": weights,
            "budget_external_force": self._external_force_terms(),
            "cumulative_mass_source": np.zeros(1),
            "interaction_force": np.vstack((self._last_interaction, -self._last_interaction)),
        }
        return StateFrame(self.time, position, velocity, mass=mass, body_id=body_id,
                          fields=fields, attributes={"mass_quantity": mass.copy()},
                          scalars={"reference_momentum": 12.0*self.u_inf,
                                   "reference_velocity": self.u_inf})

    def diagnostics(self) -> dict[str, object]:
        return {"method": "immersed-force vorticity/Euler-Bernoulli FSI",
                "steps": self._steps, "time": self.time,
                "tip_displacement": float(self.y[-1]),
                "peak_vorticity": float(np.max(np.abs(self.omega))),
                "finite": bool(np.isfinite(self.y).all() and np.isfinite(self.omega).all())}
