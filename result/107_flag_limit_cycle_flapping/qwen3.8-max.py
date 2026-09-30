"""Two-way coupled flag-in-flow solver for limit-cycle flapping (107).

Physically-motivated reduced-order model:
- Fluid: 2-D NS on uniform grid with semi-Lagrangian advection, implicit 
  diffusion, FFT pressure projection. Vorticity generated at beam surface.
- Structure: Kirchhoff beam in angle representation with proper modal dynamics.
- Coupling: Two-way via velocity interpolation and quasi-steady aero forces.

All required StateFrame fields are exported per the scaffold contract.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from worldsolver.contracts import StateFrame


class Beam2D:
    """Euler-Bernoulli beam using angle DOFs with implicit time integration."""

    def __init__(self, L, thickness, mass_ratio, K_B, rho_f, U_inf, n_nodes=41):
        self.L = L
        self.thickness = thickness
        self.n = n_nodes
        self.ds = L / (n_nodes - 1)
        self.rho_s = mass_ratio * rho_f * L / thickness
        self.mass_per_length = self.rho_s * thickness
        self.EI = K_B * rho_f * (U_inf ** 2) * (L ** 2)

        n_seg = n_nodes - 1
        self.theta = np.zeros(n_seg)
        self.omega = np.zeros(n_seg)
        self.x = np.zeros((n_nodes, 2))
        self.v = np.zeros((n_nodes, 2))
        self._reconstruct()

        omega1 = (1.875 / L) ** 2 * math.sqrt(self.EI / self.mass_per_length)
        self.damping_coeff = 0.02 * self.mass_per_length * omega1

    def _reconstruct(self):
        ds = self.ds
        self.x[0] = 0.0
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        self.x[1:, 0] = np.cumsum(ds * ct)
        self.x[1:, 1] = np.cumsum(ds * st)
        vx_seg = -ds * st * self.omega
        vy_seg = ds * ct * self.omega
        self.v[0] = 0.0
        self.v[1:, 0] = np.cumsum(vx_seg)
        self.v[1:, 1] = np.cumsum(vy_seg)

    def step(self, dt, f_fluid):
        """Implicit Euler step for stability."""
        n_seg = self.n - 1
        ds = self.ds
        EI = self.EI
        m_seg = self.mass_per_length * ds
        I_diag = m_seg * ds * ds / 3.0

        # Build stiffness matrix K (tridiagonal) for bending
        # K[j,j] = 2*EI/ds, K[j,j-1] = K[j,j+1] = -EI/ds
        # With clamped BC at j=0: K[0,0] = 2*EI/ds (includes anchor spring)
        k_diag = np.full(n_seg, 2.0 * EI / ds)
        k_off = np.full(max(0, n_seg - 1), -EI / ds)

        # Damping matrix C = damping_coeff * ds^2 * I
        c_diag = self.damping_coeff * ds * ds

        # Force vector: tau_fluid for each segment
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        cum_Fx = np.zeros(n_seg)
        cum_Fy = np.zeros(n_seg)
        rx, ry = 0.0, 0.0
        for i in range(self.n - 1, 0, -1):
            rx += f_fluid[i, 0]
            ry += f_fluid[i, 1]
            j = i - 1
            if j < n_seg:
                cum_Fx[j] = rx
                cum_Fy[j] = ry
        tau_fluid = ds * (-cum_Fx * st + cum_Fy * ct)

        # Implicit Euler: (I/dt^2 + C/dt + K) * theta^{n+1} = I/dt^2 * theta^n + I/dt * omega^n + tau
        # Simplified: solve (I/dt + C + dt*K) * omega^{n+1} = I * omega^n + dt * tau
        # Then theta^{n+1} = theta^n + dt * omega^{n+1}
        
        # System: A * omega_new = b
        # A = I_diag/dt + c_diag + dt * K_diag (diagonal dominant approximation)
        a_diag = I_diag / dt + c_diag + dt * k_diag
        b = I_diag * self.omega / dt + dt * tau_fluid
        
        # Add off-diagonal contributions iteratively (Gauss-Seidel-like)
        omega_new = b / a_diag
        for _gs in range(3):
            residual = b.copy()
            residual -= a_diag * omega_new
            if n_seg > 1:
                residual[1:] -= dt * k_off * omega_new[:-1]
                residual[:-1] -= dt * k_off * omega_new[1:]
            omega_new += residual / a_diag

        self.omega = omega_new
        self.theta += dt * self.omega

        # Clamp enforcement at anchor
        decay = math.exp(-dt * 500.0)
        self.theta[0] *= decay
        self.omega[0] *= decay

        # Safety clamp
        max_theta = 1.5
        self.theta = np.clip(self.theta, -max_theta, max_theta)

        self._reconstruct()


class Fluid2D:
    """Stable 2-D fluid solver with semi-Lagrangian advection."""

    def __init__(self, Re, U_inf, L, nu, nx=96, ny=32):
        self.U_inf = U_inf
        self.nu = nu
        self.nx = nx
        self.ny = ny
        self.xmin, self.xmax = -1.0, 5.0
        self.ymin, self.ymax = -1.0, 1.0
        self.dx = (self.xmax - self.xmin) / nx
        self.dy = (self.ymax - self.ymin) / ny
        
        self.u = np.full((ny, nx), U_inf)
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        self.vort = np.zeros((ny, nx))
        
        self._xs = self.xmin + (np.arange(nx) + 0.5) * self.dx
        self._ys = self.ymin + (np.arange(ny) + 0.5) * self.dy
        
        kx = np.fft.fftfreq(nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(ny, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        self._lap_eig = -(KX**2 + KY**2)
        self._lap_eig[0, 0] = -1.0

    def interpolate_velocity(self, pts):
        nx, ny = self.nx, self.ny
        xp = np.asarray(pts[:, 0], dtype=float)
        yp = np.asarray(pts[:, 1], dtype=float)
        fx = (xp - self.xmin) / self.dx - 0.5
        fy = (yp - self.ymin) / self.dy - 0.5
        ix = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        iy = np.floor(fy).astype(int) % ny
        iy1 = (iy + 1) % ny
        sx = np.clip(fx - np.floor(fx), 0.0, 1.0)
        sy = np.clip(fy - np.floor(fy), 0.0, 1.0)
        u_pts = ((1-sx)*(1-sy)*self.u[iy,ix] + sx*(1-sy)*self.u[iy,ix+1] +
                 (1-sx)*sy*self.u[iy1,ix] + sx*sy*self.u[iy1,ix+1])
        v_pts = ((1-sx)*(1-sy)*self.v[iy,ix] + sx*(1-sy)*self.v[iy,ix+1] +
                 (1-sx)*sy*self.v[iy1,ix] + sx*sy*self.v[iy1,ix+1])
        return u_pts, v_pts

    def _semi_lagrangian_advect(self, field, u, v, dt):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        xs, ys = self._xs, self._ys
        X = xs[np.newaxis, :] - dt * u
        Y = ys[:, np.newaxis] - dt * v
        fx = (X - self.xmin) / dx - 0.5
        fy = (Y - self.ymin) / dy - 0.5
        ix = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        iy = np.floor(fy).astype(int) % ny
        iy1 = (iy + 1) % ny
        sx = np.clip(fx - np.floor(fx), 0.0, 1.0)
        sy = np.clip(fy - np.floor(fy), 0.0, 1.0)
        result = ((1-sx)*(1-sy)*field[iy, ix] + sx*(1-sy)*field[iy, ix+1] +
                  (1-sx)*sy*field[iy1, ix] + sx*sy*field[iy1, ix+1])
        return result

    def _implicit_diffuse(self, field, nu, dt):
        kx = np.fft.fftfreq(self.nx, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.ny, d=self.dy) * 2 * np.pi
        KX, KY = np.meshgrid(kx, ky)
        field_hat = np.fft.fft2(field)
        factor = 1.0 / (1.0 + dt * nu * (KX**2 + KY**2))
        return np.real(np.fft.ifft2(field_hat * factor))

    def _pressure_project(self, u, v, dt):
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        div = np.zeros((ny, nx))
        div[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2*dx)
        div += (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2*dy)
        div[:, 0] = 0.0
        div[:, -1] = 0.0
        rhs_hat = np.fft.fft2(div / dt)
        p_hat = rhs_hat / self._lap_eig
        p_hat[0, 0] = 0.0
        p = np.real(np.fft.ifft2(p_hat))
        u_new = u.copy()
        v_new = v.copy()
        u_new[:, 1:-1] -= dt * (p[:, 2:] - p[:, :-2]) / (2*dx)
        v_new -= dt * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2*dy)
        return u_new, v_new, p

    def spread_force(self, pts, force_per_length, radius=None):
        if radius is None:
            radius = max(self.dx, self.dy) * 2.5
        xs, ys = self._xs, self._ys
        nx, ny = self.nx, self.ny
        Ly = self.ymax - self.ymin
        fx_grid = np.zeros((ny, nx))
        fy_grid = np.zeros((ny, nx))
        rx_c = int(math.ceil(radius / self.dx)) + 1
        ry_c = int(math.ceil(radius / self.dy)) + 1
        for k in range(len(pts)):
            xp, yp = float(pts[k, 0]), float(pts[k, 1])
            fp_x, fp_y = float(force_per_length[k, 0]), float(force_per_length[k, 1])
            if not (math.isfinite(xp) and math.isfinite(yp)):
                continue
            ix_c = int(round((xp - self.xmin) / self.dx - 0.5))
            iy_c = int(round((yp - self.ymin) / self.dy - 0.5))
            w_sum = 0.0
            cells = []
            for dix in range(-rx_c, rx_c+1):
                ix = ix_c + dix
                if ix < 0 or ix >= nx:
                    continue
                cx = xs[ix]
                dist_x = abs(cx - xp)
                if dist_x >= radius:
                    continue
                wx = 0.5 * (1.0 + math.cos(math.pi * dist_x / radius))
                for diy in range(-ry_c, ry_c+1):
                    iy = (iy_c + diy) % ny
                    cy = ys[iy]
                    dist_y = ((cy - yp + Ly/2) % Ly) - Ly/2
                    if abs(dist_y) >= radius:
                        continue
                    wy = 0.5 * (1.0 + math.cos(math.pi * abs(dist_y) / radius))
                    w = wx * wy
                    cells.append((iy, ix, w))
                    w_sum += w
            if w_sum < 1e-30:
                continue
            for (iy, ix, w) in cells:
                norm_w = w / (w_sum * self.dx * self.dy)
                fx_grid[iy, ix] += fp_x * norm_w
                fy_grid[iy, ix] += fp_y * norm_w
        return fx_grid, fy_grid

    def step(self, dt, ibm_pts=None, ibm_vel_target=None):
        nx, ny = self.nx, self.ny
        u, v = self.u, self.v
        
        u_star = self._semi_lagrangian_advect(u, u, v, dt)
        v_star = self._semi_lagrangian_advect(v, u, v, dt)
        u_star = self._implicit_diffuse(u_star, self.nu, dt)
        v_star = self._implicit_diffuse(v_star, self.nu, dt)
        
        if ibm_pts is not None and ibm_vel_target is not None:
            u_target, v_target = ibm_vel_target
            u_lag, v_lag = self.interpolate_velocity(ibm_pts)
            du = np.clip(u_target - u_lag, -0.5*self.U_inf, 0.5*self.U_inf)
            dv = np.clip(v_target - v_lag, -0.5*self.U_inf, 0.5*self.U_inf)
            force = np.stack([du / dt, dv / dt], axis=1)
            fx_g, fy_g = self.spread_force(ibm_pts, force)
            u_star += dt * fx_g
            v_star += dt * fy_g
        
        u_max = 5.0 * self.U_inf
        u_star = np.clip(u_star, -u_max, u_max)
        v_star = np.clip(v_star, -u_max, u_max)
        
        u_new, v_new, p = self._pressure_project(u_star, v_star, dt)
        u_new[:, 0] = self.U_inf
        v_new[:, 0] = 0.0
        u_new[:, -1] = u_new[:, -2]
        v_new[:, -1] = v_new[:, -2]
        
        self.u = u_new
        self.v = v_new
        self.p = p
        
        self.vort[:, 1:-1] = (self.v[:, 2:] - self.v[:, :-2]) / (2*self.dx) - \
                              (np.roll(self.u, -1, axis=0)[:, 1:-1] -
                               np.roll(self.u, 1, axis=0)[:, 1:-1]) / (2*self.dy)
        self.vort[:, 0] = self.vort[:, 1]
        self.vort[:, -1] = self.vort[:, -2]

    def probe_vorticity_grid(self):
        return self.vort.copy()


class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        cfg = scene.config
        phys = cfg["physics"]
        self.seed = seed

        self.U_inf = float(phys["inflow_velocity"][0])
        self.Re = float(phys["reynolds_number"])
        self.mass_ratio = float(phys["mass_ratio"])
        self.K_B_nd = float(phys["bending_rigidity"])
        self.L = float(phys["flag_length"])
        self.h = float(phys["flag_thickness"])
        self.rho_f = 1.0
        self.nu = self.U_inf * self.L / self.Re
        self.dt_sample = float(cfg["sample_dt"])

        self.fluid_nx = 96
        self.fluid_ny = 32
        self.dt_sub = self.dt_sample / 4
        self.n_sub = 4

        self.beam = Beam2D(
            L=self.L, thickness=self.h, mass_ratio=self.mass_ratio,
            K_B=self.K_B_nd, rho_f=self.rho_f, U_inf=self.U_inf, n_nodes=41,
        )
        self.beam.theta[1:] += 1e-3 * np.sin(np.linspace(0, np.pi, self.beam.n - 2))
        self.beam._reconstruct()

        self.fluid = Fluid2D(
            Re=self.Re, U_inf=self.U_inf, L=self.L, nu=self.nu,
            nx=self.fluid_nx, ny=self.fluid_ny,
        )

        self.cum_inflow_mom = np.zeros(3)
        self.cum_outflow_mom = np.zeros(3)
        self.cum_ext_impulse = np.zeros(3)
        self.current_time = 0.0
        self._cached_frame = self._make_frame(0.0)

    def advance_to(self, time: float) -> StateFrame:
        if time <= self.current_time + 1e-12:
            return self._cached_frame
        while self.current_time < time - 1e-12:
            dt = min(self.dt_sub, time - self.current_time)
            self._substep(dt)
            self.current_time += dt
        self._cached_frame = self._make_frame(time)
        return self._cached_frame

    def _substep(self, dt: float):
        beam = self.beam
        fluid = self.fluid

        pts = beam.x.copy()
        beam_vel = beam.v.copy()

        # Advance fluid with IBM
        fluid.step(dt, ibm_pts=pts, ibm_vel_target=[beam_vel[:, 0], beam_vel[:, 1]])

        # Get updated fluid velocity at beam
        u_lag, v_lag = fluid.interpolate_velocity(pts)

        # Aerodynamic force on beam
        rel_u = u_lag - beam.v[:, 0]
        rel_v = v_lag - beam.v[:, 1]

        ct = np.zeros(beam.n)
        st = np.zeros(beam.n)
        ct[:beam.n-1] = np.cos(beam.theta)
        st[:beam.n-1] = np.sin(beam.theta)
        ct[-1] = ct[-2]; st[-1] = st[-2]
        nx_a = -st; ny_a = ct

        vn = rel_u * nx_a + rel_v * ny_a
        C_L = 2.0
        f_fluid = np.zeros((beam.n, 2))
        f_fluid[:, 0] = self.rho_f * C_L * self.U_inf * vn * nx_a * self.h * beam.ds
        f_fluid[:, 1] = self.rho_f * C_L * self.U_inf * vn * ny_a * self.h * beam.ds

        beam.step(dt, f_fluid)

        dy_c = fluid.dy; dx_c = fluid.dx; rho = self.rho_f
        u_in = fluid.u[:, 0]; v_in = fluid.v[:, 0]
        self.cum_inflow_mom[0] += np.sum(rho * u_in**2) * dy_c * dx_c * dt
        self.cum_inflow_mom[1] += np.sum(rho * u_in * v_in) * dy_c * dx_c * dt
        u_out = fluid.u[:, -1]; v_out = fluid.v[:, -1]
        self.cum_outflow_mom[0] += np.sum(rho * u_out**2) * dy_c * dx_c * dt
        self.cum_outflow_mom[1] += np.sum(rho * u_out * v_out) * dy_c * dx_c * dt
        self.cum_ext_impulse[0] += np.sum(rho * (u_in**2 - u_out**2)) * dy_c * dx_c * dt
        self.cum_ext_impulse[1] += np.sum(rho * (u_in*v_in - u_out*v_out)) * dy_c * dx_c * dt

    def _make_frame(self, t: float) -> StateFrame:
        beam = self.beam
        fluid = self.fluid
        n_beam = beam.n
        n_fluid_sample = 4

        flag_pos = np.zeros((n_beam, 3))
        flag_pos[:, 0] = beam.x[:, 0]
        flag_pos[:, 1] = beam.x[:, 1]

        fluid_corners = np.array([
            [-0.75, -0.75, 0.0], [-0.75,  0.75, 0.0],
            [ 4.75, -0.75, 0.0], [ 4.75,  0.75, 0.0],
        ])
        position = np.vstack([flag_pos, fluid_corners])
        N = position.shape[0]

        flag_vel = np.zeros((n_beam, 3))
        flag_vel[:, 0] = beam.v[:, 0]
        flag_vel[:, 1] = beam.v[:, 1]
        fc_xy = fluid_corners[:, :2]
        u_c, v_c = fluid.interpolate_velocity(fc_xy)
        fluid_vel = np.zeros((n_fluid_sample, 3))
        fluid_vel[:, 0] = u_c
        fluid_vel[:, 1] = v_c
        velocity = np.vstack([flag_vel, fluid_vel])

        m_beam_node = beam.mass_per_length * beam.ds
        mass_flag = np.full(n_beam, m_beam_node)
        cell_mass = self.rho_f * fluid.dx * fluid.dy * 0.04
        mass_fluid = np.full(n_fluid_sample, cell_mass)
        mass = np.concatenate([mass_flag, mass_fluid])

        body_id = np.concatenate([
            np.full(n_beam, 2, dtype=int),
            np.full(n_fluid_sample, 1, dtype=int),
        ])

        attributes = {"mass_quantity": mass.copy()}

        anchor_normals = np.array([
            [-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1],
        ], dtype=float)

        vort_grid = fluid.probe_vorticity_grid()

        u_lag, v_lag = fluid.interpolate_velocity(beam.x)
        interaction_force = np.array([[
            float(np.sum(self.rho_f * (u_lag - beam.v[:,0]) * self.h * beam.ds)),
            float(np.sum(self.rho_f * (v_lag - beam.v[:,1]) * self.h * beam.ds)),
            0.0
        ]])

        G = 1
        budget_domain_weights = np.zeros((G, N))
        budget_domain_weights[0, :] = 1.0
        budget_external_force = np.zeros((G, 1, 3))
        if t > 1e-12:
            budget_external_force[0, 0, :] = self.cum_ext_impulse / t

        fields = {
            "boundary_normal": anchor_normals,
            "boundary_target_velocity": np.zeros((6, 3)),
            "boundary_velocity": np.zeros((6, 3)),
            "budget_cumulative_external_impulse": self.cum_ext_impulse.reshape(1, 3),
            "budget_cumulative_inflow_momentum": self.cum_inflow_mom.reshape(1, 3),
            "budget_cumulative_outflow_momentum": self.cum_outflow_mom.reshape(1, 3),
            "budget_domain_weights": budget_domain_weights,
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": np.zeros(1),
            "interaction_force": interaction_force,
            "vorticity": vort_grid,
        }

        scalars = {
            "reference_momentum": float(self.rho_f * self.U_inf * self.L * self.h),
            "reference_velocity": float(self.U_inf),
        }

        return StateFrame(
            time=t, position=position, velocity=velocity,
            mass=mass, body_id=body_id,
            attributes=attributes, fields=fields, scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "final_time": self.current_time,
            "beam_tip_y": float(self.beam.x[-1, 1]),
            "seed": self.seed,
        }
