"""Grid-based shallow-water + rigid-body solver for frigate wave response.

Vectorised Lax-Friedrichs scheme on a staggered grid coupled to a rigid
frigate hull free to heave and pitch.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = seed

        fluid = cfg["fluid"]
        self.rho0 = float(fluid["density"])
        self.domain_min = np.asarray(fluid["domain_min"], dtype=float)
        self.domain_max = np.asarray(fluid["domain_max"], dtype=float)
        self.still_water_level = float(fluid["still_water_level"])
        self.init_vel = np.asarray(fluid["initial_velocity"], dtype=float)
        self.marker_spacing = float(fluid["marker_spacing"])

        wave = cfg["incoming_waves"]
        self.wave_amp = float(wave["amplitude"])
        self.wave_len = float(wave["wavelength"])
        self.wave_period = float(wave["period"])
        self.wave_phase = float(wave["phase_radians"])
        self.wave_omega = 2.0 * math.pi / self.wave_period
        self.wave_k = 2.0 * math.pi / self.wave_len

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.g = abs(self.gravity[2])

        ship = cfg["frigate"]
        self.ship_mass = float(ship["mass"])
        self.ship_body_id = int(ship["body_id"])
        self.ship_center0 = np.asarray(ship["initial_center"], dtype=float)
        self.ship_vel = np.asarray(ship["initial_velocity"], dtype=float).copy()
        self.ship_angvel = np.asarray(ship["initial_angular_velocity"], dtype=float).copy()

        dof = ship["degrees_of_freedom"]
        self.trans_free = np.asarray(dof["translation"], dtype=bool)
        self.rot_free = np.asarray(dof["rotation"], dtype=bool)

        self.hull_half_len = 4.35
        self.hull_max_beam = 0.92
        self.hull_bottom_offset = -1.12
        self.hull_top_local = 0.42

        Lx, Ly, Lz = 8.7, 1.6, 1.5
        self.Iyy = self.ship_mass * (Lx**2 + Lz**2) / 12.0

        # Grid
        self.dx = 0.48
        Lx_dom = self.domain_max[0] - self.domain_min[0]
        Ly_dom = self.domain_max[1] - self.domain_min[1]
        self.nx = max(int(Lx_dom / self.dx), 10)
        self.ny = max(int(Ly_dom / self.dx), 4)
        self.x_edges = np.linspace(self.domain_min[0], self.domain_max[0], self.nx + 1)
        self.y_edges = np.linspace(self.domain_min[1], self.domain_max[1], self.ny + 1)
        self.xc = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        self.yc = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        self.X, self.Y = np.meshgrid(self.xc, self.yc, indexing="ij")

        eta0 = self.still_water_level + self.wave_amp * np.sin(
            self.wave_k * self.X + self.wave_phase
        )
        self.h_sw = np.maximum(eta0 - self.domain_min[2], 0.1)
        self.u_sw = np.full_like(self.h_sw, self.init_vel[0])
        self.v_sw = np.zeros_like(self.h_sw)

        self.ship_pos = self.ship_center0.copy()
        self.ship_quat = np.array([0., 0., 0., 1.])
        self.pitch_angle = 0.0

        self.cum_ext_impulse = np.zeros(3, dtype=float)
        self.cum_inflow_mom = np.zeros(3, dtype=float)
        self.cum_outflow_mom = np.zeros(3, dtype=float)
        self.prev_time = 0.0
        self.interaction_force_z = 0.0

        self.ref_velocity = max(abs(self.init_vel[0]), self.wave_amp * self.wave_omega, 0.1)
        self.ref_momentum = self.ship_mass * self.ref_velocity

        self._build_particles_from_grid()

    # ---- Hull helpers ----
    def _hull_half_beam(self, xl):
        t = np.clip(1.0 - np.abs(xl) / self.hull_half_len, 0., 1.)
        return self.hull_max_beam * t ** 0.35

    def _hull_bottom_z(self, xl):
        return self.hull_bottom_offset + 0.16 * np.abs(xl) / self.hull_half_len

    # ---- Wave helpers ----
    def _wave_eta(self, x, t):
        return self.wave_amp * np.sin(self.wave_k * x - self.wave_omega * t + self.wave_phase)

    def _wave_u(self, x, t):
        return self.wave_amp * self.wave_omega * np.cos(
            self.wave_k * x - self.wave_omega * t + self.wave_phase
        )

    # ---- Quaternion helpers ----
    @staticmethod
    def _quat_multiply(q1, q2):
        x1, y1, z1, w1 = q1; x2, y2, z2, w2 = q2
        return np.array([w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2,
                         w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2])

    @staticmethod
    def _quat_rotate_vec(q, v):
        qv = np.array([v[0], v[1], v[2], 0.])
        qc = np.array([-q[0], -q[1], -q[2], q[3]])
        r = BenchmarkSolver._quat_multiply(BenchmarkSolver._quat_multiply(q, qv), qc)
        return r[:3]

    @staticmethod
    def _quat_from_axis_angle(axis, angle):
        s = math.sin(angle*0.5); c = math.cos(angle*0.5)
        n = axis / (np.linalg.norm(axis)+1e-30)
        return np.array([n[0]*s, n[1]*s, n[2]*s, c])

    # ---- Particles ----
    def _build_particles_from_grid(self):
        h = self.marker_spacing
        lo, hi = self.domain_min, self.domain_max
        xs = np.arange(lo[0]+0.5*h, hi[0], h)
        ys = np.arange(lo[1]+0.5*h, hi[1], h)
        zs = np.arange(lo[2]+0.5*h, min(hi[2], self.still_water_level+self.wave_amp+h), h)
        xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))

        sx, sy, sz = self.ship_pos
        xl = pts[:, 0] - sx
        hb = self._hull_half_beam(xl)
        bz = self._hull_bottom_z(xl) + sz
        tz = sz + self.hull_top_local
        inside = ((np.abs(xl) < self.hull_half_len) & (np.abs(pts[:,1]-sy) < hb)
                  & (pts[:,2] > bz) & (pts[:,2] < tz))
        pts = pts[~inside]

        mx = 8000
        if len(pts) > mx:
            pts = pts[np.linspace(0, len(pts)-1, mx, dtype=int)]

        self.fluid_pts = pts.copy()
        self.N_fluid = len(pts)
        self.N = 1 + self.N_fluid
        self.particle_mass_fluid = self.rho0 * h**3

    def _interp_to_particles(self):
        xi = np.clip(np.searchsorted(self.x_edges, self.fluid_pts[:,0])-1, 0, self.nx-1)
        yi = np.clip(np.searchsorted(self.y_edges, self.fluid_pts[:,1])-1, 0, self.ny-1)
        fx = np.clip((self.fluid_pts[:,0]-self.x_edges[xi])/self.dx, 0., 1.)
        fy = np.clip((self.fluid_pts[:,1]-self.y_edges[yi])/self.dx, 0., 1.)
        xi1 = np.minimum(xi+1, self.nx-1); yi1 = np.minimum(yi+1, self.ny-1)

        def blerp(F):
            return ((1-fx)*(1-fy)*F[xi,yi] + fx*(1-fy)*F[xi1,yi] +
                    (1-fx)*fy*F[xi,yi1] + fx*fy*F[xi1,yi1])

        return blerp(self.h_sw), blerp(self.u_sw), blerp(self.v_sw)

    # ---- Vectorised LF shallow-water step ----
    def _sw_substep(self, dt, t):
        nx, ny = self.nx, self.ny
        dx = self.dx
        g = self.g
        alpha = 0.3  # LF averaging coefficient

        h = self.h_sw
        u = self.u_sw
        v = self.v_sw

        # --- Hull coupling ---
        F_hull_z = 0.0
        T_hull_y = 0.0
        sx = self.ship_pos[0]; sz = self.ship_pos[2]
        xl = self.X - sx
        in_x = np.abs(xl) < self.hull_half_len
        hb = self._hull_half_beam(xl)
        in_y = np.abs(self.Y) < hb
        bz = self._hull_bottom_z(xl) + sz
        eta_f = self.still_water_level + self._wave_eta(self.X, t)
        wet = in_x & in_y & (bz < eta_f)
        dA = dx*dx

        if np.any(wet):
            sd = np.maximum(eta_f[wet] - bz[wet], 0.)
            F_buoy = self.rho0 * g * np.sum(sd) * dA
            u_rel = u[wet] - self.ship_vel[0]
            Cd = 0.3
            F_drag = -Cd * self.rho0 * np.sum(np.abs(u_rel)*u_rel) * dA
            arm = xl[wet]
            T_pitch = self.rho0 * g * np.sum(sd * arm) * dA
            # Wave slope contribution to pitch
            ws = self.wave_amp * self.wave_k * np.cos(
                self.wave_k * self.X[wet] - self.wave_omega*t + self.wave_phase)
            T_pitch += self.rho0 * g * np.sum(sd * arm * ws) * dA * 0.3

            F_hull_z = F_buoy - self.ship_mass * g
            T_hull_y = T_pitch

            nw = np.sum(wet)
            if nw > 0:
                decel = F_drag / (nw * self.rho0 * max(np.mean(h[wet]),0.1) * dA + 1e-12)
                u_arr = u.copy()
                u_arr[wet] -= decel * dt
                u = u_arr

        # --- Vectorised LF update (interior cells only) ---
        hu = h * u
        hv = h * v

        # Neighbour averages (interior: [1:-1, 1:-1])
        h_avg = 0.25*(h[2:,1:-1] + h[:-2,1:-1] + h[1:-1,2:] + h[1:-1,:-2])
        hu_avg = 0.25*(hu[2:,1:-1] + hu[:-2,1:-1] + hu[1:-1,2:] + hu[1:-1,:-2])
        hv_avg = 0.25*(hv[2:,1:-1] + hv[:-2,1:-1] + hv[1:-1,2:] + hv[1:-1,:-2])

        # Flux divergences
        dhdx = (hu[2:,1:-1] - hu[:-2,1:-1]) / (2*dx)
        dhdy = (hv[1:-1,2:] - hv[1:-1,:-2]) / (2*dx)

        h_safe = np.maximum(h, 0.01)
        # Momentum fluxes
        Fhu_x = (hu[2:,1:-1]**2/h_safe[2:,1:-1] + 0.5*g*h[2:,1:-1]**2 -
                 hu[:-2,1:-1]**2/h_safe[:-2,1:-1] - 0.5*g*h[:-2,1:-1]**2) / (2*dx)
        Fhu_y = (hu[1:-1,2:]*hv[1:-1,2:]/h_safe[1:-1,2:] -
                 hu[1:-1,:-2]*hv[1:-1,:-2]/h_safe[1:-1,:-2]) / (2*dx)

        Fhv_x = (hu[2:,1:-1]*hv[2:,1:-1]/h_safe[2:,1:-1] -
                 hu[:-2,1:-1]*hv[:-2,1:-1]/h_safe[:-2,1:-1]) / (2*dx)
        Fhv_y = (hv[1:-1,2:]**2/h_safe[1:-1,2:] + 0.5*g*h[1:-1,2:]**2 -
                 hv[1:-1,:-2]**2/h_safe[1:-1,:-2] - 0.5*g*h[1:-1,:-2]**2) / (2*dx)

        h_new = np.copy(h)
        hu_new = np.copy(hu)
        hv_new = np.copy(hv)

        sl = (slice(1,-1), slice(1,-1))
        h_new[sl] = (1-alpha)*h[sl] + alpha*h_avg - dt*(dhdx + dhdy)
        hu_new[sl] = (1-alpha)*hu[sl] + alpha*hu_avg - dt*(Fhu_x + Fhu_y)
        hv_new[sl] = (1-alpha)*hv[sl] + alpha*hv_avg - dt*(Fhv_x + Fhv_y)

        h_new = np.maximum(h_new, 0.01)
        u_new = hu_new / h_new
        v_new = hv_new / h_new

        mv = 10.0
        u_new = np.clip(u_new, -mv, mv)
        v_new = np.clip(v_new, -mv, mv)

        # Friction
        Cf = 0.003
        spd = np.sqrt(u_new**2 + v_new**2)
        ff = np.maximum(1.0 - Cf*spd*dt/np.maximum(h_new, 0.05), 0.5)
        u_new *= ff; v_new *= ff

        # BCs
        eta_in = self.still_water_level + self._wave_eta(self.x_edges[0], t+dt)
        h_new[0,:] = max(eta_in - self.domain_min[2], 0.1)
        u_new[0,:] = self._wave_u(self.x_edges[0], t+dt)
        v_new[0,:] = 0.0
        h_new[-1,:] = h_new[-2,:]; u_new[-1,:] = u_new[-2,:]; v_new[-1,:] = v_new[-2,:]
        v_new[:,0] = 0.0; v_new[:,-1] = 0.0

        h_new = np.nan_to_num(h_new, nan=1.0, posinf=5.0, neginf=0.1)
        u_new = np.nan_to_num(u_new, nan=0., posinf=mv, neginf=-mv)
        v_new = np.nan_to_num(v_new, nan=0., posinf=mv, neginf=-mv)

        self.h_sw = h_new; self.u_sw = u_new; self.v_sw = v_new
        return F_hull_z, T_hull_y

    # ---- Advance ----
    def advance_to(self, time: float) -> StateFrame:
        max_c = np.sqrt(self.g*np.max(self.h_sw)) + np.max(np.abs(self.u_sw)) + 1.0
        dt_max = min(0.4*self.dx/max(max_c, 0.1), 0.02)

        t = self.prev_time
        while t < time - 1e-12:
            dt = min(dt_max, time - t)
            F_z, T_y = self._sw_substep(dt, t)

            if self.trans_free[2]:
                self.ship_vel[2] += (F_z/self.ship_mass)*dt
                self.ship_vel[2] *= max(1.-1.5*dt, 0.)
                self.ship_pos[2] += self.ship_vel[2]*dt
                self.ship_pos[2] = np.clip(self.ship_pos[2],
                                            self.still_water_level-1., self.still_water_level+1.)
            if self.rot_free[1]:
                self.ship_angvel[1] += (T_y/self.Iyy)*dt
                self.ship_angvel[1] *= max(1.-2.*dt, 0.)
                self.pitch_angle += self.ship_angvel[1]*dt
                self.pitch_angle = np.clip(self.pitch_angle, -0.25, 0.25)

            self.ship_quat = self._quat_from_axis_angle(np.array([0.,1.,0.]), self.pitch_angle)
            self.cum_ext_impulse[2] += F_z*dt
            self.interaction_force_z = F_z

            mom_in = self.rho0*np.sum(self.h_sw[0,:]*np.maximum(self.u_sw[0,:],0))*self.dx**2
            self.cum_inflow_mom[0] += mom_in*dt
            mom_out = self.rho0*np.sum(self.h_sw[-1,:]*np.maximum(self.u_sw[-1,:],0))*self.dx**2
            self.cum_outflow_mom[0] += mom_out*dt

            t += dt

        self.prev_time = time
        self._build_particles_from_grid()
        h_p, u_p, v_p = self._interp_to_particles()

        N = self.N
        pos = np.zeros((N,3)); vel = np.zeros((N,3))
        mass_arr = np.zeros(N); body_id = np.zeros(N, dtype=np.int64)
        ori = np.tile([0.,0.,0.,1.], (N,1))
        av = np.zeros((N,3))
        phase = np.ones(N); sv = np.zeros(N)

        pos[0] = self.ship_pos; vel[0] = self.ship_vel
        mass_arr[0] = self.ship_mass; body_id[0] = self.ship_body_id
        ori[0] = self.ship_quat; av[0] = self.ship_angvel
        phase[0] = 0.; sv[0] = 1.

        pos[1:,0] = self.fluid_pts[:,0]; pos[1:,1] = self.fluid_pts[:,1]
        zfrac = np.clip((self.fluid_pts[:,2]-self.domain_min[2])/
                        max(self.still_water_level+self.wave_amp-self.domain_min[2],0.1), 0., 1.)
        pos[1:,2] = self.domain_min[2] + zfrac*np.maximum(h_p, 0.1)
        vel[1:,0] = u_p; vel[1:,1] = v_p
        mass_arr[1:] = self.particle_mass_fluid; phase[1:] = 1.

        B = 20
        angles = np.linspace(0, 2*np.pi, B, endpoint=False)
        bn = np.zeros((B,3)); bv = np.zeros((B,3)); bt = np.zeros((B,3))
        for bi in range(B):
            th = angles[bi]; rs = 0.8
            lp = np.array([0., rs*math.cos(th), rs*math.sin(th)-0.3])
            wp = self.ship_pos + self._quat_rotate_vec(self.ship_quat, lp)
            n = lp/(np.linalg.norm(lp)+1e-12)
            bn[bi] = self._quat_rotate_vec(self.ship_quat, n)
            rel = wp - self.ship_pos
            bv[bi] = self.ship_vel + np.cross(self.ship_angvel, rel)
            bt[bi] = bv[bi]

        G = 2
        bw = np.zeros((G,N)); bw[0,1:] = 1.; bw[1,0] = 1.
        bef = np.zeros((G,1,3))
        bef[0,0,2] = -self.rho0*self.g*np.sum(self.h_sw)*self.dx**2
        bef[1,0,2] = -self.ship_mass*self.g
        intf = np.zeros((1,3)); intf[0,2] = self.interaction_force_z

        fields = {
            "boundary_normal": bn, "boundary_target_velocity": bt,
            "boundary_velocity": bv,
            "budget_cumulative_external_impulse": self.cum_ext_impulse.reshape(1,3),
            "budget_cumulative_inflow_momentum": self.cum_inflow_mom.reshape(1,3),
            "budget_cumulative_outflow_momentum": self.cum_outflow_mom.reshape(1,3),
            "budget_domain_weights": bw, "budget_external_force": bef,
            "cumulative_mass_source": np.zeros(1), "interaction_force": intf,
        }
        attrs = {"phase": phase, "ship_visual": sv, "mass_quantity": mass_arr.copy()}
        sclrs = {"reference_momentum": self.ref_momentum,
                 "reference_velocity": self.ref_velocity,
                 "still_water_level": self.still_water_level}

        return StateFrame(time=time, position=pos, velocity=vel, orientation=ori,
                          angular_velocity=av, mass=mass_arr, body_id=body_id,
                          attributes=attrs, fields=fields, scalars=sclrs)

    def diagnostics(self) -> dict[str, object]:
        return {"seed": self.seed}
