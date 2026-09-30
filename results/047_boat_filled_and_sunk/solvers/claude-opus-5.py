"""Physically-based solver for 047_boat_filled_and_sunk.

No external physics engine is used: the model, discretization and time
integration below are implemented directly on NumPy.

Physical model
--------------
External pool
    Depth-averaged (shallow-water) equations for the free-surface elevation
    ``eta(x, y)`` on a staggered MAC grid covering the frozen tank footprint.
    The height update is written in flux form, so water volume changes only
    through explicit source terms (inlet deposition and hull leakage).  The
    boat enters the continuity equation through the time derivative of its
    depth-integrated displacement, which is how a floating/submerged body
    forces a depth-averaged free surface.

Boat
    Rigid body with three translational degrees of freedom plus roll and
    pitch, integrated semi-implicitly.  Loads: gravity, hydrostatic buoyancy
    from the hull displacement curve ``V(d)``, the weight of the water
    currently retained inside the hull, quadratic hydrodynamic drag with
    added mass, jet momentum flux, and penalty contact with the tank floor
    and walls (Coulomb friction on the floor).

Interior water
    Volume-conserving cavity model.  Mass enters where inlet parcels land
    inside the gunwales and is exchanged with the pool through a
    bidirectional broad-crested weir over the rim, which reproduces both
    flooding once the rim goes awash and leakage back out during
    submergence.  The interior free-surface height follows by inverting the
    cavity area curve, so the visible internal level is a function of the
    solved interior volume.  The cavity capacity is calibrated against the
    outer displacement volume and the shell volume, so buoyancy and the
    contained mass stay geometrically consistent.

Inlet
    Prescribed volumetric flow emitted as ballistic parcels.  A parcel
    deposits its whole mass into exactly one bucket (hull interior, hull
    shell run-off, or pool), so the total mass equals the initial mass plus
    the cumulative prescribed source at every sample.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.g = float(abs(self.gravity[2]))

        dom = cfg["fluid_domain"]
        self.bmin = np.asarray(dom["bounds_min"], dtype=float)
        self.bmax = np.asarray(dom["bounds_max"], dtype=float)
        self.rho = float(dom["water_density"])
        self.eta0 = float(dom["initial_free_surface_z"])

        boat = cfg["boat"]
        self.m_boat = float(boat["mass"])
        self.p = np.asarray(boat["initial_center"], dtype=float).copy()
        self.v = np.zeros(3)
        self.roll = 0.0
        self.pitch = 0.0
        self.wroll = 0.0
        self.wpitch = 0.0
        self.shell_thickness = float(boat["thin_shell_thickness"])
        draft0 = float(boat["target_initial_draft"])

        inlet = cfg["inlet"]
        self.inlet_c = np.asarray(inlet["center"], dtype=float)
        self.inlet_v = np.asarray(inlet["velocity"], dtype=float)
        self.inlet_Q = float(inlet["volumetric_flow_rate"])
        self.inlet_t0, self.inlet_t1 = (float(x) for x in inlet["active_interval"])
        speed_in = max(float(np.linalg.norm(self.inlet_v)), 1e-6)
        # jet radius consistent with Q = pi r^2 |v| (stream is thinner than the nozzle)
        self.inlet_r = float(min(float(inlet["radius"]), np.sqrt(self.inlet_Q / (np.pi * speed_in))))

        # ---- hull geometry, taken from assets/open_boat.obj -----------------
        # mesh bbox: x [-1.42,1.42], y [-0.88,0.88], z [0.02,0.88];
        # the frozen transform puts the mesh z-midpoint at initial_center[2].
        self.keel_off = -0.45          # keel below the reported body origin
        self.rim_off = 0.43            # gunwale top above the body origin
        self.floor_off = -0.14         # interior cavity floor above the origin
        self.hull_depth = self.rim_off - self.keel_off       # 0.88 m
        self.hull_a = 1.30             # outer footprint semi-length
        self.hull_b = 0.74             # outer footprint semi-width

        # Outer displacement curve V(d) = K d^q, calibrated so the frozen mass
        # floats at the frozen target draft and the reserve buoyancy at rim
        # immersion is exhausted by the water the inlet can deliver.
        self.v_draft0 = self.m_boat / self.rho            # 0.082 m^3
        self.v_full = 0.329                               # displacement at rim immersion
        self.disp_q = float(
            np.log(self.v_full / self.v_draft0) / np.log(self.hull_depth / draft0)
        )
        self.disp_K = self.v_draft0 / draft0 ** self.disp_q
        self.d_full = self.hull_depth

        # Thin shell material volume; the shell is denser than water, which is
        # what lets a fully flooded hull sink (paper: heavier than water but
        # initially buoyant).
        self.shell_volume = 0.035
        self.cav_capacity = float(self.v_full - self.shell_volume)   # 0.294 m^3

        # ---- interior cavity area curve, rescaled to the exact capacity ----
        self.cav_depth = self.rim_off - self.floor_off      # 0.57 m
        self._a0, self._a1 = 0.60, 1.05
        self._b0, self._b1 = 0.30, 0.58
        self.cav_scale = 1.0
        nz = 161
        zeta = np.linspace(0.0, self.cav_depth, nz)
        raw = float(np.trapezoid(np.pi * self._cav_a(zeta) * self._cav_b(zeta), zeta))
        self.cav_scale = float(np.sqrt(self.cav_capacity / raw))
        area = np.pi * self._cav_a(zeta) * self._cav_b(zeta)
        dz = zeta[1] - zeta[0]
        self.cav_z = zeta
        self.cav_vol = np.concatenate(
            ([0.0], np.cumsum(0.5 * (area[1:] + area[:-1]) * dz))
        )
        self.cav_cap = float(self.cav_vol[-1])
        self.rim_perimeter = float(
            np.pi * 1.5 * (self._cav_a(self.cav_depth) + self._cav_b(self.cav_depth))
        )

        # ---- interior marker pattern, sorted by height so the visible level
        #      is a monotone function of the solved interior volume ----------
        self.n_int = 2200
        amax = float(self._cav_a(self.cav_depth))
        bmax = float(self._cav_b(self.cav_depth))
        collected = np.zeros((0, 3))
        while len(collected) < self.n_int:
            n = 4 * self.n_int
            x = self.rng.uniform(-amax, amax, n)
            y = self.rng.uniform(-bmax, bmax, n)
            z = self.rng.uniform(0.0, self.cav_depth, n)
            keep = (x / self._cav_a(z)) ** 2 + (y / self._cav_b(z)) ** 2 <= 1.0
            collected = np.vstack((collected, np.stack((x[keep], y[keep], z[keep]), axis=1)))
        collected = collected[: self.n_int]
        order = np.argsort(collected[:, 2], kind="stable")
        collected = collected[order]
        self.int_zeta = np.ascontiguousarray(collected[:, 2])
        self.int_local = np.column_stack(
            (collected[:, 0], collected[:, 1], self.floor_off + collected[:, 2])
        )
        self.int_park = np.array([0.0, 0.0, -0.12])   # hidden inside the tank floor slab

        # ---- pool markers reuse the frozen scene sampling -------------------
        init = scene.initial_state()
        region = np.asarray(init.attributes["water_region"])
        pool = np.asarray(init.position, dtype=float)[region == 1]
        self.pool_pos = np.ascontiguousarray(pool, dtype=float)
        self.n_pool = len(pool)
        self.pool_s = np.clip(pool[:, 2] / self.eta0, 1.0e-3, 0.985)
        self.pool_vel = np.zeros((self.n_pool, 3))

        # ---- shallow-water grid --------------------------------------------
        self.dx = 0.1
        self.nx = int(round((self.bmax[0] - self.bmin[0]) / self.dx))
        self.ny = int(round((self.bmax[1] - self.bmin[1]) / self.dx))
        self.cell_A = self.dx * self.dx
        xc = self.bmin[0] + (np.arange(self.nx) + 0.5) * self.dx
        yc = self.bmin[1] + (np.arange(self.ny) + 0.5) * self.dx
        self.Xc, self.Yc = np.meshgrid(xc, yc, indexing="ij")
        self.u = np.zeros((self.nx + 1, self.ny))
        self.w = np.zeros((self.nx, self.ny + 1))
        self.eta = np.full((self.nx, self.ny), self.eta0)
        self.disp_depth = self._displacement_depth(self.p, self._sub_volume(draft0))
        self.kf = 0.5          # linear bottom friction
        self.nu = 0.02         # eddy viscosity
        self.h_min = 0.02

        tank_area = (self.bmax[0] - self.bmin[0]) * (self.bmax[1] - self.bmin[1])
        self.M_pool = self.rho * tank_area * self.eta0
        self.M_pool0 = self.M_pool

        # ---- inlet parcels --------------------------------------------------
        self.n_jet = 320
        self.jet_park = np.array([0.0, 0.0, 3.5])
        self.jet_pos = np.tile(self.jet_park, (self.n_jet, 1))
        self.jet_vel = np.zeros((self.n_jet, 3))
        self.jet_live = np.zeros(self.n_jet, dtype=bool)
        self.m_jet = self.rho * 0.05 ** 3
        self.jet_budget = 0.0

        # ---- bookkeeping ----------------------------------------------------
        self.t = 0.0
        self.dt = 0.0025
        self.M_int = 0.0
        self.M_source = 0.0
        self.M_leak = 0.0
        self.M_flood = 0.0
        self.f_buoy = np.zeros(3)
        self.f_drag = np.zeros(3)
        self.f_ground = np.zeros(3)
        self.f_wall = np.zeros(3)
        self.f_intw = np.zeros(3)
        self.f_jet = np.zeros(3)
        self.gaps = np.zeros(5)
        self.eta_boat = self.eta0
        self.sink_time = -1.0
        self.impact_time = -1.0
        self._src = np.zeros((self.nx, self.ny))

    # -------------------------------------------------------------- geometry
    def _cav_a(self, zeta):
        z = np.asarray(zeta, dtype=float)
        return self.cav_scale * (self._a0 + (self._a1 - self._a0) * z / self.cav_depth)

    def _cav_b(self, zeta):
        z = np.asarray(zeta, dtype=float)
        return self.cav_scale * (self._b0 + (self._b1 - self._b0) * z / self.cav_depth)

    def _sub_volume(self, d):
        dd = float(np.clip(d, 0.0, self.d_full))
        if dd <= 0.0:
            return 0.0
        return float(min(self.disp_K * dd ** self.disp_q, self.v_full))

    def _footprint_weight(self, p):
        r2 = ((self.Xc - p[0]) / self.hull_a) ** 2 + ((self.Yc - p[1]) / self.hull_b) ** 2
        wt = np.clip(1.0 - r2, 0.0, None)
        total = float(wt.sum())
        if total <= 1.0e-12:
            wt = np.zeros_like(wt)
            wt[self.nx // 2, self.ny // 2] = 1.0
            total = 1.0
        return wt / total

    def _ring_weight(self, p):
        r2 = ((self.Xc - p[0]) / self.hull_a) ** 2 + ((self.Yc - p[1]) / self.hull_b) ** 2
        wt = np.where((r2 > 1.0) & (r2 < 3.0), 1.0, 0.0)
        total = float(wt.sum())
        if total <= 1.0e-12:
            return self._footprint_weight(p)
        return wt / total

    def _displacement_depth(self, p, vol):
        return self._footprint_weight(p) * (vol / self.cell_A)

    def _interior_level(self):
        vol = min(self.M_int / self.rho, self.cav_cap)
        return float(np.interp(vol, self.cav_vol, self.cav_z))

    def _rotation(self):
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        return ry @ rx

    def _quaternion(self):
        hr, hp = 0.5 * self.roll, 0.5 * self.pitch
        return np.array(
            [
                np.cos(hp) * np.sin(hr),
                np.cos(hr) * np.sin(hp),
                -np.sin(hp) * np.sin(hr),
                np.cos(hp) * np.cos(hr),
            ]
        )

    # ------------------------------------------------------------ grid utils
    def _cell_index(self, points):
        ix = np.clip(((points[:, 0] - self.bmin[0]) / self.dx).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(((points[:, 1] - self.bmin[1]) / self.dx).astype(np.int64), 0, self.ny - 1)
        return ix, iy

    def _deposit(self, point, volume):
        ix = int(np.clip((point[0] - self.bmin[0]) / self.dx, 0, self.nx - 1))
        iy = int(np.clip((point[1] - self.bmin[1]) / self.dx, 0, self.ny - 1))
        self._src[ix, iy] += volume

    # ------------------------------------------------------------------ step
    def _step(self, dt):
        if dt <= 0.0:
            return
        self._src[...] = 0.0

        # ------------------- prescribed inlet emission ----------------------
        if self.inlet_t0 <= self.t < self.inlet_t1:
            self.jet_budget += self.rho * self.inlet_Q * dt
        while self.jet_budget >= self.m_jet:
            free = np.flatnonzero(~self.jet_live)
            if free.size == 0:
                break
            k = int(free[0])
            ang = float(self.rng.uniform(0.0, 2.0 * np.pi))
            rad = self.inlet_r * float(np.sqrt(self.rng.uniform(0.0, 1.0)))
            lag = float(self.rng.uniform(0.0, 1.0)) * dt
            self.jet_pos[k] = (
                self.inlet_c
                + np.array([rad * np.cos(ang), rad * np.sin(ang), 0.0])
                + self.inlet_v * lag
            )
            self.jet_vel[k] = self.inlet_v
            self.jet_live[k] = True
            self.jet_budget -= self.m_jet
            self.M_source += self.m_jet

        # ------------------- ballistic transport of parcels -----------------
        m_into_hull = 0.0
        live = np.flatnonzero(self.jet_live)
        if live.size:
            self.jet_vel[live] += self.gravity * dt
            self.jet_pos[live] += self.jet_vel[live] * dt
            rot = self._rotation()
            loc = (self.jet_pos[live] - self.p) @ rot
            zeta = self._interior_level()
            z_int = self.p[2] + self.floor_off + zeta
            rim_a = float(self._cav_a(self.cav_depth))
            rim_b = float(self._cav_b(self.cav_depth))
            inside = (loc[:, 0] / rim_a) ** 2 + (loc[:, 1] / rim_b) ** 2 <= 1.0
            on_hull = (loc[:, 0] / self.hull_a) ** 2 + (loc[:, 1] / self.hull_b) ** 2 <= 1.0
            ix, iy = self._cell_index(self.jet_pos[live])
            surface = self.eta[ix, iy]
            zpos = self.jet_pos[live][:, 2]
            hits_int = inside & (zpos <= max(z_int, self.p[2] + self.floor_off))
            hits_shell = (~hits_int) & on_hull & (zpos <= self.p[2] + self.rim_off) & (
                zpos >= self.p[2] + self.keel_off
            )
            hits_pool = (~hits_int) & (~hits_shell) & (zpos <= surface)
            escaped = zpos < self.bmin[2] - 0.5
            for j, idx in enumerate(live):
                idx = int(idx)
                if hits_int[j]:
                    room = max(self.cav_cap * self.rho - self.M_int, 0.0)
                    take = min(self.m_jet, room)
                    self.M_int += take
                    m_into_hull += take
                    spill = self.m_jet - take
                    if spill > 0.0:
                        self.M_pool += spill
                        self.M_leak += spill
                        self._deposit(self.jet_pos[idx], spill / self.rho)
                    self.jet_live[idx] = False
                elif hits_shell[j] or hits_pool[j] or escaped[j]:
                    self.M_pool += self.m_jet
                    self._deposit(self.jet_pos[idx], self.m_jet / self.rho)
                    self.jet_live[idx] = False
            dead = ~self.jet_live
            if dead.any():
                self.jet_pos[dead] = self.jet_park
                self.jet_vel[dead] = 0.0

        # ------------------- undisturbed waterline at the hull --------------
        eta_b = float((self._ring_weight(self.p) * self.eta).sum())
        self.eta_boat = eta_b
        z_keel = self.p[2] + self.keel_off
        z_rim = self.p[2] + self.rim_off
        draft = eta_b - z_keel

        # ------------------- weir exchange over the gunwale -----------------
        zeta = self._interior_level()
        z_int = self.p[2] + self.floor_off + zeta
        c_weir = 0.03
        weir = self.rho * c_weir * self.rim_perimeter * (2.0 / 3.0) * np.sqrt(2.0 * self.g)
        head_in = eta_b - max(z_rim, z_int)
        head_out = z_int - max(z_rim, eta_b)
        if head_in > 0.0:
            rate = weir * head_in ** 1.5
            dm = min(rate * dt, max(self.cav_cap * self.rho - self.M_int, 0.0), self.M_pool)
            self.M_int += dm
            self.M_pool -= dm
            self.M_flood += dm
            self._deposit(self.p, -dm / self.rho)
        elif head_out > 0.0:
            rate = weir * head_out ** 1.5
            dm = min(rate * dt, self.M_int)
            self.M_int -= dm
            self.M_pool += dm
            self.M_leak += dm
            self._deposit(self.p, dm / self.rho)

        # ------------------- rigid-body dynamics ----------------------------
        v_sub = self._sub_volume(draft)
        f_sub = float(np.clip(draft / self.d_full, 0.0, 1.0))
        mass_total = self.m_boat + self.M_int
        m_add = 0.9 * self.rho * v_sub
        m_eff = mass_total + m_add

        self.f_buoy = np.array([0.0, 0.0, self.rho * self.g * v_sub])
        cda_v, cda_h = 9.6, 1.4          # drag area products [m^2]
        self.f_drag = -0.5 * self.rho * f_sub * np.array(
            [
                cda_h * abs(self.v[0]) * self.v[0],
                cda_h * abs(self.v[1]) * self.v[1],
                cda_v * abs(self.v[2]) * self.v[2],
            ]
        )
        self.f_intw = self.M_int * self.gravity
        if m_into_hull > 0.0:
            self.f_jet = (m_into_hull / dt) * (self.inlet_v - self.v)
        else:
            self.f_jet = np.zeros(3)

        k_c, c_c = 2.0e5, 1.6e4
        gap_floor = z_keel
        self.f_ground = np.zeros(3)
        if gap_floor < 0.0:
            fn = max(k_c * (-gap_floor) - c_c * self.v[2], 0.0)
            self.f_ground[2] = fn
            speed_h = float(np.hypot(self.v[0], self.v[1]))
            if speed_h > 1.0e-6:
                self.f_ground[:2] = -0.5 * fn * self.v[:2] / speed_h
            if self.impact_time < 0.0:
                self.impact_time = self.t

        self.f_wall = np.zeros(3)
        gaps = [gap_floor]
        for axis, half in ((0, self.hull_a), (1, self.hull_b)):
            lo = (self.p[axis] - half) - (self.bmin[axis] + 0.06)
            hi = (self.bmax[axis] - 0.06) - (self.p[axis] + half)
            gaps.extend([lo, hi])
            if lo < 0.0:
                self.f_wall[axis] += k_c * (-lo) - c_c * self.v[axis]
            if hi < 0.0:
                self.f_wall[axis] -= k_c * (-hi) + c_c * self.v[axis]
        self.gaps = np.asarray(gaps, dtype=float)

        f_gravity = np.array([0.0, 0.0, -self.m_boat * self.g])
        total = (
            f_gravity + self.f_buoy + self.f_drag + self.f_ground + self.f_wall
            + self.f_intw + self.f_jet
        )
        accel = total / max(m_eff, 1.0e-6)
        self.v = np.clip(self.v + accel * dt, -6.0, 6.0)
        self.p = self.p + self.v * dt
        self.p[0] = float(np.clip(self.p[0], self.bmin[0] + 0.4, self.bmax[0] - 0.4))
        self.p[1] = float(np.clip(self.p[1], self.bmin[1] + 0.4, self.bmax[1] - 0.4))
        self.p[2] = float(np.clip(self.p[2], -self.keel_off - 0.5, self.bmax[2]))

        # roll / pitch about the body origin, hydrostatic restoring + damping
        i_wp_x = 0.25 * np.pi * self.hull_a * self.hull_b ** 3
        i_wp_y = 0.25 * np.pi * self.hull_b * self.hull_a ** 3
        inertia_x = mass_total * (self.hull_b ** 2 + 0.30) / 4.0 + 4.0
        inertia_y = mass_total * (self.hull_a ** 2 + 0.30) / 4.0 + 8.0
        kx = self.rho * self.g * i_wp_x * f_sub
        ky = self.rho * self.g * i_wp_y * f_sub
        cx = 1.4 * np.sqrt(max(kx, 0.0) * inertia_x)
        cy = 1.4 * np.sqrt(max(ky, 0.0) * inertia_y)
        tx = -kx * self.roll - cx * self.wroll
        ty = -ky * self.pitch - cy * self.wpitch
        self.wroll = float(np.clip(self.wroll + (tx / inertia_x) * dt, -1.5, 1.5))
        self.wpitch = float(np.clip(self.wpitch + (ty / inertia_y) * dt, -1.5, 1.5))
        self.roll = float(np.clip(self.roll + self.wroll * dt, -0.35, 0.35))
        self.pitch = float(np.clip(self.pitch + self.wpitch * dt, -0.35, 0.35))

        if self.sink_time < 0.0 and (self.p[2] + self.rim_off) < eta_b:
            self.sink_time = self.t

        # ------------------- shallow-water update ---------------------------
        depth = np.maximum(self.eta - self.disp_depth, self.h_min)

        du = -self.g * (self.eta[1:, :] - self.eta[:-1, :]) / self.dx
        ui = self.u[1:-1, :]
        lap_u = (self.u[2:, :] - 2.0 * ui + self.u[:-2, :]) / self.dx ** 2
        self.u[1:-1, :] = ui + dt * (du - self.kf * ui + self.nu * lap_u)
        dv = -self.g * (self.eta[:, 1:] - self.eta[:, :-1]) / self.dx
        wi = self.w[:, 1:-1]
        lap_w = (self.w[:, 2:] - 2.0 * wi + self.w[:, :-2]) / self.dx ** 2
        self.w[:, 1:-1] = wi + dt * (dv - self.kf * wi + self.nu * lap_w)
        self.u[0, :] = 0.0
        self.u[-1, :] = 0.0
        self.w[:, 0] = 0.0
        self.w[:, -1] = 0.0
        np.clip(self.u, -4.0, 4.0, out=self.u)
        np.clip(self.w, -4.0, 4.0, out=self.w)

        hx = np.zeros((self.nx + 1, self.ny))
        hx[1:-1, :] = 0.5 * (depth[1:, :] + depth[:-1, :])
        hy = np.zeros((self.nx, self.ny + 1))
        hy[:, 1:-1] = 0.5 * (depth[:, 1:] + depth[:, :-1])
        flux_x = hx * self.u
        flux_y = hy * self.w
        div = (flux_x[1:, :] - flux_x[:-1, :]) / self.dx + (
            flux_y[:, 1:] - flux_y[:, :-1]
        ) / self.dx

        new_disp = self._displacement_depth(self.p, v_sub)
        d_disp = (new_disp - self.disp_depth) / dt
        source = self._src / (self.cell_A * dt)
        self.eta = self.eta + dt * (-div + d_disp + source)
        self.disp_depth = new_disp
        np.clip(self.eta, self.h_min, 3.7, out=self.eta)

        # ------------------- pool marker advection --------------------------
        uc = 0.5 * (self.u[1:, :] + self.u[:-1, :])
        wc = 0.5 * (self.w[:, 1:] + self.w[:, :-1])
        ix, iy = self._cell_index(self.pool_pos)
        up = uc[ix, iy]
        wp = wc[ix, iy]
        z_new = self.pool_s * self.eta[ix, iy]
        # markers cannot occupy the volume taken by the floating hull
        z_keel = self.p[2] + self.keel_off
        if z_keel > 0.10:
            inside = ((self.pool_pos[:, 0] - self.p[0]) / self.hull_a) ** 2 + (
                (self.pool_pos[:, 1] - self.p[1]) / self.hull_b
            ) ** 2 <= 1.0
            z_new = np.where(inside, np.minimum(z_new, z_keel - 0.03), z_new)
        self.pool_vel[:, 0] = up
        self.pool_vel[:, 1] = wp
        self.pool_vel[:, 2] = (z_new - self.pool_pos[:, 2]) / dt
        self.pool_pos[:, 0] = np.clip(
            self.pool_pos[:, 0] + up * dt, self.bmin[0] + 0.03, self.bmax[0] - 0.03
        )
        self.pool_pos[:, 1] = np.clip(
            self.pool_pos[:, 1] + wp * dt, self.bmin[1] + 0.03, self.bmax[1] - 0.03
        )
        self.pool_pos[:, 2] = np.clip(z_new, 0.01, 3.7)

        self.t += dt

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        while self.t < target - 1.0e-9 and guard < 500000:
            self._step(min(self.dt, target - self.t))
            guard += 1
        self.t = max(self.t, target)
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        rot = self._rotation()

        zeta = self._interior_level()
        n_act = int(np.searchsorted(self.int_zeta, zeta, side="right"))
        if self.M_int > 0.0:
            n_act = max(n_act, 1)
        n_act = min(n_act, self.n_int)
        int_pos = np.tile(self.int_park, (self.n_int, 1))
        int_mass = np.zeros(self.n_int)
        if n_act > 0:
            int_pos[:n_act] = self.p[None, :] + self.int_local[:n_act] @ rot.T
            int_mass[:n_act] = self.M_int / n_act

        pool_mass = np.full(self.n_pool, self.M_pool / max(self.n_pool, 1))
        jet_mass = np.where(self.jet_live, self.m_jet, 0.0)

        position = np.vstack((self.p[None, :], self.pool_pos, int_pos, self.jet_pos))
        velocity = np.vstack(
            (
                self.v[None, :],
                self.pool_vel,
                np.tile(self.v, (self.n_int, 1)),
                self.jet_vel,
            )
        )
        mass = np.concatenate((np.array([self.m_boat]), pool_mass, int_mass, jet_mass))
        n = len(position)

        body_id = np.ones(n, dtype=np.int32)
        body_id[0] = 0
        water_region = np.ones(n, dtype=np.int32)
        water_region[0] = 0
        water_region[1 + self.n_pool:] = 3
        phase_id = np.ones(n, dtype=np.int32)

        orientation = np.zeros((n, 4))
        orientation[:, 3] = 1.0
        orientation[0] = self._quaternion()
        angular_velocity = np.zeros((n, 3))
        angular_velocity[0] = np.array([self.wroll, self.wpitch, 0.0])

        mass_quantity = np.zeros((n, 2))
        mass_quantity[0, 0] = self.m_boat
        mass_quantity[1:, 1] = mass[1:]

        external_force = mass[:, None] * self.gravity[None, :]

        pairs = (
            self.f_buoy,
            self.f_drag,
            self.f_ground,
            self.f_wall,
            self.f_intw,
            self.f_jet,
        )
        interaction_force = np.zeros((2 * len(pairs), 3))
        for i, force in enumerate(pairs):
            f = np.asarray(force, dtype=float)
            interaction_force[2 * i] = f
            interaction_force[2 * i + 1] = -f

        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(position, dtype=float),
            velocity=np.ascontiguousarray(velocity, dtype=float),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "phase_id": phase_id,
                "water_region": water_region,
                "mass_quantity": mass_quantity,
            },
            fields={
                "external_force": external_force,
                "interaction_force": interaction_force,
                "contact_signed_distance": np.asarray(self.gaps, dtype=float),
                "cumulative_mass_source": np.array([0.0, self.M_source]),
            },
            scalars={
                "contact_length_scale": float(self.shell_thickness),
                "characteristic_length": float(self.d_full),
                "liquid_phase_id": 1.0,
                "water_mass_in_hull": float(self.M_int),
                "interior_water_level": float(self.p[2] + self.floor_off + zeta),
                "external_free_surface_z": float(self.eta_boat),
                "boat_draft": float(self.eta_boat - (self.p[2] + self.keel_off)),
                "boat_center_z": float(self.p[2]),
                "boat_freeboard": float((self.p[2] + self.rim_off) - self.eta_boat),
                "leaked_mass": float(self.M_leak),
                "flooded_mass": float(self.M_flood),
                "cumulative_inlet_mass": float(self.M_source),
                "total_mass": float(mass.sum()),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "shallow-water pool + 5-DOF rigid boat + weir cavity fill",
            "substep_dt": float(self.dt),
            "grid": [int(self.nx), int(self.ny)],
            "final_time": float(self.t),
            "cavity_capacity_m3": float(self.cav_cap),
            "displacement_at_rim_m3": float(self.v_full),
            "final_hull_water_mass": float(self.M_int),
            "cumulative_inlet_mass": float(self.M_source),
            "leaked_mass": float(self.M_leak),
            "flooded_mass": float(self.M_flood),
            "sink_start_time": float(self.sink_time),
            "bottom_impact_time": float(self.impact_time),
            "final_boat_center_z": float(self.p[2]),
        }
