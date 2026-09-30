"""Physically-based solver for sponge saturation and wringing.

Models:
  * Sponge: 36-vertex mass-spring poro-elastic lattice (structural + shear +
    bending springs), per-vertex pore-liquid storage (saturation), capillary
    diffusion of stored water, gravity, floor contact with friction.
  * Water: particle liquid (soft repulsion + cohesion + normal damping),
    gravity, basin contact with friction; faucet is a prescribed mass and
    momentum source until stops_at.
  * Two-way coupling: Darcy-type drag between liquid particles and sponge
    vertices with equal-and-opposite reaction; absorption converts liquid
    particles into stored pore water (inelastic momentum merge); torsion and
    drip-driven expulsion releases stored water with momentum-conserving
    recoil on the solid.
  * Grips: kinematic lift + opposite twist ramps applied as hard constraints
    to end regions from active_from; constraint forces feed momentum budgets.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _smooth_d(u: float) -> float:
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return 6.0 * u * (1.0 - u)


class BenchmarkSolver:
    DT = 0.002
    NSUB = 4

    H = 0.022        # water interaction radius
    KR = 0.35        # water repulsion strength (N)
    KA = 0.03        # water cohesion strength (N)
    CV = 0.04        # water pair damping
    RP = 0.006       # water particle contact radius

    R_DRAG = 0.075   # water-sponge drag radius
    R_ABS = 0.06     # absorption radius
    TAU_DRAG = 0.02  # drag relaxation time (s)
    F_DRAG_CAP = 0.25

    CAP = 0.013      # pore water capacity per vertex (kg)
    D_PAIR = 2.5     # capillary diffusion rate per edge (1/s)
    S_RET = 0.32     # residual saturation after wringing
    K_WRING = 2.5    # twist-driven expulsion coefficient
    K_DRIP = 5.0     # gravity drip coefficient
    V_EJECT = 1.2    # ejection speed (m/s)

    RATE = 0.06      # faucet mass rate (kg/s)
    T_STOP = 6.8
    GRAV = np.array([0.0, 0.0, -9.81])

    Z_FLOOR = 0.015
    X_WALL = 0.384
    Y_BACK = 0.264
    Y_FRONT = -0.264
    NOZZLE = np.array([-0.07, 0.04, 0.5245])

    K_X = 300.0
    K_X2 = 40.0
    K_RING = 180.0
    K_DIAG = 40.0
    ZETA = 0.4
    M_DRY_V = 0.0124 / 36.0
    RV = 0.010       # sponge vertex contact radius
    KC_SP = 800.0
    CC_SP = 0.8
    MU_SP = 0.7
    CT_SP = 0.4
    KC_W = 60.0
    CC_W = 0.05
    MU_W = 0.4
    CT_W = 0.08

    LIFT_T0 = 6.5
    LIFT_DUR = 1.5
    LIFT_H = 0.22
    TWIST_T0 = 8.0
    TWIST_DUR = 3.0
    TWIST_DEG = 150.0
    M_GRIP = 0.02

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        rng = np.random.default_rng(int(seed))
        frame0 = scene.initial_state()
        p0 = np.asarray(frame0.position, dtype=float)
        v0 = np.asarray(frame0.velocity, dtype=float)
        b0 = np.asarray(frame0.body_id)

        sp0 = p0[b0 == 1]                      # 36 sponge vertices
        wcol0 = p0[b0 == 2]                    # initial stream column
        wcolv = v0[b0 == 2]
        grips0 = p0[b0 >= 10]                  # 2 grips

        self.n_sp = len(sp0)
        self.m_p = 0.00095
        interval = self.m_p / self.RATE
        n_emit = int(np.floor(self.T_STOP / interval)) + 1
        self.emit_t = np.arange(n_emit) * interval
        ang = rng.uniform(0.0, 2.0 * np.pi, n_emit)
        rad = 0.012 * np.sqrt(rng.uniform(0.0, 1.0, n_emit))
        self.emit_off = np.stack([rad * np.cos(ang), rad * np.sin(ang),
                                  np.zeros(n_emit)], axis=1)

        n_water = len(wcol0) + n_emit
        self.n_water = n_water
        self.w0 = self.n_sp
        self.w1 = self.n_sp + n_water
        n_tot = self.n_sp + n_water + 2
        self.n_tot = n_tot

        self.pos = np.zeros((n_tot, 3))
        self.vel = np.zeros((n_tot, 3))
        self.mass = np.zeros(n_tot)
        self.phase = np.zeros(n_tot, dtype=np.int8)   # 0 solid/parked, 1 free, 2 stored
        self.body = np.zeros(n_tot, dtype=np.int32)

        self.pos[:self.n_sp] = sp0
        self.body[:self.n_sp] = 1
        self.mass[:self.n_sp] = self.M_DRY_V

        ncol = len(wcol0)
        self.pos[self.w0:self.w0 + ncol] = wcol0
        self.vel[self.w0:self.w0 + ncol] = wcolv
        self.mass[self.w0:self.w0 + ncol] = self.m_p
        self.phase[self.w0:self.w0 + ncol] = 1
        self.body[self.w0:self.w1] = 2

        park = self.NOZZLE + np.array([0.0, 0.0, 0.028])
        self.pos[self.w0 + ncol:self.w1] = park + np.concatenate(
            [self.emit_off[:, :2], np.zeros((n_emit, 1))], axis=1)
        self.emit_idx = np.arange(self.w0 + ncol, self.w1)

        self.pos[self.w1] = grips0[0]
        self.pos[self.w1 + 1] = grips0[1]
        self.body[self.w1:] = np.array([10, 11])
        self.mass[self.w1:] = self.M_GRIP

        self.pdir = rng.normal(size=(n_water, 3))
        self.pdir /= np.linalg.norm(self.pdir, axis=1, keepdims=True)
        self.noabs = np.zeros(n_water)

        self.m_w = np.zeros(self.n_sp)         # stored pore water mass per vertex
        self.stored = [[] for _ in range(self.n_sp)]

        self._build_lattice(sp0)

        controls = scene.config["controls"]
        self.grip_init = np.array(controls["left_end"]["initial_position"]
                                  + controls["right_end"]["initial_position"]).reshape(2, 3)
        lx = np.array(controls["left_end"]["region_x"])
        rx = np.array(controls["right_end"]["region_x"])
        x0 = sp0[:, 0]
        self.left_v = np.where((x0 >= lx[0] - 1e-9) & (x0 <= lx[1] + 1e-9))[0]
        self.right_v = np.where((x0 >= rx[0] - 1e-9) & (x0 <= rx[1] + 1e-9))[0]
        self.grabbed = False
        self.d0 = np.zeros((self.n_sp, 3))

        m_water_total = ncol * self.m_p + n_emit * self.m_p
        self.ref_momentum = float(m_water_total * 1.0)
        self.injected = 0.0
        self.absorbed_cum = 0.0
        self.expelled_cum = 0.0
        self.emit_ptr = 0

        self.t = 0.0
        self.step = 0

        self.imp = np.zeros((3, 3))
        self.inflow = np.zeros((3, 3))
        self.outflow = np.zeros((3, 3))
        self.expulse_acc = np.zeros(self.n_sp)

        self.ext_force = np.zeros((n_tot, 3))
        self.f_grav_dom = np.zeros((3, 3))
        self.f_contact_dom = np.zeros((3, 3))
        self.f_constr_dom = np.zeros((3, 3))
        self.f_pair_water = np.zeros(3)        # net drag force on water from solid
        self.prev_grip_v = np.zeros((2, 3))

    # -------------------------------------------------------------- lattice
    def _build_lattice(self, sp0):
        edges = []  # (a, b, k)
        for i in range(9):
            for j in range(4):
                v = i * 4 + j
                if i < 8:
                    edges.append((v, v + 4, self.K_X))
                if i < 7:
                    edges.append((v, v + 8, self.K_X2))
                edges.append((v, i * 4 + (j + 1) % 4, self.K_RING))
                if j < 2:
                    edges.append((v, i * 4 + j + 2, self.K_DIAG))
        seen = set()
        uniq = []
        for a, b, k in edges:
            key = (min(a, b), max(a, b))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((a, b, k))
        ea = np.array([e[0] for e in uniq])
        eb = np.array([e[1] for e in uniq])
        kk = np.array([e[2] for e in uniq])
        rest0 = np.linalg.norm(sp0[eb] - sp0[ea], axis=1)
        m = self.M_DRY_V
        cc = 2.0 * self.ZETA * np.sqrt(kk * m * 0.5)
        self.sp_ea, self.sp_eb = ea, eb
        self.sp_k, self.sp_L0, self.sp_c = kk, rest0, cc

        # diffusion edges: x-neighbours + ring neighbours
        d_edges = set()
        for i in range(9):
            for j in range(4):
                v = i * 4 + j
                if i < 8:
                    d_edges.add((v, v + 4))
                d_edges.add((v, i * 4 + (j + 1) % 4))
        self.diff_a = np.array([e[0] for e in d_edges])
        self.diff_b = np.array([e[1] for e in d_edges])

        # x-springs for strain diagnostic
        xm = kk >= self.K_X - 1e-9
        self.x_ea, self.x_eb = ea[xm], eb[xm]
        self.x_L0 = rest0[xm]

        # smoothing operator for drag reaction: W = (I + A)/(1 + deg)
        n = self.n_sp
        A = np.zeros((n, n))
        for a, b in zip(ea, eb):
            A[a, b] = 1.0
            A[b, a] = 1.0
        W = A + np.eye(n)
        self.W_smooth = W / W.sum(axis=1, keepdims=True)

    # --------------------------------------------------------- grip kinematics
    def _grip_pose(self, t):
        lam = _smooth((t - self.LIFT_T0) / self.LIFT_DUR)
        lamd = _smooth_d((t - self.LIFT_T0) / self.LIFT_DUR) / self.LIFT_DUR
        u = (t - self.TWIST_T0) / self.TWIST_DUR
        th = np.deg2rad(self.TWIST_DEG) * _smooth(u)
        thd = np.deg2rad(self.TWIST_DEG) * _smooth_d(u) / self.TWIST_DUR
        centers = self.grip_init + np.array([0.0, 0.0, self.LIFT_H * lam])
        return centers, np.array([0.0, 0.0, self.LIFT_H * lamd]), th, thd

    @staticmethod
    def _rot_x(d, th):
        c, s = np.cos(th), np.sin(th)
        out = np.array(d, dtype=float, copy=True)
        y = d[..., 1] * c - d[..., 2] * s
        z = d[..., 1] * s + d[..., 2] * c
        out[..., 1] = y
        out[..., 2] = z
        return out

    def _grip_quat(self, th_s):
        return np.array([np.sin(th_s * 0.5), 0.0, 0.0, np.cos(th_s * 0.5)])

    def _constrained_targets(self, t, side):
        """Target pose for gripped vertices of one side at time t."""
        centers, cdot, th, thd = self._grip_pose(t)
        idx = self.left_v if side == 0 else self.right_v
        sign = 1.0 if side == 0 else -1.0
        d = self._rot_x(self.d0[idx], sign * th)
        p = centers[side] + d
        om = sign * thd
        v = np.stack([np.zeros(len(idx)),
                      np.full(len(idx), cdot[2]) - om * d[:, 2],
                      om * d[:, 1]], axis=1)
        return p, v

    # ------------------------------------------------------------- main step
    def _step(self):
        dt = self.DT
        t1 = self.t + dt
        mp = self.m_p

        # ---- 1) faucet emission (prescribed mass + momentum source)
        while self.emit_ptr < len(self.emit_t) and self.emit_t[self.emit_ptr] <= t1 + 1e-12:
            k = self.emit_ptr
            idx = self.emit_idx[k]
            self.pos[idx] = self.NOZZLE + self.emit_off[k]
            self.vel[idx] = np.array([0.0, 0.0, -1.0])
            self.mass[idx] = mp
            self.phase[idx] = 1
            self.inflow[1] += mp * self.vel[idx]
            self.injected += mp
            self.emit_ptr += 1

        free = np.where(self.phase == 1)[0]
        wloc = free - self.w0
        sp_pos = self.pos[:self.n_sp]
        sp_vel = self.vel[:self.n_sp]
        sp_mass = self.M_DRY_V + self.m_w

        # ---- 2) water dynamics
        n_f = len(free)
        F = np.zeros((n_f, 3))
        m_free = self.mass[free]
        F += m_free[:, None] * self.GRAV

        contact_sum = np.zeros(3)
        if n_f >= 2:
            P = self.pos[free]
            V = self.vel[free]
            dx = P[:, None, :] - P[None, :, :]
            r2 = np.einsum('ijk,ijk->ij', dx, dx)
            np.fill_diagonal(r2, 1.0)
            r = np.sqrt(r2)
            dv = V[:, None, :] - V[None, :, :]
            dot = np.einsum('ijk,ijk->ij', dv, dx)
            q = np.clip(1.0 - r / self.H, 0.0, None)
            near = r < self.H
            A = np.where(near, self.KR * q / r, 0.0)
            B = np.where(near, -self.CV * q * dot / r2, 0.0)
            coh = (~near) & (r < 1.5 * self.H)
            A = np.where(coh, -self.KA * (1.0 - (r - self.H) / (0.5 * self.H)) / r, A)
            F += np.einsum('ij,ijk->ik', A + B, dx)

            # drag with sponge vertices (Darcy resistance) + absorption pairs
            d = P[:, None, :] - sp_pos[None, :, :]
            rr2 = np.einsum('ijk,ijk->ij', d, d)
            rr = np.sqrt(rr2)
            wd = np.clip(1.0 - rr / self.R_DRAG, 0.0, None)
            dm = rr < self.R_DRAG
            dvel = V[:, None, :] - sp_vel[None, :, :]
            cd = mp / self.TAU_DRAG
            fd = -cd * (wd * dm)[:, :, None] * dvel
            fn = np.linalg.norm(fd, axis=2, keepdims=True)
            fd = np.where(fn > self.F_DRAG_CAP, fd * (self.F_DRAG_CAP / np.maximum(fn, 1e-12)), fd)
            F += fd.sum(axis=1)
            f_vert_raw = -fd.sum(axis=0)  # reaction on solid, [36,3]
            self._absorb_pairs = (rr < self.R_ABS)
            self._absorb_rr = rr
        else:
            f_vert_raw = np.zeros((self.n_sp, 3))
            self._absorb_pairs = np.zeros((n_f, self.n_sp), dtype=bool)
            self._absorb_rr = np.zeros((n_f, self.n_sp))

        # basin contact for water particles
        Pf = self.pos[free]
        Vf = self.vel[free]
        zc = self.Z_FLOOR + self.RP
        pen = np.clip(zc - Pf[:, 2], 0.0, None)
        if np.any(pen > 0):
            fn = self.KC_W * pen - self.CC_W * np.minimum(Vf[:, 2], 0.0)
            fn = np.maximum(fn, 0.0) * (pen > 0)
            contact_force = np.zeros((n_f, 3))
            contact_force[:, 2] += fn
            vt = Vf[:, :2]
            sp = np.linalg.norm(vt, axis=1)
            ft = np.minimum(self.CT_W * sp, self.MU_W * fn)
            scale = np.where(sp > 1e-9, ft / np.maximum(sp, 1e-9), 0.0)
            contact_force[:, 0] -= scale * vt[:, 0]
            contact_force[:, 1] -= scale * vt[:, 1]
            F += contact_force
            contact_sum += contact_force.sum(axis=0)
        wall_force = np.zeros((n_f, 3))
        if n_f:
            pen_x = np.clip(np.abs(Pf[:, 0]) - self.X_WALL, 0.0, None)
            if np.any(pen_x > 0):
                wall_force[:, 0] += -np.sign(Pf[:, 0]) * 30.0 * pen_x
            pen_b = np.clip(Pf[:, 1] - self.Y_BACK, 0.0, None)
            if np.any(pen_b > 0):
                wall_force[:, 1] += -30.0 * pen_b
            pen_f = np.clip(-Pf[:, 1] + self.Y_FRONT, 0.0, None)
            if np.any(pen_f > 0):
                wall_force[:, 1] += 30.0 * pen_f
        F += wall_force
        contact_sum += wall_force.sum(axis=0)

        # integrate water
        a = F / np.maximum(m_free, 1e-12)[:, None]
        self.vel[free] += a * dt
        np.clip(self.vel[free], -15.0, 15.0, out=self.vel[free])
        self.pos[free] += self.vel[free] * dt
        self._water_ext = np.zeros((self.n_tot, 3))
        self._water_ext[free] = F
        self._water_contact_sum = contact_sum
        self._water_grav_sum = (m_free.sum()) * self.GRAV

        # ---- 3) absorption (liquid -> stored pore water)
        self._absorption_step(free, wloc, t1)

        # ---- 4) sponge: drag reaction impulse + spring substeps
        f_react = self.W_smooth @ f_vert_raw
        self._sponge_advance(dt, f_react)
        self.f_pair_water = f_vert_raw.sum(axis=0) * -1.0  # force on water from solid

        # ---- 5) expulsion (stored -> liquid)
        self._expulsion_step(t1)

        # ---- 6) capillary diffusion of stored water
        self._diffusion(dt)

        # ---- 7) stored tracers follow their vertices
        tr = np.where(self.phase == 2)[0]
        if len(tr):
            va = self.tracer_vertex[tr]
            self.pos[tr] = self.pos[va] + 0.010 * self.pdir[tr - self.w0]
            self.vel[tr] = self.vel[va]

        # ---- 8) budgets
        sp_mass_now = self.M_DRY_V + self.m_w
        g_sp = sp_mass_now.sum() * self.GRAV
        g_w = self._water_grav_sum
        g_g = 2.0 * self.M_GRIP * self.GRAV
        self.f_grav_dom[0] = g_sp
        self.f_grav_dom[1] = g_w
        self.f_grav_dom[2] = g_g
        self.f_contact_dom[0] = self._sp_contact_imp / dt
        self.f_contact_dom[1] = contact_sum
        self.f_contact_dom[2] = 0.0
        self.f_constr_dom[0] = self._sp_constr_imp / dt
        centers, cdot, th, thd = self._grip_pose(t1)
        grip_v = np.stack([cdot, cdot])
        m_dv = self.M_GRIP * (grip_v - self.prev_grip_v) / dt
        f_c_side = self._constr_side_force  # [2,3]
        f_act = m_dv - self.M_GRIP * self.GRAV + f_c_side
        self.f_constr_dom[2] = f_act.sum(axis=0)
        self.prev_grip_v = grip_v

        self.imp[0] += g_sp * dt + self._sp_contact_imp + self._sp_constr_imp
        self.imp[1] += g_w * dt + contact_sum * dt
        self.imp[2] += (m_dv + f_c_side.sum(axis=0) * dt)

        # grip + gripped-vertex poses at t1 (kinematic)
        self.pos[self.w1] = centers[0]
        self.pos[self.w1 + 1] = centers[1]
        self.vel[self.w1] = cdot
        self.vel[self.w1 + 1] = cdot

        self.t = t1
        self.step += 1

    # ----------------------------------------------------------- absorption
    def _absorption_step(self, free, wloc, t):
        pairs = np.argwhere(self._absorb_pairs)
        if not len(pairs):
            return
        rr = self._absorb_rr
        order = np.argsort(rr[pairs[:, 0], pairs[:, 1]])
        pairs = pairs[order]
        for pi, vi in pairs:
            gidx = free[pi]
            if self.phase[gidx] != 1:
                continue
            if self.noabs[wloc[pi]] > t:
                continue
            if self.mass[gidx] <= 0.0:
                continue
            room = self.CAP - self.m_w[vi]
            if room <= 0.0:
                continue
            mpart = self.mass[gidx]
            take = min(mpart, room)
            vp = self.vel[gidx].copy()
            M_old = self.M_DRY_V + self.m_w[vi]
            # inelastic momentum merge into the solid vertex
            self.vel[vi] = (M_old * self.vel[vi] + take * vp) / (M_old + take)
            self.m_w[vi] += take
            self.inflow[0] += take * vp
            self.outflow[1] += take * vp
            self.absorbed_cum += take
            if take >= mpart - 1e-15:
                # fully absorbed: becomes a massless stored tracer on vertex vi
                self.mass[gidx] = 0.0
                self.phase[gidx] = 2
                self.tracer_vertex[wloc[pi]] = vi
                self.stored[vi].append(gidx)
                self.pos[gidx] = self.pos[vi] + 0.010 * self.pdir[wloc[pi]]
                self.vel[gidx] = self.vel[vi]
            else:
                self.mass[gidx] = mpart - take

    # -------------------------------------------------------------- sponge
    def _sponge_advance(self, dt, f_react):
        dt_s = dt / self.NSUB
        m = self.M_DRY_V + self.m_w
        self._sp_contact_imp = np.zeros(3)
        self._sp_constr_imp = np.zeros(3)
        self._constr_side_force = np.zeros((2, 3))
        self._sp_contact_f = np.zeros((self.n_sp, 3))
        self._sp_constr_f = np.zeros((self.n_sp, 3))
        for k in range(self.NSUB):
            t_s = self.t + k * dt_s
            if not self.grabbed and t_s >= self.LIFT_T0 - 1e-9:
                centers, _, _, _ = self._grip_pose(self.LIFT_T0)
                pos = self.pos[:self.n_sp]
                for side, idx in ((0, self.left_v), (1, self.right_v)):
                    self.d0[idx] = pos[idx] - centers[side]
                self.grabbed = True
            Fs = m[:, None] * self.GRAV + f_react
            # springs
            pos = self.pos[:self.n_sp]
            vel = self.vel[:self.n_sp]
            d = pos[self.sp_eb] - pos[self.sp_ea]
            L = np.linalg.norm(d, axis=1)
            dirv = d / np.maximum(L, 1e-12)[:, None]
            fmag = self.sp_k * (L - self.sp_L0)
            dv = vel[self.sp_eb] - vel[self.sp_ea]
            fdamp = self.sp_c * np.einsum('ij,ij->i', dv, dirv)
            Fe = (fmag + fdamp)[:, None] * dirv
            np.add.at(Fs, self.sp_ea, Fe)
            np.add.at(Fs, self.sp_eb, -Fe)
            # floor contact with friction (no penetration, no teleportation)
            zc = self.Z_FLOOR + self.RV
            pen = np.clip(zc - pos[:, 2], 0.0, None)
            act = pen > 0
            if np.any(act):
                fn = np.maximum(self.KC_SP * pen - self.CC_SP * np.minimum(vel[:, 2], 0.0), 0.0) * act
                Fc = np.zeros((self.n_sp, 3))
                Fc[:, 2] = fn
                vt = vel[:, :2]
                spt = np.linalg.norm(vt, axis=1)
                ft = np.minimum(self.CT_SP * spt, self.MU_SP * fn)
                scale = np.where(spt > 1e-9, ft / np.maximum(spt, 1e-9), 0.0)
                Fc[:, 0] = -scale * vt[:, 0]
                Fc[:, 1] = -scale * vt[:, 1]
                Fs += Fc
                self._sp_contact_imp += Fc.sum(axis=0) * dt_s
                self._sp_contact_f = Fc
            vel_new = vel + Fs / m[:, None] * dt_s
            if self.grabbed:
                t_n = t_s + dt_s
                for side, idx in ((0, self.left_v), (1, self.right_v)):
                    p_t, v_t = self._constrained_targets(t_n, side)
                    f_c = m[idx, None] * (v_t - vel_new[idx]) / dt_s
                    vel_new[idx] = v_t
                    self._sp_constr_imp += f_c.sum(axis=0) * dt_s
                    self._constr_side_force[side] += f_c.sum(axis=0)
                    self._sp_constr_f[idx] = f_c
                    self.pos[idx] = p_t
            self.vel[:self.n_sp] = vel_new * 0.999
            self.pos[:self.n_sp] += self.vel[:self.n_sp] * dt_s
            np.clip(self.vel[:self.n_sp], -10.0, 10.0, out=self.vel[:self.n_sp])

    # ------------------------------------------------------------ expulsion
    def _expulsion_step(self, t):
        centers, cdot, th, thd = self._grip_pose(t)
        s = self.m_w / self.CAP
        rate = (self.K_WRING * abs(thd) * np.maximum(s - self.S_RET, 0.0)
                + self.K_DRIP * np.maximum(s - 0.9, 0.0))
        self.expulse_acc += rate * self.DT
        order = np.argsort(-self.m_w)
        for v in range(self.n_sp):
            while self.expulse_acc[v] >= 1.0:
                if self.m_w[v] < self.m_p - 1e-12:
                    self.expulse_acc[v] = 0.0
                    break
                src = v
                gidx = -1
                if self.stored[src]:
                    gidx = self.stored[src].pop()
                else:
                    for u in order:
                        if self.stored[u]:
                            gidx = self.stored[u].pop()
                            break
                if gidx < 0:
                    self.expulse_acc[v] = 0.0
                    break
                self.expulse_acc[v] -= 1.0
                w = gidx - self.w0
                direction = self.pdir[w]
                M_old = self.M_DRY_V + self.m_w[v]
                v_new_p = self.vel[v] + self.V_EJECT * direction
                M_after = M_old - self.m_p
                if M_after > 1e-9:
                    self.vel[v] = (M_old * self.vel[v] - self.m_p * v_new_p) / M_after
                self.m_w[v] -= self.m_p
                self.phase[gidx] = 1
                self.mass[gidx] = self.m_p
                self.pos[gidx] = self.pos[v] + 0.07 * direction
                self.vel[gidx] = v_new_p
                self.noabs[w] = t + 0.4
                self.outflow[0] += self.m_p * v_new_p
                self.inflow[1] += self.m_p * v_new_p
                self.expelled_cum += self.m_p

    # ------------------------------------------------------------ diffusion
    def _diffusion(self, dt):
        s = self.m_w / self.CAP
        flux = self.D_PAIR * dt * self.CAP * (s[self.diff_b] - s[self.diff_a])
        dm = np.zeros(self.n_sp)
        np.add.at(dm, self.diff_a, flux)
        np.add.at(dm, self.diff_b, -flux)
        self.m_w = np.clip(self.m_w + dm, 0.0, self.CAP)
