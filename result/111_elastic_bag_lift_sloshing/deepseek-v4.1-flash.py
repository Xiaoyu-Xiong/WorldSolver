"""Two-way coupled SPH water / elastic-bag shell solver.

Physics implemented from scratch on NumPy + SciPy:

* Water: weakly compressible SPH (cubic-spline kernel, Tait EOS, Monaghan
  artificial viscosity) with gravity, tank-wall collisions and an open top.
* Bag: dynamic triangle-mesh shell integrated as a mass-spring membrane
  (structural edges + bending/second-ring springs, Kelvin-Voigt damping),
  sub-cycled relative to the fluid so the light cloth stays stable.
* Coupling: the shell nodes act as SPH boundary particles (mirrored pressure
  plus tangential drag); the equal-and-opposite reaction deforms the shell.
  Only rim vertices are kinematic (prescribed lift trajectory); every other
  shell vertex responds to fluid traction, gravity and elastic forces.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81])


class BenchmarkSolver:
    """Deterministic lifted elastic bag carrying and spilling water."""

    FLUID_DT = 8.0e-4
    SHELL_SUBSTEPS = 8
    SOUND_SPEED = 20.0
    EOS_GAMMA = 7.0
    ALPHA_VISC = 0.2
    ALPHA_VISC_WALL = 0.5
    DENSITY_FLOOR = 0.35
    DENSITY_FLOOR_WALL = 0.6
    K_STRUCT = 4000.0
    K_BEND = 800.0
    SPRING_ZETA = 0.3
    C_TANGENTIAL = 4.0
    RESTITUTION = 0.05
    WALL_FRICTION = 0.98
    MAX_SPEED = 14.0
    MAX_PAIR_FORCE = 600.0

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.cfg = cfg
        meta = scene.metadata()
        frame0 = scene.initial_state()

        pos = np.array(frame0.position, dtype=float)
        self.n_total = int(pos.shape[0])
        body = np.asarray(frame0.body_id, dtype=int)
        self.is_bag = body == 0
        self.is_water = body == 1
        self.nb = int(np.count_nonzero(self.is_bag))
        self.nw = int(np.count_nonzero(self.is_water))

        self.pos = pos
        self.vel = np.zeros_like(pos)
        self.mass = np.array(frame0.mass, dtype=float)
        controlled = np.asarray(frame0.attributes["rim_controlled"], dtype=float)
        self.rim_mask_global = controlled > 0.5
        self.rim_bag_idx = np.nonzero(self.rim_mask_global[: self.nb])[0]
        self.rim0 = self.pos[self.rim_bag_idx].copy()

        # frozen pool / tank parameters
        pool = cfg["initial_conditions"]["pool"]
        self.spacing = float(pool["particle_spacing"])
        self.rho_fallback = float(pool["rest_density"])
        tank = cfg["boundary_conditions"]["tank"]
        self.tank_lo = np.array(tank["interior_min"], dtype=float)
        self.tank_hi = np.array(tank["interior_max"], dtype=float)
        radius = np.asarray(frame0.attributes["particle_radius"], dtype=float)
        if radius.shape[0] == self.n_total:
            self.wall_radius = np.where(self.is_water, np.maximum(radius, 0.02), 0.004)
        else:  # pragma: no cover - defensive
            self.wall_radius = np.full(self.n_total, 0.02)

        lift = dict(meta.get("rim_control", {}))
        self.lift_start = float(lift.get("start_time", 0.0))
        self.lift_dur = float(lift.get("rise_duration", 1.0e-3))
        start_z = float(np.asarray(lift.get("start_center", [0, 0, 0]))[2])
        end_z = float(np.asarray(lift.get("end_center", [0, 0, 0]))[2])
        self.lift_height = end_z - start_z
        self.rim_center0 = np.asarray(lift.get("start_center", self.rim0.mean(axis=0)), dtype=float)

        self.h = self.spacing                       # SPH smoothing length
        self.sigma = 1.0 / (np.pi * self.h ** 3)    # cubic-spline normalisation
        self.rcut = 2.0 * self.h
        self.water_mass = float(np.mean(self.mass[self.is_water]))
        self.m_b_eff = self.rho_fallback * self.spacing ** 3
        self.rho0 = self._initial_rest_density()
        self.eos_B = self.rho0 * self.SOUND_SPEED ** 2 / self.EOS_GAMMA
        self.dt = self.FLUID_DT
        self.dt_shell = self.dt / self.SHELL_SUBSTEPS

        self._build_shell_topology(meta)
        self._build_wall_samples()

        # running bookkeeping
        self.steps = 0
        self.time = 0.0
        self.n_samples_out = 0
        n_domains = 3
        self.impulse = np.zeros((n_domains, 3))
        self.domain_weights = np.zeros((n_domains, self.n_total))
        self.domain_weights[0, :] = 1.0
        self.domain_weights[1, :] = self.is_bag.astype(float)
        self.domain_weights[2, :] = self.is_water.astype(float)
        self.last_domain_force = np.zeros((n_domains, 4, 3))
        self.coupling_net_bag = np.zeros(3)
        self.shell_wall_impulse = np.zeros(3)
        self._wall_force = np.zeros(3)
        self._wall_force_bag = np.zeros(3)
        self.ctrl_force = np.zeros(3)
        self.mass_quantity = np.zeros((self.n_total, 2))
        self.mass_quantity[self.is_bag, 0] = self.mass[self.is_bag]
        self.mass_quantity[self.is_water, 1] = self.mass[self.is_water]
        self.reference_velocity = float(np.sqrt(abs(GRAVITY[2]) * max(self.tank_hi[2], 1.0)))
        total_mass = float(self.mass.sum())
        self.reference_momentum = float(total_mass * self.reference_velocity)
        self._face_loop = np.asarray(meta["topology"]["bag_faces"], dtype=np.int64)
        self._last_diag: dict[str, object] = {"steps": 0}

    # ---------------------------------------------------------- fluid helpers
    def _kernel_w(self, q):
        w = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        w[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
        w[m2] = 0.25 * (2.0 - q[m2]) ** 3
        return w * self.sigma

    def _kernel_dw(self, q):
        d = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        d[m1] = -3.0 * q[m1] + 2.25 * q[m1] ** 2
        d[m2] = -0.75 * (2.0 - q[m2]) ** 2
        return d * self.sigma / self.h

    def _initial_rest_density(self) -> float:
        pos = self.pos
        tree = cKDTree(pos)
        pairs = tree.query_pairs(self.rcut, output_type="ndarray")
        if pairs.size == 0:
            return self.rho_fallback
        i, j = pairs[:, 0], pairs[:, 1]
        r = np.linalg.norm(pos[i] - pos[j], axis=1)
        w = self._kernel_w(np.clip(r / self.h, 0.0, 2.0))
        water = self.is_water
        rho = self.water_mass * float(self._kernel_w(np.array([0.0]))[0]) * np.ones(self.n_total)
        contrib = np.zeros(self.n_total)
        mask = water[i]
        mp = np.where(water[j], self.water_mass, self.m_b_eff)
        contrib += np.bincount(i[mask], weights=(mp * w)[mask], minlength=self.n_total)
        mask = water[j]
        mp = np.where(water[i], self.water_mass, self.m_b_eff)
        contrib += np.bincount(j[mask], weights=(mp * w)[mask], minlength=self.n_total)
        rho = rho + contrib
        vals = rho[water]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return self.rho_fallback
        mean = float(np.mean(vals))
        if not (0.5 * self.rho_fallback < mean < 2.0 * self.rho_fallback):
            return self.rho_fallback
        return mean

    # ---------------------------------------------------------- shell helpers
    def _build_shell_topology(self, meta) -> None:
        faces = np.asarray(meta["topology"]["bag_faces"], dtype=np.int64)
        self.faces = faces
        edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        bag_pos = self.pos[: self.nb]
        edge_len = np.linalg.norm(bag_pos[edges[:, 0]] - bag_pos[edges[:, 1]], axis=1)
        self.edge_len_mean = float(np.mean(edge_len))

        tree = cKDTree(bag_pos)
        cand = tree.query_pairs(2.3 * self.edge_len_mean, output_type="ndarray")
        if cand.size:
            key_self = edges[:, 0] * self.nb + edges[:, 1]
            key_cand = np.sort(cand, axis=1)
            key_cand = key_cand[:, 0] * self.nb + key_cand[:, 1]
            keep = ~np.isin(key_cand, np.sort(key_self))
            bend = cand[keep]
        else:
            bend = np.zeros((0, 2), dtype=np.int64)

        if bend.shape[0]:
            si = np.concatenate([edges[:, 0], bend[:, 0]])
            sj = np.concatenate([edges[:, 1], bend[:, 1]])
            kk = np.concatenate([np.full(len(edges), self.K_STRUCT),
                                 np.full(len(bend), self.K_BEND)])
        else:
            si, sj = edges[:, 0], edges[:, 1]
            kk = np.full(len(edges), self.K_STRUCT)
        l0 = np.linalg.norm(bag_pos[si] - bag_pos[sj], axis=1)
        l0 = np.maximum(l0, 1.0e-4)
        m_eff = 0.5 * float(np.mean(self.mass[self.is_bag]))
        cc = 2.0 * self.SPRING_ZETA * np.sqrt(kk * m_eff)
        self.si, self.sj = si, sj
        self.sk, self.sl0, self.sc = kk, l0, cc

    def _build_wall_samples(self) -> None:
        lo, hi = self.tank_lo, self.tank_hi
        step = 0.2
        xs = np.arange(lo[0] + 0.1, hi[0] - 0.05, step)
        ys = np.arange(lo[1] + 0.1, hi[1] - 0.05, step)
        zs = np.arange(lo[2] + 0.1, hi[2] - 0.05, 0.25)
        pts, nrm = [], []
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts.append(np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, lo[2])]))
        nrm.append(np.tile(np.array([0.0, 0.0, 1.0]), (gx.size, 1)))
        gy2, gz2 = np.meshgrid(ys, zs, indexing="ij")
        for plane, sign in ((lo[0], 1.0), (hi[0], -1.0)):
            pts.append(np.column_stack([np.full(gy2.size, plane), gy2.ravel(), gz2.ravel()]))
            nrm.append(np.tile(np.array([sign, 0.0, 0.0]), (gy2.size, 1)))
        gx2, gz3 = np.meshgrid(xs, zs, indexing="ij")
        pts.append(np.column_stack([gx2.ravel(), np.full(gx2.size, lo[1]), gz3.ravel()]))
        nrm.append(np.tile(np.array([0.0, 1.0, 0.0]), (gx2.size, 1)))
        self.wall_pts = np.vstack(pts)
        self.wall_nrm = np.vstack(nrm)
        self.n_wall = int(self.wall_pts.shape[0])

    # ------------------------------------------------------------- rim control
    def _rim_kinematics(self, t: float):
        if t <= self.lift_start:
            s = 0.0
        elif t >= self.lift_start + self.lift_dur:
            s = 1.0
        else:
            s = (t - self.lift_start) / self.lift_dur
        dz = self.lift_height * (s * s * (3.0 - 2.0 * s))
        ddz = 0.0
        if 0.0 < s < 1.0:
            ddz = self.lift_height * 6.0 * s * (1.0 - s) / self.lift_dur
        return dz, ddz

    # -------------------------------------------------------------- one step
    def _step(self) -> None:
        dt = self.dt
        nb, nw, nt = self.nb, self.nw, self.n_total
        water = self.is_water
        pos, vel, mass = self.pos, self.vel, self.mass

        tree = cKDTree(pos)
        pairs = tree.query_pairs(self.rcut, output_type="ndarray")
        if pairs.size:
            ii, jj = pairs[:, 0], pairs[:, 1]
            wi = water[ii]
            wj = water[jj]
            ww = wi & wj
        else:
            ii = jj = np.zeros(0, dtype=np.int64)
            ww = np.zeros(0, dtype=bool)

        rho = self.water_mass * float(self._kernel_w(np.array([0.0]))[0]) * np.ones(nw)
        acc_w = np.tile(GRAVITY, (nw, 1)).copy()
        self.coupling_net_bag = np.zeros(3)
        f_pair_cap = self.MAX_PAIR_FORCE

        # ---------------- water/water SPH pressure + viscosity
        if np.any(ww):
            i = ii[ww] - nb
            j = jj[ww] - nb
            rvec = pos[ii[ww]] - pos[jj[ww]]
            r2 = np.einsum("ij,ij->i", rvec, rvec)
            r2 = np.maximum(r2, 1.0e-8)
            r = np.sqrt(r2)
            q = r / self.h
            w = self._kernel_w(q)
            dw = self._kernel_dw(q)
            rho += (np.bincount(i, weights=self.water_mass * w, minlength=nw)
                    + np.bincount(j, weights=self.water_mass * w, minlength=nw))
            # keep for force stage
            ww_i, ww_j, ww_rvec, ww_r, ww_dw = i, j, rvec, r, dw
        else:
            ww_i = ww_j = np.zeros(0, dtype=np.int64)
            ww_rvec = np.zeros((0, 3))
            ww_r = np.zeros(0)
            ww_dw = np.zeros(0)

        # ---------------- water/bag boundary pairs
        bw = water[ii] & ~water[jj]
        wb = ~water[ii] & water[jj]
        if np.any(bw) or np.any(wb):
            wi_idx = np.concatenate([ii[bw] - nb, jj[wb] - nb])
            bi_idx = np.concatenate([jj[bw], ii[wb]])
            rvec_bw = pos[wi_idx + nb] - pos[bi_idx]
            r2b = np.maximum(np.einsum("ij,ij->i", rvec_bw, rvec_bw), 1.0e-8)
            rb = np.sqrt(r2b)
            qb = rb / self.h
            rho += np.bincount(wi_idx, weights=self.m_b_eff * self._kernel_w(qb), minlength=nw)
            dwb = self._kernel_dw(qb)
        else:
            wi_idx = bi_idx = np.zeros(0, dtype=np.int64)
            rvec_bw = np.zeros((0, 3))
            rb = np.zeros(0)
            dwb = np.zeros(0)

        inv_mw = 1.0 / self.water_mass
        rho = np.maximum(rho, self.DENSITY_FLOOR * self.rho0)
        p = self.eos_B * (np.maximum(rho / self.rho0, 1.0e-6) ** self.EOS_GAMMA - 1.0)
        p = np.maximum(p, -0.15 * self.eos_B)
        pinv = 1.0 / rho

        # ---------------- water/water momentum
        if ww_i.size:
            vrel = vel[ww_i + nb] - vel[ww_j + nb]
            vdotr = np.einsum("ij,ij->i", vrel, ww_rvec)
            mu = self.h * vdotr / (ww_r ** 2 + 0.01 * self.h ** 2)
            pi = np.where(vdotr < 0.0,
                          -self.ALPHA_VISC * self.SOUND_SPEED * mu * 0.5 * (pinv[ww_i] + pinv[ww_j]),
                          0.0)
            coef = -(p[ww_i] * pinv[ww_i] ** 2 + p[ww_j] * pinv[ww_j] ** 2 + pi) * ww_dw / ww_r
            f = (self.water_mass ** 2) * coef[:, None] * ww_rvec
            mag = np.linalg.norm(f, axis=1)
            scale = np.where(mag > f_pair_cap, f_pair_cap / np.maximum(mag, 1e-12), 1.0)
            f *= scale[:, None]
            for c in range(3):
                acc_w[:, c] += np.bincount(ww_i, weights=f[:, c], minlength=nw) * inv_mw
                acc_w[:, c] -= np.bincount(ww_j, weights=f[:, c], minlength=nw) * inv_mw

        # ---------------- water/bag coupling (boundary-particle SPH)
        couple_bag = np.zeros((nb, 3))
        if wi_idx.size:
            vrel = vel[wi_idx + nb] - vel[bi_idx]
            vdotr = np.einsum("ij,ij->i", vrel, rvec_bw)
            mu = self.h * vdotr / (rb ** 2 + 0.01 * self.h ** 2)
            pi = np.where(vdotr < 0.0,
                          -self.ALPHA_VISC_WALL * self.SOUND_SPEED * mu * pinv[wi_idx], 0.0)
            p_b = np.maximum(p[wi_idx], 0.0) * self.DENSITY_FLOOR_WALL * 3.0
            pterm = p[wi_idx] * pinv[wi_idx] ** 2 + p_b / self.rho0 ** 2 + pi
            coef = -pterm * dwb / rb
            f = (self.water_mass * self.m_b_eff) * coef[:, None] * rvec_bw
            mag = np.linalg.norm(f, axis=1)
            scale = np.where(mag > f_pair_cap, f_pair_cap / np.maximum(mag, 1e-12), 1.0)
            f *= scale[:, None]
            # tangential drag (pairwise, momentum conserving)
            rhat = rvec_bw / rb[:, None]
            vtan = vrel - np.einsum("ij,ij->i", vrel, rhat)[:, None] * rhat
            f_drag = -self.C_TANGENTIAL * (self.water_mass * self.m_b_eff) ** 0.5 * vtan
            f = f + f_drag
            for c in range(3):
                acc_w[:, c] += np.bincount(wi_idx, weights=f[:, c], minlength=nw) * inv_mw
                couple_bag[:, c] += np.bincount(bi_idx, weights=f[:, c], minlength=nb)
            self.coupling_net_bag = couple_bag.sum(axis=0)

        # ---------------- integrate fluid (symplectic Euler)
        speed = np.linalg.norm(vel[nb:], axis=1)
        scale = np.where(speed > self.MAX_SPEED, self.MAX_SPEED / np.maximum(speed, 1e-12), 1.0)
        acc_w *= scale[:, None]
        vel[nb:] += acc_w * dt
        pos[nb:] += vel[nb:] * dt
        self._clamp_walls(nb, nt, dt)

        # ---------------- shell sub-cycled integration
        self.shell_wall_impulse = np.zeros(3)
        self._advance_shell(dt, couple_bag)

        # ---------------- cumulative external impulse per budget domain
        grav_all = GRAVITY * float(self.mass.sum())
        grav_bag = GRAVITY * float(np.sum(self.mass[:nb]))
        grav_water = grav_all - grav_bag
        wall_all = self._wall_force
        wall_bag = self._wall_force_bag + self.shell_wall_impulse / dt
        wall_water = wall_all - self._wall_force_bag
        ctrl = self.ctrl_force
        ext_world = grav_all + wall_all + ctrl
        ext_bag = grav_bag + wall_bag + ctrl + self.coupling_net_bag
        ext_water = grav_water + wall_water - self.coupling_net_bag
        self.impulse[0] += ext_world * dt
        self.impulse[1] += ext_bag * dt
        self.impulse[2] += ext_water * dt
        self.last_domain_force[0, 1] = wall_all
        self.last_domain_force[1, 1] = wall_bag
        self.last_domain_force[2, 1] = wall_water
        self.steps += 1
        self.time = self.steps * self.dt

    # -------------------------------------------------------- wall collisions
    def _clamp_walls(self, nb: int, nt: int, dt: float) -> None:
        """Water-only tank-wall collision (shell nodes are handled separately)."""
        water = self.is_water
        pos = self.pos
        vel = self.vel
        mass = self.mass
        lo = self.tank_lo + self.wall_radius
        hi = self.tank_hi - self.wall_radius
        impulse = np.zeros(3)
        for axis in range(3):
            low = water & (pos[:, axis] < lo[axis])
            if np.any(low):
                pos[low, axis] = lo[axis]
                v = vel[low, axis]
                dv = -(1.0 + self.RESTITUTION) * np.minimum(v, 0.0)
                vel[low, axis] = v + dv
                impulse[axis] += float(np.sum(mass[low] * dv))
            high = water & (pos[:, axis] > hi[axis])
            if np.any(high):
                pos[high, axis] = hi[axis]
                v = vel[high, axis]
                dv = -(1.0 + self.RESTITUTION) * np.maximum(v, 0.0)
                vel[high, axis] = v + dv
                impulse[axis] += float(np.sum(mass[high] * dv))
        # tangential wall friction
        for axis in range(3):
            contact = water & ((np.abs(pos[:, axis] - lo[axis]) < 1e-9)
                               | (np.abs(pos[:, axis] - hi[axis]) < 1e-9))
            if np.any(contact):
                for a in range(3):
                    if a == axis:
                        continue
                    v = vel[contact, a]
                    dv = (self.WALL_FRICTION - 1.0) * v
                    vel[contact, a] = v + dv
                    impulse[a] += float(np.sum(mass[contact] * dv))
        self._wall_force = impulse / dt
        self._wall_force_bag = np.zeros(3)

    # ------------------------------------------------------------ shell step
    def _advance_shell(self, dt: float, couple_bag: np.ndarray) -> None:
        dts = self.dt_shell
        nb = self.nb
        pos = self.pos[:nb]
        vel = self.vel[:nb]
        mass = self.mass[:nb]
        invm = 1.0 / mass
        rim = self.rim_bag_idx
        rim_mask = np.zeros(nb, dtype=bool)
        rim_mask[rim] = True
        dyn = ~rim_mask
        si, sj, sk, sl0, sc = self.si, self.sj, self.sk, self.sl0, self.sc
        gz = GRAVITY[None, :].repeat(nb, axis=0) * mass[:, None]
        ctrl_impulse = np.zeros(3)
        start_time = self.time
        for sub in range(self.SHELL_SUBSTEPS):
            rvec = pos[sj] - pos[si]
            d = np.sqrt(np.maximum(np.einsum("ij,ij->i", rvec, rvec), 1.0e-10))
            rhat = rvec / d[:, None]
            vrel = vel[sj] - vel[si]
            fmag = sk * (d - sl0) + sc * np.einsum("ij,ij->i", vrel, rhat)
            f = fmag[:, None] * rhat
            force = np.zeros((nb, 3))
            for c in range(3):
                force[:, c] = (np.bincount(si, weights=f[:, c], minlength=nb)
                               - np.bincount(sj, weights=f[:, c], minlength=nb))
            force += gz + couple_bag
            acc = force * invm[:, None]
            vel[dyn] += acc[dyn] * dts
            pos += vel * dts
            # prescribed rim trajectory (rigid z translation, shape preserved)
            t_sub = start_time + (sub + 1) * dts
            dz, ddz = self._rim_kinematics(t_sub)
            target_pos = self.rim0 + np.array([0.0, 0.0, dz])
            target_vel = np.array([0.0, 0.0, ddz])
            for c in range(3):
                ctrl_impulse[c] += float(np.sum(mass[rim] * (target_vel[c] - vel[rim, c])))
                ctrl_impulse[c] -= float(np.sum(force[rim, c]) * dts)
            pos[rim] = target_pos
            vel[rim] = target_vel
            # non-rim shell nodes obey the tank walls
            self._clamp_shell(pos, vel, dyn)
        self.coupling_net_bag = couple_bag.sum(axis=0)
        self.ctrl_force = ctrl_impulse / dt
        self.last_domain_force[0, 2] = self.ctrl_force
        self.last_domain_force[1, 2] = self.ctrl_force

    def _clamp_shell(self, pos, vel, dyn) -> None:
        lo = self.tank_lo + 0.004
        hi = self.tank_hi - 0.004
        idx = np.nonzero(dyn)[0]
        mass = self.mass[: self.nb]
        for axis in range(3):
            low = pos[idx, axis] < lo[axis]
            if np.any(low):
                sel = idx[low]
                pos[sel, axis] = lo[axis]
                v = vel[sel, axis]
                dv = -(1.0 + self.RESTITUTION) * np.minimum(v, 0.0)
                vel[sel, axis] = v + dv
                self.shell_wall_impulse[axis] += float(np.sum(mass[sel] * dv))
            high = pos[idx, axis] > hi[axis]
            if np.any(high):
                sel = idx[high]
                pos[sel, axis] = hi[axis]
                v = vel[sel, axis]
                dv = -(1.0 + self.RESTITUTION) * np.maximum(v, 0.0)
                vel[sel, axis] = v + dv
                self.shell_wall_impulse[axis] += float(np.sum(mass[sel] * dv))

    # -------------------------------------------------------------- advancing
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        n = int(round((target - self.time) / self.dt))
        if n < 0:
            n = 0
        for _ in range(n):
            self._step()
        return self._build_frame(self.time)

    # ------------------------------------------------------------- frame build
    def _build_frame(self, t: float) -> StateFrame:
        pos = self.pos.copy()
        vel = self.vel.copy()
        mass = self.mass.copy()
        nb = self.nb

        grav_bag = float(np.sum(mass[:nb])) * GRAVITY
        grav_water = float(np.sum(mass[nb:])) * GRAVITY
        grav_all = grav_bag + grav_water
        couple_bag = self.coupling_net_bag.copy()
        ctrl = getattr(self, "ctrl_force", np.zeros(3))
        wall = np.zeros((3, 3))
        wall[0] = self.last_domain_force[0, 1]
        wall[1] = self.last_domain_force[1, 1]
        wall[2] = self.last_domain_force[2, 1]

        ext = np.zeros((3, 4, 3))
        ext[0, 0] = grav_all
        ext[1, 0] = grav_bag
        ext[2, 0] = grav_water
        ext[0, 1] = wall[0]
        ext[1, 1] = wall[1]
        ext[2, 1] = wall[2]
        ext[0, 2] = ctrl
        ext[1, 2] = ctrl
        ext[2, 2] = 0.0
        ext[0, 3] = 0.0
        ext[1, 3] = couple_bag
        ext[2, 3] = -couple_bag
        self.last_domain_force = ext

        if t > self.time - 0.5 * self.dt or self.steps == 0:
            pass
        # rim normals from the current shell configuration
        f = self._face_loop
        v0 = pos[f[:, 0]]
        v1 = pos[f[:, 1]]
        v2 = pos[f[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-12)[:, None]
        node_n = np.zeros((nb, 3))
        for c in range(3):
            for a in range(3):
                node_n[:, c] += np.bincount(f[:, a], weights=fn[:, c], minlength=nb)
        node_n /= np.maximum(np.linalg.norm(node_n, axis=1), 1e-12)[:, None]
        rim = self.rim_bag_idx
        rim_n = node_n[rim]
        rim_n[:, 2] = 0.0
        nn = np.linalg.norm(rim_n, axis=1)
        bad = nn < 1e-9
        rim_n[bad] = np.array([1.0, 0.0, 0.0])
        rim_n /= np.maximum(np.linalg.norm(rim_n, axis=1), 1e-12)[:, None]

        boundary_normal = np.vstack([self.wall_nrm, rim_n])
        boundary_velocity = np.vstack([np.zeros((self.n_wall, 3)), vel[rim]])
        dz, ddz = self._rim_kinematics(t)
        target = np.zeros_like(boundary_velocity)
        target[self.n_wall:, 2] = ddz
        rim_pos = self.rim0 + np.array([0.0, 0.0, dz])
        boundary_normal[self.n_wall:, :] = rim_n

        scalars = {
            "reference_velocity": float(self.reference_velocity),
            "reference_momentum": float(self.reference_momentum),
            "water_rest_density": float(self.rho0),
            "bag_thickness": float(self.cfg.get("initial_conditions", {}).get("bag", {}).get("thickness", 1.5e-3)),
        }
        fields = {
            "boundary_normal": boundary_normal,
            "boundary_velocity": boundary_velocity,
            "boundary_target_velocity": target,
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": ext.copy(),
            "budget_cumulative_external_impulse": self.impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((3, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((3, 3)),
            "cumulative_mass_source": np.zeros(2),
            "interaction_force": np.vstack([couple_bag, -couple_bag]),
            "gravity_acceleration": GRAVITY[None, :].copy(),
        }
        attributes = {
            "mass_quantity": self.mass_quantity.copy(),
            "material_phase": (self.is_water.astype(float)),
            "particle_radius": np.asarray(
                np.where(self.is_water, self.wall_radius, 0.008), dtype=float),
        }
        frame = StateFrame(
            time=float(t),
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=np.where(self.is_bag, 0, 1).astype(np.int32),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        self.n_samples_out += 1
        self._last_diag = {
            "time": float(t),
            "steps": self.steps,
            "max_speed": float(np.max(np.linalg.norm(vel, axis=1))),
            "water_z_max": float(np.max(pos[nb:, 2])),
            "bag_min_z": float(np.min(pos[:nb, 2])),
        }
        return frame

    def diagnostics(self) -> dict[str, object]:
        return dict(self._last_diag)
