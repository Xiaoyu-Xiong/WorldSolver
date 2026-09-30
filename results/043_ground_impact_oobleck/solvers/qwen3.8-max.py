"""Physically-based particle solver for the ground-impact oobleck benchmark.

Model: a bonded discrete-element continuum (peridynamics-style spring network)
for two granular bodies plus penalty contact with the ground plane.

* Each body is sampled as particles (from ``scene.initial_state``).  A fixed
  material bond network is built once from the initial configuration
  (neighbors within 1.45 * particle spacing).
* Bonds carry an axial spring (with damage-softened stiffness), axial and
  tangential dashpots, rest-length plasticity (viscoplastic flow), fatigue
  damage, and brittle rupture.
* Body 1 (warm coral) uses a viscoplastic law: tiny yield strain, fast
  rest-length flow, strong damping, early rupture -> it splats and flows on
  the first impact.
* Body 2 (yellow-green oobleck) uses a shear-thickening law: stiff elastic
  bonds with light axial damping (stores impact energy -> first rebound),
  strongly shear-rate-thickening tangential damping, high yield strain, and
  cumulative damage that softens/plasticizes the network so the slower second
  impact collapses it into a flowing, spreading puddle.
* Short-range dynamics (rebuilt neighbor list): hard core repulsion
  (excluded volume), smooth cohesion (surface tension / keeps piles together),
  and pair damping (viscosity of the fluidized phase).
* Ground: penalty normal spring + dashpot, Coulomb-limited tangential
  friction.  Gravity from the scene config.

Explicit symplectic Euler with dt = 5e-4 s, stable for the stiffest pair
(bond + core) and the ground spring.  All required trajectory fields are
exported per frame.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------- lifecycle ----------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        frame0 = scene.initial_state()
        meta = scene.metadata()
        config = scene.config
        initial = config["initial_conditions"]

        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.floor_z = float(config.get("boundary_conditions", {}).get("ground", {}).get("plane_z", 0.0))
        self.duration = float(config.get("duration", 2.8))
        self.sample_dt = float(config.get("sample_dt", 0.02))

        self.n = int(frame0.position.shape[0])
        self.pos = np.array(frame0.position, dtype=np.float64)
        self.vel = np.array(frame0.velocity, dtype=np.float64)
        self.mass = np.array(frame0.mass, dtype=np.float64)
        self.body_id = np.array(frame0.body_id)
        self.rest_position = np.array(frame0.position, dtype=np.float64)
        self.particle_radius = np.array(frame0.attributes.get(
            "particle_radius", np.full(self.n, 0.5 * float(initial["particle_spacing"]))))
        self.r_p = float(np.mean(self.particle_radius))

        self.dx = float(initial.get("particle_spacing", 0.045))
        self.t = 0.0
        self.step = 0
        self.dt = 5.0e-4
        self.rebuild_every = 4
        self.steps_done = 0

        # --- interaction ranges ---
        h_coh = 2.0 * self.dx            # cohesion / pair-damping cutoff
        self.h_coh = h_coh
        self.dcut_list = 2.55 * self.dx  # pair-list cutoff (buffer for rebuilds)
        self.dcell = self.dcut_list      # uniform-grid cell size
        self.r_core = 0.85 * self.dx     # hard-core repulsion radius
        self.k_core = 6.0e4              # core repulsion stiffness [N/m]
        self.c_coh = 2.5                 # cohesion magnitude [N] at r -> r_core
        self.c_dyn_ax = 22.0             # pair axial damping [N s/m]
        self.c_dyn_tg = 12.0              # pair tangential damping [N s/m]
        self.f_pair_max = 200.0          # clamp on pair force magnitude [N]

        # --- ground contact ---
        self.k_ground = 1.0e5
        zeta_g = 0.04
        self.c_ground = 2.0 * zeta_g * np.sqrt(self.k_ground * self.mass)
        self.mu_ground = 0.55
        self.c_fric = 200.0
        self.f_ground_max = 5.0e3

        # --- bond network ---
        self._build_bonds()

        # --- per-body material laws (per bond arrays) ---
        is_obl = self.b_body == 2
        self.b_k0 = np.where(is_obl, 7.0e4, 6.0e4)
        self.b_cax = np.where(is_obl, 3.0, 30.0)
        self.b_ctg = np.where(is_obl, 1.5, 25.0)
        self.b_st_gain = np.where(is_obl, 6.0, 0.0)   # shear-thickening gain
        self.b_st_v = 0.25                            # shear-rate scale [m/s]
        self.b_eps_y = np.where(is_obl, 0.09, 0.015)  # yield strain
        self.b_beta_pl = np.where(is_obl, 60.0, 500.0)  # plastic flow rate [1/s]
        self.b_break = np.where(is_obl, 0.50, 0.15)   # rupture strain
        self.b_D0 = np.where(is_obl, 100.0, 300.0)    # damage rate [1/s]
        self.b_eps_d = np.where(is_obl, 0.06, 0.03)   # damage threshold strain
        self.b_dam = np.zeros(self.n_bonds)
        self.b_alive = np.ones(self.n_bonds, dtype=bool)
        self.alive_idx = np.flatnonzero(self.b_alive)
        self.f_bond_max = 800.0

        # --- deformation gradient operators (material coords = rest) ---
        self._build_defgrad_operators()

        # --- dynamic pair list (built lazily) ---
        self.dsrc = np.zeros(0, dtype=np.int64)
        self.ddst = np.zeros(0, dtype=np.int64)
        self._rebuild_pairs()

        # diagnostics accumulators
        self.first_contact_t = None
        self.min_com_z_body2 = float("inf")
        self.stats = {}

    # ---------------- neighbor search ----------------
    def _neighbor_pairs(self, pos, cell, cutoff):
        """Uniform-grid all-pairs within cutoff, deduplicated (i<j)."""
        inv = 1.0 / cell
        ixyz = np.floor(pos * inv).astype(np.int64)
        ixyz -= ixyz.min(axis=0) - 1
        P = np.int64(4096)
        key = (ixyz[:, 0] * P + ixyz[:, 1]) * P + ixyz[:, 2]
        order = np.argsort(key, kind="stable")
        skeys = key[order]
        n = pos.shape[0]
        ar = np.arange(n)
        src_parts = []
        dst_parts = []
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    nkey = ((ixyz[:, 0] + ox) * P + (ixyz[:, 1] + oy)) * P + (ixyz[:, 2] + oz)
                    lo = np.searchsorted(skeys, nkey, side="left")
                    hi = np.searchsorted(skeys, nkey, side="right")
                    cnt = hi - lo
                    tot = int(cnt.sum())
                    if tot == 0:
                        continue
                    s = np.repeat(ar, cnt)
                    base = np.repeat(lo, cnt)
                    within = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    dst_parts.append(order[base + within])
                    src_parts.append(s)
        if not src_parts:
            e = np.zeros(0, dtype=np.int64)
            return e, e
        src = np.concatenate(src_parts)
        dst = np.concatenate(dst_parts)
        m = src < dst
        src = src[m]
        dst = dst[m]
        dvec = pos[dst] - pos[src]
        r2 = np.einsum("ij,ij->i", dvec, dvec)
        m2 = r2 < cutoff * cutoff
        return src[m2], dst[m2]

    def _build_bonds(self):
        cutoff = 1.45 * self.dx
        srcs = []
        dsts = []
        dXs = []
        bodies = []
        for bid in (1, 2):
            idx = np.flatnonzero(self.body_id == bid)
            local = self.rest_position[idx]
            ls, ld = self._neighbor_pairs(local, cutoff, cutoff)
            srcs.append(idx[ls])
            dsts.append(idx[ld])
            dXs.append(local[ld] - local[ls])
            bodies.append(np.full(len(ls), bid, dtype=np.int64))
        self.b_i = np.concatenate(srcs).astype(np.int64)
        self.b_j = np.concatenate(dsts).astype(np.int64)
        self.b_dX = np.concatenate(dXs)
        self.b_body = np.concatenate(bodies)
        self.n_bonds = len(self.b_i)
        self.b_rest0 = np.linalg.norm(self.b_dX, axis=1)
        self.b_rest = self.b_rest0.copy()

    def _rebuild_pairs(self):
        self.dsrc, self.ddst = self._neighbor_pairs(self.pos, self.dcell, self.dcut_list)

    # ---------------- deformation gradient ----------------
    def _build_defgrad_operators(self):
        n, nb = self.n, self.n_bonds
        dX = self.b_dX
        comps = np.empty((nb, 9))
        for a in range(3):
            for b in range(3):
                comps[:, 3 * a + b] = dX[:, a] * dX[:, b]
        M = np.zeros((n, 9))
        for k in range(9):
            M[:, k] = (np.bincount(self.b_i, weights=comps[:, k], minlength=n)
                       + np.bincount(self.b_j, weights=comps[:, k], minlength=n))
        self._dg_A = np.linalg.pinv(M.reshape(n, 3, 3))
        self._dg_dX_static = dX

    def _deformation_gradient(self):
        u = self.pos - self.rest_position
        du = u[self.b_j] - u[self.b_i]
        dX = self._dg_dX_static
        n, nb = self.n, self.n_bonds
        G = np.zeros((n, 9))
        for a in range(3):
            for b in range(3):
                w = du[:, a] * dX[:, b]
                G[:, 3 * a + b] = (np.bincount(self.b_i, weights=w, minlength=n)
                                   + np.bincount(self.b_j, weights=w, minlength=n))
        F = np.eye(3)[None, :, :] + np.einsum("nab,nbc->nac", G.reshape(n, 3, 3), self._dg_A)
        return np.clip(F, -50.0, 50.0)

    # ---------------- forces ----------------
    def _accumulate(self, forces, src, dst, fvec):
        idx = np.concatenate((src, dst))
        for k in range(3):
            w = np.concatenate((fvec[:, k], -fvec[:, k]))
            forces[:, k] += np.bincount(idx, weights=w, minlength=self.n)

    def _ground_force(self, pos, vel):
        """Penalty ground contact; returns (force[N,3], normal force[N])."""
        f = np.zeros_like(pos)
        pen = self.r_p - (pos[:, 2] - self.floor_z)
        act = pen > 0.0
        if np.any(act):
            fn = self.k_ground * pen[act] - self.c_ground[act] * vel[act, 2]
            fn = np.clip(fn, 0.0, self.f_ground_max)
            f[act, 2] += fn
            vt = vel[act, :2]
            vtm = np.linalg.norm(vt, axis=1)
            ft = np.minimum(self.mu_ground * fn, self.c_fric * vtm)
            ok = vtm > 1e-9
            ft_vec = np.zeros_like(vt)
            ft_vec[ok] = -ft[ok, None] * (vt[ok] / vtm[ok, None])
            f[act, 0] += ft_vec[:, 0]
            f[act, 1] += ft_vec[:, 1]
        return f, np.where(act, np.maximum(f[:, 2], 0.0), 0.0)

    def _substep(self, dt):
        n = self.n
        pos, vel = self.pos, self.vel

        if self.step % self.rebuild_every == 0:
            self._rebuild_pairs()

        forces = np.empty((n, 3))
        forces[:, 0] = self.mass * self.gravity[0]
        forces[:, 1] = self.mass * self.gravity[1]
        forces[:, 2] = self.mass * self.gravity[2]

        # ---- dynamic short-range pairs: core repulsion + cohesion + damping ----
        src, dst = self.dsrc, self.ddst
        if len(src):
            dvec = pos[dst] - pos[src]
            r = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
            r = np.maximum(r, 1e-9)
            rhat = dvec / r[:, None]
            vrel = vel[dst] - vel[src]
            vr = np.einsum("ij,ij->i", vrel, rhat)
            # normal: repulsive core + cohesive tail + axial damping
            fmag = np.where(r < self.r_core, self.k_core * (self.r_core - r), 0.0)
            w = np.clip((self.h_coh - r) / max(self.h_coh - self.r_core, 1e-12), 0.0, 1.0)
            fmag -= self.c_coh * w
            fmag -= self.c_dyn_ax * vr
            fmag = np.clip(fmag, -self.f_pair_max, self.f_pair_max)
            # tangential damping
            vt = vrel - vr[:, None] * rhat
            ft = -self.c_dyn_tg * vt
            ft_norm = np.linalg.norm(ft, axis=1)
            over = ft_norm > self.f_pair_max
            if np.any(over):
                ft[over] *= (self.f_pair_max / ft_norm[over])[:, None]
            fvec = fmag[:, None] * rhat + ft
            self._accumulate(forces, src, dst, fvec)

        # ---- material bonds ----
        bi = self.b_i[self.alive_idx]
        bj = self.b_j[self.alive_idx]
        if len(bi):
            ai = self.alive_idx
            rest = self.b_rest[ai]
            rest0 = self.b_rest0[ai]
            k0 = self.b_k0[ai]
            dam = self.b_dam[ai]
            dvec = pos[bj] - pos[bi]
            r = np.sqrt(np.einsum("ij,ij->i", dvec, dvec))
            r = np.maximum(r, 1e-9)
            rhat = dvec / r[:, None]
            vrel = vel[bj] - vel[bi]
            vr = np.einsum("ij,ij->i", vrel, rhat)
            vt = vrel - vr[:, None] * rhat
            vtm = np.linalg.norm(vt, axis=1)
            s = (r - rest) / rest
            # axial spring (damage-softened) + axial dashpot
            fmag = k0 * (1.0 - dam) * (r - rest) - self.b_cax[ai] * vr
            fmag = np.clip(fmag, -self.f_bond_max, self.f_bond_max)
            # tangential dashpot with shear thickening (oobleck)
            stf = 1.0 + self.b_st_gain[ai] * np.minimum(vtm / self.b_st_v, 1.0)
            ft = -(self.b_ctg[ai] * stf)[:, None] * vt
            ft_norm = np.linalg.norm(ft, axis=1)
            over = ft_norm > self.f_bond_max
            if np.any(over):
                ft[over] *= (self.f_bond_max / ft_norm[over])[:, None]
            fvec = fmag[:, None] * rhat + ft
            self._accumulate(forces, bi, bj, fvec)

            # ---- plastic rest-length flow ----
            eps_y_eff = self.b_eps_y[ai] * (1.0 - dam) ** 2
            excess = np.abs(s) - eps_y_eff
            rate = self.b_beta_pl[ai] * np.clip(excess, 0.0, 0.3)
            new_rest = rest * (1.0 + np.sign(s) * rate * dt)
            new_rest = np.clip(new_rest, 0.6 * rest0, 1.7 * rest0)
            self.b_rest[ai] = np.where(excess > 0.0, new_rest, rest)

            # ---- fatigue damage and rupture ----
            grow = self.b_D0[ai] * dt * np.clip(np.abs(s) - self.b_eps_d[ai], 0.0, None)
            new_dam = np.minimum(dam + grow, 1.0)
            self.b_dam[ai] = new_dam
            strain0 = np.abs(r / rest0 - 1.0)
            broke = (strain0 > self.b_break[ai]) | (new_dam >= 1.0)
            if np.any(broke):
                self.b_alive[ai[broke]] = False
                self.alive_idx = np.flatnonzero(self.b_alive)

        # ---- ground contact ----
        gf, _ = self._ground_force(pos, vel)
        forces += gf

        # ---- integrate (symplectic Euler) ----
        vel += forces * (dt / self.mass)[:, None]
        speed = np.linalg.norm(vel, axis=1)
        over = speed > 40.0
        if np.any(over):
            vel[over] *= (40.0 / speed[over])[:, None]
        pos += vel * dt
        # absolute floor guard (should never engage with a working penalty)
        below = pos[:, 2] < self.floor_z
        if np.any(below):
            pos[below, 2] = self.floor_z
            vel[below, 2] = np.maximum(vel[below, 2], 0.0)

        if self.first_contact_t is None and np.any(self.r_p - (pos[:, 2] - self.floor_z) > 1e-6):
            self.first_contact_t = float(self.t + dt)
        self.step += 1
        self.steps_done += 1

    # ---------------- frame output ----------------
    def _make_frame(self, time):
        gf, _ = self._ground_force(self.pos, self.vel)
        ext = self.mass[:, None] * self.gravity[None, :] + gf

        F = self._deformation_gradient()

        gap = self.pos[:, 2] - self.floor_z - self.r_p
        sel = np.flatnonzero(gap < 0.30)
        if len(sel) < 8:
            sel = np.argsort(gap)[:64]
        contact_sd = gap[sel]

        ke = float(0.5 * np.sum(self.mass * np.einsum("ij,ij->i", self.vel, self.vel)))
        com2 = self.pos[self.body_id == 2].mean(axis=0)

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass.copy(),
                "rest_position": self.rest_position.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.body_id.copy(),
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": ext,
            },
            scalars={
                "contact_length_scale": float(self.dx),
                "total_mass": float(np.sum(self.mass)),
                "kinetic_energy": ke,
                "oobleck_com_height": float(com2[2]),
                "oobleck_broken_bond_fraction": float(
                    1.0 - np.count_nonzero(self.b_alive) / max(self.n_bonds, 1)),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        time = float(time)
        guard = 0
        while time - self.t > 1e-9:
            dt = min(self.dt, time - self.t)
            self._substep(dt)
            self.t += dt
            guard += 1
            if guard > int(4 * self.duration / self.dt) + 10000:
                break
        self.t = time
        return self._make_frame(time)

    def diagnostics(self):
        return {
            "method": "bonded_particle_continuum_shear_thickening_oobleck",
            "n_particles": int(self.n),
            "n_bonds": int(self.n_bonds),
            "dt": self.dt,
            "steps": int(self.steps_done),
            "first_contact_time": self.first_contact_t,
            "broken_bond_fraction": float(
                1.0 - np.count_nonzero(self.b_alive) / max(self.n_bonds, 1)),
        }
