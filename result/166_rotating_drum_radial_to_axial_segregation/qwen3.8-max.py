"""Discrete-element solver for the rotating-drum radial-to-axial segregation scene.

Physics
-------
Soft-sphere (linear spring-dashpot) DEM for N ~ 4000 frictional spheres inside a
horizontal cylinder of radius R rotating about the x axis with angular speed
omega.  Forces on each grain:

  * gravity (external force field),
  * grain-grain contacts: linear normal spring + dashpot, tangential spring with
    contact-history memory limited by Coulomb friction |Ft| <= mu*Fn,
  * grain-wall contacts (cylindrical mantle and the two flat end caps): the wall
    is kinematic with surface velocity v_w = omega x (p - c); same linear
    contact law with friction, which is what drags the bed around and creates
    the flowing surface layer that drives percolation (radial segregation) and
    axial band formation.

Integration is semi-implicit Euler with a fixed step well below the Rayleigh
contact-time fraction, plus a hard geometric containment guard so grains can
never leave the drum (no teleportation, no numerical explosion).

All StateFrame fields required by the trajectory contract are exported:
state.mass, state.velocity, attributes.mass_quantity, fields.external_force,
fields.cumulative_mass_source, fields.contact_signed_distance and
scalars.contact_length_scale.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts.state import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        c = cfg["centrifuge"]
        frame = scene.initial_state()

        self.seed = int(seed)
        self.R = float(c["radius"])                 # drum inner radius
        self.Lh = float(c["length"]) * 0.5          # drum half length (axis = x)
        self.cz = float(c["center_height"])         # axis height
        self.omega_target = float(c["angular_velocity_rad_s"])
        self.h = float(c["spacing"])
        g = cfg.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.asarray(g, dtype=float)
        self.K = int(c["species_count"])

        self.pos = np.ascontiguousarray(frame.position, dtype=float)
        self.vel = np.ascontiguousarray(frame.velocity, dtype=float)
        self.mass = np.ascontiguousarray(frame.mass, dtype=float)
        self.rad = np.ascontiguousarray(frame.attributes["particle_radius"], dtype=float)
        self.species = np.ascontiguousarray(frame.attributes["species_id"]).astype(np.int32)
        self.mass_quantity = np.ascontiguousarray(frame.attributes["mass_quantity"], dtype=float)
        self.N = int(self.pos.shape[0])
        self.invm = 1.0 / self.mass
        self.body_id = np.zeros(self.N, dtype=np.int32)

        # ---- contact model parameters (soft-sphere DEM)
        self.kn = 3.0e4            # normal stiffness  [N/m]
        self.kt = 1.5e4            # tangential stiffness
        self.kn_w = 4.0e4          # wall normal stiffness
        self.kt_w = 2.0e4          # wall tangential stiffness
        self.mu = 0.45             # grain-grain friction
        self.mu_w = 0.6            # grain-wall friction
        self.zeta = 0.4            # damping ratio (restitution ~0.25)
        self.dt = 8.0e-4           # fixed integration step
        self.ramp_time = 0.4       # spin-up ramp of the drum
        self.settle_time = 0.5     # extra numerical damping window
        self.settle_tau = 0.15
        self.vmax = 60.0

        # ---- uniform background grid for neighbour search
        rmax = float(self.rad.max())
        self.rmax = rmax
        self.cutoff = 2.0 * rmax
        self.cell = self.cutoff
        self.grid_min = np.array([-self.Lh - 0.25, -self.R - 0.25, self.cz - self.R - 0.25])
        self.grid_max = np.array([self.Lh + 0.25, self.R + 0.25, self.cz + self.R + 0.25])
        gd = np.ceil((self.grid_max - self.grid_min) / self.cell).astype(np.int64)
        gd = np.maximum(gd, 1)
        self.gdim = gd
        half = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    lead = dx if dx != 0 else (dy if dy != 0 else dz)
                    if lead > 0:
                        half.append((dx, dy, dz))
        deltas = [(0, 0, 0)] + half
        self.delta_ids = np.array(
            [d[0] + d[1] * gd[0] + d[2] * gd[0] * gd[1] for d in deltas], dtype=np.int64
        )

        # ---- persistent contact history (tangential springs)
        self.prev_keys = np.zeros(0, dtype=np.int64)
        self.prev_td = np.zeros((0, 3), dtype=float)
        self.wall_td = np.zeros((self.N, 3), dtype=float)

        self.t = 0.0
        self.steps = 0
        self.max_overlap = 0.0
        self.contact_count = 0
        self._cached_frame = None

    # --------------------------------------------------------- neighbour list
    def _candidate_pairs(self):
        pos = self.pos
        gd = self.gdim
        ci = np.floor((pos - self.grid_min) / self.cell).astype(np.int64)
        ci[:, 0] = np.clip(ci[:, 0], 0, gd[0] - 1)
        ci[:, 1] = np.clip(ci[:, 1], 0, gd[1] - 1)
        ci[:, 2] = np.clip(ci[:, 2], 0, gd[2] - 1)
        idx = ci[:, 0] + ci[:, 1] * gd[0] + ci[:, 2] * gd[0] * gd[1]

        order = np.argsort(idx, kind="stable")
        sidx = idx[order]
        N = self.N
        ar = np.arange(N)
        i_list = []
        j_list = []
        for k, did in enumerate(self.delta_ids):
            q = idx + did
            lo = np.searchsorted(sidx, q, side="left")
            hi = np.searchsorted(sidx, q, side="right")
            cnt = hi - lo
            tot = int(cnt.sum())
            if tot <= 0:
                continue
            rep = np.repeat(ar, cnt)
            starts = np.cumsum(cnt) - cnt
            offs = np.arange(tot) - starts[rep]
            j = order[lo[rep] + offs]
            i = rep
            if k == 0:
                m = j > i
                i = i[m]
                j = j[m]
            if len(i):
                i_list.append(i)
                j_list.append(j)
        if not i_list:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        return np.concatenate(i_list), np.concatenate(j_list)

    # ------------------------------------------------------------------ step
    def _step(self, dt: float) -> None:
        N = self.N
        pos = self.pos
        vel = self.vel
        rad = self.rad
        mass = self.mass

        om = self.omega_target * min(1.0, max(0.0, (self.t + dt) / self.ramp_time))

        acc = np.empty((N, 3), dtype=float)
        acc[:, 0] = self.gravity[0]
        acc[:, 1] = self.gravity[1]
        acc[:, 2] = self.gravity[2]

        # ---------------- grain-grain contacts
        pi, pj = self._candidate_pairs()
        overlap_max = 0.0
        if len(pi):
            d = pos[pj] - pos[pi]
            dist = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)
            rsum = rad[pi] + rad[pj]
            ov = rsum - dist
            m = ov > 0.0
            if m.any():
                pi = pi[m]
                pj = pj[m]
                dist = np.maximum(dist[m], 1e-12)
                ov = ov[m]
                d = d[m]
                overlap_max = float(ov.max())
                nvec = d / dist[:, None]

                vrel = vel[pi] - vel[pj]
                vn = np.einsum("ij,ij->i", vrel, nvec)
                vt = vrel - vn[:, None] * nvec

                mi = mass[pi]
                mj = mass[pj]
                mred = mi * mj / (mi + mj)
                gn = 2.0 * self.zeta * np.sqrt(self.kn * mred)
                gt = 2.0 * self.zeta * np.sqrt(self.kt * mred)

                Fn = self.kn * ov + gn * vn
                np.maximum(Fn, 0.0, out=Fn)

                # tangential spring with pair memory
                keys = pi.astype(np.int64) * N + pj
                if self.prev_keys.size:
                    pk = self.prev_keys
                    loc = np.searchsorted(pk, keys)
                    loc_c = np.minimum(loc, pk.size - 1)
                    hit = pk[loc_c] == keys
                    td = np.where(hit[:, None], self.prev_td[loc_c], 0.0)
                else:
                    td = np.zeros((len(pi), 3))
                td = td + vt * dt

                Ft = -(self.kt * td + gt[:, None] * vt)
                Ft_norm = np.sqrt(np.einsum("ij,ij->i", Ft, Ft))
                cap = self.mu * Fn
                slip = Ft_norm > cap
                if slip.any():
                    scale = np.ones_like(Ft_norm)
                    scale[slip] = cap[slip] / np.maximum(Ft_norm[slip], 1e-12)
                    Ft = Ft * scale[:, None]
                    # prune stored history so it is consistent with sliding
                    td = (-Ft - gt[:, None] * vt) / self.kt

                Ftot = -Fn[:, None] * nvec + Ft
                for a in range(3):
                    acc[:, a] += np.bincount(pi, weights=Ftot[:, a] * self.invm[pi], minlength=N)
                    acc[:, a] -= np.bincount(pj, weights=Ftot[:, a] * self.invm[pj], minlength=N)

                srt = np.argsort(keys, kind="stable")
                self.prev_keys = np.ascontiguousarray(keys[srt])
                self.prev_td = np.ascontiguousarray(td[srt])
                self.contact_count = int(len(pi))
            else:
                self.prev_keys = np.zeros(0, dtype=np.int64)
                self.prev_td = np.zeros((0, 3))
                self.contact_count = 0
        else:
            self.prev_keys = np.zeros(0, dtype=np.int64)
            self.prev_td = np.zeros((0, 3))
            self.contact_count = 0

        # ---------------- cylindrical wall (rotating mantle)
        y = pos[:, 1]
        z = pos[:, 2] - self.cz
        s = np.sqrt(y * y + z * z)
        s_safe = np.maximum(s, 1e-9)
        ovw = s + rad - self.R
        cw = ovw > 0.0
        # wall surface velocity: omega x (p - c), omega = (om, 0, 0)
        vw_y = -om * z
        vw_z = om * y

        if cw.any():
            iw = np.flatnonzero(cw)
            n_out = np.zeros((N, 3))
            n_out[iw, 1] = y[iw] / s_safe[iw]
            n_out[iw, 2] = z[iw] / s_safe[iw]
            vrel = np.zeros((N, 3))
            vrel[:, 0] = vel[:, 0]
            vrel[:, 1] = vel[:, 1] - vw_y
            vrel[:, 2] = vel[:, 2] - vw_z
            vn = np.einsum("ij,ij->i", vrel, n_out)
            Fn = self.kn_w * ovw + 2.0 * self.zeta * np.sqrt(self.kn_w * mass) * vn
            Fn = np.maximum(Fn, 0.0)
            Fn = np.where(cw, Fn, 0.0)
            vt = vrel - vn[:, None] * n_out
            vt = np.where(cw[:, None], vt, 0.0)
            self.wall_td = np.where(cw[:, None], self.wall_td + vt * dt, 0.0)
            gt_w = 2.0 * self.zeta * np.sqrt(self.kt_w * mass)
            Ft = -(self.kt_w * self.wall_td + gt_w[:, None] * vt)
            Ft = np.where(cw[:, None], Ft, 0.0)
            Ft_norm = np.sqrt(np.einsum("ij,ij->i", Ft, Ft))
            cap = self.mu_w * Fn
            over = Ft_norm > cap
            scale = np.ones_like(Ft_norm)
            scale[over] = cap[over] / np.maximum(Ft_norm[over], 1e-12)
            Ft = Ft * scale[:, None]
            Fw = -Fn[:, None] * n_out + Ft
            acc += Fw * self.invm[:, None]
            overlap_max = max(overlap_max, float(ovw[cw].max()))

        # ---------------- flat end caps (also rotating with the drum)
        # +x cap
        ov1 = pos[:, 0] + rad - self.Lh
        c1 = ov1 > 0.0
        # -x cap
        ov2 = rad - pos[:, 0] - self.Lh
        c2 = ov2 > 0.0
        for mask, ov, sgn in ((c1, ov1, 1.0), (c2, ov2, -1.0)):
            if not mask.any():
                continue
            nx = np.zeros(N)
            nx[mask] = sgn  # outward normal (+x or -x)
            vrel = np.zeros((N, 3))
            vrel[:, 0] = vel[:, 0]
            vrel[:, 1] = vel[:, 1] - vw_y
            vrel[:, 2] = vel[:, 2] - vw_z
            vn = vrel[:, 0] * nx
            Fn = np.maximum(self.kn_w * ov + 2.0 * self.zeta * np.sqrt(self.kn_w * mass) * vn, 0.0)
            Fn = np.where(mask, Fn, 0.0)
            vt = vrel - vn[:, None] * nx[:, None]
            vt_norm = np.sqrt(np.einsum("ij,ij->i", vt, vt))
            gt_v = 2.0 * self.zeta * np.sqrt(self.kt_w * mass)
            fmag = np.minimum(gt_v * vt_norm, self.mu_w * Fn)
            with np.errstate(invalid="ignore", divide="ignore"):
                unit = np.where(vt_norm[:, None] > 1e-12, vt / np.maximum(vt_norm[:, None], 1e-12), 0.0)
            Ft = -fmag[:, None] * unit
            Fn_vec = -Fn[:, None] * nx[:, None]
            acc += (Fn_vec + Ft) * self.invm[:, None]
            overlap_max = max(overlap_max, float(ov[mask].max()))

        # ---------------- integrate (semi-implicit Euler)
        vel += acc * dt
        if self.t < self.settle_time:
            vel *= np.exp(-dt / self.settle_tau)
        sp = np.sqrt(np.einsum("ij,ij->i", vel, vel))
        bad = sp > self.vmax
        if bad.any():
            vel[bad] *= (self.vmax / sp[bad])[:, None]
        pos += vel * dt

        # ---------------- hard containment guard (never allows escape)
        y = pos[:, 1]
        z = pos[:, 2] - self.cz
        s = np.sqrt(y * y + z * z)
        pen = s + rad - self.R
        tol = 0.02
        fix = pen > tol
        if fix.any():
            s_safe = np.maximum(s, 1e-9)
            ny = y[fix] / s_safe[fix]
            nz = z[fix] / s_safe[fix]
            back = pen[fix] - 0.5 * tol
            pos[fix, 1] -= ny * back
            pos[fix, 2] -= nz * back
            vn = vel[fix, 1] * ny + vel[fix, 2] * nz
            out = vn > 0.0
            if out.any():
                idx = np.flatnonzero(fix)[out]
                vnn = vn[out]
                nyy = ny[out]
                nzz = nz[out]
                vel[idx, 1] -= vnn * nyy
                vel[idx, 2] -= vnn * nzz
        penx1 = pos[:, 0] + rad - self.Lh
        fx1 = penx1 > tol
        if fx1.any():
            pos[fx1, 0] = self.Lh - rad[fx1] + 0.5 * tol
            vel[fx1, 0] = np.minimum(vel[fx1, 0], 0.0)
        penx2 = rad - pos[:, 0] - self.Lh
        fx2 = penx2 > tol
        if fx2.any():
            pos[fx2, 0] = -self.Lh + rad[fx2] - 0.5 * tol
            vel[fx2, 0] = np.maximum(vel[fx2, 0], 0.0)

        self.max_overlap = overlap_max
        self.t += dt
        self.steps += 1

    # ----------------------------------------------------------------- frame
    def _contact_signed_distances(self):
        """Signed gaps (negative = penetration) for active/near contacts."""
        parts = []
        pi, pj = self._candidate_pairs()
        if len(pi):
            d = self.pos[pj] - self.pos[pi]
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            gap = dist - (self.rad[pi] + self.rad[pj])
            near = gap < 0.25 * self.cutoff
            if near.any():
                parts.append(gap[near])
        y = self.pos[:, 1]
        z = self.pos[:, 2] - self.cz
        s = np.sqrt(y * y + z * z)
        gw = self.R - self.rad - s          # negative -> penetration
        lim = 0.25 * self.cutoff
        sel = np.abs(gw) < lim
        if sel.any():
            parts.append(gw[sel])
        gx1 = self.Lh - self.rad - self.pos[:, 0]
        sel = np.abs(gx1) < lim
        if sel.any():
            parts.append(gx1[sel])
        gx2 = self.Lh - self.rad + self.pos[:, 0]
        sel = np.abs(gx2) < lim
        if sel.any():
            parts.append(gx2[sel])
        if not parts:
            return np.zeros(1, dtype=float)
        return np.concatenate(parts).astype(float)

    def _make_frame(self, time: float) -> StateFrame:
        om = self.omega_target * min(1.0, max(0.0, time / self.ramp_time))
        ext = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "species_id": self.species.copy(),
                "particle_radius": self.rad.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "external_force": np.ascontiguousarray(ext),
                "cumulative_mass_source": np.zeros(self.K, dtype=float),
                "contact_signed_distance": self._contact_signed_distances(),
            },
            scalars={
                "contact_length_scale": float(self.h),
                "grid_spacing": float(self.h),
                "characteristic_length": float(2.0 * self.R),
                "angular_velocity_rad_s": float(om),
                "kinetic_energy": float(0.5 * np.sum(self.mass * np.einsum("ij,ij->i", self.vel, self.vel))),
                "max_overlap": float(self.max_overlap),
                "contact_count": float(self.contact_count),
            },
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- public
    def advance_to(self, time: float):
        time = float(time)
        guard = 0.0
        while self.t < time - 1e-12:
            dt = self.dt
            if self.t + dt > time:
                dt = time - self.t
            self._step(dt)
            guard += dt
            if guard > 3600.0:
                break
        return self._make_frame(time)

    def diagnostics(self) -> dict:
        sp = np.sqrt(np.einsum("ij,ij->i", self.vel, self.vel))
        return {
            "method": "soft_sphere_DEM_rotating_drum",
            "particles": int(self.N),
            "steps": int(self.steps),
            "dt": float(self.dt),
            "time": float(self.t),
            "max_overlap": float(self.max_overlap),
            "contacts_last_step": int(self.contact_count),
            "mean_speed": float(sp.mean()) if self.N else 0.0,
            "max_speed": float(sp.max()) if self.N else 0.0,
        }
