"""Two-way coupled poro-elasto-capillary sponge wetting and wringing solver.

Pure NumPy implementation: a mass-spring deformable sponge mesh with per-vertex
pore-liquid storage, a particle liquid phase driven by gravity and coupled to the
solid by interface drag, sink floor/wall boundary conditions, and kinematic end
grips that lift then counter-twist the sponge.  Fluid mass transfer into and out
of the pore space is resolved from the current mechanical strain and saturation
state, and the evolving pore content changes the solid's mass, its cross-section
rest lengths (desaturation contraction) and the resolved liquid motion.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_G = np.array([0.0, 0.0, -9.81])
_FLOOR_Z = 0.015
_BASIN_X = 0.384
_BASIN_Y = 0.264
_NSEC, _PER = 9, 4
_NSP = _NSEC * _PER
_DT = 0.001
_WMASS = 0.00095
_EMIT_DT = 0.02
_EMIT_COUNT = 3
_EMIT_STOP = 6.8
_MAXW = 2400
_RHO = 1000.0
_PORO = 0.72
_VV = (0.36 * 0.16 * 0.08) / _NSP
_CAP = _RHO * _PORO * _VV * 0.4
_K = 60.0
_C = 0.9
_KDRAG = 0.05
_RDRAG = 0.04
_KEXP = 0.15
_ACTIVE = 6.5
_LIFT_END = 8.0
_TWIST_END = 11.0
_LIFT = 0.22
_TWIST = np.deg2rad(150.0)
_GAMMA = 0.55
_DOM_SPONGE, _DOM_WATER, _DOM_GLOBAL = 0, 1, 2
_K_TERMS = 6


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, float(u)))
    return u * u * (3.0 - 2.0 * u)


def _rx(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.config = dict(getattr(scene, "config", {}))
        self.seed = int(seed)
        init = scene.initial_state()
        pos = np.asarray(init.position, dtype=float)
        vel = np.asarray(init.velocity, dtype=float)
        mass = np.asarray(init.mass, dtype=float)
        bid = np.asarray(init.body_id).astype(np.int64)
        attrs = init.attributes

        sp_mask = bid == 1
        self.sp = pos[sp_mask].copy()
        self.spv = vel[sp_mask].copy()
        n_sp = self.sp.shape[0]
        self.sp_ms = np.full(n_sp, 0.0124 / n_sp)
        self.sp_m_pore = np.zeros(n_sp)
        self.sp_cap = np.full(n_sp, _CAP)
        self.sp_sat = np.zeros(n_sp)
        self.sp_smax = np.zeros(_NSEC)
        self._expel_accum = np.zeros(n_sp)
        self._emit_timer = 0.0
        self._gas_prev = None

        w_mask = bid == 2
        self.wp = pos[w_mask].copy()
        self.wv = vel[w_mask].copy()
        self.wm = mass[w_mask].copy()

        g_mask = (bid == 10) | (bid == 11)
        self.gp = pos[g_mask].copy()
        self.gv = np.zeros_like(self.gp)
        self.gm = mass[g_mask].copy()

        self._c0L = np.array([-0.205, 0.0, 0.065])
        self._c0R = np.array([0.205, 0.0, 0.065])
        sec = np.arange(n_sp) // _PER
        self.sec = sec
        self.grip_L = np.where(self.sp[:, 0] <= -0.125)[0]
        self.grip_R = np.where(self.sp[:, 0] >= 0.125)[0]
        self._engaged = False
        self._anchor_L = None
        self._anchor_R = None
        self._target_prev = self.sp.copy()

        # spring network (perimeter + section diagonals + axial edges)
        e0, e1, esec = [], [], []
        for i in range(_NSEC):
            b = i * _PER
            for k in range(_PER):
                e0.append(b + k)
                e1.append(b + (k + 1) % _PER)
                esec.append(i)
            e0.append(b + 0)
            e1.append(b + 2)
            esec.append(i)
            e0.append(b + 1)
            e1.append(b + 3)
            esec.append(i)
        for i in range(_NSEC - 1):
            for k in range(_PER):
                e0.append(i * _PER + k)
                e1.append((i + 1) * _PER + k)
                esec.append(-1)
        self.e0 = np.asarray(e0, dtype=np.int64)
        self.e1 = np.asarray(e1, dtype=np.int64)
        self.esec = np.asarray(esec, dtype=np.int64)
        self.rest = np.linalg.norm(self.sp[self.e1] - self.sp[self.e0], axis=1)
        self.rest = np.maximum(self.rest, 1e-4)

        n = self.sp.shape[0] + self.wp.shape[0] + self.gp.shape[0]
        self.impulse = np.zeros((3, 3))
        self.inflow = np.zeros((3, 3))
        self.outflow = np.zeros((3, 3))
        self.mass_source = np.zeros(2)
        self.interaction = np.zeros((2, 3))
        self._last_Fext = np.zeros((3, _K_TERMS, 3))
        self._sp_contact = np.zeros(3)
        self._w_contact = np.zeros(3)
        self._grip_force = np.zeros(3)
        self._faucet_force = np.zeros(3)
        self._drag_on_water = np.zeros_like(self.wp)
        self._drag_on_sponge = np.zeros(3)
        self.c_comp = np.zeros(n_sp)
        n0 = float(np.sum(mass))
        self.reference_momentum = float(max(1e-3, n0 + 1.0 * _WMASS * 1020.0))
        self.t = 0.0
        self.steps = 0
        self.max_sat = 0.0

    # --------------------------------------------------------------- controls
    def _grip_targets(self, t: float):
        lift = _LIFT * _smoothstep((t - _ACTIVE) / (_LIFT_END - _ACTIVE))
        ang = _TWIST * _smoothstep((t - _LIFT_END) / (_TWIST_END - _LIFT_END))
        tgt = self.sp.copy()
        if t >= _ACTIVE:
            if not self._engaged:
                self._engaged = True
                self._anchor_L = self.sp[self.grip_L] - self._c0L
                self._anchor_R = self.sp[self.grip_R] - self._c0R
                self._target_prev = tgt.copy()
            cL = self._c0L + np.array([0.0, 0.0, lift])
            cR = self._c0R + np.array([0.0, 0.0, lift])
            RL, RR = _rx(ang), _rx(-ang)
            tgt[self.grip_L] = cL + (RL @ self._anchor_L.T).T
            tgt[self.grip_R] = cR + (RR @ self._anchor_R.T).T
            gp = np.stack((cL, cR))
        else:
            gp = self.gp.copy()
        return tgt, gp

    # ------------------------------------------------------------------ step
    def _step(self, dt: float) -> None:
        t, tn = self.t, self.t + dt
        tgt, gp_t = self._grip_targets(tn)
        tgt_prev = self._target_prev
        self._target_prev = tgt.copy()

        p, v = self.sp, self.spv
        mtot = self.sp_ms + self.sp_m_pore
        F = mtot[:, None] * _G

        # desaturation-driven section contraction (state based)
        f_i = np.ones(_NSEC)
        for i in range(_NSEC):
            s = float(np.mean(self.sp_sat[i * _PER:(i + 1) * _PER]))
            self.sp_smax[i] = max(self.sp_smax[i], s)
            d = max(0.0, self.sp_smax[i] - s)
            f_i[i] = min(1.0, max(0.5, 1.0 - _GAMMA * d))
        escale = np.where(self.esec >= 0, f_i[np.clip(self.esec, 0, _NSEC - 1)], 1.0)

        # elastic springs with damping
        dvec = p[self.e1] - p[self.e0]
        L = np.linalg.norm(dvec, axis=1)
        L = np.maximum(L, 1e-6)
        dirv = dvec / L[:, None]
        L0 = self.rest * escale
        rel = ((v[self.e1] - v[self.e0]) * dirv).sum(axis=1)
        fmag = _K * (L - L0) + _C * rel
        fe = fmag[:, None] * dirv
        np.add.at(F, self.e0, fe)
        np.add.at(F, self.e1, -fe)
        strain = (L - L0) / L0
        comp = np.maximum(0.0, -strain)
        self.c_comp[:] = 0.0
        np.maximum.at(self.c_comp, self.e0, comp)
        np.maximum.at(self.c_comp, self.e1, comp)

        # two-way drag coupling with nearby liquid particles
        self._drag_on_sponge[:] = 0.0
        self._drag_on_water = np.zeros_like(self.wp)
        if self.wp.shape[0] > 0:
            lo = self.sp.min(axis=0) - _RDRAG - 0.05
            hi = self.sp.max(axis=0) + _RDRAG + 0.05
            near = np.all((self.wp > lo) & (self.wp < hi), axis=1)
            if np.any(near):
                pw = self.wp[near]
                vw = self.wv[near]
                diff = self.sp[:, None, :] - pw[None, :, :]
                d2 = np.einsum("ijk,ijk->ij", diff, diff)
                mask = d2 < _RDRAG * _RDRAG
                dist = np.sqrt(d2 + 1e-12)
                ker = np.where(mask, 1.0 - dist / _RDRAG, 0.0)
                u = vw[None, :, :] - v[None, :, :]
                fd = _KDRAG * ker[:, :, None] * u
                fs = fd.sum(axis=1)
                F += fs
                fw = -fd.sum(axis=0)
                full = np.zeros_like(self.wp)
                full[near] = fw
                self._drag_on_water = full
                self._drag_on_sponge = fs.sum(axis=0)

        # integrate solid
        acc = F / mtot[:, None]
        v[:] = v + acc * dt
        v *= (1.0 - 0.5 * dt)

        # floor contact for sponge (no penetration)
        self._sp_contact[:] = 0.0
        below = p[:, 2] < _FLOOR_Z
        if np.any(below):
            p[below, 2] = _FLOOR_Z
            vz = v[below, 2]
            hit = vz < 0.0
            idx = np.where(below)[0][hit]
            if len(idx):
                self._sp_contact[2] += float(np.sum(-mtot[idx] * v[idx, 2] / dt))
                v[idx, 2] = 0.0
            v[below, 0] *= 0.7
            v[below, 1] *= 0.7

        # kinematic grips: continuous projection onto prescribed pose
        old_p = p.copy()
        old_v = v.copy()
        if tn >= _ACTIVE:
            gidx = np.concatenate((self.grip_L, self.grip_R))
            newv = (tgt[gidx] - old_p[gidx]) / dt
            self._grip_force = (mtot[gidx, None] * (newv - old_v[gidx]) / dt).sum(axis=0)
            p[gidx] = tgt[gidx]
            v[gidx] = newv
        else:
            self._grip_force = np.zeros(3)

        p += v * dt
        if tn >= _ACTIVE:
            gidx = np.concatenate((self.grip_L, self.grip_R))
            p[gidx] = tgt[gidx]
            v[gidx] = (tgt[gidx] - tgt_prev[gidx]) / dt
        # keep contact guarantee after position update
        np.maximum(p[:, 2], _FLOOR_Z, out=p[:, 2])

        # grip bodies follow prescribed pose continuously
        self.gv = (gp_t - self.gp) / dt
        self.gp = gp_t

        # ------------------------------------------------------------ liquid
        M = self.wp.shape[0]
        wv = self.wv
        self._w_contact[:] = 0.0
        if M > 0:
            wv[:, 2] += _G[2] * dt
            wv *= (1.0 - 0.1 * dt)
            wm = self.wm
            wv += self._drag_on_water / np.maximum(wm, 1e-9)[:, None] * dt
            wp = self.wp + wv * dt
            low = wp[:, 2] < _FLOOR_Z
            if np.any(low):
                wp[low, 2] = _FLOOR_Z
                self._w_contact[2] += float(np.sum(-wm[low] * np.minimum(wv[low, 2], 0.0) / dt))
                wv[low, 2] = 0.0
                wv[low, 0] *= 0.85
                wv[low, 1] *= 0.85
            for axis, lim in ((0, _BASIN_X), (1, _BASIN_Y)):
                over = wp[:, axis] > lim
                under = wp[:, axis] < -lim
                if np.any(over):
                    wp[over, axis] = lim
                    self._w_contact[axis] += float(np.sum(wm[over] * np.maximum(wv[over, axis], 0.0) / dt))
                    wv[over, axis] = np.minimum(wv[over, axis], 0.0)
                if np.any(under):
                    wp[under, axis] = -lim
                    self._w_contact[axis] += float(np.sum(-wm[under] * np.minimum(wv[under, axis], 0.0) / dt))
                    wv[under, axis] = np.maximum(wv[under, axis], 0.0)
            self.wp = wp
            self.wv = wv

        # absorption into pore space (mass/momentum transfer solid<-liquid)
        self._absorb(dt, tn)
        # desaturation squeeze-out (mass/momentum transfer liquid<-solid)
        self._expel(dt, tn)
        # faucet emission
        self._emit(dt, tn)
        # drain / count management
        self._drain(dt, tn)

        # saturations from stored pore mass
        self.sp_sat = np.clip(self.sp_m_pore / self.sp_cap, 0.0, 1.0)
        self.max_sat = max(self.max_sat, float(self.sp_sat.max()) if self.sp_sat.size else 0.0)

        # ------------------------------------------------------- bookkeeping
        Fext = np.zeros((3, _K_TERMS, 3))
        mg = self.gm.sum()
        Fext[_DOM_SPONGE, 0] = mtot.sum(axis=0) * _G
        Fext[_DOM_WATER, 0] = self.wm.sum() * _G
        Fext[_DOM_GLOBAL, 0] = (mtot.sum() + self.wm.sum() + mg) * _G
        Fext[_DOM_SPONGE, 1] = self._sp_contact
        Fext[_DOM_GLOBAL, 1] = self._sp_contact
        Fext[_DOM_SPONGE, 2] = self._grip_force
        Fext[_DOM_GLOBAL, 2] = self._grip_force
        Fext[_DOM_WATER, 3] = self._w_contact
        Fext[_DOM_GLOBAL, 3] = self._w_contact
        Fext[_DOM_WATER, 4] = self._faucet_force
        Fext[_DOM_GLOBAL, 4] = self._faucet_force
        Fext[_DOM_SPONGE, 5] = self._drag_on_sponge
        Fext[_DOM_WATER, 5] = -self._drag_on_sponge
        self._last_Fext = Fext
        self.impulse += Fext.sum(axis=1) * dt
        self.interaction[0] = self._drag_on_sponge
        self.interaction[1] = -self._drag_on_sponge

        self.t = tn
        self.steps += 1

    # ------------------------------------------------------------ transfers
    def _absorb(self, dt: float, tn: float) -> None:
        M = self.wp.shape[0]
        if M == 0:
            return
        lo = self.sp.min(axis=0) - 0.012
        hi = self.sp.max(axis=0) + 0.012
        inside = np.all((self.wp > lo) & (self.wp < hi), axis=1)
        idx = np.where(inside)[0]
        if idx.size == 0:
            return
        pts = self.wp[idx]
        vs = self.wv[idx]
        ms = self.wm[idx]
        d = np.linalg.norm(pts[:, None, :] - self.sp[None, :, :], axis=2)
        k = min(4, self.sp.shape[0])
        order = np.argsort(d, axis=1)[:, :k]
        w = 1.0 / (np.take_along_axis(d, order, axis=1) + 1e-4)
        w = w / w.sum(axis=1, keepdims=True)
        contrib = ms[:, None] * w
        np.add.at(self.sp_m_pore, order.ravel(), contrib.ravel())
        self.inflow[_DOM_SPONGE] += np.sum(ms[:, None] * vs, axis=0)
        self.outflow[_DOM_WATER] += np.sum(ms[:, None] * vs, axis=0)
        keep = np.ones(M, dtype=bool)
        keep[idx] = False
        self.wp = self.wp[keep]
        self.wv = self.wv[keep]
        self.wm = self.wm[keep]

    def _expel(self, dt: float, tn: float) -> None:
        if tn < _ACTIVE:
            return
        cent = self.sp.reshape(_NSEC, _PER, 3).mean(axis=1)
        ang = np.arctan2(cent[:, 2] - 0.065, cent[:, 1])
        dth = np.diff(ang)
        dth = (dth + np.pi) % (2.0 * np.pi) - np.pi
        dx = np.maximum(np.abs(np.diff(cent[:, 0])), 1e-6)
        shear_int = np.abs(dth) / dx
        shear_sec = np.empty(_NSEC)
        shear_sec[0] = shear_int[0]
        shear_sec[-1] = shear_int[-1]
        if _NSEC > 2:
            shear_sec[1:-1] = np.maximum(shear_int[:-1], shear_int[1:])
        r = np.sqrt((self.sp[:, 1] - cent[self.sec, 1]) ** 2 + (self.sp[:, 2] - cent[self.sec, 2]) ** 2)
        c_twist = shear_sec[self.sec] * r
        drive = c_twist + self.c_comp
        rate = _KEXP * self.sp_sat * np.where(drive > 0.08, drive, 0.0) * self.sp_cap
        rate = np.minimum(rate, self.sp_m_pore / max(dt, 1e-9) * 0.5)
        self._expel_accum += rate * dt
        emit = self._expel_accum >= _WMASS
        if not np.any(emit):
            return
        idx = np.where(emit)[0]
        self._expel_accum[idx] -= _WMASS
        self.sp_m_pore[idx] -= _WMASS
        cy = cent[self.sec[idx], 1]
        cz = cent[self.sec[idx], 2]
        out = np.stack((np.zeros(idx.size), self.sp[idx, 1] - cy, self.sp[idx, 2] - cz), axis=1)
        nrm = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.where(nrm > 1e-8, out / np.maximum(nrm, 1e-8), np.array([0.0, 0.0, -1.0]))
        newp = self.sp[idx] + out * 0.012
        newv = self.spv[idx] + out * 0.5 + np.array([0.0, 0.0, -0.25])
        self.wp = np.vstack((self.wp, newp))
        self.wv = np.vstack((self.wv, newv))
        self.wm = np.concatenate((self.wm, np.full(idx.size, _WMASS)))
        mom = np.sum(_WMASS * newv, axis=0)
        self.outflow[_DOM_SPONGE] += mom
        self.inflow[_DOM_WATER] += mom

    def _emit(self, dt: float, tn: float) -> None:
        self._emit_timer += dt
        spawned_p = []
        spawned_v = []
        while self._emit_timer >= _EMIT_DT and tn < _EMIT_STOP:
            self._emit_timer -= _EMIT_DT
            offs = np.array([[0.0, 0.0], [0.008, 0.0], [-0.008, 0.0]])
            for ox, oy in offs:
                spawned_p.append([-0.07 + ox, 0.04 + oy, 0.552])
                spawned_v.append([0.0, 0.0, -1.0])
        if not spawned_p:
            return
        np_ = np.asarray(spawned_p)
        nv = np.asarray(spawned_v)
        nm = np.full(len(np_), _WMASS)
        self.wp = np.vstack((self.wp, np_))
        self.wv = np.vstack((self.wv, nv))
        self.wm = np.concatenate((self.wm, nm))
        mom = np.sum(nm[:, None] * nv, axis=0)
        self.inflow[_DOM_WATER] += mom
        self.inflow[_DOM_GLOBAL] += mom
        self.mass_source[1] += float(nm.sum())
        self._faucet_force = mom / max(dt, 1e-9)

    def _drain(self, dt: float, tn: float) -> None:
        M = self.wp.shape[0]
        if M == 0:
            return
        # open drain: only matter that escapes the basin or reaches the drain
        # hole leaves the domain; water pooling on the sink floor stays resolved.
        drained = (
            (np.abs(self.wp[:, 0]) >= _BASIN_X - 1e-6)
            | (np.abs(self.wp[:, 1]) >= _BASIN_Y - 1e-6)
            | (self.wp[:, 2] <= _FLOOR_Z - 0.004)
        )
        if M > _MAXW:
            extra = M - _MAXW
            order = np.argsort(self.wp[:, 2])
            drained[order[:extra]] = True
        if not np.any(drained):
            return
        idx = np.where(drained)[0]
        mom = np.sum(self.wm[idx, None] * self.wv[idx], axis=0)
        self.outflow[_DOM_WATER] += mom
        self.outflow[_DOM_GLOBAL] += mom
        self.mass_source[1] -= float(self.wm[idx].sum())
        keep = np.ones(M, dtype=bool)
        keep[idx] = False
        self.wp = self.wp[keep]
        self.wv = self.wv[keep]
        self.wm = self.wm[keep]

    # ------------------------------------------------------------------ io
    def advance_to(self, time: float):
        target = float(time)
        while self.t < target - 1e-12:
            dt = min(_DT, target - self.t)
            if dt <= 1e-12:
                break
            self._step(dt)
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        n_sp = self.sp.shape[0]
        n_w = self.wp.shape[0]
        pos = np.vstack((self.sp, self.wp, self.gp))
        vel = np.vstack((self.spv, self.wv, self.gv))
        n = pos.shape[0]
        mass = np.concatenate((self.sp_ms + self.sp_m_pore, self.wm, self.gm))
        body = np.concatenate((
            np.full(n_sp, 1, dtype=np.int64),
            np.full(n_w, 2, dtype=np.int64),
            np.array([10, 11], dtype=np.int64),
        ))
        sat = np.concatenate((self.sp_sat, np.ones(n_w), np.zeros(2)))
        poro = np.concatenate((np.full(n_sp, _PORO), np.zeros(n_w), np.zeros(2)))
        phase = np.concatenate((np.zeros(n_sp), np.ones(n_w), np.zeros(2)))
        mq = np.zeros((n, 2))
        mq[:n_sp, 0] = self.sp_ms
        mq[:n_sp, 1] = self.sp_m_pore
        mq[n_sp:n_sp + n_w, 1] = self.wm
        mq[n_sp + n_w:, 0] = self.gm

        w = np.zeros((3, n))
        w[0, :n_sp] = 1.0
        w[1, n_sp:n_sp + n_w] = 1.0
        w[2, :] = 1.0

        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=body,
            attributes={
                "saturation": sat,
                "porosity": poro,
                "phase": phase,
                "mass_quantity": mq,
            },
            fields={
                "budget_domain_weights": w,
                "budget_external_force": self._last_Fext.copy(),
                "budget_cumulative_external_impulse": self.impulse.copy(),
                "budget_cumulative_inflow_momentum": self.inflow.copy(),
                "budget_cumulative_outflow_momentum": self.outflow.copy(),
                "cumulative_mass_source": self.mass_source.copy(),
                "interaction_force": self.interaction.copy(),
            },
            scalars={"reference_momentum": self.reference_momentum},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": int(self.steps),
            "max_saturation": float(self.max_sat),
            "water_particles": int(self.wp.shape[0]),
            "stored_pore_mass": float(self.sp_m_pore.sum()),
            "mass_source_water": float(self.mass_source[1]),
        }
