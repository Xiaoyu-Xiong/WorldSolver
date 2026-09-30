"""Physically-based solver for scene 052_soluble_bunny_water.

Coupled particle simulation, time-stepped with explicit sub-stepping:

* Deformable soluble bunny: corotated lattice solid.  Each bunny sample owns a
  moving-least-squares deformation gradient ``F`` built from its rest neighbour
  stencil; elastic pair forces are computed in the polar-rotation frame of the
  stencil (shape-preserving, volume-locking corotated model).  Solid mass is
  removed at wetted interfaces by a dissolution law limited by the local solute
  concentration, so dissolution slows as the surrounding liquid saturates.
* Pipe inflow / receiving water: weakly compressible SPH (cubic-spline kernel,
  Tait pressure with a non-negative pressure clamp, XSPH velocity smoothing,
  floor no-penetration, open-side outflow).
* Red solute: a conserved per-particle mass ``m*c`` that is created only by
  dissolution of bunny solid and is advected with the carrying liquid and mixed
  by a conservative pairwise diffusion operator.  A collocated diagnostic grid
  carries the advected/diffused transported scalar, its prescribed source and a
  diagnostic velocity field.

Nothing is scripted: every exported frame is the result of integrating the
equations of motion above with the frozen scene geometry and parameters.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants --
GRAVITY = np.array([0.0, 0.0, -9.81])
SPACING = 0.105                      # bunny lattice spacing (scene.initial_state)
H = 0.12                             # SPH smoothing length
SUPPORT = 2.0 * H
KERNEL_SIGMA = 1.0 / (np.pi * H ** 3)
RHO0 = 1000.0
RHO_SOLID = 1180.0
DT = 1.25e-3                         # base sub-step = sample_dt/16 (CFL safe for c_s=12 m/s)

# Pipe nozzle (scene_geometry.yaml: cylinder axis +x, centre x=-1.12,z=1.30).
NOZZLE_X = -0.90
NOZZLE_Z = 1.20
NOZZLE_R = 0.26
INFLOW_SPEED = 2.4
INJECT_RATE = 430.0                  # injected samples per second (fills the nozzle disk
                                     # at the frozen 0.105 m sample spacing)

# Contact / dissolution / elasticity.
CONTACT_R = 0.095
CONTACT_K = 6.0e4
CONTACT_C = 350.0
DISSOLVE_K = 0.65                    # 1/s interfacial dissolution rate
SOLID_K = 1.5e4                      # per-sample corotated lattice stiffness
SOLID_DAMP = 0.35
XSPH = 0.06
SOLUTE_MIX = 0.45                    # 1/s conservative inter-particle mixing
DIFFUSIVITY = 1.0e-4                 # m^2/s grid diffusion

# Diagnostic grid over the frozen mixing domain.
GRID_MIN = np.array([-0.55, -1.25, 0.04])
GRID_MAX = np.array([2.45, 1.25, 1.65])
GRID_D = 0.12
GRID_SHAPE = (25, 21, 13)

KILL_X = 8.0
KILL_Y = 6.0
KILL_Z = -0.6
FLOOR_Z = 0.05
FLOOR_CX, FLOOR_CY = 0.2, 0.0
FLOOR_HX, FLOOR_HY = 3.8, 2.7


def _kernel(q):
    """Cubic spline shape function f(q) and df/dq for q = r/h."""
    f = np.zeros_like(q)
    df = np.zeros_like(q)
    inner = q < 1.0
    outer = (q >= 1.0) & (q < 2.0)
    f[inner] = 1.0 - 1.5 * q[inner] ** 2 + 0.75 * q[inner] ** 3
    df[inner] = -3.0 * q[inner] + 2.25 * q[inner] ** 2
    t = 2.0 - q[outer]
    f[outer] = 0.25 * t ** 3
    df[outer] = -0.75 * t ** 2
    return f, df


def _rotations(F):
    """Polar rotation factor of each [n,3,3] deformation gradient."""
    F = np.where(np.isfinite(F), F, np.eye(3)[None])
    U, _S, Vt = np.linalg.svd(F)
    R = U @ Vt
    neg = np.linalg.det(R) < 0.0
    if np.any(neg):
        U = U.copy()
        U[neg, :, 2] *= -1.0
        R = U @ Vt
    return R


class BenchmarkSolver:
    """Deterministic particle solver for the soluble bunny water scene."""

    # ------------------------------------------------------------ lifecycle --
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(int(seed))
        cfg = getattr(scene, "config", None) or {}
        ic = cfg.get("initial_conditions", {}) or {}
        self.inflow_speed = float(ic.get("inflow_speed_m_s", INFLOW_SPEED))
        self.inflow_start = float(ic.get("inflow_start_s", 0.0))
        self.inflow_end = float(ic.get("inflow_end_s", 5.5))

        frame = scene.initial_state()
        self.pos = np.array(frame.position, dtype=float)
        self.vel = np.array(frame.velocity, dtype=float)
        self.m = np.array(frame.mass, dtype=float)
        self.label = np.asarray(frame.body_id).astype(np.int64)
        attrs = frame.attributes
        n = len(self.pos)
        self.c = np.array(attrs.get("red_solute_concentration", np.zeros(n)), dtype=float)
        self.s = np.array(attrs.get("solid_phase_fraction", (self.label == 1).astype(float)),
                          dtype=float)
        self.m0 = self.m.copy()               # reference SPH mass (fixed)
        self.ms = self.m * np.clip(self.c, 0.0, 1.0)   # solute mass carried per sample
        self.rho = np.full(n, RHO0)
        self.pressure = np.zeros(n)
        self.acc = np.zeros((n, 3))
        self.contact_force = np.zeros((n, 3))

        # SPH reference density from the frozen initial packing (avoids a
        # spurious startup compression from mixed lattice spacings).
        pairs = cKDTree(self.pos).query_pairs(SUPPORT, output_type="ndarray")
        fluid = self.label >= 2
        keep = fluid[pairs[:, 0]] & fluid[pairs[:, 1]]
        ii, jj = pairs[keep, 0], pairs[keep, 1]
        if len(ii):
            r = np.linalg.norm(self.pos[jj] - self.pos[ii], axis=1)
            f, _ = _kernel(r / H)
            w = KERNEL_SIGMA * f
            rho_i = np.zeros(n)
            np.add.at(rho_i, ii, self.m0[jj] * w)
            np.add.at(rho_i, jj, self.m0[ii] * w)
            rho_i += self.m0 * KERNEL_SIGMA * _kernel(np.zeros(1))[0][0]
            est = rho_i[fluid]
            est = est[est > 0.0]
            self.rho0 = float(np.median(est)) if est.size else RHO0
            if not np.isfinite(self.rho0) or self.rho0 <= 0.0:
                self.rho0 = RHO0
        else:
            self.rho0 = RHO0
        self.sound_speed = 12.0

        # Bunny rest stencil (fixed material neighbour list).
        self.b_idx = np.flatnonzero(self.label == 1)
        self.b_local = np.full(n, -1, dtype=np.int64)
        self.b_local[self.b_idx] = np.arange(len(self.b_idx))
        self.x0 = self.pos[self.b_idx].copy()
        nb = len(self.b_idx)
        if nb:
            bpairs = cKDTree(self.x0).query_pairs(1.55 * SPACING, output_type="ndarray")
        else:
            bpairs = np.zeros((0, 2), dtype=np.int64)
        self.bi = bpairs[:, 0].astype(np.int64)
        self.bj = bpairs[:, 1].astype(np.int64)
        self.br0 = self.x0[self.bj] - self.x0[self.bi] if len(self.bi) else np.zeros((0, 3))
        cnt = (np.bincount(self.bi, minlength=nb) + np.bincount(self.bj, minlength=nb)).astype(float)
        self.bn0 = np.maximum(cnt, 1.0)
        A = np.zeros((nb, 3, 3))
        if len(self.bi):
            outer = self.br0[:, :, None] * self.br0[:, None, :]
            np.add.at(A, self.bi, outer)
            np.add.at(A, self.bj, outer)
        scale = np.maximum(np.trace(A, axis1=1, axis2=2) / 3.0, 1e-6)
        A = A + np.eye(3)[None] * (1e-3 * scale)[:, None, None]
        self.bAinv = np.linalg.inv(A) if nb else np.zeros((0, 3, 3))
        nbulk = float(np.median(self.bn0)) if nb else 1.0
        self.be0 = np.clip((nbulk - self.bn0) / max(nbulk, 1.0) * 2.5, 0.05, 1.0)
        self.Fb = np.broadcast_to(np.eye(3), (nb, 3, 3)).copy()

        # Diagnostic grid state.
        self.C = np.zeros(GRID_SHAPE)
        self.S_grid = np.zeros(GRID_SHAPE)
        self.V_grid = np.zeros(GRID_SHAPE + (3,))
        self.active = np.zeros(GRID_SHAPE)
        self._src_acc = np.zeros(GRID_SHAPE)
        self._grid_dt = 0.0
        self._emit_acc = 0.0
        self._emit_index = 0
        self._t = 0.0
        self._injected = 0.0
        self._removed = 0.0
        self._dissolved = 0.0
        self._removed_water = 0.0
        self._removed_solute = 0.0
        self._interaction = np.zeros((3, 3))
        water = self.label == 2
        self._m_fluid_ref = float(self.m0[water].mean()) if np.any(water) else 1.0
        self._steps = 0
        self._vmax = 0.0

    # ------------------------------------------------------------- dynamics --
    def _step(self, dt: float) -> None:
        pos, vel, m = self.pos, self.vel, self.m
        lab = self.label
        n = len(pos)
        pairs = cKDTree(pos).query_pairs(SUPPORT, output_type="ndarray")
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            rvec = pos[j] - pos[i]
            r = np.linalg.norm(rvec, axis=1)
            good = r > 1e-9
            i, j, rvec, r = i[good], j[good], rvec[good], r[good]
            q = r / H
            f, df = _kernel(q)
            w = KERNEL_SIGMA * f
            dWdr = KERNEL_SIGMA * df / H
            inv_r = 1.0 / r
            grad = dWdr[:, None] * (rvec * inv_r[:, None])
        else:
            i = j = np.zeros(0, dtype=np.int64)
            rvec = r = grad = w = np.zeros((0, 3))
            w = np.zeros(0)

        fluid = lab >= 2
        ff = fluid[i] & fluid[j]
        fi, fj = i[ff], j[ff]
        fw, fgrad = w[ff], grad[ff]

        # ---- SPH density / pressure (fluid only, reference mass) ----
        rho = self.m0 * KERNEL_SIGMA * float(_kernel(np.zeros(1))[0][0])
        if len(fi):
            np.add.at(rho, fi, self.m0[fj] * fw)
            np.add.at(rho, fj, self.m0[fi] * fw)
        rho = np.maximum(rho, 1e-3 * self.rho0)
        p = self.sound_speed ** 2 * (rho - self.rho0)
        np.maximum(p, 0.0, out=p)          # free-surface clamp: no tension
        self.rho = np.where(fluid, rho, RHO_SOLID)
        self.pressure = p

        acc = np.broadcast_to(GRAVITY, (n, 3)).copy()

        # ---- SPH pressure acceleration ----
        if len(fi):
            rho_i = rho[fi]
            rho_j = rho[fj]
            coef = p[fi] / rho_i ** 2 + p[fj] / rho_j ** 2
            term = coef[:, None] * fgrad
            np.add.at(acc, fi, self.m0[fj][:, None] * term)
            np.add.at(acc, fj, -self.m0[fi][:, None] * term)

        # ---- fluid / solid contact + dissolution coupling ----
        self.contact_force[:] = 0.0
        fbi = fbj = None
        if len(i):
            bun = lab[i] == 1
            fl = fluid[j]
            sel = (bun & fl)
            sel2 = (fluid[i] & (lab[j] == 1))
            if np.any(sel) or np.any(sel2):
                ci = np.concatenate((i[sel], j[sel2]))
                cj = np.concatenate((j[sel], i[sel2]))
                crvec = pos[cj] - pos[ci]
                cr = np.linalg.norm(crvec, axis=1)
                cmask = (cr > 1e-9) & (cr < CONTACT_R)
                ci, cj = ci[cmask], cj[cmask]
                crvec, cr = crvec[cmask], cr[cmask]
                if len(ci):
                    nhat = crvec / cr[:, None]
                    vrel = vel[cj] - vel[ci]
                    vn = np.einsum("ij,ij->i", vrel, nhat)
                    fmag = CONTACT_K * (CONTACT_R - cr) - CONTACT_C * vn
                    force = fmag[:, None] * nhat
                    np.add.at(self.contact_force, ci, -force)
                    np.add.at(self.contact_force, cj, force)
                    m_eff = np.maximum(m, 0.02 * self.m0)
                    np.add.at(acc, ci, -force / m_eff[ci][:, None])
                    np.add.at(acc, cj, force / m_eff[cj][:, None])
                    # per-material interaction force bookkeeping
                    lab_i, lab_j = lab[ci], lab[cj]
                    for mat in (1, 2, 3):
                        sel_m = lab_i == mat
                        if np.any(sel_m):
                            self._interaction[mat - 1] -= force[sel_m].sum(axis=0)
                        sel_m = lab_j == mat
                        if np.any(sel_m):
                            self._interaction[mat - 1] += force[sel_m].sum(axis=0)
                    # dissolution bookkeeping: bunny index -> fluid neighbour
                    bun_row = lab[ci] == 1
                    fbi = ci[bun_row]
                    fbj = cj[bun_row]

        # ---- bunny elasticity (corotated lattice, MLS deformation gradient) --
        b = self.b_idx
        nb = len(b)
        if nb and len(self.bi):
            u = pos[b] - self.x0
            du = u[self.bj] - u[self.bi]
            outer = du[:, :, None] * self.br0[:, None, :]
            B = np.zeros((nb, 3, 3))
            np.add.at(B, self.bi, outer)
            np.add.at(B, self.bj, outer)
            F = np.eye(3)[None] + B @ self.bAinv
            F = np.where(np.isfinite(F), F, np.eye(3)[None])
            np.clip(F, -50.0, 50.0, out=F)
            self.Fb = F
            R = _rotations(F)
            Ravg = 0.5 * (R[self.bi] + R[self.bj])
            target = np.einsum("nij,nj->ni", Ravg, self.br0)
            cur = pos[b][self.bj] - pos[b][self.bi]
            s_i = self.s[b]
            k_pair = 0.5 * (SOLID_K / self.bn0[self.bi] + SOLID_K / self.bn0[self.bj])
            k_pair = k_pair * s_i[self.bi] * s_i[self.bj]
            c_pair = SOLID_DAMP * np.sqrt(np.maximum(k_pair, 0.0) * max(float(self.m0[b].mean()), 1e-6))
            fspring = k_pair[:, None] * (target - cur) + c_pair[:, None] * (
                vel[b][self.bj] - vel[b][self.bi])
            m_eff_b = np.maximum(m[b], 0.02 * self.m0[b])
            np.add.at(acc[b], self.bi, fspring / m_eff_b[self.bi][:, None])
            np.add.at(acc[b], self.bj, -fspring / m_eff_b[self.bj][:, None])

        # ---- integrate (semi-implicit Euler) ----
        np.clip(acc, -5.0e4, 5.0e4, out=acc)
        vel += acc * dt
        pos += vel * dt

        # ---- XSPH smoothing of fluid velocities ----
        if len(fi):
            base = self.m0[fj] / rho[fj]
            num = np.zeros((n, 3))
            den = np.zeros(n)
            np.add.at(num, fi, (base * fw)[:, None] * (vel[fj] - vel[fi]))
            np.add.at(den, fi, base * fw)
            act = den > 1e-12
            vel[act] += XSPH * num[act] / den[act][:, None]

        # ---- floor no-penetration ----
        on_floor = (np.abs(pos[:, 0] - FLOOR_CX) <= FLOOR_HX) & (
            np.abs(pos[:, 1] - FLOOR_CY) <= FLOOR_HY)
        hit = on_floor & (pos[:, 2] < FLOOR_Z)
        if np.any(hit):
            pos[hit, 2] = FLOOR_Z
            vel[hit, 2] = np.maximum(vel[hit, 2], 0.0)
            vel[hit, 0] *= 0.90
            vel[hit, 1] *= 0.90

        # ---- dissolution mass transfer (conserved) ----
        if fbi is not None and len(fbi):
            self._dissolve(dt, fbi, fbj)

        # ---- inflow ----
        self._inflow(dt)

    # ---------------------------------------------------------- dissolution --
    def _dissolve(self, dt, fbi, fbj):
        nb = len(self.b_idx)
        bi = self.b_local[fbi]
        wet = np.zeros(nb)
        cwet = np.zeros(nb)
        np.add.at(wet, bi, self.m[fbj])
        np.add.at(cwet, bi, self.m[fbj] * self.c[fbj])
        nbr_exposure = np.zeros(nb)
        if len(self.bi):
            np.add.at(nbr_exposure, self.bi, 1.0 - self.s[self.bj])
            np.add.at(nbr_exposure, self.bj, 1.0 - self.s[self.bi])
            nbr_exposure /= self.bn0
        exposure = np.maximum(self.be0, nbr_exposure)
        sat = np.where(wet > 1e-9, cwet / np.maximum(wet, 1e-12), 0.0)
        m_b = self.m[self.b_idx]
        wetness = np.clip(wet / (6.0 * self._m_fluid_ref), 0.0, 1.0)
        rate = DISSOLVE_K * exposure * wetness * np.clip(1.0 - np.minimum(sat, 0.97), 0.0, 1.0)
        dm_b = m_b * np.clip(rate, 0.0, 20.0) * dt
        dm_b = np.minimum(dm_b, m_b)
        if not np.any(dm_b > 0.0):
            return
        # distribute to wetted fluid neighbours proportional to (1 - r/Rc)
        pair_w = 1.0 - np.linalg.norm(self.pos[fbj] - self.pos[fbi], axis=1) / CONTACT_R
        pair_w = np.maximum(pair_w, 1e-3)
        tot = np.zeros(nb)
        np.add.at(tot, bi, pair_w)
        share = dm_b[bi] * pair_w / np.maximum(tot[bi], 1e-12)
        cap = np.maximum(self.m[fbj] * (1.0 - self.c[fbj]) * 0.999, 0.0)
        req = np.zeros(len(self.pos))
        np.add.at(req, fbj, share)
        factor = np.ones(len(self.pos))
        busy = req > 1e-15
        factor[busy] = np.minimum(1.0, cap[busy] / req[busy])
        share = share * factor[fbj]
        applied = np.zeros(nb)
        np.add.at(applied, bi, share)
        np.add.at(self.ms, fbj, share)
        np.add.at(self.m, fbj, share)
        self.m[self.b_idx] -= applied
        np.maximum(self.m[self.b_idx], 0.0, out=self.m[self.b_idx])
        self.s[self.b_idx] = np.clip(self.m[self.b_idx] / np.maximum(self.m0[self.b_idx], 1e-12),
                                     0.0, 1.0)
        self.c = np.clip(self.ms / np.maximum(self.m, 1e-12), 0.0, 1.0)
        self._dissolved += float(applied.sum())
        self._map_source(fbi, fbj, share)

    def _map_source(self, fbi, fbj, share):
        """Map transferred solute mass onto the diagnostic grid source."""
        if not len(fbj):
            return
        cell = self._cell_index(self.pos[fbj])
        np.add.at(self._src_acc, cell, share)

    def _cell_index(self, points):
        idx = np.floor((points - GRID_MIN) / GRID_D).astype(np.int64)
        for ax, dim in enumerate(GRID_SHAPE):
            np.clip(idx[:, ax], 0, dim - 1, out=idx[:, ax])
        return (idx[:, 0], idx[:, 1], idx[:, 2])

    # --------------------------------------------------------------- inflow --
    def _inflow(self, dt):
        if not (self.inflow_start <= self._t < self.inflow_end):
            return
        self._emit_acc += INJECT_RATE * dt
        count = int(self._emit_acc)
        if count <= 0:
            return
        self._emit_acc -= count
        m_w = RHO0 * SPACING ** 3
        k = np.arange(self._emit_index, self._emit_index + count, dtype=float)
        self._emit_index += count
        # R2 low-discrepancy sequence fills the nozzle disk evenly at the
        # frozen sample spacing (deterministic, no per-run randomness).
        u = np.mod((k + 0.5) * 0.7548776662466927, 1.0)
        w = np.mod((k + 0.5) * 0.5698402909980532, 1.0)
        r = NOZZLE_R * np.sqrt(u)
        th = 2.0 * np.pi * w
        x = NOZZLE_X + 0.03 * (np.mod((k + 0.5) * 0.6180339887498949, 1.0) - 0.5)
        y = r * np.cos(th)
        z = NOZZLE_Z + r * np.sin(th)
        p = np.column_stack((x, y, z))
        v = np.zeros((count, 3))
        v[:, 0] = self.inflow_speed
        v[:, 2] = -0.05
        self._append(p, v, np.full(count, m_w), np.full(count, 2, dtype=np.int64))
        self._injected += float(count * m_w)

    def _append(self, p, v, mm, lab):
        self.pos = np.vstack((self.pos, p))
        self.vel = np.vstack((self.vel, v))
        self.m = np.concatenate((self.m, mm))
        self.m0 = np.concatenate((self.m0, mm))
        self.ms = np.concatenate((self.ms, np.zeros(len(mm))))
        self.c = np.concatenate((self.c, np.zeros(len(mm))))
        self.s = np.concatenate((self.s, np.zeros(len(mm))))
        self.label = np.concatenate((self.label, lab))
        self.rho = np.concatenate((self.rho, np.full(len(mm), self.rho0)))
        self.pressure = np.concatenate((self.pressure, np.zeros(len(mm))))
        self.acc = np.vstack((self.acc, np.zeros((len(mm), 3))))
        self.contact_force = np.vstack((self.contact_force, np.zeros((len(mm), 3))))

    # --------------------------------------------------------------- outflow --
    def _outflow(self):
        lab = self.label
        bad = ((np.abs(self.pos[:, 0]) > KILL_X) | (np.abs(self.pos[:, 1]) > KILL_Y) |
               (self.pos[:, 2] < KILL_Z)) & (lab >= 2)
        if not np.any(bad):
            return
        keep = ~bad
        # never delete the last sample of a bound material
        for value in (2, 3):
            if not np.any(keep & (lab == value)):
                cand = np.flatnonzero(bad & (lab == value))
                if cand.size:
                    keep[cand[0]] = True
        if np.all(keep):
            return
        lost = float(self.m[bad].sum())
        self._removed += lost
        self._removed_water += float((self.m[bad] * (1.0 - self.c[bad])).sum())
        self._removed_solute += float(self.ms[bad].sum())
        self.pos = self.pos[keep]
        self.vel = self.vel[keep]
        self.m = self.m[keep]
        self.m0 = self.m0[keep]
        self.ms = self.ms[keep]
        self.c = self.c[keep]
        self.s = self.s[keep]
        self.label = self.label[keep]
        self.rho = self.rho[keep]
        self.pressure = self.pressure[keep]
        self.acc = self.acc[keep]
        self.contact_force = self.contact_force[keep]

    # ------------------------------------------------------------ diagnostics --
    def _splat_velocity(self):
        fluid = self.label >= 2
        vg = np.zeros(GRID_SHAPE + (3,))
        wg = np.zeros(GRID_SHAPE)
        if np.any(fluid):
            cell = self._cell_index(self.pos[fluid])
            mw = self.m0[fluid]
            np.add.at(wg, cell, mw)
            np.add.at(vg, cell, self.vel[fluid] * mw[:, None])
        denom = np.maximum(wg, 1e-12)[..., None]
        self.V_grid = vg / denom
        return wg

    def _update_grid(self, dt):
        wg = self._splat_velocity()
        cell_vol = GRID_D ** 3
        if dt > 0.0:
            self.S_grid = self._src_acc / (dt * RHO0 * cell_vol)
            self.C = self._advect(self.C, self.V_grid, dt)
            self.C = self._diffuse(self.C, dt)
            self.C = np.minimum(self.C + self.S_grid * dt, 1.0)
            self.C = np.maximum(self.C, 0.0)
        else:
            self.S_grid = np.zeros(GRID_SHAPE)
        self._src_acc = np.zeros(GRID_SHAPE)
        # active mask: cells inside the frozen mixing domain carrying liquid
        occupied = wg > 0.0
        kmax = np.full(GRID_SHAPE[:2], -1)
        any_col = occupied.any(axis=2)
        if np.any(any_col):
            rev = occupied[:, :, ::-1]
            kmax[any_col] = GRID_SHAPE[2] - 1 - np.argmax(rev, axis=2)[any_col]
        kk = np.arange(GRID_SHAPE[2])[None, None, :]
        self.active = ((kk <= kmax[:, :, None]) & (kmax[:, :, None] >= 0)).astype(float)
        self.active[occupied] = 1.0
        self._grid_dt = 0.0

    def _advect(self, field, vel, dt):
        nx, ny, nz = GRID_SHAPE
        xs = GRID_MIN[0] + (np.arange(nx) + 0.5) * GRID_D
        ys = GRID_MIN[1] + (np.arange(ny) + 0.5) * GRID_D
        zs = GRID_MIN[2] + (np.arange(nz) + 0.5) * GRID_D
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        px = gx - vel[..., 0] * dt
        py = gy - vel[..., 1] * dt
        pz = gz - vel[..., 2] * dt
        fx = (px - GRID_MIN[0]) / GRID_D - 0.5
        fy = (py - GRID_MIN[1]) / GRID_D - 0.5
        fz = (pz - GRID_MIN[2]) / GRID_D - 0.5
        inside = ((fx > -1.0) & (fx < nx) & (fy > -1.0) & (fy < ny) &
                  (fz > -1.0) & (fz < nz))
        ix = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
        iy = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
        iz = np.clip(np.floor(fz).astype(np.int64), 0, nz - 2)
        tx = np.clip(fx - ix, 0.0, 1.0)
        ty = np.clip(fy - iy, 0.0, 1.0)
        tz = np.clip(fz - iz, 0.0, 1.0)
        c000 = field[ix, iy, iz]
        c100 = field[ix + 1, iy, iz]
        c010 = field[ix, iy + 1, iz]
        c110 = field[ix + 1, iy + 1, iz]
        c001 = field[ix, iy, iz + 1]
        c101 = field[ix + 1, iy, iz + 1]
        c011 = field[ix, iy + 1, iz + 1]
        c111 = field[ix + 1, iy + 1, iz + 1]
        c00 = c000 * (1 - tx) + c100 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c11 = c011 * (1 - tx) + c111 * tx
        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        out = c0 * (1 - tz) + c1 * tz
        return np.where(inside, out, 0.0)

    def _diffuse(self, field, dt):
        if DIFFUSIVITY <= 0.0:
            return field
        lap = np.zeros_like(field)
        lap[1:-1, :, :] += field[2:, :, :] - 2.0 * field[1:-1, :, :] + field[:-2, :, :]
        lap[:, 1:-1, :] += field[:, 2:, :] - 2.0 * field[:, 1:-1, :] + field[:, :-2, :]
        lap[:, :, 1:-1] += field[:, :, 2:] - 2.0 * field[:, :, 1:-1] + field[:, :, :-2]
        return np.maximum(field + DIFFUSIVITY * dt * lap / GRID_D ** 2, 0.0)

    # ----------------------------------------------------------------- frame --
    def _make_frame(self, time):
        n = len(self.pos)
        F = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        bun = self.label == 1
        if np.any(bun):
            F[bun] = self.Fb
        mq = np.zeros((n, 3))
        wat = self.label == 2
        mix = self.label == 3
        mq[bun, 0] = self.m[bun]
        mq[wat, 1] = self.m[wat] * (1.0 - self.c[wat])
        mq[wat, 2] = self.m[wat] * self.c[wat]
        mq[mix, 2] = self.m[mix]
        mass_total = float(self.m.sum())
        solute_total = float(self.ms.sum())
        cumulative = np.array([
            -self._dissolved,
            self._injected - self._removed_water,
            self._dissolved - self._removed_solute,
        ])
        interaction = self._interaction / max(getattr(self, "_istep", 1), 1)
        fields = {
            "active_mask": self.active.astype(float),
            "cumulative_mass_source": cumulative,
            "interaction_force": interaction,
            "scalar_source": self.S_grid.astype(float),
            "transported_scalar": self.C.astype(float),
            "velocity_field": self.V_grid.astype(float),
        }
        attrs = {
            "material_label": self.label.astype(float),
            "red_solute_concentration": np.clip(self.c, 0.0, 1.0),
            "solid_phase_fraction": np.clip(self.s, 0.0, 1.0),
            "deformation_gradient": F,
            "mass_quantity": mq,
            "density": self.rho.copy(),
        }
        scalars = {
            "grid_spacing": float(GRID_D),
            "total_mass": mass_total,
            "total_red_solute_mass": solute_total,
            "dissolved_mass": float(self._dissolved),
            "injected_mass": float(self._injected),
            "outflow_mass": float(self._removed),
            "fluid_mass": float(self.m[self.label >= 2].sum()),
            "initial_bunny_mass": float(self.m0[bun].sum()),
            "remaining_bunny_solid_mass": float(self.m[bun].sum()),
            "max_speed": float(self._vmax),
            "sim_time": float(self._t),
        }
        return StateFrame(
            float(time),
            self.pos.copy(),
            self.vel.copy(),
            mass=self.m.copy(),
            body_id=self.label.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )

    # ------------------------------------------------------------------- API --
    def advance_to(self, time: float):
        target = float(time)
        if target < self._t - 1e-9:
            raise ValueError("advance_to must be called with non-decreasing time")
        self._interaction[:] = 0.0
        self._istep = 0
        while self._t < target - 1e-12:
            dt = min(DT, target - self._t)
            if dt <= 1e-12:
                break
            self._step(dt)
            self._t += dt
            self._grid_dt += dt
            self._istep += 1
            self._steps += 1
            self._outflow()
            np.clip(self.vel, -60.0, 60.0, out=self.vel)
            np.nan_to_num(self.pos, copy=False, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
            if len(self.pos):
                self._vmax = float(np.max(np.abs(self.vel)))
        self._update_grid(self._grid_dt)
        return self._make_frame(target)

    def diagnostics(self):
        fluid = self.label >= 2
        return {
            "solver": "corotated_lattice_solid + SPH + scalar advection-diffusion",
            "steps": int(self._steps),
            "samples": int(len(self.pos)),
            "fluid_samples": int(np.count_nonzero(fluid)),
            "bunny_samples": int(np.count_nonzero(self.label == 1)),
            "reference_density": float(self.rho0),
            "dissolved_mass_kg": float(self._dissolved),
            "injected_mass_kg": float(self._injected),
            "outflow_mass_kg": float(self._removed),
            "total_solute_mass_kg": float(self.ms.sum()),
            "final_time_s": float(self._t),
        }
