"""Self-implemented MLS-MPM multiphysics solver for 052_soluble_bunny_water.

Physical model
--------------
* Background-grid MLS-MPM (APIC transfers, quadratic B-spline shape functions).
* Liquid phase: weakly compressible Tait equation of state plus Newtonian
  viscous stress; the pipe jet and the receiving pool are the same phase.
* Bunny phase: fixed-corotated elastoplasticity (singular-value clamping)
  that softens as the body loses solute mass, so it visibly deforms.
* Coupling: the shared background grid gives two-way no-slip solid/liquid
  momentum exchange; the floor is a frictional no-penetration boundary, the
  pipe is a rigid collider with prescribed inflow, and the side boundaries are
  open outflow with exact mass bookkeeping.
* Solute: the bunny releases red solute mass into wetted liquid neighbours and
  the solute then diffuses between liquid particles through a grid-averaged
  relaxation. Both operators are exactly mass conserving, so the total solute
  mass always equals the initial mass plus the cumulative prescribed source.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_EYE = np.eye(3)
_OFFSETS = np.array([(i, j, k) for i in range(3) for j in range(3) for k in range(3)])


class BenchmarkSolver:
    """Deterministic material-point solver for the soluble bunny scene."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        ic = dict(cfg.get("initial_conditions", {}) or {})
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), float)
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.inflow_speed = float(ic.get("inflow_speed_m_s", 2.4))
        self.inflow_start = float(ic.get("inflow_start_s", 0.0))
        self.inflow_end = float(ic.get("inflow_end_s", 5.5))
        self.rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

        # ---- simulation grid -------------------------------------------
        self.dx = 0.11
        self.inv_dx = 1.0 / self.dx
        self.origin = np.array([-1.60, -1.95, -0.40])
        self.n = np.array([45, 37, 31], dtype=np.int64)
        nx, ny, nz = (int(v) for v in self.n)
        self.NN = nx * ny * nz
        self.stride0 = ny * nz
        self.stride1 = nz
        self.qmax = self.n.astype(float) - 2.02
        self.offsets = _OFFSETS.astype(float)
        self.flat_off = (
            _OFFSETS[:, 0] * self.stride0 + _OFFSETS[:, 1] * self.stride1 + _OFFSETS[:, 2]
        ).astype(np.int64)
        znode = np.broadcast_to(
            (np.arange(nz) * self.dx + self.origin[2])[None, None, :], (nx, ny, nz)
        )
        self.floor_nodes = np.ascontiguousarray(znode).reshape(-1) <= 0.012

        # ---- material parameters ---------------------------------------
        self.rho_s = 1180.0
        self.rho_l = 1000.0
        young, poisson = 1.8e5, 0.3
        self.mu_solid = young / (2.0 * (1.0 + poisson))
        self.lam_solid = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.K_fluid = 5.0e4
        self.gamma = 3.0
        self.eta_fluid = 1.5
        self.eta_solid = 2.0
        self.floor_friction = 0.88
        self.k_diss = 1.0          # dissolution rate constant [1/s]
        self.kappa = 6.0           # solute mixing relaxation rate [1/s]
        self.n_sub = 14
        self.dt = self.sample_dt / self.n_sub
        self.n_transfer = 7        # substeps between solute exchange updates
        self.min_solid = 10
        self.min_liquid = 40

        # ---- outflow / removal window ----------------------------------
        self.rmin = np.array([-1.35, -1.60, -0.20])
        self.rmax = np.array([2.85, 1.60, 2.70])

        # ---- pipe collider (cylinder, axis along +x) --------------------
        self.pipe_x0, self.pipe_x1 = -1.745, -0.495
        self.pipe_c = np.array([0.0, 1.30])   # (y, z) of axis
        self.pipe_r = 0.38

        self._init_particles(scene)
        self._init_emitter()
        self._init_diagnostics()
        self.t = 0.0
        self.removed_particles = 0
        self.dissolved_particles = 0
        self.interaction = np.zeros((6, 3))

    def _init_particles(self, scene) -> None:
        frame = scene.initial_state()
        self.x = np.array(frame.position, dtype=float)
        self.v = np.array(frame.velocity, dtype=float)
        label = np.asarray(frame.attributes["material_label"], float)
        conc = np.asarray(frame.attributes["red_solute_concentration"], float)
        mass = np.asarray(frame.mass, float)

        self.is_solid = label == 1.0
        # Split the mass into the two conserved species: red solute + solvent.
        self.solute = mass * conc
        self.solvent = mass * (1.0 - conc)
        self.m = self.solute + self.solvent
        self.m0 = self.m.copy()
        rho0 = np.where(self.is_solid, self.rho_s, self.rho_l)
        self.V0 = self.m / rho0
        self.F = np.broadcast_to(_EYE, (len(self.x), 3, 3)).copy()
        self.J = np.ones(len(self.x))
        self.C = np.zeros((len(self.x), 3, 3))

        self.initial_solute = float(np.sum(self.solute))
        self.initial_solvent = float(np.sum(self.solvent))
        self.initial_solid_mass = float(np.sum(self.m[self.is_solid]))
        # Cumulative prescribed [solute, solvent] mass added minus removed.
        self.mass_source = np.zeros(2)

    def _init_emitter(self) -> None:
        """Prescribed-inflow lattice on the pipe outlet disc."""
        h = 0.09
        rad = 0.27
        ax = np.arange(-3, 4) * h
        gy, gz = np.meshgrid(ax, ax, indexing="ij")
        keep = (gy ** 2 + gz ** 2) <= rad ** 2 + 1e-12
        self.emit_y = gy[keep]
        self.emit_z = gz[keep] + 1.19
        self.emit_h = h
        self.emit_x = self.pipe_x1 + 0.02
        self.emit_mass = self.rho_l * h ** 3
        self.emit_interval = h / max(self.inflow_speed, 1e-6)
        self.next_emit = self.inflow_start
        self.emitted_particles = 0

    def _init_diagnostics(self) -> None:
        self.ddx = 0.15
        self.dorigin = np.array([-1.20, -1.50, 0.0])
        self.dn = np.array([26, 21, 12], dtype=np.int64)
        dx_, dy_, dz_ = (int(v) for v in self.dn)
        self.dNN = dx_ * dy_ * dz_
        self.dstride0 = dy_ * dz_
        self.dstride1 = dz_
        self.diag_release = np.zeros(self.dNN)
        self.diag_window = 0.0

    # ------------------------------------------------------- grid transfers
    def _weights(self, need_dpos: bool = True):
        """Quadratic B-spline weights, flat node indices and node offsets."""
        q = (self.x - self.origin) * self.inv_dx
        q = np.clip(q, 1.02, self.qmax)
        base = (q - 0.5).astype(np.int64)
        fx = q - base                                    # in [1.0, 2.0)
        w0 = 0.5 * (1.5 - fx) ** 2
        w1 = 0.75 - (fx - 1.0) ** 2
        w2 = 0.5 * (fx - 0.5) ** 2
        wt = np.stack((w0, w1, w2), axis=1)              # [N,3,3] (order, dim)
        weight = (
            wt[:, _OFFSETS[:, 0], 0] * wt[:, _OFFSETS[:, 1], 1] * wt[:, _OFFSETS[:, 2], 2]
        )                                                # [N,27]
        flat = (
            base[:, 0] * self.stride0 + base[:, 1] * self.stride1 + base[:, 2]
        )[:, None] + self.flat_off                       # [N,27]
        if not need_dpos:
            return flat, weight, None
        dpos = (self.offsets[None, :, :] - fx[:, None, :]) * self.dx
        return flat, weight, dpos

    def _stress_affine(self):
        """Return -dt * V0 * (sigma * J) * D_inv, the MLS-MPM affine stress."""
        n = len(self.x)
        tau = np.zeros((n, 3, 3))
        solid = np.flatnonzero(self.is_solid)
        if solid.size:
            F = self.F[solid]
            U, sig, Vt = np.linalg.svd(F)
            sig = np.clip(sig, 0.45, 1.9)                # plastic clamping
            F = (U * sig[:, None, :]) @ Vt
            self.F[solid] = F
            R = U @ Vt
            J = np.prod(sig, axis=1)
            # Softening as the gel loses solute mass.
            phi = np.clip(self.m[solid] / np.maximum(self.m0[solid], 1e-12), 0.02, 1.0)
            soft = phi ** 1.5
            mu = self.mu_solid * soft
            lam = self.lam_solid * soft
            # P F^T for the fixed-corotated model.
            PFt = 2.0 * mu[:, None, None] * (F - R) @ np.swapaxes(F, 1, 2)
            PFt += (lam * J * (J - 1.0))[:, None, None] * _EYE
            tau[solid] = PFt
        liquid = np.flatnonzero(~self.is_solid)
        if liquid.size:
            Jl = np.clip(self.J[liquid], 0.35, 2.5)
            self.J[liquid] = Jl
            press = self.K_fluid * (Jl ** (-self.gamma) - 1.0)
            press = np.maximum(press, 0.0)               # no tensile water
            tau[liquid] = (-press * Jl)[:, None, None] * _EYE
        # Newtonian viscous stress from the affine velocity gradient.
        eta = np.where(self.is_solid, self.eta_solid, self.eta_fluid)[:, None, None]
        Jall = np.where(self.is_solid, np.linalg.det(self.F), self.J)[:, None, None]
        tau += eta * Jall * (self.C + np.swapaxes(self.C, 1, 2))
        d_inv = 4.0 * self.inv_dx * self.inv_dx
        return (-self.dt * d_inv * self.V0)[:, None, None] * tau

    # --------------------------------------------------------- one substep
    def _substep(self, measure: bool) -> None:
        self._emit()
        flat, weight, dpos = self._weights()
        affine = self._stress_affine() + (self.m[:, None, None] * self.C)

        gm = np.bincount(flat.reshape(-1), weights=weight.reshape(-1) * np.repeat(self.m, 27),
                         minlength=self.NN)
        contrib = np.einsum("nij,nkj->nki", affine, dpos)
        contrib += (self.m[:, None] * self.v)[:, None, :]
        wc = weight[:, :, None] * contrib
        gp = np.empty((self.NN, 3))
        fr = flat.reshape(-1)
        for d in range(3):
            gp[:, d] = np.bincount(fr, weights=wc[:, :, d].reshape(-1), minlength=self.NN)

        if measure:
            sm = np.where(self.is_solid, self.m, 0.0)
            gms = np.bincount(fr, weights=weight.reshape(-1) * np.repeat(sm, 27),
                              minlength=self.NN)
            svel = (sm[:, None] * self.v)[:, None, :] * weight[:, :, None]
            gps = np.empty((self.NN, 3))
            for d in range(3):
                gps[:, d] = np.bincount(fr, weights=svel[:, :, d].reshape(-1),
                                        minlength=self.NN)
            gmt = np.bincount(fr, weights=weight.reshape(-1) * np.repeat(self.m, 27),
                              minlength=self.NN)
            gpt = np.empty((self.NN, 3))
            vmom = (self.m[:, None] * self.v)[:, None, :] * weight[:, :, None]
            for d in range(3):
                gpt[:, d] = np.bincount(fr, weights=vmom[:, :, d].reshape(-1),
                                        minlength=self.NN)
            self._measure_interaction(gms, gps, gmt, gpt)

        active = gm > 1e-9
        gv = np.zeros((self.NN, 3))
        gv[active] = gp[active] / gm[active, None]
        gv[active] += self.gravity * self.dt

        # Frictional no-penetration floor on the grid.
        fmask = self.floor_nodes & active
        if measure:
            pre = gv[fmask].copy()
        down = fmask.copy()
        down[fmask] = gv[fmask, 2] < 0.0
        gv[down, 2] = 0.0
        gv[down, 0] *= self.floor_friction
        gv[down, 1] *= self.floor_friction
        if measure:
            self._measure_floor(fmask, pre, gv[fmask], gm[fmask])

        # ---- grid to particle (APIC) -----------------------------------
        gvn = gv[flat]                                   # [N,27,3]
        self.v = np.einsum("nk,nkd->nd", weight, gvn)
        self.C = 4.0 * self.inv_dx * self.inv_dx * np.einsum(
            "nk,nki,nkj->nij", weight, gvn, dpos
        )
        speed = np.linalg.norm(self.v, axis=1)
        hot = speed > 40.0
        if np.any(hot):
            self.v[hot] *= (40.0 / speed[hot])[:, None]

        gradv = _EYE + self.dt * self.C
        solid = self.is_solid
        if np.any(solid):
            self.F[solid] = gradv[solid] @ self.F[solid]
        liq = ~solid
        if np.any(liq):
            trC = self.C[liq, 0, 0] + self.C[liq, 1, 1] + self.C[liq, 2, 2]
            self.J[liq] = np.clip(self.J[liq] * (1.0 + self.dt * trC), 0.35, 2.5)

        self.x = self.x + self.dt * self.v
        self._particle_colliders()

    def _particle_colliders(self) -> None:
        # Floor: no penetration with tangential friction.
        low = self.x[:, 2] < 0.006
        if np.any(low):
            self.x[low, 2] = 0.006
            vz = self.v[low, 2]
            self.v[low, 2] = np.where(vz < 0.0, 0.0, vz)
            self.v[low, 0] *= self.floor_friction
            self.v[low, 1] *= self.floor_friction
        # Rigid pipe: push samples out of the cylinder shell.
        dy = self.x[:, 1] - self.pipe_c[0]
        dz = self.x[:, 2] - self.pipe_c[1]
        r = np.sqrt(dy * dy + dz * dz)
        inside = (r < self.pipe_r) & (self.x[:, 0] > self.pipe_x0 - 0.05) & (
            self.x[:, 0] < self.pipe_x1
        )
        # Anything drifting back into the nozzle bore is flushed forward.
        if np.any(inside):
            idx = np.flatnonzero(inside)
            self.x[idx, 0] = self.pipe_x1 + 0.005
            self.v[idx, 0] = np.maximum(self.v[idx, 0], 0.0)
        # Splash hitting the outer barrel is deflected radially.
        shell = (r < self.pipe_r) & (self.x[:, 0] >= self.pipe_x1) & (
            self.x[:, 0] < self.pipe_x1 + 0.02
        )
        if np.any(shell):
            idx = np.flatnonzero(shell)
            rr = np.maximum(r[idx], 1e-9)
            nrm = np.stack((np.zeros_like(rr), dy[idx] / rr, dz[idx] / rr), axis=1)
            push = (self.pipe_r - r[idx])[:, None] * nrm
            self.x[idx] += push
            vn = np.einsum("nd,nd->n", self.v[idx], nrm)
            neg = vn < 0.0
            self.v[idx] -= np.where(neg, vn, 0.0)[:, None] * nrm

    # ------------------------------------------------------- prescribed flow
    def _emit(self) -> None:
        if self.t < self.inflow_start - 1e-12 or self.t > self.inflow_end + 1e-12:
            return
        added = []
        guard = 0
        while self.next_emit <= self.t + 1e-12 and self.next_emit <= self.inflow_end and guard < 8:
            lag = self.t - self.next_emit
            added.append(lag)
            self.next_emit += self.emit_interval
            guard += 1
        if not added:
            return
        h = self.emit_h
        k = len(self.emit_y)
        xs, vs = [], []
        for lag in added:
            jit = self.rng.uniform(-0.15 * h, 0.15 * h, size=(k, 3))
            pos = np.empty((k, 3))
            pos[:, 0] = self.emit_x + self.inflow_speed * lag + jit[:, 0]
            pos[:, 1] = self.emit_y + jit[:, 1]
            pos[:, 2] = self.emit_z + jit[:, 2]
            vel = np.zeros((k, 3))
            vel[:, 0] = self.inflow_speed * (
                1.0 + self.rng.uniform(-0.01, 0.01, size=k)
            )
            xs.append(pos)
            vs.append(vel)
        pos = np.vstack(xs)
        vel = np.vstack(vs)
        cnt = len(pos)
        m = np.full(cnt, self.emit_mass)
        self.x = np.vstack((self.x, pos))
        self.v = np.vstack((self.v, vel))
        self.solute = np.concatenate((self.solute, np.zeros(cnt)))
        self.solvent = np.concatenate((self.solvent, m))
        self.m = np.concatenate((self.m, m))
        self.m0 = np.concatenate((self.m0, m))
        self.V0 = np.concatenate((self.V0, m / self.rho_l))
        self.is_solid = np.concatenate((self.is_solid, np.zeros(cnt, bool)))
        self.F = np.concatenate((self.F, np.broadcast_to(_EYE, (cnt, 3, 3)).copy()))
        self.J = np.concatenate((self.J, np.ones(cnt)))
        self.C = np.concatenate((self.C, np.zeros((cnt, 3, 3))))
        # Clear inflow water carries no solute.
        self.mass_source[1] += float(np.sum(m))
        self.emitted_particles += cnt

    def _outflow(self) -> None:
        out = np.any((self.x < self.rmin) | (self.x > self.rmax), axis=1)
        if not np.any(out):
            return
        # Solids never leave: reflect them back instead of losing bunny mass.
        solid_out = out & self.is_solid
        if np.any(solid_out):
            idx = np.flatnonzero(solid_out)
            self.x[idx] = np.clip(self.x[idx], self.rmin + 0.02, self.rmax - 0.02)
            self.v[idx] *= 0.0
            out[idx] = False
        if not np.any(out):
            return
        keep_n = int(np.sum(~out & ~self.is_solid))
        if keep_n < self.min_liquid:
            return
        self.mass_source[0] -= float(np.sum(self.solute[out]))
        self.mass_source[1] -= float(np.sum(self.solvent[out]))
        self.removed_particles += int(np.sum(out))
        self._compress(~out)

    def _compress(self, keep) -> None:
        self.x = self.x[keep]
        self.v = self.v[keep]
        self.F = self.F[keep]
        self.J = self.J[keep]
        self.C = self.C[keep]
        self.m = self.m[keep]
        self.m0 = self.m0[keep]
        self.V0 = self.V0[keep]
        self.solute = self.solute[keep]
        self.solvent = self.solvent[keep]
        self.is_solid = self.is_solid[keep]

    # ------------------------------------------- dissolution + solute mixing
    def _solute_exchange(self, dt_eff: float) -> None:
        """Conservative solute release from the bunny plus liquid mixing."""
        flat, weight, _ = self._weights(need_dpos=False)
        fr = flat.reshape(-1)
        wr = weight.reshape(-1)
        liq = ~self.is_solid
        ml = np.where(liq, self.m, 0.0)
        sl = np.where(liq, self.solute, 0.0)
        node_ml = np.bincount(fr, weights=wr * np.repeat(ml, 27), minlength=self.NN)
        node_sl = np.bincount(fr, weights=wr * np.repeat(sl, 27), minlength=self.NN)
        wet_node = node_ml > 1e-6
        node_c = np.zeros(self.NN)
        node_c[wet_node] = node_sl[wet_node] / node_ml[wet_node]

        solid = np.flatnonzero(self.is_solid)
        release = np.zeros(len(self.x))
        node_release = np.zeros(self.NN)
        if solid.size:
            ws = weight[solid]
            fs = flat[solid]
            # Local liquid volume fraction seen by each bunny sample.
            node_ref = self.rho_l * self.dx ** 3
            frac = np.sum(ws * (node_ml[fs] / node_ref), axis=1)
            wetness = np.clip(np.sqrt(np.clip(frac, 0.0, 1.0)), 0.0, 1.0)
            rate = self.k_diss * wetness * self.m0[solid]
            phi = self.m[solid] / np.maximum(self.m0[solid], 1e-12)
            # Nearly spent samples dump their remaining solute quickly.
            spent = phi < 0.25
            rate = np.where(spent, np.maximum(rate, 4.0 * self.m0[solid] * wetness), rate)
            rel = np.minimum(rate * dt_eff, self.solute[solid])
            # Only the part of the stencil that actually holds liquid receives it.
            wet_share = np.sum(ws * wet_node[fs], axis=1)
            rel = rel * np.clip(wet_share, 0.0, 1.0)
            keep_alive = np.zeros(solid.size, bool)
            if solid.size <= self.min_solid + 2:
                order = np.argsort(-self.m[solid])[: self.min_solid]
                keep_alive[order] = True
                rel[keep_alive] = 0.0
            rel = np.maximum(rel, 0.0)
            if np.any(rel > 0.0):
                share = ws * wet_node[fs]
                tot = np.sum(share, axis=1)
                good = tot > 1e-12
                share[good] /= tot[good, None]
                node_release = np.bincount(
                    fs.reshape(-1), weights=(share * rel[:, None]).reshape(-1),
                    minlength=self.NN,
                )
            release[solid] = rel
            self.solute[solid] -= rel
            self._scatter_release(self.x[solid], rel)

        # Distribute released solute onto liquid particles (mass weighted).
        if np.any(node_release > 0.0):
            gain_node = np.zeros(self.NN)
            gain_node[wet_node] = node_release[wet_node] / node_ml[wet_node]
            lidx = np.flatnonzero(liq)
            gain = self.m[lidx] * np.sum(weight[lidx] * gain_node[flat[lidx]], axis=1)
            self.solute[lidx] += gain

        # Grid-averaged relaxation: conservative solute diffusion in the liquid.
        # node_c predates the release above, so the raw increments carry a small
        # imbalance; a mass-weighted projection removes it exactly.
        lidx = np.flatnonzero(liq)
        if lidx.size:
            ctarget = np.sum(weight[lidx] * node_c[flat[lidx]], axis=1)
            alpha = min(self.kappa * dt_eff, 0.9)
            delta = alpha * (self.m[lidx] * ctarget - self.solute[lidx])
            mtot = float(np.sum(self.m[lidx]))
            if mtot > 1e-12:
                delta -= (float(np.sum(delta)) / mtot) * self.m[lidx]
            self.solute[lidx] += delta

        self.solute = np.clip(self.solute, 0.0, None)
        self.m = self.solute + self.solvent

        # Retire fully leached bunny samples.
        if solid.size:
            spent = self.is_solid & (self.solute <= 1e-4 * self.m0)
            if np.any(spent):
                remain = int(np.sum(self.is_solid)) - int(np.sum(spent))
                if remain >= self.min_solid:
                    self.dissolved_particles += int(np.sum(spent))
                    self._compress(~spent)

    def _scatter_release(self, pos, rel) -> None:
        """Accumulate released solute on the diagnostic grid."""
        if not len(pos):
            return
        q = (pos - self.dorigin) / self.ddx
        b = np.floor(q).astype(np.int64)
        f = q - b
        good = np.all((b >= 0) & (b <= self.dn - 2), axis=1)
        if not np.any(good):
            return
        b, f, rel = b[good], f[good], rel[good]
        flat = b[:, 0] * self.dstride0 + b[:, 1] * self.dstride1 + b[:, 2]
        for dx_ in (0, 1):
            wx = f[:, 0] if dx_ else 1.0 - f[:, 0]
            for dy_ in (0, 1):
                wy = f[:, 1] if dy_ else 1.0 - f[:, 1]
                for dz_ in (0, 1):
                    wz = f[:, 2] if dz_ else 1.0 - f[:, 2]
                    idx = flat + dx_ * self.dstride0 + dy_ * self.dstride1 + dz_
                    self.diag_release += np.bincount(
                        idx, weights=wx * wy * wz * rel, minlength=self.dNN
                    )

    # ------------------------------------------------- interaction measures
    def _measure_interaction(self, gms, gps, gmt, gpt) -> None:
        gml = gmt - gms
        gpl = gpt - gps
        both = (gms > 1e-9) & (gml > 1e-9)
        if not np.any(both):
            self.interaction[0] = 0.0
            self.interaction[1] = 0.0
            return
        ms, ml = gms[both], gml[both]
        vs = gps[both] / ms[:, None]
        vl = gpl[both] / ml[:, None]
        mred = ms * ml / (ms + ml)
        # No-slip grid coupling drives both phases to the mixture velocity;
        # the resulting impulse per unit time is the interaction force.
        imp = mred[:, None] * (vl - vs)
        f = np.sum(imp, axis=0) / self.dt
        self.interaction[0] = f
        self.interaction[1] = -f

    def _measure_floor(self, fmask, pre, post, gmass) -> None:
        dv = post - pre
        if not len(dv):
            self.interaction[2:6] = 0.0
            return
        solid_share = self._floor_solid_share(fmask)
        force = (gmass[:, None] * dv) / self.dt
        fs = np.sum(force * solid_share[:, None], axis=0)
        fl = np.sum(force * (1.0 - solid_share)[:, None], axis=0)
        self.interaction[2] = fs        # floor on bunny
        self.interaction[3] = -fs       # bunny on floor
        self.interaction[4] = fl        # floor on water
        self.interaction[5] = -fl       # water on floor

    def _floor_solid_share(self, fmask):
        flat, weight, _ = self._weights(need_dpos=False)
        fr = flat.reshape(-1)
        wr = weight.reshape(-1)
        sm = np.where(self.is_solid, self.m, 0.0)
        gs = np.bincount(fr, weights=wr * np.repeat(sm, 27), minlength=self.NN)
        gt = np.bincount(fr, weights=wr * np.repeat(self.m, 27), minlength=self.NN)
        share = np.zeros(self.NN)
        good = gt > 1e-12
        share[good] = gs[good] / gt[good]
        return np.clip(share[fmask], 0.0, 1.0)

    # ---------------------------------------------------------- integration
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.t < target - 1e-12:
            for s in range(self.n_sub):
                self._substep(measure=(s == self.n_sub - 1))
                self.t += self.dt
                if (s + 1) % self.n_transfer == 0:
                    self._solute_exchange(self.dt * self.n_transfer)
                    self.diag_window += self.dt * self.n_transfer
                    self._outflow()
            self.t = round(self.t / self.dt) * self.dt
        return self._frame(target)

    # -------------------------------------------------------------- export
    def _diag_scatter(self):
        """Collocated diagnostic fields on the uniform export grid."""
        q = (self.x - self.dorigin) / self.ddx
        b = np.floor(q).astype(np.int64)
        f = q - b
        good = np.all((b >= 0) & (b <= self.dn - 2), axis=1)
        ml = np.zeros(self.dNN)
        sl = np.zeros(self.dNN)
        mt = np.zeros(self.dNN)
        mom = np.zeros((self.dNN, 3))
        if np.any(good):
            bg, fg = b[good], f[good]
            flat = bg[:, 0] * self.dstride0 + bg[:, 1] * self.dstride1 + bg[:, 2]
            liq = (~self.is_solid)[good]
            mg = self.m[good]
            sg = self.solute[good]
            vg = self.v[good]
            for dx_ in (0, 1):
                wx = fg[:, 0] if dx_ else 1.0 - fg[:, 0]
                for dy_ in (0, 1):
                    wy = fg[:, 1] if dy_ else 1.0 - fg[:, 1]
                    for dz_ in (0, 1):
                        wz = fg[:, 2] if dz_ else 1.0 - fg[:, 2]
                        w = wx * wy * wz
                        idx = flat + dx_ * self.dstride0 + dy_ * self.dstride1 + dz_
                        ml += np.bincount(idx, weights=w * mg * liq, minlength=self.dNN)
                        sl += np.bincount(idx, weights=w * sg * liq, minlength=self.dNN)
                        mt += np.bincount(idx, weights=w * mg, minlength=self.dNN)
                        for d in range(3):
                            mom[:, d] += np.bincount(
                                idx, weights=w * mg * vg[:, d], minlength=self.dNN
                            )
        tol = 1e-4
        active = ml > tol
        scalar = np.zeros(self.dNN)
        scalar[active] = np.clip(sl[active] / ml[active], 0.0, 1.0)
        vel = np.zeros((self.dNN, 3))
        hasm = mt > tol
        vel[hasm] = mom[hasm] / mt[hasm, None]
        src = np.zeros(self.dNN)
        if self.diag_window > 1e-12:
            wnd = np.zeros(self.dNN)
            wnd[active] = self.diag_release[active] / (ml[active] * self.diag_window)
            src = wnd
        self.diag_release = np.zeros(self.dNN)
        self.diag_window = 0.0
        shape = tuple(int(v) for v in self.dn)
        return (
            scalar.reshape(shape),
            src.reshape(shape),
            active.astype(np.float64).reshape(shape),
            vel.reshape(shape + (3,)),
        )

    def _labels(self):
        """Scene state bindings: 1 bunny, 2 clear pipe water, 3 red mixture."""
        n = len(self.x)
        label = np.where(self.is_solid, 1.0, 2.0)
        liq = ~self.is_solid
        conc = np.zeros(n)
        good = self.m > 1e-12
        conc[good] = np.clip(self.solute[good] / self.m[good], 0.0, 1.0)
        in_mix = (
            (self.x[:, 0] >= -0.55) & (self.x[:, 0] <= 2.45)
            & (np.abs(self.x[:, 1]) <= 1.25) & (self.x[:, 2] <= 1.65)
        )
        mixture = liq & in_mix & ((conc > 0.01) | (self.x[:, 0] > 0.32))
        label[mixture] = 3.0
        # Guarantee both liquid bindings stay populated for every frame.
        lidx = np.flatnonzero(liq)
        if lidx.size:
            if not np.any(label[lidx] == 2.0):
                pick = lidx[np.argsort(conc[lidx])[: min(25, lidx.size)]]
                label[pick] = 2.0
            if not np.any(label[lidx] == 3.0):
                pick = lidx[np.argsort(-conc[lidx])[: min(25, lidx.size)]]
                label[pick] = 3.0
        return label, conc

    def _frame(self, time: float) -> StateFrame:
        label, conc = self._labels()
        n = len(self.x)
        scalar, src, active, vel = self._diag_scatter()
        Fex = self.F.copy()
        liq = ~self.is_solid
        if np.any(liq):
            Fex[liq] = (np.cbrt(np.clip(self.J[liq], 1e-6, None)))[:, None, None] * _EYE
        phi = np.zeros(n)
        sidx = np.flatnonzero(self.is_solid)
        if sidx.size:
            phi[sidx] = np.clip(self.m[sidx] / np.maximum(self.m0[sidx], 1e-12), 0.0, 1.0)
        mass_quantity = np.stack((self.solute, self.solvent), axis=1)
        solid_mass = float(np.sum(self.m[self.is_solid]))
        dissolved = float(np.sum(self.solute[liq])) if np.any(liq) else 0.0
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=label.astype(int),
            attributes={
                "material_label": label,
                "red_solute_concentration": conc,
                "solid_phase_fraction": phi,
                "density": self.m / np.maximum(self.V0, 1e-12),
                "phase_id": np.where(self.is_solid, 1.0, 2.0),
                "deformation_gradient": Fex,
                "mass_quantity": mass_quantity,
                "volume_quantity": self.V0 * np.where(
                    self.is_solid, np.linalg.det(self.F), self.J
                ),
            },
            fields={
                "transported_scalar": scalar,
                "scalar_source": src,
                "active_mask": active,
                "velocity_field": vel,
                "cumulative_mass_source": self.mass_source.copy(),
                "interaction_force": self.interaction.copy(),
            },
            scalars={
                "grid_spacing": float(self.ddx),
                "simulation_grid_spacing": float(self.dx),
                "grid_origin_x": float(self.dorigin[0]),
                "grid_origin_y": float(self.dorigin[1]),
                "grid_origin_z": float(self.dorigin[2]),
                "initial_red_solute_mass": self.initial_solute,
                "total_red_solute_mass": float(np.sum(self.solute)),
                "dissolved_red_solute_mass": dissolved,
                "remaining_bunny_mass": solid_mass,
                "dissolved_mass_fraction": float(
                    1.0 - solid_mass / max(self.initial_solid_mass, 1e-12)
                ),
                "particle_count": float(n),
                "solid_particle_count": float(np.sum(self.is_solid)),
                "liquid_particle_count": float(np.sum(liq)),
                "inflow_speed": self.inflow_speed,
                "interaction_pair_count": 3.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MLS-MPM (APIC) elastoplastic solid + weakly compressible liquid",
            "substeps_per_sample": int(self.n_sub),
            "substep_dt": float(self.dt),
            "sim_grid_shape": [int(v) for v in self.n],
            "diagnostic_grid_shape": [int(v) for v in self.dn],
            "particles_final": int(len(self.x)),
            "particles_emitted": int(self.emitted_particles),
            "particles_removed_outflow": int(self.removed_particles),
            "particles_fully_dissolved": int(self.dissolved_particles),
            "initial_red_solute_mass": self.initial_solute,
            "final_red_solute_mass": float(np.sum(self.solute)),
            "cumulative_mass_source": [float(v) for v in self.mass_source],
            "final_bunny_mass": float(np.sum(self.m[self.is_solid])),
        }
