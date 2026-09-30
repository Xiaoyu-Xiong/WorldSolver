"""WorldSolver solver for scene 093: elastic rod meandering on a moving belt.

Physically-based discrete elastic rod simulation (no external physics engine):

* The rod is a naturally straight, inextensible rod discretized by nodes with
  a unit-tangent discrete bending energy (rigidity B = E*I) and edge length
  constraints solved by a position-based Gauss-Seidel (edge parity) projection
  every substep.
* The injector prescribes the leading node: it is kinematically fed downward
  at the frozen injection speed, and a new material node is spliced at the
  outlet whenever one edge worth of material has passed, so the rod grows
  exactly at the injection rate.
* The belt is a rigid horizontal substrate at z = 0 translating in +x.  Free
  rod nodes receive one-sided normal contact; Coulomb kinetic friction drags
  them toward the belt velocity.  Once a contacting node settles onto the
  belt velocity it sticks (frozen source substrate condition) and is carried
  away in +x.  The source neglects self-contact, so none is modeled.
* Uniform gravity acts throughout; light damping keeps the response in the
  source quasi-static regime.  A tiny deterministic lateral perturbation
  (seeded) breaks the perfect x-z symmetry so the compressed heel can buckle
  laterally; the meandering itself is produced entirely by the dynamics: with
  belt speed below injection speed the deposited trace is compressed and the
  contact point swings side to side, laying a serpentine trace.  After the
  belt speed is matched to the injection speed at t = 24 s the feed/take-up
  mismatch vanishes and the newly laid trace becomes straight.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

# Budget layout: G = 2 domains (0 = whole rod, 1 = deposited/stuck part),
# K = 6 external force terms.
_DOMAINS = 2
_TERMS = 6
_TERM_GRAVITY = 0
_TERM_CONTACT = 1
_TERM_FRICTION = 2
_TERM_PIN = 3
_TERM_ATTACH = 4
_TERM_PERTURB = 5


class BenchmarkSolver:
    """Discrete elastic rod fed by an injector onto a translating belt."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.seed = int(seed)
        self.g = np.asarray(cfg["gravity"], dtype=float)

        rod = cfg["rod"]
        self.radius = float(rod["radius"])
        self.rho = float(rod["density"])
        self.E = float(rod["young_modulus"])
        self.rhoA = self.rho * math.pi * self.radius ** 2
        self.B = self.E * math.pi * self.radius ** 4 / 4.0
        self.L_avail = float(rod["total_available_length"])

        inj = cfg["injector"]
        self.outlet = np.asarray(inj["outlet_position"], dtype=float)
        self.v_inj = float(inj["injection_speed"])
        self.t_start = float(inj["start_time"])
        self.t_end = float(inj["end_time"])

        belt = cfg["belt"]
        self.belt_v0 = float(belt["initial_speed"])
        self.belt_v1 = float(belt["matched_speed"])
        self.t_switch = float(belt["speed_switch_time"])
        center = np.asarray(belt["center"], dtype=float)
        he = np.asarray(belt["half_extents"], dtype=float)
        self.xmin = float(center[0] - he[0])
        self.xmax = float(center[0] + he[0])
        self.ymax = float(he[1])

        self.r = self.radius  # belt surface at z = 0, rod centerline at z = r

        # Numerical parameters (deterministic).
        self.h = 0.005          # target node spacing [m]
        self.dt = 0.00025       # substep [s]; 80 substeps per sample dt
        self.iters = 10         # constraint iterations per substep
        self.mu = 1.0           # kinetic friction coefficient
        self.vtol = 1.5e-3      # relative-speed tolerance for sticking [m/s]
        self.t_stick = 0.01     # settled contact time before sticking [s]
        self.damp = 2.5         # linear velocity damping [1/s]
        self.sigma = 2.0e-7     # lateral symmetry-breaking force scale [N]
        self.rng = np.random.default_rng((abs(self.seed) % 100000) * 977 + 13)

        self._init_from_frame(scene.initial_state())

        self.x_delete = self.xmax + 0.01
        self.nsteps = 0
        self.fed = 0.0
        self.fed_cap = max(self.L_avail - float(self.rest.sum()), 0.0)
        self.descent = 0.0
        self.mass_added = 0.0
        self.mass_removed = 0.0
        self.imp = np.zeros((_DOMAINS, _TERMS, 3))
        self.last_export = 0.0
        self.last_contact_pos = np.array([0.0, 0.0, self.r])
        self.first_step = True

    # ------------------------------------------------------------------ setup

    def _init_from_frame(self, frame) -> None:
        p0 = np.asarray(frame.position, dtype=float)
        v0 = np.asarray(frame.velocity, dtype=float)
        dep = np.asarray(frame.attributes["is_initially_deposited"], dtype=float) > 0.5
        pts = [p0[0]]
        vel = [v0[0]]
        depf = [bool(dep[0])]
        for k in range(len(p0) - 1):
            a, b = p0[k], p0[k + 1]
            va, vb = v0[k], v0[k + 1]
            L = float(np.linalg.norm(b - a))
            n = max(1, int(round(L / self.h)))
            for q in range(1, n + 1):
                f = q / n
                pts.append(a + (b - a) * f)
                vel.append(va + (vb - va) * f)
                depf.append(bool(dep[k + 1]) if q == n else False)
        self.x = np.asarray(pts, dtype=float)
        self.v = np.asarray(vel, dtype=float)
        self.stuck = np.asarray(depf, dtype=bool)
        self.timer = np.zeros(self.N)
        self.orig_dep = self.stuck.copy()
        d = np.diff(self.x, axis=0)
        self.rest = np.linalg.norm(d, axis=1)
        self._refresh_mass()

    @property
    def N(self) -> int:
        return len(self.x)

    def _refresh_mass(self) -> None:
        n = self.N
        rest = self.rest
        m = np.empty(n)
        m[0] = rest[0] / 2.0
        m[-1] = rest[-1] / 2.0
        if n > 2:
            m[1:-1] = (rest[:-1] + rest[1:]) / 2.0
        m *= self.rhoA
        self.m = m
        w = 1.0 / m
        w[0] = 0.0  # prescribed injector node
        w[self.stuck] = 0.0  # stuck to the belt
        self.w = w

    # ------------------------------------------------------------------ helpers

    def _belt_speed(self, t: float) -> float:
        if t < self.t_switch:
            return self.belt_v0
        ramp = 0.5
        if t < self.t_switch + ramp:
            f = (t - self.t_switch) / ramp
            return self.belt_v0 + (self.belt_v1 - self.belt_v0) * f
        return self.belt_v1

    def _in_footprint(self, x: np.ndarray) -> np.ndarray:
        return (
            (x[:, 0] > self.xmin)
            & (x[:, 0] < self.xmax)
            & (np.abs(x[:, 1]) < self.ymax)
        )

    def _bending_forces(self) -> np.ndarray:
        x = self.x
        n = self.N
        F = np.zeros((n, 3))
        if n < 3:
            return F
        ea = x[1:-1] - x[:-2]
        eb = x[2:] - x[1:-1]
        La = np.maximum(np.linalg.norm(ea, axis=1), 1e-12)
        Lb = np.maximum(np.linalg.norm(eb, axis=1), 1e-12)
        ta = ea / La[:, None]
        tb = eb / Lb[:, None]
        d = ta - tb
        ra = self.rest[:-1]
        rb = self.rest[1:]
        hbar = 0.5 * (ra + rb)
        # transverse projections of the tangent mismatch
        pa = d - ta * np.sum(ta * d, axis=1)[:, None]
        pb = d - tb * np.sum(tb * d, axis=1)[:, None]
        fa = (self.B / (hbar * La))[:, None] * pa
        fb = (self.B / (hbar * Lb))[:, None] * pb
        F[:-2] += fa
        F[2:] += fb
        F[1:-1] -= fa + fb
        return F

    # ------------------------------------------------------------------ stepping

    def _step(self) -> None:
        dt = self.dt
        self.nsteps += 1
        t = self.nsteps * dt
        vb = self._belt_speed(t)
        x = self.x
        v = self.v
        n = self.N

        # --- prescribed injection of the leading node -------------------
        feeding = (self.t_start <= t <= self.t_end) and (self.fed < self.fed_cap - 1e-15)
        if feeding:
            dlen = self.v_inj * dt
            self.descent += dlen
            self.fed += dlen
            self.mass_added += self.rhoA * dlen
            x[0] = (self.outlet[0], self.outlet[1], self.outlet[2] - self.descent)
            v[0] = (0.0, 0.0, -self.v_inj)
        else:
            v[0] = 0.0

        free = self.w > 0.0
        stuck = self.stuck

        # --- internal bending (explicit, small rigidity: stable) --------
        Fb = self._bending_forces()

        # --- tiny deterministic lateral perturbation ---------------------
        fp = np.zeros((n, 3))
        nf = int(free.sum())
        if nf:
            fp[free, 1] = self.rng.standard_normal(nf) * self.sigma
        if self.first_step:
            kick = np.zeros((n, 3))
            kick[free, 1] = 5.0e-5
            v += kick
            self.imp[0, _TERM_PERTURB] += (self.m[:, None] * kick).sum(axis=0)
            self.first_step = False

        # --- explicit integration of free nodes --------------------------
        acc = np.zeros((n, 3))
        acc[free] = self.g[None, :] + (Fb[free] + fp[free]) / self.m[free][:, None]
        v[free] += acc[free] * dt
        v[free] *= math.exp(-self.damp * dt)
        xprev = x.copy()
        x[free] += v[free] * dt
        x[stuck] += np.array([vb, 0.0, 0.0])
        x[stuck, 2] = self.r

        # --- contact bookkeeping (pre-solve penetration) ------------------
        inb = self._in_footprint(x)
        cfree = free & inb & (x[:, 2] < self.r + 1e-9)
        imp_contact = np.zeros(3)
        if cfree.any():
            pen = self.r - x[cfree, 2]
            imp_contact[2] = float((self.m[cfree] * pen).sum())
            self.last_contact_pos = x[cfree].mean(axis=0).copy()

        # --- constraint solve (position based, edge-parity Gauss-Seidel) --
        w = self.w
        fict = np.zeros((n, 3))
        m_edges = n - 1
        kk = np.arange(m_edges)
        for it in range(self.iters):
            cm = free & inb
            x[cm, 2] = np.maximum(x[cm, 2], self.r)
            sel = kk[(kk % 2) == (it % 2)]
            a = sel
            b = sel + 1
            d = x[b] - x[a]
            L = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
            dirv = d / L[:, None]
            diff = L - self.rest[sel]
            wa = w[a]
            wb = w[b]
            ws = wa + wb
            ok = ws > 0
            ca = np.zeros((len(sel), 3))
            cb = np.zeros((len(sel), 3))
            if ok.any():
                ca[ok] = (wa[ok] / ws[ok] * diff[ok])[:, None] * dirv[ok]
                cb[ok] = -(wb[ok] / ws[ok] * diff[ok])[:, None] * dirv[ok]
            x[a] += ca
            x[b] += cb
            # fictive (anchored-endpoint) corrections -> constraint reactions
            ma_ = self.m[a]
            mb_ = self.m[b]
            anch_a = (wa <= 0.0) & (wb > 0.0) & (ma_ > 0.0)
            if anch_a.any():
                wfa = 1.0 / ma_[anch_a]
                den = wfa + wb[anch_a]
                cfa = np.zeros((len(sel), 3))
                cfa[anch_a] = (wfa / den * diff[anch_a])[:, None] * dirv[anch_a]
                fict[a] += cfa
            anch_b = (wb <= 0.0) & (wa > 0.0) & (mb_ > 0.0)
            if anch_b.any():
                wfb = 1.0 / mb_[anch_b]
                den = wfb + wa[anch_b]
                cfb = np.zeros((len(sel), 3))
                cfb[anch_b] = -(wfb / den * diff[anch_b])[:, None] * dirv[anch_b]
                fict[b] += cfb
            # weak injector-channel alignment of the first free nodes
            if n > 2:
                x[1, 0:2] -= 0.002 * x[1, 0:2]
                x[2, 0:2] -= 0.001 * x[2, 0:2]

        # --- velocity update ---------------------------------------------
        v[free] = (x[free] - xprev[free]) / dt
        norms = np.linalg.norm(v[free], axis=1)
        if norms.size and norms.max() > 10.0:
            idx = np.where(free)[0]
            over = norms > 10.0
            v[idx[over]] *= (10.0 / norms[over])[:, None]

        # --- belt friction and sticking ------------------------------------
        cmask = free & inb & (x[:, 2] <= self.r + 1e-9)
        imp_fric = np.zeros(3)
        if cmask.any():
            rel = v[cmask, 0:2] - np.array([vb, 0.0])
            sp = np.linalg.norm(rel, axis=1)
            cap = self.mu * abs(float(self.g[2])) * dt
            scale = np.where(sp > 1e-12, np.minimum(1.0, cap / np.maximum(sp, 1e-12)), 0.0)
            dvv = -rel * scale[:, None]
            v[cmask, 0:2] += dvv
            imp_fric[0:2] = (self.m[cmask][:, None] * dvv).sum(axis=0)
            idx = np.where(cmask)[0]
            slow = sp < self.vtol
            self.timer[idx] = np.where(slow, self.timer[idx] + dt, 0.0)
            newly = idx[self.timer[idx] >= self.t_stick]
            if newly.size:
                self.stuck[newly] = True
                self.timer[newly] = 0.0
                x[newly, 2] = self.r
                self.w[newly] = 0.0
                v[newly] = (vb, 0.0, 0.0)

        # --- external-force budget accumulation ----------------------------
        stuck = self.stuck
        gdt = self.g * dt
        mst = float(self.m[stuck].sum()) if stuck.any() else 0.0
        self.imp[0, _TERM_GRAVITY] += float(self.m.sum()) * gdt
        self.imp[1, _TERM_GRAVITY] += mst * gdt
        self.imp[0, _TERM_CONTACT] += imp_contact
        self.imp[0, _TERM_FRICTION] += imp_fric
        pin = -(
            self.m[0] * gdt
            + Fb[0] * dt
            + fp[0] * dt
            + self.m[0] * fict[0] / dt
        )
        self.imp[0, _TERM_PIN] += pin
        if stuck.any():
            rod_on_stuck = Fb[stuck].sum(axis=0) * dt + (
                self.m[stuck][:, None] * fict[stuck]
            ).sum(axis=0) / dt
        else:
            rod_on_stuck = np.zeros(3)
        attach = -(mst * gdt + rod_on_stuck)
        self.imp[0, _TERM_ATTACH] += attach
        self.imp[1, _TERM_ATTACH] += attach
        self.imp[0, _TERM_PERTURB] += (fp * dt).sum(axis=0)

        # --- splice newly injected material at the outlet -------------------
        if self.descent >= self.h:
            newrest = self.descent
            self.x = np.insert(self.x, 0, self.outlet, axis=0)
            self.v = np.insert(self.v, 0, np.array([0.0, 0.0, -self.v_inj]), axis=0)
            self.rest = np.insert(self.rest, 0, newrest)
            self.stuck = np.insert(self.stuck, 0, False)
            self.timer = np.insert(self.timer, 0, 0.0)
            self.orig_dep = np.insert(self.orig_dep, 0, False)
            self.descent = 0.0
            self._refresh_mass()

        # --- remove material carried past the far end of the belt -----------
        while self.N > 8 and self.stuck[-1] and self.x[-1, 0] > self.x_delete:
            self.mass_removed += self.rhoA * float(self.rest[-1])
            self.x = self.x[:-1].copy()
            self.v = self.v[:-1].copy()
            self.rest = self.rest[:-1].copy()
            self.stuck = self.stuck[:-1].copy()
            self.timer = self.timer[:-1].copy()
            self.orig_dep = self.orig_dep[:-1].copy()
            self._refresh_mass()

        np.nan_to_num(self.x, copy=False)
        np.nan_to_num(self.v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------ export

    def advance_to(self, time: float):
        t = float(time)
        target = int(round(t / self.dt))
        while self.nsteps < target:
            self._step()
        return self._export(t)

    def _export(self, t: float) -> StateFrame:
        x = self.x.copy()
        n = self.N
        vb = self._belt_speed(max(t, 0.0))
        v = self.v.copy()
        self.v[self.stuck] = (vb, 0.0, 0.0)
        v[self.stuck] = (vb, 0.0, 0.0)
        v[0] = self.v[0]

        interval = t - self.last_export
        if interval <= 1e-12:
            interval = self.dt
        F = self.imp / interval
        stuck = self.stuck
        free = self.w > 0.0

        # gravity is exact and instantaneous
        F[0, _TERM_GRAVITY] = float(self.m.sum()) * self.g
        F[1, _TERM_GRAVITY] = (
            float(self.m[stuck].sum()) * self.g if stuck.any() else np.zeros(3)
        )

        mall = float(self.m.sum())
        com_all = (self.m[:, None] * x).sum(axis=0) / max(mall, 1e-12)
        if stuck.any():
            mst = float(self.m[stuck].sum())
            com_st = (self.m[stuck][:, None] * x[stuck]).sum(axis=0) / max(mst, 1e-12)
            mean_st = x[stuck].mean(axis=0)
        else:
            com_st = com_all.copy()
            mean_st = com_all.copy()
        mean_free = x[free].mean(axis=0) if free.any() else com_all

        pos = np.zeros((_DOMAINS, _TERMS, 3))
        for gi, base in enumerate((com_all, com_st)):
            for k in range(_TERMS):
                pos[gi, k] = base
        pos[:, _TERM_CONTACT] = self.last_contact_pos
        pos[:, _TERM_FRICTION] = self.last_contact_pos
        pos[:, _TERM_PIN] = self.outlet
        pos[:, _TERM_ATTACH] = mean_st
        pos[:, _TERM_PERTURB] = mean_free

        fields = {
            "budget_external_force": F,
            "budget_external_couple": np.zeros((_DOMAINS, _TERMS, 3)),
            "budget_force_position": pos,
            "constraint_value": np.array(
                [x[0, 0], x[0, 1], x[0, 2], vb], dtype=float
            ),
            "constraint_target": np.array(
                [self.outlet[0], self.outlet[1], self.outlet[2], self.v_inj],
                dtype=float,
            ),
            "cumulative_mass_source": np.array(
                [self.mass_added - self.mass_removed], dtype=float
            ),
        }

        arc = np.concatenate(([0.0], np.cumsum(self.rest))) if n > 1 else np.zeros(n)
        attributes = {
            "arc_coordinate": arc,
            "is_injector_controlled": (np.arange(n) == 0).astype(float),
            "is_initially_deposited": self.orig_dep.astype(float),
            "is_sticking_to_belt": self.stuck.astype(float),
            "rod_radius": np.full(n, self.radius),
            "natural_curvature": np.zeros(n),
            "mass_quantity": self.m.copy(),
        }

        ref_force = float(self.rhoA * self.L_avail * abs(float(self.g[2])))
        scalars = {
            "injection_speed": self.v_inj,
            "belt_speed": vb,
            "initial_belt_speed": self.belt_v0,
            "matched_belt_speed": self.belt_v1,
            "speed_switch_time": self.t_switch,
            "constraint_scale": float(abs(self.outlet[2])),
            "reference_force": ref_force,
            "reference_torque": ref_force * 0.033,
        }

        frame = StateFrame(
            time=t,
            position=x,
            velocity=v,
            orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1)),
            angular_velocity=np.zeros((n, 3)),
            mass=self.m.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        self.imp = np.zeros((_DOMAINS, _TERMS, 3))
        self.last_export = t
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "success": True,
            "solver": "discrete_elastic_rod_belt_meandering",
            "substeps": int(self.nsteps),
            "nodes": int(self.N),
            "stuck_nodes": int(self.stuck.sum()),
            "mass_injected_kg": float(self.mass_added),
            "mass_removed_kg": float(self.mass_removed),
            "fed_length_m": float(self.fed),
        }

