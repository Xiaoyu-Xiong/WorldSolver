"""Physically-based particle solver for the soluble bunny / water scene.

Method (Lagrangian particles, unified-particle-system spirit):
  * Deformable soluble bunny: shape-matching elasticity (polar-decomposition
    clamped best-fit affine map) + gravity + fluid contact forces.
  * Water: particle pressure (near-pressure repulsion), pairwise viscosity,
    gravity, soft no-penetration floor, prescribed pipe inflow, open-side
    outflow removal, continuous re-emission between inflow_start/inflow_end.
  * Soluble phase evolution: contact-driven dissolution transfers red solute
    mass from bunny particles to touching water particles with capacity
    limiting (C <= 1) so total red solute mass is conserved exactly; pairwise
    diffusion mixes the solute; fully dissolved bunny particles convert to
    mixture fluid (label 3) carrying their residual solute.
All required StateFrame trajectory fields are exported each sample.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ---- numerical / physical parameters -------------------------------------
H = 0.18            # interaction radius (m)
DT = 0.005          # substep (s)
G = np.array([0.0, 0.0, -9.81])

KP1 = 80.0          # fluid pressure (linear)
KP2 = 250.0         # fluid near-pressure (quadratic)
VISC = 6.0          # pairwise viscosity coefficient (1/s)
DIFF = 1.5          # solute diffusion coefficient

KC_F = 120.0        # bunny -> fluid contact push (m/s^2 scale)
KC_B = 25.0         # fluid -> bunny contact push
KS = 250.0          # shape-matching spring stiffness
KD = 14.0           # shape-matching damping

KD_ISS = 0.10       # dissolution rate (1/s) per kg of contacting water
S_CONV = 0.05       # solid fraction below which a bunny particle becomes fluid

FLOOR_Z = 0.05
KB_FLOOR = 4000.0
KD_FLOOR = 60.0

VMAX = 10.0
SPAWN_RATE = 36.0   # inflow particles per second
MAX_PARTICLES = 9000

# removal (open sides / outflow) bounds
RM_XMIN, RM_XMAX = -1.02, 2.80
RM_YMAX = 1.60
RM_ZMIN = -0.05

# diagnostic grid
GS = 0.14
GX0, GX1 = -1.15, 2.55
GY0, GY1 = -1.45, 1.45
GZ0, GZ1 = 0.0, 1.75


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))
        cfg = scene.config
        ic = cfg.get("initial_conditions", {})
        self.inflow_start = float(ic.get("inflow_start_s", 0.0))
        self.inflow_end = float(ic.get("inflow_end_s", 5.5))
        self.inflow_speed = float(ic.get("inflow_speed_m_s", 2.4))

        frame = scene.initial_state()
        self.pos = np.array(frame.position, dtype=float)
        self.vel = np.array(frame.velocity, dtype=float)
        mass0 = np.array(frame.mass, dtype=float)
        self.label = np.array(frame.attributes["material_label"], dtype=int)
        self.rest = np.zeros_like(self.pos)          # bunny rest configuration
        bunny = self.label == 1
        self.rest[bunny] = self.pos[bunny]
        self.m0ref = mass0.copy()                    # reference (original) mass
        self.mass = mass0.copy()                     # current carrier mass
        # red solute mass (kg): bunny is pure solute, water starts clear
        self.msol = np.where(bunny, mass0, 0.0)
        self.src_gain = np.zeros(len(self.pos))      # per-particle solute gain accumulator
        fluid = self.label != 1
        self.m_fluid = float(np.median(mass0[fluid])) if fluid.any() else 1.157

        self.t = 0.0
        self.spawn_acc = 0.0
        self.cum_source = np.zeros(2)                # [solute, water] kg in - out
        self.injected_mass = 0.0
        self.removed_mass = 0.0
        self.initial_solute = float(self.msol.sum())
        self.initial_solid_mass = float(self.mass[bunny].sum())
        self.last_force = np.zeros((3, 3))
        self.A_fit = np.eye(3)

        # diagnostic grid
        self.nx = int(np.ceil((GX1 - GX0) / GS)) + 1
        self.ny = int(np.ceil((GY1 - GY0) / GS)) + 1
        self.nz = int(np.ceil((GZ1 - GZ0) / GS)) + 1
        self._frames_built = 0

    # ------------------------------------------------------------- helpers
    def _solid_fraction(self):
        s = np.zeros(len(self.pos))
        b = self.label == 1
        s[b] = self.mass[b] / self.m0ref[b]
        return s

    def _concentration(self):
        c = np.zeros(len(self.pos))
        b = self.label == 1
        f = ~b
        c[b] = self.mass[b] / self.m0ref[b]                       # = s
        c[f] = np.minimum(1.0, self.msol[f] / np.maximum(self.mass[f], 1e-9))
        return c

    def _append_particles(self, new_pos, new_vel):
        n = len(new_pos)
        self.pos = np.vstack((self.pos, new_pos))
        self.vel = np.vstack((self.vel, new_vel))
        self.label = np.concatenate((self.label, np.full(n, 2)))
        self.rest = np.vstack((self.rest, np.zeros((n, 3))))
        self.m0ref = np.concatenate((self.m0ref, np.full(n, self.m_fluid)))
        self.mass = np.concatenate((self.mass, np.full(n, self.m_fluid)))
        self.msol = np.concatenate((self.msol, np.zeros(n)))
        self.src_gain = np.concatenate((self.src_gain, np.zeros(n)))
        self.cum_source[1] += n * self.m_fluid
        self.injected_mass += n * self.m_fluid

    def _remove(self, keep):
        if keep.all():
            return
        gone = ~keep
        self.cum_source[0] -= float(self.msol[gone].sum())
        self.cum_source[1] -= float(self.mass[gone].sum())
        self.removed_mass += float(self.mass[gone].sum())
        for name in ("pos", "vel", "rest"):
            setattr(self, name, getattr(self, name)[keep])
        for name in ("label", "m0ref", "mass", "msol", "src_gain"):
            setattr(self, name, getattr(self, name)[keep])

    # ------------------------------------------------------------- substep
    def _substep(self, dt):
        n = len(self.pos)
        acc = np.tile(G, (n, 1))
        f_contact = np.zeros(3)   # fluid -> bunny total force
        f_floor = np.zeros(3)     # floor -> matter total force
        f_elastic = np.zeros(3)   # elastic restoring net force

        tree = cKDTree(self.pos)
        try:
            pairs = tree.query_pairs(H, output_mode="ndarray")
        except TypeError:
            pairs = np.array(sorted(tree.query_pairs(H)), dtype=int).reshape(-1, 2)
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            d = self.pos[i] - self.pos[j]
            r = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
            rhat = d / r[:, None]
            W = np.maximum(0.0, 1.0 - r / H)
            li, lj = self.label[i], self.label[j]
            ff = (li != 1) & (lj != 1)
            bf = (li == 1) ^ (lj == 1)

            # --- fluid-fluid: pressure + viscosity -----------------------
            if ff.any():
                ii, jj, ww, rh = i[ff], j[ff], W[ff], rhat[ff]
                mag = KP1 * ww + KP2 * ww * ww
                fa = mag[:, None] * rh
                np.add.at(acc, ii, fa)
                np.add.at(acc, jj, -fa)
                dv = VISC * ww[:, None] * (self.vel[jj] - self.vel[ii])
                np.add.at(acc, ii, dv)
                np.add.at(acc, jj, -dv)

            # --- bunny-fluid contact --------------------------------------
            if bf.any():
                ib, ifl = np.where(bf & (li == 1), i, j), np.where(bf & (li == 1), j, i)
                # rhat points j -> i; recompute bunny-relative direction
                dd = self.pos[ib] - self.pos[ifl]
                rr = np.maximum(np.linalg.norm(dd, axis=1), 1e-6)
                dh = dd / rr[:, None]
                wk = np.maximum(0.0, 1.0 - rr / H)
                mf = self.mass[ifl]
                mb = self.mass[ib]
                ab = (KC_B * wk * (mf / np.maximum(mb, 1e-6)))[:, None] * dh
                af = -(KC_F * wk)[:, None] * dh
                np.add.at(acc, ib, ab)
                np.add.at(acc, ifl, af)
                fpair = (KC_B * wk * mf)[:, None] * dh * 1.0  # force on bunny (N scale)
                f_contact = fpair.sum(axis=0)

            # --- solute diffusion on fluid-fluid pairs --------------------
            if ff.any():
                ii, jj, ww = i[ff], j[ff], W[ff]
                mi, mj = self.mass[ii], self.mass[jj]
                Ci = self.msol[ii] / np.maximum(mi, 1e-9)
                Cj = self.msol[jj] / np.maximum(mj, 1e-9)
                meff = mi * mj / np.maximum(mi + mj, 1e-9)
                dM = DIFF * ww * dt * (Cj - Ci) * meff
                hi = np.minimum(self.msol[jj], mi * np.maximum(0.0, 1.0 - Ci))
                hj = np.minimum(self.msol[ii], mj * np.maximum(0.0, 1.0 - Cj))
                dM = np.clip(dM, -hj, hi)
                np.add.at(self.msol, ii, dM)
                np.add.at(self.msol, jj, -dM)
                np.add.at(self.src_gain, ii, dM)
                np.add.at(self.src_gain, jj, -dM)

            # --- dissolution on bunny-fluid pairs --------------------------
            if bf.any():
                ib, ifl = np.where(bf & (li == 1), i, j), np.where(bf & (li == 1), j, i)
                dd = self.pos[ib] - self.pos[ifl]
                rr = np.maximum(np.linalg.norm(dd, axis=1), 1e-6)
                wk = np.maximum(0.0, 1.0 - rr / H)
                mf = self.mass[ifl]
                # contacting water mass per bunny particle
                w_b = np.zeros(n)
                np.add.at(w_b, ib, wk * mf)
                desired = np.minimum(self.mass[ib], KD_ISS * w_b[ib] * dt)
                Cf = self.msol[ifl] / np.maximum(mf, 1e-9)
                cap_pair = wk * mf * np.maximum(0.0, 1.0 - Cf)
                cap_b = np.zeros(n)
                np.add.at(cap_b, ib, cap_pair)
                scale = np.where(cap_b[ib] > 1e-12,
                                 np.minimum(1.0, desired / np.maximum(cap_b[ib], 1e-12)),
                                 0.0)
                dM = scale * cap_pair
                np.add.at(self.msol, ifl, dM)
                np.add.at(self.src_gain, ifl, dM)
                loss = np.zeros(n)
                np.add.at(loss, ib, dM)
                self.mass[ib] -= loss[ib]
                self.mass = np.maximum(self.mass, 0.0)

        # --- bunny elasticity (shape matching) ------------------------------
        bidx = np.nonzero(self.label == 1)[0]
        if len(bidx) > 8:
            x = self.pos[bidx]
            X = self.rest[bidx]
            c = x.mean(axis=0)
            C0 = X.mean(axis=0)
            dx = x - c
            dX = X - C0
            M2 = dX.T @ dX + 1e-6 * np.eye(3)
            A = (dx.T @ dX) @ np.linalg.inv(M2)
            U, S, Vt = np.linalg.svd(A)
            S = np.clip(S, 0.75, 1.30)
            A = U @ np.diag(S) @ Vt
            self.A_fit = A
            tgt = c + dX @ A.T
            a_sp = KS * (tgt - x) - KD * self.vel[bidx]
            acc[bidx] += a_sp
            f_elastic = (self.mass[bidx, None] * a_sp).sum(axis=0)
        else:
            self.A_fit = np.eye(3)

        # --- floor (no penetration) + friction ------------------------------
        low = self.pos[:, 2] < FLOOR_Z
        if low.any():
            pen = FLOOR_Z - self.pos[low, 2]
            az = KB_FLOOR * pen - KD_FLOOR * np.minimum(self.vel[low, 2], 0.0)
            acc[low, 2] += az
            near = self.pos[:, 2] < FLOOR_Z + 0.03
            acc[near, 0] -= 3.0 * self.vel[near, 0]
            acc[near, 1] -= 3.0 * self.vel[near, 1]
            mfloor = self.mass[low] * az
            f_floor = np.array([0.0, 0.0, float(mfloor.sum())])

        # --- integrate -------------------------------------------------------
        self.vel += acc * dt
        sp = np.linalg.norm(self.vel, axis=1)
        over = sp > VMAX
        if over.any():
            self.vel[over] *= (VMAX / sp[over])[:, None]
        self.pos += self.vel * dt

        # --- bunny remains pure solute: msol tracks current solid mass -------
        bmask = self.label == 1
        self.msol[bmask] = self.mass[bmask]

        # --- conversion of fully dissolved bunny particles -------------------
        s = self._solid_fraction()
        conv = (self.label == 1) & (s <= S_CONV)
        if conv.any():
            self.msol[conv] = self.mass[conv]        # residual solute dissolves
            self.mass[conv] = self.m_fluid
            self.m0ref[conv] = self.m_fluid
            self.label[conv] = 3

        # --- prescribed inflow ------------------------------------------------
        if self.inflow_start <= self.t < self.inflow_end and n < MAX_PARTICLES:
            self.spawn_acc += SPAWN_RATE * dt
            while self.spawn_acc >= 1.0 and len(self.pos) < MAX_PARTICLES:
                self.spawn_acc -= 1.0
                p = np.array([[-0.90,
                               self.rng.uniform(-0.16, 0.16),
                               self.rng.uniform(1.05, 1.40)]])
                v = np.array([[self.inflow_speed, 0.0, 0.0]])
                self._append_particles(p, v)

        # --- open-side outflow removal ---------------------------------------
        keep = ~(
            ((self.label != 1) & (
                (self.pos[:, 0] < RM_XMIN) | (self.pos[:, 0] > RM_XMAX)
                | (np.abs(self.pos[:, 1]) > RM_YMAX) | (self.pos[:, 2] < RM_ZMIN)))
        )
        self._remove(keep)

        self.last_force = np.vstack((f_contact, -f_contact + f_elastic * 0.0, f_floor))
        self._force_acc = getattr(self, "_force_acc", np.zeros((3, 3))) + self.last_force * dt

    # ------------------------------------------------------------- advance
    def advance_to(self, time: float):
        time = float(time)
        self._force_acc = np.zeros((3, 3))
        t_start = self.t
        while self.t < time - 1e-9:
            dt = min(DT, time - self.t)
            self._substep(dt)
            self.t += dt
        elapsed = max(time - t_start, DT)
        frame = self._build_frame(time, elapsed)
        self.src_gain[:] = 0.0
        self._frames_built += 1
        return frame

    # --------------------------------------------------------------- frame
    def _grid_cells(self):
        ix = np.clip(((self.pos[:, 0] - GX0) / GS).astype(int), 0, self.nx - 1)
        iy = np.clip(((self.pos[:, 1] - GY0) / GS).astype(int), 0, self.ny - 1)
        iz = np.clip(((self.pos[:, 2] - GZ0) / GS).astype(int), 0, self.nz - 1)
        return (ix * self.ny + iy) * self.nz + iz

    def _build_frame(self, time, elapsed):
        n = len(self.pos)
        bunny = self.label == 1
        total_mass = np.where(bunny, self.mass, self.mass + self.msol)
        conc = self._concentration()
        sfrac = self._solid_fraction()
        sfrac = np.where(bunny, sfrac, 0.0)
        density = np.where(bunny, 1180.0, 1000.0)

        F = np.zeros((n, 3, 3))
        F[:, np.arange(3), np.arange(3)] = 1.0
        F[bunny] = self.A_fit

        mq = np.zeros((n, 2))
        mq[:, 0] = np.where(bunny, self.mass, self.msol)   # red solute mass
        mq[:, 1] = np.where(bunny, 0.0, self.mass)         # water mass

        ncell = self.nx * self.ny * self.nz
        flat = self._grid_cells()
        msum = np.zeros(ncell)
        np.add.at(msum, flat, total_mass)
        vsum = np.zeros((ncell, 3))
        np.add.at(vsum, flat, total_mass[:, None] * self.vel)
        ssum = np.zeros(ncell)
        np.add.at(ssum, flat, self.msol)
        qsum = np.zeros(ncell)
        np.add.at(qsum, flat, self.src_gain)

        active = (msum > 1e-9).astype(float)
        safe = np.where(msum > 1e-9, msum, 1.0)
        vel_grid = (vsum / safe[:, None]).reshape(self.nx, self.ny, self.nz, 3)
        scalar = (ssum / safe).reshape(self.nx, self.ny, self.nz)
        source = (qsum / elapsed / safe).reshape(self.nx, self.ny, self.nz)
        mask3 = active.reshape(self.nx, self.ny, self.nz)

        dissolved = 1.0 - float(self.mass[bunny].sum()) / max(self.initial_solid_mass, 1e-9)
        solute_now = float(np.where(bunny, self.mass, self.msol).sum())

        frame = StateFrame(
            time,
            self.pos.copy(),
            self.vel.copy(),
            mass=total_mass,
            body_id=self.label.astype(int),
            attributes={
                "material_label": self.label.astype(float),
                "red_solute_concentration": conc,
                "solid_phase_fraction": sfrac,
                "density": density,
                "deformation_gradient": F,
                "mass_quantity": mq,
            },
            fields={
                "active_mask": mask3,
                "cumulative_mass_source": self.cum_source.copy(),
                "interaction_force": (self._force_acc / elapsed),
                "scalar_source": source,
                "transported_scalar": np.clip(scalar, 0.0, 1.0),
                "velocity_field": vel_grid,
            },
            scalars={
                "grid_spacing": GS,
                "dissolved_mass_fraction": dissolved,
                "total_red_solute_mass": solute_now,
                "initial_red_solute_mass": self.initial_solute,
                "injected_water_mass": self.injected_mass,
                "removed_mass": self.removed_mass,
                "particle_count": float(n),
            },
        )
        frame.validate()
        return frame

    # --------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        bunny = self.label == 1
        solute_now = float(np.where(bunny, self.mass, self.msol).sum())
        return {
            "final_time": self.t,
            "particle_count": len(self.pos),
            "initial_red_solute_mass": self.initial_solute,
            "final_red_solute_mass": solute_now,
            "solute_conservation_error": abs(solute_now - self.initial_solute)
            / max(self.initial_solute, 1e-9),
            "dissolved_mass_fraction": 1.0
            - float(self.mass[bunny].sum()) / max(self.initial_solid_mass, 1e-9),
            "injected_water_mass": self.injected_mass,
            "removed_mass": self.removed_mass,
        }
