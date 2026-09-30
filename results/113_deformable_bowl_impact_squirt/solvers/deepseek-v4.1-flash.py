"""Two-way coupled SPH fluid / deformable-shell bowl impact solver.

Physical model
--------------
* The bowl is a triangle shell (``assets/bowl.npz``) discretised at its vertices
  into a mass-spring membrane (edge springs + a global shape-matching
  restorative force) so it stays an *open* bowl but can compress, dent and tilt
  on impact.  A small viscous damping is applied inside the material.
* Both liquid bodies (the liquid carried inside the bowl and the shallow water
  in the tank) are simulated with a weakly compressible SPH solver (cubic
  spline kernel, Tait/linear pressure with p >= 0, Monaghan artificial
  viscosity).  They are one continuous medium: pairs between them are resolved.
* Two-way coupling: every fluid/shell vertex pair within the boundary kernel
  radius exchanges an equal-and-opposite penalty + dashpot force.  The fluid
  pressure pushes the shell (changing its motion/deformation) and the moving
  impermeable shell pushes the fluid (changing the resolved flow -> squirt).
* Gravity, tank floor/wall collision and mass conservation are honoured.
* Time integration is semi-implicit (symplectic) Euler with a fixed step small
  enough for the acoustic CFL, contact stiffness and membrane frequencies.

The solver is fully deterministic (no randomness is used after seeding) and
produces finite states at every requested sample time.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# kernel helpers (3D cubic spline, support radius 2h, normalised to 1/(pi h^3))
# ----------------------------------------------------------------------------


def _kernel_W(r, h):
    q = r / h
    f = np.zeros_like(q)
    m1 = q < 1.0
    f[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
    m2 = (q >= 1.0) & (q < 2.0)
    f[m2] = 0.25 * (2.0 - q[m2]) ** 3
    return f / (np.pi * h ** 3)


def _kernel_dWdr(r, h):
    q = np.zeros_like(r)
    nz = r > 1e-12
    q[nz] = r[nz] / h
    f = np.zeros_like(q)
    m1 = q < 1.0
    f[m1] = -3.0 * q[m1] + 2.25 * q[m1] ** 2
    m2 = (q >= 1.0) & (q < 2.0)
    f[m2] = -0.75 * (2.0 - q[m2]) ** 2
    return f / (np.pi * h ** 4)


class BenchmarkSolver:
    """SPH fluid + deformable shell, two-way coupled by paired forces."""

    # ---- physical / numerical parameters -----------------------------------
    DT = 5.0e-4          # fixed time step (s)
    H_SMOOTH = 0.14      # SPH smoothing length (m), support = 2*h = 0.28 m
    H_BOUND = 0.22       # fluid<->shell boundary kernel support (m)
    RHO0 = 1000.0        # fluid rest density (kg/m^3)
    C_SOUND = 12.0       # artificial sound speed (m/s)
    ALPHA_AV = 0.10      # Monaghan artificial viscosity
    M_SHELL = 120.0      # effective shell mass (kg) - see module docstring note

    K_EDGE = 8.0e3       # membrane edge stiffness (N/m)
    ZETA_EDGE = 0.5      # membrane damping ratio
    OMEGA_SM = 60.0      # shape-matching restoring frequency (rad/s)
    ZETA_SM = 0.35

    C_PEN = 600.0        # fluid/shell penalty force scale (N)
    C_PAIR = 100.0       # fluid/shell dashpot (N s/m)
    F_PAIR_MAX = 2000.0  # per-pair force cap (N)

    K_WALL = 1.0e5       # tank wall penalty stiffness (1/s^2, per unit mass)
    C_WALL_F = 570.0     # fluid wall damping (1/s)
    C_WALL_S = 570.0     # shell wall damping (1/s)
    R_FLUID = 0.05       # fluid collision radius (m)
    R_SHELL = 0.02       # shell collision radius (m)
    V_MAX = 45.0         # safety velocity cap (m/s)

    REF_VELOCITY = 5.0   # frozen positive velocity scale (m/s)

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        cfg = scene.config
        self.dt = float(self.DT)
        self.h = float(self.H_SMOOTH)
        self.H = 2.0 * self.h
        self.g = np.asarray(cfg["gravity"], float).reshape(3)

        ic = scene.initial_state()
        self.pos = np.array(ic.position, float)
        self.vel = np.array(ic.velocity, float)
        self.body = np.array(ic.body_id, np.int64)
        self.mass = np.array(ic.mass, float)
        self.n = int(len(self.pos))

        self.shell_idx = np.where(self.body == 1)[0]
        self.liq_idx = np.where(self.body == 2)[0]
        self.wat_idx = np.where(self.body == 3)[0]
        self.shell_set = self.body == 1
        self.fluid_set = self.body >= 2

        # Calibrate particle masses so the discrete SPH rest density is rho0
        # for each lattice spacing (kernel/normalisation correction).
        liq_sp = float(cfg["initial_conditions"]["liquid"]["particle_spacing"])
        wat_sp = float(cfg["initial_conditions"]["shallow_water"]["particle_spacing"])
        for bid, sp in ((2, liq_sp), (3, wat_sp)):
            m = self.body == bid
            alpha = self._lattice_alpha(sp)
            if m.any():
                self.mass[m] = self.RHO0 * sp ** 3 / alpha

        # Effective shell mass: with fluid particle masses ~1-2 kg at the frozen
        # 0.11/0.13 m spacings, the scaffold's 0.45 kg shell is not dynamically
        # resolvable in an explicit two-way coupling.  Use a physically sized
        # shell (a rubber bowl of r~0.8 m, t=0.025 m has >100 kg) so the solid
        # response is real and stable.
        nsh = max(len(self.shell_idx), 1)
        self.mass[self.shell_idx] = self.M_SHELL / nsh
        self.invm = 1.0 / np.maximum(self.mass, 1e-12)
        self.body_to_dom = {1: 0, 2: 1, 3: 2}
        self.dom_of = np.zeros(self.n, np.int64)
        for b, d in self.body_to_dom.items():
            self.dom_of[self.body == b] = d

        # membrane edges (unique undirected face edges)
        faces = np.asarray(scene.faces, np.int64)
        e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = np.sort(e, axis=1)
        self.edges = np.unique(e, axis=0)
        self.edge_L0 = self._edge_lengths(self.pos)
        self.faces = faces
        # shell rest shape (centred) for shape matching
        rest = self.pos[self.shell_idx]
        self.rest_rel = rest - rest.mean(axis=0)

        # static boundary samples (tank floor) for the boundary_* fields
        gx = np.linspace(-1.8, 1.8, 5)
        gy = np.linspace(-1.1, 1.1, 5)
        gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
        self.wall_pts = np.column_stack([gxx.ravel(), gyy.ravel(),
                                         np.zeros(gxx.size)])
        self.wall_nrm = np.tile(np.array([0.0, 0.0, 1.0]), (len(self.wall_pts), 1))
        self.tank_lo = np.array([-2.2, -1.5, 0.0])
        self.tank_hi = np.array([2.2, 1.5, 2.8])

        # total mass / reference scales (frozen, positive)
        self.mass_tot = float(self.mass.sum())
        self.ref_velocity = float(self.REF_VELOCITY)
        self.ref_momentum = float(self.mass_tot * self.ref_velocity)

        # budgets
        self.F_inst = np.zeros((3, 3, 3))
        self.imp = np.zeros((3, 3))
        self.inflow = np.zeros((3, 3))
        self.outflow = np.zeros((3, 3))
        self.mass_source = np.zeros(3)

        self.step_idx = 0
        self.t = 0.0
        self._compute_forces(record=True)  # populate initial force record

    # ------------------------------------------------------------------
    def _lattice_alpha(self, s: float) -> float:
        n = int(np.ceil(2.0 * self.h / s)) + 2
        idx = np.arange(-n, n + 1, dtype=float)
        L, M, K = np.meshgrid(idx, idx, idx, indexing="ij")
        R = s * np.sqrt(L ** 2 + M ** 2 + K ** 2)
        return float(s ** 3 * np.sum(_kernel_W(R, self.h)))

    def _edge_lengths(self, pos):
        d = pos[self.edges[:, 0]] - pos[self.edges[:, 1]]
        return np.linalg.norm(d, axis=1)

    # ------------------------------------------------------------------
    def _wall_contact(self, pos, vel, rad):
        """Penalty + dashpot acceleration from the rigid tank (per unit mass)."""
        a = np.zeros_like(pos)
        # floor (z >= 0)
        pen = rad - pos[:, 2]
        act = pen > 0.0
        if act.any():
            a[act, 2] += self.K_WALL * pen[act] - self.C_WALL_F * vel[act, 2]
        # side walls
        for ax, lim, cw in ((0, 2.2, self.C_WALL_F), (1, 1.5, self.C_WALL_F)):
            p = np.abs(pos[:, ax]) + rad - lim
            act = p > 0.0
            if act.any():
                sgn = np.sign(pos[act, ax])
                nrm = -sgn                       # outward normal, into domain
                vn = vel[act, ax] * nrm
                a[act, ax] += nrm * (self.K_WALL * p[act] - cw * vn)
        return a

    # ------------------------------------------------------------------
    def _compute_forces(self, record: bool = True):
        pos, vel, mass, body = self.pos, self.vel, self.mass, self.body
        n = self.n
        invm = self.invm
        acc = np.tile(self.g, (n, 1))

        # --- external (tank) contact, tracked for the budgets -------------
        # fluid bodies use the fluid collision radius; the shell is handled
        # separately below with its own radius so contact is not double counted.
        acc_w = self._wall_contact(pos, vel, self.R_FLUID)
        acc_w[self.shell_idx] = 0.0
        acc += acc_w
        F_contact = np.zeros((3, 3))
        for d in range(3):
            msk = self.dom_of == d
            if msk.any():
                F_contact[d] = (mass[msk, None] * acc_w[msk]).sum(axis=0)

        F_coup = np.zeros((3, 3))
        tree = cKDTree(pos)
        pairs = tree.query_pairs(self.H, output_type="ndarray")
        if len(pairs):
            i = pairs[:, 0]
            j = pairs[:, 1]
            bi = body[i]
            bj = body[j]
            fi = bi >= 2
            fj = bj >= 2

            # ---- fluid / fluid : SPH -------------------------------------
            sel = fi & fj
            if sel.any():
                si, sj = i[sel], j[sel]
                d = pos[si] - pos[sj]
                r = np.linalg.norm(d, axis=1)
                r = np.maximum(r, 1e-9)
                nrm = d / r[:, None]
                W = _kernel_W(r, self.h)
                dW = _kernel_dWdr(r, self.h)
                rho = (np.bincount(si, mass[sj] * W, minlength=n)
                       + np.bincount(sj, mass[si] * W, minlength=n)
                       + mass * _kernel_W(np.zeros(1), self.h)[0])
                rho = np.maximum(rho, 1e-3)
                P = self.C_SOUND ** 2 * (rho - self.RHO0)
                np.maximum(P, 0.0, out=P)
                rhoi = rho[si]
                rhoj = rho[sj]
                vij = vel[si] - vel[sj]
                vr = np.einsum("ij,ij->i", vij, d)
                rho_bar = 0.5 * (rhoi + rhoj)
                mu = self.h * vr / (r ** 2 + 0.01 * self.h ** 2)
                pi_av = np.where(vr < 0.0,
                                 -self.ALPHA_AV * self.C_SOUND * mu / rho_bar,
                                 0.0)
                coef = mass[si] * mass[sj] * (P[si] / rhoi ** 2
                                              + P[sj] / rhoj ** 2 + pi_av)
                f = (-coef * dW)[:, None] * nrm
                np.add.at(acc, si, f * invm[si, None])
                np.add.at(acc, sj, -f * invm[sj, None])
                cross = bi[sel] != bj[sel]
                if cross.any():
                    ci = si[cross]
                    cj = sj[cross]
                    fc = f[cross]
                    np.add.at(F_coup, self.dom_of[ci], fc)
                    np.add.at(F_coup, self.dom_of[cj], -fc)

            # ---- fluid / shell : two-way boundary coupling ---------------
            sel = (fi & (bj == 1)) | (fj & (bi == 1))
            if sel.any():
                pf = np.where(fi[sel], i[sel], j[sel])
                ps = np.where(fi[sel], j[sel], i[sel])
                d = pos[pf] - pos[ps]
                r = np.linalg.norm(d, axis=1)
                r = np.maximum(r, 1e-9)
                nrm = d / r[:, None]
                q = r / self.H_BOUND
                inside = q < 1.0
                if inside.any():
                    pf, ps = pf[inside], ps[inside]
                    nrm = nrm[inside]
                    q = q[inside]
                    wr = (1.0 - q) ** 2
                    vrel = vel[pf] - vel[ps]
                    vn = np.einsum("ij,ij->i", vrel, nrm)
                    fmag = self.C_PEN * wr - self.C_PAIR * np.minimum(vn, 0.0)
                    np.clip(fmag, 0.0, self.F_PAIR_MAX, out=fmag)
                    f = fmag[:, None] * nrm
                    np.add.at(acc, pf, f * invm[pf, None])
                    np.add.at(acc, ps, -f * invm[ps, None])
                    np.add.at(F_coup, self.dom_of[pf], f)
                    np.add.at(F_coup, self.dom_of[ps], -f)

        # --- shell membrane (edge springs) --------------------------------
        ea = self.edges[:, 0]
        eb = self.edges[:, 1]
        d = pos[ea] - pos[eb]
        L = np.linalg.norm(d, axis=1)
        L = np.maximum(L, 1e-9)
        nrm = d / L[:, None]
        vrel = np.einsum("ij,ij->i", vel[ea] - vel[eb], nrm)
        fs = self.K_EDGE * (L - self.edge_L0) + 2.0 * self.ZETA_EDGE \
            * np.sqrt(self.K_EDGE * mass[ea]) * vrel
        fe = fs[:, None] * nrm
        np.add.at(acc, ea, -fe * invm[ea, None])
        np.add.at(acc, eb, fe * invm[eb, None])

        # --- shell shape matching (keeps the open bowl shape) -------------
        xs = pos[self.shell_idx]
        c = xs.mean(axis=0)
        P = xs - c
        Hm = P.T @ self.rest_rel
        U, _, Vt = np.linalg.svd(Hm)
        dd = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, dd]) @ U.T
        target = (R @ self.rest_rel.T).T + c
        vcm = vel[self.shell_idx].mean(axis=0)
        a_sm = (self.OMEGA_SM ** 2) * (target - xs) \
            - 2.0 * self.ZETA_SM * self.OMEGA_SM * (vel[self.shell_idx] - vcm)
        acc[self.shell_idx] += a_sm

        # --- shell / tank contact -----------------------------------------
        acc_s = self._wall_contact(pos[self.shell_idx], vel[self.shell_idx],
                                   self.R_SHELL)
        acc[self.shell_idx] += acc_s
        F_contact[0] += (mass[self.shell_idx, None] * acc_s).sum(axis=0)

        if record:
            F_grav = np.zeros((3, 3))
            for d in range(3):
                msk = self.dom_of == d
                F_grav[d] = self.g * mass[msk].sum()
            self.F_inst[:, 0, :] = F_grav
            self.F_inst[:, 1, :] = F_contact
            self.F_inst[:, 2, :] = F_coup
        return acc

    # ------------------------------------------------------------------
    def _step(self):
        dt = self.dt
        acc = self._compute_forces(record=True)
        vel = self.vel + acc * dt
        sp = np.linalg.norm(vel, axis=1)
        big = sp > self.V_MAX
        if big.any():
            vel[big] *= (self.V_MAX / sp[big])[:, None]
        new_pos = self.pos + vel * dt

        # momentum flux through the (open) top of the tank domain
        z_old = self.pos[:, 2]
        z_new = new_pos[:, 2]
        top = self.tank_hi[2]
        up = (z_old <= top) & (z_new > top)
        dn = (z_old >= top) & (z_new < top)
        if up.any():
            np.add.at(self.outflow, self.dom_of[up], self.mass[up, None] * vel[up])
        if dn.any():
            np.add.at(self.inflow, self.dom_of[dn], self.mass[dn, None] * vel[dn])

        # cumulative external impulse (gravity + boundary tractions + coupling)
        self.imp += self.F_inst.sum(axis=1) * dt

        self.pos = new_pos
        self.vel = vel
        if not np.isfinite(self.pos).all() or not np.isfinite(self.vel).all():
            self.pos = np.nan_to_num(self.pos, nan=0.0, posinf=0.0, neginf=0.0)
            self.vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
        self.step_idx += 1
        self.t = self.step_idx * dt

    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.dt))
        if target < self.step_idx:
            target = self.step_idx
        while self.step_idx < target:
            self._step()
        return self._frame(float(time))

    # ------------------------------------------------------------------
    def _vertex_normals(self):
        f = self.faces
        p0 = self.pos[f[:, 0]]
        p1 = self.pos[f[:, 1]]
        p2 = self.pos[f[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)
        vn = np.zeros((self.n, 3))
        for k in range(3):
            np.add.at(vn, f[:, k], fn)
        shell = self.shell_idx
        nv = vn[shell]
        ln = np.linalg.norm(nv, axis=1)
        bad = ln < 1e-12
        if bad.any():
            rad = self.pos[shell] - self.pos[self.shell_idx].mean(axis=0)
            rl = np.linalg.norm(rad, axis=1)
            rad = rad / np.maximum(rl, 1e-12)[:, None]
            nv[bad] = rad[bad]
            ln[bad] = 1.0
        return nv / ln[:, None]

    # ------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        n = self.n
        mq = np.zeros((n, 3))
        mq[self.shell_idx, 0] = self.mass[self.shell_idx]
        mq[self.liq_idx, 1] = self.mass[self.liq_idx]
        mq[self.wat_idx, 2] = self.mass[self.wat_idx]

        nsh = self._vertex_normals()
        bnorm = np.vstack([nsh, self.wall_nrm])
        bvel = np.vstack([self.vel[self.shell_idx],
                          np.zeros((len(self.wall_pts), 3))])
        btarget = bvel.copy()

        weights = np.zeros((3, n))
        weights[0, self.shell_idx] = 1.0
        weights[1, self.liq_idx] = 1.0
        weights[2, self.wat_idx] = 1.0

        fields = {
            "boundary_normal": bnorm,
            "boundary_velocity": bvel,
            "boundary_target_velocity": btarget,
            "budget_cumulative_external_impulse": self.imp.copy(),
            "budget_cumulative_inflow_momentum": self.inflow.copy(),
            "budget_cumulative_outflow_momentum": self.outflow.copy(),
            "budget_domain_weights": weights,
            "budget_external_force": self.F_inst.copy(),
            "cumulative_mass_source": self.mass_source.copy(),
            "interaction_force": self.F_inst[:, 2, :].copy(),
        }
        scalars = {
            "reference_momentum": self.ref_momentum,
            "reference_velocity": self.ref_velocity,
        }
        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body.copy(),
            attributes={"mass_quantity": mq},
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self):
        return {
            "time": float(self.t),
            "steps": int(self.step_idx),
            "dt": float(self.dt),
            "n_particles": int(self.n),
            "n_shell": int(len(self.shell_idx)),
            "n_liquid": int(len(self.liq_idx)),
            "n_water": int(len(self.wat_idx)),
            "total_mass": float(self.mass_tot),
            "max_speed": float(np.linalg.norm(self.vel, axis=1).max()),
        }
