"""Poro-elasto-capillary sponge absorption and swelling solver.

Self-contained physical model implemented with NumPy/SciPy only.

Physics
-------
* The sponge is a damped elastic network of its solid samples (surface shell +
  resolved porous interior), loaded by gravity, a floor contact and the
  hydrodynamic reaction of the adjacent free-liquid block.
* A liquid-saturation field lives on the resolved porous diagnostic grid
  (material / Lagrangian samples).  It advances by an explicit finite-volume
  Richards/Darcy law with capillary pressure ``p_c(S)`` and relative
  permeability ``k_r(S)``, including the dilation term ``S div(v_s)`` produced
  by the swelling solid velocity.
* Free water is a weakly compressible SPH liquid (Tait equation of state plus
  Monaghan artificial viscosity) with a penalty floor contact and a momentum
  conserving sponge/liquid boundary force.
* Coupling: capillary suction at the wetted sponge face pulls liquid out of the
  SPH particles (mass, momentum and volume conserving transfer); the absorbed
  liquid is stored, raises the local saturation and drives the swelling
  eigenstrain of the elastic network, which in turn pushes the free liquid.

All external/internal force bookkeeping is closed so that the reported
momentum budgets match the integrated state.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81])

# ---- fluid / SPH -----------------------------------------------------------
RHO_L = 1000.0            # frozen incompressible pore-liquid density [kg/m^3]
MU_L = 1.0e-3             # dynamic viscosity [Pa*s]
SOUND_SPEED = 16.0        # artificial sound speed (weakly compressible SPH)
SPH_H = 0.105             # SPH smoothing length [m]
KERNEL_SIGMA = 0.5028     # normalization of the support-h cubic spline kernel
AV_ALPHA = 0.10           # Monaghan artificial viscosity coefficient
LIST_RADIUS = 0.135       # neighbour-list search radius [m]

# ---- porous medium ---------------------------------------------------------
PHI0 = 0.9                # initial porosity (scene metadata)
K0 = 3.0e-8               # intrinsic permeability [m^2] (scene metadata)
P_ENTRY = 3000.0          # capillary entry pressure scale [Pa]
CAP_LAMBDA = 1.0          # capillary curve exponent
S_RES = 0.0               # residual saturation

# ---- sponge elasticity -----------------------------------------------------
E_SPONGE = 1.0e5          # Young modulus of the wet sponge skeleton [Pa]
ZETA_SPONGE = 0.35        # Rayleigh (viscoelastic) damping ratio
BETA_SWELL = 0.25         # free swelling eigenstrain at S = 1
FLOOR_Z = 0.0             # support top surface [m]

# ---- time stepping ---------------------------------------------------------
DT_OUTER = 8.0e-4         # fluid / coupling time step
N_SUB = 5                 # solid sub-steps per outer step
REBUILD_EVERY = 15        # neighbour list / tree refresh interval
REP_RANGE = 0.09          # sponge/liquid contact force range [m]
REP_STIFF = 100.0         # sponge/liquid contact stiffness [N]

# ---- budget bookkeeping ----------------------------------------------------
G_DOM = 3                 # global, water, sponge
K_TERM = 3                # gravity, support contact, sponge<->water coupling
MATERIALS = 2             # 0 = solid, 1 = liquid


def _spline(q):
    """Cubic-spline SPH kernel and its derivative (support radius = h)."""
    w = np.zeros_like(q)
    dw = np.zeros_like(q)
    m = q < 1.0
    qm = q[m]
    w[m] = 1.0 - 1.5 * qm * qm + 0.75 * qm * qm * qm
    dw[m] = -3.0 * qm + 2.25 * qm * qm
    return w, dw


class BenchmarkSolver:
    """Two-way coupled poro-elasto-capillary sponge imbibition solver."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.meta = dict(scene.metadata())
        frame = scene.initial_state()
        self.t = 0.0
        self.step_index = 0
        self.rebuild_counter = 0

        self.x = np.array(frame.position, dtype=np.float64)
        self.v = np.zeros_like(self.x)
        self.N = int(self.x.shape[0])
        mass = np.array(frame.mass, dtype=np.float64)
        self.body = np.array(frame.body_id, dtype=np.int64)
        self.phase = np.array(frame.attributes["phase"], dtype=np.float64)

        self.i_surf = np.where(self.body == 1)[0]
        self.i_water = np.where(self.body == 2)[0]
        self.i_pore = np.where(self.body == 3)[0]
        self.i_sponge = np.concatenate((self.i_surf, self.i_pore))
        self.n_surf = len(self.i_surf)
        self.n_water = len(self.i_water)
        self.n_pore = len(self.i_pore)

        # conserved material split: solid vs (absorbed) liquid
        self.m_solid = np.zeros(self.N)
        self.m_liquid = np.zeros(self.N)
        self.m_solid[self.i_sponge] = mass[self.i_sponge]
        self.m_liquid[self.i_water] = mass[self.i_water]
        self.floor_w = 0.35 * self.m_liquid[self.i_water].copy()

        # ---- porous diagnostic grid (material samples, fixed layout) --------
        self.pore_lo = np.array([-1.22, -0.62, 0.03])
        self.pore_hi = np.array([0.22, 0.62, 1.27])
        self.spacing = 0.06
        counts = [
            int(np.ceil((self.pore_hi[d] - self.pore_lo[d] + 0.25 * self.spacing - 1e-9)
                        / self.spacing))
            for d in range(3)
        ]
        self.grid_shape = (counts[0], counts[1], counts[2])
        flat = np.arange(self.n_pore)
        self.gidx = np.empty(self.grid_shape, dtype=np.int64)
        if counts[0] * counts[1] * counts[2] == self.n_pore:
            cell = np.rint((self.x[self.i_pore] - self.pore_lo) / self.spacing).astype(np.int64)
            cell[:, 0] = np.clip(cell[:, 0], 0, counts[0] - 1)
            cell[:, 1] = np.clip(cell[:, 1], 0, counts[1] - 1)
            cell[:, 2] = np.clip(cell[:, 2], 0, counts[2] - 1)
            lin = np.ravel_multi_index((cell[:, 0], cell[:, 1], cell[:, 2]), self.grid_shape)
            if len(np.unique(lin)) == self.n_pore:
                self.gidx.flat[lin] = self.i_pore
            else:
                self.gidx = self.i_pore.reshape(self.grid_shape)
        else:
            raise RuntimeError("Unexpected porous-interior sample count")

        axes = [
            self.pore_lo[d] + self.spacing * np.arange(self.grid_shape[d])
            for d in range(3)
        ]
        self.cell_center = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

        # surface nodes -> nearest diagnostic cell (material mapping)
        sc = np.rint((self.x[self.i_surf] - self.pore_lo) / self.spacing).astype(np.int64)
        for d in range(3):
            sc[:, d] = np.clip(sc[:, d], 0, self.grid_shape[d] - 1)
        self.surf_cell = sc
        pc_ = np.rint((self.x[self.i_pore] - self.pore_lo) / self.spacing).astype(np.int64)
        self.pore_cell = (np.clip(pc_[:, 0], 0, counts[0] - 1),
                          np.clip(pc_[:, 1], 0, counts[1] - 1),
                          np.clip(pc_[:, 2], 0, counts[2] - 1))

        self.x0_pore = self.x[self.gidx].copy()
        self.x0_surf = self.x[self.i_surf].copy()

        # ---- saturation / porosity state ------------------------------------
        self.S = np.zeros(self.grid_shape, dtype=np.float64)
        self.phi = np.full(self.grid_shape, PHI0, dtype=np.float64)
        self.vol_source = np.zeros(self.grid_shape, dtype=np.float64)
        self.pore_mass_in = np.zeros(self.n_pore, dtype=np.float64)
        self.H_water = 1.11

        # ---- sponge elastic network -----------------------------------------
        xsp = self.x[self.i_sponge]
        self.x0_sp = xsp.copy()
        surf_local = np.arange(self.n_surf)
        pore_local = np.arange(self.n_surf, self.n_surf + self.n_pore)
        pairs_i = []
        pairs_j = []
        t_surf = cKDTree(xsp[surf_local])
        t_pore = cKDTree(xsp[pore_local])
        for a, b in t_surf.query_pairs(0.12):
            pairs_i.append(surf_local[a])
            pairs_j.append(surf_local[b])
        for a, b in t_pore.query_pairs(0.066):
            pairs_i.append(pore_local[a])
            pairs_j.append(pore_local[b])
        for a, near in enumerate(t_surf.query_ball_tree(t_pore, 0.10)):
            for b in near:
                pairs_i.append(surf_local[a])
                pairs_j.append(pore_local[b])
        self.ei = np.asarray(pairs_i, dtype=np.int64)
        self.ej = np.asarray(pairs_j, dtype=np.int64)
        d0 = xsp[self.ei] - xsp[self.ej]
        self.l0 = np.linalg.norm(d0, axis=1)
        keep = self.l0 > 1e-6
        self.ei, self.ej, self.l0 = self.ei[keep], self.ej[keep], self.l0[keep]
        self.k_edge = E_SPONGE * self.l0
        m_sp = self.m_solid[self.i_sponge]
        m_red = 2.0 * m_sp[self.ei] * m_sp[self.ej] / np.maximum(
            m_sp[self.ei] + m_sp[self.ej], 1e-12)
        self.c_edge = 2.0 * ZETA_SPONGE * np.sqrt(self.k_edge * m_red)
        omega_max = float(np.max(np.sqrt(self.k_edge / np.maximum(m_red, 1e-12))))
        self.omega_max = omega_max
        # nodes of the sponge base resting on (and adhering to) the support
        self.anchor = np.where(self.x0_sp[:, 2] < 0.02)[0]
        self._anchor_imp = np.zeros(3)
        self.dt_solid = min(1.6e-4, 0.9 / max(omega_max, 1.0))
        self.n_sub = max(N_SUB, int(np.ceil(DT_OUTER / max(self.dt_solid, 1e-9))))
        self.dt_solid = DT_OUTER / self.n_sub

        # ---- mass-transfer (supply) coupling geometry ------------------------
        self.inlet = (self.grid_shape[0] - 1,
                      np.arange(self.grid_shape[1])[:, None],
                      np.arange(self.grid_shape[2])[None, :])
        self.inlet_center = self.cell_center[self.inlet[0], self.inlet[1], self.inlet[2]]
        self.inlet_center = self.inlet_center.reshape(-1, 3)
        self.n_inlet = self.inlet_center.shape[0]
        self.inlet_ij = np.stack(
            np.meshgrid(np.arange(self.grid_shape[1]), np.arange(self.grid_shape[2]),
                        indexing="ij"), axis=-1).reshape(-1, 2)
        self.supply_radius = 0.13
        self.supply_cell = np.zeros(0, dtype=np.int64)
        self.supply_part = np.zeros(0, dtype=np.int64)

        # ---- neighbour structures (filled by _rebuild) -----------------------
        self.wpair_i = np.zeros(0, dtype=np.int64)
        self.wpair_j = np.zeros(0, dtype=np.int64)
        self.rep_pairs = np.zeros((0, 2), dtype=np.int64)
        self.wtree = cKDTree(self.x[self.i_water])
        self._rebuild(force=True)

        # frozen reference masses used by the Lagrangian SPH density ratio
        self.mw0 = self.m_liquid[self.i_water].copy()
        self.mw0 = np.maximum(self.mw0, 1e-9)

        # ---- budget accumulators --------------------------------------------
        self.impulse = np.zeros((G_DOM, 3))
        self.inflow = np.zeros((G_DOM, 3))
        self.outflow = np.zeros((G_DOM, 3))
        self.cum_mass_source = np.zeros(MATERIALS)

        # ---- reference scales ------------------------------------------------
        m_tot = float(self.m_solid.sum() + self.m_liquid.sum())
        self.ref_momentum = float(max(1.0, m_tot * np.sqrt(9.81 * 1.3)))
        self.ref_darcy = float(max(1.0e-6, 0.05))
        self.ref_storage = 1.0
        self._ext_force = np.zeros((G_DOM, K_TERM, 3))
        self._interaction = np.zeros((K_TERM, 3))
        self._last_inlet_flux = np.zeros(self.n_inlet)
        self._flux_max = 0.0
        self._rho_w = np.full(self.n_water, RHO_L)
        self._pres_w = np.zeros(self.n_water)
        self._swell_state()

    # ------------------------------------------------------------- neighbours
    def _rebuild(self, force: bool = False) -> None:
        xw = self.x[self.i_water]
        self.wtree = cKDTree(xw)
        pi, pj = self.wtree.query_pairs(LIST_RADIUS, output_type="ndarray")
        self.wpair_i = np.ascontiguousarray(pi[:, 0])
        self.wpair_j = np.ascontiguousarray(pi[:, 1])

        ts = cKDTree(self.x[self.i_sponge])
        near_s = ts.query_ball_tree(self.wtree, REP_RANGE)
        a = []
        b = []
        for k, near in enumerate(near_s):
            for w in near:
                a.append(k)
                b.append(w)
        if a:
            self.rep_pairs = np.stack(
                [np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)], axis=1
            )
        else:
            self.rep_pairs = np.zeros((0, 2), dtype=np.int64)

        ti = cKDTree(self.inlet_center)
        near_c = ti.query_ball_tree(self.wtree, self.supply_radius)
        c = []
        p = []
        for k, near in enumerate(near_c):
            for w in near:
                c.append(k)
                p.append(w)
        if c:
            self.supply_cell = np.asarray(c, dtype=np.int64)
            self.supply_part = np.asarray(p, dtype=np.int64)
        else:
            self.supply_cell = np.zeros(0, dtype=np.int64)
            self.supply_part = np.zeros(0, dtype=np.int64)

        if len(xw):
            self.H_water = float(np.clip(np.quantile(xw[:, 2], 0.97), 0.25, 1.25))

    # ------------------------------------------------------------- pore laws
    def _pc(self, s):
        """Capillary (matric) pressure of the pore liquid, zero at S = 1."""
        se = np.clip(s, 0.0, 1.0)
        return -P_ENTRY * (1.0 - se)

    def _pore_fields(self):
        """Capillary pressure, relative permeability and their face fluxes."""
        s = self.S
        pc = self._pc(s)
        kr = np.clip(s, 0.0, 1.0) ** 3
        kfac = K0 / MU_L
        dx = self.spacing
        qx = -kfac * (0.5 * (kr[:-1] + kr[1:])) * (pc[1:] - pc[:-1]) / dx
        qy = -kfac * (0.5 * (kr[:, :-1] + kr[:, 1:])) * (pc[:, 1:] - pc[:, :-1]) / dx
        qz = -kfac * (0.5 * (kr[:, :, :-1] + kr[:, :, 1:])) * (pc[:, :, 1:] - pc[:, :, :-1]) / dx
        return pc, kr, qx, qy, qz

    # ------------------------------------------------------------ SPH forces
    def _water_forces(self, mw):
        xw = self.x[self.i_water]
        vw = self.v[self.i_water]
        i = self.wpair_i
        j = self.wpair_j
        nw = self.n_water
        dx = xw[i] - xw[j]
        r2 = np.einsum("ij,ij->i", dx, dx)
        r = np.sqrt(np.maximum(r2, 1e-12))
        w, dw = _spline(r / SPH_H)
        kern = KERNEL_SIGMA / (SPH_H ** 3) * w
        dkern = KERNEL_SIGMA / (SPH_H ** 4) * dw
        rho = (np.bincount(i, weights=mw[j] * kern, minlength=nw)
               + np.bincount(j, weights=mw[i] * kern, minlength=nw))
        # Lagrangian reference density: same packing, original masses.  This
        # makes the pressure a pure function of the configuration change, so
        # absorbed mass does not create spurious suction.
        rho0 = (np.bincount(i, weights=self.mw0[j] * kern, minlength=nw)
                + np.bincount(j, weights=self.mw0[i] * kern, minlength=nw))
        rho = np.maximum(rho, 1e-9)
        rho0 = np.maximum(rho0, 1e-9)
        bcoef = RHO_L * SOUND_SPEED ** 2 / 7.0
        pres = np.maximum(0.0, bcoef * ((rho / rho0) ** 7 - 1.0))
        pterm = pres[i] / rho[i] ** 2 + pres[j] / rho[j] ** 2
        vdotr = (vw[i] - vw[j])[:, 0] * dx[:, 0] + (vw[i] - vw[j])[:, 1] * dx[:, 1] \
            + (vw[i] - vw[j])[:, 2] * dx[:, 2]
        mu_ij = SPH_H * vdotr / (r2 + 0.01 * SPH_H ** 2)
        pvisc = np.where(vdotr < 0.0,
                         -AV_ALPHA * SOUND_SPEED * mu_ij / (0.5 * (rho[i] + rho[j])),
                         0.0)
        c = (pterm + pvisc) * dkern / r
        vec = c[:, None] * dx
        acc = np.empty((nw, 3))
        for d in range(3):
            acc[:, d] = (np.bincount(i, weights=-mw[j] * vec[:, d], minlength=nw)
                         - np.bincount(j, weights=-mw[i] * vec[:, d], minlength=nw))
        return acc, rho, pres

    def _repulsion(self):
        """Momentum-conserving sponge boundary force on the free liquid."""
        nsp = self.n_surf + self.n_pore
        if len(self.rep_pairs) == 0:
            return np.zeros((self.n_water, 3)), np.zeros((nsp, 3))
        si = self.rep_pairs[:, 0]
        wi = self.rep_pairs[:, 1]
        xs = self.x[self.i_sponge][si]
        xw = self.x[self.i_water][wi]
        dx = xw - xs
        r = np.sqrt(np.einsum("ij,ij->i", dx, dx) + 1e-12)
        n = dx / r[:, None]
        ratio = np.clip(1.0 - r / REP_RANGE, 0.0, 1.0)
        dv = self.v[self.i_water][wi] - self.v[self.i_sponge][si]
        vn = np.einsum("ij,ij->i", dv, n)
        fmag = np.maximum(0.0, REP_STIFF * ratio ** 2 - 40.0 * vn)
        f = fmag[:, None] * n
        fw = np.zeros((self.n_water, 3))
        fs = np.zeros((nsp, 3))
        for d in range(3):
            fw[:, d] = np.bincount(wi, weights=f[:, d], minlength=self.n_water)
            fs[:, d] = -np.bincount(si, weights=f[:, d], minlength=nsp)
        return fw, fs

    # ------------------------------------------------------------- contacts
    def _water_floor(self):
        zw = self.x[self.i_water, 2]
        vz = self.v[self.i_water, 2]
        pen = np.maximum(0.0, 0.5 * self.spacing - zw)
        fn = 5.0e4 * pen - 300.0 * np.minimum(vz, 0.0)
        f = np.zeros((self.n_water, 3))
        f[:, 2] = fn
        slip = np.maximum(0.0, 1.0 - pen / (0.5 * self.spacing + 1e-12))
        vt = self.v[self.i_water, :2]
        ft = -50.0 * vt * slip[:, None]
        cap = 0.5 * fn
        mag = np.linalg.norm(ft, axis=1)
        scale = np.where(mag > cap, cap / np.maximum(mag, 1e-12), 1.0)
        f[:, :2] = ft * scale[:, None]
        return f

    def _sponge_floor(self):
        zs = self.x[self.i_sponge, 2]
        vz = self.v[self.i_sponge, 2]
        pen = np.maximum(0.0, -zs)
        fn = 3000.0 * pen - 0.5 * np.minimum(vz, 0.0)
        fn = np.maximum(fn, 0.0)
        f = np.zeros((self.n_surf + self.n_pore, 3))
        f[:, 2] = fn
        slip = np.maximum(0.0, 1.0 - pen / (0.5 * self.spacing + 1e-12))
        vt = self.v[self.i_sponge, :2]
        ft = -30.0 * vt * slip[:, None]
        cap = 0.5 * fn
        mag = np.linalg.norm(ft, axis=1)
        scale = np.where(mag > cap, cap / np.maximum(mag, 1e-12), 1.0)
        f[:, :2] = ft * scale[:, None]
        return f

    # ------------------------------------------------------------- one step
    def _step(self, dt: float) -> None:
        self.step_index += 1
        self.rebuild_counter += 1
        if self.rebuild_counter >= REBUILD_EVERY:
            self._rebuild()
            self.rebuild_counter = 0

        iw = self.i_water
        isp = self.i_sponge
        mw = self.m_liquid[iw]
        m_sp = (self.m_solid[isp] + self.m_liquid[isp])
        m_sp = np.maximum(m_sp, 1e-9)
        self._anchor_imp[:] = 0.0

        # --- free liquid internal forces
        a_sph, rho_w, pres_w = self._water_forces(mw)
        f_rep_w, f_rep_s = self._repulsion()
        f_floor_w = self._water_floor()

        # --- external force inventory (per domain and per term)
        f_grav_w = mw[:, None] * GRAVITY[None, :]
        f_grav_s = m_sp[:, None] * GRAVITY[None, :]

        acc_w = a_sph + GRAVITY[None, :] + (f_rep_w + f_floor_w) / mw[:, None]
        self.v[iw] += acc_w * dt
        self.x[iw] += self.v[iw] * dt

        # --- sponge sub-cycled elastic integration (frozen coupling force)
        s_avg = 0.5 * (self.S_edge[:, 0] + self.S_edge[:, 1])
        l0_eff = self.l0 * (1.0 + BETA_SWELL * s_avg)
        xsp = self.x[isp]
        vsp = self.v[isp]
        f_spring = np.zeros((len(isp), 3))
        f_floorsum = np.zeros((len(isp), 3))
        dt_s = self.dt_solid
        for _ in range(self.n_sub):
            dxe = xsp[self.ei] - xsp[self.ej]
            r = np.sqrt(np.einsum("ij,ij->i", dxe, dxe) + 1e-18)
            n = dxe / r[:, None]
            dv = vsp[self.ei] - vsp[self.ej]
            vn = np.einsum("ij,ij->i", dv, n)
            fsc = -(self.k_edge * (r - l0_eff) + self.c_edge * vn)
            fe = fsc[:, None] * n
            f_spring[:] = 0.0
            for d in range(3):
                f_spring[:, d] = (np.bincount(self.ei, weights=fe[:, d], minlength=len(isp))
                                  - np.bincount(self.ej, weights=fe[:, d], minlength=len(isp)))
            # floor contact for the sponge nodes (recomputed per sub-step)
            zs = xsp[:, 2]
            vz = vsp[:, 2]
            pen = np.maximum(0.0, -zs)
            fnz = np.maximum(3000.0 * pen - 0.5 * np.minimum(vz, 0.0), 0.0)
            slip = np.maximum(0.0, 1.0 - pen / (0.5 * self.spacing + 1e-12))
            ft = -30.0 * vsp[:, :2] * slip[:, None]
            mag = np.linalg.norm(ft, axis=1)
            cap = 0.5 * fnz
            sc = np.where(mag > cap, cap / np.maximum(mag, 1e-12), 1.0)
            f_floorsum[:, :2] = ft * sc[:, None]
            f_floorsum[:, 2] = fnz
            ftot = f_spring + f_floorsum + f_grav_s + f_rep_s
            vsp += (ftot / m_sp[:, None]) * dt_s
            vsp[:] = np.clip(vsp, -60.0, 60.0)
            if self.anchor.size:
                # no-slip support contact: the sponge base sticks to the floor
                vt = vsp[self.anchor, :2]
                self._anchor_imp[:2] -= (m_sp[self.anchor, None] * vt).sum(axis=0)
                vsp[self.anchor, :2] = 0.0
            xsp += vsp * dt_s
            self.x[isp] = xsp
            self.v[isp] = vsp
            self.impulse[2] += (f_grav_s.sum(axis=0) + f_floorsum.sum(axis=0)
                                + f_rep_s.sum(axis=0)) * dt_s
            self.impulse[0] += (f_grav_s.sum(axis=0) + f_floorsum.sum(axis=0)) * dt_s
            self._ext_force[2, 0] = f_grav_s.sum(axis=0)
            self._ext_force[2, 1] = f_floorsum.sum(axis=0)
            self._ext_force[2, 2] = f_rep_s.sum(axis=0)

        self.impulse[2] += self._anchor_imp
        self.impulse[0] += self._anchor_imp
        self._ext_force[2, 1] = self._ext_force[2, 1] + self._anchor_imp / dt

        self.impulse[1] += (f_grav_w.sum(axis=0) + f_floor_w.sum(axis=0)
                            + f_rep_w.sum(axis=0)) * dt
        self.impulse[0] += (f_grav_w.sum(axis=0) + f_floor_w.sum(axis=0)) * dt
        self._ext_force[1, 0] = f_grav_w.sum(axis=0)
        self._ext_force[1, 1] = f_floor_w.sum(axis=0)
        self._ext_force[1, 2] = f_rep_w.sum(axis=0)
        self._ext_force[0, 0] = f_grav_w.sum(axis=0) + f_grav_s.sum(axis=0)
        self._ext_force[0, 1] = f_floor_w.sum(axis=0) + f_floorsum.sum(axis=0)
        self._ext_force[0, 2] = 0.0
        self._interaction[0] = f_rep_w.sum(axis=0)
        self._interaction[1] = f_rep_s.sum(axis=0)
        self._interaction[2] = 0.0
        self._rho_w = rho_w
        self._pres_w = pres_w

        # --- pore-liquid transport and sponge/liquid mass exchange
        self._transfer(dt)
        self._update_pore(dt)
        self._swell_state()

    # --------------------------------------------------- pore transport
    def _update_pore(self, dt: float) -> None:
        s = self.S
        phi = self.phi
        dx = self.spacing
        pc, kr, qx, qy, qz = self._pore_fields()
        divq = np.zeros_like(s)
        divq[:-1] += qx / dx
        divq[1:] -= qx / dx
        divq[:, :-1] += qy / dx
        divq[:, 1:] -= qy / dx
        divq[:, :, :-1] += qz / dx
        divq[:, :, 1:] -= qz / dx

        # solid velocity divergence (dilation from swelling / elasticity)
        vs = self.v[self.gidx].reshape(self.grid_shape[0], self.grid_shape[1],
                                       self.grid_shape[2], 3)
        div_vs = np.zeros_like(s)
        for d in range(3):
            g = np.gradient(vs[..., d], dx, axis=d, edge_order=1)
            div_vs += g

        rhs = -divq - s * div_vs + self.vol_source
        s = s + dt * rhs / np.maximum(phi, 1e-3)
        self.S = np.clip(s, 0.0, 1.0)

        # porosity update from the volumetric deformation of the skeleton
        u = self.x[self.gidx] - self.x0_pore
        eps_v = np.zeros_like(s)
        for d in range(3):
            eps_v += np.gradient(u[..., d], dx, axis=d, edge_order=1)
        jac = np.clip(1.0 + eps_v, 0.2, 5.0)
        self.phi = np.clip(1.0 - (1.0 - PHI0) / jac, 0.05, 0.99)

    # --------------------------------------------------- mass transfer
    def _transfer(self, dt: float) -> None:
        self.vol_source[:] = 0.0
        if self.supply_part.size == 0:
            return
        s = self.S
        pc, kr, _, _, _ = self._pore_fields()
        dx = self.spacing
        area = dx * dx
        kr_in = 0.5 * (kr[-1] + 1.0)
        qreq = (K0 / MU_L) * kr_in * (-pc[-1]) / dx          # requested inlet Darcy flux
        req_all = RHO_L * qreq * area * dt                   # kg per inlet cell per step
        part = self.supply_part
        cell = self.supply_cell
        mwl = self.m_liquid[self.i_water]
        avail_full = np.maximum(mwl - self.floor_w, 0.0)
        avail = avail_full[part]
        csum = np.bincount(cell, weights=avail, minlength=self.n_inlet)
        req = req_all.reshape(-1)
        tot_req = float(req.sum())
        tot_avail = float(avail_full.sum())
        scale = min(1.0, tot_avail / max(tot_req, 1e-30))
        share = req[cell] / np.maximum(csum[cell], 1e-30)
        raw = share * avail * scale
        dm_p = np.bincount(part, weights=raw, minlength=self.n_water)
        over = dm_p > avail_full
        if np.any(over):
            fac = np.where(over, avail_full / np.maximum(dm_p, 1e-30), 1.0)
            raw = raw * fac[part]
            dm_p = np.bincount(part, weights=raw, minlength=self.n_water)
        dm_cell = np.bincount(cell, weights=raw, minlength=self.n_inlet)
        vw = self.v[self.i_water]
        mom = np.zeros((self.n_inlet, 3))
        for d in range(3):
            mom[:, d] = np.bincount(cell, weights=raw * vw[part, d], minlength=self.n_inlet)
        v_cell = mom / np.maximum(dm_cell, 1e-30)[:, None]
        v_cell[dm_cell <= 0.0] = 0.0

        # take liquid out of the free surface block (mass and momentum)
        mwl -= dm_p
        self.m_liquid[self.i_water] = mwl
        self.outflow[1] += (dm_cell[:, None] * v_cell).sum(axis=0)
        self.inflow[2] += (dm_cell[:, None] * v_cell).sum(axis=0)

        # store it on the inlet pore nodes (momentum conserving merge)
        recv = self.gidx[-1].reshape(-1)
        add = dm_cell
        m_old = self.m_solid[recv] + self.m_liquid[recv]
        p_old = m_old[:, None] * self.v[recv]
        m_new = m_old + add
        self.m_liquid[recv] = self.m_liquid[recv] + add
        self.v[recv] = (p_old + add[:, None] * v_cell) / np.maximum(m_new, 1e-12)[:, None]
        loc = ((self.grid_shape[0] - 1) * self.grid_shape[1] * self.grid_shape[2]
               + np.arange(self.n_inlet))
        self.pore_mass_in[loc] += add

        # volume source feeding the saturation equation
        qin_act = dm_cell / np.maximum(RHO_L * area * dt, 1e-30)
        src = np.zeros(self.grid_shape)
        src[-1] = qin_act.reshape(self.grid_shape[1], self.grid_shape[2]) / dx
        self.vol_source = src
        self._last_inlet_flux = qin_act

    # --------------------------------------------------- node saturation map
    def _swell_state(self) -> None:
        s = self.S
        s_surf = s[self.surf_cell[:, 0], self.surf_cell[:, 1], self.surf_cell[:, 2]]
        s_pore = s[self.pore_cell]
        self.S_node = np.concatenate((s_surf, s_pore))
        self.S_edge = np.stack((self.S_node[self.ei], self.S_node[self.ej]), axis=1)
        self._s_surf = s_surf
        self._s_pore = s_pore

    # ------------------------------------------------------------- frames
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t < target - 1e-12 and guard < 200000:
            dt = min(DT_OUTER, target - self.t)
            self._step(dt)
            self.t += dt
            guard += 1
        self.t = target
        return self._frame(target)

    def _frame(self, t: float) -> StateFrame:
        n = self.N
        s = self.S
        nx, ny, nz = self.grid_shape
        s_surf = s[self.surf_cell[:, 0], self.surf_cell[:, 1], self.surf_cell[:, 2]]
        s_pore = s[self.pore_cell]
        pc, kr, qx, qy, qz = self._pore_fields()

        zc = self.x[self.gidx][..., 2]
        press = pc + RHO_L * np.maximum(self.H_water - zc, 0.0) * 9.81

        qc = np.zeros((nx, ny, nz, 3))
        qc[:-1, :, :, 0] += 0.5 * qx
        qc[1:, :, :, 0] += 0.5 * qx
        qc[:, :-1, :, 1] += 0.5 * qy
        qc[:, 1:, :, 1] += 0.5 * qy
        qc[:, :, :-1, 2] += 0.5 * qz
        qc[:, :, 1:, 2] += 0.5 * qz
        qin = self._last_inlet_flux.reshape(ny, nz)
        qc[-1, :, :, 0] += 0.5 * qin
        self._flux_max = float(np.max(np.abs(qc)))

        perm = np.zeros((nx, ny, nz, 3, 3))
        for d in range(3):
            perm[..., d, d] = K0 * kr
        vs = self.v[self.gidx].reshape(nx, ny, nz, 3)
        mask = np.ones((nx, ny, nz))

        # ---- per-sample state
        sat = np.zeros(n)
        sat[self.i_water] = 1.0
        sat[self.i_surf] = s_surf
        sat[self.i_pore] = s_pore
        poro = np.zeros(n)
        poro[self.i_surf] = self.phi[self.surf_cell[:, 0], self.surf_cell[:, 1],
                                    self.surf_cell[:, 2]]
        poro[self.i_pore] = self.phi[self.pore_cell]
        ppress = np.zeros(n)
        ppress[self.i_surf] = press[self.surf_cell[:, 0], self.surf_cell[:, 1],
                                    self.surf_cell[:, 2]]
        ppress[self.i_pore] = press[self.pore_cell]
        dflux = np.zeros((n, 3))
        dflux[self.i_surf] = qc[self.surf_cell[:, 0], self.surf_cell[:, 1],
                                self.surf_cell[:, 2]]
        dflux[self.i_pore] = qc[self.pore_cell]
        svol = np.zeros((n, 3))
        svol[self.i_sponge] = self.v[self.i_sponge]

        mass_q = np.stack((self.m_solid, self.m_liquid), axis=1)
        total_mass = self.m_solid + self.m_liquid

        weights = np.zeros((G_DOM, n))
        weights[0] = 1.0
        weights[1, self.i_water] = 1.0
        weights[2, self.i_sponge] = 1.0

        fields = {
            "pore_active_mask": mask,
            "pore_permeability": perm,
            "pore_porosity": self.phi.copy(),
            "pore_pressure": press,
            "pore_relative_flux": qc,
            "pore_saturation": s.copy(),
            "pore_solid_velocity": vs,
            "pore_volume_source": self.vol_source.copy(),
            "budget_domain_weights": weights,
            "budget_external_force": self._ext_force.copy(),
            "budget_cumulative_external_impulse": self.impulse.copy(),
            "budget_cumulative_inflow_momentum": self.inflow.copy(),
            "budget_cumulative_outflow_momentum": self.outflow.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "gravity_acceleration": GRAVITY[None, :].copy(),
            "interaction_force": self._interaction.copy(),
        }
        attributes = {
            "mass_quantity": mass_q,
            "saturation": sat,
            "phase": self.phase.copy(),
            "porosity": poro,
            "pore_pressure": ppress,
            "darcy_flux": dflux,
            "solid_velocity": svol,
        }
        scalars = {
            "dynamic_viscosity": MU_L,
            "grid_spacing": float(self.spacing),
            "pore_fluid_density": RHO_L,
            "reference_darcy_flux": float(self.ref_darcy),
            "reference_momentum": float(self.ref_momentum),
            "reference_pore_storage_rate": float(self.ref_storage),
            "time": float(t),
        }
        return StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=total_mass,
            body_id=self.body,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict:
        s = self.S
        prof = s.mean(axis=(1, 2))
        wet = np.where(prof > 0.10)[0]
        front_x = float(self.pore_lo[0] + self.spacing * wet.min()) if wet.size else None
        return {
            "steps": int(self.step_index),
            "time": float(self.t),
            "dt_outer": float(DT_OUTER),
            "dt_solid": float(self.dt_solid),
            "n_sub": int(self.n_sub),
            "omega_max": float(self.omega_max),
            "n_edges": int(len(self.ei)),
            "n_water": int(self.n_water),
            "n_pore": int(self.n_pore),
            "n_surface": int(self.n_surf),
            "absorbed_liquid_mass": float(self.m_liquid[self.i_pore].sum()),
            "remaining_water_mass": float(self.m_liquid[self.i_water].sum()),
            "wet_front_x": front_x,
            "max_saturation": float(s.max()),
            "max_water_speed": float(np.linalg.norm(self.v[self.i_water], axis=1).max()),
            "max_solid_speed": float(np.linalg.norm(self.v[self.i_sponge], axis=1).max()),
            "max_darcy_flux": float(self._flux_max),
            "max_pore_reynolds": float(RHO_L * self._flux_max * np.sqrt(K0) / MU_L),
        }
