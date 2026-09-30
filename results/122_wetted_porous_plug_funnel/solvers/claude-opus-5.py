"""Two-way coupled wet-softening porous plug / funnel solver.

Physics
-------
* Liquid: weakly compressible SPH (cubic-spline kernel, Tait equation of state,
  Monaghan artificial viscosity), symplectic-Euler time stepping.
* Porous plug: linear tetrahedral FEM on the frozen plug mesh with a compressible
  Neo-Hookean material and Kelvin-Voigt element damping, sub-cycled inside each
  fluid step.
* Wetting: liquid contact drives a Darcy/imbibition-type saturation ODE with
  permeability-scaled diffusion through the plug; the element Young modulus is
  interpolated between the dry and the wet (softened) value from saturation, so
  passage results from constitutive softening and deformation only.
* Coupling: plug surface markers embedded in the FEM (constant barycentric
  weights) act as Akinci boundary samples for the SPH pressure/viscous forces.
  Fluid forces on the markers are scattered to FEM nodes with the transposed
  weights, so the fluid/solid interaction is an exactly antisymmetric pair.
* Boundaries: the frozen axisymmetric funnel profile and the catch floor are
  analytic penalty contacts (normal spring + damper + regularised friction) for
  both liquid particles and plug markers.

All velocity updates use the forces that are simultaneously recorded in the
momentum ledger, so the exported per-domain cumulative external impulse matches
the momentum change of every budget domain by construction.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_EPS = 1.0e-12


def _kernel_w(q: np.ndarray, h: float) -> np.ndarray:
    """Cubic spline kernel value, support radius 2h."""
    sigma = 1.0 / (np.pi * h ** 3)
    w = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    w[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
    w[m2] = 0.25 * (2.0 - q[m2]) ** 3
    return sigma * w


def _kernel_dw(q: np.ndarray, h: float) -> np.ndarray:
    """dW/dr of the cubic spline kernel (negative on 0 < q < 2)."""
    sigma = 1.0 / (np.pi * h ** 4)
    d = np.zeros_like(q)
    m1 = (q > 0.0) & (q < 1.0)
    m2 = (q >= 1.0) & (q < 2.0)
    d[m1] = -3.0 * q[m1] + 2.25 * q[m1] ** 2
    d[m2] = -0.75 * (2.0 - q[m2]) ** 2
    return sigma * d


class BenchmarkSolver:
    """Explicit multiphysics solver for the wetted porous plug / funnel scene."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg
        self.rng = np.random.default_rng(self.seed)

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])
        self.spacing = float(cfg["particle_spacing"])

        frame0 = scene.initial_state()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int64).copy()
        pos0 = np.asarray(frame0.position, dtype=float).copy()
        vel0 = np.asarray(frame0.velocity, dtype=float).copy()
        mass0 = np.asarray(frame0.mass, dtype=float).copy()
        self.solid_idx = np.flatnonzero(self.body_id == 1)
        self.fluid_idx = np.flatnonzero(self.body_id == 2)
        self.n_total = pos0.shape[0]
        self.n_s = int(self.solid_idx.size)
        self.n_f = int(self.fluid_idx.size)

        self.phase_id = np.asarray(frame0.attributes["phase_id"], dtype=float).copy()
        self.porosity_attr = np.asarray(frame0.attributes["porosity"], dtype=float).copy()

        # --- solid state -------------------------------------------------
        self.x_s = pos0[self.solid_idx].copy()
        self.v_s = vel0[self.solid_idx].copy()
        self.X_s = self.x_s.copy()
        self.m_s = mass0[self.solid_idx].copy()

        plug = cfg["plug"]
        self.E_dry = float(plug["dry_young_modulus"])
        self.E_wet = self.E_dry * float(plug["wet_softening_factor"])
        self.poisson = 0.2
        self.permeability = float(plug["permeability"])
        self.porosity = float(plug["porosity"])
        self.cos_theta = float(np.cos(np.radians(float(plug["wetting_contact_angle_degrees"]))))
        self.saturation_s = np.full(self.n_s, float(plug["initial_saturation"]))

        # --- fluid state -------------------------------------------------
        self.x_f = pos0[self.fluid_idx].copy()
        self.v_f = vel0[self.fluid_idx].copy()
        self.m_f = mass0[self.fluid_idx].copy()
        liq = cfg["liquid"]
        self.rho0 = float(liq["density"])
        self.h = 1.3 * self.spacing
        self.support = 2.0 * self.h
        self.c_snd = 30.0
        self.alpha_visc = 0.18
        self.rho_f = np.full(self.n_f, self.rho0)

        # --- boundaries --------------------------------------------------
        funnel = cfg["funnel"]
        prof = None
        geom = cfg.get("scene_geometry")
        if isinstance(geom, dict):
            for obj in geom.get("objects", []):
                if obj.get("id") == "funnel":
                    ip = obj.get("geometry", {}).get("inner_profile")
                    if ip:
                        prof = np.asarray(ip, dtype=float)
                    break
        if prof is None:
            prof = np.array([[0.31, 0.86], [0.31, 1.02], [0.58, 1.48],
                             [0.88, 1.88], [1.10, 2.15]], dtype=float)
        self.prof_r = prof[:, 0].copy()
        self.prof_z = prof[:, 1].copy()
        dz = np.diff(self.prof_z)
        self.prof_slope = np.diff(self.prof_r) / np.maximum(dz, _EPS)
        self.z_bot = float(self.prof_z[0])
        self.z_top = float(self.prof_z[-1])
        self.throat_radius = float(funnel["throat_radius"])
        self.floor_z = 0.06
        self.floor_r = 1.35

        self.k_wall = 4.0e4              # N/m, liquid particle wall penalty
        self.k_press = 8.0e5             # Pa/m, plug contact pressure per unit overlap
        self.mu_fric = 0.05
        self.time = 0.0

        # --- solid discretisation (needs h/support/rho0 and funnel profile) --
        self._build_tets()
        self._build_markers()

        # --- time stepping ----------------------------------------------
        self.substeps_per_sample = 50
        self.dt = self.sample_dt / self.substeps_per_sample
        self.solid_substeps = 10
        self.dt_s = self.dt / self.solid_substeps
        self.v_cap = 40.0

        # --- ledgers -----------------------------------------------------
        self.n_domains = 3          # 0: plug, 1: water, 2: global
        self.n_terms = 4            # gravity, boundary traction, coupling, stabilisation
        self.cum_impulse = np.zeros((self.n_domains, 3))
        self.term_force = np.zeros((self.n_domains, self.n_terms, 3))
        self.coupling_force = np.zeros(3)      # fluid -> solid, latest step
        self.bnd_force_s = np.zeros(3)
        self.bnd_force_f = np.zeros(3)
        self.stab_force_s = np.zeros(3)
        self.stab_force_f = np.zeros(3)
        self.ext_force_particles = np.zeros((self.n_total, 3))
        self.steps_taken = 0

        weights = np.zeros((self.n_domains, self.n_total))
        weights[0, self.solid_idx] = 1.0
        weights[1, self.fluid_idx] = 1.0
        weights[2, :] = 1.0
        self.domain_weights = weights

        total_mass = float(self.m_s.sum() + self.m_f.sum())
        self.reference_momentum = float(total_mass * np.sqrt(2.0 * 9.81 * 1.0))
        self.mass_total = total_mass
        self._update_external_force_field()

    # ------------------------------------------------------------ mesh setup
    def _build_tets(self) -> None:
        """Tetrahedralise the 16-sided closed cylindrical plug mesh."""
        n_side = 16
        bot = np.arange(n_side)                 # ring at z = z_min
        top = np.arange(n_side) + n_side        # ring at z = z_max
        c_bot = 2 * n_side                      # bottom cap centre
        c_top = 2 * n_side + 1                  # top cap centre
        tets = []
        for i in range(n_side):
            j = (i + 1) % n_side
            b0, b1, t0, t1 = bot[i], bot[j], top[i], top[j]
            tets.append([b0, b1, c_bot, t0])
            tets.append([b1, c_bot, t0, t1])
            tets.append([c_bot, t0, t1, c_top])
        tets = np.asarray(tets, dtype=np.int64)

        X = self.X_s
        d1 = X[tets[:, 1]] - X[tets[:, 0]]
        d2 = X[tets[:, 2]] - X[tets[:, 0]]
        d3 = X[tets[:, 3]] - X[tets[:, 0]]
        vol6 = np.einsum("ij,ij->i", np.cross(d1, d2), d3)
        flip = vol6 < 0.0
        if np.any(flip):
            tets[flip] = tets[flip][:, [0, 2, 1, 3]]
        self.tets = tets

        Dm = np.stack((X[tets[:, 1]] - X[tets[:, 0]],
                       X[tets[:, 2]] - X[tets[:, 0]],
                       X[tets[:, 3]] - X[tets[:, 0]]), axis=2)
        self.Dm_inv = np.linalg.inv(Dm)
        self.rest_vol = np.abs(np.linalg.det(Dm)) / 6.0
        self.n_tet = tets.shape[0]

        # lumped nodal mass consistent with the frozen plug mass
        m_node = np.zeros(self.n_s)
        share = self.rest_vol / max(float(self.rest_vol.sum()), _EPS)
        m_total = float(self.m_s.sum())
        for k in range(4):
            np.add.at(m_node, tets[:, k], 0.25 * share * m_total)
        m_node = np.maximum(m_node, 1.0e-6)
        m_node *= m_total / float(m_node.sum())
        self.m_node = m_node
        self.m_s = m_node.copy()
        self.inv_m_node = 1.0 / m_node

        # graph Laplacian (edge list) for saturation diffusion through the plug
        edges = set()
        for a in range(4):
            for b in range(a + 1, 4):
                for e in zip(tets[:, a].tolist(), tets[:, b].tolist()):
                    edges.add((min(e), max(e)))
        edge_arr = np.asarray(sorted(edges), dtype=np.int64)
        self.edge_i = edge_arr[:, 0]
        self.edge_j = edge_arr[:, 1]
        el = np.linalg.norm(X[self.edge_i] - X[self.edge_j], axis=1)
        self.edge_len = np.maximum(el, 1.0e-6)

    def _build_markers(self) -> None:
        """Surface coupling/contact markers embedded with constant FEM weights."""
        n_side = 16
        rows = []
        areas = []
        X = self.X_s
        # lateral surface: bilinear samples on each side quad
        us = np.array([1.0 / 6.0, 0.5, 5.0 / 6.0])
        vs = (np.arange(8) + 0.5) / 8.0
        r_out = float(np.hypot(X[0, 0], X[0, 1]))
        h_plug = float(X[n_side, 2] - X[0, 2])
        side_area = 2.0 * np.pi * r_out * h_plug / (n_side * us.size * vs.size)
        for i in range(n_side):
            j = (i + 1) % n_side
            b0, b1, t0, t1 = i, j, i + n_side, j + n_side
            for u in us:
                for v in vs:
                    w = np.zeros(self.n_s)
                    w[b0] = (1.0 - u) * (1.0 - v)
                    w[b1] = u * (1.0 - v)
                    w[t0] = (1.0 - u) * v
                    w[t1] = u * v
                    rows.append(w)
                    areas.append(side_area)
        # caps: barycentric samples on each triangular sector
        bary = np.array([[0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6],
                         [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        cap_area = np.pi * r_out ** 2 / (n_side * bary.shape[0])
        for ring_off, centre in ((0, 2 * n_side), (n_side, 2 * n_side + 1)):
            for i in range(n_side):
                j = (i + 1) % n_side
                a, b, c = ring_off + i, ring_off + j, centre
                for bc in bary:
                    w = np.zeros(self.n_s)
                    w[a] += bc[0]
                    w[b] += bc[1]
                    w[c] += bc[2]
                    rows.append(w)
                    areas.append(cap_area)
        self.W = np.asarray(rows, dtype=float)          # [M, n_s]
        self.marker_area = np.asarray(areas, dtype=float)
        self.n_mark = self.W.shape[0]

        # Akinci-style boundary volume from marker self-density
        pm = self.W @ self.X_s
        tree = cKDTree(pm)
        pairs = tree.query_pairs(self.support, output_type="ndarray")
        dens = np.full(self.n_mark, _kernel_w(np.zeros(1), self.h)[0])
        if pairs.size:
            d = np.linalg.norm(pm[pairs[:, 0]] - pm[pairs[:, 1]], axis=1)
            w = _kernel_w(d / self.h, self.h)
            np.add.at(dens, pairs[:, 0], w)
            np.add.at(dens, pairs[:, 1], w)
        self.marker_vol = 1.0 / np.maximum(dens, _EPS)
        self.marker_psi = self.rho0 * self.marker_vol

        # frozen initial overlap: the given placement at the throat is the
        # stress-free contact reference, further squeeze is resisted
        pen0, _, _ = self._wall_penetration(pm)
        self.marker_rest_gap = np.maximum(pen0, 0.0)

        # per-marker contact stiffness/damping from the contact pressure scale
        self.marker_k = self.k_press * self.marker_area
        m_eff = float(self.m_s.sum()) / self.n_mark
        self.marker_c = 2.0 * np.sqrt(self.marker_k * m_eff)

    # ---------------------------------------------------------- funnel walls
    def _wall_radius(self, z: np.ndarray):
        """Inner funnel radius and radial slope dr/dz at heights z."""
        r = np.interp(z, self.prof_z, self.prof_r)
        idx = np.clip(np.searchsorted(self.prof_z, z, side="right") - 1,
                      0, self.prof_slope.size - 1)
        return r, self.prof_slope[idx]

    def _wall_penetration(self, pts: np.ndarray):
        """Outward penetration depth, inward unit normal and radial dir."""
        rxy = np.hypot(pts[:, 0], pts[:, 1])
        inv_r = 1.0 / np.maximum(rxy, _EPS)
        e_r = np.zeros_like(pts)
        e_r[:, 0] = pts[:, 0] * inv_r
        e_r[:, 1] = pts[:, 1] * inv_r
        r_w, slope = self._wall_radius(pts[:, 2])
        inside_band = (pts[:, 2] >= self.z_bot) & (pts[:, 2] <= self.z_top)
        radial_over = rxy - r_w
        denom = np.sqrt(1.0 + slope ** 2)
        pen = np.where(inside_band & (radial_over > 0.0), radial_over / denom, 0.0)
        # inward normal of the cone surface: -e_r + slope * e_z, normalised
        n = np.zeros_like(pts)
        n[:, 0] = -e_r[:, 0] / denom
        n[:, 1] = -e_r[:, 1] / denom
        n[:, 2] = slope / denom
        return pen, n, e_r

    def _contact_force(self, pts, vels, k_pen, c_pen, rest_gap=None):
        """Penalty normal + damping + regularised friction from funnel and floor.

        ``k_pen``/``c_pen`` are scalars or per-point arrays. ``rest_gap`` is the
        frozen initial overlap: contact is measured from the given initial
        configuration, so the frozen placement is stress free and any further
        squeeze of the plug against the wall is resisted.
        """
        force = np.zeros_like(pts)
        kk = np.broadcast_to(np.asarray(k_pen, dtype=float), (pts.shape[0],))
        cc = np.broadcast_to(np.asarray(c_pen, dtype=float), (pts.shape[0],))
        pen, nrm, _ = self._wall_penetration(pts)
        if rest_gap is not None:
            pen = np.maximum(pen - rest_gap, 0.0)
        act = pen > 0.0
        if np.any(act):
            p = pen[act]
            n = nrm[act]
            v = vels[act]
            vn = np.einsum("ij,ij->i", v, n)
            fn_mag = kk[act] * p - cc[act] * np.minimum(vn, 0.0)
            fn_mag = np.maximum(fn_mag, 0.0)
            fn = fn_mag[:, None] * n
            vt = v - vn[:, None] * n
            vt_n = np.linalg.norm(vt, axis=1)[:, None]
            ft = -self.mu_fric * fn_mag[:, None] * vt / np.sqrt(vt_n ** 2 + 0.04)
            force[act] += fn + ft
        # catch floor (top face at floor_z, finite radius)
        rxy = np.hypot(pts[:, 0], pts[:, 1])
        fpen = self.floor_z - pts[:, 2]
        fact = (fpen > 0.0) & (rxy <= self.floor_r)
        if np.any(fact):
            p = fpen[fact]
            vz = vels[fact, 2]
            fn_mag = kk[fact] * p - cc[fact] * np.minimum(vz, 0.0)
            fn_mag = np.maximum(fn_mag, 0.0)
            force[fact, 2] += fn_mag
            vt = vels[fact][:, :2]
            vt_n = np.linalg.norm(vt, axis=1)[:, None]
            force[fact, :2] += -self.mu_fric * fn_mag[:, None] * vt / np.sqrt(vt_n ** 2 + 0.04)
        return force

    # ------------------------------------------------------------- solid FEM
    def _fem_forces(self) -> np.ndarray:
        """Neo-Hookean + Kelvin-Voigt nodal forces (self-equilibrated per element)."""
        tets = self.tets
        x = self.x_s
        v = self.v_s
        Ds = np.stack((x[tets[:, 1]] - x[tets[:, 0]],
                       x[tets[:, 2]] - x[tets[:, 0]],
                       x[tets[:, 3]] - x[tets[:, 0]]), axis=2)
        F = Ds @ self.Dm_inv
        Dv = np.stack((v[tets[:, 1]] - v[tets[:, 0]],
                       v[tets[:, 2]] - v[tets[:, 0]],
                       v[tets[:, 3]] - v[tets[:, 0]]), axis=2)
        Fdot = Dv @ self.Dm_inv

        eye = np.broadcast_to(np.eye(3), F.shape)
        J = np.linalg.det(F)
        J = np.where(np.abs(J) < 0.05, np.sign(J + _EPS) * 0.05, J)
        Freg = F + 1.0e-7 * eye
        F_invT = np.linalg.inv(Freg).transpose(0, 2, 1)

        sat_e = self.saturation_s[tets].mean(axis=1)
        E = self.E_dry + (self.E_wet - self.E_dry) * np.clip(sat_e, 0.0, 1.0)
        mu = E / (2.0 * (1.0 + self.poisson))
        lam = E * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        logJ = np.log(np.abs(J))
        P = (mu[:, None, None] * (F - F_invT)
             + (lam * logJ)[:, None, None] * F_invT)
        # Kelvin-Voigt viscous stress (symmetric strain rate)
        eta = 30.0
        D = 0.5 * (Fdot + Fdot.transpose(0, 2, 1))
        P = P + eta * D

        H = -self.rest_vol[:, None, None] * (P @ self.Dm_inv.transpose(0, 2, 1))
        f1 = H[:, :, 0]
        f2 = H[:, :, 1]
        f3 = H[:, :, 2]
        f0 = -(f1 + f2 + f3)

        # element-wise stress limiter (scales the whole self-equilibrated set)
        mag = np.linalg.norm(f1, axis=1) + np.linalg.norm(f2, axis=1) + np.linalg.norm(f3, axis=1)
        cap = 6.0e3
        scale = np.where(mag > cap, cap / np.maximum(mag, _EPS), 1.0)[:, None]
        f0, f1, f2, f3 = f0 * scale, f1 * scale, f2 * scale, f3 * scale

        out = np.zeros_like(x)
        np.add.at(out, tets[:, 0], f0)
        np.add.at(out, tets[:, 1], f1)
        np.add.at(out, tets[:, 2], f2)
        np.add.at(out, tets[:, 3], f3)
        return out

    def _update_saturation(self, dt: float, tree_f: cKDTree) -> None:
        """Imbibition from liquid contact + permeability-scaled internal diffusion."""
        near = tree_f.query_ball_point(self.x_s, 1.2 * self.support, return_length=True)
        wet = (np.asarray(near, dtype=float) > 0.0).astype(float)
        rate = 0.9 * self.permeability * max(self.cos_theta, 0.05) * self.porosity
        s = self.saturation_s
        s = s + dt * rate * wet * (1.0 - s)
        # graph diffusion along mesh edges (conservative)
        diff = 0.04 * self.permeability
        flux = diff * (s[self.edge_i] - s[self.edge_j]) / self.edge_len ** 2
        ds = np.zeros_like(s)
        np.add.at(ds, self.edge_i, -flux)
        np.add.at(ds, self.edge_j, flux)
        s = np.clip(s + dt * ds, 0.0, 1.0)
        self.saturation_s = s

    # ------------------------------------------------------------ SPH fluid
    def _fluid_forces(self, marker_pos, marker_vel):
        """SPH pressure/viscous forces on liquid, and reaction on plug markers."""
        n_f = self.n_f
        x = self.x_f
        v = self.v_f
        tree_f = cKDTree(x)
        h = self.h

        dens = self.m_f * _kernel_w(np.zeros(1), h)[0]
        pairs = tree_f.query_pairs(self.support, output_type="ndarray")
        if pairs.size:
            i, j = pairs[:, 0], pairs[:, 1]
            dxv = x[i] - x[j]
            r = np.linalg.norm(dxv, axis=1)
            w = _kernel_w(r / h, h)
            np.add.at(dens, i, self.m_f[j] * w)
            np.add.at(dens, j, self.m_f[i] * w)
        else:
            i = j = np.zeros(0, dtype=np.int64)
            dxv = np.zeros((0, 3))
            r = np.zeros(0)

        tree_m = cKDTree(marker_pos)
        sdm = tree_f.sparse_distance_matrix(tree_m, self.support, output_type="ndarray")
        fi = sdm["i"].astype(np.int64)
        mi = sdm["j"].astype(np.int64)
        rm = sdm["v"]
        if fi.size:
            wm = _kernel_w(rm / h, h)
            np.add.at(dens, fi, self.marker_psi[mi] * wm)

        rho = np.maximum(dens, 0.35 * self.rho0)
        self.rho_f = rho
        p = (self.rho0 * self.c_snd ** 2 / 7.0) * ((rho / self.rho0) ** 7 - 1.0)
        p = np.maximum(p, 0.0)

        f_fluid = np.zeros((n_f, 3))
        # fluid-fluid pressure + Monaghan artificial viscosity
        if r.size:
            rsafe = np.maximum(r, 1.0e-9)
            dw = _kernel_dw(rsafe / h, h)
            grad = (dw / rsafe)[:, None] * dxv
            pr = (p[i] / rho[i] ** 2 + p[j] / rho[j] ** 2)
            fp = -(self.m_f[i] * self.m_f[j] * pr)[:, None] * grad
            dv = v[i] - v[j]
            vdotr = np.einsum("ij,ij->i", dv, dxv)
            mu_ij = np.zeros_like(r)
            appr = vdotr < 0.0
            if np.any(appr):
                rho_ij = 0.5 * (rho[i] + rho[j])
                mu_ij[appr] = (self.alpha_visc * h * self.c_snd
                               * vdotr[appr] / (rsafe[appr] ** 2 + 0.01 * h ** 2)
                               / rho_ij[appr])
            fv = (self.m_f[i] * self.m_f[j] * mu_ij)[:, None] * grad
            fpair = fp + fv
            np.add.at(f_fluid, i, fpair)
            np.add.at(f_fluid, j, -fpair)

        # fluid-marker coupling (Akinci boundary pressure + viscous drag + repulsion)
        f_mark = np.zeros((self.n_mark, 3))
        if fi.size:
            dxm = x[fi] - marker_pos[mi]
            rsafe = np.maximum(rm, 1.0e-9)
            dwm = _kernel_dw(rsafe / h, h)
            gradm = (dwm / rsafe)[:, None] * dxm
            pr = 2.0 * p[fi] / rho[fi] ** 2
            fp = -(self.m_f[fi] * self.marker_psi[mi] * pr)[:, None] * gradm
            dvm = v[fi] - marker_vel[mi]
            vdotr = np.einsum("ij,ij->i", dvm, dxm)
            mu_ij = np.zeros_like(rm)
            appr = vdotr < 0.0
            if np.any(appr):
                mu_ij[appr] = (2.0 * self.alpha_visc * h * self.c_snd
                               * vdotr[appr] / (rsafe[appr] ** 2 + 0.01 * h ** 2)
                               / rho[fi][appr])
            fv = (self.m_f[fi] * self.marker_psi[mi] * mu_ij)[:, None] * gradm
            # short range repulsion prevents liquid leaking through the plug
            d0 = 0.6 * self.spacing
            close = rm < d0
            frep = np.zeros_like(fp)
            if np.any(close):
                amp = 8.0e2 * (d0 - rm[close]) / d0
                frep[close] = amp[:, None] * dxm[close] / rsafe[close][:, None]
            fpair = fp + fv + frep
            np.add.at(f_fluid, fi, fpair)
            np.add.at(f_mark, mi, -fpair)
        return f_fluid, f_mark, tree_f

    # --------------------------------------------------------------- stepping
    def _step(self) -> None:
        dt = self.dt
        marker_pos = self.W @ self.x_s
        marker_vel = self.W @ self.v_s
        f_fluid, f_mark, tree_f = self._fluid_forces(marker_pos, marker_vel)

        # coupling force pair (internal to the global domain)
        f_coupling_on_solid = self.W.T @ f_mark
        self.coupling_force = f_mark.sum(axis=0)

        # ---- liquid update (gravity + SPH + analytic wall contact) --------
        g_f = self.m_f[:, None] * self.gravity
        c_wall = 2.0 * np.sqrt(self.k_wall * float(self.m_f[0]))
        f_bnd_f = self._contact_force(self.x_f, self.v_f, self.k_wall, c_wall)
        total_f = f_fluid + g_f + f_bnd_f
        dv_f = dt * total_f / self.m_f[:, None]
        self.v_f += dv_f
        speed = np.linalg.norm(self.v_f, axis=1)
        stab_f = np.zeros(3)
        hot = speed > self.v_cap
        if np.any(hot):
            v_old = self.v_f[hot]
            v_new = v_old * (self.v_cap / speed[hot])[:, None]
            stab_f = ((v_new - v_old) * self.m_f[hot][:, None]).sum(axis=0) / dt
            self.v_f[hot] = v_new
        self.x_f += dt * self.v_f

        # ---- solid sub-cycling (FEM + gravity + contact + coupling) -------
        acc_bnd_s = np.zeros(3)
        acc_stab_s = np.zeros(3)
        g_s = self.m_s[:, None] * self.gravity
        for _ in range(self.solid_substeps):
            f_fem = self._fem_forces()
            mp = self.W @ self.x_s
            mv = self.W @ self.v_s
            f_c_mark = self._contact_force(mp, mv, self.marker_k, self.marker_c,
                                           self.marker_rest_gap)
            np.clip(f_c_mark, -400.0, 400.0, out=f_c_mark)
            f_bnd_s = self.W.T @ f_c_mark
            acc_bnd_s += f_c_mark.sum(axis=0) * self.dt_s
            total_s = f_fem + g_s + f_bnd_s + f_coupling_on_solid
            self.v_s += self.dt_s * total_s * self.inv_m_node[:, None]
            sp = np.linalg.norm(self.v_s, axis=1)
            hot_s = sp > self.v_cap
            if np.any(hot_s):
                v_old = self.v_s[hot_s]
                v_new = v_old * (self.v_cap / sp[hot_s])[:, None]
                acc_stab_s += ((v_new - v_old) * self.m_s[hot_s][:, None]).sum(axis=0)
                self.v_s[hot_s] = v_new
            self.x_s += self.dt_s * self.v_s

        self.bnd_force_s = acc_bnd_s / dt
        self.bnd_force_f = f_bnd_f.sum(axis=0)
        self.stab_force_s = acc_stab_s / dt
        self.stab_force_f = stab_f

        # ---- wetting / softening -----------------------------------------
        self._update_saturation(dt, tree_f)

        # ---- momentum ledger ---------------------------------------------
        grav_s = g_s.sum(axis=0)
        grav_f = g_f.sum(axis=0)
        terms = np.zeros((self.n_domains, self.n_terms, 3))
        terms[0, 0] = grav_s
        terms[0, 1] = self.bnd_force_s
        terms[0, 2] = self.coupling_force
        terms[0, 3] = self.stab_force_s
        terms[1, 0] = grav_f
        terms[1, 1] = self.bnd_force_f
        terms[1, 2] = -self.coupling_force
        terms[1, 3] = self.stab_force_f
        terms[2, 0] = grav_s + grav_f
        terms[2, 1] = self.bnd_force_s + self.bnd_force_f
        terms[2, 2] = np.zeros(3)      # internal pair cancels globally
        terms[2, 3] = self.stab_force_s + self.stab_force_f
        self.term_force = terms
        self.cum_impulse += dt * terms.sum(axis=1)

        self.time += dt
        self.steps_taken += 1
        self._update_external_force_field()

    def _update_external_force_field(self) -> None:
        ext = np.zeros((self.n_total, 3))
        ext[self.solid_idx] = self.m_s[:, None] * self.gravity
        ext[self.fluid_idx] = self.m_f[:, None] * self.gravity
        self.ext_force_particles = ext

    # ----------------------------------------------------------------- frames
    def _deformation_gradients(self) -> np.ndarray:
        tets = self.tets
        x = self.x_s
        Ds = np.stack((x[tets[:, 1]] - x[tets[:, 0]],
                       x[tets[:, 2]] - x[tets[:, 0]],
                       x[tets[:, 3]] - x[tets[:, 0]]), axis=2)
        F = Ds @ self.Dm_inv
        acc = np.zeros((self.n_s, 3, 3))
        cnt = np.zeros(self.n_s)
        for k in range(4):
            np.add.at(acc, tets[:, k], F)
            np.add.at(cnt, tets[:, k], 1.0)
        Fn = acc / np.maximum(cnt, 1.0)[:, None, None]
        out = np.tile(np.eye(3), (self.n_total, 1, 1))
        out[self.solid_idx] = np.where(np.isfinite(Fn), Fn, np.eye(3))
        return out

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        while self.time < target - 0.5 * self.dt and guard < 10_000_000:
            self._step()
            guard += 1

        pos = np.zeros((self.n_total, 3))
        vel = np.zeros((self.n_total, 3))
        mass = np.zeros(self.n_total)
        pos[self.solid_idx] = self.x_s
        pos[self.fluid_idx] = self.x_f
        vel[self.solid_idx] = self.v_s
        vel[self.fluid_idx] = self.v_f
        mass[self.solid_idx] = self.m_s
        mass[self.fluid_idx] = self.m_f
        pos = np.nan_to_num(pos, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        vel = np.nan_to_num(vel, nan=0.0, posinf=self.v_cap, neginf=-self.v_cap)

        sat = np.zeros(self.n_total)
        sat[self.solid_idx] = np.clip(self.saturation_s, 0.0, 1.0)
        sat[self.fluid_idx] = 1.0

        mass_quantity = np.zeros((self.n_total, 2))
        mass_quantity[self.solid_idx, 0] = self.m_s
        mass_quantity[self.fluid_idx, 1] = self.m_f

        interaction = np.stack((self.coupling_force, -self.coupling_force), axis=0)

        frame = StateFrame(
            time=max(target, 0.0),
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=self.body_id,
            attributes={
                "phase_id": self.phase_id,
                "saturation": sat,
                "porosity": self.porosity_attr,
                "mass_quantity": mass_quantity,
                "deformation_gradient": self._deformation_gradients(),
            },
            fields={
                "external_force": self.ext_force_particles,
                "interaction_force": np.nan_to_num(interaction),
                "cumulative_mass_source": np.zeros(2),
                "budget_domain_weights": self.domain_weights,
                "budget_external_force": np.nan_to_num(self.term_force),
                "budget_cumulative_external_impulse": np.nan_to_num(self.cum_impulse),
                "budget_cumulative_inflow_momentum": np.zeros((self.n_domains, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((self.n_domains, 3)),
            },
            scalars={
                "reference_momentum": self.reference_momentum,
                "dry_young_modulus": self.E_dry,
                "wet_softening_factor": float(self.cfg["plug"]["wet_softening_factor"]),
                "permeability": self.permeability,
                "throat_radius": self.throat_radius,
                "mean_saturation": float(np.mean(self.saturation_s)),
                "plug_min_z": float(np.min(self.x_s[:, 2])),
                "characteristic_length": float(2.0 * self.throat_radius),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "wcsph_fluid_neohookean_fem_plug_wetting_softening",
            "steps": int(self.steps_taken),
            "dt": float(self.dt),
            "solid_substeps": int(self.solid_substeps),
            "n_fluid": int(self.n_f),
            "n_solid_nodes": int(self.n_s),
            "n_markers": int(self.n_mark),
            "final_mean_saturation": float(np.mean(self.saturation_s)),
            "final_plug_min_z": float(np.min(self.x_s[:, 2])),
        }
