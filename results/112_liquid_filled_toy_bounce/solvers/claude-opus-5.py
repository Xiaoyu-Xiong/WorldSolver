"""Two-way coupled liquid-filled deformable toy bounce solver (self-implemented).

Physics implemented from scratch with NumPy/SciPy only:

* Deformable closed shell: lumped-mass triangulated membrane with structural,
  shear and bending springs (linear elastic plus pair viscous damping) and a
  polytropic enclosed-gas pressure acting on the closed surface. The wall is
  impermeable and is the physical boundary seen by the liquid.
* Enclosed liquid: weakly compressible SPH (cubic-spline kernel, linearised Cole
  equation of state with free-surface clamping, Monaghan artificial viscosity).
* Two-way coupling: the fluid sees the shell as a moving penalty wall built from
  barycentric surface samples of the deforming shell triangles. Every wall force
  on a fluid particle is scattered back, exactly equal and opposite, onto the
  shell vertices through the same barycentric weights.
* Floor: penalty normal contact with damping and Coulomb-limited friction against
  the static floor slab of the frozen geometry.
* Markers: massless passive tracers advected by the Shepard-interpolated SPH
  velocity field and kept inside the impermeable wall geometrically.

Internal force pairs cancel exactly (scatter-add of antisymmetric pair forces),
so the reported momentum budgets close against gravity plus floor contact.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


def _cubic_spline_kernel(r: np.ndarray, h: float):
    """3D cubic B-spline kernel W and dW/dr (support 2h)."""
    sigma = 1.0 / (np.pi * h ** 3)
    q = r / h
    w = np.zeros_like(q)
    dw = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    q1 = q[m1]
    w[m1] = 1.0 - 1.5 * q1 ** 2 + 0.75 * q1 ** 3
    dw[m1] = (-3.0 * q1 + 2.25 * q1 ** 2) / h
    q2 = q[m2]
    t = 2.0 - q2
    w[m2] = 0.25 * t ** 3
    dw[m2] = -0.75 * t ** 2 / h
    return sigma * w, sigma * dw


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _scatter_pairs(idx_i: np.ndarray, idx_j: np.ndarray, f: np.ndarray, n: int) -> np.ndarray:
    """Accumulate antisymmetric pair forces: +f on i, -f on j (momentum conserving)."""
    out = np.zeros((n, 3))
    for c in range(3):
        out[:, c] = (np.bincount(idx_i, weights=f[:, c], minlength=n)
                     - np.bincount(idx_j, weights=f[:, c], minlength=n))
    return out


class BenchmarkSolver:
    """Self-implemented SPH-liquid / mass-spring-shell two-way coupled solver."""

    N_SUB = 50                 # substeps per sample interval
    GAMMA_GAS = 1.4            # polytropic exponent of the trapped gas
    P_GAS_REF = 450.0          # Pa, gauge stiffness scale of the enclosed gas
    K_STRUCT = 900.0           # N/m membrane (hoop / meridian) springs
    K_SHEAR = 350.0            # N/m diagonal shear springs
    K_BEND = 120.0             # N/m second-neighbour bending springs
    K_SPOKE = 700.0            # N/m pole spokes
    ZETA_SPRING = 0.08         # spring damping ratio
    K_FLOOR = 4000.0           # N/m floor penalty stiffness
    ZETA_FLOOR = 0.35
    MU_FLOOR = 0.40
    K_WALL = 1200.0            # N/m impermeable-wall penalty stiffness
    WALL_BAND = 0.05           # m penalty band thickness inside the wall
    WALL_CN = 0.60             # N*s/m wall normal damping
    WALL_CT = 0.25             # N*s/m wall tangential (viscous no-slip) drag
    WALL_KNN = 3
    SPH_H = 0.105              # m smoothing length (support 2h)
    SPH_C = 30.0               # m/s numerical sound speed
    SPH_ALPHA = 0.10           # Monaghan artificial viscosity
    V_CLAMP = 25.0             # m/s numerical safety speed limit

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.cfg = cfg
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.dt = self.sample_dt / float(self.N_SUB)

        self.z_floor = 0.0
        for obj in (cfg.get("scene_geometry", {}) or {}).get("objects", []) or []:
            if obj.get("id") == "floor":
                he = (obj.get("geometry", {}) or {}).get("half_extents", [0.0, 0.0, 0.0])
                tr = (obj.get("transform", {}) or {}).get("translation", [0.0, 0.0, 0.0])
                self.z_floor = float(tr[2]) + float(he[2])

        frame0 = scene.initial_state()
        pos0 = np.array(frame0.position, dtype=float)
        vel0 = np.array(frame0.velocity, dtype=float)
        mass0 = np.array(frame0.mass, dtype=float)
        bid = np.array(frame0.body_id, dtype=np.int32)
        self.body_id = bid
        self.n_all = len(pos0)
        self.i_shell = np.flatnonzero(bid == 1)
        self.i_fluid = np.flatnonzero(bid == 2)
        self.i_mark = np.flatnonzero((bid == 3) | (bid == 4))

        self.sp = pos0[self.i_shell].copy()
        self.sv = vel0[self.i_shell].copy()
        self.sm = mass0[self.i_shell].copy()
        self.fp = pos0[self.i_fluid].copy()
        self.fv = vel0[self.i_fluid].copy()
        self.fm = mass0[self.i_fluid].copy()
        self.mp = pos0[self.i_mark].copy()
        self.mv = vel0[self.i_mark].copy()
        self.mass_all = mass0.copy()
        self.m_shell = float(self.sm.sum())
        self.m_fluid = float(self.fm.sum())
        self.m_total = self.m_shell + self.m_fluid

        self._build_shell_topology()
        self._build_springs()
        self._build_wall_samples()
        self._build_sph()
        self._init_reports()
        self.t = 0.0
        self.n_step = 0
        self._stab_s = np.zeros(3)
        self._stab_f = np.zeros(3)
        self._forces()  # populate physical force reports for the t=0 frame
        self._assemble_budget(self._stab_s, self._stab_f)
    # --------------------------------------------------- shell mesh topology
    def _build_shell_topology(self) -> None:
        P = self.sp
        n = len(P)
        rad = np.hypot(P[:, 0], P[:, 1])
        poles = np.flatnonzero(rad < 1e-9)
        tris = None
        self.structured = False
        if poles.size == 2:
            ring_idx = np.setdiff1d(np.arange(n), poles)
            z = P[ring_idx, 2]
            ntheta = int(np.count_nonzero(np.isclose(z, z[0], atol=1e-9)))
            if ntheta >= 3 and ring_idx.size % ntheta == 0:
                nring = ring_idx.size // ntheta
                ring = ring_idx.reshape(nring, ntheta)
                ok = all(np.allclose(P[ring[r], 2], P[ring[r, 0], 2], atol=1e-9)
                         for r in range(nring))
                if ok:
                    self.structured = True
                    self.nring = nring
                    self.ntheta = ntheta
                    self.ring = ring
                    lo, hi = poles if P[poles[0], 2] <= P[poles[1], 2] else poles[::-1]
                    self.pole_lo = int(lo)
                    self.pole_hi = int(hi)
                    faces = []
                    for r in range(nring - 1):
                        for j in range(ntheta):
                            k = (j + 1) % ntheta
                            a, b = ring[r, j], ring[r, k]
                            c, d = ring[r + 1, k], ring[r + 1, j]
                            faces.append((a, b, c))
                            faces.append((a, c, d))
                    for j in range(ntheta):
                        k = (j + 1) % ntheta
                        faces.append((self.pole_lo, ring[0, j], ring[0, k]))
                        faces.append((self.pole_hi, ring[nring - 1, k], ring[nring - 1, j]))
                    tris = np.asarray(faces, dtype=np.int64)
        if tris is None:
            from scipy.spatial import ConvexHull
            tris = np.asarray(ConvexHull(P).simplices, dtype=np.int64)
        self.tris = tris
        v0, v1, v2 = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
        vol = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
        self.vol_sign = 1.0 if vol >= 0.0 else -1.0
        if self.vol_sign < 0.0:
            self.tris = tris[:, ::-1].copy()
            self.vol_sign = 1.0
        self.vol0 = abs(vol)

    def _shell_geom(self, P: np.ndarray):
        """Triangle cross products (2*A*n), enclosed volume, vertex area-normals."""
        T = self.tris
        v0, v1, v2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        cr = np.cross(v1 - v0, v2 - v0)
        vol = float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0)
        vn = np.zeros_like(P)
        third = cr / 6.0
        for k in range(3):
            np.add.at(vn, T[:, k], third)
        return cr, vol, vn

    # ------------------------------------------------------- spring network
    def _build_springs(self) -> None:
        edges: dict[tuple[int, int], float] = {}

        def add(a: int, b: int, k: float) -> None:
            a, b = int(a), int(b)
            if a == b:
                return
            key = (a, b) if a < b else (b, a)
            edges[key] = max(edges.get(key, 0.0), k)

        if self.structured:
            R, NT, ring = self.nring, self.ntheta, self.ring
            for r in range(R):
                for j in range(NT):
                    add(ring[r, j], ring[r, (j + 1) % NT], self.K_STRUCT)
                    add(ring[r, j], ring[r, (j + 2) % NT], self.K_BEND)
                    if r + 1 < R:
                        add(ring[r, j], ring[r + 1, j], self.K_STRUCT)
                        add(ring[r, j], ring[r + 1, (j + 1) % NT], self.K_SHEAR)
                        add(ring[r, (j + 1) % NT], ring[r + 1, j], self.K_SHEAR)
                    if r + 2 < R:
                        add(ring[r, j], ring[r + 2, j], self.K_BEND)
            for j in range(NT):
                add(self.pole_lo, ring[0, j], self.K_SPOKE)
                add(self.pole_hi, ring[R - 1, j], self.K_SPOKE)
                if R > 1:
                    add(self.pole_lo, ring[1, j], self.K_BEND)
                    add(self.pole_hi, ring[R - 2, j], self.K_BEND)
            add(self.pole_lo, self.pole_hi, self.K_BEND)
        else:
            for t in self.tris:
                add(t[0], t[1], self.K_STRUCT)
                add(t[1], t[2], self.K_STRUCT)
                add(t[2], t[0], self.K_STRUCT)

        keys = sorted(edges)
        self.e_i = np.array([k[0] for k in keys], dtype=np.int64)
        self.e_j = np.array([k[1] for k in keys], dtype=np.int64)
        self.e_k = np.array([edges[k] for k in keys], dtype=float)
        self.e_L0 = np.linalg.norm(self.sp[self.e_i] - self.sp[self.e_j], axis=1)
        self.e_L0 = np.maximum(self.e_L0, 1e-6)
        m_eff = float(np.mean(self.sm))
        self.e_c = 2.0 * self.ZETA_SPRING * np.sqrt(self.e_k * m_eff)
    # ------------------------------------------- impermeable wall sampling
    def _build_wall_samples(self) -> None:
        """Barycentric surface samples of the shell triangles (moving wall)."""
        bary = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ])
        T = self.tris
        nt, nb = len(T), len(bary)
        ns = nt * nb
        cols = np.empty(ns * 3, dtype=np.int64)
        vals = np.empty(ns * 3, dtype=float)
        for b in range(nb):
            sl = slice(b * nt * 3, (b + 1) * nt * 3)
            cols[sl] = T.reshape(-1)
            vals[sl] = np.tile(bary[b], nt)
        rows_s = np.concatenate([np.repeat(np.arange(b * nt, (b + 1) * nt), 3)
                                 for b in range(nb)])
        self.S = csr_matrix((vals, (rows_s, cols)), shape=(ns, len(self.sp)))
        self.S_T = self.S.T.tocsr()
        self.samp_tri = np.concatenate([np.arange(nt) for _ in range(nb)])
        self.n_samp = ns

    # ------------------------------------------------------------ SPH setup
    def _build_sph(self) -> None:
        h = self.SPH_H
        self.h_support = 2.0 * h
        tree = cKDTree(self.fp)
        pairs = np.asarray(list(tree.query_pairs(self.h_support)), dtype=np.int64)
        if pairs.size == 0:
            pairs = np.zeros((0, 2), dtype=np.int64)
        d = np.linalg.norm(self.fp[pairs[:, 0]] - self.fp[pairs[:, 1]], axis=1)
        w, _ = _cubic_spline_kernel(d, h)
        w0, _ = _cubic_spline_kernel(np.zeros(1), h)
        rho = np.full(len(self.fp), float(w0[0]) * float(self.fm[0]))
        np.add.at(rho, pairs[:, 0], w * self.fm[pairs[:, 1]])
        np.add.at(rho, pairs[:, 1], w * self.fm[pairs[:, 0]])
        self.rho0 = float(np.median(rho))
        self.B_eos = self.rho0 * self.SPH_C ** 2
        _, self.vol_gas0 = self._gas_state_init()

    def _gas_state_init(self):
        _, vol, _ = self._shell_geom(self.sp)
        vol = abs(vol)
        vol_liq = self.m_fluid / max(self.rho0, 1e-9)
        return vol, max(vol - vol_liq, 1e-3)

    # --------------------------------------------------------- report setup
    def _init_reports(self) -> None:
        n = self.n_all
        w = np.zeros((3, n))
        w[0, self.i_shell] = 1.0
        w[0, self.i_fluid] = 1.0
        w[1, self.i_shell] = 1.0
        w[2, self.i_fluid] = 1.0
        self.domain_weights = w
        self.cum_impulse = np.zeros((3, 3))
        self.cum_inflow = np.zeros((3, 3))
        self.cum_outflow = np.zeros((3, 3))
        self.cum_mass_source = np.zeros(2)
        self.budget_force = np.zeros((3, 4, 3))
        self.f_grav_shell = self.m_shell * self.gravity
        self.f_grav_fluid = self.m_fluid * self.gravity
        v_imp = float(np.sqrt(max(np.dot(self.fv[0], self.fv[0]), 0.0)
                              + 2.0 * abs(self.gravity[2]) * 0.58))
        self.ref_velocity = max(v_imp, 1e-3)
        self.ref_momentum = max(self.m_total * self.ref_velocity, 1e-6)
        self.f_wall_on_shell = np.zeros(3)
        self.f_wall_on_fluid = np.zeros(3)
        self.f_floor_shell = np.zeros(3)
        self.f_floor_fluid = np.zeros(3)
        self.f_stab = np.zeros((3, 3))
        self.fluid_tree = cKDTree(self.fp)
    # ------------------------------------------------------------- forces
    def _forces(self) -> None:
        P, V = self.sp, self.sv
        fp, fv = self.fp, self.fv

        # --- shell membrane springs (elastic + pair viscous damping) --------
        d = P[self.e_i] - P[self.e_j]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        nhat = d / L[:, None]
        stretch = np.clip(L - self.e_L0, -0.9 * self.e_L0, 3.0 * self.e_L0)
        vrel = V[self.e_i] - V[self.e_j]
        fmag = -self.e_k * stretch - self.e_c * np.einsum("ij,ij->i", vrel, nhat)
        F_shell = _scatter_pairs(self.e_i, self.e_j, fmag[:, None] * nhat, len(P))

        # --- enclosed gas pressure on the closed impermeable surface -------
        cr, vol_signed, vn_area = self._shell_geom(P)
        vol = abs(vol_signed)
        vol_liq = self.m_fluid / max(self.rho0, 1e-9)
        vol_gas = max(vol - vol_liq, 0.15 * self.vol_gas0)
        p_gas = self.P_GAS_REF * ((self.vol_gas0 / vol_gas) ** self.GAMMA_GAS - 1.0)
        p_gas = float(np.clip(p_gas, -5.0 * self.P_GAS_REF, 40.0 * self.P_GAS_REF))
        f_gas = p_gas * vn_area
        f_gas -= f_gas.mean(axis=0)          # closed surface: exact zero net force
        F_shell += f_gas

        # --- gravity -------------------------------------------------------
        F_shell += self.sm[:, None] * self.gravity
        F_fluid = self.fm[:, None] * self.gravity

        # --- floor contact on the shell ------------------------------------
        f_floor_s = self._floor_force(P, V, self.sm, self.K_FLOOR, self.ZETA_FLOOR)
        F_shell += f_floor_s
        self.f_floor_shell = f_floor_s.sum(axis=0)

        # --- weakly compressible SPH liquid --------------------------------
        h = self.SPH_H
        tree = cKDTree(fp)
        self.fluid_tree = tree
        pl = tree.query_pairs(self.h_support, output_type="ndarray")
        pairs = np.asarray(pl, dtype=np.int64).reshape(-1, 2)
        w0, _ = _cubic_spline_kernel(np.zeros(1), h)
        rho = float(w0[0]) * self.fm.copy()
        if len(pairs):
            i, j = pairs[:, 0], pairs[:, 1]
            xij = fp[i] - fp[j]
            r = np.maximum(np.linalg.norm(xij, axis=1), 1e-9)
            w, dw = _cubic_spline_kernel(r, h)
            np.add.at(rho, i, w * self.fm[j])
            np.add.at(rho, j, w * self.fm[i])
            press = np.maximum(self.B_eos * (rho / self.rho0 - 1.0), 0.0)
            eij = xij / r[:, None]
            term = press[i] / rho[i] ** 2 + press[j] / rho[j] ** 2
            vij = fv[i] - fv[j]
            mu = h * np.einsum("ij,ij->i", vij, eij) / np.maximum(r, 0.25 * h)
            rho_bar = 0.5 * (rho[i] + rho[j])
            visc = np.where(mu < 0.0, -self.SPH_ALPHA * self.SPH_C * mu / rho_bar, 0.0)
            coef = -(self.fm[i] * self.fm[j]) * (term + visc) * dw
            F_fluid += _scatter_pairs(i, j, coef[:, None] * eij, len(fp))
        self.fluid_rho = rho

        # --- impermeable moving wall: two-way fluid/shell coupling ---------
        sx = self.S @ P
        svv = self.S @ V
        nrm = _unit(cr)[self.samp_tri]
        wtree = cKDTree(sx)
        kk = min(self.WALL_KNN, self.n_samp)
        dist, idx = wtree.query(fp, k=kk, distance_upper_bound=0.20)
        dist = np.atleast_2d(dist.reshape(len(fp), kk))
        idx = np.atleast_2d(idx.reshape(len(fp), kk))
        valid = np.isfinite(dist)
        idx = np.where(valid, np.minimum(idx, self.n_samp - 1), 0)
        nn = nrm[idx]
        sd = np.einsum("ijk,ijk->ij", fp[:, None, :] - sx[idx], nn)
        pen = sd + self.WALL_BAND
        act = valid & (pen > 0.0)
        wgt = act.astype(float)
        wsum = wgt.sum(axis=1)
        good = wsum > 0.0
        wgt[good] = wgt[good] / wsum[good, None]
        pen_c = np.clip(pen, 0.0, 2.0 * self.WALL_BAND)
        vrel_w = fv[:, None, :] - svv[idx]
        vn_rel = np.einsum("ijk,ijk->ij", vrel_w, nn)
        fn_mag = self.K_WALL * pen_c + self.WALL_CN * np.maximum(vn_rel, 0.0)
        f_pair = -fn_mag[:, :, None] * nn
        vt_rel = vrel_w - vn_rel[:, :, None] * nn
        f_pair = (f_pair - self.WALL_CT * vt_rel) * wgt[:, :, None]
        f_on_fluid = f_pair.sum(axis=1)
        flat = idx.reshape(-1)
        fw = (-f_pair).reshape(-1, 3)
        samp_f = np.stack([np.bincount(flat, weights=fw[:, c], minlength=self.n_samp)
                           for c in range(3)], axis=1)
        f_wall_shell = np.asarray(self.S_T @ samp_f)
        F_fluid += f_on_fluid
        F_shell += f_wall_shell
        self.f_wall_on_fluid = f_on_fluid.sum(axis=0)
        self.f_wall_on_shell = f_wall_shell.sum(axis=0)

        # --- floor safety contact for any stray fluid sample ---------------
        f_floor_f = self._floor_force(fp, fv, self.fm, 0.5 * self.K_FLOOR, self.ZETA_FLOOR)
        F_fluid += f_floor_f
        self.f_floor_fluid = f_floor_f.sum(axis=0)

        self.F_shell = F_shell
        self.F_fluid = F_fluid
        self.p_gas = p_gas
        self.shell_volume = vol
        self.wall_sx = sx
        self.wall_nrm = nrm
        self.wall_tree = wtree

    def _floor_force(self, X, Vv, M, k_n, zeta) -> np.ndarray:
        out = np.zeros_like(X)
        pen = self.z_floor - X[:, 2]
        act = pen > 0.0
        if not np.any(act):
            return out
        c_n = 2.0 * zeta * np.sqrt(k_n * M[act])
        fn = np.maximum(k_n * pen[act] - c_n * Vv[act, 2], 0.0)
        out[act, 2] = fn
        vt = Vv[act][:, :2]
        speed = np.linalg.norm(vt, axis=1)
        ft = np.minimum(self.MU_FLOOR * fn, c_n * speed)
        out[np.flatnonzero(act)[:, None], np.array([0, 1])] = (
            -(ft / np.maximum(speed, 1e-9))[:, None] * vt
        )
        return out
    # --------------------------------------------------- time integration
    def _limit(self, V: np.ndarray) -> np.ndarray:
        sp = np.linalg.norm(V, axis=1)
        bad = sp > self.V_CLAMP
        if not np.any(bad):
            return V
        out = V.copy()
        out[bad] = V[bad] * (self.V_CLAMP / sp[bad])[:, None]
        return out

    def _assemble_budget(self, stab_s: np.ndarray, stab_f: np.ndarray) -> None:
        b = np.zeros((3, 4, 3))
        b[0, 0] = self.f_grav_shell + self.f_grav_fluid
        b[0, 1] = self.f_floor_shell + self.f_floor_fluid
        b[0, 2] = 0.0                                  # internal pair, cancels
        b[0, 3] = stab_s + stab_f
        b[1, 0] = self.f_grav_shell
        b[1, 1] = self.f_floor_shell
        b[1, 2] = self.f_wall_on_shell
        b[1, 3] = stab_s
        b[2, 0] = self.f_grav_fluid
        b[2, 1] = self.f_floor_fluid
        b[2, 2] = self.f_wall_on_fluid
        b[2, 3] = stab_f
        self.budget_force = b

    def _step(self) -> None:
        dt = self.dt
        self._forces()
        sv_raw = self.sv + dt * (self.F_shell / self.sm[:, None])
        fv_raw = self.fv + dt * (self.F_fluid / self.fm[:, None])
        sv_new = self._limit(sv_raw)
        fv_new = self._limit(fv_raw)
        stab_s = (self.sm[:, None] * (sv_new - sv_raw)).sum(axis=0) / dt
        stab_f = (self.fm[:, None] * (fv_new - fv_raw)).sum(axis=0) / dt
        self._stab_s = stab_s
        self._stab_f = stab_f
        self._assemble_budget(stab_s, stab_f)
        self.cum_impulse += dt * self.budget_force.sum(axis=1)
        self.sv = sv_new
        self.fv = fv_new
        self.sp = self.sp + dt * self.sv
        self.fp = self.fp + dt * self.fv
        self._advect_markers()
        self._contain_markers()

    def _advect_markers(self) -> None:
        if len(self.mp) == 0:
            return
        self.fluid_tree = cKDTree(self.fp)
        k = min(10, len(self.fp))
        dist, idx = self.fluid_tree.query(self.mp, k=k,
                                         distance_upper_bound=self.h_support)
        nm = len(self.mp)
        dist = np.asarray(dist).reshape(nm, k)
        idx = np.asarray(idx).reshape(nm, k)
        valid = np.isfinite(dist)
        idxc = np.where(valid, np.minimum(idx, len(self.fp) - 1), 0)
        w, _ = _cubic_spline_kernel(np.where(valid, dist, self.h_support), self.SPH_H)
        w = np.where(valid, w, 0.0)
        wsum = w.sum(axis=1)
        vel = np.einsum("ij,ijk->ik", w, self.fv[idxc])
        ok = wsum > 1e-12
        vel[ok] = vel[ok] / wsum[ok, None]
        vel[~ok] = self.mv[~ok] + self.dt * self.gravity
        self.mv = vel
        self.mp = self.mp + self.dt * self.mv

    def _contain_markers(self) -> None:
        """Massless tracers stay inside the impermeable wall (geometric clamp)."""
        if len(self.mp) == 0:
            return
        sx, nrm = self.wall_sx, self.wall_nrm
        _, idx = self.wall_tree.query(self.mp, k=1)
        idx = np.asarray(idx).reshape(-1)
        n1 = nrm[idx]
        sd = np.einsum("ij,ij->i", self.mp - sx[idx], n1)
        margin = -0.02
        bad = sd > margin
        if np.any(bad):
            self.mp[bad] = self.mp[bad] - (sd[bad] - margin)[:, None] * n1[bad]
            vn = np.einsum("ij,ij->i", self.mv[bad], n1[bad])
            self.mv[bad] = self.mv[bad] - np.maximum(vn, 0.0)[:, None] * n1[bad]

    # ------------------------------------------------------------ stepping
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        n_target = int(round(target / self.dt))
        while self.n_step < n_target:
            self._step()
            self.n_step += 1
        self.t = self.n_step * self.dt
        self._forces()          # report forces consistent with the sampled state
        self._assemble_budget(self._stab_s, self._stab_f)
        return self._frame(target)
    # -------------------------------------------------------- frame export
    def _frame(self, time: float) -> StateFrame:
        n = self.n_all
        pos = np.zeros((n, 3))
        vel = np.zeros((n, 3))
        pos[self.i_shell] = self.sp
        vel[self.i_shell] = self.sv
        pos[self.i_fluid] = self.fp
        vel[self.i_fluid] = self.fv
        pos[self.i_mark] = self.mp
        vel[self.i_mark] = self.mv

        # wall boundary samples: shell vertices with outward unit normals
        _, _, vn_area = self._shell_geom(self.sp)
        b_norm = _unit(vn_area)
        b_target = self.sv.copy()
        probe = self.sp - 0.5 * self.SPH_H * b_norm
        k = min(8, len(self.fp))
        dist, idx = self.fluid_tree.query(probe, k=k,
                                         distance_upper_bound=self.h_support)
        nb = len(probe)
        dist = np.asarray(dist).reshape(nb, k)
        idx = np.asarray(idx).reshape(nb, k)
        valid = np.isfinite(dist)
        idxc = np.where(valid, np.minimum(idx, len(self.fp) - 1), 0)
        w, _ = _cubic_spline_kernel(np.where(valid, dist, self.h_support), self.SPH_H)
        w = np.where(valid, w, 0.0)
        wsum = w.sum(axis=1)
        b_vel = np.einsum("ij,ijk->ik", w, self.fv[idxc])
        wet = wsum > 1e-12
        b_vel[wet] = b_vel[wet] / wsum[wet, None]
        b_vel[~wet] = b_target[~wet]          # dry (gas-side) wall samples

        mass_q = np.zeros((n, 2))
        mass_q[self.i_shell, 0] = self.sm
        mass_q[self.i_fluid, 1] = self.fm

        interaction = np.vstack((self.f_wall_on_shell, self.f_wall_on_fluid))

        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=self.mass_all.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": mass_q,
                "passive_marker": np.isin(self.body_id, (3, 4)).astype(float),
            },
            fields={
                "boundary_normal": b_norm,
                "boundary_target_velocity": b_target,
                "boundary_velocity": b_vel,
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": self.budget_force.copy(),
                "budget_cumulative_external_impulse": self.cum_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "interaction_force": interaction,
            },
            scalars={
                "reference_velocity": float(self.ref_velocity),
                "reference_momentum": float(self.ref_momentum),
                "gravity_z": float(self.gravity[2]),
                "cavity_is_closed": 1.0,
                "gas_gauge_pressure": float(self.p_gas),
                "shell_volume": float(self.shell_volume),
                "liquid_density_mean": float(np.mean(self.fluid_rho)),
                "substep_dt": float(self.dt),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "WCSPH liquid + mass-spring shell, penalty two-way coupling",
            "substep_dt": float(self.dt),
            "substeps_taken": int(self.n_step),
            "n_shell_vertices": int(len(self.sp)),
            "n_shell_springs": int(len(self.e_i)),
            "n_shell_triangles": int(len(self.tris)),
            "n_wall_samples": int(self.n_samp),
            "n_fluid_particles": int(len(self.fp)),
            "n_markers": int(len(self.mp)),
            "sph_h": float(self.SPH_H),
            "sph_c": float(self.SPH_C),
            "rho0": float(self.rho0),
            "structured_shell": bool(self.structured),
            "final_time": float(self.t),
        }
