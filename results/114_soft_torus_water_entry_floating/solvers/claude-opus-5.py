"""Two-way coupled soft-torus water-entry solver (self-implemented, no Genesis).

Physics
-------
* Water: weakly compressible SPH (cubic-spline kernel, linear equation of
  state, Monaghan artificial viscosity) on the scaffold water samples.
* Tub walls: analytic mirror-half-space boundary (density completion plus
  pressure repulsion) with a viscous no-slip layer, plus a hard
  non-penetration projection whose impulse is booked as a wall impulse.
* Soft torus: mass-spring-damper shell over the scaffold surface vertices
  (structural, shear, bending and cross-section springs) plus an enclosed
  volume pressure term, so ring outline and cross-section both deform.
* Two-way coupling: the deformed shell is expanded into layered solid
  coupling samples (Akinci-style boundary handling). Fluid feels boundary
  density, pressure, viscous drag and short-range repulsion; the exact
  equal-and-opposite reaction is redistributed to the shell vertices, so
  buoyancy, impact recoil and wave generation are all emergent.

Integration is semi-implicit (symplectic) Euler with a CFL-limited substep
count per output interval, so every sample time is hit exactly.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)

# --- fluid model -----------------------------------------------------------
PARTICLE_SPACING = 0.04
SMOOTHING_H = 0.042
SOUND_SPEED = 12.0
VISC_ALPHA = 0.22
WALL_NORMAL_DRAG = 26.0
WALL_TANGENT_DRAG = 9.0

# --- solid model -----------------------------------------------------------
K_STRUCT = 150.0
K_SHEAR = 50.0
K_BEND = 40.0
K_CROSS = 40.0
EDGE_DAMPING = 0.6
K_VOLUME = 1.2e4
SOLID_WALL_STIFFNESS = 1.5e4
SOLID_WALL_DAMPING = 30.0

# --- coupling --------------------------------------------------------------
COUPLING_DRAG = 3.0
REPULSION_RADIUS = 0.022
REPULSION_STIFFNESS = 500.0
LAYER_ALPHAS = (1.0, 0.72, 0.42)

# --- integration -----------------------------------------------------------
DT_MAX = 1.0e-3
CFL_COEFF = 0.35
MAX_SUBSTEPS = 60
NEIGHBOR_SKIN = 0.015


def _kernel_w(r: np.ndarray, h: float) -> np.ndarray:
    """Cubic spline kernel, support 2h, 3D normalisation."""
    sigma = 1.0 / (math.pi * h**3)
    q = r / h
    out = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    qa = q[m1]
    out[m1] = 1.0 - 1.5 * qa * qa + 0.75 * qa**3
    qb = 2.0 - q[m2]
    out[m2] = 0.25 * qb**3
    return sigma * out


def _kernel_dwdr(r: np.ndarray, h: float) -> np.ndarray:
    """dW/dr of the cubic spline (negative inside the support)."""
    sigma = 1.0 / (math.pi * h**3)
    q = r / h
    out = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    qa = q[m1]
    out[m1] = -3.0 * qa + 2.25 * qa * qa
    qb = 2.0 - q[m2]
    out[m2] = -0.75 * qb * qb
    return sigma * out / h


class BenchmarkSolver:
    """Weakly compressible SPH water two-way coupled to a soft-shell torus."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.config = cfg
        self.gravity = np.asarray(cfg.get("gravity", GRAVITY), dtype=float).reshape(3)

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.body_id = np.array(frame0.body_id, dtype=np.int32)
        self.n_total = self.pos.shape[0]

        self.solid_idx = np.flatnonzero(self.body_id == 1)
        self.fluid_idx = np.flatnonzero(self.body_id == 2)
        self.n_solid = int(self.solid_idx.size)
        self.n_fluid = int(self.fluid_idx.size)

        self.h = SMOOTHING_H
        self.support = 2.0 * self.h
        self.rho_water = float(
            cfg.get("initial_conditions", {}).get("water", {}).get("density", 1000.0)
        )
        self.c_sound = SOUND_SPEED

        self.fluid_mass = self.mass[self.fluid_idx].copy()
        self.solid_mass = self.mass[self.solid_idx].copy()
        self.solid_mass_total = float(self.solid_mass.sum())
        self.fluid_mass_total = float(self.fluid_mass.sum())

        # ---- domain (tub interior) -------------------------------------
        self.bounds_min = np.array([-0.5, -0.5, 0.0], dtype=float)
        self.bounds_max = np.array([0.5, 0.5, 1.4], dtype=float)
        # (axis, sign): sign = +1 -> wall at bounds_min (inward normal +axis)
        self.walls = (
            (0, +1, -0.5, np.array([1.0, 0.0, 0.0])),
            (0, -1, 0.5, np.array([-1.0, 0.0, 0.0])),
            (1, +1, -0.5, np.array([0.0, 1.0, 0.0])),
            (1, -1, 0.5, np.array([0.0, -1.0, 0.0])),
            (2, +1, 0.0, np.array([0.0, 0.0, 1.0])),
        )

        self._build_wall_tables()
        self._build_reference_density()
        self._build_solid_topology()
        self._build_coupling_map()
        self._build_boundary_samples()

        # ---- bookkeeping ------------------------------------------------
        self.time = 0.0
        self.n_steps = 0
        self.n_rebuilds = 0
        self.pairs = np.zeros((0, 2), dtype=np.int64)
        self.pair_ref_pos = self.pos[self.fluid_idx].copy()
        self._needs_rebuild = True
        self.min_solid_z = float(self.pos[self.solid_idx, 2].min())
        self.max_density = self.rho_water

        n_dom = 3
        self.cum_impulse = np.zeros((n_dom, 3), dtype=float)
        self.cum_inflow = np.zeros((n_dom, 3), dtype=float)
        self.cum_outflow = np.zeros((n_dom, 3), dtype=float)
        self.cum_mass_source = np.zeros(2, dtype=float)

        weights = np.zeros((n_dom, self.n_total), dtype=float)
        weights[0, self.solid_idx] = 1.0
        weights[1, self.fluid_idx] = 1.0
        weights[2, :] = 1.0
        self.domain_weights = weights

        self.last_force_terms = np.zeros((n_dom, 3, 3), dtype=float)
        self.last_interaction = np.zeros((2, 3), dtype=float)
        self.last_boundary_velocity = np.zeros((self.boundary_pos.shape[0], 3), dtype=float)

        self.reference_velocity = 2.0
        self.reference_momentum = float(
            (self.solid_mass_total + self.fluid_mass_total) * self.reference_velocity
        )

        # density / pressure state for the current configuration
        self.density = np.full(self.n_fluid, self.rho_ref, dtype=float)
        self.pressure = np.zeros(self.n_fluid, dtype=float)

        # frozen reference configuration (per-sample, N rows)
        rest_full = np.array(self.pos, dtype=float)
        rest_full[self.solid_idx] = self.rest_solid
        self.rest_position_full = rest_full

        # working state (compact per-phase views)
        self.xs = self.pos[self.solid_idx].copy()
        self.vs = self.vel[self.solid_idx].copy()
        self.xf = self.pos[self.fluid_idx].copy()
        self.vf = self.vel[self.fluid_idx].copy()

        self._warmup_force_snapshot()

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _build_wall_tables(self) -> None:
        """Analytic mirror-half-space kernel integrals for the tub walls.

        S(d)  = 2*pi * int_d^{2h} z W(z) dz          [1/m]
        Sint(d) = int_d^{2h} S(u) du                 [dimensionless, ->1/2]
        Density completion:  rho += rho_ref * Sint(d)
        Pressure repulsion:  a   = (2 p / rho^2) * rho_ref * S(d) * n_inward
        """
        n = 401
        d = np.linspace(0.0, self.support, n)
        integrand = 2.0 * math.pi * d * _kernel_w(d, self.h)
        # reverse cumulative trapezoid: S[k] = int_{d[k]}^{2h} integrand
        dx = d[1] - d[0]
        seg = 0.5 * (integrand[:-1] + integrand[1:]) * dx
        s_tab = np.concatenate([np.cumsum(seg[::-1])[::-1], [0.0]])
        seg2 = 0.5 * (s_tab[:-1] + s_tab[1:]) * dx
        sint_tab = np.concatenate([np.cumsum(seg2[::-1])[::-1], [0.0]])
        self.wall_d = d
        self.wall_S = s_tab
        self.wall_Sint = sint_tab
        self.wall_S0 = float(s_tab[0]) if s_tab[0] > 0.0 else 1.0
        self.w_self = float(_kernel_w(np.zeros(1), self.h)[0])
        self.wall_axes = np.asarray([w[0] for w in self.walls], dtype=np.int64)
        self.wall_signs = np.asarray([float(w[1]) for w in self.walls], dtype=float)
        self.wall_planes = np.asarray([float(w[2]) for w in self.walls], dtype=float)
        self.wall_normals = np.asarray([w[3] for w in self.walls], dtype=float)
        self.solid_contact = 0.012

    def _build_reference_density(self) -> None:
        """SPH-consistent reference density of the frozen water lattice."""
        fpos = self.pos[self.fluid_idx]
        tree = cKDTree(fpos)
        probe = np.array([0.0, 0.0, 0.5 * 0.4], dtype=float)
        _, i0 = tree.query(probe)
        i0 = int(i0)
        nb = np.asarray(tree.query_ball_point(fpos[i0], self.support), dtype=np.int64)
        rr = np.linalg.norm(fpos[nb] - fpos[i0], axis=1)
        rho = float(np.sum(self.fluid_mass[nb] * _kernel_w(rr, self.h)))
        self.rho_ref = rho if rho > 1.0 else self.rho_water

    def _build_solid_topology(self) -> None:
        """Mass-spring shell topology over the frozen torus surface vertices."""
        attrs = dict(getattr(self.scene.initial_state(), "attributes", {}) or {})
        rest_all = attrs.get("rest_position")
        rest = (
            np.asarray(rest_all, dtype=float)[self.solid_idx]
            if rest_all is not None
            else self.pos[self.solid_idx].copy()
        )
        self.rest_solid = rest
        center = np.asarray(
            self.config.get("initial_conditions", {})
            .get("torus", {})
            .get("center", [0.0, 0.0, 0.66]),
            dtype=float,
        ).reshape(3)
        self.torus_center0 = center

        rel = rest - center
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        nv = 1
        while nv < self.n_solid and abs(float(ang[nv] - ang[0])) < 1e-9:
            nv += 1
        if nv < 4 or self.n_solid % nv != 0:
            nv = 20 if self.n_solid % 20 == 0 else 1
        nu = self.n_solid // max(nv, 1)
        if nu < 4 or nu * nv != self.n_solid:
            nv, nu = self.n_solid, 1
        self.nv = int(nv)
        self.nu = int(nu)

        idx = np.arange(self.n_solid).reshape(self.nu, self.nv)
        faces = []
        edges = []
        stiff = []
        for i in range(self.nu):
            ip1 = (i + 1) % self.nu
            ip2 = (i + 2) % self.nu
            for j in range(self.nv):
                jp1 = (j + 1) % self.nv
                jp2 = (j + 2) % self.nv
                a = int(idx[i, j])
                b = int(idx[ip1, j])
                c = int(idx[ip1, jp1])
                dd = int(idx[i, jp1])
                faces.append([a, b, c])
                faces.append([a, c, dd])
                edges.append([a, b])
                stiff.append(K_STRUCT)
                edges.append([a, dd])
                stiff.append(K_STRUCT)
                edges.append([a, c])
                stiff.append(K_SHEAR)
                edges.append([b, dd])
                stiff.append(K_SHEAR)
                if self.nu > 4:
                    edges.append([a, int(idx[ip2, j])])
                    stiff.append(K_BEND)
                if self.nv > 4:
                    edges.append([a, int(idx[i, jp2])])
                    stiff.append(K_BEND)
                if j < self.nv // 2:
                    edges.append([a, int(idx[i, (j + self.nv // 2) % self.nv])])
                    stiff.append(K_CROSS)

        self.faces = np.asarray(faces, dtype=np.int64)
        self.edges = np.asarray(edges, dtype=np.int64)
        self.edge_k = np.asarray(stiff, dtype=float)
        self.edge_rest = np.linalg.norm(
            rest[self.edges[:, 1]] - rest[self.edges[:, 0]], axis=1
        )
        self.edge_rest = np.maximum(self.edge_rest, 1e-9)

        # orient faces so that the accumulated vertex normals point outward
        self.face_sign = 1.0
        nrm = self._vertex_area_normals(rest)
        ring_c = rest.reshape(self.nu, self.nv, 3).mean(axis=1)
        outward = rest - np.repeat(ring_c, self.nv, axis=0)
        if float(np.sum(nrm * outward)) < 0.0:
            self.face_sign = -1.0
        self.rest_volume = abs(self._enclosed_volume(rest))
        rho_solid = float(
            self.config.get("initial_conditions", {}).get("torus", {}).get("density", 650.0)
        )
        self.rho_solid = rho_solid
        self.solid_volume = self.solid_mass_total / max(rho_solid, 1e-9)

    def _build_coupling_map(self) -> None:
        """Layered solid coupling samples as a linear map of shell vertices.

        Each coupling sample p is an affine combination of shell vertices,
        p = sum_k M[p,k] x_k with sum_k M[p,k] = 1, so mapping forces back with
        M^T conserves linear momentum exactly.
        """
        nu, nv = self.nu, self.nv
        rows, cols, vals = [], [], []
        volumes = []
        alphas = [a for a in LAYER_ALPHAS if a > 1e-6]
        edges_a = [1.0]
        for k in range(len(alphas) - 1):
            edges_a.append(0.5 * (alphas[k] + alphas[k + 1]))
        edges_a.append(0.5 * alphas[-1])
        fracs = []
        for k in range(len(alphas)):
            hi, lo = edges_a[k], edges_a[k + 1]
            fracs.append(max(hi * hi - lo * lo, 0.0))
        core_frac = max(edges_a[-1] ** 2, 0.0)
        total = sum(fracs) + core_frac
        fracs = [f / total for f in fracs]
        core_frac = core_frac / total

        prow = 0
        for k, alpha in enumerate(alphas):
            vol_per = fracs[k] * self.solid_volume / float(nu * nv)
            for i in range(nu):
                ring = [int(i * nv + j) for j in range(nv)]
                for j in range(nv):
                    vtx = int(i * nv + j)
                    for c in ring:
                        rows.append(prow)
                        cols.append(c)
                        vals.append((1.0 - alpha) / float(nv))
                    rows.append(prow)
                    cols.append(vtx)
                    vals.append(alpha)
                    volumes.append(vol_per)
                    prow += 1
        # centreline layer
        vol_core = core_frac * self.solid_volume / float(nu)
        for i in range(nu):
            for j in range(nv):
                rows.append(prow)
                cols.append(int(i * nv + j))
                vals.append(1.0 / float(nv))
            volumes.append(vol_core)
            prow += 1

        self.coupling_map = csr_matrix(
            (np.asarray(vals, dtype=float), (np.asarray(rows), np.asarray(cols))),
            shape=(prow, self.n_solid),
        )
        self.coupling_map_t = self.coupling_map.T.tocsr()
        self.coupling_volume = np.asarray(volumes, dtype=float)
        self.n_coupling = int(prow)

    def _build_boundary_samples(self) -> None:
        """Static wall sample points with inward unit normals."""
        spacing = 0.08
        lo, hi = self.bounds_min, self.bounds_max
        z_top = 0.45
        pts, nrm = [], []
        gx = np.arange(lo[0] + 0.5 * spacing, hi[0], spacing)
        gy = np.arange(lo[1] + 0.5 * spacing, hi[1], spacing)
        gz = np.arange(0.5 * spacing, z_top, spacing)
        for x in gx:
            for y in gy:
                pts.append([float(x), float(y), float(lo[2])])
                nrm.append([0.0, 0.0, 1.0])
        for z in gz:
            for y in gy:
                pts.append([float(lo[0]), float(y), float(z)])
                nrm.append([1.0, 0.0, 0.0])
                pts.append([float(hi[0]), float(y), float(z)])
                nrm.append([-1.0, 0.0, 0.0])
            for x in gx:
                pts.append([float(x), float(lo[1]), float(z)])
                nrm.append([0.0, 1.0, 0.0])
                pts.append([float(x), float(hi[1]), float(z)])
                nrm.append([0.0, -1.0, 0.0])
        self.boundary_pos = np.asarray(pts, dtype=float)
        self.boundary_normal = np.asarray(nrm, dtype=float)
        self.boundary_target = np.zeros_like(self.boundary_pos)

    # ------------------------------------------------------------------
    # solid geometry helpers
    # ------------------------------------------------------------------
    def _vertex_area_normals(self, x: np.ndarray) -> np.ndarray:
        """Area-weighted vertex normals; sums to zero over a closed mesh."""
        a = x[self.faces[:, 0]]
        b = x[self.faces[:, 1]]
        c = x[self.faces[:, 2]]
        fn = 0.5 * np.cross(b - a, c - a) * self.face_sign
        out = np.zeros_like(x)
        third = fn / 3.0
        for k in range(3):
            for comp in range(3):
                out[:, comp] += np.bincount(
                    self.faces[:, k], weights=third[:, comp], minlength=x.shape[0]
                )
        return out

    def _enclosed_volume(self, x: np.ndarray) -> float:
        a = x[self.faces[:, 0]]
        b = x[self.faces[:, 1]]
        c = x[self.faces[:, 2]]
        return float(np.einsum("ij,ij->", a, np.cross(b, c)) / 6.0) * self.face_sign

    def _solid_internal_force(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Springs + damping + enclosed-volume pressure (all momentum neutral)."""
        i0 = self.edges[:, 0]
        i1 = self.edges[:, 1]
        d = x[i1] - x[i0]
        length = np.linalg.norm(d, axis=1)
        length = np.maximum(length, 1e-9)
        u = d / length[:, None]
        rel = np.einsum("ij,ij->i", v[i1] - v[i0], u)
        scal = self.edge_k * (length - self.edge_rest) + EDGE_DAMPING * rel
        f = scal[:, None] * u
        out = np.zeros_like(x)
        for comp in range(3):
            out[:, comp] += np.bincount(i0, weights=f[:, comp], minlength=x.shape[0])
            out[:, comp] -= np.bincount(i1, weights=f[:, comp], minlength=x.shape[0])

        vol = abs(self._enclosed_volume(x))
        ratio = float(np.clip(self.rest_volume / max(vol, 1e-9), 0.2, 5.0))
        press = K_VOLUME * (ratio - 1.0)
        out += press * self._vertex_area_normals(x)
        return out

    def _rebuild_neighbors(self) -> None:
        xf = self.xf
        radius = self.support + NEIGHBOR_SKIN
        tree = cKDTree(xf)
        self.pairs = tree.query_pairs(radius, output_type="ndarray")
        self.pair_ref_pos = xf.copy()
        self.n_rebuilds += 1

    def _wall_geometry(self, x: np.ndarray) -> np.ndarray:
        """Signed inward distances to the five tub walls, shape [n,5]."""
        return self.wall_signs[None, :] * (x[:, self.wall_axes] - self.wall_planes[None, :])

    def _step(self, dt: float) -> None:
        h = self.h
        xf, vf = self.xf, self.vf
        xs, vs = self.xs, self.vs
        mf = self.fluid_mass
        ms = self.solid_mass

        if self.pairs.shape[0] == 0 or float(
            np.max(np.abs(xf - self.pair_ref_pos))
        ) > 0.5 * NEIGHBOR_SKIN:
            self._rebuild_neighbors()

        # ---- solid coupling samples ------------------------------------
        cp = self.coupling_map @ xs
        cv = self.coupling_map @ vs
        ctree = cKDTree(cp)
        ftree = cKDTree(xf)
        cpl = ftree.sparse_distance_matrix(
            ctree, self.support, output_type="ndarray"
        )
        ci = cpl["i"].astype(np.int64)
        cj = cpl["j"].astype(np.int64)
        cr = cpl["v"].astype(float)

        # ---- fluid density ---------------------------------------------
        pi_ = self.pairs[:, 0]
        pj_ = self.pairs[:, 1]
        dvec = xf[pi_] - xf[pj_]
        r2 = np.einsum("ij,ij->i", dvec, dvec)
        r = np.sqrt(r2)
        w = _kernel_w(r, h)
        rho = mf * self.w_self
        rho += np.bincount(pi_, weights=mf[pj_] * w, minlength=self.n_fluid)
        rho += np.bincount(pj_, weights=mf[pi_] * w, minlength=self.n_fluid)

        psi = self.rho_ref * self.coupling_volume
        if ci.size:
            wc = _kernel_w(cr, h)
            rho += np.bincount(ci, weights=psi[cj] * wc, minlength=self.n_fluid)

        dw = self._wall_geometry(xf)
        dwc = np.clip(dw, 0.02 * h, self.support)
        near = dw < self.support
        sint = np.where(near, np.interp(dwc, self.wall_d, self.wall_Sint), 0.0)
        swall = np.where(near, np.interp(dwc, self.wall_d, self.wall_S), 0.0)
        rho = rho + self.rho_ref * sint.sum(axis=1)

        press = self.c_sound**2 * (rho - self.rho_ref)
        np.maximum(press, 0.0, out=press)
        self.density = rho
        self.pressure = press

        # ---- fluid-fluid pressure + artificial viscosity ---------------
        Ff = np.zeros((self.n_fluid, 3), dtype=float)
        if pi_.size:
            rsafe = np.maximum(r, 1e-9)
            dwdr = _kernel_dwdr(r, h)
            pterm = press[pi_] / rho[pi_] ** 2 + press[pj_] / rho[pj_] ** 2
            vij = vf[pi_] - vf[pj_]
            dotp = np.einsum("ij,ij->i", vij, dvec)
            mu = h * dotp / (r2 + 0.01 * h * h)
            mu = np.minimum(mu, 0.0)
            rho_bar = 0.5 * (rho[pi_] + rho[pj_])
            visc = -VISC_ALPHA * self.c_sound * mu / rho_bar
            scal = -(mf[pi_] * mf[pj_]) * (pterm + visc) * dwdr / rsafe
            fpair = scal[:, None] * dvec
            for comp in range(3):
                Ff[:, comp] += np.bincount(
                    pi_, weights=fpair[:, comp], minlength=self.n_fluid
                )
                Ff[:, comp] -= np.bincount(
                    pj_, weights=fpair[:, comp], minlength=self.n_fluid
                )

        # ---- fluid <-> solid coupling ----------------------------------
        f_couple_fluid = np.zeros((self.n_fluid, 3), dtype=float)
        f_couple_pts = np.zeros((self.n_coupling, 3), dtype=float)
        if ci.size:
            rc = np.maximum(cr, 1e-9)
            dc = xf[ci] - cp[cj]
            dwdrc = _kernel_dwdr(cr, h)
            pc = 2.0 * press[ci] / rho[ci] ** 2
            scal_c = -(mf[ci] * psi[cj]) * pc * dwdrc / rc
            fc = scal_c[:, None] * dc
            wcn = _kernel_w(cr, h)
            drel = vf[ci] - cv[cj]
            fc -= (COUPLING_DRAG * mf[ci] * self.coupling_volume[cj] * wcn)[:, None] * drel
            over = REPULSION_RADIUS - cr
            act = over > 0.0
            if np.any(act):
                mag = REPULSION_STIFFNESS * np.where(act, over, 0.0) / rc
                fc += mag[:, None] * dc
            for comp in range(3):
                f_couple_fluid[:, comp] += np.bincount(
                    ci, weights=fc[:, comp], minlength=self.n_fluid
                )
                f_couple_pts[:, comp] -= np.bincount(
                    cj, weights=fc[:, comp], minlength=self.n_coupling
                )
        f_couple_solid = self.coupling_map_t @ f_couple_pts
        Ff += f_couple_fluid

        # ---- tub walls acting on the fluid -----------------------------
        amag = (2.0 * press / rho**2 * self.rho_ref)[:, None] * swall
        f_wall_f = mf[:, None] * (amag @ self.wall_normals)
        wn = swall / self.wall_S0
        vn = vf @ self.wall_normals.T
        wvn = wn * vn
        f_wall_f -= mf[:, None] * (
            WALL_NORMAL_DRAG * (wvn @ self.wall_normals)
            + WALL_TANGENT_DRAG * (wn.sum(axis=1)[:, None] * vf - wvn @ self.wall_normals)
        )
        Ff += f_wall_f

        # ---- solid internal + walls ------------------------------------
        Fs = self._solid_internal_force(xs, vs)
        dws = self._wall_geometry(xs)
        pen = np.maximum(self.solid_contact - dws, 0.0)
        vns = vs @ self.wall_normals.T
        active = pen > 0.0
        fmag = SOLID_WALL_STIFFNESS * pen - SOLID_WALL_DAMPING * ms[:, None] * np.where(
            active, np.minimum(vns, 0.0), 0.0
        )
        f_wall_s = np.where(active, fmag, 0.0) @ self.wall_normals
        Fs = Fs + f_wall_s + f_couple_solid

        # ---- gravity ----------------------------------------------------
        g = self.gravity
        Ff += mf[:, None] * g[None, :]
        Fs += ms[:, None] * g[None, :]

        # ---- integrate (semi-implicit Euler) ---------------------------
        vf += (Ff / mf[:, None]) * dt
        vs += (Fs / ms[:, None]) * dt
        xf += vf * dt
        xs += vs * dt

        imp_f = self._project(xf, vf, mf)
        imp_s = self._project(xs, vs, ms)

        # ---- momentum bookkeeping --------------------------------------
        gf = float(mf.sum()) * g
        gs = float(ms.sum()) * g
        wall_f = f_wall_f.sum(axis=0) + imp_f / dt
        wall_s = f_wall_s.sum(axis=0) + imp_s / dt
        coup_s = f_couple_solid.sum(axis=0)
        coup_f = -coup_s
        terms = np.zeros((3, 3, 3), dtype=float)
        terms[0, 0] = gs
        terms[0, 1] = wall_s
        terms[0, 2] = coup_s
        terms[1, 0] = gf
        terms[1, 1] = wall_f
        terms[1, 2] = coup_f
        terms[2, 0] = gs + gf
        terms[2, 1] = wall_s + wall_f
        terms[2, 2] = np.zeros(3)
        self.last_force_terms = terms
        self.cum_impulse += terms.sum(axis=1) * dt
        self.last_interaction = np.vstack([coup_s, coup_f])

        self.time += dt
        self.n_steps += 1
        self.max_density = max(self.max_density, float(rho.max()))
        self.min_solid_z = min(self.min_solid_z, float(xs[:, 2].min()))

    def _project(self, x: np.ndarray, v: np.ndarray, m: np.ndarray) -> np.ndarray:
        """Hard non-penetration clamp; returns the applied wall impulse."""
        eps = 1e-4
        v_before = v.copy()
        lo = self.bounds_min + eps
        hi = self.bounds_max - eps
        for axis in range(3):
            below = x[:, axis] < lo[axis]
            if np.any(below):
                x[below, axis] = lo[axis]
                v[below, axis] = np.maximum(v[below, axis], 0.0)
            above = x[:, axis] > hi[axis]
            if np.any(above):
                x[above, axis] = hi[axis]
                v[above, axis] = np.minimum(v[above, axis], 0.0)
        return (m[:, None] * (v - v_before)).sum(axis=0)

    # ------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------
    def _warmup_force_snapshot(self) -> None:
        """Evaluate forces at t=0 without advancing the physical state."""
        xs0 = self.xs.copy()
        vs0 = self.vs.copy()
        xf0 = self.xf.copy()
        vf0 = self.vf.copy()
        self._step(1e-9)
        self.xs, self.vs = xs0, vs0
        self.xf, self.vf = xf0, vf0
        self.time = 0.0
        self.n_steps = 0
        self.cum_impulse[:] = 0.0
        self.min_solid_z = float(self.xs[:, 2].min())
        self._needs_rebuild = True
        self.pairs = np.zeros((0, 2), dtype=np.int64)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while target - self.time > 1e-12:
            remaining = target - self.time
            vmax = max(
                float(np.abs(self.vf).max(initial=0.0)),
                float(np.abs(self.vs).max(initial=0.0)),
                1e-6,
            )
            dt_cfl = CFL_COEFF * self.h / (self.c_sound + vmax)
            dt_lim = min(DT_MAX, dt_cfl)
            n_sub = int(math.ceil(remaining / dt_lim))
            n_sub = max(1, min(n_sub, MAX_SUBSTEPS))
            dt = remaining / n_sub
            for _ in range(n_sub):
                self._step(dt)
        self.time = target
        return self._build_frame(target)

    def _boundary_velocity(self) -> np.ndarray:
        """Shepard-normalised fluid velocity measured at the wall samples."""
        nb = self.boundary_pos.shape[0]
        out = np.zeros((nb, 3), dtype=float)
        btree = cKDTree(self.boundary_pos)
        ftree = cKDTree(self.xf)
        pairs = btree.sparse_distance_matrix(ftree, self.h, output_type="ndarray")
        if pairs.size == 0:
            return out
        bi = pairs["i"].astype(np.int64)
        fj = pairs["j"].astype(np.int64)
        r = pairs["v"].astype(float)
        w = _kernel_w(r, 0.5 * self.h)
        denom = np.bincount(bi, weights=w, minlength=nb)
        for comp in range(3):
            num = np.bincount(bi, weights=w * self.vf[fj, comp], minlength=nb)
            good = denom > 1e-12
            out[good, comp] = num[good] / denom[good]
        return out

    def _build_frame(self, t: float) -> StateFrame:
        pos = np.empty((self.n_total, 3), dtype=float)
        vel = np.empty((self.n_total, 3), dtype=float)
        pos[self.solid_idx] = self.xs
        pos[self.fluid_idx] = self.xf
        vel[self.solid_idx] = self.vs
        vel[self.fluid_idx] = self.vf
        self.pos = pos
        self.vel = vel

        mass_quantity = np.zeros((self.n_total, 2), dtype=float)
        mass_quantity[self.solid_idx, 0] = self.solid_mass
        mass_quantity[self.fluid_idx, 1] = self.fluid_mass

        bvel = self._boundary_velocity()
        self.last_boundary_velocity = bvel

        frame = StateFrame(
            time=float(t),
            position=pos.copy(),
            velocity=vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": mass_quantity,
                "rest_position": self._rest_position_full(),
                "phase_id": self.body_id.astype(float),
            },
            fields={
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_target.copy(),
                "boundary_velocity": bvel,
                "budget_cumulative_external_impulse": self.cum_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": self.last_force_terms.copy(),
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "interaction_force": self.last_interaction.copy(),
                "gravity_acceleration": self.gravity.reshape(1, 3).copy(),
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "reference_velocity": float(self.reference_velocity),
                "water_free_surface_z": float(
                    self.config.get("initial_conditions", {})
                    .get("water", {})
                    .get("free_surface_z", 0.4)
                ),
                "liquid_phase_id": 2.0,
                "characteristic_length": float(PARTICLE_SPACING),
            },
        )
        frame.validate()
        return frame

    def _rest_position_full(self) -> np.ndarray:
        return self.rest_position_full.copy()

    def diagnostics(self) -> dict[str, object]:
        momentum = float(
            np.linalg.norm(
                (self.solid_mass[:, None] * self.vs).sum(axis=0)
                + (self.fluid_mass[:, None] * self.vf).sum(axis=0)
            )
        )
        return {
            "method": "WCSPH water + mass-spring-pressure soft shell, Akinci coupling",
            "genesis_used": False,
            "n_fluid_particles": int(self.n_fluid),
            "n_solid_vertices": int(self.n_solid),
            "n_coupling_samples": int(self.n_coupling),
            "torus_grid": [int(self.nu), int(self.nv)],
            "smoothing_length": float(self.h),
            "sound_speed": float(self.c_sound),
            "reference_density": float(self.rho_ref),
            "substeps_taken": int(self.n_steps),
            "neighbor_rebuilds": int(self.n_rebuilds),
            "time": float(self.time),
            "torus_center_z": float(self.xs[:, 2].mean()),
            "torus_min_z": float(self.min_solid_z),
            "max_density": float(self.max_density),
            "net_momentum_norm": momentum,
        }
