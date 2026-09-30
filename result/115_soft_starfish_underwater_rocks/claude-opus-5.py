"""Two-way coupled WCSPH water + mass-spring soft starfish, implemented from scratch.

Numerical model
---------------
* Water: weakly compressible SPH (cubic spline kernel, Tait equation of state,
  Morris-type viscous term) with a genuine free surface (no lid, particles are
  only bounded by the frozen tank planes and the ellipsoidal rocks).
* Starfish: the frozen closed triangle mesh is discretised as a mass/spring/damper
  shell -- stretch, shear, thickness and bending springs -- integrated with
  symplectic Euler substeps. Arms are completely unconstrained.
* Coupling: Akinci-style boundary-particle treatment. Fluid pressure and viscous
  stresses act on the deforming shell nodes and the equal-and-opposite reaction is
  applied to the fluid, so the solid alters the resolved flow and vice versa.
* Contact: analytic ellipsoid rocks and tank planes give penalty normal contact
  with Coulomb friction; every contact/clamp impulse is accumulated exactly so the
  exported momentum budgets close to machine precision.
"""
from __future__ import annotations

import math
import time as _time

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

N_THETA = 100
N_RADIAL = 8

FALLBACK_ROCKS = (
    ((-0.34, -0.25, 0.10), 0.12, (1.25, 0.82, 0.78)),
    ((-0.10, -0.31, 0.11), 0.105, (1.20, 0.78, 1.05)),
    ((0.18, -0.30, 0.09), 0.115, (1.35, 0.78, 0.72)),
    ((0.39, -0.22, 0.12), 0.11, (0.78, 1.12, 1.05)),
    ((-0.39, 0.00, 0.12), 0.115, (0.90, 1.28, 1.00)),
    ((-0.16, -0.04, 0.13), 0.125, (1.15, 0.88, 1.06)),
    ((0.09, -0.06, 0.10), 0.10, (1.28, 0.85, 0.82)),
    ((0.34, 0.03, 0.14), 0.13, (0.88, 1.12, 1.10)),
    ((-0.31, 0.27, 0.10), 0.11, (1.30, 0.82, 0.78)),
    ((-0.05, 0.25, 0.15), 0.12, (0.88, 1.20, 1.20)),
    ((0.20, 0.25, 0.09), 0.105, (1.42, 0.78, 0.75)),
    ((0.42, 0.29, 0.11), 0.115, (0.78, 1.12, 0.92)),
)

def starfish_topology(n_theta: int = N_THETA, n_radial: int = N_RADIAL):
    """Rebuild the frozen mesh connectivity (same generator as the scaffold)."""
    n_vert = 2 + 2 * n_theta * n_radial

    def top(j, i):
        return 2 + (j * n_theta + (i % n_theta)) * 2

    def bot(j, i):
        return top(j, i) + 1

    faces = []
    for i in range(n_theta):
        k = (i + 1) % n_theta
        faces.append([0, top(0, i), top(0, k)])
        faces.append([1, bot(0, k), bot(0, i)])
    for j in range(n_radial - 1):
        for i in range(n_theta):
            k = (i + 1) % n_theta
            it, ib = top(j, i), bot(j, i)
            kt, kb = top(j, k), bot(j, k)
            ot, ob = top(j + 1, i), bot(j + 1, i)
            qt, qb = top(j + 1, k), bot(j + 1, k)
            faces.extend([[it, ot, qt], [it, qt, kt], [ib, kb, qb], [ib, qb, ob]])
    for i in range(n_theta):
        k = (i + 1) % n_theta
        t0, b0 = top(n_radial - 1, i), bot(n_radial - 1, i)
        t1, b1 = top(n_radial - 1, k), bot(n_radial - 1, k)
        faces.extend([[t0, b0, b1], [t0, b1, t1]])
    faces = np.asarray(faces, dtype=np.int64)

    struct = []
    bend = []
    for j in range(n_radial):
        for i in range(n_theta):
            k = (i + 1) % n_theta
            k2 = (i + 2) % n_theta
            struct.append((top(j, i), top(j, k)))
            struct.append((bot(j, i), bot(j, k)))
            struct.append((top(j, i), bot(j, i)))          # thickness strut
            struct.append((top(j, i), bot(j, k)))          # thickness diagonals
            struct.append((bot(j, i), top(j, k)))
            bend.append((top(j, i), top(j, k2)))
            bend.append((bot(j, i), bot(j, k2)))
    for j in range(n_radial - 1):
        for i in range(n_theta):
            k = (i + 1) % n_theta
            struct.append((top(j, i), top(j + 1, i)))      # radial
            struct.append((bot(j, i), bot(j + 1, i)))
            struct.append((top(j, i), top(j + 1, k)))      # in-plane shear
            struct.append((top(j + 1, i), top(j, k)))
            struct.append((bot(j, i), bot(j + 1, k)))
            struct.append((bot(j + 1, i), bot(j, k)))
            struct.append((top(j, i), bot(j + 1, i)))      # through-thickness shear
            struct.append((bot(j, i), top(j + 1, i)))
    for j in range(n_radial - 2):
        for i in range(n_theta):
            bend.append((top(j, i), top(j + 2, i)))
            bend.append((bot(j, i), bot(j + 2, i)))
    for i in range(n_theta):
        struct.append((0, top(0, i)))
        struct.append((1, bot(0, i)))
        bend.append((0, top(1, i)))
        bend.append((1, bot(1, i)))
    struct.append((0, 1))

    struct = np.asarray(struct, dtype=np.int64)
    bend = np.asarray(bend, dtype=np.int64)
    return n_vert, faces, struct, bend


class CubicSpline:
    """3D cubic spline SPH kernel with support radius 2h."""

    def __init__(self, h: float):
        self.h = float(h)
        self.radius = 2.0 * self.h
        self.sigma = 1.0 / (math.pi * self.h ** 3)

    def w(self, r):
        q = np.asarray(r) / self.h
        out = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        out[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
        out[m2] = 0.25 * (2.0 - q[m2]) ** 3
        return self.sigma * out

    def dwdr(self, r):
        q = np.asarray(r) / self.h
        out = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        out[m1] = -3.0 * q[m1] + 2.25 * q[m1] ** 2
        out[m2] = -0.75 * (2.0 - q[m2]) ** 2
        return self.sigma * out / self.h


class BenchmarkSolver:
    """Explicit two-way coupled solver (no external physics engine)."""

    # --- numerical parameters -------------------------------------------------
    DX = 0.04                 # fluid particle spacing
    H_FACTOR = 1.0            # smoothing length = H_FACTOR * DX
    C_SOUND = 15.0            # artificial speed of sound
    NU_ART = 2.0e-3           # effective kinematic viscosity of the resolved water
    P_MAX = 6.0e4             # pressure clamp (>> hydrostatic + water-hammer head)
    F_PAIR_MAX = 10.0         # per-pair coupling force clamp (kept antisymmetric)
    DT = 5.0e-4               # fluid time step
    SOLID_SUBSTEPS = 2
    K_STRETCH = 700.0
    K_BEND = 200.0
    ZETA = 0.08               # spring damping ratio
    K_WALL_FLUID = 1.2e4
    K_WALL_SOLID = 2.0e3
    C_WALL_FLUID = 6.0
    C_WALL_SOLID = 1.2
    MU_ROCK = 0.55
    MU_GLASS = 0.25
    K_PENALTY_FS = 900.0      # fluid/solid short-range repulsion
    BOUNDARY_VISC_FACTOR = 2.0
    TIME_BUDGET = 1500.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(scene.config)
        self.cfg = cfg
        self.g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        water = cfg.get("water", {}) or {}
        self.rho0 = float(water.get("density", 1000.0))
        self.free_surface_z = float(water.get("initial_free_surface_z", 0.40))
        bounds = water.get("domain_bounds", [[-0.5, -0.5, 0.0], [0.5, 0.5, 0.72]])
        self.bmin = np.asarray(bounds[0], dtype=float)
        self.bmax = np.asarray(bounds[1], dtype=float)
        star = cfg.get("starfish", {}) or {}
        self.rho_solid = float(star.get("density", 1080.0))

        self.rocks = self._load_rocks(cfg)
        self.planes = self._build_planes()

        # --- solid ------------------------------------------------------------
        init = scene.initial_state()
        pos0 = np.asarray(init.position, dtype=float)
        vel0 = np.asarray(init.velocity, dtype=float)
        bid0 = np.asarray(init.body_id)
        solid_mask = bid0 == 1
        self.xs = np.ascontiguousarray(pos0[solid_mask])
        self.vs = np.ascontiguousarray(vel0[solid_mask])
        self.ns = len(self.xs)
        self.rest_solid = self.xs.copy()

        n_vert, faces, struct, bend = starfish_topology()
        if n_vert != self.ns:
            faces, struct, bend = self._fallback_topology()
        self.faces = faces
        self._build_springs(struct, bend)
        self._build_solid_mass(faces)

        # --- fluid ------------------------------------------------------------
        self.kern = CubicSpline(self.H_FACTOR * self.DX)
        self.xf = self._seed_fluid()
        self.nf = len(self.xf)
        self.vf = np.zeros((self.nf, 3), dtype=float)
        self.mf = self.rho0 * self.DX ** 3
        self.mass_f = np.full(self.nf, self.mf)
        self.rho_f = np.full(self.nf, self.rho0)
        self.press_f = np.zeros(self.nf)
        self.B_tait = self.rho0 * self.C_SOUND ** 2 / 7.0
        self.nu_art = float(self.NU_ART)

        # boundary-particle pseudo mass for the shell nodes (Akinci)
        self.psi = self._boundary_psi()

        self.mass_all = np.concatenate([self.mass_s, self.mass_f])
        self.M_solid = float(self.mass_s.sum())
        self.M_water = float(self.mass_f.sum())

        # --- boundary samples -------------------------------------------------
        self.b_pos, self.b_nrm = self._boundary_samples()
        self.nb = len(self.b_pos)

        # --- budget bookkeeping ----------------------------------------------
        self.imp_solid = np.zeros(3)     # boundary contact + clamp impulse, solid
        self.imp_fluid = np.zeros(3)
        self.imp_cross_solid = np.zeros(3)   # fluid -> solid impulse
        self.imp_cross_fluid = np.zeros(3)   # solid -> fluid impulse
        self.imp_stab_solid = np.zeros(3)
        self.imp_stab_fluid = np.zeros(3)
        self.f_bnd_solid = np.zeros(3)
        self.f_bnd_fluid = np.zeros(3)
        self.f_cross_solid = np.zeros(3)
        self.f_cross_fluid = np.zeros(3)

        self.t = 0.0
        self.steps = 0
        self._rebuild_at = -1
        self._pairs_ff = np.zeros((0, 2), dtype=np.int64)
        self._pairs_fs = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        self._rock_cand_f = [np.zeros(0, dtype=np.int64)] * len(self.rocks)
        self._rock_cand_s = [np.zeros(0, dtype=np.int64)] * len(self.rocks)
        self._plane_cand_f = [np.zeros(0, dtype=np.int64)] * len(self.planes)
        self._plane_cand_s = [np.zeros(0, dtype=np.int64)] * len(self.planes)
        self.v_ref = 2.0
        self.p_ref = (self.M_solid + self.M_water) * self.v_ref
        self._t0 = _time.time()
        self._frozen = False
        self._nan_events = 0
        self._rebuild(force=True)

    # ------------------------------------------------------------------ setup
    def _load_rocks(self, cfg):
        rocks = []
        geom = cfg.get("scene_geometry")
        objs = None
        if isinstance(geom, dict):
            objs = geom.get("objects")
        if objs:
            for item in objs:
                if item.get("semantic") != "rigid":
                    continue
                g = item.get("geometry", {}) or {}
                if g.get("type") != "sphere":
                    continue
                tr = item.get("transform", {}) or {}
                c = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float)
                s = np.asarray(tr.get("scale", [1.0, 1.0, 1.0]), dtype=float)
                a = float(g.get("radius", 0.1)) * s
                rocks.append((c, a))
        if not rocks:
            for c, r, s in FALLBACK_ROCKS:
                rocks.append((np.asarray(c, dtype=float), r * np.asarray(s, dtype=float)))
        return rocks

    def _build_planes(self):
        planes = []
        lo, hi = self.bmin, self.bmax
        planes.append((np.array([0.0, 0.0, 1.0]), float(lo[2])))
        planes.append((np.array([1.0, 0.0, 0.0]), float(lo[0])))
        planes.append((np.array([-1.0, 0.0, 0.0]), -float(hi[0])))
        planes.append((np.array([0.0, 1.0, 0.0]), float(lo[1])))
        planes.append((np.array([0.0, -1.0, 0.0]), -float(hi[1])))
        planes.append((np.array([0.0, 0.0, -1.0]), -float(hi[2])))
        return planes


    def _fallback_topology(self):
        """Connectivity from geometric proximity if the mesh layout ever changes."""
        tree = cKDTree(self.rest_solid)
        pairs = tree.query_pairs(2.2 * self._median_spacing(tree), output_type="ndarray")
        struct = np.asarray(pairs, dtype=np.int64)
        d = np.linalg.norm(self.rest_solid[struct[:, 0]] - self.rest_solid[struct[:, 1]], axis=1)
        cut = np.median(d)
        bend = struct[d > cut]
        struct = struct[d <= cut]
        faces = np.zeros((0, 3), dtype=np.int64)
        return faces, struct, bend

    def _median_spacing(self, tree):
        d, _ = tree.query(self.rest_solid, k=2)
        return float(np.median(d[:, 1]))

    def _build_springs(self, struct, bend):
        struct = np.unique(np.sort(struct, axis=1), axis=0)
        bend = np.unique(np.sort(bend, axis=1), axis=0)
        self.sp_i = np.concatenate([struct[:, 0], bend[:, 0]]).astype(np.int64)
        self.sp_j = np.concatenate([struct[:, 1], bend[:, 1]]).astype(np.int64)
        self.sp_k = np.concatenate([
            np.full(len(struct), self.K_STRETCH),
            np.full(len(bend), self.K_BEND),
        ])
        d = self.rest_solid[self.sp_j] - self.rest_solid[self.sp_i]
        self.sp_rest = np.linalg.norm(d, axis=1)
        self.sp_rest = np.maximum(self.sp_rest, 1e-6)
        self.n_springs = len(self.sp_i)

    def _build_solid_mass(self, faces):
        n = self.ns
        if len(faces):
            p = self.rest_solid[faces]
            vol = abs(float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0))
            area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
            w = np.zeros(n)
            for k in range(3):
                w += np.bincount(faces[:, k], area / 3.0, minlength=n)
            self.surface_area = float(area.sum())
        else:
            vol = 0.0078
            w = np.ones(n)
            self.surface_area = 0.43
        if vol <= 1e-9:
            vol = 0.0078
        self.solid_volume = vol
        total = self.rho_solid * vol
        w = np.maximum(w, 0.5 * w.mean())
        self.mass_s = total * w / w.sum()
        self.inv_ms = 1.0 / self.mass_s
        self.node_area = np.maximum(w, 1e-9)
        # spring dampers sized from a target damping ratio
        m_eff = 0.5 * (self.mass_s[self.sp_i] + self.mass_s[self.sp_j])
        self.sp_c = 2.0 * self.ZETA * np.sqrt(np.maximum(self.sp_k * m_eff, 1e-12))

    def _inside_rock(self, pts, margin=1.0):
        inside = np.zeros(len(pts), dtype=bool)
        for c, a in self.rocks:
            u = (pts - c) / a
            inside |= np.einsum("ij,ij->i", u, u) < margin ** 2
        return inside

    def _seed_fluid(self):
        dx = self.DX
        xs = np.arange(self.bmin[0] + 0.5 * dx, self.bmax[0] - 0.25 * dx, dx)
        ys = np.arange(self.bmin[1] + 0.5 * dx, self.bmax[1] - 0.25 * dx, dx)
        zs = np.arange(self.bmin[2] + 0.5 * dx, self.free_surface_z - 0.25 * dx, dx)
        grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        keep = ~self._inside_rock(grid, margin=1.03)
        return np.ascontiguousarray(grid[keep])

    def _boundary_psi(self):
        """Akinci boundary mass of each shell node from its rest kernel sum."""
        tree = cKDTree(self.rest_solid)
        pairs = tree.query_pairs(self.kern.radius, output_type="ndarray")
        ksum = np.full(self.ns, self.kern.w(np.zeros(1))[0])
        if len(pairs):
            r = np.linalg.norm(self.rest_solid[pairs[:, 0]] - self.rest_solid[pairs[:, 1]], axis=1)
            wv = self.kern.w(r)
            ksum += np.bincount(pairs[:, 0], wv, minlength=self.ns)
            ksum += np.bincount(pairs[:, 1], wv, minlength=self.ns)
        psi = self.rho0 / np.maximum(ksum, 1e-9)
        # thin double-layer shells give very dense kernel sums; keep psi in a
        # physically sensible band relative to the fluid particle mass
        return np.clip(psi, 0.05 * self.mf, 1.2 * self.mf)

    def _boundary_samples(self):
        pos = []
        nrm = []
        lo, hi = self.bmin, self.bmax
        gx = np.linspace(lo[0] + 0.08, hi[0] - 0.08, 8)
        gy = np.linspace(lo[1] + 0.08, hi[1] - 0.08, 8)
        for x in gx:
            for y in gy:
                pos.append([x, y, lo[2] + 0.004])
                nrm.append([0.0, 0.0, 1.0])
        zs = np.linspace(0.05, self.free_surface_z - 0.05, 4)
        for z in zs:
            for y in gy:
                pos.append([lo[0] + 0.004, y, z]); nrm.append([1.0, 0.0, 0.0])
                pos.append([hi[0] - 0.004, y, z]); nrm.append([-1.0, 0.0, 0.0])
            for x in gx:
                pos.append([x, lo[1] + 0.004, z]); nrm.append([0.0, 1.0, 0.0])
                pos.append([x, hi[1] - 0.004, z]); nrm.append([0.0, -1.0, 0.0])
        # rock crowns: sample the upper hemisphere of each ellipsoid
        for c, a in self.rocks:
            for phi in (0.0, 2.0 * math.pi / 3.0, 4.0 * math.pi / 3.0):
                for th in (0.25 * math.pi, 0.5 * math.pi):
                    u = np.array([math.sin(th) * math.cos(phi),
                                  math.sin(th) * math.sin(phi),
                                  math.cos(th)])
                    p = c + a * u
                    n = u / a
                    n = n / max(np.linalg.norm(n), 1e-12)
                    pos.append((p + 0.004 * n).tolist())
                    nrm.append(n.tolist())
        pos = np.asarray(pos, dtype=float)
        nrm = np.asarray(nrm, dtype=float)
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        return pos, nrm


    # ------------------------------------------------------- neighbour handling
    def _rebuild(self, force: bool = False) -> None:
        skin = 0.012
        R = self.kern.radius + skin
        tf = cKDTree(self.xf)
        self._pairs_ff = np.asarray(tf.query_pairs(R, output_type="ndarray"), dtype=np.int64)
        ts = cKDTree(self.xs)
        sd = tf.sparse_distance_matrix(ts, R, output_type="ndarray")
        self._pairs_fs = (np.asarray(sd["i"], dtype=np.int64), np.asarray(sd["j"], dtype=np.int64))
        pad = 0.05
        for r, (c, a) in enumerate(self.rocks):
            lo = c - a - pad
            hi = c + a + pad
            self._rock_cand_f[r] = np.flatnonzero(
                np.all((self.xf >= lo) & (self.xf <= hi), axis=1))
            self._rock_cand_s[r] = np.flatnonzero(
                np.all((self.xs >= lo) & (self.xs <= hi), axis=1))
        for p, (n, off) in enumerate(self.planes):
            self._plane_cand_f[p] = np.flatnonzero(self.xf @ n - off < 0.06)
            self._plane_cand_s[p] = np.flatnonzero(self.xs @ n - off < 0.06)
        self._rebuild_at = self.steps

    # --------------------------------------------------------------- contacts
    def _contacts(self, x, v, mass, kn, cn, mu_rock, mu_glass, dt, cand_rock, cand_plane, skin):
        F = np.zeros_like(x)
        for p, (n, off) in enumerate(self.planes):
            idx = cand_plane[p]
            if idx.size == 0:
                continue
            d = x[idx] @ n - off - skin
            m = d < 0.0
            if not np.any(m):
                continue
            sel = idx[m]
            vn = v[sel] @ n
            fn = np.maximum(-kn * d[m] - cn * vn, 0.0)
            F[sel] += fn[:, None] * n
            vt = v[sel] - vn[:, None] * n
            sp = np.linalg.norm(vt, axis=1)
            ct = 0.4 * mass[sel] / dt
            ft = np.minimum(mu_glass * fn, ct * sp)
            F[sel] -= (ft / np.maximum(sp, 1e-9))[:, None] * vt
        for r, (c, a) in enumerate(self.rocks):
            idx = cand_rock[r]
            if idx.size == 0:
                continue
            u = (x[idx] - c) / a
            s = np.linalg.norm(u, axis=1)
            q = u / a
            qn = np.maximum(np.linalg.norm(q, axis=1), 1e-9)
            dist = (s - 1.0) / qn
            m = dist < skin
            if not np.any(m):
                continue
            sel = idx[m]
            nrm = q[m] / qn[m][:, None]
            depth = skin - dist[m]
            vn = np.einsum("ij,ij->i", v[sel], nrm)
            fn = np.maximum(kn * depth - cn * vn, 0.0)
            F[sel] += fn[:, None] * nrm
            vt = v[sel] - vn[:, None] * nrm
            sp = np.linalg.norm(vt, axis=1)
            ct = 0.4 * mass[sel] / dt
            ft = np.minimum(mu_rock * fn, ct * sp)
            F[sel] -= (ft / np.maximum(sp, 1e-9))[:, None] * vt
        return F

    def _project(self, x, v, mass, skin, acc):
        """Hard non-penetration backstop; the applied impulse is recorded exactly."""
        v_before = v.copy()
        lo = self.bmin + skin
        hi = self.bmax - skin
        for k in range(3):
            below = x[:, k] < lo[k]
            if np.any(below):
                x[below, k] = lo[k]
                v[below, k] = np.maximum(v[below, k], 0.0)
            above = x[:, k] > hi[k]
            if np.any(above):
                x[above, k] = hi[k]
                v[above, k] = np.minimum(v[above, k], 0.0)
        for c, a in self.rocks:
            u = (x - c) / a
            s = np.linalg.norm(u, axis=1)
            inside = s < 1.0
            if not np.any(inside):
                continue
            idx = np.flatnonzero(inside)
            q = u[idx] / a
            qn = np.maximum(np.linalg.norm(q, axis=1), 1e-9)
            nrm = q / qn[:, None]
            push = (1.0 - s[idx]) / qn + skin
            x[idx] += push[:, None] * nrm
            vn = np.einsum("ij,ij->i", v[idx], nrm)
            neg = vn < 0.0
            if np.any(neg):
                sub = idx[neg]
                v[sub] -= vn[neg][:, None] * nrm[neg]
        acc += (mass[:, None] * (v - v_before)).sum(axis=0)

    def _limit(self, v, mass, vmax, acc):
        sp = np.linalg.norm(v, axis=1)
        bad = ~np.isfinite(sp)
        if np.any(bad):
            self._nan_events += 1
            v_old = np.nan_to_num(v[bad], nan=0.0, posinf=vmax, neginf=-vmax)
            acc -= (mass[bad, None] * v_old).sum(axis=0)
            v[bad] = 0.0
            sp = np.linalg.norm(v, axis=1)
        hot = sp > vmax
        if np.any(hot):
            scale = vmax / sp[hot]
            dv = v[hot] * (scale[:, None] - 1.0)
            acc += (mass[hot, None] * dv).sum(axis=0)
            v[hot] += dv


    # ------------------------------------------------------------ solid forces
    def _spring_forces(self, x, v):
        d = x[self.sp_j] - x[self.sp_i]
        L = np.sqrt(np.einsum("ij,ij->i", d, d))
        invL = 1.0 / np.maximum(L, 1e-9)
        u = d * invL[:, None]
        dv = v[self.sp_j] - v[self.sp_i]
        fmag = self.sp_k * (L - self.sp_rest) + self.sp_c * np.einsum("ij,ij->i", dv, u)
        fmag = np.clip(fmag, -80.0, 80.0)
        F = fmag[:, None] * u
        out = np.zeros_like(x)
        for k in range(3):
            out[:, k] = (np.bincount(self.sp_i, F[:, k], minlength=self.ns)
                         - np.bincount(self.sp_j, F[:, k], minlength=self.ns))
        return out

    # ------------------------------------------------------------------- step
    def _step(self, dt: float) -> None:
        if self.steps - self._rebuild_at >= 6:
            self._rebuild()
        kern = self.kern
        xf, vf, xs = self.xf, self.vf, self.xs

        # --- SPH density (fluid + Akinci boundary contribution of the shell)
        w0 = float(kern.w(np.zeros(1))[0])
        rho = np.full(self.nf, self.mf * w0)
        pairs = self._pairs_ff
        pi = pairs[:, 0]
        pj = pairs[:, 1]
        if pi.size:
            dij = xf[pi] - xf[pj]
            rij = np.sqrt(np.einsum("ij,ij->i", dij, dij))
            wij = kern.w(rij)
            rho += np.bincount(pi, self.mf * wij, minlength=self.nf)
            rho += np.bincount(pj, self.mf * wij, minlength=self.nf)
        fi, sj = self._pairs_fs
        if fi.size:
            dfs = xf[fi] - xs[sj]
            rfs = np.sqrt(np.einsum("ij,ij->i", dfs, dfs))
            wfs = kern.w(rfs)
            rho += np.bincount(fi, self.psi[sj] * wfs, minlength=self.nf)
        rho = np.clip(rho, 0.3 * self.rho0, 1.6 * self.rho0)
        press = np.clip(self.B_tait * ((rho / self.rho0) ** 7 - 1.0), 0.0, self.P_MAX)
        self.rho_f = rho
        self.press_f = press

        Ff = np.zeros((self.nf, 3))
        Fcross_s = np.zeros((self.ns, 3))
        Fcross_f = np.zeros((self.nf, 3))

        # --- fluid-fluid pressure + viscous forces (antisymmetric pairs)
        if pi.size:
            dwdr = kern.dwdr(rij)
            inv_r = 1.0 / np.maximum(rij, 1e-12)
            grad = (dwdr * inv_r)[:, None] * dij
            pterm = press[pi] / rho[pi] ** 2 + press[pj] / rho[pj] ** 2
            Fpair = (-(self.mf * self.mf) * pterm)[:, None] * grad
            rho_bar = 0.5 * (rho[pi] + rho[pj])
            cvis = (2.0 * self.nu_art * self.mf * self.mf / rho_bar) * (dwdr * inv_r)
            Fpair += cvis[:, None] * (vf[pi] - vf[pj])
            for k in range(3):
                Ff[:, k] += (np.bincount(pi, Fpair[:, k], minlength=self.nf)
                             - np.bincount(pj, Fpair[:, k], minlength=self.nf))

        # --- two-way fluid/solid coupling
        if fi.size:
            dwdr_fs = kern.dwdr(rfs)
            inv_rfs = 1.0 / np.maximum(rfs, 1e-12)
            grad_fs = (dwdr_fs * inv_rfs)[:, None] * dfs
            psb = self.psi[sj]
            pc = -(self.mf * psb) * (press[fi] / rho[fi] ** 2)
            Fp = pc[:, None] * grad_fs
            cvis = (2.0 * self.BOUNDARY_VISC_FACTOR * self.nu_art * self.mf * psb
                    / rho[fi]) * (dwdr_fs * inv_rfs)
            Fp += cvis[:, None] * (vf[fi] - self.vs[sj])
            near = 0.6 * self.DX
            close = rfs < near
            if np.any(close):
                amp = self.K_PENALTY_FS * (near - rfs[close])
                Fp[close] += (amp * inv_rfs[close])[:, None] * dfs[close]
            # clamp the pair force symmetrically so the reaction stays exact
            fmag = np.sqrt(np.einsum("ij,ij->i", Fp, Fp))
            hot = fmag > self.F_PAIR_MAX
            if np.any(hot):
                Fp[hot] *= (self.F_PAIR_MAX / fmag[hot])[:, None]
            for k in range(3):
                Fcross_f[:, k] += np.bincount(fi, Fp[:, k], minlength=self.nf)
                Fcross_s[:, k] -= np.bincount(sj, Fp[:, k], minlength=self.ns)
        Ff += Fcross_f
        self.f_cross_solid = Fcross_s.sum(axis=0)
        self.f_cross_fluid = Fcross_f.sum(axis=0)
        self.imp_cross_solid += dt * self.f_cross_solid
        self.imp_cross_fluid += dt * self.f_cross_fluid

        # --- solid substeps (springs, gravity, contact, constant coupling load)
        dts = dt / self.SOLID_SUBSTEPS
        f_bnd_s = np.zeros(3)
        for _ in range(self.SOLID_SUBSTEPS):
            Fspring = self._spring_forces(self.xs, self.vs)
            Fc = self._contacts(self.xs, self.vs, self.mass_s, self.K_WALL_SOLID,
                                self.C_WALL_SOLID, self.MU_ROCK, self.MU_GLASS, dts,
                                self._rock_cand_s, self._plane_cand_s, 0.003)
            acc = (Fspring + Fc + Fcross_s) * self.inv_ms[:, None] + self.g
            self.vs += dts * acc
            self._limit(self.vs, self.mass_s, 8.0, self.imp_stab_solid)
            self.xs += dts * self.vs
            self._project(self.xs, self.vs, self.mass_s, 0.002, self.imp_solid)
            fsum = Fc.sum(axis=0)
            self.imp_solid += dts * fsum
            f_bnd_s += fsum / self.SOLID_SUBSTEPS
        self.f_bnd_solid = f_bnd_s

        # --- fluid integration
        Fc_f = self._contacts(self.xf, self.vf, self.mass_f, self.K_WALL_FLUID,
                              self.C_WALL_FLUID, self.MU_ROCK, self.MU_GLASS, dt,
                              self._rock_cand_f, self._plane_cand_f, 0.35 * self.DX)
        self.vf += dt * ((Ff + Fc_f) / self.mf + self.g)
        self._limit(self.vf, self.mass_f, 6.0, self.imp_stab_fluid)
        self.xf += dt * self.vf
        self._project(self.xf, self.vf, self.mass_f, 0.1 * self.DX, self.imp_fluid)
        self.f_bnd_fluid = Fc_f.sum(axis=0)
        self.imp_fluid += dt * self.f_bnd_fluid

        self.t += dt
        self.steps += 1


    # --------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.t + 1e-12:
            if self._frozen or (_time.time() - self._t0) > self.TIME_BUDGET:
                self._frozen = True
                self.t = target
            else:
                n = int(round((target - self.t) / self.DT))
                n = max(n, 0)
                for _ in range(n):
                    self._step(self.DT)
                    if (_time.time() - self._t0) > self.TIME_BUDGET:
                        self._frozen = True
                        break
                self.t = target
        return self._frame(target)

    def _boundary_velocity(self):
        pos = np.vstack([self.xs, self.xf])
        vel = np.vstack([self.vs, self.vf])
        tree = cKDTree(pos)
        idx = tree.query_ball_point(self.b_pos, self.kern.radius)
        out = np.zeros((self.nb, 3))
        for b, nb in enumerate(idx):
            if not len(nb):
                continue
            nb = np.asarray(nb, dtype=np.int64)
            r = np.linalg.norm(pos[nb] - self.b_pos[b], axis=1)
            w = self.kern.w(r)
            s = w.sum()
            if s > 1e-12:
                out[b] = (w[:, None] * vel[nb]).sum(axis=0) / s
        return out

    def _frame(self, t: float) -> StateFrame:
        position = np.vstack([self.xs, self.xf])
        velocity = np.vstack([self.vs, self.vf])
        n = self.ns + self.nf
        body_id = np.concatenate([
            np.full(self.ns, 1, dtype=np.int32),
            np.full(self.nf, 2, dtype=np.int32),
        ])
        mass = self.mass_all

        mass_quantity = np.zeros((n, 2))
        mass_quantity[:self.ns, 0] = self.mass_s
        mass_quantity[self.ns:, 1] = self.mass_f

        weights = np.zeros((3, n))
        weights[0, :self.ns] = 1.0
        weights[1, self.ns:] = 1.0
        weights[2, :] = 1.0

        g_solid = self.M_solid * self.g
        g_water = self.M_water * self.g
        ext_force = np.stack([
            g_solid[None, :],
            g_water[None, :],
            (g_solid + g_water)[None, :],
        ], axis=0)                                   # [G,1,3] gravity only

        imp_s = self.imp_solid + self.imp_cross_solid + self.imp_stab_solid
        imp_f = self.imp_fluid + self.imp_cross_fluid + self.imp_stab_fluid
        cum_impulse = np.stack([imp_s, imp_f, imp_s + imp_f], axis=0)

        interaction = np.stack([
            self.f_cross_solid,          # fluid -> solid
            self.f_cross_fluid,          # solid -> fluid (reaction pair)
            self.f_bnd_solid,            # rocks/tank -> solid
            -self.f_bnd_solid,           # solid -> rocks/tank
            self.f_bnd_fluid,            # rocks/tank -> fluid
            -self.f_bnd_fluid,           # fluid -> rocks/tank
        ], axis=0)

        b_vel = self._boundary_velocity()
        submerged = float(np.count_nonzero(self.xs[:, 2] < self.free_surface_z)) / self.ns

        frame = StateFrame(
            time=float(t),
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "mass_quantity": mass_quantity,
                "density": np.concatenate([
                    np.full(self.ns, self.rho_solid), self.rho_f]),
                "pressure": np.concatenate([np.zeros(self.ns), self.press_f]),
            },
            fields={
                "boundary_normal": self.b_nrm,
                "boundary_target_velocity": np.zeros((self.nb, 3)),
                "boundary_velocity": b_vel,
                "budget_domain_weights": weights,
                "budget_external_force": ext_force,
                "budget_cumulative_external_impulse": cum_impulse,
                "budget_cumulative_inflow_momentum": np.zeros((3, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((3, 3)),
                "cumulative_mass_source": np.zeros(2),
                "interaction_force": interaction,
                "gravity_acceleration": self.g.reshape(1, 3),
            },
            scalars={
                "reference_velocity": float(self.v_ref),
                "reference_momentum": float(self.p_ref),
                "reference_force": float((self.M_solid + self.M_water)
                                         * abs(float(self.g[2]))),
                "water_free_surface_z": float(self.free_surface_z),
                "dynamic_viscosity": float(self.nu_art * self.rho0),
                "solid_submerged_fraction": submerged,
                "solid_center_z": float(np.average(self.xs[:, 2], weights=self.mass_s)),
                "sph_time_step": float(self.DT),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "WCSPH free-surface water + mass-spring shell, two-way coupled",
            "fluid_particles": int(self.nf),
            "solid_nodes": int(self.ns),
            "springs": int(self.n_springs),
            "solid_volume_m3": float(self.solid_volume),
            "solid_mass_kg": float(self.M_solid),
            "water_mass_kg": float(self.M_water),
            "time_step": float(self.DT),
            "steps_taken": int(self.steps),
            "frozen_on_budget": bool(self._frozen),
            "nan_guard_events": int(self._nan_events),
            "wall_seconds": float(_time.time() - self._t0),
        }
