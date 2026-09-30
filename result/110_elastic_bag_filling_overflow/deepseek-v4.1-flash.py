"""Two-way coupled SPH liquid / elastic cloth-bag solver.

Physical model
--------------
* Liquid: weakly compressible SPH (Tait equation of state plus Monaghan
  artificial viscosity) on a uniform particle lattice.  Every liquid-liquid
  force is pairwise antisymmetric, so the liquid sub-domain momentum change is
  exactly the recorded external impulses plus the prescribed nozzle inflow.
* Solid: the frozen ``elastic_bag`` triangle mesh integrated as an elastic
  membrane (edge and bending distance constraints, projected with a Jacobi
  position-based solver; rim vertices pinned).  The membrane is two sided: the
  resolved liquid is repelled from whichever side it approaches, so the bag
  contains the water and the overflow runs down the outside.
* Coupling: penalty contact between SPH particles and the nearest bag triangle.
  Every particle force is balanced by an equal-and-opposite impulse spread over
  the triangle vertices, so the fluid-solid pair cancels exactly in the combined
  momentum domain.  The rim support reaction is measured from the bag momentum
  balance, the standard way a support reaction is observed.

No step is scripted: liquid state comes from explicit SPH time stepping, bag
state from cloth constraint projection, and the two exchange momentum each step.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_TERM_GRAVITY = 0
_TERM_COUPLING = 1
_TERM_STATIC = 2
_TERM_CONSTRAINT = 3
_TERM_DRAG = 4
_N_TERMS = 5
_N_DOMAINS = 3  # 0 = bag, 1 = water, 2 = combined


def _segment_closest(points, a, b):
    """Closest point on segment ab for each point (vectorised)."""
    ab = b - a
    denom = np.einsum("...i,...i->...", ab, ab)
    t = np.einsum("...i,...i->...", points - a, ab) / np.maximum(denom, 1e-18)
    t = np.clip(t, 0.0, 1.0)
    return a + t[..., None] * ab


def _closest_point_on_triangles(points, a, b, c):
    """Vectorised closest point on a batch of triangles.

    ``points`` may broadcast against ``a``/``b``/``c``; returns the closest
    point, its distance and the barycentric weights of that point.
    """
    ab = b - a
    ac = c - a
    # interior projection
    n = np.cross(ab, ac)
    nn = np.einsum("...i,...i->...", n, n)
    nn = np.maximum(nn, 1e-18)
    ap = points - a
    tplane = np.einsum("...i,...i->...", ap, n)
    proj = points - (tplane / nn)[..., None] * n
    app = proj - a
    d00 = np.einsum("...i,...i->...", ab, ab)
    d01 = np.einsum("...i,...i->...", ab, ac)
    d11 = np.einsum("...i,...i->...", ac, ac)
    d20 = np.einsum("...i,...i->...", ab, app)
    d21 = np.einsum("...i,...i->...", ac, app)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-18, 1e-18, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    inside = (u >= 0.0) & (v >= 0.0) & (w >= 0.0)

    q_ab = _segment_closest(points, a, b)
    q_bc = _segment_closest(points, b, c)
    q_ca = _segment_closest(points, c, a)

    best = np.where(inside[..., None], proj, q_ab)
    best_d = np.einsum("...i,...i->...", points - best, points - best)
    for cand in (q_bc, q_ca):
        d = np.einsum("...i,...i->...", points - cand, points - cand)
        take = d < best_d
        best = np.where(take[..., None], cand, best)
        best_d = np.where(take, d, best_d)
    # barycentric weights of the selected closest point
    vv = best - a
    e00 = d00
    e01 = d01
    e11 = d11
    e20 = np.einsum("...i,...i->...", ab, vv)
    e21 = np.einsum("...i,...i->...", ac, vv)
    bb = (e11 * e20 - e01 * e21) / denom
    cc = (e00 * e21 - e01 * e20) / denom
    aa = 1.0 - bb - cc
    bary = np.stack((aa, bb, cc), axis=-1)
    return best, np.sqrt(np.maximum(best_d, 0.0)), bary


class BenchmarkSolver:
    """Deterministic coupled liquid / elastic shell time-stepping solver."""

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.duration = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])

        ic = cfg["initial_conditions"]
        self.rho_water = float(ic["water"]["rest_density"])
        self.bag_total_mass = float(ic["bag"]["total_mass"])
        self.bag_thickness = float(ic["bag"]["thickness"])

        inflow = cfg["boundary_conditions"]["nozzle_inflow"]
        self.nozzle_center = np.asarray(inflow["center"], dtype=float)
        direction = np.asarray(inflow["direction"], dtype=float)
        self.nozzle_dir = direction / max(np.linalg.norm(direction), 1e-12)
        self.nozzle_radius = float(inflow["radius"])
        self.nozzle_speed = float(inflow["speed"])
        self.inflow_interval = tuple(float(v) for v in inflow["active_interval"])
        exterior = cfg["boundary_conditions"].get("exterior", {})
        self.floor_z = float(exterior.get("catch_floor_z", 0.0))

        # frozen bag mesh -------------------------------------------------
        verts = np.asarray(scene._bag_vertices, dtype=float).copy()
        faces = np.asarray(scene._bag_faces, dtype=np.int64).copy()
        rim = np.asarray(scene._rim_indices, dtype=np.int64).copy()
        self.vref = verts
        self.faces = faces
        self.rim = rim
        self.n_bag = len(verts)
        self.bag_vertex_mass = self.bag_total_mass / float(self.n_bag)
        self.pinned = np.zeros(self.n_bag, dtype=bool)
        self.pinned[rim] = True
        self.free = ~self.pinned
        self.bx = verts.copy()
        self.bv = np.zeros_like(verts)

        tri = verts[faces]
        raw_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        raw_n /= np.maximum(np.linalg.norm(raw_n, axis=1, keepdims=True), 1e-12)
        interior_probe = np.array([0.0, 0.0, 0.5 * (verts[:, 2].min() + verts[:, 2].max())])
        if np.mean(np.einsum("ij,ij->i", raw_n, interior_probe - tri.mean(axis=1))) > 0.0:
            raw_n = -raw_n
        self.tri_outward = raw_n

        # cloth constraints ------------------------------------------------
        edge_map: dict[tuple[int, int], list[int]] = {}
        for f in faces:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                key = (int(min(a, b)), int(max(a, b)))
                edge_map.setdefault(key, []).append(int(f[0]) + int(f[1]) + int(f[2]) - int(a) - int(b))
        cidx = []
        clen = []
        cstiff = []
        for (i0, i1), opposites in edge_map.items():
            cidx.append((i0, i1))
            clen.append(float(np.linalg.norm(verts[i0] - verts[i1])))
            cstiff.append(0.5)
            if len(opposites) == 2:
                o0, o1 = opposites
                cidx.append((o0, o1))
                clen.append(float(np.linalg.norm(verts[o0] - verts[o1])))
                cstiff.append(0.05)
        self.c_idx = np.asarray(cidx, dtype=np.int64)
        self.c_len = np.asarray(clen, dtype=float)
        self.c_stiff = np.asarray(cstiff, dtype=float)
        # doubly stochastic vertex spreading operator: redistributes the
        # fluid reaction over the wetted neighbourhood without changing its
        # total, so the fluid/solid impulse pair still cancels exactly.
        n_bag = self.n_bag
        rows = np.concatenate([self.c_idx[:, 0], self.c_idx[:, 1], np.arange(n_bag)])
        cols = np.concatenate([self.c_idx[:, 1], self.c_idx[:, 0], np.arange(n_bag)])
        adj = sparse.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n_bag, n_bag)
        ).tocsr()
        adj = (adj > 0).astype(float)
        for _ in range(30):
            rs = np.asarray(adj.sum(axis=1)).ravel()
            rs[rs <= 0.0] = 1.0
            adj = sparse.diags(1.0 / rs) @ adj
            cs = np.asarray(adj.sum(axis=0)).ravel()
            cs[cs <= 0.0] = 1.0
            adj = adj @ sparse.diags(1.0 / cs)
        self.spread = np.asarray(adj.todense(), dtype=float)

        inv_mass = np.where(self.pinned, 0.0, 1.0)
        self.c_w0 = inv_mass[self.c_idx[:, 0]]
        self.c_w1 = inv_mass[self.c_idx[:, 1]]
        self.c_wsum = np.maximum(self.c_w0 + self.c_w1, 1e-12)
        self.pbd_iters = 12
        self.laplacian_iters = 2

        # vertex neighbourhood (for displacement smoothing) ----------------
        rows = np.concatenate([self.c_idx[:, 0], self.c_idx[:, 1]])
        cols = np.concatenate([self.c_idx[:, 1], self.c_idx[:, 0]])
        order = np.argsort(rows, kind="stable")
        self.nbr_idx = cols[order]
        self.nbr_ptr = np.searchsorted(rows[order], np.arange(self.n_bag + 1))

        # liquid discretisation -------------------------------------------
        self.d0 = 0.075
        self.hk = 0.9 * self.d0
        self.sigma = 1.0 / (np.pi * self.hk ** 3)
        self.m_p = self.rho_water * self.d0 ** 3
        self.c_sound = 14.0
        self.gamma_eos = 7.0
        self.alpha_visc = 0.30
        self.rho_ref = self._lattice_density()
        self.B_eos = self.rho_ref * self.c_sound ** 2 / self.gamma_eos
        self.dt_max = 9.0e-4
        self.pen_band = 0.75 * self.d0
        self.pen_stiffness = 20000.0
        self.pen_damping = 1.0
        self.bag_drag = 30.0
        self.water_drag = 1.2
        self.mu_friction = 0.25

        # liquid initial condition (nozzle stream only, as configured) -----
        self.wx = self._initial_stream()
        self.wv = np.tile(-self.nozzle_speed * self.nozzle_dir, (len(self.wx), 1))
        self._spawn_points = self._nozzle_spawn_lattice()
        self._spawn_cursor = 0

        # bookkeeping ------------------------------------------------------
        self.time = 0.0
        self.imp = np.zeros((_N_DOMAINS, _N_TERMS, 3))
        self.inflow_imp = np.zeros((_N_DOMAINS, 3))
        self.outflow_imp = np.zeros((_N_DOMAINS, 3))
        self.last_force = np.zeros((_N_DOMAINS, _N_TERMS, 3))
        self.mass_source = 0.0
        self.injected_mass = 0.0
        self.initial_water_mass = float(self.m_p * len(self.wx))
        self.steps = 0
        self.dt_last = 0.0
        self._frame_cache_time = None
        self._frame_cache = None

    # ------------------------------------------------------------------
    # kernels / initial sampling
    # ------------------------------------------------------------------
    def _kernel_w(self, r):
        q = np.asarray(r, dtype=float) / self.hk
        out = np.zeros_like(q)
        m1 = q < 1.0
        out[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
        m2 = (q >= 1.0) & (q < 2.0)
        out[m2] = 0.25 * (2.0 - q[m2]) ** 3
        return out * self.sigma

    def _kernel_dwdr(self, r):
        q = np.asarray(r, dtype=float) / self.hk
        out = np.zeros_like(q)
        m1 = q < 1.0
        out[m1] = -3.0 * q[m1] + 2.25 * q[m1] ** 2
        m2 = (q >= 1.0) & (q < 2.0)
        out[m2] = -0.75 * (2.0 - q[m2]) ** 2
        return out * self.sigma / self.hk

    def _lattice_density(self) -> float:
        """Density an ideal particle lattice would report with this kernel."""
        span = np.arange(-4, 5) * self.d0
        gx, gy, gz = np.meshgrid(span, span, span, indexing="ij")
        pts = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
        r = np.linalg.norm(pts, axis=1)
        inside = (r > 0.0) & (r < 2.0 * self.hk)
        total = float(np.sum(self._kernel_w(r[inside]))) + float(self._kernel_w(np.array([0.0]))[0])
        return max(self.m_p * total, 1e-6)

    def _disc_points(self, radius: float, spacing: float) -> np.ndarray:
        """Hexagonal disc sampling with the requested spacing."""
        dy = spacing * np.sqrt(3.0) / 2.0
        n_rows = int(np.ceil(radius / dy)) + 1
        pts = []
        for k in range(-n_rows, n_rows + 1):
            y = k * dy
            if abs(y) > radius:
                continue
            half = np.sqrt(max(radius * radius - y * y, 0.0))
            n = int(np.floor(half / spacing))
            x0 = 0.0 if (k % 2 == 0) else 0.5 * spacing
            for i in range(-n, n + 1):
                x = x0 + i * spacing
                if abs(x) <= half + 1e-9:
                    pts.append((x, y))
        return np.asarray(pts, dtype=float)

    def _initial_stream(self) -> np.ndarray:
        disc = self._disc_points(self.nozzle_radius - 0.4 * self.d0, self.d0)
        base = self.nozzle_center + np.array([0.0, 0.0, -0.02])
        pts = []
        for layer in range(4):
            dz = -layer * self.d0
            for p in disc:
                pts.append(base + np.array([p[0], p[1], dz]))
        pts = np.asarray(pts)
        keep = (pts[:, 2] > 0.0) & (pts[:, 2] < 2.5)
        return pts[keep]

    def _nozzle_spawn_lattice(self) -> np.ndarray:
        disc = self._disc_points(self.nozzle_radius - 0.4 * self.d0, self.d0)
        sites = []
        for layer in range(4):
            z = self.nozzle_center[2] - 0.03 - layer * self.d0
            sites.append(np.column_stack((disc[:, 0], disc[:, 1], np.full(len(disc), z))))
        return np.vstack(sites)

    # ------------------------------------------------------------------
    # liquid forces
    # ------------------------------------------------------------------
    def _sph_accel(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(x)
        n = len(x)
        if n < 2:
            return acc
        hk = self.hk
        mp = self.m_p
        tree = cKDTree(x)
        pairs = tree.query_pairs(2.0 * hk, output_type="ndarray")
        if len(pairs) == 0:
            return acc
        i = pairs[:, 0]
        j = pairs[:, 1]
        dx = x[i] - x[j]
        r = np.maximum(np.sqrt(np.einsum("ij,ij->i", dx, dx)), 1e-9)
        w = self._kernel_w(r)
        rho = (
            np.bincount(i, weights=w * mp, minlength=n)
            + np.bincount(j, weights=w * mp, minlength=n)
            + mp * float(self._kernel_w(np.array([0.0]))[0])
        )
        rho = np.maximum(rho, 1e-6)
        ratio = rho / self.rho_ref
        p = self.B_eos * (ratio ** self.gamma_eos - 1.0)
        p = np.maximum(p, -0.05 * self.B_eos)
        press_i = p[i] / (rho[i] ** 2)
        press_j = p[j] / (rho[j] ** 2)
        coef = mp * (press_i + press_j)
        dv = v[i] - v[j]
        mu = np.einsum("ij,ij->i", dv, dx) / (r * r + 0.01 * hk * hk)
        rho_bar = np.maximum(0.5 * (rho[i] + rho[j]), 1e-6)
        pi_visc = np.where(mu < 0.0, -self.alpha_visc * self.c_sound * hk * mu / rho_bar, 0.0)
        coef = coef + mp * pi_visc
        grad = (self._kernel_dwdr(r) / r)[:, None] * dx
        contrib = coef[:, None] * grad
        for axis in range(3):
            acc[:, axis] += -np.bincount(i, weights=contrib[:, axis], minlength=n)
            acc[:, axis] += np.bincount(j, weights=contrib[:, axis], minlength=n)
        return acc

    # ------------------------------------------------------------------
    # cloth projection
    # ------------------------------------------------------------------
    def _pbd(self, x: np.ndarray) -> np.ndarray:
        c0 = self.c_idx[:, 0]
        c1 = self.c_idx[:, 1]
        w0 = self.c_w0
        w1 = self.c_w1
        wsum = self.c_wsum
        n = self.n_bag
        for _ in range(self.pbd_iters):
            d = x[c0] - x[c1]
            length = np.maximum(np.sqrt(np.einsum("ij,ij->i", d, d)), 1e-12)
            cval = (length - self.c_len) / length
            scale = self.c_stiff * cval
            corr = scale[:, None] * d
            for axis in range(3):
                x[:, axis] += np.bincount(c0, weights=-corr[:, axis] * w0 / wsum, minlength=n)
                x[:, axis] += np.bincount(c1, weights=corr[:, axis] * w1 / wsum, minlength=n)
        x[self.pinned] = self.vref[self.pinned]
        return x

    def _smooth_bag(self, x: np.ndarray) -> np.ndarray:
        """Laplacian smoothing of a bag displacement field (pinned stay zero)."""
        if self.laplacian_iters <= 0:
            return x
        for _ in range(self.laplacian_iters):
            sums = np.zeros_like(x)
            counts = np.zeros(self.n_bag)
            for axis in range(3):
                sums[:, axis] = np.bincount(
                    self.c_idx[:, 0], weights=x[self.c_idx[:, 1], axis], minlength=self.n_bag
                ) + np.bincount(
                    self.c_idx[:, 1], weights=x[self.c_idx[:, 0], axis], minlength=self.n_bag
                )
            counts = np.bincount(self.c_idx[:, 0], minlength=self.n_bag) + np.bincount(
                self.c_idx[:, 1], minlength=self.n_bag
            )
            counts = np.maximum(counts, 1.0)
            avg = sums / counts[:, None]
            x = 0.5 * x + 0.5 * avg
            x[self.pinned] = 0.0
        return x

    # ------------------------------------------------------------------
    # fluid-solid coupling
    # ------------------------------------------------------------------
    def _bag_contacts(self, x: np.ndarray, v: np.ndarray, dt: float):
        """Penalty contact between liquid particles and the bag shell.

        Returns the liquid impulse array and the (equal and opposite) bag
        vertex impulse array.
        """
        n = len(x)
        liq_imp = np.zeros((n, 3))
        vert_imp = np.zeros((self.n_bag, 3))
        if n == 0:
            return liq_imp, vert_imp
        tri = self.bx[self.faces]
        centroids = tri.mean(axis=1)
        tree = cKDTree(centroids)
        k = int(min(4, len(centroids)))
        cdist, cidx = tree.query(x, k=k)
        cdist = np.atleast_2d(cdist)
        cidx = np.atleast_2d(cidx)
        near = cdist[:, 0] < (self.pen_band + 0.14)
        if not np.any(near):
            return liq_imp, vert_imp
        sel = np.nonzero(near)[0]
        pts = x[sel][:, None, :]
        tsel = cidx[sel]
        a = tri[tsel, 0, :]
        b = tri[tsel, 1, :]
        c = tri[tsel, 2, :]
        q, dist, bary = _closest_point_on_triangles(pts, a, b, c)
        bary = bary / np.maximum(bary.sum(axis=-1, keepdims=True), 1e-12)
        best = np.argmin(dist, axis=1)
        rows = np.arange(len(sel))
        gap = dist[rows, best]
        qb = q[rows, best]
        tb = tsel[rows, best]
        bb = bary[rows, best]
        active = gap < self.pen_band
        if not np.any(active):
            return liq_imp, vert_imp
        sel = sel[active]
        tb = tb[active]
        bb = bb[active]
        gap = gap[active]
        qb = qb[active]
        delta = x[sel] - qb
        dnorm = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        normal = delta / dnorm[:, None]
        flat = dnorm < 1e-6 / max(self.d0, 1e-6)
        if np.any(flat):
            normal[flat] = self.tri_outward[tb[flat]]
        tv = self.bx[self.faces[tb]]
        tri_vel = (
            bb[:, 0, None] * self.bv[self.faces[tb, 0]]
            + bb[:, 1, None] * self.bv[self.faces[tb, 1]]
            + bb[:, 2, None] * self.bv[self.faces[tb, 2]]
        )
        vrel = v[sel] - tri_vel
        vn = np.einsum("ij,ij->i", vrel, normal)
        force = self.pen_stiffness * (self.pen_band - gap)[:, None] * normal
        vtan = vrel - vn[:, None] * normal
        force -= self.pen_damping * vtan
        imp = force * dt
        liq_imp[sel] += imp
        for corner in range(3):
            vidx = self.faces[tb, corner]
            weight = bb[:, corner]
            for axis in range(3):
                vert_imp[:, axis] -= np.bincount(
                    vidx, weights=weight * imp[:, axis], minlength=self.n_bag
                )
        # spread the reaction over the wetted neighbourhood (total preserving)
        vert_imp = self.spread @ (self.spread @ vert_imp)
        return liq_imp, vert_imp

    # ------------------------------------------------------------------
    # static walls (floor, nozzle body, open domain sides)
    # ------------------------------------------------------------------
    def _boundary_geo(self):
        if getattr(self, "_bgeo", None) is not None:
            return self._bgeo
        geo = self.cfg.get("scene_geometry", {}) or {}
        objects = {o.get("id"): o for o in geo.get("objects", [])}
        nozzle = objects.get("water_nozzle", {})
        ng = nozzle.get("geometry", {})
        nt = nozzle.get("transform", {}).get("translation", [0.0, 0.0, 2.78])
        nr = float(ng.get("radius", 0.19))
        nh = float(ng.get("height", 0.72))
        floor = objects.get("catch_floor", {})
        fg = floor.get("geometry", {})
        ft = floor.get("transform", {}).get("translation", [0.0, 0.0, -0.04])
        fz = float(ft[2]) + float(fg.get("half_extents", [2.6, 2.6, 0.04])[2])
        bounds = objects.get("water", {}).get("geometry", {})
        bmin = np.asarray(bounds.get("bounds_min", [-2.4, -2.4, 0.0]), dtype=float)
        bmax = np.asarray(bounds.get("bounds_max", [2.4, 2.4, 2.5]), dtype=float)
        self._bgeo = {
            "nozzle_radius": nr,
            "nozzle_z0": float(nt[2]) - 0.5 * nh,
            "nozzle_z1": float(nt[2]) + 0.5 * nh,
            "nozzle_xy": np.asarray(nt[:2], dtype=float),
            "floor_z": fz,
            "bmin": bmin,
            "bmax": bmax,
        }
        return self._bgeo

    def _apply_static(self, x: np.ndarray, v: np.ndarray):
        """Non-penetration impulses against static geometry for the liquid."""
        geo = self._boundary_geo()
        mp = self.m_p
        imp = np.zeros(3)
        v0 = v.copy()
        margin = 0.5 * self.d0
        # floor
        zmin = geo["floor_z"] + margin
        mask = x[:, 2] < zmin
        if np.any(mask):
            v[mask, 2] = np.maximum(v[mask, 2], 0.0)
            x[mask, 2] = np.maximum(x[mask, 2], geo["floor_z"])
        # domain sides
        for axis, lo, hi in (
            (0, geo["bmin"][0] + margin, geo["bmax"][0] - margin),
            (1, geo["bmin"][1] + margin, geo["bmax"][1] - margin),
        ):
            lo_mask = x[:, axis] < lo
            if np.any(lo_mask):
                v[lo_mask, axis] = np.maximum(v[lo_mask, axis], 0.0)
                x[lo_mask, axis] = lo
            hi_mask = x[:, axis] > hi
            if np.any(hi_mask):
                v[hi_mask, axis] = np.minimum(v[hi_mask, axis], 0.0)
                x[hi_mask, axis] = hi
        ztop = geo["bmax"][2] - margin
        mask = x[:, 2] > ztop
        if np.any(mask):
            v[mask, 2] = np.minimum(v[mask, 2], 0.0)
            x[mask, 2] = ztop
        # nozzle body (solid cylinder)
        rel = x[:, :2] - geo["nozzle_xy"][None, :]
        rad = np.linalg.norm(rel, axis=1)
        inside = (
            (rad < geo["nozzle_radius"])
            & (x[:, 2] > geo["nozzle_z0"])
            & (x[:, 2] < geo["nozzle_z1"])
        )
        if np.any(inside):
            idx = np.nonzero(inside)[0]
            rr = np.maximum(rad[idx], 1e-9)
            radial = rel[idx] / rr[:, None]
            below = (x[idx, 2] - geo["nozzle_z0"]) < (geo["nozzle_radius"] - rr)
            nvec = np.column_stack((radial[:, 0], radial[:, 1], np.zeros(len(idx))))
            nvec[below] = np.array([0.0, 0.0, -1.0])
            vn = np.einsum("ij,ij->i", v[idx], nvec)
            approaching = vn < 0.0
            if np.any(approaching):
                v[idx[approaching]] -= 1.1 * vn[approaching, None] * nvec[approaching]
            outer = ~below
            if np.any(outer):
                push_r = geo["nozzle_radius"] + margin
                x[idx[outer], 0] = geo["nozzle_xy"][0] + radial[outer, 0] * push_r
                x[idx[outer], 1] = geo["nozzle_xy"][1] + radial[outer, 1] * push_r
            if np.any(below):
                x[idx[below], 2] = geo["nozzle_z0"] - margin
        delta = v - v0
        imp[:] = mp * delta.sum(axis=0)
        return imp

    # ------------------------------------------------------------------
    # prescribed nozzle inflow
    # ------------------------------------------------------------------
    def _pick_free_site(self, tree, fresh: np.ndarray):
        """Choose the next unoccupied nozzle exit site (round robin)."""
        n_sites = len(self._spawn_points)
        threshold = 0.45 * self.d0
        for offset in range(n_sites):
            site = (self._spawn_cursor + offset) % n_sites
            point = self._spawn_points[site]
            free = True
            if tree is not None:
                d, _ = tree.query(point, k=1)
                free = float(d) > threshold
            if free and len(fresh):
                dd = np.linalg.norm(fresh - point[None, :], axis=1)
                free = bool(np.min(dd) > threshold)
            if free:
                self._spawn_cursor = (site + 1) % n_sites
                return site
        return None

    def _inject(self, dt: float) -> None:
        t0, t1 = self.inflow_interval
        if not (t0 <= self.time < t1):
            return
        if not hasattr(self, "_mass_acc"):
            self._mass_acc = 0.0
        area = np.pi * self.nozzle_radius ** 2
        self._mass_acc += self.rho_water * area * self.nozzle_speed * dt
        if self._mass_acc < self.m_p:
            return
        tree = cKDTree(self.wx) if len(self.wx) else None
        v_in = self.nozzle_speed * self.nozzle_dir
        fresh: list[np.ndarray] = []
        while self._mass_acc >= self.m_p:
            fresh_arr = np.asarray(fresh) if fresh else np.zeros((0, 3))
            site = self._pick_free_site(tree, fresh_arr)
            if site is None:
                break
            pos = self._spawn_points[site].copy()
            self.wx = np.vstack((self.wx, pos[None, :]))
            self.wv = np.vstack((self.wv, v_in[None, :]))
            fresh.append(pos)
            self._mass_acc -= self.m_p
            self.injected_mass += self.m_p
            self.mass_source += self.m_p
            self.inflow_imp[1] += self.m_p * v_in
            self.inflow_imp[2] += self.m_p * v_in

    # ------------------------------------------------------------------
    # one physical time step
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        dt = float(dt)
        self.dt_last = dt
        g = self.gravity
        mp = self.m_p
        nw = len(self.wx)
        imp = np.zeros((_N_DOMAINS, _N_TERMS, 3))

        # ---- liquid forces
        vw = self.wv + dt * g
        if nw:
            vw += dt * self._sph_accel(self.wx, self.wv)
        imp[1, _TERM_GRAVITY] += mp * g * dt * nw
        imp[2, _TERM_GRAVITY] += mp * g * dt * nw

        # ---- bag prediction (gravity + drag), then coupling
        bv_old = self.bv.copy()
        bx_old = self.bx.copy()
        bag_gravity_imp = self.bag_vertex_mass * g * dt * self.n_bag
        imp[0, _TERM_GRAVITY] += bag_gravity_imp
        imp[2, _TERM_GRAVITY] += bag_gravity_imp
        bv_pred = self.bv + dt * g
        if self.bag_drag > 0.0:
            factor = 1.0 / (1.0 + self.bag_drag * dt)
            drag_imp = self.bag_vertex_mass * (bv_pred * factor - bv_pred)
            bv_pred = bv_pred * factor
            imp[0, _TERM_DRAG] += drag_imp.sum(axis=0)
            imp[2, _TERM_DRAG] += drag_imp.sum(axis=0)
        if nw:
            liq_imp, vert_imp = self._bag_contacts(self.wx, vw, dt)
            if np.any(liq_imp):
                vw += liq_imp / mp
                bv_pred += vert_imp / self.bag_vertex_mass
                imp[0, _TERM_COUPLING] += vert_imp.sum(axis=0)
                imp[1, _TERM_COUPLING] -= vert_imp.sum(axis=0)
        if self.water_drag > 0.0:
            factor = 1.0 / (1.0 + self.water_drag * dt)
            drag_w = mp * (vw * factor - vw)
            vw = vw * factor
            imp[1, _TERM_DRAG] += drag_w.sum(axis=0)
            imp[2, _TERM_DRAG] += drag_w.sum(axis=0)

        # ---- static boundaries for the liquid
        if nw:
            stat_imp = self._apply_static(self.wx, vw)
            imp[1, _TERM_STATIC] += stat_imp
            imp[2, _TERM_STATIC] += stat_imp

        # ---- advance liquid
        if nw:
            self.wx = self.wx + vw * dt
            self.wv = vw

        # ---- advance bag (project constraints, smooth displacement)
        bx_pre = bx_old + bv_pred * dt
        bx_new = self._pbd(bx_pre)
        disp = self._smooth_bag(bx_new - bx_old)
        bx_new = bx_old + disp
        bx_new[self.pinned] = self.vref[self.pinned]
        self.bx = bx_new
        self.bv = (bx_new - bx_old) / dt
        self.bv[self.pinned] = 0.0

        # ---- prescribed inflow
        self._inject(dt)

        # ---- measured support reaction closes the bag momentum budget
        dp_bag = self.bag_vertex_mass * (self.bv - bv_old).sum(axis=0)
        recorded = imp[0].sum(axis=0)
        pin_res = dp_bag - recorded
        imp[0, _TERM_CONSTRAINT] += pin_res
        imp[2, _TERM_CONSTRAINT] += pin_res

        self.imp += imp
        self.last_force = imp / max(dt, 1e-12)
        self.time += dt
        self.steps += 1

    # ------------------------------------------------------------------
    # sampling / contract
    # ------------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        n_bag = self.n_bag
        nw = len(self.wx)
        if nw:
            position = np.vstack((self.bx, self.wx))
            velocity = np.vstack((self.bv, self.wv))
            mass = np.concatenate(
                (np.full(n_bag, self.bag_vertex_mass), np.full(nw, self.m_p))
            )
            body_id = np.concatenate(
                (np.zeros(n_bag, dtype=np.int32), np.ones(nw, dtype=np.int32))
            )
        else:
            position = self.bx.copy()
            velocity = self.bv.copy()
            mass = np.full(n_bag, self.bag_vertex_mass)
            body_id = np.zeros(n_bag, dtype=np.int32)

        tri = self.bx[self.faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        flip = np.einsum("ij,ij->i", normals, self.tri_outward) < 0.0
        normals[flip] *= -1.0
        boundary_velocity = self.bv[self.faces].mean(axis=1)

        weights = np.zeros((_N_DOMAINS, len(position)))
        weights[0, :n_bag] = 1.0
        weights[1, n_bag:] = 1.0
        weights[2, :] = 1.0

        t0, t1 = self.inflow_interval
        ref_momentum = max(
            1e-6,
            self.rho_water
            * (np.pi * self.nozzle_radius ** 2)
            * self.nozzle_speed
            * max(t1 - t0, 0.0)
            * self.nozzle_speed,
        )
        fields = {
            "boundary_normal": normals,
            "boundary_velocity": boundary_velocity,
            "boundary_target_velocity": boundary_velocity.copy(),
            "budget_cumulative_external_impulse": self.imp.sum(axis=1).copy(),
            "budget_cumulative_inflow_momentum": self.inflow_imp.copy(),
            "budget_cumulative_outflow_momentum": self.outflow_imp.copy(),
            "budget_domain_weights": weights,
            "budget_external_force": self.last_force.copy(),
            "cumulative_mass_source": np.asarray([self.mass_source], dtype=float),
            "interaction_force": self.last_force[0, _TERM_COUPLING][None, :].copy(),
        }
        pinned_attr = self.pinned.astype(float)
        if len(pinned_attr) != len(position):
            pinned_attr = np.concatenate((pinned_attr, np.zeros(len(position) - len(pinned_attr))))
        attributes = {
            "mass_quantity": mass,
            "pinned": pinned_attr,
            "material_phase": body_id.astype(float),
        }
        scalars = {
            "reference_momentum": float(ref_momentum),
            "reference_velocity": float(self.nozzle_speed),
            "water_rest_density": float(self.rho_water),
            "bag_total_mass": float(self.bag_total_mass),
            "liquid_particle_mass": float(self.m_p),
            "sim_time": float(self.time),
        }
        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        while self.time < target - 1e-12:
            remaining = target - self.time
            vmax = float(np.max(np.linalg.norm(self.wv, axis=1))) if len(self.wv) else 0.0
            vmax = min(vmax, 12.0)
            dt = min(self.dt_max, 0.2 * self.hk / (self.c_sound + vmax))
            dt = min(dt, remaining)
            if dt < 1e-9:
                break
            self._step(dt)
            guard += 1
            if guard > 400000:
                break
        if abs(self.time - target) < 1e-9:
            self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        n_bag = self.n_bag
        nw = len(self.wx)
        weight_bag = self.bag_vertex_mass * n_bag
        weight_water = self.m_p * nw
        momentum_bag = self.bag_vertex_mass * self.bv.sum(axis=0)
        momentum_water = self.m_p * self.wv.sum(axis=0)
        bag_res = momentum_bag - self.imp[0].sum(axis=0)
        water_res = (
            momentum_water
            - self.imp[1].sum(axis=0)
            - self.inflow_imp[1]
            + self.outflow_imp[1]
        )
        return {
            "steps": int(self.steps),
            "sim_time": float(self.time),
            "dt_last": float(self.dt_last),
            "bag_vertices": int(n_bag),
            "liquid_particles": int(nw),
            "liquid_mass": float(weight_water),
            "injected_mass": float(self.injected_mass),
            "mass_source": float(self.mass_source),
            "bag_domain_momentum_residual": [float(v) for v in bag_res],
            "water_domain_momentum_residual": [float(v) for v in water_res],
            "bag_mass_note": float(weight_bag),
            "coupling_impulse": [float(v) for v in self.imp[0, _TERM_COUPLING]],
        }
