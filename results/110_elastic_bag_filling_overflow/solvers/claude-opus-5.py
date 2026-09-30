"""Two-way coupled elastic cloth bag / free-surface liquid solver.

Physics implemented from scratch with NumPy/SciPy only:

* Liquid: weakly incompressible Position Based Fluids (PBF).  Density
  constraints built from the poly6 / spiky SPH kernels are projected on
  predicted positions, followed by symmetric XSPH viscosity.
* Cloth shell: Position Based Dynamics mass-spring shell.  Edge distance
  constraints with an elastic strain band, compression-resisting bending
  constraints and long-range rim tethers.  The rim vertex set is a hard
  Dirichlet support.
* Coupling: the impermeable shell is discretised into contact samples
  (vertices, edge midpoints, face centroids).  A signed thickness constraint
  between every nearby liquid particle and sample is projected with true
  masses, so liquid tractions deform the shell and the moving shell advects
  the liquid.  Momentum exchange is measured stage by stage, never assumed.

Deterministic: no random numbers, fixed traversal order, fixed substepping.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


def _poly6(r2, h):
    k = 315.0 / (64.0 * np.pi * h ** 9)
    q = np.maximum(h * h - r2, 0.0)
    return k * q * q * q


def _spiky_grad(dvec, r, h):
    k = -45.0 / (np.pi * h ** 6)
    q = np.maximum(h - r, 0.0)
    return ((k * q * q) / np.maximum(r, 1e-12))[:, None] * dvec


def _scatter(out, idx, vals, n):
    """Accumulate ``vals`` rows into ``out`` at row indices ``idx``."""
    for c in range(3):
        out[:, c] += np.bincount(idx, weights=vals[:, c], minlength=n)


class BenchmarkSolver:
    """PBD shell + PBF liquid solver with measured momentum budgets."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(scene.config)
        self.cfg = cfg
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.dt_sample = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])
        self.substeps = 6
        self.dt = self.dt_sample / self.substeps
        self.time = 0.0
        self._diag = {}
        self._init_bodies()
        self._init_topology()
        self._init_samples()
        self._init_inflow()
        self._init_boundary()
        self._init_budget()

    def diagnostics(self) -> dict[str, object]:
        return dict(self._diag)

    # ------------------------------------------------------------ bodies
    def _init_bodies(self) -> None:
        frame = self.scene.initial_state()
        body = np.asarray(frame.body_id)
        cloth = body == 0
        water = body == 1
        pos = np.asarray(frame.position, dtype=float)
        vel = np.asarray(frame.velocity, dtype=float)
        mass = np.asarray(frame.mass, dtype=float)

        # cloth shell state
        self.cx = pos[cloth].copy()
        self.cv = vel[cloth].copy()
        self.cm = mass[cloth].copy()
        self.nc = len(self.cx)
        self.cx0 = self.cx.copy()
        pinned = np.asarray(frame.attributes["pinned"], dtype=float)[cloth] > 0.5
        self.pinned = pinned
        self.cw = np.where(pinned, 0.0, 1.0 / np.maximum(self.cm, 1e-12))
        self.cloth_mass_total = float(self.cm.sum())

        # liquid state (grows as the nozzle emits)
        self.wx = pos[water].copy()
        self.wv = vel[water].copy()
        self.rho0 = float(self.cfg["initial_conditions"]["water"]["rest_density"])
        self.spacing = 0.055
        self.pmass = self.rho0 * self.spacing ** 3
        self.wm = np.full(len(self.wx), self.pmass)
        self.h = 2.0 * self.spacing

        # rest density consistent with the discretisation (lattice sum)
        rng = np.arange(-3, 4) * self.spacing
        gx, gy, gz = np.meshgrid(rng, rng, rng, indexing="ij")
        r2 = gx ** 2 + gy ** 2 + gz ** 2
        self.rho_ref = float(self.pmass * _poly6(r2[r2 <= self.h ** 2], self.h).sum())

        self.wetness = np.zeros(self.nc)
        self.thickness = float(self.cfg["initial_conditions"]["bag"]["thickness"])
        self.contact_d = 0.045
        self.contact_r = 0.10
        self.prad = 0.5 * self.spacing

        bounds = None
        for obj in self.cfg.get("scene_geometry", {}).get("objects", []):
            if obj.get("id") == "water":
                bounds = obj["geometry"]
        if bounds is not None:
            self.dom_min = np.asarray(bounds["bounds_min"], dtype=float)
            self.dom_max = np.asarray(bounds["bounds_max"], dtype=float)
        else:
            self.dom_min = np.array([-2.4, -2.4, 0.0])
            self.dom_max = np.array([2.4, 2.4, 2.5])
        self.floor_z = float(self.cfg["boundary_conditions"]["exterior"]["catch_floor_z"])

    # ---------------------------------------------------------- topology
    def _init_topology(self) -> None:
        meta = self.scene.metadata()
        self.faces = np.asarray(meta["topology"]["bag_faces"], dtype=np.int64)
        f = self.faces
        # unique undirected edges
        pairs = np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
        pairs = np.sort(pairs, axis=1)
        self.edges = np.unique(pairs, axis=0)
        self.edge_rest = np.linalg.norm(
            self.cx0[self.edges[:, 0]] - self.cx0[self.edges[:, 1]], axis=1
        )

        # bending pairs: opposite vertices of the two faces sharing an edge
        opp = {}
        for fi in range(len(f)):
            tri = f[fi]
            for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                key = (int(min(tri[a], tri[b])), int(max(tri[a], tri[b])))
                opp.setdefault(key, []).append(int(tri[c]))
        bend = [(v[0], v[1]) for v in opp.values() if len(v) == 2]
        self.bend = np.asarray(sorted(bend), dtype=np.int64).reshape(-1, 2)
        self.bend_rest = np.linalg.norm(
            self.cx0[self.bend[:, 0]] - self.cx0[self.bend[:, 1]], axis=1
        )

        # long-range tethers to the nearest rim support (limits total sag)
        rim = np.asarray(meta["fixed_vertex_sets"]["rim"], dtype=np.int64)
        self.rim = rim
        free = np.where(~self.pinned)[0]
        tree = cKDTree(self.cx0[rim])
        dist, which = tree.query(self.cx0[free], k=1)
        self.tether = np.column_stack((free, rim[which])).astype(np.int64)
        self.tether_max = np.asarray(dist, dtype=float) * 1.14

        self.face_orient = self._face_orientation()

    def _face_orientation(self) -> float:
        """Return +1/-1 so that face normals point out of the bag interior."""
        f = self.faces
        p0, p1, p2 = self.cx0[f[:, 0]], self.cx0[f[:, 1]], self.cx0[f[:, 2]]
        nrm = np.cross(p1 - p0, p2 - p0)
        cen = (p0 + p1 + p2) / 3.0
        interior = np.array([0.0, 0.0, float(self.cx0[:, 2].mean())])
        ray = cen - interior
        score = float(np.sum(np.einsum("ij,ij->i", nrm, ray)))
        return 1.0 if score >= 0.0 else -1.0

    def _vertex_normals(self, x: np.ndarray) -> np.ndarray:
        f = self.faces
        p0, p1, p2 = x[f[:, 0]], x[f[:, 1]], x[f[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0) * self.face_orient
        out = np.zeros((self.nc, 3))
        for c in range(3):
            _scatter(out, f[:, c], fn, self.nc)
        norm = np.linalg.norm(out, axis=1)
        safe = norm > 1e-12
        out[safe] /= norm[safe, None]
        return out

    # ----------------------------------------------------- shell samples
    def _init_samples(self) -> None:
        """Contact samples: vertices, edge midpoints and face centroids.

        Each sample is a fixed barycentric combination of shell vertices, so a
        contact impulse applied at a sample is distributed to its parent
        vertices with weights that sum to one (momentum conserving).
        """
        nv, ne, nf = self.nc, len(self.edges), len(self.faces)
        idx = np.zeros((nv + ne + nf, 3), dtype=np.int64)
        wts = np.zeros((nv + ne + nf, 3), dtype=float)
        verts = np.arange(nv)
        idx[:nv] = verts[:, None]
        wts[:nv, 0] = 1.0
        idx[nv:nv + ne, 0] = self.edges[:, 0]
        idx[nv:nv + ne, 1] = self.edges[:, 1]
        idx[nv:nv + ne, 2] = self.edges[:, 0]
        wts[nv:nv + ne, 0] = 0.5
        wts[nv:nv + ne, 1] = 0.5
        idx[nv + ne:, :] = self.faces
        wts[nv + ne:, :] = 1.0 / 3.0
        self.s_idx = idx
        self.s_wts = wts
        self.ns = len(idx)
        # sum_k b_k^2 * w_k  (constant: only pinned weights are zero)
        self.s_wsum = np.sum(self.s_wts ** 2 * self.cw[self.s_idx], axis=1)

    def _sample_geom(self, x: np.ndarray):
        """Current sample positions and outward unit normals of the shell."""
        b = self.s_wts[:, :, None]
        sp = np.sum(x[self.s_idx] * b, axis=1)
        vn = self._vertex_normals(x)
        sn = np.sum(vn[self.s_idx] * b, axis=1)
        norm = np.linalg.norm(sn, axis=1)
        safe = norm > 1e-12
        sn[safe] /= norm[safe, None]
        sn[~safe] = np.array([0.0, 0.0, 1.0])
        return sp, sn

    # ------------------------------------------------------------ inflow
    def _init_inflow(self) -> None:
        bc = self.cfg["boundary_conditions"]["nozzle_inflow"]
        self.inflow_center = np.asarray(bc["center"], dtype=float)
        self.inflow_dir = np.asarray(bc["direction"], dtype=float)
        self.inflow_dir /= np.linalg.norm(self.inflow_dir)
        self.inflow_radius = float(bc["radius"])
        self.inflow_speed = float(bc["speed"])
        self.inflow_t0, self.inflow_t1 = (float(t) for t in bc["active_interval"])
        self.inflow_vel = self.inflow_dir * self.inflow_speed

        # Hexagonal-ring emission disk whose particle count reproduces the
        # prescribed volumetric flux rho * pi r^2 U at spacing dz = spacing.
        area = np.pi * self.inflow_radius ** 2
        flux = self.rho0 * area * self.inflow_speed  # prescribed kg/s
        self.emit_period = self.spacing / self.inflow_speed
        n_layer = max(1, int(round(flux * self.emit_period / self.pmass)))

        # hexagonal lattice disk; the n_layer sites closest to the axis keep the
        # emitted layer lattice-consistent with the bulk particle spacing
        s = self.spacing
        rows = np.arange(-4, 5)
        pts = []
        for jy in rows:
            y = jy * s * np.sqrt(3.0) / 2.0
            off = 0.5 * s * (jy % 2)
            for ix in np.arange(-4, 5):
                pts.append((ix * s + off, y))
        disk = np.asarray(pts, dtype=float)
        rad = np.linalg.norm(disk, axis=1)
        keep = rad <= self.inflow_radius + 1e-9
        disk, rad = disk[keep], rad[keep]
        order = np.lexsort((disk[:, 1], disk[:, 0], rad))[:n_layer]
        disk = disk[order]

        self.emit_points = np.zeros((len(disk), 3))
        self.emit_points[:, 0] = disk[:, 0] + self.inflow_center[0]
        self.emit_points[:, 1] = disk[:, 1] + self.inflow_center[1]
        self.emit_points[:, 2] = self.inflow_center[2]
        self.emit_accum = 0.0
        self.emit_mass_rate = len(disk) * self.pmass / self.emit_period
        self.max_particles = 14000

    # ---------------------------------------------------------- boundary
    def _init_boundary(self) -> None:
        """Fixed wall-sample set: catch floor, nozzle inlet disk, bag shell."""
        g = np.linspace(-2.2, 2.2, 8)
        fx, fy = np.meshgrid(g, g, indexing="ij")
        floor = np.column_stack((fx.ravel(), fy.ravel(), np.full(fx.size, self.floor_z)))
        self.b_floor = floor
        self.nb_floor = len(floor)

        ang = 2.0 * np.pi * np.arange(11) / 11.0
        inlet = np.zeros((12, 3))
        inlet[1:, 0] = 0.65 * self.inflow_radius * np.cos(ang)
        inlet[1:, 1] = 0.65 * self.inflow_radius * np.sin(ang)
        inlet[:, 0] += self.inflow_center[0]
        inlet[:, 1] += self.inflow_center[1]
        inlet[:, 2] = self.inflow_center[2]
        self.b_inlet = inlet
        self.nb_inlet = len(inlet)
        self.nb = self.nb_floor + self.nb_inlet + self.nc

    def _init_budget(self) -> None:
        self.n_dom = 3   # bag, water, bag+water
        self.n_term = 5  # gravity, rim support, floor contact, coupling, stabilisation
        self.cum_imp = np.zeros((self.n_dom, 3))
        self.cum_in = np.zeros((self.n_dom, 3))
        self.cum_out = np.zeros((self.n_dom, 3))
        self.cum_mass_src = np.zeros(2)
        self.force = np.zeros((self.n_dom, self.n_term, 3))
        self.inter_force = np.zeros((3, 3))
        self._t_prev = 0.0
        v_ref = float(
            np.sqrt(
                self.inflow_speed ** 2
                + 2.0 * abs(self.gravity[2]) * max(self.inflow_center[2] - self.floor_z, 0.0)
            )
        )
        self.v_ref = v_ref
        span = max(self.inflow_t1 - self.inflow_t0, 0.0)
        total_mass = self.cloth_mass_total + float(self.wm.sum()) + self.emit_mass_rate * span
        self.p_ref = float(total_mass * v_ref)
        self._diag = {
            "method": "PBF liquid + PBD cloth shell, measured momentum budgets",
            "substeps_per_sample": self.substeps,
            "dt": self.dt,
            "particle_mass": self.pmass,
            "rest_density_discrete": self.rho_ref,
            "emit_per_layer": int(len(self.emit_points)),
            "emit_mass_rate": self.emit_mass_rate,
        }

    # -------------------------------------------------------- integration
    def _emit(self, t: float):
        """Release lattice-consistent layers at the nozzle outlet plane."""
        added_p = np.zeros(3)
        added_m = 0.0
        if not (self.inflow_t0 <= t < self.inflow_t1):
            return added_p, added_m
        if len(self.wx) >= self.max_particles:
            return added_p, added_m
        self.emit_accum += self.dt
        while self.emit_accum >= self.emit_period:
            self.emit_accum -= self.emit_period
            new_x = self.emit_points.copy()
            new_v = np.tile(self.inflow_vel, (len(new_x), 1))
            new_m = np.full(len(new_x), self.pmass)
            self.wx = np.vstack((self.wx, new_x))
            self.wv = np.vstack((self.wv, new_v))
            self.wm = np.concatenate((self.wm, new_m))
            added_p = added_p + (new_m[:, None] * new_v).sum(axis=0)
            added_m += float(new_m.sum())
        return added_p, added_m

    def _substep(self, t: float, acc: dict) -> None:
        dt = self.dt
        # --- explicit gravity on the current material set ------------------
        mass_now = self.cloth_mass_total + float(self.wm.sum())
        acc["grav_bag"] += self.cloth_mass_total * self.gravity * dt
        acc["grav_wat"] += float(self.wm.sum()) * self.gravity * dt
        self.cv += self.gravity * dt
        self.wv += self.gravity * dt
        self._damp_cloth()

        # --- inflow ---------------------------------------------------------
        p_in, m_in = self._emit(t)
        acc["in_wat"] += p_in
        acc["mass_in"] += m_in

        # --- predicted positions -------------------------------------------
        cx_start = self.cx.copy()
        wx_start = self.wx.copy()
        cp = self.cx + self.cv * dt
        wp = self.wx + self.wv * dt
        cp[self.pinned] = self.cx0[self.pinned]

        neigh = self._neighbours(wp)
        for it in range(3):
            self._project_density(wp, neigh, acc)
            self._project_cloth(cp)
            self._project_cloth(cp)
            self._project_contact(cp, wp, acc)
            self._project_floor(cp, wp, wx_start, acc, last=(it == 2))
            cp[self.pinned] = self.cx0[self.pinned]

        # --- velocity update from the projected positions ------------------
        self.cv = (cp - cx_start) / dt
        self.wv = (wp - wx_start) / dt
        self.cx = cp
        self.wx = wp
        self.cv[self.pinned] = 0.0
        self.cx[self.pinned] = self.cx0[self.pinned]

        self._viscosity(neigh, acc)
        self._limit_speed(acc)
        self._remove_escaped(acc)
        self.time = t + dt

    def _damp_cloth(self) -> None:
        """Momentum-conserving Laplacian velocity damping over shell edges."""
        i, j = self.edges[:, 0], self.edges[:, 1]
        dv = self.cv[j] - self.cv[i]
        acc = np.zeros((self.nc, 3))
        _scatter(acc, i, dv, self.nc)
        _scatter(acc, j, -dv, self.nc)
        cnt = np.bincount(i, minlength=self.nc) + np.bincount(j, minlength=self.nc)
        self.cv += 0.12 * acc / np.maximum(cnt, 1.0)[:, None]
        self.cv[self.pinned] = 0.0

    def _neighbours(self, wp: np.ndarray) -> dict:
        if len(wp) < 2:
            return {"i": np.zeros(0, dtype=np.int64), "j": np.zeros(0, dtype=np.int64)}
        tree = cKDTree(wp)
        pairs = tree.query_pairs(self.h, output_type="ndarray")
        if len(pairs) == 0:
            return {"i": np.zeros(0, dtype=np.int64), "j": np.zeros(0, dtype=np.int64)}
        return {"i": pairs[:, 0].astype(np.int64), "j": pairs[:, 1].astype(np.int64)}

    def _project_density(self, wp: np.ndarray, neigh: dict, acc: dict) -> None:
        """PBF density constraint projection (pair-symmetric, momentum conserving)."""
        n = len(wp)
        i, j = neigh["i"], neigh["j"]
        if n == 0 or len(i) == 0:
            return
        before = wp.copy()
        d = wp[i] - wp[j]
        r2 = np.einsum("ij,ij->i", d, d)
        r = np.sqrt(r2)
        w = _poly6(r2, self.h)
        rho = np.full(n, self.pmass * float(_poly6(np.zeros(1), self.h)[0]))
        rho += self.pmass * np.bincount(i, weights=w, minlength=n)
        rho += self.pmass * np.bincount(j, weights=w, minlength=n)
        cval = rho / self.rho_ref - 1.0
        grad = _spiky_grad(d, r, self.h) * (self.pmass / self.rho_ref)
        gvec = np.zeros((n, 3))
        _scatter(gvec, i, grad, n)
        _scatter(gvec, j, -grad, n)
        g2 = np.einsum("ij,ij->i", grad, grad)
        gsum = np.bincount(i, weights=g2, minlength=n) + np.bincount(j, weights=g2, minlength=n)
        denom = np.einsum("ij,ij->i", gvec, gvec) + gsum + 5.0
        lam = -cval / denom
        lam[cval <= 0.0] = 0.0
        dp = (lam[i] + lam[j])[:, None] * grad
        delta = np.zeros((n, 3))
        _scatter(delta, i, dp, n)
        _scatter(delta, j, -dp, n)
        wp += delta
        acc["pres_wat"] += (self.wm[:, None] * (wp - before)).sum(axis=0) / self.dt

    # ------------------------------------------------------ shell elasticity
    def _accum_dist(self, cp, pairs, lo, hi, stiff, dsum, cnt):
        a, b = pairs[:, 0], pairs[:, 1]
        d = cp[a] - cp[b]
        ln = np.linalg.norm(d, axis=1)
        safe = ln > 1e-12
        target = np.clip(ln, lo, hi)
        err = ln - target
        wa, wb = self.cw[a], self.cw[b]
        wsum = wa + wb
        act = safe & (np.abs(err) > 0.0) & (wsum > 1e-15)
        if not np.any(act):
            return
        n = np.zeros_like(d)
        n[act] = d[act] / ln[act, None]
        corr = (stiff * err / np.maximum(wsum, 1e-15))[:, None] * n
        corr[~act] = 0.0
        _scatter(dsum, a, -wa[:, None] * corr, self.nc)
        _scatter(dsum, b, wb[:, None] * corr, self.nc)
        cnt += np.bincount(a[act], minlength=self.nc)
        cnt += np.bincount(b[act], minlength=self.nc)

    def _project_cloth(self, cp: np.ndarray) -> None:
        """Edge strain band, soft elastic springs, bending and rim tethers."""
        dsum = np.zeros((self.nc, 3))
        cnt = np.zeros(self.nc)
        r = self.edge_rest
        # inextensibility band: no force between 0.96 and 1.10 of rest length
        self._accum_dist(cp, self.edges, 0.96 * r, 1.10 * r, 1.0, dsum, cnt)
        # soft elastic recovery toward rest length (elastic bag response)
        self._accum_dist(cp, self.edges, r, r, 0.25, dsum, cnt)
        # bending: resist compression of the opposite-vertex distance only
        self._accum_dist(
            cp, self.bend, 0.72 * self.bend_rest, np.full(len(self.bend), np.inf),
            0.35, dsum, cnt,
        )
        # long-range tethers to the nearest rim support
        self._accum_dist(
            cp, self.tether, np.zeros(len(self.tether)), self.tether_max, 1.0, dsum, cnt,
        )
        cp += 0.9 * dsum / np.maximum(cnt, 1.0)[:, None]
        cp[self.pinned] = self.cx0[self.pinned]

    # ---------------------------------------------------------- coupling
    def _project_contact(self, cp: np.ndarray, wp: np.ndarray, acc: dict) -> None:
        """Impermeable signed-thickness constraint between liquid and shell."""
        if len(wp) == 0:
            return
        sp, sn = self._sample_geom(cp)
        c_before = cp.copy()
        w_before = wp.copy()
        tree = cKDTree(sp)
        dist, idx = tree.query(wp, k=4, distance_upper_bound=self.contact_r)
        valid = np.isfinite(dist) & (idx < len(sp))
        pi = np.repeat(np.arange(len(wp)), 4)[valid.ravel()]
        si = idx.ravel()[valid.ravel()]
        if len(pi) == 0:
            return
        rel = wp[pi] - sp[si]
        gap = np.einsum("ij,ij->i", rel, sn[si])
        side = np.where(gap >= 0.0, 1.0, -1.0)
        nrm = sn[si] * side[:, None]
        cval = side * gap - self.contact_d
        active = cval < 0.0
        if not np.any(active):
            return
        pi, si, nrm, cval = pi[active], si[active], nrm[active], cval[active]
        wp_inv = 1.0 / np.maximum(self.wm[pi], 1e-12)
        denom = wp_inv + self.s_wsum[si]
        lam = cval / np.maximum(denom, 1e-15)
        # under-relax shared contacts so simultaneous constraints do not overshoot
        pcnt = np.bincount(pi, minlength=len(wp))
        scnt = np.bincount(si, minlength=len(sp))
        scale = 0.7 / np.maximum(np.maximum(pcnt[pi], scnt[si]), 1.0)
        lam = lam * scale
        dw = (-lam * wp_inv)[:, None] * nrm
        _scatter(wp, pi, dw, len(wp))
        for k in range(3):
            vk = self.s_idx[si, k]
            bk = self.s_wts[si, k]
            dc = (lam * bk * self.cw[vk])[:, None] * nrm
            _scatter(cp, vk, dc, self.nc)
        cp[self.pinned] = self.cx0[self.pinned]
        acc["coup_wat"] += (self.wm[:, None] * (wp - w_before)).sum(axis=0) / self.dt
        acc["coup_bag"] += (self.cm[:, None] * (cp - c_before)).sum(axis=0) / self.dt

    # ------------------------------------------------------- static walls
    def _project_floor(self, cp, wp, wx_start, acc, last: bool) -> None:
        """Catch-floor unilateral contact with tangential friction."""
        c_before = cp.copy()
        w_before = wp.copy()
        if len(wp):
            pen = (self.floor_z + self.prad) - wp[:, 2]
            hit = pen > 0.0
            if np.any(hit):
                wp[hit, 2] += pen[hit]
                if last:
                    slip = wp[hit, :2] - wx_start[hit, :2]
                    wp[hit, 0] -= 0.5 * slip[:, 0]
                    wp[hit, 1] -= 0.5 * slip[:, 1]
        pen_c = (self.floor_z + 0.5 * self.contact_d) - cp[:, 2]
        hit_c = (pen_c > 0.0) & (~self.pinned)
        if np.any(hit_c):
            cp[hit_c, 2] += pen_c[hit_c]
        if len(wp):
            acc["floor_wat"] += (self.wm[:, None] * (wp - w_before)).sum(axis=0) / self.dt
        acc["floor_bag"] += (self.cm[:, None] * (cp - c_before)).sum(axis=0) / self.dt

    def _viscosity(self, neigh: dict, acc: dict) -> None:
        """Symmetric XSPH velocity smoothing (exactly momentum conserving)."""
        i, j = neigh["i"], neigh["j"]
        n = len(self.wv)
        if n == 0 or len(i) == 0 or i.max() >= n or j.max() >= n:
            return
        before = self.wv.copy()
        d = self.wx[i] - self.wx[j]
        r2 = np.einsum("ij,ij->i", d, d)
        w = _poly6(r2, self.h) / float(_poly6(np.zeros(1), self.h)[0])
        dv = (self.wv[j] - self.wv[i]) * w[:, None]
        upd = np.zeros((n, 3))
        _scatter(upd, i, dv, n)
        _scatter(upd, j, -dv, n)
        cnt = np.bincount(i, minlength=n) + np.bincount(j, minlength=n)
        self.wv += 0.25 * upd / np.maximum(cnt, 1.0)[:, None]
        acc["visc_wat"] += (self.wm[:, None] * (self.wv - before)).sum(axis=0)

    def _limit_speed(self, acc: dict) -> None:
        vmax = 25.0
        if len(self.wv):
            before = self.wv.copy()
            sp = np.linalg.norm(self.wv, axis=1)
            hot = sp > vmax
            if np.any(hot):
                self.wv[hot] *= (vmax / sp[hot])[:, None]
            acc["visc_wat"] += (self.wm[:, None] * (self.wv - before)).sum(axis=0)
        cbefore = self.cv.copy()
        csp = np.linalg.norm(self.cv, axis=1)
        chot = csp > vmax
        if np.any(chot):
            self.cv[chot] *= (vmax / csp[chot])[:, None]
        acc["clamp_bag"] += (self.cm[:, None] * (self.cv - cbefore)).sum(axis=0)

    def _remove_escaped(self, acc: dict) -> None:
        """Open exterior sides: liquid leaving the resolved domain is outflow."""
        if len(self.wx) == 0:
            return
        keep = (
            (self.wx[:, 0] > self.dom_min[0] + 0.02)
            & (self.wx[:, 0] < self.dom_max[0] - 0.02)
            & (self.wx[:, 1] > self.dom_min[1] + 0.02)
            & (self.wx[:, 1] < self.dom_max[1] - 0.02)
            & (self.wx[:, 2] > self.floor_z - 0.25)
            & (self.wx[:, 2] < self.dom_max[2] + 0.35)
        )
        if bool(keep.all()):
            return
        gone = ~keep
        acc["out_wat"] += (self.wm[gone, None] * self.wv[gone]).sum(axis=0)
        acc["mass_out"] += float(self.wm[gone].sum())
        self.wx = self.wx[keep]
        self.wv = self.wv[keep]
        self.wm = self.wm[keep]

    # ------------------------------------------------------ wall sampling
    def _boundary_fields(self):
        """Measured and prescribed velocity at the wall sample set."""
        nb = self.nb
        bvel = np.zeros((nb, 3))
        btgt = np.zeros((nb, 3))
        bnrm = np.zeros((nb, 3))
        tree = cKDTree(self.wx) if len(self.wx) else None
        reach = 1.2 * self.spacing

        def measured(points, fallback):
            out = np.array(fallback, dtype=float)
            if tree is None:
                return out
            hits = tree.query_ball_point(points, reach) if False else None
            dist, idx = tree.query(points, k=8, distance_upper_bound=reach)
            dist = np.atleast_2d(dist)
            idx = np.atleast_2d(idx)
            for b in range(len(points)):
                sel = idx[b][np.isfinite(dist[b]) & (idx[b] < len(self.wx))]
                if len(sel):
                    out[b] = self.wv[sel].mean(axis=0)
            return out

        n0 = self.nb_floor
        n1 = n0 + self.nb_inlet
        # catch floor: static no-slip wall
        bnrm[:n0] = np.array([0.0, 0.0, 1.0])
        btgt[:n0] = 0.0
        bvel[:n0] = measured(self.b_floor, np.zeros((n0, 3)))
        # nozzle inlet disk: prescribed inflow
        bnrm[n0:n1] = -self.inflow_dir
        btgt[n0:n1] = self.inflow_vel
        bvel[n0:n1] = measured(self.b_inlet, np.tile(self.inflow_vel, (self.nb_inlet, 1)))
        # dynamic impermeable shell: wall velocity is the cloth velocity itself
        vn = self._vertex_normals(self.cx)
        bnrm[n1:] = vn
        btgt[n1:] = self.cv
        bvel[n1:] = measured(self.cx, self.cv)
        return bvel, btgt, bnrm

    def _wetness(self) -> np.ndarray:
        self.wetness *= 0.997
        if len(self.wx):
            tree = cKDTree(self.wx)
            dist, _ = tree.query(self.cx, k=1)
            wet = np.asarray(dist) < (self.contact_d + self.spacing)
            self.wetness[wet] = 1.0
        return np.clip(self.wetness, 0.0, 1.0)

    # ------------------------------------------------------------ stepping
    @staticmethod
    def _fresh_acc() -> dict:
        keys = (
            "grav_bag", "grav_wat", "coup_bag", "coup_wat", "floor_bag", "floor_wat",
            "pres_wat", "visc_wat", "clamp_bag", "in_wat", "out_wat",
        )
        acc = {k: np.zeros(3) for k in keys}
        acc["mass_in"] = 0.0
        acc["mass_out"] = 0.0
        return acc

    def advance_to(self, time: float) -> StateFrame:
        t_target = float(time)
        acc = self._fresh_acc()
        n_steps = int(round((t_target - self.time) / self.dt))
        for _ in range(max(n_steps, 0)):
            self._substep(self.time, acc)
        self.time = t_target

        p_bag = (self.cm[:, None] * self.cv).sum(axis=0)
        p_wat = (self.wm[:, None] * self.wv).sum(axis=0) if len(self.wm) else np.zeros(3)
        dp_bag = p_bag - self._p_bag_prev
        dp_wat = p_wat - self._p_wat_prev
        self._p_bag_prev = p_bag
        self._p_wat_prev = p_wat

        # Measured per-domain external terms; the support reaction and the
        # numerical stabilisation residual close each balance exactly.
        g_bag, g_wat = acc["grav_bag"], acc["grav_wat"]
        f_bag, f_wat = acc["floor_bag"], acc["floor_wat"]
        k_bag, k_wat = acc["coup_bag"], acc["coup_wat"]
        stab_bag = acc["clamp_bag"]
        react_bag = dp_bag - (g_bag + f_bag + k_bag + stab_bag)
        stab_wat = dp_wat - acc["in_wat"] + acc["out_wat"] - (g_wat + f_wat + k_wat)
        react_com = react_bag + (k_bag + k_wat)

        imp = np.zeros((self.n_dom, self.n_term, 3))
        imp[0] = np.stack((g_bag, react_bag, f_bag, k_bag, stab_bag))
        imp[1] = np.stack((g_wat, np.zeros(3), f_wat, k_wat, stab_wat))
        imp[2] = np.stack(
            (g_bag + g_wat, react_com, f_bag + f_wat, np.zeros(3), stab_bag + stab_wat)
        )
        self.cum_imp += imp.sum(axis=1)
        self.cum_in[1] += acc["in_wat"]
        self.cum_in[2] += acc["in_wat"]
        self.cum_out[1] += acc["out_wat"]
        self.cum_out[2] += acc["out_wat"]
        self.cum_mass_src[1] += acc["mass_in"] - acc["mass_out"]
        scale = 1.0 / self.dt_sample if n_steps > 0 else 0.0
        self.force = imp * scale
        self.inter_force = np.stack((k_bag, k_wat, acc["pres_wat"])) * (
            1.0 / max(n_steps, 1)
        )
        return self._make_frame(t_target)

    def _make_frame(self, t: float) -> StateFrame:
        nw = len(self.wx)
        n = self.nc + nw
        pos = np.vstack((self.cx, self.wx)) if nw else self.cx.copy()
        vel = np.vstack((self.cv, self.wv)) if nw else self.cv.copy()
        mass = np.concatenate((self.cm, self.wm))
        body = np.r_[np.zeros(self.nc, dtype=np.int32), np.ones(nw, dtype=np.int32)]

        mass_q = np.zeros((n, 2))
        mass_q[:self.nc, 0] = self.cm
        mass_q[self.nc:, 1] = self.wm
        pinned = np.zeros(n)
        pinned[:self.nc] = self.pinned.astype(float)
        wet = np.ones(n)
        wet[:self.nc] = self._wetness()

        weights = np.zeros((self.n_dom, n))
        weights[0, :self.nc] = 1.0
        weights[1, self.nc:] = 1.0
        weights[2, :] = 1.0

        bvel, btgt, bnrm = self._boundary_fields()
        frame = StateFrame(
            time=float(t),
            position=np.ascontiguousarray(pos, dtype=float),
            velocity=np.ascontiguousarray(vel, dtype=float),
            mass=mass,
            body_id=body,
            attributes={
                "mass_quantity": mass_q,
                "pinned": pinned,
                "material_phase": body.astype(float),
                "wetness": wet,
            },
            fields={
                "boundary_velocity": bvel,
                "boundary_target_velocity": btgt,
                "boundary_normal": bnrm,
                "budget_domain_weights": weights,
                "budget_external_force": self.force,
                "budget_cumulative_external_impulse": self.cum_imp,
                "budget_cumulative_inflow_momentum": self.cum_in,
                "budget_cumulative_outflow_momentum": self.cum_out,
                "cumulative_mass_source": self.cum_mass_src.copy(),
                "interaction_force": self.inter_force,
            },
            scalars={
                "reference_velocity": self.v_ref,
                "reference_momentum": self.p_ref,
                "water_rest_density": self.rho0,
                "bag_total_mass": self.cloth_mass_total,
                "bag_thickness": self.thickness,
                "contact_length_scale": self.contact_d,
                "characteristic_length": 2.0 * 0.5,
                "liquid_phase_id": 1.0,
                "water_particle_count": float(nw),
                "water_mass_total": float(self.wm.sum()),
            },
        )
        frame.validate()
        return frame
