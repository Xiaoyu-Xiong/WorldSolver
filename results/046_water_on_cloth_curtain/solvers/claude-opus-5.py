"""Two-way coupled water / cloth-curtain solver.

Physical model (implemented from scratch, no external physics engine):

* Cloth curtain: lumped-mass triangle-mesh shell.  Membrane (structural +
  shear) behaviour is imposed with inextensible distance constraints taken
  from the frozen topology asset, bending with skip-one distance
  constraints.  Constraints are solved with an XPBD / position-based
  Gauss-Seidel-Jacobi projection, which is unconditionally stable for the
  time step used here.  The top vertex row is pinned (zero inverse mass).

* Water: weakly-compressible SPH written in position-based form (PBF).
  A density constraint  C_i = rho_i/rho_0 - 1 >= 0  is projected every
  iteration using the poly6 / spiky kernel pair, followed by XSPH
  velocity smoothing.  Mass is injected by a prescribed nozzle inflow that
  reproduces the frozen flux rho * A * v.

* Coupling: the water particles and the cloth triangles interact through
  symmetric, mass-weighted non-penetration constraints (point vs. triangle
  with barycentric load transfer) plus Coulomb-style tangential friction
  impulses.  Because the constraints are solved in the same iteration loop
  as the cloth membrane constraints, the hydrodynamic load is transmitted
  through the sheet to the pinned support, so the water is redirected by
  the sheet while the sheet is pushed by the water (full two-way coupling).

* Static boundaries (catch-tank floor and walls) act as one-sided
  constraints on both phases.

Everything is deterministic: no random numbers are used anywhere.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# small vector helpers
# ----------------------------------------------------------------------------


def _dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def _norm(a):
    return np.sqrt(np.maximum(_dot(a, a), 0.0))


def _closest_point_on_triangle(p, a, b, c):
    """Vectorised closest point on triangle (Ericson, Real-Time Collision Detection)."""
    eps = 1e-30
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    res = a.copy()
    done = (d1 <= 0.0) & (d2 <= 0.0)

    bp = p - b
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    m = (~done) & (d3 >= 0.0) & (d4 <= d3)
    res[m] = b[m]
    done = done | m

    vc = d1 * d4 - d3 * d2
    m = (~done) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    if np.any(m):
        t = np.clip(d1[m] / (d1[m] - d3[m] + eps), 0.0, 1.0)
        res[m] = a[m] + t[:, None] * ab[m]
    done = done | m

    cp = p - c
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    m = (~done) & (d6 >= 0.0) & (d5 <= d6)
    res[m] = c[m]
    done = done | m

    vb = d5 * d2 - d1 * d6
    m = (~done) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    if np.any(m):
        t = np.clip(d2[m] / (d2[m] - d6[m] + eps), 0.0, 1.0)
        res[m] = a[m] + t[:, None] * ac[m]
    done = done | m

    va = d3 * d6 - d5 * d4
    m = (~done) & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    if np.any(m):
        t = np.clip((d4[m] - d3[m]) / ((d4[m] - d3[m]) + (d5[m] - d6[m]) + eps), 0.0, 1.0)
        res[m] = b[m] + t[:, None] * (c[m] - b[m])
    done = done | m

    m = ~done
    if np.any(m):
        denom = 1.0 / (va[m] + vb[m] + vc[m] + eps)
        v = vb[m] * denom
        w = vc[m] * denom
        res[m] = a[m] + v[:, None] * ab[m] + w[:, None] * ac[m]
    return res


def _barycentric(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = _dot(v0, v0)
    d01 = _dot(v0, v1)
    d11 = _dot(v1, v1)
    d20 = _dot(v2, v0)
    d21 = _dot(v2, v1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-24, 1e-24, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    v = np.clip(v, 0.0, 1.0)
    w = np.clip(w, 0.0, 1.0)
    u = np.clip(1.0 - v - w, 0.0, 1.0)
    s = u + v + w
    s = np.where(s < 1e-12, 1.0, s)
    return u / s, v / s, w / s


# ----------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------


class BenchmarkSolver:
    # numerical parameters
    SUBSTEP_DT = 0.005
    N_ITER = 5
    N_CLOTH_PASS = 3
    CONTACT_SEARCH = 0.16
    MAX_CONTACT_ROWS = 20000

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(scene.config)
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.duration = float(cfg.get("duration", 4.0))

        frame0 = scene.initial_state()
        meta = scene.metadata()
        self.scene_scalars = {k: float(v) for k, v in dict(frame0.scalars).items()}

        pos = np.asarray(frame0.position, dtype=np.float64)
        vel = np.asarray(frame0.velocity, dtype=np.float64)
        mass = np.asarray(frame0.mass, dtype=np.float64)
        body = np.asarray(frame0.body_id).astype(np.int64)
        pinned_attr = np.asarray(frame0.attributes["pinned"], dtype=np.float64)

        cloth_sel = body == 0
        water_sel = body == 1

        # ------------------------------------------------------------ cloth
        self.xc = pos[cloth_sel].copy()
        self.vc = vel[cloth_sel].copy()
        self.mc = mass[cloth_sel].copy()
        self.nc = len(self.xc)
        self.pinned = pinned_attr[cloth_sel] > 0.5
        self.wc = np.where(self.pinned, 0.0, 1.0 / np.maximum(self.mc, 1e-12))
        self.pin_rest = self.xc[self.pinned].copy()
        self.cloth_thickness = float(self.scene_scalars.get("cloth_thickness", 0.0015))

        topo = self._load_topology()
        self.faces = topo["faces"].astype(np.int64)
        edges = topo["edges"].astype(np.int64)
        ny, nz = int(scene.CLOTH_NY), int(scene.CLOTH_NZ)
        self.grid_ny, self.grid_nz = ny, nz

        bend = []
        idx = np.arange(self.nc).reshape(nz, ny)
        for k in range(ny - 2):
            bend.append(np.stack((idx[:, k], idx[:, k + 2]), axis=1))
        for k in range(nz - 2):
            bend.append(np.stack((idx[k, :], idx[k + 2, :]), axis=1))
        bend = np.concatenate(bend, axis=0)

        self.edges = np.vstack((edges, bend)).astype(np.int64)
        self.e_rest = _norm(self.xc[self.edges[:, 1]] - self.xc[self.edges[:, 0]])
        self.e_rest = np.maximum(self.e_rest, 1e-6)
        self.e_stiff = np.concatenate(
            (np.full(len(edges), 1.0), np.full(len(bend), 0.08))
        )
        # only membrane edges carry the support reaction bookkeeping
        self.e_structural = np.zeros(len(self.edges), dtype=bool)
        self.e_structural[: len(edges)] = True
        self.mean_edge = float(np.mean(self.e_rest[: len(edges)]))

        # ------------------------------------------------------------ water
        w_pos = pos[water_sel]
        w_vel = vel[water_sel]
        self.mw = float(np.mean(mass[water_sel]))
        self.rho_phys = float(self.scene_scalars.get("water_density", 1000.0))
        self.p_vol = self.mw / self.rho_phys
        self.spacing = self.p_vol ** (1.0 / 3.0)
        self.h = 1.62 * self.spacing
        self._kernel_setup()

        inflow = dict(meta.get("inflow", {}))
        self.in_center = np.asarray(
            inflow.get("center", [-1.255, 0.0, 1.72]), dtype=np.float64
        )
        self.in_radius = float(inflow.get("radius", 0.20))
        self.in_vel = np.asarray(inflow.get("velocity", [3.2, 0.0, 0.0]), dtype=np.float64)
        interval = inflow.get("active_interval", [0.0, self.duration])
        self.in_t0, self.in_t1 = float(interval[0]), float(interval[1])
        self.in_speed = float(_norm(self.in_vel[None, :])[0])
        self.in_dir = self.in_vel / max(self.in_speed, 1e-12)

        self.layer_template = self._emit_template()
        self.n_layer = len(self.layer_template)
        area = np.pi * self.in_radius ** 2
        # layer pitch chosen so the discrete emission reproduces rho*A*v exactly
        self.layer_dx = self.n_layer * self.p_vol / max(area, 1e-9)
        self.emit_phase = 0.0
        self.emitted_mass = 0.0
        self.removed_mass = 0.0

        n_layers_max = int(np.ceil(self.in_speed * (self.duration + 0.1) / self.layer_dx)) + 2
        cap = len(w_pos) + n_layers_max * self.n_layer + 64
        self.xw = np.zeros((cap, 3))
        self.vw = np.zeros((cap, 3))
        self.side = np.ones(cap)
        self.nw = len(w_pos)
        self.xw[: self.nw] = w_pos
        self.vw[: self.nw] = w_vel

        # ------------------------------------------------------- boundaries
        self.p_rad = 0.45 * self.spacing
        self.tank_lo = np.array([-1.580, -1.380, 0.0])
        self.tank_hi = np.array([2.080, 1.380, 3.20])
        self.contact_len = 0.5 * self.spacing + 0.5 * self.cloth_thickness

        self.side[: self.nw] = self._sheet_side(self.xw[: self.nw])

        # ------------------------------------------------ reporting buffers
        self.time = 0.0
        self._ext_bnd_w = np.zeros((cap, 3))
        self._ext_bnd_c = np.zeros((self.nc, 3))
        self._cpl_pairs = None
        self._diag = {
            "substep_dt": self.SUBSTEP_DT,
            "iterations": self.N_ITER,
            "water_particles_initial": int(self.nw),
            "cloth_vertices": int(self.nc),
            "method": "PBF water + XPBD cloth shell, symmetric two-way contact coupling",
        }

    def _load_topology(self):
        import os

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "assets", "curtain_topology.npz")
        with np.load(path) as data:
            return {k: np.array(data[k]) for k in data.files}

    def _kernel_setup(self):
        h = self.h
        self.k_poly6 = 315.0 / (64.0 * np.pi * h ** 9)
        self.k_spiky = -45.0 / (np.pi * h ** 6)
        # effective rest density of the discretisation on the rest lattice
        s = self.spacing
        n = int(np.ceil(h / s)) + 1
        g = np.arange(-n, n + 1) * s
        gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
        r2 = gx ** 2 + gy ** 2 + gz ** 2
        r2 = r2[r2 < h * h]
        self.rho0 = float(self.mw * np.sum(self.k_poly6 * (h * h - r2) ** 3))

    def _poly6(self, r):
        q = np.maximum(self.h * self.h - r * r, 0.0)
        return self.k_poly6 * q ** 3

    def _emit_template(self):
        s = self.spacing
        n = int(np.ceil(self.in_radius / s)) + 1
        k = (np.arange(-n, n + 1) + 0.5) * s
        a, b = np.meshgrid(k, k, indexing="ij")
        a = a.ravel()
        b = b.ravel()
        keep = (a * a + b * b) <= self.in_radius ** 2
        return np.stack((a[keep], b[keep]), axis=1)

    # ------------------------------------------------------------- geometry
    def _tri_data(self):
        f = self.faces
        a = self.xc[f[:, 0]]
        b = self.xc[f[:, 1]]
        c = self.xc[f[:, 2]]
        nrm = np.cross(b - a, c - a)
        ln = _norm(nrm)
        nrm = nrm / np.maximum(ln, 1e-14)[:, None]
        cen = (a + b + c) / 3.0
        return a, b, c, nrm, cen

    def _sheet_side(self, pts):
        """Signed side (+/-1) of points relative to the cloth sheet."""
        a, b, c, nrm, cen = self._tri_data()
        tree = cKDTree(cen)
        d, j = tree.query(pts, k=1)
        j = np.clip(j, 0, len(cen) - 1)
        q = _closest_point_on_triangle(pts, a[j], b[j], c[j])
        g = _dot(pts - q, nrm[j])
        s = np.sign(g)
        s[s == 0.0] = 1.0
        return s

    def _contact_pairs(self, xw, nw):
        """Candidate water/cloth contact pairs (particle index, triangle index)."""
        a, b, c, nrm, cen = self._tri_data()
        lo = cen.min(axis=0) - (self.CONTACT_SEARCH + 0.12)
        hi = cen.max(axis=0) + (self.CONTACT_SEARCH + 0.12)
        p = xw[:nw]
        near = np.all((p >= lo) & (p <= hi), axis=1)
        cand = np.nonzero(near)[0]
        if cand.size == 0:
            return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
        tree = cKDTree(cen)
        d, j = tree.query(p[cand], k=4, distance_upper_bound=self.CONTACT_SEARCH)
        d = np.atleast_2d(d)
        j = np.atleast_2d(j)
        ok = np.isfinite(d) & (j < len(cen))
        pi = np.repeat(cand, d.shape[1])[ok.ravel()]
        tj = j.ravel()[ok.ravel()]
        return pi.astype(np.int64), tj.astype(np.int64)

    # ------------------------------------------------------------ time step
    def _emit(self, t, dt):
        if not (self.in_t0 - 1e-12 <= t <= self.in_t1 + 1e-12):
            return
        self.emit_phase += self.in_speed * dt
        while self.emit_phase >= self.layer_dx:
            self.emit_phase -= self.layer_dx
            k = self.n_layer
            if self.nw + k > len(self.xw):
                return
            tpl = self.layer_template
            # nozzle axis is +x: template spans the (y, z) disc
            pts = np.zeros((k, 3))
            pts[:, 0] = self.in_center[0] + 0.02
            pts[:, 1] = self.in_center[1] + tpl[:, 0]
            pts[:, 2] = self.in_center[2] + tpl[:, 1]
            sl = slice(self.nw, self.nw + k)
            self.xw[sl] = pts
            self.vw[sl] = self.in_vel
            self.side[sl] = 1.0
            self.nw += k
            self.emitted_mass += k * self.mw

    def _substep(self, t, dt, record):
        self._emit(t, dt)
        nw = self.nw
        g = self.gravity
        pin = self.pinned

        # Accumulated displacement imposed by the pinned-support constraint.
        # m * (sum of pin corrections) / dt^2 is exactly the support reaction.
        pin_disp = np.zeros((self.nc, 3))

        def _pin(x):
            if record:
                pin_disp[pin] += self.pin_rest - x[pin]
            x[pin] = self.pin_rest

        # ---- prediction ------------------------------------------------
        vw = self.vw[:nw]
        vw += dt * g
        xw0 = self.xw[:nw].copy()
        xw = xw0 + dt * vw

        self.vc *= 0.995  # light aerodynamic damping of the sheet
        self.vc += dt * g
        xc0 = self.xc.copy()
        xc = xc0 + dt * self.vc
        _pin(xc)

        # ---- neighbour structures (fixed within the substep) -----------
        tree = cKDTree(xw)
        pairs = tree.query_pairs(self.h, output_type="ndarray")
        if len(pairs):
            pi = pairs[:, 0]
            pj = pairs[:, 1]
        else:
            pi = np.zeros(0, dtype=np.int64)
            pj = np.zeros(0, dtype=np.int64)

        xc_save = self.xc
        self.xc = xc
        cpi, ctj = self._contact_pairs(xw, nw)
        self.xc = xc_save

        lam_edge = np.zeros(len(self.edges)) if record else None
        cpl_force_w = np.zeros((nw, 3)) if record else None
        cpl_force_c = np.zeros((self.nc, 3)) if record else None
        lam_cpl = np.zeros(len(cpi)) if record else None
        cpl_nrm = np.zeros((len(cpi), 3)) if record else None
        bnd_w = np.zeros((nw, 3))
        bnd_c = np.zeros((self.nc, 3))

        w_p = 1.0 / self.mw

        for _ in range(self.N_ITER):
            # ---------- cloth membrane / bending constraints ------------
            for _p in range(self.N_CLOTH_PASS):
                i0 = self.edges[:, 0]
                i1 = self.edges[:, 1]
                d = xc[i1] - xc[i0]
                ln = _norm(d)
                inv = 1.0 / np.maximum(ln, 1e-12)
                n = d * inv[:, None]
                Cv = ln - self.e_rest
                w0 = self.wc[i0]
                w1 = self.wc[i1]
                wsum = w0 + w1
                lam = np.where(wsum > 0.0, -Cv / np.maximum(wsum, 1e-18), 0.0)
                lam *= self.e_stiff
                if record:
                    lam_edge += np.where(self.e_structural, lam, 0.0)
                imp = lam[:, None] * n
                acc = np.zeros_like(xc)
                cnt = np.zeros(self.nc)
                for comp in range(3):
                    acc[:, comp] += np.bincount(
                        i0, weights=(-w0 * imp[:, comp]), minlength=self.nc
                    )
                    acc[:, comp] += np.bincount(
                        i1, weights=(w1 * imp[:, comp]), minlength=self.nc
                    )
                cnt += np.bincount(i0, minlength=self.nc)
                cnt += np.bincount(i1, minlength=self.nc)
                xc += 0.85 * acc / np.maximum(cnt, 1.0)[:, None]
                _pin(xc)

            # ---------- water density constraint (PBF) ------------------
            if len(pi):
                d = xw[pi] - xw[pj]
                r = _norm(d)
                Wv = self._poly6(r)
                rho = self.mw * self._poly6(np.zeros(nw))
                rho += self.mw * np.bincount(pi, weights=Wv, minlength=nw)
                rho += self.mw * np.bincount(pj, weights=Wv, minlength=nw)
                Cd = np.maximum(rho / self.rho0 - 1.0, 0.0)
                coef = self.k_spiky * (self.h - r) ** 2 / np.maximum(r, 1e-12)
                gij = coef[:, None] * d  # grad_{x_i} W(x_i - x_j)
                sc = self.mw / self.rho0
                gs = gij * sc
                sq = _dot(gs, gs)
                sumg = np.zeros((nw, 3))
                for comp in range(3):
                    sumg[:, comp] += np.bincount(pi, weights=gs[:, comp], minlength=nw)
                    sumg[:, comp] -= np.bincount(pj, weights=gs[:, comp], minlength=nw)
                S = _dot(sumg, sumg)
                S += np.bincount(pi, weights=sq, minlength=nw)
                S += np.bincount(pj, weights=sq, minlength=nw)
                lam_f = -Cd / (S + 30.0)
                fac = (lam_f[pi] + lam_f[pj]) * sc
                dp = fac[:, None] * gij
                corr = np.zeros((nw, 3))
                for comp in range(3):
                    corr[:, comp] += np.bincount(pi, weights=dp[:, comp], minlength=nw)
                    corr[:, comp] -= np.bincount(pj, weights=dp[:, comp], minlength=nw)
                mx = 0.35 * self.spacing
                cl = _norm(corr)
                over = cl > mx
                if np.any(over):
                    corr[over] *= (mx / cl[over])[:, None]
                xw += corr

            # ---------- two-way water / cloth contact -------------------
            if len(cpi):
                f = self.faces[ctj]
                a = xc[f[:, 0]]
                b = xc[f[:, 1]]
                c = xc[f[:, 2]]
                nrm = np.cross(b - a, c - a)
                nrm = nrm / np.maximum(_norm(nrm), 1e-14)[:, None]
                p = xw[cpi]
                q = _closest_point_on_triangle(p, a, b, c)
                s = self.side[cpi]
                gap = s * _dot(p - q, nrm)
                u, v, w2 = _barycentric(q, a, b, c)
                bw = np.stack((u, v, w2), axis=1)
                wk = self.wc[f]
                w_cl = np.einsum("ij,ij->i", bw * bw, wk)
                act = gap < self.contact_len
                Cc = np.where(act, gap - self.contact_len, 0.0)
                den = w_p + w_cl
                lam_c = np.where(act, -Cc / np.maximum(den, 1e-18), 0.0)
                nvec = (s[:, None] * nrm)
                # particle side (Jacobi averaged over its own contacts)
                dpw = (w_p * lam_c)[:, None] * nvec
                accw = np.zeros((nw, 3))
                cntw = np.zeros(nw)
                for comp in range(3):
                    accw[:, comp] += np.bincount(cpi, weights=dpw[:, comp], minlength=nw)
                cntw += np.bincount(cpi, weights=act.astype(np.float64), minlength=nw)
                xw += accw / np.maximum(cntw, 1.0)[:, None]
                # cloth side
                accc = np.zeros((self.nc, 3))
                cntc = np.zeros(self.nc)
                for kk in range(3):
                    vid = f[:, kk]
                    coefk = -(bw[:, kk] * self.wc[vid] * lam_c)
                    for comp in range(3):
                        accc[:, comp] += np.bincount(
                            vid, weights=coefk * nvec[:, comp], minlength=self.nc
                        )
                    cntc += np.bincount(vid, weights=act.astype(np.float64), minlength=self.nc)
                xc += accc / np.maximum(cntc, 1.0)[:, None]
                xc[self.pinned] = self.pin_rest
                if record:
                    lam_cpl += lam_c
                    cpl_nrm[:] = nvec
                    for comp in range(3):
                        cpl_force_w[:, comp] += np.bincount(
                            cpi, weights=lam_c * nvec[:, comp], minlength=nw
                        )
                    for kk in range(3):
                        vid = f[:, kk]
                        for comp in range(3):
                            cpl_force_c[:, comp] -= np.bincount(
                                vid,
                                weights=bw[:, kk] * lam_c * nvec[:, comp],
                                minlength=self.nc,
                            )

            # ---------- static boundaries -------------------------------
            lo = self.tank_lo + self.p_rad
            hi = self.tank_hi - self.p_rad
            new = np.clip(xw, lo, hi)
            bnd_w += new - xw
            xw = new
            clo = self.tank_lo + 0.01
            chi = self.tank_hi - 0.01
            newc = np.clip(xc, clo, chi)
            bnd_c += newc - xc
            xc = newc
            xc[self.pinned] = self.pin_rest

        # ---- velocity update -------------------------------------------
        inv_dt = 1.0 / dt
        vw_new = (xw - xw0) * inv_dt
        vc_new = (xc - xc0) * inv_dt
        vc_new[self.pinned] = 0.0

        # tangential friction at water/cloth contacts (momentum conserving)
        if len(cpi):
            f = self.faces[ctj]
            a = xc[f[:, 0]]
            b = xc[f[:, 1]]
            c = xc[f[:, 2]]
            nrm = np.cross(b - a, c - a)
            nrm = nrm / np.maximum(_norm(nrm), 1e-14)[:, None]
            p = xw[cpi]
            q = _closest_point_on_triangle(p, a, b, c)
            gap = self.side[cpi] * _dot(p - q, nrm)
            act = gap < 1.15 * self.contact_len
            if np.any(act):
                sel = np.nonzero(act)[0]
                fs = f[sel]
                u, v, w2 = _barycentric(q[sel], a[sel], b[sel], c[sel])
                bw = np.stack((u, v, w2), axis=1)
                vcl = (
                    bw[:, 0:1] * vc_new[fs[:, 0]]
                    + bw[:, 1:2] * vc_new[fs[:, 1]]
                    + bw[:, 2:3] * vc_new[fs[:, 2]]
                )
                rel = vw_new[cpi[sel]] - vcl
                nn = nrm[sel]
                vt = rel - _dot(rel, nn)[:, None] * nn
                w_cl = np.einsum("ij,ij->i", bw * bw, self.wc[fs])
                den = np.maximum(w_p + w_cl, 1e-18)
                mu = 0.22
                jimp = -(mu * vt) / den[:, None]
                accw = np.zeros((nw, 3))
                cw = np.zeros(nw)
                for comp in range(3):
                    accw[:, comp] += np.bincount(
                        cpi[sel], weights=w_p * jimp[:, comp], minlength=nw
                    )
                cw += np.bincount(cpi[sel], weights=np.ones(len(sel)), minlength=nw)
                vw_new += accw / np.maximum(cw, 1.0)[:, None]
                accc = np.zeros((self.nc, 3))
                cc = np.zeros(self.nc)
                for kk in range(3):
                    vid = fs[:, kk]
                    coefk = -bw[:, kk] * self.wc[vid]
                    for comp in range(3):
                        accc[:, comp] += np.bincount(
                            vid, weights=coefk * jimp[:, comp], minlength=self.nc
                        )
                    cc += np.bincount(vid, weights=np.ones(len(sel)), minlength=self.nc)
                vc_new += accc / np.maximum(cc, 1.0)[:, None]
                vc_new[self.pinned] = 0.0

        # XSPH viscosity for the liquid
        if len(pi):
            d = xw[pi] - xw[pj]
            r = _norm(d)
            Wv = self._poly6(r)
            dv = vw_new[pj] - vw_new[pi]
            acc = np.zeros((nw, 3))
            for comp in range(3):
                acc[:, comp] += np.bincount(pi, weights=Wv * dv[:, comp], minlength=nw)
                acc[:, comp] -= np.bincount(pj, weights=Wv * dv[:, comp], minlength=nw)
            vw_new += 0.06 * (self.mw / self.rho0) * acc

        # boundary velocity response (no-penetration + mild wall friction)
        lo = self.tank_lo + self.p_rad
        hi = self.tank_hi - self.p_rad
        for axis in range(3):
            at_lo = xw[:, axis] <= lo[axis] + 1e-9
            at_hi = xw[:, axis] >= hi[axis] - 1e-9
            m = at_lo & (vw_new[:, axis] < 0.0)
            vw_new[m, axis] = 0.0
            m = at_hi & (vw_new[:, axis] > 0.0)
            vw_new[m, axis] = 0.0
            touch = at_lo | at_hi
            if np.any(touch):
                for other in range(3):
                    if other != axis:
                        vw_new[touch, other] *= 0.92

        # speed limiters (stability guards, far above the physical scales)
        sp = _norm(vw_new)
        cap = sp > 15.0
        if np.any(cap):
            vw_new[cap] *= (15.0 / sp[cap])[:, None]
        sp = _norm(vc_new)
        cap = sp > 12.0
        if np.any(cap):
            vc_new[cap] *= (12.0 / sp[cap])[:, None]

        self.xw[:nw] = xw
        self.vw[:nw] = vw_new
        self.xc = xc
        self.vc = vc_new

        # refresh the persistent sheet side away from the sheet
        if nw:
            s_new = self._sheet_side(xw)
            a, b, c, nrm, cen = self._tri_data()
            tree_c = cKDTree(cen)
            dmin, _ = tree_c.query(xw, k=1)
            far = dmin > 0.16
            self.side[:nw][far] = s_new[far]

        if record:
            self._record = {
                "bnd_w": bnd_w / (dt * dt) * self.mw,
                "bnd_c": bnd_c / (dt * dt) * self.mc[:, None],
                "lam_edge": lam_edge / (dt * dt) if lam_edge is not None else None,
                "cpl_force_c": cpl_force_c / (dt * dt) if cpl_force_c is not None else None,
                "cpi": cpi,
                "ctj": ctj,
                "lam_cpl": lam_cpl / (dt * dt) if lam_cpl is not None else None,
                "cpl_nrm": cpl_nrm,
                "dt": dt,
            }

    # ------------------------------------------------------------- stepping
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 1e-12:
            dt = min(self.SUBSTEP_DT, target - self.time)
            if dt < 1e-9:
                break
            last = (target - (self.time + dt)) <= 1e-12
            self._substep(self.time, dt, record=last)
            self.time += dt
        if not hasattr(self, "_record"):
            # t = 0 frame: the reporting quantities (constraint reactions) are
            # only defined for a finite step, so probe one regular substep on a
            # saved copy of the state and then restore the initial condition
            # exactly.  A micro dt is not usable here because the constraint
            # impulses are converted to forces through 1/dt^2.
            saved = (
                self.xc.copy(), self.vc.copy(), self.xw.copy(), self.vw.copy(),
                self.side.copy(), self.nw, self.emit_phase, self.emitted_mass,
                self.removed_mass,
            )
            self._substep(0.0, self.SUBSTEP_DT, record=True)
            (self.xc, self.vc, self.xw, self.vw, self.side, self.nw,
             self.emit_phase, self.emitted_mass, self.removed_mass) = saved
            # recorded per-particle arrays were sized for the probed step; trim
            # or pad them back to the restored particle count
            self._resize_record(self.nw)
            self.time = 0.0
        return self._build_frame(target)

    def _resize_record(self, nw: int) -> None:
        """Match recorded per-water-particle arrays to the current count."""
        rec = self._record
        bw = rec["bnd_w"]
        if len(bw) != nw:
            out = np.zeros((nw, 3))
            k = min(len(bw), nw)
            out[:k] = bw[:k]
            rec["bnd_w"] = out
        cpi = rec["cpi"]
        if cpi is not None and len(cpi):
            keep = cpi < nw
            rec["cpi"] = cpi[keep]
            rec["ctj"] = rec["ctj"][keep]
            if rec["lam_cpl"] is not None:
                rec["lam_cpl"] = rec["lam_cpl"][keep]
            if rec["cpl_nrm"] is not None:
                rec["cpl_nrm"] = rec["cpl_nrm"][keep]

    # -------------------------------------------------------------- output
    def _build_frame(self, t: float) -> StateFrame:
        nw = self.nw
        nc = self.nc
        n = nc + nw
        rec = self._record

        position = np.vstack((self.xc, self.xw[:nw]))
        velocity = np.vstack((self.vc, self.vw[:nw]))
        mass = np.concatenate((self.mc, np.full(nw, self.mw)))
        body_id = np.concatenate(
            (np.zeros(nc, dtype=np.int32), np.ones(nw, dtype=np.int32))
        )

        # ---- external force: gravity + static-boundary / support reaction
        ext = mass[:, None] * self.gravity[None, :]
        ext[nc:] += rec["bnd_w"]
        ext[:nc] += rec["bnd_c"]
        lam_edge = rec["lam_edge"]
        if lam_edge is not None:
            i0 = self.edges[:, 0]
            i1 = self.edges[:, 1]
            d = self.xc[i1] - self.xc[i0]
            nrm = d / np.maximum(_norm(d), 1e-12)[:, None]
            fint = np.zeros((nc, 3))
            imp = lam_edge[:, None] * nrm
            for comp in range(3):
                fint[:, comp] -= np.bincount(i0, weights=imp[:, comp], minlength=nc)
                fint[:, comp] += np.bincount(i1, weights=imp[:, comp], minlength=nc)
            fc = rec["cpl_force_c"]
            if fc is not None:
                fint += fc
            pin = self.pinned
            # the pinned support supplies exactly the reaction that holds the
            # top row: m*a = F_ext + F_int = 0
            ext[:nc][pin] = -fint[pin]

        # ---- interaction (coupling) forces, listed as +/- pairs ---------
        cpi = rec["cpi"]
        lam_cpl = rec["lam_cpl"]
        if cpi is not None and len(cpi) and lam_cpl is not None:
            fvec = lam_cpl[:, None] * rec["cpl_nrm"]
            keep = np.abs(lam_cpl) > 0.0
            fvec = fvec[keep]
            if len(fvec) > self.MAX_CONTACT_ROWS // 2:
                order = np.argsort(-_norm(fvec))[: self.MAX_CONTACT_ROWS // 2]
                fvec = fvec[order]
            interaction = np.vstack((fvec, -fvec)) if len(fvec) else np.zeros((2, 3))
        else:
            interaction = np.zeros((2, 3))

        # ---- contact signed distances at the reported state ------------
        gaps = []
        pi2, tj2 = self._contact_pairs(self.xw, nw)
        if len(pi2):
            a, b, c, nrm, cen = self._tri_data()
            f = self.faces[tj2]
            p = self.xw[pi2]
            q = _closest_point_on_triangle(p, a[tj2], b[tj2], c[tj2])
            gg = self.side[pi2] * _dot(p - q, nrm[tj2]) - self.contact_len
            gaps.append(gg)
        lo = self.tank_lo + self.p_rad
        hi = self.tank_hi - self.p_rad
        dw = np.minimum(self.xw[:nw] - lo, hi - self.xw[:nw]).min(axis=1)
        near = dw < 0.10
        if np.any(near):
            gaps.append(dw[near])
        dc = np.minimum(self.xc - (self.tank_lo + 0.01), (self.tank_hi - 0.01) - self.xc).min(
            axis=1
        )
        nearc = dc < 0.10
        if np.any(nearc):
            gaps.append(dc[nearc])
        if gaps:
            csd = np.concatenate(gaps)
        else:
            csd = np.array([self.contact_len])
        if len(csd) > self.MAX_CONTACT_ROWS:
            order = np.argsort(csd)[: self.MAX_CONTACT_ROWS]
            csd = csd[order]

        # ---- mass bookkeeping -------------------------------------------
        mass_quantity = np.zeros((n, 2))
        mass_quantity[:nc, 0] = self.mc
        mass_quantity[nc:, 1] = self.mw
        cum_source = np.array([0.0, self.emitted_mass - self.removed_mass])

        pinned = np.zeros(n)
        pinned[:nc] = self.pinned.astype(np.float64)
        phase = body_id.astype(np.float64)

        water = self.xw[:nw]
        frame = StateFrame(
            time=float(t),
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "pinned": pinned,
                "material_phase": phase,
                "mass_quantity": mass_quantity,
                "phase_id": phase,
            },
            fields={
                "contact_signed_distance": np.ascontiguousarray(csd, dtype=np.float64),
                "cumulative_mass_source": cum_source,
                "external_force": np.ascontiguousarray(ext, dtype=np.float64),
                "interaction_force": np.ascontiguousarray(interaction, dtype=np.float64),
            },
            scalars={
                "contact_length_scale": float(self.contact_len),
                "characteristic_length": float(self.mean_edge),
                "water_density": float(self.rho_phys),
                "cloth_thickness": float(self.cloth_thickness),
                "particle_spacing": float(self.spacing),
                "smoothing_radius": float(self.h),
                "liquid_phase_id": 1.0,
                "water_particle_count": float(nw),
                "water_mass_total": float(nw * self.mw),
                "cloth_mass_total": float(self.mc.sum()),
                "cloth_max_x": float(self.xc[:, 0].max()),
                "water_beyond_cloth_mass": float(
                    self.mw * np.count_nonzero(self.side[:nw] < 0.0)
                ),
                "water_top_z": float(water[:, 2].max()) if nw else 0.0,
            },
        )
        frame.validate()
        self._diag["water_particles"] = int(nw)
        self._diag["final_time"] = float(t)
        self._diag["cloth_max_deflection_x"] = float(self.xc[:, 0].max())
        self._diag["water_mass_total"] = float(nw * self.mw)
        self._diag["min_contact_gap"] = float(csd.min())
        return frame

    def diagnostics(self) -> dict:
        return dict(self._diag)
