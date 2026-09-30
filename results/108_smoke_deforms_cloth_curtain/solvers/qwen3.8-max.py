"""Two-way coupled smoke-flow / hanging-cloth-curtain solver.

Physics:
  * Fluid: incompressible air on a coarse collocated grid (63x35x35, exactly
    2x downsampled from the declared 126x70x70 voxel grid) advanced with a
    semi-Lagrangian advection + DCT (Neumann) pressure-projection scheme.
    A prescribed smoke-bearing jet enters through the -x face inside the
    nozzle radius with the configured start/ramp; the floor is a solid wall,
    the remaining faces are open.
  * Cloth: mass-spring triangular lattice (structural + shear + bending
    springs with damping), semi-implicit Euler at 0.0025 s substeps, two top
    corners pinned to the clamps, floor contact clamped.
  * Two-way coupling: the sampled fluid velocity exerts normal (pressure-like)
    and tangential (friction) drag on the cloth surface; the moving
    impermeable cloth removes the relative normal fluid velocity in a band
    around the surface via an immersed-boundary projection (A u - b penalty),
    so the flow and smoke bend around the moving sheet and re-route as it
    opens and closes.

All required raw trajectory fields are exported per frame; budget terms are
accumulated in time (never duplicated from exported snapshots).
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import map_coordinates

RHO_AIR = 1.2


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)
        dom = cfg["domain"]
        self.low = np.asarray(dom["bounds_min"], float)
        self.high = np.asarray(dom["bounds_max"], float)
        self.fine_shape = tuple(int(n) for n in dom["grid_shape"])  # 126,70,70
        self.ref = 2
        self.shape = tuple(n // self.ref for n in self.fine_shape)  # 63,35,35
        nx, ny, nz = self.shape
        self.dx = float((self.high[0] - self.low[0]) / nx)
        self.dy = float((self.high[1] - self.low[1]) / ny)
        self.dz = float((self.high[2] - self.low[2]) / nz)
        self.xc = self.low[0] + (np.arange(nx) + 0.5) * self.dx
        self.yc = self.low[1] + (np.arange(ny) + 0.5) * self.dy
        self.zc = self.low[2] + (np.arange(nz) + 0.5) * self.dz
        X, Y, Z = np.meshgrid(self.xc, self.yc, self.zc, indexing="ij")
        self.X, self.Y, self.Z = X, Y, Z
        self.ncells = nx * ny * nz

        jet = cfg["jet"]
        self.jet = jet
        self.jc = np.asarray(jet["center"], float)
        self.jr = float(jet["radius"])
        self.jspeed = float(jet["speed"])
        self.jstart = float(jet["start_time"])
        self.jramp = float(jet["ramp_duration"])
        self.juntil = float(jet["active_until"])
        self.jdens = float(jet["smoke_density"])
        Yf, Zf = np.meshgrid(self.yc, self.zc, indexing="ij")
        self.inflow_mask = ((Yf - self.jc[1]) ** 2 + (Zf - self.jc[2]) ** 2) <= self.jr ** 2

        # initial state fields
        self.u = np.zeros(self.shape)
        self.v = np.zeros(self.shape)
        self.w = np.zeros(self.shape)
        self.smoke = np.zeros(self.shape)
        plug = (X >= self.low[0]) & (X <= self.jc[0] + 0.18) & \
               ((Y - self.jc[1]) ** 2 + (Z - self.jc[2]) ** 2 <= self.jr ** 2)
        self.smoke[plug] = self.jdens

        # DCT Poisson eigenvalues (Neumann, cell-centered)
        lam = (2 * np.cos(np.pi * np.arange(nx) / nx) - 2) / self.dx ** 2
        lamy = (2 * np.cos(np.pi * np.arange(ny) / ny) - 2) / self.dy ** 2
        lamz = (2 * np.cos(np.pi * np.arange(nz) / nz) - 2) / self.dz ** 2
        L = lam[:, None, None] + lamy[None, :, None] + lamz[None, None, :]
        L[0, 0, 0] = 1.0
        self.poisson_lam = L

        # ---------------- cloth lattice ----------------
        cloth = cfg["cloth"]
        self.cnx, self.cnz = (int(v) for v in cloth["grid_shape"])  # 5 (y), 7 (z)
        init = scene.initial_state()
        self.P = np.array(init.position, float)      # [35,3]
        self.V = np.array(init.velocity, float)
        self.M = np.array(init.mass, float)
        self.uv = np.array(init.attributes["uv"], float)
        self.pinned_attr = np.array(init.attributes["pinned"], float)
        self.pin = np.array(cfg["cloth"]["pinned_vertices"], int)
        self.pin_P = self.P[self.pin].copy()
        self.free = np.ones(len(self.P), bool)
        self.free[self.pin] = False
        g = np.asarray(cfg["initial_conditions"]["gravity"], float)
        self.gravity = g
        self.F_grav_total = self.M.sum() * g

        nx_c, nz_c = self.cnx, self.cnz
        dy = float(cloth["width"]) / (nx_c - 1)
        dz = float(cloth["height"]) / (nz_c - 1)
        edges = []  # (ia, ib, rest, k)
        def idx(j, i):
            return j * nx_c + i
        for j in range(nz_c):
            for i in range(nx_c):
                a = idx(j, i)
                if i + 1 < nx_c:
                    edges.append((a, idx(j, i + 1), dy, 3000.0))
                if j + 1 < nz_c:
                    edges.append((a, idx(j + 1, i), dz, 3000.0))
                if i + 1 < nx_c and j + 1 < nz_c:
                    d = float(np.hypot(dy, dz))
                    edges.append((a, idx(j + 1, i + 1), d, 1500.0))
                    edges.append((idx(j, i + 1), idx(j + 1, i), d, 1500.0))
                if i + 2 < nx_c:
                    edges.append((a, idx(j, i + 2), 2 * dy, 200.0))
                if j + 2 < nz_c:
                    edges.append((a, idx(j + 2, i), 2 * dz, 200.0))
        self.e_ia = np.array([e[0] for e in edges])
        self.e_ib = np.array([e[1] for e in edges])
        self.e_rest = np.array([e[2] for e in edges])
        self.e_k = np.array([e[3] for e in edges])
        self.e_c = 0.2 * np.sqrt(self.e_k * 0.0333)

        # ---------------- cloth surface sample points (barycentric) --------
        faces = []
        for j in range(nz_c - 1):
            for i in range(nx_c - 1):
                a = idx(j, i)
                faces.append((a, a + nx_c, a + nx_c + 1))
                faces.append((a, a + nx_c + 1, a + 1))
        self.faces = np.array(faces, int)
        ns = 6
        bary = []
        for i in range(ns):
            for j in range(ns):
                u = (i + 0.5) / ns
                v = (j + 0.5) / ns
                if u + v < 1.0:
                    bary.append((1.0 - u - v, u, v))
        self.bary = np.array(bary)  # [S,3]
        tri = self.faces[:, None, :] + 0  # [F,1,3]
        wts = self.bary[None, :, :]       # [1,S,3]
        self.pt_tri = np.repeat(np.arange(len(self.faces)), len(self.bary))
        self.pt_w = np.tile(wts, (len(self.faces), 1, 1)).reshape(-1, 3)  # [F*S,3]
        self.pt_idx = tri.reshape(len(self.faces), 1, 3).repeat(len(self.bary), 1).reshape(-1, 3)
        self.npts = len(self.pt_tri)
        self.pt_area = (0.5 * dy * dz) / len(self.bary)

        # splat stencil offsets
        offs = np.array([(a, b, c) for a in range(-2, 3) for b in range(-2, 3) for c in range(-2, 3)], float)
        self.offs = offs
        self.splat_R = 0.24

        # ---------------- boundary samples (walls / inflow) ----------------
        jj, kk = np.nonzero(self.inflow_mask)
        self.b_in_j, self.b_in_k = jj, kk
        fi = np.arange(0, nx, 4)
        fj = np.arange(0, ny, 4)
        FI, FJ = np.meshgrid(fi, fj, indexing="ij")
        self.b_fl_i, self.b_fl_j = FI.ravel(), FJ.ravel()
        B = len(jj) + len(self.b_fl_i)
        self.bnd_normal = np.zeros((B, 3))
        self.bnd_normal[: len(jj), 0] = 1.0
        self.bnd_normal[len(jj):, 2] = 1.0
        self.bnd_target = np.zeros((B, 3))
        self.dA_x = self.dy * self.dz
        self.dA_y = self.dx * self.dz
        self.dA_z = self.dx * self.dy

        # ---------------- accumulators / budgets ----------------
        self.t = 0.0
        self.dt_master = 0.01
        self.n_sub = 4
        N = len(self.P)
        self.weights = np.zeros((2, N))
        self.weights[0, :] = 1.0  # cloth body domain; fluid domain has no state samples
        self.imp_ext = np.zeros((2, 3))
        self.mom_in = np.zeros((2, 3))
        self.mom_out = np.zeros((2, 3))
        self.mass_source = np.zeros(2)
        self.F_aero = np.zeros(3)
        self.F_pin = np.zeros(3)
        self.ref_momentum = RHO_AIR * self.jspeed ** 2 * np.pi * self.jr ** 2
        self.diag = {"steps": 0, "max_smoke": 0.0, "cloth_tip_x": 0.0, "max_speed": 0.0}

        # precompute backtrace index base
        I, J, K = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        self.idx_base = np.stack([I.ravel(), J.ravel(), K.ravel()]).astype(float)

    # -------------------------------------------------------------- helpers
    def jet_speed(self, t: float) -> float:
        if t < self.jstart or t >= self.juntil:
            return 0.0
        r = min(1.0, (t - self.jstart) / self.jramp)
        return self.jspeed * r

    def _advect(self, f, coords, cval=0.0, mode="constant"):
        return map_coordinates(f, coords, order=1, mode=mode, cval=cval).reshape(self.shape)

    def _fluid_step(self, dt, U):
        nx, ny, nz = self.shape
        # semi-Lagrangian backtrace
        s = np.empty_like(self.idx_base)
        s[0] = self.idx_base[0] - dt * self.u.ravel() / self.dx
        s[1] = self.idx_base[1] - dt * self.v.ravel() / self.dy
        s[2] = self.idx_base[2] - dt * self.w.ravel() / self.dz
        un = self._advect(self.u, s, mode="nearest")
        vn = self._advect(self.v, s, mode="nearest")
        wn = self._advect(self.w, s, mode="nearest")
        self.smoke = np.clip(self._advect(self.smoke, s, cval=0.0), 0.0, 1.5)
        # weak buoyancy of hot smoke
        wn = wn + 0.15 * self.smoke * dt
        self.u, self.v, self.w = un, vn, wn
        # smoke injection with the jet
        if U > 0.0:
            self.smoke[0][self.inflow_mask] = self.jdens
        self._project(dt)
        # walls / inflow
        self.w[:, :, 0] = 0.0
        if U > 0.0:
            self.u[0][self.inflow_mask] = U
            self.v[0][self.inflow_mask] = 0.0
            self.w[0][self.inflow_mask] = 0.0

    def _project(self, dt):
        ux = np.pad(self.u, 1, mode="edge")
        vy = np.pad(self.v, 1, mode="edge")
        wz = np.pad(self.w, 1, mode="edge")
        div = (ux[2:, 1:-1, 1:-1] - ux[:-2, 1:-1, 1:-1]) / (2 * self.dx) + \
              (vy[1:-1, 2:, 1:-1] - vy[1:-1, :-2, 1:-1]) / (2 * self.dy) + \
              (wz[1:-1, 1:-1, 2:] - wz[1:-1, 1:-1, :-2]) / (2 * self.dz)
        rhs = dctn(div / dt, norm="ortho")
        rhs[0, 0, 0] = 0.0
        p = idctn(rhs / self.poisson_lam, norm="ortho")
        pp = np.pad(p, 1, mode="edge")
        self.u -= dt * (pp[2:, 1:-1, 1:-1] - pp[:-2, 1:-1, 1:-1]) / (2 * self.dx)
        self.v -= dt * (pp[1:-1, 2:, 1:-1] - pp[1:-1, :-2, 1:-1]) / (2 * self.dy)
        self.w -= dt * (pp[1:-1, 1:-1, 2:] - pp[1:-1, 1:-1, :-2]) / (2 * self.dz)

    # -------------------------------------------------- cloth point states
    def vertex_normals(self):
        p = self.P.reshape(self.cnz, self.cnx, 3)
        ti = np.zeros_like(p)
        tj = np.zeros_like(p)
        ti[:, 1:-1] = p[:, 2:] - p[:, :-2]
        ti[:, 0] = p[:, 1] - p[:, 0]
        ti[:, -1] = p[:, -1] - p[:, -2]
        tj[1:-1] = p[2:] - p[:-2]
        tj[0] = p[1] - p[0]
        tj[-1] = p[-1] - p[-2]
        n = np.cross(ti, tj)
        ln = np.linalg.norm(n, axis=-1, keepdims=True)
        n = np.divide(n, np.maximum(ln, 1e-12))
        return n.reshape(-1, 3)

    def cloth_points(self, NV):
        pts = (self.pt_w[:, 0:1] * self.P[self.pt_idx[:, 0]] +
               self.pt_w[:, 1:2] * self.P[self.pt_idx[:, 1]] +
               self.pt_w[:, 2:3] * self.P[self.pt_idx[:, 2]])
        vel = (self.pt_w[:, 0:1] * NV[self.pt_idx[:, 0]] +
               self.pt_w[:, 1:2] * NV[self.pt_idx[:, 1]] +
               self.pt_w[:, 2:3] * NV[self.pt_idx[:, 2]])
        return pts, vel

    def sample_fluid(self, pts):
        c = np.stack([(pts[:, 0] - self.low[0]) / self.dx - 0.5,
                      (pts[:, 1] - self.low[1]) / self.dy - 0.5,
                      (pts[:, 2] - self.low[2]) / self.dz - 0.5])  # (3, npts)
        uf = map_coordinates(self.u, c, order=1, mode="nearest")
        vf = map_coordinates(self.v, c, order=1, mode="nearest")
        wf = map_coordinates(self.w, c, order=1, mode="nearest")
        return np.stack([uf, vf, wf], axis=1)

    # ------------------------------------------------------- cloth physics
    def aero_forces(self, pts, pvel, nrm):
        uf = self.sample_fluid(pts)
        wrel = uf - pvel
        wn = np.einsum("pd,pd->p", wrel, nrm)
        Fn = (0.5 * RHO_AIR * 1.4 * self.pt_area * np.abs(wn) * wn)[:, None] * nrm
        wt = wrel - wn[:, None] * nrm
        speed_t = np.linalg.norm(wt, axis=1)
        Ft = (0.5 * RHO_AIR * 0.06 * self.pt_area * speed_t)[:, None] * wt
        Fp = Fn + Ft  # per point
        Fv = np.zeros_like(self.P)
        for c in range(3):
            np.add.at(Fv, self.pt_idx[:, c], self.pt_w[:, c:c + 1] * Fp)
        return Fv

    def cloth_substep(self, dt, Faero):
        d = self.P[self.e_ib] - self.P[self.e_ia]
        L = np.linalg.norm(d, axis=1)
        L = np.maximum(L, 1e-9)
        dh = d / L[:, None]
        relv = self.V[self.e_ib] - self.V[self.e_ia]
        dL = np.einsum("ed,ed->e", relv, dh)
        fm = self.e_k * (L - self.e_rest) + self.e_c * dL
        F = np.zeros_like(self.P)
        Fd = fm[:, None] * dh
        np.add.at(F, self.e_ia, Fd)
        np.add.at(F, self.e_ib, -Fd)
        F += self.M[:, None] * self.gravity[None, :]
        F += Faero
        F -= 0.02 * self.V
        self.V += dt * F / self.M[:, None]
        # pin constraints (record reaction)
        Fpin = np.zeros(3)
        vp = self.V[self.pin]
        Fpin = (-self.M[self.pin, None] * vp / dt).sum(axis=0)
        self.V[self.pin] = 0.0
        self.P += dt * self.V
        self.P[self.pin] = self.pin_P
        # floor + domain clamp
        below = self.P[:, 2] < 0.03
        if np.any(below):
            self.P[below, 2] = 0.03
            self.V[below, 2] = np.maximum(self.V[below, 2], 0.0)
        self.P = np.clip(self.P, self.low + 0.01, self.high - 0.01)
        return Fpin

    def cloth_normals_for_points(self):
        nv = self.vertex_normals()
        n = (self.pt_w[:, 0:1] * nv[self.pt_idx[:, 0]] +
             self.pt_w[:, 1:2] * nv[self.pt_idx[:, 1]] +
             self.pt_w[:, 2:3] * nv[self.pt_idx[:, 2]])
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, np.maximum(ln, 1e-12))

    # --------------------------------------------- cloth -> fluid blocking
    def apply_blocking(self, pts, pvel, nrm):
        nx, ny, nz = self.shape
        base = np.stack([np.round((pts[:, 0] - self.low[0]) / self.dx - 0.5),
                         np.round((pts[:, 1] - self.low[1]) / self.dy - 0.5),
                         np.round((pts[:, 2] - self.low[2]) / self.dz - 0.5)]).astype(int)
        P = base[:, None, :] + self.offs[None, :, :].astype(int)  # [npts,125,3]
        inside = (P[..., 0] >= 0) & (P[..., 0] < nx) & (P[..., 1] >= 0) & \
                 (P[..., 1] < ny) & (P[..., 2] >= 0) & (P[..., 2] < nz)
        Pc = np.clip(P, 0, [nx - 1, ny - 1, nz - 1])
        cx = self.low[0] + (Pc[..., 0] + 0.5) * self.dx
        cy = self.low[1] + (Pc[..., 1] + 0.5) * self.dy
        cz = self.low[2] + (Pc[..., 2] + 0.5) * self.dz
        ddx = cx - pts[:, None, 0]
        ddy = cy - pts[:, None, 1]
        ddz = cz - pts[:, None, 2]
        dist = np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
        w = np.clip(1.0 - dist / self.splat_R, 0.0, 1.0) * inside
        flat = (Pc[..., 0] * ny + Pc[..., 1]) * nz + Pc[..., 2]
        flat = flat.ravel()
        wf = w.ravel()
        keep = wf > 1e-6
        flat, wf = flat[keep], wf[keep]
        nxv = np.repeat(nrm, w.shape[1], axis=0)[keep]
        vpv = np.repeat(pvel, w.shape[1], axis=0)[keep]
        A = np.zeros((self.ncells, 9))
        b = np.zeros((self.ncells, 3))
        W = np.zeros(self.ncells)
        outer = nxv[:, :, None] * nxv[:, None, :]  # [k,3,3]
        coef = wf * (np.einsum("kd,kd->k", vpv, nxv))
        for a in range(3):
            for bb in range(3):
                A[:, a * 3 + bb] += np.bincount(flat, weights=outer[:, a, bb], minlength=self.ncells)
            b[:, a] += np.bincount(flat, weights=coef * nxv[:, a], minlength=self.ncells)
        W += np.bincount(flat, weights=wf, minlength=self.ncells)
        Wc = np.clip(W, 0.0, 1.0)
        Uf = np.stack([self.u.ravel(), self.v.ravel(), self.w.ravel()], axis=1)
        for _ in range(2):
            Au = np.einsum("cij,cj->ci", A.reshape(self.ncells, 3, 3), Uf)
            Uf -= Wc[:, None] * (Au - b)
        self.u = Uf[:, 0].reshape(self.shape)
        self.v = Uf[:, 1].reshape(self.shape)
        self.w = Uf[:, 2].reshape(self.shape)

    # ------------------------------------------------------- budget fluxes
    def boundary_flux(self, dt):
        # inflow/outflow momentum + mass across open faces (floor excluded)
        def face(flux_n, comps_u, dA):
            fin = np.maximum(-flux_n, 0.0)
            fout = np.maximum(flux_n, 0.0)
            for c in range(3):
                self.mom_in[1, c] += RHO_AIR * dA * dt * float(np.sum(fin * comps_u[c]))
                self.mom_out[1, c] += RHO_AIR * dA * dt * float(np.sum(fout * comps_u[c]))
            return float(np.sum(fin)) * dA
        # -x face: outward normal (-1,0,0), flux_n = -u
        m_in = face(-self.u[0], (self.u[0], self.v[0], self.w[0]), self.dA_x)
        # +x face
        m_out = face(self.u[-1], (self.u[-1], self.v[-1], self.w[-1]), self.dA_x)
        # -y / +y
        face(-self.v[:, 0], (self.u[:, 0], self.v[:, 0], self.w[:, 0]), self.dA_y)
        face(self.v[:, -1], (self.u[:, -1], self.v[:, -1], self.w[:, -1]), self.dA_y)
        # +z top
        face(self.w[:, :, -1], (self.u[:, :, -1], self.v[:, :, -1], self.w[:, :, -1]), self.dA_z)
        self.mass_source[1] += RHO_AIR * dt * m_in

    # --------------------------------------------------------------- step
    def _master_step(self, dt):
        t1 = self.t + dt
        U = self.jet_speed(t1)
        nv = self.V.copy()
        nrm_v = self.vertex_normals()
        pts, pvel = self.cloth_points(nv)
        nrm = self.cloth_normals_for_points()
        # aero force on cloth from the current fluid state
        Faero_v = self.aero_forces(pts, pvel, nrm)
        F_aero_net = Faero_v.sum(axis=0)
        # fluid advance
        self._fluid_step(dt, U)
        # cloth acts back on the fluid (impermeable moving boundary)
        self.apply_blocking(pts, pvel, nrm)
        # cloth substeps
        dt_s = dt / self.n_sub
        Fpin_acc = np.zeros(3)
        for _ in range(self.n_sub):
            Fpin_acc += self.cloth_substep(dt_s, Faero_v)
        F_pin = Fpin_acc / self.n_sub
        # budgets
        self.F_aero = F_aero_net
        self.F_pin = F_pin
        ext0 = self.F_grav_total + F_aero_net + F_pin
        ext1 = -F_aero_net
        self.imp_ext[0] += ext0 * dt
        self.imp_ext[1] += ext1 * dt
        self.boundary_flux(dt)
        self.t = t1
        self.diag["steps"] += 1

    # -------------------------------------------------------------- frame
    def _make_frame(self, time):
        from worldsolver.contracts import StateFrame
        r = self.ref
        smoke_f = np.repeat(np.repeat(np.repeat(self.smoke, r, 0), r, 1), r, 2).astype(np.float32)
        vel_c = np.stack([self.u, self.v, self.w], axis=-1)
        vel_f = np.repeat(np.repeat(np.repeat(vel_c, r, 0), r, 1), r, 2).astype(np.float32)
        U = self.jet_speed(self.t)
        nin = len(self.b_in_j)
        B = nin + len(self.b_fl_i)
        target = np.zeros((B, 3))
        target[:nin, 0] = U
        bvel = np.zeros((B, 3))
        bvel[:nin, 0] = self.u[0][self.b_in_j, self.b_in_k]
        bvel[:nin, 1] = self.v[0][self.b_in_j, self.b_in_k]
        bvel[:nin, 2] = self.w[0][self.b_in_j, self.b_in_k]
        bvel[nin:, 0] = self.u[self.b_fl_i, self.b_fl_j, 0]
        bvel[nin:, 1] = self.v[self.b_fl_i, self.b_fl_j, 0]
        bvel[nin:, 2] = self.w[self.b_fl_i, self.b_fl_j, 0]
        ext_force = np.zeros((2, 3, 3))
        ext_force[0, 0] = self.F_grav_total
        ext_force[0, 1] = self.F_aero
        ext_force[0, 2] = self.F_pin
        ext_force[1, 1] = -self.F_aero
        mass_quantity = np.zeros((len(self.P), 2))
        mass_quantity[:, 0] = self.M
        N = len(self.P)
        frame = StateFrame(
            time=float(time),
            position=self.P.copy(),
            velocity=self.V.copy(),
            mass=self.M.copy(),
            body_id=np.ones(N, dtype=np.int32),
            attributes={
                "uv": self.uv.copy(),
                "pinned": self.pinned_attr.copy(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "smoke_density": smoke_f,
                "fluid_velocity": vel_f,
                "boundary_normal": self.bnd_normal.copy(),
                "boundary_target_velocity": target,
                "boundary_velocity": bvel,
                "budget_domain_weights": self.weights.copy(),
                "budget_external_force": ext_force,
                "budget_cumulative_external_impulse": self.imp_ext.copy(),
                "budget_cumulative_inflow_momentum": self.mom_in.copy(),
                "budget_cumulative_outflow_momentum": self.mom_out.copy(),
                "cumulative_mass_source": self.mass_source.copy(),
                "interaction_force": self.F_aero[None, :].copy(),
            },
            scalars={
                "reference_momentum": float(self.ref_momentum),
                "reference_velocity": float(self.jspeed),
                "jet_start_time": self.jstart,
                "jet_speed": self.jspeed,
            },
        )
        frame.validate()
        self.diag["max_smoke"] = float(self.smoke.max())
        self.diag["max_speed"] = float(max(np.abs(self.u).max(), np.abs(self.v).max(), np.abs(self.w).max()))
        self.diag["cloth_tip_x"] = float(self.P[:, 0].max())
        return frame

    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t < target - 1e-9 and guard < 100000:
            dt = min(self.dt_master, target - self.t)
            self._master_step(dt)
            guard += 1
        return self._make_frame(target)

    def diagnostics(self) -> dict:
        return dict(self.diag)
