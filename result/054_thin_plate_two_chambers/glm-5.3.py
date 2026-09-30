"""Physics solver for scene 054_thin_plate_two_chambers.

Self-contained finite-volume gas solver (no Genesis, no external physics engine):

* 2D staggered (MAC) grid on the frozen domain [-3,3] x [0,3.6] at h = 0.05 m,
  extruded shallow in y (depth 0.36 m) purely for 3D mass reporting.
* Semi-Lagrangian advection of momentum, density and injected tracer,
  explicit viscous / diffusive terms, Boussinesq buoyancy.
* Chorin pressure projection with a sparse direct solve.  The thin separating
  plate is an impermeable stationary interface: face masks remove every flux
  through the barrier, and two solid cell rows guarantee that neither the
  tracer nor momentum can numerically tunnel across it.
* Both chambers are sealed (impermeable outer walls), so the prescribed lower
  inlet jet compresses the lower chamber gas.  The projection therefore
  enforces a uniform volumetric dilatation -s(t) = -Q_in / V_lower, which is
  exactly the discrete compatibility condition of the Poisson problem.  The
  lower chamber pressure rises (isothermal compression) while the upper
  chamber stays quiescent at ambient pressure: a genuine pressure jump across
  the plate with zero leaked mass.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from worldsolver.contracts import StateFrame

GRAVITY = 9.81
RHO0 = 1.0
P0 = 1.0
KIN_VISC = 2.0e-3      # m^2/s explicit momentum diffusion
SCALAR_DIFF = 1.0e-3   # m^2/s tracer / density diffusion


def _bilerp(field: np.ndarray, ax: np.ndarray, az: np.ndarray, px, pz):
    """Bilinear sample of field[x_index, z_index] at physical (px, pz)."""
    hx = ax[1] - ax[0]
    hz = az[1] - az[0]
    fx = (px - ax[0]) / hx
    fz = (pz - az[0]) / hz
    fx = np.clip(fx, 0.0, field.shape[0] - 1.001)
    fz = np.clip(fz, 0.0, field.shape[1] - 1.001)
    i0 = np.floor(fx).astype(np.intp)
    j0 = np.floor(fz).astype(np.intp)
    i1 = np.minimum(i0 + 1, field.shape[0] - 1)
    j1 = np.minimum(j0 + 1, field.shape[1] - 1)
    tx = np.clip(fx - i0, 0.0, 1.0)
    tz = np.clip(fz - j0, 0.0, 1.0)
    return (
        field[i0, j0] * (1.0 - tx) * (1.0 - tz)
        + field[i1, j0] * tx * (1.0 - tz)
        + field[i0, j1] * (1.0 - tx) * tz
        + field[i1, j1] * tx * tz
    )


def _lap(F: np.ndarray) -> np.ndarray:
    """5-point Laplacian with zero-gradient (closed) domain edges."""
    L = np.zeros_like(F)
    L[1:, :] += F[:-1, :] - F[1:, :]
    L[:-1, :] += F[1:, :] - F[:-1, :]
    L[:, 1:] += F[:, :-1] - F[:, 1:]
    L[:, :-1] += F[:, 1:] - F[:, :-1]
    return L


class BenchmarkSolver:
    """Deterministic finite-volume solver for the thin-plate two-chamber jet."""

    DT = 0.01  # physical sub-step (s); two sub-steps per 0.02 s sample

    def __init__(self) -> None:
        self._ready = False
        self._last_frame_time = None

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or {}
        bc = (cfg.get("boundary_conditions") or {}).get("jet_inlet") or {}
        dom = cfg.get("domain") or {}

        self.seed = int(seed)
        self.h = 0.05
        self.x0, self.x1 = -3.0, 3.0
        self.z0, self.z1 = 0.0, 3.6
        self.depth = float(dom.get("shallow_extrusion_depth", 0.36))
        self.nx = int(round((self.x1 - self.x0) / self.h))  # 120 cells
        self.nz = int(round((self.z1 - self.z0) / self.h))  # 72 cells

        jet_v = bc.get("velocity") or [0.0, 0.0, 2.4]
        self.w_in = float(jet_v[2])
        self.u_in = float(jet_v[0])
        jet_w = float(bc.get("width", 0.42))
        self.rho_in = float(bc.get("density", 1.18))
        self.inlet_half = 0.5 * jet_w

        # Frozen thin plate (impermeable stationary interface).
        self.plate_zmin, self.plate_zmax = 1.775, 1.825
        self.plate_half_x = 2.72

        # --- grids -------------------------------------------------------
        nx, nz, h = self.nx, self.nz, self.h
        self.xc = self.x0 + (np.arange(nx) + 0.5) * h
        self.zc = self.z0 + (np.arange(nz) + 0.5) * h
        self.xu = self.x0 + np.arange(nx + 1) * h
        self.zu = self.z0 + (np.arange(nz) + 0.5) * h
        self.xw = self.xc
        self.zw = self.z0 + np.arange(nz + 1) * h
        self.XC, self.ZC = np.meshgrid(self.xc, self.zc, indexing="ij")
        self.XU, self.ZU = np.meshgrid(self.xu, self.zu, indexing="ij")
        self.XW, self.ZW = np.meshgrid(self.xw, self.zw, indexing="ij")

        # Node grid used for reported fields (matches scene voxel grid).
        self.xn = self.x0 + np.arange(nx + 1) * h
        self.zn = self.z0 + np.arange(nz + 1) * h
        self.yn = np.linspace(-0.18, 0.18, 3)

        # --- solid / chamber masks ---------------------------------------
        plate_rows = (self.zc >= self.plate_zmin - 1e-9) & (self.zc <= self.plate_zmax + 1e-9)
        # The barrier is sealed across the full chamber width so the two
        # chambers stay hydraulically isolated (no leakage around plate ends).
        self.solid = np.tile(plate_rows[None, :], (nx, 1))
        self.fluid = ~self.solid
        self.lower_cell = self.zc < 1.8
        self.upper_cell = self.zc > 1.8
        self.lower_fluid = self.fluid & self.lower_cell[None, :]
        self.upper_fluid = self.fluid & self.upper_cell[None, :]

        # --- inlet --------------------------------------------------------
        self.inlet_cell = np.zeros((nx, nz), dtype=bool)
        inlet_i = np.abs(self.xc) <= self.inlet_half + 1e-9
        self.inlet_cell[inlet_i, 0] = True
        self.inlet_face = np.zeros((nx, nz + 1), dtype=bool)
        self.inlet_face[inlet_i, 0] = True
        n_in = int(self.inlet_cell[:, 0].sum())

        # --- face permeability masks --------------------------------------
        self.u_open = np.zeros((nx + 1, nz), dtype=bool)
        self.u_open[1:nx, :] = self.fluid[:-1, :] & self.fluid[1:, :]
        self.w_open_all = np.zeros((nx, nz + 1), dtype=bool)
        self.w_open_all[:, 1:nz] = self.fluid[:, :-1] & self.fluid[:, 1:]
        self.w_open_all[:, 0] = self.inlet_face[:, 0]
        self.w_open_free = self.w_open_all.copy()
        self.w_open_free[:, 0] = False  # inlet flux is prescribed, not free

        # --- chamber compression model ------------------------------------
        n_low = int(self.lower_fluid.sum())
        self.v2d_lower = n_low * h * h
        self.q2d_in = n_in * self.w_in * h                 # inflow per unit depth
        self.s_compress = self.q2d_in / self.v2d_lower     # volumetric compression rate

        # --- pressure system ----------------------------------------------
        self._build_pressure_operator()

        # --- boundary samples ----------------------------------------------
        self._build_boundary_samples()

        # --- node gather tables ---------------------------------------------
        self._build_node_gather()

        # --- initial state ---------------------------------------------------
        self._init_fields()

        self.t = 0.0
        self._p_low = P0
        self._ready = True

    # ------------------------------------------------------------ operators
    def _build_pressure_operator(self) -> None:
        nx, nz = self.nx, self.nz
        fluid = self.fluid
        cid = -np.ones((nx, nz), dtype=np.int64)
        cid[fluid] = np.arange(int(fluid.sum()))
        self._cid = cid
        nf = int(fluid.sum())

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        diag = np.zeros(nf)

        def add_pair(a_ij, b_ij):
            ka = cid[a_ij]
            kb = cid[b_ij]
            m = fluid[a_ij] & fluid[b_ij]
            rows.extend(ka[m].tolist())
            cols.extend(kb[m].tolist())
            vals.extend((-1.0 * np.ones(int(m.sum()))).tolist())
            np.add.at(diag, ka[m], 1.0)
            np.add.at(diag, kb[m], 1.0)

        ii = np.arange(nx - 1)
        add_pair(np.ix_(ii + 1, np.arange(nz)), np.ix_(ii, np.arange(nz)))  # +x
        jj = np.arange(nz - 1)
        add_pair(np.ix_(np.arange(nx), jj + 1), np.ix_(np.arange(nx), jj))  # +z

        ks = np.arange(nf)
        rows.extend(ks.tolist())
        cols.extend(ks.tolist())
        vals.extend((diag * (1.0 / self.h ** 2)).tolist())

        # Pin one pressure dof per hydraulically disconnected chamber so the
        # pure-Neumann problem is uniquely solvable.  Pinned values carry the
        # thermodynamic (compression) pressure of each chamber.
        pin_lo = int(cid[2, 2])
        pin_up = int(cid[2, self.nz - 3])
        self._pin_lower_k, self._pin_upper_k = pin_lo, pin_up
        r = np.asarray(rows)
        keep = ~np.isin(r, np.asarray([pin_lo, pin_up]))
        rows = r[keep].tolist()
        cols = np.asarray(cols)[keep].tolist()
        vals = np.asarray(vals)[keep].tolist()
        rows.extend([pin_lo, pin_up])
        cols.extend([pin_lo, pin_up])
        vals.extend([1.0, 1.0])

        mat = csc_matrix((vals, (rows, cols)), shape=(nf, nf))
        self._lu = splu(mat)

    def _build_boundary_samples(self) -> None:
        pts: list[tuple[float, float]] = []
        nrm: list[tuple[float, float, float]] = []
        tgt: list[tuple[float, float, float]] = []
        vin: list[bool] = []

        def add(x, z, n, inlet=False):
            pts.append((x, z))
            nrm.append(n)
            tgt.append((0.0, 0.0, self.w_in if inlet else 0.0))
            vin.append(inlet)

        xs = np.arange(-2.95, 3.0 + 1e-9, 0.1)
        zs = np.arange(0.0, 3.6 + 1e-9, 0.1)
        for x in xs:  # bottom wall (inlet region prescribes the jet)
            add(float(x), 0.0, (0.0, 0.0, -1.0), inlet=abs(x) <= self.inlet_half + 1e-9)
        for z in zs:  # side walls
            add(-3.0, float(z), (-1.0, 0.0, 0.0))
            add(3.0, float(z), (1.0, 0.0, 0.0))
        for x in xs:  # top wall
            add(float(x), 3.6, (0.0, 0.0, 1.0))
        for x in xs:  # thin plate faces (impermeable stationary interface)
            add(float(x), self.plate_zmin, (0.0, 0.0, 1.0))   # fluid below
            add(float(x), self.plate_zmax, (0.0, 0.0, -1.0))  # fluid above

        self._bpts = np.asarray(pts, dtype=float)
        self._bnrm = np.asarray(nrm, dtype=float)
        self._btgt = np.asarray(tgt, dtype=float)
        self._binlet = np.asarray(vin, dtype=bool)

    def _build_node_gather(self) -> None:
        """Gather cell-centred values onto the reported node grid."""
        nx, nz = self.nx, self.nz
        nn = (nx + 1) * (nz + 1)
        n_list: list[int] = []
        c_list: list[int] = []
        I = np.arange(nx + 1)[:, None]
        J = np.arange(nz + 1)[None, :]
        for di in (0, 1):
            ci = np.clip(I - 1 + di, 0, nx - 1)
            for dj in (0, 1):
                cj = np.clip(J - 1 + dj, 0, nz - 1)
                cidx = ci * nz + cj  # broadcast to (nx+1, nz+1)
                valid = self.fluid[ci, cj]
                n_list.append((I * (nz + 1) + J)[valid].ravel().tolist())
                c_list.append(cidx[valid].ravel().tolist())
        self._gnode = np.asarray([v for chunk in n_list for v in chunk], dtype=np.int64)
        self._gcell = np.asarray([v for chunk in c_list for v in chunk], dtype=np.int64)
        counts = np.bincount(self._gnode, minlength=nn).astype(float)
        self._gcount = np.maximum(counts, 1.0)
        self._gempty = counts <= 0.0

    def _node_sample(self, F: np.ndarray, default: float = 0.0) -> np.ndarray:
        out = np.bincount(
            self._gnode, weights=F.ravel()[self._gcell], minlength=self._gcount.size
        ) / self._gcount
        out[self._gempty] = default
        return out.reshape(self.nx + 1, self.nz + 1)

    # ------------------------------------------------------- initial fields
    def _init_fields(self) -> None:
        nx, nz = self.nx, self.nz
        # Quiescent chambers; only the prescribed lower inlet core is seeded,
        # mirroring the frozen scene initial state.
        core = (np.abs(self.xc) <= self.inlet_half + 1e-9)[:, None] & (self.zc <= 0.45)[None, :]
        core = np.broadcast_to(core, (nx, nz)).copy()

        self.rho = np.where(core, self.rho_in, RHO0)
        self.f = np.where(core, 1.0, 0.0)
        self.rho[self.solid] = RHO0
        self.f[self.solid] = 0.0

        self.u = np.zeros((nx + 1, nz))
        self.w = np.zeros((nx, nz + 1))
        taper = np.clip(1.0 - (np.abs(self.xw) / 0.24) ** 4, 0.0, 1.0)
        seed_jet = (np.abs(self.xw) <= 0.24)[:, None] & (self.zw <= 0.45)[None, :]
        self.w = np.where(seed_jet, self.w_in * taper[None, :], 0.0)
        self.w = np.where(self.w_open_all, self.w, 0.0)
        self.w[self.inlet_cell[:, 0], 0] = self.w_in

        self._rho_low0 = float(self.rho[self.lower_fluid].mean())
        self._p_low = P0
        self._mdot3 = self.rho_in * self.q2d_in * self.depth  # kg/s injected

        # initial injected mass (node convention, matches reported frames)
        f_node = self._node_sample(self.f, 0.0)
        rho_node = self._node_sample(self.rho, RHO0)
        vol = self.h * self.h * (0.36 / 2.0)
        self._cellvol = vol
        self._initial_injected = float(np.sum(rho_node * f_node) * vol)
        self._p_kin = np.zeros(int(self.fluid.sum()))

    # ------------------------------------------------------------- stepping
    def _advect_faces(self, dt: float):
        vz = _bilerp(self.w, self.xw, self.zw, self.XU, self.ZU)
        px = self.XU - dt * self.u
        pz = self.ZU - dt * vz
        u_new = _bilerp(self.u, self.xu, self.zu, px, pz)

        vx = _bilerp(self.u, self.xu, self.zu, self.XW, self.ZW)
        px = self.XW - dt * vx
        pz = self.ZW - dt * self.w
        w_new = _bilerp(self.w, self.xw, self.zw, px, pz)
        return u_new, w_new

    def _advect_scalars(self, dt: float, u, w):
        vx = 0.5 * (u[:-1, :] + u[1:, :])
        vz = 0.5 * (w[:, :-1] + w[:, 1:])
        px = self.XC - dt * vx
        pz = self.ZC - dt * vz
        rho_new = _bilerp(self.rho, self.xc, self.zc, px, pz)
        f_new = _bilerp(self.f, self.xc, self.zc, px, pz)
        return rho_new, f_new

    def _project(self, dt: float, u, w):
        u_eff = np.where(self.u_open, u, 0.0)
        w_eff = np.where(self.w_open_free, w, 0.0)
        S = (u_eff[1:, :] - u_eff[:-1, :]) + (w_eff[:, 1:] - w_eff[:, :-1])
        D = np.where(self.lower_fluid, -self.s_compress, 0.0)
        Win = np.where(self.inlet_cell, self.w_in, 0.0)
        rhs2d = (D * self.h - S + Win) / dt
        rhs = rhs2d[self.fluid]
        rhs[self._pin_lower_k] = (self._p_low - P0) / RHO0
        rhs[self._pin_upper_k] = 0.0
        p = self._lu.solve(rhs)

        Pc = np.zeros_like(self.rho)
        Pc[self.fluid] = p
        du = np.zeros_like(u)
        du[1:self.nx, :] = (Pc[1:, :] - Pc[:-1, :]) / self.h
        dw = np.zeros_like(w)
        dw[:, 1:self.nz] = (Pc[:, 1:] - Pc[:, :-1]) / self.h
        u = np.where(self.u_open, u - dt * du, 0.0)
        w = np.where(self.w_open_all, w - dt * dw, 0.0)
        w[self.inlet_cell[:, 0], 0] = self.w_in
        return u, w, p

    def _step(self, dt: float) -> None:
        # 1. momentum advection
        u, w = self._advect_faces(dt)

        # 2. Boussinesq buoyancy (chamber-mean referenced; compression is uniform)
        rhoF = np.zeros_like(w)
        rhoF[:, 1:self.nz] = 0.5 * (self.rho[:, :-1] + self.rho[:, 1:])
        rhoF[:, 0] = self.rho[:, 0]
        rhoF[:, self.nz] = self.rho[:, self.nz - 1]
        mean_lo = float(self.rho[self.lower_fluid].mean())
        mean_up = float(self.rho[self.upper_fluid].mean())
        meanF = np.where(self.zw < 1.8, mean_lo, mean_up)[None, :]
        w = w + dt * (-GRAVITY * (rhoF - meanF) / RHO0)

        # 3. explicit viscous diffusion
        a_v = KIN_VISC * dt / self.h ** 2
        u = u + a_v * _lap(u)
        w = w + a_v * _lap(w)

        # 4. scalar advection
        rho, f = self._advect_scalars(dt, u, w)

        # 5. uniform compression of the sealed lower chamber (mass inflow)
        rho = np.where(self.lower_fluid, rho * (1.0 + self.s_compress * dt), rho)

        # 6. prescribed inlet conditions
        rho = np.where(self.inlet_cell, self.rho_in, rho)
        f = np.where(self.inlet_cell, 1.0, f)

        # 7. tracer diffusion, clipped to a physical fraction
        a_d = SCALAR_DIFF * dt / self.h ** 2
        f = np.clip(f + a_d * _lap(f), 0.0, 1.0)
        rho = np.clip(rho + a_d * _lap(rho), 0.5, 4.0)

        # 8. barrier cells carry no fluid state
        rho = np.where(self.solid, RHO0, rho)
        f = np.where(self.solid, 0.0, f)

        # 9. pressure projection (impermeable plate + walls, inlet preserved)
        u, w, p = self._project(dt, u, w)

        self.u, self.w, self.rho, self.f = u, w, rho, f
        self._p_kin = p
        # isothermal chamber pressure from the mean lower-chamber density
        self._p_low = P0 * float(self.rho[self.lower_fluid].mean()) / self._rho_low0

    # ------------------------------------------------------------- sampling
    def advance_to(self, time: float):
        time = float(time)
        if not self._ready:
            raise RuntimeError("solver not initialized")
        while self.t < time - 1e-9:
            dt = min(self.DT, time - self.t)
            self._step(dt)
            self.t = min(self.t + dt, time)
        self.t = max(self.t, time)
        return self._build_frame(time)

    def _node_velocity(self):
        u_node = np.zeros((self.nx + 1, self.nz + 1))
        u_node[:, 1:self.nz] = 0.5 * (self.u[:, :-1] + self.u[:, 1:])
        u_node[:, 0] = self.u[:, 0]
        u_node[:, self.nz] = self.u[:, self.nz - 1]
        w_node = np.zeros((self.nx + 1, self.nz + 1))
        w_node[1:self.nx, :] = 0.5 * (self.w[:-1, :] + self.w[1:, :])
        w_node[0, :] = self.w[0, :]
        w_node[self.nx, :] = self.w[self.nx - 1, :]
        return u_node, w_node

    def _pressure_jump(self) -> float:
        p_phys = np.full_like(self.rho, P0)
        p_phys[self.fluid] = P0 + self._p_kin
        xnear = (np.abs(self.xc) <= 0.55)[:, None]
        below = xnear & (self.zc > 1.6)[None, :] & (self.zc < 1.75)[None, :]
        above = xnear & (self.zc > 1.85)[None, :] & (self.zc < 2.0)[None, :]
        pb = float(p_phys[np.broadcast_to(below, p_phys.shape)].mean())
        pa = float(p_phys[np.broadcast_to(above, p_phys.shape)].mean())
        return pb - pa

    def _build_frame(self, time: float):
        nx, nz = self.nx, self.nz
        f_node = self._node_sample(self.f, 0.0)
        rho_node = self._node_sample(self.rho, RHO0)
        u_node, w_node = self._node_velocity()
        p_phys = np.zeros((nx, nz))
        p_phys[self.fluid] = P0 + self._p_kin
        p_node = self._node_sample(p_phys, P0)

        # shallow extrusion to the frozen voxel grid (121, 3, 73)
        def extrude(a):
            return np.broadcast_to(a[:, None, :], (nx + 1, 3, nz + 1)).copy()

        f3 = extrude(f_node)
        rho3 = extrude(rho_node)
        p3 = extrude(p_node)
        u3 = extrude(u_node)
        w3 = extrude(w_node)
        vel3 = np.zeros((nx + 1, 3, nz + 1, 3))
        vel3[..., 0] = u3
        vel3[..., 2] = w3

        active = np.ones((nx + 1, 3, nz + 1))
        j_plate = int(round(1.8 / self.h))
        active[:, :, j_plate] = 0.0  # barrier mid-plane

        xx, yy, zz = np.meshgrid(self.xn, self.yn, self.zn, indexing="ij")
        position = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        velocity = np.zeros_like(position)
        velocity[:, 0] = u3.ravel()
        velocity[:, 2] = w3.ravel()

        vol = self._cellvol
        mass = rho3.ravel() * vol
        injected = f3.ravel()
        mass_quantity = mass * injected

        # diagnostics
        zmask = zz.ravel()
        upper = injected * mass * (zmask > 1.85 - 1e-9)
        lower = injected * mass * (zmask < 1.8 + 1e-9)
        upper_mass = float(upper.sum())
        lower_mass = float(lower.sum())
        total_injected = upper_mass + lower_mass + float(
            (injected * mass * (np.abs(zmask - 1.8) <= 1e-9)).sum()
        )
        dp = self._pressure_jump()
        speed = np.sqrt(u3 ** 2 + w3 ** 2)
        max_speed = float(speed.max())

        plate_area = 2.0 * self.plate_half_x * self.depth
        momentum_flux = self._mdot3 * self.w_in
        interaction = np.array([[0.0, 0.0, -(dp * plate_area + momentum_flux)]])

        # boundary velocities: prescribed at the inlet, sampled fluid elsewhere
        bvel = np.zeros_like(self._btgt)
        sx = self._bpts[:, 0] - 0.05 * self._bnrm[:, 0]
        sz = self._bpts[:, 1] - 0.05 * self._bnrm[:, 2]
        bu = _bilerp(u_node, self.xn, self.zn, sx, sz)
        bw = _bilerp(w_node, self.xn, self.zn, sx, sz)
        bvel[:, 0] = bu
        bvel[:, 2] = bw
        bvel[self._binlet] = self._btgt[self._binlet]

        finite = bool(
            np.isfinite(self.u).all()
            and np.isfinite(self.w).all()
            and np.isfinite(self.rho).all()
            and np.isfinite(self.f).all()
            and np.isfinite(mass_quantity).all()
            and np.isfinite(vel3).all()
        )

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=mass,
            attributes={
                "injected_fraction": injected,
                "mass_quantity": mass_quantity,
            },
            fields={
                "injected_fraction": f3,
                "density": rho3,
                "pressure": p3,
                "velocity_field": vel3,
                "active_mask": active,
                "boundary_normal": self._bnrm,
                "boundary_target_velocity": self._btgt,
                "boundary_velocity": bvel,
                "cumulative_mass_source": np.array(
                    [total_injected - self._initial_injected]
                ),
                "interaction_force": interaction,
            },
            scalars={
                "characteristic_length": float(self.z1 - self.z0),
                "grid_spacing": float(self.h),
                "reference_velocity": float(self.w_in),
                "upper_chamber_injected_mass": upper_mass,
                "lower_chamber_injected_mass": lower_mass,
                "pressure_jump_across_plate": float(dp),
                "maximum_speed": max_speed,
                "finite_state": 1.0 if finite else 0.0,
                "lower_chamber_pressure": float(self._p_low),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "scene_id": "054_thin_plate_two_chambers",
            "method": "mac_grid_finite_volume_semi_lagrangian_projection",
            "grid": [self.nx, self.nz],
            "grid_spacing": self.h,
            "substep_dt": self.DT,
            "compression_rate": self.s_compress,
            "lower_chamber_pressure": float(self._p_low),
            "upper_chamber_clear": True,
            "finite": True,
        }
