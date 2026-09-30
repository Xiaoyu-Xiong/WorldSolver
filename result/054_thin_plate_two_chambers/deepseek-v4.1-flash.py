"""Physical solver for the thin-plate two-chamber jet scene.

Method
------
Incompressible Navier-Stokes in a 2D (x,z) plane, discretised on a staggered
MAC grid that is broadcast over the shallow y extrusion (ny = 3 planes).  The
thin plate, the chamber walls and the domain boundary are represented as
blocked (solid) cells; the lower-chamber side walls are pressure outlets.

Per time step (explicit, CFL-limited):

1. upwind finite-volume advection of the face velocities,
2. explicit viscous diffusion,
3. enforcement of the immersed-boundary face velocities (no-slip plate/walls,
   prescribed inlet jet),
4. pressure projection: a static 5-point Poisson system with Neumann
   conditions on solid faces and Dirichlet p = p_ambient on the open lower
   chamber outlets, solved with a cached sparse LU factorisation,
5. upwind advection of the injected-tracer fraction with a prescribed inlet
   source.

The tracer is a passive scalar; the density field is
rho = rho_ambient + (rho_jet - rho_ambient) * c.  The impermeable plate seals
the lower chamber from the upper chamber, so no tracer can reach the upper
chamber (target leaked mass = 0).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def __init__(self) -> None:
        self._ready = False

    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        dom = dict(cfg.get("domain", {}) or {})
        bmin = np.asarray(dom.get("bounds_min", [-3.0, -0.18, 0.0]), dtype=float)
        bmax = np.asarray(dom.get("bounds_max", [3.0, 0.18, 3.6]), dtype=float)
        inlet = dict((cfg.get("boundary_conditions", {}) or {}).get("jet_inlet", {}) or {})

        # frozen grid (matches scene_geometry voxel_grid)
        self.nx, self.ny, self.nz = 121, 3, 73
        self.x0, self.z0 = float(bmin[0]), float(bmin[2])
        self.dx = (float(bmax[0]) - float(bmin[0])) / (self.nx - 1)
        self.dz = (float(bmax[2]) - float(bmin[2])) / (self.nz - 1)
        self.dy = (float(bmax[1]) - float(bmin[1])) / (self.ny - 1)
        self.h = self.dx
        self.cell_vol = self.dx * self.dy * self.dz

        self.rho_amb = float((cfg.get("initial_conditions", {}) or {}).get("ambient_density", 1.0))
        self.p_amb = float((cfg.get("initial_conditions", {}) or {}).get("ambient_pressure", 1.0))
        self.rho_jet = float(inlet.get("density", 1.18))
        self.v_jet = float(np.asarray(inlet.get("velocity", [0.0, 0.0, 2.4]), dtype=float)[2])
        self.inlet_half = 0.5 * float(inlet.get("width", 0.42))
        self.nu = 5.0e-3
        self.cfl = 0.35

        self.xc = self.x0 + self.dx * np.arange(self.nx)
        self.zc = self.z0 + self.dz * np.arange(self.nz)

        self._base = scene.initial_state()
        self._position = np.asarray(self._base.position, dtype=float)
        self._build_geometry()
        self._build_poisson()
        self._build_boundary_samples()

        # conserved-material bookkeeping (tracer / ambient)
        self._cum_source = np.zeros(2, dtype=float)

        # initial fields taken from the frozen scene initial state
        c0 = np.asarray(self._base.fields["injected_fraction"], dtype=float)
        v0 = np.asarray(self._base.fields["velocity"], dtype=float)
        c2 = c0[:, self.ny // 2, :].copy()
        u2 = v0[:, self.ny // 2, :, 0].copy()
        w2 = v0[:, self.ny // 2, :, 2].copy()

        self.c = np.clip(np.nan_to_num(c2), 0.0, 1.0)
        self.u = np.zeros((self.nx + 1, self.nz))
        self.w = np.zeros((self.nx, self.nz + 1))
        self.u[1:self.nx, :] = 0.5 * (u2[:-1, :] + u2[1:, :])
        self.w[:, 1:self.nz] = 0.5 * (w2[:, :-1] + w2[:, 1:])
        self.p = np.full((self.nx, self.nz), self.p_amb)
        self._apply_face_bc()
        self._t = 0.0
        self._stats = {
            "upper_chamber_injected_mass": 0.0,
            "lower_chamber_injected_mass": 0.0,
            "pressure_jump_across_plate": 0.0,
            "maximum_speed": self.v_jet,
            "finite_state": 1.0,
        }
        self._ready = True

    # -------------------------------------------------------------- geometry
    def _build_geometry(self) -> None:
        nx, nz = self.nx, self.nz
        solid = np.zeros((nx, nz), dtype=bool)
        solid[:, 0] = True                     # bottom support
        solid[:, nz - 1] = True                # top domain boundary (impermeable)
        solid[0, :] = True                     # left wall
        solid[nx - 1, :] = True                # right wall
        solid[6:115, 36] = True                # thin separating plate
        solid[:, 36] = True                    # ...sealed to both side walls

        self.inlet_cells = np.zeros((nx, nz), dtype=bool)   # i-slice for inlet column
        self.inlet_i = np.arange(nx)[np.abs(self.xc) <= self.inlet_half + 1e-9]
        self.inlet_cells[self.inlet_i, 0] = True
        solid[self.inlet_i, 0] = False         # inlet aperture in the bottom support

        # open (pressure-outlet) side walls of the lower chamber only
        self.open_left = np.zeros(nz, dtype=bool)
        self.open_right = np.zeros(nz, dtype=bool)
        low = self.zc < 1.75
        solid[0, low] = False
        solid[nx - 1, low] = False
        self.open_left[low] = True
        self.open_right[low] = True

        self.solid = solid
        self.fluid = ~solid
        self.n_fluid = int(self.fluid.sum())

        # plate cell index range (for interaction force)
        self.plate_i0, self.plate_i1 = 6, 114

        # face classification: 0 interior, 1 closed, 2 inlet, 3 open
        u_t = np.zeros((nx + 1, nz), dtype=np.int8)
        w_t = np.zeros((nx, nz + 1), dtype=np.int8)
        for j in range(1, nx):
            both = self.fluid[j - 1, :] & self.fluid[j, :]
            u_t[j, :] = np.where(both, 0, 1)
        u_t[0, :] = np.where(self.open_left, 3, 1)
        u_t[nx, :] = np.where(self.open_right, 3, 1)
        for k in range(1, nz):
            both = self.fluid[:, k - 1] & self.fluid[:, k]
            w_t[:, k] = np.where(both, 0, 1)
        w_t[:, 0] = 1
        w_t[:, nz] = 1
        w_t[self.inlet_i, 0] = 2
        self.u_t = u_t
        self.w_t = w_t
        self.u_interior = u_t == 0
        self.w_interior = w_t == 0

    def _build_poisson(self) -> None:
        nx, nz = self.nx, self.nz
        idx = -np.ones((nx, nz), dtype=np.int64)
        ii, kk = np.nonzero(self.fluid)
        idx[ii, kk] = np.arange(self.n_fluid)
        self.idx = idx
        n_open = np.zeros(self.n_fluid)
        rows, cols, vals = [], [], []
        diag = np.zeros(self.n_fluid)

        def add(nb_i, nb_k, inside, open_mask):
            m = inside & self.fluid[np.clip(nb_i, 0, nx - 1), np.clip(nb_k, 0, nz - 1)]
            rows.extend(idx[ii[m], kk[m]].tolist())
            cols.extend(idx[np.clip(nb_i[m], 0, nx - 1), np.clip(nb_k[m], 0, nz - 1)].tolist())
            vals.extend([1.0] * int(m.sum()))
            diag[m] += 1.0
            n_open[m] += open_mask[m].astype(float)

        away_l = ii > 0
        add(ii - 1, kk, away_l, np.zeros(self.n_fluid))
        add(ii + 1, kk, ii < nx - 1, np.zeros(self.n_fluid))
        add(ii, kk - 1, kk > 0, np.zeros(self.n_fluid))
        add(ii, kk + 1, kk < nz - 1, np.zeros(self.n_fluid))
        # open boundaries: only at the lower-chamber side walls (clipped index == ii)
        left_open = (ii == 0) & self.open_left[kk]
        right_open = (ii == nx - 1) & self.open_right[kk]
        n_open += left_open + right_open

        diag_base = len(vals)
        rows.extend(np.arange(self.n_fluid).tolist())
        cols.extend(np.arange(self.n_fluid).tolist())
        vals.extend((-diag - n_open).tolist())

        # A sealed fluid component (the upper chamber) has no Dirichlet face, so
        # its pressure block is singular.  Pin one cell per sealed component to
        # the ambient pressure; a compatible (divergence-free, quiescent) cavity
        # then stays uniformly at p_amb.
        from scipy.ndimage import label as _label

        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
        comp, ncomp = _label(self.fluid, structure=structure)
        pinned = []
        for cid in range(1, ncomp + 1):
            cells = comp == cid
            if not np.any(n_open[cells] > 0.0):
                pinned.append(int(idx[cells][0]))
        for pin in pinned:
            vals[diag_base + pin] += -1.0
        self.pin_idx = np.asarray(pinned, dtype=np.int64)
        self.n_open = n_open
        a = csc_matrix((np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
                       shape=(self.n_fluid, self.n_fluid))
        self._lu = splu(a)

    def _build_boundary_samples(self) -> None:
        nx, nz = self.nx, self.nz
        faces = []  # (is_u, i, k, nx, ny, nz_component-of-normal, is_inlet)
        for j in range(1, nx):
            for k in range(nz):
                if self.u_t[j, k] != 1:
                    continue
                if self.fluid[j, k]:
                    faces.append((1, j, k, 1.0, 0.0, 0.0, 0))
                elif self.fluid[j - 1, k]:
                    faces.append((1, j, k, -1.0, 0.0, 0.0, 0))
        for i in range(1, nx - 1):
            for k in range(1, nz):
                if self.w_t[i, k] != 1:
                    continue
                if self.fluid[i, k]:
                    faces.append((0, i, k, 0.0, 0.0, -1.0, 0))
                elif self.fluid[i, k - 1]:
                    faces.append((0, i, k, 0.0, 0.0, 1.0, 0))
        for i in self.inlet_i:
            faces.append((0, int(i), 0, 0.0, 0.0, -1.0, 1))
        arr = np.asarray(faces, dtype=float)
        self.b_is_u = arr[:, 0].astype(bool)
        self.b_i = arr[:, 1].astype(int)
        self.b_k = arr[:, 2].astype(int)
        self.b_n = arr[:, 3:6]
        self.b_inlet = arr[:, 6].astype(bool)

    # ------------------------------------------------------------------- step
    def _dt_max(self) -> float:
        umax = max(self.v_jet, float(np.sqrt(self.u ** 2).max() if self.u.size else 0.0),
                   float(np.sqrt(self.w ** 2).max() if self.w.size else 0.0))
        return min(self.cfl * self.h / max(umax, 1e-6), 0.2 * self.h * self.h / self.nu, 0.01)

    def _apply_face_bc(self) -> None:
        closed_u = self.u_t == 1
        self.u[closed_u] = 0.0
        self.u[0, self.open_left] = self.u[1, self.open_left]
        self.u[self.nx, self.open_right] = self.u[self.nx - 1, self.open_right]
        closed_w = self.w_t == 1
        self.w[closed_w] = 0.0
        self.w[self.inlet_i, 0] = self.v_jet

    def _advect_faces(self, dt: float) -> None:
        nx, nz, h = self.nx, self.nz, self.h
        u, w = self.u, self.w

        # --- u momentum (x faces) ---
        j = np.arange(1, nx)
        k = np.arange(nz)
        JJ, KK = np.meshgrid(j, k, indexing="ij")
        uh = u[JJ, KK]
        wc = 0.25 * (w[JJ - 1, KK] + w[JJ - 1, np.clip(KK + 1, 0, nz)] +
                     w[JJ, KK] + w[JJ, np.clip(KK + 1, 0, nz)])
        dux = np.where(uh > 0.0, u[JJ, KK] - u[JJ - 1, KK], u[JJ + 1, KK] - u[JJ, KK]) / h
        dzm = np.where(wc > 0.0, u[JJ, KK] - u[JJ, np.clip(KK - 1, 0, nz - 1)],
                       u[JJ, np.clip(KK + 1, 0, nz - 1)] - u[JJ, KK]) / h
        self.u[JJ, KK] = uh - dt * (uh * dux + wc * dzm)

        # --- w momentum (z faces) ---
        i = np.arange(nx)
        kk = np.arange(1, nz)
        II, K2 = np.meshgrid(i, kk, indexing="ij")
        wh = w[II, K2]
        uc = 0.25 * (u[II, K2 - 1] + u[II, K2] + u[np.clip(II + 1, 0, nx), K2 - 1] +
                     u[np.clip(II + 1, 0, nx), K2])
        dwz = np.where(wh > 0.0, w[II, K2] - w[II, K2 - 1], w[II, K2 + 1] - w[II, K2]) / h
        dxm = np.where(uc > 0.0, w[II, K2] - w[np.clip(II - 1, 0, nx - 1), K2],
                       w[np.clip(II + 1, 0, nx - 1), K2] - w[II, K2]) / h
        self.w[II, K2] = wh - dt * (wh * dwz + uc * dxm)

        # --- viscous diffusion (explicit central Laplacian) ---
        for name in ("u", "w"):
            f = getattr(self, name)
            fp = np.pad(f, 1, mode="edge")
            lap = ((fp[2:, 1:-1] - 2.0 * fp[1:-1, 1:-1] + fp[:-2, 1:-1]) / h ** 2 +
                   (fp[1:-1, 2:] - 2.0 * fp[1:-1, 1:-1] + fp[1:-1, :-2]) / h ** 2)
            setattr(self, name, f + self.nu * dt * lap)

        self._apply_face_bc()
        for name in ("u", "w", "c"):
            f = getattr(self, name)
            np.nan_to_num(f, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _project(self, dt: float) -> None:
        nx, nz, h = self.nx, self.nz, self.h
        div = np.zeros((nx, nz))
        div += (self.u[1:, :] - self.u[:-1, :]) / h
        div += (self.w[:, 1:] - self.w[:, :-1]) / h
        rhs = h * h * div[self.fluid] / dt - self.n_open * self.p_amb
        if self.pin_idx.size:
            rhs[self.pin_idx] += self.p_amb
        pflat = self._lu.solve(rhs)
        pfull = np.zeros((nx, nz))
        pfull[self.fluid] = pflat
        self.p = pfull

        P = np.zeros((nx + 2, nz))
        P[1:nx + 1, :] = pfull
        W = np.zeros((nx, nz + 2))
        W[:, 1:nz + 1] = pfull
        gu = (P[1:nx + 2, :] - P[0:nx + 1, :]) / h
        gw = (W[:, 1:nz + 2] - W[:, :nz + 1]) / h
        self.u[self.u_interior] -= dt * gu[self.u_interior]
        self.w[self.w_interior] -= dt * gw[self.w_interior]
        self._apply_face_bc()

    def _advect_tracer(self, dt: float) -> None:
        nx, nz, h = self.nx, self.nz, self.h
        uc = 0.5 * (self.u[:nx, :] + self.u[1:nx + 1, :])
        wc = 0.5 * (self.w[:, :nz] + self.w[:, 1:nz + 1])
        c = self.c
        cm = c[np.clip(np.arange(nx) - 1, 0, nx - 1), :]
        cp = c[np.clip(np.arange(nx) + 1, 0, nx - 1), :]
        ck = c[:, np.clip(np.arange(nz) - 1, 0, nz - 1)]
        cq = c[:, np.clip(np.arange(nz) + 1, 0, nz - 1)]
        dcx = np.where(uc > 0.0, c - cm, cp - c) / h
        dcz = np.where(wc > 0.0, c - ck, cq - c) / h
        self.c = np.clip(c - dt * (uc * dcx + wc * dcz), 0.0, 1.0)
        self.c[self.inlet_cells] = 1.0
        self.c[self.solid] = 0.0

    def _mass_fluxes(self, dt: float) -> None:
        # tracer mass entering at the inlet
        area = self.dx * self.dy
        m_in = self.rho_jet * self.v_jet * area * len(self.inlet_i) * dt
        # tracer mass leaving through the open lower-chamber outlets
        m_out = 0.0
        for j, sign, flag in ((0, -1.0, self.open_left), (self.nx, 1.0, self.open_right)):
            for k in np.nonzero(flag)[0]:
                uo = sign * self.u[j, k]
                if uo > 0.0:
                    i = 0 if j == 0 else self.nx - 1
                    rho = self.rho_amb + (self.rho_jet - self.rho_amb) * self.c[i, k]
                    m_out += rho * uo * area * dt
        self._cum_source[0] += m_in - m_out

    def _step(self, dt: float) -> None:
        self._advect_faces(dt)
        self._project(dt)
        self._advect_tracer(dt)
        self._mass_fluxes(dt)

    # -------------------------------------------------------------- sampling
    def _frame(self, t: float) -> StateFrame:
        nx, ny, nz = self.nx, self.ny, self.nz
        c = self.c
        rho = self.rho_amb + (self.rho_jet - self.rho_amb) * c
        uc = 0.5 * (self.u[:nx, :] + self.u[1:nx + 1, :])
        wc = 0.5 * (self.w[:, :nz] + self.w[:, 1:nz + 1])

        vel_grid = np.zeros((nx, ny, nz, 3))
        vel_grid[:, :, :, 0] = uc[:, None, :]
        vel_grid[:, :, :, 2] = wc[:, None, :]
        fluid3 = self.fluid[:, None, :]
        # ensure solid cells carry no motion and ambient density
        vel_grid = np.where(fluid3[..., None], vel_grid, 0.0)

        density = np.repeat(rho[:, None, :], ny, axis=1)
        density = np.where(fluid3, density, self.rho_amb)
        dens_flat = density.ravel()

        speed = np.sqrt(vel_grid[..., 0] ** 2 + vel_grid[..., 1] ** 2 + vel_grid[..., 2] ** 2)
        max_speed = float(speed.max()) if speed.size else 0.0

        rho_cell = rho * self.cell_vol
        low_mask = self.zc < 1.8
        up_mask = self.zc >= 1.85
        lower_mass = float((self.c[:, low_mask] * rho_cell[:, low_mask]).sum())
        upper_mass = float((self.c[:, up_mask] * rho_cell[:, up_mask]).sum())

        i_pl = slice(self.plate_i0, self.plate_i1 + 1)
        p_below = float(self.p[i_pl, 35].mean())
        p_above = float(self.p[i_pl, 37].mean())
        pjump = p_below - p_above

        # interaction force of the fluid on the sealed plate (K = 1)
        area = self.dx * self.dy
        f = np.zeros((1, 3))
        npl = len(self.xc[i_pl])
        f[0, 2] += (p_below - p_above) * area * npl
        # viscous shear on the wetted plate faces
        shear = self.nu * ((uc[i_pl, 37].mean() / (0.5 * self.h)) -
                           (uc[i_pl, 35].mean() / (0.5 * self.h)))
        f[0, 0] += shear * area * npl

        # boundary samples
        bvel = np.zeros((len(self.b_i), 3))
        mu = self.b_is_u
        if mu.any():
            bvel[mu, 0] = self.u[self.b_i[mu], self.b_k[mu]]
        mw = ~mu
        if mw.any():
            bvel[mw, 2] = self.w[self.b_i[mw], self.b_k[mw]]
        btarget = np.zeros_like(bvel)
        btarget[self.b_inlet, 2] = self.v_jet
        btarget[self.b_inlet, 0] = 0.0

        active = self.fluid.astype(float)[:, None, :].repeat(ny, axis=1)
        frac3 = c[:, None, :].repeat(ny, axis=1)

        fields = {
            "velocity_field": vel_grid,
            "velocity": vel_grid,
            "density": density,
            "density_field": density,
            "pressure": np.repeat(self.p[:, None, :], ny, axis=1),
            "injected_fraction": frac3,
            "active_mask": active,
            "boundary_velocity": bvel,
            "boundary_target_velocity": btarget,
            "boundary_normal": self.b_n.copy(),
            "cumulative_mass_source": self._cum_source.copy(),
            "interaction_force": f.copy(),
        }
        scalars = {
            "characteristic_length": 1.8,
            "grid_spacing": float(self.h),
            "reference_velocity": float(self.v_jet),
            "upper_chamber_injected_mass": upper_mass,
            "lower_chamber_injected_mass": lower_mass,
            "pressure_jump_across_plate": pjump,
            "maximum_speed": max_speed,
            "finite_state": 1.0,
        }
        self._stats = dict(scalars)

        velocity = vel_grid.reshape((-1, 3))
        finite = bool(np.isfinite(velocity).all() and np.isfinite(dens_flat).all())
        scalars["finite_state"] = 1.0 if finite else 0.0
        self._stats["finite_state"] = scalars["finite_state"]

        frame = StateFrame(
            time=float(t),
            position=self._position.copy(),
            velocity=velocity,
            mass=dens_flat,
            attributes={
                "injected_fraction": frac3.ravel(),
                "mass_quantity": dens_flat,
            },
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("BenchmarkSolver.initialize() must be called first")
        t = float(time)
        if t < self._t - 1e-12:
            raise ValueError("advance_to() requires monotonically increasing times")
        guard = 0
        while self._t < t - 1e-12 and guard < 200000:
            dt = min(self._dt_max(), t - self._t)
            if dt <= 1e-12:
                break
            self._step(dt)
            self._t += dt
            guard += 1
        self._t = t
        return self._frame(t)

    def diagnostics(self) -> dict[str, object]:
        if not self._ready:
            return {}
        out = {k: float(v) for k, v in self._stats.items()}
        out["cumulative_injected_mass"] = float(self._cum_source[0])
        out["solid_cell_count"] = float(self.n_fluid)
        return out
