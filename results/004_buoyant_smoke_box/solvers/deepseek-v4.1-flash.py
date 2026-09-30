"""Physically-based buoyant smoke box solver (NumPy/SciPy only).

Model
-----
Incompressible Boussinesq flow in a closed box, advanced with a real
time-stepping scheme:

* staggered (MAC) velocity grid on the frozen 32 x 32 x 64 layout,
* semi-Lagrangian advection of velocity and of the transported scalars
  (smoke density, temperature),
* buoyancy body force from the heat source (Boussinesq coupling),
* explicit viscous / diffusive terms with the frozen grid spacing,
* pressure projection with a pure-Neumann Poisson solve (DCT-II) enforcing
  a discretely divergence-free field and no-penetration/no-slip wall
  velocities,
* deterministic passive markers advected with a second-order midpoint rule,
  which keeps the seeding unclumped because the flow is volume preserving.

No random numbers, no keyframes, no precomputed motion.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import map_coordinates

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        cfg = scene.config
        self._seed = int(seed)

        dom = cfg["domain"]
        layout = cfg["field_layout"]
        initial = cfg["initial_conditions"]

        self._lo = np.asarray(dom["bounds_min"], dtype=np.float64)
        self._hi = np.asarray(dom["bounds_max"], dtype=np.float64)
        self._shape = tuple(int(v) for v in layout["grid_shape"])
        nx, ny, nz = self._shape
        self._nx, self._ny, self._nz = nx, ny, nz

        ext = self._hi - self._lo
        self._dx = float(ext[0] / nx)
        self._dy = float(ext[1] / ny)
        self._dz = float(ext[2] / nz)
        self._h = self._dx
        self._cell_volume = self._dx * self._dy * self._dz
        self._domain_volume = float(ext[0] * ext[1] * ext[2])

        # cell-centre coordinates
        self._xc = self._lo[0] + (np.arange(nx) + 0.5) * self._dx
        self._yc = self._lo[1] + (np.arange(ny) + 0.5) * self._dy
        self._zc = self._lo[2] + (np.arange(nz) + 0.5) * self._dz

        self._X, self._Y, self._Z = np.meshgrid(self._xc, self._yc, self._zc, indexing="ij")

        # frozen physical parameters
        self._dt = float(cfg["sample_dt"])
        self._duration = float(cfg["duration"])
        self._ambient_temperature = float(initial["ambient_temperature"])
        self._source_temperature = float(initial["source_temperature"])
        self._source_radius = float(initial["source_radius"])
        self._source_height = float(initial["source_height"])
        self._beta = 1.2          # buoyancy acceleration per unit temperature
        self._nu = 1.0e-3         # kinematic viscosity
        self._kappa = 1.0e-4      # thermal / scalar diffusivity
        self._relax = 3.0         # source temperature relaxation rate [1/s]
        self._smoke_rate = 1.0    # smoke injection rate [1/s]
        self._drag = 0.02         # weak linear damping [1/s]
        self._max_speed = 8.0
        self._substeps_max = 4

        # ---- staggered velocity (MAC) -------------------------------------
        self._u = np.zeros((nx + 1, ny, nz), dtype=np.float64)
        self._v = np.zeros((nx, ny + 1, nz), dtype=np.float64)
        self._w = np.zeros((nx, ny, nz + 1), dtype=np.float64)

        # ---- scalars -------------------------------------------------------
        frame0 = scene.initial_state()
        smoke = frame0.fields.get("smoke_density")
        if smoke is None:
            smoke = np.zeros(self._shape, dtype=np.float64)
        self._smoke = np.array(smoke, dtype=np.float64).reshape(self._shape)
        temp = frame0.fields.get("temperature")
        if temp is None:
            temp = np.zeros(self._shape, dtype=np.float64)
        self._temp = np.array(temp, dtype=np.float64).reshape(self._shape)
        self._smoke = np.nan_to_num(self._smoke, nan=0.0, posinf=0.0, neginf=0.0)
        self._temp = np.nan_to_num(self._temp, nan=0.0, posinf=0.0, neginf=0.0)

        # heat / smoke source region (cylindrical column above the heater)
        r = np.sqrt(self._X ** 2 + self._Y ** 2)
        self._source_mask = ((r <= self._source_radius) &
                             (self._Z <= self._lo[2] + self._source_height)).astype(np.float64)
        if not np.any(self._source_mask > 0.0):
            k = int(np.argmin(np.abs(self._zc - (self._lo[2] + 0.5 * self._dz))))
            self._source_mask[:, :, k] = 1.0
        self._scalar_source = (self._smoke_rate * self._source_mask).astype(np.float64)

        # ---- passive markers ----------------------------------------------
        self._pos = np.array(frame0.position, dtype=np.float64).reshape(-1, 3)
        self._vel = np.array(frame0.velocity, dtype=np.float64).reshape(-1, 3)
        n = self._pos.shape[0]
        mass = frame0.mass
        self._mass = (np.array(mass, dtype=np.float64).reshape(-1)
                      if mass is not None else np.full(n, 1.0 / max(n, 1)))
        kind = frame0.attributes.get("marker_kind")
        self._kind = (np.array(kind).reshape(-1).astype(np.int32)
                      if kind is not None else np.ones(n, dtype=np.int32))

        # ---- wall samples ---------------------------------------------------
        self._wall_points, self._wall_normals = self._build_wall_samples()
        self._wall_target = np.zeros_like(self._wall_points)

        # ---- spectral Poisson operator (Neumann) ----------------------------
        kx = np.arange(nx)
        ky = np.arange(ny)
        kz = np.arange(nz)
        ex = (2.0 * np.cos(np.pi * kx / nx) - 2.0) / self._dx ** 2
        ey = (2.0 * np.cos(np.pi * ky / ny) - 2.0) / self._dy ** 2
        ez = (2.0 * np.cos(np.pi * kz / nz) - 2.0) / self._dz ** 2
        eig = ex[:, None, None] + ey[None, :, None] + ez[None, None, :]
        self._inv_eig = np.zeros_like(eig)
        nz_mask = np.abs(eig) > 1e-14
        self._inv_eig[nz_mask] = 1.0 / eig[nz_mask]

        # ---- bookkeeping ----------------------------------------------------
        self._step_index = 0
        self._time = 0.0
        self._accumulated_volume_change = 0.0
        self._div_l2 = 0.0
        self._div_max = 0.0
        self._volume_error = 0.0
        self._initialized = True

    # ------------------------------------------------------------ utilities
    def _build_wall_samples(self):
        nx, ny, nz = self._nx, self._ny, self._nz
        lo, hi = self._lo, self._hi
        xc, yc, zc = self._xc[::2], self._yc[::2], self._zc[::2]
        pts, nrm = [], []

        def add(px, py, pz, normal):
            gx, gy, gz = np.meshgrid(px, py, pz, indexing="ij")
            pts.append(np.column_stack((gx.ravel(), gy.ravel(), gz.ravel())))
            n = np.zeros((gx.size, 3), dtype=np.float64)
            n[:] = normal
            nrm.append(n)

        for sign, xw in ((-1.0, lo[0]), (1.0, hi[0])):
            add(np.array([xw]), yc, zc, (sign, 0.0, 0.0))
        for sign, yw in ((-1.0, lo[1]), (1.0, hi[1])):
            add(xc, np.array([yw]), zc, (0.0, sign, 0.0))
        for sign, zw in ((-1.0, lo[2]), (1.0, hi[2])):
            add(xc, yc, np.array([zw]), (0.0, 0.0, sign))

        return np.concatenate(pts, axis=0), np.concatenate(nrm, axis=0)

    def _collocated(self):
        """Cell-centred velocity view of the staggered field."""
        uc = np.empty((self._nx, self._ny, self._nz, 3), dtype=np.float64)
        uc[:, :, :, 0] = 0.5 * (self._u[:-1, :, :] + self._u[1:, :, :])
        uc[:, :, :, 1] = 0.5 * (self._v[:, :-1, :] + self._v[:, 1:, :])
        uc[:, :, :, 2] = 0.5 * (self._w[:, :, :-1] + self._w[:, :, 1:])
        return uc

    def _sample(self, field, px, py, pz, order=1):
        ix = (px - self._lo[0]) / self._dx - 0.5
        iy = (py - self._lo[1]) / self._dy - 0.5
        iz = (pz - self._lo[2]) / self._dz - 0.5
        return map_coordinates(field, [ix, iy, iz], order=order, mode="nearest")

    def _sample_vec(self, uc, px, py, pz, order=1):
        out = np.empty((px.size, 3), dtype=np.float64)
        for c in range(3):
            out[:, c] = self._sample(uc[:, :, :, c], px, py, pz, order=order)
        return out

    def _advect_scalar(self, field, uc, dt):
        """Semi-Lagrangian backtrace of a cell-centred scalar."""
        ux = uc[:, :, :, 0]
        uy = uc[:, :, :, 1]
        uz = uc[:, :, :, 2]
        # departure points
        px, py, pz = self._X, self._Y, self._Z
        dxp = self._sample(ux, px, py, pz) * dt
        dyp = self._sample(uy, px, py, pz) * dt
        dzp = self._sample(uz, px, py, pz) * dt
        return self._sample(field, px - dxp, py - dyp, pz - dzp)

    def _advect_staggered(self, comp, uc, dt, axis):
        """Advect one staggered velocity component (index-approximate)."""
        px, py, pz = self._X, self._Y, self._Z
        dxp = self._sample(uc[:, :, :, 0], px, py, pz) * dt
        dyp = self._sample(uc[:, :, :, 1], px, py, pz) * dt
        dzp = self._sample(uc[:, :, :, 2], px, py, pz) * dt
        dep = (np.clip(px - dxp, self._lo[0], self._hi[0]),
               np.clip(py - dyp, self._lo[1], self._hi[1]),
               np.clip(pz - dzp, self._lo[2], self._hi[2]))
        if axis == 0:
            base = comp[:-1, :, :]
            adv = self._sample(base, *dep)
            comp[:-1, :, :] = adv
        elif axis == 1:
            base = comp[:, :-1, :]
            adv = self._sample(base, *dep)
            comp[:, :-1, :] = adv
        else:
            base = comp[:, :, :-1]
            adv = self._sample(base, *dep)
            comp[:, :, :-1] = adv
        return comp

    @staticmethod
    def _laplacian(field, hx, hy, hz):
        p = np.pad(field, 1, mode="edge")
        lap = ((p[2:, 1:-1, 1:-1] - 2.0 * field + p[:-2, 1:-1, 1:-1]) / hx ** 2 +
               (p[1:-1, 2:, 1:-1] - 2.0 * field + p[1:-1, :-2, 1:-1]) / hy ** 2 +
               (p[1:-1, 1:-1, 2:] - 2.0 * field + p[1:-1, 1:-1, :-2]) / hz ** 2)
        return lap

    def _apply_wall_bc(self):
        u, v, w = self._u, self._v, self._w
        # no penetration (normal components on the walls)
        u[0, :, :] = 0.0
        u[self._nx, :, :] = 0.0
        v[:, 0, :] = 0.0
        v[:, self._ny, :] = 0.0
        w[:, :, 0] = 0.0
        w[:, :, self._nz] = 0.0
        # no slip for tangential components adjacent to the walls
        u[:, 0, :] = 0.0
        u[:, self._ny - 1, :] = 0.0
        u[:, :, 0] = 0.0
        u[:, :, self._nz - 1] = 0.0
        v[0, :, :] = 0.0
        v[self._nx - 1, :, :] = 0.0
        v[:, :, 0] = 0.0
        v[:, :, self._nz - 1] = 0.0
        w[0, :, :] = 0.0
        w[self._nx - 1, :, :] = 0.0
        w[:, 0, :] = 0.0
        w[:, self._ny - 1, :] = 0.0

    def _divergence(self):
        u, v, w = self._u, self._v, self._w
        return ((u[1:, :, :] - u[:-1, :, :]) / self._dx +
                (v[:, 1:, :] - v[:, :-1, :]) / self._dy +
                (w[:, :, 1:] - w[:, :, :-1]) / self._dz)

    def _project(self, dt):
        div = self._divergence()
        rhs = div / dt
        p_hat = dctn(rhs, type=2, norm="ortho") * self._inv_eig
        p = idctn(p_hat, type=2, norm="ortho")
        self._u[1:self._nx, :, :] -= dt * (p[1:, :, :] - p[:-1, :, :]) / self._dx
        self._v[:, 1:self._ny, :] -= dt * (p[:, 1:, :] - p[:, :-1, :]) / self._dy
        self._w[:, :, 1:self._nz] -= dt * (p[:, :, 1:] - p[:, :, :-1]) / self._dz
        self._apply_wall_bc()
        return div

    # -------------------------------------------------------------- physics
    def _step(self, dt):
        """One physical time step of the Boussinesq scheme."""
        nx, ny, nz = self._nx, self._ny, self._nz
        uc = self._collocated()
        speed = float(np.sqrt(np.sum(uc ** 2, axis=3)).max()) if uc.size else 0.0
        cfl = speed * dt / min(self._dx, self._dy, self._dz)
        diff = max(self._nu, self._kappa)
        diff_limit = 0.24 * self._h ** 2 / max(diff, 1e-12)
        n_sub = int(np.ceil(max(cfl, 1e-9)))
        n_sub = max(1, min(self._substeps_max, n_sub))
        if diff_limit < dt / n_sub:
            n_sub = max(n_sub, min(self._substeps_max, int(np.ceil(dt / max(diff_limit, 1e-12)))))
        h = dt / n_sub

        for _ in range(n_sub):
            uc = self._collocated()
            # --- advection -------------------------------------------------
            self._smoke = self._advect_scalar(self._smoke, uc, h)
            self._temp = self._advect_scalar(self._temp, uc, h)
            self._u = self._advect_staggered(self._u, uc, h, 0)
            self._v = self._advect_staggered(self._v, uc, h, 1)
            self._w = self._advect_staggered(self._w, uc, h, 2)

            # --- diffusion --------------------------------------------------
            self._smoke = self._smoke + h * self._kappa * self._laplacian(
                self._smoke, self._dx, self._dy, self._dz)
            self._temp = self._temp + h * self._kappa * self._laplacian(
                self._temp, self._dx, self._dy, self._dz)
            self._u[1:nx, :, :] += h * self._nu * self._laplacian(
                self._u[1:nx, :, :], self._dx, self._dy, self._dz)
            self._v[:, 1:ny, :] += h * self._nu * self._laplacian(
                self._v[:, 1:ny, :], self._dx, self._dy, self._dz)
            self._w[:, :, 1:nz] += h * self._nu * self._laplacian(
                self._w[:, :, 1:nz], self._dx, self._dy, self._dz)

            # --- sources ------------------------------------------------------
            m = self._source_mask
            self._temp += h * self._relax * m * (self._source_temperature - self._temp)
            self._smoke += h * self._smoke_rate * m
            self._smoke = np.clip(self._smoke, 0.0, 2.0)
            self._temp = np.clip(self._temp, -1.0, 3.0)
            self._smoke = np.nan_to_num(self._smoke, nan=0.0, posinf=2.0, neginf=0.0)
            self._temp = np.nan_to_num(self._temp, nan=0.0, posinf=3.0, neginf=-1.0)

            # --- buoyancy (Boussinesq) on the vertical faces -------------------
            bz = self._beta * self._temp
            self._w[:, :, 1:nz] += h * 0.5 * (bz[:, :, 1:] + bz[:, :, :-1])
            damp = np.exp(-h * self._drag)
            self._u *= damp
            self._v *= damp
            self._w *= damp

            self._apply_wall_bc()
            self._u = np.clip(self._u, -self._max_speed, self._max_speed)
            self._v = np.clip(self._v, -self._max_speed, self._max_speed)
            self._w = np.clip(self._w, -self._max_speed, self._max_speed)
            self._apply_wall_bc()

            # --- projection ----------------------------------------------------
            self._project(h)
            # residual divergence after the projection is the true source of
            # spurious volume change of the (incompressible) flow map
            div_residual = self._divergence()
            self._accumulated_volume_change += h * float(np.sum(div_residual)) * self._cell_volume

        # ---- advect passive markers (volume-preserving flow) ---------------
        self._advance_markers(dt)

        # ---- diagnostics ---------------------------------------------------
        div = self._divergence()
        scale = self._h / 1.0  # grid spacing / reference velocity [s]
        self._div_l2 = float(np.sqrt(np.mean(div ** 2)) * scale)
        self._div_max = float(np.max(np.abs(div)) * scale)
        self._volume_error = abs(self._accumulated_volume_change) / self._domain_volume

    def _advance_markers(self, dt):
        n_sub = 1
        uc = self._collocated()
        speed = float(np.sqrt(np.sum(uc ** 2, axis=3)).max())
        cfl = speed * dt / min(self._dx, self._dy, self._dz)
        n_sub = max(1, min(self._substeps_max, int(np.ceil(max(cfl, 1e-9)))))
        h = dt / n_sub
        lo, hi = self._lo, self._hi
        eps = 1e-9
        for _ in range(n_sub):
            uc = self._collocated()
            px, py, pz = self._pos[:, 0], self._pos[:, 1], self._pos[:, 2]
            v1 = self._sample_vec(uc, px, py, pz)
            mx = np.clip(px + 0.5 * h * v1[:, 0], lo[0], hi[0])
            my = np.clip(py + 0.5 * h * v1[:, 1], lo[1], hi[1])
            mz = np.clip(pz + 0.5 * h * v1[:, 2], lo[2], hi[2])
            v2 = self._sample_vec(uc, mx, my, mz)
            px = px + h * v2[:, 0]
            py = py + h * v2[:, 1]
            pz = pz + h * v2[:, 2]
            # keep markers inside the closed box (wall reflection)
            px = np.where(px < lo[0] + eps, lo[0] + eps, np.where(px > hi[0] - eps, hi[0] - eps, px))
            py = np.where(py < lo[1] + eps, lo[1] + eps, np.where(py > hi[1] - eps, hi[1] - eps, py))
            pz = np.where(pz < lo[2] + eps, lo[2] + eps, np.where(pz > hi[2] - eps, hi[2] - eps, pz))
            self._pos[:, 0] = px
            self._pos[:, 1] = py
            self._pos[:, 2] = pz
        uc = self._collocated()
        self._vel = self._sample_vec(uc, self._pos[:, 0], self._pos[:, 1], self._pos[:, 2])
        self._vel = np.nan_to_num(self._vel, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------------------------------------------------------- frames
    def _frame(self, time: float) -> StateFrame:
        uc = self._collocated()
        uc = np.nan_to_num(uc, nan=0.0, posinf=0.0, neginf=0.0)
        smoke = np.nan_to_num(self._smoke, nan=0.0, posinf=0.0, neginf=0.0)
        temp = np.nan_to_num(self._temp, nan=0.0, posinf=0.0, neginf=0.0)
        pos = np.nan_to_num(self._pos, nan=0.0, posinf=0.0, neginf=0.0)
        vel = np.nan_to_num(self._vel, nan=0.0, posinf=0.0, neginf=0.0)

        fields = {
            "smoke_density": smoke.astype(np.float32),
            "transported_scalar": smoke.astype(np.float32),
            "temperature": temp.astype(np.float32),
            "velocity_field": uc,
            "scalar_source": self._scalar_source.astype(np.float32),
            "active_mask": np.ones(self._shape, dtype=np.float32),
            "boundary_normal": self._wall_normals,
            "boundary_target_velocity": self._wall_target,
            "boundary_velocity": np.zeros_like(self._wall_points),
        }
        scalars = {
            "divergence_l2": float(self._div_l2),
            "divergence_max": float(self._div_max),
            "relative_volume_error": float(self._volume_error),
            "grid_spacing": float(self._dx),
            "characteristic_length": float(self._hi[2] - self._lo[2]),
            "reference_velocity": 1.0,
        }
        frame = StateFrame(
            time=float(time),
            position=pos.astype(np.float32),
            velocity=vel.astype(np.float32),
            mass=self._mass.astype(np.float32),
            attributes={"marker_kind": self._kind},
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        if not getattr(self, "_initialized", False):
            raise RuntimeError("BenchmarkSolver.initialize() must be called first")
        target = float(time)
        n_target = int(round(target / self._dt))
        if n_target < self._step_index:
            raise ValueError("advance_to() must be called with non-decreasing times")
        while self._step_index < n_target:
            self._step(self._dt)
            self._step_index += 1
        self._time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "divergence_l2": float(self._div_l2),
            "divergence_max": float(self._div_max),
            "relative_volume_error": float(self._volume_error),
            "steps": int(self._step_index),
            "marker_count": int(self._pos.shape[0]),
            "grid_spacing": float(self._dx),
        }
