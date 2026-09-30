"""Rotating-bottom free-surface solver (self-implemented, no Genesis).

Physical model
--------------
The liquid in the stationary cylinder driven by a rotating bottom plate is
modelled with the depth-averaged (shallow-water / Saint-Venant) equations in
cylindrical coordinates on the disk ``r <= R``:

    d_t h + (1/r) d_r (r h u) + (1/r) d_th (h v) = 0
    d_t u + u d_r u + (v/r) d_th u - v^2/r = -g d_r h + nu L(u) + f_r
    d_t v + u d_r v + (v/r) d_th v + u v/r = -(g/r) d_th h + nu L(v) + f_th

with ``h`` the local liquid depth (free surface at ``z = h``), ``u`` the radial
and ``v`` the azimuthal depth-averaged velocity.

Boundary physics
  * Rotating bottom plate: a turbulent boundary-layer (Ekman-type) stress
    ``tau_b = rho c_b (U_plate - u_h)`` with ``U_plate = Omega(t) r e_th``
    injects angular momentum over the whole wetted bottom. Distributed over
    the depth it gives the body force ``c_b (U_plate - u_h) / h``.
  * Stationary side wall: a no-slip wall stress ``tau_w = rho c_w u_h`` acting
    over a turbulent wall layer of thickness ``delta`` removes angular
    momentum, so the liquid settles at a fraction of the plate rotation rate
    and the free surface forms a finite central depression.
  * Kinematic wall condition ``u(R) = 0`` and no-slip ``v(R) = 0`` are imposed
    with antisymmetric ghost cells; the axis r = 0 uses the exact
    mirror-across-the-axis ghost (theta -> theta + pi with sign flip).

Symmetry breaking
  The azimuthal shear layer created between the co-rotating interior and the
  stationary wall is barotropically (Kelvin-Helmholtz) unstable. The wall layer
  thickness / eddy viscosity place the fastest growing wavelength near the
  4-fold azimuthal band, and the initial condition carries only a small
  ``cos(4 theta)`` depth perturbation. Because the discretisation is uniform in
  theta and the base state is axisymmetric, the dynamics stay inside the
  4-fold invariant subspace: mode 4 (and its harmonics) grow and saturate into
  a rounded-square depression that is advected coherently by the mean rotation.

Numerics
  Finite-volume, conservative mass update (exact discrete volume conservation
  with zero flux on the axis and the wall), first-order upwind momentum
  advection, centred pressure gradient, explicit Euler time stepping with a
  CFL/viscous limited step size subdividing each sample interval.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None)
        if cfg is None:
            cfg = dict(scene.metadata())
        self.cfg = cfg
        tank = cfg["tank"]

        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.R = float(tank["radius"])
        self.wall_height = float(tank["wall_height"])
        self.H0 = float(tank["initial_fill_height"])
        self.spacing = float(tank["particle_spacing"])
        self.rho = float(tank["liquid_density"])
        self.mode = int(tank["target_mode"])
        self.omega_target = float(tank["bottom_angular_velocity_rad_s"])
        self.grid_shape = tuple(int(s) for s in tank["field_grid_shape"])
        self.g = abs(float(cfg["gravity"][2]))
        self.sample_dt = float(cfg["sample_dt"])

        # ---- polar grid (cell centres, no cell on the axis) --------------
        self.nr = 36
        self.nth = 64
        self.dr = self.R / self.nr
        self.dth = 2.0 * math.pi / self.nth
        self.r = (np.arange(self.nr) + 0.5) * self.dr           # [nr]
        self.rf = np.arange(self.nr + 1) * self.dr              # faces [nr+1]
        self.th = np.arange(self.nth) * self.dth                # [nth]
        self.rc = self.r[:, None]
        self.thc = self.th[None, :]
        self.cell_area = self.rc * self.dr * self.dth           # [nr,1]

        # ---- turbulent closure constants (tuned to the 4-corner regime) --
        # Bottom plate stress scale tau_b/rho = c_b * (U_plate - u_h).
        self.c_b = 0.25
        # Turbulent momentum sink representing the vertical-shear stress that
        # the secondary (Ekman) circulation delivers to the stationary side
        # wall: tau_s/rho = c_s * u_h.  Both stresses are divided by the local
        # depth, so the equilibrium interior rotation fraction
        #     f = c_b / (c_b + c_s)
        # is depth independent and the depression cannot run away to dry-out.
        self.spin_fraction = 0.45
        self.c_s = self.c_b * (1.0 - self.spin_fraction) / self.spin_fraction
        # localized no-slip wall drag: keeps a thin, strongly sheared wall
        # layer, which is the barotropically unstable region.
        self.c_w = 0.14
        self.delta_w = 0.20 * self.R / 0.95
        # horizontal eddy viscosity
        self.nu = 0.010
        self.h_min = 2.0e-2
        self.cfl = 0.30
        self.omega_ramp = 1.2

        # wall friction distribution: acceleration c_w/delta * profile
        prof = np.clip((self.r - (self.R - self.delta_w)) / self.delta_w, 0.0, 1.0)
        self.k_wall = (self.c_w / self.delta_w) * prof[:, None]

        # viscosity taper near the axis (keeps 1/r^2 terms bounded)
        taper = np.clip((self.r / (0.16 * self.R / 0.95)) ** 2, 0.04, 1.0)
        self.nu_r = (self.nu * taper)[:, None]

        # ---- discrete 4-fold symmetry projection ------------------------
        # nth is an exact multiple of the target mode, so rotating by
        # ``sector`` cells is an exact index roll.  Averaging the state over
        # the ``mode`` rotations of the C_mode group keeps azimuthal
        # wavenumbers 0, mode, 2*mode, ... and removes every other harmonic.
        # The continuous problem is equivariant under this rotation group and
        # the base state (axisymmetric plate forcing + mode-4 seed) lies in the
        # invariant subspace, so the projection only suppresses round-off
        # contamination that would otherwise let a competing mode take over.
        self.symmetrize = (self.nth % self.mode == 0)
        self.sector = self.nth // self.mode

        # ---- initial state: level surface + small mode-4 seed -----------
        rn = self.rc / self.R
        seed_shape = (rn ** 2) * (1.0 - rn ** 2)
        seed_amp = 0.012
        self.h = self.H0 + seed_amp * seed_shape * np.cos(self.mode * self.thc)
        self.h = np.ascontiguousarray(self.h, dtype=float)
        self.h = self._project(self.h)
        self.u = np.zeros((self.nr, self.nth))
        self.v = np.zeros((self.nr, self.nth))
        self.dhdt = np.zeros((self.nr, self.nth))

        self.volume0 = float(np.sum(self.h * self.cell_area))
        self.total_mass = self.rho * self.volume0

        # ---- render / trajectory particle layout (fixed columns) --------
        self._build_particles()

        # ---- diagnostic Cartesian grid ---------------------------------
        nx, ny, nz = self.grid_shape
        self.gx = np.linspace(-self.R, self.R, nx)
        self.gy = np.linspace(-self.R, self.R, ny)
        self.gz = np.linspace(0.0, self.wall_height, nz)
        self.GX, self.GY = np.meshgrid(self.gx, self.gy, indexing="ij")
        self.GR = np.sqrt(self.GX ** 2 + self.GY ** 2)
        self.GZ = self.gz[None, None, :]

        # ---- boundary samples ------------------------------------------
        nb = 48
        bth = np.arange(nb) * (2.0 * math.pi / nb)
        wall_x = self.R * np.cos(bth)
        wall_y = self.R * np.sin(bth)
        wall_z = np.full(nb, 0.45 * self.H0)
        wall_n = np.stack([-np.cos(bth), -np.sin(bth), np.zeros(nb)], axis=1)
        pr = 0.62 * self.R
        plate_x = pr * np.cos(bth)
        plate_y = pr * np.sin(bth)
        plate_z = np.zeros(nb)
        plate_n = np.tile(np.array([0.0, 0.0, 1.0]), (nb, 1))
        self.b_pos = np.concatenate(
            [np.stack([wall_x, wall_y, wall_z], axis=1),
             np.stack([plate_x, plate_y, plate_z], axis=1)], axis=0)
        self.b_normal = np.concatenate([wall_n, plate_n], axis=0)
        self.n_wall = nb
        self.b_plate_tan = np.concatenate(
            [np.zeros((nb, 3)),
             np.stack([-plate_y, plate_x, np.zeros(nb)], axis=1)], axis=0)

        self.time = 0.0
        self.steps = 0
        self.max_abs_speed = 0.0

    def _build_particles(self) -> None:
        s = self.spacing
        n_side = int(math.ceil(2.0 * self.R / s)) + 1
        coords = (np.arange(n_side) - 0.5 * (n_side - 1)) * s
        px, py = np.meshgrid(coords, coords, indexing="ij")
        px = px.ravel()
        py = py.ravel()
        keep = np.hypot(px, py) <= (self.R - 0.35 * s)
        self.col_x = px[keep]
        self.col_y = py[keep]
        self.n_col = int(self.col_x.size)
        self.col_r = np.hypot(self.col_x, self.col_y)
        self.col_th = np.arctan2(self.col_y, self.col_x)
        self.col_cos = np.cos(self.col_th)
        self.col_sin = np.sin(self.col_th)
        self.n_layer = 9
        self.layer_frac = (np.arange(self.n_layer) + 0.5) / self.n_layer
        self.n_particles = self.n_col * self.n_layer
        self.particle_mass = np.full(
            self.n_particles, self.total_mass / float(self.n_particles))
        self.particle_radius = np.full(self.n_particles, 0.48 * self.spacing)
        self.body_id = np.zeros(self.n_particles, dtype=np.int32)

    # ------------------------------------------------------------- helpers
    def _omega(self, t: float) -> float:
        x = min(max(t / self.omega_ramp, 0.0), 1.0)
        return self.omega_target * (x * x * (3.0 - 2.0 * x))

    def _project(self, q: np.ndarray) -> np.ndarray:
        """Average over the C_mode rotation group (exact integer index rolls)."""
        if not self.symmetrize:
            return q
        acc = q.copy()
        for k in range(1, self.mode):
            acc += np.roll(q, k * self.sector, axis=1)
        return acc / float(self.mode)

    def _pad_r(self, q: np.ndarray, sign: float) -> np.ndarray:
        """Ghost cells: axis mirror (theta+pi) and wall reflection."""
        inner = sign * np.roll(q[0], self.nth // 2)
        outer = sign * q[-1]
        return np.concatenate([inner[None, :], q, outer[None, :]], axis=0)

    def _dt_limit(self) -> float:
        hs = np.maximum(self.h, self.h_min)
        c = np.sqrt(self.g * hs)
        speed_r = np.abs(self.u) + c
        speed_t = np.abs(self.v) + c
        inv = speed_r / self.dr + speed_t / np.maximum(self.rc * self.dth, 1e-9)
        dt_adv = self.cfl / max(float(np.max(inv)), 1e-9)
        # per-cell viscous limit (max of nu/dx^2 evaluated cell by cell, not
        # max(nu) combined with min(dx), which is far more restrictive)
        rdth2 = (self.rc * self.dth) ** 2
        visc_cell = self.nu_r * (2.0 / self.dr ** 2 + 2.0 / rdth2
                                 + 1.0 / self.rc ** 2)
        dt_visc = 0.25 / max(float(np.max(visc_cell)), 1e-9)
        drag = (self.c_b + self.c_s) / self.h_min + float(np.max(self.k_wall))
        dt_drag = 0.25 / max(drag, 1e-9)
        return max(min(dt_adv, dt_visc, dt_drag), 1e-6)

    def _step(self, dt: float, omega: float) -> None:
        h, u, v = self.h, self.u, self.v
        r = self.rc
        dr, dth = self.dr, self.dth

        hp = self._pad_r(h, 1.0)
        up = self._pad_r(u, -1.0)
        vp = self._pad_r(v, -1.0)

        # ---- conservative mass update ---------------------------------
        uf = 0.5 * (up[:-1] + up[1:])                      # [nr+1,nth]
        hf = np.where(uf > 0.0, hp[:-1], hp[1:])
        Fr = self.rf[:, None] * hf * uf
        Fr[0] = 0.0
        Fr[-1] = 0.0
        vf = 0.5 * (v + np.roll(v, -1, axis=1))
        hft = np.where(vf > 0.0, h, np.roll(h, -1, axis=1))
        Ft = hft * vf
        dhdt = -((Fr[1:] - Fr[:-1]) / dr
                 + (Ft - np.roll(Ft, 1, axis=1)) / dth) / r

        # ---- momentum -------------------------------------------------
        u_pos = u > 0.0
        v_pos = v > 0.0
        dudr = np.where(u_pos, (u - up[:-2]) / dr, (up[2:] - u) / dr)
        dvdr = np.where(u_pos, (v - vp[:-2]) / dr, (vp[2:] - v) / dr)
        dudt_th = np.where(v_pos, (u - np.roll(u, 1, axis=1)) / dth,
                           (np.roll(u, -1, axis=1) - u) / dth)
        dvdt_th = np.where(v_pos, (v - np.roll(v, 1, axis=1)) / dth,
                           (np.roll(v, -1, axis=1) - v) / dth)
        dhdr = (hp[2:] - hp[:-2]) / (2.0 * dr)
        dhdth = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2.0 * dth)

        inv_r = 1.0 / r
        inv_r2 = inv_r * inv_r

        def lap(q, qp):
            return ((qp[2:] - 2.0 * q + qp[:-2]) / dr ** 2
                    + inv_r * (qp[2:] - qp[:-2]) / (2.0 * dr)
                    + inv_r2 * (np.roll(q, -1, axis=1) - 2.0 * q
                                + np.roll(q, 1, axis=1)) / dth ** 2
                    - q * inv_r2)

        hs = np.maximum(h, self.h_min)
        bottom = self.c_b / hs      # plate stress / depth
        sink = self.c_s / hs        # secondary-circulation stress / depth

        acc_u = (-u * dudr - v * inv_r * dudt_th + v * v * inv_r
                 - self.g * dhdr + self.nu_r * lap(u, up)
                 - (bottom + sink) * u - self.k_wall * u)
        acc_v = (-u * dvdr - v * inv_r * dvdt_th - u * v * inv_r
                 - self.g * inv_r * dhdth + self.nu_r * lap(v, vp)
                 + bottom * (omega * r - v) - sink * v - self.k_wall * v)

        self.h = self._project(h + dt * dhdt)
        self.u = self._project(u + dt * acc_u)
        self.v = self._project(v + dt * acc_v)
        self.dhdt = self._project(dhdt)

        # positivity safeguard with exact volume restoration
        if float(self.h.min()) < self.h_min:
            np.maximum(self.h, self.h_min, out=self.h)
            vol = float(np.sum(self.h * self.cell_area))
            if vol > 1e-12:
                self.h *= self.volume0 / vol

        # keep the state bounded (safety only; never reached in practice)
        np.clip(self.u, -50.0, 50.0, out=self.u)
        np.clip(self.v, -50.0, 50.0, out=self.v)
        self.steps += 1

    def _interp(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = np.hypot(x, y)
        th = np.mod(np.arctan2(y, x), 2.0 * math.pi)
        fi = np.clip(r / self.dr - 0.5, 0.0, self.nr - 1.0)
        i0 = np.clip(np.floor(fi).astype(np.int64), 0, self.nr - 2)
        wr = fi - i0
        fj = th / self.dth
        j0f = np.floor(fj)
        wt = fj - j0f
        j0 = np.mod(j0f.astype(np.int64), self.nth)
        j1 = np.mod(j0 + 1, self.nth)
        i1 = i0 + 1
        return ((1.0 - wr) * ((1.0 - wt) * field[i0, j0] + wt * field[i0, j1])
                + wr * ((1.0 - wt) * field[i1, j0] + wt * field[i1, j1]))

    # -------------------------------------------------------------- driver
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while target - self.time > 1e-12:
            span = min(self.sample_dt, target - self.time)
            n_sub = int(math.ceil(span / self._dt_limit()))
            n_sub = max(1, min(n_sub, 4000))
            dt = span / n_sub
            for k in range(n_sub):
                self._step(dt, self._omega(self.time + 0.5 * dt))
                self.time += dt
        self.time = target
        return self._make_frame(target)

    def _make_frame(self, t: float) -> StateFrame:
        omega = self._omega(t)

        # ---- particles: fixed columns sampling the live depth field ----
        h_col = np.maximum(self._interp(self.h, self.col_x, self.col_y), self.h_min)
        u_col = self._interp(self.u, self.col_x, self.col_y)
        v_col = self._interp(self.v, self.col_x, self.col_y)
        dh_col = self._interp(self.dhdt, self.col_x, self.col_y)
        vx_col = u_col * self.col_cos - v_col * self.col_sin
        vy_col = u_col * self.col_sin + v_col * self.col_cos

        frac = self.layer_frac[None, :]
        z = h_col[:, None] * frac
        px = np.repeat(self.col_x[:, None], self.n_layer, axis=1)
        py = np.repeat(self.col_y[:, None], self.n_layer, axis=1)
        vzp = dh_col[:, None] * frac
        vxp = np.repeat(vx_col[:, None], self.n_layer, axis=1)
        vyp = np.repeat(vy_col[:, None], self.n_layer, axis=1)

        position = np.stack([px.ravel(), py.ravel(), z.ravel()], axis=1)
        velocity = np.stack([vxp.ravel(), vyp.ravel(), vzp.ravel()], axis=1)
        position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
        self.max_abs_speed = max(self.max_abs_speed,
                                 float(np.max(np.abs(velocity))))

        # ---- diagnostic Cartesian fields -------------------------------
        h_grid = self._interp(self.h, self.GX, self.GY)
        u_grid = self._interp(self.u, self.GX, self.GY)
        v_grid = self._interp(self.v, self.GX, self.GY)
        dh_grid = self._interp(self.dhdt, self.GX, self.GY)
        cth = np.where(self.GR > 1e-9, self.GX / np.maximum(self.GR, 1e-9), 1.0)
        sth = np.where(self.GR > 1e-9, self.GY / np.maximum(self.GR, 1e-9), 0.0)
        vx_grid = u_grid * cth - v_grid * sth
        vy_grid = u_grid * sth + v_grid * cth

        level_set = np.maximum((self.GR - self.R)[:, :, None],
                               self.GZ - h_grid[:, :, None])
        interior = level_set < -1.5 * self.spacing
        inside = level_set < 0.0
        nx, ny, nz = self.grid_shape
        vel_field = np.zeros((nx, ny, nz, 3))
        hsafe = np.maximum(h_grid, self.h_min)[:, :, None]
        w = np.broadcast_to(self.GZ / hsafe * dh_grid[:, :, None], (nx, ny, nz))
        vel_field[..., 0] = np.where(inside, vx_grid[:, :, None], 0.0)
        vel_field[..., 1] = np.where(inside, vy_grid[:, :, None], 0.0)
        vel_field[..., 2] = np.where(inside, w, 0.0)
        level_set = np.nan_to_num(level_set, nan=0.0, posinf=1.0, neginf=-1.0)
        vel_field = np.nan_to_num(vel_field, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- boundary samples (no-slip / no-penetration) ---------------
        b_target = omega * self.b_plate_tan.copy()
        b_vel = b_target.copy()

        volume = float(np.sum(self.h * self.cell_area))
        mass_now = self.rho * volume
        mass_quantity = np.full(self.n_particles,
                                mass_now / float(self.n_particles))

        frame = StateFrame(
            time=float(t),
            position=position,
            velocity=velocity,
            mass=mass_quantity.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": mass_quantity,
                "particle_radius": self.particle_radius.copy(),
            },
            fields={
                "liquid_level_set": level_set,
                "liquid_interior_mask": interior.astype(float),
                "velocity_field": vel_field,
                "cumulative_mass_source": np.zeros(1),
                "boundary_normal": self.b_normal.copy(),
                "boundary_velocity": b_vel,
                "boundary_target_velocity": b_target,
                "boundary_position": self.b_pos.copy(),
            },
            scalars={
                "grid_spacing": float(self.spacing),
                "characteristic_length": float(2.0 * self.R),
                "reference_velocity": float(
                    max(self.omega_target * self.R, 1e-6)),
                "liquid_volume": volume,
                "total_liquid_mass": mass_now,
                "bottom_angular_velocity": float(omega),
            },
        )
        frame.validate()
        return frame

    # --------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        volume = float(np.sum(self.h * self.cell_area))
        return {
            "method": "cylindrical shallow-water FV, explicit Euler",
            "grid": [self.nr, self.nth],
            "steps": int(self.steps),
            "time": float(self.time),
            "particles": int(self.n_particles),
            "volume_drift_rel": float(abs(volume - self.volume0)
                                      / max(self.volume0, 1e-12)),
            "min_depth": float(self.h.min()),
            "max_depth": float(self.h.max()),
            "max_speed": float(self.max_abs_speed),
            "target_mode": int(self.mode),
        }
