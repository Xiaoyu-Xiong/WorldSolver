"""Egg frying on a hot pan: two-layer thin-film (lubrication) multiphysics solver.

Physical model
--------------
The deposited egg is represented as two coexisting incompressible viscous films
resting on the hot pan surface: the albumen (egg white, ``body_id == 1``) and the
yolk (``body_id == 2``).  Each film ``m`` carries a thickness field
``h_m(x, y, t)`` on a fixed uniform Cartesian grid covering the pan interior.

Lateral motion uses the lubrication / depth-averaged Stokes limit of the
incompressible Navier-Stokes equations for a shallow gravity current on a
no-slip plate,

    q_m = -(g / (3 nu_m)) * h_liq_m^3 * grad(Phi_m),
    d h_m / dt + div(q_m) = 0,

with the piezometric head coupling both phases through their hydrostatic load,

    Phi_w = h_w + lam * (rho_y / rho_w) * h_y,
    Phi_y = h_y + lam * (rho_w / rho_y) * h_w.

``h_liq_m = h_m * (1 - s_m)`` is the still-liquid part of the column, so the
coagulated (cooked) fraction removes mobility exactly where the protein has set:
the spreading front is arrested by cooking rather than by any scripted rule.

Heat transfer is solved in a terrain-following coordinate zeta = (z - z_pan)/h
with ``NZ`` layers per column: vertical conduction (thermal diffusivity alpha_m)
with a strong contact-conductance boundary to the frozen 443 K pan surface at
zeta = 0, convective loss to ambient air at zeta = 1, plus conservative upwind
horizontal advection of h*T by the depth-averaged velocity.

Coagulation is a first-order Arrhenius phase transition integrated with an
unconditionally stable exponential integrator,

    d s / dt = A_m * exp(-Ea / (R T)) * (1 - s),

so the solid-phase front nucleates on the pan-contact plane and diffuses upward
(phase diffusion enabled).  Material concentrations are one-hot per particle and
never mixed (concentration diffusion disabled), which conserves the mass of each
material exactly.

Particles are Lagrangian tracers of this continuum: they are advected by the
depth-averaged horizontal velocity u = q/h and ride at z = z_pan + zeta * h with
zeta materially conserved (plug-flow limit), so the particle cloud reproduces the
film thickness field it is drawn from.

Numerics: explicit finite volume, cell-centred, adaptive time step limited by the
degenerate-diffusion, advective CFL and conduction stability bounds.  No RNG is
used, so the trajectory is bit-deterministic.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

R_GAS = 8.314462618  # J/(mol K)

# --- Grid / discretisation -------------------------------------------------
NX = 61          # cells per horizontal direction of the diagnostic film grid
NZ = 6           # terrain-following layers resolving the vertical phase front
DT_MAX = 5.0e-4  # s, hard ceiling on the adaptive step
DT_MIN = 2.0e-6  # s, floor so the loop always terminates
CFL_DIFF = 0.20  # safety factor for the degenerate h^3 diffusion bound
CFL_ADV = 0.35   # safety factor for the advective bound
C_VERT_MAX = 0.15  # clamp on the vertical conduction diffusion number
H_DRY = 1.0e-5   # m, film thickness below which a column counts as dry

# --- Material properties (SI) ---------------------------------------------
# Kinematic viscosities of the raw phases. Albumen is a shear-thinning, weakly
# viscoelastic fluid; the yolk is far more viscous and membrane-bound, so it
# holds its dome while the white runs out into a thin sheet.
NU_WHITE_RAW = 3.0e-3   # m^2/s
NU_YOLK_RAW = 5.0       # m^2/s
NU_SET_GAIN = 40.0      # viscosity multiplier as the column coagulates

ALPHA_WHITE = 2.8e-5    # m^2/s thermal diffusivity of albumen
ALPHA_YOLK = 2.0e-5     # m^2/s thermal diffusivity of yolk
K_WHITE = 0.60          # W/(m K)
K_YOLK = 0.45           # W/(m K)
H_AIR = 15.0            # W/(m^2 K) free-convection loss to ambient air
H_PAN_WHITE = 3.0e3     # W/(m^2 K) pan-to-albumen contact conductance
H_PAN_YOLK = 1.2e3      # W/(m^2 K) pan-to-yolk contact conductance (white film between)

# First-order protein coagulation (Arrhenius). Ea is the classic ovalbumin
# denaturation scale; the prefactors are set so albumen sets near 340-350 K and
# the yolk sets more slowly, as observed when frying.
EA_COAG = 2.50e5        # J/mol (ovalbumin denaturation scale)
# Prefactors are calibrated so albumen coagulates on a ~1 s timescale once a
# column passes ~345 K and the yolk sets more slowly, compressing the transition
# into the scene's 8 s window while keeping the Arrhenius temperature response.
A_WHITE = 7.0e37        # 1/s
A_YOLK = 5.0e35         # 1/s
RATE_MAX = 200.0        # 1/s clamp keeping the exponential integrator tame
D_PHASE = 6.0e-6        # m^2/s upward phase (solid-fraction) diffusivity
D_PHASE_LAT = 4.0e-7    # m^2/s weak lateral phase diffusivity

LAMBDA_XLOAD = 0.5      # partial-overlap hydrostatic cross-load coefficient


def _box_smooth(field: np.ndarray) -> np.ndarray:
    """One separable [1,2,1]/4 pass with zero-extension outside the domain."""
    padded = np.zeros((field.shape[0] + 2, field.shape[1] + 2), dtype=float)
    padded[1:-1, 1:-1] = field
    out = 0.25 * (padded[:-2, 1:-1] + 2.0 * padded[1:-1, 1:-1] + padded[2:, 1:-1])
    padded[1:-1, 1:-1] = out
    padded[0, :] = 0.0
    padded[-1, :] = 0.0
    padded[:, 0] = 0.0
    padded[:, -1] = 0.0
    return 0.25 * (padded[1:-1, :-2] + 2.0 * padded[1:-1, 1:-1] + padded[1:-1, 2:])


def _grad_x(field: np.ndarray, dx: float) -> np.ndarray:
    """Face-centred x derivative, shape [nx+1, ny]; outer faces are closed."""
    out = np.zeros((field.shape[0] + 1, field.shape[1]), dtype=float)
    out[1:-1, :] = (field[1:, :] - field[:-1, :]) / dx
    return out


def _grad_y(field: np.ndarray, dx: float) -> np.ndarray:
    """Face-centred y derivative, shape [nx, ny+1]; outer faces are closed."""
    out = np.zeros((field.shape[0], field.shape[1] + 1), dtype=float)
    out[:, 1:-1] = (field[:, 1:] - field[:, :-1]) / dx
    return out


def _face_avg_x(field: np.ndarray) -> np.ndarray:
    out = np.zeros((field.shape[0] + 1, field.shape[1]), dtype=float)
    out[1:-1, :] = 0.5 * (field[1:, :] + field[:-1, :])
    return out


def _face_avg_y(field: np.ndarray) -> np.ndarray:
    out = np.zeros((field.shape[0], field.shape[1] + 1), dtype=float)
    out[:, 1:-1] = 0.5 * (field[:, 1:] + field[:, :-1])
    return out


def _divergence(flux_x: np.ndarray, flux_y: np.ndarray, dx: float) -> np.ndarray:
    return (flux_x[1:, :] - flux_x[:-1, :]) / dx + (flux_y[:, 1:] - flux_y[:, :-1]) / dx


def _upwind_face_x(scalar: np.ndarray, flux_x: np.ndarray) -> np.ndarray:
    """Upwind interior face value of ``scalar`` given face volume flux."""
    out = np.zeros_like(flux_x)
    left = scalar[:-1, :]
    right = scalar[1:, :]
    inner = flux_x[1:-1, :]
    out[1:-1, :] = np.where(inner >= 0.0, left, right)
    return out


def _upwind_face_y(scalar: np.ndarray, flux_y: np.ndarray) -> np.ndarray:
    out = np.zeros_like(flux_y)
    lower = scalar[:, :-1]
    upper = scalar[:, 1:]
    inner = flux_y[:, 1:-1]
    out[:, 1:-1] = np.where(inner >= 0.0, lower, upper)
    return out


def _bilinear(field: np.ndarray, xs: np.ndarray, ys: np.ndarray,
              origin: float, dx: float) -> np.ndarray:
    """Bilinear sample of a cell-centred field at scattered points."""
    nx, ny = field.shape
    fx = (xs - origin) / dx
    fy = (ys - origin) / dx
    i0 = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
    j0 = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
    tx = np.clip(fx - i0, 0.0, 1.0)
    ty = np.clip(fy - j0, 0.0, 1.0)
    i1 = i0 + 1
    j1 = j0 + 1
    return (
        field[i0, j0] * (1.0 - tx) * (1.0 - ty)
        + field[i1, j0] * tx * (1.0 - ty)
        + field[i0, j1] * (1.0 - tx) * ty
        + field[i1, j1] * tx * ty
    )


class _Film:
    """State and frozen properties of one incompressible egg phase."""

    def __init__(self, name, body_id, mass, nu_raw, alpha, conduct,
                 h_pan, a_coag, shape):
        self.name = name
        self.body_id = int(body_id)
        self.mass = float(mass)
        self.nu_raw = float(nu_raw)
        self.alpha = float(alpha)
        self.conduct = float(conduct)
        self.h_pan = float(h_pan)
        self.a_coag = float(a_coag)
        self.h = np.zeros(shape, dtype=float)         # [nx,ny] film thickness (m)
        self.temp = np.zeros(shape + (NZ,), dtype=float)   # [nx,ny,NZ] K
        self.solid = np.zeros(shape + (NZ,), dtype=float)  # [nx,ny,NZ] 0..1
        self.ux = np.zeros(shape, dtype=float)        # depth-averaged velocity
        self.uy = np.zeros(shape, dtype=float)
        self.dhdt = np.zeros(shape, dtype=float)
        self.rho = 1000.0
        self.volume = 0.0
        self.rho_c = 1.0e6                            # rho * c_p (J/m^3/K)
        self.dp = 0.007                                # lattice pitch of this phase
        self.heat_rate = np.zeros(shape, dtype=float)  # column-mean dT/dt source


class BenchmarkSolver:
    """Deterministic two-layer thin-film multiphysics solver for egg frying."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(scene.config) if hasattr(scene, "config") else {}
        self.config = config

        initial = dict(config.get("initial_conditions", {}))
        boundary = dict(config.get("boundary_conditions", {}))

        self.t_ambient = float(initial.get("ambient_temperature_k", 296.0))
        self.t_egg0 = float(initial.get("egg_temperature_k", 291.0))
        self.t_pan = float(boundary.get("pan_surface_temperature_k",
                                        initial.get("pan_temperature_k", 443.0)))
        gravity = np.asarray(initial.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.gravity = gravity
        self.g = float(abs(gravity[2])) if abs(gravity[2]) > 1e-9 else 9.81

        self.z_pan = float(boundary.get("pan_contact_height_m", 0.018))
        self.r_pan = float(boundary.get("pan_no_escape_radius_m", 0.185))
        self.phase_diffusion = bool(initial.get("phase_diffusion_enabled", True))
        self.conc_diffusion = bool(initial.get("concentration_diffusion_enabled", False))

        # --- frozen initial particle state from the scaffold scene ----------
        frame0 = scene.initial_state()
        self.p_xy = np.array(frame0.position[:, :2], dtype=float)
        self.p_z0 = np.array(frame0.position[:, 2], dtype=float)
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.p_mass = np.asarray(frame0.mass, dtype=float).copy()
        self.n_particles = int(self.p_xy.shape[0])
        attrs0 = frame0.attributes
        self.concentration = np.array(attrs0["material_concentration"], dtype=float)
        self.material_label = np.array(attrs0["material_label"], dtype=float)

        # --- diagnostic film grid over the pan interior ---------------------
        half = self.r_pan + 0.015
        self.nx = NX
        self.ny = NX
        self.dx = (2.0 * half) / self.nx
        self.origin = -half + 0.5 * self.dx
        idx = np.arange(self.nx, dtype=float)
        self.xc = self.origin + idx * self.dx
        self.yc = self.origin + idx * self.dx
        gx, gy = np.meshgrid(self.xc, self.yc, indexing="ij")
        self.gx = gx
        self.gy = gy
        radius = np.hypot(gx, gy)
        self.inside = radius <= self.r_pan
        # Closed pan-wall faces: no material may escape past the rim.
        inside_f = self.inside.astype(float)
        self.open_x = np.zeros((self.nx + 1, self.ny), dtype=float)
        self.open_x[1:-1, :] = inside_f[1:, :] * inside_f[:-1, :]
        self.open_y = np.zeros((self.nx, self.ny + 1), dtype=float)
        self.open_y[:, 1:-1] = inside_f[:, 1:] * inside_f[:, :-1]

        masses = {1: float(initial.get("white_mass_kg", 0.034)),
                  2: float(initial.get("yolk_mass_kg", 0.018))}
        shape = (self.nx, self.ny)
        self.white = _Film("egg_white", 1, masses[1], NU_WHITE_RAW, ALPHA_WHITE,
                           K_WHITE, H_PAN_WHITE, A_WHITE, shape)
        self.yolk = _Film("egg_yolk", 2, masses[2], NU_YOLK_RAW, ALPHA_YOLK,
                          K_YOLK, H_PAN_YOLK, A_YOLK, shape)
        self.films = (self.white, self.yolk)

        self.zeta = np.zeros(self.n_particles, dtype=float)
        for film in self.films:
            film.dp = self._particle_spacing(self.body_id == film.body_id)
        self.dp = float(np.mean([film.dp for film in self.films]))
        for film in self.films:
            self._seed_film(film)

        # zeta is the materially-conserved vertical coordinate of each tracer.
        self.zeta = np.clip(self.zeta, 0.02, 0.98)
        self.p_vel = np.zeros((self.n_particles, 3), dtype=float)
        self.p_temp = np.full(self.n_particles, self.t_egg0, dtype=float)
        self.p_solid = np.zeros(self.n_particles, dtype=float)

        self.time = 0.0
        self.steps = 0
        self.substep_min_dt = DT_MAX
        self._sync_particles()

    def _particle_spacing(self, select: np.ndarray) -> float:
        """Recover one body's lattice pitch without assuming its value."""
        best = None
        for axis in range(2):
            values = np.unique(np.round(self.p_xy[select, axis], 9))
            if values.size > 1:
                gaps = np.diff(values)
                gaps = gaps[gaps > 1e-6]
                if gaps.size:
                    candidate = float(np.min(gaps))
                    best = candidate if best is None else min(best, candidate)
        if best is None or not np.isfinite(best) or best <= 0.0:
            best = 0.007
        return float(best)

    def _seed_film(self, film: _Film) -> None:
        """Build h(x,y), rho, T and s from the frozen particle cloud."""
        select = self.body_id == film.body_id
        xs = self.p_xy[select, 0]
        ys = self.p_xy[select, 1]
        zs = self.p_z0[select]
        ii = np.clip(((xs - self.origin) / self.dx + 0.5).astype(np.int64),
                     0, self.nx - 1)
        jj = np.clip(((ys - self.origin) / self.dx + 0.5).astype(np.int64),
                     0, self.ny - 1)
        flat = ii * self.ny + jj
        size = self.nx * self.ny
        zmax = np.full(size, -np.inf)
        zmin = np.full(size, np.inf)
        np.maximum.at(zmax, flat, zs)
        np.minimum.at(zmin, flat, zs)
        occupied = np.isfinite(zmax)
        raw = np.zeros(size, dtype=float)
        raw[occupied] = (zmax[occupied] - zmin[occupied]) + film.dp
        raw = raw.reshape(self.nx, self.ny)
        smoothed = _box_smooth(raw)
        # Retain the true footprint: keep every occupied column, trim the fringe.
        smoothed = np.maximum(smoothed, 0.5 * raw)
        smoothed = np.where(smoothed < 0.15 * film.dp, 0.0, smoothed)
        smoothed *= self.inside

        volume = float(np.sum(smoothed)) * self.dx * self.dx
        if volume <= 0.0:
            volume = float(np.sum(raw)) * self.dx * self.dx
            smoothed = raw * self.inside
        film.volume = volume
        film.rho = film.mass / max(volume, 1e-12)
        # Water-like specific heat scaled by the effective (scene) density so the
        # conduction and contact-flux terms stay consistent with alpha and k.
        film.rho_c = film.conduct / film.alpha
        film.h = smoothed

        active = film.h > H_DRY
        film.temp[:] = self.t_ambient
        film.temp[active, :] = self.t_egg0
        film.solid[:] = 0.0

        # Materially-conserved zeta of each tracer inside its own column.
        h_at = _bilinear(film.h, xs, ys, self.origin, self.dx)
        h_at = np.maximum(h_at, film.dp)
        zeta = (zs - self.z_pan + 0.5 * film.dp) / h_at
        self.zeta[select] = np.clip(zeta, 0.02, 0.98)


    # -------------------------------------------------------------- physics
    def _mobility(self, film: _Film):
        """Depth-integrated lubrication mobility of the still-liquid column."""
        s_col = np.mean(film.solid, axis=2)
        nu_eff = film.nu_raw * (1.0 + NU_SET_GAIN * s_col ** 3)
        h_liq = np.maximum(film.h * (1.0 - s_col), 0.0)
        return (self.g / (3.0 * nu_eff)) * h_liq ** 3

    def _heads(self):
        """Piezometric heads with the hydrostatic cross-load of both phases."""
        rho_w = self.white.rho
        rho_y = self.yolk.rho
        head_w = self.white.h + LAMBDA_XLOAD * (rho_y / rho_w) * self.yolk.h
        head_y = self.yolk.h + LAMBDA_XLOAD * (rho_w / rho_y) * self.white.h
        return head_w, head_y

    def _fluxes(self, film: _Film, head: np.ndarray):
        mob = self._mobility(film)
        flux_x = -_face_avg_x(mob) * _grad_x(head, self.dx) * self.open_x
        flux_y = -_face_avg_y(mob) * _grad_y(head, self.dx) * self.open_y
        return mob, flux_x, flux_y

    def _advect_layers(self, layers: np.ndarray, h_old: np.ndarray,
                       h_new: np.ndarray, flux_x: np.ndarray,
                       flux_y: np.ndarray, dt: float, fill: np.ndarray):
        """Conservative upwind transport of h*phi for every vertical layer."""
        out = np.empty_like(layers)
        wet = h_new > H_DRY
        h_safe = np.where(wet, h_new, 1.0)
        for k in range(layers.shape[2]):
            phi = layers[:, :, k]
            amount = h_old * phi
            fx = _upwind_face_x(phi, flux_x) * flux_x
            fy = _upwind_face_y(phi, flux_y) * flux_y
            amount = amount - dt * _divergence(fx, fy, self.dx)
            value = amount / h_safe
            out[:, :, k] = np.where(wet, value, fill[:, :, k])
        return out

    def _conduct(self, film: _Film, dt: float) -> None:
        """Vertical conduction with hot-pan contact and ambient-air convection."""
        h = film.h
        wet = h > H_DRY
        dz = np.maximum(h, H_DRY) / NZ
        dz = np.maximum(dz, 1.0e-5)
        coeff = film.alpha * dt / (dz * dz)
        bi_pan = film.h_pan * dz / film.conduct
        bi_air = H_AIR * dz / film.conduct
        limit = 0.5 / (1.0 + np.maximum(bi_pan, bi_air))
        coeff = np.minimum(np.minimum(coeff, C_VERT_MAX), limit)
        coeff = np.where(wet, coeff, 0.0)

        temp = film.temp
        new = temp.copy()
        if NZ > 2:
            new[:, :, 1:-1] = temp[:, :, 1:-1] + coeff[:, :, None] * (
                temp[:, :, 2:] - 2.0 * temp[:, :, 1:-1] + temp[:, :, :-2]
            )
        # Hot pan boundary (Robin contact conductance) at zeta = 0.
        new[:, :, 0] = temp[:, :, 0] + coeff * (
            bi_pan * (self.t_pan - temp[:, :, 0]) + (temp[:, :, 1] - temp[:, :, 0])
        )
        # Free surface losing heat to ambient air at zeta = 1.
        new[:, :, -1] = temp[:, :, -1] + coeff * (
            bi_air * (self.t_ambient - temp[:, :, -1]) + (temp[:, :, -2] - temp[:, :, -1])
        )
        film.temp = np.clip(new, 150.0, 1200.0)

        q_bot = film.h_pan * (self.t_pan - film.temp[:, :, 0])
        q_top = H_AIR * (self.t_ambient - film.temp[:, :, -1])
        denom = film.rho_c * np.maximum(h, H_DRY)
        film.heat_rate = np.where(wet, (q_bot + q_top) / denom, 0.0)

    def _coagulate(self, film: _Film, dt: float) -> None:
        """Arrhenius protein coagulation plus upward phase diffusion."""
        wet = film.h > H_DRY
        temp = np.maximum(film.temp, 1.0)
        rate = film.a_coag * np.exp(-EA_COAG / (R_GAS * temp))
        rate = np.minimum(rate, RATE_MAX)
        rate = np.where(wet[:, :, None], rate, 0.0)
        solid = 1.0 - (1.0 - film.solid) * np.exp(-rate * dt)

        if self.phase_diffusion:
            dz = np.maximum(np.maximum(film.h, H_DRY) / NZ, 1.0e-5)
            cz = np.minimum(D_PHASE * dt / (dz * dz), C_VERT_MAX)
            cz = np.where(wet, cz, 0.0)[:, :, None]
            padded = np.concatenate(
                (solid[:, :, :1], solid, solid[:, :, -1:]), axis=2
            )
            solid = solid + cz * (
                padded[:, :, 2:] - 2.0 * solid + padded[:, :, :-2]
            )
            c_lat = min(D_PHASE_LAT * dt / (self.dx * self.dx), 0.15)
            if c_lat > 0.0:
                wetf = wet.astype(float)
                gate_x = (self.open_x[1:-1, :] * wetf[1:, :] * wetf[:-1, :])[:, :, None]
                gate_y = (self.open_y[:, 1:-1] * wetf[:, 1:] * wetf[:, :-1])[:, :, None]
                lap = np.zeros_like(solid)
                dsx = (solid[1:, :, :] - solid[:-1, :, :]) * gate_x
                lap[:-1, :, :] += dsx
                lap[1:, :, :] -= dsx
                dsy = (solid[:, 1:, :] - solid[:, :-1, :]) * gate_y
                lap[:, :-1, :] += dsy
                lap[:, 1:, :] -= dsy
                solid = solid + c_lat * lap
        film.solid = np.clip(solid, 0.0, 1.0)

    def _step(self, dt_budget: float) -> float:
        """Advance one adaptive explicit substep, returning the step taken."""
        head_w, head_y = self._heads()
        prepared = []
        dt = min(dt_budget, DT_MAX)
        for film, head in ((self.white, head_w), (self.yolk, head_y)):
            mob, flux_x, flux_y = self._fluxes(film, head)
            prepared.append((film, mob, flux_x, flux_y))
            mob_max = float(np.max(mob)) if mob.size else 0.0
            if mob_max > 0.0:
                dt = min(dt, CFL_DIFF * self.dx * self.dx / mob_max)
            h_safe = np.maximum(film.h, H_DRY)
            speed = max(
                float(np.max(np.abs(flux_x[1:-1, :]) / _face_avg_x(h_safe)[1:-1, :])),
                float(np.max(np.abs(flux_y[:, 1:-1]) / _face_avg_y(h_safe)[:, 1:-1])),
            )
            if speed > 1.0e-9:
                dt = min(dt, CFL_ADV * self.dx / speed)
        dt = float(max(min(dt, dt_budget), DT_MIN))

        for film, _mob, flux_x, flux_y in prepared:
            h_old = film.h
            dhdt = -_divergence(flux_x, flux_y, self.dx)
            h_new = np.maximum(h_old + dt * dhdt, 0.0)
            fill_t = np.full_like(film.temp, self.t_ambient)
            film.temp = self._advect_layers(
                film.temp, h_old, h_new, flux_x, flux_y, dt, fill_t
            )
            film.solid = np.clip(
                self._advect_layers(
                    film.solid, h_old, h_new, flux_x, flux_y, dt, film.solid
                ),
                0.0,
                1.0,
            )
            # Incompressibility: restore the exact frozen phase volume.
            total = float(np.sum(h_new)) * self.dx * self.dx
            if total > 1.0e-12:
                h_new *= film.volume / total
            film.h = h_new
            film.dhdt = dhdt
            h_safe = np.maximum(h_new, H_DRY)
            wet = h_new > H_DRY
            qx = 0.5 * (flux_x[:-1, :] + flux_x[1:, :])
            qy = 0.5 * (flux_y[:, :-1] + flux_y[:, 1:])
            film.ux = np.where(wet, qx / h_safe, 0.0)
            film.uy = np.where(wet, qy / h_safe, 0.0)

        for film in self.films:
            self._conduct(film, dt)
            self._coagulate(film, dt)

        self._advect_particles(dt)
        self.time += dt
        self.steps += 1
        self.substep_min_dt = min(self.substep_min_dt, dt)
        return dt

    def _advect_particles(self, dt: float) -> None:
        """Lagrangian tracers follow the depth-averaged material velocity."""
        for film in self.films:
            select = self.body_id == film.body_id
            xs = self.p_xy[select, 0]
            ys = self.p_xy[select, 1]
            ux = _bilinear(film.ux, xs, ys, self.origin, self.dx)
            uy = _bilinear(film.uy, xs, ys, self.origin, self.dx)
            self.p_xy[select, 0] = xs + dt * ux
            self.p_xy[select, 1] = ys + dt * uy
        limit = self.r_pan - 0.5 * self.dp
        radius = np.hypot(self.p_xy[:, 0], self.p_xy[:, 1])
        over = radius > limit
        if np.any(over):
            scale = limit / np.maximum(radius[over], 1.0e-12)
            self.p_xy[over, 0] *= scale
            self.p_xy[over, 1] *= scale


    # ------------------------------------------------------------- sampling
    def _sample_layered(self, field3: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                        zeta: np.ndarray) -> np.ndarray:
        """Trilinear sample of a [nx,ny,NZ] terrain-following field."""
        nx, ny = field3.shape[0], field3.shape[1]
        fx = (xs - self.origin) / self.dx
        fy = (ys - self.origin) / self.dx
        i0 = np.clip(np.floor(fx).astype(np.int64), 0, nx - 2)
        j0 = np.clip(np.floor(fy).astype(np.int64), 0, ny - 2)
        tx = np.clip(fx - i0, 0.0, 1.0)
        ty = np.clip(fy - j0, 0.0, 1.0)
        fz = zeta * NZ - 0.5
        k0 = np.clip(np.floor(fz).astype(np.int64), 0, NZ - 2)
        tz = np.clip(fz - k0, 0.0, 1.0)
        out = np.zeros(xs.shape, dtype=float)
        for di, wx in ((0, 1.0 - tx), (1, tx)):
            for dj, wy in ((0, 1.0 - ty), (1, ty)):
                low = field3[i0 + di, j0 + dj, k0]
                high = field3[i0 + di, j0 + dj, k0 + 1]
                out += wx * wy * ((1.0 - tz) * low + tz * high)
        return out

    def _sync_particles(self) -> None:
        """Place and label tracers from the current continuum film state."""
        for film in self.films:
            select = self.body_id == film.body_id
            xs = self.p_xy[select, 0]
            ys = self.p_xy[select, 1]
            zeta = self.zeta[select]
            h_at = np.maximum(_bilinear(film.h, xs, ys, self.origin, self.dx), 0.0)
            self.p_z0[select] = self.z_pan + zeta * h_at
            self.p_vel[select, 0] = _bilinear(film.ux, xs, ys, self.origin, self.dx)
            self.p_vel[select, 1] = _bilinear(film.uy, xs, ys, self.origin, self.dx)
            self.p_vel[select, 2] = zeta * _bilinear(
                film.dhdt, xs, ys, self.origin, self.dx
            )
            self.p_temp[select] = self._sample_layered(film.temp, xs, ys, zeta)
            self.p_solid[select] = self._sample_layered(film.solid, xs, ys, zeta)
        self.p_temp = np.clip(self.p_temp, 150.0, 1200.0)
        self.p_solid = np.clip(self.p_solid, 0.0, 1.0)

    def _cell_gradient(self, field: np.ndarray):
        gx = np.zeros_like(field)
        gy = np.zeros_like(field)
        gx[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * self.dx)
        gy[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / (2.0 * self.dx)
        return gx, gy

    def _interaction_forces(self) -> np.ndarray:
        """Paired internal contact and phase-coupling forces (Newton's third law)."""
        area = self.dx * self.dx
        head_w, head_y = self._heads()
        rows = []
        for film, head in ((self.white, head_w), (self.yolk, head_y)):
            gx, gy = self._cell_gradient(head)
            wet = film.h > H_DRY
            factor = film.rho * self.g * np.where(wet, film.h, 0.0) * area
            # Wall shear reaction balancing the lubrication pressure gradient.
            fx = float(np.sum(factor * gx))
            fy = float(np.sum(factor * gy))
            fz = film.rho * self.g * float(np.sum(np.where(wet, film.h, 0.0))) * area
            rows.append([fx, fy, fz])       # pan on phase
            rows.append([-fx, -fy, -fz])    # phase on pan
        # Hydrostatic cross-load between the two coexisting phases.
        gyx, gyy = self._cell_gradient(self.yolk.h)
        overlap = (self.white.h > H_DRY) & (self.yolk.h > H_DRY)
        load = LAMBDA_XLOAD * self.yolk.rho * self.g * np.where(
            overlap, self.yolk.h, 0.0
        ) * area
        cx = -float(np.sum(load * gyx))
        cy = -float(np.sum(load * gyy))
        cz = -float(np.sum(load))
        rows.append([cx, cy, cz])           # yolk on white
        rows.append([-cx, -cy, -cz])        # white on yolk
        forces = np.asarray(rows, dtype=float)
        return np.where(np.isfinite(forces), forces, 0.0)

    def _build_frame(self, time: float) -> StateFrame:
        self._sync_particles()
        position = np.column_stack((self.p_xy[:, 0], self.p_xy[:, 1], self.p_z0))
        velocity = np.array(self.p_vel, dtype=float)

        h_w = self.white.h
        h_y = self.yolk.h
        wet = (h_w + h_y) > H_DRY
        active_mask = (wet & self.inside).astype(float)

        m_w = self.white.rho * h_w
        m_y = self.yolk.rho * h_y
        m_tot = m_w + m_y
        m_safe = np.where(m_tot > 1.0e-12, m_tot, 1.0)
        t_w = np.mean(self.white.temp, axis=2)
        t_y = np.mean(self.yolk.temp, axis=2)
        transported = np.where(
            m_tot > 1.0e-12, (m_w * t_w + m_y * t_y) / m_safe, self.t_ambient
        )
        scalar_source = np.where(
            m_tot > 1.0e-12,
            (m_w * self.white.heat_rate + m_y * self.yolk.heat_rate) / m_safe,
            0.0,
        )
        velocity_field = np.zeros((self.nx, self.ny, 2), dtype=float)
        velocity_field[:, :, 0] = np.where(
            m_tot > 1.0e-12, (m_w * self.white.ux + m_y * self.yolk.ux) / m_safe, 0.0
        )
        velocity_field[:, :, 1] = np.where(
            m_tot > 1.0e-12, (m_w * self.white.uy + m_y * self.yolk.uy) / m_safe, 0.0
        )

        # Contact gap against the frozen pan base plane and rim wall.
        radial = np.hypot(self.p_xy[:, 0], self.p_xy[:, 1])
        half = 0.5 * self.dp
        gap_base = (self.p_z0 - self.z_pan) - half
        gap_rim = (self.r_pan - radial) - half
        contact_sdf = np.minimum(gap_base, gap_rim)

        mass_quantity = np.zeros((self.n_particles, 2), dtype=float)
        is_white = self.body_id == 1
        mass_quantity[is_white, 0] = self.p_mass[is_white]
        mass_quantity[~is_white, 1] = self.p_mass[~is_white]

        s_white = float(np.mean(self.p_solid[is_white])) if np.any(is_white) else 0.0
        s_yolk = float(np.mean(self.p_solid[~is_white])) if np.any(~is_white) else 0.0
        spread = float(np.max(radial[is_white])) if np.any(is_white) else 0.0

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=np.array(self.p_mass, dtype=float),
            body_id=np.array(self.body_id, dtype=np.int32),
            attributes={
                "mass_quantity": mass_quantity,
                "material_concentration": np.array(self.concentration, dtype=float),
                "material_label": np.array(self.material_label, dtype=float),
                "solid_phase_fraction": np.array(self.p_solid, dtype=float),
                "liquid_phase_fraction": 1.0 - np.array(self.p_solid, dtype=float),
                "temperature_k": np.array(self.p_temp, dtype=float),
                "pan_contact_distance": self.p_z0 - self.z_pan,
            },
            fields={
                "active_mask": active_mask,
                "contact_signed_distance": contact_sdf,
                "cumulative_mass_source": np.zeros(2, dtype=float),
                "interaction_force": self._interaction_forces(),
                "scalar_source": scalar_source,
                "transported_scalar": transported,
                "velocity_field": velocity_field,
                "film_thickness_white": h_w,
                "film_thickness_yolk": h_y,
            },
            scalars={
                "contact_length_scale": float(self.dp),
                "grid_spacing": float(self.dx),
                "pan_temperature_k": float(self.t_pan),
                "ambient_temperature_k": float(self.t_ambient),
                "total_egg_mass_kg": float(np.sum(self.p_mass)),
                "white_mass_kg": float(np.sum(mass_quantity[:, 0])),
                "yolk_mass_kg": float(np.sum(mass_quantity[:, 1])),
                "white_solid_fraction_mean": s_white,
                "yolk_solid_fraction_mean": s_yolk,
                "white_spread_radius_m": spread,
                "white_volume_m3": float(np.sum(h_w) * self.dx * self.dx),
                "yolk_volume_m3": float(np.sum(h_y) * self.dx * self.dx),
                "substeps": float(self.steps),
            },
        )
        frame.validate()
        return frame

    # --------------------------------------------------------------- driver
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.time < target - 1.0e-12 and guard < 4_000_000:
            self._step(target - self.time)
            guard += 1
        self.time = target
        return self._build_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "two-layer lubrication thin film + terrain-following conduction",
            "grid": [int(self.nx), int(self.ny), int(NZ)],
            "grid_spacing_m": float(self.dx),
            "substeps": int(self.steps),
            "min_substep_s": float(self.substep_min_dt),
            "particles": int(self.n_particles),
            "white_density_kg_m3": float(self.white.rho),
            "yolk_density_kg_m3": float(self.yolk.rho),
            "white_volume_m3": float(self.white.volume),
            "yolk_volume_m3": float(self.yolk.volume),
            "final_time_s": float(self.time),
            "phase_diffusion_enabled": bool(self.phase_diffusion),
            "concentration_diffusion_enabled": bool(self.conc_diffusion),
        }
