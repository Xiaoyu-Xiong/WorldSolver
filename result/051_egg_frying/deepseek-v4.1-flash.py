"""Solver Agent-owned implementation stub.

Implement ``BenchmarkSolver`` in this file.

The runner calls::

    solver.initialize(scene, seed)
    frames = [solver.advance_to(float(t)) for t in scene.sample_times()]

``advance_to(time)`` must return ``worldsolver.contracts.StateFrame`` with
finite state arrays and all trajectory fields requested in the Solver prompt.
The class must define ``initialize``, ``advance_to``, and ``diagnostics``.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

try:  # optional accelerator, a brute force fallback is provided below
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

# --------------------------------------------------------------------------
# frozen physical / geometric constants (mirrored from config + scene geometry)
# --------------------------------------------------------------------------
_GRAVITY = 9.81
_PAN_TOP = 0.018            # pan_contact_height_m
_R_NO_ESCAPE = 0.185        # pan_no_escape_radius_m
_RHO_WHITE = 1030.0
_RHO_YOLK = 1020.0
_NU_WHITE = 1.0e-5          # kinematic viscosity of raw egg white
_NU_YOLK = 1.0e-3           # yolk (with vitelline membrane) is far more viscous
_H_MIN = 1.0e-6
_K_EGG = 0.55               # thermal conductivity W/(m K)
_CP_RHO = 3.8e6             # volumetric heat capacity J/(m^3 K)
_Z_CONTACT = 3.0e-4         # pan/egg contact resistance length
_H_CONV = 15.0              # free surface convection coefficient W/(m^2 K)
_T_DENATURE = {1: 335.0, 2: 343.0}
_T_WIDTH = 5.0
_COOK_RATE = {1: 1.5, 2: 0.8}
_PHASE_DIFFUSION = {1: 0.6, 2: 0.25}
_N_NEIGHBOURS = 10
_DRAG_FLOOR = 2.0e-4
_CFL = 0.35
_DT_MAX = 1.0e-2
_DT_MIN = 1.0e-6

# diagnostic grid ------------------------------------------------------------
_DX = 0.005
_NGRID = 76
_ORIGIN = -0.19

# frozen particle sampling domains (scene_geometry.yaml) ---------------------
_WHITE_CENTER = (0.012, 0.006, 0.035)
_WHITE_RADII = (0.076, 0.062, 0.017)
_YOLK_CENTER = (-0.026, -0.003, 0.052)
_YOLK_RADII = (0.034, 0.031, 0.032)
_CONTACT_PER_MATERIAL = 48


def _dome(x, y, center, radii):
    q = ((x - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2
    return np.sqrt(np.maximum(0.0, 1.0 - q))


def _material_coordinate(points, center, radii):
    """Relative vertical material coordinate inside the frozen sampling blob."""
    q = ((points[:, 0] - center[0]) / radii[0]) ** 2 + (
        (points[:, 1] - center[1]) / radii[1]
    ) ** 2
    q = np.clip(q, 0.0, 1.0 - 1e-9)
    half = radii[2] * np.sqrt(1.0 - q)
    denom = 2.0 * half
    eta = np.where(
        denom > 1e-9,
        (points[:, 2] - (center[2] - half)) / np.maximum(denom, 1e-12),
        0.5,
    )
    return np.clip(eta, 0.05, 0.97)


def _brute_neighbours(points, k):
    n = len(points)
    kk = int(min(k, n - 1))
    idx = np.zeros((n, kk), dtype=np.int64)
    dist = np.zeros((n, kk))
    for start in range(0, n, 128):
        chunk = points[start : start + 128]
        d = np.linalg.norm(chunk[:, None, :] - points[None, :, :], axis=2)
        for i in range(len(chunk)):
            d[i, start + i] = np.inf
        part = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        rows = np.arange(len(chunk))[:, None]
        dd = d[rows, part]
        order = np.argsort(dd, axis=1)
        idx[start : start + len(chunk)] = part[rows, order]
        dist[start : start + len(chunk)] = dd[rows, order]
    return dist, idx


def _neighbour_lists(points, k):
    n = len(points)
    if n <= 1:
        return np.zeros((n, 1)), np.zeros((n, 1), dtype=np.int64)
    kk = int(min(k, n - 1))
    if cKDTree is not None:
        tree = cKDTree(points)
        dist, idx = tree.query(points, k=kk + 1)
        dist = np.atleast_2d(np.asarray(dist, dtype=float))[:, 1:]
        idx = np.atleast_2d(np.asarray(idx, dtype=np.int64))[:, 1:]
        return dist, idx
    return _brute_neighbours(points, kk)


class BenchmarkSolver:
    """Deterministic thin-film + Lagrangian thermal solver for the frying egg."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None)
        if not isinstance(cfg, dict):
            cfg = {}
        ic = dict(cfg.get("initial_conditions", {}) or {})
        bc = dict(cfg.get("boundary_conditions", {}) or {})
        self._pan_temperature = float(bc.get("pan_surface_temperature_k", 443.0))
        self._ambient_temperature = float(ic.get("ambient_temperature_k", 296.0))
        egg_temperature = float(ic.get("egg_temperature_k", 291.0))
        self._seed = int(seed)

        frame = scene.initial_state()
        self._sample_times = [float(v) for v in scene.sample_times()]

        position = np.array(frame.position, dtype=float)
        n = len(position)
        if frame.body_id is None:
            body = np.ones(n, dtype=np.int64)
        else:
            body = np.asarray(frame.body_id).astype(np.int64)
        if frame.mass is None:
            mass = np.full(n, float(ic.get("white_mass_kg", 0.034)) / max(n, 1))
        else:
            mass = np.array(frame.mass, dtype=float)

        self._position = position.copy()
        self._velocity = np.zeros_like(position)
        self._body = body
        self._mass = mass
        self._n = n
        self._is_white = body == 1
        self._is_yolk = body == 2
        white_idx = np.flatnonzero(self._is_white)
        yolk_idx = np.flatnonzero(self._is_yolk)
        self._white_idx = white_idx
        self._yolk_idx = yolk_idx

        mass_white = float(np.sum(mass[white_idx])) if white_idx.size else 0.0
        mass_yolk = float(np.sum(mass[yolk_idx])) if yolk_idx.size else 0.0
        self._mass_white_initial = mass_white
        self._mass_yolk_initial = mass_yolk

        density = np.where(self._is_yolk, _RHO_YOLK, _RHO_WHITE)
        self._volume = mass / density
        self._area = np.power(np.maximum(self._volume, 1e-18), 2.0 / 3.0)
        self._heat_capacity = _CP_RHO * self._volume

        # material coordinate inside each film
        eta = np.full(n, 0.5)
        if white_idx.size:
            eta[white_idx] = _material_coordinate(
                position[white_idx], _WHITE_CENTER, _WHITE_RADII
            )
        if yolk_idx.size:
            eta[yolk_idx] = _material_coordinate(position[yolk_idx], _YOLK_CENTER, _YOLK_RADII)
        self._eta = eta

        # ---------------- diagnostic grid ----------------
        coords = _ORIGIN + (np.arange(_NGRID) + 0.5) * _DX
        self._X, self._Y = np.meshgrid(coords, coords)
        radius = np.hypot(self._X, self._Y)
        self._wall = (radius <= _R_NO_ESCAPE).astype(float)
        faces = _ORIGIN + (np.arange(_NGRID) + 1.0) * _DX
        fx, fy = np.meshgrid(faces, coords)
        self._mask_fx = (np.hypot(fx, fy) <= _R_NO_ESCAPE).astype(float)
        fx2, fy2 = np.meshgrid(coords, faces)
        self._mask_fy = (np.hypot(fx2, fy2) <= _R_NO_ESCAPE).astype(float)

        white_shape = _dome(self._X, self._Y, _WHITE_CENTER[:2], _WHITE_RADII[:2])
        yolk_shape = _dome(self._X, self._Y, _YOLK_CENTER[:2], _YOLK_RADII[:2])
        # film thickness chosen so that integral(rho * h dA) equals the frozen mass
        area_white = float(np.sum(white_shape) * _DX * _DX)
        area_yolk = float(np.sum(yolk_shape) * _DX * _DX)
        h1 = (
            white_shape * (mass_white / _RHO_WHITE) / area_white
            if area_white > 0.0
            else np.zeros_like(white_shape)
        )
        h2 = (
            yolk_shape * (mass_yolk / _RHO_YOLK) / area_yolk
            if area_yolk > 0.0
            else np.zeros_like(yolk_shape)
        )
        self._h1 = h1 * self._wall
        self._h2 = h2 * self._wall
        self._u1 = np.zeros_like(self._h1)
        self._v1 = np.zeros_like(self._h1)
        self._u2 = np.zeros_like(self._h1)
        self._v2 = np.zeros_like(self._h1)

        # ---------------- particle fields ----------------
        self._temperature = np.full(n, egg_temperature)
        solid = np.zeros(n)
        self._solid = solid
        self._liquid = 1.0 - solid
        self._phase_rate = np.zeros(n)
        self._d_phase = np.where(self._is_yolk, _PHASE_DIFFUSION[2], _PHASE_DIFFUSION[1])
        self._denature = np.where(self._is_yolk, _T_DENATURE[2], _T_DENATURE[1])
        self._cook_rate = np.where(self._is_yolk, _COOK_RATE[2], _COOK_RATE[1])

        concentration = None
        if frame.attributes is not None:
            concentration = frame.attributes.get("material_concentration")
        if concentration is not None:
            concentration = np.array(concentration, dtype=float)
            if concentration.shape != (n, 2):
                concentration = None
        if concentration is None:
            concentration = np.zeros((n, 2))
            concentration[self._is_white, 0] = 1.0
            concentration[self._is_yolk, 1] = 1.0
        self._concentration = concentration

        z = self._initial_height(eta)
        self._position[:, 2] = z
        self._depth = np.maximum(z - _PAN_TOP, 0.0)

        # ---------------- neighbours ----------------
        dist, idx = _neighbour_lists(self._position, _N_NEIGHBOURS)
        self._neighbour = idx
        d_eff = np.maximum(dist, 1.0e-4)
        area_ij = 0.5 * (self._area[:, None] + self._area[idx])
        cond = _K_EGG * area_ij / (d_eff * self._heat_capacity[:, None])
        self._conductance = np.minimum(cond, 40.0)
        same = self._body[:, None] == self._body[idx]
        weights = np.where(same, 1.0 / (d_eff * d_eff), 0.0)
        weight_sum = np.sum(weights, axis=1, keepdims=True)
        self._phase_weight = np.where(
            weight_sum > 0.0, weights / np.maximum(weight_sum, 1e-30), 0.0
        )

        # ---------------- contact bookkeeping ----------------
        cw = np.argsort(eta[white_idx])[: min(_CONTACT_PER_MATERIAL, white_idx.size)]
        cy = np.argsort(eta[yolk_idx])[: min(_CONTACT_PER_MATERIAL, yolk_idx.size)]
        self._contact_white = white_idx[cw]
        self._contact_yolk = yolk_idx[cy]
        self._contact_length_scale = float(np.mean(dist)) if dist.size else _DX

        # ---------------- bookkeeping ----------------
        self._time = 0.0
        self._mass_source = np.zeros(2)
        self._max_temperature = float(np.max(self._temperature))
        self._ready = True

    # --------------------------------------------------------------- geometry
    def _initial_height(self, eta):
        h1 = self._gather(self._h1, self._position[:, 0], self._position[:, 1])
        h2 = self._gather(self._h2, self._position[:, 0], self._position[:, 1])
        z = np.where(
            self._is_white,
            _PAN_TOP + eta * h1,
            _PAN_TOP + h1 + eta * h2,
        )
        return np.maximum(z, _PAN_TOP)

    # ------------------------------------------------------------ grid helpers
    def _gather(self, grid, x, y):
        fx = (np.asarray(x, dtype=float) - _ORIGIN) / _DX - 0.5
        fy = (np.asarray(y, dtype=float) - _ORIGIN) / _DX - 0.5
        i0 = np.clip(np.floor(fx).astype(np.int64), 0, _NGRID - 2)
        j0 = np.clip(np.floor(fy).astype(np.int64), 0, _NGRID - 2)
        tx = np.clip(fx - i0, 0.0, 1.0)
        ty = np.clip(fy - j0, 0.0, 1.0)
        g00 = grid[j0, i0]
        g10 = grid[j0, i0 + 1]
        g01 = grid[j0 + 1, i0]
        g11 = grid[j0 + 1, i0 + 1]
        return (
            (1.0 - tx) * (1.0 - ty) * g00
            + tx * (1.0 - ty) * g10
            + (1.0 - tx) * ty * g01
            + tx * ty * g11
        )

    def _splat(self, values, weights, x, y):
        num = np.zeros((_NGRID, _NGRID))
        den = np.zeros((_NGRID, _NGRID))
        fx = (np.asarray(x, dtype=float) - _ORIGIN) / _DX - 0.5
        fy = (np.asarray(y, dtype=float) - _ORIGIN) / _DX - 0.5
        i0 = np.clip(np.floor(fx).astype(np.int64), 0, _NGRID - 2)
        j0 = np.clip(np.floor(fy).astype(np.int64), 0, _NGRID - 2)
        tx = np.clip(fx - i0, 0.0, 1.0)
        ty = np.clip(fy - j0, 0.0, 1.0)
        corners = (
            (i0, j0, (1.0 - tx) * (1.0 - ty)),
            (i0 + 1, j0, tx * (1.0 - ty)),
            (i0, j0 + 1, (1.0 - tx) * ty),
            (i0 + 1, j0 + 1, tx * ty),
        )
        for ii, jj, ww in corners:
            np.add.at(num, (jj, ii), values * weights * ww)
            np.add.at(den, (jj, ii), weights * ww)
        return num, den

    def _flux_divergence(self, velocity, height, axis, face_mask):
        if axis == 1:
            uf = 0.5 * (velocity[:, :-1] + velocity[:, 1:])
            hup = np.where(uf > 0.0, height[:, :-1], height[:, 1:])
            flux = uf * hup * face_mask
            padded = np.zeros((flux.shape[0], flux.shape[1] + 2))
            padded[:, 1:-1] = flux
            return -(padded[:, 1:] - padded[:, :-1]) / _DX
        uf = 0.5 * (velocity[:-1, :] + velocity[1:, :])
        hup = np.where(uf > 0.0, height[:-1, :], height[1:, :])
        flux = uf * hup * face_mask
        padded = np.zeros((flux.shape[0] + 2, flux.shape[1]))
        padded[1:-1, :] = flux
        return -(padded[1:, :] - padded[:-1, :]) / _DX

    # ---------------------------------------------------------------- stepping
    def _suggest_dt(self, target):
        remaining = target - self._time
        if remaining <= 0.0:
            return 0.0
        speed = 1.0e-6
        for field in (self._u1, self._v1, self._u2, self._v2):
            value = float(np.max(np.abs(field)))
            if value > speed:
                speed = value
        dt = _CFL * _DX / (2.0 * speed)
        dt = min(dt, _DT_MAX)
        dt = max(dt, _DT_MIN)
        return min(dt, remaining)

    def _step(self, dt: float) -> None:
        h1 = self._h1
        h2 = self._h2
        wet1 = h1 > _H_MIN
        wet2 = h2 > _H_MIN
        rho_ratio = _RHO_YOLK / _RHO_WHITE

        dh1x = np.gradient(h1, _DX, axis=1)
        dh1y = np.gradient(h1, _DX, axis=0)
        dh2x = np.gradient(h2, _DX, axis=1)
        dh2y = np.gradient(h2, _DX, axis=0)
        d1x = np.where(wet1, np.clip(-_GRAVITY * (dh1x + rho_ratio * dh2x), -50.0, 50.0), 0.0)
        d1y = np.where(wet1, np.clip(-_GRAVITY * (dh1y + rho_ratio * dh2y), -50.0, 50.0), 0.0)
        d2x = np.where(wet2, np.clip(-_GRAVITY * (dh1x + dh2x), -50.0, 50.0), 0.0)
        d2y = np.where(wet2, np.clip(-_GRAVITY * (dh1y + dh2y), -50.0, 50.0), 0.0)

        c1 = np.minimum(3.0 * _NU_WHITE / np.maximum(h1, _DRAG_FLOOR) ** 2, 1.0e5)
        c2 = np.minimum(3.0 * _NU_YOLK / np.maximum(h2, _DRAG_FLOOR) ** 2, 1.0e5)
        coupling = 0.25 * np.minimum(c1, c2) * wet1 * wet2

        a11 = 1.0 + dt * (c1 + coupling)
        a12 = -dt * coupling
        a21 = -dt * coupling
        a22 = 1.0 + dt * (c2 + coupling)
        det = np.maximum(a11 * a22 - a12 * a21, 1e-12)

        r1 = self._u1 + dt * d1x
        r2 = self._u2 + dt * d2x
        u1 = (a22 * r1 - a12 * r2) / det
        u2 = (a11 * r2 - a21 * r1) / det
        r1 = self._v1 + dt * d1y
        r2 = self._v2 + dt * d2y
        v1 = (a22 * r1 - a12 * r2) / det
        v2 = (a11 * r2 - a21 * r1) / det

        u1 = np.where(wet1, np.clip(u1, -2.0, 2.0), 0.0)
        v1 = np.where(wet1, np.clip(v1, -2.0, 2.0), 0.0)
        u2 = np.where(wet2, np.clip(u2, -2.0, 2.0), 0.0)
        v2 = np.where(wet2, np.clip(v2, -2.0, 2.0), 0.0)

        div1 = self._flux_divergence(u1, h1, 1, self._mask_fx) + self._flux_divergence(
            v1, h1, 0, self._mask_fy
        )
        div2 = self._flux_divergence(u2, h2, 1, self._mask_fx) + self._flux_divergence(
            v2, h2, 0, self._mask_fy
        )
        h1_new = h1 + dt * div1
        h2_new = h2 + dt * div2

        removed1 = float(np.sum(np.maximum(-h1_new, 0.0))) * _DX * _DX * _RHO_WHITE
        removed2 = float(np.sum(np.maximum(-h2_new, 0.0))) * _DX * _DX * _RHO_YOLK
        h1_new = np.maximum(h1_new, 0.0) * self._wall
        h2_new = np.maximum(h2_new, 0.0) * self._wall
        self._mass_source[0] -= removed1
        self._mass_source[1] -= removed2

        wet1n = h1_new > _H_MIN
        wet2n = h2_new > _H_MIN
        vel_div1 = np.where(
            wet1n, np.gradient(u1, _DX, axis=1) + np.gradient(v1, _DX, axis=0), 0.0
        )
        vel_div2 = np.where(
            wet2n, np.gradient(u2, _DX, axis=1) + np.gradient(v2, _DX, axis=0), 0.0
        )

        self._h1 = h1_new
        self._h2 = h2_new
        self._u1, self._v1, self._u2, self._v2 = u1, v1, u2, v2

        self._advect_particles(dt, u1, v1, u2, v2, vel_div1, vel_div2)
        self._thermal_step(dt)

    def _advect_particles(self, dt, u1, v1, u2, v2, vel_div1, vel_div2) -> None:
        position = self._position
        velocity = self._velocity
        x = position[:, 0].copy()
        y = position[:, 1].copy()
        white = self._is_white

        ux1 = self._gather(u1, x, y)
        vy1 = self._gather(v1, x, y)
        ux2 = self._gather(u2, x, y)
        vy2 = self._gather(v2, x, y)
        ux = np.where(white, ux1, ux2)
        vy = np.where(white, vy1, vy2)

        x = x + dt * ux
        y = y + dt * vy
        radius = np.hypot(x, y)
        outside = radius > _R_NO_ESCAPE
        if np.any(outside):
            scale = _R_NO_ESCAPE / np.maximum(radius[outside], 1e-30)
            x[outside] = x[outside] * scale
            y[outside] = y[outside] * scale
            nx = x[outside] / _R_NO_ESCAPE
            ny = y[outside] / _R_NO_ESCAPE
            radial = ux[outside] * nx + vy[outside] * ny
            push = radial > 0.0
            ux[outside] = np.where(push, ux[outside] - radial * nx, ux[outside])
            vy[outside] = np.where(push, vy[outside] - radial * ny, vy[outside])

        h1p = np.maximum(self._gather(self._h1, x, y), 0.0)
        h2p = np.maximum(self._gather(self._h2, x, y), 0.0)
        d1p = self._gather(vel_div1, x, y)
        d2p = self._gather(vel_div2, x, y)
        eta = self._eta

        z = np.where(white, _PAN_TOP + eta * h1p, _PAN_TOP + h1p + eta * h2p)
        z = np.maximum(z, _PAN_TOP)
        w = np.where(white, -(eta * h1p * d1p), -(h1p * d1p + eta * h2p * d2p))
        w = np.where(h1p > _H_MIN, w, 0.0)

        position[:, 0] = x
        position[:, 1] = y
        position[:, 2] = z
        velocity[:, 0] = ux
        velocity[:, 1] = vy
        velocity[:, 2] = w
        self._depth = np.maximum(z - _PAN_TOP, 0.0)

    def _thermal_step(self, dt: float) -> None:
        temperature = self._temperature
        neighbour = self._neighbour
        delta = temperature[neighbour] - temperature[:, None]
        rate = np.sum(self._conductance * delta, axis=1)
        pan_rate = _K_EGG * self._area / (self._heat_capacity * (self._depth + _Z_CONTACT))
        rate += pan_rate * (self._pan_temperature - temperature)
        surface_rate = (
            _H_CONV * self._area / self._heat_capacity * np.clip(self._eta, 0.0, 1.0) ** 3
        )
        rate += surface_rate * (self._ambient_temperature - temperature)
        temperature = temperature + dt * rate
        np.clip(temperature, 1.0, 3000.0, out=temperature)
        self._temperature = temperature
        self._max_temperature = float(np.max(temperature))

        activation = 1.0 / (
            1.0 + np.exp(-np.clip((temperature - self._denature) / _T_WIDTH, -60.0, 60.0))
        )
        cook = self._cook_rate * (1.0 - self._solid) * activation
        phase_delta = np.sum(
            self._phase_weight * (self._solid[neighbour] - self._solid[:, None]), axis=1
        )
        phase = self._d_phase * phase_delta
        solid = np.clip(self._solid + dt * (cook + phase), 0.0, 1.0)
        self._solid = solid
        self._liquid = 1.0 - solid
        self._phase_rate = cook + phase

    # ------------------------------------------------------------------ output
    def advance_to(self, time: float):
        if not getattr(self, "_ready", False):
            raise RuntimeError("initialize() must be called before advance_to()")
        target = float(time)
        if target < self._time - 1e-12:
            raise ValueError("advance_to() requires non-decreasing sample times")
        while self._time < target - 1e-12:
            remaining = target - self._time
            if remaining <= 1e-12:
                break
            dt = self._suggest_dt(target)
            if dt <= 0.0:
                break
            self._step(dt)
            self._time = min(self._time + dt, target)
        self._time = target
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        position = np.nan_to_num(self._position, nan=0.0, posinf=0.0, neginf=0.0)
        velocity = np.nan_to_num(self._velocity, nan=0.0, posinf=0.0, neginf=0.0)
        solid = np.clip(np.nan_to_num(self._solid, nan=0.0), 0.0, 1.0)
        liquid = np.clip(np.nan_to_num(self._liquid, nan=1.0), 0.0, 1.0)
        temperature = np.nan_to_num(self._temperature, nan=float(self._ambient_temperature))
        concentration = np.nan_to_num(self._concentration, nan=0.0)

        mass_quantity = np.zeros((self._n, 2))
        mass_quantity[self._white_idx, 0] = self._mass[self._white_idx]
        mass_quantity[self._yolk_idx, 1] = self._mass[self._yolk_idx]

        white = self._is_white
        num, den = self._splat(
            solid[white], self._mass[white], position[white, 0], position[white, 1]
        )
        scalar_grid = np.where(den > 0.0, num / np.maximum(den, 1e-30), 0.0)
        num_s, den_s = self._splat(
            self._phase_rate[white], self._mass[white], position[white, 0], position[white, 1]
        )
        source_grid = np.where(den_s > 0.0, num_s / np.maximum(den_s, 1e-30), 0.0)

        active = (self._h1 > _H_MIN).astype(float)
        velocity_field = np.stack(
            [np.nan_to_num(self._u1, nan=0.0), np.nan_to_num(self._v1, nan=0.0)], axis=-1
        )

        h1p = np.maximum(
            self._gather(
                self._h1, position[self._contact_yolk, 0], position[self._contact_yolk, 1]
            ),
            0.0,
        )
        gap_white = np.maximum(position[self._contact_white, 2] - _PAN_TOP, 0.0)
        gap_yolk = np.maximum(position[self._contact_yolk, 2] - (_PAN_TOP + h1p), 0.0)
        contact = np.concatenate([gap_white, gap_yolk])
        if contact.size == 0:
            contact = np.zeros(1)

        c1 = np.minimum(3.0 * _NU_WHITE / np.maximum(self._h1, _DRAG_FLOOR) ** 2, 1.0e5)
        c2 = np.minimum(3.0 * _NU_YOLK / np.maximum(self._h2, _DRAG_FLOOR) ** 2, 1.0e5)
        cell = _DX * _DX
        weight_white = float(np.sum(self._mass[self._white_idx])) * _GRAVITY
        weight_yolk = float(np.sum(self._mass[self._yolk_idx])) * _GRAVITY
        shear_white = -_RHO_WHITE * cell * np.array(
            [
                float(np.sum(self._h1 * c1 * self._u1)),
                float(np.sum(self._h1 * c1 * self._v1)),
            ]
        )
        shear_yolk = -_RHO_YOLK * cell * np.array(
            [
                float(np.sum(self._h2 * c2 * self._u2)),
                float(np.sum(self._h2 * c2 * self._v2)),
            ]
        )
        interaction = np.array(
            [
                [shear_white[0], shear_white[1], weight_white],
                [shear_yolk[0], shear_yolk[1], weight_yolk],
            ]
        )

        egg_mass = float(np.sum(self._mass))
        if self._white_idx.size:
            radius_white = float(
                np.max(np.hypot(position[self._white_idx, 0], position[self._white_idx, 1]))
            )
        else:
            radius_white = 0.0
        if self._yolk_idx.size:
            radius_yolk = float(
                np.max(np.hypot(position[self._yolk_idx, 0], position[self._yolk_idx, 1]))
            )
        else:
            radius_yolk = 0.0

        attributes = {
            "mass_quantity": mass_quantity,
            "material_concentration": concentration,
            "material_label": self._body.astype(float),
            "solid_phase_fraction": solid,
            "liquid_phase_fraction": liquid,
            "temperature_k": temperature,
            "pan_contact_distance": np.maximum(position[:, 2] - _PAN_TOP, 0.0),
        }
        fields = {
            "active_mask": np.nan_to_num(active, nan=0.0),
            "contact_signed_distance": np.nan_to_num(contact, nan=0.0),
            "cumulative_mass_source": self._mass_source.copy(),
            "interaction_force": interaction,
            "scalar_source": np.nan_to_num(source_grid, nan=0.0),
            "transported_scalar": np.nan_to_num(scalar_grid, nan=0.0),
            "velocity_field": velocity_field,
        }
        scalars = {
            "contact_length_scale": float(self._contact_length_scale),
            "grid_spacing": float(_DX),
            "total_egg_mass_kg": egg_mass,
            "white_mass_kg": float(np.sum(self._mass[self._white_idx]))
            if self._white_idx.size
            else 0.0,
            "yolk_mass_kg": float(np.sum(self._mass[self._yolk_idx]))
            if self._yolk_idx.size
            else 0.0,
            "white_spread_radius_m": radius_white,
            "yolk_spread_radius_m": radius_yolk,
            "mean_temperature_k": float(np.mean(temperature)),
            "max_temperature_k": float(np.max(temperature)),
            "mean_solid_fraction": float(np.mean(solid)),
            "max_solid_fraction": float(np.max(solid)),
            "pan_temperature_k": float(self._pan_temperature),
            "pan_no_escape_radius_m": float(_R_NO_ESCAPE),
        }
        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=self._mass.copy(),
            body_id=self._body.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        position = self._position
        radius = np.hypot(position[:, 0], position[:, 1])
        white_radius = (
            float(np.max(radius[self._white_idx])) if self._white_idx.size else 0.0
        )
        return {
            "time": float(self._time),
            "particle_count": int(self._n),
            "white_particle_count": int(self._white_idx.size),
            "yolk_particle_count": int(self._yolk_idx.size),
            "white_mass_kg": float(np.sum(self._mass[self._white_idx]))
            if self._white_idx.size
            else 0.0,
            "yolk_mass_kg": float(np.sum(self._mass[self._yolk_idx]))
            if self._yolk_idx.size
            else 0.0,
            "numerical_mass_removed_kg": [float(v) for v in self._mass_source],
            "white_spread_radius_m": white_radius,
            "max_temperature_k": float(np.max(self._temperature)),
            "mean_solid_fraction": float(np.mean(self._solid)),
            "grid_spacing_m": float(_DX),
        }
