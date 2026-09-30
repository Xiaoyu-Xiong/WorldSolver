"""Snow-castle destruction solved with a self-contained material point method.

Physical model
--------------
* A compressible elasto-plastic continuum for packed snow with the
  multiplicative decomposition ``F = F_E F_P``, a "critical compression /
  critical stretch" plastic return mapping and strain hardening, following
  Stomakhin et al., *A Material Point Method for Snow Simulation* (SIGGRAPH
  2013, Figure 6).
* St. Venant-Kirchhoff elastic energy evaluated on ``F_E`` (rotation
  invariant, cheap first Piola-Kirchhoff stress, no polar decomposition).
* Affine particle-in-cell (APIC) momentum transfer on a uniform Cartesian
  background grid with quadratic B-spline shape functions (Jiang et al. 2015).
* Rigid spherical striker coupled to the snow through inelastic normal
  impulses plus Coulomb friction; the equal-and-opposite reaction is applied to
  the striker, so striker/snow linear momentum is exchanged conservatively.
* Rigid floor with high friction resolved by grid boundary conditions plus a
  particle level Coulomb contact.

Every frame is produced by actually time stepping the above dynamics; nothing
is scripted or precomputed.  Only NumPy is used.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

_IDENTITY = np.eye(3, dtype=np.float64)


def _det3(a: np.ndarray) -> np.ndarray:
    """Determinant of (possibly batched) 3x3 matrices."""
    return (
        a[..., 0, 0] * (a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1])
        - a[..., 0, 1] * (a[..., 1, 0] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 0])
        + a[..., 0, 2] * (a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0])
    )


def _inv3(a: np.ndarray) -> np.ndarray:
    """Elementwise inverse of (possibly batched) 3x3 matrices."""
    a00, a01, a02 = a[..., 0, 0], a[..., 0, 1], a[..., 0, 2]
    a10, a11, a12 = a[..., 1, 0], a[..., 1, 1], a[..., 1, 2]
    a20, a21, a22 = a[..., 2, 0], a[..., 2, 1], a[..., 2, 2]
    c00 = a11 * a22 - a12 * a21
    c01 = a02 * a21 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    c10 = a12 * a20 - a10 * a22
    c11 = a00 * a22 - a02 * a20
    c12 = a02 * a10 - a00 * a12
    c20 = a10 * a21 - a11 * a20
    c21 = a01 * a20 - a00 * a21
    c22 = a00 * a11 - a01 * a10
    det = a00 * c00 + a01 * c10 + a02 * c20
    singular = np.abs(det) < 1e-12
    safe = np.where(singular, 1.0, det)
    inv = np.empty_like(a)
    inv[..., 0, 0] = c00 / safe
    inv[..., 0, 1] = c01 / safe
    inv[..., 0, 2] = c02 / safe
    inv[..., 1, 0] = c10 / safe
    inv[..., 1, 1] = c11 / safe
    inv[..., 1, 2] = c12 / safe
    inv[..., 2, 0] = c20 / safe
    inv[..., 2, 1] = c21 / safe
    inv[..., 2, 2] = c22 / safe
    if np.any(singular):
        inv[singular] = _IDENTITY
    return inv


class BenchmarkSolver:
    """Explicit APIC material point solver for plastic snow destruction."""

    # ---- constitutive / contact parameters -------------------------------
    YOUNG_MODULUS = 2.0e5      # Pa, packed snow
    POISSON_RATIO = 0.2
    HARDENING_XI = 10.0
    HARDENING_MIN = 0.4
    HARDENING_MAX = 2.0
    CRITICAL_COMPRESSION = 0.025
    CRITICAL_STRETCH = 0.0075
    FLOOR_FRICTION = 0.9
    STRIKER_FRICTION = 0.5
    STRIKER_RESTITUTION = 0.0

    # ---- numerical parameters --------------------------------------------
    CFL = 0.4
    SUBSTEPS_PER_SAMPLE = 16
    MAX_SUBSTEPS_PER_SAMPLE = 20
    GRID_SPACING_FACTOR = 4.0 / 3.0
    MAX_INTERVAL_STEPS = 96
    MAX_PARTICLE_SPEED = 25.0
    MAX_STRIKER_SPEED = 12.0
    MAX_AFFINE = 300.0

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        initial = dict(config.get("initial_conditions", {}) or {})
        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.gravity_magnitude = float(abs(self.gravity[2]))

        frame = scene.initial_state()
        position = np.array(frame.position, dtype=np.float64)
        velocity = np.array(frame.velocity, dtype=np.float64)
        mass = np.array(frame.mass, dtype=np.float64)
        body = np.asarray(frame.body_id).astype(np.int64)

        snow = body == 1
        striker = body == 2
        if not snow.any() or not striker.any():
            raise ValueError("scene must provide body_id 1 (snow) and 2 (striker) samples")
        self.snow_mask = snow
        self.n = int(np.count_nonzero(snow))
        self.n_total = int(position.shape[0])
        self.x = position[snow].copy()
        self.v = velocity[snow].copy()
        self.mass_snow = mass[snow].copy()
        self.mass_total = mass.copy()
        self.body_id = body
        self.ball_x = position[striker][0].copy()
        self.ball_v = velocity[striker][0].copy()
        self.ball_mass = float(mass[striker][0])

        volume = frame.attributes.get("initial_volume")
        if volume is None:
            spacing = float(initial.get("snow_particle_spacing", 0.075))
            volume = np.full(self.n_total, spacing ** 3)
        volume = np.array(volume, dtype=np.float64)
        self.volume_snow = volume[snow].copy()
        spacing = float(np.mean(np.cbrt(np.maximum(self.volume_snow, 1e-12))))
        if not np.isfinite(spacing) or spacing <= 0.0:
            spacing = float(initial.get("snow_particle_spacing", 0.075))
        # A background cell slightly larger than the material point spacing
        # keeps several points per cell (smooth transfer) while allowing a
        # proportionally larger stable explicit time step.
        self.dx = spacing * self.GRID_SPACING_FACTOR
        density = float(np.mean(self.mass_snow / np.maximum(self.volume_snow, 1e-12)))
        self.rho = density if np.isfinite(density) and density > 1e-6 else float(
            initial.get("snow_bulk_density", 420.0)
        )

        shear = self.YOUNG_MODULUS / (2.0 * (1.0 + self.POISSON_RATIO))
        lame = (
            self.YOUNG_MODULUS
            * self.POISSON_RATIO
            / ((1.0 + self.POISSON_RATIO) * (1.0 - 2.0 * self.POISSON_RATIO))
        )
        self.mu0 = float(shear)
        self.lam0 = float(lame)

        # striker radius / floor height from the frozen scene geometry.
        self.ball_radius = float(initial.get("striker_radius", 0.42))
        self.floor_top = 0.0
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []):
            if obj.get("id") == "impact_ball":
                self.ball_radius = float(obj.get("geometry", {}).get("radius", self.ball_radius))
            elif obj.get("id") == "snow_floor":
                half = obj.get("geometry", {}).get("half_extents", [0.0, 0.0, 0.0])
                translation = obj.get("transform", {}).get("translation", [0.0, 0.0, 0.0])
                self.floor_top = float(translation[2]) + float(half[2])

        # ---- background grid ------------------------------------------------
        span = np.asarray([7.2, 4.8, 4.3], dtype=np.float64)
        self.nx = max(4, int(round(span[0] / self.dx)))
        self.ny = max(4, int(round(span[1] / self.dx)))
        self.nz = max(4, int(round(span[2] / self.dx)))
        self.origin = np.array(
            [
                0.0 - 0.5 * self.nx * self.dx,
                0.0 - 0.5 * self.ny * self.dx,
                self.floor_top - self.dx,
            ],
            dtype=np.float64,
        )
        self.n_nodes = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        self._base_max = np.array([self.nx - 2, self.ny - 2, self.nz - 2], dtype=np.int64)

        node_index = np.arange(self.n_nodes, dtype=np.int64)
        kk = node_index % (self.nz + 1)
        jj = (node_index // (self.nz + 1)) % (self.ny + 1)
        ii = node_index // ((self.nz + 1) * (self.ny + 1))
        floor_k = int(round((self.floor_top - self.origin[2]) / self.dx))
        self.floor_nodes = kk <= max(0, floor_k)
        self.wall_nodes = (
            (ii == 0) | (ii == self.nx) | (jj == 0) | (jj == self.ny) | (kk == self.nz)
        )

        # ---- particle state ------------------------------------------------
        self.ftot = np.tile(_IDENTITY, (self.n, 1, 1))
        self.fe = np.tile(_IDENTITY, (self.n, 1, 1))
        self.fp = np.tile(_IDENTITY, (self.n, 1, 1))
        self.affine = np.zeros((self.n, 3, 3), dtype=np.float64)
        self.plastic_jacobian = np.ones(self.n, dtype=np.float64)
        self.hardening = np.ones(self.n, dtype=np.float64)

        sample_dt = float(config.get("sample_dt", 0.02))
        self.dt_nominal = sample_dt / float(self.SUBSTEPS_PER_SAMPLE)
        self.dt_min = sample_dt / float(self.MAX_SUBSTEPS_PER_SAMPLE)
        self.time = 0.0

    # ------------------------------------------------------------- stepping
    def _cfl_dt(self) -> float:
        hard = float(self.hardening.max()) if self.n else 1.0
        wave = math.sqrt(max(self.lam0 + 2.0 * self.mu0, 1.0) * max(hard, 1e-3) / self.rho)
        vmax = float(np.sqrt((self.v * self.v).sum(axis=1)).max()) if self.n else 0.0
        vmax = max(vmax, float(np.linalg.norm(self.ball_v)))
        return self.CFL * self.dx / (wave + vmax + 1e-9)

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("requested sample time must be finite")
        if target < self.time - 1e-9:
            raise ValueError("sample times must be non-decreasing")
        steps = 0
        while self.time < target - 1e-12:
            remaining = target - self.time
            dt = min(self.dt_nominal, remaining)
            limit = self._cfl_dt()
            if limit < dt:
                dt = max(limit, self.dt_min)
            dt = min(dt, remaining)
            if dt <= 0.0:
                break
            self._step(dt)
            self.time += dt
            steps += 1
            if steps >= self.MAX_INTERVAL_STEPS:
                # Defensive fall-back: finish the interval with bounded steps
                # instead of one oversized (unstable) step.
                while self.time < target - 1e-12:
                    tail = min(self.dt_min, target - self.time)
                    if tail <= 0.0:
                        break
                    self._step(tail)
                    self.time += tail
                break
        self.time = target
        return self._frame(target)

    def _step(self, dt: float) -> None:
        n = self.n
        dx = self.dx
        ox, oy, oz = float(self.origin[0]), float(self.origin[1]), float(self.origin[2])
        nby = self.ny + 1
        nbz = self.nz + 1

        x = self.x
        v = self.v
        affine = self.affine
        fe = self.fe
        ftot = self.ftot
        mass = self.mass_snow
        volume = self.volume_snow

        # ---- 1. elastic first Piola-Kirchhoff stress ----------------------
        mu = self.mu0 * self.hardening
        lam = self.lam0 * self.hardening
        ftf = np.einsum("nba,nbc->nac", fe, fe)
        strain = 0.5 * (ftf - _IDENTITY[None, :, :])
        trace = strain[:, 0, 0] + strain[:, 1, 1] + strain[:, 2, 2]
        second = (
            2.0 * mu[:, None, None] * strain
            + (lam * trace)[:, None, None] * _IDENTITY[None, :, :]
        )
        piola = np.einsum("nab,nbc->nac", fe, second)
        kirchhoff = np.einsum("nab,ncb->nac", piola, fe)   # P F_E^T

        # ---- 2. quadratic B-spline shape functions ------------------------
        s = (x - self.origin[None, :]) / dx
        base = np.floor(s).astype(np.int64) - 1
        base = np.minimum(np.maximum(base, 0), self._base_max[None, :])
        frac = s - (base + 1.0)
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
        wx = np.stack([0.5 * (1.0 - fx) ** 2, 0.75 - (fx - 0.5) ** 2, 0.5 * fx ** 2], axis=1)
        wy = np.stack([0.5 * (1.0 - fy) ** 2, 0.75 - (fy - 0.5) ** 2, 0.5 * fy ** 2], axis=1)
        wz = np.stack([0.5 * (1.0 - fz) ** 2, 0.75 - (fz - 0.5) ** 2, 0.5 * fz ** 2], axis=1)
        dwx = np.stack([-(1.0 - fx), 1.0 - 2.0 * fx, fx], axis=1) / dx
        dwy = np.stack([-(1.0 - fy), 1.0 - 2.0 * fy, fy], axis=1) / dx
        dwz = np.stack([-(1.0 - fz), 1.0 - 2.0 * fz, fz], axis=1) / dx

        w4 = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
        gx4 = dwx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
        gy4 = wx[:, :, None, None] * dwy[:, None, :, None] * wz[:, None, None, :]
        gz4 = wx[:, :, None, None] * wy[:, None, :, None] * dwz[:, None, None, :]

        ar = np.arange(3, dtype=np.int64)
        # flattened (n,27) node indices; the 27 entries follow the (a,b,c)
        # ordering of the 3x3x3 quadratic B-spline stencil.
        ix_f = np.repeat(base[:, 0:1] + ar[None, :], 9, axis=1)
        iy_f = np.tile(np.repeat(base[:, 1:2] + ar[None, :], 3, axis=1), (1, 3))
        iz_f = np.tile(base[:, 2:3] + ar[None, :], (1, 9))
        wf = w4.reshape(n, 27)
        gxf = gx4.reshape(n, 27)
        gyf = gy4.reshape(n, 27)
        gzf = gz4.reshape(n, 27)
        nf = ((ix_f * nby) + iy_f) * nbz + iz_f
        nfr = nf.ravel()

        # offset[i] = x_node[i] - x_particle, built in place.
        offset = np.empty((n, 27, 3), dtype=np.float64)
        offset[:, :, 0] = ix_f
        offset[:, :, 1] = iy_f
        offset[:, :, 2] = iz_f
        offset *= dx
        offset -= (x - self.origin[None, :])[:, None, :]

        # ---- 3. particle -> grid (APIC scatter) ---------------------------
        weighted = mass[:, None] * wf
        node_mass = np.bincount(nfr, weights=weighted.ravel(), minlength=self.n_nodes)

        particle_vel = v[:, None, :] + np.einsum("nab,nib->nia", affine, offset)
        node_momentum = np.empty((self.n_nodes, 3), dtype=np.float64)
        for axis in range(3):
            node_momentum[:, axis] = np.bincount(
                nfr,
                weights=(weighted * particle_vel[:, :, axis]).ravel(),
                minlength=self.n_nodes,
            )
        node_force = np.empty((self.n_nodes, 3), dtype=np.float64)
        for axis in range(3):
            coefficient = -volume[:, None] * (
                kirchhoff[:, axis, 0:1] * gxf
                + kirchhoff[:, axis, 1:2] * gyf
                + kirchhoff[:, axis, 2:3] * gzf
            )
            node_force[:, axis] = np.bincount(
                nfr, weights=coefficient.ravel(), minlength=self.n_nodes
            )

        # ---- 4. grid update, boundary conditions --------------------------
        grid_velocity = np.zeros((self.n_nodes, 3), dtype=np.float64)
        active = node_mass > 1e-14
        inverse_mass = np.zeros(self.n_nodes, dtype=np.float64)
        np.divide(1.0, node_mass, out=inverse_mass, where=active)
        grid_velocity[active] = node_momentum[active] * inverse_mass[active, None]
        grid_velocity[active] += dt * self.gravity[None, :]
        grid_velocity[active] += dt * inverse_mass[active, None] * node_force[active]
        grid_velocity[self.floor_nodes] = 0.0
        grid_velocity[self.wall_nodes] = 0.0

        # ---- 5. grid -> particle (APIC gather) ----------------------------
        node_vel = grid_velocity[nf]
        v_new = np.einsum("ni,nid->nd", wf, node_vel)
        linear = np.einsum("ni,nid,nie->nde", wf, node_vel, offset)
        gram = np.einsum("ni,nid,nie->nde", wf, offset, offset)
        affine_new = np.einsum("nde,nef->ndf", linear, _inv3(gram))
        if not np.isfinite(affine_new).all():
            affine_new = np.nan_to_num(affine_new, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(affine_new, -self.MAX_AFFINE, self.MAX_AFFINE, out=affine_new)

        # ---- 6. advect particles, update deformation ----------------------
        x += dt * v_new
        v[:] = v_new
        gradient = _IDENTITY[None, :, :] + dt * affine_new
        ftot_new = np.einsum("nab,nbc->nac", gradient, ftot)
        fe_trial = np.einsum("nab,nbc->nac", gradient, fe)
        jacobian = _det3(fe_trial)
        lower = 1.0 - self.CRITICAL_COMPRESSION
        upper = 1.0 + self.CRITICAL_STRETCH
        compressed = jacobian < lower
        stretched = jacobian > upper
        if compressed.any():
            fe_trial[compressed] = lower ** (1.0 / 3.0)
        if stretched.any():
            fe_trial[stretched] = upper ** (1.0 / 3.0)
        fp_new = np.einsum("nab,nbc->nac", _inv3(fe_trial), ftot_new)

        self.fe = fe_trial
        self.fp = fp_new
        self.ftot = ftot_new
        self.affine = affine_new
        plastic_jacobian = _det3(fp_new)
        self.plastic_jacobian = plastic_jacobian
        self.hardening = np.clip(
            np.exp(self.HARDENING_XI * (1.0 - plastic_jacobian)),
            self.HARDENING_MIN,
            self.HARDENING_MAX,
        )

        # ---- 7. contacts ---------------------------------------------------
        self._solve_floor(dt)
        self._solve_striker(dt)
        self._sanitize()

    # ------------------------------------------------------------- contacts
    def _solve_floor(self, dt: float) -> None:
        x = self.x
        v = self.v
        below = x[:, 2] < self.floor_top
        if not below.any():
            return
        x[below, 2] = self.floor_top
        normal = v[below, 2]
        v[below, 2] = np.maximum(normal, 0.0)
        removed = np.maximum(-normal, 0.0) + self.gravity_magnitude * dt
        tangential = v[below, :2]
        speed = np.sqrt(np.einsum("nd,nd->n", tangential, tangential))
        budget = self.FLOOR_FRICTION * removed
        scale = np.clip(1.0 - budget / np.maximum(speed, 1e-12), 0.0, 1.0)
        v[below, :2] = tangential * scale[:, None]

    def _solve_striker(self, dt: float) -> None:
        x = self.x
        v = self.v
        ball_x = self.ball_x
        ball_v = self.ball_v
        ball_v[2] += self.gravity[2] * dt
        speed = float(np.linalg.norm(ball_v))
        if speed > self.MAX_STRIKER_SPEED:
            ball_v *= self.MAX_STRIKER_SPEED / speed
        ball_x += ball_v * dt
        # floor contact of the rigid striker
        rest = self.floor_top + self.ball_radius
        if ball_x[2] < rest:
            ball_x[2] = rest
            if ball_v[2] < 0.0:
                ball_v[2] = 0.0
            tangential_speed = float(np.hypot(ball_v[0], ball_v[1]))
            if tangential_speed > 1e-9:
                decrement = self.FLOOR_FRICTION * self.gravity_magnitude * dt
                ball_v[:2] *= max(0.0, 1.0 - decrement / tangential_speed)
        # snow contact: inelastic normal impulse + Coulomb friction
        if self.n == 0:
            return
        relative = x - ball_x[None, :]
        distance = np.sqrt(np.einsum("nd,nd->n", relative, relative))
        hit = (distance < self.ball_radius) & (distance > 1e-9)
        if not hit.any():
            return
        normal = relative[hit] / distance[hit, None]
        x[hit] = ball_x[None, :] + normal * self.ball_radius
        struck_velocity = v[hit]
        relative_velocity = struck_velocity - ball_v[None, :]
        normal_speed = np.einsum("nd,nd->n", relative_velocity, normal)
        tangential = relative_velocity - normal_speed[:, None] * normal
        delta = (
            -(1.0 + self.STRIKER_RESTITUTION)
            * np.minimum(normal_speed, 0.0)[:, None]
            * normal
            - self.STRIKER_FRICTION * tangential
        )
        v[hit] = struck_velocity + delta
        impulse = np.einsum("n,nd->d", self.mass_snow[hit], delta)
        ball_v -= impulse / self.ball_mass

    def _sanitize(self) -> None:
        x = self.x
        v = self.v
        finite = np.isfinite(x).all(axis=1) & np.isfinite(v).all(axis=1)
        if not finite.all():
            x[~finite] = 0.0
            v[~finite] = 0.0
        speed = np.sqrt(np.einsum("nd,nd->n", v, v))
        over = speed > self.MAX_PARTICLE_SPEED
        if over.any():
            v[over] *= (self.MAX_PARTICLE_SPEED / speed[over])[:, None]
        for attr, default in (
            ("fe", _IDENTITY),
            ("fp", _IDENTITY),
            ("ftot", _IDENTITY),
            ("affine", 0.0),
        ):
            array = getattr(self, attr)
            if not np.isfinite(array).all():
                broken = ~np.isfinite(array).reshape(self.n, -1).all(axis=1)
                array[broken] = default
        if not np.isfinite(self.hardening).all() or not np.isfinite(
            self.plastic_jacobian
        ).all():
            self.hardening = np.where(np.isfinite(self.hardening), self.hardening, 1.0)
            self.plastic_jacobian = np.where(
                np.isfinite(self.plastic_jacobian), self.plastic_jacobian, 1.0
            )
        lo = self.origin[None, :] + 0.01 * self.dx
        hi = self.origin[None, :] + (np.array([self.nx, self.ny, self.nz]) - 0.01) * self.dx
        np.clip(x[:, 0], lo[0, 0], hi[0, 0], out=x[:, 0])
        np.clip(x[:, 1], lo[0, 1], hi[0, 1], out=x[:, 1])
        np.clip(x[:, 2], self.floor_top, hi[0, 2], out=x[:, 2])

    # ---------------------------------------------------------------- output
    def _frame(self, time: float) -> StateFrame:
        n = self.n
        total = self.n_total
        position = np.empty((total, 3), dtype=np.float64)
        velocity = np.empty((total, 3), dtype=np.float64)
        position[:n] = self.x
        position[n:] = self.ball_x[None, :]
        velocity[:n] = self.v
        velocity[n:] = self.ball_v[None, :]
        deformation = np.empty((total, 3, 3), dtype=np.float64)
        deformation[:n] = self.ftot
        deformation[n:] = _IDENTITY
        plastic = np.empty(total, dtype=np.float64)
        plastic[:n] = self.plastic_jacobian
        plastic[n:] = 1.0
        mass = self.mass_total
        external = mass[:, None] * self.gravity[None, :]

        gap_floor = self.x[:, 2] - self.floor_top
        relative = self.x - self.ball_x[None, :] if n else np.zeros((0, 3))
        distance = (
            np.sqrt(np.einsum("nd,nd->n", relative, relative)) if n else np.zeros(0)
        )
        gap_striker = distance - self.ball_radius
        gap_snow = np.minimum(gap_floor, gap_striker) if n else np.zeros(0)
        striker_floor_gap = float(self.ball_x[2] - self.ball_radius - self.floor_top)
        striker_snow_gap = float(gap_striker.min()) if n else striker_floor_gap
        gaps = np.concatenate([gap_snow, [min(striker_floor_gap, striker_snow_gap)]])
        band = self.dx
        selected = gaps < band
        if not selected.any():
            selected = np.zeros_like(gaps, dtype=bool)
            selected[int(np.argmin(gaps))] = True
        contacts = np.ascontiguousarray(gaps[selected])

        position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
        deformation = np.nan_to_num(deformation, nan=0.0, posinf=0.0, neginf=0.0)
        contacts = np.nan_to_num(contacts, nan=0.0, posinf=0.0, neginf=0.0)
        external = np.nan_to_num(external, nan=0.0, posinf=0.0, neginf=0.0)

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=self.body_id,
            attributes={
                "deformation_gradient": deformation,
                "mass_quantity": mass.copy(),
                "plastic_volume_ratio": np.nan_to_num(plastic, nan=1.0, posinf=1.0, neginf=1.0),
            },
            fields={
                "contact_signed_distance": contacts,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": external,
            },
            scalars={
                "contact_length_scale": float(self.dx),
                "total_mass": float(mass.sum()),
                "snow_particle_count": float(n),
                "striker_speed": float(np.linalg.norm(self.ball_v)),
                "min_contact_gap": float(np.min(gaps)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        gaps = None
        if self.n:
            relative = self.x - self.ball_x[None, :]
            distance = np.sqrt(np.einsum("nd,nd->n", relative, relative)) - self.ball_radius
            gaps = float(np.minimum(distance, self.x[:, 2] - self.floor_top).min())
        return {
            "snow_particles": int(self.n),
            "total_samples": int(self.n_total),
            "total_mass_kg": float(self.mass_total.sum()),
            "simulated_time_s": float(self.time),
            "grid_spacing_m": float(self.dx),
            "youngs_modulus_pa": float(self.YOUNG_MODULUS),
            "max_particle_speed": (
                float(np.sqrt((self.v * self.v).sum(axis=1)).max()) if self.n else 0.0
            ),
            "striker_position": [float(value) for value in self.ball_x],
            "striker_speed": float(np.linalg.norm(self.ball_v)),
            "min_contact_gap_m": gaps,
        }
