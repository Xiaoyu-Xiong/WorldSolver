"""Self-contained MLS-MPM viscoplastic solver for repeated topological merging.

Physical model
--------------
The 24 rounded blocks are discretized as a single material-point continuum.
Momentum is exchanged through a fixed Eulerian background grid, which is what
makes independent blocks weld into one accumulating mass as soon as their
material points share grid support: the topological merge is an emergent
consequence of the continuum discretization, not scripted.

Constitutive law: finite-strain Neo-Hookean elasticity written on the elastic
left Cauchy-Green tensor ``b_e``, closed by a von Mises radial-return map with
rate-dependent (viscous) deviatoric stress.  This is the standard
viscoplastic model used for large plastic flow and needs no SVD, so the whole
step stays vectorized in NumPy.

Discretization: MLS-MPM with quadratic B-spline weights and APIC affine
velocity transfer, symplectic-Euler grid update, explicit substeps well inside
the elastic CFL limit.  Ground and domain walls are enforced on the grid
(sticky below the support, Coulomb friction at the support surface).
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_I3 = np.eye(3)
_OFFSETS = tuple((i, j, k) for i in range(3) for j in range(3) for k in range(3))


class BenchmarkSolver:
    # --- material (cohesive viscoplastic solid) ---------------------------
    YOUNG = 6.0e5           # Pa
    POISSON = 0.30
    YIELD_STRESS = 1.8e4    # Pa, von Mises
    VISCOSITY = 2.0e2       # Pa s, deviatoric rate dependence
    # --- discretization ---------------------------------------------------
    GRID_DX = 0.30          # m
    TARGET_DT = 9.0e-4      # s (elastic CFL allows ~1.1e-2 s)
    FRICTION = 0.55         # support friction coefficient
    MAX_SPEED = 40.0        # m/s numerical safety clip
    PAD = 0.05

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        frame0 = scene.initial_state()

        self.x = np.array(frame0.position, dtype=np.float64, copy=True)
        self.v = np.array(frame0.velocity, dtype=np.float64, copy=True)
        n = self.x.shape[0]
        self.n = n
        mass = frame0.mass
        self.m = (np.array(mass, dtype=np.float64, copy=True)
                  if mass is not None else np.full(n, 1.0))
        self.body_id = (np.array(frame0.body_id, dtype=np.int32, copy=True)
                        if frame0.body_id is not None
                        else np.zeros(n, dtype=np.int32))

        attributes = dict(frame0.attributes or {})
        self.color_id = np.array(
            attributes.get("color_id", np.zeros(n)), dtype=np.float64)
        self.group0 = np.array(
            attributes.get("initial_group_id", self.body_id), dtype=np.float64)
        scalars = dict(frame0.scalars or {})
        self.spacing = float(scalars.get("particle_spacing", 0.205))
        self.radius = np.array(
            attributes.get("particle_radius",
                           np.full(n, 0.546 * self.spacing)), dtype=np.float64)
        self.rest_density = float(scalars.get("rest_density", 1050.0))
        # One material point per lattice cell of the sampling grid.
        self.vol0 = float(self.spacing ** 3)

        gravity = config.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.asarray(gravity, dtype=np.float64).reshape(3)

        # Lame parameters and the elastic wave speed used for the CFL check.
        self.mu = self.YOUNG / (2.0 * (1.0 + self.POISSON))
        self.lame = (self.YOUNG * self.POISSON
                     / ((1.0 + self.POISSON) * (1.0 - 2.0 * self.POISSON)))
        self.bulk = self.lame + 2.0 * self.mu / 3.0
        self.wave_speed = float(
            np.sqrt((self.bulk + 4.0 * self.mu / 3.0) / max(self.rest_density, 1e-9)))

        self.sample_dt = float(config.get("sample_dt", 1.0 / 30.0))
        cfl_dt = 0.4 * self.GRID_DX / max(self.wave_speed, 1e-9)
        step_target = min(self.TARGET_DT, cfl_dt)
        self.substeps = max(1, int(np.ceil(self.sample_dt / step_target)))
        self.dt = self.sample_dt / self.substeps

        # Per-particle state: elastic left Cauchy-Green tensor, total
        # deformation gradient, APIC affine velocity matrix.
        self.b_e = np.tile(_I3, (n, 1, 1))
        self.F = np.tile(_I3, (n, 1, 1))
        self.C = np.zeros((n, 3, 3), dtype=np.float64)

        self._build_grid(config)
        self.time = 0.0
        self.steps_taken = 0
        self.bnd_impulse = np.zeros(3, dtype=np.float64)
        self.interval = 0.0
        self.max_speed_seen = 0.0
        self.group_count = int(len(np.unique(self.body_id)))
        self._nonfinite_events = 0

    # ----------------------------------------------------------------- grid
    def _build_grid(self, config) -> None:
        dx = self.GRID_DX
        lo = np.array([-4.35, -3.65, -0.9], dtype=np.float64)
        hi = np.array([4.35, 3.65, 9.3], dtype=np.float64)
        geometry = config.get("scene_geometry") or {}
        for item in geometry.get("objects", ()) or ():
            spec = (item or {}).get("geometry") or {}
            bmin, bmax = spec.get("bounds_min"), spec.get("bounds_max")
            if bmin is not None and bmax is not None:
                lo = np.minimum(lo, np.asarray(bmin, dtype=np.float64) - 3.0 * dx)
                hi = np.maximum(hi, np.asarray(bmax, dtype=np.float64) + 3.0 * dx)
        # Snap the origin so that the support surface z = 0 lands on a node row.
        lo[2] = -3.0 * dx
        self.origin = lo
        shape = np.ceil((hi - lo) / dx).astype(np.int64) + 3
        self.shape = tuple(int(s) for s in shape)
        nx, ny, nz = self.shape
        self.ncell = nx * ny * nz
        self.stride = np.array([ny * nz, nz, 1], dtype=np.int64)
        self.floor_k = int(round(-lo[2] / dx))  # node row at z = 0

        kk = np.arange(nz)
        ii = np.arange(nx)
        jj = np.arange(ny)
        iii, jjj, kkk = np.meshgrid(ii, jj, kk, indexing="ij")
        sticky = (kkk < self.floor_k) | (kkk >= nz - 2)
        sticky |= (iii < 2) | (iii >= nx - 2) | (jjj < 2) | (jjj >= ny - 2)
        floor = kkk == self.floor_k
        self.sticky_idx = np.flatnonzero(sticky.reshape(-1))
        self.floor_idx = np.flatnonzero((floor & ~sticky).reshape(-1))
        self.bnd_idx = np.union1d(self.sticky_idx, self.floor_idx)
        self.bnd_sticky_mask = np.isin(self.bnd_idx, self.sticky_idx)
        # Interior clamp box for particle positions (keeps grid support valid).
        self.pos_lo = lo + 2.5 * dx
        self.pos_hi = lo + (np.asarray(self.shape, dtype=np.float64) - 3.5) * dx
        self.pos_lo[2] = 0.0

        # Scratch buffers for the 27-node scatter/gather.
        n = self.n
        self.buf_idx = np.empty(27 * n, dtype=np.int64)
        self.buf_w = np.empty(27 * n, dtype=np.float64)
        self.buf_mom = np.empty((27 * n, 3), dtype=np.float64)
        self.m_tiled = np.tile(self.m, 27)

    # ------------------------------------------------------- interpolation
    def _weights(self):
        dx = self.GRID_DX
        xi = (self.x - self.origin) / dx
        base = np.floor(xi - 0.5).astype(np.int64)
        np.clip(base, 0, np.asarray(self.shape, dtype=np.int64) - 3, out=base)
        fx = xi - base
        w = np.empty((self.n, 3, 3), dtype=np.float64)
        w[:, :, 0] = 0.5 * (1.5 - fx) ** 2
        w[:, :, 1] = 0.75 - (fx - 1.0) ** 2
        w[:, :, 2] = 0.5 * (fx - 0.5) ** 2
        rel = base * dx + self.origin - self.x
        return base, w, rel

    # ------------------------------------------------------ constitutive law
    def _kirchhoff_stress(self):
        """Neo-Hookean elastic + viscous deviatoric Kirchhoff stress tau = J sigma."""
        det_b = np.linalg.det(self.b_e)
        det_b = np.clip(det_b, 1.0e-6, 1.0e6)
        j_e = np.clip(np.sqrt(det_b), 0.35, 2.8)
        b_bar = self.b_e / (j_e ** (2.0 / 3.0))[:, None, None]
        trace = np.trace(b_bar, axis1=1, axis2=2)
        dev = b_bar - (trace / 3.0)[:, None, None] * _I3
        tau = self.mu * dev + (self.bulk * np.log(j_e))[:, None, None] * _I3
        # Rate-dependent (viscoplastic) deviatoric part from the APIC gradient.
        strain_rate = 0.5 * (self.C + np.swapaxes(self.C, 1, 2))
        tr_d = np.trace(strain_rate, axis1=1, axis2=2)
        tau += 2.0 * self.VISCOSITY * (
            strain_rate - (tr_d / 3.0)[:, None, None] * _I3)
        norm = np.sqrt(np.einsum("nij,nij->n", tau, tau))
        limit = 60.0 * (self.YIELD_STRESS + self.bulk)
        over = norm > limit
        if np.any(over):
            tau[over] *= (limit / norm[over])[:, None, None]
        return tau

    def _return_map(self) -> None:
        """von Mises radial return in the elastic left Cauchy-Green space."""
        det_b = np.clip(np.linalg.det(self.b_e), 1.0e-6, 1.0e6)
        j_raw = np.sqrt(det_b)
        b_bar = self.b_e / (j_raw ** (2.0 / 3.0))[:, None, None]
        trace = np.trace(b_bar, axis1=1, axis2=2)
        dev = b_bar - (trace / 3.0)[:, None, None] * _I3
        # von Mises equivalent stress of the elastic deviator.
        norm_dev = np.sqrt(np.einsum("nij,nij->n", dev, dev))
        vm = np.sqrt(1.5) * self.mu * norm_dev
        scale = np.where(vm > self.YIELD_STRESS,
                         self.YIELD_STRESS / np.maximum(vm, 1.0e-12), 1.0)
        b_bar = (trace / 3.0)[:, None, None] * _I3 + scale[:, None, None] * dev
        # Keep the plastic flow exactly volume preserving.
        det_bar = np.clip(np.linalg.det(b_bar), 1.0e-9, 1.0e9)
        b_bar /= (det_bar ** (1.0 / 3.0))[:, None, None]
        j_e = np.clip(j_raw, 0.45, 2.2)
        self.b_e = b_bar * (j_e ** (2.0 / 3.0))[:, None, None]

    # ------------------------------------------------------------- one step
    def _step(self) -> None:
        dx = self.GRID_DX
        dt = self.dt
        n = self.n
        base, w, rel = self._weights()
        tau = self._kirchhoff_stress()

        affine = (self.m[:, None, None] * self.C
                  - (4.0 * dt * self.vol0 / (dx * dx)) * tau)
        a_rel = np.einsum("nij,nj->ni", affine, rel)
        col = affine * dx
        mv = self.m[:, None] * self.v
        base_flat = base @ self.stride

        for o, (i, j, k) in enumerate(_OFFSETS):
            sl = slice(o * n, (o + 1) * n)
            weight = w[:, 0, i] * w[:, 1, j] * w[:, 2, k]
            self.buf_w[sl] = weight
            self.buf_idx[sl] = base_flat + (
                i * self.stride[0] + j * self.stride[1] + k * self.stride[2])
            term = a_rel + i * col[:, :, 0] + j * col[:, :, 1] + k * col[:, :, 2]
            self.buf_mom[sl] = weight[:, None] * (mv + term)

        grid_m = np.bincount(self.buf_idx, weights=self.buf_w * self.m_tiled,
                             minlength=self.ncell)
        grid_v = np.empty((self.ncell, 3), dtype=np.float64)
        for d in range(3):
            grid_v[:, d] = np.bincount(self.buf_idx, weights=self.buf_mom[:, d],
                                       minlength=self.ncell)
        active = grid_m > 1.0e-12
        inv_m = np.zeros(self.ncell, dtype=np.float64)
        inv_m[active] = 1.0 / grid_m[active]
        grid_v *= inv_m[:, None]
        grid_v[active] += dt * self.gravity

        # --- boundary conditions on the grid (support + domain walls) ------
        idx = self.bnd_idx
        node_v = grid_v[idx]
        sticky_mask = self.bnd_sticky_mask
        node_v[sticky_mask] = 0.0
        floor_sel = ~sticky_mask
        if np.any(floor_sel):
            sub = node_v[floor_sel]
            hit = sub[:, 2] < 0.0
            if np.any(hit):
                normal_v = sub[hit, 2]
                tangent = sub[hit, :2]
                speed_t = np.sqrt(np.sum(tangent * tangent, axis=1))
                allowed = np.maximum(
                    speed_t + self.FRICTION * normal_v, 0.0)
                factor = np.where(speed_t > 1.0e-12, allowed / np.maximum(
                    speed_t, 1.0e-12), 0.0)
                sub[hit, :2] = tangent * factor[:, None]
                sub[hit, 2] = 0.0
                node_v[floor_sel] = sub
        grid_v[idx] = node_v

        # --- grid to particle (APIC) --------------------------------------
        new_v = np.zeros((n, 3), dtype=np.float64)
        new_c = np.zeros((n, 3, 3), dtype=np.float64)
        for o, (i, j, k) in enumerate(_OFFSETS):
            sl = slice(o * n, (o + 1) * n)
            weight = self.buf_w[sl]
            vn = grid_v[self.buf_idx[sl]]
            wv = weight[:, None] * vn
            new_v += wv
            dpos = rel + dx * np.array([i, j, k], dtype=np.float64)
            new_c += wv[:, :, None] * dpos[:, None, :]
        new_c *= 4.0 / (dx * dx)

        old_v = self.v
        self.v = new_v
        self.C = new_c

        # --- deformation update -------------------------------------------
        grad = _I3 + dt * new_c
        self.F = np.matmul(grad, self.F)
        self.b_e = np.matmul(np.matmul(grad, self.b_e), np.swapaxes(grad, 1, 2))
        self._return_map()
        norm_f = np.sqrt(np.einsum("nij,nij->n", self.F, self.F))
        huge = norm_f > 1.0e3
        if np.any(huge):
            self.F[huge] *= (1.0e3 / norm_f[huge])[:, None, None]

        # --- advection with numerical safety clips -------------------------
        speed = np.sqrt(np.sum(self.v * self.v, axis=1))
        fast = speed > self.MAX_SPEED
        if np.any(fast):
            self.v[fast] *= (self.MAX_SPEED / speed[fast])[:, None]
        self.x += dt * self.v
        below = self.x[:, 2] < self.pos_lo[2]
        if np.any(below):
            self.x[below, 2] = self.pos_lo[2]
            self.v[below, 2] = np.maximum(self.v[below, 2], 0.0)
        np.clip(self.x, self.pos_lo, self.pos_hi, out=self.x)

        # Non-gravity impulse actually delivered this substep: support
        # reaction plus internal contact/coupling, from the momentum balance
        # sum(m dv) = dt * (sum(m g) + reaction).
        total_mass = float(self.m.sum())
        self.bnd_impulse += (np.einsum("n,nd->d", self.m, self.v - old_v)
                             - dt * total_mass * self.gravity)

        if not np.isfinite(self.x).all() or not np.isfinite(self.v).all():
            self._nonfinite_events += 1
            self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
            self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
            self.b_e = np.tile(_I3, (n, 1, 1))
            self.C = np.zeros((n, 3, 3), dtype=np.float64)
        self.max_speed_seen = max(self.max_speed_seen, float(speed.max()))
        self.steps_taken += 1

    # -------------------------------------------------------------- driving
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.time + 1.0e-12:
            span = target - self.time
            count = max(1, int(round(span / self.dt)))
            self.dt = span / count
            self.bnd_impulse = np.zeros(3, dtype=np.float64)
            self.interval = span
            for _ in range(count):
                self._step()
            self.time = target
            self.dt = self.sample_dt / self.substeps
        return self._frame(target)

    def _group_count(self) -> int:
        """Connected components of the occupied grid: merges reduce this count."""
        try:
            from scipy import ndimage
        except Exception:  # pragma: no cover - SciPy is expected to exist
            return self.group_count
        pitch = self.GRID_DX
        cells = np.floor((self.x - self.origin) / pitch).astype(np.int64)
        np.clip(cells, 0, np.asarray(self.shape, dtype=np.int64) - 1, out=cells)
        occupancy = np.zeros(self.shape, dtype=bool)
        occupancy[cells[:, 0], cells[:, 1], cells[:, 2]] = True
        _, count = ndimage.label(occupancy, structure=np.ones((3, 3, 3), bool))
        return int(count)

    def _frame(self, time: float) -> StateFrame:
        gravity_force = self.m[:, None] * self.gravity[None, :]
        if self.interval > 1.0e-12:
            interaction = (self.bnd_impulse / self.interval).reshape(1, 3)
        else:
            interaction = np.zeros((1, 3), dtype=np.float64)
        self.group_count = self._group_count()
        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(self.x, dtype=np.float64),
            velocity=np.ascontiguousarray(self.v, dtype=np.float64),
            mass=np.ascontiguousarray(self.m, dtype=np.float64),
            body_id=np.ascontiguousarray(self.body_id, dtype=np.int32),
            attributes={
                "deformation_gradient": np.ascontiguousarray(
                    self.F, dtype=np.float64),
                "mass_quantity": np.ascontiguousarray(self.m, dtype=np.float64),
                "initial_group_id": self.group0,
                "color_id": self.color_id,
                "particle_radius": self.radius,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": np.ascontiguousarray(
                    gravity_force, dtype=np.float64),
                "interaction_force": np.ascontiguousarray(
                    interaction, dtype=np.float64),
            },
            scalars={
                "gravity_x": float(self.gravity[0]),
                "gravity_y": float(self.gravity[1]),
                "gravity_z": float(self.gravity[2]),
                "particle_spacing": self.spacing,
                "rest_density": self.rest_density,
                "grid_spacing": self.GRID_DX,
                "yield_stress": float(self.YIELD_STRESS),
                "shear_modulus": float(self.mu),
                "bulk_modulus": float(self.bulk),
                "viscosity": float(self.VISCOSITY),
                "time_step": float(self.dt),
                "group_count": float(self.group_count),
                "characteristic_length": float(self.GRID_DX),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "MLS-MPM, quadratic B-spline, APIC transfer",
            "constitutive_model": "Neo-Hookean + von Mises radial return + viscous deviator",
            "particles": int(self.n),
            "grid_shape": list(self.shape),
            "grid_spacing": self.GRID_DX,
            "substeps_per_sample": int(self.substeps),
            "time_step": float(self.sample_dt / self.substeps),
            "elastic_wave_speed": self.wave_speed,
            "cfl_number": float(self.wave_speed * (self.sample_dt / self.substeps)
                                / self.GRID_DX),
            "steps_taken": int(self.steps_taken),
            "final_group_count": int(self.group_count),
            "max_speed": float(self.max_speed_seen),
            "nonfinite_recoveries": int(self._nonfinite_events),
            "total_mass": float(self.m.sum()),
        }
