"""Physically-based particle solver for water poured into a clear cylindrical glass.

The liquid phase is discretised with Lagrangian particles advanced by a
position-based incompressible-fluid (PBF) scheme: gravity is integrated, a
density constraint (Tait-free, per-particle Lagrange multiplier) is solved with
Gauss-Seidel style iterations on the predicted positions, particles exchange
momentum through an XSPH viscosity, and a standard SPH vorticity-confinement
sub-grid term restores the turbulent detail produced by the plunging jet.  The
glass wall, floor, rim flare and exterior ground are handled as impermeable
no-penetration boundaries derived from the frozen axisymmetric inner profile.
A prescribed circular jet injects mass at the configured flow rate (the same
conserved particle mass is used everywhere), and a coarse Eulerian aeration
field models the air entrained at the jet plunge point, which appears as voids
inside the liquid level set.  All diagnostic fields are sampled from the
particles at every requested sample time.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)
_MAX_SPEED = 8.0
_POCKET_CAP = 6000


class BenchmarkSolver:
    def __init__(self) -> None:
        self.t = 0.0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", {}) or {}
        self.cfg = cfg
        ic = cfg["initial_conditions"]
        self.seed = int(seed)
        self.rng = np.random.default_rng(
            (int(seed) * 6364136223846793005 + 1442695040888963407) % (2 ** 63)
        )

        # --- frozen physical parameters -----------------------------------
        self.rho0 = float(ic["liquid_density"])
        self.spacing = float(ic["particle_spacing"])
        self.glass_r = float(ic["glass_inner_radius"])
        self.glass_h = float(ic["glass_inner_height"])
        self.r_part = 0.5 * self.spacing
        self.vol_p = self.spacing ** 3
        self.m_p = self.rho0 * self.vol_p

        # --- smoothing kernel (cubic-spline-like poly6 / spiky gradient) ---
        self.h = 2.5 * self.spacing
        self.h2 = self.h * self.h
        self.k_poly6 = 315.0 / (64.0 * np.pi * self.h ** 9)
        self.k_spiky = -45.0 / (np.pi * self.h ** 6)
        self.w_self = self.k_poly6 * self.h ** 6
        self.w_dq = self.k_poly6 * (self.h2 - (0.2 * self.h) ** 2) ** 3
        self.rho_ref = self.m_p * self._lattice_density()
        self.eps_lambda = 1e-6

        # --- time stepping -------------------------------------------------
        self.dt_sub = 0.002
        self.iters = 2
        self.xsph = 0.12
        self.vort_eps = 1.5

        # --- glass wall profile (authoritative scene geometry) -------------
        self._load_profile(cfg)

        # --- jet source ----------------------------------------------------
        self.jet_start = np.asarray(ic["jet_start"], dtype=np.float64)
        self.jet_end = np.asarray(ic["jet_end"], dtype=np.float64)
        direction = np.asarray(ic.get("source_direction"), dtype=np.float64)
        axis = self.jet_end - self.jet_start
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        if direction.shape != (3,) or np.linalg.norm(direction) < 1e-9:
            direction = axis
        self.jet_dir = direction / np.linalg.norm(direction)
        self.jet_radius = float(ic["jet_radius"])
        self.jet_speed = float(ic["jet_speed"])
        self.t_start = float(ic.get("source_start_time", 0.0))
        self.t_end = float(ic.get("source_end_time", 0.0))
        rate = float(ic["target_inflow_rate_m3_s"])
        self.particles_per_second = max(rate, 0.0) / self.vol_p
        self.inj_accum = 0.0
        self.n_injected = 0
        self.particle_cap = 40000
        side = np.array([0.0, 1.0, 0.0])
        cross = np.cross(self.jet_dir, side)
        n = np.linalg.norm(cross)
        self.e1 = side if n < 1e-9 else cross / n
        self.e2 = np.cross(self.jet_dir, self.e1)

        # --- diagnostic grid (frozen reference grid) -----------------------
        domain = cfg.get("domain", {})
        shape = tuple(int(v) for v in domain.get("reference_grid", (55, 55, 120)))
        self.dmin = np.asarray(domain.get("bounds_min", (-0.33, -0.33, 0.0)), dtype=np.float64)
        self.dmax = np.asarray(domain.get("bounds_max", (0.33, 0.33, 1.1)), dtype=np.float64)
        self.dshape = shape
        self.dcell = (self.dmax - self.dmin) / np.asarray(shape, dtype=np.float64)
        self.dcell_vol = float(np.prod(self.dcell))
        self.grid_spacing = float(self.dcell[0])
        self.characteristic_length = self.glass_r
        self.reference_velocity = self.jet_speed
        self.ncell = int(np.prod(shape))
        self.aeration = np.zeros(shape, dtype=np.float64)
        self._gcoords = tuple(
            c.astype(np.float64) for c in np.indices(shape, dtype=np.float32)
        )
        self._u_prev = np.zeros(shape + (3,), dtype=np.float64)

        # --- inflow / state initialisation ---------------------------------
        frame0 = scene.initial_state()
        pos = np.array(frame0.position, dtype=np.float64)
        vel = np.array(frame0.velocity, dtype=np.float64)
        src = np.asarray(frame0.attributes.get("source_region", np.ones(len(pos))), dtype=bool)
        pos = pos + (self.rng.random(pos.shape) - 0.5) * (0.02 * self.spacing)
        self.pos = np.ascontiguousarray(pos)
        self.vel = np.ascontiguousarray(vel)
        self.jet_tag = np.array(src, dtype=bool)
        self.mass_source_initial = len(self.pos) * self.m_p
        self.cumulative_mass_source = np.zeros(1, dtype=np.float64)

        # --- boundary sample set (glass wall + floor) ----------------------
        self._build_boundary()
        self.t = 0.0
        self.steps = 0

    # ------------------------------------------------------------- geometry
    def _load_profile(self, cfg) -> None:
        profile = None
        geo = cfg.get("scene_geometry") or {}
        for obj in geo.get("objects", []):
            if obj.get("id") == "clear_cylindrical_glass":
                profile = (obj.get("geometry") or {}).get("inner_profile")
        if profile is None:
            profile = [
                [0.225, 0.0],
                [self.glass_r, 0.03],
                [self.glass_r, self.glass_h - 0.03],
                [self.glass_h * 1.0 + 0.005, self.glass_h],
            ]
        prof = np.asarray(profile, dtype=np.float64)
        order = np.argsort(prof[:, 1])
        self.prof_z = prof[order, 1].copy()
        self.prof_r = prof[order, 0].copy()

    def _rmax(self, z: np.ndarray) -> np.ndarray:
        r = np.interp(z, self.prof_z, self.prof_r)
        return np.where(z > self.prof_z[-1], np.inf, r)

    def _build_boundary(self) -> None:
        pts = []
        nrm = []
        theta = np.arange(24) * (2.0 * np.pi / 24.0)
        ct, st = np.cos(theta), np.sin(theta)

        def add_ring(radius, z, nr, nz):
            for c, s in zip(ct, st):
                pts.append([radius * c, radius * s, z])
                nrm.append([nr * c, nr * s, nz])

        def add_ring_axisym(radius, z, slope):
            # inward normal of a meridian line with dr/dz = slope
            nn = np.hypot(1.0, slope)
            add_ring(radius, z, -1.0 / nn, slope / nn)

        zs = self.prof_z
        zlo, zhi = float(zs[0]), float(zs[-1])
        for z in np.linspace(zlo + 0.02, zhi - 0.02, 11):
            r = float(np.interp(z, self.prof_z, self.prof_r))
            add_ring(r - 1e-4, float(z), -1.0, 0.0)
        if zs[1] > zs[0]:
            slope = (self.prof_r[1] - self.prof_r[0]) / (zs[1] - zs[0])
            for z in np.linspace(zlo, zs[1], 4, endpoint=False):
                add_ring_axisym(float(np.interp(z, self.prof_z, self.prof_r)), float(z), slope)
        if self.prof_r[-1] != self.prof_r[-2] and zs[-1] > zs[-2]:
            slope = (self.prof_r[-1] - self.prof_r[-2]) / (zs[-1] - zs[-2])
            for z in np.linspace(zs[-2], zhi, 3, endpoint=False):
                add_ring_axisym(float(np.interp(z, self.prof_z, self.prof_r)), float(z), slope)
        for r in (0.05, 0.12, 0.19):
            for k in range(8):
                a = 2.0 * np.pi * k / 8.0
                pts.append([r * np.cos(a), r * np.sin(a), 0.0])
                nrm.append([0.0, 0.0, 1.0])

        self.bsamp = np.asarray(pts, dtype=np.float64)
        self.bnorm = np.asarray(nrm, dtype=np.float64)
        self.bnorm /= np.linalg.norm(self.bnorm, axis=1, keepdims=True)
        self.btarget = np.zeros_like(self.bsamp)
        self.bvel = np.zeros_like(self.bsamp)

    # --------------------------------------------------------------- kernels
    def _lattice_density(self) -> float:
        k = int(np.ceil(self.h / self.spacing))
        off = np.arange(-k, k + 1, dtype=np.float64) * self.spacing
        x, y, z = np.meshgrid(off, off, off, indexing="ij")
        d2 = x * x + y * y + z * z
        w = self.k_poly6 * np.maximum(self.h2 - d2, 0.0) ** 3
        return float(w.sum())

    def _pairs(self, pos: np.ndarray) -> np.ndarray:
        if len(pos) < 2:
            return np.zeros((0, 2), dtype=np.int64)
        tree = cKDTree(pos, balanced_tree=False, compact_nodes=False)
        return np.asarray(tree.query_pairs(r=self.h, output_type="ndarray"), dtype=np.int64)

    def _geom(self, pos: np.ndarray, pairs: np.ndarray):
        n = len(pos)
        if len(pairs) == 0:
            ii = np.zeros(0, dtype=np.int64)
            jj = np.zeros(0, dtype=np.int64)
            rv = np.zeros((0, 3))
            r = np.zeros(0)
            w = np.zeros(0)
            gw = np.zeros((0, 3))
            return ii, jj, rv, r, w, gw
        ii = np.concatenate((pairs[:, 0], pairs[:, 1]))
        jj = np.concatenate((pairs[:, 1], pairs[:, 0]))
        rv = pos[ii] - pos[jj]
        r2 = np.einsum("ij,ij->i", rv, rv)
        r = np.sqrt(np.maximum(r2, 1e-24))
        w = self.k_poly6 * np.maximum(self.h2 - r2, 0.0) ** 3
        hr = np.maximum(self.h - r, 0.0)
        gw = (self.k_spiky * hr * hr / r)[:, None] * rv
        return ii, jj, rv, r, w, gw

    def _bincount(self, idx: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
        if values.ndim == 1:
            return np.bincount(idx, weights=values, minlength=n)
        return np.column_stack(
            [np.bincount(idx, weights=values[:, d], minlength=n) for d in range(values.shape[1])]
        )

    # -------------------------------------------------------------- stepping
    def _inject(self, t_mid: float, dt: float) -> None:
        if t_mid < self.t_start or t_mid >= self.t_end:
            return
        if len(self.pos) >= self.particle_cap:
            return
        self.inj_accum += self.particles_per_second * dt
        k = int(self.inj_accum)
        if k <= 0:
            return
        self.inj_accum -= k
        k = min(k, self.particle_cap - len(self.pos))
        if k <= 0:
            return
        u = self.rng.random(k)
        rad = self.jet_radius * np.sqrt(u)
        ang = self.rng.random(k) * (2.0 * np.pi)
        offs = (
            np.cos(ang)[:, None] * self.e1[None, :]
            + np.sin(ang)[:, None] * self.e2[None, :]
        ) * rad[:, None]
        new_pos = self.jet_start[None, :] + offs
        new_vel = np.repeat(self.jet_dir[None, :] * self.jet_speed, k, axis=0)
        new_vel += (self.rng.random((k, 3)) - 0.5) * (0.01 * self.jet_speed)
        self.pos = np.ascontiguousarray(np.concatenate((self.pos, new_pos)))
        self.vel = np.ascontiguousarray(np.concatenate((self.vel, new_vel)))
        self.jet_tag = np.concatenate((self.jet_tag, np.ones(k, dtype=bool)))
        self.n_injected += k
        self.cumulative_mass_source = np.array([self.n_injected * self.m_p], dtype=np.float64)

    def _project(self, pos: np.ndarray) -> None:
        rho = np.hypot(pos[:, 0], pos[:, 1])
        rmax = self._rmax(pos[:, 2])
        limit = np.where(np.isfinite(rmax), rmax - self.r_part, np.inf)
        hit = np.isfinite(rmax) & (rho > limit) & (rho > 1e-12)
        if np.any(hit):
            scale = limit[hit] / rho[hit]
            pos[hit, 0] *= scale
            pos[hit, 1] *= scale
        zmin = self.r_part
        zmax = self.dmax[2] - self.r_part
        np.clip(pos[:, 2], zmin, zmax, out=pos[:, 2])

    def _velocity_bc(self, pos: np.ndarray, vel: np.ndarray) -> None:
        rho = np.hypot(pos[:, 0], pos[:, 1])
        rmax = self._rmax(pos[:, 2])
        limit = np.where(np.isfinite(rmax), rmax - self.r_part, np.inf)
        hit = np.isfinite(rmax) & (rho > limit - 2e-3) & (rho > 1e-12)
        if np.any(hit):
            ux = pos[hit, 0] / rho[hit]
            uy = pos[hit, 1] / rho[hit]
            vrad = vel[hit, 0] * ux + vel[hit, 1] * uy
            push = np.where(vrad > 0.0, vrad, 0.0)
            vel[hit, 0] -= push * ux
            vel[hit, 1] -= push * uy
            vel[hit, 0] *= 0.99
            vel[hit, 1] *= 0.99
        floor = pos[:, 2] <= (self.r_part + 1e-9)
        if np.any(floor):
            vel[floor, 2] = np.maximum(vel[floor, 2], 0.0)
            vel[floor, 0] *= 0.985
            vel[floor, 1] *= 0.985
        top = pos[:, 2] >= (self.dmax[2] - self.r_part - 1e-9)
        if np.any(top):
            vel[top, 2] = np.minimum(vel[top, 2], 0.0)

    def _substep(self, dt: float) -> None:
        self._inject(self.t + 0.5 * dt, dt)
        pos = self.pos
        vel = self.vel
        vel += _GRAVITY * dt
        x0 = pos.copy()
        pos = pos + dt * vel

        pairs = self._pairs(pos)
        n = len(pos)
        geom = self._geom(pos, pairs)
        for _ in range(self.iters):
            lam = self._lambda(geom, n)
            pos = pos + self._delta(geom, lam, n)
            self._project(pos)
            if _ < self.iters - 1:
                geom = self._geom(pos, pairs)

        new_vel = (pos - x0) / dt
        self._velocity_bc(pos, new_vel)

        ii, jj, rv, r, w, gw = geom
        if len(ii):
            # XSPH momentum smoothing
            wsum = self._bincount(ii, w, n) + self.w_self
            dv = vel[jj] - vel[ii]
            xs = self._bincount(ii, (self.m_p * w / self.rho0)[:, None] * dv, n)
            new_vel += self.xsph * xs
            # SPH vorticity confinement (sub-grid turbulence model)
            cross = np.cross(dv, gw)
            omega = self._bincount(ii, cross, n) / wsum[:, None]
            mag = np.linalg.norm(omega, axis=1)
            np.clip(mag, 0.0, 500.0, out=mag)
            dmag = mag[jj] - mag[ii]
            gmag = self._bincount(ii, dmag[:, None] * gw, n) / wsum[:, None]
            gn = np.linalg.norm(gmag, axis=1)
            nrm = gmag / np.maximum(gn, 1e-9)[:, None]
            fc = self.vort_eps * self.grid_spacing * np.cross(nrm, omega)
            new_vel += dt * fc

        np.clip(new_vel, -_MAX_SPEED, _MAX_SPEED, out=new_vel)
        bad = ~np.isfinite(new_vel).all(axis=1)
        if np.any(bad):
            new_vel[bad] = 0.0
        self.pos = pos
        self.vel = new_vel
        self.steps += 1

    def _lambda(self, geom, n: int) -> np.ndarray:
        ii, jj, rv, r, w, gw = geom
        if len(ii) == 0:
            return np.zeros(n)
        rho = self.m_p * (self._bincount(ii, w, n) + self.w_self)
        grad_sum = self._bincount(ii, gw, n)
        inv = 1.0 / (self.rho_ref * self.rho_ref)
        sum_sq = self._bincount(ii, np.einsum("ij,ij->i", gw, gw), n) * inv
        sum_sq += np.einsum("ij,ij->i", grad_sum, grad_sum) * inv
        c = np.maximum(rho / self.rho_ref - 1.0, 0.0)
        return -c / (sum_sq + self.eps_lambda)

    def _delta(self, geom, lam: np.ndarray, n: int) -> np.ndarray:
        ii, jj, rv, r, w, gw = geom
        if len(ii) == 0:
            return np.zeros((n, 3))
        # Artificial anti-clustering pressure is omitted: in strict SI kernel
        # units its usual k*(W/W_dq)^n form is not commensurate with lambda and
        # would dominate the constraint correction.  The clamped density
        # constraint alone enforces incompressibility stably.
        coef = (lam[ii] + lam[jj]) / self.rho_ref
        dp = self._bincount(ii, coef[:, None] * gw, n)
        mag = np.linalg.norm(dp, axis=1)
        scale = np.minimum(1.0, (0.5 * self.spacing) / np.maximum(mag, 1e-12))
        return dp * scale[:, None]

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        while self.t < target - 1e-12:
            dt = min(self.dt_sub, target - self.t)
            if dt <= 1e-12:
                break
            self._substep(dt)
            self.t += dt
            guard += 1
            if guard > 200000:
                break
        self.t = target
        return self._emit(target)

    # ------------------------------------------------------------ diagnostics
    def _flat_index(self, points: np.ndarray) -> np.ndarray:
        shape = self.dshape
        idx = np.empty((len(points), 3), dtype=np.int64)
        for d in range(3):
            v = (points[:, d] - self.dmin[d]) / self.dcell[d]
            idx[:, d] = np.clip(np.floor(v).astype(np.int64), 0, shape[d] - 1)
        return (idx[:, 2] * shape[1] + idx[:, 1]) * shape[0] + idx[:, 0]

    def _update_aeration(self, dt: float, pos: np.ndarray, vel: np.ndarray,
                         mask: np.ndarray, u: np.ndarray) -> np.ndarray:
        shape = self.dshape
        aer = self.aeration
        finite = np.isfinite(aer).all()
        if not finite:
            aer[:] = 0.0
        if aer.any():
            d = self.dcell
            zz, yy, xx = self._gcoords
            xi = xx - dt * u[..., 0] / d[0]
            yi = yy - dt * u[..., 1] / d[1]
            zi = zz - dt * (u[..., 2] + 0.12) / d[2]
            aer = ndimage.map_coordinates(
                aer, [zi.ravel(), yi.ravel(), xi.ravel()], order=1, mode="nearest"
            ).reshape(shape)
        jet = self.jet_tag
        if np.any(jet):
            pj = pos[jet]
            inside = np.isfinite(self._rmax(pj[:, 2]))
            pj = pj[inside]
            if len(pj):
                surf = self._pool_surface(pos, jet)
                sel = pj[:, 2] < (surf + 0.04)
                if np.any(sel):
                    ci = self._flat_index(pj[sel])
                    inc = np.bincount(ci, minlength=self.ncell).astype(np.float64)
                    inc *= 0.04
                    aer = aer + inc.reshape(shape)
        aer = aer * np.exp(-0.5 * dt)
        aer[~mask] = 0.0
        np.clip(aer, 0.0, 0.9, out=aer)
        self.aeration = aer
        return aer

    def _pool_surface(self, pos: np.ndarray, jet: np.ndarray) -> float:
        pool = (~jet) & np.isfinite(self._rmax(pos[:, 2]))
        if not np.any(pool):
            return float(self.cfg["initial_conditions"]["initial_water_depth"])
        return float(np.max(pos[pool, 2]))

    def _emit(self, t: float) -> StateFrame:
        pos = self.pos
        vel = self.vel
        n = len(pos)
        shape = self.dshape
        fi = self._flat_index(pos)
        mass = np.full(n, self.m_p, dtype=np.float64)
        mass_g = np.bincount(fi, weights=mass, minlength=self.ncell).reshape(shape)
        mom = np.zeros((self.ncell, 3), dtype=np.float64)
        for d in range(3):
            mom[:, d] = np.bincount(fi, weights=mass * vel[:, d], minlength=self.ncell)
        mom = mom.reshape(shape + (3,))
        filled = mass_g > 1e-12
        u = np.zeros(shape + (3,), dtype=np.float64)
        u[filled] = mom[filled] / mass_g[filled][:, None]
        volfrac = mass_g / (self.rho0 * self.dcell_vol)
        mask = volfrac > 0.06

        aer = self._update_aeration(
            max(self.dt_sub, 0.02), pos, vel, mask, self._u_prev
        )
        interior = ndimage.binary_erosion(mask, iterations=1)
        pockets = (aer > 0.5) & interior
        if int(pockets.sum()) > _POCKET_CAP:
            pockets = np.zeros_like(pockets)
        liquid = mask & ~pockets

        if liquid.any():
            inside = ndimage.distance_transform_edt(liquid, sampling=tuple(self.dcell))
            outside = ndimage.distance_transform_edt(~liquid, sampling=tuple(self.dcell))
            level = (outside - inside).astype(np.float32)
        else:
            level = np.full(shape, float(self.dmax[2]), dtype=np.float32)

        self._u_prev = u
        self.bvel = self._sample_boundary_velocity(pos, vel)
        self.aeration = aer

        return StateFrame(
            time=float(t),
            position=np.array(pos, dtype=np.float64),
            velocity=np.array(vel, dtype=np.float64),
            mass=mass,
            attributes={
                "mass_quantity": mass,
                "volume_quantity": np.full(n, self.vol_p, dtype=np.float64),
                "phase_id": np.zeros(n, dtype=np.float64),
            },
            fields={
                "active_mask": liquid.astype(np.float32),
                "liquid_level_set": level,
                "velocity_field": u.astype(np.float32),
                "boundary_normal": self.bnorm,
                "boundary_target_velocity": self.btarget,
                "boundary_velocity": self.bvel,
                "cumulative_mass_source": self.cumulative_mass_source,
            },
            scalars={
                "grid_spacing": self.grid_spacing,
                "characteristic_length": self.characteristic_length,
                "reference_velocity": self.reference_velocity,
                "rest_density": self.rho0,
                "particle_spacing": self.spacing,
                "initial_liquid_volume": float(
                    self.cfg["initial_conditions"]["initial_liquid_volume_m3"]
                ),
                "target_inflow_rate": float(
                    self.cfg["initial_conditions"]["target_inflow_rate_m3_s"]
                ),
                "source_end_time": self.t_end,
            },
        )

    def _sample_boundary_velocity(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        out = np.zeros_like(self.bsamp)
        if len(pos) == 0:
            return out
        tree = cKDTree(pos, balanced_tree=False, compact_nodes=False)
        lists = tree.query_ball_point(self.bsamp, 1.5 * self.h)
        for k, neighbours in enumerate(lists):
            if not neighbours:
                continue
            idx = np.asarray(neighbours, dtype=np.int64)
            mean = vel[idx].mean(axis=0)
            nrm = self.bnorm[k]
            out[k] = mean - np.dot(mean, nrm) * nrm
        return out

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "steps": int(self.steps),
            "particles": int(len(self.pos)),
            "injected_particles": int(self.n_injected),
            "total_liquid_mass": float(len(self.pos) * self.m_p),
            "initial_liquid_mass": float(self.mass_source_initial),
            "cumulative_mass_source": float(self.cumulative_mass_source[0]),
            "max_speed": float(np.max(np.linalg.norm(self.vel, axis=1)))
            if len(self.vel)
            else 0.0,
            "boundary_samples": int(len(self.bsamp)),
        }
