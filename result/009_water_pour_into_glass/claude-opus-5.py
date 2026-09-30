"""Incompressible SPH (position-based fluid) solver for water poured into a glass.

Every sample frame is produced by an actual time-stepping procedure:

1. semi-implicit gravity integration and position prediction,
2. SPH density-constraint projection (position based fluids) with a poly6
   density kernel and spiky kernel gradients, enforcing incompressibility,
3. analytic no-penetration projection against the frozen axisymmetric glass
   geometry (inner wall, outer wall, ground plane),
4. velocity recovery from the projected positions plus XSPH viscosity,
5. prescribed volume-flux inflow at the pour spout while the source is active.

No keyframes, scripted motion, or precomputed trajectories are used.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Position-based incompressible fluid solver for scene 009."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(getattr(scene, "config", {}) or {})
        ic = dict(cfg.get("initial_conditions", {}) or {})
        dom = dict(cfg.get("domain", {}) or {})
        self.rng = np.random.default_rng(abs(int(seed)) % (2 ** 31))

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=np.float64)
        self.vel = np.array(frame0.velocity, dtype=np.float64)
        self.dx = float(frame0.scalars.get("particle_spacing", ic.get("particle_spacing", 0.018)))
        self.rho_phys = float(frame0.scalars.get("rest_density", ic.get("liquid_density", 998.0)))
        if frame0.mass is not None and len(np.asarray(frame0.mass)):
            self.m_p = float(np.asarray(frame0.mass, dtype=np.float64).reshape(-1)[0])
        else:
            self.m_p = self.rho_phys * self.dx ** 3
        n0 = int(len(self.pos))
        self.mass = np.full(n0, self.m_p, dtype=np.float64)
        tag = frame0.attributes.get("source_region")
        self.tag = (np.array(tag, dtype=np.float64).reshape(-1) if tag is not None
                    else np.zeros(n0, dtype=np.float64))
        self.n_initial = n0
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.bmin = np.asarray(dom.get("bounds_min", [-0.33, -0.33, 0.0]), dtype=np.float64)
        self.bmax = np.asarray(dom.get("bounds_max", [0.33, 0.33, 1.1]), dtype=np.float64)
        self.v0_volume = float(
            frame0.scalars.get("initial_liquid_volume", n0 * self.m_p / self.rho_phys)
        )
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self._init_geometry(cfg, ic)
        self._init_source(ic)
        self._init_sph()
        self._init_grid()
        self._init_boundary_samples()
        self.time = 0.0
        self.mass_added = 0.0
        self.mass_removed = 0.0
        self.emit_credit = 0.0
        self.steps = 0
        self.step_index = 0
        self.max_density_error = 0.0

    def _init_geometry(self, cfg: dict, ic: dict) -> None:
        """Frozen axisymmetric glass: inner radius profile R(z), wall thickness."""
        profile = None
        geom = cfg.get("scene_geometry")
        if isinstance(geom, dict):
            for obj in geom.get("objects", []) or []:
                g = dict((obj or {}).get("geometry", {}) or {})
                if g.get("type") == "axisymmetric_container":
                    profile = g.get("inner_profile")
                    self.wall_thickness = float(g.get("wall_thickness", 0.012))
                    break
        if profile is None:
            r = float(ic.get("glass_inner_radius", 0.25))
            h = float(ic.get("glass_inner_height", 0.78))
            profile = [[0.9 * r, 0.0], [r, 0.03], [r, h - 0.03], [r, h]]
            self.wall_thickness = 0.012
        prof = np.asarray(profile, dtype=np.float64).reshape(-1, 2)
        order = np.argsort(prof[:, 1])
        self.prof_r = np.ascontiguousarray(prof[order, 0])
        self.prof_z = np.ascontiguousarray(prof[order, 1])
        self.rim_z = float(self.prof_z[-1])
        self.floor_z = float(max(self.bmin[2], self.prof_z[0]))
        self.inner_radius = float(np.max(self.prof_r))

    def _inner_radius(self, z: np.ndarray) -> np.ndarray:
        """Piecewise-linear inner radius, clamped outside the profile range."""
        return np.interp(z, self.prof_z, self.prof_r)

    def _init_source(self, ic: dict) -> None:
        start = np.asarray(ic.get("jet_start", [-0.14, 0.0, 0.98]), dtype=np.float64)
        end = np.asarray(ic.get("jet_end", [-0.035, 0.0, 0.58]), dtype=np.float64)
        direction = np.asarray(ic.get("source_direction", (end - start)), dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-12:
            direction = np.array([0.0, 0.0, -1.0])
            norm = 1.0
        self.jet_dir = direction / norm
        self.jet_start = start
        self.jet_radius = float(ic.get("jet_radius", 0.042))
        self.jet_speed = float(ic.get("jet_speed", 2.25))
        self.t_src0 = float(ic.get("source_start_time", 0.0))
        self.t_src1 = float(ic.get("source_end_time", 3.25))
        self.flow_rate = float(ic.get("target_inflow_rate_m3_s", 0.01247))
        # Orthonormal frame spanning the inlet disk.
        helper = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(helper, self.jet_dir))) > 0.9:
            helper = np.array([1.0, 0.0, 0.0])
        e1 = helper - float(np.dot(helper, self.jet_dir)) * self.jet_dir
        e1 /= float(np.linalg.norm(e1))
        e2 = np.cross(self.jet_dir, e1)
        e2 /= float(np.linalg.norm(e2))
        self.jet_e1, self.jet_e2 = e1, e2
        # Deterministic Vogel-spiral disk template (near-uniform inlet sampling).
        area = math.pi * self.jet_radius ** 2
        n_disk = max(4, int(round(area / (self.dx * self.dx))))
        golden = math.pi * (3.0 - math.sqrt(5.0))
        idx = np.arange(n_disk, dtype=np.float64)
        rr = self.jet_radius * np.sqrt((idx + 0.5) / n_disk)
        th = idx * golden
        self.disk = (rr * np.cos(th))[:, None] * e1[None, :] + (rr * np.sin(th))[:, None] * e2[None, :]
        self.n_disk = n_disk
        # Particles per second required to realise the prescribed volume flux.
        self.emit_rate = self.flow_rate * self.rho_phys / self.m_p
        self.layer_period = float(n_disk) / max(self.emit_rate, 1e-12)

    def _init_sph(self) -> None:
        h = 2.0 * self.dx
        self.h = h
        self.h2 = h * h
        self.poly6 = 315.0 / (64.0 * math.pi * h ** 9)
        self.spiky = -45.0 / (math.pi * h ** 6)
        # Rest density of a regular lattice at the frozen particle spacing:
        # guarantees an interior particle at spacing dx satisfies C = 0.
        span = int(math.ceil(h / self.dx))
        offs = np.arange(-span, span + 1, dtype=np.float64) * self.dx
        gx, gy, gz = np.meshgrid(offs, offs, offs, indexing="ij")
        r2 = (gx ** 2 + gy ** 2 + gz ** 2).ravel()
        r2 = r2[r2 < self.h2]
        self.rho0 = float(self.m_p * np.sum(self.poly6 * (self.h2 - r2) ** 3))
        self.iterations = 3
        self.substeps = 10
        self.dt = self.sample_dt / self.substeps
        self.xsph = 0.045
        self.wall_margin = 0.4 * self.dx
        self.tangent_damping = 0.985
        self.v_max = 12.0
        self.max_correction = 0.35 * self.dx
        self.char_length = float(self.inner_radius)
        self.ref_velocity = float(max(self.jet_speed, 1e-6))

    def _init_grid(self) -> None:
        """Uniform diagnostic grid covering the frozen domain."""
        extent = self.bmax - self.bmin
        target = 0.022
        counts = np.maximum(np.round(extent / target).astype(int), 2)
        self.grid_shape = (int(counts[0]), int(counts[1]), int(counts[2]))
        self.grid_h = float(np.min(extent / counts))
        axes = [
            self.bmin[d] + (np.arange(counts[d], dtype=np.float64) + 0.5) * (extent[d] / counts[d])
            for d in range(3)
        ]
        gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        self.grid_pts = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
        # Cells occupied by the glass shell are excluded from the active mask.
        rad = np.hypot(self.grid_pts[:, 0], self.grid_pts[:, 1])
        r_in = self._inner_radius(self.grid_pts[:, 2])
        self.grid_solid = (
            (self.grid_pts[:, 2] >= self.floor_z)
            & (self.grid_pts[:, 2] <= self.rim_z)
            & (rad >= r_in)
            & (rad <= r_in + self.wall_thickness)
        )
        self.ls_radius = 2.0 * self.grid_h
        self.vel_radius = 3.0 * self.grid_h
        self.grid_knn = 12
        self.surface_offset = 0.62 * self.dx

    def _init_boundary_samples(self) -> None:
        """Fixed wall sample set with outward-of-solid (fluid-side) unit normals."""
        n_theta = 24
        theta = np.arange(n_theta, dtype=np.float64) * (2.0 * math.pi / n_theta)
        ct, st = np.cos(theta), np.sin(theta)
        pts, nrm = [], []
        # Bottom plane samples (normal +z into the fluid).
        radii = np.array([0.0, 0.25, 0.5, 0.75, 0.95]) * self._inner_radius(
            np.array([self.floor_z])
        )[0]
        for r in radii:
            if r <= 1e-9:
                pts.append([0.0, 0.0, self.floor_z + self.wall_margin])
                nrm.append([0.0, 0.0, 1.0])
                continue
            for k in range(n_theta):
                pts.append([r * ct[k], r * st[k], self.floor_z + self.wall_margin])
                nrm.append([0.0, 0.0, 1.0])
        # Side wall samples: normal is -grad of (rad - R(z)), normalised.
        z_levels = np.linspace(self.floor_z + 0.02, self.rim_z - 0.01, 16)
        eps = 1e-4
        for z in z_levels:
            slope = float(
                (self._inner_radius(np.array([z + eps]))[0]
                 - self._inner_radius(np.array([z - eps]))[0]) / (2.0 * eps)
            )
            r_in = float(self._inner_radius(np.array([z]))[0]) - self.wall_margin
            for k in range(n_theta):
                n_vec = np.array([-ct[k], -st[k], slope], dtype=np.float64)
                n_vec /= float(np.linalg.norm(n_vec))
                pts.append([r_in * ct[k], r_in * st[k], float(z)])
                nrm.append(n_vec.tolist())
        self.bnd_pts = np.asarray(pts, dtype=np.float64)
        self.bnd_nrm = np.asarray(nrm, dtype=np.float64)
        self.bnd_target = np.zeros_like(self.bnd_pts)  # static glass
        self.bnd_radius = 1.6 * self.dx
        self.bnd_knn = 8

    # -------------------------------------------------------------- boundaries
    def _project(self, p: np.ndarray) -> np.ndarray:
        """Analytic no-penetration projection onto the frozen glass geometry."""
        if len(p) == 0:
            return p
        p = np.array(p, dtype=np.float64, copy=True)
        margin = self.wall_margin
        np.maximum(p[:, 2], self.floor_z + margin, out=p[:, 2])
        rad = np.hypot(p[:, 0], p[:, 1])
        safe = np.maximum(rad, 1e-12)
        r_wall = self._inner_radius(p[:, 2])
        r_in = r_wall - margin
        r_out = r_wall + self.wall_thickness + margin
        mid = 0.5 * (r_in + r_out)
        in_band = (p[:, 2] <= self.rim_z) & (rad > r_in) & (rad < r_out)
        push_in = in_band & (rad <= mid)
        push_out = in_band & (rad > mid)
        scale = np.ones_like(rad)
        scale[push_in] = r_in[push_in] / safe[push_in]
        scale[push_out] = r_out[push_out] / safe[push_out]
        moved = push_in | push_out
        p[moved, 0] *= scale[moved]
        p[moved, 1] *= scale[moved]
        return p

    def _wall_velocity(self) -> None:
        """Remove wall-normal inflow so that n . (v - v_wall) vanishes at contact."""
        if len(self.pos) == 0:
            return
        z = self.pos[:, 2]
        rad = np.hypot(self.pos[:, 0], self.pos[:, 1])
        safe = np.maximum(rad, 1e-12)
        r_wall = self._inner_radius(z)
        eps = 1e-4
        slope = (self._inner_radius(z + eps) - self._inner_radius(z - eps)) / (2.0 * eps)
        band = 1.35 * self.wall_margin
        split = r_wall + 0.5 * self.wall_thickness
        # Interior of the glass: fluid-side normal points toward the axis.
        inner = (z <= self.rim_z) & (rad >= r_wall - band) & (rad <= split)
        # Exterior of the shell (spilled water): normal points away from the axis.
        outer = (
            (z <= self.rim_z)
            & (rad > split)
            & (rad <= r_wall + self.wall_thickness + band)
        )
        for mask, sign in ((inner, -1.0), (outer, 1.0)):
            if not np.any(mask):
                continue
            k = np.flatnonzero(mask)
            ct = self.pos[k, 0] / safe[k]
            st = self.pos[k, 1] / safe[k]
            n = np.column_stack((sign * ct, sign * st, -sign * slope[k]))
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            vn = np.einsum("ij,ij->i", self.vel[k], n)
            # Cancel only motion into the solid; tangential slip stays physical.
            self.vel[k] -= np.minimum(vn, 0.0)[:, None] * n
            vn2 = np.einsum("ij,ij->i", self.vel[k], n)
            v_t = self.vel[k] - vn2[:, None] * n
            self.vel[k] = vn2[:, None] * n + self.tangent_damping * v_t
        floor = self.pos[:, 2] <= self.floor_z + 1.35 * self.wall_margin
        if np.any(floor):
            k = np.flatnonzero(floor)
            down = self.vel[k, 2] < 0.0
            kk = k[down]
            self.vel[kk, 2] = 0.0
            self.vel[kk, 0] *= self.tangent_damping
            self.vel[kk, 1] *= self.tangent_damping

    # ----------------------------------------------------------------- inflow
    def _emit(self, dt: float, t: float) -> None:
        """Prescribed volume-flux inflow at the pour spout (mass-exact ledger)."""
        if t < self.t_src0 - 1e-12 or t >= self.t_src1:
            return
        self.emit_credit += self.emit_rate * dt
        layers = int(self.emit_credit // self.n_disk)
        if layers <= 0:
            return
        chunks_p, chunks_v = [], []
        jet_v = self.jet_speed * self.jet_dir
        for _ in range(layers):
            self.emit_credit -= self.n_disk
            frac = min(max(self.emit_credit / float(self.n_disk), 0.0), 1.0)
            base = self.jet_start + (frac * self.layer_period * self.jet_speed) * self.jet_dir
            chunks_p.append(base[None, :] + self.disk)
            chunks_v.append(np.repeat(jet_v[None, :], self.n_disk, axis=0))
        pos_new = np.vstack(chunks_p)
        vel_new = np.vstack(chunks_v)
        n_new = int(len(pos_new))
        pos_new = pos_new + self.rng.normal(0.0, 0.10 * self.dx, size=pos_new.shape)
        vel_new = vel_new + self.rng.normal(0.0, 0.02 * self.jet_speed, size=vel_new.shape)
        self.pos = np.vstack((self.pos, pos_new))
        self.vel = np.vstack((self.vel, vel_new))
        self.mass = np.concatenate((self.mass, np.full(n_new, self.m_p, dtype=np.float64)))
        self.tag = np.concatenate((self.tag, np.ones(n_new, dtype=np.float64)))
        self.mass_added += n_new * self.m_p

    def _cull(self) -> None:
        """Open lateral/top boundaries: outflow is removed and book-kept."""
        p, v = self.pos, self.vel
        keep = (
            np.isfinite(p).all(axis=1)
            & np.isfinite(v).all(axis=1)
            & (p[:, 0] > self.bmin[0]) & (p[:, 0] < self.bmax[0])
            & (p[:, 1] > self.bmin[1]) & (p[:, 1] < self.bmax[1])
            & (p[:, 2] > self.bmin[2]) & (p[:, 2] < self.bmax[2])
        )
        if not bool(np.all(keep)):
            self.mass_removed += float(np.sum(self.mass[~keep]))
            self.pos = self.pos[keep]
            self.vel = self.vel[keep]
            self.mass = self.mass[keep]
            self.tag = self.tag[keep]

    def _clamp_velocity(self) -> None:
        speed = np.linalg.norm(self.vel, axis=1)
        over = speed > self.v_max
        if np.any(over):
            self.vel[over] *= (self.v_max / speed[over])[:, None]

    # -------------------------------------------------------------- time step
    def _substep(self, dt: float, t: float) -> None:
        self._emit(dt, t)
        if len(self.pos) == 0:
            return
        # 1. semi-implicit gravity integration and position prediction
        self.vel += dt * self.gravity[None, :]
        self._clamp_velocity()
        p_old = self.pos
        pred = self._project(p_old + dt * self.vel)

        # 2. incompressibility: SPH density constraint projection
        pairs = cKDTree(pred).query_pairs(self.h, output_type="ndarray")
        n = len(pred)
        if len(pairs):
            ia = pairs[:, 0].astype(np.intp)
            ib = pairs[:, 1].astype(np.intp)
        else:
            ia = np.empty(0, dtype=np.intp)
            ib = np.empty(0, dtype=np.intp)
        w_self = self.poly6 * self.h2 ** 3
        for _ in range(self.iterations):
            if len(ia):
                d = pred[ia] - pred[ib]
                r2 = np.einsum("ij,ij->i", d, d)
                inside = r2 < self.h2
                ja, jb, dv, r2 = ia[inside], ib[inside], d[inside], r2[inside]
                r = np.sqrt(np.maximum(r2, 1e-24))
                w = self.poly6 * (self.h2 - r2) ** 3
                rho = np.full(n, self.m_p * w_self, dtype=np.float64)
                rho += self.m_p * np.bincount(ja, weights=w, minlength=n)
                rho += self.m_p * np.bincount(jb, weights=w, minlength=n)
                c = np.maximum(rho / self.rho0 - 1.0, 0.0)
                # grad_{p_j} C_i for the pair (spiky kernel), scaled by 1/rho0
                coeff = (self.m_p / self.rho0) * self.spiky * (self.h - r) ** 2 / r
                gw = coeff[:, None] * dv
                gsum = np.zeros((n, 3), dtype=np.float64)
                for col in range(3):
                    gsum[:, col] = (
                        np.bincount(ja, weights=gw[:, col], minlength=n)
                        - np.bincount(jb, weights=gw[:, col], minlength=n)
                    )
                g2 = np.einsum("ij,ij->i", gw, gw)
                denom = np.einsum("ij,ij->i", gsum, gsum)
                denom += np.bincount(ja, weights=g2, minlength=n)
                denom += np.bincount(jb, weights=g2, minlength=n)
                lam = -c / (denom + 1.0e-3)
                lam[c <= 0.0] = 0.0
                scale = (lam[ja] + lam[jb])[:, None] * gw
                dp = np.zeros((n, 3), dtype=np.float64)
                for col in range(3):
                    dp[:, col] = (
                        np.bincount(ja, weights=scale[:, col], minlength=n)
                        - np.bincount(jb, weights=scale[:, col], minlength=n)
                    )
                norm = np.linalg.norm(dp, axis=1)
                big = norm > self.max_correction
                if np.any(big):
                    dp[big] *= (self.max_correction / norm[big])[:, None]
                pred = self._project(pred + dp)
                self.max_density_error = max(self.max_density_error, float(np.max(c) if n else 0.0))

        # 3. velocity from the projected positions
        self.pos = pred
        self.vel = (pred - p_old) / dt

        # 4. XSPH viscosity (physical momentum diffusion between neighbours)
        if len(ia) and self.xsph > 0.0:
            d = self.pos[ia] - self.pos[ib]
            r2 = np.einsum("ij,ij->i", d, d)
            inside = r2 < self.h2
            ja, jb, r2 = ia[inside], ib[inside], r2[inside]
            w = self.poly6 * (self.h2 - r2) ** 3
            dvel = self.vel[jb] - self.vel[ja]
            acc = np.zeros((n, 3), dtype=np.float64)
            wsum = np.bincount(ja, weights=w, minlength=n) + np.bincount(
                jb, weights=w, minlength=n
            )
            for col in range(3):
                acc[:, col] = (
                    np.bincount(ja, weights=w * dvel[:, col], minlength=n)
                    - np.bincount(jb, weights=w * dvel[:, col], minlength=n)
                )
            self.vel += self.xsph * acc / np.maximum(wsum + w_self, 1e-12)[:, None]

        self._wall_velocity()
        self._clamp_velocity()
        self._cull()
        self.steps += 1

    def advance_to(self, time: float) -> StateFrame:
        """Step with a uniform dt using integer step indexing (no float drift)."""
        target = float(time)
        total = max(int(round(target / self.dt)), self.step_index)
        while self.step_index < total:
            self._substep(self.dt, self.step_index * self.dt)
            self.step_index += 1
        self.time = target
        return self._build_frame(target)

    # ---------------------------------------------------------- diagnostics
    def _grid_fields(self, tree, n):
        shape = self.grid_shape
        phi = np.full(shape, 3.0 * self.grid_h, dtype=np.float64).ravel()
        vel = np.zeros((phi.size, 3), dtype=np.float64)
        if n > 0:
            k = int(min(self.grid_knn, n))
            dist, idx = tree.query(self.grid_pts, k=k, workers=-1)
            dist = np.atleast_2d(dist.reshape(len(self.grid_pts), k))
            idx = np.atleast_2d(idx.reshape(len(self.grid_pts), k))
            valid = np.isfinite(dist) & (idx < n)
            idx = np.where(valid, idx, 0)
            dist = np.where(valid, dist, np.inf)
            nearest = dist[:, 0]

            # Blended level set (Zhu & Bridson style weighted particle centre)
            w = np.clip(1.0 - (dist / self.ls_radius) ** 2, 0.0, None) ** 3
            w = np.where(valid, w, 0.0)
            wsum = np.sum(w, axis=1)
            centre = np.einsum("gk,gkd->gd", w, self.pos[idx])
            ok = wsum > 1e-12
            centre[ok] /= wsum[ok][:, None]
            d_near = np.where(np.isfinite(nearest), nearest, 3.0 * self.grid_h)
            phi_blend = np.linalg.norm(self.grid_pts - centre, axis=1) - self.surface_offset
            phi = np.where(ok, phi_blend, d_near - self.surface_offset)

            # Velocity: Shepard-normalised SPH interpolation with extrapolation
            wv = np.clip(1.0 - (dist / self.vel_radius) ** 2, 0.0, None) ** 3
            wv = np.where(valid, wv, 0.0)
            wvs = np.sum(wv, axis=1)
            vel = np.einsum("gk,gkd->gd", wv, self.vel[idx])
            okv = wvs > 1e-12
            vel[okv] /= wvs[okv][:, None]
            vel[~okv] = 0.0
        phi = np.clip(phi, -0.5, 0.5)
        active = ((phi <= 0.0) & (~self.grid_solid)).astype(np.float32)
        return (
            phi.reshape(shape).astype(np.float32),
            vel.reshape(shape + (3,)).astype(np.float32),
            active.reshape(shape),
        )

    def _boundary_fields(self, tree, n):
        vel = np.zeros_like(self.bnd_pts)
        if n > 0:
            k = int(min(self.bnd_knn, n))
            dist, idx = tree.query(
                self.bnd_pts, k=k, distance_upper_bound=self.bnd_radius, workers=-1
            )
            dist = np.atleast_2d(dist.reshape(len(self.bnd_pts), k))
            idx = np.atleast_2d(idx.reshape(len(self.bnd_pts), k))
            valid = np.isfinite(dist) & (idx < n)
            idx = np.where(valid, idx, 0)
            w = np.where(valid, np.clip(1.0 - (dist / self.bnd_radius) ** 2, 0.0, None) ** 3, 0.0)
            wsum = np.sum(w, axis=1)
            vel = np.einsum("bk,bkd->bd", w, self.vel[idx])
            ok = wsum > 1e-12
            vel[ok] /= wsum[ok][:, None]
            vel[~ok] = 0.0
        return vel

    def _build_frame(self, t: float) -> StateFrame:
        n = int(len(self.pos))
        pos = np.ascontiguousarray(self.pos, dtype=np.float64).reshape(n, 3)
        vel = np.ascontiguousarray(self.vel, dtype=np.float64).reshape(n, 3)
        pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        mass = np.ascontiguousarray(self.mass, dtype=np.float64).reshape(n)
        tree = cKDTree(pos) if n > 0 else None
        phi, vfield, active = self._grid_fields(tree, n)
        bnd_vel = self._boundary_fields(tree, n)
        cumulative = np.array([self.mass_added - self.mass_removed], dtype=np.float64)
        frame = StateFrame(
            time=float(t),
            position=pos,
            velocity=vel,
            mass=mass,
            attributes={
                "mass_quantity": mass.copy(),
                "source_region": np.ascontiguousarray(self.tag, dtype=np.float64).reshape(n),
            },
            fields={
                "velocity_field": vfield,
                "liquid_level_set": phi,
                "active_mask": active,
                "boundary_velocity": np.ascontiguousarray(bnd_vel, dtype=np.float64),
                "boundary_target_velocity": np.ascontiguousarray(
                    self.bnd_target, dtype=np.float64
                ),
                "boundary_normal": np.ascontiguousarray(self.bnd_nrm, dtype=np.float64),
                "cumulative_mass_source": cumulative,
            },
            scalars={
                "grid_spacing": float(self.grid_h),
                "characteristic_length": float(self.char_length),
                "reference_velocity": float(self.ref_velocity),
                "rest_density": float(self.rho_phys),
                "sph_rest_density": float(self.rho0),
                "particle_spacing": float(self.dx),
                "smoothing_length": float(self.h),
                "time_step": float(self.dt),
                "liquid_mass": float(np.sum(mass)),
                "liquid_volume": float(np.sum(mass) / self.rho_phys),
                "initial_liquid_volume": float(self.v0_volume),
                "particle_count": float(n),
                "source_active": 1.0 if (self.t_src0 <= t < self.t_src1) else 0.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position_based_incompressible_sph",
            "substeps": int(self.substeps),
            "constraint_iterations": int(self.iterations),
            "time_step": float(self.dt),
            "smoothing_length": float(self.h),
            "sph_rest_density": float(self.rho0),
            "total_substeps": int(self.steps),
            "final_particle_count": int(len(self.pos)),
            "initial_particle_count": int(self.n_initial),
            "mass_added_kg": float(self.mass_added),
            "mass_removed_kg": float(self.mass_removed),
            "final_liquid_mass_kg": float(np.sum(self.mass)) if len(self.mass) else 0.0,
            "max_density_error": float(self.max_density_error),
            "grid_shape": [int(v) for v in self.grid_shape],
            "grid_spacing": float(self.grid_h),
            "boundary_sample_count": int(len(self.bnd_pts)),
        }
