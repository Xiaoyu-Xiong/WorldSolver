"""Physically based solver for scene 133_viscosity_contrast_splash.

Method
------
Weakly compressible SPH (WCSPH) with

* poly6 kernel for the density estimate,
* spiky kernel gradient for the pressure / artificial-viscosity forces,
* Tait-like linear equation of state  P = c0^2 (rho - rho0),  P >= 0,
* Monaghan artificial viscosity with a *material dependent* coefficient
  alpha (low-viscosity liquid ~ water, high-viscosity liquid ~ thick syrup),
* an XSPH velocity-relaxation term whose rate lambda is again material
  dependent (this is the discrete form of a shear viscosity and produces the
  visible cohesion / damping contrast of the paper's Figure 8),
* explicit penalty-free enforcement of the rigid boundaries (ground, divider
  and the outer basin walls) by projection plus normal restitution and
  tangential friction.

Two matched basins (left = low viscosity, right = high viscosity) each contain
a pool and an identical falling drop, so both drops impact their pool with the
same speed while the splash response differs strongly.

All required trajectory fields (mass quantity, active mask, boundary samples,
cumulative mass source, signed liquid level set, diagnostic velocity grid and
the frozen scalars) are exported every sample time.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

try:  # optional, only used for smoothing of the diagnostic grid
    from scipy.ndimage import uniform_filter
except Exception:  # pragma: no cover
    uniform_filter = None

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        rng = np.random.default_rng(int(seed))
        self._rng = rng
        self.scene = scene
        config = dict(scene.config)
        self.config = config

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=float, copy=True)
        self.vel = np.array(frame0.velocity, dtype=float, copy=True)
        self.mass = np.array(frame0.mass, dtype=float, copy=True)
        self.body_id = np.array(frame0.body_id, dtype=np.int64, copy=True)
        self.labels = np.array(frame0.attributes["material_label"], dtype=np.int64, copy=True)
        self.particle_radius = np.array(frame0.attributes["particle_radius"], dtype=float, copy=True)
        self.deformation_gradient = np.array(
            frame0.attributes["deformation_gradient"], dtype=float, copy=True
        )
        self.n = int(self.pos.shape[0])

        sampling = config["sampling"]
        self.spacing = float(sampling["spacing"])
        self.density0_config = float(sampling["density"])
        self.gravity = np.asarray(config["gravity"], dtype=float)

        # ---- SPH discretisation --------------------------------------
        self.h = 2.0 * self.spacing
        self.h2 = self.h * self.h
        self.poly6_c = 315.0 / (64.0 * np.pi * self.h ** 9)
        self.spiky_c = 45.0 / (np.pi * self.h ** 6)
        self.m = float(self.mass[0]) if self.n else self.density0_config * self.spacing ** 3

        self.c0 = 30.0                 # artificial speed of sound [m/s]
        self.dt_max = 1.0e-3           # substep cap (CFL safe for c0, h, vmax)
        self.vmax = 20.0               # velocity clamp for robustness

        # rest density: calibrate on the (bulk) initial lattice packing
        rho_init = self._density(self.pos)
        self.rho0 = float(max(np.percentile(rho_init, 75.0), 0.85 * self.density0_config))

        # ---- material dependent viscosity ----------------------------
        high = self._high_viscosity_labels()
        self.high_mask = np.isin(self.labels, list(high))
        self.alpha = np.where(self.high_mask, 1.0, 0.02)     # Monaghan alpha
        self.lambda_xsph = np.where(self.high_mask, 10.0, 0.5)  # relaxation rate [1/s]

        # ---- rigid boundaries ----------------------------------------
        self.margin = 0.25 * self.spacing
        self.ground_z = 0.0
        self.domain = np.array([[-3.15, -1.35, self.ground_z], [3.15, 1.35, 3.75]])
        self.divider_min = np.array([-0.08, -0.9, 0.0])
        self.divider_max = np.array([0.08, 0.9, 1.3])

        # ---- diagnostic grid -----------------------------------------
        shape = tuple(int(s) for s in sampling["field_grid_shape"])
        self.grid_shape = shape
        self.grid_min = np.array([-3.0, -1.2, 0.0])
        self.grid_max = np.array([3.0, 1.2, 3.8])
        cell = (self.grid_max - self.grid_min) / np.array(shape, dtype=float)
        self.grid_cell = cell
        centers = [self.grid_min[k] + (np.arange(shape[k]) + 0.5) * cell[k] for k in range(3)]
        gx, gy, gz = np.meshgrid(*centers, indexing="ij")
        self.grid_points = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))

        # ---- boundary samples (static walls) --------------------------
        self.bnd_pos, self.bnd_normal = self._build_boundary_samples()
        self.bnd_target = np.zeros_like(self.bnd_pos)

        # ---- frozen scalars (identical to the scene's initial frame) ---
        self.scalars = {k: float(v) for k, v in frame0.scalars.items()}
        self.cumulative_mass_source = np.array(frame0.fields["cumulative_mass_source"], dtype=float)

        self.time = 0.0
        self.substeps = 0

    # ------------------------------------------------------------- helpers
    def _high_viscosity_labels(self) -> set:
        labels = {1, 3}
        geom = self.config.get("scene_geometry") or {}
        try:
            for obj in geom.get("objects", []):
                binding = obj.get("state_binding") or {}
                if binding.get("attribute_name") != "material_label":
                    continue
                mat = str(obj.get("material_id", ""))
                if "high" in mat:
                    labels.discard(int(binding.get("value", -1)))
                    labels.add(int(binding.get("value", -1)))
                elif "low" in mat:
                    labels.discard(int(binding.get("value", -1)))
        except Exception:
            labels = {1, 3}
        return labels

    def _build_boundary_samples(self):
        pos_list, nrm_list = [], []

        def add(points, normal):
            pos_list.append(np.asarray(points, dtype=float).reshape(-1, 3))
            nrm_list.append(np.tile(np.asarray(normal, dtype=float), (pos_list[-1].shape[0], 1)))

        # ground top face (z = 0)
        xs = np.arange(-3.2, 3.21, 0.4)
        ys = np.arange(-1.4, 1.41, 0.4)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        add(np.column_stack((X.ravel(), Y.ravel(), np.zeros(X.size))), (0.0, 0.0, 1.0))

        # divider side faces (x = +-0.08) and top (z = 1.3)
        ys_d = np.arange(-0.9, 0.91, 0.3)
        zs_d = np.arange(0.05, 1.31, 0.3)
        Y, Z = np.meshgrid(ys_d, zs_d, indexing="ij")
        add(np.column_stack((np.full(Y.size, -0.08), Y.ravel(), Z.ravel())), (-1.0, 0.0, 0.0))
        add(np.column_stack((np.full(Y.size, 0.08), Y.ravel(), Z.ravel())), (1.0, 0.0, 0.0))
        xs_t = np.array([-0.08, 0.0, 0.08])
        X, Y = np.meshgrid(xs_t, ys_d, indexing="ij")
        add(np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, 1.3))), (0.0, 0.0, 1.0))

        # outer basin walls (numerical containment of the rigid basin)
        ys_w = np.arange(-1.2, 1.21, 0.4)
        zs_w = np.arange(0.1, 1.31, 0.3)
        Y, Z = np.meshgrid(ys_w, zs_w, indexing="ij")
        add(np.column_stack((np.full(Y.size, -3.2), Y.ravel(), Z.ravel())), (-1.0, 0.0, 0.0))
        add(np.column_stack((np.full(Y.size, 3.2), Y.ravel(), Z.ravel())), (1.0, 0.0, 0.0))
        xs_w = np.arange(-3.0, 3.01, 0.4)
        X, Z = np.meshgrid(xs_w, zs_w, indexing="ij")
        add(np.column_stack((X.ravel(), np.full(X.size, -1.4), Z.ravel())), (0.0, -1.0, 0.0))
        add(np.column_stack((X.ravel(), np.full(X.size, 1.4), Z.ravel())), (0.0, 1.0, 0.0))

        pos = np.concatenate(pos_list, axis=0)
        nrm = np.concatenate(nrm_list, axis=0)
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        return pos, nrm

    @staticmethod
    def _scatter(vec, idx, n):
        """Accumulate rows of vec (P,3) into out[idx] (n,3)."""
        flat = np.bincount(
            (idx[:, None] * 3 + np.arange(3)).ravel(),
            weights=np.ascontiguousarray(vec).ravel(),
            minlength=3 * n,
        )
        return flat.reshape(n, 3)

    def _neighbors(self, pos):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(self.h, output_type="ndarray")
        if pairs.ndim != 2 or pairs.shape[0] == 0:
            pairs = np.zeros((0, 2), dtype=np.int64)
        return tree, pairs

    def _density(self, pos):
        _, pairs = self._neighbors(pos)
        w0 = self.poly6_c * self.h2 ** 3
        rho = np.full(self.n, self.m * w0)
        if pairs.shape[0]:
            i, j = pairs[:, 0], pairs[:, 1]
            rij = pos[i] - pos[j]
            r2 = np.einsum("ij,ij->i", rij, rij)
            w = self.poly6_c * np.maximum(self.h2 - r2, 0.0) ** 3
            c = self.m * w
            rho += np.bincount(i, weights=c, minlength=self.n)
            rho += np.bincount(j, weights=c, minlength=self.n)
        return rho

    # ------------------------------------------------------------ physics
    def _step(self, dt):
        n, h, h2 = self.n, self.h, self.h2
        pos, vel = self.pos, self.vel
        tree, pairs = self._neighbors(pos)

        # ---- density ------------------------------------------------
        w0 = self.poly6_c * h2 ** 3
        rho = np.full(n, self.m * w0)
        if pairs.shape[0]:
            i, j = pairs[:, 0], pairs[:, 1]
            rij = pos[i] - pos[j]
            r2 = np.einsum("ij,ij->i", rij, rij)
            r = np.maximum(np.sqrt(np.maximum(r2, 1e-12)), 0.1 * self.spacing)
            w = self.poly6_c * np.maximum(h2 - r2, 0.0) ** 3
            c = self.m * w
            rho += np.bincount(i, weights=c, minlength=n)
            rho += np.bincount(j, weights=c, minlength=n)
        rho = np.maximum(rho, 0.25 * self.rho0)

        acc = np.tile(self.gravity, (n, 1))

        if pairs.shape[0]:
            # ---- pressure (linear EOS, tensile clipped) -------------
            P = self.c0 ** 2 * np.maximum(rho - self.rho0, 0.0)
            gfac = self.spiky_c * np.maximum(h - r, 0.0) ** 2 / r

            psym = self.m * (P[i] / rho[i] ** 2 + P[j] / rho[j] ** 2)

            # ---- Monaghan artificial viscosity (material dependent) --
            vij = vel[i] - vel[j]
            vdotr = np.einsum("ij,ij->i", vij, rij)
            mu = h * vdotr / (r2 + 0.01 * h2)
            rho_bar = 0.5 * (rho[i] + rho[j])
            alpha_bar = 0.5 * (self.alpha[i] + self.alpha[j])
            Pi = np.where(vdotr < 0.0, (-alpha_bar * self.c0 * mu) / rho_bar, 0.0)

            coef = (psym + self.m * Pi) * gfac
            fpair = coef[:, None] * rij
            acc += self._scatter(fpair, i, n)
            acc -= self._scatter(fpair, j, n)

        # ---- integrate velocity -------------------------------------
        vel = vel + dt * acc
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        np.copyto(vel, vel * (self.vmax / np.maximum(speed, 1e-12)), where=speed > self.vmax)

        # ---- XSPH viscous relaxation (material dependent rate) -------
        if pairs.shape[0]:
            wj = (self.m / rho[j]) * w
            wi = (self.m / rho[i]) * w
            num = np.zeros((n, 3))
            den = np.full(n, (self.m / rho) * w0)
            num += self._scatter(wj[:, None] * vel[j], i, n)
            den += np.bincount(i, weights=wj, minlength=n)
            num += self._scatter(wi[:, None] * vel[i], j, n)
            den += np.bincount(j, weights=wi, minlength=n)
            vbar = num / np.maximum(den, 1e-12)[:, None]
            blend = 1.0 - np.exp(-self.lambda_xsph * dt)
            vel = vel + blend[:, None] * (vbar - vel)

        # ---- integrate position --------------------------------------
        pos = pos + dt * vel
        self._apply_boundaries(pos, vel)

        # robustness guard: never propagate non-finite state
        np.nan_to_num(pos, copy=False, nan=0.0, posinf=self.domain[1], neginf=self.domain[0])
        np.nan_to_num(vel, copy=False, nan=0.0, posinf=self.vmax, neginf=-self.vmax)

        self.pos, self.vel = pos, vel
        self.substeps += 1

    def _apply_boundaries(self, pos, vel):
        m = self.margin
        rest, fric = 0.1, 0.9

        def clamp_axis(axis, lo, hi):
            low = pos[:, axis] < lo
            if np.any(low):
                pos[low, axis] = lo
                v = vel[low, axis]
                vel[low, axis] = np.where(v < 0.0, -rest * v, v)
                tan = [a for a in range(3) if a != axis]
                vel[np.ix_(low, tan)] *= fric
            high = pos[:, axis] > hi
            if np.any(high):
                pos[high, axis] = hi
                v = vel[high, axis]
                vel[high, axis] = np.where(v > 0.0, -rest * v, v)
                tan = [a for a in range(3) if a != axis]
                vel[np.ix_(high, tan)] *= fric

        lo, hi = self.domain[0], self.domain[1]
        for axis in range(3):
            clamp_axis(axis, lo[axis] + m, hi[axis] - m)

        # rigid divider box: project out along the least penetrating axis
        bmin, bmax = self.divider_min - m, self.divider_max + m
        inside = np.all(pos >= bmin, axis=1) & np.all(pos <= bmax, axis=1)
        if np.any(inside):
            idx = np.nonzero(inside)[0]
            p = pos[idx]
            depths = np.stack(
                (p[:, 0] - bmin[0], bmax[0] - p[:, 0],
                 p[:, 1] - bmin[1], bmax[1] - p[:, 1],
                 np.full(p.shape[0], np.inf), bmax[2] - p[:, 2]),
                axis=1,
            )
            k = np.argmin(depths, axis=1)
            axis_of = (0, 0, 1, 1, 2, 2)
            sign_of = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
            for case in range(6):
                sel = k == case
                if not np.any(sel):
                    continue
                ax = axis_of[case]
                sgn = sign_of[case]
                rows = idx[sel]
                pos[rows, ax] = bmin[ax] if sgn < 0 else bmax[ax]
                vn = vel[rows, ax]
                vel[rows, ax] = np.where(sgn * vn > 0.0, -rest * vn, vn)
                tan = [a for a in range(3) if a != ax]
                vel[np.ix_(rows, tan)] *= fric

    # ------------------------------------------------------------- output
    def _frame(self, time: float) -> StateFrame:
        pos, vel = self.pos, self.vel
        tree = cKDTree(pos)

        # ---- diagnostic velocity grid (mass weighted, lightly smoothed) --
        rel = (pos - self.grid_min) / self.grid_cell
        idx = np.floor(rel).astype(np.int64)
        idx = np.clip(idx, 0, np.array(self.grid_shape) - 1)
        nx, ny, nz = self.grid_shape
        flat = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]
        ncell = nx * ny * nz
        wnum = self._scatter(self.mass[:, None] * vel, flat, ncell)
        wden = np.bincount(flat, weights=self.mass, minlength=ncell)
        if uniform_filter is not None:
            comps = []
            for c in range(3):
                num_s = uniform_filter(wnum[:, c].reshape(self.grid_shape), size=2, mode="constant")
                den_s = uniform_filter(wden.reshape(self.grid_shape), size=2, mode="constant")
                comps.append(num_s / np.maximum(den_s, 1e-9))
            velocity_field = np.stack(comps, axis=-1)
            wden_s = uniform_filter(wden.reshape(self.grid_shape), size=2, mode="constant")
        else:
            velocity_field = (wnum / np.maximum(wden, 1e-12)[:, None]).reshape(self.grid_shape + (3,))
            wden_s = wden.reshape(self.grid_shape)

        # ---- signed liquid level set (metres, negative inside liquid) ----
        dist, _ = tree.query(self.grid_points, k=1)
        level_set = (dist.reshape(self.grid_shape) - 0.6 * self.spacing)
        active_mask = level_set <= 0.0
        if not np.any(active_mask):
            active_mask = wden_s > 1e-9

        # ---- wall sample velocities (fluid side, distance weighted) -------
        cutoff = 1.5 * self.spacing
        k_near = min(8, max(1, self.n))
        d, nn = tree.query(self.bnd_pos, k=k_near)
        d = np.atleast_2d(d)
        nn = np.atleast_2d(nn)
        wgt = np.where(d < cutoff, 1.0 / np.maximum(d, 1e-3), 0.0)
        bvel = np.einsum("bk,bkd->bd", wgt, vel[nn]) / np.maximum(wgt.sum(axis=1), 1e-12)[:, None]

        fields = {
            "active_mask": active_mask.astype(np.float64),
            "boundary_normal": self.bnd_normal.copy(),
            "boundary_target_velocity": self.bnd_target.copy(),
            "boundary_velocity": bvel,
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "liquid_level_set": level_set,
            "velocity_field": velocity_field,
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": self.particle_radius.copy(),
            "material_label": self.labels.copy(),
            "deformation_gradient": self.deformation_gradient.copy(),
        }
        frame = StateFrame(
            float(time),
            pos.copy(),
            vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=dict(self.scalars),
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- driver
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.time + 1e-12:
            span = target - self.time
            nsub = max(1, int(np.ceil(span / self.dt_max)))
            dt = span / nsub
            for _ in range(nsub):
                self._step(dt)
            self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "weakly_compressible_sph_with_material_viscosity_contrast",
            "particles": int(self.n),
            "substeps": int(self.substeps),
            "smoothing_radius": float(self.h),
            "rest_density": float(self.rho0),
            "sound_speed": float(self.c0),
            "time": float(self.time),
        }
