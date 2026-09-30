"""Multiphase incompressible particle solver for the octagonal two-phase drop.

Physical model
--------------
Two immiscible weakly-compressible liquid phases (heavy blue, rho = 1000, and
light pink, rho = 250) are advanced with a position-based incompressible SPH
scheme (PBF / density-constraint projection) written from scratch on NumPy /
SciPy:

  1. explicit gravity + physical (Morris/Hu-Adams style) shear viscosity +
     Akinci-style intra-phase cohesion,
  2. semi-implicit position prediction,
  3. iterative projection of the volume-fraction constraint
     C_i = sum_j V_j W_ij / n0 - 1 >= 0 with inverse-mass weighted
     corrections.  The 1/m weighting is what makes the light phase rise
     through the heavy phase (buoyancy) with exact pairwise momentum
     exchange, so the density ratio of 4 is honoured.
  4. velocity update v = (x* - x)/dt, no-slip wall drag, hard projection on
     the analytic signed distance of the frozen octagonal basin.

Static boundary particles sampled inside the basin solid (from the frozen
octagonal geometry) supply the missing kernel support so the fluid cannot
leak through the floor or the eight walls.

Nothing about the motion is scripted: only gravity, the staged release times
frozen in the configuration, and the boundary geometry are prescribed.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

try:  # optional, only used to extend the diagnostic level set
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover
    _ndimage = None

from worldsolver.contracts import StateFrame


# --- frozen basin geometry (assets/octagonal_basin.obj) ----------------------
OCT_APOTHEM = 1.060660          # inner octagon face offset (m)
OCT_CIRCUMRADIUS = 1.148246     # inner octagon vertex radius (m)
Z_FLOOR = 0.08                  # inner floor height (m)
WALL_TOP = 0.62                 # rim height (m)


def _acc3(index: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    """Scatter-add [P,3] pair contributions onto [count,3]."""
    out = np.zeros((count, 3), dtype=float)
    if index.size:
        for k in range(3):
            out[:, k] = np.bincount(index, weights=values[:, k], minlength=count)
    return out


def _acc1(index: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    if index.size == 0:
        return np.zeros(count, dtype=float)
    return np.bincount(index, weights=values, minlength=count)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config

        frame0 = scene.initial_state()
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.n = int(self.position.shape[0])

        attributes = dict(frame0.attributes)
        self.phase = np.asarray(attributes["phase_id"], dtype=float).round().astype(np.int32)
        self.density0 = np.asarray(attributes["density"], dtype=float).copy()
        self.release_time = np.asarray(attributes["release_time"], dtype=float).copy()
        self.volume = np.asarray(attributes["target_volume"], dtype=float).copy()
        self.mass = self.density0 * self.volume
        self.inv_mass = 1.0 / self.mass

        gravity = config.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)

        # ---- discretisation lengths
        self.d = float(np.mean(self.volume) ** (1.0 / 3.0))     # particle spacing
        self.h = 2.2 * self.d                                   # kernel support
        self.inv_h = 1.0 / self.h
        self.sigma = 8.0 / (np.pi * self.h ** 3)                # cubic spline norm
        self.volume_b = self.d ** 3                             # boundary volume
        self.margin = 0.40 * self.d

        # ---- time stepping
        self.dt_sub = 2.0e-3
        self.iterations = 3
        self.relaxation = 0.55
        self.max_correction = 0.30 * self.d
        self.max_speed = 15.0
        self.max_accel = 2.5e3

        # ---- material coefficients
        kinematic_viscosity = 4.0e-3                            # m^2 / s
        self.eta = kinematic_viscosity * self.density0          # dynamic viscosity
        self.cohesion = 0.01

        # ---- octagonal wall planes (outward face normals every 45 degrees)
        angles = np.arange(8) * (np.pi / 4.0)
        self.wall_n = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        self.wall_limit = OCT_APOTHEM - self.margin

        # ---- static boundary particles filling the basin solid
        self.boundary = self._build_boundary_particles()
        self.tree_b = cKDTree(self.boundary) if len(self.boundary) else None

        # ---- reference volume-fraction of a full lattice packing
        self.n0 = self._reference_number_density()

        # ---- diagnostic grid
        self._build_diagnostic_grid()

        # ---- neighbour bookkeeping
        self.search_radius = self.h + 0.6 * self.d
        self.rebuild_tolerance = 0.5 * (self.search_radius - self.h)
        self._pos_reference = None
        self.ff_i = np.zeros(0, dtype=np.intp)
        self.ff_j = np.zeros(0, dtype=np.intp)
        self.fb_i = np.zeros(0, dtype=np.intp)
        self.fb_b = np.zeros(0, dtype=np.intp)

        self.time = 0.0
        self.steps = 0
        self._update_active()
        self._rebuild_neighbors()

        self.characteristic_length = 2.0 * OCT_APOTHEM
        self.phase_count = 2
        self.phase_mass = np.array(
            [float(self.mass[self.phase == k].sum()) for k in range(self.phase_count)]
        )

    # ------------------------------------------------------- boundary sampling
    def _build_boundary_particles(self) -> np.ndarray:
        d = self.d
        thickness = 3.05 * d
        span = OCT_CIRCUMRADIUS + thickness + d
        axis = np.arange(-span, span + 1e-9, d)
        zs = np.arange(Z_FLOOR - thickness, WALL_TOP - 0.06, d)
        gx, gy, gz = np.meshgrid(axis, axis, zs, indexing="ij")
        pts = np.stack((gx.ravel(), gy.ravel(), gz.ravel()), axis=1)

        lateral = OCT_APOTHEM - (pts[:, :2] @ self.wall_n.T).max(axis=1)
        phi = np.minimum(pts[:, 2] - Z_FLOOR, lateral)
        keep = (phi < 0.0) & (phi > -thickness)
        return np.ascontiguousarray(pts[keep])

    def _reference_number_density(self) -> float:
        reach = int(np.ceil(self.h / self.d)) + 1
        offsets = np.arange(-reach, reach + 1) * self.d
        gx, gy, gz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
        r = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2).ravel()
        w, _ = self._kernel(r)
        return float((self.d ** 3) * w.sum())

    def _build_diagnostic_grid(self) -> None:
        self.grid_dx = 0.07
        dx = self.grid_dx
        self.nx = 35
        self.ny = 35
        self.nz = 62
        self.gx = (np.arange(self.nx) - (self.nx - 1) / 2.0) * dx
        self.gy = (np.arange(self.ny) - (self.ny - 1) / 2.0) * dx
        self.gz = np.arange(self.nz) * dx + 0.5 * dx
        self.grid_origin = np.array([self.gx[0], self.gy[0], self.gz[0]])
        cx, cy, cz = np.meshgrid(self.gx, self.gy, self.gz, indexing="ij")
        self.cell_centers = np.stack((cx, cy, cz), axis=-1)
        lateral = OCT_APOTHEM - (
            np.stack((cx.ravel(), cy.ravel()), axis=1) @ self.wall_n.T
        ).max(axis=1)
        inside_container = (lateral > 0.0) & (cz.ravel() > Z_FLOOR - 0.5 * dx)
        self.container_mask = inside_container.reshape(self.nx, self.ny, self.nz)
        self.splat_radius = 2.0 * dx
        self.splat_reach = int(np.ceil(self.splat_radius / dx))
        self.particle_radius = 0.75 * self.d

    # ----------------------------------------------------------------- kernels
    def _kernel(self, r: np.ndarray):
        """Cubic spline W and dW/dr with compact support ``h``."""
        q = np.asarray(r, dtype=float) * self.inv_h
        w = np.zeros_like(q)
        dw = np.zeros_like(q)
        near = q < 0.5
        far = (q >= 0.5) & (q < 1.0)
        qn = q[near]
        w[near] = 6.0 * (qn ** 3 - qn ** 2) + 1.0
        dw[near] = 18.0 * qn ** 2 - 12.0 * qn
        t = 1.0 - q[far]
        w[far] = 2.0 * t ** 3
        dw[far] = -6.0 * t ** 2
        return self.sigma * w, (self.sigma * self.inv_h) * dw

    def _cohesion_kernel(self, r: np.ndarray) -> np.ndarray:
        h = self.h
        out = np.zeros_like(r)
        outer = (r > 0.5 * h) & (r < h)
        inner = (r > 1e-9) & (r <= 0.5 * h)
        out[outer] = (h - r[outer]) ** 3 * r[outer] ** 3
        out[inner] = 2.0 * (h - r[inner]) ** 3 * r[inner] ** 3 - h ** 6 / 64.0
        return (32.0 / (np.pi * h ** 9)) * out

    # -------------------------------------------------------------- neighbours
    def _update_active(self) -> None:
        self.active = self.time >= self.release_time - 1e-9
        self.inv_mass_eff = np.where(self.active, self.inv_mass, 0.0)

    def _rebuild_neighbors(self) -> None:
        tree_f = cKDTree(self.position)
        pairs = tree_f.query_pairs(self.search_radius, output_type="ndarray")
        if pairs.size:
            self.ff_i = np.ascontiguousarray(pairs[:, 0].astype(np.intp))
            self.ff_j = np.ascontiguousarray(pairs[:, 1].astype(np.intp))
        else:
            self.ff_i = np.zeros(0, dtype=np.intp)
            self.ff_j = np.zeros(0, dtype=np.intp)

        if self.tree_b is not None:
            lists = tree_f.query_ball_tree(self.tree_b, self.search_radius)
            counts = np.fromiter((len(item) for item in lists), dtype=np.intp, count=self.n)
            total = int(counts.sum())
            if total:
                flat = np.empty(total, dtype=np.intp)
                cursor = 0
                for item in lists:
                    size = len(item)
                    if size:
                        flat[cursor:cursor + size] = item
                        cursor += size
                self.fb_b = flat
                self.fb_i = np.repeat(np.arange(self.n, dtype=np.intp), counts)
            else:
                self.fb_b = np.zeros(0, dtype=np.intp)
                self.fb_i = np.zeros(0, dtype=np.intp)
        self._pos_reference = self.position.copy()

    def _maybe_rebuild(self) -> None:
        if self._pos_reference is None:
            self._rebuild_neighbors()
            return
        shift = np.abs(self.position - self._pos_reference).max()
        if shift > self.rebuild_tolerance:
            self._rebuild_neighbors()

    # ------------------------------------------------------------------- forces
    def _pair_geometry(self, positions: np.ndarray):
        dv_ff = positions[self.ff_i] - positions[self.ff_j]
        r_ff = np.sqrt(np.einsum("ij,ij->i", dv_ff, dv_ff))
        e_ff = dv_ff / np.maximum(r_ff, 1e-12)[:, None]
        if self.fb_i.size:
            dv_fb = positions[self.fb_i] - self.boundary[self.fb_b]
            r_fb = np.sqrt(np.einsum("ij,ij->i", dv_fb, dv_fb))
            e_fb = dv_fb / np.maximum(r_fb, 1e-12)[:, None]
        else:
            r_fb = np.zeros(0)
            e_fb = np.zeros((0, 3))
        return r_ff, e_ff, r_fb, e_fb

    def _explicit_accel(self, r_ff, e_ff, r_fb, e_fb) -> np.ndarray:
        accel = np.zeros((self.n, 3), dtype=float)
        accel[self.active] = self.gravity

        _, dw_ff = self._kernel(r_ff)
        i, j = self.ff_i, self.ff_j

        if i.size:
            # --- physical shear viscosity (harmonic mean dynamic viscosity)
            eta_i = self.eta[i]
            eta_j = self.eta[j]
            eta_ij = 2.0 * eta_i * eta_j / np.maximum(eta_i + eta_j, 1e-12)
            coef = (
                2.0
                * eta_ij
                * self.volume[i]
                * self.volume[j]
                * (-dw_ff / np.maximum(r_ff, 1e-9))
            )
            force = coef[:, None] * (self.velocity[j] - self.velocity[i])

            # --- intra-phase cohesion (immiscible surface tension surrogate)
            same = self.phase[i] == self.phase[j]
            if np.any(same):
                cval = self._cohesion_kernel(r_ff)
                cval = np.where(same, cval, 0.0)
                coh = (self.cohesion * cval)[:, None] * e_ff
                force = force - coh * (self.mass[i] * self.mass[j])[:, None]

            accel += _acc3(i, force, self.n) * self.inv_mass[:, None]
            accel -= _acc3(j, force, self.n) * self.inv_mass[:, None]

        if self.fb_i.size:
            _, dw_fb = self._kernel(r_fb)
            bi = self.fb_i
            coef_b = (
                2.0
                * self.eta[bi]
                * self.volume[bi]
                * self.volume_b
                * (-dw_fb / np.maximum(r_fb, 1e-9))
            )
            force_b = coef_b[:, None] * (-self.velocity[bi])   # no-slip walls
            accel += _acc3(bi, force_b, self.n) * self.inv_mass[:, None]

        accel[~self.active] = 0.0
        norm = np.sqrt(np.einsum("ij,ij->i", accel, accel))
        over = norm > self.max_accel
        if np.any(over):
            accel[over] *= (self.max_accel / norm[over])[:, None]
        return accel

    # ------------------------------------------------------------- projection
    def _density_projection(self, pred: np.ndarray) -> None:
        i, j = self.ff_i, self.ff_j
        bi, bb = self.fb_i, self.fb_b
        v = self.volume
        inv_m = self.inv_mass_eff
        w_self = self.sigma
        eps = 1.0e-9

        for _ in range(self.iterations):
            r_ff, e_ff, r_fb, e_fb = self._pair_geometry(pred)
            w_ff, dw_ff = self._kernel(r_ff)
            if bi.size:
                w_fb, dw_fb = self._kernel(r_fb)
            else:
                w_fb = dw_fb = np.zeros(0)

            # volume fraction
            number = v * w_self
            number += _acc1(i, v[j] * w_ff, self.n)
            number += _acc1(j, v[i] * w_ff, self.n)
            if bi.size:
                number += _acc1(bi, self.volume_b * w_fb, self.n)

            constraint = np.maximum(number / self.n0 - 1.0, 0.0)
            constraint[~self.active] = 0.0

            # gradient of C_i with respect to its own position
            grad = _acc3(i, (v[j] * dw_ff)[:, None] * e_ff, self.n)
            grad += _acc3(j, (-v[i] * dw_ff)[:, None] * e_ff, self.n)
            if bi.size:
                grad += _acc3(bi, (self.volume_b * dw_fb)[:, None] * e_fb, self.n)

            denom = inv_m * np.einsum("ij,ij->i", grad, grad)
            denom += _acc1(i, inv_m[j] * (v[j] * dw_ff) ** 2, self.n)
            denom += _acc1(j, inv_m[i] * (v[i] * dw_ff) ** 2, self.n)

            lam = -constraint / (denom / self.n0 ** 2 + eps)
            lam[~self.active] = 0.0

            coef = lam[i] * v[j] + lam[j] * v[i]
            contrib = (coef * dw_ff)[:, None] * e_ff
            delta = _acc3(i, contrib, self.n) - _acc3(j, contrib, self.n)
            if bi.size:
                delta += _acc3(bi, (lam[bi] * self.volume_b * dw_fb)[:, None] * e_fb, self.n)
            delta *= (inv_m / self.n0)[:, None]

            step = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            over = step > self.max_correction
            if np.any(over):
                delta[over] *= (self.max_correction / step[over])[:, None]
            delta[~self.active] = 0.0

            pred += self.relaxation * delta
            self._project_to_domain(pred)

    def _project_to_domain(self, pos: np.ndarray) -> None:
        for _ in range(2):
            signed = pos[:, :2] @ self.wall_n.T
            over = signed - self.wall_limit
            worst = over.max(axis=1)
            hit = worst > 0.0
            if not np.any(hit):
                break
            plane = over.argmax(axis=1)[hit]
            pos[hit, :2] -= worst[hit, None] * self.wall_n[plane]
        np.clip(pos[:, 2], Z_FLOOR + self.margin, 5.5, out=pos[:, 2])

    # ------------------------------------------------------------------- steps
    def _substep(self, dt: float) -> None:
        self._update_active()
        self._maybe_rebuild()

        r_ff, e_ff, r_fb, e_fb = self._pair_geometry(self.position)
        accel = self._explicit_accel(r_ff, e_ff, r_fb, e_fb)

        self.velocity += dt * accel
        self.velocity[~self.active] = 0.0
        speed = np.sqrt(np.einsum("ij,ij->i", self.velocity, self.velocity))
        over = speed > self.max_speed
        if np.any(over):
            self.velocity[over] *= (self.max_speed / speed[over])[:, None]

        pred = self.position + dt * self.velocity
        pred[~self.active] = self.position[~self.active]
        self._project_to_domain(pred)

        self._density_projection(pred)

        new_velocity = (pred - self.position) / dt
        new_velocity[~self.active] = 0.0
        self.velocity = new_velocity
        speed = np.sqrt(np.einsum("ij,ij->i", self.velocity, self.velocity))
        over = speed > self.max_speed
        if np.any(over):
            self.velocity[over] *= (self.max_speed / speed[over])[:, None]

        self.position = pred
        if not np.all(np.isfinite(self.position)):
            self.position = np.nan_to_num(self.position, nan=0.0, posinf=0.0, neginf=0.0)
            self._project_to_domain(self.position)
        if not np.all(np.isfinite(self.velocity)):
            self.velocity = np.nan_to_num(self.velocity, nan=0.0, posinf=0.0, neginf=0.0)
        self.steps += 1

    def advance_to(self, time: float):
        target = float(time)
        if target > self.time + 1e-12:
            remaining = target - self.time
            count = max(1, int(np.ceil(remaining / self.dt_sub - 1e-9)))
            dt = remaining / count
            for _ in range(count):
                self._substep(dt)
                self.time += dt
            self.time = target
        self._update_active()
        return self._make_frame(target)

    # ------------------------------------------------------------ diagnostics
    def _grid_fields(self):
        dx = self.grid_dx
        reach = self.splat_reach
        radius = self.splat_radius
        nx, ny, nz = self.nx, self.ny, self.nz
        cells = nx * ny * nz

        base = np.rint((self.position - self.grid_origin) / dx).astype(np.int64)
        offsets = np.arange(-reach, reach + 1)
        ox, oy, oz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
        stencil = np.stack((ox.ravel(), oy.ravel(), oz.ravel()), axis=1)
        stencil_size = stencil.shape[0]

        idx = base[:, None, :] + stencil[None, :, :]
        valid = (
            (idx[:, :, 0] >= 0) & (idx[:, :, 0] < nx)
            & (idx[:, :, 1] >= 0) & (idx[:, :, 1] < ny)
            & (idx[:, :, 2] >= 0) & (idx[:, :, 2] < nz)
        )
        pid = np.repeat(np.arange(self.n), stencil_size).reshape(self.n, stencil_size)
        pid = pid[valid]
        idx = idx[valid]
        flat = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]

        centre = self.grid_origin[None, :] + idx * dx
        diff = centre - self.position[pid]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        q2 = dist2 / (radius * radius)
        keep = q2 < 1.0
        flat = flat[keep]
        pid = pid[keep]
        weight = (1.0 - q2[keep]) ** 3

        wsum = np.bincount(flat, weights=weight, minlength=cells)
        xbar = np.empty((cells, 3))
        for k in range(3):
            xbar[:, k] = np.bincount(
                flat, weights=weight * self.position[pid, k], minlength=cells
            )
        rbar = np.bincount(
            flat, weights=weight * self.particle_radius, minlength=cells
        )
        vsum = np.empty((cells, 3))
        for k in range(3):
            vsum[:, k] = np.bincount(
                flat, weights=weight * self.velocity[pid, k], minlength=cells
            )

        occupied = wsum > 1e-12
        safe = np.where(occupied, wsum, 1.0)
        xbar /= safe[:, None]
        rbar = rbar / safe
        vsum /= safe[:, None]

        centres = self.cell_centers.reshape(cells, 3)
        delta = centres - xbar
        phi_zb = np.sqrt(np.einsum("ij,ij->i", delta, delta)) - rbar
        phi_zb = np.where(occupied, phi_zb, radius)

        velocity_grid = np.where(occupied[:, None], vsum, 0.0)
        velocity_grid = velocity_grid.reshape(nx, ny, nz, 3)

        inside = (occupied & (phi_zb < 0.0)).reshape(nx, ny, nz)
        phi_zb = phi_zb.reshape(nx, ny, nz)
        occupied_grid = occupied.reshape(nx, ny, nz)

        if _ndimage is not None and inside.any() and not inside.all():
            sampling = (dx, dx, dx)
            d_out = _ndimage.distance_transform_edt(~inside, sampling=sampling)
            d_in = _ndimage.distance_transform_edt(inside, sampling=sampling)
            phi_far = d_out - d_in
        else:
            phi_far = np.where(inside, -dx, dx).astype(float)

        band = occupied_grid & (np.abs(phi_far) <= 1.2 * dx)
        level_set = np.where(band, phi_zb, phi_far)
        level_set = np.clip(np.nan_to_num(level_set, nan=radius), -1.0, 4.0)

        active = self.container_mask & (level_set <= -0.5 * dx)
        return level_set, velocity_grid, active

    def _make_frame(self, time: float) -> StateFrame:
        level_set, velocity_grid, active = self._grid_fields()

        mass_quantity = np.zeros((self.n, self.phase_count), dtype=float)
        for k in range(self.phase_count):
            sel = self.phase == k
            mass_quantity[sel, k] = self.mass[sel]

        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(self.position, dtype=float),
            velocity=np.ascontiguousarray(self.velocity, dtype=float),
            mass=np.ascontiguousarray(self.mass, dtype=float),
            body_id=self.phase.astype(np.int32),
            attributes={
                "phase_id": self.phase.astype(float),
                "density": self.density0.copy(),
                "release_time": self.release_time.copy(),
                "target_volume": self.volume.copy(),
                "volume_quantity": self.volume.copy(),
                "mass_quantity": mass_quantity,
                "active_flag": self.active.astype(float),
            },
            fields={
                "liquid_level_set": level_set.astype(np.float32),
                "velocity_field": velocity_grid.astype(np.float32),
                "active_mask": active.astype(np.float32),
                "cumulative_mass_source": np.zeros(self.phase_count, dtype=float),
                "cumulative_volume_source": np.zeros(self.phase_count, dtype=float),
            },
            scalars={
                "characteristic_length": float(self.characteristic_length),
                "grid_spacing": float(self.grid_dx),
                "reference_velocity": float(np.sqrt(9.81 * self.characteristic_length)),
                "heavy_phase_mass": float(self.phase_mass[0]),
                "light_phase_mass": float(self.phase_mass[1]),
                "substeps": float(self.steps),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "multiphase position-based incompressible SPH (self-implemented)",
            "particles": int(self.n),
            "boundary_particles": int(len(self.boundary)),
            "particle_spacing": float(self.d),
            "kernel_support": float(self.h),
            "reference_number_density": float(self.n0),
            "substep_dt": float(self.dt_sub),
            "projection_iterations": int(self.iterations),
            "substeps": int(self.steps),
            "grid_shape": [int(self.nx), int(self.ny), int(self.nz)],
            "grid_spacing": float(self.grid_dx),
            "final_time": float(self.time),
        }
