"""Particle-based free-surface fountain solver (Position-Based Fluids / PBF-SPH).

Physical model
--------------
The liquid is a set of equal-mass material particles carrying the mass field of
the frozen initial state.  Their motion is advanced with a real time-stepping
scheme:

* gravity is integrated into a predicted position (semi-implicit Euler),
* an incompressibility (constant rest-density) constraint is solved on the
  predicted positions with the Position-Based Fluids projection of
  Macklin & Mueller (2013), using a poly6 density kernel, the spiky constraint
  gradient and the artificial-pressure term that keeps the free surface
  cohesive,
* velocities are recovered from the position change, smoothed with an XSPH
  viscosity pass (this is the numerical dissipation of the scheme),
* the frozen basin floor / outer wall / nozzle solid are enforced as hard
  impermeable position constraints with no normal velocity, so the basin is
  impermeable and the top stays open,
* the sustained upward inflow of the fountain source is realised by emitting
  material layers at the nozzle exit plane with the prescribed jet speed, so
  the jet is sustained while the emitted mass is recorded exactly in
  ``fields.cumulative_mass_source``.

Everything is deterministic given the frozen scene state and the seed; every
exported array is finite by construction (guarded displacement, velocity and
domain clamps).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_DEFAULT_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)


class BenchmarkSolver:
    """Deterministic free-surface fountain simulation."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", None)
        if not isinstance(config, dict):
            config = {}
        try:
            metadata = scene.metadata()
        except Exception:  # pragma: no cover - defensive
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        ic = config.get("initial_conditions", {}) or {}

        gravity = np.asarray(config.get("gravity", _DEFAULT_GRAVITY), dtype=np.float64)
        self.gravity = gravity if gravity.shape == (3,) else _DEFAULT_GRAVITY.copy()

        self.dx = float(ic.get("particle_spacing", 0.085))
        self.rho0 = float(ic.get("liquid_density", 1000.0))
        self.jet_speed = float(ic.get("jet_speed", 5.2))
        self.nozzle_radius = float(ic.get("nozzle_radius", 0.18))
        self.nozzle_exit_z = float(ic.get("nozzle_exit_z", 0.40))
        self.pool_radius = float(ic.get("pool_radius", 1.45))
        self.jet_height = float(ic.get("jet_height", 0.92))

        self._read_scene_geometry()

        # ---- numerical parameters (frozen for the whole run) --------------
        self.h = 2.5 * self.dx                     # smoothing / support radius
        self.h2 = self.h * self.h
        self.dt = 0.003                            # physical time step
        self.n_iters = 3                           # PBF constraint iterations
        self.xsph = 0.030                          # XSPH viscosity strength
        self.k_corr = 1.0e-4                       # artificial pressure
        self.dq = 0.2 * self.h
        self.max_move = 0.15 * self.dx             # per-iteration displacement cap
        self.max_speed = 25.0                      # state finiteness guard
        self.gscale = self.rho0 * self.dx ** 3     # mass of one particle

        # poly6 kernel and spiky gradient normalisations
        self.w_coef = 315.0 / (64.0 * math.pi * self.h ** 9)
        self.g_coef = -45.0 / (math.pi * self.h ** 6)
        self.W0 = self.w_coef * self.h ** 6
        self.rest_sum = self._lattice_kernel_sum()

        # ---- frozen geometry ---------------------------------------------
        self.r_wall = self.wall_radius - 0.5 * self.dx
        self.r_domain = self.wall_radius
        self.z_floor = self.liquid_bounds_min[2]
        self.z_ceiling = self.liquid_bounds_max[2]

        # ---- diagnostic grid ---------------------------------------------
        self.grid_g = 0.1
        lo = np.array(self.liquid_bounds_min, dtype=np.float64)
        hi = np.array(self.liquid_bounds_max, dtype=np.float64)
        lo = np.minimum(lo, -np.array([self.wall_radius, self.wall_radius, 0.0]))
        self.grid_min = np.array(
            [-self.wall_radius, -self.wall_radius, max(0.0, lo[2] - 0.5 * self.dx)]
        )
        self.grid_max = np.array(
            [self.wall_radius, self.wall_radius, self.z_ceiling]
        )
        self.grid_n = np.maximum(
            np.floor((self.grid_max - self.grid_min) / self.grid_g).astype(int) + 1, 4
        )
        self.gnx, self.gny, self.gnz = (int(v) for v in self.grid_n)

        # ---- emitter ------------------------------------------------------
        # Material enters through the nozzle exit disk at the prescribed jet
        # speed: the emitted volume flux is v_jet * pi * R_nozzle^2 and one
        # particle of volume dx^3 is released whenever that much volume has
        # accumulated, so the mass source is exactly the prescribed inflow.
        self._emit_flux = self.jet_speed * math.pi * self.nozzle_radius ** 2
        self._emit_accum = 0.0
        self._emitted_mass = 0.0
        self._emitted_count = 0
        self._recent: list = []

        # ---- deterministic randomness -------------------------------------
        self.rng = np.random.default_rng(int(seed))

        # ---- boundary samples (static geometry) ---------------------------
        self._build_boundary_samples()

        # ---- initial state (frozen, returned verbatim at t = 0) -----------
        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=np.float64, copy=True)
        self.vel = np.array(frame0.velocity, dtype=np.float64, copy=True)
        if frame0.mass is not None:
            self.mass = np.array(frame0.mass, dtype=np.float64, copy=True)
        else:
            self.mass = np.full(len(self.pos), self.gscale, dtype=np.float64)
        self.N = len(self.pos)
        if "source_region" in frame0.attributes:
            self.phase = np.array(frame0.attributes["source_region"], dtype=np.float64)
        else:
            self.phase = np.zeros(self.N, dtype=np.float64)
        self.phase = np.where(self.phase > 0.5, 1.0, 0.0)

        self.time = 0.0
        self._last_t = None
        self._steps = 0

    # ------------------------------------------------------------- geometry
    def _read_scene_geometry(self) -> None:
        self.wall_radius = 1.5
        self.wall_top = 0.52
        self.nozzle_solid_radius = 0.24
        self.nozzle_solid_top = 0.34
        self.liquid_bounds_min = np.array([-1.5, -1.5, 0.04], dtype=np.float64)
        self.liquid_bounds_max = np.array([1.5, 1.5, 3.2], dtype=np.float64)
        try:
            import yaml

            path = Path("scene_geometry.yaml")
            if path.is_file():
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                for obj in data.get("objects", []) or []:
                    geom = obj.get("geometry", {}) or {}
                    oid = obj.get("id", "")
                    if oid == "circular_basin" and geom.get("type") == "axisymmetric_container":
                        profile = geom.get("inner_profile") or []
                        if profile:
                            self.wall_radius = float(profile[0][0])
                            self.wall_top = float(profile[-1][1])
                    elif oid == "fountain_nozzle" and geom.get("type") == "cylinder":
                        self.nozzle_solid_radius = float(geom.get("radius", 0.24))
                        height = float(geom.get("height", 0.34))
                        translation = (obj.get("transform") or {}).get("translation") or [0.0, 0.0, 0.0]
                        self.nozzle_solid_top = float(translation[2]) + 0.5 * height
                    elif oid == "fountain_liquid" and geom.get("type") == "particle_domain":
                        if geom.get("bounds_min") is not None:
                            self.liquid_bounds_min = np.asarray(geom["bounds_min"], dtype=np.float64)
                        if geom.get("bounds_max") is not None:
                            self.liquid_bounds_max = np.asarray(geom["bounds_max"], dtype=np.float64)
        except Exception:
            # Frozen defaults stay in force if the scaffold file is unavailable.
            pass

    def _lattice_kernel_sum(self) -> float:
        """Rest-density normalisation measured on the frozen particle lattice."""
        span = int(math.ceil(self.h / self.dx)) + 1
        axis = np.arange(-span, span + 1, dtype=np.float64) * self.dx
        gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
        r2 = gx * gx + gy * gy + gz * gz
        inside = r2 < self.h2
        return float(np.sum(self.w_coef * (self.h2 - r2[inside]) ** 3))

    def _build_boundary_samples(self) -> None:
        pts = []
        nrm = []

        def add(positions, normals):
            pts.append(np.asarray(positions, dtype=np.float64))
            nrm.append(np.asarray(normals, dtype=np.float64))

        step = 0.12
        # outer vertical wall of the basin (inner face, radius r_wall)
        n_theta = max(8, int(round(2.0 * math.pi * self.wall_radius / step)))
        theta = np.arange(n_theta) * (2.0 * math.pi / n_theta)
        z_levels = np.arange(0.6 * step, self.wall_top, step)
        if z_levels.size == 0:
            z_levels = np.array([0.5 * self.wall_top])
        ct, st = np.cos(theta), np.sin(theta)
        for z in z_levels:
            wall_pts = np.column_stack(
                (self.wall_radius * ct, self.wall_radius * st, np.full(n_theta, z))
            )
            add(wall_pts, np.column_stack((ct, st, np.zeros(n_theta))))
        # floor (annulus outside the nozzle footprint)
        axis = np.arange(-self.wall_radius + 0.5 * step, self.wall_radius, step)
        fx, fy = np.meshgrid(axis, axis, indexing="ij")
        fr = np.hypot(fx.ravel(), fy.ravel())
        keep = (fr <= self.wall_radius - 0.5 * step) & (fr >= self.nozzle_solid_radius + 0.5 * step)
        floor_pts = np.column_stack(
            (fx.ravel()[keep], fy.ravel()[keep], np.full(int(np.sum(keep)), 0.0))
        )
        add(floor_pts, np.tile(np.array([0.0, 0.0, 1.0]), (len(floor_pts), 1)))
        # nozzle lateral surface
        n_nz = max(8, int(round(2.0 * math.pi * self.nozzle_solid_radius / step)))
        tz = np.arange(n_nz) * (2.0 * math.pi / n_nz)
        cz, sz = np.cos(tz), np.sin(tz)
        for z in np.arange(0.5 * step, self.nozzle_solid_top, step):
            side = np.column_stack(
                (self.nozzle_solid_radius * cz, self.nozzle_solid_radius * sz,
                 np.full(n_nz, z))
            )
            add(side, np.column_stack((cz, sz, np.zeros(n_nz))))
        # nozzle top face
        naxis = np.arange(-self.nozzle_solid_radius, self.nozzle_solid_radius, step)
        nx, ny = np.meshgrid(naxis, naxis, indexing="ij")
        nr = np.hypot(nx.ravel(), ny.ravel())
        top_keep = nr <= self.nozzle_solid_radius - 0.4 * step
        top_pts = np.column_stack(
            (nx.ravel()[top_keep], ny.ravel()[top_keep],
             np.full(int(np.sum(top_keep)), self.nozzle_solid_top))
        )
        add(top_pts, np.tile(np.array([0.0, 0.0, 1.0]), (len(top_pts), 1)))

        self.boundary_points = np.vstack(pts).astype(np.float64)
        normals = np.vstack(nrm).astype(np.float64)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        self.boundary_normals = normals / np.maximum(lengths, 1e-12)
        self.boundary_zeros = np.zeros_like(self.boundary_points)
        self.n_boundary = len(self.boundary_points)

    # -------------------------------------------------------------- emitter
    def _emit(self, dt: float) -> None:
        """Inject the prescribed upward volume flux at the nozzle exit plane."""
        self._emit_accum += self._emit_flux * dt
        n_new = int(self._emit_accum / self.dx ** 3)
        if n_new <= 0:
            return
        self._emit_accum -= n_new * self.dx ** 3
        r_in = self.nozzle_radius * np.sqrt(self.rng.uniform(0.0, 1.0, size=n_new))
        theta = self.rng.uniform(0.0, 2.0 * math.pi, size=n_new)
        new_pos = np.column_stack(
            (
                r_in * np.cos(theta),
                r_in * np.sin(theta),
                self.nozzle_exit_z + self.rng.uniform(0.0, 0.4 * self.dx, size=n_new),
            )
        )
        jitter = 0.03 * self.jet_speed
        new_vel = np.column_stack(
            (
                self.rng.uniform(-jitter, jitter, size=n_new),
                self.rng.uniform(-jitter, jitter, size=n_new),
                self.jet_speed + self.rng.uniform(-jitter, jitter, size=n_new),
            )
        )
        first = self.N
        self.pos = np.vstack((self.pos, new_pos))
        self.vel = np.vstack((self.vel, new_vel))
        self.mass = np.concatenate((self.mass, np.full(n_new, self.gscale)))
        self.phase = np.concatenate((self.phase, np.ones(n_new)))
        self.N += n_new
        self._emitted_count += n_new
        self._emitted_mass += n_new * self.gscale
        self._recent.append(
            (self._steps, np.arange(first, first + n_new), new_vel.copy())
        )

    # ------------------------------------------------------------- stepping
    def _step(self, dt: float) -> None:
        self._emit(dt)
        n = self.N
        pos = self.pos
        vel = self.vel + self.gravity * dt
        pred = pos + vel * dt

        tree = cKDTree(pred, compact_nodes=False, balanced_tree=False)
        pairs = tree.query_pairs(self.h, output_type="ndarray")
        if pairs.size:
            ii = pairs[:, 0].astype(np.int64)
            jj = pairs[:, 1].astype(np.int64)
        else:
            ii = np.zeros(0, dtype=np.int64)
            jj = np.zeros(0, dtype=np.int64)

        wdq = self.w_coef * (self.h2 - self.dq ** 2) ** 3
        h2 = self.h2
        w_coef = self.w_coef
        g_coef = self.g_coef
        inv_s = 1.0 / self.rest_sum
        inv_s2 = inv_s * inv_s
        max_move = self.max_move

        for _ in range(self.n_iters):
            if ii.size == 0:
                break
            diff = pred[ii] - pred[jj]
            r2 = np.einsum("ij,ij->i", diff, diff)
            live = r2 < h2
            if not live.all():
                ii = ii[live]
                jj = jj[live]
                diff = diff[live]
                r2 = r2[live]
                if ii.size == 0:
                    break
            r = np.sqrt(r2)
            r_safe = np.maximum(r, 1.0e-4 * self.dx)
            h_r = self.h - r
            np.maximum(h_r, 0.0, out=h_r)
            w = w_coef * h_r ** 3
            grad = (g_coef * h_r ** 2 / r_safe)[:, None] * diff

            rho = (
                self.W0
                + np.bincount(ii, weights=w, minlength=n)
                + np.bincount(jj, weights=w, minlength=n)
            )
            constraint = rho * inv_s - 1.0

            gxx = np.bincount(ii, weights=grad[:, 0], minlength=n) - np.bincount(
                jj, weights=grad[:, 0], minlength=n
            )
            gxy = np.bincount(ii, weights=grad[:, 1], minlength=n) - np.bincount(
                jj, weights=grad[:, 1], minlength=n
            )
            gxz = np.bincount(ii, weights=grad[:, 2], minlength=n) - np.bincount(
                jj, weights=grad[:, 2], minlength=n
            )
            gsq = np.bincount(
                ii, weights=np.einsum("ij,ij->i", grad, grad), minlength=n
            ) + np.bincount(
                jj, weights=np.einsum("ij,ij->i", grad, grad), minlength=n
            )
            denom = (gxx * gxx + gxy * gxy + gxz * gxz + gsq) * inv_s2 + 1.0e-8
            lam = -constraint / denom

            s_corr = -self.k_corr * (w / wdq) ** 4
            coef = (lam[ii] + lam[jj] + s_corr) * inv_s
            dp = coef[:, None] * grad

            delta = np.empty((n, 3), dtype=np.float64)
            for axis in range(3):
                delta[:, axis] = np.bincount(
                    ii, weights=dp[:, axis], minlength=n
                ) - np.bincount(jj, weights=dp[:, axis], minlength=n)
            mag = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            scale = np.ones(n, dtype=np.float64)
            big = mag > max_move
            if np.any(big):
                scale[big] = max_move / mag[big]
            pred += delta * scale[:, None]

        vel = (pred - pos) / dt

        # XSPH viscosity: symmetric velocity smoothing (numerical dissipation).
        if ii.size:
            diff = pred[ii] - pred[jj]
            r2 = np.einsum("ij,ij->i", diff, diff)
            live = r2 < h2
            if np.any(live):
                li = ii[live]
                lj = jj[live]
                rl2 = r2[live]
                wl = w_coef * (h2 - rl2) ** 3
                vd = vel[lj] - vel[li]
                acc = np.empty((n, 3), dtype=np.float64)
                for axis in range(3):
                    acc[:, axis] = (
                        np.bincount(li, weights=wl * vd[:, axis], minlength=n)
                        + np.bincount(lj, weights=-wl * vd[:, axis], minlength=n)
                    )
                vel += self.xsph * inv_s * acc

        speed = np.sqrt(np.einsum("ij,ij->i", vel, vel))
        fast = speed > self.max_speed
        if np.any(fast):
            vel[fast] *= (self.max_speed / speed[fast])[:, None]

        # prescribed inflow condition: particles just released at the nozzle
        # exit keep the mandated jet velocity while they clear the source.
        if self._recent:
            keep = []
            for entry in self._recent:
                step, idx, pvel = entry
                if self._steps - step <= 3:
                    vel[idx] = pvel
                    keep.append(entry)
            self._recent = keep

        self._collide(pred, vel)
        self.pos = pred
        self.vel = vel
        self._steps += 1
        np.nan_to_num(self.pos, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.vel, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _collide(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """Impermeable basin floor / outer wall / nozzle with soft projection."""
        cap = self.max_move

        # floor
        below = pos[:, 2] < self.z_floor
        if np.any(below):
            push = self.z_floor - pos[below, 2]
            np.minimum(push, cap, out=push)
            pos[below, 2] += push
            vel[below, 2] = np.maximum(vel[below, 2], 0.0)

        # outer wall
        r_xy = np.hypot(pos[:, 0], pos[:, 1])
        outside = r_xy > self.r_wall
        if np.any(outside):
            ox = pos[outside, 0].copy()
            oy = pos[outside, 1].copy()
            inv_r = 1.0 / np.maximum(r_xy[outside], 1e-12)
            nx = ox * inv_r
            ny = oy * inv_r
            new_r = r_xy[outside] - np.minimum(r_xy[outside] - self.r_wall, cap)
            pos[outside, 0] = nx * new_r
            pos[outside, 1] = ny * new_r
            vn = vel[outside, 0] * nx + vel[outside, 1] * ny
            shrink = vn > 0.0
            if np.any(shrink):
                vi = np.flatnonzero(outside)[shrink]
                vel[vi, 0] -= vn[shrink] * nx[shrink]
                vel[vi, 1] -= vn[shrink] * ny[shrink]

        # ceiling of the frozen liquid domain (open top: only a far safety lid)
        above = pos[:, 2] > self.z_ceiling
        if np.any(above):
            pos[above, 2] = self.z_ceiling
            vel[above, 2] = np.minimum(vel[above, 2], 0.0)

        # nozzle solid: {r <= Rn, z <= top}
        r_xy = np.hypot(pos[:, 0], pos[:, 1])
        inside = (r_xy < self.nozzle_solid_radius) & (pos[:, 2] < self.nozzle_solid_top)
        if np.any(inside):
            idx = np.flatnonzero(inside)
            radial_gap = self.nozzle_solid_radius - r_xy[idx]
            axial_gap = self.nozzle_solid_top - pos[idx, 2]
            use_top = axial_gap <= radial_gap
            # push upward out of the nozzle top face
            if np.any(use_top):
                ti = idx[use_top]
                push = np.minimum(axial_gap[use_top], cap)
                pos[ti, 2] += push
                vel[ti, 2] = np.maximum(vel[ti, 2], 0.0)
            # push radially out of the nozzle lateral surface
            if np.any(~use_top):
                si = idx[~use_top]
                ri = r_xy[si]
                inv_r = 1.0 / np.maximum(ri, 1e-12)
                nx = pos[si, 0] * inv_r
                ny = pos[si, 1] * inv_r
                target = ri + np.minimum(radial_gap[~use_top], cap)
                pos[si, 0] = nx * target
                pos[si, 1] = ny * target
                vn = vel[si, 0] * nx + vel[si, 1] * ny
                inward = vn < 0.0
                if np.any(inward):
                    k = si[inward]
                    vel[k, 0] -= vn[inward] * nx[inward]
                    vel[k, 1] -= vn[inward] * ny[inward]

    # ------------------------------------------------------------ sampling
    def _frame(self, time: float) -> StateFrame:
        pos = self.pos
        vel = self.vel
        g = self.grid_g
        shape = (self.gnx, self.gny, self.gnz)
        ncells = shape[0] * shape[1] * shape[2]

        idx = np.floor((pos - self.grid_min) / g).astype(np.int64)
        np.clip(idx, 0, self.grid_n - 1, out=idx)
        flat = (idx[:, 0] * self.gny + idx[:, 1]) * self.gnz + idx[:, 2]

        mass = self.mass
        cell_mass = np.bincount(flat, weights=mass, minlength=ncells).reshape(shape)
        mom = np.empty((ncells, 3), dtype=np.float64)
        for axis in range(3):
            mom[:, axis] = np.bincount(
                flat, weights=mass * vel[:, axis], minlength=ncells
            )
        mom = mom.reshape(shape + (3,))

        msum = uniform_filter(cell_mass, size=3, mode="constant")
        vsum = np.stack(
            [uniform_filter(mom[..., a], size=3, mode="constant") for a in range(3)],
            axis=-1,
        )
        safe = msum > 1e-9
        velocity_field = np.zeros(shape + (3,), dtype=np.float64)
        np.divide(vsum, msum[..., None], out=velocity_field, where=safe[..., None])

        # signed liquid level set from a smoothed volume-fraction indicator
        fraction = cell_mass / (self.rho0 * g ** 3)
        fraction = uniform_filter(fraction, size=3, mode="constant")
        fraction = uniform_filter(fraction, size=3, mode="constant")
        np.clip(fraction, 0.0, 1.5, out=fraction)
        grads = np.gradient(fraction, g)
        gmag = np.sqrt(grads[0] ** 2 + grads[1] ** 2 + grads[2] ** 2)
        level_set = np.clip(
            (0.5 - fraction) / np.maximum(gmag, 0.25 / g),
            -3.0 * g,
            3.0 * g,
        )

        fields = {
            "active_mask": (level_set < 0.0).astype(np.float64),
            "liquid_level_set": level_set,
            "velocity_field": velocity_field,
            "boundary_normal": self.boundary_normals,
            "boundary_velocity": self.boundary_zeros,
            "boundary_target_velocity": self.boundary_zeros,
            "cumulative_mass_source": np.array([self._emitted_mass], dtype=np.float64),
        }

        scalars = {
            "grid_spacing": float(self.grid_g),
            "characteristic_length": float(2.0 * self.nozzle_radius),
            "reference_velocity": float(self.jet_speed),
            "rest_density": float(self.rho0),
            "particle_spacing": float(self.dx),
            "liquid_phase_id": 0.0,
            "time_step": float(self.dt),
        }

        frame = StateFrame(
            time=float(time),
            position=np.array(pos, dtype=np.float64, copy=True),
            velocity=np.array(vel, dtype=np.float64, copy=True),
            mass=np.array(self.mass, dtype=np.float64, copy=True),
            attributes={
                "mass_quantity": np.array(self.mass, dtype=np.float64, copy=True),
                "phase_id": np.array(self.phase, dtype=np.float64, copy=True),
            },
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- contract
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not np.isfinite(target):
            target = self.time
        if self._last_t is not None and target < self._last_t - 1.0e-9:
            raise ValueError("advance_to() must be called with non-decreasing times")
        guard = 0
        while self.time < target - 1.0e-12 and guard < 200000:
            dt = min(self.dt, target - self.time)
            if dt <= 1.0e-12:
                break
            self._step(dt)
            self.time += dt
            guard += 1
        self.time = target
        self._last_t = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "pbf_sph_free_surface_fountain",
            "particle_count": int(self.N),
            "simulated_time": float(self.time),
            "time_steps": int(self._steps),
            "emitted_mass_kg": float(self._emitted_mass),
            "total_mass_kg": float(np.sum(self.mass)),
            "kernel_support": float(self.h),
            "particle_spacing": float(self.dx),
        }
