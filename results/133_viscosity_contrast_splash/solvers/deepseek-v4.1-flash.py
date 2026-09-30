"""Viscosity Contrast Splash -- weakly compressible SPH (WCSPH) solver.

Physical model
--------------
Two matched liquid pools sit side by side inside a rigid basin, separated by a
thin rigid divider.  Above each pool an identical liquid volume (the "drop")
starts at the same height and falls under gravity, so both impacts occur at the
same speed.  The left pair of objects is nearly inviscid (material_label 0/2,
"low viscosity") and the right pair is strongly viscous (material_label 1/3,
"high viscosity"), reproducing the viscosity-contrast splash of the source
figure: a broad, thin, rapidly spreading sheet with filaments on the low
viscosity side versus a thick, cohesive, strongly damped mound on the high
viscosity side.

Numerical method (implemented from scratch with NumPy/SciPy):
  * SPH particles with a cubic-spline kernel, smoothing length h = 1.3 dx,
    uniform particle mass m = rho0 dx^3 (mass is constant => conservation).
  * Density by direct kernel summation; pressure from the Tait equation of
    state p = B ((rho/rho_ref)^7 - 1), clamped to p >= 0 (free surface),
    c = 30 m/s so the flow is nearly incompressible at the relevant speeds.
  * Momentum: pressure gradient + gravity + Monaghan artificial viscosity
    (material dependent, for shock/impact stability) + an explicit laminar
    (Morris) viscous stress with a material dependent kinematic viscosity.
  * Time integration: kick-drift-kick leapfrog with a fixed maximum step,
    sub-stepped so each requested sample time is reached exactly.  The scheme
    is fully deterministic.
  * Rigid walls (ground slab, divider, basin side walls) are handled with an
    exact box signed-distance projection: penetrating samples are pushed back
    onto the wall surface, the normal velocity is made inelastic and the
    tangential velocity is friction damped (approximate no-slip).

Diagnostics exported per sample: collocated velocity grid, liquid level set,
active mask, conserved mass, cumulative mass source and wall samples.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# --- frozen scene constants (from config.yaml / scene_geometry.yaml) --------
DEFAULT_SPACING = 0.14
DEFAULT_DENSITY = 1000.0
DEFAULT_GRID_SHAPE = (24, 16, 28)
DEFAULT_DOMAIN_MIN = np.array([-3.0, -1.2, 0.0])
DEFAULT_DOMAIN_MAX = np.array([3.0, 1.2, 3.8])
GRAVITY = np.array([0.0, 0.0, -9.81])

# --- numerical parameters ---------------------------------------------------
SMOOTH_RATIO = 1.3          # h = SMOOTH_RATIO * dx
SOUND_SPEED = 30.0          # Tait sound speed [m/s]
EOS_GAMMA = 7.0
DT_MAX = 8.0e-4             # largest integration step [s] (CFL ~= 0.16)
ALPHA_LOW = 0.005           # Monaghan artificial viscosity, low viscosity fluid
ALPHA_HIGH = 0.12           # Monaghan artificial viscosity, high viscosity fluid
NU_LOW = 2.0e-4             # kinematic viscosity of the low viscosity liquid
NU_HIGH = 0.25              # kinematic viscosity of the high viscosity liquid
EPS2_FACTOR = 0.01          # r^2 + EPS2_FACTOR*h^2 softening
WALL_FRICTION = 0.08        # tangential velocity damping for wall contacts
WALL_RESTITUTION = 0.0      # inelastic wall response
VMAX = 40.0                 # safety cap on particle speed [m/s]
K_MATERIALS = 2             # conserved material groups (low / high viscosity)

_CSPLINE_SIGMA = 1.0 / np.pi  # 3-D cubic spline normalisation


class BenchmarkSolver:
    """Deterministic WCSPH solver for the viscosity contrast splash scene."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = getattr(scene, "config", None) or {}
        sampling = cfg.get("sampling", {}) or {}
        self.dx = float(sampling.get("spacing", DEFAULT_SPACING))
        self.rho0 = float(sampling.get("density", DEFAULT_DENSITY))

        frame = scene.initial_state()
        position = np.ascontiguousarray(np.asarray(frame.position, dtype=float))
        velocity = np.ascontiguousarray(np.asarray(frame.velocity, dtype=float))
        mass = np.asarray(frame.mass, dtype=float).copy().reshape(-1)
        labels = frame.attributes.get("material_label")
        if labels is None:
            labels = frame.body_id
        labels = np.asarray(labels).astype(np.int64).reshape(-1)

        self.n = position.shape[0]
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.label = labels
        self.mass_p = float(mass[0])
        self.particle_radius = 0.45 * self.dx

        self.h = SMOOTH_RATIO * self.dx
        self.h2 = self.h * self.h
        self.w0 = _CSPLINE_SIGMA / (self.h ** 3)

        # material split: even labels = low viscosity, odd labels = high
        low = (labels % 2) == 0
        self.alpha = np.where(low, ALPHA_LOW, ALPHA_HIGH).astype(float)
        self.nu = np.where(low, NU_LOW, NU_HIGH).astype(float)

        # frozen normalisation quantities
        self.characteristic_length = float(np.max(np.ptp(position, axis=0)))
        if not np.isfinite(self.characteristic_length) or self.characteristic_length <= 0.0:
            self.characteristic_length = self.dx
        self.reference_velocity = 1.0

        # rigid geometry
        self._load_geometry(cfg)
        self._build_wall_samples()

        # density reference from the initial lattice (median interior density)
        rho_init = self._density(position)
        interior = rho_init > 0.6 * float(np.max(rho_init))
        self.rho_ref = float(np.median(rho_init[interior])) if interior.any() else self.rho0
        self.B = self.rho_ref * SOUND_SPEED * SOUND_SPEED / EOS_GAMMA
        self.rho_floor = 0.25 * self.rho_ref

        # diagnostic grid (frozen field_grid_shape over the frozen domain)
        shape = sampling.get("field_grid_shape")
        if shape is not None and len(shape) == 3:
            self.grid_shape = tuple(int(v) for v in shape)
        if self.grid_shape is None:
            self.grid_shape = tuple(int(v) for v in DEFAULT_GRID_SHAPE)
        self.cell = (self.domain_max - self.domain_min) / np.asarray(self.grid_shape, float)
        self.n_full = float(np.prod(self.cell) / self.dx ** 3)
        self.time = 0.0

    # ----------------------------------------------------------- geometry
    def _load_geometry(self, cfg: dict) -> None:
        boxes = []
        self.grid_shape = None
        dmin, dmax = DEFAULT_DOMAIN_MIN.copy(), DEFAULT_DOMAIN_MAX.copy()
        sg = cfg.get("scene_geometry")
        if isinstance(sg, dict):
            for obj in sg.get("objects", []) or []:
                if not obj.get("collision_enabled", True):
                    continue
                geom = obj.get("geometry") or {}
                gtype = geom.get("type")
                if gtype == "box":
                    he = geom.get("half_extents")
                    if he is None:
                        continue
                    tr = (obj.get("transform") or {}).get("translation", [0.0, 0.0, 0.0])
                    boxes.append(
                        (np.asarray(tr, dtype=float), np.abs(np.asarray(he, dtype=float)))
                    )
                elif gtype in {"particle_domain", "voxel_grid"}:
                    bmin = geom.get("bounds_min")
                    bmax = geom.get("bounds_max")
                    if bmin is not None and bmax is not None:
                        dmin = np.asarray(bmin, dtype=float)
                        dmax = np.asarray(bmax, dtype=float)
                    gs = geom.get("grid_shape")
                    if gs is not None and len(gs) == 3:
                        self.grid_shape = tuple(int(v) for v in gs)
        if not boxes:
            boxes = [
                (np.array([0.0, 0.0, -0.05]), np.array([3.2, 1.4, 0.05])),
                (np.array([0.0, 0.0, 0.65]), np.array([0.08, 0.9, 0.65])),
            ]
        self.boxes = boxes
        self.domain_min = dmin
        self.domain_max = dmax

    @staticmethod
    def _box_sdf_and_normal(point: np.ndarray, center: np.ndarray, he: np.ndarray):
        d = point - center
        q = np.abs(d) - he
        outside = np.maximum(q, 0.0)
        dist_out = np.linalg.norm(outside, axis=1)
        inside = np.maximum(q, axis=1) < 0.0
        sdf = dist_out + np.minimum(np.max(q, axis=1), 0.0)
        normals = np.zeros_like(d)
        safe = dist_out > 1e-12
        normals[safe] = outside[safe] / dist_out[safe, None]
        if np.any(inside) or np.any(~safe):
            axis = np.argmax(q, axis=1)
            rows = np.arange(len(point))
            fallback = np.zeros_like(d)
            fallback[rows, axis] = np.where(d[rows, axis] >= 0.0, 1.0, -1.0)
            use_fallback = inside | (~safe)
            normals[use_fallback] = fallback[use_fallback]
        return sdf, normals

    def _boxes_sdf(self, point: np.ndarray) -> np.ndarray:
        if not self.boxes:
            return np.full(len(point), 1e9)
        out = np.full(len(point), 1e9)
        for center, he in self.boxes:
            sdf, _ = self._box_sdf_and_normal(point, center, he)
            out = np.minimum(out, sdf)
        return out

    # -------------------------------------------------------- wall samples
    def _build_wall_samples(self) -> None:
        r_p = 0.5 * self.dx * 0.95
        samples: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        spacing = 0.28
        lo, hi = self.domain_min, self.domain_max
        for center, he in self.boxes:
            for axis in range(3):
                for sgn in (1.0, -1.0):
                    other = [a for a in range(3) if a != axis]
                    axes_pts = []
                    for a in other:
                        a_lo = center[a] - he[a]
                        a_hi = center[a] + he[a]
                        k = max(2, int(np.floor((a_hi - a_lo) / spacing)))
                        axes_pts.append(a_lo + (np.arange(k) + 0.5) * (a_hi - a_lo) / k)
                    g0, g1 = np.meshgrid(axes_pts[0], axes_pts[1], indexing="ij")
                    pts = np.empty((g0.size, 3))
                    pts[:, axis] = center[axis] + sgn * (he[axis] + 0.02)
                    pts[:, other[0]] = g0.ravel()
                    pts[:, other[1]] = g1.ravel()
                    nrm = np.zeros((len(pts), 3))
                    nrm[:, axis] = sgn
                    samples.append(pts)
                    normals.append(nrm)
        pts = np.concatenate(samples, axis=0)
        nrm = np.concatenate(normals, axis=0)
        # keep only samples inside the frozen liquid domain, on the fluid side
        margin = np.array([0.06, 0.06, 0.02])
        keep = np.all(pts >= lo + margin, axis=1) & np.all(pts <= hi - margin, axis=1)
        pts, nrm = pts[keep], nrm[keep]
        if len(pts):
            outside_all = np.ones(len(pts), dtype=bool)
            for center, he in self.boxes:
                sdf, _ = self._box_sdf_and_normal(pts, center, he)
                outside_all &= sdf > 0.005
            pts, nrm = pts[outside_all], nrm[outside_all]
        if len(pts) == 0:
            pts = np.array([[0.0, 0.0, 0.02]])
            nrm = np.array([[0.0, 0.0, 1.0]])
        self.wall_points = np.ascontiguousarray(pts)
        self.wall_normals = np.ascontiguousarray(nrm)
        self.wall_targets = np.zeros_like(self.wall_points)
        self.wall_radius = max(0.75 * self.dx, r_p * 1.4)

    # -------------------------------------------------------------- kernel
    def _neighbours(self, position: np.ndarray):
        tree = cKDTree(position)
        pairs = tree.query_pairs(2.0 * self.h, output_type="ndarray")
        if pairs.size == 0:
            empty_i = np.zeros(0, dtype=np.int64)
            return (
                empty_i,
                empty_i.copy(),
                np.zeros((0, 3)),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
                np.zeros(0),
            )
        i = pairs[:, 0].astype(np.int64)
        j = pairs[:, 1].astype(np.int64)
        rij = position[i] - position[j]
        r2 = np.einsum("ij,ij->i", rij, rij)
        np.maximum(r2, 1e-12, out=r2)
        r = np.sqrt(r2)
        q = r / self.h
        W = np.zeros_like(r)
        dW = np.zeros_like(r)
        m1 = q < 1.0
        if m1.any():
            qq = q[m1]
            W[m1] = _CSPLINE_SIGMA / self.h ** 3 * (1.0 - 1.5 * qq ** 2 + 0.75 * qq ** 3)
            dW[m1] = _CSPLINE_SIGMA / self.h ** 4 * (-3.0 * qq + 2.25 * qq ** 2)
        m2 = (~m1) & (q < 2.0)
        if m2.any():
            qq = 2.0 - q[m2]
            W[m2] = _CSPLINE_SIGMA / self.h ** 3 * 0.25 * qq ** 3
            dW[m2] = _CSPLINE_SIGMA / self.h ** 4 * (-0.75 * qq ** 2)
        return i, j, rij, r2, r, W, dW

    def _density(self, position: np.ndarray) -> np.ndarray:
        i, j, _rij, _r2, _r, W, _dW = self._neighbours(position)
        rho = np.full(self.n, self.mass_p * self.w0)
        if i.size:
            rho += np.bincount(i, weights=self.mass_p * W, minlength=self.n)
            rho += np.bincount(j, weights=self.mass_p * W, minlength=self.n)
        return rho

    # ---------------------------------------------------------- forces
    def _accelerations(self, position: np.ndarray, velocity: np.ndarray):
        n = self.n
        i, j, rij, r2, r, W, dW = self._neighbours(position)
        rho = np.full(n, self.mass_p * self.w0)
        if i.size:
            mW = self.mass_p * W
            rho += np.bincount(i, weights=mW, minlength=n)
            rho += np.bincount(j, weights=mW, minlength=n)
        rho_safe = np.maximum(rho, self.rho_floor)
        p = self.B * ((rho_safe / self.rho_ref) ** EOS_GAMMA - 1.0)
        np.maximum(p, 0.0, out=p)

        acc = np.zeros((n, 3))
        if i.size:
            inv = 1.0 / r
            gW = dW[:, None] * (rij * inv[:, None])
            # pressure gradient (symmetric, momentum conserving)
            press = p / (rho_safe ** 2)
            coef_p = -self.mass_p * (press[i] + press[j])
            w = coef_p[:, None] * gW
            for d in range(3):
                acc[:, d] += np.bincount(i, weights=w[:, d], minlength=n)
                acc[:, d] -= np.bincount(j, weights=w[:, d], minlength=n)
            # Monaghan artificial viscosity (impact / shock stabilisation)
            vij = velocity[i] - velocity[j]
            vr = np.einsum("ij,ij->i", vij, rij)
            mu = self.h * vr / (r2 + EPS2_FACTOR * self.h2)
            alpha_ij = 0.5 * (self.alpha[i] + self.alpha[j])
            rho_bar = np.maximum(0.5 * (rho_safe[i] + rho_safe[j]), 1e-6)
            pi_ij = np.where(vr < 0.0, -alpha_ij * SOUND_SPEED * mu / rho_bar, 0.0)
            coef_av = -self.mass_p * pi_ij
            w = coef_av[:, None] * gW
            for d in range(3):
                acc[:, d] += np.bincount(i, weights=w[:, d], minlength=n)
                acc[:, d] -= np.bincount(j, weights=w[:, d], minlength=n)
            # explicit laminar (Morris) viscous stress, material dependent
            rvw = np.einsum("ij,ij->i", rij, gW)  # = dW * r  (negative)
            denom = r2 + EPS2_FACTOR * self.h2
            nu_ij = 0.5 * (self.nu[i] + self.nu[j])
            base = 2.0 * nu_ij * rvw / denom
            c_i = (self.mass_p / rho_safe[j]) * base
            c_j = (self.mass_p / rho_safe[i]) * base
            dv = velocity[i] - velocity[j]
            f_i = c_i[:, None] * dv
            f_j = -c_j[:, None] * dv
            for d in range(3):
                acc[:, d] += np.bincount(i, weights=f_i[:, d], minlength=n)
                acc[:, d] += np.bincount(j, weights=f_j[:, d], minlength=n)
            del gW
        acc += GRAVITY[None, :]
        return acc, rho

    # -------------------------------------------------------- boundaries
    def _apply_boundaries(self, position: np.ndarray, velocity: np.ndarray):
        r_p = 0.5 * self.dx * 0.95
        for center, he in self.boxes:
            sdf, normals = self._box_sdf_and_normal(position, center, he)
            contact = sdf < r_p
            if not np.any(contact):
                continue
            idx = np.nonzero(contact)[0]
            nrm = normals[idx]
            push = (r_p - sdf[idx])[:, None] * nrm
            position[idx] += push
            vn = np.einsum("ij,ij->i", velocity[idx], nrm)
            moving_in = vn < 0.0
            if np.any(moving_in):
                sub = idx[moving_in]
                velocity[sub] -= (1.0 + WALL_RESTITUTION) * vn[moving_in, None] * nrm[moving_in]
            # tangential friction (approximate no-slip)
            vn2 = np.einsum("ij,ij->i", velocity[idx], nrm)
            vt = velocity[idx] - vn2[:, None] * nrm
            velocity[idx] = velocity[idx] - WALL_FRICTION * vt
        # basin side walls (frozen particle-domain bounds) and lid
        walls = (
            (0, self.domain_min[0], 1.0),
            (0, self.domain_max[0], -1.0),
            (1, self.domain_min[1], 1.0),
            (1, self.domain_max[1], -1.0),
        )
        for axis, coord, sgn in walls:
            sdf = (position[:, axis] - coord) * sgn
            contact = sdf > -r_p
            if not np.any(contact):
                continue
            idx = np.nonzero(contact)[0]
            position[idx, axis] -= sgn * (sdf[idx] + r_p)
            vn = velocity[idx, axis] * sgn
            moving_in = vn > 0.0
            if np.any(moving_in):
                sub = idx[moving_in]
                velocity[sub, axis] -= (1.0 + WALL_RESTITUTION) * vn[moving_in] * sgn
            vt = velocity[idx].copy()
            vt[:, axis] = 0.0
            velocity[idx] -= WALL_FRICTION * vt
        return position, velocity

    # ------------------------------------------------------------ stepping
    def _step(self, dt: float) -> None:
        position = self.position
        velocity = self.velocity
        acc, _rho = self._accelerations(position, velocity)
        velocity += (0.5 * dt) * acc
        position = position + dt * velocity
        position, velocity = self._apply_boundaries(position, velocity)
        acc, _rho = self._accelerations(position, velocity)
        velocity += (0.5 * dt) * acc
        speed = np.linalg.norm(velocity, axis=1)
        too_fast = speed > VMAX
        if np.any(too_fast):
            velocity[too_fast] *= (VMAX / speed[too_fast])[:, None]
        self.position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
        self.velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

    def advance_to(self, time: float):
        target = float(time)
        remaining = target - self.time
        if remaining > 1e-12:
            steps = int(np.ceil(remaining / DT_MAX))
            steps = max(1, steps)
            dt = remaining / steps
            for _ in range(steps):
                self._step(dt)
        self.time = target
        return self._frame()

    # -------------------------------------------------------------- output
    def _diagnostic_fields(self, position: np.ndarray, velocity: np.ndarray):
        nx, ny, nz = self.grid_shape
        lo = self.domain_min
        cell = self.cell
        xs = lo[0] + (np.arange(nx) + 0.5) * cell[0]
        ys = lo[1] + (np.arange(ny) + 0.5) * cell[1]
        zs = lo[2] + (np.arange(nz) + 0.5) * cell[2]
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        centers = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
        m_cells = nx * ny * nz

        wsum = np.zeros(m_cells)
        vsum = np.zeros((m_cells, 3))
        g = (position - lo) / cell
        base = np.floor(g).astype(np.int64)
        base = np.clip(base, 0, np.array([nx - 2, ny - 2, nz - 2]))
        frac = np.clip(g - base, 0.0, 1.0)
        for corner in range(8):
            off = np.array([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1])
            weight = np.ones(len(position))
            for axis in range(3):
                weight *= frac[:, axis] if off[axis] else (1.0 - frac[:, axis])
            idx = (
                (base[:, 0] + off[0]) * ny * nz
                + (base[:, 1] + off[1]) * nz
                + (base[:, 2] + off[2])
            )
            np.add.at(wsum, idx, weight)
            np.add.at(vsum, idx, weight[:, None] * velocity)

        velocity_field = np.zeros((m_cells, 3))
        good = wsum > 1e-9
        velocity_field[good] = vsum[good] / wsum[good, None]
        velocity_field = velocity_field.reshape(nx, ny, nz, 3)

        inside = wsum >= 0.75 * self.n_full
        mask = (wsum >= 0.5 * self.n_full).astype(float).reshape(nx, ny, nz)

        tree = cKDTree(position)
        dist, _idx = tree.query(centers)
        half = 0.5 * self.dx
        level = np.where(inside, -(dist + half), np.maximum(dist - half, 0.0))
        level = np.clip(level, -1.0, 1.0).reshape(nx, ny, nz)

        return velocity_field, mask, level

    def _boundary_velocity(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        out = np.zeros_like(self.wall_points)
        if len(position) == 0:
            return out
        tree = cKDTree(position)
        lists = tree.query_ball_point(self.wall_points, self.wall_radius)
        for k, neighbours in enumerate(lists):
            if len(neighbours) == 0:
                continue
            idx = np.asarray(neighbours, dtype=np.int64)
            out[k] = velocity[idx].mean(axis=0)
        return out

    def _frame(self) -> StateFrame:
        position = np.ascontiguousarray(self.position)
        velocity = np.ascontiguousarray(self.velocity)
        velocity_field, mask, level = self._diagnostic_fields(position, velocity)
        fields = {
            "velocity_field": velocity_field,
            "active_mask": mask,
            "liquid_level_set": level,
            "boundary_normal": self.wall_normals.copy(),
            "boundary_target_velocity": self.wall_targets.copy(),
            "boundary_velocity": self._boundary_velocity(position, velocity),
            "cumulative_mass_source": np.zeros(K_MATERIALS),
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "material_label": self.label.copy(),
            "particle_radius": np.full(self.n, self.particle_radius),
        }
        scalars = {
            "contact_length_scale": float(self.dx),
            "grid_spacing": float(self.dx),
            "characteristic_length": float(self.characteristic_length),
            "reference_velocity": float(self.reference_velocity),
        }
        frame = StateFrame(
            float(self.time),
            position.copy(),
            velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.label.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "WCSPH (Tait EOS, Morris viscosity, Monaghan AV)",
            "particles": int(self.n),
            "smoothing_length": float(self.h),
            "sound_speed": float(SOUND_SPEED),
            "viscosity_low": float(NU_LOW),
            "viscosity_high": float(NU_HIGH),
            "dt_max": float(DT_MAX),
            "wall_samples": int(len(self.wall_points)),
            "final_time": float(self.time),
        }
