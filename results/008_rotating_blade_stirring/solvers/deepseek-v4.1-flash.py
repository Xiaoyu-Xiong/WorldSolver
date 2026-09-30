"""Three-dimensional weakly-compressible SPH solver for the rotating-blade tank.

Physical model
--------------
The liquid is a Lagrangian set of samples with fixed mass advanced by a
symplectic (kick-drift) integrator under gravity, a Tait equation-of-state
pressure (weakly compressible, zero in tension), a Monaghan artificial
viscosity shock term and a laminar Morris viscous term carrying the prescribed
kinematic viscosity.

Solid boundaries (tank floor, four side walls, rotating blade) are enforced by
an impermeable free-slip projection: penetrating samples are pushed back onto
the surface and the fluid-side normal velocity relative to the solid is
removed.  Air is not simulated, so the free surface needs no special handling.

The blade is a kinematic actuator: its pose, angular velocity and surface
velocity come from the frozen smoothstep drive in ``scene.blade_kinematics``.
No animation tables, keyframes or precomputed motion are used.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_SIGMA3 = 1.0 / math.pi
_GAMMA = 7.0


def _kernel(q):
    """Cubic-spline profile f, f' and f'' for q = r/h in [0, 2]."""
    q = np.clip(np.asarray(q, dtype=float), 0.0, 2.0)
    w = np.zeros_like(q)
    dw = np.zeros_like(q)
    d2w = np.zeros_like(q)
    inner = q < 1.0
    outer = ~inner
    qi = q[inner]
    w[inner] = 1.0 - 1.5 * qi * qi + 0.75 * qi * qi * qi
    dw[inner] = -3.0 * qi + 2.25 * qi * qi
    d2w[inner] = -3.0 + 4.5 * qi
    t = 2.0 - q[outer]
    w[outer] = 0.25 * t * t * t
    dw[outer] = -0.75 * t * t
    d2w[outer] = 1.5 * t
    return w, dw, d2w


def _scatter(target, i, j, contribution_i, contribution_j):
    """Accumulate symmetric per-pair contributions without slow np.add.at."""
    n = len(target)
    for axis in range(3):
        target[:, axis] += np.bincount(i, weights=contribution_i[:, axis], minlength=n)
        target[:, axis] += np.bincount(j, weights=contribution_j[:, axis], minlength=n)


class BenchmarkSolver:
    """Deterministic 3D SPH liquid driven by a prescribed rotating blade."""

    def __init__(self):
        self.ready = False

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.scene = scene
        self.config = cfg
        self.seed = int(seed)

        liquid = cfg["liquid"]
        tank = cfg["tank"]
        blade = cfg["blade"]
        domain = cfg["domain"]

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.rho0 = float(liquid["density"])
        self.nu = float(liquid["kinematic_viscosity"])
        self.depth = float(liquid["depth"])
        self.tank_half = float(tank["interior_half_width"])
        self.tank_top = float(tank["height"])
        self.floor_half = 0.84
        self.blade_half = np.asarray(blade["half_extents"], dtype=float)

        self.grid_spacing = float(domain["grid_spacing"])
        self.grid_shape = tuple(int(v) for v in domain["reference_grid"])
        self.grid_origin = np.asarray(domain["bounds_min"], dtype=float)
        axes = [
            lo + (np.arange(n) + 0.5) * self.grid_spacing
            for lo, n in zip(self.grid_origin, self.grid_shape, strict=True)
        ]
        self.grid_points = (
            np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
            .reshape(-1, 3)
            .copy()
        )

        # ---- Lagrangian liquid samples on a regular lattice ------------
        hp = 0.05
        self.hp = hp
        half = self.tank_half
        xs = np.arange(-half + hp / 2, half, hp)
        zs = np.arange(hp / 2, self.depth, hp)
        points = np.stack(np.meshgrid(xs, xs, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        keep = scene.blade_signed_distance(points, 0.0) >= 0.5 * hp
        points = np.ascontiguousarray(points[keep])
        count = len(points)
        if count < 32:
            raise ValueError("liquid lattice produced too few samples")

        bh = self.blade_half
        center = np.asarray(blade["center"], dtype=float)
        submerged = float(np.clip(self.depth - (center[2] - bh[2]), 0.0, 2.0 * bh[2]))
        self.fluid_volume = (2.0 * half) ** 2 * self.depth - 4.0 * bh[0] * bh[1] * submerged
        mass = np.full(count, self.rho0 * self.fluid_volume / count)

        self.x = points.copy()
        self.v = np.zeros_like(self.x)
        self.m = mass

        # ---- SPH resolution, reference density and sound speed ---------
        self.hs = 1.35 * hp
        self.support = 2.0 * self.hs
        self.rho_ref = self._lattice_density(self.x, self.m)
        self.c_s = 15.0
        self.B = self.rho_ref * self.c_s * self.c_s / _GAMMA
        self.alpha_av = 0.3
        self.cal = self._calibrate()

        # ---- tank wall sampling template (static geometry) -------------
        step = self.grid_spacing
        wall_xy = np.arange(-half + step / 2, half, step)
        wall_z = np.arange(step / 2, 0.5, step)
        gx, gy = np.meshgrid(wall_xy, wall_xy, indexing="ij")
        positions = [np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])]
        normals = [np.tile([0.0, 0.0, 1.0], (gx.size, 1))]
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                gz, gt = np.meshgrid(wall_z, wall_xy, indexing="ij")
                p = np.zeros((gz.size, 3))
                p[:, axis] = sign * half
                p[:, 1 - axis] = gt.ravel()
                p[:, 2] = gz.ravel()
                n = np.zeros_like(p)
                n[:, axis] = -sign
                positions.append(p)
                normals.append(n)
        self.wall_position = np.concatenate(positions)
        self.wall_normal = np.concatenate(normals)

        # ---- blade surface sampling template (rotated per frame) -------
        blade_samples = scene.blade_surface_samples(0.0, spacing=step)
        self.blade_local_position = (
            (blade_samples["position"] - center) @ scene.blade_kinematics(0.0)["rotation_matrix"]
        )
        self.blade_local_normal = blade_samples["normal"].copy()
        self.blade_center = center

        self.t = 0.0
        self.steps = 0
        self.a = self._accelerations()
        self.ready = True

    # ------------------------------------------------------------------
    # geometry and interpolation helpers
    # ------------------------------------------------------------------
    def _lattice_density(self, x, m) -> float:
        """SPH density plateau of the initial lattice: the EOS reference."""
        tree = cKDTree(x)
        pairs = tree.query_pairs(self.support, output_type="ndarray")
        if len(pairs) == 0:
            raise ValueError("degenerate initial lattice")
        i, j = pairs[:, 0], pairs[:, 1]
        r = np.linalg.norm(x[i] - x[j], axis=1)
        w, _, _ = _kernel(r / self.hs)
        rho = np.bincount(i, weights=m[j] * w, minlength=len(x))
        rho += np.bincount(j, weights=m[i] * w, minlength=len(x))
        big = rho[rho > 0.8 * rho.max()]
        return float(np.median(big))

    def _calibrate(self) -> float:
        """Truncation bias of the k-nearest sampling used for the fields."""
        probe = np.array([[0.3, 0.3, 0.15]])
        weight, _ = self._sample(probe, cKDTree(self.x), k=48)
        return float(max(weight[0] / self.rho_ref, 1e-6))

    def _sample(self, points, tree, k: int = 48):
        """Shepard interpolation: returns (sum m*W, sum m*v*W / sum m*W)."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.size == 0:
            return np.zeros(0), np.zeros((0, 3))
        dist, idx = tree.query(points, k=k, distance_upper_bound=self.support)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        valid = np.isfinite(dist) & (idx < len(self.x))
        safe = np.where(valid, idx, 0)
        q = np.where(valid, dist, 0.0) / self.hs
        w, _, _ = _kernel(q)
        mw = np.where(valid, w * self.m[safe], 0.0)
        weight = mw.sum(axis=1)
        vel = np.einsum("pk,pkd->pd", mw, self.v[safe])
        return weight, vel

    def _blade_sdf(self, points, pose):
        """Exact oriented-box distance and outward unit normal gradient."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.size == 0:
            return np.zeros(0), np.zeros((0, 3))
        local = (points - pose["position"]) @ pose["rotation_matrix"]
        q = np.abs(local) - self.blade_half
        mx = np.maximum(q, 0.0)
        nm = np.linalg.norm(mx, axis=1)
        dist = nm + np.minimum(q.max(axis=1), 0.0)
        sign = np.sign(local)
        sign[sign == 0.0] = 1.0
        rows = np.arange(len(local))
        idx = np.argmax(q, axis=1)
        grad = np.zeros_like(local)
        grad[rows, idx] = sign[rows, idx]
        outside = dist > 0.0
        if outside.any():
            scale = sign[outside] * mx[outside] / np.maximum(nm[outside], 1e-12)[:, None]
            grad[outside] = scale
        grad = grad @ pose["rotation_matrix"].T
        grad /= np.maximum(np.linalg.norm(grad, axis=1), 1e-12)[:, None]
        return dist, grad

    def _blade_samples(self, time):
        """Currently wetted blade samples with rotated positions/normals."""
        pose = self.scene.blade_kinematics(time)
        rot = pose["rotation_matrix"]
        pos = self.blade_local_position @ rot.T + pose["position"]
        nrm = self.blade_local_normal @ rot.T
        return pose, pos, nrm

    def _wetted(self, sample_pos, normal, tree, threshold=0.35):
        """Liquid presence just off a solid face (volume-fraction test)."""
        sample_pos = np.atleast_2d(np.asarray(sample_pos, dtype=float))
        if sample_pos.size == 0:
            return np.zeros(0, dtype=bool)
        probe = sample_pos + normal * (0.6 * self.grid_spacing)
        weight, _ = self._sample(probe, tree, k=48)
        return weight / (self.rho_ref * self.cal) > threshold

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def _accelerations(self):
        x, v, m = self.x, self.v, self.m
        n = len(x)
        acc = np.zeros_like(x)
        acc[:, 2] += self.gravity[2]
        acc[:, 0] += self.gravity[0]
        acc[:, 1] += self.gravity[1]
        pairs = cKDTree(x).query_pairs(self.support, output_type="ndarray")
        if len(pairs) == 0:
            return acc
        i, j = pairs[:, 0], pairs[:, 1]
        dx = x[i] - x[j]
        dv = v[i] - v[j]
        r2 = np.einsum("ij,ij->i", dx, dx)
        r = np.sqrt(np.maximum(r2, 1e-18))
        w, dw, _ = _kernel(r / self.hs)
        rho = np.bincount(i, weights=m[j] * w, minlength=n)
        rho += np.bincount(j, weights=m[i] * w, minlength=n)
        rho = np.maximum(rho, 0.25 * self.rho_ref)
        p = self.B * ((rho / self.rho_ref) ** _GAMMA - 1.0)
        np.maximum(p, 0.0, out=p)
        term = p[i] / (rho[i] * rho[i]) + p[j] / (rho[j] * rho[j])
        # Monaghan artificial viscosity: only approaching pairs, shock capturing.
        mu = self.hs * np.einsum("ij,ij->i", dv, dx) / (r2 + 0.01 * self.hs * self.hs)
        rho_bar = 0.5 * (rho[i] + rho[j])
        art = np.where(mu < 0.0, -self.alpha_av * self.c_s * mu / rho_bar, 0.0)
        kn = _SIGMA3 / self.hs ** 4 * dw / np.maximum(r, 1e-12)
        coef = (term + art) * kn
        grad = dx * coef[:, None]
        _scatter(acc, i, j, -m[j, None] * grad, m[i, None] * grad)
        # laminar (Morris) viscosity with the prescribed kinematic viscosity
        if self.nu > 0.0:
            eta = 0.01 * self.hs * self.hs
            fac = 2.0 * self.nu * np.einsum("ij,ij->i", dv, dx) / (r2 + eta)
            visc = (fac * kn)[:, None] * dx
            _scatter(
                acc,
                i,
                j,
                (m[j] / rho[j])[:, None] * visc,
                -(m[i] / rho[i])[:, None] * visc,
            )
        return acc

    def _apply_solids(self):
        """Impermeable free-slip projection against blade and tank solids."""
        x, v = self.x, self.v
        half = self.tank_half
        # --- rotating blade: push out of the box, kill inward relative flow
        pose = self.scene.blade_kinematics(self.t)
        dist, nrm = self._blade_sdf(x, pose)
        vblade = np.cross(pose["angular_velocity"], x - pose["position"])
        vrel = v - vblade
        vn = np.einsum("ij,ij->i", vrel, nrm)
        inside = dist < 0.0
        if inside.any():
            x[inside] += nrm[inside] * (0.3 * self.hp - dist[inside])[:, None]
        touching = (dist < 0.5 * self.hp) & (vn < 0.0)
        if touching.any():
            vrel[touching] -= vn[touching][:, None] * nrm[touching]
            v = vrel + vblade
            self.v = v
            v = self.v
        # --- tank floor slab (finite extent; outside it liquid falls free)
        on_slab = (np.abs(x[:, 0]) <= self.floor_half) & (np.abs(x[:, 1]) <= self.floor_half)
        hit = on_slab & (x[:, 2] < 0.0)
        if hit.any():
            x[hit, 2] = -x[hit, 2]
            v[hit, 2] = np.abs(v[hit, 2])
        # --- four side walls, only where the wall slab exists
        z_ok = (x[:, 2] >= 0.0) & (x[:, 2] <= self.tank_top)
        for axis in (0, 1):
            within = z_ok & (np.abs(x[:, 1 - axis]) <= self.floor_half)
            over = within & (x[:, axis] > half)
            if over.any():
                x[over, axis] = 2.0 * half - x[over, axis]
                v[over, axis] = -np.abs(v[over, axis])
            under = within & (x[:, axis] < -half)
            if under.any():
                x[under, axis] = -2.0 * half - x[under, axis]
                v[under, axis] = np.abs(v[under, axis])

    def _timestep(self):
        speed = np.linalg.norm(self.v, axis=1).max() if len(self.v) else 0.0
        accel = np.linalg.norm(self.a, axis=1).max() if len(self.a) else 0.0
        dt = 0.4 * self.hs / (self.c_s + float(speed))
        dt = min(dt, 0.25 * math.sqrt(self.hs / max(float(accel), 1e-9)))
        return float(min(max(dt, 1e-5), 2.0e-3))

    def _step(self, dt: float) -> None:
        self.v += self.a * dt
        self.x += self.v * dt
        self._apply_solids()
        self.a = self._accelerations()
        self.steps += 1

    def advance_to(self, time: float) -> StateFrame:
        if not self.ready:
            raise RuntimeError("BenchmarkSolver.initialize must be called first")
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("advance_to requires a finite non-negative time")
        guard = 0
        while self.t < target - 1e-12:
            dt = self._timestep()
            remaining = target - self.t
            if dt > remaining:
                dt = remaining
            if dt <= 0.0:
                break
            self._step(dt)
            self.t += dt
            guard += 1
            if guard > 2_000_000:
                raise RuntimeError("time stepping failed to reach the requested time")
        return self._build_frame(target)

    # ------------------------------------------------------------------
    # diagnostics and frame export
    # ------------------------------------------------------------------
    def _build_frame(self, time: float) -> StateFrame:
        scene = self.scene
        half = self.tank_half
        step = self.grid_spacing
        count = len(self.x)
        tree = cKDTree(self.x)

        pose = scene.blade_kinematics(time)
        position = np.vstack([self.x, pose["position"]])
        velocity = np.vstack([self.v, np.zeros(3)])
        orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (count + 1, 1))
        orientation[-1] = pose["orientation"]
        angular = np.zeros((count + 1, 3))
        angular[-1] = pose["angular_velocity"]
        mass = np.concatenate([self.m, [0.0]])
        body_id = np.concatenate([np.zeros(count, dtype=np.int64), [1]])
        radius = np.concatenate([np.full(count, self.hp / 2), [0.0]])
        volume = np.concatenate([self.m / self.rho0, [0.0]])

        # ---- collocated diagnostic grid -------------------------------
        weight, gvel = self._sample(self.grid_points, tree, k=48)
        frac = (weight / (self.rho_ref * self.cal)).reshape(self.grid_shape)
        vel_field = gvel.reshape(self.grid_shape + (3,))
        grad = np.gradient(frac, step)
        gnorm = np.sqrt(sum(component * component for component in grad))
        phi = np.clip((0.5 - frac) / np.maximum(gnorm, 0.5 / step), -1.0, 1.0)
        blade_dist, _ = self._blade_sdf(self.grid_points, pose)
        blade_dist = blade_dist.reshape(self.grid_shape)
        gx = self.grid_points[:, 0].reshape(self.grid_shape)
        gy = self.grid_points[:, 1].reshape(self.grid_shape)
        gz = self.grid_points[:, 2].reshape(self.grid_shape)
        active = (
            (phi < -0.5 * step)
            & (gz > 0.5 * step)
            & (np.abs(gx) < half - 0.5 * step)
            & (np.abs(gy) < half - 0.5 * step)
            & (blade_dist > 0.5 * step)
        )

        # ---- stationary wall samples -----------------------------------
        wet = self._wetted(self.wall_position, self.wall_normal, tree)
        wall_pos = self.wall_position[wet]
        wall_nrm = self.wall_normal[wet]
        _, wall_vel = self._sample(wall_pos, tree, k=48)

        # ---- moving blade samples --------------------------------------
        _, blade_pos, blade_nrm = self._blade_samples(time)
        wetted = self._wetted(blade_pos, blade_nrm, tree, threshold=0.32)
        blade_pos = blade_pos[wetted]
        blade_nrm = blade_nrm[wetted]
        blade_target = np.cross(
            pose["angular_velocity"], blade_pos - pose["position"]
        )
        _, blade_vel = self._sample(blade_pos, tree, k=48)

        speed = np.linalg.norm(self.v, axis=1)
        frames_fields = {
            "velocity_field": vel_field,
            "liquid_level_set": phi,
            "active_mask": active.astype(float),
            "density_field": np.maximum(frac, 0.0) * self.rho0,
            "cumulative_mass_source": np.zeros(1),
            "boundary_position": wall_pos,
            "boundary_normal": wall_nrm,
            "boundary_velocity": wall_vel,
            "boundary_target_velocity": np.zeros_like(wall_pos),
            "blade_boundary_position": blade_pos,
            "blade_boundary_normal": blade_nrm,
            "blade_boundary_velocity": blade_vel,
            "blade_boundary_target_velocity": blade_target,
            "grid_origin": self.grid_origin,
        }
        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular,
            mass=mass,
            body_id=body_id,
            attributes={
                "mass_quantity": mass,
                "particle_volume": volume,
                "particle_radius": radius,
            },
            fields=frames_fields,
            scalars={
                "grid_spacing": step,
                "characteristic_length": float(2.0 * self.blade_half[0]),
                "reference_velocity": float(abs(scene.config["blade"]["angular_speed"]) * self.blade_half[0]),
                "liquid_volume": float(self.fluid_volume),
                "max_speed": float(speed.max()) if count else 0.0,
                "kinetic_energy": float(0.5 * np.sum(self.m * speed * speed)),
            },
        )
        frame.validate()
        self.last_max_speed = float(speed.max()) if count else 0.0
        return frame

    def diagnostics(self):
        mass = getattr(self, "m", np.zeros(0))
        return {
            "solver": "weakly_compressible_sph_3d",
            "initialized": bool(self.ready),
            "liquid_particles": int(len(mass)),
            "substeps": int(getattr(self, "steps", 0)),
            "simulated_time": float(getattr(self, "t", 0.0)),
            "particle_mass_kg": float(mass[0]) if len(mass) else 0.0,
            "total_liquid_mass_kg": float(mass.sum()) if len(mass) else 0.0,
            "initial_liquid_volume_m3": float(getattr(self, "fluid_volume", 0.0)),
            "sph_smoothing_length_m": float(getattr(self, "hs", 0.0)),
            "sound_speed_m_per_s": float(getattr(self, "c_s", 0.0)),
            "last_max_speed_m_per_s": float(getattr(self, "last_max_speed", 0.0)),
        }
