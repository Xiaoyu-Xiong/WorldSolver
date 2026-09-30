"""Self-implemented smoke solver: buoyant plume pumped by a rotating cube.

Physical model
-------------
Incompressible inviscid gas (Boussinesq buoyancy) on a collocated 3D grid that
matches the frozen ``subject`` voxel grid (24 x 16 x 28 over
[-3,-1.2,0]..[3,1.2,3.8]).  Each sub-step performs

  1. kinematic update of the rotating cube obstacle (prescribed omega about y),
  2. semi-Lagrangian advection of velocity and of the transported smoke scalar,
  3. buoyancy body force + mild drag + vorticity confinement,
  4. prescribed emitter source (scalar injection + weak vertical jet),
  5. solid-boundary enforcement (closed box walls + moving cube),
  6. pressure projection: face fluxes -> masked Neumann Poisson solve (PCG) ->
     flux correction -> interpolation back to cell centres,
  7. passive advection of the sampled material particles.

Only NumPy / SciPy are used.  No external physics engine, no scripted motion:
the plume shape, the wrap-around, and the trailing vortices all emerge from the
time stepping.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

# ----------------------------------------------------------------------------
# physical / numerical parameters
# ----------------------------------------------------------------------------
CUBE_OMEGA = 3.0          # rad/s, prescribed rotation about +y
BUOYANCY = 5.0            # m/s^2 per unit smoke concentration
LINEAR_DRAG = 0.06        # 1/s
VORT_EPS = 1.1            # vorticity-confinement strength (dimensionless)
SOURCE_RATE = 2.6         # smoke concentration per second in emitter cells
JET_SPEED = 1.25          # m/s vertical inflow at the emitter
JET_RELAX = 9.0           # 1/s relaxation of emitter momentum
SCALAR_DISSIPATION = 0.02  # 1/s
SCALAR_MAX = 2.2
SPEED_CLAMP = 8.0         # m/s hard clamp (guarantees finite states)
SUBSTEPS = 4              # sub-steps per sample interval
CG_MAX_IT = 200
CG_TOL = 1e-6


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        config = dict(scene.config)
        self.sample_dt = float(config["sample_dt"])

        geometry = config.get("scene_geometry") or {}
        objects = {o["id"]: o for o in geometry.get("objects", [])}

        subject = objects.get("subject", {}).get("geometry", {})
        self.lo = np.asarray(subject.get("bounds_min", [-3.0, -1.2, 0.0]), float)
        self.hi = np.asarray(subject.get("bounds_max", [3.0, 1.2, 3.8]), float)
        shape = tuple(int(s) for s in subject.get("grid_shape",
                      config["sampling"]["field_grid_shape"]))
        self.shape = shape
        self.nx, self.ny, self.nz = shape

        # node-centred axes (identical to the frozen initial-condition axes)
        self.xs = np.linspace(self.lo[0], self.hi[0], self.nx)
        self.ys = np.linspace(self.lo[1], self.hi[1], self.ny)
        self.zs = np.linspace(self.lo[2], self.hi[2], self.nz)
        self.dx = float(self.xs[1] - self.xs[0])
        self.dy = float(self.ys[1] - self.ys[0])
        self.dz = float(self.zs[1] - self.zs[0])
        self.h = float((self.dx * self.dy * self.dz) ** (1.0 / 3.0))

        X = self.xs[:, None, None]
        Y = self.ys[None, :, None]
        Z = self.zs[None, None, :]
        self.X = np.broadcast_to(X, shape).copy()
        self.Y = np.broadcast_to(Y, shape).copy()
        self.Z = np.broadcast_to(Z, shape).copy()
        self.I, self.J, self.K = np.meshgrid(
            np.arange(self.nx, dtype=float),
            np.arange(self.ny, dtype=float),
            np.arange(self.nz, dtype=float),
            indexing="ij",
        )

        # rotating cube obstacle -------------------------------------------
        cube = objects.get("rotating_cube", {})
        self.cube_center = np.asarray(
            cube.get("transform", {}).get("translation", [0.0, 0.0, 1.55]), float)
        self.cube_half = np.asarray(
            cube.get("geometry", {}).get("half_extents", [0.45, 0.45, 0.45]), float)
        self.omega_vec = np.array([0.0, CUBE_OMEGA, 0.0])
        self.theta = 0.0

        # emitter ----------------------------------------------------------
        emitter = objects.get("smoke_source", {})
        e_geom = emitter.get("geometry", {})
        self.emit_center = np.asarray(
            emitter.get("transform", {}).get("translation", [0.0, 0.0, 0.06]), float)
        self.emit_radius = float(e_geom.get("radius", 0.34))
        self.emit_height = float(e_geom.get("height", 0.12))

        # closed-box wall cells (outermost node layer on every face)
        wall = np.zeros(shape, dtype=bool)
        wall[0, :, :] = True
        wall[-1, :, :] = True
        wall[:, 0, :] = True
        wall[:, -1, :] = True
        wall[:, :, 0] = True
        wall[:, :, -1] = True
        self.wall = wall

        # emitter source field (prescribed, time independent shape)
        rad = np.hypot(self.X - self.emit_center[0], self.Y - self.emit_center[1])
        weight = np.zeros(shape)
        band = np.clip(1.0 - np.maximum(
            self.Z - (self.emit_center[2] + 0.5 * self.emit_height), 0.0) / (2.0 * self.dz), 0.0, 1.0)
        inside = (rad <= self.emit_radius + 0.5 * max(self.dx, self.dy)) & (~wall)
        weight[inside] = band[inside]
        weight[self.Z > self.emit_center[2] + 0.5 * self.emit_height + 2.0 * self.dz] = 0.0
        if weight.max() <= 0.0:                          # degenerate fallback
            weight[self.nx // 2, self.ny // 2, 1] = 1.0
        self.emit_weight = weight
        self.scalar_source = SOURCE_RATE * weight

        # state ------------------------------------------------------------
        init = scene.initial_state()
        self.frozen_scalars = {k: float(v) for k, v in init.scalars.items()}
        self.grid_spacing = float(self.frozen_scalars.get("grid_spacing", self.h))
        self.u = np.zeros(shape + (3,))
        scalar0 = np.asarray(init.fields.get("transported_scalar",
                                             np.zeros(shape)), float).copy()
        if scalar0.shape != shape:
            scalar0 = np.zeros(shape)
        self.s = np.clip(scalar0, 0.0, SCALAR_MAX)

        solid, usol = self._solid_state(0.0)
        self.s[solid] = 0.0
        self.u[solid] = usol[solid]

        # sampled material particles (passive smoke tracers)
        self.p_pos = np.asarray(init.position, float).copy()
        self.p_vel = np.zeros_like(self.p_pos)
        self.p_mass = (np.asarray(init.mass, float).copy()
                       if init.mass is not None else np.ones(len(self.p_pos)))
        self.p_body = (np.asarray(init.body_id).astype(np.int32).copy()
                       if init.body_id is not None
                       else np.zeros(len(self.p_pos), np.int32))
        self.p_attr = {}
        for name, value in init.attributes.items():
            self.p_attr[name] = np.asarray(value).copy()
        self.cumulative_mass_source = np.asarray(
            init.fields.get("cumulative_mass_source",
                            np.zeros(len(self.p_pos))), float).copy()

        self._build_boundary_samples()
        self.time = 0.0
        self.steps = 0
        self.cg_iters = 0
        self.max_speed = 0.0

    # -------------------------------------------------------------- geometry
    def _rotation(self, theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def _solid_state(self, theta: float):
        """Solid mask (walls + rotated cube) and prescribed solid velocity."""
        R = self._rotation(theta)
        rel = np.stack((self.X - self.cube_center[0],
                        self.Y - self.cube_center[1],
                        self.Z - self.cube_center[2]), axis=-1)
        body = rel @ R                      # rel @ R == (R^T rel^T)^T
        cube = np.all(np.abs(body) <= self.cube_half[None, None, None, :], axis=-1)
        solid = cube | self.wall
        usol = np.zeros(self.shape + (3,))
        if cube.any():
            vel = np.cross(np.broadcast_to(self.omega_vec, rel.shape), rel)
            usol[cube] = vel[cube]
        usol[self.wall] = 0.0
        return solid, usol

    def _build_boundary_samples(self):
        """Fixed wall samples (closed box) + rotating-cube surface samples."""
        pts, nrm = [], []
        sy = max(1, self.ny // 8)
        sz = max(1, self.nz // 14)
        sx = max(1, self.nx // 12)
        idx = []
        for j in range(0, self.ny, sy):
            for k in range(0, self.nz, sz):
                idx.append((0, j, k, (1.0, 0.0, 0.0)))
                idx.append((self.nx - 1, j, k, (-1.0, 0.0, 0.0)))
        for i in range(0, self.nx, sx):
            for k in range(0, self.nz, sz):
                idx.append((i, 0, k, (0.0, 1.0, 0.0)))
                idx.append((i, self.ny - 1, k, (0.0, -1.0, 0.0)))
        for i in range(0, self.nx, sx):
            for j in range(0, self.ny, sy):
                idx.append((i, j, 0, (0.0, 0.0, 1.0)))
                idx.append((i, j, self.nz - 1, (0.0, 0.0, -1.0)))
        self.wall_idx = np.array([(i, j, k) for i, j, k, _ in idx], dtype=int)
        for i, j, k, n in idx:
            pts.append((self.xs[i], self.ys[j], self.zs[k]))
            nrm.append(n)
        self.wall_points = np.asarray(pts, float)
        self.wall_normals = np.asarray(nrm, float)

        # cube surface samples in body frame
        t = np.array([-0.3, 0.0, 0.3])
        bp, bn = [], []
        for axis in range(3):
            for sign in (-1.0, 1.0):
                for a in t:
                    for b in t:
                        p = np.zeros(3)
                        p[axis] = sign * self.cube_half[axis]
                        other = [q for q in range(3) if q != axis]
                        p[other[0]] = a * 2.0 * self.cube_half[other[0]]
                        p[other[1]] = b * 2.0 * self.cube_half[other[1]]
                        n = np.zeros(3)
                        n[axis] = sign
                        bp.append(p)
                        bn.append(n)
        self.cube_body_points = np.asarray(bp, float)
        self.cube_body_normals = np.asarray(bn, float)
        self.n_boundary = len(self.wall_points) + len(self.cube_body_points)

    # ------------------------------------------------------- interpolation
    def _interp_vector(self, points: np.ndarray) -> np.ndarray:
        """Trilinear sample of the collocated velocity field at world points."""
        ci = (points[:, 0] - self.lo[0]) / self.dx
        cj = (points[:, 1] - self.lo[1]) / self.dy
        ck = (points[:, 2] - self.lo[2]) / self.dz
        coords = np.stack((np.clip(ci, 0.0, self.nx - 1.0),
                           np.clip(cj, 0.0, self.ny - 1.0),
                           np.clip(ck, 0.0, self.nz - 1.0)))
        out = np.empty((len(points), 3))
        for c in range(3):
            out[:, c] = map_coordinates(self.u[..., c], coords, order=1,
                                        mode="nearest")
        return out

    # ------------------------------------------------------------ projection
    def _project(self, solid, usol):
        fluid = ~solid
        idx2 = 1.0 / self.dx ** 2
        idy2 = 1.0 / self.dy ** 2
        idz2 = 1.0 / self.dz ** 2

        ox = fluid[:-1, :, :] & fluid[1:, :, :]
        oy = fluid[:, :-1, :] & fluid[:, 1:, :]
        oz = fluid[:, :, :-1] & fluid[:, :, 1:]

        ux, uy, uz = self.u[..., 0], self.u[..., 1], self.u[..., 2]
        sx, sy, sz = usol[..., 0], usol[..., 1], usol[..., 2]

        fx = np.where(ox, 0.5 * (ux[:-1] + ux[1:]),
                      np.where(solid[:-1, :, :], sx[:-1], sx[1:]))
        fy = np.where(oy, 0.5 * (uy[:, :-1] + uy[:, 1:]),
                      np.where(solid[:, :-1, :], sy[:, :-1], sy[:, 1:]))
        fz = np.where(oz, 0.5 * (uz[:, :, :-1] + uz[:, :, 1:]),
                      np.where(solid[:, :, :-1], sz[:, :, :-1], sz[:, :, 1:]))

        # divergence from staggered face fluxes; outer domain faces are solid
        # walls and therefore carry zero normal flux.
        div = np.zeros(self.shape)
        fxf = np.zeros((self.nx + 1, self.ny, self.nz))
        fyf = np.zeros((self.nx, self.ny + 1, self.nz))
        fzf = np.zeros((self.nx, self.ny, self.nz + 1))
        fxf[1:-1] = fx
        fyf[:, 1:-1] = fy
        fzf[:, :, 1:-1] = fz
        div += (fxf[1:] - fxf[:-1]) / self.dx
        div += (fyf[:, 1:] - fyf[:, :-1]) / self.dy
        div += (fzf[:, :, 1:] - fzf[:, :, :-1]) / self.dz
        div[solid] = 0.0

        nfluid = int(fluid.sum())
        if nfluid == 0:
            return
        rhs = div.copy()
        rhs[fluid] -= rhs[fluid].mean()      # Neumann compatibility

        diag = np.zeros(self.shape)
        diag[:-1] += ox * idx2
        diag[1:] += ox * idx2
        diag[:, :-1] += oy * idy2
        diag[:, 1:] += oy * idy2
        diag[:, :, :-1] += oz * idz2
        diag[:, :, 1:] += oz * idz2
        invdiag = np.where(diag > 0.0, 1.0 / np.maximum(diag, 1e-30), 0.0)
        invdiag[solid] = 0.0

        def apply_neg_L(p):
            out = np.zeros(self.shape)
            gx = ox * (p[1:] - p[:-1]) * idx2
            out[:-1] -= gx
            out[1:] += gx
            gy = oy * (p[:, 1:] - p[:, :-1]) * idy2
            out[:, :-1] -= gy
            out[:, 1:] += gy
            gz = oz * (p[:, :, 1:] - p[:, :, :-1]) * idz2
            out[:, :, :-1] -= gz
            out[:, :, 1:] += gz
            out[solid] = 0.0
            return out

        b = -rhs
        b[solid] = 0.0
        p = np.zeros(self.shape)
        r = b.copy()
        znew = r * invdiag
        d = znew.copy()
        rz = float(np.sum(r * znew))
        b_norm = float(np.sqrt(np.sum(b * b)))
        it = 0
        if b_norm > 1e-14 and rz > 0.0:
            tol = CG_TOL * b_norm
            for it in range(1, CG_MAX_IT + 1):
                Ad = apply_neg_L(d)
                dAd = float(np.sum(d * Ad))
                if not np.isfinite(dAd) or abs(dAd) < 1e-30:
                    break
                alpha = rz / dAd
                p += alpha * d
                r -= alpha * Ad
                if float(np.sqrt(np.sum(r * r))) <= tol:
                    break
                znew = r * invdiag
                rz_new = float(np.sum(r * znew))
                if not np.isfinite(rz_new):
                    break
                d = znew + (rz_new / rz) * d
                rz = rz_new
        self.cg_iters += it
        if not np.all(np.isfinite(p)):
            p = np.zeros(self.shape)
        p[solid] = 0.0

        fx = fx - ox * (p[1:] - p[:-1]) / self.dx
        fy = fy - oy * (p[:, 1:] - p[:, :-1]) / self.dy
        fz = fz - oz * (p[:, :, 1:] - p[:, :, :-1]) / self.dz

        fxf[:] = 0.0
        fyf[:] = 0.0
        fzf[:] = 0.0
        fxf[1:-1] = fx
        fyf[:, 1:-1] = fy
        fzf[:, :, 1:-1] = fz
        # solid faces on the outer boundary carry zero normal flux already
        self.u[..., 0] = 0.5 * (fxf[1:] + fxf[:-1])
        self.u[..., 1] = 0.5 * (fyf[:, 1:] + fyf[:, :-1])
        self.u[..., 2] = 0.5 * (fzf[:, :, 1:] + fzf[:, :, :-1])
        self.u[solid] = usol[solid]

    # ----------------------------------------------------------- one substep
    def _advect(self, dt):
        bi = np.clip(self.I - dt * self.u[..., 0] / self.dx, 0.0, self.nx - 1.0)
        bj = np.clip(self.J - dt * self.u[..., 1] / self.dy, 0.0, self.ny - 1.0)
        bk = np.clip(self.K - dt * self.u[..., 2] / self.dz, 0.0, self.nz - 1.0)
        coords = np.stack((bi, bj, bk))
        new_u = np.empty_like(self.u)
        for c in range(3):
            new_u[..., c] = map_coordinates(self.u[..., c], coords, order=1,
                                            mode="nearest")
        self.s = map_coordinates(self.s, coords, order=1, mode="nearest")
        self.u = new_u

    def _vorticity_confinement(self, dt, fluid):
        ux, uy, uz = self.u[..., 0], self.u[..., 1], self.u[..., 2]
        duz_dy, duy_dz = np.gradient(uz, self.dy, axis=1), np.gradient(uy, self.dz, axis=2)
        dux_dz, duz_dx = np.gradient(ux, self.dz, axis=2), np.gradient(uz, self.dx, axis=0)
        duy_dx, dux_dy = np.gradient(uy, self.dx, axis=0), np.gradient(ux, self.dy, axis=1)
        w = np.stack((duz_dy - duy_dz, dux_dz - duz_dx, duy_dx - dux_dy), axis=-1)
        mag = np.sqrt(np.sum(w * w, axis=-1))
        gx = np.gradient(mag, self.dx, axis=0)
        gy = np.gradient(mag, self.dy, axis=1)
        gz = np.gradient(mag, self.dz, axis=2)
        g = np.stack((gx, gy, gz), axis=-1)
        gn = np.sqrt(np.sum(g * g, axis=-1))[..., None]
        N = g / np.maximum(gn, 1e-8)
        f = VORT_EPS * self.h * np.cross(N, w)
        fmag = np.sqrt(np.sum(f * f, axis=-1))[..., None]
        f = f * np.minimum(1.0, 20.0 / np.maximum(fmag, 1e-12))
        self.u[fluid] += dt * f[fluid]

    def _step(self, dt):
        self.theta += CUBE_OMEGA * dt
        solid, usol = self._solid_state(self.theta)
        fluid = ~solid

        self._advect(dt)

        # buoyancy + drag
        self.u[..., 2] += dt * BUOYANCY * np.clip(self.s, 0.0, SCALAR_MAX)
        self.u /= (1.0 + dt * LINEAR_DRAG)

        self._vorticity_confinement(dt, fluid)

        # prescribed emitter: scalar injection and weak vertical jet
        self.s = self.s + dt * self.scalar_source
        k = np.clip(dt * JET_RELAX, 0.0, 1.0) * self.emit_weight
        self.u[..., 2] += k * (JET_SPEED - self.u[..., 2])

        # dissipation and bounds on the transported scalar
        self.s *= np.exp(-SCALAR_DISSIPATION * dt)
        self.s = np.clip(self.s, 0.0, SCALAR_MAX)
        self.s[solid] = 0.0

        # boundary conditions then incompressibility
        self.u[solid] = usol[solid]
        speed = np.sqrt(np.sum(self.u * self.u, axis=-1))
        big = speed > SPEED_CLAMP
        if big.any():
            self.u[big] *= (SPEED_CLAMP / speed[big])[:, None]
        self._project(solid, usol)

        if not np.all(np.isfinite(self.u)):
            self.u = np.nan_to_num(self.u, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(self.s)):
            self.s = np.nan_to_num(self.s, nan=0.0, posinf=0.0, neginf=0.0)

        self._advance_particles(dt, solid)
        self.time += dt
        self.steps += 1
        self.max_speed = max(self.max_speed,
                             float(np.sqrt(np.sum(self.u ** 2, axis=-1)).max()))

    def _advance_particles(self, dt, solid):
        vel = self._interp_vector(self.p_pos)
        self.p_pos = self.p_pos + dt * vel
        # keep inside the closed domain (one cell margin = wall layer)
        margin = np.array([self.dx, self.dy, self.dz]) * 0.6
        self.p_pos = np.clip(self.p_pos, self.lo + margin, self.hi - margin)
        # push tracers out of the rotating cube
        R = self._rotation(self.theta)
        rel = self.p_pos - self.cube_center
        body = rel @ R
        inside = np.all(np.abs(body) <= self.cube_half[None, :], axis=1)
        if inside.any():
            b = body[inside]
            frac = np.abs(b) / self.cube_half[None, :]
            axis = np.argmax(frac, axis=1)
            rows = np.arange(len(b))
            sgn = np.sign(b[rows, axis])
            sgn[sgn == 0.0] = 1.0
            b[rows, axis] = sgn * (self.cube_half[axis] + 0.02)
            body[inside] = b
            self.p_pos = self.cube_center + body @ R.T
            vel[inside] = np.cross(np.broadcast_to(self.omega_vec, rel[inside].shape),
                                   rel[inside])
        self.p_vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        self.p_pos = np.nan_to_num(self.p_pos, nan=0.0, posinf=0.0, neginf=0.0)

    # ---------------------------------------------------------------- output
    def _boundary_arrays(self, solid, usol):
        idx = self.wall_idx
        wall_vel = self.u[idx[:, 0], idx[:, 1], idx[:, 2], :]
        wall_target = np.zeros_like(wall_vel)

        R = self._rotation(self.theta)
        cube_pts = self.cube_center + self.cube_body_points @ R.T
        cube_nrm = self.cube_body_normals @ R.T
        offset = 0.35 * np.array([self.dx, self.dy, self.dz])
        sample_pts = cube_pts + cube_nrm * offset[None, :]
        cube_vel = self._interp_vector(sample_pts)
        rel = cube_pts - self.cube_center
        cube_target = np.cross(np.broadcast_to(self.omega_vec, rel.shape), rel)

        vel = np.concatenate((wall_vel, cube_vel), axis=0)
        target = np.concatenate((wall_target, cube_target), axis=0)
        normal = np.concatenate((self.wall_normals, cube_nrm), axis=0)
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal / np.maximum(norm, 1e-12)
        return (np.nan_to_num(vel), np.nan_to_num(target), np.nan_to_num(normal))

    def _frame(self, time):
        from worldsolver.contracts import StateFrame

        solid, usol = self._solid_state(self.theta)
        vel_b, tgt_b, nrm_b = self._boundary_arrays(solid, usol)
        active = (~solid)

        fields = {
            "transported_scalar": np.ascontiguousarray(self.s, dtype=float),
            "scalar_source": np.ascontiguousarray(self.scalar_source, dtype=float),
            "active_mask": np.ascontiguousarray(active),
            "velocity_field": np.ascontiguousarray(self.u, dtype=float),
            "boundary_velocity": np.ascontiguousarray(vel_b, dtype=float),
            "boundary_target_velocity": np.ascontiguousarray(tgt_b, dtype=float),
            "boundary_normal": np.ascontiguousarray(nrm_b, dtype=float),
            "cumulative_mass_source": np.ascontiguousarray(
                self.cumulative_mass_source, dtype=float),
        }
        scalars = dict(self.frozen_scalars)
        scalars["grid_spacing"] = self.grid_spacing
        scalars.setdefault("characteristic_length", 0.72)
        scalars.setdefault("reference_velocity", 1.0)

        attributes = {k: np.asarray(v) for k, v in self.p_attr.items()}
        frame = StateFrame(
            float(time),
            np.ascontiguousarray(self.p_pos, dtype=float),
            np.ascontiguousarray(self.p_vel, dtype=float),
            mass=np.ascontiguousarray(self.p_mass, dtype=float),
            body_id=np.ascontiguousarray(self.p_body),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # --------------------------------------------------------------- stepping
    def advance_to(self, time: float):
        target = float(time)
        dt_max = self.sample_dt / SUBSTEPS
        guard = 0
        while target - self.time > 1e-12 and guard < 100000:
            dt = min(dt_max, target - self.time)
            if dt <= 1e-12:
                break
            self._step(dt)
            guard += 1
        self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "self-implemented semi-Lagrangian incompressible smoke solver",
            "grid_shape": list(self.shape),
            "cell_size": [self.dx, self.dy, self.dz],
            "substeps_per_sample": SUBSTEPS,
            "steps_taken": int(self.steps),
            "cg_iterations_total": int(self.cg_iters),
            "cube_omega_rad_s": CUBE_OMEGA,
            "max_speed_m_s": float(self.max_speed),
            "final_scalar_sum": float(np.sum(self.s)),
        }
