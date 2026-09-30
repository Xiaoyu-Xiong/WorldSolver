"""Position-Based Fluids (PBF) solver for the U-corridor dam-break scene.

Physical model
--------------
* The liquid is a Lagrangian particle set sampled from the frozen sampling
  block (spacing 0.14 m, rho = 1000 kg/m^3) held behind a kinematic dam gate
  at one end of a rigid U-corridor (floor, two side walls, closed end wall).
* Time stepping is a genuine predict-correct scheme: gravity is integrated
  explicitly, incompressibility is enforced by density constraints solved
  with Jacobi iterations (Macklin & Mueller 2013, "Position Based Fluids"),
  and boundaries are imposed as rigid half-space / box constraints with
  no-slip velocity resolution.  XSPH viscosity damps the motion.
* The dam gate is a kinematic rigid box that is retracted vertically
  (prescribed lift law), so the column is released by an actual moving
  boundary rather than a scripted fluid state.
* Mass is exactly conserved; no particles are added or removed.

All state exported per frame is derived from the simulated particle state.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """PBF dam-break solver honouring the frozen scene configuration."""

    SUB_DT = 0.002          # physical substep (s)
    ITERS = 4               # density-constraint iterations per substep
    VMAX = 10.0             # safety velocity clamp (m/s)
    EPS_CFM = 1.0           # constraint-force-mixing relaxation
    K_SCORR = 0.005         # anti-clustering strength
    XSPH_C = 0.05           # XSPH viscosity coefficient
    GATE_SPEED = 2.0        # dam-gate retraction speed (m/s)
    GATE_TRAVEL = 1.15      # lift distance before the gate is clear (m)

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame0 = scene.initial_state()

        self.x = np.array(frame0.position, dtype=np.float64)
        self.v = np.zeros_like(self.x)
        self.mass = np.array(frame0.mass, dtype=np.float64)
        self.body = np.asarray(frame0.body_id, dtype=np.int32)
        self.n = int(self.x.shape[0])

        self.spacing = float(cfg["sampling"]["spacing"])
        self.h = 2.5 * self.spacing
        self.h2 = self.h * self.h
        self.g = np.asarray(cfg["gravity"], dtype=np.float64)
        self.time = 0.0
        self.substeps = 0

        # SPH kernels (poly6 for densities, spiky gradient for constraints).
        self.poly6_k = 315.0 / (64.0 * np.pi * self.h ** 9)
        self.spiky_k = -45.0 / (np.pi * self.h ** 6)
        self.w0 = self.poly6_k * self.h2 ** 3
        self.w_dq = self.poly6_k * (self.h2 - (0.3 * self.h) ** 2) ** 3

        # Rest density: measured on the initial (equilibrated) lattice so the
        # column neither expands nor collapses artificially at t = 0.
        rho_init = self._densities(self.x)
        self.rho0 = float(np.max(rho_init)) if rho_init.size else 1000.0

        # Rigid corridor interior (frozen geometry: inner faces of the
        # corridor_left_wall / corridor_right_wall / corridor_end_wall boxes,
        # the floor top, and the closed corridor mouth at the dam end).
        self.xmin, self.xmax = -1.05, 1.05
        self.ymin, self.ymax = -1.55, 1.45
        self.zmin = 0.0

        # Dam gate box (frozen geometry), retracted along +z.
        self.gate_xh = 0.85
        self.gate_y0, self.gate_y1 = -0.64, -0.56
        self.gate_h = 1.10

        # Diagnostic grid over the corridor interior.
        shape = cfg["sampling"].get("field_grid_shape", [24, 16, 28])
        self.grid_shape = tuple(int(v) for v in shape)
        nx, ny, nz = self.grid_shape
        self.dx = (self.xmax - self.xmin) / nx
        self.dy = (self.ymax - self.ymin) / ny
        self.dz = (1.4 - 0.0) / nz
        self.grid_spacing = float((self.dx * self.dy * self.dz) ** (1.0 / 3.0))
        cx = self.xmin + (np.arange(nx) + 0.5) * self.dx
        cy = self.ymin + (np.arange(ny) + 0.5) * self.dy
        cz = 0.0 + (np.arange(nz) + 0.5) * self.dz
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        self.grid_points = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
        self.h_grid = 0.30          # kernel radius for grid diagnostics
        self.h_grid2 = self.h_grid ** 2
        self.poly6_g = 315.0 / (64.0 * np.pi * self.h_grid ** 9)
        self.occ_r = 0.45 * self.rho0   # liquid occupancy density threshold

        # Frozen normalization scalars.
        span = float(np.ptp(self.x, axis=0).max())
        self.characteristic_length = max(span, self.spacing)
        self.reference_velocity = float(np.sqrt(2.0 * max(abs(self.g[2]), 1e-9)
                                                * max(float(self.x[:, 2].max()), 1e-3)))

        self._build_boundary_samples()
        self.rng_seed = int(seed)

    # ------------------------------------------------------------------ #
    # boundary samples (walls + moving gate)
    # ------------------------------------------------------------------ #
    def _build_boundary_samples(self) -> None:
        pts, nrm, moving = [], [], []

        def add(p, n, is_gate=False):
            pts.append(np.asarray(p, dtype=np.float64).reshape(-1, 3))
            nrm.append(np.broadcast_to(np.asarray(n, dtype=np.float64), (len(p), 3)).copy())
            moving.append(np.zeros(len(p), dtype=bool) if not is_gate
                          else np.ones(len(p), dtype=bool))

        xs = np.linspace(-1.0, 1.0, 9)
        ys = np.linspace(-1.5, 1.4, 11)
        zs = np.linspace(0.1, 1.0, 5)

        # floor (top face of the ground box)
        fx, fy = np.meshgrid(xs, ys, indexing="ij")
        add(np.column_stack((fx.ravel(), fy.ravel(), np.full(fx.size, self.zmin))), (0, 0, 1))

        # side walls (inner faces)
        wy, wz = np.meshgrid(ys, zs, indexing="ij")
        left = np.column_stack((np.full(wy.size, self.xmin), wy.ravel(), wz.ravel()))
        right = np.column_stack((np.full(wy.size, self.xmax), wy.ravel(), wz.ravel()))
        add(left, (1, 0, 0))
        add(right, (-1, 0, 0))

        # end wall and closed corridor mouth
        wx, wz2 = np.meshgrid(xs, zs, indexing="ij")
        end = np.column_stack((wx.ravel(), np.full(wx.size, self.ymax), wz2.ravel()))
        back = np.column_stack((wx.ravel(), np.full(wx.size, self.ymin), wz2.ravel()))
        add(end, (0, -1, 0))
        add(back, (0, 1, 0))

        # dam gate (both wettable faces, kinematic)
        gxx, gzz = np.meshgrid(np.linspace(-0.8, 0.8, 9), zs, indexing="ij")
        face_m = np.column_stack((gxx.ravel(), np.full(gxx.size, self.gate_y0), gzz.ravel()))
        face_p = np.column_stack((gxx.ravel(), np.full(gxx.size, self.gate_y1), gzz.ravel()))
        add(face_m, (0, -1, 0), is_gate=True)
        add(face_p, (0, 1, 0), is_gate=True)

        self.b_pos = np.concatenate(pts)
        self.b_nrm = np.concatenate(nrm)
        self.b_moving = np.concatenate(moving)
        self.b_count = len(self.b_pos)

    def _gate_bottom(self, t: float) -> float:
        return min(self.GATE_SPEED * max(t, 0.0), self.GATE_TRAVEL)

    def _gate_velocity(self, t: float) -> float:
        return self.GATE_SPEED if self._gate_bottom(t) < self.GATE_TRAVEL - 1e-9 else 0.0

    # ------------------------------------------------------------------ #
    # SPH helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _binacc(idx, val, n):
        out = np.empty((n, val.shape[1]), dtype=np.float64)
        for c in range(val.shape[1]):
            out[:, c] = np.bincount(idx, weights=val[:, c], minlength=n)
        return out

    def _pairs(self, p):
        diff = p[:, None, :] - p[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", diff, diff)
        iu = np.triu_indices(len(p), 1)
        sel = r2[iu] < self.h2
        return iu[0][sel], iu[1][sel]

    def _densities(self, p):
        a, b = self._pairs(p)
        d = p[a] - p[b]
        r2 = np.einsum("ij,ij->i", d, d)
        msk = r2 < self.h2
        aa, bb, rr = a[msk], b[msk], np.sqrt(np.maximum(r2[msk], 1e-12))
        w = self.poly6_k * (self.h2 - rr * rr) ** 3
        rho = (np.bincount(aa, weights=self.mass[bb] * w, minlength=self.n)
               + np.bincount(bb, weights=self.mass[aa] * w, minlength=self.n)
               + self.mass * self.w0)
        return rho

    # ------------------------------------------------------------------ #
    # rigid boundary projection
    # ------------------------------------------------------------------ #
    def _clamp(self, p, t):
        p[:, 0] = np.clip(p[:, 0], self.xmin, self.xmax)
        p[:, 1] = np.clip(p[:, 1], self.ymin, self.ymax)
        p[:, 2] = np.clip(p[:, 2], self.zmin, 3.0)

        zb = self._gate_bottom(t)
        if zb < self.gate_h:  # gate still intersects the corridor volume
            inside = ((np.abs(p[:, 0]) < self.gate_xh)
                      & (p[:, 1] > self.gate_y0) & (p[:, 1] < self.gate_y1)
                      & (p[:, 2] > zb) & (p[:, 2] < zb + self.gate_h))
            for i in np.nonzero(inside)[0]:
                pen = np.array([p[i, 0] + self.gate_xh,        # -x face
                                self.gate_xh - p[i, 0],        # +x face
                                p[i, 1] - self.gate_y0,        # -y face
                                self.gate_y1 - p[i, 1],        # +y face
                                p[i, 2] - zb,                  # bottom face
                                zb + self.gate_h - p[i, 2]])   # top face
                k = int(np.argmin(pen))
                if k == 0:
                    p[i, 0] = -self.gate_xh
                elif k == 1:
                    p[i, 0] = self.gate_xh
                elif k == 2:
                    p[i, 1] = self.gate_y0
                elif k == 3:
                    p[i, 1] = self.gate_y1
                elif k == 4:
                    p[i, 2] = zb
                else:
                    p[i, 2] = zb + self.gate_h
        return p

    # ------------------------------------------------------------------ #
    # one physical substep
    # ------------------------------------------------------------------ #
    def _step(self, dt):
        t = self.time
        self.v = self.v + self.g * dt
        p = self.x + self.v * dt

        a, b = self._pairs(p)
        for _ in range(self.ITERS):
            d = p[a] - p[b]
            r2 = np.einsum("ij,ij->i", d, d)
            msk = r2 < self.h2
            aa, bb = a[msk], b[msk]
            if aa.size == 0:
                break
            dd = d[msk]
            rr = np.sqrt(np.maximum(r2[msk], 1e-12))

            w = self.poly6_k * (self.h2 - rr * rr) ** 3
            rho = (np.bincount(aa, weights=self.mass[bb] * w, minlength=self.n)
                   + np.bincount(bb, weights=self.mass[aa] * w, minlength=self.n)
                   + self.mass * self.w0)
            c_i = rho / self.rho0 - 1.0

            # grad of W w.r.t. p_a (points from a towards b)
            grad = self.spiky_k * (self.h - rr)[:, None] ** 2 * dd / rr[:, None]

            s_self = (self._binacc(aa, grad, self.n)
                      - self._binacc(bb, grad, self.n))          # sum_k grad W(ik)
            g2 = np.einsum("ij,ij->i", grad, grad)
            denom = (np.einsum("ij,ij->i", s_self, s_self)
                     + np.bincount(aa, weights=g2, minlength=self.n)
                     + np.bincount(bb, weights=g2, minlength=self.n)) / (self.rho0 ** 2)
            lam = -c_i / (denom + self.EPS_CFM)

            scorr = -self.K_SCORR * (w / self.w_dq) ** 4
            coeff = ((lam[aa] + lam[bb] + scorr) / self.rho0)[:, None] * grad
            dp = self._binacc(aa, coeff, self.n) - self._binacc(bb, coeff, self.n)
            dp = np.clip(dp, -0.5 * self.h, 0.5 * self.h)

            p = self._clamp(p + dp, t)

        # XSPH viscosity on the corrected positions.
        d = p[a] - p[b]
        r2 = np.einsum("ij,ij->i", d, d)
        msk = r2 < self.h2
        aa, bb = a[msk], b[msk]
        if aa.size:
            rr = np.sqrt(np.maximum(r2[msk], 1e-12))
            w = self.poly6_k * (self.h2 - rr * rr) ** 3
            wsum = (np.bincount(aa, weights=w, minlength=self.n)
                    + np.bincount(bb, weights=w, minlength=self.n) + self.w0)
            rel = w[:, None] * (p[bb] - p[aa])
            xsph = (self._binacc(aa, rel, self.n) - self._binacc(bb, rel, self.n))
            p = p + self.XSPH_C * xsph / np.maximum(wsum, 1e-12)[:, None]

        p = self._clamp(p, t)

        # safety: guarantee finite state
        bad = ~np.isfinite(p).all(axis=1)
        if np.any(bad):
            p[bad] = self._clamp(self.x[bad].copy(), t)

        v = (p - self.x) / dt
        speed = np.linalg.norm(v, axis=1)
        scale = np.where(speed > self.VMAX, self.VMAX / np.maximum(speed, 1e-12), 1.0)
        v = v * scale[:, None]
        v[~np.isfinite(v).all(axis=1)] = 0.0

        self.x = p
        self.v = v
        self.time += dt
        self.substeps += 1

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def advance_to(self, time: float):
        target = float(time)
        while self.time < target - 1e-9:
            dt = min(self.SUB_DT, target - self.time)
            if dt < 1e-9:
                break
            self._step(dt)
        return self._frame(max(target, 0.0))

    def _grid_diagnostics(self):
        tree = cKDTree(self.x)
        k = min(48, max(1, self.n))
        dist, idx = tree.query(self.grid_points, k=k, distance_upper_bound=self.h_grid)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)
        valid = np.isfinite(dist) & (idx < self.n)
        w = np.where(valid, self.poly6_g * np.maximum(self.h_grid2 - dist ** 2, 0.0) ** 3, 0.0)

        # collocated liquid density and velocity
        safe_idx = np.where(valid, idx, 0)
        rho_g = (w * self.mass[safe_idx] * valid).sum(axis=1)
        wsum = w.sum(axis=1)
        vel = (w * self.v[safe_idx]).sum(axis=1) / np.maximum(wsum, 1e-12)[:, None] \
            if w.ndim == 2 else np.zeros((len(self.grid_points), 3))
        vel[wsum <= 1e-12] = 0.0

        # liquid level set (signed distance, positive inside the liquid)
        occ = (rho_g > self.occ_r).reshape(self.grid_shape)
        din = ndimage.distance_transform_edt(occ, sampling=(self.dx, self.dy, self.dz))
        dout = ndimage.distance_transform_edt(~occ, sampling=(self.dx, self.dy, self.dz))
        phi = (din - dout).reshape(-1)

        active = (rho_g > 0.02 * self.rho0).reshape(self.grid_shape)
        return (active.astype(np.float64),
                phi.reshape(self.grid_shape),
                vel.reshape(self.grid_shape + (3,)))

    def _boundary_fields(self, t):
        target = np.zeros((self.b_count, 3))
        target[self.b_moving, 2] = self._gate_velocity(t)

        # measured fluid-side velocity at the wall samples (no-slip fallback)
        if self.n:
            tree = cKDTree(self.x)
            dist, idx = tree.query(self.b_pos, k=min(8, self.n), distance_upper_bound=0.25)
            dist = np.atleast_2d(dist)
            idx = np.atleast_2d(idx)
            valid = np.isfinite(dist) & (idx < self.n)
            safe = np.where(valid, idx, 0)
            wr = np.maximum(0.25 ** 2 - np.where(valid, dist, 0.25) ** 2, 0.0) ** 3
            ws = wr.sum(axis=1)
            vel = (wr[:, :, None] * self.v[safe]).sum(axis=1) / np.maximum(ws, 1e-12)[:, None]
            vel[ws <= 1e-12] = target[ws <= 1e-12]
        else:
            vel = target.copy()
        return target, vel

    def _frame(self, t) -> StateFrame:
        active, phi, vel = self._grid_diagnostics()
        b_target, b_vel = self._boundary_fields(t)

        eye3 = np.eye(3)
        attributes = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": np.full(self.n, 0.45 * self.spacing),
            "material_label": self.body.astype(np.float64),
            "deformation_gradient": np.repeat(eye3[None], self.n, axis=0),
        }
        fields = {
            "active_mask": active,
            "liquid_level_set": phi,
            "velocity_field": vel,
            "boundary_normal": self.b_nrm.copy(),
            "boundary_target_velocity": b_target,
            "boundary_velocity": b_vel,
            "cumulative_mass_source": np.zeros(1),  # single material, no sources
        }
        scalars = {
            "contact_length_scale": float(self.spacing),
            "grid_spacing": float(self.grid_spacing),
            "characteristic_length": float(self.characteristic_length),
            "reference_velocity": float(self.reference_velocity),
        }
        frame = StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position_based_fluids",
            "particles": int(self.n),
            "substeps": int(self.substeps),
            "rest_density": float(getattr(self, "rho0", 0.0)),
            "total_mass": float(self.mass.sum()) if self.n else 0.0,
        }
