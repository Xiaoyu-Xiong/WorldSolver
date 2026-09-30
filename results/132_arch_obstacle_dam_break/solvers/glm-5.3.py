"""Dam break through an arch obstacle (scene 132).

Physical model
--------------
Incompressible free-surface liquid simulated with position-based fluids
(Macklin & Mueller 2013): the liquid is carried by Lagrangian particles
whose density constraint C_i = rho_i / rho0 - 1 is enforced by a Jacobi
Newton step on predicted positions, iterated several times per substep.
Poly6 kernels measure density, spiky kernels provide constraint gradients,
and virtual mirror particles across the rigid walls/obstacle faces remove
the wall density deficiency.  Gravity, a rising release gate (kinematic
box), the rectangular tank and the rigid arch obstacle (three static
boxes) are honored exactly as frozen in the scene geometry.

Time stepping is a fixed explicit scheme: 120 Hz substeps, 4 constraint
iterations, XSPH viscosity for sheet coherence, collision projection
against every rigid box (with standoff), and a CFL velocity clamp.  No
keyframes, animation tables or precomputed trajectories are used: every
sampled state is produced by advancing the physics from the previous one.

Diagnostics are exported on a uniform Eulerian grid (signed liquid level
set, collocated velocity, active mask) plus wall sample normals and
velocities, per the StateFrame contract.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Position-based incompressible fluid solver for the arch dam break."""

    SUB_DT = 1.0 / 120.0          # physics substep (s)
    SOLVER_ITERATIONS = 4         # density-constraint iterations per substep
    H_FACTOR = 2.5                # smoothing radius in particle spacings
    RELAX_EPS = 1e-9              # constraint regularization
    MAX_DP_FACTOR = 0.25          # position correction cap, in radii of h
    XSPH = 0.2                    # velocity blending coefficient
    VMAX = 7.0                    # CFL safety clamp (m/s)

    GATE_T0 = 0.15                # barrier release time (s)
    GATE_SPEED = 2.5              # gate lift speed (m/s)
    GATE_LIFT = 1.2               # total gate lift (m)

    TANK_X = 3.2                  # tank half length (ground box extent)
    TANK_Y = 1.4                  # tank half width

    def __init__(self) -> None:
        self._ready = False
        self._steps = 0

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        config = scene.config
        frame = scene.initial_state()
        sampling = config["sampling"]

        self.spacing = float(sampling["spacing"])
        bulk_density = float(sampling["density"])
        self.gravity = np.asarray(config["gravity"], dtype=float)

        self.h = self.H_FACTOR * self.spacing
        self.h2 = self.h * self.h
        self.rr = 0.5 * self.spacing            # rigid-wall standoff
        self.level_radius = 0.75 * self.spacing  # liquid level-set radius
        self.m = bulk_density * self.spacing ** 3
        self.poly6_c = 315.0 / (64.0 * np.pi * self.h ** 9)
        self.spiky_c = 45.0 / (np.pi * self.h ** 6)
        self.rho0 = self._rest_density()

        self.x = np.array(frame.position, dtype=float)
        self.v = np.zeros_like(self.x)
        self.n = self.x.shape[0]
        self.mass = np.array(frame.mass, dtype=float)
        self.body_id = np.array(frame.body_id)
        self.particle_radius = np.full(self.n, 0.45 * self.spacing)
        self.material_label = np.array(frame.attributes["material_label"])
        self.deformation_gradient = np.repeat(np.eye(3)[None, :, :], self.n, axis=0)

        head = float(self.x[:, 2].max() - self.x[:, 2].min())
        self.characteristic_length = max(head, float(np.ptp(self.x, axis=0).max()))
        gmag = float(np.linalg.norm(self.gravity))
        self.reference_velocity = float(np.sqrt(gmag * head)) if gmag > 0.0 else 1.0

        # static rigid obstacles (axis-aligned boxes), frozen geometry
        self.static_boxes = [
            (np.array([-0.75, 0.0, 0.70]), np.array([0.22, 0.70, 0.70])),  # arch_left
            (np.array([0.75, 0.0, 0.70]), np.array([0.22, 0.70, 0.70])),   # arch_right
            (np.array([0.0, 0.0, 1.40]), np.array([0.95, 0.70, 0.18])),    # arch_top
        ]
        self.gate_center0 = np.array([-1.55, 0.0, 0.55])
        self.gate_half = np.array([0.04, 0.65, 0.55])

        # uniform diagnostic grid spanning the tank
        shape = tuple(int(v) for v in sampling["field_grid_shape"])
        self.grid_shape = shape
        self.grid_dx = (2.0 * self.TANK_X) / shape[0]
        y_half = 0.5 * self.grid_dx * shape[1]
        self.grid_axes = (
            -self.TANK_X + self.grid_dx * (np.arange(shape[0]) + 0.5),
            -y_half + self.grid_dx * (np.arange(shape[1]) + 0.5),
            self.grid_dx * (np.arange(shape[2]) + 0.5),
        )

        self._build_boundary_samples()
        self._steps = 0
        self._ready = True

    def _rest_density(self) -> float:
        """Rest density of the initial regular lattice under the poly6 kernel."""
        k = np.arange(-3, 4) * self.spacing
        ox, oy, oz = np.meshgrid(k, k, k, indexing="ij")
        pts = np.column_stack((ox.ravel(), oy.ravel(), oz.ravel()))
        r2 = (pts * pts).sum(axis=1)
        w = np.where(r2 < self.h2, self.poly6_c * (self.h2 - r2) ** 3, 0.0)
        return self.m * float(w.sum())

    # ------------------------------------------------------------------
    # boundary sample table (fixed sample count across frames)
    # ------------------------------------------------------------------
    def _face_grid(self, base, u_axis, u_range, v_axis, v_range, n_u, n_v, normal):
        us = np.linspace(u_range[0], u_range[1], n_u)
        vs = np.linspace(v_range[0], v_range[1], n_v)
        uu, vv = np.meshgrid(us, vs, indexing="ij")
        pts = np.tile(np.asarray(base, dtype=float), (uu.size, 1))
        pts[:, u_axis] = uu.ravel()
        pts[:, v_axis] = vv.ravel()
        nrm = np.tile(np.asarray(normal, dtype=float), (uu.size, 1))
        return pts, nrm

    def _build_boundary_samples(self) -> None:
        pos, nrm = [], []

        def add(base, ua, ur, va, vr, nu, nv, normal):
            p, q = self._face_grid(base, ua, ur, va, vr, nu, nv, normal)
            pos.append(p)
            nrm.append(q)

        # tank floor (ground top face) and side walls at the ground extents
        add([0.0, 0.0, 0.0], 0, (-3.0, 3.0), 1, (-1.2, 1.2), 9, 5, (0, 0, 1))
        add([-self.TANK_X, 0.0, 0.0], 1, (-1.2, 1.2), 2, (0.1, 1.6), 5, 6, (1, 0, 0))
        add([self.TANK_X, 0.0, 0.0], 1, (-1.2, 1.2), 2, (0.1, 1.6), 5, 6, (-1, 0, 0))
        add([0.0, -self.TANK_Y, 0.0], 0, (-3.0, 3.0), 2, (0.1, 1.6), 9, 6, (0, 1, 0))
        add([0.0, self.TANK_Y, 0.0], 0, (-3.0, 3.0), 2, (0.1, 1.6), 9, 6, (0, -1, 0))

        # arch pillars (inner, outer and lateral faces)
        for sx in (-1.0, 1.0):
            cx = sx * 0.75
            add([sx * 0.53, 0.0, 0.70], 1, (-0.7, 0.7), 2, (0.1, 1.3), 3, 4, (sx, 0, 0))
            add([sx * 0.97, 0.0, 0.70], 1, (-0.7, 0.7), 2, (0.1, 1.3), 3, 4, (-sx, 0, 0))
            add([cx, -0.7, 0.70], 0, (cx - 0.22, cx + 0.22), 2, (0.1, 1.3), 3, 4, (0, -1, 0))
            add([cx, 0.7, 0.70], 0, (cx - 0.22, cx + 0.22), 2, (0.1, 1.3), 3, 4, (0, 1, 0))

        # arch top slab
        add([0.0, 0.0, 1.22], 0, (-0.95, 0.95), 1, (-0.7, 0.7), 5, 3, (0, 0, -1))
        add([0.0, 0.0, 1.58], 0, (-0.95, 0.95), 1, (-0.7, 0.7), 5, 3, (0, 0, 1))
        add([0.0, -0.7, 1.40], 0, (-0.95, 0.95), 2, (1.24, 1.56), 5, 2, (0, -1, 0))
        add([0.0, 0.7, 1.40], 0, (-0.95, 0.95), 2, (1.24, 1.56), 5, 2, (0, 1, 0))
        add([-0.95, 0.0, 1.40], 1, (-0.7, 0.7), 2, (1.24, 1.56), 3, 2, (-1, 0, 0))
        add([0.95, 0.0, 1.40], 1, (-0.7, 0.7), 2, (1.24, 1.56), 3, 2, (1, 0, 0))

        self._static_pos = np.concatenate(pos, axis=0)
        self._static_normal = np.concatenate(nrm, axis=0)

        gpos, gnrm = [], []

        def gadd(base, ua, ur, va, vr, nu, nv, normal):
            p, q = self._face_grid(base, ua, ur, va, vr, nu, nv, normal)
            gpos.append(p)
            gnrm.append(q)

        gadd([-1.59, 0.0, 0.55], 1, (-0.65, 0.65), 2, (0.05, 1.05), 3, 4, (-1, 0, 0))
        gadd([-1.51, 0.0, 0.55], 1, (-0.65, 0.65), 2, (0.05, 1.05), 3, 4, (1, 0, 0))
        gadd([-1.55, -0.65, 0.55], 0, (-1.59, -1.51), 2, (0.05, 1.05), 2, 4, (0, -1, 0))
        gadd([-1.55, 0.65, 0.55], 0, (-1.59, -1.51), 2, (0.05, 1.05), 2, 4, (0, 1, 0))
        gadd([-1.55, 0.0, 1.10], 0, (-1.59, -1.51), 1, (-0.65, 0.65), 3, 3, (0, 0, 1))

        self._gate_pos0 = np.concatenate(gpos, axis=0)
        self.boundary_normal = np.concatenate((self._static_normal, np.concatenate(gnrm, axis=0)), axis=0)

    # ------------------------------------------------------------------
    # kinematic gate
    # ------------------------------------------------------------------
    def _gate_offset(self, t: float) -> float:
        return min(self.GATE_LIFT, max(0.0, t - self.GATE_T0) * self.GATE_SPEED)

    def _gate_speed(self, t: float) -> float:
        if self.GATE_T0 < t < self.GATE_T0 + self.GATE_LIFT / self.GATE_SPEED:
            return self.GATE_SPEED
        return 0.0

    # ------------------------------------------------------------------
    # rigid collisions (boxes + tank containment)
    # ------------------------------------------------------------------
    def _collide(self, p, gate_c):
        boxes = self.static_boxes + [(gate_c, self.gate_half)]
        for _ in range(2):  # two relaxation passes for corner consistency
            for c, h in boxes:
                lo = c - h
                hi = c + h
                d_lo = p - lo
                d_hi = hi - p
                inside = (d_lo > 0.0).all(axis=1) & (d_hi > 0.0).all(axis=1)
                if inside.any():
                    ii = np.where(inside)[0]
                    pl, ph = d_lo[ii], d_hi[ii]
                    axis = np.argmin(np.minimum(pl, ph), axis=1)
                    rows = np.arange(ii.size)
                    to_lo = pl[rows, axis] < ph[rows, axis]
                    lo_a, hi_a = lo[axis], hi[axis]
                    p[ii, axis] = np.where(to_lo, lo_a - self.rr, hi_a + self.rr)
                # keep a standoff from exposed faces
                q = np.clip(p, lo, hi)
                dvec = p - q
                dist = np.sqrt((dvec * dvec).sum(axis=1))
                near = (dist > 1e-12) & (dist < self.rr)
                if near.any():
                    ii = np.where(near)[0]
                    p[ii] = q[ii] + dvec[ii] / dist[ii, None] * self.rr
            # tank walls (ground extents) and floor
            p[:, 0] = np.clip(p[:, 0], -self.TANK_X, self.TANK_X)
            p[:, 1] = np.clip(p[:, 1], -self.TANK_Y, self.TANK_Y)
            p[:, 2] = np.maximum(p[:, 2], self.rr)
        return p

    def _mirrors(self, p, gate_c):
        """Virtual mirror particles across rigid boundaries (wall density)."""
        idx, pos = [], []
        planes = (
            (0, -self.TANK_X, 1.0), (0, self.TANK_X, -1.0),
            (1, -self.TANK_Y, 1.0), (1, self.TANK_Y, -1.0),
            (2, 0.0, 1.0),
        )
        for axis, value, sign in planes:
            d = sign * (p[:, axis] - value)
            sel = np.where((d >= 0.0) & (d < self.h))[0]
            if sel.size:
                m = p[sel].copy()
                m[:, axis] -= 2.0 * sign * d[sel]
                idx.append(sel)
                pos.append(m)
        for c, h in self.static_boxes + [(gate_c, self.gate_half)]:
            lo, hi = c - h, c + h
            q = np.clip(p, lo, hi)
            dvec = p - q
            dist = np.sqrt((dvec * dvec).sum(axis=1))
            sel = np.where((dist > 1e-12) & (dist < self.h))[0]
            if sel.size:
                idx.append(sel)
                pos.append(p[sel] - 2.0 * dvec[sel])
        if not idx:
            return np.zeros(0, dtype=int), np.zeros((0, 3))
        return np.concatenate(idx), np.concatenate(pos, axis=0)

    # ------------------------------------------------------------------
    # position-based density constraint projection
    # ------------------------------------------------------------------
    def _project(self, p, gate_c):
        diff = p[:, None, :] - p[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", diff, diff)
        w = np.where(r2 < self.h2, self.poly6_c * (self.h2 - r2) ** 3, 0.0)
        rho = self.m * w.sum(axis=1)

        midx, mpos = self._mirrors(p, gate_c)
        if midx.size:
            dm = p[midx] - mpos
            r2m = (dm * dm).sum(axis=1)
            np.add.at(rho, midx, self.m * np.where(r2m < self.h2, self.poly6_c * (self.h2 - r2m) ** 3, 0.0))

        c_val = rho / self.rho0 - 1.0
        r = np.sqrt(np.maximum(r2, 1e-16))
        valid = (r2 < self.h2) & (r2 > 1e-12)
        coef = np.where(valid, self.spiky_c * (self.h - r) ** 2, 0.0)
        a_mat = coef[:, :, None] * diff / r[:, :, None]   # points from j toward i
        gp = (self.m / self.rho0) * a_mat                 # grad_{p_j} C_i
        sum_sq = np.einsum("ijk,ijk->i", gp, gp)
        vec = gp.sum(axis=1)                              # = -grad_{p_i} C_i
        denom = sum_sq + np.einsum("ij,ij->i", vec, vec)

        # one-sided incompressibility: correct only compression
        lam = np.where(c_val > 0.0, -c_val / np.maximum(denom, self.RELAX_EPS), 0.0)
        lam_sum = lam[:, None] + lam[None, :]
        dp = -np.einsum("ij,ijk->ik", lam_sum, a_mat) / self.rho0
        dp_mag = np.linalg.norm(dp, axis=1, keepdims=True)
        limit = self.MAX_DP_FACTOR * self.h
        dp = np.where(dp_mag > limit, dp * (limit / np.maximum(dp_mag, 1e-12)), dp)
        return self._collide(p + dp, gate_c)

    def _xsph(self, p, v):
        diff = p[:, None, :] - p[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", diff, diff)
        w = np.where(r2 < self.h2, self.poly6_c * (self.h2 - r2) ** 3, 0.0)
        np.fill_diagonal(w, 0.0)
        wsum = w.sum(axis=1)
        avg = w @ v
        blend = np.where(wsum[:, None] > 1e-12,
                         avg / np.maximum(wsum, 1e-12)[:, None], v)
        return v + self.XSPH * (blend - v)

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def _step(self) -> None:
        dt = self.SUB_DT
        t_next = (self._steps + 1) * dt
        gate_c = self.gate_center0 + np.array([0.0, 0.0, self._gate_offset(t_next)])

        v = self.v + self.gravity * dt
        p = self.x + v * dt
        p = self._collide(p, gate_c)
        for _ in range(self.SOLVER_ITERATIONS):
            p = self._project(p, gate_c)

        v_new = (p - self.x) / dt
        v_new = self._xsph(p, v_new)
        speed = np.linalg.norm(v_new, axis=1, keepdims=True)
        v_new = np.where(speed > self.VMAX, v_new * (self.VMAX / np.maximum(speed, 1e-12)), v_new)

        self.x = p
        self.v = v_new
        self._steps += 1

    def advance_to(self, time: float) -> StateFrame:
        if not self._ready:
            raise RuntimeError("solver is not initialized")
        time = float(time)
        target_steps = int(round(time / self.SUB_DT))
        while self._steps < target_steps:
            self._step()
        return self._build_frame(time)

    # ------------------------------------------------------------------
    # diagnostics exported on the StateFrame
    # ------------------------------------------------------------------
    def _grid_fields(self):
        gx, gy, gz = self.grid_axes
        px, py, pz = np.meshgrid(gx, gy, gz, indexing="ij")
        nodes = np.column_stack((px.ravel(), py.ravel(), pz.ravel()))
        npts = nodes.shape[0]
        phi = np.empty(npts)
        vel = np.zeros((npts, 3))
        r_inf = 2.0 * self.grid_dx
        chunk = 1024
        for s in range(0, npts, chunk):
            block = nodes[s:s + chunk]
            d2 = ((block[:, None, :] - self.x[None, :, :]) ** 2).sum(axis=-1)
            phi[s:s + chunk] = np.sqrt(d2.min(axis=1)) - self.level_radius
            wgt = np.maximum(0.0, r_inf - np.sqrt(d2))
            wsum = wgt.sum(axis=1)
            vel[s:s + chunk] = np.where(
                wsum[:, None] > 1e-12,
                (wgt @ self.v) / np.maximum(wsum, 1e-12)[:, None],
                0.0,
            )
        phi = np.minimum(phi, 2.0)
        active = phi < 0.0
        return (phi.reshape(self.grid_shape),
                active.reshape(self.grid_shape),
                vel.reshape(self.grid_shape + (3,)))

    def _boundary_fields(self, t: float):
        off = self._gate_offset(t)
        vg = self._gate_speed(t)
        pos = np.vstack((self._static_pos, self._gate_pos0 + np.array([0.0, 0.0, off])))
        target = np.zeros_like(pos)
        target[len(self._static_pos):, 2] = vg

        d = pos[:, None, :] - self.x[None, :, :]
        dist = np.sqrt((d * d).sum(axis=-1))
        wgt = np.maximum(0.0, 0.4 - dist)
        wsum = wgt.sum(axis=1)
        avg = wgt @ self.v
        bvel = np.where(wsum[:, None] > 1e-12,
                        avg / np.maximum(wsum, 1e-12)[:, None], target)
        return target, bvel

    def _build_frame(self, t: float) -> StateFrame:
        phi, active, vel = self._grid_fields()
        target, bvel = self._boundary_fields(t)
        fields = {
            "active_mask": active,
            "boundary_normal": self.boundary_normal,
            "boundary_target_velocity": target,
            "boundary_velocity": bvel,
            "cumulative_mass_source": np.zeros(1),
            "liquid_level_set": phi,
            "velocity_field": vel,
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": self.particle_radius,
            "material_label": self.material_label,
            "deformation_gradient": self.deformation_gradient,
        }
        scalars = {
            "characteristic_length": float(self.characteristic_length),
            "grid_spacing": float(self.grid_dx),
            "reference_velocity": float(self.reference_velocity),
        }
        return StateFrame(
            t,
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position_based_fluids",
            "particles": int(self.n),
            "substeps": int(self._steps),
            "rest_density": float(self.rho0),
            "smoothing_radius": float(self.h),
            "max_particle_speed": float(np.linalg.norm(self.v, axis=1).max()) if self.n else 0.0,
        }
