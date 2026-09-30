"""Physically-based WCSPH solver for the breaking-wave-over-shelf scene.

A dam-break water column (particles supplied by the frozen scene) collapses
under gravity, the resulting bore steepens over the submerged shelf, breaks
with a forward-curling crest, and runs out toward the end wall.  The fluid is
simulated with weakly-compressible SPH (Tait equation of state, Monaghan
artificial viscosity, leapfrog time stepping) against the frozen rigid
boundaries (ground, tank walls, submerged shelf, kinematic lift gate).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------- setup ----------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame0 = scene.initial_state()
        self.seed = int(seed)

        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.body_id = np.array(frame0.body_id)
        self.N = len(self.pos)
        self.radius = np.array(frame0.attributes["particle_radius"], dtype=float)

        g = np.asarray(cfg["gravity"], dtype=float)
        self.g = g
        s = float(cfg["sampling"]["spacing"])
        self.spacing = s
        self.rho0 = float(cfg["sampling"]["density"])
        self.m0 = float(self.mass.mean())

        # SPH parameters (cubic spline, support 2h)
        self.h = s
        self.support = 2.0 * self.h
        self.sigma_w = 1.0 / (np.pi * self.h ** 3)
        self.c_sos = 12.0                 # numerical sound speed
        self.gamma = 7.0
        self.stiff = self.rho0 * self.c_sos ** 2 / self.gamma
        self.alpha_v = 0.5                # artificial viscosity
        self.beta_v = 1.0
        self.dt_target = 0.0024

        self.t = 0.0
        self.rho = np.full(self.N, self.rho0)
        self.acc = np.zeros((self.N, 3))
        self._compute_acceleration()      # prime leapfrog

        # Frozen rigid geometry (inner faces), from scene_geometry.yaml
        self.mr = 0.35 * s                # boundary margin
        self.floor_z = 0.0
        self.wall_y = 0.92                # tank side inner faces at y = +-0.92
        self.end_x = 3.06                 # end wall inner face
        self.left_x = -3.06               # safety wall (ground extent)
        # shelf expanded to full tank width so no side leakage
        self.shelf_lo = np.array([0.25, -self.wall_y, -0.1])
        self.shelf_hi = np.array([2.65, self.wall_y, 0.6])
        # kinematic gate
        self.gate_lo0 = np.array([-0.49, -0.72, 0.0])
        self.gate_hi0 = np.array([-0.41, 0.72, 1.1])
        self.gate_off = 0.0
        self.gate_vel = 0.0

        # Diagnostic grid (uniform spacing)
        self.gshape = tuple(int(v) for v in cfg["sampling"]["field_grid_shape"])
        self.gdx = 0.25
        self.gorigin = np.array([-3.0, -2.0, -0.5])
        cx = self.gorigin[0] + (np.arange(self.gshape[0]) + 0.5) * self.gdx
        cy = self.gorigin[1] + (np.arange(self.gshape[1]) + 0.5) * self.gdx
        cz = self.gorigin[2] + (np.arange(self.gshape[2]) + 0.5) * self.gdx
        CX, CY, CZ = np.meshgrid(cx, cy, cz, indexing="ij")
        self.active_mask = (
            (CX >= -self.end_x) & (CX <= self.end_x)
            & (CY >= -self.wall_y) & (CY <= self.wall_y)
            & (CZ >= 0.0) & (CZ <= 1.3)
        )
        self._cell_centers = np.stack((CX, CY, CZ), axis=-1)

        # Boundary samples for exported wall fields
        self._build_boundary_samples()

        # Frozen scalars
        ptp = float(np.ptp(self.pos, axis=0).max())
        depth = float(self.pos[:, 2].max() - self.pos[:, 2].min())
        self.scalars = {
            "contact_length_scale": s,
            "grid_spacing": self.gdx,
            "characteristic_length": max(ptp, s),
            "reference_velocity": float(np.sqrt(max(abs(g[2]), 1e-9) * max(depth, s))),
        }

    # ---------------- boundary sample points ----------------
    def _plane_samples(self, fix_axis, fix_val, u_range, v_range, du, dv, normal):
        u = np.arange(u_range[0], u_range[1] + 1e-9, du)
        v = np.arange(v_range[0], v_range[1] + 1e-9, dv)
        U, V = np.meshgrid(u, v, indexing="ij")
        pts = np.zeros((U.size, 3))
        nrm = np.zeros((U.size, 3))
        free = [a for a in range(3) if a != fix_axis]
        pts[:, fix_axis] = fix_val
        pts[:, free[0]] = U.ravel()
        pts[:, free[1]] = V.ravel()
        nrm[:] = normal
        return pts, nrm

    def _build_boundary_samples(self):
        pts_l, nrm_l, tag_l = [], [], []

        def add(pts, nrm, tag):
            pts_l.append(pts); nrm_l.append(nrm); tag_l.append(np.full(len(pts), tag))

        # ground top (tag 0)
        p, n = self._plane_samples(2, 0.0, (-3.0, 3.0), (-0.8, 0.8), 0.4, 0.4, (0, 0, 1))
        add(p, n, 0)
        # shelf top (tag 0)
        p, n = self._plane_samples(2, 0.6, (0.3, 2.6), (-0.8, 0.8), 0.4, 0.4, (0, 0, 1))
        add(p, n, 0)
        # shelf upstream face (tag 0)
        p, n = self._plane_samples(0, 0.25, (-0.8, 0.8), (0.1, 0.5), 0.4, 0.4, (-1, 0, 0))
        add(p, n, 0)
        # left side wall inner face y=-0.92 (tag 0)
        p, n = self._plane_samples(1, -self.wall_y, (-3.0, 3.0), (0.1, 1.0), 0.4, 0.45, (0, 1, 0))
        add(p, n, 0)
        # right side wall inner face y=+0.92 (tag 0)
        p, n = self._plane_samples(1, self.wall_y, (-3.0, 3.0), (0.1, 1.0), 0.4, 0.45, (0, -1, 0))
        add(p, n, 0)
        # end wall inner face x=3.06 (tag 0)
        p, n = self._plane_samples(0, self.end_x, (-0.8, 0.8), (0.1, 1.0), 0.4, 0.45, (-1, 0, 0))
        add(p, n, 0)
        # gate face (tag 1, kinematic)
        p, n = self._plane_samples(0, -0.49, (-0.6, 0.6), (0.15, 1.0), 0.4, 0.42, (-1, 0, 0))
        add(p, n, 1)

        self.bnd_pts = np.concatenate(pts_l, axis=0)
        self.bnd_normal = np.concatenate(nrm_l, axis=0)
        self.bnd_tag = np.concatenate(tag_l, axis=0)
        self.gate_bnd_mask = self.bnd_tag == 1

    # ---------------- gate kinematics ----------------
    def _update_gate(self, t):
        t0, v_lift, lift_max = 0.1, 1.5, 1.3
        if t < t0:
            self.gate_off, self.gate_vel = 0.0, 0.0
        else:
            off = min(v_lift * (t - t0), lift_max)
            self.gate_vel = v_lift if off < lift_max - 1e-9 else 0.0
            self.gate_off = off

    # ---------------- SPH acceleration ----------------
    def _compute_acceleration(self):
        N, h = self.N, self.h
        pos, vel = self.pos, self.vel
        tree = cKDTree(pos)
        pairs = tree.query_pairs(self.support, output_type="ndarray")
        rho = np.full(N, self.m0 * self.sigma_w)
        acc = np.tile(self.g, (N, 1))
        if len(pairs) == 0:
            self.rho = rho
            return
        i = pairs[:, 0]
        j = pairs[:, 1]
        rij = pos[i] - pos[j]
        r2 = np.einsum("ij,ij->i", rij, rij)
        r = np.sqrt(np.maximum(r2, 1e-12))
        q = r / h
        sw = self.sigma_w
        W = sw * np.where(q <= 1.0, 1.0 - 1.5 * q * q + 0.75 * q ** 3,
                          0.25 * (2.0 - q) ** 3)
        dWdr = (sw / h) * np.where(q <= 1.0, -3.0 * q + 2.25 * q * q,
                                   -0.75 * (2.0 - q) ** 2)
        # density
        wm = self.m0 * W
        both = np.concatenate((i, j))
        rho += np.bincount(both, weights=np.concatenate((wm, wm)), minlength=N)
        self.rho = rho
        # pressure (Tait, clamped to reject tension)
        pr = self.stiff * (np.maximum(rho / self.rho0, 1e-6) ** self.gamma - 1.0)
        pr = np.maximum(pr, 0.0)
        # artificial viscosity
        dv = vel[i] - vel[j]
        vr = np.einsum("ij,ij->i", dv, rij)
        mu = h * vr / (r2 + 0.01 * h * h)
        rbar = 0.5 * (rho[i] + rho[j])
        Pi = np.where(vr < 0.0,
                      (-self.alpha_v * self.c_sos * mu + self.beta_v * mu * mu) / rbar,
                      0.0)
        coef = self.m0 * (pr[i] / rho[i] ** 2 + pr[j] / rho[j] ** 2 + Pi)
        f = -(coef * dWdr / r)[:, None] * rij  # force/mass on i from j
        for d in range(3):
            vals = np.concatenate((f[:, d], -f[:, d]))
            acc[:, d] += np.bincount(both, weights=vals, minlength=N)
        self.acc = acc

    # ---------------- rigid boundary enforcement ----------------
    def _push_out_aabb(self, lo, hi):
        pos, vel, mr = self.pos, self.vel, self.mr
        lo = lo - mr
        hi = hi + mr
        inside = np.all(pos > lo, axis=1) & np.all(pos < hi, axis=1)
        if not np.any(inside):
            return
        idx = np.nonzero(inside)[0]
        p = pos[idx]
        pen_lo = p - lo
        pen_hi = hi - p
        pen = np.minimum(pen_lo, pen_hi)
        axis = np.argmin(pen, axis=1)
        exit_lo = pen_lo[np.arange(len(idx)), axis] <= pen_hi[np.arange(len(idx)), axis]
        newp = p.copy()
        newp[np.arange(len(idx)), axis] = np.where(
            exit_lo, lo[axis], hi[axis])
        pos[idx] = newp
        v = vel[idx, axis]
        vel[idx, axis] = np.where(exit_lo, np.minimum(v, 0.0), np.maximum(v, 0.0))
        # light tangential friction
        for d in range(3):
            if True:
                tang = vel[idx, d].copy()
                mask_ax = axis == d
                vel[idx[mask_ax], d] = tang[mask_ax] * 0.995

    def _enforce_boundaries(self):
        pos, vel, mr = self.pos, self.vel, self.mr
        # floor
        k = pos[:, 2] < self.floor_z + mr
        if np.any(k):
            pos[k, 2] = self.floor_z + mr
            vel[k, 2] = np.maximum(vel[k, 2], 0.0)
            vel[k, 0] *= 0.995
            vel[k, 1] *= 0.995
        # side walls
        k = pos[:, 1] < -self.wall_y + mr
        if np.any(k):
            pos[k, 1] = -self.wall_y + mr
            vel[k, 1] = np.maximum(vel[k, 1], 0.0)
        k = pos[:, 1] > self.wall_y - mr
        if np.any(k):
            pos[k, 1] = self.wall_y - mr
            vel[k, 1] = np.minimum(vel[k, 1], 0.0)
        # end wall and safety left wall
        k = pos[:, 0] > self.end_x - mr
        if np.any(k):
            pos[k, 0] = self.end_x - mr
            vel[k, 0] = np.minimum(vel[k, 0], 0.0)
        k = pos[:, 0] < self.left_x + mr
        if np.any(k):
            pos[k, 0] = self.left_x + mr
            vel[k, 0] = np.maximum(vel[k, 0], 0.0)
        # submerged shelf
        self._push_out_aabb(self.shelf_lo, self.shelf_hi)
        # kinematic gate (only while it still blocks the column)
        if self.gate_off < 1.3:
            self._push_out_aabb(self.gate_lo0 + np.array([0, 0, self.gate_off]),
                                self.gate_hi0 + np.array([0, 0, self.gate_off]))
        # global safety clamp (keeps state finite and inside the world)
        np.clip(pos[:, 0], self.left_x + mr, self.end_x - mr, out=pos[:, 0])
        np.clip(pos[:, 1], -self.wall_y + mr, self.wall_y - mr, out=pos[:, 1])
        np.clip(pos[:, 2], self.floor_z + mr, 4.0, out=pos[:, 2])
        vmax = 30.0
        sp = np.linalg.norm(vel, axis=1)
        bad = sp > vmax
        if np.any(bad):
            vel[bad] *= (vmax / sp[bad])[:, None]

    # ---------------- time stepping ----------------
    def _substep(self, dt):
        self.vel += 0.5 * dt * self.acc
        self.pos += dt * self.vel
        self._enforce_boundaries()
        self._compute_acceleration()
        self.vel += 0.5 * dt * self.acc

    def advance_to(self, time: float):
        time = float(time)
        rem = time - self.t
        if rem > 1e-9:
            n = max(1, int(np.ceil(rem / self.dt_target)))
            dt = rem / n
            for step in range(n):
                self._update_gate(self.t + (step + 1) * dt)
                self._substep(dt)
            self.t = time
        else:
            self._update_gate(time)
        return self._make_frame(time)

    # ---------------- diagnostic grids ----------------
    def _grid_fields(self):
        shape, gdx, org = self.gshape, self.gdx, self.gorigin
        idx = np.floor((self.pos - org) / gdx).astype(int)
        idx[:, 0] = np.clip(idx[:, 0], 0, shape[0] - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, shape[1] - 1)
        idx[:, 2] = np.clip(idx[:, 2], 0, shape[2] - 1)
        key = (idx[:, 0], idx[:, 1], idx[:, 2])
        count = np.zeros(shape)
        np.add.at(count, key, 1.0)
        mom = np.zeros(shape + (3,))
        np.add.at(mom, key, self.vel * self.m0)
        cs = gaussian_filter(count, 1.0, mode="constant")
        ms = gaussian_filter(mom, (1.0, 1.0, 1.0, 0.0), mode="constant")
        ref = (gdx / self.spacing) ** 3  # interior particles per cell
        vfield = np.zeros(shape + (3,))
        good = cs > 0.05 * ref
        for d in range(3):
            vfield[..., d] = np.where(good, ms[..., d] / (self.m0 * np.maximum(cs, 1e-9)), 0.0)
        level = (0.5 * ref - cs) / ref * 0.5  # meters; negative inside liquid
        return vfield, level

    def _boundary_fields(self):
        pts = self.bnd_pts.copy()
        pts[self.gate_bnd_mask, 2] += self.gate_off
        target = np.zeros_like(pts)
        target[self.gate_bnd_mask, 2] = self.gate_vel
        tree = cKDTree(self.pos)
        k = min(8, self.N)
        dist, idx = tree.query(pts, k=k)
        if k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        w = (dist < 0.4).astype(float)
        wsum = w.sum(axis=1)
        v = np.einsum("bk,bkd->bd", w, self.vel[idx]) / np.maximum(wsum, 1e-9)[:, None]
        empty = wsum == 0
        if np.any(empty):
            v[empty] = target[empty]
        return self.bnd_normal.copy(), target, v

    # ---------------- frame assembly ----------------
    def _make_frame(self, time: float):
        vfield, level = self._grid_fields()
        bn, bt, bv = self._boundary_fields()
        attrs = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": self.radius.copy(),
            "material_label": self.body_id.astype(np.int32),
        }
        fields = {
            "active_mask": self.active_mask.copy(),
            "boundary_normal": bn,
            "boundary_target_velocity": bt,
            "boundary_velocity": bv,
            "cumulative_mass_source": np.zeros(1),
            "liquid_level_set": level,
            "velocity_field": vfield,
        }
        frame = StateFrame(
            float(time),
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attrs,
            fields=fields,
            scalars=dict(self.scalars),
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_particles": int(self.N),
            "time": float(self.t),
            "sound_speed": float(self.c_sos),
            "max_speed": float(np.abs(self.vel).max()) if self.N else 0.0,
            "total_mass": float(self.mass.sum()),
        }
