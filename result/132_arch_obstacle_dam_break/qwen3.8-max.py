"""Physically-based particle solver for the arch-obstacle dam break scene.

Method: Position Based Fluids (Macklin & Mueller 2013) -- a particle
incompressible-flow solver.  The liquid column is represented by ~340 SPH
particles; the ground, arch obstacle boxes and the kinematic gate are rigid
boundaries enforced by position projection.  Gravity drives the dam break;
the gate lifts at t=0 (barrier release), the column collapses, passes through
/ around the arch opening, and settles downstream.

All required StateFrame fields are exported every sample time:
collocated particle state, mass attributes, diagnostic grid fields
(velocity_field, liquid_level_set, active_mask), boundary samples
(positions/normals/target & measured velocities) and scalars.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------
# Kernels / constants
# ----------------------------------------------------------------------
_H = 0.28                 # SPH smoothing radius (2 x particle spacing)
_CPOLY = 315.0 / (64.0 * np.pi * _H ** 9)
_W0 = _CPOLY * _H ** 6    # poly6 at r = 0
_CGRAD = 45.0 / (np.pi * _H ** 6)
_RHO0 = 1000.0
_DT = 1.0 / 120.0
_ITERS = 4
_EPS_RELAX = 0.5
_MAX_DP = 0.05
_K_TENSILE = 0.1
_N_TENSILE = 4
_WDQ = _CPOLY * (_H ** 2 - (0.3 * _H) ** 2) ** 3   # W(delta_q)
_XSPH = 0.08
_GRAV = np.array([0.0, 0.0, -9.81])

# implicit tank walls (keep fluid on the ground slab)
_WALL_X = 3.15
_WALL_Y = 1.35
_FLOOR = 0.005
_MARGIN = 0.015           # boundary stand-off for particle projection

# diagnostic grid (field_grid_shape = [24, 16, 28])
_GX, _GY, _GZ = 24, 16, 28
_GS = 0.27                       # uniform grid spacing
_GMIN = np.array([-3.24, -2.16, 0.0])


def _grid_centers():
    ax = [_GMIN[i] + (np.arange(s) + 0.5) * _GS for i, s in
          enumerate((_GX, _GY, _GZ))]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


class BenchmarkSolver:
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        frame0 = scene.initial_state()
        self.p = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.p_prev = self.p.copy()
        self.m = np.array(frame0.mass, dtype=float)
        self.n = len(self.p)
        self.body_id = np.array(frame0.body_id)
        self.t = 0.0

        self.radius = np.array(frame0.attributes["particle_radius"], dtype=float)
        self.labels = np.array(frame0.attributes["material_label"])
        self.eye3 = np.repeat(np.eye(3)[None], self.n, axis=0)

        pos = self.p
        self.char_len = max(float(np.ptp(pos, axis=0).max()), 0.14)
        z_top = float(pos[:, 2].max())
        self.ref_vel = float(np.sqrt(2.0 * 9.81 * z_top))

        # rigid boxes (static): arch pieces
        self.boxes = [
            (np.array([-0.75, 0.0, 0.7]), np.array([0.22, 0.7, 0.7])),
            (np.array([0.75, 0.0, 0.7]), np.array([0.22, 0.7, 0.7])),
            (np.array([0.0, 0.0, 1.4]), np.array([0.95, 0.7, 0.18])),
        ]
        # kinematic gate
        self.gate_c0 = np.array([-1.55, 0.0, 0.55])
        self.gate_e = np.array([0.04, 0.65, 0.55])
        self.lift_v = 3.0
        self.lift_max = 1.4
        self.lift_end = self.lift_max / self.lift_v

        self.gpts_grid = _grid_centers()
        self._build_boundary_samples()
        self._freeze_levelset_threshold()
        self.diag = {"n_particles": self.n, "method": "PBF"}

    # ------------------------------------------------------------------
    def _gate_offset(self, t):
        return float(np.clip(self.lift_v * t, 0.0, self.lift_max))

    def _gate_center(self, t):
        c = self.gate_c0.copy()
        c[2] += self._gate_offset(t)
        return c

    def _build_boundary_samples(self):
        pts, nrm, tag = [], [], []   # tag 0 static, 1 gate

        def face_samples(c, e, i_axis, side, nu=3, nv=3):
            u = (i_axis + 1) % 3
            v_ = (i_axis + 2) % 3
            n = np.zeros(3); n[i_axis] = side
            base = np.asarray(c, float).copy()
            base[i_axis] += side * (e[i_axis] + 0.005)
            us = np.linspace(-e[u] * 0.8, e[u] * 0.8, nu)
            vs = np.linspace(-e[v_] * 0.8, e[v_] * 0.8, nv)
            for a in us:
                for b in vs:
                    q = base.copy(); q[u] += a; q[v_] += b
                    pts.append(q); nrm.append(n); tag.append(0)

        # ground top
        for x in np.linspace(-3.0, 3.0, 13):
            for y in np.linspace(-1.2, 1.2, 7):
                pts.append([x, y, 0.005]); nrm.append([0, 0, 1]); tag.append(0)
        # arch boxes: all faces
        for c, e in self.boxes:
            for ax in range(3):
                for side in (-1.0, 1.0):
                    face_samples(c, e, ax, side)
        # implicit side walls
        for sx in (-1.0, 1.0):
            for y in np.linspace(-1.2, 1.2, 5):
                for z in np.linspace(0.15, 1.35, 4):
                    pts.append([sx * _WALL_X, y, z])
                    nrm.append([-sx, 0, 0]); tag.append(0)
        for sy in (-1.0, 1.0):
            for x in np.linspace(-3.0, 3.0, 9):
                for z in np.linspace(0.15, 1.35, 4):
                    pts.append([x, sy * _WALL_Y, z])
                    nrm.append([0, -sy, 0]); tag.append(0)
        # gate samples (kinematic, tag 1)
        c, e = self.gate_c0, self.gate_e
        n_gate = len(pts)
        for ax in range(3):
            for side in (-1.0, 1.0):
                u = (ax + 1) % 3; v_ = (ax + 2) % 3
                n = np.zeros(3); n[ax] = side
                base = c.copy(); base[ax] += side * (e[ax] + 0.005)
                for a in np.linspace(-e[u] * 0.8, e[u] * 0.8, 3):
                    for b in np.linspace(-e[v_] * 0.8, e[v_] * 0.8, 3):
                        q = base.copy(); q[u] += a; q[v_] += b
                        pts.append(q); nrm.append(n); tag.append(1)
        self.b_base = np.array(pts)
        self.b_normal = np.array(nrm)
        self.b_tag = np.array(tag)
        self.b_gate_mask = self.b_tag == 1

    def _boundary_points(self, t):
        pts = self.b_base.copy()
        pts[self.b_gate_mask, 2] += self._gate_offset(t)
        return pts

    # ------------------------------------------------------------------
    def _freeze_levelset_threshold(self):
        tree = cKDTree(self.p)
        f = self._gaussian_density(self.gpts_grid)
        self.ls_threshold = 0.3 * float(f.max())
        self._ls_ref = max(float(f.max()), 1e-9)
        del tree

    def _gaussian_density(self, pts, sigma=None):
        sigma = _GS if sigma is None else sigma
        inv2s2 = 1.0 / (2.0 * sigma * sigma)
        out = np.zeros(len(pts))
        step = 2048
        for a in range(0, len(pts), step):
            d2 = ((pts[a:a + step, None, :] - self.p[None, :, :]) ** 2).sum(-1)
            out[a:a + step] = np.exp(-d2 * inv2s2).sum(1)
        return out

    # ------------------------------------------------------------------
    def _project_boundaries(self, P, t):
        # floor & tank walls
        P[:, 2] = np.maximum(P[:, 2], _FLOOR)
        P[:, 0] = np.clip(P[:, 0], -_WALL_X, _WALL_X)
        P[:, 1] = np.clip(P[:, 1], -_WALL_Y, _WALL_Y)
        # box obstacles + gate
        all_boxes = list(self.boxes) + [(self._gate_center(t), self.gate_e)]
        for c, e in all_boxes:
            d = P - c
            ad = np.abs(d) - e
            inside = np.all(ad < 0.0, axis=1)
            if inside.any():
                k = np.argmax(ad[inside], axis=1)
                rows = np.nonzero(inside)[0]
                sgn = np.sign(d[rows, k])
                sgn[sgn == 0] = 1.0
                P[rows, k] = c[k] + sgn * (e[k] + _MARGIN)
            cl = c + np.clip(d, -e, e)
            vec = P - cl
            dist = np.linalg.norm(vec, axis=1)
            near = (~inside) & (dist > 1e-9) & (dist < _MARGIN)
            if near.any():
                P[near] = cl[near] + vec[near] / dist[near, None] * _MARGIN

    # ------------------------------------------------------------------
    def _substep(self, dt, t0):
        p = self.p
        v = self.v + _GRAV * dt
        p_pred = p + v * dt
        self._project_boundaries(p_pred, t0 + dt)

        tree = cKDTree(p_pred)
        pairs = tree.query_pairs(_H, output_type="ndarray")
        if len(pairs):
            I = np.concatenate((pairs[:, 0], pairs[:, 1]))
            J = np.concatenate((pairs[:, 1], pairs[:, 0]))
        else:
            I = J = np.zeros(0, dtype=np.int64)

        m = float(self.m[0])
        mr0 = m / _RHO0

        for _ in range(_ITERS):
            if len(I):
                q = p_pred[I] - p_pred[J]
                r2 = (q * q).sum(1)
                r = np.sqrt(r2) + 1e-12
                diff = _H * _H - r2
                Wp = _CPOLY * np.maximum(diff, 0.0) ** 3
                rho = m * (np.bincount(I, weights=Wp, minlength=self.n) + _W0)
                C = rho / _RHO0 - 1.0
                A = _CGRAD * np.maximum(_H - r, 0.0) ** 2 / r   # grad W = -A q
                gA = mr0 * A                                     # |g| = gA * r
                denom = np.bincount(I, weights=(gA * r) ** 2, minlength=self.n)
                gx = np.bincount(I, weights=-gA * q[:, 0], minlength=self.n)
                gy = np.bincount(I, weights=-gA * q[:, 1], minlength=self.n)
                gz = np.bincount(I, weights=-gA * q[:, 2], minlength=self.n)
                denom += gx * gx + gy * gy + gz * gz + _EPS_RELAX
                lam = -C / denom
                lam = np.clip(lam, -50.0, 50.0)
                scorr = -_K_TENSILE * (Wp / _WDQ) ** _N_TENSILE
                coef = lam[I] + lam[J] + scorr
                wq = coef * gA
                dp = np.empty_like(p_pred)
                dp[:, 0] = -np.bincount(I, weights=wq * q[:, 0], minlength=self.n)
                dp[:, 1] = -np.bincount(I, weights=wq * q[:, 1], minlength=self.n)
                dp[:, 2] = -np.bincount(I, weights=wq * q[:, 2], minlength=self.n)
                dpl = np.linalg.norm(dp, axis=1, keepdims=True)
                dp *= np.minimum(1.0, _MAX_DP / np.maximum(dpl, 1e-12))
                p_pred += dp
            self._project_boundaries(p_pred, t0 + dt)

        v = (p_pred - p) / dt
        # XSPH viscosity
        if len(I):
            q = p_pred[I] - p_pred[J]
            r2 = (q * q).sum(1)
            Wp = _CPOLY * np.maximum(_H * _H - r2, 0.0) ** 3
            dv = v[J] - v[I]
            wgt = (mr0 * Wp)[:, None] * dv
            visc = np.zeros_like(v)
            for k in range(3):
                visc[:, k] = np.bincount(I, weights=wgt[:, k], minlength=self.n)
            v += _XSPH * visc
        sp = np.linalg.norm(v, axis=1, keepdims=True)
        v *= np.minimum(1.0, 15.0 / np.maximum(sp, 1e-9))
        self.p = p_pred
        self.v = v
        self.t = t0 + dt

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        time = float(time)
        while self.t < time - 1e-12:
            dt = min(_DT, time - self.t)
            self._substep(dt, self.t)
        return self._make_frame(time)

    # ------------------------------------------------------------------
    def _make_frame(self, time):
        p, v = self.p, self.v
        tree = cKDTree(p)

        # ---- grid diagnostics ----------------------------------------
        gpts = self.gpts_grid
        dmin, _ = tree.query(gpts, k=1)
        sigma = _GS
        inv2s2 = 1.0 / (2.0 * sigma * sigma)
        f = np.zeros(len(gpts))
        vf = np.zeros((len(gpts), 3))
        step = 1024
        for a in range(0, len(gpts), step):
            chunk = gpts[a:a + step]
            d2 = ((chunk[:, None, :] - p[None, :, :]) ** 2).sum(-1)
            w = np.exp(-d2 * inv2s2)
            fs = w.sum(1)
            f[a:a + step] = fs
            safe = np.maximum(fs, 1e-6)
            vf[a:a + step] = (w @ v) / safe[:, None]
        vf[f < 1e-4] = 0.0
        sgn = np.clip((self.ls_threshold - f) / max(self.ls_threshold, 1e-9), -1.0, 1.0)
        level = dmin * sgn
        mask = (f > self.ls_threshold).astype(np.uint8)
        shape = (_GX, _GY, _GZ)

        # ---- boundary samples ----------------------------------------
        bpts = self._boundary_points(time)
        bnrm = self.b_normal
        btarget = np.zeros_like(bpts)
        if time < self.lift_end:
            btarget[self.b_gate_mask, 2] = self.lift_v
        inv2s2b = 1.0 / (2.0 * 0.25 * 0.25)
        bvel = np.zeros_like(bpts)
        for a in range(0, len(bpts), 512):
            chunk = bpts[a:a + 512]
            d2 = ((chunk[:, None, :] - p[None, :, :]) ** 2).sum(-1)
            w = np.exp(-d2 * inv2s2b)
            fs = w.sum(1)
            wet = fs > 1e-3
            if wet.any():
                bvel[a:a + 512][wet] = (w[wet] @ v) / fs[wet, None]

        attributes = {
            "mass_quantity": self.m.copy(),
            "particle_radius": self.radius.copy(),
            "material_label": self.labels.copy(),
            "deformation_gradient": self.eye3.copy(),
        }
        fields = {
            "cumulative_mass_source": np.zeros(self.n),
            "active_mask": mask.reshape(shape),
            "boundary_normal": bnrm,
            "boundary_target_velocity": btarget,
            "boundary_velocity": bvel,
            "liquid_level_set": level.reshape(shape),
            "velocity_field": vf.reshape(shape + (3,)),
        }
        scalars = {
            "characteristic_length": self.char_len,
            "grid_spacing": _GS,
            "reference_velocity": self.ref_vel,
            "contact_length_scale": 0.14,
            "liquid_phase_id": 0.0,
        }
        frame = StateFrame(
            float(time), p.copy(), v.copy(), mass=self.m.copy(),
            body_id=self.body_id.copy(), attributes=attributes,
            fields=fields, scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        return dict(self.diag)
