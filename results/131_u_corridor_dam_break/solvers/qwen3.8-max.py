"""Weakly-compressible SPH dam break in a U-shaped corridor.

Physics
-------
WCSPH (Monaghan): density summation with a Poly6 kernel, pressure from a
clamped linear equation of state p = K (rho - rho0) (weakly compressible),
Monaghan artificial viscosity, gravity, and rigid axis-aligned box
boundaries (floor, left/right corridor walls, end wall, kinematic dam gate
that slides downward at release).  Particles are advected with symplectic
Euler sub-steps limited by a CFL condition; boundary handling reflects and
penalizes intrusions along the axis of least penetration.

The solver consumes the frozen scene sampling (initial compact liquid
column confined upstream of the gate in the y<0 corridor leg), so the U
path is: leg y in [-1.55,-0.6] -> gate at y=-0.6 -> leg y in [-0.6,1.45]
between x=-1.05 and x=1.05 -> end wall at y=1.55 with run-up and
reflection back down the corridor.

Diagnostic exports every frame:
  fields.velocity_field      [nx,ny,nz,3] mass-weighted particle velocity
  fields.active_mask         [nx,ny,nz]  bool cells containing liquid,
                                         restricted to corridor interior
  fields.liquid_level_set    [nx,ny,nz]  signed distance-style level set
  fields.boundary_normal     [B,3]       unit inward wall normals
  fields.boundary_target_velocity [B,3]  prescribed wall velocity
  fields.boundary_velocity   [B,3]       fluid velocity at wall samples
  fields.cumulative_mass_source [K=1]    cumulative added-minus-removed mass
  attributes.mass_quantity   [N]         per-particle mass
  scalars.characteristic_length / grid_spacing / reference_velocity
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


# ------------------------------------------------------------------ geometry --
def _load_geometry(scene):
    cfg = scene.config
    geom_cfg = cfg.get("scene_geometry")
    if geom_cfg is None:
        try:
            from worldsolver.config import load_yaml
            from pathlib import Path
            p = Path("scene_geometry.yaml")
            if p.is_file():
                geom_cfg = load_yaml(p)
        except Exception:
            geom_cfg = None
    return geom_cfg


def _boxes_from_geometry(geom_cfg):
    out = []
    if not geom_cfg:
        return out
    for obj in geom_cfg.get("objects", []):
        g = obj.get("geometry", {})
        if g.get("type") != "box":
            continue
        tr = obj.get("transform", {})
        c = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), float)
        h = np.asarray(g.get("half_extents", [0.1, 0.1, 0.1]), float)
        out.append((c.copy(), h.copy(), obj.get("motion", "static"), obj.get("id", "")))
    return out


def _default_boxes():
    return [
        (np.array([0.0, 0.0, -0.05]), np.array([3.2, 1.4, 0.05]), "static", "ground"),
        (np.array([-1.15, 0.0, 0.55]), np.array([0.1, 1.55, 0.55]), "static", "corridor_left_wall"),
        (np.array([1.15, 0.0, 0.55]), np.array([0.1, 1.55, 0.55]), "static", "corridor_right_wall"),
        (np.array([0.0, 1.55, 0.55]), np.array([1.15, 0.1, 0.55]), "static", "corridor_end_wall"),
        (np.array([0.0, -0.6, 0.55]), np.array([0.85, 0.04, 0.55]), "kinematic", "dam_gate"),
    ]


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup --
    def initialize(self, scene, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        cfg = scene.config
        geom_cfg = _load_geometry(scene)

        init = scene.initial_state()
        p = np.asarray(init.position, float).copy()
        # deterministic jitter to break lattice symmetry
        spacing = float(cfg["sampling"]["spacing"])
        p += self._rng.uniform(-0.1, 0.1, size=p.shape) * spacing
        p[:, 2] = np.maximum(p[:, 2], 0.5 * spacing)

        self._p = p
        self._v = np.zeros_like(p)
        self._m = np.asarray(init.mass, float).copy()
        self._body = np.asarray(init.body_id).copy()
        self._mass_quantity = np.asarray(init.attributes["mass_quantity"], float).copy()
        self._n = len(p)

        self._dx = spacing
        rho0 = float(cfg["sampling"]["density"])
        self._rho0 = rho0
        self._gravity = np.asarray(cfg["gravity"], float)
        self._dt_out = float(cfg["sample_dt"])
        self._time = 0.0

        # SPH parameters
        h = 1.6 * spacing
        self._h = h
        self._K = 40.0 * float(np.linalg.norm(self._gravity)) * 2.8  # ~ 1100 Pa per unit over-density
        self._visc_beta = 0.35
        self._c0 = max(np.sqrt(self._K / rho0 * 7.0 * rho0 / rho0), 1.0)
        # wave speed for CFL: use sqrt(K/rho0 * (rho/rho0)^6) upper bound via c0
        self._c_cfl = float(np.sqrt(self._K * 7.0 / rho0)) + 3.0

        self._poly6 = 315.0 / (64.0 * np.pi * h ** 9)
        self._spiky_grad = -45.0 / (np.pi * h ** 6)
        self._w_self = self._poly6 * h ** 6
        self._support_r2 = h * h

        boxes = _boxes_from_geometry(geom_cfg)
        if not boxes:
            boxes = _default_boxes()
        self._boxes = boxes
        self._gate_id = next((k for k, b in enumerate(boxes) if b[2] == "kinematic"), -1)
        self._gate_c0 = boxes[self._gate_id][0].copy() if self._gate_id >= 0 else None
        self._gate_drop = 1.4          # slide distance
        self._gate_time = 0.35         # seconds to fully retract

        self._grid_shape = tuple(int(s) for s in cfg["sampling"]["field_grid_shape"])
        self._grid_axes = self._make_grid_axes()
        self._build_boundary_samples()

        self._rho = np.full(self._n, rho0)
        self._a = np.zeros_like(p)
        self._diag = {"n_particles": int(self._n), "h": float(h),
                      "stiffness": float(self._K)}

    def _make_grid_axes(self):
        gs = self._grid_shape
        x = np.linspace(-1.05, 1.05, gs[0])
        y = np.linspace(-1.45, 1.45, gs[1])
        z = np.linspace(0.05, 1.05, gs[2])
        return x, y, z

    def _build_boundary_samples(self):
        pts = []
        nrm = []
        tgt = []
        d = self._dx
        xs = np.arange(-1.0, 1.001, 2 * d)
        ys = np.arange(-1.4, 1.401, 2 * d)
        zs = np.arange(0.08, 1.05, 2 * d)
        # floor
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts.append(np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)]))
        nrm.append(np.tile(np.array([0.0, 0.0, 1.0]), (gx.size, 1)))
        tgt.append(np.zeros((gx.size, 3)))
        # left wall inner face x=-1.05
        gy2, gz2 = np.meshgrid(np.arange(-1.4, 1.401, 2 * d), zs, indexing="ij")
        s = gy2.size
        pts.append(np.column_stack([np.full(s, -1.05), gy2.ravel(), gz2.ravel()]))
        nrm.append(np.tile(np.array([1.0, 0.0, 0.0]), (s, 1)))
        tgt.append(np.zeros((s, 3)))
        # right wall inner face x=+1.05
        pts.append(np.column_stack([np.full(s, 1.05), gy2.ravel(), gz2.ravel()]))
        nrm.append(np.tile(np.array([-1.0, 0.0, 0.0]), (s, 1)))
        tgt.append(np.zeros((s, 3)))
        # end wall inner face y=1.45
        gx3, gz3 = np.meshgrid(xs, zs, indexing="ij")
        s = gx3.size
        pts.append(np.column_stack([gx3.ravel(), np.full(s, 1.45), gz3.ravel()]))
        nrm.append(np.tile(np.array([0.0, -1.0, 0.0]), (s, 1)))
        tgt.append(np.zeros((s, 3)))
        self._b_samples = np.concatenate(pts, axis=0)
        nn = np.concatenate(nrm, axis=0)
        self._b_normals = nn / np.linalg.norm(nn, axis=1, keepdims=True)
        self._b_target = np.concatenate(tgt, axis=0)

    # ------------------------------------------------------------- neighbors --
    def _neighbor_pairs(self):
        """Vectorized uniform-grid neighbor search. Returns I, J, rv, r2 (symmetric pairs, I!=J)."""
        h = self._h
        p = self._p
        cs = h
        ci = np.floor(p / cs).astype(np.int64)
        ci[:, 0] += 4096
        ci[:, 1] += 4096
        ci[:, 2] += 4096
        key = (ci[:, 0] * 8192 + ci[:, 1]) * 8192 + ci[:, 2]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        uniq, start = np.unique(ks, return_index=True)
        end = np.append(start[1:], len(ks))
        cell_of_sorted = np.arange(len(uniq)).repeat(end - start)
        cell_of = np.empty(len(p), np.int64)
        cell_of[order] = cell_of_sorted
        # neighbor cell offsets (13 of 27 for symmetry, plus self handled below)
        offs = []
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for oz in (-1, 0, 1):
                    if (ox, oy, oz) > (0, 0, 0):
                        offs.append((ox, oy, oz))
        offs = np.asarray(offs, np.int64)
        cell_idx = np.tile(ci[:, None, :], (1, len(offs), 1)) + offs[None, :, :]
        nkey = (cell_idx[:, :, 0] * 8192 + cell_idx[:, :, 1]) * 8192 + cell_idx[:, :, 2]
        nkey = nkey.ravel()
        pos = np.searchsorted(uniq, nkey)
        pos_c = np.clip(pos, 0, len(uniq) - 1)
        valid = uniq[pos_c] == nkey
        rep_i = np.repeat(np.arange(len(p)), len(offs))
        valid = valid & (pos < len(uniq))
        ri = rep_i[valid]
        cell_j = pos_c[valid]
        s0 = start[cell_j]
        s1 = end[cell_j]
        counts = s1 - s0
        total = int(counts.sum())
        I = np.repeat(ri, counts)
        Jrep = np.repeat(s0, counts) + np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        J = order[Jrep]
        rv = p[I] - p[J]
        r2 = (rv * rv).sum(1)
        keep = r2 < self._support_r2
        I, J, rv, r2 = I[keep], J[keep], rv[keep], r2[keep]
        # add symmetric counterparts
        I2 = np.concatenate([I, J])
        J2 = np.concatenate([J, I])
        rv2 = np.concatenate([rv, -rv])
        r22 = np.concatenate([r2, r2])
        return I2, J2, rv2, r22

    # ------------------------------------------------------------------- step --
    def _gate_center(self, t):
        if self._gate_c0 is None:
            return None
        frac = min(max(t / self._gate_time, 0.0), 1.0)
        c = self._gate_c0.copy()
        c[2] -= self._gate_drop * frac
        return c

    def _active_boxes(self, t):
        boxes = []
        for k, (c, he, motion, oid) in enumerate(self._boxes):
            if motion == "kinematic":
                cc = self._gate_center(t)
                if cc is None:
                    continue
                if cc[2] + he[2] <= 0.02:  # fully retracted below floor
                    continue
                boxes.append((cc, he))
            else:
                boxes.append((c, he))
        return boxes

    def _compute_accel(self, t):
        p, v, m = self._p, self._v, self._m
        h = self._h
        rho0 = self._rho0
        I, J, rv, r2 = self._neighbor_pairs()
        r = np.sqrt(np.maximum(r2, 1e-12))
        # density
        w = self._poly6 * np.maximum(h * h - r2, 0.0) ** 3
        rho = m * self._w_self
        np.add.at(rho, I, m[J] * w)
        rho = np.maximum(rho, 0.25 * rho0)
        self._rho = rho
        # pressure (clamped linear EOS)
        pr = np.maximum(self._K * (rho - rho0), 0.0)
        a = np.zeros_like(p)
        if len(I):
            eps = 1e-12
            spiky = self._spiky_grad * (h - r) ** 2 / (r + eps)  # <0
            # pressure: symmetric
            pterm = m[J] * (pr[I] / rho[I] ** 2 + pr[J] / rho[J] ** 2)
            # artificial viscosity (Monaghan)
            dv = v[I] - v[J]
            mu = h * (dv * rv).sum(1) / (r2 + 0.01 * h * h)
            c_bar = self._c_cfl
            rho_bar = 0.5 * (rho[I] + rho[J])
            pi_ij = np.where(mu < 0.0,
                             (-self._visc_beta * c_bar * mu + 0.1 * self._visc_beta ** 2 * mu ** 2) / rho_bar,
                             0.0)
            fac = spiky * m[J] * (pterm + pi_ij)   # dv/dr direction along rv
            # acceleration on I: a_i = -sum_j m_j (p_i/rho_i^2 + p_j/rho_j^2 + Pi) grad W
            # grad_i W = spiky * rv/r  (spiky already includes 1/r factor via (h-r)^2/r)
            f = fac[:, None] * rv
            np.add.at(a, I, -f)
        a += self._gravity[None, :]
        return a

    def _enforce_boundaries(self, t_after):
        p, v = self._p, self._v
        margin = 0.0
        for c, he in self._active_boxes(t_after):
            lo = c - he
            hi = c + he
            inside = np.all((p >= lo) & (p <= hi), axis=1)
            idx = np.nonzero(inside)[0]
            if len(idx) == 0:
                continue
            q = p[idx]
            d_lo = q - lo
            d_hi = hi - q
            # least penetration face
            pen = np.minimum(d_lo, d_hi)              # [k,3]
            axis = np.argmin(pen, axis=1)             # [k]
            side_lo = d_lo[np.arange(len(idx)), axis] < d_hi[np.arange(len(idx)), axis]
            rows = np.arange(len(idx))
            pos_new = q.copy()
            pos_new[rows, axis] = np.where(side_lo, lo[axis], hi[axis])
            p[idx] = pos_new
            vn = v[idx, axis]
            v[idx, axis] = np.where(side_lo, np.minimum(vn, 0.0), np.maximum(vn, 0.0))
            # tangential damping (friction)
            for b in range(3):
                if b == axis:
                    continue
                v[idx, b] *= 0.998
        # absolute floor safety
        fmin = 0.5 * self._dx
        below = p[:, 2] < fmin
        p[below, 2] = fmin
        v[below, 2] = np.maximum(v[below, 2], 0.0)

    def _substep(self, dt, t):
        a = self._compute_accel(t)
        self._a = a
        self._v += a * dt
        # velocity clamp for stability
        vmax = 8.0
        sp = np.linalg.norm(self._v, axis=1)
        over = sp > vmax
        if np.any(over):
            self._v[over] *= vmax / sp[over, None]
        self._p += self._v * dt
        self._enforce_boundaries(t + dt)

    # -------------------------------------------------------------- diagnostics
    def _make_fields(self):
        gs = self._grid_shape
        x, y, z = self._grid_axes
        hx = x[1] - x[0]; hy = y[1] - y[0]; hz = z[1] - z[0]
        p = self._p
        # cell indices for particles (clamp into grid)
        ix = np.clip(np.floor((p[:, 0] - x[0]) / hx).astype(int), 0, gs[0] - 1)
        iy = np.clip(np.floor((p[:, 1] - y[0]) / hy).astype(int), 0, gs[1] - 1)
        iz = np.clip(np.floor((p[:, 2] - z[0]) / hz).astype(int), 0, gs[2] - 1)
        flat = (ix * gs[1] + iy) * gs[2] + iz
        m = self._m
        # mass per cell
        cell_mass = np.bincount(flat, weights=m, minlength=np.prod(gs)).reshape(gs)
        cell_vol = hx * hy * hz
        cell_rho = cell_mass / cell_vol
        # active mask: corridor-interior cells containing meaningful liquid
        mask = cell_rho > 0.25 * self._rho0
        # level set: signed distance approximation from local density
        # negative inside liquid, positive outside, scale = half cell diagonal
        scale = 0.5 * np.sqrt(hx * hx + hy * hy + hz * hz)
        ls = scale * (1.0 - 2.0 * np.clip(cell_rho / self._rho0, 0.0, 1.0))
        # velocity field: momentum-weighted
        vf = np.zeros(gs + (3,))
        for k in range(3):
            comp = np.bincount(flat, weights=m * self._v[:, k], minlength=np.prod(gs)).reshape(gs)
            vf[..., k] = comp / np.maximum(cell_mass, 1e-12)
        # boundary velocity: nearest-particle fluid velocity
        bs = self._b_samples
        bv = np.zeros_like(bs)
        # coarse search: assign each boundary sample to grid cell of nearest particle
        # use KD-free approach: bucket particles by coarse cells
        # boundary samples are on corridor surfaces; find particles within 2*dx
        dmax2 = (2.5 * self._dx) ** 2
        # chunk over boundary samples
        B = len(bs)
        step = 512
        for s0 in range(0, B, step):
            d2 = ((bs[s0:s0 + step, None, :] - p[None, :, :]) ** 2).sum(2)
            j = np.argmin(d2, axis=1)
            close = d2[np.arange(len(j)), j] < dmax2
            rows = np.arange(s0, s0 + len(j))[close]
            bv[rows] = self._v[j[close]]
        fields = {
            "active_mask": mask.astype(float),
            "boundary_normal": self._b_normals.copy(),
            "boundary_target_velocity": self._b_target.copy(),
            "boundary_velocity": bv,
            "cumulative_mass_source": np.zeros(1),
            "liquid_level_set": ls,
            "velocity_field": vf,
        }
        return fields

    def _frame(self):
        fields = self._make_fields()
        frame = StateFrame(
            float(self._time),
            self._p.copy(),
            self._v.copy(),
            mass=self._m.copy(),
            body_id=self._body.copy(),
            attributes={
                "mass_quantity": self._mass_quantity.copy(),
                "particle_radius": np.full(self._n, 0.45 * self._dx),
                "material_label": self._body.copy(),
                "deformation_gradient": np.repeat(np.eye(3)[None], self._n, axis=0),
            },
            fields=fields,
            scalars={
                "contact_length_scale": self._dx,
                "grid_spacing": self._dx,
                "characteristic_length": 1.4,
                "reference_velocity": float(np.sqrt(2.0 * 9.81 * 1.4)),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        time = float(time)
        if time <= self._time + 1e-12:
            return self._frame()
        dt_cfl = 0.3 * self._h / max(self._c_cfl, 1.0)
        dt_max = min(dt_cfl, 0.004)
        remaining = time - self._time
        guard = 0
        max_steps = 200000
        while remaining > 1e-12 and guard < max_steps:
            dt = min(dt_max, remaining)
            self._substep(dt, self._time)
            self._time += dt
            remaining = time - self._time
            guard += 1
        self._diag["last_steps_per_frame"] = int(guard)
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return dict(self._diag)
