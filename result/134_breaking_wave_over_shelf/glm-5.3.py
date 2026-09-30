"""Position-Based Fluids solver: breaking wave over a submerged shelf.

The frozen scene is a dam-break style wave tank: a water column is held
behind a kinematic gate on the left; when the gate rises the released surge
travels right, shoals over the submerged shelf (top at z = 0.6), steepens,
plunges forward entraining air, and continues as a breaking bore that is
reflected by the downstream end wall.

Physics: 3D Position-Based Fluids (Macklin & Mueller 2013) with
 - poly6 density kernel / spiky gradient kernel,
 - iterated density constraints (lambda formulation),
 - XSPH viscosity,
 - rigid AABB collision projection (static tank, shelf, ground, walls and
   the moving gate) with normal-velocity removal,
 - deterministic cell-list neighbor search.

No Genesis: the discretization and time stepping are implemented directly
with NumPy.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

G_ACC = np.array([0.0, 0.0, -9.81])


def _unit(vecs):
    n = np.linalg.norm(vecs, axis=-1, keepdims=True)
    return vecs / np.maximum(n, 1e-12)


class BenchmarkSolver:
    # ---- numerical parameters -------------------------------------------
    sub_dt = 1.0 / 240.0        # physics substep
    pbf_iters = 4               # density-constraint iterations
    xsph_c = 0.05               # XSPH viscosity coefficient
    max_speed = 7.0             # m/s safety clamp
    dp_cap_frac = 0.3           # max position correction per iteration (x H)
    gate_speed = 1.5            # m/s upward gate velocity
    gate_travel = 1.9           # gate bottom travel before it stops
    left_wall_x = -3.15         # virtual left containment wall
    y_limit = 1.02

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.spacing = float(cfg["sampling"]["spacing"])
        density = float(cfg["sampling"]["density"])
        self.mp = density * self.spacing ** 3

        frame = scene.initial_state()
        self.pos = np.array(frame.position, dtype=float)
        self.n = len(self.pos)
        self.vel = np.zeros((self.n, 3))
        if frame.mass is not None:
            self.mass = np.array(frame.mass, dtype=float)
        else:
            self.mass = np.full(self.n, self.mp)
        if frame.body_id is not None:
            self.labels = np.array(frame.body_id).astype(np.int32)
        else:
            self.labels = np.zeros(self.n, np.int32)
        self.t = 0.0
        self.step_count = 0

        # ---- SPH kernels --------------------------------------------------
        self.H = 2.0 * self.spacing              # kernel support radius
        self.H2 = self.H ** 2
        self.lr = 1.1 * self.H                   # neighbor list radius
        self.poly6 = 315.0 / (64.0 * np.pi * self.H ** 9)
        self.W0 = self.poly6 * self.H2 ** 3
        self.spiky = -45.0 / (np.pi * self.H ** 6)
        self.eps_cf = 0.05
        self.dp_cap = self.dp_cap_frac * self.H

        # ---- frozen rigid boxes (center, half extents) ---------------------
        self.static_boxes = [
            (np.array([0.0, 0.0, -0.05]), np.array([3.2, 1.4, 0.05])),    # ground
            (np.array([0.0, -0.98, 0.55]), np.array([3.2, 0.06, 0.55])),  # side wall -y
            (np.array([0.0, 0.98, 0.55]), np.array([3.2, 0.06, 0.55])),   # side wall +y
            (np.array([3.12, 0.0, 0.55]), np.array([0.06, 0.98, 0.55])),  # end wall
            (np.array([1.45, 0.0, 0.30]), np.array([1.2, 0.84, 0.30])),   # submerged shelf
        ]

        # ---- neighbor grid -------------------------------------------------
        self.dom_min = np.array([-3.3, -1.2, -0.2])
        self.dom_max = np.array([3.3, 1.2, 3.2])
        self.ncells = (np.ceil((self.dom_max - self.dom_min) / self.lr).astype(int) + 1)

        # rest density measured on the initial lattice (interior value)
        I, J = self._pairs(self.pos)
        rho0 = self._density(self.pos, I, J).max()
        self.rho0 = max(float(rho0), 1e-9)

        # ---- diagnostic grid (frozen field_grid_shape 24 x 16 x 28) ---------
        self.gx = np.linspace(-3.2, 3.2, 25)
        self.gy = np.linspace(-1.04, 1.04, 17)
        self.gz = np.linspace(0.0, 1.6, 29)
        self.edges = [self.gx, self.gy, self.gz]
        self.gshape = (24, 16, 28)
        self.dxyz = (self.gx[1] - self.gx[0], self.gy[1] - self.gy[0],
                     self.gz[1] - self.gz[0])
        self.base_active = self._static_active_mask()

        # ---- frozen scalars -------------------------------------------------
        self.char_len = float(max(np.ptp(self.pos, axis=0).max(), self.spacing))
        depth = float(np.ptp(self.pos[:, 2])) if self.n else 0.5
        self.ref_vel = float(np.sqrt(9.81 * max(depth, 0.1)))

        self._init_boundary_samples()

        self.particle_radius = np.full(self.n, 0.45 * self.spacing)
        self.eye3 = np.repeat(np.eye(3)[None, :, :], self.n, axis=0)

    # ------------------------------------------------------------------ gate
    def _gate(self, t):
        zb = min(self.gate_speed * t, self.gate_travel)
        vg = self.gate_speed if zb < self.gate_travel else 0.0
        center = np.array([-0.45, 0.0, zb + 0.55])
        half = np.array([0.04, 0.72, 0.55])
        return center, half, zb, vg

    # -------------------------------------------------------- neighbor pairs
    def _pairs(self, pos):
        ci = np.floor((pos - self.dom_min) / self.lr).astype(np.int64)
        np.clip(ci, 0, self.ncells - 1, out=ci)
        nx, ny, nz = (int(v) for v in self.ncells)
        flat = (ci[:, 0] * ny + ci[:, 1]) * nz + ci[:, 2]
        ncell = nx * ny * nz
        order = np.argsort(flat, kind="stable")
        sflat = flat[order]
        cnt = np.bincount(sflat, minlength=ncell).astype(np.int64)
        start = np.zeros(ncell + 1, np.int64)
        np.cumsum(cnt, out=start[1:])
        present = np.nonzero(cnt)[0]
        px = present // (ny * nz)
        rem = present - px * (ny * nz)
        py = rem // nz
        pz = rem - py * nz
        Is, Js = [], []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    tx, ty, tz = px + a, py + b, pz + c
                    ok = ((tx >= 0) & (tx < nx) & (ty >= 0) & (ty < ny)
                          & (tz >= 0) & (tz < nz))
                    if not ok.any():
                        continue
                    src = present[ok]
                    tflat = (tx[ok] * ny + ty[ok]) * nz + tz[ok]
                    tot = cnt[src] * cnt[tflat]
                    K = int(tot.sum())
                    if K == 0:
                        continue
                    crep = np.repeat(np.arange(len(src)), tot)
                    base = np.repeat(np.cumsum(tot) - tot, tot)
                    k = np.arange(K) - base
                    nbr = tot[crep]
                    ia = start[src[crep]] + k // nbr
                    ib = start[tflat[crep]] + k % nbr
                    Is.append(order[ia])
                    Js.append(order[ib])
        if not Is:
            empty = np.empty(0, np.int64)
            return empty, empty
        I = np.concatenate(Is)
        J = np.concatenate(Js)
        keep = I != J
        I, J = I[keep], J[keep]
        d = pos[I] - pos[J]
        r2 = np.einsum("ij,ij->i", d, d)
        m = r2 < self.lr ** 2
        return I[m], J[m]

    def _kernel_pairs(self, p, I, J):
        """Distances/kernel values restricted to the kernel support."""
        dx = p[I] - p[J]
        r2 = np.einsum("ij,ij->i", dx, dx)
        m = r2 < self.H2
        I, J, dx, r2 = I[m], J[m], dx[m], r2[m]
        r = np.maximum(np.sqrt(r2), 1e-9)
        w = self.poly6 * (self.H2 - r2) ** 3
        dw = self.spiky * (self.H - r) ** 2       # dW/dr  (negative)
        gv = dw[:, None] * dx / r[:, None]        # grad W(x_i - x_j)
        return I, J, w, gv

    def _density(self, p, I, J):
        I2, J2, w, _ = self._kernel_pairs(p, I, J)
        rho = np.full(self.n, self.mp * self.W0)
        if len(I2):
            rho += np.bincount(I2, weights=self.mp * w, minlength=self.n)
        return rho

    # ------------------------------------------------------ PBF constraint
    def _project(self, p, I, J):
        I2, J2, w, gv = self._kernel_pairs(p, I, J)
        if not len(I2):
            return
        rho = self.mp * self.W0 + np.bincount(
            I2, weights=self.mp * w, minlength=self.n)
        C = rho / self.rho0 - 1.0
        np.maximum(C, 0.0, out=C)
        g = self.mp / self.rho0
        gx = g * gv[:, 0]
        gy = g * gv[:, 1]
        gz = g * gv[:, 2]
        sgx = np.bincount(I2, weights=gx, minlength=self.n)
        sgy = np.bincount(I2, weights=gy, minlength=self.n)
        sgz = np.bincount(I2, weights=gz, minlength=self.n)
        denom = (np.bincount(I2, weights=gx * gx + gy * gy + gz * gz,
                             minlength=self.n)
                 + sgx * sgx + sgy * sgy + sgz * sgz)
        lam = -C / (denom + self.eps_cf)
        coef = (lam[I2] + lam[J2])[:, None] * gv
        dp = np.empty((self.n, 3))
        for k in range(3):
            dp[:, k] = g * np.bincount(I2, weights=coef[:, k],
                                       minlength=self.n)
        dn = np.linalg.norm(dp, axis=1)
        scale = np.ones(self.n)
        big = dn > self.dp_cap
        scale[big] = self.dp_cap / dn[big]
        p += dp * scale[:, None]

    # ------------------------------------------------------------ collisions
    def _collide(self, p, gate_center, gate_half):
        for c, e in self.static_boxes:
            self._box_project(p, c, e)
        self._box_project(p, gate_center, gate_half)
        # virtual left wall and lateral containment
        p[:, 0] = np.maximum(p[:, 0], self.left_wall_x)
        p[:, 1] = np.clip(p[:, 1], -self.y_limit, self.y_limit)

    @staticmethod
    def _box_project(p, c, e):
        q = p - c
        pen = e - np.abs(q)
        inside = np.all(pen > 0.0, axis=1)
        if not inside.any():
            return
        idx = np.argmin(pen[inside], axis=1)
        rows = np.nonzero(inside)[0]
        target = c[idx] + np.sign(q[rows, idx]) * e[idx]
        p[rows, idx] = target

    def _vel_fix(self, v, p, gate_center, gate_half, gate_vel):
        boxes = [(c, e, np.zeros(3)) for c, e in self.static_boxes]
        boxes.append((gate_center, gate_half, np.array([0.0, 0.0, gate_vel])))
        for c, e, bv in boxes:
            q = p - c
            pen = e - np.abs(q)
            near = np.all(pen > -0.005, axis=1)
            if not near.any():
                continue
            rows = np.nonzero(near)[0]
            idx = np.argmin(pen[near], axis=1)
            nrm = np.zeros((len(rows), 3))
            nrm[np.arange(len(rows)), idx] = np.sign(q[rows, idx])
            vrel = v[rows] - bv
            vn = np.einsum("ij,ij->i", vrel, nrm)
            hit = vn < 0.0
            if not hit.any():
                continue
            hr = rows[hit]
            v[hr] = (vrel[hit] - vn[hit, None] * nrm[hit]) + bv
        # virtual walls
        ml = p[:, 0] < self.left_wall_x + 0.01
        v[ml, 0] = np.maximum(v[ml, 0], 0.0)
        my = p[:, 1] > self.y_limit - 0.01
        v[my, 1] = np.minimum(v[my, 1], 0.0)
        my = p[:, 1] < -self.y_limit + 0.01
        v[my, 1] = np.maximum(v[my, 1], 0.0)

    # ------------------------------------------------------------- substep
    def _step(self, dt, t_now):
        v = self.vel + dt * G_ACC
        sp = np.linalg.norm(v, axis=1)
        scale = np.ones(self.n)
        fast = sp > self.max_speed
        scale[fast] = self.max_speed / sp[fast]
        v *= scale[:, None]
        p = self.pos + dt * v

        gate_center, gate_half, _, gate_vel = self._gate(t_now + dt)

        I, J = self._pairs(p)
        for _ in range(self.pbf_iters):
            self._project(p, I, J)
            self._collide(p, gate_center, gate_half)

        v = (p - self.pos) / dt

        # XSPH viscosity
        if len(I):
            I2, J2, w, _ = self._kernel_pairs(p, I, J)
            if len(I2):
                k = self.xsph_c * self.mp / self.rho0
                for d in range(3):
                    dv = v[J2, d] - v[I2, d]
                    v[:, d] += k * np.bincount(I2, weights=w * dv,
                                               minlength=self.n)

        sp = np.linalg.norm(v, axis=1)
        scale = np.ones(self.n)
        fast = sp > self.max_speed
        scale[fast] = self.max_speed / sp[fast]
        v *= scale[:, None]

        self._vel_fix(v, p, gate_center, gate_half, gate_vel)

        # numerical safety net
        bad = ~np.isfinite(p)
        if bad.any():
            rows = np.nonzero(bad.any(axis=1))[0]
            p[rows] = self.pos[rows]
            v[rows] = 0.0
        badv = ~np.isfinite(v)
        if badv.any():
            rows = np.nonzero(badv.any(axis=1))[0]
            v[rows] = 0.0

        self.pos = p
        self.vel = v
        self.t += dt
        self.step_count += 1

    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t < target - 1e-9 and guard < 100000:
            dt = min(self.sub_dt, target - self.t)
            self._step(dt, self.t)
            guard += 1
        if self.t > target:
            self.t = target
        return self._frame(target)

    # ----------------------------------------------------- boundary samples
    def _init_boundary_samples(self):
        pts, nrm = [], []

        def add(P, N):
            P = np.asarray(P, float).reshape(-1, 3)
            N = np.asarray(N, float).reshape(-1, 3)
            pts.append(P)
            nrm.append(N)

        # tank floor
        xs = np.linspace(-3.0, 3.0, 16)
        ys = np.linspace(-0.84, 0.84, 7)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        P = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        keep = ~((P[:, 0] > 0.24) & (P[:, 0] < 2.66) & (np.abs(P[:, 1]) < 0.85))
        add(P[keep], np.tile([0.0, 0.0, 1.0], (keep.sum(), 1)))

        # shelf top
        xs = np.linspace(0.35, 2.55, 8)
        ys = np.linspace(-0.7, 0.7, 7)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        P = np.column_stack([X.ravel(), Y.ravel(),
                             np.full(X.size, 0.6)])
        add(P, np.tile([0.0, 0.0, 1.0], (X.size, 1)))

        # shelf upstream / downstream faces
        ys = np.linspace(-0.7, 0.7, 7)
        zs = np.linspace(0.1, 0.5, 4)
        for xf, nvec in ((0.25, [-1.0, 0.0, 0.0]), (2.65, [1.0, 0.0, 0.0])):
            Y, Z = np.meshgrid(ys, zs, indexing="ij")
            P = np.column_stack([np.full(Y.size, xf), Y.ravel(), Z.ravel()])
            add(P, np.tile(nvec, (Y.size, 1)))

        # shelf side faces
        xs = np.linspace(0.45, 2.45, 5)
        zs = np.linspace(0.15, 0.55, 4)
        for yf, nvec in ((-0.84, [0.0, -1.0, 0.0]), (0.84, [0.0, 1.0, 0.0])):
            X, Z = np.meshgrid(xs, zs, indexing="ij")
            P = np.column_stack([X.ravel(), np.full(X.size, yf), Z.ravel()])
            add(P, np.tile(nvec, (X.size, 1)))

        # side walls (interior faces)
        xs = np.linspace(-3.0, 3.0, 13)
        zs = np.linspace(0.1, 1.0, 5)
        for yf, nvec in ((-0.92, [0.0, 1.0, 0.0]), (0.92, [0.0, -1.0, 0.0])):
            X, Z = np.meshgrid(xs, zs, indexing="ij")
            P = np.column_stack([X.ravel(), np.full(X.size, yf), Z.ravel()])
            add(P, np.tile(nvec, (X.size, 1)))

        # downstream end wall
        ys = np.linspace(-0.84, 0.84, 7)
        zs = np.linspace(0.1, 1.0, 5)
        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        P = np.column_stack([np.full(Y.size, 3.06), Y.ravel(), Z.ravel()])
        add(P, np.tile([-1.0, 0.0, 0.0], (Y.size, 1)))

        # virtual upstream end wall
        ys = np.linspace(-0.84, 0.84, 5)
        zs = np.linspace(0.1, 1.0, 4)
        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        P = np.column_stack([np.full(Y.size, self.left_wall_x),
                             Y.ravel(), Z.ravel()])
        add(P, np.tile([1.0, 0.0, 0.0], (Y.size, 1)))

        self.bpos_static = np.concatenate(pts)
        self.bnrm_static = _unit(np.concatenate(nrm))

    def _gate_samples(self, t):
        center, half, zb, vg = self._gate(t)
        if zb >= 1.7:
            zc = np.zeros((0, 3))
            return zc, zc, zc
        pts, nrm, tgt = [], [], []
        vel = np.array([0.0, 0.0, vg])

        def add(P, N, T):
            P = np.asarray(P, float).reshape(-1, 3)
            N = np.asarray(N, float).reshape(-1, 3)
            T = np.asarray(T, float).reshape(-1, 3)
            pts.append(P)
            nrm.append(N)
            tgt.append(T)

        # gate bottom lip
        ys = np.linspace(-0.6, 0.6, 5)
        P = np.column_stack([np.full(5, -0.45), ys, np.full(5, zb)])
        add(P, np.tile([0.0, 0.0, -1.0], (5, 1)), np.tile(vel, (5, 1)))
        # gate faces normal to the flow
        zs = np.linspace(zb + 0.1, min(zb + 1.0, 1.3), 3)
        for xf, nvec in ((-0.49, [-1.0, 0.0, 0.0]), (-0.41, [1.0, 0.0, 0.0])):
            Y, Z = np.meshgrid(ys, zs, indexing="ij")
            P = np.column_stack([np.full(Y.size, xf), Y.ravel(), Z.ravel()])
            add(P, np.tile(nvec, (Y.size, 1)), np.tile(vel, (Y.size, 1)))
        return (np.concatenate(pts), _unit(np.concatenate(nrm)),
                np.concatenate(tgt))

    # -------------------------------------------------------- grid helpers
    def _sdf_boxes(self, centers, boxes):
        """Negative inside any box."""
        inside = np.zeros(centers.shape[0], bool)
        for c, e in boxes:
            q = np.abs(centers - c) - e
            d = np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(
                q.max(axis=1), 0.0)
            inside |= d < 0.0
        return inside

    def _static_active_mask(self):
        X, Y, Z = np.meshgrid(0.5 * (self.gx[:-1] + self.gx[1:]),
                              0.5 * (self.gy[:-1] + self.gy[1:]),
                              0.5 * (self.gz[:-1] + self.gz[1:]),
                              indexing="ij")
        centers = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        blocked = self._sdf_boxes(centers, self.static_boxes)
        return (~blocked).reshape(self.gshape)

    # ---------------------------------------------------------------- frame
    def _frame(self, t):
        pos = np.array(self.pos, float)
        vel = np.array(self.vel, float)

        counts, _ = np.histogramdd(pos, bins=self.edges)
        counts = np.asarray(counts, float)
        vel_grid = np.zeros(self.gshape + (3,))
        for k in range(3):
            Hk, _ = np.histogramdd(pos, bins=self.edges, weights=vel[:, k])
            vel_grid[..., k] = np.asarray(Hk) / np.maximum(counts, 1.0)

        # signed liquid level set from particle occupancy
        filled = counts >= 1.0
        if filled.any() and not filled.all():
            filled = ndimage.binary_closing(filled)
            d_out = ndimage.distance_transform_edt(~filled, sampling=self.dxyz)
            d_in = ndimage.distance_transform_edt(filled, sampling=self.dxyz)
            level = np.where(filled, -d_in, d_out)
        else:
            level = np.where(filled, -self.spacing, self.spacing)

        # active mask: cells not blocked by solids (incl. current gate)
        gate_center, gate_half, _, _ = self._gate(t)
        active = self.base_active.copy()
        X, Y, Z = np.meshgrid(0.5 * (self.gx[:-1] + self.gx[1:]),
                              0.5 * (self.gy[:-1] + self.gy[1:]),
                              0.5 * (self.gz[:-1] + self.gz[1:]),
                              indexing="ij")
        centers = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        gate_block = self._sdf_boxes(centers, [(gate_center, gate_half)])
        active &= (~gate_block).reshape(self.gshape)

        # boundary samples
        gpos, gnrm, gtgt = self._gate_samples(t)
        B0 = len(self.bpos_static)
        bpos = np.concatenate([self.bpos_static, gpos]) if len(gpos) \
            else self.bpos_static
        bnrm = np.concatenate([self.bnrm_static, gnrm]) if len(gnrm) \
            else self.bnrm_static
        btgt = np.concatenate([np.zeros((B0, 3)), gtgt]) if len(gtgt) \
            else np.zeros((B0, 3))
        bvel = btgt.copy()
        if len(bpos):
            tree = cKDTree(pos)
            neighbors = tree.query_ball_point(bpos, r=0.3)
            for k, idx in enumerate(neighbors):
                if idx:
                    vf = vel[idx].mean(axis=0)
                    vrel = vf - btgt[k]
                    vn = float(vrel @ bnrm[k])
                    bvel[k] = btgt[k] + (vrel - vn * bnrm[k])

        attributes = {
            "mass_quantity": self.mass.copy(),
            "particle_radius": self.particle_radius.copy(),
            "material_label": self.labels.astype(np.float64),
            "deformation_gradient": self.eye3.copy(),
        }
        fields = {
            "active_mask": active,
            "liquid_level_set": np.asarray(level, float),
            "velocity_field": vel_grid,
            "boundary_normal": bnrm,
            "boundary_target_velocity": btgt,
            "boundary_velocity": bvel,
            "cumulative_mass_source": np.zeros((1,)),
        }
        scalars = {
            "contact_length_scale": self.spacing,
            "grid_spacing": self.spacing,
            "characteristic_length": self.char_len,
            "reference_velocity": self.ref_vel,
        }

        from worldsolver.contracts import StateFrame
        frame = StateFrame(
            float(t), pos, vel, mass=self.mass.copy(),
            body_id=self.labels.copy(),
            attributes=attributes, fields=fields, scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "position_based_fluids",
            "particles": int(self.n),
            "rest_density_kernel": float(self.rho0),
            "particle_mass_kg": float(self.mp),
            "kernel_support_m": float(self.H),
            "substep_dt": float(self.sub_dt),
            "substeps_taken": int(self.step_count),
            "current_time": float(self.t),
        }
