"""Self-contained multiphysics solver for 050_water_collapses_towers.

Physical model (implemented from scratch on NumPy/SciPy):

* Water: weakly compressible SPH.  Cubic-spline kernel, Tait equation of
  state, Monaghan artificial viscosity, symplectic-Euler substepping under a
  CFL-limited step.
* Coupling: Akinci-style one-layer boundary particles for the static basin and
  for every dynamic rigid body.  Boundary particles carry a rest-density
  weighted volume, contribute to fluid density and pressure, and receive the
  equal-and-opposite reaction, which is reduced to a force/torque pair on the
  owning rigid body (two-way momentum exchange).
* Rigid towers: 18 free bodies (cylinders and cubes) integrated with quaternion
  attitude and analytic inertia tensors.  Contact against the basin and between
  bodies uses a penalty (spring-damper) law with regularized Coulomb friction
  evaluated on a sphere decomposition of each shape.

Nothing is key-framed: every reported state comes from time-stepping the model
from the frozen initial condition.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_RHO0 = 1000.0


def _w_kernel(r, h):
    """Cubic spline kernel with support 2h (3D normalization)."""
    q = np.asarray(r, dtype=float) / h
    sigma = 1.0 / (np.pi * h ** 3)
    out = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    q1 = q[m1]
    out[m1] = 1.0 - 1.5 * q1 ** 2 + 0.75 * q1 ** 3
    q2 = q[m2]
    out[m2] = 0.25 * (2.0 - q2) ** 3
    return sigma * out


def _grad_coeff(r, h):
    """Scalar s with grad_i W(x_i - x_j) = s * (x_i - x_j)."""
    r = np.asarray(r, dtype=float)
    q = r / h
    sigma = 1.0 / (np.pi * h ** 3)
    dw = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    q1 = q[m1]
    dw[m1] = -3.0 * q1 + 2.25 * q1 ** 2
    q2 = q[m2]
    dw[m2] = -0.75 * (2.0 - q2) ** 2
    return sigma * dw / (h * np.maximum(r, 1e-9))


def _scatter3(index, values, n):
    out = np.zeros((n, 3))
    if len(index):
        idx = np.asarray(index)
        for k in range(3):
            out[:, k] = np.bincount(idx, weights=values[:, k], minlength=n)[:n]
    return out


def _quat_to_mat(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((len(q), 3, 3))
    m[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    m[:, 0, 1] = 2.0 * (x * y - z * w)
    m[:, 0, 2] = 2.0 * (x * z + y * w)
    m[:, 1, 0] = 2.0 * (x * y + z * w)
    m[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    m[:, 1, 2] = 2.0 * (y * z - x * w)
    m[:, 2, 0] = 2.0 * (x * z - y * w)
    m[:, 2, 1] = 2.0 * (y * z + x * w)
    m[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return m


def _quat_rate(omega, q):
    """dq/dt = 0.5 * (omega, 0) * q for world-frame omega, xyzw layout."""
    wx, wy, wz = omega[:, 0], omega[:, 1], omega[:, 2]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.empty_like(q)
    out[:, 0] = 0.5 * (w * wx + wy * z - wz * y)
    out[:, 1] = 0.5 * (w * wy + wz * x - wx * z)
    out[:, 2] = 0.5 * (w * wz + wx * y - wy * x)
    out[:, 3] = 0.5 * (-wx * x - wy * y - wz * z)
    return out


def _cylinder_boundary(radius, length, spacing):
    nz = max(2, int(round(length / spacing)))
    zs = -0.5 * length + (np.arange(nz) + 0.5) * (length / nz)
    nt = max(4, int(round(2.0 * np.pi * radius / spacing)))
    pts = []
    for k, z in enumerate(zs):
        ang = 2.0 * np.pi * (np.arange(nt) + 0.5 * (k % 2)) / nt
        pts.append(np.column_stack((radius * np.cos(ang), radius * np.sin(ang),
                                    np.full(nt, z))))
    ang4 = 2.0 * np.pi * np.arange(4) / 4.0
    cap_xy = np.vstack((np.zeros((1, 2)),
                        np.column_stack((0.5 * radius * np.cos(ang4),
                                         0.5 * radius * np.sin(ang4)))))
    for sign in (-1.0, 1.0):
        pts.append(np.column_stack((cap_xy, np.full(len(cap_xy), sign * 0.5 * length))))
    return np.vstack(pts)


def _box_boundary(hx, hy, hz, spacing):
    def axis(half):
        n = max(2, int(round(2.0 * half / spacing)))
        return -half + (np.arange(n) + 0.5) * (2.0 * half / n)

    xs, ys, zs = axis(hx), axis(hy), axis(hz)
    pts = []
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    for sx in (-hx, hx):
        pts.append(np.column_stack((np.full(yy.size, sx), yy.ravel(), zz.ravel())))
    xx, zz2 = np.meshgrid(xs, zs, indexing="ij")
    for sy in (-hy, hy):
        pts.append(np.column_stack((xx.ravel(), np.full(xx.size, sy), zz2.ravel())))
    xx2, yy2 = np.meshgrid(xs, ys, indexing="ij")
    for sz in (-hz, hz):
        pts.append(np.column_stack((xx2.ravel(), yy2.ravel(), np.full(xx2.size, sz))))
    return np.vstack(pts)


def _cylinder_spheres(radius, length):
    nz = max(2, int(np.ceil((length - 2.0 * radius) / (0.9 * radius))) + 1)
    zc = np.linspace(-0.5 * length + radius, 0.5 * length - radius, nz)
    centers = np.column_stack((np.zeros(nz), np.zeros(nz), zc))
    radii = np.full(nz, radius)
    rs = 0.5 * radius
    ang = 2.0 * np.pi * np.arange(4) / 4.0
    for sign in (-1.0, 1.0):
        ring = np.column_stack((rs * np.cos(ang), rs * np.sin(ang),
                                np.full(4, sign * (0.5 * length - rs))))
        centers = np.vstack((centers, ring))
        radii = np.concatenate((radii, np.full(4, rs)))
    return centers, radii


def _box_spheres(hx, hy, hz):
    rs = float(np.hypot(0.5 * hx, 0.5 * hy))
    nz = max(2, int(np.ceil((2.0 * hz - 2.0 * rs) / (1.6 * rs))) + 1)
    zc = np.linspace(-hz + rs, hz - rs, nz)
    offs = np.array([[-0.5 * hx, -0.5 * hy], [0.5 * hx, -0.5 * hy],
                     [-0.5 * hx, 0.5 * hy], [0.5 * hx, 0.5 * hy]])
    blocks = [np.column_stack((np.full(nz, ox), np.full(nz, oy), zc)) for ox, oy in offs]
    centers = np.vstack(blocks)
    return centers, np.full(len(centers), rs)


def _psi_from_cloud(points, h):
    """Akinci boundary volume weight Psi = rho0 / sum_k W."""
    total = np.full(len(points), float(_w_kernel(np.zeros(1), h)[0]))
    if len(points) > 1:
        pairs = np.asarray(cKDTree(points).query_pairs(2.0 * h,
                                                       output_type="ndarray")).reshape(-1, 2)
        if len(pairs):
            d = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
            w = _w_kernel(d, h)
            total += np.bincount(pairs[:, 0], weights=w, minlength=len(points))
            total += np.bincount(pairs[:, 1], weights=w, minlength=len(points))
    return _RHO0 / np.maximum(total, 1e-9)


def _basin_boundary(dom_min, dom_max, spacing, wall_top):
    def grid(lo, hi):
        n = max(2, int(round((hi - lo) / spacing)) + 1)
        return np.linspace(lo, hi, n)

    xs = grid(dom_min[0], dom_max[0])
    ys = grid(dom_min[1], dom_max[1])
    zs = grid(dom_min[2] + spacing * 0.5, wall_top)
    pts = []
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    pts.append(np.column_stack((xx.ravel(), yy.ravel(),
                                np.full(xx.size, dom_min[2]))))
    xx2, zz = np.meshgrid(xs, zs, indexing="ij")
    for sy in (dom_min[1], dom_max[1]):
        pts.append(np.column_stack((xx2.ravel(), np.full(xx2.size, sy), zz.ravel())))
    yy2, zz2 = np.meshgrid(ys, zs, indexing="ij")
    for sx in (dom_min[0], dom_max[0]):
        pts.append(np.column_stack((np.full(yy2.size, sx), yy2.ravel(), zz2.ravel())))
    return np.vstack(pts)


class BenchmarkSolver:
    """WCSPH water two-way coupled to penalty-contact rigid towers."""

    # ---- frozen model constants -------------------------------------------
    DX = 0.16                 # fluid sample spacing (scene initial state)
    H_FACTOR = 1.2            # smoothing length / spacing
    C_SOUND = 20.0            # artificial speed of sound [m/s]
    GAMMA = 7.0               # Tait exponent
    ALPHA_F = 0.10            # artificial viscosity, fluid-fluid
    ALPHA_B = 0.25            # artificial viscosity, fluid-boundary
    BSPACING = 0.115          # rigid boundary-particle spacing
    SSPACING = 0.20           # basin boundary-particle spacing
    WALL_TOP = 1.30           # sampled basin wall height
    K_FLUID_RIGID = 3.0e4     # fluid non-penetration penalty [N/m]
    C_FLUID_RIGID = 60.0
    SKIN = 0.02
    K_CONTACT = 5.0e5         # rigid contact penalty stiffness [N/m]
    C_CONTACT = 3.5e3         # rigid contact damping [N s/m]
    MU_CONTACT = 0.5          # Coulomb friction coefficient
    V_EPS = 0.05              # friction regularization velocity [m/s]
    V_MAX_FLUID = 25.0
    V_MAX_BODY = 15.0
    W_MAX_BODY = 25.0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        cfg = dict(getattr(scene, "config", {}) or {})
        self._rng = np.random.default_rng(int(seed))
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        basin = (cfg.get("boundaries", {}) or {}).get("basin_interior", {})
        self.dom_min = np.asarray(basin.get("min", [-4.0, -2.0, 0.0]), dtype=float)
        self.dom_max = np.asarray(basin.get("max", [4.0, 2.0, 2.2]), dtype=float)

        frame0 = scene.initial_state()
        pos0 = np.asarray(frame0.position, dtype=float)
        vel0 = np.asarray(frame0.velocity, dtype=float)
        mass0 = np.asarray(frame0.mass, dtype=float)
        groups = np.asarray(frame0.attributes["scene_group"], dtype=float)
        self.groups = groups.copy()
        fluid_mask = groups == 10.0
        self.n_fluid = int(fluid_mask.sum())
        self.n_body = int((~fluid_mask).sum())

        # ---- fluid state
        self.fx = pos0[fluid_mask].copy()
        self.fv = vel0[fluid_mask].copy()
        self.fm = mass0[fluid_mask].copy()
        self.h = self.H_FACTOR * self.DX
        self.support = 2.0 * self.h
        self.rho0 = float(cfg.get("initial_conditions", {}).get(
            "water_rest_density", _RHO0))
        self.B_tait = self.rho0 * self.C_SOUND ** 2 / self.GAMMA
        self.rho = np.full(self.n_fluid, self.rho0)

        # ---- rigid bodies (order preserved from the scene initial state)
        bg = groups[~fluid_mask]
        self.bgroups = bg.copy()
        self.bx = pos0[~fluid_mask].copy()
        self.bv = vel0[~fluid_mask].copy()
        self.bq = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n_body, 1))
        self.bw = np.zeros((self.n_body, 3))
        self.bm = mass0[~fluid_mask].copy()
        self.is_cube = (np.mod(bg, 10.0) == 1.0)

        cyl_r, cyl_l = 0.115, 1.85
        hx, hy, hz = 0.115, 0.115, 0.925
        inertia = np.zeros((self.n_body, 3, 3))
        for i in range(self.n_body):
            m = self.bm[i]
            if self.is_cube[i]:
                ixx = m * ((2.0 * hy) ** 2 + (2.0 * hz) ** 2) / 12.0
                iyy = m * ((2.0 * hx) ** 2 + (2.0 * hz) ** 2) / 12.0
                izz = m * ((2.0 * hx) ** 2 + (2.0 * hy) ** 2) / 12.0
            else:
                ixx = iyy = m * (3.0 * cyl_r ** 2 + cyl_l ** 2) / 12.0
                izz = 0.5 * m * cyl_r ** 2
            inertia[i] = np.diag([ixx, iyy, izz])
        self.Ib = inertia
        self.Ib_inv = np.linalg.inv(inertia)

        # ---- boundary particle clouds
        cyl_pts = _cylinder_boundary(cyl_r, cyl_l, self.BSPACING)
        box_pts = _box_boundary(hx, hy, hz, self.BSPACING)
        cyl_psi = _psi_from_cloud(cyl_pts, self.h) * (self.rho0 / _RHO0)
        box_psi = _psi_from_cloud(box_pts, self.h) * (self.rho0 / _RHO0)
        cyl_sc, cyl_sr = _cylinder_spheres(cyl_r, cyl_l)
        box_sc, box_sr = _box_spheres(hx, hy, hz)

        bpts, bpsi, bbody = [], [], []
        scen, srad, sbody = [], [], []
        for i in range(self.n_body):
            pts, psi = (box_pts, box_psi) if self.is_cube[i] else (cyl_pts, cyl_psi)
            bpts.append(pts)
            bpsi.append(psi)
            bbody.append(np.full(len(pts), i, dtype=int))
            sc, sr = (box_sc, box_sr) if self.is_cube[i] else (cyl_sc, cyl_sr)
            scen.append(sc)
            srad.append(sr)
            sbody.append(np.full(len(sc), i, dtype=int))
        self.bp_local = np.vstack(bpts)
        self.bp_psi = np.concatenate(bpsi)
        self.bp_body = np.concatenate(bbody)
        self.sc_local = np.vstack(scen)
        self.sr = np.concatenate(srad)
        self.s_body = np.concatenate(sbody)
        counts = np.bincount(self.s_body, minlength=self.n_body)
        self.s_offsets = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(int)
        self.s_counts = counts

        # ---- static basin boundary
        self.sb_x = _basin_boundary(self.dom_min, self.dom_max, self.SSPACING,
                                    self.WALL_TOP)
        self.sb_psi = _psi_from_cloud(self.sb_x, self.h) * (self.rho0 / _RHO0)
        self.n_static = len(self.sb_x)
        self.n_bp = len(self.bp_local)

        # ---- body-pair bookkeeping
        pairs = [(i, j) for i in range(self.n_body) for j in range(i + 1, self.n_body)]
        self.body_pairs = np.asarray(pairs, dtype=int)
        self.n_pairs = len(pairs)
        pidx = -np.ones((self.n_body, self.n_body), dtype=int)
        for k, (i, j) in enumerate(pairs):
            pidx[i, j] = k
            pidx[j, i] = k
        self.pair_index = pidx

        # ---- material bookkeeping (water / light cylinders / heavy cubes)
        self.material_id = np.zeros(self.n_fluid + self.n_body, dtype=int)
        self.material_id[self.n_fluid:] = np.where(self.is_cube, 2, 1)
        self.contact_length_scale = float(self.DX)

        # ---- time stepping
        self.sample_dt = float(cfg.get("sample_dt", 0.02))
        self.n_sub = 12
        self.dt = self.sample_dt / self.n_sub
        self.time = 0.0
        self._steps = 0
        self._min_gap = 0.0
        self._Rm = _quat_to_mat(self.bq)
        self._f_clamp = np.zeros((self.n_fluid, 3))
        self._b_clamp = np.zeros((self.n_body, 3))
        self._compute_forces()

    # ------------------------------------------------------------- kinematics
    def _body_world_points(self, local, body_idx):
        return self.bx[body_idx] + np.einsum(
            "nij,nj->ni", self._Rm[body_idx], local)

    def _point_velocity(self, world_pts, body_idx):
        r = world_pts - self.bx[body_idx]
        return self.bv[body_idx] + np.cross(self.bw[body_idx], r)

    # ------------------------------------------------------------------ forces
    def _compute_forces(self):
        nf, nst, nbp = self.n_fluid, self.n_static, self.n_bp
        self._Rm = _quat_to_mat(self.bq)
        bp_w = self._body_world_points(self.bp_local, self.bp_body)
        bp_v = self._point_velocity(bp_w, self.bp_body)
        s_w = self._body_world_points(self.sc_local, self.s_body)
        s_v = self._point_velocity(s_w, self.s_body)
        self.s_world = s_w

        f_force = np.zeros((nf, 3))
        f_ext = self.fm[:, None] * self.gravity[None, :]
        b_force = self.bm[:, None] * self.gravity[None, :]
        b_ext = b_force.copy()
        b_torque = np.zeros((self.n_body, 3))
        coupling = np.zeros((self.n_body, 3))
        pair_force = np.zeros((self.n_pairs, 3))

        all_x = np.vstack((self.fx, self.sb_x, bp_w))
        pairs = np.asarray(cKDTree(all_x).query_pairs(
            self.support, output_type="ndarray")).reshape(-1, 2)
        if len(pairs):
            pairs = pairs[pairs[:, 0] < nf]
        if len(pairs):
            i_all, j_all = pairs[:, 0], pairs[:, 1]
            ff = j_all < nf
            ii, jj = i_all[ff], j_all[ff]
            fb = ~ff
            bi, bj = i_all[fb], j_all[fb] - nf
        else:
            ii = jj = bi = bj = np.zeros(0, dtype=int)

        # ---- density (SPH summation incl. boundary contributions)
        rho = self.fm * float(_w_kernel(np.zeros(1), self.h)[0])
        if len(ii):
            dij = self.fx[ii] - self.fx[jj]
            rij = np.linalg.norm(dij, axis=1)
            wij = _w_kernel(rij, self.h)
            rho += np.bincount(ii, weights=self.fm[jj] * wij, minlength=nf)
            rho += np.bincount(jj, weights=self.fm[ii] * wij, minlength=nf)
        if len(bi):
            b_is_static = bj < nst
            psi_b = np.where(b_is_static,
                             self.sb_psi[np.clip(bj, 0, nst - 1)],
                             self.bp_psi[np.clip(bj - nst, 0, nbp - 1)])
            xb = np.where(b_is_static[:, None],
                          self.sb_x[np.clip(bj, 0, nst - 1)],
                          bp_w[np.clip(bj - nst, 0, nbp - 1)])
            vb = np.where(b_is_static[:, None], 0.0,
                          bp_v[np.clip(bj - nst, 0, nbp - 1)])
            dib = self.fx[bi] - xb
            rib = np.linalg.norm(dib, axis=1)
            rho += np.bincount(bi, weights=psi_b * _w_kernel(rib, self.h),
                               minlength=nf)
        self.rho = np.maximum(rho, 1e-3 * self.rho0)
        press = np.maximum(self.B_tait * ((self.rho / self.rho0) ** self.GAMMA - 1.0),
                           0.0)
        self.pressure = press

        # ---- fluid-fluid pressure + Monaghan artificial viscosity
        if len(ii):
            rho_i, rho_j = self.rho[ii], self.rho[jj]
            pterm = press[ii] / rho_i ** 2 + press[jj] / rho_j ** 2
            vij = self.fv[ii] - self.fv[jj]
            vdotx = np.einsum("ij,ij->i", vij, dij)
            mu = np.where(vdotx < 0.0,
                          self.h * vdotx / (rij ** 2 + 0.01 * self.h ** 2), 0.0)
            visc = -self.ALPHA_F * self.C_SOUND * mu / (0.5 * (rho_i + rho_j))
            s = _grad_coeff(rij, self.h)
            scal = -self.fm[ii] * self.fm[jj] * (pterm + visc) * s
            fpair = scal[:, None] * dij
            f_force += _scatter3(ii, fpair, nf)
            f_force -= _scatter3(jj, fpair, nf)

        # ---- fluid-boundary coupling (Akinci): pressure + wall viscosity
        if len(bi):
            rho_i = self.rho[bi]
            sb = _grad_coeff(rib, self.h)
            vib = self.fv[bi] - vb
            vdotxb = np.einsum("ij,ij->i", vib, dib)
            mub = np.where(vdotxb < 0.0,
                           self.h * vdotxb / (rib ** 2 + 0.01 * self.h ** 2), 0.0)
            viscb = -self.ALPHA_B * self.C_SOUND * mub / rho_i
            ptermb = press[bi] / rho_i ** 2
            scalb = -self.fm[bi] * psi_b * (ptermb + viscb) * sb
            fb_pair = scalb[:, None] * dib
            f_force += _scatter3(bi, fb_pair, nf)
            # reaction on the boundary samples
            react = -fb_pair
            stat = b_is_static
            if np.any(stat):
                # The basin is not part of the sampled set, so its action on
                # the fluid is bookkept as an external force.
                f_ext += _scatter3(bi[stat], fb_pair[stat], nf)
            dyn = ~stat
            if np.any(dyn):
                bidx = self.bp_body[np.clip(bj[dyn] - nst, 0, nbp - 1)]
                rarm = xb[dyn] - self.bx[bidx]
                b_force += _scatter3(bidx, react[dyn], self.n_body)
                coupling += _scatter3(bidx, react[dyn], self.n_body)
                b_torque += _scatter3(bidx, np.cross(rarm, react[dyn]), self.n_body)

        # ---- fluid / rigid non-penetration penalty on the sphere hull
        stree = cKDTree(s_w)
        dist, nn = stree.query(self.fx, k=1)
        rad = self.sr[nn]
        pen = (rad + self.SKIN) - dist
        act = pen > 0.0
        if np.any(act):
            ia = np.nonzero(act)[0]
            sa = nn[ia]
            dvec = self.fx[ia] - s_w[sa]
            dn = np.linalg.norm(dvec, axis=1)
            nrm = np.where(dn[:, None] > 1e-9, dvec / np.maximum(dn, 1e-9)[:, None],
                           np.array([0.0, 0.0, 1.0])[None, :])
            vrel = self.fv[ia] - s_v[sa]
            vn = np.einsum("ij,ij->i", vrel, nrm)
            fn = np.clip(self.K_FLUID_RIGID * pen[ia] - self.C_FLUID_RIGID * vn,
                         0.0, None)
            fvec = fn[:, None] * nrm
            f_force += _scatter3(ia, fvec, nf)
            bidx = self.s_body[sa]
            cpt = s_w[sa] + rad[ia][:, None] * nrm
            b_force += _scatter3(bidx, -fvec, self.n_body)
            coupling += _scatter3(bidx, -fvec, self.n_body)
            b_torque += _scatter3(bidx, np.cross(cpt - self.bx[bidx], -fvec),
                                  self.n_body)

        # ---- rigid / rigid penalty contacts on the sphere decomposition
        rmax = float(self.sr.max())
        sp = np.asarray(cKDTree(s_w).query_pairs(2.0 * rmax,
                                                 output_type="ndarray")).reshape(-1, 2)
        if len(sp):
            sp = sp[self.s_body[sp[:, 0]] != self.s_body[sp[:, 1]]]
        if len(sp):
            a, b = sp[:, 0], sp[:, 1]
            d = s_w[a] - s_w[b]
            dl = np.linalg.norm(d, axis=1)
            gap = dl - (self.sr[a] + self.sr[b])
            hit = gap < 0.0
            if np.any(hit):
                a, b, d, dl, gap = a[hit], b[hit], d[hit], dl[hit], gap[hit]
                nrm = d / np.maximum(dl, 1e-9)[:, None]
                vrel = s_v[a] - s_v[b]
                vn = np.einsum("ij,ij->i", vrel, nrm)
                fn = np.clip(-self.K_CONTACT * gap - self.C_CONTACT * vn, 0.0, None)
                vt = vrel - vn[:, None] * nrm
                vtn = np.linalg.norm(vt, axis=1)
                ft = -(self.MU_CONTACT * fn / (vtn + self.V_EPS))[:, None] * vt
                fvec = fn[:, None] * nrm + ft
                ba, bb = self.s_body[a], self.s_body[b]
                cpt = 0.5 * (s_w[a] + s_w[b])
                b_force += _scatter3(ba, fvec, self.n_body)
                b_force -= _scatter3(bb, fvec, self.n_body)
                b_torque += _scatter3(ba, np.cross(cpt - self.bx[ba], fvec),
                                      self.n_body)
                b_torque += _scatter3(bb, np.cross(cpt - self.bx[bb], -fvec),
                                      self.n_body)
                pk = self.pair_index[ba, bb]
                sgn = np.where(ba < bb, 1.0, -1.0)[:, None]
                pair_force += _scatter3(pk, fvec * sgn, self.n_pairs)

        # ---- rigid / basin penalty contacts (floor + four walls)
        planes = (
            (np.array([0.0, 0.0, 1.0]), self.dom_min[2], 2, 1.0),
            (np.array([1.0, 0.0, 0.0]), self.dom_min[0], 0, 1.0),
            (np.array([-1.0, 0.0, 0.0]), self.dom_max[0], 0, -1.0),
            (np.array([0.0, 1.0, 0.0]), self.dom_min[1], 1, 1.0),
            (np.array([0.0, -1.0, 0.0]), self.dom_max[1], 1, -1.0),
        )
        wall_gaps = []
        for nrm_p, level, axis, sgn_p in planes:
            gap = sgn_p * (s_w[:, axis] - level) - self.sr
            wall_gaps.append(gap)
            hit = gap < 0.0
            if not np.any(hit):
                continue
            idx = np.nonzero(hit)[0]
            nvec = np.tile(nrm_p, (len(idx), 1))
            vrel = s_v[idx]
            vn = np.einsum("ij,ij->i", vrel, nvec)
            fn = np.clip(-self.K_CONTACT * gap[idx] - self.C_CONTACT * vn, 0.0, None)
            vt = vrel - vn[:, None] * nvec
            vtn = np.linalg.norm(vt, axis=1)
            ft = -(self.MU_CONTACT * fn / (vtn + self.V_EPS))[:, None] * vt
            fvec = fn[:, None] * nvec + ft
            bidx = self.s_body[idx]
            cpt = s_w[idx] - self.sr[idx][:, None] * nvec
            b_force += _scatter3(bidx, fvec, self.n_body)
            b_ext += _scatter3(bidx, fvec, self.n_body)
            b_torque += _scatter3(bidx, np.cross(cpt - self.bx[bidx], fvec),
                                  self.n_body)
        self._wall_gaps = np.asarray(wall_gaps)

        self._f_force = f_force + self.fm[:, None] * self.gravity[None, :]
        self._f_ext = f_ext
        self._b_force = b_force
        self._b_ext = b_ext
        self._b_torque = b_torque
        self._coupling = coupling
        self._pair_force = pair_force

    # -------------------------------------------------------------- integration
    def _substep(self, dt: float) -> None:
        self._compute_forces()

        # --- water: symplectic Euler
        self.fv += dt * self._f_force / self.fm[:, None]
        speed = np.linalg.norm(self.fv, axis=1)
        over = speed > self.V_MAX_FLUID
        if np.any(over):
            self.fv[over] *= (self.V_MAX_FLUID / speed[over])[:, None]
        self.fx += dt * self.fv

        # --- basin containment for the fluid samples (recorded as external)
        margin = 0.02
        v_before = self.fv.copy()
        lo = self.dom_min + margin
        hi = self.dom_max - margin
        for k in range(3):
            below = self.fx[:, k] < lo[k]
            if np.any(below):
                self.fx[below, k] = lo[k]
                self.fv[below, k] = np.where(self.fv[below, k] < 0.0, 0.0,
                                             self.fv[below, k])
            above = self.fx[:, k] > hi[k]
            if np.any(above):
                self.fx[above, k] = hi[k]
                self.fv[above, k] = np.where(self.fv[above, k] > 0.0, 0.0,
                                             self.fv[above, k])
        self._f_clamp = self.fm[:, None] * (self.fv - v_before) / dt

        # --- rigid bodies: velocity, position, quaternion attitude
        self.bv += dt * self._b_force / self.bm[:, None]
        bspeed = np.linalg.norm(self.bv, axis=1)
        bover = bspeed > self.V_MAX_BODY
        if np.any(bover):
            self.bv[bover] *= (self.V_MAX_BODY / bspeed[bover])[:, None]
        self.bx += dt * self.bv

        Iw = np.einsum("nij,njk,nlk->nil", self._Rm, self.Ib, self._Rm)
        Lw = np.einsum("nij,nj->ni", Iw, self.bw)
        rhs = self._b_torque - np.cross(self.bw, Lw)
        alpha = np.linalg.solve(Iw, rhs[:, :, None])[:, :, 0]
        self.bw += dt * alpha
        wnorm = np.linalg.norm(self.bw, axis=1)
        wover = wnorm > self.W_MAX_BODY
        if np.any(wover):
            self.bw[wover] *= (self.W_MAX_BODY / wnorm[wover])[:, None]
        self.bq += dt * _quat_rate(self.bw, self.bq)
        self.bq /= np.maximum(np.linalg.norm(self.bq, axis=1, keepdims=True), 1e-12)

        vb_before = self.bv.copy()
        blo = self.dom_min + 0.05
        bhi = self.dom_max - 0.05
        for k in range(3):
            below = self.bx[:, k] < blo[k]
            if np.any(below):
                self.bx[below, k] = blo[k]
                self.bv[below, k] = np.where(self.bv[below, k] < 0.0, 0.0,
                                             self.bv[below, k])
            above = self.bx[:, k] > bhi[k]
            if np.any(above):
                self.bx[above, k] = bhi[k]
                self.bv[above, k] = np.where(self.bv[above, k] > 0.0, 0.0,
                                             self.bv[above, k])
        self._b_clamp = self.bm[:, None] * (self.bv - vb_before) / dt
        self._steps += 1

    # ------------------------------------------------------------- diagnostics
    def _contact_gaps(self) -> np.ndarray:
        """[18 floor] + [72 wall] + [153 body-pair] signed gaps."""
        gaps = [np.minimum.reduceat(row, self.s_offsets) for row in self._wall_gaps]
        floor_gap = gaps[0]
        wall_gap = np.concatenate(gaps[1:])
        sw = self.s_world
        d = np.linalg.norm(sw[:, None, :] - sw[None, :, :], axis=2)
        d -= self.sr[:, None] + self.sr[None, :]
        pair_gap = np.empty(self.n_pairs)
        for k, (i, j) in enumerate(self.body_pairs):
            oi, ci = self.s_offsets[i], self.s_counts[i]
            oj, cj = self.s_offsets[j], self.s_counts[j]
            pair_gap[k] = d[oi:oi + ci, oj:oj + cj].min()
        self._min_gap = float(min(floor_gap.min(), wall_gap.min(), pair_gap.min()))
        return np.concatenate((floor_gap, wall_gap, pair_gap))

    def _emit(self, time: float) -> StateFrame:
        nf, nb = self.n_fluid, self.n_body
        n = nf + nb
        position = np.vstack((self.fx, self.bx))
        velocity = np.vstack((self.fv, self.bv))
        orientation = np.vstack((np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (nf, 1)),
                                 self.bq))
        angular = np.vstack((np.zeros((nf, 3)), self.bw))
        mass = np.concatenate((self.fm, self.bm))
        body_id = np.concatenate((np.zeros(nf, dtype=int),
                                  np.arange(1, nb + 1, dtype=int)))
        body_density = self.bm / np.where(
            self.is_cube, 0.23 * 0.23 * 1.85, np.pi * 0.115 ** 2 * 1.85)
        density = np.concatenate((self.rho, body_density))
        phase = np.concatenate((np.zeros(nf), np.where(self.is_cube, 2.0, 1.0)))

        mass_quantity = np.zeros((n, 3))
        mass_quantity[np.arange(n), self.material_id] = mass

        external = np.vstack((self._f_ext + self._f_clamp,
                              self._b_ext + self._b_clamp))
        interaction = np.vstack((self._coupling, self._pair_force))

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular,
            mass=mass,
            body_id=body_id,
            attributes={
                "scene_group": self.groups.copy(),
                "density": density,
                "phase_id": phase,
                "mass_quantity": mass_quantity,
            },
            fields={
                "density": density.copy(),
                "external_force": external,
                "interaction_force": interaction,
                "contact_signed_distance": self._contact_gaps(),
                "cumulative_mass_source": np.zeros(3),
                "pressure": self.pressure.copy(),
            },
            scalars={
                "contact_length_scale": self.contact_length_scale,
                "characteristic_length": self.contact_length_scale,
                "water_rest_density": self.rho0,
                "liquid_phase_id": 0.0,
                "reference_velocity": 3.2,
                "smoothing_length": self.h,
                "time_step": self.dt,
                "min_contact_gap": self._min_gap,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 1e-9:
            dt = min(self.dt, target - self.time)
            self._substep(dt)
            self.time += dt
        self.time = target
        self._compute_forces()
        return self._emit(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "WCSPH water + Akinci rigid-fluid coupling + penalty rigid contacts",
            "fluid_samples": int(self.n_fluid),
            "rigid_bodies": int(self.n_body),
            "static_boundary_particles": int(self.n_static),
            "rigid_boundary_particles": int(self.n_bp),
            "collision_spheres": int(len(self.sr)),
            "substeps_taken": int(self._steps),
            "dt": float(self.dt),
            "smoothing_length": float(self.h),
            "sound_speed": float(self.C_SOUND),
            "final_time": float(self.time),
            "min_contact_gap": float(self._min_gap),
            "max_fluid_speed": float(np.linalg.norm(self.fv, axis=1).max()),
            "max_body_speed": float(np.linalg.norm(self.bv, axis=1).max()),
        }


