"""Two-way coupled WCSPH fluid + rigid-body solver (boat cargo overload sinking).

Physics implemented here from scratch (NumPy/SciPy only):

* Weakly compressible SPH for the pool water (cubic spline kernel, linear
  equation of state, Monaghan artificial viscosity), integrated with a
  symplectic-Euler scheme at a CFL-stable fixed step.
* Akinci-style boundary particles rigidly attached to the boat hull shell and
  to both cargo spheres.  Fluid pressure/viscous pair forces act on the fluid
  and their exact reactions (force + torque) drive the rigid bodies, giving
  genuine two-way coupling (buoyancy, added mass, slam loads, flooding).
* Analytic solid pool walls/bottom: the kernel half-space integral supplies the
  missing boundary density so hydrostatics are correct, plus a no-penetration
  projection with recorded wall impulses and near-wall friction.
* Rigid bodies (open-top boat hull, light sphere, heavy sphere) integrated with
  quaternion attitude, world-frame inertia, penalty contacts with Coulomb
  friction, and hard fluid impermeability projection on the hull shell.

Nothing about the scenario is scripted: the spheres are simply held at their
frozen start poses until their frozen release times, then released.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------- kernel utils
def _kernel_w(r, h):
    q = np.asarray(r, dtype=float) / h
    sig = 1.0 / (np.pi * h ** 3)
    w = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    w[m1] = sig * (1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3)
    w[m2] = sig * 0.25 * (2.0 - q[m2]) ** 3
    return w


def _kernel_dw(r, h):
    """dW/dr (negative inside the support)."""
    q = np.asarray(r, dtype=float) / h
    sig = 1.0 / (np.pi * h ** 3) / h
    d = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    d[m1] = sig * (-3.0 * q[m1] + 2.25 * q[m1] ** 2)
    d[m2] = sig * (-0.75 * (2.0 - q[m2]) ** 2)
    return d


def _quat_to_mat(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_integrate(q, omega, dt):
    x, y, z, w = q
    wx, wy, wz = omega
    dq = 0.5 * np.array(
        [
            w * wx + y * wz - z * wy,
            w * wy + z * wx - x * wz,
            w * wz + x * wy - y * wx,
            -(x * wx + y * wy + z * wz),
        ]
    )
    qn = q + dq * dt
    n = np.linalg.norm(qn)
    return qn / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def _trapz(y, x, axis=-1):
    """Trapezoidal rule that works on both NumPy 1.x and 2.x."""
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x=x, axis=axis)


def _wall_tables(h, radius, n_s=97):
    """Half-space kernel integral F(s) and |dF/ds| for analytic solid walls.

    F(s) = integral of W over the solid half-space seen by a fluid sample that
    sits a distance ``s`` inside the domain.  It restores the density deficit
    next to a wall so hydrostatic pressure is correct, and -dF/ds supplies the
    matching wall pressure gradient.
    """
    s = np.linspace(0.0, radius, n_s)
    f = np.zeros(n_s)
    t = np.linspace(0.0, 1.0, 96)[None, :]
    for i, si in enumerate(s):
        d = np.linspace(si, radius, 160)[:, None]
        rmax = np.sqrt(np.maximum(radius * radius - d * d, 0.0))
        rho = t * rmax
        integ = _kernel_w(np.sqrt(d * d + rho * rho), h) * 2.0 * np.pi * rho
        inner = _trapz(integ, rho, axis=1)
        f[i] = float(_trapz(inner, d[:, 0], axis=0))
    fp = np.maximum(-np.gradient(f, s), 0.0)
    return s, f, fp


class _Rigid:
    """Minimal rigid body: centre of mass state + attached boundary samples."""

    def __init__(self, mass, com, inertia_local, bpts_local, held_until):
        self.mass = float(mass)
        self.pos = np.asarray(com, dtype=float).copy()   # world centre of mass
        self.vel = np.zeros(3)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.omega = np.zeros(3)
        self.rot = np.eye(3)
        self.inertia = np.asarray(inertia_local, dtype=float)
        self.inertia_inv = np.linalg.inv(self.inertia)
        self.bpts = np.asarray(bpts_local, dtype=float)  # relative to com
        self.held_until = float(held_until)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)

    def world_points(self):
        return self.pos + self.bpts @ self.rot.T

    def point_velocity(self, world_pts):
        return self.vel + np.cross(self.omega, world_pts - self.pos)

    def apply_force(self, force, point):
        self.force += force
        self.torque += np.cross(np.asarray(point) - self.pos, force)

    def integrate(self, dt):
        self.vel += (self.force / self.mass) * dt
        i_world = self.rot @ self.inertia @ self.rot.T
        rhs = self.torque - np.cross(self.omega, i_world @ self.omega)
        self.omega += np.linalg.solve(i_world, rhs) * dt
        sp = np.linalg.norm(self.vel)
        if sp > 12.0:
            self.vel *= 12.0 / sp
        so = np.linalg.norm(self.omega)
        if so > 15.0:
            self.omega *= 15.0 / so
        self.pos += self.vel * dt
        self.quat = _quat_integrate(self.quat, self.omega, dt)
        self.rot = _quat_to_mat(self.quat)

    def apply_impulse(self, imp, point):
        self.vel += imp / self.mass
        i_world = self.rot @ self.inertia @ self.rot.T
        self.omega += np.linalg.solve(
            i_world, np.cross(np.asarray(point) - self.pos, imp)
        )


_BOAT_VERTS = np.array(
    [
        [-0.95, -0.45, 0.00],
        [0.95, -0.45, 0.00],
        [0.95, 0.45, 0.00],
        [-0.95, 0.45, 0.00],
        [-0.72, -0.34, 0.48],
        [0.72, -0.34, 0.48],
        [0.72, 0.34, 0.48],
        [-0.72, 0.34, 0.48],
    ]
)
_BOAT_FACES = [(0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7), (0, 1, 2, 3)]


def _load_boat_mesh(path="assets/open_boat.obj"):
    """Read the frozen hull asset; fall back to the identical embedded copy."""
    try:
        verts, faces = [], []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                p = line.split()
                if not p:
                    continue
                if p[0] == "v":
                    verts.append([float(v) for v in p[1:4]])
                elif p[0] == "f":
                    faces.append(tuple(int(v.split("/")[0]) - 1 for v in p[1:5]))
        verts = np.asarray(verts, dtype=float)
        if len(verts) != 8 or len(faces) != 4:
            raise ValueError("unexpected hull asset")
        faces = [f for f in faces if len(f) == 4]
        faces.append((0, 1, 2, 3))  # closed (impermeable) hull floor plate
        return verts, faces
    except Exception:
        return _BOAT_VERTS.copy(), list(_BOAT_FACES)


def _quad_frames(verts, faces, interior):
    """Per-quad centre, outward normal, in-plane axes and half extents."""
    out = []
    for idx in faces:
        v = verts[list(idx)]
        c = v.mean(axis=0)
        n = np.cross(v[1] - v[0], v[3] - v[0])
        n /= max(np.linalg.norm(n), 1e-12)
        if np.dot(n, c - interior) < 0.0:
            n = -n
        u = 0.5 * (v[1] + v[2]) - 0.5 * (v[0] + v[3])
        u = u - np.dot(u, n) * n
        nu = np.linalg.norm(u)
        u = u / nu if nu > 1e-12 else np.array([1.0, 0.0, 0.0])
        w = np.cross(n, u)
        rel = v - c
        out.append(
            (
                c,
                n,
                u,
                w,
                float(np.abs(rel @ u).max()),
                float(np.abs(rel @ w).max()),
                v,
            )
        )
    return out


def _sample_quad(v, spacing):
    """Bilinear samples covering the exact (possibly trapezoidal) quad."""
    e0 = np.linalg.norm(v[1] - v[0])
    e1 = np.linalg.norm(v[3] - v[0])
    nu = max(int(round(e0 / spacing)), 2)
    nv = max(int(round(e1 / spacing)), 2)
    a = (np.arange(nu) + 0.5) / nu
    b = (np.arange(nv) + 0.5) / nv
    aa, bb = np.meshgrid(a, b, indexing="ij")
    aa = aa.ravel()[:, None]
    bb = bb.ravel()[:, None]
    bot = v[0][None, :] * (1 - aa) + v[1][None, :] * aa
    top = v[3][None, :] * (1 - aa) + v[2][None, :] * aa
    return bot * (1 - bb) + top * bb


def _fib_sphere(n, radius):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return radius * np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )


class BenchmarkSolver:
    """WCSPH water two-way coupled to an open hull and two cargo spheres."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        ic = cfg["initial_conditions"]
        self.gvec = np.asarray(cfg["gravity"], dtype=float)
        self.rho_w = float(cfg["fluid"]["density"])
        self.depth = float(cfg["fluid"]["initial_water_depth"])
        self.dx = float(ic["particle_spacing"])
        self.rng = np.random.default_rng(int(seed))

        # ---- fluid particles (identical lattice to the frozen scene) ----
        frame0 = scene.initial_state()
        bid = np.asarray(frame0.body_id)
        fl = bid == 0
        self.fp = np.asarray(frame0.position, dtype=float)[fl].copy()
        self.fv = np.zeros_like(self.fp)
        self.nf = len(self.fp)
        self.m_f = self.rho_w * self.dx ** 3

        # ---- kernel / EOS constants ----
        self.h = self.dx
        self.R = 2.0 * self.h
        self.w0 = float(_kernel_w(np.array([0.0]), self.h)[0])
        off = np.arange(-2, 3) * self.dx
        gx, gy, gz = np.meshgrid(off, off, off, indexing="ij")
        rr = np.sqrt(gx.ravel() ** 2 + gy.ravel() ** 2 + gz.ravel() ** 2)
        self.rho0 = float(np.sum(_kernel_w(rr, self.h)) * self.m_f)
        self.c = 20.0
        self.cs2 = self.c ** 2
        self.alpha_f = 0.12
        self.alpha_b = 0.35
        self.k_fric = 22.0
        self.dt = 1.0e-3
        self.knn = 40
        self.skin = 0.05
        self.rebuild = 5
        self.v_cap = 8.0
        self.nstep = 0
        self.t = 0.0
        self.pi = np.zeros(0, dtype=np.int64)
        self.pj = np.zeros(0, dtype=np.int64)

        # ---- solid pool walls (analytic half-space boundary) ----
        self.walls = [
            (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.0])),
            (np.array([-1.0, 0.0, 0.0]), np.array([2.32, 0.0, 0.0])),
            (np.array([1.0, 0.0, 0.0]), np.array([-2.32, 0.0, 0.0])),
            (np.array([0.0, -1.0, 0.0]), np.array([0.0, 1.32, 0.0])),
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.32, 0.0])),
        ]
        self.w_tab_s, self.w_tab_f, self.w_tab_fp = _wall_tables(self.h, self.R)
        self._build_bodies(ic)
        self._build_boundary_samples()
        self._reset_budgets()
        self._record_forces(first=True)

    # ------------------------------------------------------------------ setup
    def _build_bodies(self, ic):
        verts, faces = _load_boat_mesh()
        interior = verts.mean(axis=0)
        self.quads = _quad_frames(verts, faces, interior)
        boat_pts = np.vstack([_sample_quad(q[6], 0.09) for q in self.quads])
        boat_mass = float(ic["boat_mass"])
        mp = boat_mass / len(boat_pts)
        r2 = np.einsum("ij,ij->i", boat_pts, boat_pts)
        inertia = mp * (
            np.eye(3) * r2.sum() - np.einsum("ij,ik->jk", boat_pts, boat_pts)
        )
        inertia += np.eye(3) * 1e-3
        self.boat = _Rigid(boat_mass, ic["boat_position"], inertia, boat_pts, -1.0)

        rad = float(ic["sphere_radius"])
        self.sphere_r = rad
        vol = 4.0 / 3.0 * np.pi * rad ** 3
        sph_pts = _fib_sphere(190, rad)
        spheres = []
        for dens, pos, trel in (
            (float(ic["light_density"]), ic["light_sphere_position"],
             float(ic["light_release_time"])),
            (float(ic["heavy_density"]), ic["heavy_sphere_position"],
             float(ic["heavy_release_time"])),
        ):
            m = dens * vol
            iner = np.eye(3) * (0.4 * m * rad ** 2)
            spheres.append(_Rigid(m, pos, iner, sph_pts.copy(), trel))
        self.light, self.heavy = spheres
        self.bodies = [self.boat, self.light, self.heavy]

        # Akinci boundary volumes (static in each body frame)
        psi = []
        owner = []
        for k, body in enumerate(self.bodies):
            pts = body.bpts
            tree = cKDTree(pts)
            pairs = tree.query_ball_point(pts, self.R)
            wsum = np.empty(len(pts))
            for i, nb in enumerate(pairs):
                d = np.linalg.norm(pts[nb] - pts[i], axis=1)
                wsum[i] = _kernel_w(d, self.h).sum()
            vol_b = np.clip(1.0 / np.maximum(wsum, 1e-9), 0.0, self.dx ** 3)
            psi.append(self.rho0 * vol_b)
            owner.append(np.full(len(pts), k, dtype=np.int64))
        self.psi = np.concatenate(psi)
        self.b_owner = np.concatenate(owner)
        self.nb = len(self.psi)
        self.b_off = np.cumsum([0] + [len(b.bpts) for b in self.bodies])
        self.proj_r = rad - 0.035
        self.band = 0.035

    def _build_boundary_samples(self):
        pts, nrm = [], []
        xs = np.linspace(-2.2, 2.2, 9)
        ys = np.linspace(-1.2, 1.2, 5)
        zs = np.linspace(0.06, 0.68, 4)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts.append(np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, 0.03)]))
        nrm.append(np.tile([0.0, 0.0, 1.0], (gx.size, 1)))
        gy2, gz2 = np.meshgrid(ys, zs, indexing="ij")
        for sx, nx in ((2.32, -1.0), (-2.32, 1.0)):
            pts.append(np.column_stack([
                np.full(gy2.size, sx + 0.03 * nx), gy2.ravel(), gz2.ravel()]))
            nrm.append(np.tile([nx, 0.0, 0.0], (gy2.size, 1)))
        gx3, gz3 = np.meshgrid(xs, zs, indexing="ij")
        for sy, ny in ((1.32, -1.0), (-1.32, 1.0)):
            pts.append(np.column_stack([
                gx3.ravel(), np.full(gx3.size, sy + 0.03 * ny), gz3.ravel()]))
            nrm.append(np.tile([0.0, ny, 0.0], (gx3.size, 1)))
        self.wall_pts = np.vstack(pts)
        self.wall_nrm = np.vstack(nrm)
        self.nw = len(self.wall_pts)

    def _reset_budgets(self):
        self.imp_cum = np.zeros((5, 3))
        self.wall_imp_fluid = np.zeros(3)
        self.proj_imp = np.zeros((4, 3))   # water, boat, light, heavy
        n = self.nf + 3
        dw = np.zeros((5, n))
        dw[0, : self.nf] = 1.0
        dw[1, self.nf] = 1.0
        dw[2, self.nf + 1] = 1.0
        dw[3, self.nf + 2] = 1.0
        dw[4, :] = 1.0
        self.domain_weights = dw
        mq = np.zeros((n, 2))
        mq[: self.nf, 0] = self.m_f
        mq[self.nf, 1] = self.boat.mass
        mq[self.nf + 1, 1] = self.light.mass
        mq[self.nf + 2, 1] = self.heavy.mass
        self.mass_quantity = mq
        self.mass_all = mq.sum(axis=1)
        self.v_ref = float(np.sqrt(abs(self.gvec[2]) * self.depth))
        self.p_ref = float(self.mass_all.sum() * self.v_ref)
        self.f_wall_water = np.zeros(3)
        self.f_coup = np.zeros((3, 3))
        self.f_cont = np.zeros((3, 3))
        self.f_wallb = np.zeros((3, 3))
        self.pair_c = np.zeros((3, 3))   # boat-light, boat-heavy, light-heavy
        self.b_vel_meas = np.zeros((self.nw, 3))

    # ------------------------------------------------------- force accounting
    def _record_forces(self, first=False):
        g = self.gvec
        mw = self.m_f * self.nf
        masses = [self.boat.mass, self.light.mass, self.heavy.mass]
        ext = np.zeros((5, 5, 3))
        ext[0, 0] = mw * g
        ext[0, 1] = self.f_wall_water
        ext[0, 2] = -self.f_coup.sum(axis=0)
        for k in range(3):
            ext[k + 1, 0] = masses[k] * g
            ext[k + 1, 1] = self.f_wallb[k]
            ext[k + 1, 2] = self.f_coup[k]
            ext[k + 1, 3] = self.f_cont[k]
        for k, body in enumerate(self.bodies):
            if self.t < body.held_until:
                ext[k + 1, 4] = -ext[k + 1, :4].sum(axis=0)
        ext[4, 0] = (mw + sum(masses)) * g
        ext[4, 1] = self.f_wall_water + self.f_wallb.sum(axis=0)
        ext[4, 4] = ext[1:4, 4].sum(axis=0)
        self.ext_force = ext
        inter = np.zeros((6, 3))
        inter[0] = self.f_coup[0]
        inter[1] = self.f_coup[1]
        inter[2] = self.f_coup[2]
        inter[3] = self.pair_c[0]
        inter[4] = self.pair_c[1]
        inter[5] = self.pair_c[2]
        self.inter_force = inter
        imp = np.zeros((5, 3))
        imp[0] = self.wall_imp_fluid + self.proj_imp[0]
        imp[1] = self.proj_imp[1]
        imp[2] = self.proj_imp[2]
        imp[3] = self.proj_imp[3]
        imp[4] = self.wall_imp_fluid
        self.imp_cum = imp
        if first:
            self.b_vel_meas = np.zeros((self.nw, 3))

    def _measure_boundary_velocity(self):
        tree = cKDTree(self.fp)
        nb = tree.query_ball_point(self.wall_pts, self.R)
        out = np.zeros((self.nw, 3))
        for i, ids in enumerate(nb):
            if not ids:
                continue
            ids = np.asarray(ids)
            d = np.linalg.norm(self.fp[ids] - self.wall_pts[i], axis=1)
            w = _kernel_w(d, self.h)
            s = w.sum()
            if s > 1e-12:
                out[i] = (w[:, None] * self.fv[ids]).sum(axis=0) / s
        self.b_vel_meas = out

    # --------------------------------------------------------------- stepping
    def _rebuild_pairs(self, allp):
        tree = cKDTree(allp)
        _, idx = tree.query(
            self.fp, k=self.knn, distance_upper_bound=self.R + self.skin, workers=-1
        )
        rows = np.repeat(np.arange(self.nf, dtype=np.int64), self.knn)
        cols = idx.ravel().astype(np.int64)
        ok = (cols < len(allp)) & (cols != rows)
        self.pi = rows[ok]
        self.pj = cols[ok]

    def _step(self):
        dt = self.dt
        bp = np.empty((self.nb, 3))
        bv = np.empty((self.nb, 3))
        for k, body in enumerate(self.bodies):
            a0, a1 = self.b_off[k], self.b_off[k + 1]
            wp = body.world_points()
            bp[a0:a1] = wp
            bv[a0:a1] = body.point_velocity(wp)
        self.bp, self.bv = bp, bv
        allp = np.vstack([self.fp, bp])
        if self.nstep % self.rebuild == 0 or len(self.pi) == 0:
            self._rebuild_pairs(allp)
        pi, pj = self.pi, self.pj
        d = allp[pj] - self.fp[pi]
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        r = np.maximum(r, 1e-9)
        w = _kernel_w(r, self.h)
        dwr = _kernel_dw(r, self.h)
        isb = pj >= self.nf
        jb = np.where(isb, pj - self.nf, 0)
        pj_f = np.where(isb, 0, pj)
        mj = np.where(isb, self.psi[jb], self.m_f)

        rho = np.bincount(pi, weights=w * mj, minlength=self.nf)
        rho += self.m_f * self.w0
        for n, p0 in self.walls:
            s = (self.fp - p0) @ n
            rho += self.rho0 * np.interp(
                np.clip(s, 0.0, None), self.w_tab_s, self.w_tab_f
            )
        rho = np.maximum(rho, 1e-3)
        pr = self.cs2 * np.maximum(rho - self.rho0, 0.0)

        p_j = np.where(isb, pr[pi], pr[pj_f])
        rho_j = np.where(isb, self.rho0, rho[pj_f])
        vall = np.vstack([self.fv, bv])
        vij = self.fv[pi] - vall[pj]
        rij = -d
        dotvr = np.einsum("ij,ij->i", vij, rij)
        rho_av = 0.5 * (rho[pi] + rho_j)
        alpha = np.where(isb, self.alpha_b, self.alpha_f)
        visc = np.where(
            dotvr < 0.0,
            -alpha * self.c * self.h * dotvr
            / (rho_av * (r * r + 0.01 * self.h * self.h)),
            0.0,
        )
        coef = mj * (pr[pi] / rho[pi] ** 2 + p_j / rho_j ** 2 + visc)
        contrib = (coef * dwr / r)[:, None] * d

        acc = np.empty((self.nf, 3))
        for k in range(3):
            acc[:, k] = np.bincount(pi, weights=contrib[:, k], minlength=self.nf)

        fb = np.zeros((self.nb, 3))
        if isb.any():
            cf = -self.m_f * contrib[isb]
            ib = jb[isb]
            for k in range(3):
                fb[:, k] = np.bincount(ib, weights=cf[:, k], minlength=self.nb)
        f_coup = np.zeros((3, 3))
        t_coup = np.zeros((3, 3))
        for k, body in enumerate(self.bodies):
            a0, a1 = self.b_off[k], self.b_off[k + 1]
            seg = fb[a0:a1]
            f_coup[k] = seg.sum(axis=0)
            t_coup[k] = np.cross(bp[a0:a1] - body.pos, seg).sum(axis=0)
        self.f_coup = f_coup

        fww = np.zeros(3)
        for n, p0 in self.walls:
            s = (self.fp - p0) @ n
            fpn = np.interp(np.clip(s, 0.0, None), self.w_tab_s, self.w_tab_fp)
            cw = self.rho0 * (pr / rho ** 2 + pr / self.rho0 ** 2) * fpn
            aw = cw[:, None] * n[None, :]
            ramp = np.clip(1.0 - s / (1.5 * self.dx), 0.0, 1.0)
            vt = self.fv - (self.fv @ n)[:, None] * n[None, :]
            aw = aw - (self.k_fric * ramp)[:, None] * vt
            acc += aw
            fww -= self.m_f * aw.sum(axis=0)
        self.f_wall_water = fww

        acc += self.gvec[None, :]
        self.fv += acc * dt
        sp = np.linalg.norm(self.fv, axis=1)
        big = sp > self.v_cap
        if big.any():
            self.fv[big] *= (self.v_cap / sp[big])[:, None]
        self.fp += self.fv * dt
        self._clamp_walls()
        self._rigid_step(f_coup, t_coup, dt)
        self._project_fluid(dt)
        self.nstep += 1
        self.t = self.nstep * dt

    def _clamp_walls(self):
        for n, p0 in self.walls:
            s = (self.fp - p0) @ n
            pen = s < 0.0
            if not pen.any():
                continue
            idx = np.flatnonzero(pen)
            self.fp[idx] -= s[idx, None] * n[None, :]
            vn = self.fv[idx] @ n
            hit = idx[vn < 0.0]
            if len(hit):
                vnh = self.fv[hit] @ n
                self.fv[hit] -= vnh[:, None] * n[None, :]
                self.wall_imp_fluid += self.m_f * float(-vnh.sum()) * n

    # ---------------------------------------------------------------- rigids
    def _rigid_step(self, f_coup, t_coup, dt):
        for body in self.bodies:
            body.force = np.zeros(3)
            body.torque = np.zeros(3)
        f_cont = np.zeros((3, 3))
        f_wallb = np.zeros((3, 3))
        pair_c = np.zeros((3, 3))

        # sphere-sphere penalty contact
        fss = self._sphere_sphere(self.light, self.heavy)
        pair_c[2] = fss
        f_cont[1] += fss
        f_cont[2] -= fss
        # sphere-hull contacts
        for k, sph in ((1, self.light), (2, self.heavy)):
            fs = self._sphere_boat(sph)
            pair_c[k - 1] = fs
            f_cont[k] += fs
            f_cont[0] -= fs
        # pool walls / bottom
        f_wallb[0] = self._boat_walls()
        for k, sph in ((1, self.light), (2, self.heavy)):
            f_wallb[k] = self._sphere_walls(sph)

        for k, body in enumerate(self.bodies):
            body.force += body.mass * self.gvec + f_coup[k]
            body.torque += t_coup[k]
        self.f_cont = f_cont
        self.f_wallb = f_wallb
        self.pair_c = pair_c

        for body in self.bodies:
            if self.t < body.held_until:
                body.force = np.zeros(3)
                body.torque = np.zeros(3)
                body.vel = np.zeros(3)
                body.omega = np.zeros(3)
            else:
                body.integrate(dt)

    def _sphere_sphere(self, a, b):
        d = b.pos - a.pos
        dist = float(np.linalg.norm(d))
        lim = 2.0 * self.sphere_r
        if dist >= lim or dist < 1e-9:
            return np.zeros(3)
        n = d / dist
        pen = lim - dist
        meff = a.mass * b.mass / (a.mass + b.mass)
        k = 3.0e5
        cd = 2.0 * 0.6 * np.sqrt(k * meff)
        vrel = float((b.vel - a.vel) @ n)
        fn = max(k * pen - cd * vrel, 0.0)
        force_on_b = fn * n
        cp = a.pos + n * self.sphere_r
        a.apply_force(-force_on_b, cp)
        b.apply_force(force_on_b, cp)
        return force_on_b

    def _sphere_boat(self, sph):
        """Penalty contact of one cargo sphere against the hull quads."""
        rot = self.boat.rot
        xl = (sph.pos - self.boat.pos) @ rot
        total = np.zeros(3)
        k = 1.2e6
        meff = sph.mass * self.boat.mass / (sph.mass + self.boat.mass)
        cd = 2.0 * 0.4 * np.sqrt(k * meff)
        for c, n, u, wv, hu, hv, _v in self.quads:
            rel = xl - c
            du = float(np.clip(rel @ u, -hu, hu))
            dv = float(np.clip(rel @ wv, -hv, hv))
            cp_l = c + du * u + dv * wv
            delta = xl - cp_l
            dist = float(np.linalg.norm(delta))
            if dist >= self.sphere_r:
                continue
            if dist < 1e-9:
                nrm_l = n
                dist = 1e-9
            else:
                nrm_l = delta / dist
            pen = self.sphere_r - dist
            nw = rot @ nrm_l
            cp_w = self.boat.pos + rot @ cp_l
            vrel = sph.point_velocity(cp_w) - self.boat.point_velocity(cp_w)
            vn = float(vrel @ nw)
            fn = max(k * pen - cd * vn, 0.0)
            vt = vrel - vn * nw
            svt = float(np.linalg.norm(vt))
            ft = np.zeros(3)
            if svt > 1e-6:
                ft = -min(0.5 * fn, 4000.0 * svt) * (vt / svt)
            force = fn * nw + ft
            sph.apply_force(force, cp_w)
            self.boat.apply_force(-force, cp_w)
            total += force
        return total

    def _boat_walls(self):
        pts = self.boat.world_points()
        total = np.zeros(3)
        k = 2.5e4
        cd = 300.0
        for n, p0 in self.walls:
            s = (pts - p0) @ n
            m = s < 0.0
            if not m.any():
                continue
            idx = np.flatnonzero(m)
            vel = self.boat.point_velocity(pts[idx])
            vn = vel @ n
            fn = np.maximum(-k * s[idx] - cd * vn, 0.0)
            forces = fn[:, None] * n[None, :]
            tot = forces.sum(axis=0)
            self.boat.force += tot
            self.boat.torque += np.cross(pts[idx] - self.boat.pos, forces).sum(axis=0)
            total += tot
        return total

    def _sphere_walls(self, sph):
        total = np.zeros(3)
        k = 3.0e5
        cd = 2.0 * 0.7 * np.sqrt(k * sph.mass)
        for n, p0 in self.walls:
            s = float((sph.pos - p0) @ n) - self.sphere_r
            if s >= 0.0:
                continue
            vn = float(sph.vel @ n)
            fn = max(-k * s - cd * vn, 0.0)
            force = fn * n
            sph.apply_force(force, sph.pos - self.sphere_r * n)
            total += force
        return total

    # ------------------------------------------- hard fluid impermeability
    def _project_fluid(self, dt):
        rot = self.boat.rot
        xl = (self.fp - self.boat.pos) @ rot
        imp_boat = np.zeros(3)
        tq_boat = np.zeros(3)
        for c, n, u, wv, hu, hv, _v in self.quads:
            rel = xl - c
            sd = rel @ n
            inside = (np.abs(rel @ u) <= hu) & (np.abs(rel @ wv) <= hv)
            m = inside & (np.abs(sd) < self.band)
            if not m.any():
                continue
            idx = np.flatnonzero(m)
            sgn = np.where(sd[idx] >= 0.0, 1.0, -1.0)
            nw = rot @ n
            shift = sgn * self.band - sd[idx]
            self.fp[idx] += shift[:, None] * nw[None, :]
            xl[idx] += shift[:, None] * n[None, :]
            vb = self.boat.point_velocity(self.fp[idx])
            vn = (self.fv[idx] - vb) @ nw
            app = sgn * vn < 0.0
            if not app.any():
                continue
            hit = idx[app]
            dvn = -((self.fv[hit] - self.boat.point_velocity(self.fp[hit])) @ nw)
            self.fv[hit] += dvn[:, None] * nw[None, :]
            impulses = self.m_f * dvn[:, None] * nw[None, :]
            tot = impulses.sum(axis=0)
            self.proj_imp[0] += tot
            imp_boat -= tot
            tq_boat -= np.cross(self.fp[hit] - self.boat.pos, impulses).sum(axis=0)
        if np.any(imp_boat != 0.0) or np.any(tq_boat != 0.0):
            self.proj_imp[1] += imp_boat
            self.boat.vel += imp_boat / self.boat.mass
            i_world = rot @ self.boat.inertia @ rot.T
            self.boat.omega += np.linalg.solve(i_world, tq_boat)

        for k, sph in ((2, self.light), (3, self.heavy)):
            if self.t < sph.held_until:
                continue
            rel = self.fp - sph.pos
            dist = np.sqrt(np.einsum("ij,ij->i", rel, rel))
            m = dist < self.proj_r
            if not m.any():
                continue
            idx = np.flatnonzero(m)
            dirs = rel[idx] / np.maximum(dist[idx, None], 1e-9)
            self.fp[idx] = sph.pos + dirs * self.proj_r
            vrel = self.fv[idx] - sph.point_velocity(self.fp[idx])
            vn = np.einsum("ij,ij->i", vrel, dirs)
            app = vn < 0.0
            if not app.any():
                continue
            sel = idx[app]
            dsel = dirs[app]
            dvn = -vn[app]
            self.fv[sel] += dvn[:, None] * dsel
            impulses = self.m_f * dvn[:, None] * dsel
            tot = impulses.sum(axis=0)
            self.proj_imp[0] += tot
            self.proj_imp[k] -= tot
            sph.vel -= tot / sph.mass
            i_world = sph.rot @ sph.inertia @ sph.rot.T
            sph.omega -= np.linalg.solve(
                i_world, np.cross(self.fp[sel] - sph.pos, impulses).sum(axis=0)
            )

    # -------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t < target - 1e-9 and guard < 200000:
            self._step()
            guard += 1
        self._measure_boundary_velocity()
        self._record_forces()
        return self._make_frame(target)

    def _make_frame(self, time: float) -> StateFrame:
        rb_pos = np.array([b.pos for b in self.bodies])
        rb_vel = np.array([b.vel for b in self.bodies])
        rb_quat = np.array([b.quat for b in self.bodies])
        rb_om = np.array([b.omega for b in self.bodies])
        rb_m = np.array([b.mass for b in self.bodies])

        pos = np.vstack([self.fp, rb_pos]).astype(np.float32)
        vel = np.vstack([self.fv, rb_vel]).astype(np.float32)
        mass = np.concatenate([np.full(self.nf, self.m_f), rb_m]).astype(np.float32)
        quat = np.vstack(
            [np.tile([0.0, 0.0, 0.0, 1.0], (self.nf, 1)), rb_quat]
        ).astype(np.float32)
        omega = np.vstack([np.zeros((self.nf, 3)), rb_om]).astype(np.float32)
        body_id = np.concatenate(
            [np.zeros(self.nf, dtype=np.int32), np.arange(1, 4, dtype=np.int32)]
        )

        attributes = {
            "phase": body_id.astype(np.float32),
            "phase_id": body_id.astype(np.float32),
            "mass_quantity": self.mass_quantity.astype(np.float32),
            "volume_quantity": (
                self.mass_quantity.sum(axis=1)
                / np.where(body_id == 0, self.rho_w, 1000.0)
            ).astype(np.float32),
        }
        tgt = np.zeros((self.nw, 3), dtype=np.float32)
        fields = {
            "boundary_normal": self.wall_nrm.astype(np.float32),
            "boundary_velocity": self.b_vel_meas.astype(np.float32),
            "boundary_target_velocity": tgt,
            "budget_domain_weights": self.domain_weights.astype(np.float32),
            "budget_external_force": self.ext_force.astype(np.float32),
            "budget_cumulative_external_impulse": self.imp_cum.astype(np.float32),
            "budget_cumulative_inflow_momentum": np.zeros((5, 3), dtype=np.float32),
            "budget_cumulative_outflow_momentum": np.zeros((5, 3), dtype=np.float32),
            "cumulative_mass_source": np.zeros(2, dtype=np.float32),
            "interaction_force": self.inter_force.astype(np.float32),
        }
        scalars = {
            "reference_velocity": self.v_ref,
            "reference_momentum": self.p_ref,
            "liquid_phase_id": 0.0,
        }
        frame = StateFrame(
            time=max(float(time), 0.0),
            position=pos,
            velocity=vel,
            orientation=quat,
            angular_velocity=omega,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "WCSPH fluid + Akinci boundary coupling + rigid bodies",
            "fluid_particles": int(self.nf),
            "boundary_particles": int(self.nb),
            "dt": float(self.dt),
            "sound_speed": float(self.c),
            "rest_density_eos": float(self.rho0),
            "steps_taken": int(self.nstep),
            "boat_z": float(self.boat.pos[2]),
            "light_z": float(self.light.pos[2]),
            "heavy_z": float(self.heavy.pos[2]),
        }

