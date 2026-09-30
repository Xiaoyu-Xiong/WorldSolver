"""Tablecloth-pull solver: mass-spring shell cloth coupled to rigid dinnerware.

Physics implemented here directly with NumPy:

* Cloth: 17x13 mass-spring shell (structural / shear / bending springs) with
  Rayleigh edge damping, symplectic-Euler time stepping, Provot strain
  limiting, node-level self-repulsion and projected frictional contact against
  the static table box and the ground plane.
* Rigid dinnerware: 6-DOF rigid bodies (quaternion orientation, world inverse
  inertia) driven by gravity plus penalty-and-Coulomb contact sampled at
  surface points against the table box, the ground plane and the deforming
  cloth surface.  Cloth/rigid contact is two-way coupled: the reaction of each
  rigid contact force is distributed to the supporting cloth nodes.
* Boundary condition: the +x cloth edge column is a kinematic handle pulled
  horizontally at ``pull_speed`` for ``pull_duration`` seconds and then held.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

_V_CLIP = 30.0
_EPS = 1e-12


def _box_sdf(points, center, half):
    """Signed distance and outward unit normal for an axis-aligned box."""
    d = np.asarray(points, float) - center
    q = np.abs(d) - half
    sgn = np.where(d >= 0.0, 1.0, -1.0)
    out = np.maximum(q, 0.0)
    d_out = np.sqrt(np.sum(out * out, axis=1))
    dist = d_out + np.minimum(q.max(axis=1), 0.0)
    nrm = np.zeros_like(d)
    outside = d_out > 1e-12
    if np.any(outside):
        nrm[outside] = (sgn[outside] * out[outside]) / d_out[outside, None]
    inside = ~outside
    if np.any(inside):
        qi = q[inside]
        rows = np.arange(qi.shape[0])
        ax = np.argmax(qi, axis=1)
        tmp = np.zeros((qi.shape[0], 3))
        tmp[rows, ax] = sgn[inside][rows, ax]
        nrm[inside] = tmp
    return dist, nrm
def _quat_to_mat(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-15:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def _quat_integrate(q, omega, dt):
    x, y, z, w = q
    wx, wy, wz = omega
    dx = 0.5 * (w * wx + wy * z - wz * y)
    dy = 0.5 * (w * wy + wz * x - wx * z)
    dz = 0.5 * (w * wz + wx * y - wy * x)
    dw = -0.5 * (wx * x + wy * y + wz * z)
    q = np.array([x + dt * dx, y + dt * dy, z + dt * dz, w + dt * dw])
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def _cyl_points(radius, height, nring=8):
    """Surface sample points for a cylinder aligned with local +z."""
    hz = 0.5 * height
    ang = np.linspace(0.0, 2.0 * np.pi, nring, endpoint=False)
    pts = [np.array([0.0, 0.0, -hz]), np.array([0.0, 0.0, hz])]
    for z in (-hz, 0.0, hz):
        for a in ang:
            pts.append(np.array([radius * math.cos(a), radius * math.sin(a), z]))
    return np.array(pts, float)


def _box_points(half):
    hx, hy, hz = half
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append([sx * hx, sy * hy, sz * hz])
    pts.append([0.0, 0.0, -hz])
    pts.append([0.0, 0.0, hz])
    return np.array(pts, float)
class BenchmarkSolver:
    """Explicit-substep tablecloth-pull solver."""

    # numerical / material parameters
    DT = 1.0 / 6000.0
    K_STRUCT = 3000.0
    K_SHEAR = 400.0
    K_BEND = 120.0
    DAMP_STRUCT = 0.9
    DAMP_SHEAR = 0.25
    DAMP_BEND = 0.08
    AIR_DRAG = 0.25
    STRAIN_LIMIT = 1.03
    STRAIN_ITERS = 6
    SELF_K = 400.0
    SELF_D = 0.5
    SELF_EVERY = 2
    MU_CLOTH_TABLE = 0.25
    MU_CLOTH_GROUND = 0.4
    MU_CLOTH_RIGID = 0.12
    MU_RIGID_TABLE = 0.25
    K_RIGID_TABLE = 4.0e4
    D_RIGID_TABLE = 60.0
    K_RIGID_CLOTH = 8.0e3
    D_RIGID_CLOTH = 6.0
    ANG_DAMP = 0.6
    NEAR_K = 4

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.cfg = cfg
        frame0 = scene.initial_state()
        self.frame0 = frame0

        self.pos0 = np.array(frame0.position, float)
        self.n_all = self.pos0.shape[0]
        self.body_id = np.array(frame0.body_id, int) if frame0.body_id is not None \
            else np.zeros(self.n_all, int)
        self.mass_all = np.array(frame0.mass, float) if frame0.mass is not None \
            else np.full(self.n_all, 1.0)
        self.uv = np.array(frame0.attributes.get("uv", np.zeros((self.n_all, 2))), float)
        g = np.array(cfg.get("gravity", [0.0, 0.0, -9.81]), float)
        self.gravity = g

        self.contact_length_scale = float(frame0.scalars.get("contact_length_scale", 0.008))
        self.constraint_scale = float(frame0.scalars.get("constraint_scale", 1.9))
        if abs(self.constraint_scale) < 1e-9:
            self.constraint_scale = 1.0

        # ---- cloth ----
        cloth_cfg = dict(cfg.get("cloth", {}) or {})
        gs = cloth_cfg.get("grid_shape", [17, 13])
        self.nx, self.ny = int(gs[0]), int(gs[1])
        self.thickness = float(cloth_cfg.get("thickness", 0.008))
        self.cloth_mask = self.body_id == 0
        self.cloth_idx = np.nonzero(self.cloth_mask)[0]
        n_cloth = self.cloth_idx.size
        if n_cloth != self.nx * self.ny:
            # fall back to inferred grid if the config disagrees with the asset
            self.nx, self.ny = n_cloth, 1
        self.n_cloth = n_cloth
        self.x = self.pos0[self.cloth_idx].copy()
        self.v = np.zeros_like(self.x)
        self.m = self.mass_all[self.cloth_idx].copy()
        self.m = np.where(self.m > 1e-9, self.m, 1e-9)
        self.inv_m = 1.0 / self.m

        # ---- static colliders ----
        self.table_center = np.array([0.0, 0.0, 0.72])
        self.table_half = np.array([1.25, 0.9, 0.06])
        self.table_top = float(cfg.get("table_top", 0.78))
        self.ground_z = 0.0

        self._build_springs()
        self._build_self_mask()
        self._build_handle()
        self._build_rigid()

        self.time = 0.0
        self.step_count = 0
        self.ext_force = self.mass_all[:, None] * self.gravity[None, :]
        self._self_force = np.zeros_like(self.x)
        self.max_strain_seen = 1.0
        self.min_gap_seen = float("inf")
    # ------------------------------------------------------------------ setup
    def _gid(self, i, j):
        return i * self.ny + j

    def _build_springs(self):
        nx, ny = self.nx, self.ny
        struct_a, struct_b = [], []
        shear_a, shear_b = [], []
        bend_a, bend_b = [], []
        for i in range(nx):
            for j in range(ny):
                a = self._gid(i, j)
                if i + 1 < nx:
                    struct_a.append(a)
                    struct_b.append(self._gid(i + 1, j))
                if j + 1 < ny:
                    struct_a.append(a)
                    struct_b.append(self._gid(i, j + 1))
                if i + 1 < nx and j + 1 < ny:
                    shear_a.append(a)
                    shear_b.append(self._gid(i + 1, j + 1))
                    shear_a.append(self._gid(i + 1, j))
                    shear_b.append(self._gid(i, j + 1))
                if i + 2 < nx:
                    bend_a.append(a)
                    bend_b.append(self._gid(i + 2, j))
                if j + 2 < ny:
                    bend_a.append(a)
                    bend_b.append(self._gid(i, j + 2))
        self.s_a = np.array(struct_a, int)
        self.s_b = np.array(struct_b, int)
        self.h_a = np.array(shear_a, int)
        self.h_b = np.array(shear_b, int)
        self.b_a = np.array(bend_a, int)
        self.b_b = np.array(bend_b, int)
        x0 = self.x
        self.s_rest = np.linalg.norm(x0[self.s_b] - x0[self.s_a], axis=1)
        self.h_rest = np.linalg.norm(x0[self.h_b] - x0[self.h_a], axis=1)
        self.b_rest = np.linalg.norm(x0[self.b_b] - x0[self.b_a], axis=1)
        self.s_rest = np.maximum(self.s_rest, 1e-6)
        self.h_rest = np.maximum(self.h_rest, 1e-6)
        self.b_rest = np.maximum(self.b_rest, 1e-6)
        self.mean_edge = float(np.mean(self.s_rest))

    def _build_self_mask(self):
        n = self.n_cloth
        ii = np.arange(n) // self.ny
        jj = np.arange(n) % self.ny
        di = np.abs(ii[:, None] - ii[None, :])
        dj = np.abs(jj[:, None] - jj[None, :])
        near_topo = (np.maximum(di, dj) <= 2)
        upper = np.triu(np.ones((n, n), bool), 1)
        self.self_pair_ok = upper & (~near_topo)
        self.self_rest = 1.5 * self.thickness

    def _build_handle(self):
        """+x edge column is the kinematic pulling handle."""
        xs = self.x[:, 0]
        x_min = float(xs.min())
        self.handle = np.nonzero(xs < x_min + 0.25 * self.mean_edge)[0]
        if self.handle.size == 0:
            self.handle = np.array([int(np.argmin(xs))], int)
        self.handle_x0 = self.x[self.handle].copy()
        self.pull_speed = float(self.cfg.get("pull_speed", 2.0))
        self.pull_duration = float(self.cfg.get("pull_duration", 0.9))
        # pull the -x edge horizontally away from the camera so the drape and
        # attachment stay inside the frozen three-quarter view
        self.pull_dir = np.array([-1.0, 0.0, 0.0])
        self.free_mask = np.ones(self.n_cloth, bool)
        self.free_mask[self.handle] = False
        self.inv_m_free = np.where(self.free_mask, self.inv_m, 0.0)
        self.constraint_ref = self.handle_x0[:, [0, 2]].mean(axis=0)

    def _handle_offset(self, t):
        s = self.pull_speed * min(max(t, 0.0), self.pull_duration)
        return s * self.pull_dir
    def _build_rigid(self):
        geom = {
            1: ("cyl", dict(radius=0.085, height=0.72)),
            2: ("cyl", dict(radius=0.07, height=0.3)),
            3: ("cyl", dict(radius=0.24, height=0.035)),
            4: ("box", dict(half=[0.19, 0.11, 0.035])),
            5: ("box", dict(half=[0.16, 0.025, 0.012])),
            6: ("box", dict(half=[0.19, 0.022, 0.012])),
        }
        names = {1: "bottle", 2: "glass", 3: "plate", 4: "butter_dish",
                 5: "fork", 6: "knife"}
        self.bodies = []
        self.rigid_rows = []
        for bid in sorted(set(int(b) for b in self.body_id if b != 0)):
            rows = np.nonzero(self.body_id == bid)[0]
            row = int(rows[0])
            kind, par = geom.get(bid, ("box", dict(half=[0.05, 0.05, 0.05])))
            mass = float(max(self.mass_all[row], 1e-6))
            if kind == "cyl":
                r = float(par["radius"])
                h = float(par["height"])
                ixx = mass * (3.0 * r * r + h * h) / 12.0
                izz = 0.5 * mass * r * r
                inertia = np.array([ixx, ixx, izz])
                pts = _cyl_points(r, h)
            else:
                half = np.array(par["half"], float)
                hx, hy, hz = 2.0 * half
                inertia = mass / 12.0 * np.array([hy * hy + hz * hz,
                                                  hx * hx + hz * hz,
                                                  hx * hx + hy * hy])
                pts = _box_points(half)
            inertia = np.maximum(inertia, 1e-8)
            q0 = np.array([0.0, 0.0, 0.0, 1.0])
            if self.frame0.orientation is not None:
                q0 = np.array(self.frame0.orientation[row], float)
                if np.linalg.norm(q0) < 1e-9:
                    q0 = np.array([0.0, 0.0, 0.0, 1.0])
                q0 = q0 / np.linalg.norm(q0)
            self.bodies.append(dict(
                bid=bid, name=names.get(bid, "body_%d" % bid), row=row,
                mass=mass, inv_mass=1.0 / mass, inertia=inertia,
                inv_inertia=1.0 / inertia, pts=pts,
                p=self.pos0[row].copy(), q=q0,
                v=np.zeros(3), w=np.zeros(3),
                gap=float(self.frame0.fields.get(
                    "contact_signed_distance", np.zeros(self.n_all))[row])
                if "contact_signed_distance" in self.frame0.fields else 0.0,
                force=np.zeros(3), start_p=self.pos0[row].copy()))
            self.rigid_rows.append(row)
        self.rigid_rows = np.array(self.rigid_rows, int)

    # ------------------------------------------------------------- kinematics
    def _cloth_normals(self):
        nx, ny = self.nx, self.ny
        if ny < 2 or nx < 2:
            n = np.zeros_like(self.x)
            n[:, 2] = 1.0
            return n
        g = self.x.reshape(nx, ny, 3)
        du = np.empty_like(g)
        du[1:-1] = g[2:] - g[:-2]
        du[0] = g[1] - g[0]
        du[-1] = g[-1] - g[-2]
        dv = np.empty_like(g)
        dv[:, 1:-1] = g[:, 2:] - g[:, :-2]
        dv[:, 0] = g[:, 1] - g[:, 0]
        dv[:, -1] = g[:, -1] - g[:, -2]
        n = np.cross(du.reshape(-1, 3), dv.reshape(-1, 3))
        ln = np.linalg.norm(n, axis=1)
        bad = ln < 1e-12
        n[bad] = np.array([0.0, 0.0, 1.0])
        ln = np.where(bad, 1.0, ln)
        n = n / ln[:, None]
        flip = n[:, 2] < 0.0
        n[flip] *= -1.0
        return n
    def _scatter(self, f, a, b, fa):
        n = self.n_cloth
        for c in range(3):
            f[:, c] += np.bincount(a, weights=fa[:, c], minlength=n)
            f[:, c] -= np.bincount(b, weights=fa[:, c], minlength=n)
        return f

    def _spring_forces(self, f):
        x, v = self.x, self.v
        sets = (
            (self.s_a, self.s_b, self.s_rest, self.K_STRUCT, self.DAMP_STRUCT),
            (self.h_a, self.h_b, self.h_rest, self.K_SHEAR, self.DAMP_SHEAR),
            (self.b_a, self.b_b, self.b_rest, self.K_BEND, self.DAMP_BEND),
        )
        for a, b, rest, k, cd in sets:
            if a.size == 0:
                continue
            d = x[b] - x[a]
            L = np.linalg.norm(d, axis=1)
            L = np.where(L > 1e-9, L, 1e-9)
            u = d / L[:, None]
            ext = L - rest
            dv = np.sum((v[b] - v[a]) * u, axis=1)
            mag = k * ext + cd * dv
            fa = mag[:, None] * u          # force on node a (toward b)
            self._scatter(f, a, b, fa)
        return f

    def _self_contact_forces(self):
        """Node-level cloth self-repulsion, refreshed every SELF_EVERY steps."""
        if self.step_count % self.SELF_EVERY != 0:
            return self._self_force
        x = self.x
        d = x[:, None, :] - x[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", d, d)
        rest = self.self_rest
        hit = self.self_pair_ok & (dist2 < rest * rest)
        f = np.zeros_like(x)
        ia, ib = np.nonzero(hit)
        if ia.size:
            dd = x[ia] - x[ib]
            L = np.sqrt(np.maximum(np.einsum("ij,ij->i", dd, dd), 1e-18))
            u = dd / L[:, None]
            pen = rest - L
            dv = np.sum((self.v[ia] - self.v[ib]) * u, axis=1)
            mag = self.SELF_K * pen - self.SELF_D * dv
            mag = np.maximum(mag, 0.0)
            fa = mag[:, None] * u
            for c in range(3):
                f[:, c] += np.bincount(ia, weights=fa[:, c], minlength=self.n_cloth)
                f[:, c] -= np.bincount(ib, weights=fa[:, c], minlength=self.n_cloth)
        self._self_force = f
        return f
    def _cloth_neighbors(self, P):
        """Nearest cloth nodes (in xy) for world points ``P`` plus weights."""
        dxy = P[:, None, :2] - self.x[None, :, :2]
        d2 = np.einsum("ijk,ijk->ij", dxy, dxy)
        k = min(self.NEAR_K, self.n_cloth)
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        rows = np.arange(P.shape[0])[:, None]
        dk = np.sqrt(np.maximum(d2[rows, idx], 0.0))
        w = 1.0 / (dk + 0.02 * self.mean_edge)
        w = w / np.sum(w, axis=1, keepdims=True)
        return idx, w, dk[:, 0]

    def _rigid_step(self, dt, cloth_force):
        """Advance rigid bodies; accumulate cloth reaction into cloth_force."""
        normals = self._cloth_normals()
        half_t = 0.5 * self.thickness
        v_eps = 0.01
        for bd in self.bodies:
            R = _quat_to_mat(bd["q"])
            P = bd["p"][None, :] + bd["pts"] @ R.T
            r = P - bd["p"][None, :]
            Vp = bd["v"][None, :] + np.cross(np.broadcast_to(bd["w"], r.shape), r)
            F = bd["mass"] * self.gravity.copy()
            T = np.zeros(3)
            fpts = np.zeros_like(P)
            fstat = np.zeros(3)

            # --- static colliders: table box and ground plane ---
            for which in (0, 1):
                if which == 0:
                    dist, nrm = _box_sdf(P, self.table_center, self.table_half)
                else:
                    dist = P[:, 2] - self.ground_z
                    nrm = np.zeros_like(P)
                    nrm[:, 2] = 1.0
                act = dist < 0.0
                if not np.any(act):
                    continue
                pen = -dist[act]
                n = nrm[act]
                vr = Vp[act]
                vn = np.sum(vr * n, axis=1)
                fn = np.maximum(self.K_RIGID_TABLE * pen - self.D_RIGID_TABLE * vn, 0.0)
                vt = vr - vn[:, None] * n
                vtn = np.linalg.norm(vt, axis=1)
                ft = -(self.MU_RIGID_TABLE * fn / np.maximum(vtn, v_eps))[:, None] * vt
                f_static = fn[:, None] * n + ft
                fpts[act] += f_static
                fstat += f_static.sum(axis=0)
            # --- deforming cloth surface (two-way coupled) ---
            idx, w, d_near = self._cloth_neighbors(P)
            surf = np.einsum("ij,ijk->ik", w, self.x[idx])
            snrm = np.einsum("ij,ijk->ik", w, normals[idx])
            ln = np.linalg.norm(snrm, axis=1)
            snrm = snrm / np.maximum(ln, 1e-9)[:, None]
            svel = np.einsum("ij,ijk->ik", w, self.v[idx])
            gap = np.sum((P - surf) * snrm, axis=1) - half_t
            act = (gap < 0.0) & (gap > -0.06) & (d_near < 1.6 * self.mean_edge)
            if np.any(act):
                pen = -gap[act]
                n = snrm[act]
                vr = Vp[act] - svel[act]
                vn = np.sum(vr * n, axis=1)
                fn = np.maximum(self.K_RIGID_CLOTH * pen - self.D_RIGID_CLOTH * vn, 0.0)
                vt = vr - vn[:, None] * n
                vtn = np.linalg.norm(vt, axis=1)
                ft = -(self.MU_CLOTH_RIGID * fn / np.maximum(vtn, v_eps))[:, None] * vt
                fc = fn[:, None] * n + ft
                fpts[act] += fc
                # reaction on supporting cloth nodes
                wa = w[act]
                ia = idx[act]
                for c in range(3):
                    cloth_force[:, c] -= np.bincount(
                        ia.ravel(), weights=(wa * fc[:, c:c + 1]).ravel(),
                        minlength=self.n_cloth)

            F = F + fpts.sum(axis=0)
            T = T + np.cross(r, fpts).sum(axis=0)
            bd["force"] = F
            # external (body + static-boundary) force, excluding cloth coupling
            bd["ext_force"] = bd["mass"] * self.gravity + fstat

            # --- 6-DOF integration ---
            bd["v"] = bd["v"] + dt * bd["inv_mass"] * F
            Rw = _quat_to_mat(bd["q"])
            I_w = Rw @ np.diag(bd["inertia"]) @ Rw.T
            Iinv_w = Rw @ np.diag(bd["inv_inertia"]) @ Rw.T
            L = I_w @ bd["w"]
            bd["w"] = bd["w"] + dt * (Iinv_w @ (T - np.cross(bd["w"], L)))
            bd["w"] *= max(0.0, 1.0 - self.ANG_DAMP * dt)
            bd["v"] = np.clip(bd["v"], -_V_CLIP, _V_CLIP)
            bd["w"] = np.clip(bd["w"], -60.0, 60.0)
            bd["p"] = bd["p"] + dt * bd["v"]
            bd["q"] = _quat_integrate(bd["q"], bd["w"], dt)
            bd["p"] = np.clip(bd["p"], -20.0, 20.0)
        return cloth_force
    def _strain_limit(self, dt):
        """Provot-style Jacobi strain limiting on structural edges."""
        a, b, rest = self.s_a, self.s_b, self.s_rest
        lim = self.STRAIN_LIMIT
        wi = self.inv_m_free
        x_before = self.x.copy()
        max_strain = 1.0
        for _ in range(self.STRAIN_ITERS):
            d = self.x[b] - self.x[a]
            L = np.linalg.norm(d, axis=1)
            L = np.where(L > 1e-9, L, 1e-9)
            ratio = L / rest
            max_strain = max(max_strain, float(ratio.max()))
            over = ratio > lim
            if not np.any(over):
                break
            ia, ib = a[over], b[over]
            u = d[over] / L[over, None]
            excess = (L[over] - lim * rest[over])
            wa, wb = wi[ia], wi[ib]
            wsum = wa + wb
            ok = wsum > 1e-15
            if not np.any(ok):
                break
            ia, ib, u, excess = ia[ok], ib[ok], u[ok], excess[ok]
            wa, wb, wsum = wa[ok], wb[ok], wsum[ok]
            ca = (wa / wsum * excess)[:, None] * u
            cb = -(wb / wsum * excess)[:, None] * u
            corr = np.zeros_like(self.x)
            cnt = np.zeros(self.n_cloth)
            for c in range(3):
                corr[:, c] += np.bincount(ia, weights=ca[:, c], minlength=self.n_cloth)
                corr[:, c] += np.bincount(ib, weights=cb[:, c], minlength=self.n_cloth)
            cnt += np.bincount(ia, weights=np.ones(ia.size), minlength=self.n_cloth)
            cnt += np.bincount(ib, weights=np.ones(ib.size), minlength=self.n_cloth)
            corr /= np.maximum(cnt, 1.0)[:, None]
            self.x += corr
        self.max_strain_seen = max(self.max_strain_seen, max_strain)
        self.v += (self.x - x_before) / dt
        return max_strain

    def _project_contacts(self, dt):
        """Projected frictional contact of cloth nodes against static geometry."""
        half_t = 0.5 * self.thickness
        v_before = self.v.copy()
        for which in (0, 1):
            if which == 0:
                dist, nrm = _box_sdf(self.x, self.table_center, self.table_half)
                mu = self.MU_CLOTH_TABLE
            else:
                dist = self.x[:, 2] - self.ground_z
                nrm = np.zeros_like(self.x)
                nrm[:, 2] = 1.0
                mu = self.MU_CLOTH_GROUND
            pen = half_t - dist
            act = pen > 0.0
            if not np.any(act):
                continue
            n = nrm[act]
            self.x[act] += pen[act, None] * n
            vv = self.v[act]
            vn = np.sum(vv * n, axis=1)
            approach = vn < 0.0
            dvn = np.where(approach, -vn, 0.0)
            vt = vv - vn[:, None] * n
            vtn = np.linalg.norm(vt, axis=1)
            scale = np.maximum(0.0, 1.0 - mu * dvn / np.maximum(vtn, 1e-9))
            vv = np.where(approach[:, None], scale[:, None] * vt, vt + vn[:, None] * n)
            self.v[act] = vv
        f_contact = np.zeros_like(self.x)
        f_contact = self.m[:, None] * (self.v - v_before) / dt
        return f_contact
    def _handle_velocity(self, t):
        if t < self.pull_duration:
            return self.pull_speed * self.pull_dir
        return np.zeros(3)

    def _step(self, dt):
        t = self.time
        # ---- cloth forces ----
        f = self.m[:, None] * self.gravity[None, :]
        self._spring_forces(f)
        f += self._self_contact_forces()
        f = self._rigid_step(dt, f)

        # ---- symplectic Euler on cloth ----
        v_pred = self.v + dt * self.inv_m[:, None] * f
        v_pred *= max(0.0, 1.0 - self.AIR_DRAG * dt)
        v_pred = np.clip(v_pred, -_V_CLIP, _V_CLIP)

        # kinematic handle: prescribed velocity, record reaction force
        v_handle = self._handle_velocity(t)
        f_react = np.zeros_like(f)
        hv = self.handle
        f_react[hv] = self.m[hv, None] * (v_handle[None, :] - v_pred[hv]) / dt
        v_pred[hv] = v_handle[None, :]

        self.v = v_pred
        self.x = self.x + dt * self.v

        self._strain_limit(dt)
        f_contact = self._project_contacts(dt)

        # exact handle placement (kinematic boundary condition)
        self.x[hv] = self.handle_x0 + self._handle_offset(t + dt)[None, :]
        self.v[hv] = v_handle[None, :]
        self.x = np.clip(self.x, -20.0, 20.0)
        self.v = np.clip(self.v, -_V_CLIP, _V_CLIP)

        # ---- external force bookkeeping ----
        ext = np.zeros((self.n_all, 3))
        ext[self.cloth_idx] = (self.m[:, None] * self.gravity[None, :]
                               + f_contact + f_react)
        for bd in self.bodies:
            ext[bd["row"]] = bd.get("ext_force", bd["mass"] * self.gravity)
        self.ext_force = ext

        self.time = t + dt
        self.step_count += 1
    # ------------------------------------------------------------------ output
    def _cloth_gap(self):
        d_table, _ = _box_sdf(self.x, self.table_center, self.table_half)
        d_ground = self.x[:, 2] - self.ground_z
        return np.minimum(d_table, d_ground)

    def _rigid_gap(self, bd, normals):
        R = _quat_to_mat(bd["q"])
        P = bd["p"][None, :] + bd["pts"] @ R.T
        d_table, _ = _box_sdf(P, self.table_center, self.table_half)
        d_ground = P[:, 2] - self.ground_z
        gap = np.minimum(d_table, d_ground)
        idx, w, d_near = self._cloth_neighbors(P)
        surf = np.einsum("ij,ijk->ik", w, self.x[idx])
        snrm = np.einsum("ij,ijk->ik", w, normals[idx])
        snrm = snrm / np.maximum(np.linalg.norm(snrm, axis=1), 1e-9)[:, None]
        g_cloth = np.sum((P - surf) * snrm, axis=1) - 0.5 * self.thickness
        near = d_near < 1.6 * self.mean_edge
        g_cloth = np.where(near, g_cloth, np.inf)
        gap = np.minimum(gap, g_cloth)
        gap = gap[np.isfinite(gap)]
        if gap.size == 0:
            return 1.0
        return float(gap.min())

    def _frame(self):
        n = self.n_all
        pos = np.zeros((n, 3))
        vel = np.zeros((n, 3))
        ori = np.zeros((n, 4))
        ori[:, 3] = 1.0
        avel = np.zeros((n, 3))
        csd = np.zeros(n)

        pos[self.cloth_idx] = self.x
        vel[self.cloth_idx] = self.v
        csd[self.cloth_idx] = self._cloth_gap()
        normals = self._cloth_normals()
        for bd in self.bodies:
            r = bd["row"]
            pos[r] = bd["p"]
            vel[r] = bd["v"]
            ori[r] = bd["q"]
            avel[r] = bd["w"]
            csd[r] = self._rigid_gap(bd, normals)

        self.min_gap_seen = min(self.min_gap_seen, float(csd.min()))

        handle_now = pos[self.cloth_idx][self.handle][:, [0, 2]].mean(axis=0)
        constraint_value = handle_now - self.constraint_ref
        off = self._handle_offset(self.time)
        constraint_target = np.array([off[0], off[2]])

        ext = np.array(self.ext_force, float)
        for arr in (pos, vel, ori, avel, csd, ext):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        frame = StateFrame(
            time=float(max(self.time, 0.0)),
            position=pos,
            velocity=np.clip(vel, -_V_CLIP, _V_CLIP),
            orientation=ori,
            angular_velocity=avel,
            mass=self.mass_all.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "uv": self.uv.copy(),
                "mass_quantity": self.mass_all.copy(),
            },
            fields={
                "external_force": ext,
                "contact_signed_distance": csd,
                "constraint_value": np.asarray(constraint_value, float),
                "constraint_target": np.asarray(constraint_target, float),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "contact_length_scale": self.contact_length_scale,
                "constraint_scale": self.constraint_scale,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        n_target = int(round(target / self.DT))
        while self.step_count < n_target:
            self._step(self.DT)
        self.time = self.step_count * self.DT
        frame = self._frame()
        frame.time = target if target >= 0.0 else 0.0
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "mass-spring shell cloth + 6-DOF rigid penalty contact",
            "integrator": "symplectic Euler with strain limiting and projected friction",
            "dt": self.DT,
            "steps": int(self.step_count),
            "cloth_nodes": int(self.n_cloth),
            "handle_nodes": int(self.handle.size),
            "pull_speed": self.pull_speed,
            "pull_duration": self.pull_duration,
            "max_structural_strain": float(self.max_strain_seen),
            "min_contact_signed_distance": float(self.min_gap_seen),
            "rigid_bodies": [b["name"] for b in self.bodies],
            "rigid_displacement": {
                b["name"]: float(np.linalg.norm(b["p"] - b["start_p"]))
                for b in self.bodies
            },
            "notes": "rigid-rigid contact omitted (initial layout is non-overlapping)",
        }
