"""Physically-based rigid-body solver: nonconvex bones falling through a funnel.

Method
------
* Each bone (assets/bone.obj: capsule shaft + two end knobs, a nonconvex
  union) is represented by a rigidly attached cloud of spheres.
* Contacts
    - sphere vs funnel triangle shell (loaded from assets/funnel.obj,
      consistently outward-oriented),
    - sphere vs floor plane z = 0 (top face of the floor box),
    - sphere vs sphere between different bones,
  are resolved with a penalty (spring-damper with cubic stiffening) normal
  force plus regularized Coulomb friction applied at the contact point.
  Forces and torques on the rigid bodies produce genuine multi-point
  contact, bridging/jamming at the throat, and force-chain break-free.
* Integration: semi-implicit Euler; attitude propagated with quaternions and
  the bone's principal (diagonal, body-frame) inertia tensor.
* The frozen initial configuration overlaps; a one-time position-based
  de-penetration relaxation at initialize() removes it so the trajectory
  starts (and stays) free of persistent overlap.

Exported per sample (StateFrame):
    state.position/velocity/orientation/angular_velocity/mass/body_id
    fields.external_force          [N,3] gravity per body
    fields.interaction_force       [K,3] paired contact forces (action=-reaction)
    fields.contact_signed_distance [C]   signed gaps, negative = penetration
    scalars.contact_length_scale
"""
from __future__ import annotations

import os

import numpy as np

_ASSET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets"
)


def _load_obj(path):
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == "f":
                faces.append([int(tok.split("/")[0]) - 1 for tok in parts[1:]])
    return np.asarray(verts, dtype=float), [list(f) for f in faces]


def _bone_spheres():
    """Sphere cloud reproducing assets/bone.obj (local frame, shaft along z)."""
    zs = [0.0, 0.08, 0.16, 0.24, 0.32]
    rows = [(0.0, 0.0, z, 0.105) for z in zs]
    rows += [(0.0, 0.0, -z, 0.105) for z in zs[1:]]
    rows += [(0.0, 0.0, 0.38, 0.2312), (0.0, 0.0, -0.38, 0.2312)]
    return np.asarray(rows, dtype=float)


def _orient_faces(faces):
    """Make triangle windings consistently oriented across the shell."""
    n_f = len(faces)
    adj = {}
    for fi, f in enumerate(faces):
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            adj.setdefault((a, b), []).append(fi)
    flipped = [False] * n_f
    seen = [False] * n_f
    stack = [0]
    seen[0] = True
    while stack:
        fi = stack.pop()
        f = faces[fi]
        sgn = -1 if flipped[fi] else 1
        edges = ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))
        if sgn < 0:
            edges = tuple((b, a) for a, b in edges)
        for a, b in edges:
            for fj in adj.get((a, b), []):
                if fj != fi and not seen[fj]:
                    seen[fj] = True
                    flipped[fj] = False
                    stack.append(fj)
            for fj in adj.get((b, a), []):
                if fj != fi and not seen[fj]:
                    seen[fj] = True
                    flipped[fj] = True
                    stack.append(fj)
    out = []
    for fi, f in enumerate(faces):
        out.append([f[0], f[2], f[1]] if flipped[fi] else list(f))
    return out


class BenchmarkSolver:
    SUB_DT = 5.0e-4
    K_N = 6.0e5        # normal penalty stiffness [N/m]
    K_N3 = 4.0e7       # cubic stiffening to bound deep penetration
    GAMMA_N = 80.0     # normal damping [N/(m/s)]
    GAMMA_T = 100.0    # tangential (friction) damping [N/(m/s)]
    BAND = 0.002       # near-contact reporting band [m]

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        contact = cfg.get("contact", {})
        self.mu = float(contact.get("friction", 0.55))
        self.length_scale = float(contact.get("length_scale", 0.02))
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        frame0 = scene.initial_state()
        self.N = int(frame0.position.shape[0])
        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.quat = np.array(frame0.orientation, dtype=float)
        self.quat /= np.linalg.norm(self.quat, axis=1, keepdims=True)
        self.ang = np.zeros_like(self.pos)
        self.mass = np.array(frame0.mass, dtype=float)
        self.body_id = np.array(frame0.body_id, dtype=np.int32)
        self.ext_force = self.mass[:, None] * self.gravity[None, :]

        self._randomize_orientation(seed)

        sp = _bone_spheres()
        self.sphere_off = np.ascontiguousarray(sp[:, :3])   # [S,3]
        self.sphere_r = np.ascontiguousarray(sp[:, 3])      # [S]
        self.S = self.sphere_off.shape[0]
        self.r_sum_mat = self.sphere_r[:, None] + self.sphere_r[None, :]
        self.body_radius = float(
            np.linalg.norm(self.sphere_off, axis=1).max() + self.sphere_r.max()
        )

        # inertia about the center of mass (sphere-cloud model; the bone's
        # principal axes coincide with its body frame by symmetry)
        m_i = float(self.mass.mean()) / self.S
        off = self.sphere_off
        r2 = (off * off).sum(1)
        sp_in = 0.4 * m_i * self.sphere_r ** 2
        Ixx = float(np.sum(m_i * (r2 - off[:, 0] ** 2) + sp_in))
        Iyy = float(np.sum(m_i * (r2 - off[:, 1] ** 2) + sp_in))
        Izz = float(np.sum(m_i * (r2 - off[:, 2] ** 2) + sp_in))
        self.I_body = np.array([Ixx, Iyy, Izz])
        self.I_body_inv = 1.0 / self.I_body

        # statics: funnel triangle shell with consistent outward orientation
        fv, ff = _load_obj(os.path.join(_ASSET_DIR, "funnel.obj"))
        ff = _orient_faces(ff)
        tri = fv[np.asarray(ff, dtype=int)]                 # [T,3,3]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        tri_n = nrm / np.maximum(ln, 1e-12)
        # global sign: outward = away from shell centroid for an extreme vertex
        centroid = tri.mean(axis=(0, 1))
        vx = tri.reshape(-1, 3)
        k = int(np.argmax(vx[:, 0]))
        p = vx[k]
        faces_at = np.unique(np.asarray(ff, dtype=int)[np.nonzero(
            np.asarray(ff, dtype=int) == k)[0]])
        avg = tri_n[faces_at].mean(0)
        if float(avg @ (p - centroid)) < 0.0:
            tri_n = -tri_n
        self.tri = np.ascontiguousarray(tri)
        self.tri_n = np.ascontiguousarray(tri_n)
        self.v0 = np.ascontiguousarray(tri[:, 0])
        self.e0 = np.ascontiguousarray(tri[:, 1] - tri[:, 0])
        self.e1 = np.ascontiguousarray(tri[:, 2] - tri[:, 0])
        self.dd00 = (self.e0 * self.e0).sum(1)
        self.dd01 = (self.e0 * self.e1).sum(1)
        self.dd11 = (self.e1 * self.e1).sum(1)
        self.tden = self.dd00 * self.dd11 - self.dd01 ** 2
        self.T = self.tri.shape[0]
        self._ray_dir = np.array([0.31, 0.47, 0.83])

        self.t = 0.0
        self._rec_f: list = []
        self._rec_d: list = []
        self._recording = False

        self._relax_initial()

    def _randomize_orientation(self, seed: int) -> None:
        rng = np.random.default_rng(int(seed) + 101)
        axes = rng.normal(size=(self.N, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        angles = rng.uniform(0.4, 1.1, size=self.N)
        s = np.sin(angles / 2.0)
        self.quat = np.column_stack([axes * s[:, None], np.cos(angles / 2.0)])
        self.quat /= np.linalg.norm(self.quat, axis=1, keepdims=True)

    # -------------------------------------------------------------- helpers
    def _quat_to_R(self, q):
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = np.empty((q.shape[0], 3, 3))
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    def _sphere_centers(self):
        R = self._quat_to_R(self.quat)
        return self.pos[:, None, :] + np.einsum("nij,sj->nsi", R, self.sphere_off)

    def _gap_statics(self, cen):
        """Signed gap and contact normal of every sphere vs every funnel
        triangle.  gap = dist(center, closest point) - radius  (>0 separated,
        <0 penetrating).  The normal points from the closest surface point
        toward the sphere center, which is valid on both sides of the thin
        funnel shell."""
        flat = cen.reshape(-1, 3)
        rad = np.tile(self.sphere_r, self.N)
        diff = flat[:, None, :] - self.v0[None, :, :]           # [NS,T,3]
        d0 = np.einsum("ktj,tj->kt", diff, self.e0)
        d1 = np.einsum("ktj,tj->kt", diff, self.e1)
        u = (self.dd11[None, :] * d0 - self.dd01[None, :] * d1) / self.tden[None, :]
        v = (self.dd00[None, :] * d1 - self.dd01[None, :] * d0) / self.tden[None, :]
        uu = np.clip(u, 0.0, None)
        vv = np.clip(v, 0.0, None)
        ssum = uu + vv
        sc = np.where(ssum > 1.0, 1.0 / np.maximum(ssum, 1e-12), 1.0)
        uu = uu * sc
        vv = vv * sc
        cp = (self.v0[None, :, :]
              + uu[..., None] * self.e0[None, :, :]
              + vv[..., None] * self.e1[None, :, :])
        delta = flat[:, None, :] - cp                            # [NS,T,3]
        dist = np.sqrt((delta * delta).sum(2))                   # [NS,T]
        nrm = delta / np.maximum(dist, 1e-12)[..., None]
        gap = dist - rad[:, None]
        # Centers that crossed INTO the thin funnel shell (dist under-reports
        # penetration there): detect with a ray-parity inside test and flip
        # the contact so the sphere is pushed back out to the nearer side.
        inside = self._inside_shell(flat)                        # [NS]
        if inside.any():
            kmin = np.argmin(dist, axis=1)
            rows = np.nonzero(inside)[0]
            d_in = dist[rows, kmin[rows]]
            n_in = -nrm[rows, kmin[rows], :]
            gap[rows, :] = np.inf
            gap[rows, kmin[rows]] = -(d_in + rad[rows])
            nrm[rows, kmin[rows], :] = n_in
        return gap, nrm

    def _inside_shell(self, pts):
        """Ray-parity inside test of points vs the closed funnel shell."""
        rdir = self._ray_dir                                     # [3]
        pvec = np.cross(rdir, self.e1)                             # [T,3]
        det = (self.e0 * pvec).sum(1)                              # [T]
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = pts[:, None, :] - self.v0[None, :, :]                  # [P,T,3]
        u = np.einsum("ptj,tj->pt", s, pvec) * inv[None, :]
        qvec = np.cross(s, self.e0[None, :, :])                    # [P,T,3]
        v = np.einsum("ptj,j->pt", qvec, rdir) * inv[None, :]
        tpar = (qvec * self.e1[None, :, :]).sum(2) * inv[None, :]
        hit = ok[None, :] & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) \
            & (u + v <= 1.0) & (tpar > 1e-9)
        return (hit.sum(1) % 2) == 1

    # ------------------------------------------------- initial de-overlap
    def _relax_initial(self) -> None:
        for _ in range(600):
            cen = self._sphere_centers()
            worst = 0.0
            d = cen[:, None, :, :] - cen[None, :, None, :]      # [N,N,S,S,3]
            dist = np.sqrt((d * d).sum(3))
            pen = self.r_sum_mat[None, None, :, :] - dist
            ii, jj, si, sj = np.nonzero(pen > 1e-4)
            keep = ii < jj
            ii, jj, si, sj = ii[keep], jj[keep], si[keep], sj[keep]
            if len(ii):
                depth = np.clip(pen[ii, jj, si, sj], 0.0, None)
                worst = float(depth.max())
                dirs = d[ii, jj, si, sj] / np.maximum(dist[ii, jj, si, sj], 1e-12)
                corr = np.zeros_like(self.pos)
                np.add.at(corr, ii, 0.5 * depth[:, None] * dirs)
                np.add.at(corr, jj, -0.5 * depth[:, None] * dirs)
                self.pos += corr
            zpen = self.sphere_r[None, :] - cen[:, :, 2]
            if zpen.max() > 1e-4:
                worst = max(worst, float(zpen.max()))
                self.pos[:, 2] += float(zpen.max())
            # funnel shell: push each penetrating sphere out along its
            # closest-surface normal (rigid translation per body)
            gap_s, nrm_s = self._gap_statics(cen)
            gmin = gap_s.min(axis=1).reshape(self.N, self.S)     # [N,S]
            kmin = gap_s.argmin(axis=1).reshape(self.N, self.S)
            bad = gmin < -1e-4
            if bad.any():
                worst = max(worst, float(-gmin.min()))
                nb = nrm_s.reshape(self.N, self.S, self.T, 3)[
                    np.arange(self.N)[:, None], np.arange(self.S)[None, :], kmin]
                push = np.zeros((self.N, 3))
                for b in range(self.N):
                    rows = np.nonzero(bad[b])[0]
                    if rows.size:
                        push[b] = np.mean(
                            -gmin[b, rows][:, None] * nb[b, rows], axis=0)
                self.pos += push
            if worst <= 5e-4:
                break

    # ------------------------------------------------------------- stepping
    def advance_to(self, time: float):
        time = float(time)
        if time <= self.t:
            return self._make_frame(self.t)
        n_sub = max(1, int(round((time - self.t) / self.SUB_DT)))
        dt = (time - self.t) / n_sub
        for k in range(n_sub):
            self._recording = (k == n_sub - 1)
            self._rec_f = []
            self._rec_d = []
            self._step(dt)
        self.t = time
        return self._make_frame(time)

    def _contact_force_mag(self, gap, vn):
        if gap >= 0.0:
            return 0.0
        d = -gap
        fn = self.K_N * d + self.K_N3 * d * d * d - self.GAMMA_N * vn
        return fn if fn > 0.0 else 0.0

    def _step(self, dt: float) -> None:
        cen = self._sphere_centers()
        force = np.zeros_like(self.pos)
        torque = np.zeros_like(self.pos)
        rec = self._recording
        mu = self.mu

        # ---------- spheres vs funnel shell ----------
        gap, nrm = self._gap_statics(cen)
        idx = np.argwhere(gap < self.BAND)
        for k, t in idx:
            b = k // self.S
            s = k - b * self.S
            g = float(gap[k, t])
            nn = nrm[k, t]
            r_vec = cen[b, s] - self.pos[b]
            rv = self.vel[b] + np.cross(self.ang[b], r_vec)
            vn = float(rv @ nn)
            fn = self._contact_force_mag(g, vn)
            if fn <= 0.0:
                if rec:
                    self._rec_f.append(np.zeros(3))
                    self._rec_d.append(g)
                continue
            f = fn * nn
            vt = rv - vn * nn
            sp = float(vt @ vt)
            if sp > 1e-12:
                ft = min(self.GAMMA_T * np.sqrt(sp), mu * fn)
                f = f - (ft / np.sqrt(sp)) * vt
            force[b] += f
            torque[b] += np.cross(r_vec, f)
            if rec:
                self._rec_f.append(f.copy())
                self._rec_d.append(g)

        # ---------- spheres vs floor plane z = 0 ----------
        flat_cen = cen.reshape(-1, 3)
        rad_flat = np.tile(self.sphere_r, self.N)
        g_floor = flat_cen[:, 2] - rad_flat
        for k in np.nonzero(g_floor < self.BAND)[0]:
            g = float(g_floor[k])
            b = k // self.S
            s = k - b * self.S
            r_vec = cen[b, s] - self.pos[b]
            rv = self.vel[b] + np.cross(self.ang[b], r_vec)
            vn = float(rv[2])
            fn = self._contact_force_mag(g, vn)
            if fn <= 0.0:
                if rec:
                    self._rec_f.append(np.zeros(3))
                    self._rec_d.append(g)
                continue
            f = np.array([0.0, 0.0, fn])
            vt0, vt1 = float(rv[0]), float(rv[1])
            sp = vt0 * vt0 + vt1 * vt1
            if sp > 1e-12:
                ft = min(self.GAMMA_T * np.sqrt(sp), mu * fn) / np.sqrt(sp)
                f[0] -= ft * vt0
                f[1] -= ft * vt1
            force[b] += f
            torque[b] += np.cross(r_vec, f)
            if rec:
                self._rec_f.append(f.copy())
                self._rec_d.append(g)

        # ---------- bone vs bone (sphere-sphere) ----------
        dc = self.pos[:, None, :] - self.pos[None, :, :]
        bdist = np.sqrt((dc * dc).sum(2))
        touch = bdist < 2.0 * self.body_radius
        np.fill_diagonal(touch, False)
        bi, bj = np.nonzero(np.triu(touch))
        for a, b in zip(bi, bj):
            dvec = cen[b][None, :, :] - cen[a][:, None, :]      # [S,S,3]
            dist = np.sqrt((dvec * dvec).sum(2))
            gmat = dist - self.r_sum_mat
            si_l, sj_l = np.nonzero(gmat < self.BAND)
            for sa, sb in zip(si_l, sj_l):
                g = float(gmat[sa, sb])
                dn = float(dist[sa, sb])
                if dn < 1e-9:
                    continue
                nn = dvec[sa, sb] / dn          # from sphere(a) toward sphere(b)
                ra = cen[a, sa] - self.pos[a]
                rb = cen[b, sb] - self.pos[b]
                rva = self.vel[a] + np.cross(self.ang[a], ra)
                rvb = self.vel[b] + np.cross(self.ang[b], rb)
                rv = rvb - rva
                vn = float(rv @ nn)
                fn = self._contact_force_mag(g, vn)
                if fn <= 0.0:
                    if rec:
                        self._rec_f.append(np.zeros(3))
                        self._rec_d.append(g)
                    continue
                f = fn * nn                     # force on body b from body a
                vt = rv - vn * nn
                sp = float(vt @ vt)
                if sp > 1e-12:
                    ft = min(self.GAMMA_T * np.sqrt(sp), mu * fn)
                    f = f - (ft / np.sqrt(sp)) * vt
                force[b] += f
                torque[b] += np.cross(rb, f)
                force[a] -= f
                torque[a] -= np.cross(ra, f)
                if rec:
                    self._rec_f.append(f.copy())
                    self._rec_d.append(g)

        # ---------- integrate ----------
        self.vel += (force / self.mass[:, None] + self.gravity[None, :]) * dt
        R = self._quat_to_R(self.quat)
        Rt = np.transpose(R, (0, 2, 1))
        torque_body = np.einsum("nij,nj->ni", Rt, torque)
        omega_body = np.einsum("nij,nj->ni", Rt, self.ang)
        alpha_body = torque_body * self.I_body_inv[None, :] - 0.05 * omega_body
        self.ang += np.einsum("nij,nj->ni", R, alpha_body) * dt
        self.pos += self.vel * dt
        q, w = self.quat, self.ang
        qd = 0.5 * np.stack([
            w[:, 0] * q[:, 3] + w[:, 1] * q[:, 2] - w[:, 2] * q[:, 1],
            -w[:, 0] * q[:, 2] + w[:, 1] * q[:, 3] + w[:, 2] * q[:, 0],
            w[:, 0] * q[:, 1] - w[:, 1] * q[:, 0] + w[:, 2] * q[:, 3],
            -w[:, 0] * q[:, 0] - w[:, 1] * q[:, 1] - w[:, 2] * q[:, 2],
        ], axis=1)
        self.quat += qd * dt
        self.quat /= np.linalg.norm(self.quat, axis=1, keepdims=True)

    # --------------------------------------------------------------- frame
    def _make_frame(self, time: float):
        from worldsolver.contracts import StateFrame

        if self._rec_f:
            inter = np.asarray(self._rec_f, dtype=float).reshape(-1, 3)
            dists = np.asarray(self._rec_d, dtype=float).reshape(-1)
        else:
            inter = np.zeros((1, 3))
            cen = self._sphere_centers()
            gap, _ = self._gap_statics(cen)
            gap_body = gap.reshape(self.N, self.S, self.T).min(axis=(1, 2))
            z_gap = (cen[:, :, 2] - self.sphere_r[None, :]).min(axis=1)
            dists = np.minimum(gap_body, z_gap)

        return StateFrame(
            float(time),
            self.pos.copy(),
            self.vel.copy(),
            orientation=self.quat.copy(),
            angular_velocity=self.ang.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "external_force": self.ext_force.copy(),
                "interaction_force": inter,
                "contact_signed_distance": dists,
            },
            scalars={"contact_length_scale": self.length_scale},
        )

    def diagnostics(self) -> dict:
        return {
            "method": "penalty sphere-cloud rigid-body contact",
            "n_bodies": int(self.N),
            "sub_dt": self.SUB_DT,
        }
