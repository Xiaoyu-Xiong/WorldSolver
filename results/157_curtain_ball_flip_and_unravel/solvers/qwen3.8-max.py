"""Physically-based cloth solver: pinned curtain wrapped around a rigid ball.

Mass-spring thin sheet (stretch/shear/bend) integrated with semi-implicit Euler
at small substeps. Penalty contact + Coulomb-ish friction against the kinematic
sphere, the ground plane, and cloth self-contact (brute-force pair search).
The ball first falls to the ground, then slides backward out of the cloth
pocket; the pocket turns over / collapses and the curtain hangs freely again.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------- setup ----------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.nx, self.ny = map(int, cloth["grid_shape"])
        self.N = self.nx * self.ny
        self.thickness = float(cloth.get("thickness", 0.015))
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.duration = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])

        frame0 = scene.initial_state()
        p = np.array(frame0.position, dtype=float, copy=True)
        self.mass = np.array(frame0.mass, dtype=float, copy=True)
        # material coordinates of the flat reference grid (x, z)
        self.X0 = np.column_stack((p[:, 0], p[:, 2])).copy()
        self.dX = float(cloth["size"][0]) / (self.nx - 1)
        self.dY = float(cloth["size"][1]) / (self.ny - 1)

        # ---- initial wrap: drape cloth over the front hemisphere of the ball
        R = 0.62
        c0 = np.array([0.0, 0.0, 0.68])
        eps = 0.02
        dz = p[:, 2] - c0[2]
        rr = R + eps
        inside = (p[:, 0] ** 2 + dz ** 2) < rr ** 2
        p[inside, 1] = -np.sqrt(np.maximum(rr ** 2 - p[inside, 0] ** 2 - dz[inside] ** 2, 0.0))
        # keep wrapped vertices clear of the ground
        p[:, 2] = np.maximum(p[:, 2], 0.02)

        # ---- pins: top row (z = 2.8) attached to the rod
        yy = np.linspace(-float(cloth["size"][1]) * .5, float(cloth["size"][1]) * .5, self.ny)
        ggx, ggy = np.meshgrid(np.linspace(-1, 1, self.nx), yy, indexing="xy")
        self.pinned = np.isclose(ggy.ravel(), yy[-1])
        self.pin_pos = p[self.pinned].copy()

        self.p = p
        self.v = np.zeros_like(p)

        # ---- spring lattice
        ei, ej, L, K, C = [], [], [], [], []
        k_s, c_s = 1200.0, 0.08
        k_sh, c_sh = 300.0, 0.05
        k_b, c_b = 1.5, 0.02
        nx, ny = self.nx, self.ny
        def idx(i, j):
            return i * nx + j
        for i in range(ny):
            for j in range(nx):
                a = idx(i, j)
                if j + 1 < nx:
                    b = idx(i, j + 1)
                    ei.append(a); ej.append(b); L.append(self.dX); K.append(k_s); C.append(c_s)
                if i + 1 < ny:
                    b = idx(i + 1, j)
                    ei.append(a); ej.append(b); L.append(self.dY); K.append(k_s); C.append(c_s)
                if i + 1 < ny and j + 1 < nx:
                    b = idx(i + 1, j + 1)
                    ei.append(a); ej.append(b); L.append(np.hypot(self.dX, self.dY)); K.append(k_sh); C.append(c_sh)
                    b2 = idx(i + 1, j); a2 = idx(i, j + 1)
                    ei.append(a2); ej.append(b2); L.append(np.hypot(self.dX, self.dY)); K.append(k_sh); C.append(c_sh)
                if j + 2 < nx:
                    b = idx(i, j + 2)
                    ei.append(a); ej.append(b); L.append(2 * self.dX); K.append(k_b); C.append(c_b)
                if i + 2 < ny:
                    b = idx(i + 2, j)
                    ei.append(a); ej.append(b); L.append(2 * self.dY); K.append(k_b); C.append(c_b)
        self.ei = np.asarray(ei, dtype=np.int64)
        self.ej = np.asarray(ej, dtype=np.int64)
        self.eL = np.asarray(L, dtype=float)
        self.eK = np.asarray(K, dtype=float)
        self.eC = np.asarray(C, dtype=float)

        # ---- contact parameters
        self.k_c = 1.0e4      # sphere / ground penalty stiffness
        self.c_c = 0.6        # contact normal damping
        self.k_self = 4.0e3
        self.c_self = 0.25
        self.d_self = 0.022   # self-contact gap threshold
        self.mu_ball = 0.35
        self.mu_ground = 0.45
        self.mu_self = 0.2
        self.half_t = 0.5 * self.thickness

        self.substeps_per_sample = 112
        self.h = self.sample_dt / self.substeps_per_sample
        self.time = 0.0
        self.n_steps = 0
        self.seed = int(seed)

    # ---------- kinematic ball ----------
    def ball_state(self, t: float):
        R = 0.62
        z0 = 0.68
        g = 9.81
        x = y = 0.0
        z = z0
        vx = vy = vz = 0.0
        t0 = 0.9
        if t > t0:
            tf = np.sqrt(2.0 * (z0 - R) / g)
            dtf = min(t - t0, tf)
            z = max(z0 - 0.5 * g * dtf * dtf, R)
            vz = -g * dtf if (t - t0) < tf else 0.0
        t1, t2 = 1.5, 3.8
        dx, dy = 0.45, 0.85
        if t > t1:
            tau = min((t - t1) / (t2 - t1), 1.0)
            s = tau * tau * (3.0 - 2.0 * tau)
            ds = 6.0 * tau * (1.0 - tau) / (t2 - t1)
            x = dx * s
            y = dy * s
            vx = dx * ds
            vy = dy * ds
        return np.array([x, y, z]), np.array([vx, vy, vz])

    # ---------- forces ----------
    def _spring_forces(self, F):
        d = self.p[self.ej] - self.p[self.ei]
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        r = np.maximum(r, 1e-9)
        n = d / r[:, None]
        dv = self.v[self.ej] - self.v[self.ei]
        vr = np.einsum("ij,ij->i", dv, n)
        fm = self.eK * (r - self.eL) + self.eC * vr
        fvec = n * fm[:, None]
        np.add.at(F, self.ei, fvec)
        np.add.at(F, self.ej, -fvec)

    def _apply_contact(self, F, pen, i_arr, normal, vrel, mu, fn_mag):
        """Penalty normal force (already in F via caller) plus clipped friction."""
        vt = vrel - np.einsum("ij,ij->i", vrel, normal)[:, None]
        sp = np.sqrt(np.einsum("ij,ij->i", vt, vt))
        cf = np.minimum(mu * fn_mag, 0.4 * self.mass[i_arr] / self.h * np.maximum(sp, 1e-9)) / np.maximum(sp, 1e-9)
        np.add.at(F, i_arr, -cf[:, None] * vt)

    def _contacts(self, F, dist_out):
        h = self.h
        p, v = self.p, self.v
        # signed gaps (negative = penetration), start with ground & sphere
        d_ground = p[:, 2] - self.half_t
        bc, bv = self.ball_state(self.time)
        rel = p - bc
        rl = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        rl = np.maximum(rl, 1e-9)
        n_s = rel / rl[:, None]
        d_ball = rl - (0.62 + self.half_t)
        dmin = np.minimum(d_ground, d_ball)

        # ground
        act = (d_ground < 0.005) & (~self.pinned)
        if np.any(act):
            ia = np.nonzero(act)[0]
            pen = np.maximum(-d_ground[ia], 0.0)
            nrm = np.zeros((len(ia), 3)); nrm[:, 2] = 1.0
            vn = v[ia, 2]
            fn = self.k_c * pen - self.c_c * np.minimum(vn, 0.0)
            fn = np.maximum(fn, 0.0)
            np.add.at(F, ia, fn[:, None] * nrm)
            vrel = v[ia] - 0.0
            self._apply_contact(F, pen, ia, nrm, vrel, self.mu_ground, fn)
        # sphere
        act = (d_ball < 0.005) & (~self.pinned)
        if np.any(act):
            ia = np.nonzero(act)[0]
            pen = np.maximum(-d_ball[ia], 0.0)
            nrm = n_s[ia]
            vrel = v[ia] - bv
            vn = np.einsum("ij,ij->i", vrel, nrm)
            fn = np.maximum(self.k_c * pen - self.c_c * np.minimum(vn, 0.0), 0.0)
            np.add.at(F, ia, fn[:, None] * nrm)
            self._apply_contact(F, pen, ia, nrm, vrel, self.mu_ball, fn)
        # self-contact (brute force pairs)
        diff = p[:, None, :] - p[None, :, :]
        dd = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        np.fill_diagonal(dd, np.inf)
        ii, jj = np.nonzero(dd < self.d_self)
        keep = ii < jj
        ii, jj = ii[keep], jj[keep]
        if len(ii):
            dij = dd[ii, jj]
            nvec = diff[ii, jj] / np.maximum(dij, 1e-9)[:, None]
            pen = self.d_self - dij
            dvv = v[ii] - v[jj]
            vn = np.einsum("ij,ij->i", dvv, nvec)
            fn = np.maximum(self.k_self * pen - self.c_self * vn, 0.0)
            fvec = fn[:, None] * nvec
            ok_i = ~self.pinned[ii]
            ok_j = ~self.pinned[jj]
            np.add.at(F, ii[ok_i], fvec[ok_i])
            np.add.at(F, jj[ok_j], -fvec[ok_j])
            # light self friction
            vt = dvv - vn[:, None] * nvec
            sp = np.sqrt(np.einsum("ij,ij->i", vt, vt))
            m_pair = np.minimum(self.mass[ii], self.mass[jj])
            cf = np.minimum(self.mu_self * fn, 0.3 * m_pair / h * np.maximum(sp, 1e-9)) / np.maximum(sp, 1e-9)
            ff = cf[:, None] * vt
            np.add.at(F, ii[ok_i], -ff[ok_i])
            np.add.at(F, jj[ok_j], ff[ok_j])
            # self distances into per-vertex minimum
            ds = np.full(self.N, np.inf)
            np.minimum.at(ds, ii, dij - self.d_self)
            np.minimum.at(ds, jj, dij - self.d_self)
            dmin = np.minimum(dmin, ds)
        dist_out[:] = np.minimum(dmin, 1.0)

    # ---------- time integration ----------
    def _step(self):
        h = self.h
        F = np.zeros_like(self.p)
        F += self.mass[:, None] * self.gravity
        self._spring_forces(F)
        self._contact_buf = np.full(self.N, 1.0)
        self._contacts(F, self._contact_buf)
        F[self.pinned] = 0.0
        a = F / self.mass[:, None]
        self.v += a * h
        self.v *= (1.0 - 0.3 * h)  # light air drag
        # clamp velocities for robustness
        sp = np.sqrt(np.einsum("ij,ij->i", self.v, self.v))
        bad = sp > 60.0
        if np.any(bad):
            self.v[bad] *= (60.0 / sp[bad])[:, None]
        self.p += self.v * h
        self.p[self.pinned] = self.pin_pos
        self.v[self.pinned] = 0.0
        # hard positional fixes for deep penetration (projection, no tunneling)
        bc, _ = self.ball_state(self.time)
        rel = self.p - bc
        rl = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        rmin = 0.62 + 0.4 * self.thickness
        fix = (rl < rmin) & (~self.pinned)
        if np.any(fix):
            self.p[fix] = bc + rel[fix] * (rmin / np.maximum(rl[fix], 1e-9))[:, None]
            nrm = rel[fix] / np.maximum(rl[fix], 1e-9)[:, None]
            vn = np.einsum("ij,ij->i", self.v[fix], nrm)
            self.v[fix] -= np.minimum(vn, 0.0)[:, None] * nrm
        low = (self.p[:, 2] < 0.4 * self.thickness) & (~self.pinned)
        if np.any(low):
            self.p[low, 2] = 0.4 * self.thickness
            self.v[low, 2] = np.maximum(self.v[low, 2], 0.0)
        self.time += h
        self.n_steps += 1

    # ---------- output ----------
    def _deformation_gradients(self):
        nx, ny = self.nx, self.ny
        q = self.p.reshape(ny, nx, 3)
        F = np.zeros((self.N, 2, 2))
        a = np.zeros((ny, nx, 3))
        b = np.zeros((ny, nx, 3))
        a[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (2 * self.dX)
        a[:, 0] = (q[:, 1] - q[:, 0]) / self.dX
        a[:, -1] = (q[:, -1] - q[:, -2]) / self.dX
        b[1:-1] = (q[2:] - q[:-2]) / (2 * self.dY)
        b[0] = (q[1] - q[0]) / self.dY
        b[-1] = (q[-1] - q[-2]) / self.dY
        ar = a.reshape(self.N, 3)
        br = b.reshape(self.N, 3)
        F[:, 0, 0] = ar[:, 0]; F[:, 0, 1] = br[:, 0]
        F[:, 1, 0] = ar[:, 2]; F[:, 1, 1] = br[:, 2]
        return F

    def _make_frame(self, t):
        dist = np.full(self.N, 1.0)
        if hasattr(self, "_contact_buf"):
            dist = self._contact_buf
        bc, _ = self.ball_state(t)
        rel = self.p - bc
        rl = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        d_ball = rl - (0.62 + self.half_t)
        d_ground = self.p[:, 2] - self.half_t
        dist = np.minimum(np.minimum(dist, d_ball), np.minimum(d_ground, 1.0))
        ext = self.mass[:, None] * self.gravity
        return StateFrame(
            float(t),
            self.p.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradients(),
                "boundary_vertex": self.pinned.astype(np.int32),
            },
            fields={
                "cumulative_mass_source": np.zeros(self.N),
                "external_force": ext,
                "contact_signed_distance": dist,
            },
            scalars={"contact_length_scale": self.thickness},
        )

    def advance_to(self, time: float):
        target = float(time)
        if target <= self.time + 1e-12:
            return self._make_frame(self.time)
        n = int(np.ceil((target - self.time) / self.h - 1e-9))
        for _ in range(n):
            self._step()
        self.time = max(self.time, target)
        fr = self._make_frame(self.time)
        fr.time = target
        return fr

    def diagnostics(self) -> dict:
        return {
            "substeps": self.n_steps,
            "time": self.time,
            "n_springs": int(len(self.eL)),
            "method": "mass-spring + penalty contact, semi-implicit Euler",
        }
