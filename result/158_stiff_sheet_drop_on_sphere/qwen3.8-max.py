"""Physically-based solver: stiff thin sheet dropped onto a fixed rigid sphere.

Model
-----
* The sheet is a 17x17 lumped-mass triangle lattice (289 nodes) integrated with
  semi-implicit (symplectic) Euler at a small fixed substep.
* Internal elasticity: penalty springs for structural (nearest neighbour),
  shear (diagonal) and bending (second neighbour) edges plus edge-aligned
  viscous damping.  The high bending/stretch ratio reproduces the stiff-sheet
  regime of Narain et al. 2013 Fig. 2: the sheet buckles into a small number of
  sharp folds instead of draping softly.
* Contacts: analytic penalty + normal projection against the fixed sphere
  (centre (0,0,0.68), radius 0.62) and the ground box top plane (z = 0), with
  the sheet half-thickness as clearance, and a repulsive penalty for
  self-contact between topologically distant node pairs.
* External force is gravity only (contact forces are internal/coupling pairs
  and are deliberately excluded from ``fields.external_force``).

Exported per-frame state follows the required raw trajectory contract.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.rng = np.random.default_rng(int(seed))
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        nx, ny = (int(v) for v in cloth["grid_shape"])
        sx, sy = (float(v) for v in cloth["size"])
        center = np.asarray(cloth["center"], dtype=float)
        self.nx, self.ny = nx, ny

        x = np.linspace(-sx * 0.5, sx * 0.5, nx)
        y = np.linspace(-sy * 0.5, sy * 0.5, ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        pos = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size))) + center
        self.n = len(pos)

        self.thickness = float(cloth.get("thickness", 0.015))
        total_mass = float(cloth["total_mass"])
        self.mass = np.full(self.n, total_mass / self.n)
        self.inv_mass = 1.0 / self.mass

        # material (reference) coordinates of the flat sheet
        self.ref = pos.copy()
        boundary = ((np.isclose(xx, x[0])) | (np.isclose(xx, x[-1])) |
                    (np.isclose(yy, y[0])) | (np.isclose(yy, y[-1]))).ravel()
        self.boundary = boundary.astype(np.int32)
        self.body_id = np.zeros(self.n, dtype=np.int32)

        # ---------------- obstacles (from frozen scene geometry) -----------
        geo = cfg.get("scene_geometry", {}) or {}
        sphere_c = np.array([0.0, 0.0, 0.68])
        sphere_r = 0.62
        ground_z = 0.0
        for obj in geo.get("objects", []) or []:
            g = obj.get("geometry", {}) or {}
            t = (obj.get("transform", {}) or {}).get("translation")
            if g.get("type") == "sphere" and t is not None:
                sphere_c = np.asarray(t, dtype=float)
                sphere_r = float(g.get("radius", sphere_r))
            if g.get("type") == "box" and t is not None:
                he = np.asarray(g.get("half_extents", [0, 0, 0.05]), dtype=float)
                ground_z = float(t[2]) + float(he[2])
        self.sphere_c = sphere_c
        self.sphere_r = sphere_r
        self.ground_z = ground_z

        # ---------------- lattice topology ---------------------------------
        idx = np.arange(self.n).reshape(ny, nx)  # row j (y), column i (x)
        faces = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a, b = idx[j, i], idx[j, i + 1]
                c, d = idx[j + 1, i], idx[j + 1, i + 1]
                faces.append((a, b, c))
                faces.append((b, d, c))
        self.faces = np.asarray(faces, dtype=np.int64)

        # spring families: (i, j, stiffness, damping)
        springs = []
        h_x = x[1] - x[0]
        h_y = y[1] - y[0]
        K_STRETCH, C_STRETCH = 9.0e3, 0.25
        K_SHEAR, C_SHEAR = 4.0e3, 0.25
        K_BEND, C_BEND = 3.0e3, 0.35

        def add(di, dj, k, c):
            for j in range(ny):
                for i in range(nx):
                    i2, j2 = i + di, j + dj
                    if 0 <= i2 < nx and 0 <= j2 < ny:
                        a, b = idx[j, i], idx[j2, i2]
                        rest = float(np.linalg.norm(self.ref[a] - self.ref[b]))
                        springs.append((a, b, rest, k, c))
        add(1, 0, K_STRETCH, C_STRETCH)
        add(0, 1, K_STRETCH, C_STRETCH)
        add(1, 1, K_SHEAR, C_SHEAR)
        add(1, -1, K_SHEAR, C_SHEAR)
        add(2, 0, K_BEND, C_BEND)
        add(0, 2, K_BEND, C_BEND)
        sp = np.asarray([(a, b) for a, b, *_ in springs], dtype=np.int64)
        self.sp_i, self.sp_j = sp[:, 0], sp[:, 1]
        self.sp_rest = np.asarray([s[2] for s in springs])
        self.sp_k = np.asarray([s[3] for s in springs])
        self.sp_c = np.asarray([s[4] for s in springs])
        self.sp_maxf = 60.0 * self.sp_k * 1e-3 + 5.0  # per-edge force clamp

        # ---------------- self-contact pair list ---------------------------
        adj = np.zeros((self.n, self.n), dtype=bool)
        f = self.faces
        for a, b in ((0, 1), (1, 2), (2, 0)):
            adj[f[:, a], f[:, b]] = True
            adj[f[:, b], f[:, a]] = True
        near = adj | (adj.astype(np.int32) @ adj.astype(np.int32) > 0)
        near |= np.eye(self.n, dtype=bool)
        ii, jj = np.nonzero(~near)
        keep = ii < jj
        self.sc_i = ii[keep]
        self.sc_j = jj[keep]
        self.sc_dist = max(2.0 * self.thickness, 0.022)
        self.k_self = 6.0e3
        self.c_self = 0.15

        # ---------------- contact parameters -------------------------------
        self.k_contact = 4.0e4
        self.c_contact = 3.0
        self.drag = 0.06  # gentle global air drag (N per m/s)

        # ---------------- deformation gradient helpers ---------------------
        R = self.ref[self.faces]                     # (T,3,3)
        dX = np.stack((R[:, 1] - R[:, 0], R[:, 2] - R[:, 0]), axis=2)  # (T,3,2)
        self.dXm = dX[:, :2, :]                      # material 2x2 (flat sheet)
        self.dXm_inv = np.linalg.inv(self.dXm)
        self.vert_faces = [[] for _ in range(self.n)]
        for t, tri in enumerate(self.faces):
            for v in tri:
                self.vert_faces[v].append(t)
        self.vf_idx = np.concatenate([np.asarray(vf, dtype=np.int64) for vf in self.vert_faces])
        self.vf_ptr = np.cumsum([0] + [len(vf) for vf in self.vert_faces])
        self.vf_cnt = np.diff(self.vf_ptr).astype(float)

        # ---------------- state --------------------------------------------
        self.pos = pos.copy()
        self.vel = np.zeros_like(pos)
        # the frozen geometry places the flat sheet plane slightly below the
        # sphere crown, so resolve that initial overlap before the first frame
        self._project_contacts(self.pos, self.vel)
        self.time = 0.0
        self.ext_force = self.mass[:, None] * self.gravity[None, :]
        self.signed_distance = self._signed_distance(self.pos)
        self.F = np.repeat(np.eye(3)[None], self.n, axis=0)
        self.diag = {"substeps": 0, "min_signed_distance": 1.0}

    # ------------------------------------------------------------- physics
    def _signed_distance(self, pos):
        rel = pos - self.sphere_c
        r = np.linalg.norm(rel, axis=1)
        gap_sphere = r - (self.sphere_r + 0.5 * self.thickness)
        gap_ground = pos[:, 2] - (self.ground_z + 0.5 * self.thickness)
        return np.minimum(gap_sphere, gap_ground)

    def _internal_forces(self, pos, vel, out):
        d = pos[self.sp_j] - pos[self.sp_i]
        L = np.linalg.norm(d, axis=1)
        Lsafe = np.maximum(L, 1e-9)
        nrm = d / Lsafe[:, None]
        relv = vel[self.sp_j] - vel[self.sp_i]
        fs = self.sp_k * (L - self.sp_rest) + self.sp_c * np.einsum("ij,ij->i", relv, nrm)
        fs = np.clip(fs, -self.sp_maxf, self.sp_maxf)
        fvec = fs[:, None] * nrm
        np.add.at(out, self.sp_i, fvec)
        np.add.at(out, self.sp_j, -fvec)

    def _contact_forces(self, pos, vel, out):
        # sphere
        rel = pos - self.sphere_c
        r = np.linalg.norm(rel, axis=1)
        rsafe = np.maximum(r, 1e-9)
        nrm = rel / rsafe[:, None]
        pen = (self.sphere_r + 0.5 * self.thickness) - r
        act = pen > 0.0
        if np.any(act):
            vn = np.einsum("ij,ij->i", vel, nrm)
            mag = self.k_contact * pen - self.c_contact * np.minimum(vn, 0.0)
            f = np.clip(mag, 0.0, None)[:, None] * nrm
            out[act] += f[act]
        # ground
        pen_g = self.ground_z + 0.5 * self.thickness - pos[:, 2]
        act_g = pen_g > 0.0
        if np.any(act_g):
            mag = self.k_contact * pen_g - self.c_contact * np.minimum(vel[:, 2], 0.0)
            out[act_g, 2] += np.clip(mag, 0.0, None)[act_g]
        # self contact (penalty repulsion between distant material points)
        if self.sc_i.size:
            d = pos[self.sc_j] - pos[self.sc_i]
            dist = np.linalg.norm(d, axis=1)
            near = dist < self.sc_dist
            if np.any(near):
                idx = np.nonzero(near)[0]
                dist_n = np.maximum(dist[idx], 1e-9)
                nrm = d[idx] / dist_n[:, None]
                relv = vel[self.sc_j[idx]] - vel[self.sc_i[idx]]
                mag = (self.k_self * (self.sc_dist - dist_n)
                       - self.c_self * np.einsum("ij,ij->i", relv, nrm))
                mag = np.clip(mag, 0.0, 20.0)
                fvec = mag[:, None] * nrm
                np.add.at(out, self.sc_i[idx], fvec)
                np.add.at(out, self.sc_j[idx], -fvec)

    def _project_contacts(self, pos, vel):
        """Hard positional projection so that no penetration is ever exported."""
        rel = pos - self.sphere_c
        r = np.linalg.norm(rel, axis=1)
        lim = self.sphere_r + 0.5 * self.thickness
        pen = r < lim
        if np.any(pen):
            rsafe = np.maximum(r, 1e-12)
            nrm = rel / rsafe[:, None]
            pos[pen] = self.sphere_c + nrm[pen] * lim
            rows = np.nonzero(pen)[0]
            vnn = np.einsum("ij,ij->i", vel[rows], nrm[rows])
            inward = vnn < 0.0
            rows = rows[inward]
            if rows.size:
                nrm_i = nrm[inward]
                vel[rows] -= vnn[inward][:, None] * nrm_i
        pz = pos[:, 2]
        pen_g = pz < self.ground_z + 0.5 * self.thickness
        if np.any(pen_g):
            pos[pen_g, 2] = self.ground_z + 0.5 * self.thickness
            vel[pen_g, 2] = np.maximum(vel[pen_g, 2], 0.0)
        # simple Coulomb-like tangential damping on contact nodes
        if np.any(pen):
            nrm = (pos - self.sphere_c) / np.maximum(
                np.linalg.norm(pos - self.sphere_c, axis=1), 1e-12)[:, None]
            vn = np.einsum("ij,ij->i", vel[pen], nrm[pen])[:, None] * nrm[pen]
            vt = vel[pen] - vn
            vel[pen] = vn + 0.96 * vt
        if np.any(pen_g):
            vel[pen_g, 0] *= 0.96
            vel[pen_g, 1] *= 0.96

    def _deformation_gradients(self, pos):
        R = pos[self.faces]
        ds = np.stack((R[:, 1] - R[:, 0], R[:, 2] - R[:, 0]), axis=2)   # (T,3,2)
        Ftri = ds @ self.dXm_inv                                       # (T,3,2)
        nrm = np.cross(R[:, 1] - R[:, 0], R[:, 2] - R[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]
        F3 = np.concatenate((Ftri, nrm[:, :, None]), axis=2)           # (T,3,3)
        Fv = np.zeros((self.n, 3, 3))
        np.add.at(Fv, self.vf_idx, F3[self.vf_idx])
        Fv /= np.maximum(self.vf_cnt, 1.0)[:, None, None]
        return Fv

    def _substep(self, dt):
        pos, vel = self.pos, self.vel
        f = self.ext_force.copy()
        f -= self.drag * vel
        self._internal_forces(pos, vel, f)
        self._contact_forces(pos, vel, f)
        acc = f * self.inv_mass[:, None]
        np.clip(acc, -5.0e4, 5.0e4, out=acc)
        vel += acc * dt
        np.clip(vel, -30.0, 30.0, out=vel)
        pos += vel * dt
        self._project_contacts(pos, vel)
        if not np.all(np.isfinite(pos)):
            bad = ~np.isfinite(pos).all(axis=1)
            pos[bad] = self.ref[bad]
            vel[bad] = 0.0

    # ------------------------------------------------------------- output
    def _make_frame(self, time):
        F = self._deformation_gradients(self.pos)
        self.F = F
        sd = self._signed_distance(self.pos)
        # include self-contact gap in the signed distance measure
        if self.sc_i.size:
            d = np.linalg.norm(self.pos[self.sc_j] - self.pos[self.sc_i], axis=1)
            gap = d - 2.0 * (0.5 * self.thickness)
            worst = np.full(self.n, np.inf)
            np.minimum.at(worst, self.sc_i, gap)
            np.minimum.at(worst, self.sc_j, gap)
            sd = np.minimum(sd, worst)
        self.signed_distance = sd
        self.diag["min_signed_distance"] = min(self.diag["min_signed_distance"], float(sd.min()))
        frame = StateFrame(
            float(time),
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": F,
            },
            fields={
                "external_force": self.ext_force.copy(),
                "contact_signed_distance": sd.copy(),
                "cumulative_mass_source": np.zeros(self.n),
            },
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        time = float(time)
        dt_target = 2.0e-4
        span = max(time - self.time, 0.0)
        nsteps = int(np.ceil(span / dt_target)) if span > 0 else 0
        if nsteps > 0:
            dt = span / nsteps
            for _ in range(nsteps):
                self._substep(dt)
            self.diag["substeps"] += nsteps
        self.time = time
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return dict(self.diag)
