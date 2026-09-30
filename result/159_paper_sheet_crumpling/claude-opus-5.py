"""Paper sheet crumpling solver.

Physical model
-------------
The sheet is a discrete thin shell on the frozen 17x17 grid, integrated with a
substepped XPBD scheme (symplectic prediction + compliant constraint projection):

* structural edge constraints  -> nearly inextensible in-plane stretching (paper)
* diagonal (shear) constraints -> in-plane shear resistance
* cross-node bend constraints  -> bending stiffness with rate-independent plastic
  yielding of the rest curvature, which makes creases persist after folding
  (sharp ridges and conical vertices).

Boundary conditions come from the frozen geometry: a static ground box plus two
kinematic wall boxes driven inwards with a prescribed smoothstep compression.
Contact uses unilateral non-penetration projection against ground/walls with
Coulomb-like friction, and node-node plus node-triangle repulsion at the finite
collision thickness for self-contact so folded layers cannot pass through.

Determinism: fixed timestep, fixed iteration counts, seeded RNG used only for the
tiny initial out-of-plane imperfection that seeds buckling.
"""
from __future__ import annotations

import numpy as np


def _smoothstep(t: float, t0: float, t1: float) -> float:
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    u = (t - t0) / (t1 - t0)
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(getattr(scene, "config", {}) or {})
        self.cfg = cfg
        cloth = dict(cfg.get("cloth", {}))
        self.nx, self.ny = (int(v) for v in cloth.get("grid_shape", (17, 17)))
        sx, sy = (float(v) for v in cloth.get("size", (3.0, 2.5)))
        self.thickness = float(cloth.get("thickness", 0.015))
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        self.duration = float(cfg.get("duration", 7.0))
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.seed = int(seed if seed is not None else cfg.get("seed", 0))

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)
        self.n = len(self.x)
        self.v = np.zeros_like(self.x)
        if frame0.mass is not None:
            self.m = np.array(frame0.mass, dtype=float)
        else:
            self.m = np.full(self.n, float(cloth.get("total_mass", 0.55)) / self.n)
        self.w = 1.0 / np.maximum(self.m, 1e-12)
        self.body_id = np.zeros(self.n, dtype=np.int32)

        # material (rest) coordinates on the frozen grid
        self.dx0 = sx / max(self.nx - 1, 1)
        self.dy0 = sy / max(self.ny - 1, 1)
        gi, gj = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="xy")
        self.gi = gi.ravel().astype(np.int64)
        self.gj = gj.ravel().astype(np.int64)

        self._build_topology()
        self._build_masks()

        rng = np.random.default_rng(self.seed)
        pert = rng.standard_normal((self.n, 3))
        self.x[:, 0] += 0.0006 * pert[:, 0]
        self.x[:, 1] += 0.0006 * pert[:, 1]
        self.x[:, 2] += 0.0035 * pert[:, 2]

        self.L_struct = self._rest_len(self.e_struct)
        self.L_shear = self._rest_len(self.e_shear)
        self.L_bend0 = self._rest_len(self.e_bend)
        self.L_bend = self.L_bend0.copy()

        # numerics
        self.substeps = 16
        self.h = self.sample_dt / self.substeps
        self.iters = 8
        self.alpha_struct = 1.0e-9
        self.alpha_shear = 1.0e-7
        self.alpha_bend = 2.0e-4
        self.omega = 0.55
        self.drag = 0.010
        self.vel_damp = 0.999
        self.bend_yield = 0.05
        self.bend_plastic_rate = 0.004

        # contact geometry (frozen scene_geometry.yaml)
        self.r_c = 0.5 * max(self.thickness, 1e-4) + 0.017
        self.h_self = 2.0 * self.r_c
        self.ground_top = 0.0
        self.wall_half = np.array([0.12, 1.5, 1.3])
        self.wall_x0 = 1.55
        self.wall_x1 = 0.60
        self.wall_z = 1.3
        self.t_close0 = 0.9
        self.t_close1 = 5.4
        self.mu_ground = 0.45
        self.mu_wall = 0.35

        self.t = 0.0
        self._steps = 0
        self._max_speed = 0.0
        self.f_ext = self._external_force()

    # --------------------------------------------------------------- topology
    def _idx(self, i, j):
        return j * self.nx + i

    def _build_topology(self) -> None:
        nx, ny = self.nx, self.ny
        struct, shear, bend, tris = [], [], [], []
        for j in range(ny):
            for i in range(nx):
                a = self._idx(i, j)
                if i + 1 < nx:
                    struct.append((a, self._idx(i + 1, j)))
                if j + 1 < ny:
                    struct.append((a, self._idx(i, j + 1)))
                if i + 1 < nx and j + 1 < ny:
                    b = self._idx(i + 1, j)
                    c = self._idx(i, j + 1)
                    d = self._idx(i + 1, j + 1)
                    shear.append((a, d))
                    shear.append((b, c))
                    if (i + j) % 2 == 0:
                        tris.append((a, b, d))
                        tris.append((a, d, c))
                    else:
                        tris.append((a, b, c))
                        tris.append((b, d, c))
                if i + 2 < nx:
                    bend.append((a, self._idx(i + 2, j)))
                if j + 2 < ny:
                    bend.append((a, self._idx(i, j + 2)))
        self.e_struct = np.asarray(struct, dtype=np.int64)
        self.e_shear = np.asarray(shear, dtype=np.int64)
        self.e_bend = np.asarray(bend, dtype=np.int64)
        self.tris = np.asarray(tris, dtype=np.int64)

    def _rest_len(self, edges: np.ndarray) -> np.ndarray:
        di = (self.gi[edges[:, 0]] - self.gi[edges[:, 1]]).astype(float) * self.dx0
        dj = (self.gj[edges[:, 0]] - self.gj[edges[:, 1]]).astype(float) * self.dy0
        return np.sqrt(di * di + dj * dj)

    def _build_masks(self) -> None:
        di = np.abs(self.gi[:, None] - self.gi[None, :])
        dj = np.abs(self.gj[:, None] - self.gj[None, :])
        cheb = np.maximum(di, dj)
        self.self_pairs = np.argwhere(np.triu(cheb >= 3, 1))
        tv = self.tris
        near = np.zeros((self.n, len(tv)), dtype=bool)
        for k in range(3):
            near |= cheb[:, tv[:, k]] <= 2
        self.pt_v, self.pt_t = np.nonzero(~near)
        self.self_mask = (cheb >= 3)

    # ------------------------------------------------------------ boundaries
    def _wall_center_x(self, t: float) -> float:
        s = _smoothstep(t, self.t_close0, self.t_close1)
        return self.wall_x0 + (self.wall_x1 - self.wall_x0) * s

    def _external_force(self) -> np.ndarray:
        return self.m[:, None] * self.gravity[None, :] - self.drag * self.v

    # ----------------------------------------------------------- constraints
    def _solve_distance(self, x, edges, rest, alpha, lam):
        if len(edges) == 0:
            return
        i0 = edges[:, 0]
        i1 = edges[:, 1]
        d = x[i0] - x[i1]
        ln = np.sqrt(np.einsum("ij,ij->i", d, d))
        safe = np.maximum(ln, 1e-12)
        nrm = d / safe[:, None]
        C = ln - rest
        w0 = self.w[i0]
        w1 = self.w[i1]
        at = alpha / (self.h * self.h)
        dlam = (-C - at * lam) / (w0 + w1 + at)
        lam += dlam
        corr = self.omega * dlam[:, None] * nrm
        np.add.at(x, i0, w0[:, None] * corr)
        np.add.at(x, i1, -w1[:, None] * corr)

    def _solve_self_contact(self, x) -> None:
        if len(self.self_pairs) == 0:
            return
        i0 = self.self_pairs[:, 0]
        i1 = self.self_pairs[:, 1]
        d = x[i0] - x[i1]
        ln = np.sqrt(np.einsum("ij,ij->i", d, d))
        act = ln < self.h_self
        if not np.any(act):
            return
        i0 = i0[act]
        i1 = i1[act]
        d = d[act]
        ln = np.maximum(ln[act], 1e-9)
        nrm = d / ln[:, None]
        corr = 0.8 * 0.5 * (self.h_self - ln)[:, None] * nrm
        np.add.at(x, i0, corr)
        np.add.at(x, i1, -corr)

    def _solve_node_triangle(self, x) -> None:
        vi = self.pt_v
        ti = self.pt_t
        if len(vi) == 0:
            return
        tri = self.tris[ti]
        a = x[tri[:, 0]]
        b = x[tri[:, 1]]
        c = x[tri[:, 2]]
        p = x[vi]
        cen = (a + b + c) / 3.0
        rel = p - cen
        cand = np.einsum("ij,ij->i", rel, rel) < 0.0625
        if not np.any(cand):
            return
        vi = vi[cand]
        tri = tri[cand]
        a, b, c, p = a[cand], b[cand], c[cand], p[cand]
        e0 = b - a
        e1 = c - a
        nvec = np.cross(e0, e1)
        nl = np.sqrt(np.einsum("ij,ij->i", nvec, nvec))
        ok = nl > 1e-12
        if not np.any(ok):
            return
        vi, tri = vi[ok], tri[ok]
        a, b, c, p = a[ok], b[ok], c[ok], p[ok]
        e0, e1, nvec, nl = e0[ok], e1[ok], nvec[ok], nl[ok]
        nrm = nvec / nl[:, None]
        d00 = np.einsum("ij,ij->i", e0, e0)
        d01 = np.einsum("ij,ij->i", e0, e1)
        d11 = np.einsum("ij,ij->i", e1, e1)
        pa = p - a
        d20 = np.einsum("ij,ij->i", pa, e0)
        d21 = np.einsum("ij,ij->i", pa, e1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
        bv = (d11 * d20 - d01 * d21) / den
        bw = (d00 * d21 - d01 * d20) / den
        bary = np.stack((1.0 - bv - bw, bv, bw), axis=1)
        bary = np.clip(bary, 0.0, 1.0)
        bary /= np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)
        q = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
        diff = p - q
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        act = dist < self.h_self
        if not np.any(act):
            return
        vi, tri, bary = vi[act], tri[act], bary[act]
        diff, dist, nrm = diff[act], dist[act], nrm[act]
        side = np.where(np.einsum("ij,ij->i", diff, nrm) >= 0.0, 1.0, -1.0)
        dirv = np.where(
            dist[:, None] > 1e-7,
            diff / np.maximum(dist, 1e-7)[:, None],
            side[:, None] * nrm,
        )
        pen = (self.h_self - dist)[:, None]
        wv = self.w[vi]
        wt = (
            bary[:, 0] * self.w[tri[:, 0]]
            + bary[:, 1] * self.w[tri[:, 1]]
            + bary[:, 2] * self.w[tri[:, 2]]
        )
        wsum = np.maximum(wv + wt, 1e-12)
        scale = 0.6
        np.add.at(x, vi, scale * (wv / wsum)[:, None] * pen * dirv)
        for k in range(3):
            coef = scale * bary[:, k] * (self.w[tri[:, k]] / wsum)
            np.add.at(x, tri[:, k], -(coef[:, None] * pen * dirv))

    def _solve_boundaries(self, x, wall_cx):
        zmin = self.ground_top + self.r_c
        hit_g = x[:, 2] < zmin
        x[hit_g, 2] = zmin
        hit_w = np.zeros(self.n, dtype=bool)
        for sgn in (-1.0, 1.0):
            cx = sgn * wall_cx
            face = cx - sgn * self.wall_half[0]
            inside_y = np.abs(x[:, 1]) < (self.wall_half[1] + self.r_c)
            inside_z = np.abs(x[:, 2] - self.wall_z) < (self.wall_half[2] + self.r_c)
            if sgn > 0.0:
                pen = (x[:, 0] > face - self.r_c) & inside_y & inside_z
                x[pen, 0] = face - self.r_c
            else:
                pen = (x[:, 0] < face + self.r_c) & inside_y & inside_z
                x[pen, 0] = face + self.r_c
            hit_w |= pen
        return hit_g, hit_w

    # -------------------------------------------------------------- time step
    def _substep(self) -> None:
        h = self.h
        self.f_ext = self._external_force()
        self.v += h * self.f_ext * self.w[:, None]
        self.v *= self.vel_damp
        x_prev = self.x
        x = x_prev + h * self.v

        lam_s = np.zeros(len(self.e_struct))
        lam_h = np.zeros(len(self.e_shear))
        lam_b = np.zeros(len(self.e_bend))
        for _ in range(self.iters):
            self._solve_distance(x, self.e_struct, self.L_struct, self.alpha_struct, lam_s)
            self._solve_distance(x, self.e_shear, self.L_shear, self.alpha_shear, lam_h)
            self._solve_distance(x, self.e_bend, self.L_bend, self.alpha_bend, lam_b)

        wall_cx = self._wall_center_x(self.t + h)
        for _ in range(2):
            self._solve_self_contact(x)
            self._solve_node_triangle(x)
        hit_g, hit_w = self._solve_boundaries(x, wall_cx)

        v_new = (x - x_prev) / h
        if np.any(hit_g):
            v_new[hit_g, 0:2] *= 1.0 - self.mu_ground
            v_new[hit_g, 2] = np.maximum(v_new[hit_g, 2], 0.0)
        if np.any(hit_w):
            v_new[hit_w, 1:3] *= 1.0 - self.mu_wall
        self.x = x
        self.v = np.clip(np.nan_to_num(v_new, nan=0.0, posinf=0.0, neginf=0.0), -25.0, 25.0)
        self.t += h
        self._steps += 1
        self._plastic_update()
        if self.n:
            self._max_speed = max(self._max_speed, float(np.max(np.abs(self.v))))

    def _plastic_update(self) -> None:
        i0 = self.e_bend[:, 0]
        i1 = self.e_bend[:, 1]
        d = self.x[i0] - self.x[i1]
        ln = np.sqrt(np.einsum("ij,ij->i", d, d))
        strain = (ln - self.L_bend) / np.maximum(self.L_bend, 1e-9)
        yielding = np.abs(strain) > self.bend_yield
        if np.any(yielding):
            self.L_bend[yielding] += self.bend_plastic_rate * (
                ln[yielding] - self.L_bend[yielding]
            )
            np.clip(self.L_bend, 0.2 * self.L_bend0, self.L_bend0, out=self.L_bend)

    # ------------------------------------------------------------ diagnostics
    def _deformation_gradient(self) -> np.ndarray:
        grid = self.x.reshape(self.ny, self.nx, 3)
        dpdx = np.gradient(grid, self.dx0, axis=1).reshape(-1, 3)
        dpdy = np.gradient(grid, self.dy0, axis=0).reshape(-1, 3)
        F3 = np.stack((dpdx, dpdy), axis=2)  # [N,3,2]
        _, R = np.linalg.qr(F3)              # [N,2,2] material-frame gradient
        diag = np.einsum("nii->ni", R)
        sgn = np.where(diag >= 0.0, 1.0, -1.0)
        R = R * sgn[:, :, None]
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(R)

    def _box_sdf(self, x, center, half):
        q = np.abs(x - center[None, :]) - half[None, :]
        outside = np.sqrt(np.sum(np.maximum(q, 0.0) ** 2, axis=1))
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def _contact_signed_distance(self) -> np.ndarray:
        x = self.x
        gaps = [(x[:, 2] - self.ground_top) - self.r_c]
        cx = self._wall_center_x(self.t)
        for sgn in (-1.0, 1.0):
            center = np.array([sgn * cx, 0.0, self.wall_z])
            gaps.append(self._box_sdf(x, center, self.wall_half) - self.r_c)
        d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
        dsel = np.where(self.self_mask, d2, 1.0e6)
        self_gap = np.sqrt(np.min(dsel, axis=1)) - self.h_self
        gaps.append(np.minimum(self_gap, 1.0e3))
        return np.concatenate(gaps).astype(float)

    def _frame(self):
        from worldsolver.contracts import StateFrame

        x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        fext = np.nan_to_num(self.f_ext, nan=0.0, posinf=0.0, neginf=0.0)
        csd = np.nan_to_num(
            self._contact_signed_distance(), nan=0.0, posinf=1.0e3, neginf=-1.0e3
        )
        frame = StateFrame(
            float(max(self.t, 0.0)),
            np.ascontiguousarray(x),
            np.ascontiguousarray(v),
            mass=np.ascontiguousarray(self.m),
            body_id=self.body_id,
            attributes={
                "mass_quantity": np.ascontiguousarray(self.m.copy()),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "external_force": np.ascontiguousarray(fext),
                "contact_signed_distance": csd,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- driver
    def advance_to(self, time: float):
        target = float(time)
        eps = 0.25 * self.h
        while self.t < target - eps:
            self._substep()
        if abs(self.t - target) < 2.0 * self.h:
            self.t = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "XPBD thin shell, plastic bending, self-contact",
            "substeps_taken": int(self._steps),
            "substep_dt": float(self.h),
            "iterations": int(self.iters),
            "nodes": int(self.n),
            "max_speed": float(self._max_speed),
            "final_wall_gap": float(2.0 * (self.wall_x1 - self.wall_half[0])),
        }
