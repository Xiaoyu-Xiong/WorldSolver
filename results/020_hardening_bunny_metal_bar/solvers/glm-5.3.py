"""Viscoplastic bunny-on-bar solver (no Genesis; pure NumPy physics).

Physical model
--------------
The bunny is a closed surface triangle mesh.  Its mechanics are represented by
a network of elastoplastic springs connecting each vertex to its neighbours
within a short cutoff (mesh edges plus near diagonals).  Each spring has an
*elastic* rest length ``r`` and obeys:

* elastic force        f = k_eff * (l - r)  (Hooke along the spring),
* plastic yield        |strain| > y  ->  the rest length flows so that the
  elastic strain relaxes back to the yield surface (rate-independent
  perfect viscoplastic flow),
* work hardening       the local yield strain grows with the accumulated
  plastic strain ``p``:   y = y0 * (1 + H * p),
  and the local elastic stiffness grows as well, so strongly deformed
  regions stop flowing and become more elastic -> post-impact jiggle.

Integration is semi-implicit Euler with many deterministic substeps, spring
damping (oscillation decay), gravity, and collision projection (penalty-free
position projection with restitution + Coulomb friction) against the frozen
metal bar cylinder and the ground plane.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.bar_z = float(cfg.get("bar_z", 0.66))
        self.bar_r = float(cfg.get("bar_radius", 0.085))
        self.restitution = float(cfg.get("restitution", 0.3))
        self.friction = float(cfg.get("friction", 0.5))
        self.damping = float(cfg.get("damping", 1.0))
        self.stiffness = float(cfg.get("stiffness", 10.0))
        self.H = float(cfg.get("hardening_rate", 1.0))
        self.l_c = float(cfg.get("contact_length_scale", 0.04))
        self.y0 = 0.045          # initial yield strain
        self.p_cap = 4.0         # cap on accumulated plastic strain
        self.k_cap = 2.5         # cap on hardening stiffness multiplier

        # --- mesh ---------------------------------------------------------
        X = np.asarray(scene.vertices, dtype=float)
        self.X0 = X.copy()
        self.n = len(X)

        # faces come from the asset next to the scene definition
        import inspect
        import os
        scene_file = inspect.getfile(type(scene))
        base = os.path.dirname(os.path.abspath(scene_file))
        npz = os.path.join(base, cfg["subject_asset"])
        with np.load(npz) as data:
            self.faces = np.asarray(data["faces"], dtype=np.int64)
        faces = self.faces

        self.x = X.copy()
        v0 = np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float)
        self.v = np.tile(v0, (self.n, 1))
        total_mass = float(cfg.get("subject_mass", 1.0))
        self.m = np.full(self.n, total_mass / self.n, dtype=float)

        # --- springs --------------------------------------------------------
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = np.unique(np.sort(e, axis=1), axis=0)
        try:
            from scipy.spatial import cKDTree

            cut = 1.8 * float(np.linalg.norm(X[e[:, 0]] - X[e[:, 1]], axis=1).mean())
            pairs = np.array(sorted(cKDTree(X).query_pairs(cut)), dtype=np.int64)
            if len(pairs) >= len(e):
                e = np.unique(
                    np.sort(np.vstack([e, pairs]), axis=1), axis=0
                )
        except Exception:
            pass
        self.ei = np.ascontiguousarray(e[:, 0])
        self.ej = np.ascontiguousarray(e[:, 1])
        self.r0 = np.linalg.norm(X[self.ei] - X[self.ej], axis=1)
        self.r = self.r0.copy()
        self.p = np.zeros(self.n, dtype=float)      # accumulated plastic strain
        mean_len = float(self.r0.mean())
        # spring stiffness: E_eff * length  (stiffness interpreted as kPa)
        self.k0 = 200.0 * self.stiffness * mean_len
        self.m_red = 0.5 * self.m[0]
        self.zeta = 0.35 * self.damping

        # --- rest normals (for deformation-gradient fit) ---------------------
        self.n0 = self._vertex_normals(self.X0, faces)

        # --- neighbour structure for per-vertex F ---------------------------
        # flat list: for every edge both (a->b) and (b->a)
        src = np.concatenate([self.ei, self.ej])
        dst = np.concatenate([self.ej, self.ei])
        self.nbr_src = src
        self.nbr_dst = dst
        self.h_n = mean_len
        self.M0 = None  # precomputed rest metric (built lazily)

        # --- time stepping ---------------------------------------------------
        self.dt_sub = float(cfg["sample_dt"]) / 24.0
        self.t = 0.0
        self.v_max = 15.0
        self.contact_impulse = np.zeros(3, dtype=float)
        self.contact_window = 0.0
        self._ready = True

    # ------------------------------------------------------------- utilities
    @staticmethod
    def _vertex_normals(V, faces):
        a, b, c = V[faces[:, 0]], V[faces[:, 1]], V[faces[:, 2]]
        fn = np.cross(b - a, c - a)
        n = np.zeros_like(V)
        for k in range(3):
            np.add.at(n, faces[:, k], fn)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        bad = ln[:, 0] < 1e-12
        n[bad] = np.array([0.0, 0.0, 1.0])
        n[~bad] /= ln[~bad]
        return n

    # ------------------------------------------------------------- sampling
    def _signed_distances(self):
        """Signed gaps to ground and bar per vertex (min of obstacles)."""
        gap_ground = self.x[:, 2] - self.ground_z
        d = np.sqrt(self.x[:, 0] ** 2 + (self.x[:, 2] - self.bar_z) ** 2)
        gap_bar = d - self.bar_r
        return np.minimum(gap_ground, gap_bar), gap_ground, gap_bar

    def _deformation_gradients(self):
        X0, x = self.X0, self.x
        u = X0[self.nbr_dst] - X0[self.nbr_src]          # rest offsets
        w = x[self.nbr_dst] - x[self.nbr_src]            # current offsets
        ncur = self._vertex_normals(x, self.faces)
        s = 0.5 * self.h_n                                # virtual normal sample
        A = np.zeros((self.n, 3, 3))
        M = np.zeros((self.n, 3, 3))
        # neighbor contributions (unit weight): A += w u^T ; M += u u^T
        np.add.at(A, self.nbr_src, w[:, :, None] * u[:, None, :])
        np.add.at(M, self.nbr_src, u[:, :, None] * u[:, None, :])
        # virtual normal neighbour: rest n0 -> current ncur, weight s
        A += s * s * ncur[:, :, None] * self.n0[:, None, :]
        M += s * s * self.n0[:, :, None] * self.n0[:, None, :]
        # regularisation
        lam = 1e-4 * self.h_n * self.h_n
        M[:, 0, 0] += lam
        M[:, 1, 1] += lam
        M[:, 2, 2] += lam
        # F = A M^{-1}  =>  F^T = solve(M^T, A^T)
        try:
            Ft = np.linalg.solve(M.transpose(0, 2, 1), A.transpose(0, 2, 1))
            F = Ft.transpose(0, 2, 1)
        except np.linalg.LinAlgError:
            F = np.tile(np.eye(3), (self.n, 1, 1))
        if not np.isfinite(F).all():
            bad = ~np.isfinite(F).all(axis=(1, 2))
            F[bad] = np.eye(3)
        return F

    # --------------------------------------------------------------- stepping
    def _substep(self, dt):
        x, v, m = self.x, self.v, self.m
        ei, ej = self.ei, self.ej

        dx = x[ej] - x[ei]
        l = np.linalg.norm(dx, axis=1)
        l = np.maximum(l, 1e-9)
        d = dx / l[:, None]

        # --- plasticity with work hardening --------------------------------
        p_loc = 0.5 * (self.p[ei] + self.p[ej])
        y = self.y0 * (1.0 + self.H * p_loc)
        strain = (l - self.r) / self.r
        over = np.abs(strain) > y
        if np.any(over):
            dlam = np.abs(strain[over]) - y[over]
            sgn = np.sign(strain[over])
            self.r[over] = np.clip(
                self.r[over] * (1.0 + sgn * dlam),
                0.25 * self.r0[over],
                3.0 * self.r0[over],
            )
            add = 0.5 * dlam
            np.add.at(self.p, ei[over], add)
            np.add.at(self.p, ej[over], add)
            # elastic strain after the plastic relaxation
            strain = (l - self.r) / self.r
            y = self.y0 * (1.0 + self.H * 0.5 * (self.p[ei] + self.p[ej]))
            strain = np.clip(strain, -y, y)
        self.p = np.minimum(self.p, self.p_cap)

        # --- elastic + damping forces ---------------------------------------
        p_loc = np.minimum(0.5 * (self.p[ei] + self.p[ej]), self.p_cap)
        k_eff = np.minimum(self.k0 * (1.0 + 0.4 * self.H * p_loc), self.k_cap * self.k0)
        c_eff = 2.0 * self.zeta * np.sqrt(k_eff * self.m_red)
        vrel = np.einsum("ij,ij->i", v[ej] - v[ei], d)
        fmag = k_eff * strain + c_eff * vrel
        fvec = fmag[:, None] * d          # force on i (towards j when stretched)

        fx = np.bincount(ei, weights=fvec[:, 0], minlength=self.n) - np.bincount(
            ej, weights=fvec[:, 0], minlength=self.n
        )
        fy = np.bincount(ei, weights=fvec[:, 1], minlength=self.n) - np.bincount(
            ej, weights=fvec[:, 1], minlength=self.n
        )
        fz = np.bincount(ei, weights=fvec[:, 2], minlength=self.n) - np.bincount(
            ej, weights=fvec[:, 2], minlength=self.n
        )
        acc = np.stack([fx, fy, fz], axis=1) / m[:, None] + self.gravity[None, :]

        # --- integrate -----------------------------------------------------
        v += acc * dt
        # mild global damping (oscillation decay)
        v *= max(0.0, 1.0 - 0.15 * self.damping * dt)
        speed = np.linalg.norm(v, axis=1, keepdims=True)
        scale = np.minimum(1.0, self.v_max / np.maximum(speed, 1e-12))
        v *= scale
        x += v * dt

        # --- collisions ----------------------------------------------------
        self._collide(dt)

    def _collide(self, dt):
        x, v, m = self.x, self.v, self.m
        e, mu = self.restitution, self.friction
        J = np.zeros(3)

        # ground plane
        gap = x[:, 2] - self.ground_z
        hit = gap < 0.0
        if np.any(hit):
            idx = np.flatnonzero(hit)
            x[idx, 2] = self.ground_z
            vn = v[idx, 2]
            app = vn < 0.0
            if np.any(app):
                ia = idx[app]
                dv_n = -(1.0 + e) * v[ia, 2]
                v[ia, 2] += dv_n
                J[2] += m[0] * dv_n.sum()
                # friction on tangential velocity
                vt = v[ia, :2]
                vm = np.linalg.norm(vt, axis=1)
                need = mu * np.abs(dv_n)
                fr = np.minimum(need, vm)
                sc = np.where(vm > 1e-12, 1.0 - fr / np.maximum(vm, 1e-12), 0.0)
                v[ia, 0] = vt[:, 0] * sc
                v[ia, 1] = vt[:, 1] * sc
                J[0] += m[0] * np.sum(vt[:, 0] * sc - vt[:, 0])
                J[1] += m[0] * np.sum(vt[:, 1] * sc - vt[:, 1])

        # metal bar (cylinder along y at height bar_z)
        rad = np.sqrt(x[:, 0] ** 2 + (x[:, 2] - self.bar_z) ** 2)
        hit = rad < self.bar_r
        if np.any(hit):
            idx = np.flatnonzero(hit)
            rr = np.maximum(rad[idx], 1e-9)
            nx = x[idx, 0] / rr
            nz = (x[idx, 2] - self.bar_z) / rr
            # project to surface
            x[idx, 0] = nx * self.bar_r
            x[idx, 2] = self.bar_z + nz * self.bar_r
            vn = v[idx, 0] * nx + v[idx, 2] * nz
            app = vn < 0.0
            if np.any(app):
                ia = idx[app]
                na = np.stack([nx[app], np.zeros(len(ia)), nz[app]], axis=1)
                vna = v[ia, 0] * na[:, 0] + v[ia, 2] * na[:, 2]
                dv_n_vec = -(1.0 + e) * vna[:, None] * na
                v[ia] += dv_n_vec
                J += m[0] * dv_n_vec.sum(axis=0)
                # friction: tangential component (includes y and tangential-in-plane)
                vt = v[ia] - np.einsum("ij,ij->i", v[ia], na)[:, None] * na
                vm = np.linalg.norm(vt, axis=1)
                need = mu * (1.0 + e) * np.abs(vna)
                fr = np.minimum(need, vm)
                sc = np.where(vm > 1e-12, 1.0 - fr / np.maximum(vm, 1e-12), 0.0)
                dv_t = vt * sc[:, None] - vt
                v[ia] += dv_t
                J += m[0] * dv_t.sum(axis=0)

        self.contact_impulse += J
        self.contact_window += dt

    # ---------------------------------------------------------------- frames
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("initialize() must be called first")
        t_target = float(time)
        n_steps = int(np.floor((t_target - self.t) / self.dt_sub + 1e-9))
        for _ in range(n_steps):
            self._substep(self.dt_sub)
            self.t += self.dt_sub
        # snap the remaining epsilon
        rem = t_target - self.t
        if rem > 1e-9:
            self._substep(rem)
            self.t = t_target
        self.t = max(self.t, t_target)

        gap, _, _ = self._signed_distances()
        near = gap < 4.0 * self.l_c
        csd = gap[near] if np.any(near) else np.zeros(0)

        if self.contact_window > 1e-12:
            interaction = (self.contact_impulse / self.contact_window).reshape(1, 3)
        else:
            interaction = np.zeros((1, 3))
        self.contact_impulse[:] = 0.0
        self.contact_window = 0.0

        F = self._deformation_gradients()
        frame = StateFrame(
            time=t_target,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.m.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.m[:, None] * self.gravity[None, :],
                "contact_signed_distance": csd,
                "interaction_force": interaction,
            },
            scalars={"contact_length_scale": self.l_c},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "vertices": int(self.n),
            "springs": int(len(self.r0)),
            "max_plastic_strain": float(np.max(self.p)) if self._ready else 0.0,
            "mean_plastic_strain": float(np.mean(self.p)) if self._ready else 0.0,
            "time": float(self.t) if self._ready else 0.0,
        }
