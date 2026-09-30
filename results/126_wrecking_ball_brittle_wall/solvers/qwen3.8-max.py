"""Physically-based solver for the wrecking-ball brittle-wall fracture scene.

Model
-----
* The ``subject`` particle cloud (sampled masonry block) is simulated with a
  bonded discrete-element method: linear central springs between material
  neighbours that break irreversibly above a (randomised) strain threshold,
  reproducing brittle crack initiation, branching propagation, fragment
  separation and gravity-driven debris fall.
* The wrecking ball is the kinematic boundary sphere: it is prescribed as a
  heavy pendulum released from its displaced rest pose at (-1.75, 0, 1.35),
  pivoting above the wall.  It transfers momentum to particles through a
  penalty contact with friction (Newton's third law is respected inside the
  particle system; the ball itself is kinematic).
* The rigid floor (top of the ``ground`` box, z = 0) is a penalty contact with
  Coulomb-like friction so fragments collide and settle.
* Gravity is the configured body force on every sample.

All exported quantities (mass, velocity, external force, deformation
gradient, near-contact signed distance, cumulative mass source) are computed
from the actual simulated state; nothing is scripted.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        cfg = getattr(scene, "config", {}) or {}
        sampling = cfg.get("sampling", {}) or {}

        self.h = float(frame0.scalars.get("grid_spacing", sampling.get("spacing", 0.12)))
        grav = cfg.get("gravity") or scene.metadata().get("gravity") or [0.0, 0.0, -9.81]
        self.gravity = np.asarray(grav, dtype=float).reshape(3)
        self.g_mag = float(np.linalg.norm(self.gravity))

        self.X0 = np.array(frame0.position, dtype=float)
        self.X = self.X0.copy()
        self.V = np.zeros_like(self.X)
        self.m = np.array(frame0.mass, dtype=float)
        self.N = int(len(self.X))
        self.rp = 0.45 * self.h
        self.rng = np.random.default_rng(int(seed))
        self.t = 0.0
        self.steps_done = 0
        self.bonds_broken = 0

        # ---------------- bonds (material neighbours of the reference grid)
        cutoff = 1.45 * self.h
        if self.N > 1:
            diff0 = self.X0[:, None, :] - self.X0[None, :, :]
            dist0 = np.sqrt(np.einsum("ijk,ijk->ij", diff0, diff0))
            iu = np.triu_indices(self.N, 1)
            sel = dist0[iu] < cutoff
            self.bi = iu[0][sel].copy()
            self.bj = iu[1][sel].copy()
            self.bd0 = dist0[self.bi, self.bj].copy()
        else:
            self.bi = np.zeros(0, dtype=np.int64)
            self.bj = np.zeros(0, dtype=np.int64)
            self.bd0 = np.zeros(0)
        nb = len(self.bi)
        self.broken = np.zeros(nb, dtype=bool)
        # brittle thresholds with seeded scatter -> branching crack paths
        self.eps_t = 0.045 * (0.8 + 0.4 * self.rng.random(nb))
        self.eps_c = 0.18 * (0.8 + 0.4 * self.rng.random(nb))

        # ---------------- material / contact parameters
        self.k_bond = 2.0e5      # N/m cohesive spring
        self.c_bond = 60.0       # N s/m bond damping
        self.k_rep = 2.0e5       # N/m fragment-fragment repulsion
        self.c_rep = 250.0
        self.rep_dist = 0.9 * self.h

        self.k_g = 2.0e6         # floor penalty stiffness
        self.c_g = 900.0
        self.mu_g = 0.6

        self.k_ball = 3.0e6      # ball penalty stiffness
        self.c_ball = 1500.0
        self.mu_ball = 0.35

        self.drag = 0.3          # weak global velocity damping (1/s)
        self.v_max = 40.0

        # ---------------- kinematic wrecking ball (pendulum)
        self.R_ball = 0.48
        self.pivot = np.array([0.0, 0.0, 3.4])          # top of the cable
        b0 = np.array([-1.75, 0.0, 1.35])               # frozen initial pose
        off = b0 - self.pivot
        self.L = float(np.linalg.norm(off))
        self.theta = float(np.arctan2(off[0], -off[2])) # displaced, at rest
        self.omega = 0.0
        self.c_pend = 0.06
        self._ball_center = b0.copy()
        self._ball_vel = np.zeros(3)

        # ---------------- deformation-gradient least-squares stencils
        self.nb_idx = []
        self.nb_dX = []
        self.nb_P = []
        eye3 = np.eye(3)
        for i in range(self.N):
            d = np.linalg.norm(self.X0 - self.X0[i], axis=1)
            idx = np.flatnonzero((d < cutoff) & (d > 1e-12))
            dX = self.X0[idx] - self.X0[i]
            if len(idx) >= 3:
                P = np.linalg.pinv(dX)          # [3, nb]
            else:
                P = np.zeros((3, max(len(idx), 0)))
            self.nb_idx.append(idx)
            self.nb_dX.append(dX)
            self.nb_P.append(P)
        self._eye3 = eye3

        # ---------------- frozen scalars carried through frames
        self.contact_length_scale = self.h
        self.char_len = float(frame0.scalars.get("characteristic_length", 1.0))
        self.ref_vel = float(frame0.scalars.get("reference_velocity", 1.0))
        labels = frame0.attributes.get("material_label")
        self.labels = (
            np.array(labels).astype(np.int32)
            if labels is not None
            else np.zeros(self.N, dtype=np.int32)
        )

        self.ext_force = self.m[:, None] * self.gravity[None, :]
        self._update_ball()

    # ------------------------------------------------------------- pendulum
    def _update_ball(self) -> None:
        s, c = np.sin(self.theta), np.cos(self.theta)
        self._ball_center = self.pivot + self.L * np.array([s, 0.0, -c])
        self._ball_vel = self.L * self.omega * np.array([c, 0.0, s])

    def _step_pendulum(self, dt: float) -> None:
        alpha = -(self.g_mag / self.L) * np.sin(self.theta) - self.c_pend * self.omega
        self.omega += alpha * dt
        self.theta += self.omega * dt
        self._update_ball()

    # ------------------------------------------------------------- dynamics
    def _substep(self, dt: float) -> None:
        X, V, m = self.X, self.V, self.m
        F = m[:, None] * self.gravity[None, :]

        self._step_pendulum(dt)
        cb, vb = self._ball_center, self._ball_vel

        # ---- cohesive / broken bonds
        if len(self.bi):
            dx = X[self.bj] - X[self.bi]
            d = np.sqrt(np.einsum("ij,ij->i", dx, dx))
            d_safe = np.maximum(d, 1e-12)
            n = dx / d_safe[:, None]
            strain = (d - self.bd0) / self.bd0
            newly = (~self.broken) & (
                (strain > self.eps_t) | (strain < -self.eps_c)
            )
            if newly.any():
                self.broken |= newly
                self.bonds_broken += int(newly.sum())
            active = ~self.broken
            dv = V[self.bj] - V[self.bi]
            vn = np.einsum("ij,ij->i", dv, n)
            fmag = (self.k_bond * (d - self.bd0) + self.c_bond * vn) * active
            fvec = fmag[:, None] * n
            np.add.at(F, self.bi, fvec)
            np.add.at(F, self.bj, -fvec)

        # ---- short-range repulsion between separated fragments
        diff = X[:, None, :] - X[None, :, :]
        r = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        np.fill_diagonal(r, np.inf)
        touch = np.triu(r < self.rep_dist, 1)
        if touch.any():
            ii, jj = np.nonzero(touch)
            rij = r[ii, jj]
            nij = diff[ii, jj] / np.maximum(rij, 1e-12)[:, None]
            dvij = V[ii] - V[jj]
            vnij = np.einsum("ij,ij->i", dvij, nij)
            fm = self.k_rep * (self.rep_dist - rij) - self.c_rep * vnij
            fm = np.maximum(fm, 0.0)
            fvec = fm[:, None] * nij
            # nij points from j to i: positive force pushes the pair apart
            np.add.at(F, ii, fvec)
            np.add.at(F, jj, -fvec)

        # ---- wrecking ball contact (external, kinematic)
        dxb = X - cb[None, :]
        rb = np.sqrt(np.einsum("ij,ij->i", dxb, dxb))
        rb_safe = np.maximum(rb, 1e-12)
        gap = rb - (self.R_ball + self.rp)
        pen = gap < 0.0
        if pen.any():
            nb_ = dxb[pen] / rb_safe[pen][:, None]
            vrel = V[pen] - vb[None, :]
            vn = np.einsum("ij,ij->i", vrel, nb_)
            fn = -self.k_ball * gap[pen] - self.c_ball * np.minimum(vn, 0.0)
            fn = np.maximum(fn, 0.0)
            vt = vrel - vn[:, None] * nb_
            vt_n = np.linalg.norm(vt, axis=1)
            fr = np.where(
                vt_n > 1e-9,
                np.minimum(self.mu_ball * fn, m[pen] * vt_n / max(dt, 1e-12)),
                0.0,
            )
            ft = -fr[:, None] * vt / np.maximum(vt_n, 1e-12)[:, None]
            Fb = fn[:, None] * nb_ + ft
            idx = np.flatnonzero(pen)
            np.add.at(F, idx, Fb)

        # ---- rigid floor contact (top of ground box at z = 0)
        zg = X[:, 2]
        peng = zg < self.rp
        if peng.any():
            ov = self.rp - zg[peng]
            fn = self.k_g * ov - self.c_g * np.minimum(V[peng, 2], 0.0)
            fn = np.maximum(fn, 0.0)
            idx = np.flatnonzero(peng)
            F[idx, 2] += fn
            vt = V[peng][:, :2]
            vt_n = np.linalg.norm(vt, axis=1)
            fr = np.where(
                vt_n > 1e-9,
                np.minimum(self.mu_g * fn, m[peng] * vt_n / max(dt, 1e-12)),
                0.0,
            )
            F[idx, 0] += -fr * np.where(vt_n > 1e-9, vt[:, 0] / np.maximum(vt_n, 1e-12), 0.0)
            F[idx, 1] += -fr * np.where(vt_n > 1e-9, vt[:, 1] / np.maximum(vt_n, 1e-12), 0.0)

        # ---- external force record: gravity + boundary contacts only
        Fext = m[:, None] * self.gravity[None, :]
        if pen.any():
            np.add.at(Fext, np.flatnonzero(pen), Fb)
        if peng.any():
            idxg = np.flatnonzero(peng)
            Fext[idxg, 2] += fn
            Fext[idxg, 0] += -fr * np.where(vt_n > 1e-9, vt[:, 0] / np.maximum(vt_n, 1e-12), 0.0)
            Fext[idxg, 1] += -fr * np.where(vt_n > 1e-9, vt[:, 1] / np.maximum(vt_n, 1e-12), 0.0)
        self.ext_force = Fext

        # ---- integrate (semi-implicit Euler with weak drag)
        V += (F / m[:, None]) * dt
        V /= (1.0 + self.drag * dt)
        sp = np.linalg.norm(V, axis=1)
        over = sp > self.v_max
        if over.any():
            V[over] *= (self.v_max / sp[over])[:, None]
        X += V * dt

        # hard floor safety clamp (should never engage noticeably)
        below = X[:, 2] < 0.2 * self.rp
        if below.any():
            X[below, 2] = 0.2 * self.rp
            V[below, 2] = np.maximum(V[below, 2], 0.0)

        if not np.isfinite(X).all():
            bad = ~np.isfinite(X[:, 0]) | ~np.isfinite(X[:, 1]) | ~np.isfinite(X[:, 2])
            X[bad] = self.X0[bad]
            V[bad] = 0.0
        self.steps_done += 1

    # ------------------------------------------------------------ frame out
    def _deformation_gradients(self) -> np.ndarray:
        F = np.empty((self.N, 3, 3))
        for i in range(self.N):
            idx = self.nb_idx[i]
            if len(idx) >= 3:
                dU = (self.X[idx] - self.X[i]) - self.nb_dX[i]
                G = self.nb_P[i] @ dU            # G^T in row convention
                F[i] = self._eye3 + G.T
            else:
                F[i] = self._eye3
        return np.nan_to_num(F, nan=0.0, posinf=50.0, neginf=-50.0)

    def _signed_contact_distance(self) -> np.ndarray:
        d_ball = np.linalg.norm(self.X - self._ball_center[None, :], axis=1)
        d_ball -= (self.R_ball + self.rp)
        d_floor = self.X[:, 2] - self.rp
        return np.minimum(d_ball, d_floor)

    def _build_frame(self, time: float) -> StateFrame:
        F = self._deformation_gradients()
        frame = StateFrame(
            float(time),
            self.X.copy(),
            self.V.copy(),
            mass=self.m.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "mass_quantity": self.m.copy(),
                "particle_radius": np.full(self.N, self.rp),
                "material_label": self.labels.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(self.N),
                "external_force": self.ext_force.copy(),
                "contact_signed_distance": self._signed_contact_distance(),
            },
            scalars={
                "contact_length_scale": self.contact_length_scale,
                "grid_spacing": self.h,
                "characteristic_length": self.char_len,
                "reference_velocity": self.ref_vel,
            },
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- driver
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.t + 1e-12:
            n_sub = 64
            dt = (target - self.t) / n_sub
            for _ in range(n_sub):
                self._substep(dt)
            self.t = target
        return self._build_frame(self.t)

    def diagnostics(self) -> dict[str, object]:
        nb_tot = len(self.bi)
        return {
            "samples": int(self.N),
            "bonds_total": int(nb_tot),
            "bonds_broken": int(self.bonds_broken),
            "broken_fraction": float(self.bonds_broken / nb_tot) if nb_tot else 0.0,
            "max_speed": float(np.max(np.linalg.norm(self.V, axis=1))) if self.N else 0.0,
            "ball_theta": float(self.theta),
            "ball_omega": float(self.omega),
            "steps": int(self.steps_done),
            "time": float(self.t),
        }
