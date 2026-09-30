"""Physically-based solver for scene 024_stretched_cube.

Method: fixed-corotated linear elasticity (StVK-free hyperelastic model of
Klar et al. / Teran et al., "Optimization Integrator for Large Time Steps")
on a Delaunay tetrahedralization of the subject mesh, integrated with a
symplectic explicit scheme using small substeps.  Kinematic grips grab the
two end faces of the cube, smoothly stretch it to ``stretch_ratio`` by
``stretch_end``, hold, and then inject a small compressive pulse at
``pulse_time`` producing large-deformation response, element inversion
recovery and damped oscillation (damping coefficient from config).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)

        v = np.asarray(scene.vertices, dtype=float)
        self.X0 = v.copy()
        self.n = len(v)

        # ---- tetrahedralization (subject is a convex box) ----
        tets = Delaunay(v).simplices
        X = v[tets]
        e0 = X[:, 1] - X[:, 0]
        e1 = X[:, 2] - X[:, 0]
        e2 = X[:, 3] - X[:, 0]
        vol = np.einsum("ti,ti->t", e0, np.cross(e1, e2)) / 6.0
        keep = vol > 1e-9  # drop degenerate coplanar slivers
        tets = tets[keep]
        X = X[keep]
        e0, e1, e2 = e0[keep], e1[keep], e2[keep]
        vol = vol[keep]
        self.tets = tets
        self.V0 = vol
        # Dm columns are edge vectors; F = Ds @ inv(Dm)
        Dm = np.stack([e0, e1, e2], axis=2)
        self.B = np.linalg.inv(Dm)          # (T,3,3)
        self.BT = np.swapaxes(self.B, 1, 2)

        # ---- material ----
        E = float(cfg.get("stiffness", 28.0))
        nu = 0.30
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # ---- lumped masses (rho from total subject mass / tet volume) ----
        total_mass = float(cfg.get("subject_mass", 1.3))
        rho = total_mass / max(vol.sum(), 1e-12)
        m = np.zeros(self.n)
        for k in range(4):
            m += np.bincount(tets[:, k], weights=rho * vol / 4.0,
                             minlength=self.n)
        m = np.maximum(m, 1e-9)
        m *= total_mass / m.sum()
        self.mass = m
        self.inv_mass = 1.0 / m

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]),
                                  dtype=float)

        # ---- kinematic grips ----
        self.grip_x0 = 0.72          # initial grip center |x|
        self.grip_half = np.array([0.12, 0.55, 0.55])
        self.grip_z = 0.86
        self.ratio = float(cfg.get("stretch_ratio", 2.15))
        self.stretch_end = float(cfg.get("stretch_end", 1.25))
        self.pulse_time = float(cfg.get("pulse_time", 1.55))
        self.pulse_amp = 0.09        # small compressive pulse [m]
        self.pulse_tau = 0.20        # pulse duration [s]
        side = np.sign(v[:, 0])
        self.gripped = np.abs(v[:, 0]) > 0.55
        self.grip_side = side[self.gripped]
        self.grip_X0 = v[self.gripped, 0]
        self.face0 = float(np.max(np.abs(v[:, 0])))

        self.damping = float(cfg.get("damping", 0.42))
        self.contact_len = float(cfg.get("contact_length_scale", 0.04))
        self.ground_z = float(cfg.get("ground_z", 0.0))

        # ---- dynamic state ----
        self.x = v.copy()
        self.vel = np.tile(np.asarray(cfg.get("initial_velocity",
                                              [0.0, 0.0, 0.0]),
                                      dtype=float), (self.n, 1))
        self.t = 0.0
        self.dt_sub = 0.0015
        self.ext_force = self.mass[:, None] * self.gravity[None, :]
        self.steps_done = 0
        self._apply_grips(0.0)

    # ------------------------------------------------------- grip schedule
    def _schedule(self, t: float):
        """Grip stretch factor s(t), pulse p(t) and their derivatives."""
        te = self.stretch_end
        if t < te:
            u = t / te
            smooth = u * u * (3.0 - 2.0 * u)
            dsmooth = (6.0 * u - 6.0 * u * u) / te
            s = 1.0 + (self.ratio - 1.0) * smooth
            sdot = (self.ratio - 1.0) * dsmooth
        else:
            s = self.ratio
            sdot = 0.0
        tp, tau, A = self.pulse_time, self.pulse_tau, self.pulse_amp
        if tp <= t <= tp + tau:
            ph = np.pi * (t - tp) / tau
            p = A * np.sin(ph)
            pdot = A * (np.pi / tau) * np.cos(ph)
        else:
            p = 0.0
            pdot = 0.0
        return s, sdot, p, pdot

    def _apply_grips(self, t: float) -> None:
        s, sdot, p, pdot = self._schedule(t)
        self.x[self.gripped, 0] = self.grip_X0 * s - self.grip_side * p
        self.vel[self.gripped, 0] = self.grip_X0 * sdot - self.grip_side * pdot
        self.vel[self.gripped, 1] = 0.0
        self.vel[self.gripped, 2] = 0.0

    # ------------------------------------------------------------- forces
    def _tet_F(self):
        xs = self.x[self.tets]                      # (T,4,3)
        Ds = np.stack([xs[:, 1] - xs[:, 0],
                       xs[:, 2] - xs[:, 0],
                       xs[:, 3] - xs[:, 0]], axis=2)
        return Ds @ self.B                          # (T,3,3)

    def _internal_forces(self, F):
        U, S, Vt = np.linalg.svd(F)
        R = U @ Vt
        # polar rotation must stay in SO(3) even for inverted elements
        bad = np.linalg.det(R) < 0.0
        if np.any(bad):
            U = U.copy()
            U[bad, :, 2] *= -1.0
            R = U @ Vt
        c1, c2, c3 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        J = np.einsum("ti,ti->t", c1, np.cross(c2, c3))
        cof = np.stack([np.cross(c2, c3),
                        np.cross(c3, c1),
                        np.cross(c1, c2)], axis=2)  # = J F^-T
        P = 2.0 * self.mu * (F - R) \
            + (self.lam * (J - 1.0))[:, None, None] * cof
        # dE/dDs = V0 * P B^T ; force on node j+1 = -column j,
        # force on node 0 = +sum(columns)
        G = self.V0[:, None, None] * (P @ self.BT)
        f = np.zeros((self.n, 3))
        for j in range(3):
            gj = G[:, :, j]
            idx = self.tets[:, j + 1]
            for c in range(3):
                f[:, c] -= np.bincount(idx, weights=gj[:, c],
                                       minlength=self.n)
        g0 = G.sum(axis=2)
        idx0 = self.tets[:, 0]
        for c in range(3):
            f[:, c] += np.bincount(idx0, weights=g0[:, c],
                                   minlength=self.n)
        return f
    # --------------------------------------------------------------- step
    def _step(self, dt: float) -> None:
        t_next = self.t + dt
        F = self._tet_F()
        f = self._internal_forces(F)
        ext = self.mass[:, None] * self.gravity[None, :]
        a = (f + ext) * self.inv_mass[:, None]
        self.vel = self.vel + a * dt
        decay = float(np.exp(-self.damping * dt))
        self.vel *= decay

        # kinematic grip constraints at end of step; record actuation force
        s, sdot, p, pdot = self._schedule(t_next)
        x_tar = self.grip_X0 * s - self.grip_side * p
        v_tar_x = self.grip_X0 * sdot - self.grip_side * pdot
        ext_g = ext.copy()
        ext_g[self.gripped, 0] += self.mass[self.gripped] * \
            (v_tar_x - self.vel[self.gripped, 0]) / dt
        ext_g[self.gripped, 1] -= self.mass[self.gripped] * \
            self.vel[self.gripped, 1] / dt
        ext_g[self.gripped, 2] -= self.mass[self.gripped] * \
            self.vel[self.gripped, 2] / dt
        self.ext_force = ext_g

        self.vel[self.gripped, 0] = v_tar_x
        self.vel[self.gripped, 1] = 0.0
        self.vel[self.gripped, 2] = 0.0

        self.x = self.x + self.vel * dt
        self.x[self.gripped, 0] = x_tar

        # ground contact (frictional, restitution 0) - defensive; body is
        # held aloft so this should never activate
        z = self.x[:, 2]
        hit = z < self.ground_z
        if np.any(hit):
            self.x[hit, 2] = self.ground_z
            vz = self.vel[hit, 2]
            self.vel[hit, 2] = np.where(vz < 0.0, -vz * 0.0, vz)
            fric = float(self.cfg.get("friction", 0.5))
            self.vel[hit, 0] *= (1.0 - fric)
            self.vel[hit, 1] *= (1.0 - fric)

        # safety clamp (should not trigger)
        np.clip(self.vel, -100.0, 100.0, out=self.vel)

        self.t = t_next
        self.steps_done += 1

    # ----------------------------------------------------------- outputs
    def _vertex_F(self):
        F = self._tet_F()
        Fv = np.zeros((self.n, 3, 3))
        w = np.zeros(self.n)
        wF = self.V0[:, None, None] * F
        for k in range(4):
            idx = self.tets[:, k]
            for i in range(3):
                for j in range(3):
                    Fv[:, i, j] += np.bincount(idx, weights=wF[:, i, j],
                                               minlength=self.n)
            w += np.bincount(idx, weights=self.V0, minlength=self.n)
        w = np.maximum(w, 1e-12)
        Fv /= w[:, None, None]
        return Fv

    def _signed_distances(self, t: float):
        s, _, p, _ = self._schedule(t)
        # grip box inner face tracks the stretched cube face; center x:
        gx = self.face0 * s + (self.grip_x0 - self.face0) - p
        boxes = [
            (np.array([0.0, 0.0, -0.04]), np.array([2.3, 1.7, 0.04])),
            (np.array([-gx, 0.0, self.grip_z]), self.grip_half),
            (np.array([gx, 0.0, self.grip_z]), self.grip_half),
        ]
        sd = np.full(self.n, np.inf)
        for c, h in boxes:
            q = np.abs(self.x - c[None, :]) - h[None, :]
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(np.max(q, axis=1), 0.0)
            sd = np.minimum(sd, outside + inside)
        return sd

    def _frame(self, time: float) -> StateFrame:
        sd = self._signed_distances(time)
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._vertex_F(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.ext_force.copy(),
                "contact_signed_distance": sd,
            },
            scalars={"contact_length_scale": self.contact_len},
        )
        frame.validate()
        return frame

    # --------------------------------------------------------- public API
    def advance_to(self, time: float):
        time = float(time)
        while self.t < time - 1e-12:
            dt = min(self.dt_sub, time - self.t)
            self._step(dt)
        return self._frame(time)

    def diagnostics(self) -> dict[str, object]:
        F = self._tet_F()
        c1, c2, c3 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        J = np.einsum("ti,ti->t", c1, np.cross(c2, c3))
        ke = 0.5 * float(np.sum(self.mass[:, None] * self.vel ** 2))
        return {
            "n_vertices": int(self.n),
            "n_tets": int(len(self.tets)),
            "substeps": int(self.steps_done),
            "substep_dt": float(self.dt_sub),
            "kinetic_energy": ke,
            "min_detF": float(J.min()),
            "max_detF": float(J.max()),
            "time": float(self.t),
        }
