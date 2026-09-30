"""Physically-based solver for the bread dough drop scene.

The dough slab (triangle mesh, 702 vertices) is simulated as a set of
overlapping surface clusters using shape matching with plastic flow
(Mueller et al. style meshless deformation), which is a standard
animation-oriented approximation of large viscoplastic flow.

Physics:
  - gravity (external force),
  - shape-matching elastic forces toward a slowly creeping (plastic)
    rest configuration -> impact deformation, partial recovery, spreading,
  - velocity damping,
  - penalty contact with the ground plane (normal spring/damper with
    restitution clamp + Coulomb friction).

All required StateFrame fields are exported.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ----------------------------- setup -----------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        v = np.asarray(scene.vertices, dtype=float)
        # faces come from the same asset the scene loaded
        import pathlib
        with np.load(pathlib.Path(__file__).resolve().parent.parent / cfg["subject_asset"]) as data:
            f = np.asarray(data["faces"], dtype=np.int64)

        self.n = len(v)
        self.faces = f
        self.nf = len(f)

        self.g = np.asarray(cfg["gravity"], dtype=float)
        self.ground_z = float(cfg.get("ground_z", 0.0))
        total_mass = float(cfg["subject_mass"])
        self.mass = np.full(self.n, total_mass / self.n)
        self.inv_m = 1.0 / self.mass

        self.stiff = float(cfg.get("stiffness", 5.0)) * 450.0      # shape-match rate (1/s^2)
        self.damp = float(cfg.get("damping", 3.4))                  # velocity damping (1/s)
        self.plasticity = float(cfg.get("plasticity", 0.86))
        self.creep = 3.0 * self.plasticity                          # plastic flow rate (1/s)
        self.restitution = float(cfg.get("restitution", 0.02))
        self.friction = float(cfg.get("friction", 0.82))
        self.cls = float(cfg.get("contact_length_scale", 0.04))

        # contact penalty parameters
        self.k_contact = 6.0e3                                      # N/m
        self.c_contact = 150.0                                      # N/(m/s)

        # cluster (per-face) rest configuration
        self.cl_idx = f.ravel()                                     # (3*nf,)
        self.cl_of_v = np.repeat(np.arange(self.nf), 3)
        vrest = v[f].reshape(-1, 3)
        self.r_c = vrest.reshape(self.nf, 3, 3).mean(axis=1).copy()  # rest centroids
        self.y = vrest - np.repeat(self.r_c, 3, axis=0)              # rest offsets (3nf,3)

        self.x = v.copy()
        self.vel = np.tile(np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]),
                                      dtype=float), (self.n, 1))
        self.F = np.tile(np.eye(3), (self.n, 1, 1))
        self.t = 0.0
        self._target_dt = 5.0e-4
        self.gap = np.maximum(self.x[:, 2] - self.ground_z, 0.0)

    # ----------------------------- stepping -----------------------------
    def _shape_match(self):
        """Compute per-cluster rotation, deformation gradient and elastic force."""
        nv, nf = self.n, self.nf
        idx = self.cl_idx
        xv = self.x[idx]                                             # (3nf,3)
        J = xv.reshape(nf, 3, 3).mean(axis=1)                        # deformed centroids
        rel = xv - np.repeat(J, 3, axis=0)                           # deformed offsets
        # covariance A_q = sum m_i rel_i y_i^T  (uniform masses cancel)
        A = np.einsum('qai,qaj->qij', rel.reshape(nf, 3, 3), self.y.reshape(nf, 3, 3))
        U, S, Vt = np.linalg.svd(A)
        d = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
        Dm = np.tile(np.eye(3), (nf, 1, 1))
        Dm[:, 2, 2] = d
        R = U @ Dm @ Vt                                              # rotations (nf,3,3)

        # deformation gradient per cluster: A_q (y^T y)^-1 in rest frame
        G = np.einsum('qai,qaj->qij', self.y.reshape(nf, 3, 3), self.y.reshape(nf, 3, 3))
        G[:, np.arange(3), np.arange(3)] += 1e-12
        Aq = np.einsum('qai,qbi->qab', rel.reshape(nf, 3, 3), self.y.reshape(nf, 3, 3))
        Fq = Aq @ np.linalg.inv(G)                                   # (nf,3,3)

        # goal positions and elastic force (mass-normalized spring)
        goal_rel = np.einsum('qij,qaj->qai', R, self.y.reshape(nf, 3, 3)).reshape(-1, 3)
        goal = J.repeat(3, axis=0) * 0 + np.repeat(J, 3, axis=0) + goal_rel
        acc = self.stiff * (goal - xv)                               # acceleration (m/s^2)
        f_el = np.zeros((nv, 3))
        np.add.at(f_el, idx, acc * self.mass[idx, None])

        # weak anti-collapse (volume preservation penalty)
        detF = np.linalg.det(Fq)
        pen = np.maximum(0.5 - detF, 0.0)
        f_vol = np.zeros((nv, 3))
        np.add.at(f_vol, idx,
                  (pen * 4000.0)[:, None] * self.mass[idx, None] * rel)

        # plastic creep of the rest configuration toward deformed state
        alpha = min(1.0, self.creep * self._dt_used)
        y_new = (1.0 - alpha) * self.y.reshape(nf, 3, 3) + alpha * np.einsum(
            'qij,qaj->qai', R, self.y.reshape(nf, 3, 3))
        self.y = y_new.reshape(-1, 3)
        self.r_c = (1.0 - alpha) * self.r_c + alpha * J

        # per-vertex deformation gradient (cluster average)
        F = np.zeros((nv, 3, 3))
        np.add.at(F, idx, Fq[self.cl_of_v])
        cnt = np.zeros(nv)
        np.add.at(cnt, idx, 1.0)
        self.F = F / np.maximum(cnt, 1.0)[:, None, None]
        return f_el + f_vol

    def _contact(self):
        pen = self.ground_z - self.x[:, 2]
        act = pen > 0.0
        f_c = np.zeros((self.n, 3))
        if np.any(act):
            pn = pen[act]
            vn = self.vel[act, 2]
            fn = self.k_contact * pn - self.c_contact * np.minimum(vn, 0.0)
            fn = np.maximum(fn, 0.0)
            f_c[act, 2] = fn
            # restitution clamp
            if np.any(vn < 0.0):
                v_after = vn + (fn * self._dt_used) * self.inv_m[act]
                lim = -self.restitution * vn
                over = (v_after > lim) & (vn < 0.0)
                if np.any(over):
                    f_c[np.where(act)[0][over], 2] = \
                        (lim[over] - vn[over]) * self.mass[act][over] / self._dt_used
            # Coulomb friction on tangential velocity
            vt = self.vel[act, :2]
            sp = np.linalg.norm(vt, axis=1)
            mu_fn = self.friction * f_c[act, 2]
            drag = np.minimum(mu_fn, sp * self.mass[act] / self._dt_used)
            safe = sp > 1e-9
            f_c[np.where(act)[0][safe], :2] = \
                -(vt[safe] / sp[safe, None]) * drag[safe, None]
        self.gap = np.maximum(self.x[:, 2] - self.ground_z, -pen)
        return f_c

    def _step(self, h):
        self._dt_used = h
        f_int = self._shape_match()
        f_c = self._contact()
        f_ext = self.mass[:, None] * self.g[None, :]
        acc = (f_int + f_c + f_ext) * self.inv_m[:, None] - self.damp * self.vel
        self.vel = self.vel + acc * h
        self.x = self.x + self.vel * h
        # hard positional clamp (safety, penalty should handle it)
        below = self.x[:, 2] < self.ground_z
        if np.any(below):
            self.x[below, 2] = self.ground_z
            self.vel[below, 2] = np.maximum(self.vel[below, 2], 0.0)
        self.t += h

    def advance_to(self, time: float):
        time = float(time)
        if time > self.t + 1e-12:
            total = time - self.t
            nsub = int(np.clip(np.ceil(total / self._target_dt), 1, 40000))
            h = total / nsub
            for _ in range(nsub):
                self._step(h)
        return self._frame(time)

    # ----------------------------- output -----------------------------
    def _frame(self, time: float) -> StateFrame:
        x = np.asarray(self.x, dtype=float)
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        vel = np.nan_to_num(np.asarray(self.vel, dtype=float))
        frame = StateFrame(
            time=float(time),
            position=x.copy(),
            velocity=vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": np.nan_to_num(np.asarray(self.F, dtype=float)),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.g[None, :],
                "contact_signed_distance": x[:, 2] - self.ground_z,
            },
            scalars={"contact_length_scale": self.cls},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "n_vertices": self.n,
            "n_faces": self.nf,
            "time": self.t,
            "stiffness": self.stiff,
            "creep": self.creep,
        }
