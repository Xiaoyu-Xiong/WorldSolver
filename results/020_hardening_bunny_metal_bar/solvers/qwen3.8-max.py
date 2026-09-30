"""Viscoplastic bunny falling onto a metal bar (work-hardening soft body).

Method: mass-spring discretization of the subject triangle mesh.
- Axial springs along mesh edges carry an elastic-plastic constitutive law:
  strain beyond a yield threshold flows plastically (permanent deformation),
  and accumulated plastic strain hardens the spring (higher yield and higher
  stiffness), reproducing work hardening -> localized elasticity -> jiggle.
- Weak bending springs across opposite vertices of adjacent faces keep the
  shell from inverting/collapsing.
- Contacts: positional projection + velocity reflection (restitution) with
  Coulomb-style friction against the ground plane and the finite cylinder bar.
- Per-vertex deformation gradient F from a regularized least-squares fit of
  the current neighborhood against the rest neighborhood.
- Semi-implicit Euler with small substeps between output samples.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        asset = __import__("pathlib").Path(__file__).resolve().parent.parent / cfg["subject_asset"]
        with np.load(asset) as data:
            V = np.asarray(data["vertices"], dtype=float)
            Fc = np.asarray(data["faces"], dtype=np.int64)
        self.X0 = V.copy()
        self.N = len(V)

        self.x = V.copy()
        v0 = np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float)
        self.v = np.tile(v0, (self.N, 1))
        self.m = np.full(self.N, float(cfg["subject_mass"]) / self.N)
        self.g = np.asarray(cfg["gravity"], dtype=float)
        self.t = 0.0

        # ---- edges (axial springs)
        e = np.vstack([Fc[:, :2], Fc[:, 1:], Fc[:, ::2]])
        e = np.sort(e, axis=1)
        e = np.unique(e, axis=0)
        self.E = e
        d = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
        self.L0 = d.copy()
        self.Lrest = d.copy()
        self.acc_pl = np.zeros(len(e))          # accumulated plastic strain

        # ---- bending springs: opposite vertices across each interior edge
        fe = np.vstack([
            np.sort(Fc[:, [1, 2]], axis=1),
            np.sort(Fc[:, [2, 0]], axis=1),
            np.sort(Fc[:, [0, 1]], axis=1),
        ])
        key = fe[:, 0] * (self.N + 1) + fe[:, 1]
        order = np.argsort(key)
        ks, ksorted = key[order], fe[order]
        dup = np.nonzero(ks[:-1] == ks[1:])[0]
        bpair = np.stack([ksorted[dup, 2], ksorted[dup + 1, 2]], axis=1)
        bpair = np.sort(bpair, axis=1)
        bpair = np.unique(bpair, axis=0)
        bpair = bpair[bpair[:, 0] != bpair[:, 1]]
        self.B = bpair
        self.B0 = np.linalg.norm(V[bpair[:, 0]] - V[bpair[:, 1]], axis=1)

        # ---- material parameters
        stiffness = float(cfg.get("stiffness", 18.0))
        k_scale = 150.0 / max(float(self.L0.mean()), 1e-6) * float(self.L0.mean())  # = 150
        self.k_ax = stiffness * 150.0
        self.k_bend = 0.06 * self.k_ax
        self.hard = float(cfg.get("hardening_rate", 1.8))
        self.eps_y0 = 0.05
        self.flow = 0.5
        damp = float(cfg.get("damping", 1.4))
        self.cd = 0.12 * damp * np.sqrt(self.k_ax * self.m.mean())
        self.drag = 0.35 * damp
        self.rest = float(cfg.get("restitution", 0.28))
        self.mu = float(cfg.get("friction", 0.55))
        self.kcap = 3.5  # hardening stiffness multiplier cap

        # ---- obstacle geometry
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.bar_c = np.array([0.0, 0.0, float(cfg.get("bar_z", 0.66))])
        self.bar_r = float(cfg.get("bar_radius", 0.085))
        self.bar_hy = 1.2  # half height (axis along y)

        # ---- deformation-gradient neighborhoods (padded adjacency)
        deg = np.bincount(e.ravel(), minlength=self.N)
        self.Dmax = int(max(deg.max(), 1))
        self.nbr = np.full((self.N, self.Dmax), 0, dtype=np.int64)
        self.wgt = np.zeros((self.N, self.Dmax))
        cnt = np.zeros(self.N, dtype=np.int64)
        dX0 = V[e[:, 0]] - V[e[:, 1]]
        dL = np.maximum(np.linalg.norm(dX0, axis=1), 1e-9)
        w = 1.0 / dL
        for (a, b, ww) in zip(e[:, 0], e[:, 1], w):
            for (i, j) in ((a, b), (b, a)):
                s = cnt[i]
                if s < self.Dmax:
                    self.nbr[i, s] = j
                    self.wgt[i, s] = ww
                    cnt[i] += 1
        self.dx0 = (V[self.nbr] - V[:, None, :]) * self.wgt[:, :, None]
        A = np.einsum("ndi,ndj->nij", self.dx0, self.dx0)
        A += 1e-3 * np.trace(A).mean() * np.eye(3)[None]
        self.Ainv = np.linalg.inv(A)

        self.cls = float(cfg.get("contact_length_scale", 0.04))
        self._contact_forces = np.zeros((2, 3))

    # -------------------------------------------------------------- contacts
    def _signed_distances(self, x):
        dg = x[:, 2] - self.ground_z
        rx = x[:, 0]
        rz = x[:, 2] - self.bar_c[2]
        r = np.sqrt(rx * rx + rz * rz)
        in_span = np.abs(x[:, 1]) <= self.bar_hy
        db = np.where(in_span, r - self.bar_r, 1e3)
        return dg, db, r, in_span

    def _project_contacts(self, dt):
        x, v = self.x, self.v
        imp = np.zeros((2, 3))
        # ground
        pen = self.ground_z - x[:, 2]
        act = pen > 0
        if act.any():
            x[act, 2] = self.ground_z
            vn = v[act, 2]
            neg = vn < 0
            dvn = np.where(neg, -(1.0 + self.rest) * vn, 0.0)
            v[act, 2] = np.where(neg, -self.rest * vn, vn)
            vt = v[act][:, :2]
            sp = np.linalg.norm(vt, axis=1)
            sc = np.maximum(0.0, 1.0 - self.mu * dvn / np.maximum(sp, 1e-9))
            v[np.nonzero(act)[0], :2] = vt * sc[:, None]
            imp[0, 2] += np.sum(self.m[act] * dvn) / max(dt, 1e-12)
        # bar (cylinder, axis along y)
        dg, db, r, in_span = self._signed_distances(x)
        act = (db < 0) & (r > 1e-9)
        if act.any():
            idx = np.nonzero(act)[0]
            n = np.zeros((len(idx), 3))
            n[:, 0] = x[idx, 0] / r[idx]
            n[:, 2] = (x[idx, 2] - self.bar_c[2]) / r[idx]
            x[idx, 0] = self.bar_c[0] + n[:, 0] * self.bar_r
            x[idx, 2] = self.bar_c[2] + n[:, 2] * self.bar_r
            vn = np.einsum("ij,ij->i", v[idx], n)
            neg = vn < 0
            dvn = np.where(neg, -(1.0 + self.rest) * vn, 0.0)
            vn_new = np.where(neg, -self.rest * vn, vn)
            vt = v[idx] - vn[:, None] * n
            sp = np.linalg.norm(vt, axis=1)
            sc = np.maximum(0.0, 1.0 - self.mu * dvn / np.maximum(sp, 1e-9))
            v[idx] = vt * sc[:, None] + vn_new[:, None] * n
            imp[1] += (self.m[idx, None] * (dvn[:, None] * n)).sum(0) / max(dt, 1e-12)
        self._contact_forces += imp

    # ------------------------------------------------------------------ step
    def _substep(self, dt):
        x, v = self.x, self.v
        i, j = self.E[:, 0], self.E[:, 1]
        d = x[j] - x[i]
        L = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        n = d / L[:, None]

        # --- plasticity with work hardening (axial springs)
        delta = L - self.Lrest
        eps_el = np.abs(delta) / self.L0
        eps_y = self.eps_y0 + self.hard * self.acc_pl
        over = eps_el > eps_y
        if over.any():
            dpl = self.flow * (np.abs(delta[over]) - eps_y[over] * self.L0[over])
            self.Lrest[over] += np.sign(delta[over]) * dpl
            self.acc_pl[over] += dpl / self.L0[over]

        ks = np.minimum(1.0 + self.hard * self.acc_pl, self.kcap) * self.k_ax
        relv = v[j] - v[i]
        vn = np.einsum("ij,ij->i", relv, n)
        fmag = ks * (L - self.Lrest) - self.cd * vn
        f = fmag[:, None] * n
        np.add.at(v, i, (f * dt) / self.m[i, None])
        np.add.at(v, j, (-f * dt) / self.m[j, None])

        # --- bending springs (elastic, weak)
        bi, bj = self.B[:, 0], self.B[:, 1]
        db = x[bj] - x[bi]
        Lb = np.maximum(np.linalg.norm(db, axis=1), 1e-9)
        nb = db / Lb[:, None]
        rv = v[bj] - v[bi]
        vnb = np.einsum("ij,ij->i", rv, nb)
        fb = (self.k_bend * (Lb - self.B0) - 0.5 * self.cd * vnb)[:, None] * nb
        np.add.at(v, bi, (fb * dt) / self.m[bi, None])
        np.add.at(v, bj, (-fb * dt) / self.m[bj, None])

        # --- gravity + drag, integrate
        v += (self.g[None, :] - self.drag * v) * dt
        x += v * dt
        self._project_contacts(dt)

    # ---------------------------------------------------------------- frames
    def _deformation_gradient(self):
        dx = (self.x[self.nbr] - self.x[:, None, :]) * self.wgt[:, :, None]
        Bmat = np.einsum("ndi,ndj->nij", dx, self.dx0)
        return np.einsum("nij,njk->nik", Bmat, self.Ainv)

    def _make_frame(self, t):
        dg, db, r, in_span = self._signed_distances(self.x)
        gap = np.minimum(dg, db)
        F = self._deformation_gradient()
        frame = StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=np.zeros(self.N, dtype=np.int64),
            attributes={
                "mass_quantity": self.m.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.m[:, None] * self.g[None, :],
                "contact_signed_distance": gap,
                "interaction_force": self._contact_forces.copy(),
            },
            scalars={"contact_length_scale": self.cls},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        time = float(time)
        span = time - self.t
        dt_max = 4.0e-4
        if span > 1e-12:
            n = int(np.ceil(span / dt_max))
            dt = span / n
            self._contact_forces = np.zeros((2, 3))
            for _ in range(n):
                self._substep(dt)
            self._contact_forces /= max(n, 1)  # mean over substeps
            self.t = time
        return self._make_frame(time)

    def diagnostics(self) -> dict:
        return {
            "n_vertices": self.N,
            "n_springs": len(self.E),
            "mean_plastic_strain": float(self.acc_pl.mean()),
            "max_plastic_strain": float(self.acc_pl.max()),
        }
