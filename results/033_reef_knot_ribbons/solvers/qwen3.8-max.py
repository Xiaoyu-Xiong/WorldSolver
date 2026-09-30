"""Physically-based solver for scene 033_reef_knot_ribbons.

Two intertwined ribbons (29x5 vertex grids each, 290 vertices total) are pulled
at their scripted end attachments into a tight reef knot.  The dynamics are
solved with a position-based (XPBD-style) method:

* in-plane stretch constraints on the triangulated mesh edges (stiff),
* a *hard* inequality strain cap: no edge may exceed ``strain_upper`` x rest
  length (the paper's strict strain limit, 1.134),
* weak second-neighbour bending constraints giving the ribbons flexural rigidity,
* point-to-segment contact projection between all non-neighbouring feature
  pairs (self- and inter-ribbon) with a minimum separation of two thicknesses,
* position-space Coulomb friction (mu = 0.02) applied at active contacts,
* kinematically scripted end pulling at ``pull_speed`` along each ribbon axis.

Gravity is zero per the scene configuration.  All required raw trajectory
outputs (deformation gradient, mass quantity, contact signed distance,
cumulative mass source, contact length scale) are exported every frame.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        rc = cfg.get("ribbon", {}) if isinstance(cfg, dict) else {}
        self.thickness = float(rc.get("thickness", 0.006))
        self.strain_upper = float(rc.get("strain_upper", 1.134))
        self.mu = float(rc.get("friction", 0.02))
        self.pull_speed = float(cfg.get("pull_speed", 0.32))
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), float)
        self.seed = int(seed)

        root = Path(__file__).resolve().parents[1] / "assets"
        topo = json.loads((root / "ribbon_topology.json").read_text())
        init = json.loads((root / "initial_state.json").read_text())

        self.X0 = np.asarray(init["position"], float)
        self.N = len(self.X0)
        self.mass = np.asarray(init["mass"], float)
        self.body_id = np.asarray(init["body_id"], int)
        self.uv = np.asarray(init["attributes"]["uv"], float)

        nu, nv = 29, 5
        faces_local = np.asarray(topo["faces"], int)          # (224,3)
        self.faces = np.concatenate([faces_local, faces_local + nu * nv])

        # ---------------- edges (unique, includes triangle diagonals) -------
        eset = set()
        for f in self.faces:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                eset.add((int(min(a, b)), int(max(a, b))))
        self.edges = np.array(sorted(eset), int)              # (E,2)
        self.ea, self.eb = self.edges[:, 0], self.edges[:, 1]
        d0 = self.X0[self.eb] - self.X0[self.ea]
        self.L0 = np.linalg.norm(d0, axis=1)
        self.Lcap = self.strain_upper * self.L0

        # ---------------- bending pairs (second neighbours) ----------------
        bp = []
        for body in range(2):
            base = body * nu * nv
            for j in range(nv):
                for i in range(nu):
                    v = base + j * nu + i
                    if i + 2 < nu:
                        bp.append((v, base + j * nu + i + 2))
                    if j + 2 < nv:
                        bp.append((v, base + (j + 2) * nu + i))
        self.bends = np.array(bp, int)
        self.ba, self.bb = self.bends[:, 0], self.bends[:, 1]
        db = self.X0[self.bb] - self.X0[self.ba]
        self.Lb = np.linalg.norm(db, axis=1)

        # ---------------- kinematic pulled ends -----------------------------
        # Local vertex id -> grid (i along length, j across width).
        li = np.arange(self.N) % (nu * nv)
        gi, gj = li % nu, li // nu
        self.pinned = np.zeros(self.N, bool)
        self.pin_dir = np.zeros((self.N, 3))
        axes = {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 1.0, 0.0])}
        for body in range(2):
            m = self.body_id == body
            for end, sgn in ((0, -1.0), (28, 1.0)):
                sel = m & (gi == end)
                self.pinned[sel] = True
                self.pin_dir[sel] = sgn * axes[body]

        # ---------------- inverse masses ------------------------------------
        self.w = 1.0 / np.maximum(self.mass, 1e-12)
        self.w[self.pinned] = 0.0

        # ---------------- contact candidate mask -----------------------------
        # vertex v vs edge e allowed if different ribbons, or same ribbon but
        # topologically far (grid Chebyshev distance > 2 from both endpoints).
        vi, vj = gi[:, None], gj[:, None]
        ei = np.minimum(gi[self.ea], gi[self.eb])[None, :]
        ei2 = np.maximum(gi[self.ea], gi[self.eb])[None, :]
        ej = np.minimum(gj[self.ea], gj[self.eb])[None, :]
        ej2 = np.maximum(gj[self.ea], gj[self.eb])[None, :]
        cheb = np.maximum(np.maximum(np.abs(vi - ei), np.abs(vi - ei2)),
                          np.maximum(np.abs(vj - ej), np.abs(vj - ej2)))
        same_body = self.body_id[:, None] == np.maximum(self.body_id[self.ea],
                                                        self.body_id[self.eb])[None, :]
        same_body = same_body & (self.body_id[:, None] ==
                                 np.minimum(self.body_id[self.ea],
                                            self.body_id[self.eb])[None, :])
        self.allowed = (~same_body) | (cheb > 2)

        # ---------------- deformation gradient rest frames -------------------
        f0 = self.X0[self.faces[:, 0]]
        e1 = self.X0[self.faces[:, 1]] - f0
        e2 = self.X0[self.faces[:, 2]] - f0
        n0 = np.cross(e1, e2)
        n0 /= np.maximum(np.linalg.norm(n0, axis=1, keepdims=True), 1e-15)
        self.face_n0 = n0
        h = self.thickness
        Dm = np.stack((e1, e2, n0 * h), axis=2)               # (F,3,3) columns
        self.Dm_inv = np.linalg.inv(Dm)

        # ---------------- state ----------------------------------------------
        self.eps_contact = 2.0 * self.thickness
        self.x = self.X0.copy()
        self.v = np.zeros((self.N, 3))
        self.t = 0.0
        self.frame_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.substeps = 16
        self.dt_sub = self.frame_dt / self.substeps
        self.ramp_tau = 0.05
        self._diag = {}

    # ------------------------------------------------------------- kinematics
    def _pull_disp(self, t: float) -> float:
        """Smoothly ramped scripted pull displacement."""
        tau = self.ramp_tau
        return self.pull_speed * (t - tau * (1.0 - np.exp(-t / tau)))

    def _pull_vel(self, t: float) -> float:
        return self.pull_speed * (1.0 - np.exp(-t / self.ramp_tau))

    def _pin_targets(self, t: float):
        return self.X0 + self.pin_dir * self._pull_disp(t)

    # ------------------------------------------------------- constraint passes
    def _project_constraints(self, x: np.ndarray, dt: float) -> None:
        dt2 = dt * dt
        alpha_s = 1e-5 / dt2      # stiff stretch (k ~ 1e5 N/m)
        alpha_b = 1e-2 / dt2      # soft bending
        dx = np.zeros_like(x)
        cnt = np.zeros(self.N)

        # stretch equality constraints
        d = x[self.eb] - x[self.ea]
        l = np.linalg.norm(d, axis=1)
        n = d / np.maximum(l, 1e-15)[:, None]
        wa, wb = self.w[self.ea], self.w[self.eb]
        C = l - self.L0
        dlam = -C / (wa + wb + alpha_s)
        ca = (-wa * dlam)[:, None] * n
        cb = (wb * dlam)[:, None] * n
        np.add.at(dx, self.ea, ca)
        np.add.at(dx, self.eb, cb)
        np.add.at(cnt, self.ea, 1.0)
        np.add.at(cnt, self.eb, 1.0)

        # hard strain-cap inequality: l <= strain_upper * L0
        over = l > self.Lcap
        if np.any(over):
            idx = np.nonzero(over)[0]
            Cc = l[idx] - self.Lcap[idx]
            dl = -Cc / np.maximum(wa[idx] + wb[idx], 1e-15)
            ca = (-wa[idx] * dl)[:, None] * n[idx]
            cb = (wb[idx] * dl)[:, None] * n[idx]
            np.add.at(dx, self.ea[idx], ca)
            np.add.at(dx, self.eb[idx], cb)
            np.add.at(cnt, self.ea[idx], 1.0)
            np.add.at(cnt, self.eb[idx], 1.0)

        # weak bending constraints
        d = x[self.bb] - x[self.ba]
        l = np.linalg.norm(d, axis=1)
        n = d / np.maximum(l, 1e-15)[:, None]
        wa, wb = self.w[self.ba], self.w[self.bb]
        C = l - self.Lb
        dlam = -C / (wa + wb + alpha_b)
        np.add.at(dx, self.ba, (-wa * dlam)[:, None] * n)
        np.add.at(dx, self.bb, (wb * dlam)[:, None] * n)
        np.add.at(cnt, self.ba, 1.0)
        np.add.at(cnt, self.bb, 1.0)

        x += dx / np.maximum(cnt, 1.0)[:, None]

    # ---------------------------------------------------------- contact pass
    def _contact_info(self, x: np.ndarray):
        """Point-to-segment distances for all allowed vertex/edge pairs."""
        A = x[self.ea]                       # (E,3)
        E = x[self.eb] - A                   # (E,3)
        d0 = x[:, None, :] - A[None, :, :]   # (N,E,3)
        ee = np.maximum((E * E).sum(1)[None, :], 1e-24)
        t = np.clip((d0 * E[None, :, :]).sum(2) / ee, 0.0, 1.0)
        r = d0 - t[:, :, None] * E[None, :, :]
        dist = np.linalg.norm(r, axis=2)
        dist = np.where(self.allowed, dist, np.inf)
        return dist, t, r

    def _solve_contacts(self, x: np.ndarray, x_prev: np.ndarray) -> None:
        dist, t, r = self._contact_info(x)
        jmin = np.argmin(dist, axis=1)
        dmin = dist[np.arange(self.N), jmin]
        act = np.nonzero(np.isfinite(dmin) & (dmin < self.eps_contact))[0]
        self._dmin = np.where(np.isfinite(dmin), dmin, self.eps_contact)
        self._tmin = t[np.arange(self.N), jmin]
        self._jmin = jmin
        if len(act) == 0:
            return
        j = jmin[act]
        ia, ib = self.ea[j], self.eb[j]
        tt = t[act, j]
        nvec = r[act, j, :]
        dn = dmin[act]
        nvec = nvec / np.maximum(dn, 1e-15)[:, None]
        delta = self.eps_contact - dn

        wp = self.w[act]
        wa = self.w[ia] * (1.0 - tt) ** 2
        wb = self.w[ib] * tt ** 2
        S = np.maximum(wp + wa + wb, 1e-15)

        dx = np.zeros_like(x)
        cnt = np.zeros(self.N)
        # normal push-out
        np.add.at(dx, act, (wp * delta / S)[:, None] * nvec)
        np.add.at(dx, ia, (-(wa * delta) / S)[:, None] * nvec)
        np.add.at(dx, ib, (-(wb * delta) / S)[:, None] * nvec)
        np.add.at(cnt, act, 1.0)
        np.add.at(cnt, ia, 1.0)
        np.add.at(cnt, ib, 1.0)

        # position-space Coulomb friction opposing previous tangential slip
        if self.mu > 0.0:
            A0 = x_prev[self.ea[j]]
            E0 = x_prev[self.eb[j]] - A0
            c0 = A0 + tt[:, None] * E0
            s = x_prev[act] - c0
            sn = (s * nvec).sum(1, keepdims=True) * nvec
            st = s - sn
            mag = np.linalg.norm(st, axis=1)
            ok = mag > 1e-12
            if np.any(ok):
                u = st[ok] / mag[ok, None]
                ft = np.minimum(self.mu * delta[ok], mag[ok])
                np.add.at(dx, act[ok], (-(wp[ok] * ft) / S[ok])[:, None] * u)
                np.add.at(dx, ia[ok], ((wa[ok] * ft) / S[ok])[:, None] * u)
                np.add.at(dx, ib[ok], ((wb[ok] * ft) / S[ok])[:, None] * u)

        x += dx / np.maximum(cnt, 1.0)[:, None]

    # --------------------------------------------------------------- stepping
    def _step(self, dt: float, t_new: float) -> None:
        x_prev = self.x.copy()
        x = self.x + self.v * dt + 0.5 * self.gravity * dt * dt
        target = self._pin_targets(t_new)
        x[self.pinned] = target[self.pinned]
        for _ in range(3):
            self._project_constraints(x, dt)
            x[self.pinned] = target[self.pinned]
        for _ in range(2):
            self._solve_contacts(x, x_prev)
            x[self.pinned] = target[self.pinned]
        self.v = (x - x_prev) / dt
        self.v[self.pinned] = self.pin_dir[self.pinned] * self._pull_vel(t_new)
        self.v *= 0.999
        self.x = x
        self.t = t_new

    # -------------------------------------------------- deformation gradient
    def _deformation_gradients(self) -> np.ndarray:
        f0 = self.x[self.faces[:, 0]]
        e1 = self.x[self.faces[:, 1]] - f0
        e2 = self.x[self.faces[:, 2]] - f0
        n = np.cross(e1, e2)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-15), self.face_n0)
        Ds = np.stack((e1, e2, n * self.thickness), axis=2)
        Ff = Ds @ self.Dm_inv                          # (F,3,3)
        Fv = np.zeros((self.N, 3, 3))
        cnt = np.zeros(self.N)
        for k in range(3):
            np.add.at(Fv, self.faces[:, k], Ff)
            np.add.at(cnt, self.faces[:, k], 1.0)
        return Fv / np.maximum(cnt, 1.0)[:, None, None]

    # ---------------------------------------------------------------- frames
    def _make_frame(self, time: float) -> StateFrame:
        dist, _, _ = self._contact_info(self.x)
        dmin = dist.min(1)
        csd = np.where(np.isfinite(dmin), dmin, self.eps_contact) - self.eps_contact
        F = self._deformation_gradients()
        frame = StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "uv": self.uv.copy(),
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F,
            },
            fields={
                "contact_signed_distance": csd,
                "cumulative_mass_source": np.zeros(2),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        time = float(time)
        if time < self.t:
            time = self.t
        dt = self.dt_sub
        while self.t < time - 1e-12:
            h = min(dt, time - self.t)
            self._step(h, self.t + h)
        # diagnostics bookkeeping
        d = self.x[self.eb] - self.x[self.ea]
        l = np.linalg.norm(d, axis=1)
        strain = l / self.L0
        self._diag = {
            "time": self.t,
            "max_edge_strain": float(strain.max()),
            "strain_violation": float(max(strain.max() - self.strain_upper, 0.0)),
            "min_contact_gap": float(self._dmin.min() - self.eps_contact)
            if hasattr(self, "_dmin") else 0.0,
            "n_active_contacts": int(np.sum(self._dmin < self.eps_contact))
            if hasattr(self, "_dmin") else 0,
        }
        return self._make_frame(self.t)

    def diagnostics(self) -> dict:
        return dict(self._diag)
