"""Knitted legwarmer deformation solver (scene 155).

Physically-based yarn-level-inspired cloth simulation:

* The frozen 17x17 knit tube (scaffold topology) is simulated as a
  tension-stiff / compression-soft mass-spring shell.  Knit yarn resists
  extension but buckles almost freely in compression, which reproduces the
  characteristic behaviour of a knitted sleeve: it stretches on the outside
  of a joint bend and bunches (folds) on the compressed inside.
* The articulated leg (two capsules + knee sphere + foot capsule) is a
  kinematic collider set.  The lower limb group rotates about the knee
  pivot, driving the bend.
* Penalty contact with Coulomb friction keeps the sleeve off the body and
  the ground; both sleeve ends are elically coupled to the limb they sit on
  so the sleeve follows the articulation (yarn friction follow).
* Time integration: semi-implicit (symplectic) Euler with fixed deterministic
  substeps (dt = sample_dt / 200), velocity clipping for robustness.

No keyframes, no precomputed trajectories: every frame is the result of the
time-stepping procedure above.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    SUBSTEPS_PER_SAMPLE = 200

    # ---- knit constitutive parameters (per edge category) -------------
    # categories: 0 circumferential (course), 1 axial (wale), 2 shear, 3 bend
    K_TENSION = np.array([28.0, 28.0, 7.0, 0.45])   # N/m  yarn stretch stiffness
    K_COMPRESS = np.array([1.6, 1.6, 0.6, 0.45])    # N/m  buckled-yarn stiffness
    K_DAMP = np.array([0.22, 0.22, 0.05, 0.015])    # N s/m along-edge damping

    # ---- contact parameters ------------------------------------------
    CONTACT_K = 1500.0        # N/m normal penalty stiffness
    CONTACT_MARGIN = 5.0e-4   # m
    CONTACT_CN = 1.0          # N s/m normal damping
    FRICTION_MU = 0.55
    FRICTION_GAMMA = 8.0      # N s/m viscous cap (stability-limited)

    # ---- end coupling (sleeve follows the limbs) ----------------------
    ATTACH_K = 80.0
    ATTACH_D = 1.2

    GLOBAL_DAMP = 0.01        # N s/m
    VMAX = 3.0                # m/s safety clip

    # ---- driven knee bend ---------------------------------------------
    THETA_MAX = -0.95         # rad (negative: shin swings up, knee bends)
    T_BEND0 = 0.8
    T_BEND1 = 5.2

    PIVOT = np.array([0.0, 0.0, 1.1])

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.config = cfg
        self.sample_dt = float(cfg["sample_dt"])
        self.dt = self.sample_dt / self.SUBSTEPS_PER_SAMPLE

        frame0 = scene.initial_state()
        self.P0 = np.array(frame0.position, dtype=float)
        self.n = len(self.P0)
        self.NR, self.NA = (int(v) for v in cfg["cloth"]["grid_shape"])
        self.P = self.P0.copy()
        self.V = np.zeros_like(self.P)
        self.mass = np.array(frame0.mass, dtype=float)
        self.gravity = np.array(cfg["gravity"], dtype=float)
        self.thickness = float(cfg["cloth"]["thickness"])

        self._build_topology()
        self._build_frames()
        self._resolve_initial_overlap()
        self.Prest = self.P.copy()

        self.ring_first = np.arange(self.NA)
        self.ring_last = np.arange((self.NR - 1) * self.NA, self.NR * self.NA)

        self.step_count = 0
        self.sim_time = 0.0

    # ------------------------------------------------------------------
    def _grid_faces(self):
        faces = []
        for i in range(self.NR - 1):
            for j in range(self.NA):
                j2 = (j + 1) % self.NA
                a = i * self.NA + j
                b = i * self.NA + j2
                c = (i + 1) * self.NA + j
                d = (i + 1) * self.NA + j2
                faces.append([a, c, d])
                faces.append([a, d, b])
        return np.asarray(faces, dtype=np.int64)

    def _build_topology(self):
        faces = None
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "assets", "cloth_topology.json")
        try:
            with open(path) as fh:
                topo = json.load(fh)
            f = np.asarray(topo["faces"], dtype=np.int64)
            if f.ndim == 2 and f.shape[1] == 3 and len(topo["vertices"]) == self.n \
                    and f.max() < self.n and f.min() >= 0:
                faces = f
        except Exception:
            faces = None
        if faces is None:
            faces = self._grid_faces()

        ea, eb, cat = [], [], []
        seen = set()
        for tri in faces:
            for u, v in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                a, b = (int(u), int(v)) if u < v else (int(v), int(u))
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                ia, ja = divmod(a, self.NA)
                ib, jb = divmod(b, self.NA)
                if ia == ib:
                    c = 0          # circumferential (course) yarn
                elif ja == jb:
                    c = 1          # axial (wale) yarn
                else:
                    c = 2          # shear / diagonal
                ea.append(a)
                eb.append(b)
                cat.append(c)
        # second-neighbour bending yarns (loop-to-loop coupling)
        for a in range(self.n):
            ia, ja = divmod(a, self.NA)
            if ia + 2 < self.NR:
                ea.append(a)
                eb.append(a + 2 * self.NA)
                cat.append(3)
            ea.append(a)
            eb.append(ia * self.NA + (ja + 2) % self.NA)
            cat.append(3)

        self.ea = np.asarray(ea, dtype=np.int64)
        self.eb = np.asarray(eb, dtype=np.int64)
        ecat = np.asarray(cat, dtype=np.int64)
        self.k_t = self.K_TENSION[ecat]
        self.k_c = self.K_COMPRESS[ecat]
        self.k_d = self.K_DAMP[ecat]
        self.L0 = np.linalg.norm(self.P0[self.eb] - self.P0[self.ea], axis=1)

    def _build_frames(self):
        """Neighbour indices + initial tangent frames for the 2x2 surface
        deformation gradient exported per sample."""
        idx = np.arange(self.n)
        ia = idx // self.NA
        ja = idx % self.NA
        ia0 = np.maximum(ia - 1, 0)
        ia1 = np.minimum(ia + 1, self.NR - 1)
        self.ia_prev = ia0 * self.NA + ja
        self.ia_next = ia1 * self.NA + ja
        self.ja_prev = ia * self.NA + (ja - 1) % self.NA
        self.ja_next = ia * self.NA + (ja + 1) % self.NA
        E1 = self.P0[self.ia_next] - self.P0[self.ia_prev]
        E2 = self.P0[self.ja_next] - self.P0[self.ja_prev]
        self.A0 = self._frame_matrix(E1, E2)
        self.A0_inv = np.linalg.inv(self.A0 + 1e-12 * np.eye(2)[None])

    @staticmethod
    def _frame_matrix(E1, E2):
        l1 = np.maximum(np.linalg.norm(E1, axis=1), 1e-9)
        t1 = E1 / l1[:, None]
        p2 = E2 - np.einsum("ij,ij->i", E2, t1)[:, None] * t1
        l2 = np.maximum(np.linalg.norm(p2, axis=1), 1e-9)
        t2 = p2 / l2[:, None]
        A = np.empty((len(E1), 2, 2))
        A[:, 0, 0] = np.einsum("ij,ij->i", E1, t1)
        A[:, 0, 1] = np.einsum("ij,ij->i", E2, t1)
        A[:, 1, 0] = np.einsum("ij,ij->i", E1, t2)
        A[:, 1, 1] = np.einsum("ij,ij->i", E2, t2)
        return A

    # ------------------------------------------------------------------
    def _rot_y(self, th):
        c, s = math.cos(th), math.sin(th)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    def _bend(self, t):
        """Smooth knee-angle profile theta(t) and angular rate."""
        u = (t - self.T_BEND0) / (self.T_BEND1 - self.T_BEND0)
        uc = min(max(u, 0.0), 1.0)
        s = uc * uc * (3.0 - 2.0 * uc)
        th = self.THETA_MAX * s
        w = 0.0
        if 0.0 < u < 1.0:
            w = self.THETA_MAX * 6.0 * uc * (1.0 - uc) / (self.T_BEND1 - self.T_BEND0)
        return th, w

    def _colliders(self, t):
        th, w = self._bend(t)
        R = self._rot_y(th)
        om = np.array([0.0, w, 0.0])
        xax = np.array([1.0, 0.0, 0.0])
        cols = [
            dict(kind="cap", c=np.array([-0.62, 0.0, 1.1]), ax=xax,
                 r=0.34, h=0.65, om=None, pivot=None),                 # upper limb
            dict(kind="sph", c=np.array([0.0, 0.0, 1.1]), r=0.36,
                 om=None, pivot=None),                                  # knee
        ]
        for c0, r, h in (((0.62, 0.0, 1.1), 0.31, 0.65),               # lower limb
                         ((1.55, -0.1, 0.92), 0.25, 0.36)):            # foot
            c0 = np.asarray(c0, dtype=float)
            cols.append(dict(kind="cap", c=R @ (c0 - self.PIVOT) + self.PIVOT,
                             ax=R @ xax, r=r, h=h, om=om, pivot=self.PIVOT))
        return cols, om, R

    @staticmethod
    def _collider_distance(P, col):
        if col["kind"] == "sph":
            q = P - col["c"]
            dist = np.maximum(np.linalg.norm(q, axis=1), 1e-12)
            return dist - col["r"], q / dist[:, None]
        q = P - col["c"]
        t = np.clip(q @ col["ax"], -col["h"], col["h"])
        diff = q - t[:, None] * col["ax"]
        dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
        return dist - col["r"], diff / dist[:, None]

    def _resolve_initial_overlap(self):
        """Quasi-static projection so the frozen initial tube does not start
        penetrating the (slightly offset) foot capsule."""
        cols, _, _ = self._colliders(0.0)
        for _ in range(400):
            disp = np.zeros_like(self.P)
            worst = 0.0
            for col in cols:
                d, nrm = self._collider_distance(self.P, col)
                pen = np.maximum(-d, 0.0)
                worst = max(worst, float(pen.max()))
                disp += 0.25 * pen[:, None] * nrm
            pen = np.maximum(-self.P[:, 2], 0.0)  # ground plane z = 0
            worst = max(worst, float(pen.max()))
            disp += 0.25 * pen[:, None] * np.array([0.0, 0.0, 1.0])
            self.P += disp
            if worst < 1.0e-4:
                break
        self.V[:] = 0.0

    # ------------------------------------------------------------------
    def _spring_forces(self):
        d = self.P[self.eb] - self.P[self.ea]
        L = np.linalg.norm(d, axis=1)
        Ln = np.maximum(L, 1e-9)
        n = d / Ln[:, None]
        ext = L - self.L0
        k = np.where(ext > 0.0, self.k_t, self.k_c)
        vrel = np.einsum("ij,ij->i", self.V[self.eb] - self.V[self.ea], n)
        f = (k * ext - self.k_d * vrel)[:, None] * n
        out = np.zeros((self.n, 3))
        for c in range(3):
            out[:, c] = (np.bincount(self.ea, weights=f[:, c], minlength=self.n)
                         - np.bincount(self.eb, weights=f[:, c], minlength=self.n))
        return out

    def _contact_forces(self, cols):
        F = np.zeros((self.n, 3))
        margin = self.CONTACT_MARGIN
        for col in cols:
            d, nrm = self._collider_distance(self.P, col)
            if col["om"] is None:
                vsurf = np.zeros_like(self.P)
            else:
                vsurf = np.cross(col["om"], self.P - col["pivot"])
            vrel = self.V - vsurf
            vn = np.einsum("ij,ij->i", vrel, nrm)
            act = d < margin
            pen = np.where(act, margin - d, 0.0)
            Fn = self.CONTACT_K * pen - self.CONTACT_CN * np.minimum(vn, 0.0)
            Fn = np.where(act, Fn, 0.0)
            vt = vrel - vn[:, None] * nrm
            lvt = np.linalg.norm(vt, axis=1)
            ft = np.where(act & (lvt > 1e-9),
                          np.minimum(self.FRICTION_GAMMA * lvt,
                                     self.FRICTION_MU * Fn) / np.maximum(lvt, 1e-12),
                          0.0)
            F += Fn[:, None] * nrm - ft[:, None] * vt
        # ground (top face at z = 0)
        d = self.P[:, 2]
        nrm = np.tile(np.array([0.0, 0.0, 1.0]), (self.n, 1))
        act = d < margin
        pen = np.where(act, margin - d, 0.0)
        vn = self.V[:, 2]
        Fn = np.where(act, self.CONTACT_K * pen - self.CONTACT_CN * np.minimum(vn, 0.0), 0.0)
        vt = np.column_stack([self.V[:, 0], self.V[:, 1],
                              np.zeros(self.n)])
        lvt = np.linalg.norm(vt, axis=1)
        ft = np.where(act & (lvt > 1e-9),
                      np.minimum(self.FRICTION_GAMMA * lvt,
                                 self.FRICTION_MU * Fn) / np.maximum(lvt, 1e-12),
                      0.0)
        F += Fn[:, None] * nrm - ft[:, None] * vt
        return F

    def _step(self, t):
        dt = self.dt
        cols, om, R = self._colliders(t)

        F = self._spring_forces()
        F += self._contact_forces(cols)

        # elastic end coupling: sleeve ends ride on their limbs
        tgt_last = (R @ (self.Prest[self.ring_last] - self.PIVOT).T).T + self.PIVOT
        vtl = np.cross(om, tgt_last - self.PIVOT)
        F[self.ring_last] += (self.ATTACH_K * (tgt_last - self.P[self.ring_last])
                              - self.ATTACH_D * (self.V[self.ring_last] - vtl))
        F[self.ring_first] += (self.ATTACH_K * (self.Prest[self.ring_first]
                                                - self.P[self.ring_first])
                               - self.ATTACH_D * self.V[self.ring_first])

        # gravity + light global damping
        F += self.mass[:, None] * self.gravity[None, :]
        F -= self.GLOBAL_DAMP * self.V

        acc = F / self.mass[:, None]
        V = self.V + dt * acc
        sp = np.linalg.norm(V, axis=1)
        scale = np.minimum(1.0, self.VMAX / np.maximum(sp, 1e-12))
        self.V = V * scale[:, None]
        self.P = self.P + dt * self.V

    # ------------------------------------------------------------------
    def _deformation_gradient(self):
        E1 = self.P[self.ia_next] - self.P[self.ia_prev]
        E2 = self.P[self.ja_next] - self.P[self.ja_prev]
        A = self._frame_matrix(E1, E2)
        Fd = np.clip(A @ self.A0_inv, -20.0, 20.0)
        return Fd

    def _contact_distances(self):
        cols, _, _ = self._colliders(self.sim_time)
        dmin = np.full(self.n, 1e9)
        for col in cols:
            d, _ = self._collider_distance(self.P, col)
            dmin = np.minimum(dmin, d)
        dmin = np.minimum(dmin, self.P[:, 2])  # ground
        near = dmin < 0.2
        if not np.any(near):
            near = np.zeros(self.n, dtype=bool)
            near[int(np.argmin(dmin))] = True
        return dmin[near]

    def _frame(self, time: float) -> StateFrame:
        boundary = np.zeros(self.n, dtype=np.int32)
        boundary[:self.NA] = 1
        boundary[-self.NA:] = 1
        external = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(time),
            self.P.copy(),
            self.V.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": boundary,
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self._contact_distances(),
                "cumulative_mass_source": np.zeros(self.n),
                "external_force": external,
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        t = max(float(time), 0.0)
        target_step = int(round(t / self.dt))
        while self.step_count < target_step:
            self._step(self.step_count * self.dt)
            self.step_count += 1
        self.sim_time = self.step_count * self.dt
        return self._frame(t)

    def diagnostics(self) -> dict:
        th, w = self._bend(self.sim_time)
        return {
            "time": float(self.sim_time),
            "steps": int(self.step_count),
            "substep_dt": float(self.dt),
            "knee_angle_rad": float(th),
            "knee_rate_rad_s": float(w),
            "n_vertices": int(self.n),
            "n_edges": int(len(self.ea)),
        }
