"""Physically-based solver for the ductile hollow ball drop and tear scene.

A hollow elastoplastic shell (particle samples welded onto the ball topology)
falls under gravity onto a rigid floor.  The shell is simulated with an XPBD
distance-constraint network (stretch + bending) featuring:

* rate-independent plastic flow once the elastic strain exceeds a yield
  strain (permanent flattening / buckling, no elastic recovery),
* ductile necking (local compliance softening after plastic accumulation),
* ductile fracture: constraints tear irreversibly once the accumulated
  plastic strain exceeds a ductility limit, leaving a permanently collapsed
  and torn shell.

All required raw trajectory fields are exported per sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_WORKSPACE = Path(__file__).resolve().parents[1]


class BenchmarkSolver:
    # ---------------- material / numerics parameters ----------------
    DT = 1.0 / 240.0            # internal substep
    ITERS = 6                   # XPBD constraint iterations per substep
    E_STRETCH = 5.0e6           # Young-like stretch modulus (Pa)
    E_BEND = 1.0e5              # bending modulus (Pa)
    YIELD = 0.02                # elastic yield strain
    FLOW = 1.0                  # plastic flow rate
    DUCTILITY = 0.40            # accumulated plastic strain at tear
    BEND_DUCTILITY = 0.18       # bending constraints tear earlier
    NECK_START = 0.12           # plastic strain where necking softens
    NECK_SOFTEN = 6.0           # compliance multiplier when necking
    L0_MIN = 0.45               # min rest length ratio (densification cap)
    L0_MAX = 1.9                # max rest length ratio
    FLOOR_R = 0.02              # particle contact radius above z=0 floor
    FRICTION = 0.985            # tangential velocity retention on contact
    DAMP = 0.9992               # mild global damping per substep
    V_MAX = 30.0                # velocity clamp for robustness

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.scene = scene
        self.seed = int(seed)
        self.gravity = np.asarray(cfg["gravity"], float)
        sampling = cfg["sampling"]
        self.spacing = float(sampling["spacing"])

        frame0 = scene.initial_state()
        self.frame0 = frame0
        X = np.asarray(frame0.position, float).copy()
        self.N = len(X)
        self.mass = np.asarray(frame0.mass, float).copy()
        self.body_id = np.asarray(frame0.body_id).copy()
        self.material_label = np.asarray(frame0.attributes["material_label"]).copy()
        self.particle_radius = np.asarray(frame0.attributes["particle_radius"]).copy()

        # ---- topology: particles welded onto unique mesh vertices ----
        topo = json.loads(
            (_WORKSPACE / "assets" / "hollow_ball_topology.json").read_text(encoding="utf-8")
        )
        faces = np.asarray(topo["faces"], dtype=np.int64)
        _, inv, counts = np.unique(
            np.round(X, 6), axis=0, return_inverse=True, return_counts=True
        )
        inv = inv.ravel()
        self.map_u = inv
        M = int(counts.size)
        self.M = M
        # unique particle positions (duplicates welded) and lumped masses
        Xu = np.zeros((M, 3))
        Xu[inv] = X
        mass_u = np.bincount(inv, weights=self.mass, minlength=M)

        self.X0_u = Xu.copy()
        self.pos_u = Xu.copy()
        self.vel_u = np.zeros((M, 3))

        # ---- stretch constraints from mesh edges ----
        fu = inv[faces]
        estretch = set()
        bend_pairs = {}
        edge_faces: dict[tuple[int, int], list[int]] = {}
        for fi, f in enumerate(fu):
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                if a == b:
                    continue
                key = (min(a, b), max(a, b))
                estretch.add(key)
                edge_faces.setdefault(key, []).append(fi)
        self.s_i0 = np.array([e[0] for e in estretch], dtype=np.int64)
        self.s_i1 = np.array([e[1] for e in estretch], dtype=np.int64)

        # ---- bending constraints: opposite vertices across interior edges ----
        ebend = set()
        for key, fl in edge_faces.items():
            if len(fl) != 2:
                continue
            a, b = key
            for f in fl:
                verts = [v for v in (fu[f][0], fu[f][1], fu[f][2]) if v != a and v != b]
                if not verts:
                    continue
                opp = verts[0]
                for end in (a, b):
                    pair = (min(end, opp), max(end, opp))
                    if pair not in estretch:
                        ebend.add(pair)
        ebend = sorted(ebend)
        if ebend:
            self.b_i0 = np.array([e[0] for e in ebend], dtype=np.int64)
            self.b_i1 = np.array([e[1] for e in ebend], dtype=np.int64)
        else:
            self.b_i0 = np.zeros(0, dtype=np.int64)
            self.b_i1 = np.zeros(0, dtype=np.int64)

        self._init_constraints(self.s_i0, self.s_i1, mass_u, is_bend=False)
        self.s_L0, self.s_Linit, self.s_w0, self.s_w1, self.s_alpha0, self.s_pl, \
            self.s_active, self.s_duct = (
                self._L0, self._Linit, self._w0, self._w1, self._alpha0,
                self._pl, self._active, self._duct,
            )
        self._init_constraints(self.b_i0, self.b_i1, mass_u, is_bend=True)
        self.b_L0, self.b_Linit, self.b_w0, self.b_w1, self.b_alpha0, self.b_pl, \
            self.b_active, self.b_duct = (
                self._L0, self._Linit, self._w0, self._w1, self._alpha0,
                self._pl, self._active, self._duct,
            )

        # ---- local least-squares neighbourhoods for deformation gradient ----
        self._init_deformation_gradient()

        self.time = 0.0
        self.torn_count = 0

    # ------------------------- helpers -------------------------
    def _init_constraints(self, i0, i1, mass_u, is_bend):
        L = np.linalg.norm(self.X0_u[i0] - self.X0_u[i1], axis=1)
        L = np.maximum(L, 1e-4)
        w = 1.0 / mass_u
        E = self.E_BEND if is_bend else self.E_STRETCH
        alpha0 = (1.0 / E) / (self.DT ** 2)
        self._L0 = L.copy()
        self._Linit = L.copy()
        self._w0 = w[i0].copy()
        self._w1 = w[i1].copy()
        self._alpha0 = np.full(len(L), alpha0)
        self._pl = np.zeros(len(L))
        self._active = np.ones(len(L), dtype=bool)
        self._duct = np.full(
            len(L), self.BEND_DUCTILITY if is_bend else self.DUCTILITY
        )

    def _init_deformation_gradient(self):
        tree = cKDTree(self.X0_u)
        cutoff = 2.7 * 0.0592  # ~ mean nearest-neighbour spacing on shell
        nbrs = tree.query_ball_point(self.X0_u, cutoff)
        Kmax = max(1, max(len(n) for n in nbrs))
        Kmax = min(Kmax, 64)
        M = self.M
        idx = np.zeros((M, Kmax), dtype=np.int64)
        wgt = np.zeros((M, Kmax))
        for i, n in enumerate(nbrs):
            n = [j for j in n if j != i][:Kmax]
            k = len(n)
            if k == 0:
                n = [i]
                k = 1
            idx[i, :k] = n
            d0 = self.X0_u[n] - self.X0_u[i]
            r2 = (d0 ** 2).sum(axis=1)
            wgt[i, :k] = np.exp(-r2 / (0.13 ** 2)) + 1e-3
        self.F_idx = idx
        dX = self.X0_u[idx] - self.X0_u[:, None, :]          # [M,K,3]
        w3 = wgt[:, :, None, None]
        A = (w3 * dX[:, :, :, None] * dX[:, :, None, :]).sum(axis=1)  # [M,3,3]
        # regularise with the local normal direction so F n = n (shell is 2D)
        evals, evecs = np.linalg.eigh(A)
        nvec = evecs[:, :, 0]                                # smallest eigenvector
        c = np.trace(A, axis1=1, axis2=2)[:, None, None] + 1e-9
        nnT = nvec[:, :, None] * nvec[:, None, :]
        A_reg = A + c[:, :, 0, 0][:, None, None] * nnT
        self.F_Ainv = np.linalg.inv(A_reg)
        self.F_cnn = c * nnT                                 # [M,3,3]
        self.F_dX = dX
        self.F_w = wgt

    # ------------------------- simulation -------------------------
    def _solve_set(self, i0, i1, L0, w0, w1, alpha, active, x, lam):
        d = x[i1] - x[i0]
        L = np.sqrt((d * d).sum(axis=1))
        Lsafe = np.maximum(L, 1e-9)
        C = L - L0
        dl = (-C - alpha * lam) / (w0 + w1 + alpha)
        dl = np.where(active, dl, 0.0)
        lam += dl
        corr = (dl / Lsafe)[:, None] * d
        np.add.at(x, i0, -w0[:, None] * corr)
        np.add.at(x, i1, w1[:, None] * corr)
        return lam, L

    def _plasticity(self, L0, Linit, pl, active, duct, L):
        strain = L / np.maximum(L0, 1e-9) - 1.0
        hi = active & (strain > self.YIELD)
        lo = active & (strain < -self.YIELD)
        dL0 = np.zeros_like(L0)
        dL0[hi] = self.FLOW * (L[hi] - (1.0 + self.YIELD) * L0[hi])
        dL0[lo] = self.FLOW * (L[lo] - (1.0 - self.YIELD) * L0[lo])
        L0_new = np.clip(L0 + dL0, self.L0_MIN * Linit, self.L0_MAX * Linit)
        dpl = np.abs(L0_new - L0) / Linit
        pl += np.where(active, dpl, 0.0)
        L0[:] = L0_new
        # ductile fracture (only while the constraint is being pulled open)
        tear = active & (pl > duct) & (strain > 0.0)
        active[tear] = False
        self.torn_count += int(tear.sum())

    def _substep(self, dt):
        x = self.pos_u
        v = self.vel_u
        v += self.gravity * dt
        x_start = x.copy()
        x_pred = x + v * dt
        x[:] = x_pred

        s_lam = np.zeros(len(self.s_L0))
        b_lam = np.zeros(len(self.b_L0))
        s_soft = np.where(self.s_pl > self.NECK_START, self.NECK_SOFTEN, 1.0)
        s_alpha = self.s_alpha0 * s_soft
        b_soft = np.where(self.b_pl > self.NECK_START, self.NECK_SOFTEN, 1.0)
        b_alpha = self.b_alpha0 * b_soft
        for _ in range(self.ITERS):
            s_lam, s_L = self._solve_set(
                self.s_i0, self.s_i1, self.s_L0, self.s_w0, self.s_w1,
                s_alpha, self.s_active, x, s_lam,
            )
            if len(self.b_L0):
                b_lam, b_L = self._solve_set(
                    self.b_i0, self.b_i1, self.b_L0, self.b_w0, self.b_w1,
                    b_alpha, self.b_active, x, b_lam,
                )

        # plastic flow, necking and ductile tearing (base alpha stays fixed)
        self._plasticity(
            self.s_L0, self.s_Linit, self.s_pl,
            self.s_active, self.s_duct, s_L,
        )
        if len(self.b_L0):
            self._plasticity(
                self.b_L0, self.b_Linit, self.b_pl,
                self.b_active, self.b_duct, b_L,
            )

        # rigid floor at z = 0 (position projection)
        pen = x[:, 2] < self.FLOOR_R
        x[pen, 2] = self.FLOOR_R

        v = (x - x_start) / dt
        v *= self.DAMP
        if pen.any():
            vt = v[pen]
            vt[:, 0] *= self.FRICTION
            vt[:, 1] *= self.FRICTION
            vt[:, 2] = np.maximum(vt[:, 2], 0.0)
            v[pen] = vt
        sp = np.sqrt((v * v).sum(axis=1))
        bad = sp > self.V_MAX
        if bad.any():
            v[bad] *= (self.V_MAX / sp[bad])[:, None]
        self.vel_u = v
        self.pos_u = x

    def _deformation_gradients(self):
        x = self.pos_u
        dx = x[self.F_idx] - x[:, None, :]                    # [M,K,3]
        w3 = self.F_w[:, :, None, None]
        B = (w3 * dx[:, :, :, None] * self.F_dX[:, :, None, :]).sum(axis=1)
        B = B + self.F_cnn
        F = np.einsum("mij,mjk->mik", B, self.F_Ainv)
        return np.clip(F, -50.0, 50.0)

    # ------------------------- driver -------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        while self.time < target - 1e-12 and guard < 10_000_000:
            dt = min(self.DT, target - self.time)
            self._substep(dt)
            self.time += dt
            guard += 1

        m = self.map_u
        pos = self.pos_u[m].copy()
        vel = self.vel_u[m].copy()
        F_u = self._deformation_gradients()
        F = F_u[m].copy()

        ext_force = self.mass[:, None] * self.gravity[None, :]
        csd = pos[:, 2] - self.FLOOR_R  # signed gap to rigid floor plane
        return StateFrame(
            self.time,
            pos,
            vel,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.material_label.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(self.N),
                "contact_signed_distance": csd,
                "external_force": ext_force,
            },
            scalars={
                "contact_length_scale": self.spacing,
                "grid_spacing": self.spacing,
                "characteristic_length": max(
                    float(np.ptp(self.frame0.position, axis=0).max()), self.spacing
                ),
                "reference_velocity": 1.0,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "torn_constraints": self.torn_count,
            "stretch_constraints": int(len(self.s_L0)),
            "bend_constraints": int(len(self.b_L0)),
            "unique_particles": self.M,
            "samples": self.N,
            "min_z": float(self.pos_u[:, 2].min()) if self.M else 0.0,
        }
