"""Viscoplastic bunny-staircase solver.

Physically-based deformable solid: an elastic-plastic mass/spring (XPBD) model on
the bunny surface mesh.  Repeated impacts drive strain past a yield threshold,
accumulate plastic work, and locally *soften* the material (Bargteil 2007 style
work-softening), so worked regions droop and flow over the stair edges.  Collisions
with the axis-aligned stair/ground boxes are resolved with signed-distance
projection plus restitution and Coulomb-style friction.

Exports the required raw trajectory fields:
  attributes.deformation_gradient  [N,3,3]
  attributes.mass_quantity         [N]
  fields.contact_signed_distance   [C]
  fields.cumulative_mass_source    [K]
  fields.interaction_force         [K,3]
  scalars.contact_length_scale     scalar
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)

        # ---- mesh -------------------------------------------------------
        asset = Path(__file__).resolve().parent.parent / cfg["subject_asset"]
        with np.load(asset) as data:
            verts = np.asarray(data["vertices"], dtype=float)
            faces = np.asarray(data["faces"], dtype=np.int64)
        self.faces = faces
        self.X0 = verts.copy()          # frozen material (reference) configuration
        self.n = len(verts)
        self.ar_n = np.arange(self.n)

        # ---- initial dynamic state from the scene contract -------------
        init = scene.initial_state()
        self.X = np.asarray(init.position, dtype=float).copy()
        self.Vel = np.asarray(init.velocity, dtype=float).copy()
        self.mass = np.asarray(init.mass, dtype=float).copy()
        self.w = 1.0 / np.maximum(self.mass, 1e-12)          # inverse mass [N]
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        # ---- material / contact parameters -----------------------------
        self.stiffness = float(cfg.get("stiffness", 22.0))
        self.softening_rate = float(cfg.get("softening_rate", 0.34))
        self.damping = float(cfg.get("damping", 2.3))
        self.restitution = float(cfg.get("restitution", 0.12))
        self.friction = float(min(1.0, max(0.0, float(cfg.get("friction", 0.72)))))
        self.contact_length_scale = float(cfg.get("contact_length_scale", 0.04))

        # ---- integrator settings ---------------------------------------
        self.dt_max = 1.0 / 180.0
        self.iterations = 5
        self.damp_factor_scale = 0.05

        # ---- plasticity constants --------------------------------------
        self.yield0 = 0.10
        self.plastic_flow = 0.40
        self.YFAC = 0.60          # yield softening with plastic work
        self.KSOFT = 2.0          # stiffness softening with plastic work

        self._build_edges(faces)
        self._build_obstacles(cfg)
        self._build_deformation_fit(faces)

        # plastic state (per edge)
        self.L0 = self.rest_len.copy()
        self.ep = np.zeros(len(self.rest_len))
        self.yield_eff = np.full(len(self.rest_len), self.yield0)
        self.k_base = self.stiffness * 30.0
        self.k_eff = np.full(len(self.rest_len), self.k_base) * self.stiff_mult
        self.alpha_e = 1.0 / np.maximum(self.k_eff, 1e-6)
        self.wsum = self.w[self.ei] + self.w[self.ej]

        self.t = 0.0
        self.contact_impulse = np.zeros((self.n, 3))
        self.total_plastic = 0.0

    # ------------------------------------------------------------- topology
    def _build_edges(self, faces):
        struct = set()
        for a, b, c in faces:
            for u, v in ((a, b), (b, c), (c, a)):
                struct.add((min(u, v), max(u, v)))
        struct = np.array(sorted(struct), dtype=np.int64)

        opp = defaultdict(list)
        for a, b, c in faces:
            for (u, v), o in (((a, b), c), ((b, c), a), ((c, a), b)):
                opp[(min(u, v), max(u, v))].append(o)
        bend = set()
        for os_ in opp.values():
            if len(os_) == 2 and os_[0] != os_[1]:
                u, v = os_
                bend.add((min(u, v), max(u, v)))
        bend = np.array(sorted(bend), dtype=np.int64) if bend else np.zeros((0, 2), np.int64)

        edges = np.vstack([struct, bend])
        kind = np.concatenate([np.zeros(len(struct)), np.ones(len(bend))])
        self.ei = edges[:, 0].copy()
        self.ej = edges[:, 1].copy()
        self.rest_len = np.linalg.norm(self.X0[self.ei] - self.X0[self.ej], axis=1)
        self.rest_len = np.maximum(self.rest_len, 1e-6)
        # bending (diagonal) springs are softer than structural membrane springs
        self.stiff_mult = np.where(kind < 0.5, 1.0, 0.5)

    def _build_obstacles(self, cfg):
        centers = []
        halves = []
        geom = cfg.get("scene_geometry")
        if geom is not None:
            for obj in geom.get("objects", []):
                if not obj.get("collision_enabled", True):
                    continue
                g = obj.get("geometry", {})
                if g.get("type") != "box":
                    continue
                tr = obj.get("transform", {})
                centers.append(list(tr.get("translation", [0.0, 0.0, 0.0])))
                halves.append(list(g["half_extents"]))
        if not centers:  # fallback to the frozen staircase layout
            centers = [[0.0, 0.0, -0.04], [-0.75, 0.0, 0.96], [-0.15, 0.0, 0.68],
                       [0.45, 0.0, 0.40], [1.05, 0.0, 0.12]]
            halves = [[3.0, 1.6, 0.04], [0.36, 1.05, 0.14], [0.36, 1.05, 0.14],
                      [0.36, 1.05, 0.14], [0.36, 1.05, 0.14]]
        self.obs_c = np.asarray(centers, dtype=float)
        self.obs_h = np.asarray(halves, dtype=float)

    def _build_deformation_fit(self, faces):
        nbr = defaultdict(list)
        for a, b, c in faces:
            nbr[a] += [b, c]
            nbr[b] += [a, c]
            nbr[c] += [a, b]
        dmax = max(len(set(v)) for v in nbr.values())
        self.nbr_idx = np.zeros((self.n, dmax), dtype=np.int64)
        self.nbr_mask = np.zeros((self.n, dmax), dtype=float)
        for i in range(self.n):
            us = sorted(set(nbr[i]))
            self.nbr_idx[i, :len(us)] = us
            self.nbr_mask[i, :len(us)] = 1.0
        # reference neighbour vectors (material coordinates)
        self.ref_vec = (self.X0[self.nbr_idx] - self.X0[:, None, :]) * self.nbr_mask[..., None]
        M = np.einsum('ndi,ndj->nij', self.ref_vec, self.ref_vec)
        ridge = 1e-6 * (np.trace(M, axis1=1, axis2=2) / 3.0 + 1e-9)
        M = M + ridge[:, None, None] * np.eye(3)
        self.Minv = np.linalg.inv(M)

    # -------------------------------------------------------------- helpers
    def _obstacle_sdf(self, P):
        """Signed distance and outward normal to the nearest box obstacle."""
        d = P[:, None, :] - self.obs_c[None, :, :]           # [N,B,3]
        q = np.abs(d) - self.obs_h[None, :, :]               # [N,B,3]
        outside = np.linalg.norm(np.clip(q, 0.0, None), axis=2)
        inside = np.minimum(np.max(q, axis=2), 0.0)
        sd_all = outside + inside                            # [N,B]
        b = np.argmin(sd_all, axis=1)                        # [N]
        sd = sd_all[self.ar_n, b]
        db = d[self.ar_n, b]                                 # [N,3]
        hb = self.obs_h[b]                                   # [N,3]
        clamped = np.clip(db, -hb, hb)
        vec = db - clamped
        nv = np.linalg.norm(vec, axis=1)
        pen = hb - np.abs(db)                                # positive inside
        axis = np.argmin(pen, axis=1)
        inside_n = np.zeros_like(db)
        inside_n[self.ar_n, axis] = np.sign(db[self.ar_n, axis] + 1e-12)
        safe = np.where(nv > 1e-9, nv, 1.0)
        n = np.where((nv > 1e-9)[:, None], vec / safe[:, None], inside_n)
        return sd, n

    def _deformation_gradient(self, X):
        cur = (X[self.nbr_idx] - X[:, None, :]) * self.nbr_mask[..., None]
        S = np.einsum('ndi,ndj->nij', cur, self.ref_vec)
        return np.einsum('nij,njk->nik', S, self.Minv)

    # ------------------------------------------------------- constraint solve
    def _solve_distance(self, P, lam, dt):
        dvec = P[self.ei] - P[self.ej]
        dist = np.sqrt(np.einsum('ei,ei->e', dvec, dvec)) + 1e-9
        ndir = dvec / dist[:, None]
        C = dist - self.L0
        alpha_t = self.alpha_e / (dt * dt)
        denom = self.wsum + alpha_t
        dlam = (-C - alpha_t * lam) / denom
        lam += dlam
        np.add.at(P, self.ei, (self.w[self.ei] * dlam)[:, None] * ndir)
        np.add.at(P, self.ej, -(self.w[self.ej] * dlam)[:, None] * ndir)

    def _update_plasticity(self, X):
        dvec = X[self.ei] - X[self.ej]
        dist = np.sqrt(np.einsum('ei,ei->e', dvec, dvec)) + 1e-9
        strain = (dist - self.L0) / self.L0
        astrain = np.abs(strain)
        mask = astrain > self.yield_eff
        if not mask.any():
            return
        sign = np.sign(strain)
        new_L0 = dist / (1.0 + sign * self.yield_eff)
        dL0 = np.where(mask, self.plastic_flow * (new_L0 - self.L0), 0.0)
        self.L0 = np.clip(self.L0 + dL0, 0.3 * self.rest_len, 3.0 * self.rest_len)
        dep = np.abs(dL0) / self.L0
        self.ep += dep
        sr = self.softening_rate
        self.yield_eff = np.clip(self.yield0 * np.exp(-sr * self.ep * self.YFAC),
                                 0.02, 2.0 * self.yield0)
        self.k_eff = np.clip(self.k_base * self.stiff_mult *
                             np.exp(-sr * self.ep * self.KSOFT),
                             0.05 * self.k_base, 2.0 * self.k_base)
        self.alpha_e = 1.0 / np.maximum(self.k_eff, 1e-6)
        self.total_plastic = float(self.ep.sum())

    # ------------------------------------------------------------- one step
    def _substep(self, dt):
        self.Vel = self.Vel + self.gravity * dt
        P = self.X + self.Vel * dt
        lam = np.zeros(len(self.L0))
        for _ in range(self.iterations):
            self._solve_distance(P, lam, dt)

        Vpre = (P - self.X) / dt
        sd, nrm = self._obstacle_sdf(P)
        pen = sd < 0.0
        V = Vpre.copy()
        if pen.any():
            idx = np.flatnonzero(pen)
            corr = -(sd[idx])[:, None] * nrm[idx]
            P[idx] += corr
            V[idx] = (P[idx] - self.X[idx]) / dt
            nn = nrm[idx]
            vn = np.einsum('ki,ki->k', V[idx], nn)
            vn_in = np.einsum('ki,ki->k', Vpre[idx], nn)
            vn_target = -self.restitution * np.minimum(vn_in, 0.0)
            # Only raise the normal velocity up to the bounce target; never remove
            # an already-outward velocity (avoids trapping penetrating vertices).
            dvn = np.maximum(vn_target - vn, 0.0)
            V[idx] += dvn[:, None] * nn
            vt = V[idx] - np.einsum('ki,ki->k', V[idx], nn)[:, None] * nn
            V[idx] -= vt * self.friction
            self.contact_impulse[idx] += self.mass[idx, None] * (V[idx] - Vpre[idx])

        # mild viscous damping (does not suppress free fall appreciably)
        V *= float(np.exp(-self.damping * dt * self.damp_factor_scale))

        self.X = P
        self.Vel = V
        self._update_plasticity(self.X)

    # ---------------------------------------------------------- public API
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        frame_dt = time - self.t
        self.contact_impulse = np.zeros((self.n, 3))
        if frame_dt > 1e-12:
            n_sub = max(1, int(np.ceil(frame_dt / self.dt_max)))
            dt = frame_dt / n_sub
            for _ in range(n_sub):
                self._substep(dt)
            self.t = time
        elif frame_dt < 0.0:
            # should not happen with monotonic sample times; stay put
            pass

        F = self._deformation_gradient(self.X)
        sd, _ = self._obstacle_sdf(self.X)
        if frame_dt > 1e-12:
            contact_force = self.contact_impulse / frame_dt
        else:
            contact_force = np.zeros((self.n, 3))

        frame = StateFrame(
            time=self.t,
            position=self.X.copy(),
            velocity=self.Vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F.astype(float),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "contact_signed_distance": sd.astype(float),
                "interaction_force": contact_force.astype(float),
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "solver": "xpbd_viscoplastic",
            "n_vertices": int(self.n),
            "n_edges": int(len(self.L0)),
            "total_plastic_strain": float(self.total_plastic),
            "min_deformation_gradient_det": float(
                np.min(np.linalg.det(self._deformation_gradient(self.X)))
            ) if self.n else 0.0,
        }
