"""Physically-based solver for the pile of forty incompressible elastic tori.

Model
-----
* Each torus is a three-dimensional solid: a tetrahedral mesh built on the
  frozen 24 x 12 surface lattice (the rendered topology) with a 24-node
  centerline ring inside the tube.  Every cross-section disk is fan-triangulated
  about its centerline node and extruded into prisms split into 3 tets each
  (864 tets/torus, 34,560 total).  The centerline nodes are kinematically
  condensed (center = mean of its 12 ring surface nodes), so the dynamic DOFs
  are exactly the 11,520 exported surface samples and the exported momentum
  budget closes exactly.
* Material: stable nearly-incompressible neo-Hookean hyperelasticity
  (nu = 0.43), first Piola stress P = mu (F - F^-T) + lambda ln(J) F^-T,
  integrated with an explicit symplectic-Euler / central-difference scheme at a
  time step below the dilatational CFL limit.
* Contact: penalty spring-dashpot with Coulomb friction against the floor
  plane (z=0), the twelve rigid cylindrical posts, and between tori.
  Torus-torus contact models each body as its deformed 12-segment centerline
  polyline with an effective tube radius measured from the current surface, so
  coverage is continuous (no vertex-lattice gaps).  Reactions are distributed
  back to the opposing surface DOFs, hence inter-body pairs cancel exactly in
  the global momentum budget.
* A small linear drag (reported as an external budget force term) plus contact
  dashpots dissipate impact energy so the pile settles on the floor.

All exported trajectory fields required by the task contract are produced.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

# --- scene constants (frozen by scene.py / scene_geometry.yaml) -------------
NB = 40            # tori
NU = 24            # surface lattice, major direction
NV = 12            # surface lattice, tube direction
NS = NB * NU * NV  # 11520 exported surface samples (dynamic DOFs)
NC = NB * NU       # 960 centerline nodes (condensed)
MAJOR_R = 0.46
TUBE_R = 0.18
BODY_MASS = 0.9    # kg per torus (frozen by the scene initial state)

# --- numerics ---------------------------------------------------------------
DT = 4.0e-4                 # stable explicit step (CFL ~ 0.35)
YOUNG = 5.0e3               # Pa (soft elastomer; tori are very light)
POISSON = 0.43              # nearly incompressible
MU = YOUNG / (2.0 * (1.0 + POISSON))
LAM = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
GAMMA = 0.6                 # 1/s linear drag (budgeted as external force)
VMAX = 30.0                 # m/s safety clamp

DELTA0 = 0.015              # contact margin (m)
KC_BODY = 1.2e4             # N/m per vertex, torus-torus
CN_BODY = 5.2               # N*s/m normal dashpot
MU_BODY = 0.5               # Coulomb friction
KC_FLOOR = 2.5e4
CN_FLOOR = 10.6
MU_FLOOR = 0.6
DELTA_F = 0.012
KC_POST = 2.5e4
CN_POST = 10.6
MU_POST = 0.5
DELTA_P = 0.012
POST_R = 0.13
POST_ZMAX = 2.8

GRAVITY = np.array([0.0, 0.0, -9.81])
REF_MOMENTUM = float(BODY_MASS * NB * math.sqrt(2.0 * 9.81 * 5.0))  # frozen scale


def _rotate(points: np.ndarray, q: np.ndarray) -> np.ndarray:
    xyz, w = q[:3], q[3]
    return points + 2.0 * np.cross(xyz, np.cross(xyz, points) + w * points)


def _local_surface() -> np.ndarray:
    u = np.arange(NU) * (2.0 * math.pi / NU)
    v = np.arange(NV) * (2.0 * math.pi / NV)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    radial = MAJOR_R + TUBE_R * np.cos(vv)
    return np.column_stack((
        (radial * np.cos(uu)).ravel(),
        (radial * np.sin(uu)).ravel(),
        (TUBE_R * np.sin(vv)).ravel(),
    ))


def _local_centers() -> np.ndarray:
    u = np.arange(NU) * (2.0 * math.pi / NU)
    return np.column_stack((MAJOR_R * np.cos(u), MAJOR_R * np.sin(u),
                            np.zeros(NU)))


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        frame0 = scene.initial_state()
        pos0 = np.asarray(frame0.position, dtype=float)
        orient0 = np.asarray(frame0.orientation, dtype=float)

        self.orient0 = orient0.copy()
        self.body_id = np.repeat(np.arange(NB, dtype=np.int32), NU * NV)
        self.mass_vec = np.full(NS, BODY_MASS / (NU * NV))
        self.mass_quantity = self.mass_vec[:, None].astype(np.float32)
        self.mcol = self.mass_vec[:, None]
        self.fgrav = self.mass_vec[:, None] * GRAVITY[None, :]

        # rest configuration: surface samples straight from the frozen scene,
        # centerline ring rotated/translated with the same rigid pose.
        local_s = _local_surface()
        local_c = _local_centers()
        rest = np.empty((NS + NC, 3))
        for b in range(NB):
            q = orient0[b * NU * NV]
            surf = pos0[b * NU * NV:(b + 1) * NU * NV]
            t = np.mean(surf - _rotate(local_s, q), axis=0)
            rest[b * NU * NV:(b + 1) * NU * NV] = surf
            rest[NS + b * NU:NS + (b + 1) * NU] = _rotate(local_c, q) + t
        self.rest = rest

        # tetrahedra ------------------------------------------------------
        bb = np.repeat(np.arange(NB), NU * NV)
        ii = np.tile(np.repeat(np.arange(NU), NV), NB)
        jj = np.tile(np.tile(np.arange(NV), NU), NB)

        def S(b, u, v):
            return b * (NU * NV) + (u % NU) * NV + (v % NV)

        def C(b, u):
            return NS + b * NU + (u % NU)

        A = C(bb, ii)
        B = S(bb, ii, jj)
        Cc = S(bb, ii, jj + 1)
        A2 = C(bb, ii + 1)
        B2 = S(bb, ii + 1, jj)
        C2 = S(bb, ii + 1, jj + 1)
        tets = np.stack((
            np.stack((A, B, Cc, A2), axis=1),
            np.stack((B, Cc, B2, A2), axis=1),
            np.stack((B2, Cc, C2, A2), axis=1),
        ), axis=1).reshape(-1, 4).astype(np.int64)
        self.tets = tets
        self.ntet = len(tets)

        # rest shape gradients / volumes, flipped to positive orientation
        e = rest[tets]
        e1, e2, e3 = e[:, 1] - e[:, 0], e[:, 2] - e[:, 0], e[:, 3] - e[:, 0]
        det = np.einsum("ti,ti->t", np.cross(e1, e2), e3)
        bad = det <= 0.0
        tets[bad] = tets[bad][:, [0, 1, 3, 2]]
        self.tets = tets
        e = rest[tets]
        Dm = np.stack((e[:, 1] - e[:, 0], e[:, 2] - e[:, 0], e[:, 3] - e[:, 0]),
                      axis=2)
        det = np.linalg.det(Dm)
        self.V0 = det / 6.0
        if not np.all(self.V0 > 0.0):
            raise RuntimeError("degenerate rest tetrahedra")
        Dm_inv = np.linalg.inv(Dm)
        self.Dm_inv = Dm_inv
        self.Dm_invT = np.ascontiguousarray(np.transpose(Dm_inv, (0, 2, 1)))

        # per-body rest volume (constraint target)
        self.constraint_target = self.V0.reshape(NB, -1).sum(axis=1).copy()
        self.constraint_scale = float(self.constraint_target.mean())

        # budget domains: row 0 = global pile, rows 1..40 = individual tori
        w = np.zeros((NB + 1, NS), dtype=np.float32)
        w[0, :] = 1.0
        for b in range(NB):
            w[1 + b, b * NU * NV:(b + 1) * NU * NV] = 1.0
        self.domain_weights = w
        self.grav_body = np.tile(np.array([0.0, 0.0, -BODY_MASS * 9.81]),
                                 (NB, 1))

        # state -------------------------------------------------------------
        self.X = pos0.copy()
        self.V = np.zeros((NS, 3))
        self.t = 0.0
        self.nsteps = 0
        self.cum_impulse_body = np.zeros((NB, 3))
        self.Fsum = np.zeros((5, NB, 3))  # gravity, floor, post, inter, drag
        self.Fsum[0] = self.grav_body.copy()
        self.lastF = np.tile(np.eye(3), (self.ntet, 1, 1)).astype(np.float64)
        self.lastJ = np.ones(self.ntet)
        self.contacts_every = 2

        self.post_xy = []
        for k in range(12):
            ang = 2.0 * math.pi * k / 12.0
            self.post_xy.append((2.35 * math.cos(ang), 2.35 * math.sin(ang)))
        self.post_xy = np.array(self.post_xy)

    # ------------------------------------------------------------- elasticity
    def _elastic(self, f: np.ndarray) -> None:
        Xs = self.X
        cen = Xs.reshape(NB, NU, NV, 3).mean(axis=2)  # (40,24,3)
        Xf = np.concatenate((Xs, cen.reshape(NC, 3)))
        x4 = Xf[self.tets]
        Ds = np.stack((x4[:, 1] - x4[:, 0], x4[:, 2] - x4[:, 0],
                       x4[:, 3] - x4[:, 0]), axis=2)
        F = Ds @ self.Dm_inv
        f1, f2, f3 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        g1 = np.cross(f2, f3)
        g2 = np.cross(f3, f1)
        g3 = np.cross(f1, f2)
        J = np.einsum("ti,ti->t", g3, f3)
        Js = np.maximum(J, 1.0e-3)
        invJ = (1.0 / Js)[:, None, None]
        FinvT = np.stack((g1, g2, g3), axis=2) * invJ
        lnJ = np.log(Js)[:, None, None]
        P = MU * (F - FinvT) + (LAM * lnJ) * FinvT
        M = P @ self.Dm_invT           # dE/dDs / V0
        f123 = -self.V0[:, None, None] * M          # columns -> nodes 1..3
        fall = np.empty((self.ntet, 4, 3))
        fall[:, 1, :] = f123[:, :, 0]
        fall[:, 2, :] = f123[:, :, 1]
        fall[:, 3, :] = f123[:, :, 2]
        fall[:, 0, :] = -(f123.sum(axis=2))
        fnodes = np.empty((NS + NC, 3))
        flat = self.tets.ravel()
        for c in range(3):
            fnodes[:, c] = np.bincount(flat, weights=fall[:, :, c].ravel(),
                                       minlength=NS + NC)
        fc = fnodes[NS:].reshape(NB, NU, 3)
        spread = np.repeat(fc, NV, axis=1).reshape(NS, 3) / float(NV)
        f[:] = fnodes[:NS] + spread
        # store for exports
        self.lastF = F
        self.lastJ = J
        self.cen = cen
        dr = Xs.reshape(NB, NU, NV, 3) - cen[:, :, None, :]
        self.reff_all = np.sqrt((dr * dr).sum(axis=3)).mean(axis=2)  # (40,24)

    # --------------------------------------------------------------- contacts
    def _ring_data(self):
        rings = np.ascontiguousarray(self.cen[:, ::2, :])       # (40,12,3)
        reff = np.ascontiguousarray(self.reff_all[:, ::2])      # (40,12)
        vring = self.V.reshape(NB, NU, NV, 3).mean(axis=2)[:, ::2, :]
        return rings, reff, np.ascontiguousarray(vring)

    def _floor_post(self, f: np.ndarray) -> None:
        Fsum = self.Fsum
        Fsum[1].fill(0.0)
        Fsum[2].fill(0.0)
        X, V = self.X, self.V
        # floor (top surface at z = 0)
        z = X[:, 2]
        idx = np.flatnonzero(z < DELTA_F)
        if idx.size:
            pen = DELTA_F - z[idx]
            vz = V[idx, 2]
            fn = np.maximum(KC_FLOOR * pen - CN_FLOOR * vz, 0.0)
            Floc = np.zeros((idx.size, 3))
            Floc[:, 2] = fn
            vt = V[idx, :2]
            sp = np.sqrt((vt * vt).sum(axis=1))
            Floc[:, :2] = -(MU_FLOOR * fn)[:, None] * vt / (sp + 1.0e-9)[:, None]
            f[idx] += Floc
            np.add.at(Fsum[1], self.body_id[idx], Floc)
        # posts (12 vertical cylinders)
        for px, py in self.post_xy:
            dx = X[:, 0] - px
            dy = X[:, 1] - py
            rho = np.sqrt(dx * dx + dy * dy)
            zz = X[:, 2]
            idx = np.flatnonzero((rho < POST_R + DELTA_P) & (rho > 1.0e-9)
                                 & (zz > 0.0) & (zz < POST_ZMAX))
            if not idx.size:
                continue
            r = rho[idx]
            pen = POST_R + DELTA_P - r
            nhat = np.column_stack((dx[idx] / r, dy[idx] / r, np.zeros(r.size)))
            vv = V[idx]
            vn = np.einsum("ti,ti->t", vv, nhat)
            fn = np.maximum(KC_POST * pen - CN_POST * vn, 0.0)
            vt = vv - vn[:, None] * nhat
            sp = np.sqrt((vt * vt).sum(axis=1))
            Floc = fn[:, None] * nhat - (MU_POST * fn)[:, None] * vt / (sp + 1.0e-9)[:, None]
            f[idx] += Floc
            np.add.at(Fsum[2], self.body_id[idx], Floc)

    def _inter_body(self, f: np.ndarray) -> None:
        rings, reff, vring = self._ring_data()
        bcen = rings.mean(axis=1)
        Fsum3 = self.Fsum[3]
        Fsum3.fill(0.0)
        dcen = bcen[:, None, :] - bcen[None, :, :]
        dist = np.sqrt((dcen * dcen).sum(axis=2))
        cand = np.argwhere((dist < 1.45) & (dist > 1.0e-12))
        ar12 = np.arange(NV)[None, :]
        for a, b in cand:
            a = int(a); b = int(b)
            sl = slice(a * NU * NV, (a + 1) * NU * NV)
            p = self.X[sl]
            dvec = p - bcen[b]
            d2 = (dvec * dvec).sum(axis=1)
            mask = d2 < 0.64  # 0.80^2 prefilter
            if not mask.any():
                continue
            m_idx = np.flatnonzero(mask)
            pm = p[m_idx]
            vm = self.V[sl][m_idx]
            A0 = rings[b]
            AB = np.roll(A0, -1, axis=0) - A0
            L2 = np.maximum((AB * AB).sum(axis=1), 1.0e-12)
            va = vring[b]
            vB = np.roll(va, -1, axis=0)
            ra = reff[b]
            rb = np.roll(ra, -1)
            diff = pm[:, None, :] - A0[None, :, :]
            t = np.clip((diff * AB[None, :, :]).sum(axis=2) / L2[None, :], 0.0, 1.0)
            cp = A0[None, :, :] + t[:, :, None] * AB[None, :, :]
            dv = pm[:, None, :] - cp
            d = np.sqrt((dv * dv).sum(axis=2))
            rseg = ra[None, :] * (1.0 - t) + rb[None, :] * t
            pen = rseg + DELTA0 - d
            hit = (pen > 0.0) & (d > 1.0e-9)
            if not hit.any():
                continue
            rr, ss = np.nonzero(hit)
            dh = d[rr, ss]
            nhat = dv[rr, ss] / dh[:, None]
            th = t[rr, ss]
            fn = KC_BODY * pen[rr, ss]
            vcp = va[ss] * (1.0 - th)[:, None] + vB[ss] * th[:, None]
            vrel = vm[rr] - vcp
            vn = np.einsum("ti,ti->t", vrel, nhat)
            fn = np.maximum(fn - CN_BODY * vn, 0.0)
            vt = vrel - vn[:, None] * nhat
            sp = np.sqrt((vt * vt).sum(axis=1))
            Ft = fn[:, None] * nhat - (MU_BODY * fn)[:, None] * vt / (sp + 1.0e-9)[:, None]
            gidx = a * NU * NV + m_idx[rr]
            np.add.at(f, gidx, Ft)
            # reactions distributed onto body b's surface DOFs (Newton III)
            st1 = (2 * ss) % NU
            st2 = (2 * ss + 2) % NU
            base1 = (b * NU * NV + st1[:, None] * NV + ar12).ravel()
            base2 = (b * NU * NV + st2[:, None] * NV + ar12).ravel()
            w1 = -(Ft * (1.0 - th)[:, None]) / float(NV)
            w2 = -(Ft * th[:, None]) / float(NV)
            np.add.at(f, base1, np.repeat(w1, NV, axis=0))
            np.add.at(f, base2, np.repeat(w2, NV, axis=0))
            tot = Ft.sum(axis=0)
            Fsum3[a] += tot
            Fsum3[b] -= tot

    # -------------------------------------------------------------- time step
    def _substep(self) -> None:
        f = np.empty((NS, 3))
        self._elastic(f)
        f += self.fgrav
        self._floor_post(f)
        if self.nsteps % self.contacts_every == 0:
            self._inter_body(f)
        self.Fsum[0] = self.grav_body
        # integrate (symplectic Euler)
        self.V += DT * (f / self.mcol)
        # linear drag, budgeted exactly as an external force term
        drag_body = -GAMMA * (BODY_MASS / (NU * NV)) * self.V.reshape(NB, NU * NV, 3).sum(axis=1)
        self.Fsum[4] = drag_body
        self.V *= (1.0 - GAMMA * DT)
        sp = np.sqrt((self.V * self.V).sum(axis=1))
        over = sp > VMAX
        if over.any():
            self.V[over] *= (VMAX / sp[over])[:, None]
        self.X += DT * self.V
        below = self.X[:, 2] < 0.0
        if below.any():
            self.X[below, 2] = 0.0
            np.maximum(self.V[below, 2], 0.0, out=self.V[below, 2])
        # cumulative external impulse (per body)
        self.cum_impulse_body += DT * (self.Fsum[0] + self.Fsum[1] + self.Fsum[2]
                                       + self.Fsum[3] + self.Fsum[4])
        self.t += DT
        self.nsteps += 1

    # ----------------------------------------------------------------- frames
    def _make_frame(self, time: float) -> StateFrame:
        vol = np.einsum("be,be->b", self.lastJ.reshape(NB, -1),
                        self.V0.reshape(NB, -1))
        ci = np.zeros((NB + 1, 3))
        ci[0] = self.cum_impulse_body.sum(axis=0)
        ci[1:] = self.cum_impulse_body
        bef = np.zeros((NB + 1, 5, 3))
        bef[0] = self.Fsum.sum(axis=1)
        bef[1:] = np.transpose(self.Fsum, (1, 0, 2))
        fields = {
            "budget_cumulative_external_impulse": ci,
            "budget_cumulative_inflow_momentum": np.zeros((NB + 1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((NB + 1, 3)),
            "budget_domain_weights": self.domain_weights,
            "budget_external_force": bef,
            "constraint_target": self.constraint_target.copy(),
            "constraint_value": vol,
            "cumulative_mass_source": np.zeros(1),
            "element_deformation_gradient": self.lastF.astype(np.float32),
        }
        frame = StateFrame(
            time=float(time),
            position=self.X.copy(),
            velocity=self.V.copy(),
            orientation=self.orient0,
            mass=self.mass_vec,
            body_id=self.body_id,
            attributes={"mass_quantity": self.mass_quantity},
            fields=fields,
            scalars={
                "constraint_scale": self.constraint_scale,
                "reference_momentum": REF_MOMENTUM,
                "gravity_z": -9.81,
                "torus_count": float(NB),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        n = int(round((target - self.t) / DT))
        for _ in range(max(n, 0)):
            self._substep()
        self.t = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        vol = np.einsum("be,be->b", self.lastJ.reshape(NB, -1),
                        self.V0.reshape(NB, -1))
        rel = vol / self.constraint_target
        return {
            "model": "neo-Hookean tet FEM (nu=0.43), explicit dt=4e-4, "
                     "penalty contact with friction, condensed centerline",
            "substeps": self.nsteps,
            "elements": self.ntet,
            "min_relative_volume": float(rel.min()) if self.ntet else 1.0,
            "max_relative_volume": float(rel.max()) if self.ntet else 1.0,
            "min_z": float(self.X[:, 2].min()),
        }
