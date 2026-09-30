"""Elastoplastic particle solver for the lava spout periodic folding scene.

Physical model
--------------
The lava is discretized with material particles carrying mass and a reference
(lagrangian) position.  Cohesion is enforced with pairwise distance
constraints solved by a position-based dynamics (PBD/XPBD) projection each
substep.  The material is a von-Mises-like elastoplastic solid: whenever a
constraint is strained beyond a small yield band its rest length flows
plastically, so the material behaves elastically at small strain and
permanently deforms (folds) under large strain -- the regime needed for a
viscous lava spout that buckles and accumulates alternating folds.

Strong viscosity is modeled with an XSPH-style pairwise velocity averaging
(very high dynamic viscosity for a hot silicate melt).

Collisions use exact signed distances against the frozen rigid geometry
(ground box, catch-basin cylinder) with basal friction so the incoming stream
cannot spread fast enough and must buckle laterally into a folded pile.

A prescribed inlet at the nozzle emits new material layers with a fixed
downward velocity, giving a steady vertical stream; the cumulative emitted
mass is tracked for the trajectory contract.

All stepping is explicit, deterministic and finite: fixed substep count,
fixed iteration counts, no randomness.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is expected but not essential
    cKDTree = None


class BenchmarkSolver:
    """Deterministic XPBD elastoplastic solver for scene 129."""

    SUB_DT = 1.0 / 600.0          # physical substep (s)
    N_ITER = 6                    # constraint-projection iterations
    REBUILD_EVERY = 12            # substeps between neighbor-graph rebuilds
    EMIT_EVERY = 36               # substeps between inlet layers (0.06 s)
    STIFF = 0.35                  # Jacobi relaxation for distance constraints
    YIELD_STRAIN = 0.07           # elastic band before plastic flow
    VISCOUS = 0.7                 # XSPH velocity-averaging strength
    FRICTION = 0.9                # basal tangential damping per substep
    CONTACT_EPS = 0.03            # contact margin for friction (m)
    VMAX = 25.0                   # safety velocity clamp (m/s)

    # Frozen rigid geometry (from scene_geometry.yaml)
    GROUND_C = np.array([0.0, 0.0, -0.05])
    GROUND_H = np.array([3.2, 1.4, 0.05])
    BASIN_R = 1.35
    BASIN_Z0 = 0.0
    BASIN_Z1 = 0.12
    NOZZLE_EXIT_Z = 2.81          # bottom face of the nozzle cylinder
    V_IN = 1.2                    # inlet speed (m/s)

    _BIG = 1 << 21                # pair-key stride

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None)
        sampling = cfg.get("sampling", {}) if isinstance(cfg, dict) else {}
        self.spacing = float(sampling.get("spacing", 0.14))
        self.rho = float(sampling.get("density", 1350.0))
        grav = cfg.get("gravity", [0.0, 0.0, -9.81]) if isinstance(cfg, dict) else [0.0, 0.0, -9.81]
        self.gravity = np.asarray(grav, dtype=float)

        self.r_neigh = 1.5 * self.spacing
        self.r_drop = 2.2 * self.spacing

        f0 = scene.initial_state()
        self.X = np.array(f0.position, dtype=float)          # reference coords
        self.p = self.X.copy()
        self.v = np.zeros_like(self.p)
        self.pprev = self.p.copy()
        m0 = np.asarray(f0.mass, dtype=float)
        self.m = m0.copy()
        self.pm = float(m0[0]) if m0.size else self.rho * self.spacing ** 3
        self.body = (np.asarray(f0.body_id, dtype=np.int32).copy()
                     if f0.body_id is not None else np.zeros(len(self.p), np.int32))
        scal = dict(f0.scalars) if f0.scalars else {}
        self.char_len = float(scal.get("characteristic_length", 2.8))
        self.ref_vel = float(scal.get("reference_velocity", self.V_IN))

        self.step = 0
        self.emitted_mass = 0.0
        self.pi = np.zeros(0, np.int64)
        self.pj = np.zeros(0, np.int64)
        self.rest = np.zeros(0)
        self._rebuild()

    # --------------------------------------------------------------- emitter
    def _emit(self) -> None:
        off = np.array([-0.045, 0.095])
        gx, gy = np.meshgrid(off, off)
        pos = np.column_stack((gx.ravel(), gy.ravel(),
                               np.full(4, self.NOZZLE_EXIT_Z)))
        vel = np.zeros((4, 3))
        vel[:, 2] = -self.V_IN
        self.p = np.vstack((self.p, pos))
        self.v = np.vstack((self.v, vel))
        self.pprev = np.vstack((self.pprev, pos - vel * self.SUB_DT))
        self.X = np.vstack((self.X, pos))
        self.m = np.append(self.m, np.full(4, self.pm))
        self.body = np.append(self.body, np.zeros(4, np.int32))
        self.emitted_mass += 4.0 * self.pm

    # --------------------------------------------------------- neighbor graph
    def _rebuild(self) -> None:
        n = len(self.p)
        rest_map: dict[int, float] = {}

        # keep existing (possibly plastically stretched) constraints
        if self.pi.size:
            d = np.linalg.norm(self.p[self.pj] - self.p[self.pi], axis=1)
            keep = d <= 1.45 * self.rest
            for a, b, r in zip(self.pi[keep].tolist(),
                               self.pj[keep].tolist(),
                               self.rest[keep].tolist()):
                rest_map[a * self._BIG + b] = r

        # add new cohesive contacts within the interaction radius
        if n > 1:
            if cKDTree is not None:
                pairs = cKDTree(self.p).query_pairs(self.r_neigh,
                                                    output_type="ndarray")
                pairs = np.sort(np.asarray(pairs, dtype=np.int64), axis=1)
            else:
                dist = np.linalg.norm(self.p[:, None, :] - self.p[None, :, :],
                                      axis=2)
                a, b = np.nonzero(np.triu(dist < self.r_neigh, 1))
                pairs = np.column_stack((a, b))
            if pairs.size:
                d = np.linalg.norm(self.p[pairs[:, 1]] - self.p[pairs[:, 0]],
                                   axis=1)
                for a, b, dd in zip(pairs[:, 0].tolist(),
                                    pairs[:, 1].tolist(), d.tolist()):
                    key = a * self._BIG + b
                    if key not in rest_map:
                        rest_map[key] = dd

        if rest_map:
            keys = np.fromiter(rest_map.keys(), dtype=np.int64,
                               count=len(rest_map))
            self.pi = keys // self._BIG
            self.pj = keys % self._BIG
            self.rest = np.fromiter(rest_map.values(), dtype=float,
                                    count=len(rest_map))
        else:
            self.pi = np.zeros(0, np.int64)
            self.pj = np.zeros(0, np.int64)
            self.rest = np.zeros(0)

    # ------------------------------------------------------------ material
    def _viscosity(self) -> None:
        i, j = self.pi, self.pj
        if i.size == 0:
            return
        d = np.linalg.norm(self.p[j] - self.p[i], axis=1)
        w = np.clip(1.0 - (d / self.r_neigh) ** 2, 0.0, 1.0)
        dv = self.v[j] - self.v[i]
        acc = np.zeros_like(self.v)
        np.add.at(acc, i, w[:, None] * dv)
        np.add.at(acc, j, -w[:, None] * dv)
        wsum = np.zeros(len(self.p))
        np.add.at(wsum, i, w)
        np.add.at(wsum, j, w)
        # move a fraction of the way toward the local mass-weighted mean
        # velocity: (v_i + sum w v_j) / (1 + sum w); never overshoots.
        self.v += self.VISCOUS * acc / (1.0 + wsum)[:, None]

    def _solve_distances(self, plastic: bool) -> None:
        i, j = self.pi, self.pj
        if i.size == 0:
            return
        rest = self.rest
        d = self.p[j] - self.p[i]
        L = np.linalg.norm(d, axis=1)
        ok = L > 1e-12
        if plastic:
            ex = L - rest
            band = self.YIELD_STRAIN * rest
            rest = np.where(ex > band, rest + (ex - band), rest)
            rest = np.where(ex < -band, rest + (ex + band), rest)
            rest = np.maximum(rest, 0.02 * self.spacing)
            self.rest = rest
        Ls = np.where(ok, L, 1.0)
        corr = np.where(ok, (L - rest) / Ls, 0.0)
        move = (0.5 * self.STIFF * corr)[:, None] * d
        dp = np.zeros_like(self.p)
        np.add.at(dp, i, move)
        np.add.at(dp, j, -move)
        self.p += dp

    # ------------------------------------------------------------ collisions
    def _collide(self) -> None:
        p = self.p
        # ground box
        q = p - self.GROUND_C
        aq = np.abs(q)
        inside = np.all(aq < self.GROUND_H, axis=1)
        rows = np.nonzero(inside)[0]
        if rows.size:
            sub = q[rows]
            pen = self.GROUND_H - np.abs(sub)
            k = np.argmin(pen, axis=1)
            ar = np.arange(rows.size)
            sgn = np.sign(sub[ar, k])
            sgn[sgn == 0] = 1.0
            p[rows, k] = self.GROUND_C[k] + sgn * self.GROUND_H[k]
        # catch-basin cylinder
        r = np.hypot(p[:, 0], p[:, 1])
        inside = (r < self.BASIN_R) & (p[:, 2] > self.BASIN_Z0) & (p[:, 2] < self.BASIN_Z1)
        rows = np.nonzero(inside)[0]
        if rows.size:
            rr = r[rows]
            zz = p[rows, 2]
            d_up = self.BASIN_Z1 - zz
            d_dn = zz - self.BASIN_Z0
            d_rad = self.BASIN_R - rr
            choice = np.argmin(np.column_stack((d_up, d_dn, d_rad)), axis=1)
            up = choice == 0
            dn = choice == 1
            rad = choice == 2
            p[rows[up], 2] = self.BASIN_Z1
            p[rows[dn], 2] = self.BASIN_Z0
            rows_rad = rows[rad]
            if rows_rad.size:
                rr2 = r[rows_rad]
                scale = np.where(rr2 > 1e-9, self.BASIN_R / np.maximum(rr2, 1e-9), 1.0)
                p[rows_rad, 0] *= scale
                p[rows_rad, 1] *= scale

    def _sdf(self) -> np.ndarray:
        """Signed distance to the union of ground box and basin cylinder."""
        p = self.p
        q = np.abs(p - self.GROUND_C) - self.GROUND_H
        box = (np.linalg.norm(np.maximum(q, 0.0), axis=1)
               + np.minimum(np.max(q, axis=1), 0.0))
        zl = np.abs(p[:, 2] - 0.5 * (self.BASIN_Z0 + self.BASIN_Z1)) - 0.5 * (self.BASIN_Z1 - self.BASIN_Z0)
        rq = np.hypot(p[:, 0], p[:, 1]) - self.BASIN_R
        cyl = (np.minimum(np.maximum(rq, zl), 0.0)
               + np.hypot(np.maximum(rq, 0.0), np.maximum(zl, 0.0)))
        return np.minimum(box, cyl)

    # ------------------------------------------------------------------ step
    def _step(self) -> None:
        self.step += 1
        s = self.step
        if s % self.EMIT_EVERY == 0:
            self._emit()
        if s % self.REBUILD_EVERY == 0:
            self._rebuild()

        dt = self.SUB_DT
        self.v += self.gravity * dt
        self._viscosity()

        self.pprev = self.p.copy()
        self.p = self.p + self.v * dt
        for it in range(self.N_ITER):
            self._solve_distances(plastic=(it == 0))
            self._collide()
        self.v = (self.p - self.pprev) / dt

        # basal friction on contacting samples (surfaces are near-horizontal)
        contact = self._sdf() < self.CONTACT_EPS
        if np.any(contact):
            self.v[contact, 0] *= self.FRICTION
            self.v[contact, 1] *= self.FRICTION

        # safety clamp keeps everything finite
        sp = np.linalg.norm(self.v, axis=1)
        fast = sp > self.VMAX
        if np.any(fast):
            self.v[fast] *= (self.VMAX / sp[fast])[:, None]

    # ------------------------------------------------------------ diagnostics
    def _deformation_gradient(self) -> np.ndarray:
        n = len(self.p)
        F = np.tile(np.eye(3), (n, 1, 1))
        i, j = self.pi, self.pj
        if i.size == 0:
            return F
        d = np.linalg.norm(self.p[j] - self.p[i], axis=1)
        sel = d < self.r_neigh
        if not np.any(sel):
            return F
        i, j = i[sel], j[sel]
        dX = self.X[j] - self.X[i]
        dx = self.p[j] - self.p[i]
        w = 1.0 / (np.linalg.norm(dX, axis=1) + 1e-9)
        A = np.zeros((n, 3, 3))
        B = np.zeros((n, 3, 3))
        np.add.at(A, i, (dX * w[:, None])[:, :, None] * dX[:, None, :])
        np.add.at(B, i, (dx * w[:, None])[:, :, None] * dX[:, None, :])
        lam = 1e-8 + 1e-4 * self.spacing ** 2
        A += lam * np.eye(3)
        B += lam * np.eye(3)
        try:
            Ft = np.linalg.solve(A, np.transpose(B, (0, 2, 1)))
            Fs = np.transpose(Ft, (0, 2, 1))
        except np.linalg.LinAlgError:
            return F
        bad = ~np.isfinite(Fs).all(axis=(1, 2))
        Fs[bad] = np.eye(3)
        np.clip(Fs, -100.0, 100.0, out=Fs)
        return Fs

    def _frame(self, time: float) -> StateFrame:
        n = len(self.p)
        sd = self._sdf()
        near = np.nonzero(sd < 3.0 * self.spacing)[0]
        if near.size == 0:
            near = np.array([int(np.argmin(sd))])
        frame = StateFrame(
            float(time),
            self.p.copy(),
            self.v.copy(),
            mass=self.m.copy(),
            body_id=self.body.copy(),
            attributes={
                "mass_quantity": self.m.copy(),
                "deformation_gradient": self._deformation_gradient(),
                "particle_radius": np.full(n, 0.45 * self.spacing),
                "material_label": np.zeros(n, np.int32),
            },
            fields={
                "contact_signed_distance": sd[near].copy(),
                "cumulative_mass_source": np.array([self.emitted_mass]),
                "external_force": self.m[:, None] * self.gravity[None, :],
            },
            scalars={
                "contact_length_scale": float(self.spacing),
                "grid_spacing": float(self.spacing),
                "characteristic_length": float(self.char_len),
                "reference_velocity": float(self.ref_vel),
            },
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.SUB_DT))
        while self.step < target:
            self._step()
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        return {
            "scene": "129_lava_spout_periodic_folding",
            "substeps": int(self.step),
            "particles": int(len(self.p)),
            "constraints": int(self.pi.size),
            "emitted_mass": float(self.emitted_mass),
            "total_mass": float(np.sum(self.m)),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))) if len(self.v) else 0.0,
        }
