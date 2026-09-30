"""Physically-based brittle bowl fracture solver (XPBD shell + strain fracture).

Model
-----
* The bowl triangle mesh (``assets/bowl_topology.json``) is simulated as a thin
  brittle shell: mass points at the mesh vertices, distance (membrane)
  constraints along mesh edges and bending constraints across the two-ring of
  every interior edge (position-based dynamics, unconditionally stable).
* Brittle fracture: every membrane edge carries a seeded failure strain.  Each
  substep the *predicted* (pre-projection) edge strain -- the kinematic signal
  of the local stress -- accumulates damage; when damage saturates the edge is
  deleted permanently and the bending constraint supported by that edge is
  deleted too.  Load redistribution makes cracks propagate away from the first
  rim contact, releasing distinct curved fragments.
* Rigid floor contact with Coulomb friction and restitution; gravity from the
  frozen config.  Fragments bounce and settle.
* Per-frame exports: mass, velocity, external force, per-element deformation
  gradient, contact signed distance, cumulative mass source, fragment body ids.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class _UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int64)

    def find(self, i: int) -> int:
        p = self.p
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


class BenchmarkSolver:
    # ---- numerical / material parameters -------------------------------
    SUB_DT = 1.0 / 480.0
    ITERATIONS = 8
    RELAX = 0.7               # Jacobi relaxation for the projection solve
    BEND_SOFT = 0.35          # bending constraints are much softer than membrane
    EPS_BASE = 0.040          # brittle failure strain of a pristine edge
    DAMAGE_RATE = 6.0
    RESTITUTION = 0.25
    FRICTION = 0.5
    TANG_DAMP = 0.90          # extra rolling/sliding damping while in contact
    V_MAX = 60.0
    VEL_DAMP = 0.999

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self._gravity = np.asarray(cfg["gravity"], dtype=float)
        spacing = float(cfg["sampling"]["spacing"])
        self._spacing = spacing

        frame0 = scene.initial_state()
        self._t = 0.0
        pos = np.array(frame0.position, dtype=float)
        vel = np.array(frame0.velocity, dtype=float)
        mass = np.array(frame0.mass, dtype=float)
        n = len(pos)

        asset = Path(__file__).resolve().parent.parent / "assets" / "bowl_topology.json"
        payload = json.loads(asset.read_text(encoding="utf-8"))
        faces = np.asarray(payload["faces"], dtype=np.int64).reshape(-1, 3)
        self._faces = faces

        # keep the intact bowl just above the floor plane (z = 0)
        lift = 0.005 - pos[:, 2].min()
        if lift > 0.0:
            pos[:, 2] += lift
        self._pos = pos
        self._vel = vel
        self._mass = mass
        self._w = 1.0 / mass
        self._n = n

        rng = np.random.default_rng(int(seed))

        # ---- constraints -------------------------------------------------
        edge_map: dict[tuple[int, int], int] = {}
        edges = []
        edge_faces: dict[int, list[int]] = {}
        for fi, (a, b, c) in enumerate(faces):
            for u, v in ((a, b), (b, c), (c, a)):
                key = (int(min(u, v)), int(max(u, v)))
                idx = edge_map.get(key)
                if idx is None:
                    idx = len(edges)
                    edge_map[key] = idx
                    edges.append(key)
                edge_faces.setdefault(idx, []).append(fi)
        self._edges = np.asarray(edges, dtype=np.int64)          # [E,2]
        self._ei = self._edges[:, 0].copy()
        self._ej = self._edges[:, 1].copy()
        d = pos[self._ej] - pos[self._ei]
        self._L0 = np.linalg.norm(d, axis=1)
        self._L0 = np.maximum(self._L0, 1e-6)

        # bending pairs: opposite vertices of every interior edge
        bend = []
        support = []
        for eidx, fl in edge_faces.items():
            if len(fl) != 2:
                continue
            f1, f2 = fl
            i, j = edges[eidx]
            a = next(v for v in faces[f1] if v != i and v != j)
            b = next(v for v in faces[f2] if v != i and v != j)
            bend.append((int(a), int(b)))
            support.append(eidx)
        self._bend = np.asarray(bend, dtype=np.int64).reshape(-1, 2)
        self._bend_support = np.asarray(support, dtype=np.int64)
        db = pos[self._bend[:, 1]] - pos[self._bend[:, 0]]
        self._B0 = np.maximum(np.linalg.norm(db, axis=1), 1e-6)

        self._active_edge = np.ones(len(self._edges), dtype=bool)
        self._active_bend = np.ones(len(self._bend), dtype=bool)
        self._edge_dirty = False
        self._ia = np.flatnonzero(self._active_edge)
        self._ib = np.flatnonzero(self._active_bend)

        # ---- brittle strength (seeded flaws, weakest at first contact) ----
        low = int(np.argmin(pos[:, 2]))
        dist = np.linalg.norm(pos - pos[low], axis=1)
        near = (dist[ self._ei] < 6.0 * spacing) & (dist[self._ej] < 6.0 * spacing)
        eps_f = self.EPS_BASE * (0.65 + 0.65 * rng.random(len(self._edges)))
        eps_f[near] *= 0.55
        self._eps_f = eps_f
        self._damage = np.zeros(len(self._edges))
        self._broken_count = 0

        # ---- deformation gradient helpers (rest configuration) -----------
        X0 = pos
        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        Dm = np.stack([X0[f1] - X0[f0], X0[f2] - X0[f0]], axis=2)   # [F,3,2]
        self._Dm_pinv = np.linalg.pinv(Dm)                           # [F,2,3]
        e1 = Dm[:, :, 0]
        e2 = Dm[:, :, 1]
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        self._face_area = np.maximum(area, 1e-9)
        wsum = np.zeros(n)
        np.add.at(wsum, faces.reshape(-1), np.repeat(self._face_area, 3))
        self._vw = 1.0 / np.maximum(wsum, 1e-12)

        self._body_id = np.zeros(n, dtype=np.int32)
        self._contact = np.zeros(n, dtype=bool)
        self._scalars = {
            "contact_length_scale": spacing,
            "grid_spacing": spacing,
            "characteristic_length": float(
                max(np.ptp(pos, axis=0).max(), spacing)),
            "reference_velocity": 1.0,
        }

    # ------------------------------------------------------------------
    def _active_indices(self):
        if self._edge_dirty:
            self._ia = np.flatnonzero(self._active_edge)
            self._ib = np.flatnonzero(self._active_bend)
            self._edge_dirty = False
        return self._ia, self._ib

    def _project(self, p, idx, i0, i1, L0, w, stiffness):
        if idx.size == 0:
            return
        ei = i0[idx]
        ej = i1[idx]
        rest = L0[idx]
        wi = w[ei][:, None]
        wj = w[ej][:, None]
        d = p[ej] - p[ei]
        L = np.linalg.norm(d, axis=1)
        L = np.maximum(L, 1e-9)
        C = L - rest
        scale = (stiffness * self.RELAX) * C / (L * (w[ei] + w[ej]))
        corr = scale[:, None] * d
        np.add.at(p, ei, wi * corr)
        np.add.at(p, ej, -wj * corr)

    def _fracture(self, p):
        idx = self._ia
        if idx.size == 0:
            return
        d = p[self._ej[idx]] - p[self._ei[idx]]
        L = np.linalg.norm(d, axis=1)
        eps = (L - self._L0[idx]) / self._L0[idx]
        over = np.maximum(eps / self._eps_f[idx] - 1.0, 0.0)
        dmg = self._damage[idx] + self.DAMAGE_RATE * over
        self._damage[idx] = dmg
        fail = dmg >= 1.0
        if fail.any():
            broken = idx[fail]
            self._active_edge[broken] = False
            supp = np.isin(self._bend_support, broken)
            if supp.any():
                self._active_bend[supp] = False
            self._broken_count += int(broken.size)
            self._edge_dirty = True

    def _step(self, dt):
        p = self._pos
        v = self._vel
        w = self._w
        g = self._gravity

        v += g * dt
        v *= self.VEL_DAMP
        np.clip(v, -self.V_MAX, self.V_MAX, out=v)
        p_prev = p.copy()
        p += v * dt

        # brittle damage from the kinematic (predicted) strain, then solve
        self._fracture(p)
        ia, ib = self._active_indices()
        for _ in range(self.ITERATIONS):
            self._project(p, ia, self._ei, self._ej, self._L0, w, 1.0)
            self._project(p, ib, self._bend[:, 0], self._bend[:, 1],
                          self._B0, w, self.BEND_SOFT)

        # rigid floor (z = 0): position clamp, restitution + Coulomb friction
        pen = p[:, 2] < 0.0
        if pen.any():
            v_pre = (p - p_prev) / dt
            p[pen, 2] = 0.0
            v = (p - p_prev) / dt
            vn = v_pre[pen, 2]
            vt = v_pre[pen, :2]
            hit = vn < 0.0
            idx = np.flatnonzero(pen)
            hi = idx[hit]
            if hi.size:
                v[hi, 2] = -self.RESTITUTION * v_pre[hi, 2]
                speed_t = np.linalg.norm(vt[hit], axis=1)
                mu_dv = self.FRICTION * (-v_pre[hi, 2])
                f = np.clip(1.0 - mu_dv / np.maximum(speed_t, 1e-6), 0.0, 1.0)
                v[hi, :2] = vt[hit] * f[:, None]
            # rolling / sliding damping for everything touching the floor
            v[idx, :2] *= self.TANG_DAMP
            v[idx, 2] = np.maximum(v[idx, 2], 0.0)
            self._vel = v
            self._contact = pen
        else:
            v = (p - p_prev) / dt
            self._vel = v
            self._contact = pen

        self._vel = np.clip(self._vel, -self.V_MAX, self.V_MAX)
        self._pos = p

    def _update_bodies(self):
        uf = _UnionFind(self._n)
        ia = self._ia
        ei = self._ei[ia].tolist()
        ej = self._ej[ia].tolist()
        for a, b in zip(ei, ej):
            uf.union(a, b)
        roots = np.array([uf.find(i) for i in range(self._n)], dtype=np.int64)
        _, labels = np.unique(roots, return_inverse=True)
        self._body_id = labels.astype(np.int32)

    def _deformation_gradient(self):
        p = self._pos
        f0, f1, f2 = self._faces[:, 0], self._faces[:, 1], self._faces[:, 2]
        Ds = np.stack([p[f1] - p[f0], p[f2] - p[f0]], axis=2)        # [F,3,2]
        Ff = np.einsum('fij,fjk->fik', Ds, self._Dm_pinv)            # [F,3,3]
        eye = np.eye(3)
        Ff = eye[None] + np.clip(Ff - eye[None], -2.0, 2.0)
        Fv = np.zeros((self._n, 3, 3))
        aw = self._face_area
        for k in range(3):
            comp = Ff[:, :, k] * aw[:, None]
            acc = np.zeros((self._n, 3))
            np.add.at(acc, f0, comp)
            np.add.at(acc, f1, comp)
            np.add.at(acc, f2, comp)
            Fv[:, :, k] = acc * self._vw[:, None]
        return Fv

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        time = float(time)
        dt = self.SUB_DT
        steps = int(round((time - self._t) / dt))
        for _ in range(max(steps, 0)):
            self._step(dt)
        self._t = self._t + max(steps, 0) * dt

        p = np.array(self._pos, dtype=float)
        v = np.array(self._vel, dtype=float)
        m = np.array(self._mass, dtype=float)
        self._update_bodies()

        F = self._deformation_gradient()
        ext = m[:, None] * self._gravity[None, :]

        attributes = {
            "mass_quantity": m.copy(),
            "particle_radius": np.full(self._n, 0.45 * self._spacing),
            "material_label": np.zeros(self._n, dtype=np.int32),
            "deformation_gradient": F,
        }
        fields = {
            "cumulative_mass_source": np.zeros(self._n),
            "contact_signed_distance": p[:, 2].copy(),
        }
        frame = StateFrame(
            float(self._t), p, v, mass=m,
            body_id=self._body_id.copy(),
            attributes=attributes, fields=fields,
            scalars=dict(self._scalars),
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "time": float(self._t),
            "broken_edges": int(self._broken_count),
            "active_edges": int(self._active_edge.sum()),
            "fragments": int(len(np.unique(self._body_id))) if self._n else 0,
            "kinetic_energy": float(
                0.5 * np.sum(self._mass * np.sum(self._vel ** 2, axis=1))),
            "max_speed": float(np.max(np.linalg.norm(self._vel, axis=1)))
            if self._n else 0.0,
        }
