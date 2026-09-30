"""Physics solver for scene 161_pinned_denim_gravity_sag.

Inextensible cloth (denim) pinned at its two upper corners, sagging under
gravity.  The cloth is integrated with a substepped position-based dynamics
(Verlet / strain-limiting) scheme:

  * semi-implicit Euler prediction with gravity and linear drag,
  * repeated Jacobi-projected distance constraints (structural, shear and
    bending edges) which enforce near-inextensibility,
  * pinned vertices held exactly at their prescribed rest positions,
  * one-sided ground contact with friction.

No external physics engine is used; only NumPy.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict

import numpy as np

from worldsolver.contracts import StateFrame

try:  # optional, only used to read the frozen scene geometry
    import yaml  # type: ignore
    _HAVE_YAML = True
except Exception:  # pragma: no cover
    _HAVE_YAML = False


class BenchmarkSolver:
    """Substepped PBD cloth solver with pinned upper corners."""

    SUBSTEP_DT = 1.0e-3
    ITERS = 30            # constraint projections per substep
    RELAX = 0.5           # Jacobi relaxation
    DAMPING = 2.0         # linear velocity drag [1/s]
    FRICTION = 0.3        # tangential velocity factor while in contact
    NEAR_CONTACT = 0.12   # band for reported signed distances [m]

    K_STRUCT = 1.0
    K_SHEAR = 0.9
    K_BEND = 0.4

    def __init__(self) -> None:
        self._t = 0.0
        self._steps_done = 0
        self._diag: dict[str, object] = {}

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        f0 = scene.initial_state()
        x0 = np.array(f0.position, dtype=float)
        n = len(x0)
        self._n = n
        self._mass = np.array(f0.mass, dtype=float)
        self._gravity = np.asarray(cfg["gravity"], dtype=float)

        cloth = cfg["cloth"]
        self._thickness = float(cloth.get("thickness", 0.015))
        self._contact_length_scale = self._thickness

        # --- pin set: the two upper corners of the sheet -------------------
        top = x0[:, 2].max()
        top_idx = np.where(x0[:, 2] > top - 1.0e-9)[0]
        i_lo = int(top_idx[np.argmin(x0[top_idx, 0])])
        i_hi = int(top_idx[np.argmax(x0[top_idx, 0])])
        self._pinned = np.array(sorted({i_lo, i_hi}), dtype=np.int64)
        self._diag["pinned_vertices"] = [int(i) for i in self._pinned]

        # --- rest state -----------------------------------------------------
        self._rest = x0.copy()

        # tiny deterministic out-of-plane imperfection (material unevenness)
        # so that compressive regions buckle into folds instead of staying on
        # the exactly mirror-symmetric planar configuration.
        extents = x0.max(axis=0) - x0.min(axis=0)
        normal_axis = int(np.argmin(extents))
        min_feat = float(min(extents[extents > 1e-9].min(), self._thickness * 10.0)) \
            if np.any(extents > 1e-9) else 0.01
        rng = np.random.default_rng(int(seed))
        pert = rng.normal(0.0, 0.01 * min_feat, size=(n,))
        pert[self._pinned] = 0.0
        x = x0.copy()
        x[:, normal_axis] = x[:, normal_axis] + pert

        # --- topology -------------------------------------------------------
        faces = self._load_faces(cfg, n)
        self._faces = np.asarray(faces, dtype=np.int64)

        # material (reference) coordinates on the initial flat sheet
        plane = str(cloth.get("plane", "xy")).lower()
        if plane == "xz":
            mat = np.column_stack((x0[:, 0], x0[:, 2]))
        else:
            mat = np.column_stack((x0[:, 0], x0[:, 1]))
        self._mat = mat

        # --- constraint edges ------------------------------------------------
        eset = set()
        for f in self._faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            eset.add((min(a, b), max(a, b)))
            eset.add((min(b, c), max(b, c)))
            eset.add((min(a, c), max(a, c)))
        edge_list = sorted(eset)
        E = np.array(edge_list, dtype=np.int64)
        rest_len = np.linalg.norm(x0[E[:, 0]] - x0[E[:, 1]], axis=1)

        min_len = float(rest_len.min())
        struct_mask = rest_len < 0.95 * math.sqrt(2.0) * min_len

        # bending edges: collinear extensions of structural edges (skip-2)
        adj: dict[int, set[int]] = defaultdict(set)
        for k in range(len(E)):
            if struct_mask[k]:
                a, b = int(E[k, 0]), int(E[k, 1])
                adj[a].add(b)
                adj[b].add(a)
        struct_pairs = [(int(E[k, 0]), int(E[k, 1])) for k in range(len(E))
                        if struct_mask[k]]
        ordered = struct_pairs + [(b, a) for (a, b) in struct_pairs]
        bend = set()
        for a, b in ordered:
            la = float(np.linalg.norm(x0[a] - x0[b]))
            for c in adj[b]:
                if c == a:
                    continue
                lb = float(np.linalg.norm(x0[b] - x0[c]))
                lac = float(np.linalg.norm(x0[a] - x0[c]))
                if lac > 1e-9 and abs(lac - (la + lb)) < 0.1 * min(la, lb):
                    key = (min(a, c), max(a, c))
                    if key not in eset:
                        bend.add(key)

        bend_list = sorted(bend)
        shear_list = [edge_list[k] for k in range(len(edge_list))
                      if not struct_mask[k]]

        parts = [(edge_list, self.K_STRUCT), (shear_list, self.K_SHEAR),
                 (bend_list, self.K_BEND)]
        Eall, kall = [], []
        for lst, k in parts:
            for (a, b) in lst:
                Eall.append((a, b))
                kall.append(k)
        self._E0 = np.array([p[0] for p in Eall], dtype=np.int64)
        self._E1 = np.array([p[1] for p in Eall], dtype=np.int64)
        self._kedge = np.array(kall, dtype=float)
        self._rest_len = np.linalg.norm(
            x0[self._E0] - x0[self._E1], axis=1)

        # mass-weighted (pinned = infinite mass) correction fractions
        w = np.where(np.isin(np.arange(n), self._pinned), 0.0, 1.0 / self._mass)
        wi = w[self._E0]
        wj = w[self._E1]
        tot = wi + wj
        tot = np.where(tot > 0.0, tot, 1.0)
        self._fi = np.where(tot > 0.0, wi / tot, 0.0)
        self._fj = np.where(tot > 0.0, wj / tot, 0.0)
        degree = np.zeros(n)
        np.add.at(degree, self._E0, (self._fi > 0.0).astype(float))
        np.add.at(degree, self._E1, (self._fj > 0.0).astype(float))
        self._inv_degree = 1.0 / np.maximum(degree, 1.0)

        # --- ground ----------------------------------------------------------
        self._ground_c = np.array([0.0, 0.0, -0.05])
        self._ground_h = np.array([2.4, 2.0, 0.05])
        geo = self._load_geometry()
        if geo is not None:
            self._ground_c = geo[0]
            self._ground_h = geo[1]
        self._ground_top = float(self._ground_c[2] + self._ground_h[2])
        self._contact_z = self._ground_top + 0.5 * self._thickness

        # --- state ------------------------------------------------------------
        self._x = x
        self._v = np.zeros_like(x)
        self._t = 0.0
        self._steps_done = 0
        self._f0_attrs = dict(f0.attributes)

    # ------------------------------------------------------------- file loads
    @staticmethod
    def _load_faces(cfg, n) -> list[list[int]]:
        path = os.path.join("assets", "cloth_topology.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as fh:
                    data = json.load(fh)
                faces = data.get("faces")
                if faces and len(data.get("vertices", [])) == n:
                    return [list(map(int, f)) for f in faces]
            except Exception:
                pass
        nx, ny = (int(v) for v in cfg["cloth"]["grid_shape"])
        if nx * ny != n:
            nx = ny = int(round(math.sqrt(n)))
        faces = []
        for j in range(ny):
            for i in range(nx - 1):
                a = j * nx + i
                b = a + 1
                c = a + nx
                d = c + 1
                faces.append([a, b, d])
                faces.append([a, d, c])
        return faces

    @staticmethod
    def _load_geometry():
        if not _HAVE_YAML or not os.path.exists("scene_geometry.yaml"):
            return None
        try:
            with open("scene_geometry.yaml", "r") as fh:
                data = yaml.safe_load(fh)
            for obj in data.get("objects", []) or []:
                if obj.get("id") == "ground":
                    geo = obj.get("geometry", {}) or {}
                    if geo.get("type") == "box":
                        c = np.asarray(obj["transform"]["translation"], float)
                        h = np.asarray(geo["half_extents"], float)
                        return c, h
        except Exception:
            return None
        return None

    # ---------------------------------------------------------------- physics
    def _project(self, x: np.ndarray) -> np.ndarray:
        E0, E1 = self._E0, self._E1
        for _ in range(self.ITERS):
            d = x[E0] - x[E1]
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            dist = np.maximum(dist, 1.0e-12)
            C = dist - self._rest_len
            s = (self._kedge * C / dist)[:, None]
            corr = d * s
            dx = np.zeros_like(x)
            np.add.at(dx, E0, -self._fi[:, None] * corr)
            np.add.at(dx, E1, self._fj[:, None] * corr)
            x = x + (self.RELAX * self._inv_degree)[:, None] * dx
            # pinned vertices stay exactly at their prescribed positions
            x[self._pinned] = self._rest[self._pinned]
            # one-sided ground contact
            below = x[:, 2] < self._contact_z
            if below.any():
                x[below, 2] = self._contact_z
        return x

    def _substep(self) -> None:
        dt = self.SUBSTEP_DT
        v = self._v + self._gravity * dt
        v *= math.exp(-self.DAMPING * dt)
        xp = self._x + v * dt
        xn = self._project(xp)
        v = (xn - self._x) / dt
        # contact friction / non-penetration on velocity
        near = xn[:, 2] < self._contact_z + 0.02
        if near.any():
            v[near, 0] *= self.FRICTION
            v[near, 1] *= self.FRICTION
            v[near, 2] = np.maximum(v[near, 2], 0.0)
        v[self._pinned] = 0.0
        self._v = np.clip(v, -30.0, 30.0)
        self._x = xn

    def advance_to(self, time: float) -> StateFrame:
        t = float(time)
        if t > self._t:
            target = int(round(t / self.SUBSTEP_DT))
            target = max(target, self._steps_done)
            while self._steps_done < target:
                self._substep()
                self._steps_done += 1
            self._t = self._steps_done * self.SUBSTEP_DT
        return self._frame(t)

    # ----------------------------------------------------------------- output
    def _box_sdf(self, p: np.ndarray) -> np.ndarray:
        q = np.abs(p - self._ground_c[None, :]) - self._ground_h[None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def _deformation_gradient(self, pos: np.ndarray) -> np.ndarray:
        """Per-vertex in-plane (2x2) deformation gradient in material coords."""
        f = self._faces
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        pa, pb, pc = pos[a], pos[b], pos[c]
        ma, mb, mc = self._mat[a], self._mat[b], self._mat[c]
        e1 = pb - pa                       # (T,3) current edge vectors
        e2 = pc - pa
        m1 = mb - ma                       # (T,2) material edge vectors
        m2 = mc - ma
        det = m1[:, 0] * m2[:, 1] - m1[:, 1] * m2[:, 0]
        sign = np.where(det >= 0.0, 1.0, -1.0)
        det = np.maximum(np.abs(det), 1.0e-12) * sign
        inv = 1.0 / det
        # F = Ds @ inv(Dm);  columns f1, f2
        f1 = (e1 * (m2[:, 1] * inv)[:, None]
              + e2 * (-m1[:, 1] * inv)[:, None])
        f2 = (e1 * (-m2[:, 0] * inv)[:, None]
              + e2 * (m1[:, 0] * inv)[:, None])
        # local tangent frame of the deformed triangle
        nrm = np.cross(e1, e2)
        nrm = nrm / np.maximum(
            np.linalg.norm(nrm, axis=1), 1.0e-12)[:, None]
        t1 = e1 / np.maximum(np.linalg.norm(e1, axis=1), 1.0e-12)[:, None]
        t2 = np.cross(nrm, t1)
        G = np.empty((len(f), 2, 2))
        G[:, 0, 0] = np.einsum("ij,ij->i", f1, t1)
        G[:, 0, 1] = np.einsum("ij,ij->i", f2, t1)
        G[:, 1, 0] = np.einsum("ij,ij->i", f1, t2)
        G[:, 1, 1] = np.einsum("ij,ij->i", f2, t2)
        # scatter-average to vertices
        Vg = np.zeros((self._n, 2, 2))
        np.add.at(Vg, a, G)
        np.add.at(Vg, b, G)
        np.add.at(Vg, c, G)
        cnt = np.zeros(self._n)
        for col in (a, b, c):
            np.add.at(cnt, col, 1.0)
        Vg /= np.maximum(cnt, 1.0)[:, None, None]
        return Vg

    def _frame(self, time: float) -> StateFrame:
        pos = self._x.copy()
        vel = self._v.copy()
        attrs = {
            "deformation_gradient": self._deformation_gradient(pos),
            "mass_quantity": self._mass.copy(),
        }
        bv = self._f0_attrs.get("boundary_vertex")
        if bv is not None:
            attrs["boundary_vertex"] = np.asarray(bv).copy()
        sd = self._box_sdf(pos)
        near = sd < self.NEAR_CONTACT
        fields = {
            "contact_signed_distance": sd[near],
            "cumulative_mass_source": np.zeros(1),
            "external_force": self._mass[:, None] * self._gravity[None, :],
        }
        scalars = {"contact_length_scale": float(self._contact_length_scale)}
        # diagnostics bookkeeping
        d = np.linalg.norm(pos[self._E0] - pos[self._E1], axis=1)
        stretch = float(np.mean(np.abs(d / self._rest_len - 1.0)))
        self._diag = {
            "time": float(time),
            "substeps": int(self._steps_done),
            "pinned_vertices": [int(i) for i in self._pinned],
            "mean_edge_strain": stretch,
            "max_speed": float(np.max(np.linalg.norm(vel, axis=1))),
            "contact_samples": int(np.count_nonzero(near)),
        }
        return StateFrame(
            time, pos, vel,
            mass=self._mass.copy(),
            body_id=np.zeros(self._n, dtype=np.int32),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return dict(self._diag)
