"""Hard Sphere Ground solver (scene 026).

Physically-based time stepping of a stiff elastic sphere (triangle surface
mesh) impacting a static ground plane:

* elasticity: network of Hookean springs on mesh edges (stretch) and on
  graph-distance-2 vertex pairs (bending / shear resistance), with viscous
  damping along each spring;
* additional rotation-invariant shape-matching stiffness (best-fit rigid
  frame via Kabsch, per-vertex spring toward the rigid-frame rest position)
  which keeps the stiff sphere shape stable and prevents inversion;
* ground contact handled with an impulse/penalty-free "postprocess" law:
  penetrations are projected out and normal impulses apply restitution,
  tangential impulses apply Coulomb friction;
* integrator: semi-implicit (symplectic) Euler with fixed substeps,
  fully deterministic (no randomness, no keyframes).

Only NumPy / SciPy are used; no external physics engine.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp

try:
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - fallback if contracts unavailable
    StateFrame = object


class BenchmarkSolver:
    SUB_DT = 1.0 / 6000.0  # fixed physics substep (s), stable for spring set
    REST_VEL_THRESH = 0.02  # m/s: below this, contact kills normal velocity

    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        self.seed = int(seed)
        self.cfg = cfg

        vertices = np.asarray(scene.vertices, dtype=float)
        faces = self._load_faces(cfg, vertices.shape[0])
        self.x0 = vertices.copy()
        self.n = vertices.shape[0]

        # --- physical parameters (frozen config) ---
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        total_mass = float(cfg.get("subject_mass", 2.0))
        self.mass = np.full(self.n, total_mass / self.n, dtype=float)
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.restitution = float(cfg.get("restitution", 0.5))
        self.friction = float(cfg.get("friction", 0.4))
        self.contact_scale = float(cfg.get("contact_length_scale", 0.04))
        stiffness = float(cfg.get("stiffness", 40.0))
        damping = float(cfg.get("damping", 0.5))
        v0 = np.asarray(cfg.get("initial_velocity", (0.0, 0.0, 0.0)), dtype=float)

        # --- spring network ---
        edges = self._unique_edges(faces)
        bend = self._distance2_pairs(edges)
        springs = np.vstack([edges, bend])
        self.n_edges = edges.shape[0]

        rest = self.x0[springs[:, 0]] - self.x0[springs[:, 1]]
        self.rest_len = np.linalg.norm(rest, axis=1)
        k_edge = 60.0 * stiffness
        k_bend = 0.25 * k_edge
        self.k_spring = np.concatenate(
            [np.full(edges.shape[0], k_edge), np.full(bend.shape[0], k_bend)]
        )
        self.c_spring = np.full(springs.shape[0], damping)

        # sparse incidence matrix: column m has +1 at head, -1 at tail
        m_idx = np.arange(springs.shape[0])
        rows = np.concatenate([springs[:, 0], springs[:, 1]])
        cols = np.concatenate([m_idx, m_idx])
        vals = np.concatenate([np.ones(springs.shape[0]), -np.ones(springs.shape[0])])
        self.D = sp.csr_matrix((vals, (rows, cols)), shape=(self.n, springs.shape[0]))
        self.DT = self.D.T.tocsr()
        self.n_springs = springs.shape[0]

        # --- shape matching (rotation-invariant stiffness) ---
        self.rest_centroid = self.x0.mean(axis=0)
        self.P = self.x0 - self.rest_centroid  # centered rest positions
        self.k_shape = 2000.0  # per-vertex spring toward rigid-frame target
        self.c_shape = damping

        # --- state ---
        self.x = self.x0.copy()
        self.v = np.tile(v0, (self.n, 1))
        self.steps = 0
        self.time = 0.0

        # one-ring neighbors (for per-vertex deformation gradients)
        adj = [set() for _ in range(self.n)]
        for a, b in edges:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
        self.neighbors = [np.array(sorted(a), dtype=np.int64) for a in adj]

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_faces(cfg, n_vertices):
        candidates = []
        asset = cfg.get("subject_asset")
        if asset:
            root = Path(__file__).resolve().parents[1]
            candidates.append(root / asset)
            candidates.append(Path(__file__).resolve().parent / asset)
        for path in candidates:
            try:
                if path.exists():
                    with np.load(path) as data:
                        faces = np.asarray(data["faces"], dtype=np.int64)
                    if faces.ndim == 2 and faces.shape[1] == 3:
                        return faces
            except Exception:
                continue
        # no connectivity found; return empty edge set (physics still runs
        # through the shape-matching stiffness alone)
        return np.zeros((0, 3), dtype=np.int64)

    @staticmethod
    def _unique_edges(faces):
        if faces.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int64)
        e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = np.sort(e, axis=1)
        e = np.unique(e, axis=0)
        return e.astype(np.int64)

    def _distance2_pairs(self, edges):
        if edges.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int64)
        adj = [set() for _ in range(self.n)]
        edge_set = set()
        for a, b in edges:
            a, b = int(a), int(b)
            adj[a].add(b)
            adj[b].add(a)
            edge_set.add((min(a, b), max(a, b)))
        pairs = set()
        for k in range(self.n):
            nb = sorted(adj[k])
            for i in range(len(nb)):
                for j in range(i + 1, len(nb)):
                    a, b = nb[i], nb[j]
                    key = (min(a, b), max(a, b))
                    if key not in edge_set:
                        pairs.add(key)
        if not pairs:
            return np.zeros((0, 2), dtype=np.int64)
        return np.array(sorted(pairs), dtype=np.int64)

    # ------------------------------------------------------------------
    # physics
    # ------------------------------------------------------------------
    def _best_fit_rotation(self, x):
        q = x - x.mean(axis=0)
        m = self.P.T @ q  # sum_i p_i q_i^T
        u, _, vt = np.linalg.svd(m)
        d = np.eye(3)
        d[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
        if d[2, 2] == 0.0:
            d[2, 2] = 1.0
        return vt.T @ d @ u.T

    def _substep(self, dt):
        x, v, m = self.x, self.v, self.mass

        # spring forces (stretch + bending) with viscous damping
        d = np.asarray(self.DT @ x).reshape(self.n_springs, 3)
        length = np.linalg.norm(d, axis=1)
        length = np.maximum(length, 1e-12)
        dhat = d / length[:, None]
        fspring = (self.k_spring * (length - self.rest_len))[:, None] * dhat

        dv = np.asarray(self.DT @ v).reshape(self.n_springs, 3)
        rel = np.einsum("ij,ij->i", dv, dhat)
        fdamp = -(self.c_spring * rel)[:, None] * dhat

        fint = np.asarray(self.D @ (fspring + fdamp)).reshape(self.n, 3)

        # rotation-invariant shape-matching stiffness
        rot = self._best_fit_rotation(x)
        target = x.mean(axis=0) + self.P @ rot.T
        v_mean = v.mean(axis=0)
        fshape = self.k_shape * (target - x) - self.c_shape * (v - v_mean)

        force = fint + fshape + m[:, None] * self.gravity[None, :]

        # semi-implicit Euler
        v = v + (force / m[:, None]) * dt
        x = x + v * dt

        # ground contact (impulse / postprocess): project penetration,
        # restitution on the normal, Coulomb friction on the tangential.
        pen = x[:, 2] < self.ground_z
        if np.any(pen):
            x[pen, 2] = self.ground_z
            vz = v[pen, 2]
            new_vz = np.where(
                np.abs(vz) > self.REST_VEL_THRESH,
                -self.restitution * vz,
                0.0,
            )
            djn = m[pen] * (new_vz - vz)  # >= 0
            v[pen, 2] = new_vz
            vt = v[pen, :2]
            vt_norm = np.linalg.norm(vt, axis=1)
            moving = vt_norm > 1e-9
            jt_max = self.friction * djn
            jt = np.minimum(m[pen] * vt_norm, jt_max)
            scale = np.where(moving, 1.0 - jt / np.maximum(m[pen] * vt_norm, 1e-12), 1.0)
            v[pen, 0] = np.where(moving, vt[:, 0] * scale, 0.0)
            v[pen, 1] = np.where(moving, vt[:, 1] * scale, 0.0)

        self.x, self.v = x, v

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _deformation_gradients(self, x):
        grad = np.empty((self.n, 3, 3))
        for i in range(self.n):
            nb = self.neighbors[i]
            if nb.size < 3:
                grad[i] = np.eye(3)
                continue
            xm = self.x0[nb] - self.x0[i]
            ym = x[nb] - x[i]
            a = xm.T @ xm
            reg = 1e-9 * (1.0 + np.trace(a) / 3.0)
            a = a + reg * np.eye(3)
            b = ym.T @ xm
            grad[i] = np.linalg.solve(a.T, b.T).T
        return grad

    def _contact_gaps(self, x):
        gaps = x[:, 2] - self.ground_z
        mask = gaps < 3.0 * self.contact_scale
        if not np.any(mask):
            idx = np.array([int(np.argmin(gaps))])
            return gaps[idx]
        return gaps[mask]

    def _build_frame(self, t):
        grad = self._deformation_gradients(self.x)
        gaps = self._contact_gaps(self.x)
        frame = StateFrame(
            time=float(max(t, 0.0)),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": grad,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "contact_signed_distance": np.asarray(gaps, dtype=float),
            },
            scalars={"contact_length_scale": self.contact_scale},
        )
        return frame

    # ------------------------------------------------------------------
    # solver API
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        t = float(time)
        target_steps = int(math.ceil(t / self.SUB_DT - 1e-9))
        target_steps = max(target_steps, self.steps)
        while self.steps < target_steps:
            self._substep(self.SUB_DT)
            self.steps += 1
        self.time = self.steps * self.SUB_DT
        frame = self._build_frame(t)
        if hasattr(frame, "validate"):
            frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps": int(self.steps),
            "sub_dt": self.SUB_DT,
            "sim_time": float(self.time),
            "n_vertices": int(self.n),
            "n_springs": int(self.n_springs),
            "min_gap": float(np.min(self.x[:, 2] - self.ground_z)),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))) if self.n else 0.0,
        }
