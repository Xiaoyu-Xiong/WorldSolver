"""Physically-based solver for the falling deformable torus (ground contact +
self-collision).

Method overview
---------------
* The subject is the triangle-mesh torus from ``assets/subject.npz`` (N=1536
  sample points = mesh vertices, lumped masses summing to ``subject_mass``).
* Elasticity: a mass-spring network built from mesh edges (structural) and
  two-hop neighborhood pairs (bending/shape keeping).  Spring constants are
  derived from the configured ``stiffness``; per-spring damping from
  ``damping``.
* Time integration: linearly-implicit (backward) Euler.  Each substep solves

      A v' = M v + dt * (f_gravity + f_spring(x) + f_self(x)),
      A    = M + dt^2 * K(x) + dt * C,

  where K contains the (negative-semidefinite) spring/self-contact Jacobians
  (-k u u^T blocks) and C the velocity-damping Jacobians.  This is
  unconditionally stable for the stiff spring terms, an optimization-style
  large-time-step integrator in the spirit of the reference paper.
* Ground contact: constraint projection (as in the paper's "constraint ground
  collision"): positions clamped to z >= ground_z, normal velocity reflected
  with ``restitution`` (below a rest threshold it is killed), Coulomb friction
  with coefficient ``friction`` applied to the tangential velocity using the
  normal impulse bound.
* Self-collision: penalty repulsion (paper's "penalty self-collision") for
  vertex pairs closer than ``contact_length_scale`` whose rest-configuration
  graph distance exceeds three hops (found each substep with a KD-tree).

Required trajectory outputs are produced per StateFrame:
  attributes.deformation_gradient  [N,3,3] least-squares per-vertex F
  attributes.mass_quantity         [N]
  fields.contact_signed_distance   [N] signed gap to ground (pre-projection,
                                   negative = penetration)
  fields.cumulative_mass_source    [1] (no mass added/removed)
  fields.external_force            [N,3] gravity (contact excluded)
  scalars.contact_length_scale     frozen length scale
  state.mass, state.velocity       [N], [N,3]
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        X0 = np.asarray(scene.vertices, dtype=float).copy()
        from pathlib import Path
        try:
            scene_dir = Path(scene.__class__.__init__.__code__.co_filename).resolve().parent
        except Exception:
            scene_dir = Path.cwd()
        asset = scene_dir / cfg["subject_asset"]
        if not asset.exists():
            asset = Path(cfg["subject_asset"])
        with np.load(asset, allow_pickle=False) as data:
            faces = np.asarray(data["faces"], dtype=np.int64)
        self.X0 = X0
        self.faces = faces
        n = len(X0)
        self.n = n

        self.duration = float(cfg["duration"])
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.mu = float(cfg.get("friction", 0.5))
        self.restitution = float(cfg.get("restitution", 0.2))
        self.h_contact = float(cfg.get("contact_length_scale", 0.04))
        stiffness = float(cfg.get("stiffness", 13.0))
        damping = float(cfg.get("damping", 1.2))

        total_mass = float(cfg["subject_mass"])
        self.mass = np.full(n, total_mass / n)
        self.m_node = total_mass / n

        # spring constants calibrated from configured stiffness
        self.k_edge = stiffness * 30.0
        self.k_bend = 0.2 * self.k_edge
        self.k_self = 1.0 * self.k_edge
        self.c_edge = damping * np.sqrt(self.k_edge * self.m_node)
        self.c_bend = damping * np.sqrt(self.k_bend * self.m_node)
        self.c_self = 0.5 * damping * np.sqrt(self.k_self * self.m_node)
        self.c_glob = 0.10 * self.m_node  # light air drag -> settling

        # ---------------- topology: edges and bending (2-hop) pairs
        e = np.unique(np.sort(np.vstack(
            [faces[:, :2], faces[:, 1:], faces[:, ::2]]), axis=1), axis=0)
        self.edges = e
        A = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        A = A + A.T
        A.data[:] = 1.0
        I = sp.identity(n, format="csr")
        R1 = (I + A)
        R2 = (R1 @ R1); R2.data[:] = 1.0
        R3 = (R2 @ R1); R3.data[:] = 1.0
        # exclusion mask for self-collision (graph distance <= 3)
        self.excl = np.zeros((n, n), dtype=bool)
        R3c = R3.tocoo()
        self.excl[R3c.row, R3c.col] = True
        # bending pairs = 2-hop but not 1-hop/identity
        R2c = R2.tocoo()
        r1 = np.asarray(R1.todense(), dtype=bool)
        mask = ~r1[R2c.row, R2c.col] & (R2c.row < R2c.col)
        self.bends = np.stack([R2c.row[mask], R2c.col[mask]], axis=1)

        self.L_edge = np.linalg.norm(X0[e[:, 0]] - X0[e[:, 1]], axis=1)
        b = self.bends
        self.L_bend = np.linalg.norm(X0[b[:, 0]] - X0[b[:, 1]], axis=1) \
            if len(b) else np.zeros(0)

        # ---------------- rest vertex normals (area weighted)
        self.N0 = self._vertex_normals(X0)

        # ---------------- neighbor padding for deformation gradient
        nbr = [set() for _ in range(n)]
        for i, j in e:
            nbr[i].add(j)
            nbr[j].add(i)
        self.kmax = max(len(s) for s in nbr)
        self.nbr_idx = np.zeros((n, self.kmax), dtype=np.int64)
        self.nbr_mask = np.zeros((n, self.kmax))
        for i in range(n):
            js = sorted(nbr[i])
            self.nbr_idx[i, :len(js)] = js
            self.nbr_mask[i, :len(js)] = 1.0
        d = X0[self.nbr_idx] - X0[:, None, :]
        self.dX_pad = d * self.nbr_mask[:, :, None]
        self.s_pseudo = 0.5 * float(self.L_edge.mean())

        # ---------------- state
        self.x = X0.copy()
        self.v = np.tile(np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]),
                                    dtype=float), (n, 1))
        self.t = 0.0
        self.pen_gap = self.x[:, 2] - self.ground_z
        self.dt_target = float(cfg["sample_dt"]) / 6.0
        self.steps_done = 0

    # ------------------------------------------------------------- helpers
    def _vertex_normals(self, x):
        f = self.faces
        n = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
        vn = np.zeros_like(x)
        np.add.at(vn, f[:, 0], n)
        np.add.at(vn, f[:, 1], n)
        np.add.at(vn, f[:, 2], n)
        ln = np.linalg.norm(vn, axis=1, keepdims=True)
        return vn / np.maximum(ln, 1e-15)

    # ----------------------------------------------------------- mechanics
    def _spring_terms(self, pairs, L, k, c, dt):
        """Return force on i, and COO pieces of dt^2*k*u u^T + dt*c*u u^T."""
        i, j = pairs[:, 0], pairs[:, 1]
        dvec = self.x[i] - self.x[j]
        d = np.linalg.norm(dvec, axis=1)
        d = np.maximum(d, 1e-12)
        u = dvec / d[:, None]
        f = (-k * (d - L))[:, None] * u  # on i
        w = (dt * dt * k + dt * c)
        uuT = u[:, :, None] * u[:, None, :]  # (P,3,3)
        blk = w * uuT
        return i, j, f, blk

    def _self_contact(self, dt):
        tree = cKDTree(self.x)
        pairs = np.array(sorted(tree.query_pairs(self.h_contact)),
                         dtype=np.int64).reshape(-1, 2)
        if len(pairs) == 0:
            return (np.zeros((self.n, 3)), [], [], [])
        keep = ~self.excl[pairs[:, 0], pairs[:, 1]]
        pairs = pairs[keep]
        if len(pairs) == 0:
            return (np.zeros((self.n, 3)), [], [], [])
        i, j = pairs[:, 0], pairs[:, 1]
        dvec = self.x[i] - self.x[j]
        d = np.linalg.norm(dvec, axis=1)
        d = np.maximum(d, 1e-12)
        act = d < self.h_contact
        i, j, dvec, d = i[act], j[act], dvec[act], d[act]
        if len(i) == 0:
            return (np.zeros((self.n, 3)), [], [], [])
        u = dvec / d[:, None]
        fmag = self.k_self * (self.h_contact - d)
        fs = np.zeros((self.n, 3))
        np.add.at(fs, i, fmag[:, None] * u)
        np.add.at(fs, j, -fmag[:, None] * u)
        blk = (dt * dt * self.k_self + dt * self.c_self) * (u[:, :, None] * u[:, None, :])
        return fs, i, j, blk

    def _substep(self, dt):
        n = self.n
        # ---- forces at current state
        f = np.zeros((n, 3))
        f += self.mass[:, None] * self.gravity[None, :]

        rows, cols, vals = [], [], []

        def add_blocks(i, j, blk):
            # symmetric 2x2 block coupling of pair (i,j):
            #   A[i,i] += blk, A[j,j] += blk, A[i,j] -= blk, A[j,i] -= blk
            a3 = np.arange(3)
            ri = np.concatenate([3 * i[:, None] + a3] * 3, axis=1)  # (P,9)
            cj = np.concatenate([3 * j[:, None] + a3] * 3, axis=1)
            b = blk.reshape(len(i), 9)
            rows.append(ri.ravel()); cols.append(ri.ravel()); vals.append(b.ravel())
            rows.append(cj.ravel()); cols.append(cj.ravel()); vals.append(b.ravel())
            rows.append(ri.ravel()); cols.append(cj.ravel()); vals.append((-b).ravel())
            rows.append(cj.ravel()); cols.append(ri.ravel()); vals.append((-b).ravel())

        for pairs, L, k, c in ((self.edges, self.L_edge, self.k_edge, self.c_edge),
                               (self.bends, self.L_bend, self.k_bend, self.c_bend)):
            if len(pairs) == 0:
                continue
            i, j, fi, blk = self._spring_terms(pairs, L, k, c, dt)
            np.add.at(f, i, fi)
            np.add.at(f, j, -fi)
            add_blocks(i, j, blk)

        fs, si, sj, sblk = self._self_contact(dt)
        f += fs
        if len(si):
            add_blocks(si, sj, sblk)

        # ---- assemble A
        diag = np.full(3 * n, self.m_node + dt * self.c_glob)
        A = sp.diags(diag, format="coo")
        if rows:
            r = np.concatenate([A.row] + rows)
            c = np.concatenate([A.col] + cols)
            v = np.concatenate([A.data] + vals)
            A = sp.coo_matrix((v, (r, c)), shape=(3 * n, 3 * n)).tocsr()
        else:
            A = A.tocsr()

        rhs = (self.mass[:, None] * self.v + dt * f).reshape(3 * n)
        v_new = spsolve(A, rhs).reshape(n, 3)
        if not np.all(np.isfinite(v_new)):
            raise FloatingPointError("solver diverged")
        self.v = v_new
        x_pre = self.x + dt * self.v
        self.pen_gap = x_pre[:, 2] - self.ground_z

        # ---- ground constraint projection + friction
        gz = self.ground_z
        contact = x_pre[:, 2] < gz
        if contact.any():
            vn_in = self.v[contact, 2]
            vn_out = np.where(vn_in < -0.3, -self.restitution * vn_in,
                              np.where(vn_in < 0.0, 0.0, vn_in))
            jn_m = np.abs(vn_in - vn_out) + dt * 9.81  # normal impulse / mass
            vt = self.v[contact, :2]
            sp_t = np.linalg.norm(vt, axis=1)
            scale = np.clip(1.0 - self.mu * jn_m / np.maximum(sp_t, 1e-9), 0.0, 1.0)
            self.v[contact, :2] = vt * scale[:, None]
            self.v[contact, 2] = vn_out
            x_pre[contact, 2] = gz
        self.x = x_pre
        self.t += dt
        self.steps_done += 1

    # -------------------------------------------------------------- driving
    def advance_to(self, time: float):
        time = float(time)
        rem = time - self.t
        if rem > 1e-12:
            nsub = max(1, int(np.ceil(rem / self.dt_target)))
            dt = rem / nsub
            for _ in range(nsub):
                self._substep(dt)
        return self._make_frame(time)

    def _make_frame(self, t):
        x = self.x
        # current vertex normals + deformation gradient (least squares with a
        # pseudo normal-direction observation so F = I at rest)
        nc = self._vertex_normals(x)
        dx = (x[self.nbr_idx] - x[:, None, :]) * self.nbr_mask[:, :, None]
        G = np.einsum("nki,nkj->nij", dx, self.dX_pad)
        H = np.einsum("nki,nkj->nij", self.dX_pad, self.dX_pad)
        s2 = self.s_pseudo ** 2
        G += s2 * (nc[:, :, None] * self.N0[:, None, :])
        H += s2 * (self.N0[:, :, None] * self.N0[:, None, :])
        F = G @ np.linalg.inv(H)

        mass = self.mass
        gap = np.array(self.pen_gap, dtype=float)
        return StateFrame(
            time=float(t),
            position=x.copy(),
            velocity=self.v.copy(),
            mass=mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": mass.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": mass[:, None] * self.gravity[None, :],
                "contact_signed_distance": gap,
            },
            scalars={"contact_length_scale": self.h_contact},
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": int(self.steps_done),
            "time": float(self.t),
            "min_z": float(self.x[:, 2].min()) if hasattr(self, "x") else 0.0,
            "kinetic_energy": float(0.5 * np.sum(self.mass[:, None] * self.v ** 2))
            if hasattr(self, "v") else 0.0,
        }
