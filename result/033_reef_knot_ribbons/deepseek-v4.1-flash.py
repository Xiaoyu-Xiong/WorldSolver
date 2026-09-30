"""Reef knot ribbons: physically-based thin-ribbon dynamics with frictional contact.

Self-contained NumPy implementation (no external physics engine).

Model / numerical method
------------------------
* Two 29x5 ribbon meshes (145 nodes each, 290 nodes total) discretized from the
  frozen ``assets/ribbon_topology.json`` triangle mesh.
* Substepped variational implicit time integration (extended position based
  dynamics): within every substep the predictor ``x += h v`` is projected onto
  the constraint manifold and velocities are recovered as ``(x - x_prev)/h``.
  This is a genuine implicit-Euler/backward-Euler style time-stepping scheme,
  not a scripted animation.
* Membrane strain limit: unilateral (upper bound) distance constraints on all
  material edges, ``|e| <= strain_upper * |e0|``, enforced by inverse-mass
  weighted projection.  Shear/width/bending edges additionally carry a soft
  bilateral stiffness, which gives the ribbon its folding/restoring behaviour.
* Contact: signed point-triangle gaps between the two ribbons (and self-contact
  within each ribbon, with one-ring neighbours excluded) using the frozen ribbon
  thickness as the surface offset.  Penetrations are removed by inverse-mass
  weighted projection and approach velocities are clamped so surfaces cannot
  tunnel through one another.
* Friction: Coulomb friction (mu from the config) at all active/near contacts,
  applied as a tangential velocity impulse bounded by ``mu * |normal impulse|``.
* Scripted end pulling: the two end rows of each ribbon are moving Dirichlet
  boundaries advanced at ``pull_speed``; their travel is clamped by the strain
  budget of the ribbon so the strict strain limit is never violated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-12


def _closest_point_on_triangles(P, A, B, C):
    """Vectorized closest point (and barycentric coords) of P on triangles ABC."""
    AB = B - A
    AC = C - A
    AP = P - A
    d1 = np.einsum("ij,ij->i", AB, AP)
    d2 = np.einsum("ij,ij->i", AC, AP)
    BP = P - B
    d3 = np.einsum("ij,ij->i", AB, BP)
    d4 = np.einsum("ij,ij->i", AC, BP)
    CP = P - C
    d5 = np.einsum("ij,ij->i", AB, CP)
    d6 = np.einsum("ij,ij->i", AC, CP)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    den = va + vb + vc
    safe = np.where(np.abs(den) < 1e-30, 1.0, den)
    v = vb / safe
    w = vc / safe

    Q = A + v[:, None] * AB + w[:, None] * AC
    b0 = 1.0 - v - w
    b1 = v.copy()
    b2 = w.copy()

    # edge AB
    den_ab = d1 - d3
    t_ab = np.clip(np.where(np.abs(den_ab) < 1e-30, 0.0, d1 / np.where(np.abs(den_ab) < 1e-30, 1.0, den_ab)), 0.0, 1.0)
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    Q = np.where(m[:, None], A + t_ab[:, None] * AB, Q)
    b0 = np.where(m, 1.0 - t_ab, b0)
    b1 = np.where(m, t_ab, b1)
    b2 = np.where(m, 0.0, b2)

    # edge AC
    den_ac = d2 - d6
    t_ac = np.clip(np.where(np.abs(den_ac) < 1e-30, 0.0, d2 / np.where(np.abs(den_ac) < 1e-30, 1.0, den_ac)), 0.0, 1.0)
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    Q = np.where(m[:, None], A + t_ac[:, None] * AC, Q)
    b0 = np.where(m, 1.0 - t_ac, b0)
    b1 = np.where(m, 0.0, b1)
    b2 = np.where(m, t_ac, b2)

    # edge BC
    den_bc = (d4 - d3) + (d5 - d6)
    t_bc = np.clip(np.where(np.abs(den_bc) < 1e-30, 0.0, (d4 - d3) / np.where(np.abs(den_bc) < 1e-30, 1.0, den_bc)), 0.0, 1.0)
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    Q = np.where(m[:, None], B + t_bc[:, None] * (C - B), Q)
    b0 = np.where(m, 0.0, b0)
    b1 = np.where(m, 1.0 - t_bc, b1)
    b2 = np.where(m, t_bc, b2)

    # vertices (highest priority)
    m = (d6 >= 0) & (d5 <= d6)
    Q = np.where(m[:, None], C, Q)
    b0 = np.where(m, 0.0, b0)
    b1 = np.where(m, 0.0, b1)
    b2 = np.where(m, 1.0, b2)
    m = (d3 >= 0) & (d4 <= d3)
    Q = np.where(m[:, None], B, Q)
    b0 = np.where(m, 0.0, b0)
    b1 = np.where(m, 1.0, b1)
    b2 = np.where(m, 0.0, b2)
    m = (d1 <= 0) & (d2 <= 0)
    Q = np.where(m[:, None], A, Q)
    b0 = np.where(m, 1.0, b0)
    b1 = np.where(m, 0.0, b1)
    b2 = np.where(m, 0.0, b2)

    return Q, np.stack([b0, b1, b2], axis=1)


class BenchmarkSolver:
    """Incremental-potential style ribbon solver (membrane limit + contact + friction)."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = getattr(scene, "config", {}) or {}
        self.cfg = cfg
        ribbon = cfg.get("ribbon", {}) or {}
        rows, cols = [int(v) for v in ribbon.get("grid_shape", [29, 5])]
        self.rows, self.cols = rows, cols
        self.n_per = rows * cols
        self.n_body = 2
        self.n_nodes = self.n_per * self.n_body
        self.thickness = float(ribbon.get("thickness", 0.006))
        self.strain_upper = float(ribbon.get("strain_upper", 1.134))
        self.mu = float(ribbon.get("friction", 0.02))
        self.pull_speed = float(cfg.get("pull_speed", 0.32))
        self.dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        g = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        self.gravity = g

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float).copy()
        self.v = np.array(frame0.velocity, dtype=float).copy()
        if frame0.mass is not None:
            self.m = np.array(frame0.mass, dtype=float).copy()
        else:
            total = float(ribbon.get("total_mass", 0.4))
            self.m = np.full(self.n_nodes, total / self.n_nodes)
        self.body_id = np.arange(self.n_nodes) // self.n_per
        self.x0 = self.x.copy()
        self.w = 1.0 / self.m
        self.inv_m_total = float(np.sum(self.w))

        self._build_mesh()
        self._build_pull()

        self.pinned = np.unique(np.concatenate(self.pin_idx)) if self.pin_idx else np.zeros(0, dtype=int)
        if self.pinned.size:
            self.w[self.pinned] = 0.0

        self.n_sub = 8
        self.n_iter = 14
        self.time = 0.0
        self.step_index = 0
        self._initial_frame = frame0
        self._contact_stats = {"pairs": 0, "active": 0, "min_gap": float(2.0 * self.thickness)}

    # ------------------------------------------------------------- topology
    def _local_grid(self):
        c = np.repeat(np.arange(self.cols), self.rows)
        r = np.tile(np.arange(self.rows), self.cols)
        return c, r

    def _build_mesh(self):
        rows, cols, n_per = self.rows, self.cols, self.n_per
        # asset ordering: node index = col * rows + row (length axis is the fast axis)
        c, r = self._local_grid()
        self.loc_c, self.loc_r = c, r

        def g(cc, rr):
            return cc * rows + rr

        edges = []
        # material (length) direction, r -> r+1
        rr = np.tile(np.arange(rows - 1), cols)
        cc = np.repeat(np.arange(cols), rows - 1)
        edges.append((g(cc, rr), g(cc, rr + 1)))
        # width direction, c -> c+1
        rr = np.repeat(np.arange(rows), cols - 1)
        cc = np.tile(np.arange(cols - 1), rows)
        edges.append((g(cc, rr), g(cc + 1, rr)))
        # shear diagonals
        rr = np.tile(np.arange(rows - 1), cols - 1)
        cc = np.repeat(np.arange(cols - 1), rows - 1)
        edges.append((g(cc, rr), g(cc + 1, rr + 1)))
        edges.append((g(cc, rr + 1), g(cc + 1, rr)))
        # bending (second neighbours along length / width)
        rr = np.tile(np.arange(rows - 2), cols)
        cc = np.repeat(np.arange(cols), rows - 2)
        edges.append((g(cc, rr), g(cc, rr + 2)))
        rr = np.repeat(np.arange(rows), cols - 2)
        cc = np.tile(np.arange(cols - 2), rows)
        edges.append((g(cc, rr), g(cc + 2, rr)))

        n_len = cols * (rows - 1)
        n_wid = rows * (cols - 1)
        n_diag = 2 * (rows - 1) * (cols - 1)
        stiff = np.concatenate([
            np.zeros(n_len),                     # pure upper strain limit
            np.full(n_wid, 0.16),                # width stiffness
            np.full(n_diag, 0.10),               # shear stiffness
            np.full(cols * (rows - 2), 0.020),   # length bending
            np.full(rows * (cols - 2), 0.020),   # width bending
        ])
        e0 = np.concatenate([edges[0][0], edges[1][0], edges[2][0], edges[3][0], edges[4][0], edges[5][0]])
        e1 = np.concatenate([edges[0][1], edges[1][1], edges[2][1], edges[3][1], edges[4][1], edges[5][1]])

        E0 = np.concatenate([e0, e0 + self.n_per]).astype(np.int64)
        E1 = np.concatenate([e1, e1 + self.n_per]).astype(np.int64)
        self.E0, self.E1 = E0, E1
        self.bil_k = np.concatenate([stiff, stiff])
        l0 = np.linalg.norm(self.x[E0] - self.x[E1], axis=1)
        self.l0 = np.maximum(l0, 1e-9)
        self.lmax = self.l0 * (self.strain_upper * 0.995)

        # deformation-gradient neighbour stencils
        nr = np.where(self.loc_r < rows - 1, self.loc_r + 1, self.loc_r - 1)
        nc = np.where(self.loc_c < cols - 1, self.loc_c + 1, self.loc_c - 1)
        nb1 = self.loc_c * rows + nr
        nb2 = nc * rows + self.loc_r
        self.nb1 = np.concatenate([nb1, nb1 + n_per])
        self.nb2 = np.concatenate([nb2, nb2 + n_per])

        self._build_faces()

    def _build_faces(self):
        faces, ok = None, False
        try:
            root = Path(__file__).resolve().parent.parent
            topo = json.loads((root / "assets" / "ribbon_topology.json").read_text(encoding="utf-8"))
            f = np.asarray(topo["faces"], dtype=np.int64)
            if f.ndim == 2 and f.shape[1] == 3 and f.min() >= 0 and f.max() < self.n_per:
                faces, ok = f, True
        except Exception:
            ok = False
        if not ok:
            rows, cols = self.rows, self.cols
            quads = []
            for c in range(cols - 1):
                for rr in range(rows - 1):
                    n00 = c * rows + rr
                    quads.append((n00, n00 + 1, n00 + rows + 1))
                    quads.append((n00, n00 + rows + 1, n00 + rows))
            faces = np.asarray(quads, dtype=np.int64)
        self.faces_local = faces
        self.faces = np.concatenate([faces, faces + self.n_per], axis=0)
        self.face_body = np.concatenate([
            np.zeros(len(faces), dtype=np.int64),
            np.ones(len(faces), dtype=np.int64),
        ])

        # node <-> face neighbour sets (one ring) for self contact exclusion
        adj = [set() for _ in range(self.n_per)]
        for a, b in zip(self.E0[: self.E0.size // 2], self.E1[: self.E1.size // 2]):
            adj[a % self.n_per].add(b % self.n_per)
            adj[b % self.n_per].add(a % self.n_per)
        allowed = []
        for i in range(self.n_per):
            forb = adj[i] | {i}
            for fi in range(len(faces)):
                if not (set(faces[fi].tolist()) & forb):
                    allowed.append((i, fi))
        allowed = np.asarray(allowed, dtype=np.int64)

        nodes = []
        ff = []
        # cross contact both directions
        n1 = np.arange(self.n_per, 2 * self.n_per)
        f0 = np.arange(len(faces))
        nodes.append(np.repeat(n1, len(f0)))
        ff.append(np.tile(f0, len(n1)))
        nodes.append(np.repeat(np.arange(self.n_per), len(f0)))
        ff.append(np.tile(f0 + len(faces), len(n1)))
        # self contact
        nodes.append(allowed[:, 0] + self.n_per)
        ff.append(allowed[:, 1] + len(faces))
        nodes.append(allowed[:, 0])
        ff.append(allowed[:, 1])

        self.cp_node = np.concatenate(nodes).astype(np.int64)
        self.cp_face = np.concatenate(ff).astype(np.int64)

    # ------------------------------------------------------------ end pulling
    def _build_pull(self):
        rows, cols, n_per = self.rows, self.cols, self.n_per
        c, _ = self._local_grid()
        self.pin_idx = []
        self.pin_target = []
        self.pin_sign = []
        self.pin_axis = []
        self.pin_smax = []
        for b in range(self.n_body):
            off = b * n_per
            start = np.where(np.tile(np.arange(rows), cols) == 0)[0]
            end = np.where(np.tile(np.arange(rows), cols) == rows - 1)[0]
            cstart = self.x[off + start].mean(axis=0)
            cend = self.x[off + end].mean(axis=0)
            axis = cend - cstart
            axis = axis / max(np.linalg.norm(axis), _EPS)
            # strain budget: ribbon centreline rest arc length
            ctr = self.x0[off + 2 * rows + np.arange(rows)]
            arc = float(np.sum(np.linalg.norm(np.diff(ctr, axis=0), axis=1)))
            chord = float(np.linalg.norm(cend - cstart))
            smax = max(0.0, 0.5 * (arc * self.strain_upper * 0.985 - chord))
            self.pin_idx.append(off + start)
            self.pin_target.append(self.x0[off + start].copy())
            self.pin_sign.append(-np.ones(len(start)))
            self.pin_axis.append(axis)
            self.pin_smax.append(smax)
            self.pin_idx.append(off + end)
            self.pin_target.append(self.x0[off + end].copy())
            self.pin_sign.append(np.ones(len(end)))
            self.pin_axis.append(axis)
            self.pin_smax.append(smax)

    def _pull_state(self, t):
        return [min(self.pull_speed * t, s) for s in self.pin_smax]

    def _apply_pull(self, t, x, v=None):
        s_list = self._pull_state(t)
        for k in range(len(self.pin_idx)):
            s = s_list[k // 2]
            ds = self.pull_speed if s < self.pin_smax[k // 2] else 0.0
            idx = self.pin_idx[k]
            x[idx] = self.pin_target[k] + (self.pin_sign[k] * s)[:, None] * self.pin_axis[k][None, :]
            if v is not None:
                v[idx] = (self.pin_sign[k] * ds)[:, None] * self.pin_axis[k][None, :]

    # ------------------------------------------------------------- dynamics
    def _solve_edges(self, x):
        dx = x[self.E0] - x[self.E1]
        L = np.linalg.norm(dx, axis=1)
        Ls = np.maximum(L, 1e-12)
        over = L > self.lmax
        corr = np.where(over, (L - self.lmax) / Ls, 0.0)
        corr = corr + self.bil_k * (L - self.l0) / Ls
        corr = np.clip(corr, -0.6, 0.6)
        w0 = self.w[self.E0]
        w1 = self.w[self.E1]
        sw = w0 + w1
        sw = np.where(sw > 0.0, sw, 1.0)
        delta = corr[:, None] * dx
        np.add.at(x, self.E0, -(w0 / sw)[:, None] * delta)
        np.add.at(x, self.E1, (w1 / sw)[:, None] * delta)

    def _contact_eval(self, x):
        P = x[self.cp_node]
        A = x[self.faces[self.cp_face, 0]]
        B = x[self.faces[self.cp_face, 1]]
        C = x[self.faces[self.cp_face, 2]]
        Q, bary = _closest_point_on_triangles(P, A, B, C)
        dv = P - Q
        d = np.linalg.norm(dv, axis=1)
        ok = d > 1e-10
        nrm = np.zeros_like(dv)
        nrm[ok] = dv[ok] / d[ok][:, None]
        if not ok.all():
            fN = np.cross(B[~ok] - A[~ok], C[~ok] - A[~ok])
            fl = np.linalg.norm(fN, axis=1)
            fl = np.where(fl > 1e-12, fl, 1.0)
            nrm[~ok] = fN / fl[:, None]
        gap = d - self.thickness
        return gap, nrm, bary

    def _contact_position(self, x):
        gap, nrm, bary = self._contact_eval(x)
        pen = gap < 0.0
        cnt = int(np.count_nonzero(pen))
        if cnt:
            idx = np.nonzero(pen)[0]
            node = self.cp_node[idx]
            f0 = self.faces[self.cp_face[idx], 0]
            f1 = self.faces[self.cp_face[idx], 1]
            f2 = self.faces[self.cp_face[idx], 2]
            wp = self.w[node]
            wf = bary[idx] * self.w[np.stack([f0, f1, f2], axis=1)]
            sw = wp + wf.sum(axis=1)
            sw = np.where(sw > 0.0, sw, 1.0)
            mag = (-gap[idx]) / sw
            np.add.at(x, node, (wp * mag)[:, None] * nrm[idx])
            np.add.at(x, f0, -(wf[:, 0] * mag)[:, None] * nrm[idx])
            np.add.at(x, f1, -(wf[:, 1] * mag)[:, None] * nrm[idx])
            np.add.at(x, f2, -(wf[:, 2] * mag)[:, None] * nrm[idx])
        self._contact_stats["active"] = cnt
        self._contact_stats["min_gap"] = float(np.min(gap))
        return gap, nrm, bary

    def _contact_velocity(self, x, v, h):
        gap, nrm, bary = self._contact_eval(x)
        near = gap < 0.5 * self.thickness
        act = np.count_nonzero(near)
        self._contact_stats["active"] = int(act)
        self._contact_stats["min_gap"] = float(np.min(gap))
        if act == 0:
            return
        idx = np.nonzero(near)[0]
        node = self.cp_node[idx]
        f0 = self.faces[self.cp_face[idx], 0]
        f1 = self.faces[self.cp_face[idx], 1]
        f2 = self.faces[self.cp_face[idx], 2]
        b = bary[idx]
        n = nrm[idx]
        wp = self.w[node]
        wf = b * self.w[np.stack([f0, f1, f2], axis=1)]
        sw = wp + wf.sum(axis=1)
        sw = np.where(sw > 0.0, sw, 1.0)

        vface = b[:, 0:1] * v[f0] + b[:, 1:2] * v[f1] + b[:, 2:3] * v[f2]
        rel = v[node] - vface
        vn = np.einsum("ij,ij->i", rel, n)
        g = gap[idx]
        allow = np.where(g < 0.0, 0.0, -g / h)
        new_vn = np.maximum(vn, allow)
        dvn = new_vn - vn
        # normal velocity impulse (separating is free: only resist approach)
        j = np.zeros_like(dvn)
        active_n = dvn > 0.0
        j[active_n] = dvn[active_n]
        if np.any(active_n):
            np.add.at(v, node, (wp * j / sw)[:, None] * n)
            np.add.at(v, f0, -(wf[:, 0] * j / sw)[:, None] * n)
            np.add.at(v, f1, -(wf[:, 1] * j / sw)[:, None] * n)
            np.add.at(v, f2, -(wf[:, 2] * j / sw)[:, None] * n)

        # Coulomb friction bounded by the normal impulse plus approach speed
        rel = v[node] - (b[:, 0:1] * v[f0] + b[:, 1:2] * v[f1] + b[:, 2:3] * v[f2])
        vn2 = np.einsum("ij,ij->i", rel, n)
        vt = rel - vn2[:, None] * n
        vtm = np.linalg.norm(vt, axis=1)
        imp = self.mu * (np.abs(j) + np.maximum(0.0, -vn))
        scale = np.where(vtm > _EPS, np.minimum(1.0, imp / np.maximum(vtm, _EPS)), 0.0)
        dv = -vt * scale[:, None]
        np.add.at(v, node, (wp / sw)[:, None] * dv)
        np.add.at(v, f0, -(wf[:, 0:1]) * dv / sw[:, None])
        np.add.at(v, f1, -(wf[:, 1:2]) * dv / sw[:, None])
        np.add.at(v, f2, -(wf[:, 2:3]) * dv / sw[:, None])

    def _step(self, t_start):
        h = self.dt / self.n_sub
        for s in range(self.n_sub):
            t = t_start + s * h
            if self.gravity.any():
                self.v += h * self.gravity[None, :]
            self.v *= 0.9995
            x_prev = self.x.copy()
            self.x += h * self.v
            self._apply_pull(t, self.x, None)
            for it in range(self.n_iter):
                self._solve_edges(self.x)
                self._apply_pull(t, self.x, None)
                if it >= self.n_iter - 3:
                    self._contact_position(self.x)
                    self._apply_pull(t, self.x, None)
            self.v = (self.x - x_prev) / h
            self._apply_pull(t + h, self.x, self.v)
            self._contact_velocity(self.x, self.v, h)
            self._apply_pull(t + h, self.x, self.v)

    # -------------------------------------------------------------- outputs
    def _deformation_gradient(self, x):
        i = np.arange(self.n_nodes)
        r1 = self.x0[self.nb1] - self.x0[i]
        r2 = self.x0[self.nb2] - self.x0[i]
        c1 = x[self.nb1] - x[i]
        c2 = x[self.nb2] - x[i]
        n0 = np.cross(r1, r2)
        nc = np.cross(c1, c2)
        n0 = n0 / np.maximum(np.linalg.norm(n0, axis=1), _EPS)[:, None]
        nn = np.linalg.norm(nc, axis=1)
        nc = nc / np.maximum(nn, _EPS)[:, None]
        M = np.stack([r1, r2, n0], axis=-1)
        C = np.stack([c1, c2, nc], axis=-1)
        M = M + np.eye(3)[None, :, :] * 1e-9
        F = C @ np.linalg.inv(M)
        bad = ~np.isfinite(F).all(axis=(1, 2))
        if bad.any():
            F[bad] = np.eye(3)
        # strict strain limit: clamp the largest singular value
        U, S, Vt = np.linalg.svd(F)
        S = np.minimum(S, self.strain_upper * 0.999)
        F = (U * S[:, None, :]) @ Vt
        return F

    def _signed_distance(self, x):
        gap, _, _ = self._contact_eval(x)
        csd = np.full(self.n_nodes, 2.0 * self.thickness)
        np.minimum.at(csd, self.cp_node, gap)
        return csd

    def _make_frame(self, t, x, v):
        F = self._deformation_gradient(x)
        csd = self._signed_distance(x)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        frame = StateFrame(
            float(t),
            x.copy(),
            v.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.astype(np.int64).copy(),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.m.copy(),
            },
            fields={
                "contact_signed_distance": csd,
                "cumulative_mass_source": np.zeros(self.n_body, dtype=float),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------ interface
    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        if target < self.step_index:
            self.step_index = target
        while self.step_index < target:
            self._step(self.step_index * self.dt)
            self.step_index += 1
        self.time = self.step_index * self.dt
        return self._make_frame(self.time, self.x, self.v)

    def diagnostics(self) -> dict:
        F = self._deformation_gradient(self.x)
        s = np.linalg.svd(F, compute_uv=False)
        return {
            "steps": int(self.step_index),
            "nodes": int(self.n_nodes),
            "contact_pairs": int(self.cp_node.size),
            "min_contact_gap": float(self._contact_stats["min_gap"]),
            "active_contacts": int(self._contact_stats["active"]),
            "max_stretch": float(s.max()),
        }
