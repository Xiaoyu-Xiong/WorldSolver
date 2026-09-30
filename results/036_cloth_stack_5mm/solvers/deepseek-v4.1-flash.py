"""Physically-based 5 mm cloth-stack solver (shell dynamics + contact + friction).

Method
------
Every panel is a triangle-mesh shell discretised by its frozen 13x13 material
grid.  Elasticity is enforced with distance constraints on material edges:

  * warp / weft (grid) edges    : zero compliance (inextensible yarn),
  * cell diagonals (shear)      : finite small compliance,
  * second neighbours (bending) : soft finite compliance.

They are integrated with the Extended Position Based Dynamics small-substep
scheme, a variationally consistent implicit-Euler time discretisation of the
shell dynamics: every substep applies gravity as a velocity increment,
predicts positions, projects the constraint set, and recovers velocity from
the projected positions.

Contact
-------
board  : analytic signed distance to the frozen board box; shell mid-surfaces
         are kept half a shell thickness above the surface.
panels : exact point/triangle signed gaps against the frozen 5 mm thickness
         (mid-surface separation of two shells), collected with a kd-tree
         broad phase.  Corrections are mass weighted on both sides using the
         barycentric weights of the closest point.
friction: position-level Coulomb friction with the frozen mu, clipped by the
         normal correction, both for board and panel-panel contacts.

No animation, keyframes, tables or precomputed motion are used: the exported
state is the integrated physical state of the shells.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_EPS = 1e-12


def _box_sdf(points, half):
    """Signed distance and outward unit normal of an axis-aligned box."""
    q = np.abs(points) - half
    outside = np.maximum(q, 0.0)
    sd = np.linalg.norm(outside, axis=1) + np.minimum(np.max(q, axis=1), 0.0)
    sgn = np.where(points >= 0.0, 1.0, -1.0)
    grad = sgn * (q > 0.0)
    norm = np.linalg.norm(grad, axis=1)
    axis = np.argmax(q, axis=1)
    fallback = np.zeros_like(points)
    fallback[np.arange(points.shape[0]), axis] = 1.0
    fallback = sgn * fallback
    bad = norm < 1e-12
    unit = np.where(bad[:, None], fallback, grad / np.maximum(norm, 1e-30)[:, None])
    return sd, unit


def _triangle_barycentric(p, a, b, c):
    """Vectorised closest point (barycentric coordinates) on triangles to points.

    Implements the standard point/triangle region decomposition (Ericson,
    Real-Time Collision Detection, section 5.1.5) for arrays of points and
    triangle corners.  Returns an (M, 3) array of barycentric weights.
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    m = p.shape[0]
    bary = np.zeros((m, 3))
    done = np.zeros(m, dtype=bool)

    sel = (d1 <= 0.0) & (d2 <= 0.0)
    bary[sel] = np.array([1.0, 0.0, 0.0])
    done |= sel

    sel = (~done) & (d3 >= 0.0) & (d4 <= d3)
    bary[sel] = np.array([0.0, 1.0, 0.0])
    done |= sel

    sel = (~done) & (d6 >= 0.0) & (d5 <= d6)
    bary[sel] = np.array([0.0, 0.0, 1.0])
    done |= sel

    sel = (~done) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    if np.any(sel):
        t = d1[sel] / np.maximum(d1[sel] - d3[sel], 1e-30)
        bary[sel] = np.stack([1.0 - t, t, np.zeros_like(t)], axis=1)
        done |= sel

    sel = (~done) & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    if np.any(sel):
        t = d2[sel] / np.maximum(d2[sel] - d6[sel], 1e-30)
        bary[sel] = np.stack([1.0 - t, np.zeros_like(t), t], axis=1)
        done |= sel

    sel = (~done) & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    if np.any(sel):
        num = d4[sel] - d3[sel]
        den = num + (d5[sel] - d6[sel])
        t = num / np.maximum(den, 1e-30)
        bary[sel] = np.stack([np.zeros_like(t), 1.0 - t, t], axis=1)
        done |= sel

    sel = ~done
    if np.any(sel):
        den = va[sel] + vb[sel] + vc[sel]
        inv = 1.0 / np.where(np.abs(den) < 1e-30, 1e-30, den)
        v = vb[sel] * inv
        w = vc[sel] * inv
        bary[sel] = np.stack([1.0 - v - w, v, w], axis=1)
    return bary


class BenchmarkSolver:
    """Deterministic shell/contact time-stepping solver for the cloth stack."""

    substeps_per_frame = 80
    k_nn = 40
    damping = 0.3
    alpha_shear = 5.0e-4
    alpha_bend = 2.0e-3

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        meta = dict(scene.metadata() or {})
        cfg = dict(getattr(scene, "config", {}) or {})
        frame0 = scene.initial_state()

        self.x = np.array(frame0.position, dtype=float)
        self.v = np.zeros_like(self.x)
        n = int(self.x.shape[0])
        self.n = n

        body = frame0.body_id
        self.body_id = (
            np.zeros(n, dtype=np.int64) if body is None else np.asarray(body, dtype=np.int64)
        )

        mass = None
        attrs = dict(frame0.attributes or {})
        if "mass_quantity" in attrs:
            candidate = np.asarray(attrs["mass_quantity"], dtype=float).reshape(-1)
            if candidate.shape == (n,):
                mass = candidate
        if mass is None and frame0.mass is not None:
            mass = np.asarray(frame0.mass, dtype=float).reshape(-1)
        if mass is None or mass.shape != (n,):
            mass = np.full(n, 0.12 * 10.0 / n)
        self.mass = np.maximum(mass, 1e-12)
        self.w = 1.0 / self.mass
        self.uv = (
            np.asarray(attrs["uv"], dtype=float) if "uv" in attrs else None
        )

        scalars = dict(frame0.scalars or {})
        panel_cfg = dict(cfg.get("panel", {}) or {})
        self.thickness = float(meta.get("thickness", panel_cfg.get("thickness", 0.005)))
        self.contact_length_scale = float(
            scalars.get("contact_length_scale", self.thickness)
        )
        self.constraint_scale = float(scalars.get("constraint_scale", 1.35))
        # two 5 mm mid-surfaces touch when their centres are one thickness apart
        self.offset = self.thickness
        self.board_offset = 0.5 * self.thickness

        gravity = meta.get("gravity", cfg.get("gravity", [0.0, 0.0, -9.81]))
        self.g = np.asarray(gravity, dtype=float).reshape(1, 3)
        self.mu = float(cfg.get("friction", 0.1))

        geo = cfg.get("scene_geometry") or {}
        half = None
        for obj in geo.get("objects", []) if isinstance(geo, dict) else []:
            if obj.get("id") == "board":
                half = obj.get("geometry", {}).get("half_extents")
        if half is None:
            half = [0.9, 0.9, float(meta.get("board_top", 0.05))]
        self.board_half = np.asarray(half, dtype=float).reshape(1, 3)

        topo = dict(meta.get("topology") or {})
        grid = topo.get("grid_shape") or panel_cfg.get("grid_shape") or [13, 13]
        rows, cols = int(grid[0]), int(grid[1])
        faces = topo.get("faces")
        if faces is None:
            faces = self._faces_from_asset(geo)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

        panel_ids = np.unique(self.body_id)
        self.panel_count = int(panel_ids.size)
        n_local = rows * cols
        self.rows, self.cols, self.n_local = rows, cols, n_local
        if n != self.panel_count * n_local:
            raise ValueError("initial state node count does not match panel grid")

        first = np.where(self.body_id == panel_ids[0])[0]
        self.rest = self.x[first[:n_local]].copy()

        self._build_topology()

        sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.dt_sub = sample_dt / float(self.substeps_per_frame)
        self.t = 0.0
        self.last_nsub = 0

    def _faces_from_asset(self, geo):
        candidates = []
        for obj in geo.get("objects", []) if isinstance(geo, dict) else []:
            path = obj.get("geometry", {}).get("topology_asset")
            if path:
                candidates.append(Path(path))
        candidates.append(
            Path(__file__).resolve().parent.parent / "assets" / "panel_topology.json"
        )
        for path in candidates:
            try:
                if path.is_file():
                    import json as _json

                    data = _json.loads(path.read_text(encoding="utf-8"))
                    if "faces" in data:
                        return data["faces"]
            except Exception:
                continue
        raise ValueError("panel topology faces are unavailable")

    def _build_topology(self) -> None:
        rows, cols, local = self.rows, self.cols, self.n_local
        rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        idx = (rr * cols + cc).astype(np.int64)

        def pairs(a, b):
            return a.ravel(), b.ravel()

        struct = [
            pairs(idx[:, :-1], idx[:, 1:]),
            pairs(idx[:-1, :], idx[1:, :]),
        ]
        shear = [
            pairs(idx[:-1, :-1], idx[1:, 1:]),
            pairs(idx[:-1, 1:], idx[1:, :-1]),
        ]
        bend = [
            pairs(idx[:, :-2], idx[:, 2:]),
            pairs(idx[:-2, :], idx[2:, :]),
        ]

        rest = self.rest
        si, sj, sl0, salpha = [], [], [], []
        for group, alpha in (
            (struct, 0.0),
            (shear, self.alpha_shear),
            (bend, self.alpha_bend),
        ):
            for a, b in group:
                si.append(a)
                sj.append(b)
                sl0.append(np.linalg.norm(rest[b] - rest[a], axis=1))
                salpha.append(np.full(a.size, alpha, dtype=float))
        si = np.concatenate(si)
        sj = np.concatenate(sj)
        sl0 = np.concatenate(sl0)
        salpha = np.concatenate(salpha)
        self.struct_mask = salpha == 0.0

        gi, gj, gl0, galpha = [], [], [], []
        tri = []
        node_local, tri_local = [], []
        for p in range(self.panel_count):
            base = p * local
            gi.append(base + si)
            gj.append(base + sj)
            gl0.append(sl0)
            galpha.append(salpha)
            tri.append(base + self.faces)
            node_local.append(np.arange(local))
            tri_local.append(np.arange(self.faces.shape[0]))
        self.si = np.concatenate(gi).astype(np.int64)
        self.sj = np.concatenate(gj).astype(np.int64)
        self.sl0 = np.concatenate(gl0)
        self.spring_alpha = np.concatenate(galpha)
        self.salpha = np.zeros_like(self.spring_alpha)
        self.tris = np.concatenate(tri, axis=0).astype(np.int64)
        self.node_local = np.concatenate(node_local).astype(np.int64)
        self.tri_local = np.concatenate(tri_local).astype(np.int64)
        self.tri_panel = np.repeat(np.arange(self.panel_count), self.faces.shape[0])

        # a vertex never collides with the triangles touching it (its own ring)
        nr = idx // cols
        nc = idx % cols
        allowed = np.ones((local, self.faces.shape[0]), dtype=bool)
        for k in range(3):
            vr = nr[self.faces[:, k]]
            vc = nc[self.faces[:, k]]
            near = (np.abs(nr[:, None] - vr[None, :]) <= 1) & (
                np.abs(nc[:, None] - vc[None, :]) <= 1
            )
            allowed &= ~near
        self.allowed_local = allowed

        self.r_tri = np.full(self.tris.shape[0], 1e6)

    # -------------------------------------------------------------- detection
    def _detect(self, x):
        """Return candidate node/triangle pairs with their exact signed gaps."""
        tri_x = x[self.tris]
        centroids = tri_x.mean(axis=1)
        self.r_tri = np.max(
            np.linalg.norm(tri_x - centroids[:, None, :], axis=2), axis=1
        )
        tree = cKDTree(centroids)
        k = min(self.k_nn, centroids.shape[0])
        dist_c, ids = tree.query(x, k=k)
        if k == 1:
            dist_c = dist_c[:, None]
            ids = ids[:, None]
        a_ids = np.repeat(np.arange(x.shape[0]), k)
        t_ids = ids.reshape(-1)
        cdist = dist_c.reshape(-1)

        keep = cdist <= self.r_tri[t_ids] + self.offset + 1e-9
        a_ids = a_ids[keep]
        t_ids = t_ids[keep]
        if a_ids.size == 0:
            return (
                a_ids.astype(np.int64),
                t_ids.astype(np.int64),
                np.zeros((0, 3)),
                np.zeros(0),
                np.zeros((0, 3)),
            )

        same = self.tri_panel[t_ids] == self.body_id[a_ids]
        if np.any(same):
            ok = ~same
            ok[same] = self.allowed_local[
                self.node_local[a_ids[same]], self.tri_local[t_ids[same]]
            ]
            a_ids = a_ids[ok]
            t_ids = t_ids[ok]
        if a_ids.size == 0:
            return (
                a_ids.astype(np.int64),
                t_ids.astype(np.int64),
                np.zeros((0, 3)),
                np.zeros(0),
                np.zeros((0, 3)),
            )

        a = x[self.tris[t_ids, 0]]
        b = x[self.tris[t_ids, 1]]
        c = x[self.tris[t_ids, 2]]
        p = x[a_ids]
        bary = _triangle_barycentric(p, a, b, c)
        closest = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
        delta = p - closest
        dist = np.linalg.norm(delta, axis=1)
        safe = np.maximum(dist, 1e-14)
        nrm = delta / safe[:, None]
        degen = dist < 1e-12
        if np.any(degen):
            e1 = b - a
            e2 = c - a
            tn = np.cross(e1, e2)
            tn = tn / np.maximum(np.linalg.norm(tn, axis=1), 1e-30)[:, None]
            sgn = np.sign(np.einsum("ij,ij->i", p - a, tn))
            sgn = np.where(sgn == 0.0, 1.0, sgn)
            nrm[degen] = tn[degen] * sgn[degen, None]
        return a_ids.astype(np.int64), t_ids.astype(np.int64), bary, dist, nrm

    # ---------------------------------------------------------------- solvers
    def _solve_springs(self, x):
        i = self.si
        j = self.sj
        d = x[j] - x[i]
        length = np.sqrt(np.einsum("ij,ij->i", d, d))
        length = np.maximum(length, 1e-14)
        c = length - self.sl0
        nrm = d / length[:, None]
        wi = self.w[i]
        wj = self.w[j]
        denom = wi + wj + self.salpha
        dlam = -c / denom
        np.add.at(x, i, (-(wi * dlam))[:, None] * nrm)
        np.add.at(x, j, ((wj * dlam))[:, None] * nrm)

    def _solve_board(self, x, v, h):
        sd, nrm = _box_sdf(x, self.board_half)
        gap = sd - self.board_offset
        active = gap < 0.0
        if not np.any(active):
            return
        corr = -gap[active][:, None] * nrm[active]
        x[active] += corr
        mag = np.abs(gap[active])
        rel = v[active]
        vn = np.einsum("ij,ij->i", rel, nrm[active])
        vt = rel - vn[:, None] * nrm[active]
        want = -vt * h
        want_mag = np.linalg.norm(want, axis=1)
        cap = self.mu * mag
        scale = np.where(want_mag > 1e-30, np.minimum(1.0, cap / np.maximum(want_mag, 1e-30)), 0.0)
        x[active] += want * scale[:, None]

    def _solve_contacts(self, x, v, h):
        a_ids, t_ids, bary, dist, nrm = self._detect(x)
        if a_ids.size == 0:
            return
        active = dist < self.offset
        if not np.any(active):
            return
        a_ids = a_ids[active]
        t_ids = t_ids[active]
        bary = bary[active]
        dist = dist[active]
        nrm = nrm[active]

        tri_nodes = self.tris[t_ids]
        wn = self.w[a_ids]
        wt = np.zeros(a_ids.shape[0])
        for kk in range(3):
            wt += bary[:, kk] ** 2 * self.w[tri_nodes[:, kk]]
        denom = wn + wt
        c = dist - self.offset
        dlam = -c / denom
        dlam = np.maximum(dlam, 0.0)

        base = (wn * dlam)[:, None] * nrm
        np.add.at(x, a_ids, base)
        for kk in range(3):
            share = -(self.w[tri_nodes[:, kk]] * bary[:, kk] * dlam)[:, None] * nrm
            np.add.at(x, tri_nodes[:, kk], share)

        # Coulomb friction: tangential slip limited by mu * normal correction.
        rel_v = v[a_ids].copy()
        for kk in range(3):
            rel_v -= bary[:, kk : kk + 1] * v[tri_nodes[:, kk]]
        vn = np.einsum("ij,ij->i", rel_v, nrm)
        vt = rel_v - vn[:, None] * nrm
        want = -vt * h
        want_mag = np.linalg.norm(want, axis=1)
        cap = self.mu * (wn * dlam)
        scale = np.where(
            want_mag > 1e-30, np.minimum(1.0, cap / np.maximum(want_mag, 1e-30)), 0.0
        )
        slip = want * scale[:, None]
        np.add.at(x, a_ids, slip)
        for kk in range(3):
            np.add.at(x, tri_nodes[:, kk], -bary[:, kk : kk + 1] * slip)

    def _step(self, h):
        x = self.x
        v = self.v + self.g * h
        v *= 1.0 / (1.0 + self.damping * h)
        self.salpha = self.spring_alpha / (h * h)
        x = x + v * h
        self._solve_springs(x)
        self._solve_contacts(x, v, h)
        self._solve_board(x, v, h)
        self.v = (x - self.x) / h
        self.x = x

    # ---------------------------------------------------------------- outputs
    def _measure(self):
        gap = np.full(self.n, np.inf)
        sd, _ = _box_sdf(self.x, self.board_half)
        gap = np.minimum(gap, sd - self.board_offset)
        a_ids, _t, _b, dist, _n = self._detect(self.x)
        if a_ids.size:
            np.minimum.at(gap, a_ids, dist - self.offset)
        # inextensibility constraint: largest material edge deviation
        i = self.si[self.struct_mask]
        j = self.sj[self.struct_mask]
        d = self.x[j] - self.x[i]
        length = np.sqrt(np.einsum("ij,ij->i", d, d))
        violation = float(np.max(np.abs(length - self.sl0[self.struct_mask])))
        return gap, violation

    def advance_to(self, time: float):
        target = float(time)
        if target > self.t + 1e-12:
            nsub = max(1, int(round((target - self.t) / self.dt_sub)))
            h = (target - self.t) / nsub
            for _ in range(nsub):
                self._step(h)
            self.t = target
            self.last_nsub = nsub
        gap, violation = self._measure()
        attributes = {"mass_quantity": self.mass.copy()}
        if self.uv is not None:
            attributes["uv"] = self.uv
        fields = {
            "contact_signed_distance": gap,
            "constraint_value": np.array([violation]),
            "constraint_target": np.zeros(1),
            "cumulative_mass_source": np.zeros(1),
        }
        scalars = {
            "contact_length_scale": self.contact_length_scale,
            "constraint_scale": self.constraint_scale,
        }
        frame = StateFrame(
            time=target,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "xpbd_shell_contact",
            "nodes": int(self.n),
            "panels": int(self.panel_count),
            "thickness": float(self.thickness),
            "friction": float(self.mu),
            "substep": float(self.dt_sub),
            "time": float(self.t),
        }
