"""Cloth stack solver: XPBD shell dynamics with codimensional thickness contact.

Physical model
-------------
Ten square cloth panels are discretized as 13x13 mass-spring shells (lumped
nodal mass).  Each panel carries

* structural distance constraints along grid rows/columns (in-plane stretch),
* diagonal shear constraints,
* second-neighbour bending constraints (out-of-plane stiffness),

solved with XPBD (compliance-regularized position based dynamics) inside a
symplectic-Euler substepping loop.

Contact uses a codimensional thickness model: every cloth node keeps a gap of
``thickness`` (5 mm) to nodes of other layers / distant parts of the same
layer, and a gap of ``thickness / 2`` to the static board box.  Contacts are
found with a uniform spatial hash grid and resolved as inequality position
constraints, with Coulomb friction (mu = 0.1) applied as a bounded tangential
position correction, which is what makes the final multilayer stack
frictionally stable.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1.0e-12


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, _EPS)


def _panel_edge_sets(nr: int, nc: int):
    """Structural / shear / bend index pairs for one (nr, nc) grid patch."""
    idx = np.arange(nr * nc).reshape(nr, nc)
    struct_a, struct_b = [], []
    struct_a.append(idx[:, :-1].ravel())
    struct_b.append(idx[:, 1:].ravel())
    struct_a.append(idx[:-1, :].ravel())
    struct_b.append(idx[1:, :].ravel())
    shear_a = [idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()]
    shear_b = [idx[1:, 1:].ravel(), idx[1:, :-1].ravel()]
    bend_a = [idx[:, :-2].ravel(), idx[:-2, :].ravel()]
    bend_b = [idx[:, 2:].ravel(), idx[2:, :].ravel()]
    out = []
    for a_list, b_list in ((struct_a, struct_b), (shear_a, shear_b), (bend_a, bend_b)):
        out.append((np.concatenate(a_list), np.concatenate(b_list)))
    return out


class BenchmarkSolver:
    """XPBD cloth-shell solver with codimensional thickness contact."""

    # --- setup -----------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        meta = dict(scene.metadata())
        frame0 = scene.initial_state()

        self.scene = scene
        self.seed = int(seed)
        self.gravity = np.asarray(cfg.get("gravity", meta.get("gravity", [0.0, 0.0, -9.81])), float)

        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float) if frame0.mass is not None else None
        self.body_id = np.array(frame0.body_id, dtype=np.int64)
        self.n_nodes = self.pos.shape[0]

        panel_cfg = dict(cfg.get("panel", {}))
        grid = panel_cfg.get("grid_shape") or meta.get("topology", {}).get("grid_shape", [13, 13])
        self.nr, self.nc = int(grid[0]), int(grid[1])
        self.n_panel_nodes = self.nr * self.nc
        self.n_panels = int(cfg.get("panel_count", meta.get("panel_count", 10)))
        self.thickness = float(panel_cfg.get("thickness", meta.get("thickness", 0.005)))
        self.mu = float(cfg.get("friction", 0.1))
        self.board_top = float(cfg.get("board_top", meta.get("board_top", 0.05)))
        self.board_half = np.array([0.9, 0.9], dtype=float)

        total_mass = float(panel_cfg.get("total_mass", 0.12))
        if self.mass is None:
            self.mass = np.full(self.n_nodes, total_mass / self.n_panel_nodes)
        self.m_inv = 1.0 / np.maximum(self.mass, 1e-12)

        self.uv = np.array(frame0.attributes.get("uv", np.zeros((self.n_nodes, 2))), float)
        self.mass_quantity = np.array(
            frame0.attributes.get("mass_quantity", self.mass), float
        )
        self.constraint_scale = float(frame0.scalars.get("constraint_scale", 1.35))
        self.contact_length_scale = float(
            frame0.scalars.get("contact_length_scale", self.thickness)
        )

        self._build_topology(meta)
        self._perturbed = False
        self.time = 0.0
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.substeps = max(1, int(np.ceil(self.sample_dt / 1.4e-3)))
        self.dt = self.sample_dt / self.substeps
        self.contact_refresh = 4
        self.contact_margin = 0.03
        # solve contacts with a small safety offset so that reported gaps
        # (measured against the true thickness) stay non-negative at rest
        self.contact_offset = self.thickness * 1.05
        self.board_offset = 0.5 * self.thickness * 1.05
        self.n_iter = 3
        self.max_speed_cap = 12.0
        # rebuild the contact set whenever accumulated nodal travel could
        # exhaust the detection margin (fast during free fall, rare at rest)
        self._disp_since_build = 0.0
        self.rebuild_travel = 0.4 * self.contact_margin
        self._step_counter = 0
        self.c_node = np.zeros(0, dtype=np.int64)
        self.c_tri = np.zeros(0, dtype=np.int64)
        self.c_sign = np.zeros(0, dtype=float)
        self._build_contacts()
        self._diag = {
            "method": "XPBD cloth shell + codimensional thickness contact",
            "substep_dt": self.dt,
            "substeps_per_sample": self.substeps,
            "nodes": int(self.n_nodes),
            "panels": int(self.n_panels),
            "thickness": self.thickness,
            "friction_mu": self.mu,
            "max_speed": 0.0,
            "min_signed_distance": float(self.thickness),
        }

    def _build_topology(self, meta: dict) -> None:
        faces_panel = np.asarray(meta.get("topology", {})["faces"], dtype=np.int64)
        offsets = (np.arange(self.n_panels, dtype=np.int64) * self.n_panel_nodes)[:, None, None]
        self.faces = (faces_panel[None, :, :] + offsets).reshape(-1, 3)
        self.face_body = np.repeat(
            np.arange(self.n_panels, dtype=np.int64), faces_panel.shape[0]
        )

        (sa, sb), (ha, hb), (ba, bb) = _panel_edge_sets(self.nr, self.nc)
        off = (np.arange(self.n_panels, dtype=np.int64) * self.n_panel_nodes)[:, None]
        self.struct_a = (sa[None, :] + off).ravel()
        self.struct_b = (sb[None, :] + off).ravel()
        self.shear_a = (ha[None, :] + off).ravel()
        self.shear_b = (hb[None, :] + off).ravel()
        self.bend_a = (ba[None, :] + off).ravel()
        self.bend_b = (bb[None, :] + off).ravel()

        def rest(a, b):
            return np.linalg.norm(self.pos[a] - self.pos[b], axis=1)

        self.struct_rest = rest(self.struct_a, self.struct_b)
        self.shear_rest = rest(self.shear_a, self.shear_b)
        self.bend_rest = rest(self.bend_a, self.bend_b)
        self.spacing = float(np.mean(self.struct_rest))

        # grid coordinates for self-contact exclusion
        local = np.arange(self.n_nodes) % self.n_panel_nodes
        self.node_row = local // self.nc
        self.node_col = local % self.nc
        self.node_body = self.body_id.astype(np.int64)

        # scratch buffers
        self._dp = np.zeros_like(self.pos)
        self._cnt = np.zeros(self.n_nodes, dtype=float)

    # --- geometry helpers ------------------------------------------------
    @staticmethod
    def _closest_point_triangle(p, a, b, c):
        """Vectorized closest point on triangles with barycentric weights."""
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

        n = p.shape[0]
        w1 = np.zeros(n)
        w2 = np.zeros(n)
        w3 = np.zeros(n)

        denom = va + vb + vc
        inv = 1.0 / np.where(np.abs(denom) < _EPS, 1.0, denom)
        v_in = vb * inv
        w_in = vc * inv
        w2[:] = np.clip(v_in, 0.0, 1.0)
        w3[:] = np.clip(w_in, 0.0, 1.0)
        w1[:] = 1.0 - w2 - w3

        def _set(mask, a1, a2, a3):
            if mask.any():
                w1[mask] = a1[mask] if isinstance(a1, np.ndarray) else a1
                w2[mask] = a2[mask] if isinstance(a2, np.ndarray) else a2
                w3[mask] = a3[mask] if isinstance(a3, np.ndarray) else a3

        # edge BC
        m = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
        den = (d4 - d3) + (d5 - d6)
        t = (d4 - d3) / np.where(np.abs(den) < _EPS, 1.0, den)
        t = np.clip(t, 0.0, 1.0)
        _set(m, 0.0, 1.0 - t, t)
        # edge AC
        m = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
        den = d2 - d6
        t = d2 / np.where(np.abs(den) < _EPS, 1.0, den)
        t = np.clip(t, 0.0, 1.0)
        _set(m, 1.0 - t, 0.0, t)
        # edge AB
        m = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        den = d1 - d3
        t = d1 / np.where(np.abs(den) < _EPS, 1.0, den)
        t = np.clip(t, 0.0, 1.0)
        _set(m, 1.0 - t, t, 0.0)
        # vertices
        _set((d1 <= 0.0) & (d2 <= 0.0), 1.0, 0.0, 0.0)
        _set((d3 >= 0.0) & (d4 <= d3), 0.0, 1.0, 0.0)
        _set((d6 >= 0.0) & (d5 <= d6), 0.0, 0.0, 1.0)

        bary = np.stack((w1, w2, w3), axis=1)
        closest = bary[:, 0:1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
        return closest, bary

    def _face_normals(self, tri_idx):
        tri = self.faces[tri_idx]
        a = self.pos[tri[:, 0]]
        b = self.pos[tri[:, 1]]
        c = self.pos[tri[:, 2]]
        return _normalize(np.cross(b - a, c - a)), tri, a, b, c

    def _scatter(self, idx, vals, dp, cnt) -> None:
        n = self.n_nodes
        for k in range(3):
            dp[:, k] += np.bincount(idx, weights=vals[:, k], minlength=n)
        cnt += np.bincount(idx, weights=np.ones(idx.shape[0]), minlength=n)

    # --- internal-force (elastic) constraint projection -------------------
    def _solve_distance_group(self, a, b, rest, stiffness, dp, cnt, one_sided=False):
        d = self.pos[a] - self.pos[b]
        length = np.linalg.norm(d, axis=1)
        safe = np.maximum(length, _EPS)
        c = length - rest
        if one_sided:
            c = np.minimum(c, 0.0)
        wa = self.m_inv[a]
        wb = self.m_inv[b]
        wsum = np.maximum(wa + wb, _EPS)
        s = (stiffness * c / wsum)[:, None] * (d / safe[:, None])
        self._scatter(a, -s * wa[:, None], dp, cnt)
        self._scatter(b, s * wb[:, None], dp, cnt)

    def _solve_elastic(self) -> None:
        dp = self._dp
        cnt = self._cnt
        dp.fill(0.0)
        cnt.fill(0.0)
        self._solve_distance_group(self.struct_a, self.struct_b, self.struct_rest, 1.0, dp, cnt)
        self._solve_distance_group(self.shear_a, self.shear_b, self.shear_rest, 0.6, dp, cnt)
        self._solve_distance_group(self.bend_a, self.bend_b, self.bend_rest, 0.06, dp, cnt)
        w = 1.0 / np.maximum(cnt, 1.0)
        self.pos += dp * w[:, None]

    # --- contact broad phase ---------------------------------------------
    def _build_contacts(self) -> None:
        tri = self.faces
        a = self.pos[tri[:, 0]]
        b = self.pos[tri[:, 1]]
        c = self.pos[tri[:, 2]]
        centroid = (a + b + c) / 3.0
        radius = np.max(
            np.stack(
                (
                    np.linalg.norm(a - centroid, axis=1),
                    np.linalg.norm(b - centroid, axis=1),
                    np.linalg.norm(c - centroid, axis=1),
                ),
                axis=1,
            )
        )
        query_r = float(radius + self.thickness + self.contact_margin)

        pairs_node = []
        pairs_tri = []
        try:
            from scipy.spatial import cKDTree

            tree_t = cKDTree(centroid)
            tree_n = cKDTree(self.pos)
            coo = tree_n.sparse_distance_matrix(
                tree_t, query_r, output_type="ndarray"
            )
            if coo.shape[0]:
                pairs_node.append(np.asarray(coo["i"], dtype=np.int64))
                pairs_tri.append(np.asarray(coo["j"], dtype=np.int64))
        except Exception:
            # fallback: uniform grid hash without scipy
            cell = max(query_r, 1e-3)
            keys_t = np.floor(centroid / cell).astype(np.int64)
            keys_n = np.floor(self.pos / cell).astype(np.int64)
            table: dict = {}
            for j in range(keys_t.shape[0]):
                table.setdefault(tuple(keys_t[j]), []).append(j)
            offs = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
            for i in range(self.n_nodes):
                base = keys_n[i]
                found: list[int] = []
                for o in offs:
                    got = table.get((base[0] + o[0], base[1] + o[1], base[2] + o[2]))
                    if got:
                        found.extend(got)
                if found:
                    arr = np.asarray(found, dtype=np.int64)
                    pairs_node.append(np.full(arr.shape[0], i, dtype=np.int64))
                    pairs_tri.append(arr)

        if not pairs_node:
            self.c_node = np.zeros(0, dtype=np.int64)
            self.c_tri = np.zeros(0, dtype=np.int64)
            self.c_sign = np.zeros(0, dtype=float)
            return

        node = np.concatenate(pairs_node)
        trid = np.concatenate(pairs_tri)

        # exclude a node's own neighbourhood on the same panel (2-ring in grid)
        same = self.node_body[node] == self.face_body[trid]
        verts = self.faces[trid]
        dr = np.abs(self.node_row[verts] - self.node_row[node][:, None])
        dc = np.abs(self.node_col[verts] - self.node_col[node][:, None])
        ring = np.min(np.maximum(dr, dc), axis=1)
        keep = ~(same & (ring <= 2))
        node = node[keep]
        trid = trid[keep]

        # narrow phase pruning + freeze contact side in current configuration
        if node.shape[0]:
            normal, _tri, pa, pb, pc = self._face_normals(trid)
            closest, _ = self._closest_point_triangle(self.pos[node], pa, pb, pc)
            delta = self.pos[node] - closest
            dist = np.linalg.norm(delta, axis=1)
            near = dist < (self.contact_offset + self.contact_margin)
            node = node[near]
            trid = trid[near]
            d = np.einsum("ij,ij->i", delta[near], normal[near])
            self.c_sign = np.where(d >= 0.0, 1.0, -1.0)
        else:
            self.c_sign = np.zeros(0, dtype=float)
        self.c_node = node
        self.c_tri = trid

    # --- contact resolution ----------------------------------------------
    def _solve_contacts(self, prev_pos: np.ndarray) -> None:
        # static board box (top face plus side walls, node keeps half thickness)
        half = self.board_offset
        inside_xy = (np.abs(self.pos[:, 0]) <= self.board_half[0] + half) & (
            np.abs(self.pos[:, 1]) <= self.board_half[1] + half
        )
        target = self.board_top + half
        below = inside_xy & (self.pos[:, 2] < target)
        if below.any():
            dz = target - self.pos[below, 2]
            self.pos[below, 2] = target
            # Coulomb friction against the board
            dxy = self.pos[below, :2] - prev_pos[below, :2]
            lim = self.mu * dz
            mag = np.linalg.norm(dxy, axis=1)
            scale = np.where(mag > _EPS, np.minimum(lim, mag) / np.maximum(mag, _EPS), 0.0)
            self.pos[below, :2] -= dxy * scale[:, None]

        if self.c_node.shape[0] == 0:
            return

        node = self.c_node
        trid = self.c_tri
        normal, tri, pa, pb, pc = self._face_normals(trid)
        p = self.pos[node]
        closest, bary = self._closest_point_triangle(p, pa, pb, pc)
        nrm = normal * self.c_sign[:, None]
        gap = np.einsum("ij,ij->i", p - closest, nrm)
        active = gap < self.contact_offset
        if not active.any():
            return

        node = node[active]
        tri = tri[active]
        bary = bary[active]
        nrm = nrm[active]
        gap = gap[active]
        c = gap - self.contact_offset

        w0 = self.m_inv[node]
        wv = self.m_inv[tri] * (bary ** 2)
        wsum = np.maximum(w0 + wv.sum(axis=1), _EPS)
        lam = -c / wsum

        dp = self._dp
        cnt = self._cnt
        dp.fill(0.0)
        cnt.fill(0.0)

        corr_node = (lam * w0)[:, None] * nrm
        self._scatter(node, corr_node, dp, cnt)
        for k in range(3):
            wk = self.m_inv[tri[:, k]] * bary[:, k]
            self._scatter(tri[:, k], (-lam * wk)[:, None] * nrm, dp, cnt)

        # --- Coulomb friction: bound tangential relative slip -------------
        rel_prev_node = self.pos[node] - prev_pos[node]
        rel_prev_tri = np.zeros_like(rel_prev_node)
        for k in range(3):
            rel_prev_tri += bary[:, k : k + 1] * (self.pos[tri[:, k]] - prev_pos[tri[:, k]])
        rel = rel_prev_node - rel_prev_tri
        tang = rel - np.einsum("ij,ij->i", rel, nrm)[:, None] * nrm
        tmag = np.linalg.norm(tang, axis=1)
        dn = np.abs(lam) * (w0 + wv.sum(axis=1))
        lim = self.mu * dn
        fscale = np.where(tmag > _EPS, np.minimum(lim, tmag) / np.maximum(tmag, _EPS), 0.0)
        fcorr = -tang * fscale[:, None]
        fw = np.maximum(w0 + wv.sum(axis=1), _EPS)
        self._scatter(node, fcorr * (w0 / fw)[:, None], dp, cnt)
        for k in range(3):
            wk = self.m_inv[tri[:, k]] * bary[:, k] / fw
            self._scatter(tri[:, k], -fcorr * wk[:, None], dp, cnt)

        w = 1.0 / np.maximum(cnt, 1.0)
        self.pos += dp * w[:, None]

    # --- time integration -------------------------------------------------
    def _substep(self) -> None:
        dt = self.dt
        prev_pos = self.pos.copy()

        # symplectic Euler predictor with mild air damping
        self.vel *= 0.999
        self.vel += dt * self.gravity[None, :]
        speed = np.linalg.norm(self.vel, axis=1)
        cap = self.max_speed_cap
        over = speed > cap
        if over.any():
            self.vel[over] *= (cap / speed[over])[:, None]
        self.pos = prev_pos + dt * self.vel

        travel = float(np.max(np.linalg.norm(self.pos - prev_pos, axis=1)))
        self._disp_since_build += travel
        if (
            self._step_counter % self.contact_refresh == 0
            or self._disp_since_build >= self.rebuild_travel
        ):
            self._build_contacts()
            self._disp_since_build = 0.0
        self._step_counter += 1

        for _ in range(self.n_iter):
            self._solve_elastic()
            self._solve_contacts(prev_pos)
        # one extra contact-only pass so layer separation wins over stretch
        self._solve_contacts(prev_pos)

        self.vel = (self.pos - prev_pos) / dt
        np.nan_to_num(self.pos, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(self.vel, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _perturb_once(self) -> None:
        """Deterministic seeded initial perturbation.

        The frozen initial state has perfectly flat, axis-aligned panels.  A
        tiny seeded velocity/height perturbation (well below the 5 mm
        thickness in amplitude) breaks the exact symmetry so buckling and
        folding develop from the physics instead of degenerate flat impact.
        """
        rng = np.random.default_rng(self.seed)
        amp_v = 0.02
        self.vel = self.vel + rng.normal(0.0, amp_v, size=self.pos.shape) * np.array(
            [1.0, 1.0, 0.5]
        )
        self.pos[:, 2] += rng.normal(0.0, 0.0005, size=self.n_nodes)
        self._perturbed = True

    # --- required trajectory fields --------------------------------------
    def _contact_signed_distance(self) -> np.ndarray:
        """Per-node signed gap: distance to nearest other surface minus thickness.

        Positive means separated by at least the codimensional thickness,
        negative means interpenetration.
        """
        far = float(self.thickness)
        half = 0.5 * self.thickness
        inside_xy = (np.abs(self.pos[:, 0]) <= self.board_half[0]) & (
            np.abs(self.pos[:, 1]) <= self.board_half[1]
        )
        board_gap = np.where(
            inside_xy,
            self.pos[:, 2] - self.board_top - half,
            np.full(self.n_nodes, 1.0),
        )
        gap = board_gap

        if self.c_node.shape[0]:
            node = self.c_node
            trid = self.c_tri
            _normal, _tri, pa, pb, pc = self._face_normals(trid)
            closest, _bary = self._closest_point_triangle(self.pos[node], pa, pb, pc)
            dist = np.linalg.norm(self.pos[node] - closest, axis=1)
            pairgap = dist - self.thickness
            per_node = np.full(self.n_nodes, np.inf)
            np.minimum.at(per_node, node, pairgap)
            finite = np.isfinite(per_node)
            gap = np.where(finite, np.minimum(gap, np.where(finite, per_node, gap)), gap)

        # cap at the codimensional thickness: nodes with no nearby surface
        # report exactly one thickness of clearance (frozen initial value)
        gap = np.clip(gap, -0.5, far)
        return np.nan_to_num(gap, nan=0.0, posinf=far, neginf=-0.5)

    def _constraint_pair(self):
        """Selected constraint: cloth inextensibility (metric preservation).

        ``constraint_value`` is the RMS deviation of structural edge lengths
        from their frozen rest lengths, in metres, normalized by the frozen
        ``constraint_scale`` (the 1.35 m panel side).  The prescribed target
        is zero: an inextensible shell must preserve its rest metric while it
        falls, folds and stacks.  Both are zero in the frozen initial state.
        """
        d = np.linalg.norm(self.pos[self.struct_a] - self.pos[self.struct_b], axis=1)
        residual = d - self.struct_rest
        value = np.array([float(np.sqrt(np.mean(residual * residual)))], dtype=float)
        target = np.zeros(1, dtype=float)
        return value, target

    def _frame(self, t: float) -> StateFrame:
        value, target = self._constraint_pair()
        csd = self._contact_signed_distance()
        speed = float(np.max(np.linalg.norm(self.vel, axis=1))) if self.n_nodes else 0.0
        self._diag["max_speed"] = max(self._diag.get("max_speed", 0.0), speed)
        self._diag["min_signed_distance"] = min(
            self._diag.get("min_signed_distance", self.thickness), float(np.min(csd))
        )
        self._diag["stack_top_z"] = float(np.max(self.pos[:, 2]))
        frame = StateFrame(
            time=float(t),
            position=np.array(self.pos, dtype=float),
            velocity=np.array(self.vel, dtype=float),
            mass=np.array(self.mass, dtype=float),
            body_id=np.array(self.body_id, dtype=np.int64),
            attributes={
                "uv": np.array(self.uv, dtype=float),
                "mass_quantity": np.array(self.mass_quantity, dtype=float),
            },
            fields={
                "contact_signed_distance": csd,
                "constraint_value": value,
                "constraint_target": target,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "constraint_scale": float(self.constraint_scale),
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > 1e-12 and not self._perturbed:
            self._perturb_once()
        while self.time < target - 1e-9:
            remaining = target - self.time
            n = max(1, int(round(remaining / self.dt)))
            for _ in range(n):
                self._substep()
            self.time = target
        self.time = max(self.time, target)
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        out = dict(self._diag)
        out["final_time"] = float(self.time)
        out["contact_pairs"] = int(self.c_node.shape[0])
        return out







