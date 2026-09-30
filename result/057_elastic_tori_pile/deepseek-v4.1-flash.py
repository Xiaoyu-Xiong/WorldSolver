"""Position-based volumetric solver for the pile of forty incompressible elastic tori.

Physical model
--------------
Every torus is treated as a three-dimensional solid.  Its 288 scaffold surface
samples are completed by 24 interior ring-centre samples, which yields a closed
volumetric tetrahedral mesh (a triangular-prism decomposition of the tube,
864 signed-volume tetrahedra per body).  The material is modelled as a volume
conserving elastic solid: one incompressibility (volume) constraint per
tetrahedron plus distance constraints per mesh edge supply the stored energy,
gravity supplies the body force, and non-penetration contacts against the rigid
floor, the twelve vertical posts and the other tori supply the boundary
tractions.

Integration
-----------
An XPBD (extended position based dynamics) time stepping scheme is used with a
fixed sub-step.  Gravity is integrated explicitly, the elastic constraints are
projected with Gauss-Seidel sweeps over colour classes (volume) and an
averaged Jacobi sweep (edges), and contacts are resolved by positional
projection plus Coulomb friction impulses.  The scheme is deterministic and
every produced state is finite.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Frozen scaffold quantities (fallbacks; live values are read from the config).
# ---------------------------------------------------------------------------
U_COUNT = 24
V_COUNT = 12
SURF_PER_BODY = U_COUNT * V_COUNT            # 288 scaffold surface samples
NODE_PER_BODY = SURF_PER_BODY + U_COUNT      # + 24 interior ring centres
TETS_PER_BODY = U_COUNT * V_COUNT * 3        # 864 tetrahedra
EDGES_PER_BODY = U_COUNT * V_COUNT * 3 + U_COUNT

DEFAULT_BODY_COUNT = 40
DEFAULT_BODY_MASS = 0.90
DEFAULT_SAMPLE_DT = 0.02
DEFAULT_DURATION = 10.0

SUBSTEPS = 4              # sub-steps per sample interval (dt = 5 ms)
ITERATIONS = 6            # constraint sweeps per sub-step
EDGE_COMPLIANCE = 5.0e-4   # m/N, elastic stretch compliance of a rim/rim edge
CORE_COMPLIANCE = 5.0e-2   # m/N, soft interior spokes (allows tube flattening)
VOLUME_COMPLIANCE = 1.0e-8  # 1/Pa, near-rigid incompressibility
DAMPING = 0.9998          # velocity damping per sub-step (material damping)
MAX_SPEED = 30.0          # m/s safety clamp
MAX_ITER_DISP = 0.05      # m, per-sweep displacement clamp
MAX_CONTACT_DISP = 0.03   # m, penetration recovery clamp per sub-step
CONTACT_RADIUS = 0.04     # m, effective node radius for body-body contact
CONTACT_CUTOFF = 2.0 * CONTACT_RADIUS
FRICTION = 0.5            # Coulomb coefficient for all dry contacts
FLOOR_Z = 0.0
POST_RADIUS = 0.13
POST_HEIGHT = 2.8
POST_XZ = (
    (2.35, 0.0), (2.0352, 1.175), (1.175, 2.0352), (0.0, 2.35),
    (-1.175, 2.0352), (-2.0352, 1.175), (-2.35, 0.0), (-2.0352, -1.175),
    (-1.175, -2.0352), (0.0, -2.35), (1.175, -2.0352), (2.0352, -1.175),
)


# ---------------------------------------------------------------------------
# Frozen mesh topology (independent of the current nodal positions)
# ---------------------------------------------------------------------------
def _local_topology():
    """Return (tets, edges, kinds, tet_colour, tet_slot) in local index space."""
    tets = np.zeros((TETS_PER_BODY, 4), dtype=np.int64)
    colours = np.zeros(TETS_PER_BODY, dtype=np.int64)
    slots = np.zeros(TETS_PER_BODY, dtype=np.int64)
    edges = []
    kinds = []
    cursor = 0
    for i in range(U_COUNT):
        i2 = (i + 1) % U_COUNT
        c0 = SURF_PER_BODY + i
        c1 = SURF_PER_BODY + i2
        for j in range(V_COUNT):
            k = (j + 1) % V_COUNT
            b0 = i * V_COUNT + j
            b1 = i * V_COUNT + k
            b2 = i2 * V_COUNT + j
            b3 = i2 * V_COUNT + k
            # triangular prism (c0,b0,b1) -> (c1,b2,b3) split into 3 tets
            for slot, tet in enumerate(((c0, b0, b1, c1), (b0, b1, c1, b2), (b1, c1, b2, b3))):
                tets[cursor] = tet
                colours[cursor] = (i % 2) * 2 + (j % 2)
                slots[cursor] = slot
                cursor += 1
    for i in range(U_COUNT):
        i2 = (i + 1) % U_COUNT
        for j in range(V_COUNT):
            j2 = (j + 1) % V_COUNT
            edges.append((i * V_COUNT + j, i2 * V_COUNT + j))   # axial ring edge
            edges.append((i * V_COUNT + j, i * V_COUNT + j2))   # circumferential edge
            edges.append((SURF_PER_BODY + i, i * V_COUNT + j))  # cross-section spoke
            kinds.extend((0, 0, 1))
        edges.append((SURF_PER_BODY + i, SURF_PER_BODY + i2))   # centre-line edge
        kinds.append(1)
    return (tets, np.asarray(edges, dtype=np.int64), np.asarray(kinds, dtype=np.int64),
            colours, slots)


def _inverse_edge_matrices(ref, tets):
    """Batched inverse of the reference edge matrices (Ne,3,3)."""
    a = ref[tets[:, 1]] - ref[tets[:, 0]]
    b = ref[tets[:, 2]] - ref[tets[:, 0]]
    c = ref[tets[:, 3]] - ref[tets[:, 0]]
    bc = np.cross(b, c)
    ca = np.cross(c, a)
    ab = np.cross(a, b)
    det = np.einsum("ij,ij->i", a, bc)
    safe = np.where(np.abs(det) < 1e-14, 1e-14, det)
    inv = np.stack((bc, ca, ab), axis=1) / safe[:, None, None]
    return inv


def _tet_volumes(x, tets):
    """Signed tetrahedron volumes for the given nodal configuration."""
    e1 = x[tets[:, 1]] - x[tets[:, 0]]
    e2 = x[tets[:, 2]] - x[tets[:, 0]]
    e3 = x[tets[:, 3]] - x[tets[:, 0]]
    return np.einsum("ij,ij->i", e1, np.cross(e2, e3)) / 6.0


def _scatter(rows, values, size):
    """Accumulate values into a flat array of the given size (3 components)."""
    out = np.zeros((size, 3), dtype=float)
    for comp in range(3):
        out[:, comp] = np.bincount(rows, weights=values[:, comp], minlength=size)
    return out


class BenchmarkSolver:
    """Deterministic XPBD solid solver for the forty-torus pile scene."""

    def __init__(self) -> None:
        self._ready = False
        self.time = 0.0
        self._report = {}
        self._notes = {"substeps": SUBSTEPS, "iterations": ITERATIONS}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        geometry = config.get("scene_geometry", {}) or {}
        initial = config.get("initial_conditions", {}) or {}

        self.sample_dt = float(config.get("sample_dt", DEFAULT_SAMPLE_DT))
        self.duration = float(config.get("duration", DEFAULT_DURATION))
        self.gravity = np.asarray(config.get("gravity", (0.0, 0.0, -9.81)), dtype=float).reshape(3)
        self.n_body = int(initial.get("torus_count", DEFAULT_BODY_COUNT))
        self.body_mass = DEFAULT_BODY_MASS
        self._read_static_geometry(geometry)

        frame = scene.initial_state()
        position = np.asarray(frame.position, dtype=float)
        body_id = np.asarray(frame.body_id).reshape(-1).astype(np.int64)
        n_surf = self.n_body * SURF_PER_BODY
        if position.shape != (n_surf, 3):
            raise ValueError("initial state does not match the frozen torus count")
        # Rebuild the surface samples body by body in scaffold vertex order.
        surface = np.empty_like(position)
        for b in range(self.n_body):
            rows = np.flatnonzero(body_id == b)
            if rows.size != SURF_PER_BODY:
                raise ValueError("each torus must expose 288 surface samples")
            surface[b * SURF_PER_BODY:(b + 1) * SURF_PER_BODY] = position[rows]
        self.body_ids = np.repeat(np.arange(self.n_body, dtype=np.int64), SURF_PER_BODY)

        # Interior ring centres are the mean of each cross-section ring.
        rings = surface.reshape(self.n_body, U_COUNT, V_COUNT, 3)
        centres = rings.mean(axis=2).reshape(-1, 3)

        n_node = self.n_body * NODE_PER_BODY
        self.n_node = n_node
        self.surf_idx = (np.arange(n_surf) // SURF_PER_BODY) * NODE_PER_BODY + (
            np.arange(n_surf) % SURF_PER_BODY
        )
        self.centre_idx = (np.arange(self.n_body * U_COUNT) // U_COUNT) * NODE_PER_BODY + (
            SURF_PER_BODY + np.arange(self.n_body * U_COUNT) % U_COUNT
        )

        x0 = np.zeros((n_node, 3), dtype=float)
        x0[self.surf_idx] = surface
        x0[self.centre_idx] = centres

        # Global topology.
        tets_l, edges_l, kinds_l, colours_l, slots_l = _local_topology()
        body_off = np.repeat(np.arange(self.n_body) * NODE_PER_BODY, TETS_PER_BODY)
        self.tets = np.tile(tets_l, (self.n_body, 1)) + body_off[:, None]
        self.tet_colour = np.tile(colours_l, self.n_body)
        self.tet_slot = np.tile(slots_l, self.n_body)
        self.tet_body = np.repeat(np.arange(self.n_body), TETS_PER_BODY)
        self.tet_sel = {
            (c, s): np.flatnonzero((self.tet_colour == c) & (self.tet_slot == s))
            for c in range(4)
            for s in range(3)
        }
        edge_off = np.repeat(np.arange(self.n_body) * NODE_PER_BODY, EDGES_PER_BODY)
        self.edges = np.tile(edges_l, (self.n_body, 1)) + edge_off[:, None]
        edge_kind = np.tile(kinds_l, self.n_body)
        self.edge_alpha = np.where(edge_kind == 1, CORE_COMPLIANCE, EDGE_COMPLIANCE)

        self.node_mass = np.full(n_node, self.body_mass / NODE_PER_BODY, dtype=float)
        self.w = 1.0 / self.node_mass
        self.m_node = self.body_mass / NODE_PER_BODY

        self.ref = x0.copy()
        self.l0 = np.linalg.norm(x0[self.edges[:, 0]] - x0[self.edges[:, 1]], axis=1)
        self.v0 = _tet_volumes(self.ref, self.tets)
        self.dm_inv = _inverse_edge_matrices(self.ref, self.tets)
        self.edge_count = np.maximum(
            np.bincount(self.edges.reshape(-1), minlength=n_node), 1
        ).astype(float)

        self.x = x0.copy()
        self.v = np.zeros_like(x0)
        self.lam_tet = np.zeros(self.tets.shape[0], dtype=float)
        self.lam_edge = np.zeros(self.edges.shape[0], dtype=float)

        self.coll_idx = self.surf_idx.copy()
        self.coll_body = self.body_ids.copy()

        # ---- frozen budgets -------------------------------------------
        self.n_domain = self.n_body + 1
        self.n_term = 4  # gravity, floor, posts, body-body contact
        weights = np.zeros((self.n_domain, n_surf), dtype=np.float32)
        for b in range(self.n_body):
            weights[b, self.body_ids == b] = 1.0
        weights[self.n_body, :] = 1.0
        self.domain_weights = weights
        self.cum = np.zeros((self.n_domain, self.n_term, 3), dtype=float)
        self.cum_snapshot = np.zeros_like(self.cum)
        self.imp_cum = np.zeros((self.n_domain, 3), dtype=float)
        self.imp_snapshot = np.zeros_like(self.imp_cum)
        self.v0_body = np.bincount(self.tet_body, weights=self.v0, minlength=self.n_body)
        self.cumulative_mass_source = np.zeros(1, dtype=float)

        self.surf_mass = np.full(n_surf, self.body_mass / SURF_PER_BODY, dtype=float)
        self._first_frame = True
        self._last_good = (self.x.copy(), self.v.copy())
        self._ready = True

    def _read_static_geometry(self, geometry) -> None:
        posts = []
        floor = FLOOR_Z
        for item in (geometry or {}).get("objects", []) or []:
            geo = item.get("geometry", {}) or {}
            if not item.get("collision_enabled", True):
                continue
            if geo.get("type") == "box":
                half = np.asarray(geo.get("half_extents", (0.0, 0.0, 0.0)), dtype=float)
                trans = np.asarray(item.get("transform", {}).get("translation", (0, 0, 0)), dtype=float)
                floor = max(floor, float(trans[2] + half[2]))
            elif geo.get("type") == "cylinder":
                trans = np.asarray(item.get("transform", {}).get("translation", (0, 0, 0)), dtype=float)
                radius = float(geo.get("radius", POST_RADIUS))
                height = float(geo.get("height", POST_HEIGHT))
                posts.append((float(trans[0]), float(trans[1]), radius, float(trans[2] - height / 2.0),
                              float(trans[2] + height / 2.0)))
        self.floor_z = floor
        if posts:
            self.posts = tuple(posts)
        else:
            self.posts = tuple((x, y, POST_RADIUS, 0.0, POST_HEIGHT) for x, y in POST_XZ)

    # ------------------------------------------------------------------
    # Elastic constraints (XPBD projections)
    # ------------------------------------------------------------------
    def _solve_volumes(self, alpha_tilde: float) -> None:
        x = self.x
        w = self.w
        for colour in range(4):
            for slot in range(3):
                sel = self.tet_sel[(colour, slot)]
                if sel.size == 0:
                    continue
                t = self.tets[sel]
                p0 = x[t[:, 0]]
                p1 = x[t[:, 1]]
                p2 = x[t[:, 2]]
                p3 = x[t[:, 3]]
                e1 = p1 - p0
                e2 = p2 - p0
                e3 = p3 - p0
                vol = np.einsum("ij,ij->i", e1, np.cross(e2, e3)) / 6.0
                g1 = np.cross(e2, e3) / 6.0
                g2 = np.cross(e3, e1) / 6.0
                g3 = np.cross(e1, e2) / 6.0
                g0 = -(g1 + g2 + g3)
                c = vol - self.v0[sel]
                lam = self.lam_tet[sel]
                w0 = w[t[:, 0]]
                w1 = w[t[:, 1]]
                w2 = w[t[:, 2]]
                w3 = w[t[:, 3]]
                denom = (
                    w0 * np.einsum("ij,ij->i", g0, g0)
                    + w1 * np.einsum("ij,ij->i", g1, g1)
                    + w2 * np.einsum("ij,ij->i", g2, g2)
                    + w3 * np.einsum("ij,ij->i", g3, g3)
                    + alpha_tilde
                )
                dlam = (-c - alpha_tilde * lam) / denom
                self.lam_tet[sel] = lam + dlam
                dx = np.zeros((self.n_node, 3), dtype=float)
                for node, grad, wk in (
                    (t[:, 0], g0, w0),
                    (t[:, 1], g1, w1),
                    (t[:, 2], g2, w2),
                    (t[:, 3], g3, w3),
                ):
                    weighted = (wk * dlam)[:, None] * grad
                    for comp in range(3):
                        dx[:, comp] += np.bincount(
                            node, weights=weighted[:, comp], minlength=self.n_node
                        )
                norm = np.linalg.norm(dx, axis=1)
                scale = np.where(norm > MAX_ITER_DISP, MAX_ITER_DISP / np.maximum(norm, 1e-12), 1.0)
                x += dx * scale[:, None]

    def _solve_edges(self, alpha_tilde) -> None:
        x = self.x
        e = self.edges
        p0 = x[e[:, 0]]
        p1 = x[e[:, 1]]
        d = p0 - p1
        dist = np.linalg.norm(d, axis=1)
        safe = np.maximum(dist, 1e-12)
        n = d / safe[:, None]
        c = dist - self.l0
        w0 = self.w[e[:, 0]]
        w1 = self.w[e[:, 1]]
        denom = w0 + w1 + alpha_tilde
        dlam = (-c - alpha_tilde * self.lam_edge) / denom
        self.lam_edge += dlam
        contrib = (w0 * dlam)[:, None] * n
        acc = np.zeros((self.n_node, 3), dtype=float)
        for comp in range(3):
            acc[:, comp] = np.bincount(
                e[:, 0], weights=contrib[:, comp], minlength=self.n_node
            ) - np.bincount(e[:, 1], weights=contrib[:, comp], minlength=self.n_node)
        acc /= self.edge_count[:, None]
        norm = np.linalg.norm(acc, axis=1)
        scale = np.where(norm > MAX_ITER_DISP, MAX_ITER_DISP / np.maximum(norm, 1e-12), 1.0)
        x += acc * scale[:, None]

    # ------------------------------------------------------------------
    # Contacts
    # ------------------------------------------------------------------
    def _hash_pairs(self, xs: np.ndarray):
        """Candidate contact pairs (i<j, distinct bodies) via a spatial hash."""
        n = xs.shape[0]
        h = CONTACT_CUTOFF
        cell = np.floor(xs / h).astype(np.int64)
        cell -= cell.min(axis=0)
        dims = cell.max(axis=0) + 1
        stride_y = int(dims[2])
        stride_x = int(dims[1] * dims[2])
        key = (cell[:, 0] * dims[1] + cell[:, 1]) * dims[2] + cell[:, 2]
        order = np.argsort(key, kind="stable")
        keys = key[order]
        left, right = [], []
        base = np.arange(n, dtype=np.int64)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx < 0 or (dx == 0 and (dy < 0 or (dy == 0 and dz <= 0))):
                        continue
                    nk = key + dx * stride_x + dy * stride_y + dz
                    lo = np.searchsorted(keys, nk, side="left")
                    hi = np.searchsorted(keys, nk, side="right")
                    cnt = hi - lo
                    total = int(cnt.sum())
                    if total == 0:
                        continue
                    src = np.repeat(base, cnt)
                    ramp = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    dst = order[np.repeat(lo, cnt) + ramp]
                    left.append(src)
                    right.append(dst)
        if not left:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        i = np.concatenate(left)
        j = np.concatenate(right)
        keep = self.coll_body[i] != self.coll_body[j]
        i, j = i[keep], j[keep]
        return i, j

    def _detect_contacts(self):
        """Geometric contact query for the floor, the posts and other bodies."""
        xs = self.x[self.coll_idx]
        empty3 = (np.empty(0, dtype=np.int64),) * 3
        z = xs[:, 2]
        mask = z < self.floor_z + 1e-4
        if np.any(mask):
            si = np.flatnonzero(mask)
            normal = np.zeros((si.size, 3))
            normal[:, 2] = 1.0
            floor_contact = (si, normal, self.floor_z - z[si])
        else:
            floor_contact = empty3
        post_parts = []
        for px, py, radius, z0, z1 in self.posts:
            dx = xs[:, 0] - px
            dy = xs[:, 1] - py
            r = np.sqrt(dx * dx + dy * dy)
            mask = (r < radius + 1e-4) & (z >= z0 - CONTACT_RADIUS) & (z <= z1 + CONTACT_RADIUS)
            if not np.any(mask):
                continue
            si = np.flatnonzero(mask)
            rs = np.maximum(r[si], 1e-12)
            normal = np.column_stack((dx[si] / rs, dy[si] / rs, np.zeros(si.size)))
            post_parts.append((si, normal, radius - r[si]))
        if len(post_parts) == 1:
            post_contact = post_parts[0]
        elif post_parts:
            post_contact = (
                np.concatenate([p[0] for p in post_parts]),
                np.vstack([p[1] for p in post_parts]),
                np.concatenate([p[2] for p in post_parts]),
            )
        else:
            post_contact = empty3

        i, j = self._hash_pairs(xs)
        pairs = (np.empty(0, dtype=np.int64),) * 4
        if i.size:
            d = xs[i] - xs[j]
            dist2 = np.einsum("ij,ij->i", d, d)
            keep = (dist2 < CONTACT_CUTOFF * CONTACT_CUTOFF) & (dist2 > 1e-18)
            i, j, d = i[keep], j[keep], d[keep]
            if i.size:
                dist = np.sqrt(np.einsum("ij,ij->i", d, d))
                pairs = (i, j, d / dist[:, None], CONTACT_CUTOFF - dist)
        return floor_contact, post_contact, pairs

    def _apply_positional_contacts(self, floor_contact, post_contact, pairs) -> None:
        """Split-impulse penetration recovery (does not inject velocity)."""
        xs = self.x[self.coll_idx]
        corr = np.zeros_like(xs)
        for contact in (floor_contact, post_contact):
            si, normal, pen = contact
            if si.size:
                corr += _scatter(si, pen[:, None] * normal, xs.shape[0])
        i, j, normal, pen = pairs
        if i.size:
            half = 0.5 * pen[:, None] * normal
            corr += _scatter(i, half, xs.shape[0]) - _scatter(j, half, xs.shape[0])
        norm = np.linalg.norm(corr, axis=1)
        scale = np.where(norm > MAX_CONTACT_DISP, MAX_CONTACT_DISP / np.maximum(norm, 1e-12), 1.0)
        xs += corr * scale[:, None]
        self.x[self.coll_idx] = xs

    def _static_impulses(self, contact, term: int, dv_all) -> None:
        si, normal, _pen = contact
        if si.size == 0:
            return
        nodes = self.coll_idx[si]
        vloc = self.v[nodes]
        vn = np.einsum("ij,ij->i", vloc, normal)
        jn = np.maximum(-vn, 0.0)
        dv = jn[:, None] * normal
        tangential = vloc - vn[:, None] * normal
        speed = np.linalg.norm(tangential, axis=1)
        jt = np.minimum(FRICTION * jn, speed)
        tdir = tangential / np.maximum(speed, 1e-12)[:, None]
        dv = dv - jt[:, None] * tdir
        dv_all += _scatter(nodes, dv, self.n_node)
        weight = self.node_mass * dv
        bodies = self.coll_body[si]
        for comp in range(3):
            contrib = np.bincount(bodies, weights=weight[:, comp], minlength=self.n_body)
            self.cum[: self.n_body, term, comp] += contrib
            self.cum[self.n_body, term, comp] += float(contrib.sum())

    def _apply_contact_impulses(self, floor_contact, post_contact, pairs) -> None:
        dv_all = np.zeros((self.n_node, 3), dtype=float)
        self._static_impulses(floor_contact, 1, dv_all)
        self._static_impulses(post_contact, 2, dv_all)
        i, j, normal, _pen = pairs
        if i.size:
            nodes_i = self.coll_idx[i]
            nodes_j = self.coll_idx[j]
            vrel = self.v[nodes_i] - self.v[nodes_j]
            vn = np.einsum("ij,ij->i", vrel, normal)
            jn = 0.5 * np.maximum(-vn, 0.0)
            dvi = jn[:, None] * normal
            dvj = -dvi
            tangential = vrel - vn[:, None] * normal
            speed = np.linalg.norm(tangential, axis=1)
            jt = np.minimum(FRICTION * jn, 0.5 * speed)
            tdir = tangential / np.maximum(speed, 1e-12)[:, None]
            dvi = dvi - jt[:, None] * tdir
            dvj = dvj + jt[:, None] * tdir
            dv_all += _scatter(nodes_i, dvi, self.n_node) + _scatter(nodes_j, dvj, self.n_node)
            weight_i = self.node_mass * dvi
            weight_j = self.node_mass * dvj
            bodies_i = self.coll_body[i]
            bodies_j = self.coll_body[j]
            for comp in range(3):
                self.cum[: self.n_body, 3, comp] += np.bincount(
                    bodies_i, weights=weight_i[:, comp], minlength=self.n_body
                )
                self.cum[: self.n_body, 3, comp] += np.bincount(
                    bodies_j, weights=weight_j[:, comp], minlength=self.n_body
                )
        self.v += dv_all

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        x = self.x
        v = self.v
        v += self.gravity * dt
        np.clip(v, -MAX_SPEED, MAX_SPEED, out=v)
        x_prev = x.copy()
        x += v * dt

        self.lam_tet[:] = 0.0
        self.lam_edge[:] = 0.0
        a_vol = VOLUME_COMPLIANCE / (dt * dt)
        a_edge = self.edge_alpha / (dt * dt)
        for _ in range(ITERATIONS):
            self._solve_volumes(a_vol)
            self._solve_edges(a_edge)

        floor_contact, post_contact, pairs = self._detect_contacts()

        # Velocity increment produced by gravity plus the elastic projections.
        v[:] = (x - x_prev) / dt
        v *= DAMPING
        np.clip(v, -MAX_SPEED, MAX_SPEED, out=v)
        self._apply_contact_impulses(floor_contact, post_contact, pairs)
        # Split impulse: geometric penetration recovery never injects momentum.
        self._apply_positional_contacts(floor_contact, post_contact, pairs)

        # Gravity impulse budget.
        g_imp = self.body_mass * self.gravity * dt
        self.cum[: self.n_body, 0, :] += g_imp[None, :]
        self.cum[self.n_body, 0, :] += self.n_body * g_imp

        if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()):
            good_x, good_v = self._last_good
            self.x = good_x.copy()
            self.v = good_v.copy()
            self._notes["recovered"] = self._notes.get("recovered", 0) + 1
        else:
            self._last_good = (self.x.copy(), self.v.copy())

    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("initialize() must be called before advance_to()")
        time = float(time)
        if time > self.time + 1e-12:
            steps = max(1, int(round((time - self.time) / self.sample_dt)))
            dt = (time - self.time) / steps
            for _ in range(steps):
                self._substep(dt / SUBSTEPS)
            self.time = time
        return self._make_frame(self.time)

    # ------------------------------------------------------------------
    # Exported state
    # ------------------------------------------------------------------
    def _gravity_forces(self) -> np.ndarray:
        out = np.zeros((self.n_domain, 3), dtype=float)
        out[: self.n_body, :] = self.body_mass * self.gravity[None, :]
        out[self.n_body, :] = self.n_body * self.body_mass * self.gravity
        return out

    def _deformation_gradient(self) -> np.ndarray:
        t = self.tets
        x = self.x
        ds = np.stack(
            (x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]], x[t[:, 3]] - x[t[:, 0]]),
            axis=2,
        )
        return np.einsum("nij,njk->nik", ds, self.dm_inv)

    def _make_frame(self, time: float) -> StateFrame:
        x = self.x
        v = self.v
        si = self.surf_idx

        delta = self.cum - self.cum_snapshot
        force = delta / max(self.sample_dt, 1e-12)
        if self._first_frame:
            force[:, 0, :] = self._gravity_forces()
            self._first_frame = False
        impulse = self.cum.sum(axis=1)
        self.cum_snapshot = self.cum.copy()

        volumes = np.bincount(
            self.tet_body, weights=_tet_volumes(x, self.tets), minlength=self.n_body
        )
        gradient = self._deformation_gradient().astype(np.float32)

        fields = {
            "element_deformation_gradient": gradient,
            "constraint_value": volumes.astype(np.float64),
            "constraint_target": self.v0_body.copy(),
            "budget_domain_weights": self.domain_weights,
            "budget_external_force": force.astype(np.float32),
            "budget_cumulative_external_impulse": impulse.astype(np.float32),
            "budget_cumulative_inflow_momentum": np.zeros((self.n_domain, 3), dtype=np.float32),
            "budget_cumulative_outflow_momentum": np.zeros((self.n_domain, 3), dtype=np.float32),
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
        }
        scalars = {
            "constraint_scale": float(np.mean(self.v0_body)),
            "reference_momentum": float(self.n_body * self.body_mass * 10.0),
        }
        frame = StateFrame(
            time=float(time),
            position=np.array(x[si], dtype=float),
            velocity=np.array(v[si], dtype=float),
            mass=self.surf_mass.copy(),
            body_id=self.body_ids.copy(),
            attributes={"mass_quantity": self.surf_mass.copy()},
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        out = {
            "solver": "xpbd_volumetric_solid",
            "bodies": int(self.n_body),
            "nodes": int(self.n_node),
            "elements": int(self.tets.shape[0]),
            "substeps": int(SUBSTEPS),
            "iterations": int(ITERATIONS),
            "time": float(self.time),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))) if self._ready else 0.0,
            "min_height": float(np.min(self.x[:, 2])) if self._ready else 0.0,
        }
        out.update(self._notes)
        return out
