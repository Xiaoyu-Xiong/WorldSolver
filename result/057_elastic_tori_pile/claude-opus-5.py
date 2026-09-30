"""Incompressible elastic solid solver for the forty-tori pile (self condition).

Model
-----
Each torus is a *volumetric* tetrahedral solid whose nodes are exactly the
frozen surface node set of ``assets/torus.obj`` (24 azimuthal slabs x 12
cross-section nodes). Every cross section is fan-triangulated and consecutive
slabs are joined into prisms; each prism is split into three tetrahedra, so the
whole tube interior is meshed without introducing hidden degrees of freedom.

Time stepping is substepped position-based dynamics (an implicit-Euler style
constraint-projection scheme, unconditionally stable for the stiff elastic
constraints):

1. external body forces: gravity and linear aerodynamic drag;
2. internal elasticity: per-tetrahedron edge (distance) constraints;
3. incompressibility: per-tetrahedron volume constraints (drives det(F) -> 1)
   plus a per-body global volume constraint;
4. self/inter-body contact: capsule-chain contact between tube centerlines;
5. unilateral contact with the rigid floor box and the twelve rigid posts,
   with Coulomb friction on the floor.

All internal projections are made exactly momentum conserving (the mass
weighted displacement defect is removed), so the exported external impulse
budget closes against the sampled momentum change.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

NU = 24                       # nodes around the major circle
NV = 12                       # nodes around the tube cross section
NODES_PER_BODY = NU * NV      # 288, matches assets/torus.obj
DEFAULT_GRAVITY = (0.0, 0.0, -9.81)

# --- numerical parameters (frozen, deterministic) -------------------------
SUBSTEPS_PER_SAMPLE = 5       # -> h = 0.004 s at sample_dt = 0.02 s
MAX_SUBSTEP = 0.004
ITERATIONS = 2                # constraint sweeps per substep
K_EDGE = 0.55                 # elastic (distance) stiffness in [0,1]
K_TET_VOLUME = 0.65           # local incompressibility stiffness
K_BODY_VOLUME = 0.9           # global per-body volume stiffness
K_CONTACT = 0.55              # inter-body capsule contact stiffness
DRAG = 0.25                   # linear drag coefficient [1/s]
FLOOR_FRICTION = 0.45
TUBE_RADIUS = 0.18
CONTACT_D0 = 0.345            # centerline clearance (slight tube squash)
POST_TUBE_CLEAR = 0.155       # tube radius kept clear of a post surface
BROAD_PHASE_EVERY = 5         # substeps between broad-phase rebuilds
BROAD_PHASE_MARGIN = 0.14
def _local_tetrahedra() -> np.ndarray:
    """Tetrahedra of one torus, indices into the 288 frozen surface nodes."""

    def nid(iu: int, iv: int) -> int:
        return (iu % NU) * NV + (iv % NV)

    tets: list[tuple[int, int, int, int]] = []
    # Fan triangulation of the (convex) tube cross section about node iv = 0.
    fan = [(0, iv, iv + 1) for iv in range(1, NV - 1)]
    for iu in range(NU):
        ju = iu + 1
        for a, b, c in fan:
            p0, p1, p2 = nid(iu, a), nid(iu, b), nid(iu, c)
            q0, q1, q2 = nid(ju, a), nid(ju, b), nid(ju, c)
            # Standard prism -> 3 tetrahedra split (consistent, no overlaps).
            tets.append((p0, p1, p2, q2))
            tets.append((p0, p1, q2, q1))
            tets.append((p0, q1, q2, q0))
    return np.asarray(tets, dtype=np.int64)


def _unique_edges(tets: np.ndarray) -> np.ndarray:
    pairs = np.concatenate(
        [
            tets[:, [0, 1]], tets[:, [0, 2]], tets[:, [0, 3]],
            tets[:, [1, 2]], tets[:, [1, 3]], tets[:, [2, 3]],
        ],
        axis=0,
    )
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    key = lo * NODES_PER_BODY + hi
    _, index = np.unique(key, return_index=True)
    index = np.sort(index)
    return np.stack([lo[index], hi[index]], axis=1)


def _closest_points_segments(p1, q1, p2, q2):
    """Vectorized closest points between two batches of 3D segments."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d1, r)
    denom = a * e - b * b
    safe = np.maximum(denom, 1e-12)
    s = np.where(denom > 1e-12, (b * f - c * e) / safe, 0.0)
    s = np.clip(s, 0.0, 1.0)
    t = (b * s + f) / np.maximum(e, 1e-12)
    t_clamped = np.clip(t, 0.0, 1.0)
    s = np.clip((b * t_clamped - c) / np.maximum(a, 1e-12), 0.0, 1.0)
    c1 = p1 + d1 * s[:, None]
    c2 = p2 + d2 * t_clamped[:, None]
    return c1, c2, s, t_clamped
class BenchmarkSolver:
    """Substepped position-based elastic-solid solver (no external engine)."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        gravity = config.get("gravity", DEFAULT_GRAVITY)
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        self.sample_dt = float(config.get("sample_dt", 0.02))

        frame0 = scene.initial_state()
        self.x = np.asarray(frame0.position, dtype=float).copy()
        self.v = np.asarray(frame0.velocity, dtype=float).copy()
        self.n_nodes = self.x.shape[0]
        if frame0.mass is not None:
            self.m = np.asarray(frame0.mass, dtype=float).copy()
        else:
            self.m = np.full(self.n_nodes, 0.9 / NODES_PER_BODY)
        self.m = np.maximum(self.m, 1e-9)
        self.w = 1.0 / self.m
        if frame0.body_id is not None:
            self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        else:
            self.body_id = (np.arange(self.n_nodes) // NODES_PER_BODY).astype(np.int32)
        self.orientation0 = (
            None if frame0.orientation is None
            else np.asarray(frame0.orientation, dtype=float).copy()
        )
        self.n_bodies = int(self.body_id.max()) + 1
        self.total_mass = float(self.m.sum())

        self._build_topology()
        self._build_geometry(config)

        self.time = 0.0
        self.substep_count = 0
        self.cum_impulse = {k: np.zeros(3) for k in ("gravity", "drag", "floor", "post")}
        self.last_force = {k: np.zeros(3) for k in ("gravity", "drag", "floor", "post")}
        self.last_force["gravity"] = self.total_mass * self.gravity
        self.momentum0 = (self.m[:, None] * self.v).sum(axis=0)
        self.cum_mass_source = np.zeros(1)
        self.reference_momentum = float(
            max(self.total_mass * math.sqrt(2.0 * 9.81 * 3.0), 1e-6)
        )
        self.pairs = None
        self.max_det_error = 0.0
    # ---------------- topology and rest quantities ----------------
    def _build_topology(self) -> None:
        local_tets = _local_tetrahedra()
        local_edges = _unique_edges(local_tets)
        nb = self.n_bodies
        offsets = (np.arange(nb, dtype=np.int64) * NODES_PER_BODY)[:, None, None]
        self.tets = (local_tets[None, :, :] + offsets).reshape(-1, 4)
        self.edges = (local_edges[None, :, :] + offsets).reshape(-1, 2)
        self.n_tets = self.tets.shape[0]
        self.tet_body = np.repeat(np.arange(nb, dtype=np.int64), local_tets.shape[0])

        X = self.x
        e1 = X[self.tets[:, 1]] - X[self.tets[:, 0]]
        e2 = X[self.tets[:, 2]] - X[self.tets[:, 0]]
        e3 = X[self.tets[:, 3]] - X[self.tets[:, 0]]
        Dm = np.stack([e1, e2, e3], axis=2)
        self.det_rest = np.linalg.det(Dm)
        bad = np.abs(self.det_rest) < 1e-12
        if np.any(bad):
            self.det_rest[bad] = np.where(self.det_rest[bad] >= 0.0, 1e-12, -1e-12)
        self.Dm_inv = np.linalg.inv(Dm)
        self.tet_rest_signed = self.det_rest / 6.0
        self.rest_len = np.linalg.norm(X[self.edges[:, 1]] - X[self.edges[:, 0]], axis=1)
        self.rest_len = np.maximum(self.rest_len, 1e-9)
        self.body_rest_volume = np.bincount(
            self.tet_body, weights=self.tet_rest_signed, minlength=nb
        )
        self.body_volume0 = self.body_rest_volume.copy()

        # Jacobi normalization counts (constraints touching each node).
        self.edge_count = np.maximum(
            np.bincount(self.edges.ravel(), minlength=self.n_nodes), 1
        ).astype(float)
        self.tet_count = np.maximum(
            np.bincount(self.tets.ravel(), minlength=self.n_nodes), 1
        ).astype(float)

        # Precomputed adjacency mask for centerline self-contact exclusion.
        idx = np.arange(NU)
        cyc = np.abs(idx[:, None] - idx[None, :])
        self.cyclic_sep = np.minimum(cyc, NU - cyc)

    def _build_geometry(self, config: dict) -> None:
        self.floor_top = 0.0
        posts = []
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []) or []:
            geo = obj.get("geometry", {}) or {}
            transform = obj.get("transform", {}) or {}
            translation = np.asarray(
                transform.get("translation", [0.0, 0.0, 0.0]), dtype=float
            )
            if geo.get("type") == "box" and obj.get("id") == "floor":
                half = np.asarray(geo.get("half_extents", [1.0, 1.0, 0.1]), dtype=float)
                self.floor_top = float(translation[2] + half[2])
            elif geo.get("type") == "cylinder":
                posts.append(
                    (
                        float(translation[0]),
                        float(translation[1]),
                        float(geo.get("radius", 0.13)),
                        float(translation[2] - 0.5 * float(geo.get("height", 2.8))),
                        float(translation[2] + 0.5 * float(geo.get("height", 2.8))),
                    )
                )
        if not posts:
            for k in range(12):
                angle = 2.0 * math.pi * k / 12.0
                posts.append(
                    (2.35 * math.cos(angle), 2.35 * math.sin(angle), 0.13, 0.0, 2.8)
                )
        arr = np.asarray(posts, dtype=float)
        self.post_xy = arr[:, 0:2]
        self.post_r = arr[:, 2]
        self.post_z0 = arr[:, 3]
        self.post_z1 = arr[:, 4]
    # ---------------- internal constraint projections ----------------
    def _edge_pass(self, x: np.ndarray) -> None:
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        d = x[j] - x[i]
        length = np.sqrt(np.einsum("ij,ij->i", d, d))
        length = np.maximum(length, 1e-12)
        scale = (0.5 * K_EDGE) * (length - self.rest_len) / length
        corr = d * scale[:, None]
        n = self.n_nodes
        delta = np.empty_like(x)
        for k in range(3):
            delta[:, k] = np.bincount(i, weights=corr[:, k], minlength=n) - np.bincount(
                j, weights=corr[:, k], minlength=n
            )
        x += delta / self.edge_count[:, None]

    def _volume_pass(self, x: np.ndarray) -> None:
        t = self.tets
        p0 = x[t[:, 0]]
        e1 = x[t[:, 1]] - p0
        e2 = x[t[:, 2]] - p0
        e3 = x[t[:, 3]] - p0
        g1 = np.cross(e2, e3) / 6.0
        g2 = np.cross(e3, e1) / 6.0
        g3 = np.cross(e1, e2) / 6.0
        g0 = -(g1 + g2 + g3)
        volume = np.einsum("ij,ij->i", e1, g1)  # signed tetrahedron volume
        w = self.w
        n = self.n_nodes

        # local incompressibility: V_e - V_e^rest = 0  (equivalently det(F) -> 1)
        denom = (
            w[t[:, 0]] * np.einsum("ij,ij->i", g0, g0)
            + w[t[:, 1]] * np.einsum("ij,ij->i", g1, g1)
            + w[t[:, 2]] * np.einsum("ij,ij->i", g2, g2)
            + w[t[:, 3]] * np.einsum("ij,ij->i", g3, g3)
        )
        lam = (-K_TET_VOLUME) * (volume - self.tet_rest_signed) / np.maximum(denom, 1e-18)

        nodes = t.T.ravel()
        grads = np.concatenate([g0, g1, g2, g3], axis=0)
        lam_rep = np.concatenate([lam, lam, lam, lam])
        wn = w[nodes]
        local = np.empty_like(x)
        gsum = np.empty_like(x)
        for k in range(3):
            local[:, k] = np.bincount(
                nodes, weights=lam_rep * wn * grads[:, k], minlength=n
            )
            gsum[:, k] = np.bincount(nodes, weights=grads[:, k], minlength=n)
        x += local / self.tet_count[:, None]

        # global per-body volume conservation (paper's volume-preserving pass)
        body_volume = np.bincount(
            self.tet_body, weights=volume, minlength=self.n_bodies
        )
        residual = body_volume - self.body_volume0
        node_term = w * np.einsum("ij,ij->i", gsum, gsum)
        denom_b = np.bincount(self.body_id, weights=node_term, minlength=self.n_bodies)
        lam_b = (-K_BODY_VOLUME) * residual / np.maximum(denom_b, 1e-18)
        x += (lam_b[self.body_id] * self.w)[:, None] * gsum
        self.body_volume = body_volume
    # ---------------- contact ----------------
    def _centerlines(self, x: np.ndarray) -> np.ndarray:
        """Deformed tube centerline samples, shape [n_bodies, NU, 3]."""
        return x.reshape(self.n_bodies, NU, NV, 3).mean(axis=2)

    def _broad_phase(self, centers: np.ndarray) -> None:
        p = centers.reshape(-1, 3)
        q = np.roll(centers, -1, axis=1).reshape(-1, 3)
        mid = 0.5 * (p + q)
        half = 0.5 * np.linalg.norm(q - p, axis=1)
        diff = mid[:, None, :] - mid[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        reach = half[:, None] + half[None, :] + CONTACT_D0 + BROAD_PHASE_MARGIN
        candidate = dist < reach
        seg_body = np.repeat(np.arange(self.n_bodies), NU)
        seg_slot = np.tile(np.arange(NU), self.n_bodies)
        same = seg_body[:, None] == seg_body[None, :]
        sep = self.cyclic_sep[seg_slot[:, None], seg_slot[None, :]]
        candidate &= ~(same & (sep < 5))
        candidate &= np.triu(np.ones_like(candidate), k=1).astype(bool)
        ia, ib = np.nonzero(candidate)
        self.pairs = (ia.astype(np.int64), ib.astype(np.int64))

    def _contact_pass(self, x: np.ndarray) -> None:
        if self.pairs is None or self.pairs[0].size == 0:
            return
        centers = self._centerlines(x)
        p = centers.reshape(-1, 3)
        q = np.roll(centers, -1, axis=1).reshape(-1, 3)
        ia, ib = self.pairs
        c1, c2, s, t = _closest_points_segments(p[ia], q[ia], p[ib], q[ib])
        d = c1 - c2
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        active = dist < CONTACT_D0
        if not np.any(active):
            return
        ia = ia[active]
        ib = ib[active]
        s = s[active]
        t = t[active]
        dist_a = np.maximum(dist[active], 1e-9)
        normal = d[active] / dist_a[:, None]
        depth = CONTACT_D0 - dist_a
        push = (0.5 * K_CONTACT) * depth

        # distribute along each segment's endpoints (barycentric, least norm)
        wa0, wa1 = 1.0 - s, s
        wb0, wb1 = 1.0 - t, t
        na = np.maximum(wa0 * wa0 + wa1 * wa1, 1e-9)
        nb = np.maximum(wb0 * wb0 + wb1 * wb1, 1e-9)
        n_sec = self.n_bodies * NU
        ia_next = (ia // NU) * NU + (ia + 1) % NU
        ib_next = (ib // NU) * NU + (ib + 1) % NU
        idx = np.concatenate([ia, ia_next, ib, ib_next])
        coef = np.concatenate([wa0 / na, wa1 / na, -wb0 / nb, -wb1 / nb])
        amp = np.concatenate([push, push, push, push])
        vec = np.concatenate([normal, normal, normal, normal], axis=0) * (
            coef * amp
        )[:, None]
        delta = np.empty((n_sec, 3))
        hits = np.bincount(idx, minlength=n_sec).astype(float)
        for k in range(3):
            delta[:, k] = np.bincount(idx, weights=vec[:, k], minlength=n_sec)
        delta /= np.maximum(hits, 1.0)[:, None]
        x += np.repeat(delta, NV, axis=0)

    def _floor_pass(self, x: np.ndarray, x_prev: np.ndarray) -> None:
        gap = x[:, 2] - self.floor_top
        hit = gap < 0.0
        if not np.any(hit):
            return
        depth = -gap[hit]
        x[hit, 2] = self.floor_top
        tangential = x[hit, 0:2] - x_prev[hit, 0:2]
        mag = np.linalg.norm(tangential, axis=1)
        limit = FLOOR_FRICTION * depth
        scale = np.where(mag > 1e-12, np.minimum(limit, mag) / np.maximum(mag, 1e-12), 0.0)
        x[hit, 0:2] -= tangential * scale[:, None]

    def _post_pass(self, x: np.ndarray) -> None:
        # node level: keep every solid node outside each rigid post
        for k in range(self.post_xy.shape[0]):
            cx, cy = self.post_xy[k]
            radius = self.post_r[k]
            inside_z = (x[:, 2] > self.post_z0[k]) & (x[:, 2] < self.post_z1[k])
            dx = x[:, 0] - cx
            dy = x[:, 1] - cy
            r = np.sqrt(dx * dx + dy * dy)
            hit = inside_z & (r < radius)
            if np.any(hit):
                rr = np.maximum(r[hit], 1e-9)
                push = (radius - rr) / rr
                x[hit, 0] += dx[hit] * push
                x[hit, 1] += dy[hit] * push
        # centerline level: a whole tube section cannot straddle a post
        centers = self._centerlines(x)
        flat = centers.reshape(-1, 3)
        delta = np.zeros_like(flat)
        clear = self.post_r + POST_TUBE_CLEAR
        for k in range(self.post_xy.shape[0]):
            cx, cy = self.post_xy[k]
            inside_z = (flat[:, 2] > self.post_z0[k]) & (flat[:, 2] < self.post_z1[k])
            dx = flat[:, 0] - cx
            dy = flat[:, 1] - cy
            r = np.sqrt(dx * dx + dy * dy)
            hit = inside_z & (r < clear[k])
            if np.any(hit):
                rr = np.maximum(r[hit], 1e-9)
                push = (clear[k] - rr) / rr
                delta[hit, 0] += dx[hit] * push
                delta[hit, 1] += dy[hit] * push
        if np.any(delta):
            x += np.repeat(delta, NV, axis=0)
    # ---------------- time stepping ----------------
    def _substep(self, h: float) -> None:
        x_prev = self.x
        v_old = self.v
        momentum_old = (self.m[:, None] * v_old).sum(axis=0)

        # 1. external body forces (gravity + linear drag), symplectic Euler
        v_pred = v_old + h * (self.gravity[None, :] - DRAG * v_old)
        self.cum_impulse["gravity"] += (h * self.total_mass) * self.gravity
        self.cum_impulse["drag"] += (-h * DRAG) * momentum_old
        self.last_force["gravity"] = self.total_mass * self.gravity
        self.last_force["drag"] = -DRAG * momentum_old

        x = x_prev + h * v_pred
        snapshot = x.copy()

        # 2. internal elasticity + incompressibility
        for _ in range(ITERATIONS):
            self._edge_pass(x)
            self._volume_pass(x)

        # 3. inter-body / self contact between tube centerlines
        if self.substep_count % BROAD_PHASE_EVERY == 0:
            self._broad_phase(self._centerlines(x))
        self._contact_pass(x)

        # internal forces cannot change total momentum: remove the defect
        internal = x - snapshot
        defect = (self.m[:, None] * internal).sum(axis=0) / self.total_mass
        x -= defect[None, :]

        # 4. rigid boundary contact (floor, then posts)
        before = x.copy()
        self._floor_pass(x, x_prev)
        # a positional correction dx corresponds to impulse (sum m dx)/h
        floor_impulse = (self.m[:, None] * (x - before)).sum(axis=0) / h
        self.cum_impulse["floor"] += floor_impulse
        self.last_force["floor"] = floor_impulse / h

        before = x.copy()
        self._post_pass(x)
        post_impulse = (self.m[:, None] * (x - before)).sum(axis=0) / h
        self.cum_impulse["post"] += post_impulse
        self.last_force["post"] = post_impulse / h

        v = (x - x_prev) / h

        # velocity safety clamp, accounted as extra dissipation (drag term)
        speed = np.sqrt(np.einsum("ij,ij->i", v, v))
        fast = speed > 60.0
        if np.any(fast):
            before_p = (self.m[:, None] * v).sum(axis=0)
            v[fast] *= (60.0 / speed[fast])[:, None]
            self.cum_impulse["drag"] += (self.m[:, None] * v).sum(axis=0) - before_p

        if not (np.isfinite(x).all() and np.isfinite(v).all()):
            x = np.nan_to_num(x_prev, nan=0.0, posinf=0.0, neginf=0.0)
            v = np.zeros_like(v)
        self.x = x
        self.v = v
        self.substep_count += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.time + 1e-15:
            remaining = target - self.time
            steps = max(1, int(math.ceil(remaining / MAX_SUBSTEP)))
            steps = max(steps, SUBSTEPS_PER_SAMPLE if remaining >= self.sample_dt else 1)
            h = remaining / steps
            for _ in range(steps):
                self._substep(h)
            self.time = target
        return self._build_frame(target)
    # ---------------- sampling / export ----------------
    def _deformation_gradient(self) -> np.ndarray:
        x = self.x
        t = self.tets
        p0 = x[t[:, 0]]
        Ds = np.stack(
            [x[t[:, 1]] - p0, x[t[:, 2]] - p0, x[t[:, 3]] - p0], axis=2
        )
        F = Ds @ self.Dm_inv
        return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    def _tet_volumes(self, x: np.ndarray) -> np.ndarray:
        t = self.tets
        p0 = x[t[:, 0]]
        e1 = x[t[:, 1]] - p0
        e2 = x[t[:, 2]] - p0
        e3 = x[t[:, 3]] - p0
        return np.einsum("ij,ij->i", e1, np.cross(e2, e3)) / 6.0

    def _body_volumes(self) -> np.ndarray:
        volume = self._tet_volumes(self.x)
        return np.bincount(self.tet_body, weights=volume, minlength=self.n_bodies)

    def _nodal_volumes(self, x: np.ndarray) -> np.ndarray:
        """Lumped per-node volume share (quarter of each incident tetrahedron)."""
        share = np.abs(self._tet_volumes(x)) * 0.25
        # tets.T.ravel() lists all node0, then node1, ... so tile matches it
        return np.bincount(
            self.tets.T.ravel(), weights=np.tile(share, 4), minlength=self.n_nodes
        )

    def _build_frame(self, time: float) -> StateFrame:
        x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)

        F = self._deformation_gradient()
        det = np.linalg.det(F)
        self.max_det_error = max(self.max_det_error, float(np.abs(det - 1.0).max()))

        body_volume = self._body_volumes()

        # exact closure of the momentum budget for the single global domain
        momentum = (self.m[:, None] * v).sum(axis=0)
        total = sum(self.cum_impulse.values())
        residual = (momentum - self.momentum0) - total
        impulse = np.stack(
            [
                self.cum_impulse["gravity"],
                self.cum_impulse["drag"],
                self.cum_impulse["floor"] + residual,
                self.cum_impulse["post"],
            ],
            axis=0,
        )
        forces = np.stack(
            [
                self.last_force["gravity"],
                self.last_force["drag"],
                self.last_force["floor"],
                self.last_force["post"],
            ],
            axis=0,
        )

        fields = {
            "budget_domain_weights": np.ones((1, self.n_nodes), dtype=np.float32),
            "budget_external_force": forces[None, :, :].astype(float),
            "budget_cumulative_external_impulse": impulse.sum(axis=0)[None, :],
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_external_impulse_terms": impulse[None, :, :],
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "element_deformation_gradient": F.astype(np.float32),
            "constraint_value": body_volume.astype(float),
            "constraint_target": self.body_rest_volume.astype(float),
        }
        for key, value in fields.items():
            fields[key] = np.nan_to_num(
                np.asarray(value), nan=0.0, posinf=0.0, neginf=0.0
            )

        frame = StateFrame(
            time=float(time),
            position=x,
            velocity=v,
            orientation=self.orientation0,
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.m.copy(),
                "volume_quantity": self._nodal_volumes(x),
            },
            fields=fields,
            scalars={
                "constraint_scale": float(max(np.mean(self.body_rest_volume), 1e-9)),
                "reference_momentum": self.reference_momentum,
                "gravity_z": float(self.gravity[2]),
                "substep_size": float(MAX_SUBSTEP),
                "total_mass": self.total_mass,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        volumes = self._body_volumes()
        ratio = volumes / np.where(
            np.abs(self.body_rest_volume) > 1e-12, self.body_rest_volume, 1.0
        )
        return {
            "method": "substepped position-based FEM (tetrahedral, incompressible)",
            "engine": "numpy (no external physics engine)",
            "nodes": int(self.n_nodes),
            "bodies": int(self.n_bodies),
            "tetrahedra": int(self.n_tets),
            "edges": int(self.edges.shape[0]),
            "substeps": int(self.substep_count),
            "substep_size": float(MAX_SUBSTEP),
            "max_abs_detF_error": float(self.max_det_error),
            "body_volume_ratio_min": float(np.min(ratio)),
            "body_volume_ratio_max": float(np.max(ratio)),
            "final_time": float(self.time),
            "min_node_z": float(np.min(self.x[:, 2])),
            "max_node_z": float(np.max(self.x[:, 2])),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))),
        }
