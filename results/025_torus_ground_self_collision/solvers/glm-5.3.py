"""Physics solver: deformable torus falling on ground with self-collision.

Physical model
--------------
The subject is a triangulated torus surface (64 minor rings x 24 vertices,
tube axis along y).  It is simulated as an elastic network of mass points
(one per mesh vertex, total mass from config) connected by:

* structural springs on every mesh edge          (stretch stiffness)
* shear springs on the two diagonals of each grid quad
* bending springs between 2-hop grid neighbours  (curvature resistance)
* cross-tube "diameter" springs joining opposite vertices of every minor
  ring, which carry the tube's bulk (through-thickness) stiffness so the
  tube squashes under load instead of behaving like cloth.

Time integration is semi-implicit (symplectic) Euler with a fixed substep,
Rayleigh mass damping plus spring damping along each spring axis.

Ground contact is a constraint (projective) contact on the box top surface
(z = ground_z) with Coulomb friction and restitution, matching the paper
configuration "constraint ground collision".  Self-collision is a penalty
repulsion between non-neighbouring vertices closer than the frozen contact
length scale, matching "penalty self-collision".

Per-vertex deformation gradients are recovered by a weighted least-squares
fit of the 1-ring (plus the cross-tube diameter direction, which supplies
the missing normal rank) against rest offsets.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

try:  # scipy is available in the runtime image
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - fallback
    cKDTree = None


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _pack_pairs(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Pack unordered index pairs into unique int64 keys (lo * n + hi)."""
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return np.unique(lo.astype(np.int64) * n + hi.astype(np.int64))


def _pair_keys(pairs: np.ndarray, n: int) -> np.ndarray:
    lo = pairs.min(axis=1)
    hi = pairs.max(axis=1)
    return (lo.astype(np.int64) * n + hi.astype(np.int64))


class BenchmarkSolver:
    """Spring-network elastic-solid solver for the torus scene."""

    SUBSTEP = 4.0e-4          # integration substep [s]
    MAX_SPEED = 40.0          # safety clamp [m/s]
    SELF_COLLISION_QUERY_R = 0.12  # kd-tree query radius [m]

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.contact_scale = float(cfg["contact_length_scale"])
        self.restitution = float(cfg.get("restitution", 0.2))
        self.friction = float(cfg.get("friction", 0.58))
        self.rayleigh = float(cfg.get("damping", 1.2))

        init = scene.initial_state()
        self.x = np.array(init.position, dtype=float).copy()
        self.v = np.array(init.velocity, dtype=float).copy()
        self.n = len(self.x)
        self.rest = self.x.copy()
        self.mass = np.array(init.mass, dtype=float).copy()
        self.time = 0.0

        # ---- mesh + asset (faces are needed for topology) ----------------
        root = Path(__file__).resolve().parents[1]
        with np.load(root / cfg["subject_asset"]) as data:
            faces = np.asarray(data["faces"], dtype=np.int64)
        self.faces = faces

        stiffness = float(cfg.get("stiffness", 13.0))
        m_node = float(self.mass.mean())

        k_struct = 12.0 * stiffness     # structural (mesh edges)
        k_shear = 6.0 * stiffness       # quad diagonals
        k_bend = 2.5 * stiffness        # 2-hop grid neighbours
        k_diam = 0.5 * stiffness        # cross-tube diameters (bulk squash)
        k_self = 24.0 * stiffness       # self-collision penalty
        self.k_self = k_self
        self.c_self = 0.35 * np.sqrt(k_self * m_node)

        # ---- assemble springs --------------------------------------------
        pairs: list[tuple[np.ndarray, np.ndarray, float, float]] = []

        edges = set()
        for f in faces:
            for a, b in itertools.combinations(sorted(f.tolist()), 2):
                edges.add((a, b))
        edges = np.array(sorted(edges), dtype=np.int64)
        c_struct = 0.10 * np.sqrt(k_struct * m_node)
        pairs.append((edges[:, 0], edges[:, 1], k_struct, c_struct))

        grid = self._detect_grid()          # (n_rings, n_minor) vertex ids
        if grid is not None:
            nr, nm = grid.shape
            ii = np.arange(nr)[:, None]
            jj = np.arange(nm)[None, :]
            nxt_i = (ii + 1) % nr
            nxt_j = (jj + 1) % nm
            # shear diagonals of each quad
            d1a, d1b = grid[ii, jj], grid[nxt_i, nxt_j]
            d2a, d2b = grid[nxt_i, jj], grid[ii, nxt_j]
            diag = np.concatenate([
                np.stack([d1a.ravel(), d1b.ravel()], axis=1),
                np.stack([d2a.ravel(), d2b.ravel()], axis=1),
            ])
            c_shear = 0.10 * np.sqrt(k_shear * m_node)
            pairs.append((diag[:, 0], diag[:, 1], k_shear, c_shear))
            # bending: 2-hop along the tube and around the tube
            hop_j = (jj + 2) % nm
            hop_i = (nxt_i + 1) % nr
            bend = np.concatenate([
                np.stack([grid[ii, jj].ravel(), grid[ii, hop_j].ravel()], axis=1),
                np.stack([grid[ii, jj].ravel(), grid[hop_i, jj].ravel()], axis=1),
            ])
            c_bend = 0.10 * np.sqrt(k_bend * m_node)
            pairs.append((bend[:, 0], bend[:, 1], k_bend, c_bend))
            # cross-tube diameters (opposite points of the minor ring)
            if nm % 2 == 0:
                diam = np.stack([grid.ravel(), grid[:, (jj + nm // 2) % nm].ravel()], axis=1)
                c_diam = 0.10 * np.sqrt(k_diam * m_node)
                pairs.append((diam[:, 0], diam[:, 1], k_diam, c_diam))
            self.grid = grid
        else:
            self.grid = None

        # flatten spring arrays
        si, sk, sc = [], [], []
        for a, b, k, c in pairs:
            si.append(np.stack([a, b], axis=1))
            sk.append(np.full(len(a), k, dtype=float))
            sc.append(np.full(len(a), c, dtype=float))
        self.springs = np.concatenate(si, axis=0)
        self.k_spring = np.concatenate(sk)
        self.c_spring = np.concatenate(sc)
        self.rest_len = np.linalg.norm(
            self.rest[self.springs[:, 0]] - self.rest[self.springs[:, 1]], axis=1
        )
        # drop degenerate (zero-length) springs
        good = self.rest_len > 1e-9
        self.springs = self.springs[good]
        self.k_spring = self.k_spring[good]
        self.c_spring = self.c_spring[good]
        self.rest_len = self.rest_len[good]

        # ---- self-collision exclusion: graph distance <= 2 ---------------
        adj: list[set] = [set() for _ in range(self.n)]
        for a, b in edges:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
        ex = set()
        for v, nbrs in enumerate(adj):
            for u in nbrs:
                ex.add((min(v, u), max(v, u)))
            for u1, u2 in itertools.combinations(sorted(nbrs), 2):
                ex.add((min(u1, u2), max(u1, u2)))
        self.excluded_keys = np.unique(
            np.array([a * self.n + b for a, b in ex], dtype=np.int64)
        ) if ex else np.zeros(0, dtype=np.int64)

        # ---- per-vertex deformation-gradient fit data --------------------
        diag_pairs = self._diameter_pairs()
        fit_a = [edges[:, 0], edges[:, 1]]
        fit_b = [edges[:, 1], edges[:, 0]]
        fit_w = [np.ones(len(edges)), np.ones(len(edges))]
        if diag_pairs is not None:
            w = np.full(len(diag_pairs), 0.5)
            fit_a += [diag_pairs[:, 0], diag_pairs[:, 1]]
            fit_b += [diag_pairs[:, 1], diag_pairs[:, 0]]
            fit_w += [w, w]
        self.fit_i = np.concatenate(fit_a)
        self.fit_j = np.concatenate(fit_b)
        self.fit_w = np.concatenate(fit_w)
        self.fit_rest = self.rest[self.fit_j] - self.rest[self.fit_i]

        self.external_force = self.mass[:, None] * self.gravity[None, :]
        self.initial_frame_emitted = False

    # ------------------------------------------------------------------
    def _detect_grid(self):
        """Detect the (rings x minor) structured grid of the torus mesh."""
        V = self.rest
        zc = 0.5 * (V[:, 2].max() + V[:, 2].min())
        psi = np.arctan2(V[:, 2] - zc, V[:, 0])
        order = np.argsort(psi)
        ps = psi[order]
        splits = np.where(np.diff(ps) > 1e-6)[0]
        rings = [r for r in np.split(order, splits + 1)]
        # merge the cluster wrapped across the +-pi branch cut
        if len(rings) > 1:
            wrap_gap = ps[0] + 2.0 * np.pi - ps[-1]
            inner_gaps = np.diff(ps)
            if inner_gaps.size and wrap_gap < inner_gaps.max():
                rings[0] = np.concatenate([rings[-1], rings[0]])
                rings = rings[:-1]
        if len(rings) < 8:
            return None
        sizes = {len(r) for r in rings}
        if len(sizes) != 1:
            return None
        nm = sizes.pop()
        if nm % 2 != 0 or nm < 6:
            return None
        grid = np.empty((len(rings), nm), dtype=np.int64)
        yax = np.array([0.0, 1.0, 0.0])
        for i, ring in enumerate(rings):
            P = V[ring]
            ctr = P.mean(axis=0)
            rad = np.array([ctr[0], 0.0, ctr[2] - zc])
            rad /= max(np.linalg.norm(rad), 1e-12)
            loc = P - ctr
            phi = np.arctan2(loc @ yax, loc @ rad)
            grid[i] = ring[np.argsort(phi)]
        return grid

    def _diameter_pairs(self):
        if self.grid is None:
            return None
        nr, nm = self.grid.shape
        jj = np.arange(nm)[None, :]
        return np.stack(
            [self.grid.ravel(), self.grid[:, (jj + nm // 2) % nm].ravel()], axis=1
        )

    # ------------------------------------------------------------------
    def _spring_forces(self, f: np.ndarray) -> None:
        s = self.springs
        i, j = s[:, 0], s[:, 1]
        dx = self.x[j] - self.x[i]
        d = np.linalg.norm(dx, axis=1)
        d = np.maximum(d, 1e-12)
        n = dx / d[:, None]
        rel = self.v[j] - self.v[i]
        vn = np.sum(rel * n, axis=1)
        mag = self.k_spring * (d - self.rest_len) + self.c_spring * vn
        F = mag[:, None] * n
        np.add.at(f, i, F)
        np.add.at(f, j, -F)

    def _self_collision(self, f: np.ndarray):
        """Penalty self-collision between non-neighbouring vertices."""
        r0 = self.contact_scale
        if cKDTree is None:
            return np.zeros((0, 2), dtype=np.int64), np.zeros(0)
        tree = cKDTree(self.x)
        try:
            pairs = tree.query_pairs(r=self.SELF_COLLISION_QUERY_R, output_type="ndarray")
        except TypeError:  # older scipy
            pairs = np.array(sorted(tree.query_pairs(r=self.SELF_COLLISION_QUERY_R)),
                             dtype=np.int64).reshape(-1, 2)
        if len(pairs) == 0:
            return pairs, np.zeros(0)
        keys = _pair_keys(pairs, self.n)
        idx = np.searchsorted(self.excluded_keys, keys)
        idx = np.clip(idx, 0, max(len(self.excluded_keys) - 1, 0))
        if len(self.excluded_keys):
            keep = self.excluded_keys[idx] != keys
        else:
            keep = np.ones(len(keys), dtype=bool)
        pairs = pairs[keep]
        if len(pairs) == 0:
            return pairs, np.zeros(0)
        i, j = pairs[:, 0], pairs[:, 1]
        dx = self.x[j] - self.x[i]
        d = np.maximum(np.linalg.norm(dx, axis=1), 1e-12)
        n = dx / d[:, None]
        rel = self.v[j] - self.v[i]
        vn = np.sum(rel * n, axis=1)
        pen = r0 - d
        active = pen > 0.0
        if not np.any(active):
            return pairs, d - r0
        mag = np.where(active, self.k_self * pen - self.c_self * vn, 0.0)
        F = mag[:, None] * n
        np.add.at(f, i[active], -F[active])
        np.add.at(f, j[active], F[active])
        return pairs, d - r0

    def _ground_contact(self) -> None:
        """Constraint contact with the ground box top (z = ground_z)."""
        pen = self.ground_z - self.x[:, 2]
        hit = pen > 0.0
        if not np.any(hit):
            return
        idx = np.where(hit)[0]
        self.x[idx, 2] = self.ground_z
        vz = self.v[idx, 2]
        vn_new = np.where(vz < 0.0, -self.restitution * vz, vz)
        jn = self.mass[idx] * (vn_new - vz)      # normal impulse (>= 0)
        # Coulomb friction on the tangential velocity
        vt = self.v[idx, :2].copy()
        vt_mag = np.linalg.norm(vt, axis=1)
        needed = self.mass[idx] * vt_mag
        jmax = self.friction * jn
        scale = np.where(vt_mag > 1e-12, np.minimum(1.0, jmax / np.maximum(needed, 1e-15)), 0.0)
        self.v[idx, :2] = vt * (1.0 - scale)[:, None]
        self.v[idx, 2] = vn_new

    def _step(self, h: float) -> None:
        f = np.broadcast_to(self.external_force, self.x.shape).copy()
        self._spring_forces(f)
        self._self_collision(f)
        acc = f / self.mass[:, None]
        self.v += h * acc
        self.v -= h * self.rayleigh * self.v
        # safety clamp keeps states finite under extreme impacts
        speed = np.linalg.norm(self.v, axis=1)
        fast = speed > self.MAX_SPEED
        if np.any(fast):
            self.v[fast] *= (self.MAX_SPEED / speed[fast])[:, None]
        self.x += h * self.v
        self._ground_contact()

    # ------------------------------------------------------------------
    def _deformation_gradients(self) -> np.ndarray:
        """Least-squares per-vertex F from 1-ring + cross-tube offsets."""
        x = self.x
        p = self.fit_rest
        q = x[self.fit_j] - x[self.fit_i]
        w = self.fit_w
        A = np.zeros((self.n, 3, 3))
        Bm = np.zeros((self.n, 3, 3))
        wp = w[:, None, None] * p[:, :, None] * p[:, None, :]   # (M,3,3)
        wq = w[:, None, None] * q[:, :, None] * p[:, None, :]
        np.add.at(A, self.fit_i, wp)
        np.add.at(A, self.fit_j, wp)
        np.add.at(Bm, self.fit_i, wq)
        np.add.at(Bm, self.fit_j, wq)
        trace = np.trace(A, axis1=1, axis2=2)
        reg = 1e-3 * np.maximum(trace / 3.0, 1e-9)
        A += reg[:, None, None] * np.eye(3)[None, :, :]
        F = np.linalg.solve(A.transpose(0, 2, 1), Bm.transpose(0, 2, 1)).transpose(0, 2, 1)
        return F

    def _contact_distances(self):
        """Signed gaps for ground contacts and self-collision pairs."""
        l = self.contact_scale
        gaps = []
        gz = self.x[:, 2] - self.ground_z
        near_ground = np.where(gz < 3.0 * l)[0]
        if len(near_ground):
            gaps.append(gz[near_ground])
        if cKDTree is not None:
            tree = cKDTree(self.x)
            try:
                prs = tree.query_pairs(r=3.0 * l, output_type="ndarray")
            except TypeError:
                prs = np.array(sorted(tree.query_pairs(r=3.0 * l)), dtype=np.int64).reshape(-1, 2)
            if len(prs):
                keys = _pair_keys(prs, self.n)
                idx = np.searchsorted(self.excluded_keys, keys)
                idx = np.clip(idx, 0, max(len(self.excluded_keys) - 1, 0))
                keep = (self.excluded_keys[idx] != keys) if len(self.excluded_keys) else np.ones(len(keys), bool)
                prs = prs[keep]
                if len(prs):
                    d = np.linalg.norm(self.x[prs[:, 0]] - self.x[prs[:, 1]], axis=1)
                    gaps.append(d - l)
        if gaps:
            return np.concatenate(gaps)
        # always report at least the closest approach so the field is non-empty
        return np.array([gz.min()])

    def _make_frame(self, t: float):
        from worldsolver.contracts import StateFrame

        n = self.n
        return StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradients(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.external_force.copy(),
                "contact_signed_distance": np.asarray(self._contact_distances(), dtype=float),
            },
            scalars={"contact_length_scale": self.contact_scale},
        )

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        eps = 1e-9
        while self.time < target - eps:
            h = min(self.SUBSTEP, target - self.time)
            self._step(h)
            self.time = self.time + h
        self.time = max(self.time, target)
        return self._make_frame(target)

    def diagnostics(self) -> dict:
        return {
            "time": float(self.time),
            "n_vertices": int(self.n),
            "n_springs": int(len(self.springs)),
            "min_vertex_z": float(self.x[:, 2].min()),
            "max_speed": float(np.linalg.norm(self.v, axis=1).max()),
        }
