"""Cloth through a narrow funnel - self-contained thin-shell solver.

Physical model
--------------
* The cloth sheet is a constrained mass lattice: stretch (structural edges),
  shear (diagonals) and discrete bending (two-span) constraints with physical
  compliances (inverse stiffnesses).
* Integration is substepped XPBD, an implicit-Euler-type variational scheme:
  symplectic velocity prediction under gravity plus linear air drag,
  Gauss-Seidel constraint projection over graph-coloured groups, contact
  projection, then a first-order velocity update.
* Rigid contact uses the frozen funnel triangle mesh (unsigned closest-point
  distance against an offset shell) and the ground half-space, with Coulomb
  friction as a tangential position correction bounded by the resolved normal
  displacement.
* Dense cloth self-contact is vertex-vertex repulsion at a thickness proxy,
  producing many simultaneous folds that slide and compress to the throat.
"""
from __future__ import annotations

import os

import numpy as np

from worldsolver.contracts import StateFrame


def _load_obj(path):
    """Minimal OBJ reader (vertices + triangulated faces)."""
    verts = []
    faces = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                p = line.split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                ids = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                for k in range(1, len(ids) - 1):
                    faces.append([ids[0], ids[k], ids[k + 1]])
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces, dtype=np.int64) - 1
    if v.size == 0 or f.size == 0:
        raise ValueError("empty obstacle mesh")
    return v, f
def _fallback_funnel():
    """Analytic tapered shell used only if the frozen asset cannot be read."""
    hx0, hy0, z0 = 0.34, 0.24, 0.48
    hx1, hy1, z1 = 1.31, 0.91, 1.72
    low = np.array([[-hx0, -hy0, z0], [hx0, -hy0, z0], [hx0, hy0, z0], [-hx0, hy0, z0]])
    up = np.array([[-hx1, -hy1, z1], [hx1, -hy1, z1], [hx1, hy1, z1], [-hx1, hy1, z1]])
    verts = np.vstack((low, up))
    faces = []
    for k in range(4):
        j = (k + 1) % 4
        faces.append([k, j, 4 + j])
        faces.append([k, 4 + j, 4 + k])
    return verts, np.asarray(faces, dtype=np.int64)


def _closest_points_on_triangles(points, tri):
    """Vectorised closest point of every point on every triangle (Ericson)."""
    a, b, c = tri["a"], tri["b"], tri["c"]
    ab, ac, bc = tri["ab"], tri["ac"], tri["bc"]
    tiny = 1e-30
    ap = points[:, None, :] - a[None, :, :]
    d1 = np.einsum("nmk,mk->nm", ap, ab)
    d2 = np.einsum("nmk,mk->nm", ap, ac)
    bp = points[:, None, :] - b[None, :, :]
    d3 = np.einsum("nmk,mk->nm", bp, ab)
    d4 = np.einsum("nmk,mk->nm", bp, ac)
    cp = points[:, None, :] - c[None, :, :]
    d5 = np.einsum("nmk,mk->nm", cp, ab)
    d6 = np.einsum("nmk,mk->nm", cp, ac)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    inv = 1.0 / np.where(np.abs(denom) < tiny, tiny, denom)
    v = vb * inv
    w = vc * inv
    closest = a[None] + ab[None] * v[..., None] + ac[None] * w[..., None]

    den = d1 - d3
    t = np.clip(d1 / np.where(np.abs(den) < tiny, tiny, den), 0.0, 1.0)
    mask = ((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0))[..., None]
    closest = np.where(mask, a[None] + ab[None] * t[..., None], closest)

    den = d2 - d6
    t = np.clip(d2 / np.where(np.abs(den) < tiny, tiny, den), 0.0, 1.0)
    mask = ((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0))[..., None]
    closest = np.where(mask, a[None] + ac[None] * t[..., None], closest)

    num = d4 - d3
    den = num + (d5 - d6)
    t = np.clip(num / np.where(np.abs(den) < tiny, tiny, den), 0.0, 1.0)
    mask = ((va <= 0.0) & (num >= 0.0) & ((d5 - d6) >= 0.0))[..., None]
    closest = np.where(mask, b[None] + bc[None] * t[..., None], closest)

    shape = closest.shape
    mask = ((d1 <= 0.0) & (d2 <= 0.0))[..., None]
    closest = np.where(mask, np.broadcast_to(a[None], shape), closest)
    mask = ((d3 >= 0.0) & (d4 <= d3))[..., None]
    closest = np.where(mask, np.broadcast_to(b[None], shape), closest)
    mask = ((d6 >= 0.0) & (d5 <= d6))[..., None]
    closest = np.where(mask, np.broadcast_to(c[None], shape), closest)
    return closest
class BenchmarkSolver:
    """Substepped XPBD thin-shell solver with rigid and self contact."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(getattr(scene, "config", {}) or {})
        cloth = dict(cfg.get("cloth", {}) or {})
        frame0 = scene.initial_state()

        self.nx, self.ny = (int(v) for v in cloth.get("grid_shape", (17, 17)))
        self.thickness = float(cloth.get("thickness", 0.015))
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))

        self.x = np.array(frame0.position, dtype=float)
        self.n = len(self.x)
        self.v = np.zeros_like(self.x)
        mass = frame0.mass
        if mass is None:
            total = float(cloth.get("total_mass", 0.32))
            mass = np.full(self.n, total / self.n)
        self.mass = np.array(mass, dtype=float)
        self.w = np.where(self.mass > 0.0, 1.0 / np.maximum(self.mass, 1e-12), 0.0)
        self.boundary = np.array(
            frame0.attributes.get("boundary_vertex", np.zeros(self.n)), dtype=np.int32
        )

        # material / numerical parameters
        self.substeps = 20
        self.iters = 3
        self.a_stretch = 1.0e-8       # near inextensible
        self.a_shear = 2.0e-3
        self.a_bend = 6.0             # very soft -> many fine folds
        self.drag_tau = 6.0           # linear air drag time constant [s]
        self.v_max = 12.0             # CFL guard: 12 m/s * dt_sub << shell width
        self.mu_obstacle = 0.22
        self.mu_self = 0.14

        rest = self._rest_grid(cloth)
        self.groups = self._build_constraints(rest)
        edge = min(
            float(np.linalg.norm(rest[1] - rest[0])),
            float(np.linalg.norm(rest[self.nx] - rest[0])),
        )
        self.mean_edge = edge
        # discrete-thickness proxy: vertex repulsion wide enough that no
        # triangle interior can be pierced between sampled vertices
        self.self_d = 0.70 * edge
        self.obs_offset = 0.5 * self.thickness + 0.005
        self.ground_offset = 0.5 * self.thickness + 0.004
        self.ground_top = 0.0

        self._build_topology(rest)
        self._load_obstacle(cfg)

        rng = np.random.default_rng(int(seed))
        self._jitter = rng.normal(0.0, 1.0, size=self.x.shape)
        self._jitter[:, :2] *= 2.0e-4
        self._jitter[:, 2] *= 6.0e-4
        self._jitter_applied = False
        self.time = 0.0
        self._steps = 0
        self._min_gap = float("inf")
    # ---------------------------------------------------------------- setup
    def _rest_grid(self, cloth):
        sx, sy = (float(v) for v in cloth.get("size", (3.0, 2.5)))
        ax = np.linspace(-0.5 * sx, 0.5 * sx, self.nx)
        ay = np.linspace(-0.5 * sy, 0.5 * sy, self.ny)
        xx, yy = np.meshgrid(ax, ay, indexing="xy")
        self.uv = np.column_stack((xx.ravel(), yy.ravel()))
        return np.array(self.x, dtype=float)

    def _build_constraints(self, rest):
        nx, ny = self.nx, self.ny
        ids = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx)
        groups = []

        def add(i, j, alpha):
            i = np.asarray(i).ravel()
            j = np.asarray(j).ravel()
            if i.size == 0:
                return
            length = np.linalg.norm(rest[j] - rest[i], axis=1)
            groups.append({"i": i, "j": j, "L": length, "alpha": float(alpha)})

        for p in (0, 1):
            cols = np.arange(p, nx - 1, 2)
            if cols.size:
                add(ids[:, cols], ids[:, cols + 1], self.a_stretch)
            rows = np.arange(p, ny - 1, 2)
            if rows.size:
                add(ids[rows, :], ids[rows + 1, :], self.a_stretch)
        for px in (0, 1):
            for py in (0, 1):
                cols = np.arange(px, nx - 1, 2)
                rows = np.arange(py, ny - 1, 2)
                if not (cols.size and rows.size):
                    continue
                add(ids[np.ix_(rows, cols)], ids[np.ix_(rows + 1, cols + 1)], self.a_shear)
                add(ids[np.ix_(rows, cols + 1)], ids[np.ix_(rows + 1, cols)], self.a_shear)
        for p in range(4):
            cols = np.arange(p, nx - 2, 4)
            if cols.size:
                add(ids[:, cols], ids[:, cols + 2], self.a_bend)
            rows = np.arange(p, ny - 2, 4)
            if rows.size:
                add(ids[rows, :], ids[rows + 2, :], self.a_bend)
        return groups

    def _build_topology(self, rest):
        nx, ny = self.nx, self.ny
        ids = np.arange(nx * ny, dtype=np.int64).reshape(ny, nx)
        v00 = ids[:-1, :-1].ravel()
        v10 = ids[:-1, 1:].ravel()
        v11 = ids[1:, 1:].ravel()
        v01 = ids[1:, :-1].ravel()
        self.faces = np.vstack(
            (np.column_stack((v00, v10, v11)), np.column_stack((v00, v11, v01)))
        )
        fa, fb, fc = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        dm = np.stack((self.uv[fb] - self.uv[fa], self.uv[fc] - self.uv[fa]), axis=2)
        det = dm[:, 0, 0] * dm[:, 1, 1] - dm[:, 0, 1] * dm[:, 1, 0]
        det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        self.dm_inv = np.empty_like(dm)
        self.dm_inv[:, 0, 0] = dm[:, 1, 1] / det
        self.dm_inv[:, 1, 1] = dm[:, 0, 0] / det
        self.dm_inv[:, 0, 1] = -dm[:, 0, 1] / det
        self.dm_inv[:, 1, 0] = -dm[:, 1, 0] / det
        counts = np.bincount(self.faces.ravel(), minlength=self.n).astype(float)
        self.vertex_face_weight = 1.0 / np.maximum(counts, 1.0)

        # grid neighbour mask: exclude near neighbours from self-contact pairs
        gi = np.arange(self.n) % nx
        gj = np.arange(self.n) // nx
        self.grid_i = gi
        self.grid_j = gj

    def _load_obstacle(self, cfg):
        here = os.path.dirname(os.path.abspath(__file__))
        roots = [os.getcwd(), os.path.dirname(here), here]
        verts = faces = None
        for root in roots:
            path = os.path.join(root, "assets", "funnel.obj")
            if os.path.isfile(path):
                try:
                    verts, faces = _load_obj(path)
                    break
                except Exception:
                    verts = faces = None
        if verts is None:
            verts, faces = _fallback_funnel()
        a = verts[faces[:, 0]]
        b = verts[faces[:, 1]]
        c = verts[faces[:, 2]]
        nrm = np.cross(b - a, c - a)
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        self.tri = {
            "a": np.ascontiguousarray(a),
            "b": np.ascontiguousarray(b),
            "c": np.ascontiguousarray(c),
            "ab": np.ascontiguousarray(b - a),
            "ac": np.ascontiguousarray(c - a),
            "bc": np.ascontiguousarray(c - b),
            "n": nrm / np.where(ln < 1e-16, 1.0, ln),
        }
        self.n_tri = len(a)
    # ------------------------------------------------------------- physics
    def _project(self, x, dt2):
        """XPBD Gauss-Seidel projection over vertex-disjoint groups."""
        for g in self.groups:
            i, j, rest, alpha = g["i"], g["j"], g["L"], g["alpha"]
            d = x[j] - x[i]
            ln = np.linalg.norm(d, axis=1)
            safe = np.where(ln < 1e-12, 1.0, ln)
            nvec = d / safe[:, None]
            cval = ln - rest
            wi = self.w[i]
            wj = self.w[j]
            denom = wi + wj + alpha / dt2
            lam = -cval / np.where(denom < 1e-30, 1e-30, denom)
            corr = lam[:, None] * nvec
            x[i] -= wi[:, None] * corr
            x[j] += wj[:, None] * corr

    def _obstacle_contact(self, x):
        """Unsigned closest-point contact against the rigid funnel shell."""
        cp = _closest_points_on_triangles(x, self.tri)
        d = x[:, None, :] - cp
        dist = np.linalg.norm(d, axis=2)
        k = np.argmin(dist, axis=1)
        rows = np.arange(len(x))
        dmin = dist[rows, k]
        delta = d[rows, k]
        pen = self.obs_offset - dmin
        active = pen > 0.0
        push = np.zeros_like(x)
        if np.any(active):
            safe = np.maximum(dmin, 1e-9)[:, None]
            nrm = np.where(dmin[:, None] > 1e-7, delta / safe, self.tri["n"][k])
            push = np.where(active[:, None], nrm * np.maximum(pen, 0.0)[:, None], 0.0)
        return push, dmin

    def _ground_contact(self, x):
        push = np.zeros_like(x)
        pen = (self.ground_top + self.ground_offset) - x[:, 2]
        act = pen > 0.0
        if np.any(act):
            push[act, 2] = pen[act]
        return push

    def _self_pairs(self, x):
        """Non-adjacent vertex pairs closer than the cloth thickness proxy."""
        sq = np.sum(x * x, axis=1)
        gram = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
        np.fill_diagonal(gram, np.inf)
        ii, jj = np.nonzero(gram < self.self_d * self.self_d)
        keep = ii < jj
        ii, jj = ii[keep], jj[keep]
        if ii.size:
            chev = np.maximum(
                np.abs(self.grid_i[ii] - self.grid_i[jj]),
                np.abs(self.grid_j[ii] - self.grid_j[jj]),
            )
            keep = chev > 1
            ii, jj = ii[keep], jj[keep]
        return ii, jj

    def _self_contact(self, x):
        ii, jj = self._self_pairs(x)
        push = np.zeros_like(x)
        if ii.size == 0:
            return push, 0
        d = x[jj] - x[ii]
        ln = np.linalg.norm(d, axis=1)
        safe = np.where(ln < 1e-9, 1e-9, ln)
        nvec = d / safe[:, None]
        pen = np.maximum(self.self_d - ln, 0.0)
        wi = self.w[ii]
        wj = self.w[jj]
        tot = np.where(wi + wj < 1e-30, 1.0, wi + wj)
        corr = (pen / tot)[:, None] * nvec
        np.add.at(push, ii, -wi[:, None] * corr)
        np.add.at(push, jj, wj[:, None] * corr)
        return push, int(ii.size)
    def _substep(self, dt):
        x0 = self.x
        acc = self.gravity[None, :] - self.v / self.drag_tau
        v_pred = self.v + dt * acc
        x = x0 + dt * v_pred

        dt2 = dt * dt
        for _ in range(self.iters):
            self._project(x, dt2)

        n_self = 0
        total_push = np.zeros_like(x)
        for _ in range(2):
            push_o, _ = self._obstacle_contact(x)
            push_s, n_self = self._self_contact(x)
            push_g = self._ground_contact(x)
            push = push_o + push_s + push_g
            x += push
            total_push += push

        # Coulomb friction: bound tangential slip by resolved normal motion
        pmag = np.linalg.norm(total_push, axis=1)
        act = pmag > 1e-9
        if np.any(act):
            nrm = total_push[act] / pmag[act][:, None]
            slip = x[act] - x0[act]
            tang = slip - np.sum(slip * nrm, axis=1)[:, None] * nrm
            tmag = np.linalg.norm(tang, axis=1)
            limit = self.mu_obstacle * pmag[act]
            scale = np.minimum(limit, tmag) / np.where(tmag < 1e-12, 1e-12, tmag)
            x[act] -= tang * scale[:, None]

        self.v = (x - x0) / dt
        speed = np.linalg.norm(self.v, axis=1)
        hot = speed > self.v_max
        if np.any(hot):
            self.v[hot] *= (self.v_max / speed[hot])[:, None]
        self.x = x
        self._steps += 1
        self._n_self = n_self

    # ------------------------------------------------------------- outputs
    def _deformation_gradient(self):
        fa, fb, fc = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        e1 = self.x[fb] - self.x[fa]
        e2 = self.x[fc] - self.x[fa]
        l1 = np.linalg.norm(e1, axis=1, keepdims=True)
        t1 = e1 / np.where(l1 < 1e-12, 1.0, l1)
        nrm = np.cross(e1, e2)
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        nhat = nrm / np.where(ln < 1e-14, 1.0, ln)
        t2 = np.cross(nhat, t1)
        ds = np.empty((len(fa), 2, 2))
        ds[:, 0, 0] = np.sum(e1 * t1, axis=1)
        ds[:, 1, 0] = np.sum(e1 * t2, axis=1)
        ds[:, 0, 1] = np.sum(e2 * t1, axis=1)
        ds[:, 1, 1] = np.sum(e2 * t2, axis=1)
        f_tri = ds @ self.dm_inv
        f_vert = np.zeros((self.n, 2, 2))
        for col in (fa, fb, fc):
            np.add.at(f_vert, col, f_tri)
        f_vert *= self.vertex_face_weight[:, None, None]
        return np.nan_to_num(f_vert, nan=0.0, posinf=0.0, neginf=0.0)

    def _contact_gap(self):
        """True geometric surface gaps: mid-surface distance minus thickness."""
        half = 0.5 * self.thickness
        _, dmin = self._obstacle_contact(self.x)
        gap = dmin - half
        gap = np.minimum(gap, self.x[:, 2] - self.ground_top - half)
        ii, jj = self._self_pairs(self.x)
        if ii.size:
            d = np.linalg.norm(self.x[jj] - self.x[ii], axis=1) - self.thickness
            self_gap = np.full(self.n, np.inf)
            np.minimum.at(self_gap, ii, d)
            np.minimum.at(self_gap, jj, d)
            gap = np.where(np.isfinite(self_gap), np.minimum(gap, self_gap), gap)
        gap = np.nan_to_num(gap, nan=0.0, posinf=1.0, neginf=-1.0)
        self._min_gap = min(self._min_gap, float(gap.min()))
        return gap
    def _frame(self, time) -> StateFrame:
        ext = self.mass[:, None] * (self.gravity[None, :] - self.v / self.drag_tau)
        frame = StateFrame(
            float(time),
            np.ascontiguousarray(self.x, dtype=float),
            np.ascontiguousarray(self.v, dtype=float),
            mass=np.ascontiguousarray(self.mass, dtype=float),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": np.ascontiguousarray(self.mass, dtype=float),
                "deformation_gradient": self._deformation_gradient(),
                "boundary_vertex": np.asarray(self.boundary, dtype=np.int32),
            },
            fields={
                "contact_signed_distance": self._contact_gap(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": np.ascontiguousarray(ext, dtype=float),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- driver
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > 0.0 and not self._jitter_applied:
            # symmetry-breaking surface roughness (sub-millimetre), deterministic
            self.x = self.x + self._jitter
            self._jitter_applied = True
        span = target - self.time
        if span > 1e-12:
            base = self.sample_dt / self.substeps
            nsub = max(1, int(round(span / base)))
            dt = span / nsub
            for _ in range(nsub):
                self._substep(dt)
            self.time = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "substepped XPBD thin shell (stretch/shear/bend) with "
                      "mesh contact, self-contact and Coulomb friction",
            "vertices": int(self.n),
            "faces": int(len(self.faces)),
            "obstacle_triangles": int(self.n_tri),
            "substeps_per_sample": int(self.substeps),
            "solver_iterations": int(self.iters),
            "substeps_taken": int(self._steps),
            "self_contact_distance": float(self.self_d),
            "obstacle_offset": float(self.obs_offset),
            "min_contact_gap": float(self._min_gap),
            "self_contact_pairs_last_step": int(getattr(self, "_n_self", 0)),
        }
