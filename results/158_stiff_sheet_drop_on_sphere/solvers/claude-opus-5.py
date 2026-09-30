"""Stiff-sheet drop on a sphere: elasto-plastic discrete-shell solver.

Physical model
--------------
The sheet is a triangulated discrete shell:

* In-plane: mesh edges (grid edges plus one diagonal per quad) are treated as
  inextensible distance constraints, so triangles are effectively in-plane
  rigid (paper-like: negligible stretch or shear).
* Bending: every interior edge carries a hinge spring between the two opposite
  vertices of its adjacent triangles; the rest length encodes the hinge angle.
  The spring is stiff (stiff-sheet regime) and elasto-plastic: once the hinge
  force exceeds a yield value, a return mapping permanently shortens the rest
  length, creasing the sheet. That is what turns the drop into a small number
  of sharp, persistent folds rather than a soft cloth drape.
* Contact: analytic signed distances against the frozen static ground box and
  the frozen kinematic sphere, plus vertex-vertex self contact with a cloth
  thickness offset, resolved as inequality position constraints with
  Coulomb-like tangential velocity damping.

Time stepping is a fixed-step symplectic-Euler predictor followed by XPBD
constraint projection over precomputed vertex-disjoint constraint colours,
which is stable for the near-rigid in-plane constraints and deterministic.
"""
from __future__ import annotations

import json
import os

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-12


def _greedy_colors(pairs: np.ndarray, n_vertices: int) -> list[np.ndarray]:
    """Partition two-vertex constraints into vertex-disjoint groups."""
    if len(pairs) == 0:
        return []
    colors = np.full(len(pairs), -1, dtype=np.int64)
    used: list[set] = [set() for _ in range(n_vertices)]
    for k in range(len(pairs)):
        a = int(pairs[k, 0])
        b = int(pairs[k, 1])
        taken = used[a] | used[b]
        c = 0
        while c in taken:
            c += 1
        colors[k] = c
        used[a].add(c)
        used[b].add(c)
    n_colors = int(colors.max()) + 1
    return [np.nonzero(colors == c)[0].astype(np.int64) for c in range(n_colors)]


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=np.float64)
        self.v = np.array(frame0.velocity, dtype=np.float64)
        self.mass = np.array(frame0.mass, dtype=np.float64)
        self.n = int(len(self.x))
        self.boundary_vertex = np.asarray(
            frame0.attributes.get("boundary_vertex", np.zeros(self.n)), dtype=np.int32
        )
        self.w = 1.0 / np.maximum(self.mass, 1e-12)

        cloth = cfg["cloth"]
        self.grid_shape = (int(cloth["grid_shape"][0]), int(cloth["grid_shape"][1]))
        self.thickness = float(cloth["thickness"])
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)

        # frozen geometry from scene_geometry.yaml
        self.sphere_c = np.array([0.0, 0.0, 0.68], dtype=np.float64)
        self.sphere_r = 0.62
        self.ground_top = 0.0
        self.ground_half = np.array([2.4, 2.0], dtype=np.float64)

        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])
        self.n_sub = 40
        self.dt = self.sample_dt / self.n_sub
        self.n_iter = 5

        self._resolve_initial_overlap()
        self._build_topology()
        self._build_material()
        self._build_contact_params()
        self._build_deformation_basis()

        # deterministic, tiny symmetry break so folds are not a perfectly
        # symmetric drape; amplitude far below any physical velocity scale
        self.v += rng.normal(0.0, 1.0e-3, size=self.v.shape)
        self.v[:, 2] *= 0.5

        self.time = 0.0
        self.steps = 0
        self._plastic_flow = 0.0

    def _resolve_initial_overlap(self) -> None:
        """Keep the frozen flat sheet flat, translated clear of the sphere crown.

        The frozen cloth centre plane (z = 1.22) lies just below the sphere
        crown (z = 1.30), so the prescribed configuration would start already
        interpenetrating.  A rigid vertical translation is the minimal
        correction that preserves the frozen flat horizontal initial state and
        the required "starts above the sphere" condition.
        """
        crown = float(self.sphere_c[2] + self.sphere_r)
        clear = crown + 0.5 * self.thickness + 0.01
        z0 = float(self.x[:, 2].min())
        if z0 < clear:
            self.x[:, 2] += clear - z0
        self._initial_lift = float(self.x[:, 2].min() - z0)

    def _load_faces(self) -> np.ndarray:
        path = os.path.join(os.getcwd(), "assets", "cloth_topology.json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            faces = np.asarray(data["faces"], dtype=np.int64).reshape(-1, 3)
            if int(faces.max()) < self.n:
                return faces
        except Exception:
            pass
        nx, ny = self.grid_shape
        tri = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                b = a + 1
                c = a + nx
                d = c + 1
                tri.append((a, b, d))
                tri.append((a, d, c))
        return np.asarray(tri, dtype=np.int64)

    def _build_topology(self) -> None:
        faces = self._load_faces()
        self.faces = faces
        edge_map: dict = {}
        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            for (p, q, o) in ((a, b, c), (b, c, a), (c, a, b)):
                key = (p, q) if p < q else (q, p)
                edge_map.setdefault(key, []).append(o)

        keys = sorted(edge_map.keys())
        self.edges = np.asarray(keys, dtype=np.int64).reshape(-1, 2)
        d0 = self.x[self.edges[:, 0]] - self.x[self.edges[:, 1]]
        self.edge_rest = np.linalg.norm(d0, axis=1)
        self.edge_groups = _greedy_colors(self.edges, self.n)
        self.mean_edge = float(np.mean(self.edge_rest))

        # discrete umbrella Laplacian over the primary (non-diagonal) stencil
        adj = np.zeros((self.n, self.n), dtype=np.float64)
        for k in range(len(self.edges)):
            a = int(self.edges[k, 0])
            b = int(self.edges[k, 1])
            adj[a, b] = 1.0
            adj[b, a] = 1.0
        valence = np.maximum(adj.sum(axis=1), 1.0)
        lap = adj / valence[:, None]
        lap[np.arange(self.n), np.arange(self.n)] -= 1.0
        self.lap = lap
        self.lap_t = lap.T.copy()
        self.n_bend_stencils = int(np.count_nonzero(valence > 3))

    def _build_material(self) -> None:
        """Stiff plate bending with an elasto-plastic reference curvature.

        Bending energy is the standard isometric (quadratic) plate energy
        ``E = 1/2 * kb * sum_i |L x_i - p_i|^2`` where ``L`` is the discrete
        umbrella Laplacian (a curvature measure) and ``p_i`` is the plastic
        reference curvature.  ``kb = 16 B / h^2`` follows from matching
        ``1/2 B kappa^2`` per unit area for a regular stencil of spacing ``h``.
        Once the elastic curvature ``|L x - p|`` exceeds the yield value the
        return mapping transfers the excess into ``p``, i.e. a permanent crease.
        """
        self.bending_modulus = 0.25            # B [N m], stiff-sheet regime
        h = max(self.mean_edge, 1e-6)
        self.kb = 16.0 * self.bending_modulus / (h * h)   # [N/m]
        self.cb = 0.0015                       # Rayleigh bending damping [s]
        self.curv_yield = 0.020                # yield on |L x| [m]
        self.curv_plastic_max = 0.080          # crease saturation [m]
        self.plastic_rate = 0.35               # return-mapping fraction/substep
        # scalar plastic curvature magnitude per vertex (frame invariant; the
        # crease direction always follows the current bending direction)
        self.plastic = np.zeros(self.n, dtype=np.float64)
        self.f_bend_max = 60.0                 # per-vertex bending force clamp [N]

    def _build_contact_params(self) -> None:
        self.contact_offset = 0.5 * self.thickness + 0.0035
        self.self_dist = 0.34 * self.mean_edge + 0.5 * self.thickness
        self.self_broad = 2.0 * self.self_dist
        self.mu_obstacle = 0.55
        self.max_correction = 0.02
        self.v_max = 25.0
        self.damping = 0.9994

        nx, ny = self.grid_shape
        idx = np.arange(self.n)
        gi = (idx % nx).astype(np.int64)
        gj = (idx // nx).astype(np.int64)
        ring = np.maximum(np.abs(gi[:, None] - gi[None, :]), np.abs(gj[:, None] - gj[None, :]))
        self.self_allowed = ring > 2

    def _build_deformation_basis(self) -> None:
        mat = self.x[:, :2] - self.x[:, :2].mean(axis=0)
        self.mat_uv = mat
        faces = self.faces
        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        dm = np.empty((len(faces), 2, 2), dtype=np.float64)
        dm[:, :, 0] = mat[f1] - mat[f0]
        dm[:, :, 1] = mat[f2] - mat[f0]
        det = dm[:, 0, 0] * dm[:, 1, 1] - dm[:, 0, 1] * dm[:, 1, 0]
        det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        inv = np.empty_like(dm)
        inv[:, 0, 0] = dm[:, 1, 1] / det
        inv[:, 1, 1] = dm[:, 0, 0] / det
        inv[:, 0, 1] = -dm[:, 0, 1] / det
        inv[:, 1, 0] = -dm[:, 1, 0] / det
        self.dm_inv = inv
        counts = np.zeros(self.n, dtype=np.float64)
        np.add.at(counts, faces.ravel(), 1.0)
        self.vertex_face_count = np.maximum(counts, 1.0)

    # -------------------------------------------------------------- stepping
    def _project_edges(self, x: np.ndarray) -> None:
        """Inextensible in-plane distance constraints (rigid projection)."""
        lim = self.max_correction * 20.0
        for g in self.edge_groups:
            i = self.edges[g, 0]
            j = self.edges[g, 1]
            d = x[i] - x[j]
            length = np.sqrt(np.einsum("ij,ij->i", d, d))
            nrm = d / np.maximum(length, _EPS)[:, None]
            c = length - self.edge_rest[g]
            wi = self.w[i]
            wj = self.w[j]
            s = np.clip(c / np.maximum(wi + wj, _EPS), -lim, lim)
            x[i] -= (wi * s)[:, None] * nrm
            x[j] += (wj * s)[:, None] * nrm

    def _bending_force(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Elasto-plastic plate bending force, f = -kb L^T (L x - p) - damping.

        ``L x`` is the discrete mean-curvature (umbrella) vector.  The plastic
        reference curvature ``p`` is stored as a magnitude and reintroduced
        along the current curvature direction, so creases persist under rigid
        motion without introducing a preferred world direction.
        """
        curv = self.lap @ x                                # (N,3) curvature vec
        mag = np.linalg.norm(curv, axis=1)
        direction = curv / np.maximum(mag, _EPS)[:, None]
        # the relaxed (creased) curvature never exceeds the current curvature,
        # so the plastic term can only unload stress, never inject energy
        ref = np.minimum(self.plastic, mag)
        elastic = curv - ref[:, None] * direction
        f = -self.kb * (self.lap_t @ elastic)
        # Rayleigh (curvature-rate) damping keeps the stiff explicit term stable
        f -= (self.kb * self.cb) * (self.lap_t @ (self.lap @ v))
        norm = np.linalg.norm(f, axis=1)
        over = norm > self.f_bend_max
        if np.any(over):
            f[over] *= (self.f_bend_max / norm[over])[:, None]
        self._last_curv_mag = mag
        return f

    def _obstacle_gaps(self, x: np.ndarray) -> np.ndarray:
        rel = x - self.sphere_c
        dist = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        gap_sphere = dist - (self.sphere_r + self.contact_offset)
        gap_ground = x[:, 2] - (self.ground_top + self.contact_offset)
        outside = ~np.all(np.abs(x[:, :2]) <= self.ground_half, axis=1)
        gap_ground = np.where(outside, gap_ground + 10.0, gap_ground)
        return np.minimum(gap_sphere, gap_ground)

    def _project_obstacles(self, x: np.ndarray) -> None:
        rel = x - self.sphere_c
        dist = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        target = self.sphere_r + self.contact_offset
        pen = target - dist
        hit = pen > 0.0
        if np.any(hit):
            nrm = rel[hit] / np.maximum(dist[hit], _EPS)[:, None]
            corr = np.minimum(pen[hit], self.max_correction)
            x[hit] += corr[:, None] * nrm
        inside = np.all(np.abs(x[:, :2]) <= self.ground_half, axis=1)
        floor = self.ground_top + self.contact_offset
        below = inside & (x[:, 2] < floor)
        if np.any(below):
            x[below, 2] += np.minimum(floor - x[below, 2], self.max_correction)

    def _self_pairs(self, x: np.ndarray) -> np.ndarray:
        diff = x[:, None, :] - x[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        close = (d2 < self.self_broad * self.self_broad) & self.self_allowed
        i, j = np.nonzero(np.triu(close, 1))
        return np.column_stack((i, j)).astype(np.int64)

    def _project_self(self, x: np.ndarray, pairs: np.ndarray) -> None:
        if len(pairs) == 0:
            return
        i = pairs[:, 0]
        j = pairs[:, 1]
        d = x[i] - x[j]
        length = np.sqrt(np.einsum("ij,ij->i", d, d))
        act = length < self.self_dist
        if not np.any(act):
            return
        i = i[act]
        j = j[act]
        length = length[act]
        d = d[act]
        nrm = d / np.maximum(length, _EPS)[:, None]
        deg = length < 1e-9
        if np.any(deg):
            nrm[deg] = np.array([0.0, 0.0, 1.0])
        pen = np.minimum(self.self_dist - length, self.max_correction)
        wi = self.w[i]
        wj = self.w[j]
        s = pen / np.maximum(wi + wj, _EPS)
        dx = np.zeros_like(x)
        np.add.at(dx, i, (wi * s)[:, None] * nrm)
        np.add.at(dx, j, -(wj * s)[:, None] * nrm)
        cnt = np.zeros(len(x), dtype=np.float64)
        np.add.at(cnt, i, 1.0)
        np.add.at(cnt, j, 1.0)
        x += dx / np.maximum(cnt, 1.0)[:, None]

    def _apply_friction(self, x: np.ndarray) -> None:
        tang = 1.0 - self.mu_obstacle * 0.35
        rel = x - self.sphere_c
        dist = np.sqrt(np.einsum("ij,ij->i", rel, rel))
        band = self.contact_offset + 0.02 * self.mean_edge
        touch = dist < self.sphere_r + band
        if np.any(touch):
            nrm = rel[touch] / np.maximum(dist[touch], _EPS)[:, None]
            vsel = self.v[touch]
            vn = np.einsum("ij,ij->i", vsel, nrm)
            vt = vsel - vn[:, None] * nrm
            vn = np.maximum(vn, 0.0)  # no inward normal velocity at contact
            self.v[touch] = vt * tang + vn[:, None] * nrm
        floor = self.ground_top + self.contact_offset
        gnd = (x[:, 2] < floor + 0.02 * self.mean_edge) & np.all(
            np.abs(x[:, :2]) <= self.ground_half, axis=1
        )
        if np.any(gnd):
            self.v[gnd, 0] *= tang
            self.v[gnd, 1] *= tang
            self.v[gnd, 2] = np.maximum(self.v[gnd, 2], 0.0)

    def _plastic_update(self, x: np.ndarray) -> None:
        """Return mapping: curvature beyond yield becomes a permanent crease."""
        mag = np.linalg.norm(self.lap @ x, axis=1)
        elastic = mag - np.minimum(self.plastic, mag)
        excess = elastic - self.curv_yield
        act = excess > 0.0
        if not np.any(act):
            return
        dp = self.plastic_rate * excess[act]
        new = np.minimum(self.plastic[act] + dp, self.curv_plastic_max)
        self._plastic_flow += float(np.abs(new - self.plastic[act]).sum())
        self.plastic[act] = new

    def _substep(self) -> None:
        dt = self.dt
        x_prev = self.x.copy()
        self.v *= self.damping
        self.v += dt * self.gravity[None, :]
        speed = np.linalg.norm(self.v, axis=1)
        over = speed > self.v_max
        if np.any(over):
            self.v[over] *= (self.v_max / speed[over])[:, None]
        # stiff elasto-plastic bending as an explicit force with rate damping
        f_bend = self._bending_force(x_prev, self.v)
        self.v += dt * (f_bend * self.w[:, None])
        speed = np.linalg.norm(self.v, axis=1)
        over = speed > self.v_max
        if np.any(over):
            self.v[over] *= (self.v_max / speed[over])[:, None]
        x = x_prev + dt * self.v

        pairs = self._self_pairs(x)
        for _ in range(self.n_iter):
            self._project_edges(x)
            self._project_self(x, pairs)
            self._project_obstacles(x)
        self._project_edges(x)
        self._project_self(x, pairs)
        self._project_obstacles(x)

        if not np.all(np.isfinite(x)):
            x = np.where(np.isfinite(x), x, x_prev)
        self.x = x
        self.v = np.clip((self.x - x_prev) / dt, -self.v_max, self.v_max)
        self._apply_friction(self.x)
        self._plastic_update(self.x)
        self.steps += 1
        self.time = self.steps * dt

    # ---------------------------------------------------------------- output
    def _deformation_gradient(self) -> np.ndarray:
        """Per-vertex 2x2 surface deformation gradient in material coords."""
        faces = self.faces
        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        ds = np.empty((len(faces), 3, 2), dtype=np.float64)
        ds[:, :, 0] = self.x[f1] - self.x[f0]
        ds[:, :, 1] = self.x[f2] - self.x[f0]
        f3 = np.einsum("nik,nkj->nij", ds, self.dm_inv)  # (F,3,2)
        fu = f3[:, :, 0]
        fv = f3[:, :, 1]
        t1 = fu / np.maximum(np.linalg.norm(fu, axis=1), _EPS)[:, None]
        nvec = np.cross(fu, fv)
        nn = np.linalg.norm(nvec, axis=1)
        nvec = nvec / np.maximum(nn, _EPS)[:, None]
        degenerate = nn < 1e-12
        if np.any(degenerate):
            nvec[degenerate] = np.array([0.0, 0.0, 1.0])
            t1[degenerate] = np.array([1.0, 0.0, 0.0])
        t2 = np.cross(nvec, t1)
        f2d = np.empty((len(faces), 2, 2), dtype=np.float64)
        f2d[:, 0, 0] = np.einsum("ij,ij->i", t1, fu)
        f2d[:, 0, 1] = np.einsum("ij,ij->i", t1, fv)
        f2d[:, 1, 0] = np.einsum("ij,ij->i", t2, fu)
        f2d[:, 1, 1] = np.einsum("ij,ij->i", t2, fv)
        out = np.zeros((self.n, 2, 2), dtype=np.float64)
        for k in range(3):
            np.add.at(out, faces[:, k], f2d)
        out /= self.vertex_face_count[:, None, None]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _self_gap(self) -> np.ndarray:
        diff = self.x[:, None, :] - self.x[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        masked = np.where(self.self_allowed, d2, np.inf)
        nearest = np.sqrt(np.min(masked, axis=1))
        nearest = np.where(np.isfinite(nearest), nearest, 10.0)
        return nearest - self.self_dist

    def _frame(self) -> StateFrame:
        pos = np.ascontiguousarray(self.x, dtype=np.float64)
        vel = np.ascontiguousarray(self.v, dtype=np.float64)
        ext = np.ascontiguousarray(self.mass[:, None] * self.gravity[None, :])
        gaps = np.ascontiguousarray(self._obstacle_gaps(pos), dtype=np.float64)
        frame = StateFrame(
            time=float(max(self.time, 0.0)),
            position=pos,
            velocity=vel,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary_vertex.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": gaps,
                "self_contact_signed_distance": np.ascontiguousarray(
                    self._self_gap(), dtype=np.float64
                ),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": ext,
            },
            scalars={
                "contact_length_scale": float(self.thickness),
                "total_mass": float(self.mass.sum()),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        n_target = int(round(target / self.dt))
        while self.steps < n_target:
            self._substep()
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        gaps = self._obstacle_gaps(self.x)
        return {
            "method": "XPBD elasto-plastic discrete shell (self-implemented)",
            "substeps_taken": int(self.steps),
            "substep_dt": float(self.dt),
            "iterations_per_substep": int(self.n_iter),
            "vertices": int(self.n),
            "edges": int(len(self.edges)),
            "bend_stencils": int(self.n_bend_stencils),
            "bending_modulus": float(self.bending_modulus),
            "bend_stiffness": float(self.kb),
            "curvature_yield": float(self.curv_yield),
            "accumulated_plastic_flow": float(self._plastic_flow),
            "min_obstacle_gap": float(np.min(gaps)),
            "initial_lift": float(getattr(self, "_initial_lift", 0.0)),
            "mean_plastic_curvature": float(np.mean(self.plastic)),
            "creased_hinge_fraction": float(
                np.mean(self.plastic > 1e-4)
            ),
        }
