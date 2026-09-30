"""Mass-spring cloth solver with triangle-mesh contact for the narrow funnel scene.

The solver advances a physical time-stepping loop using only NumPy.  It treats the
cloth as a damped mass-spring network, applies gravity, resolves self-contact with a
vertex repulsion model, and keeps the sheet outside the rigid funnel and ground with
a penalty + projection scheme.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = getattr(scene, "config", {})

        # Gravity vector from the public config.
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # Cloth parameters.
        cloth_cfg = self.config.get("cloth", {})
        self.thickness = float(cloth_cfg.get("thickness", 0.015))
        self.nx, self.ny = map(int, cloth_cfg.get("grid_shape", [17, 17]))

        # Initial state from the scaffold.
        frame = scene.initial_state()
        self.x = np.asarray(frame.position, dtype=float).copy()
        self.v = np.asarray(frame.velocity, dtype=float).copy()
        if frame.mass is not None:
            self.m = np.asarray(frame.mass, dtype=float).copy()
        else:
            total_mass = float(cloth_cfg.get("total_mass", 0.32))
            self.m = np.full(len(self.x), total_mass / len(self.x), dtype=float)
        if frame.body_id is not None:
            self.body_id = np.asarray(frame.body_id, dtype=np.int64).copy()
        else:
            self.body_id = np.zeros(len(self.x), dtype=np.int64)
        self.N = len(self.x)
        self.t = float(frame.time)

        # Load cloth rest topology (vertices and faces).
        self._load_cloth_topology()

        # Build spring edges (structural, shear, bending) from the grid.
        self._build_springs()

        # Load the funnel triangle mesh and pre-compute per-triangle data.
        self._load_funnel()

        # Time stepping parameters.
        self.h = 0.0005  # 0.5 ms substep

        # Material parameters.
        self.k_stretch = 15.0
        self.k_shear = 8.0
        self.k_bend = 0.3
        self.c_damp = 0.2  # global velocity damping [1/s]

        # Contact parameters.
        self.margin = 1.5 * self.thickness
        self.k_contact = 2000.0
        self.c_contact = 10.0

        # Self-contact parameters.
        self.self_cutoff = 4.0 * self.thickness
        self.k_self = 120.0

        # Cumulative source (no mass is added or removed).
        self.cum_mass_source = np.zeros(1, dtype=float)

        # Pre-allocate scratch arrays.
        self._f = np.zeros((self.N, 3), dtype=float)

        self._diag: dict[str, object] = {}

    # ------------------------------------------------------------------ helpers
    def _load_cloth_topology(self) -> None:
        paths = [
            Path(__file__).resolve().parent.parent / "assets" / "cloth_topology.json",
            Path("assets") / "cloth_topology.json",
        ]
        data = None
        for p in paths:
            if p.is_file():
                with open(p) as fh:
                    data = json.load(fh)
                break
        if data is None:
            raise FileNotFoundError("cloth_topology.json not found")
        self.rest_pos = np.asarray(data["vertices"], dtype=float)
        self.faces = np.asarray(data["faces"], dtype=np.int64)

    def _build_springs(self) -> None:
        edges = []
        nx, ny = self.nx, self.ny

        def idx(i: int, j: int) -> int:
            return i * ny + j

        # Structural and shear edges.
        for i in range(nx):
            for j in range(ny):
                v0 = idx(i, j)
                if i + 1 < nx:
                    v1 = idx(i + 1, j)
                    edges.append((v0, v1, "structural"))
                    if j + 1 < ny:
                        edges.append((v0, idx(i + 1, j + 1), "shear"))
                if j + 1 < ny:
                    v1 = idx(i, j + 1)
                    edges.append((v0, v1, "structural"))
                if i + 1 < nx and j - 1 >= 0:
                    edges.append((v0, idx(i + 1, j - 1), "shear"))
                # Bending edges (skip one vertex).
                if i + 2 < nx:
                    edges.append((v0, idx(i + 2, j), "bending"))
                if j + 2 < ny:
                    edges.append((v0, idx(i, j + 2), "bending"))

        edges = list({(min(a, b), max(a, b), k) for a, b, k in edges})
        self.edges_i = np.array([e[0] for e in edges], dtype=np.int64)
        self.edges_j = np.array([e[1] for e in edges], dtype=np.int64)
        self.edges_type = np.array([e[2] for e in edges])

        rest_vec = self.rest_pos[self.edges_j] - self.rest_pos[self.edges_i]
        self.rest_len = np.linalg.norm(rest_vec, axis=1)

        k_map = {"structural": self.k_stretch, "shear": self.k_shear, "bending": self.k_bend}
        self.edge_k = np.array([k_map[t] for t in self.edges_type], dtype=float)

        # Connected mask for self-contact (share an edge or are the same vertex).
        conn = np.zeros((self.N, self.N), dtype=bool)
        conn[self.edges_i, self.edges_j] = True
        conn[self.edges_j, self.edges_i] = True
        conn[np.arange(self.N), np.arange(self.N)] = True
        self.connected_mask = conn

        # Per-vertex structural neighbors for the deformation gradient.
        self._neighbors: list[list[int]] = [[] for _ in range(self.N)]
        for a, b in zip(self.edges_i, self.edges_j):
            if self.edges_type[list(zip(self.edges_i, self.edges_j)).index((a, b))] == "structural":
                self._neighbors[a].append(b)
                self._neighbors[b].append(a)
        # Ensure a non-singular neighbor set by adding diagonal shear if needed.
        for i in range(self.N):
            if len(self._neighbors[i]) < 2:
                # Fallback to any connected vertex.
                self._neighbors[i] = list(set(self._neighbors[i]) | set(self.edges_j[self.edges_i == i]) | set(self.edges_i[self.edges_j == i]))

    def _load_funnel(self) -> None:
        paths = [
            Path(__file__).resolve().parent.parent / "assets" / "funnel.obj",
            Path("assets") / "funnel.obj",
        ]
        verts: list[list[float]] = []
        faces: list[list[int]] = []
        for p in paths:
            if not p.is_file():
                continue
            with open(p) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if parts[0] == "v":
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif parts[0] == "f":
                        faces.append([int(s.split("/")[0]) - 1 for s in parts[1:4]])
            break
        if not verts:
            raise FileNotFoundError("funnel.obj not found")
        self.funnel_v = np.asarray(verts, dtype=float)
        self.funnel_f = np.asarray(faces, dtype=np.int64)
        a = self.funnel_v[self.funnel_f[:, 0]]
        b = self.funnel_v[self.funnel_f[:, 1]]
        c = self.funnel_v[self.funnel_f[:, 2]]
        self.funnel_e0 = b - a
        self.funnel_e1 = c - a
        self.funnel_a = a
        self.funnel_n = np.cross(self.funnel_e0, self.funnel_e1)
        self.funnel_n /= np.linalg.norm(self.funnel_n, axis=1)[:, None]

    # ------------------------------------------------------------------- physics
    def _funnel_closest(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (distance, normal) from each vertex to the nearest funnel triangle.

        The normal points from the closest surface point toward the vertex, so it
        can be used as a double-sided contact direction.
        """
        N = x.shape[0]
        best_d = np.full(N, np.inf, dtype=float)
        best_n = np.zeros((N, 3), dtype=float)
        p = x
        a = self.funnel_a
        e0 = self.funnel_e0
        e1 = self.funnel_e1
        # Precompute triangle metric terms.
        dot00 = np.einsum("ij,ij->i", e0, e0)
        dot01 = np.einsum("ij,ij->i", e0, e1)
        dot11 = np.einsum("ij,ij->i", e1, e1)
        denom = dot00 * dot11 - dot01 * dot01
        for t in range(a.shape[0]):
            # vector from triangle vertex to all cloth points
            w = p - a[t]
            dot02 = np.dot(e0[t], w.T)
            dot12 = np.dot(e1[t], w.T)
            u = (dot11[t] * dot02 - dot01[t] * dot12) / denom[t]
            v = (dot00[t] * dot12 - dot01[t] * dot02) / denom[t]
            # Clamp barycentric coordinates to the triangle.
            u_c = np.clip(u, 0.0, 1.0)
            v_c = np.clip(v, 0.0, 1.0)
            sum_uv = u_c + v_c
            # If outside the diagonal, project back to the edge.
            mask = sum_uv > 1.0
            u_c[mask] = u_c[mask] / sum_uv[mask]
            v_c[mask] = v_c[mask] / sum_uv[mask]
            closest = a[t] + u_c[:, None] * e0[t] + v_c[:, None] * e1[t]
            diff = p - closest
            d = np.linalg.norm(diff, axis=1)
            update = d < best_d
            if np.any(update):
                best_d[update] = d[update]
                best_n[update] = diff[update] / d[update, None]
        return best_d, best_n

    def _signed_distance(self) -> np.ndarray:
        """Per-vertex signed distance to the union of funnel and ground obstacles."""
        # Ground signed distance is simply the height above the plane z=0 minus margin.
        sd_ground = self.x[:, 2]
        # Funnel signed distance using the oriented nearest triangle.
        sd_funnel = self._funnel_signed_distance(self.x)
        # Union of two implicit solids: keep the minimum (most negative) value.
        out = np.minimum(sd_ground, sd_funnel)
        return out

    def _funnel_signed_distance(self, x: np.ndarray) -> np.ndarray:
        """Signed distance to the funnel mesh (negative inside the solid wall)."""
        N = x.shape[0]
        best_sd = np.full(N, np.inf, dtype=float)
        p = x
        a = self.funnel_a
        e0 = self.funnel_e0
        e1 = self.funnel_e1
        n = self.funnel_n
        dot00 = np.einsum("ij,ij->i", e0, e0)
        dot01 = np.einsum("ij,ij->i", e0, e1)
        dot11 = np.einsum("ij,ij->i", e1, e1)
        denom = dot00 * dot11 - dot01 * dot01
        for t in range(a.shape[0]):
            w = p - a[t]
            dot02 = np.dot(e0[t], w.T)
            dot12 = np.dot(e1[t], w.T)
            u = (dot11[t] * dot02 - dot01[t] * dot12) / denom[t]
            v = (dot00[t] * dot12 - dot01[t] * dot02) / denom[t]
            u_c = np.clip(u, 0.0, 1.0)
            v_c = np.clip(v, 0.0, 1.0)
            sum_uv = u_c + v_c
            mask_out = sum_uv > 1.0
            u_c[mask_out] /= sum_uv[mask_out]
            v_c[mask_out] /= sum_uv[mask_out]
            closest = a[t] + u_c[:, None] * e0[t] + v_c[:, None] * e1[t]
            diff = p - closest
            d = np.linalg.norm(diff, axis=1)
            sign = np.sign(np.einsum("ij,j->i", diff, n[t]))
            sign[sign == 0.0] = 1.0
            sd = sign * d
            update = sd < best_sd
            if np.any(update):
                best_sd[update] = sd[update]
        return best_sd

    def _deformation_gradient(self) -> np.ndarray:
        """Return per-vertex 2x2 in-plane deformation gradient in material coordinates."""
        F = np.zeros((self.N, 2, 2), dtype=float)
        X0 = self.rest_pos[:, :2]
        x0 = self.x[:, :2]
        for i in range(self.N):
            nbrs = self._neighbors[i]
            if len(nbrs) < 2:
                F[i] = np.eye(2)
                continue
            dX = X0[nbrs] - X0[i]
            dx = x0[nbrs] - x0[i]
            A = dx.T @ dX  # 2x2
            B = dX.T @ dX  # 2x2
            try:
                invB = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                invB = np.eye(2)
            Fi = A @ invB
            # Clamp singular values to avoid runaway numbers.
            uu, s_s, vv = np.linalg.svd(Fi)
            s_s = np.clip(s_s, 0.1, 10.0)
            F[i] = uu @ np.diag(s_s) @ vv
        return F

    def _external_force(self) -> np.ndarray:
        return self.m[:, None] * self.gravity

    def _spring_forces(self, f: np.ndarray) -> None:
        dx = self.x[self.edges_j] - self.x[self.edges_i]
        dist = np.linalg.norm(dx, axis=1)
        # Guard against zero-length springs.
        safe = dist > 1e-12
        force = np.zeros_like(dx)
        force[safe] = (self.edge_k[safe, None] * (dist[safe] - self.rest_len[safe])[:, None]
                       * dx[safe] / dist[safe, None])
        np.add.at(f, self.edges_i, force)
        np.add.at(f, self.edges_j, -force)

    def _self_contact(self, f: np.ndarray) -> None:
        if self.k_self <= 0.0:
            return
        diff = self.x[:, None, :] - self.x[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        mask = (d2 > 0.0) & (d2 < self.self_cutoff * self.self_cutoff) & (~self.connected_mask)
        if not np.any(mask):
            return
        d = np.sqrt(d2)
        # Vectorized repulsion; only the masked pairs contribute.
        mag = np.zeros_like(d)
        mag[mask] = self.k_self * (self.self_cutoff - d[mask])
        # diff[i,j] = x_i - x_j, so it points the force on i away from j.
        pair_forces = np.zeros_like(diff)
        pair_forces[mask] = mag[mask, None] * diff[mask] / d[mask, None]
        f += pair_forces.sum(axis=1)

    def _obstacle_contact(self, f: np.ndarray) -> None:
        # Funnel contact.
        d, n = self._funnel_closest(self.x)
        pen = self.margin - d
        active = pen > 0.0
        if np.any(active):
            f[active] += (self.k_contact * pen[active])[:, None] * n[active]
            v_n = np.einsum("ij,ij->i", self.v, n)
            f[active] += (self.c_contact * np.maximum(0.0, -v_n[active]))[:, None] * n[active]
        # Ground contact.
        active_g = self.x[:, 2] < self.margin
        if np.any(active_g):
            pen_g = self.margin - self.x[active_g, 2]
            f[active_g, 2] += self.k_contact * pen_g
            f[active_g, 2] += self.c_contact * np.maximum(0.0, -self.v[active_g, 2])

    def _project_obstacles(self) -> None:
        # Funnel projection.
        d, n = self._funnel_closest(self.x)
        pen = self.margin - d
        active = pen > 0.0
        self.x[active] += (pen[active])[:, None] * n[active]
        # Ground projection.
        self.x[:, 2] = np.maximum(self.x[:, 2], self.margin)

    def step(self) -> None:
        f = self._f
        f.fill(0.0)

        # External (gravity) and spring forces.
        f += self._external_force()
        self._spring_forces(f)

        # Contact forces (self + obstacles) before the velocity update.
        self._self_contact(f)
        self._obstacle_contact(f)

        # Velocity update with damping.
        self.v += self.h * (f / self.m[:, None] - self.c_damp * self.v)
        # Position update.
        self.x += self.h * self.v

        # Positional projection to enforce hard constraints and remove drift.
        self._project_obstacles()

    def advance_to(self, time: float) -> StateFrame:
        if time < self.t:
            return self._make_frame(time)
        # Deterministic fixed substep count.
        n_steps = int(round((time - self.t) / self.h))
        for _ in range(n_steps):
            self.step()
        self.t = time
        return self._make_frame(time)

    def _make_frame(self, time: float) -> StateFrame:
        return StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": self._deformation_gradient(),
                "mass_quantity": self.m.copy(),
            },
            fields={
                "contact_signed_distance": self._signed_distance(),
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "external_force": self._external_force(),
            },
            scalars={
                "contact_length_scale": self.thickness,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return self._diag
