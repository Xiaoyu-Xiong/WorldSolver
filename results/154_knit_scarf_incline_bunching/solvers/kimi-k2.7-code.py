"""Mass-spring cloth solver with signed-distance contact for a scarf on an incline."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.frame0 = scene.initial_state()

        # Basic state
        self.X = np.asarray(self.frame0.position, dtype=float).copy()
        self.V = np.asarray(self.frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(self.frame0.mass, dtype=float).copy()
        self.N = self.X.shape[0]
        self.body_id = (
            np.asarray(self.frame0.body_id, dtype=np.int32).copy()
            if self.frame0.body_id is not None
            else np.zeros(self.N, dtype=np.int32)
        )

        # Helpers / bookkeeping
        self.time = 0.0
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.contact_length_scale = float(
            self.frame0.scalars.get("contact_length_scale", 0.015)
        )
        self.margin = 2.0 * self.contact_length_scale

        # Attributes from the scaffold
        self.mass_quantity = np.asarray(
            self.frame0.attributes.get("mass_quantity", self.mass.copy()),
            dtype=float,
        ).copy()
        self.boundary = np.asarray(
            self.frame0.attributes.get("boundary_vertex", np.zeros(self.N, dtype=int)),
            dtype=bool,
        )

        # Load topology and build springs / triangle data
        topo_path = Path("assets/cloth_topology.json")
        topo = json.loads(topo_path.read_text())
        faces = np.asarray(topo["faces"], dtype=int)
        self.faces = faces
        self.n_faces = faces.shape[0]

        # Unique edges (stretch springs)
        edge_set = set()
        for a, b, c in faces:
            edge_set.add((min(a, b), max(a, b)))
            edge_set.add((min(b, c), max(b, c)))
            edge_set.add((min(c, a), max(c, a)))
        self.edges = np.asarray(sorted(edge_set), dtype=int)
        self.rest_len = np.linalg.norm(
            self.X[self.edges[:, 1]] - self.X[self.edges[:, 0]], axis=1
        )

        # Bending springs: pairs of opposite vertices across internal edges
        self.bend_edges, self.bend_rest_len = self._build_bending_springs(faces)

        # Triangle rest data for per-vertex deformation gradient
        self.tri_t1 = np.zeros((self.n_faces, 3), dtype=float)
        self.tri_t2 = np.zeros((self.n_faces, 3), dtype=float)
        self.tri_Dm_inv = np.zeros((self.n_faces, 2, 2), dtype=float)
        self._build_triangle_rest()

        # Neighbor triangles for per-vertex averaging
        self.vertex_tri_count = np.zeros(self.N, dtype=int)
        for tri_idx, (a, b, c) in enumerate(faces):
            self.vertex_tri_count[a] += 1
            self.vertex_tri_count[b] += 1
            self.vertex_tri_count[c] += 1

        # Obstacles (frozen scene geometry, hard-coded meters)
        self.ground_R = np.eye(3)
        self.ground_t = np.array([0.0, 0.0, -0.05])
        self.ground_h = np.array([2.4, 2.0, 0.05])

        self.incline_R = self._quat_to_rot(np.array([0.0, 0.300706, 0.0, 0.953717]))
        self.incline_t = np.array([0.0, 0.0, 0.8])
        self.incline_h = np.array([2.2, 1.1, 0.08])

        # Material parameters
        self.k_stretch = 2.0e3
        self.k_bend = 5.0
        self.c_damp = 0.5
        self.c_bend_damp = 0.2
        self.k_contact = 5.0e4
        self.mu = 0.6
        self.restitution = 0.0
        self.dt = 1.0e-4

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def advance_to(self, target_time: float) -> StateFrame:
        if target_time <= self.time + 1e-12:
            return self._make_frame(self.time)

        while self.time < target_time - 1e-12:
            dt_step = min(self.dt, target_time - self.time)
            self._step(dt_step)
            self.time += dt_step

        return self._make_frame(self.time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "mass_spring_cloth",
            "substep_dt": self.dt,
            "vertices": int(self.N),
            "faces": int(self.n_faces),
            "stretch_springs": int(self.edges.shape[0]),
            "bend_springs": int(self.bend_edges.shape[0]),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        X = self.X
        V = self.V
        F = np.zeros_like(X)

        # Gravity
        F += self.mass[:, None] * self.gravity

        # Stretch springs
        dx = X[self.edges[:, 1]] - X[self.edges[:, 0]]
        dist = np.linalg.norm(dx, axis=1)
        dist = np.maximum(dist, 1e-12)
        strain = dist - self.rest_len
        f_mag = self.k_stretch * strain

        dv = V[self.edges[:, 1]] - V[self.edges[:, 0]]
        v_rel = np.einsum("ij,ij->i", dv, dx / dist[:, None])
        f_total = (f_mag + self.c_damp * v_rel) / dist
        f_vec = f_total[:, None] * dx

        np.add.at(F, self.edges[:, 0], -f_vec)
        np.add.at(F, self.edges[:, 1], f_vec)

        # Bending springs
        if self.bend_edges.size > 0:
            dx_b = X[self.bend_edges[:, 1]] - X[self.bend_edges[:, 0]]
            dist_b = np.linalg.norm(dx_b, axis=1)
            dist_b = np.maximum(dist_b, 1e-12)
            strain_b = dist_b - self.bend_rest_len
            f_mag_b = self.k_bend * strain_b

            dv_b = V[self.bend_edges[:, 1]] - V[self.bend_edges[:, 0]]
            v_rel_b = np.einsum("ij,ij->i", dv_b, dx_b / dist_b[:, None])
            f_total_b = (f_mag_b + self.c_bend_damp * v_rel_b) / dist_b
            f_vec_b = f_total_b[:, None] * dx_b

            np.add.at(F, self.bend_edges[:, 0], -f_vec_b)
            np.add.at(F, self.bend_edges[:, 1], f_vec_b)

        # Contact against union of ground + incline
        phi, normal = self._signed_distance(X)
        near = phi < self.margin
        if np.any(near):
            penetration = self.margin - phi[near]
            f_n_mag = self.k_contact * penetration
            f_n = f_n_mag[:, None] * normal[near]

            # Normal contact damping (critical damping per vertex)
            vn = np.einsum("ij,ij->i", V[near], normal[near])
            c_n = 2.0 * np.sqrt(self.mass[near] * self.k_contact)
            f_damp = -c_n[:, None] * vn[:, None] * normal[near]

            # Tangential Coulomb friction
            Vt = V[near] - vn[:, None] * normal[near]
            vt_mag = np.linalg.norm(Vt, axis=1)
            f_t = -self.mu * f_n_mag[:, None] * Vt / (vt_mag[:, None] + 1e-6)

            F[near] += f_n + f_damp + f_t

            # Reflect normal velocity for penetrating vertices to avoid tunneling
            penetrated = phi[near] < 0.0
            if np.any(penetrated):
                near_idx = np.nonzero(near)[0]
                idx = near_idx[penetrated]
                vn_p = np.einsum("ij,ij->i", V[idx], normal[idx])
                V[idx] = V[idx] - (1.0 + self.restitution) * vn_p[:, None] * normal[idx]

        # Semi-implicit damping-free integration of internal spring forces,
        # with contact forces treated explicitly.  Stable for the chosen dt.
        a = F / self.mass[:, None]
        V_new = V + a * dt
        X_new = X + V_new * dt

        self.V = V_new
        self.X = X_new

    def _make_frame(self, t: float) -> StateFrame:
        pos = self.X.copy()
        vel = self.V.copy()

        # Per-vertex deformation gradient (2x2) averaged from triangles
        F = self._compute_deformation_gradient(pos)

        # Signed distance to obstacle union
        phi, _ = self._signed_distance(pos)

        # External force = gravity + contact/friction at this state
        ext = self._compute_external_force(pos, vel)

        return StateFrame(
            time=float(t),
            position=pos,
            velocity=vel,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass_quantity.copy(),
                "boundary_vertex": self.boundary.astype(np.int32),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.array([0.0], dtype=float),
                "external_force": ext,
                "contact_signed_distance": phi,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )

    def _compute_external_force(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        F = self.mass[:, None] * self.gravity
        phi, normal = self._signed_distance(X)
        near = phi < self.margin
        if not np.any(near):
            return F

        penetration = self.margin - phi[near]
        f_n_mag = self.k_contact * penetration
        f_n = f_n_mag[:, None] * normal[near]

        vn = np.einsum("ij,ij->i", V[near], normal[near])
        c_n = 2.0 * np.sqrt(self.mass[near] * self.k_contact)
        f_damp = -c_n[:, None] * vn[:, None] * normal[near]

        Vt = V[near] - vn[:, None] * normal[near]
        vt_mag = np.linalg.norm(Vt, axis=1)
        f_t = -self.mu * f_n_mag[:, None] * Vt / (vt_mag[:, None] + 1e-6)

        F[near] += f_n + f_damp + f_t
        return F

    # ------------------------------------------------------------------
    # Geometry / contact
    # ------------------------------------------------------------------
    def _signed_distance(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        phi1, n1 = self._box_phi(X, self.ground_R, self.ground_t, self.ground_h)
        phi2, n2 = self._box_phi(X, self.incline_R, self.incline_t, self.incline_h)
        choose = (phi1 < phi2)[:, None]
        phi = np.where(choose[:, 0], phi1, phi2)
        normal = np.where(choose, n1, n2)
        return phi, normal

    def _box_phi(
        self,
        X: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        h: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Vectorised signed distance + outward normal for an oriented box.
        pl = (X - t[None, :]) @ R  # (N,3) in local frame
        q = np.abs(pl) - h[None, :]

        inside_mask = np.all(q <= 0.0, axis=1)
        phi = np.empty(X.shape[0], dtype=float)
        normal = np.empty_like(X)

        # Outside branch
        out = ~inside_mask
        if np.any(out):
            qpos = np.maximum(q[out], 0.0)
            dist = np.linalg.norm(qpos, axis=1)
            n_local = np.sign(pl[out]) * qpos / (dist[:, None] + 1e-12)
            phi[out] = dist
            normal[out] = n_local @ R.T

        # Inside branch
        if np.any(inside_mask):
            q_in = q[inside_mask]
            pl_in = pl[inside_mask]
            axis = np.argmax(q_in, axis=1)
            signs = np.copysign(1.0, pl_in[np.arange(pl_in.shape[0]), axis])
            n_local = np.zeros_like(q_in)
            n_local[np.arange(n_local.shape[0]), axis] = signs
            phi[inside_mask] = q_in[np.arange(q_in.shape[0]), axis]
            normal[inside_mask] = n_local @ R.T

        return phi, normal

    @staticmethod
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        x, y, z, w = q / np.linalg.norm(q)
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ]
        )

    # ------------------------------------------------------------------
    # Triangle / spring helpers
    # ------------------------------------------------------------------
    def _build_bending_springs(self, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Map each edge to the triangles it belongs to and their opposite vertices.
        edge_tris: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for tri_idx, (a, b, c) in enumerate(faces):
            edges = [(a, b, c), (b, c, a), (c, a, b)]
            for u, v, opp in edges:
                key = (min(u, v), max(u, v))
                edge_tris.setdefault(key, []).append((tri_idx, opp))

        pairs = []
        lengths = []
        for (u, v), tri_list in edge_tris.items():
            if len(tri_list) == 2:
                opp_a = tri_list[0][1]
                opp_b = tri_list[1][1]
                if opp_a != opp_b:
                    pairs.append((opp_a, opp_b))
                    diff = self.X[opp_b] - self.X[opp_a]
                    lengths.append(np.linalg.norm(diff))

        if pairs:
            return np.asarray(pairs, dtype=int), np.asarray(lengths, dtype=float)
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)

    def _build_triangle_rest(self) -> None:
        p = self.X
        faces = self.faces
        for i, (a, b, c) in enumerate(faces):
            e1 = p[b] - p[a]
            e2 = p[c] - p[a]
            t1 = e1 / (np.linalg.norm(e1) + 1e-12)
            e2_dot_t1 = np.dot(e2, t1)
            t2 = e2 - e2_dot_t1 * t1
            t2 /= np.linalg.norm(t2) + 1e-12

            # Material edge matrix in (t1,t2) coordinates: [e1,e2]
            Dm = np.array([[np.dot(e1, t1), np.dot(e2, t1)], [0.0, np.dot(e2, t2)]])
            self.tri_t1[i] = t1
            self.tri_t2[i] = t2
            self.tri_Dm_inv[i] = np.linalg.inv(Dm + 1e-12 * np.eye(2))

    def _compute_deformation_gradient(self, X: np.ndarray) -> np.ndarray:
        # Per-triangle 2x2 deformation gradient, then averaged to vertices.
        F_tri = np.zeros((self.n_faces, 2, 2), dtype=float)
        p = X
        faces = self.faces

        for i, (a, b, c) in enumerate(faces):
            e1 = p[b] - p[a]
            e2 = p[c] - p[a]
            t1 = self.tri_t1[i]
            t2 = self.tri_t2[i]
            Ds = np.array(
                [
                    [np.dot(e1, t1), np.dot(e2, t1)],
                    [np.dot(e1, t2), np.dot(e2, t2)],
                ]
            )
            F_tri[i] = Ds @ self.tri_Dm_inv[i]

        F_vert = np.zeros((self.N, 2, 2), dtype=float)
        for i, (a, b, c) in enumerate(faces):
            F_vert[a] += F_tri[i]
            F_vert[b] += F_tri[i]
            F_vert[c] += F_tri[i]

        F_vert /= np.maximum(self.vertex_tri_count[:, None, None], 1)
        return F_vert
