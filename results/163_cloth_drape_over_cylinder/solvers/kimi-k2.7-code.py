"""Solver Agent-owned cloth simulation.

Implements a mass-spring thin sheet with in-plane edge springs, bending springs,
gravity, and penalty contact against the fixed horizontal cylinder and ground.
The model is integrated explicitly with adaptive substeps.
"""
from __future__ import annotations

import json
import math

import numpy as np
import scipy.sparse as sp

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.seed = 0

        # Per-vertex state
        self.position: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        self.mass: np.ndarray | None = None
        self.n_vertices = 0

        # Mesh topology
        self.faces: np.ndarray | None = None
        self.n_faces = 0
        self.edges: np.ndarray | None = None
        self.edge_rest: np.ndarray | None = None
        self.edge_inc: sp.csr_matrix | None = None
        self.bend: np.ndarray | None = None
        self.bend_rest: np.ndarray | None = None
        self.bend_inc: sp.csr_matrix | None = None

        # Material / simulation parameters
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.external_force: np.ndarray | None = None
        self.thickness = 0.015
        self.contact_scale = 0.015
        self.cylinder_center = np.array([0.0, 0.0, 0.7])
        self.cylinder_radius = 0.52
        self.cylinder_axis = np.array([1.0, 0.0, 0.0])

        # Spring parameters
        self.k_edge = 300.0
        self.c_edge = 0.3
        self.k_bend = 5.0
        self.c_bend = 0.05
        self.k_contact = 1.0e4
        self.mu_contact = 0.3
        self.c_global = 0.05

        # Time bookkeeping
        self.time = 0.0
        self._last_time = 0.0
        self._min_substep = 2.0e-5
        self._cumulative_mass_source = np.zeros(1, dtype=float)

        # Cached initial data
        self._initial_position: np.ndarray | None = None
        self._boundary_vertex: np.ndarray | None = None

    def diagnostics(self) -> dict[str, object]:
        return {}

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed

        frame = scene.initial_state()
        self.position = np.asarray(frame.position, dtype=float).copy()
        self.velocity = np.asarray(frame.velocity, dtype=float).copy()
        self.mass = np.asarray(frame.mass, dtype=float).copy()
        self.n_vertices = len(self.position)

        self.thickness = float(frame.scalars.get("contact_length_scale", 0.015))
        self.contact_scale = self.thickness

        self._boundary_vertex = np.asarray(
            frame.attributes.get("boundary_vertex", np.zeros(self.n_vertices, dtype=np.int32)),
            dtype=np.int32,
        )

        # Load cloth topology (triangles) from the asset.
        try:
            with open("assets/cloth_topology.json", "r", encoding="utf-8") as f:
                topo = json.load(f)
            self.faces = np.asarray(topo["faces"], dtype=np.int32)
        except Exception:
            self.faces = self._build_grid_faces(self.position)
        self.n_faces = len(self.faces)

        # Build edge and bending data structures.
        self._build_edges_and_bending(self.position)

        # Gravity from configuration.
        config = getattr(scene, "config", {})
        if "gravity" in config:
            self.gravity = np.array(config["gravity"], dtype=float)
        self.external_force = self.mass[:, None] * self.gravity[None, :]

        # Cylinder geometry from the frozen scene.
        self.cylinder_center = np.array([0.0, 0.0, 0.7])
        self.cylinder_radius = 0.52
        self.cylinder_axis = np.array([1.0, 0.0, 0.0])

        self._cumulative_mass_source = np.zeros(1, dtype=float)
        self.time = float(frame.time) if frame.time is not None else 0.0
        self._last_time = self.time
        self._initial_position = self.position.copy()

    def _build_grid_faces(self, position: np.ndarray) -> np.ndarray:
        """Fallback regular-grid triangulation if the topology asset is missing."""
        n = int(round(len(position) ** 0.5))
        faces = []
        for j in range(n - 1):
            for i in range(n - 1):
                a = j * n + i
                b = a + 1
                c = a + n
                d = c + 1
                faces.append([a, b, d])
                faces.append([a, d, c])
        return np.asarray(faces, dtype=np.int32)

    def _build_edges_and_bending(self, position: np.ndarray) -> None:
        """Create unique edge list, incidence matrix, and bending-opposite pairs."""
        edge_info: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for fi, tri in enumerate(self.faces):
            # Each edge (a,b) has opposite vertex c.
            for a, b, c in [(tri[0], tri[1], tri[2]),
                              (tri[1], tri[2], tri[0]),
                              (tri[2], tri[0], tri[1])]:
                key = (min(a, b), max(a, b))
                edge_info.setdefault(key, []).append((a, b, c))

        # Unique edges.  Keep an arbitrary orientation per edge.
        edges = []
        bend_pairs = []
        for key, info in edge_info.items():
            a, b = key
            edges.append((a, b))
            if len(info) == 2:
                # Two opposite vertices, one from each adjacent face.
                _, _, opp1 = info[0]
                _, _, opp2 = info[1]
                bend_pairs.append((opp1, opp2))

        self.edges = np.asarray(edges, dtype=np.int32)
        self.edge_rest = np.linalg.norm(
            position[self.edges[:, 0]] - position[self.edges[:, 1]], axis=1
        )

        # Incidence matrix for edge springs: maps edge forces to vertices.
        row = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        col = np.tile(np.arange(len(self.edges)), 2)
        vals = np.concatenate([np.ones(len(self.edges)), -np.ones(len(self.edges))])
        self.edge_inc = sp.csr_matrix(
            (vals, (row, col)), shape=(self.n_vertices, len(self.edges))
        )

        # Bending springs between opposite vertices of interior edges.
        if bend_pairs:
            self.bend = np.asarray(bend_pairs, dtype=np.int32)
            self.bend_rest = np.linalg.norm(
                position[self.bend[:, 0]] - position[self.bend[:, 1]], axis=1
            )
            row = np.concatenate([self.bend[:, 0], self.bend[:, 1]])
            col = np.tile(np.arange(len(self.bend)), 2)
            vals = np.concatenate([np.ones(len(self.bend)), -np.ones(len(self.bend))])
            self.bend_inc = sp.csr_matrix(
                (vals, (row, col)), shape=(self.n_vertices, len(self.bend))
            )
        else:
            self.bend = np.zeros((0, 2), dtype=np.int32)
            self.bend_rest = np.zeros(0, dtype=float)
            self.bend_inc = sp.csr_matrix((self.n_vertices, 0), dtype=float)

    def _edge_spring_forces(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Vectorized in-plane stretching forces from Hookean edge springs."""
        tail = self.edges[:, 0]
        head = self.edges[:, 1]
        p_t = position[tail]
        p_h = position[head]
        v_t = velocity[tail]
        v_h = velocity[head]
        diff = p_t - p_h
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-12)
        dir_vec = diff / dist

        strain = dist[:, 0] - self.edge_rest
        rel_v = np.sum((v_t - v_h) * dir_vec, axis=1)
        mag = self.k_edge * strain + self.c_edge * rel_v

        f_edge = mag[:, None] * dir_vec  # force on tail vertex
        return self.edge_inc @ f_edge

    def _bending_forces(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Vectorized bending resistance via springs between opposite hinge vertices."""
        if self.bend.size == 0:
            return np.zeros((self.n_vertices, 3), dtype=float)

        tail = self.bend[:, 0]
        head = self.bend[:, 1]
        p_t = position[tail]
        p_h = position[head]
        v_t = velocity[tail]
        v_h = velocity[head]
        diff = p_t - p_h
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-12)
        dir_vec = diff / dist

        strain = dist[:, 0] - self.bend_rest
        rel_v = np.sum((v_t - v_h) * dir_vec, axis=1)
        mag = self.k_bend * strain + self.c_bend * rel_v

        f_bend = mag[:, None] * dir_vec
        return self.bend_inc @ f_bend

    def _contact_forces(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Penalty contact against the cylinder and the ground."""
        n = self.n_vertices
        forces = np.zeros((n, 3), dtype=float)

        # ---- Cylinder ------------------------------------------------------
        d = position - self.cylinder_center[None, :]
        proj = (
            np.sum(d * self.cylinder_axis[None, :], axis=1, keepdims=True)
            * self.cylinder_axis[None, :]
        )
        radial = d - proj
        rad_len = np.linalg.norm(radial, axis=1, keepdims=True)
        rad_len = np.maximum(rad_len, 1e-12)
        rad_dir = radial / rad_len

        penetration = self.cylinder_radius - rad_len[:, 0]  # >0 when inside cylinder
        active = penetration > 0.0
        if active.any():
            normal = rad_dir[active]  # outward normal
            fn_mag = self.k_contact * penetration[active]
            forces[active] += fn_mag[:, None] * normal

            # Coulomb-like tangential friction.
            v_rel = velocity[active]
            v_n = np.sum(v_rel * normal, axis=1, keepdims=True) * normal
            v_t = v_rel - v_n
            v_t_norm = np.linalg.norm(v_t, axis=1, keepdims=True)
            mask = v_t_norm[:, 0] > 1e-12
            if mask.any():
                ft = -self.mu_contact * fn_mag[mask][:, None] * (v_t[mask] / v_t_norm[mask])
                forces[active][mask] += ft  # safe: forces[active] is a writable copy here

        # ---- Ground (z = 0) ------------------------------------------------
        ground_pen = -position[:, 2]
        active_g = ground_pen > 0.0
        if active_g.any():
            forces[active_g, 2] += self.k_contact * ground_pen[active_g]
            v_t = velocity[active_g, :2]
            v_t_norm = np.linalg.norm(v_t, axis=1, keepdims=True)
            mask = v_t_norm[:, 0] > 1e-12
            if mask.any():
                idx = np.nonzero(active_g)[0]
                forces[idx[mask], :2] -= (
                    self.mu_contact
                    * self.k_contact
                    * ground_pen[active_g][mask][:, None]
                    * (v_t[mask] / v_t_norm[mask])
                )

        return forces

    def _damping_forces(self, velocity: np.ndarray) -> np.ndarray:
        return -self.c_global * velocity

    def _deformation_gradients(self, position: np.ndarray) -> np.ndarray:
        """Per-vertex 2x2 in-plane deformation gradient (material coordinates)."""
        ref = self._initial_position[:, :2]
        cur = position[:, :2]
        n = self.n_vertices
        grad = np.zeros((n, 2, 2), dtype=float)
        counts = np.zeros(n, dtype=int)

        for tri in self.faces:
            i, j, k = tri
            r1 = ref[j] - ref[i]
            r2 = ref[k] - ref[i]
            Dm = np.column_stack([r1, r2])
            det = np.linalg.det(Dm)
            if abs(det) < 1e-12:
                continue
            Dm_inv = np.linalg.inv(Dm)
            c1 = cur[j] - cur[i]
            c2 = cur[k] - cur[i]
            Ds = np.column_stack([c1, c2])
            F = Ds @ Dm_inv
            for v in tri:
                grad[v] += F
                counts[v] += 1

        for v in range(n):
            if counts[v] > 0:
                grad[v] /= counts[v]
        return grad

    def _contact_signed_distance(self, position: np.ndarray) -> np.ndarray:
        """Signed geometric gap to the cylinder and the ground."""
        d = position - self.cylinder_center[None, :]
        proj = (
            np.sum(d * self.cylinder_axis[None, :], axis=1, keepdims=True)
            * self.cylinder_axis[None, :]
        )
        radial = d - proj
        rad_len = np.linalg.norm(radial, axis=1)
        cyl_sd = rad_len - self.cylinder_radius
        ground_sd = position[:, 2]
        return np.minimum(cyl_sd, ground_sd)

    def _step_explicit(self, dt: float) -> None:
        position = self.position
        velocity = self.velocity
        mass = self.mass

        f_edge = self._edge_spring_forces(position, velocity)
        f_bend = self._bending_forces(position, velocity)
        f_contact = self._contact_forces(position, velocity)
        f_damp = self._damping_forces(velocity)

        total_force = self.external_force + f_edge + f_bend + f_contact + f_damp
        acceleration = total_force / mass[:, None]

        # Clamp pathological accelerations to keep the simulation deterministic.
        max_acc = 1.0e6
        acc_norm = np.linalg.norm(acceleration, axis=1)
        if acc_norm.max() > max_acc:
            scale = np.minimum(acc_norm, max_acc) / (acc_norm + 1e-12)
            acceleration *= scale[:, None]

        velocity_new = velocity + acceleration * dt
        position_new = position + velocity_new * dt

        self.velocity = velocity_new
        self.position = position_new
        self.time += dt

    def advance_to(self, time: float) -> StateFrame:
        dt = time - self._last_time
        if dt < 0:
            raise ValueError("Time must advance")

        if dt > 0:
            n_sub = max(1, int(math.ceil(dt / self._min_substep)))
            sub_dt = dt / n_sub
            for _ in range(n_sub):
                self._step_explicit(sub_dt)
            self._last_time = time
        else:
            self.time = time

        # Required trajectory fields.
        F = self._deformation_gradients(self.position)
        sd = self._contact_signed_distance(self.position)
        ext_force = self.mass[:, None] * self.gravity[None, :]

        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n_vertices, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self._boundary_vertex.copy(),
                "deformation_gradient": F,
            },
            fields={
                "contact_signed_distance": sd,
                "cumulative_mass_source": self._cumulative_mass_source.copy(),
                "external_force": ext_force,
            },
            scalars={
                "contact_length_scale": self.contact_scale,
            },
        )
        frame.validate()
        return frame
