"""Physically-based solver for the soft spine cantilever benchmark.

Implements a 3-D mass-spring truss on the surface mesh, advances it with
explicit time stepping, and exports the required trajectory fields.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from worldsolver.contracts import StateFrame


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = seed
        self.scene = scene

        # Load the frozen configuration for physical parameters.
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        if config_path.is_file():
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = scene.config

        self.gravity = np.array(self.config["gravity"], dtype=float)
        self.density = float(self.config["material"]["density"])
        self.young = float(self.config["material"]["youngs_modulus_pa"])
        self.poisson = float(self.config["material"]["poisson_ratio"])

        # Initial state from the scene.
        init = scene.initial_state()
        self.pos0 = init.position.copy().astype(float)
        self.vel = init.velocity.copy().astype(float)
        if init.mass is not None:
            self.mass = init.mass.copy().astype(float)
        else:
            self.mass = np.ones(len(self.pos0), dtype=float)
        n = len(self.pos0)

        # Fixed material region: left root block.
        fixed_attr = init.attributes.get("fixed")
        if fixed_attr is not None:
            self.fixed_mask = np.asarray(fixed_attr, dtype=bool)
        else:
            x_max = float(self.config["boundary_conditions"]["fixed_region"]["x_max"])
            self.fixed_mask = self.pos0[:, 0] <= x_max

        self.current_time = float(init.time)

        # Reference force scale: total weight, always positive.
        self.total_mass = float(np.sum(self.mass))
        self.weight_vector = self.total_mass * self.gravity
        self.reference_force = float(np.linalg.norm(self.weight_vector))
        if self.reference_force <= 0.0:
            self.reference_force = 1.0

        # Parse the mesh asset.
        self.vertices, self.faces = self._load_obj()
        assert len(self.vertices) == n

        # Build the elastic truss: intra-component face edges + diagonals,
        # plus inter-component springs between adjacent spine segments.
        self._build_elastic_network()

        # Element (per original quad) reference geometry for deformation gradient.
        self._build_element_deformation_gradient()

        # Time-stepping parameters.
        self.dt = 2.0e-5  # seconds
        self.zeta = 0.25  # spring damping ratio
        self.c_global = 0.8  # global velocity damping (N s / m)

        # Precompute edge unit vectors at rest not needed, just rest lengths.
        self.L0 = np.linalg.norm(
            self.pos0[self.edges[:, 0]] - self.pos0[self.edges[:, 1]], axis=1
        )

        # Edge stiffness: E * A / L0, where A is a representative cross-section.
        self.k = self._edge_stiffness()

        # Damping coefficient per edge (Rayleigh-like, critical scaling).
        m_edge = (self.mass[self.edges[:, 0]] + self.mass[self.edges[:, 1]]) * 0.5
        self.c = 2.0 * self.zeta * np.sqrt(m_edge * self.k)

        # Safety clamping for pathological short edges.
        min_len = np.min(self.L0)
        if min_len <= 0.0:
            raise ValueError("Zero-length edge detected in reference mesh")

    def _load_obj(self) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        path = Path(__file__).resolve().parent.parent / "assets" / "spine_arm.obj"
        vertices: List[List[float]] = []
        faces: List[Tuple[int, int, int, int]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                if len(idx) == 4:
                    faces.append((idx[0], idx[1], idx[2], idx[3]))
                elif len(idx) == 3:
                    faces.append((idx[0], idx[1], idx[2], idx[2]))
        return np.asarray(vertices, dtype=float), faces

    def _build_elastic_network(self) -> None:
        # Intra-component edges from the original quad faces.
        intra_edges: set[frozenset] = set()
        for f in self.faces:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[3]), (f[3], f[0])):
                if a != b:
                    intra_edges.add(frozenset({a, b}))
            # Add both diagonals to give each face in-plane shear rigidity.
            for a, b in ((f[0], f[2]), (f[1], f[3])):
                if a != b:
                    intra_edges.add(frozenset({a, b}))

        # Determine connected components from face edges only.
        uf = _UnionFind(len(self.vertices))
        for f in self.faces:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[3]), (f[3], f[0])):
                uf.union(a, b)

        comp_of = np.array([uf.find(i) for i in range(len(self.vertices))])
        unique_comps = np.unique(comp_of)
        centroids = {cid: np.mean(self.vertices[comp_of == cid], axis=0) for cid in unique_comps}
        sorted_comps = sorted(unique_comps, key=lambda c: centroids[c][0])

        # Build a map from vertex to component.
        comp_index = {cid: idx for idx, cid in enumerate(sorted_comps)}
        self.vertex_comp = np.array([comp_index[comp_of[i]] for i in range(len(self.vertices))])
        self.components = sorted_comps

        # Component cross-sectional area (in y-z plane, roughly the loading plane).
        self.comp_area: np.ndarray = np.zeros(len(sorted_comps), dtype=float)
        for idx, cid in enumerate(sorted_comps):
            mask = comp_of == cid
            ys = self.vertices[mask, 1]
            zs = self.vertices[mask, 2]
            self.comp_area[idx] = max(0.0, (ys.max() - ys.min()) * (zs.max() - zs.min()))

        # Inter-component edges between adjacent spine segments.
        dmax = 0.12  # meters; tuned to connect overlapping box/neck pairs.
        for i in range(len(sorted_comps) - 1):
            v_i = np.where(self.vertex_comp == i)[0]
            v_j = np.where(self.vertex_comp == i + 1)[0]
            if len(v_i) == 0 or len(v_j) == 0:
                continue
            # Ensure connectivity: always keep the closest pair.
            diff = self.pos0[v_i][:, None, :] - self.pos0[v_j][None, :, :]  # type: ignore[index]
            dists = np.linalg.norm(diff, axis=2)
            min_i, min_j = np.unravel_index(int(np.argmin(dists)), dists.shape)
            intra_edges.add(frozenset({int(v_i[min_i]), int(v_j[min_j])}))
            # Add all pairs within the tolerance for distributed coupling.
            close = np.argwhere(dists <= dmax)
            for a_idx, b_idx in close:
                a = int(v_i[a_idx])
                b = int(v_j[b_idx])
                intra_edges.add(frozenset({a, b}))

        # Convert to arrays.
        self.edges = np.array([[min(e), max(e)] for e in intra_edges], dtype=int)

    def _edge_stiffness(self) -> np.ndarray:
        areas = np.zeros(len(self.edges), dtype=float)
        for idx, (i, j) in enumerate(self.edges):
            ci = self.vertex_comp[i]
            cj = self.vertex_comp[j]
            if ci == cj:
                a = self.comp_area[ci]
            else:
                a = min(self.comp_area[ci], self.comp_area[cj])
            # Guard against tiny/numerical area.
            if a <= 0.0:
                a = np.median(self.comp_area[self.comp_area > 0.0])
            areas[idx] = a
        return self.young * areas / self.L0

    def _build_element_deformation_gradient(self) -> None:
        # Per-quad reference basis.
        p0 = self.pos0[self.faces[:, 0]]
        p1 = self.pos0[self.faces[:, 1]]
        p3 = self.pos0[self.faces[:, 3]]
        e1 = p1 - p0
        e2 = p3 - p0
        n = np.cross(e1, e2)
        n_len = np.linalg.norm(n, axis=1)
        n = n / np.maximum(n_len[:, None], 1e-12)
        # Thickness scale comparable to in-plane edge length.
        self.thickness = 0.5 * (np.linalg.norm(e1, axis=1) + np.linalg.norm(e2, axis=1))
        e3 = n * self.thickness[:, None]
        D_ref = np.stack([e1, e2, e3], axis=2)  # Ne x 3 x 3
        self.D_ref_inv = np.linalg.inv(D_ref)

    def _compute_element_F(self, pos: np.ndarray) -> np.ndarray:
        p0 = pos[self.faces[:, 0]]
        p1 = pos[self.faces[:, 1]]
        p3 = pos[self.faces[:, 3]]
        e1 = p1 - p0
        e2 = p3 - p0
        n = np.cross(e1, e2)
        n_len = np.linalg.norm(n, axis=1)
        n = n / np.maximum(n_len[:, None], 1e-12)
        e3 = n * self.thickness[:, None]
        D = np.stack([e1, e2, e3], axis=2)
        return D @ self.D_ref_inv

    def _frame(self, time: float, pos: np.ndarray, vel: np.ndarray) -> StateFrame:
        element_F = self._compute_element_F(pos)
        budget = np.zeros((1, 1, 3), dtype=float)
        budget[0, 0, :] = self.weight_vector
        return StateFrame(
            time=float(time),
            position=pos.astype(float),
            velocity=vel.astype(float),
            mass=self.mass.astype(float),
            attributes={"mass_quantity": self.mass.astype(float)},
            fields={
                "budget_external_force": budget,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "element_deformation_gradient": element_F.astype(float),
            },
            scalars={"reference_force": float(self.reference_force)},
        )

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.current_time:
            raise ValueError("advance_to target must be >= current simulation time")

        if target == self.current_time:
            return self._frame(target, self.pos0 if target == 0.0 else self.pos0, self.vel)

        elapsed = target - self.current_time
        n_steps = max(1, int(np.ceil(elapsed / self.dt)))
        h = elapsed / n_steps

        pos = self.pos0.copy()
        vel = self.vel.copy()

        for _ in range(n_steps):
            # Elastic forces on edges.
            diff = pos[self.edges[:, 0]] - pos[self.edges[:, 1]]
            dist = np.linalg.norm(diff, axis=1)
            e_unit = diff / np.maximum(dist[:, None], 1e-12)

            # Hookean spring force: positive when stretched.
            f_mag = self.k * (dist - self.L0)
            f_spring = f_mag[:, None] * e_unit

            # Internal nodal force accumulator.
            f_node = np.zeros_like(pos)
            np.add.at(f_node, self.edges[:, 0], f_spring)
            np.add.at(f_node, self.edges[:, 1], -f_spring)

            # Parallel spring damping along each edge.
            dv = vel[self.edges[:, 0]] - vel[self.edges[:, 1]]
            v_rel = np.einsum("ij,ij->i", dv, e_unit)
            f_damp = (self.c * v_rel)[:, None] * e_unit
            np.add.at(f_node, self.edges[:, 0], f_damp)
            np.add.at(f_node, self.edges[:, 1], -f_damp)

            # Body force (gravity) and global velocity damping.
            f_node += self.mass[:, None] * self.gravity[None, :]
            f_node -= self.c_global * vel

            # Symplectic Euler update.
            vel_new = vel + h * f_node / self.mass[:, None]
            pos_new = pos + h * vel_new

            # Enforce Dirichlet boundary conditions (clamped root).
            vel_new[self.fixed_mask] = 0.0
            pos_new[self.fixed_mask] = self.pos0[self.fixed_mask]

            vel = vel_new
            pos = pos_new

        self.current_time = target
        self.pos0 = pos
        self.vel = vel
        return self._frame(target, pos, vel)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "mass_spring_spine_cantilever",
            "nodes": int(len(self.pos0)),
            "edges": int(len(self.edges)),
            "elements": int(len(self.faces)),
            "dt": float(self.dt),
            "reference_force": float(self.reference_force),
        }
