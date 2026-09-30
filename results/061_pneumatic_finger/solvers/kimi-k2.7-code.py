"""Physically-based solver for the pneumatic soft finger scene.

The finger is modelled as a chain of hexahedral soft segments (base, six
chamber sections and a tip).  Each segment is tetrahedralised and integrated
with a stable compressible Neo-Hookean material.  Adjacent segments are glued
with stiff elastic springs and the base is fixed.  Internal pressure is applied
as a surface traction on the six chamber segments, ramped according to the
frozen scene parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.spatial import Delaunay

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        # Scene / config filled in initialize()
        self.config: dict = {}
        self.rest_position: np.ndarray | None = None
        self.mass: np.ndarray | None = None
        self.fixed_mask: np.ndarray | None = None
        self.time: float = 0.0
        self.position: np.ndarray | None = None
        self.velocity: np.ndarray | None = None

        # Geometry-derived structures
        self.block_vertex_indices: List[np.ndarray] = []   # per block, local 0..7 -> global
        self.block_faces: List[List[np.ndarray]] = []      # per block, list of 4-vertex faces (global)
        self.chamber_block_ids = list(range(1, 7))         # base=0, chambers=1..6, tip=7
        self.base_block_id = 0
        self.tip_block_id = 7

        # Tetrahedral elements: list of (v0, v1, v2, v3, block_id)
        self.tets: List[Tuple[int, int, int, int, int]] = []
        self.tet_ref_inv = []   # D_ref^{-1}
        self.tet_volume = []    # reference volumes

        # Attachment springs between adjacent blocks
        self.attach_springs: List[Tuple[int, int, float, float]] = []

        # Material / integration parameters
        self.rho = 1.20e3          # kg/m^3  (silicone-like)
        self.E = 2.0e5             # Pa      (soft elastomer)
        self.nu = 0.45             # -
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        self.damping = 1.2e2        # 1/s  velocity damping
        self.k_attach = 1.0e6      # N/m  inter-block attachment spring
        self.dt_sub = 1.0e-4       # s    physics sub-step
        self.gravity = np.array([0.0, 0.0, -9.81])

        # Pressure schedule
        self.p_target = 0.0
        self.p_ramp_start = 0.0
        self.p_ramp_end = 0.0
        self.p_hold = 0.0

    # ------------------------------------------------------------------ #
    # Solver protocol
    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        self.config = scene.config if hasattr(scene, "config") else {}
        pressure_cfg = self.config.get("pressure", {})
        self.p_target = float(pressure_cfg.get("chamber_pressure_pa", 18000.0))
        self.p_ramp_start = float(pressure_cfg.get("ramp_start_s", 0.4))
        self.p_ramp_end = float(pressure_cfg.get("ramp_end_s", 2.2))
        self.p_hold = float(pressure_cfg.get("hold_until_s", 4.0))

        # Start from the scene's own initial state for consistency.
        frame0 = scene.initial_state()
        self.rest_position = np.asarray(frame0.position, dtype=float).copy()
        n = len(self.rest_position)
        self.position = self.rest_position.copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.time = 0.0

        fixed_attr = frame0.attributes.get("fixed_base")
        if fixed_attr is None:
            # Fall back to the frozen base support region.
            base_range = self.config.get("initial_conditions", {}).get("base_support_region_x", [0.0, 0.03])
            fixed_attr = (self.rest_position[:, 0] <= base_range[1]).astype(float)
        self.fixed_mask = np.asarray(fixed_attr, dtype=bool)

        # Parse the wavefront asset and group into 8 hexahedral blocks.
        asset_path = Path(__file__).parent.parent / "assets" / "pneumatic_finger.obj"
        if not asset_path.is_file():
            raise FileNotFoundError(f"Scene asset not found: {asset_path}")
        verts, faces = self._parse_obj(asset_path)
        if len(verts) % 8 != 0:
            raise ValueError("Vertex count is not a multiple of 8; cannot split into hexahedra")
        num_blocks = len(verts) // 8
        self.block_vertex_indices = [np.arange(i * 8, (i + 1) * 8) for i in range(num_blocks)]

        # Group faces by block (6 faces per block).
        self.block_faces = []
        for b in range(num_blocks):
            start = b * 6
            block_faces = [np.asarray(f, dtype=int) for f in faces[start : start + 6]]
            self.block_faces.append(block_faces)

        # Tetrahedralise each block.
        self.tets = []
        self.tet_ref_inv = []
        self.tet_volume = []
        for b, local_to_global in enumerate(self.block_vertex_indices):
            block_verts = self.rest_position[local_to_global]
            try:
                tri = Delaunay(block_verts)
            except Exception as exc:
                raise RuntimeError(f"Delaunay failed for block {b}") from exc
            for simplex in tri.simplices:
                v0, v1, v2, v3 = local_to_global[simplex]
                X = self.rest_position[[v0, v1, v2, v3]]
                D = np.column_stack((X[0] - X[3], X[1] - X[3], X[2] - X[3]))
                vol = abs(np.linalg.det(D)) / 6.0
                if vol < 1e-12:
                    # Degenerate tetrahedron; skip rather than crash.
                    continue
                self.tets.append((v0, v1, v2, v3, b))
                self.tet_ref_inv.append(np.linalg.inv(D))
                self.tet_volume.append(vol)

        # Per-vertex mass from material density.
        vertex_volume = np.zeros(n)
        for (v0, v1, v2, v3, b), vol in zip(self.tets, self.tet_volume):
            for v in (v0, v1, v2, v3):
                vertex_volume[v] += vol / 4.0
        self.mass = self.rho * vertex_volume
        self.mass = np.maximum(self.mass, 1e-8)

        # Build inter-block attachment springs between adjacent blocks.
        self._build_attachment_springs()

        # Seed is not used; solver is deterministic.
        _ = seed

    def advance_to(self, time: float) -> StateFrame:
        if time <= self.time:
            return self._make_frame(self.time)

        dt_total = time - self.time
        n_steps = max(1, int(np.ceil(dt_total / self.dt_sub)))
        dt = dt_total / n_steps

        for _ in range(n_steps):
            self._step(dt)
        self.time = time
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "substep_dt": self.dt_sub,
            "num_vertices": len(self.position) if self.position is not None else 0,
            "num_tets": len(self.tets),
            "mu": self.mu,
            "lambda": self.lam,
            "reference_force_n": 1.0,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _parse_obj(self, path: Path) -> Tuple[np.ndarray, List[List[int]]]:
        verts = []
        faces: List[List[int]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f":
                # Only quad faces are expected; triangulate if a triangle is found.
                indices = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                faces.append(indices)
        return np.asarray(verts, dtype=float), faces

    def _build_attachment_springs(self) -> None:
        """Connect vertices of adjacent blocks at their common interface."""
        blocks = self.block_vertex_indices
        for i in range(len(blocks)):
            for j in range(i + 1, len(blocks)):
                pairs = self._interface_pairs(i, j)
                for a, b in pairs:
                    rest = np.linalg.norm(self.rest_position[a] - self.rest_position[b])
                    self.attach_springs.append((int(a), int(b), rest, self.k_attach))

    def _interface_pairs(self, bi: int, bj: int) -> List[Tuple[int, int]]:
        """Return pairs of vertices belonging to a shared interface between two blocks."""
        vi = self.block_vertex_indices[bi]
        vj = self.block_vertex_indices[bj]
        Xi = self.rest_position[vi]
        Xj = self.rest_position[vj]

        # Determine the most-aligned pair of faces (share a coordinate plane).
        extents_i = np.ptp(Xi, axis=0)
        extents_j = np.ptp(Xj, axis=0)
        # Candidate separating directions: coordinate axes.
        best_axis = -1
        best_score = -1.0
        for axis in range(3):
            # Check if blocks are adjacent along this axis and overlap in the other two.
            if extents_i[axis] < 1e-9 or extents_j[axis] < 1e-9:
                continue
            min_i, max_i = Xi[:, axis].min(), Xi[:, axis].max()
            min_j, max_j = Xj[:, axis].min(), Xj[:, axis].max()
            gap = min(max_i, max_j) - max(min_i, min_j)
            if gap < -1e-6:
                # Overlap in this direction, not a separating plane.
                continue
            # Overlap in the other two directions?
            other = [k for k in range(3) if k != axis]
            overlap_other = True
            for k in other:
                o_min = max(Xi[:, k].min(), Xj[:, k].min())
                o_max = min(Xi[:, k].max(), Xj[:, k].max())
                if o_max - o_min < 1e-6:
                    overlap_other = False
                    break
            if overlap_other:
                # Choose the smallest gap as the best interface.
                if best_axis == -1 or gap < best_score:
                    best_axis = axis
                    best_score = gap

        if best_axis == -1:
            return []

        axis = best_axis
        # Select the face of each block with the extreme coordinate value.
        val_i = Xi[:, axis].max() if (Xi[:, axis].mean() < Xj[:, axis].mean()) else Xi[:, axis].min()
        val_j = Xj[:, axis].max() if val_i > Xj[:, axis].mean() else Xj[:, axis].min()

        face_i = [v for v in vi if abs(self.rest_position[v, axis] - val_i) < 1e-6]
        face_j = [v for v in vj if abs(self.rest_position[v, axis] - val_j) < 1e-6]

        if len(face_i) != 4 or len(face_j) != 4:
            return []

        # Pair closest vertices.
        pairs: List[Tuple[int, int]] = []
        used_j = set()
        for a in face_i:
            best_b = None
            best_d = np.inf
            for b in face_j:
                if b in used_j:
                    continue
                d = np.linalg.norm(self.rest_position[a] - self.rest_position[b])
                if d < best_d:
                    best_d = d
                    best_b = b
            if best_b is not None:
                pairs.append((a, best_b))
                used_j.add(best_b)
        return pairs

    def _pressure(self, t: float) -> float:
        if t < self.p_ramp_start:
            return 0.0
        if t <= self.p_ramp_end:
            return self.p_target * (t - self.p_ramp_start) / (self.p_ramp_end - self.p_ramp_start)
        return self.p_target if t <= self.p_hold else 0.0

    def _step(self, dt: float) -> None:
        pos = self.position
        vel = self.velocity
        n = pos.shape[0]
        force = np.zeros((n, 3), dtype=float)

        # Gravity.
        force += self.mass[:, None] * self.gravity

        # Elastic forces from tetrahedral elements.
        for (v0, v1, v2, v3, b), Dinv, Vref in zip(
            self.tets, self.tet_ref_inv, self.tet_volume
        ):
            X = pos[[v0, v1, v2, v3]]
            D = np.column_stack((X[0] - X[3], X[1] - X[3], X[2] - X[3]))
            F = D @ Dinv
            # Stable Neo-Hookean first Piola stress.
            try:
                Finv = np.linalg.inv(F)
            except np.linalg.LinAlgError:
                continue
            J = max(np.linalg.det(F), 1e-6)
            P = self.mu * (F - Finv.T) + self.lam * np.log(J) * Finv.T
            # Nodal forces: -V * P * G^T, where G = Dinv.
            GT = Dinv.T
            f0 = -Vref * P @ GT[:, 0]
            f1 = -Vref * P @ GT[:, 1]
            f2 = -Vref * P @ GT[:, 2]
            f3 = -(f0 + f1 + f2)
            force[v0] += f0
            force[v1] += f1
            force[v2] += f2
            force[v3] += f3

        # Attachment springs between blocks.
        for a, b, rest, k in self.attach_springs:
            dvec = pos[b] - pos[a]
            dist = np.linalg.norm(dvec)
            if dist < 1e-12:
                continue
            f = k * (dist - rest) * dvec / dist
            force[a] += f
            force[b] -= f

        # Pressure traction on chamber blocks.
        p = self._pressure(self.time)
        if p != 0.0:
            for b in self.chamber_block_ids:
                for face in self.block_faces[b]:
                    fvec = self._face_area_vector(pos[face])
                    # outward area vector; pressure pushes outward from the chamber.
                    f_pressure = p * fvec / 4.0
                    force[face] += f_pressure

        # Integrate with semi-implicit damping.
        # v' = (f/m - d v)  =>  v_new = (v_old + dt * f/m) / (1 + d*dt)
        acc = force / self.mass[:, None]
        vel_new = (vel + dt * acc) / (1.0 + self.damping * dt)
        pos_new = pos + dt * vel_new

        # Enforce fixed base.
        pos_new[self.fixed_mask] = self.rest_position[self.fixed_mask]
        vel_new[self.fixed_mask] = 0.0

        self.position = pos_new
        self.velocity = vel_new

    def _face_area_vector(self, quad: np.ndarray) -> np.ndarray:
        """Return the oriented area vector of a planar quadrilateral."""
        # Split quad 0-1-2-3 into two triangles sharing the diagonal 0-2.
        v0, v1, v2, v3 = quad
        a = 0.5 * np.cross(self.position[v1] - self.position[v0], self.position[v2] - self.position[v0])
        b = 0.5 * np.cross(self.position[v3] - self.position[v2], self.position[v0] - self.position[v2])
        return a + b

    def _make_frame(self, t: float) -> StateFrame:
        # Deformation gradient for each element at the current state.
        F_list = []
        for (v0, v1, v2, v3, b), Dinv in zip(self.tets, self.tet_ref_inv):
            X = self.position[[v0, v1, v2, v3]]
            D = np.column_stack((X[0] - X[3], X[1] - X[3], X[2] - X[3]))
            F = D @ Dinv
            F_list.append(F)
        element_F = np.asarray(F_list) if F_list else np.empty((0, 3, 3), dtype=float)

        # Budget external forces: gravity + pressure.
        total_gravity = np.sum(self.mass[:, None] * self.gravity, axis=0)
        total_pressure = np.zeros(3)
        p = self._pressure(t)
        if p != 0.0:
            for b in self.chamber_block_ids:
                for face in self.block_faces[b]:
                    fvec = self._face_area_vector(face)
                    total_pressure += p * fvec
        budget = np.zeros((1, 2, 3), dtype=float)
        budget[0, 0, :] = total_gravity
        budget[0, 1, :] = total_pressure

        mass_quantity = self.mass.copy()

        frame = StateFrame(
            time=float(t),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            attributes={
                "fixed_base": self.fixed_mask.astype(float),
                "rest_position": self.rest_position.copy(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "element_deformation_gradient": element_F,
                "budget_external_force": budget,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "chamber_pressure_pa": float(p),
                "reference_force": 1.0,
            },
        )
        frame.validate()
        return frame
