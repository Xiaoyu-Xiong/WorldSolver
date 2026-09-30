"""BenchmarkSolver implementation for the hanging rubber spine cantilever benchmark."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.seed = 0
        self.time = 0.0

        # Mesh / Node data
        self.rest_position = None
        self.position = None
        self.velocity = None
        self.mass = None
        self.inv_mass = None
        self.fixed = None
        self.num_nodes = 0

        # Box / Element definitions
        self.num_boxes = 21
        self.box_node_indices = []
        self.box_rest_positions = []
        self.box_rest_com = []
        self.box_mass = []

        # Constraints / Springs
        self.distance_constraints = []  # (i, j, rest_len, stiffness)
        self.tet_elements = []  # (i, j, k, l, Dm_inv, volume)
        self.overlap_constraints = []  # (box_a_nodes, box_b_nodes, rest_offsets)

        # Physics params
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.density = 1070.0
        self.youngs_modulus = 150000.0
        self.poisson_ratio = 0.4999
        self.reference_force = 1.0
        self.total_mass = 0.0

        # Substepping
        self.sub_dt = 1e-4

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.time = 0.0

        config = scene.config
        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        mat = config.get("material", {})
        self.density = float(mat.get("density", 1070.0))
        self.youngs_modulus = float(mat.get("youngs_modulus_pa", 150000.0))
        self.poisson_ratio = float(mat.get("poisson_ratio", 0.4999))

        init_frame = scene.initial_state()
        self.rest_position = np.array(init_frame.position, dtype=float, copy=True)
        self.position = np.array(init_frame.position, dtype=float, copy=True)
        self.velocity = np.zeros_like(self.position)
        self.num_nodes = len(self.position)

        # Boundary conditions (fixed region)
        x_max_fixed = float(config["boundary_conditions"]["fixed_region"]["x_max"])
        self.fixed = (self.rest_position[:, 0] <= x_max_fixed)

        # Total mass
        envelope_volume = 0.075 * 0.18 * 1.18
        self.total_mass = self.density * envelope_volume
        self.mass = np.full(self.num_nodes, self.total_mass / self.num_nodes, dtype=float)
        self.inv_mass = np.where(self.fixed, 0.0, 1.0 / self.mass)
        self.reference_force = float(np.linalg.norm(self.mass * self.gravity[2]))
        if self.reference_force == 0.0:
            self.reference_force = float(self.total_mass * 9.81)

        self._build_elements_and_constraints()

    def _build_elements_and_constraints(self) -> None:
        """Decompose each of the 21 boxes into 5 tetrahedra and set up springs/coupling."""
        self.box_node_indices = []
        self.box_rest_positions = []
        self.box_rest_com = []
        self.tet_elements = []

        # Standard 5-tet decomposition of a hexahedron [0..7]
        # Box vertices: 0:(-,-,-), 1:(+,-,-), 2:(+,+,-), 3:(-,+,-),
        #               4:(-,-,+), 5:(+,-,+), 6:(+,+,+), 7:(-,+,+)
        tet_patterns_even = [
            (0, 1, 3, 4),
            (1, 2, 3, 6),
            (1, 4, 5, 6),
            (3, 4, 6, 7),
            (1, 3, 4, 6)
        ]
        tet_patterns_odd = [
            (0, 1, 2, 5),
            (0, 2, 3, 7),
            (0, 4, 5, 7),
            (2, 5, 6, 7),
            (0, 2, 5, 7)
        ]

        edge_set = set()

        for b in range(self.num_boxes):
            nodes = np.arange(b * 8, (b + 1) * 8)
            self.box_node_indices.append(nodes)
            X_b = self.rest_position[nodes]
            self.box_rest_positions.append(X_b)
            self.box_rest_com.append(np.mean(X_b, axis=0))

            tets = tet_patterns_even if (b % 2 == 0) else tet_patterns_odd
            for (n0, n1, n2, n3) in tets:
                i0, i1, i2, i3 = nodes[n0], nodes[n1], nodes[n2], nodes[n3]
                x0, x1, x2, x3 = self.rest_position[i0], self.rest_position[i1], self.rest_position[i2], self.rest_position[i3]
                Dm = np.column_stack([x1 - x0, x2 - x0, x3 - x0])
                vol = np.abs(np.linalg.det(Dm)) / 6.0
                if vol > 1e-12:
                    Dm_inv = np.linalg.inv(Dm)
                    self.tet_elements.append((i0, i1, i2, i3, Dm_inv, vol, b))

                # Add edges for distance constraints
                for pair in [(i0, i1), (i0, i2), (i0, i3), (i1, i2), (i1, i3), (i2, i3)]:
                    edge = tuple(sorted(pair))
                    edge_set.add(edge)

            # Add all 12 box edges and diagonals for extra structural integrity
            for i in range(8):
                for j in range(i + 1, 8):
                    edge_set.add((nodes[i], nodes[j]))

        # Inter-box connection springs: connect adjacent boxes at overlapping / adjacent faces
        for b in range(self.num_boxes - 1):
            nodes_a = self.box_node_indices[b]
            nodes_b = self.box_node_indices[b + 1]
            pos_a = self.rest_position[nodes_a]
            pos_b = self.rest_position[nodes_b]

            # Find closest pairs of vertices between consecutive boxes
            for i_idx, ia in enumerate(nodes_a):
                for j_idx, ib in enumerate(nodes_b):
                    d = np.linalg.norm(pos_a[i_idx] - pos_b[j_idx])
                    if d < 0.12:  # Close enough to be coupled across joint
                        edge_set.add(tuple(sorted((ia, ib))))

        # Build distance constraints list
        self.distance_constraints = []
        for (i, j) in edge_set:
            rest_len = np.linalg.norm(self.rest_position[i] - self.rest_position[j])
            if rest_len > 1e-6:
                self.distance_constraints.append((i, j, rest_len))

    def _compute_deformation_gradients(self) -> np.ndarray:
        """Compute average deformation gradient F for each of the 21 boxes."""
        F_boxes = np.zeros((self.num_boxes, 3, 3))
        counts = np.zeros(self.num_boxes)

        for (i0, i1, i2, i3, Dm_inv, vol, b) in self.tet_elements:
            x0, x1, x2, x3 = self.position[i0], self.position[i1], self.position[i2], self.position[i3]
            Ds = np.column_stack([x1 - x0, x2 - x0, x3 - x0])
            F = Ds @ Dm_inv
            F_boxes[b] += F * vol
            counts[b] += vol

        for b in range(self.num_boxes):
            if counts[b] > 0:
                F_boxes[b] /= counts[b]
            else:
                F_boxes[b] = np.eye(3)
        return F_boxes

    def _step(self, dt: float) -> None:
        """Position-based dynamics / XPBD step."""
        # 1. External forces & explicit velocity integration
        g = self.gravity
        self.velocity[~self.fixed] += g * dt

        # Viscous damping
        self.velocity *= 0.999

        p = self.position.copy()
        p[~self.fixed] += self.velocity[~self.fixed] * dt

        # 2. Constraint projections (XPBD / PBD iterations)
        num_solver_iters = 10
        # Compliances / stiffness factors
        for _ in range(num_solver_iters):
            # Distance constraints
            for (i, j, rest_len) in self.distance_constraints:
                w1 = self.inv_mass[i]
                w2 = self.inv_mass[j]
                w_sum = w1 + w2
                if w_sum == 0.0:
                    continue

                diff = p[i] - p[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-9:
                    c = dist - rest_len
                    correction = (c / w_sum) * (diff / dist)
                    if not self.fixed[i]:
                        p[i] -= w1 * correction * 0.5
                    if not self.fixed[j]:
                        p[j] += w2 * correction * 0.5

            # Volume preservation constraints for tetrahedra (nearly incompressible rubber)
            for (i0, i1, i2, i3, Dm_inv, rest_vol, b) in self.tet_elements:
                w0, w1, w2, w3 = self.inv_mass[i0], self.inv_mass[i1], self.inv_mass[i2], self.inv_mass[i3]
                w_sum = w0 + w1 + w2 + w3
                if w_sum == 0.0:
                    continue

                x0, x1, x2, x3 = p[i0], p[i1], p[i2], p[i3]
                cur_vol = np.dot(np.cross(x1 - x0, x2 - x0), x3 - x0) / 6.0
                c_vol = cur_vol - rest_vol

                # Gradients with respect to vertices
                grad1 = np.cross(x2 - x0, x3 - x0) / 6.0
                grad2 = np.cross(x3 - x0, x1 - x0) / 6.0
                grad3 = np.cross(x1 - x0, x2 - x0) / 6.0
                grad0 = -(grad1 + grad2 + grad3)

                grad_sq_sum = w0 * np.dot(grad0, grad0) + w1 * np.dot(grad1, grad1) + \
                              w2 * np.dot(grad2, grad2) + w3 * np.dot(grad3, grad3)

                if grad_sq_sum > 1e-12:
                    lambda_vol = -c_vol / (grad_sq_sum + 1e-6)
                    if not self.fixed[i0]:
                        p[i0] += w0 * lambda_vol * grad0 * 0.1
                    if not self.fixed[i1]:
                        p[i1] += w1 * lambda_vol * grad1 * 0.1
                    if not self.fixed[i2]:
                        p[i2] += w2 * lambda_vol * grad2 * 0.1
                    if not self.fixed[i3]:
                        p[i3] += w3 * lambda_vol * grad3 * 0.1

            # Ensure fixed nodes strictly maintain rest position
            p[self.fixed] = self.rest_position[self.fixed]

        # 3. Velocity update
        self.velocity[~self.fixed] = (p[~self.fixed] - self.position[~self.fixed]) / dt
        self.position = p

    def advance_to(self, time: float) -> StateFrame:
        if time <= self.time:
            return self._build_frame(self.time)

        target_time = float(time)
        while self.time + 1e-8 < target_time:
            dt = min(self.sub_dt, target_time - self.time)
            self._step(dt)
            self.time += dt

        self.time = target_time
        return self._build_frame(self.time)

    def _build_frame(self, time: float) -> StateFrame:
        # Compute element deformation gradients (shape: [21, 3, 3])
        F_elem = self._compute_deformation_gradients()

        # Budget external force: shape [G, K, 3] = [1, 1, 3]
        # Total external gravity force acting on dynamic body
        ext_force = np.sum(self.mass[:, None] * self.gravity[None, :], axis=0).reshape(1, 1, 3)

        # Cumulative mass source: shape [K] = [1]
        mass_source = np.zeros(1, dtype=float)

        frame = StateFrame(
            time=time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "fixed": self.fixed.astype(float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "rest_position": self.rest_position.copy(),
            },
            fields={
                "budget_external_force": ext_force,
                "cumulative_mass_source": mass_source,
                "element_deformation_gradient": F_elem,
            },
            scalars={
                "reference_force": float(self.reference_force),
                "gravity_x": float(self.gravity[0]),
                "gravity_y": float(self.gravity[1]),
                "gravity_z": float(self.gravity[2]),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        tip_z = float(self.position[-1, 2]) if self.position is not None else 0.0
        return {
            "time": self.time,
            "tip_z": tip_z,
            "num_nodes": self.num_nodes,
        }

