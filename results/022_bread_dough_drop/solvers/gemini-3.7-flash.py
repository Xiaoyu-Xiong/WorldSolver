"""Physical solver for bread dough drop simulation.

Simulates elastoplastic deformable solid (bread dough) falling under gravity and impacting the ground.
Uses a particle/spring/viscoplastic lattice model with volume preservation, plastic deformation,
contact/friction with the ground plane, and accurate trajectory output matching StateFrame contracts.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        # Load mesh topology
        asset_path = Path(__file__).resolve().parent.parent / self.config["subject_asset"]
        with np.load(asset_path) as data:
            self.rest_vertices = np.asarray(data["vertices"], dtype=np.float64)
            self.faces = np.asarray(data["faces"], dtype=np.int64) if "faces" in data else None

        self.n = len(self.rest_vertices)
        self.dim = 3

        # Physical parameters from config
        self.total_mass = float(self.config.get("subject_mass", 0.55))
        self.mass = np.full(self.n, self.total_mass / self.n, dtype=np.float64)
        self.inv_mass = 1.0 / self.mass
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.ground_z = float(self.config.get("ground_z", 0.0))
        self.contact_length_scale = float(self.config.get("contact_length_scale", 0.04))

        self.stiffness = float(self.config.get("stiffness", 5.0))
        self.plasticity = float(self.config.get("plasticity", 0.86))
        self.damping = float(self.config.get("damping", 3.4))
        self.restitution = float(self.config.get("restitution", 0.02))
        self.friction = float(self.config.get("friction", 0.82))

        # State variables
        self.pos = self.rest_vertices.copy()
        v0 = np.asarray(self.config.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.vel = np.tile(v0, (self.n, 1))
        self.current_time = 0.0

        # Build connectivity (edges from faces + k-nearest neighbors for volumetric structural integrity)
        edges = set()
        if self.faces is not None:
            for f in self.faces:
                for i in range(3):
                    j = (i + 1) % 3
                    v1, v2 = int(f[i]), int(f[j])
                    if v1 > v2:
                        v1, v2 = v2, v1
                    edges.add((v1, v2))

        # Also connect internal/nearby vertices to maintain 3D structure
        coords = self.rest_vertices
        dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        # For each vertex, connect to 12 nearest neighbors
        k_neighbors = min(12, self.n - 1)
        for i in range(self.n):
            nn = np.argsort(dist_matrix[i])[:k_neighbors + 1]
            for j in nn:
                if i != j:
                    v1, v2 = min(i, int(j)), max(i, int(j))
                    edges.add((v1, v2))

        self.edge_i = np.array([e[0] for e in edges], dtype=np.int64)
        self.edge_j = np.array([e[1] for e in edges], dtype=np.int64)
        self.num_edges = len(self.edge_i)

        # Rest lengths of edges
        d_rest = np.linalg.norm(self.rest_vertices[self.edge_i] - self.rest_vertices[self.edge_j], axis=1)
        self.edge_rest_len = d_rest.copy()
        self.edge_orig_len = d_rest.copy()

        # Centroid rest offsets for shape matching / volume retention
        self.rest_com = np.mean(self.rest_vertices, axis=0)
        self.rest_rel_pos = self.rest_vertices - self.rest_com

        # Deformation gradient per vertex (identity initially)
        self.F = np.tile(np.eye(3, dtype=np.float64), (self.n, 1, 1))

        # Internal substep dt for numerical stability
        self.sub_dt = 0.001

    def _compute_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute internal elastoplastic forces, contact forces, and gravity."""
        forces = np.zeros_like(self.pos)

        # 1. Edge spring forces with viscoplastic yielding
        diff = self.pos[self.edge_j] - self.pos[self.edge_i]
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-8)
        dir_vec = diff / dist

        # Strain
        strain = (dist.squeeze() - self.edge_rest_len) / np.maximum(self.edge_rest_len, 1e-6)

        # Plastic deformation of rest length: if strain exceeds yield threshold, flow plastically
        # Yield strain threshold
        yield_strain = 0.05 / (1.0 + self.plasticity * 2.0)
        plastic_flow_rate = self.plasticity * 15.0

        excess_pos = np.maximum(strain - yield_strain, 0.0)
        excess_neg = np.minimum(strain + yield_strain, 0.0)
        plastic_rate = np.where(strain > yield_strain, excess_pos, np.where(strain < -yield_strain, excess_neg, 0.0))

        # Update rest lengths plastically
        self.edge_rest_len += plastic_rate * self.edge_rest_len * (plastic_flow_rate * self.sub_dt)
        # Bound plastic deformation from collapsing to 0 or blowing up
        self.edge_rest_len = np.clip(self.edge_rest_len, 0.15 * self.edge_orig_len, 3.0 * self.edge_orig_len)

        # Elastic spring force
        k_spring = self.stiffness * 120.0
        f_spring_mag = k_spring * (dist.squeeze() - self.edge_rest_len)

        # Damping along edge
        rel_vel = self.vel[self.edge_j] - self.vel[self.edge_i]
        v_proj = np.sum(rel_vel * dir_vec, axis=1)
        c_damp = self.damping * 4.0
        f_damp_mag = c_damp * v_proj

        total_f_edge = (f_spring_mag + f_damp_mag)[:, None] * dir_vec

        np.add.at(forces, self.edge_i, total_f_edge)
        np.add.at(forces, self.edge_j, -total_f_edge)

        # 2. Global volume / shape retention damping
        com = np.mean(self.pos, axis=0)
        v_com = np.mean(self.vel, axis=0)
        # Relative velocity damping toward rigid/soft body motion
        forces -= self.damping * 1.5 * (self.vel - v_com) * self.mass[:, None]

        # 3. Ground contact forces (penalty + friction)
        # Ground plane is at z = ground_z
        z_pen = self.ground_z - self.pos[:, 2]
        contact_mask = z_pen > 0.0

        k_contact = 25000.0
        c_contact = 50.0

        if np.any(contact_mask):
            # Normal spring-damper penalty force
            fn = k_contact * z_pen[contact_mask] - c_contact * np.minimum(self.vel[contact_mask, 2], 0.0)
            fn = np.maximum(fn, 0.0)
            forces[contact_mask, 2] += fn

            # Coulomb friction force
            v_horiz = self.vel[contact_mask, :2]
            v_horiz_norm = np.linalg.norm(v_horiz, axis=1, keepdims=True)
            v_horiz_dir = np.zeros_like(v_horiz)
            nonzero_v = (v_horiz_norm[:, 0] > 1e-6)
            v_horiz_dir[nonzero_v] = v_horiz[nonzero_v] / v_horiz_norm[nonzero_v]

            f_fric_mag = np.minimum(self.friction * fn, 300.0 * v_horiz_norm[:, 0])
            forces[contact_mask, 0] -= f_fric_mag * v_horiz_dir[:, 0]
            forces[contact_mask, 1] -= f_fric_mag * v_horiz_dir[:, 1]

        # External force is pure gravity
        ext_force = self.mass[:, None] * self.gravity[None, :]
        total_forces = forces + ext_force

        return total_forces, ext_force

    def _step(self, dt: float) -> None:
        """Advance the physics by dt using semi-implicit Euler integration."""
        total_forces, _ = self._compute_forces()

        acc = total_forces * self.inv_mass[:, None]
        self.vel += acc * dt
        self.pos += self.vel * dt

        # Hard contact projection / boundary enforcement for ground
        penetration = self.ground_z - self.pos[:, 2]
        pen_mask = penetration > 0.0
        if np.any(pen_mask):
            self.pos[pen_mask, 2] = self.ground_z
            # Restitution and friction on velocity
            self.vel[pen_mask, 2] = -self.restitution * self.vel[pen_mask, 2]
            if self.restitution == 0.0 or np.all(self.vel[pen_mask, 2] < 0.05):
                self.vel[pen_mask, 2] = np.maximum(self.vel[pen_mask, 2], 0.0)
            # Tangential friction deceleration
            friction_factor = np.exp(-self.friction * 20.0 * dt)
            self.vel[pen_mask, 0] *= friction_factor
            self.vel[pen_mask, 1] *= friction_factor

        self.current_time += dt

    def _update_deformation_gradient(self) -> None:
        """Estimate deformation gradient F per vertex from neighbor displacements."""
        diff_curr = self.pos[self.edge_j] - self.pos[self.edge_i]
        diff_rest = self.rest_vertices[self.edge_j] - self.rest_vertices[self.edge_i]

        # Compute local affine transformation for each vertex using least squares
        for i in range(self.n):
            # Edges incident to vertex i
            mask_i = (self.edge_i == i)
            mask_j = (self.edge_j == i)

            d_curr = np.vstack([diff_curr[mask_i], -diff_curr[mask_j]])
            d_rest = np.vstack([diff_rest[mask_i], -diff_rest[mask_j]])

            if len(d_rest) >= 3:
                # F * d_rest^T = d_curr^T => F = (d_curr^T @ d_rest) @ inv(d_rest^T @ d_rest)
                A = d_rest.T @ d_rest + 1e-5 * np.eye(3)
                B = d_curr.T @ d_rest
                self.F[i] = B @ np.linalg.inv(A)
            else:
                self.F[i] = np.eye(3)

    def advance_to(self, time: float) -> StateFrame:
        # Step simulation forward to target time
        while self.current_time + 1e-9 < time:
            dt = min(self.sub_dt, time - self.current_time)
            if dt <= 1e-9:
                break
            self._step(dt)

        self.current_time = time
        self._update_deformation_gradient()

        # Compute signed distance to ground (positive above ground, negative inside)
        gap = self.pos[:, 2] - self.ground_z
        ext_force = self.mass[:, None] * self.gravity[None, :]

        frame = StateFrame(
            time=float(self.current_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self.F.copy(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": ext_force,
                "contact_signed_distance": gap,
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": self.n,
            "num_edges": self.num_edges,
        }

