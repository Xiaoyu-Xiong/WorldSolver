"""Physically-based deformable solid solver for hard sphere ground impact."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import spatial
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        np.random.seed(seed)

        # Load mesh
        asset_path = Path(scene.config.get("subject_asset", "assets/subject.npz"))
        if not asset_path.is_absolute():
            asset_path = Path(scene.__module__).resolve().parent / asset_path if hasattr(scene, "__file__") else Path.cwd() / asset_path
            if not asset_path.exists():
                asset_path = Path(__file__).resolve().parent.parent / scene.config.get("subject_asset", "assets/subject.npz")

        with np.load(asset_path) as data:
            self.rest_positions = np.asarray(data["vertices"], dtype=float).copy()
            self.faces = np.asarray(data["faces"], dtype=np.int64).copy()

        self.num_vertices = len(self.rest_positions)
        self.positions = self.rest_positions.copy()

        # Initial velocity
        init_vel = np.asarray(self.config.get("initial_velocity", [0.0, 0.0, -2.5]), dtype=float)
        self.velocities = np.tile(init_vel, (self.num_vertices, 1))

        # Masses
        total_mass = float(self.config.get("subject_mass", 2.0))
        self.mass = np.full(self.num_vertices, total_mass / self.num_vertices, dtype=float)

        # Simulation parameters
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.stiffness = float(self.config.get("stiffness", 44.0))
        self.damping = float(self.config.get("damping", 0.75))
        self.restitution = float(self.config.get("restitution", 0.62))
        self.friction = float(self.config.get("friction", 0.42))
        self.sphere_radius = float(self.config.get("sphere_radius", 0.42))
        self.contact_length_scale = float(self.config.get("contact_length_scale", 0.04))
        self.ground_z = float(self.config.get("ground_z", 0.0))

        # Build spring mesh: surface mesh edges + volumetric interior cross-springs
        edges_set = set()
        for f in self.faces:
            for i in range(3):
                u, v = f[i], f[(i + 1) % 3]
                if u > v:
                    u, v = v, u
                edges_set.add((u, v))

        # Connect interior struts across the sphere (k-NN or diameter/cross connections)
        # Using KDTree to find internal structural springs
        kdtree = spatial.KDTree(self.rest_positions)
        # Find neighbors within ~2.5x edge length or k-nearest
        dists, indices = kdtree.query(self.rest_positions, k=16)
        for i in range(self.num_vertices):
            for j in indices[i, 1:]:
                u, v = min(i, j), max(i, j)
                edges_set.add((u, v))

        # Also add diametrically opposed / central chords to preserve 3D spherical volume & shape
        rest_center = np.mean(self.rest_positions, axis=0)
        rel_pos = self.rest_positions - rest_center
        # Find antipodal vertices
        for i in range(self.num_vertices):
            target = rest_center - rel_pos[i]
            _, opp_idx = kdtree.query(target, k=3)
            for j in opp_idx:
                u, v = min(i, j), max(i, j)
                if u != v:
                    edges_set.add((u, v))

        self.edges = np.array(list(edges_set), dtype=np.int64)
        e0 = self.edges[:, 0]
        e1 = self.edges[:, 1]
        self.rest_lengths = np.linalg.norm(self.rest_positions[e0] - self.rest_positions[e1], axis=1)

        # Precompute local rest neighborhood for deformation gradient F computation
        # For each vertex, take k nearest neighbors in rest state
        self.nn_k = 8
        _, self.nn_indices = kdtree.query(self.rest_positions, k=self.nn_k)
        self.inv_dX0 = []
        for i in range(self.num_vertices):
            nbrs = self.nn_indices[i, 1:]  # (k-1, 3)
            dX0 = (self.rest_positions[nbrs] - self.rest_positions[i]).T  # (3, k-1)
            # pseudo-inverse of dX0: dX0_pinv = (dX0 dX0^T)^(-1) dX0
            pinv = np.linalg.pinv(dX0)  # (k-1, 3)
            self.inv_dX0.append(pinv)

        self.current_time = 0.0
        self.substep_dt = 0.001  # 1 ms substep for numerical stability of stiff springs

    def _compute_deformation_gradients(self) -> np.ndarray:
        F = np.zeros((self.num_vertices, 3, 3), dtype=float)
        for i in range(self.num_vertices):
            nbrs = self.nn_indices[i, 1:]
            dx = (self.positions[nbrs] - self.positions[i]).T  # (3, k-1)
            # F = dx @ pinv
            F[i] = dx @ self.inv_dX0[i]
        return F

    def _substep(self, dt: float) -> None:
        # Internal spring forces + damping
        e0 = self.edges[:, 0]
        e1 = self.edges[:, 1]

        diff = self.positions[e1] - self.positions[e0]
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-12)
        direction = diff / dist

        # Strain & spring force
        strain = dist[:, 0] - self.rest_lengths
        # Spring stiffness scaled appropriately for volumetric preservation
        k_s = self.stiffness * 800.0
        spring_mag = k_s * strain

        # Relative velocity damping
        rel_vel = self.velocities[e1] - self.velocities[e0]
        damping_mag = self.damping * 40.0 * np.sum(rel_vel * direction, axis=1)

        f_total_mag = spring_mag + damping_mag
        f_spring = direction * f_total_mag[:, None]

        # Accumulate forces
        forces = np.zeros_like(self.positions)
        np.add.at(forces, e0, f_spring)
        np.add.at(forces, e1, -f_spring)

        # Global damping to avoid high frequency ringing
        forces -= self.damping * 0.5 * self.velocities * self.mass[:, None]

        # External forces (gravity)
        forces += self.mass[:, None] * self.gravity[None, :]

        # Symplectic Euler integration
        acc = forces / self.mass[:, None]
        self.velocities += acc * dt
        self.positions += self.velocities * dt

        # Ground collision handling
        penetration = self.ground_z - self.positions[:, 2]
        colliding = penetration > 0.0

        if np.any(colliding):
            # Position projection
            self.positions[colliding, 2] = self.ground_z

            # Velocity postprocessing / impulse
            # Normal velocity
            v_n = self.velocities[colliding, 2]
            approaching = v_n < 0.0
            idx = np.where(colliding)[0][approaching]

            if len(idx) > 0:
                # Normal rebound
                self.velocities[idx, 2] = -self.restitution * self.velocities[idx, 2]

                # Tangential friction
                v_tan = self.velocities[idx, :2]
                v_tan_norm = np.linalg.norm(v_tan, axis=1, keepdims=True)
                friction_reduction = np.maximum(0.0, 1.0 - self.friction * (1.0 + self.restitution) * np.abs(self.velocities[idx, 2:3]) / np.maximum(v_tan_norm, 1e-8))
                self.velocities[idx, :2] *= friction_reduction

    def advance_to(self, time: float) -> StateFrame:
        target_time = float(time)
        while self.current_time < target_time - 1e-9:
            dt = min(self.substep_dt, target_time - self.current_time)
            self._substep(dt)
            self.current_time += dt

        self.current_time = target_time

        # Compute StateFrame outputs
        deformation_gradient = self._compute_deformation_gradients()
        contact_gap = np.maximum(self.positions[:, 2] - self.ground_z, 0.0)
        ext_force = self.mass[:, None] * self.gravity[None, :]

        frame = StateFrame(
            time=float(target_time),
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.num_vertices, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": deformation_gradient,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": ext_force,
                "contact_signed_distance": contact_gap,
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_vertices": self.num_vertices,
            "num_edges": len(self.edges),
        }

