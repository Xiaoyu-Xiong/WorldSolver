"""Deterministic physics solver for 024_stretched_cube scenario."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.time = 0.0
        self.dt = 0.005  # internal sub-step dt

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        # Load mesh
        asset = Path(scene.config["subject_asset"])
        if not asset.is_absolute():
            asset = Path(__file__).resolve().parent.parent / asset
        with np.load(asset) as data:
            self.rest_pos = np.asarray(data["vertices"], dtype=np.float64)
            self.faces = np.asarray(data["faces"], dtype=np.int64)

        self.n = len(self.rest_pos)
        self.pos = self.rest_pos.copy()
        self.vel = np.tile(np.asarray(self.config.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=np.float64), (self.n, 1))

        self.total_mass = float(self.config.get("subject_mass", 1.3))
        self.mass = np.full(self.n, self.total_mass / self.n, dtype=np.float64)
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.ground_z = float(self.config.get("ground_z", 0.0))
        self.contact_length_scale = float(self.config.get("contact_length_scale", 0.04))

        self.stiffness = float(self.config.get("stiffness", 28.0))
        self.damping = float(self.config.get("damping", 0.42))
        self.stretch_ratio = float(self.config.get("stretch_ratio", 2.15))
        self.stretch_end = float(self.config.get("stretch_end", 1.25))
        self.pulse_time = float(self.config.get("pulse_time", 1.55))

        self.time = 0.0

        # Identify left and right boundary vertices (grips)
        min_x = self.rest_pos[:, 0].min()
        max_x = self.rest_pos[:, 0].max()
        tol = 1e-4
        self.left_mask = (self.rest_pos[:, 0] <= min_x + tol)
        self.right_mask = (self.rest_pos[:, 0] >= max_x - tol)
        self.free_mask = ~(self.left_mask | self.right_mask)

        # Build spring network from surface mesh and cross-springs
        edges = set()
        for f in self.faces:
            for i in range(3):
                u, v = sorted([f[i], f[(i+1)%3]])
                edges.add((u, v))

        # Also build interior/cross-springs between nearby vertices to maintain volume/elastic cube behavior
        from scipy.spatial import KDTree
        tree = KDTree(self.rest_pos)
        # Find neighbors within a certain radius to create shear/bending/cross springs
        # Mesh grid spacing in rest is ~0.07375
        h_rest = 0.07375
        pairs = tree.query_pairs(r=h_rest * 2.5)
        for u, v in pairs:
            edges.add((min(u, v), max(u, v)))

        edge_list = list(edges)
        self.spring_i = np.array([e[0] for e in edge_list], dtype=np.int64)
        self.spring_j = np.array([e[1] for e in edge_list], dtype=np.int64)

        diff = self.rest_pos[self.spring_j] - self.rest_pos[self.spring_i]
        self.rest_len = np.linalg.norm(diff, axis=1)

        # Spring stiffness adjusted per spring
        # Normalize spring stiffness so overall bulk stiffness matches config
        num_springs = len(self.rest_len)
        self.k_spring = np.full(num_springs, self.stiffness * 50.0 / (num_springs / self.n), dtype=np.float64)

        # For deformation gradient estimation, compute local neighbor rest frames
        self.nbr_indices = []
        self.nbr_inv_cov = []
        for i in range(self.n):
            dists = np.linalg.norm(self.rest_pos - self.rest_pos[i], axis=1)
            nbrs = np.where((dists > 1e-6) & (dists <= h_rest * 3.1))[0]
            if len(nbrs) < 3:
                nbrs = np.argsort(dists)[1:5]
            self.nbr_indices.append(nbrs)
            dX = self.rest_pos[nbrs] - self.rest_pos[i]  # (K, 3)
            cov = dX.T @ dX + 1e-6 * np.eye(3)
            self.nbr_inv_cov.append(np.linalg.inv(cov))

        # Target displacement bounds
        self.min_x = min_x
        self.max_x = max_x
        self.center_x = (min_x + max_x) * 0.5

    def _grip_target_x(self, t: float, side: str) -> float:
        """Compute kinematic x-position of left or right grip at time t."""
        # Initial stretch phase: from t=0 to t=stretch_end, stretch symmetrically
        # stretch_ratio is 2.15
        L0 = self.max_x - self.min_x
        half_L0 = L0 * 0.5

        if t <= self.stretch_end:
            s = t / self.stretch_end
            # smooth step
            alpha = s * s * (3.0 - 2.0 * s)
            current_ratio = 1.0 + (self.stretch_ratio - 1.0) * alpha
        elif t <= self.pulse_time:
            # Hold stretched
            current_ratio = self.stretch_ratio
        elif t <= self.pulse_time + 0.3:
            # Compressive pulse / perturbation then release/hold
            # Small compressive dip
            p_dt = t - self.pulse_time
            pulse_phase = np.sin(np.pi * p_dt / 0.3)
            current_ratio = self.stretch_ratio - 0.25 * pulse_phase
        else:
            current_ratio = self.stretch_ratio

        if side == 'left':
            return self.center_x - half_L0 * current_ratio
        else:
            return self.center_x + half_L0 * current_ratio

    def _step(self, dt: float) -> None:
        # Update grip positions kinematically
        target_left_x = self._grip_target_x(self.time + dt, 'left')
        target_right_x = self._grip_target_x(self.time + dt, 'right')

        # Calculate spring forces
        # F_ij = k * (||x_j - x_i|| - L0) * (x_j - x_i) / ||x_j - x_i||
        pos_i = self.pos[self.spring_i]
        pos_j = self.pos[self.spring_j]
        diff = pos_j - pos_i
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist_clamped = np.maximum(dist, 1e-8)
        dir_vec = diff / dist_clamped

        # Elastic force
        stretch = dist - self.rest_len[:, None]
        f_spring = self.k_spring[:, None] * stretch * dir_vec

        # Damping force on relative velocity
        vel_i = self.vel[self.spring_i]
        vel_j = self.vel[self.spring_j]
        rel_vel = vel_j - vel_i
        proj_vel = np.sum(rel_vel * dir_vec, axis=1, keepdims=True)
        f_damp = self.damping * self.k_spring[:, None] * 0.05 * proj_vel * dir_vec

        f_total_spring = f_spring + f_damp

        # Accumulate forces on vertices
        force = np.zeros_like(self.pos)
        np.add.at(force, self.spring_i, f_total_spring)
        np.add.at(force, self.spring_j, -f_total_spring)

        # External forces (gravity, global drag)
        force += self.mass[:, None] * self.gravity[None, :]
        force -= self.damping * self.mass[:, None] * self.vel

        # Integrate free vertices with symplectic Euler
        acc = force / self.mass[:, None]
        self.vel[self.free_mask] += dt * acc[self.free_mask]
        self.pos[self.free_mask] += dt * self.vel[self.free_mask]

        # Kinematic boundary conditions on grips
        left_dx = (target_left_x - self.pos[self.left_mask, 0])
        self.vel[self.left_mask, 0] = left_dx / dt
        self.pos[self.left_mask, 0] = target_left_x
        self.vel[self.left_mask, 1:] = 0.0

        right_dx = (target_right_x - self.pos[self.right_mask, 0])
        self.vel[self.right_mask, 0] = right_dx / dt
        self.pos[self.right_mask, 0] = target_right_x
        self.vel[self.right_mask, 1:] = 0.0

        # Ground collision
        penetration = self.ground_z - self.pos[:, 2]
        contact = penetration > 0
        if np.any(contact):
            self.pos[contact, 2] = self.ground_z
            self.vel[contact, 2] = np.maximum(self.vel[contact, 2], 0.0)
            self.vel[contact, :2] *= (1.0 - float(self.config.get("friction", 0.5)))

        self.time += dt

    def _compute_deformation_gradient(self) -> np.ndarray:
        """Compute per-vertex deformation gradient F."""
        F_all = np.zeros((self.n, 3, 3), dtype=np.float64)
        for i in range(self.n):
            nbrs = self.nbr_indices[i]
            dX = self.rest_pos[nbrs] - self.rest_pos[i]  # (K, 3)
            dx = self.pos[nbrs] - self.pos[i]            # (K, 3)
            # F * dX.T ~ dx.T => F * (dX.T @ dX) ~ dx.T @ dX
            # F = (dx.T @ dX) @ (dX.T @ dX)^-1
            F = (dx.T @ dX) @ self.nbr_inv_cov[i]
            F_all[i] = F
        return F_all

    def advance_to(self, target_time: float) -> StateFrame:
        # Step solver up to target_time
        sub_dt = 0.001  # high stability sub-step
        while self.time < target_time - 1e-8:
            step_size = min(sub_dt, target_time - self.time)
            self._step(step_size)

        # Compute required outputs
        F = self._compute_deformation_gradient()
        gap = np.maximum(self.pos[:, 2] - self.ground_z, 0.0)
        ext_force = self.mass[:, None] * self.gravity[None, :]

        frame = StateFrame(
            time=target_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": ext_force,
                "contact_signed_distance": gap,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_particles": self.n,
            "simulated_time": self.time,
        }

