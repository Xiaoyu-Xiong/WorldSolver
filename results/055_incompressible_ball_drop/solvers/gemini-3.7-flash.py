"""Physical solver for 055_incompressible_ball_drop.

Simulates a homogeneous, incompressible soft elastic ball falling onto a flat rigid floor.
Models 3D volumetric deformation, volume conservation, elastic recovery, and contact.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        init_frame = scene.initial_state()

        self.N = len(init_frame.position)  # 12 surface vertices
        self.pos = np.array(init_frame.position, dtype=np.float64)  # shape (12, 3)
        self.vel = np.array(init_frame.velocity, dtype=np.float64)  # shape (12, 3)
        self.mass = np.array(init_frame.mass, dtype=np.float64)    # shape (12,)
        self.total_mass = float(np.sum(self.mass))                # 4.0 kg

        self.sim_time = float(init_frame.time)

        # Mesh faces (1-based to 0-based indexing from assets/ball_sphere.obj)
        faces_1based = [
            (1, 12, 6), (1, 6, 2), (1, 2, 8), (1, 8, 11), (1, 11, 12),
            (2, 6, 10), (6, 12, 5), (12, 11, 3), (11, 8, 7), (8, 2, 9),
            (4, 10, 5), (4, 5, 3), (4, 3, 7), (4, 7, 9), (4, 9, 10),
            (5, 10, 6), (3, 5, 12), (7, 3, 11), (9, 7, 8), (10, 9, 2),
        ]
        self.faces = np.array([[a - 1, b - 1, c - 1] for a, b, c in faces_1based], dtype=np.int32)
        self.Ne = len(self.faces)  # 20 elements (tetrahedra formed with ball center)

        # Build unique edges for structural spring network
        edge_set = set()
        for face in self.faces:
            for i in range(3):
                u, v = sorted((face[i], face[(i + 1) % 3]))
                edge_set.add((u, v))
        self.edges = np.array(list(edge_set), dtype=np.int32)

        # Compute initial rest center and rest vertex offsets relative to center
        self.rest_center = np.mean(self.pos, axis=0)
        self.rest_offsets = self.pos - self.rest_center  # unit vectors of icosphere
        self.rest_lengths = np.linalg.norm(self.pos[self.edges[:, 0]] - self.pos[self.edges[:, 1]], axis=1)
        self.rest_radius = np.linalg.norm(self.rest_offsets, axis=1)  # all 1.0

        # Reference element volumes and inverse reference matrices Dm_inv for shape matching / FEM
        # Element i: tetrahedron formed by center c and face vertices (v0, v1, v2)
        self.Dm_inv = np.zeros((self.Ne, 3, 3), dtype=np.float64)
        for e, face in enumerate(self.faces):
            r0 = self.rest_offsets[face[0]]
            r1 = self.rest_offsets[face[1]]
            r2 = self.rest_offsets[face[2]]
            Dm = np.column_stack([r0, r1, r2])
            self.Dm_inv[e] = np.linalg.inv(Dm)

        # Initial total reference volume
        self.V0 = self._compute_volume(self.pos)

        # Physics parameters
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.k_edge = 380.0       # Edge spring stiffness
        self.k_radial = 300.0     # Radial spring stiffness from center
        self.k_vol = 1200.0       # Hydrostatic / incompressibility volume pressure stiffness
        self.damping = 0.08       # Internal damping

        # Floor contact parameters
        self.floor_z = 0.0
        self.k_contact = 6000.0
        self.d_contact = 25.0

        # Budget tracking
        self.cumulative_external_impulse = np.zeros((1, 3), dtype=np.float64)
        self.budget_domain_weights = np.ones((1, self.N), dtype=np.float64)
        self.current_ext_force = np.zeros((1, 1, 3), dtype=np.float64)

        # Reference scales
        self.reference_momentum = 4.0 * np.sqrt(2.0 * 9.81 * 1.25)  # ~19.8 kg*m/s
        self.constraint_scale = float(self.V0)

    def _compute_volume(self, pos: np.ndarray) -> float:
        """Compute closed mesh signed volume using center-based tetrahedra."""
        center = np.mean(pos, axis=0)
        rel = pos - center
        # For each triangular face (v0, v1, v2), volume is (1/6) * det([v0, v1, v2])
        v0 = rel[self.faces[:, 0]]
        v1 = rel[self.faces[:, 1]]
        v2 = rel[self.faces[:, 2]]
        cross = np.cross(v1, v2)
        dets = np.sum(v0 * cross, axis=1)
        return float(np.sum(dets) / 6.0)

    def _compute_deformation_gradients(self, pos: np.ndarray) -> np.ndarray:
        """Compute deformation gradient F for each of the 20 volumetric elements."""
        center = np.mean(pos, axis=0)
        rel = pos - center
        x0 = rel[self.faces[:, 0]]
        x1 = rel[self.faces[:, 1]]
        x2 = rel[self.faces[:, 2]]
        F = np.zeros((self.Ne, 3, 3), dtype=np.float64)
        for e in range(self.Ne):
            Ds = np.column_stack([x0[e], x1[e], x2[e]])
            F[e] = Ds @ self.Dm_inv[e]
        return F

    def _physics_step(self, dt: np.float64) -> None:
        """Advance the physics by a small sub-step dt using semi-implicit / symplectic Euler."""
        center = np.mean(self.pos, axis=0)
        center_vel = np.mean(self.vel, axis=0)
        rel_pos = self.pos - center
        rel_vel = self.vel - center_vel

        forces = np.zeros_like(self.pos)

        # 1. Gravity on each vertex
        forces += self.mass[:, None] * self.gravity

        # 2. Edge elastic springs + damping along edges
        p0 = self.pos[self.edges[:, 0]]
        p1 = self.pos[self.edges[:, 1]]
        dp = p1 - p0
        dist = np.linalg.norm(dp, axis=1, keepdims=True)
        dist = np.maximum(dist, 1e-8)
        dir_vec = dp / dist
        stretch = dist.squeeze(-1) - self.rest_lengths
        spring_f = self.k_edge * stretch[:, None] * dir_vec

        v0 = self.vel[self.edges[:, 0]]
        v1 = self.vel[self.edges[:, 1]]
        dv = v1 - v0
        damp_f = self.damping * np.sum(dv * dir_vec, axis=1, keepdims=True) * dir_vec

        total_edge_f = spring_f + damp_f
        np.add.at(forces, self.edges[:, 0], total_edge_f)
        np.add.at(forces, self.edges[:, 1], -total_edge_f)

        # 3. Radial volumetric elastic springs (relative to center of mass)
        curr_rad = np.linalg.norm(rel_pos, axis=1, keepdims=True)
        curr_rad = np.maximum(curr_rad, 1e-8)
        rad_dir = rel_pos / curr_rad
        rad_stretch = curr_rad - self.rest_radius[:, None]
        forces -= self.k_radial * rad_stretch * rad_dir
        forces -= self.damping * np.sum(rel_vel * rad_dir, axis=1, keepdims=True) * rad_dir

        # 4. Incompressible volume preservation forces
        current_vol = self._compute_volume(self.pos)
        vol_err = current_vol - self.V0
        pressure = -self.k_vol * vol_err  # positive pressure if compressed (vol < V0)

        # Volume gradient w.r.t. vertex positions
        # For each face (v0, v1, v2), dV/dv0 = (v1 x v2) / 6
        v0 = rel_pos[self.faces[:, 0]]
        v1 = rel_pos[self.faces[:, 1]]
        v2 = rel_pos[self.faces[:, 2]]
        grad_v0 = np.cross(v1, v2) / 6.0
        grad_v1 = np.cross(v2, v0) / 6.0
        grad_v2 = np.cross(v0, v1) / 6.0

        np.add.at(forces, self.faces[:, 0], pressure * grad_v0)
        np.add.at(forces, self.faces[:, 1], pressure * grad_v1)
        np.add.at(forces, self.faces[:, 2], pressure * grad_v2)

        # 5. Floor contact forces and external force budget
        contact_force_total = np.zeros(3, dtype=np.float64)
        for i in range(self.N):
            z = self.pos[i, 2]
            if z < self.floor_z:
                penetration = self.floor_z - z
                normal_f = self.k_contact * penetration
                if self.vel[i, 2] < 0:
                    normal_f -= self.d_contact * self.vel[i, 2]
                normal_f = max(normal_f, 0.0)
                forces[i, 2] += normal_f
                contact_force_total[2] += normal_f

                # Horizontal friction damping during contact
                v_horiz = self.vel[i, :2]
                forces[i, :2] -= 0.5 * self.d_contact * v_horiz

        # Total external force = gravity + floor contact
        total_ext_f = self.total_mass * self.gravity + contact_force_total
        self.current_ext_force[0, 0, :] = total_ext_f
        self.cumulative_external_impulse[0, :] += total_ext_f * dt

        # 6. Integration: Symplectic Euler
        acc = forces / self.mass[:, None]
        self.vel += acc * dt

        # Floor velocity penalty / penetration prevention
        for i in range(self.N):
            if self.pos[i, 2] < self.floor_z and self.vel[i, 2] < 0:
                self.vel[i, 2] = -0.1 * self.vel[i, 2]

        self.pos += self.vel * dt

        # Post-integration floor projection
        for i in range(self.N):
            if self.pos[i, 2] < self.floor_z:
                self.pos[i, 2] = self.floor_z

        self.sim_time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        # Step simulation to reach target_time
        sub_dt = 0.0002  # 0.2 ms stable sub-stepping
        while self.sim_time + 1e-9 < target_time:
            dt = min(sub_dt, target_time - self.sim_time)
            self._physics_step(dt)

        # Compute trajectory contract fields
        F = self._compute_deformation_gradients(self.pos)
        current_vol = self._compute_volume(self.pos)

        fields = {
            "budget_cumulative_external_impulse": np.array(self.cumulative_external_impulse, dtype=np.float64),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_domain_weights": np.array(self.budget_domain_weights, dtype=np.float64),
            "budget_external_force": np.array(self.current_ext_force, dtype=np.float64),
            "constraint_target": np.array([self.V0], dtype=np.float64),
            "constraint_value": np.array([current_vol], dtype=np.float64),
            "cumulative_mass_source": np.zeros((1,), dtype=np.float64),
            "element_deformation_gradient": np.array(F, dtype=np.float64),
        }

        scalars = {
            "constraint_scale": float(self.constraint_scale),
            "reference_momentum": float(self.reference_momentum),
        }

        return StateFrame(
            time=float(target_time),
            position=np.array(self.pos, dtype=np.float64),
            velocity=np.array(self.vel, dtype=np.float64),
            mass=np.array(self.mass, dtype=np.float64),
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {}

