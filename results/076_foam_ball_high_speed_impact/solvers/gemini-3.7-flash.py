"""High-Speed Foam Practice Ball Impact Solver (Self-contained Continuum / Spring-Lattice Model).

Simulates high-speed volumetric elastic ball impact against a rigid wall at x = 0.
Produces deterministic, physically-grounded StateFrames matching contract requirements.
"""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = {}
        self.dt = 1e-6  # internal sub-step
        self.time = 0.0

        # Physical parameters
        self.E = 1.0e7
        self.nu = 0.45
        self.density = 1150.0
        self.radius = 0.02
        self.total_mass = 0.0
        self.g = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # Particles / Nodal representation
        self.pos = None
        self.vel = None
        self.mass = None
        self.rest_pos = None
        self.N = 0

        # Spring-lattice / volumetric connections
        self.spring_indices = None
        self.spring_rest_len = None
        self.spring_k = None
        self.spring_damping = None

        # Deformation gradient elements (local neighborhoods)
        self.elem_neighbors = None
        self.elem_inv_Dm = None
        self.Ne = 0

        # Budget tracking
        self.cumulative_impulse = np.zeros((1, 3), dtype=np.float64)
        self.reference_momentum = 1.0
        self.wall_contact_force = np.zeros(3, dtype=np.float64)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        ball = self.config["foam_ball"]
        self.E = float(ball["youngs_modulus"])
        self.nu = float(ball["poisson_ratio"])
        self.density = float(ball["density"])
        self.radius = float(ball["radius"])
        self.g = np.asarray(self.config.get("initial_conditions", {}).get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        init_frame = scene.initial_state()
        self.time = float(init_frame.time)
        self.pos = np.ascontiguousarray(init_frame.position, dtype=np.float64).copy()
        self.vel = np.ascontiguousarray(init_frame.velocity, dtype=np.float64).copy()
        self.mass = np.ascontiguousarray(init_frame.mass, dtype=np.float64).copy()
        self.rest_pos = self.pos.copy()
        self.N = len(self.pos)
        self.total_mass = float(np.sum(self.mass))

        # Initial momentum magnitude as base scale
        init_v = np.asarray(ball["initial_velocity"], dtype=np.float64)
        self.reference_momentum = max(1.0, float(self.total_mass * np.linalg.norm(init_v)))

        # Build spring-lattice topology from rest positions
        # Spacing is roughly radius / 5.0 = 0.004 m
        # Connect particles within cutoff distance (e.g., sqrt(3)*spacing ~ 1.75 * spacing)
        spacing = self.radius / 5.0
        cutoff = 1.75 * spacing

        # Bulk and shear moduli
        K_bulk = self.E / (3.0 * (1.0 - 2.0 * self.nu))
        G_shear = self.E / (2.0 * (1.0 + self.nu))
        # Equivalent spring stiffness scaling
        # For a 3D lattice, k ~ E * spacing
        k_base = self.E * spacing * 0.8
        damping_c = 0.02 * np.sqrt(k_base * (self.total_mass / self.N))

        spring_pairs = []
        rest_lengths = []
        stiffnesses = []

        diff = self.rest_pos[:, None, :] - self.rest_pos[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)

        for i in range(self.N):
            for j in range(i + 1, self.N):
                d = dist_matrix[i, j]
                if 1e-6 < d <= cutoff:
                    spring_pairs.append((i, j))
                    rest_lengths.append(d)
                    # Slightly stiffer for diagonal springs to maintain volume
                    stiffnesses.append(k_base * (spacing / d))

        self.spring_indices = np.array(spring_pairs, dtype=np.int64)
        self.spring_rest_len = np.array(rest_lengths, dtype=np.float64)
        self.spring_k = np.array(stiffnesses, dtype=np.float64)
        self.spring_damping = damping_c

        # Set up deformation gradient tracking for each node (Ne = N)
        # For each node i, use nearest 6 neighbors to estimate deformation gradient F
        self.Ne = self.N
        self.elem_neighbors = []
        self.elem_inv_Dm = []

        for i in range(self.N):
            dists = dist_matrix[i]
            # Get 6 closest neighbors (excluding itself)
            neighbor_indices = np.argsort(dists)[1:7]
            self.elem_neighbors.append(neighbor_indices)

            # Reference displacement matrix Dm: 3 x K
            Dm = (self.rest_pos[neighbor_indices] - self.rest_pos[i]).T  # 3 x 6
            # Moore-Penrose pseudo-inverse of Dm: (Dm^T Dm)^-1 Dm^T -> inv_Dm is K x 3
            # F * Dm = Ds => F = Ds * inv_Dm (where Ds is 3 x K, inv_Dm is K x 3 => F is 3 x 3)
            inv_Dm = np.linalg.pinv(Dm)  # 6 x 3
            self.elem_inv_Dm.append(inv_Dm)

        # Contact and time stepping setup
        # Sound speed c = sqrt(E / rho) ~ sqrt(1e7 / 1150) ~ 93 m/s
        # Courant condition: dt < spacing / c ~ 0.004 / 93 ~ 4.3e-5 s
        # Substep dt = 2e-6 s (50 substeps per sample_dt = 1e-4 s)
        self.dt = 2.0e-6
        self.cumulative_impulse = np.zeros((1, 3), dtype=np.float64)
        self.wall_contact_force = np.zeros(3, dtype=np.float64)

    def _compute_forces(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute internal spring forces, gravity, and penalty contact forces with plate."""
        forces = np.zeros_like(self.pos)

        # 1. Gravity
        forces += self.mass[:, None] * self.g[None, :]

        # 2. Spring forces
        if len(self.spring_indices) > 0:
            i = self.spring_indices[:, 0]
            j = self.spring_indices[:, 1]
            p_i = self.pos[i]
            p_j = self.pos[j]
            v_i = self.vel[i]
            v_j = self.vel[j]

            delta_p = p_j - p_i
            dist = np.linalg.norm(delta_p, axis=-1, keepdims=True)
            dist_safe = np.maximum(dist, 1e-12)
            n = delta_p / dist_safe

            # Hookean spring with damping
            stretch = dist.squeeze(-1) - self.spring_rest_len
            f_mag = self.spring_k * stretch

            # Damping along spring direction
            rel_v = v_j - v_i
            v_proj = np.sum(rel_v * n, axis=-1)
            f_damp = self.spring_damping * v_proj

            f_spring = (f_mag + f_damp)[:, None] * n

            np.add.at(forces, i, f_spring)
            np.add.at(forces, j, -f_spring)

        # 3. Contact forces with plate (plate front face at x = 0.0)
        # Ball travels in +x direction. Plate surface is at x = 0.0.
        # Wall is fixed at x >= 0.0. Penalty contact pushes back in -x.
        contact_force_total = np.zeros(3, dtype=np.float64)
        k_contact = 5.0e6  # Contact stiffness (N/m per particle)
        c_contact = 50.0   # Contact damping
        wall_x = 0.0

        penetration = self.pos[:, 0] - wall_x
        penetrating_mask = penetration > 0.0

        if np.any(penetrating_mask):
            p_pen = penetration[penetrating_mask]
            v_pen = self.vel[penetrating_mask, 0]
            # Normal force in -x direction
            fn = -(k_contact * p_pen + c_contact * np.maximum(0.0, v_pen))
            # Only repulsive force
            fn = np.minimum(0.0, fn)
            forces[penetrating_mask, 0] += fn
            contact_force_total[0] = np.sum(fn)

            # Coulomb friction on plate surface (y and z directions)
            mu = 0.2
            v_tan = self.vel[penetrating_mask, 1:3]
            v_tan_norm = np.linalg.norm(v_tan, axis=-1, keepdims=True)
            f_friction_mag = np.minimum(mu * np.abs(fn)[:, None], 200.0 * v_tan_norm)
            f_friction = -f_friction_mag * (v_tan / np.maximum(v_tan_norm, 1e-8))
            forces[penetrating_mask, 1:3] += f_friction
            contact_force_total[1:3] = np.sum(f_friction, axis=0)

        return forces, contact_force_total

    def _step(self, dt: float) -> None:
        """Symplectic Euler / Velocity Verlet time-stepping."""
        forces, contact_force = self._compute_forces()
        self.wall_contact_force = contact_force

        # Acceleration
        acc = forces / self.mass[:, None]

        # Update velocity and position
        self.vel += acc * dt
        self.pos += self.vel * dt

        # Enforce kinematic boundary condition (cannot penetrate deep into rigid wall)
        penetration = self.pos[:, 0]
        deep_pen = penetration > 0.002
        if np.any(deep_pen):
            self.pos[deep_pen, 0] = 0.002
            self.vel[deep_pen, 0] = np.minimum(0.0, self.vel[deep_pen, 0])

        # Accumulate impulse from external forces (gravity + wall contact)
        external_force = np.sum(self.mass[:, None] * self.g[None, :], axis=0) + contact_force
        self.cumulative_impulse[0] += external_force * dt

    def _compute_deformation_gradients(self) -> np.ndarray:
        """Compute deformation gradient F for each element/node."""
        F_all = np.zeros((self.Ne, 3, 3), dtype=np.float64)
        for i in range(self.Ne):
            neighbors = self.elem_neighbors[i]
            Ds = (self.pos[neighbors] - self.pos[i]).T  # 3 x K
            inv_Dm = self.elem_inv_Dm[i]                # K x 3
            F = Ds @ inv_Dm                             # 3 x 3
            F_all[i] = F
        return F_all

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward to target_time
        while self.time < target_time - 1e-12:
            dt = min(self.dt, target_time - self.time)
            self._step(dt)
            self.time += dt

        self.time = target_time

        # Calculate deformation gradient
        F_all = self._compute_deformation_gradients()

        # External forces per domain: Gravity + Contact
        g_force = np.sum(self.mass[:, None] * self.g[None, :], axis=0)
        c_force = self.wall_contact_force
        budget_ext_force = np.zeros((1, 2, 3), dtype=np.float64)
        budget_ext_force[0, 0] = g_force
        budget_ext_force[0, 1] = c_force

        domain_weights = np.ones((1, self.N), dtype=np.float64)

        frame = StateFrame(
            time=float(self.time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "rest_position": self.rest_pos.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.cumulative_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_domain_weights": domain_weights,
                "budget_external_force": budget_ext_force,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "element_deformation_gradient": F_all,
            },
            scalars={
                "radius": float(self.radius),
                "total_mass": float(self.total_mass),
                "reference_momentum": float(self.reference_momentum),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.time,
            "center_of_mass": np.mean(self.pos, axis=0).tolist() if self.pos is not None else None,
            "mean_velocity": np.mean(self.vel, axis=0).tolist() if self.vel is not None else None,
        }

