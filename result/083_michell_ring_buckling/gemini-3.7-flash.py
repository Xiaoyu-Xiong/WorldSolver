"""Physical solver for Michell buckling of a pre-twisted elastic ring."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed

        # Rod configuration
        rod_cfg = self.config["rod"]
        self.n = int(rod_cfg["vertex_count"])
        self.R = float(rod_cfg["radius"])
        self.L = float(rod_cfg["length"])
        self.r_cross = float(rod_cfg["cross_section_radius"])
        self.density = float(rod_cfg["density"])
        self.ds = self.L / self.n

        # Cross section properties
        self.A = np.pi * self.r_cross**2
        self.m_node = self.density * self.A * self.ds
        self.mass = np.full(self.n, self.m_node)

        # Stiffness parameters
        # In DER, bending stiffness B = alpha, twisting stiffness C = beta
        # (or alpha * (pi/4)*r^4 * E, but config provides bending_stiffness_alpha and twisting_stiffness_beta)
        self.alpha = float(rod_cfg.get("bending_stiffness_alpha", 1.0))
        self.beta = float(rod_cfg.get("twisting_stiffness_beta", 1.0))

        # Initial conditions
        init_cfg = self.config["initial_conditions"]
        self.winding = int(init_cfg["material_frame_winding_number"])
        self.stored_twist = float(init_cfg["stored_total_twist_rad"])

        # Initial state from scene
        initial_frame = scene.initial_state()
        self.pos = np.array(initial_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(initial_frame.velocity, dtype=np.float64, copy=True)

        self.time = 0.0
        self.history = {}

        # Precompute reference / invariant quantities
        # Budget domains
        self.num_domains = 1
        self.domain_weights = np.ones((1, self.n), dtype=np.float64)

        # Reference scales
        # Characteristic scales based on total mass and ring geometry/dynamics
        self.total_mass = np.sum(self.mass)
        # reference momentum: m_total * v_char
        # characteristic velocity ~ sqrt(B / (m_total * R))
        v_char = np.sqrt(self.alpha / (self.total_mass * self.R + 1e-6)) + 0.1
        self.ref_momentum = float(self.total_mass * v_char)
        self.ref_angular_momentum = float(self.total_mass * v_char * self.R)

        # Internal sim step
        self.sub_dt = 1e-4  # stable sub-stepping dt
        self.damping = 0.05  # slight numerical/physical damping for relaxation

    def _compute_forces(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Discrete Elastic Rods forces for a closed pre-twisted ring."""
        n = self.n
        forces = np.zeros_like(pos)

        # 1. Edge vectors and lengths: e_i = pos_{i+1} - pos_i
        e = np.roll(pos, -1, axis=0) - pos  # shape (n, 3)
        l = np.linalg.norm(e, axis=1, keepdims=True)  # shape (n, 1)
        l_safe = np.maximum(l, 1e-12)
        t = e / l_safe  # unit tangents (n, 3)

        # 1a. Inextensibility / stretching spring forces
        # Strain epsilon = (l - ds) / ds
        k_stretch = 2.0e5 * self.alpha / (self.ds**2)  # high stretching stiffness
        stretch_force_mag = k_stretch * (l - self.ds)  # (n, 1)
        f_stretch_edge = stretch_force_mag * t  # (n, 3)
        # f_stretch on node i is +f_stretch_edge[i] - f_stretch_edge[i-1]
        forces += f_stretch_edge - np.roll(f_stretch_edge, 1, axis=0)

        # 2. Curvature binormal and bending forces
        # At node i, incoming tangent t_{i-1} = roll(t, 1), outgoing tangent t_i = t
        t_prev = np.roll(t, 1, axis=0)
        e_prev = np.roll(e, 1, axis=0)
        l_prev = np.roll(l_safe, 1, axis=0)

        # Discrete curvature binormal (kb)_i = 2 * (t_{i-1} x t_i) / (1 + t_{i-1} . t_i)
        cross_t = np.cross(t_prev, t)  # (n, 3)
        dot_t = np.sum(t_prev * t, axis=1, keepdims=True)  # (n, 1)
        denom = np.maximum(1.0 + dot_t, 1e-8)
        kb = 2.0 * cross_t / denom  # (n, 3)
        kappa_sq = np.sum(kb * kb, axis=1, keepdims=True)  # |kb|^2

        # Dual length associated with node i: l_dual_i = (l_{i-1} + l_i) / 2
        l_dual = 0.5 * (l_prev + l_safe)

        # Bending energy gradient forces: E_bend = sum_i alpha * |kb_i|^2 / (2 * l_dual_i)
        # Force on node i and neighbors from bending
        # Standard DER gradient of (kb)_i with respect to x_{i-1}, x_i, x_{i+1}:
        # Using the standard Discrete Elastic Rods formulation (Bergou et al. 2008 / 2010):
        # Grad_{x_{i-1}} E_bend = ...
        # For simplicity and robust stability, we compute the bending forces using the curvature vectors:
        f_bend = np.zeros_like(pos)
        coeff_bend = self.alpha / l_dual  # (n, 1)

        # Variation of curvature binormal kb_i w.r.t positions:
        # J_prev = (2 [t_i]_x + kb_i (t_i)^T) / (l_{i-1} (1 + t_{i-1}.t_i))
        # J_next = (2 [t_{i-1}]_x - kb_i (t_{i-1})^T) / (l_i (1 + t_{i-1}.t_i))
        # f_{i-1} -= coeff * J_prev^T kb_i, f_{i+1} -= coeff * J_next^T kb_i, f_i += coeff * (J_prev^T + J_next^T) kb_i
        # Let's write the exact vector formulas:
        # For a vector v = coeff_bend * kb:
        v = coeff_bend * kb  # (n, 3)
        # J_prev^T v = (2 (t_i x v) + (kb . v) t_i) / (l_prev * denom)
        kb_dot_v = np.sum(kb * v, axis=1, keepdims=True)  # (n, 1)
        grad_prev = (2.0 * np.cross(t, v) + kb_dot_v * t) / (l_prev * denom)
        # J_next^T v = (-2 (t_prev x v) - (kb . v) t_prev) / (l_safe * denom)
        grad_next = (-2.0 * np.cross(t_prev, v) - kb_dot_v * t_prev) / (l_safe * denom)
        grad_curr = -(grad_prev + grad_next)

        # Accumulate bending forces:
        # For vertex k, it acts as (i-1) for node k+1, (i+1) for node k-1, and (i) for node k
        f_bend += -grad_curr  # from node i
        f_bend += -np.roll(grad_prev, -1, axis=0)  # from node i+1 (where current node is i-1)
        f_bend += -np.roll(grad_next, 1, axis=0)   # from node i-1 (where current node is i+1)
        forces += f_bend

        # 3. Michell Buckling Twist-Coupling Force
        # In Michell instability of a pre-twisted ring, total stored twist is tau_0 = 2*pi*winding.
        # As the ring deforms, writhe Wr changes, and Twist Tw = Tw_0 - Wr.
        # The twist energy is E_twist = 0.5 * beta * (Tw_total)^2 / L
        # The variation of Writhe (or total twist) with respect to centerline position x_k produces
        # a transverse force proportional to beta * tau_0 * (t x d^2 x / ds^2) or beta * tau_0 * (t_i x kb_i / ds)
        # According to Michell (1889) and Goriely & Tabor / Discrete Elastic Rods:
        # The twisting moment M_t = beta * (stored_twist / L)
        # The resulting distributed out-of-plane buckling force is f_twist = M_t * (t x kappa) = M_t * (t x (kb / ds))
        # This exact term couples the in-plane curvature to out-of-plane buckling forces!
        tau_density = self.stored_twist / self.L
        M_twist = self.beta * tau_density  # constant twisting moment density
        # At node i, the local cross product of tangent and curvature binormal:
        # t_mid = 0.5 * (t_prev + t)
        t_mid = 0.5 * (t_prev + t)
        t_mid_norm = np.maximum(np.linalg.norm(t_mid, axis=1, keepdims=True), 1e-8)
        t_mid = t_mid / t_mid_norm
        # Force per node: F_twist_i = M_twist * (t_mid x kb_i)
        f_twist_node = M_twist * np.cross(t_mid, kb)
        forces += f_twist_node

        # 4. Self-contact repulsion (prevent self-penetration if loops approach)
        r_contact = 2.0 * self.r_cross
        r_contact_sq = r_contact**2
        k_contact = 5.0e4 * self.alpha
        # Broad-phase / direct check since N=128 is small
        # Vectorized pairwise distance
        diff = pos[:, None, :] - pos[None, :, :]  # (N, N, 3)
        dist_sq = np.sum(diff**2, axis=-1)  # (N, N)
        # Exclude neighboring vertices along the ring (modulo N distance <= 3)
        idx = np.arange(n)
        ring_dist = np.abs(idx[:, None] - idx[None, :])
        ring_dist = np.minimum(ring_dist, n - ring_dist)
        mask = (ring_dist > 4) & (dist_sq < r_contact_sq) & (dist_sq > 1e-12)

        if np.any(mask):
            dist = np.sqrt(dist_sq[mask])
            overlap = r_contact - dist
            f_mag = k_contact * overlap / dist  # (M,)
            contact_vecs = diff[mask] * f_mag[:, None]  # (M, 3)
            # Accumulate
            row_indices, _ = np.nonzero(mask)
            np.add.at(forces, row_indices, contact_vecs)

        # 5. Damping forces (proportional to velocity)
        forces -= self.damping * self.m_node * vel

        # Compute angular velocity at nodes for StateFrame
        # omega_i = (pos_i x vel_i) / |pos_i|^2 + local rotation
        ang_vel = np.cross(pos, vel) / np.maximum(np.sum(pos**2, axis=1, keepdims=True), 1e-6)

        return forces, stretch_force_mag.squeeze(-1), ang_vel

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward until target_time
        while self.time < target_time - 1e-9:
            dt = min(self.sub_dt, target_time - self.time)

            # Semi-implicit Euler / Verlet
            forces, stretch_mag, ang_vel = self._compute_forces(self.pos, self.vel)
            acc = forces / self.mass[:, None]
            self.vel += acc * dt
            self.pos += self.vel * dt
            self.time += dt

        # Final force evaluation at target_time for frame outputs
        forces, stretch_mag, ang_vel = self._compute_forces(self.pos, self.vel)

        # Inextensibility constraint residuals
        e = np.roll(self.pos, -1, axis=0) - self.pos
        edge_lengths = np.linalg.norm(e, axis=1)
        constraint_val = edge_lengths  # actual edge lengths
        constraint_tgt = np.full(self.n, self.ds)  # target edge length ds

        # Budget fields (global domain G=1)
        # External forces are 0 (free relaxation)
        # K=1 term for external force / couples
        budget_ext_force = np.zeros((1, 1, 3), dtype=np.float64)
        budget_force_pos = np.zeros((1, 1, 3), dtype=np.float64)
        budget_ext_couple = np.zeros((1, 1, 3), dtype=np.float64)

        budget_cum_ext_impulse = np.zeros((1, 3), dtype=np.float64)
        budget_cum_ext_ang_impulse = np.zeros((1, 3), dtype=np.float64)
        budget_cum_inflow_mom = np.zeros((1, 3), dtype=np.float64)
        budget_cum_inflow_ang_mom = np.zeros((1, 3), dtype=np.float64)
        budget_cum_outflow_mom = np.zeros((1, 3), dtype=np.float64)
        budget_cum_outflow_ang_mom = np.zeros((1, 3), dtype=np.float64)

        # Material mass quantity
        mass_quantity = self.mass.copy()  # [N]

        # Inertia tensor per node: I_particle = 0 for point mass resolution
        inertia_tensor = np.zeros((self.n, 3, 3), dtype=np.float64)

        # Cumulative mass source [K] with K=1
        cumulative_mass_source = np.zeros(1, dtype=np.float64)

        # Phase angle for material attributes
        phase = 2.0 * np.pi * np.arange(self.n) / self.n

        frame = StateFrame(
            time=float(target_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            angular_velocity=ang_vel,
            mass=self.mass.copy(),
            attributes={
                "inertia_tensor": inertia_tensor,
                "mass_quantity": mass_quantity,
                "arc_length": self.ds * np.arange(self.n),
                "closed_curve_index": np.arange(self.n, dtype=np.float64),
                "natural_curvature": np.zeros(self.n),
                "natural_twist": np.zeros(self.n),
                "material_frame_angle": self.winding * phase,
                "stored_twist_density": np.full(self.n, self.stored_twist / self.L),
            },
            fields={
                "budget_cumulative_external_angular_impulse": budget_cum_ext_ang_impulse,
                "budget_cumulative_external_impulse": budget_cum_ext_impulse,
                "budget_cumulative_inflow_angular_momentum": budget_cum_inflow_ang_mom,
                "budget_cumulative_inflow_momentum": budget_cum_inflow_mom,
                "budget_cumulative_outflow_angular_momentum": budget_cum_outflow_ang_mom,
                "budget_cumulative_outflow_momentum": budget_cum_outflow_mom,
                "budget_domain_weights": self.domain_weights,
                "budget_external_couple": budget_ext_couple,
                "budget_external_force": budget_ext_force,
                "budget_force_position": budget_force_pos,
                "constraint_target": constraint_tgt,
                "constraint_value": constraint_val,
                "cumulative_mass_source": cumulative_mass_source,
            },
            scalars={
                "constraint_scale": self.ds,
                "reference_angular_momentum": self.ref_angular_momentum,
                "reference_momentum": self.ref_momentum,
                "rod_radius": self.r_cross,
                "stored_total_twist_rad": self.stored_twist,
                "material_frame_winding_number": float(self.winding),
                "closed_ring": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "vertex_count": self.n,
            "mean_edge_length": float(np.mean(np.linalg.norm(np.roll(self.pos, -1, axis=0) - self.pos, axis=1))),
        }

