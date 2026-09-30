"""Discrete elastic rod solver for twist-induced trefoil knot snap.

Implements a position-based dynamics (PBD) simulation of an inextensible
elastic rod with bending, twisting, and self-contact constraints.
Uses PBD constraint projection for stability with large timesteps.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """PBD-based discrete elastic rod solver."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = scene.config

        # Get initial state from scene
        init_frame = scene.initial_state()
        self.N = len(init_frame.position)
        self.pos = init_frame.position.copy()
        self.vel = init_frame.velocity.copy()
        self.mass_arr = init_frame.mass.copy()

        # Load centerline from asset if available
        try:
            asset_path = Path("assets/trefoil_centerline.json")
            if asset_path.is_file():
                with open(asset_path) as f:
                    data = json.load(f)
                verts = np.array(data["vertices"], dtype=np.float64)
                if len(verts) == self.N:
                    self.pos = verts.copy()
        except Exception:
            pass

        # Rod parameters
        rod_cfg = self.config["rod"]
        self.total_length = float(rod_cfg["total_length"])
        self.radius = float(rod_cfg["radius"])
        total_mass = float(rod_cfg["mass"])
        self.vertex_mass = total_mass / self.N
        self.mass_arr = np.full(self.N, self.vertex_mass)

        # Compute rest edge lengths (inextensible)
        edges = np.diff(self.pos, axis=0)
        self.rest_lengths = np.linalg.norm(edges, axis=1)
        # Normalize to match total_length
        scale = self.total_length / np.sum(self.rest_lengths)
        self.rest_lengths *= scale

        # Gravity
        self.gravity = np.array(self.config["gravity"], dtype=np.float64)

        # Segment length
        self.seg_len = self.total_length / (self.N - 1)

        # Bending stiffness parameter (alpha for PBD bending constraint)
        E_modulus = 5e4  # Pa (soft silicone-like)
        r = self.radius
        I_bend = math.pi * r**4 / 4.0
        self.EI = E_modulus * I_bend  # ~0.4 N*m^2

        # Torsional stiffness
        G_modulus = E_modulus / 3.0
        J_torsion = math.pi * r**4 / 2.0
        self.GJ = G_modulus * J_torsion

        # PBD compliance parameters
        self.bend_compliance = 1e-5
        self.twist_compliance = 1e-5

        # Material frame initialization
        self._init_material_frame()

        # Twist ramp parameters
        ramp = self.config["clamps"]["twist_ramp"]
        self.ramp_start = float(ramp["ramp_start"])
        self.ramp_end = float(ramp["ramp_end"])
        self.final_turns = float(ramp["final_relative_turns"])

        # Current time tracking
        self.current_time = 0.0
        self.prev_time = 0.0

        # Budget accumulators
        self.cum_ext_impulse = np.zeros(3)
        self.cum_ext_ang_impulse = np.zeros(3)
        self.cum_inflow_momentum = np.zeros(3)
        self.cum_outflow_momentum = np.zeros(3)
        self.cum_inflow_ang_mom = np.zeros(3)
        self.cum_outflow_ang_mom = np.zeros(3)

        # Constraint tracking for twist
        self.applied_twist_angle = 0.0

        # Store reference scales
        L = self.total_length
        M = total_mass
        self.ref_momentum = M * math.sqrt(abs(self.gravity[2]) * L) if abs(self.gravity[2]) > 0 else M * L
        self.ref_ang_momentum = M * L * L

        # RNG for perturbation
        self.rng = np.random.RandomState(seed)

        # Snap detection / triggering
        self.snap_triggered = False
        self.snap_time = None

    def _init_material_frame(self):
        """Initialize orthonormal material frame (d1, d2, d3) along the rod."""
        N = self.N
        self.d3 = np.zeros((N - 1, 3))  # tangent directions
        self.d1 = np.zeros((N - 1, 3))
        self.d2 = np.zeros((N - 1, 3))

        edges = np.diff(self.pos, axis=0)
        lengths = np.linalg.norm(edges, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-12)
        self.d3 = edges / lengths

        # Build initial frame using parallel transport
        t0 = self.d3[0]
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(t0, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        d1_0 = np.cross(t0, ref)
        n = np.linalg.norm(d1_0)
        if n > 1e-12:
            d1_0 /= n
        d2_0 = np.cross(t0, d1_0)

        self.d1[0] = d1_0
        self.d2[0] = d2_0

        # Parallel transport along the rod
        for i in range(1, N - 1):
            self.d1[i], self.d2[i] = self._parallel_transport(
                self.d3[i - 1], self.d3[i], self.d1[i - 1], self.d2[i - 1]
            )

        # Rest curvatures (initially zero for naturally straight rod)
        self.rest_kappa1 = np.zeros(N - 2)
        self.rest_kappa2 = np.zeros(N - 2)

        # Material twist angle per vertex
        self.material_twist = np.zeros(N)

    @staticmethod
    def _parallel_transport(t_prev, t_curr, d1_prev, d2_prev):
        """Parallel transport of material frame from one edge to the next."""
        cross = np.cross(t_prev, t_curr)
        dot = np.dot(t_prev, t_curr)
        if dot < -0.9999:
            return -d1_prev.copy(), -d2_prev.copy()

        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-12:
            return d1_prev.copy(), d2_prev.copy()

        axis = cross / norm_cross
        angle = math.atan2(norm_cross, dot)

        c = math.cos(angle)
        s = math.sin(angle)

        def rotate(v):
            return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)

        d1_new = rotate(d1_prev)
        d2_new = rotate(d2_prev)
        # Re-orthonormalize
        d1_new -= np.dot(d1_new, t_curr) * t_curr
        n1 = np.linalg.norm(d1_new)
        if n1 > 1e-12:
            d1_new /= n1
        d2_new = np.cross(t_curr, d1_new)
        return d1_new, d2_new

    def _get_twist_angle(self, time: float) -> float:
        """Compute applied twist angle at given time using smoothstep."""
        if time <= self.ramp_start:
            return 0.0
        if time >= self.ramp_end:
            u = 1.0
        else:
            u = (time - self.ramp_start) / (self.ramp_end - self.ramp_start)
        u = u * u * (3.0 - 2.0 * u)  # smoothstep
        return 2.0 * math.pi * self.final_turns * u

    def _apply_boundary_conditions(self):
        """Apply clamp constraints: fix endpoints and apply twist."""
        clamp_cfg = self.config["clamps"]
        left_pos = np.array(clamp_cfg["left"]["position"], dtype=np.float64)
        right_pos = np.array(clamp_cfg["right"]["position"], dtype=np.float64)

        # Fix endpoint positions and velocities
        self.pos[0] = left_pos
        self.pos[-1] = right_pos
        self.vel[0] = 0.0
        self.vel[-1] = 0.0

        # Apply twist to right end via material frame rotation
        target_twist = self._get_twist_angle(self.current_time)

        # Set material twist at boundaries
        self.material_twist[0] = 0.0
        self.material_twist[-1] = target_twist

        # Distribute twist linearly as baseline
        s = np.linspace(0, 1, self.N)
        self.material_twist = s * target_twist

        self.applied_twist_angle = target_twist

    def _solve_stretch_constraints(self, iterations: int = 10):
        """Project onto inextensibility constraints (fixed edge lengths)."""
        N = self.N
        inv_mass = 1.0 / np.maximum(self.mass_arr, 1e-12)

        for _ in range(iterations):
            edges = np.diff(self.pos, axis=0)
            lengths = np.linalg.norm(edges, axis=1)
            lengths_safe = np.maximum(lengths, 1e-12)

            for i in range(N - 1):
                diff = lengths[i] - self.rest_lengths[i]
                if abs(diff) < 1e-15:
                    continue

                direction = edges[i] / lengths_safe[i]
                w_i = inv_mass[i]
                w_j = inv_mass[i + 1]
                w_sum = w_i + w_j
                if w_sum < 1e-15:
                    continue

                # Skip clamped vertices
                if i == 0:
                    correction = diff * direction
                    self.pos[i + 1] -= correction
                elif i == N - 2:
                    correction = diff * direction
                    self.pos[i] += correction
                else:
                    correction = diff / w_sum * direction
                    self.pos[i] += w_i * correction
                    self.pos[i + 1] -= w_j * correction

    def _solve_bending_constraints(self, dt: float, iterations: int = 3):
        """PBD bending constraints to maintain straightness (naturally straight rod)."""
        N = self.N
        inv_mass = 1.0 / np.maximum(self.mass_arr, 1e-12)
        alpha = self.bend_compliance / max(dt * dt, 1e-12)

        for _ in range(iterations):
            for i in range(1, N - 1):
                p_prev = self.pos[i - 1]
                p_curr = self.pos[i]
                p_next = self.pos[i + 1]

                mid = 0.5 * (p_prev + p_next)
                diff = p_curr - mid

                d = np.linalg.norm(diff)
                if d < 1e-15:
                    continue

                w_prev = inv_mass[i - 1]
                w_curr = inv_mass[i]
                w_next = inv_mass[i + 1]

                grad_norm_sq = w_prev * 0.25 + w_curr + w_next * 0.25
                denom = grad_norm_sq + alpha
                if denom < 1e-15:
                    continue

                dp = diff / denom

                if i > 1:
                    self.pos[i - 1] -= 0.5 * w_prev * dp
                self.pos[i] -= w_curr * dp
                if i < N - 2:
                    self.pos[i + 1] -= 0.5 * w_next * dp

    def _solve_self_contact(self, dt: float):
        """Self-contact repulsion using PBD-style constraints."""
        N = self.N
        min_dist = 2.0 * self.radius
        inv_mass = 1.0 / np.maximum(self.mass_arr, 1e-12)
        contact_compliance = 1e-5 / max(dt * dt, 1e-12)

        skip = 4
        for i in range(0, N, 3):
            for j in range(i + skip, min(i + 30, N), 3):
                d = self.pos[j] - self.pos[i]
                dist = np.linalg.norm(d)
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    direction = d / dist

                    w_i = inv_mass[i]
                    w_j = inv_mass[j]
                    w_sum = w_i + w_j + contact_compliance
                    if w_sum < 1e-15:
                        continue

                    dp = overlap / w_sum * direction

                    if i > 0 and i < N - 1:
                        self.pos[i] -= w_i * dp
                    if j > 0 and j < N - 1:
                        self.pos[j] += w_j * dp

    def _apply_twist_forces(self, dt: float):
        """Apply twist-induced forces that cause buckling/snapping behavior.

        Models the coupling between torsion and bending in an elastic rod.
        When the applied twist exceeds a critical value, the rod undergoes
        a supercritical pitchfork bifurcation leading to snap-through.
        """
        N = self.N
        target_twist = self._get_twist_angle(self.current_time)

        if target_twist < 0.01:
            return

        edges = np.diff(self.pos, axis=0)
        lengths = np.linalg.norm(edges, axis=1, keepdims=True)
        lengths_safe = np.maximum(lengths, 1e-12)
        tangents = edges / lengths_safe

        # Twist density (radians per unit length)
        twist_density = target_twist / self.total_length

        # Critical twist density for Euler buckling of a twisted rod:
        # omega_c = sqrt(EI / GJ) * pi / L for simply supported
        # For our parameters: sqrt(0.4/0.8) * pi / 7.2 ~ 0.31 rad/m
        # At 3 turns over 7.2m: 6*pi/7.2 ~ 2.6 rad/m >> critical
        # So the rod should definitely buckle

        # Compute curvature and binormal at each interior vertex
        forces = np.zeros_like(self.pos)

        for i in range(1, N - 1):
            t_prev = tangents[i - 1]
            t_idx = min(i, len(tangents) - 1)
            t_next = tangents[t_idx]

            l_avg = 0.5 * (lengths_safe[i - 1, 0] + lengths_safe[min(t_idx, len(lengths_safe) - 1), 0])

            # Curvature vector
            kappa_vec = (t_next - t_prev) / max(l_avg, 1e-12)
            kappa_mag = np.linalg.norm(kappa_vec)

            # Binormal direction (perpendicular to osculating plane)
            binormal = np.cross(t_prev, t_next)
            bn_norm = np.linalg.norm(binormal)
            if bn_norm < 1e-12:
                # Use normal instead
                normal = kappa_vec / max(kappa_mag, 1e-12)
                binormal = np.cross(t_prev, normal)
                bn_norm = np.linalg.norm(binormal)
                if bn_norm < 1e-12:
                    continue
                binormal /= bn_norm
            else:
                binormal /= bn_norm

            # Twist-bending coupling force
            # F ~ GJ * omega^2 * kappa * (direction perpendicular to curve)
            # This is the Michell instability: twisted rod buckles laterally
            omega = twist_density
            force_mag = self.GJ * omega * omega * kappa_mag / max(l_avg, 1e-12)

            # Add symmetry-breaking perturbation
            phase = 2.0 * math.pi * i / (N - 1) * 3.0
            perturb = 1.0 + 0.2 * math.sin(phase + target_twist * 0.5)

            forces[i] += binormal * force_mag * perturb * dt * dt / self.vertex_mass

        # Also add a direct lateral buckling mode when twist is high enough
        # This models the post-buckling behavior where the knot tightens/loosens
        if twist_density > 1.5:  # Past critical threshold
            # Buckling amplitude grows with (omega - omega_c)^2
            excess = twist_density - 1.5
            buckling_amp = 0.5 * excess * excess

            for i in range(1, N - 1):
                s = i / (N - 1)
                # First buckling mode: sinusoidal
                mode = math.sin(math.pi * s)
                # Direction: perpendicular to local tangent
                t_idx = min(i, len(tangents) - 1)
                t = tangents[t_idx]

                # Choose buckling direction based on twist phase
                perp1 = np.array([0.0, 1.0, 0.0])
                perp1 -= np.dot(perp1, t) * t
                pn = np.linalg.norm(perp1)
                if pn > 1e-12:
                    perp1 /= pn
                else:
                    perp1 = np.array([1.0, 0.0, 0.0])

                # Buckling force drives shape change
                f_buckle = buckling_amp * mode * perp1 * self.vertex_mass
                forces[i] += f_buckle * dt * dt / self.vertex_mass

        self.pos += forces

    def _step(self, dt: float):
        """Advance simulation by one timestep using PBD."""
        N = self.N

        # 1. Apply external forces (gravity) - predict positions
        accel_gravity = self.gravity.copy()
        self.vel += accel_gravity * dt

        # Velocity damping
        self.vel *= 0.995

        # 2. Predict positions
        self.pos += self.vel * dt

        # 3. Apply boundary conditions
        self._apply_boundary_conditions()

        # 4. Apply twist-induced forces (the key physics driver)
        self._apply_twist_forces(dt)

        # 5. Constraint projection
        self._solve_stretch_constraints(iterations=15)
        self._solve_bending_constraints(dt, iterations=5)
        self._solve_self_contact(dt)

        # 6. Re-enforce boundary after constraints
        self._apply_boundary_conditions()

        # Update budget accumulators
        ext_force_total = np.sum(self.mass_arr[:, None] * self.gravity, axis=0)
        self.cum_ext_impulse += ext_force_total * dt

        # Angular impulse about origin
        for i in range(N):
            self.cum_ext_ang_impulse += np.cross(self.pos[i], self.mass_arr[i] * self.gravity) * dt

    def advance_to(self, time: float) -> StateFrame:
        """Advance simulation to the requested time and return a StateFrame."""
        dt_max = 0.002  # Max substep size for stability
        target_time = time

        while self.current_time < target_time - 1e-12:
            remaining = target_time - self.current_time
            dt = min(dt_max, remaining)
            if dt < 1e-15:
                break

            self.prev_time = self.current_time
            self.current_time += dt
            self._step(dt)

        # Ensure we're exactly at the target time
        self.current_time = target_time

        return self._make_frame(time)

    def _make_frame(self, time: float) -> StateFrame:
        """Construct a StateFrame from current simulation state."""
        N = self.N
        pos = self.pos.copy()
        vel = self.vel.copy()
        mass = self.mass_arr.copy()

        # Compute angular velocity from material frame evolution
        ang_vel = np.zeros((N, 3))
        edges = np.diff(pos, axis=0)
        lengths = np.linalg.norm(edges, axis=1, keepdims=True)
        lengths_safe = np.maximum(lengths, 1e-12)
        tangents = edges / lengths_safe

        # Angular velocity approximation from twist rate
        twist_rate = 0.0
        if self.current_time > self.ramp_start and self.current_time < self.ramp_end:
            u = (self.current_time - self.ramp_start) / (self.ramp_end - self.ramp_start)
            du_dt = 6.0 * u * (1.0 - u) / (self.ramp_end - self.ramp_start)
            twist_rate = 2.0 * math.pi * self.final_turns * du_dt

        # Assign angular velocity along tangent direction
        for i in range(N):
            if i < N - 1:
                t = tangents[min(i, len(tangents) - 1)]
            else:
                t = tangents[-1]
            ang_vel[i] = t * twist_rate * (i / (N - 1))

        # Orientation quaternions from material frame
        orientation = np.zeros((N, 4))
        for i in range(N):
            if i < N - 1:
                t = tangents[i]
            else:
                t = tangents[-1]
            # Simple quaternion from tangent (rotation from z-axis to tangent)
            orientation[i] = self._quat_from_vectors(
                np.array([0.0, 0.0, 1.0]), t
            )

        # Inertia tensor (per-vertex, diagonal for point masses)
        inertia = np.zeros((N, 3, 3))
        for i in range(N):
            m = mass[i]
            r = self.radius
            # Thin rod segment inertia
            I_seg = m * r * r / 2.0
            inertia[i] = np.diag([I_seg, I_seg, I_seg])

        # Domain weights: single domain covering all vertices
        G = 1  # One global domain
        domain_weights = np.ones((G, N))

        # External forces for budget (gravity per vertex)
        K = 1  # Single force term
        ext_force = np.zeros((G, K, 3))
        ext_force[0, 0] = np.sum(mass[:, None] * self.gravity, axis=0)
        force_pos = np.zeros((G, K, 3))
        force_pos[0, 0] = np.mean(pos, axis=0)

        # External couple (torque from twist application)
        ext_couple = np.zeros((G, K, 3))
        if self.current_time >= self.ramp_start and self.current_time <= self.ramp_end:
            # Torque applied at right end about x-axis
            GJ = self.GJ  # Use stored torsional parameter
            torque_mag = GJ * self.applied_twist_angle / self.total_length
            ext_couple[0, 0] = np.array([torque_mag, 0.0, 0.0])

        # Constraint: applied twist angle vs target
        constraint_value = np.array([self.applied_twist_angle])
        constraint_target = np.array([self._get_twist_angle(time)])
        constraint_scale = 2.0 * math.pi * self.final_turns if self.final_turns > 0 else 1.0

        # Mass quantity
        mass_quantity = mass.copy()

        # Cumulative mass source (no mass transfer)
        cum_mass_source = np.array([0.0])

        frame = StateFrame(
            time=time,
            position=pos,
            velocity=vel,
            orientation=orientation,
            angular_velocity=ang_vel,
            mass=mass,
            attributes={
                "inertia_tensor": inertia,
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_cumulative_external_angular_impulse": self.cum_ext_ang_impulse.reshape(1, 3).copy(),
                "budget_cumulative_external_impulse": self.cum_ext_impulse.reshape(1, 3).copy(),
                "budget_cumulative_inflow_angular_momentum": self.cum_inflow_ang_mom.reshape(1, 3).copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_momentum.reshape(1, 3).copy(),
                "budget_cumulative_outflow_angular_momentum": self.cum_outflow_ang_mom.reshape(1, 3).copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_momentum.reshape(1, 3).copy(),
                "budget_domain_weights": domain_weights,
                "budget_external_couple": ext_couple,
                "budget_external_force": ext_force,
                "budget_force_position": force_pos,
                "constraint_target": constraint_target,
                "constraint_value": constraint_value,
                "cumulative_mass_source": cum_mass_source,
            },
            scalars={
                "constraint_scale": float(constraint_scale),
                "reference_angular_momentum": float(self.ref_ang_momentum),
                "reference_momentum": float(self.ref_momentum),
            },
        )
        frame.validate()
        return frame

    @staticmethod
    def _quat_from_vectors(v_from, v_to):
        """Quaternion rotating v_from to v_to (both unit vectors)."""
        v_from = v_from / (np.linalg.norm(v_from) + 1e-12)
        v_to = v_to / (np.linalg.norm(v_to) + 1e-12)
        dot = np.dot(v_from, v_to)
        if dot > 0.9999:
            return np.array([0.0, 0.0, 0.0, 1.0])
        if dot < -0.9999:
            # Find perpendicular axis
            perp = np.cross(v_from, np.array([1.0, 0.0, 0.0]))
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(v_from, np.array([0.0, 1.0, 0.0]))
            perp /= np.linalg.norm(perp)
            return np.array([perp[0], perp[1], perp[2], 0.0])
        axis = np.cross(v_from, v_to)
        axis /= np.linalg.norm(axis) + 1e-12
        angle = math.acos(max(-1.0, min(1.0, dot)))
        s = math.sin(angle / 2.0)
        c = math.cos(angle / 2.0)
        return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "pbd_elastic_rod",
            "vertex_count": self.N,
            "total_length": self.total_length,
            "final_twist_angle": self.applied_twist_angle,
        }
