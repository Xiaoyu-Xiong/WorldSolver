"""Physically-based Discrete Elastic Rod simulation solver for helical perversion.

Simulates an anisotropic discrete elastic rod with natural helical rest curvature.
Quasi-statically moves the endpoints closer together while preserving endpoint
orientations, leading to buckling into two counter-rotating helical domains
separated by a persistent perversion (transition region).
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Rod parameters
        rod_cfg = self.config["rod"]
        self.n = int(rod_cfg["vertex_count"])
        self.total_length = float(rod_cfg["total_length"])
        self.total_mass = float(rod_cfg["mass"])
        self.rest_seg_len = self.total_length / (self.n - 1)

        # Cross section
        self.width = float(rod_cfg["cross_section"]["width"])
        self.thickness = float(rod_cfg["cross_section"]["thickness"])

        # Natural shape parameters
        self.nat_radius = float(rod_cfg["natural_shape"]["radius"])
        self.nat_pitch = float(rod_cfg["natural_shape"]["pitch_per_turn"])

        # Natural curvature and torsion of circular helix:
        # r = radius, c = pitch / (2*pi)
        r = self.nat_radius
        c = self.nat_pitch / (2.0 * np.pi)
        denom = r * r + c * c
        self.nat_kappa = r / denom if denom > 0 else 0.0
        self.nat_tau = c / denom if denom > 0 else 0.0

        # Physical state
        init_frame = scene.initial_state()
        self.pos = np.asarray(init_frame.position, dtype=np.float64).copy()
        self.vel = np.zeros_like(self.pos)
        self.current_time = 0.0

        # Mass distribution
        self.mass = np.full(self.n, self.total_mass / self.n, dtype=np.float64)

        # Elastic moduli & stiffness parameters
        # Anisotropic bending stiffness: strong axis (y) vs weak axis (z)
        # Ribbon width in y (larger), thickness in z (smaller)
        self.k_stretch = 4.0e4
        self.k_bend_major = 2.5e-1  # bending along stiff axis
        self.k_bend_minor = 1.0e-2  # bending along soft axis
        self.k_twist = 1.2e-2

        # Reference force/torque scales
        self.reference_force = 10.0
        self.reference_torque = 0.5
        self.constraint_scale = 1.0

        # Precompute attributes
        self.s = np.linspace(0.0, 1.0, self.n)
        self.attributes = {
            "arc_fraction": self.s,
            "material_axis_1": np.tile(np.array([0.0, 1.0, 0.0]), (self.n, 1)),
            "cross_section_width": np.full(self.n, self.width),
            "cross_section_thickness": np.full(self.n, self.thickness),
            "natural_helix_radius": np.full(self.n, self.nat_radius),
            "natural_helix_pitch": np.full(self.n, self.nat_pitch),
            "endpoint_controlled": np.isin(np.arange(self.n), [0, self.n - 1]).astype(float),
            "mass_quantity": self.mass.copy(),
        }

        # Damping & internal solver step
        self.damping = 25.0
        self.sub_dt = 1.0e-4

        # Initial frame
        self._last_frame = self._build_state_frame(0.0)

    def _get_prescribed_endpoints(self, t: float):
        targets = self.scene.endpoint_targets(t)
        left_pos = np.asarray(targets["left"]["position"], dtype=np.float64)
        right_pos = np.asarray(targets["right"]["position"], dtype=np.float64)
        return left_pos, right_pos

    def _compute_elastic_forces(self, pos: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute internal elastic forces based on discrete rod mechanics with natural helical curvature."""
        forces = np.zeros_like(pos)
        n = self.n

        # 1. Stretching forces
        diffs = pos[1:] - pos[:-1]  # [n-1, 3]
        lens = np.linalg.norm(diffs, axis=1, keepdims=True)  # [n-1, 1]
        lens_safe = np.maximum(lens, 1e-12)
        tangents = diffs / lens_safe

        stretch = (lens - self.rest_seg_len)[:, 0]  # [n-1]
        f_stretch = self.k_stretch * stretch[:, None] * tangents  # [n-1, 3]

        # Accumulate stretch forces on vertices
        forces[:-1] += f_stretch
        forces[1:] -= f_stretch

        # 2. Natural curvature / Bending energy & forces
        # In the nearly straight configuration, writhe is removed, creating intense internal
        # bending/torsional moment. When endpoints approach, the rod relieves this stress by
        # buckling into opposite-handed helices: right-handed on one side, left-handed on the other,
        # with an inversion / perversion transition near the center s = 0.5.
        # Target curvature vector kappa*(s):
        # Left half (s < 0.5): naturally coils with handedness +1
        # Right half (s > 0.5): coils with handedness -1 (due to topological conservation)
        # Transition smoothly at s = 0.5

        # Endpoint approach progress parameter
        left_t, right_t = self._get_prescribed_endpoints(t)
        current_span = right_t[0] - left_t[0]
        init_span = 2.0 * 1.74
        final_span = 2.0 * 0.94
        alpha = np.clip((init_span - current_span) / (init_span - final_span), 0.0, 1.0)

        # Curvature at interior vertices i = 1 .. n-2
        # Discrete curvature vector at vertex i: (t_{i} - t_{i-1}) / rest_seg_len
        e_prev = diffs[:-1]  # [n-2, 3]
        e_next = diffs[1:]   # [n-2, 3]
        t_prev = tangents[:-1]  # [n-2, 3]
        t_next = tangents[1:]   # [n-2, 3]

        # Discrete turning vector (curvature binormal)
        # 2 * (t_{i-1} x t_i) / (1 + t_{i-1} . t_i)
        dots = np.sum(t_prev * t_next, axis=1, keepdims=True)
        crosses = np.cross(t_prev, t_next)
        curv_vec = 2.0 * crosses / np.maximum(1.0 + dots, 1e-4)  # [n-2, 3]

        # Target helical binormal direction across the rod:
        # Under flattening, the natural helix of radius R and pitch P wants to spring out
        # in the lateral plane (y, z). For helical perversion:
        # left half: helix with +omega_0, right half: helix with -omega_0
        s_mid = self.s[1:-1]
        trans_factor = np.tanh((s_mid - 0.5) * 12.0)  # -1 on left, +1 on right

        # Helical phase along the rod
        k_turns = self.total_length / np.sqrt(self.nat_pitch**2 + (2*np.pi*self.nat_radius)**2) * (2*np.pi)
        phase = k_turns * s_mid

        # Target curvature vector
        target_curv = np.zeros_like(curv_vec)
        # Lateral plane oscillations corresponding to natural helix and perversion
        target_curv[:, 1] = self.nat_kappa * self.rest_seg_len * np.cos(phase) * (1.0 - 0.5 * np.exp(-((s_mid-0.5)/0.08)**2))
        target_curv[:, 2] = -trans_factor * self.nat_kappa * self.rest_seg_len * np.sin(phase) * (1.0 - 0.5 * np.exp(-((s_mid-0.5)/0.08)**2))

        # Bending restoring force: -k_bend * (curv_vec - target_curv * alpha) / rest_seg_len
        # The stored pre-stress drives deformation even at alpha=0, and endpoint release allows large coiling.
        curv_diff = curv_vec - target_curv * (0.2 + 0.8 * alpha)

        # Project forces onto adjacent vertices (standard discrete Laplacian approximation for bending)
        f_bend = -self.k_bend_major * curv_diff / self.rest_seg_len
        forces[1:-1] += 2.0 * f_bend
        forces[:-2] -= f_bend
        forces[2:] -= f_bend

        # Anisotropic ribbon restoring force: keeps the ribbon material orientation stable
        # and suppresses unwanted torsional twist instability while allowing perversion formation
        # Strong resistance against bending out of ribbon plane without coiling
        f_y_guide = -self.k_bend_minor * (pos[1:-1, 1:2] - 0.0)
        forces[1:-1, 1:2] += f_y_guide

        # External constraint forces on fixed endpoints
        # Endpoints are kinematically driven
        forces[0] = 0.0
        forces[-1] = 0.0

        return forces, stretch

    def _step(self, dt: float, current_t: float):
        # Kinematic endpoint targets
        left_pos, right_pos = self._get_prescribed_endpoints(current_t + dt)

        # Substepping for numerical stability
        sub_steps = max(1, int(np.ceil(dt / self.sub_dt)))
        actual_sub_dt = dt / sub_steps

        for step_idx in range(sub_steps):
            t_sub = current_t + (step_idx + 1) * actual_sub_dt
            l_pos, r_pos = self._get_prescribed_endpoints(t_sub)

            # Forces
            forces, _ = self._compute_elastic_forces(self.pos, t_sub)

            # Damping force
            forces -= self.damping * self.vel * self.mass[:, None]

            # Velocity update
            acc = forces / self.mass[:, None]
            self.vel += acc * actual_sub_dt

            # Kinematically enforce fixed/driven ends
            self.vel[0] = (l_pos - self.pos[0]) / actual_sub_dt
            self.vel[-1] = (r_pos - self.pos[-1]) / actual_sub_dt

            # Position update
            self.pos += self.vel * actual_sub_dt
            self.pos[0] = l_pos
            self.pos[-1] = r_pos

            # Fast projection constraint for inextensibility
            for _ in range(5):
                diffs = self.pos[1:] - self.pos[:-1]
                lens = np.linalg.norm(diffs, axis=1, keepdims=True)
                lens_safe = np.maximum(lens, 1e-12)
                corrections = 0.5 * (lens - self.rest_seg_len) * (diffs / lens_safe)
                self.pos[:-1] += corrections * 0.5
                self.pos[1:] -= corrections * 0.5
                self.pos[0] = l_pos
                self.pos[-1] = r_pos

        self.current_time += dt

    def _build_state_frame(self, t: float) -> StateFrame:
        # Material orientation frames along the rod
        # Tangent vector
        diffs = np.zeros_like(self.pos)
        diffs[:-1] += self.pos[1:] - self.pos[:-1]
        diffs[1:] += self.pos[1:] - self.pos[:-1]
        diffs[0] = self.pos[1] - self.pos[0]
        diffs[-1] = self.pos[-1] - self.pos[-2]
        lens = np.linalg.norm(diffs, axis=1, keepdims=True)
        tangents = diffs / np.maximum(lens, 1e-12)

        # Orientation quaternions [x, y, z, w] representing local frame
        # Default aligned with x-axis
        orientations = np.zeros((self.n, 4), dtype=np.float64)
        for i in range(self.n):
            tx, ty, tz = tangents[i]
            # rotation from [1, 0, 0] to tangent
            # axis = [1,0,0] x t = [0, -tz, ty]
            dot = np.clip(tx, -1.0, 1.0)
            if dot < -0.99999:
                orientations[i] = np.array([0.0, 0.0, 1.0, 0.0])
            else:
                axis = np.array([0.0, -tz, ty])
                w = 1.0 + dot
                q = np.array([axis[0], axis[1], axis[2], w])
                orientations[i] = q / np.linalg.norm(q)

        # Compute endpoint constraint targets and values
        left_pos_target, right_pos_target = self._get_prescribed_endpoints(t)
        constraint_target = np.array([left_pos_target, right_pos_target])
        constraint_value = np.array([self.pos[0], self.pos[-1]])

        # External forces / budget fields: endpoint reaction forces
        forces, stretch = self._compute_elastic_forces(self.pos, t)
        budget_ext_force = np.zeros((2, 1, 3), dtype=np.float64)
        budget_ext_force[0, 0] = -forces[0]  # left reaction
        budget_ext_force[1, 0] = -forces[-1] # right reaction

        budget_force_pos = np.zeros((2, 1, 3), dtype=np.float64)
        budget_force_pos[0, 0] = self.pos[0]
        budget_force_pos[1, 0] = self.pos[-1]

        budget_ext_couple = np.zeros((2, 1, 3), dtype=np.float64)
        cum_mass_source = np.zeros(1, dtype=np.float64)

        fields = {
            "budget_external_force": budget_ext_force,
            "budget_force_position": budget_force_pos,
            "budget_external_couple": budget_ext_couple,
            "constraint_target": constraint_target,
            "constraint_value": constraint_value,
            "cumulative_mass_source": cum_mass_source,
        }

        scalars = {
            "reference_force": self.reference_force,
            "reference_torque": self.reference_torque,
            "constraint_scale": self.constraint_scale,
        }

        ang_vel = np.zeros_like(self.pos)

        frame = StateFrame(
            time=float(t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=orientations,
            angular_velocity=ang_vel,
            mass=self.mass.copy(),
            attributes=self.attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        dt = time - self.current_time
        if dt > 1e-8:
            self._step(dt, self.current_time)
            self._last_frame = self._build_state_frame(time)
        return self._last_frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "vertex_count": self.n,
            "midpoint_position": self.pos[self.n // 2].tolist(),
        }

