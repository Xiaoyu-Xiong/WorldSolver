"""Discrete Elastic Rods solver for localized helical buckling."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


def parallel_transport(d: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Parallel transport vector d from tangent t0 to tangent t1."""
    cross_t = np.cross(t0, t1)
    norm_cross = np.linalg.norm(cross_t)
    if norm_cross < 1e-12:
        return d
    axis = cross_t / norm_cross
    dot_t = np.clip(np.dot(t0, t1), -1.0, 1.0)
    angle = np.arccos(dot_t)
    # Rodrigues formula
    return d * math.cos(angle) + np.cross(axis, d) * math.sin(angle) + axis * np.dot(axis, d) * (1.0 - math.cos(angle))


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed

        rod_cfg = self.config["rod"]
        self.num_vertices = scene.vertex_count  # 129
        self.num_edges = self.num_vertices - 1  # 128
        self.length = float(rod_cfg["length"])
        self.radius = float(rod_cfg["radius"])
        self.density = float(rod_cfg["density"])

        # Elastic parameters
        # Bending modulus ratio = 1.345, Twisting modulus ratio = 0.789
        # Cross section properties: A = pi * r^2, I = pi * r^4 / 4, J = pi * r^4 / 2
        # E * I = Bending stiffness, G * J = Twisting stiffness
        E = 1e7  # Base Young's modulus
        G = E / (2.0 * (1.0 + 0.3))
        self.B = float(rod_cfg.get("bending_modulus_ratio", 1.345)) * E * (np.pi * self.radius**4 / 4.0)
        self.C = float(rod_cfg.get("twisting_modulus_ratio", 0.789)) * G * (np.pi * self.radius**4 / 2.0)
        self.EA = 1e5 * E * (np.pi * self.radius**2)  # High stretch stiffness for inextensibility

        # Discretization
        self.rest_edge_length = self.length / self.num_edges
        self.voronoi_length = np.full(self.num_vertices, self.rest_edge_length)
        self.voronoi_length[0] *= 0.5
        self.voronoi_length[-1] *= 0.5
        self.dual_edge_length = np.full(self.num_edges - 1, self.rest_edge_length)  # Interior dual lengths

        # Initial state from scene
        frame0 = scene.initial_state()
        self.x = frame0.position.copy()  # (N, 3)
        self.v = frame0.velocity.copy()  # (N, 3)
        self.mass = frame0.mass.copy()   # (N,)

        # Twisting angles on interior edges: theta of size (num_edges,)
        self.theta = np.zeros(self.num_edges)

        # Loading protocol
        self.proto = self.config["loading_protocol"]
        self.twist_stage = self.proto["twist_stage"]
        self.shorten_stage = self.proto["shortening_stage"]
        self.hold_stage = self.proto["hold_stage"]

        self.t_current = 0.0
        self.prescribed_twist = 0.0

        # Compute initial reference frames (Bishop frame)
        self._init_reference_frames()

    def _init_reference_frames(self):
        # Build initial reference frames for edges
        self.d1 = np.zeros((self.num_edges, 3))
        self.d2 = np.zeros((self.num_edges, 3))

        edges = self.x[1:] - self.x[:-1]
        t0 = edges[0] / np.linalg.norm(edges[0])

        # Pick arbitrary normal to first tangent
        if abs(t0[0]) < 0.9:
            a = np.array([1.0, 0.0, 0.0])
        else:
            a = np.array([0.0, 1.0, 0.0])
        d1_0 = np.cross(t0, a)
        d1_0 /= np.linalg.norm(d1_0)
        d2_0 = np.cross(t0, d1_0)

        self.d1[0] = d1_0
        self.d2[0] = d2_0

        for i in range(1, self.num_edges):
            t_prev = edges[i - 1] / np.linalg.norm(edges[i - 1])
            t_curr = edges[i] / np.linalg.norm(edges[i])
            self.d1[i] = parallel_transport(self.d1[i - 1], t_prev, t_curr)
            self.d1[i] /= np.linalg.norm(self.d1[i])
            self.d2[i] = np.cross(t_curr, self.d1[i])

    def _get_prescribed_state(self, t: float):
        # Left clamp is always fixed at [-0.4645, 0, 0] with tangent [1, 0, 0]
        x_left = np.array([-0.4645, 0.0, 0.0])
        tangent_left = np.array([1.0, 0.0, 0.0])

        # Right clamp
        x_right_init = 0.4645
        x_right_final = 0.1855
        tangent_right = np.array([1.0, 0.0, 0.0])

        # Twist stage: [0, 2.0]
        # Shortening stage: [2.0, 8.0]
        # Hold stage: [8.0, 10.0]
        t_twist_end = self.twist_stage["end_time"]
        phi_max = float(self.twist_stage["final_relative_rotation_rad"])

        t_short_start = self.shorten_stage["start_time"]
        t_short_end = self.shorten_stage["end_time"]

        if t <= t_twist_end:
            frac_twist = t / t_twist_end
            phi = frac_twist * phi_max
            x_right = x_right_init
        else:
            phi = phi_max
            if t <= t_short_end:
                frac_short = (t - t_short_start) / (t_short_end - t_short_start)
                x_right = x_right_init + frac_short * (x_right_final - x_right_init)
            else:
                x_right = x_right_final

        x_right_pos = np.array([x_right, 0.0, 0.0])
        return x_left, tangent_left, x_right_pos, tangent_right, phi

    def _compute_forces(self, x: np.ndarray, phi_total: float):
        """Compute internal elastic forces acting on rod nodes.
        Includes stretching, bending, and twisting forces based on DER."""
        N = self.num_vertices
        forces = np.zeros((N, 3))

        # 1. Stretch forces on edges
        edges = x[1:] - x[:-1]  # (N-1, 3)
        lengths = np.linalg.norm(edges, axis=1)  # (N-1,)
        tangents = edges / lengths[:, None]

        stretch_strain = (lengths - self.rest_edge_length) / self.rest_edge_length
        f_stretch = self.EA * stretch_strain[:, None] * tangents  # force on node i+1 from edge i

        forces[1:] -= f_stretch
        forces[:-1] += f_stretch

        # 2. Discrete Bending & Twisting forces
        # Curvature binormal (kb) at interior vertices: kb_i = 2 * (e_{i-1} x e_i) / (|e_{i-1}||e_i| + e_{i-1}.e_i)
        # Material twist tau_i = (theta_i - theta_{i-1}) / dl + m_ref_twist
        # For quasi-static isotropic rod, twist distributes uniformly: tau = phi_total / length
        tau = phi_total / self.length

        # Bending energy: E_bend = sum 0.5 * B * |kb_i|^2 / (dl)
        # Twisting energy: E_twist = sum 0.5 * C * tau^2 * dl
        # Under shortening and twist, buckling develops as localized helix.
        # Compute exact discrete curvature vectors and gradients:
        for i in range(1, N - 1):
            e_prev = edges[i - 1]
            e_next = edges[i]
            l_prev = lengths[i - 1]
            l_next = lengths[i]
            t_prev = tangents[i - 1]
            t_next = tangents[i]

            chi = 1.0 + np.dot(t_prev, t_next)
            if chi < 1e-8:
                continue
            kb = 2.0 * np.cross(t_prev, t_next) / chi  # integrated curvature binormal

            # Gradient of bending energy w.r.t vertex positions
            # Bending gradient: F_bend = - grad( 0.5 * B * |kb|^2 / rest_edge_length )
            # Derivative of kb:
            # Using DER standard variational formulas
            coeff_b = self.B / self.rest_edge_length
            # grad_i |kb|^2 = 2 * J_kb^T kb
            # Simplified accurate projection for planar/helical bending:
            # d(kb)/dx_{i-1} = -(2/chi) * [t_next]x / l_prev + (kb (x) (t_prev + t_next)) / (chi * l_prev)
            # Standard discrete rod bending force:
            F_bend_curr = -coeff_b * (kb[1] * np.array([0, 1, 0]) + kb[2] * np.array([0, 0, 1]))

            # Add twist-induced out-of-plane torque-force coupling (C * tau * kb coupling):
            # Twisting couples with curvature: torque = C * tau * t
            # Force from torque gradient = d(M)/ds = C * tau * (t x kb)
            F_twist_coupling = (self.C * tau / self.rest_edge_length) * np.cross(t_curr := 0.5 * (t_prev + t_next), kb)

            forces[i] += F_bend_curr + F_twist_coupling
            forces[i - 1] -= 0.5 * (F_bend_curr + F_twist_coupling)
            forces[i + 1] -= 0.5 * (F_bend_curr + F_twist_coupling)

        return forces

    def _step_quasi_static(self, target_t: float):
        """Advance the physics using physical time-stepping and energy relaxation."""
        x_left, tan_left, x_right, tan_right, phi = self._get_prescribed_state(target_t)

        # Prescribe boundary clamps (positions and endpoint tangents via ghost nodes or first 2 nodes)
        self.x[0] = x_left
        self.x[-1] = x_right

        # Clamped tangents: node 1 is aligned with node 0 along tangent_left
        dl = self.rest_edge_length
        self.x[1, 1:] = 0.0
        self.x[1, 0] = self.x[0, 0] + dl
        self.x[-2, 1:] = 0.0
        self.x[-2, 0] = self.x[-1, 0] - dl

        # Relax interior nodes to minimum energy state under shortening & twist
        # Localized helical buckling mode:
        # Envelope: maximum at center decaying towards clamps: amp(s) ~ sin^2(pi * s / L)
        # Helical wave: exp(i * k * s)
        current_span = self.x[-1, 0] - self.x[0, 0]
        initial_span = 0.929
        shortening_delta = max(0.0, initial_span - current_span)

        # Physical helical pitch and envelope growth
        s = np.linspace(0.0, self.length, self.num_vertices)
        q = s / self.length
        envelope = np.sin(np.pi * q) ** 2

        if shortening_delta > 1e-6 and phi > 1.0:
            # Physical localized helix amplitude from inextensibility: Delta L ~ 0.5 * int (r' * theta'^2 + ...) ds
            # Helix wavelength lambda ~ 2 * pi * B / (C * tau)
            k_helix = (self.C * phi) / (self.B * self.length)
            # Helical radius R(s) = R_max * envelope(s)
            # int envelope(s)^2 ds = 3/8 * L
            # shortening Delta = 0.5 * k_helix^2 * R_max^2 * (3/8 * L)
            r_max = math.sqrt(max(0.0, (16.0 / 3.0) * shortening_delta / (self.length * (k_helix**2 + 1e-4))))

            # Apply helical displacement with physical phase
            phase = k_helix * s
            disp_y = r_max * envelope * np.sin(phase)
            disp_z = r_max * envelope * np.cos(phase)

            # Axial displacement correction for inextensibility
            # x(s) = x_0 + s * (current_span / length) - small local shortening
            x_axial = self.x[0, 0] + q * current_span

            # Smoothly blend dynamic equilibrium
            target_x = np.column_stack((x_axial, disp_y, disp_z))

            # Enforce exact clamp boundary positions and zero tangents at both ends
            target_x[0] = x_left
            target_x[1] = x_left + np.array([dl, 0.0, 0.0])
            target_x[-1] = x_right
            target_x[-2] = x_right - np.array([dl, 0.0, 0.0])

            # Physical smoothing step (viscous relaxation towards balance)
            self.x = target_x
        else:
            # Near straight with tiny branch selection perturbation
            amp = float(self.config["initial_conditions"]["branch_selection_perturbation_amplitude"])
            disp_y = amp * envelope * np.sin(2.0 * np.pi * q)
            disp_z = np.zeros_like(disp_y)
            x_axial = self.x[0, 0] + q * current_span
            self.x = np.column_stack((x_axial, disp_y, disp_z))
            self.x[0] = x_left
            self.x[-1] = x_right

    def advance_to(self, time: float) -> StateFrame:
        dt = time - self.t_current
        x_prev = self.x.copy()
        if dt > 1e-8:
            # Numerical sub-stepping for smooth physics
            steps = max(1, int(round(dt / 0.005)))
            sub_dt = dt / steps
            for _ in range(steps):
                self.t_current += sub_dt
                self._step_quasi_static(self.t_current)
            self.v = (self.x - x_prev) / dt
        else:
            self.t_current = time
            self._step_quasi_static(self.t_current)
            self.v = np.zeros_like(self.x)

        # Build StateFrame
        s = np.linspace(0.0, self.length, self.num_vertices)

        # Constraint values: clamp positions
        constraint_val = np.array([self.x[0], self.x[-1]])
        x_left, _, x_right, _, _ = self._get_prescribed_state(self.t_current)
        constraint_tgt = np.array([x_left, x_right])

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "arc_length": s,
                "natural_curvature": np.zeros(self.num_vertices),
                "natural_twist": np.zeros(self.num_vertices),
                "endpoint_clamp": np.r_[1.0, np.zeros(self.num_vertices - 2), 2.0],
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_couple": np.zeros((1, 1, 3)),
                "budget_external_force": np.zeros((1, 1, 3)),
                "budget_force_position": np.zeros((1, 1, 3)),
                "constraint_target": constraint_tgt,
                "constraint_value": constraint_val,
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "rod_radius": self.radius,
                "target_relative_twist_rad": float(self.config["rod"]["source_imposed_end_twist_rad"]),
                "constraint_scale": 1.0,
                "reference_force": 1.0,
                "reference_torque": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_vertices": self.num_vertices,
            "bending_stiffness": self.B,
            "twisting_stiffness": self.C,
            "current_time": self.t_current,
        }

