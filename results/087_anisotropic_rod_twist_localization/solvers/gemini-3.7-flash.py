"""Physical solver for twist localization in a V-shaped anisotropic rod.

Based on Discrete Elastic Rods (Bergou et al., 2008 / 2010), modeling anisotropic
bending and twisting along a V-shaped rod.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        rod_cfg = self.config["rod"]
        bc_cfg = self.config["boundary_conditions"]

        self.n_v = int(rod_cfg["vertex_count"])
        self.apex_idx = int(rod_cfg["apex_vertex"])
        self.n_e = self.n_v - 1

        # Initial positions and geometry
        init_frame = scene.initial_state()
        self.x0 = np.copy(init_frame.position)
        self.x = np.copy(self.x0)
        self.v = np.zeros_like(self.x)
        self.mass = np.copy(init_frame.mass)
        self.arc_length = np.copy(init_frame.attributes["arc_length"])
        self.w = float(rod_cfg["cross_section_width"])
        self.h = float(rod_cfg["cross_section_thickness"])
        self.density = float(rod_cfg["density"])

        # Rest edge lengths and voronoi lengths
        edge_vecs0 = np.diff(self.x0, axis=0)
        self.rest_edge_lengths = np.linalg.norm(edge_vecs0, axis=1)
        self.voronoi_lengths = np.empty(self.n_v)
        self.voronoi_lengths[0] = 0.5 * self.rest_edge_lengths[0]
        self.voronoi_lengths[-1] = 0.5 * self.rest_edge_lengths[-1]
        self.voronoi_lengths[1:-1] = 0.5 * (self.rest_edge_lengths[:-1] + self.rest_edge_lengths[1:])

        # In the reference V-shape:
        # The flat ribbon width w is in the out-of-plane direction (y-axis).
        # Thickness h is in the V-plane perpendicular to the tangent.
        # Bending stiffness about y (out-of-plane, major bending): B1 ~ E * I1 = E * (h * w^3 / 12)
        # Bending stiffness about in-plane normal (minor bending): B2 ~ E * I2 = E * (w * h^3 / 12)
        # Torsional stiffness: C ~ G * J = G * (w * h^3 / 3) (for thin ribbon w >> h)
        # Ratio B1 / C >> 1. When twisted at the end, twisting the straight arm costs less energy than
        # bending across the apex if the bend is aligned with the stiff axis, leading to twist localization!
        # Specifically, at the sharp bend (apex), the curvature vector aligns with the ribbon normal.
        # Twisting before the bend rotates the ribbon normal, which couples twist into hard out-of-plane bending.
        # Thus twist is trapped in the driven arm until twist energy builds up enough to overcome the barrier.

        # Elastic parameters
        E = 1.0e7  # Young's modulus (Pa)
        nu = 0.4   # Poisson ratio
        G = E / (2.0 * (1.0 + nu))

        I1 = self.h * (self.w ** 3) / 12.0  # stiff axis (out of plane)
        I2 = self.w * (self.h ** 3) / 12.0  # easy axis (in plane)
        # Torsion constant for rectangle w x h (w > h)
        J = (self.w * (self.h ** 3) / 3.0) * (1.0 - 0.63 * (self.h / self.w))

        self.EI1 = E * I1
        self.EI2 = E * I2
        self.GJ = G * J

        # Schedule for driven end rotation
        self.rotation_schedule = bc_cfg["driven_end"]["rotation_schedule"]

        # Twist angles theta on edges: theta[i] is the rotation of the material frame relative to reference Bishop frame on edge i
        # Initially untwisted: theta = 0
        self.theta = np.zeros(self.n_e)
        self.theta_dot = np.zeros(self.n_e)

        # Vertex twist angle tracking (integrated/interpolated for output)
        self.material_twist = np.zeros(self.n_v)

        # Simulation time tracking
        self.current_time = 0.0

        # Fixed reference force / torque scales for contract
        self.reference_force = float(self.EI1 / (self.voronoi_lengths[self.apex_idx] ** 2))
        self.reference_torque = float(self.GJ / self.rest_edge_lengths[0])
        self.constraint_scale = 1.0

    def get_driven_angle(self, t: float) -> float:
        sched = self.rotation_schedule
        if t <= sched[0][0]:
            return sched[0][1]
        if t >= sched[-1][0]:
            return sched[-1][1]
        for i in range(len(sched) - 1):
            t0, a0 = sched[i]
            t1, a1 = sched[i + 1]
            if t0 <= t <= t1:
                frac = (t - t0) / (t1 - t0)
                # Smooth cosine blend or linear
                # Linear interpolation matching schedule points:
                return a0 + frac * (a1 - a0)
        return sched[-1][1]

    def _compute_twist_equilibrium(self, alpha_driven: float) -> np.ndarray:
        """Solve for the quasi-static / equilibrated twist distribution theta across edges.

        In the discrete anisotropic rod formulation (Bergou et al. 2008/2010):
        At the interior vertices (especially the apex k = apex_idx),
        the bending energy depends on the material frame angles theta_{k-1} and theta_k on adjacent edges.
        The discrete curvature binormal (kb)_k = 2 * (e_{k-1} x e_k) / (|e_{k-1}||e_k| + e_{k-1} . e_k)
        For straight arms, (kb)_k = 0 for all k != apex_idx.
        At apex_idx, there is a large rest turning angle phi_0 in the x-z plane (normal = +y).

        The bending energy at vertex k is:
        E_bend = 0.5 / l_k * [ B1 * (omega_1 - omega_1^0)^2 + B2 * (omega_2 - omega_2^0)^2 ]
        where (omega_1, omega_2) are components of the discrete Darboux vector in material frame.
        Since the bend is in the x-z plane with normal along y,
        in the untwisted reference frame, the curvature is entirely along the stiff direction (y) with magnitude kappa_0.
        When edge k-1 twists by theta_{k-1} and edge k twists by theta_k, the average twist at the vertex is
        theta_bar_k = 0.5 * (theta_{k-1} + theta_k).
        The material axes rotate by theta_bar_k relative to the bend plane:
        omega_1 = kappa_0 * cos(theta_bar_k)
        omega_2 = -kappa_0 * sin(theta_bar_k)
        omega_1^0 = kappa_0, omega_2^0 = 0

        So Delta E_bend(vertex k) = 0.5 * (B1 / l_k) * (kappa_0 * cos(theta_bar) - kappa_0)^2 + 0.5 * (B2 / l_k) * (-kappa_0 * sin(theta_bar))^2
                                  = 0.5 * (kappa_0^2 / l_k) * [ B1 * (1 - cos(theta_bar))^2 + B2 * sin^2(theta_bar) ]

        The twisting energy along edge i is:
        E_twist(edge i) = 0.5 * (GJ / |e_i|) * (theta_i - theta_{i-1})^2   (relative twist across vertices)
        or more accurately, for discrete twist m_i = (theta_i - theta_{i-1} + holonomy_i):
        E_twist = sum_{k=1}^{n_v-2} 0.5 * (GJ / l_k) * (theta_k - theta_{k-1})^2
        with boundary conditions: theta_0 = alpha_driven, theta_{n_e-1} = 0 (fixed clamp).

        Notice that at the apex, the bending barrier is:
        V(theta_apex) = 0.5 * (kappa_0^2 / l_apex) * [ B1 * (1 - cos(theta_apex))^2 + B2 * sin^2(theta_apex) ]
        Since B1 >> B2 and B1 >> GJ, V(theta_apex) creates a potential barrier preventing twist from passing the apex
        until the driven arm has twisted significantly and built enough torque to push over or through the barrier.
        """
        # Discretization:
        # N_e edges: i = 0, ..., N_e-1.
        # On driven arm (edges 0 .. apex_idx-1), fixed arm (edges apex_idx .. N_e-1).
        # We can minimize total energy E(theta_0, ..., theta_{n_e-1}) with theta_0 clamped to alpha_driven
        # and theta_{n_e-1} clamped to 0.

        # Compute apex geometric curvature kappa_0
        e_left = self.x0[self.apex_idx] - self.x0[self.apex_idx - 1]
        e_right = self.x0[self.apex_idx + 1] - self.x0[self.apex_idx]
        t_left = e_left / np.linalg.norm(e_left)
        t_right = e_right / np.linalg.norm(e_right)

        # Turning angle at apex
        dot_t = np.clip(np.dot(t_left, t_right), -1.0, 1.0)
        turning_angle = np.arccos(dot_t)
        l_apex = self.voronoi_lengths[self.apex_idx]
        kappa_0 = turning_angle  # integrated discrete curvature

        # Build numerical optimization / relaxation using gradient descent / Newton
        # Unknowns: theta[1 : n_e - 1] (or if driven end specifies edge 0, theta[0] = alpha_driven, theta[-1] = 0)
        # For smooth twist distribution:
        # E_twist = sum_{k=1}^{n_e-1} 0.5 * (GJ / l_k) * (theta_k - theta_{k-1})^2
        # E_bend_apex = 0.5 * (kappa_0^2 / l_apex) * [ B1 * (1 - np.cos(theta_mid))^2 + B2 * np.sin(theta_mid)**2 ]
        # where theta_mid = 0.5 * (theta[apex_idx - 1] + theta[apex_idx])

        # Let's solve for theta given theta[0] = alpha_driven and theta[-1] = 0
        n_unknowns = self.n_e
        # Initial guess: previous theta or linear interpolation
        theta = np.copy(self.theta)
        theta[0] = alpha_driven
        theta[-1] = 0.0

        # Damped gradient relaxation / L-BFGS to find the physical equilibrium state
        for step in range(500):
            grad = np.zeros(n_unknowns)
            # Twisting energy gradients
            for k in range(1, n_unknowns):
                lk = self.voronoi_lengths[k]
                k_twist = self.GJ / lk
                d_diff = theta[k] - theta[k - 1]
                grad[k] += k_twist * d_diff
                grad[k - 1] -= k_twist * d_diff

            # Apex bending energy gradient
            k_l = self.apex_idx - 1
            k_r = self.apex_idx
            theta_mid = 0.5 * (theta[k_l] + theta[k_r])

            # d/d(theta_mid) of [ B1 * (1 - cos(theta_mid))^2 + B2 * sin^2(theta_mid) ]
            # = B1 * 2 * (1 - cos(theta_mid)) * sin(theta_mid) + B2 * 2 * sin(theta_mid) * cos(theta_mid)
            # = 2 * sin(theta_mid) * [ B1 * (1 - cos(theta_mid)) + B2 * cos(theta_mid) ]
            dV_dtheta_mid = (0.5 * (kappa_0 ** 2) / l_apex) * (
                2.0 * np.sin(theta_mid) * (self.EI1 * (1.0 - np.cos(theta_mid)) + self.EI2 * np.cos(theta_mid))
            )

            grad[k_l] += 0.5 * dV_dtheta_mid
            grad[k_r] += 0.5 * dV_dtheta_mid

            # Boundary conditions (fixed values)
            grad[0] = 0.0
            grad[-1] = 0.0

            # Adaptive learning rate step
            step_size = 0.05 / (1.0 + step * 0.01)
            theta -= step_size * grad
            theta[0] = alpha_driven
            theta[-1] = 0.0

            if np.max(np.abs(grad[1:-1])) < 1e-6:
                break

        return theta

    def _quaternion_from_axis_angle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0])
        u = axis / norm
        half = 0.5 * angle
        s = np.sin(half)
        return np.array([u[0] * s, u[1] * s, u[2] * s, np.cos(half)])

    def advance_to(self, time: float) -> StateFrame:
        alpha_driven = self.get_driven_angle(time)

        # Equilibrate twist along the rod
        self.theta = self._compute_twist_equilibrium(alpha_driven)

        # Map edge twist theta to vertex twist
        # For vertices 0..apex_idx..n_v-1:
        self.material_twist[0] = self.theta[0]
        self.material_twist[-1] = self.theta[-1]
        for i in range(1, self.n_v - 1):
            self.material_twist[i] = 0.5 * (self.theta[i - 1] + self.theta[i])

        # Compute vertex orientation quaternions
        # For each vertex, the material frame is rotated by material_twist[i] around the local tangent.
        # In the reference state, initial orientation is identity [0, 0, 0, 1].
        # The tangent vector at vertex i:
        tangents = np.zeros((self.n_v, 3))
        tangents[0] = self.x0[1] - self.x0[0]
        tangents[-1] = self.x0[-1] - self.x0[-2]
        for i in range(1, self.n_v - 1):
            tangents[i] = self.x0[i + 1] - self.x0[i - 1]
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)

        orientations = np.zeros((self.n_v, 4))
        for i in range(self.n_v):
            orientations[i] = self._quaternion_from_axis_angle(tangents[i], self.material_twist[i])

        # Physical forces and torques for trajectory budget
        # External torque at driven clamp:
        k_twist0 = self.GJ / self.voronoi_lengths[0]
        driven_torque_mag = k_twist0 * (self.theta[0] - self.theta[1])
        driven_couple = driven_torque_mag * tangents[0]

        # External torque at fixed clamp:
        k_twist_end = self.GJ / self.voronoi_lengths[-1]
        fixed_torque_mag = k_twist_end * (self.theta[-2] - self.theta[-1])
        fixed_couple = fixed_torque_mag * tangents[-1]

        # Budget arrays: [G, K, 3] where G=1 domain, K=2 clamps/terms
        budget_external_couple = np.zeros((1, 2, 3))
        budget_external_couple[0, 0] = driven_couple
        budget_external_couple[0, 1] = fixed_couple

        budget_external_force = np.zeros((1, 2, 3))
        budget_force_position = np.zeros((1, 2, 3))
        budget_force_position[0, 0] = self.x[0]
        budget_force_position[0, 1] = self.x[-1]

        # Constraint target & value:
        # Driven end twist angle
        constraint_target = np.array([alpha_driven])
        constraint_value = np.array([self.material_twist[0]])

        # Cumulative mass source: [K] = [0.0]
        cumulative_mass_source = np.array([0.0])

        # Mass quantity: [N]
        mass_quantity = np.copy(self.mass)

        frame = StateFrame(
            time=time,
            position=np.copy(self.x),
            velocity=np.copy(self.v),
            orientation=orientations,
            angular_velocity=np.zeros_like(self.x),
            mass=self.mass,
            attributes={
                "arc_length": self.arc_length,
                "material_twist": np.copy(self.material_twist),
                "cross_section_width": np.full(self.n_v, self.w),
                "cross_section_thickness": np.full(self.n_v, self.h),
                "is_apex": (np.arange(self.n_v) == self.apex_idx).astype(float),
                "is_driven_end": (np.arange(self.n_v) == 0).astype(float),
                "is_fixed_end": (np.arange(self.n_v) == self.n_v - 1).astype(float),
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_external_couple": budget_external_couple,
                "budget_external_force": budget_external_force,
                "budget_force_position": budget_force_position,
                "constraint_target": constraint_target,
                "constraint_value": constraint_value,
                "cumulative_mass_source": cumulative_mass_source,
            },
            scalars={
                "driven_end_angle": alpha_driven,
                "constraint_scale": self.constraint_scale,
                "reference_force": self.reference_force,
                "reference_torque": self.reference_torque,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "theta_driven": float(self.theta[0]) if hasattr(self, "theta") else 0.0,
            "theta_apex": float(self.material_twist[self.apex_idx]) if hasattr(self, "material_twist") else 0.0,
            "theta_fixed": float(self.theta[-1]) if hasattr(self, "theta") else 0.0,
        }

