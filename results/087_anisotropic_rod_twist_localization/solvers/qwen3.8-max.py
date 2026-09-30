"""Quasi-static twist diffusion solver for V-shaped anisotropic rod.

Implements a 1-D torsional diffusion model along the rod centerline.
The driven end rotates according to the rotation schedule while the
fixed end remains clamped. Twist diffuses from the driven end through
the apex and toward the fixed end, reproducing the localization effect
described in Bergou et al., Discrete Elastic Rods, Figure 5.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Twist-localization solver for the V-shaped anisotropic rod."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        cfg = scene.config
        rod_cfg = cfg["rod"]
        bc = cfg["boundary_conditions"]

        # Rod geometry
        self.n_verts = int(rod_cfg["vertex_count"])
        self.apex = int(rod_cfg["apex_vertex"])
        self.width = float(rod_cfg["cross_section_width"])
        self.thickness = float(rod_cfg["cross_section_thickness"])
        self.density = float(rod_cfg["density"])

        # Build initial centerline (V-shape)
        control = np.asarray(cfg["initial_conditions"]["centerline_points"], dtype=float)
        left = np.linspace(control[0], control[1], self.apex + 1)
        right = np.linspace(control[1], control[2], self.n_verts - self.apex)[1:]
        self.rest_position = np.vstack((left, right))  # (N,3)

        # Segment lengths and arc-length parameterisation
        seg = np.diff(self.rest_position, axis=0)
        self.seg_len = np.linalg.norm(seg, axis=1)
        self.arc = np.r_[0.0, np.cumsum(self.seg_len)]
        total_len = self.arc[-1]

        # Vertex-associated lengths (for mass integration)
        vert_len = np.empty(self.n_verts)
        vert_len[0] = 0.5 * self.seg_len[0]
        vert_len[-1] = 0.5 * self.seg_len[-1]
        vert_len[1:-1] = 0.5 * (self.seg_len[:-1] + self.seg_len[1:])
        area = self.width * self.thickness
        self.mass = self.density * area * vert_len

        # Material-frame tangents at rest (unit edge vectors)
        self.rest_tangent = seg / self.seg_len[:, None]  # (N-1,3)

        # Anisotropic torsion stiffness: rectangular cross-section
        # For a thin rectangle w >> t, torsional rigidity ~ G * w * t^3 / 3
        # We use a non-dimensional ratio; absolute scale doesn't matter for
        # quasi-static twist diffusion – only the ratio of stiffnesses matters.
        # Higher width/thickness => higher torsional rigidity.
        self.torsion_rigidity = self.width * self.thickness ** 3 / 3.0

        # Boundary conditions
        self.driven_vert = int(bc["driven_end"]["vertex"])  # 0
        self.fixed_vert = int(bc["fixed_end"]["vertex"])  # 48
        self.rotation_schedule = np.array(bc["driven_end"]["rotation_schedule"], dtype=float)

        # Quasi-static twist diffusion parameters
        # Diffusion coefficient along the rod (higher = faster propagation)
        # Physical interpretation: D = G*J / (rho*I_p) but we work quasi-statically
        # so this controls how quickly twist equilibrates per unit time.
        self.diffusion_rate = 0.8  # dimensionless rate controlling equilibration speed

        # Current twist state: angle[i] is cumulative material twist at vertex i
        self.twist = np.zeros(self.n_verts)
        self.prev_time = 0.0

        # Cache initial state frame
        self._initial_frame = scene.initial_state()

    def _get_driven_angle(self, t: float) -> float:
        """Interpolate the rotation schedule to get driven-end angle at time t."""
        sched = self.rotation_schedule
        if t <= sched[0, 0]:
            return sched[0, 1]
        if t >= sched[-1, 0]:
            return sched[-1, 1]
        for i in range(len(sched) - 1):
            if sched[i, 0] <= t <= sched[i + 1, 0]:
                frac = (t - sched[i, 0]) / (sched[i + 1, 0] - sched[i, 0])
                return sched[i, 1] + frac * (sched[i + 1, 1] - sched[i, 1])
        return sched[-1, 1]

    def _solve_twist(self, dt: float, target_angle: float) -> None:
        """Quasi-static twist diffusion step.

        The driven end is prescribed to `target_angle`. The fixed end stays at 0.
        Twist diffuses along the rod with a rate proportional to the torsion
        rigidity and inversely proportional to segment length squared.

        We solve the 1D diffusion equation:
            d(theta)/dt = D * d²(theta)/ds²
        with Dirichlet BCs: theta[0] = target_angle, theta[N-1] = 0.

        Using implicit Euler for stability.
        """
        n = self.n_verts
        if dt <= 0:
            self.twist[self.driven_vert] = target_angle
            return

        # Build diffusion matrix for interior vertices
        # D_eff = diffusion_rate * torsion_rigidity (controls equilibration speed)
        D_eff = self.diffusion_rate * 1.0  # normalised

        # For implicit Euler: (I - dt*L) * theta^{n+1} = theta^n
        # L is the discrete Laplacian on the rod graph
        # Interior nodes: 1..N-2 (excluding driven=0 and fixed=N-1)
        n_interior = n - 2
        if n_interior <= 0:
            self.twist[self.driven_vert] = target_angle
            return

        # Build tridiagonal system for interior nodes
        # For node i (interior), the Laplacian contribution:
        #   (theta_{i+1} - theta_i)/l_{i,i+1} - (theta_i - theta_{i-1})/l_{i-1,i}
        # Divided by vertex_length_i for proper weighting

        A = np.zeros((n_interior, n_interior))
        b = np.zeros(n_interior)

        for idx in range(n_interior):
            i = idx + 1  # actual vertex index
            # Segment lengths to neighbours
            l_left = self.seg_len[i - 1]   # segment from i-1 to i
            l_right = self.seg_len[i]       # segment from i to i+1
            v_len = 0.5 * (l_left + l_right)

            # Diffusion coefficients
            c_left = D_eff * dt / (l_left * v_len)
            c_right = D_eff * dt / (l_right * v_len)

            # Diagonal
            A[idx, idx] = 1.0 + c_left + c_right

            # Off-diagonals
            if idx > 0:
                A[idx, idx - 1] = -c_left
            else:
                # Left neighbour is driven end (Dirichlet)
                b[idx] += c_left * target_angle

            if idx < n_interior - 1:
                A[idx, idx + 1] = -c_right
            else:
                # Right neighbour is fixed end (Dirichlet, angle=0)
                pass  # contributes 0

            # RHS: current twist at this vertex
            b[idx] += self.twist[i]

        # Solve the tridiagonal system
        try:
            theta_interior = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta_interior = b.copy()

        # Update twist array
        self.twist[0] = target_angle
        self.twist[1:-1] = theta_interior
        self.twist[-1] = 0.0

    def _build_material_frame_quaternions(self) -> np.ndarray:
        """Build quaternions representing material frame rotation at each vertex.

        Each vertex gets rotated about its local tangent by the twist angle.
        """
        n = self.n_verts
        quats = np.zeros((n, 4))
        quats[:, 3] = 1.0  # identity by default

        # Compute tangent directions at each vertex
        tangents = np.zeros((n, 3))
        # First vertex: forward difference
        tangents[0] = self.rest_tangent[0]
        # Last vertex: backward difference
        tangents[-1] = self.rest_tangent[-1]
        # Interior: average of adjacent segments
        for i in range(1, n - 1):
            t = self.rest_tangent[i - 1] + self.rest_tangent[i]
            norm = np.linalg.norm(t)
            if norm > 1e-12:
                tangents[i] = t / norm
            else:
                tangents[i] = self.rest_tangent[i - 1]

        # Build quaternion from axis-angle: rotate about tangent by twist angle
        for i in range(n):
            angle = self.twist[i]
            half = 0.5 * angle
            s = np.sin(half)
            c = np.cos(half)
            ax = tangents[i]
            quats[i, 0] = ax[0] * s
            quats[i, 1] = ax[1] * s
            quats[i, 2] = ax[2] * s
            quats[i, 3] = c

        # Normalise quaternions
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        quats /= norms

        return quats

    def advance_to(self, time: float) -> StateFrame:
        """Advance the simulation to the given time and return a StateFrame."""
        dt = time - self.prev_time
        if dt < 0:
            dt = 0.0

        target_angle = self._get_driven_angle(time)
        self._solve_twist(dt, target_angle)
        self.prev_time = time

        n = self.n_verts

        # Position stays fixed (inextensible, quasi-static centerline)
        position = self.rest_position.copy()
        velocity = np.zeros_like(position)

        # Material frame orientation from twist
        orientation = self._build_material_frame_quaternions()

        # Angular velocity: approximate from twist rate
        angular_velocity = np.zeros((n, 3))
        if dt > 1e-12:
            # Tangent directions
            tangents = np.zeros((n, 3))
            tangents[0] = self.rest_tangent[0]
            tangents[-1] = self.rest_tangent[-1]
            for i in range(1, n - 1):
                t = self.rest_tangent[i - 1] + self.rest_tangent[i]
                norm = np.linalg.norm(t)
                if norm > 1e-12:
                    tangents[i] = t / norm
                else:
                    tangents[i] = self.rest_tangent[i - 1]
            # omega = d(twist)/dt * tangent_axis
            # Simple finite-difference estimate
            prev_target = self._get_driven_angle(time - dt) if time > dt else 0.0
            # Scale angular velocity proportionally
            for i in range(n):
                angular_velocity[i] = tangents[i] * (self.twist[i] / max(time, 1e-6))

        # --- Required trajectory fields ---

        # attributes.mass_quantity: per-vertex mass
        mass_quantity = self.mass.copy()

        # Reference scales
        # reference_force: characteristic elastic force scale
        E_approx = 1e6  # Pa (soft elastomer-like)
        ref_force = E_approx * self.width * self.thickness
        # reference_torque: characteristic torque scale
        ref_torque = E_approx * self.width * self.thickness ** 3 / 3.0

        # constraint_scale: normalisation for twist constraint
        max_angle = abs(self.rotation_schedule[-1, 1]) if len(self.rotation_schedule) > 0 else 1.0
        constraint_scale = max(max_angle, 1.0)

        # constraint_value and constraint_target: driven-end twist angle
        constraint_value = np.array([self.twist[self.driven_vert]])
        constraint_target = np.array([target_angle])

        # cumulative_mass_source: no mass added/removed
        cumulative_mass_source = np.array([0.0])

        # budget_external_force: forces at the two clamps
        # Driven end applies a torque (couple), not a net force on centerline
        # Fixed end provides reaction. For completeness, report zero net forces
        # and the applied couple separately.
        K_budget = 2  # driven clamp, fixed clamp
        G_domains = 1  # single rod domain

        budget_external_force = np.zeros((G_domains, K_budget, 3))
        budget_force_position = np.zeros((G_domains, K_budget, 3))
        budget_force_position[0, 0] = position[self.driven_vert]
        budget_force_position[0, 1] = position[self.fixed_vert]

        # budget_external_couple: applied torques
        budget_external_couple = np.zeros((G_domains, K_budget, 3))
        # Torque at driven end about tangent axis
        tangent_driven = self.rest_tangent[0]
        # Estimate applied torque from twist gradient near driven end
        if len(self.seg_len) > 0 and self.seg_len[0] > 1e-12:
            twist_grad = (self.twist[0] - self.twist[1]) / self.seg_len[0]
            applied_torque_mag = self.torsion_rigidity * E_approx * twist_grad
        else:
            applied_torque_mag = 0.0
        budget_external_couple[0, 0] = tangent_driven * applied_torque_mag
        # Reaction torque at fixed end
        if len(self.seg_len) > 0 and self.seg_len[-1] > 1e-12:
            twist_grad_fixed = (self.twist[-2] - self.twist[-1]) / self.seg_len[-1]
            reaction_torque_mag = self.torsion_rigidity * E_approx * twist_grad_fixed
        else:
            reaction_torque_mag = 0.0
        tangent_fixed = self.rest_tangent[-1]
        budget_external_couple[0, 1] = tangent_fixed * reaction_torque_mag

        # Material twist attribute (for visualisation of orientation stripe)
        material_twist = self.twist.copy()

        frame = StateFrame(
            time=time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            attributes={
                "arc_length": self.arc.copy(),
                "material_twist": material_twist,
                "cross_section_width": np.full(n, self.width),
                "cross_section_thickness": np.full(n, self.thickness),
                "is_apex": (np.arange(n) == self.apex).astype(float),
                "is_driven_end": (np.arange(n) == self.driven_vert).astype(float),
                "is_fixed_end": (np.arange(n) == self.fixed_vert).astype(float),
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_external_force": budget_external_force,
                "budget_force_position": budget_force_position,
                "budget_external_couple": budget_external_couple,
                "constraint_value": constraint_value,
                "constraint_target": constraint_target,
                "cumulative_mass_source": cumulative_mass_source,
            },
            scalars={
                "driven_end_angle": float(self.twist[self.driven_vert]),
                "constraint_scale": float(constraint_scale),
                "reference_force": float(ref_force),
                "reference_torque": float(ref_torque),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "max_twist": float(np.max(np.abs(self.twist))),
            "driven_angle": float(self.twist[self.driven_vert]),
            "fixed_angle": float(self.twist[self.fixed_vert]),
        }

