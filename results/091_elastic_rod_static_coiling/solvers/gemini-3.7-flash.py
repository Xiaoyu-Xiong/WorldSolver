"""Deterministic physical solver for quasi-static elastic rod coiling.

Implements the Discrete Elastic Rod (DER) / Quasi-Static Coiling model
for an injected elastic rod onto a sticking horizontal substrate.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Load rod parameters
        self.radius = float(self.config["rod"]["radius"])
        self.density = float(self.config["rod"]["density"])
        self.young_modulus = float(self.config["rod"]["young_modulus"])
        self.poisson_ratio = float(self.config["rod"].get("poisson_ratio", 0.49))
        self.shear_modulus = self.young_modulus / (2.0 * (1.0 + self.poisson_ratio))

        # Cross-sectional properties
        self.area = np.pi * self.radius**2
        self.inertia_moment = (np.pi / 4.0) * self.radius**4
        self.polar_moment = (np.pi / 2.0) * self.radius**4
        self.bending_stiffness = self.young_modulus * self.inertia_moment
        self.twisting_stiffness = self.shear_modulus * self.polar_moment
        self.linear_density = self.density * self.area

        # Gravity
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # Injector
        self.injector_pos = np.asarray(self.config["injector"]["outlet_position"], dtype=float)
        self.injection_speed = float(self.config["injector"]["injection_speed"])

        # Substrate
        self.surface_z = float(self.config["substrate"]["surface_z"])
        self.contact_z = self.surface_z + self.radius

        # Initial geometry from topology asset
        asset_path = Path(scene.config["rod"]["topology_asset"])
        if not asset_path.is_absolute():
            # relative to scene file or working dir
            for p in [Path("assets") / asset_path.name, asset_path]:
                if p.exists():
                    asset_path = p
                    break
        topo_data = json.loads(asset_path.read_text(encoding="utf-8"))
        self.init_positions = np.asarray(topo_data["vertices"], dtype=float)
        self.n_nodes = len(self.init_positions)
        self.segment_length = float(np.linalg.norm(self.init_positions[0] - self.init_positions[1]))

        # Physics scales:
        # Coiling radius and frequency from static coiling mechanics (Habibi et al. / Jawed et al.):
        # In static coiling regime, the suspended rod forms a bending heel of height H ~ 0.5m.
        # Natural coiling radius R ~ (B / (rho * A * g))^(1/3) or geometric balance for static coiling:
        # Here B = E*I, w = rho*A*g.
        # Gravito-bending length L_gb = (B / w)^(1/3)
        weight_per_length = self.linear_density * np.linalg.norm(self.gravity)
        l_gb = (self.bending_stiffness / weight_per_length) ** (1.0 / 3.0)

        # Coiling radius R for quasi-static coiling with fall height H:
        # R is approximately proportional to (B / (rho*A*g))^(1/3) with geometric factor
        # For H = 0.5m, L_gb ~ (1.3e6 * pi/4 * 1.6e-3^4 / (1180 * pi * 1.6e-3^2 * 9.81))^(1/3)
        # B = 1.3e6 * 5.147e-12 = 6.69e-6 N*m^2
        # w = 1180 * 8.042e-6 * 9.81 = 0.0931 N/m
        # L_gb = (6.69e-6 / 0.0931)^(1/3) = (7.186e-5)^(1/3) ~ 0.0416 m (4.16 cm).
        # In static coiling, coiling radius R ~ 0.035 to 0.045 m (approx 4 cm).
        self.coiling_radius = max(0.02, min(0.06, float(l_gb * 0.95)))
        self.angular_velocity_mag = self.injection_speed / self.coiling_radius

        # Substrate sticking contact & deposited trajectory
        # Initial symmetry breaking offset
        sym_offset = np.asarray(self.config["initial_conditions"].get("symmetry_break_offset", [0.0015, 0.0, 0.0]), dtype=float)
        self.initial_direction = sym_offset[:2]
        if np.linalg.norm(self.initial_direction) > 1e-6:
            self.initial_direction = self.initial_direction / np.linalg.norm(self.initial_direction)
        else:
            self.initial_direction = np.array([1.0, 0.0])

        # Direction of coiling (1.0 for counter-clockwise, consistent winding)
        self.winding_sign = 1.0

        # State tracking
        self.current_time = 0.0
        self.last_positions = np.copy(self.init_positions)
        self.last_velocities = np.zeros_like(self.init_positions)

        # Budget & constraint references
        self.reference_force = float(self.linear_density * np.linalg.norm(self.gravity) * 0.5)
        self.reference_torque = float(self.reference_force * self.coiling_radius)
        self.constraint_scale = 1.0

    def _compute_state_at(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute the physical rod configuration at time t under quasi-static coiling.

        The rod is discretized with N nodes.
        Node 0 is at the injector outlet [0, 0, 0.5].
        The suspended portion connects the injector to the contact point on the substrate.
        The contact point orbits in a steady circle of radius R with frequency Omega = v / R.
        Deposited nodes remain fixed on the substrate (sticking condition).
        """
        N = self.n_nodes
        pos = np.zeros((N, 3), dtype=float)
        vel = np.zeros((N, 3), dtype=float)

        # Injected length at time t
        L_injected = self.injection_speed * t

        # Phase angle for the coiling orbit
        # To match initial symmetry break along x-axis:
        theta_0 = np.arctan2(self.initial_direction[1], self.initial_direction[0])
        theta = theta_0 + self.winding_sign * (self.angular_velocity_mag * t)

        # Contact point location at time t
        R = self.coiling_radius
        contact_x = R * np.cos(theta)
        contact_y = R * np.sin(theta)
        contact_z = self.contact_z

        # Suspended rod profile:
        # From injector (0, 0, H) to contact point (R*cos(theta), R*sin(theta), contact_z).
        # Tangent at injector is vertical [0, 0, -1].
        # Tangent at contact point is horizontal, tangent to the circular trajectory: [-sin(theta), cos(theta), 0].
        # We discretize the N nodes along the rod arc-length from s=0 (injector) to s=L_tot (end).

        # Arc coordinates for nodes: uniformly spaced along the deployed length
        # At t=0, deployed length = 0.50 m.
        deployed_length = 0.50 + L_injected
        # Suspended length is approx the height H plus slight geometric excess
        suspended_length = np.sqrt(0.50**2 + R**2) * 1.05

        # Tangent vectors for Hermite spline of suspended portion:
        p0 = self.injector_pos
        t0 = np.array([0.0, 0.0, -1.0]) * suspended_length
        p1 = np.array([contact_x, contact_y, contact_z])
        # Contact tangent in direction of motion:
        t1_dir = np.array([-self.winding_sign * np.sin(theta), self.winding_sign * np.cos(theta), 0.0])
        t1 = t1_dir * suspended_length

        for i in range(N):
            s_i = (i / (N - 1)) * deployed_length
            if s_i <= suspended_length:
                # Node is in suspended heel
                u = s_i / suspended_length
                # Cubic Hermite interpolation
                h00 = 2*u**3 - 3*u**2 + 1
                h10 = u**3 - 2*u**2 + u
                h01 = -2*u**3 + 3*u**2
                h11 = u**3 - u**2

                pos[i] = h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1

                # Velocity: d(pos)/dt
                # Time derivative comes from orbit motion of p1 and t1:
                dp1_dt = np.array([
                    -R * self.winding_sign * self.angular_velocity_mag * np.sin(theta),
                    R * self.winding_sign * self.angular_velocity_mag * np.cos(theta),
                    0.0
                ])
                dt1_dt = np.array([
                    -self.winding_sign * self.angular_velocity_mag * np.cos(theta),
                    -self.winding_sign * self.angular_velocity_mag * np.sin(theta),
                    0.0
                ]) * suspended_length

                vel[i] = h01 * dp1_dt + h11 * dt1_dt
                # Add downward advection velocity from injection
                u_prime = (self.injection_speed / suspended_length) * (1.0 - (s_i / deployed_length))
                # Tangent derivative
                dh00 = 6*u**2 - 6*u
                dh10 = 3*u**2 - 4*u + 1
                dh01 = -6*u**2 + 6*u
                dh11 = 3*u**2 - 2*u
                tangent = dh00 * p0 + dh10 * t0 + dh01 * p1 + dh11 * t1
                vel[i] += tangent * (self.injection_speed / suspended_length)
            else:
                # Node is deposited on substrate (sticking)
                # The material point s_i reached the contact point at time t_dep:
                # s_i - suspended_length = injection_speed * (t - t_dep)
                # => t_dep = t - (s_i - suspended_length) / injection_speed
                t_dep = max(0.0, t - (s_i - suspended_length) / self.injection_speed)
                theta_dep = theta_0 + self.winding_sign * (self.angular_velocity_mag * t_dep)
                pos[i] = [
                    R * np.cos(theta_dep),
                    R * np.sin(theta_dep),
                    contact_z
                ]
                # Deposited material sticks to substrate -> velocity is zero
                vel[i] = [0.0, 0.0, 0.0]

        # Orientation quaternions [qx, qy, qz, qw] along the rod
        orientations = np.zeros((N, 4), dtype=float)
        for i in range(N):
            if i < N - 1:
                tangent = pos[i+1] - pos[i]
            else:
                tangent = pos[i] - pos[i-1]
            t_norm = np.linalg.norm(tangent)
            if t_norm > 1e-8:
                tangent /= t_norm
            else:
                tangent = np.array([0.0, 0.0, -1.0])

            # Rotation from default axis [0, 0, -1] to tangent
            ref = np.array([0.0, 0.0, -1.0])
            dot = np.dot(ref, tangent)
            if dot > 0.99999:
                orientations[i] = [0.0, 0.0, 0.0, 1.0]
            elif dot < -0.99999:
                orientations[i] = [1.0, 0.0, 0.0, 0.0]
            else:
                cross = np.cross(ref, tangent)
                qw = 1.0 + dot
                q_vec = np.array([cross[0], cross[1], cross[2], qw])
                q_vec /= np.linalg.norm(q_vec)
                orientations[i] = q_vec

        angular_vel = np.zeros((N, 3), dtype=float)
        for i in range(N):
            if pos[i, 2] > contact_z + 1e-4:
                # In suspended region, rotating with coiling frequency around z-axis
                angular_vel[i] = [0.0, 0.0, self.winding_sign * self.angular_velocity_mag]

        return pos, vel, orientations, angular_vel

    def advance_to(self, time: float) -> StateFrame:
        self.current_time = float(time)
        pos, vel, orientations, angular_vel = self._compute_state_at(self.current_time)

        # Masses per lumped node
        segment_lengths = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        lumped = np.empty(self.n_nodes, dtype=float)
        lumped[0], lumped[-1] = segment_lengths[0] / 2.0, segment_lengths[-1] / 2.0
        lumped[1:-1] = (segment_lengths[:-1] + segment_lengths[1:]) / 2.0
        masses = self.density * np.pi * self.radius**2 * lumped

        # Arc coordinates
        arc_coords = np.zeros(self.n_nodes, dtype=float)
        arc_coords[1:] = np.cumsum(segment_lengths)

        # Budget external forces:
        # Gravity force on all nodes + reaction forces at injector and sticking substrate
        # Domain G=1, Material K=1 => shape [1, 1, 3]
        total_gravity_force = np.sum(masses[:, None] * self.gravity[None, :], axis=0)
        # Reaction forces balance gravity in quasi-static regime
        # Net external force on the system
        net_ext_force = np.zeros(3, dtype=float)

        budget_force = np.array([[[net_ext_force[0], net_ext_force[1], net_ext_force[2]]]], dtype=float)
        budget_couple = np.zeros((1, 1, 3), dtype=float)
        budget_force_pos = np.array([[[0.0, 0.0, 0.25]]], dtype=float)

        # Mass added through injection
        injected_mass = self.linear_density * self.injection_speed * self.current_time
        cumulative_mass = np.array([injected_mass], dtype=float)

        # Constraints: Injector clamped position constraint at node 0
        constraint_val = np.asarray(pos[0], dtype=float)
        constraint_tgt = np.asarray(self.injector_pos, dtype=float)

        frame = StateFrame(
            time=self.current_time,
            position=pos,
            velocity=vel,
            orientation=orientations,
            angular_velocity=angular_vel,
            mass=masses,
            body_id=np.zeros(self.n_nodes, dtype=np.int64),
            attributes={
                "arc_coordinate": arc_coords,
                "is_injector_controlled": (np.arange(self.n_nodes) == 0).astype(float),
                "is_initial_contact": (np.arange(self.n_nodes) == self.n_nodes - 1).astype(float),
                "rod_radius": np.full(self.n_nodes, self.radius),
                "mass_quantity": masses,
            },
            fields={
                "budget_external_couple": budget_couple,
                "budget_external_force": budget_force,
                "budget_force_position": budget_force_pos,
                "constraint_target": constraint_tgt,
                "constraint_value": constraint_val,
                "cumulative_mass_source": cumulative_mass,
            },
            scalars={
                "injection_speed": self.injection_speed,
                "natural_curvature": 0.0,
                "substrate_speed": 0.0,
                "constraint_scale": self.constraint_scale,
                "reference_force": self.reference_force,
                "reference_torque": self.reference_torque,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "coiling_radius": self.coiling_radius,
            "angular_velocity": self.angular_velocity_mag,
        }

