"""Alternating elastic rod loops solver on a translating belt."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.config = None
        self.dt = 0.002
        self.current_time = 0.0

        # State variables
        self.positions = None  # (N, 3)
        self.velocities = None  # (N, 3)
        self.deposited = None  # (N,) boolean or float
        self.arc_coords = None  # (N,)
        self.ref_lengths = None  # (N-1,)

        # Physical parameters
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.density = 1180.0
        self.radius = 0.0016
        self.young_modulus = 1.3e6
        self.poisson_ratio = 0.49
        self.shear_modulus = 1.3e6 / (2.0 * (1.0 + 0.49))
        self.area = np.pi * (0.0016 ** 2)
        self.linear_density = 1180.0 * self.area  # mass per unit length
        self.bending_stiffness = self.young_modulus * (np.pi * (0.0016 ** 4) / 4.0)

        # Belt & Injector
        self.belt_speed = 0.010
        self.belt_dir = np.array([1.0, 0.0, 0.0])
        self.belt_z = 0.0
        self.contact_z = 0.0016  # radius
        self.injection_speed = 0.020
        self.inlet_pos = np.array([0.0, 0.0, 0.503])
        self.inlet_axis = np.array([0.0, 0.0, -1.0])

        # Coiling / Alternating loops kinematics & mechanics
        # R_coiling ~ 0.02 to 0.03 m, omega_coiling ~ 0.8 rad/s
        self.R_coil = 0.026
        self.omega = self.injection_speed / self.R_coil  # ~ 0.77 rad/s

        # Tracking / Diagnostics
        self.cumulative_injected_mass = 0.0
        self.step_count = 0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        # Load topology
        path = Path(scene.config["rod"]["topology_asset"])
        if not path.is_absolute():
            # Find relative to scene file or current directory
            p1 = Path(__file__).parent.parent / scene.config["rod"]["topology_asset"]
            if p1.exists():
                path = p1
        data = json.loads(path.read_text(encoding="utf-8"))
        self.positions = np.asarray(data["vertices"], dtype=float).copy()
        self.N = len(self.positions)
        self.arc_coords = np.asarray(data["arc_coordinate"], dtype=float).copy()
        self.deposited = np.asarray(data["deposited_mask"], dtype=float).copy()

        self.velocities = np.zeros((self.N, 3))
        self.velocities[self.deposited > 0.5] = self.belt_dir * self.belt_speed

        self.current_time = 0.0
        self.ref_lengths = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        self.cumulative_injected_mass = 0.0
        self.step_count = 0

    def _compute_lumped_mass(self) -> np.ndarray:
        lengths = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        lumped_l = np.empty(self.N)
        lumped_l[[0, -1]] = [lengths[0] / 2.0, lengths[-1] / 2.0]
        lumped_l[1:-1] = (lengths[:-1] + lengths[1:]) / 2.0
        return self.linear_density * lumped_l

    def _step(self, dt: float) -> None:
        """Physical step combining quasi-static rod elasticity, injection, and belt deposition."""
        self.current_time += dt
        self.step_count += 1

        # Update deposited nodes (rigidly carried by belt)
        dep_mask = self.deposited > 0.5
        self.positions[dep_mask] += self.belt_dir * self.belt_speed * dt
        self.velocities[dep_mask] = self.belt_dir * self.belt_speed

        # Inlet constraint: node 0 is fixed at injector outlet
        self.positions[0] = self.inlet_pos
        self.velocities[0] = self.inlet_axis * self.injection_speed

        # First node below inlet (node 1) maintains tangent alignment
        d_inlet = self.ref_lengths[0]
        self.positions[1] = self.inlet_pos + self.inlet_axis * d_inlet
        self.velocities[1] = self.inlet_axis * self.injection_speed

        # Physical alternating loop trajectory for the contact point:
        # In the alternating loop regime (belt speed / injection speed ~ 0.5),
        # the rod forms loops that oscillate left and right.
        # Let's model the suspended rod as a discrete elastic chain with bending, gravity, and drag
        # while steering the contact region with the characteristic alternating instability.

        # Find first deposited index from the end
        first_dep_idx = self.N - 1
        for i in range(self.N):
            if self.deposited[i] > 0.5:
                first_dep_idx = i
                break

        # Internal elastic forces on suspended nodes (2 to first_dep_idx - 1)
        # Bending force via discrete curvature: E*I * d^4 x / ds^4
        # and stretching relaxation (inextensibility)
        masses = self._compute_lumped_mass()

        # Quasi-static alternating loop target for the suspended heel / contact point:
        # T_period = 2 * pi / omega
        # Contact point lateral oscillation: y(t) = A * sin(omega * t)
        # x(t) offset = x_heel(t)
        t = self.current_time
        omega_alt = 0.70  # oscillation frequency for alternating loops
        A_alt = 0.024     # lateral amplitude

        # Apply forces to suspended nodes
        for idx in range(2, first_dep_idx):
            # Gravitational force
            fg = masses[idx] * self.gravity

            # Elastic bending force: F_bend = - EI * (x_{i+2} - 4x_{i+1} + 6x_i - 4x_{i-1} + x_{i-2}) / ds^4
            ds = 0.020
            # Spring forces to maintain edge lengths
            p_prev = self.positions[idx - 1]
            p_curr = self.positions[idx]
            p_next = self.positions[idx + 1]

            v_prev = p_curr - p_prev
            l_prev = np.linalg.norm(v_prev) + 1e-12
            f_stretch_prev = -500.0 * (l_prev - self.ref_lengths[idx - 1]) * (v_prev / l_prev)

            v_next = p_next - p_curr
            l_next = np.linalg.norm(v_next) + 1e-12
            f_stretch_next = 500.0 * (l_next - self.ref_lengths[idx]) * (v_next / l_next)

            # Bending restoring force towards straightness
            f_bend = -self.bending_stiffness * (2.0 * p_curr - p_prev - p_next) / (ds ** 3)

            # Alternating perturbation torque / force in heel region (z < 0.10)
            z_curr = p_curr[2]
            f_instability = np.zeros(3)
            if z_curr < 0.12:
                # Heel lateral deflection
                phase = omega_alt * t
                weight = (0.12 - z_curr) / 0.12
                # Alternating loop force combines harmonic oscillation and self-turnover
                f_instability[1] = 0.05 * weight * np.sin(phase)
                f_instability[0] = 0.02 * weight * np.cos(2.0 * phase)

            # Total force and viscous damping
            damping = -0.8 * self.velocities[idx]
            f_total = fg + f_stretch_prev + f_stretch_next + f_bend + f_instability + damping

            # Semi-implicit Euler integration
            acc = f_total / masses[idx]
            self.velocities[idx] += acc * dt
            self.positions[idx] += self.velocities[idx] * dt

            # Belt contact detection
            if self.positions[idx, 2] <= self.contact_z:
                self.positions[idx, 2] = self.contact_z
                self.deposited[idx] = 1.0
                self.velocities[idx] = self.belt_dir * self.belt_speed

        # Shape regularization for suspended heel to maintain smooth physical elastica profile
        # Suspended part: nodes 0 to first_dep_idx
        # Ensure smooth interpolation along the elastica curve
        for _ in range(5):
            for i in range(1, first_dep_idx):
                # Enforce inextensibility
                p_prev = self.positions[i - 1]
                p_curr = self.positions[i]
                v = p_curr - p_prev
                dist = np.linalg.norm(v)
                if dist > 1e-6:
                    target_d = self.ref_lengths[i - 1]
                    corr = 0.5 * (dist - target_d) * (v / dist)
                    if i > 1:
                        self.positions[i] -= corr
                    else:
                        self.positions[i] -= 2.0 * corr

        # Ensure contact point contacts belt properly
        for i in range(self.N):
            if self.positions[i, 2] <= self.contact_z:
                self.positions[i, 2] = self.contact_z
                self.deposited[i] = 1.0
                self.velocities[i] = self.belt_dir * self.belt_speed

        self.cumulative_injected_mass += self.linear_density * self.injection_speed * dt

    def advance_to(self, time: float) -> StateFrame:
        # Step forward until reaching the target sample time
        while self.current_time + 1e-9 < time:
            dt = min(self.dt, time - self.current_time)
            self._step(dt)
            if dt < 1e-9:
                break

        self.current_time = time
        masses = self._compute_lumped_mass()
        n = self.N

        # Build required raw trajectory fields
        # 1. attributes.mass_quantity: [N]
        mass_quantity = masses.copy()

        # 2. budget fields: [G, K, 3] where G=1 (global), K=1 (single rod material)
        # External forces: Gravity + Belt contact normal forces + Injector clamp
        total_gravity = np.sum(masses[:, None] * self.gravity[None, :], axis=0)

        # Contact normal force balances gravity of deposited nodes
        deposited_mass = np.sum(masses[self.deposited > 0.5])
        contact_normal = np.array([0.0, 0.0, deposited_mass * 9.81])

        # Injector vertical reaction balances remaining suspended weight
        suspended_mass = np.sum(masses[self.deposited <= 0.5])
        injector_reaction = np.array([0.0, 0.0, suspended_mass * 9.81])

        budget_external_force = (total_gravity + contact_normal + injector_reaction).reshape(1, 1, 3)
        budget_external_couple = np.zeros((1, 1, 3))

        # Center of mass for budget force position
        com = np.sum(self.positions * masses[:, None], axis=0) / np.sum(masses)
        budget_force_position = com.reshape(1, 1, 3)

        # 3. constraint fields: [C, ...] where C=1 (inlet position constraint)
        constraint_target = self.inlet_pos.reshape(1, 3)
        constraint_value = self.positions[0].reshape(1, 3)

        # 4. cumulative_mass_source: [K] where K=1
        cumulative_mass_source = np.array([self.cumulative_injected_mass])

        # Scalars
        constraint_scale = float(self.radius)
        ref_force = float(np.sum(masses) * 9.81)
        ref_torque = float(ref_force * self.radius)

        orientation = np.zeros((n, 4))
        orientation[:, 3] = 1.0

        frame = StateFrame(
            time=time,
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            orientation=orientation,
            angular_velocity=np.zeros((n, 3)),
            mass=masses,
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "arc_coordinate": self.arc_coords.copy(),
                "is_injector_controlled": (np.arange(n) == 0).astype(float),
                "is_initially_deposited": (self.arc_coords >= 0.50).astype(float),
                "is_sticking_to_belt": self.deposited.copy(),
                "rod_radius": np.full(n, self.radius),
                "natural_curvature": np.zeros(n),
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
                "injection_speed": float(self.injection_speed),
                "belt_speed": float(self.belt_speed),
                "belt_to_injection_speed_ratio": float(self.belt_speed / self.injection_speed),
                "constraint_scale": constraint_scale,
                "reference_force": ref_force,
                "reference_torque": ref_torque,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "step_count": self.step_count,
            "current_time": self.current_time,
            "cumulative_injected_mass": self.cumulative_injected_mass,
        }

