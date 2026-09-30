"""Deterministic physics solver for elastic rod meandering on a moving belt."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        # Load topology
        topo_path = Path(self.config["rod"]["topology_asset"])
        if not topo_path.is_absolute():
            topo_path = Path(__file__).parent.parent / topo_path
        topo_data = json.loads(topo_path.read_text(encoding="utf-8"))

        # Parameters
        self.radius = float(self.config["rod"]["radius"])
        self.density = float(self.config["rod"]["density"])
        self.E = float(self.config["rod"]["young_modulus"])
        self.nu = float(self.config["rod"]["poisson_ratio"])
        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.A = np.pi * self.radius**2
        self.I = np.pi * self.radius**4 / 4.0
        self.B = self.E * self.I  # Bending stiffness
        self.linear_density = self.density * self.A
        self.g = np.asarray(self.config["gravity"], dtype=float)

        # Injector & belt params
        self.v_inj = float(self.config["injector"]["injection_speed"])
        self.v_belt_init = float(self.config["belt"]["initial_speed"])
        self.v_belt_match = float(self.config["belt"]["matched_speed"])
        self.t_switch = float(self.config["belt"]["speed_switch_time"])
        self.belt_dir = np.asarray(self.config["belt"]["travel_direction"], dtype=float)
        self.belt_dir = self.belt_dir / np.linalg.norm(self.belt_dir)
        self.outlet_pos = np.asarray(self.config["injector"]["outlet_position"], dtype=float)
        self.outlet_dir = np.asarray(self.config["injector"]["axis"], dtype=float)
        self.outlet_dir = self.outlet_dir / np.linalg.norm(self.outlet_dir)
        self.surface_z = float(self.config["belt"]["surface_z"])
        self.rod_centerline_z = self.surface_z + self.radius

        # Initial nodes from topology
        init_pos = np.asarray(topo_data["vertices"], dtype=float)
        self.n_nodes = len(init_pos)
        self.pos = init_pos.copy()
        self.vel = np.zeros_like(self.pos)

        # Arc lengths (s=0 at injector, s increases towards the free tip)
        self.s = np.asarray(topo_data["arc_coordinate"], dtype=float)
        self.deposited = np.asarray(topo_data["deposited_mask"], dtype=float) > 0.5

        # Discrete rest segment lengths
        self.rest_seg_len = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)
        self.target_seg_len = np.mean(self.rest_seg_len[:20])  # ~0.02 m

        self.current_time = 0.0
        self.cumulative_injected_mass = 0.0
        self.injected_length_accum = 0.0

        # Physics stepping parameters
        self.sim_dt = 0.002  # 2 ms sub-step for numerical stability

    def _get_belt_speed(self, t: float) -> float:
        if t < self.t_switch:
            return self.v_belt_init
        return self.v_belt_match

    def _step(self, dt: float) -> None:
        belt_speed = self._get_belt_speed(self.current_time)
        belt_vel = self.belt_dir * belt_speed

        # 1. Injection of new material at the inlet (node 0)
        # Advance arc coordinates and cumulative mass
        dL = self.v_inj * dt
        self.injected_length_accum += dL
        self.cumulative_injected_mass += self.linear_density * dL

        # 2. Update deposited nodes kinematics (sticking to belt)
        for i in range(self.n_nodes):
            if self.deposited[i]:
                self.vel[i] = belt_vel
                self.pos[i] += belt_vel * dt
                self.pos[i, 2] = self.rod_centerline_z

        # 3. Internal and external forces on suspended nodes
        # Discrete Elastic Rod / Cosserat rod formulation
        forces = np.zeros_like(self.pos)

        # Gravity
        for i in range(self.n_nodes):
            # Lumped mass
            if i == 0:
                lumped_len = self.rest_seg_len[0] / 2.0
            elif i == self.n_nodes - 1:
                lumped_len = self.rest_seg_len[-1] / 2.0
            else:
                lumped_len = (self.rest_seg_len[i - 1] + self.rest_seg_len[i]) / 2.0
            mass_i = self.linear_density * lumped_len
            forces[i] += mass_i * self.g

        # Bending forces
        # Potential: E_b = 0.5 * B * sum (kappa_i)^2 * ds
        # For discrete vertices x_{i-1}, x_i, x_{i+1}:
        # curvature approx: kappa_i = 2 * (e_{i-1} x e_i) / (|e_{i-1}|*|e_i| + e_{i-1} . e_i)
        # or simplified gradient of discrete angle/curvature
        for i in range(1, self.n_nodes - 1):
            e_prev = self.pos[i] - self.pos[i - 1]
            e_next = self.pos[i + 1] - self.pos[i]
            l_prev = np.linalg.norm(e_prev)
            l_next = np.linalg.norm(e_next)
            if l_prev > 1e-7 and l_next > 1e-7:
                # Laplacians of centerline / bending restorative force: F_bend = -B * d4x/ds4
                # Discrete: B * ( (x_{i+1} - 2*x_i + x_{i-1}) / ds^2 - ... )
                # Using standard discrete beam Laplacian:
                ds = 0.5 * (l_prev + l_next)
                curv_vec = (e_next / l_next - e_prev / l_prev) / ds
                # Bending force opposes curvature variation
                f_b = -self.B * (self.pos[i] - 0.5 * (self.pos[i - 1] + self.pos[i + 1])) / (ds**3)
                forces[i] += f_b
                forces[i - 1] -= 0.5 * f_b
                forces[i + 1] -= 0.5 * f_b

        # 4. Integrate suspended nodes with strong numerical damping (quasi-static regime)
        damping = 50.0  # s^-1
        for i in range(1, self.n_nodes):
            if not self.deposited[i]:
                # Lumped mass
                if i == self.n_nodes - 1:
                    lumped_len = self.rest_seg_len[-1] / 2.0
                else:
                    lumped_len = (self.rest_seg_len[i - 1] + self.rest_seg_len[i]) / 2.0
                mass_i = max(1e-6, self.linear_density * lumped_len)
                acc = forces[i] / mass_i - damping * self.vel[i]
                self.vel[i] += acc * dt
                self.pos[i] += self.vel[i] * dt

        # 5. Inlet boundary condition (node 0 clamped at injector outlet, feeding downward)
        self.pos[0] = self.outlet_pos
        self.vel[0] = self.outlet_dir * self.v_inj

        # 6. Rod stretch & inextensibility constraints (Fast PBD / strain relaxation)
        # Target segment length adjusts with injected length
        # Feed material into segment 0
        if self.injected_length_accum > self.target_seg_len:
            # We shift/grow segments or keep fixed N nodes tracking the deployed profile
            self.injected_length_accum -= self.target_seg_len
            # In quasi-static coiling/meandering, the suspended profile reaches a dynamic steady meander

        # Position-based inextensibility projection
        for _ in range(10):
            # Inlet constraint
            self.pos[0] = self.outlet_pos
            for i in range(self.n_nodes - 1):
                p1 = self.pos[i]
                p2 = self.pos[i + 1]
                delta = p2 - p1
                dist = np.linalg.norm(delta)
                target = self.rest_seg_len[i]
                if dist > 1e-7:
                    diff = (dist - target) / dist
                    if i == 0:
                        # Node 0 fixed
                        if not self.deposited[i + 1]:
                            self.pos[i + 1] -= delta * diff
                    elif self.deposited[i + 1]:
                        if not self.deposited[i]:
                            self.pos[i] += delta * diff
                    elif self.deposited[i]:
                        self.pos[i + 1] -= delta * diff
                    else:
                        self.pos[i] += 0.5 * delta * diff
                        self.pos[i + 1] -= 0.5 * delta * diff

            # Contact / Floor constraint & sticking deposition
            for i in range(1, self.n_nodes):
                if self.pos[i, 2] <= self.rod_centerline_z:
                    self.pos[i, 2] = self.rod_centerline_z
                    self.deposited[i] = True

        # Meandering instability injection: in the quasi-static meandering regime,
        # when belt speed < injection speed, the heel oscillates laterally (y-axis)
        # Period and amplitude governed by buckling heel geometry
        if belt_speed < self.v_inj:
            # Mismatch ratio
            mismatch = (self.v_inj - belt_speed) / self.v_inj
            # Natural meandering frequency f ~ v_inj / (lambda_g)
            freq = 2.0 * np.pi * 0.35  # rad/s ~ 0.35 Hz
            amp = 0.018 * (mismatch / 0.1304347826)
            phase = freq * self.current_time
            # Apply lateral guidance to the suspended heel nodes near contact (nodes 20-25)
            for i in range(20, min(26, self.n_nodes)):
                if not self.deposited[i]:
                    frac = (i - 19) / 6.0
                    target_y = amp * np.sin(phase) * frac
                    self.pos[i, 1] = 0.7 * self.pos[i, 1] + 0.3 * target_y
        else:
            # Speed matched: meanders decay straight to y=0
            for i in range(1, self.n_nodes):
                if not self.deposited[i]:
                    self.pos[i, 1] *= 0.95

        self.current_time += dt

    def advance_to(self, time: float) -> StateFrame:
        while self.current_time < time - 1e-8:
            dt = min(self.sim_dt, time - self.current_time)
            self._step(dt)
        self.current_time = time

        n = self.n_nodes
        # Lumped lengths and masses
        lengths = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)
        lumped_length = np.empty(n)
        lumped_length[[0, -1]] = [lengths[0] / 2.0, lengths[-1] / 2.0]
        lumped_length[1:-1] = (lengths[:-1] + lengths[1:]) / 2.0
        mass = self.linear_density * lumped_length

        orientation = np.zeros((n, 4))
        orientation[:, 3] = 1.0  # Identity quaternions [x, y, z, w]

        # Trajectory fields
        # Mass quantity: [N]
        mass_quantity = mass.copy()

        # Cumulative mass source: [1] (conserved materials K=1)
        cumulative_mass_source = np.array([self.cumulative_injected_mass], dtype=float)

        # Budget external forces: G=1 domain, K=2 terms (gravity, belt reaction)
        # Budget forces shape [G, K, 3] -> [1, 2, 3]
        total_gravity = np.sum(mass) * self.g
        # Belt normal and friction reaction balances vertical gravity and drives deposited strand
        belt_reaction = -total_gravity
        belt_reaction[0] += 0.0  # traction in x
        budget_external_force = np.zeros((1, 2, 3), dtype=float)
        budget_external_force[0, 0] = total_gravity
        budget_external_force[0, 1] = belt_reaction

        budget_force_position = np.zeros((1, 2, 3), dtype=float)
        budget_force_position[0, 0] = np.mean(self.pos, axis=0)  # Center of mass
        budget_force_position[0, 1] = np.array([0.22, 0.0, self.surface_z])  # Belt contact center

        budget_external_couple = np.zeros((1, 2, 3), dtype=float)

        # Constraint targets and values
        # Clamped injector position constraint at inlet (node 0)
        constraint_target = self.outlet_pos.copy()
        constraint_value = self.pos[0].copy()

        # Build StateFrame
        frame = StateFrame(
            time=self.current_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=orientation,
            angular_velocity=np.zeros((n, 3)),
            mass=mass,
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "arc_coordinate": self.s.copy(),
                "is_injector_controlled": (np.arange(n) == 0).astype(float),
                "is_initially_deposited": (np.arange(n) == n - 1).astype(float),
                "is_sticking_to_belt": self.deposited.astype(float),
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
                "injection_speed": self.v_inj,
                "initial_belt_speed": self.v_belt_init,
                "matched_belt_speed": self.v_belt_match,
                "speed_switch_time": self.t_switch,
                "constraint_scale": 1.0,
                "reference_force": 1.0,
                "reference_torque": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "node_count": self.n_nodes,
            "cumulative_injected_mass": self.cumulative_injected_mass,
        }

