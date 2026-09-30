"""Elastic rod meandering on a moving belt — quasi-static discrete elastic rod solver.

Implements ``BenchmarkSolver`` for scene 093_elastic_rod_belt_meandering.

The solver uses an explicit Euler time integration of a discrete elastic rod with:
- Bending elasticity (naturally straight reference)
- Inextensibility via edge-length constraint projection
- Gravity loading
- Injector kinematic control at node 0 (position and direction fixed)
- Belt sticking condition: deposited nodes move with belt velocity
- Contact detection with z=0 plane; newly contacted nodes become deposited

Quasi-static regime is used: velocities are damped heavily so the rod shape
is determined by force balance rather than inertial dynamics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        cfg = scene.config
        self.cfg = cfg

        # Load topology
        topo_path = Path(__file__).parent.parent / cfg["rod"]["topology_asset"]
        topo = json.loads(topo_path.read_text(encoding="utf-8"))
        self.init_position = np.asarray(topo["vertices"], dtype=np.float64)
        self.edges = np.asarray(topo["edges"], dtype=np.int64)
        self.arc_coord = np.asarray(topo["arc_coordinate"], dtype=np.float64)
        self.deposited_mask_init = np.asarray(topo["deposited_mask"], dtype=np.float64)

        self.n_nodes = len(self.init_position)
        self.n_edges = len(self.edges)

        # Rod parameters
        self.radius = float(cfg["rod"]["radius"])
        self.density = float(cfg["rod"]["density"])
        self.E = float(cfg["rod"]["young_modulus"])
        self.nu = float(cfg["rod"]["poisson_ratio"])
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)
        self.inextensible = bool(cfg["rod"].get("inextensible", True))

        # Cross-section properties
        area = np.pi * self.radius ** 2
        self.mass_per_length = self.density * area
        # Second moment of area for circular cross-section
        self.I_bend = np.pi * self.radius ** 4 / 4.0
        self.bending_stiffness = self.E * self.I_bend

        # Injector parameters
        inj = cfg["injector"]
        self.injector_pos = np.asarray(inj["outlet_position"], dtype=np.float64)
        self.injector_axis = np.asarray(inj["axis"], dtype=np.float64)
        self.injector_axis /= np.linalg.norm(self.injector_axis)
        self.injection_speed = float(inj["injection_speed"])
        self.inj_start = float(inj["start_time"])
        self.inj_end = float(inj["end_time"])

        # Belt parameters
        belt_cfg = cfg["belt"]
        self.belt_travel = np.asarray(belt_cfg["travel_direction"], dtype=np.float64)
        self.belt_travel /= np.linalg.norm(self.belt_travel)
        self.belt_surface_z = float(belt_cfg["surface_z"])
        self.belt_initial_speed = float(belt_cfg["initial_speed"])
        self.belt_matched_speed = float(belt_cfg["matched_speed"])
        self.belt_switch_time = float(belt_cfg["speed_switch_time"])

        # Compute rest edge lengths from initial configuration
        self.rest_lengths = np.linalg.norm(
            self.init_position[self.edges[:, 1]] - self.init_position[self.edges[:, 0]], axis=1
        )

        # Compute lumped masses
        self._compute_masses()

        # Initialize state
        self.position = self.init_position.copy()
        self.velocity = np.zeros((self.n_nodes, 3), dtype=np.float64)
        # Set initial velocity for deposited nodes
        dep = self.deposited_mask_init > 0.5
        self.velocity[dep] = self.belt_travel * self.belt_initial_speed

        self.orientation = np.zeros((self.n_nodes, 4), dtype=np.float64)
        self.orientation[:, 3] = 1.0
        self.angular_velocity = np.zeros((self.n_nodes, 3), dtype=np.float64)

        # Deposited mask tracks which nodes are stuck to belt
        self.is_deposited = self.deposited_mask_init.copy()

        # Cumulative mass injected
        self.cumulative_injected_mass = 0.0

        # Time tracking
        self.current_time = 0.0

        # Initial mass
        self.initial_total_mass = float(np.sum(self.mass))

        # Meandering parameters
        # Natural meandering frequency from elastica theory
        # For a rod falling under gravity onto a surface, the meandering wavelength
        # depends on the bending length and speed mismatch
        self.gravito_bending_length = float(cfg["rod"]["gravito_bending_length"])

        # Damping coefficient for quasi-static behavior
        self.damping = 50.0

        # Reference scales for trajectory output
        self.reference_force = self.mass_per_length * 9.81 * self.gravito_bending_length
        self.reference_torque = self.reference_force * self.gravito_bending_length

        # Constraint: inextensibility — track max edge strain
        self.constraint_scale = 1.0  # normalized by rest length

    def _compute_masses(self):
        """Compute lumped nodal masses based on edge lengths."""
        n = self.n_nodes
        lengths = self.rest_lengths
        lumped_length = np.zeros(n, dtype=np.float64)
        lumped_length[self.edges[:, 0]] += lengths / 2.0
        lumped_length[self.edges[:, 1]] += lengths / 2.0
        self.mass = self.mass_per_length * lumped_length

    def _belt_speed(self, t: float) -> float:
        """Get belt speed at time t, with smooth transition at switch time."""
        if t < self.belt_switch_time:
            return self.belt_initial_speed
        else:
            # Smooth transition over ~1 second
            tau = 0.5
            alpha = min(1.0, (t - self.belt_switch_time) / tau)
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
            return self.belt_initial_speed + alpha * (self.belt_matched_speed - self.belt_initial_speed)

    def _compute_bending_forces(self, pos):
        """Compute bending forces using discrete elastic rod model.
        
        Uses the angle between consecutive edges to compute bending moments.
        For a naturally straight rod, the bending energy is proportional to
        the squared curvature.
        """
        forces = np.zeros_like(pos)
        
        # Edge vectors
        edge_vecs = pos[self.edges[:, 1]] - pos[self.edges[:, 0]]
        edge_lengths = np.linalg.norm(edge_vecs, axis=1)
        edge_lengths = np.maximum(edge_lengths, 1e-15)
        edge_dirs = edge_vecs / edge_lengths[:, None]
        
        # For each interior node (connected to two edges), compute bending force
        # Bending energy: E = 0.5 * EI * integral(kappa^2 ds)
        # Discrete: E ≈ 0.5 * EI * sum((theta_i)^2 / l_i)
        # where theta_i is the angle between consecutive edges
        
        for i in range(self.n_edges - 1):
            e0 = edge_dirs[i]
            e1 = edge_dirs[i + 1]
            
            # Angle between edges (for small angles, sin(theta) ≈ theta)
            cos_angle = np.clip(np.dot(e0, e1), -1.0, 1.0)
            
            # Curvature vector (discrete)
            # kappa_vec = (e1 - e0) / avg_length
            avg_len = 0.5 * (edge_lengths[i] + edge_lengths[i + 1])
            
            # Tangent change
            dt = e1 - e0
            
            # Bending moment magnitude
            # M = EI * kappa ≈ EI * |dt| / avg_len
            kappa_mag = np.linalg.norm(dt) / avg_len
            
            # Force from bending: F = -grad(E_bend)
            # For discrete rod, this gives forces at three consecutive nodes
            # The force is perpendicular to the rod centerline
            
            # Effective bending stiffness scaled by element length
            k_bend = self.bending_stiffness / (avg_len ** 2)
            
            # Node indices
            n0 = self.edges[i, 0]
            n1 = self.edges[i, 1]  # = self.edges[i+1, 0]
            n2 = self.edges[i + 1, 1]
            
            # Bending force (linearized for small deflections)
            # F = k_bend * (pos[n0] - 2*pos[n1] + pos[n2])
            laplacian = pos[n0] - 2.0 * pos[n1] + pos[n2]
            
            fb = k_bend * laplacian
            
            forces[n0] += fb
            forces[n1] -= 2.0 * fb
            forces[n2] += fb
        
        return forces

    def _project_inextensibility(self, pos, n_iter=20):
        """Project positions to satisfy inextensibility constraints."""
        if not self.inextensible:
            return pos
        
        for _ in range(n_iter):
            for e_idx in range(self.n_edges):
                i, j = self.edges[e_idx]
                diff = pos[j] - pos[i]
                dist = np.linalg.norm(diff)
                if dist < 1e-15:
                    continue
                rest = self.rest_lengths[e_idx]
                error = dist - rest
                
                # Correction along edge direction
                correction = 0.5 * error * diff / dist
                
                # Don't move injector node
                if i != 0:
                    pos[i] += correction
                if j != 0:
                    pos[j] -= correction
        
        return pos

    def _detect_contact_and_deposit(self, pos, vel, belt_vel, dt):
        """Detect contact with belt surface and mark nodes as deposited."""
        z_surface = self.belt_surface_z
        
        for i in range(self.n_nodes):
            if self.is_deposited[i] > 0.5:
                continue
            
            # Check if node has reached or passed through the surface
            if pos[i, 2] <= z_surface + self.radius:
                # Project to surface
                pos[i, 2] = z_surface
                # Mark as deposited
                self.is_deposited[i] = 1.0
                # Set velocity to belt velocity
                vel[i] = belt_vel * self.belt_travel

    def _step(self, dt: float, t: float):
        """Advance simulation by one timestep."""
        belt_speed = self._belt_speed(t)
        belt_vel_vec = belt_speed * self.belt_travel
        
        pos = self.position.copy()
        vel = self.velocity.copy()
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Gravity
        forces += self.mass[:, None] * self.gravity[None, :]
        
        # Bending forces
        bend_forces = self._compute_bending_forces(pos)
        forces += bend_forces
        
        # Damping (quasi-static: heavy damping)
        forces -= self.damping * self.mass[:, None] * vel
        
        # Injector constraint: node 0 fixed at injector position, directed downward
        # This is handled by directly setting position after integration
        
        # Semi-implicit Euler integration
        acc = np.zeros_like(pos)
        nonzero_mass = self.mass > 1e-20
        acc[nonzero_mass] = forces[nonzero_mass] / self.mass[nonzero_mass, None]
        
        vel += dt * acc
        pos += dt * vel
        
        # Apply injector boundary condition
        pos[0] = self.injector_pos
        vel[0] = self.injection_speed * (-self.injector_axis)  # downward injection
        
        # Enforce direction at injector: first edge points along injector axis
        if self.n_nodes > 1:
            edge_dir = pos[1] - pos[0]
            edge_len = np.linalg.norm(edge_dir)
            if edge_len > 1e-15:
                # Project edge direction onto injector axis
                target_dir = -self.injector_axis  # rod goes downward
                pos[1] = pos[0] + edge_len * target_dir
        
        # Contact detection and deposition
        self._detect_contact_and_deposit(pos, vel, belt_speed, dt)
        
        # Enforce deposited nodes move with belt
        dep = self.is_deposited > 0.5
        pos[dep, 2] = self.belt_surface_z  # keep on surface
        vel[dep] = belt_vel_vec
        
        # Project inextensibility
        pos = self._project_inextensibility(pos, n_iter=10)
        
        # Re-enforce constraints after projection
        pos[0] = self.injector_pos
        vel[0] = self.injection_speed * (-self.injector_axis)
        pos[dep, 2] = self.belt_surface_z
        vel[dep] = belt_vel_vec
        
        self.position = pos
        self.velocity = vel
        
        # Update cumulative injected mass
        # Mass injection rate = mass_per_length * injection_speed
        dm = self.mass_per_length * self.injection_speed * dt
        self.cumulative_injected_mass += dm

    def advance_to(self, time: float) -> StateFrame:
        """Advance simulation to the given time and return state frame."""
        if time < self.current_time:
            # Should not happen, but handle gracefully
            pass
        elif time > self.current_time:
            # Sub-step for stability
            max_dt = 0.001  # 1ms substeps for stability
            total_dt = time - self.current_time
            n_steps = max(1, int(np.ceil(total_dt / max_dt)))
            dt = total_dt / n_steps
            
            for _ in range(n_steps):
                self._step(dt, self.current_time)
                self.current_time += dt
        
        # Build state frame
        n = self.n_nodes
        
        # Compute external forces for budget output
        ext_forces = np.zeros((n, 3), dtype=np.float64)
        # Gravity
        ext_forces += self.mass[:, None] * self.gravity[None, :]
        # Belt reaction on deposited nodes (normal force balancing gravity)
        dep = self.is_deposited > 0.5
        ext_forces[dep, 2] -= self.mass[dep] * self.gravity[2]  # cancel gravity on surface
        
        # External force application points
        force_positions = self.position.copy()
        
        # Budget: single domain (G=1), single term per node (K=n)
        budget_ext_force = ext_forces[np.newaxis, :, :]  # [1, N, 3]
        budget_force_pos = force_positions[np.newaxis, :, :]  # [1, N, 3]
        budget_ext_couple = np.zeros((1, n, 3), dtype=np.float64)  # no applied couples
        
        # Constraint: inextensibility - report edge strains
        edge_strains = np.zeros(self.n_edges, dtype=np.float64)
        for e_idx in range(self.n_edges):
            i, j = self.edges[e_idx]
            curr_len = np.linalg.norm(self.position[j] - self.position[i])
            edge_strains[e_idx] = (curr_len - self.rest_lengths[e_idx]) / self.rest_lengths[e_idx]
        
        # Constraint target is zero strain (inextensible)
        constraint_target = np.zeros(self.n_edges, dtype=np.float64)
        
        # Orientation quaternions from local frame
        orientation = self.orientation.copy()
        
        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=orientation,
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "arc_coordinate": self.arc_coord.copy(),
                "is_injector_controlled": (np.arange(n) == 0).astype(np.float64),
                "is_initially_deposited": self.deposited_mask_init.copy(),
                "is_sticking_to_belt": self.is_deposited.copy(),
                "rod_radius": np.full(n, self.radius),
                "natural_curvature": np.zeros(n, dtype=np.float64),
            },
            fields={
                "budget_external_force": budget_ext_force,
                "budget_force_position": budget_force_pos,
                "budget_external_couple": budget_ext_couple,
                "constraint_value": edge_strains,
                "constraint_target": constraint_target,
                "cumulative_mass_source": np.array([self.cumulative_injected_mass], dtype=np.float64),
            },
            scalars={
                "constraint_scale": self.constraint_scale,
                "reference_force": self.reference_force,
                "reference_torque": self.reference_torque,
                "injection_speed": self.injection_speed,
                "initial_belt_speed": self.belt_initial_speed,
                "matched_belt_speed": self.belt_matched_speed,
                "speed_switch_time": self.belt_switch_time,
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "final_time": self.current_time,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "total_injected_mass": self.cumulative_injected_mass,
            "n_deposited": int(np.sum(self.is_deposited > 0.5)),
        }
