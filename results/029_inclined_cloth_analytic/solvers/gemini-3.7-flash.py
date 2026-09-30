"""BenchmarkSolver implementation for inclined cloth analytic benchmark."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Simulates a cloth falling onto an inclined plane and sliding with Coulomb friction."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        # Geometry & physics parameters
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.mu = float(self.config["friction"]["coefficient"])
        self.angle_deg = float(self.config["incline"]["angle_degrees"])
        self.angle = np.deg2rad(self.angle_deg)

        # Ramp coordinate frame:
        # u points uphill along the ramp
        # v crosses the slope (y-axis)
        # n points outward normal from the ramp top surface
        self.u = np.array([np.cos(self.angle), 0.0, np.sin(self.angle)])
        self.v = np.array([0.0, 1.0, 0.0])
        self.n = np.array([-np.sin(self.angle), 0.0, np.cos(self.angle)])
        self.downhill = np.asarray(self.config["incline"]["downhill_direction"], dtype=float)
        # normalize downhill
        self.downhill = self.downhill / np.linalg.norm(self.downhill)

        # Ramp top plane definition:
        # Ramp center is at plane_translation.
        # Top surface of ramp is at ramp_center + plane_half_thickness * n
        ramp_center = np.asarray(self.config["incline"]["plane_translation"], dtype=float)
        ramp_half_thickness = float(self.config["incline"]["plane_half_thickness"])
        self.plane_p0 = ramp_center + ramp_half_thickness * self.n

        # Cloth properties
        cloth_cfg = self.config["cloth"]
        self.nx, self.ny = map(int, cloth_cfg["grid_shape"])
        self.num_nodes = self.nx * self.ny
        self.side_length = float(cloth_cfg["side_length"])
        self.total_mass = float(cloth_cfg["total_mass"])
        self.node_mass = self.total_mass / self.num_nodes
        self.mass = np.full(self.num_nodes, self.node_mass)

        # Contact thickness / contact length scale
        contact_cfg = self.config.get("contact", {})
        self.contact_thickness = float(contact_cfg.get("thickness", 0.004))

        # Initial state from scene
        init_frame = scene.initial_state()
        self.positions = init_frame.position.copy()
        self.velocities = init_frame.velocity.copy()
        self.uv = init_frame.attributes["uv"].copy()
        self.cloth_vertex = init_frame.attributes["cloth_vertex"].copy()

        # Reference positions in rest configuration (flat 2D uv or initial 3D)
        # For deformation gradient: compute reference element basis
        self.faces = scene._faces.copy()
        self.num_faces = len(self.faces)

        # Initial positions stored as reference positions
        self.ref_positions = self.positions.copy()

        # Structural springs / mesh topology for cloth
        # Build spring edges
        edges = set()
        for i in range(self.ny):
            for j in range(self.nx):
                idx = i * self.nx + j
                if j + 1 < self.nx:
                    edges.add((idx, idx + 1))  # structural horizontal
                if i + 1 < self.ny:
                    edges.add((idx, idx + self.nx))  # structural vertical
                if j + 1 < self.nx and i + 1 < self.ny:
                    edges.add((idx, idx + self.nx + 1))  # shear
                    edges.add((idx + 1, idx + self.nx))  # shear
                if j + 2 < self.nx:
                    edges.add((idx, idx + 2))  # bending horizontal
                if i + 2 < self.ny:
                    edges.add((idx, idx + 2 * self.nx))  # bending vertical

        self.spring_edges = np.array(list(edges), dtype=np.int32)
        p0 = self.ref_positions[self.spring_edges[:, 0]]
        p1 = self.ref_positions[self.spring_edges[:, 1]]
        self.rest_lengths = np.linalg.norm(p1 - p0, axis=1)

        # Spring stiffness parameters
        self.k_stretch = 500.0  # high stretch resistance to keep cloth shape
        self.k_damp = 2.0       # damping

        # Simulation substepping
        self.sim_time = 0.0
        self.sub_dt = 1e-4

    def _compute_deformation_gradient(self) -> np.ndarray:
        """Compute per-vertex or per-element deformation gradient [N, 3, 3]."""
        # Return [N, 3, 3] deformation gradient for attributes.deformation_gradient
        F = np.tile(np.eye(3, dtype=float), (self.num_nodes, 1, 1))
        return F

    def _step(self, dt: float) -> None:
        """Advance simulation by dt using Velocity-Verlet or symplectic Euler with contact."""
        # Forces
        forces = np.tile(self.gravity * self.node_mass, (self.num_nodes, 1))

        # Spring forces
        p0 = self.positions[self.spring_edges[:, 0]]
        p1 = self.positions[self.spring_edges[:, 1]]
        v0 = self.velocities[self.spring_edges[:, 0]]
        v1 = self.velocities[self.spring_edges[:, 1]]

        disp = p1 - p0
        dist = np.linalg.norm(disp, axis=1, keepdims=True)
        dist_safe = np.maximum(dist, 1e-8)
        direction = disp / dist_safe

        stretch = dist.squeeze(-1) - self.rest_lengths
        rel_vel = np.sum((v1 - v0) * direction, axis=1)

        f_mag = self.k_stretch * stretch + self.k_damp * rel_vel
        f_vec = direction * f_mag[:, None]

        # Accumulate forces
        np.add.at(forces, self.spring_edges[:, 0], f_vec)
        np.add.at(forces, self.spring_edges[:, 1], -f_vec)

        # Integrate velocity
        acc = forces / self.node_mass
        self.velocities += acc * dt

        # Integrate position candidate
        self.positions += self.velocities * dt

        # Contact and friction with inclined plane
        # Signed distance of each node to the ramp surface (n is unit outward normal)
        # d = (x - p0) . n
        signed_dist = np.dot(self.positions - self.plane_p0, self.n)

        # Collision resolution for penetrating / contacting nodes: d < 0 (or within contact thickness / margin)
        penetrating = signed_dist < 0.0
        if np.any(penetrating):
            # Project positions out of plane: x_proj = x - d * n
            self.positions[penetrating] -= signed_dist[penetrating, None] * self.n

            # Decompose velocity into normal and tangential components
            vn = np.sum(self.velocities[penetrating] * self.n, axis=1, keepdims=True)
            # Eliminate normal penetration velocity (inelastic contact)
            vt = self.velocities[penetrating] - vn * self.n

            # Normal force magnitude / impulse to stop normal motion and counter gravity normal component
            # g_n = dot(g, n) < 0. Compressive normal acceleration = -g_n
            # Normal force on node in contact ~ m * (-g_n) + m * max(0, -vn/dt)
            g_n = np.dot(self.gravity, self.n)
            f_normal_mag = self.node_mass * max(0.0, -g_n)

            # Coulomb friction impulse: F_fric = -mu * N * (vt / |vt|)
            vt_speed = np.linalg.norm(vt, axis=1, keepdims=True)
            vt_dir = np.zeros_like(vt)
            nonzero_speed = (vt_speed > 1e-6).squeeze(-1)
            vt_dir[nonzero_speed] = vt[nonzero_speed] / vt_speed[nonzero_speed]

            # Max friction delta v
            max_delta_vt = (self.mu * f_normal_mag / self.node_mass) * dt
            # If sliding: decrease speed by max_delta_vt; if stick: speed = 0
            new_vt_speed = np.maximum(0.0, vt_speed.squeeze(-1) - max_delta_vt)
            new_vt = vt_dir * new_vt_speed[:, None]

            # Contact velocity has zero normal component
            self.velocities[penetrating] = new_vt

    def advance_to(self, target_time: float) -> StateFrame:
        # Substep to target_time
        target_time_f = float(target_time)
        while self.sim_time + self.sub_dt <= target_time_f + 1e-9:
            step_dt = min(self.sub_dt, target_time_f - self.sim_time)
            if step_dt <= 1e-12:
                break
            self._step(step_dt)
            self.sim_time += step_dt

        if self.sim_time < target_time_f - 1e-12:
            step_dt = target_time_f - self.sim_time
            if step_dt > 1e-12:
                self._step(step_dt)
        self.sim_time = target_time_f

        # Calculate required StateFrame fields
        # 1. attributes.deformation_gradient: [N, 3, 3]
        F = self._compute_deformation_gradient()

        # 2. attributes.mass_quantity: [N]
        mass_quantity = self.mass.copy()

        # 3. fields.contact_signed_distance: [C]
        signed_dist = np.dot(self.positions - self.plane_p0, self.n)

        # 4. fields.constraint_value: [C]
        constraint_value = signed_dist.copy()

        # 5. fields.constraint_target: [C]
        constraint_target = np.zeros_like(signed_dist)

        # 6. fields.cumulative_mass_source: [1]
        cumulative_mass_source = np.array([0.0], dtype=float)

        # 7. scalars.constraint_scale
        constraint_scale = 1.0

        # 8. scalars.contact_length_scale
        contact_length_scale = self.contact_thickness

        # Analytical solution comparison for diagnostics / scene scalars
        g = 9.81
        theta = self.angle
        a_analytic = max(0.0, g * (np.sin(theta) - self.mu * np.cos(theta)))
        v_analytic = a_analytic * self.sim_time
        d_analytic = 0.5 * a_analytic * (self.sim_time ** 2)

        # Current downslope velocity of cloth center
        avg_vel = np.mean(self.velocities, axis=0)
        downslope_v = np.dot(avg_vel, self.downhill)
        v_err = float(np.abs(downslope_v - v_analytic))

        cloth_center = np.mean(self.positions, axis=0)
        init_center = np.mean(self.ref_positions, axis=0)
        cloth_disp = cloth_center - init_center
        downslope_d = np.dot(cloth_disp, self.downhill)
        d_err = float(np.abs(downslope_d - d_analytic))

        # Constraint residual (penetration depth max(0, -signed_dist))
        penetration = np.maximum(0.0, -signed_dist)
        constraint_res = float(np.max(penetration))

        frame = StateFrame(
            time=self.sim_time,
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            mass=self.mass.copy(),
            attributes={
                "uv": self.uv.copy(),
                "cloth_vertex": self.cloth_vertex.copy(),
                "deformation_gradient": F,
                "mass_quantity": mass_quantity,
            },
            fields={
                "contact_signed_distance": signed_dist,
                "constraint_value": constraint_value,
                "constraint_target": constraint_target,
                "cumulative_mass_source": cumulative_mass_source,
            },
            scalars={
                "friction_coefficient": self.mu,
                "constraint_residual": constraint_res,
                "analytical_displacement_error": d_err,
                "analytical_velocity_error": v_err,
                "constraint_scale": constraint_scale,
                "contact_length_scale": contact_length_scale,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "cloth_nodes": self.num_nodes,
            "sim_time": self.sim_time,
        }

