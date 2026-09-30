"""Tablecloth pull physics simulation solver."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.meta = scene.metadata()

        # Load initial frame
        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)

        # Positions and velocities
        self.pos = np.array(init_frame.position, dtype=float, copy=True) # (227, 3)
        self.vel = np.array(init_frame.velocity, dtype=float, copy=True) # (227, 3)
        self.mass = np.array(init_frame.mass, dtype=float, copy=True)   # (227,)
        self.body_id = np.array(init_frame.body_id, dtype=int, copy=True) # (227,)
        self.orientation = np.array(init_frame.orientation, dtype=float, copy=True) # (227, 4)

        # Cloth parameters
        self.num_cloth_nodes = 221  # 17 * 13
        self.grid_shape = tuple(self.meta.get("grid_shape", [17, 13])) # (17, 13)
        self.cloth_faces = np.array(self.meta.get("topology", {}).get("faces", []), dtype=int)

        # Build cloth edge springs and initial rest lengths
        edges_set = set()
        for f in self.cloth_faces:
            edges_set.add((min(f[0], f[1]), max(f[0], f[1])))
            edges_set.add((min(f[1], f[2]), max(f[1], f[2])))
            edges_set.add((min(f[2], f[0]), max(f[2], f[0])))

        # Also add bending springs across 2D grid
        nx, ny = self.grid_shape
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                if i + 2 < nx:
                    edges_set.add((idx, idx + 2))
                if j + 2 < ny:
                    edges_set.add((idx, idx + 2 * nx))

        self.edges = np.array(list(edges_set), dtype=int)
        diff = self.pos[self.edges[:, 0]] - self.pos[self.edges[:, 1]]
        self.rest_lengths = np.linalg.norm(diff, axis=-1)

        # Simulation parameters
        self.pull_speed = float(self.config.get("pull_speed", 2.0))
        self.pull_duration = float(self.config.get("pull_duration", 0.9))
        self.table_top = float(self.config.get("table_top", 0.78))
        self.cloth_thickness = float(self.config.get("cloth", {}).get("thickness", 0.008))
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # Pulled nodes: the positive-X edge of the tablecloth
        # In grid of 17x13, row j from 0 to 12, column i = 16 (or x = +0.95)
        # Check nodes where initial x > 0.94
        self.pulled_node_indices = np.where(self.pos[:self.num_cloth_nodes, 0] > 0.94)[0]

        # Physical parameters
        self.cloth_k = 1500.0   # cloth stiffness
        self.cloth_damp = 2.0   # cloth damping
        self.friction_table = 0.25
        self.friction_cloth = 0.20
        self.sub_dt = 0.002

        # Dinnerware geometry info (indices 221 to 226)
        # body_id 1: bottle (r=0.085, h=0.72)
        # body_id 2: glass (r=0.07, h=0.3)
        # body_id 3: plate (r=0.24, h=0.035)
        # body_id 4: butter_dish (half_extents=[0.19, 0.11, 0.035])
        # body_id 5: fork (half_extents=[0.16, 0.025, 0.012])
        # body_id 6: knife (half_extents=[0.19, 0.022, 0.012])
        self.rigid_bottom_offsets = {
            221: 0.72 / 2.0,   # bottle
            222: 0.3 / 2.0,    # glass
            223: 0.035 / 2.0,  # plate
            224: 0.035,        # butter dish
            225: 0.012,        # fork
            226: 0.012,        # knife
        }
        self.rigid_radii = {
            221: 0.085,
            222: 0.07,
            223: 0.24,
            224: 0.20,
            225: 0.16,
            226: 0.19,
        }

        # Metadata / attributes template
        self.attributes = {k: np.asarray(v) for k, v in init_frame.attributes.items()}
        self.scalars = {
            "contact_length_scale": float(self.cloth_thickness),
            "constraint_scale": 1.9,
        }

    def _step(self, dt: float, current_sim_time: float) -> None:
        N = len(self.pos)
        forces = np.zeros_like(self.pos)

        # 1. Gravity on all objects
        for i in range(N):
            forces[i] += self.mass[i] * self.gravity

        # 2. Cloth internal spring forces
        p0 = self.pos[self.edges[:, 0]]
        p1 = self.pos[self.edges[:, 1]]
        v0 = self.vel[self.edges[:, 0]]
        v1 = self.vel[self.edges[:, 1]]

        delta_p = p0 - p1
        dist = np.linalg.norm(delta_p, axis=-1, keepdims=True)
        dist = np.maximum(dist, 1e-8)
        direction = delta_p / dist

        # Hookean spring with strain damping
        stretch = dist.squeeze(-1) - self.rest_lengths
        rel_vel = np.sum((v0 - v1) * direction, axis=-1)
        spring_f_mag = -self.cloth_k * stretch - self.cloth_damp * rel_vel
        f_vec = spring_f_mag[:, None] * direction

        np.add.at(forces, self.edges[:, 0], f_vec)
        np.add.at(forces, self.edges[:, 1], -f_vec)

        # 3. Pulling boundary condition during pull_duration
        is_pulling = current_sim_time < self.pull_duration

        # Update velocities and positions for non-prescribed cloth nodes
        inv_mass = 1.0 / np.maximum(self.mass, 1e-6)

        # Aerodynamic/numerical damping
        self.vel[:self.num_cloth_nodes] *= 0.999
        self.vel[self.num_cloth_nodes:] *= 0.9995

        # Integrate cloth nodes
        for i in range(self.num_cloth_nodes):
            if is_pulling and (i in self.pulled_node_indices):
                # Prescribed horizontal pull velocity +X
                self.vel[i] = np.array([self.pull_speed, 0.0, 0.0])
                self.pos[i] += self.vel[i] * dt
            else:
                self.vel[i] += (forces[i] * inv_mass[i]) * dt
                self.pos[i] += self.vel[i] * dt

            # Table contact for cloth
            cloth_z_min = self.table_top + self.cloth_thickness * 0.5
            if self.pos[i, 2] < cloth_z_min:
                # If on table surface (within table half extents roughly 1.25, 0.9)
                if abs(self.pos[i, 0]) <= 1.25 and abs(self.pos[i, 1]) <= 0.9:
                    self.pos[i, 2] = cloth_z_min
                    if self.vel[i, 2] < 0:
                        self.vel[i, 2] = 0.0
                    # Table friction on cloth
                    self.vel[i, :2] *= max(0.0, 1.0 - self.friction_table * 9.81 * dt)

        # 4. Integrate rigid bodies & interact with cloth / table
        cloth_pts = self.pos[:self.num_cloth_nodes]
        cloth_vels = self.vel[:self.num_cloth_nodes]

        for idx in range(self.num_cloth_nodes, N):
            self.vel[idx] += (forces[idx] * inv_mass[idx]) * dt
            self.pos[idx] += self.vel[idx] * dt

            bottom_offset = self.rigid_bottom_offsets[idx]
            radius = self.rigid_radii[idx]
            obj_pos = self.pos[idx]
            obj_bottom_z = obj_pos[2] - bottom_offset

            # Find nearby cloth nodes under/near the object base
            d_xy = np.linalg.norm(cloth_pts[:, :2] - obj_pos[:2], axis=-1)
            under_mask = d_xy < radius

            support_z = self.table_top
            support_vel_xy = np.zeros(2)
            has_cloth_contact = False

            if np.any(under_mask):
                cloth_near_z = cloth_pts[under_mask, 2]
                max_cloth_z = np.max(cloth_near_z)
                support_z = max(support_z, max_cloth_z + self.cloth_thickness * 0.5)
                # Average cloth velocity under object
                support_vel_xy = np.mean(cloth_vels[under_mask, :2], axis=0)
                has_cloth_contact = True

            if obj_bottom_z < support_z:
                self.pos[idx, 2] = support_z + bottom_offset
                if self.vel[idx, 2] < 0:
                    self.vel[idx, 2] = 0.0

                # Frictional coupling
                if has_cloth_contact and is_pulling:
                    # Cloth sliding beneath dinnerware transmits friction force
                    rel_v = self.vel[idx, :2] - support_vel_xy
                    # Dynamic friction impulse
                    fric_acc = min(np.linalg.norm(rel_v) / dt, self.friction_cloth * 9.81)
                    if np.linalg.norm(rel_v) > 1e-5:
                        fric_dir = -rel_v / np.linalg.norm(rel_v)
                        self.vel[idx, :2] += fric_dir * fric_acc * dt
                else:
                    # Table friction
                    self.vel[idx, :2] *= max(0.0, 1.0 - self.friction_table * 9.81 * dt)

    def advance_to(self, time: float) -> StateFrame:
        # Step forward until requested sample time
        while self.current_time < time - 1e-7:
            dt = min(self.sub_dt, time - self.current_time)
            self._step(dt, self.current_time)
            self.current_time += dt

        self.current_time = time

        # Construct required fields
        # contact_signed_distance: cloth nodes signed distance to table_top
        contact_sd = self.pos[:self.num_cloth_nodes, 2] - self.table_top

        # external_force: gravity + external pulling force
        ext_forces = np.zeros_like(self.pos)
        for i in range(len(self.pos)):
            ext_forces[i] += self.mass[i] * self.gravity

        fields = {
            "cumulative_mass_source": np.array([0.0], dtype=float),
            "contact_signed_distance": np.array(contact_sd, dtype=float),
            "constraint_value": np.array([0.0, 0.0], dtype=float),
            "constraint_target": np.array([0.0, 0.0], dtype=float),
            "external_force": np.array(ext_forces, dtype=float),
        }

        frame = StateFrame(
            time=float(time),
            position=np.array(self.pos, dtype=float, copy=True),
            velocity=np.array(self.vel, dtype=float, copy=True),
            mass=np.array(self.mass, dtype=float, copy=True),
            body_id=np.array(self.body_id, dtype=int, copy=True),
            orientation=np.array(self.orientation, dtype=float, copy=True),
            attributes=self.attributes,
            fields=fields,
            scalars=self.scalars,
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "tablecloth_pull_pbd"}

