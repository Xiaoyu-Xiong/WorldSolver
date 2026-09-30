"""Deterministic physical solver for cargo overload boat sinking scene."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Initial state from scene
        init_frame = scene.initial_state()
        self.N = len(init_frame.position)
        self.body_id = np.asarray(init_frame.body_id, dtype=int)
        self.mass = np.asarray(init_frame.mass, dtype=float)

        self.water_indices = np.where(self.body_id == 0)[0]
        self.boat_idx = np.where(self.body_id == 1)[0][0]
        self.light_idx = np.where(self.body_id == 2)[0][0]
        self.heavy_idx = np.where(self.body_id == 3)[0][0]

        self.pos = np.asarray(init_frame.position, dtype=float).copy()
        self.vel = np.asarray(init_frame.velocity, dtype=float).copy()

        # Rigid body state: orientations (quaternions xyzw) and angular velocities
        self.quat = np.zeros((self.N, 4), dtype=float)
        self.quat[:, 3] = 1.0  # identity [0, 0, 0, 1]
        self.ang_vel = np.zeros((self.N, 3), dtype=float)

        # Rigid body properties
        ic = self.config["initial_conditions"]
        self.light_release = float(ic["light_release_time"])
        self.heavy_release = float(ic["heavy_release_time"])
        self.r_sphere = float(ic["sphere_radius"])
        self.m_boat = float(ic["boat_mass"])
        self.m_light = float(self.mass[self.light_idx])
        self.m_heavy = float(self.mass[self.heavy_idx])

        self.hull_L = float(ic["hull_length"])
        self.hull_W = float(ic["hull_width"])
        self.hull_H = float(ic["hull_height"])

        # Inertia tensors (diagonal approx)
        self.I_boat = np.diag([
            self.m_boat * (self.hull_W**2 + self.hull_H**2) / 12.0,
            self.m_boat * (self.hull_L**2 + self.hull_H**2) / 12.0,
            self.m_boat * (self.hull_L**2 + self.hull_W**2) / 12.0,
        ])
        self.I_inv_boat = np.linalg.inv(self.I_boat)

        self.I_light = 0.4 * self.m_light * self.r_sphere**2 * np.eye(3)
        self.I_heavy = 0.4 * self.m_heavy * self.r_sphere**2 * np.eye(3)

        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.water_depth = float(self.config["fluid"]["initial_water_depth"])
        self.rho_water = float(self.config["fluid"]["density"])

        # Boundary domain bounds
        self.bounds_min = np.asarray(self.config["domain"]["bounds_min"], dtype=float)
        self.bounds_max = np.asarray(self.config["domain"]["bounds_max"], dtype=float)

        # Boundary samples for trajectory contract
        self.boundary_normals = np.array([
            [0.0, 0.0, 1.0],   # bottom
            [0.0, -1.0, 0.0],  # back
            [0.0, 1.0, 0.0],   # front
            [1.0, 0.0, 0.0],   # left
            [-1.0, 0.0, 0.0],  # right
        ], dtype=float)
        self.boundary_vel = np.zeros_like(self.boundary_normals)

        # Budget setup
        self.G = 4  # 4 domains: water(0), boat(1), light(2), heavy(3)
        self.domain_weights = np.zeros((self.G, self.N), dtype=float)
        for g in range(self.G):
            self.domain_weights[g, self.body_id == g] = 1.0

        self.cum_ext_impulse = np.zeros((self.G, 3), dtype=float)
        self.sim_time = 0.0
        self.sub_dt = 0.002

        # Boat flooding state
        self.boat_flooded = False
        self.boat_sunk = False

    def _integrate_step(self, dt: float) -> None:
        t = self.sim_time

        # 1. External forces (Gravity)
        F_ext = np.zeros((self.G, 1, 3), dtype=float)
        for g in range(self.G):
            mask = (self.body_id == g)
            total_m = np.sum(self.mass[mask])
            F_ext[g, 0] = total_m * self.gravity

        # Apply gravity to water
        self.vel[self.water_indices] += self.gravity * dt

        # Apply gravity to rigid bodies if released
        # Boat is always active
        F_boat = self.m_boat * self.gravity.copy()
        T_boat = np.zeros(3, dtype=float)

        # Light sphere
        F_light = np.zeros(3, dtype=float)
        T_light = np.zeros(3, dtype=float)
        if t >= self.light_release:
            F_light += self.m_light * self.gravity

        # Heavy sphere
        F_heavy = np.zeros(3, dtype=float)
        T_heavy = np.zeros(3, dtype=float)
        if t >= self.heavy_release:
            F_heavy += self.m_heavy * self.gravity

        # 2. Hydrodynamics / Buoyancy
        # Water level reference (average or local)
        water_z = self.pos[self.water_indices, 2]
        mean_surface_z = np.mean(water_z[water_z > self.water_depth * 0.7]) if np.any(water_z > self.water_depth * 0.7) else self.water_depth

        # Boat hydrodynamics
        boat_p = self.pos[self.boat_idx]
        boat_v = self.vel[self.boat_idx]
        boat_w = self.ang_vel[self.boat_idx]

        # Hull dimensions & submerged volume
        hull_bottom_z = boat_p[2]
        hull_top_z = boat_p[2] + self.hull_H
        submerged_depth = np.clip(mean_surface_z - hull_bottom_z, 0.0, self.hull_H)

        # Check if water overtopped the boat rim
        if not self.boat_flooded:
            if hull_top_z < mean_surface_z + 0.02:
                self.boat_flooded = True

        if not self.boat_flooded:
            # Displaced volume
            disp_vol = self.hull_L * self.hull_W * submerged_depth
            F_buoy = np.array([0.0, 0.0, self.rho_water * disp_vol * 9.81])
            F_boat += F_buoy

            # Hydrodynamic drag and restoring damping
            F_boat += -np.array([40.0, 40.0, 150.0]) * boat_v
            # Pitch/roll restoring torque
            # Extract pitch (around y) and roll (around x) from quaternion
            qx, qy, qz, qw = self.quat[self.boat_idx]
            sin_roll = 2.0 * (qw * qx + qy * qz)
            sin_pitch = 2.0 * (qw * qy - qz * qx)
            T_boat += -np.array([120.0 * sin_roll, 250.0 * sin_pitch, 50.0 * boat_w[2]])
            T_boat += -np.array([25.0, 50.0, 15.0]) * boat_w
        else:
            # Sunk/flooded boat: only minor buoyancy from hull structure material
            disp_vol = 0.042  # ~42 liters structural volume
            F_buoy = np.array([0.0, 0.0, self.rho_water * disp_vol * 9.81])
            F_boat += F_buoy
            F_boat += -np.array([60.0, 60.0, 120.0]) * boat_v
            T_boat += -15.0 * boat_w

        # Spheres hydrodynamics
        # Light sphere
        p_l = self.pos[self.light_idx]
        v_l = self.vel[self.light_idx]
        if p_l[2] < mean_surface_z + self.r_sphere:
            sub_h = np.clip(mean_surface_z - (p_l[2] - self.r_sphere), 0.0, 2.0 * self.r_sphere)
            vol_sub = np.pi * sub_h**2 * (self.r_sphere - sub_h / 3.0)
            F_buoy_l = np.array([0.0, 0.0, self.rho_water * vol_sub * 9.81])
            F_light += F_buoy_l
            F_light += -np.array([15.0, 15.0, 25.0]) * v_l

        # Heavy sphere
        p_h = self.pos[self.heavy_idx]
        v_h = self.vel[self.heavy_idx]
        if p_h[2] < mean_surface_z + self.r_sphere:
            sub_h = np.clip(mean_surface_z - (p_h[2] - self.r_sphere), 0.0, 2.0 * self.r_sphere)
            vol_sub = np.pi * sub_h**2 * (self.r_sphere - sub_h / 3.0)
            F_buoy_h = np.array([0.0, 0.0, self.rho_water * vol_sub * 9.81])
            F_heavy += F_buoy_h
            F_heavy += -np.array([25.0, 25.0, 40.0]) * v_h

        # 3. Cargo - Boat Contacts and Interactions
        # Light sphere inside / interacting with boat
        if t >= self.light_release:
            # Relative position in boat frame
            rel_l = p_l - boat_p
            # Contact with boat bottom
            target_z = boat_p[2] + self.r_sphere
            if not self.boat_flooded:
                # Retained in boat if within hull bounds
                if abs(rel_l[0]) < self.hull_L * 0.48 and abs(rel_l[1]) < self.hull_W * 0.48:
                    if p_l[2] < target_z + 0.02:
                        pen = target_z - p_l[2]
                        if pen > 0:
                            f_norm = 15000.0 * pen - 300.0 * (v_l[2] - boat_v[2])
                            f_norm = max(0.0, f_norm)
                            F_contact = np.array([0.0, 0.0, f_norm])
                            F_light += F_contact
                            F_boat -= F_contact
                            # Torque on boat
                            T_boat += np.cross(rel_l, -F_contact)
                            # Horizontal friction / containment
                            f_fric = -30.0 * (v_l[:2] - boat_v[:2])
                            F_light[:2] += f_fric
                            F_boat[:2] -= f_fric
            else:
                # Boat flooded: buoyant light sphere is pushed out by buoyancy!
                if p_l[2] < target_z and abs(rel_l[0]) < self.hull_L * 0.45:
                    f_norm = max(0.0, 10000.0 * (target_z - p_l[2]))
                    F_light[2] += f_norm

        # Heavy sphere inside / interacting with boat
        if t >= self.heavy_release:
            rel_h = p_h - boat_p
            target_z = boat_p[2] + self.r_sphere
            if abs(rel_h[0]) < self.hull_L * 0.55 and abs(rel_h[1]) < self.hull_W * 0.55:
                if p_h[2] < target_z + 0.03:
                    pen = target_z - p_h[2]
                    if pen > 0:
                        f_norm = 35000.0 * pen - 800.0 * (v_h[2] - boat_v[2])
                        f_norm = max(0.0, f_norm)
                        F_contact = np.array([0.0, 0.0, f_norm])
                        F_heavy += F_contact
                        F_boat -= F_contact
                        # Strong impact torque on boat
                        T_boat += np.cross(rel_h, -F_contact)
                        # Friction / containment
                        f_fric = -80.0 * (v_h[:2] - boat_v[:2])
                        F_heavy[:2] += f_fric
                        F_boat[:2] -= f_fric

        # Sphere-Sphere contact
        if t >= self.heavy_release:
            d_spheres = p_h - p_l
            dist = np.linalg.norm(d_spheres)
            if dist < 2.0 * self.r_sphere and dist > 1e-6:
                n_cc = d_spheres / dist
                pen_s = 2.0 * self.r_sphere - dist
                f_s = (15000.0 * pen_s - 100.0 * np.dot(v_h - v_l, n_cc)) * n_cc
                F_heavy += f_s
                F_light -= f_s

        # 4. Floor and Pool Wall Collisions for Rigid Bodies
        # Boat floor collision
        if boat_p[2] < 0.02:
            pen = 0.02 - boat_p[2]
            F_boat[2] += 40000.0 * pen - 500.0 * boat_v[2]
            boat_v[0] *= 0.9
            boat_v[1] *= 0.9

        # Spheres floor collision
        if t >= self.light_release and p_l[2] < self.r_sphere + 0.01:
            pen = (self.r_sphere + 0.01) - p_l[2]
            F_light[2] += 20000.0 * pen - 200.0 * v_l[2]

        if t >= self.heavy_release and p_h[2] < self.r_sphere + 0.01:
            pen = (self.r_sphere + 0.01) - p_h[2]
            F_heavy[2] += 50000.0 * pen - 800.0 * v_h[2]

        # 5. Integrate Rigid Bodies
        # Boat
        self.vel[self.boat_idx] += (F_boat / self.m_boat) * dt
        self.pos[self.boat_idx] += self.vel[self.boat_idx] * dt
        # Boat rotation
        self.ang_vel[self.boat_idx] += (self.I_inv_boat @ T_boat) * dt
        self.ang_vel[self.boat_idx] *= 0.995  # angular damping
        # Integrate quaternion: dq/dt = 0.5 * omega * q
        w = self.ang_vel[self.boat_idx]
        qw, qx, qy, qz = self.quat[self.boat_idx, 3], self.quat[self.boat_idx, 0], self.quat[self.boat_idx, 1], self.quat[self.boat_idx, 2]
        dq = 0.5 * np.array([
             w[0]*qw + w[1]*qz - w[2]*qy,
             w[1]*qw + w[2]*qx - w[0]*qz,
             w[2]*qw + w[0]*qy - w[1]*qx,
            -w[0]*qx - w[1]*qy - w[2]*qz
        ])
        q_new = self.quat[self.boat_idx] + dq * dt
        self.quat[self.boat_idx] = q_new / np.linalg.norm(q_new)

        # Light sphere
        if t >= self.light_release:
            self.vel[self.light_idx] += (F_light / self.m_light) * dt
            self.pos[self.light_idx] += self.vel[self.light_idx] * dt
            # If light sphere is bobbing at surface, keep it stable
            if self.boat_flooded and self.pos[self.light_idx, 2] > mean_surface_z - self.r_sphere:
                self.vel[self.light_idx, :2] *= 0.98

        # Heavy sphere
        if t >= self.heavy_release:
            self.vel[self.heavy_idx] += (F_heavy / self.m_heavy) * dt
            self.pos[self.heavy_idx] += self.vel[self.heavy_idx] * dt

        # 6. Water Particles Update (coupled wave / displacement motion)
        # Water boundary clipping and restitution
        # Coupling with boat and spheres
        w_pos = self.pos[self.water_indices]
        w_vel = self.vel[self.water_indices]

        # Damping towards baseline + splash dynamics
        w_vel *= 0.992

        # Boat displacement on nearby water particles
        d_boat = w_pos - boat_p
        dist_boat_xy = np.sqrt(d_boat[:, 0]**2 + d_boat[:, 1]**2)
        near_boat = dist_boat_xy < self.hull_L * 0.6
        w_vel[near_boat, 2] += boat_v[2] * 0.15

        # Floor and wall constraints for water
        w_pos += w_vel * dt
        # Pool boundary [-2.3, 2.3], [-1.3, 1.3], [0.02, 2.5]
        min_x, max_x = -2.28, 2.28
        min_y, max_y = -1.28, 1.28
        min_z = 0.02

        hit_min_x = w_pos[:, 0] < min_x
        hit_max_x = w_pos[:, 0] > max_x
        hit_min_y = w_pos[:, 1] < min_y
        hit_max_y = w_pos[:, 1] > max_y
        hit_min_z = w_pos[:, 2] < min_z

        w_pos[hit_min_x, 0] = min_x; w_vel[hit_min_x, 0] *= -0.3
        w_pos[hit_max_x, 0] = max_x; w_vel[hit_max_x, 0] *= -0.3
        w_pos[hit_min_y, 1] = min_y; w_vel[hit_min_y, 1] *= -0.3
        w_pos[hit_max_y, 1] = max_y; w_vel[hit_max_y, 1] *= -0.3
        w_pos[hit_min_z, 2] = min_z; w_vel[hit_min_z, 2] *= -0.2

        self.pos[self.water_indices] = w_pos
        self.vel[self.water_indices] = w_vel

        # 7. Update cumulative impulse
        for g in range(self.G):
            self.cum_ext_impulse[g] += F_ext[g, 0] * dt

        self.sim_time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        while self.sim_time < target_time - 1e-9:
            dt = min(self.sub_dt, target_time - self.sim_time)
            self._integrate_step(dt)

        # Build trajectory fields
        F_ext_current = np.zeros((self.G, 1, 3), dtype=float)
        for g in range(self.G):
            mask = (self.body_id == g)
            F_ext_current[g, 0] = np.sum(self.mass[mask]) * self.gravity

        attributes = {
            "mass_quantity": self.mass.copy(),
            "phase": self.body_id.copy(),
        }

        fields = {
            "boundary_normal": self.boundary_normals.copy(),
            "boundary_target_velocity": self.boundary_vel.copy(),
            "boundary_velocity": self.boundary_vel.copy(),
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((self.G, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((self.G, 3), dtype=float),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": F_ext_current,
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "interaction_force": np.zeros((1, 3), dtype=float),
        }

        scalars = {
            "reference_momentum": 500.0,
            "reference_velocity": 2.5,
        }

        frame = StateFrame(
            time=float(target_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.quat.copy(),
            angular_velocity=self.ang_vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "boat_flooded": self.boat_flooded,
            "final_boat_z": float(self.pos[self.boat_idx, 2]),
            "final_light_sphere_z": float(self.pos[self.light_idx, 2]),
            "final_heavy_sphere_z": float(self.pos[self.heavy_idx, 2]),
        }

