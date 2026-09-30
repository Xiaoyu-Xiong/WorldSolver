"""Physics-based solver for 077_elastic_rod_ground_stick_slip.

Recreates the stick-slip dynamics of an elastic rod pressed and dragged along a frictional plane.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = {}
        self.seed = 0
        self.current_time = 0.0

        # Physical state
        self.rest_positions = None
        self.positions = None
        self.velocities = None
        self.masses = None
        self.driven_mask = None
        self.num_vertices = 0

        # Discretization info
        self.rings = 10
        self.sides = 6
        self.radius = 0.055
        self.height = 1.45

        # Boundary condition params
        self.press_interval = [0.0, 0.8]
        self.press_displacement = np.array([0.0, 0.0, -0.05])
        self.drag_interval = [0.8, 8.0]
        self.drag_velocity = np.array([0.22, 0.0, 0.0])
        self.gravity = np.array([0.0, 0.0, -9.81])
        self.mu_s = 0.35
        self.mu_d = 0.35

        # Structural connectivity
        self.springs = []  # list of (i, j, rest_len, stiffness, damping)
        self.ring_indices = []

        # Reference scales and tracking
        self.ref_velocity = 0.22
        self.ref_force = 1.0
        self.ref_momentum = 1.0
        self.cum_external_impulse = np.zeros((1, 3))
        self.last_contact_forces = np.zeros(3)

        # Simulation substepping
        self.sub_dt = 1e-4

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        self.current_time = 0.0

        init_state = scene.initial_state()
        self.rest_positions = np.array(init_state.position, dtype=np.float64, copy=True)
        self.positions = np.array(init_state.position, dtype=np.float64, copy=True)
        self.velocities = np.array(init_state.velocity, dtype=np.float64, copy=True)
        self.masses = np.array(init_state.mass, dtype=np.float64, copy=True)
        self.num_vertices = len(self.positions)

        total_mass = np.sum(self.masses)
        self.ref_velocity = float(np.linalg.norm(self.drag_velocity))
        self.ref_force = float(total_mass * np.linalg.norm(self.gravity) + 0.5)
        self.ref_momentum = float(total_mass * self.ref_velocity + 1e-3)

        if "driven_top" in init_state.attributes:
            self.driven_mask = np.array(init_state.attributes["driven_top"] > 0.5, dtype=bool)
        else:
            self.driven_mask = (self.positions[:, 2] > (self.height - 1e-3))

        bc = self.config.get("boundary_conditions", {}).get("driven_top", {})
        self.press_interval = list(bc.get("press_interval", [0.0, 0.8]))
        self.press_displacement = np.array(bc.get("press_displacement", [0.0, 0.0, -0.05]), dtype=np.float64)
        self.drag_interval = list(bc.get("drag_interval", [0.8, 8.0]))
        self.drag_velocity = np.array(bc.get("drag_velocity", [0.22, 0.0, 0.0]), dtype=np.float64)

        grav = self.config.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.array(grav, dtype=np.float64)

        contact_cfg = self.config.get("contact", {})
        self.mu_s = float(contact_cfg.get("ground_friction_coefficient", 0.35))
        self.mu_d = self.mu_s

        # Build ring structure
        self.ring_indices = []
        for k in range(self.rings):
            r_idx = list(range(k * self.sides, (k + 1) * self.sides))
            self.ring_indices.append(r_idx)

        # Build spring mesh for 3D volumetric beam elasticity
        self._build_elastic_network()
        self.cum_external_impulse = np.zeros((1, 3), dtype=np.float64)

    def _build_elastic_network(self) -> None:
        self.springs = []
        spring_dict = {}

        def add_spring(i, j, k_stiffness, damping_ratio=0.08):
            if i > j:
                i, j = j, i
            if i == j:
                return
            key = (i, j)
            if key not in spring_dict:
                rest_len = float(np.linalg.norm(self.rest_positions[i] - self.rest_positions[j]))
                c_damp = 2.0 * damping_ratio * np.sqrt(k_stiffness * min(self.masses[i], self.masses[j]))
                spring_dict[key] = (rest_len, k_stiffness, c_damp)

        # Base stiffness scales for rod material (soft elastic polymer / silicone rod)
        k_long = 1200.0   # Longitudinal stretch stiffness
        k_diag = 800.0    # Shear diagonal stiffness
        k_ring = 1500.0   # Ring circumference stiffness
        k_bend = 400.0    # 2-ring bend stiffness
        k_cross = 600.0   # Ring internal diagonal struts

        # Intra-ring springs (sides & cross diagonals)
        for k in range(self.rings):
            ring = self.ring_indices[k]
            for s in range(self.sides):
                s_next = (s + 1) % self.sides
                add_spring(ring[s], ring[s_next], k_ring)
                s_opp = (s + 3) % self.sides
                add_spring(ring[s], ring[s_opp], k_cross)

        # Inter-ring springs (longitudinal, face diagonals, cross diagonals)
        for k in range(self.rings - 1):
            r1 = self.ring_indices[k]
            r2 = self.ring_indices[k + 1]
            for s in range(self.sides):
                s_next = (s + 1) % self.sides
                # Longitudinal
                add_spring(r1[s], r2[s], k_long)
                # Face diagonals
                add_spring(r1[s], r2[s_next], k_diag)
                add_spring(r1[s_next], r2[s], k_diag)
                # Internal cross diagonals
                s_opp = (s + 3) % self.sides
                add_spring(r1[s], r2[s_opp], k_diag)

        # Bending springs across 2 rings (k to k+2)
        for k in range(self.rings - 2):
            r1 = self.ring_indices[k]
            r3 = self.ring_indices[k + 2]
            for s in range(self.sides):
                add_spring(r1[s], r3[s], k_bend)

        # Convert to arrays for vectorization
        spring_list = []
        for (i, j), (rest_len, k_stiff, c_damp) in spring_dict.items():
            spring_list.append((i, j, rest_len, k_stiff, c_damp))

        self.spring_i = np.array([s[0] for s in spring_list], dtype=np.int32)
        self.spring_j = np.array([s[1] for s in spring_list], dtype=np.int32)
        self.spring_l0 = np.array([s[2] for s in spring_list], dtype=np.float64)
        self.spring_k = np.array([s[3] for s in spring_list], dtype=np.float64)
        self.spring_c = np.array([s[4] for s in spring_list], dtype=np.float64)

    def _get_top_target(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute target position offset and velocity for driven top vertices."""
        disp = np.zeros(3, dtype=np.float64)
        vel = np.zeros(3, dtype=np.float64)

        t_p0, t_p1 = self.press_interval
        t_d0, t_d1 = self.drag_interval

        # Press phase
        if t <= t_p0:
            disp += 0.0
            vel += 0.0
        elif t < t_p1:
            frac = (t - t_p0) / (t_p1 - t_p0)
            disp += frac * self.press_displacement
            vel += self.press_displacement / (t_p1 - t_p0)
        else:
            disp += self.press_displacement

        # Drag phase
        if t >= t_d0:
            dt_drag = min(t, t_d1) - t_d0
            disp += dt_drag * self.drag_velocity
            if t <= t_d1:
                vel += self.drag_velocity

        return disp, vel

    def _step(self, dt: float) -> None:
        """Advance physical state by sub-step dt using semi-implicit integration with contact."""
        # 1. Enforce driven top kinematics
        disp_top, vel_top = self._get_top_target(self.current_time + dt)
        self.positions[self.driven_mask] = self.rest_positions[self.driven_mask] + disp_top
        self.velocities[self.driven_mask] = vel_top

        # 2. Compute internal elastic forces
        p_i = self.positions[self.spring_i]
        p_j = self.positions[self.spring_j]
        v_i = self.velocities[self.spring_i]
        v_j = self.velocities[self.spring_j]

        diff = p_j - p_i
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist_safe = np.maximum(dist, 1e-8)
        dir_vec = diff / dist_safe

        # Hookean stretch + relative velocity damping
        rel_vel = v_j - v_i
        v_proj = np.sum(rel_vel * dir_vec, axis=1, keepdims=True)

        stretch = dist - self.spring_l0[:, None]
        f_mag = self.spring_k[:, None] * stretch + self.spring_c[:, None] * v_proj
        f_vec = f_mag * dir_vec

        forces = np.zeros_like(self.positions)
        np.add.at(forces, self.spring_i, f_vec)
        np.add.at(forces, self.spring_j, -f_vec)

        # 3. Add gravity and ambient damping
        forces += self.masses[:, None] * self.gravity[None, :]
        forces -= 0.02 * self.velocities  # gentle medium damping

        # 4. Integrate unconstrained vertices
        free_mask = ~self.driven_mask
        acc = forces[free_mask] / self.masses[free_mask, None]
        self.velocities[free_mask] += acc * dt
        self.positions[free_mask] += self.velocities[free_mask] * dt

        # 5. Frictional ground contact resolution (z=0 is ground surface)
        # Rod bottom vertices have radius offset so contact plane is at z = 0.0
        ground_z = 0.0
        contact_nodes = np.where(self.positions[:, 2] <= ground_z + 1e-4)[0]

        total_contact_force = np.zeros(3, dtype=np.float64)

        for idx in contact_nodes:
            if self.driven_mask[idx]:
                continue
            pen = ground_z - self.positions[idx, 2]
            if pen >= 0:
                # Position projection
                self.positions[idx, 2] = ground_z

                # Normal contact impulse / force
                # Spring-damper penalty contact
                k_pen = 2e4
                d_pen = 200.0
                f_norm_mag = max(0.0, k_pen * pen - d_pen * self.velocities[idx, 2])
                # Ensure velocity along normal is stopped/reflected
                if self.velocities[idx, 2] < 0:
                    self.velocities[idx, 2] = 0.0

                f_norm = np.array([0.0, 0.0, f_norm_mag])

                # Tangential friction (Coulomb stick-slip)
                v_tang = np.array([self.velocities[idx, 0], self.velocities[idx, 1], 0.0])
                v_speed = np.linalg.norm(v_tang)

                max_static_f = self.mu_s * f_norm_mag
                dynamic_f = self.mu_d * f_norm_mag

                # Tangential stopping force needed to stick
                f_stop_mag = (self.masses[idx] * v_speed) / dt

                if f_stop_mag <= max_static_f:
                    # Stick
                    self.velocities[idx, 0] = 0.0
                    self.velocities[idx, 1] = 0.0
                    f_tang = -v_tang / dt * self.masses[idx]
                else:
                    # Slip: apply kinetic friction opposing slip velocity
                    slip_dir = v_tang / (v_speed + 1e-8)
                    f_tang = -dynamic_f * slip_dir
                    self.velocities[idx, :2] += (f_tang[:2] / self.masses[idx]) * dt

                total_contact_force += f_norm + f_tang

        self.last_contact_forces = total_contact_force
        ext_f = total_contact_force + np.sum(self.masses[:, None] * self.gravity[None, :], axis=0)
        self.cum_external_impulse[0] += ext_f * dt

        self.current_time += dt

    def advance_to(self, target_time: float) -> StateFrame:
        # Step forward to target_time
        while self.current_time < target_time - 1e-8:
            dt = min(self.sub_dt, target_time - self.current_time)
            self._step(dt)

        self.current_time = target_time

        # Recompute driven top exactly at sample time
        disp_top, vel_top = self._get_top_target(self.current_time)
        self.positions[self.driven_mask] = self.rest_positions[self.driven_mask] + disp_top
        self.velocities[self.driven_mask] = vel_top

        # Compute required StateFrame fields
        N = self.num_vertices

        # Deformation gradient per element (9 rod segments between successive rings)
        Ne = self.rings - 1
        element_F = np.zeros((Ne, 3, 3), dtype=np.float64)
        for k in range(Ne):
            r1 = self.ring_indices[k]
            r2 = self.ring_indices[k + 1]
            p_curr = (self.positions[r2] - self.positions[r1])  # (6, 3)
            p_rest = (self.rest_positions[r2] - self.rest_positions[r1])  # (6, 3)
            # Least-squares affine deformation: F = (p_curr^T p_rest) (p_rest^T p_rest)^-1
            AtA = p_rest.T @ p_rest + 1e-6 * np.eye(3)
            AtB = p_rest.T @ p_curr
            element_F[k] = AtB.T @ np.linalg.inv(AtA)

        # Contact and friction fields for bottom ring (persistent contact region C=6)
        bottom_ring = self.ring_indices[0]
        C = len(bottom_ring)

        f_normal = np.tile(np.array([0.0, 0.0, 1.0]), (C, 1))
        f_norm_force = np.zeros(C, dtype=np.float64)
        f_tang_force = np.zeros((C, 3), dtype=np.float64)
        f_rel_vel = np.zeros((C, 3), dtype=np.float64)
        f_active = np.zeros(C, dtype=np.float64)

        for i, v_idx in enumerate(bottom_ring):
            pz = self.positions[v_idx, 2]
            f_rel_vel[i] = self.velocities[v_idx]
            if pz <= 0.01:
                f_active[i] = 1.0
                f_norm_force[i] = max(0.0, -self.last_contact_forces[2] / C)
                f_tang_force[i] = self.last_contact_forces / C
                f_tang_force[i, 2] = 0.0

        f_dyn_coef = np.full(C, self.mu_d, dtype=np.float64)
        f_stat_coef = np.full(C, self.mu_s, dtype=np.float64)

        # External force per budget domain [G=1, K=1, 3]
        budget_ext_f = np.zeros((1, 1, 3), dtype=np.float64)
        total_ext = self.last_contact_forces + np.sum(self.masses[:, None] * self.gravity[None, :], axis=0)
        budget_ext_f[0, 0] = total_ext

        fields = {
            "budget_cumulative_external_impulse": np.array(self.cum_external_impulse, dtype=np.float64),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_domain_weights": np.ones((1, N), dtype=np.float64),
            "budget_external_force": budget_ext_f,
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "element_deformation_gradient": element_F,
            "friction_active_mask": f_active,
            "friction_dynamic_coefficient": f_dyn_coef,
            "friction_normal": f_normal,
            "friction_normal_force": f_norm_force,
            "friction_relative_velocity": f_rel_vel,
            "friction_static_coefficient": f_stat_coef,
            "friction_tangential_force": f_tang_force,
        }

        attributes = {
            "mass_quantity": np.array(self.masses, dtype=np.float64),
            "driven_top": np.array(self.driven_mask, dtype=np.float64),
            "rod_vertex": np.ones(N, dtype=np.float64),
        }

        scalars = {
            "friction_stick_speed_tolerance": 0.005,
            "reference_force": float(self.ref_force),
            "reference_momentum": float(self.ref_momentum),
            "reference_velocity": float(self.ref_velocity),
            "rod_radius": float(self.radius),
            "rod_height": float(self.height),
        }

        frame = StateFrame(
            time=float(self.current_time),
            position=np.array(self.positions, dtype=np.float64, copy=True),
            velocity=np.array(self.velocities, dtype=np.float64, copy=True),
            mass=np.array(self.masses, dtype=np.float64, copy=True),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_vertices": self.num_vertices,
            "num_springs": len(self.spring_l0) if hasattr(self, "spring_l0") else 0,
        }

