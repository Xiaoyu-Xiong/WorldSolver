"""Deterministic physical solver for open-trefoil knot twist instability."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        # Rod parameters
        self.n = int(self.config["rod"]["vertex_count"])
        self.total_length = float(self.config["rod"]["total_length"])
        self.total_mass = float(self.config["rod"]["mass"])
        self.radius = float(self.config["rod"]["radius"])
        self.gravity = np.array(self.config["gravity"], dtype=float)

        # Initial positions from scene
        init_frame = scene.initial_state()
        self.pos = np.array(init_frame.position, dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.ang_vel = np.zeros_like(self.pos)
        self.mass = np.full(self.n, self.total_mass / self.n, dtype=float)

        # Rest segment lengths
        self.rest_lens = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)
        self.rest_edge_len = float(np.mean(self.rest_lens))

        # Clamp specifications
        self.clamps = self.config["clamps"]
        self.twist_ramp = self.clamps["twist_ramp"]
        self.controlled_indices = [0, self.n - 1]

        # Initial time
        self.current_time = 0.0

        # Physical constants
        r = self.radius
        area = np.pi * r**2
        E = 3.0e6  # Pa
        G = E / 3.0  # Pa
        self.k_stretch = E * area / self.rest_edge_len  # N
        self.k_bend = E * (np.pi / 4.0 * r**4)  # N*m^2
        self.k_twist = G * (np.pi / 2.0 * r**4)  # N*m^2

        self.contact_dist = 2.0 * self.radius
        self.k_contact = 5.0e4

        # Budget tracking
        self.cumulative_external_impulse = np.zeros((1, 3), dtype=float)
        self.cumulative_external_angular_impulse = np.zeros((1, 3), dtype=float)

        # Step size
        self.step_dt = 0.001

    def _get_clamp_twist_angle(self, t: float) -> float:
        ramp = self.twist_ramp
        u = np.clip((t - ramp["ramp_start"]) / (ramp["ramp_end"] - ramp["ramp_start"]), 0.0, 1.0)
        u_smooth = u * u * (3.0 - 2.0 * u)
        angle = 2.0 * np.pi * float(ramp["final_relative_turns"]) * u_smooth
        return float(angle)

    def _compute_forces(self, pos: np.ndarray, vel: np.ndarray, current_twist: float) -> np.ndarray:
        n = self.n
        forces = np.zeros_like(pos)

        # 1. Gravity
        forces += self.mass[:, None] * self.gravity[None, :]

        # 2. Inextensible stretch forces
        diffs = pos[1:] - pos[:-1]
        lens = np.linalg.norm(diffs, axis=1, keepdims=True)
        lens_safe = np.maximum(lens, 1e-8)
        tangents = diffs / lens_safe

        stretch_strain = (lens[:, 0] - self.rest_lens) / self.rest_lens
        stretch_f_mag = self.k_stretch * stretch_strain[:, None] * tangents

        forces[:-1] += stretch_f_mag
        forces[1:] -= stretch_f_mag

        # 3. Discrete bending forces
        t_prev = tangents[:-1]
        t_next = tangents[1:]
        d2x = (t_next - t_prev) / self.rest_edge_len
        f_bend_mid = - (self.k_bend / (self.rest_edge_len**2)) * (t_prev - t_next)
        forces[1:-1] += f_bend_mid
        forces[:-2] -= 0.5 * f_bend_mid
        forces[2:] -= 0.5 * f_bend_mid

        # 4. Twist forces / Writhing instability coupling
        twist_gradient = current_twist / self.total_length
        f_writhe = self.k_twist * twist_gradient * np.cross(tangents[:-1], d2x)
        forces[1:-1] += f_writhe
        forces[:-2] -= 0.5 * f_writhe
        forces[2:] -= 0.5 * f_writhe

        # 5. Finite-thickness self-contact
        pos_expand_i = pos[:, None, :]
        pos_expand_j = pos[None, :, :]
        diff_pairs = pos_expand_i - pos_expand_j
        dist_sq = np.sum(diff_pairs**2, axis=2)

        idx = np.arange(n)
        topology_dist = np.abs(idx[:, None] - idx[None, :])
        mask = (topology_dist > 5) & (dist_sq < (self.contact_dist * 1.5)**2) & (dist_sq > 1e-10)

        if np.any(mask):
            dist = np.sqrt(dist_sq[mask])
            penetration = np.maximum(0.0, self.contact_dist - dist)
            contact_f_mag = self.k_contact * (penetration**1.5) / dist
            dir_pairs = diff_pairs[mask] * contact_f_mag[:, None]

            i_idx, j_idx = np.where(mask)
            np.add.at(forces, i_idx, dir_pairs)

        # 6. Damping
        forces -= 2.5 * self.mass[:, None] * vel
        v_diff = vel[1:] - vel[:-1]
        v_stretch = np.sum(v_diff * tangents, axis=1, keepdims=True) * tangents
        f_internal_damp = 50.0 * v_stretch
        forces[:-1] += f_internal_damp
        forces[1:] -= f_internal_damp

        return forces

    def _step(self, dt: float, current_twist: float) -> None:
        forces = self._compute_forces(self.pos, self.vel, current_twist)
        acc = forces / self.mass[:, None]

        self.vel += acc * dt
        self.pos += self.vel * dt

        # Enforce kinematic clamp boundary conditions
        left_target = np.asarray(self.clamps["left"]["position"], dtype=float)
        right_target = np.asarray(self.clamps["right"]["position"], dtype=float)

        self.pos[0] = left_target
        self.vel[0] = 0.0
        self.pos[-1] = right_target
        self.vel[-1] = 0.0

        tangent_left = np.array([1.0, 0.0, 0.0])
        self.pos[1] = self.pos[0] + self.rest_lens[0] * tangent_left
        self.vel[1] = 0.0

        tangent_right = np.array([-1.0, 0.0, 0.0])
        self.pos[-2] = self.pos[-1] + self.rest_lens[-1] * tangent_right
        self.vel[-2] = 0.0

        # Fast inextensibility projection
        for _ in range(5):
            for i in range(self.n - 1):
                p1 = self.pos[i]
                p2 = self.pos[i + 1]
                delta = p2 - p1
                cur_len = np.linalg.norm(delta)
                if cur_len > 1e-8:
                    diff = (cur_len - self.rest_lens[i]) / cur_len
                    correction = 0.5 * diff * delta
                    if i > 1:
                        self.pos[i] += correction
                    if i + 1 < self.n - 2:
                        self.pos[i + 1] -= correction

    def advance_to(self, time: float) -> StateFrame:
        while self.current_time < time - 1e-7:
            dt = min(self.step_dt, time - self.current_time)
            twist_angle = self._get_clamp_twist_angle(self.current_time)
            self._step(dt, twist_angle)
            self.current_time += dt

        current_twist = self._get_clamp_twist_angle(self.current_time)
        ramp = self.twist_ramp
        if ramp["ramp_start"] <= self.current_time <= ramp["ramp_end"]:
            u = (self.current_time - ramp["ramp_start"]) / (ramp["ramp_end"] - ramp["ramp_start"])
            du_dt = 6.0 * u * (1.0 - u) / (ramp["ramp_end"] - ramp["ramp_start"])
            twist_rate = 2.0 * np.pi * float(ramp["final_relative_turns"]) * du_dt
        else:
            twist_rate = 0.0

        s_frac = np.linspace(0.0, 1.0, self.n)
        self.ang_vel[:, 0] = twist_rate * s_frac

        # Inertia tensors
        inertia_tensors = np.zeros((self.n, 3, 3), dtype=float)
        for i in range(self.n):
            m_i = self.mass[i]
            r = self.radius
            l_i = self.rest_edge_len
            i_xx = 0.5 * m_i * r**2
            i_yy = (1.0 / 12.0) * m_i * (3.0 * r**2 + l_i**2)
            inertia_tensors[i, 0, 0] = i_xx
            inertia_tensors[i, 1, 1] = i_yy
            inertia_tensors[i, 2, 2] = i_yy

        # Budget fields
        domain_weights = np.ones((1, self.n), dtype=float)
        ext_force = np.zeros((1, 1, 3), dtype=float)
        ext_force[0, 0] = np.sum(self.mass[:, None] * self.gravity[None, :], axis=0)
        force_pos = np.zeros((1, 1, 3), dtype=float)
        force_pos[0, 0] = np.mean(self.pos, axis=0)
        ext_couple = np.zeros((1, 1, 3), dtype=float)

        self.cumulative_external_impulse += ext_force[:, 0, :] * float(self.config["sample_dt"])

        constraint_val = np.array([self.pos[0], self.pos[-1]], dtype=float)
        constraint_tgt = np.array([
            self.clamps["left"]["position"],
            self.clamps["right"]["position"]
        ], dtype=float)

        frame = StateFrame(
            time=float(self.current_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=np.tile([0.0, 0.0, 0.0, 1.0], (self.n, 1)),
            angular_velocity=self.ang_vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "inertia_tensor": inertia_tensors,
                "mass_quantity": self.mass.copy(),
                "arc_fraction": np.linspace(0.0, 1.0, self.n),
                "rod_radius": np.full(self.n, self.radius),
                "contact_diameter": np.full(self.n, 2.0 * self.radius),
                "endpoint_controlled": np.isin(np.arange(self.n), [0, self.n - 1]).astype(float),
                "material_twist_turns": np.full(self.n, current_twist / (2.0 * np.pi)),
            },
            fields={
                "budget_cumulative_external_angular_impulse": self.cumulative_external_angular_impulse.copy(),
                "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
                "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_domain_weights": domain_weights,
                "budget_external_couple": ext_couple,
                "budget_external_force": ext_force,
                "budget_force_position": force_pos,
                "constraint_target": constraint_tgt,
                "constraint_value": constraint_val,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "constraint_scale": 1.0,
                "reference_angular_momentum": 1.0,
                "reference_momentum": 1.0,
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "vertex_count": self.n,
            "total_mass": self.total_mass,
        }

