"""Physics solver for curly hair clump vertical oscillation."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        # Initial frame
        init_frame = scene.initial_state()
        self.time = float(init_frame.time)
        self.pos = np.array(init_frame.position, dtype=np.float64)
        self.vel = np.array(init_frame.velocity, dtype=np.float64)
        self.mass = np.array(init_frame.mass, dtype=np.float64)
        self.N = len(self.pos)

        # Physical parameters
        guide = self.config["physical_guide"]
        self.material_length = float(guide["material_length"])
        self.radius = float(guide["radius"])
        self.density = float(guide["density"])
        self.gamma_internal = float(guide["internal_damping_gamma"])
        self.ambient_nu = float(guide["ambient_drag_nu"])
        self.gravity = np.array(self.config["gravity"], dtype=np.float64)

        # Root motion parameters
        root_cfg = self.config["root_motion"]
        self.rest_pos = np.array(root_cfg["rest_position"], dtype=np.float64)
        self.start_time = float(root_cfg["start_time"])
        self.amplitude = float(root_cfg["amplitude"])
        self.frequency = float(root_cfg["frequency"])
        self.phase = float(root_cfg["phase"])
        self.axis = np.array(root_cfg["axis"], dtype=np.float64)
        self.axis = self.axis / (np.linalg.norm(self.axis) + 1e-12)

        # Initial rest geometry (natural helix)
        self.rest_edge_lengths = np.linalg.norm(self.pos[1:] - self.pos[:-1], axis=1) # (N-1,)
        self.rest_pos_local = self.pos.copy() - self.pos[0]  # rest shape relative to root

        # Rod physical stiffness
        # Hair keratin Young's modulus E ~ 1-3 GPa
        E = 2.0e9 # Pa
        r = self.radius
        A = np.pi * r**2
        I = (np.pi / 4.0) * (r**4)
        self.EI = E * I
        self.EA = E * A

        # Edge rest vectors in initial shape
        self.rest_e = self.pos[1:] - self.pos[:-1] # (N-1, 3)
        self.rest_L = np.linalg.norm(self.rest_e, axis=1) # (N-1,)

        # Dual edge lengths for bending: l_dual[i] = 0.5 * (rest_L[i-1] + rest_L[i])
        self.rest_dual_L = 0.5 * (self.rest_L[:-1] + self.rest_L[1:]) # (N-2,)

        # Total physical mass
        self.total_mass = np.sum(self.mass)
        # Nonzero mass for dynamic nodes
        self.inv_mass = np.zeros(self.N, dtype=np.float64)
        self.inv_mass[1:] = 1.0 / self.mass[1:]

        # Budget tracking (cumulative impulse, cumulative mass source)
        self.cum_ext_impulse = np.zeros((1, 3), dtype=np.float64)

        # Reference momentum scale
        v_ref = max(self.amplitude * 2.0 * np.pi * self.frequency, 0.5)
        self.reference_momentum = float(self.total_mass * v_ref)
        if self.reference_momentum < 1e-6:
            self.reference_momentum = 1e-4

    def _get_root_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute root position, velocity, and acceleration at time t."""
        if t <= self.start_time:
            p = self.rest_pos.copy()
            v = np.zeros(3, dtype=np.float64)
            a = np.zeros(3, dtype=np.float64)
        else:
            omega = 2.0 * np.pi * self.frequency
            dt_driven = t - self.start_time
            disp = self.amplitude * np.sin(omega * dt_driven + self.phase)
            vel = self.amplitude * omega * np.cos(omega * dt_driven + self.phase)
            acc = -self.amplitude * (omega**2) * np.sin(omega * dt_driven + self.phase)
            p = self.rest_pos + disp * self.axis
            v = vel * self.axis
            a = acc * self.axis
        return p, v, a

    def _compute_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute all internal and external forces on nodes."""
        forces = np.zeros((self.N, 3), dtype=np.float64)

        # 1. Gravity (external)
        for i in range(1, self.N):
            forces[i] += self.mass[i] * self.gravity

        # 2. Ambient drag (external)
        dl = self.material_length / (self.N - 1)
        forces[1:] -= self.ambient_nu * dl * vel[1:]

        # 3. Inextensibility / axial stretching forces (internal)
        e = pos[1:] - pos[:-1] # (N-1, 3)
        l_cur = np.linalg.norm(e, axis=1) # (N-1,)
        e_hat = e / (l_cur[:, None] + 1e-14) # (N-1, 3)

        strain = (l_cur - self.rest_L) / self.rest_L # (N-1,)
        stretch_tension = self.EA * strain # (N-1,)

        # Internal axial damping
        rel_v = vel[1:] - vel[:-1]
        strain_rate = np.sum(rel_v * e_hat, axis=1) / self.rest_L
        stretch_damping = self.gamma_internal * 1e8 * strain_rate

        total_tension = stretch_tension + stretch_damping
        tension_force = total_tension[:, None] * e_hat # (N-1, 3)

        # Apply tension to nodes
        forces[:-1] += tension_force
        forces[1:] -= tension_force

        # 4. Bending forces with natural curvature (internal)
        e_prev_hat = e_hat[:-1] # (N-2, 3)
        e_next_hat = e_hat[1:]  # (N-2, 3)

        rest_e_prev_hat = self.rest_e[:-1] / self.rest_L[:-1, None]
        rest_e_next_hat = self.rest_e[1:] / self.rest_L[1:, None]

        curv_cur = (e_next_hat - e_prev_hat) # (N-2, 3)
        curv_rest = (rest_e_next_hat - rest_e_prev_hat) # (N-2, 3)
        delta_curv = curv_cur - curv_rest # (N-2, 3)

        kb = self.EI / self.rest_dual_L # (N-2,)

        # Internal bending damping
        curv_rate = ( (vel[2:] - vel[1:-1]) / (l_cur[1:, None] + 1e-14)
                     - (vel[1:-1] - vel[:-2]) / (l_cur[:-1, None] + 1e-14) )
        gamma_b = self.gamma_internal * 1e4

        bending_moment = kb[:, None] * delta_curv + gamma_b * curv_rate # (N-2, 3)

        f_bend_prev = bending_moment / (l_cur[:-1, None] + 1e-14) # on i-1
        f_bend_next = bending_moment / (l_cur[1:, None] + 1e-14)  # on i+1
        f_bend_mid = -(f_bend_prev + f_bend_next)                  # on i

        for k in range(len(bending_moment)):
            forces[k] += f_bend_prev[k]
            forces[k + 1] += f_bend_mid[k]
            forces[k + 2] += f_bend_next[k]

        return forces

    def _substep(self, dt: float) -> None:
        """Perform one numerical integration step with constraint projection."""
        root_pos, root_vel, _ = self._get_root_state(self.time + dt)

        forces = self._compute_forces(self.pos, self.vel)

        dl = self.material_length / (self.N - 1)
        ext_forces_on_dynamic = (self.mass[1:, None] * self.gravity
                                - self.ambient_nu * dl * self.vel[1:])

        # Velocity update for dynamic nodes
        self.vel[1:] += (forces[1:] * self.inv_mass[1:, None]) * dt

        # Root node velocity is kinematic
        self.vel[0] = root_vel
        self.pos[0] = root_pos

        # Predict positions
        self.pos[1:] += self.vel[1:] * dt

        # Inextensibility distance constraint projection
        iterations = 10
        for _ in range(iterations):
            p0 = self.pos[:-1]
            p1 = self.pos[1:]
            delta = p1 - p0
            dists = np.linalg.norm(delta, axis=1)
            err = dists - self.rest_L
            dir_vec = delta / (dists[:, None] + 1e-14)

            w0 = self.inv_mass[:-1]
            w1 = self.inv_mass[1:]
            w_sum = w0 + w1

            corr = (err / w_sum)[:, None] * dir_vec

            self.pos[0] = root_pos
            self.pos[1:] -= (w1[:, None] / w_sum[:, None]) * corr
            self.pos[:-1] += (w0[:, None] / w_sum[:, None]) * corr
            self.pos[0] = root_pos

        self.time += dt

        total_ext_force = np.sum(ext_forces_on_dynamic, axis=0)
        self.cum_ext_impulse[0] += total_ext_force * dt

    def advance_to(self, target_time: float) -> StateFrame:
        max_substep_dt = 2.0e-4
        while self.time < target_time - 1e-12:
            dt = min(max_substep_dt, target_time - self.time)
            pos_old = self.pos.copy()
            self._substep(dt)
            self.vel[1:] = (self.pos[1:] - pos_old[1:]) / dt
            self.vel[0] = (self.pos[0] - pos_old[0]) / dt

        self.time = target_time
        root_pos, root_vel, _ = self._get_root_state(self.time)
        self.pos[0] = root_pos
        self.vel[0] = root_vel

        node_index = np.arange(self.N, dtype=int)

        mass_quantity = self.mass.copy()
        budget_cumulative_external_impulse = self.cum_ext_impulse.copy()
        budget_cumulative_inflow_momentum = np.zeros((1, 3), dtype=np.float64)
        budget_cumulative_outflow_momentum = np.zeros((1, 3), dtype=np.float64)
        budget_domain_weights = np.ones((1, self.N), dtype=np.float64)

        dl = self.material_length / (self.N - 1)
        ext_f = np.sum(self.mass[1:, None] * self.gravity - self.ambient_nu * dl * self.vel[1:], axis=0)
        budget_external_force = ext_f.reshape(1, 1, 3)

        edge_lengths = np.linalg.norm(self.pos[1:] - self.pos[:-1], axis=1)
        constraint_value = edge_lengths
        constraint_target = self.rest_L
        cumulative_mass_source = np.zeros(1, dtype=np.float64)
        constraint_scale = float(self.material_length / (self.N - 1))
        reference_momentum = float(self.reference_momentum)

        frame = StateFrame(
            time=self.time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "node_index": node_index,
                "is_root": (node_index == 0).astype(int),
                "is_free_tip": (node_index == self.N - 1).astype(int),
                "kinematic_root": (node_index == 0).astype(int),
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_cumulative_external_impulse": budget_cumulative_external_impulse,
                "budget_cumulative_inflow_momentum": budget_cumulative_inflow_momentum,
                "budget_cumulative_outflow_momentum": budget_cumulative_outflow_momentum,
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "constraint_target": constraint_target,
                "constraint_value": constraint_value,
                "cumulative_mass_source": cumulative_mass_source,
            },
            scalars={
                "material_length": float(self.material_length),
                "internal_damping_gamma": float(self.gamma_internal),
                "ambient_drag_nu": float(self.ambient_nu),
                "constraint_scale": constraint_scale,
                "reference_momentum": reference_momentum,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "node_count": self.N,
            "time": self.time,
            "total_mass": self.total_mass,
        }

