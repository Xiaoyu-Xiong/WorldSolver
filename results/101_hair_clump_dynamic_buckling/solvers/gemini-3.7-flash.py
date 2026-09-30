"""Physics simulation solver for dynamic buckling of a vertically shaken hair clump."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.config = None
        self.current_time = 0.0
        self.N = 49
        self.pos = None
        self.vel = None
        self.mass = None
        self.rest_edge_len = None
        self.g = np.array([0.0, 0.0, -9.81])
        self.cum_impulse = np.zeros((1, 3), dtype=float)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        f0 = scene.initial_state()
        self.pos = np.array(f0.position, dtype=float, copy=True)
        self.vel = np.array(f0.velocity, dtype=float, copy=True)
        self.mass = np.array(f0.mass, dtype=float, copy=True)
        self.N = len(self.pos)
        self.current_time = 0.0

        hair = self.config["hair_clump"]
        self.total_length = float(hair["material_length"])
        self.radius = float(hair["effective_radius"])
        self.density = float(hair["density"])
        self.gamma = float(hair["internal_damping_gamma"])
        self.nu = float(hair["ambient_drag_nu"])
        self.g = np.asarray(self.config["gravity"], dtype=float)

        # Rest segment length
        self.rest_edge_len = self.total_length / (self.N - 1)
        self.cum_impulse = np.zeros((1, 3), dtype=float)

    def _root_state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute root position, velocity, acceleration at time t."""
        rm = self.config["root_motion"]
        rest_pos = np.asarray(rm["rest_position"], dtype=float)
        axis = np.asarray(rm["axis"], dtype=float)
        start_t = float(rm["start_time"])
        amp = float(rm["amplitude"])
        freq = float(rm["frequency"])
        phase = float(rm["phase"])
        omega = 2.0 * np.pi * freq

        if t < start_t:
            return rest_pos.copy(), np.zeros(3), np.zeros(3)
        dt_start = t - start_t
        pos = rest_pos + axis * (amp * np.sin(omega * dt_start + phase))
        vel = axis * (amp * omega * np.cos(omega * dt_start + phase))
        acc = axis * (-amp * (omega**2) * np.sin(omega * dt_start + phase))
        return pos, vel, acc

    def _internal_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute bending and damping forces along the rod."""
        # Young's modulus for hair clump: E ~ 1.0 GPa - 3.0 GPa
        # Second moment of area I = pi * r^4 / 4
        # Bending stiffness B = E * I
        E = 2.0e9  # Pa
        I = np.pi * (self.radius**4) / 4.0
        B = E * I

        forces = np.zeros_like(pos)
        dl = self.rest_edge_len
        k_bend = B / (dl**3)

        # Curvatures kappa_i = pos[i+1] - 2*pos[i] + pos[i-1]
        # At root boundary (node 0), tangent is vertical downwards (-Z)
        tangent_root = np.array([0.0, 0.0, -1.0])
        x_minus_1 = pos[0] - tangent_root * dl

        kappas = np.zeros((self.N, 3))
        kappas[0] = pos[1] - 2 * pos[0] + x_minus_1
        for i in range(1, self.N - 1):
            kappas[i] = pos[i + 1] - 2 * pos[i] + pos[i - 1]
        kappas[-1] = np.zeros(3)  # zero curvature at free tip

        # Gradient of bending energy 0.5 * sum_i |kappa_i|^2 w.r.t pos[j]:
        for j in range(1, self.N - 1):
            kap_prev = kappas[j - 1]
            kap_curr = kappas[j]
            kap_next = kappas[j + 1] if j + 1 < self.N else np.zeros(3)
            grad = kap_prev - 2.0 * kap_curr + kap_next
            forces[j] -= k_bend * grad

        kap_prev = kappas[self.N - 2]
        grad_tip = kap_prev - 2.0 * kappas[self.N - 1]
        forces[-1] -= k_bend * grad_tip

        # Internal damping proportional to curvature rate
        d_kappas = np.zeros((self.N, 3))
        d_x_minus_1 = vel[0]
        d_kappas[0] = vel[1] - 2 * vel[0] + d_x_minus_1
        for i in range(1, self.N - 1):
            d_kappas[i] = vel[i + 1] - 2 * vel[i] + vel[i - 1]

        c_int = 1e-4
        for j in range(1, self.N - 1):
            d_prev = d_kappas[j - 1]
            d_curr = d_kappas[j]
            d_next = d_kappas[j + 1] if j + 1 < self.N else np.zeros(3)
            forces[j] -= c_int * (d_prev - 2.0 * d_curr + d_next)

        return forces

    def _step(self, sub_dt: float) -> None:
        """Advance simulation by sub_dt using Position-Based Dynamics / Verlet."""
        t_next = self.current_time + sub_dt

        # 1. External forces (gravity and ambient drag)
        forces = np.zeros_like(self.pos)
        for i in range(1, self.N):
            forces[i] += self.mass[i] * self.g

        # Internal bending forces
        f_bend = self._internal_forces(self.pos, self.vel)
        forces += f_bend

        # Ambient drag
        for i in range(1, self.N):
            v = self.vel[i]
            v_norm = np.linalg.norm(v)
            f_drag = -self.nu * self.rest_edge_len * v - 0.5 * 1.225 * (2.0 * self.radius) * self.rest_edge_len * 1.0 * v_norm * v
            forces[i] += f_drag

        # 2. Integrate velocities and predict positions
        new_vel = np.array(self.vel, copy=True)
        new_pos = np.array(self.pos, copy=True)

        for i in range(1, self.N):
            acc = forces[i] / self.mass[i]
            new_vel[i] += acc * sub_dt
            new_pos[i] += new_vel[i] * sub_dt

        # Kinematic root update
        root_p, root_v, _ = self._root_state(t_next)
        new_pos[0] = root_p
        new_vel[0] = root_v

        # 3. Inextensibility (Distance constraint projection / PBD)
        pbd_iters = 50
        inv_mass = np.zeros(self.N)
        inv_mass[1:] = 1.0 / self.mass[1:]
        inv_mass[0] = 0.0  # infinite mass for root

        target_d = self.rest_edge_len
        for _ in range(pbd_iters):
            for i in range(self.N - 1):
                p1 = new_pos[i]
                p2 = new_pos[i + 1]
                delta = p2 - p1
                dist = np.linalg.norm(delta)
                if dist > 1e-12:
                    diff = (dist - target_d) / dist
                    w1 = inv_mass[i]
                    w2 = inv_mass[i + 1]
                    w_sum = w1 + w2
                    if w_sum > 0:
                        corr = (diff / w_sum) * delta
                        new_pos[i] += w1 * corr
                        new_pos[i + 1] -= w2 * corr
            new_pos[0] = root_p

        # 4. Update velocities from corrected positions
        new_vel[1:] = (new_pos[1:] - self.pos[1:]) / sub_dt
        new_vel[0] = root_v

        self.pos = new_pos
        self.vel = new_vel
        self.current_time = t_next

    def advance_to(self, target_time: float) -> StateFrame:
        sub_dt = 1e-4  # 0.1 ms sub-step for physical fidelity and stability
        while self.current_time < target_time - 1e-9:
            dt = min(sub_dt, target_time - self.current_time)
            self._step(dt)

        self.current_time = target_time

        # Prepare StateFrame
        root_p, root_v, _ = self._root_state(target_time)
        self.pos[0] = root_p
        self.vel[0] = root_v

        # Calculate constraint values
        edge_diffs = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)
        rest_edge_diffs = np.full(self.N - 1, self.rest_edge_len)

        total_hair_mass = np.sum(self.mass[1:])
        grav_force = total_hair_mass * self.g

        domain_weights = np.ones((1, self.N), dtype=float)
        ext_force = np.zeros((1, 1, 3), dtype=float)
        ext_force[0, 0] = grav_force

        cum_ext_impulse = np.zeros((1, 3), dtype=float)
        cum_inflow_p = np.zeros((1, 3), dtype=float)
        cum_outflow_p = np.zeros((1, 3), dtype=float)
        cum_mass_source = np.zeros(1, dtype=float)

        node_index = np.arange(self.N, dtype=int)
        is_root = (node_index == 0).astype(int)
        is_free_tip = (node_index == self.N - 1).astype(int)

        frame = StateFrame(
            time=float(target_time),
            position=np.array(self.pos, dtype=float, copy=True),
            velocity=np.array(self.vel, dtype=float, copy=True),
            mass=np.array(self.mass, dtype=float, copy=True),
            attributes={
                "mass_quantity": np.array(self.mass, dtype=float, copy=True),
                "node_index": node_index,
                "is_root": is_root,
                "is_free_tip": is_free_tip,
                "kinematic_root": is_root,
            },
            fields={
                "budget_cumulative_external_impulse": cum_ext_impulse,
                "budget_cumulative_inflow_momentum": cum_inflow_p,
                "budget_cumulative_outflow_momentum": cum_outflow_p,
                "budget_domain_weights": domain_weights,
                "budget_external_force": ext_force,
                "constraint_target": rest_edge_diffs,
                "constraint_value": edge_diffs,
                "cumulative_mass_source": cum_mass_source,
            },
            scalars={
                "constraint_scale": float(self.rest_edge_len),
                "reference_momentum": float(total_hair_mass * 1.0 + 1e-4),
                "material_length": float(self.total_length),
                "internal_damping_gamma": float(self.gamma),
                "ambient_drag_nu": float(self.nu),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "node_count": self.N,
        }

