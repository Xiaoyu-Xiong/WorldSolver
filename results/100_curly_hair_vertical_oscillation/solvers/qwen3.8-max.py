"""Solver for vertical oscillation of a curly hair clump.

PBD-based inextensible rod with bending relaxation towards natural helix,
gravity, internal damping, ambient drag. Uses simple, stable integration:
- External forces (gravity, drag) predict new positions
- Inextensibility projected via PBD
- Bending applied as gentle positional relaxation towards reference shape
- Velocity derived from constrained position change
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


NODE_COUNT = 65


class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        cfg = scene.config

        guide = cfg["physical_guide"]
        drive = cfg["root_motion"]
        grav = np.asarray(cfg["gravity"], dtype=float)

        self.material_length = float(guide["material_length"])
        self.cross_radius = float(guide["radius"])
        self.density = float(guide["density"])
        self.helix_radius = float(guide["natural_helix_radius"])
        self.helix_pitch = float(guide["natural_helix_pitch_per_turn"])
        self.internal_gamma = float(guide["internal_damping_gamma"])
        self.ambient_nu = float(guide["ambient_drag_nu"])
        self.gravity = grav

        self.root_rest = np.asarray(drive["rest_position"], dtype=float)
        self.root_axis = np.asarray(drive["axis"], dtype=float)
        self.root_start = float(drive["start_time"])
        self.root_amp = float(drive["amplitude"])
        self.root_freq = float(drive["frequency"])
        self.root_phase = float(drive["phase"])

        area = math.pi * self.cross_radius ** 2
        self.line_density = self.density * area

        init_frame = scene.initial_state()
        self.position = np.array(init_frame.position, dtype=float)
        self.velocity = np.array(init_frame.velocity, dtype=float)
        self.mass = np.array(init_frame.mass, dtype=float)

        N = len(self.position)
        self.N = N

        diffs = np.diff(self.position, axis=0)
        self.rest_lengths = np.linalg.norm(diffs, axis=1)

        # Reference helix relative to root
        turns = self.material_length / math.sqrt(
            (2.0 * math.pi * self.helix_radius) ** 2 + self.helix_pitch ** 2
        )
        theta_ref = np.linspace(0.0, 2.0 * math.pi * turns, N)
        ref = np.empty((N, 3), dtype=float)
        ref[:, 0] = self.helix_radius * (1.0 - np.cos(theta_ref))
        ref[:, 1] = self.helix_radius * np.sin(theta_ref)
        ref[:, 2] = -(self.helix_pitch / (2.0 * math.pi)) * theta_ref
        poly_len = np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()
        ref *= self.material_length / poly_len
        ref -= ref[0]
        self.ref_shape = ref

        E_modulus = 5.0e9
        I_area = math.pi * self.cross_radius ** 4 / 4.0
        self.EI = E_modulus * I_area

        seg_len = self.material_length / (N - 1)
        m_node = self.line_density * seg_len

        # Use moderate number of substeps
        self.n_substeps = 20
        self.dt_sub = 0.02 / self.n_substeps  # 1ms

        # Bending relaxation: fraction of displacement towards reference per substep
        # Choose so strand relaxes over ~0.5s timescale
        # With 1000 substeps/s and alpha per step, effective rate = alpha * 1000 s^-1
        # For tau = 0.5s: alpha = 1/(tau * n_substeps_per_s) = 1/500 = 0.002
        self.bend_alpha = 0.0002

        self.current_time = 0.0
        self.cum_ext_impulse = np.zeros(3, dtype=float)
        self.cum_inflow_mom = np.zeros(3, dtype=float)
        self.cum_outflow_mom = np.zeros(3, dtype=float)

        self.node_index = np.arange(N, dtype=int)
        self.is_root = (self.node_index == 0).astype(float)
        self.is_tip = (self.node_index == N - 1).astype(float)

        self.inv_mass = np.zeros(N, dtype=float)
        mask = self.mass > 1e-30
        self.inv_mass[mask] = 1.0 / self.mass[mask]
        self.inv_mass[0] = 0.0

    def _root_position(self, t: float) -> np.ndarray:
        if t < self.root_start:
            return self.root_rest.copy()
        dt = t - self.root_start
        disp = self.root_amp * math.sin(2.0 * math.pi * self.root_freq * dt + self.root_phase)
        return self.root_rest + disp * self.root_axis

    def _root_velocity(self, t: float) -> np.ndarray:
        if t < self.root_start:
            return np.zeros(3)
        dt = t - self.root_start
        omega = 2.0 * math.pi * self.root_freq
        vel = self.root_amp * omega * math.cos(omega * dt + self.root_phase)
        return vel * self.root_axis

    def _project_inextensibility(self, pos, max_iters=50):
        N = self.N
        inv_m = self.inv_mass
        for _it in range(max_iters):
            diffs = pos[1:] - pos[:-1]
            dists = np.linalg.norm(diffs, axis=1)
            dists_safe = np.maximum(dists, 1e-15)
            errors = dists - self.rest_lengths
            directions = diffs / dists_safe[:, None]
            w_sum = inv_m[:-1] + inv_m[1:]
            w_sum = np.maximum(w_sum, 1e-30)
            corr_mag = errors / w_sum
            corrections = corr_mag[:, None] * directions
            pos[1:] -= inv_m[1:, None] * corrections
            pos[:-1] += inv_m[:-1, None] * corrections
            if np.max(np.abs(errors)) < 1e-12:
                break
        return pos

    def advance_to(self, time: float) -> "StateFrame":
        from worldsolver.contracts import StateFrame

        target_time = float(time)
        if target_time > self.current_time:
            self._step_to(target_time)

        pos = self.position.copy()
        vel = self.velocity.copy()
        pos[0] = self._root_position(target_time)
        vel[0] = self._root_velocity(target_time)

        N = self.N
        ext_force_gravity = self.mass[:, None] * self.gravity[None, :]
        ext_force_drag = -self.ambient_nu * vel
        ext_force_total = ext_force_gravity + ext_force_drag
        domain_weights = np.ones((1, N), dtype=float)

        edge_vecs = np.diff(pos, axis=0)
        edge_lens = np.linalg.norm(edge_vecs, axis=1)
        constraint_scale = self.material_length / (N - 1)

        total_mass = float(np.sum(self.mass))
        ref_momentum = total_mass * self.root_amp * 2.0 * math.pi * self.root_freq
        if ref_momentum < 1e-20:
            ref_momentum = 1.0

        frame = StateFrame(
            time=target_time,
            position=pos,
            velocity=vel,
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "node_index": self.node_index.copy(),
                "is_root": self.is_root.copy(),
                "is_free_tip": self.is_tip.copy(),
                "kinematic_root": self.is_root.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_ext_impulse.reshape(1, 3).copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_mom.reshape(1, 3).copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_mom.reshape(1, 3).copy(),
                "budget_domain_weights": domain_weights,
                "budget_external_force": ext_force_total.reshape(1, N, 3).copy(),
                "constraint_value": edge_lens,
                "constraint_target": self.rest_lengths,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "constraint_scale": float(constraint_scale),
                "reference_momentum": float(ref_momentum),
                "material_length": self.material_length,
                "internal_damping_gamma": self.internal_gamma,
                "ambient_drag_nu": self.ambient_nu,
            },
        )
        frame.validate()
        return frame

    def _step_to(self, target_time: float):
        dt = self.dt_sub
        t = self.current_time
        # Use integer substep count to avoid floating-point drift
        total_dt = target_time - t
        if total_dt <= 0:
            return
        n_steps = max(1, round(total_dt / dt))
        actual_dt = total_dt / n_steps
        for _ in range(n_steps):
            self._substep(actual_dt, t, t + actual_dt)
            t += actual_dt
        self.current_time = target_time

    def _substep(self, dt, t_old, t_new):
        """Stable PBD substep with bending relaxation."""
        N = self.N
        pos_old = self.position.copy()

        # Compute external acceleration
        f_grav = self.mass[:, None] * self.gravity[None, :]
        f_drag = -self.ambient_nu * self.velocity

        # Internal damping along edges
        edge_vecs = np.diff(self.position, axis=0)
        edge_lens = np.linalg.norm(edge_vecs, axis=1)
        edge_dirs = edge_vecs / np.maximum(edge_lens, 1e-15)[:, None]
        rel_vel = np.diff(self.velocity, axis=0)
        strain_rate = np.sum(rel_vel * edge_dirs, axis=1)
        f_damp = np.zeros((N, 3), dtype=float)
        damp_f = (self.internal_gamma * strain_rate)[:, None] * edge_dirs
        f_damp[:-1] -= damp_f
        f_damp[1:] += damp_f

        f_ext = f_grav + f_drag + f_damp

        # Predict positions
        pos_pred = self.position.copy()
        accel = np.zeros((N, 3), dtype=float)
        mask = self.mass > 1e-30
        accel[mask] = f_ext[mask] / self.mass[mask, None]
        accel[0] = 0.0

        pos_pred[1:] += dt * self.velocity[1:] + 0.5 * dt * dt * accel[1:]
        pos_pred[0] = self._root_position(t_new)

        # Project inextensibility FIRST
        self._project_inextensibility(pos_pred)

        # THEN apply bending relaxation towards reference shape
        # This gently pulls nodes towards their reference positions
        root_pos = self._root_position(t_new)
        target = root_pos + self.ref_shape
        bend_disp = target - pos_pred
        bend_disp[0] = 0.0  # don't move root
        
        # Apply small fraction
        pos_pred[1:] += self.bend_alpha * bend_disp[1:]
        
        # Re-project after bending (to maintain inextensibility)
        self._project_inextensibility(pos_pred)

        # Derive velocity from constrained position change
        self.velocity = (pos_pred - pos_old) / dt
        self.velocity[0] = self._root_velocity(t_new)

        self.position = pos_pred

        # Accumulate external impulse
        ext_force = f_grav + f_drag
        self.cum_ext_impulse += dt * np.sum(ext_force, axis=0)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "pbd_bending_relax",
            "node_count": self.N,
            "material_length": self.material_length,
            "dt_sub": self.dt_sub,
            "bend_alpha": self.bend_alpha,
        }
