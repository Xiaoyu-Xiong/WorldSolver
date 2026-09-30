"""Particle-based sponge saturation and wringing solver.

Implements a two-way coupled explicit time-stepping simulation for the scene
"A Sponge Is Saturated and Wrung Out".  The model uses a spring-network
sponge, Lagrangian water particles, absorption/expulsion of pore water, and
kinematic grip controllers.  It exports all required StateFrame trajectory
fields.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import Solver, StateFrame


class BenchmarkSolver(Solver):
    """Solver for the sponge saturation/wringing benchmark."""

    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.rng = np.random.default_rng(seed)

        # Time / stepping
        self.sample_dt: float = float(self.config["sample_dt"])
        self.duration: float = float(self.config["duration"])
        self.gravity = np.array(self.config["gravity"], dtype=float)

        # Event intervals from config
        self.wet_start, self.wet_end = self.config["event_sequence"]["wetting_interval"]
        self.lift_start, self.lift_end = self.config["event_sequence"]["lift_interval"]
        self.wring_start, self.wring_end = self.config["event_sequence"]["wring_interval"]
        src = self.config["water_source"]
        self.source_start = float(src["starts_at"])
        self.source_end = float(src["stops_at"])
        self.source_center = np.array(src["center"], dtype=float)
        self.source_dir = np.array(src["direction"], dtype=float)
        self.source_radius = float(src["radius"])
        self.source_speed = float(src["speed"])

        # Controls
        self.left_cfg = self.config["controls"]["left_end"]
        self.right_cfg = self.config["controls"]["right_end"]
        self.left_init = np.array(self.left_cfg["initial_position"], dtype=float)
        self.right_init = np.array(self.right_cfg["initial_position"], dtype=float)
        self.left_lift = np.array(self.left_cfg["lift_displacement"], dtype=float)
        self.right_lift = np.array(self.right_cfg["lift_displacement"], dtype=float)
        self.left_twist_deg = float(self.left_cfg["twist_degrees"])
        self.right_twist_deg = float(self.right_cfg["twist_degrees"])
        self.twist_axis = np.array(self.left_cfg["twist_axis"], dtype=float)
        self.twist_axis /= np.linalg.norm(self.twist_axis)

        # Initial state
        init = scene.initial_state()
        self.pos: np.ndarray = init.position.copy()
        self.vel: np.ndarray = init.velocity.copy()
        self.mass: np.ndarray = init.mass.copy()
        self.body_id: np.ndarray = np.asarray(init.body_id, dtype=np.int32)
        self.saturation: np.ndarray = np.asarray(init.attributes["saturation"], dtype=float).copy()
        self.porosity: np.ndarray = np.asarray(init.attributes["porosity"], dtype=float).copy()
        self.phase: np.ndarray = np.asarray(init.attributes["phase"], dtype=float).copy()

        # Material mass bookkeeping: columns [solid, liquid, other]
        self.K_MAT = 3
        self.mat_mass = np.zeros((len(self.mass), self.K_MAT), dtype=float)
        sponge_mask = self.body_id == 1
        water_mask = self.body_id == 2
        grip_mask = np.isin(self.body_id, [10, 11])
        self.mat_mass[sponge_mask, 0] = self.mass[sponge_mask]
        self.mat_mass[water_mask, 1] = self.mass[water_mask]
        self.mat_mass[grip_mask, 2] = self.mass[grip_mask]

        # Sponge capacity per particle for pore water (kg)
        self.capacity = np.full(len(self.mass), 0.0, dtype=float)
        self.capacity[sponge_mask] = 0.001

        # Store rest configuration for sponge control / springs
        self.rest_pos = self.pos.copy()
        self.rest_x = self.rest_pos[:, 0].copy()
        self.left_ctrl_mask = sponge_mask & (self.rest_x >= self.left_cfg["region_x"][0]) & (
            self.rest_x <= self.left_cfg["region_x"][1]
        )
        self.right_ctrl_mask = sponge_mask & (self.rest_x >= self.right_cfg["region_x"][0]) & (
            self.rest_x <= self.right_cfg["region_x"][1]
        )

        # Build sponge spring network
        self.sponge_mask = sponge_mask
        self.water_mask = water_mask
        self.grip_mask = grip_mask
        self.sponge_idx = np.where(sponge_mask)[0]
        self.springs: list[tuple[int, int, float]] = []
        for i in range(len(self.sponge_idx)):
            for j in range(i + 1, len(self.sponge_idx)):
                a = self.sponge_idx[i]
                b = self.sponge_idx[j]
                r0 = np.linalg.norm(self.rest_pos[a] - self.rest_pos[b])
                if r0 < 0.07:
                    self.springs.append((a, b, r0))

        # Budget bookkeeping
        self.G = 4  # total, sponge, water, grips
        self.K_TERMS = 4  # gravity, contact, drag/coupling, control
        self.budget_impulse = np.zeros((self.G, 3), dtype=float)
        self.budget_inflow = np.zeros((self.G, 3), dtype=float)
        self.budget_outflow = np.zeros((self.G, 3), dtype=float)
        self.cum_mass_source = np.zeros(self.K_MAT, dtype=float)

        self.reference_momentum = 1.0

        # Source emitter state
        self.emit_interval = 0.015
        self.next_emit_time = self.source_start
        self.emit_pattern = np.array(
            [[0.0, 0.0], [0.008, 0.0], [-0.008, 0.0], [0.0, 0.008], [0.0, -0.008]],
            dtype=float,
        )
        self.emit_count = 0

        # Substep
        self.dt = 0.0002
        self.t = 0.0

        # Caps
        self.max_water_particles = 220
        self.water_cut = 0.015
        self.sponge_r_cut = 0.028
        self.drag_coeff = 60.0
        self.contact_stiff = 300.0
        self.sponge_k = 100.0
        self.sponge_c = 8.0
        self.ctrl_k = 200.0
        self.ctrl_c = 12.0
        self.floor_k = 800.0
        self.floor_c = 40.0
        self.diffusion = 0.02
        self.ejection_rate = 2.0

    # ------------------------------------------------------------------ helpers
    def _ramp(self, t: float, t0: float, t1: float, y0: float, y1: float) -> float:
        if t <= t0:
            return y0
        if t >= t1:
            return y1
        u = (t - t0) / (t1 - t0)
        return y0 + u * (y1 - y0)

    def _rotation_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)
        c = math.cos(angle)
        s = math.sin(angle)
        a = axis
        # Rodrigues' rotation matrix
        R = np.array(
            [
                [c + a[0] * a[0] * (1 - c), a[0] * a[1] * (1 - c) - a[2] * s, a[0] * a[2] * (1 - c) + a[1] * s],
                [a[1] * a[0] * (1 - c) + a[2] * s, c + a[1] * a[1] * (1 - c), a[1] * a[2] * (1 - c) - a[0] * s],
                [a[2] * a[0] * (1 - c) - a[1] * s, a[2] * a[1] * (1 - c) + a[0] * s, c + a[2] * a[2] * (1 - c)],
            ],
            dtype=float,
        )
        return R

    def _grip_targets(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return target positions and linear velocities for the two grips at time t."""
        active = t >= self.left_cfg["active_from"]
        if not active:
            return self.left_init.copy(), np.zeros(3), self.right_init.copy(), np.zeros(3)

        lift_t = self._ramp(t, self.lift_start, self.lift_end, 0.0, 1.0)
        twist_t = self._ramp(t, self.wring_start, self.wring_end, 0.0, 1.0)

        left_center = self.left_init + lift_t * self.left_lift
        right_center = self.right_init + lift_t * self.right_lift

        left_angle = math.radians(self.left_twist_deg * twist_t)
        right_angle = math.radians(self.right_twist_deg * twist_t)

        # Derivatives of centers (only lift contributes)
        if self.lift_start <= t <= self.lift_end:
            lift_rate = 1.0 / (self.lift_end - self.lift_start)
        else:
            lift_rate = 0.0
        left_vel = lift_rate * self.left_lift
        right_vel = lift_rate * self.right_lift

        return left_center, left_vel, right_center, right_vel

    def _sponge_target(self, t: float, rest: np.ndarray, side: str) -> np.ndarray:
        """Return target positions for a set of sponge rest points on one end."""
        left_center, _, right_center, _ = self._grip_targets(t)
        if side == "left":
            center = left_center
            angle = math.radians(self.left_twist_deg * self._ramp(t, self.wring_start, self.wring_end, 0.0, 1.0))
            rest_center = self.left_init
        else:
            center = right_center
            angle = math.radians(self.right_twist_deg * self._ramp(t, self.wring_start, self.wring_end, 0.0, 1.0))
            rest_center = self.right_init
        rel = rest - rest_center
        rotated = rel @ self._rotation_matrix(self.twist_axis, angle).T
        return center + rotated

    # --------------------------------------------------------------------- step
    def _add_water_particle(self, pos: np.ndarray, vel: np.ndarray, mass: float) -> None:
        self.pos = np.vstack([self.pos, pos.reshape(1, 3)])
        self.vel = np.vstack([self.vel, vel.reshape(1, 3)])
        self.mass = np.append(self.mass, mass)
        self.body_id = np.append(self.body_id, 2)
        self.mat_mass = np.vstack([self.mat_mass, [[0.0, mass, 0.0]]])
        self.saturation = np.append(self.saturation, 1.0)
        self.porosity = np.append(self.porosity, 0.0)
        self.phase = np.append(self.phase, 1.0)

    def _remove_water(self, idx: int) -> None:
        for arr_name in ("pos", "vel", "mass", "body_id", "mat_mass", "saturation", "porosity", "phase"):
            arr = getattr(self, arr_name)
            setattr(self, arr_name, np.delete(arr, idx, axis=0))
        self.water_mask = self.body_id == 2

    def _domain_masks(self) -> np.ndarray:
        n = len(self.body_id)
        weights = np.zeros((self.G, n), dtype=float)
        weights[0] = 1.0
        weights[1] = (self.body_id == 1).astype(float)
        weights[2] = (self.body_id == 2).astype(float)
        weights[3] = np.isin(self.body_id, [10, 11]).astype(float)
        return weights

    def _step(self, dt: float) -> None:
        t = self.t
        n = len(self.body_id)
        force = np.zeros((n, 3), dtype=float)
        term = np.zeros((n, self.K_TERMS, 3), dtype=float)

        masks = {
            "sponge": self.body_id == 1,
            "water": self.body_id == 2,
            "grip": np.isin(self.body_id, [10, 11]),
        }
        dynamic = masks["sponge"] | masks["water"]

        # Gravity
        g_force = self.mass[:, None] * self.gravity
        force[dynamic] += g_force[dynamic]
        term[dynamic, 0] += g_force[dynamic]

        # Sponge elastic springs + damping
        for a, b, r0 in self.springs:
            dx = self.pos[b] - self.pos[a]
            dist = np.linalg.norm(dx)
            if dist < 1e-12:
                continue
            u = dx / dist
            dv = self.vel[b] - self.vel[a]
            fmag = self.sponge_k * (dist - r0) + self.sponge_c * np.dot(dv, u)
            f = fmag * u
            force[a] += f
            force[b] -= f
            # internal spring force: not attributed to any external budget term

        # Water-sponge drag / coupling and contact
        if np.any(masks["water"]) and np.any(masks["sponge"]):
            water_idx = np.where(masks["water"])[0]
            sponge_idx = np.where(masks["sponge"])[0]
            sp = self.pos[sponge_idx]
            sw = self.pos[water_idx]
            for i_w in range(len(water_idx)):
                w = water_idx[i_w]
                deltas = sp - sw[i_w]
                d2 = np.einsum("ij,ij->i", deltas, deltas)
                close = sponge_idx[d2 < self.sponge_r_cut**2]
                if close.size == 0:
                    continue
                for s in close:
                    dx = self.pos[s] - self.pos[w]
                    d = np.linalg.norm(dx)
                    if d < 1e-12:
                        continue
                    u = dx / d
                    # repulsion
                    if d < self.sponge_r_cut:
                        f_rep = self.contact_stiff * (self.sponge_r_cut - d) * u
                        force[w] += f_rep
                        force[s] -= f_rep
                        term[w, 1] += f_rep
                        term[s, 1] -= f_rep
                    # drag coupling
                    v_rel = self.vel[w] - self.vel[s]
                    f_drag = -self.drag_coeff * self.mass[w] * v_rel * (1.0 - d / self.sponge_r_cut)
                    force[w] += f_drag
                    force[s] -= f_drag
                    term[w, 2] += f_drag
                    term[s, 2] -= f_drag

        # Control forces on sponge end regions
        if t >= self.left_cfg["active_from"]:
            left_target = self._sponge_target(t, self.rest_pos[self.left_ctrl_mask], "left")
            left_vel = (self._sponge_target(t + dt, self.rest_pos[self.left_ctrl_mask], "left") - left_target) / dt
            idx = np.where(self.left_ctrl_mask)[0]
            for k, pidx in enumerate(idx):
                f = self.ctrl_k * (left_target[k] - self.pos[pidx]) + self.ctrl_c * (left_vel[k] - self.vel[pidx])
                force[pidx] += f
                term[pidx, 3] += f

            right_target = self._sponge_target(t, self.rest_pos[self.right_ctrl_mask], "right")
            right_vel = (self._sponge_target(t + dt, self.rest_pos[self.right_ctrl_mask], "right") - right_target) / dt
            idx = np.where(self.right_ctrl_mask)[0]
            for k, pidx in enumerate(idx):
                f = self.ctrl_k * (right_target[k] - self.pos[pidx]) + self.ctrl_c * (right_vel[k] - self.vel[pidx])
                force[pidx] += f
                term[pidx, 3] += f

        # Floor / sink contact for dynamic particles
        # Sponge floor only before lift; water always drains
        for pidx in range(n):
            if self.body_id[pidx] in (10, 11):
                continue
            z = self.pos[pidx, 2]
            is_sponge = self.body_id[pidx] == 1
            if is_sponge and t >= self.lift_start:
                continue
            if z < self.water_cut:
                pen = self.water_cut - z
                v_n = self.vel[pidx, 2]
                f_z = self.floor_k * pen - self.floor_c * v_n
                f = np.array([0.0, 0.0, f_z], dtype=float)
                force[pidx] += f
                term[pidx, 1] += f

        # Update velocities
        inv_m = np.zeros(n, dtype=float)
        inv_m[dynamic] = 1.0 / self.mass[dynamic]
        self.vel[dynamic] += dt * force[dynamic] * inv_m[dynamic, None]

        # Kinematic grip velocities
        if np.any(masks["grip"]):
            left_center_now, left_vel, right_center_now, right_vel = self._grip_targets(t)
            left_center_next, _, right_center_next, _ = self._grip_targets(t + dt)
            # Simple derivative target velocity
            grip_vel = {10: left_vel, 11: right_vel}
            for g in np.where(masks["grip"])[0]:
                bid = self.body_id[g]
                self.vel[g] = grip_vel[bid]

        # Update positions
        self.pos += dt * self.vel

        # Hard floor for sponge before lift
        sponge_idx_arr = np.where(masks["sponge"])[0]
        if t < self.lift_start and sponge_idx_arr.size:
            below = sponge_idx_arr[self.pos[sponge_idx_arr, 2] < self.water_cut]
            self.pos[below, 2] = self.water_cut
            self.vel[below, 2] = np.maximum(self.vel[below, 2], 0.0)

        # Update budgets: cumulative external impulse per domain
        weights = self._domain_masks()
        total_force_per_domain = (weights[:, :, None] * term).sum(axis=1)  # [G, K_terms, 3]
        total_force = total_force_per_domain.sum(axis=1)  # [G, 3]
        self.budget_impulse += dt * total_force

        # Source emission
        while t + dt >= self.next_emit_time and self.next_emit_time < self.source_end:
            if np.sum(self.water_mask) < self.max_water_particles:
                pat = self.emit_pattern[self.emit_count % len(self.emit_pattern)]
                r = self.source_radius * (0.6 + 0.4 * ((self.emit_count % 7) / 6.0))
                offset = np.array([pat[0] * r, pat[1] * r, 0.0], dtype=float)
                pos = self.source_center + offset
                vel = self.source_dir * self.source_speed
                m = 0.00095
                self._add_water_particle(pos, vel, m)
                # inflow momentum
                p = m * vel
                self.budget_inflow[0] += p
                self.budget_inflow[2] += p
                self.cum_mass_source[1] += m
                self.emit_count += 1
                self.next_emit_time += self.emit_interval
            else:
                break

        # Absorption of water into dry sponge
        if np.any(self.water_mask) and np.any(masks["sponge"]):
            water_idx = np.where(self.water_mask)[0]
            sponge_idx = np.where(masks["sponge"])[0]
            sp = self.pos[sponge_idx]
            to_remove: list[int] = []
            for i_w in water_idx:
                if self.mass[i_w] <= 0.0:
                    to_remove.append(i_w)
                    continue
                deltas = sp - self.pos[i_w]
                d2 = np.einsum("ij,ij->i", deltas, deltas)
                if d2.min() >= self.sponge_r_cut**2:
                    continue
                s = sponge_idx[np.argmin(d2)]
                cap = self.capacity[s]
                room = (1.0 - self.saturation[s]) * cap
                if room <= 0.0:
                    continue
                take = min(self.mass[i_w], room)
                self.mat_mass[s, 1] += take
                self.saturation[s] += take / cap
                self.mass[s] = self.mat_mass[s].sum()
                self.mass[i_w] -= take
                self.mat_mass[i_w, 1] -= take
                if self.mass[i_w] <= 1e-8:
                    to_remove.append(i_w)
            # remove from highest index to keep indices valid
            for i_w in sorted(set(to_remove), reverse=True):
                self._remove_water(i_w)

        # Diffusion of saturation within sponge
        if self.sponge_idx.size > 1:
            for a, b, _ in self.springs:
                sa = self.saturation[a]
                sb = self.saturation[b]
                cap = self.capacity[a]
                if cap <= 0.0:
                    continue
                delta = self.diffusion * (sb - sa) * dt
                if abs(delta) < 1e-12:
                    continue
                # transfer mass from b to a
                m_transfer = delta * cap
                total_liquid_a = self.mat_mass[a, 1]
                total_liquid_b = self.mat_mass[b, 1]
                # clamp to available water
                if m_transfer > 0:
                    m_transfer = min(m_transfer, total_liquid_b)
                else:
                    m_transfer = max(m_transfer, -total_liquid_a)
                self.mat_mass[a, 1] += m_transfer
                self.mat_mass[b, 1] -= m_transfer
                self.saturation[a] = self.mat_mass[a, 1] / cap
                self.saturation[b] = self.mat_mass[b, 1] / self.capacity[b]
                self.mass[a] = self.mat_mass[a].sum()
                self.mass[b] = self.mat_mass[b].sum()

        # Ejection of water from sponge during wringing (middle narrows)
        if self.wring_start <= t <= self.wring_end:
            sponge_idx = np.where(self.sponge_mask)[0]
            for s in sponge_idx:
                x = self.pos[s, 0]
                if abs(x) > 0.07:
                    continue
                sval = self.saturation[s]
                if sval <= 0.05:
                    continue
                if np.sum(self.water_mask) >= self.max_water_particles:
                    break
                cap = self.capacity[s]
                available = sval * cap
                ejected = min(available, available * self.ejection_rate * dt)
                if ejected < 1e-8:
                    continue
                self.mat_mass[s, 1] -= ejected
                self.saturation[s] = self.mat_mass[s, 1] / cap
                self.mass[s] = self.mat_mass[s].sum()
                # new water particle expelled downward/outward
                dir_out = np.array([0.0, self.pos[s, 1] * 2.0, -1.0], dtype=float)
                norm = np.linalg.norm(dir_out)
                if norm > 0:
                    dir_out /= norm
                else:
                    dir_out = np.array([0.0, 0.0, -1.0], dtype=float)
                v = dir_out * 0.6
                self._add_water_particle(self.pos[s].copy(), v, ejected)

        # Drain / outflow
        if np.any(self.water_mask):
            water_idx = np.where(self.water_mask)[0]
            to_remove = []
            for i_w in water_idx:
                if self.pos[i_w, 2] < self.water_cut:
                    # outflow
                    p = self.mass[i_w] * self.vel[i_w]
                    self.budget_outflow[0] += p
                    self.budget_outflow[2] += p
                    self.cum_mass_source[1] -= self.mass[i_w]
                    to_remove.append(i_w)
            for i_w in sorted(set(to_remove), reverse=True):
                self._remove_water(i_w)

        # Sync water mask
        self.water_mask = self.body_id == 2
        self.t += dt

    def _build_frame(self) -> StateFrame:
        # Recompute current external-force terms for the budget field
        n = len(self.body_id)
        term = np.zeros((n, self.K_TERMS, 3), dtype=float)

        masks = {
            "sponge": self.body_id == 1,
            "water": self.body_id == 2,
            "grip": np.isin(self.body_id, [10, 11]),
        }
        dynamic = masks["sponge"] | masks["water"]

        # gravity
        g_force = self.mass[:, None] * self.gravity
        term[dynamic, 0] = g_force[dynamic]

        # contact / floor
        for pidx in range(n):
            if self.body_id[pidx] in (10, 11):
                continue
            z = self.pos[pidx, 2]
            is_sponge = self.body_id[pidx] == 1
            if is_sponge and self.t >= self.lift_start:
                continue
            if z < self.water_cut:
                pen = self.water_cut - z
                v_n = self.vel[pidx, 2]
                f = np.array([0.0, 0.0, self.floor_k * pen - self.floor_c * v_n], dtype=float)
                term[pidx, 1] = f

        # drag coupling (current snapshot)
        if np.any(masks["water"]) and np.any(masks["sponge"]):
            water_idx = np.where(masks["water"])[0]
            sponge_idx = np.where(masks["sponge"])[0]
            sp = self.pos[sponge_idx]
            for i_w in range(len(water_idx)):
                w = water_idx[i_w]
                deltas = sp - self.pos[w]
                d2 = np.einsum("ij,ij->i", deltas, deltas)
                close = sponge_idx[d2 < self.sponge_r_cut**2]
                for s in close:
                    dx = self.pos[s] - self.pos[w]
                    d = np.linalg.norm(dx)
                    if d < 1e-12:
                        continue
                    u = dx / d
                    f_rep = self.contact_stiff * (self.sponge_r_cut - d) * u
                    term[w, 1] += f_rep
                    term[s, 1] -= f_rep
                    v_rel = self.vel[w] - self.vel[s]
                    f_drag = -self.drag_coeff * self.mass[w] * v_rel * (1.0 - d / self.sponge_r_cut)
                    term[w, 2] += f_drag
                    term[s, 2] -= f_drag

        # control
        if self.t >= self.left_cfg["active_from"]:
            left_target = self._sponge_target(self.t, self.rest_pos[self.left_ctrl_mask], "left")
            for k, pidx in enumerate(np.where(self.left_ctrl_mask)[0]):
                term[pidx, 3] += self.ctrl_k * (left_target[k] - self.pos[pidx])
            right_target = self._sponge_target(self.t, self.rest_pos[self.right_ctrl_mask], "right")
            for k, pidx in enumerate(np.where(self.right_ctrl_mask)[0]):
                term[pidx, 3] += self.ctrl_k * (right_target[k] - self.pos[pidx])

        weights = self._domain_masks()
        budget_external_force = (weights[:, :, None, None] * term[None, :, :, :]).sum(axis=1)  # [G,K,3]

        # Interaction force snapshot
        interaction = np.zeros((3, 3), dtype=float)
        # 0: sponge-water coupling (drag on sponge)
        if np.any(masks["sponge"]):
            for s in np.where(masks["sponge"])[0]:
                interaction[0] -= term[s, 2]
        # 1: water-floor contact
        if np.any(masks["water"]):
            for w in np.where(masks["water"])[0]:
                interaction[1] += term[w, 1]
        # 2: grip-sponge control force on sponge
        if np.any(masks["sponge"]):
            for s in np.where(masks["sponge"])[0]:
                interaction[2] += term[s, 3]

        # Ensure body id present
        body_id = self.body_id

        # Update grip positions to kinematic targets
        left_center, _, right_center, _ = self._grip_targets(self.t)
        grip_idx = np.where(self.grip_mask)[0]
        if len(grip_idx) >= 2:
            self.pos[grip_idx[0]] = left_center
            self.pos[grip_idx[1]] = right_center

        frame = StateFrame(
            time=float(self.t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=body_id.copy(),
            attributes={
                "saturation": self.saturation.copy(),
                "porosity": self.porosity.copy(),
                "phase": self.phase.copy(),
                "mass_quantity": self.mat_mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.budget_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.budget_inflow.copy(),
                "budget_cumulative_outflow_momentum": self.budget_outflow.copy(),
                "budget_domain_weights": weights,
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "interaction_force": interaction,
            },
            scalars={"reference_momentum": float(self.reference_momentum)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        while self.t < time - 1e-12:
            step_dt = min(self.dt, time - self.t)
            self._step(step_dt)
        return self._build_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "BenchmarkSolver",
            "substep_dt": self.dt,
            "final_time": self.t,
            "num_particles": len(self.mass),
        }
