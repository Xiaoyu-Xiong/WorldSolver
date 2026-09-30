"""BenchmarkSolver for 103_cable_cylinder_winding.

XPBD-based extensible elastic cable with bending resistance,
cylinder contact with generalized power-law friction, and floor contact.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        cable_cfg = cfg["cable"]
        cyl_cfg = cfg["cylinder"]
        contact_cfg = cfg["contact"]

        self.n = int(cable_cfg["vertex_count"])
        self.rest_length = float(cable_cfg["length"])
        self.radius = float(cable_cfg["radius"])
        self.density = float(cable_cfg["density"])
        self.E = float(cable_cfg["young_modulus_pa"])
        self.G = float(cable_cfg["shear_modulus_pa"])
        area = math.pi * self.radius ** 2
        self.segment_rest = self.rest_length / (self.n - 1)
        nodal_mass = np.full(self.n, self.density * area * self.segment_rest)
        nodal_mass[0] *= 0.5
        nodal_mass[-1] *= 0.5
        self.mass = nodal_mass.astype(np.float64)
        self.inv_mass = np.where(self.mass > 0, 1.0 / self.mass, 0.0)

        self.area = area
        I_area = math.pi * self.radius ** 4 / 4.0
        self.EI = self.E * I_area
        self.EA = self.E * area

        # Cylinder
        self.cyl_center = np.asarray(cyl_cfg["center"], dtype=np.float64)
        self.cyl_axis = np.asarray(cyl_cfg["axis"], dtype=np.float64)
        self.cyl_axis /= np.linalg.norm(self.cyl_axis)
        self.cyl_radius = float(cyl_cfg["radius"])
        self.cyl_height = float(cyl_cfg["height"])
        self.cyl_half_h = self.cyl_height / 2.0
        az = self.cyl_axis
        tmp = np.array([1.0, 0.0, 0.0]) if abs(az[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        ax = np.cross(az, tmp); ax /= np.linalg.norm(ax)
        ay = np.cross(az, ax)
        self.cyl_basis = np.stack([ax, ay, az], axis=0)

        self.friction_kappa = float(contact_cfg["kappa"])
        self.friction_N = float(contact_cfg["exponent_N"])
        self.contact_enabled = bool(contact_cfg.get("cable_cylinder_enabled", True))
        self.self_contact_enabled = bool(contact_cfg.get("cable_self_contact_enabled", True))

        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)

        self.time = 0.0
        self.substep_dt = 0.0002
        self.max_substeps = 3000
        self.xpbd_iters = 12
        self.vel_damping = 0.999

        frame0 = scene.initial_state()
        self.position = np.array(frame0.position, dtype=np.float64).copy()
        self.velocity = np.array(frame0.velocity, dtype=np.float64).copy()

        self.cum_ext_impulse = np.zeros(3, dtype=np.float64)
        self.cum_ext_ang_impulse = np.zeros(3, dtype=np.float64)
        self.cum_in_mom = np.zeros(3, dtype=np.float64)
        self.cum_out_mom = np.zeros(3, dtype=np.float64)
        self.cum_in_amom = np.zeros(3, dtype=np.float64)
        self.cum_out_amom = np.zeros(3, dtype=np.float64)

        total_mass = float(self.mass.sum())
        self.ref_momentum = max(total_mass * 0.01, 1e-9)
        char_L = max(self.rest_length, 0.01)
        self.ref_ang_momentum = max(self.ref_momentum * char_L, 1e-9)

        self.last_ext_force = np.zeros((1, 1, 3), dtype=np.float64)
        self.last_ext_pos = np.zeros((1, 1, 3), dtype=np.float64)
        self.last_ext_couple = np.zeros((1, 1, 3), dtype=np.float64)

        self._setup_compliance()

    def _setup_compliance(self):
        dt = self.substep_dt
        self.alpha_stretch = 0.0  # PBD (inextensible)
        self.alpha_bend = self.segment_rest / (self.EI * dt * dt)
        self.alpha_contact = 1e-12 / (dt * dt)

    def _local_of(self, p):
        return (p - self.cyl_center) @ self.cyl_basis.T

    def _substep(self, dt):
        n = self.n
        pos_prev = self.position.copy()
        vel = self.velocity.copy()
        inv_m = self.inv_mass.copy()

        vel_pred = vel + self.gravity * dt
        vel_pred *= self.vel_damping
        pos = pos_prev + vel_pred * dt

        tgt = self.scene.grip_target(self.time + dt)
        drv = int(tgt["controlled_vertex"])
        target_pos = np.asarray(tgt["position"], dtype=np.float64)
        pos[drv] = target_pos
        inv_m_drv = inv_m.copy()
        inv_m_drv[drv] = 0.0

        lam_s = np.zeros(n - 1, dtype=np.float64)
        lam_b = np.zeros(max(n - 2, 0), dtype=np.float64)
        lam_c = np.zeros(n, dtype=np.float64)

        for _it in range(self.xpbd_iters):
            # === STRETCHING ===
            edges = pos[1:] - pos[:-1]
            seg_len = np.linalg.norm(edges, axis=1)
            seg_safe = np.maximum(seg_len, 1e-15)
            C_s = seg_len - self.segment_rest
            grad_dirs = edges / seg_safe[:, None]
            w_s = inv_m_drv[:-1] + inv_m_drv[1:]
            denom_s = w_s + self.alpha_stretch
            dl_s = -C_s / np.maximum(denom_s, 1e-30)
            lam_s += dl_s
            corr = (dl_s * inv_m_drv[1:])[:, None] * grad_dirs
            pos[1:] += corr
            pos[:-1] -= (dl_s * inv_m_drv[:-1])[:, None] * grad_dirs
            pos[drv] = target_pos

            # === BENDING (deviation-from-straight-line formulation) ===
            if n >= 3:
                mid_expected = 0.5 * (pos[:-2] + pos[2:])
                dev = pos[1:-1] - mid_expected
                dev_norm = np.linalg.norm(dev, axis=1)
                dev_safe = np.maximum(dev_norm, 1e-15)
                e1 = pos[1:-1] - pos[:-2]
                e2 = pos[2:] - pos[1:-1]
                l1 = np.linalg.norm(e1, axis=1)
                l2 = np.linalg.norm(e2, axis=1)
                l_eff = 0.5 * (l1 + l2)
                l_eff_safe = np.maximum(l_eff, 1e-15)
                # Bending angle approximation: theta ≈ 2*dev_norm/l_eff
                C_b = 2.0 * dev_norm / l_eff_safe
                # Gradients
                grad_mid = (2.0 / l_eff_safe)[:, None] * (dev / dev_safe[:, None])
                grad_end = (-1.0 / l_eff_safe)[:, None] * (dev / dev_safe[:, None])
                w_b = (np.sum(grad_end**2, axis=1) * (inv_m_drv[:-2] + inv_m_drv[2:]) +
                       np.sum(grad_mid**2, axis=1) * inv_m_drv[1:-1])
                denom_b = w_b + self.alpha_bend
                dl_b = -(C_b + self.alpha_bend * lam_b) / np.maximum(denom_b, 1e-30)
                lam_b += dl_b
                pos[:-2] += (dl_b * inv_m_drv[:-2])[:, None] * grad_end
                pos[2:] += (dl_b * inv_m_drv[2:])[:, None] * grad_end
                pos[1:-1] += (dl_b * inv_m_drv[1:-1])[:, None] * grad_mid
                pos[drv] = target_pos

            # === CYLINDER CONTACT ===
            if self.contact_enabled:
                local = self._local_of(pos)
                rho = np.sqrt(local[:, 0]**2 + local[:, 1]**2)
                rho_safe = np.maximum(rho, 1e-15)
                pen = (self.cyl_radius + self.radius) - rho
                zloc = local[:, 2]
                within_z = (zloc >= -self.cyl_half_h) & (zloc <= self.cyl_half_h)
                active = (pen > 0.0) & within_z
                if np.any(active):
                    idx = np.where(active)[0]
                    nx_loc = local[idx, 0] / rho_safe[idx]
                    ny_loc = local[idx, 1] / rho_safe[idx]
                    n_loc = np.stack([nx_loc, ny_loc, np.zeros_like(nx_loc)], axis=1)
                    n_world = n_loc @ self.cyl_basis
                    C_c = pen[idx]
                    grad_c = -n_world
                    w_c = inv_m_drv[idx] * np.sum(grad_c**2, axis=1)
                    denom_c = w_c + self.alpha_contact
                    dl_c = -(C_c + self.alpha_contact * lam_c[idx]) / np.maximum(denom_c, 1e-30)
                    dl_c = np.minimum(dl_c, 0.0)
                    lam_c[idx] += dl_c
                    pos[idx] += (dl_c * inv_m_drv[idx])[:, None] * grad_c
                    pos[drv] = target_pos

        # === FRICTION ===
        if self.contact_enabled:
            local = self._local_of(pos)
            rho = np.sqrt(local[:, 0]**2 + local[:, 1]**2)
            rho_safe = np.maximum(rho, 1e-15)
            pen = (self.cyl_radius + self.radius) - rho
            zloc = local[:, 2]
            within_z = (zloc >= -self.cyl_half_h) & (zloc <= self.cyl_half_h)
            in_contact = ((pen > -1e-6) & within_z) | (lam_c < -1e-10)
            if np.any(in_contact):
                idx = np.where(in_contact)[0]
                nx_loc = local[idx, 0] / rho_safe[idx]
                ny_loc = local[idx, 1] / rho_safe[idx]
                n_loc = np.stack([nx_loc, ny_loc, np.zeros_like(nx_loc)], axis=1)
                n_world = n_loc @ self.cyl_basis
                vel_new = (pos - pos_prev) / dt
                v_rel = vel_new[idx]
                vn = np.sum(v_rel * n_world, axis=1, keepdims=True) * n_world
                vt = v_rel - vn
                vt_norm = np.linalg.norm(vt, axis=1)
                vt_safe = np.maximum(vt_norm, 1e-15)
                vt_dir = vt / vt_safe[:, None]
                fn_mag = np.abs(lam_c[idx]) / dt
                fn_mag = np.maximum(fn_mag, 0.0)
                ft_bound = self.friction_kappa * np.power(np.maximum(fn_mag, 1e-15), self.friction_N)
                ft_desired = self.mass[idx] * vt_norm / dt
                ft_mag = np.minimum(ft_desired, ft_bound)
                dv = -(ft_mag * dt / self.mass[idx])[:, None] * vt_dir
                vel_new[idx] += dv
                pos = pos_prev + vel_new * dt
                pos[drv] = target_pos

        # === FLOOR ===
        floor_z = 0.0
        below = pos[:, 2] < (floor_z + self.radius)
        if np.any(below):
            idx_f = np.where(below)[0]
            pos[idx_f, 2] = floor_z + self.radius
            vel_new = (pos - pos_prev) / dt
            vel_new[idx_f, 2] = np.maximum(vel_new[idx_f, 2], 0.0)
            vh = vel_new[idx_f, :2]
            vh_norm = np.linalg.norm(vh, axis=1)
            vh_safe = np.maximum(vh_norm, 1e-15)
            fn_floor = self.mass[idx_f] * abs(self.gravity[2])
            ft_bound_f = self.friction_kappa * np.power(np.maximum(fn_floor, 1e-15), self.friction_N)
            ft_desired_f = self.mass[idx_f] * vh_norm
            ft_mag_f = np.minimum(ft_desired_f, ft_bound_f)
            dv_h = -(ft_mag_f / self.mass[idx_f])[:, None] * (vh / vh_safe[:, None])
            vel_new[idx_f, :2] += dv_h
            pos = pos_prev + vel_new * dt
            pos[drv] = target_pos

        # === SELF-CONTACT ===
        if self.self_contact_enabled and n > 3:
            min_sep = 2.0 * self.radius
            stride = 4
            for i in range(0, n - 2, stride):
                for j in range(i + 2, n, stride):
                    diff = pos[j] - pos[i]
                    dist = np.linalg.norm(diff)
                    if dist < min_sep and dist > 1e-15:
                        overlap = min_sep - dist
                        dir_ij = diff / dist
                        w_i = inv_m_drv[i]
                        w_j = inv_m_drv[j]
                        w_total = w_i + w_j
                        if w_total > 0:
                            correction = overlap * dir_ij
                            pos[i] -= (w_i / w_total) * correction
                            pos[j] += (w_j / w_total) * correction
            pos[drv] = target_pos

        vel_final = (pos - pos_prev) / dt
        vel_final[drv] = (target_pos - pos_prev[drv]) / dt
        self.position = pos
        self.velocity = vel_final

        grav_force = np.sum(self.mass[:, None] * self.gravity, axis=0)
        grip_reaction = self.mass[drv] * (vel_final[drv] - vel[drv]) / dt - self.mass[drv] * self.gravity
        ext_f = grav_force + grip_reaction
        ext_tau = np.sum(np.cross(pos, self.mass[:, None] * self.gravity), axis=0)
        ext_tau += np.cross(pos[drv], grip_reaction)
        return ext_f, ext_tau

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time:
            target = self.time
        remaining = target - self.time
        if remaining <= 0:
            return self._build_frame(target)
        steps = int(math.ceil(remaining / self.substep_dt))
        steps = min(steps, self.max_substeps)
        dt_used = remaining / steps
        ext_f_acc = np.zeros(3)
        ext_tau_acc = np.zeros(3)
        for _ in range(steps):
            ef, et = self._substep(dt_used)
            ext_f_acc += ef * dt_used
            ext_tau_acc += et * dt_used
            self.time += dt_used
        self.cum_ext_impulse += ext_f_acc
        self.cum_ext_ang_impulse += ext_tau_acc
        grav_total = np.sum(self.mass[:, None] * self.gravity, axis=0)
        self.last_ext_force[0, 0] = grav_total
        self.last_ext_pos[0, 0] = np.average(self.position, weights=self.mass, axis=0)
        self.last_ext_couple[0, 0] = 0.0
        return self._build_frame(target)

    def _build_frame(self, t: float) -> StateFrame:
        n = self.n
        pos = self.position.copy()
        vel = self.velocity.copy()
        mass = self.mass.copy()
        omega = np.zeros((n, 3), dtype=np.float64)
        inertia = np.zeros((n, 3, 3), dtype=np.float64)
        domain_weights = np.ones((1, n), dtype=np.float64)
        cum_mass_src = np.zeros(1, dtype=np.float64)
        fields = {
            "budget_cumulative_external_angular_impulse": self.cum_ext_ang_impulse.reshape(1, 3).copy(),
            "budget_cumulative_external_impulse": self.cum_ext_impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_angular_momentum": self.cum_in_amom.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": self.cum_in_mom.reshape(1, 3).copy(),
            "budget_cumulative_outflow_angular_momentum": self.cum_out_amom.reshape(1, 3).copy(),
            "budget_cumulative_outflow_momentum": self.cum_out_mom.reshape(1, 3).copy(),
            "budget_domain_weights": domain_weights,
            "budget_external_couple": self.last_ext_couple.copy(),
            "budget_external_force": self.last_ext_force.copy(),
            "budget_force_position": self.last_ext_pos.copy(),
            "cumulative_mass_source": cum_mass_src,
        }
        attributes = {
            "inertia_tensor": inertia,
            "mass_quantity": mass.copy(),
        }
        scalars = {
            "reference_angular_momentum": float(self.ref_ang_momentum),
            "reference_momentum": float(self.ref_momentum),
        }
        frame = StateFrame(
            time=float(t), position=pos, velocity=vel,
            orientation=None, angular_velocity=omega, mass=mass,
            attributes=attributes, fields=fields, scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {"solver": "xpbd_cable_power_friction", "current_time": float(self.time)}
