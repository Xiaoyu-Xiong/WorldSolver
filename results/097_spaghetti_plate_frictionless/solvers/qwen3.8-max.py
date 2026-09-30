"""Frictionless spaghetti-in-plate solver using Position-Based Dynamics (PBD)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """PBD-based strand simulation with zero friction."""

    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.config = scene.config
        self.gravity = np.asarray(self.config["gravity"], dtype=float)

        strand_cfg = self.config["strand"]
        self.strand_count = int(self.config["strand_count"])
        self.vertices_per_strand = int(strand_cfg["vertex_count"])
        self.strand_length = float(strand_cfg["length"])
        self.radius = float(strand_cfg["radius"])
        self.mass_per_strand = float(strand_cfg["total_mass_each"])
        self.node_mass = self.mass_per_strand / self.vertices_per_strand
        self.total_nodes = self.strand_count * self.vertices_per_strand

        n_edges = self.vertices_per_strand - 1
        self.rest_edge_len = self.strand_length / n_edges

        self.stretch_compliance = 1e-6
        self.bending_strength = 0.02  # fraction of angle corrected per iteration

        plate_cfg = self.config["plate"]
        geom = self.config.get("scene_geometry", {})
        self.plate_profile = None
        if geom:
            for obj in geom.get("objects", []):
                if obj.get("id") == "plate":
                    g = obj.get("geometry", {})
                    prof = g.get("inner_profile")
                    if prof:
                        self.plate_profile = np.asarray(prof, dtype=float)
        if self.plate_profile is None:
            self.plate_profile = np.array(
                [[0.10, 0.075], [0.88, 0.080], [1.05, 0.135], [1.14, 0.220]],
                dtype=float,
            )
        self.plate_wall_thickness = float(plate_cfg.get("wall_thickness", 0.055))

        frame0 = scene.initial_state()
        self.position = np.array(frame0.position, dtype=float).copy()
        self.velocity = np.array(frame0.velocity, dtype=float).copy()
        self.mass = np.array(frame0.mass, dtype=float).copy()
        self.body_id = np.array(frame0.body_id, dtype=int).copy()

        self.attributes_template: dict[str, np.ndarray] = {}
        for name, value in frame0.attributes.items():
            self.attributes_template[name] = np.asarray(value).copy()

        self.time = 0.0
        self.sim_dt = 0.002   # 2ms substeps
        self.max_substeps = 5000
        self.pbd_iterations = 2

        total_mass = float(np.sum(self.mass))
        g_mag = float(np.linalg.norm(self.gravity))
        if g_mag < 1e-12:
            g_mag = 9.81
        self.reference_force = max(total_mass * g_mag, 1e-6)
        self.reference_velocity = max(math.sqrt(g_mag * self.strand_length), 1e-3)
        self.reference_momentum = max(total_mass * self.reference_velocity, 1e-9)
        self.friction_stick_tol = 1e-4

        self.cum_ext_impulse = np.zeros(3, dtype=float)
        self._w = 1.0 / self.mass

        self._profile_radii = self.plate_profile[:, 0]
        self._profile_heights = self.plate_profile[:, 1]
        self._max_rim_r = float(self._profile_radii[-1])
        self._rim_z = float(self._profile_heights[-1])

        if len(self.plate_profile) >= 4:
            self._slope_r_inner = float(self._profile_radii[2])
            self._slope_z_inner = float(self._profile_heights[2])
            self._slope_r_outer = float(self._profile_radii[3])
            dr = self._slope_r_outer - self._slope_r_inner
            dz = float(self._profile_heights[3]) - self._slope_z_inner
            self._slope_dzdr = dz / dr if abs(dr) > 1e-12 else 0.0
        else:
            self._slope_dzdr = 0.0
            self._slope_r_inner = 0.0
            self._slope_r_outer = 0.0
            self._slope_z_inner = 0.0

        # Precompute edge index pairs (all strands at once)
        vps = self.vertices_per_strand
        all_ei = []
        all_ej = []
        for s in range(self.strand_count):
            base = s * vps
            ei = np.arange(base, base + vps - 1)
            ej = ei + 1
            all_ei.append(ei)
            all_ej.append(ej)
        self._all_ei = np.concatenate(all_ei)
        self._all_ej = np.concatenate(all_ej)

        # Bending quad indices (i0,i1,i2,i3) for each strand
        all_b0, all_b1, all_b2, all_b3 = [], [], [], []
        for s in range(self.strand_count):
            base = s * vps
            nb = vps - 3
            if nb > 0:
                b0 = np.arange(base, base + nb)
                all_b0.append(b0)
                all_b1.append(b0 + 1)
                all_b2.append(b0 + 2)
                all_b3.append(b0 + 3)
        if all_b0:
            self._b0 = np.concatenate(all_b0)
            self._b1 = np.concatenate(all_b1)
            self._b2 = np.concatenate(all_b2)
            self._b3 = np.concatenate(all_b3)
        else:
            self._b0 = np.array([], dtype=int)
            self._b1 = np.array([], dtype=int)
            self._b2 = np.array([], dtype=int)
            self._b3 = np.array([], dtype=int)

    def _project_stretch(self, pos):
        w = self._w
        rest = self.rest_edge_len
        alpha = self.stretch_compliance / (self.sim_dt * self.sim_dt)
        ei = self._all_ei
        ej = self._all_ej
        diff = pos[ej] - pos[ei]
        d = np.linalg.norm(diff, axis=1)
        d_safe = np.maximum(d, 1e-12)
        err = d - rest
        denom = w[ei] + w[ej] + alpha
        denom_safe = np.maximum(denom, 1e-20)
        dp = err / denom_safe
        n = diff / d_safe[:, None]
        corr = dp[:, None] * n
        np.add.at(pos, ei, w[ei][:, None] * corr)
        np.subtract.at(pos, ej, w[ej][:, None] * corr)

    def _project_bending(self, pos):
        """Vectorized bending projection."""
        if len(self._b0) == 0:
            return
        strength = self.bending_strength
        p0 = pos[self._b0]
        p1 = pos[self._b1]
        p2 = pos[self._b2]
        p3 = pos[self._b3]
        e1 = p1 - p0
        e2 = p2 - p1
        e3 = p3 - p2
        n1 = np.cross(e1, e2)
        n2 = np.cross(e2, e3)
        n1_norm = np.linalg.norm(n1, axis=1, keepdims=True)
        n2_norm = np.linalg.norm(n2, axis=1, keepdims=True)
        valid = (n1_norm.squeeze() > 1e-12) & (n2_norm.squeeze() > 1e-12)
        if not np.any(valid):
            return
        n1 = n1 / np.maximum(n1_norm, 1e-12)
        n2 = n2 / np.maximum(n2_norm, 1e-12)
        dots = np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0)
        phi = np.arccos(dots)
        mask = valid & (phi > 1e-10)
        if not np.any(mask):
            return
        idx1 = self._b1[mask]
        idx2 = self._b2[mask]
        phi_m = phi[mask]
        p0m = p0[mask]
        p1m = p1[mask]
        p2m = p2[mask]
        p3m = p3[mask]
        target1 = 0.5 * (p0m + p2m)
        target2 = 0.5 * (p1m + p3m)
        corr1 = (target1 - p1m) * np.minimum(phi_m * strength, 0.02)[:, None]
        corr2 = (target2 - p2m) * np.minimum(phi_m * strength, 0.02)[:, None]
        np.add.at(pos, idx1, corr1)
        np.add.at(pos, idx2, corr2)

    def _resolve_plate_contact(self, pos, vel):
        profile_radii = self._profile_radii
        profile_heights = self._profile_heights
        max_rim_r = self._max_rim_r
        rim_z = self._rim_z

        xy = pos[:, :2]
        rho = np.linalg.norm(xy, axis=1)
        z_vals = pos[:, 2]
        rho_safe = np.maximum(rho, 1e-12)

        all_normals = []
        all_fn = []
        all_relv = []

        floor_at_rho = np.interp(rho, profile_radii, profile_heights)
        pen_floor = floor_at_rho + self.radius - z_vals
        mask = pen_floor > 0.0
        if np.any(mask):
            pos[mask, 2] += pen_floor[mask]
            vz = vel[mask, 2]
            down = vz < 0.0
            if np.any(down):
                idx_down = np.where(mask)[0][down]
                vel[idx_down, 2] *= -0.1
            nf = np.zeros((int(mask.sum()), 3), dtype=float)
            nf[:, 2] = 1.0
            fn_approx = pen_floor[mask] * self.node_mass / max(self.sim_dt**2, 1e-12)
            all_normals.append(nf)
            all_fn.append(fn_approx)
            all_relv.append(vel[mask].copy())

        pen_wall = rho - max_rim_r + self.radius
        mask_w = (pen_wall > 0.0) & (z_vals < rim_z + self.radius)
        if np.any(mask_w):
            scale = (max_rim_r - self.radius) / rho_safe[mask_w]
            pos[mask_w, 0] *= scale
            pos[mask_w, 1] *= scale
            vr = vel[mask_w, 0] * (xy[mask_w, 0] / rho_safe[mask_w]) + \
                 vel[mask_w, 1] * (xy[mask_w, 1] / rho_safe[mask_w])
            outward = vr > 0.0
            if np.any(outward):
                idx_out = np.where(mask_w)[0][outward]
                er_x = xy[idx_out, 0] / rho_safe[idx_out]
                er_y = xy[idx_out, 1] / rho_safe[idx_out]
                vr_o = vr[outward]
                vel[idx_out, 0] -= 1.1 * vr_o * er_x
                vel[idx_out, 1] -= 1.1 * vr_o * er_y
            nw = np.zeros((int(mask_w.sum()), 3), dtype=float)
            nw[:, 0] = -xy[mask_w, 0] / rho_safe[mask_w]
            nw[:, 1] = -xy[mask_w, 1] / rho_safe[mask_w]
            fn_approx = pen_wall[mask_w] * self.node_mass / max(self.sim_dt**2, 1e-12)
            all_normals.append(nw)
            all_fn.append(fn_approx)
            all_relv.append(vel[mask_w].copy())

        if self._slope_dzdr != 0.0:
            r_in = self._slope_r_inner
            r_out = self._slope_r_outer
            z_in = self._slope_z_inner
            slope = self._slope_dzdr
            in_slope = (rho >= r_in) & (rho <= r_out)
            if np.any(in_slope):
                surf_z = z_in + slope * (rho[in_slope] - r_in)
                pen = surf_z + self.radius - z_vals[in_slope]
                mp = pen > 0.0
                if np.any(mp):
                    idx = np.where(in_slope)[0][mp]
                    pos[idx, 2] += pen[mp]
                    ns = np.zeros((len(idx), 3), dtype=float)
                    ns[:, 0] = -slope * xy[idx, 0] / rho_safe[idx]
                    ns[:, 1] = -slope * xy[idx, 1] / rho_safe[idx]
                    ns[:, 2] = 1.0
                    norms = np.linalg.norm(ns, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-12)
                    ns /= norms
                    fn_approx = pen[mp] * self.node_mass / max(self.sim_dt**2, 1e-12)
                    all_normals.append(ns)
                    all_fn.append(fn_approx)
                    all_relv.append(vel[idx].copy())

        if all_normals:
            c_normals = np.vstack(all_normals)
            c_fn = np.concatenate(all_fn)
            c_relv = np.vstack(all_relv)
        else:
            c_normals = np.empty((0, 3), dtype=float)
            c_fn = np.empty((0,), dtype=float)
            c_relv = np.empty((0, 3), dtype=float)

        return c_normals, c_fn, c_relv

    def _resolve_strand_strand(self, pos, vel):
        contact_radius = 2.0 * self.radius
        cr2 = contact_radius * contact_radius
        slab_h = 4.0 * self.radius
        z_vals = pos[:, 2]
        z_min = float(z_vals.min()) - 1.0
        slab_idx = np.floor((z_vals - z_min) / slab_h).astype(int)
        unique_slabs = np.unique(slab_idx)
        ss_normals = []
        ss_fn = []
        ss_relv = []
        body = self.body_id

        for slab in unique_slabs:
            idxs = np.where(slab_idx == slab)[0]
            m = len(idxs)
            if m < 2:
                continue
            if m > 200:
                idxs = idxs[::2]
                m = len(idxs)
            sub_pos = pos[idxs]
            sub_vel = vel[idxs]
            sub_body = body[idxs]
            diff = sub_pos[:, None, :] - sub_pos[None, :, :]
            d2 = np.sum(diff * diff, axis=2)
            same_or_adj = (sub_body[:, None] == sub_body[None, :]) & \
                          (np.abs(idxs[:, None] - idxs[None, :]) <= 2)
            close = (d2 < cr2) & (d2 > 1e-20) & (~same_or_adj)
            pairs = np.argwhere(close)
            if len(pairs) == 0:
                continue
            a = pairs[:, 0]
            b = pairs[:, 1]
            dvec = diff[a, b]
            dist = np.sqrt(d2[a, b])
            pen = contact_radius - dist
            nhat = dvec / dist[:, None]
            correction = 0.5 * pen[:, None] * nhat
            pos[idxs[a]] += correction
            pos[idxs[b]] -= correction
            rel_v = sub_vel[a] - sub_vel[b]
            vn = np.sum(rel_v * nhat, axis=1)
            approaching = vn < 0.0
            if np.any(approaching):
                ia = idxs[a[approaching]]
                ib = idxs[b[approaching]]
                n_appr = nhat[approaching]
                vn_appr = vn[approaching]
                vel[ia] -= 0.5 * vn_appr[:, None] * n_appr
                vel[ib] += 0.5 * vn_appr[:, None] * n_appr
            fn_approx = pen * self.node_mass / max(self.sim_dt**2, 1e-12)
            ss_normals.append(nhat)
            ss_fn.append(fn_approx)
            ss_relv.append(rel_v)

        if ss_normals:
            c_normals = np.vstack(ss_normals)
            c_fn = np.concatenate(ss_fn)
            c_relv = np.vstack(ss_relv)
        else:
            c_normals = np.empty((0, 3), dtype=float)
            c_fn = np.empty((0,), dtype=float)
            c_relv = np.empty((0, 3), dtype=float)
        return c_normals, c_fn, c_relv

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time:
            return self._build_frame(target)

        remaining = target - self.time
        if remaining <= 0.0:
            return self._build_frame(target)

        steps = max(1, int(math.ceil(remaining / self.sim_dt)))
        steps = min(steps, self.max_substeps)
        dt = remaining / steps

        last_contact = (
            np.empty((0, 3), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0, 3), dtype=float),
        )
        ext_impulse_step = np.zeros(3, dtype=float)

        for _ in range(steps):
            self.velocity += self.gravity[None, :] * dt
            prev_pos = self.position.copy()
            self.position += self.velocity * dt

            last_pn = np.empty((0, 3), dtype=float)
            last_pfn = np.empty((0,), dtype=float)
            last_prv = np.empty((0, 3), dtype=float)
            last_sn = np.empty((0, 3), dtype=float)
            last_sfn = np.empty((0,), dtype=float)
            last_srv = np.empty((0, 3), dtype=float)

            for _it in range(self.pbd_iterations):
                self._project_stretch(self.position)
                last_pn, last_pfn, last_prv = self._resolve_plate_contact(self.position, self.velocity)
                last_sn, last_sfn, last_srv = self._resolve_strand_strand(self.position, self.velocity)

            self._project_bending(self.position)
            self.velocity = (self.position - prev_pos) / dt
            self.time += dt

            normal_parts = [last_pn, last_sn]
            fn_parts = [last_pfn, last_sfn]
            rv_parts = [last_prv, last_srv]
            normals_list = [a for a in normal_parts if len(a) > 0]
            fn_list = [a for a in fn_parts if len(a) > 0]
            rv_list = [a for a in rv_parts if len(a) > 0]
            if normals_list:
                c_normals = np.vstack(normals_list)
                c_fn = np.concatenate(fn_list)
                c_relv = np.vstack(rv_list)
            else:
                c_normals = np.empty((0, 3), dtype=float)
                c_fn = np.empty((0,), dtype=float)
                c_relv = np.empty((0, 3), dtype=float)
            last_contact = (c_normals, c_fn, c_relv)

            grav_imp = np.sum(self.mass[:, None] * self.gravity[None, :], axis=0) * dt
            if len(c_fn) > 0:
                contact_imp = np.sum(c_fn[:, None] * c_normals, axis=0) * dt
            else:
                contact_imp = np.zeros(3, dtype=float)
            ext_impulse_step += grav_imp + contact_imp

        self.cum_ext_impulse += ext_impulse_step
        return self._build_frame(target, last_contact)

    def _build_frame(self, time, contact_info=None):
        n = len(self.position)
        attrs = {k: v.copy() for k, v in self.attributes_template.items()}
        attrs["mass_quantity"] = self.mass.copy()

        if contact_info is None:
            c_normals = np.empty((0, 3), dtype=float)
            c_fn = np.empty((0,), dtype=float)
            c_relv = np.empty((0, 3), dtype=float)
        else:
            c_normals, c_fn, c_relv = contact_info

        nc = len(c_fn)
        fields = {
            "budget_cumulative_external_impulse": self.cum_ext_impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_domain_weights": np.ones((1, n), dtype=float),
            "budget_external_force": np.zeros((1, 1, 3), dtype=float),
            "cumulative_mass_source": np.zeros((1,), dtype=float),
            "friction_active_mask": np.zeros(nc, dtype=float),
            "friction_dynamic_coefficient": np.zeros(nc, dtype=float),
            "friction_normal": c_normals.astype(float, copy=True) if nc else np.empty((0, 3), dtype=float),
            "friction_normal_force": c_fn.astype(float, copy=True) if nc else np.empty((0,), dtype=float),
            "friction_relative_velocity": c_relv.astype(float, copy=True) if nc else np.empty((0, 3), dtype=float),
            "friction_static_coefficient": np.zeros(nc, dtype=float),
            "friction_tangential_force": np.zeros((nc, 3), dtype=float),
        }

        scalars = {
            "friction_stick_speed_tolerance": self.friction_stick_tol,
            "reference_force": self.reference_force,
            "reference_momentum": self.reference_momentum,
            "reference_velocity": self.reference_velocity,
        }

        return StateFrame(
            time=time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self):
        return {
            "solver": "pbd_discrete_rod_frictionless",
            "sim_dt": self.sim_dt,
            "pbd_iterations": self.pbd_iterations,
        }
