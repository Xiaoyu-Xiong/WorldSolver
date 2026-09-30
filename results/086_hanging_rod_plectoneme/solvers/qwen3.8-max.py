"""Discrete elastic rod solver for hanging-rod plectoneme formation."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Mass-spring-chain elastic rod with bending/twist stiffness, inextensibility,
    self-contact, gravity, ambient drag, and kinematic endpoint twist control."""

    def initialize(self, scene: Any, seed: int) -> None:
        cfg = scene.config if hasattr(scene, "config") else dict(scene.metadata())
        self.scene = scene
        self.cfg = cfg
        self.seed = seed

        rod = cfg["rod"]
        controls = cfg["end_controls"]
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        drag_cfg = cfg.get("ambient_drag", {}) or {}
        self.drag_enabled = bool(drag_cfg.get("enabled", False))
        self.drag_coeff = float(drag_cfg.get("coefficient", 0.0))

        self.n = int(rod["vertex_count"])
        self.total_length = float(rod["total_length"])
        self.radius = float(rod["radius"])
        self.total_mass = float(rod["mass"])
        self.rest_ds = self.total_length / (self.n - 1)

        # Material parameters (tuned for visible plectoneme under 8 turns over ~12s)
        self.EI = 2.5e-4       # bending stiffness N*m^2
        self.GJ = 1.5e-4       # torsional stiffness N*m^2
        self.EA = 5.0e3        # axial stiffness N (near-inextensible penalty)
        self.self_contact_k = 6.0  # radial contact spring stiffness N/m
        self.bending_damping = 4.0e-5
        self.stretch_damping = 0.05

        # State arrays
        init = scene.initial_state()
        self.pos = np.array(init.position, dtype=float).copy()
        self.vel = np.array(init.velocity, dtype=float).copy()
        self.mass_arr = np.array(init.mass, dtype=float).copy()

        # Compute initial tangent frame along planar curve
        self._recompute_rest_frame()

        # Twist state: material twist angle at each vertex relative to rest frame
        self.twist = np.zeros(self.n, dtype=float)
        self.twist_vel = np.zeros(self.n, dtype=float)

        # Track cumulative twist applied at endpoints (for constraint_target/value)
        self.applied_twist_left = 0.0
        self.applied_twist_right = 0.0

        # Endpoint indices
        self.iL = int(controls["controlled_vertices"][0])
        self.iR = int(controls["controlled_vertices"][1])

        # Solver timekeeping
        self.time = 0.0
        self.substeps = 12  # per sample interval

        # Budget accumulators (single domain G=1, K=1 force term)
        self.cum_ext_impulse = np.zeros(3, dtype=float)
        self.cum_ext_ang_impulse = np.zeros(3, dtype=float)
        self.cum_inflow_mom = np.zeros(3, dtype=float)
        self.cum_outflow_mom = np.zeros(3, dtype=float)
        self.cum_inflow_am = np.zeros(3, dtype=float)
        self.cum_outflow_am = np.zeros(3, dtype=float)

    # ------------------------------------------------------------------ helpers
    def _recompute_rest_frame(self):
        """Build parallel-transported reference frame d1,d2 along current centerline."""
        n = self.n
        t = np.zeros((n, 3), dtype=float)
        ds = np.zeros(n - 1, dtype=float)
        for i in range(n - 1):
            e = self.pos[i + 1] - self.pos[i]
            ds[i] = np.linalg.norm(e) + 1e-14
            t[i] = e / ds[i]
        t[-1] = t[-2]
        self.tangent = t
        self.seg_len = ds

        # Build initial d1 perpendicular to t[0]; choose world y unless degenerate
        d1 = np.zeros((n, 3), dtype=float)
        d2 = np.zeros((n, 3), dtype=float)
        ref = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(t[0], ref)) > 0.95:
            ref = np.array([0.0, 0.0, 1.0])
        d1[0] = np.cross(t[0], ref)
        d1[0] /= np.linalg.norm(d1[0]) + 1e-14
        d2[0] = np.cross(t[0], d1[0])
        # Parallel transport
        for i in range(n - 1):
            d1[i + 1], d2[i + 1] = self._parallel_transport(
                t[i], t[i + 1], d1[i], d2[i]
            )
        self.d1_ref = d1.copy()
        self.d2_ref = d2.copy()

    @staticmethod
    def _parallel_transport(t0, t1, d1_0, d2_0):
        v = np.cross(t0, t1)
        c = np.dot(t0, t1)
        if c < -0.999999:
            return -d1_0.copy(), -d2_0.copy()
        if c > 0.999999:
            return d1_0.copy(), d2_0.copy()
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        R = np.eye(3) + vx + vx @ vx / (1.0 + c)
        d1_new = R @ d1_0
        d1_new -= np.dot(d1_new, t1) * t1
        nrm = np.linalg.norm(d1_new)
        if nrm < 1e-12:
            ref = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(t1, ref)) > 0.95:
                ref = np.array([0.0, 0.0, 1.0])
            d1_new = np.cross(t1, ref)
            nrm = np.linalg.norm(d1_new)
        d1_new /= nrm
        d2_new = np.cross(t1, d1_new)
        return d1_new, d2_new

    def _twist_ramp_angle(self, time: float) -> float:
        ramp = self.cfg["end_controls"]["twist_ramp"]
        t0 = float(ramp["ramp_start"])
        t1 = float(ramp["ramp_end"])
        final_turns = float(ramp["final_relative_turns"])
        if time <= t0:
            u = 0.0
        elif time >= t1:
            u = 1.0
        else:
            u = (time - t0) / (t1 - t0)
            u = u * u * (3.0 - 2.0 * u)  # smoothstep
        return 2.0 * np.pi * final_turns * u

    def _endpoint_targets(self, time: float):
        targets = self.scene.end_control_targets(time)
        left_pos = np.asarray(targets["left"]["position"], dtype=float)
        right_pos = np.asarray(targets["right"]["position"], dtype=float)
        left_ang = float(targets["left"]["angle_radians"])
        right_ang = float(targets["right"]["angle_radians"])
        return left_pos, right_pos, left_ang, right_ang

    # -------------------------------------------------------------- physics step
    def _step(self, dt: float, time_now: float):
        n = self.n
        pos = self.pos
        vel = self.vel
        mass = self.mass_arr
        g = self.gravity
        drag_c = self.drag_coeff if self.drag_enabled else 0.0

        F = np.zeros_like(pos)

        # Gravity
        F += mass[:, None] * g[None, :]

        # Ambient drag: F_drag = -c * v
        if drag_c > 0:
            F -= drag_c * vel

        # Axial springs (near-inextensible)
        seg = pos[1:] - pos[:-1]
        seg_len = np.linalg.norm(seg, axis=1) + 1e-14
        t_hat = seg / seg_len[:, None]
        stretch = seg_len - self.rest_ds
        rel_v = (vel[1:] - vel[:-1]) * t_hat
        axial_f_mag = self.EA * stretch + self.EA * self.stretch_damping * rel_v.sum(axis=1)
        f_axial = axial_f_mag[:, None] * t_hat
        F[:-1] += f_axial
        F[1:] -= f_axial

        # Bending forces via discrete curvature gradient
        # E_bend = 0.5 * EI * sum |kappa|^2 * ds, kappa approx (t_{i+1}-t_i)/ds_avg
        if n >= 3:
            t_vec = np.diff(pos, axis=0)
            t_len = np.linalg.norm(t_vec, axis=1) + 1e-14
            t_hat_seg = t_vec / t_len[:, None]
            # Curvature vector at interior vertices
            for i in range(1, n - 1):
                ds_prev = t_len[i - 1]
                ds_next = t_len[i]
                ds_avg = 0.5 * (ds_prev + ds_next)
                kappa = (t_hat_seg[i] - t_hat_seg[i - 1]) / ds_avg
                # Gradient of 0.5*EI*|kappa|^2*ds_avg w.r.t positions
                # dE/dp_i ~ EI * kappa / ds_avg contributions
                grad = self.EI * kappa / ds_avg
                F[i] -= 2.0 * grad
                F[i - 1] += grad
                F[i + 1] += grad

            # Bending damping (velocity-proportional on curvature rate)
            vt = np.diff(vel, axis=0)
            for i in range(1, n - 1):
                ds_prev = t_len[i - 1]
                ds_next = t_len[i]
                ds_avg = 0.5 * (ds_prev + ds_next)
                kappa_rate = (vt[i] / t_len[i] - vt[i - 1] / t_len[i - 1]) / ds_avg
                damp = self.bending_damping * kappa_rate / ds_avg
                F[i] -= 2.0 * damp
                F[i - 1] += damp
                F[i + 1] += damp

        # Torsion: twist energy E_tw = 0.5 * GJ * sum (twist_grad)^2 * ds
        # twist_grad[i] = (twist[i+1]-twist[i])/ds
        if n >= 2:
            tw_diff = np.diff(self.twist)
            tw_grad = tw_diff / self.seg_len
            torque_mag = self.GJ * tw_grad  # internal torque on segment
            # Torque balance: vertex i gets +torque from left seg, -torque from right seg
            torque_vertex = np.zeros(n, dtype=float)
            torque_vertex[:-1] -= torque_mag
            torque_vertex[1:] += torque_mag
            # Convert torque about tangent into angular accel => we track twist DOF directly
            # I_twist * alpha = torque; treat unit inertia per vertex for twist DOF
            # Use simple explicit Euler on twist DOF
            I_tw = mass * (self.radius ** 2) * 0.5 + 1e-12
            self.twist_vel += dt * torque_vertex / I_tw
            self.twist_vel *= 0.995  # small numerical damping
            self.twist += dt * self.twist_vel

        # Self-contact (finite thickness): penalize pairs closer than 2*radius
        min_sep = 2.0 * self.radius
        skip = max(3, n // 40)  # skip nearby neighbors
        # Subsample for performance
        stride = max(1, n // 80)
        idx = np.arange(0, n, stride)
        for ii in idx:
            pi = pos[ii]
            diffs = pos - pi
            dist2 = np.sum(diffs ** 2, axis=1)
            mask = dist2 < (min_sep * 3.0) ** 2
            mask[:max(0, ii - skip)] = False
            mask[ii:min(n, ii + skip + 1)] = False
            js = np.where(mask)[0]
            if len(js) == 0:
                continue
            dvecs = diffs[js]
            dists = np.sqrt(dist2[js]) + 1e-14
            overlap = min_sep - dists
            active = overlap > 0
            if not np.any(active):
                continue
            normals = dvecs[active] / dists[active, None]
            forces = self.self_contact_k * overlap[active, None] * normals
            F[js[active]] += forces
            F[ii] -= forces.sum(axis=0)

        # Endpoint position constraints (kinematic)
        lp, rp, l_ang, r_ang = self._endpoint_targets(time_now)
        # Strong positional pin via velocity correction after integration
        # (handled post-step)

        # Semi-implicit Euler
        acc = F / mass[:, None]
        vel += dt * acc
        pos += dt * vel

        # Pin endpoints
        pos[self.iL] = lp
        pos[self.iR] = rp
        vel[self.iL] = 0.0
        vel[self.iR] = 0.0

        # Apply endpoint twist as Dirichlet on twist DOF
        # Left rotates about outward_axis (-x) by l_ang; this maps to twist sign
        # Right rotates about +x by r_ang
        # Total relative twist distributed: left gets -0.5*rel, right +0.5*rel
        self.twist[self.iL] = l_ang
        self.twist[self.iR] = r_ang
        self.twist_vel[self.iL] = 0.0
        self.twist_vel[self.iR] = 0.0

        # Accumulate external impulse (gravity + drag only; constraint forces are internal-to-domain)
        ext_force = mass[:, None] * g[None, :]
        if drag_c > 0:
            ext_force -= drag_c * vel
        self.cum_ext_impulse += dt * ext_force.sum(axis=0)
        origin = np.zeros(3)
        torque_about_origin = np.cross(pos, ext_force).sum(axis=0)
        self.cum_ext_ang_impulse += dt * torque_about_origin

    # --------------------------------------------------------------- advance API
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time:
            # Should not happen; just return current
            return self._make_frame(self.time)
        sample_dt = float(self.cfg["sample_dt"])
        sub_dt = sample_dt / max(1, self.substeps)
        while self.time < target - 1e-12:
            step = min(sub_dt, target - self.time)
            self._step(step, self.time + step)
            self.time += step
        return self._make_frame(self.time)

    def _make_frame(self, time: float) -> StateFrame:
        n = self.n
        pos = self.pos.copy()
        vel = self.vel.copy()
        mass = self.mass_arr.copy()

        # Orientation: quaternion aligned with tangent, twisted by material twist
        orient = np.zeros((n, 4), dtype=float)
        for i in range(n):
            t = self.tangent[i]
            # Rotate d1_ref by twist[i] about tangent
            ct = math.cos(self.twist[i])
            st = math.sin(self.twist[i])
            d1 = ct * self.d1_ref[i] + st * self.d2_ref[i]
            d2 = np.cross(t, d1)
            # Rotation matrix columns: d1, d2, t -> quaternion
            R = np.column_stack((d1, d2, t))
            q = self._rotmat_to_quat(R)
            orient[i] = q

        ang_vel = np.zeros((n, 3), dtype=float)
        # Approximate angular velocity from twist rate about tangent
        for i in range(n):
            ang_vel[i] = self.twist_vel[i] * self.tangent[i]

        # Inertia tensor per vertex (sphere approximation for point mass)
        I_diag = mass * self.radius * self.radius * 0.4  # solid sphere 2/5 m r^2 ≈ 0.4
        inertia = np.zeros((n, 3, 3), dtype=float)
        for i in range(n):
            inertia[i] = np.diag([I_diag[i], I_diag[i], I_diag[i]])

        # Budget fields (G=1 domain, K=1 force term)
        G, K = 1, 1
        budget_domain_weights = np.ones((G, n), dtype=float)
        budget_external_force = np.zeros((G, K, 3), dtype=float)
        budget_force_position = np.zeros((G, K, 3), dtype=float)
        budget_external_couple = np.zeros((G, K, 3), dtype=float)

        # Net external force at this instant: gravity + drag
        ext_f = mass[:, None] * self.gravity[None, :]
        if self.drag_enabled:
            ext_f -= self.drag_coeff * vel
        budget_external_force[0, 0] = ext_f.sum(axis=0)
        budget_force_position[0, 0] = (pos * mass[:, None]).sum(axis=0) / (mass.sum() + 1e-30)

        # Constraint: total relative twist vs target
        _, _, l_ang, r_ang = self._endpoint_targets(time)
        target_rel = r_ang - l_ang
        actual_rel = self.twist[self.iR] - self.twist[self.iL]
        constraint_value = np.array([actual_rel], dtype=float)
        constraint_target = np.array([target_rel], dtype=float)
        constraint_scale = max(abs(target_rel), 2.0 * np.pi)

        # Momentum scales
        mom = (mass[:, None] * vel).sum(axis=0)
        am = np.cross(pos, mass[:, None] * vel).sum(axis=0)
        ref_mom = max(float(np.linalg.norm(mom)), 1e-6)
        ref_am = max(float(np.linalg.norm(am)), 1e-9)

        fields = {
            "budget_cumulative_external_angular_impulse": self.cum_ext_ang_impulse.reshape(1, 3),
            "budget_cumulative_external_impulse": self.cum_ext_impulse.reshape(1, 3),
            "budget_cumulative_inflow_angular_momentum": self.cum_inflow_am.reshape(1, 3),
            "budget_cumulative_inflow_momentum": self.cum_inflow_mom.reshape(1, 3),
            "budget_cumulative_outflow_angular_momentum": self.cum_outflow_am.reshape(1, 3),
            "budget_cumulative_outflow_momentum": self.cum_outflow_mom.reshape(1, 3),
            "budget_domain_weights": budget_domain_weights,
            "budget_external_couple": budget_external_couple,
            "budget_external_force": budget_external_force,
            "budget_force_position": budget_force_position,
            "constraint_target": constraint_target,
            "constraint_value": constraint_value,
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }
        attributes = {
            "inertia_tensor": inertia,
            "mass_quantity": mass,
        }
        scalars = {
            "constraint_scale": float(constraint_scale),
            "reference_angular_momentum": float(ref_am),
            "reference_momentum": float(ref_mom),
        }

        frame = StateFrame(
            time=time,
            position=pos,
            velocity=vel,
            orientation=orient,
            angular_velocity=ang_vel,
            mass=mass,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    @staticmethod
    def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
        """Convert 3x3 rotation matrix to xyzw quaternion."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / math.sqrt(tr + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        q = np.array([x, y, z, w], dtype=float)
        q /= np.linalg.norm(q) + 1e-30
        return q

    def diagnostics(self) -> dict[str, object]:
        return {"time": self.time, "seed": self.seed}
