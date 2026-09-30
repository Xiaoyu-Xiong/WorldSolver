"""Two-way coupled liquid-filled deformable toy bounce solver.

PBD-based simulation with optimized vectorized constraints:
- Shell: distance constraints + global shape matching
- Liquid: pairwise repulsion + ellipsoidal confinement inside shell
- Markers: passive tracers advected by interpolated liquid velocity
- Floor: hard constraint with restitution
- Two-way coupling via shared PBD iterations
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        cfg = scene.config
        self.gravity = np.array(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])

        init = scene.initial_state()
        self.pos = np.array(init.position, dtype=float).copy()
        self.vel = np.array(init.velocity, dtype=float).copy()
        self.mass = np.array(init.mass, dtype=float).copy()
        self.body_id = np.array(init.body_id, dtype=np.int32).copy()

        self.n_total = len(self.pos)
        self.shell_mask = self.body_id == 1
        self.liquid_mask = self.body_id == 2
        self.marker_mask = np.isin(self.body_id, (3, 4))

        self.shell_idx = np.where(self.shell_mask)[0]
        self.liq_idx = np.where(self.liquid_mask)[0]
        self.marker_idx = np.where(self.marker_mask)[0]

        self.n_shell = len(self.shell_idx)
        self.n_liquid = len(self.liq_idx)

        # Shell rest configuration
        self.shell_rest = self.pos[self.shell_idx].copy()
        self.shell_com_init = self.shell_rest.mean(axis=0)
        self.shell_rest_centered = self.shell_rest - self.shell_com_init

        # Build shell springs
        self._build_springs()

        # Precompute spring global indices
        if len(self.spring_pairs) > 0:
            self.spring_gi = self.shell_idx[self.spring_pairs[:, 0]]
            self.spring_gj = self.shell_idx[self.spring_pairs[:, 1]]
        else:
            self.spring_gi = np.empty(0, dtype=np.int32)
            self.spring_gj = np.empty(0, dtype=np.int32)

        # Inverse masses
        self.inv_mass = np.zeros(self.n_total, dtype=float)
        has_mass = self.mass > 1e-12
        self.inv_mass[has_mass] = 1.0 / self.mass[has_mass]

        # PBD parameters
        self.pbd_iterations = 6
        self.substeps = 10
        self.dt = self.sample_dt / self.substeps

        # Shell
        self.spring_compliance = 0.0003
        self.shape_stiffness = 0.15

        # Liquid
        self.liq_rest_spacing = 0.105
        self.liq_interaction_radius = 0.14
        self.liq_compliance = 0.0001

        # Floor
        self.floor_z = 0.0
        self.restitution = 0.4
        self.friction = 0.2

        # Reference scales
        total_mass = self.mass[~self.marker_mask].sum()
        self.ref_velocity = 3.0
        self.ref_momentum = total_mass * self.ref_velocity

        self.current_time = 0.0
        self._last_frame = self._make_frame(0.0)

    def _build_springs(self):
        shell_pos = self.shell_rest
        n = len(shell_pos)
        pairs = []
        rest_lens = []
        for i in range(n):
            dists = np.linalg.norm(shell_pos - shell_pos[i], axis=1)
            dists[i] = np.inf
            nn = np.argsort(dists)[:6]
            for j in nn:
                if j > i:
                    pairs.append((i, j))
                    rest_lens.append(float(dists[j]))
        self.spring_pairs = np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)
        self.spring_rest = np.array(rest_lens, dtype=float) if rest_lens else np.empty((0,), dtype=float)

    def _solve_liquid_constraints(self, pred):
        """Vectorized pairwise repulsion for liquid particles."""
        liq_pred = pred[self.liq_idx]
        h = self.liq_interaction_radius
        h2 = h * h
        rest_d = self.liq_rest_spacing
        n = self.n_liquid
        
        if n == 0:
            return

        batch = 120
        for ls in range(0, n, batch):
            le = min(ls + batch, n)
            ni = le - ls
            
            for js in range(0, n, batch):
                je = min(js + batch, n)
                nj = je - js
                
                dx = liq_pred[ls:le, np.newaxis, :] - liq_pred[np.newaxis, js:je, :]
                r2 = np.sum(dx * dx, axis=2)
                
                close = (r2 < h2) & (r2 > 1e-10)
                if not close.any():
                    continue
                
                r = np.sqrt(np.where(close, r2, 1.0))
                overlap = np.maximum(0.0, rest_d - r) * close
                
                if overlap.max() < 1e-10:
                    continue
                
                normal = dx / (r[:, :, np.newaxis] + 1e-12)
                
                # Vectorized correction computation
                wi_batch = self.inv_mass[self.liq_idx[ls:le]]  # (ni,)
                wj_batch = self.inv_mass[self.liq_idx[js:je]]  # (nj,)
                
                # For each close pair (i,j): s = overlap / (wi + wj + compliance)
                denom = wi_batch[:, np.newaxis] + wj_batch[np.newaxis, :] + self.liq_compliance
                s = overlap / denom  # (ni, nj)
                
                # Correction for i particles: sum_j(s * normal * wj)
                # Correction for j particles: -sum_i(s * normal * wi) 
                f_vec = s[:, :, np.newaxis] * normal  # (ni, nj, 3)
                
                # Accumulate corrections for i
                corr_i = np.sum(f_vec * wj_batch[np.newaxis, :, np.newaxis], axis=1)  # (ni, 3)
                mask_i = close.any(axis=1)
                if mask_i.any():
                    active_i = np.where(mask_i)[0]
                    gi_global = self.liq_idx[ls + active_i]
                    pred[gi_global] += corr_i[active_i]
                
                # Accumulate corrections for j  
                corr_j = -np.sum(f_vec * wi_batch[:, np.newaxis, np.newaxis], axis=0)  # (nj, 3)
                mask_j = close.any(axis=0)
                if mask_j.any():
                    active_j = np.where(mask_j)[0]
                    gj_global = self.liq_idx[js + active_j]
                    pred[gj_global] += corr_j[active_j]

    def _step(self, dt):
        pos = self.pos.copy()
        vel = self.vel.copy()
        inv_mass = self.inv_mass

        # Apply gravity
        has_mass = self.mass > 1e-12
        vel[has_mass] += self.gravity * dt

        # Predict positions
        pred = pos + vel * dt

        # PBD iterations
        for iteration in range(self.pbd_iterations):
            # --- Shell spring constraints (fully vectorized) ---
            if len(self.spring_gi) > 0:
                gi = self.spring_gi
                gj = self.spring_gj
                
                diff = pred[gj] - pred[gi]
                dist = np.linalg.norm(diff, axis=1, keepdims=True) + 1e-12
                direction = diff / dist
                error = dist.flatten() - self.spring_rest

                w_i = inv_mass[gi]
                w_j = inv_mass[gj]
                denom = w_i + w_j + self.spring_compliance
                s = error / denom

                dp = direction * s[:, np.newaxis]
                pred[gi] += dp * w_i[:, np.newaxis]
                pred[gj] -= dp * w_j[:, np.newaxis]

            # --- Shell shape matching ---
            shell_pred = pred[self.shell_idx]
            shell_com = shell_pred.mean(axis=0)
            target = shell_com + self.shell_rest_centered
            disp = target - shell_pred
            pred[self.shell_idx] += disp * self.shape_stiffness

            # --- Liquid pairwise repulsion (vectorized) ---
            self._solve_liquid_constraints(pred)

            # --- Liquid confinement inside shell ellipsoid ---
            if self.n_liquid > 0:
                shell_pred_now = pred[self.shell_idx]
                shell_com_now = shell_pred_now.mean(axis=0)
                
                centered_shell = shell_pred_now - shell_com_now
                current_extents = np.array([
                    max(np.abs(centered_shell[:, 0]).max(), 0.1),
                    max(np.abs(centered_shell[:, 1]).max(), 0.1),
                    max(np.abs(centered_shell[:, 2]).max(), 0.1),
                ])
                
                liq_rel = pred[self.liq_idx] - shell_com_now
                norm_pos = liq_rel / current_extents
                ellipsoid_r = np.linalg.norm(norm_pos, axis=1)
                
                margin = 0.90
                outside = ellipsoid_r > margin
                if outside.any():
                    out_idx = self.liq_idx[outside]
                    out_rel = liq_rel[outside]
                    out_r = ellipsoid_r[outside]
                    
                    push_dir = -out_rel / (np.linalg.norm(out_rel, axis=1, keepdims=True) + 1e-12)
                    target_r = margin * 0.9
                    push_dist = ((out_r - target_r) * np.mean(current_extents))[:, np.newaxis]
                    pred[out_idx] += push_dir * push_dist * 0.5

            # --- Floor constraint ---
            pen = self.floor_z - pred[:, 2]
            touching = pen > 0.0
            if touching.any():
                pred[touching, 2] = self.floor_z

        # Update velocity
        new_vel = (pred - pos) / dt

        # Floor bounce
        at_floor = np.abs(pred[:, 2] - self.floor_z) < 1e-6
        if at_floor.any():
            fidx = np.where(at_floor)[0]
            old_vz = vel[fidx, 2]
            incoming = old_vz < 0.0
            if incoming.any():
                inc_idx = fidx[incoming]
                new_vel[inc_idx, 2] = -old_vz[incoming] * self.restitution
                vt = new_vel[inc_idx, :2].copy()
                vt_mag = np.linalg.norm(vt, axis=1) + 1e-12
                fric = np.minimum(self.friction * np.abs(new_vel[inc_idx, 2]) / vt_mag, 1.0)
                new_vel[inc_idx, :2] *= (1.0 - fric)[:, np.newaxis]

        # Advect markers
        if len(self.marker_idx) > 0 and self.n_liquid > 0:
            liq_vel_new = new_vel[self.liq_idx]
            liq_pos_new = pred[self.liq_idx]
            for mi in self.marker_idx:
                diffs = liq_pos_new - pred[mi]
                dsq = np.sum(diffs**2, axis=1)
                weights = np.exp(-dsq / 0.01)
                w_sum = weights.sum()
                if w_sum > 1e-12:
                    interp_vel = np.sum(weights[:, np.newaxis] * liq_vel_new, axis=0) / w_sum
                    pred[mi] = pred[mi] + interp_vel * dt
                    new_vel[mi] = interp_vel

        self.pos = pred
        self.vel = new_vel

    def advance_to(self, time: float):
        if hasattr(self, '_last_frame') and abs(time - self._last_frame.time) < 1e-12:
            return self._last_frame

        while self.current_time < time - 1e-12:
            remaining = time - self.current_time
            step_dt = min(self.dt, remaining)
            if step_dt < 1e-15:
                break
            self._step(step_dt)
            self.current_time += step_dt

        frame = self._make_frame(time)
        self._last_frame = frame
        return frame

    def _make_frame(self, time: float) -> StateFrame:
        pos = self.pos.copy()
        vel = self.vel.copy()
        mass = self.mass.copy()
        body_id = self.body_id.copy()
        n = len(pos)

        boundary_normal = np.array([[0.0, 0.0, 1.0]])
        boundary_velocity = np.array([[0.0, 0.0, 0.0]])
        boundary_target_velocity = np.array([[0.0, 0.0, 0.0]])

        domain_weights = np.zeros((2, n), dtype=float)
        domain_weights[0, self.shell_mask] = 1.0
        domain_weights[1, self.liquid_mask] = 1.0

        ext_force = np.zeros((2, 2, 3), dtype=float)
        shell_mass_total = mass[self.shell_mask].sum()
        liq_mass_total = mass[self.liquid_mask].sum()
        ext_force[0, 0] = shell_mass_total * self.gravity
        ext_force[1, 1] = liq_mass_total * self.gravity

        cum_ext_impulse = np.zeros((2, 3), dtype=float)
        cum_ext_impulse[0] = ext_force[0, 0] * time
        cum_ext_impulse[1] = ext_force[1, 1] * time

        cum_inflow = np.zeros((2, 3), dtype=float)
        cum_outflow = np.zeros((2, 3), dtype=float)
        interaction_force = np.zeros((2, 3), dtype=float)
        cum_mass_source = np.zeros(2, dtype=float)

        fields = {
            "boundary_normal": boundary_normal,
            "boundary_target_velocity": boundary_target_velocity,
            "boundary_velocity": boundary_velocity,
            "budget_cumulative_external_impulse": cum_ext_impulse,
            "budget_cumulative_inflow_momentum": cum_inflow,
            "budget_cumulative_outflow_momentum": cum_outflow,
            "budget_domain_weights": domain_weights,
            "budget_external_force": ext_force,
            "cumulative_mass_source": cum_mass_source,
            "interaction_force": interaction_force,
        }

        scalars = {
            "reference_momentum": float(self.ref_momentum),
            "reference_velocity": float(self.ref_velocity),
        }

        attributes = {
            "mass_quantity": mass.copy(),
        }

        return StateFrame(
            time=time,
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "n_shell": self.n_shell,
            "n_liquid": self.n_liquid,
        }
