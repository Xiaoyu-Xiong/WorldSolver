"""Two-way coupled deformable bowl impact solver.

Mass-spring shell for the bowl + simplified repulsive-pressure particle fluid.
Uses position-based dynamics (PBD) for the bowl to ensure unconditional stability,
and simple repulsive particles for the fluid.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        cfg = scene.config
        self.gravity = np.array(cfg["gravity"], dtype=float)
        self.dt = float(cfg["sample_dt"])

        bc = cfg.get("boundary_conditions", {}).get("tank", {})
        self.tank_min = np.array(bc.get("interior_min", [-2.2, -1.5, 0.0]), dtype=float)
        self.tank_max = np.array(bc.get("interior_max", [2.2, 1.5, 2.8]), dtype=float)

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=float).copy()
        self.vel = np.array(frame0.velocity, dtype=float).copy()
        self.mass = np.array(frame0.mass, dtype=float).copy()
        self.body_id = np.array(frame0.body_id, dtype=np.int32).copy()
        self.N = len(self.pos)

        self.bowl_mask = self.body_id == 1
        self.liquid_mask = self.body_id == 2
        self.water_mask = self.body_id == 3
        self.nb = int(self.bowl_mask.sum())
        self.nl = int(self.liquid_mask.sum())
        self.nw = int(self.water_mask.sum())

        self.bowl_faces = np.asarray(scene.faces, dtype=np.int32)
        ref_pos = self.pos[self.bowl_mask].copy()
        self._ref_bowl_pos = ref_pos.copy()
        self._compute_face_geom(ref_pos)

        bowl_cfg = cfg.get("initial_conditions", {}).get("bowl", {})
        liq_cfg = cfg.get("initial_conditions", {}).get("liquid", {})
        wat_cfg = cfg.get("initial_conditions", {}).get("shallow_water", {})

        self.thickness = float(bowl_cfg.get("thickness", 0.025))
        self.E_shell = 5e5
        self.bowl_damping_ratio = 0.1  # Fraction of critical damping

        self.rho_liq = float(liq_cfg.get("rest_density", 1000.0))
        self.h_liq = float(liq_cfg.get("particle_spacing", 0.11))
        self.rho_wat = float(wat_cfg.get("rest_density", 1000.0))
        self.h_wat = float(wat_cfg.get("particle_spacing", 0.13))

        self.h_int = max(self.h_liq, self.h_wat) * 1.3

        self._build_shell_springs(ref_pos)
        self._build_boundary_samples()

        # Fluid pressure calibration
        m_wat = 1000.0 * self.h_wat ** 3
        n_nbrs_est = 20.0
        self.k_pressure = 9.81 / (n_nbrs_est * m_wat) * 1.5
        self.k_visc = 0.05

        # Budget accumulators
        self.cum_ext_impulse = np.zeros((3, 3), dtype=float)
        self.cum_inflow_mom = np.zeros((3, 3), dtype=float)
        self.cum_outflow_mom = np.zeros((3, 3), dtype=float)
        self.cum_mass_source = np.zeros(3, dtype=float)
        self.interaction_force = np.zeros((3, 3), dtype=float)

        self.domain_weights = np.zeros((3, self.N), dtype=float)
        self.domain_weights[0, :] = 1.0
        self.domain_weights[1, self.bowl_mask | self.liquid_mask] = 1.0
        self.domain_weights[2, self.water_mask] = 1.0

        total_mass = float(self.mass.sum())
        v_scale = max(float(np.linalg.norm(self.vel, axis=1).max()), 1.0)
        self.ref_momentum = total_mass * v_scale
        self.ref_velocity = v_scale

        self.current_time = 0.0
        self.max_vel = 50.0

    def _compute_face_geom(self, bowl_pos):
        f = self.bowl_faces
        v0 = bowl_pos[f[:, 0]]; v1 = bowl_pos[f[:, 1]]; v2 = bowl_pos[f[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(cross, axis=1, keepdims=True) + 1e-30
        self.face_normals = cross / norms
        self.face_areas = 0.5 * norms.ravel()

    def _build_shell_springs(self, ref_pos):
        f = self.bowl_faces
        edges = set()
        for tri in f:
            for i in range(3):
                a, b = int(tri[i]), int(tri[(i + 1) % 3])
                edges.add((min(a, b), max(a, b)))
        self.spring_edges = np.array(sorted(edges), dtype=np.int32)
        d = ref_pos[self.spring_edges[:, 1]] - ref_pos[self.spring_edges[:, 0]]
        self.spring_rest = np.linalg.norm(d, axis=1)
        mean_e = float(self.spring_rest.mean()) + 1e-12
        self.spring_k = self.E_shell * self.thickness * mean_e / (self.spring_rest + 1e-30)
        # Compliance for PBD: alpha = 1/(k * dt^2) but we use iterative constraint projection
        self.spring_compliance = 1.0 / (self.spring_k + 1e-30)

    def _build_boundary_samples(self):
        xs = np.linspace(self.tank_min[0], self.tank_max[0], 10)
        ys = np.linspace(self.tank_min[1], self.tank_max[1], 6)
        floor_pts = np.array([[x, y, 0.0] for x in xs for y in ys])
        floor_n = np.tile([0., 0., 1.], (len(floor_pts), 1))
        zs = np.linspace(0., self.tank_max[2], 8)
        back_pts = np.array([[x, self.tank_max[1], z] for x in xs for z in zs])
        back_n = np.tile([0., -1., 0.], (len(back_pts), 1))
        front_pts = np.array([[x, self.tank_min[1], z] for x in xs for z in zs])
        front_n = np.tile([0., 1., 0.], (len(front_pts), 1))
        sy = np.linspace(self.tank_min[1], self.tank_max[1], 5)
        sz = np.linspace(0., self.tank_max[2], 6)
        left_pts = np.array([[self.tank_min[0], y, z] for y in sy for z in sz])
        left_n = np.tile([1., 0., 0.], (len(left_pts), 1))
        right_pts = np.array([[self.tank_max[0], y, z] for y in sy for z in sz])
        right_n = np.tile([-1., 0., 0.], (len(right_pts), 1))
        self.bnd_pos = np.vstack([floor_pts, back_pts, front_pts, left_pts, right_pts])
        self.bnd_normal = np.vstack([floor_n, back_n, front_n, left_n, right_n])
        self.B = len(self.bnd_pos)

    def _build_grid(self, pos, cell_size):
        if len(pos) == 0:
            return {}, cell_size, np.zeros(3), np.ones(3, dtype=int)
        mins = pos.min(axis=0) - cell_size
        keys = np.floor((pos - mins) / cell_size).astype(int)
        dims = keys.max(axis=0) + 2
        encoded = keys[:, 0] * dims[1] * dims[2] + keys[:, 1] * dims[2] + keys[:, 2]
        grid = {}
        for idx in range(len(encoded)):
            k = int(encoded[idx])
            if k not in grid:
                grid[k] = []
            grid[k].append(idx)
        return grid, cell_size, mins, dims

    def _get_neighbors(self, grid, cs, mins, dims, pos_i):
        ci = np.floor((pos_i - mins) / cs).astype(int)
        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    cx, cy, cz = ci[0]+dx, ci[1]+dy, ci[2]+dz
                    if 0 <= cx < dims[0] and 0 <= cy < dims[1] and 0 <= cz < dims[2]:
                        k = cx * dims[1] * dims[2] + cy * dims[2] + cz
                        if k in grid:
                            neighbors.extend(grid[k])
        return neighbors

    def advance_to(self, time):
        target = float(time)
        dt = self.dt
        sub_dt = min(dt, 0.002)
        while self.current_time < target - 1e-12:
            step = min(sub_dt, target - self.current_time)
            self._substep(step)
            self.current_time += step
        return self._make_frame(target)

    def _pbd_constrain_bowl(self, dt):
        """Position-Based Dynamics constraint projection for bowl springs."""
        bowl_idx = np.where(self.bowl_mask)[0]
        bp = self.pos[bowl_idx]
        bm = self.mass[bowl_idx]
        w = 1.0 / (bm + 1e-30)  # inverse mass

        edges = self.spring_edges
        rest = self.spring_rest
        compliance = self.spring_compliance / (dt * dt + 1e-30)  # scaled compliance

        n_iters = 4
        for _ in range(n_iters):
            d = bp[edges[:, 1]] - bp[edges[:, 0]]
            L = np.linalg.norm(d, axis=1) + 1e-30
            C = L - rest  # constraint violation
            n_hat = d / L[:, None]

            # PBD correction: delta_p = -C / (w_i + w_j + alpha) * n
            w_sum = w[edges[:, 0]] + w[edges[:, 1]] + compliance
            dp_mag = -C / (w_sum + 1e-30)

            dp0 = dp_mag[:, None] * n_hat * w[edges[:, 0], None] / (w_sum[:, None] + 1e-30)
            dp1 = -dp_mag[:, None] * n_hat * w[edges[:, 1], None] / (w_sum[:, None] + 1e-30)

            np.add.at(bp, edges[:, 0], dp0)
            np.add.at(bp, edges[:, 1], dp1)

        # Shape matching toward reference (soft constraint) - preserve shape not position
        ref = self._ref_bowl_pos
        ref_com = ref.mean(axis=0)
        cur_com = bp.mean(axis=0)
        # Shift reference to current COM so we only restore shape, not position
        ref_shifted = ref - ref_com + cur_com
        shape_stiffness = 0.02  # Gentle restoration
        bp += shape_stiffness * (ref_shifted - bp)

        self.pos[bowl_idx] = bp

    def _substep(self, dt):
        N = self.N
        acc = np.zeros((N, 3), dtype=float)
        acc[:, 2] = self.gravity[2]

        fluid_mask = self.liquid_mask | self.water_mask
        fluid_idx = np.where(fluid_mask)[0]
        n_f = len(fluid_idx)

        if n_f > 0:
            pos_f = self.pos[fluid_idx]
            vel_f = self.vel[fluid_idx]
            mass_f = self.mass[fluid_idx]

            h = self.h_int
            grid, cs, mins, dims = self._build_grid(pos_f, h)

            k_p = self.k_pressure
            k_v = self.k_visc

            f_fluid = np.zeros((n_f, 3), dtype=float)
            for i in range(n_f):
                nbrs = self._get_neighbors(grid, cs, mins, dims, pos_f[i])
                nbrs_arr = np.array([j for j in nbrs if j != i])
                if len(nbrs_arr) == 0:
                    continue
                diff = pos_f[i] - pos_f[nbrs_arr]
                r_sq = np.sum(diff * diff, axis=1)
                r = np.sqrt(r_sq) + 1e-12
                q = r / h
                active = q < 1.0
                if not np.any(active):
                    continue
                qa = q[active]
                ra = r[active]
                diff_a = diff[active]
                r_hat = diff_a / ra[:, None]

                overlap = 1.0 - qa
                f_mag = k_p * overlap * overlap * mass_f[i] * mass_f[nbrs_arr[active]]
                f_fluid[i] += np.sum(f_mag[:, None] * r_hat, axis=0)

                dv = vel_f[nbrs_arr[active]] - vel_f[i]
                f_visc = k_v * overlap[:, None] * dv * mass_f[nbrs_arr[active], None]
                f_fluid[i] += np.sum(f_visc, axis=0)

            acc[fluid_idx] += f_fluid / mass_f[:, None]

        # Fluid-bowl coupling
        bowl_idx = np.where(self.bowl_mask)[0]
        if self.nb > 0 and n_f > 0:
            bp = self.pos[bowl_idx]
            fp = self.pos[fluid_idx]
            fm = self.mass[fluid_idx]
            h_coup = self.h_int * 0.8

            f_on_fluid = np.zeros((n_f, 3), dtype=float)
            f_on_bowl_acc = np.zeros((self.nb, 3), dtype=float)

            bgrid, bcs, bmins, bdims = self._build_grid(bp, h_coup)

            k_pen = self.k_pressure * 2.0
            for i in range(n_f):
                nbrs = self._get_neighbors(bgrid, bcs, bmins, bdims, fp[i])
                if not nbrs:
                    continue
                nbrs = np.array(nbrs)
                diff = fp[i] - bp[nbrs]
                r = np.sqrt(np.sum(diff * diff, axis=1)) + 1e-12
                q = r / h_coup
                active = q < 1.0
                if not np.any(active):
                    continue
                overlap = 1.0 - q[active]
                r_hat = -diff[active] / r[active, None]
                f_rep = k_pen * overlap * overlap * fm[i]
                f_vec = f_rep[:, None] * r_hat
                f_on_fluid[i] += np.sum(f_vec, axis=0)
                n_active = np.sum(active)
                for jj, j_local in enumerate(np.where(active)[0]):
                    j = nbrs[j_local]
                    f_on_bowl_acc[j] -= f_vec[jj] / n_active

            acc[fluid_idx] += f_on_fluid / fm[:, None]
            bm = self.mass[bowl_idx]
            acc[bowl_idx] += f_on_bowl_acc / bm[:, None]

            self.interaction_force[0] = np.sum(f_on_fluid * fm[:, None], axis=0)
            self.interaction_force[1] = -self.interaction_force[0]

        # Semi-implicit Euler for all particles
        new_vel = self.vel + acc * dt

        # Velocity cap
        v_norm = np.linalg.norm(new_vel, axis=1, keepdims=True) + 1e-30
        cap = v_norm > self.max_vel
        if np.any(cap):
            new_vel[cap.ravel()] *= self.max_vel / v_norm[cap.ravel()]

        new_pos = self.pos + new_vel * dt

        # Apply PBD constraints for bowl
        self.pos = new_pos
        self.vel = new_vel
        self._pbd_constrain_bowl(dt)

        # Update bowl velocity from position change (for PBD)
        # Actually recompute vel from constrained positions
        # For bowl: v = (x_new - x_old) / dt approximately
        # But we already updated pos, so just damp bowl velocity
        bi = np.where(self.bowl_mask)[0]
        self.vel[bi] *= (1.0 - self.bowl_damping_ratio)

        # Tank boundaries
        below = self.pos[:, 2] < 0.0
        self.pos[below, 2] = 0.0
        self.vel[below, 2] = np.abs(self.vel[below, 2]) * 0.2

        for dim in range(2):
            lo, hi = self.tank_min[dim], self.tank_max[dim]
            under = self.pos[:, dim] < lo
            over = self.pos[:, dim] > hi
            self.pos[under, dim] = lo
            self.vel[under, dim] = np.abs(self.vel[under, dim]) * 0.2
            self.pos[over, dim] = hi
            self.vel[over, dim] = -np.abs(self.vel[over, dim]) * 0.2

        top_cap = self.pos[:, 2] > self.tank_max[2]
        self.pos[top_cap, 2] = self.tank_max[2]
        self.vel[top_cap, 2] *= 0.1

        if self.nb > 0:
            self._compute_face_geom(self.pos[self.bowl_mask])

        ext_imp = self.mass[:, None] * self.gravity * dt
        for g in range(3):
            w = self.domain_weights[g]
            self.cum_ext_impulse[g] += np.sum(ext_imp * w[:, None], axis=0)

    def _make_frame(self, time):
        scalars = {
            "reference_momentum": float(self.ref_momentum),
            "reference_velocity": float(self.ref_velocity),
        }
        fields = {
            "boundary_normal": self.bnd_normal.copy(),
            "boundary_target_velocity": np.zeros_like(self.bnd_normal),
            "boundary_velocity": np.zeros_like(self.bnd_normal),
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow_mom.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow_mom.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "interaction_force": self.interaction_force.copy(),
        }
        ext_force = np.zeros((3, 1, 3), dtype=float)
        for g in range(3):
            w = self.domain_weights[g]
            ext_force[g, 0] = np.sum(self.mass[:, None] * self.gravity * w[:, None], axis=0)
        fields["budget_external_force"] = ext_force

        return StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"mass_quantity": self.mass.copy()},
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self):
        return {"N": self.N, "nb": self.nb, "nl": self.nl, "nw": self.nw, "t": self.current_time}
