"""Two-way coupled cloth-water solver for the suspended cloth scene.

The cloth is modeled as a constrained mass-spring network integrated with
position-based dynamics (PBD).  The water is modeled with weakly compressible
Smoothed Particle Hydrodynamics (SPH).  Coupling is enforced through a
one-sided penalty that keeps water particles above the local cloth tangent
plane inside the cloth domain, while letting them fall freely after passing
over the free edges.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        cfg = scene.config
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.duration = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])

        # Substepping for stability (0.02 / 0.001 = 20 substeps per sample).
        self.dt = 0.001

        cloth_cfg = cfg["cloth"]
        water_cfg = cfg["water"]
        inflow_cfg = cfg["inflow"]

        self.rho0 = float(water_cfg["density"])
        self.spacing = float(water_cfg["initial_particle_spacing"])
        self.inflow_active = (float(inflow_cfg["active_interval"][0]),
                                float(inflow_cfg["active_interval"][1]))
        self.inflow_center = np.asarray(inflow_cfg["center"], dtype=float)
        self.inflow_axis = np.asarray(inflow_cfg["axis"], dtype=float)
        self.inflow_speed = float(inflow_cfg["speed"])
        self.inflow_radius = float(inflow_cfg["radius"])

        # Initial frame from the scene.
        frame0 = scene.initial_state()
        self.time = 0.0

        self.pos = np.asarray(frame0.position, dtype=float).copy()
        self.vel = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.attributes = {
            k: np.asarray(v).copy()
            for k, v in frame0.attributes.items()
        }

        # Partition into cloth and water.
        self.cloth_idx = np.where(self.body_id == 0)[0]
        self.water_idx = np.where(self.body_id == 1)[0]
        self.n_cloth = len(self.cloth_idx)
        self.n_water = len(self.water_idx)

        # Cloth grid.
        grid_shape = tuple(scene.metadata()["topology"]["grid_shape"])
        self.nx, self.ny = grid_shape
        self._build_cloth_edges_faces()

        self.pinned = self.attributes["pinned"].astype(bool)
        self.cloth_pos0 = self.pos[self.cloth_idx].copy()
        self.cloth_vel = self.vel[self.cloth_idx].copy()

        # SPH / coupling parameters.
        self.h_smooth = 1.5 * self.spacing          # smoothing length
        self.support = 2.0 * self.h_smooth          # compact support radius
        self.k_stiff = 2.0e4                      # pressure stiffness
        self.visc = 0.4                           # artificial viscosity coefficient
        self.k_wall = 2.0e3                         # boundary penalty stiffness
        self.k_cloth = 1.5e3                        # cloth-water coupling stiffness

        # Domain for boundary penalties (water particle domain plus floor).
        self.domain_min = np.array([-2.05, -1.80, 0.00], dtype=float)
        self.domain_max = np.array([2.05, 1.80, 3.22], dtype=float)

        # Static boundary samples (floor + four side walls, open top).
        self.boundary_normal = np.array([
            [0.0, 0.0, 1.0],   # floor
            [1.0, 0.0, 0.0],   # x max wall
            [-1.0, 0.0, 0.0],  # x min wall
            [0.0, 1.0, 0.0],   # y max wall
            [0.0, -1.0, 0.0],  # y min wall
        ], dtype=float)
        self.boundary_target_velocity = np.zeros_like(self.boundary_normal)
        self.boundary_velocity = np.zeros_like(self.boundary_normal)

        # Inflow bookkeeping.
        self._inflow_accum = 0.0
        self._water_volume_added = 0.0

        # Cumulative budget / bookkeeping.
        self.cum_ext_impulse = np.zeros((2, 3), dtype=float)
        self.cum_inflow_momentum = np.zeros((2, 3), dtype=float)
        self.cum_outflow_momentum = np.zeros((2, 3), dtype=float)
        self.cum_mass_source = np.zeros(1, dtype=float)
        self.last_budget_ext_force = np.zeros((2, 1, 3), dtype=float)
        self.last_interaction_force = np.zeros((1, 3), dtype=float)

        self.ref_velocity = self.inflow_speed
        self.ref_momentum = float(np.sum(self.mass) * self.ref_velocity)

        self.diag_info: dict[str, object] = {
            "dt": self.dt,
            "cloth_vertices": self.n_cloth,
            "initial_water": self.n_water,
        }

    def _build_cloth_edges_faces(self) -> None:
        nx, ny = self.nx, self.ny
        edges = []
        faces = []
        # Structural and shear (diagonal) edges; each edge stored once.
        for j in range(ny):
            for i in range(nx):
                v = j * nx + i
                if i + 1 < nx:
                    edges.append((v, v + 1))
                if j + 1 < ny:
                    edges.append((v, v + nx))
                if i + 1 < nx and j + 1 < ny:
                    edges.append((v, v + nx + 1))
                    edges.append((v + 1, v + nx))
                if i + 1 < nx and j + 1 < ny:
                    faces.append((v, v + 1, v + nx + 1))
                    faces.append((v, v + nx + 1, v + nx))
        self.edges = np.asarray(edges, dtype=np.int32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.rest_length = np.linalg.norm(
            self.pos[self.cloth_idx][self.edges[:, 1]] -
            self.pos[self.cloth_idx][self.edges[:, 0]],
            axis=1,
        )

    # ------------------------------------------------------------------
    # Kernel helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _kernel(r: np.ndarray, h: float) -> np.ndarray:
        q = r / h
        alpha = 1.0 / (np.pi * h ** 3)
        w = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        w[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
        w[m2] = 0.25 * (2.0 - q[m2]) ** 3
        return alpha * w

    @staticmethod
    def _dW_dr(r: np.ndarray, h: float) -> np.ndarray:
        q = r / h
        alpha = 1.0 / (np.pi * h ** 3)
        dw = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        dw[m1] = (-3.0 * q[m1] + 2.25 * q[m1] ** 2) / h
        dw[m2] = -0.75 * (2.0 - q[m2]) ** 2 / h
        return alpha * dw

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------
    def _boundary_force(self, pos: np.ndarray) -> np.ndarray:
        f = np.zeros_like(pos)
        # x walls
        d = self.domain_min[0] - pos[:, 0]
        mask = d > 0.0
        f[mask, 0] += self.k_wall * d[mask]
        d = pos[:, 0] - self.domain_max[0]
        mask = d > 0.0
        f[mask, 0] -= self.k_wall * d[mask]
        # y walls
        d = self.domain_min[1] - pos[:, 1]
        mask = d > 0.0
        f[mask, 1] += self.k_wall * d[mask]
        d = pos[:, 1] - self.domain_max[1]
        mask = d > 0.0
        f[mask, 1] -= self.k_wall * d[mask]
        # floor only; top is open
        d = self.domain_min[2] - pos[:, 2]
        mask = d > 0.0
        f[mask, 2] += self.k_wall * d[mask]
        return f

    def _cloth_normals(self, cloth_pos: np.ndarray) -> np.ndarray:
        # Per-face normal.
        p0 = cloth_pos[self.faces[:, 0]]
        p1 = cloth_pos[self.faces[:, 1]]
        p2 = cloth_pos[self.faces[:, 2]]
        n = np.cross(p1 - p0, p2 - p0)
        # Accumulate onto vertices.
        normals = np.zeros_like(cloth_pos)
        counts = np.zeros(len(cloth_pos))
        np.add.at(normals, self.faces[:, 0], n)
        np.add.at(normals, self.faces[:, 1], n)
        np.add.at(normals, self.faces[:, 2], n)
        np.add.at(counts, self.faces[:, 0], 1)
        np.add.at(counts, self.faces[:, 1], 1)
        np.add.at(counts, self.faces[:, 2], 1)
        normals /= counts[:, None] + 1e-12
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= norm + 1e-12
        # Keep a consistent upward orientation.
        if normals[:, 2].mean() < 0.0:
            normals = -normals
        return normals

    def _cloth_coupling(
        self,
        wpos: np.ndarray,
        cloth_pos: np.ndarray,
        cloth_norm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return water and cloth coupling forces."""
        f_water = np.zeros_like(wpos)
        f_cloth = np.zeros_like(cloth_pos)
        if len(wpos) == 0 or len(cloth_pos) == 0:
            return f_water, f_cloth

        tree = cKDTree(cloth_pos)
        dist, idx = tree.query(wpos, k=4)
        # Mask for particles close enough to the cloth surface.
        mask = dist[:, 0] < self.support
        if not np.any(mask):
            return f_water, f_cloth

        # Build inverse-distance weights over the k nearest vertices.
        eps = 1e-12
        valid = dist < self.support
        weights = np.where(valid, 1.0 / (dist + eps), 0.0)
        sum_w = weights.sum(axis=1, keepdims=True)
        weights = weights / (sum_w + 1e-12)

        # Weighted cloth position / normal for each water particle.
        cpos_n = cloth_pos[idx]                       # (N,4,3)
        cnorm_n = cloth_norm[idx]
        wc = np.einsum("nk,nki->ni", weights, cpos_n)
        n = np.einsum("nk,nki->ni", weights, cnorm_n)
        n_norm = np.linalg.norm(n, axis=1, keepdims=True)
        n /= n_norm + 1e-12

        s = np.einsum("ni,ni->n", wpos - wc, n)
        active = mask & (s < self.h_smooth)
        if not np.any(active):
            return f_water, f_cloth

        mag = self.k_cloth * (self.h_smooth - s[active])
        fvec = mag[:, None] * n[active]
        f_water[active] += fvec

        # Distribute the opposite reaction to the nearest cloth vertices.
        w_active = weights[active]
        idx_active = idx[active]
        for k in range(4):
            vertex = idx_active[:, k]
            contrib = -fvec * w_active[:, k:k+1]
            np.add.at(f_cloth, vertex, contrib)

        return f_water, f_cloth

    def _sph_forces(self, wpos: np.ndarray, wvel: np.ndarray, wm: np.ndarray) -> np.ndarray:
        """Return SPH pressure + viscosity forces for water particles."""
        f = np.zeros_like(wpos)
        if len(wpos) == 0:
            return f

        tree = cKDTree(wpos)
        neighbors = tree.query_ball_tree(tree, r=self.support)

        i_idx = []
        j_idx = []
        for i, nbr in enumerate(neighbors):
            i_idx.extend([i] * len(nbr))
            j_idx.extend(nbr)
        i_idx = np.asarray(i_idx, dtype=np.int32)
        j_idx = np.asarray(j_idx, dtype=np.int32)

        r_vec = wpos[i_idx] - wpos[j_idx]
        r = np.linalg.norm(r_vec, axis=1)

        # Density (includes self-pair r=0).
        W = self._kernel(r, self.h_smooth)
        rho = np.zeros(len(wpos), dtype=float)
        np.add.at(rho, i_idx, wm[j_idx] * W)
        # Guard against isolated particles.
        rho = np.maximum(rho, 1e-6)

        # Pressure force (only r>0 pairs).
        mask = r > 0.0
        if np.any(mask):
            ri = r[mask]
            r_vec_m = r_vec[mask]
            ii = i_idx[mask]
            jj = j_idx[mask]
            p_i = self.k_stiff * (rho[ii] - self.rho0)
            p_j = self.k_stiff * (rho[jj] - self.rho0)
            mj = wm[jj]
            dW = self._dW_dr(ri, self.h_smooth)
            # -m_j * (p_i/rho_i^2 + p_j/rho_j^2) * dW/dr * (r_vec/r)
            factor = -(mj * (p_i / rho[ii] ** 2 + p_j / rho[jj] ** 2) * dW / ri)
            force = factor[:, None] * r_vec_m
            np.add.at(f, ii, force)
            np.add.at(f, jj, -force)

            # Viscosity (pairwise, anti-symmetric).
            vi = wvel[ii]
            vj = wvel[jj]
            Wvis = self._kernel(ri, self.h_smooth)
            # mu = nu * m_j / rho_j
            mu = self.visc * mj / rho[jj]
            v_ij = vi - vj
            fvis = -mu[:, None] * v_ij * Wvis[:, None]
            np.add.at(f, ii, fvis)
            np.add.at(f, jj, -fvis)

        return f

    def _pbd_step(self, cloth_pos: np.ndarray, cloth_vel: np.ndarray,
                  external_f: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """One PBD substep for the cloth."""
        m = self.mass[self.cloth_idx][:, None]
        vel = cloth_vel + dt * external_f / m
        pos = cloth_pos + dt * vel

        # Satisfy edge constraints.
        for _ in range(5):
            d = pos[self.edges[:, 1]] - pos[self.edges[:, 0]]
            l = np.linalg.norm(d, axis=1)
            l = np.maximum(l, 1e-12)
            delta = 0.5 * (1.0 - self.rest_length / l)[:, None] * d
            pos[self.edges[:, 0]] += delta
            pos[self.edges[:, 1]] -= delta
            # Re-attach pinned corners.
            pos[self.pinned] = self.cloth_pos0[self.pinned]

        vel = (pos - cloth_pos) / dt
        vel *= 0.99
        vel[self.pinned] = 0.0
        pos[self.pinned] = self.cloth_pos0[self.pinned]
        return pos, vel

    # ------------------------------------------------------------------
    # Inflow
    # ------------------------------------------------------------------
    def _maybe_emit(self, dt: float) -> None:
        if not (self.inflow_active[0] <= self.time < self.inflow_active[1]):
            return

        layer_dt = self.spacing / self.inflow_speed
        self._inflow_accum += dt
        if self._inflow_accum < layer_dt:
            return

        # Number of layers elapsed.
        n_layers = int(self._inflow_accum // layer_dt)
        self._inflow_accum -= n_layers * layer_dt

        n_per_layer = 14
        for _ in range(n_layers):
            self._emit_layer(n_per_layer)

    def _emit_layer(self, n: int) -> None:
        # Random positions inside the inflow disk.
        u = self.rng.uniform(0.0, 1.0, n)
        theta = self.rng.uniform(0.0, 2.0 * np.pi, n)
        r = self.inflow_radius * np.sqrt(u)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = self.rng.uniform(3.10, 3.18, n)
        pos_new = np.column_stack((x, y, z))
        vel_new = np.tile(self.inflow_axis * self.inflow_speed, (n, 1))
        mass_new = np.full(n, self.rho0 * self.spacing ** 3, dtype=float)
        body_new = np.ones(n, dtype=np.int32)

        self.pos = np.vstack([self.pos, pos_new])
        self.vel = np.vstack([self.vel, vel_new])
        self.mass = np.concatenate([self.mass, mass_new])
        self.body_id = np.concatenate([self.body_id, body_new])

        # Extend attributes.
        uv_new = np.zeros((n, 2), dtype=float)
        pinned_new = np.zeros(n, dtype=float)
        phase_new = np.ones(n, dtype=float)
        self.attributes["uv"] = np.vstack([self.attributes["uv"], uv_new])
        self.attributes["pinned"] = np.concatenate([self.attributes["pinned"], pinned_new])
        self.attributes["material_phase"] = np.concatenate(
            [self.attributes["material_phase"], phase_new]
        )

        # Update indices.
        new_water = np.arange(self.n_water, self.n_water + n)
        self.water_idx = np.concatenate([self.water_idx, new_water + self.n_cloth])
        self.n_water += n

        # Budget updates.
        self.cum_mass_source[0] += np.sum(mass_new)
        self.cum_inflow_momentum[1] += np.sum(mass_new[:, None] * vel_new, axis=0)

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    def _step(self) -> None:
        self._maybe_emit(self.dt)

        cloth_pos = self.pos[self.cloth_idx].copy()
        cloth_vel = self.vel[self.cloth_idx].copy()
        water_pos = self.pos[self.water_idx].copy()
        water_vel = self.vel[self.water_idx].copy()
        water_mass = self.mass[self.water_idx].copy()

        # Cloth normals on the current (deformed) configuration.
        cloth_norm = self._cloth_normals(cloth_pos)

        # SPH internal forces for water.
        f_sph = self._sph_forces(water_pos, water_vel, water_mass)

        # Cloth-water coupling.
        f_w_couple, f_c_couple = self._cloth_coupling(water_pos, cloth_pos, cloth_norm)

        # Boundary forces.
        f_w_bound = self._boundary_force(water_pos)
        f_c_bound = self._boundary_force(cloth_pos)

        # Gravity.
        f_w_grav = water_mass[:, None] * self.gravity
        f_c_grav = self.mass[self.cloth_idx][:, None] * self.gravity

        # Total water force.
        f_water_total = f_sph + f_w_couple + f_w_bound + f_w_grav
        # Total cloth external force (internal forces are handled by PBD).
        f_cloth_ext = f_c_couple + f_c_bound + f_c_grav

        # Update water.
        water_vel = water_vel + self.dt * f_water_total / water_mass[:, None]
        water_pos = water_pos + self.dt * water_vel

        # Update cloth with PBD.
        cloth_pos, cloth_vel = self._pbd_step(cloth_pos, cloth_vel, f_cloth_ext, self.dt)

        # Store back.
        self.pos[self.cloth_idx] = cloth_pos
        self.pos[self.water_idx] = water_pos
        self.vel[self.cloth_idx] = cloth_vel
        self.vel[self.water_idx] = water_vel
        self.mass[self.water_idx] = water_mass

        # Budget bookkeeping.
        ext_group = np.zeros((2, 3), dtype=float)
        ext_group[0] = f_c_bound.sum(axis=0) + f_c_grav.sum(axis=0)
        ext_group[1] = f_w_bound.sum(axis=0) + f_w_grav.sum(axis=0)
        self.cum_ext_impulse += self.dt * ext_group
        self.last_budget_ext_force[:, 0, :] = ext_group

        interaction = f_w_couple.sum(axis=0)
        self.last_interaction_force[0] = interaction

        self.time += self.dt

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time + 1e-12 < target:
            dt = min(self.dt, target - self.time)
            if dt < self.dt:
                # For the final fractional substep, temporarily shrink dt and step.
                saved_dt = self.dt
                self.dt = dt
                self._step()
                self.dt = saved_dt
            else:
                self._step()
        return self._make_frame()

    # ------------------------------------------------------------------
    # Frame export
    # ------------------------------------------------------------------
    def _make_frame(self) -> StateFrame:
        n = len(self.pos)
        # Domain weights: [2, N]
        weights = np.zeros((2, n), dtype=float)
        weights[0, self.cloth_idx] = 1.0
        weights[1, self.water_idx] = 1.0

        frame = StateFrame(
            time=float(self.time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                **{k: v.copy() for k, v in self.attributes.items()},
            },
            fields={
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_target_velocity.copy(),
                "boundary_velocity": self.boundary_velocity.copy(),
                "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_momentum.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_momentum.copy(),
                "budget_domain_weights": weights,
                "budget_external_force": self.last_budget_ext_force.copy(),
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "interaction_force": self.last_interaction_force.copy(),
            },
            scalars={
                "reference_velocity": float(self.ref_velocity),
                "reference_momentum": float(self.ref_momentum),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            **self.diag_info,
            "current_time": self.time,
            "n_cloth": self.n_cloth,
            "n_water": self.n_water,
            "total_particles": self.n_cloth + self.n_water,
        }
