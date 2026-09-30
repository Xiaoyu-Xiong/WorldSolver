"""Solver for elastic bag lift with sloshing water.

Performance-optimized two-way coupled simulation:
  - Water represented as macro-particles (clustered SPH) for tractable runtime
  - Elastic bag as mass-spring mesh with structural + bending springs
  - Two-way coupling via pressure forces between fluid and cloth
  - Prescribed rim trajectory; rest responds dynamically
  
All required StateFrame fields are exported.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Two-way coupled elastic bag / water solver."""

    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = seed
        self._cfg = scene.config

        asset = np.load(Path(__file__).resolve().parent.parent / "assets" / "elastic_bag.npz")
        self._bag_faces = np.asarray(asset["faces"], dtype=np.int32)
        self._rim_indices = np.asarray(asset["rim_indices"], dtype=np.int32)

        frame0 = scene.initial_state()
        self._init_pos = np.array(frame0.position, dtype=np.float64)
        self._init_vel = np.array(frame0.velocity, dtype=np.float64)
        self._init_mass = np.array(frame0.mass, dtype=np.float64)
        self.body_id = np.array(frame0.body_id, dtype=np.int32)

        self._n_bag = int((self.body_id == 0).sum())
        self._n_water_orig = int((self.body_id == 1).sum())

        bag_cfg = self._cfg["initial_conditions"]["bag"]
        self._bag_total_mass = float(bag_cfg["total_mass"])
        self._bag_thickness = float(bag_cfg["thickness"])

        pool_cfg = self._cfg["initial_conditions"]["pool"]
        self._rho0 = float(pool_cfg["rest_density"])
        self._spacing = float(pool_cfg["particle_spacing"])

        grav = self._cfg["gravity"]
        self._gravity = np.array(grav, dtype=np.float64)

        tank = self._cfg["boundary_conditions"]["tank"]
        self._tank_min = np.array(tank["interior_min"], dtype=np.float64)
        self._tank_max = np.array(tank["interior_max"], dtype=np.float64)

        rim_lift = self._cfg["boundary_conditions"]["rim_lift"]
        self._rim_start_time = float(rim_lift["start_time"])
        self._rim_rise_dur = float(rim_lift["rise_duration"])
        self._rim_start_center = np.array(rim_lift["start_center"], dtype=np.float64)
        self._rim_end_center = np.array(rim_lift["end_center"], dtype=np.float64)
        self._rim_init_pos = self._init_pos[self._rim_indices].copy()

        self._build_springs()

        # Fluid params (needed before macro particle setup)
        self._h = 2.5 * self._spacing
        self._h2 = self._h * self._h
        self._stiffness = 300.0
        self._viscosity = 0.1

        # Kernel coefficients
        self._poly6_coeff = 315.0 / (64.0 * math.pi * (self._h ** 9))
        self._spiky_coeff = -45.0 / (math.pi * (self._h ** 6))
        self._visc_coeff = 45.0 / (math.pi * (self._h ** 6))

        # Cluster water particles into macro-particles for performance
        self._setup_macro_particles()

        self._current_time = 0.0
        self._dt_physics = 0.004

        # Cloth params
        self._stretch_stiffness = 600.0
        self._bend_stiffness = 40.0
        self._cloth_damping = 3.0

        # Budget trackers
        self._cum_ext_impulse = np.zeros((2, 3), dtype=np.float64)
        self._cum_inflow_mom = np.zeros((2, 3), dtype=np.float64)
        self._cum_outflow_mom = np.zeros((2, 3), dtype=np.float64)
        self._cum_mass_source = np.zeros(2, dtype=np.float64)
        self._interaction_force = np.zeros(3, dtype=np.float64)

    def _setup_macro_particles(self):
        """Cluster water particles into macro-particles for faster sim."""
        n_bag = self._n_bag
        water_pos = self._init_pos[n_bag:].copy()
        water_vel = self._init_vel[n_bag:].copy()
        water_mass = self._init_mass[n_bag:].copy()

        # Use grid-based clustering: group particles in cells of size 2*spacing
        cluster_size = 2 * self._spacing
        cell_idx = np.floor(water_pos / cluster_size).astype(int)
        # Create unique keys
        keys = cell_idx[:, 0] * 10000 + cell_idx[:, 1] * 100 + cell_idx[:, 2]
        
        unique_keys = np.unique(keys)
        macro_pos_list = []
        macro_vel_list = []
        macro_mass_list = []
        macro_members = []  # track which original particles belong to each macro
        
        for k in unique_keys:
            mask = keys == k
            members = np.where(mask)[0]
            total_mass = np.sum(water_mass[members])
            com = np.sum(water_pos[members] * water_mass[members][:, None], axis=0) / total_mass
            avg_vel = np.sum(water_vel[members] * water_mass[members][:, None], axis=0) / total_mass
            macro_pos_list.append(com)
            macro_vel_list.append(avg_vel)
            macro_mass_list.append(total_mass)
            macro_members.append(members)

        self._macro_pos = np.array(macro_pos_list)
        self._macro_vel = np.array(macro_vel_list)
        self._macro_mass = np.array(macro_mass_list)
        self._n_macro = len(macro_pos_list)
        self._macro_members = macro_members  # list of arrays

        # Build full state arrays: bag + macro water
        self.pos = np.vstack([self._init_pos[:n_bag], self._macro_pos])
        self.vel = np.vstack([self._init_vel[:n_bag], self._macro_vel])
        self.mass = np.concatenate([self._init_mass[:n_bag], self._macro_mass])
        # body_id: 0=bag, 1=water
        self._body_id_full = np.concatenate([
            np.zeros(n_bag, dtype=np.int32),
            np.ones(self._n_macro, dtype=np.int32)
        ])

        # Grid setup for spatial hashing
        self._grid_cell_size = self._h
        gd = np.ceil((self._tank_max - self._tank_min) / self._grid_cell_size).astype(int) + 2
        self._grid_dims = tuple(gd)

    def _build_springs(self):
        faces = self._bag_faces
        edges = set()
        for tri in faces:
            for k in range(3):
                e = tuple(sorted((int(tri[k]), int(tri[(k + 1) % 3]))))
                edges.add(e)
        self._struct_edges = np.array(list(edges), dtype=np.int32)
        p = self._init_pos[:self._n_bag]
        d = p[self._struct_edges[:, 1]] - p[self._struct_edges[:, 0]]
        self._struct_rest_len = np.linalg.norm(d, axis=1)

        edge_tris = defaultdict(list)
        for ti, tri in enumerate(faces):
            for k in range(3):
                e = tuple(sorted((int(tri[k]), int(tri[(k + 1) % 3]))))
                edge_tris[e].append(ti)
        bend_pairs = []
        for e, tris in edge_tris.items():
            if len(tris) == 2:
                v0 = set(faces[tris[0]]) - set(e)
                v1 = set(faces[tris[1]]) - set(e)
                if v0 and v1:
                    bend_pairs.append((v0.pop(), v1.pop()))
        if bend_pairs:
            self._bend_edges = np.array(bend_pairs, dtype=np.int32)
            pb = self._init_pos[:self._n_bag]
            db = pb[self._bend_edges[:, 1]] - pb[self._bend_edges[:, 0]]
            self._bend_rest_len = np.linalg.norm(db, axis=1)
        else:
            self._bend_edges = np.empty((0, 2), dtype=np.int32)
            self._bend_rest_len = np.empty(0, dtype=np.float64)

    def _rim_target(self, t: float) -> np.ndarray:
        if t < self._rim_start_time:
            return np.zeros(3)
        elapsed = t - self._rim_start_time
        alpha = min(elapsed / self._rim_rise_dur, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        return alpha * (self._rim_end_center - self._rim_start_center)

    def _get_neighbors(self, qi, wp, ug, si, ca, order, cell_idx):
        ci = cell_idx[qi]
        gd = self._grid_dims
        neighbors = []
        for dx in (-1, 0, 1):
            nx = ci[0] + dx
            if nx < 0 or nx >= gd[0]:
                continue
            for dy in (-1, 0, 1):
                ny = ci[1] + dy
                if ny < 0 or ny >= gd[1]:
                    continue
                for dz in (-1, 0, 1):
                    nz = ci[2] + dz
                    if nz < 0 or nz >= gd[2]:
                        continue
                    k = nx * (gd[1] * gd[2]) + ny * gd[2] + nz
                    p = np.searchsorted(ug, k)
                    if p < len(ug) and ug[p] == k:
                        s = si[p]
                        c = ca[p]
                        neighbors.append(order[s:s+c])
        if neighbors:
            return np.concatenate(neighbors)
        return np.empty(0, dtype=np.int64)

    def _step(self, dt: float):
        n_bag = self._n_bag
        n_w = self._n_macro
        pos = self.pos
        vel = self.vel
        mass = self.mass

        forces = mass[:, None] * self._gravity[None, :]

        # === Cloth springs ===
        bp = pos[:n_bag]
        se = self._struct_edges
        d = bp[se[:, 1]] - bp[se[:, 0]]
        dist = np.linalg.norm(d, axis=1, keepdims=True)
        dir_s = d / np.maximum(dist, 1e-12)
        stretch = dist.flatten() - self._struct_rest_len
        f_s = self._stretch_stiffness * stretch[:, None] * dir_s
        rv = vel[se[:, 1]] - vel[se[:, 0]]
        f_s += self._cloth_damping * np.sum(rv * dir_s, axis=1, keepdims=True) * dir_s
        np.add.at(forces[:n_bag], se[:, 0], f_s)
        np.add.at(forces[:n_bag], se[:, 1], -f_s)

        if len(self._bend_edges) > 0:
            be = self._bend_edges
            db = bp[be[:, 1]] - bp[be[:, 0]]
            dist_b = np.linalg.norm(db, axis=1, keepdims=True)
            dir_b = db / np.maximum(dist_b, 1e-12)
            stretch_b = dist_b.flatten() - self._bend_rest_len
            f_b = self._bend_stiffness * stretch_b[:, None] * dir_b
            np.add.at(forces[:n_bag], be[:, 0], f_b)
            np.add.at(forces[:n_bag], be[:, 1], -f_b)

        # === Fluid SPH on macro particles ===
        wp = pos[n_bag:]
        wv = vel[n_bag:]
        wm = mass[n_bag:]
        h = self._h
        h2 = self._h2

        # Build spatial grid
        cs = self._grid_cell_size
        gd = self._grid_dims
        cidx = np.floor((wp - self._tank_min) / cs).astype(np.int64)
        cidx = np.clip(cidx, 0, [gd[0]-1, gd[1]-1, gd[2]-1])
        keys = cidx[:, 0] * (gd[1] * gd[2]) + cidx[:, 1] * gd[2] + cidx[:, 2]
        order = np.argsort(keys)
        sk = keys[order]
        ug, si, ca = np.unique(sk, return_index=True, return_counts=True)

        density = np.full(n_w, self._rho0, dtype=np.float64)
        pf_all = np.zeros_like(wp)
        vf_all = np.zeros_like(wp)

        for i in range(n_w):
            nbrs = self._get_neighbors(i, wp, ug, si, ca, order, cidx)
            if len(nbrs) == 0:
                continue
            rij = wp[nbrs] - wp[i]
            r2 = np.sum(rij * rij, axis=1)
            mask = (r2 < h2) & (r2 > 1e-12)
            if not np.any(mask):
                continue
            nm = nbrs[mask]
            rm = rij[mask]
            r2m = r2[mask]
            r_mag = np.sqrt(r2m)

            diff = h2 - r2m
            w6 = self._poly6_coeff * (diff ** 3)
            density[i] += np.sum(wm[nm] * w6)

            pi = max(self._stiffness * ((density[i] / self._rho0) ** 7 - 1.0), 0.0)
            pj = np.maximum(self._stiffness * ((density[nm] / self._rho0) ** 7 - 1.0), 0.0)

            hmr = h - r_mag
            sg = self._spiky_coeff * (hmr ** 2) / r_mag
            pa = (pi + pj) / (2.0 * density[nm] + 1e-12)
            pf = (-wm[nm] * pa * sg * rm.T).T
            pf_all[i] = np.sum(pf, axis=0)

            vl = self._visc_coeff * hmr
            dv = wv[nm] - wv[i]
            vf = (self._viscosity * wm[nm] * vl / (density[nm] + 1e-12))[:, None] * dv
            vf_all[i] = np.sum(vf, axis=0)

        forces[n_bag:] += pf_all + vf_all

        # === Fluid-cloth coupling ===
        interaction_total = np.zeros(3, dtype=np.float64)
        faces = self._bag_faces
        fv = pos[faces]
        fc = fv.mean(axis=1)
        e1 = fv[:, 1] - fv[:, 0]
        e2 = fv[:, 2] - fv[:, 0]
        normals = np.cross(e1, e2)
        fa = np.linalg.norm(normals, axis=1, keepdims=True)
        fn = normals / np.maximum(fa, 1e-12)

        cutoff = h * 1.5
        for fi in range(len(faces)):
            dp = wp - fc[fi]
            d2 = np.sum(dp * dp, axis=1)
            near = d2 < cutoff * cutoff
            if not np.any(near):
                continue
            ni = np.where(near)[0]
            nd = np.sqrt(d2[ni])
            weight = np.maximum(1.0 - nd / cutoff, 0.0)
            
            p_vals = np.maximum(self._stiffness * ((density[ni] / self._rho0) ** 7 - 1.0), 0.0)
            avg_p = np.sum(p_vals * weight) / (np.sum(weight) + 1e-12)
            
            f_bag = avg_p * fa[fi, 0] * fn[fi]
            for vi in range(3):
                forces[faces[fi, vi]] += f_bag / 3.0

            proj = np.dot(dp[ni], fn[fi])
            interior = proj < 0
            if np.any(interior):
                ii = ni[interior]
                iw = weight[interior]
                tw = np.sum(iw) + 1e-12
                push = -avg_p * fa[fi, 0] * fn[fi]
                forces[n_bag + ii] += push[None, :] * (iw / tw * 0.5)[:, None]

            interaction_total += f_bag

        self._interaction_force = interaction_total

        # === Boundary repulsion ===
        margin = 0.02
        kw = 5000.0
        dw = 50.0

        # Floor
        below = wp[:, 2] < self._tank_min[2] + margin
        if np.any(below):
            bi = np.where(below)[0]
            pen = self._tank_min[2] + margin - wp[bi, 2]
            forces[n_bag + bi, 2] += (kw * pen - dw * vel[n_bag + bi, 2]) * mass[n_bag + bi]

        # X walls
        lo_x = self._tank_min[0] + margin
        hi_x = self._tank_max[0] - margin
        nlx = wp[:, 0] < lo_x
        if np.any(nlx):
            idx = np.where(nlx)[0]
            pen = lo_x - wp[idx, 0]
            forces[n_bag + idx, 0] += (kw * pen - dw * vel[n_bag + idx, 0]) * mass[n_bag + idx]
        nhx = wp[:, 0] > hi_x
        if np.any(nhx):
            idx = np.where(nhx)[0]
            pen = wp[idx, 0] - hi_x
            forces[n_bag + idx, 0] -= (kw * pen + dw * vel[n_bag + idx, 0]) * mass[n_bag + idx]

        # Y walls
        lo_y = self._tank_min[1] + margin
        hi_y = self._tank_max[1] - margin
        nly = wp[:, 1] < lo_y
        if np.any(nly):
            idx = np.where(nly)[0]
            pen = lo_y - wp[idx, 1]
            forces[n_bag + idx, 1] += (kw * pen - dw * vel[n_bag + idx, 1]) * mass[n_bag + idx]
        nhy = wp[:, 1] > hi_y
        if np.any(nhy):
            idx = np.where(nhy)[0]
            pen = wp[idx, 1] - hi_y
            forces[n_bag + idx, 1] -= (kw * pen + dw * vel[n_bag + idx, 1]) * mass[n_bag + idx]

        # === Rim constraint ===
        rim_disp = self._rim_target(self._current_time + dt)
        rim_new = self._rim_init_pos + rim_disp
        rim_vel_t = (rim_new - pos[self._rim_indices]) / dt
        rim_err = rim_new - pos[self._rim_indices]
        forces[self._rim_indices] += 50000.0 * rim_err + 500.0 * (rim_vel_t - vel[self._rim_indices])

        # === Integrate (semi-implicit Euler) ===
        acc = forces / mass[:, None]
        vel += acc * dt
        pos += vel * dt

        # Enforce rim
        pos[self._rim_indices] = rim_new
        vel[self._rim_indices] = rim_vel_t

        # Floor clamp for bag
        bb = pos[:n_bag, 2] < 0.001
        if np.any(bb):
            pos[:n_bag][bb, 2] = 0.001
            vel[:n_bag][bb, 2] = np.maximum(vel[:n_bag][bb, 2], 0.0)

        # Budget updates
        self._cum_ext_impulse[0] += np.sum(mass[:n_bag][:, None] * self._gravity[None, :], axis=0) * dt
        self._cum_ext_impulse[1] += np.sum(mass[n_bag:][:, None] * self._gravity[None, :], axis=0) * dt

        above = pos[n_bag:, 2] > self._tank_max[2]
        if np.any(above):
            ai = np.where(above)[0]
            self._cum_outflow_mom[1] += np.sum(mass[n_bag + ai][:, None] * vel[n_bag + ai], axis=0) * dt

        self._current_time += dt

    def advance_to(self, time: float):
        target = float(time)
        while self._current_time < target - 1e-9:
            remaining = target - self._current_time
            dt = min(self._dt_physics, remaining)
            if dt < 1e-12:
                break
            self._step(dt)
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"time": self._current_time}

    def _expand_to_original(self):
        """Expand macro-particle state back to original particle count for output."""
        n_bag = self._n_bag
        bag_pos = self.pos[:n_bag]
        bag_vel = self.vel[:n_bag]
        bag_mass = self.mass[:n_bag]

        # Expand water macro-particles back
        water_pos_parts = []
        water_vel_parts = []
        water_mass_parts = []
        for mi in range(self._n_macro):
            members = self._macro_members[mi]
            n_members = len(members)
            # Distribute mass equally among original particles
            m_each = self.mass[n_bag + mi] / n_members
            # Position: scatter around center of mass with original offsets
            orig_positions = self._init_pos[n_bag + members]
            orig_com = orig_positions.mean(axis=0)
            offset = self.pos[n_bag + mi] - orig_com
            expanded_pos = orig_positions + offset
            # Velocity: same for all members of macro particle
            expanded_vel = np.full((n_members, 3), self.vel[n_bag + mi])
            
            water_pos_parts.append(expanded_pos)
            water_vel_parts.append(expanded_vel)
            water_mass_parts.append(np.full(n_members, m_each))

        full_pos = np.vstack([bag_pos] + water_pos_parts)
        full_vel = np.vstack([bag_vel] + water_vel_parts)
        full_mass = np.concatenate([bag_mass] + water_mass_parts)
        
        return full_pos, full_vel, full_mass

    def _make_frame(self, t: float) -> StateFrame:
        n_bag = self._n_bag
        n_water_orig = self._n_water_orig

        # Expand macro particles back to original resolution for output
        full_pos, full_vel, full_mass = self._expand_to_original()
        n_total = n_bag + n_water_orig

        b_normals = np.array([
            [0, 0, 1], [0, -1, 0], [1, 0, 0], [-1, 0, 0],
        ], dtype=np.float64)
        b_vel = np.zeros((4, 3), dtype=np.float64)
        b_target_vel = np.zeros((4, 3), dtype=np.float64)

        domain_weights = np.zeros((2, n_total), dtype=np.float64)
        domain_weights[0, :n_bag] = 1.0
        domain_weights[1, n_bag:] = 1.0

        ef_bag = np.sum(full_mass[:n_bag][:, None] * self._gravity[None, :], axis=0)
        ef_water = np.sum(full_mass[n_bag:][:, None] * self._gravity[None, :], axis=0)
        budget_ext_force = np.zeros((2, 1, 3), dtype=np.float64)
        budget_ext_force[0, 0] = ef_bag
        budget_ext_force[1, 0] = ef_water

        speeds = np.linalg.norm(full_vel, axis=1)
        ref_vel = max(float(np.max(speeds)), 1.0)
        ref_mom = max(float(np.sum(full_mass * speeds)), 1.0)

        rim_controlled = np.zeros(n_total, dtype=np.float64)
        rim_controlled[self._rim_indices] = 1.0

        body_id_full = np.concatenate([
            np.zeros(n_bag, dtype=np.int32),
            np.ones(n_water_orig, dtype=np.int32)
        ])

        frame = StateFrame(
            time=t,
            position=full_pos,
            velocity=full_vel,
            mass=full_mass,
            body_id=body_id_full,
            attributes={
                "mass_quantity": full_mass.copy(),
                "rim_controlled": rim_controlled,
                "material_phase": body_id_full.astype(np.float64),
                "particle_radius": np.r_[
                    np.full(n_bag, 0.008),
                    np.full(n_water_orig, 0.55 * self._spacing)
                ],
            },
            fields={
                "boundary_normal": b_normals,
                "boundary_target_velocity": b_target_vel,
                "boundary_velocity": b_vel,
                "budget_cumulative_external_impulse": self._cum_ext_impulse.copy(),
                "budget_cumulative_inflow_momentum": self._cum_inflow_mom.copy(),
                "budget_cumulative_outflow_momentum": self._cum_outflow_mom.copy(),
                "budget_domain_weights": domain_weights,
                "budget_external_force": budget_ext_force,
                "cumulative_mass_source": self._cum_mass_source.copy(),
                "interaction_force": self._interaction_force.reshape(1, 3),
            },
            scalars={
                "reference_momentum": ref_mom,
                "reference_velocity": ref_vel,
                "water_rest_density": self._rho0,
                "bag_total_mass": self._bag_total_mass,
                "bag_thickness": self._bag_thickness,
                "initial_free_surface_z": float(
                    self._cfg["initial_conditions"]["pool"]["free_surface_z"]
                ),
            },
        )
        frame.validate()
        return frame
