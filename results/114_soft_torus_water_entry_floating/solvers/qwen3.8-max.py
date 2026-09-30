"""Solver for 114_soft_torus_water_entry_floating.

Two-way coupled deformable torus entering water and floating. Uses a simplified
SPH-like particle method with buoyancy coupling to an elastic shell model for
the torus surface. The torus is represented by its rest-configuration surface
vertices; elasticity uses spring forces between neighboring vertices. Water
particles interact via pressure-based repulsion and buoyancy forces on the
torus, while the torus exerts reaction forces back on nearby water particles
(two-way coupling).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# Geometry helpers (mirrors scene.py so we don't import it at runtime)
# ---------------------------------------------------------------------------
MAJOR_RADIUS = 0.255
MINOR_RADIUS = 0.105
TORUS_CENTER_INIT = np.array([0.0, 0.0, 0.66], dtype=float)


def _build_torus_neighbors(n_major: int = 48, n_minor: int = 20):
    """Return edge list for the torus mesh connectivity."""
    edges = []
    for i in range(n_major):
        for j in range(n_minor):
            a = i * n_minor + j
            b = ((i + 1) % n_major) * n_minor + j
            c = ((i + 1) % n_major) * n_minor + (j + 1) % n_minor
            d = i * n_minor + (j + 1) % n_minor
            edges.append((a, b))
            edges.append((b, c))
            edges.append((c, d))
            edges.append((d, a))
    return edges


def _rest_positions(n_major: int = 48, n_minor: int = 20):
    verts = []
    for i in range(n_major):
        u = 2.0 * math.pi * i / n_major
        for j in range(n_minor):
            v = 2.0 * math.pi * j / n_minor
            r = MAJOR_RADIUS + MINOR_RADIUS * math.cos(v)
            verts.append([r * math.cos(u), r * math.sin(u), MINOR_RADIUS * math.sin(v)])
    return np.asarray(verts, dtype=float)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        cfg = scene.config
        self.gravity = np.array(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.dt_sample = float(cfg["sample_dt"])
        # Sub-step for stability
        self.substeps = 10
        self.dt = self.dt_sample / self.substeps

        # Initial state from scene
        frame0 = scene.initial_state()
        self.position = frame0.position.copy()
        self.velocity = frame0.velocity.copy()
        self.mass = frame0.mass.copy()
        self.body_id = frame0.body_id.copy()
        self.rest_position = frame0.attributes["rest_position"].copy()

        self.n_total = len(self.position)
        self.torus_mask = self.body_id == 1
        self.water_mask = self.body_id == 2
        self.n_torus = int(self.torus_mask.sum())
        self.n_water = int(self.water_mask.sum())

        # Torus elastic network
        self.edges = np.array(_build_torus_neighbors(), dtype=np.int32)
        self.edge_a = self.edges[:, 0]
        self.edge_b = self.edges[:, 1]
        self.edge_rest_len = np.linalg.norm(
            self.rest_position[self.edge_a] - self.rest_position[self.edge_b], axis=1
        )
        # Spring stiffness – moderate to avoid instability
        self.k_spring = 800.0
        self.damping_spring = 5.0

        # Water parameters
        self.water_density = float(cfg["initial_conditions"]["water"]["density"])
        self.torus_density = float(cfg["initial_conditions"]["torus"]["density"])
        self.free_surface_z0 = float(cfg["initial_conditions"]["water"]["free_surface_z"])

        # Tub boundaries (axis-aligned boxes from scene_geometry)
        self.walls = [
            # (center, half_extents, outward_normal)
            (np.array([0.0, 0.0, -0.025]), np.array([0.56, 0.56, 0.025]), np.array([0.0, 0.0, 1.0])),   # floor top
            (np.array([0.0, 0.53, 0.35]), np.array([0.56, 0.03, 0.35]), np.array([0.0, -1.0, 0.0])),     # back
            (np.array([-0.53, 0.0, 0.35]), np.array([0.03, 0.5, 0.35]), np.array([1.0, 0.0, 0.0])),      # left
            (np.array([0.53, 0.0, 0.35]), np.array([0.03, 0.5, 0.35]), np.array([-1.0, 0.0, 0.0])),      # right
            (np.array([0.0, -0.525, 0.16]), np.array([0.56, 0.025, 0.16]), np.array([0.0, 1.0, 0.0])),   # front
        ]

        # Boundary samples for trajectory output
        bpos, bnorm = self._boundary_samples()
        self.boundary_pos = bpos
        self.boundary_normal = bnorm
        self.n_boundary = len(bpos)

        # Budget accumulators
        self.cum_ext_impulse = np.zeros((2, 3), dtype=float)  # G=2 domains: torus, water
        self.cum_inflow_mom = np.zeros((2, 3), dtype=float)
        self.cum_outflow_mom = np.zeros((2, 3), dtype=float)

        # Reference scales
        torus_vol = 2.0 * math.pi**2 * MAJOR_RADIUS * MINOR_RADIUS**2
        self.ref_momentum = self.torus_density * torus_vol * 1.0
        self.ref_velocity = 1.0

        self.current_time = 0.0
        
        # Precompute torus vertex volume
        self.torus_vol = 2.0 * math.pi**2 * MAJOR_RADIUS * MINOR_RADIUS**2
        self.vol_per_vertex = self.torus_vol / self.n_torus

    # ------------------------------------------------------------------
    def _boundary_samples(self, spacing: float = 0.12):
        pts, norms = [], []
        for center, he, normal in self.walls:
            if abs(normal[2]) > 0.5:
                xs = np.arange(center[0] - he[0] + spacing/2, center[0] + he[0], spacing)
                ys = np.arange(center[1] - he[1] + spacing/2, center[1] + he[1], spacing)
                for x in xs:
                    for y in ys:
                        pts.append([x, y, center[2]])
                        norms.append(normal)
            elif abs(normal[1]) > 0.5:
                xs = np.arange(center[0] - he[0] + spacing/2, center[0] + he[0], spacing)
                zs = np.arange(center[2] - he[2] + spacing/2, center[2] + he[2], spacing)
                for x in xs:
                    for z in zs:
                        pts.append([x, center[1], z])
                        norms.append(normal)
            else:
                ys = np.arange(center[1] - he[1] + spacing/2, center[1] + he[1], spacing)
                zs = np.arange(center[2] - he[2] + spacing/2, center[2] + he[2], spacing)
                for y in ys:
                    for z in zs:
                        pts.append([center[0], y, z])
                        norms.append(normal)
        return np.array(pts, dtype=float), np.array(norms, dtype=float)

    # ------------------------------------------------------------------
    def _clamp_velocities(self, max_speed: float = 5.0):
        """Clamp velocities to prevent blowup."""
        speeds = np.linalg.norm(self.velocity, axis=1)
        mask = speeds > max_speed
        if np.any(mask):
            self.velocity[mask] *= (max_speed / speeds[mask])[:, None]

    # ------------------------------------------------------------------
    def _step(self, dt: float):
        pos = self.position
        vel = self.velocity
        mass = self.mass
        n = self.n_total
        force = np.zeros_like(pos)

        # --- Gravity ---
        force += mass[:, None] * self.gravity[None, :]

        # --- Torus elastic springs (vectorized) ---
        diff = pos[self.edge_a] - pos[self.edge_b]
        dist = np.linalg.norm(diff, axis=1)
        dist = np.maximum(dist, 1e-12)
        stretch = dist - self.edge_rest_len
        direction = diff / dist[:, None]
        
        rel_v = vel[self.edge_a] - vel[self.edge_b]
        damp_proj = np.sum(rel_v * direction, axis=1)
        f_mag = self.k_spring * stretch + self.damping_spring * damp_proj
        
        f_vec = f_mag[:, None] * direction
        np.add.at(force, self.edge_a, -f_vec)
        np.add.at(force, self.edge_b, f_vec)

        # --- Buoyancy / fluid-solid coupling ---
        t_idx = np.where(self.torus_mask)[0]
        w_idx = np.where(self.water_mask)[0]
        
        smooth_h = 0.08
        
        torus_pos = pos[t_idx]
        water_pos = pos[w_idx]
        
        # Estimate local free surface for each torus vertex
        surface_est = np.full(self.n_torus, self.free_surface_z0)
        
        # For submerged/near-surface torus vertices, use nearby water to estimate surface
        # Only do this for vertices near the water level to save time
        near_water = np.abs(torus_pos[:, 2] - self.free_surface_z0) < 0.3
        near_indices = np.where(near_water)[0]
        
        for ti_local in near_indices:
            diffs = water_pos - torus_pos[ti_local]
            dists_sq = np.sum(diffs**2, axis=1)
            close = dists_sq < smooth_h**2
            if np.any(close):
                weights = np.exp(-dists_sq[close] / (smooth_h**2 * 0.25))
                w_sum = weights.sum()
                if w_sum > 1e-10:
                    surface_est[ti_local] = np.dot(weights, water_pos[close, 2]) / w_sum
        
        # Buoyancy + drag on torus
        for k, ti in enumerate(t_idx):
            z = pos[ti, 2]
            # Submerged fraction: how much of this vertex's patch is below surface
            sub_frac = np.clip((surface_est[k] - z + MINOR_RADIUS * 0.5) / MINOR_RADIUS, 0.0, 1.0)
            
            if sub_frac > 0.001:
                # Buoyancy
                buoyancy = self.water_density * abs(self.gravity[2]) * self.vol_per_vertex * sub_frac
                force[ti, 2] += buoyancy
                
                # Drag
                v = vel[ti]
                speed = np.linalg.norm(v)
                if speed > 1e-8:
                    area = self.vol_per_vertex ** (2.0/3.0)
                    drag_coeff = 0.5 * self.water_density * 0.6 * area * sub_frac
                    drag_force = -drag_coeff * speed * v
                    force[ti] += drag_force
                    
                    # Two-way coupling: push nearby water particles
                    diffs = water_pos - pos[ti]
                    dists_sq = np.sum(diffs**2, axis=1)
                    close = dists_sq < smooth_h**2
                    close_and_safe = close & (dists_sq > 1e-10)
                    if np.any(close_and_safe):
                        w_kernel = (1.0 - dists_sq[close_and_safe] / smooth_h**2)**2
                        reaction_scale = drag_coeff * speed * 0.005
                        reaction = reaction_scale * v[None, :] * w_kernel[:, None]
                        force[w_idx[close_and_safe]] += reaction

        # --- Water pressure (SPH-like repulsion) ---
        p_stiffness = 2000.0
        rest_dist_water = 0.035
        cell_size = smooth_h
        
        if len(w_idx) > 0:
            wp = pos[w_idx]
            grid_min = wp.min(axis=0) - cell_size
            cell_coords = np.floor((wp - grid_min) / cell_size).astype(int)
            
            grid = defaultdict(list)
            for li in range(len(cell_coords)):
                grid[tuple(cell_coords[li])].append(li)
            
            for li in range(len(cell_coords)):
                wi = w_idx[li]
                cc = cell_coords[li]
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        for dk in (-1, 0, 1):
                            key = (cc[0]+di, cc[1]+dj, cc[2]+dk)
                            for lj in grid.get(key, ()):
                                if lj <= li:
                                    continue
                                wj = w_idx[lj]
                                d = pos[wi] - pos[wj]
                                dist_sq = np.dot(d, d)
                                if dist_sq < cell_size**2 and dist_sq > 1e-12:
                                    dist_val = math.sqrt(dist_sq)
                                    overlap = rest_dist_water - dist_val
                                    if overlap > 0:
                                        f_mag = p_stiffness * overlap
                                    else:
                                        f_mag = -p_stiffness * 0.005 * overlap
                                    direction = d / dist_val
                                    rel_v = vel[wi] - vel[wj]
                                    visc = 1.5 * np.dot(rel_v, direction)
                                    f_scalar = f_mag + visc
                                    f_vec = f_scalar * direction
                                    force[wi] += f_vec
                                    force[wj] -= f_vec

        # --- Wall collisions ---
        for center, he, normal in self.walls:
            lo = center - he
            hi = center + he
            inside = (pos[:, 0] >= lo[0]) & (pos[:, 0] <= hi[0]) & \
                     (pos[:, 1] >= lo[1]) & (pos[:, 1] <= hi[1]) & \
                     (pos[:, 2] >= lo[2]) & (pos[:, 2] <= hi[2])
            idx_inside = np.where(inside)[0]
            for ii in idx_inside:
                best_pen = 1e10
                best_sign = 0
                best_ax = 0
                for ax in range(3):
                    pen_lo = pos[ii, ax] - lo[ax]
                    pen_hi = hi[ax] - pos[ii, ax]
                    if pen_lo < best_pen:
                        best_pen = pen_lo
                        best_sign = -1
                        best_ax = ax
                    if pen_hi < best_pen:
                        best_pen = pen_hi
                        best_sign = 1
                        best_ax = ax
                correction = -best_sign * best_pen
                pos[ii, best_ax] += correction
                if vel[ii, best_ax] * best_sign < 0:
                    vel[ii, best_ax] *= -0.2
                force[ii, best_ax] += -best_sign * 5000.0 * best_pen

        # --- Integrate (semi-implicit Euler) ---
        acc = force / mass[:, None]
        vel += acc * dt
        # Global damping
        vel *= 0.999
        pos += vel * dt

        # Clamp velocities
        self._clamp_velocities(5.0)

        # Track external impulse (gravity)
        ext_grav_torus = mass[self.torus_mask].sum() * self.gravity * dt
        ext_grav_water = mass[self.water_mask].sum() * self.gravity * dt
        self.cum_ext_impulse[0] += ext_grav_torus
        self.cum_ext_impulse[1] += ext_grav_water

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        while self.current_time < target - 1e-12:
            remaining = target - self.current_time
            step = min(self.dt, remaining)
            self._step(step)
            self.current_time += step

        return self._make_frame(target)

    # ------------------------------------------------------------------
    def _make_frame(self, time: float) -> StateFrame:
        pos = self.position.copy()
        vel = self.velocity.copy()
        mass = self.mass.copy()
        body_id = self.body_id.copy()

        n = self.n_total
        w_torus = np.zeros(n, dtype=float)
        w_torus[self.torus_mask] = 1.0
        w_water = np.zeros(n, dtype=float)
        w_water[self.water_mask] = 1.0
        domain_weights = np.stack([w_torus, w_water], axis=0)

        grav_torus = (mass[self.torus_mask].sum() * self.gravity).reshape(1, 1, 3)
        grav_water = (mass[self.water_mask].sum() * self.gravity).reshape(1, 1, 3)
        budget_ext_force = np.concatenate([grav_torus, grav_water], axis=0)

        # Interaction force: net buoyancy
        t_positions = pos[self.torus_mask]
        below_surface = t_positions[:, 2] < self.free_surface_z0
        if self.n_torus > 0:
            submerged_frac = float(below_surface.sum()) / self.n_torus
        else:
            submerged_frac = 0.0
        buoyancy_total = self.water_density * abs(self.gravity[2]) * self.torus_vol * submerged_frac
        interaction_force = np.array([[0.0, 0.0, buoyancy_total]])

        mass_quantity = mass.copy()
        cum_mass_source = np.zeros(1, dtype=float)

        frame = StateFrame(
            time=time,
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=body_id,
            attributes={
                "rest_position": self.rest_position.copy(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": np.zeros_like(self.boundary_normal),
                "boundary_velocity": np.zeros_like(self.boundary_normal),
                "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_mom.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_mom.copy(),
                "budget_domain_weights": domain_weights,
                "budget_external_force": budget_ext_force,
                "cumulative_mass_source": cum_mass_source,
                "interaction_force": interaction_force,
            },
            scalars={
                "reference_momentum": float(self.ref_momentum),
                "reference_velocity": float(self.ref_velocity),
                "water_free_surface_z": float(self.free_surface_z0),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "n_torus": self.n_torus,
            "n_water": self.n_water,
        }
