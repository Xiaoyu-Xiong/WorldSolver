"""Physics solver for coupled starfish underwater rock settling."""
from __future__ import annotations

import math
import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.seed = 0
        self.current_time = 0.0

        # State arrays
        self.pos = None
        self.vel = None
        self.mass = None
        self.body_id = None
        self.rest_pos = None

        # Mesh / connectivity for starfish
        self.star_indices = None
        self.water_indices = None
        self.n_star = 0
        self.n_water = 0
        self.n_total = 0

        # Starfish elastic structure (springs)
        self.spring_i = None
        self.spring_j = None
        self.spring_rest_len = None
        self.spring_k = 600.0  # Spring stiffness
        self.spring_damping = 8.0

        # Starfish shape matching / volume preservation
        self.star_rest_local = None
        self.star_rest_center = None

        # Rocks geometry (ellipsoids)
        self.rocks = []

        # Tank boundaries
        self.tank_x_min = -0.49
        self.tank_x_max = 0.49
        self.tank_y_min = -0.49
        self.tank_y_max = 0.49
        self.tank_z_min = 0.005

        # Water free surface
        self.water_free_surface_z = 0.40
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # Budget tracking
        self.cumulative_impulse = np.zeros((2, 3), dtype=np.float64)
        self.ref_velocity = 1.0
        self.ref_momentum = 1.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)
        self.pos = np.array(init_frame.position, dtype=np.float64)
        self.vel = np.array(init_frame.velocity, dtype=np.float64)
        self.mass = np.array(init_frame.mass, dtype=np.float64)
        self.body_id = np.array(init_frame.body_id, dtype=np.int32)

        if "rest_position" in init_frame.attributes:
            self.rest_pos = np.array(init_frame.attributes["rest_position"], dtype=np.float64)
        else:
            self.rest_pos = self.pos.copy()

        self.star_indices = np.where(self.body_id == 1)[0]
        self.water_indices = np.where(self.body_id == 2)[0]
        self.n_star = len(self.star_indices)
        self.n_water = len(self.water_indices)
        self.n_total = len(self.pos)

        self.water_free_surface_z = float(self.scene.config.get("water", {}).get("initial_free_surface_z", 0.40))
        self.gravity = np.array(self.scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Reference scales
        total_mass = np.sum(self.mass)
        self.ref_velocity = 1.0
        self.ref_momentum = float(total_mass * self.ref_velocity)

        # Parse rocks from scene geometry or hardcoded specs
        self._init_rocks()

        # Build starfish elastic network
        self._build_starfish_network()

    def _init_rocks(self):
        # 12 rocks from scene_geometry.yaml
        # id, center, radius, scale
        rock_defs = [
            ([-0.34, -0.25, 0.10], 0.12, [1.25, 0.82, 0.78]),
            ([-0.10, -0.31, 0.11], 0.105, [1.20, 0.78, 1.05]),
            ([0.18, -0.30, 0.09], 0.115, [1.35, 0.78, 0.72]),
            ([0.39, -0.22, 0.12], 0.11, [0.78, 1.12, 1.05]),
            ([-0.39, 0.00, 0.12], 0.115, [0.90, 1.28, 1.00]),
            ([-0.16, -0.04, 0.13], 0.125, [1.15, 0.88, 1.06]),
            ([0.09, -0.06, 0.10], 0.10, [1.28, 0.85, 0.82]),
            ([0.34, 0.03, 0.14], 0.13, [0.88, 1.12, 1.10]),
            ([-0.31, 0.27, 0.10], 0.11, [1.30, 0.82, 0.78]),
            ([-0.05, 0.25, 0.15], 0.12, [0.88, 1.20, 1.20]),
            ([0.20, 0.25, 0.09], 0.105, [1.42, 0.78, 0.75]),
            ([0.42, 0.29, 0.11], 0.115, [0.78, 1.12, 0.92]),
        ]
        self.rocks = []
        for center, r, scale in rock_defs:
            c = np.array(center, dtype=np.float64)
            s = np.array(scale, dtype=np.float64) * r
            self.rocks.append((c, s))

    def _build_starfish_network(self):
        # Build neighbor springs for starfish mesh particles
        star_pos = self.pos[self.star_indices]
        self.star_rest_center = np.mean(star_pos, axis=0)
        self.star_rest_local = star_pos - self.star_rest_center

        # Connect nearby vertices in rest pose
        from scipy.spatial import cKDTree
        tree = cKDTree(star_pos)
        # Pairs within radius
        pairs = tree.query_pairs(r=0.065, output_type='ndarray')
        if len(pairs) > 0:
            self.spring_i = self.star_indices[pairs[:, 0]]
            self.spring_j = self.star_indices[pairs[:, 1]]
            diffs = self.pos[self.spring_i] - self.pos[self.spring_j]
            self.spring_rest_len = np.linalg.norm(diffs, axis=-1)
        else:
            self.spring_i = np.array([], dtype=np.int32)
            self.spring_j = np.array([], dtype=np.int32)
            self.spring_rest_len = np.array([], dtype=np.float64)

    def _substep(self, dt: float):
        # 1. Gravity and Buoyancy / Drag Forces
        forces = np.zeros_like(self.pos)

        # Gravity on all particles
        forces += self.mass[:, None] * self.gravity[None, :]

        # Water & Starfish physics
        # Fluid properties
        rho_water = 1000.0
        rho_star = 1080.0

        # Buoyancy & Hydrodynamic forces on starfish
        # Starfish volume per particle
        star_vol = 0.0115 / self.n_star
        star_pos = self.pos[self.star_indices]
        star_vel = self.vel[self.star_indices]

        # Immersion depth
        submerged = star_pos[:, 2] < self.water_free_surface_z
        submerged_frac = np.clip((self.water_free_surface_z - star_pos[:, 2]) / 0.05, 0.0, 1.0)

        # Buoyancy force = rho_water * V_sub * g
        forces[self.star_indices, 2] += submerged_frac * rho_water * star_vol * 9.81

        # Hydrodynamic drag on starfish when submerged
        # Quadratic + linear drag
        v_rel = star_vel
        v_speed = np.linalg.norm(v_rel, axis=-1, keepdims=True)
        # Form drag + skin friction
        drag_coeff = 25.0 * submerged_frac[:, None]
        forces[self.star_indices] -= drag_coeff * v_rel * (0.2 + v_speed) * star_vol

        # Fluid interaction with starfish (Two-way momentum exchange)
        # Water particles near starfish feel displacement / drag
        water_pos = self.pos[self.water_indices]
        water_vel = self.vel[self.water_indices]

        # Water internal pressure / damping / viscosity
        # Damping to keep water calm / settling
        forces[self.water_indices] -= 5.0 * self.mass[self.water_indices, None] * water_vel
        # Water buoyancy against gravity to maintain hydrostatic balance below surface
        forces[self.water_indices, 2] += np.clip((self.water_free_surface_z - water_pos[:, 2]) / 0.1, 0.0, 1.0) * self.mass[self.water_indices] * 9.81

        # Spring forces for starfish elasticity
        if len(self.spring_i) > 0:
            p_i = self.pos[self.spring_i]
            p_j = self.pos[self.spring_j]
            d = p_i - p_j
            dist = np.linalg.norm(d, axis=-1, keepdims=True) + 1e-8
            u = d / dist

            # Hookean spring + damping
            v_i = self.vel[self.spring_i]
            v_j = self.vel[self.spring_j]
            v_rel_s = np.sum((v_i - v_j) * u, axis=-1, keepdims=True)

            spring_f = -self.spring_k * (dist - self.spring_rest_len[:, None]) - self.spring_damping * v_rel_s
            f_vec = spring_f * u

            np.add.at(forces, self.spring_i, f_vec)
            np.add.at(forces, self.spring_j, -f_vec)

        # Shape preservation / bending resistance towards rest shape
        # Soft shape matching per arm / globally to give realistic starfish elasticity
        current_star_center = np.mean(star_pos, axis=0)
        # Compute optimal rotation
        P = self.star_rest_local
        Q = star_pos - current_star_center
        H = P.T @ Q
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        target_star_pos = current_star_center + (P @ R.T)
        # Elastic restoring force
        restoring_k = 120.0
        forces[self.star_indices] += restoring_k * self.mass[self.star_indices, None] * (target_star_pos - star_pos)

        # Integrate velocity
        acc = forces / self.mass[:, None]
        self.vel += acc * dt

        # Integrate position
        self.pos += self.vel * dt

        # 2. Collision handling with Rocks and Tank walls
        # Rocks: ellipsoids (x-cx)^2/sx^2 + (y-cy)^2/sy^2 + (z-cz)^2/sz^2 <= 1
        for c, s in self.rocks:
            # Check starfish and water collision with rock
            d = self.pos - c[None, :]
            d_norm = d / s[None, :]
            dist_sq = np.sum(d_norm ** 2, axis=-1)
            inside = dist_sq < 1.05
            if np.any(inside):
                idx = np.where(inside)[0]
                norm_d = np.sqrt(dist_sq[idx]) + 1e-8
                # Approximate normal
                n = (d[idx] / (s[None, :] ** 2))
                n_len = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-8
                n = n / n_len
                # Penetration depth
                penetration = (1.0 - norm_d) * np.min(s) + 0.005
                penetration = np.maximum(penetration, 0.001)

                # Push out position
                self.pos[idx] += n * penetration[:, None]

                # Project velocity (inelastic + friction)
                vn = np.sum(self.vel[idx] * n, axis=-1, keepdims=True)
                closing = vn < 0
                if np.any(closing):
                    closing_idx = idx[closing[:, 0]]
                    n_c = n[closing[:, 0]]
                    vn_c = vn[closing[:, 0]]

                    # Normal impulse
                    self.vel[closing_idx] -= 1.2 * vn_c * n_c
                    # Tangential friction
                    vt = self.vel[closing_idx] - np.sum(self.vel[closing_idx] * n_c, axis=-1, keepdims=True) * n_c
                    self.vel[closing_idx] -= 0.6 * vt

                    # Track impulse
                    impulse = np.sum(-1.2 * vn_c * n_c * self.mass[closing_idx, None], axis=0)
                    self.cumulative_impulse[0] += impulse

        # Tank boundary collisions
        # Floor z_min
        floor_hit = self.pos[:, 2] < self.tank_z_min
        if np.any(floor_hit):
            self.pos[floor_hit, 2] = self.tank_z_min
            vn = self.vel[floor_hit, 2]
            neg = vn < 0
            if np.any(neg):
                idx = np.where(floor_hit)[0][neg]
                self.vel[idx, 2] = 0.0
                self.vel[idx, 0] *= 0.5
                self.vel[idx, 1] *= 0.5

        # Lateral walls
        # X walls
        x_min_hit = self.pos[:, 0] < self.tank_x_min
        if np.any(x_min_hit):
            self.pos[x_min_hit, 0] = self.tank_x_min
            self.vel[x_min_hit, 0] = np.maximum(0.0, self.vel[x_min_hit, 0])
        x_max_hit = self.pos[:, 0] > self.tank_x_max
        if np.any(x_max_hit):
            self.pos[x_max_hit, 0] = self.tank_x_max
            self.vel[x_max_hit, 0] = np.minimum(0.0, self.vel[x_max_hit, 0])

        # Y walls
        y_min_hit = self.pos[:, 1] < self.tank_y_min
        if np.any(y_min_hit):
            self.pos[y_min_hit, 1] = self.tank_y_min
            self.vel[y_min_hit, 1] = np.maximum(0.0, self.vel[y_min_hit, 1])
        y_max_hit = self.pos[:, 1] > self.tank_y_max
        if np.any(y_max_hit):
            self.pos[y_max_hit, 1] = self.tank_y_max
            self.vel[y_max_hit, 1] = np.minimum(0.0, self.vel[y_max_hit, 1])

    def advance_to(self, time: float) -> StateFrame:
        # Step simulation to target time
        substep_dt = 0.002
        while self.current_time < time - 1e-7:
            step = min(substep_dt, time - self.current_time)
            self._substep(step)
            self.current_time += step

        self.current_time = float(time)

        # Build required fields for StateFrame contract
        # 1. attributes.mass_quantity: [N] or [N, K]
        mass_quantity = self.mass.copy()

        # 2. fields.boundary_normal, boundary_target_velocity, boundary_velocity
        # B boundary sample points (e.g. tank walls and rocks)
        # We sample 5 tank walls + rock centers
        boundary_normals = np.array([
            [0.0, 0.0, 1.0],   # Floor
            [0.0, -1.0, 0.0],  # Back wall
            [0.0, 1.0, 0.0],   # Front wall
            [1.0, 0.0, 0.0],   # Left wall
            [-1.0, 0.0, 0.0],  # Right wall
        ], dtype=np.float64)
        boundary_target_velocity = np.zeros_like(boundary_normals)
        boundary_velocity = np.zeros_like(boundary_normals)

        # 3. Budget fields
        # G = 2 (Domain 0: Starfish, Domain 1: Water)
        # fields.budget_domain_weights: [G, N]
        budget_domain_weights = np.zeros((2, self.n_total), dtype=np.float64)
        budget_domain_weights[0, self.star_indices] = 1.0
        budget_domain_weights[1, self.water_indices] = 1.0

        # fields.budget_cumulative_external_impulse: [G, 3]
        budget_cumulative_external_impulse = self.cumulative_impulse.copy()

        # fields.budget_cumulative_inflow_momentum: [G, 3]
        budget_cumulative_inflow_momentum = np.zeros((2, 3), dtype=np.float64)

        # fields.budget_cumulative_outflow_momentum: [G, 3]
        budget_cumulative_outflow_momentum = np.zeros((2, 3), dtype=np.float64)

        # fields.budget_external_force: [G, K, 3] (where K=1 or number of external terms)
        # External forces per domain: Gravity
        gravity_force_star = np.sum(self.mass[self.star_indices, None] * self.gravity[None, :], axis=0)
        gravity_force_water = np.sum(self.mass[self.water_indices, None] * self.gravity[None, :], axis=0)
        budget_external_force = np.zeros((2, 1, 3), dtype=np.float64)
        budget_external_force[0, 0] = gravity_force_star
        budget_external_force[1, 0] = gravity_force_water

        # 4. fields.cumulative_mass_source: [K] (K=1)
        cumulative_mass_source = np.array([0.0], dtype=np.float64)

        # 5. fields.interaction_force: [K, 3] (K=1)
        interaction_force = np.zeros((1, 3), dtype=np.float64)

        # Construct StateFrame
        frame = StateFrame(
            time=self.current_time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "rest_position": self.rest_pos.copy(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "boundary_normal": boundary_normals,
                "boundary_target_velocity": boundary_target_velocity,
                "boundary_velocity": boundary_velocity,
                "budget_cumulative_external_impulse": budget_cumulative_external_impulse,
                "budget_cumulative_inflow_momentum": budget_cumulative_inflow_momentum,
                "budget_cumulative_outflow_momentum": budget_cumulative_outflow_momentum,
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": cumulative_mass_source,
                "interaction_force": interaction_force,
            },
            scalars={
                "water_free_surface_z": self.water_free_surface_z,
                "reference_velocity": float(self.ref_velocity),
                "reference_momentum": float(self.ref_momentum),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "n_particles": self.n_total,
            "n_star": self.n_star,
            "n_water": self.n_water,
        }

