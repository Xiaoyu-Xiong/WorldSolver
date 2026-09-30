"""Physically-based wet-towel solver using a mass-spring cloth and Lagrangian water particles.

The towel is modelled as a damped mass-spring network with stretch and bending
springs.  Water is represented by the initial particle set; each particle carries
a variable liquid mass and can exchange mass with the porous towel.  Two-way
coupling is achieved through (a) drag/viscous forces between free water and the
towel, and (b) reversible absorption/release of liquid mass governed by the
towel's dynamic storage capacity.  The end regions of the towel are driven by
the prescribed grip kinematics from the scene contract.

Only NumPy/SciPy are used; no Genesis code is imported or invoked.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver for the wet-towel lift-and-wring scene."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Solver protocol
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed

        # ------------------------------------------------------------------
        # Time parameters
        # ------------------------------------------------------------------
        self.duration = float(self.config["duration"])
        self.sample_dt = float(self.config["sample_dt"])
        controls = self.config["controls"]
        self.lift_cfg = controls["lift"]
        self.twist_cfg = controls["twist"]
        self.hold_until = float(controls["hold_after_twist_until"])

        # ------------------------------------------------------------------
        # Current state (will be advanced by _substep)
        # ------------------------------------------------------------------
        frame0 = scene.initial_state()
        self.time: float = 0.0
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.attributes = {
            name: np.asarray(value, dtype=float).copy()
            for name, value in frame0.attributes.items()
        }

        # Towel / water split
        self.towel_mask = self.body_id == 0
        self.water_mask = self.body_id == 1
        self.n_towel = int(self.towel_mask.sum())
        self.n_water = int(self.water_mask.sum())

        # ------------------------------------------------------------------
        # Towel geometry and mechanical data
        # ------------------------------------------------------------------
        towel_cfg = self.config["towel"]
        self.nx = int(towel_cfg["grid_shape"][0])
        self.ny = int(towel_cfg["grid_shape"][1])
        self.towel_size = [float(towel_cfg["size"][0]), float(towel_cfg["size"][1])]
        self.towel_area = self.towel_size[0] * self.towel_size[1]
        self.thickness = float(towel_cfg["thickness"])

        # Per-node dry mass (constant) and maximum storable liquid mass
        dry_areal = float(towel_cfg["dry_areal_density"])
        self.dry_mass = np.full(self.n_towel, dry_areal * self.towel_area / self.n_towel, dtype=float)
        storage_capacity = float(towel_cfg["liquid_storage_capacity_kg_m2"])
        self.max_stored = np.full(
            self.n_towel, storage_capacity * self.towel_area / self.n_towel, dtype=float
        )

        # Stored water per towel node (split out from total towel mass)
        self.stored = self.mass[: self.n_towel].copy() - self.dry_mass
        self.stored = np.clip(self.stored, 0.0, self.max_stored)
        self.mass[: self.n_towel] = self.dry_mass + self.stored

        # Rest positions for grip kinematics
        self.rest_pos = self.attributes["rest_position"][: self.n_towel].copy()

        # Build cloth stretch and bending element tables
        self._build_cloth_elements()

        # ------------------------------------------------------------------
        # Water data
        # ------------------------------------------------------------------
        water_cfg = self.config["water"]
        self.water_density = float(water_cfg["density"])
        self.water_particle_mass = self.water_density * (float(water_cfg["particle_spacing"]) ** 3)
        self.water_mass = self.mass[self.n_towel :].copy()
        # Free/hidden: mass > eps means free and active
        self.water_mass = np.clip(self.water_mass, 0.0, None)

        # ------------------------------------------------------------------
        # Physics parameters
        # ------------------------------------------------------------------
        self.gravity = np.asarray(self.config["gravity"], dtype=float)

        # Cloth: stretch and bending spring constants
        # Scale with a reference edge length so that cloth is reasonably
        # stiff but still allows the visible sag/twist.
        self.edge_stiffness = float(towel_cfg.get("edge_stiffness", 1.2e4))
        self.bend_stiffness = float(towel_cfg.get("bend_stiffness", 4.0e2))
        self.damping = float(towel_cfg.get("damping", 1.5))

        # Towel-water coupling
        self.drag_coeff = 0.25  # N per (m/s) of relative speed
        self.capture_rate = 12.0  # 1/s, effective absorption speed
        self.release_rate = 8.0  # 1/s, effective drainage speed
        self.capture_radius = 0.18
        self.release_radius = 0.25
        self.chunk_mass = 0.015  # kg released per transfer event
        self.release_velocity = np.array([0.0, 0.0, -1.2])  # m/s added to towel velocity

        # Basin (simple interior box)
        basin = self.config["basin"]
        self.basin_half = np.asarray(basin["interior_half_extents"], dtype=float)
        self.floor_z = 0.0
        self.basin_restitution = 0.2

        # ------------------------------------------------------------------
        # Grip kinematics
        # ------------------------------------------------------------------
        self.left_grip_x = -1.25
        self.right_grip_x = 1.25
        self.left_grasp = np.asarray(self.attributes["left_grasp"][: self.n_towel], dtype=bool)
        self.right_grasp = np.asarray(self.attributes["right_grasp"][: self.n_towel], dtype=bool)
        self.grasp_mask = self.left_grasp | self.right_grasp

        z0 = float(towel_cfg["initial_end_height"])
        self.left_center0 = np.array([self.left_grip_x, 0.0, z0])
        self.right_center0 = np.array([self.right_grip_x, 0.0, z0])

        # ------------------------------------------------------------------
        # Budget / diagnostic accumulators
        # ------------------------------------------------------------------
        # Two domains: 0 = towel, 1 = water.  Two materials: 0 = solid, 1 = liquid.
        self.G = 2
        self.K = 2
        self.budget_cumulative_ext_imp = np.zeros((self.G, 3), dtype=float)
        self.budget_cumulative_in = np.zeros((self.G, 3), dtype=float)
        self.budget_cumulative_out = np.zeros((self.G, 3), dtype=float)

        # Reference momentum scale (positive, frozen at init)
        total_mass = self.mass.sum()
        char_speed = np.sqrt(abs(self.gravity[2]) * float(towel_cfg["size"][0]))
        if char_speed <= 0.0 or not np.isfinite(char_speed):
            char_speed = 1.0
        self.reference_momentum = float(total_mass * char_speed)
        if self.reference_momentum <= 0.0 or not np.isfinite(self.reference_momentum):
            self.reference_momentum = 1.0

        # Substep size
        self.n_sub = 10
        self.dt_sub = self.sample_dt / self.n_sub

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1e-12:
            raise ValueError("Cannot advance backwards in time")
        n_steps = int(np.rint((target - self.time) / self.sample_dt))
        if n_steps < 0:
            n_steps = 0
        # If the caller requests a time that is not an integer multiple of
        # sample_dt, take fractional substeps to reach it exactly.
        for _ in range(n_steps):
            for _ in range(self.n_sub):
                self._substep(self.dt_sub)
        # Handle any remaining small drift
        remaining = target - self.time
        if remaining > 1e-12:
            self._substep(remaining)
            self.time = target
        else:
            self.time = target
        return self._make_frame()

    def diagnostics(self) -> dict:
        return {
            "solver": "mass_spring_cloth_lagrangian_water",
            "n_towel": self.n_towel,
            "n_water": self.n_water,
            "total_mass": float(self.mass.sum()),
            "stored_water_mass": float(self.stored.sum()),
            "free_water_mass": float(self.water_mass.sum()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_cloth_elements(self) -> None:
        """Construct edge (stretch) and bending springs for the towel grid."""
        pos = self.position[: self.n_towel]
        nx, ny = self.nx, self.ny

        edges = []

        # Horizontal edges
        for iy in range(ny):
            for ix in range(nx - 1):
                i = iy * nx + ix
                j = i + 1
                edges.append((i, j))
        # Vertical edges
        for iy in range(ny - 1):
            for ix in range(nx):
                i = iy * nx + ix
                j = i + nx
                edges.append((i, j))
        # Diagonals (consistent with scene.py triangulation)
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                a = iy * nx + ix
                edges.append((a, a + nx + 1))

        edges = np.asarray(edges, dtype=np.int32)
        i, j = edges[:, 0], edges[:, 1]
        dx = pos[j] - pos[i]
        rest = np.linalg.norm(dx, axis=1)

        # Spring stiffness proportional to 1/rest to avoid super-stiff short edges
        k = self.edge_stiffness * np.ones_like(rest)
        self.edge_i = i
        self.edge_j = j
        self.edge_rest = rest
        self.edge_k = k

        # Build face list
        faces = []
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                a = iy * nx + ix
                faces.extend([(a, a + 1, a + nx + 1), (a, a + nx + 1, a + nx)])
        faces = np.asarray(faces, dtype=np.int32)

        # Bending: for each interior edge, spring between opposite vertices
        edge_to_opp: dict[tuple[int, int], list[int]] = {}
        for f in faces:
            a, b, c = f
            for u, v, opp in [(a, b, c), (b, c, a), (c, a, b)]:
                key = (min(u, v), max(u, v))
                edge_to_opp.setdefault(key, []).append(opp)

        bend_i, bend_j, bend_rest = [], [], []
        for (u, v), opps in edge_to_opp.items():
            if len(opps) == 2:
                o1, o2 = opps
                bend_i.append(o1)
                bend_j.append(o2)
                d = pos[o2] - pos[o1]
                bend_rest.append(np.linalg.norm(d))

        bend_i = np.asarray(bend_i, dtype=np.int32)
        bend_j = np.asarray(bend_j, dtype=np.int32)
        bend_rest = np.asarray(bend_rest, dtype=float)
        self.bend_i = bend_i
        self.bend_j = bend_j
        self.bend_rest = bend_rest
        self.bend_k = np.full_like(bend_rest, self.bend_stiffness)

    def _grip_pose(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """Return target positions/velocities for towel grasp nodes at time."""
        kin = self.scene.grip_kinematics(time)

        # Left grip
        left_t = np.asarray(kin["grip_left"]["translation"], dtype=float)
        left_theta = float(kin["grip_left"]["rotation_about_x_radians"])
        # Right grip
        right_t = np.asarray(kin["grip_right"]["translation"], dtype=float)
        right_theta = float(kin["grip_right"]["rotation_about_x_radians"])

        # Rotation matrices about x
        cl, sl = np.cos(left_theta), np.sin(left_theta)
        cr, sr = np.cos(right_theta), np.sin(right_theta)
        Rl = np.array([[1.0, 0.0, 0.0],
                       [0.0, cl, -sl],
                       [0.0, sl, cl]])
        Rr = np.array([[1.0, 0.0, 0.0],
                       [0.0, cr, -sr],
                       [0.0, sr, cr]])

        # Target positions for grasp nodes using their rest local coordinates
        pos = self.rest_pos
        left_local = pos[self.left_grasp] - self.left_center0
        right_local = pos[self.right_grasp] - self.right_center0
        left_targets = left_t + left_local @ Rl.T
        right_targets = right_t + right_local @ Rr.T

        target_pos = np.zeros_like(pos)
        target_vel = np.zeros_like(pos)

        # Finite-difference grip velocity for the substep
        dt = self.dt_sub
        if dt > 0.0:
            kin_prev = self.scene.grip_kinematics(max(0.0, time - dt))
            left_t_prev = np.asarray(kin_prev["grip_left"]["translation"], dtype=float)
            left_theta_prev = float(kin_prev["grip_left"]["rotation_about_x_radians"])
            right_t_prev = np.asarray(kin_prev["grip_right"]["translation"], dtype=float)
            right_theta_prev = float(kin_prev["grip_right"]["rotation_about_x_radians"])

            clp, slp = np.cos(left_theta_prev), np.sin(left_theta_prev)
            crp, srp = np.cos(right_theta_prev), np.sin(right_theta_prev)
            Rl_prev = np.array([[1.0, 0.0, 0.0],
                                [0.0, clp, -slp],
                                [0.0, slp, clp]])
            Rr_prev = np.array([[1.0, 0.0, 0.0],
                                [0.0, crp, -srp],
                                [0.0, srp, crp]])

            left_targets_prev = left_t_prev + left_local @ Rl_prev.T
            right_targets_prev = right_t_prev + right_local @ Rr_prev.T

            target_vel[self.left_grasp] = (left_targets - left_targets_prev) / dt
            target_vel[self.right_grasp] = (right_targets - right_targets_prev) / dt

        target_pos[self.left_grasp] = left_targets
        target_pos[self.right_grasp] = right_targets
        return target_pos, target_vel

    def _capacity_factor(self, time: float) -> float:
        """Reduction factor for liquid storage during wringing.

        Before wringing the towel can hold its full capacity.  During the
        twist phase the factor drops to a small residual, and it stays there
        so that water continues to drip after twisting stops.
        """
        twist = self.twist_cfg
        t0, t1 = float(twist["start_time"]), float(twist["end_time"])
        if time < t0:
            return 1.0
        if time > t1:
            return 0.05
        u = (time - t0) / (t1 - t0)
        u = max(0.0, min(1.0, u))
        smooth = u * u * (3.0 - 2.0 * u)
        return 1.0 - 0.95 * smooth

    def _spring_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Vectorized evaluation of cloth stretch + bending forces."""
        force = np.zeros_like(pos)

        # Stretch
        i, j = self.edge_i, self.edge_j
        dx = pos[j] - pos[i]
        dist = np.linalg.norm(dx, axis=1)
        dist[dist < 1e-12] = 1e-12
        dir_vec = dx / dist[:, None]
        rel_v = vel[j] - vel[i]
        fs = self.edge_k * (dist - self.edge_rest) + self.damping * np.einsum("ij,ij->i", rel_v, dir_vec)
        fvec = fs[:, None] * dir_vec

        np.add.at(force, i, fvec)
        np.add.at(force, j, -fvec)

        # Bending
        if len(self.bend_i):
            i, j = self.bend_i, self.bend_j
            dx = pos[j] - pos[i]
            dist = np.linalg.norm(dx, axis=1)
            dist[dist < 1e-12] = 1e-12
            dir_vec = dx / dist[:, None]
            rel_v = vel[j] - vel[i]
            fb = self.bend_k * (dist - self.bend_rest) + self.damping * np.einsum("ij,ij->i", rel_v, dir_vec)
            fvec = fb[:, None] * dir_vec
            np.add.at(force, i, fvec)
            np.add.at(force, j, -fvec)

        return force

    def _substep(self, dt: float) -> None:
        """Advance by a single sub-step using semi-implicit Euler."""
        nt = self.n_towel
        pos_t = self.position[:nt]
        vel_t = self.velocity[:nt]
        pos_w = self.position[nt:]
        vel_w = self.velocity[nt:]

        # ------------------------------------------------------------------
        # Cloth internal forces (elastic + bending + gravity)
        # ------------------------------------------------------------------
        cloth_force = self._spring_forces(pos_t, vel_t)
        towel_mass = self.dry_mass + self.stored
        cloth_force += towel_mass[:, None] * self.gravity[None, :]

        # ------------------------------------------------------------------
        # Towel-water drag and mass transfer
        # ------------------------------------------------------------------
        # Build a tree of towel nodes for near-neighbor queries
        tree = cKDTree(pos_t)

        # Free water particle indices and current data
        free_mask = self.water_mass > 1e-9
        free_indices = np.nonzero(free_mask)[0]

        interaction_solid = np.zeros(3, dtype=float)
        interaction_liquid = np.zeros(3, dtype=float)

        # Drag forces + capture from free water -> towel
        node_momentum_gain = np.zeros((nt, 3), dtype=float)
        node_mass_gain = np.zeros(nt, dtype=float)
        capture_in = np.zeros(3, dtype=float)
        capture_out = np.zeros(3, dtype=float)

        if free_indices.size:
            # Map local free index -> global water index
            p = pos_w[free_indices]
            v = vel_w[free_indices]
            masses = self.water_mass[free_indices]

            # k-nearest towel nodes per particle for capture distribution
            # (small k for speed; interactions are local)
            dists, idx = tree.query(p, k=min(8, nt), distance_upper_bound=self.capture_radius)
            if idx.ndim == 1:
                idx = idx[:, None]
                dists = dists[:, None]

            valid = idx < nt
            for local_i, glob in enumerate(free_indices):
                drow = dists[local_i]
                irow = idx[local_i]
                valid_mask = valid[local_i]
                if not np.any(valid_mask):
                    continue
                nodes = irow[valid_mask]
                ds = drow[valid_mask]
                v_part = v[local_i]
                m_part = masses[local_i]

                # Drag on the water from the nearest towel node
                nearest = nodes[np.argmin(ds)]
                v_node = vel_t[nearest]
                f_drag = self.drag_coeff * (v_node - v_part)
                # Apply drag to water particle and to towel node
                cloth_force[nearest] -= f_drag
                vel_w[glob] = v_part + (f_drag / max(m_part, 1e-12)) * dt

                interaction_solid -= f_drag
                interaction_liquid += f_drag

                # Capture: distribute transferred mass over the available nodes
                available = self.max_stored[nodes] - self.stored[nodes]
                available = np.clip(available, 0.0, None)
                total_avail = available.sum()
                if total_avail <= 0.0 or m_part <= 0.0:
                    continue

                # Weight by inverse distance and remaining capacity
                weights = available / (ds + 0.01)
                wsum = weights.sum()
                if wsum <= 0.0:
                    continue
                weights /= wsum

                max_transfer = min(m_part, dt * self.capture_rate * total_avail)
                transfer = max_transfer * weights

                # Update particle and node bookkeeping
                self.water_mass[glob] = m_part - transfer.sum()
                np.add.at(node_mass_gain, nodes, transfer)
                np.add.at(node_momentum_gain, nodes, transfer[:, None] * v_part)
                capture_in += (transfer.sum() * v_part)
                capture_out += (transfer.sum() * v_part)

        # Apply captured mass to towel nodes, preserving momentum
        if np.any(node_mass_gain > 0.0):
            dm = node_mass_gain
            old_mass = towel_mass
            new_mass = old_mass + dm
            # Avoid divide by zero
            new_mass = np.where(new_mass > 0.0, new_mass, 1.0)
            vel_t[:] = (
                old_mass[:, None] * vel_t + node_momentum_gain
            ) / new_mass[:, None]
            self.stored += dm
            towel_mass = self.dry_mass + self.stored
            self.mass[:nt] = towel_mass

        # ------------------------------------------------------------------
        # Release: towel -> water when storage exceeds dynamic capacity
        # ------------------------------------------------------------------
        cap_factor = self._capacity_factor(self.time)
        dynamic_max = self.max_stored * cap_factor
        excess = np.clip(self.stored - dynamic_max, 0.0, None)
        release_in = np.zeros(3, dtype=float)
        release_out = np.zeros(3, dtype=float)

        if np.any(excess > 0.0):
            # Candidates: water particles we can revive or grow (any particle)
            all_water_indices = np.arange(self.n_water)
            # Query nearest candidate for each node with excess
            # For efficiency, process only a subset per substep
            release_nodes = np.nonzero(excess > 1e-6)[0]
            if release_nodes.size:
                # Use the water positions (including currently hidden ones)
                # Hidden particles are kept below the basin; only revive those near the node.
                # Query nearest towel node for all water particles:
                dists_to_towel, idx_to_towel = tree.query(self.position[nt:], k=1)
                for node in release_nodes:
                    rel = min(excess[node], dt * self.release_rate * self.stored[node])
                    if rel <= 1e-9:
                        continue
                    # Find water particles near this node
                    near = np.nonzero(dists_to_towel < self.release_radius)[0]
                    if near.size == 0:
                        continue
                    # Choose the least massive nearby particle to avoid runaway growth
                    chosen = near[np.argmin(self.water_mass[near])]
                    # Grow the chosen particle; cap at the nominal particle mass so that
                    # multiple distinct droplets can be formed over time.
                    room = max(0.0, self.water_particle_mass - self.water_mass[chosen])
                    add = min(rel, room, self.chunk_mass)
                    if add <= 1e-9:
                        continue
                    self.water_mass[chosen] += add
                    self.stored[node] -= add

                    # Eject droplet below the towel with release velocity
                    v_node = vel_t[node]
                    eject_dir = np.array([0.0, 0.0, -1.0])
                    # Add a small outward horizontal component based on side
                    if self.rest_pos[node, 0] < -0.1:
                        eject_dir = eject_dir + np.array([-0.15, 0.0, 0.0])
                    elif self.rest_pos[node, 0] > 0.1:
                        eject_dir = eject_dir + np.array([0.15, 0.0, 0.0])
                    eject_dir[2] = -1.0
                    eject_dir = eject_dir / (np.linalg.norm(eject_dir) + 1e-12)
                    v_rel = v_node + np.linalg.norm(v_node) * 0.1 * eject_dir + self.release_velocity
                    vel_w[chosen] = v_rel
                    # Place just below the node
                    pos_w[chosen] = pos_t[node] + eject_dir * 0.03

                    release_out += add * v_node
                    release_in += add * (v_node + self.release_velocity)

        # ------------------------------------------------------------------
        # Basin collision for water particles
        # ------------------------------------------------------------------
        water_pos = self.position[nt:]
        water_vel = self.velocity[nt:]
        active = self.water_mass > 1e-9
        # Floor
        below = (water_pos[:, 2] < self.floor_z) & active
        if np.any(below):
            water_pos[below, 2] = self.floor_z
            water_vel[below, 2] = -self.basin_restitution * water_vel[below, 2]
        # Walls
        for axis, bound in [(0, self.basin_half[0]), (1, self.basin_half[1])]:
            too_lo = (water_pos[:, axis] < -bound) & active
            too_hi = (water_pos[:, axis] > bound) & active
            if np.any(too_lo):
                water_pos[too_lo, axis] = -bound
                water_vel[too_lo, axis] = abs(water_vel[too_lo, axis]) * self.basin_restitution
            if np.any(too_hi):
                water_pos[too_hi, axis] = bound
                water_vel[too_hi, axis] = -abs(water_vel[too_hi, axis]) * self.basin_restitution

        # ------------------------------------------------------------------
        # Integrate towel
        # ------------------------------------------------------------------
        # Ensure towel mass is current
        towel_mass = self.dry_mass + self.stored
        self.mass[:nt] = towel_mass

        # Semi-implicit Euler
        acc_t = cloth_force / towel_mass[:, None]
        vel_t += acc_t * dt
        pos_t += vel_t * dt

        # Floor collision for towel (basin floor)
        below_t = pos_t[:, 2] < self.floor_z
        pos_t[below_t, 2] = self.floor_z
        vel_t[below_t, 2] = np.maximum(0.0, vel_t[below_t, 2])

        # ------------------------------------------------------------------
        # Grip constraints: override end regions
        # ------------------------------------------------------------------
        grip_pos, grip_vel = self._grip_pose(self.time + dt)
        grasp = self.grasp_mask
        pos_t[grasp] = grip_pos[grasp]
        vel_t[grasp] = grip_vel[grasp]

        # ------------------------------------------------------------------
        # Integrate free water particles (gravity)
        # ------------------------------------------------------------------
        acc_w = self.gravity[None, :].copy()
        # Only integrate particles with mass
        active = self.water_mass > 1e-9
        m_inv = np.where(active, 1.0 / np.maximum(self.water_mass, 1e-12), 0.0)
        vel_w[active] += acc_w * dt
        pos_w[active] += vel_w[active] * dt

        # Hide zero-mass water particles below the basin so they do not render
        hidden = self.water_mass <= 1e-9
        if np.any(hidden):
            pos_w[hidden, 0] = 0.0
            pos_w[hidden, 1] = 0.0
            pos_w[hidden, 2] = -2.0
            vel_w[hidden] = 0.0

        # ------------------------------------------------------------------
        # Update global state arrays
        # ------------------------------------------------------------------
        self.position[:nt] = pos_t
        self.velocity[:nt] = vel_t
        self.position[nt:] = pos_w
        self.velocity[nt:] = vel_w
        self.mass[nt:] = self.water_mass

        # ------------------------------------------------------------------
        # Budget bookkeeping
        # ------------------------------------------------------------------
        # External gravity forces per domain/material
        ext_force = np.zeros((self.G, self.K, 3), dtype=float)
        # Domain 0 (towel)
        ext_force[0, 0] = (self.dry_mass * self.gravity).sum(axis=0)
        ext_force[0, 1] = (self.stored * self.gravity).sum(axis=0)
        # Domain 1 (water)
        ext_force[1, 1] = (self.water_mass * self.gravity).sum(axis=0)

        # Cumulative external impulse (gravity + grip would also be external,
        # but grip is not represented as a continuous force here)
        self.budget_cumulative_ext_imp += ext_force.sum(axis=1) * dt

        # Inflow/outflow momentum from mass transfer
        self.budget_cumulative_in[0] += capture_in + release_in
        self.budget_cumulative_in[1] += release_in + capture_in
        self.budget_cumulative_out[1] += capture_out + release_out
        self.budget_cumulative_out[0] += release_out + capture_out

        # Domain weights: towel nodes vs water particles
        weights = np.zeros((self.G, nt + self.n_water), dtype=float)
        weights[0, :nt] = 1.0
        weights[1, nt:] = 1.0

        # Interaction force: paired towel <-> water drag
        interaction = np.zeros((self.K, 3), dtype=float)
        interaction[0] = interaction_solid
        interaction[1] = interaction_liquid

        # Save for frame construction
        self._last_ext_force = ext_force
        self._last_weights = weights
        self._last_interaction = interaction

        self.time += dt

    def _make_frame(self) -> StateFrame:
        nt = self.n_towel
        # Total per-sample mass
        total_mass = np.empty_like(self.mass)
        total_mass[:nt] = self.dry_mass + self.stored
        total_mass[nt:] = self.water_mass

        # Mass split into materials: [solid, liquid]
        mass_quantity = np.zeros((self.n_towel + self.n_water, 2), dtype=float)
        mass_quantity[:nt, 0] = self.dry_mass
        mass_quantity[:nt, 1] = self.stored
        mass_quantity[nt:, 1] = self.water_mass

        # Budget fields
        budget_external_force = self._last_ext_force
        budget_cumulative_external_impulse = self.budget_cumulative_ext_imp.copy()
        budget_cumulative_inflow_momentum = self.budget_cumulative_in.copy()
        budget_cumulative_outflow_momentum = self.budget_cumulative_out.copy()
        budget_domain_weights = self._last_weights
        cumulative_mass_source = np.zeros(self.K, dtype=float)
        interaction_force = self._last_interaction

        # Attributes: keep original plus add mass_quantity
        attrs = {k: v.copy() for k, v in self.attributes.items()}
        attrs["mass_quantity"] = mass_quantity

        # Scalars
        scalars = dict(self.config.get("scalars", {}))
        scalars["reference_momentum"] = self.reference_momentum

        frame = StateFrame(
            time=self.time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=total_mass,
            body_id=self.body_id,
            attributes=attrs,
            fields={
                "budget_cumulative_external_impulse": budget_cumulative_external_impulse,
                "budget_cumulative_inflow_momentum": budget_cumulative_inflow_momentum,
                "budget_cumulative_outflow_momentum": budget_cumulative_outflow_momentum,
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": cumulative_mass_source,
                "interaction_force": interaction_force,
            },
            scalars=scalars,
        )
        frame.validate()
        return frame
