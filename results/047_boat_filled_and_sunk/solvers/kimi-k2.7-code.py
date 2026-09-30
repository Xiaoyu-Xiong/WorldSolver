"""Physically-based solver for the boat-filled-and-sunk scene.

The implementation uses an explicit time-stepping particle scheme for the
water and a single-degree-of-freedom rigid-body model for the open boat.
Water particles feel gravity, tank-boundary penalties and the moving boat
hull; the boat feels gravity, hydrostatic buoyancy, bottom contact, damping
and the equal-and-opposite contact forces from the water.  All required
StateFrame fields are produced from the current simulation state.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver implementing the fill-and-sink multiphysics scene."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = getattr(scene, "config", {}) or {}

        init = scene.initial_state()
        self.positions = np.asarray(init.position, dtype=float).copy()
        self.velocities = np.asarray(init.velocity, dtype=float).copy()
        self.mass = np.asarray(init.mass, dtype=float).copy()
        self.body_id = np.asarray(init.body_id, dtype=np.int32).copy()
        self.orientation = np.asarray(init.orientation, dtype=float).copy()
        self.angular_velocity = np.asarray(init.angular_velocity, dtype=float).copy()
        self.phase_id = np.asarray(init.attributes["phase_id"], dtype=np.int32).copy()
        self.water_region = np.asarray(init.attributes["water_region"], dtype=np.int32).copy()

        self.time = float(init.time)
        self.boat_idx = int(np.where(self.body_id == 0)[0][0])
        self.boat_pos = self.positions[self.boat_idx].copy()
        self.boat_vel = self.velocities[self.boat_idx].copy()

        cfg = self.config
        self.gravity = np.array(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        boat_cfg = cfg.get("boat", {})
        self.boat_mass_val = float(boat_cfg.get("mass", 82.0))
        self.target_draft = float(boat_cfg.get("target_initial_draft", 0.28))
        self.boat_thickness = float(boat_cfg.get("thin_shell_thickness", 0.018))

        fluid_cfg = cfg.get("fluid_domain", {})
        self.eta = float(fluid_cfg.get("initial_free_surface_z", 1.65))
        self.rho = float(fluid_cfg.get("water_density", 1000.0))

        inlet_cfg = cfg.get("inlet", {})
        self.inlet_center = np.array(inlet_cfg.get("center", [0.0, 0.0, 3.35]), dtype=float)
        self.inlet_radius = float(inlet_cfg.get("radius", 0.18))
        self.inlet_vel = np.array(inlet_cfg.get("velocity", [0.0, 0.0, -2.6]), dtype=float)
        active = inlet_cfg.get("active_interval", [0.35, 6.2])
        self.inlet_t0 = float(active[0])
        self.inlet_t1 = float(active[1])
        self.inlet_flow = float(inlet_cfg.get("volumetric_flow_rate", 0.055))

        # Tank interior extents (matches scene_geometry.yaml).
        self.tank_min = np.array([-2.28, -1.68, 0.0], dtype=float)
        self.tank_max = np.array([2.28, 1.68, 3.8], dtype=float)

        # Boat prismatic hull chosen so that the empty boat floats at the
        # prescribed draft.  L*W = m / (rho * draft).  Keep a plausible aspect.
        self.boat_half_L = 0.30
        base_area = self.boat_mass_val / (self.rho * self.target_draft)
        self.boat_half_W = base_area / (4.0 * self.boat_half_L)
        # Total height of the open boat (visual/physical hull).
        self.boat_H = 0.90
        self.boat_half_H = 0.5 * self.boat_H

        # Particle data derived from the initial scaffold.
        self.spacing = 0.18
        self.vol_particle = self.spacing ** 3
        self.mass_particle = self.rho * self.vol_particle

        # Time-stepping.
        self.dt_sub = 0.001

        # Contact/penalty parameters.
        self.k_wall = 2.0e5
        self.c_wall = 2.0e3
        self.k_boat = 2.0e5
        self.c_boat = 2.0e3
        self.k_bottom = 5.0e6
        self.c_bottom = 1.0e4
        self.c_damp = 0.8
        self.c_boat_damp = 5.0
        self.v_max = 15.0

        # Source tracking.
        self.flow_mass_rate = self.rho * self.inlet_flow
        self.emit_counter = 0.0
        self.cum_source = np.zeros(2, dtype=float)

        # Deterministic RNG for inlet jitter.
        self.rng = np.random.default_rng(seed)

        # Ensure boat mass is exact.
        self.mass[self.boat_idx] = self.boat_mass_val

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target <= self.time + 1e-12:
            return self._build_frame()

        dt = self.dt_sub
        # Whole substeps.
        while self.time + dt < target:
            self._substep(dt)
        # Final partial substep to land exactly on target.
        remaining = target - self.time
        if remaining > 1e-12:
            self._substep(remaining)
        return self._build_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "BenchmarkSolver",
            "particles": int(len(self.positions)),
            "water_mass_source_kg": float(self.cum_source[1]),
        }

    # ------------------------------------------------------------------ #
    # Substepping
    # ------------------------------------------------------------------ #
    def _substep(self, dt: float) -> None:
        # Prescribed inlet source.
        self._emit_inlet(dt)

        water_mask = self.body_id == 1

        # External forces on water: gravity + tank boundaries.
        f_water = np.zeros_like(self.positions)
        if np.any(water_mask):
            f_water[water_mask] += self.mass[water_mask][:, None] * self.gravity[None, :]
            f_water += self._tank_forces(self.positions, self.velocities)

        # Interaction: boat hull <-> water contact.
        f_hull_on_water, boat_contact = self._boat_hull_forces(
            self.positions, self.velocities
        )

        # Update water particles (semi-implicit Euler).
        total_f = f_water + f_hull_on_water
        # Add mild viscous drag.
        total_f[water_mask] -= self.c_damp * self.velocities[water_mask]
        accel = total_f[water_mask] / self.mass[water_mask][:, None]
        self.velocities[water_mask] += accel * dt
        self.positions[water_mask] += self.velocities[water_mask] * dt

        # Velocity cap to avoid integration blow-up.
        v2 = np.sum(self.velocities[water_mask] ** 2, axis=1)
        over = v2 > self.v_max ** 2
        if np.any(over):
            scale = self.v_max / np.sqrt(v2[over])
            self.velocities[water_mask][over] *= scale[:, None]

        # Update boat.
        boat_ext = self._boat_external_force()
        total_boat_f = boat_ext + boat_contact
        # Constrain horizontal motion: the tank and scene are symmetric.
        total_boat_f[:2] = 0.0
        self.boat_vel += total_boat_f / self.boat_mass_val * dt
        self.boat_vel[:2] = 0.0
        self.boat_pos += self.boat_vel * dt

        # Keep boat within tank horizontally.
        self.boat_pos[0] = 0.0
        self.boat_pos[1] = 0.0

        # Sync arrays.
        self.positions[self.boat_idx] = self.boat_pos
        self.velocities[self.boat_idx] = self.boat_vel

        self.time += dt

    # ------------------------------------------------------------------ #
    # Inlet emission
    # ------------------------------------------------------------------ #
    def _emit_inlet(self, dt: float) -> None:
        t0 = self.time
        t1 = self.time + dt
        active_start = max(t0, self.inlet_t0)
        active_end = min(t1, self.inlet_t1)
        dt_active = max(0.0, active_end - active_start)
        if dt_active <= 0.0:
            return

        rate = self.inlet_flow / self.vol_particle
        self.emit_counter += rate * dt_active
        n_emit = int(self.emit_counter)
        if n_emit <= 0:
            return
        self.emit_counter -= n_emit

        # Uniform disk distribution (deterministic given the seed).
        u = self.rng.random(n_emit, dtype=float)
        v = self.rng.random(n_emit, dtype=float)
        r = self.inlet_radius * np.sqrt(u)
        theta = 2.0 * np.pi * v
        xs = r * np.cos(theta)
        ys = r * np.sin(theta)
        zs = np.full(n_emit, self.inlet_center[2], dtype=float)
        new_pos = np.column_stack([xs, ys, zs])
        new_vel = np.tile(self.inlet_vel, (n_emit, 1))

        self.positions = np.vstack([self.positions, new_pos])
        self.velocities = np.vstack([self.velocities, new_vel])
        self.mass = np.concatenate([self.mass, np.full(n_emit, self.mass_particle, dtype=float)])
        self.body_id = np.concatenate([self.body_id, np.full(n_emit, 1, dtype=np.int32)])
        self.phase_id = np.concatenate([self.phase_id, np.full(n_emit, 1, dtype=np.int32)])
        self.water_region = np.concatenate([self.water_region, np.full(n_emit, 3, dtype=np.int32)])

        orient = np.zeros((n_emit, 4), dtype=float)
        orient[:, 3] = 1.0
        self.orientation = np.vstack([self.orientation, orient])
        self.angular_velocity = np.vstack(
            [self.angular_velocity, np.zeros((n_emit, 3), dtype=float)]
        )

        self.cum_source[1] += n_emit * self.mass_particle

    # ------------------------------------------------------------------ #
    # Force helpers
    # ------------------------------------------------------------------ #
    def _halfspace_force(self, p: np.ndarray, v: np.ndarray, n: np.ndarray, offset: float) -> np.ndarray:
        """Penalty force for half-space n·x >= offset (n points into domain)."""
        d = p @ n - offset
        pen = -d
        active = pen > 0.0
        f = np.zeros_like(p)
        if not np.any(active):
            return f
        f_mag = self.k_wall * pen[active] - self.c_wall * (v[active] @ n)
        f[active] = f_mag[:, None] * n
        return f

    def _tank_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        f = np.zeros_like(pos)
        water_mask = self.body_id == 1
        if not np.any(water_mask):
            return f
        pw = pos[water_mask]
        vw = vel[water_mask]
        # x walls
        f[water_mask] += self._halfspace_force(pw, vw, np.array([1.0, 0.0, 0.0]), self.tank_min[0])
        f[water_mask] += self._halfspace_force(pw, vw, np.array([-1.0, 0.0, 0.0]), self.tank_max[0])
        # y walls
        f[water_mask] += self._halfspace_force(pw, vw, np.array([0.0, 1.0, 0.0]), self.tank_min[1])
        f[water_mask] += self._halfspace_force(pw, vw, np.array([0.0, -1.0, 0.0]), self.tank_max[1])
        # floor
        f[water_mask] += self._halfspace_force(pw, vw, np.array([0.0, 0.0, 1.0]), self.tank_min[2])
        return f

    def _boat_hull_forces(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (force_on_water, contact_force_on_boat) for the moving hull."""
        f = np.zeros_like(pos)
        boat_contact = np.zeros(3, dtype=float)
        water_mask = self.body_id == 1
        if not np.any(water_mask):
            return f, boat_contact

        rel = pos[water_mask] - self.boat_pos
        Lx = self.boat_half_L
        Ly = self.boat_half_W
        t = self.boat_thickness
        hh = self.boat_half_H
        Lx_in = Lx - t
        Ly_in = Ly - t

        in_x = np.abs(rel[:, 0]) < Lx
        in_y = np.abs(rel[:, 1]) < Ly
        in_lateral = in_x & in_y

        # Floor (bottom of the hull, including interior floor).
        floor_local = (rel[:, 2] < -hh + t) & in_lateral
        if np.any(floor_local):
            water_idx = np.where(water_mask)[0]
            floor_idx = water_idx[floor_local]
            penetration = (-hh + t) - rel[floor_local, 2]
            n = np.array([0.0, 0.0, 1.0], dtype=float)
            rel_v = vel[floor_idx] - self.boat_vel
            f_mag = self.k_boat * penetration - self.c_boat * (rel_v @ n)
            f_vec = np.zeros((floor_idx.size, 3), dtype=float)
            f_vec[:, 2] = f_mag
            f[floor_idx] += f_vec
            boat_contact -= np.sum(f_vec, axis=0)

        # Side walls (vertical shell, open at the top).
        in_vertical = (rel[:, 2] > -hh) & (rel[:, 2] < hh)
        inside_lateral = (np.abs(rel[:, 0]) < Lx_in) & (np.abs(rel[:, 1]) < Ly_in)
        wall_local = in_vertical & (~inside_lateral) & in_lateral
        if np.any(wall_local):
            water_idx = np.where(water_mask)[0]
            wall_idx = water_idx[wall_local]
            rxy = rel[wall_local, :2]
            cx = np.clip(rxy[:, 0], -Lx_in, Lx_in)
            cy = np.clip(rxy[:, 1], -Ly_in, Ly_in)
            diff_xy = np.column_stack([rxy[:, 0] - cx, rxy[:, 1] - cy])
            dist = np.linalg.norm(diff_xy, axis=1)
            valid = dist > 1e-9
            if np.any(valid):
                wall_idx = wall_idx[valid]
                diff_xy = diff_xy[valid]
                dist = dist[valid]
                n_vec = np.zeros((wall_idx.size, 3), dtype=float)
                n_vec[:, :2] = diff_xy / dist[:, None]
                rel_v = vel[wall_idx] - self.boat_vel
                vn = np.einsum("ij,ij->i", rel_v, n_vec)
                f_mag = self.k_boat * dist - self.c_boat * vn
                f_vec = f_mag[:, None] * n_vec
                f[wall_idx] += f_vec
                boat_contact -= np.sum(f_vec, axis=0)

        return f, boat_contact

    def _boat_external_force(self) -> np.ndarray:
        g = self.gravity
        # Gravity on the rigid boat hull.
        f = self.boat_mass_val * g.copy()

        # Hydrostatic buoyancy based on the volume of the hull below the
        # external free surface.
        bottom_z = self.boat_pos[2] - self.boat_half_H
        h_sub = np.clip(self.eta - bottom_z, 0.0, self.boat_H)
        base_area = 4.0 * self.boat_half_L * self.boat_half_W
        buoyancy_mag = self.rho * (-g[2]) * base_area * h_sub
        f[2] += buoyancy_mag

        # Bottom collision with the tank floor.
        if bottom_z < 0.0:
            penetration = -bottom_z
            rel_v = self.boat_vel[2]
            f[2] += self.k_bottom * penetration - self.c_bottom * rel_v

        # Damping.
        f -= self.c_boat_damp * self.boat_vel
        return f

    # ------------------------------------------------------------------ #
    # Frame construction
    # ------------------------------------------------------------------ #
    def _signed_distance(self, pos: np.ndarray) -> np.ndarray:
        """Signed geometric gap to the nearest tank/boat boundary (negative = penetration)."""
        N = pos.shape[0]
        sd = np.full(N, 1e9, dtype=float)
        # Tank walls/floor.
        sd = np.minimum(sd, pos[:, 0] - self.tank_min[0])
        sd = np.minimum(sd, self.tank_max[0] - pos[:, 0])
        sd = np.minimum(sd, pos[:, 1] - self.tank_min[1])
        sd = np.minimum(sd, self.tank_max[1] - pos[:, 1])
        sd = np.minimum(sd, pos[:, 2] - self.tank_min[2])
        # Boat outer box proxy.
        rel = pos - self.boat_pos
        d_box = np.maximum(
            np.maximum(
                np.abs(rel[:, 0]) - self.boat_half_L,
                np.abs(rel[:, 1]) - self.boat_half_W,
            ),
            np.abs(rel[:, 2]) - self.boat_half_H,
        )
        sd = np.minimum(sd, d_box)
        return sd

    def _external_force_frame(self) -> np.ndarray:
        f = np.zeros_like(self.positions)
        water_mask = self.body_id == 1
        if np.any(water_mask):
            f[water_mask] += self.mass[water_mask][:, None] * self.gravity[None, :]
            f += self._tank_forces(self.positions, self.velocities)
        f[self.boat_idx] = self._boat_external_force()
        return f

    def _interaction_force_frame(self) -> np.ndarray:
        f_hull, boat_contact = self._boat_hull_forces(self.positions, self.velocities)
        total_on_water = np.sum(f_hull, axis=0)
        # material 0 = boat, material 1 = water
        return np.vstack([boat_contact, total_on_water])

    def _build_frame(self) -> StateFrame:
        # Ensure orientation/angular velocity arrays match current particle count.
        N = self.positions.shape[0]
        if self.orientation.shape[0] != N:
            old_n = self.orientation.shape[0]
            new_orient = np.zeros((N, 4), dtype=float)
            new_orient[:, 3] = 1.0
            new_orient[:old_n] = self.orientation
            self.orientation = new_orient
            new_ang = np.zeros((N, 3), dtype=float)
            new_ang[:old_n] = self.angular_velocity
            self.angular_velocity = new_ang

        # Per-material mass quantity [N, 2].
        mass_quantity = np.zeros((N, 2), dtype=float)
        mass_quantity[:, 0] = self.mass * (self.body_id == 0)
        mass_quantity[:, 1] = self.mass * (self.body_id == 1)

        external_force = self._external_force_frame()
        interaction_force = self._interaction_force_frame()
        signed_distance = self._signed_distance(self.positions)

        attributes = {
            "phase_id": self.phase_id,
            "water_region": self.water_region,
            "mass_quantity": mass_quantity,
        }
        fields = {
            "contact_signed_distance": signed_distance,
            "cumulative_mass_source": self.cum_source.copy(),
            "external_force": external_force,
            "interaction_force": interaction_force,
        }
        scalars = {
            "contact_length_scale": self.boat_thickness,
        }

        return StateFrame(
            time=self.time,
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
