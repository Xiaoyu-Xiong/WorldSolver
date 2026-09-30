"""Two-way coupled particle fluid + rigid-body solver for overloaded boat scene.

A lightweight weakly-compressible particle fluid is coupled to three rigid bodies
(boat hull, light cargo sphere, heavy cargo sphere) through hydrostatic
buoyancy and penalty contact.  The hull is modelled as an axis-aligned box
with an open top; when the rim dips below the free surface the interior floods,
removing the cavity buoyancy and allowing the overloaded hull to sink.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldsolver.contracts import Scene, StateFrame  # type: ignore


class BenchmarkSolver:
    def initialize(self, scene: Scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = scene.config
        ic = self.config["initial_conditions"]

        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.g = float(np.linalg.norm(self.gravity))
        if self.g <= 0.0:
            self.g = 9.81

        self.sample_dt = float(self.config["sample_dt"])
        self.duration = float(self.config["duration"])
        self.rho0 = float(self.config["fluid"]["density"])
        self.water_depth = float(self.config["fluid"]["initial_water_depth"])

        self.h = float(ic["particle_spacing"])
        self.r_sp = float(ic["sphere_radius"])
        self.light_rho = float(ic["light_density"])
        self.heavy_rho = float(ic["heavy_density"])
        self.boat_mass = float(ic["boat_mass"])
        self.hull_half = np.asarray(
            [float(ic["hull_length"]) / 2.0, float(ic["hull_width"]) / 2.0, float(ic["hull_height"]) / 2.0],
            dtype=float,
        )

        frame0 = scene.initial_state()

        # Per-particle arrays: water particles first, then boat/light/heavy markers
        self.pos = np.asarray(frame0.position, dtype=float).copy()
        self.vel = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=int).copy()
        self.phase = np.asarray(frame0.attributes["phase"], dtype=int).copy()

        self.n_water = int(np.sum(self.body_id == 0))
        self.n_total = self.pos.shape[0]

        # Rigid-body state indexed 1..3 (0 unused)
        self.body_pos = np.zeros((4, 3), dtype=float)
        self.body_vel = np.zeros((4, 3), dtype=float)
        self.body_omega = np.zeros((4, 3), dtype=float)
        self.body_quat = np.zeros((4, 4), dtype=float)
        self.body_quat[:, 3] = 1.0
        for i in range(1, 4):
            self.body_pos[i] = self.pos[self.n_water + i - 1]
            self.body_vel[i] = self.vel[self.n_water + i - 1]

        # Per-particle orientation and angular velocity (identity/no spin)
        self.orientation = np.zeros((self.n_total, 4), dtype=float)
        self.orientation[:, 3] = 1.0
        self.angular_velocity = np.zeros((self.n_total, 3), dtype=float)

        # Release times
        self.light_release = float(ic["light_release_time"])
        self.heavy_release = float(ic["heavy_release_time"])

        # Domain boundaries for fluid and solids
        self.xmin, self.xmax = -2.3, 2.3
        self.ymin, self.ymax = -1.3, 1.3
        self.zmin, self.zmax = 0.0, 2.8

        # Numerical parameters
        self.max_dt = 0.001  # internal substep for stability
        self.fluid_damping = 2.0
        self.solid_damping = 5.0
        self.contact_stiff = 2e5
        self.contact_damp = 4e3
        self.fluid_solid_stiff = 1e5
        self.fluid_solid_damp = 2e3
        self.drag_coeff = 300.0

        # Volume / mass of a fluid particle
        self.fluid_vol = self.h ** 3
        self.fluid_mass = self.rho0 * self.fluid_vol

        # Cumulative budget fields
        self.cum_ext_impulse = np.zeros((5, 3), dtype=float)
        self.cum_in_mom = np.zeros((5, 3), dtype=float)
        self.cum_out_mom = np.zeros((5, 3), dtype=float)
        self.cum_mass_source = np.zeros(1, dtype=float)

        # Interaction force buffer [water, boat, light, heavy]
        self.interaction = np.zeros((4, 3), dtype=float)

        self.time = 0.0
        self.substeps = 0
        self._diagnostics = {
            "solver": "coupled_particle_fluid_rigid",
            "seed": seed,
            "fluid_particles": int(self.n_water),
            "rigid_bodies": 3,
        }

    # ----------------------------------------------------------------- helpers
    def _body_mass(self, idx: int) -> float:
        if idx == 1:
            return self.boat_mass
        if idx == 2:
            return 4.0 / 3.0 * np.pi * self.r_sp ** 3 * self.light_rho
        if idx == 3:
            return 4.0 / 3.0 * np.pi * self.r_sp ** 3 * self.heavy_rho
        return 0.0

    def _sphere_volume(self) -> float:
        return 4.0 / 3.0 * np.pi * self.r_sp ** 3

    def _inside_hull_xy(self, pos: np.ndarray) -> np.ndarray:
        c = self.body_pos[1]
        return (
            (pos[:, 0] > c[0] - self.hull_half[0])
            & (pos[:, 0] < c[0] + self.hull_half[0])
            & (pos[:, 1] > c[1] - self.hull_half[1])
            & (pos[:, 1] < c[1] + self.hull_half[1])
        )

    def _hull_flooded(self) -> bool:
        rim = self.body_pos[1, 2] + self.hull_half[2]
        return rim < self.water_depth + 0.03

    # ----------------------------------------------------------------- forces
    def _body_buoyancy_force(self, idx: int) -> np.ndarray:
        """Hydrostatic buoyancy plus drag."""
        F = np.zeros(3, dtype=float)
        if idx == 1:
            c = self.body_pos[1]
            z_bot = c[2] - self.hull_half[2]
            draft = min(max(self.water_depth - z_bot, 0.0), 2.0 * self.hull_half[2])
            # If flooded, cavity fills with water -> only material volume provides buoyancy
            if self._hull_flooded():
                # Treat hull as solid with effective density 1500 kg/m^3 for the shell
                material_volume = self.boat_mass / 1500.0
                F[2] = self.rho0 * self.g * material_volume
            else:
                base_area = 4.0 * self.hull_half[0] * self.hull_half[1]
                F[2] = self.rho0 * self.g * base_area * draft
            F[2] -= self.drag_coeff * base_area * self.body_vel[idx, 2]
            return F
        # Sphere
        c = self.body_pos[idx]
        if c[2] + self.r_sp <= 0.0:
            return F
        if c[2] - self.r_sp >= self.water_depth:
            return F
        h = min(max((self.water_depth - c[2] + self.r_sp) / (2.0 * self.r_sp), 0.0), 1.0)
        Vsub = self._sphere_volume() * h
        F[2] = self.rho0 * self.g * Vsub
        F[2] -= 5.0 * self._body_mass(idx) * self.body_vel[idx, 2]
        return F

    def _closest_point_on_hull(self, p: np.ndarray) -> np.ndarray:
        c = self.body_pos[1]
        lo = c - self.hull_half
        hi = c + self.hull_half
        return np.clip(p, lo, hi)

    def _wall_floor_force(self, p: np.ndarray, v: np.ndarray, radius: float, mass: float) -> np.ndarray:
        f = np.zeros(3, dtype=float)
        k = self.contact_stiff * mass
        c = self.contact_damp * mass
        if p[0] - radius < self.xmin:
            pen = self.xmin - (p[0] - radius)
            f[0] = -k * pen - c * v[0]
        if p[0] + radius > self.xmax:
            pen = (p[0] + radius) - self.xmax
            f[0] = k * pen - c * v[0]
        if p[1] - radius < self.ymin:
            pen = self.ymin - (p[1] - radius)
            f[1] = -k * pen - c * v[1]
        if p[1] + radius > self.ymax:
            pen = (p[1] + radius) - self.ymax
            f[1] = k * pen - c * v[1]
        if p[2] - radius < self.zmin:
            pen = self.zmin - (p[2] - radius)
            f[2] = -k * pen - c * v[2]
        return f

    def _hull_floor_wall_force(self) -> np.ndarray:
        c = self.body_pos[1]
        v = self.body_vel[1]
        hx, hy, hz = self.hull_half
        f = np.zeros(3, dtype=float)
        k = self.contact_stiff * self.boat_mass
        cd = self.contact_damp * self.boat_mass
        if c[2] - hz < self.zmin:
            pen = self.zmin - (c[2] - hz)
            f[2] = -k * pen - cd * v[2]
        if c[0] - hx < self.xmin:
            pen = self.xmin - (c[0] - hx)
            f[0] = -k * pen - cd * v[0]
        if c[0] + hx > self.xmax:
            pen = (c[0] + hx) - self.xmax
            f[0] = k * pen - cd * v[0]
        if c[1] - hy < self.ymin:
            pen = self.ymin - (c[1] - hy)
            f[1] = -k * pen - cd * v[1]
        if c[1] + hy > self.ymax:
            pen = (c[1] + hy) - self.ymax
            f[1] = k * pen - cd * v[1]
        return f

    def _sphere_sphere_contact(self, i1: int, i2: int) -> np.ndarray:
        p1 = self.body_pos[i1]
        p2 = self.body_pos[i2]
        d = p1 - p2
        dist = np.linalg.norm(d)
        if dist >= 2.0 * self.r_sp:
            return np.zeros(3)
        n = d / (dist + 1e-12)
        pen = 2.0 * self.r_sp - dist
        m_eff = 1.0 / (1.0 / self._body_mass(i1) + 1.0 / self._body_mass(i2))
        k = self.contact_stiff * m_eff
        c = self.contact_damp * m_eff
        rel = self.body_vel[i1] - self.body_vel[i2]
        fn = -k * pen - c * np.dot(rel, n)
        f = fn * n
        vt = rel - np.dot(rel, n) * n
        f -= 0.1 * c * vt
        return f

    def _sphere_hull_contact(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        sp = self.body_pos[idx]
        cp = self._closest_point_on_hull(sp)
        d = sp - cp
        dist = np.linalg.norm(d)
        if dist >= self.r_sp:
            return np.zeros(3), np.zeros(3)
        n = d / (dist + 1e-12)
        pen = self.r_sp - dist
        m = self._body_mass(idx)
        k = self.contact_stiff * m
        c = self.contact_damp * m
        rel = self.body_vel[idx] - self.body_vel[1]
        fn = -k * pen - c * np.dot(rel, n)
        f = fn * n
        vt = rel - np.dot(rel, n) * n
        f -= 0.1 * c * vt
        rvec = cp - self.body_pos[1]
        torque = np.cross(rvec, f)
        return f, torque

    # ----------------------------------------------------------------- coupling
    def _fluid_solid_coupling(self) -> tuple[np.ndarray, np.ndarray]:
        """Penalty coupling of water particles with rigid bodies.

        Returns
        -------
        acc : [N_water, 3] updated fluid accelerations
        reactions : [4, 3] net reaction forces on water(0), boat(1), light(2), heavy(3)
        """
        n = self.n_water
        if n == 0:
            return np.zeros_like(self.pos[:0]), np.zeros((4, 3), dtype=float)

        pos = self.pos[:n]
        vel = self.vel[:n]
        acc = np.zeros((n, 3), dtype=float)
        reactions = np.zeros((4, 3), dtype=float)

        flooded = self._hull_flooded()
        bottom, top = self.body_pos[1, 2] - self.hull_half[2], self.body_pos[1, 2] + self.hull_half[2]
        inside_xy = self._inside_hull_xy(pos)

        # Spheres
        for b in (2, 3):
            center = self.body_pos[b]
            v_b = self.body_vel[b]
            dvec = pos - center[None, :]
            dist = np.linalg.norm(dvec, axis=1)
            threshold = self.r_sp + self.h * 0.5
            mask = dist < threshold
            if not np.any(mask):
                continue
            m = self._body_mass(b)
            k = self.fluid_solid_stiff * m
            c = self.fluid_solid_damp * m
            nvec = dvec[mask] / (dist[mask][:, None] + 1e-12)
            pen = threshold - dist[mask]
            rel = vel[mask] - v_b[None, :]
            fn = (k * pen)[:, None] * nvec - c * np.sum(rel * nvec, axis=1)[:, None] * nvec
            fn -= 0.1 * c * (rel - np.sum(rel * nvec, axis=1)[:, None] * nvec)
            acc[mask] += fn / self.fluid_mass
            # reaction on sphere
            reactions[b] -= fn.sum(axis=0)

        # Hull
        cp = self._closest_point_on_hull(pos)
        dvec = pos - cp
        dist = np.linalg.norm(dvec, axis=1)
        mask = (dist < self.h) & (pos[:, 2] < top + self.h)
        if np.any(mask):
            nvec = dvec[mask] / (dist[mask][:, None] + 1e-12)
            pen = self.h - dist[mask]
            # If particle is inside the cavity and flooding, allow it (weak drag only)
            interior_mask = inside_xy[mask] & (pos[mask, 2] < top - 0.03) & flooded
            if np.any(interior_mask):
                drag = -0.5 * (vel[mask][interior_mask] - self.body_vel[1])
                acc[mask][interior_mask] += drag
                reactions[1] += drag.sum(axis=0) * self.fluid_mass
            exterior_mask = ~interior_mask
            if np.any(exterior_mask):
                m = self.boat_mass
                k = self.fluid_solid_stiff * m
                c = self.fluid_solid_damp * m
                rel = vel[mask][exterior_mask] - self.body_vel[1][None, :]
                fn = (k * pen[exterior_mask])[:, None] * nvec[exterior_mask] - c * np.sum(rel * nvec[exterior_mask], axis=1)[:, None] * nvec[exterior_mask]
                fn -= 0.1 * c * (rel - np.sum(rel * nvec[exterior_mask], axis=1)[:, None] * nvec[exterior_mask])
                acc[mask][exterior_mask] += fn / self.fluid_mass
                reactions[1] -= fn.sum(axis=0)

        return acc, reactions

    def _fluid_acceleration(self, solid_reactions: np.ndarray) -> np.ndarray:
        n = self.n_water
        if n == 0:
            return np.zeros_like(self.pos[:0])
        pos = self.pos[:n]
        vel = self.vel[:n]

        acc = np.zeros((n, 3), dtype=float)
        acc += self.gravity[None, :]

        # Hydrostatic pressure: below surface -> net vertical acceleration zero;
        # above surface -> gravity only
        below = pos[:, 2] < self.water_depth
        acc[below, 2] += self.g

        # Damping
        acc -= self.fluid_damping * vel

        # Boundary repulsion (keeps particles in domain)
        boundary_acc = np.zeros((n, 3), dtype=float)
        k = 1e4
        # floor
        pen = self.zmin - (pos[:, 2] - self.h * 0.3)
        mask = pen > 0
        boundary_acc[mask, 2] += k * pen[mask] + 100.0 * vel[mask, 2]
        # walls
        for axis, lo, hi in ((0, self.xmin, self.xmax), (1, self.ymin, self.ymax)):
            pen_lo = lo - (pos[:, axis] - self.h * 0.3)
            pen_hi = (pos[:, axis] + self.h * 0.3) - hi
            mask_lo = pen_lo > 0
            mask_hi = pen_hi > 0
            boundary_acc[mask_lo, axis] += k * pen_lo[mask_lo] + 100.0 * vel[mask_lo, axis]
            boundary_acc[mask_hi, axis] -= k * pen_hi[mask_hi] + 100.0 * vel[mask_hi, axis]

        acc += boundary_acc
        acc += solid_reactions[:n]
        return acc

    # ----------------------------------------------------------------- integration
    def _step(self, dt: float) -> None:
        n = self.n_water

        # Rigid-body forces (without fluid reactions)
        ext_forces = np.zeros((4, 3), dtype=float)
        ext_torques = np.zeros((4, 3), dtype=float)
        for idx in (1, 2, 3):
            ext_forces[idx] += self._body_mass(idx) * self.gravity
            ext_forces[idx] += self._body_buoyancy_force(idx)

        # Rigid-rigid contacts
        f12, t12 = self._sphere_hull_contact(2)
        ext_forces[1] += f12
        ext_torques[1] += t12
        ext_forces[2] -= f12

        f13, t13 = self._sphere_hull_contact(3)
        ext_forces[1] += f13
        ext_torques[1] += t13
        ext_forces[3] -= f13

        f23 = self._sphere_sphere_contact(2, 3)
        ext_forces[2] += f23
        ext_forces[3] -= f23

        ext_forces[1] += self._hull_floor_wall_force()
        for idx in (2, 3):
            ext_forces[idx] += self._wall_floor_force(
                self.body_pos[idx], self.body_vel[idx], self.r_sp, self._body_mass(idx)
            )

        # Fluid-solid coupling
        fluid_acc, reactions = self._fluid_solid_coupling()
        for idx in (1, 2, 3):
            ext_forces[idx] += reactions[idx]

        # Fluid update
        self.vel[:n] += (self._fluid_acceleration(reactions[0]) + fluid_acc) * dt
        self.pos[:n] += self.vel[:n] * dt

        # Boundary clamp (hard floor/walls)
        pos = self.pos[:n]
        vel = self.vel[:n]
        floor_pen = pos[:, 2] - self.zmin
        mask = floor_pen < 0.0
        vel[mask, 2] = -0.3 * vel[mask, 2]
        pos[mask, 2] = self.zmin + 1e-5
        for axis, lo, hi in ((0, self.xmin, self.xmax), (1, self.ymin, self.ymax)):
            lo_pen = pos[:, axis] - lo
            hi_pen = hi - pos[:, axis]
            mask_lo = lo_pen < 0.0
            mask_hi = hi_pen < 0.0
            pos[mask_lo, axis] = lo + 1e-5
            pos[mask_hi, axis] = hi - 1e-5
            vel[mask_lo, axis] = -0.3 * vel[mask_lo, axis]
            vel[mask_hi, axis] = -0.3 * vel[mask_hi, axis]

        # Rigid-body integration with held-sphere constraints
        held = [False, False, False, False]
        if self.time < self.light_release:
            held[2] = True
        if self.time < self.heavy_release:
            held[3] = True

        for idx in (1, 2, 3):
            m = self._body_mass(idx)
            if held[idx]:
                self.body_vel[idx] = np.zeros(3)
            else:
                self.body_vel[idx] += ext_forces[idx] / m * dt
                self.body_vel[idx] -= self.solid_damping * self.body_vel[idx] * dt
            self.body_pos[idx] += self.body_vel[idx] * dt

        # Update per-particle markers
        for i in range(1, 4):
            self.pos[n + i - 1] = self.body_pos[i]
            self.vel[n + i - 1] = self.body_vel[i]

        # Budget accumulators
        masks = [
            self.body_id == 0,
            self.body_id == 1,
            self.body_id == 2,
            self.body_id == 3,
            np.ones_like(self.body_id, dtype=bool),
        ]
        for g, mask in enumerate(masks):
            total_mass = self.mass[mask].sum()
            self.cum_ext_impulse[g] += total_mass * self.gravity * dt

        # Store interaction forces (contact + coupling reactions on each body)
        self.interaction.fill(0.0)
        # Water net interaction is the reaction from solids
        self.interaction[0] = -reactions[0]
        # Bodies: include contact forces + coupling reactions
        # contact contributions
        self.interaction[1] += f12 + f13 + self._hull_floor_wall_force()
        self.interaction[2] += -f12 + f23 + self._wall_floor_force(self.body_pos[2], self.body_vel[2], self.r_sp, self._body_mass(2))
        self.interaction[3] += -f13 - f23 + self._wall_floor_force(self.body_pos[3], self.body_vel[3], self.r_sp, self._body_mass(3))
        # Add coupling reactions (already include wall/floor above, but coupling is separate; keep both for completeness)
        self.interaction[1] += reactions[1]
        self.interaction[2] += reactions[2]
        self.interaction[3] += reactions[3]

        self.time += dt
        self.substeps += 1

    # ----------------------------------------------------------------- StateFrame
    def _make_frame(self, target_time: float) -> StateFrame:
        pos = self.pos.copy()
        vel = self.vel.copy()
        mass = self.mass.copy()
        body_id = self.body_id.copy()
        orientation = self.orientation.copy()
        angular_velocity = self.angular_velocity.copy()

        # Boundary wall samples (6 walls)
        wall_normals = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        )
        boundary_velocity = np.zeros((6, 3), dtype=float)
        boundary_target_velocity = np.zeros((6, 3), dtype=float)

        # Domain weights [G, N]
        G = 5
        weights = np.zeros((G, pos.shape[0]), dtype=float)
        water_mask = body_id == 0
        boat_mask = body_id == 1
        light_mask = body_id == 2
        heavy_mask = body_id == 3
        weights[0, water_mask] = 1.0
        weights[1, boat_mask] = 1.0
        weights[2, light_mask] = 1.0
        weights[3, heavy_mask] = 1.0
        weights[4, :] = 1.0

        # External force budget [G, K, 3], K=1 (gravity)
        external_force = np.zeros((G, 1, 3), dtype=float)
        for g, mask in enumerate([water_mask, boat_mask, light_mask, heavy_mask, np.ones_like(body_id, dtype=bool)]):
            external_force[g, 0] = np.sum(mass[mask][:, None] * self.gravity[None, :], axis=0)

        reference_velocity = 1.0
        reference_momentum = self.boat_mass * np.sqrt(self.g * 1.0)

        frame = StateFrame(
            time=target_time,
            position=pos,
            velocity=vel,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "mass_quantity": mass.copy(),
                "phase": self.phase.copy(),
            },
            fields={
                "boundary_normal": wall_normals,
                "boundary_target_velocity": boundary_target_velocity,
                "boundary_velocity": boundary_velocity,
                "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_in_mom.copy(),
                "budget_cumulative_outflow_momentum": self.cum_out_mom.copy(),
                "budget_domain_weights": weights,
                "budget_external_force": external_force,
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "interaction_force": self.interaction.copy(),
            },
            scalars={
                "reference_momentum": reference_momentum,
                "reference_velocity": reference_velocity,
            },
        )
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time:
            return self._make_frame(target)
        while self.time < target - 1e-12:
            dt = min(self.max_dt, target - self.time)
            self._step(dt)
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return self._diagnostics
