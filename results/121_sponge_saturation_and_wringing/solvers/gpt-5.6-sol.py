"""Deterministic porous-sponge and particle-water dynamics for scene 121.

The implementation deliberately uses only NumPy.  The sponge is a damped
nonlinear spring lattice with a saturation-dependent torsional contraction
law.  Water is represented by weakly-compressible particles.  Momentum is
exchanged by porous drag and by conservative resolved/pore phase transfer.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Small, deterministic two-way coupled solid/liquid solver."""

    def initialize(self, scene, seed: int) -> None:
        del seed  # The discretization is deterministic and needs no noise.
        self.scene = scene
        self.config = scene.config
        initial = scene.initial_state()

        self.time = 0.0
        self.x = np.asarray(initial.position, dtype=np.float64).copy()
        self.v = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.body = np.asarray(initial.body_id, dtype=np.int32).copy()
        initial_mass = np.asarray(initial.mass, dtype=np.float64)
        self.n = len(self.x)
        self.si = np.flatnonzero(self.body == 1)
        self.wi = np.flatnonzero(self.body == 2)
        self.gi = np.flatnonzero((self.body == 10) | (self.body == 11))
        self.ns, self.nw = len(self.si), len(self.wi)

        self.rest = self.x[self.si].copy()
        self.dry_mass = np.zeros(self.n, dtype=np.float64)
        self.dry_mass[self.si] = initial_mass[self.si]
        self.dry_mass[self.gi] = initial_mass[self.gi]
        self.free_water = initial_mass[self.wi].copy()
        self.pore_water = np.zeros(self.ns, dtype=np.float64)
        # About 36 g capacity: enough to visibly load the very light dry sponge.
        self.pore_capacity0 = np.full(self.ns, 0.0010, dtype=np.float64)

        # The mesh consists of nine ordered four-corner cross-sections.
        x_levels = np.unique(np.round(self.rest[:, 0], 10))
        self.sections = [np.flatnonzero(np.isclose(self.rest[:, 0], q)) for q in x_levels]
        self.section_x = np.array([self.rest[s, 0].mean() for s in self.sections])
        self.node_section = np.zeros(self.ns, dtype=np.int32)
        for k, section in enumerate(self.sections):
            self.node_section[section] = k

        edges: dict[tuple[int, int], float] = {}

        def add_edge(a: int, b: int, stiffness: float) -> None:
            key = (min(a, b), max(a, b))
            edges[key] = max(edges.get(key, 0.0), stiffness)

        # Complete cross-section frames resist shear and collapse.
        for section in self.sections:
            for a in range(len(section)):
                for b in range(a + 1, len(section)):
                    add_edge(int(section[a]), int(section[b]), 0.42)
        # Longitudinal fibers and diagonal fibers transmit lift and torsion.
        for sa, sb in zip(self.sections[:-1], self.sections[1:]):
            for c in range(4):
                add_edge(int(sa[c]), int(sb[c]), 0.82)
                add_edge(int(sa[c]), int(sb[(c + 1) % 4]), 0.34)
                add_edge(int(sa[c]), int(sb[(c - 1) % 4]), 0.34)
        # Second-neighbour fibers supply a modest bending stiffness.
        for sa, sb in zip(self.sections[:-2], self.sections[2:]):
            for c in range(4):
                add_edge(int(sa[c]), int(sb[c]), 0.18)

        ordered = sorted(edges)
        self.ei = np.array([p[0] for p in ordered], dtype=np.int32)
        self.ej = np.array([p[1] for p in ordered], dtype=np.int32)
        self.edge_k = np.array([edges[p] for p in ordered], dtype=np.float64)
        self.edge_l0 = np.linalg.norm(self.rest[self.ej] - self.rest[self.ei], axis=1)
        self.diff_i = self.ei.copy()
        self.diff_j = self.ej.copy()

        # Fixed particle pairs for symmetric pressure and viscosity forces.
        self.wp_i, self.wp_j = np.triu_indices(self.nw, 1)
        self.particle_nominal_mass = float(np.median(initial_mass[self.wi]))
        self.emit_mass = min(self.particle_nominal_mass, 0.00045)
        self.emit_accumulator = 0.0
        self.emit_counter = 0
        self.water_age = np.zeros(self.nw, dtype=np.float64)

        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.floor_z = 0.015
        self.internal_dt = 0.0025

        # Momentum-accounting domains: all bodies, sponge, free water, grips.
        self.domain_weights = np.zeros((4, self.n), dtype=np.float64)
        self.domain_weights[0, :] = 1.0
        self.domain_weights[1, self.si] = 1.0
        self.domain_weights[2, self.wi] = 1.0
        self.domain_weights[3, self.gi] = 1.0
        self.cum_external_impulse = np.zeros((4, 3), dtype=np.float64)
        self.cum_inflow_momentum = np.zeros((4, 3), dtype=np.float64)
        self.cum_outflow_momentum = np.zeros((4, 3), dtype=np.float64)
        self.cum_mass_source = np.zeros(2, dtype=np.float64)
        self.current_external = np.zeros((4, 5, 3), dtype=np.float64)
        self.current_interaction = np.zeros((4, 3), dtype=np.float64)
        self.reference_momentum = max(
            0.01, float(np.sum(initial_mass[self.wi] * np.linalg.norm(self.v[self.wi], axis=1)))
        )

        self.steps = 0
        self.absorbed_total = 0.0
        self.expelled_total = 0.0
        self.source_total = 0.0
        self.drain_total = 0.0
        self.current_external[:, 0, :] = self._gravity_by_domain()

    @staticmethod
    def _smooth_interval(t: float, start: float, end: float) -> tuple[float, float, float]:
        """C2 position ramp and its first two time derivatives."""
        if t <= start:
            return 0.0, 0.0, 0.0
        if t >= end:
            return 1.0, 0.0, 0.0
        u = (t - start) / (end - start)
        span = end - start
        return (
            u * u * (3.0 - 2.0 * u),
            6.0 * u * (1.0 - u) / span,
            (6.0 - 12.0 * u) / (span * span),
        )

    def _control_state(self, t: float):
        lift, lift_rate, lift_accel = self._smooth_interval(t, 6.5, 8.0)
        twist, twist_rate, _ = self._smooth_interval(t, 8.0, 11.0)
        lift_height = 0.22 * lift
        vz = 0.22 * lift_rate
        az = 0.22 * lift_accel
        angle = math.radians(150.0) * twist
        omega = math.radians(150.0) * twist_rate
        centers = np.array([[-0.205, 0.0, 0.065 + lift_height],
                            [0.205, 0.0, 0.065 + lift_height]], dtype=np.float64)
        velocities = np.array([[0.0, 0.0, vz], [0.0, 0.0, vz]], dtype=np.float64)
        accelerations = np.array([[0.0, 0.0, az], [0.0, 0.0, az]], dtype=np.float64)
        angles = np.array([angle, -angle], dtype=np.float64)
        omegas = np.array([omega, -omega], dtype=np.float64)
        return centers, velocities, accelerations, angles, omegas

    @staticmethod
    def _rotate_about_x(points: np.ndarray, angle: float) -> np.ndarray:
        out = points.copy()
        c, s = math.cos(angle), math.sin(angle)
        out[:, 1] = c * points[:, 1] - s * points[:, 2]
        out[:, 2] = s * points[:, 1] + c * points[:, 2]
        return out

    def _sponge_geometry(self):
        sx = self.x[self.si]
        centers = np.array([sx[q].mean(axis=0) for q in self.sections])
        areas = np.empty(len(self.sections), dtype=np.float64)
        angles = np.empty(len(self.sections), dtype=np.float64)
        for k, q in enumerate(self.sections):
            p = sx[q]
            areas[k] = np.linalg.norm(np.cross(p[1] - p[0], p[3] - p[0]))
            radial = p[0] - centers[k]
            angles[k] = math.atan2(radial[2], radial[1])
        angles = np.unwrap(angles)
        dx = np.maximum(np.gradient(centers[:, 0]), 0.020)
        twist_strain = np.gradient(angles) / dx
        rest_area = 0.16 * 0.08
        area_ratio = np.clip(areas / rest_area, 0.20, 1.35)
        return centers, area_ratio, twist_strain

    def _gravity_by_domain(self) -> np.ndarray:
        mass = self._state_mass()
        result = np.zeros((4, 3), dtype=np.float64)
        result[0] = mass.sum() * self.gravity
        result[1] = mass[self.si].sum() * self.gravity
        result[2] = mass[self.wi].sum() * self.gravity
        result[3] = mass[self.gi].sum() * self.gravity
        return result

    def _state_mass(self) -> np.ndarray:
        mass = self.dry_mass.copy()
        mass[self.si] += self.pore_water
        mass[self.wi] += self.free_water
        return mass

    def _emit_from_faucet(self, dt: float, t: float) -> None:
        source = self.config["initial_conditions"]["water_source"]
        if not (float(source["starts_at"]) <= t < float(source["stops_at"])):
            return
        # A 9 g/s faucet is intentionally modest for the small numerical sponge.
        self.emit_accumulator += 0.0090 * dt
        inactive = np.flatnonzero(self.free_water <= 1.0e-12)
        while self.emit_accumulator >= self.emit_mass and len(inactive):
            p = int(inactive[0])
            self.emit_accumulator -= self.emit_mass
            ring = self.emit_counter % 5
            offsets = ((0.0, 0.0), (0.008, 0.0), (-0.008, 0.0),
                       (0.0, 0.008), (0.0, -0.008))
            dx, dy = offsets[ring]
            center = np.asarray(source["center"], dtype=np.float64)
            self.x[self.wi[p]] = center + np.array([dx, dy, -0.055])
            direction = np.asarray(source["direction"], dtype=np.float64)
            self.v[self.wi[p]] = float(source["speed"]) * direction
            self.free_water[p] = self.emit_mass
            self.water_age[p] = 0.0
            momentum = self.emit_mass * self.v[self.wi[p]]
            self.cum_inflow_momentum[0] += momentum
            self.cum_inflow_momentum[2] += momentum
            self.cum_mass_source[1] += self.emit_mass
            self.source_total += self.emit_mass
            self.emit_counter += 1
            inactive = inactive[1:]

    def _fluid_forces(self, dt: float):
        del dt
        wx = self.x[self.wi]
        wv = self.v[self.wi]
        active = self.free_water > 1.0e-12
        force = self.free_water[:, None] * self.gravity
        boundary_force = np.zeros((self.nw, 3), dtype=np.float64)

        # Symmetric weakly-compressible particle pressure and viscosity.
        i, j = self.wp_i, self.wp_j
        pair_active = active[i] & active[j]
        delta = wx[j] - wx[i]
        distance = np.linalg.norm(delta, axis=1)
        mask = pair_active & (distance < 0.019) & (distance > 1.0e-8)
        if np.any(mask):
            ii, jj = i[mask], j[mask]
            rr = distance[mask]
            normal = delta[mask] / rr[:, None]
            overlap = 0.019 - rr
            scale = (self.free_water[ii] * self.free_water[jj]) / (self.emit_mass ** 2)
            pressure = -normal * (0.34 * overlap * scale)[:, None]
            visc = 0.00010 * (wv[jj] - wv[ii]) * (overlap / 0.019)[:, None] * scale[:, None]
            pf = pressure + visc
            np.add.at(force, ii, pf)
            np.add.at(force, jj, -pf)

        # A shallow receiving basin guides settled water toward an open drain.
        low = active & (wx[:, 2] < 0.040)
        if np.any(low):
            drain_xy = np.array([0.0, -0.205])
            toward = drain_xy - wx[low, :2]
            norm = np.linalg.norm(toward, axis=1)
            direction = toward / np.maximum(norm[:, None], 1.0e-8)
            boundary_force[low, :2] += self.free_water[low, None] * 1.1 * direction
        force += boundary_force
        return force, boundary_force

    def _solid_forces(self, centers, area_ratio, twist_strain):
        del area_ratio
        sx = self.x[self.si]
        sv = self.v[self.si]
        sm = self.dry_mass[self.si] + self.pore_water
        force = sm[:, None] * self.gravity

        delta = sx[self.ej] - sx[self.ei]
        length = np.linalg.norm(delta, axis=1)
        normal = delta / np.maximum(length[:, None], 1.0e-9)
        rel_speed = np.sum((sv[self.ej] - sv[self.ei]) * normal, axis=1)
        edge_force = normal * (self.edge_k * (length - self.edge_l0) + 0.012 * rel_speed)[:, None]
        np.add.at(force, self.ei, edge_force)
        np.add.at(force, self.ej, -edge_force)

        # Wet helical fibers produce radial compression under dynamically
        # generated torsional strain (a simple Poynting/constriction model).
        saturation = np.clip(self.pore_water / self.pore_capacity0, 0.0, 1.0)
        for k, section in enumerate(self.sections):
            radial = sx[section] - centers[k]
            radial[:, 0] = 0.0
            rnorm = np.linalg.norm(radial, axis=1)
            wetness = 0.25 + 0.75 * float(np.mean(saturation[section]))
            magnitude = 0.00058 * min(abs(twist_strain[k]), 24.0) * wetness
            inward = -radial / np.maximum(rnorm[:, None], 1.0e-8) * magnitude
            inward -= inward.mean(axis=0)
            force[section] += inward
        return force

    def _grip_attachment(self, t: float):
        force = np.zeros((self.ns, 3), dtype=np.float64)
        if t < 6.5:
            return force
        _, _, _, angles, omegas = self._control_state(t)
        left = np.flatnonzero(self.rest[:, 0] <= -0.125 + 1.0e-9)
        right = np.flatnonzero(self.rest[:, 0] >= 0.125 - 1.0e-9)
        lift, lift_rate, _ = self._smooth_interval(t, 6.5, 8.0)
        for nodes, side in ((left, 0), (right, 1)):
            rel = self.rest[nodes].copy()
            rel[:, 0] -= -0.18 if side == 0 else 0.18
            rel[:, 2] -= 0.065
            rotated = self._rotate_about_x(rel, float(angles[side]))
            target = rotated + np.array([-0.18 if side == 0 else 0.18, 0.0,
                                          0.065 + 0.22 * lift])
            omega_vec = np.array([omegas[side], 0.0, 0.0])
            target_v = np.cross(np.broadcast_to(omega_vec, rotated.shape), rotated)
            target_v[:, 2] += 0.22 * lift_rate
            force[nodes] += 2.6 * (target - self.x[self.si[nodes]])
            force[nodes] += 0.035 * (target_v - self.v[self.si[nodes]])
        return force

    def _porous_coupling(self, dt: float):
        """Return water and solid forces plus the phase-transfer impulse."""
        wx = self.x[self.wi]
        wv = self.v[self.wi]
        sx = self.x[self.si]
        sv = self.v[self.si]
        active = self.free_water > 1.0e-12
        delta = wx[:, None, :] - sx[None, :, :]
        d2 = np.sum(delta * delta, axis=2)
        weights = np.exp(-d2 / (0.068 ** 2))
        weights[d2 > 0.078 ** 2] = 0.0
        weights[~active, :] = 0.0
        sums = weights.sum(axis=1)
        normalized = weights / np.maximum(sums[:, None], 1.0e-12)
        contact = np.clip(sums / 2.0, 0.0, 1.0)
        solid_velocity = normalized @ sv
        drag_water = -self.free_water[:, None] * (7.0 * contact)[:, None] * (wv - solid_velocity)
        drag_solid = -(normalized.T @ drag_water)

        # Absorb resolved water into available pore volume.  This mass and its
        # momentum are transferred, rather than deleted.
        saturation = np.clip(self.pore_water / self.pore_capacity0, 0.0, 1.0)
        local_sat = normalized @ saturation
        absorb = self.free_water * (1.8 * dt) * contact * np.clip(1.0 - local_sat, 0.0, 1.0)
        absorb = np.minimum(absorb, self.free_water)
        proposed = normalized * absorb[:, None]
        room = np.maximum(self.pore_capacity0 - self.pore_water, 0.0)
        node_gain = proposed.sum(axis=0)
        limiter = np.ones(self.ns)
        positive = node_gain > 1.0e-15
        limiter[positive] = np.minimum(1.0, room[positive] / node_gain[positive])
        proposed *= limiter[None, :]
        actual_particle = proposed.sum(axis=1)
        node_gain = proposed.sum(axis=0)
        transfer_impulse = proposed.T @ wv
        old_sm = self.dry_mass[self.si] + self.pore_water
        new_sm = old_sm + node_gain
        self.v[self.si] = (old_sm[:, None] * self.v[self.si] + transfer_impulse) / new_sm[:, None]
        self.pore_water += node_gain
        self.free_water -= actual_particle
        total_impulse = np.sum(transfer_impulse, axis=0)
        if np.any(actual_particle > 0.0):
            ptotal = np.sum(actual_particle[:, None] * wv, axis=0)
            self.cum_inflow_momentum[1] += ptotal
            self.cum_outflow_momentum[2] += ptotal
        self.absorbed_total += float(node_gain.sum())
        return drag_water, drag_solid, total_impulse / max(dt, 1.0e-12)

    def _diffuse_and_expel(self, dt: float, centers, area_ratio, twist_strain):
        # Conservative graph diffusion of pore liquid.
        saturation = self.pore_water / self.pore_capacity0
        flux = 0.10 * (saturation[self.diff_j] - saturation[self.diff_i]) * dt
        flux *= np.minimum(self.pore_capacity0[self.diff_i], self.pore_capacity0[self.diff_j])
        delta_mass = np.zeros(self.ns)
        np.add.at(delta_mass, self.diff_i, flux)
        np.add.at(delta_mass, self.diff_j, -flux)
        scale = 1.0
        negative = delta_mass < 0.0
        if np.any(negative):
            scale = min(scale, float(np.min(self.pore_water[negative] / -delta_mass[negative])))
        self.pore_water += delta_mass * min(1.0, max(0.0, scale))

        # Deformed pore capacity and pressure-driven drainage.  No time-based
        # wringing switch is used: release follows the actual current strain.
        capacity = self.pore_capacity0.copy()
        release = np.zeros(self.ns)
        for k, section in enumerate(self.sections):
            capacity[section] *= 0.30 + 0.70 * area_ratio[k]
            excess = np.maximum(self.pore_water[section] - 0.96 * capacity[section], 0.0)
            pressure_rate = 0.42 * min((abs(twist_strain[k]) / 10.0) ** 2, 4.0)
            release[section] = excess + pressure_rate * dt * self.pore_water[section]
        release = np.minimum(release, self.pore_water)

        expelled_impulse = np.zeros(3)
        for k, section in enumerate(self.sections):
            dm = float(release[section].sum())
            if dm <= 1.0e-12:
                continue
            inactive = np.flatnonzero(self.free_water <= 1.0e-12)
            if not len(inactive):
                continue
            p = int(inactive[0])
            shares = release[section]
            self.pore_water[section] -= shares
            local_v = np.average(self.v[self.si[section]], axis=0, weights=np.maximum(shares, 1.0e-15))
            side = -1.0 if (k % 2 == 0) else 1.0
            eject = local_v + np.array([0.0, 0.30 * side, -0.42])
            self.free_water[p] = dm
            self.x[self.wi[p]] = centers[k] + np.array([0.0, 0.045 * side, -0.010])
            self.v[self.wi[p]] = eject
            self.water_age[p] = 0.0
            momentum = dm * eject
            expelled_impulse += momentum
            self.cum_outflow_momentum[1] += momentum
            self.cum_inflow_momentum[2] += momentum
            self.expelled_total += dm
        return expelled_impulse / max(dt, 1.0e-12)

    def _apply_collisions_and_drain(self, dt: float):
        sponge_impulse = np.zeros((self.ns, 3))
        water_impulse = np.zeros((self.nw, 3))

        # No penetration of the physical sink floor or side/back walls.
        sx = self.x[self.si]
        sv = self.v[self.si]
        below = sx[:, 2] < self.floor_z
        if np.any(below):
            old = sv[below].copy()
            sx[below, 2] = self.floor_z
            sv[below, 2] = np.maximum(0.0, -0.06 * sv[below, 2])
            mass = self.dry_mass[self.si][below] + self.pore_water[below]
            sponge_impulse[below] += mass[:, None] * (sv[below] - old)
        for axis, lo, hi in ((0, -0.384, 0.384), (1, -0.282, 0.264)):
            low = sx[:, axis] < lo
            high = sx[:, axis] > hi
            for mask, bound in ((low, lo), (high, hi)):
                if np.any(mask):
                    old = sv[mask].copy()
                    sx[mask, axis] = bound
                    sv[mask, axis] *= -0.10
                    mass = self.dry_mass[self.si][mask] + self.pore_water[mask]
                    sponge_impulse[mask] += mass[:, None] * (sv[mask] - old)

        wx = self.x[self.wi]
        wv = self.v[self.wi]
        active = self.free_water > 1.0e-12
        below = active & (wx[:, 2] < self.floor_z + 0.003)
        if np.any(below):
            old = wv[below].copy()
            wx[below, 2] = self.floor_z + 0.003
            wv[below, 2] = np.maximum(0.0, -0.12 * wv[below, 2])
            wv[below, :2] *= 0.90
            water_impulse[below] += self.free_water[below, None] * (wv[below] - old)
        for axis, lo, hi in ((0, -0.384, 0.384), (1, -0.282, 0.264)):
            low = active & (wx[:, axis] < lo)
            high = active & (wx[:, axis] > hi)
            for mask, bound in ((low, lo), (high, hi)):
                if np.any(mask):
                    old = wv[mask].copy()
                    wx[mask, axis] = bound
                    wv[mask, axis] *= -0.18
                    water_impulse[mask] += self.free_water[mask, None] * (wv[mask] - old)

        # Remove water only through the receiving basin's open drain.
        drain_distance = np.linalg.norm(wx[:, :2] - np.array([0.0, -0.205]), axis=1)
        draining = active & (wx[:, 2] <= self.floor_z + 0.004) & (drain_distance < 0.052)
        for p in np.flatnonzero(draining):
            dm = float(self.free_water[p])
            momentum = dm * wv[p]
            self.cum_outflow_momentum[0] += momentum
            self.cum_outflow_momentum[2] += momentum
            self.cum_mass_source[1] -= dm
            self.drain_total += dm
            self.free_water[p] = 0.0
            wx[p] = np.array([0.0, -0.205, self.floor_z + 0.001])
            wv[p] = 0.0
            self.water_age[p] = 0.0
        return sponge_impulse.sum(axis=0) / dt, water_impulse.sum(axis=0) / dt

    def _step(self, dt: float) -> None:
        t_mid = self.time + 0.5 * dt
        self._emit_from_faucet(dt, t_mid)
        centers, area_ratio, twist_strain = self._sponge_geometry()
        water_force, water_boundary = self._fluid_forces(dt)
        solid_force = self._solid_forces(centers, area_ratio, twist_strain)
        grip_force = self._grip_attachment(t_mid)
        drag_water, drag_solid, absorb_force = self._porous_coupling(dt)
        expel_force = self._diffuse_and_expel(dt, centers, area_ratio, twist_strain)
        phase_force = absorb_force - expel_force

        water_force += drag_water
        solid_force += drag_solid + grip_force

        sm = self.dry_mass[self.si] + self.pore_water
        active = self.free_water > 1.0e-12
        self.v[self.si] += dt * solid_force / np.maximum(sm[:, None], 1.0e-8)
        self.x[self.si] += dt * self.v[self.si]
        if np.any(active):
            self.v[self.wi[active]] += dt * water_force[active] / self.free_water[active, None]
            self.x[self.wi[active]] += dt * self.v[self.wi[active]]
            self.water_age[active] += dt

        sponge_boundary_total, water_collision_total = self._apply_collisions_and_drain(dt)
        water_boundary_total = water_boundary.sum(axis=0) + water_collision_total

        # Kinematic grip states are sampled analytically and never teleported.
        centers_g, velocity_g, accel_g, _, _ = self._control_state(self.time + dt)
        self.x[self.gi] = centers_g
        self.v[self.gi] = velocity_g

        grav = self._gravity_by_domain()
        actuator = self.dry_mass[self.gi].sum() * accel_g.mean(axis=0) - grav[3] + grip_force.sum(axis=0)
        current = np.zeros((4, 5, 3), dtype=np.float64)
        current[:, 0, :] = grav
        current[0, 1] = sponge_boundary_total + water_boundary_total
        current[1, 1] = sponge_boundary_total
        current[2, 1] = water_boundary_total
        current[0, 2] = actuator
        current[3, 2] = actuator
        fluid_on_sponge = drag_solid.sum(axis=0) + phase_force
        current[1, 3] = fluid_on_sponge
        current[2, 3] = -fluid_on_sponge
        current[1, 4] = grip_force.sum(axis=0)
        current[3, 4] = -grip_force.sum(axis=0)
        self.current_external = current
        self.cum_external_impulse += current.sum(axis=1) * dt
        self.current_interaction = np.array([
            fluid_on_sponge, -fluid_on_sponge,
            grip_force.sum(axis=0), -grip_force.sum(axis=0),
        ])
        self.time += dt
        self.steps += 1

    @staticmethod
    def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ])

    def _make_frame(self, requested_time: float) -> StateFrame:
        mass = self._state_mass()
        mass_quantity = np.zeros((self.n, 2), dtype=np.float64)
        mass_quantity[:, 0] = self.dry_mass
        mass_quantity[self.si, 1] = self.pore_water
        mass_quantity[self.wi, 1] = self.free_water

        saturation = np.zeros(self.n, dtype=np.float64)
        saturation[self.si] = np.clip(self.pore_water / self.pore_capacity0, 0.0, 1.0)
        saturation[self.wi] = (self.free_water > 1.0e-12).astype(np.float64)
        porosity = np.zeros(self.n, dtype=np.float64)
        _, area_ratio, _ = self._sponge_geometry()
        porosity[self.si] = 0.72 * np.clip(area_ratio[self.node_section], 0.30, 1.0)
        phase = np.zeros(self.n, dtype=np.float64)
        phase[self.wi] = 1.0

        orientation = np.zeros((self.n, 4), dtype=np.float64)
        orientation[:, 3] = 1.0
        angular_velocity = np.zeros((self.n, 3), dtype=np.float64)
        _, _, _, angles, omegas = self._control_state(requested_time)
        base = np.array([0.0, -math.sqrt(0.5), 0.0, -math.sqrt(0.5)])
        for q, angle, omega in zip(self.gi, angles, omegas):
            twist = np.array([math.sin(0.5 * angle), 0.0, 0.0, math.cos(0.5 * angle)])
            orientation[q] = self._quat_multiply(twist, base)
            angular_velocity[q, 0] = omega

        frame = StateFrame(
            time=float(requested_time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=self.body.copy(),
            attributes={
                "saturation": saturation,
                "porosity": porosity,
                "phase": phase,
                "mass_quantity": mass_quantity,
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_momentum.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_momentum.copy(),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": self.current_external.copy(),
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "interaction_force": self.current_interaction.copy(),
            },
            scalars={
                "gravity_z": float(self.gravity[2]),
                "reference_momentum": float(self.reference_momentum),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-9:
            raise ValueError("advance_to times must be finite and nondecreasing")
        while self.time < target - 1.0e-12:
            self._step(min(self.internal_dt, target - self.time))
        # Suppress harmless accumulated floating-point drift in reported time.
        self.time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "nonlinear spring lattice + WC particles + porous drag/storage",
            "internal_steps": int(self.steps),
            "absorbed_water_kg": float(self.absorbed_total),
            "expelled_water_kg": float(self.expelled_total),
            "faucet_source_kg": float(self.source_total),
            "drain_outflow_kg": float(self.drain_total),
            "max_saturation": float(np.max(self.pore_water / self.pore_capacity0)),
        }
