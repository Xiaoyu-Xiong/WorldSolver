"""Soft-sphere discrete-element solver for the pouring-sand benchmark.

Model
-----
Each sand sample is a soft-sphere discrete element (DEM) grain integrated from

    m_i dv_i/dt = m_i g + sum_j f_ij + f_wall,i

with a linear normal spring-dashpot law and a Coulomb-capped viscous
tangential law:

    f_n = max(0, k_n delta - c_n v_n) n
    f_t = -min(mu |f_n|, c_t |v_t|) v_t / |v_t|

Boundaries are the frozen scaffold geometry.  The receiving floor is the exact
box signed-distance field; the supply hopper is its axisymmetric inner profile
(neck, cone, barrel) evaluated as a polyline signed distance in the (rho, z)
half plane, so the outlet at the bottom of the neck stays open and grains leave
through it under gravity alone.

Time stepping is symplectic (semi-implicit) Euler with a step tied to the
contact stiffness, dt ~ 0.21 sqrt(m/k), and contact pairs come from a Verlet
list rebuilt from a KD-tree whenever grains travel more than half the skin.
Discharge, stream breakup, floor impact, pile growth, runout and the stable
angle of repose all emerge from gravity, contact and friction; no keyframes,
scripted motion or precomputed trajectories are used, and no grains are created
or destroyed, so the total mass is conserved exactly.
"""
from __future__ import annotations

import time as _wallclock

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_FLOOR_HALF = (1.65, 1.35, 0.04)
_FLOOR_CENTER = (0.0, 0.0, -0.04)
_HOPPER_PROFILE = ((0.075, 1.42), (0.075, 1.52), (0.34, 1.72), (0.34, 2.28))


def _scene_object(config: dict, object_id: str) -> dict:
    """Return the frozen scene-geometry entry for ``object_id`` if present."""
    geometry = config.get("scene_geometry") or {}
    for item in geometry.get("objects", ()) or ():
        if isinstance(item, dict) and item.get("id") == object_id:
            return item
    return {}


def _norm_rows(array: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,ij->i", array, array))


class BenchmarkSolver:
    """Explicit soft-sphere DEM granular solver on the frozen scaffold geometry."""

    WALL_BUDGET_SECONDS = 1350.0
    MAX_DEGRADATIONS = 3
    SPEED_LIMIT = 15.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = dict(scene.config)
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        initial = scene.initial_state()
        self.position = np.ascontiguousarray(np.asarray(initial.position, dtype=float))
        self.velocity = np.ascontiguousarray(np.asarray(initial.velocity, dtype=float))
        self.count = int(len(self.position))
        conditions = config["initial_conditions"]
        if initial.mass is None:
            spacing = float(conditions["particle_spacing"])
            grain_mass = float(conditions["bulk_density"]) * spacing ** 3
            self.mass = np.full(self.count, grain_mass, dtype=float)
        else:
            self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.inverse_mass = 1.0 / self.mass

        attributes = dict(initial.attributes)
        nominal = attributes.get("nominal_radius")
        if nominal is None:
            self.radius = float(conditions["nominal_grain_radius"])
        else:
            self.radius = float(np.mean(np.asarray(nominal, dtype=float)))
        self.nominal_radius = np.full(self.count, self.radius, dtype=float)
        phase = attributes.get("material_phase")
        self.material_phase = (
            np.zeros(self.count, dtype=float)
            if phase is None
            else np.asarray(phase, dtype=float).astype(float).copy()
        )

        self.contact_diameter = 2.0 * self.radius
        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.gravity_force = self.mass[:, None] * self.gravity[None, :]
        self.total_mass = float(np.sum(self.mass))

        self._setup_geometry(config)
        self._setup_contact_model()

        self.time = 0.0
        self.sample_dt = float(config["sample_dt"])
        self.substeps = max(1, int(np.ceil(self.sample_dt / self.dt_target)))
        self.substeps = min(self.substeps, 200)
        self.dt = self.sample_dt / self.substeps
        self._apply_contact_coefficients()

        self.pair_indices = np.zeros((0, 2), dtype=np.intp)
        self.reference_position = self.position.copy()
        self.pairs_valid = False
        self.step_count = 0
        self.rebuild_count = 0
        self.degradations = 0
        self.max_speed_seen = 0.0
        self.max_penetration = 0.0
        self.start_wallclock = _wallclock.perf_counter()

    # ------------------------------------------------------------------
    # frozen geometry
    # ------------------------------------------------------------------
    def _setup_geometry(self, config: dict) -> None:
        floor = _scene_object(config, "receiving_floor")
        geometry = floor.get("geometry", {}) if floor else {}
        half = geometry.get("half_extents") or list(_FLOOR_HALF)
        transform = floor.get("transform", {}) if floor else {}
        center = transform.get("translation") or list(_FLOOR_CENTER)
        self.floor_half = np.asarray(half, dtype=float)
        self.floor_center = np.asarray(center, dtype=float)
        self.floor_top = float(self.floor_center[2] + self.floor_half[2])

        hopper = _scene_object(config, "supply_hopper")
        hopper_geometry = hopper.get("geometry", {}) if hopper else {}
        profile = hopper_geometry.get("inner_profile") or [list(p) for p in _HOPPER_PROFILE]
        self.hopper_profile = np.asarray(profile, dtype=float)
        self.hopper_z_min = float(self.hopper_profile[0, 1])
        self.hopper_z_max = float(self.hopper_profile[-1, 1])
        self.hopper_r_max = float(np.max(self.hopper_profile[:, 0]))

        domain = _scene_object(config, "sand").get("geometry", {})
        bounds_min = domain.get("bounds_min") or [-1.55, -1.25, 0.0]
        bounds_max = domain.get("bounds_max") or [1.55, 1.25, 2.25]
        self.domain_min = np.asarray(bounds_min, dtype=float)
        self.domain_max = np.asarray(bounds_max, dtype=float)

    def _setup_contact_model(self) -> None:
        """Linear spring-dashpot stiffness sized to keep overlaps a few percent of a grain."""
        grain_mass = float(np.mean(self.mass))
        self.stiffness_pair = 2.0e4
        self.stiffness_wall = 4.0e4
        self.damping_ratio = 0.55
        self.friction_pair = 0.50
        self.friction_wall = 0.40
        self.max_overlap = 0.75 * self.radius
        self.grain_mass = grain_mass
        self.reduced_mass = 0.5 * grain_mass
        omega_pair = np.sqrt(self.stiffness_pair / self.reduced_mass)
        omega_wall = np.sqrt(self.stiffness_wall / grain_mass)
        self.dt_target = 0.21 / max(omega_pair, omega_wall)
        self.skin = 0.012
        self.cutoff = self.contact_diameter + self.skin

    def _apply_contact_coefficients(self) -> None:
        """Damping and friction viscosities consistent with the current dt and stiffness."""
        self.damping_pair = (
            2.0 * self.damping_ratio * np.sqrt(self.stiffness_pair * self.reduced_mass)
        )
        self.damping_wall = (
            2.0 * self.damping_ratio * np.sqrt(self.stiffness_wall * self.grain_mass)
        )
        # Explicit dashpot stability: dt * c / m must stay well below 2.
        self.damping_pair = min(self.damping_pair, 0.6 * self.reduced_mass / self.dt)
        self.damping_wall = min(self.damping_wall, 0.6 * self.grain_mass / self.dt)
        self.tangential_viscosity_pair = self.damping_pair
        self.tangential_viscosity_wall = self.damping_wall
        self.gap_report = 0.1 * self.contact_diameter

    # ------------------------------------------------------------------
    # neighbour bookkeeping
    # ------------------------------------------------------------------
    def _rebuild_pairs(self) -> None:
        tree = cKDTree(self.position)
        pairs = tree.query_pairs(self.cutoff, output_type="ndarray")
        self.pair_indices = np.ascontiguousarray(
            np.asarray(pairs, dtype=np.intp).reshape(-1, 2)
        )
        self.reference_position = self.position.copy()
        self.pairs_valid = True
        self.rebuild_count += 1

    def _maybe_rebuild(self) -> None:
        if not self.pairs_valid:
            self._rebuild_pairs()
            return
        offset = self.position - self.reference_position
        travelled = np.max(np.einsum("ij,ij->i", offset, offset))
        if travelled > (0.5 * self.skin) ** 2:
            self._rebuild_pairs()

    # ------------------------------------------------------------------
    # grain-grain contact
    # ------------------------------------------------------------------
    def _pair_forces(self, force: np.ndarray, collect: bool):
        index = self.pair_indices
        if index.shape[0] == 0:
            return None
        first = index[:, 0]
        second = index[:, 1]
        offset = self.position[second] - self.position[first]
        distance = _norm_rows(offset)
        np.maximum(distance, 1e-12, out=distance)
        gap = distance - self.contact_diameter
        active = gap < 0.0
        if np.any(active):
            hit_a = first[active]
            hit_b = second[active]
            normal = offset[active] / distance[active][:, None]
            overlap = np.minimum(-gap[active], self.max_overlap)
            relative = self.velocity[hit_b] - self.velocity[hit_a]
            normal_speed = np.einsum("ij,ij->i", relative, normal)
            normal_force = np.maximum(
                self.stiffness_pair * overlap - self.damping_pair * normal_speed, 0.0
            )
            contact_force = normal_force[:, None] * normal
            tangent = relative - normal_speed[:, None] * normal
            slip = _norm_rows(tangent)
            sliding = slip > 1e-9
            if np.any(sliding):
                limit = np.minimum(
                    self.friction_pair * normal_force[sliding],
                    self.tangential_viscosity_pair * slip[sliding],
                )
                contact_force[sliding] -= (limit / slip[sliding])[:, None] * tangent[sliding]
            for axis in range(3):
                column = np.ascontiguousarray(contact_force[:, axis])
                force[:, axis] += np.bincount(hit_b, weights=column, minlength=self.count)
                force[:, axis] -= np.bincount(hit_a, weights=column, minlength=self.count)
            deepest = float(np.max(-gap[active]))
            if deepest > self.max_penetration:
                self.max_penetration = deepest
        if not collect:
            return None
        return gap[gap < self.gap_report]

    def _wall_response(
        self,
        index: np.ndarray,
        penetration: np.ndarray,
        normal: np.ndarray,
        stiffness: float,
        damping: float,
        friction: float,
        force: np.ndarray,
    ) -> None:
        """Apply spring-dashpot plus Coulomb friction from a static boundary."""
        velocity = self.velocity[index]
        normal_speed = np.einsum("ij,ij->i", velocity, normal)
        overlap = np.minimum(penetration, self.max_overlap)
        normal_force = np.maximum(stiffness * overlap - damping * normal_speed, 0.0)
        contact_force = normal_force[:, None] * normal
        tangent = velocity - normal_speed[:, None] * normal
        slip = _norm_rows(tangent)
        sliding = slip > 1e-9
        if np.any(sliding):
            limit = np.minimum(
                friction * normal_force[sliding],
                self.tangential_viscosity_wall * slip[sliding],
            )
            contact_force[sliding] -= (limit / slip[sliding])[:, None] * tangent[sliding]
        force[index] += contact_force
        deepest = float(np.max(penetration))
        if deepest > self.max_penetration:
            self.max_penetration = deepest

    # ------------------------------------------------------------------
    # receiving floor (exact box signed distance)
    # ------------------------------------------------------------------
    def _floor_forces(self, force: np.ndarray, collect: bool):
        candidate = np.flatnonzero(self.position[:, 2] < self.floor_top + 4.0 * self.radius)
        if candidate.size == 0:
            return None
        local = self.position[candidate] - self.floor_center[None, :]
        outside = np.abs(local) - self.floor_half[None, :]
        clamped = np.maximum(outside, 0.0)
        exterior = _norm_rows(clamped)
        interior = np.minimum(np.max(outside, axis=1), 0.0)
        distance = exterior + interior
        gap = distance - self.radius
        active = gap < 0.0
        if not np.any(active):
            return gap[gap < self.gap_report] if collect else None
        index = candidate[active]
        sign = np.sign(local[active])
        sign[sign == 0.0] = 1.0
        out_active = clamped[active]
        gradient = sign * out_active
        magnitude = _norm_rows(gradient)
        # Inside the box the gradient follows the least-penetrated axis.
        interior_hit = magnitude <= 1e-12
        if np.any(interior_hit):
            axis = np.argmax(outside[active][interior_hit], axis=1)
            replacement = np.zeros((int(np.count_nonzero(interior_hit)), 3))
            rows = np.arange(replacement.shape[0])
            replacement[rows, axis] = sign[interior_hit][rows, axis]
            gradient[interior_hit] = replacement
            magnitude[interior_hit] = 1.0
        normal = gradient / magnitude[:, None]
        self._wall_response(
            index,
            -gap[active],
            normal,
            self.stiffness_wall,
            self.damping_wall,
            self.friction_wall,
            force,
        )
        return gap[gap < self.gap_report] if collect else None

    # ------------------------------------------------------------------
    # supply hopper (axisymmetric inner profile, open outlet)
    # ------------------------------------------------------------------
    def _hopper_forces(self, force: np.ndarray, collect: bool):
        z = self.position[:, 2]
        band = (z > self.hopper_z_min - 6.0 * self.radius) & (
            z < self.hopper_z_max + 4.0 * self.radius
        )
        candidate = np.flatnonzero(band)
        if candidate.size == 0:
            return None
        x = self.position[candidate, 0]
        y = self.position[candidate, 1]
        rho = np.sqrt(x * x + y * y)
        inner = np.interp(
            self.position[candidate, 2], self.hopper_profile[:, 1], self.hopper_profile[:, 0]
        )
        near = rho > inner - 0.05
        if not np.any(near):
            return None
        candidate = candidate[near]
        rho = rho[near]
        x = x[near]
        y = y[near]
        zc = self.position[candidate, 2]
        distance, closest = self._profile_distance(rho, zc)
        inside = rho <= np.interp(zc, self.hopper_profile[:, 1], self.hopper_profile[:, 0])
        signed = np.where(inside, distance, -distance)
        gap = signed - self.radius
        active = gap < 0.0
        if not np.any(active):
            return gap[gap < self.gap_report] if collect else None
        index = candidate[active]
        safe_rho = np.maximum(rho[active], 1e-9)
        radial_x = x[active] / safe_rho
        radial_y = y[active] / safe_rho
        delta_rho = rho[active] - closest[active, 0]
        delta_z = zc[active] - closest[active, 1]
        length = np.sqrt(delta_rho * delta_rho + delta_z * delta_z)
        degenerate = length <= 1e-12
        if np.any(degenerate):
            delta_rho[degenerate] = np.where(inside[active][degenerate], -1.0, 1.0)
            delta_z[degenerate] = 0.0
            length[degenerate] = 1.0
        normal = np.empty((index.size, 3))
        normal[:, 0] = radial_x * delta_rho / length
        normal[:, 1] = radial_y * delta_rho / length
        normal[:, 2] = delta_z / length
        self._wall_response(
            index,
            -gap[active],
            normal,
            self.stiffness_wall,
            self.damping_wall,
            self.friction_wall,
            force,
        )
        return gap[gap < self.gap_report] if collect else None

    def _profile_distance(self, rho: np.ndarray, z: np.ndarray):
        """Distance and closest point to the inner-wall polyline in the (rho, z) half plane."""
        best_distance = np.full(rho.shape, np.inf)
        best_point = np.zeros((rho.size, 2))
        profile = self.hopper_profile
        for start, end in zip(profile[:-1], profile[1:]):
            edge = end - start
            length_squared = float(edge[0] * edge[0] + edge[1] * edge[1])
            if length_squared <= 0.0:
                continue
            parameter = ((rho - start[0]) * edge[0] + (z - start[1]) * edge[1]) / length_squared
            np.clip(parameter, 0.0, 1.0, out=parameter)
            point_rho = start[0] + parameter * edge[0]
            point_z = start[1] + parameter * edge[1]
            offset_rho = rho - point_rho
            offset_z = z - point_z
            distance = np.sqrt(offset_rho * offset_rho + offset_z * offset_z)
            closer = distance < best_distance
            best_distance[closer] = distance[closer]
            best_point[closer, 0] = point_rho[closer]
            best_point[closer, 1] = point_z[closer]
        return best_distance, best_point

    # ------------------------------------------------------------------
    # explicit time stepping
    # ------------------------------------------------------------------
    def _step(self, step_size: float) -> None:
        """One symplectic-Euler DEM step: forces, velocity update, position update."""
        self._maybe_rebuild()
        force = self.gravity_force.copy()
        self._pair_forces(force, False)
        self._floor_forces(force, False)
        self._hopper_forces(force, False)
        self.velocity += (step_size * self.inverse_mass)[:, None] * force
        speed = _norm_rows(self.velocity)
        fast = speed > self.SPEED_LIMIT
        if np.any(fast):
            self.velocity[fast] *= (self.SPEED_LIMIT / speed[fast])[:, None]
        peak = float(np.max(speed))
        if peak > self.max_speed_seen:
            self.max_speed_seen = peak
        self.position += step_size * self.velocity
        self._clamp_domain()
        self.step_count += 1

    def _clamp_domain(self) -> None:
        """Final safety net so a stray grain can never leave the frozen domain."""
        low = self.domain_min - 0.15
        high = np.maximum(self.domain_max, np.array([0.0, 0.0, self.hopper_z_max])) + 0.3
        below = self.position < low[None, :]
        if np.any(below):
            self.position[below] = np.broadcast_to(low, self.position.shape)[below]
            self.velocity[below] = 0.0
        above = self.position > high[None, :]
        if np.any(above):
            self.position[above] = np.broadcast_to(high, self.position.shape)[above]
            self.velocity[above] = 0.0

    def _collect_gaps(self) -> list:
        """Signed contact gaps at the current state, for the reported contact field."""
        scratch = np.zeros_like(self.position)
        self._maybe_rebuild()
        gaps = [
            self._pair_forces(scratch, True),
            self._floor_forces(scratch, True),
            self._hopper_forces(scratch, True),
        ]
        return [np.asarray(item, dtype=float).ravel() for item in gaps if item is not None]

    def _budget_check(self) -> None:
        """Coarsen the step only if the run would otherwise miss its wall-clock ceiling."""
        if self.degradations >= self.MAX_DEGRADATIONS or self.substeps <= 8:
            return
        duration = float(self.config.get("duration", 8.0))
        # Ignore the first samples so start-up overhead cannot skew the projection.
        if self.time < min(0.2, 0.05 * duration):
            return
        elapsed = _wallclock.perf_counter() - self.start_wallclock
        progress = max(self.time / max(duration, 1e-9), 1e-6)
        if elapsed / progress <= self.WALL_BUDGET_SECONDS:
            return
        self.substeps = max(8, self.substeps // 2)
        self.dt = self.sample_dt / self.substeps
        self.stiffness_pair *= 0.25
        self.stiffness_wall *= 0.25
        self.max_overlap = min(1.2 * self.radius, 4.0 * self.max_overlap)
        self._apply_contact_coefficients()
        self.degradations += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 1e-12:
            span = min(self.sample_dt, target - self.time)
            substeps = max(1, int(round(span / self.dt)))
            step_size = span / substeps
            for _ in range(substeps):
                self._step(step_size)
            self.time += span
            self._budget_check()
        self.time = max(self.time, target)
        return self._build_frame(target)

    def _build_frame(self, time: float) -> StateFrame:
        gaps = self._collect_gaps()
        contact = (
            np.concatenate(gaps)
            if gaps and sum(item.size for item in gaps) > 0
            else np.asarray([self.gap_report], dtype=float)
        )
        if contact.size == 0:
            contact = np.asarray([self.gap_report], dtype=float)
        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "nominal_radius": self.nominal_radius.copy(),
                "material_phase": self.material_phase.copy(),
            },
            fields={
                "contact_signed_distance": np.ascontiguousarray(contact, dtype=float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.gravity_force.copy(),
            },
            scalars={
                "contact_length_scale": float(self.contact_diameter),
                "initial_total_mass": float(self.total_mass),
                "total_mass": float(np.sum(self.mass)),
                "grain_radius": float(self.radius),
                "time_step": float(self.dt),
                "contact_count": float(contact.size),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        deposited = float(np.sum(self.mass[self.position[:, 2] < self.hopper_z_min]))
        return {
            "method": "soft_sphere_dem_symplectic_euler",
            "grain_count": int(self.count),
            "time_step": float(self.dt),
            "substeps_per_sample": int(self.substeps),
            "total_steps": int(self.step_count),
            "neighbour_rebuilds": int(self.rebuild_count),
            "stiffness_pair": float(self.stiffness_pair),
            "stiffness_wall": float(self.stiffness_wall),
            "damping_ratio": float(self.damping_ratio),
            "friction_pair": float(self.friction_pair),
            "friction_wall": float(self.friction_wall),
            "max_penetration": float(self.max_penetration),
            "max_speed": float(self.max_speed_seen),
            "total_mass": float(np.sum(self.mass)),
            "mass_below_outlet": deposited,
            "step_degradations": int(self.degradations),
            "wall_seconds": float(_wallclock.perf_counter() - self.start_wallclock),
        }
