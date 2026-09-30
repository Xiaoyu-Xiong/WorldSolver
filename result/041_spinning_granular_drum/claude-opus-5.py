"""Discrete-element solver for a half-filled spinning granular drum.

Physics
-------
Soft-sphere DEM (Cundall-Strack style, viscous-regularized friction):

* Grain-grain contact: Hookean normal spring ``kn * delta`` plus normal
  dashpot, tangential viscous friction capped by Coulomb ``mu * Fn``, and a
  directional rolling-resistance torque.
* Boundary: the drum is an inner cylindrical wall of radius ``R`` whose axis is
  the scene's y axis through the drum center, closed by two end plates at
  ``|y - cy| = depth/2``. The whole boundary is kinematic and spins at the
  frozen angular velocity ``Omega``, so the wall surface velocity
  ``Omega x (contact_point - center)`` enters the frictional contact law. That
  wall friction is what lifts grains up the rising side until they avalanche
  down the free surface.
* Time integration: semi-implicit (symplectic) Euler on translation and spin
  with a substep well below the Hertz-Rayleigh contact period.

Neighbor pairs come from a Verlet-skinned ``cKDTree``, rebuilt only when the
largest displacement since the last rebuild exceeds half the skin.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit DEM time-stepper for the frozen spinning-drum scene."""

    # Contact / material parameters (SI).
    NORMAL_STIFFNESS = 3.0e4          # N/m, grain-grain and grain-wall spring
    DAMPING_RATIO = 0.35              # normal dashpot as a fraction of critical
    TANGENTIAL_VISCOSITY = 50.0       # N s/m, regularizes stick below the cap
    WALL_TANGENTIAL_VISCOSITY = 80.0
    FRICTION_GRAIN = 0.6              # Coulomb coefficient, grain-grain
    FRICTION_WALL = 0.85              # drum wall grips grains and lifts them
    ROLLING_FRICTION = 0.08           # directional rolling-resistance torque
    TARGET_SUBSTEP = 1.2e-4           # s, ~35 substeps per contact period
    VERLET_SKIN = 0.010               # m
    LEAK_ALLOWANCE = 0.6              # projection sits outside the contact band
    MAX_SPEED = 15.0                  # m/s numerical safety clamp
    MAX_SPIN = 400.0                  # rad/s numerical safety clamp
    NEAR_CONTACT_BAND = 0.15          # gaps within 0.15*diameter are reported

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = dict(scene.config)
        initial = config["initial_conditions"]
        drum = initial["drum"]
        grains = initial["grains"]

        frame0 = scene.initial_state()
        self.position = np.array(frame0.position, dtype=np.float64)
        self.velocity = np.array(frame0.velocity, dtype=np.float64)
        self.spin = np.zeros_like(self.position)
        self.mass = np.array(frame0.mass, dtype=np.float64)
        self.body_id = np.array(frame0.body_id, dtype=np.int32)
        self.base_attributes = {
            key: np.array(value) for key, value in dict(frame0.attributes).items()
        }
        self.count = int(self.position.shape[0])

        self.radius = np.array(
            self.base_attributes.get(
                "particle_radius",
                np.full(self.count, float(grains["radius"])),
            ),
            dtype=np.float64,
        )
        self.inertia = 0.4 * self.mass * self.radius**2
        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        # Frozen drum geometry / kinematics.
        self.center = np.asarray(drum["center"], dtype=np.float64)
        self.drum_radius = float(drum["radius"])
        self.half_depth = 0.5 * float(drum["depth"])
        axis = np.asarray(drum.get("rotation_axis", [0.0, 1.0, 0.0]), dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        self.axis = axis / norm if norm > 0.0 else np.array([0.0, 1.0, 0.0])
        self.omega_scalar = float(drum["angular_velocity_rad_s"])
        self.omega = self.omega_scalar * self.axis

        # Contact coefficients derived from the reduced contact mass.
        mean_mass = float(np.mean(self.mass))
        self.pair_damping = 2.0 * self.DAMPING_RATIO * np.sqrt(
            self.NORMAL_STIFFNESS * 0.5 * mean_mass
        )
        self.wall_damping = 2.0 * self.DAMPING_RATIO * np.sqrt(
            self.NORMAL_STIFFNESS * mean_mass
        )
        self.contact_length_scale = float(2.0 * np.mean(self.radius))
        self.cutoff = float(2.0 * np.max(self.radius) + self.VERLET_SKIN)
        self.near_band = self.NEAR_CONTACT_BAND * self.contact_length_scale

        self._seat_initial_packing(seed)

        self.time = 0.0
        self.pair_i = np.empty(0, dtype=np.int64)
        self.pair_j = np.empty(0, dtype=np.int64)
        self._rebuild_reference = self.position.copy()
        self._rebuild_pairs()
        self._history: dict[str, list[float]] = {
            "time": [],
            "drum_angle": [],
            "lifted_grain_height": [],
            "free_surface_slope": [],
            "avalanche_layer_speed": [],
            "escaped_particle_count": [],
            "kinetic_energy": [],
        }
        self._steps = 0

    def _seat_initial_packing(self, seed: int) -> None:
        """Deterministic sub-gap jitter so the seeded lattice cannot lock as a crystal."""
        rng = np.random.default_rng(int(seed) + 977)
        self.position += rng.uniform(-8.0e-4, 8.0e-4, size=self.position.shape)
        self._project_inside_drum()

    def _project_inside_drum(self) -> None:
        """Hard non-penetration projection; guarantees grains never leak out."""
        local = self.position - self.center
        along = local @ self.axis
        radial = local - np.outer(along, self.axis)
        rho = np.linalg.norm(radial, axis=1)
        limit = self.drum_radius - self.radius - 1.0e-5
        outside = rho > limit
        if np.any(outside):
            scale = np.ones_like(rho)
            safe = np.maximum(rho, 1.0e-12)
            scale[outside] = limit[outside] / safe[outside]
            radial = radial * scale[:, None]
        cap = self.half_depth - self.radius - 1.0e-5
        along = np.clip(along, -cap, cap)
        self.position = self.center + radial + np.outer(along, self.axis)

    # -------------------------------------------------------- neighbor search
    def _rebuild_pairs(self) -> None:
        tree = cKDTree(self.position)
        pairs = tree.query_pairs(self.cutoff, output_type="ndarray")
        if pairs.size:
            self.pair_i = pairs[:, 0].astype(np.int64)
            self.pair_j = pairs[:, 1].astype(np.int64)
        else:
            self.pair_i = np.empty(0, dtype=np.int64)
            self.pair_j = np.empty(0, dtype=np.int64)
        self._rebuild_reference = self.position.copy()

    def _maybe_rebuild_pairs(self) -> None:
        drift = self.position - self._rebuild_reference
        if float(np.max(np.einsum("ij,ij->i", drift, drift))) > (0.5 * self.VERLET_SKIN) ** 2:
            self._rebuild_pairs()

    def _scatter(self, index, values, out) -> None:
        """Accumulate per-pair vector contributions into per-particle arrays."""
        for axis in range(3):
            out[:, axis] += np.bincount(index, weights=values[:, axis], minlength=self.count)

    # ------------------------------------------------------------------ forces
    def _pair_forces(self, force, torque):
        """Grain-grain normal spring/dashpot, capped Coulomb friction, rolling drag."""
        i, j = self.pair_i, self.pair_j
        if i.size == 0:
            return np.empty(0, dtype=np.float64)
        delta = self.position[j] - self.position[i]
        distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        radius_sum = self.radius[i] + self.radius[j]
        gap = distance - radius_sum
        active = (gap < 0.0) & (distance > 1.0e-12)
        if not np.any(active):
            return gap
        ai, aj = i[active], j[active]
        overlap = -gap[active]
        normal = delta[active] / distance[active][:, None]

        relative = self.velocity[aj] - self.velocity[ai]
        normal_rate = np.einsum("ij,ij->i", relative, normal)
        normal_force = self.NORMAL_STIFFNESS * overlap - self.pair_damping * normal_rate
        normal_force = np.maximum(normal_force, 0.0)

        # Surface-relative slip includes spin at both contact points.
        spin_term = np.cross(
            self.radius[ai][:, None] * self.spin[ai] + self.radius[aj][:, None] * self.spin[aj],
            normal,
        )
        surface_relative = relative - spin_term
        slip = surface_relative - np.einsum("ij,ij->i", surface_relative, normal)[:, None] * normal
        slip_speed = np.sqrt(np.einsum("ij,ij->i", slip, slip))
        safe_slip = np.maximum(slip_speed, 1.0e-12)
        friction_magnitude = np.minimum(
            self.TANGENTIAL_VISCOSITY * slip_speed, self.FRICTION_GRAIN * normal_force
        )
        tangent = slip / safe_slip[:, None]
        friction = friction_magnitude[:, None] * tangent

        force_on_i = -normal_force[:, None] * normal + friction
        self._scatter(ai, force_on_i, force)
        self._scatter(aj, -force_on_i, force)

        moment_axis = np.cross(normal, tangent)
        torque_i = (self.radius[ai] * friction_magnitude)[:, None] * moment_axis
        torque_j = (self.radius[aj] * friction_magnitude)[:, None] * moment_axis
        self._scatter(ai, torque_i, torque)
        self._scatter(aj, torque_j, torque)

        if self.ROLLING_FRICTION > 0.0:
            spin_relative = self.spin[ai] - self.spin[aj]
            spin_norm = np.sqrt(np.einsum("ij,ij->i", spin_relative, spin_relative))
            effective_radius = (self.radius[ai] * self.radius[aj]) / radius_sum[active]
            scale = self.ROLLING_FRICTION * normal_force * effective_radius
            rolling = -(scale / np.maximum(spin_norm, 1.0e-9))[:, None] * spin_relative
            self._scatter(ai, rolling, torque)
            self._scatter(aj, -rolling, torque)
        return gap

    def _wall_forces(self, force, torque):
        """Kinematic drum: inner cylindrical wall plus two rotating end plates."""
        local = self.position - self.center
        along = local @ self.axis
        radial = local - np.outer(along, self.axis)
        rho = np.sqrt(np.einsum("ij,ij->i", radial, radial))
        safe_rho = np.maximum(rho, 1.0e-12)
        unit_radial = radial / safe_rho[:, None]

        cyl_gap = (self.drum_radius - rho) - self.radius
        gaps = [cyl_gap]
        active = np.flatnonzero(cyl_gap < 0.0)
        if active.size:
            overlap = -cyl_gap[active]
            inward = -unit_radial[active]
            contact = unit_radial[active] * self.drum_radius + np.outer(along[active], self.axis)
            wall_velocity = np.cross(np.broadcast_to(self.omega, contact.shape), contact)
            arm = self.radius[active][:, None] * unit_radial[active]
            surface_velocity = self.velocity[active] + np.cross(self.spin[active], arm)
            relative = surface_velocity - wall_velocity

            normal_rate = np.einsum("ij,ij->i", relative, inward)
            normal_force = np.maximum(
                self.NORMAL_STIFFNESS * overlap - self.wall_damping * normal_rate, 0.0
            )
            slip = relative - normal_rate[:, None] * inward
            slip_speed = np.sqrt(np.einsum("ij,ij->i", slip, slip))
            friction_magnitude = np.minimum(
                self.WALL_TANGENTIAL_VISCOSITY * slip_speed, self.FRICTION_WALL * normal_force
            )
            friction = -(friction_magnitude / np.maximum(slip_speed, 1.0e-12))[:, None] * slip

            total = normal_force[:, None] * inward + friction
            self._scatter(active, total, force)
            self._scatter(active, np.cross(arm, friction), torque)

            if self.ROLLING_FRICTION > 0.0:
                spin_relative = self.spin[active] - self.omega
                spin_norm = np.sqrt(np.einsum("ij,ij->i", spin_relative, spin_relative))
                scale = self.ROLLING_FRICTION * normal_force * self.radius[active]
                rolling = -(scale / np.maximum(spin_norm, 1.0e-9))[:, None] * spin_relative
                self._scatter(active, rolling, torque)

        for sign in (1.0, -1.0):
            plate_gap = (self.half_depth - sign * along) - self.radius
            gaps.append(plate_gap)
            hits = np.flatnonzero(plate_gap < 0.0)
            if hits.size == 0:
                continue
            overlap = -plate_gap[hits]
            inward = np.broadcast_to(-sign * self.axis, (hits.size, 3))
            contact = radial[hits] + sign * self.half_depth * self.axis
            wall_velocity = np.cross(np.broadcast_to(self.omega, contact.shape), contact)
            arm = -self.radius[hits][:, None] * inward
            surface_velocity = self.velocity[hits] + np.cross(self.spin[hits], arm)
            relative = surface_velocity - wall_velocity

            normal_rate = np.einsum("ij,ij->i", relative, inward)
            normal_force = np.maximum(
                self.NORMAL_STIFFNESS * overlap - self.wall_damping * normal_rate, 0.0
            )
            slip = relative - normal_rate[:, None] * inward
            slip_speed = np.sqrt(np.einsum("ij,ij->i", slip, slip))
            friction_magnitude = np.minimum(
                self.WALL_TANGENTIAL_VISCOSITY * slip_speed, self.FRICTION_WALL * normal_force
            )
            friction = -(friction_magnitude / np.maximum(slip_speed, 1.0e-12))[:, None] * slip

            total = normal_force[:, None] * inward + friction
            self._scatter(hits, total, force)
            self._scatter(hits, np.cross(arm, friction), torque)
        return gaps

    # ------------------------------------------------------------ integration
    def _external_force(self):
        return self.mass[:, None] * self.gravity[None, :]

    def _step(self, dt: float) -> None:
        self._maybe_rebuild_pairs()
        force = self._external_force()
        torque = np.zeros_like(self.position)
        self._pair_forces(force, torque)
        self._wall_forces(force, torque)

        self.velocity += dt * (force / self.mass[:, None])
        self.spin += dt * (torque / self.inertia[:, None])

        speed = np.sqrt(np.einsum("ij,ij->i", self.velocity, self.velocity))
        excess = speed > self.MAX_SPEED
        if np.any(excess):
            self.velocity[excess] *= (self.MAX_SPEED / speed[excess])[:, None]
        spin_norm = np.sqrt(np.einsum("ij,ij->i", self.spin, self.spin))
        hot = spin_norm > self.MAX_SPIN
        if np.any(hot):
            self.spin[hot] *= (self.MAX_SPIN / spin_norm[hot])[:, None]

        self.position += dt * self.velocity
        self._enforce_containment()
        self._steps += 1

    def _enforce_containment(self) -> None:
        """Leak-proof safety net that must never pre-empt wall contact.

        The allowance keeps this projection strictly *outside* the contact band:
        grains may overlap the wall by up to ``LEAK_ALLOWANCE * radius``, which is
        what lets the wall spring and its Coulomb friction engage and lift
        material. Only grains that would otherwise escape get projected back.
        """
        local = self.position - self.center
        along = local @ self.axis
        radial = local - np.outer(along, self.axis)
        rho = np.sqrt(np.einsum("ij,ij->i", radial, radial))
        limit = self.drum_radius - (1.0 - self.LEAK_ALLOWANCE) * self.radius

        outside = np.flatnonzero(rho > limit)
        if outside.size:
            unit = radial[outside] / np.maximum(rho[outside], 1.0e-12)[:, None]
            radial[outside] = unit * limit[outside][:, None]
            outward = np.einsum("ij,ij->i", self.velocity[outside], unit)
            self.velocity[outside] -= np.maximum(outward, 0.0)[:, None] * unit

        cap = self.half_depth - (1.0 - self.LEAK_ALLOWANCE) * self.radius
        beyond = np.flatnonzero(np.abs(along) > cap)
        if beyond.size:
            axial_velocity = self.velocity[beyond] @ self.axis
            sign = np.sign(along[beyond])
            remove = np.maximum(sign * axial_velocity, 0.0) * sign
            self.velocity[beyond] -= remove[:, None] * self.axis[None, :]
            along[beyond] = np.clip(along[beyond], -cap[beyond], cap[beyond])

        self.position = self.center + radial + np.outer(along, self.axis)

    # ------------------------------------------------------------ observables
    def _contact_gaps(self) -> np.ndarray:
        """Signed gaps at active and near-contact grain-grain and wall samples."""
        collected = []
        if self.pair_i.size:
            delta = self.position[self.pair_j] - self.position[self.pair_i]
            distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            gap = distance - (self.radius[self.pair_i] + self.radius[self.pair_j])
            collected.append(gap[gap < self.near_band])

        local = self.position - self.center
        along = local @ self.axis
        radial = local - np.outer(along, self.axis)
        rho = np.sqrt(np.einsum("ij,ij->i", radial, radial))
        wall_gap = (self.drum_radius - rho) - self.radius
        collected.append(wall_gap[wall_gap < self.near_band])
        for sign in (1.0, -1.0):
            plate_gap = (self.half_depth - sign * along) - self.radius
            collected.append(plate_gap[plate_gap < self.near_band])

        gaps = np.concatenate(collected) if collected else np.empty(0, dtype=np.float64)
        if gaps.size == 0:
            gaps = np.array([float(np.min(wall_gap))], dtype=np.float64)
        if gaps.size > 24000:
            gaps = gaps[np.linspace(0, gaps.size - 1, 24000, dtype=np.int64)]
        return np.ascontiguousarray(gaps, dtype=np.float64)

    def _surface_observables(self):
        """Free-surface slope and speed of the flowing layer on top of the bed."""
        local = self.position - self.center
        x = local[:, 0]
        z = local[:, 2]
        speed = np.sqrt(np.einsum("ij,ij->i", self.velocity, self.velocity))
        edges = np.linspace(-0.6 * self.drum_radius, 0.6 * self.drum_radius, 9)
        centers, tops = [], []
        surface_speed = []
        band = 2.5 * self.contact_length_scale
        for lo, hi in zip(edges[:-1], edges[1:]):
            inside = np.flatnonzero((x >= lo) & (x < hi))
            if inside.size < 4:
                continue
            top = float(np.percentile(z[inside], 96.0))
            centers.append(0.5 * (lo + hi))
            tops.append(top)
            near = inside[z[inside] > top - band]
            if near.size:
                surface_speed.append(float(np.mean(speed[near])))
        if len(centers) >= 2:
            slope = float(np.polyfit(np.asarray(centers), np.asarray(tops), 1)[0])
        else:
            slope = 0.0
        layer_speed = float(np.mean(surface_speed)) if surface_speed else 0.0
        return slope, layer_speed

    def _lifted_height(self) -> float:
        """Height reached by grains on the rising side of the drum wall."""
        local = self.position - self.center
        rising = np.cross(self.omega, np.array([0.0, 0.0, -1.0]))
        rising_axis = rising / max(float(np.linalg.norm(rising)), 1e-12)
        side = local @ rising_axis
        selected = np.flatnonzero(side > 0.25 * self.drum_radius)
        if selected.size == 0:
            return float(np.max(local[:, 2]))
        return float(np.percentile(local[selected, 2], 98.0))

    def _escaped_count(self) -> float:
        local = self.position - self.center
        along = local @ self.axis
        radial = local - np.outer(along, self.axis)
        rho = np.sqrt(np.einsum("ij,ij->i", radial, radial))
        escaped = (rho > self.drum_radius + 0.5 * self.radius) | (
            np.abs(along) > self.half_depth + 0.5 * self.radius
        )
        return float(np.count_nonzero(escaped))

    # ------------------------------------------------------------------ frames
    def _make_frame(self, time: float) -> StateFrame:
        gaps = self._contact_gaps()
        external = self._external_force()
        slope, layer_speed = self._surface_observables()
        lifted = self._lifted_height()
        escaped = self._escaped_count()
        drum_angle = float(self.omega_scalar * time)
        kinetic = float(
            0.5 * np.sum(self.mass * np.einsum("ij,ij->i", self.velocity, self.velocity))
        )

        attributes = {key: np.array(value) for key, value in self.base_attributes.items()}
        attributes["mass_quantity"] = np.ascontiguousarray(self.mass, dtype=np.float64)
        attributes["speed"] = np.sqrt(
            np.einsum("ij,ij->i", self.velocity, self.velocity)
        ).astype(np.float64)

        fields = {
            "contact_signed_distance": gaps,
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "external_force": np.ascontiguousarray(external, dtype=np.float64),
        }
        scalars = {
            "contact_length_scale": float(self.contact_length_scale),
            "drum_radius": float(self.drum_radius),
            "drum_depth": float(2.0 * self.half_depth),
            "drum_angular_velocity_rad_s": float(self.omega_scalar),
            "drum_angle": drum_angle,
            "lifted_grain_height": lifted,
            "free_surface_slope": slope,
            "free_surface_angle_deg": float(np.degrees(np.arctan(slope))),
            "avalanche_layer_speed": layer_speed,
            "escaped_particle_count": escaped,
            "kinetic_energy": kinetic,
            "total_mass": float(np.sum(self.mass)),
            "particle_count": float(self.count),
            "contact_count": float(gaps.size),
        }

        self._history["time"].append(float(time))
        self._history["drum_angle"].append(drum_angle)
        self._history["lifted_grain_height"].append(lifted)
        self._history["free_surface_slope"].append(slope)
        self._history["avalanche_layer_speed"].append(layer_speed)
        self._history["escaped_particle_count"].append(escaped)
        self._history["kinetic_energy"].append(kinetic)

        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(self.position, dtype=np.float64),
            velocity=np.ascontiguousarray(self.velocity, dtype=np.float64),
            angular_velocity=np.ascontiguousarray(self.spin, dtype=np.float64),
            mass=np.ascontiguousarray(self.mass, dtype=np.float64),
            body_id=np.ascontiguousarray(self.body_id, dtype=np.int32),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.time:
            span = target - self.time
            substeps = max(1, int(np.ceil(span / self.TARGET_SUBSTEP)))
            dt = span / substeps
            for _ in range(substeps):
                self._step(dt)
            self.time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        history = {key: list(map(float, value)) for key, value in self._history.items()}
        slopes = np.asarray(history["free_surface_slope"]) if history["free_surface_slope"] else np.zeros(1)
        speeds = np.asarray(history["avalanche_layer_speed"]) if history["avalanche_layer_speed"] else np.zeros(1)
        lifted = np.asarray(history["lifted_grain_height"]) if history["lifted_grain_height"] else np.zeros(1)
        return {
            "method": "explicit_soft_sphere_DEM_rotating_drum",
            "particle_count": int(self.count),
            "substeps_taken": int(self._steps),
            "substep_size": float(self.TARGET_SUBSTEP),
            "normal_stiffness": float(self.NORMAL_STIFFNESS),
            "friction_grain": float(self.FRICTION_GRAIN),
            "friction_wall": float(self.FRICTION_WALL),
            "contact_length_scale": float(self.contact_length_scale),
            "drum_angle_final": float(history["drum_angle"][-1]) if history["drum_angle"] else 0.0,
            "max_free_surface_angle_deg": float(np.degrees(np.arctan(np.max(np.abs(slopes))))),
            "mean_avalanche_layer_speed": float(np.mean(speeds)),
            "max_lifted_grain_height": float(np.max(lifted)),
            "escaped_particle_count_max": (
                int(max(history["escaped_particle_count"]))
                if history["escaped_particle_count"]
                else 0
            ),
            "history": history,
        }
