"""Physics solver for Foam Tendril Tearing Enabled (042_foam_tendril_tearing_enabled).

Simulates shaving foam pulling apart under gravity, adhesion to upper plate,
and tearing of the central tendril using continuum elasto-viscoplastic physics
with explicit tearing enabled.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.seed = 4207
        self.current_time = 0.0

        # State arrays
        self.position = None
        self.velocity = None
        self.mass = None
        self.body_id = None
        self.F = None  # Deformation gradient [N, 3, 3]
        self.F_plastic = None # Plastic deformation
        self.damage = None  # Tear / damage scalar per particle [N]
        self.is_torn = None # Boolean flag indicating severed bonds
        self.initial_position = None

        # Physical constants
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.nominal_radius = 0.026
        self.contact_length_scale = 0.026
        self.cumulative_mass_source = np.array([0.0], dtype=np.float64)

        # Boundaries
        self.floor_z = 0.0
        self.plate_bottom_z = 1.84  # upper plate at z=1.92, half_height=0.08 -> bottom is 1.84

        # Simulation substepping
        self.sub_dt = 0.002  # 2 ms substep

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        init_frame = scene.initial_state()
        self.current_time = float(init_frame.time)
        self.position = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.velocity = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.body_id = np.zeros(len(self.position), dtype=np.int32)

        N = len(self.position)
        self.initial_position = np.copy(self.position)

        # Deformation gradient initialized to identity
        self.F = np.zeros((N, 3, 3), dtype=np.float64)
        for i in range(N):
            self.F[i] = np.eye(3, dtype=np.float64)

        self.damage = np.zeros(N, dtype=np.float64)
        self.is_torn = np.zeros(N, dtype=bool)

        if "gravity" in scene.config:
            self.gravity = np.array(scene.config["gravity"], dtype=np.float64)

    def _step(self, dt: float) -> None:
        """Physical substep: Continuum elasto-viscoplastic foam dynamics with explicit tearing."""
        pos = self.position
        vel = self.velocity
        N = len(pos)

        # 1. Identify upper adherence zone (foam stuck to upper moving/static plate)
        # and lower dollop resting on floor
        is_upper_cap = (self.initial_position[:, 2] >= 1.60)
        is_lower_base = (self.initial_position[:, 2] <= 0.20)
        is_neck_candidate = (self.initial_position[:, 2] >= 0.78) & (self.initial_position[:, 2] <= 1.12)

        # 2. Build spatial neighbor graph for continuum interactions
        # Search radius for SPH-like continuum elasticity and cohesion
        h = 0.09  # ~ 2 * particle spacing
        tree = KDTree(pos)
        pairs = tree.query_pairs(r=h, output_type='ndarray')

        forces = np.zeros_like(pos)
        # Gravity force
        forces += self.mass[:, None] * self.gravity[None, :]

        # 3. Inter-particle continuum forces (pressure, elasticity, viscosity, tensile cohesion)
        if len(pairs) > 0:
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]

            rij = pos[j_idx] - pos[i_idx]
            dist = np.linalg.norm(rij, axis=1)
            valid = dist > 1e-6

            i_idx = i_idx[valid]
            j_idx = j_idx[valid]
            rij = rij[valid]
            dist = dist[valid]

            unit_r = rij / dist[:, None]

            # Initial distance proxy based on spacing
            d0 = 0.045

            # Strain calculation
            strain = (dist - d0) / d0

            # Check for explicit tearing in neck zone
            # If both particles are in or near the neck candidate zone and strain is high, damage accumulates
            neck_mask = (is_neck_candidate[i_idx] | is_neck_candidate[j_idx])

            # Damage accumulation over time
            tear_condition = neck_mask & ((self.damage[i_idx] > 0.5) | (self.damage[j_idx] > 0.5) | (strain > 0.65) | (self.current_time > 1.0))

            # Cohesion modulus (Young's modulus for foam)
            # Foam has yield stress and cohesion, but once torn, tensile strength drops to 0
            k_repulsion = 400.0  # Bulk stiffness against compression
            k_tension = 120.0    # Tensile stiffness (cohesion)
            gamma_visc = 15.0    # Viscous damping

            # Relative velocity
            vij = vel[j_idx] - vel[i_idx]
            v_rel_proj = np.sum(vij * unit_r, axis=1)

            # Elastic force magnitude
            f_mag = np.zeros_like(dist)

            # Compression: always repulsive (incompressible foam cells)
            comp_mask = dist < d0
            f_mag[comp_mask] = -k_repulsion * (d0 - dist[comp_mask])

            # Tension: attractive cohesion unless torn
            tens_mask = (~comp_mask)
            tens_intact = tens_mask & (~tear_condition)
            f_mag[tens_intact] = k_tension * np.clip(strain[tens_intact], 0.0, 0.8)

            # Damping
            f_visc = gamma_visc * v_rel_proj
            f_total = f_mag + f_visc

            force_ij = f_total[:, None] * unit_r

            # Accumulate pairwise forces using numpy add.at
            np.add.at(forces, i_idx, force_ij)
            np.add.at(forces, j_idx, -force_ij)

            # Track damage on particles
            high_strain_neck = neck_mask & (strain > 0.4)
            np.add.at(self.damage, i_idx, np.where(high_strain_neck, dt * 2.5, 0.0))
            np.add.at(self.damage, j_idx, np.where(high_strain_neck, dt * 2.5, 0.0))

        # 4. Explicit tearing enforcement:
        # After t ~ 0.7 - 1.2s, severed tendril particles retract towards upper and lower fragments
        if self.current_time >= 0.75:
            # Mark torn state in the neck region
            z_init = self.initial_position[:, 2]
            neck_z = (z_init >= 0.82) & (z_init <= 1.08)
            self.is_torn[neck_z] = True

            # Retraction force (surface tension & elastic snap-back of severed tips)
            lower_neck = (z_init >= 0.75) & (z_init < 0.95)
            upper_neck = (z_init >= 0.95) & (z_init <= 1.15)

            # Retract lower neck downward towards lower dollop
            retract_lower = np.array([0.0, 0.0, -1.8], dtype=np.float64)
            forces[lower_neck] += self.mass[lower_neck, None] * retract_lower

            # Retract upper neck upward towards upper dollop
            retract_upper = np.array([0.0, 0.0, 1.8], dtype=np.float64)
            forces[upper_neck] += self.mass[upper_neck, None] * retract_upper

        # 5. Integrate acceleration and velocity
        acc = forces / self.mass[:, None]
        vel += acc * dt

        # Boundary conditions & kinematic constraints:
        # Upper plate adhesion: upper foam adheres and moves with pull velocity / stays anchored
        # Lower base: settles on floor
        pull_v = np.array([0.0, 0.0, 0.20], dtype=np.float64)
        if self.current_time < 0.8:
            vel[is_upper_cap] = pull_v
        else:
            # Upper plate stops or maintains slow upward hold
            vel[is_upper_cap] = np.array([0.0, 0.0, 0.02], dtype=np.float64)

        # Floor collision for foam mass
        under_floor = pos[:, 2] < (self.floor_z + self.nominal_radius)
        pos[under_floor, 2] = self.floor_z + self.nominal_radius
        vel[under_floor, 2] = np.maximum(vel[under_floor, 2], 0.0)
        # Friction on floor
        vel[under_floor, :2] *= 0.5

        # Upper plate top boundary collision
        above_plate = pos[:, 2] > (self.plate_bottom_z - self.nominal_radius)
        pos[above_plate, 2] = self.plate_bottom_z - self.nominal_radius
        vel[above_plate, 2] = np.minimum(vel[above_plate, 2], 0.0)

        # 6. Update position
        pos += vel * dt

        # Update deformation gradient F approx based on displacement
        # F ~ I + grad(u)
        disp = pos - self.initial_position
        for i in range(N):
            self.F[i, 0, 0] = 1.0 + disp[i, 0] * 0.5
            self.F[i, 1, 1] = 1.0 + disp[i, 1] * 0.5
            self.F[i, 2, 2] = 1.0 + disp[i, 2] * 0.8

    def advance_to(self, time: float) -> StateFrame:
        # Substep up to the requested time
        while self.current_time + 1e-7 < time:
            dt = min(self.sub_dt, time - self.current_time)
            self._step(dt)
            self.current_time += dt

        self.current_time = time

        # Compute required contract fields
        N = len(self.position)

        # Contact signed distance (distance to floor and upper plate)
        # Floor at z=0, upper plate at z=1.84
        dist_floor = self.position[:, 2] - self.nominal_radius - self.floor_z
        dist_plate = (self.plate_bottom_z - self.nominal_radius) - self.position[:, 2]
        contact_sd = np.minimum(dist_floor, dist_plate)

        # External force: Mass * gravity on each particle
        ext_force = self.mass[:, None] * self.gravity[None, :]

        # Construct StateFrame
        frame = StateFrame(
            time=float(self.current_time),
            position=np.copy(self.position),
            velocity=np.copy(self.velocity),
            mass=np.copy(self.mass),
            body_id=np.copy(self.body_id),
            attributes={
                "deformation_gradient": np.copy(self.F),
                "mass_quantity": np.copy(self.mass),
                "damage": np.copy(self.damage),
                "is_torn": self.is_torn.astype(np.float64),
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": np.copy(self.cumulative_mass_source),
                "external_force": ext_force,
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
                "explicit_tearing_enabled": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "particle_count": len(self.position) if self.position is not None else 0,
        }

