"""Explicit particle-bond solver for the sticky snowball-wall scene.

The snowball is modelled as a network of equal-mass particles connected by
fracturable cohesive bonds.  Sticky contact attaches particles that touch the
vertical wall; the remaining material is free to rebound, break internal bonds,
and fall under gravity.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physically time-step the granular snowball with sticky wall contact."""

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.config = scene.config

        # Time stepping
        self.sample_dt = float(self.config["sample_dt"])
        self.substeps = 120
        self.dt_sub = self.sample_dt / self.substeps

        # Environment
        gravity = np.asarray(self.config["gravity"], dtype=float)
        self.gravity = gravity
        contact_cfg = self.config["contact"]
        self.wall_inner_x = float(contact_cfg["wall_inner_x"])
        self.floor_z = float(contact_cfg["floor_z"])
        self.contact_length_scale = float(contact_cfg["contact_length_scale"])

        # Initial state from the frozen scene contract
        frame0 = scene.initial_state()
        self.current_time = 0.0
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.volume = np.asarray(frame0.attributes["reference_volume"], dtype=float).copy()
        self.deformation_gradient = np.asarray(
            frame0.attributes["deformation_gradient"], dtype=float
        ).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int64).copy()
        self.component_id = np.asarray(
            frame0.attributes["initial_component_id"], dtype=np.int64
        ).copy()
        self.n = self.position.shape[0]

        # Derived particle size and area
        self.particle_radius = ((3.0 * self.volume) / (4.0 * np.pi)) ** (1.0 / 3.0)
        self.particle_radius = float(self.particle_radius.mean())
        self.particle_area = np.pi * self.particle_radius * self.particle_radius
        self.spacing = float(self.config["snowball"]["particle_spacing"])
        self.mass_particle = float(self.mass.mean())

        # Cohesive bond material parameters
        self.young_modulus = 1.0e7  # Pa
        self.k_bond = self.young_modulus * self.particle_area / self.spacing
        self.zeta_bond = 0.4
        self.c_bond = (
            2.0
            * self.zeta_bond
            * np.sqrt(self.k_bond * 0.5 * self.mass_particle)
        )
        self.break_strain = 0.35

        # Rigid boundary contact parameters
        self.k_wall = 1.0e5
        self.k_floor = 1.0e5
        self.zeta_contact = 0.7
        self.c_wall = 2.0 * self.zeta_contact * np.sqrt(self.k_wall * self.mass_particle)
        self.c_floor = 2.0 * self.zeta_contact * np.sqrt(self.k_floor * self.mass_particle)
        self.mu_floor = 0.5
        self.mu_wall = 0.3

        # Build the initial cohesive bond network
        tree = cKDTree(self.position)
        pairs = tree.query_pairs(r=self.spacing * 1.5, output_type="ndarray")
        self.i_idx = pairs[:, 0].astype(np.int64)
        self.j_idx = pairs[:, 1].astype(np.int64)
        self.rest_length = np.linalg.norm(
            self.position[self.j_idx] - self.position[self.i_idx], axis=1
        )
        self.broken = np.zeros(len(self.i_idx), dtype=bool)

        # Sticky wall state
        self.stuck = np.zeros(self.n, dtype=bool)

        # Preserve frozen scalars
        self.initial_total_mass = float(frame0.scalars["initial_total_mass"])
        self.initial_total_volume = float(frame0.scalars["initial_total_volume"])

    def _compute_bond_forces(self, force: np.ndarray) -> None:
        """Add cohesive/fracturable bond forces to the per-particle force array."""
        if len(self.i_idx) == 0:
            return

        dx = self.position[self.j_idx] - self.position[self.i_idx]
        dist = np.linalg.norm(dx, axis=1)
        # Avoid division by zero for coincident particles
        dist = np.where(dist < 1.0e-12, 1.0e-12, dist)
        strain = (dist - self.rest_length) / self.rest_length

        # Break bonds that have been stretched beyond the fracture limit
        break_now = (~self.broken) & (strain > self.break_strain)
        self.broken[break_now] = True

        active = ~self.broken
        if not np.any(active):
            return

        dx_a = dx[active]
        dist_a = dist[active]
        rest_a = self.rest_length[active]
        i_a = self.i_idx[active]
        j_a = self.j_idx[active]

        direction = dx_a / dist_a[:, None]
        vrel = self.velocity[j_a] - self.velocity[i_a]
        vn = np.einsum("ij,ij->i", vrel, direction)

        fmag = self.k_bond * (dist_a - rest_a) - self.c_bond * vn
        fvec = fmag[:, None] * direction

        np.add.at(force, i_a, fvec)
        np.add.at(force, j_a, -fvec)

    def _compute_boundary_forces(self, force: np.ndarray) -> None:
        """Add wall and floor contact forces; mark particles as stuck on the wall."""
        # ----- Sticky wall -----
        gap_wall = self.wall_inner_x - self.position[:, 0]
        near_wall = gap_wall < self.particle_radius
        if np.any(near_wall):
            # Any particle that reaches the wall becomes permanently attached there
            new_stuck = near_wall & (~self.stuck)
            self.stuck |= new_stuck

        # Only apply wall contact impulses to particles that are not yet stuck
        unstuck_near_wall = near_wall & (~self.stuck)
        if np.any(unstuck_near_wall):
            penetration = -gap_wall[unstuck_near_wall]
            penetration = np.maximum(penetration, 0.0)
            f_x = -self.k_wall * penetration - self.c_wall * self.velocity[unstuck_near_wall, 0]
            force[unstuck_near_wall, 0] += f_x

            # Tangential Coulomb friction against the wall
            v_t = self.velocity[unstuck_near_wall, 1:3]
            v_t_norm = np.linalg.norm(v_t, axis=1)
            normal_mag = self.k_wall * penetration
            f_t_mag = np.minimum(self.mu_wall * normal_mag, v_t_norm / self.dt_sub * self.mass_particle)
            mask = v_t_norm > 1.0e-12
            f_t = np.zeros_like(v_t)
            f_t[mask] = -(
                f_t_mag[mask][:, None]
                * v_t[mask] / v_t_norm[mask][:, None]
            )
            force[unstuck_near_wall, 1:3] += f_t

        # ----- Floor -----
        gap_floor = self.position[:, 2] - self.floor_z
        near_floor = gap_floor < self.particle_radius
        if np.any(near_floor):
            penetration = self.particle_radius - gap_floor[near_floor]
            penetration = np.maximum(penetration, 0.0)
            f_z = self.k_floor * penetration - self.c_floor * self.velocity[near_floor, 2]
            force[near_floor, 2] += f_z

            # Tangential Coulomb friction on the floor
            v_t = self.velocity[near_floor, :2]
            v_t_norm = np.linalg.norm(v_t, axis=1)
            normal_mag = self.k_floor * penetration + self.c_floor * np.maximum(
                -self.velocity[near_floor, 2], 0.0
            )
            f_t_mag = np.minimum(
                self.mu_floor * normal_mag,
                v_t_norm / self.dt_sub * self.mass_particle,
            )
            mask = v_t_norm > 1.0e-12
            f_t = np.zeros_like(v_t)
            f_t[mask] = -(
                f_t_mag[mask][:, None] * v_t[mask] / v_t_norm[mask][:, None]
            )
            force[near_floor, :2] += f_t

    def _substep(self) -> None:
        """Advance the particle state by one sub-step using symplectic Euler."""
        force = self.mass[:, None] * self.gravity[None, :]

        self._compute_bond_forces(force)
        self._compute_boundary_forces(force)

        acceleration = force / self.mass[:, None]
        self.velocity += acceleration * self.dt_sub
        self.position += self.velocity * self.dt_sub

        # Enforce sticky-wall constraint: stuck particles are motionless and
        # cannot move past the wall or below the floor
        if np.any(self.stuck):
            self.velocity[self.stuck] = 0.0
            self.position[self.stuck, 0] = np.minimum(
                self.position[self.stuck, 0],
                self.wall_inner_x - self.particle_radius,
            )
            self.position[self.stuck, 2] = np.maximum(
                self.position[self.stuck, 2], self.particle_radius
            )

    def _update_components(self) -> None:
        """Label connected components of the unbroken bond graph."""
        if len(self.i_idx) == 0:
            self.component_id = np.zeros(self.n, dtype=np.int64)
            return
        active = ~self.broken
        rows = np.concatenate([self.i_idx[active], self.j_idx[active]])
        cols = np.concatenate([self.j_idx[active], self.i_idx[active]])
        data = np.ones(2 * int(active.sum()), dtype=np.int64)
        graph = csr_matrix((data, (rows, cols)), shape=(self.n, self.n))
        _, labels = connected_components(graph, directed=False, return_labels=True)
        self.component_id = labels.astype(np.int64)

    def _build_frame(self, time: float) -> StateFrame:
        """Assemble a StateFrame that satisfies the frozen contract."""
        self._update_components()

        # Signed distances to the wall and floor
        wall_gap = self.wall_inner_x - self.position[:, 0]
        floor_gap = self.position[:, 2] - self.floor_z

        # Active/near contact samples for the required trajectory field
        margin = 1.5 * self.contact_length_scale
        wall_active = wall_gap < margin
        floor_active = floor_gap < margin
        contact_signed_distance = np.concatenate(
            [wall_gap[wall_active], floor_gap[floor_active]]
        ).astype(float)
        if contact_signed_distance.size == 0:
            contact_signed_distance = np.zeros(0, dtype=float)

        external_force = self.mass[:, None] * self.gravity[None, :]

        return StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "reference_volume": self.volume.copy(),
                "deformation_gradient": self.deformation_gradient.copy(),
                "initial_component_id": np.zeros(self.n, dtype=np.int64),
                "component_id": self.component_id.copy(),
            },
            fields={
                "external_force": external_force,
                "wall_signed_distance": wall_gap.astype(float),
                "floor_signed_distance": floor_gap.astype(float),
                "contact_signed_distance": contact_signed_distance,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "contact_length_scale": self.contact_length_scale,
                "initial_total_mass": self.initial_total_mass,
                "initial_total_volume": self.initial_total_volume,
            },
        )

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.current_time - 1.0e-12:
            raise ValueError(
                f"Requested time {target} is before current time {self.current_time}"
            )

        # Integrate with a fixed number of sub-steps per sample interval
        n_steps = int(round((target - self.current_time) / self.dt_sub))
        for _ in range(n_steps):
            self._substep()
            self.current_time += self.dt_sub

        # Avoid floating-point drift: snap to the requested sample time
        self.current_time = target
        return self._build_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "explicit_bond_fracture",
            "particles": int(self.n),
            "bonds": int(len(self.i_idx)),
            "broken_bonds": int(self.broken.sum()),
            "stuck_particles": int(self.stuck.sum()),
            "substep_dt": float(self.dt_sub),
        }
