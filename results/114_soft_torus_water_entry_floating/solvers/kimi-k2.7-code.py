"""Solver implementation for soft torus water entry and floating.

A particle-based two-way coupled simulation:
- The soft torus is modelled as a deformable elastic surface (edge springs +
  volume-preserving pressure).
- The water is modelled as a weakly-compressible particle fluid using
  pairwise repulsion and viscous damping.
- Solid and fluid particles are coupled through short-range repulsive contact.
- Static tub walls provide penalty-based boundary conditions.

Time integration uses a symplectic (velocity-stormer) Euler sub-stepping scheme.
"""
from __future__ import annotations

import importlib
import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = seed
        self.scene = scene

        # Initial state from the scene contract
        frame0 = scene.initial_state()
        self.position = frame0.position.copy()
        self.velocity = frame0.velocity.copy()
        self.mass = frame0.mass.copy()
        self.body_id = frame0.body_id.copy()
        self.rest_position = frame0.attributes["rest_position"].copy()

        self.n = len(self.position)
        self.solid_mask = self.body_id == 1
        self.fluid_mask = self.body_id == 2
        self.solid_idx = np.where(self.solid_mask)[0]
        self.fluid_idx = np.where(self.fluid_mask)[0]

        # Torus topology from the scaffold
        scene_mod = importlib.import_module("scene")
        torus_verts, self.torus_faces = scene_mod.torus_surface()
        self.num_solid = len(torus_verts)

        # Edge network and rest shape quantities for the torus
        self.edges = self._build_edges(self.torus_faces)
        self.rest_edge_lengths = self._edge_lengths(
            self.rest_position[self.solid_idx], self.edges
        )
        self.rest_triangle_areas = self._triangle_areas(
            self.rest_position[self.solid_idx], self.torus_faces
        )
        self.rest_volume = self._enclosed_volume(
            self.rest_position[self.solid_idx], self.torus_faces
        )

        # Physical parameters
        self.gravity = np.array(
            scene.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        )
        self.k_stretch = 5e3
        self.k_volume = 2e4
        self.k_damp = 2.0
        self.k_fluid = 2e3
        self.k_coupling = 5e3
        self.h = 0.08
        self.nu_fluid = 0.5
        self.wall_k = 5e4
        self.wall_c = 100.0

        # Time-stepping
        self.t = 0.0
        self.sample_dt = float(scene.config["sample_dt"])
        self.substeps = 20
        self.dt = self.sample_dt / self.substeps

        # Tub interior bounds (top is open)
        self.bounds_min = np.array([-0.50, -0.50, 0.0], dtype=float)
        self.bounds_max = np.array([0.50, 0.50, 2.0], dtype=float)

        # Budget bookkeeping
        self.G = 2  # torus, water
        self.K = 3  # gravity, wall, coupling
        self.cum_external_impulse = np.zeros((self.G, 3), dtype=float)
        self.cum_inflow_mom = np.zeros((self.G, 3), dtype=float)
        self.cum_outflow_mom = np.zeros((self.G, 3), dtype=float)
        self.cum_mass_source = np.zeros(2, dtype=float)
        self.interaction_force = np.zeros((1, 3), dtype=float)
        self.budget_external_force = np.zeros((self.G, self.K, 3), dtype=float)

        # Reference scales
        total_mass = float(np.sum(self.mass))
        self.ref_momentum = total_mass * 1.0
        self.ref_velocity = 1.0

        # Static boundary samples: x=±0.5, y=±0.5, z=0
        self.boundary_normals = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.boundary_target_velocity = np.zeros((5, 3), dtype=float)
        self.boundary_velocity = np.zeros((5, 3), dtype=float)

    # ------------------------------------------------------------------ helpers
    def _build_edges(self, faces):
        edges = set()
        for f in faces:
            for i in range(3):
                a, b = f[i], f[(i + 1) % 3]
                if a > b:
                    a, b = b, a
                edges.add((a, b))
        return np.array(list(edges), dtype=int)

    def _edge_lengths(self, pos, edges):
        diff = pos[edges[:, 0]] - pos[edges[:, 1]]
        return np.linalg.norm(diff, axis=1)

    def _triangle_areas(self, pos, faces):
        v0 = pos[faces[:, 1]] - pos[faces[:, 0]]
        v1 = pos[faces[:, 2]] - pos[faces[:, 0]]
        cross = np.cross(v0, v1)
        return 0.5 * np.linalg.norm(cross, axis=1)

    def _enclosed_volume(self, pos, faces):
        a = pos[faces[:, 0]]
        b = pos[faces[:, 1]]
        c = pos[faces[:, 2]]
        cross = np.cross(b - a, c - a)
        area = 0.5 * np.linalg.norm(cross, axis=1, keepdims=True)
        unit_n = cross / (2.0 * area + 1e-12)
        centroid = (a + b + c) / 3.0
        return float(
            np.sum(np.einsum("ij,ij->i", centroid, unit_n) * area[:, 0]) / 3.0
        )

    # ---------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        target = max(0.0, float(time))
        while self.t < target - 1e-12:
            self._step()
        return self._build_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": self.dt,
            "substeps_per_frame": self.substeps,
            "num_particles": self.n,
            "num_solid": int(np.sum(self.solid_mask)),
            "num_fluid": int(np.sum(self.fluid_mask)),
        }

    # ---------------------------------------------------------------- time step
    def _step(self):
        pos = self.position
        vel = self.velocity
        mass = self.mass

        # Accumulate forces
        force = np.zeros_like(pos)

        # Gravity
        grav_force = mass[:, None] * self.gravity[None, :]
        force += grav_force

        # Elastic solid forces on the torus
        f_elastic = self._elastic_forces(pos)
        force[self.solid_idx] += f_elastic

        # Fluid-fluid and fluid-solid coupling
        f_coupling = self._fluid_coupling_forces(pos, vel)
        force += f_coupling

        # Tub wall penalty
        f_wall = self._wall_forces(pos, vel)
        force += f_wall

        # Symplectic Euler update
        acc = force / mass[:, None]
        vel_new = vel + self.dt * acc
        pos_new = pos + self.dt * vel_new

        self.position = pos_new
        self.velocity = vel_new
        self.t += self.dt

        # Budget updates
        self._update_budgets(grav_force, f_wall, f_coupling)

    def _elastic_forces(self, pos):
        solid_pos = pos[self.solid_idx]
        force = np.zeros_like(solid_pos)

        # Edge springs (stretch resistance)
        a = solid_pos[self.edges[:, 0]]
        b = solid_pos[self.edges[:, 1]]
        diff = a - b
        dist = np.linalg.norm(diff, axis=1)
        stretch = dist - self.rest_edge_lengths
        f_mag = self.k_stretch * stretch
        # Force on vertex a is toward b when stretched
        f_vec = -f_mag[:, None] * diff / (dist[:, None] + 1e-12)
        np.add.at(force, self.edges[:, 0], f_vec)
        np.add.at(force, self.edges[:, 1], -f_vec)

        # Volume-preserving pressure
        a = solid_pos[self.torus_faces[:, 0]]
        b = solid_pos[self.torus_faces[:, 1]]
        c = solid_pos[self.torus_faces[:, 2]]
        n = np.cross(b - a, c - a)
        area = 0.5 * np.linalg.norm(n, axis=1, keepdims=True)
        unit_n = n / (2.0 * area + 1e-12)
        centroid = (a + b + c) / 3.0
        V = float(
            np.sum(np.einsum("ij,ij->i", centroid, unit_n) * area[:, 0]) / 3.0
        )
        p = self.k_volume * (self.rest_volume - V) / (self.rest_volume + 1e-12)
        p = float(np.clip(p, -1e6, 1e6))
        f_pressure = (p * area / 3.0) * unit_n
        np.add.at(force, self.torus_faces[:, 0], f_pressure)
        np.add.at(force, self.torus_faces[:, 1], f_pressure)
        np.add.at(force, self.torus_faces[:, 2], f_pressure)

        # Velocity damping to remove spurious elastic oscillations
        force -= self.k_damp * self.velocity[self.solid_idx]

        return force

    def _fluid_coupling_forces(self, pos, vel):
        force = np.zeros_like(pos)
        tree = cKDTree(pos)
        pairs = tree.sparse_distance_matrix(
            tree, max_distance=self.h, output_type="coo"
        )
        i = pairs.row
        j = pairs.col
        d = pairs.data
        valid = (i < j) & (d > 1e-12)
        i = i[valid]
        j = j[valid]
        d = d[valid]

        if i.size == 0:
            return force

        dx = pos[j] - pos[i]
        r = d[:, None]
        unit = dx / (r + 1e-12)

        fi = self.fluid_mask[i]
        fj = self.fluid_mask[j]
        si = self.solid_mask[i]
        sj = self.solid_mask[j]
        ff = fi & fj
        sf = (si & fj) | (fi & sj)

        # Repulsive contact
        mag = np.zeros(len(i), dtype=float)
        mag[ff] = self.k_fluid * (self.h - d[ff])
        mag[sf] += self.k_coupling * (self.h - d[sf])

        f_ij = mag[:, None] * unit
        np.add.at(force, i, f_ij)
        np.add.at(force, j, -f_ij)

        # Viscous damping for fluid pairs
        if self.nu_fluid > 0.0 and np.any(ff):
            dv = vel[j] - vel[i]
            mag_visc = np.where(ff, self.nu_fluid * (self.h - d), 0.0)
            f_v = mag_visc[:, None] * dv
            np.add.at(force, i, f_v)
            np.add.at(force, j, -f_v)

        return force

    def _wall_forces(self, pos, vel):
        force = np.zeros_like(pos)

        # Side walls in x and y
        for dim, min_b, max_b in (
            (0, self.bounds_min[0], self.bounds_max[0]),
            (1, self.bounds_min[1], self.bounds_max[1]),
        ):
            mask = pos[:, dim] < min_b
            if np.any(mask):
                depth = min_b - pos[mask, dim]
                force[mask, dim] += self.wall_k * depth
                force[mask, dim] -= self.wall_c * vel[mask, dim]

            mask = pos[:, dim] > max_b
            if np.any(mask):
                depth = pos[mask, dim] - max_b
                force[mask, dim] -= self.wall_k * depth
                force[mask, dim] -= self.wall_c * vel[mask, dim]

        # Floor only (free surface at top)
        mask = pos[:, 2] < 0.0
        if np.any(mask):
            depth = -pos[mask, 2]
            force[mask, 2] += self.wall_k * depth
            force[mask, 2] -= self.wall_c * vel[mask, 2]

        return force

    def _update_budgets(self, grav_force, wall_force, coupling_force):
        domain_forces = np.zeros((self.G, self.K, 3), dtype=float)

        # Torus domain
        domain_forces[0, 0] = np.sum(grav_force[self.solid_idx], axis=0)
        domain_forces[0, 1] = np.sum(wall_force[self.solid_idx], axis=0)
        domain_forces[0, 2] = np.sum(coupling_force[self.solid_idx], axis=0)

        # Water domain
        domain_forces[1, 0] = np.sum(grav_force[self.fluid_idx], axis=0)
        domain_forces[1, 1] = np.sum(wall_force[self.fluid_idx], axis=0)
        domain_forces[1, 2] = np.sum(coupling_force[self.fluid_idx], axis=0)

        self.budget_external_force = domain_forces
        self.cum_external_impulse += domain_forces.sum(axis=1) * self.dt
        self.interaction_force[0] = np.sum(
            coupling_force[self.solid_idx], axis=0
        )

    # ------------------------------------------------------------------ frame
    def _build_frame(self):
        domain_weights = np.zeros((self.G, self.n), dtype=float)
        domain_weights[0, self.solid_idx] = 1.0
        domain_weights[1, self.fluid_idx] = 1.0

        mass_quantity = np.zeros((self.n, 2), dtype=float)
        mass_quantity[self.solid_idx, 0] = self.mass[self.solid_idx]
        mass_quantity[self.fluid_idx, 1] = self.mass[self.fluid_idx]

        frame = StateFrame(
            time=self.t,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "rest_position": self.rest_position.copy(),
                "mass_quantity": mass_quantity,
            },
            fields={
                "boundary_normal": self.boundary_normals.copy(),
                "boundary_target_velocity": self.boundary_target_velocity.copy(),
                "boundary_velocity": self.boundary_velocity.copy(),
                "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_mom.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_mom.copy(),
                "budget_domain_weights": domain_weights,
                "budget_external_force": self.budget_external_force.copy(),
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "interaction_force": self.interaction_force.copy(),
            },
            scalars={
                "reference_momentum": self.ref_momentum,
                "reference_velocity": self.ref_velocity,
                "water_free_surface_z": 0.40,
            },
        )
        frame.validate()
        return frame
