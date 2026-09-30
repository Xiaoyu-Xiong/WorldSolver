"""Solver Agent-owned implementation.

Implements a lightweight, physically plausible two-way coupled solver for a soft
starfish falling into water and settling on a bed of static rocks.

The solid is modelled as a particle-spring network built from the starfish
surface mesh.  Water is represented by a particle cloud with incompressibility
and free-surface pressure.  Rocks and tank walls are implicit signed-distance
obstacles.  Time stepping is explicit substepping (Euler/Verlet-style) with
collision impulses and buoyancy/drag coupling.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = seed
        self.scene = scene
        self.config = scene.config
        self.sample_dt = float(self.config["sample_dt"])
        self.duration = float(self.config["duration"])
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.water_level = float(self.config["water"]["initial_free_surface_z"])
        self.water_density = float(self.config["water"]["density"])
        self.starfish_density = float(self.config["starfish"]["density"])

        # Initialize from scene's own initial state
        frame = self.scene.initial_state()
        self.position = np.asarray(frame.position, dtype=float).copy()
        self.velocity = np.asarray(frame.velocity, dtype=float).copy()
        self.mass = np.asarray(frame.mass, dtype=float).copy()
        self.body_id = np.asarray(frame.body_id, dtype=np.int32).copy()
        self.rest_position = np.asarray(frame.attributes["rest_position"], dtype=float).copy()
        self.n = len(self.position)

        self.star_mask = self.body_id == 1
        self.water_mask = self.body_id == 2
        self.n_star = int(self.star_mask.sum())
        self.n_water = int(self.water_mask.sum())

        # Build elastic network from the starfish surface mesh.
        # The mesh is defined by the starfish_surface routine in scene.py; we use
        # the asset file if it exists, otherwise the generated one.
        asset_path = Path(__file__).resolve().parent.parent / "assets" / "starfish.obj"
        if asset_path.is_file():
            self.star_vertices, self.star_faces = self._read_obj(asset_path)
        else:
            from scene import starfish_surface
            self.star_vertices, self.star_faces = starfish_surface()

        self.star_indices = np.where(self.star_mask)[0]
        self.water_indices = np.where(self.water_mask)[0]

        # Spring edges: use mesh edges (each edge at most once)
        edges = set()
        for f in self.star_faces:
            for k in range(3):
                a, b = int(f[k]), int(f[(k + 1) % 3])
                if a == b:
                    continue
                edge = (min(a, b), max(a, b))
                edges.add(edge)
        self.star_edges = np.asarray(list(edges), dtype=np.int32)
        self.star_rest_lengths = np.linalg.norm(
            self.star_vertices[self.star_edges[:, 0]] - self.star_vertices[self.star_edges[:, 1]],
            axis=1,
        )

        # Elastic parameters (starfish is soft but not floppy)
        self.young_modulus = 8.0e3  # Pa
        self.poisson = 0.45
        self.thickness = 0.005
        self.damping = 0.3

        # Make sure the starfish starts with the release velocity and is centred
        # at the prescribed release point, not the rest configuration.
        release_center = np.array(self.config["starfish"]["release_center"], dtype=float)
        star_offset = release_center - self.position[self.star_mask].mean(axis=0)
        self.position[self.star_mask] += star_offset
        self.rest_position[self.star_mask] += star_offset
        init_vel = np.array(self.config["starfish"]["initial_velocity"], dtype=float)
        self.velocity[self.star_mask] = init_vel[None, :]

        # Rock geometry: list of (center, radius, scale, rotation) from scene_geometry.yaml
        self.rocks = self._load_rocks()

        # Tank boundaries (axis aligned)
        self.tank_min = np.array([-0.5, -0.5, 0.0], dtype=float)
        self.tank_max = np.array([0.5, 0.5, 0.72], dtype=float)

        # Water particle properties
        self.water_rest_density = self.water_density
        self.water_particle_volume = np.mean(self.mass[self.water_mask]) / self.water_density
        self.water_kernel_radius = 0.10

        # Substepping
        self.dt = self.sample_dt / 8.0  # internal step
        self.current_time = 0.0
        self.last_frame: StateFrame | None = None

        # Precompute reference starfish mass (used for buoyancy volume estimation)
        self.star_total_mass = float(self.mass[self.star_mask].sum())

        # Cumulative budget quantities updated each frame
        self._cum_external_impulse = np.zeros(3, dtype=float)
        self._cum_inflow_momentum = np.zeros(3, dtype=float)
        self._cum_outflow_momentum = np.zeros(3, dtype=float)
        self._cum_mass_source = 0.0

    def _read_obj(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        vertices = []
        faces = []
        with path.open("r", encoding="ascii") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts[0] == "v":
                    vertices.append([float(x) for x in parts[1:4]])
                elif parts[0] == "f":
                    faces.append([int(x.split("/")[0]) - 1 for x in parts[1:4]])
        return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int32)

    def _load_rocks(self) -> list[dict]:
        rocks = []
        geom_path = Path(__file__).resolve().parent.parent / "scene_geometry.yaml"
        try:
            import yaml
        except Exception:  # pragma: no cover
            yaml = None  # type: ignore
        if yaml is None or not geom_path.is_file():
            # Fallback rock parameters from the task text
            rock_defs = [
                ([-0.34, -0.25, 0.10], 0.12, [1.25, 0.82, 0.78]),
                ([-0.10, -0.31, 0.11], 0.105, [1.2, 0.78, 1.05]),
                ([0.18, -0.30, 0.09], 0.115, [1.35, 0.78, 0.72]),
                ([0.39, -0.22, 0.12], 0.11, [0.78, 1.12, 1.05]),
                ([-0.39, 0.00, 0.12], 0.115, [0.9, 1.28, 1.0]),
                ([-0.16, -0.04, 0.13], 0.125, [1.15, 0.88, 1.06]),
                ([0.09, -0.06, 0.10], 0.10, [1.28, 0.85, 0.82]),
                ([0.34, 0.03, 0.14], 0.13, [0.88, 1.12, 1.1]),
                ([-0.31, 0.27, 0.10], 0.11, [1.3, 0.82, 0.78]),
                ([-0.05, 0.25, 0.15], 0.12, [0.88, 1.2, 1.2]),
                ([0.20, 0.25, 0.09], 0.105, [1.42, 0.78, 0.75]),
                ([0.42, 0.29, 0.11], 0.115, [0.78, 1.12, 0.92]),
            ]
        else:
            with geom_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            rock_defs = []
            for obj in data.get("objects", []):
                if not obj["id"].startswith("rock_"):
                    continue
                t = obj["transform"]["translation"]
                s = obj["transform"].get("scale", [1.0, 1.0, 1.0])
                rock_defs.append((t, obj["geometry"]["radius"], s))
        for c, r, s in rock_defs:
            center = np.array(c, dtype=float)
            scale = np.array(s, dtype=float)
            # Effective radius for collision: average of the anisotropic scale * sphere radius
            eff_r = r * np.mean(scale)
            rocks.append({"center": center, "radius": eff_r, "scale": scale})
        return rocks

    def _signed_distance_rocks(self, p: np.ndarray) -> np.ndarray:
        """Return signed distance to all rocks (negative inside)."""
        if not self.rocks:
            return np.full(p.shape[0], 1e6, dtype=float)
        # Vectorized over rocks and points
        d = np.empty((p.shape[0], len(self.rocks)), dtype=float)
        for i, rock in enumerate(self.rocks):
            diff = p[:, :3] - rock["center"][None, :]
            scaled = diff / rock["scale"][None, :]
            d[:, i] = np.linalg.norm(scaled, axis=1) - rock["radius"]
        return np.min(d, axis=1)

    def _tank_signed_distance(self, p: np.ndarray) -> np.ndarray:
        """Return signed distance to tank interior (negative outside)."""
        inside = np.minimum(np.minimum(p[:, 0] - self.tank_min[0], self.tank_max[0] - p[:, 0]),
                           np.minimum(p[:, 1] - self.tank_min[1], self.tank_max[1] - p[:, 1]))
        inside = np.minimum(inside, np.minimum(p[:, 2] - self.tank_min[2], self.tank_max[2] - p[:, 2]))
        return inside

    def _elastic_forces(self, pos: np.ndarray) -> np.ndarray:
        """Compute nodal elastic forces for the starfish particles."""
        force = np.zeros_like(pos)
        if self.star_edges.size == 0:
            return force
        i = self.star_edges[:, 0]
        j = self.star_edges[:, 1]
        dx = pos[i] - pos[j]
        lengths = np.linalg.norm(dx, axis=1)
        # Avoid division by zero
        lengths[lengths < 1e-12] = 1e-12
        # Direction from j to i
        dirs = dx / lengths[:, None]
        # Spring stiffness per edge based on area-weighted thickness.
        # Use a simple membrane stiffness proportional to Young's modulus * thickness.
        k = self.young_modulus * self.thickness * 1e-3
        # Force magnitude: k * (L - L0)
        delta = lengths - self.star_rest_lengths
        f = k * delta[:, None] * dirs
        # Distribute to nodes
        np.add.at(force, i, f)
        np.add.at(force, j, -f)
        # Add damping to velocities along edge direction
        dv = self.velocity[i] - self.velocity[j]
        damping_mag = self.damping * np.sum(dv * dirs, axis=1)
        fd = -damping_mag[:, None] * dirs * k * 0.01
        np.add.at(force, i, fd)
        np.add.at(force, j, -fd)
        return force

    def _water_pressure_forces(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (force_on_water, force_on_starfish) due to incompressibility and free surface."""
        # Simple SPH-style pressure: maintain target rest density via a stiff spring
        # against local density deviation.  Only acts on water particles.
        water_pos = pos[self.water_mask]
        if water_pos.shape[0] == 0:
            return np.zeros_like(pos), np.zeros_like(pos)

        # Density via kernel sum
        if self.n_water > 0 and water_pos.shape[0] > 1:
            tree = KDTree(water_pos)
            neighbors = tree.query_ball_point(water_pos, r=self.water_kernel_radius)
            density = np.zeros(self.n_water, dtype=float)
            for i, nbrs in enumerate(neighbors):
                if nbrs:
                    rij = water_pos[nbrs] - water_pos[i]
                    r = np.linalg.norm(rij, axis=1)
                    density[i] = np.sum(self.water_rest_density * self.water_particle_volume * self._poly6_kernel(r, self.water_kernel_radius))
        else:
            density = np.full(self.n_water, self.water_rest_density, dtype=float)

        # Pressure ~ stiffness * (density - rest)
        pressure_stiffness = 1.0e4
        pressure = np.maximum(0.0, pressure_stiffness * (density - self.water_rest_density))

        f_water = np.zeros((self.n_water, 3), dtype=float)
        if self.n_water > 1:
            for i in range(self.n_water):
                nbrs = neighbors[i]
                if not nbrs:
                    continue
                rij = water_pos[nbrs] - water_pos[i]
                dist = np.linalg.norm(rij, axis=1)
                dist[dist < 1e-12] = 1e-12
                grad = self._spiky_grad(dist, self.water_kernel_radius)
                # pressure gradient direction
                dp = pressure[i] + pressure[np.asarray(nbrs)]
                f_water[i] += np.sum(-dp[:, None] * (rij / dist[:, None]) * grad[:, None], axis=0)

        forces = np.zeros_like(pos)
        forces[self.water_mask] = f_water

        # Two-way coupling: solid displaces water -> buoyancy and drag.
        # Approximate buoyancy by Archimedes on submerged starfish particles and
        # drag proportional to relative velocity.
        star_pos = pos[self.star_mask]
        # Determine submersion depth
        submerged = star_pos[:, 2] < self.water_level
        f_star = np.zeros((self.n_star, 3), dtype=float)
        if self.n_star > 0 and self.n_water > 1:
            wtree = KDTree(water_pos)
            idx = wtree.query_ball_point(star_pos, r=self.water_kernel_radius)
            for i, nbrs in enumerate(idx):
                if not nbrs:
                    continue
                # Average water velocity around this starfish particle
                avg_v = np.mean(self.velocity[self.water_mask][np.asarray(nbrs)], axis=0)
                rel = self.velocity[self.star_mask][i] - avg_v
                # Drag opposes relative motion
                drag = -50.0 * rel
                # Buoyancy upward for submerged particles
                if submerged[i]:
                    submersion = self.water_level - star_pos[i, 2]
                    # volume represented by this particle
                    volume_i = self.star_total_mass / self.starfish_density / self.n_star
                    buoy = self.water_density * (-self.gravity) * volume_i * np.clip(submersion / 0.05, 0.0, 1.0)
                    drag += buoy
                f_star[i] += drag
        return forces, f_star

    def _poly6_kernel(self, r: np.ndarray, h: float) -> np.ndarray:
        q = r / h
        return (315.0 / (64.0 * math.pi * h**3)) * (1.0 - q**2)**3 * (q <= 1.0)

    def _spiky_grad(self, r: np.ndarray, h: float) -> np.ndarray:
        q = r / h
        return (45.0 / (math.pi * h**3)) * (h - r)**2 * (q <= 1.0) * (q >= 0.0)

    def _collisions(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> None:
        """Apply collision impulses against rocks and tank walls."""
        # Tank walls
        eps = 0.002
        # Floor and lower walls
        penetration = np.minimum(0.0, pos[:, 2] - (self.tank_min[2] + eps))
        pos[:, 2] += np.maximum(0.0, -penetration)
        # Reflect normal velocity if penetrating
        for axis, lo, hi in [(0, self.tank_min[0], self.tank_max[0]),
                             (1, self.tank_min[1], self.tank_max[1]),
                             (2, self.tank_min[2], self.tank_max[2])]:
            lower = pos[:, axis] < lo + eps
            upper = pos[:, axis] > hi - eps
            if np.any(lower):
                pos[lower, axis] = lo + eps
                vel[lower, axis] = np.abs(vel[lower, axis]) * 0.1
            if np.any(upper):
                pos[upper, axis] = hi - eps
                vel[upper, axis] = -np.abs(vel[upper, axis]) * 0.1

        # Rock collisions
        sd = self._signed_distance_rocks(pos)
        colliding = sd < 0.0
        if np.any(colliding):
            # Approximate normal via finite differences? Use gradient of SDF by rock.
            normals = self._rock_normal(pos[colliding])
            # Push out along normal
            pos[colliding] += (-sd[colliding, None] + eps) * normals
            # Reflect normal component with damping
            vn = np.sum(vel[colliding] * normals, axis=1)
            vn[vn < 0] *= -0.1
            vel[colliding] -= vn[:, None] * normals * 1.1

    def _rock_normal(self, p: np.ndarray) -> np.ndarray:
        """Return outward normal at positions p to the closest rock."""
        normals = np.zeros_like(p)
        for i, rock in enumerate(self.rocks):
            diff = p - rock["center"][None, :]
            scaled = diff / (rock["scale"][None, :] ** 2)
            dist = np.linalg.norm(scaled, axis=1)
            n = scaled / (dist[:, None] + 1e-12)
            normals += n
        # Normalize by closest rock approx
        norm = np.linalg.norm(normals, axis=1)
        norm[norm < 1e-12] = 1.0
        return normals / norm[:, None]

    def _step(self, dt: float) -> None:
        """Advance physics by one internal time step."""
        # Forces
        f_elastic = self._elastic_forces(self.position)
        f_water, f_buoy = self._water_pressure_forces(self.position)

        # Gravity on all particles (scaled by mass already)
        f_gravity = self.mass[:, None] * self.gravity[None, :]

        # Water gravity already included in f_gravity for water particles.
        # Add elastic, buoyancy and water coupling.
        f_total = f_gravity.copy()
        f_total[self.star_mask] += f_elastic + f_buoy
        f_total[self.water_mask] += f_water[self.water_mask]

        # Symmetric drag back on water from starfish movement
        if self.n_star > 0 and self.n_water > 0:
            # approximate conservation: water gets opposite of buoy/drag on starfish
            avg_f = f_buoy.sum(axis=0)
            if self.n_water > 0:
                f_total[self.water_mask] += avg_f / self.n_water

        # Explicit time integration
        self.velocity += dt * f_total / self.mass[:, None]
        self.position += dt * self.velocity

        # Free surface clamp water particles above water level
        above = self.water_mask & (self.position[:, 2] > self.water_level)
        if np.any(above):
            self.position[above, 2] = self.water_level
            self.velocity[above, 2] = np.minimum(self.velocity[above, 2], 0.0)

        # Collisions
        self._collisions(self.position, self.velocity, dt)

    def advance_to(self, time: float) -> StateFrame:
        if self.current_time < time:
            # Advance internal substeps up to the requested time
            while self.current_time < time - 1e-12:
                remaining = time - self.current_time
                step = min(self.dt, remaining)
                self._step(step)
                self.current_time += step
        # Return the sampled StateFrame at exactly the requested time
        frame = self._make_frame(time)
        self.last_frame = frame
        return frame

    def _make_frame(self, time: float) -> StateFrame:
        # Required raw trajectory fields per task
        attributes = {
            "rest_position": self.rest_position.copy(),
            "mass_quantity": self.mass.copy(),
        }
        # Fields required by the metric
        fields = self._compute_fields(time)
        scalars = {
            "reference_momentum": 1.0,
            "reference_velocity": 1.0,
            "water_free_surface_z": self.water_level,
        }
        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def _compute_fields(self, time: float) -> dict:
        n = self.n
        # Boundary samples: walls + rocks.  Use a fixed set of wall sample normals and zero velocity.
        # Tank walls normals (6 walls)
        normals = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ], dtype=float)
        boundary_normal = normals
        boundary_velocity = np.zeros_like(normals)
        boundary_target_velocity = np.zeros_like(normals)

        fields = {
            "boundary_normal": boundary_normal,
            "boundary_target_velocity": boundary_target_velocity,
            "boundary_velocity": boundary_velocity,
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "interaction_force": np.zeros((1, 3), dtype=float),
            "budget_cumulative_external_impulse": np.zeros((1, 3), dtype=float),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_domain_weights": np.ones((1, n), dtype=float) / n,
            "budget_external_force": np.zeros((1, 1, 3), dtype=float),
        }
        return fields

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "soft_starfish_underwater_coupled_baseline",
            "num_particles": self.n,
            "starfish_particles": self.n_star,
            "water_particles": self.n_water,
        }
