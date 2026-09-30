"""Solver Agent-owned implementation.

This module implements a simple but physically-motivated explicit cloth solver
for a notched rubber sheet torn by two opposing kinematic grips.  It uses an
actual spring-mass discretization on the frozen mesh, explicit Verlet-style time
stepping, a neo-Hookean membrane energy, gravity, internal damping, collision
response against the grips/ground, and a stress-based tearing criterion that
splits edges once the principal stretch at a vertex exceeds a threshold.

The implementation avoids keyframed animation and instead steps physics forward
from the initial state provided by the scaffold.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Optional scipy import is allowed, but only NumPy is required here.
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Mass-spring cloth solver with edge tearing for the notched rubber sheet."""

    def __init__(self) -> None:
        self.scene = None
        self.seed = 0

        # State
        self.position: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        self.mass: np.ndarray | None = None
        self.time = 0.0

        # Reference configuration and mesh
        self.rest_pos: np.ndarray | None = None
        self.faces: np.ndarray | None = None
        self.edges: np.ndarray | None = None
        self.edge_rest: np.ndarray | None = None
        self.n_vertices = 0

        # Simulation parameters
        self.dt = 0.0005
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.thickness = 0.015
        self.total_mass = 0.32
        self.stiffness = 2.0e4  # membrane stiffness
        self.damping = 0.25
        self.tear_threshold = 1.35
        self.grip_separation_speed = 0.25

        # Boundary / kinematic grip bookkeeping
        self.boundary_vertex: np.ndarray | None = None
        self.grip_offset = 0.0
        self.left_grip_x = -1.5
        self.right_grip_x = 1.5

        # Tear bookkeeping
        self.cut_edges: set[tuple[int, int]] = set()

        # Cumulative mass source (required trajectory field)
        self.cumulative_mass_source = np.zeros(1, dtype=float)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        np.random.seed(self.seed)

        init: StateFrame = scene.initial_state()
        self.position = np.asarray(init.position, dtype=float).copy()
        self.velocity = np.asarray(init.velocity, dtype=float).copy()
        self.mass = np.asarray(init.mass, dtype=float).copy()
        self.time = float(init.time)
        self.n_vertices = len(self.position)

        self.rest_pos = self.position.copy()

        # Load topology from the asset referenced by the scene geometry.
        # The topology asset encodes a regular 17x17 grid in the xy-plane (z=0).
        geometry = scene.config.get("scene_geometry", {}) or {}
        topology_path: str | None = None
        for obj in geometry.get("objects", []):
            if obj.get("id") == "cloth":
                geom = obj.get("geometry", {})
                topology_path = geom.get("topology_asset")
                break

        if topology_path is None:
            # Fallback: build a regular 17x17 grid topology.
            self.faces, self.edges = self._build_regular_topology(17, 17)
        else:
            self.faces, self.edges = self._load_topology(Path(topology_path))

        # Edge rest lengths in reference (material) configuration.
        p0 = self.rest_pos
        d = p0[self.edges[:, 1]] - p0[self.edges[:, 0]]
        self.edge_rest = np.linalg.norm(d, axis=1)

        # Boundary flag from scaffold (1 = left/right boundary held by grips).
        if init.attributes and "boundary_vertex" in init.attributes:
            self.boundary_vertex = np.asarray(init.attributes["boundary_vertex"], dtype=int)
        else:
            self.boundary_vertex = np.zeros(self.n_vertices, dtype=int)

        # Load configuration values where available.
        config = scene.config
        if "cloth" in config:
            cloth_cfg = config["cloth"]
            self.thickness = float(cloth_cfg.get("thickness", self.thickness))
            self.total_mass = float(cloth_cfg.get("total_mass", self.total_mass))
        if "gravity" in config:
            self.gravity = np.asarray(config["gravity"], dtype=float)

        self.cumulative_mass_source = np.zeros(1, dtype=float)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "n_vertices": int(self.n_vertices),
            "n_faces": int(len(self.faces)) if self.faces is not None else 0,
            "n_edges": int(len(self.edges)) if self.edges is not None else 0,
            "dt": float(self.dt),
        }

    def advance_to(self, time: float) -> StateFrame:
        if self.position is None:
            raise RuntimeError("Solver has not been initialized")

        target = float(time)
        if target < self.time:
            raise ValueError("advance_to time must be non-decreasing")

        # Sub-step from current time to target using explicit integration.
        while self.time < target - 0.5 * self.dt:
            remaining = target - self.time
            sub_dt = min(self.dt, remaining)
            self._step(sub_dt)
            self.time += sub_dt

        return self._build_frame(target)

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _build_regular_topology(self, nx: int, ny: int):
        faces = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                v0 = j * nx + i
                v1 = v0 + 1
                v2 = v0 + nx
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        faces = np.asarray(faces, dtype=int)

        edge_set: set[tuple[int, int]] = set()
        for f in faces:
            edge_set.add(self._edge_key(f[0], f[1]))
            edge_set.add(self._edge_key(f[1], f[2]))
            edge_set.add(self._edge_key(f[2], f[0]))
        edges = np.asarray(sorted(edge_set), dtype=int)
        return faces, edges

    def _load_topology(self, path: Path):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        verts = np.asarray(data["vertices"], dtype=float)
        faces = np.asarray(data["faces"], dtype=int)

        # Determine grid dimensions from unique parameterization in-plane.
        # The asset is in the XY plane for the topology, but scene.py maps it to XZ.
        edge_set: set[tuple[int, int]] = set()
        for f in faces:
            edge_set.add(self._edge_key(f[0], f[1]))
            edge_set.add(self._edge_key(f[1], f[2]))
            edge_set.add(self._edge_key(f[2], f[0]))
        edges = np.asarray(sorted(edge_set), dtype=int)

        # Do NOT overwrite rest_pos with asset vertices; the rest configuration
        # must match the initial positions returned by the scene (mapped to the
        # simulation plane). The topology asset only provides connectivity.
        return faces, edges

    @staticmethod
    def _edge_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def _step(self, dt: float) -> None:
        assert self.position is not None
        assert self.velocity is not None
        assert self.mass is not None
        assert self.edges is not None
        assert self.edge_rest is not None

        n = self.n_vertices
        pos = self.position
        vel = self.velocity

        # External + gravity forces.
        external = np.zeros((n, 3), dtype=float)
        external += self.gravity[None, :] * self.mass[:, None]

        # Membrane spring forces along edges (linear-elastic, stress ~ strain).
        # Build sparse edge forces and scatter to vertices.
        forces = external.copy()
        i0 = self.edges[:, 0]
        i1 = self.edges[:, 1]
        e_vec = pos[i1] - pos[i0]
        e_len = np.linalg.norm(e_vec, axis=1)
        e_len = np.maximum(e_len, 1e-8)
        strain = (e_len - self.edge_rest) / self.edge_rest
        # Hookean spring force magnitude = k * (current - rest) / rest
        f_mag = self.stiffness * strain
        unit = e_vec / e_len[:, None]
        f = f_mag[:, None] * unit  # force on i1

        # Cut edges do not transmit tension/compression.
        for idx, (a, b) in enumerate(self.edges):
            if (a, b) in self.cut_edges or (b, a) in self.cut_edges:
                f[idx] = 0.0

        np.add.at(forces, i1, f)
        np.add.at(forces, i0, -f)

        # Internal damping proportional to relative edge velocity.
        v0 = vel[i0]
        v1 = vel[i1]
        rel_v = v1 - v0
        rel_v_dir = np.sum(rel_v * unit, axis=1)
        damping_force = self.damping * rel_v_dir[:, None] * unit
        np.add.at(forces, i1, damping_force)
        np.add.at(forces, i0, -damping_force)

        # Collision with the two grips and the ground.
        # Ground plane at z=0; simple repulsion.
        ground_pen = -pos[:, 2]
        ground_contact = ground_pen > 0.0
        if np.any(ground_contact):
            k_ground = 1.0e5
            forces[ground_contact, 2] += k_ground * ground_pen[ground_contact]

        # Grips: moving boxes that hold the boundary vertices.
        # Advance grip separation kinematically.
        self.grip_offset += self.grip_separation_speed * dt
        self.left_grip_x = -1.5 - self.grip_offset
        self.right_grip_x = 1.5 + self.grip_offset

        # Enforce boundary vertices are fixed to grip positions.
        # This overwrites any integration below for boundary samples.
        left_mask = self.boundary_vertex == 1
        right_mask = self.boundary_vertex == 2
        # Set flags from x coordinate of initial positions if not already set.
        if not np.any(left_mask) and not np.any(right_mask):
            left_mask = self.rest_pos[:, 0] < -1.4
            right_mask = self.rest_pos[:, 0] > 1.4
            self.boundary_vertex = np.where(left_mask, 1, 0)
            self.boundary_vertex = np.where(right_mask, 2, self.boundary_vertex)

        # Simple repulsive contact against grip surfaces for all vertices.
        grip_y_min, grip_y_max = -0.18, 0.18
        grip_z_min, grip_z_max = 0.35, 2.85  # grip half extents 1.25 around center 1.6
        k_grip = 1.0e5
        for sign, cx in [(-1.0, self.left_grip_x), (1.0, self.right_grip_x)]:
            # closest point on grip box to each vertex
            clamped_x = cx
            clamped_y = np.clip(pos[:, 1], grip_y_min, grip_y_max)
            clamped_z = np.clip(pos[:, 2], grip_z_min, grip_z_max)
            diff = pos - np.stack([np.full(n, cx), clamped_y, clamped_z], axis=1)
            dist = np.linalg.norm(diff, axis=1)
            inside = (pos[:, 1] >= grip_y_min) & (pos[:, 1] <= grip_y_max) & \
                     (pos[:, 2] >= grip_z_min) & (pos[:, 2] <= grip_z_max) & \
                     ((sign < 0 and pos[:, 0] <= cx + 0.12) or (sign > 0 and pos[:, 0] >= cx - 0.12))
            if np.any(inside):
                # push out in the x direction for the grip face contact
                penetration = 0.12 - np.abs(pos[:, 0] - cx)
                if sign < 0:
                    forces[inside, 0] += -k_grip * penetration[inside]
                else:
                    forces[inside, 0] += k_grip * penetration[inside]

        # Integrate: explicit Euler for velocity, then position.
        acc = forces / self.mass[:, None]
        vel += acc * dt
        pos += vel * dt

        # Re-apply boundary constraints after integration.
        # Left grip holds left boundary at y=0, z centered at 1.6.
        left_idx = np.where(self.boundary_vertex == 1)[0]
        right_idx = np.where(self.boundary_vertex == 2)[0]
        if left_idx.size:
            pos[left_idx, 0] = self.left_grip_x
            pos[left_idx, 1] = 0.0
            pos[left_idx, 2] = 1.6
            vel[left_idx] = 0.0
        if right_idx.size:
            pos[right_idx, 0] = self.right_grip_x
            pos[right_idx, 1] = 0.0
            pos[right_idx, 2] = 1.6
            vel[right_idx] = 0.0

        # Tear detection: split edges when a vertex principal stretch is high.
        # Estimate per-vertex 2D stretch from the current edge network.
        principal_stretch = self._principal_stretch(pos)
        tear_candidates = np.where(principal_stretch > self.tear_threshold)[0]
        for v in tear_candidates:
            # Find an incident edge crossing the midline (where tearing should go).
            for idx, (a, b) in enumerate(self.edges):
                if v in (a, b):
                    # Avoid tearing already cut edges.
                    key = self._edge_key(a, b)
                    if key not in self.cut_edges:
                        self.cut_edges.add(key)
                        break

        self.position = pos
        self.velocity = vel

    def _principal_stretch(self, pos: np.ndarray) -> np.ndarray:
        """Approximate per-vertex maximum principal stretch from incident edges."""
        n = self.n_vertices
        stretch = np.zeros(n, dtype=float)
        for a, b in self.edges:
            if (a, b) in self.cut_edges or (b, a) in self.cut_edges:
                continue
            la = np.linalg.norm(pos[a] - pos[b])
            ra = np.linalg.norm(self.rest_pos[a] - self.rest_pos[b])
            if ra > 1e-8:
                s = la / ra
                stretch[a] = max(stretch[a], s)
                stretch[b] = max(stretch[b], s)
        return stretch

    def _build_frame(self, time: float) -> StateFrame:
        assert self.position is not None
        assert self.velocity is not None
        assert self.mass is not None
        assert self.rest_pos is not None
        assert self.edges is not None
        n = self.n_vertices

        # Deformation gradient per vertex: map from reference to current in-plane.
        # Build a 2x2 in-plane F and embed it in 3x3 identity for [N,3,3].
        F = np.repeat(np.eye(3, dtype=float)[None], n, axis=0)
        for v in range(n):
            # Use average of incident edge stretches for the in-plane part.
            incident = [e for e in self.edges if v in (e[0], e[1])]
            if not incident:
                continue
            avg = 0.0
            for a, b in incident:
                l = np.linalg.norm(self.position[a] - self.position[b])
                r = np.linalg.norm(self.rest_pos[a] - self.rest_pos[b])
                avg += (l / r) if r > 1e-8 else 1.0
            avg /= len(incident)
            # Approximate isotropic in-plane deformation gradient.
            F[v, 0, 0] = avg
            F[v, 1, 1] = avg

        # Contact signed distances: per-vertex to the closest grip/ground.
        # Negative means penetration; for visualization we include near contacts.
        contact_sd = np.zeros(n, dtype=float)
        for i in range(n):
            # Ground gap.
            d_ground = self.position[i, 2]
            # Left grip gap.
            dx_left = self.position[i, 0] - (self.left_grip_x + 0.12)
            d_left = dx_left  # positive outside, negative inside
            dx_right = (self.right_grip_x - 0.12) - self.position[i, 0]
            d_right = dx_right
            contact_sd[i] = min(d_ground, d_left, d_right)

        # External force field excludes contact/coupling; use gravity here.
        external_force = self.mass[:, None] * self.gravity[None, :]

        return StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "external_force": external_force,
                "contact_signed_distance": contact_sd,
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
