"""Cloth sleeve on articulated cylinders physics solver.

Implements a simple mass-spring cloth with explicit time integration and
capsule collision response. The solver advances physics at a small sub-step and
samples StateFrame objects at the times requested by the scene.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = int(seed)

        # Read configuration and geometry
        self._config = scene.config
        cloth_cfg = self._config["cloth"]
        self._gravity = np.asarray(self._config["gravity"], dtype=float)
        self._thickness = float(cloth_cfg["thickness"])
        self._total_mass = float(cloth_cfg["total_mass"])
        self._sample_dt = float(self._config["sample_dt"])
        self._duration = float(self._config["duration"])
        self._seed = seed

        # Mesh topology from the provided asset
        topo_path = Path(self._config.get("scene_geometry", {})).parent / "assets" / "cloth_topology.json"
        if not topo_path.exists():
            topo_path = Path("assets/cloth_topology.json")
        import json
        with open(topo_path) as fh:
            topo = json.load(fh)
        self._rest_pos = np.asarray(topo["vertices"], dtype=float)  # initial parametric positions
        # Build the sleeve from the flat grid, mapping (u in [-1.5,1.5], v in [-1.25,1.25])
        # onto a tube that encloses the two capsules.
        self._build_tubular_sleeve()
        self._faces = np.asarray(topo["faces"], dtype=int)

        # Make the rest mesh into a tubular sleeve around the cylinders
        # The initial scene already provides positions; we use it as initial state.
        init_frame = scene.initial_state()
        self._pos = np.array(init_frame.position, dtype=float)
        self._vel = np.array(init_frame.velocity, dtype=float)
        self._mass = np.array(init_frame.mass, dtype=float)
        self._body_id = np.array(init_frame.body_id, dtype=int)

        # Build edge list and rest lengths
        edges = set()
        for f in self._faces:
            edges.add((min(f[0], f[1]), max(f[0], f[1])))
            edges.add((min(f[1], f[2]), max(f[1], f[2])))
            edges.add((min(f[2], f[0]), max(f[2], f[0])))
        self._edges = np.asarray(sorted(edges), dtype=int)
        self._edge_rest = np.linalg.norm(
            self._rest_pos[self._edges[:, 1]] - self._rest_pos[self._edges[:, 0]], axis=1
        )

        # Per-vertex area (lumped mass) and material parameters
        self._area = self._compute_vertex_area()
        self._young = 2.0e4  # Pa
        self._poisson = 0.3
        self._damping = 0.5  # Rayleigh-like velocity damping
        self._friction = 0.35
        self._substeps = 20
        self._dt = self._sample_dt / self._substeps

        # Kinematic cylinders
        self._capsules = [
            {"radius": 0.34, "height": 1.3, "base": np.array([-0.55, 0.0, 1.0]), "axis_dir": np.array([1.0, 0.0, 0.0])},
            {"radius": 0.31, "height": 1.3, "base": np.array([0.55, 0.0, 1.0]), "axis_dir": np.array([1.0, 0.0, 0.0])},
        ]

        # Deformation gradient per element (per triangle)
        self._F = np.repeat(np.eye(2, dtype=float)[None], len(self._faces), axis=0)

        # External force and contact arrays
        self._ext_force = np.zeros_like(self._pos)
        self._contact_signed = np.zeros(len(self._pos), dtype=float)

        self._current_time = 0.0
        self._frame_index = 0
        self._rng = np.random.default_rng(seed)

    def _build_tubular_sleeve(self) -> None:
        # Map each rest vertex (x,y,0) to a cylinder around the global y-axis.
        # We use the v coordinate as the axial (y) direction and wrap the
        # u coordinate around the circumference.
        p = self._rest_pos.copy()
        u = p[:, 0]
        v = p[:, 1]
        # Circumferential angle proportional to u, radius chosen so the sleeve
        # loosely fits around the capsules (radii ~0.34 / 0.31, plus clearance).
        radius = 0.46
        theta = (u / 1.5) * np.pi  # map u in [-1.5,1.5] -> theta in [-pi,pi]
        x = radius * np.sin(theta)
        z = radius * np.cos(theta) + 1.0
        y = v
        self._rest_pos = np.column_stack([x, y, z])

    def _compute_vertex_area(self) -> np.ndarray:
        area = np.zeros(self._pos.shape[0], dtype=float)
        p = self._rest_pos
        for f in self._faces:
            a, b, c = p[f]
            tri_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
            area[f] += tri_area / 3.0
        return area

    def diagnostics(self) -> dict[str, object]:
        return {"substeps": self._substeps, "dt": self._dt}

    def advance_to(self, time: float) -> StateFrame:
        while self._current_time < time - 1e-12:
            self._step()
        self._current_time = time
        self._frame_index += 1
        return self._build_frame(time)

    def _build_frame(self, time: float) -> StateFrame:
        # Update per-element deformation gradient from current state
        self._F = self._compute_deformation_gradient(self._pos)
        # Signed contact distance per vertex
        self._contact_signed = self._compute_signed_distance(self._pos)
        # External force per vertex (gravity only here)
        self._ext_force = np.zeros_like(self._pos)
        self._ext_force += self._mass[:, None] * self._gravity

        return StateFrame(
            time=time,
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={
                "deformation_gradient": self._F.copy(),
                "mass_quantity": self._mass.copy(),
            },
            fields={
                "contact_signed_distance": self._contact_signed.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self._ext_force.copy(),
            },
            scalars={
                "contact_length_scale": float(self._thickness),
            },
        )

    def _step(self) -> None:
        dt = self._dt
        # Gravity
        force = self._mass[:, None] * self._gravity
        # Spring forces from edges
        force += self._spring_forces(self._pos)
        # Damping
        force -= self._damping * self._vel

        # Explicit integration
        self._vel += (force / self._mass[:, None]) * dt
        self._pos += self._vel * dt

        # Collision and contact response
        self._resolve_collisions()

        self._current_time += dt

    def _spring_forces(self, pos: np.ndarray) -> np.ndarray:
        i, j = self._edges[:, 0], self._edges[:, 1]
        diff = pos[j] - pos[i]
        dist = np.linalg.norm(diff, axis=1)
        # Avoid divide by zero
        mask = dist > 1e-12
        strain = np.zeros_like(dist)
        strain[mask] = (dist[mask] - self._edge_rest[mask]) / self._edge_rest[mask]
        # Spring stiffness based on Young's modulus and edge cross-section area
        # thickness / area gives an approximate membrane stiffness
        k = self._young * self._thickness
        force_mag = k * strain
        dirs = np.zeros_like(diff)
        dirs[mask] = diff[mask] / dist[mask, None]
        fvec = force_mag[:, None] * dirs

        f = np.zeros_like(pos)
        np.add.at(f, i, fvec)
        np.subtract.at(f, j, fvec)
        return f

    def _resolve_collisions(self) -> None:
        # For each vertex, project outside the kinematic capsules with friction
        for cap in self._capsule_pose(self._current_time):
            self._pos, self._vel = self._resolve_capsule(self._pos, self._vel, cap)

    def _capsule_pose(self, t: float) -> list[dict]:
        # Articulated motion: both cylinders remain at their base positions,
        # with the upper limb rotating around the common center at the origin.
        # We drive a simple sinusoidal bend.
        bend = 0.25 * (1.0 - math.cos(math.pi * t / self._duration)) * math.pi  # 0 -> pi/2
        poses = []
        # Upper cylinder rotates about the x-axis? Actually original orientation is y-axis
        # aligned. We rotate the upper limb around z or y? Scene description: aligned along y
        # in xz plane (rotation about y). Bending the joint means rotating upper limb around
        # the x or z axis at the common center. We'll rotate upper around z to bring it
        # toward lower.
        base_u = np.array([-0.55, 0.0, 1.0])
        # Rotate around the z-axis by bend at origin (common joint near x=0)
        R = self._rotation_matrix_z(bend)
        base_u_rot = R @ base_u
        axis = R @ np.array([1.0, 0.0, 0.0])
        poses.append({"radius": 0.34, "height": 1.3, "base": base_u_rot, "axis_dir": axis})
        # Lower stays as configured
        poses.append({"radius": 0.31, "height": 1.3, "base": np.array([0.55, 0.0, 1.0]), "axis_dir": np.array([1.0, 0.0, 0.0])})
        return poses

    @staticmethod
    def _rotation_matrix_z(angle: float) -> np.ndarray:
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _resolve_capsule(self, pos: np.ndarray, vel: np.ndarray, cap: dict) -> tuple[np.ndarray, np.ndarray]:
        p = pos.copy()
        v = vel.copy()
        r = cap["radius"]
        h = cap["height"]
        base = cap["base"]
        axis = cap["axis_dir"]
        axis = axis / np.linalg.norm(axis)
        half = axis * (h * 0.5)
        a = base - half
        b = base + half
        for cap in [a, b]:
            diff = p - cap
            dist = np.linalg.norm(diff, axis=1)
            mask = dist < r
            if mask.any():
                n = diff[mask] / dist[mask, None]
                p[mask] += n * (r - dist[mask, None])
                vn = np.sum(v[mask] * n, axis=1)
                vt = v[mask] - n * vn[:, None]
                # Reflect normal velocity and apply friction
                v[mask] = -vn[:, None] * n * 0.0 + vt * (1.0 - self._friction)
        # Cylindrical shaft
        ab = b - a
        ab_norm_sq = np.dot(ab, ab)
        ap = p - a
        t = np.clip(np.dot(ap, ab) / ab_norm_sq, 0.0, 1.0)[:, None]
        closest = a + t * ab
        diff = p - closest
        dist = np.linalg.norm(diff, axis=1)
        mask = dist < r
        if mask.any():
            n = diff[mask] / dist[mask, None]
            p[mask] += n * (r - dist[mask, None])
            vn = np.sum(v[mask] * n, axis=1)
            vt = v[mask] - n * vn[:, None]
            v[mask] = -vn[:, None] * n * 0.0 + vt * (1.0 - self._friction)
        return p, v

    def _compute_signed_distance(self, pos: np.ndarray) -> np.ndarray:
        # Minimum signed distance to the two capsules (positive outside)
        sd = np.full(len(pos), 1e6)
        for cap in self._capsule_pose(self._current_time):
            sd = np.minimum(sd, self._capsule_signed_distance(pos, cap))
        return sd

    def _capsule_signed_distance(self, pos: np.ndarray, cap: dict) -> np.ndarray:
        r = cap["radius"]
        h = cap["height"]
        base = cap["base"]
        axis = cap["axis_dir"]
        axis = axis / np.linalg.norm(axis)
        half = axis * (h * 0.5)
        a = base - half
        b = base + half
        ap = pos - a
        ab = b - a
        t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0.0, 1.0)[:, None]
        closest = a + t * ab
        dist = np.linalg.norm(pos - closest, axis=1)
        return dist - r

    def _compute_deformation_gradient(self, pos: np.ndarray) -> np.ndarray:
        # Return per-triangle 2x2 deformation gradients in the local tangent plane.
        F = np.zeros((len(self._faces), 2, 2), dtype=float)
        for idx, f in enumerate(self._faces):
            # Use rest edge vectors projected to local 2D (first two components)
            r1 = self._rest_pos[f[1]] - self._rest_pos[f[0]]
            r2 = self._rest_pos[f[2]] - self._rest_pos[f[0]]
            # World space current edge vectors
            x1 = pos[f[1]] - pos[f[0]]
            x2 = pos[f[2]] - pos[f[0]]
            # Build approximate 2D coordinates in the triangle plane
            R = np.stack([r1, r2], axis=0)[:, :2]
            X = np.stack([x1, x2], axis=0)[:, :2]
            det = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
            if abs(det) > 1e-12:
                inv = np.linalg.inv(R)
                F[idx] = inv.T @ X
            else:
                F[idx] = np.eye(2)
        return F
