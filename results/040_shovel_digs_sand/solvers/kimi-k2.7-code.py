"""Shovel Digs Sand - DEM-style granular solver.

The solver advances a bed of spherical sand grains with explicit sub-stepping,
cKDTree broad-phase contacts, and penalty-based contact forces against the
static container and the kinematic shovel blade/shaft.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


def _quat_to_matrix(q):
    """Convert a quaternion in xyzw order to a rotation matrix.
    Columns of the returned matrix are the local basis vectors in world coords.
    """
    x, y, z, w = q
    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _box_signed_distance_and_normal(points, center, rotation, half_extents):
    """Signed distance and outward normal for an oriented box.

    Returns
    -------
    sd : (N,) array
        Signed distance (negative inside the solid box).
    normal : (N,3) array
        Unit outward normal in world coordinates.
    """
    q = (points - center) @ rotation
    abs_q = np.abs(q)
    inside = np.all(abs_q <= half_extents, axis=1)
    gap = half_extents - abs_q
    axis = np.argmin(gap, axis=1)
    signs = np.sign(q[np.arange(len(q)), axis])
    n_local = np.zeros_like(q)
    n_local[np.arange(len(q)), axis] = signs

    clamped = np.clip(q, -half_extents, half_extents)
    diff = q - clamped
    out_dist = np.linalg.norm(diff, axis=1)
    mask = out_dist > 1e-12
    n_local[mask] = diff[mask] / out_dist[mask, None]

    min_gap = np.min(gap, axis=1)
    sd = np.where(inside, -min_gap, out_dist)
    return sd, n_local @ rotation.T


def _capsule_signed_distance_and_normal(points, center, rotation, radius, height):
    """Signed distance and outward normal for a capsule aligned with the local y-axis."""
    q = (points - center) @ rotation
    half_height = 0.5 * height
    y = np.clip(q[:, 1], -half_height, half_height)
    center_points = np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=1)
    diff = q - center_points
    dist = np.linalg.norm(diff, axis=1)
    sd = dist - radius
    n_local = np.zeros_like(q)
    mask = dist > 1e-12
    n_local[mask] = diff[mask] / dist[mask, None]
    return sd, n_local @ rotation.T


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.config = scene.config if hasattr(scene, "config") else {}

        # Gravity and timing.
        self.gravity = np.array(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.sample_dt = float(self.config.get("sample_dt", 0.02))
        self.duration = float(self.config.get("duration", 4.0))

        # Initial particle state from the scene.
        frame0 = scene.initial_state()
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.N = len(self.position)
        self.t = 0.0

        # Particle radius is half the lattice spacing used by the scene, so grains are
        # touching in the initial configuration.
        init_cond = self.config.get("initial_conditions", {})
        self.spacing = float(init_cond.get("sand_particle_spacing", 0.075))
        self.radius = 0.5 * self.spacing
        self.density = float(init_cond.get("sand_bulk_density", 1550.0))

        # Sub-stepping.
        self.sub_dt = 0.0005
        if self.sample_dt / self.sub_dt != int(round(self.sample_dt / self.sub_dt)):
            # Choose a sub-step that divides sample_dt.
            self.sub_dt = self.sample_dt / max(1, int(round(self.sample_dt / self.sub_dt)))

        # Contact parameters.
        self.k0 = 1.0e4  # stiffness / unit mass (1/s^2)
        self.zeta = 0.3  # damping ratio
        self.mu = 0.5  # Coulomb friction
        self.drag = 1.0  # global velocity damping (1/s)

        # Static container planes: (point, inward-normal) pairs.
        bounds = self.config.get("boundary_conditions", {}).get("container_inner_bounds", [-2.35, 2.35, -1.5, 1.5])
        floor_z = float(self.config.get("boundary_conditions", {}).get("container_floor_z", 0.0))
        self.walls = [
            (np.array([bounds[0], 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
            (np.array([bounds[1], 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),
            (np.array([0.0, bounds[2], 0.0]), np.array([0.0, 1.0, 0.0])),
            (np.array([0.0, bounds[3], 0.0]), np.array([0.0, -1.0, 0.0])),
            (np.array([0.0, 0.0, floor_z]), np.array([0.0, 0.0, 1.0])),
        ]

        # Kinematic shovel motion.
        motion = self.config.get("kinematic_motion", {}).get("shovel_assembly", {})
        keyframes = motion.get("translation_keyframes", [])
        if len(keyframes) < 2:
            keyframes = [
                {"time": 0.0, "value": [-2.05, 0.0, 0.38]},
                {"time": 4.0, "value": [-2.05, 0.0, 0.38]},
            ]
        times = np.array([k["time"] for k in keyframes], dtype=float)
        vals = np.array([k["value"] for k in keyframes], dtype=float)
        self._traj_x = PchipInterpolator(times, vals[:, 0], extrapolate=True)
        self._traj_y = PchipInterpolator(times, vals[:, 1], extrapolate=True)
        self._traj_z = PchipInterpolator(times, vals[:, 2], extrapolate=True)
        self._traj_times = times
        self._traj_vals = vals

        # Load frozen geometry to extract shovel shapes.
        self.blade_spec: dict | None = None
        self.shaft_spec: dict | None = None
        geom_path = Path("scene_geometry.yaml")
        if geom_path.is_file():
            with open(geom_path, "r", encoding="utf-8") as fh:
                geom = yaml.safe_load(fh) or {}
            for obj in geom.get("objects", []):
                if obj.get("id") == "shovel_blade":
                    t = np.array(obj["transform"]["translation"], dtype=float)
                    q = np.array(obj["transform"].get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float)
                    he = np.array(obj["geometry"]["half_extents"], dtype=float)
                    self.blade_spec = {
                        "translation": t,
                        "rotation": _quat_to_matrix(q),
                        "half_extents": he,
                    }
                elif obj.get("id") == "shovel_shaft":
                    t = np.array(obj["transform"]["translation"], dtype=float)
                    q = np.array(obj["transform"].get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float)
                    self.shaft_spec = {
                        "translation": t,
                        "rotation": _quat_to_matrix(q),
                        "radius": float(obj["geometry"]["radius"]),
                        "height": float(obj["geometry"]["height"]),
                    }

        # If the frozen geometry is missing, fall back to the config keyframe origin.
        if self.blade_spec is None:
            t = np.array(keyframes[0]["value"], dtype=float)
            self.blade_spec = {
                "translation": t,
                "rotation": np.eye(3),
                "half_extents": np.array([0.14, 0.64, 0.34]),
            }
        if self.shaft_spec is None:
            self.shaft_spec = {
                "translation": t + np.array([-0.71, 0.0, 0.87]),
                "rotation": np.eye(3),
                "radius": 0.065,
                "height": 2.0,
            }

        # Fixed offset between blade and shaft so the shaft follows the blade motion.
        self._shaft_offset = self.shaft_spec["translation"] - self.blade_spec["translation"]

    def _shovel_state(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return blade position/velocity and shaft position/velocity at the given time."""
        blade_pos = np.array([self._traj_x(time), self._traj_y(time), self._traj_z(time)], dtype=float)
        # Numerical derivative for velocity.
        dt = 1e-6
        t0 = max(time - dt, 0.0)
        t1 = min(time + dt, self.duration)
        p0 = np.array([self._traj_x(t0), self._traj_y(t0), self._traj_z(t0)], dtype=float)
        p1 = np.array([self._traj_x(t1), self._traj_y(t1), self._traj_z(t1)], dtype=float)
        blade_vel = (p1 - p0) / (t1 - t0)
        shaft_pos = blade_pos + self._shaft_offset
        return blade_pos, blade_vel, shaft_pos, blade_vel

    def _boundary_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute contact forces from static container walls."""
        force = np.zeros_like(vel)
        for point, normal in self.walls:
            sd = np.dot(pos - point, normal)
            mask = sd < 0.0
            if not np.any(mask):
                continue
            overlap = -sd[mask]
            n = normal
            v = vel[mask]
            rel_n = np.einsum("ij,j->i", v, n)
            k = self.k0 * self.mass[mask]
            c = self.zeta * np.sqrt(self.mass[mask] * k)
            f_n_mag = k * overlap - c * rel_n
            f_n_mag = np.maximum(f_n_mag, 0.0)
            f = -f_n_mag[:, None] * n
            # Coulomb friction
            v_t = v - rel_n[:, None] * n
            v_t_mag = np.linalg.norm(v_t, axis=1)
            max_fric = self.mu * f_n_mag
            ft_mag = np.minimum(max_fric, c * v_t_mag)
            v_t_mag_safe = np.where(v_t_mag > 1e-12, v_t_mag, 1.0)
            f -= (ft_mag / v_t_mag_safe)[:, None] * v_t
            force[mask] += f
        return force

    def _kinematic_forces(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        center: np.ndarray,
        v_boundary: np.ndarray,
        sd: np.ndarray,
        normal: np.ndarray,
    ) -> np.ndarray:
        """Penalty + friction force against a moving boundary with given signed distance/normals."""
        force = np.zeros_like(vel)
        mask = sd < 0.0
        if not np.any(mask):
            return force
        overlap = -sd[mask]
        n = normal[mask]
        v = vel[mask] - v_boundary
        rel_n = np.einsum("ij,ij->i", v, n)
        k = self.k0 * self.mass[mask]
        c = self.zeta * np.sqrt(self.mass[mask] * k)
        f_n_mag = k * overlap - c * rel_n
        f_n_mag = np.maximum(f_n_mag, 0.0)
        f = -f_n_mag[:, None] * n
        v_t = v - rel_n[:, None] * n
        v_t_mag = np.linalg.norm(v_t, axis=1)
        max_fric = self.mu * f_n_mag
        ft_mag = np.minimum(max_fric, c * v_t_mag)
        v_t_mag_safe = np.where(v_t_mag > 1e-12, v_t_mag, 1.0)
        f -= (ft_mag / v_t_mag_safe)[:, None] * v_t
        force[mask] += f
        return force

    def _pair_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute particle-particle contact forces using cKDTree broad-phase."""
        force = np.zeros_like(vel)
        if self.N < 2:
            return force
        tree = cKDTree(pos)
        pairs = tree.query_pairs(2.0 * self.radius)
        if not pairs:
            return force
        pairs = np.asarray(list(pairs), dtype=int).reshape(-1, 2)
        i = pairs[:, 0]
        j = pairs[:, 1]
        diff = pos[j] - pos[i]
        dist = np.linalg.norm(diff, axis=1)
        mask = dist > 1e-12
        if not np.any(mask):
            return force
        i = i[mask]
        j = j[mask]
        diff = diff[mask]
        dist = dist[mask]
        n = diff / dist[:, None]
        overlap = 2.0 * self.radius - dist
        v_rel = vel[j] - vel[i]
        rel_n = np.einsum("ij,ij->i", v_rel, n)
        k = self.k0 * self.mass[i]
        c = self.zeta * np.sqrt(self.mass[i] * k)
        f_n_mag = k * overlap - c * rel_n
        f_n_mag = np.maximum(f_n_mag, 0.0)
        f = -f_n_mag[:, None] * n
        v_t = v_rel - rel_n[:, None] * n
        v_t_mag = np.linalg.norm(v_t, axis=1)
        max_fric = self.mu * f_n_mag
        ft_mag = np.minimum(max_fric, c * v_t_mag)
        v_t_mag_safe = np.where(v_t_mag > 1e-12, v_t_mag, 1.0)
        f -= (ft_mag / v_t_mag_safe)[:, None] * v_t
        np.add.at(force, i, f)
        np.add.at(force, j, -f)
        return force

    def _step(self) -> None:
        pos = self.position
        vel = self.velocity

        # External body force (gravity).
        f = self.mass[:, None] * self.gravity

        # Static container contacts.
        f += self._boundary_forces(pos, vel)

        # Kinematic shovel contacts.
        blade_pos, blade_vel, shaft_pos, shaft_vel = self._shovel_state(self.t)
        if self.blade_spec is not None:
            sd, normal = _box_signed_distance_and_normal(
                pos,
                blade_pos,
                self.blade_spec["rotation"],
                self.blade_spec["half_extents"],
            )
            f += self._kinematic_forces(pos, vel, blade_pos, blade_vel, sd, normal)
        if self.shaft_spec is not None:
            sd, normal = _capsule_signed_distance_and_normal(
                pos,
                shaft_pos,
                self.shaft_spec["rotation"],
                self.shaft_spec["radius"],
                self.shaft_spec["height"],
            )
            f += self._kinematic_forces(pos, vel, shaft_pos, shaft_vel, sd, normal)

        # Particle-particle contacts.
        f += self._pair_forces(pos, vel)

        # Integrate.
        vel += (f / self.mass[:, None]) * self.sub_dt
        # Global damping to represent inelastic granular collisions.
        vel *= np.exp(-self.drag * self.sub_dt)
        pos += vel * self.sub_dt

        self.t += self.sub_dt

    def _collect_contact_signed_distances(self) -> np.ndarray:
        """Collect signed distances for all active contacts at the current state."""
        sds = []
        # Particle-particle.
        if self.N >= 2:
            tree = cKDTree(self.position)
            pairs = tree.query_pairs(2.0 * self.radius)
            if pairs:
                pairs = np.asarray(list(pairs), dtype=int).reshape(-1, 2)
                diff = self.position[pairs[:, 1]] - self.position[pairs[:, 0]]
                dist = np.linalg.norm(diff, axis=1)
                sd = dist - 2.0 * self.radius
                sds.append(sd[sd < 0.0])
        # Static walls.
        for point, normal in self.walls:
            sd = np.dot(self.position - point, normal)
            sds.append(sd[sd < 0.0])
        # Shovel blade.
        blade_pos, _, shaft_pos, _ = self._shovel_state(self.t)
        if self.blade_spec is not None:
            sd, _ = _box_signed_distance_and_normal(
                self.position,
                blade_pos,
                self.blade_spec["rotation"],
                self.blade_spec["half_extents"],
            )
            sds.append(sd[sd < 0.0])
        # Shovel shaft.
        if self.shaft_spec is not None:
            sd, _ = _capsule_signed_distance_and_normal(
                self.position,
                shaft_pos,
                self.shaft_spec["rotation"],
                self.shaft_spec["radius"],
                self.shaft_spec["height"],
            )
            sds.append(sd[sd < 0.0])
        if sds:
            return np.concatenate(sds)
        return np.zeros(0, dtype=float)

    def _build_frame(self, time: float) -> StateFrame:
        contact_sd = self._collect_contact_signed_distances()
        ext_force = self.mass[:, None] * self.gravity
        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=np.ones(self.N, dtype=np.int32),
            attributes={"mass_quantity": self.mass[:, None].copy()},
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": ext_force,
            },
            scalars={"contact_length_scale": float(self.spacing)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target <= self.t:
            return self._build_frame(target)
        # Step until the requested time is reached or exceeded, then clamp.
        while self.t < target - 0.5 * self.sub_dt:
            self._step()
        if self.t > target:
            self.t = target
        return self._build_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "dem_sand",
            "particle_count": int(self.N),
            "particle_radius": float(self.radius),
            "sub_dt": float(self.sub_dt),
        }
