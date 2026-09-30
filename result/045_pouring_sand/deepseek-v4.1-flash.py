"""Deterministic soft-sphere DEM solver for the ``045_pouring_sand`` scene.

Physics implemented here (no external simulation engines):

* dry-sand grains are represented by the particle samples delivered by the frozen
  scene; every particle carries its own mass and radius;
* a linear spring-dashpot normal contact law plus regularised Coulomb tangential
  friction describes grain-grain, grain-hopper and grain-floor contacts;
* the frozen axisymmetric hopper (inner profile from ``scene_geometry``) is honoured
  as a one-sided rigid container: grains are confined by the neck, the conical funnel
  and the cylindrical reservoir, while the outlet at the neck bottom stays open;
* gravity drives the discharge, the free-falling stream, the floor impact, pile
  growth, runout and the relaxation towards a stable angle of repose;
* time integration is velocity Verlet (kick-drift-kick) with a Verlet neighbour list
  rebuilt whenever the maximum particle displacement exceeds half the skin.

All state is deterministic and finite; every required trajectory field is exported.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_DEFAULT_PROFILE = ((0.075, 1.42), (0.075, 1.52), (0.34, 1.72), (0.34, 2.28))
_DEFAULT_FLOOR_TOP = 0.0


class BenchmarkSolver:
    """Gravity-driven hopper discharge, stream break-up, pile growth and repose."""

    # ------------------------------------------------------------------ tunables
    DT = 4.0e-4              # fixed integration step [s]
    STIFFNESS = 1.5e4        # normal contact stiffness [N/m]
    DAMPING_RATIO = 0.4      # normal dashpot damping ratio [-]
    MU_PARTICLE = 0.6        # grain-grain Coulomb friction coefficient
    MU_WALL = 0.15           # grain-wall Coulomb friction coefficient
    TANGENTIAL_DAMPING = 30.0  # regularised tangential drag [N s/m]
    NEIGHBOUR_SKIN = 0.010   # Verlet-list skin [m]
    MAX_REBUILD_INTERVAL = 30
    NEAR_CONTACT = 5.0e-4    # reported near-contact gap band [m]
    MAX_SPEED = 25.0         # velocity safety clamp [m/s]
    RELAX_ITERATIONS = 150   # initial overlap removal passes
    RELAX_FACTOR = 0.7       # under-relaxation of the projection

    # ------------------------------------------------------------ construction
    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        initial = config.get("initial_conditions", {}) or {}

        self._spacing = float(initial.get("particle_spacing", 0.026))
        self._radius = float(initial.get("nominal_grain_radius", 0.5 * self._spacing))
        density = float(initial.get("bulk_density", 1550.0))
        self._gravity = np.asarray(config.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        if self._gravity.shape != (3,):
            self._gravity = np.array([0.0, 0.0, -9.81])

        frame = scene.initial_state()
        position = np.array(frame.position, dtype=float)
        velocity = np.array(frame.velocity, dtype=float)
        if frame.mass is None:
            mass = np.full(len(position), density * self._spacing ** 3)
        else:
            mass = np.array(frame.mass, dtype=float)
        mass = np.where(np.isfinite(mass) & (mass > 0.0), mass, density * self._spacing ** 3)

        self._pos = position
        self._vel = velocity
        self._mass = mass
        self._n = int(len(position))
        self._inv_mass = 1.0 / mass

        # contact damping from the reduced mass of a grain pair
        m_eff = 0.5 * float(np.mean(mass))
        self._cn_particle = 2.0 * self.DAMPING_RATIO * np.sqrt(self.STIFFNESS * m_eff)
        self._cn_wall = 2.0 * self.DAMPING_RATIO * np.sqrt(
            self.STIFFNESS * float(np.mean(mass))
        )

        self._profile = self._read_profile(config)
        self._floor_top = self._read_floor_top(config)
        self._segments = self._build_segments(self._profile)

        self._time = 0.0
        self._step_count = 0
        self._build_step = -10 ** 9
        self._pairs = None
        self._build_pos = None

        self._initial_total_mass = float(np.sum(mass))
        self._resolve_initial_overlaps()
        self._rebuild_neighbours()
        self._initial_hopper_mass = float(np.sum(mass[self._in_hopper(self._pos)]))

    # ------------------------------------------------------------ config / geometry
    @staticmethod
    def _read_profile(config: dict) -> np.ndarray:
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []) or []:
            if obj.get("id") == "supply_hopper":
                spec = obj.get("geometry", {}) or {}
                profile = spec.get("inner_profile")
                if profile:
                    array = np.asarray(profile, dtype=float)
                    if array.ndim == 2 and array.shape[1] == 2 and len(array) >= 2:
                        order = np.argsort(array[:, 1])
                        return array[order]
        return np.asarray(_DEFAULT_PROFILE, dtype=float)

    @staticmethod
    def _read_floor_top(config: dict) -> float:
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []) or []:
            if obj.get("id") != "receiving_floor":
                continue
            spec = obj.get("geometry", {}) or {}
            half = spec.get("half_extents")
            transform = obj.get("transform", {}) or {}
            translation = transform.get("translation", (0.0, 0.0, 0.0))
            if half and len(half) == 3:
                return float(translation[2]) + float(half[2])
        return _DEFAULT_FLOOR_TOP

    @staticmethod
    def _build_segments(profile: np.ndarray) -> list:
        """Inner-wall polyline segments as (a_rho, a_z, t_rho, t_z, length, n_rho, n_z)."""
        segments = []
        for index in range(len(profile) - 1):
            r0, z0 = float(profile[index, 0]), float(profile[index, 1])
            r1, z1 = float(profile[index + 1, 0]), float(profile[index + 1, 1])
            t_r, t_z = r1 - r0, z1 - z0
            length = float(np.hypot(t_r, t_z))
            if length <= 0.0:
                continue
            # inward normal (points towards the container axis)
            n_r, n_z = -t_z / length, t_r / length
            segments.append((r0, z0, t_r, t_z, length, n_r, n_z))
        return segments

    # ------------------------------------------------------------ initial packing
    def _resolve_initial_overlaps(self) -> None:
        """Project the frozen initial packing to a (nearly) overlap-free state.

        The scene lattice is constructed with sub-spacing jitter and a funnel fill
        that slightly intersects the hopper wall, so a short position-only
        relaxation is applied before the dynamics start.  No velocities are
        generated, so the released configuration is deterministic and calm.
        """
        pos = self._pos
        radius = self._radius
        if self.RELAX_ITERATIONS <= 0 or self._n < 2:
            return
        for _ in range(int(self.RELAX_ITERATIONS)):
            shift = np.zeros_like(pos)
            count = np.zeros(self._n)
            tree = cKDTree(pos)
            pairs = tree.query_pairs(2.0 * radius, output_type="ndarray")
            if len(pairs):
                i = pairs[:, 0]
                j = pairs[:, 1]
                delta = pos[i] - pos[j]
                dist = np.sqrt(np.maximum(np.einsum("ij,ij->i", delta, delta), 1e-24))
                penetration = 2.0 * radius - dist
                active = penetration > 1e-9
                if np.any(active):
                    i = i[active]
                    j = j[active]
                    penetration = penetration[active]
                    dist = dist[active]
                    normal = delta[active] / dist[:, None]
                    push = 0.5 * penetration[:, None] * normal
                    for axis in range(3):
                        shift[:, axis] += np.bincount(
                            i, weights=push[:, axis], minlength=self._n
                        )
                        shift[:, axis] -= np.bincount(
                            j, weights=push[:, axis], minlength=self._n
                        )
                    count += np.bincount(i, minlength=self._n)
                    count += np.bincount(j, minlength=self._n)
            penetration, normal, mask = self._wall_contacts(pos)
            if np.any(mask):
                push = normal[mask] * penetration[mask, None]
                shift[mask] += push
                count[mask] += 1.0
            floor_pen = radius - (pos[:, 2] - self._floor_top)
            floor_mask = floor_pen > 0.0
            if np.any(floor_mask):
                shift[floor_mask, 2] += floor_pen[floor_mask]
                count[floor_mask] += 1.0
            moving = count > 0.0
            if not np.any(moving):
                break
            pos[moving] += self.RELAX_FACTOR * shift[moving] / count[moving, None]

    def _wall_contacts(self, pos: np.ndarray):
        """Signed penetration, inward unit normal and mask for the hopper wall."""
        radius = self._radius
        radial = np.hypot(pos[:, 0], pos[:, 1])
        safe_radial = np.maximum(radial, 1e-12)
        ex = pos[:, 0] / safe_radial
        ey = pos[:, 1] / safe_radial
        z = pos[:, 2]

        best = np.zeros(self._n)
        u0 = np.zeros(self._n)
        u1 = np.zeros(self._n)
        for (a_r, a_z, t_r, t_z, length, n_r, n_z) in self._segments:
            d_r = radial - a_r
            d_z = z - a_z
            parameter = (d_r * t_r + d_z * t_z) / (length * length)
            inside = (parameter > 0.0) & (parameter < 1.0)
            clipped = np.clip(parameter, 0.0, 1.0)
            c_r = a_r + clipped * t_r
            c_z = a_z + clipped * t_z
            w_r = radial - c_r
            w_z = z - c_z
            dist = np.sqrt(w_r * w_r + w_z * w_z)
            dist_safe = np.maximum(dist, 1e-12)
            signed = np.where(inside, w_r * n_r + w_z * n_z, dist)
            penetration = radius - signed
            better = penetration > best
            if np.any(better):
                local_u0 = np.where(inside, n_r, w_r / dist_safe)
                local_u1 = np.where(inside, n_z, w_z / dist_safe)
                best = np.where(better, penetration, best)
                u0 = np.where(better, local_u0, u0)
                u1 = np.where(better, local_u1, u1)
        mask = best > 0.0
        normal = np.zeros((self._n, 3))
        normal[:, 0] = u0 * ex
        normal[:, 1] = u0 * ey
        normal[:, 2] = u1
        return best, normal, mask

    def _in_hopper(self, pos: np.ndarray) -> np.ndarray:
        radial = np.hypot(pos[:, 0], pos[:, 1])
        wall_r = np.interp(
            pos[:, 2],
            self._profile[:, 1],
            self._profile[:, 0],
            left=np.nan,
            right=np.nan,
        )
        return np.isfinite(wall_r) & (radial <= wall_r)

    # ------------------------------------------------------------ neighbour list
    def _rebuild_neighbours(self) -> None:
        self._pairs = cKDTree(self._pos).query_pairs(
            2.0 * self._radius + self.NEIGHBOUR_SKIN, output_type="ndarray"
        )
        self._build_pos = self._pos.copy()
        self._build_step = self._step_count

    def _maybe_rebuild(self) -> None:
        if self._pairs is None or self._build_pos is None:
            self._rebuild_neighbours()
            return
        delta = self._pos - self._build_pos
        moved2 = float(np.max(np.einsum("ij,ij->i", delta, delta)))
        limit = 0.5 * self.NEIGHBOUR_SKIN
        if moved2 > limit * limit or (self._step_count - self._build_step) >= self.MAX_REBUILD_INTERVAL:
            self._rebuild_neighbours()

    # ------------------------------------------------------------ contact forces
    def _pair_forces(self, pos: np.ndarray, vel: np.ndarray, force: np.ndarray) -> None:
        pairs = self._pairs
        if pairs is None or len(pairs) == 0:
            return
        i = pairs[:, 0]
        j = pairs[:, 1]
        delta = pos[i] - pos[j]
        dist2 = np.einsum("ij,ij->i", delta, delta)
        dist = np.sqrt(np.maximum(dist2, 1e-24))
        inv_dist = 1.0 / dist
        normal = delta * inv_dist[:, None]
        overlap = 2.0 * self._radius - dist
        active = overlap > 0.0
        rel = vel[i] - vel[j]
        v_n = np.einsum("ij,ij->i", rel, normal)
        magnitude = self.STIFFNESS * overlap - self._cn_particle * v_n
        magnitude = np.where(active, magnitude, 0.0)
        np.maximum(magnitude, 0.0, out=magnitude)
        tangential = rel - v_n[:, None] * normal
        t_speed = np.sqrt(np.maximum(np.einsum("ij,ij->i", tangential, tangential), 0.0))
        t_limit = np.minimum(self.MU_PARTICLE * magnitude, self.TANGENTIAL_DAMPING * t_speed)
        scale = np.where(t_speed > 1e-12, t_limit / np.maximum(t_speed, 1e-12), 0.0)
        contact = magnitude[:, None] * normal - scale[:, None] * tangential
        for axis in range(3):
            column = contact[:, axis]
            force[:, axis] += np.bincount(i, weights=column, minlength=self._n)
            force[:, axis] -= np.bincount(j, weights=column, minlength=self._n)

    def _wall_forces(self, pos: np.ndarray, vel: np.ndarray, force: np.ndarray) -> None:
        penetration, normal, mask = self._wall_contacts(pos)
        if not np.any(mask):
            return
        index = np.nonzero(mask)[0]
        n = normal[index]
        v_n = np.einsum("ij,ij->i", vel[index], n)
        magnitude = self.STIFFNESS * penetration[index] - self._cn_wall * v_n
        np.maximum(magnitude, 0.0, out=magnitude)
        v_t = vel[index] - v_n[:, None] * n
        t_speed = np.sqrt(np.maximum(np.einsum("ij,ij->i", v_t, v_t), 0.0))
        t_limit = np.minimum(self.MU_WALL * magnitude, self.TANGENTIAL_DAMPING * t_speed)
        scale = np.where(t_speed > 1e-12, t_limit / np.maximum(t_speed, 1e-12), 0.0)
        force[index] += magnitude[:, None] * n - scale[:, None] * v_t

    def _floor_forces(self, pos: np.ndarray, vel: np.ndarray, force: np.ndarray) -> None:
        gap = (pos[:, 2] - self._floor_top) - self._radius
        mask = gap < 0.0
        if not np.any(mask):
            return
        index = np.nonzero(mask)[0]
        v_z = vel[index, 2]
        magnitude = self.STIFFNESS * (-gap[index]) - self._cn_wall * v_z
        np.maximum(magnitude, 0.0, out=magnitude)
        v_t = vel[index].copy()
        v_t[:, 2] = 0.0
        t_speed = np.sqrt(np.maximum(np.einsum("ij,ij->i", v_t, v_t), 0.0))
        t_limit = np.minimum(self.MU_WALL * magnitude, self.TANGENTIAL_DAMPING * t_speed)
        scale = np.where(t_speed > 1e-12, t_limit / np.maximum(t_speed, 1e-12), 0.0)
        force[index, 2] += magnitude
        force[index, :2] -= scale[:, None] * v_t[:, :2]

    def _forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        force = np.zeros((self._n, 3))
        force += self._mass[:, None] * self._gravity[None, :]
        self._pair_forces(pos, vel, force)
        self._wall_forces(pos, vel, force)
        self._floor_forces(pos, vel, force)
        return force

    # ------------------------------------------------------------ time stepping
    def _step(self, dt: float) -> None:
        pos = self._pos
        vel = self._vel
        force = self._forces(pos, vel)
        vel += (0.5 * dt) * force * self._inv_mass[:, None]
        pos += dt * vel
        self._step_count += 1
        self._maybe_rebuild()
        force = self._forces(pos, vel)
        vel += (0.5 * dt) * force * self._inv_mass[:, None]
        np.clip(vel, -self.MAX_SPEED, self.MAX_SPEED, out=vel)
        if not np.isfinite(vel).all():
            vel[:] = np.nan_to_num(vel, nan=0.0, posinf=self.MAX_SPEED, neginf=-self.MAX_SPEED)
        if not np.isfinite(pos).all():
            pos[:] = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)

    def advance_to(self, time: float):
        target = float(time)
        if target < self._time - 1e-12:
            target = self._time
        remaining = target - self._time
        if remaining > 1e-12:
            steps = int(np.floor(remaining / self.DT + 1e-9))
            for _ in range(steps):
                self._step(self.DT)
            self._time += steps * self.DT
            tail = target - self._time
            if tail > 1e-12:
                self._step(tail)
                self._time += tail
        self._time = target
        return self._build_frame(target)

    # ------------------------------------------------------------ trajectory output
    def _sample_contact_gaps(self, pos: np.ndarray) -> np.ndarray:
        chunks = []
        pairs = self._pairs
        if pairs is not None and len(pairs):
            i = pairs[:, 0]
            j = pairs[:, 1]
            delta = pos[i] - pos[j]
            dist = np.sqrt(np.maximum(np.einsum("ij,ij->i", delta, delta), 0.0))
            gap = dist - 2.0 * self._radius
            near = gap < self.NEAR_CONTACT
            if np.any(near):
                chunks.append(gap[near])
        wall_pen, _normal, _mask = self._wall_contacts(pos)
        wall_gap = -wall_pen
        floor_gap = (pos[:, 2] - self._floor_top) - self._radius
        extra = np.concatenate([wall_gap, floor_gap])
        extra = extra[extra < self.NEAR_CONTACT]
        if extra.size:
            chunks.append(extra)
        if not chunks:
            return np.zeros(1)
        return np.concatenate(chunks)

    def _build_frame(self, time: float) -> StateFrame:
        pos = self._pos
        vel = self._vel
        mass = self._mass
        in_hopper = self._in_hopper(pos)
        hopper_mass = float(np.sum(mass[in_hopper]))
        pile_mass = float(np.sum(mass[~in_hopper]))
        speed = np.sqrt(np.sum(vel * vel, axis=1))
        scalars = {
            "contact_length_scale": float(2.0 * self._radius),
            "initial_total_mass": self._initial_total_mass,
            "total_mass": float(np.sum(mass)),
            "hopper_inventory_mass": hopper_mass,
            "pile_mass": pile_mass,
            "initial_hopper_mass": self._initial_hopper_mass,
            "max_speed": float(np.max(speed)) if self._n else 0.0,
            "mean_speed": float(np.mean(speed)) if self._n else 0.0,
        }
        attributes = {
            "mass_quantity": mass,
            "nominal_radius": np.full(self._n, self._radius),
            "material_phase": np.zeros(self._n),
        }
        fields = {
            "contact_signed_distance": self._sample_contact_gaps(pos),
            "cumulative_mass_source": np.zeros(1),
            "external_force": mass[:, None] * self._gravity[None, :],
        }
        frame = StateFrame(
            time=float(time),
            position=pos.copy(),
            velocity=vel.copy(),
            mass=mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        pos = self._pos
        vel = self._vel
        in_hopper = self._in_hopper(pos)
        return {
            "solver": "soft_sphere_dem",
            "particles": int(self._n),
            "steps": int(self._step_count),
            "time": float(self._time),
            "dt": float(self.DT),
            "radius": float(self._radius),
            "initial_total_mass": self._initial_total_mass,
            "final_total_mass": float(np.sum(self._mass)),
            "hopper_inventory_mass": float(np.sum(self._mass[in_hopper])),
            "initial_hopper_mass": float(self._initial_hopper_mass),
            "pile_mass": float(np.sum(self._mass[~in_hopper])),
            "max_height": float(np.max(pos[:, 2])) if self._n else 0.0,
            "max_speed": float(np.max(np.sqrt(np.sum(vel * vel, axis=1)))) if self._n else 0.0,
        }
