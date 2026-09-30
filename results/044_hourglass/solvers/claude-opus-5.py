"""Self-contained discrete-element solver for the flipped 2D hourglass.

Physical model
--------------
* Thin-slab granular medium: soft frictional grains whose dynamics live in the
  x-z plane.  The frozen slab depth at the neck is thinner than a grain, so the
  out-of-plane coordinate is held at its seeded value and never integrated.
* Grain/grain and grain/glass contacts use a linear spring-dashpot normal law
  (one sided, cohesionless) with regularized Coulomb friction.
* The hourglass is a kinematic shell rebuilt from the frozen inner profile.  Its
  two 180 degree flips are analytic smoothstep rotations about the y axis at the
  frozen event times; the wall velocity enters the contact law so the sand is
  genuinely dragged around by the rotating glass.
* Time stepping: fixed-step semi-implicit Euler with Verlet-skin neighbour lists
  from a cKDTree.  Mass is strictly conserved: no grain is created, deleted or
  re-weighted, and no motion is scripted.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


def _smoothstep(tau: float) -> tuple[float, float]:
    """C1 smoothstep and its derivative with respect to the parameter."""
    if tau <= 0.0:
        return 0.0, 0.0
    if tau >= 1.0:
        return 1.0, 0.0
    return 3.0 * tau * tau - 2.0 * tau**3, 6.0 * tau - 6.0 * tau * tau


class BenchmarkSolver:
    """Grain-scale DEM solver honouring the frozen hourglass geometry."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = dict(scene.config)
        frame0 = scene.initial_state()
        self.n = int(len(frame0.position))
        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.inv_mass = 1.0 / self.mass
        conditions = config["initial_conditions"]
        self.radius = float(conditions["particle_radius"])
        self.grain_volume = 4.0 * math.pi * self.radius**3 / 3.0
        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.duration = float(config["duration"])
        self.sample_dt = float(config["sample_dt"])
        self.initial_volume = float(
            frame0.scalars.get("initial_bulk_particle_volume", self.n * self.grain_volume)
        )
        # Thin-slab confinement: the out-of-plane coordinate is frozen.
        self.slab_y = self.pos[:, 1].copy()
        self.vel[:, 1] = 0.0
        self.seed = int(seed)
        self._parse_geometry(config)
        self._parse_events(config)
        self._setup_material()
        self.time = 0.0
        self._steps = 0
        self._projections = 0
        self._max_penetration = 0.0
        self._pairs = np.empty((0, 2), dtype=np.int64)
        self._pair_reference = None
        self._fext_sum = np.zeros((self.n, 3), dtype=float)
        self._fext_count = 0

    def _parse_geometry(self, config: dict) -> None:
        geometry = config.get("scene_geometry") or {}
        objects = {item["id"]: item for item in geometry.get("objects", [])}
        shell = objects.get("hourglass_shell")
        if shell is not None:
            profile = np.asarray(shell["geometry"]["inner_profile"], dtype=float)
            transform = shell.get("transform", {}) or {}
            shift = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            scale = np.asarray(transform.get("scale", [1.0, 1.0, 1.0]), dtype=float)
        else:  # pragma: no cover - frozen geometry always provides the shell
            profile = np.asarray(
                [
                    [0.46, 0.02], [0.44, 0.14], [0.32, 0.36], [0.12, 0.6], [0.038, 0.72],
                    [0.12, 0.84], [0.32, 1.08], [0.44, 1.3], [0.46, 1.42],
                ],
                dtype=float,
            )
            shift = np.array([0.0, 0.0, 0.08])
            scale = np.array([1.0, 0.16, 1.0])
        # World-space profile of the glass interior (radial half-width, height).
        self.prof_r = profile[:, 0] * float(scale[0])
        self.prof_z = profile[:, 1] * float(scale[2]) + float(shift[2])
        self.z_bottom = float(self.prof_z[0])
        self.z_top = float(self.prof_z[-1])
        # Flip pivot: the geometric centre of the closed glass body.
        self.pivot = np.array([float(shift[0]), 0.0, 0.5 * (self.z_bottom + self.z_top)])
        # Local (pivot-centred) polyline used for signed-distance queries.
        self.loc_r = self.prof_r.copy()
        self.loc_z = self.prof_z - self.pivot[2]
        self.loc_z_bottom = float(self.loc_z[0])
        self.loc_z_top = float(self.loc_z[-1])
        self.seg_a = np.stack((self.loc_r[:-1], self.loc_z[:-1]), axis=1)
        self.seg_b = np.stack((self.loc_r[1:], self.loc_z[1:]), axis=1)
        self.seg_d = self.seg_b - self.seg_a
        self.seg_len2 = np.maximum(np.sum(self.seg_d**2, axis=1), 1e-18)
        floor = objects.get("support_floor")
        if floor is not None:
            half = np.asarray(floor["geometry"]["half_extents"], dtype=float)
            centre = np.asarray(
                (floor.get("transform", {}) or {}).get("translation", [0.0, 0.0, 0.0]),
                dtype=float,
            )
            self.floor_z = float(centre[2] + half[2])
        else:  # pragma: no cover
            self.floor_z = 0.0

    def _parse_events(self, config: dict) -> None:
        self.flips: list[tuple[float, float, float]] = []
        for event in config.get("events", []) or []:
            if str(event.get("type")) != "flip":
                continue
            start = float(event.get("time", 0.0))
            span = max(float(event.get("duration", 0.8)), 1e-6)
            angle = math.radians(float(event.get("angle_degrees", 180.0)))
            self.flips.append((start, span, angle))
        self.flips.sort(key=lambda item: item[0])

    def _setup_material(self) -> None:
        mean_mass = float(np.mean(self.mass))
        self.dt = 1.0e-4
        self.substeps = max(int(round(self.sample_dt / self.dt)), 1)
        self.dt = self.sample_dt / self.substeps
        # Normal stiffness chosen so the static pile compresses ~3% of a radius.
        self.k_n = 4000.0
        self.k_wall = 2.0 * self.k_n
        m_eff = 0.5 * mean_mass
        self.c_n = 2.0 * 0.45 * math.sqrt(self.k_n * m_eff)
        self.c_wall = 2.0 * 0.45 * math.sqrt(self.k_wall * mean_mass)
        # Sand-on-sand and sand-on-glass Coulomb coefficients.  Grain spin is not
        # integrated, so sliding-only contacts already give a steep repose angle;
        # keeping mu moderate avoids a permanent arch at the narrow neck.
        self.mu_pp = 0.35
        self.mu_wall = 0.25
        self.background_damping = mean_mass / 3.0
        g = float(np.linalg.norm(self.gravity))
        self.f_cap_pair = 150.0 * mean_mass * g
        self.f_cap_wall = 400.0 * mean_mass * g
        self.skin = 0.5 * self.radius
        self.cutoff = 2.0 * self.radius + self.skin
        self.v_max_late = 8.0
        self.v_max_early = 1.0

    # -------------------------------------------------------------- kinematics
    def _rotation(self, t: float) -> tuple[float, float]:
        """Accumulated flip angle and angular rate about +y at time ``t``."""
        angle = 0.0
        rate = 0.0
        for start, span, total in self.flips:
            tau = (t - start) / span
            value, derivative = _smoothstep(tau)
            angle += total * value
            rate += total * derivative / span
        return angle, rate

    @staticmethod
    def _basis(angle: float) -> tuple[float, float]:
        return math.cos(angle), math.sin(angle)

    def _to_local(self, xz: np.ndarray, cos_a: float, sin_a: float) -> np.ndarray:
        """World x-z offsets from the pivot expressed in the glass frame.

        A rotation by ``angle`` about +y maps local (x,z) to world
        (x cos + z sin, -x sin + z cos); the inverse is applied here.
        """
        out = np.empty_like(xz)
        out[:, 0] = cos_a * xz[:, 0] - sin_a * xz[:, 1]
        out[:, 1] = sin_a * xz[:, 0] + cos_a * xz[:, 1]
        return out

    def _to_world(self, xz: np.ndarray, cos_a: float, sin_a: float) -> np.ndarray:
        out = np.empty_like(xz)
        out[:, 0] = cos_a * xz[:, 0] + sin_a * xz[:, 1]
        out[:, 1] = -sin_a * xz[:, 0] + cos_a * xz[:, 1]
        return out

    # ------------------------------------------------------------- geometry ops
    def _wall_query(self, local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Signed gaps and inward normals for the glass side wall and caps.

        ``local`` holds pivot-centred (x, z) glass-frame coordinates.  Returns
        (side gap, side inward normal in local x-z, top gap, bottom gap) where a
        positive gap means the grain centre is inside that surface.
        """
        u = np.abs(local[:, 0])
        z = local[:, 1]
        query = np.stack((u, z), axis=1)
        # Distance to every profile segment, vectorised over grains x segments.
        delta = query[:, None, :] - self.seg_a[None, :, :]
        param = np.clip(
            np.einsum("nsd,sd->ns", delta, self.seg_d) / self.seg_len2[None, :], 0.0, 1.0
        )
        closest = self.seg_a[None, :, :] + param[:, :, None] * self.seg_d[None, :, :]
        offset = query[:, None, :] - closest
        dist2 = np.sum(offset**2, axis=2)
        best = np.argmin(dist2, axis=1)
        rows = np.arange(len(u))
        near = closest[rows, best, :]
        gap_vec = query - near
        dist = np.sqrt(np.maximum(np.sum(gap_vec**2, axis=1), 0.0))
        # Inside test against the interpolated half-width at this height.
        r_at_z = np.interp(z, self.loc_z, self.loc_r)
        inside = (u <= r_at_z) & (z >= self.loc_z_bottom) & (z <= self.loc_z_top)
        side_gap = np.where(inside, dist, -dist)
        # Inward normal points from the grain towards the wall foot point.
        safe = np.maximum(dist, 1e-12)
        normal_u = -gap_vec[:, 0] / safe
        normal_z = -gap_vec[:, 1] / safe
        outside = ~inside
        normal_u[outside] *= -1.0
        normal_z[outside] *= -1.0
        degenerate = dist < 1e-12
        if np.any(degenerate):
            normal_u[degenerate] = -1.0
            normal_z[degenerate] = 0.0
        sign_x = np.where(local[:, 0] >= 0.0, 1.0, -1.0)
        normal = np.stack((normal_u * sign_x, normal_z), axis=1)
        top_gap = self.loc_z_top - z
        bottom_gap = z - self.loc_z_bottom
        return side_gap, normal, top_gap, bottom_gap

    # ------------------------------------------------------------ neighbourhood
    def _refresh_pairs(self, xz: np.ndarray) -> None:
        if self._pair_reference is not None:
            moved = np.max(np.abs(xz - self._pair_reference))
            if moved < 0.5 * self.skin:
                return
        tree = cKDTree(xz)
        pairs = tree.query_pairs(self.cutoff, output_type="ndarray")
        self._pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self._pair_reference = xz.copy()

    # ------------------------------------------------------------------ forces
    def _forces(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Total in-plane force and the external (gravity + wall) force."""
        xz = self.pos[:, [0, 2]]
        force = np.zeros((self.n, 2), dtype=float)
        vxz = self.vel[:, [0, 2]]
        weight = np.outer(self.mass, self.gravity[[0, 2]])
        force += weight
        external = weight.copy()

        # --- grain/grain contacts -------------------------------------------
        self._refresh_pairs(xz)
        pairs = self._pairs
        if len(pairs):
            i = pairs[:, 0]
            j = pairs[:, 1]
            d = xz[i] - xz[j]
            dist = np.sqrt(np.maximum(np.sum(d**2, axis=1), 1e-24))
            overlap = 2.0 * self.radius - dist
            active = overlap > 0.0
            if np.any(active):
                i = i[active]
                j = j[active]
                overlap = overlap[active]
                dist_a = dist[active]
                normal = d[active] / dist_a[:, None]
                rel = vxz[i] - vxz[j]
                v_n = np.sum(rel * normal, axis=1)
                f_n = self.k_n * overlap - self.c_n * v_n
                f_n = np.clip(f_n, 0.0, self.f_cap_pair)
                v_t = rel - v_n[:, None] * normal
                speed_t = np.sqrt(np.maximum(np.sum(v_t**2, axis=1), 1e-24))
                mag_t = np.minimum(self.mu_pp * f_n, self.c_n * speed_t)
                unit_t = v_t / speed_t[:, None]
                contact = f_n[:, None] * normal - mag_t[:, None] * unit_t
                index = np.concatenate((i, j))
                for axis in (0, 1):
                    signed = np.concatenate((contact[:, axis], -contact[:, axis]))
                    force[:, axis] += np.bincount(index, weights=signed, minlength=self.n)
                self._max_penetration = max(self._max_penetration, float(np.max(overlap)))

        # --- grain/glass contacts -------------------------------------------
        angle, rate = self._rotation(t)
        cos_a, sin_a = self._basis(angle)
        offset = np.empty((self.n, 2), dtype=float)
        offset[:, 0] = xz[:, 0] - self.pivot[0]
        offset[:, 1] = xz[:, 1] - self.pivot[2]
        local = self._to_local(offset, cos_a, sin_a)
        side_gap, side_normal_local, top_gap, bottom_gap = self._wall_query(local)
        # Wall material velocity from rigid rotation about +y: v = omega x r.
        wall_v = np.empty((self.n, 2), dtype=float)
        wall_v[:, 0] = rate * offset[:, 1]
        wall_v[:, 1] = -rate * offset[:, 0]

        side_normal = self._to_world(side_normal_local, cos_a, sin_a)
        wall_force = np.zeros((self.n, 2), dtype=float)
        wall_force += self._contact_with_wall(side_gap, side_normal, vxz, wall_v)
        cap_top = np.tile(np.array([0.0, -1.0]), (self.n, 1))
        cap_bottom = np.tile(np.array([0.0, 1.0]), (self.n, 1))
        radial = np.abs(local[:, 0])
        inside_cap = radial <= (self.loc_r[0] + self.loc_r[-1]) * 0.5 + self.radius
        top_gap = np.where(inside_cap, top_gap, np.inf)
        bottom_gap = np.where(inside_cap, bottom_gap, np.inf)
        wall_force += self._contact_with_wall(
            top_gap, self._to_world(cap_top, cos_a, sin_a), vxz, wall_v
        )
        wall_force += self._contact_with_wall(
            bottom_gap, self._to_world(cap_bottom, cos_a, sin_a), vxz, wall_v
        )

        # --- static support floor (safety boundary below the glass) ----------
        floor_gap = self.pos[:, 2] - self.floor_z
        wall_force += self._contact_with_wall(
            floor_gap, cap_bottom, vxz, np.zeros((self.n, 2), dtype=float)
        )

        force += wall_force
        external += wall_force
        # Weak background drag: quiets grain-scale jitter, negligible in flight.
        force -= self.background_damping * vxz
        return force, external

    def _contact_with_wall(
        self,
        gap: np.ndarray,
        normal: np.ndarray,
        vxz: np.ndarray,
        wall_v: np.ndarray,
    ) -> np.ndarray:
        """Spring-dashpot + friction response for one boundary surface."""
        out = np.zeros((self.n, 2), dtype=float)
        active = np.flatnonzero(gap < self.radius)
        if len(active) == 0:
            return out
        overlap = self.radius - gap[active]
        nrm = normal[active]
        rel = vxz[active] - wall_v[active]
        v_n = np.sum(rel * nrm, axis=1)
        f_n = self.k_wall * overlap - self.c_wall * v_n
        f_n = np.clip(f_n, 0.0, self.f_cap_wall)
        v_t = rel - v_n[:, None] * nrm
        speed_t = np.sqrt(np.maximum(np.sum(v_t**2, axis=1), 1e-24))
        mag_t = np.minimum(self.mu_wall * f_n, self.c_wall * speed_t)
        unit_t = v_t / speed_t[:, None]
        out[active] = f_n[:, None] * nrm - mag_t[:, None] * unit_t
        if len(overlap):
            self._max_penetration = max(self._max_penetration, float(np.max(overlap)))
        return out

    # -------------------------------------------------------------- integration
    def _velocity_cap(self, t: float) -> float:
        """Ramped speed limit that regularizes the seeded grain overlaps."""
        if t <= 0.2:
            return self.v_max_early
        if t >= 0.55:
            return self.v_max_late
        blend = (t - 0.2) / 0.35
        return self.v_max_early + blend * (self.v_max_late - self.v_max_early)

    def _step(self) -> None:
        t_mid = self.time + 0.5 * self.dt
        force, external = self._forces(t_mid)
        acceleration = force * self.inv_mass[:, None]
        vxz = self.vel[:, [0, 2]] + self.dt * acceleration
        limit = self._velocity_cap(t_mid)
        speed = np.sqrt(np.sum(vxz**2, axis=1))
        scale = np.where(speed > limit, limit / np.maximum(speed, 1e-12), 1.0)
        vxz *= scale[:, None]
        self.vel[:, 0] = vxz[:, 0]
        self.vel[:, 2] = vxz[:, 1]
        self.vel[:, 1] = 0.0
        self.pos[:, 0] += self.dt * self.vel[:, 0]
        self.pos[:, 2] += self.dt * self.vel[:, 2]
        self.pos[:, 1] = self.slab_y
        self.time += self.dt
        self._steps += 1
        self._fext_sum[:, 0] += external[:, 0]
        self._fext_sum[:, 2] += external[:, 1]
        self._fext_count += 1
        if self._steps % 50 == 0:
            self._confine()

    def _confine(self) -> None:
        """Hard fallback so no grain can ever leave the glass interior."""
        angle, _ = self._rotation(self.time)
        cos_a, sin_a = self._basis(angle)
        offset = np.empty((self.n, 2), dtype=float)
        offset[:, 0] = self.pos[:, 0] - self.pivot[0]
        offset[:, 1] = self.pos[:, 2] - self.pivot[2]
        local = self._to_local(offset, cos_a, sin_a)
        z = np.clip(local[:, 1], self.loc_z_bottom + self.radius, self.loc_z_top - self.radius)
        r_at_z = np.interp(z, self.loc_z, self.loc_r)
        limit = np.maximum(r_at_z - self.radius, 0.25 * self.radius)
        u = np.abs(local[:, 0])
        bad = (u > limit + 2.0 * self.radius) | (
            np.abs(local[:, 1] - z) > 2.0 * self.radius
        )
        if not np.any(bad):
            return
        idx = np.flatnonzero(bad)
        sign_x = np.where(local[idx, 0] >= 0.0, 1.0, -1.0)
        local[idx, 0] = sign_x * np.minimum(u[idx], limit[idx])
        local[idx, 1] = z[idx]
        fixed = self._to_world(local[idx], cos_a, sin_a)
        self.pos[idx, 0] = fixed[:, 0] + self.pivot[0]
        self.pos[idx, 2] = fixed[:, 1] + self.pivot[2]
        self.vel[idx] *= 0.1
        self._projections += len(idx)

    # ------------------------------------------------------------------- output
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 0.25 * self.dt:
            self._step()
        return self._frame(target)

    def _frame(self, target: float) -> StateFrame:
        if self._fext_count > 0:
            external = self._fext_sum / float(self._fext_count)
        else:
            _, ext2 = self._forces(target)
            external = np.zeros((self.n, 3), dtype=float)
            external[:, 0] = ext2[:, 0]
            external[:, 2] = ext2[:, 1]
        self._fext_sum = np.zeros((self.n, 3), dtype=float)
        self._fext_count = 0

        angle, rate = self._rotation(target)
        cos_a, sin_a = self._basis(angle)
        offset = np.empty((self.n, 2), dtype=float)
        offset[:, 0] = self.pos[:, 0] - self.pivot[0]
        offset[:, 1] = self.pos[:, 2] - self.pivot[2]
        local = self._to_local(offset, cos_a, sin_a)
        side_gap, _, top_gap, bottom_gap = self._wall_query(local)
        wall_gap = np.minimum(np.minimum(side_gap, top_gap), bottom_gap) - self.radius

        xz = self.pos[:, [0, 2]]
        self._refresh_pairs(xz)
        pair_gap = np.empty(0, dtype=float)
        if len(self._pairs):
            d = xz[self._pairs[:, 0]] - xz[self._pairs[:, 1]]
            dist = np.sqrt(np.maximum(np.sum(d**2, axis=1), 1e-24))
            pair_gap = dist - 2.0 * self.radius
        gaps = np.concatenate((wall_gap, pair_gap))
        near = gaps < 0.5 * self.radius
        if np.count_nonzero(near) < 32:
            order = np.argsort(gaps, kind="stable")[: min(32, len(gaps))]
            contact_gap = gaps[order]
        else:
            contact_gap = gaps[near]
        contact_gap = np.ascontiguousarray(contact_gap, dtype=float)

        position = np.ascontiguousarray(self.pos, dtype=float)
        velocity = np.ascontiguousarray(self.vel, dtype=float)
        frame = StateFrame(
            time=float(target),
            position=position,
            velocity=velocity,
            mass=np.ascontiguousarray(self.mass, dtype=float),
            attributes={
                "mass_quantity": np.ascontiguousarray(self.mass, dtype=float),
                "volume_quantity": np.full(self.n, self.grain_volume, dtype=float),
                "particle_radius": np.full(self.n, self.radius, dtype=float),
            },
            fields={
                "contact_signed_distance": contact_gap,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": np.ascontiguousarray(external, dtype=float),
            },
            scalars={
                "contact_length_scale": float(self.radius),
                "hourglass_rotation_angle": float(angle),
                "hourglass_rotation_rate": float(rate),
                "initial_bulk_particle_volume": float(self.initial_volume),
                "bulk_particle_volume": float(self.n * self.grain_volume),
                "total_mass": float(np.sum(self.mass)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "2d_thin_slab_dem_spring_dashpot_coulomb",
            "grain_count": int(self.n),
            "dt": float(self.dt),
            "steps": int(self._steps),
            "substeps_per_sample": int(self.substeps),
            "normal_stiffness": float(self.k_n),
            "friction_grain": float(self.mu_pp),
            "friction_wall": float(self.mu_wall),
            "flip_count": int(len(self.flips)),
            "final_rotation_deg": float(math.degrees(self._rotation(self.duration)[0])),
            "max_penetration_m": float(self._max_penetration),
            "containment_projections": int(self._projections),
            "total_mass_kg": float(np.sum(self.mass)),
            "seed": int(self.seed),
        }
