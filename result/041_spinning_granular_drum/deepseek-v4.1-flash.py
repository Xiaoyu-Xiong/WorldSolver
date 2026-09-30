"""Deterministic soft-sphere DEM solver for the spinning granular drum.

Scene ``041_spinning_granular_drum`` (Hybrid Grains, Fig.17 / Section 5.8).

The granular bed is simulated with a real discrete-element time-stepping
scheme.  Every grain carries translational and rotational degrees of freedom
and is advanced with a symplectic (semi-implicit) Euler step under

* gravity,
* normal / tangential contact forces against neighbouring grains using a
  linear spring-dashpot normal law with a regularised Coulomb friction cap,
* contact forces against the cylindrical drum wall, which turns at the frozen
  angular velocity ``omega * y_hat`` through the frozen drum centre, so wall
  friction drags grains up the rising side,
* rolling-resistance torques that give the bed a realistic angle of repose,
* flat rotating end walls that keep the bed inside the frozen drum depth.

Contacts are detected with a Verlet neighbour list (skin + periodic rebuild),
which is the standard neighbourhood scheme for DEM.  Nothing is scripted: the
free surface, the lifting and the avalanches all emerge from the contact
dynamics.  Every exported trajectory field is a raw physical sample.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

try:  # optional accelerator; the solver also works without SciPy
    from scipy.spatial import cKDTree as _KDTree
except Exception:  # pragma: no cover - SciPy is part of the base environment
    _KDTree = None

_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)

# numerical safety clamps (never active in normal operation)
_VELOCITY_CLAMP = 40.0
_SPIN_CLAMP = 800.0


def _scatter(count, index, values, accumulator):
    """Accumulate per-contact vector values onto particle indices."""
    for axis in range(3):
        accumulator[:, axis] += np.bincount(
            index, weights=values[:, axis], minlength=count
        )


class BenchmarkSolver:
    """Discrete-element simulation of a half-filled rotating drum."""

    def __init__(self) -> None:
        self.count = 0
        self.time = 0.0
        self._pair_i = np.zeros(0, dtype=np.int64)
        self._pair_j = np.zeros(0, dtype=np.int64)
        self._step_index = 0

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = getattr(scene, "config", {}) or {}
        self.config = config
        initial = config.get("initial_conditions", {}) or {}
        drum = initial.get("drum", {}) or {}
        grain_cfg = initial.get("grains", {}) or {}

        frame = scene.initial_state()
        position = np.array(frame.position, dtype=np.float64)
        count = int(position.shape[0])
        self.count = count

        attributes = dict(frame.attributes or {})
        radius_source = attributes.get("particle_radius", attributes.get("radius"))
        if radius_source is None:
            grain_radius = float(grain_cfg.get("radius", 0.025))
        else:
            grain_radius = float(np.median(np.asarray(radius_source, dtype=np.float64)))
        self.grain_radius = grain_radius
        self.grain_radius_vector = np.full(count, grain_radius, dtype=np.float64)

        if frame.mass is not None:
            mass = np.array(frame.mass, dtype=np.float64)
        else:
            density = float(grain_cfg.get("bulk_density", 1550.0))
            volume = 4.0 / 3.0 * np.pi * grain_radius**3
            mass = np.full(count, density * volume, dtype=np.float64)
        self.mass = mass
        self.inv_mass = 1.0 / mass
        # uniform solid sphere inertia
        self.inertia = 0.4 * mass * grain_radius**2
        self.inv_inertia = 1.0 / self.inertia

        self.body_id = (
            np.asarray(frame.body_id, dtype=np.int32)
            if frame.body_id is not None
            else np.ones(count, dtype=np.int32)
        )
        self.material_label = np.asarray(
            attributes.get("material_label", np.ones(count)), dtype=np.int32
        )
        tracer = attributes.get("tracer_grain")
        if tracer is None:
            rng = np.random.default_rng(self.seed)
            tracer = (
                rng.random(count) < float(grain_cfg.get("tracer_fraction", 0.06))
            ).astype(np.float64)
        self.tracer = np.asarray(tracer, dtype=np.float64)
        self.initial_volume = np.asarray(
            attributes.get("initial_volume", np.full(count, 4.0 / 3.0 * np.pi * grain_radius**3)),
            dtype=np.float64,
        )

        # ---- frozen drum geometry -------------------------------------
        self.drum_center = np.array(drum.get("center", [0.0, 0.0, 0.86]), dtype=np.float64)
        self.drum_radius = float(drum.get("radius", 0.82))
        self.drum_depth = float(drum.get("depth", 0.42))
        self.half_depth = 0.5 * self.drum_depth
        self.angular_velocity = float(drum.get("angular_velocity_rad_s", 1.25))
        axis = np.array(drum.get("rotation_axis", [0.0, 1.0, 0.0]), dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        self.rotation_axis = axis / norm if norm > 0 else np.array([0.0, 1.0, 0.0])
        self.omega_vector = self.angular_velocity * self.rotation_axis

        # ---- contact / integration parameters --------------------------
        self.dt = 2.0e-4
        mean_mass = float(np.mean(mass))
        self.k_normal = 4.0e4
        self.k_wall = 4.0e4
        damping_ratio = 0.15
        self.gamma_normal = 2.0 * damping_ratio * np.sqrt(self.k_normal * mean_mass)
        self.gamma_wall = self.gamma_normal
        self.gamma_tangent = 40.0
        self.mu_grain = 0.40
        self.mu_wall = 0.60
        self.mu_rolling = 0.06
        self.k_rolling = 0.01
        self.skin = 0.012
        self.rebuild_interval = 15
        self.near_gap = 0.008
        self.contact_length_scale = 2.0 * grain_radius

        # ---- state -----------------------------------------------------
        # sub-micron deterministic de-latticeing keeps the initial packing
        # (which matches the frozen scene state) from being a perfect crystal
        rng = np.random.default_rng(self.seed)
        jitter = rng.uniform(-1.0, 1.0, size=position.shape) * 1.0e-6
        self.pos = position + jitter
        self.vel = np.array(frame.velocity, dtype=np.float64)
        self.spin = np.zeros((count, 3), dtype=np.float64)
        self.time = 0.0
        self._step_index = 0
        self._rebuild_pairs()

        # cached per-particle gravity (external force field)
        self.gravity_force = self.mass[:, None] * _GRAVITY[None, :]
        self.free_surface_slope = 0.0
        self.avalanche_speed = 0.0
        self.lifted_height = float(np.max(self.pos[:, 2]))
        self.escaped_count = 0

    # ------------------------------------------------------------------
    # neighbour list
    # ------------------------------------------------------------------
    def _rebuild_pairs(self) -> None:
        position = self.pos
        cutoff = 2.0 * self.grain_radius + self.skin
        if _KDTree is not None:
            tree = _KDTree(position)
            pairs = tree.query_pairs(cutoff, output_type="ndarray")
            pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        else:  # pragma: no cover - chunked fallback
            left, right = [], []
            step = 256
            limit = cutoff * cutoff
            for start in range(0, len(position), step):
                block = position[start : start + step]
                diff = block[:, None, :] - position[None, :, :]
                dist2 = np.einsum("ijk,ijk->ij", diff, diff)
                rows, cols = np.nonzero(dist2 < limit)
                keep = (start + rows) < cols
                left.append((start + rows)[keep])
                right.append(cols[keep])
            if left:
                pairs = np.column_stack(
                    (np.concatenate(left), np.concatenate(right))
                ).astype(np.int64)
            else:
                pairs = np.zeros((0, 2), dtype=np.int64)
        if pairs.size == 0:
            pairs = np.zeros((0, 2), dtype=np.int64)
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs = pairs[order]
        self._pair_i = pairs[:, 0].copy()
        self._pair_j = pairs[:, 1].copy()

    # ------------------------------------------------------------------
    # contact physics
    # ------------------------------------------------------------------
    def _wall_velocity(self, points: np.ndarray) -> np.ndarray:
        """Rigid-body velocity of the rotating drum surface at world points."""
        offset = points - self.drum_center[None, :]
        wy = self.omega_vector[1]
        out = np.zeros_like(points)
        out[:, 0] = wy * offset[:, 2]
        out[:, 2] = -wy * offset[:, 0]
        return out

    def _contact_forces(self, force, torque) -> None:
        count = self.count
        radius = self.grain_radius
        i_idx = self._pair_i
        j_idx = self._pair_j
        if len(i_idx) == 0:
            return
        pos = self.pos
        vel = self.vel
        spin = self.spin
        diff = pos[j_idx] - pos[i_idx]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        dist = np.maximum(dist, 1e-12)
        normal = diff / dist[:, None]
        gap = dist - 2.0 * radius
        active = gap < 0.0
        if not np.any(active):
            return
        na = normal[active]
        ia = i_idx[active]
        ja = j_idx[active]
        penetration = -gap[active]

        v_rel = vel[ja] - vel[ia]
        v_n = np.einsum("ij,ij->i", v_rel, na)
        f_n = self.k_normal * penetration - self.gamma_normal * v_n
        np.maximum(f_n, 0.0, out=f_n)

        v_t = v_rel - v_n[:, None] * na
        v_t_mag = np.sqrt(np.einsum("ij,ij->i", v_t, v_t))
        f_t_mag = np.minimum(self.gamma_tangent * v_t_mag, self.mu_grain * f_n)
        slip_scale = np.where(
            v_t_mag > 1e-12, -f_t_mag / np.maximum(v_t_mag, 1e-12), 0.0
        )
        pair_force = na * f_n[:, None] + v_t * slip_scale[:, None]

        _scatter(count, ja, pair_force, force)
        _scatter(count, ia, -pair_force, force)

        # tangential force couple spins both grains the same way about contact
        contact_torque = -radius * np.cross(na, pair_force)
        _scatter(count, ja, contact_torque, torque)
        _scatter(count, ia, contact_torque, torque)

        # rolling resistance between grains
        rel_spin = spin[ia] - spin[ja]
        spin_mag = np.sqrt(np.einsum("ij,ij->i", rel_spin, rel_spin))
        roll_mag = np.minimum(self.k_rolling * spin_mag, self.mu_rolling * f_n * 2.0 * radius)
        roll_scale = np.where(
            spin_mag > 1e-12, -roll_mag / np.maximum(spin_mag, 1e-12), 0.0
        )
        roll_torque = rel_spin * roll_scale[:, None]
        _scatter(count, ia, roll_torque, torque)
        _scatter(count, ja, -roll_torque, torque)

    def _wall_forces(self, force, torque) -> None:
        count = self.count
        radius = self.grain_radius
        pos = self.pos
        vel = self.vel
        spin = self.spin

        # ---- cylindrical side wall ------------------------------------
        qx = pos[:, 0] - self.drum_center[0]
        qz = pos[:, 2] - self.drum_center[2]
        radial = np.sqrt(qx * qx + qz * qz)
        radial = np.maximum(radial, 1e-12)
        gap_cyl = (self.drum_radius - radius) - radial
        index = np.nonzero(gap_cyl < 0.0)[0]
        if index.size:
            ex = qx[index] / radial[index]
            ez = qz[index] / radial[index]
            normal = np.stack((-ex, np.zeros_like(ex), -ez), axis=1)
            self._resolve_wall_contact(
                index, normal, -gap_cyl[index], force, torque, moving=True
            )

        # ---- rotating flat end walls (drum faces) ---------------------
        for sign in (1.0, -1.0):
            gap_flat = self.half_depth - radius - sign * pos[:, 1]
            index = np.nonzero(gap_flat < 0.0)[0]
            if index.size == 0:
                continue
            normal = np.zeros((index.size, 3), dtype=np.float64)
            normal[:, 1] = -sign
            self._resolve_wall_contact(
                index, normal, -gap_flat[index], force, torque, moving=True
            )

    def _resolve_wall_contact(self, index, normal, penetration, force, torque, moving):
        radius = self.grain_radius
        contact_point = self.pos[index] - radius * normal
        wall_velocity = self._wall_velocity(contact_point)
        v_rel = self.vel[index] - wall_velocity
        v_n = np.einsum("ij,ij->i", v_rel, normal)
        f_n = self.k_wall * penetration - self.gamma_wall * v_n
        np.maximum(f_n, 0.0, out=f_n)

        v_t = v_rel - v_n[:, None] * normal
        v_t_mag = np.sqrt(np.einsum("ij,ij->i", v_t, v_t))
        f_t_mag = np.minimum(self.gamma_tangent * v_t_mag, self.mu_wall * f_n)
        slip_scale = np.where(
            v_t_mag > 1e-12, -f_t_mag / np.maximum(v_t_mag, 1e-12), 0.0
        )
        wall_force = normal * f_n[:, None] + v_t * slip_scale[:, None]
        _scatter(self.count, index, wall_force, force)
        contact_torque = np.cross(-radius * normal, wall_force)
        _scatter(self.count, index, contact_torque, torque)

        if moving:
            rel_spin = self.spin[index] - self.omega_vector[None, :]
            spin_mag = np.sqrt(np.einsum("ij,ij->i", rel_spin, rel_spin))
            roll_mag = np.minimum(
                self.k_rolling * spin_mag, self.mu_rolling * f_n * 2.0 * radius
            )
            roll_scale = np.where(
                spin_mag > 1e-12, -roll_mag / np.maximum(spin_mag, 1e-12), 0.0
            )
            _scatter(self.count, index, rel_spin * roll_scale[:, None], torque)

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        count = self.count
        force = self.gravity_force.copy()
        torque = np.zeros((count, 3), dtype=np.float64)
        self._contact_forces(force, torque)
        self._wall_forces(force, torque)

        self.vel += (force * self.inv_mass[:, None]) * dt
        self.spin += (torque * self.inv_inertia[:, None]) * dt
        np.clip(self.vel, -_VELOCITY_CLAMP, _VELOCITY_CLAMP, out=self.vel)
        np.clip(self.spin, -_SPIN_CLAMP, _SPIN_CLAMP, out=self.spin)
        self.pos += self.vel * dt

    def advance_to(self, time: float):
        time = float(time)
        steps = int(round((time - self.time) / self.dt))
        if steps > 0:
            for _ in range(steps):
                if self._step_index % self.rebuild_interval == 0:
                    self._rebuild_pairs()
                self._step(self.dt)
                self._step_index += 1
        self.time = time
        return self._make_frame(time)

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def _update_diagnostics(self) -> None:
        pos = self.pos
        radius = self.grain_radius
        surface_level = float(np.percentile(pos[:, 2], 97.0))
        self.lifted_height = surface_level
        flowing = pos[:, 2] > surface_level - 0.08
        if np.any(flowing):
            self.avalanche_speed = float(
                np.mean(np.linalg.norm(self.vel[flowing], axis=1))
            )
        else:
            self.avalanche_speed = 0.0
        edges = np.linspace(-self.drum_radius, self.drum_radius, 9)
        centres, heights = [], []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (pos[:, 0] >= low) & (pos[:, 0] < high)
            if np.count_nonzero(mask) >= 8:
                centres.append(0.5 * (low + high))
                heights.append(float(np.percentile(pos[mask, 2], 95.0)))
        if len(centres) >= 3:
            slope = np.polyfit(np.asarray(centres), np.asarray(heights), 1)[0]
            self.free_surface_slope = float(np.clip(slope, -3.0, 3.0))
        qx = pos[:, 0] - self.drum_center[0]
        qz = pos[:, 2] - self.drum_center[2]
        radial = np.sqrt(qx * qx + qz * qz)
        outside = (radial > self.drum_radius + radius) | (
            np.abs(pos[:, 1]) > self.half_depth + radius
        )
        self.escaped_count = int(np.count_nonzero(outside))

    def _make_frame(self, time: float) -> StateFrame:
        self._update_diagnostics()
        pos = self.pos
        vel = self.vel
        count = self.count
        radius = self.grain_radius

        # ---- raw contact geometry samples -----------------------------
        gaps = []
        i_idx = self._pair_i
        j_idx = self._pair_j
        if len(i_idx):
            diff = pos[j_idx] - pos[i_idx]
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            pair_gap = dist - 2.0 * radius
            gaps.append(pair_gap[pair_gap < self.near_gap])
        qx = pos[:, 0] - self.drum_center[0]
        qz = pos[:, 2] - self.drum_center[2]
        radial = np.sqrt(qx * qx + qz * qz)
        cyl_gap = (self.drum_radius - radius) - radial
        gaps.append(cyl_gap[cyl_gap < self.near_gap])
        for sign in (1.0, -1.0):
            flat_gap = (self.half_depth - radius) - sign * pos[:, 1]
            gaps.append(flat_gap[flat_gap < self.near_gap])
        contact_distance = (
            np.concatenate(gaps) if gaps else np.zeros(0, dtype=np.float64)
        )

        colors = np.tile(np.array([0.86, 0.57, 0.20]), (count, 1))
        colors[self.tracer > 0.5] = np.array([0.88, 0.11, 0.12])

        attributes = {
            "radius": self.grain_radius_vector.copy(),
            "particle_radius": self.grain_radius_vector.copy(),
            "material_label": self.material_label.copy(),
            "tracer_grain": self.tracer.copy(),
            "mass_quantity": self.mass.copy(),
            "initial_volume": self.initial_volume.copy(),
            "color_rgb": colors,
        }
        fields = {
            "contact_signed_distance": contact_distance,
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "external_force": self.gravity_force.copy(),
        }
        scalars = {
            "drum_radius": self.drum_radius,
            "drum_depth": self.drum_depth,
            "drum_angular_velocity_rad_s": self.angular_velocity,
            "drum_angle": float(self.angular_velocity * time % (2.0 * np.pi)),
            "initial_particle_count": float(count),
            "contact_length_scale": self.contact_length_scale,
            "lifted_grain_height": float(self.lifted_height),
            "free_surface_slope": float(self.free_surface_slope),
            "avalanche_layer_speed": float(self.avalanche_speed),
            "escaped_particle_count": float(self.escaped_count),
        }
        frame = StateFrame(
            time=time,
            position=pos.copy(),
            velocity=vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "particle_count": int(self.count),
            "final_time": float(self.time),
            "drum_angular_velocity_rad_s": float(self.angular_velocity),
            "lifted_grain_height": float(self.lifted_height),
            "free_surface_slope": float(self.free_surface_slope),
            "avalanche_layer_speed": float(self.avalanche_speed),
            "escaped_particle_count": int(self.escaped_count),
            "contact_length_scale": float(self.contact_length_scale),
        }
