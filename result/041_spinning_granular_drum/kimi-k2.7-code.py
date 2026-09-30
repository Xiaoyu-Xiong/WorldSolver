"""DEM-style solver for a half-filled rotating granular drum.

The drum rotates about the y-axis.  Grains feel gravity, particle-particle
contact, and contact with the rotating cylindrical wall and end caps.  The
integration is explicit (semi-implicit Euler) with a fixed sub-step chosen to
resolve the contact time scale.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Physically step a rotating drum filled with soft spherical grains."""

    def __init__(self) -> None:
        pass

    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # Load the authoritative scaffold configuration.
        config_path = Path(__file__).parent.parent / "config.yaml"
        if config_path.exists():
            import yaml

            with open(config_path, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh)
        else:
            config = {}

        ic = config.get("initial_conditions", {})
        drum = ic.get("drum", {})

        self.gravity = np.array(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64
        )
        self.drum_center = np.array([0.0, 0.0, 0.86], dtype=np.float64)
        self.drum_radius = float(drum.get("radius", 0.82))
        self.drum_depth = float(drum.get("depth", 0.42))
        self.omega = float(drum.get("angular_velocity_rad_s", 1.25))

        initial = scene.initial_state()
        self.time = 0.0
        self.position = np.asarray(initial.position, dtype=np.float64).copy()
        self.velocity = np.asarray(initial.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(initial.mass, dtype=np.float64).copy()
        self.body_id = (
            np.asarray(initial.body_id, dtype=np.int32).copy()
            if initial.body_id is not None
            else None
        )
        self.attributes = {
            k: np.asarray(v).copy() for k, v in initial.attributes.items()
        }
        self.initial_scalars = {
            k: float(v) for k, v in initial.scalars.items()
        }
        self.n = self.position.shape[0]

        self.particle_radius = float(self.attributes["radius"][0])
        self.particle_volume = float(self.attributes["initial_volume"][0])

        # Contact parameters.  Stiffness is chosen soft enough to allow a stable
        # 5e-4 s sub-step, yet stiff enough to prevent visible interpenetration.
        m_avg = float(np.mean(self.mass))
        self.k_n = 1.0e5  # normal stiffness [N/m]
        damping_ratio = 0.3
        self.c_n = damping_ratio * 2.0 * np.sqrt(m_avg * self.k_n)
        self.c_t = 2.0 * self.c_n  # viscous regularisation for tangential friction
        self.mu_pp = 0.5  # particle-particle friction coefficient
        self.mu_pw = 0.7  # particle-wall friction coefficient

        # Fixed physics sub-step.
        self.dt = 5.0e-4

        # End-cap locations (particle-center bounds).
        self._cap_top_y = 0.5 * self.drum_depth - self.particle_radius
        self._cap_bottom_y = -0.5 * self.drum_depth + self.particle_radius

        # Storage for contact signed distances from the most recent sub-step.
        self._contact_signed = self._compute_signed_distances()

    # --------------------------------------------------------------------- #
    # Kinematic helpers
    # --------------------------------------------------------------------- #
    def _wall_velocity(self, positions: np.ndarray) -> np.ndarray:
        """Tangential velocity of the rigid rotating drum at ``positions``."""
        x = positions[:, 0]
        z = positions[:, 2]
        v = np.zeros_like(positions)
        v[:, 0] = self.omega * (z - self.drum_center[2])
        v[:, 2] = -self.omega * x
        return v

    # --------------------------------------------------------------------- #
    # Contact detection / signed distances
    # --------------------------------------------------------------------- #
    def _compute_signed_distances(self) -> np.ndarray:
        """Return signed gaps for active and near contacts (negative = overlap)."""
        tree = cKDTree(self.position)
        pairs = tree.query_pairs(
            3.0 * self.particle_radius, output_type="ndarray"
        )
        if pairs.size > 0:
            dx = self.position[pairs[:, 1]] - self.position[pairs[:, 0]]
            d = np.sqrt(np.sum(dx * dx, axis=1))
            pp_signed = d - 2.0 * self.particle_radius
        else:
            pp_signed = np.array([], dtype=np.float64)

        # Cylindrical wall.
        r_vec = self.position[:, [0, 2]] - self.drum_center[[0, 2]]
        r_mag = np.sqrt(np.sum(r_vec * r_vec, axis=1))
        gap_cyl = self.drum_radius - r_mag - self.particle_radius
        cyl_signed = gap_cyl[gap_cyl < self.particle_radius]

        # End caps.
        gap_top = self._cap_top_y - self.position[:, 1]
        top_signed = gap_top[gap_top < self.particle_radius]
        gap_bottom = self.position[:, 1] - self._cap_bottom_y
        bottom_signed = gap_bottom[gap_bottom < self.particle_radius]

        return np.concatenate([pp_signed, cyl_signed, top_signed, bottom_signed])

    # --------------------------------------------------------------------- #
    # Time stepping
    # --------------------------------------------------------------------- #
    def _add_pp_contacts(self, force: np.ndarray) -> None:
        """Accumulate equal-and-opposite particle-particle contact forces."""
        tree = cKDTree(self.position)
        pairs = tree.query_pairs(
            2.0 * self.particle_radius, output_type="ndarray"
        )
        if pairs.size == 0:
            return

        i = pairs[:, 0]
        j = pairs[:, 1]
        dx = self.position[j] - self.position[i]
        d = np.sqrt(np.sum(dx * dx, axis=1))
        n = dx / d[:, None]

        v_rel = self.velocity[j] - self.velocity[i]
        v_n = np.sum(v_rel * n, axis=1)
        overlap = 2.0 * self.particle_radius - d
        f_n_mag = np.maximum(0.0, self.k_n * overlap - self.c_n * v_n)

        active = overlap > 0.0
        if not np.any(active):
            return

        f_n = f_n_mag[active][:, None] * n[active]
        v_t = v_rel[active] - v_n[active][:, None] * n[active]
        v_t_norm = np.linalg.norm(v_t, axis=1)
        f_t_mag = np.minimum(self.mu_pp * f_n_mag[active], self.c_t * v_t_norm)

        t = np.zeros_like(v_t)
        mask = v_t_norm > 1e-8
        t[mask] = -v_t[mask] / v_t_norm[mask][:, None]
        f_t = f_t_mag[:, None] * t

        f_total = f_n + f_t
        np.add.at(force, i[active], f_total)
        np.add.at(force, j[active], -f_total)

    def _add_wall_contacts(self, force: np.ndarray) -> None:
        """Accumulate contact forces from the rotating cylindrical drum wall."""
        r_vec = self.position[:, [0, 2]] - self.drum_center[[0, 2]]
        r_mag = np.sqrt(np.sum(r_vec * r_vec, axis=1))

        n_cyl = np.zeros((self.n, 3), dtype=np.float64)
        n_cyl[:, 0] = -r_vec[:, 0] / r_mag
        n_cyl[:, 2] = -r_vec[:, 1] / r_mag

        gap_cyl = self.drum_radius - r_mag - self.particle_radius
        mask = gap_cyl < 0.0
        if not np.any(mask):
            return

        idx = np.nonzero(mask)[0]
        n = n_cyl[idx]
        v_w = self._wall_velocity(self.position[idx])
        v_rel = self.velocity[idx] - v_w
        v_n = np.sum(v_rel * n, axis=1)
        overlap = -gap_cyl[idx]
        f_n_mag = np.maximum(0.0, self.k_n * overlap - self.c_n * v_n)

        active = overlap > 0.0
        if not np.any(active):
            return

        f_n = f_n_mag[active][:, None] * n[active]
        v_t = v_rel[active] - v_n[active][:, None] * n[active]
        v_t_norm = np.linalg.norm(v_t, axis=1)
        f_t_mag = np.minimum(self.mu_pw * f_n_mag[active], self.c_t * v_t_norm)

        t = np.zeros_like(v_t)
        m = v_t_norm > 1e-8
        t[m] = -v_t[m] / v_t_norm[m][:, None]
        f_t = f_t_mag[:, None] * t

        f_total = f_n + f_t
        np.add.at(force, idx[active], f_total)

    def _add_cap_contacts(self, force: np.ndarray, cap_normal: np.ndarray) -> None:
        """Accumulate contact forces from one end cap of the drum."""
        if np.allclose(cap_normal, (0.0, 1.0, 0.0)):
            gap = self.position[:, 1] - self._cap_bottom_y
        else:
            gap = self._cap_top_y - self.position[:, 1]

        mask = gap < 0.0
        if not np.any(mask):
            return

        idx = np.nonzero(mask)[0]
        n = np.zeros((len(idx), 3), dtype=np.float64)
        n[:, 1] = cap_normal[1]

        v_w = self._wall_velocity(self.position[idx])
        v_rel = self.velocity[idx] - v_w
        v_n = np.sum(v_rel * n, axis=1)
        overlap = -gap[idx]
        f_n_mag = np.maximum(0.0, self.k_n * overlap - self.c_n * v_n)

        active = overlap > 0.0
        if not np.any(active):
            return

        f_n = f_n_mag[active][:, None] * n[active]
        v_t = v_rel[active] - v_n[active][:, None] * n[active]
        v_t_norm = np.linalg.norm(v_t, axis=1)
        f_t_mag = np.minimum(self.mu_pw * f_n_mag[active], self.c_t * v_t_norm)

        t = np.zeros_like(v_t)
        m = v_t_norm > 1e-8
        t[m] = -v_t[m] / v_t_norm[m][:, None]
        f_t = f_t_mag[:, None] * t

        f_total = f_n + f_t
        np.add.at(force, idx[active], f_total)

    def _step(self, dt: float) -> None:
        """Advance one physical sub-step."""
        force = np.zeros((self.n, 3), dtype=np.float64)

        self._add_pp_contacts(force)
        self._add_wall_contacts(force)
        self._add_cap_contacts(force, (0.0, -1.0, 0.0))  # top cap
        self._add_cap_contacts(force, (0.0, 1.0, 0.0))  # bottom cap

        # Gravity.
        force += self.mass[:, None] * self.gravity

        # Semi-implicit Euler integration.
        self.velocity += (force / self.mass[:, None]) * dt
        self.position += self.velocity * dt

        # Recompute signed-distance field for the next sampled frame.
        self._contact_signed = self._compute_signed_distances()

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def _current_frame(self) -> StateFrame:
        attributes = dict(self.attributes)
        attributes["mass_quantity"] = self.mass.copy()

        scalars = dict(self.initial_scalars)
        scalars["contact_length_scale"] = 2.0 * self.particle_radius

        fields = {
            "contact_signed_distance": self._contact_signed.copy(),
            "cumulative_mass_source": np.array([0.0], dtype=np.float64),
            "external_force": self.mass[:, None] * self.gravity,
        }

        return StateFrame(
            time=self.time,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy() if self.body_id is not None else None,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def advance_to(self, time: float) -> StateFrame:
        if time <= self.time:
            return self._current_frame()

        n_steps = int(round((time - self.time) / self.dt))
        for _ in range(n_steps):
            self._step(self.dt)
            self.time += self.dt

        # Snap to the requested sample time to avoid floating-point drift.
        self.time = time
        return self._current_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "particle_count": self.n,
            "dt": self.dt,
            "time": self.time,
            "contact_count": int(self._contact_signed.size),
        }
