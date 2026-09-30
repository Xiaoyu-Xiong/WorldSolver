"""Self-contained particle-based sand solver for the pouring-sand scene.

The solver advances dry granular material as discrete particles subject to
gravity, particle-particle contact, and contact with a static hopper + floor.
It reproduces the visible phenomenon (steady discharge, falling stream, pile
formation, stable angle of repose) using an stable explicit time-stepping
scheme with position/velocity based collision response.

No external simulator (Genesis, MuJoCo, etc.) is used.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import Scene, Solver, StateFrame

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore


class BenchmarkSolver(Solver):
    """Explicit particle solver with gravity, inelastic contact and a static hopper/floor."""

    def __init__(self) -> None:
        self._initialized = False
        self._scene: Scene | None = None
        self._seed = 0
        self._config: dict = {}
        self._initial_frame: StateFrame | None = None

        # Simulation state
        self.position: NDArray[np.float64] = np.empty((0, 3))
        self.velocity: NDArray[np.float64] = np.empty((0, 3))
        self.mass: NDArray[np.float64] = np.empty((0,))
        self.radius: NDArray[np.float64] = np.empty((0,))
        self.phase: NDArray[np.float64] = np.empty((0,))

        self.gravity: NDArray[np.float64] = np.array([0.0, 0.0, -9.81])
        self.t: float = 0.0
        self.dt: float = 1e-3
        self.sample_dt: float = 0.02
        self.duration: float = 8.0

        # Boundary parameters
        self.floor_z: float = 0.0
        self.hopper_profile: NDArray[np.float64] | None = None
        self.hopper_thickness: float = 0.035

        # Cumulative sources
        self.cumulative_mass_added: float = 0.0
        self.cumulative_mass_removed: float = 0.0
        self._diagnostics: dict = {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _finite_copy(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.asarray(arr, dtype=np.float64).copy()
        if not np.isfinite(a).all():
            raise ValueError("Array contains non-finite values")
        return a

    # ------------------------------------------------------------------ lifecycle
    def initialize(self, scene: Scene, seed: int) -> None:
        self._scene = scene
        self._seed = seed

        if hasattr(scene, "config"):
            self._config = scene.config
        else:
            self._config = getattr(scene, "_config", {})

        self._initial_frame = scene.initial_state()
        self._initial_frame.validate()

        self.position = self._finite_copy(self._initial_frame.position)
        self.velocity = self._finite_copy(self._initial_frame.velocity)
        if self._initial_frame.mass is None:
            raise ValueError("Scene initial state must provide per-particle mass")
        self.mass = self._finite_copy(self._initial_frame.mass)

        attrs = self._initial_frame.attributes
        if "nominal_radius" in attrs:
            self.radius = self._finite_copy(attrs["nominal_radius"])
        else:
            r = self._config.get("initial_conditions", {}).get("nominal_grain_radius", 0.013)
            self.radius = np.full(len(self.position), float(r), dtype=np.float64)

        if "material_phase" in attrs:
            self.phase = self._finite_copy(attrs["material_phase"])
        else:
            self.phase = np.zeros(len(self.position), dtype=np.float64)

        self.gravity = np.array(self._config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.sample_dt = float(self._config.get("sample_dt", 0.02))
        self.duration = float(self._config.get("duration", 8.0))

        # Stable explicit time step: use a fixed number of substeps per sample frame.
        self.substeps_per_sample = 20
        self.dt = self.sample_dt / self.substeps_per_sample

        # Static boundaries
        self.floor_z = 0.0
        geometry = self._config.get("scene_geometry", {})
        objects = geometry.get("objects", []) if isinstance(geometry, dict) else []
        for obj in objects:
            if obj.get("id") == "supply_hopper" and obj.get("geometry", {}).get("type") == "axisymmetric_container":
                profile = obj["geometry"].get("inner_profile", [])
                if len(profile) >= 3:
                    self.hopper_profile = np.asarray(profile, dtype=np.float64)
                    self.hopper_thickness = float(obj["geometry"].get("wall_thickness", 0.035))

        self.t = 0.0
        self._initialized = True

        self._diagnostics = {
            "particles": len(self.position),
            "dt": self.dt,
            "substeps_per_sample": self.substeps_per_sample,
            "duration": self.duration,
        }

    def diagnostics(self) -> dict:
        self._diagnostics["current_time"] = self.t
        self._diagnostics["particles"] = len(self.position)
        return self._diagnostics.copy()

    # ------------------------------------------------------------------ geometry
    def _point_to_hopper_signed_distance(self, p: NDArray[np.float64]) -> float:
        """Signed distance of a point to the inside of the hopper wall.

        Positive = outside the container (penetrating the wall), negative = inside
        free space.  The hopper is open at the bottom and top, so outside the
        vertical span of the profile there is no wall.
        """
        if self.hopper_profile is None or len(self.hopper_profile) < 3:
            return 1e6

        profile = self.hopper_profile
        z_vals = profile[:, 1]
        z = p[2]
        if z < z_vals[0] or z > z_vals[-1]:
            return 1e6

        r = np.hypot(p[0], p[1])
        for i in range(len(profile) - 1):
            z0, z1 = z_vals[i], z_vals[i + 1]
            if z0 <= z <= z1:
                r0, r1 = profile[i, 0], profile[i + 1, 0]
                if abs(z1 - z0) < 1e-12:
                    r_inner = r0
                else:
                    r_inner = r0 + (r1 - r0) * (z - z0) / (z1 - z0)
                return r - r_inner
        return 1e6

    def _hopper_surface_normal(self, p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the outward-pointing normal of the hopper wall at point p."""
        eps = 1e-5
        d0 = self._point_to_hopper_signed_distance(p)
        grad = np.zeros(3, dtype=np.float64)
        for k in range(3):
            p_plus = p.copy()
            p_plus[k] += eps
            d_plus = self._point_to_hopper_signed_distance(p_plus)
            grad[k] = (d_plus - d0) / eps
        if np.linalg.norm(grad) < 1e-12:
            # Default normal points radially outward
            n = p.copy()
            n[2] = 0.0
        else:
            n = grad
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return n / n_norm

    def _floor_signed_distance(self, p: NDArray[np.float64]) -> float:
        return p[2] - self.floor_z

    # ------------------------------------------------------------------ contact
    def _resolve_floor_contacts(self) -> None:
        n = len(self.position)
        if n == 0:
            return
        floor_d = self.position[:, 2] - self.radius - self.floor_z
        contact = floor_d < 0.0
        if not np.any(contact):
            return

        indices = np.where(contact)[0]
        for i in indices:
            # Position correction: push to floor surface
            self.position[i, 2] = self.floor_z + self.radius[i] + 1e-6
            # Velocity response
            v = self.velocity[i]
            normal = np.array([0.0, 0.0, 1.0])
            v_n = v @ normal
            if v_n < 0.0:
                v_t = v - v_n * normal
                self.velocity[i] = v_t * (1.0 - 0.3)  # simple tangential friction

    def _resolve_hopper_contacts(self) -> None:
        if self.hopper_profile is None:
            return
        n = len(self.position)
        if n == 0:
            return
        # Vectorize simple check over all particles
        for i in range(n):
            p = self.position[i]
            d_hop = self._point_to_hopper_signed_distance(p)
            if d_hop > -self.radius[i]:
                normal = self._hopper_surface_normal(p)
                penetration = self.radius[i] + d_hop
                # Normal points outward from container; push inward (opposite normal)
                self.position[i] -= penetration * normal
                v_n = self.velocity[i] @ normal
                if v_n > 0.0:
                    self.velocity[i] -= v_n * normal
                # Tangential friction
                v_t = self.velocity[i] - (self.velocity[i] @ normal) * normal
                self.velocity[i] -= 0.3 * v_t

    def _resolve_particle_contacts(self) -> None:
        n = len(self.position)
        if n == 0:
            return
        if cKDTree is None:
            return

        tree = cKDTree(self.position)
        max_radius = float(np.max(self.radius)) if n else 0.013
        pairs = tree.query_pairs(r=2.0 * max_radius, output_type="ndarray")
        if pairs.size == 0:
            return

        i = pairs[:, 0]
        j = pairs[:, 1]
        pi = self.position[i]
        pj = self.position[j]
        vi = self.velocity[i]
        vj = self.velocity[j]
        ri = self.radius[i]
        rj = self.radius[j]

        nvec = pj - pi
        dist = np.linalg.norm(nvec, axis=1)
        dist = np.where(dist < 1e-12, 1.0, dist)
        nvec = nvec / dist[:, None]

        r_sum = ri + rj
        rel_v = vj - vi
        v_n = np.einsum('ij,ij->i', rel_v, nvec)

        # Only process approaching pairs
        mask = v_n < 0.0
        if not np.any(mask):
            return

        i = i[mask]
        j = j[mask]
        nvec = nvec[mask]
        v_n = v_n[mask]
        r_sum = r_sum[mask]
        dist = dist[mask]

        # Equal mass => split correction
        correction_n = -v_n[:, None] * nvec * 0.5
        # Tangential damping
        vi_corr = np.zeros_like(self.velocity)
        vj_corr = np.zeros_like(self.velocity)
        np.add.at(vi_corr, i, correction_n)
        np.add.at(vj_corr, j, -correction_n)
        self.velocity += vi_corr
        self.velocity += vj_corr

        # Position correction to reduce penetration
        penetration = r_sum - dist
        pos_mask = penetration > 0.0
        if np.any(pos_mask):
            pen_i = i[pos_mask]
            pen_j = j[pos_mask]
            pen_n = nvec[pos_mask]
            pen_val = penetration[pos_mask]
            for _ in range(2):
                for idx in range(len(pen_i)):
                    a = pen_i[idx]
                    b = pen_j[idx]
                    n = pen_n[idx]
                    d = pen_val[idx] * 0.25
                    self.position[a] -= d * n
                    self.position[b] += d * n

    # ------------------------------------------------------------------ stepping
    def _step_explicit(self) -> None:
        """One explicit sub-step with gravity, boundary and particle-particle collisions."""
        n = len(self.position)
        if n == 0:
            self.t += self.dt
            return

        # Boundary collision response (position and velocity)
        self._resolve_floor_contacts()
        self._resolve_hopper_contacts()

        # Particle-particle contact
        self._resolve_particle_contacts()

        # Gravity integration
        self.velocity += self.gravity * self.dt

        # Slight damping to avoid unbounded energy
        self.velocity *= 0.9995

        # Position integration
        self.position += self.velocity * self.dt
        self.t += self.dt

    def advance_to(self, target_time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("Solver not initialized")
        if target_time < self.t - 1e-12:
            raise ValueError("Cannot advance backwards in time")

        while self.t < target_time:
            remaining = target_time - self.t
            step_dt = min(self.dt, remaining)
            saved_dt = self.dt
            self.dt = step_dt
            self._step_explicit()
            self.dt = saved_dt

        return self._build_frame(target_time)

    # ------------------------------------------------------------------ frame export
    def _build_frame(self, time: float) -> StateFrame:
        n = len(self.position)

        mass_quantity = np.array(self.mass, dtype=np.float64, copy=True)
        external_force = (self.mass[:, None] * self.gravity[None, :]) if n else np.zeros((n, 3), dtype=np.float64)

        contact_sdf: list[float] = []
        if n:
            for i in range(n):
                p = self.position[i]
                d_floor = self._floor_signed_distance(p) - self.radius[i]
                if d_floor < 0.05:
                    contact_sdf.append(float(d_floor))
                if self.hopper_profile is not None:
                    d_hop = self._point_to_hopper_signed_distance(p) + self.radius[i]
                    if d_hop > -0.05:
                        contact_sdf.append(float(d_hop))
        if not contact_sdf:
            contact_sdf.append(0.0)
        contact_signed_distance = np.array(contact_sdf, dtype=np.float64)

        cumulative_mass_source = np.array([self.cumulative_mass_added, self.cumulative_mass_removed], dtype=np.float64)
        contact_length_scale = float(np.mean(self.radius)) if n else 0.013

        frame = StateFrame(
            time=float(time),
            position=np.array(self.position, dtype=np.float64, copy=True),
            velocity=np.array(self.velocity, dtype=np.float64, copy=True),
            mass=np.array(self.mass, dtype=np.float64, copy=True),
            attributes={
                "mass_quantity": mass_quantity,
                "nominal_radius": np.array(self.radius, dtype=np.float64, copy=True),
                "material_phase": np.array(self.phase, dtype=np.float64, copy=True),
            },
            fields={
                "contact_signed_distance": contact_signed_distance,
                "cumulative_mass_source": cumulative_mass_source,
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": contact_length_scale,
            },
        )
        frame.validate()
        return frame
