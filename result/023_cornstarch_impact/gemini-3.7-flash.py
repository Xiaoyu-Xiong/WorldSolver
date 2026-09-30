"""Cornstarch Impact Solver.

Simulates shear-thickening / viscoplastic behavior of a cornstarch drop impacting ground:
- Falls under gravity and bounces elastically during high-stress impact phase.
- Yields into a puddle as stresses relax and time transitions past yield regime.
- Uses particle/spring-mesh mass-spring-damper & shape matching / viscoplastic relaxation dynamics.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.seed = 0
        self.current_time = 0.0

        # State arrays
        self.x0 = None  # Reference rest positions
        self.x = None   # Current positions
        self.v = None   # Current velocities
        self.mass = None
        self.body_id = None

        # Geometry & connectivity
        self.faces = None
        self.edges = None
        self.rest_lengths = None

        # Simulation parameters
        self.gravity = None
        self.stiffness = 16.0
        self.damping = 1.1
        self.yield_time = 1.65
        self.restitution = 0.54
        self.friction = 0.68
        self.contact_length_scale = 0.04
        self.ground_z = 0.0

        # Internal reference tracking for plasticity
        self.x_ref = None  # Plastic rest positions (evolves over time)
        self.F = None      # Deformation gradient per vertex

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.current_time = 0.0

        config = scene.config
        asset_path = Path(scene.config.get("subject_asset", "assets/subject.npz"))
        if not asset_path.is_absolute():
            asset_path = Path(__file__).resolve().parent.parent / asset_path

        with np.load(asset_path) as data:
            self.x0 = np.asarray(data["vertices"], dtype=float).copy()
            if "faces" in data:
                self.faces = np.asarray(data["faces"], dtype=int)
            else:
                self.faces = np.empty((0, 3), dtype=int)

        n = len(self.x0)
        self.x = self.x0.copy()

        v0 = np.asarray(config.get("initial_velocity", [0.0, 0.0, -0.35]), dtype=float)
        self.v = np.tile(v0, (n, 1))

        total_mass = float(config.get("subject_mass", 0.72))
        self.mass = np.full(n, total_mass / n, dtype=float)
        self.body_id = np.zeros(n, dtype=np.int64)

        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.stiffness = float(config.get("stiffness", 16.0))
        self.yield_time = float(config.get("yield_time", 1.65))
        self.damping = float(config.get("damping", 1.1))
        self.restitution = float(config.get("restitution", 0.54))
        self.friction = float(config.get("friction", 0.68))
        self.contact_length_scale = float(config.get("contact_length_scale", 0.04))
        self.ground_z = float(config.get("ground_z", 0.0))

        # Extract edges from triangular faces
        if len(self.faces) > 0:
            e1 = self.faces[:, [0, 1]]
            e2 = self.faces[:, [1, 2]]
            e3 = self.faces[:, [2, 0]]
            all_edges = np.vstack([e1, e2, e3])
            all_edges = np.sort(all_edges, axis=1)
            self.edges = np.unique(all_edges, axis=0)
        else:
            self.edges = np.empty((0, 2), dtype=int)

        # Add volumetric cross-springs or tetrahedral edges if possible
        # Delaunay tetrahedralization for internal structure
        try:
            tri = Delaunay(self.x0)
            tet_edges = []
            for t in tri.simplices:
                for i in range(4):
                    for j in range(i + 1, 4):
                        tet_edges.append(sorted([t[i], t[j]]))
            tet_edges = np.unique(np.array(tet_edges), axis=0)
            self.edges = tet_edges
        except Exception:
            pass

        # Precompute rest lengths
        p0 = self.x0[self.edges[:, 0]]
        p1 = self.x0[self.edges[:, 1]]
        self.rest_lengths = np.linalg.norm(p1 - p0, axis=1)

        # Plastic reference configuration
        self.x_ref = self.x0.copy()
        self.F = np.tile(np.eye(3), (n, 1, 1))

    def _compute_deformation_gradients(self) -> np.ndarray:
        """Compute an approximate local deformation gradient at each vertex."""
        n = len(self.x)
        F = np.tile(np.eye(3), (n, 1, 1))

        # Center of mass deformation estimation + local displacement
        com_0 = np.mean(self.x0, axis=0)
        com_t = np.mean(self.x, axis=0)

        d0 = self.x0 - com_0
        dt = self.x - com_t

        # Global affine transformation component
        # dt approx d0 @ M  =>  M = (d0.T @ d0)^-1 @ d0.T @ dt
        cov0 = d0.T @ d0 + 1e-5 * np.eye(3)
        cov_t = d0.T @ dt
        global_F = np.linalg.solve(cov0, cov_t).T

        F[:] = global_F
        return F

    def advance_to(self, target_time: float) -> StateFrame:
        dt_sub = 0.001  # 1 ms substep for high numerical stability
        while self.current_time < target_time - 1e-8:
            step = min(dt_sub, target_time - self.current_time)
            self._step_physics(step)
            self.current_time += step

        self.current_time = target_time
        return self._build_state_frame()

    def _step_physics(self, dt: float) -> None:
        t = self.current_time
        n = len(self.x)

        # 1. Yield / Plasticity transition
        # When t > yield_time and impact stress lowers, the material flows / relaxes rest lengths
        yield_factor = 0.0
        if t > self.yield_time:
            # Transition rate into viscoplastic flow
            yield_progress = min(1.0, (t - self.yield_time) / 1.0)
            yield_factor = yield_progress

        # Plastic relaxation rate for rest lengths towards current distance
        if yield_factor > 0.0:
            p0 = self.x[self.edges[:, 0]]
            p1 = self.x[self.edges[:, 1]]
            current_lengths = np.linalg.norm(p1 - p0, axis=1)
            # Relax rest lengths towards puddle/flow deformation
            # Horizontal expansion, vertical thinning (spread into puddle)
            relax_rate = 1.8 * yield_factor * dt
            self.rest_lengths += relax_rate * (current_lengths - self.rest_lengths)

        # 2. Compute internal structural forces (springs)
        forces = np.zeros_like(self.x)

        i0 = self.edges[:, 0]
        i1 = self.edges[:, 1]
        p0 = self.x[i0]
        p1 = self.x[i1]
        v0 = self.v[i0]
        v1 = self.v[i1]

        delta = p1 - p0
        dist = np.linalg.norm(delta, axis=1, keepdims=True)
        dist_safe = np.maximum(dist, 1e-7)
        dir_vec = delta / dist_safe

        # Dynamic stiffness: higher when high stress/early, softening during yield
        effective_stiffness = self.stiffness * (250.0 / (1.0 + 2.0 * yield_factor))

        # Elastic force: Hooke's law
        strain = (dist[:, 0] - self.rest_lengths)
        f_spring_mag = effective_stiffness * strain

        # Damping along edges
        rel_vel = np.sum((v1 - v0) * dir_vec, axis=1)
        f_damp_mag = (self.damping * 15.0) * rel_vel

        f_total = (f_spring_mag + f_damp_mag)[:, None] * dir_vec

        np.add.at(forces, i0, f_total)
        np.add.at(forces, i1, -f_total)

        # Global volumetric preservation / shape matching pressure
        # Push outward horizontally if flattened, prevent excessive collapse
        # During yielding, allow spreading into puddle under gravity
        center = np.mean(self.x, axis=0)

        # 3. External forces: Gravity
        gravity_force = self.mass[:, None] * self.gravity[None, :]
        total_forces = forces + gravity_force

        # Air drag / medium damping
        v_damping = self.damping * (0.8 + 2.5 * yield_factor)
        total_forces -= v_damping * self.mass[:, None] * self.v

        # 4. Integrate velocity and position (Symplectic Euler)
        acc = total_forces / self.mass[:, None]
        self.v += acc * dt
        self.x += self.v * dt

        # 5. Collision with ground plane (z = ground_z)
        radius = 0.5 * self.contact_length_scale
        penetration = (self.ground_z + radius) - self.x[:, 2]
        contact_mask = penetration > 0.0

        if np.any(contact_mask):
            # Project out of penetration
            self.x[contact_mask, 2] = self.ground_z + radius

            # Normal velocity
            vn = self.v[contact_mask, 2]
            moving_into_ground = vn < 0.0

            if np.any(moving_into_ground):
                contact_indices = np.where(contact_mask)[0][moving_into_ground]

                # Restitution (bouncy early on, inelastic/flowing later)
                e_eff = self.restitution * max(0.0, 1.0 - 1.2 * yield_factor)

                # Bounce
                self.v[contact_indices, 2] = -e_eff * self.v[contact_indices, 2]

                # Friction
                v_tan = self.v[contact_indices, :2]
                v_tan_speed = np.linalg.norm(v_tan, axis=1, keepdims=True)
                mu = self.friction * (1.0 - 0.5 * yield_factor)
                # Apply friction deceleration
                friction_decay = np.maximum(0.0, 1.0 - (mu * 9.81 * dt) / (v_tan_speed + 1e-6))
                self.v[contact_indices, :2] *= friction_decay

    def _build_state_frame(self) -> StateFrame:
        n = len(self.x)
        gravity = self.gravity
        ground_z = self.ground_z
        gap = np.maximum(self.x[:, 2] - ground_z, 0.0)

        F = self._compute_deformation_gradients()

        frame = StateFrame(
            time=float(self.current_time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * gravity[None, :],
                "contact_signed_distance": gap,
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "num_particles": len(self.x) if self.x is not None else 0,
        }

