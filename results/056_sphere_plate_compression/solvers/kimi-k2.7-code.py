"""Physically-based FEM solver for a soft incompressible sphere compressed between plates.

The solver advances a tetrahedral elastic solid using explicit time stepping.
The icosahedron surface vertices are connected to a fixed centre point through
20 tetrahedral elements (one per face).  A stable Neo-Hookean constitutive
model with a strong volumetric penalty approximates incompressibility, while
kinematic contact constraints move boundary vertices with the approaching
plates.  Only NumPy/SciPy are used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver for the sphere-plate compression benchmark."""

    def __init__(self) -> None:
        # Simulation state (filled in initialize)
        self.config: dict | None = None
        self.pos: NDArray[np.float64] | None = None
        self.vel: NDArray[np.float64] | None = None
        self.time: float = 0.0

        # Derived data
        self.faces: NDArray[np.int64] | None = None
        self.ref_invDm: list[NDArray[np.float64]] = []
        self.V0: float = 0.0
        self.mass: NDArray[np.float64] | None = None

        # Material / numerical parameters
        self.mu = 10.0
        self.lam = 1000.0
        self.damping = 0.5
        self.dt = 1e-4

        # Plate data
        self.plate_motion_end: float = 0.0
        self.plate_initial_lower: float = 0.0
        self.plate_initial_upper: float = 0.0
        self.plate_final_lower: float = 0.0
        self.plate_final_upper: float = 0.0
        self.plate_speed: float = 0.0

        # Budget / constraint bookkeeping
        self.cumulative_external_impulse = np.zeros(3, dtype=np.float64)
        self.cumulative_inflow = np.zeros(3, dtype=np.float64)
        self.cumulative_outflow = np.zeros(3, dtype=np.float64)
        self.rest_volume: float = 0.0

    def initialize(self, scene, seed: int) -> None:  # noqa: ARG002 -- seed kept for contract
        """Set up the solver from the provided benchmark scene."""
        self.config = scene.config
        cfg = self.config

        # Per-sample data from the scene
        sphere_cfg = cfg["sphere"]
        plates_cfg = cfg["plates"]

        rest_volume = float(sphere_cfg["rest_volume"])
        total_mass = float(sphere_cfg["mass"])
        vertices = np.asarray(scene._vertices, dtype=np.float64)  # noqa: SLF001
        n_vertices = len(vertices)

        self.rest_volume = rest_volume
        self.V0 = rest_volume / 20.0
        self.mass = np.full(n_vertices, total_mass / n_vertices, dtype=np.float64)

        # Load and orient the 20 icosahedron faces; the centre (origin) is the
        # fourth node of every tetrahedron.
        obj_path = Path(__file__).resolve().parent.parent / "assets" / "soft_sphere.obj"
        raw_faces = self._load_obj_faces(obj_path)
        oriented_faces: list[list[int]] = []
        self.ref_invDm = []
        for a, b, c in raw_faces:
            # Make sure the face normal points outward (positive dot with vertex).
            va, vb, vc = vertices[a], vertices[b], vertices[c]
            normal = np.cross(vb - va, vc - va)
            centroid = (va + vb + vc) / 3.0
            if np.dot(normal, centroid) < 0.0:
                b, c = c, b
                vb, vc = vc, vb
            oriented_faces.append([a, b, c])

            Dm = np.column_stack([vb - 0.0, vc - 0.0, va - 0.0])  # centre is origin
            # Use a, b, c as the three non-centre nodes in this order.
            Dm = np.column_stack([vertices[a] - 0.0, vertices[b] - 0.0, vertices[c] - 0.0])
            self.ref_invDm.append(np.linalg.inv(Dm))

        self.faces = np.asarray(oriented_faces, dtype=np.int64)

        self.pos = vertices.copy()
        self.vel = np.zeros_like(self.pos)
        self.time = 0.0

        # Plate motion parameters (linear approach from initial to final inner faces)
        self.plate_initial_lower = float(plates_cfg["initial_inner_faces_z"][0])
        self.plate_initial_upper = float(plates_cfg["initial_inner_faces_z"][1])
        self.plate_final_lower = float(plates_cfg["final_inner_faces_z"][0])
        self.plate_final_upper = float(plates_cfg["final_inner_faces_z"][1])
        self.plate_motion_end = float(plates_cfg["motion_end"])
        if self.plate_motion_end <= 0.0:
            self.plate_motion_end = 1.0
        self.plate_speed = (self.plate_final_lower - self.plate_initial_lower) / self.plate_motion_end

        # Reset budget accumulators
        self.cumulative_external_impulse = np.zeros(3, dtype=np.float64)
        self.cumulative_inflow = np.zeros(3, dtype=np.float64)
        self.cumulative_outflow = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _load_obj_faces(path: Path) -> NDArray[np.int64]:
        """Read 1-based face indices from an OBJ file and convert to 0-based."""
        faces = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("f "):
                parts = line.split()
                faces.append([int(parts[1].split("/")[0]), int(parts[2].split("/")[0]), int(parts[3].split("/")[0])])
        faces = np.asarray(faces, dtype=np.int64) - 1
        return faces

    def _plate_state(self, t: float) -> tuple[float, float, float, float]:
        """Return (lower_z, upper_z, lower_speed, upper_speed) at time t."""
        alpha = max(0.0, min(1.0, t / self.plate_motion_end))
        lower = self.plate_initial_lower + alpha * (self.plate_final_lower - self.plate_initial_lower)
        upper = self.plate_initial_upper + alpha * (self.plate_final_upper - self.plate_initial_upper)
        if t <= self.plate_motion_end:
            speed = self.plate_speed
        else:
            speed = 0.0
        return lower, upper, speed, -speed

    def _elastic_forces(self) -> NDArray[np.float64]:
        """Compute internal elastic forces on the surface vertices."""
        assert self.pos is not None
        assert self.faces is not None
        forces = np.zeros_like(self.pos)
        for (a, b, c), invDm in zip(self.faces, self.ref_invDm):
            Ds = np.column_stack([self.pos[a], self.pos[b], self.pos[c]])
            F = Ds @ invDm

            # Stable Neo-Hookean stress: P = mu*(F - F^-T) + lambda*log(J)*F^-T
            J = np.linalg.det(F)
            if J <= 1e-9:
                # Degenerate element: zero force to avoid explosion.
                continue
            invF_T = np.linalg.inv(F).T
            P = self.mu * (F - invF_T) + self.lam * np.log(J) * invF_T
            H = self.V0 * P @ invDm.T

            forces[a] -= H[:, 0]
            forces[b] -= H[:, 1]
            forces[c] -= H[:, 2]
        return forces

    def _substep(self) -> None:
        """Advance the physical state by one internal time step."""
        assert self.pos is not None
        assert self.vel is not None
        assert self.mass is not None

        # Elastic forces (centre node is fixed at origin and therefore omitted).
        f_elastic = self._elastic_forces()

        # Symplectic Euler update with velocity damping.
        f_total = f_elastic
        acceleration = f_total / self.mass[:, None] - self.damping * self.vel
        self.vel += self.dt * acceleration
        self.pos += self.dt * self.vel

        # Apply plate contact as hard kinematic constraints.
        t_next = self.time + self.dt
        lower_z, upper_z, lower_speed, upper_speed = self._plate_state(t_next)

        # Ensure the plates cannot cross.
        lower_z = min(lower_z, upper_z)
        upper_z = max(lower_z, upper_z)

        for i in range(self.pos.shape[0]):
            z = self.pos[i, 2]
            if z <= lower_z:
                self.pos[i, 2] = lower_z
                self.vel[i, 2] = max(self.vel[i, 2], lower_speed)
            if z >= upper_z:
                self.pos[i, 2] = upper_z
                self.vel[i, 2] = min(self.vel[i, 2], upper_speed)

        self.time = t_next

    def _compute_deformation_gradients(self) -> NDArray[np.float64]:
        """Return per-element deformation gradients, shape (Ne, 3, 3)."""
        assert self.pos is not None
        assert self.faces is not None
        F = np.zeros((len(self.faces), 3, 3), dtype=np.float64)
        for idx, ((a, b, c), invDm) in enumerate(zip(self.faces, self.ref_invDm)):
            Ds = np.column_stack([self.pos[a], self.pos[b], self.pos[c]])
            F[idx] = Ds @ invDm
        return F

    def advance_to(self, time: float) -> StateFrame:
        """Integrate the physical state forward to ``time`` and return a frame."""
        if time < self.time:
            time = self.time

        n_steps = int(round((time - self.time) / self.dt))
        for _ in range(n_steps):
            self._substep()

        return self._build_frame(time)

    def _build_frame(self, time: float) -> StateFrame:
        """Assemble a StateFrame at the current physical state."""
        assert self.pos is not None
        assert self.vel is not None
        assert self.mass is not None

        F = self._compute_deformation_gradients()
        J = np.linalg.det(F)
        current_volume = float(np.sum(J)) * self.V0

        n = self.pos.shape[0]
        orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (n, 1))
        angular_velocity = np.zeros((n, 3), dtype=np.float64)
        body_id = np.zeros(n, dtype=np.int64)

        # Domain / budget bookkeeping (one global domain, one external-force term)
        budget_domain_weights = np.ones((1, n), dtype=np.float64)
        budget_external_force = np.zeros((1, 1, 3), dtype=np.float64)

        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=body_id,
            attributes={
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "element_deformation_gradient": F,
                "budget_domain_weights": budget_domain_weights,
                "budget_external_force": budget_external_force,
                "budget_cumulative_external_impulse": self.cumulative_external_impulse.reshape(1, 3).copy(),
                "budget_cumulative_inflow_momentum": self.cumulative_inflow.reshape(1, 3).copy(),
                "budget_cumulative_outflow_momentum": self.cumulative_outflow.reshape(1, 3).copy(),
                "constraint_value": np.array([current_volume], dtype=np.float64),
                "constraint_target": np.array([self.rest_volume], dtype=np.float64),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            },
            scalars={
                "constraint_scale": float(self.rest_volume),
                "reference_momentum": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        """Return optional diagnostic information."""
        return {
            "dt": self.dt,
            "mu": self.mu,
            "lambda": self.lam,
            "damping": self.damping,
            "current_time": self.time,
        }
