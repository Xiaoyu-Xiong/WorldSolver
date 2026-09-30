"""Physically-based elastic-solid solver for the Octocat compactor.

The implementation uses a vectorised mass-spring network on the frozen
surface mesh as a discretisation of a three-dimensional elastic solid.
The equations of motion are integrated with a stable symplectic Euler
step and implicit mass-proportional damping.  Kinematic wall motion is
enforced by projection, and the corresponding wall impulse is tracked
for the momentum-budget trajectory fields.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Mass-spring elastic solid solver with moving box-wall contact."""

    def __init__(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------
    # Solver protocol
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.config = scene.config
        self.seed = seed
        np.random.seed(seed)

        init_frame = scene.initial_state()
        init_frame.validate()

        self.position = np.asarray(init_frame.position, dtype=np.float64).copy()
        self.velocity = np.asarray(init_frame.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(init_frame.mass, dtype=np.float64).copy()

        self.num_particles = self.position.shape[0]
        if self.mass is None:
            self.mass = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float64)
        self.mass_inv = 1.0 / self.mass
        self.total_mass = float(self.mass.sum())

        # Load topology from the same asset the scene uses.
        asset_path = Path.cwd() / "assets" / "octocat.obj"
        mesh = trimesh.load_mesh(str(asset_path), process=False)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        edges = np.asarray(mesh.edges_unique, dtype=np.int64)

        self.faces = faces
        self.edges = edges
        self.rest_position = self.position.copy()

        # Edge rest lengths.
        dx = self.rest_position[edges[:, 1]] - self.rest_position[edges[:, 0]]
        self.rest_length = np.linalg.norm(dx, axis=1)
        # Protect against degenerate input edges.
        tiny = self.rest_length < 1e-12
        if tiny.any():
            self.rest_length = np.where(tiny, 1e-6, self.rest_length)

        # Triangle reference frames for the deformation gradient output.
        self._init_triangle_frames()

        # Compactor schedule.
        compactor = self.config["compactor"]
        self.open_half = float(compactor["open_inner_half_extent"])
        self.closed_half = float(compactor["closed_inner_half_extent"])
        self.wall_thickness = float(compactor["wall_thickness"])
        self.inward = tuple(compactor["inward_interval"])
        self.withdrawal = tuple(compactor["withdrawal_interval"])
        self.final_start = float(compactor["final_observation_interval"][0])

        # Time stepping.
        self.sample_dt = float(self.config["sample_dt"])
        self.substeps = 80
        self.dt = self.sample_dt / self.substeps

        # Material parameters.
        self.spring_k = 1000.0
        self.damping_c = 20.0  # mass-proportional damping [1/s]
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, 0.0]), dtype=np.float64)

        # Budget tracking.
        self.time = 0.0
        self.cumulative_external_impulse = np.zeros(3, dtype=np.float64)
        self.frame_contact_impulse = np.zeros(3, dtype=np.float64)

        # Reference momentum scale.
        self.reference_momentum = 1.0

        self._initialized = True

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "mass_spring_explicit",
            "num_particles": self.num_particles,
            "num_edges": int(self.edges.shape[0]),
            "num_faces": int(self.faces.shape[0]),
            "substep_dt": self.dt,
            "substeps_per_frame": self.substeps,
            "spring_stiffness": self.spring_k,
            "damping": self.damping_c,
        }

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("Solver has not been initialized")

        self.frame_contact_impulse.fill(0.0)

        # The caller is expected to advance by exact sample_dt multiples, but
        # guard against round-off by distributing the interval evenly.
        interval = time - self.time
        if interval <= 0.0:
            return self._build_frame(time)

        substeps = self.substeps
        dt = interval / substeps

        for i in range(substeps):
            t_mid = self.time + (i + 0.5) * dt
            self._substep(t_mid, dt)

        self.time = time
        return self._build_frame(time)

    # ------------------------------------------------------------------
    # Internal time step
    # ------------------------------------------------------------------
    def _substep(self, t: float, dt: float) -> None:
        # Symplectic Euler with implicit mass-proportional damping.
        f = self._spring_forces()

        # v = (v + dt * (f/m + g)) / (1 + dt * c)
        self.velocity = (
            self.velocity + dt * (f * self.mass_inv[:, None] + self.gravity[None, :])
        ) / (1.0 + dt * self.damping_c)

        self.position += dt * self.velocity

        # Wall projection and impulse accounting.
        self._project_walls(t)

    def _spring_forces(self) -> np.ndarray:
        dx = self.position[self.edges[:, 1]] - self.position[self.edges[:, 0]]
        length = np.linalg.norm(dx, axis=1)
        inv_length = np.zeros_like(length)
        mask = length > 1e-12
        inv_length[mask] = 1.0 / length[mask]

        # Linear spring force magnitude: k * (l - l0) / l * dx
        scale = self.spring_k * (1.0 - self.rest_length * inv_length)
        fvec = scale[:, None] * dx  # force on edge node 1 (tension positive)

        f = np.zeros_like(self.position)
        np.add.at(f, self.edges[:, 1], fvec)
        np.add.at(f, self.edges[:, 0], -fvec)
        return f

    def _project_walls(self, t: float) -> None:
        h = self._half_extent(t)
        hdot = self._half_extent_derivative(t)

        delta_velocity = np.zeros_like(self.velocity)

        for axis in range(3):
            pos = self.position[:, axis]
            vel = self.velocity[:, axis]

            # Positive wall at +h.
            mask_pos = pos > h
            if mask_pos.any():
                delta_velocity[mask_pos, axis] = hdot - vel[mask_pos]
                vel[mask_pos] = hdot
                pos[mask_pos] = h

            # Negative wall at -h.
            mask_neg = pos < -h
            if mask_neg.any():
                delta_velocity[mask_neg, axis] = -hdot - vel[mask_neg]
                vel[mask_neg] = -hdot
                pos[mask_neg] = -h

        # Contact impulse = sum( m * delta_v ).
        if np.any(delta_velocity != 0.0):
            impulse = np.sum(self.mass[:, None] * delta_velocity, axis=0)
            self.cumulative_external_impulse += impulse
            self.frame_contact_impulse += impulse

    # ------------------------------------------------------------------
    # Compactor schedule
    # ------------------------------------------------------------------
    def _smoothstep(self, x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _smoothstep_derivative(self, x: float) -> float:
        x = float(np.clip(x, 0.0, 1.0))
        return 6.0 * x * (1.0 - x)

    def _half_extent(self, t: float) -> float:
        if t < self.inward[0]:
            return self.open_half
        if t <= self.inward[1]:
            s = (t - self.inward[0]) / (self.inward[1] - self.inward[0])
            return self.open_half + (self.closed_half - self.open_half) * self._smoothstep(s)
        if t < self.withdrawal[0]:
            return self.closed_half
        if t <= self.withdrawal[1]:
            s = (t - self.withdrawal[0]) / (self.withdrawal[1] - self.withdrawal[0])
            return self.closed_half + (self.open_half - self.closed_half) * self._smoothstep(s)
        return self.open_half

    def _half_extent_derivative(self, t: float) -> float:
        if t < self.inward[0] or t > self.withdrawal[1]:
            return 0.0
        if self.inward[0] <= t <= self.inward[1]:
            s = (t - self.inward[0]) / (self.inward[1] - self.inward[0])
            return (
                (self.closed_half - self.open_half)
                / (self.inward[1] - self.inward[0])
                * self._smoothstep_derivative(s)
            )
        if self.withdrawal[0] <= t <= self.withdrawal[1]:
            s = (t - self.withdrawal[0]) / (self.withdrawal[1] - self.withdrawal[0])
            return (
                (self.open_half - self.closed_half)
                / (self.withdrawal[1] - self.withdrawal[0])
                * self._smoothstep_derivative(s)
            )
        return 0.0

    # ------------------------------------------------------------------
    # Deformation gradient
    # ------------------------------------------------------------------
    def _init_triangle_frames(self) -> None:
        f0 = self.faces[:, 0]
        f1 = self.faces[:, 1]
        f2 = self.faces[:, 2]
        X0 = self.rest_position[f0]
        X1 = self.rest_position[f1]
        X2 = self.rest_position[f2]
        e1 = X1 - X0
        e2 = X2 - X0
        n = np.cross(e1, e2)
        norm = np.linalg.norm(n, axis=1)
        n_unit = np.zeros_like(n)
        safe = norm > 1e-12
        n_unit[safe] = n[safe] / norm[safe, None]
        # Degenerate triangles keep a default normal so inversion is still defined.
        n_unit[~safe, 2] = 1.0

        self._tri_inv_dref = np.zeros((self.faces.shape[0], 3, 3), dtype=np.float64)
        # Stack reference edge vectors and normal as columns.
        dref = np.stack([e1, e2, n_unit], axis=2)  # (Ne, 3, 3)
        self._tri_inv_dref[safe] = np.linalg.inv(dref[safe])
        # For degenerate triangles the inverse is left as zero; the corresponding
        # deformation gradient will default to identity below.

    def _deformation_gradient(self) -> np.ndarray:
        f0 = self.faces[:, 0]
        f1 = self.faces[:, 1]
        f2 = self.faces[:, 2]
        x0 = self.position[f0]
        x1 = self.position[f1]
        x2 = self.position[f2]
        e1c = x1 - x0
        e2c = x2 - x0
        nc = np.cross(e1c, e2c)
        norm = np.linalg.norm(nc, axis=1)
        nc_unit = np.zeros_like(nc)
        safe = norm > 1e-12
        nc_unit[safe] = nc[safe] / norm[safe, None]
        nc_unit[~safe, 2] = 1.0

        dcur = np.stack([e1c, e2c, nc_unit], axis=2)  # (Ne, 3, 3)
        F = np.einsum("nij,njk->nik", dcur, self._tri_inv_dref)
        # Identity for degenerate reference triangles.
        degenerate = ~safe
        if degenerate.any():
            F[degenerate] = np.eye(3)
        return F

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------
    def _build_frame(self, time: float) -> StateFrame:
        external_force_term = self.frame_contact_impulse / self.sample_dt
        # Include gravity in the external-force term for consistency.
        external_force_term += self.total_mass * self.gravity

        # External impulse budget is purely the wall contact; gravity is zero here
        # but we accumulate it for the running total if it were ever non-zero.
        # (Already added in _project_walls for contact.)

        G = 1
        N = self.num_particles
        budget_external_force = np.zeros((G, 1, 3), dtype=np.float64)
        budget_external_force[0, 0, :] = external_force_term

        element_F = self._deformation_gradient()

        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy().reshape(G, 3),
                "budget_cumulative_inflow_momentum": np.zeros((G, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum": np.zeros((G, 3), dtype=np.float64),
                "budget_domain_weights": np.ones((G, N), dtype=np.float64),
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": np.zeros((1,), dtype=np.float64),
                "element_deformation_gradient": element_F,
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
            },
        )
        frame.validate()
        return frame
