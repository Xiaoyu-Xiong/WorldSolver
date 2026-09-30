"""Explicit nonlinear finite-element solver: compressed rubber cube.

Physical model
-------------
The rubber cube is a three-dimensional compressible neo-Hookean solid

    Psi(F) = mu/2 (I1 - 3) - mu ln J + lam/2 (ln J)^2
    P(F)   = mu (F - F^-T) + lam ln(J) F^-T

discretised with constant-strain (linear) tetrahedra on the frozen
9x9x9 vertex / 3072 tetrahedron mesh.  Material damping is a
Kelvin-Voigt viscous first-Piola stress proportional to the reference
elasticity tensor,

    P_visc = beta (mu (Fdot + Fdot^T) + lam tr(Fdot) I),

which is an internal (self-equilibrated) stress: the four nodal forces of
every element sum to zero, so it never enters the external momentum budget.

Momentum is advanced with lumped nodal masses and a symplectic
(semi-implicit) Euler update, sub-stepped below the elastic/viscous
stability limit of the mesh.

Boundary conditions (frozen config)
-----------------------------------
* bottom face z = 0: fixed support, all displacement components clamped;
  its constraint force is reported as the support reaction.
* top face z = 0.2: uniform prescribed compressive Piola traction along
  -z with peak magnitude 50 kPa, scaled by the frozen ramp schedule.
* no gravity is declared by the scene, so the cube is undeformed and at
  rest until the traction ramp starts.

Near incompressibility (nu = 0.45) makes the clamped base resist lateral
expansion, so the cube shortens while its free side faces bulge outward
into a barrel shape instead of shrinking uniformly.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

DENSITY = 1100.0            # kg/m^3, rubber
YOUNGS_MODULUS = 1.3e5      # Pa, soft rubber block
POISSON_RATIO = 0.45        # near incompressible -> lateral bulging
DAMPING_BETA = 1.5e-3       # s, Kelvin-Voigt stiffness-proportional damping
CFL_SAFETY = 0.25
MAX_SUBSTEPS = 2500
FLOOR_STIFFNESS = 1.0e5     # N/m, penalty against the rigid support
MAX_SPEED = 20.0            # m/s, numerical safeguard
_EYE = np.eye(3)[None, :, :]


def _inv_det3(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched analytic inverse and determinant of [...,3,3] matrices."""
    a = matrices[:, 0, 0]
    b = matrices[:, 0, 1]
    c = matrices[:, 0, 2]
    d = matrices[:, 1, 0]
    e = matrices[:, 1, 1]
    f = matrices[:, 1, 2]
    g = matrices[:, 2, 0]
    h = matrices[:, 2, 1]
    i = matrices[:, 2, 2]
    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    c10 = -(b * i - c * h)
    c11 = a * i - c * g
    c12 = -(a * h - b * g)
    c20 = b * f - c * e
    c21 = -(a * f - c * d)
    c22 = a * e - b * d
    det = a * c00 + b * c01 + c * c02
    adjugate = np.empty_like(matrices)
    adjugate[:, 0, 0] = c00
    adjugate[:, 0, 1] = c10
    adjugate[:, 0, 2] = c20
    adjugate[:, 1, 0] = c01
    adjugate[:, 1, 1] = c11
    adjugate[:, 1, 2] = c21
    adjugate[:, 2, 0] = c02
    adjugate[:, 2, 1] = c12
    adjugate[:, 2, 2] = c22
    safe = np.where(np.abs(det) < 1e-14, 1e-14, det)
    return adjugate / safe[:, None, None], det


class BenchmarkSolver:
    """Lumped-mass explicit FEM solver for the compressed rubber cube."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config

        vertices, tets = self._load_mesh(scene)
        self.rest = np.ascontiguousarray(vertices, dtype=float)
        self.n_nodes = int(self.rest.shape[0])

        # Reference shape matrices; enforce positive tetrahedron orientation.
        dm = np.stack(
            (
                self.rest[tets[:, 1]] - self.rest[tets[:, 0]],
                self.rest[tets[:, 2]] - self.rest[tets[:, 0]],
                self.rest[tets[:, 3]] - self.rest[tets[:, 0]],
            ),
            axis=2,
        )
        det = np.linalg.det(dm)
        flipped = det < 0.0
        if np.any(flipped):
            tets = tets.copy()
            tets[flipped, 2], tets[flipped, 3] = tets[flipped, 3], tets[flipped, 2].copy()
            dm = np.stack(
                (
                    self.rest[tets[:, 1]] - self.rest[tets[:, 0]],
                    self.rest[tets[:, 2]] - self.rest[tets[:, 0]],
                    self.rest[tets[:, 3]] - self.rest[tets[:, 0]],
                ),
                axis=2,
            )
            det = np.linalg.det(dm)
        self.tets = np.ascontiguousarray(tets, dtype=np.int64)
        self.n_elems = int(self.tets.shape[0])
        self.dm_inv, _ = _inv_det3(np.ascontiguousarray(dm))
        self.rest_volume = det / 6.0
        # Element force weights: -V0 * P * Dm^-T  (see nodal force assembly).
        self.dm_inv_t = np.ascontiguousarray(np.transpose(self.dm_inv, (0, 2, 1)))
        self.flat_index = self.tets.reshape(-1)

        # Lumped nodal masses (quarter of each incident element mass).
        node_mass = np.bincount(
            self.flat_index,
            weights=np.repeat(DENSITY * self.rest_volume / 4.0, 4),
            minlength=self.n_nodes,
        )
        self.node_mass = node_mass
        self.inv_mass = np.where(node_mass > 0.0, 1.0 / np.maximum(node_mass, 1e-16), 0.0)
        self.total_mass = float(node_mass.sum())

        # Lame parameters of the frozen material.
        self.mu = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
        self.lam = (
            YOUNGS_MODULUS
            * POISSON_RATIO
            / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))
        )

        self._setup_boundary_conditions()
        self._setup_time_stepping()

        # Initial state: undeformed cube at rest.
        self.position = self.rest.copy()
        self.velocity = np.zeros((self.n_nodes, 3), dtype=float)
        self.current_time = 0.0
        self.step_count = 0
        self.domain_weights = np.ones((1, self.n_nodes), dtype=float)

    # ------------------------------------------------------------------ setup
    def _load_mesh(self, scene):
        vertices = getattr(scene, "vertices", None)
        tets = getattr(scene, "tetrahedra", None)
        self.faces = np.asarray(getattr(scene, "faces", np.zeros((0, 3))), dtype=np.int64)
        if vertices is None or tets is None:
            asset = Path(__file__).resolve().parents[1] / "assets" / "rubber_cube_mesh.npz"
            with np.load(asset) as mesh:
                vertices = np.asarray(mesh["vertices"], dtype=float)
                tets = np.asarray(mesh["tetrahedra"], dtype=np.int64)
                self.faces = np.asarray(mesh["faces"], dtype=np.int64)
        return np.asarray(vertices, dtype=float), np.asarray(tets, dtype=np.int64)

    def _setup_boundary_conditions(self) -> None:
        boundary = dict(self.config.get("boundary_conditions", {}) or {})
        traction = dict(boundary.get("top_compressive_traction", {}) or {})

        z_min = float(self.rest[:, 2].min())
        z_max = float(self.rest[:, 2].max())
        tolerance = 1e-9 + 1e-6 * max(1.0, z_max - z_min)
        self.fixed_mask = self.rest[:, 2] <= z_min + tolerance
        self.free_mask = ~self.fixed_mask
        self.top_mask = self.rest[:, 2] >= z_max - tolerance
        self.cube_height_0 = z_max - z_min

        direction = np.asarray(traction.get("direction", [0.0, 0.0, -1.0]), dtype=float)
        norm = float(np.linalg.norm(direction))
        self.traction_direction = direction / norm if norm > 1e-12 else np.array([0.0, 0.0, -1.0])
        self.peak_pressure = float(traction.get("peak_magnitude_pa", 50000.0))

        schedule = traction.get("schedule") or [{"time": 0.0, "scale": 1.0}]
        times = [float(entry["time"]) for entry in schedule]
        scales = [float(entry["scale"]) for entry in schedule]
        order = np.argsort(np.asarray(times, dtype=float), kind="stable")
        self.schedule_times = np.asarray(times, dtype=float)[order]
        self.schedule_scales = np.asarray(scales, dtype=float)[order]

        # Consistent nodal load of a uniform Piola traction on the top face.
        nodal = np.zeros((self.n_nodes, 3), dtype=float)
        loaded_area = 0.0
        if self.faces.size:
            on_top = np.all(self.top_mask[self.faces], axis=1)
            top_faces = self.faces[on_top]
            if top_faces.size:
                p0 = self.rest[top_faces[:, 0]]
                edge1 = self.rest[top_faces[:, 1]] - p0
                edge2 = self.rest[top_faces[:, 2]] - p0
                areas = 0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1)
                loaded_area = float(areas.sum())
                share = np.repeat(
                    self.peak_pressure * areas / 3.0, 3
                )[:, None] * self.traction_direction[None, :]
                flat_faces = top_faces.reshape(-1)
                for axis in range(3):
                    nodal[:, axis] += np.bincount(
                        flat_faces, weights=share[:, axis], minlength=self.n_nodes
                    )
        if loaded_area <= 0.0:
            # Fallback: distribute uniformly over detected top nodes.
            span_x = float(self.rest[:, 0].max() - self.rest[:, 0].min())
            span_y = float(self.rest[:, 1].max() - self.rest[:, 1].min())
            loaded_area = max(span_x * span_y, 1e-12)
            top_nodes = np.flatnonzero(self.top_mask)
            if top_nodes.size:
                nodal[top_nodes] = (
                    self.peak_pressure * loaded_area / top_nodes.size
                ) * self.traction_direction[None, :]
        self.nodal_traction_full = nodal
        self.loaded_area = loaded_area
        self.reference_force = max(abs(self.peak_pressure) * loaded_area, 1e-9)
        self.load_centroid = (
            self.rest[self.top_mask].mean(axis=0)
            if np.any(self.top_mask)
            else self.rest.mean(axis=0)
        )
        self.support_centroid = (
            self.rest[self.fixed_mask].mean(axis=0)
            if np.any(self.fixed_mask)
            else self.rest.mean(axis=0)
        )

    def _setup_time_stepping(self) -> None:
        self.sample_dt = float(self.config.get("sample_dt", 0.02))
        wave_speed = math.sqrt((self.lam + 2.0 * self.mu) / DENSITY)
        # Smallest element altitude ~ 3 V / max face area, approximated by
        # the shortest reference edge, which bounds the element frequency.
        edges = np.stack(
            (
                self.rest[self.tets[:, 1]] - self.rest[self.tets[:, 0]],
                self.rest[self.tets[:, 2]] - self.rest[self.tets[:, 0]],
                self.rest[self.tets[:, 3]] - self.rest[self.tets[:, 0]],
                self.rest[self.tets[:, 2]] - self.rest[self.tets[:, 1]],
                self.rest[self.tets[:, 3]] - self.rest[self.tets[:, 1]],
                self.rest[self.tets[:, 3]] - self.rest[self.tets[:, 2]],
            ),
            axis=1,
        )
        h_min = float(np.linalg.norm(edges, axis=2).min())
        omega_max = 2.0 * 1.6 * wave_speed / max(h_min, 1e-9)
        # Stability of symplectic Euler with viscous damping:
        # dt < 2 zeta / omega with zeta = beta omega / 2, plus 2/omega bound.
        dt_elastic = 2.0 / omega_max
        dt_visc = 2.0 / (omega_max * omega_max * max(DAMPING_BETA, 1e-12))
        dt_limit = CFL_SAFETY * min(dt_elastic, dt_visc)
        substeps = int(math.ceil(self.sample_dt / max(dt_limit, 1e-12)))
        self.substeps = int(min(max(substeps, 1), MAX_SUBSTEPS))
        self.dt = self.sample_dt / self.substeps
        self.wave_speed = wave_speed


    # ---------------------------------------------------------------- physics
    def _deformation_gradient(self, position: np.ndarray) -> np.ndarray:
        tets = self.tets
        p0 = position[tets[:, 0]]
        ds = np.stack(
            (
                position[tets[:, 1]] - p0,
                position[tets[:, 2]] - p0,
                position[tets[:, 3]] - p0,
            ),
            axis=2,
        )
        return ds @ self.dm_inv

    def _velocity_gradient(self, velocity: np.ndarray) -> np.ndarray:
        tets = self.tets
        v0 = velocity[tets[:, 0]]
        vs = np.stack(
            (
                velocity[tets[:, 1]] - v0,
                velocity[tets[:, 2]] - v0,
                velocity[tets[:, 3]] - v0,
            ),
            axis=2,
        )
        return vs @ self.dm_inv

    def _first_piola(self, defgrad: np.ndarray) -> np.ndarray:
        """Compressible neo-Hookean first Piola-Kirchhoff stress."""
        f_inv, det = _inv_det3(np.ascontiguousarray(defgrad))
        f_inv_t = np.transpose(f_inv, (0, 2, 1))
        jac = np.maximum(det, 1e-6)
        log_j = np.log(jac)
        return self.mu * (defgrad - f_inv_t) + self.lam * log_j[:, None, None] * f_inv_t

    def _internal_forces(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        defgrad = self._deformation_gradient(position)
        stress = self._first_piola(defgrad)
        if DAMPING_BETA > 0.0:
            rate = self._velocity_gradient(velocity)
            trace = rate[:, 0, 0] + rate[:, 1, 1] + rate[:, 2, 2]
            stress = stress + DAMPING_BETA * (
                self.mu * (rate + np.transpose(rate, (0, 2, 1)))
                + self.lam * trace[:, None, None] * _EYE
            )
        # H columns are the forces on element nodes 1, 2, 3.
        h = (-self.rest_volume[:, None, None]) * (stress @ self.dm_inv_t)
        element_forces = np.empty((self.n_elems, 4, 3), dtype=float)
        element_forces[:, 1, :] = h[:, :, 0]
        element_forces[:, 2, :] = h[:, :, 1]
        element_forces[:, 3, :] = h[:, :, 2]
        element_forces[:, 0, :] = -(h[:, :, 0] + h[:, :, 1] + h[:, :, 2])
        flat = element_forces.reshape(-1, 3)
        forces = np.empty((self.n_nodes, 3), dtype=float)
        for axis in range(3):
            forces[:, axis] = np.bincount(
                self.flat_index, weights=flat[:, axis], minlength=self.n_nodes
            )
        return forces

    def _load_scale(self, time: float) -> float:
        return float(
            np.interp(
                float(time),
                self.schedule_times,
                self.schedule_scales,
                left=float(self.schedule_scales[0]),
                right=float(self.schedule_scales[-1]),
            )
        )

    def _floor_force(self, position: np.ndarray) -> np.ndarray:
        """Penalty contact against the rigid support plane z = 0."""
        penetration = np.minimum(position[:, 2], 0.0)
        force = np.zeros_like(position)
        if np.any(penetration < 0.0):
            force[:, 2] = -FLOOR_STIFFNESS * penetration
        return force

    def _assemble(self, time: float, position: np.ndarray, velocity: np.ndarray):
        internal = self._internal_forces(position, velocity)
        traction = self._load_scale(time) * self.nodal_traction_full
        contact = self._floor_force(position)
        total = internal + traction + contact
        # Reaction of the fixed support: whatever keeps the clamped nodes at rest.
        reaction = np.zeros_like(total)
        reaction[self.fixed_mask] = -total[self.fixed_mask]
        return internal, traction, contact, reaction, total

    def _step(self, time: float) -> None:
        dt = self.dt
        _, _, _, _, total = self._assemble(time, self.position, self.velocity)
        acceleration = total * self.inv_mass[:, None]
        acceleration[self.fixed_mask] = 0.0
        velocity = self.velocity + dt * acceleration
        velocity[self.fixed_mask] = 0.0
        speed = np.linalg.norm(velocity, axis=1)
        excessive = speed > MAX_SPEED
        if np.any(excessive):
            velocity[excessive] *= (MAX_SPEED / speed[excessive])[:, None]
        position = self.position + dt * velocity
        position[self.fixed_mask] = self.rest[self.fixed_mask]
        self.velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)
        self.position = np.nan_to_num(
            position, nan=0.0, posinf=0.0, neginf=0.0, copy=False
        )
        self.step_count += 1

    # ----------------------------------------------------------------- driver
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        target_step = int(round(target / self.dt))
        while self.step_count < target_step:
            self._step(self.step_count * self.dt)
        self.current_time = target
        return self._build_frame(target)

    def _build_frame(self, time: float) -> StateFrame:
        position = np.ascontiguousarray(self.position, dtype=float)
        velocity = np.ascontiguousarray(self.velocity, dtype=float)
        internal, traction, contact, reaction, _ = self._assemble(time, position, velocity)
        defgrad = self._deformation_gradient(position)

        # Single budget domain: the whole rubber cube. External terms are the
        # prescribed surface traction, the fixed-support reaction and the
        # support contact penalty; elastic and viscous pairs are internal.
        budget = np.zeros((1, 3, 3), dtype=float)
        budget[0, 0] = traction.sum(axis=0)
        budget[0, 1] = reaction.sum(axis=0)
        budget[0, 2] = contact.sum(axis=0)
        force_position = np.zeros((1, 3, 3), dtype=float)
        force_position[0, 0] = self.load_centroid
        force_position[0, 1] = self.support_centroid
        force_position[0, 2] = self.support_centroid

        scale = self._load_scale(time)
        frame = StateFrame(
            time=float(time),
            position=position.copy(),
            velocity=velocity.copy(),
            mass=self.node_mass.copy(),
            attributes={
                "mass_quantity": self.node_mass.copy(),
                "rest_position": self.rest.copy(),
                "bottom_face": self.fixed_mask.astype(float),
                "top_face": self.top_mask.astype(float),
            },
            fields={
                "budget_external_force": np.nan_to_num(budget),
                "budget_force_position": force_position,
                "budget_domain_weights": self.domain_weights.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "element_deformation_gradient": np.nan_to_num(
                    np.ascontiguousarray(defgrad, dtype=float)
                ),
                "elastic_internal_force": np.nan_to_num(internal),
                "gravity_acceleration": np.zeros((1, 3), dtype=float),
            },
            scalars={
                "reference_force": float(self.reference_force),
                "applied_traction_pa": float(scale * self.peak_pressure),
                "top_height_m": float(position[self.top_mask, 2].mean())
                if np.any(self.top_mask)
                else float(position[:, 2].max()),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        position = self.position
        height = float(position[:, 2].max() - position[:, 2].min())
        half_width = float(
            np.max(np.abs(position[:, :2] - self.rest[:, :2].mean(axis=0)[None, :]))
        )
        return {
            "substeps_per_sample": int(self.substeps),
            "dt": float(self.dt),
            "steps_taken": int(self.step_count),
            "total_mass_kg": float(self.total_mass),
            "shear_modulus_pa": float(self.mu),
            "lame_lambda_pa": float(self.lam),
            "wave_speed_m_s": float(self.wave_speed),
            "loaded_area_m2": float(self.loaded_area),
            "reference_force_n": float(self.reference_force),
            "final_height_m": height,
            "final_height_ratio": height / max(self.cube_height_0, 1e-12),
            "final_max_half_extent_m": half_width,
        }
