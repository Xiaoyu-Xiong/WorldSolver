"""Incompressible elastic FEM solver: soft sphere compressed between plates.

Physical model
--------------
Volumetric linear-tetrahedron FEM. The frozen surface asset supplies the twelve
state samples; a dependent interior node at the mass centre closes the volume so
twenty tetrahedra fan from that centre to the twenty outward surface triangles.
The centre is an affine function of the free nodes, so its elastic force and its
constraint gradients are redistributed with the mass weights and every internal
operator conserves linear momentum exactly.

Constitutive law (compressible neo-Hookean):
    psi(F) = mu/2 (tr(F^T F) - 3) - mu ln J + lambda/2 (ln J)^2
    P(F)   = mu (F - F^-T) + lambda ln(J) F^-T

Time stepping: symplectic-Euler velocity prediction with momentum-preserving
internal damping, then a position-level constraint solve, then the
implicit-Euler consistent velocity update v = (x_new - x_old) / dt. The
constraint solve alternates Jacobi projection of the per-element volume
constraint with hard non-penetration clamping against the two kinematic plates
and closes with an exact momentum-neutral lateral dilation that restores the
total volume (diag(s, s, 1) scales every tetrahedron volume by s**2, so
s = sqrt(V_rest / V) is exact and leaves the plate contacts untouched).

This reproduces the source experiment: the body cannot lose volume, so as the
gap narrows it spreads sideways past its original outline, forms broad flat
contact patches, and ends as a thin wide disc at the smallest positive gap.

Only NumPy/SciPy are used; no external physics engine is imported.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-12

def _surface_faces(points: np.ndarray) -> np.ndarray:
    """Outward-oriented triangles of the convex surface mesh."""
    pts = np.asarray(points, dtype=float)
    try:
        from scipy.spatial import ConvexHull

        faces = np.asarray(ConvexHull(pts).simplices, dtype=np.int64)
    except Exception:  # deterministic geometric fallback
        faces = _equal_edge_faces(pts)
    center = pts.mean(axis=0)
    oriented = []
    for tri in faces:
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        if float(np.cross(b - a, c - a) @ (a - center)) < 0.0:
            tri = np.array([tri[0], tri[2], tri[1]], dtype=np.int64)
        oriented.append(np.asarray(tri, dtype=np.int64))
    return np.asarray(oriented, dtype=np.int64)


def _equal_edge_faces(pts: np.ndarray) -> np.ndarray:
    """Fallback triangulation for a regular convex polyhedron."""
    count = len(pts)
    distance = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    positive = distance[distance > 1e-9]
    limit = 1.05 * float(positive.min())
    faces = []
    for i in range(count):
        for j in range(i + 1, count):
            if distance[i, j] > limit:
                continue
            for k in range(j + 1, count):
                if distance[i, k] <= limit and distance[j, k] <= limit:
                    faces.append([i, j, k])
    return np.asarray(faces, dtype=np.int64)


class BenchmarkSolver:
    """Volumetric FEM solver with kinematic plate contact."""

    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        config = dict(scene.config)
        self.config = config

        initial = scene.initial_state()
        self.rest_position = np.asarray(initial.position, dtype=float).copy()
        self.position = self.rest_position.copy()
        self.velocity = np.zeros_like(self.position)
        self.n_nodes = int(len(self.position))

        sphere = config["sphere"]
        self.total_mass = float(sphere["mass"])
        self.rest_volume_declared = float(sphere.get("rest_volume", self.total_mass))
        mass = np.asarray(initial.mass, dtype=float).copy() if initial.mass is not None else None
        if mass is None or mass.shape != (self.n_nodes,) or not np.all(mass > 0.0):
            mass = np.full(self.n_nodes, self.total_mass / self.n_nodes)
        self.mass = mass
        self.inv_mass = 1.0 / self.mass
        self.mass_total = float(self.mass.sum())
        # Interior node is dependent: x_c = sum_i w_i x_i with sum_i w_i = 1.
        self.weights = self.mass / self.mass_total

        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        plates = config["plates"]
        self.plate_half_extents = np.asarray(plates["half_extents"], dtype=float)
        self.face_lower_0 = float(plates["initial_inner_faces_z"][0])
        self.face_upper_0 = float(plates["initial_inner_faces_z"][1])
        self.face_lower_1 = float(plates["final_inner_faces_z"][0])
        self.face_upper_1 = float(plates["final_inner_faces_z"][1])
        self.motion_start = float(plates.get("motion_start", 0.0))
        self.motion_end = float(plates.get("motion_end", config["duration"]))
        span = max(self.motion_end - self.motion_start, 1e-9)
        # Constant symmetric approach reaching the frozen final inner faces at
        # motion_end, then hold (the frozen motion_profile).
        self.plate_speed = abs(self.face_lower_1 - self.face_lower_0) / span
        self.minimum_positive_gap = float(
            plates.get("minimum_positive_gap", self.face_upper_1 - self.face_lower_1)
        )

        self.faces = _surface_faces(self.rest_position)
        self.n_elements = int(len(self.faces))
        self._build_elements()

        # Material: soft, nearly incompressible rubber-like solid.
        self.mu = 200.0
        self.lam = 1000.0
        self.damping = 10.0

        self.sample_dt = float(config["sample_dt"])
        self.substep_dt_target = 4.0e-4
        self.substeps = max(1, int(math.ceil(self.sample_dt / self.substep_dt_target - 1e-9)))
        self.substep_dt = self.sample_dt / self.substeps
        self.volume_iterations = 4
        self.volume_relaxation = 0.3
        self.max_relative_speed = 60.0

        self.cumulative_gravity = np.zeros(3)
        self.cumulative_lower = np.zeros(3)
        self.cumulative_upper = np.zeros(3)
        self.interval_gravity = np.zeros(3)
        self.interval_lower = np.zeros(3)
        self.interval_upper = np.zeros(3)

        self.reference_momentum = max(self.mass_total * max(self.plate_speed, 1e-6), 1e-9)
        self.constraint_scale = max(abs(self.rest_volume_declared), 1e-12)

        self.current_time = 0.0
        self._steps_taken = 0
        self._resets = 0
        self._max_volume_error = 0.0
        self._min_jacobian = 1.0

    def _build_elements(self) -> None:
        """Tetrahedra spanning interior node -> outward surface triangle."""
        rest = self.rest_position
        center = self.weights @ rest
        elements = []
        for tri in self.faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            det = float(
                np.linalg.det(
                    np.stack((rest[a] - center, rest[b] - center, rest[c] - center), axis=1)
                )
            )
            if det < 0.0:
                b, c = c, b
            elements.append([a, b, c])
        self.elements = np.asarray(elements, dtype=np.int64)

        rest_edges = self._element_edges(rest, center)
        self.rest_shape_inverse = np.linalg.inv(rest_edges)
        self.element_rest_volume = np.abs(np.linalg.det(rest_edges)) / 6.0
        self.mesh_rest_volume = float(self.element_rest_volume.sum())
        # Report volumes in the declared physical units of the round body.
        self.volume_scale = self.rest_volume_declared / max(self.mesh_rest_volume, _EPS)
        self.rest_volume_share = np.full(self.n_nodes, self.rest_volume_declared / self.n_nodes)
        counts = np.bincount(self.elements.reshape(-1), minlength=self.n_nodes)
        self.node_element_count = np.maximum(counts, 1).astype(float)
        # Sparse selector used to scatter element node gradients densely.
        self.node_selector = np.zeros((self.n_elements, 3, self.n_nodes))
        for e in range(self.n_elements):
            for k in range(3):
                self.node_selector[e, k, self.elements[e, k]] = 1.0

    def _element_edges(self, x: np.ndarray, center: np.ndarray) -> np.ndarray:
        a = x[self.elements[:, 0]] - center
        b = x[self.elements[:, 1]] - center
        c = x[self.elements[:, 2]] - center
        return np.stack((a, b, c), axis=2)

    # ---------------------------------------------------------- plate motion
    def plate_faces(self, time: float) -> tuple[float, float]:
        elapsed = min(
            max(float(time) - self.motion_start, 0.0), self.motion_end - self.motion_start
        )
        lower = min(self.face_lower_0 + self.plate_speed * elapsed, self.face_lower_1)
        upper = max(self.face_upper_0 - self.plate_speed * elapsed, self.face_upper_1)
        return float(lower), float(upper)

    def plate_speed_at(self, time: float) -> float:
        if self.motion_start <= float(time) < self.motion_end:
            return float(self.plate_speed)
        return 0.0

    # ---------------------------------------------------------------- forces
    def _deformation_gradients(self, x: np.ndarray) -> np.ndarray:
        center = self.weights @ x
        return self._element_edges(x, center) @ self.rest_shape_inverse

    def _elastic_forces(self, x: np.ndarray) -> np.ndarray:
        gradient = self._deformation_gradients(x)
        f0 = gradient[:, :, 0]
        f1 = gradient[:, :, 1]
        f2 = gradient[:, :, 2]
        cofactor = np.stack((np.cross(f1, f2), np.cross(f2, f0), np.cross(f0, f1)), axis=2)
        jacobian = np.einsum("ei,ei->e", f0, np.cross(f1, f2))
        self._min_jacobian = min(self._min_jacobian, float(np.min(jacobian)))
        safe = np.where(jacobian > 1.0e-3, jacobian, 1.0e-3)
        inverse_transpose = cofactor / safe[:, None, None]
        stress = self.mu * (gradient - inverse_transpose) + (self.lam * np.log(safe))[
            :, None, None
        ] * inverse_transpose
        # Columns of H are the forces on the three surface nodes of the element.
        h = -self.element_rest_volume[:, None, None] * (
            stress @ np.transpose(self.rest_shape_inverse, (0, 2, 1))
        )
        forces = np.zeros_like(x)
        interior = np.zeros(3)
        for k in range(3):
            np.add.at(forces, self.elements[:, k], h[:, :, k])
            interior -= h[:, :, k].sum(axis=0)
        # Redistribute the dependent interior-node force through the affine map.
        forces += self.weights[:, None] * interior[None, :]
        forces += self.mass[:, None] * self.gravity[None, :]
        return forces

    # ----------------------------------------------------------- constraints
    def _project_volume(self, x: np.ndarray) -> None:
        """Momentum-preserving Jacobi projection of per-element volume."""
        for _ in range(self.volume_iterations):
            center = self.weights @ x
            a = x[self.elements[:, 0]] - center
            b = x[self.elements[:, 1]] - center
            c = x[self.elements[:, 2]] - center
            volume = np.einsum("ei,ei->e", a, np.cross(b, c)) / 6.0
            residual = volume - self.element_rest_volume
            local = np.stack(
                (np.cross(b, c) / 6.0, np.cross(c, a) / 6.0, np.cross(a, b) / 6.0), axis=1
            )  # [Ne,3,3] gradients wrt the element edge vectors
            interior = -local.sum(axis=1)  # gradient wrt the dependent node
            # Dense per-node gradient: direct term plus affine interior term.
            gradient = np.einsum("ekn,eki->eni", self.node_selector, local)
            gradient += interior[:, None, :] * self.weights[None, :, None]
            weighted = gradient * self.inv_mass[None, :, None]
            denominator = np.einsum("eni,eni->e", gradient, weighted)
            denominator = np.where(denominator > _EPS, denominator, _EPS)
            multiplier = residual / denominator
            correction = -np.einsum("e,eni->ni", multiplier, weighted)
            x += self.volume_relaxation * correction

    def _clamp_plates(
        self, x: np.ndarray, lower: float, upper: float, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Hard non-penetration against both plates; returns contact impulses."""
        impulse_lower = np.zeros(3)
        impulse_upper = np.zeros(3)
        inside = (np.abs(x[:, 0]) <= self.plate_half_extents[0]) & (
            np.abs(x[:, 1]) <= self.plate_half_extents[1]
        )
        below = inside & (x[:, 2] < lower)
        if np.any(below):
            shift = lower - x[below, 2]
            impulse_lower[2] = float(np.sum(self.mass[below] * shift) / dt)
            x[below, 2] = lower
        above = inside & (x[:, 2] > upper)
        if np.any(above):
            shift = upper - x[above, 2]
            impulse_upper[2] = float(np.sum(self.mass[above] * shift) / dt)
            x[above, 2] = upper
        return impulse_lower, impulse_upper

    def _restore_volume(self, x: np.ndarray) -> None:
        """Exact momentum-neutral lateral dilation about the mass centre."""
        center = self.weights @ x
        a = x[self.elements[:, 0]] - center
        b = x[self.elements[:, 1]] - center
        c = x[self.elements[:, 2]] - center
        volume = float(np.einsum("ei,ei->e", a, np.cross(b, c)).sum() / 6.0)
        if not np.isfinite(volume) or volume <= _EPS:
            return
        ratio = self.mesh_rest_volume / volume
        if not np.isfinite(ratio) or ratio <= 0.0:
            return
        scale = min(max(math.sqrt(ratio), 0.5), 2.0)
        center_xy = self.weights @ x[:, :2]
        x[:, :2] = center_xy[None, :] + scale * (x[:, :2] - center_xy[None, :])

    # -------------------------------------------------------------- stepping
    def _substep(self, time: float, dt: float) -> None:
        x_old = self.position
        forces = self._elastic_forces(x_old)
        velocity = self.velocity + dt * (forces * self.inv_mass[:, None])

        # Internal damping on relative motion only: momentum is untouched.
        mean_velocity = (self.mass[:, None] * velocity).sum(axis=0) / self.mass_total
        velocity = mean_velocity[None, :] + (velocity - mean_velocity[None, :]) / (
            1.0 + self.damping * dt
        )
        relative = velocity - mean_velocity[None, :]
        peak = float(np.max(np.abs(relative))) if relative.size else 0.0
        if peak > self.max_relative_speed:
            velocity = mean_velocity[None, :] + relative * (self.max_relative_speed / peak)

        x_new = x_old + dt * velocity
        lower, upper = self.plate_faces(time + dt)

        impulse_lower = np.zeros(3)
        impulse_upper = np.zeros(3)
        for _ in range(3):
            self._project_volume(x_new)
            step_lower, step_upper = self._clamp_plates(x_new, lower, upper, dt)
            impulse_lower += step_lower
            impulse_upper += step_upper
        self._restore_volume(x_new)
        step_lower, step_upper = self._clamp_plates(x_new, lower, upper, dt)
        impulse_lower += step_lower
        impulse_upper += step_upper

        if not np.isfinite(x_new).all():
            self._resets += 1
            self.velocity = np.zeros_like(self.velocity)
            return

        self.position = x_new
        new_velocity = (x_new - x_old) / dt
        if not np.isfinite(new_velocity).all():
            new_velocity = np.zeros_like(new_velocity)
        self.velocity = new_velocity

        gravity_impulse = self.mass_total * self.gravity * dt
        self.cumulative_gravity += gravity_impulse
        self.cumulative_lower += impulse_lower
        self.cumulative_upper += impulse_upper
        self.interval_gravity += gravity_impulse
        self.interval_lower += impulse_lower
        self.interval_upper += impulse_upper
        self._steps_taken += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        self.interval_gravity = np.zeros(3)
        self.interval_lower = np.zeros(3)
        self.interval_upper = np.zeros(3)
        interval = target - self.current_time
        if interval > 1e-12:
            count = max(1, int(math.ceil(interval / self.substep_dt - 1e-9)))
            dt = interval / count
            for index in range(count):
                self._substep(self.current_time + index * dt, dt)
            self.current_time = target
        else:
            interval = 0.0
        return self._frame(target, interval)

    # ---------------------------------------------------------------- export
    def _frame(self, time: float, interval: float) -> StateFrame:
        x = self.position
        gradient = self._deformation_gradients(x)
        jacobian = np.linalg.det(gradient)
        center = self.weights @ x
        a = x[self.elements[:, 0]] - center
        b = x[self.elements[:, 1]] - center
        c = x[self.elements[:, 2]] - center
        element_volume = np.einsum("ei,ei->e", a, np.cross(b, c)) / 6.0
        volume_report = float(element_volume.sum()) * self.volume_scale
        self._max_volume_error = max(
            self._max_volume_error,
            abs(volume_report - self.rest_volume_declared) / self.constraint_scale,
        )

        nodal_gradient = np.zeros((self.n_nodes, 3, 3))
        for k in range(3):
            np.add.at(nodal_gradient, self.elements[:, k], gradient)
        nodal_gradient /= self.node_element_count[:, None, None]

        lower, upper = self.plate_faces(time)
        speed = self.plate_speed_at(time)

        if interval > 1e-12:
            force_gravity = self.interval_gravity / interval
            force_lower = self.interval_lower / interval
            force_upper = self.interval_upper / interval
        else:
            force_gravity = np.zeros(3)
            force_lower = np.zeros(3)
            force_upper = np.zeros(3)
        external_force = np.stack((force_gravity, force_lower, force_upper), axis=0)
        cumulative_impulse = (
            self.cumulative_gravity + self.cumulative_lower + self.cumulative_upper
        )

        frame = StateFrame(
            time=float(time),
            position=x.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n_nodes, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "volume_quantity": np.full(self.n_nodes, volume_report / self.n_nodes),
                "rest_position": self.rest_position.copy(),
                "deformation_gradient": nodal_gradient,
                "rest_volume_share": self.rest_volume_share.copy(),
            },
            fields={
                "element_deformation_gradient": gradient.copy(),
                "element_volume": element_volume * self.volume_scale,
                "element_jacobian": jacobian.copy(),
                "constraint_value": np.array([volume_report], dtype=float),
                "constraint_target": np.array([self.rest_volume_declared], dtype=float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "budget_domain_weights": np.ones((1, self.n_nodes), dtype=float),
                "budget_external_force": external_force.reshape(1, 3, 3),
                "budget_cumulative_external_impulse": cumulative_impulse.reshape(1, 3),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
                "plate_inner_faces_z": np.array([lower, upper], dtype=float),
                "plate_velocity_z": np.array([speed, -speed], dtype=float),
            },
            scalars={
                "constraint_scale": float(self.constraint_scale),
                "reference_momentum": float(self.reference_momentum),
                "rest_volume": float(self.rest_volume_declared),
                "current_volume": volume_report,
                "volume_relative_error": float(
                    (volume_report - self.rest_volume_declared) / self.constraint_scale
                ),
                "plate_gap": float(upper - lower),
                "initial_plate_gap": float(self.face_upper_0 - self.face_lower_0),
                "minimum_positive_gap": float(self.minimum_positive_gap),
                "body_half_height": float(np.max(np.abs(x[:, 2]))),
                "body_half_width": float(np.max(np.linalg.norm(x[:, :2], axis=1))),
                "min_element_jacobian": float(np.min(jacobian)),
                "max_element_jacobian": float(np.max(jacobian)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "linear-tetrahedron neo-Hookean FEM with volume projection",
            "nodes": int(self.n_nodes),
            "elements": int(self.n_elements),
            "substep_dt": float(self.substep_dt),
            "substeps_taken": int(self._steps_taken),
            "shear_modulus": float(self.mu),
            "first_lame_parameter": float(self.lam),
            "damping": float(self.damping),
            "plate_speed_each": float(self.plate_speed),
            "config_speed_each": float(self.config["plates"]["speed_each"]),
            "final_plate_gap": float(self.face_upper_1 - self.face_lower_1),
            "mesh_rest_volume": float(self.mesh_rest_volume),
            "declared_rest_volume": float(self.rest_volume_declared),
            "max_relative_volume_error": float(self._max_volume_error),
            "min_element_jacobian_seen": float(self._min_jacobian),
            "nonfinite_resets": int(self._resets),
            "seed": int(self._seed),
        }

