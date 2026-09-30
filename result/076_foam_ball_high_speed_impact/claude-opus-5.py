"""Explicit finite-element solver for the high-speed foam ball impact.

Physics
-------
The volumetric foam practice ball is discretised as a tetrahedral finite
element mesh built from the frozen lattice of scene samples.  Each element
uses a stable Neo-Hookean strain energy (Smith et al. formulation) plus a
small Kelvin-Voigt viscous stress, so the model stays well behaved under the
very large compressive strains produced by a 67 m/s impact.

The fixed impact plate is a rigid boundary at x = 0 (the -x face of the frozen
plate box).  Contact is resolved with a non-penetration projection: samples
that cross the face are pushed back and their inward normal velocity is
removed.  The projection is one-sided, so material separates from the plate
as soon as the elastic recovery pushes it away, reproducing the
compression / recovery / release sequence.

Time integration is symplectic (semi-implicit) Euler with sub-stepping chosen
from the elastic CFL limit.  All momentum exchanged with the boundary and with
gravity is accumulated exactly, so the exported momentum budget closes.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_CUBE_TETS = (
    (0, 1, 3, 7),
    (0, 1, 5, 7),
    (0, 4, 5, 7),
    (0, 4, 6, 7),
    (0, 2, 6, 7),
    (0, 2, 3, 7),
)


def _build_tetrahedra(points: np.ndarray, spacing: float) -> np.ndarray:
    """Tetrahedralise the frozen cubic lattice of ball samples.

    Interior lattice cubes with all eight corners inside the ball are split
    into six tetrahedra.  Surface samples that belong to no complete cube are
    tied into the mesh with a small non-degenerate tetrahedron built from
    their nearest neighbours, so every sample carries elastic stiffness.
    """
    origin = points.min(axis=0)
    index = np.rint((points - origin) / spacing).astype(np.int64)
    shape = index.max(axis=0) + 1
    lookup = np.full(tuple(int(v) for v in shape), -1, dtype=np.int64)
    lookup[index[:, 0], index[:, 1], index[:, 2]] = np.arange(len(points))

    tets: list[tuple[int, int, int, int]] = []
    covered = np.zeros(len(points), dtype=bool)
    nx, ny, nz = (int(v) for v in shape)
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                corners = [
                    int(lookup[i + dx, j + dy, k + dz])
                    for dx in (0, 1)
                    for dy in (0, 1)
                    for dz in (0, 1)
                ]
                if min(corners) < 0:
                    continue
                for pattern in _CUBE_TETS:
                    tet = tuple(corners[p] for p in pattern)
                    tets.append(tet)
                    covered[list(tet)] = True

    orphans = np.flatnonzero(~covered)
    min_volume = 0.02 * spacing**3
    for node in orphans:
        offsets = points - points[node]
        distance = np.einsum("ij,ij->i", offsets, offsets)
        order = [int(v) for v in np.argsort(distance, kind="stable")[1:9]]
        chosen = None
        best_volume = min_volume
        for a in range(len(order) - 2):
            for b in range(a + 1, len(order) - 1):
                for c in range(b + 1, len(order)):
                    candidate = (int(node), order[a], order[b], order[c])
                    edges = points[list(candidate[1:])] - points[node]
                    volume = abs(float(np.linalg.det(edges))) / 6.0
                    if volume > best_volume:
                        best_volume = volume
                        chosen = candidate
        if chosen is not None:
            tets.append(chosen)
            covered[list(chosen)] = True

    elements = np.asarray(tets, dtype=np.int64)
    # Orient every tetrahedron positively in the reference configuration.
    edges = points[elements[:, 1:]] - points[elements[:, 0]][:, None, :]
    signed = np.linalg.det(edges)
    flip = signed < 0.0
    if np.any(flip):
        elements[flip] = elements[flip][:, [0, 2, 1, 3]]
    return elements
class BenchmarkSolver:
    """Explicit tetrahedral FEM solver with rigid-plate contact."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(scene.config)
        ball = config["foam_ball"]
        plate = config["impact_plate"]
        initial = config.get("initial_conditions", {})
        boundary = config.get("boundary_conditions", {})

        frame = scene.initial_state()
        self.position = np.asarray(frame.position, dtype=np.float64).copy()
        self.velocity = np.asarray(frame.velocity, dtype=np.float64).copy()
        self.rest_position = np.asarray(
            frame.attributes["rest_position"], dtype=np.float64
        ).copy()
        self.mass = np.asarray(frame.mass, dtype=np.float64).copy()
        self.n = int(len(self.position))

        self.radius = float(ball["radius"])
        self.density = float(ball["density"])
        self.youngs = float(ball["youngs_modulus"])
        self.poisson = float(ball["poisson_ratio"])
        self.mu = self.youngs / (2.0 * (1.0 + self.poisson))
        self.lam = (
            self.youngs
            * self.poisson
            / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        )
        self.total_mass = float(np.sum(self.mass))
        self.gravity = np.asarray(
            initial.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64
        )

        # Frozen rigid boundary: the -x face of the impact plate box.
        plate_center = np.asarray(plate["center"], dtype=np.float64)
        plate_half = np.asarray(plate["half_extents"], dtype=np.float64)
        self.plate_x = float(
            boundary.get("plate_contact_surface_x", plate_center[0] - plate_half[0])
        )
        self.plate_center = plate_center
        self.plate_half = plate_half

        spacing = self.radius / 5.0
        self.spacing = spacing
        self.elements = _build_tetrahedra(self.rest_position, spacing)
        self._setup_elements()
        self._setup_time_stepping()

        self.time = 0.0
        # Momentum bookkeeping (single global body domain, G = 1).
        self._impulse_gravity = np.zeros(3, dtype=np.float64)
        self._impulse_contact = np.zeros(3, dtype=np.float64)
        self._force_gravity = self.total_mass * self.gravity
        self._force_contact = np.zeros(3, dtype=np.float64)
        self.reference_momentum = float(
            self.total_mass
            * max(float(np.linalg.norm(np.asarray(ball["initial_velocity"], float))), 1.0)
        )
        self._contact_nodes = 0
        self._step_contact_impulse = np.zeros(3, dtype=np.float64)
        self._min_extent = float(
            self.position[:, 0].max() - self.position[:, 0].min()
        )
        self._max_substeps = 0
    # ------------------------------------------------------------------
    # Discretisation setup
    # ------------------------------------------------------------------
    def _setup_elements(self) -> None:
        rest = self.rest_position
        elements = self.elements
        edges = np.stack(
            [
                rest[elements[:, 1]] - rest[elements[:, 0]],
                rest[elements[:, 2]] - rest[elements[:, 0]],
                rest[elements[:, 3]] - rest[elements[:, 0]],
            ],
            axis=2,
        )  # [Ne,3,3] with material edge vectors as columns
        self.rest_shape = edges
        self.rest_shape_inv = np.linalg.inv(edges)
        self.rest_volume = np.abs(np.linalg.det(edges)) / 6.0
        self.ne = int(len(elements))
        self.identity = np.broadcast_to(np.eye(3), (self.ne, 3, 3)).copy()

    def _setup_time_stepping(self) -> None:
        wave_speed = float(np.sqrt((self.lam + 2.0 * self.mu) / self.density))
        self.wave_speed = wave_speed
        cfl_dt = self.spacing / max(wave_speed, 1e-12)
        self.dt_max = 0.15 * cfl_dt
        self.sample_dt = float(self.scene.config["sample_dt"])
        # Kelvin-Voigt viscosity keeps the very high frequency modes of the
        # explicit scheme bounded without visibly damping the rebound.
        self.viscosity = 5.0e1

    # ------------------------------------------------------------------
    # Elastic response
    # ------------------------------------------------------------------
    def _internal_forces(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        elements = self.elements
        p0 = position[elements[:, 0]]
        ds = np.stack(
            [
                position[elements[:, 1]] - p0,
                position[elements[:, 2]] - p0,
                position[elements[:, 3]] - p0,
            ],
            axis=2,
        )
        gradient = ds @ self.rest_shape_inv  # deformation gradient F [Ne,3,3]

        v0 = velocity[elements[:, 0]]
        dv = np.stack(
            [
                velocity[elements[:, 1]] - v0,
                velocity[elements[:, 2]] - v0,
                velocity[elements[:, 3]] - v0,
            ],
            axis=2,
        )
        gradient_rate = dv @ self.rest_shape_inv  # dF/dt

        piola = self._first_piola(gradient) + self.viscosity * gradient_rate
        # H = -V * P * Dm^{-T} : forces on the three edge nodes.
        forces = -self.rest_volume[:, None, None] * (
            piola @ np.transpose(self.rest_shape_inv, (0, 2, 1))
        )
        out = np.zeros_like(position)
        for column in range(3):
            np.add.at(out, elements[:, column + 1], forces[:, :, column])
        np.add.at(out, elements[:, 0], -forces.sum(axis=2))
        self._last_gradient = gradient
        return out

    def _first_piola(self, gradient: np.ndarray) -> np.ndarray:
        """Stable Neo-Hookean first Piola-Kirchhoff stress.

        Psi = mu/2 (tr(F^T F) - 3) - mu (J - 1) + lam/2 (J - 1)^2
        P   = mu F + (lam (J - 1) - mu) dJ/dF
        with dJ/dF built from column cross products so inverted elements stay
        well defined.
        """
        f0 = gradient[:, :, 0]
        f1 = gradient[:, :, 1]
        f2 = gradient[:, :, 2]
        cofactor = np.stack(
            [np.cross(f1, f2), np.cross(f2, f0), np.cross(f0, f1)], axis=2
        )
        jacobian = np.einsum("ij,ij->i", f0, np.cross(f1, f2))
        coefficient = self.lam * (jacobian - 1.0) - self.mu
        return self.mu * gradient + coefficient[:, None, None] * cofactor
    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    def _apply_contact(self, dt: float) -> np.ndarray:
        """Project the rigid impact-plate constraint and return its impulse."""
        inside = self.position[:, 0] > self.plate_x
        if not np.any(inside):
            self._contact_nodes = 0
            return np.zeros(3, dtype=np.float64)
        # The plate face is far wider than the ball, but keep the frozen
        # in-plane extents explicit so the constraint matches the geometry.
        offset = self.position[:, 1:] - self.plate_center[1:]
        within = np.all(np.abs(offset) <= self.plate_half[1:], axis=1)
        active = inside & within
        if not np.any(active):
            self._contact_nodes = 0
            return np.zeros(3, dtype=np.float64)

        old = self.velocity[active].copy()
        self.position[active, 0] = self.plate_x
        normal_velocity = self.velocity[active, 0]
        self.velocity[active, 0] = np.minimum(normal_velocity, 0.0)
        impulse = np.einsum(
            "i,ij->j", self.mass[active], self.velocity[active] - old
        )
        self._contact_nodes = int(np.count_nonzero(active))
        return impulse

    def _substep(self, dt: float) -> None:
        forces = self._internal_forces(self.position, self.velocity)
        forces += self.mass[:, None] * self.gravity[None, :]
        self.velocity += dt * forces / self.mass[:, None]
        self.position += dt * self.velocity
        contact_impulse = self._apply_contact(dt)
        self._impulse_gravity += dt * self._force_gravity
        self._impulse_contact += contact_impulse
        self._step_contact_impulse += contact_impulse

    def _advance_interval(self, target: float) -> None:
        remaining = target - self.time
        if remaining <= 0.0:
            return
        steps = max(int(np.ceil(remaining / self.dt_max)), 1)
        dt = remaining / steps
        self._max_substeps = max(self._max_substeps, steps)
        self._step_contact_impulse = np.zeros(3, dtype=np.float64)
        for _ in range(steps):
            self._substep(dt)
        self._force_contact = self._step_contact_impulse / remaining
        self.time = target
        extent = float(self.position[:, 0].max() - self.position[:, 0].min())
        self._min_extent = min(self._min_extent, extent)
    # ------------------------------------------------------------------
    # Contract surface
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1e-15:
            raise ValueError("advance_to must be called with non-decreasing times")
        self._advance_interval(target)
        return self._make_frame(target)

    def _deformation_gradient(self) -> np.ndarray:
        elements = self.elements
        p0 = self.position[elements[:, 0]]
        ds = np.stack(
            [
                self.position[elements[:, 1]] - p0,
                self.position[elements[:, 2]] - p0,
                self.position[elements[:, 3]] - p0,
            ],
            axis=2,
        )
        return np.ascontiguousarray(ds @ self.rest_shape_inv)

    def _make_frame(self, time: float) -> StateFrame:
        position = np.ascontiguousarray(self.position.copy())
        velocity = np.ascontiguousarray(self.velocity.copy())
        mass = np.ascontiguousarray(self.mass.copy())
        gradient = self._deformation_gradient()

        # One global budget domain covering the single deformable body.
        domain_weights = np.ones((1, self.n), dtype=np.float64)
        external_force = np.stack(
            [np.stack([self._force_gravity, self._force_contact], axis=0)], axis=0
        )
        cumulative_impulse = (self._impulse_gravity + self._impulse_contact)[None, :]
        zero_flux = np.zeros((1, 3), dtype=np.float64)

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=mass,
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "rest_position": np.ascontiguousarray(self.rest_position.copy()),
                "mass_quantity": mass.copy(),
            },
            fields={
                "budget_domain_weights": domain_weights,
                "budget_external_force": np.ascontiguousarray(external_force),
                "budget_cumulative_external_impulse": np.ascontiguousarray(
                    cumulative_impulse
                ),
                "budget_cumulative_inflow_momentum": zero_flux.copy(),
                "budget_cumulative_outflow_momentum": zero_flux.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "element_deformation_gradient": gradient,
                "elastic_reference_position": np.ascontiguousarray(
                    self.rest_position.copy()
                ),
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "total_mass": float(self.total_mass),
                "radius": float(self.radius),
                "contact_sample_count": float(self._contact_nodes),
                "plate_contact_surface_x": float(self.plate_x),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit tetrahedral FEM, stable Neo-Hookean, rigid plate contact",
            "samples": int(self.n),
            "elements": int(self.ne),
            "wave_speed": float(self.wave_speed),
            "substep_dt_limit": float(self.dt_max),
            "max_substeps_per_sample": int(self._max_substeps),
            "lame_mu": float(self.mu),
            "lame_lambda": float(self.lam),
            "min_travel_extent": float(self._min_extent),
            "final_mean_velocity_x": float(
                np.average(self.velocity[:, 0], weights=self.mass)
            ),
        }
