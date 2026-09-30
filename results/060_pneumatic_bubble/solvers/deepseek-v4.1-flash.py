"""Physically based explicit-dynamics solver for the pneumatic bubble chamber.

Physical model
--------------
The deforming rubber wall is discretised as a three-dimensional solid: the
frozen axisymmetric chamber surface is interpreted as two concentric rings
(outer / inner surface of the wall) and turned into a structured hexahedral
finite-element mesh of the *volumetric* wall.  A nearly incompressible
Neo-Hookean material (density, Young's modulus and Poisson ratio taken from the
frozen configuration) is integrated in a total-Lagrangian formulation with
2x2x2 Gauss quadrature for the isochoric response and a mean-dilatation
(element-centre J) treatment of the volumetric response to avoid volumetric
locking.

The prescribed internal pressure is applied as a follower traction on the
cavity-facing element faces, ramped over the configured window and held.  The
base and rim rings (``fixed`` vertex attribute) are Dirichlet boundaries.

Time integration is an explicit symplectic-Euler scheme with a consistent
lumped mass matrix, a CFL-stable step size and mass-proportional (Rayleigh)
damping that lets the chamber settle onto the pressurised equilibrium.  All
exported quantities are raw physical state: nodal mass, per-element deformation
gradient, cumulative mass source, and the per-domain/per-term external force
budget (pressure traction, material viscous traction, support reaction).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from worldsolver.contracts import StateFrame

_HEX_SIGNS = np.array(
    [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
_HEX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
_CFL = 0.4


def _inv3(a: np.ndarray):
    """Vectorised inverse/determinant of stacked 3x3 matrices."""
    a0 = a[..., 0, :]
    a1 = a[..., 1, :]
    a2 = a[..., 2, :]
    b0 = np.cross(a1, a2)
    b1 = np.cross(a2, a0)
    b2 = np.cross(a0, a1)
    det = np.sum(a0 * b0, axis=-1)
    safe = np.where(np.abs(det) < 1e-30, 1e-30, det)
    inv = np.stack((b0, b1, b2), axis=-1) / safe[..., None, None]
    return inv, det


def _gauss_hex():
    g = 1.0 / np.sqrt(3.0)
    pts = np.array([[sx * g, sy * g, sz * g] for sx, sy, sz in _HEX_SIGNS])
    return pts, np.ones(8)


def _shape_data(pts: np.ndarray):
    """Trilinear shape-function values and natural derivatives at ``pts``."""
    n_pts = pts.shape[0]
    values = np.empty((n_pts, 8))
    derivs = np.empty((n_pts, 8, 3))
    for gi, (xi, eta, zeta) in enumerate(pts):
        for a, (sx, sy, sz) in enumerate(_HEX_SIGNS):
            values[gi, a] = 0.125 * (1 + sx * xi) * (1 + sy * eta) * (1 + sz * zeta)
            derivs[gi, a, 0] = 0.125 * sx * (1 + sy * eta) * (1 + sz * zeta)
            derivs[gi, a, 1] = 0.125 * sy * (1 + sx * xi) * (1 + sz * zeta)
            derivs[gi, a, 2] = 0.125 * sz * (1 + sx * xi) * (1 + sy * eta)
    return values, derivs


class BenchmarkSolver:
    """Explicit finite-element solver for the pressurised chamber wall."""

    def __init__(self) -> None:
        self.ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = getattr(scene, "config", None) or {}

        initial = scene.initial_state()
        rest = np.ascontiguousarray(initial.position, dtype=np.float64)
        self.rest = rest
        self.faces = np.ascontiguousarray(np.asarray(initial.fields["surface_faces"]), dtype=np.int32)
        fixed_attr = np.asarray(initial.attributes["fixed"], dtype=np.float64)
        self.fixed_attr = fixed_attr.copy()
        n_nodes = rest.shape[0]

        n_theta, n_rings = self._detect_lattice(rest, fixed_attr)
        self.n_theta = n_theta
        self.n_rings = n_rings

        elements = self._build_elements(n_theta, n_rings)
        self.elements = elements
        self.n_elements = elements.shape[0]
        self.element_flat = elements.ravel()
        self.n_nodes = n_nodes
        self.scatter_nodes = csr_matrix(
            (
                np.ones(self.element_flat.size),
                (self.element_flat, np.arange(self.element_flat.size)),
            ),
            shape=(n_nodes, self.element_flat.size),
        )

        material = config.get("material_reference", {}) or {}
        self.rho = float(material.get("density_kg_m3", 1070.0))
        young = float(material.get("youngs_modulus_pa", 100000.0))
        nu = float(material.get("poisson_ratio", 0.499))
        nu = min(max(nu, -0.999), 0.4999)
        self.mu = young / (2.0 * (1.0 + nu))
        self.lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        loading = config.get("loading", {}) or {}
        self.pressure_max = float(loading.get("pressure_pa", 4000.0))
        self.ramp_start = float(loading.get("ramp_start_s", 0.25))
        self.ramp_end = float(loading.get("ramp_end_s", 2.5))

        self._build_quadrature(rest, elements)
        self._build_mass_and_boundary(rest, elements, fixed_attr, n_nodes)
        self._build_pressure_faces(rest, elements)

        characteristic = float(np.sqrt(self.mu / self.rho))
        length_scale = max(float(np.ptp(rest[:, 2])), float(np.max(np.linalg.norm(rest[:, :2], axis=1))))
        self.omega_ref = characteristic / max(length_scale, 1e-9)
        self.alpha = 2.0 * self.omega_ref

        c_p = float(np.sqrt((self.lam + 2.0 * self.mu) / self.rho))
        self.dt = _CFL * self.h_min / c_p
        self.reference_force = abs(self.pressure_max) * np.pi * float(
            np.max(np.linalg.norm(rest[:, :2], axis=1))
        ) ** 2
        self.reference_force = float(max(self.reference_force, 1e-6))

        self.x = rest.copy()
        self.v = np.zeros_like(rest)
        self.sim_time = 0.0
        self.steps = 0
        self.pressure_now = 0.0
        self.pressure_force = np.zeros(3)
        self.viscous_force = np.zeros(3)
        self.reaction_force = np.zeros(3)
        self.max_displacement = 0.0
        self.ready = True

    def _detect_lattice(self, rest: np.ndarray, fixed_attr: np.ndarray):
        """Infer (n_theta, n_rings) of the frozen two-surface ring lattice."""
        n = rest.shape[0]
        n_fixed = int(np.count_nonzero(fixed_attr > 0.5))
        candidates = []
        if n_fixed >= 4 and n_fixed % 4 == 0:
            candidates.append(n_fixed // 4)
        for value in (32, 64, 16, 48, 24, 96, 40, 20, 12, 8):
            if value not in candidates:
                candidates.append(value)
        radius = np.linalg.norm(rest[:, :2], axis=1)
        height = rest[:, 2]
        for n_theta in candidates:
            if n_theta < 3 or n % (2 * n_theta) != 0:
                continue
            n_rings = n // (2 * n_theta)
            if n_rings < 2:
                continue
            if self._lattice_consistent(radius, height, n_theta, n_rings):
                return int(n_theta), int(n_rings)
        raise ValueError("Could not infer the axisymmetric chamber lattice from the rest state")

    @staticmethod
    def _lattice_consistent(radius, height, n_theta: int, n_rings: int) -> bool:
        for side in (0, 1):
            base = side * n_rings * n_theta
            for j in range(n_rings):
                ring_r = radius[base + j * n_theta: base + (j + 1) * n_theta]
                ring_z = height[base + j * n_theta: base + (j + 1) * n_theta]
                if ring_r.max() - ring_r.min() > 1e-6 * max(1e-9, ring_r.max()):
                    return False
                if ring_z.max() - ring_z.min() > 1e-9 * max(1e-9, abs(ring_z[0])):
                    return False
        for j in range(n_rings):
            z_out = height[j * n_theta]
            z_in = height[n_rings * n_theta + j * n_theta]
            if abs(z_out - z_in) > 1e-9 * max(1.0, abs(z_out)):
                return False
        return True

    @staticmethod
    def _build_elements(n_theta: int, n_rings: int) -> np.ndarray:
        inner_base = n_rings * n_theta
        elements = np.empty(((n_rings - 1) * n_theta, 8), dtype=np.int64)
        e = 0
        for j in range(n_rings - 1):
            lower = j * n_theta
            upper = (j + 1) * n_theta
            for i in range(n_theta):
                k = (i + 1) % n_theta
                elements[e] = (
                    lower + i,
                    lower + k,
                    upper + k,
                    upper + i,
                    inner_base + lower + i,
                    inner_base + lower + k,
                    inner_base + upper + k,
                    inner_base + upper + i,
                )
                e += 1
        return elements

    def _build_quadrature(self, rest: np.ndarray, elements: np.ndarray) -> None:
        points, weights = _gauss_hex()
        _, derivs = _shape_data(points)
        centre_derivs = 0.125 * _HEX_SIGNS
        rest_elements = rest[elements]

        jac = np.einsum("eai,gaj->egij", rest_elements, derivs)
        inv_jac, det_jac = _inv3(jac)
        if np.any(det_jac <= 0.0):
            raise ValueError("Chamber lattice produced non-positive element Jacobians")
        self.B = np.einsum("gaj,egji->egai", derivs, inv_jac)
        self.wV0 = det_jac * weights[None, :]

        jac_c = np.einsum("eai,aj->eij", rest_elements, centre_derivs)
        inv_jac_c, det_jac_c = _inv3(jac_c)
        self.Bc = np.einsum("aj,eji->eai", centre_derivs, inv_jac_c)
        self.volume0 = np.sum(self.wV0, axis=1)
        self.total_volume0 = float(np.sum(self.volume0))

        edge_lengths = [
            np.linalg.norm(rest_elements[:, a, :] - rest_elements[:, b, :], axis=1)
            for a, b in _HEX_EDGES
        ]
        self.h_min = float(np.min(np.stack(edge_lengths, axis=1)))

    def _build_mass_and_boundary(self, rest, elements, fixed_attr, n_nodes) -> None:
        points, _ = _gauss_hex()
        values, _ = _shape_data(points)
        contributions = np.einsum("ga,eg->ega", values, self.wV0)
        nodal = self.scatter_nodes @ (self.rho * contributions).ravel()
        self.mass = np.maximum(nodal, 1e-12)
        self.total_mass = float(np.sum(self.mass))
        self.mass_col = self.mass[:, None]
        self.fixed_mask = fixed_attr > 0.5
        self.fixed_idx = np.flatnonzero(self.fixed_mask)

    def _build_pressure_faces(self, rest: np.ndarray, elements: np.ndarray) -> None:
        quads = elements[:, [2, 3, 7, 6]]
        self.pressure_quads = quads
        flat_quads = quads.ravel()
        self.scatter_faces = csr_matrix(
            (
                np.ones(flat_quads.size),
                (flat_quads, np.arange(flat_quads.size)),
            ),
            shape=(self.n_nodes, flat_quads.size),
        )
        pts = rest[quads]
        area_vec = 0.5 * np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]) + 0.5 * np.cross(
            pts[:, 2] - pts[:, 0], pts[:, 3] - pts[:, 0]
        )
        self.rest_area_vector = area_vec
        radial = rest[quads][:, :, :2]
        norms = np.linalg.norm(radial, axis=2, keepdims=True)
        rhat = radial / np.where(norms < 1e-12, 1.0, norms)
        self.pressure_sign = -1.0 if float(np.sum(area_vec[:, :2] * rhat)) > 0.0 else 1.0

    # -------------------------------------------------------------- utilities
    def _pressure_at(self, time: float) -> float:
        if time <= self.ramp_start:
            return 0.0
        if time >= self.ramp_end:
            return self.pressure_max
        span = max(self.ramp_end - self.ramp_start, 1e-12)
        return self.pressure_max * (time - self.ramp_start) / span

    def _element_deformation(self):
        rest_elements = self.rest[self.elements]
        current = self.x[self.elements]
        f_centre = np.einsum("eai,eaj->eij", current, self.Bc)
        _, det_centre = _inv3(f_centre)
        return f_centre, det_centre

    # ------------------------------------------------------------------ steps
    def _step(self, dt: float) -> None:
        x = self.x
        current = x[self.elements]

        f_gauss = np.einsum("eai,egaj->egij", current, self.B)
        inv_f, _ = _inv3(f_gauss)
        f_centre = np.einsum("eai,eaj->eij", current, self.Bc)
        _, det_centre = _inv3(f_centre)
        j_bar = np.maximum(det_centre, 1e-9)
        log_j = np.log(j_bar)[:, None, None]

        first_piola = self.mu * f_gauss + (self.lam * log_j - self.mu) * np.transpose(
            inv_f, (0, 1, 3, 2)
        )
        f_int_e = -np.einsum("egki,egai,eg->eak", first_piola, self.B, self.wV0)
        f_int = self.scatter_nodes @ f_int_e.reshape(-1, 3)

        self.pressure_now = self._pressure_at(self.sim_time + 0.5 * dt)
        quads = x[self.pressure_quads]
        area_vec = 0.5 * np.cross(quads[:, 1] - quads[:, 0], quads[:, 2] - quads[:, 0]) + 0.5 * np.cross(
            quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 0]
        )
        face_force = -self.pressure_now * self.pressure_sign * area_vec / 4.0
        f_ext = self.scatter_faces @ np.repeat(face_force, 4, axis=0)

        f_damp = -self.alpha * self.mass_col * self.v
        total = f_int + f_ext + f_damp

        reaction = np.zeros_like(x)
        reaction[self.fixed_idx] = -total[self.fixed_idx]
        total[self.fixed_idx] = 0.0

        accel = total / self.mass_col
        velocity = self.v + dt * accel
        velocity[self.fixed_idx] = 0.0
        x = x + dt * velocity

        self.x = x
        self.v = velocity
        self.steps += 1
        self.pressure_force = np.sum(f_ext, axis=0)
        self.viscous_force = np.sum(f_damp, axis=0)
        self.reaction_force = np.sum(reaction, axis=0)

    def advance_to(self, time: float):
        if not self.ready:
            raise RuntimeError("initialize() must be called before advance_to()")
        target = float(time)
        if target < self.sim_time:
            target = self.sim_time
        while self.sim_time < target - 1e-12:
            step = min(self.dt, target - self.sim_time)
            self._step(step)
            self.sim_time += step
        self.sim_time = target
        return self._frame(target)

    # ----------------------------------------------------------------- frames
    def _frame(self, time: float) -> StateFrame:
        x = self.x
        f_centre, det_centre = self._element_deformation()

        pressure = self._pressure_at(time)
        budget = np.zeros((2, 3, 3), dtype=np.float64)
        budget[0, 0] = self.pressure_force
        budget[0, 1] = self.viscous_force
        budget[0, 2] = self.reaction_force

        displacement = np.linalg.norm(x - self.rest, axis=1)
        self.max_displacement = float(np.max(displacement))

        frame = StateFrame(
            time=time,
            position=x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(x.shape[0], dtype=np.int32),
            attributes={
                "fixed": self.fixed_attr.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "surface_faces": self.faces,
                "element_deformation_gradient": f_centre,
                "budget_external_force": budget,
                "cumulative_mass_source": np.zeros(3, dtype=np.float64),
                "element_volume": np.sum(self.wV0, axis=1) * det_centre,
            },
            scalars={
                "reference_force": float(self.reference_force),
                "internal_pressure_pa": float(pressure),
                "element_count": float(self.n_elements),
                "time_step_s": float(self.dt),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "nodes": int(self.rest.shape[0]),
            "elements": int(self.n_elements),
            "time_step_s": float(self.dt),
            "integration_steps": int(self.steps),
            "total_mass_kg": float(self.total_mass),
            "rest_volume_m3": float(self.total_volume0),
            "pressure_pa": float(self.pressure_max),
            "max_displacement_m": float(self.max_displacement),
        }
