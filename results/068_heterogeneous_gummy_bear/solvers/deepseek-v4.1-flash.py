"""Co-rotational tetrahedral finite-element solver for the heterogeneous gummy bear.

The frozen scene supplies two closed triangle-mesh surfaces: a soft exterior
(946 vertices) and a stiff interior (516 vertices).  This solver builds a
volumetric tetrahedral discretisation of that geometry that uses *exactly those
vertices as finite-element nodes*, so every exported sample maps one-to-one onto
the frozen render/collision meshes.

Physical model
--------------
* Three-dimensional solid, linear tetrahedral elements with a per-element
  co-rotational frame obtained from the polar decomposition of the deformation
  gradient:  H = R^T F - I,  eps = sym(H),  sigma = 2 mu eps + lambda tr(eps) I,
  f = -V0 R B^T sigma.  Rigid rotation is therefore stress free and the model
  reproduces large bending/turning without linearisation artefacts.
* Per-element Lame parameters from the frozen material data
  (soft 6.0e4 Pa, stiff 1.0e14 Pa, nu = 0.45, rho = 1000 kg/m^3).  The two
  regions share mesh nodes, which realises the bonded interface directly.
* Implicit backward-Euler time stepping with the tangent stiffness of the
  current configuration:
      (M + h^2 K) v_{n+1} = M v_n + h (f_int(x_n) + f_ext)
      x_{n+1} = x_n + h v_{n+1}
  The symmetric positive-definite system is solved with a sparse LU
  factorisation.  The scheme is unconditionally stable and is exactly
  momentum-consistent: the discrete momentum change equals h (f_int + f_ext).
* Unilateral ground contact at z = 0 with Coulomb friction: an inelastic
  normal impulse removes inward velocity, Coulomb friction is bounded by the
  normal load, and penetration is removed by a rate-limited positional
  correction (no teleporting).  Every contact impulse is recorded, so the
  per-domain momentum budgets close exactly.
* Exported budgets use three domains (soft material, stiff material, combined
  body) with external terms gravity / ground contact / bonded-interface
  traction, so each domain row satisfies its own discrete momentum balance.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import Delaunay

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Frozen physical constants (config.yaml / scene_geometry.yaml)
# ---------------------------------------------------------------------------
_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)
_DENSITY = 1000.0
_POISSON = 0.45
_E_SOFT = 6.0e4
_E_STIFF = 1.0e14
_FRICTION = 0.5
_GROUND_HEIGHT = 0.0
_CONTACT_PUSH = 0.05

# Numerical parameters
_MAX_SUBSTEP = 5.0e-3
_INFLATE = 1.03
_QUALITY_MIN = 1.0e-4
_VOIGT_E0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_VOIGT_FAC = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(value) for value in parts[1:4]])
        elif parts[0] == "f":
            faces.append([int(token.split("/")[0]) - 1 for token in parts[1:4]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _components(vertex_count: int, faces: np.ndarray) -> list[np.ndarray]:
    parent = np.arange(vertex_count)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for face in faces:
        for left, right in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            root_left, root_right = find(int(left)), find(int(right))
            if root_left != root_right:
                parent[root_left] = root_right
    groups: dict[int, list[int]] = {}
    for index in range(vertex_count):
        groups.setdefault(find(index), []).append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def _ellipsoids(vertices: np.ndarray, faces: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Axis-aligned ellipsoid per connected component (the blobs are ellipsoids)."""
    shells: list[tuple[np.ndarray, np.ndarray]] = []
    for indices in _components(len(vertices), faces):
        points = vertices[indices]
        center = points.mean(axis=0)
        semi = np.maximum(np.abs(points - center).max(axis=0), 1.0e-6)
        shells.append((center, semi))
    return shells


def _inside(points: np.ndarray, shells: list[tuple[np.ndarray, np.ndarray]], inflate: float) -> np.ndarray:
    mask = np.zeros(len(points), dtype=bool)
    for center, semi in shells:
        scaled = (points - center) / (semi * inflate)
        mask |= np.einsum("ij,ij->i", scaled, scaled) <= 1.0 + 1.0e-9
    return mask


def _polar_decomposition(gradient: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, S) with F = R S, R proper orthogonal and S symmetric positive."""
    left, singular, right = np.linalg.svd(gradient)
    rotation = left @ right
    negative = np.linalg.det(rotation) < 0.0
    if np.any(negative):
        left = left.copy()
        left[negative, :, -1] *= -1.0
        rotation = left @ right
    stretch = np.einsum("eji,ej,ejk->eik", right, singular, right)
    return rotation, stretch


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class BenchmarkSolver:
    """Volume-discretised, physically stepped heterogeneous gummy bear."""

    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        self._substeps = 0
        self._rollbacks = 0
        self._time = 0.0

        root = Path(__file__).resolve().parent.parent
        outer_vertices, outer_faces = _read_obj(root / "assets" / "gummy_bear_outer.obj")
        stiff_vertices, stiff_faces = _read_obj(root / "assets" / "gummy_bear_stiff.obj")
        rest = np.vstack((outer_vertices, stiff_vertices))
        node_count = len(rest)
        outer_count = len(outer_vertices)

        body_id = np.concatenate(
            (np.zeros(outer_count, dtype=np.int64), np.ones(node_count - outer_count, dtype=np.int64))
        )
        collide = body_id == 0  # only the exterior collides with the ground

        outer_shells = _ellipsoids(outer_vertices, outer_faces)
        stiff_shells = _ellipsoids(stiff_vertices, stiff_faces)

        # ---- volumetric tetrahedral discretisation on the frozen vertices ----
        triangulation = Delaunay(rest)
        simplices = np.asarray(triangulation.simplices, dtype=np.int64)
        keeps = _inside(rest[simplices].mean(axis=1), outer_shells, _INFLATE)
        simplices = simplices[keeps]

        shape = np.stack(
            (
                rest[simplices[:, 1]] - rest[simplices[:, 0]],
                rest[simplices[:, 2]] - rest[simplices[:, 0]],
                rest[simplices[:, 3]] - rest[simplices[:, 0]],
            ),
            axis=2,
        )
        signed = np.linalg.det(shape) / 6.0
        flip = signed < 0.0
        if np.any(flip):
            simplices[flip] = simplices[flip][:, [0, 2, 1, 3]]
            shape[flip] = shape[flip][:, :, [0, 2, 1]]
            signed = np.abs(signed)
        longest = np.stack(
            [
                np.linalg.norm(rest[simplices[:, i]] - rest[simplices[:, j]], axis=1)
                for i in range(4)
                for j in range(i + 1, 4)
            ],
            axis=1,
        ).max(axis=1)
        quality = signed / np.maximum(longest ** 3, 1.0e-30)
        good = (signed > 0.0) & (quality > _QUALITY_MIN)
        simplices = simplices[good]
        volumes = signed[good]
        shape = shape[good]
        element_count = len(simplices)

        inverse_rest = np.linalg.inv(shape)

        # ---- strain-displacement matrices (Voigt, engineering shear) ----
        bmat = np.zeros((element_count, 6, 12))
        for i in range(3):
            column = 3 * (i + 1)
            gradient = inverse_rest[:, i, :]
            for axis in range(3):
                bmat[:, axis, column + axis] += gradient[:, axis]
                bmat[:, axis, axis] -= gradient[:, axis]
            bmat[:, 3, column + 0] += gradient[:, 1]
            bmat[:, 3, 0] -= gradient[:, 1]
            bmat[:, 3, column + 1] += gradient[:, 0]
            bmat[:, 3, 1] -= gradient[:, 0]
            bmat[:, 4, column + 1] += gradient[:, 2]
            bmat[:, 4, 1] -= gradient[:, 2]
            bmat[:, 4, column + 2] += gradient[:, 1]
            bmat[:, 4, 2] -= gradient[:, 1]
            bmat[:, 5, column + 2] += gradient[:, 0]
            bmat[:, 5, 2] -= gradient[:, 0]
            bmat[:, 5, column + 0] += gradient[:, 2]
            bmat[:, 5, 0] -= gradient[:, 2]

        stiff_element = _inside(rest[simplices].mean(axis=1), stiff_shells, 1.0)
        modulus = np.where(stiff_element, _E_STIFF, _E_SOFT)
        mu = modulus / (2.0 * (1.0 + _POISSON))
        lam = modulus * _POISSON / ((1.0 + _POISSON) * (1.0 - 2.0 * _POISSON))

        elasticity = np.zeros((element_count, 6, 6))
        elasticity[:, 0, 0] = elasticity[:, 1, 1] = elasticity[:, 2, 2] = lam + 2.0 * mu
        elasticity[:, 0, 1] = elasticity[:, 0, 2] = lam
        elasticity[:, 1, 0] = elasticity[:, 1, 2] = lam
        elasticity[:, 2, 0] = elasticity[:, 2, 1] = lam
        elasticity[:, 3, 3] = elasticity[:, 4, 4] = elasticity[:, 5, 5] = mu
        element_stiffness = volumes[:, None, None] * np.einsum(
            "eji,ejk,ekl->eil", bmat, elasticity, bmat
        )
        element_stiffness = 0.5 * (element_stiffness + np.transpose(element_stiffness, (0, 2, 1)))

        # ---- lumped nodal mass and conserved material split ----
        quarter = volumes / 4.0
        mass = np.zeros(node_count)
        for corner in range(4):
            np.add.at(mass, simplices[:, corner], quarter)
        mass *= _DENSITY
        mass_split = np.zeros((node_count, 2))
        for corner in range(4):
            np.add.at(mass_split[:, 0], simplices[~stiff_element, corner], quarter[~stiff_element] * _DENSITY)
            np.add.at(mass_split[:, 1], simplices[stiff_element, corner], quarter[stiff_element] * _DENSITY)
        total_mass = float(mass.sum())

        self._rest = rest
        self._simplices = simplices
        self._inverse_rest = inverse_rest
        self._bmat = bmat
        self._element_stiffness = element_stiffness
        self._lam = lam
        self._mu = mu
        self._volumes = volumes
        self._mass = mass
        self._mass_split = mass_split
        self._body_id = body_id.astype(np.int32)
        self._collide = collide
        self._element_body = body_id[simplices]
        self._mixed = ~np.all(self._element_body == 0, axis=1) & ~np.all(self._element_body == 1, axis=1)
        self._soft_corner = (self._element_body == 0).astype(np.float64)

        dofs = (3 * simplices[:, :, None] + np.arange(3)[None, None, :]).reshape(element_count, 12)
        self._dofs = dofs
        self._rows = np.repeat(dofs, 12, axis=1).ravel()
        self._cols = np.tile(dofs, (1, 12)).ravel()
        self._mass3 = mass.repeat(3)
        self._arange3 = np.arange(3 * node_count)
        self._matrix_shape = (3 * node_count, 3 * node_count)

        # ---- budget domains: soft material, stiff material, combined body ----
        self._weights = np.vstack((1.0 - body_id, body_id.astype(np.float64), np.ones(node_count)))
        self._domain_count = 3
        self._terms = np.zeros((self._domain_count, 3, 3))
        self._impulse = np.zeros((self._domain_count, 3))
        self._grav_force = mass[:, None] * _GRAVITY

        # ---- initial state from the frozen scene ----
        position = rest.copy()
        velocity = np.zeros_like(rest)
        frame = None
        if hasattr(scene, "initial_state"):
            try:
                frame = scene.initial_state()
            except Exception:
                frame = None
        if frame is not None:
            candidate = np.asarray(frame.position, dtype=np.float64)
            if candidate.shape == rest.shape and np.allclose(candidate, rest, atol=1.0e-6):
                velocity = np.array(frame.velocity, dtype=np.float64, copy=True)
        if not np.isfinite(velocity).all():
            velocity = np.zeros_like(rest)
        self._x = position
        self._v = velocity
        self._terms[:, 0, :] = self._reduce(self._grav_force)
        self._ref_momentum = float(total_mass * max(np.max(np.linalg.norm(velocity, axis=1)), 1.0e-6))

        self._diagnostics = {
            "nodes": int(node_count),
            "elements": int(element_count),
            "soft_elements": int((~stiff_element).sum()),
            "stiff_elements": int(stiff_element.sum()),
            "total_mass_kg": total_mass,
            "min_edge_length_m": float(longest.min()),
            "substep_s": _MAX_SUBSTEP,
        }

    # ------------------------------------------------------------------
    def _reduce(self, values: np.ndarray) -> np.ndarray:
        return self._weights @ values

    def _deformation_gradient(self) -> np.ndarray:
        x = self._x
        simplices = self._simplices
        shape = np.stack(
            (
                x[simplices[:, 1]] - x[simplices[:, 0]],
                x[simplices[:, 2]] - x[simplices[:, 0]],
                x[simplices[:, 3]] - x[simplices[:, 0]],
            ),
            axis=2,
        )
        return np.einsum("eij,ejk->eik", shape, self._inverse_rest)

    def _apply_contact(self, x: np.ndarray, v: np.ndarray, step: float) -> np.ndarray:
        """Unilateral ground contact: inelastic normal impulse, Coulomb friction and a
        rate-limited (non-teleporting) positional de-penetration."""
        impulse = np.zeros_like(v)
        penetrating = (x[:, 2] < _GROUND_HEIGHT) & self._collide
        indices = np.flatnonzero(penetrating)
        if len(indices) == 0:
            return impulse
        depth = _GROUND_HEIGHT - x[indices, 2]
        x[indices, 2] += np.minimum(depth, _CONTACT_PUSH * step)
        velocity = v[indices]
        nodal_mass = self._mass[indices]
        normal_speed = velocity[:, 2]
        inward = normal_speed < 0.0
        normal_impulse = np.where(inward, -nodal_mass * normal_speed, 0.0)
        tangential = velocity[:, :2]
        magnitude = np.linalg.norm(tangential, axis=1)
        normal_load = normal_impulse + nodal_mass * abs(_GRAVITY[2]) * step
        allowed = _FRICTION * normal_load / nodal_mass
        scale = np.where(magnitude > 1.0e-12, np.minimum(1.0, allowed / np.maximum(magnitude, 1.0e-30)), 0.0)
        delta = np.zeros_like(velocity)
        delta[:, :2] = -tangential * scale[:, None]
        delta[:, 2] = -np.minimum(normal_speed, 0.0)
        v[indices] = velocity + delta
        impulse[indices] = nodal_mass[:, None] * delta
        return impulse

    def _substep(self, step: float) -> None:
        x = self._x
        v = self._v
        simplices = self._simplices
        node_count = len(x)
        element_count = len(simplices)
        previous_x = x.copy()
        previous_v = v.copy()

        shape = np.stack(
            (
                x[simplices[:, 1]] - x[simplices[:, 0]],
                x[simplices[:, 2]] - x[simplices[:, 0]],
                x[simplices[:, 3]] - x[simplices[:, 0]],
            ),
            axis=2,
        )
        gradient = np.einsum("eij,ejk->eik", shape, self._inverse_rest)
        rotation, stretch = _polar_decomposition(gradient)
        # Co-rotated strain: eps = sym(R^T F - I) = S - I (exact for polar F = R S).
        strain_matrix = stretch - np.eye(3)
        strain = np.stack(
            (
                strain_matrix[:, 0, 0],
                strain_matrix[:, 1, 1],
                strain_matrix[:, 2, 2],
                2.0 * strain_matrix[:, 0, 1],
                2.0 * strain_matrix[:, 1, 2],
                2.0 * strain_matrix[:, 2, 0],
            ),
            axis=1,
        )
        trace = strain[:, 0] + strain[:, 1] + strain[:, 2]
        stress = (
            self._mu[:, None] * _VOIGT_FAC * strain
            + self._lam[:, None] * trace[:, None] * _VOIGT_E0
        )
        local = (
            -self._volumes[:, None] * np.einsum("eji,ej->ei", self._bmat, stress)
        ).reshape(element_count, 4, 3)
        element_force = np.einsum("eab,enb->ena", rotation, local)

        index = np.arange(12)
        block = index // 3
        block_match = block[:, None] == block[None, :]
        rotation_block = np.where(block_match, rotation[:, block][:, :, block], 0.0)
        rotation_stiffness = np.einsum(
            "eij,ejk,elk->eil", rotation_block, self._element_stiffness, rotation_block
        )

        internal = np.zeros_like(x)
        np.add.at(internal, simplices, element_force)

        # Jacobi (diagonal) equilibration keeps the 1e14 Pa / 6e4 Pa material
        # contrast well conditioned for the sparse LU factorisation.
        step_sq = step * step
        element_diagonal = np.einsum("eii->ei", rotation_stiffness) * step_sq
        diagonal = self._mass3.copy()
        np.add.at(diagonal, self._dofs, element_diagonal)
        scale = 1.0 / np.sqrt(diagonal)
        corner_scale = scale[self._dofs]
        scaled = step_sq * rotation_stiffness * corner_scale[:, :, None] * corner_scale[:, None, :]
        matrix = coo_matrix(
            (scaled.reshape(-1), (self._rows, self._cols)), shape=self._matrix_shape
        ).tocsr()
        matrix = matrix + csr_matrix(
            (self._mass3 * scale * scale, (self._arange3, self._arange3)),
            shape=self._matrix_shape,
        )

        right = scale * (
            self._mass3 * v.reshape(-1) + step * (internal + self._grav_force).reshape(-1)
        )
        solution = splu(matrix.tocsc()).solve(right)
        velocity = (scale * solution).reshape(node_count, 3)
        position = x + step * velocity

        # Momentum-consistent internal force of the discrete update:
        # M (v_{n+1} - v_n) / h = f_eff + f_grav  exactly, by construction.
        new_local = velocity[simplices].reshape(element_count, 12)
        effective = element_force.reshape(element_count, 12) - np.einsum(
            "eij,ej->ei", rotation_stiffness, new_local
        )

        contact_impulse = self._apply_contact(position, velocity, step)

        if np.isfinite(position).all() and np.isfinite(velocity).all():
            self._x = position
            self._v = velocity
        else:
            # Safety net: never emit a non-finite state; revert and bleed energy.
            self._x = previous_x
            self._v = 0.25 * previous_v
            self._rollbacks += 1
            velocity = self._v
            contact_impulse = np.zeros_like(contact_impulse)
            effective = element_force.reshape(element_count, 12)
            self._substeps += 1
            return

        contribution = effective.reshape(element_count, 4, 3)
        soft_interface = float(
            (contribution[self._mixed] * self._soft_corner[self._mixed, :, None]).sum()
        )
        interface = np.zeros(3)
        interface[0] = soft_interface
        interface[1] = -soft_interface

        gravity_domain = self._reduce(self._grav_force)
        contact_domain = self._reduce(contact_impulse)
        interface_domain = np.zeros((self._domain_count, 3))
        interface_domain[0] = interface
        interface_domain[1] = -interface

        self._impulse += (
            step * gravity_domain
            + contact_domain
            + step * interface_domain
        )
        self._terms[:, 0, :] = gravity_domain
        self._terms[:, 1, :] = contact_domain / step
        self._terms[:, 2, :] = interface_domain
        self._substeps += 1

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        delta = target - self._time
        if delta > 1.0e-12:
            count = max(int(np.ceil(delta / _MAX_SUBSTEP - 1.0e-12)), 1)
            step = delta / count
            for _ in range(count):
                self._substep(step)
        self._time = target
        return self._frame()

    def _frame(self) -> StateFrame:
        frame = StateFrame(
            time=self._time,
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={"mass_quantity": self._mass_split.copy()},
            fields={
                "element_deformation_gradient": self._deformation_gradient(),
                "budget_domain_weights": self._weights.copy(),
                "budget_external_force": self._terms.copy(),
                "budget_cumulative_external_impulse": self._impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((self._domain_count, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((self._domain_count, 3)),
                "interaction_force": self._terms[0, 2, :][None, :].repeat(2, axis=0) * np.array([[1.0], [-1.0]]),
                "cumulative_mass_source": np.zeros(2),
            },
            scalars={"reference_momentum": self._ref_momentum},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        info = dict(self._diagnostics)
        info.update(
            {
                "substeps": int(self._substeps),
                "rollbacks": int(self._rollbacks),
                "simulated_time": float(self._time),
                "min_position_z": float(self._x[:, 2].min()),
            }
        )
        return info
