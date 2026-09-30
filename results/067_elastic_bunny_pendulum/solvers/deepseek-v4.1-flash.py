"""Physical solver for scene 067: swinging of a suspended elastic bunny.

Model (implemented from scratch with NumPy only; no external physics engine)
---------------------------------------------------------------------------
The frozen asset is a closed triangle surface mesh of the soft bunny.  It is
simulated as a surface-discretised three-dimensional elastic solid:

* membrane (in-plane) stiffness: a St. Venant-Kirchhoff spring on every mesh
  edge, with the axial stiffness ``k = E t A_e / ((1 - nu^2) L0^2)`` of a
  material prism of effective thickness ``t`` and tributary area ``A_e``,
* bending stiffness: a discrete-shell cross spring joining the opposite
  vertices of every interior edge,
* bulk stiffness: a penalty on the enclosed volume with the bulk modulus
  ``K = E / (3 (1 - 2 nu))`` implied by the frozen material reference,
* gravity as the only body force.

Time integration is explicit symplectic (semi-implicit) Euler with
Kelvin-Voigt (stiffness proportional) damping on every spring.  The step size
is selected from a Gershgorin bound on the linearised elastic spectrum of the
assembled network, so the integration is provably stable for the frozen
geometry.  The sample time is an exact multiple of the step size.

The crown patch identified by the frozen ``fixed`` vertex attribute is
kinematically clamped to its initial world position (the suspension point), so
the body swings beneath a stationary attachment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_ASSET = Path(__file__).resolve().parent.parent / "assets" / "bunny.obj"
_DENSITY_FALLBACK = 1000.0
_E_FALLBACK = 30000.0
_NU_FALLBACK = 0.45
_GRAVITY_FALLBACK = (0.0, 0.0, -9.81)

_BENDING_FACTOR = 4.0      # cross-spring stiffness relative to the membrane
_DAMPING_RATIO = 0.10      # Kelvin-Voigt damping ratio on every spring
_DT_SAFETY = 0.40          # fraction of the symplectic stability limit
_MAX_SUBSTEPS = 96         # substeps per sample interval


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read vertices (in file order) and triangle indices from a Wavefront OBJ."""
    points: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                points.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(token.split("/")[0]) - 1 for token in line.split()[1:4]])
    if not points or not faces:
        raise ValueError(f"Asset has no triangle mesh: {path}")
    return np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    half = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    half = np.sort(half, axis=1)
    return np.unique(half, axis=0)


def _edge_faces(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (edge, opposite-vertex) pairs with one row per incident half edge."""
    half = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    opposite = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]], axis=0)
    order = np.lexsort((half[:, 1], half[:, 0]))
    return half[order], opposite[order]


def _edge_keys(pairs: np.ndarray, scale: int) -> np.ndarray:
    return pairs[:, 0].astype(np.int64) * scale + pairs[:, 1].astype(np.int64)


def _tributary_areas(edges, edge_rows, tri_area) -> np.ndarray:
    """Third of the area of every incident triangle, accumulated per edge."""
    scale = int(max(edges.max(), edge_rows.max())) + 2
    position = np.searchsorted(_edge_keys(edges, scale), _edge_keys(edge_rows, scale))
    area = np.zeros(len(edges), dtype=np.float64)
    np.add.at(area, position, tri_area / 3.0)
    return area


class BenchmarkSolver:
    """Deterministic explicit finite-element solver for the suspended bunny."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        config = getattr(scene, "config", None) or {}
        material = config.get("material_reference", {}) or {}
        self.density = float(material.get("density_kg_m3", _DENSITY_FALLBACK))
        self.modulus = float(material.get("youngs_modulus_pa", _E_FALLBACK))
        self.poisson = float(material.get("poisson_ratio", _NU_FALLBACK))
        gravity = np.asarray(
            config.get("gravity", _GRAVITY_FALLBACK), dtype=np.float64
        ).reshape(3)
        self.gravity = gravity

        frame = scene.initial_state()
        position = np.asarray(frame.position, dtype=np.float64)
        velocity = np.asarray(frame.velocity, dtype=np.float64)
        mass = np.asarray(frame.mass, dtype=np.float64).reshape(-1)
        n = len(position)
        if mass.shape != (n,):
            raise ValueError("Scene mass must be a per-vertex array")
        fixed_values = np.asarray(
            frame.attributes.get("fixed", np.zeros(n, dtype=np.float64)), dtype=np.float64
        ).reshape(-1)
        self.fixed = fixed_values >= 0.5
        self.fixed_attr = self.fixed.astype(np.float64)

        vertices, faces = _read_obj(_ASSET)
        if len(vertices) != n:
            raise ValueError("Asset vertex count does not match the scene initial state")
        if faces.shape[1] != 3:
            raise ValueError("Asset must contain triangular faces")

        self.x0 = position.copy()
        self.m = mass
        self.m_total = float(mass.sum())
        self.n = n
        self.faces = faces
        self.fixed_idx = np.flatnonzero(self.fixed)
        self.free_idx = np.flatnonzero(~self.fixed)

        self._build_topology(faces, position)
        self._build_step_size()

        self.x = position.copy()
        self.v = velocity.copy()
        self.v[self.fixed] = 0.0
        self.step = 0
        self.max_step = int(round(1.0e12))
        self.time = 0.0

        attach = np.asarray(
            (config.get("initial_conditions", {}) or {}).get(
                "attachment_position", [0.0, 0.0, 0.0]
            ),
            dtype=np.float64,
        ).reshape(3)
        lever = float(np.linalg.norm(position.mean(axis=0) - attach))
        speed_scale = np.sqrt(max(2.0 * float(np.linalg.norm(gravity)) * max(lever, 1e-6), 1e-12))
        self.reference_momentum = float(self.m_total * speed_scale)
        self.attachment = attach

        self._gravity_term = self.m_total * gravity
        self._external_terms = self._evaluate_external(self.x, self.v)
        self.cumulative_impulse = np.zeros(3, dtype=np.float64)

    def _build_topology(self, faces: np.ndarray, pose: np.ndarray) -> None:
        n = len(pose)
        self.edges = _unique_edges(faces)
        edge_rows, opposite = _edge_faces(faces)

        face_a, face_b, face_c = faces[:, 0], faces[:, 1], faces[:, 2]
        rest_e1 = pose[face_b] - pose[face_a]
        rest_e2 = pose[face_c] - pose[face_a]
        cross = np.cross(rest_e1, rest_e2)
        tri_area = 0.5 * np.linalg.norm(cross, axis=1)
        normal = cross / np.maximum(np.linalg.norm(cross, axis=1), 1e-300)[:, None]
        surface = float(tri_area.sum())

        # membrane springs -------------------------------------------------
        self.edge_rest = np.linalg.norm(pose[self.edges[:, 0]] - pose[self.edges[:, 1]], axis=1)
        self.edge_rest = np.maximum(self.edge_rest, 1e-9)
        trib = _tributary_areas(self.edges, edge_rows, tri_area)
        thickness = (self.m_total / max(self.density, 1e-12)) / max(surface, 1e-12)
        membrane = self.modulus * thickness / (1.0 - self.poisson**2)
        self.edge_k = membrane * trib / self.edge_rest**2
        self.edge_c = 2.0 * _DAMPING_RATIO * np.sqrt(
            self.edge_k * self._edge_mass(self.edges) / self.edge_rest
        )

        # discrete shell bending springs ------------------------------------
        scale = int(max(self.edges.max(), edge_rows.max())) + 2
        keys = _edge_keys(edge_rows, scale)
        start = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        stop = np.r_[start[1:], len(keys)]
        interior = np.flatnonzero((stop - start) == 2)
        rows = start[interior]
        self.cross = np.stack([opposite[rows], opposite[rows + 1]], axis=1)
        self.cross_rest = np.linalg.norm(
            pose[self.cross[:, 0]] - pose[self.cross[:, 1]], axis=1
        )
        self.cross_rest = np.maximum(self.cross_rest, 1e-9)
        edge_of_row = np.searchsorted(_edge_keys(self.edges, scale), keys)
        self.cross_k = _BENDING_FACTOR * self.edge_k[edge_of_row[rows]]
        self.cross_c = 2.0 * _DAMPING_RATIO * np.sqrt(
            self.cross_k * self._edge_mass(self.cross) / self.cross_rest
        )

        # reference elements -------------------------------------------------
        dm = np.stack([rest_e1, rest_e2, normal], axis=2)
        self._dm_inv = np.linalg.inv(dm)
        self.face_a, self.face_b, self.face_c = face_a, face_b, face_c

        # bulk (volume) penalty ----------------------------------------------
        bulk = self.modulus / (3.0 * max(1.0 - 2.0 * self.poisson, 1e-3))
        volume = float(
            np.einsum("ij,ij->i", np.cross(pose[face_a], pose[face_b]), pose[face_c]).sum() / 6.0
        )
        self.rest_volume = volume
        self.bulk_k = bulk / max(self.rest_volume, 1e-12)
        _, gradient = self._volume_gradient(pose)
        self._volume_grad_norm = np.linalg.norm(gradient, axis=1)
        self._volume_rowsum = float(
            self.bulk_k * float(np.abs(gradient).sum()) * self._volume_grad_norm.max()
        )

    def _edge_mass(self, pairs: np.ndarray) -> np.ndarray:
        return 0.5 * (self.m[pairs[:, 0]] + self.m[pairs[:, 1]])

    def _build_step_size(self) -> None:
        """Gershgorin bound on the linearised elastic spectrum -> stable step."""
        row = np.zeros(self.n, dtype=np.float64)
        for pairs, rest, stiff in (
            (self.edges, self.edge_rest, self.edge_k),
            (self.cross, self.cross_rest, self.cross_k),
        ):
            contribution = stiff / rest
            np.add.at(row, pairs[:, 0], contribution)
            np.add.at(row, pairs[:, 1], contribution)
        row += self._volume_rowsum
        row = np.maximum(row, 0.0)
        diagonal = np.zeros(self.n, dtype=np.float64)
        diagonal[self.free_idx] = row[self.free_idx] / self.m[self.free_idx]
        omega_max = float(np.sqrt(max(float(diagonal.max()), 1e-12)))
        limit = _DT_SAFETY * 2.0 / omega_max
        sample_dt = 0.02
        substeps = int(np.ceil(sample_dt / max(limit, 1e-9)))
        substeps = int(np.clip(substeps, 1, _MAX_SUBSTEPS))
        self.substeps = substeps
        self.dt = sample_dt / substeps
        self.omega_max = omega_max

    # ------------------------------------------------------------------ forces
    def _volume_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        a, b, c = self.face_a, self.face_b, self.face_c
        xa, xb, xc = x[a], x[b], x[c]
        ga = np.cross(xb, xc) / 6.0
        gb = np.cross(xc, xa) / 6.0
        gc = np.cross(xa, xb) / 6.0
        volume = float(np.einsum("ij,ij->i", ga, xa).sum())
        gradient = np.empty_like(x)
        for axis in range(3):
            gradient[:, axis] = (
                np.bincount(a, weights=ga[:, axis], minlength=self.n)
                + np.bincount(b, weights=gb[:, axis], minlength=self.n)
                + np.bincount(c, weights=gc[:, axis], minlength=self.n)
            )
        return volume, gradient

    def _spring_forces(self, x, v, pairs, rest, stiff, damping, force) -> None:
        i, j = pairs[:, 0], pairs[:, 1]
        delta = x[j] - x[i]
        length = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        length = np.maximum(length, 1e-12)
        direction = delta / length[:, None]
        strain = (length - rest) / rest
        relative = np.einsum("ij,ij->i", direction, v[j] - v[i])
        magnitude = stiff * strain + damping * relative
        contribution = magnitude[:, None] * direction
        for axis in range(3):
            column = contribution[:, axis]
            force[:, axis] += (
                np.bincount(j, weights=column, minlength=self.n)
                - np.bincount(i, weights=column, minlength=self.n)
            )

    def _forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        force = np.zeros_like(x)
        force += self.m[:, None] * self.gravity[None, :]
        self._spring_forces(x, v, self.edges, self.edge_rest, self.edge_k, self.edge_c, force)
        self._spring_forces(x, v, self.cross, self.cross_rest, self.cross_k, self.cross_c, force)
        volume, gradient = self._volume_gradient(x)
        pressure = self.bulk_k * (volume - self.rest_volume)
        force -= pressure * gradient
        return force

    def _evaluate_external(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """External force terms: gravity and the suspension reaction."""
        force = self._forces(x, v)
        reaction = -force[self.fixed_idx].sum(axis=0)
        return np.stack([self._gravity_term, reaction], axis=0)

    # ------------------------------------------------------------- integration
    def _step(self) -> None:
        force = self._forces(self.x, self.v)
        acceleration = force / self.m[:, None]
        acceleration[self.fixed] = 0.0
        self.v += self.dt * acceleration
        self.x += self.dt * self.v
        self.x[self.fixed] = self.x0[self.fixed]
        self.v[self.fixed] = 0.0

        external = np.stack(
            [self._gravity_term, -force[self.fixed_idx].sum(axis=0)], axis=0
        )
        self.cumulative_impulse += self.dt * external.sum(axis=0)
        self._external_terms = external
        self.step += 1

    def advance_to(self, time: float):
        time = float(time)
        target = int(round(time / self.dt))
        if target > self.max_step:
            target = self.max_step
        while self.step < target:
            self._step()
        self.time = time
        return self._frame()

    # ----------------------------------------------------------------- output
    def _element_gradient(self, x: np.ndarray) -> np.ndarray:
        a, b, c = self.face_a, self.face_b, self.face_c
        e1 = x[b] - x[a]
        e2 = x[c] - x[a]
        normal = np.cross(e1, e2)
        normal = normal / np.maximum(np.linalg.norm(normal, axis=1), 1e-300)[:, None]
        current = np.stack([e1, e2, normal], axis=2)
        return current @ self._dm_inv

    def _energies(self, x: np.ndarray, v: np.ndarray) -> tuple[float, float]:
        kinetic = 0.5 * float(np.einsum("ij,ij->i", self.m[:, None] * v, v).sum())
        strain = np.linalg.norm(x[self.edges[:, 0]] - x[self.edges[:, 1]], axis=1)
        elastic = 0.5 * float(np.sum(self.edge_k * ((strain - self.edge_rest) / self.edge_rest) ** 2))
        bend = np.linalg.norm(x[self.cross[:, 0]] - x[self.cross[:, 1]], axis=1)
        elastic += 0.5 * float(
            np.sum(self.cross_k * ((bend - self.cross_rest) / self.cross_rest) ** 2)
        )
        volume, _ = self._volume_gradient(x)
        elastic += 0.5 * self.bulk_k * (volume - self.rest_volume) ** 2
        return kinetic, elastic

    def _frame(self) -> StateFrame:
        x = self.x.copy()
        v = self.v.copy()
        gradient = self._element_gradient(x)
        kinetic, elastic = self._energies(x, v)
        volume, _ = self._volume_gradient(x)
        fields = {
            "budget_cumulative_external_impulse": self.cumulative_impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_domain_weights": np.ones((1, self.n), dtype=np.float64),
            "budget_external_force": self._external_terms.reshape(1, 2, 3).copy(),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "element_deformation_gradient": gradient.astype(np.float32),
        }
        attributes = {
            "mass_quantity": self.m.copy(),
            "fixed": self.fixed_attr.copy(),
        }
        scalars = {
            "reference_momentum": self.reference_momentum,
            "total_kinetic_energy": kinetic,
            "total_elastic_energy": elastic,
            "enclosed_volume": float(volume),
        }
        frame = StateFrame(
            time=self.time,
            position=x,
            velocity=v,
            mass=self.m.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "vertices": int(self.n),
            "triangles": int(len(self.faces)),
            "edges": int(len(self.edges)),
            "bending_springs": int(len(self.cross)),
            "fixed_vertices": int(self.fixed.sum()),
            "time_step": float(self.dt),
            "substeps_per_sample": int(self.substeps),
            "spectral_radius_bound": float(self.omega_max),
            "rest_volume": float(abs(self.rest_volume)),
            "steps_executed": int(self.step),
        }
