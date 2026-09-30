"""Embedded-lattice hyperelastic FEM solver for the suspended elastic bunny.

Physical model
--------------
The bunny is treated as a three-dimensional hyperelastic solid.  The frozen
surface mesh (in its displaced release pose) is voxelised into a hexahedral
lattice; every occupied cell is split into six Kuhn tetrahedra, giving a
conforming linear tetrahedral FEM discretisation of the volume.  Each element
uses the inversion-safe stable neo-Hookean energy

    Psi(F) = mu/2 (tr(F^T F) - 3) + lambda/2 (J - 1 - mu/lambda)^2

with mu, lambda from the frozen Young's modulus / Poisson ratio, plus a
frame-invariant Kelvin-Voigt viscous second-Piola stress S_v = beta * Edot
(Green strain rate), which vanishes for rigid motion.  Lumped nodal masses come
from rho * element volume.  Time integration is explicit symplectic Euler with a
CFL-safe substep, so the state is deterministic and finite.

The frozen surface samples are embedded in the lattice by trilinear
interpolation, so exported positions/velocities are the FEM solution evaluated
on the frozen vertex set (order preserved for the render topology).  The crown
patch flagged by the scene attribute ``fixed`` is enforced as a zero-displacement
Dirichlet boundary, which keeps the suspension point in place.
"""
from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import ndimage
from scipy.spatial import cKDTree

# Corner offsets of a lattice cell; local index = di + 2*dj + 4*dk.
_CORNER_OFFSETS = np.array(
    [(di, dj, dk) for dk in (0, 1) for dj in (0, 1) for di in (0, 1)], dtype=np.int64
)


def _corner_local_index(offset) -> int:
    di, dj, dk = offset
    return int(di + 2 * dj + 4 * dk)


def _kuhn_tetrahedra() -> np.ndarray:
    """Six-tetrahedron Kuhn decomposition of the unit cube (face conforming)."""
    tets = []
    for perm in permutations((0, 1, 2)):
        corner = [0, 0, 0]
        local = [_corner_local_index(corner)]
        for axis in perm:
            corner[axis] += 1
            local.append(_corner_local_index(corner))
        tets.append(local)
    return np.asarray(tets, dtype=np.int64)


_KUHN = _kuhn_tetrahedra()


def _load_faces(path: Path) -> np.ndarray:
    """Read triangle indices from a Wavefront OBJ (0-based)."""
    faces: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("f "):
                continue
            tokens = line.split()[1:]
            indices = [int(token.split("/")[0]) for token in tokens]
            for k in range(1, len(indices) - 1):
                faces.append((indices[0], indices[k], indices[k + 1]))
    if not faces:
        return np.zeros((0, 3), dtype=np.int64)
    return np.asarray(faces, dtype=np.int64) - 1


def _cofactor(f: np.ndarray) -> np.ndarray:
    """Cofactor matrices (dJ/dF) of a stack of 3x3 matrices."""
    c0 = np.cross(f[:, :, 1], f[:, :, 2])
    c1 = np.cross(f[:, :, 2], f[:, :, 0])
    c2 = np.cross(f[:, :, 0], f[:, :, 1])
    return np.stack((c0, c1, c2), axis=2)


class BenchmarkSolver:
    """Explicit hyperelastic FEM solver on a hexahedral lattice with surface embedding."""

    CELLS_ACROSS = 16           # target lattice resolution across the largest extent
    MAX_ELEMENTS = 9000         # coarsen the lattice until the tet count fits
    SUBSTEP_DT = 2.0e-4         # explicit substep (CFL-safe for the frozen material)
    MASS_DAMPING = 0.02         # 1/s, weak external drag
    STRAIN_RATE_DAMPING = 30.0  # Pa*s, Kelvin-Voigt on the Green strain rate

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config

        frame0 = scene.initial_state()
        self.rest_surface = np.ascontiguousarray(
            np.asarray(frame0.position, dtype=np.float64)
        )
        self.n_samples = int(self.rest_surface.shape[0])
        mass = frame0.mass
        if mass is None:
            mass = np.full(self.n_samples, 1.0 / self.n_samples, dtype=np.float64)
        self.sample_mass = np.ascontiguousarray(np.asarray(mass, dtype=np.float64))
        self.total_export_mass = float(self.sample_mass.sum())

        boundary = dict(config.get("boundary_conditions") or {})
        attr_name = str(boundary.get("fixed_attribute", "fixed"))
        fixed_value = float(boundary.get("fixed_value", 1))
        attribute = frame0.attributes.get(attr_name)
        if attribute is None:
            self.fixed_samples = np.zeros(self.n_samples, dtype=bool)
        else:
            values = np.asarray(attribute, dtype=np.float64).reshape(self.n_samples, -1)[:, 0]
            self.fixed_samples = np.isclose(values, fixed_value)
        self.fixed_attribute_name = attr_name
        self.fixed_attribute_value = fixed_value

        self.gravity = np.asarray(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64
        ).reshape(3)
        initial = dict(config.get("initial_conditions") or {})
        self.attachment = np.asarray(
            initial.get("attachment_position", [0.0, 0.0, 2.85]), dtype=np.float64
        ).reshape(3)
        self.release_velocity = np.asarray(
            initial.get("release_velocity", [0.0, 0.0, 0.0]), dtype=np.float64
        ).reshape(3)

        material = dict(config.get("material_reference") or {})
        self.density = float(material.get("density_kg_m3", 1000.0))
        youngs = float(material.get("youngs_modulus_pa", 30000.0))
        poisson = float(material.get("poisson_ratio", 0.45))
        poisson = min(max(poisson, 0.0), 0.4995)
        self.youngs = youngs
        self.poisson = poisson
        self.mu = youngs / (2.0 * (1.0 + poisson))
        self.lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self.alpha_nh = 1.0 + self.mu / self.lam

        faces = self._load_surface_faces()
        cloud = self._sample_cloud(faces)
        self.cell_size, self.origin, self.occupancy = self._build_lattice(cloud)
        self._build_mesh()
        self._build_embedding()
        self._precompute_fem()

        self.node_pos = self.node_rest.copy()
        self.node_vel = np.tile(self.release_velocity, (self.n_nodes, 1))
        self.node_vel[self.fixed_nodes] = 0.0
        self.current_time = 0.0
        self.step_count = 0

        speed = float(np.sqrt(2.0 * abs(self.gravity[2]) * max(self.cell_size, 0.25)))
        self.reference_momentum = float(max(self.total_export_mass * speed, 1e-6))

    def _load_surface_faces(self) -> np.ndarray:
        geometry = dict(self.config.get("scene_geometry") or {})
        candidates: list[str] = []
        for item in geometry.get("objects", []) or []:
            spec = dict((item or {}).get("geometry") or {})
            for key in ("topology_asset", "asset"):
                value = spec.get(key)
                if value:
                    candidates.append(str(value))
        candidates.append("assets/bunny.obj")
        bases = [Path.cwd(), Path(__file__).resolve().parent.parent]
        for candidate in candidates:
            path = Path(candidate)
            options = [path] if path.is_absolute() else [base / path for base in bases]
            for option in options:
                if option.is_file():
                    return _load_faces(option)
        return np.zeros((0, 3), dtype=np.int64)

    def _sample_cloud(self, faces: np.ndarray) -> np.ndarray:
        points = [self.rest_surface]
        if len(faces):
            tri = self.rest_surface[faces]
            points.append(tri.mean(axis=1))
            points.append(0.5 * (tri[:, 0] + tri[:, 1]))
            points.append(0.5 * (tri[:, 1] + tri[:, 2]))
            points.append(0.5 * (tri[:, 2] + tri[:, 0]))
        return np.concatenate(points, axis=0)

    def _build_lattice(self, cloud: np.ndarray):
        lo = cloud.min(axis=0)
        hi = cloud.max(axis=0)
        h = float(np.max(hi - lo)) / float(self.CELLS_ACROSS)
        origin = lo.copy()
        occupancy = np.zeros((1, 1, 1), dtype=bool)
        for _ in range(16):
            origin, occupancy = self._voxelise(cloud, lo, hi, h)
            if int(occupancy.sum()) * len(_KUHN) <= self.MAX_ELEMENTS:
                break
            h *= 1.15
        return h, origin, occupancy

    def _voxelise(self, cloud: np.ndarray, lo: np.ndarray, hi: np.ndarray, h: float):
        origin = lo - 1.5 * h
        shape = np.ceil((hi - lo) / h).astype(np.int64) + 3
        shape = np.maximum(shape, 3)
        index = np.floor((cloud - origin) / h).astype(np.int64)
        np.clip(index, 0, shape - 1, out=index)
        occupancy = np.zeros(tuple(int(s) for s in shape), dtype=bool)
        occupancy[index[:, 0], index[:, 1], index[:, 2]] = True
        occupancy = ndimage.binary_fill_holes(occupancy)
        labels, count = ndimage.label(occupancy)
        if count > 1:
            sizes = ndimage.sum(occupancy, labels, index=np.arange(1, count + 1))
            occupancy = labels == (int(np.argmax(sizes)) + 1)
        return origin, np.ascontiguousarray(occupancy)

    def _build_mesh(self) -> None:
        occupancy = self.occupancy
        cells = np.argwhere(occupancy).astype(np.int64)
        if len(cells) == 0:
            raise ValueError("Voxelisation produced an empty lattice")
        self.cells = cells
        node_shape = tuple(int(s) + 1 for s in occupancy.shape)
        self.node_shape = node_shape

        used = np.zeros(node_shape, dtype=bool)
        for offset in _CORNER_OFFSETS:
            idx = cells + offset
            used[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        flat_used = used.reshape(-1)
        mapping = np.full(flat_used.size, -1, dtype=np.int64)
        mapping[flat_used] = np.arange(int(flat_used.sum()), dtype=np.int64)
        self.node_map = mapping.reshape(node_shape)
        self.n_nodes = int(flat_used.sum())

        grid = np.argwhere(used).astype(np.float64)
        self.node_rest = np.ascontiguousarray(self.origin + grid * self.cell_size)

        corner_ids = np.empty((len(cells), 8), dtype=np.int64)
        for local, offset in enumerate(_CORNER_OFFSETS):
            idx = cells + offset
            corner_ids[:, local] = self.node_map[idx[:, 0], idx[:, 1], idx[:, 2]]
        self.cell_corner_ids = corner_ids
        self.tets = np.ascontiguousarray(corner_ids[:, _KUHN].reshape(-1, 4))
        self.n_elements = int(self.tets.shape[0])

    def _build_embedding(self) -> None:
        h = self.cell_size
        shape = np.asarray(self.occupancy.shape, dtype=np.int64)
        index = np.floor((self.rest_surface - self.origin) / h).astype(np.int64)
        np.clip(index, 0, shape - 1, out=index)
        occupied = self.occupancy[index[:, 0], index[:, 1], index[:, 2]]
        if not occupied.all():
            centers = self.origin + (self.cells + 0.5) * h
            tree = cKDTree(centers)
            missing = np.flatnonzero(~occupied)
            _, nearest = tree.query(self.rest_surface[missing])
            index[missing] = self.cells[np.asarray(nearest, dtype=np.int64)]
        self.sample_cell_index = index

        cell_origin = self.origin + index * h
        local = np.clip((self.rest_surface - cell_origin) / h, 0.0, 1.0)

        rows = np.repeat(np.arange(self.n_samples, dtype=np.int64), 8)
        cols = np.empty(self.n_samples * 8, dtype=np.int64)
        data = np.empty(self.n_samples * 8, dtype=np.float64)
        for local_id, offset in enumerate(_CORNER_OFFSETS):
            node_idx = index + offset
            cols[local_id::8] = self.node_map[node_idx[:, 0], node_idx[:, 1], node_idx[:, 2]]
            weight = np.ones(self.n_samples, dtype=np.float64)
            for axis in range(3):
                weight *= local[:, axis] if offset[axis] == 1 else (1.0 - local[:, axis])
            data[local_id::8] = weight
        self.embedding = sp.csr_matrix(
            (data, (rows, cols)), shape=(self.n_samples, self.n_nodes)
        )

        fixed_nodes = np.zeros(self.n_nodes, dtype=bool)
        if self.fixed_samples.any():
            fixed_cells = np.unique(index[self.fixed_samples], axis=0)
            for offset in _CORNER_OFFSETS:
                idx = fixed_cells + offset
                ids = self.node_map[idx[:, 0], idx[:, 1], idx[:, 2]]
                fixed_nodes[ids[ids >= 0]] = True
        if not fixed_nodes.any():
            radius = 1.25 * h
            distance = np.linalg.norm(self.node_rest - self.attachment, axis=1)
            fixed_nodes[distance <= max(radius, float(distance.min()) + 1e-9)] = True
        self.fixed_nodes = fixed_nodes
        self.free_nodes = ~fixed_nodes

    def _precompute_fem(self) -> None:
        rest = self.node_rest[self.tets]
        edges = np.stack(
            (rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0], rest[:, 3] - rest[:, 0]),
            axis=2,
        )
        determinant = np.linalg.det(edges)
        volume = np.abs(determinant) / 6.0
        valid = volume > 1e-14
        if not valid.all():
            self.tets = np.ascontiguousarray(self.tets[valid])
            self.n_elements = int(self.tets.shape[0])
            edges = edges[valid]
            volume = volume[valid]
        self.rest_inverse = np.ascontiguousarray(np.linalg.inv(edges))
        self.rest_volume = np.ascontiguousarray(volume)
        self.total_volume = float(volume.sum())

        node_mass = np.zeros(self.n_nodes, dtype=np.float64)
        share = self.density * self.rest_volume / 4.0
        for corner in range(4):
            np.add.at(node_mass, self.tets[:, corner], share)
        floor = 1e-9 * max(node_mass.max(), 1.0)
        self.node_mass = np.maximum(node_mass, floor)
        self.inv_node_mass = 1.0 / self.node_mass
        self.sim_mass = float(self.node_mass.sum())

        count = 4 * self.n_elements
        rows = self.tets.reshape(-1)
        cols = np.arange(count, dtype=np.int64)
        self.assembly = sp.csr_matrix(
            (np.ones(count, dtype=np.float64), (rows, cols)),
            shape=(self.n_nodes, count),
        )
        self.export_node_mass = np.asarray(
            self.embedding.T @ self.sample_mass, dtype=np.float64
        ).reshape(-1)

    # ---------------------------------------------------------------- physics
    def _deformation_gradients(self, position: np.ndarray) -> np.ndarray:
        p = position[self.tets]
        ds = np.stack((p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]), axis=2)
        return ds @ self.rest_inverse

    def _forces(self, position: np.ndarray, velocity: np.ndarray):
        """Nodal forces: stable neo-Hookean elasticity + Kelvin-Voigt + gravity + drag."""
        defgrad = self._deformation_gradients(position)
        v = velocity[self.tets]
        dv = np.stack((v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]), axis=2)
        rate = dv @ self.rest_inverse

        cofactor = _cofactor(defgrad)
        jacobian = np.einsum("nij,nij->n", defgrad, cofactor) / 3.0
        piola = self.mu * defgrad
        piola += (self.lam * (jacobian - self.alpha_nh))[:, None, None] * cofactor

        if self.STRAIN_RATE_DAMPING > 0.0:
            ft_rate = np.matmul(np.transpose(defgrad, (0, 2, 1)), rate)
            green_rate = 0.5 * (ft_rate + np.transpose(ft_rate, (0, 2, 1)))
            piola += self.STRAIN_RATE_DAMPING * np.matmul(defgrad, green_rate)

        stress = -self.rest_volume[:, None, None] * np.matmul(
            piola, np.transpose(self.rest_inverse, (0, 2, 1))
        )
        element_force = np.empty((self.n_elements, 4, 3), dtype=np.float64)
        element_force[:, 1, :] = stress[:, :, 0]
        element_force[:, 2, :] = stress[:, :, 1]
        element_force[:, 3, :] = stress[:, :, 2]
        element_force[:, 0, :] = -(stress[:, :, 0] + stress[:, :, 1] + stress[:, :, 2])

        force = np.asarray(self.assembly @ element_force.reshape(-1, 3), dtype=np.float64)
        force += self.node_mass[:, None] * self.gravity[None, :]
        if self.MASS_DAMPING > 0.0:
            force -= (self.MASS_DAMPING * self.node_mass)[:, None] * velocity
        return force, defgrad

    def _step(self, dt: float) -> None:
        force, _ = self._forces(self.node_pos, self.node_vel)
        acceleration = force * self.inv_node_mass[:, None]
        acceleration[self.fixed_nodes] = 0.0
        self.node_vel += dt * acceleration
        self.node_vel[self.fixed_nodes] = 0.0
        self.node_pos += dt * self.node_vel
        self.node_pos[self.fixed_nodes] = self.node_rest[self.fixed_nodes]
        self.step_count += 1

    # ----------------------------------------------------------------- output
    def advance_to(self, time: float):
        from worldsolver.contracts import StateFrame

        target = float(time)
        if target > self.current_time:
            span = target - self.current_time
            steps = max(1, int(np.ceil(span / self.SUBSTEP_DT)))
            dt = span / float(steps)
            for _ in range(steps):
                self._step(dt)
        self.current_time = target

        force, defgrad = self._forces(self.node_pos, self.node_vel)
        acceleration = force * self.inv_node_mass[:, None]
        acceleration[self.fixed_nodes] = 0.0

        position = np.ascontiguousarray(
            np.asarray(self.embedding @ self.node_pos, dtype=np.float64)
        )
        velocity = np.ascontiguousarray(
            np.asarray(self.embedding @ self.node_vel, dtype=np.float64)
        )

        # External budget of the exported domain: body weight plus the suspension
        # reaction (pin traction and drag) that closes the momentum balance.
        gravity_force = self.total_export_mass * self.gravity
        momentum_rate = self.export_node_mass @ acceleration
        reaction_force = momentum_rate - gravity_force
        external_force = np.stack((gravity_force, reaction_force), axis=0)[None, :, :]

        fields = {
            "budget_domain_weights": np.ones((1, self.n_samples), dtype=np.float64),
            "budget_external_force": np.ascontiguousarray(external_force),
            "budget_cumulative_external_impulse": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "element_deformation_gradient": np.ascontiguousarray(defgrad, dtype=np.float32),
        }
        attributes = {
            "mass_quantity": self.sample_mass.reshape(self.n_samples, 1).copy(),
            self.fixed_attribute_name: np.where(
                self.fixed_samples, self.fixed_attribute_value, 0.0
            ).astype(np.float64),
        }
        scalars = {
            "reference_momentum": self.reference_momentum,
            "density_kg_m3": self.density,
            "youngs_modulus_pa": self.youngs,
            "poisson_ratio": self.poisson,
            "element_count": float(self.n_elements),
            "simulated_mass_kg": self.sim_mass,
        }
        frame = StateFrame(
            time=target,
            position=position,
            velocity=velocity,
            mass=self.sample_mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "embedded-lattice stable neo-Hookean FEM, explicit symplectic Euler",
            "nodes": int(self.n_nodes),
            "elements": int(self.n_elements),
            "cells": int(len(self.cells)),
            "cell_size_m": float(self.cell_size),
            "substep_dt_s": float(self.SUBSTEP_DT),
            "steps_taken": int(self.step_count),
            "fixed_nodes": int(self.fixed_nodes.sum()),
            "lattice_volume_m3": float(self.total_volume),
            "simulated_mass_kg": float(self.sim_mass),
            "exported_mass_kg": float(self.total_export_mass),
            "lame_mu_pa": float(self.mu),
            "lame_lambda_pa": float(self.lam),
            "strain_rate_damping_pa_s": float(self.STRAIN_RATE_DAMPING),
            "mass_damping_per_s": float(self.MASS_DAMPING),
            "final_time_s": float(self.current_time),
        }
