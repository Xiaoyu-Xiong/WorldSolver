"""Deterministic finite-element physics solver for inflated rubber tube."""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Nonlinear solid mechanics solver for hollow rubber tube inflation."""

    def __init__(self) -> None:
        self.scene = None
        self.seed = 0
        self.current_time = 0.0

        # Geometry and node arrays
        self.rest_position = None
        self.position = None
        self.velocity = None
        self.mass = None
        self.fixed_nodes = None
        self.pressure_surface_mask = None
        self.surface_region = None

        # Mesh connectivity
        self.tets = None
        self.Dm_inv = None
        self.tet_rest_vol = None
        self.pressure_tris = None

        # Material parameters (Silicone rubber Neo-Hookean)
        self.mu = 100000.0  # Shear modulus (Pa)
        self.bulk_K = 1000000.0  # Bulk modulus (Pa)
        self.damping = 25.0  # Rayleigh/viscous damping coefficient

        # Diagnostic metrics
        self.reference_force = 10000.0 * 2.0 * np.pi * 0.4 * 1.6  # Characteristic pressure force scale
        self.external_force_budget = np.zeros((1, 1, 3), dtype=np.float64)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.current_time = 0.0

        initial_frame = scene.initial_state()
        self.rest_position = np.array(initial_frame.position, dtype=np.float64, copy=True)
        self.position = np.array(initial_frame.position, dtype=np.float64, copy=True)
        self.velocity = np.zeros_like(self.position, dtype=np.float64)
        self.mass = np.array(initial_frame.mass, dtype=np.float64, copy=True)

        self.fixed_nodes = np.array(initial_frame.attributes["fixed_rim"], dtype=bool)
        self.pressure_surface_mask = np.array(initial_frame.attributes["pressure_surface"], dtype=np.int32)
        self.surface_region = np.array(initial_frame.attributes["surface_region"], dtype=np.int32)

        g = scene.config["geometry"]
        n_theta = int(g["circumferential_segments"])
        n_z = int(g["axial_segments"])

        def get_node(layer: int, iz: int, itheta: int) -> int:
            return layer * (n_theta * (n_z + 1)) + iz * n_theta + (itheta % n_theta)

        # Build 5-tetrahedron subdivision for each hexahedral cell
        tets = []
        for iz in range(n_z):
            for it in range(n_theta):
                n0 = get_node(1, iz, it)
                n1 = get_node(0, iz, it)
                n2 = get_node(0, iz, it + 1)
                n3 = get_node(1, iz, it + 1)
                n4 = get_node(1, iz + 1, it)
                n5 = get_node(0, iz + 1, it)
                n6 = get_node(0, iz + 1, it + 1)
                n7 = get_node(1, iz + 1, it + 1)

                parity = (iz + it) % 2
                if parity == 0:
                    cell_tets = [
                        [n0, n1, n3, n4],
                        [n1, n2, n3, n6],
                        [n1, n4, n5, n6],
                        [n3, n4, n6, n7],
                        [n1, n3, n4, n6],
                    ]
                else:
                    cell_tets = [
                        [n0, n1, n2, n5],
                        [n0, n2, n3, n7],
                        [n0, n4, n5, n7],
                        [n2, n5, n6, n7],
                        [n0, n2, n5, n7],
                    ]
                tets.extend(cell_tets)

        self.tets = np.array(tets, dtype=np.int32)

        # Ensure positive reference orientation and volume
        v0 = self.rest_position[self.tets[:, 0]]
        v1 = self.rest_position[self.tets[:, 1]]
        v2 = self.rest_position[self.tets[:, 2]]
        v3 = self.rest_position[self.tets[:, 3]]
        vol = np.sum((v1 - v0) * np.cross(v2 - v0, v3 - v0), axis=1) / 6.0
        for i in range(len(self.tets)):
            if vol[i] < 0:
                self.tets[i, [1, 2]] = self.tets[i, [2, 1]]

        # Precompute reference shape matrices Dm_inv and rest volumes
        v0 = self.rest_position[self.tets[:, 0]]
        v1 = self.rest_position[self.tets[:, 1]]
        v2 = self.rest_position[self.tets[:, 2]]
        v3 = self.rest_position[self.tets[:, 3]]
        Dm = np.stack((v1 - v0, v2 - v0, v3 - v0), axis=-1)  # shape (Ne, 3, 3)
        self.Dm_inv = np.linalg.inv(Dm)
        self.tet_rest_vol = np.abs(np.linalg.det(Dm)) / 6.0

        # Build cavity pressure surface triangles with outward normals
        pressure_tris = []
        for iz in range(n_z):
            for it in range(n_theta):
                u00 = get_node(1, iz, it)
                u01 = get_node(1, iz, it + 1)
                u10 = get_node(1, iz + 1, it)
                u11 = get_node(1, iz + 1, it + 1)
                pressure_tris.append([u00, u01, u10])
                pressure_tris.append([u01, u11, u10])
        self.pressure_tris = np.array(pressure_tris, dtype=np.int32)

        # Re-distribute lump mass from tet density
        density = float(scene.config["material"]["density"])
        node_mass = np.zeros(len(self.position), dtype=np.float64)
        for i in range(4):
            np.add.at(node_mass, self.tets[:, i], 0.25 * density * self.tet_rest_vol)
        self.mass = node_mass

    def _compute_forces(self, pos: np.ndarray, vel: np.ndarray, pressure: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute internal elastic forces, damping forces, cavity pressure forces, and deformation gradient F."""
        n_nodes = len(pos)
        f_total = np.zeros((n_nodes, 3), dtype=np.float64)
        f_ext = np.zeros((n_nodes, 3), dtype=np.float64)

        # Current tet edge matrices Ds = [x1-x0, x2-x0, x3-x0]
        x0 = pos[self.tets[:, 0]]
        x1 = pos[self.tets[:, 1]]
        x2 = pos[self.tets[:, 2]]
        x3 = pos[self.tets[:, 3]]

        Ds = np.stack((x1 - x0, x2 - x0, x3 - x0), axis=-1)  # shape (Ne, 3, 3)
        F = np.matmul(Ds, self.Dm_inv)  # Deformation gradient F = Ds * Dm_inv (Ne, 3, 3)

        # First Piola-Kirchhoff stress P for compressible Neo-Hookean solid
        # Psi = mu/2 * (tr(C) - 3 - 2*ln(J)) + K/2 * (J - 1)^2
        # P = mu * (F - F^{-T}) + K * (J - 1) * J * F^{-T}
        J = np.linalg.det(F)  # shape (Ne,)
        J_safe = np.clip(J, 0.05, 10.0)
        F_inv_T = np.transpose(np.linalg.inv(F), (0, 2, 1))

        mu = self.mu
        K = self.bulk_K
        stress_coeff = K * (J_safe - 1.0) * J_safe - mu

        # P: shape (Ne, 3, 3)
        P = mu * F + stress_coeff[:, None, None] * F_inv_T

        # Element nodal forces: H = - V0 * P * Dm_inv^T
        Dm_inv_T = np.transpose(self.Dm_inv, (0, 2, 1))
        H = - self.tet_rest_vol[:, None, None] * np.matmul(P, Dm_inv_T)
        f1 = H[:, :, 0]
        f2 = H[:, :, 1]
        f3 = H[:, :, 2]
        f0 = -(f1 + f2 + f3)

        np.add.at(f_total, self.tets[:, 0], f0)
        np.add.at(f_total, self.tets[:, 1], f1)
        np.add.at(f_total, self.tets[:, 2], f2)
        np.add.at(f_total, self.tets[:, 3], f3)

        # Cavity surface pressure forces
        if pressure > 0.0:
            p0 = pos[self.pressure_tris[:, 0]]
            p1 = pos[self.pressure_tris[:, 1]]
            p2 = pos[self.pressure_tris[:, 2]]
            # Outward normal area vector (magnitude = 2 * triangle area)
            area_normals = np.cross(p1 - p0, p2 - p0)
            tri_forces = (pressure / 6.0) * area_normals  # Distributed to 3 vertices equally
            np.add.at(f_ext, self.pressure_tris[:, 0], tri_forces)
            np.add.at(f_ext, self.pressure_tris[:, 1], tri_forces)
            np.add.at(f_ext, self.pressure_tris[:, 2], tri_forces)

        f_total += f_ext

        # Velocity damping force
        f_total -= self.damping * self.mass[:, None] * vel

        return f_total, f_ext, F

    def _step(self, dt: float, target_time: float) -> np.ndarray:
        """Advance simulation by dt using semi-implicit / symplectic Euler integration."""
        pressure = self.scene.pressure_at(target_time)

        f_total, f_ext, F = self._compute_forces(self.position, self.velocity, pressure)

        # Acceleration a = f / m
        acc = f_total / self.mass[:, None]

        # Apply fixed rim boundary conditions (acc and vel are zero)
        acc[self.fixed_nodes] = 0.0

        # Update velocity and position
        self.velocity += dt * acc
        self.velocity[self.fixed_nodes] = 0.0
        self.position += dt * self.velocity

        # Track budget external force
        total_f_ext = np.sum(f_ext, axis=0)
        self.external_force_budget[0, 0] = total_f_ext

        return F

    def advance_to(self, time: float) -> StateFrame:
        """Advance solver state to the requested sample time."""
        if time < self.current_time:
            self.current_time = time

        sub_dt = 0.00025  # Stable sub-step for explicit integration
        F_last = None

        while self.current_time < time - 1e-9:
            step_size = min(sub_dt, time - self.current_time)
            self.current_time += step_size
            F_last = self._step(step_size, self.current_time)

        if F_last is None:
            pressure = self.scene.pressure_at(time)
            _, _, F_last = self._compute_forces(self.position, self.velocity, pressure)

        # Assemble StateFrame
        frame = StateFrame(
            time=float(time),
            position=np.array(self.position, dtype=np.float64, copy=True),
            velocity=np.array(self.velocity, dtype=np.float64, copy=True),
            mass=np.array(self.mass, dtype=np.float64, copy=True),
            attributes={
                "fixed_rim": self.fixed_nodes.astype(np.int32),
                "pressure_surface": self.pressure_surface_mask.copy(),
                "surface_region": self.surface_region.copy(),
                "rest_position": self.rest_position.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_force": self.external_force_budget.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "element_deformation_gradient": F_last.copy(),
            },
            scalars={
                "cavity_pressure_pa": float(self.scene.pressure_at(time)),
                "inner_radius": float(self.scene.config["geometry"]["inner_radius"]),
                "outer_radius": float(self.scene.config["geometry"]["inner_radius"] + self.scene.config["geometry"]["wall_thickness"]),
                "wall_thickness": float(self.scene.config["geometry"]["wall_thickness"]),
                "reference_force": float(self.reference_force),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "current_time": self.current_time,
            "elements_count": len(self.tets) if self.tets is not None else 0,
            "nodes_count": len(self.position) if self.position is not None else 0,
        }

