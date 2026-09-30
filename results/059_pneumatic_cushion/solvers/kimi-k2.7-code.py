"""Physical finite-element solver for the pneumatic cushion inflation scene.

The solver integrates a 3D tetrahedral solid with a nearly-incompressible
neo-Hookean material.  Volumetric locking is avoided by treating the
deviatoric response element-wise and the dilatational response as a single
global volume constraint, while the inflation pressure is applied as a
prescribed boundary traction on the inner chamber wall.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


Array = np.ndarray  # convenience alias


class BenchmarkSolver:
    """Explicit-dynamics FEM solver for the pneumatic cushion."""

    def __init__(self) -> None:
        self._seed = 0
        self._scene = None

    def initialize(self, scene, seed: int) -> None:
        self._seed = seed
        self._scene = scene

        # Frozen physical configuration
        self._config: dict = scene.config  # type: ignore[attr-defined]
        mat = self._config["material_properties"]
        loading = self._config["loading"]

        # Initial state from the scene contract
        frame0 = scene.initial_state()
        self._positions = np.asarray(frame0.position, dtype=np.float64).copy()
        self._velocities = np.asarray(frame0.velocity, dtype=np.float64).copy()
        self._mass = np.asarray(frame0.mass, dtype=np.float64).copy()
        if self._mass.ndim == 0:
            self._mass = np.full(len(self._positions), float(self._mass))
        self._inv_mass = 1.0 / np.where(self._mass > 0.0, self._mass, 1.0)

        # Keep wall_region from the scene (used for rendering / binding)
        self._wall_region = np.asarray(
            frame0.attributes.get("wall_region", np.zeros(len(self._positions), dtype=np.int32)),
            dtype=np.int32,
        )

        # Material parameters (E, nu are authoritative)
        E = float(mat["young_modulus_pa"])
        nu = float(mat["poisson_ratio"])
        self._rho = float(mat["density_kg_m3"])
        # Shear modulus and bulk modulus for the neo-Hookean model
        self._mu = E / (2.0 * (1.0 + nu))
        self._kappa = E / (3.0 * (1.0 - 2.0 * nu))

        # Pressure loading profile
        self._p0 = float(loading["initial_internal_pressure_pa"])
        self._p_target = float(loading["target_internal_pressure_pa"])
        self._ramp_start = float(loading["ramp_start_s"])
        self._ramp_end = float(loading["ramp_end_s"])
        self._hold_end = float(loading["hold_end_s"])

        # Load topology
        asset = getattr(scene, "_asset", Path("assets") / "cushion_solid.npz")
        data = np.load(asset)

        elements = None
        for key in ("elements", "tets", "tetrahedra", "cells"):
            if key in data:
                elements = np.asarray(data[key], dtype=np.int64)
                break
        if elements is None:
            raise RuntimeError("No tetrahedral element connectivity found in the solid asset.")
        if elements.ndim != 2 or elements.shape[1] not in (4, 5):
            raise ValueError(f"Unexpected element array shape {elements.shape}")
        if elements.shape[1] == 5:
            # TetGen-style marker column; drop it
            elements = elements[:, :4]
        self._elements = elements

        # Reference (material) positions
        self._X = self._positions.copy()

        # Surface faces used for the pressure traction
        self._pressure_faces = self._load_pressure_faces(Path("assets") / "cushion_rest.obj")

        # Element reference geometry
        self._rest_volume, self._grad_N = self._compute_reference_geometry()
        self._total_rest_volume = float(np.sum(self._rest_volume))

        # Integration parameters
        # The mesh length scale is ~0.1 m and the shear wave speed is
        # sqrt(mu/rho) ~ 5 m/s, so dt = 1e-4 s is comfortably stable.
        self._dt = 1e-4
        self._damping = 50.0  # s^-1, heavy enough to follow the quasi-static ramp

        self._t = 0.0

        # Reference force scale for the trajectory contract
        self._reference_force = max(1.0, abs(self._p_target))

        # Pre-compute per-element node-index lookup for assembly
        self._elem_nodes = self._elements

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _compute_reference_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-element rest volume (positive) and shape-function gradients."""
        e = self._elements  # (Ne, 4)
        x0 = self._X[e[:, 0]]
        x1 = self._X[e[:, 1]]
        x2 = self._X[e[:, 2]]
        x3 = self._X[e[:, 3]]

        # D = [x1-x0, x2-x0, x3-x0] as column vectors
        D = np.empty((e.shape[0], 3, 3), dtype=np.float64)
        D[:, :, 0] = x1 - x0
        D[:, :, 1] = x2 - x0
        D[:, :, 2] = x3 - x0

        # Signed volume = det(D)/6; use absolute value for positive rest volume
        det_D = np.linalg.det(D)
        rest_volume = np.abs(det_D) / 6.0

        inv_DT = np.transpose(np.linalg.inv(D), (0, 2, 1))
        grad_N = np.empty((e.shape[0], 4, 3), dtype=np.float64)
        # Nodes 1..3
        grad_N[:, 1, :] = inv_DT[:, :, 0]
        grad_N[:, 2, :] = inv_DT[:, :, 1]
        grad_N[:, 3, :] = inv_DT[:, :, 2]
        # Node 0 is the negative sum so that shape functions sum to 1
        grad_N[:, 0, :] = -(grad_N[:, 1, :] + grad_N[:, 2, :] + grad_N[:, 3, :])
        return rest_volume, grad_N

    def _load_pressure_faces(self, obj_path: Path) -> np.ndarray:
        """Load the OBJ surface and keep only the inner-chamber triangles."""
        faces: list[list[int]] = []
        with obj_path.open("r", encoding="utf-8") as fobj:
            for line in fobj:
                if not line.startswith("f "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                # Support "f v/vt/vn ..."
                idx = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                if len(idx) == 3:
                    faces.append(idx)
                elif len(idx) == 4:
                    faces.append([idx[0], idx[1], idx[2]])
                    faces.append([idx[0], idx[2], idx[3]])

        faces = np.asarray(faces, dtype=np.int64)

        # Re-orient so normals point outward of the solid; then inner faces have
        # normals with a component toward the chamber centre (dot(centroid, n) < 0).
        a = self._X[faces[:, 0]]
        b = self._X[faces[:, 1]]
        c = self._X[faces[:, 2]]
        nvec = 0.5 * np.cross(b - a, c - a)
        cent = (a + b + c) / 3.0
        signed = np.einsum("ij,ij->i", nvec, cent)
        if np.sum(signed) < 0.0:
            # Flip every face so the normal points outward (positive signed volume)
            faces = faces[:, [0, 2, 1]]
            nvec = -nvec

        # Inner faces are the ones whose outward normal points toward the origin
        inner = np.einsum("ij,ij->i", nvec, cent) < -1e-12
        return faces[inner]

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------
    def _pressure(self, t: float) -> float:
        if t <= self._ramp_start:
            return self._p0
        if t >= self._ramp_end:
            return self._p_target
        s = (t - self._ramp_start) / (self._ramp_end - self._ramp_start)
        return self._p0 + s * (self._p_target - self._p0)

    def _compute_element_F(self, positions: np.ndarray) -> np.ndarray:
        """Deformation gradient for every tetrahedron."""
        x_el = positions[self._elements]  # (Ne, 4, 3)
        F = np.einsum("eia,eib->eab", x_el, self._grad_N)
        return F

    def _assemble_internal_force(
        self, positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (internal_force [N,3], current element F [Ne,3,3])."""
        F = self._compute_element_F(positions)

        # Determinants and inverse transpose
        J = np.linalg.det(F)
        J_safe = np.maximum(J, 1e-8)
        F_inv = np.linalg.inv(F)
        F_inv_T = np.transpose(F_inv, (0, 2, 1))

        # Global average volume ratio (weighted by rest volume)
        J_avg = float(np.sum(self._rest_volume * J_safe) / self._total_rest_volume)

        # First Piola-Kirchhoff stress:
        #   deviatoric part = mu * F
        #   dilatational part = kappa * (J_avg - 1) * J * F^{-T}
        P = self._mu * F + self._kappa * (J_avg - 1.0) * J_safe[:, None, None] * F_inv_T

        # nodal_force_i = -V0 * P @ grad_N_i
        P_grad = np.matmul(P, self._grad_N.transpose(0, 2, 1))  # (Ne, 3, 4)
        P_grad = P_grad.transpose(0, 2, 1)  # (Ne, 4, 3)
        local_forces = -self._rest_volume[:, None, None] * P_grad

        force = np.zeros_like(positions)
        np.add.at(force, self._elem_nodes, local_forces)
        return force, F

    def _apply_pressure_force(self, positions: np.ndarray, t: float) -> np.ndarray:
        """Apply prescribed uniform internal pressure to the inner wall surface."""
        p = self._pressure(t)
        a = positions[self._pressure_faces[:, 0]]
        b = positions[self._pressure_faces[:, 1]]
        c = positions[self._pressure_faces[:, 2]]
        nvec = 0.5 * np.cross(b - a, c - a)  # oriented normal with magnitude = area
        # Pressure force on the solid: f = -p * n (n is the outward solid normal)
        total_force = -p * nvec  # (Nf, 3)
        per_node = total_force / 3.0

        force = np.zeros_like(positions)
        np.add.at(force, self._pressure_faces, per_node)
        return force

    def _remove_rigid_body_modes(self) -> None:
        """Project out linear and angular rigid-body velocity (keeps cushion centred)."""
        m = self._mass
        v = self._velocities

        # Linear
        lin = np.sum(m[:, None] * v, axis=0) / np.sum(m)
        v -= lin

        # Angular around current centre of mass
        cm = np.sum(m[:, None] * self._positions, axis=0) / np.sum(m)
        r = self._positions - cm
        L = np.sum(
            m[:, None, None]
            * np.cross(r[:, None, :], v[:, :, None]),
            axis=0,
        ).reshape(3)
        I = np.sum(
            m[:, None, None]
            * (
                np.eye(3) * np.sum(r * r, axis=1)[:, None, None]
                - r[:, :, None] * r[:, None, :]
            ),
            axis=0,
        )
        try:
            omega = np.linalg.solve(I, L)
        except np.linalg.LinAlgError:
            return
        v -= np.cross(omega, r)

    # ------------------------------------------------------------------
    # Public solver interface
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time < self._t:
            raise ValueError("advance_to() must be called with non-decreasing times")

        # Sub-step explicit dynamics until the requested sample time
        while self._t < time:
            dt = min(self._dt, time - self._t)

            f_int, _ = self._assemble_internal_force(self._positions)
            f_ext = self._apply_pressure_force(self._positions, self._t)

            # Mass-proportional damping: f_damp = -c * M * v
            acceleration = (f_int + f_ext) * self._inv_mass[:, None]
            acceleration -= self._damping * self._velocities

            self._velocities += dt * acceleration
            self._positions += dt * self._velocities
            self._t += dt

            self._remove_rigid_body_modes()

        # Build the requested StateFrame at the exact sample time
        pressure = self._pressure(self._t)
        _, F_current = self._assemble_internal_force(self._positions)

        f_ext_nodal = self._apply_pressure_force(self._positions, self._t)
        total_external = np.sum(f_ext_nodal, axis=0)

        frame = StateFrame(
            time=float(self._t),
            position=self._positions.copy(),
            velocity=self._velocities.copy(),
            mass=self._mass.copy(),
            attributes={
                "wall_region": self._wall_region.copy(),
                "mass_quantity": self._mass.copy(),
            },
            fields={
                "element_deformation_gradient": F_current.copy(),
                "budget_external_force": total_external.reshape(1, 1, 3),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            },
            scalars={
                "internal_pressure_pa": float(pressure),
                "external_pressure_pa": 0.0,
                "reference_force": float(self._reference_force),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "tetrahedral_fem_neo_hookean",
            "dt": self._dt,
            "damping": self._damping,
            "elements": int(self._elements.shape[0]),
            "nodes": int(self._positions.shape[0]),
            "pressure_faces": int(self._pressure_faces.shape[0]),
        }
