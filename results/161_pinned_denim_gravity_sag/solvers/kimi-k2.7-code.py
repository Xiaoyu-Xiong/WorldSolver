"""Position-based dynamics solver for pinned inextensible cloth.

A heavy denim sheet is modeled as a triangle-network of distance constraints.
Gravity pulls the free vertices downward while the top edge remains pinned to
the static bar.  An explicit predictor is followed by iterative PBD projection
onto the inextensibility, pin, and ground-contact constraints.  Small global
velocity damping removes excess kinetic energy so the sheet settles into a
stable drape.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        cfg = scene.config
        cloth_cfg = cfg["cloth"]
        init = scene.initial_state()

        # Dynamic state arrays (deep copies so we do not mutate the scaffold).
        self.position = np.asarray(init.position, dtype=float).copy()
        self.velocity = np.asarray(init.velocity, dtype=float).copy()
        self.mass = np.asarray(init.mass, dtype=float).copy()
        if init.body_id is not None:
            self.body_id = np.asarray(init.body_id, dtype=np.int32).copy()
        else:
            self.body_id = np.zeros(len(self.position), dtype=np.int32)
        self.boundary_vertex = np.asarray(
            init.attributes.get("boundary_vertex", np.zeros(len(self.position), dtype=np.int32)),
            dtype=np.int32,
        ).copy()

        self.thickness = float(cloth_cfg["thickness"])
        self.contact_scale = self.thickness
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.sample_dt = float(cfg["sample_dt"])

        # Time-stepping parameters.
        self.num_substeps = 40
        self.h = self.sample_dt / self.num_substeps
        self.iters = 20
        self.damping = 0.99

        # Regular grid layout from the scaffold.
        self.nx = int(cloth_cfg["grid_shape"][0])
        self.ny = int(cloth_cfg["grid_shape"][1])
        self.nv = self.nx * self.ny
        self.sx = float(cloth_cfg["size"][0])
        self.sy = float(cloth_cfg["size"][1])
        self.idx = np.arange(self.nv).reshape(self.ny, self.nx)

        # Build the full triangle-mesh edge set.  Structural edges plus one
        # diagonal per quad matches the triangulation used by the topology asset.
        edges = []
        for y in range(self.ny):
            for x in range(self.nx - 1):
                edges.append((self.idx[y, x], self.idx[y, x + 1]))
        for y in range(self.ny - 1):
            for x in range(self.nx):
                edges.append((self.idx[y, x], self.idx[y + 1, x]))
        for y in range(self.ny - 1):
            for x in range(self.nx - 1):
                edges.append((self.idx[y, x], self.idx[y + 1, x + 1]))
        self.edges = np.asarray(edges, dtype=np.int32)
        self.rest_len = np.linalg.norm(
            self.position[self.edges[:, 1]] - self.position[self.edges[:, 0]], axis=1
        )

        # Pin the top edge of the sheet (highest initial z, lying inside the bar).
        self.pinned = np.zeros(self.nv, dtype=bool)
        self.pinned[self.idx[-1, :]] = True
        self.pin_pos = self.position[self.pinned].copy()

        # Tiny deterministic out-of-plane kick to break the planar symmetry and
        # let the inextensible sheet buckle into folds.  It is applied only as
        # an initial velocity impulse during the first substep.
        self._perturb = 1e-2 * np.sin(np.pi * self.position[:, 0] / self.sx)

        # External force is just gravity; internal/contact forces are excluded.
        self.external_force = self.mass[:, None] * self.gravity[None, :]
        self.cum_mass_source = np.zeros(1, dtype=float)
        self.ground_z = 0.0
        self.time = 0.0

    def _solve_constraints(self, position: np.ndarray) -> None:
        """Project ``position`` onto distance, pin, and ground constraints."""
        for _ in range(self.iters):
            dx = position[self.edges[:, 1]] - position[self.edges[:, 0]]
            d = np.linalg.norm(dx, axis=1)
            mask = d > 1e-12
            # Equal per-vertex mass => split the correction 50/50.
            factor = 0.5 * (d[mask] - self.rest_len[mask]) / d[mask]
            delta = dx[mask] * factor[:, None]
            correction = np.zeros_like(position)
            np.add.at(correction, self.edges[mask, 0], delta)
            np.add.at(correction, self.edges[mask, 1], -delta)
            position += correction

            # Hard positional constraints.
            position[self.pinned] = self.pin_pos
            position[:, 2] = np.maximum(position[:, 2], self.ground_z + 0.5 * self.thickness)

    def _step(self, h: float) -> None:
        if self.time < 1e-12:
            # Single deterministic imperfection to trigger 3D buckling.
            self.velocity[:, 1] += self._perturb

        # Explicit gravity integration.
        self.velocity += h * self.gravity

        # Predicted configuration.
        predicted = self.position + h * self.velocity

        # Constraint projection enforces inextensibility, pins, and ground.
        self._solve_constraints(predicted)

        # Update velocity from the projected positions and damp oscillations.
        self.velocity = (predicted - self.position) / h
        self.velocity[self.pinned] = 0.0
        self.velocity *= self.damping

        self.position = predicted.copy()
        self.time += h

    def _deformation_gradient(self, position: np.ndarray) -> np.ndarray:
        """Per-vertex 2D deformation gradient in the material (x, z) basis."""
        du = self.sx / (self.nx - 1)
        dv = self.sy / (self.ny - 1)
        F = np.zeros((self.nv, 2, 2), dtype=float)
        for y in range(self.ny):
            for x in range(self.nx):
                i = self.idx[y, x]
                # Choose inward neighbors when on the boundary.
                if x < self.nx - 1:
                    j, sgnu = self.idx[y, x + 1], du
                else:
                    j, sgnu = self.idx[y, x - 1], -du
                if y < self.ny - 1:
                    k, sgnv = self.idx[y + 1, x], dv
                else:
                    k, sgnv = self.idx[y - 1, x], -dv
                dp_u = position[j] - position[i]
                dp_v = position[k] - position[i]
                F[i, 0, 0] = dp_u[0] / sgnu
                F[i, 1, 0] = dp_u[2] / sgnu
                F[i, 0, 1] = dp_v[0] / sgnv
                F[i, 1, 1] = dp_v[2] / sgnv
        return F

    def _make_frame(self, time: float) -> StateFrame:
        deformation = self._deformation_gradient(self.position)
        # Signed distance from cloth mid-plane to the ground top surface.
        contact_sd = self.position[:, 2] - (self.ground_z + 0.5 * self.thickness)
        return StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary_vertex.copy(),
                "deformation_gradient": deformation,
            },
            fields={
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "external_force": self.external_force.copy(),
                "contact_signed_distance": contact_sd,
            },
            scalars={"contact_length_scale": self.contact_scale},
        )

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 1e-12:
            h = min(self.h, target - self.time)
            self._step(h)
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "substep": self.h,
            "substeps_per_sample": self.num_substeps,
            "constraint_iterations": self.iters,
            "num_vertices": self.nv,
            "num_edges": len(self.edges),
            "pinned_vertices": int(np.count_nonzero(self.pinned)),
        }
