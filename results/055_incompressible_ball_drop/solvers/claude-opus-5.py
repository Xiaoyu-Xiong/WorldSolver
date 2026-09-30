"""Incompressible soft elastic ball dropped on a rigid floor.

Physical model
-------------
* Volumetric FEM.  The frozen closed surface (12 nodes / 20 triangles) is
  tetrahedralised by coning every surface triangle onto the body centroid.  The
  centroid is kinematically slaved to the mean of the surface nodes, so the only
  degrees of freedom are exactly the 12 samples demanded by the frozen state
  binding ``selector: all``.  Because the slaving map is linear and every
  element energy is translation invariant, the assembled internal forces sum to
  zero and total momentum is governed purely by the external terms.
* Material: nearly incompressible neo-Hookean solid,
      Psi = mu/2 (tr(F^T F) - 3) - mu ln J + lambda/2 (ln J)^2 ,  lambda >> mu,
  with a Kelvin-Voigt viscous first-Piola contribution eta * dF/dt.
* Contact: penalty normal contact plus regularised Coulomb friction against the
  frozen rigid floor (top face of the ground box), evaluated per node.  A convex
  surface whose vertices all satisfy z >= 0 never penetrates the plane.
* Time stepping: symplectic (semi-implicit) Euler at a fixed sub-step that
  resolves the dilatational wave speed and the contact frequency.

Nothing is scripted: every sampled frame is the state produced by stepping the
discretised equations of motion from the frozen initial condition.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_GRAVITY = 9.81


class BenchmarkSolver:
    """Explicit near-incompressible FEM solver for the soft ball drop."""

    # --- material and numerics -------------------------------------------
    shear_modulus = 50.0        # Pa, soft rubber-like ball (large flattening)
    bulk_lambda = 2.0e5         # Pa, lambda/mu = 4000 -> nearly incompressible
    viscosity = 0.4             # Pa*s, Kelvin-Voigt damping on dF/dt
    # Near-critically damped penalty contact (zeta ~ 0.7 at the nodal mass):
    # ringing is suppressed within one sample interval, so the exported contact
    # force is a faithful sample rather than an aliased snapshot.
    contact_stiffness = 2.0e4   # N/m per node
    contact_damping = 120.0     # N*s/m per node
    friction_coefficient = 0.3
    friction_regularisation = 40.0
    sub_dt_target = 4.0e-5      # s
    j_floor = 0.2               # clamp used only inside log terms

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})

        frame0 = scene.initial_state()
        self.rest = np.array(frame0.position, dtype=float)
        self.x = self.rest.copy()
        self.v = np.array(frame0.velocity, dtype=float)
        self.n = int(self.rest.shape[0])
        if frame0.mass is None:
            self.mass = np.full(self.n, 1.0 / self.n, dtype=float)
        else:
            self.mass = np.array(frame0.mass, dtype=float)
        self.total_mass = float(self.mass.sum())
        self.inv_mass = (1.0 / self.mass)[:, None]

        self.gravity = np.array([0.0, 0.0, -_GRAVITY], dtype=float)
        self.floor_z = self._floor_height(config)

        self._build_tetrahedra()

        sample_dt = float(config.get("sample_dt", 0.02) or 0.02)
        self.sub_steps = max(1, int(round(sample_dt / self.sub_dt_target)))
        self.dt = sample_dt / self.sub_steps
        self.sample_dt = sample_dt

        self.steps = 0
        self.sim_time = 0.0
        self.impulse = np.zeros(3, dtype=float)
        self.nonfinite_guards = 0

        drop = max(float(np.min(self.rest[:, 2]) - self.floor_z), 1e-3)
        self.reference_momentum = float(self.total_mass * np.sqrt(2.0 * _GRAVITY * drop))
        self.constraint_scale = float(np.mean(self.rest_volumes))
        self.domain_weights = np.ones((1, self.n), dtype=float)
        self.zero_momentum = np.zeros((1, 3), dtype=float)
        self.mass_source = np.zeros(1, dtype=float)
        self.last_contact = np.zeros(3, dtype=float)
    # ------------------------------------------------------------- geometry
    @staticmethod
    def _floor_height(config: dict) -> float:
        """Top surface of the frozen static ground box (fallback: z = 0)."""
        geometry = config.get("scene_geometry") or {}
        for item in geometry.get("objects", []) or []:
            if item.get("semantic") != "boundary" and item.get("motion") != "static":
                continue
            shape = item.get("geometry", {}) or {}
            if shape.get("type") != "box":
                continue
            half = shape.get("half_extents") or [1.0, 1.0, 1.0]
            translation = (item.get("transform", {}) or {}).get(
                "translation", [0.0, 0.0, 0.0]
            )
            return float(translation[2]) + float(half[2])
        return 0.0

    def _surface_faces(self) -> np.ndarray:
        """Frozen closed-surface triangles: OBJ topology, hull as fallback."""
        config = dict(getattr(self.scene, "config", {}) or {})
        geometry = config.get("scene_geometry") or {}
        for item in geometry.get("objects", []) or []:
            asset = (item.get("geometry", {}) or {}).get("asset")
            if not asset:
                continue
            try:
                faces = []
                with open(asset, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.startswith("f "):
                            continue
                        tokens = line.split()[1:]
                        loop = [int(tok.split("/")[0]) - 1 for tok in tokens]
                        for k in range(1, len(loop) - 1):
                            faces.append((loop[0], loop[k], loop[k + 1]))
                candidate = np.asarray(faces, dtype=int)
                if candidate.size and int(candidate.max()) < self.n:
                    return candidate
            except OSError:
                continue
        from scipy.spatial import ConvexHull

        return np.asarray(ConvexHull(self.rest).simplices, dtype=int)
    def _build_tetrahedra(self) -> None:
        """Cone every surface triangle onto the centroid node (index self.n)."""
        faces = self._surface_faces()
        centre = int(self.n)
        rest_aug = np.vstack((self.rest, self.rest.mean(axis=0)[None, :]))

        tets = np.column_stack(
            (faces[:, 0], faces[:, 1], faces[:, 2], np.full(len(faces), centre))
        ).astype(int)

        # Reference edge matrices; flip winding so every rest volume is positive.
        edges = np.stack(
            (
                rest_aug[tets[:, 1]] - rest_aug[tets[:, 0]],
                rest_aug[tets[:, 2]] - rest_aug[tets[:, 0]],
                rest_aug[tets[:, 3]] - rest_aug[tets[:, 0]],
            ),
            axis=2,
        )
        det = np.linalg.det(edges)
        flip = det < 0.0
        if np.any(flip):
            tets[flip] = tets[flip][:, [0, 2, 1, 3]]
            edges = np.stack(
                (
                    rest_aug[tets[:, 1]] - rest_aug[tets[:, 0]],
                    rest_aug[tets[:, 2]] - rest_aug[tets[:, 0]],
                    rest_aug[tets[:, 3]] - rest_aug[tets[:, 0]],
                ),
                axis=2,
            )
            det = np.linalg.det(edges)

        keep = np.abs(det) > 1e-12
        tets, edges, det = tets[keep], edges[keep], det[keep]

        self.tets = tets
        self.dm_inv = np.linalg.inv(edges)
        self.rest_volumes = det / 6.0
        self.n_elements = int(len(tets))
        self.rest_volume_total = float(self.rest_volumes.sum())
        self.identity = np.eye(3)[None, :, :]
        self._flat_nodes = tets.reshape(-1)

    # -------------------------------------------------------------- kinematics
    def _augment(self, nodal: np.ndarray) -> np.ndarray:
        """Append the slaved centroid value (mean of the surface nodes)."""
        return np.vstack((nodal, nodal.mean(axis=0)[None, :]))

    def _deformation_gradients(self, x_aug: np.ndarray) -> np.ndarray:
        ds = np.stack(
            (
                x_aug[self.tets[:, 1]] - x_aug[self.tets[:, 0]],
                x_aug[self.tets[:, 2]] - x_aug[self.tets[:, 0]],
                x_aug[self.tets[:, 3]] - x_aug[self.tets[:, 0]],
            ),
            axis=2,
        )
        return ds @ self.dm_inv
    # ------------------------------------------------------------- mechanics
    def _internal_forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Neo-Hookean + Kelvin-Voigt nodal forces (sum to zero by construction)."""
        x_aug = self._augment(x)
        v_aug = self._augment(v)
        f_grad = self._deformation_gradients(x_aug)
        f_rate = self._deformation_gradients(v_aug)

        cof = np.empty_like(f_grad)
        c0, c1, c2 = f_grad[:, :, 0], f_grad[:, :, 1], f_grad[:, :, 2]
        cof[:, :, 0] = np.cross(c1, c2)
        cof[:, :, 1] = np.cross(c2, c0)
        cof[:, :, 2] = np.cross(c0, c1)
        det = np.einsum("ij,ij->i", c0, cof[:, :, 0])
        det_safe = np.maximum(det, self.j_floor)

        coefficient = (self.bulk_lambda * np.log(det_safe) - self.shear_modulus) / det_safe
        piola = (
            self.shear_modulus * f_grad
            + coefficient[:, None, None] * cof
            + self.viscosity * f_rate
        )

        h = -self.rest_volumes[:, None, None] * (piola @ np.swapaxes(self.dm_inv, 1, 2))
        element_forces = np.empty((self.n_elements, 4, 3), dtype=float)
        element_forces[:, 1, :] = h[:, :, 0]
        element_forces[:, 2, :] = h[:, :, 1]
        element_forces[:, 3, :] = h[:, :, 2]
        element_forces[:, 0, :] = -(h[:, :, 0] + h[:, :, 1] + h[:, :, 2])

        flat = element_forces.reshape(-1, 3)
        size = self.n + 1
        forces_aug = np.stack(
            [
                np.bincount(self._flat_nodes, weights=flat[:, k], minlength=size)
                for k in range(3)
            ],
            axis=1,
        )
        # Chain rule for the slaved centroid: d x_centre / d x_i = I / n.
        return forces_aug[: self.n] + forces_aug[self.n] / self.n

    def _contact_forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Penalty normal contact and regularised Coulomb friction on the floor."""
        forces = np.zeros((self.n, 3), dtype=float)
        penetration = self.floor_z - x[:, 2]
        active = np.flatnonzero(penetration > 0.0)
        if active.size == 0:
            return forces

        depth = penetration[active]
        approach = -v[active, 2]
        normal = np.maximum(
            self.contact_stiffness * depth + self.contact_damping * approach, 0.0
        )
        tangent_velocity = v[active, :2]
        speed = np.linalg.norm(tangent_velocity, axis=1)
        magnitude = np.minimum(
            self.friction_coefficient * normal, self.friction_regularisation * speed
        )
        scale = np.where(speed > 1e-12, magnitude / np.maximum(speed, 1e-12), 0.0)
        forces[active, 2] = normal
        forces[active, 0] = -scale * tangent_velocity[:, 0]
        forces[active, 1] = -scale * tangent_velocity[:, 1]
        return forces

    def _step(self) -> None:
        x_prev = self.x
        v_prev = self.v
        internal = self._internal_forces(self.x, self.v)
        contact = self._contact_forces(self.x, self.v)
        gravity = self.mass[:, None] * self.gravity[None, :]
        external = gravity + contact

        velocity = self.v + self.dt * (internal + external) * self.inv_mass
        position = self.x + self.dt * velocity
        if not (np.isfinite(velocity).all() and np.isfinite(position).all()):
            # Never emit non-finite state; freeze the offending sub-step instead.
            self.nonfinite_guards += 1
            self.x = x_prev.copy()
            self.v = np.zeros_like(v_prev)
        else:
            self.x = position
            self.v = velocity
            self.impulse = self.impulse + self.dt * external.sum(axis=0)
        self.steps += 1
    # ---------------------------------------------------------------- driver
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.dt))
        while self.steps < target:
            self._step()
        self.sim_time = self.steps * self.dt
        return self._frame(float(time))

    def _frame(self, time: float) -> StateFrame:
        f_grad = self._deformation_gradients(self._augment(self.x))
        det = np.linalg.det(f_grad)
        volumes = det * self.rest_volumes

        contact = self._contact_forces(self.x, self.v)
        gravity_total = self.total_mass * self.gravity
        contact_total = contact.sum(axis=0)
        self.last_contact = contact_total

        external_force = np.empty((1, 2, 3), dtype=float)
        external_force[0, 0, :] = gravity_total
        external_force[0, 1, :] = contact_total

        fields = {
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": external_force,
            "budget_cumulative_external_impulse": self.impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": self.zero_momentum.copy(),
            "budget_cumulative_outflow_momentum": self.zero_momentum.copy(),
            "cumulative_mass_source": self.mass_source.copy(),
            "element_deformation_gradient": np.ascontiguousarray(f_grad),
            "constraint_value": np.ascontiguousarray(volumes),
            "constraint_target": np.ascontiguousarray(self.rest_volumes),
        }
        scalars = {
            "constraint_scale": float(self.constraint_scale),
            "reference_momentum": float(self.reference_momentum),
        }
        return StateFrame(
            time=float(time),
            position=np.ascontiguousarray(self.x.copy()),
            velocity=np.ascontiguousarray(self.v.copy()),
            mass=self.mass.copy(),
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        f_grad = self._deformation_gradients(self._augment(self.x))
        det = np.linalg.det(f_grad)
        volume = float((det * self.rest_volumes).sum())
        return {
            "method": "explicit neo-Hookean FEM (coned surface tets), penalty contact",
            "nodes": int(self.n),
            "elements": int(self.n_elements),
            "sub_dt": float(self.dt),
            "sub_steps_per_sample": int(self.sub_steps),
            "steps_taken": int(self.steps),
            "floor_z": float(self.floor_z),
            "shear_modulus": float(self.shear_modulus),
            "bulk_lambda": float(self.bulk_lambda),
            "rest_volume": float(self.rest_volume_total),
            "final_volume": volume,
            "final_volume_ratio": volume / self.rest_volume_total,
            "final_height": float(self.x[:, 2].max() - self.x[:, 2].min()),
            "final_width": float(
                np.linalg.norm(self.x[:, :2] - self.x[:, :2].mean(axis=0), axis=1).max()
            ),
            "cumulative_external_impulse_z": float(self.impulse[2]),
            "momentum_z": float((self.mass[:, None] * self.v)[:, 2].sum()),
            "nonfinite_guards": int(self.nonfinite_guards),
        }

