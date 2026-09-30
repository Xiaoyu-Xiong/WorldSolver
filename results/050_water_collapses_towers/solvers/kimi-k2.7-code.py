"""Physically-based SPH + rigid-particle solver for the water-collapses-towers scene.

The solver integrates point masses representing the incoming water (SPH fluid)
and the two mixed-density towers (rigid particles).  Pairwise spring-damper
contacts couple the phases, while basin walls supply external boundary forces.
All required trajectory fields are exported through ``worldsolver.contracts.StateFrame``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic SPH/rigid-particle physics solver."""

    def __init__(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.seed = seed

        # Load frozen initial frame.
        init: StateFrame = scene.initial_state()
        init.validate()

        self.time: float = float(init.time)
        self.position: np.ndarray = np.asarray(init.position, dtype=float).copy()
        self.velocity: np.ndarray = np.asarray(init.velocity, dtype=float).copy()
        self.orientation: np.ndarray = np.asarray(init.orientation, dtype=float).copy()
        self.angular_velocity: np.ndarray = np.asarray(init.angular_velocity, dtype=float).copy()
        self.mass: np.ndarray = np.asarray(init.mass, dtype=float).copy()
        self.body_id: np.ndarray = np.asarray(init.body_id, dtype=int).copy()

        self.attributes: dict[str, np.ndarray] = {
            name: np.asarray(value, dtype=float).copy()
            for name, value in init.attributes.items()
        }
        self.fields: dict[str, np.ndarray] = {
            name: np.asarray(value, dtype=float).copy()
            for name, value in init.fields.items()
        }
        self.scalars: dict[str, float] = dict(init.scalars)

        self.n = len(self.position)

        # Material/phase identification.
        self.scene_group = np.asarray(self.attributes["scene_group"], dtype=int)
        self.is_water = self.scene_group == 10
        self.is_rigid = ~self.is_water
        self.water_idx = np.nonzero(self.is_water)[0]
        self.rigid_idx = np.nonzero(self.is_rigid)[0]

        # Densities (kg/m^3) – water rest density, rigid from scaffold.
        self.density = np.asarray(self.attributes["density"], dtype=float)

        # Effective particle radius.
        self.radius = np.empty(self.n, dtype=float)
        # Water: equivalent sphere from mass/density.
        self.radius[self.is_water] = (
            3.0 * self.mass[self.is_water] / (4.0 * np.pi * self.density[self.is_water])
        ) ** (1.0 / 3.0)
        # Rigid bodies use the frozen geometry radius/half-extent.
        self.radius[self.is_rigid] = 0.115

        # Material grouping for trajectory fields (water, light, heavy).
        self.material_index = np.full(self.n, -1, dtype=int)
        self.material_index[self.is_water] = 0
        self.material_index[(self.scene_group == 20) | (self.scene_group == 30)] = 1
        self.material_index[(self.scene_group == 21) | (self.scene_group == 31)] = 2

        # Gravity from scaffold.
        self.gravity = np.asarray([0.0, 0.0, -9.81], dtype=float)

        # Basin bounds.
        self.basin_min = np.array([-4.0, -2.0, 0.0], dtype=float)
        self.basin_max = np.array([4.0, 2.0, 2.2], dtype=float)

        # SPH parameters.
        self.rho0 = 1000.0
        self.h_sph = 0.24
        self.c_s = 40.0
        self.gamma = 7.0
        self.B = (self.c_s ** 2) * self.rho0 / self.gamma
        self.alpha = 0.05
        self.rho = self.density.copy().astype(float)

        # Contact parameters.
        self.k_contact = 1.0e5
        self.zeta_contact = 0.3
        self.contact_margin = 0.05
        self.mu_rigid = 0.35
        self.mu_water_rigid = 0.10

        # Time stepping.
        self.dt = 0.001
        self.max_time = 6.0

        self._initialized = True

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("Solver has not been initialized")

        target = float(time)
        if target < self.time - 1e-12:
            raise ValueError(f"advance_to target {target} is before current time {self.time}")

        while self.time < target - 1e-12:
            step_dt = min(self.dt, target - self.time)
            self._step(step_dt)
            self.time += step_dt

        return self._build_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "sph_rigid_particle_baseline",
            "particles": self.n,
            "water_particles": int(self.is_water.sum()),
            "rigid_particles": int(self.is_rigid.sum()),
            "time": self.time,
            "dt": self.dt,
        }

    # ------------------------------------------------------------------
    # Physics core
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        n = self.n
        force = np.zeros((n, 3), dtype=float)

        # External force: gravity on all particles.
        force += self.gravity * self.mass[:, None]

        # SPH density and pressure forces for water.
        self.rho[self.is_water] = 0.0
        self._sph_density()
        self._sph_forces(force)

        # Pair contact forces (water-rigid and rigid-rigid).
        signed_distances = []
        pair_forces = []
        self._contact_forces(force, signed_distances, pair_forces)

        # Boundary forces (external) and boundary signed distances.
        self._boundary_forces(force, signed_distances)

        # Symplectic Euler integration.
        self.velocity += dt * force / self.mass[:, None]
        self.position += dt * self.velocity

        # Hard clamp against boundaries to avoid runaway penetration.
        self._clamp_to_basin()

        # Store current contact diagnostics for next frame export.
        self._last_signed = np.asarray(signed_distances, dtype=float) if signed_distances else np.empty((0,))
        self._last_pair_forces = (
            np.asarray(pair_forces, dtype=float) if pair_forces else np.empty((0, 3))
        )

    # ------------------------------------------------------------------
    # SPH helpers
    # ------------------------------------------------------------------
    def _kernel(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cubic spline kernel and its radial derivative dW/dr."""
        h = self.h_sph
        q = r / h
        sigma = 1.0 / (np.pi * h ** 3)
        w = np.zeros_like(r)
        dw = np.zeros_like(r)

        m1 = q <= 1.0
        w[m1] = sigma * (1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3)
        dw[m1] = sigma * (-3.0 * q[m1] + 2.25 * q[m1] ** 2) / h

        m2 = (q > 1.0) & (q <= 2.0)
        t = 2.0 - q[m2]
        w[m2] = 0.25 * sigma * t ** 3
        dw[m2] = -0.75 * sigma * t ** 2 / h

        return w, dw

    def _sph_density(self) -> None:
        if len(self.water_idx) == 0:
            return

        pos_w = self.position[self.is_water]
        tree = cKDTree(pos_w)
        pairs = tree.sparse_distance_matrix(tree, self.h_sph, output_type="coo_matrix")
        i = pairs.row
        j = pairs.col
        d = pairs.data
        keep = (i != j) & (i < j)
        i = i[keep]
        j = j[keep]
        d = d[keep]

        w, _ = self._kernel(d)
        # Local water indices -> global
        gi = self.water_idx[i]
        gj = self.water_idx[j]
        mi = self.mass[gi]
        mj = self.mass[gj]

        np.add.at(self.rho, gi, mj * w)
        np.add.at(self.rho, gj, mi * w)

        # Self-contribution for isolated or low-neighbour particles.
        w0, _ = self._kernel(np.zeros(1))
        self.rho[self.is_water] += self.mass[self.is_water] * w0[0]

        # Guard against zero density.
        self.rho = np.maximum(self.rho, 1.0)

    def _sph_forces(self, force: np.ndarray) -> None:
        if len(self.water_idx) < 2:
            return

        pos_w = self.position[self.is_water]
        tree = cKDTree(pos_w)
        pairs = tree.sparse_distance_matrix(tree, self.h_sph, output_type="coo_matrix")
        i = pairs.row
        j = pairs.col
        d = pairs.data
        keep = (i != j) & (i < j)
        i = i[keep]
        j = j[keep]
        d = d[keep]

        gi = self.water_idx[i]
        gj = self.water_idx[j]
        mi = self.mass[gi]
        mj = self.mass[gj]

        # Pressure
        p = np.maximum(0.0, self.B * ((self.rho[self.is_water] / self.rho0) ** self.gamma - 1.0))
        pi = p[i] / self.rho[gi] ** 2
        pj = p[j] / self.rho[gj] ** 2

        w, dw = self._kernel(d)
        rvec = self.position[gj] - self.position[gi]
        grad_w = (dw / (d[:, None] + 1e-12)) * rvec

        f_term = -mi * mj * (pi + pj)[:, None] * grad_w
        np.add.at(force, gi, f_term)
        np.add.at(force, gj, -f_term)

        # Artificial viscosity
        vij = self.velocity[gi] - self.velocity[gj]
        rv = np.einsum("ij,ij->i", rvec, vij)
        rho_avg = 0.5 * (self.rho[gi] + self.rho[gj])
        mu = self.h_sph * rv / (d ** 2 + 0.01 * self.h_sph ** 2)
        pi_visc = np.where(rv < 0.0, -self.alpha * self.c_s * mu / rho_avg, 0.0)
        f_visc = mi * mj * pi_visc[:, None] * grad_w
        np.add.at(force, gi, f_visc)
        np.add.at(force, gj, -f_visc)

    # ------------------------------------------------------------------
    # Contact mechanics
    # ------------------------------------------------------------------
    def _contact_forces(
        self,
        force: np.ndarray,
        signed_distances: list[float],
        pair_forces: list[list[float]],
    ) -> None:
        if len(self.rigid_idx) == 0:
            return

        # Water-rigid contacts.
        if len(self.water_idx) > 0:
            self._pair_contact(
                self.water_idx,
                self.rigid_idx,
                self.mu_water_rigid,
                force,
                signed_distances,
                pair_forces,
            )

        # Rigid-rigid contacts.
        if len(self.rigid_idx) > 1:
            self._pair_contact(
                self.rigid_idx,
                self.rigid_idx,
                self.mu_rigid,
                force,
                signed_distances,
                pair_forces,
            )

    def _pair_contact(
        self,
        idx_a: np.ndarray,
        idx_b: np.ndarray,
        mu: float,
        force: np.ndarray,
        signed_distances: list[float],
        pair_forces: list[list[float]],
    ) -> None:
        pos_a = self.position[idx_a]
        pos_b = self.position[idx_b]
        tree_a = cKDTree(pos_a)
        tree_b = cKDTree(pos_b)

        max_d = (
            np.max(self.radius[idx_a]) + np.max(self.radius[idx_b]) + self.contact_margin
        )
        pairs = tree_a.sparse_distance_matrix(tree_b, max_d, output_type="coo_matrix")

        i = pairs.row
        j = pairs.col
        d = pairs.data

        if len(idx_a) == len(idx_b) and np.array_equal(idx_a, idx_b):
            # Same set; avoid self and double counting.
            keep = (i != j) & (i < j)
        else:
            keep = (i != j)
        i = i[keep]
        j = j[keep]
        d = d[keep]

        gi = idx_a[i]
        gj = idx_b[j]

        r_sum = self.radius[gi] + self.radius[gj]
        signed = d - r_sum
        active = signed < self.contact_margin
        gi = gi[active]
        gj = gj[active]
        d = d[active]
        signed = signed[active]

        if len(gi) == 0:
            return

        n = (self.position[gj] - self.position[gi]) / (d[:, None] + 1e-12)
        delta = -signed
        delta = np.maximum(delta, 0.0)

        v_rel = np.einsum("ij,ij->i", self.velocity[gi] - self.velocity[gj], n)
        m_eff = (self.mass[gi] * self.mass[gj]) / (self.mass[gi] + self.mass[gj] + 1e-12)
        c = 2.0 * self.zeta_contact * np.sqrt(self.k_contact * m_eff)

        fn_mag = self.k_contact * delta - c * np.maximum(v_rel, 0.0)
        fn_mag = np.maximum(fn_mag, 0.0)

        # Tangential friction.
        v_t = (self.velocity[gi] - self.velocity[gj]) - v_rel[:, None] * n
        v_t_mag = np.linalg.norm(v_t, axis=1)
        # Avoid division by zero; friction zero if no tangential motion.
        mask = v_t_mag > 1e-9
        fric = np.zeros_like(v_t)
        fric[mask] = -mu * fn_mag[mask][:, None] * (v_t[mask] / v_t_mag[mask][:, None])

        contact_force = fn_mag[:, None] * n + fric

        np.add.at(force, gi, contact_force)
        np.add.at(force, gj, -contact_force)

        # Record signed distances and interaction forces on the acting body.
        signed_distances.extend(signed.tolist())
        pair_forces.extend(contact_force.tolist())

    # ------------------------------------------------------------------
    # Boundary mechanics
    # ------------------------------------------------------------------
    def _boundary_forces(
        self,
        force: np.ndarray,
        signed_distances: list[float],
    ) -> None:
        # Walls: x_min, x_max, y_min, y_max, z_min (floor). Top is open.
        planes = [
            (self.basin_min[0], np.array([-1.0, 0.0, 0.0]), 0),
            (self.basin_max[0], np.array([1.0, 0.0, 0.0]), 0),
            (self.basin_min[1], np.array([0.0, -1.0, 0.0]), 1),
            (self.basin_max[1], np.array([0.0, 1.0, 0.0]), 1),
            (self.basin_min[2], np.array([0.0, 0.0, -1.0]), 2),
        ]

        for limit, normal, axis in planes:
            signed = (self.position[:, axis] - limit) * (-normal[axis])
            # signed < margin are active/near contacts
            active = signed < self.contact_margin
            if not np.any(active):
                continue

            delta = np.maximum(-signed[active], 0.0)
            signed_distances.extend(signed[active].tolist())

            if np.all(delta == 0.0):
                continue

            v = self.velocity[active]
            v_approach = -np.einsum("ij,j->i", v, normal)
            m_eff = self.mass[active]
            c = 2.0 * self.zeta_contact * np.sqrt(self.k_contact * m_eff)

            fn_mag = self.k_contact * delta - c * np.maximum(v_approach, 0.0)
            fn_mag = np.maximum(fn_mag, 0.0)

            boundary_force = fn_mag[:, None] * normal[None, :]
            force[active] += boundary_force

    def _clamp_to_basin(self) -> None:
        eps = 1e-6
        # Clamp position
        for axis in range(3):
            too_low = self.position[:, axis] < self.basin_min[axis]
            self.position[too_low, axis] = self.basin_min[axis] + eps
            too_high = self.position[:, axis] > self.basin_max[axis]
            self.position[too_high, axis] = self.basin_max[axis] - eps

    # ------------------------------------------------------------------
    # Frame export
    # ------------------------------------------------------------------
    def _build_frame(self) -> StateFrame:
        # Per-material mass quantity [N, 3].
        mass_quantity = np.zeros((self.n, 3), dtype=float)
        for k in range(3):
            mask = self.material_index == k
            mass_quantity[mask, k] = self.mass[mask]

        # Gravity.
        external_force = np.zeros((self.n, 3), dtype=float)
        external_force += self.gravity * self.mass[:, None]
        # Boundary contribution.
        signed: list[float] = []
        self._boundary_forces(external_force, signed)

        # Pair interaction forces and signed distances.
        pair_signed: list[float] = []
        pair_forces: list[list[float]] = []
        force_dummy = np.zeros((self.n, 3), dtype=float)
        self._contact_forces(force_dummy, pair_signed, pair_forces)

        signed_all = np.asarray(signed + pair_signed, dtype=float)
        interaction = np.asarray(pair_forces, dtype=float) if pair_forces else np.empty((0, 3))

        # Cumulative mass source: no sources/sinks in this scene.
        cumulative_mass_source = np.zeros(3, dtype=float)

        # Contact length scale: representative element size.
        contact_length_scale = 0.115

        attributes = dict(self.attributes)
        attributes["mass_quantity"] = mass_quantity

        fields = dict(self.fields)
        fields["contact_signed_distance"] = signed_all
        fields["cumulative_mass_source"] = cumulative_mass_source
        fields["external_force"] = external_force
        fields["interaction_force"] = interaction
        fields["density"] = self.density.copy()

        scalars = dict(self.scalars)
        scalars["contact_length_scale"] = float(contact_length_scale)

        frame = StateFrame(
            time=float(self.time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame
