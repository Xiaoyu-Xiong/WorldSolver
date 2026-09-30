"""Physically-based particle solver for the Soluble Bunny Water scene.

The solver treats the bunny as a deformable solid and the pipe flow as
incompressible-ish particles.  Red solute is conserved: it leaves the bunny
and is advected/diffused by the water.  A background diagnostic grid exports
the scalar and velocity fields required by the trajectory contract.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.scene = scene
        self.config = getattr(scene, "config", {})

        # ----- physical parameters from the frozen configuration -----
        self.gravity_vec = np.array(
            self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        )
        ic = self.config.get("initial_conditions", {})
        self.inflow_speed = float(ic.get("inflow_speed_m_s", 2.4))
        self.inflow_start = float(ic.get("inflow_start_s", 0.0))
        self.inflow_end = float(ic.get("inflow_end_s", 5.5))
        self.water_density = 1000.0
        self.bunny_density = 1180.0

        # ----- copy initial state -----
        frame0: StateFrame = scene.initial_state()
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = (
            np.asarray(frame0.mass, dtype=float).copy()
            if frame0.mass is not None
            else np.ones(len(self.position), dtype=float)
        )
        self.body_id = (
            np.asarray(frame0.body_id, dtype=int).copy()
            if frame0.body_id is not None
            else np.zeros(len(self.position), dtype=int)
        )

        attrs: Mapping[str, Any] = frame0.attributes
        self.label = np.asarray(attrs["material_label"], dtype=int).copy()
        self.concentration = np.asarray(
            attrs.get("red_solute_concentration", np.zeros(len(self.position))), dtype=float
        ).copy()
        self.phase = np.asarray(
            attrs.get("solid_phase_fraction", np.zeros(len(self.position)), dtype=float)
        ).copy()
        self.density = np.asarray(
            attrs.get("density", np.where(self.label == 1, self.bunny_density, self.water_density)),
            dtype=float,
        ).copy()

        # solute mass carried by each particle (kg)
        self.solute_mass = self.concentration * self.mass

        # deformation gradient, identity for all particles initially
        self.F = np.tile(np.eye(3, dtype=float), (len(self.position), 1, 1))

        # keep original base positions so the receiving layer can stay static
        self.initial_position = self.position.copy()

        # ----- solver knobs -----
        self.substep = 0.001
        self.cutoff = 0.09
        self.target_sep = 0.085
        self.k_water = 1.0e3
        self.k_solid_water = 4.0e3
        self.k_solid = 8.0e3
        self.spring_cutoff = 0.16
        self.k_spring = 2.5e3
        self.damping = 8.0
        self.dissolve_rate = 0.25
        self.diffusivity = 0.05
        self.floor_z = 0.0

        # ----- bunny spring network -----
        self.spring_i = np.zeros(0, dtype=int)
        self.spring_j = np.zeros(0, dtype=int)
        self.spring_rest = np.zeros(0, dtype=float)
        self._build_bunny_springs()

        # ----- diagnostic grid -----
        self.h = 0.08
        self.domain_min = np.array([-0.92, -1.25, 0.04], dtype=float)
        self.domain_max = np.array([2.45, 1.25, 1.65], dtype=float)
        self.grid_shape = tuple(
            np.maximum(1, np.ceil((self.domain_max - self.domain_min) / self.h).astype(int))
        )

        # ----- bookkeeping -----
        self.time = 0.0
        self.cum_mass_source = np.zeros(3, dtype=float)
        self.interaction_force = np.zeros((3, 3), dtype=float)
        self._contact_force = np.zeros_like(self.velocity)
        self._scalar_source = np.zeros(self.grid_shape, dtype=float)
        self._grid_v = np.zeros(self.grid_shape + (3,), dtype=float)
        self._grid_c = np.zeros(self.grid_shape, dtype=float)
        self._active_mask = np.zeros(self.grid_shape, dtype=bool)
        self.dissolved_mass_total = 0.0
        self.inflow_mass_total = 0.0
        pipe_radius = 0.38
        self.inflow_mass_rate = (
            self.water_density * self.inflow_speed * math.pi * pipe_radius ** 2
        )

        # initialize grid / diagnostics from the frozen frame
        self._update_grid()

    # ------------------------------------------------------------------
    def _build_bunny_springs(self) -> None:
        bunny_ids = np.where(self.label == 1)[0]
        if bunny_ids.size == 0:
            return
        pos = self.position[bunny_ids]
        tree = cKDTree(pos)
        pairs = tree.query_pairs(self.spring_cutoff, output_type="ndarray")
        if pairs.shape[0] == 0:
            return
        i = bunny_ids[pairs[:, 0]]
        j = bunny_ids[pairs[:, 1]]
        self.spring_i = i
        self.spring_j = j
        self.spring_rest = np.linalg.norm(
            self.position[i] - self.position[j], axis=1
        )

    # ------------------------------------------------------------------
    def _apply_spring_forces(self) -> np.ndarray:
        n = len(self.position)
        force = np.zeros((n, 3), dtype=float)
        if self.spring_i.size == 0:
            return force
        pi = self.position[self.spring_i]
        pj = self.position[self.spring_j]
        rij = pj - pi
        dist = np.linalg.norm(rij, axis=1)
        safe = np.where(dist < 1e-9, 1.0, dist)
        # spring force = k * (dist - rest) / dist * rij
        f = self.k_spring * ((dist - self.spring_rest) / safe)[:, None] * rij
        # damping along the separation direction
        vi = self.velocity[self.spring_i]
        vj = self.velocity[self.spring_j]
        vrel = vj - vi
        proj = np.sum(vrel * (rij / safe[:, None]), axis=1)
        f += (self.damping * proj / safe)[:, None] * rij
        np.add.at(force, self.spring_i, f)
        np.add.at(force, self.spring_j, -f)
        return force

    # ------------------------------------------------------------------
    def _apply_pair_forces(self) -> np.ndarray:
        n = len(self.position)
        force = np.zeros((n, 3), dtype=float)
        if n < 2:
            self._contact_force = force
            return force

        tree = cKDTree(self.position)
        pairs = tree.query_pairs(self.cutoff, output_type="ndarray")
        if pairs.shape[0] == 0:
            self._contact_force = force
            self._update_interaction(force)
            return force

        i = pairs[:, 0]
        j = pairs[:, 1]
        rij = self.position[j] - self.position[i]
        dist = np.linalg.norm(rij, axis=1)
        dist = np.where(dist < 1e-9, 1e9, dist)

        li = self.label[i]
        lj = self.label[j]
        # stiffness depends on whether a solid (bunny) is involved
        k = np.full(len(i), self.k_water)
        solid_pair = (li == 1) | (lj == 1)
        k = np.where(solid_pair, self.k_solid_water, k)
        both_solid = (li == 1) & (lj == 1)
        k = np.where(both_solid, self.k_solid, k)

        mag = k * (self.target_sep - dist)
        active = dist < self.target_sep
        mag = np.where(active, mag, 0.0)
        f = (mag / dist)[:, None] * rij

        # contact damping
        vrel = self.velocity[j] - self.velocity[i]
        proj = np.sum(vrel * (rij / dist[:, None]), axis=1)
        fd = self.damping * proj
        fd = np.where(active, fd, 0.0)
        f += (fd / dist)[:, None] * rij

        np.add.at(force, i, f)
        np.add.at(force, j, -f)

        self._contact_force = force
        self._update_interaction(force)
        return force

    # ------------------------------------------------------------------
    def _update_interaction(self, contact: np.ndarray) -> None:
        interaction = np.zeros((3, 3), dtype=float)
        for k in range(3):
            ids = np.where(self.label == k + 1)[0]
            if ids.size:
                interaction[k] = contact[ids].sum(axis=0)
        self.interaction_force = interaction

    # ------------------------------------------------------------------
    def _cell_index(self, pos: np.ndarray) -> np.ndarray:
        idx = ((pos - self.domain_min) / self.h).astype(int)
        idx = np.clip(idx, 0, np.array(self.grid_shape) - 1)
        return idx

    # ------------------------------------------------------------------
    def _update_grid(self) -> None:
        cell = self._cell_index(self.position)
        grid_m = np.zeros(self.grid_shape, dtype=float)
        grid_mom = np.zeros(self.grid_shape + (3,), dtype=float)
        np.add.at(grid_m, (cell[:, 0], cell[:, 1], cell[:, 2]), self.mass)
        for d in range(3):
            np.add.at(
                grid_mom[..., d],
                (cell[:, 0], cell[:, 1], cell[:, 2]),
                self.mass * self.velocity[:, d],
            )

        grid_v = np.zeros_like(grid_mom)
        mask = grid_m > 0
        grid_v[mask] = grid_mom[mask] / grid_m[mask][:, None]

        grid_c = np.zeros(self.grid_shape, dtype=float)
        np.add.at(
            grid_c,
            (cell[:, 0], cell[:, 1], cell[:, 2]),
            self.mass * self.concentration,
        )
        grid_c[mask] /= grid_m[mask]

        self._grid_v = grid_v
        self._grid_c = grid_c
        self._active_mask = mask

    # ------------------------------------------------------------------
    def _velocity_gradient_at(self, grid_v: np.ndarray, cell: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.grid_shape
        i = cell[:, 0]
        j = cell[:, 1]
        k = cell[:, 2]
        im1 = np.clip(i - 1, 0, nx - 1)
        ip1 = np.clip(i + 1, 0, nx - 1)
        jm1 = np.clip(j - 1, 0, ny - 1)
        jp1 = np.clip(j + 1, 0, ny - 1)
        km1 = np.clip(k - 1, 0, nz - 1)
        kp1 = np.clip(k + 1, 0, nz - 1)

        dv_dx = (grid_v[ip1, j, k] - grid_v[im1, j, k]) / (2.0 * self.h)
        dv_dy = (grid_v[i, jp1, k] - grid_v[i, jm1, k]) / (2.0 * self.h)
        dv_dz = (grid_v[i, j, kp1] - grid_v[i, j, km1]) / (2.0 * self.h)

        # grad_v[...,a,b] = d v_a / d x_b
        grad = np.stack([dv_dx, dv_dy, dv_dz], axis=2)
        return grad

    # ------------------------------------------------------------------
    def _update_F(self) -> None:
        cell = self._cell_index(self.position)
        grad_v = self._velocity_gradient_at(self._grid_v, cell)
        # F^{n+1} = (I + dt * grad_v) F^n
        I = np.eye(3, dtype=float)
        self.F = self.F + self.substep * np.einsum("nij,njk->nik", grad_v, self.F)

    # ------------------------------------------------------------------
    def _update_dissolution(self, dt: float) -> None:
        bunny_ids = np.where(self.label == 1)[0]
        water_ids = np.where(self.label == 2)[0]
        scalar_source = np.zeros(self.grid_shape, dtype=float)

        if bunny_ids.size == 0 or water_ids.size == 0:
            self._scalar_source = scalar_source
            return

        bpos = self.position[bunny_ids]
        wpos = self.position[water_ids]
        tree = cKDTree(bpos)
        neighbors = tree.query_ball_point(wpos, r=self.cutoff)

        pair_w: list[int] = []
        pair_b: list[int] = []
        pair_dist: list[float] = []
        for wi, neis in enumerate(neighbors):
            if not neis:
                continue
            n = len(neis)
            pair_w.extend([wi] * n)
            pair_b.extend(neis)
            dists = np.linalg.norm(bpos[neis] - wpos[wi], axis=1)
            pair_dist.extend(dists.tolist())

        if not pair_w:
            self._scalar_source = scalar_source
            return

        pair_w = np.array(pair_w, dtype=int)
        pair_b = np.array(pair_b, dtype=int)
        pair_dist = np.array(pair_dist, dtype=float)

        # dissolve rate per bunny depends on how many water neighbors it sees
        counts = np.bincount(pair_b, minlength=len(bunny_ids))
        factor = np.minimum(1.0, counts / 4.0)
        dphase = self.dissolve_rate * dt * factor
        phase_b = self.phase[bunny_ids]
        dphase = np.minimum(dphase, phase_b)

        if np.any(dphase > 0):
            dm_b = self.mass[bunny_ids] * dphase
            self.phase[bunny_ids] = phase_b - dphase

            # remove solute from the solid bunny and give it to nearby water
            self.solute_mass[bunny_ids] -= dm_b
            weights = 1.0 / (pair_dist + 1e-6)
            sum_w = np.bincount(pair_b, weights=weights, minlength=len(bunny_ids))
            norm_w = weights / (sum_w[pair_b] + 1e-12)
            active_dm = dm_b[pair_b]
            solute_add = active_dm * norm_w
            np.add.at(self.solute_mass, water_ids[pair_w], solute_add)

            # update per-particle concentrations
            self.concentration[water_ids] = (
                self.solute_mass[water_ids] / self.mass[water_ids]
            )
            self.concentration[bunny_ids] = (
                self.solute_mass[bunny_ids] / self.mass[bunny_ids]
            )

            total_dissolved = float(dm_b.sum())
            self.dissolved_mass_total += total_dissolved
            self.cum_mass_source[0] -= total_dissolved
            self.cum_mass_source[1] += total_dissolved

            cells = self._cell_index(bpos)
            np.add.at(scalar_source, (cells[:, 0], cells[:, 1], cells[:, 2]), dm_b / dt)

        self._scalar_source = scalar_source

    # ------------------------------------------------------------------
    def _diffuse_solute(self, dt: float) -> None:
        water_ids = np.where(self.label == 2)[0]
        if water_ids.size == 0:
            return
        pos_w = self.position[water_ids]
        tree = cKDTree(pos_w)
        pairs = tree.query_pairs(self.cutoff, output_type="ndarray")
        if pairs.shape[0] == 0:
            return
        i = water_ids[pairs[:, 0]]
        j = water_ids[pairs[:, 1]]
        si = self.solute_mass[i]
        sj = self.solute_mass[j]
        exchange = self.diffusivity * dt * (sj - si)
        np.add.at(self.solute_mass, i, exchange)
        np.add.at(self.solute_mass, j, -exchange)
        self.concentration[water_ids] = (
            self.solute_mass[water_ids] / self.mass[water_ids]
        )

    # ------------------------------------------------------------------
    def _apply_inflow(self, dt: float) -> None:
        inflow_active = self.inflow_start <= self.time < self.inflow_end
        if inflow_active:
            self.cum_mass_source[1] += self.inflow_mass_rate * dt
            self.inflow_mass_total += self.inflow_mass_rate * dt

        water_ids = np.where(self.label == 2)[0]
        if water_ids.size == 0:
            return

        inflow_min = np.array([-0.92, -0.28, 0.86], dtype=float)
        inflow_max = np.array([0.32, 0.28, 1.55], dtype=float)
        pos_w = self.position[water_ids]
        inside = np.all((pos_w >= inflow_min) & (pos_w <= inflow_max), axis=1)
        if np.any(inside):
            idx = water_ids[inside]
            if inflow_active:
                self.velocity[idx] = np.array([self.inflow_speed, 0.0, 0.0])

    # ------------------------------------------------------------------
    def _apply_floor_domain(self, dt: float) -> None:
        # floor collision
        below = self.position[:, 2] < self.floor_z
        if np.any(below):
            self.position[below, 2] = self.floor_z
            self.velocity[below, 2] = np.maximum(self.velocity[below, 2], 0.0)
            self.velocity[below, :2] *= 0.9

        # recycle water that has left the combined simulation domain
        water_ids = np.where(self.label == 2)[0]
        if water_ids.size == 0:
            return
        inflow_active = self.inflow_start <= self.time < self.inflow_end
        if not inflow_active:
            return
        pos_w = self.position[water_ids]
        out = np.any((pos_w < self.domain_min) | (pos_w > self.domain_max), axis=1)
        if np.any(out):
            idx = water_ids[out]
            inflow_min = np.array([-0.92, -0.28, 0.86], dtype=float)
            inflow_max = np.array([0.32, 0.28, 1.55], dtype=float)
            r = self.rng.random((idx.size, 3))
            self.position[idx] = inflow_min + r * (inflow_max - inflow_min)
            self.velocity[idx] = np.array([self.inflow_speed, 0.0, 0.0])
            self.solute_mass[idx] = 0.0
            self.concentration[idx] = 0.0

        # keep the receiving base layer static
        base_ids = np.where(self.label == 3)[0]
        if base_ids.size:
            self.position[base_ids] = self.initial_position[base_ids]
            self.velocity[base_ids] = 0.0

    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        # forces at the beginning of the substep
        spring = self._apply_spring_forces()
        contact = self._apply_pair_forces()
        total_force = spring + contact

        # semi-implicit Euler integration
        self.velocity += (total_force / self.mass[:, None] + self.gravity_vec) * dt

        # prescribed pipe inflow
        self._apply_inflow(dt)

        # integrate positions
        self.position += self.velocity * dt

        # boundaries and recycling
        self._apply_floor_domain(dt)

        # update diagnostic grid, velocity field, and scalar field
        self._update_grid()

        # update deformation gradient
        self._update_F()

        # dissolution and diffusion of the red solute
        self._update_dissolution(dt)
        self._diffuse_solute(dt)

    # ------------------------------------------------------------------
    def _build_frame(self) -> StateFrame:
        n = len(self.position)
        mass_quantity = np.zeros((n, 3), dtype=float)
        for k in range(3):
            ids = np.where(self.label == k + 1)[0]
            if ids.size:
                mass_quantity[ids, k] = self.mass[ids]

        attributes = {
            "material_label": self.label.copy(),
            "red_solute_concentration": self.concentration.copy(),
            "solid_phase_fraction": self.phase.copy(),
            "density": self.density.copy(),
            "deformation_gradient": self.F.copy(),
            "mass_quantity": mass_quantity,
        }

        fields = {
            "active_mask": self._active_mask.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "interaction_force": self.interaction_force.copy(),
            "scalar_source": self._scalar_source.copy(),
            "transported_scalar": self._grid_c.copy(),
            "velocity_field": self._grid_v.copy(),
        }

        scalars = {"grid_spacing": float(self.h)}

        return StateFrame(
            time=float(self.time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        while self.time < time:
            dt = min(self.substep, time - self.time)
            self._step(dt)
            self.time += dt
        return self._build_frame()

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "dissolved_mass_kg": self.dissolved_mass_total,
            "inflow_mass_kg": self.inflow_mass_total,
            "grid_spacing_m": float(self.h),
        }
