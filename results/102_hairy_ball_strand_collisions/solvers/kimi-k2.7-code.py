"""Strand collision solver for the rotating hairy-ball scene.

This solver implements a Position-Based Dynamics (PBD) time integrator for the
elastic hair strands.  The sphere is kinematic and follows the prescribed
axis-angle keyframes.  Roots are attached to the sphere surface and move with it.
Free vertices are subject to gravity, stretch/bending constraints, and hard
non-penetration constraints against the sphere and against other strand vertices.
A small velocity damping models the dissipative effects that allow strand groups
to settle and separate again.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from worldsolver.contracts import StateFrame


class _SpatialHash:
    """Simple 3-D uniform grid for contact detection."""

    __slots__ = ("cell_size", "cells")

    def __init__(self, cell_size: float):
        self.cell_size = cell_size
        self.cells: dict[tuple[int, int, int], list[int]] = {}

    def build(self, points: np.ndarray) -> None:
        self.cells.clear()
        inv = 1.0 / self.cell_size
        for i in range(points.shape[0]):
            c = (int(math.floor(points[i, 0] * inv)),
                 int(math.floor(points[i, 1] * inv)),
                 int(math.floor(points[i, 2] * inv)))
            if c in self.cells:
                self.cells[c].append(i)
            else:
                self.cells[c] = [i]

    def pairs(self, points: np.ndarray) -> list[tuple[int, int]]:
        """Return unique (i, j) pairs with i < j in adjacent cells."""
        pairs: list[tuple[int, int]] = []
        inv = 1.0 / self.cell_size
        for c, idxs in self.cells.items():
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        n = (c[0] + di, c[1] + dj, c[2] + dk)
                        if n not in self.cells:
                            continue
                        others = self.cells[n]
                        for i in idxs:
                            for j in others:
                                if i < j:
                                    pairs.append((i, j))
        return pairs


class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed

        # Physical parameters.
        p = self.config["physical_parameters"]
        self.ball_r = float(p["sphere_radius"])
        self.strand_count = int(p["strand_count"])
        self.segments = int(p["segments_per_strand"])
        self.strand_length = float(p["strand_length"])
        self.strand_r = float(p["strand_radius"])
        self.density = float(p["density"])
        self.E = float(p["stretch_young_modulus"])
        self.bend_E = float(p["bending_young_modulus"])
        self.poisson = float(p["poisson_ratio"])
        self.gravity = np.array(self.config["gravity"], dtype=float)
        self.mu = float(self.config["contact"]["static_friction"])

        # Time stepping.
        self.dt = 5.0e-4  # PBD sub-step, stable for the stiff strands.
        sample_dt = float(self.config["sample_dt"])
        self.substeps = max(1, int(round(sample_dt / self.dt)))
        self.dt = sample_dt / self.substeps  # exact divisor
        self.t = 0.0

        # Initial state.
        frame0 = scene.initial_state()
        self.pos = np.asarray(frame0.position, dtype=float).copy()
        self.vel = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.root_mask = np.asarray(frame0.attributes["root_mask"], dtype=bool)
        self.strand_id = np.asarray(frame0.attributes["strand_id"], dtype=int)
        self.n = self.pos.shape[0]

        # Inverse mass for PBD (fixed roots have zero inverse mass).
        self.inv_mass = np.zeros(self.n, dtype=float)
        self.inv_mass[~self.root_mask] = 1.0 / self.mass[~self.root_mask]

        # Rest geometry: per-vertex initial positions are used for root attachment.
        self.root0 = self.pos[self.root_mask].copy()
        self.root_index = np.nonzero(self.root_mask)[0]

        # Ensure the first segment length in rest state is not exactly zero, so
        # that distance projection is well-defined even for degenerate strands.
        self.rest_len = np.zeros(0, dtype=float)  # populated below

        # Build edge list (per strand segments).
        E = self.strand_count * self.segments
        self.edges = np.zeros((E, 2), dtype=int)
        for s in range(self.strand_count):
            base = s * (self.segments + 1)
            for k in range(self.segments):
                self.edges[s * self.segments + k] = [base + k, base + k + 1]

        # Cross edges for bending (i-1 to i+1 within each strand).
        C = self.strand_count * (self.segments - 1)
        self.cross_edges = np.zeros((C, 2), dtype=int)
        for s in range(self.strand_count):
            base = s * (self.segments + 1)
            for k in range(1, self.segments):
                self.cross_edges[s * (self.segments - 1) + (k - 1)] = [base + k - 1, base + k + 1]

        # Rest lengths.
        p0 = self.pos[self.edges[:, 0]]
        p1 = self.pos[self.edges[:, 1]]
        self.rest_len = np.linalg.norm(p1 - p0, axis=1)
        self.rest_len = np.maximum(self.rest_len, 1e-8)
        c0 = self.pos[self.cross_edges[:, 0]]
        c1 = self.pos[self.cross_edges[:, 1]]
        self.rest_cross = np.linalg.norm(c1 - c0, axis=1)
        self.rest_cross = np.maximum(self.rest_cross, 1e-8)

        # PBD iteration counts.
        self.iters_stretch = 10
        self.iters_bend = 5
        self.iters_contact = 2

        # Velocity damping (per sub-step).
        self.damping = 0.01

        # Sphere rotation interpolation.
        keyframes = self.config["prescribed_sphere_rotation"]["keyframes_axis_angle_degrees"]
        self.rot_times = np.array([k["time"] for k in keyframes], dtype=float)
        rots = []
        for k in keyframes:
            axis = np.array(k["axis"], dtype=float)
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                axis = np.array([0.0, 0.0, 1.0])
            else:
                axis = axis / norm
            angle = math.radians(k["angle"])
            rots.append(Rotation.from_rotvec(angle * axis))
        rots = Rotation.from_quat([r.as_quat() for r in rots])
        self.rot_interp = Slerp(self.rot_times, rots)
        self.omega_cache: dict[float, np.ndarray] = {}

        # Budget accumulators.
        self.cum_external_impulse = np.zeros((1, 3), dtype=float)
        self.cum_inflow = np.zeros((1, 3), dtype=float)
        self.cum_outflow = np.zeros((1, 3), dtype=float)
        self.cum_mass_source = np.zeros(1, dtype=float)

        # Contact helpers.
        # Cell size large enough to make the hash cheap but small enough to avoid
        # too many spurious pairs.  Contacts are resolved when strands are within
        # a few strand radii.
        self.contact_cell = max(2.0 * self.strand_r, 1.0e-4)
        self.hash = _SpatialHash(self.contact_cell)
        self.contact_margin = 2.0 * self.strand_r

        # Reference scales.
        total_m = float(self.mass.sum())
        self.ref_force = max(1.0e-6, total_m * abs(self.gravity[2]))
        self.ref_momentum = max(1.0e-6, total_m * 1.0)
        self.ref_velocity = 1.0
        self.stick_tol = 1.0e-3

    def _rotation_matrix(self, t: float) -> np.ndarray:
        t = float(np.clip(t, self.rot_times[0], self.rot_times[-1]))
        return self.rot_interp(t).as_matrix()

    def _angular_velocity(self, t: float) -> np.ndarray:
        # Finite-difference derivative of the rotation quaternion.
        if t in self.omega_cache:
            return self.omega_cache[t]
        eps = 1.0e-7
        tp = min(self.rot_times[-1], max(self.rot_times[0], t + eps))
        tm = min(self.rot_times[-1], max(self.rot_times[0], t - eps))
        Rp = self._rotation_matrix(tp)
        Rm = self._rotation_matrix(tm)
        Rdiff = Rp @ Rm.T
        rotvec = Rotation.from_matrix(Rdiff).as_rotvec()
        omega = rotvec / (2.0 * eps)
        self.omega_cache[t] = omega
        return omega

    # ---- PBD constraint projection ----

    def _project_distance(self, p: np.ndarray, edges: np.ndarray, rest: np.ndarray) -> np.ndarray:
        """Project edge lengths back to rest lengths."""
        i = edges[:, 0]
        j = edges[:, 1]
        pi = p[i]
        pj = p[j]
        e = pj - pi
        l = np.linalg.norm(e, axis=1)
        # Avoid division by tiny lengths by clamping.
        l_safe = np.where(l < 1e-12, 1.0, l)
        u = e / l_safe[:, None]
        C = l - rest
        w0 = self.inv_mass[i]
        w1 = self.inv_mass[j]
        W = w0 + w1
        # Only project pairs with positive total compliance.
        active = W > 0.0
        s = np.zeros_like(C)
        s[active] = C[active] / W[active]
        # Accumulate corrections.
        corr = np.zeros_like(p)
        # dp_i = -w_i * s * u
        # dp_j = +w_j * s * u
        factor_i = -(w0 * s)[:, None] * u
        factor_j = (w1 * s)[:, None] * u
        np.add.at(corr, (i,), factor_i)
        np.add.at(corr, (j,), factor_j)
        return p + corr

    def _project_sphere_collision(self, p: np.ndarray) -> np.ndarray:
        free = ~self.root_mask
        d = np.linalg.norm(p[free], axis=1)
        target = self.ball_r + self.strand_r
        penetrating = d < target
        if not np.any(penetrating):
            return p
        p_new = p.copy()
        pts = p[free]
        n = pts / d[:, None]
        # Move free vertices to the sphere surface plus strand radius.
        p_new[free] = n * target
        return p_new

    def _project_strand_strand(self, p: np.ndarray) -> np.ndarray:
        self.hash.build(p)
        pairs = self.hash.pairs(p)
        if not pairs:
            return p
        p_new = p.copy()
        target = 2.0 * self.strand_r
        for i, j in pairs:
            if self.strand_id[i] == self.strand_id[j]:
                continue
            dx = p_new[i] - p_new[j]
            dist_sq = float(dx @ dx)
            if dist_sq >= target * target or dist_sq < 1e-18:
                continue
            dist = math.sqrt(dist_sq)
            n = dx / dist
            penetration = target - dist
            wi = self.inv_mass[i]
            wj = self.inv_mass[j]
            W = wi + wj
            if W <= 0.0:
                continue
            # PBD projection for contact: move apart along normal.
            p_new[i] += (wi / W) * penetration * n
            p_new[j] -= (wj / W) * penetration * n
        return p_new

    # ---- Sub-step and public API ----

    def _substep(self, t_next: float) -> None:
        dt = self.dt
        R = self._rotation_matrix(t_next)
        root_target = (R @ self.root0.T).T
        omega = self._angular_velocity(t_next)
        root_vel = np.cross(omega, root_target)

        # Gravity on free vertices.
        free = ~self.root_mask
        self.vel[free] += dt * self.gravity

        # Roots move with the sphere.
        self.pos[self.root_mask] = root_target
        self.vel[self.root_mask] = root_vel

        # Predict.
        p_pred = self.pos + dt * self.vel
        p_pred[self.root_mask] = root_target

        # Project constraints.
        for _ in range(self.iters_stretch):
            p_pred = self._project_distance(p_pred, self.edges, self.rest_len)
            p_pred[self.root_mask] = root_target
        for _ in range(self.iters_bend):
            p_pred = self._project_distance(p_pred, self.cross_edges, self.rest_cross)
            p_pred[self.root_mask] = root_target
        for _ in range(self.iters_contact):
            p_pred = self._project_sphere_collision(p_pred)
            p_pred[self.root_mask] = root_target
            p_pred = self._project_strand_strand(p_pred)
            p_pred[self.root_mask] = root_target

        # Update velocity and apply damping.
        self.vel = (p_pred - self.pos) / dt
        self.vel[self.root_mask] = root_vel
        self.vel *= (1.0 - self.damping)
        self.pos = p_pred

        # Budget: gravity impulse over the domain.
        self.cum_external_impulse[0] += self.mass.sum() * self.gravity * dt

    def advance_to(self, time: float) -> StateFrame:
        while self.t + 1e-12 < time:
            self._substep(self.t + self.dt)
            self.t = round(self.t + self.dt, 10)
        return self._make_frame(time)

    # ---- Trajectory fields ----

    def _contact_forces(self) -> dict[str, np.ndarray]:
        """Build friction/contact fields at the current sample time."""
        pos = self.pos
        vel = self.vel
        omega = self._angular_velocity(self.t)
        mu = self.mu

        normals: list[np.ndarray] = []
        rel_v: list[np.ndarray] = []
        normal_forces: list[float] = []
        tang_forces: list[np.ndarray] = []
        active: list[int] = []
        mu_list: list[float] = []

        # Sphere contacts.
        d = np.linalg.norm(pos, axis=1)
        target_s = self.ball_r + self.strand_r
        k = self.E * (np.pi * self.strand_r ** 2) / self.strand_length  # characteristic contact stiffness
        for i in range(self.n):
            if d[i] >= target_s:
                continue
            n = pos[i] / d[i]
            pen = target_s - d[i]
            Fn = k * pen
            # Velocity of sphere surface at contact point.
            surf_point = self.ball_r * n
            v_surf = np.cross(omega, surf_point)
            v_rel = vel[i] - v_surf
            v_n = float(v_rel @ n)
            v_t = v_rel - v_n * n
            v_t_mag = float(np.linalg.norm(v_t))
            if v_t_mag > self.stick_tol:
                f_t = -mu * Fn * (v_t / v_t_mag)
            else:
                f_t = np.zeros(3)
            normals.append(n)
            rel_v.append(v_rel)
            normal_forces.append(Fn)
            tang_forces.append(f_t)
            active.append(1)
            mu_list.append(mu)

        # Strand-strand contacts.
        self.hash.build(pos)
        pairs = self.hash.pairs(pos)
        target_ss = 2.0 * self.strand_r
        for i, j in pairs:
            if self.strand_id[i] == self.strand_id[j]:
                continue
            dx = pos[i] - pos[j]
            dist = float(np.linalg.norm(dx))
            if dist >= target_ss or dist < 1e-12:
                continue
            n = dx / dist
            pen = target_ss - dist
            Fn = k * pen
            v_rel = vel[i] - vel[j]
            v_n = float(v_rel @ n)
            v_t = v_rel - v_n * n
            v_t_mag = float(np.linalg.norm(v_t))
            if v_t_mag > self.stick_tol:
                f_t = -mu * Fn * (v_t / v_t_mag)
            else:
                f_t = np.zeros(3)
            normals.append(n)
            rel_v.append(v_rel)
            normal_forces.append(Fn)
            tang_forces.append(f_t)
            active.append(1)
            mu_list.append(mu)

        if not active:
            empty = np.zeros((0,), dtype=float)
            empty3 = np.zeros((0, 3), dtype=float)
            return {
                "friction_active_mask": empty.copy(),
                "friction_dynamic_coefficient": empty.copy(),
                "friction_normal": empty3.copy(),
                "friction_normal_force": empty.copy(),
                "friction_relative_velocity": empty3.copy(),
                "friction_static_coefficient": empty.copy(),
                "friction_tangential_force": empty3.copy(),
            }

        return {
            "friction_active_mask": np.array(active, dtype=float),
            "friction_dynamic_coefficient": np.array(mu_list, dtype=float),
            "friction_normal": np.vstack(normals),
            "friction_normal_force": np.array(normal_forces, dtype=float),
            "friction_relative_velocity": np.vstack(rel_v),
            "friction_static_coefficient": np.array(mu_list, dtype=float),
            "friction_tangential_force": np.vstack(tang_forces),
        }

    def _make_frame(self, time: float) -> StateFrame:
        total_m = float(self.mass.sum())
        weights = np.zeros((1, self.n), dtype=float)
        if total_m > 0.0:
            weights[0] = self.mass / total_m

        budget_external = np.zeros((1, 1, 3), dtype=float)
        budget_external[0, 0] = total_m * self.gravity

        contact = self._contact_forces()

        # Attributes from the scene's initial frame for consistency.
        init = self.scene.initial_state()
        attributes = dict(init.attributes)
        attributes["mass_quantity"] = self.mass.copy()

        fields = dict(contact)
        fields["budget_cumulative_external_impulse"] = self.cum_external_impulse.copy()
        fields["budget_cumulative_inflow_momentum"] = self.cum_inflow.copy()
        fields["budget_cumulative_outflow_momentum"] = self.cum_outflow.copy()
        fields["budget_domain_weights"] = weights
        fields["budget_external_force"] = budget_external
        fields["cumulative_mass_source"] = self.cum_mass_source.copy()

        scalars = {
            "strand_radius": self.strand_r,
            "static_friction": self.mu,
            "kinetic_friction": self.mu,
            "friction_stick_speed_tolerance": self.stick_tol,
            "reference_force": self.ref_force,
            "reference_momentum": self.ref_momentum,
            "reference_velocity": self.ref_velocity,
        }

        frame = StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": self.dt,
            "substeps_per_sample": self.substeps,
            "n_vertices": self.n,
            "total_mass": float(self.mass.sum()),
        }
