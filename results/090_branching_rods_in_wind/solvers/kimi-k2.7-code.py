"""Physically-based solver for wind-driven branching elastic rods.

The model represents the tree as a network of inextensible elastic rods
(vertices connected by edges).  Stretch is enforced with position-based
dynamics, bending is modelled by springs across grand-parent/child pairs,
gravity acts as a body force, and wind is applied as a prescribed spatial
body-force profile with per-branch phase offsets.
"""
from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        # ------------------------------------------------------------------
        # Load scaffold configuration
        # ------------------------------------------------------------------
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.seed = seed
        self.gravity = np.array(self.config["gravity"], dtype=float)

        wind_cfg = self.config["wind"]
        self.wind_dir = np.array(wind_cfg["direction"], dtype=float)
        wind_norm = np.linalg.norm(self.wind_dir)
        if wind_norm > 0.0:
            self.wind_dir = self.wind_dir / wind_norm

        rod_cfg = self.config["rod_network"]
        self.radius_by_level = np.array(rod_cfg["radius_by_level"], dtype=float)
        self.density_by_level = np.array(rod_cfg["linear_density_by_level"], dtype=float)

        junction_cfg = self.config["rigid_junctions"]
        self.rigid_junctions = np.array(junction_cfg["vertex_indices"], dtype=int)
        self.mass_by_level = np.array(junction_cfg["mass_by_level"], dtype=float)

        # ------------------------------------------------------------------
        # Scene initial state
        # ------------------------------------------------------------------
        initial = scene.initial_state()
        self.x = np.asarray(initial.position, dtype=float).copy()
        self.v = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.attributes = {
            k: np.asarray(v).copy() for k, v in initial.attributes.items()
        }
        self.n = self.x.shape[0]

        # ------------------------------------------------------------------
        # Load topology and build connectivity
        # ------------------------------------------------------------------
        topology_path = Path(__file__).parent.parent / rod_cfg["topology_asset"]
        topo = json.loads(topology_path.read_text(encoding="utf-8"))

        self.edges = np.array(topo["edges"], dtype=int)
        self.vertex_level = np.array(topo["vertex_level"], dtype=int)
        self.branch_id = np.array(topo["branch_id"], dtype=int)
        self.lumped_length = np.array(topo["lumped_length"], dtype=float)

        # Radii used for geometry/inertia/drag
        self.radii = self.radius_by_level[self.vertex_level]

        # Add extra rigid-junction mass
        for idx in self.rigid_junctions:
            lvl = self.vertex_level[idx]
            if lvl < len(self.mass_by_level):
                self.mass[idx] += self.mass_by_level[lvl]

        # Build parent/children arrays (BFS from root 0)
        self.children: list[list[int]] = [[] for _ in range(self.n)]
        self.parent = np.full(self.n, -1, dtype=int)
        adj = [[] for _ in range(self.n)]
        for i, j in self.edges:
            adj[i].append(j)
            adj[j].append(i)

        visited = {0}
        q = deque([0])
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    self.parent[nb] = cur
                    self.children[cur].append(nb)
                    q.append(nb)

        # ------------------------------------------------------------------
        # Bending springs: one per interior vertex connecting its
        # grand-parent to each child.  This penalises changes in the angle at
        # the interior vertex while remaining in the inextensible rod model.
        # ------------------------------------------------------------------
        self.bend_pairs: np.ndarray = np.empty((0, 2), dtype=int)
        self.bend_rest_len: np.ndarray = np.empty(0, dtype=float)
        self.bend_k: np.ndarray = np.empty(0, dtype=float)

        E_bend = 2.0e7  # bending modulus (Pa)
        L_ref = 0.5
        bend_pairs_list: list[tuple[int, int]] = []
        bend_rest_list: list[float] = []
        bend_k_list: list[float] = []

        for i in range(1, self.n):
            p = self.parent[i]
            if p < 0:
                continue
            for c in self.children[i]:
                L0 = np.linalg.norm(self.x[c] - self.x[p])
                # stiffness associated with the child branch level
                lvl = self.vertex_level[c]
                r = self.radius_by_level[lvl]
                I = math.pi * r ** 4 / 4.0
                k = E_bend * I / (L_ref ** 3)
                bend_pairs_list.append((p, c))
                bend_rest_list.append(float(L0))
                bend_k_list.append(float(k))

        if bend_pairs_list:
            self.bend_pairs = np.array(bend_pairs_list, dtype=int)
            self.bend_rest_len = np.array(bend_rest_list, dtype=float)
            self.bend_k = np.array(bend_k_list, dtype=float)

        # ------------------------------------------------------------------
        # Edge rest lengths and mass inverse (root is fixed)
        # ------------------------------------------------------------------
        dx = self.x[self.edges[:, 1]] - self.x[self.edges[:, 0]]
        self.rest_len = np.linalg.norm(dx, axis=1)
        self.inv_mass = 1.0 / self.mass
        self.inv_mass[0] = 0.0

        # ------------------------------------------------------------------
        # Solver parameters
        # ------------------------------------------------------------------
        self.physics_dt = 0.0005
        self.constraint_iters = 10
        self.damping = 5.0  # Rayleigh damping coefficient (1/s)

        self.rho_air = 1.225
        self.cd = 1.2
        self.wind_scale = 0.25
        self.gust_freq = float(wind_cfg.get("gust_frequency_hz", 0.55))
        self.branch_phase_offsets = np.array(
            wind_cfg.get("branch_phase_offsets", [0.0]), dtype=float
        )
        self.vertex_phase = self.branch_phase_offsets[
            self.branch_id % max(1, len(self.branch_phase_offsets))
        ]

        kf = wind_cfg.get("keyframes", [])
        self.wind_times = np.array([k["time"] for k in kf], dtype=float)
        self.wind_speeds = np.array([k["speed"] for k in kf], dtype=float)

        self.max_z = float(np.max(self.x[:, 2]))
        areas = np.pi * self.radii ** 2
        self.max_area = float(np.max(areas))

        self.time = 0.0
        self.cum_impulse = np.zeros(3)
        self.cum_ang_impulse = np.zeros(3)
        self.x0_root = self.x[0].copy()

        # Reference scales for the trajectory
        total_mass = max(float(np.sum(self.mass)), 1e-6)
        self.ref_momentum = total_mass * 5.0
        self.ref_ang_momentum = self.ref_momentum * 2.0

    # ------------------------------------------------------------------
    # Wind prescription
    # ------------------------------------------------------------------
    def _base_wind_speed(self, t: float) -> float:
        times = self.wind_times
        if t <= times[0]:
            return float(self.wind_speeds[0])
        if t >= times[-1]:
            return float(self.wind_speeds[-1])
        idx = int(np.searchsorted(times, t, side="right")) - 1
        idx = max(0, min(idx, len(times) - 2))
        t0, t1 = times[idx], times[idx + 1]
        s = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        return float(self.wind_speeds[idx] + s * (self.wind_speeds[idx + 1] - self.wind_speeds[idx]))

    def _wind_force(self, x: np.ndarray, t: float) -> np.ndarray:
        base = self._base_wind_speed(t)
        if base <= 0.0:
            return np.zeros_like(x)

        mod = 1.0 + 0.15 * np.sin(2.0 * math.pi * self.gust_freq * t + self.vertex_phase)
        mod = np.clip(mod, 0.0, None)
        speeds = base * mod

        # Height- and branch-area-weighted spatial profile
        height_factor = (x[:, 2] + 0.5) / (self.max_z + 0.5)
        height_factor = np.clip(height_factor, 0.05, 1.0)
        area = np.pi * self.radii ** 2
        area_factor = area / self.max_area if self.max_area > 0.0 else 1.0
        profile = height_factor * area_factor

        # Drag using cross-sectional area as the exposed area
        dynamic_pressure = 0.5 * self.rho_air * self.cd * speeds ** 2
        force_mag = self.wind_scale * dynamic_pressure * profile
        return force_mag[:, None] * self.wind_dir

    # ------------------------------------------------------------------
    # Internal forces
    # ------------------------------------------------------------------
    def _bending_force(self, x: np.ndarray) -> np.ndarray:
        f = np.zeros_like(x)
        if self.bend_pairs.shape[0] == 0:
            return f
        i = self.bend_pairs[:, 0]
        j = self.bend_pairs[:, 1]
        dx = x[j] - x[i]
        len_ij = np.linalg.norm(dx, axis=1)
        rest = self.bend_rest_len
        diff = len_ij - rest
        unit = dx / np.maximum(len_ij[:, None], 1e-12)
        force = self.bend_k[:, None] * diff[:, None] * unit
        np.add.at(f, i, -force)
        np.add.at(f, j, force)
        return f

    def _gravity_force(self) -> np.ndarray:
        return self.mass[:, None] * self.gravity

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    def _substep(self, h: float) -> None:
        x = self.x
        v = self.v

        f_grav = self._gravity_force()
        f_wind = self._wind_force(x, self.time + h)
        f_bend = self._bending_force(x)
        f_damp = -self.damping * self.mass[:, None] * v
        f_total = f_grav + f_wind + f_bend + f_damp

        # Semi-implicit Euler velocity update
        v_new = v + h * f_total / self.mass[:, None]
        v_new[0] = 0.0

        # Position update
        x_old = x.copy()
        x_new = x + h * v_new

        # PBD stretch constraints (Gauss-Seidel)
        for _ in range(self.constraint_iters):
            for e in range(self.edges.shape[0]):
                i = self.edges[e, 0]
                j = self.edges[e, 1]
                dx = x_new[j] - x_new[i]
                len_ij = np.linalg.norm(dx)
                if len_ij == 0.0:
                    continue
                w1 = self.inv_mass[i]
                w2 = self.inv_mass[j]
                # C = |dx| - L0 ; lambda = C / |dx| / (w1+w2)
                scale = (len_ij - self.rest_len[e]) / (len_ij * (w1 + w2))
                corr = scale * dx
                x_new[i] = x_new[i] + w1 * corr
                x_new[j] = x_new[j] - w2 * corr

        # Recompute velocity from PBD projection
        v_new = (x_new - x_old) / h
        v_new[0] = 0.0
        x_new[0] = self.x0_root

        # Update cumulative momentum budgets using external forces
        f_ext = f_grav + f_wind
        self.cum_impulse += np.sum(f_ext, axis=0) * h
        self.cum_ang_impulse += np.sum(np.cross(x_new, f_ext), axis=0) * h

        self.x = x_new
        self.v = v_new
        self.time += h

    # ------------------------------------------------------------------
    # Frame helpers
    # ------------------------------------------------------------------
    def _compute_orientation(self) -> np.ndarray:
        orient = np.zeros((self.n, 4), dtype=float)
        orient[:, 3] = 1.0

        tangents = np.zeros((self.n, 3), dtype=float)
        counts = np.zeros(self.n, dtype=float)

        dx = self.x[self.edges[:, 1]] - self.x[self.edges[:, 0]]
        np.add.at(tangents, self.edges[:, 0], dx)
        np.add.at(tangents, self.edges[:, 1], dx)
        counts += np.bincount(self.edges[:, 0], minlength=self.n)
        counts += np.bincount(self.edges[:, 1], minlength=self.n)

        # Root follows the first branch direction
        if self.children[0]:
            tangents[0] = self.x[self.children[0][0]] - self.x[0]
            counts[0] = 1.0
        else:
            tangents[0] = np.array([0.0, 0.0, 1.0])
            counts[0] = 1.0

        counts[counts == 0.0] = 1.0
        tangents = tangents / counts[:, None]
        norms = np.linalg.norm(tangents, axis=1)
        norms[norms == 0.0] = 1.0
        tangents = tangents / norms[:, None]

        z_axis = np.array([0.0, 0.0, 1.0])
        for i in range(self.n):
            t = tangents[i]
            dot = np.clip(t[2], -1.0, 1.0)
            if dot > 0.99999:
                orient[i] = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                axis = np.cross(z_axis, t)
                axis = axis / np.linalg.norm(axis)
                angle = math.acos(dot)
                s = math.sin(angle * 0.5)
                c = math.cos(angle * 0.5)
                orient[i, :3] = axis * s
                orient[i, 3] = c

        # Normalise
        qnorm = np.linalg.norm(orient, axis=1, keepdims=True)
        qnorm[qnorm == 0.0] = 1.0
        orient = orient / qnorm
        return orient

    def _com(self) -> np.ndarray:
        total = np.sum(self.mass)
        if total <= 0.0:
            return np.zeros(3)
        return np.sum(self.mass[:, None] * self.x, axis=0) / total

    def _build_frame(self, t: float) -> StateFrame:
        orientation = self._compute_orientation()
        angular_velocity = np.zeros((self.n, 3), dtype=float)

        # Per-sample inertia tensor (sphere approximation)
        inertia = np.zeros((self.n, 3, 3), dtype=float)
        diag = 0.4 * self.mass * self.radii ** 2
        inertia[:, 0, 0] = diag
        inertia[:, 1, 1] = diag
        inertia[:, 2, 2] = diag

        f_grav = self._gravity_force()
        f_wind = self._wind_force(self.x, t)
        f_ext_total = f_grav + f_wind

        total_force = np.sum(f_ext_total, axis=0)
        total_torque = np.sum(np.cross(self.x, f_ext_total), axis=0)
        com = self._com()

        domain_weights = (self.mass / np.sum(self.mass)).reshape(1, -1)
        fields = {
            "budget_cumulative_external_impulse": self.cum_impulse.reshape(1, 3).copy(),
            "budget_cumulative_external_angular_impulse": self.cum_ang_impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3), dtype=float),
            "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3), dtype=float),
            "budget_domain_weights": domain_weights,
            "budget_external_force": total_force.reshape(1, 1, 3),
            "budget_external_couple": np.zeros((1, 1, 3), dtype=float),
            "budget_force_position": com.reshape(1, 1, 3),
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }

        attributes = dict(self.attributes)
        attributes["inertia_tensor"] = inertia
        attributes["mass_quantity"] = self.mass.copy()

        return StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars={
                "reference_momentum": float(self.ref_momentum),
                "reference_angular_momentum": float(self.ref_ang_momentum),
            },
        )

    def advance_to(self, time: float) -> StateFrame:
        if time < self.time:
            return self._build_frame(time)
        dt = time - self.time
        if dt > 0.0:
            n_steps = int(round(dt / self.physics_dt))
            if n_steps < 1:
                n_steps = 1
            h = dt / n_steps
            for _ in range(n_steps):
                self._substep(h)
        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "physics_dt": self.physics_dt,
            "constraint_iterations": self.constraint_iters,
            "current_time": self.time,
        }
