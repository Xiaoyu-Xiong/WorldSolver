"""Quasi-static elastic-rod solver for static coiling on a rigid substrate.

The rod is represented as a discrete inextensible polyline.  Injection adds
material at the controlled injector outlet, the suspended portion buckles under
gravity, and vertices that reach the sticky substrate are frozen in place.
Bending is modelled with explicit linear curvature springs and inextensibility
is enforced with position-based distance constraints.  A tiny deterministic
out-of-plane imperfection is introduced only after the first frame to seed the
consistent winding direction observed in the source experiment.
"""

from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = seed
        self.scene = scene
        cfg = scene.config
        self.cfg = cfg

        rod_cfg = cfg["rod"]
        self.E = float(rod_cfg["young_modulus"])
        self.nu = float(rod_cfg["poisson_ratio"])
        self.rho = float(rod_cfg["density"])
        self.r = float(rod_cfg["radius"])
        self.A = math.pi * self.r * self.r
        self.I = math.pi * self.r**4 / 4.0
        self.EI = self.E * self.I

        self.g = np.array(cfg["gravity"], dtype=float)

        inj = cfg["injector"]
        self.inj_speed = float(inj["injection_speed"])
        self.inj_start = float(inj["start_time"])
        self.inj_end = float(inj["end_time"])
        self.outlet = np.array(inj["outlet_position"], dtype=float)

        self.surface_z = float(cfg["substrate"]["surface_z"])
        self.contact_z = self.surface_z + self.r

        # Frozen substrate half-extents from scene_geometry.yaml
        self.substrate_center = np.array([0.0, 0.0, 0.0], dtype=float)
        self.substrate_half = np.array([0.24, 0.2, 0.008], dtype=float)

        init = scene.initial_state()

        # Keep mutable copies of the initial state.
        self.x = [np.asarray(v, dtype=float).tolist() for v in init.position]
        self.v = [np.asarray(v, dtype=float).tolist() for v in init.velocity]
        self.m = init.mass.copy().astype(float)
        self.n = len(self.x)

        self.fixed = [False] * self.n
        self.stuck = [False] * self.n
        # The injector outlet is kinematically controlled.
        self.fixed[0] = True

        # Mark the initially deposited tip (already touching the substrate).
        for i in range(self.n):
            if self.x[i][2] <= self.contact_z + 1e-9:
                self.stuck[i] = True
                self.fixed[i] = True
                self.x[i][2] = self.contact_z
                self.v[i] = [0.0, 0.0, 0.0]

        # Rest edge lengths from the initial geometry.
        self.l0: list[float] = []
        for i in range(self.n - 1):
            d = np.array(self.x[i + 1]) - np.array(self.x[i])
            self.l0.append(float(np.linalg.norm(d)))

        self.initial_l0_sum = float(sum(self.l0))
        self.segment_target = self.initial_l0_sum / (self.n - 1)
        self.max_vertices = 120

        self.current_time = 0.0
        self.cumulative_mass_added = 0.0
        self.damping = 10.0  # mass-proportional damping [1/s]
        self.pbd_iters = 5
        self.h = 0.002  # internal time step [s]
        self._symmetry_bias_applied = False

        # Frozen reference scales.
        self.ref_L = self.initial_l0_sum if self.initial_l0_sum > 0.0 else 1.0
        total_mass = float(np.sum(self.m))
        self.ref_force = abs(float(self.g[2])) * total_mass
        self.ref_torque = self.EI / self.ref_L if self.ref_L > 0.0 else 1.0
        self.constraint_scale = self.ref_L

        self.diagnostics_dict: dict[str, object] = {"steps": 0}

    def _inside_substrate(self, pos: list[float]) -> bool:
        c = self.substrate_center
        h = self.substrate_half
        return abs(pos[0] - c[0]) <= h[0] and abs(pos[1] - c[1]) <= h[1]

    def _recompute_masses(self) -> None:
        m = [0.0] * self.n
        m[0] = 0.5 * self.l0[0] * self.A * self.rho
        for i in range(1, self.n - 1):
            m[i] = 0.5 * (self.l0[i - 1] + self.l0[i]) * self.A * self.rho
        m[-1] = 0.5 * self.l0[-1] * self.A * self.rho
        self.m = np.asarray(m, dtype=float)

    def _add_vertex_if_needed(self) -> None:
        l_seg = self.segment_target
        if self.n >= self.max_vertices:
            return
        if self.l0[0] <= 1.8 * l_seg:
            return

        p0 = np.array(self.x[0])
        p1 = np.array(self.x[1])
        d = p1 - p0
        len_d = float(np.linalg.norm(d))
        if len_d < 1e-12:
            return

        dir_vec = d / len_d
        new_pos = p0 + l_seg * dir_vec
        new_v = list(self.v[1])

        self.x.insert(1, new_pos.tolist())
        self.v.insert(1, new_v)
        self.fixed.insert(1, False)
        self.stuck.insert(1, False)

        old_l0 = self.l0[0]
        self.l0[0] = l_seg
        self.l0.insert(1, old_l0 - l_seg)
        self.n += 1
        self._recompute_masses()

    def _enforce_pbd(self) -> None:
        for _ in range(self.pbd_iters):
            for i in range(self.n - 1):
                p1 = np.array(self.x[i])
                p2 = np.array(self.x[i + 1])
                dvec = p2 - p1
                d2 = float(np.dot(dvec, dvec))
                if d2 < 1e-18:
                    continue
                d = math.sqrt(d2)
                l0 = self.l0[i]
                error = (d - l0) / d
                w1 = 0.0 if self.fixed[i] else (1.0 / self.m[i])
                w2 = 0.0 if self.fixed[i + 1] else (1.0 / self.m[i + 1])
                W = w1 + w2
                if W < 1e-18:
                    continue
                corr = error / W
                delta = corr * dvec
                if not self.fixed[i]:
                    self.x[i] = (p1 + w1 * delta).tolist()
                if not self.fixed[i + 1]:
                    self.x[i + 1] = (p2 - w2 * delta).tolist()

    def _apply_contact(self) -> None:
        for i in range(self.n):
            if self.fixed[i]:
                continue
            if self.x[i][2] < self.contact_z + 1e-9 and self._inside_substrate(self.x[i]):
                self.x[i][2] = self.contact_z
                self.v[i] = [0.0, 0.0, 0.0]
                self.stuck[i] = True
                self.fixed[i] = True

    def _apply_symmetry_bias(self) -> None:
        if self._symmetry_bias_applied:
            return
        # Tiny deterministic out-of-plane imperfection that seeds a consistent
        # winding direction while leaving t=0 unchanged.
        total = sum(self.l0)
        s = 0.0
        for i in range(1, self.n):
            s += self.l0[i - 1]
            if not self.fixed[i]:
                # Maximum displacement 1e-5 m, much smaller than the rod radius.
                self.x[i][1] += 1e-5 * math.sin(math.pi * s / max(total, 1e-12))
        self._symmetry_bias_applied = True

    def _step(self, h: float) -> None:
        # Inject new material by increasing the rest length of the top segment.
        if self.inj_start <= self.current_time < self.inj_end:
            self.l0[0] += self.inj_speed * h
            self.cumulative_mass_added += self.rho * self.A * self.inj_speed * h
        # Update lumped masses to account for the growing top segment.
        self._recompute_masses()

        # A one-time tiny geometric imperfection to break the planar symmetry.
        self._apply_symmetry_bias()

        # Zero velocity of constrained vertices.
        for i in range(self.n):
            if self.fixed[i]:
                self.v[i] = [0.0, 0.0, 0.0]

        # Accumulate forces on free vertices.
        f = [np.zeros(3, dtype=float) for _ in range(self.n)]

        # Gravity and mass-proportional damping.
        for i in range(self.n):
            if self.fixed[i]:
                continue
            f[i] += self.m[i] * self.g
            f[i] -= self.damping * self.m[i] * np.array(self.v[i])

        # Linear bending springs (curvature springs) on consecutive triples.
        for i in range(1, self.n - 1):
            if self.fixed[i] and self.fixed[i - 1] and self.fixed[i + 1]:
                continue
            xi = np.array(self.x[i])
            xim = np.array(self.x[i - 1])
            xip = np.array(self.x[i + 1])
            D = 2.0 * xi - xim - xip
            lav = 0.5 * (self.l0[i - 1] + self.l0[i])
            k = self.EI / (lav**3)
            f[i] -= 2.0 * k * D
            f[i - 1] += k * D
            f[i + 1] += k * D

        # Explicit velocity/position update.
        for i in range(self.n):
            if self.fixed[i]:
                continue
            v_new = np.array(self.v[i]) + h * f[i] / self.m[i]
            x_new = np.array(self.x[i]) + h * v_new
            self.v[i] = v_new.tolist()
            self.x[i] = x_new.tolist()

        # Sticky substrate contact.
        self._apply_contact()
        # Enforce inextensibility.
        self._enforce_pbd()
        # Contact may be re-established after constraints.
        self._apply_contact()
        # Add newly injected vertices if the top segment is long enough.
        self._add_vertex_if_needed()

        self.current_time += h

    def advance_to(self, time: float) -> StateFrame:
        if time < self.current_time:
            time = self.current_time
        while self.current_time < time - 1e-12:
            h = min(self.h, time - self.current_time)
            self._step(h)
        self.diagnostics_dict["steps"] += 1
        return self._build_frame(float(time))

    def _orientation_from_tangent(self, tangent: np.ndarray) -> np.ndarray:
        t = np.asarray(tangent, dtype=float)
        norm = float(np.linalg.norm(t))
        if norm < 1e-12:
            t = np.array([0.0, 0.0, 1.0])
        else:
            t = t / norm
        z = np.array([0.0, 0.0, 1.0])
        dot = float(np.dot(z, t))
        if dot > 0.999999:
            q = np.array([0.0, 0.0, 0.0, 1.0])
        elif dot < -0.999999:
            q = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = np.cross(z, t)
            axis = axis / np.linalg.norm(axis)
            half = math.acos(dot) * 0.5
            s = math.sin(half)
            c = math.cos(half)
            q = np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])
            q = q / np.linalg.norm(q)
        return q

    def _build_frame(self, time: float) -> StateFrame:
        n = self.n
        pos = np.asarray(self.x, dtype=float).reshape(n, 3)
        vel = np.asarray(self.v, dtype=float).reshape(n, 3)
        mass = np.asarray(self.m, dtype=float).reshape(n)
        body_id = np.full(n, 0, dtype=np.int64)

        orientation = np.zeros((n, 4), dtype=float)
        for i in range(n):
            if i == 0:
                t = pos[1] - pos[0]
            elif i == n - 1:
                t = pos[-1] - pos[-2]
            else:
                t = (pos[i] - pos[i - 1]) + (pos[i + 1] - pos[i])
            orientation[i] = self._orientation_from_tangent(t)
        angular_velocity = np.zeros((n, 3), dtype=float)

        # Arc-length coordinate from rest lengths.
        arc = np.zeros(n, dtype=float)
        for i in range(1, n):
            arc[i] = arc[i - 1] + self.l0[i - 1]

        # Inextensibility constraint residual per edge.
        edge_len = np.zeros(n - 1, dtype=float)
        for i in range(n - 1):
            edge_len[i] = float(np.linalg.norm(pos[i + 1] - pos[i]))
        l0_arr = np.asarray(self.l0[: n - 1], dtype=float)
        constraint_value = (edge_len - l0_arr) / np.maximum(l0_arr, 1e-12)
        constraint_target = np.zeros(n - 1, dtype=float)

        # External-force budget (two domains, one material).
        G, K = 2, 1
        budget_force = np.zeros((G, K, 3), dtype=float)
        budget_couple = np.zeros((G, K, 3), dtype=float)
        budget_pos = np.zeros((G, K, 3), dtype=float)

        # Group 0: gravity on the whole rod.
        total_gravity = np.sum(mass[:, None] * self.g[None, :], axis=0)
        budget_force[0, 0, :] = total_gravity
        budget_pos[0, 0, :] = np.mean(pos, axis=0)

        # Group 1: substrate supporting the deposited/stuck portion.
        stuck_mask = np.array(self.stuck, dtype=bool)
        if np.any(stuck_mask):
            stuck_mass = float(np.sum(mass[stuck_mask]))
            contact_force = -stuck_mass * self.g
            contact_pos = np.mean(pos[stuck_mask], axis=0)
        else:
            contact_force = np.zeros(3, dtype=float)
            contact_pos = np.array([0.0, 0.0, self.contact_z])
        budget_force[1, 0, :] = contact_force
        budget_pos[1, 0, :] = contact_pos

        cumulative = np.array([self.cumulative_mass_added], dtype=float)

        attributes = {
            "mass_quantity": mass.copy(),
            "arc_coordinate": arc,
            "is_injector_controlled": np.array(
                [1.0 if i == 0 else 0.0 for i in range(n)], dtype=float
            ),
            "is_initial_contact": np.array(
                [1.0 if self.stuck[i] else 0.0 for i in range(n)], dtype=float
            ),
            "rod_radius": np.full(n, self.r, dtype=float),
        }

        return StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields={
                "budget_external_force": budget_force,
                "budget_external_couple": budget_couple,
                "budget_force_position": budget_pos,
                "constraint_value": constraint_value,
                "constraint_target": constraint_target,
                "cumulative_mass_source": cumulative,
            },
            scalars={
                "constraint_scale": self.constraint_scale,
                "reference_force": self.ref_force,
                "reference_torque": self.ref_torque,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return self.diagnostics_dict
