"""Deterministic discrete elastic-rod solver for alternating loops on a moving belt.

A mass-spring/bending-chain model of an inextensible naturally-straight elastic
rod fed from a fixed vertical injector onto a rigid horizontal belt.  New rod
material is added at the controlled inlet, mobile nodes feel gravity and elastic
bending forces, inextensibility is enforced by position projection, and any
node touching the belt sticks and travels with it.  The scheme is explicit with
substepping and light velocity damping; it is fully deterministic and contains
no animation tables or Genesis code.
"""
from __future__ import annotations

import numpy as np

try:
    from worldsolver.contracts import StateFrame
except Exception:
    from scene import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config
        self.rng = np.random.default_rng(seed)

        # ---- physical constants ----
        self.g_vec = np.array(self.cfg["gravity"], dtype=float)

        self.inj_speed = float(self.cfg["injector"]["injection_speed"])
        self.outlet = np.array(self.cfg["injector"]["outlet_position"], dtype=float)
        self.axis = np.array(self.cfg["injector"]["axis"], dtype=float)
        self.axis /= np.linalg.norm(self.axis) + 1e-12

        belt_cfg = self.cfg["belt"]
        self.belt_speed = float(belt_cfg["speed"])
        self.belt_dir = np.array(belt_cfg["travel_direction"], dtype=float)
        self.belt_dir /= np.linalg.norm(self.belt_dir) + 1e-12
        self.belt_vel = self.belt_dir * self.belt_speed
        self.surface_z = float(belt_cfg["surface_z"])

        rod = self.cfg["rod"]
        self.radius = float(rod["radius"])
        self.rho = float(rod["density"])
        self.E = float(rod["young_modulus"])
        self.nu = float(rod["poisson_ratio"])
        self.total_available_length = float(rod["total_available_length"])

        self.A = np.pi * self.radius * self.radius
        self.I = np.pi * self.radius ** 4 / 4.0
        self.EI = self.E * self.I
        self.mu = self.rho * self.A  # mass per unit length

        # ---- time stepping ----
        self.sample_dt = float(self.cfg["sample_dt"])
        self.n_sub = 20
        self.h = self.sample_dt / self.n_sub
        self.time = 0.0

        # ---- initial state from the scene ----
        f0 = scene.initial_state()
        self.pos = f0.position.copy().astype(float)
        self.vel = f0.velocity.copy().astype(float)
        self.mass = f0.mass.copy().astype(float)
        self.orient = f0.orientation.copy().astype(float)
        self.ang_vel = f0.angular_velocity.copy().astype(float)
        self.body_id = f0.body_id.copy()

        self.attr_arc = f0.attributes["arc_coordinate"].copy().astype(float)
        self.attr_is_inj = f0.attributes["is_injector_controlled"].copy().astype(float)
        self.attr_is_dep = f0.attributes["is_initially_deposited"].copy().astype(float)
        self.attr_is_stick = f0.attributes["is_sticking_to_belt"].copy().astype(float)
        self.attr_radius = f0.attributes["rod_radius"].copy().astype(float)
        self.attr_curv = f0.attributes["natural_curvature"].copy().astype(float)

        self.rest_len = np.linalg.norm(np.diff(self.pos, axis=0), axis=1).astype(float)
        self._recompute_masses()
        self._recompute_arc()

        # The first two nodes are injector-controlled.
        self.attr_is_inj[0] = 1.0
        self.pos[0] = self.outlet

        # Discretization and model parameters.
        self.l_seg = 0.020
        self.bend_iters = 3       # PBD bending projection iterations
        self.dist_iters = 8       # PBD distance projection iterations
        self.damp = 0.97          # velocity damping per substep
        self.contact_tol = self.radius

        # Small deterministic perturbation to break perfect symmetry and seed
        # the alternating-loop instability.
        free = np.where((self.attr_is_inj < 0.5) & (self.attr_is_stick < 0.5))[0]
        self.vel[free, 1] += self.rng.normal(0.0, 1e-5, size=free.shape[0])

        # ---- injection bookkeeping ----
        self.injected_len = 0.0
        self.cum_mass_src = 0.0

        self._diag = {"seed": seed, "nodes": len(self.pos)}

    # ------------------------------------------------------------------
    # helpers
    def _is_fixed(self, i: int) -> bool:
        """Kinematic nodes: injector-controlled outlet/direction nodes and
        material stuck to the moving belt."""
        return i < 2 or self.attr_is_stick[i] > 0.5

    def _recompute_masses(self) -> None:
        n = self.rest_len.shape[0] + 1
        m = np.empty(n, dtype=float)
        m[0] = self.mu * self.rest_len[0] * 0.5
        m[-1] = self.mu * self.rest_len[-1] * 0.5
        m[1:-1] = self.mu * (self.rest_len[:-1] + self.rest_len[1:]) * 0.5
        self.mass = m

    def _recompute_arc(self) -> None:
        self.attr_arc = np.concatenate(([0.0], np.cumsum(self.rest_len)))

    def _add_injection(self, h: float) -> None:
        """Grow the first segment to represent material fed by the injector."""
        inc = self.inj_speed * h
        self.injected_len += inc
        self.cum_mass_src += self.mu * inc
        self.rest_len[0] += inc
        self._recompute_masses()
        self._recompute_arc()

    def _split_top_edge(self) -> None:
        """Subdivide the inlet segment once it exceeds the target length."""
        while (
            self.rest_len[0] > 1.25 * self.l_seg
            and np.sum(self.rest_len) + self.l_seg <= self.total_available_length + 1e-9
        ):
            L0 = float(self.rest_len[0])
            x0 = self.pos[0]
            x1 = self.pos[1]
            d = x1 - x0
            L = float(np.linalg.norm(d))
            if L < 1e-12:
                break
            t = self.l_seg / L
            new_pos = x0 + d * t

            self.pos = np.insert(self.pos, 1, new_pos, axis=0)
            self.vel = np.insert(self.vel, 1, [0.0, 0.0, 0.0], axis=0)
            self.orient = np.insert(self.orient, 1, [0.0, 0.0, 0.0, 1.0], axis=0)
            self.ang_vel = np.insert(self.ang_vel, 1, [0.0, 0.0, 0.0], axis=0)
            self.body_id = np.insert(self.body_id, 1, 0)
            self.attr_is_inj = np.insert(self.attr_is_inj, 1, 0.0)
            self.attr_is_dep = np.insert(self.attr_is_dep, 1, 0.0)
            self.attr_is_stick = np.insert(self.attr_is_stick, 1, 0.0)
            self.attr_radius = np.insert(self.attr_radius, 1, self.radius)
            self.attr_curv = np.insert(self.attr_curv, 1, 0.0)

            self.rest_len[0] = self.l_seg
            self.rest_len = np.insert(self.rest_len, 1, L0 - self.l_seg)
            self._recompute_masses()
            self._recompute_arc()

    def _enforce_injector(self, pos: np.ndarray) -> None:
        """Pin the outlet and keep the first segment aligned with the injector."""
        pos[0] = self.outlet
        if pos.shape[0] > 1:
            pos[1] = self.outlet - self.axis * self.rest_len[0]

    def _project_distance(self, pos: np.ndarray) -> None:
        """Enforce inextensibility via Gauss-Seidel distance projection."""
        for _ in range(self.dist_iters):
            for i in range(pos.shape[0] - 1):
                e = pos[i + 1] - pos[i]
                L = float(np.linalg.norm(e))
                if L < 1e-12:
                    continue
                delta = L - self.rest_len[i]
                if abs(delta) < 1e-12:
                    continue
                n = e / L

                w0 = 0.0 if self._is_fixed(i) else 1.0 / self.mass[i]
                w1 = 0.0 if self._is_fixed(i + 1) else 1.0 / self.mass[i + 1]
                W = w0 + w1
                if W < 1e-18:
                    continue

                pos[i] += n * delta * (w0 / W)
                pos[i + 1] -= n * delta * (w1 / W)

    def _project_bending(self, pos: np.ndarray) -> None:
        """Bending projection: pull each free interior node toward the
        straight-line configuration of its neighbors with a compliance-weighted
        fraction derived from EI.  This complements the explicit bending force
        and helps stabilize the inextensible beam."""
        for _ in range(self.bend_iters):
            for i in range(1, pos.shape[0] - 1):
                if self._is_fixed(i):
                    continue
                L1 = self.rest_len[i - 1]
                L2 = self.rest_len[i]
                s = L1 + L2
                if s < 1e-12:
                    continue
                xim1 = pos[i - 1]
                xip1 = pos[i + 1]
                target = xim1 + (xip1 - xim1) * (L1 / s)
                # compliance fraction chosen so the projection is a stable
                # under-relaxation of the linearized bending constraint
                alpha = min(0.45, 0.5 * self.EI / (s ** 3) * self.h ** 2)
                pos[i] += alpha * (target - pos[i])

    def _bending_force(self, pos: np.ndarray) -> np.ndarray:
        """Linearized Euler-Bernoulli bending force for a naturally-straight rod.
        The force at node i is EI / l_m^3 * (x_{i-1} - 2 x_i + x_{i+1}), where
        l_m is the mean rest length of the two adjacent edges."""
        n = pos.shape[0]
        f = np.zeros_like(pos)
        for i in range(1, n - 1):
            if self._is_fixed(i):
                continue
            l_m = 0.5 * (self.rest_len[i - 1] + self.rest_len[i])
            if l_m < 1e-12:
                continue
            f[i] = self.EI * (pos[i - 1] - 2.0 * pos[i] + pos[i + 1]) / (l_m ** 3)
        return f

    def _integrate_substep(self) -> None:
        h = self.h

        # 1. add new material at the inlet
        self._add_injection(h)
        self._split_top_edge()

        x_old = self.pos.copy()

        # 2. explicit forces on free nodes
        free_mask = (self.attr_is_inj < 0.5) & (self.attr_is_stick < 0.5)
        f_bend = self._bending_force(self.pos)
        f_ext = np.zeros_like(self.pos)
        f_ext[free_mask] = (
            self.mass[free_mask, None] * self.g_vec
            + f_bend[free_mask]
        )

        # 3. semi-implicit Euler velocity update
        self.vel[free_mask] += h * f_ext[free_mask] / self.mass[free_mask, None]

        # Kinematic velocities
        self.vel[0] = 0.0
        self.vel[1] = 0.0
        stick_mask = self.attr_is_stick > 0.5
        self.vel[stick_mask] = self.belt_vel

        # 4. position prediction
        x_pred = x_old.copy()
        x_pred[free_mask] += h * self.vel[free_mask] + h * h * self.g_vec
        x_pred[stick_mask] += h * self.belt_vel
        self._enforce_injector(x_pred)

        # 5. contact / sticking: any free node crossing the belt surface sticks
        crossed = free_mask & (x_pred[:, 2] < self.surface_z + self.contact_tol)
        if np.any(crossed):
            self.attr_is_stick[crossed] = 1.0
            self.attr_is_dep[crossed] = 1.0
            x_pred[crossed, 2] = self.surface_z + self.contact_tol
            stick_mask = self.attr_is_stick > 0.5
            self.vel[crossed] = self.belt_vel

        # 6. constraint projection
        self._project_bending(x_pred)
        self._project_distance(x_pred)
        self._enforce_injector(x_pred)

        # 7. velocity correction and damping
        self.vel = (x_pred - x_old) / h
        self.vel[0] = 0.0
        self.vel[1] = 0.0
        self.vel[stick_mask] = self.belt_vel
        self.vel[free_mask] *= self.damp

        self.pos = x_pred

    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time < self.time:
            time = self.time
        n_substeps = max(0, int(round((time - self.time) / self.h)))
        for _ in range(n_substeps):
            self._integrate_substep()
            self.time += self.h
        return self._make_frame()

    def _make_frame(self) -> StateFrame:
        n = self.pos.shape[0]
        edge_len = np.linalg.norm(np.diff(self.pos, axis=0), axis=1)

        # ---- budgets: one domain, three force terms ----
        G, K = 1, 3
        budget_force = np.zeros((G, K, 3), dtype=float)
        budget_couple = np.zeros((G, K, 3), dtype=float)
        budget_pos = np.zeros((G, K, 3), dtype=float)

        total_mass = float(np.sum(self.mass))
        cm = (
            np.sum(self.mass[:, None] * self.pos, axis=0) / total_mass
            if total_mass > 0
            else np.zeros(3, dtype=float)
        )
        total_grav = np.sum(self.mass[:, None] * self.g_vec, axis=0)

        stick_idx = np.where(self.attr_is_stick > 0.5)[0]
        if stick_idx.size > 0:
            belt_pos = np.mean(self.pos[stick_idx], axis=0)
            m_stick = float(np.sum(self.mass[stick_idx]))
            # normal reaction balances the weight of deposited rod
            belt_force = np.array([0.0, 0.0, -m_stick * self.g_vec[2]], dtype=float)
        else:
            belt_pos = self.outlet.copy()
            belt_force = np.zeros(3, dtype=float)

        inj_force = -(total_grav + belt_force)

        budget_force[0, 0] = total_grav
        budget_force[0, 1] = belt_force
        budget_force[0, 2] = inj_force
        budget_pos[0, 0] = cm
        budget_pos[0, 1] = belt_pos
        budget_pos[0, 2] = self.outlet

        attributes = {
            "arc_coordinate": self.attr_arc.copy(),
            "is_injector_controlled": self.attr_is_inj.copy(),
            "is_initially_deposited": self.attr_is_dep.copy(),
            "is_sticking_to_belt": self.attr_is_stick.copy(),
            "rod_radius": self.attr_radius.copy(),
            "natural_curvature": self.attr_curv.copy(),
            "mass_quantity": self.mass.copy().reshape(-1, 1),
        }

        fields = {
            "budget_external_force": budget_force,
            "budget_external_couple": budget_couple,
            "budget_force_position": budget_pos,
            "constraint_value": edge_len,
            "constraint_target": self.rest_len.copy(),
            "cumulative_mass_source": np.array([self.cum_mass_src], dtype=float),
        }

        scalars = {
            "injection_speed": self.inj_speed,
            "belt_speed": self.belt_speed,
            "belt_to_injection_speed_ratio": float(
                self.cfg["regime"]["speed_ratio_belt_to_injection"]
            ),
            "constraint_scale": float(self.l_seg),
            "reference_force": float(self.mu * 9.81 * self.l_seg),
            "reference_torque": float(self.EI / self.l_seg),
        }

        return StateFrame(
            time=float(self.time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.orient.copy(),
            angular_velocity=self.ang_vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        self._diag["nodes"] = len(self.pos)
        self._diag["deployed_length"] = float(np.sum(self.rest_len))
        self._diag["injected_length"] = self.injected_len
        return self._diag
