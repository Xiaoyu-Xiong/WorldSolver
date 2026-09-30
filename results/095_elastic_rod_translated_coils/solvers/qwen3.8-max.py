"""Solver for 095_elastic_rod_translated_coils.

Quasi-static discrete elastic rod model simulating the translated-coiling
regime: an inextensible rod is fed downward through an injector onto a
translating belt. Deposited material sticks to the belt while suspended
portions obey bending elasticity and gravity, forming nearly circular coils
whose centers advect with the belt.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Discrete elastic-rod solver with belt sticking contact."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        cfg = scene.config

        # ---- rod parameters -----------------------------------------
        rod_cfg = cfg["rod"]
        self.n = int(rod_cfg["vertex_count"])
        self.radius = float(rod_cfg["radius"])
        self.density = float(rod_cfg["density"])
        self.area = np.pi * self.radius ** 2
        self.line_mass = self.density * self.area

        # Young's modulus chosen so elastoviscous length scale produces
        # cm-scale coil radii at injection_speed = 0.02 m/s (soft silicone).
        self.E = 3.0e5            # Pa
        self.I_sec = np.pi * self.radius ** 4 / 4.0
        self.B = self.E * self.I_sec   # bending rigidity N*m^2

        # ---- injector -----------------------------------------------
        inj = cfg["injector"]
        self.outlet = np.asarray(inj["outlet"], dtype=np.float64)
        self.inj_axis = np.asarray(inj["axis"], dtype=np.float64)
        self.inj_speed = float(inj["injection_speed"])
        self.feed_start = float(inj["feed_start"])
        self.feed_end = float(inj["feed_end"])

        # ---- belt ---------------------------------------------------
        belt = cfg["belt"]
        self.belt_vel = float(belt["speed"]) * np.asarray(
            belt["translation_axis"], dtype=np.float64)
        self.surface_z = float(belt["surface_z"])

        # ---- gravity ------------------------------------------------
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)

        # ---- initial conditions -------------------------------------
        init = cfg["initial_conditions"]
        self.bias = float(init["lateral_branch_bias"])
        self.winding_sign = 1.0  # CCW viewed from above

        # ---- numerical parameters -----------------------------------
        self.sub_steps = 8
        self.relax_iters = 6

        # ---- state arrays (set by _init_state) ----------------------
        self.pos = None
        self.vel = None
        self.mass_arr = None
        self.arc_len = None
        self.deposited = None
        self.controlled = None
        self.current_time = 0.0

        self._init_state()

    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        dt_sample = max(time - self.current_time, 0.0)
        if dt_sample > 0:
            h = dt_sample / self.sub_steps
            for _ in range(self.sub_steps):
                self._step(h)
            self.current_time = time
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        dep_frac = float(np.mean(self.deposited)) if self.deposited is not None else 0.0
        return {"deposited_fraction": dep_frac}

    # ------------------------------------------------------------------
    def _init_state(self) -> None:
        n = self.n
        L0 = float(self.scene.config["rod"]["initial_deployed_length"])
        s = np.linspace(0.0, L0, n)
        q = s / max(L0, 1e-12)

        pos = self.outlet[None, :] + np.column_stack([
            np.zeros(n),
            self.bias * q * q,
            -s,
        ])
        tip_z = self.surface_z + float(
            self.scene.config["initial_conditions"]["lower_tip_clearance"])
        pos[:, 2] += tip_z - pos[-1, 2]

        seg_len = L0 / max(n - 1, 1)
        mass = np.full(n, self.line_mass * seg_len)
        mass[0] *= 0.5
        mass[-1] *= 0.5

        self.pos = pos
        self.vel = np.zeros_like(pos)
        self.mass_arr = mass
        self.arc_len = s.copy()
        self.deposited = np.zeros(n, dtype=np.float64)
        ctrl = np.zeros(n, dtype=np.float64)
        ctrl[0] = 1.0
        self.controlled = ctrl
        self.current_time = 0.0

    # ------------------------------------------------------------------
    def _step(self, h: float) -> None:
        n = self.n
        pos = self.pos
        vel = self.vel

        t = self.current_time + h

        # Inject new material at outlet (vertex 0 pinned)
        pos[0] = self.outlet.copy()
        vel[0] = self.inj_axis * self.inj_speed

        # --- compute forces -----------------------------------------
        force = np.zeros_like(pos)

        active = (1.0 - self.controlled) * (1.0 - self.deposited)
        force += (self.mass_arr * active)[:, None] * self.gravity[None, :]

        f_bend = self._bending_force(pos)
        force += f_bend * active[:, None]

        f_stretch = self._stretching_force(pos)
        force += f_stretch

        # Belt sticking
        stick_mask = self.deposited > 0.5
        k_stick = 200.0
        force[stick_mask] += k_stick * self.mass_arr[stick_mask, None] * (
            self.belt_vel - vel[stick_mask])

        # Ground contact
        pen = np.maximum(self.surface_z - pos[:, 2], 0.0)
        k_ground = 5000.0
        force[:, 2] += k_ground * self.mass_arr * pen * (1.0 - self.controlled)

        # --- explicit Euler integration -----------------------------
        safe_mass = np.maximum(self.mass_arr, 1e-15)
        acc = force / safe_mass[:, None]

        vel += h * acc
        pos += h * vel

        # Enforce hard constraints
        pos[0] = self.outlet.copy()
        vel[0] = self.inj_axis * self.inj_speed

        dep = self.deposited > 0.5
        pos[dep, 2] = self.surface_z
        vel[dep, 2] = 0.0

        self._update_deposition(pos, vel, h)

        self.pos = pos
        self.vel = vel

    # ------------------------------------------------------------------
    def _bending_force(self, pos: np.ndarray) -> np.ndarray:
        n = len(pos)
        f = np.zeros_like(pos)
        if n < 3:
            return f
        e = pos[1:] - pos[:-1]
        le = np.linalg.norm(e, axis=1, keepdims=True)
        le = np.maximum(le, 1e-15)
        t = e / le

        for i in range(1, n - 1):
            l_prev = le[i - 1, 0]
            l_next = le[i, 0]
            l_bar = 0.5 * (l_prev + l_next)
            if l_bar < 1e-15:
                continue
            kappa_vec = (t[i] - t[i - 1]) / l_bar
            coeff = self.B / (l_bar ** 2)
            f[i] -= 2.0 * coeff * kappa_vec
            f[i - 1] += coeff * kappa_vec
            f[i + 1] += coeff * kappa_vec
        return f

    # ------------------------------------------------------------------
    def _stretching_force(self, pos: np.ndarray) -> np.ndarray:
        n = len(pos)
        f = np.zeros_like(pos)
        if n < 2:
            return f
        L0 = float(self.scene.config["rod"]["initial_deployed_length"])
        seg0 = L0 / max(n - 1, 1)
        k_stretch = self.E * self.area
        e = pos[1:] - pos[:-1]
        le = np.linalg.norm(e, axis=1, keepdims=True)
        le_safe = np.maximum(le, 1e-15)
        strain = (le - seg0) / seg0
        tension = k_stretch * strain
        dir_vec = e / le_safe
        f_int = tension * dir_vec
        f[1:] += f_int
        f[:-1] -= f_int
        return f

    # ------------------------------------------------------------------
    def _update_deposition(self, pos: np.ndarray, vel: np.ndarray,
                           h: float) -> None:
        near_surface = pos[:, 2] <= self.surface_z + 0.002
        slow = np.linalg.norm(vel, axis=1) < 0.05
        new_dep = near_surface & slow & (self.controlled < 0.5)

        dep_indices = np.where(new_dep)[0]
        if len(dep_indices) > 0:
            start = dep_indices[0]
            self.deposited[start:] = np.maximum(self.deposited[start:], 1.0)

        dep = self.deposited > 0.5
        pos[dep] += self.belt_vel * h
        vel[dep] = self.belt_vel.copy()

    # ------------------------------------------------------------------
    def _make_frame(self, time: float) -> StateFrame:
        n = self.n
        pos = self.pos.copy()
        vel = self.vel.copy()

        mass_quantity = self.mass_arr.copy()

        g_force = (self.mass_arr * (1.0 - self.deposited))[:, None] * \
                  self.gravity[None, :]
        pen = np.maximum(self.surface_z - pos[:, 2], 0.0)
        gr_force = np.zeros_like(pos)
        gr_force[:, 2] = 5000.0 * self.mass_arr * pen
        ext_force = g_force + gr_force
        ext_couple = np.zeros_like(pos)
        force_pos = pos.copy()

        L0 = float(self.scene.config["rod"]["initial_deployed_length"])
        seg0 = L0 / max(n - 1, 1)
        if n >= 2:
            edges = pos[1:] - pos[:-1]
            seg_lens = np.linalg.norm(edges, axis=1)
        else:
            seg_lens = np.array([seg0])
        constraint_value = seg_lens
        constraint_target = np.full_like(seg_lens, seg0)

        feed_elapsed = np.clip(time - self.feed_start, 0.0,
                               self.feed_end - self.feed_start)
        supplied_len = self.inj_speed * feed_elapsed
        cumulative_mass = np.array([self.line_mass * supplied_len])

        constraint_scale = seg0 if seg0 > 0 else 1.0
        ref_force = self.line_mass * np.linalg.norm(self.gravity) * L0
        if ref_force < 1e-15:
            ref_force = 1.0
        ref_torque = self.B / max(seg0, 1e-15)
        if ref_torque < 1e-15:
            ref_torque = 1.0

        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=self.mass_arr.copy(),
            attributes={
                "mass_quantity": mass_quantity,
                "arc_length": self.arc_len.copy(),
                "natural_curvature": np.zeros(n),
                "natural_twist": np.zeros(n),
                "injector_controlled": self.controlled.copy(),
                "deposited_sticking": self.deposited.copy(),
            },
            fields={
                "budget_external_force": ext_force.reshape(n, 1, 3),
                "budget_external_couple": ext_couple.reshape(n, 1, 3),
                "budget_force_position": force_pos.reshape(n, 1, 3),
                "constraint_value": constraint_value,
                "constraint_target": constraint_target,
                "cumulative_mass_source": cumulative_mass,
            },
            scalars={
                "constraint_scale": float(constraint_scale),
                "reference_force": float(ref_force),
                "reference_torque": float(ref_torque),
                "rod_radius": self.radius,
                "injection_speed": self.inj_speed,
                "belt_speed": float(np.linalg.norm(self.belt_vel)),
            },
        )
        frame.validate()
        return frame
