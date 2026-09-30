"""Elastic-rod solver for the moving-belt meandering experiment.

The rod is discretised as a serial chain of inextensible elastic segments.
Injection is modelled by growing the first segment (held at the injector);
when it exceeds a target length a new vertex is spawned at the injector and
the previous injector vertex is released.  Vertices that reach the belt
surface stick to it and are carried downstream.  A light transverse spring
acts on the lowest free vertex to model the source meandering instability;
the spring target obeys a damped harmonic oscillator driven by the belt/feed
speed mismatch, so the meander decays once the belt speed matches the feed.
"""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver producing a physically stepped elastic rod on a moving belt."""

    def initialize(self, scene, seed: int) -> None:  # noqa: ARG002
        self.config = scene.metadata()
        self.frame0 = scene.initial_state()

        # ---- physical parameters from the frozen configuration -----------------
        rod_cfg = self.config["rod"]
        inj_cfg = self.config["injector"]
        belt_cfg = self.config["belt"]

        self.radius = float(rod_cfg["radius"])
        self.density = float(rod_cfg["density"])
        self.young = float(rod_cfg["young_modulus"])
        self.poisson = float(rod_cfg["poisson_ratio"])
        self.gravito_bending_length = float(rod_cfg["gravito_bending_length"])

        self.v_inject = float(inj_cfg["injection_speed"])
        self.outlet = np.array(inj_cfg["outlet_position"], dtype=float)

        self.belt_dir = np.array(belt_cfg["travel_direction"], dtype=float)
        self.belt_dir /= np.linalg.norm(self.belt_dir)
        self.v_belt_initial = float(belt_cfg["initial_speed"])
        self.v_belt_matched = float(belt_cfg["matched_speed"])
        self.t_switch = float(belt_cfg["speed_switch_time"])

        self.g = np.array(self.config["gravity"], dtype=float)

        # cross-section properties
        self.cross_area = np.pi * self.radius ** 2
        self.inertia = np.pi * self.radius ** 4 / 4.0
        self.EI = self.young * self.inertia

        # ---- simulation tuning --------------------------------------------------
        self.sample_dt = float(self.config.get("sample_dt", 0.02))
        self.substeps = 20
        self.dt_sub = self.sample_dt / self.substeps
        self.damping = 5.0  # s^-1, quasi-static over-damping
        self.project_iterations = 3

        # meander oscillator parameters (physical reduced-order model of the
        # straight-to-meandering instability)
        self.omega_meander = 1.5  # rad/s
        self.zeta_meander = 0.15
        self.k_perturb = 10.0  # drives the ODE target amplitude
        self.k_spring = 0.02  # N/m, transverse spring on the heel

        # ---- state --------------------------------------------------------------
        self.t = 0.0
        self.p = np.array(self.frame0.position, dtype=float, copy=True)
        self.v = np.array(self.frame0.velocity, dtype=float, copy=True)
        self.m = np.array(self.frame0.mass, dtype=float, copy=True)
        self.body_id = np.array(self.frame0.body_id, dtype=np.int64, copy=True)
        self.orientation = np.array(self.frame0.orientation, dtype=float, copy=True)
        self.ang_vel = np.array(self.frame0.angular_velocity, dtype=float, copy=True)

        # Surface is at z = rod radius (rod centre sits on the belt top)
        self.surface_z = self.radius

        # Rest lengths from the initial geometry
        self.rest_len = np.zeros(len(self.p) - 1)
        for i in range(len(self.rest_len)):
            self.rest_len[i] = np.linalg.norm(self.p[i + 1] - self.p[i])
        self.seg_len = float(np.median(self.rest_len))
        if self.seg_len <= 0.0:
            self.seg_len = 0.02

        # Sticking mask: any vertex initially touching the belt is deposited
        self.is_sticking = self.p[:, 2] <= self.surface_z + 1e-7

        # Attributes (kept in sync on remeshing)
        self.attrs = {
            k: np.array(v, dtype=float, copy=True)
            for k, v in self.frame0.attributes.items()
        }

        # Meander target state
        self.y_target = 0.0
        self.y_dot = 0.0

        # Reference scales
        total_length = float(rod_cfg["total_available_length"])
        self.ref_force = self.density * self.cross_area * 9.81 * total_length
        self.ref_torque = self.EI / self.gravito_bending_length
        self.initial_total_mass = float(self.m.sum())

        # Per-sample-step accumulators for budget fields
        self._spring_force_accum = 0.0
        self._spring_pos = self.outlet.copy()

    def _belt_speed(self, t: float) -> float:
        return self.v_belt_initial if t < self.t_switch else self.v_belt_matched

    def _update_masses(self) -> None:
        """Recompute lumped nodal masses from current rest lengths."""
        rl = self.rest_len
        n = len(self.p)
        half = np.zeros(n)
        half[0] = rl[0] / 2.0
        half[1:-1] = (rl[:-1] + rl[1:]) / 2.0
        half[-1] = rl[-1] / 2.0
        self.m = self.density * self.cross_area * half

        # Arc coordinate follows material length
        arc = np.zeros(n)
        arc[1:] = np.cumsum(rl)
        self.attrs["arc_coordinate"] = arc

    def _insert_front(self, excess: float) -> None:
        """Spawn a new fixed injector vertex and release the old one."""
        n_old = len(self.p)
        old_first = self.p[0].copy()
        # New fixed node at the outlet
        new_p = self.outlet.reshape(1, 3)
        self.p = np.vstack([new_p, self.p])
        self.v = np.vstack([np.zeros((1, 3)), self.v])
        self.body_id = np.concatenate([np.zeros(1, dtype=np.int64), self.body_id])

        # Extend orientation / angular velocity
        self.orientation = np.vstack([np.array([0.0, 0.0, 0.0, 1.0]), self.orientation])
        self.ang_vel = np.vstack([np.zeros((1, 3)), self.ang_vel])

        # Extend attributes
        new_attrs = {}
        for name, arr in self.attrs.items():
            new_arr = np.zeros(n_old + 1)
            if name == "is_injector_controlled":
                new_arr[0] = 1.0
                new_arr[1:] = arr
                new_arr[1] = 0.0  # old injector node is now free
            elif name in ("is_initially_deposited", "is_sticking_to_belt", "natural_curvature"):
                new_arr[1:] = arr
            elif name == "rod_radius":
                new_arr[1:] = arr
                new_arr[0] = self.radius
            elif name == "arc_coordinate":
                new_arr[1:] = arr + excess
            else:
                new_arr[1:] = arr
            new_attrs[name] = new_arr
        self.attrs = new_attrs

        # Extend rest lengths.  New first edge length is the actual distance
        # from the injector to the previous first vertex.  The old first edge
        # keeps its physical length.
        new_rest = np.zeros(n_old)
        new_rest[0] = float(np.linalg.norm(self.p[0] - old_first))
        if n_old > 1:
            new_rest[1:] = self.rest_len
        self.rest_len = new_rest

        # Extend sticking mask
        self.is_sticking = np.concatenate([np.zeros(1, dtype=bool), self.is_sticking])

    def _project_lengths(self) -> None:
        """Fast projection enforcing inextensibility (fixed/sticking vertices kinematic)."""
        kinematic = self.is_sticking.copy()
        kinematic[0] = True  # injector is fixed
        for _ in range(self.project_iterations):
            for i in range(len(self.rest_len)):
                a = i
                b = i + 1
                d = self.p[b] - self.p[a]
                len_d = np.linalg.norm(d)
                if len_d == 0.0:
                    continue
                diff = (len_d - self.rest_len[i]) / len_d
                if abs(diff) < 1e-9:
                    continue
                corr = diff * d
                wa = 0.0 if kinematic[a] else 1.0
                wb = 0.0 if kinematic[b] else 1.0
                w = wa + wb
                if w == 0.0:
                    continue
                self.p[a] += (wa / w) * corr
                self.p[b] -= (wb / w) * corr

    def _substep(self, t: float, dt: float) -> None:
        n = len(self.p)
        belt_speed = self._belt_speed(t)

        # --- gravity -----------------------------------------------------------
        self.v += dt * self.g

        # --- bending -----------------------------------------------------------
        for i in range(1, n - 1):
            h = 0.5 * (self.rest_len[i - 1] + self.rest_len[i])
            if h <= 0.0:
                continue
            k = self.EI / (h ** 3)
            sd = self.p[i + 1] - 2.0 * self.p[i] + self.p[i - 1]
            self.v[i] += dt * k * sd / self.m[i]

        # --- meander spring on the lowest free vertex (the heel region) ----------
        free_mask = ~self.is_sticking
        free_mask[0] = False  # injector is not part of the free heel
        free_idx = np.where(free_mask)[0]
        if free_idx.size > 0:
            lowest = free_idx[np.argmin(self.p[free_idx, 2])]
            Fy = self.k_spring * (self.y_target - self.p[lowest, 1])
            self.v[lowest, 1] += dt * Fy / self.m[lowest]
            self._spring_force_accum += Fy
            self._spring_pos = self.p[lowest].copy()

        # --- integrate the contact-point meander target ------------------------
        dv = self.v_inject - belt_speed
        omega = self.omega_meander
        zeta = self.zeta_meander
        forcing = self.k_perturb * dv * np.sin(omega * t)
        ydd = -2.0 * zeta * omega * self.y_dot - omega ** 2 * self.y_target + forcing
        self.y_dot += dt * ydd
        self.y_target += dt * self.y_dot

        # --- damping + boundary conditions ------------------------------------
        self.v *= 1.0 - self.damping * dt
        self.v[0] = 0.0
        self.p[0] = self.outlet

        # Sticking vertices move with the belt
        stick_idx = np.where(self.is_sticking)[0]
        if stick_idx.size > 0:
            self.v[stick_idx] = belt_speed * self.belt_dir

        # --- integrate positions ------------------------------------------------
        self.p += self.v * dt

        # --- floor contact ------------------------------------------------------
        below = self.p[:, 2] < self.surface_z
        if np.any(below):
            self.p[below, 2] = self.surface_z
            self.is_sticking[below] = True
            self.v[below, 2] = 0.0

        # Re-apply belt velocity to sticking vertices
        if stick_idx.size > 0:
            self.v[stick_idx] = belt_speed * self.belt_dir

        # --- inextensibility projection ----------------------------------------
        self._project_lengths()

        # --- enforce floor after projection ------------------------------------
        below = self.p[:, 2] < self.surface_z
        if np.any(below):
            self.p[below, 2] = self.surface_z
            self.is_sticking[below] = True

        # --- injector constraint ------------------------------------------------
        self.p[0] = self.outlet
        self.v[0] = 0.0

        # --- feed material through the first edge -------------------------------
        if len(self.rest_len) > 0:
            self.rest_len[0] += self.v_inject * dt
            if self.rest_len[0] > self.seg_len:
                excess = self.rest_len[0] - self.seg_len
                self._insert_front(excess)

        # --- keep masses and arc coordinate in sync -----------------------------
        self._update_masses()

    def _step(self, dt: float) -> None:
        # Reset budget accumulators for this sample step
        self._spring_force_accum = 0.0
        self._spring_pos = self.outlet.copy()

        for i in range(self.substeps):
            self._substep(self.t + i * self.dt_sub, self.dt_sub)
        self.t += dt

    def advance_to(self, time: float) -> StateFrame:
        if time < self.t - 1e-12:
            raise ValueError("Cannot advance backwards in time")
        if time == 0.0:
            return self._frame(0.0)

        dt = time - self.t
        # Should normally equal sample_dt, but be robust
        n_steps = max(1, int(round(dt / self.sample_dt)))
        for _ in range(n_steps):
            self._step(self.sample_dt)
        return self._frame(self.t)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "BenchmarkSolver",
            "num_vertices": len(self.p),
            "total_rest_length": float(self.rest_len.sum()),
            "final_belt_speed": self._belt_speed(self.t),
        }

    def _frame(self, t: float) -> StateFrame:
        # Current edge lengths for constraint_value
        current_len = np.array(
            [np.linalg.norm(self.p[i + 1] - self.p[i]) for i in range(len(self.p) - 1)],
            dtype=float,
        )

        # Budget external force: gravity + transverse meander spring
        total_mass = self.m.sum()
        gravity_force = total_mass * self.g
        external_force = np.zeros((1, 2, 3), dtype=float)
        external_force[0, 0, :] = gravity_force
        external_force[0, 1, :] = np.array([0.0, self._spring_force_accum, 0.0])

        external_couple = np.zeros((1, 2, 3), dtype=float)

        com = (self.m[:, None] * self.p).sum(axis=0) / total_mass
        force_position = np.zeros((1, 2, 3), dtype=float)
        force_position[0, 0, :] = com
        force_position[0, 1, :] = self._spring_pos

        # Cumulative injected mass
        added_mass = total_mass - self.initial_total_mass

        # Attribute: per-sample mass contribution
        attributes = {
            "mass_quantity": self.m.copy(),
            "arc_coordinate": self.attrs["arc_coordinate"].copy(),
            "is_injector_controlled": (np.arange(len(self.p)) == 0).astype(float),
            "is_initially_deposited": self.attrs["is_initially_deposited"].copy(),
            "is_sticking_to_belt": self.is_sticking.astype(float),
            "rod_radius": np.full(len(self.p), self.radius),
            "natural_curvature": np.zeros(len(self.p)),
        }

        fields = {
            "budget_external_force": external_force,
            "budget_external_couple": external_couple,
            "budget_force_position": force_position,
            "constraint_target": self.rest_len.copy(),
            "constraint_value": current_len,
            "cumulative_mass_source": np.array([added_mass], dtype=float),
        }

        scalars = {
            "constraint_scale": float(np.mean(self.rest_len)),
            "reference_force": float(self.ref_force),
            "reference_torque": float(self.ref_torque),
        }

        return StateFrame(
            time=float(t),
            position=self.p.copy(),
            velocity=self.v.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.ang_vel.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
