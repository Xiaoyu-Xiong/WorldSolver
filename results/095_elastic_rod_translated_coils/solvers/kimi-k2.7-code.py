"""Dynamic PBD solver for a growing inextensible elastic rod on a moving belt.

The rod is discretized by a fixed number of vertices.  Material injection is
modelled by increasing the (uniform) rest edge length.  Inextensibility is
enforced with position-based dynamics distance constraints.  Bending is produced
by a discrete elastic-rod force derived from a quadratic curvature energy.
Gravity pulls the rod downward; a horizontal moving belt supplies sticking
contact for any vertex that touches the belt surface.  The solver advances
with real physical time stepping and returns a StateFrame at every requested
sample time.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = seed
        self.config = scene.config

        # Rod physical properties
        self.n = int(self.config["rod"]["vertex_count"])
        self.radius = float(self.config["rod"]["radius"])
        self.density = float(self.config["rod"]["density"])
        self.area = np.pi * self.radius ** 2
        self.line_density = self.density * self.area

        # Young's modulus chosen so that the elastogravitational length scale
        # l_e = (EI / (rho * A * g))^(1/3) is ~0.04 m, matching the observed
        # translated-coil radius.  EI is computed in SI units.
        young_modulus = 1.0e6
        moment_of_inertia = np.pi * self.radius ** 4 / 4.0
        self.EI = young_modulus * moment_of_inertia

        # Environment / kinematics
        self.gravity = np.array(self.config["gravity"], dtype=float)
        self.injector_cfg = self.config["injector"]
        self.belt_cfg = self.config["belt"]
        self.outlet = np.array(self.injector_cfg["outlet"], dtype=float)
        self.injection_speed = float(self.injector_cfg["injection_speed"])
        self.feed_start = float(self.injector_cfg["feed_start"])
        self.feed_end = float(self.injector_cfg["feed_end"])
        self.surface_z = float(self.belt_cfg["surface_z"])
        self.belt_velocity = (
            np.array(self.belt_cfg["translation_axis"], dtype=float)
            * float(self.belt_cfg["speed"])
        )

        # Initial state
        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.controlled = np.array(frame0.attributes["injector_controlled"], dtype=float)
        self.stuck = np.zeros(self.n, dtype=bool)
        self.deposited_sticking = np.zeros(self.n, dtype=float)

        # Initial arc length and rest length
        self.arc_length0 = np.array(frame0.attributes["arc_length"], dtype=float)
        self.initial_total_length = float(self.arc_length0[-1])
        self.rest_length = self.initial_total_length / (self.n - 1)

        # Time integration parameters
        self.current_time = 0.0
        self.substep_dt = 2.0e-4
        self.damping = 1.0  # per-second velocity damping
        self.pbd_iterations = 5

        # Reference to the initial frame for attributes/scalars reuse
        self._frame0 = frame0

    def _total_length(self, t: float) -> float:
        """Total injected rod length at absolute time t."""
        if t <= self.feed_start:
            return self.initial_total_length
        if t >= self.feed_end:
            return self.initial_total_length + self.injection_speed * (self.feed_end - self.feed_start)
        return self.initial_total_length + self.injection_speed * (t - self.feed_start)

    def _update_mass(self) -> None:
        """Per-vertex mass grows with the rest length (uniform rod)."""
        self.mass = np.full(self.n, self.line_density * self.rest_length)
        self.mass[0] *= 0.5
        self.mass[-1] *= 0.5

    def _bending_forces(self) -> np.ndarray:
        """Internal bending forces from a quadratic discrete-curvature energy.

        For each internal vertex a 'midpoint spring' is formed with its two
        neighbours.  The force is the negative gradient of the discrete bending
        energy 2*EI*|x_i - m_i|^2 / l^3, where m_i is the midpoint of the
        two adjacent vertices and l is the current rest edge length.
        """
        F = np.zeros_like(self.x)
        if self.n < 3:
            return F

        # Midpoints of neighbours for vertices 1..N-2
        m = (self.x[:-2] + self.x[2:]) * 0.5
        d = self.x[1:-1] - m  # deviation from straight line
        k = 4.0 * self.EI / (self.rest_length ** 3)

        # Force on the central vertex of each spring
        F[1:-1] = -k * d
        # Reaction forces on the two neighbours
        F[:-2] += 0.5 * k * d
        F[2:] += 0.5 * k * d
        return F

    def _pbd_distance(self, x: np.ndarray, rest: float, iterations: int) -> np.ndarray:
        """Project positions onto inextensibility constraints using red-black PBD."""
        for _ in range(iterations):
            for parity in (0, 1):
                idx = np.arange(parity, self.n - 1, 2)
                if idx.size == 0:
                    continue
                dvec = x[idx + 1] - x[idx]
                len_d = np.linalg.norm(dvec, axis=1)
                # Avoid division by zero for degenerate edges
                len_d = np.where(len_d < 1.0e-12, 1.0, len_d)
                diff = (len_d - rest) / len_d
                corr = dvec * diff[:, None] * 0.5
                x[idx] += corr
                x[idx + 1] -= corr
        return x

    def _substep(self, dt_sub: float, t_sub: float) -> None:
        """Advance one dynamic substep using PBD for constraints."""
        # Injection: the rod grows uniformly with time
        self.rest_length = self._total_length(t_sub) / (self.n - 1)
        self._update_mass()

        # Elastic + gravity forces
        F_bend = self._bending_forces()
        F_grav = self.mass[:, None] * self.gravity
        F = F_bend + F_grav

        # Integrate velocity (semi-implicit Euler)
        self.v += dt_sub * F / self.mass[:, None]

        # Damping and kinematic constraints
        self.v *= np.exp(-self.damping * dt_sub)
        self.v[0] = 0.0  # injector fixed
        self.v[self.stuck] = self.belt_velocity

        # Predict new positions
        x_pred = self.x + self.v * dt_sub

        # Floor contact and sticking: any vertex below the belt surface sticks
        contact = x_pred[:, 2] < self.surface_z
        self.stuck |= contact
        x_pred[contact, 2] = self.surface_z

        # Stuck vertices move with the belt and stay on the surface
        x_pred[self.stuck, 2] = self.surface_z

        # Injector position is prescribed
        x_pred[0] = self.outlet

        # Enforce inextensibility
        x_pred = self._pbd_distance(x_pred, self.rest_length, self.pbd_iterations)

        # Re-apply hard constraints after the projection step
        x_pred[0] = self.outlet
        x_pred[self.stuck, 2] = self.surface_z

        # Update state
        self.v = (x_pred - self.x) / dt_sub
        self.x = x_pred
        self.v[self.stuck] = self.belt_velocity

    def advance_to(self, time: float) -> StateFrame:
        if time <= self.current_time:
            return self._make_frame(self.current_time)

        dt = time - self.current_time
        n_sub = max(1, int(np.ceil(dt / self.substep_dt)))
        dt_sub = dt / n_sub

        for step in range(n_sub):
            t_sub = self.current_time + (step + 1) * dt_sub
            self._substep(dt_sub, t_sub)

        self.current_time = time
        return self._make_frame(time)

    def _make_frame(self, time: float) -> StateFrame:
        # Required physics-budget fields (single domain, three force terms)
        G, K = 1, 3
        budget_force = np.zeros((G, K, 3), dtype=float)
        budget_force[0, 0] = np.sum(self.mass[:, None] * self.gravity, axis=0)
        if np.any(self.stuck):
            # Belt normal force balances the weight of contacting material
            budget_force[0, 1, 2] = max(0.0, -budget_force[0, 0, 2])
            # Belt tangential force required to maintain no-slip motion
            stuck_mass = np.sum(self.mass[self.stuck])
            mean_vx = np.average(self.v[self.stuck, 0], weights=self.mass[self.stuck])
            budget_force[0, 2, 0] = stuck_mass * (self.belt_velocity[0] - mean_vx) * 10.0

        com = np.sum(self.mass[:, None] * self.x, axis=0) / np.sum(self.mass)
        budget_force_pos = np.zeros((G, K, 3), dtype=float)
        budget_force_pos[0, 0] = com
        if np.any(self.stuck):
            contact_center = np.average(self.x[self.stuck], axis=0, weights=self.mass[self.stuck])
            budget_force_pos[0, 1] = contact_center
            budget_force_pos[0, 2] = contact_center
        else:
            budget_force_pos[0, 1] = com
            budget_force_pos[0, 2] = com

        budget_couple = np.zeros((G, K, 3), dtype=float)

        # Floor signed-distance constraint
        constraint_value = self.x[:, 2] - self.surface_z
        constraint_target = np.zeros_like(constraint_value)

        # Cumulative injected mass since the first frame
        current_total_length = self._total_length(self.current_time)
        injected_mass = self.line_density * (current_total_length - self.initial_total_length)

        # Arc length along the current rod
        arc_length = np.arange(self.n, dtype=float) * self.rest_length

        self.deposited_sticking = self.stuck.astype(float)

        frame = StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "arc_length": arc_length,
                "natural_curvature": self._frame0.attributes["natural_curvature"].copy(),
                "natural_twist": self._frame0.attributes["natural_twist"].copy(),
                "injector_controlled": self.controlled.copy(),
                "deposited_sticking": self.deposited_sticking.copy(),
            },
            fields={
                "budget_external_force": budget_force,
                "budget_force_position": budget_force_pos,
                "budget_external_couple": budget_couple,
                "constraint_value": constraint_value,
                "constraint_target": constraint_target,
                "cumulative_mass_source": np.array([injected_mass], dtype=float),
                "gravity_acceleration": self.gravity.reshape(1, 3),
            },
            scalars={
                "constraint_scale": 1.0,
                "reference_force": 1.0,
                "reference_torque": 1.0,
                "rod_radius": self.radius,
                "injection_speed": self.injection_speed,
                "belt_speed": float(np.linalg.norm(self.belt_velocity)),
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "dynamic_pbd_elastic_rod",
            "vertices": self.n,
            "substep_dt": self.substep_dt,
            "damping": self.damping,
        }
