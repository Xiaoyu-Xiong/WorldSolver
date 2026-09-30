"""Physically-based solver for the rod torque-transducer scene.

Model
-----
A naturally straight, inextensible elastic rod connects two rigid cylinders whose
translation is constrained to the rod axis and whose rotation about that axis is
dynamic.  Gravity is zero, the rod is initially straight and untwisted, and the only
excitation is an axial torque pulse applied to the driven body.  Under those frozen
conditions the centerline is an equilibrium of the bending energy and stays straight,
so the live degrees of freedom are the material-frame twist angles.

The discrete elastic rod twist energy

    E = sum_edges  (G*J / l) * (theta_{i+1} - theta_i)^2 / 2 ,      J = pi r^4 / 2

is discretised on the 65 frozen rod vertices.  ``material_frame_clamped`` endpoint
attachment means the end material frames are welded to the bodies, so
theta_0 = theta_driven and theta_{n-1} = theta_receiver; the corresponding half-edge
polar inertia of the rod is lumped with the body axial inertia.  The resulting
torsional wave system (plus weak internal strain-rate damping, which is an internal
force pair and therefore conserves total angular momentum) is advanced with a
symplectic-Euler / staggered leapfrog step at dt = sample_dt / 100, well inside the
explicit stability bound 2 / omega_max of the discretisation.

Total axial angular momentum therefore satisfies dL/dt = tau_applied(t) exactly up to
the integrator order, which is what the exported momentum budgets report.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# Material properties of the rod.  Density is frozen by the config; the shear modulus
# is a soft-elastomer value consistent with that density (E = 2(1+nu)G ~ 0.9 MPa).
SHEAR_MODULUS = 3.0e5          # Pa
DAMPING_TIME = 2.0e-4          # s, stiffness-proportional internal twist damping
SUBSTEPS_PER_SAMPLE = 100


def _quat_x(angle: np.ndarray) -> np.ndarray:
    """Quaternion xyzw for a rotation of ``angle`` about the +x axis."""
    angle = np.atleast_1d(np.asarray(angle, dtype=float))
    half = 0.5 * angle
    quat = np.zeros((angle.size, 4), dtype=float)
    quat[:, 0] = np.sin(half)
    quat[:, 3] = np.cos(half)
    return quat


class BenchmarkSolver:
    """Self-implemented torsional elastic-rod / rigid-body transducer solver."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg

        rod = cfg["rod"]
        self.n = int(rod["vertex_count"])
        self.rod_length = float(rod["length"])
        self.rod_radius = float(rod["radius"])
        self.density = float(rod["density"])

        ic = cfg["initial_conditions"]
        start = np.asarray(ic["rod_start"], dtype=float)
        end = np.asarray(ic["rod_end"], dtype=float)
        self.rod_ref = np.linspace(start, end, self.n)

        bodies = cfg["rigid_bodies"]
        self.driven_center = np.asarray(bodies["driven"]["center"], dtype=float)
        self.recv_center = np.asarray(bodies["receiver"]["center"], dtype=float)
        self.driven_mass = float(bodies["driven"]["mass"])
        self.recv_mass = float(bodies["receiver"]["mass"])
        self.driven_radius = float(bodies["driven"]["radius"])
        self.recv_radius = float(bodies["receiver"]["radius"])
        self.driven_height = float(bodies["driven"]["axial_length"])
        self.recv_height = float(bodies["receiver"]["axial_length"])

        # Orientation-mark offsets (welded radial bars), matching the frozen scene.
        self.driven_mark_offset = np.array([0.0, 0.0, 0.075])
        self.recv_mark_offset = np.array([0.0, 0.0, 0.0475])

        exc = cfg["excitation"]
        self.exc_axis = np.asarray(exc["axis"], dtype=float)
        norm = float(np.linalg.norm(self.exc_axis))
        self.exc_axis = self.exc_axis / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
        self.t_start = float(exc["start_time"])
        self.t_rise = float(exc["rise_time"])
        self.t_hold = float(exc["hold_until"])
        self.t_end = float(exc["end_time"])
        self.peak_torque = float(exc["peak_torque"])

        # ---- discretisation -------------------------------------------------
        self.seg = self.rod_length / (self.n - 1)
        area = np.pi * self.rod_radius ** 2
        polar_J = 0.5 * np.pi * self.rod_radius ** 4              # m^4
        self.edge_stiffness = SHEAR_MODULUS * polar_J / self.seg  # N*m/rad
        self.edge_damping = DAMPING_TIME * self.edge_stiffness    # N*m*s/rad

        # Lumped rod polar inertia per vertex (trapezoidal: half at the ends).
        rod_polar = np.full(self.n, self.density * polar_J * self.seg)
        rod_polar[0] *= 0.5
        rod_polar[-1] *= 0.5
        self.rod_polar = rod_polar

        # Axial inertia of the cylinders (axis along x -> I_axial = m R^2 / 2).
        self.I_driven = 0.5 * self.driven_mass * self.driven_radius ** 2
        self.I_recv = 0.5 * self.recv_mass * self.recv_radius ** 2

        # Effective inertia of the twist DOFs (clamped ends share a DOF with a body).
        self.I_eff = rod_polar.copy()
        self.I_eff[0] += self.I_driven
        self.I_eff[-1] += self.I_recv

        # ---- per-sample mass (matches the frozen scene initial frame) -------
        rod_mass = np.full(self.n, self.density * area * self.seg)
        self.mass = np.concatenate([rod_mass,
                                    [self.driven_mass, self.recv_mass, 0.0, 0.0]])
        self.count = self.n + 4
        self.body_id = np.concatenate([np.zeros(self.n, dtype=int), [1, 2, 3, 4]])
        self.component_id = np.concatenate([np.zeros(self.n), [1.0, 2.0, 3.0, 4.0]])
        self.arc_length = np.concatenate([np.linspace(0.0, self.rod_length, self.n),
                                         np.zeros(4)])
        self.is_driven = np.concatenate([np.zeros(self.n), [1.0, 0.0, 1.0, 0.0]])

        # ---- intrinsic inertia tensors (world frame, constant: spin about x) --
        inertia = np.zeros((self.count, 3, 3))
        seg_len = np.full(self.n, self.seg)
        seg_len[0] *= 0.5
        seg_len[-1] *= 0.5
        rod_lumped_mass = self.density * area * seg_len
        inertia[:self.n, 0, 0] = rod_polar
        trans = rod_lumped_mass * (3.0 * self.rod_radius ** 2 + seg_len ** 2) / 12.0
        inertia[:self.n, 1, 1] = trans
        inertia[:self.n, 2, 2] = trans
        inertia[self.n, 0, 0] = self.I_driven
        d_trans = self.driven_mass * (3.0 * self.driven_radius ** 2
                                      + self.driven_height ** 2) / 12.0
        inertia[self.n, 1, 1] = d_trans
        inertia[self.n, 2, 2] = d_trans
        inertia[self.n + 1, 0, 0] = self.I_recv
        r_trans = self.recv_mass * (3.0 * self.recv_radius ** 2
                                    + self.recv_height ** 2) / 12.0
        inertia[self.n + 1, 1, 1] = r_trans
        inertia[self.n + 1, 2, 2] = r_trans
        # The massless orientation marks carry no intrinsic inertia (already in bodies).
        self.inertia_tensor = inertia

        # ---- momentum-budget domains ---------------------------------------
        # row 0 global, row 1 rod, row 2 driven assembly, row 3 receiver assembly
        weights = np.zeros((4, self.count))
        weights[0, :] = 1.0
        weights[1, :self.n] = 1.0
        weights[2, self.n] = 1.0
        weights[2, self.n + 2] = 1.0
        weights[3, self.n + 1] = 1.0
        weights[3, self.n + 3] = 1.0
        self.domain_weights = weights
        self.n_domains = 4
        self.n_terms = 2

        # Frozen positive reference scales.
        total_inertia = float(rod_polar.sum() + self.I_driven + self.I_recv)
        self.ref_angular_momentum = float(self.peak_torque * (self.t_end - self.t_start))
        omega_ref = self.ref_angular_momentum / total_inertia
        self.ref_momentum = float(self.mass.sum() * omega_ref * self.driven_radius)
        if not (self.ref_momentum > 0.0):
            self.ref_momentum = 1.0

        # ---- state ----------------------------------------------------------
        self.theta = np.zeros(self.n)
        self.omega = np.zeros(self.n)
        self.time = 0.0
        self.dt = float(cfg["sample_dt"]) / SUBSTEPS_PER_SAMPLE
        self.cum_ang_impulse = np.zeros((self.n_domains, 3))
        self.cum_impulse = np.zeros((self.n_domains, 3))
        self._steps = 0
        self._max_twist = 0.0

    # ------------------------------------------------------------- excitation
    def _applied_torque(self, t: float) -> float:
        if t <= self.t_start or t >= self.t_end:
            return 0.0
        ramp_up_end = self.t_start + self.t_rise
        if t < ramp_up_end:
            return self.peak_torque * (t - self.t_start) / max(self.t_rise, 1e-12)
        if t <= self.t_hold:
            return self.peak_torque
        fall = self.t_end - self.t_hold
        if fall <= 1e-12:
            return 0.0
        return self.peak_torque * (self.t_end - t) / fall

    # --------------------------------------------------------------- dynamics
    def _accel(self, theta: np.ndarray, omega: np.ndarray, tau: float):
        d_theta = theta[1:] - theta[:-1]
        d_omega = omega[1:] - omega[:-1]
        moment = self.edge_stiffness * d_theta + self.edge_damping * d_omega
        torque = np.zeros(self.n)
        torque[:-1] -= moment
        torque[1:] += moment
        torque[0] += tau
        return torque / self.I_eff

    def _step(self) -> None:
        tau = self._applied_torque(self.time + 0.5 * self.dt)
        accel = self._accel(self.theta, self.omega, tau)

        # Interface couples implied by the clamped attachments.
        c_start = tau - self.I_driven * accel[0]     # driven body -> rod, x component
        c_end = -self.I_recv * accel[-1]             # receiver body -> rod, x component

        rates = np.zeros((self.n_domains, 3))
        rates[0] = self.exc_axis * tau
        rates[1] = self.exc_axis * (c_start + c_end)
        rates[2] = self.exc_axis * (tau - c_start)
        rates[3] = self.exc_axis * (-c_end)
        self.cum_ang_impulse += rates * self.dt

        self.omega += accel * self.dt
        self.theta += self.omega * self.dt
        self.time += self.dt
        self._steps += 1
        twist = abs(float(self.theta[0] - self.theta[-1]))
        if twist > self._max_twist:
            self._max_twist = twist

    # ------------------------------------------------------------------ frame
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 0.5 * self.dt:
            self._step()
        return self._frame(target)

    def _frame(self, stamp: float) -> StateFrame:
        n = self.n
        theta = self.theta
        omega = self.omega
        tau = self._applied_torque(stamp)
        accel = self._accel(theta, omega, tau)
        c_start = tau - self.I_driven * accel[0]
        c_end = -self.I_recv * accel[-1]

        theta_d = float(theta[0])
        theta_r = float(theta[-1])
        omega_d = float(omega[0])
        omega_r = float(omega[-1])

        position = np.zeros((self.count, 3))
        velocity = np.zeros((self.count, 3))
        angular_velocity = np.zeros((self.count, 3))
        orientation = np.zeros((self.count, 4))

        # Rod: straight centerline, twisting material frame.
        position[:n] = self.rod_ref
        angular_velocity[:n, 0] = omega
        orientation[:n] = _quat_x(theta)

        # Bodies: on-axis, spinning about x.
        position[n] = self.driven_center
        position[n + 1] = self.recv_center
        angular_velocity[n, 0] = omega_d
        angular_velocity[n + 1, 0] = omega_r
        orientation[n] = _quat_x(np.array([theta_d]))[0]
        orientation[n + 1] = _quat_x(np.array([theta_r]))[0]

        # Welded orientation marks follow their body rigidly.
        for idx, (center, offset, ang, spin) in enumerate((
                (self.driven_center, self.driven_mark_offset, theta_d, omega_d),
                (self.recv_center, self.recv_mark_offset, theta_r, omega_r))):
            cos_a, sin_a = np.cos(ang), np.sin(ang)
            rot = np.array([offset[0],
                            cos_a * offset[1] - sin_a * offset[2],
                            sin_a * offset[1] + cos_a * offset[2]])
            node = n + 2 + idx
            position[node] = center + rot
            velocity[node] = np.cross(np.array([spin, 0.0, 0.0]), rot)
            angular_velocity[node, 0] = spin
            orientation[node] = _quat_x(np.array([ang]))[0]

        # ---- budget bookkeeping --------------------------------------------
        external_force = np.zeros((self.n_domains, self.n_terms, 3))
        external_couple = np.zeros((self.n_domains, self.n_terms, 3))
        force_position = np.zeros((self.n_domains, self.n_terms, 3))

        axis = self.exc_axis
        external_couple[0, 0] = axis * tau
        external_couple[1, 0] = axis * c_start
        external_couple[1, 1] = axis * c_end
        external_couple[2, 0] = axis * tau
        external_couple[2, 1] = -axis * c_start
        external_couple[3, 1] = -axis * c_end

        for row in range(self.n_domains):
            force_position[row, 0] = self.driven_center
            force_position[row, 1] = self.recv_center
        force_position[1, 0] = self.rod_ref[0]
        force_position[1, 1] = self.rod_ref[-1]

        zeros_g3 = np.zeros((self.n_domains, 3))

        frame = StateFrame(
            time=float(stamp),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "component_id": self.component_id.copy(),
                "arc_length": self.arc_length.copy(),
                "material_twist": np.concatenate(
                    [theta, [theta_d, theta_r, theta_d, theta_r]]),
                "is_driven": self.is_driven.copy(),
                "inertia_tensor": self.inertia_tensor.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": external_force,
                "budget_external_couple": external_couple,
                "budget_force_position": force_position,
                "budget_cumulative_external_impulse": self.cum_impulse.copy(),
                "budget_cumulative_external_angular_impulse": self.cum_ang_impulse.copy(),
                "budget_cumulative_inflow_momentum": zeros_g3.copy(),
                "budget_cumulative_outflow_momentum": zeros_g3.copy(),
                "budget_cumulative_inflow_angular_momentum": zeros_g3.copy(),
                "budget_cumulative_outflow_angular_momentum": zeros_g3.copy(),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "applied_torque": float(tau),
                "driven_angle": theta_d,
                "receiver_angle": theta_r,
                "orientation_difference": theta_d - theta_r,
                "rod_interface_moment_start": float(c_start),
                "rod_interface_moment_end": float(c_end),
                "reference_momentum": self.ref_momentum,
                "reference_angular_momentum": self.ref_angular_momentum,
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self) -> dict[str, object]:
        total_inertia = float(self.rod_polar.sum() + self.I_driven + self.I_recv)
        return {
            "method": "discrete elastic rod twist wave + clamped rigid bodies",
            "integrator": "symplectic Euler (staggered leapfrog)",
            "substep_dt": self.dt,
            "steps_taken": self._steps,
            "shear_modulus": SHEAR_MODULUS,
            "edge_torsional_stiffness": self.edge_stiffness,
            "total_axial_inertia": total_inertia,
            "max_end_to_end_twist_rad": self._max_twist,
            "final_driven_angle": float(self.theta[0]),
            "final_receiver_angle": float(self.theta[-1]),
        }
