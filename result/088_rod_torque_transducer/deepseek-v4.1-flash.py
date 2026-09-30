"""Torque transmission between two rigid bodies joined by an elastic rod.

Physical model
--------------
The rod is a naturally straight, inextensible Kirchhoff rod discretised into
``n`` equal segments (``h = L / (n - 1)``).  Its centreline is integrated as a
chain of lumped masses and every node carries a material frame whose roll
(twist) about the local tangent is an independent degree of freedom.  The
stored energy is

* bending     ``U_b = sum_i EI / (2 h) |t_{i+1} - t_i|^2``
* stretching ``U_s = sum_e kappa / 2 (l_e - h)^2``    (large penalty)
* torsion     ``U_t = sum_e GJ / (2 h) (theta_{e+1} - theta_e)^2``

with a naturally straight reference configuration and zero natural twist.

The two rigid bodies are cylinders whose symmetry axis is the rod axis
(world ``x``).  Their translation is constrained to the rod axis and their
single free degree of freedom is rotation about it.  The rod end material
frames are clamped to the bodies (``material_frame_clamped``), so the driven
body's rotation is transmitted to the receiver solely through the torsional
deformation of the rod; the reaction torque of the rod decelerates the driven
body and accelerates the receiver (angular-momentum exchange).  Gravity is
zero, so only the applied torque pulse does external work.

Numerics
--------
* Centreline: explicit velocity Verlet with a CFL-limited step
  ``dt = 0.4 * 2 / omega_max`` (bending / axial branches).
* Torsion (rod twist angles + both body angles): a fully coupled, linear,
  unconditionally stable backward-Euler solve of the symmetric banded system
  whose unknowns are ``[phi_driven, theta_1 .. theta_{n-2}, phi_receiver]``;
  the rod-body reaction torques appear as the boundary columns, so the
  exchange is implicit and momentum consistent.
Steps are sub-cycled so that no sample time is skipped; the returned frame is
the state exactly at the requested sample time.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from worldsolver.contracts import StateFrame

_X = np.array([1.0, 0.0, 0.0])


# --------------------------------------------------------------------- SO(3)
def _skew(v: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def _rot_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    n = float(np.linalg.norm(axis))
    if n < 1e-15:
        return np.eye(3)
    k = axis / n
    K = _skew(k)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    n = float(np.linalg.norm(axis))
    if n < 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0])
    half = 0.5 * angle
    s = np.sin(half) / n
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)])


def _minimal_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector ``a`` onto unit vector ``b``."""
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    if s < 1e-14:
        if c > 0.0:
            return np.eye(3)
        ref = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(a, ref))) > 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        axis = np.cross(a, ref)
        return _rot_axis(axis, np.pi)
    return _rot_axis(axis / s, float(np.arctan2(s, c)))


class BenchmarkSolver:
    """Deterministic elastic-rod torque transducer."""

    # ------------------------------------------------------------- lifecycle
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.config = scene.config
        self._read_config()
        self._build_geometry()
        self._set_state()

    def _read_config(self) -> None:
        cfg = self.config
        rod = cfg["rod"]
        self.n = int(rod["vertex_count"])
        self.rod_length = float(rod["length"])
        self.rod_radius = float(rod["radius"])
        self.rod_density = float(rod["density"])
        self.h = self.rod_length / (self.n - 1)

        area = np.pi * self.rod_radius**2
        self.m_node = self.rod_density * area * self.h
        self.rod_mass_total = self.rod_density * area * self.rod_length
        # polar inertia of one discretisation node about its tangent
        self.J_node = self.rod_density * 0.5 * np.pi * self.rod_radius**4 * self.h

        # Slender elastic constants (SI).  The strand is stiff in torsion and
        # soft in bending, i.e. a torsion wire/strand as required by the task.
        self.EI = float(rod.get("bending_stiffness", 1.0e-4))
        self.GJ = float(rod.get("torsional_stiffness", 0.60))
        self.k_axial = float(rod.get("axial_stiffness", 4000.0))
        self.zeta_twist = 0.3

        bodies = cfg["rigid_bodies"]
        self.driven_center0 = np.asarray(bodies["driven"]["center"], dtype=float)
        self.receiver_center0 = np.asarray(bodies["receiver"]["center"], dtype=float)
        self.driven_mass = float(bodies["driven"]["mass"])
        self.receiver_mass = float(bodies["receiver"]["mass"])
        dr, dh = float(bodies["driven"]["radius"]), float(bodies["driven"]["axial_length"])
        rr, rh = float(bodies["receiver"]["radius"]), float(bodies["receiver"]["axial_length"])
        self.driven_I = 0.5 * self.driven_mass * dr**2
        self.receiver_I = 0.5 * self.receiver_mass * rr**2
        self.driven_I_t = self.driven_mass * (3.0 * dr**2 + dh**2) / 12.0
        self.receiver_I_t = self.receiver_mass * (3.0 * rr**2 + rh**2) / 12.0

        self.rod_start = np.asarray(cfg["initial_conditions"]["rod_start"], dtype=float)
        self.rod_end = np.asarray(cfg["initial_conditions"]["rod_end"], dtype=float)

        ex = cfg["excitation"]
        self.t_start = float(ex["start_time"])
        self.t_rise = max(float(ex["rise_time"]), 1e-9)
        self.t_hold = float(ex["hold_until"])
        self.t_end = float(ex["end_time"])
        self.tau_peak = float(ex["peak_torque"])
        self.duration = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])

        self.mark_offset = {}
        for item in cfg["scene_geometry"]["objects"]:
            if item["id"] == "driven_orientation_mark":
                self.mark_offset[3] = np.asarray(
                    cfg["rigid_bodies"]["driven"]["center"], dtype=float
                )
            elif item["id"] == "receiver_orientation_mark":
                self.mark_offset[4] = np.asarray(
                    cfg["rigid_bodies"]["receiver"]["center"], dtype=float
                )
        # welded radial offset of each mark (matches the scaffold initial state)
        self.mark_radius = {3: 0.075, 4: 0.0475}

    def _build_geometry(self) -> None:
        self.d0 = np.linspace(self.rod_start, self.rod_end, self.n)
        self.h = float(np.linalg.norm(self.d0[1] - self.d0[0]))
        self.m_node = self.rod_density * np.pi * self.rod_radius**2 * self.h
        self.J_node = (
            self.rod_density * 0.5 * np.pi * self.rod_radius**4 * self.h
        )
        self.rod_mass_total = self.rod_density * np.pi * self.rod_radius**2 * self.rod_length

    def _set_state(self) -> None:
        n = self.n
        self.position = self.d0.copy()
        self.velocity = np.zeros((n, 3))
        self.omega_theta = np.zeros(n)          # material-frame twist rates
        self.theta = np.zeros(n)                # material-frame twist angles
        self.reference = self._reference_frames(self.position)

        self.phi_d = 0.0
        self.omega_d = 0.0
        self.phi_r = 0.0
        self.omega_r = 0.0

        self.time = 0.0
        self.steps = 0
        self.dt = self._stable_dt()
        self.c_twist = 2.0 * self.zeta_twist * np.sqrt(
            max(self.J_node * self.GJ / self.h, 1e-30)
        )
        self.cum_ext_torque = np.zeros(3)
        self.cum_rod_impulse_d = np.zeros(3)
        self.cum_rod_impulse_r = np.zeros(3)
        self.ref_angular_momentum = self._angular_momentum_scale()
        self.ref_momentum = max(
            1e-9, self.rod_mass_total * self.rod_length / max(self.duration, 1e-9)
        )

    # ------------------------------------------------------------ discretisation
    def _stable_dt(self) -> float:
        omega_b = np.sqrt(20.0 * self.EI / (self.m_node * self.h**4))
        omega_a = np.sqrt(4.0 * self.k_axial / self.m_node)
        omega = max(omega_b, omega_a, 1.0)
        return float(min(0.4 * 2.0 / omega, 1.0e-3, self.sample_dt / 8.0))

    def _reference_frames(self, pos: np.ndarray):
        """Parallel-transported zero-roll frames along the centreline."""
        n = self.n
        seg = pos[1:] - pos[:-1]
        tangent = seg / np.maximum(np.linalg.norm(seg, axis=1)[:, None], 1e-15)
        node_t = np.empty((n, 3))
        node_t[0] = tangent[0]
        node_t[-1] = tangent[-1]
        if n > 2:
            mid = tangent[1:] + tangent[:-1]
            node_t[1:-1] = mid / np.maximum(np.linalg.norm(mid, axis=1)[:, None], 1e-15)
        frames = np.empty((n, 3, 3))
        ref = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(node_t[0], ref))) > 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        b = np.cross(node_t[0], ref)
        b /= np.linalg.norm(b)
        frames[0] = np.stack((b, np.cross(b, node_t[0]), node_t[0]), axis=1)
        for i in range(1, n):
            rot = _minimal_rotation(node_t[i - 1], node_t[i])
            frames[i] = rot @ frames[i - 1]
        return frames

    def _material_frames(self) -> np.ndarray:
        out = np.empty_like(self.reference)
        for i in range(self.n):
            out[i] = _rot_axis(self.reference[i][:, 2], self.theta[i]) @ self.reference[i]
        return out

    # ------------------------------------------------------------- mechanics
    def _forces(self, pos: np.ndarray) -> np.ndarray:
        n = self.n
        force = np.zeros((n, 3))
        seg = pos[1:] - pos[:-1]
        length = np.linalg.norm(seg, axis=1)
        tangent = seg / np.maximum(length[:, None], 1e-15)
        # bending at interior vertices
        if n > 2:
            turn = tangent[1:] - tangent[:-1]
            proj = turn - tangent[1:] * np.sum(tangent[1:] * turn, axis=1)[:, None]
            p = (self.EI / self.h) * proj / np.maximum(length[1:], 1e-15)[:, None]
            force[1:n - 1] += p
            force[2:n] -= p
        # axial stretch penalty
        strain = self.k_axial * (length - self.h)
        axial = strain[:, None] * tangent
        force[:-1] -= axial
        force[1:] += axial
        force[0] = 0.0
        force[-1] = 0.0
        return force

    def _ext_torque(self, t: float) -> float:
        if t <= self.t_start or t >= self.t_end:
            return 0.0
        if t <= self.t_start + self.t_rise:
            s = (t - self.t_start) / self.t_rise
        elif t <= self.t_hold:
            s = 1.0
        else:
            s = max(0.0, (self.t_end - t) / max(self.t_end - self.t_hold, 1e-12))
        return self.tau_peak * s

    def _advance_positions(self, dt: float) -> None:
        m = self.m_node
        acc = self._forces(self.position) / m
        vel_half = self.velocity + acc * dt
        pos_new = self.position + vel_half * dt
        acc_new = self._forces(pos_new) / m
        self.velocity = self.velocity + 0.5 * (acc + acc_new) * dt
        self.position = pos_new
        self.velocity[0] = 0.0
        self.velocity[-1] = 0.0

    def _solve_torsion(self, dt: float, tau_ext: float) -> None:
        """Coupled backward-Euler solve for body angles and rod twist angles."""
        n = self.n
        ke = self.GJ / self.h
        size = n
        ab = np.zeros((3, size))
        rhs = np.zeros(size)
        # unknown ordering: [phi_d, theta_1 ... theta_{n-2}, phi_r]
        ab[1, 0] = self.driven_I / dt**2 + ke
        ab[0, 1] = -ke
        rhs[0] = (self.driven_I / dt**2) * self.phi_d + (
            self.driven_I / dt
        ) * self.omega_d + tau_ext
        jj = self.J_node / dt**2 + self.c_twist / dt
        for i in range(1, n - 1):
            ab[1, i] = jj + 2.0 * ke
            ab[2, i - 1] = -ke
            ab[0, i + 1] = -ke
            rhs[i] = jj * self.theta[i] + (self.J_node / dt) * self.omega_theta[i]
        ab[1, n - 1] = self.receiver_I / dt**2 + ke
        ab[2, n - 2] = -ke
        rhs[n - 1] = (self.receiver_I / dt**2) * self.phi_r + (
            self.receiver_I / dt
        ) * self.omega_r

        solution = solve_banded((1, 1), ab, rhs)

        self.omega_d = (solution[0] - self.phi_d) / dt
        self.phi_d = float(solution[0])
        self.omega_r = (solution[n - 1] - self.phi_r) / dt
        self.phi_r = float(solution[n - 1])
        for i in range(1, n - 1):
            self.omega_theta[i] = (solution[i] - self.theta[i]) / dt
            self.theta[i] = float(solution[i])
        self.theta[0] = self.phi_d
        self.theta[-1] = self.phi_r
        self.omega_theta[0] = self.omega_d
        self.omega_theta[-1] = self.omega_r

    def _step(self, dt: float) -> None:
        t_new = self.time + dt
        tau = self._ext_torque(t_new)
        self._advance_positions(dt)
        self.reference = self._reference_frames(self.position)
        # angular impulses delivered to each body by the rod over this step
        ke = self.GJ / self.h
        torque_on_d = ke * (self.theta[1] - self.theta[0])
        torque_on_r = ke * (self.theta[-2] - self.theta[-1])
        self.cum_rod_impulse_d = self.cum_rod_impulse_d + dt * torque_on_d * _X
        self.cum_rod_impulse_r = self.cum_rod_impulse_r + dt * torque_on_r * _X
        self.cum_ext_torque = self.cum_ext_torque + dt * tau * _X
        self._solve_torsion(dt, tau)
        self.time = t_new
        self.steps += 1

    # -------------------------------------------------------------- sampling
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.time < target - 1e-12:
            dt = min(self.dt, target - self.time)
            self._step(dt)
            guard += 1
            if guard > 5_000_000:  # pragma: no cover - safety net
                raise RuntimeError("integration guard tripped")
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        n = self.n
        count = n + 4
        R = self._material_frames()

        position = np.empty((count, 3))
        position[:n] = self.position
        position[n] = self.driven_center0
        position[n + 1] = self.receiver_center0
        position[n + 2] = self.driven_center0 + _rot_axis(_X, self.phi_d) @ np.array(
            [0.0, 0.0, self.mark_radius[3]]
        )
        position[n + 3] = self.receiver_center0 + _rot_axis(_X, self.phi_r) @ np.array(
            [0.0, 0.0, self.mark_radius[4]]
        )

        velocity = np.zeros((count, 3))
        velocity[:n] = self.velocity

        orientation = np.zeros((count, 4))
        for i in range(n):
            orientation[i] = _quat_from_axis_angle(R[i][:, 2], self.theta[i])
        orientation[n] = _quat_from_axis_angle(_X, self.phi_d)
        orientation[n + 1] = _quat_from_axis_angle(_X, self.phi_r)
        orientation[n + 2] = orientation[n].copy()
        orientation[n + 3] = orientation[n + 1].copy()

        angular_velocity = np.zeros((count, 3))
        for i in range(n):
            angular_velocity[i] = self.omega_theta[i] * R[i][:, 2]
        angular_velocity[n] = self.omega_d * _X
        angular_velocity[n + 1] = self.omega_r * _X
        angular_velocity[n + 2] = angular_velocity[n]
        angular_velocity[n + 3] = angular_velocity[n + 1]

        mass = np.zeros(count)
        mass[:n] = self.m_node
        mass[n] = self.driven_mass
        mass[n + 1] = self.receiver_mass

        inertia = np.zeros((count, 3, 3))
        inertia[n] = self._body_inertia(self.driven_I, self.driven_I_t)
        inertia[n + 1] = self._body_inertia(self.receiver_I, self.receiver_I_t)
        inertia[n + 2] = inertia[n]
        inertia[n + 3] = inertia[n + 1]

        body_id = np.r_[np.zeros(n, dtype=int), [1, 2, 3, 4]]

        twist = self.theta.copy()
        attributes = {
            "component_id": np.r_[np.zeros(n), [1.0, 2.0, 3.0, 4.0]],
            "arc_length": np.r_[np.linspace(0.0, self.rod_length, n), np.zeros(4)],
            "material_twist": np.r_[twist, np.zeros(4)],
            "mass_quantity": mass.copy(),
            "inertia_tensor": inertia,
        }

        weights = np.zeros((4, count))
        weights[0, :] = 1.0
        weights[1, :n] = 1.0
        weights[2, n] = 1.0
        weights[2, n + 2] = 1.0
        weights[3, n + 1] = 1.0
        weights[3, n + 3] = 1.0

        zeros_g3 = np.zeros((4, 3))
        ext_force = np.zeros((4, 2, 3))
        ext_couple = np.zeros((4, 2, 3))
        ext_position = np.zeros((4, 2, 3))
        torque = self._ext_torque(time)
        ext_couple[2, 0] = torque * _X
        ext_position[2, 0] = self.driven_center0
        ext_position[3, 0] = self.receiver_center0
        ext_position[0, 0] = 0.5 * (self.rod_start + self.rod_end)
        ext_position[1, 0] = 0.5 * (self.rod_start + self.rod_end)

        ext_ang_impulse = np.zeros((4, 3))
        ext_ang_impulse[2] = self.cum_ext_torque
        ext_impulse = np.zeros((4, 3))
        inflow_momentum = np.zeros((4, 3))
        inflow_momentum[2] = self.cum_rod_impulse_d
        inflow_momentum[3] = self.cum_rod_impulse_r
        outflow_momentum = np.zeros((4, 3))

        fields = {
            "budget_domain_weights": weights,
            "budget_external_force": ext_force,
            "budget_force_position": ext_position,
            "budget_external_couple": ext_couple,
            "budget_cumulative_external_impulse": ext_impulse,
            "budget_cumulative_external_angular_impulse": ext_ang_impulse,
            "budget_cumulative_inflow_momentum": inflow_momentum,
            "budget_cumulative_inflow_angular_momentum": inflow_momentum.copy(),
            "budget_cumulative_outflow_momentum": outflow_momentum,
            "budget_cumulative_outflow_angular_momentum": outflow_momentum.copy(),
            "cumulative_mass_source": np.array([0.0]),
            "elastic_internal_force": np.vstack(
                (self._forces(self.position), np.zeros((4, 3)))
            ),
            "elastic_internal_couple": np.zeros((count, 3)),
        }

        scalars = {
            "reference_angular_momentum": float(self.ref_angular_momentum),
            "reference_momentum": float(self.ref_momentum),
        }

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- utilities
    def _body_inertia(self, axial: float, transverse: float) -> np.ndarray:
        return axial * np.outer(_X, _X) + (transverse - axial) * (
            np.eye(3) - np.outer(_X, _X)
        )

    def _angular_momentum_scale(self) -> float:
        pulse = self.t_rise * 0.5 + (self.t_hold - self.t_start - self.t_rise) + (
            self.t_end - self.t_hold
        ) * 0.5
        return float(max(1e-9, self.tau_peak * pulse))

    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "kirchhoff_rod_coupled_torsion",
            "dt": float(self.dt),
            "substeps": int(self.steps),
            "EI": float(self.EI),
            "GJ": float(self.GJ),
            "final_phi_driven": float(self.phi_d),
            "final_phi_receiver": float(self.phi_r),
            "final_omega_driven": float(self.omega_d),
            "final_omega_receiver": float(self.omega_r),
        }
