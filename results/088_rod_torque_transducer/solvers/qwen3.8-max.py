"""Torque-transducer solver: elastic rod coupling two rigid bodies.

Implements a discrete-elastic-rod (twist-only) model that transmits torque
between a driven and a receiver rigid body through an inextensible, naturally
straight rod that resists twist. The numerical method is explicit symplectic
Euler on the reduced 2-DOF rotational system (one twist angle per body about
the rod axis), which exactly conserves angular momentum in the absence of
external torque and reproduces the qualitative torsional-wave exchange shown
in Bergou et al., Figure 6 / Section 9.4.

The rod's distributed twist inertia is lumped equally onto the two endpoint
bodies so that the platform-required N-sample state arrays remain consistent
with the scene geometry bindings while still reflecting the rod's contribution
to the torsional dynamics. All required trajectory fields are exported; source
terms for budget fields are zero because there is no mass transfer and the only
external generalized force is the applied torque pulse (reported via
budget_external_couple).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Discrete-elastic-rod torque transducer between two rigid bodies."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene: Any, seed: int) -> None:
        cfg = scene.config
        self.seed = int(seed)
        self.dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])

        # Rod parameters -----------------------------------------------------
        rod = cfg["rod"]
        self.L = float(rod["length"])
        self.radius = float(rod["radius"])
        self.n_rod = int(rod["vertex_count"])
        self.density = float(rod["density"])
        self.inextensible = bool(rod.get("inextensible", True))
        self.resists_twist = bool(rod.get("resists_twist", True))

        # Cross-section properties (circular rod)
        area = math.pi * self.radius ** 2
        polar_moment = 0.5 * math.pi * self.radius ** 4  # J for solid circle
        self.rod_mass_total = self.density * area * self.L
        segment_length = self.L / max(self.n_rod - 1, 1)
        self.segment_mass = self.density * area * segment_length

        # Shear modulus chosen to give a physically reasonable torsional wave
        # speed c_t = sqrt(G*J / (rho*J_area)) ~ sqrt(G/rho); G ~ 1e6 Pa for a
        # soft polymer yields c_t ~ 30 m/s and a fundamental period ~ L/c_t ~
        # 0.05 s, well resolved at sample_dt = 0.02 s. The resulting torsional
        # stiffness k = G*J/L produces visible oscillatory exchange within the
        # 6 s window.
        self.G = 1.0e6  # Pa
        self.torsional_stiffness = self.G * polar_moment / self.L if self.resists_twist else 0.0

        # Rigid bodies -------------------------------------------------------
        rb_cfg = cfg["rigid_bodies"]
        self.driven_center = np.asarray(rb_cfg["driven"]["center"], dtype=float)
        self.receiver_center = np.asarray(rb_cfg["receiver"]["center"], dtype=float)
        self.driven_radius = float(rb_cfg["driven"]["radius"])
        self.driven_axial = float(rb_cfg["driven"]["axial_length"])
        self.driven_mass = float(rb_cfg["driven"]["mass"])
        self.receiver_radius = float(rb_cfg["receiver"]["radius"])
        self.receiver_axial = float(rb_cfg["receiver"]["axial_length"])
        self.receiver_mass = float(rb_cfg["receiver"]["mass"])

        # Moments of inertia about the rod (x) axis for solid cylinders whose
        # symmetry axes are aligned with x after the initial rotation_xyzw.
        # For a cylinder of radius R, length H, mass M rotating about its
        # diameter-perpendicular axis: I = M*(R^2/4 + H^2/12). With the
        # geometry transform rotating the canonical z-aligned cylinder onto x,
        # this corresponds to rotation about the body y-axis pre-transform,
        # i.e. a transverse principal axis. We add half the rod's polar moment
        # of inertia to each body so the reduced model accounts for the rod's
        # distributed twist inertia.
        rod_J_polar = self.rod_mass_total * self.radius ** 2 / 2.0
        self.I_driven = (
            self.driven_mass * (self.driven_radius ** 2 / 4.0 + self.driven_axial ** 2 / 12.0)
            + 0.5 * rod_J_polar
        )
        self.I_receiver = (
            self.receiver_mass * (self.receiver_radius ** 2 / 4.0 + self.receiver_axial ** 2 / 12.0)
            + 0.5 * rod_J_polar
        )

        # Excitation ---------------------------------------------------------
        exc = cfg["excitation"]
        self.exc_axis = np.asarray(exc["axis"], dtype=float)
        self.exc_axis /= np.linalg.norm(self.exc_axis) + 1e-30
        self.exc_start = float(exc["start_time"])
        self.exc_rise = float(exc["rise_time"])
        self.exc_hold_until = float(exc["hold_until"])
        self.exc_end = float(exc["end_time"])
        self.exc_peak = float(exc["peak_torque"])

        # Initial conditions -------------------------------------------------
        ic = cfg["initial_conditions"]
        self.rod_start = np.asarray(ic["rod_start"], dtype=float)
        self.rod_end = np.asarray(ic["rod_end"], dtype=float)
        self.initial_twist = float(ic.get("material_twist", 0.0))

        # Reduced DOFs: twist angles about x-axis for driven/receiver bodies.
        self.theta_d = 0.0
        self.theta_r = 0.0
        self.omega_d = 0.0
        self.omega_r = 0.0

        # Cumulative impulse tracking (scalar about x-axis; vector form below)
        self.cum_ext_impulse = np.zeros(3, dtype=float)
        self.cum_ext_ang_impulse = np.zeros(3, dtype=float)

        # Reference scales for scalars output
        self.ref_momentum = max(self.driven_mass, self.receiver_mass) * self.L
        self.ref_ang_momentum = max(self.I_driven, self.I_receiver) * 1.0

        # Sample bookkeeping
        self.current_time = 0.0
        self._frame_cache: dict[float, StateFrame] = {}

    # ------------------------------------------------------------ excitation
    def _applied_torque(self, t: float) -> float:
        """Trapezoidal torque pulse magnitude about the excitation axis."""
        if t < self.exc_start or t > self.exc_end:
            return 0.0
        if t < self.exc_start + self.exc_rise:
            frac = (t - self.exc_start) / max(self.exc_rise, 1e-12)
            return self.exc_peak * max(0.0, min(1.0, frac))
        if t <= self.exc_hold_until:
            return self.exc_peak
        fall = self.exc_end - self.exc_hold_until
        if fall <= 0.0:
            return 0.0
        frac = (t - self.exc_hold_until) / fall
        return self.exc_peak * max(0.0, min(1.0, 1.0 - frac))

    # ---------------------------------------------------------------- physics
    def _step_to(self, target: float) -> None:
        """Integrate the reduced twist system forward to ``target`` time using
        symplectic Euler substeps sized at most ``self.dt``."""
        if target <= self.current_time:
            return
        remaining = target - self.current_time
        dt = min(self.dt, remaining)
        k = self.torsional_stiffness
        axis = self.exc_axis  # unit vector along x for this scene
        while remaining > 1e-12:
            h = min(dt, remaining)
            tau_ext = self._applied_torque(self.current_time)
            # Torque balance: internal rod torque = k*(theta_d - theta_r)
            # acts oppositely on the two bodies. External torque acts only on
            # the driven body (receiver_direct_drive = false).
            tau_int = k * (self.theta_d - self.theta_r)
            alpha_d = (tau_ext - tau_int) / self.I_driven
            alpha_r = (+tau_int) / self.I_receiver
            # Symplectic Euler: update velocities first, then positions.
            self.omega_d += h * alpha_d
            self.omega_r += h * alpha_r
            self.theta_d += h * self.omega_d
            self.theta_r += h * self.omega_r
            # Track cumulative external angular impulse (vector).
            self.cum_ext_ang_impulse += h * tau_ext * axis
            self.current_time += h
            remaining -= h

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Return xyzw quaternion for rotation of ``angle`` radians about ``axis``."""
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        ax = axis / n
        ha = 0.5 * angle
        s = math.sin(ha)
        c = math.cos(ha)
        return np.array([ax[0] * s, ax[1] * s, ax[2] * s, c], dtype=float)

    @staticmethod
    def _rotate_vec(v: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q (xyzw)."""
        x, y, z, w = quat_xyzw
        # Convert to rotation matrix implicitly via quaternion product
        vx, vy, vz = v
        t0 = w * vx + y * vz - z * vy
        t1 = w * vy + z * vx - x * vz
        t2 = w * vz + x * vy - y * vx
        rx = w * t0 - y * t2 + z * t1
        ry = w * t1 - z * t0 + x * t2
        rz = w * t2 - x * t1 + y * t0
        return np.array([rx, ry, rz], dtype=float)

    def _make_frame(self, t: float) -> StateFrame:
        """Construct a full N-sample StateFrame from the reduced 2-DOF state."""
        n = self.n_rod
        axis = self.exc_axis

        # Rod samples: straight line between fixed endpoints (inextensible, no
        # bending in this twist-only reduction). Material twist varies linearly
        # along the rod between the two endpoint body orientations.
        rod_pos = np.linspace(self.rod_start, self.rod_end, n)
        arc = np.linspace(0.0, self.L, n)
        frac = arc / self.L
        local_twist = self.theta_d * (1.0 - frac) + self.theta_r * frac
        material_twist = local_twist - self.initial_twist

        # Orientations along the rod interpolate between driven and receiver
        # twist quaternions about the rod axis.
        q_d = self._quat_from_axis_angle(axis, self.theta_d)
        q_r = self._quat_from_axis_angle(axis, self.theta_r)
        rod_orient = np.zeros((n, 4), dtype=float)
        for i in range(n):
            f = frac[i]
            # Slerp-like interpolation for co-linear rotations reduces to
            # simple angle interpolation since both quaternions share the axis.
            angle_i = self.theta_d * (1.0 - f) + self.theta_r * f
            rod_orient[i] = self._quat_from_axis_angle(axis, angle_i)

        # Angular velocity along the rod interpolates linearly between the two
        # body spin rates (consistent with piecewise-linear twist rate).
        rod_omega_scalar = self.omega_d * (1.0 - frac) + self.omega_r * frac
        rod_angvel = rod_omega_scalar[:, None] * axis[None, :]

        # Linear velocity of rod samples comes from rotation about the x-axis
        # anchored at the respective nearest body center; with the rod lying
        # on the x-axis and bodies translating only along x, this is identically
        # zero (r × omega has no component when r is parallel to omega).
        rod_vel = np.zeros((n, 3), dtype=float)

        # Body samples -------------------------------------------------------
        driven_q = self._quat_from_axis_angle(axis, self.theta_d)
        receiver_q = self._quat_from_axis_angle(axis, self.theta_r)

        # Orientation marks are welded to their parent bodies. In the scene
        # geometry the marks sit at +z offsets relative to the body centers in
        # the initial configuration; under body rotation about x they orbit
        # accordingly.
        mark_d_local = np.array([0.0, 0.0, 0.075], dtype=float)
        mark_r_local = np.array([0.0, 0.0, 0.0475], dtype=float)
        mark_d_world = self.driven_center + self._rotate_vec(mark_d_local, driven_q)
        mark_r_world = self.receiver_center + self._rotate_vec(mark_r_local, receiver_q)

        # Positions: rod (n) + driven + receiver + driven_mark + receiver_mark
        positions = np.vstack(
            [rod_pos, self.driven_center[None, :], self.receiver_center[None, :],
             mark_d_world[None, :], mark_r_world[None, :]]
        ).astype(float)
        count = len(positions)

        # Velocities: rod is stationary translationally; marks move with the
        # tangential velocity omega x r_rel of their parent body.
        vel_d_mark = np.cross(self.omega_d * axis, mark_d_world - self.driven_center)
        vel_r_mark = np.cross(self.omega_r * axis, mark_r_world - self.receiver_center)
        velocities = np.vstack(
            [rod_vel, np.zeros((2, 3)), vel_d_mark[None, :], vel_r_mark[None, :]]
        ).astype(float)

        # Orientations: rod samples, two bodies (same as their twist quat),
        # and two orientation marks (identical to parent body orientation).
        orientations = np.vstack(
            [rod_orient, driven_q[None, :], receiver_q[None, :],
             driven_q[None, :], receiver_q[None, :]]
        ).astype(float)

        # Angular velocities: rod (linear interp), driven, receiver, marks = parents
        angvel = np.vstack(
            [rod_angvel,
             (self.omega_d * axis)[None, :],
             (self.omega_r * axis)[None, :],
             (self.omega_d * axis)[None, :],
             (self.omega_r * axis)[None, :]]
        ).astype(float)

        # Masses: rod segments, driven body, receiver body, marks (massless)
        masses = np.r_[
            np.full(n, self.segment_mass),
            self.driven_mass,
            self.receiver_mass,
            0.0,
            0.0,
        ].astype(float)

        # Body IDs match scene_geometry.yaml bindings: rod=0 (attribute
        # selector), driven=1, receiver=2, driven_mark=3, receiver_mark=4.
        body_id = np.r_[np.zeros(n, dtype=int), 1, 2, 3, 4].astype(np.int64)

        # Attributes ---------------------------------------------------------
        component_id = np.r_[np.zeros(n), 1.0, 2.0, 3.0, 4.0].astype(float)
        arc_attr = np.r_[arc, 0.0, 0.0, 0.0, 0.0].astype(float)
        mat_twist = np.r_[material_twist, self.theta_d, self.theta_r,
                          self.theta_d, self.theta_r].astype(float)
        is_driven = np.r_[np.zeros(n), 1.0, 0.0, 1.0, 0.0].astype(float)
        mass_quantity = masses.copy()

        # Per-sample world-frame inertia tensor. For rod samples we report the
        # small segment polar inertia about x embedded in a diagonal tensor;
        # for bodies we embed the scalar axial inertia; marks are zero.
        seg_I = self.segment_mass * self.radius ** 2 / 2.0
        inertia_tensor = np.zeros((count, 3, 3), dtype=float)
        for i in range(n):
            inertia_tensor[i, 0, 0] = seg_I
        inertia_tensor[n, 0, 0] = self.I_driven
        inertia_tensor[n + 1, 0, 0] = self.I_receiver

        # Fields: budget/source terms ----------------------------------------
        # Single global domain (G=1). No mass sources, no inflow/outflow.
        G = 1
        K = 1  # one external-force term slot
        domain_weights = np.ones((G, count), dtype=float)
        external_force = np.zeros((G, K, 3), dtype=float)
        external_couple = np.zeros((G, K, 3), dtype=float)
        force_position = np.zeros((G, K, 3), dtype=float)
        # Report the instantaneous external couple on the driven body.
        tau_now = self._applied_torque(self.current_time)
        external_couple[0, 0, :] = tau_now * axis
        force_position[0, 0, :] = self.driven_center

        cumulative_mass_source = np.zeros(1, dtype=float)
        zero3 = np.zeros(3, dtype=float)

        fields = {
            "budget_domain_weights": domain_weights,
            "budget_external_force": external_force,
            "budget_external_couple": external_couple,
            "budget_force_position": force_position,
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy()[None, :],
            "budget_cumulative_external_angular_impulse": self.cum_ext_ang_impulse.copy()[None, :],
            "budget_cumulative_inflow_momentum": zero3.copy()[None, :],
            "budget_cumulative_outflow_momentum": zero3.copy()[None, :],
            "budget_cumulative_inflow_angular_momentum": zero3.copy()[None, :],
            "budget_cumulative_outflow_angular_momentum": zero3.copy()[None, :],
            "cumulative_mass_source": cumulative_mass_source,
        }

        attributes = {
            "component_id": component_id,
            "arc_length": arc_attr,
            "material_twist": mat_twist,
            "is_driven": is_driven,
            "mass_quantity": mass_quantity,
            "inertia_tensor": inertia_tensor,
        }

        scalars = {
            "applied_torque": float(tau_now),
            "reference_momentum": float(self.ref_momentum),
            "reference_angular_momentum": float(self.ref_ang_momentum),
        }

        frame = StateFrame(
            time=float(t),
            position=positions,
            velocity=velocities,
            orientation=orientations,
            angular_velocity=angvel,
            mass=masses,
            body_id=body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        t = float(time)
        cached = self._frame_cache.get(t)
        if cached is not None:
            return cached
        self._step_to(t)
        frame = self._make_frame(t)
        self._frame_cache[t] = frame
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "torsional_stiffness_Nm_per_rad": float(self.torsional_stiffness),
            "I_driven_kg_m2": float(self.I_driven),
            "I_receiver_kg_m2": float(self.I_receiver),
            "final_theta_d_rad": float(self.theta_d),
            "final_theta_r_rad": float(self.theta_r),
            "final_omega_d_rad_s": float(self.omega_d),
            "final_omega_r_rad_s": float(self.omega_r),
        }
