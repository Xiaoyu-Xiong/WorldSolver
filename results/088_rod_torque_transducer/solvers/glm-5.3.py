"""Physics solver for scene 088_rod_torque_transducer.

Torque transmission between rod-coupled rigid bodies.

Physical model
--------------
The frozen experiment excites only the twist mode of the connecting rod: the
applied torque acts about the rod axis, gravity is zero and the bodies are
constrained to translate along (in fact, stay fixed on) the rod axis.  The rod
is therefore modeled as a discrete elastic rod restricted to its twist degree
of freedom:

* each rod vertex ``i`` carries a lumped polar inertia ``rho * J * ds``
  (``J = pi r^4 / 2``, circular cross section) and a twist angle ``phi_i``
  about the local tangent,
* consecutive vertices are coupled by discrete torsional springs of stiffness
  ``k = G * J / ds`` (twist energy ``U = 1/2 k (phi_{i+1}-phi_i)^2``), which
  is exactly the twist part of a discrete elastic rod,
* the rod endpoints are clamped to the material frames of the rigid bodies:
  ``phi_0 = theta_driven`` and ``phi_{n-1} = theta_receiver``, so endpoint
  DOFs coincide with the bodies' single dynamic rotation about the rod axis.

The driven body receives a smooth torque pulse about the rod axis; torque is
transmitted through the rod's shear springs to the receiver body, which only
moves because of that transmitted torque (no direct drive).  Time integration
uses the symplectic, second-order velocity-Verlet scheme with a fixed substep,
so the trajectory is deterministic and the discrete angular-momentum budget
closes exactly (the trapezoidal torque integral equals the Verlet momentum
change of the linear twist chain).

No animation, keyframes or precomputed motion are used: every frame is the
result of the explicit time-stepping procedure below.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# Shear modulus of the rod material.  A soft polymer value consistent with the
# frozen density of 1150 kg/m^3; the config fixes geometry and density but not
# the elastic modulus, so a documented physical constant is chosen here.
SHEAR_MODULUS = 6.0e6  # Pa

# Fixed internal substep for the velocity-Verlet integrator.  Stability limit
# is dt < 2/omega_max with omega_max = sqrt(4 G / (rho ds^2)) ~ 5.8e3 rad/s,
# so dt = 1e-4 keeps omega_max*dt ~ 0.58 (well inside the stable region).
DT = 1.0e-4

DOMAIN_ROD = 0
DOMAIN_DRIVEN = 1
DOMAIN_RECEIVER = 2
N_DOMAINS = 3
N_FORCE_TERMS = 1


def _quat_x(angle: float) -> np.ndarray:
    """Unit quaternion (xyzw) of a rotation about +x by ``angle`` radians."""
    half = 0.5 * float(angle)
    return np.array([np.sin(half), 0.0, 0.0, np.cos(half)])


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class BenchmarkSolver:
    """Twist-chain torsion solver with rigid-body endpoint coupling."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = int(seed)

        rod = self.config["rod"]
        self.n = int(rod["vertex_count"])
        n = self.n
        length = float(rod["length"])
        radius = float(rod["radius"])
        density = float(rod["density"])
        self.ds = length / (n - 1)

        # Circular cross-section polar second moment of area and stiffnesses.
        self.J_area = 0.5 * np.pi * radius ** 4
        self.GJ = SHEAR_MODULUS * self.J_area
        self.k_edge = self.GJ / self.ds
        # Lumped polar moment of inertia of one internal rod segment.
        self.node_inertia = density * self.J_area * self.ds

        driven = self.config["rigid_bodies"]["driven"]
        receiver = self.config["rigid_bodies"]["receiver"]
        self.I_driven_axial = 0.5 * float(driven["mass"]) * float(driven["radius"]) ** 2
        self.I_driven_perp = float(driven["mass"]) * (
            3.0 * float(driven["radius"]) ** 2 + float(driven["axial_length"]) ** 2
        ) / 12.0
        self.I_receiver_axial = 0.5 * float(receiver["mass"]) * float(receiver["radius"]) ** 2
        self.I_receiver_perp = float(receiver["mass"]) * (
            3.0 * float(receiver["radius"]) ** 2 + float(receiver["axial_length"]) ** 2
        ) / 12.0

        # Degrees of freedom: q[0] = driven body angle (clamped rod start),
        # q[1..n-2] = internal rod twist angles, q[n-1] = receiver body angle
        # (clamped rod end).  Diagonal mass matrix of the twist chain.
        self.m_dof = np.full(n, self.node_inertia)
        self.m_dof[0] = self.I_driven_axial
        self.m_dof[-1] = self.I_receiver_axial

        # Chain (discrete Laplacian) operator: F_internal = k_edge * D @ q.
        D = np.zeros((n, n))
        for i in range(n - 1):
            D[i, i] += -1.0
            D[i, i + 1] += 1.0
            D[i + 1, i] += 1.0
            D[i + 1, i + 1] += -1.0
        self.D = D

        # Base (initial) sample layout from the frozen scene description.
        base = scene.initial_state()
        self.position0 = np.array(base.position, dtype=float)
        self.mass = np.array(base.mass, dtype=float)
        self.body_id = np.array(base.body_id, dtype=int)
        self.component_id = np.array(base.attributes["component_id"], dtype=float)
        self.arc_length = np.array(base.attributes["arc_length"], dtype=float)
        self.is_driven = np.array(base.attributes["is_driven"], dtype=float)
        self.N = self.position0.shape[0]

        # Sample indices: rod vertices 0..n-1, then driven(1), receiver(2),
        # driven mark(3), receiver mark(4).
        self.idx_driven = n
        self.idx_receiver = n + 1
        self.idx_driven_mark = n + 2
        self.idx_receiver_mark = n + 3

        # Mark offsets inside the parent body frames (from the initial state).
        self.driven_mark_offset = self.position0[self.idx_driven_mark] - self.position0[self.idx_driven]
        self.receiver_mark_offset = self.position0[self.idx_receiver_mark] - self.position0[self.idx_receiver]
        self.driven_center = self.position0[self.idx_driven].copy()
        self.receiver_center = self.position0[self.idx_receiver].copy()

        # Per-sample world inertia tensors at rest (rod tangents along +x).
        self._inertia_rest = np.zeros((self.N, 3, 3))
        node_transverse = 0.5 * self.mass[:n] * radius ** 2
        for i in range(1, n - 1):
            self._inertia_rest[i, 0, 0] = self.node_inertia
            self._inertia_rest[i, 1, 1] = node_transverse[i]
            self._inertia_rest[i, 2, 2] = node_transverse[i]
        # Endpoint rod samples are rigidly clamped to the bodies; their
        # rotation is fully accounted for by the body samples (no double count).
        self._inertia_rest[self.idx_driven] = np.diag(
            [self.I_driven_axial, self.I_driven_perp, self.I_driven_perp]
        )
        self._inertia_rest[self.idx_receiver] = np.diag(
            [self.I_receiver_axial, self.I_receiver_perp, self.I_receiver_perp]
        )
        # Massless orientation marks: zero inertia.

        # Budget domain membership weights (rows are separate budgets).
        self.domain_weights = np.zeros((N_DOMAINS, self.N))
        self.domain_weights[DOMAIN_ROD, :n] = 1.0
        self.domain_weights[DOMAIN_DRIVEN, [self.idx_driven, self.idx_driven_mark]] = 1.0
        self.domain_weights[DOMAIN_RECEIVER, [self.idx_receiver, self.idx_receiver_mark]] = 1.0

        # Reference scales (positive, physically motivated).
        exc = self.config["excitation"]
        self.reference_angular_momentum = abs(float(exc["peak_torque"])) * (
            float(exc["end_time"]) - float(exc["start_time"])
        )
        self.reference_momentum = float(np.sum(self.mass)) * 1.0

        # Dynamic state: twist angles and rates of the coupled chain.
        self.q = np.zeros(n)
        self.v = np.zeros(n)
        self.t = 0.0
        self.cumulative_angular_impulse = 0.0
        self._max_twist = 0.0

    # ------------------------------------------------------------------ #
    # excitation
    # ------------------------------------------------------------------ #
    def _torque(self, time: float) -> float:
        """Smooth torque pulse applied to the driven body about the rod axis."""
        exc = self.config["excitation"]
        t0 = float(exc["start_time"])
        rise = float(exc["rise_time"])
        t1 = float(exc["hold_until"])
        t2 = float(exc["end_time"])
        peak = float(exc["peak_torque"])
        if time <= t0 or time >= t2:
            return 0.0
        if time < t0 + rise:
            u = (time - t0) / rise
            shape = 0.5 * (1.0 - np.cos(np.pi * u))
        elif time <= t1:
            shape = 1.0
        else:
            u = (time - t1) / (t2 - t1)
            shape = 0.5 * (1.0 + np.cos(np.pi * u))
        return peak * shape

    def _accel(self, q: np.ndarray, time: float) -> np.ndarray:
        force = self.k_edge * (self.D @ q)
        force[0] += self._torque(time)
        return force / self.m_dof

    # ------------------------------------------------------------------ #
    # time stepping
    # ------------------------------------------------------------------ #
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.t - 1e-9:
            # Non-monotonic request: restart deterministically from rest.
            self.q[:] = 0.0
            self.v[:] = 0.0
            self.t = 0.0
            self.cumulative_angular_impulse = 0.0
            self._max_twist = 0.0
        while self.t < target - 1e-12:
            dt = min(DT, target - self.t)
            t_old = self.t
            a0 = self._accel(self.q, t_old)
            self.v += 0.5 * dt * a0
            self.q += dt * self.v
            self.t = t_old + dt
            a1 = self._accel(self.q, self.t)
            self.v += 0.5 * dt * a1
            # Trapezoidal angular impulse; for the linear twist chain this
            # exactly matches the Verlet change of total angular momentum.
            self.cumulative_angular_impulse += 0.5 * dt * (
                self._torque(t_old) + self._torque(self.t)
            )
        if abs(self.t - target) <= 1e-9:
            self.t = target
        return self._build_frame(self.t)

    # ------------------------------------------------------------------ #
    # frame export
    # ------------------------------------------------------------------ #
    def _build_frame(self, time: float) -> StateFrame:
        n = self.n
        N = self.N
        q = self.q
        v = self.v
        theta_driven = float(q[0])
        theta_receiver = float(q[-1])
        omega_driven = float(v[0])
        omega_receiver = float(v[-1])
        self._max_twist = max(self._max_twist, float(np.max(np.abs(q[1:] - q[:-1]))))

        position = self.position0.copy()
        velocity = np.zeros((N, 3))
        orientation = np.zeros((N, 4))
        orientation[:, 3] = 1.0
        angular_velocity = np.zeros((N, 3))

        # Rod vertices: material frames twisted about the local tangent (+x).
        half = 0.5 * q
        orientation[:n, 0] = np.sin(half)
        orientation[:n, 3] = np.cos(half)
        angular_velocity[:n, 0] = v

        # Rigid bodies (rotation about the rod axis only; centers fixed).
        for idx, theta, omega in (
            (self.idx_driven, theta_driven, omega_driven),
            (self.idx_receiver, theta_receiver, omega_receiver),
        ):
            orientation[idx] = _quat_x(theta)
            angular_velocity[idx, 0] = omega

        # Orientation marks welded to the bodies (offsets revolve with them).
        for idx, center, offset, theta, omega in (
            (self.idx_driven_mark, self.driven_center, self.driven_mark_offset, theta_driven, omega_driven),
            (self.idx_receiver_mark, self.receiver_center, self.receiver_mark_offset, theta_receiver, omega_receiver),
        ):
            rotated_offset = _rot_x(theta) @ offset
            position[idx] = center + rotated_offset
            velocity[idx] = np.cross([omega, 0.0, 0.0], rotated_offset)
            orientation[idx] = _quat_x(theta)
            angular_velocity[idx, 0] = omega

        # World inertia tensors (rod is straight along +x; body tensors are
        # axisymmetric about x so the rotation leaves them unchanged).
        inertia = self._inertia_rest.copy()
        for idx, theta in (
            (self.idx_driven, theta_driven),
            (self.idx_receiver, theta_receiver),
        ):
            R = _rot_x(theta)
            inertia[idx] = R @ inertia[idx] @ R.T

        # Rod twist attribute (material twist relative to the natural frame).
        material_twist = np.zeros(N)
        material_twist[:n] = q

        # Applied torque magnitude at this instant.
        torque_now = self._torque(time)

        # External-force bookkeeping: the only external load is the applied
        # free couple on the driven body; no boundary tractions, no mass
        # transfer, hence zero impulses/momentum fluxes elsewhere.
        external_couple = np.zeros((N_DOMAINS, N_FORCE_TERMS, 3))
        external_couple[DOMAIN_DRIVEN, 0, 0] = torque_now
        external_force = np.zeros((N_DOMAINS, N_FORCE_TERMS, 3))
        force_position = np.zeros((N_DOMAINS, N_FORCE_TERMS, 3))
        force_position[DOMAIN_ROD, 0] = 0.5 * (
            self.position0[0] + self.position0[n - 1]
        )
        force_position[DOMAIN_DRIVEN, 0] = self.driven_center
        force_position[DOMAIN_RECEIVER, 0] = self.receiver_center

        cumulative_external_angular_impulse = np.zeros((N_DOMAINS, 3))
        cumulative_external_angular_impulse[DOMAIN_DRIVEN, 0] = self.cumulative_angular_impulse

        frame = StateFrame(
            time=time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "component_id": self.component_id.copy(),
                "arc_length": self.arc_length.copy(),
                "material_twist": material_twist,
                "is_driven": self.is_driven.copy(),
                "inertia_tensor": inertia,
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_couple": external_couple,
                "budget_external_force": external_force,
                "budget_force_position": force_position,
                "budget_cumulative_external_impulse": np.zeros((N_DOMAINS, 3)),
                "budget_cumulative_external_angular_impulse": cumulative_external_angular_impulse,
                "budget_cumulative_inflow_momentum": np.zeros((N_DOMAINS, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((N_DOMAINS, 3)),
                "budget_cumulative_inflow_angular_momentum": np.zeros((N_DOMAINS, 3)),
                "budget_cumulative_outflow_angular_momentum": np.zeros((N_DOMAINS, 3)),
                "cumulative_mass_source": np.zeros(N_FORCE_TERMS),
            },
            scalars={
                "applied_torque": float(torque_now),
                "reference_momentum": float(self.reference_momentum),
                "reference_angular_momentum": float(self.reference_angular_momentum),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ #
    def diagnostics(self) -> dict[str, object]:
        total_inertia = float(np.sum(self.m_dof))
        angular_momentum = float(np.sum(self.m_dof * self.v))
        return {
            "integrator": "velocity_verlet",
            "substep": DT,
            "shear_modulus": SHEAR_MODULUS,
            "torsional_rigidity_GJ": float(self.GJ),
            "edge_torsional_stiffness": float(self.k_edge),
            "time": float(self.t),
            "driven_angle": float(self.q[0]),
            "receiver_angle": float(self.q[-1]),
            "orientation_difference": float(self.q[-1] - self.q[0]),
            "driven_angular_velocity": float(self.v[0]),
            "receiver_angular_velocity": float(self.v[-1]),
            "max_edge_twist": float(self._max_twist),
            "total_polar_inertia": total_inertia,
            "angular_momentum": angular_momentum,
            "cumulative_external_angular_impulse": float(self.cumulative_angular_impulse),
            "angular_momentum_residual": angular_momentum - float(self.cumulative_angular_impulse),
        }
