"""Physically-based quasi-static solver for the clamped-free elastic rod cantilever.

Model
-----
The rod is a homogeneous, naturally straight, *inextensible* Kirchhoff elastic rod
with a circular cross-section, clamped at its root (position and horizontal
tangent) and free at its tip, loaded by uniform gravity that is ramped according
to the frozen ``load_schedule``.  Loading, reference geometry and clamp all lie in
one vertical plane, so the equilibrium is planar and the configuration is fully
described by the tangent (turning) angle ``theta_i`` at every centreline node.

Discretisation (discrete elastica, midpoint rule):
  * nodes ``i = 0 .. N-1`` equally spaced in arc length, ``ds = L/(N-1)``;
  * segment direction ``phi_j = (theta_j + theta_{j+1})/2``, so each segment keeps
    exactly the reference length ``ds`` -> inextensibility holds to machine
    precision (reported through ``fields.constraint_value``/``constraint_target``);
  * positions ``p_{i+1} = p_i + ds (cos phi_i * e_x + sin phi_i * e_z)`` with
    ``e_x`` the clamp tangent and ``(e_x, e_y, e_z)`` a right-handed in-plane frame;
  * bending energy ``U_b = sum_j EI/(2 ds) (theta_{j+1}-theta_j)^2``, the standard
    second-order accurate discretisation of ``int 1/2 EI kappa^2 ds``;
  * gravitational potential ``V = -lam(t) sum_i m_i (g . p_i)``.

Solution procedure
------------------
At every requested sample time the load factor ``lam`` follows the frozen gravity
ramp and the equilibrium angles solve

    grad_theta [ U_b(theta) + V(theta, lam) ] = 0,    theta_0 = 0 (clamped tangent)

using a damped Newton iteration with the analytic gradient and analytic
tridiagonal Hessian, an energy line search and a bending-preconditioned descent
safeguard.  Successive samples warm start from the previous equilibrium, i.e. the
load ramp is followed by quasi-static continuation, as required by
``simulation_mode: quasi_static_equilibria``.  Nodal velocities are the physical
rates of that evolving equilibrium (no scripted motion is used anywhere).

The exported momentum budget terms (gravity resultant at the centre of mass, clamp
reaction force at the root, clamp reaction couple) satisfy, for the exported
kinematics,

    sum F_ext = dP/dt        and        sum (r x F_ext) + sum C_ext = dL/dt.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Quasi-static discrete-elastica solver for the gravity-loaded cantilever."""

    #: Frozen material choice for the ``rod_elastomer`` rod (Pa).  The public
    #: configuration fixes geometry, density and gravity but not the modulus; a
    #: semi-flexible polymer value is used so that self weight produces a clearly
    #: resolved smooth large-deflection cantilever sag.
    YOUNG_MODULUS = 1.9e8

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(scene.config)
        rod = config["rod"]
        boundary = config["boundary_conditions"]

        self.gravity = np.asarray(config["gravity"], dtype=float).reshape(3)
        self.g_mag = float(np.linalg.norm(self.gravity))
        if self.g_mag <= 0.0:
            self.gravity = np.array([0.0, 0.0, -9.81])
            self.g_mag = float(np.linalg.norm(self.gravity))

        self.length = float(rod["length"])
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        self.n = int(getattr(scene, "vertex_count", 65))
        self.n_edges = self.n - 1
        self.nu = self.n - 1  # unknown angles theta_1 .. theta_{N-1}
        self.ds = self.length / self.n_edges
        self.area = float(np.pi) * self.radius ** 2
        self.second_moment = float(np.pi) * self.radius ** 4 / 4.0
        self.ei = self.YOUNG_MODULUS * self.second_moment
        self.stiffness = self.ei / self.ds

        # ---- load schedule (gravity ramp) ---------------------------------
        schedule = config.get("load_schedule") or {}
        self.ramp_start = float(schedule.get("start_time", 0.0))
        self.ramp_full = float(schedule.get("full_load_time", self.ramp_start + 1.0))
        if self.ramp_full <= self.ramp_start:
            self.ramp_full = self.ramp_start + 1.0
        self.hold_until = float(
            schedule.get("hold_until", config.get("duration", self.ramp_full))
        )

        # ---- clamp frame ---------------------------------------------------
        root = boundary["root"]
        self.root_position = np.asarray(root["position"], dtype=float).reshape(3)
        tangent = np.asarray(root["tangent"], dtype=float).reshape(3)
        tangent = tangent / max(float(np.linalg.norm(tangent)), 1e-15)
        self.axis_x = tangent
        g_dir = self.gravity / self.g_mag
        axis_y = np.cross(g_dir, self.axis_x)
        norm_y = float(np.linalg.norm(axis_y))
        if norm_y < 1e-9:  # degenerate: clamp tangent parallel to gravity
            trial = np.array([0.0, 1.0, 0.0])
            if abs(float(np.dot(trial, self.axis_x))) > 0.9:
                trial = np.array([1.0, 0.0, 0.0])
            axis_y = np.cross(trial, self.axis_x)
            norm_y = float(np.linalg.norm(axis_y))
        self.axis_y = axis_y / norm_y
        axis_z = np.cross(self.axis_x, self.axis_y)
        self.axis_z = axis_z / max(float(np.linalg.norm(axis_z)), 1e-15)
        self.g_axial = float(self.gravity @ self.axis_x)
        self.g_normal = float(self.gravity @ self.axis_z)

        # ---- reference state inherited from the frozen scene ---------------
        frame0 = scene.initial_state()
        self.mass = np.asarray(frame0.mass, dtype=float).reshape(self.n).copy()
        self.total_mass = float(self.mass.sum())
        attributes0 = {k: np.asarray(v, dtype=float) for k, v in dict(frame0.attributes).items()}
        self.arc_length = attributes0.get("arc_length", np.linspace(0.0, self.length, self.n)).copy()
        self.is_root = attributes0.get("is_clamped_root", np.r_[1.0, np.zeros(self.n - 1)]).copy()

        # tail masses T[k] = sum_{i >= k+1} m_i  (T[N-1] = 0)
        self.mass_tail = np.zeros(self.n)
        self.mass_tail[: self.n - 1] = np.cumsum(self.mass[:0:-1])[::-1]

        # constant bending Hessian over the unknown angles
        hb = np.zeros((self.nu, self.nu))
        for k in range(self.nu):
            node = k + 1
            diag = 1.0 + (1.0 if node <= self.n - 2 else 0.0)
            hb[k, k] = self.stiffness * diag
            if node + 1 <= self.n - 1:
                hb[k, k + 1] = -self.stiffness
            if node - 1 >= 1:
                hb[k, k - 1] = -self.stiffness
        self.h_bend = hb

        # ---- solver state --------------------------------------------------
        self.unk = np.zeros(self.nu)  # theta_1 .. theta_{N-1}; theta_0 = 0 (clamped)
        self.theta = self._theta_full(self.unk)
        self.position = self._positions(self.theta)
        self.orientation = self._orientations(self.theta)

        self.prev_time: float | None = None
        self.prev_position = self.position.copy()
        self.prev_theta = self.theta.copy()
        self.prev_linear_momentum = np.zeros(3)
        self.prev_angular_momentum = np.zeros(3)

        # frozen positive scales
        self.weight = self.total_mass * self.g_mag
        self.reference_force = float(max(self.weight, 1e-12))
        self.reference_torque = float(max(0.5 * self.weight * self.length, 1e-12))

        self.segment_reference = np.full(self.n_edges, self.ds)
        self.cumulative_mass_source = np.zeros(1)

        self.frame_index = 0
        self.newton_iterations = 0
        self.max_residual = 0.0
        self.solves = 0
        self.last_tip_deflection = 0.0

    # ------------------------------------------------------------- kinematics
    def _theta_full(self, unk: np.ndarray) -> np.ndarray:
        return np.concatenate((np.zeros(1), unk))

    def _positions(self, theta: np.ndarray) -> np.ndarray:
        phi = 0.5 * (theta[: self.n - 1] + theta[1:])
        local_x = self.ds * np.cos(phi)
        local_z = self.ds * np.sin(phi)
        coord_x = np.concatenate(([0.0], np.cumsum(local_x)))
        coord_z = np.concatenate(([0.0], np.cumsum(local_z)))
        return (
            self.root_position[None, :]
            + coord_x[:, None] * self.axis_x[None, :]
            + coord_z[:, None] * self.axis_z[None, :]
        )

    def _orientations(self, theta: np.ndarray) -> np.ndarray:
        # tangent t = cos(theta) e_x + sin(theta) e_z is a rotation of -theta
        # about e_y in the right-handed frame (e_x, e_y, e_z)
        half = 0.5 * theta
        quat = np.zeros((self.n, 4))
        quat[:, 0:3] = -np.sin(half)[:, None] * self.axis_y[None, :]
        quat[:, 3] = np.cos(half)
        return quat

    # ------------------------------------------------------------ energetics
    def _energy(self, theta: np.ndarray, lam: float) -> float:
        d = np.diff(theta)
        bend = 0.5 * self.stiffness * float(d @ d)
        pos = self._positions(theta)
        grav = -lam * float(self.mass @ (pos @ self.gravity))
        return bend + grav

    def _grad_hess(self, theta: np.ndarray, lam: float):
        """Analytic gradient and Hessian of U_b + V wrt theta_1 .. theta_{N-1}."""
        n = self.n
        ds = self.ds
        phi = 0.5 * (theta[: n - 1] + theta[1:])
        cos_ext = np.zeros(n)
        cos_ext[: n - 1] = np.cos(phi)
        sin_ext = np.zeros(n)
        sin_ext[: n - 1] = np.sin(phi)
        tail = self.mass_tail

        # C_k = T[k-1] cos phi_{k-1} + T[k] cos phi_k ; S_k likewise with sin
        c_prev = np.zeros(n)
        c_prev[1:] = tail[: n - 1] * cos_ext[: n - 1]
        c_cur = tail * cos_ext
        s_prev = np.zeros(n)
        s_prev[1:] = tail[: n - 1] * sin_ext[: n - 1]
        s_cur = tail * sin_ext
        big_c = c_prev + c_cur
        big_s = s_prev + s_cur

        # bending gradient: dU_b/dtheta_k = (EI/ds)(d_{k-1} - d_k)
        d = np.diff(theta)  # d[j] = theta[j+1] - theta[j], j = 0 .. n-2
        gb = np.zeros(n)
        # k = 1 .. n-1 receives +d_{k-1}; k = 0 .. n-2 receives -d_k (missing
        # end terms are simply absent, which the padded zeros express)
        gb[1:] += self.stiffness * d
        gb[: n - 1] -= self.stiffness * d

        # gravity gradient: dV/dtheta_k = lam*0.5*ds*( g_x S_k - g_z C_k )
        gv = lam * 0.5 * ds * (self.g_axial * big_s - self.g_normal * big_c)

        grad = (gb + gv)[1:]

        hess = self.h_bend.copy()
        coef = lam * 0.25 * ds
        diag_v = coef * (self.g_axial * big_c + self.g_normal * big_s)
        off_v = coef * (self.g_axial * c_cur + self.g_normal * s_cur)
        idx = np.arange(self.nu)
        hess[idx, idx] += diag_v[1:]
        if self.nu > 1:
            hess[idx[:-1], idx[:-1] + 1] += off_v[1 : n - 1]
            hess[idx[:-1] + 1, idx[:-1]] += off_v[1 : n - 1]
        return grad, hess

    # ---------------------------------------------------- equilibrium solver
    def _solve_equilibrium(self, lam: float) -> None:
        unk = self.unk.copy()
        tol = 1e-9 * max(1.0, self.reference_torque)
        residual = 0.0
        eye = np.eye(self.nu)
        trace_scale = float(np.trace(self.h_bend)) / max(self.nu, 1)
        for _ in range(100):
            theta = self._theta_full(unk)
            grad, hess = self._grad_hess(theta, lam)
            residual = float(np.max(np.abs(grad)))
            if residual < tol:
                break
            energy = self._energy(theta, lam)
            step = None
            for reg in (0.0, 1e-10, 1e-7, 1e-4, 1e-2):
                mat = hess if reg == 0.0 else hess + (reg * trace_scale) * eye
                try:
                    cand = np.linalg.solve(mat, -grad)
                except np.linalg.LinAlgError:
                    continue
                if np.all(np.isfinite(cand)) and float(cand @ grad) < 0.0:
                    step = cand
                    break
            if step is None:  # guaranteed descent direction (bending operator is SPD)
                step = np.linalg.solve(self.h_bend, -grad)
            alpha = 1.0
            improved = False
            for _ls in range(50):
                trial = unk + alpha * step
                value = self._energy(self._theta_full(trial), lam)
                if np.isfinite(value) and value <= energy - 1e-18 * abs(energy):
                    unk = trial
                    improved = True
                    break
                alpha *= 0.5
            self.newton_iterations += 1
            if not improved:
                break
        self.unk = unk
        self.theta = self._theta_full(unk)
        self.max_residual = max(self.max_residual, residual)
        self.solves += 1

    def _load_factor(self, time: float) -> float:
        if time <= self.ramp_start:
            return 0.0
        if time >= self.ramp_full:
            return 1.0
        return float((time - self.ramp_start) / (self.ramp_full - self.ramp_start))

    # ---------------------------------------------------------------- frames
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        lam = self._load_factor(time)
        self._solve_equilibrium(lam)

        theta = self.theta
        position = self._positions(theta)
        orientation = self._orientations(theta)

        dt = 0.0 if self.prev_time is None else time - self.prev_time
        if dt > 1e-12:
            velocity = (position - self.prev_position) / dt
            omega_rate = (theta - self.prev_theta) / dt
            angular_velocity = -omega_rate[:, None] * self.axis_y[None, :]
            if not np.all(np.isfinite(velocity)):
                velocity = np.zeros_like(position)
            if not np.all(np.isfinite(angular_velocity)):
                angular_velocity = np.zeros_like(position)
        else:
            velocity = np.zeros_like(position)
            angular_velocity = np.zeros_like(position)

        # ---- momentum and its rate ----------------------------------------
        linear_momentum = self.mass @ velocity
        angular_momentum = np.einsum("i,ij->j", self.mass, np.cross(position, velocity))
        if dt > 1e-12:
            dp_dt = (linear_momentum - self.prev_linear_momentum) / dt
            dl_dt = (angular_momentum - self.prev_angular_momentum) / dt
        else:
            dp_dt = np.zeros(3)
            dl_dt = np.zeros(3)

        # ---- external force/couple budget for the rod domain --------------
        force_gravity = lam * self.total_mass * self.gravity
        com = (self.mass @ position) / max(self.total_mass, 1e-15)
        force_clamp = -force_gravity + dp_dt
        torque_gravity = np.cross(com, force_gravity)
        couple_clamp = dl_dt - torque_gravity - np.cross(self.root_position, force_clamp)

        budget_force = np.zeros((1, 2, 3))
        budget_force[0, 0] = force_gravity
        budget_force[0, 1] = force_clamp
        budget_position = np.zeros((1, 2, 3))
        budget_position[0, 0] = com
        budget_position[0, 1] = self.root_position
        budget_couple = np.zeros((1, 2, 3))
        budget_couple[0, 1] = couple_clamp
        domain_weights = np.ones((1, self.n))

        # ---- inextensibility constraint (exact in this discretisation) -----
        segment_length = np.linalg.norm(np.diff(position, axis=0), axis=1)

        tip_drop = float(self.root_position[2] - position[-1, 2])
        frame = StateFrame(
            time=time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            attributes={
                "arc_length": self.arc_length.copy(),
                "is_clamped_root": self.is_root.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_force": budget_force,
                "budget_force_position": budget_position,
                "budget_external_couple": budget_couple,
                "budget_domain_weights": domain_weights,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "constraint_value": segment_length,
                "constraint_target": self.segment_reference.copy(),
            },
            scalars={
                "reference_force": self.reference_force,
                "reference_torque": self.reference_torque,
                "constraint_scale": float(self.ds),
                "characteristic_length": float(self.length),
                "rod_radius": float(self.radius),
                "bending_stiffness": float(self.ei),
                "youngs_modulus": float(self.YOUNG_MODULUS),
                "load_factor": lam,
                "tip_deflection": tip_drop,
            },
        )
        frame.validate()

        self.prev_time = time
        self.prev_position = position.copy()
        self.prev_theta = theta.copy()
        self.prev_linear_momentum = linear_momentum.copy()
        self.prev_angular_momentum = angular_momentum.copy()
        self.frame_index += 1
        self.last_tip_deflection = tip_drop
        return frame

    # ----------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        return {
            "model": "planar inextensible discrete elastica (Kirchhoff rod), quasi-static continuation",
            "nodes": int(self.n),
            "elements": int(self.n_edges),
            "segment_length_m": float(self.ds),
            "youngs_modulus_Pa": float(self.YOUNG_MODULUS),
            "bending_stiffness_Nm2": float(self.ei),
            "rod_mass_kg": float(self.total_mass),
            "reference_force_N": float(self.reference_force),
            "reference_torque_Nm": float(self.reference_torque),
            "equilibrium_solves": int(self.solves),
            "newton_iterations": int(self.newton_iterations),
            "max_gradient_residual_Nm": float(self.max_residual),
            "final_tip_deflection_m": float(self.last_tip_deflection),
            "seed": int(self.seed),
        }
