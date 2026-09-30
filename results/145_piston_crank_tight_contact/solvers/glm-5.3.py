"""Piston-crank tight-contact mechanism solver.

Physically-based constrained multibody dynamics (planar slider-crank):

* the crank is driven by a prescribed motor that spins it up smoothly
  from rest (constant target angular velocity after a smooth ramp),
* the connecting rod is a rigid body linked to the crank pin and to the
  piston pin by ideal revolute (pin) joints,
* the piston slides inside a tight rigid guide; the two guide surfaces
  act as unilateral contacts with a stiff penalty normal force and
  Coulomb friction (frozen friction coefficient),
* time integration is semi-implicit (symplectic) Euler on the
  acceleration-level constrained equations of motion, with the joint
  drift removed by Baumgarte stabilization.

No keyframes and no precomputed trajectories are used: every sample is
produced by stepping the dynamics with a fixed deterministic time step.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Slider-crank mechanism solver (rigid body dynamics, tight guide contact)."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.config = cfg
        self.seed = int(seed)

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.mu = float(cfg["contact"]["friction"])
        self.length_scale = float(cfg["contact"]["length_scale"])

        bodies = cfg["bodies"]  # order: crank, rod, piston (state bindings 0,1,2)
        self.mass = np.asarray([float(b["mass"]) for b in bodies], dtype=float)
        m_rod = self.mass[1]
        m_pis = self.mass[2]

        # ---- frozen geometry (scene_geometry.yaml + assets bounding boxes) ----
        # crank rotates about the static axle located at the crank body origin
        self.crank_c = np.asarray(bodies[0]["position"], dtype=float)  # (-0.8, 0, 1)
        # crank pin offset from the rotation axis (along local +x of the crank)
        self.r_crank = 0.45
        # connecting rod: hole centers at rod-local x = +/- h (rod length 2h)
        self.h_rod = 0.70
        # piston pin located at piston-local x = -b (rod small end presses on
        # the piston face; boss radius ~0.1 + piston half length 0.34)
        self.b_pin = 0.44
        # piston half extents (assets/piston.obj)
        self.piston_hz = 0.31
        # tight guide: inner surfaces of guide_lower / guide_upper
        self.guide_zlo = 0.63
        self.guide_zhi = 1.37
        # crank hub bore vs static axle radius (constant clearance)
        self.axle_clearance = 0.11

        # piston z bounds imposed by the tight guide (unilateral contacts)
        self.z_rest_lo = self.guide_zlo + self.piston_hz  # 0.94
        self.z_rest_hi = self.guide_zhi - self.piston_hz  # 1.06

        # ---- inertias (rod treated as a slender rigid body of length 2h) ----
        self.m_r = m_rod
        self.i_r = m_rod * (2.0 * self.h_rod) ** 2 / 12.0 + 0.02
        self.m_p = m_pis
        self.minv = np.array([1.0 / m_rod, 1.0 / m_rod, 1.0 / self.i_r,
                              1.0 / m_pis, 1.0 / m_pis])

        # ---- motor: smooth spin-up from rest to constant angular speed ----
        self.omega_s = 2.0 * np.pi  # rad/s (one revolution per second)
        self.tau = 0.8              # ramp time (s)
        self.beta0 = 0.0            # initial crank angle (identity orientation)

        # ---- guide contact model (penalty normal + Coulomb friction) ----
        self.k_contact = 2.0e4      # N/m
        self.c_contact = 2.0 * np.sqrt(self.k_contact * self.m_p)  # critical
        self.v_eps = 0.02           # m/s friction regularization

        # ---- integrator ----
        self.dt = 5.0e-4
        self.baumgarte_omega = 30.0
        self.baumgarte_zeta = 1.0

        # ---- initial assembled configuration (beta = 0 -> all identities ----
        # crank pin, rod and piston start aligned on the mechanism axis z=1
        beta = self.beta0
        pin_x = self.crank_c[0] + self.r_crank * np.cos(beta)
        pin_z = self.crank_c[2] - self.r_crank * np.sin(beta)
        # rod horizontal, big end on the crank pin, small end on the piston pin
        xr = 0.5 * (pin_x + (pin_x + 2.0 * self.h_rod))
        zr = pin_z
        psi = 0.0
        xp = pin_x + 2.0 * self.h_rod + self.b_pin
        zp = pin_z
        self.q = np.array([xr, zr, psi, xp, zp], dtype=float)
        self.v = np.zeros(5, dtype=float)

        self.n_steps = 0
        self.t = 0.0

        # constraint forces at the initial instant (static + motor spin-up)
        beta, betad, betadd = self._motor(self.t)
        _, lam, n_lo, n_hi, f_fric = self._solve_dynamics(
            self.q, self.v, beta, betad, betadd)
        self._lam = lam
        self._n_lo, self._n_hi, self._f_fric = n_lo, n_hi, f_fric

    # ----------------------------------------------------------------- motor
    def _motor(self, t: float):
        """Crank angle, angular velocity and angular acceleration at time t."""
        ch = np.cosh(t / self.tau)
        beta = self.beta0 + self.omega_s * self.tau * np.log(ch)
        betad = self.omega_s * np.tanh(t / self.tau)
        betadd = (self.omega_s / self.tau) / (ch * ch)
        return beta, betad, betadd

    # -------------------------------------------------------------- dynamics
    def _constraint_data(self, q, beta):
        """Joint constraints g(q, beta) = 0 and their Jacobians.

        q = [xr, zr, psi, xp, zp]; rotation about +y by angle a maps a local
        point (lx, 0, lz) to (lx cos a + lz sin a, 0, -lx sin a + lz cos a).
        """
        h = self.h_rod
        r = self.r_crank
        cx, cz = self.crank_c[0], self.crank_c[2]
        xr, zr, psi, xp, zp = q

        cb, sb = np.cos(beta), np.sin(beta)
        cp, sp = np.cos(psi), np.sin(psi)

        # crank pin (world)
        pin_x = cx + r * cb
        pin_z = cz - r * sb

        g = np.array([
            xr - h * cp - pin_x,               # rod big end x on crank pin
            zr + h * sp - pin_z,               # rod big end z on crank pin
            xr + h * cp - (xp - self.b_pin),   # rod small end x on piston pin
            zr - h * sp - zp,                  # rod small end z on piston pin
        ])
        # dg / dq
        j = np.array([
            [1.0, 0.0, h * sp, 0.0, 0.0],
            [0.0, 1.0, h * cp, 0.0, 0.0],
            [1.0, 0.0, -h * sp, -1.0, 0.0],
            [0.0, 1.0, -h * cp, 0.0, -1.0],
        ])
        # dg / dbeta
        gb = np.array([r * sb, -r * cb, 0.0, 0.0])
        return g, j, gb

    def _guide_forces(self, zp, zp_dot):
        """Unilateral tight-guide normal forces (penalty + damping)."""
        pen_lo = self.z_rest_lo - zp
        n_lo = 0.0
        if pen_lo > 0.0:
            n_lo = max(0.0, self.k_contact * pen_lo - self.c_contact * zp_dot)
        pen_hi = zp - self.z_rest_hi
        n_hi = 0.0
        if pen_hi > 0.0:
            n_hi = max(0.0, self.k_contact * pen_hi + self.c_contact * zp_dot)
        return n_lo, n_hi

    def _solve_dynamics(self, q, v, beta, betad, betadd):
        """Solve the acceleration-level KKT system of the constrained mechanism.

        Returns (a, lambda, n_lo, n_hi, f_fric) where lambda are the joint
        reaction forces (generalized, sign convention: force on the rod at the
        joint equals the corresponding lambda components).
        """
        g, j, gb = self._constraint_data(q, beta)
        psi = q[2]

        # constraint drift velocity  v_g = J v + g_beta * betad
        vg = j @ v + gb * betad

        # remaining second-order terms: (dJ/dt) v + (dg_beta/dt) betad
        h = self.h_rod
        r = self.r_crank
        cb, sb = np.cos(beta), np.sin(beta)
        cp, sp = np.cos(psi), np.sin(psi)
        rest = v[2] * v[2] * np.array([h * cp, -h * sp, -h * cp, h * sp])
        rest = rest + betad * betad * np.array([r * cb, -r * sb, 0.0, 0.0])

        # Baumgarte-stabilized acceleration-level constraint:  J a = rhs
        ob, zb = self.baumgarte_omega, self.baumgarte_zeta
        rhs = (-2.0 * zb * ob * vg - ob * ob * g - gb * betadd - rest)

        # guide contact (normal + friction) evaluated at the current state
        n_lo, n_hi = self._guide_forces(q[4], v[4])
        f_fric = -self.mu * (n_lo + n_hi) * np.tanh(v[3] / self.v_eps)

        # generalized external forces (gravity, guide contact, guide friction)
        f = np.array([
            0.0,
            self.m_r * self.gravity[2],
            0.0,
            f_fric,
            self.m_p * self.gravity[2] + (n_lo - n_hi),
        ])

        # KKT:  M a - J^T lambda = f ;  J a = rhs
        jm = j * self.minv[None, :]
        a_mat = jm @ j.T                      # J M^-1 J^T  (SPD, 4x4)
        lam = np.linalg.solve(a_mat, rhs - jm @ f)
        acc = self.minv * (f + j.T @ lam)
        return acc, lam, n_lo, n_hi, f_fric

    # ----------------------------------------------------------------- steps
    def _step(self) -> None:
        beta, betad, betadd = self._motor(self.t)
        acc, lam, n_lo, n_hi, f_fric = self._solve_dynamics(
            self.q, self.v, beta, betad, betadd)
        self.v = self.v + self.dt * acc
        self.q = self.q + self.dt * self.v
        self._lam = lam
        self._n_lo, self._n_hi, self._f_fric = n_lo, n_hi, f_fric
        self.n_steps += 1
        self.t = self.n_steps * self.dt

    # ---------------------------------------------------------------- frames
    def _quaternion_y(self, angle: float) -> np.ndarray:
        """xyzw quaternion of a rotation about +y by `angle`."""
        return np.array([0.0, np.sin(0.5 * angle), 0.0, np.cos(0.5 * angle)])

    def _frame(self, time: float) -> StateFrame:
        beta, betad, _ = self._motor(time)
        xr, zr, psi, xp, zp = self.q
        vxr, zvr, psid, vxp, vzp = self.v

        position = np.array([
            [self.crank_c[0], 0.0, self.crank_c[2]],
            [xr, 0.0, zr],
            [xp, 0.0, zp],
        ], dtype=float)
        velocity = np.array([
            [0.0, 0.0, 0.0],
            [vxr, 0.0, zvr],
            [vxp, 0.0, vzp],
        ], dtype=float)
        orientation = np.array([
            self._quaternion_y(beta),
            self._quaternion_y(psi),
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        angular_velocity = np.array([
            [0.0, betad, 0.0],
            [0.0, psid, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=float)

        # per-body external forces (gravity only; joints/contacts excluded)
        external_force = np.zeros((3, 3), dtype=float)
        external_force[:, 2] = self.mass * self.gravity[2]

        # paired internal interaction forces:
        #   0: crank pin -> rod big end (revolute joint reaction)
        #   1: rod small end -> piston (revolute joint reaction)
        #   2: guide normal contact force on the piston
        #   3: guide friction force on the piston
        lam = self._lam
        interaction_force = np.array([
            [lam[0], 0.0, lam[1]],
            [-lam[2], 0.0, -lam[3]],
            [0.0, 0.0, self._n_lo - self._n_hi],
            [self._f_fric, 0.0, 0.0],
        ], dtype=float)

        # signed geometric gaps of the active / near contacts
        contact_signed_distance = np.array([
            zp - self.z_rest_lo,        # piston vs lower guide (<0 = penetration)
            self.z_rest_hi - zp,        # piston vs upper guide (<0 = penetration)
            self.axle_clearance,        # crank hub bore vs static axle
        ], dtype=float)

        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=np.arange(3, dtype=np.int32),
            attributes={"release_time": np.zeros(3, dtype=float)},
            fields={
                "external_force": external_force,
                "interaction_force": interaction_force,
                "contact_signed_distance": contact_signed_distance,
            },
            scalars={"contact_length_scale": self.length_scale},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ api
    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        if target < self.n_steps:
            # requested time already passed (should not happen); rebuild frame
            return self._frame(self.n_steps * self.dt)
        while self.n_steps < target:
            self._step()
        return self._frame(self.n_steps * self.dt)

    def diagnostics(self) -> dict:
        beta, betad, _ = self._motor(self.t)
        return {
            "time": float(self.t),
            "steps": int(self.n_steps),
            "crank_angle": float(beta),
            "motor_omega": float(betad),
            "piston_x": float(self.q[3]),
            "piston_z": float(self.q[4]),
            "constraint_drift": float(np.max(np.abs(
                self._constraint_data(self.q, beta)[0]))),
        }
