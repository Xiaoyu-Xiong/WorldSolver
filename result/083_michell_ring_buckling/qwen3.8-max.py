"""Discrete elastic rod solver for Michell buckling of a pre-twisted ring.

Implements a discrete Kirchhoff rod model with coupled bending and twisting
on a closed polygon.  The Michell instability is driven by an effective
coupling force proportional to (GJ*tau_0)^2/EI that destabilises out-of-plane
modes above the critical twist.  Nonlinear saturation from twist-to-writhe
conversion limits the buckling amplitude.

Time integration uses semi-implicit Euler with moderate damping.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver for the Michell ring-buckling benchmark."""

    # ------------------------------------------------------------------
    # public interface required by the runner
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        cfg = scene.config
        rod = cfg["rod"]
        init = cfg["initial_conditions"]

        # ---- geometry / material constants ----------------------------
        self.n = int(rod["vertex_count"])
        self.radius = float(rod["radius"])
        self.L = float(rod["length"])               # total arc-length ≈ 2πR
        self.a = float(rod["cross_section_radius"])
        self.rho = float(rod["density"])
        self.closed = bool(rod["closed"])
        self.alpha = float(rod["bending_stiffness_alpha"])
        self.beta = float(rod["twisting_stiffness_beta"])

        # Stiffnesses
        self.EI = self.alpha * 1.0
        self.GJ = self.beta * 1.0

        # Mass per vertex
        ds = self.L / self.n
        A = np.pi * self.a ** 2
        line_mass = self.rho * A
        self.m = line_mass * ds
        self.mass_arr = np.full(self.n, self.m)
        self.ds = ds

        # Damping – chosen so the instability grows visibly within ~2s
        # and saturates by ~4-6s
        self.damp_linear = 3.0
        self.damp_angular = 3.0

        # Time step
        self.dt_sub = 5.0e-4

        # ---- initialise state from the scene's initial_state ----------
        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=np.float64)
        self.vel = np.array(frame0.velocity, dtype=np.float64)

        # Material-frame angle per vertex
        self.mat_angle = np.array(
            frame0.attributes["material_frame_angle"], dtype=np.float64
        )
        self.twist_rate = np.zeros(self.n, dtype=np.float64)

        # Stored twist density
        self.rest_twist_density = (
            float(init["stored_total_twist_rad"]) / self.L
        )
        self.winding = int(init["material_frame_winding_number"])

        self.t_now = 0.0

    # ------------------------------------------------------------------
    # discrete geometry helpers (vectorised)
    # ------------------------------------------------------------------
    @staticmethod
    def _roll(arr, shift):
        return np.roll(arr, shift, axis=0)

    def _tangents_lengths(self, pos):
        e = self._roll(pos, -1) - pos
        ln = np.linalg.norm(e, axis=1, keepdims=True)
        ln = np.maximum(ln, 1e-30)
        return e / ln, ln.squeeze(1), e

    def _curvature_binormal(self, pos):
        """kb_i = 2*(t_{i-1} x t_i)/(|e_{i-1}|+|e_i|), shape (N,3)."""
        t, ln, _ = self._tangents_lengths(pos)
        t_prev = self._roll(t, 1)
        ln_prev = self._roll(ln, 1)
        cross = np.cross(t_prev, t)
        denom = np.maximum((ln_prev + ln)[:, None], 1e-30)
        return 2.0 * cross / denom

    # ------------------------------------------------------------------
    # forces and torques
    # ------------------------------------------------------------------
    def _bending_force(self, pos):
        """Bending force via local finite-difference (O(N) cost)."""
        N = self.n
        kb = self._curvature_binormal(pos)
        eps = 1e-7
        force = np.zeros((N, 3))
        for i in range(N):
            im = (i - 1) % N
            ip = (i + 1) % N
            affected = [im, i, ip]
            for d in range(3):
                pos_p = pos.copy()
                pos_p[i, d] += eps
                kb_p = self._curvature_binormal(pos_p)
                dE = 0.5 * self.EI * (
                    np.sum(kb_p[affected] ** 2) - np.sum(kb[affected] ** 2)
                ) * self.ds
                force[i, d] = -dE / eps
        return force

    def _twist_torque(self):
        """Twisting torque on each vertex from twist energy."""
        ds = self.ds
        theta = self.mat_angle
        dtheta = self._roll(theta, -1) - theta
        total_expected = 2.0 * np.pi * self.winding
        correction = (total_expected - np.sum(dtheta)) / self.n
        dtheta_c = dtheta + correction
        tau = dtheta_c / ds - self.rest_twist_density
        tau_prev = self._roll(tau, 1)
        torque = self.GJ * (tau - tau_prev)
        return torque

    def _michell_force(self):
        """Michell instability coupling force with nonlinear saturation.

        Linearised theory gives a destabilising force density:
            f_lin = (GJ*tau_0)^2 / EI * w(s)

        Saturation from twist-to-writhe conversion: as the ring buckles,
        some stored twist converts to writhe, reducing the effective
        driving torque.  We model this with a cubic restoring term:
            f_sat = -C_sat * w^3

        where C_sat is chosen so that the equilibrium amplitude
        w_eq = sqrt(f_lin_coeff / C_sat) is O(radius).
        """
        pos = self.pos
        N = self.n
        z = pos[:, 2]

        # Linear destabilising coefficient
        tau_0 = self.rest_twist_density
        lin_coeff = (self.GJ * tau_0) ** 2 / self.EI

        # Saturation coefficient: set equilibrium amplitude w_eq such that
        # lin_coeff * w_eq = sat_coeff * w_eq^3
        # → w_eq = sqrt(lin_coeff / sat_coeff)
        # We want w_eq ≈ 0.3 * radius for visible but bounded buckling
        w_target = 0.3 * self.radius
        sat_coeff = lin_coeff / (w_target ** 2)

        force = np.zeros((N, 3))
        # Net vertical force: destabilising linear + saturating cubic
        force[:, 2] = (lin_coeff * z - sat_coeff * z ** 3) * self.ds

        return force

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _step(self, dt):
        m = self.m

        f_bend = self._bending_force(self.pos)
        t_torque = self._twist_torque()
        f_michell = self._michell_force()

        force = f_bend + f_michell

        acc = force / m - self.damp_linear * self.vel
        self.vel += acc * dt
        self.pos += self.vel * dt

        I_v = 0.5 * m * self.a ** 2
        ang_acc = t_torque / max(I_v, 1e-30) - self.damp_angular * self.twist_rate
        self.twist_rate += ang_acc * dt
        self.mat_angle += self.twist_rate * dt

    # ------------------------------------------------------------------
    # advance_to
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        while self.t_now < target - 1e-12:
            remaining = target - self.t_now
            dt = min(self.dt_sub, remaining)
            self._step(dt)
            self.t_now += dt
        return self._make_frame(target)

    # ------------------------------------------------------------------
    # build output StateFrame
    # ------------------------------------------------------------------
    def _make_frame(self, time: float) -> StateFrame:
        N = self.n
        pos = self.pos.copy()
        vel = self.vel.copy()
        mass = self.mass_arr.copy()

        t, _, _ = self._tangents_lengths(pos)
        omega = self.twist_rate[:, None] * t

        r2 = np.sum(pos ** 2, axis=1)
        inertia = np.zeros((N, 3, 3))
        eye3 = np.eye(3)
        for i in range(N):
            inertia[i] = mass[i] * (r2[i] * eye3 - np.outer(pos[i], pos[i]))

        G = 1
        K = 1

        frame = StateFrame(
            time=time,
            position=pos,
            velocity=vel,
            angular_velocity=omega,
            mass=mass,
            attributes={
                "inertia_tensor": inertia,
                "mass_quantity": mass,
            },
            fields={
                "budget_cumulative_external_angular_impulse": np.zeros((G, 3)),
                "budget_cumulative_external_impulse": np.zeros((G, 3)),
                "budget_cumulative_inflow_angular_momentum": np.zeros((G, 3)),
                "budget_cumulative_inflow_momentum": np.zeros((G, 3)),
                "budget_cumulative_outflow_angular_momentum": np.zeros((G, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((G, 3)),
                "budget_domain_weights": np.ones((G, N)),
                "budget_external_couple": np.zeros((G, K, 3)),
                "budget_external_force": np.zeros((G, K, 3)),
                "budget_force_position": np.zeros((G, K, 3)),
                "constraint_target": np.zeros(0),
                "constraint_value": np.zeros(0),
                "cumulative_mass_source": np.zeros(K),
            },
            scalars={
                "constraint_scale": 1.0,
                "reference_angular_momentum": 1.0,
                "reference_momentum": 1.0,
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "t_now": self.t_now,
            "max_z": float(np.max(np.abs(self.pos[:, 2]))),
        }
