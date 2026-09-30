"""Wilberforce pendulum solver (self-implemented, no external physics engine).

Physical model
-------------
The visible helical spring is treated as a one-dimensional elastic continuum
carrying two coupled generalized strains: axial extension ``e`` (change of the
coil-stack height) and torsion ``t`` (relative rotation of neighbouring coils
about the vertical axis).  For a helical spring the wire bending/torsion
compliance makes those two strains couple, which is exactly the mechanism of
the Wilberforce pendulum: the segment constitutive law is

    F = k_s * e + eps_s * t          (axial force)
    T = eps_s * e + kappa_s * t      (axial torque)

with a symmetric positive-definite 2x2 stiffness (``eps_s**2 < k_s*kappa_s``),
so the elastic energy per segment is

    W = 0.5*k_s*e**2 + 0.5*kappa_s*t**2 + eps_s*e*t.

The continuum is discretized into ``M`` segments / ``M+1`` nodes.  Every node
``j`` owns two degrees of freedom: axial displacement ``u_j`` (positive up) and
rotation ``phi_j`` about the vertical axis, lumped mass ``m_j`` and lumped
polar inertia ``I_j = m_j * R**2`` (coil of radius ``R``).  Node 0 is welded to
the frozen ``upper_attachment`` (Dirichlet boundary: ``u_0 = phi_0 = 0``), the
last node carries the rigid ``suspended_weight`` (mass and polar inertia of the
frozen cylinder), which realizes the spring-body attachment.

Gravity acts on every node.  The reference configuration is the frozen scene
helix; per-segment pre-strains ``(e0, t0)`` are obtained from a static solve so
that the internal force in each segment equals the weight hanging below it plus
a uniform extra pre-tension (the spring starts stretched below its equilibrium,
which launches the vertical oscillation).  Internal torque vanishes in the
reference state.

Time stepping is explicit velocity Verlet (kick-drift-kick) at a fixed
sub-step, with small stiffness-proportional (Rayleigh) damping that removes the
fastest discrete modes while leaving the two Wilberforce modes essentially
undamped.  The scheme is stable for the chosen ``dt`` (dt*omega_max ~ 0.25) and
fully deterministic.

Output mapping
--------------
The 128 frozen spring samples are material points of the rod at arclength
fractions ``s_i``; their state follows from the nodal fields by linear
interpolation:

    x = R*cos(Phi_i + phi(s)),  y = R*sin(Phi_i + phi(s)),  z = z_ref_i + u(s)
    vx = -R*sin(...)*phidot(s),  vy = R*cos(...)*phidot(s), vz = udot(s)

so positions and velocities are consistent (velocity is the exact time
derivative of the position map).  The weight sample follows the last node.
"""
from __future__ import annotations

import numpy as np

GRAVITY = 9.81


class BenchmarkSolver:
    # --- discretization -------------------------------------------------
    n_segments = 64
    sub_dt = 1.0e-4
    rayleigh_beta = 1.0e-4

    # --- modal design (helical spring + cylinder, weak axial/torsion mix)
    target_period = 1.0          # s, vertical mode period
    coupling_ratio = 0.25        # eps / sqrt(k*kappa) -> beat period ~3.95 s
    extra_tip_stretch = 0.15     # m, initial stretch below static equilibrium

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.sample_dt = float(cfg.get("sample_dt", 0.01))
        self.duration = float(cfg.get("duration", 12.0))

        frame0 = scene.initial_state()
        pos0 = np.asarray(frame0.position, dtype=float)
        vel0 = np.asarray(frame0.velocity, dtype=float)
        mass0 = np.asarray(frame0.mass, dtype=float)
        body0 = np.asarray(frame0.body_id, dtype=int)
        spin0 = np.asarray(frame0.angular_velocity, dtype=float)

        self.n_spring = int(np.count_nonzero(body0 == 0))
        self.n_samples = pos0.shape[0]
        self.body_id = body0.copy()
        self.sample_mass = mass0.copy()

        # Frozen reference helix (material parameterization).
        spring_ref = pos0[body0 == 0]
        self.helix_radius = float(np.mean(np.hypot(spring_ref[:, 0], spring_ref[:, 1])))
        if not np.isfinite(self.helix_radius) or self.helix_radius <= 1e-9:
            self.helix_radius = 0.16
        self.ref_phase = np.unwrap(np.arctan2(spring_ref[:, 1], spring_ref[:, 0]))
        self.ref_z = spring_ref[:, 2].copy()
        self.weight_ref = pos0[body0 == 1][0].copy()

        # Arclength fraction of each visible sample (reference height is linear).
        self.sample_s = np.linspace(0.0, 1.0, self.n_spring)

        # Rigid weight properties from the frozen cylinder geometry.
        weight_mass = float(mass0[body0 == 1].sum())
        weight_radius = 0.34
        self.weight_mass = weight_mass
        self.weight_inertia = 0.5 * weight_mass * weight_radius ** 2

        spring_mass = float(mass0[body0 == 0].sum())
        self.spring_mass = spring_mass

        # Nodal grid ----------------------------------------------------
        M = int(self.n_segments)
        self.node_s = np.linspace(0.0, 1.0, M + 1)
        seg_mass = spring_mass / M
        m_node = np.full(M + 1, seg_mass)
        m_node[0] *= 0.5
        m_node[-1] *= 0.5
        I_node = m_node * self.helix_radius ** 2
        m_node[-1] += weight_mass
        I_node[-1] += self.weight_inertia
        self.m_node = m_node
        self.I_node = I_node

        # Global modal stiffnesses (tuned resonance -> full energy exchange).
        w0 = 2.0 * np.pi / self.target_period
        m_eff = weight_mass + spring_mass / 3.0
        I_eff = self.weight_inertia + spring_mass * self.helix_radius ** 2 / 3.0
        k_axial = m_eff * w0 ** 2
        k_tors = I_eff * w0 ** 2
        eps = self.coupling_ratio * np.sqrt(k_axial * k_tors)
        self.k_axial, self.k_tors, self.eps = k_axial, k_tors, eps

        # Series assembly: each of M segments is M times stiffer.
        self.ks = k_axial * M
        self.kts = k_tors * M
        self.eps_s = eps * M

        # Static pre-strains: segment tension carries the weight below it plus a
        # uniform extra pre-tension, internal torque vanishes in the reference.
        below = np.array([m_node[j + 1:].sum() for j in range(M)], dtype=float)
        k_eff_seg = self.ks - self.eps_s ** 2 / self.kts
        f_extra = k_axial * float(self.extra_tip_stretch)
        f_target = GRAVITY * below + f_extra
        self.e0 = f_target / k_eff_seg
        self.t0 = -self.eps_s * self.e0 / self.kts

        # Degrees of freedom, initialized from the frozen initial state.
        self.u = np.zeros(M + 1)
        self.phi = np.zeros(M + 1)
        self.vu = np.zeros(M + 1)
        self.vphi = np.zeros(M + 1)

        # Initial velocities: the frozen state gives a uniform downward drift and a
        # spin of the spring coils; map both onto the nodal fields, tapering
        # linearly to zero at the welded top node.
        weight_vz = float(vel0[body0 == 1][0, 2])
        spring_vz = float(np.mean(vel0[body0 == 0][:, 2])) if self.n_spring else weight_vz
        spring_spin = float(np.mean(spin0[body0 == 0][:, 2])) if self.n_spring else 0.0
        weight_spin = float(spin0[body0 == 1][0, 2])
        self.vu[:] = self.node_s * spring_vz
        self.vu[-1] = weight_vz
        self.vphi[:] = self.node_s * spring_spin
        self.vphi[-1] = weight_spin if abs(weight_spin) > 0.0 else spring_spin
        self.vu[0] = 0.0
        self.vphi[0] = 0.0

        self.time = 0.0
        self.step_count = 0
        self.reaction_z = 0.0
        self._diag = {
            "method": "coupled axial-torsional rod (velocity Verlet)",
            "nodes": M + 1,
            "sub_dt": self.sub_dt,
            "axial_stiffness": k_axial,
            "torsional_stiffness": k_tors,
            "coupling": eps,
        }
        self._accel()  # prime the reaction force

    # ------------------------------------------------------------------
    def _accel(self):
        u, phi, vu, vphi = self.u, self.phi, self.vu, self.vphi
        e = self.e0 + (u[:-1] - u[1:])
        t = self.t0 + (phi[:-1] - phi[1:])
        ed = vu[:-1] - vu[1:]
        td = vphi[:-1] - vphi[1:]
        beta = self.rayleigh_beta
        F = self.ks * e + self.eps_s * t + beta * (self.ks * ed + self.eps_s * td)
        T = self.eps_s * e + self.kts * t + beta * (self.eps_s * ed + self.kts * td)

        fu = np.zeros_like(u)
        tau = np.zeros_like(phi)
        fu[:-1] -= F
        fu[1:] += F
        tau[:-1] -= T
        tau[1:] += T
        # boundary reaction at the welded top node (external support force)
        self.reaction_z = float(F[0] + self.m_node[0] * GRAVITY)
        fu -= self.m_node * GRAVITY

        fu[0] = 0.0
        tau[0] = 0.0
        self._elastic_F = F
        self._elastic_T = T
        return fu / self.m_node, tau / self.I_node

    def _step(self, dt: float) -> None:
        au, ap = self._accel()
        self.vu += 0.5 * dt * au
        self.vphi += 0.5 * dt * ap
        self.u += dt * self.vu
        self.phi += dt * self.vphi
        self.u[0] = 0.0
        self.phi[0] = 0.0
        au, ap = self._accel()
        self.vu += 0.5 * dt * au
        self.vphi += 0.5 * dt * ap
        self.vu[0] = 0.0
        self.vphi[0] = 0.0

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        n_target = int(round(target / self.sub_dt))
        while self.step_count < n_target:
            self._step(self.sub_dt)
            self.step_count += 1
        self.time = self.step_count * self.sub_dt
        return self._frame(max(target, 0.0))

    def _frame(self, stamp: float):
        from worldsolver.contracts import StateFrame

        ns = self.n_spring
        s = self.sample_s
        u_s = np.interp(s, self.node_s, self.u)
        phi_s = np.interp(s, self.node_s, self.phi)
        vu_s = np.interp(s, self.node_s, self.vu)
        vphi_s = np.interp(s, self.node_s, self.vphi)

        ang = self.ref_phase + phi_s
        R = self.helix_radius
        ca, sa = np.cos(ang), np.sin(ang)

        n = self.n_samples
        position = np.zeros((n, 3))
        velocity = np.zeros((n, 3))
        orientation = np.zeros((n, 4))
        angular_velocity = np.zeros((n, 3))

        position[:ns, 0] = R * ca
        position[:ns, 1] = R * sa
        position[:ns, 2] = self.ref_z + u_s
        velocity[:ns, 0] = -R * sa * vphi_s
        velocity[:ns, 1] = R * ca * vphi_s
        velocity[:ns, 2] = vu_s
        angular_velocity[:ns, 2] = vphi_s
        half = 0.5 * phi_s
        orientation[:ns, 2] = np.sin(half)
        orientation[:ns, 3] = np.cos(half)

        # rigid weight sample (last node of the rod)
        uw = float(self.u[-1])
        pw = float(self.phi[-1])
        position[ns] = (self.weight_ref[0], self.weight_ref[1], self.weight_ref[2] + uw)
        velocity[ns] = (0.0, 0.0, float(self.vu[-1]))
        angular_velocity[ns] = (0.0, 0.0, float(self.vphi[-1]))
        orientation[ns] = (0.0, 0.0, np.sin(0.5 * pw), np.cos(0.5 * pw))

        mass = self.sample_mass
        external_force = np.zeros((n, 3))
        external_force[:, 2] = -mass * GRAVITY
        external_force[0, 2] += self.reaction_z  # weld reaction from upper_attachment

        e = self.e0 + (self.u[:-1] - self.u[1:])
        t = self.t0 + (self.phi[:-1] - self.phi[1:])
        elastic = float(
            np.sum(0.5 * self.ks * e ** 2 + 0.5 * self.kts * t ** 2 + self.eps_s * e * t)
        )
        ke_trans = float(np.sum(0.5 * self.m_node * self.vu ** 2))
        ke_tors = float(np.sum(0.5 * self.I_node * self.vphi ** 2))
        grav = float(np.sum(self.m_node * GRAVITY * self.u))

        frame = StateFrame(
            float(stamp),
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"mass_quantity": mass.copy()},
            fields={
                "cumulative_mass_source": np.zeros(1),
                "external_force": external_force,
            },
            scalars={
                "translational_energy": ke_trans,
                "torsional_energy": ke_tors,
                "elastic_energy": elastic,
                "total_energy": ke_trans + ke_tors + elastic + grav,
                "weight_height": float(position[ns, 2]),
                "weight_twist": pw,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        out = dict(self._diag)
        out["steps"] = self.step_count
        out["sim_time"] = self.time
        return out
