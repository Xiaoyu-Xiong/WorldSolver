"""Wilberforce pendulum solver – reduced coupled vertical-torsional model.

The helical spring is represented by its vertex positions (for visualisation)
but the dynamics are integrated on the two coupled rigid-body DOFs of the
load: vertical translation z and rotation phi about the spring axis.  The
spring stiffnesses k_z, k_phi and the weak coupling epsilon are chosen from
the requested modal periods so that the characteristic energy exchange
("beats") between vertical and torsional motion is reproduced within the
18 s window.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Coupled vertical-torsional Wilberforce pendulum."""

    # ------------------------------------------------------------------
    # public API required by the runner
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = seed
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.g = float(-self.gravity[2])  # positive magnitude

        # Spring geometry -------------------------------------------------
        sp = cfg["spring"]
        self.n_spring = int(sp["vertex_count"])
        self.coil_count = float(sp["coil_count"])
        self.coil_radius = float(sp["coil_radius"])
        self.z_top_eq = float(sp["equilibrium_top_z"])
        self.z_bot_eq = float(sp["equilibrium_bottom_z"])
        self.spring_mass = float(sp["total_mass"])
        self.n_clamp = int(sp["upper_clamped_vertex_count"])

        # Load ------------------------------------------------------------
        ld = cfg["load"]
        self.load_mass = float(ld["mass"])
        self.I_principal = np.asarray(ld["principal_inertia"], dtype=float)
        self.attach_offset_body = np.asarray(ld["attachment_offset_body"], dtype=float)

        # Derived inertia about spring axis (z)
        self.Izz = float(self.I_principal[2])

        # Modal regime ----------------------------------------------------
        cm = cfg["coupled_mode_regime"]
        T_z = float(cm["target_vertical_period"])
        T_phi = float(cm["target_torsional_period"])
        omega_z = 2.0 * np.pi / T_z
        omega_phi = 2.0 * np.pi / T_phi

        # Stiffnesses from desired uncoupled frequencies
        self.k_z = self.load_mass * omega_z ** 2
        self.k_phi = self.Izz * omega_phi ** 2

        # Weak coupling chosen to give first rotation peak near 4.5 s
        # Beat frequency ~ (omega_phi - omega_z) for weak coupling; we add a
        # small extra coupling term so the beat envelope matches observation.
        delta_omega = abs(omega_phi - omega_z)
        # For pure detuning the beat period would be 2*pi/delta_omega ≈ 30 s
        # which is too slow.  We therefore introduce an explicit coupling
        # epsilon such that the split normal-mode frequencies differ by
        # ΔΩ ≈ π / expected_first_rotation_peak_time.
        target_split = np.pi / float(cm["expected_first_rotation_peak_time"])
        # Effective split squared ≈ delta_omega^2 + (epsilon/M_eff)^2 where
        # M_eff ~ sqrt(m*Izz).  Solve for epsilon:
        if target_split > delta_omega:
            extra = np.sqrt(target_split ** 2 - delta_omega ** 2)
        else:
            extra = 0.05 * omega_z  # fallback tiny coupling
        self.epsilon = extra * np.sqrt(self.load_mass * self.Izz)

        # Equilibrium extension under gravity -----------------------------
        # At rest the spring supports m*g → equilibrium stretch dz_eq from
        # natural length is m*g/k_z.  The provided equilibrium_bottom_z
        # already includes this sag so we treat it as the resting position.
        self.z_eq = self.z_bot_eq  # load attachment z at rest
        self.phi_eq = 0.0

        # Initial conditions ----------------------------------------------
        ic = cfg["initial_conditions"]
        self.dz0 = float(ic["upward_displacement"])
        self.v0 = np.asarray(ic["initial_linear_velocity"], dtype=float)
        self.omega0 = np.asarray(ic["initial_angular_velocity"], dtype=float)

        # Time bookkeeping ------------------------------------------------
        self.dt_sample = float(cfg["sample_dt"])
        # Internal integration timestep (sub-step for accuracy)
        self.dt_int = min(5e-4, self.dt_sample / 20.0)

        # State: [z, phi, vz, vphi]
        self.state = np.array([
            self.z_eq + self.dz0,
            self.phi_eq,
            self.v0[2],
            self.omega0[2],
        ], dtype=float)
        self.current_time = 0.0

        # Precompute spring rest vertex positions (at equilibrium)
        theta = np.linspace(0.0, 2.0 * np.pi * self.coil_count, self.n_spring)
        self.rest_r = self.coil_radius
        self.rest_theta = theta
        self.rest_z = np.linspace(self.z_top_eq, self.z_bot_eq, self.n_spring)

        # Reference scales for scalars output
        p_ref = self.load_mass * np.sqrt(self.g * abs(self.dz0))
        if p_ref < 1e-12:
            p_ref = self.load_mass * 0.01
        self.ref_momentum = float(p_ref)
        self.ref_angmom = float(p_ref * self.coil_radius)

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        t_target = float(time)
        if t_target < self.current_time:
            # Should not happen; return current state safely
            return self._make_frame(self.current_time)

        # Integrate in fixed sub-steps
        while self.current_time < t_target - 1e-12:
            dt = min(self.dt_int, t_target - self.current_time)
            self._rk4_step(dt)
            self.current_time += dt

        return self._make_frame(t_target)

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {"model": "coupled_vertical_torsional_wilberforce"}

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _rhs(self, y):
        z, phi, vz, vphi = y
        dz = z - self.z_eq
        dphi = phi - self.phi_eq
        # Forces / torques (linearised coupling)
        Fz = -self.k_z * dz - self.epsilon * dphi
        Tau = -self.k_phi * dphi - self.epsilon * dz
        az = Fz / self.load_mass
        alpha = Tau / self.Izz
        return np.array([vz, vphi, az, alpha], dtype=float)

    def _rk4_step(self, dt):
        y = self.state
        k1 = self._rhs(y)
        k2 = self._rhs(y + 0.5 * dt * k1)
        k3 = self._rhs(y + 0.5 * dt * k2)
        k4 = self._rhs(y + dt * k3)
        self.state = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ------------------------------------------------------------------
    def _make_frame(self, time: float) -> StateFrame:
        z, phi, vz, vphi = self.state

        # --- Build full N-sample arrays -----------------------------------
        n_sp = self.n_spring
        # Spring vertex positions follow the load displacement linearly
        # along the arc: upper clamp fixed, lower end tracks (z, phi).
        frac = np.linspace(0.0, 1.0, n_sp)
        # Twist interpolated along spring
        local_phi = phi * frac
        r = self.rest_r
        th = self.rest_theta + local_phi
        # Z: upper clamped vertices stay at top; others interpolate
        z_verts = self.rest_z.copy()
        # Only unclamped portion moves with the load stretch
        free_mask = frac[n_sp - 1]  # just 1.0; compute shifted z
        # Linear blend: fixed top part keeps rest_z; bottom moves with load
        # Clamp region (first n_clamp verts) stays fixed
        stretch = z - self.z_eq
        z_shift = stretch * frac
        # For clamped vertices, force zero shift
        z_shift[: self.n_clamp] = 0.0
        z_verts = self.rest_z + z_shift

        x_sp = r * np.cos(th)
        y_sp = r * np.sin(th)
        pos_spring = np.column_stack((x_sp, y_sp, z_verts))

        # Load centre
        attach_world = np.array([0.0, 0.0, z])  # attachment point on spring end
        # Orientation marker sits above load centre; load body frame offset
        # is (0,0,+0.15) relative to load CoM per config
        load_center = attach_world - self.attach_offset_body
        marker_center = load_center + np.array([0.0, 0.0, 0.08])

        pos_load = load_center.reshape(1, 3)
        pos_marker = marker_center.reshape(1, 3)
        position = np.vstack((pos_spring, pos_load, pos_marker))

        N = len(position)

        # Velocities -------------------------------------------------------
        vel_spring = np.zeros_like(pos_spring)
        # Approximate spring vertex velocities proportional to frac
        vel_spring[:, 2] = vz * frac
        # Tangential velocity from twist rate: v_tan = r * frac * vphi
        vtan = r * frac * vphi
        vel_spring[:, 0] += -vtan * np.sin(th)
        vel_spring[:, 1] += vtan * np.cos(th)
        vel_spring[: self.n_clamp] = 0.0

        vel_load = np.array([[0.0, 0.0, vz]])
        vel_marker = vel_load.copy()  # rigid attachment
        velocity = np.vstack((vel_spring, vel_load, vel_marker))

        # Orientations (quaternion xyzw) -----------------------------------
        orientation = np.zeros((N, 4), dtype=float)
        orientation[:, 3] = 1.0  # identity default
        # Load and marker share the same rotation about z
        half_phi = 0.5 * phi
        qz = np.sin(half_phi)
        qw = np.cos(half_phi)
        orientation[-2, :] = [0.0, 0.0, qz, qw]
        orientation[-1, :] = [0.0, 0.0, qz, qw]
        # Spring vertices: interpolated twist
        for i in range(n_sp):
            hp = 0.5 * local_phi[i]
            orientation[i, :] = [0.0, 0.0, np.sin(hp), np.cos(hp)]

        # Angular velocity -------------------------------------------------
        angular_velocity = np.zeros((N, 3), dtype=float)
        angular_velocity[:n_sp, 2] = vphi * frac
        angular_velocity[: self.n_clamp] = 0.0
        angular_velocity[-2, 2] = vphi
        angular_velocity[-1, 2] = vphi

        # Mass -------------------------------------------------------------
        seg_mass = self.spring_mass / n_sp
        mass = np.empty(N, dtype=float)
        mass[:n_sp] = seg_mass
        mass[-2] = self.load_mass
        mass[-1] = 0.0  # orientation marker is massless tag

        # Body id ----------------------------------------------------------
        body_id = np.zeros(N, dtype=np.int64)
        body_id[:n_sp] = 0
        body_id[-2] = 1
        body_id[-1] = 2

        # Attributes -------------------------------------------------------
        # Inertia tensor per sample (world-frame). For spring particles use
        # small isotropic placeholder; for load use principal inertia rotated
        # into world frame; marker zero.
        inertia = np.zeros((N, 3, 3), dtype=float)
        # Spring segments: negligible rotational inertia
        # Load
        Rz = np.array([
            [np.cos(phi), -np.sin(phi), 0.0],
            [np.sin(phi), np.cos(phi), 0.0],
            [0.0, 0.0, 1.0],
        ])
        I_body = np.diag(self.I_principal)
        I_world = Rz @ I_body @ Rz.T
        inertia[-2] = I_world
        # Marker zero (already)

        # mass_quantity per sample (single material)
        mass_quantity = mass.copy()

        # Fields – budget quantities --------------------------------------
        # We define G=1 global domain.
        G = 1
        K = 1  # one external-force term slot

        # Domain weights: all samples belong to global domain
        budget_domain_weights = np.ones((G, N), dtype=float)

        # External forces: gravity on every massive sample
        ext_force = np.zeros((G, K, 3), dtype=float)
        total_grav = np.array([0.0, 0.0, -self.g * self.load_mass])
        ext_force[0, 0] = total_grav
        # Force application point: load CoM
        force_pos = np.zeros((G, K, 3), dtype=float)
        force_pos[0, 0] = load_center

        # External couple (none applied directly)
        ext_couple = np.zeros((G, K, 3), dtype=float)

        # Cumulative impulse / angular impulse since release
        # Gravity has been acting for `time` seconds.
        cum_impulse = np.zeros((G, 3), dtype=float)
        cum_impulse[0] = total_grav * time
        cum_ang_impulse = np.zeros((G, 3), dtype=float)
        # τ = r × F; r = load_center, F = total_grav (both along z → zero torque)
        cum_ang_impulse[0] = np.cross(load_center, total_grav) * time

        # Momentum inflow/outflow: closed system → zeros
        zeros_G3 = np.zeros((G, 3), dtype=float)
        cum_inflow_mom = zeros_G3.copy()
        cum_outflow_mom = zeros_G3.copy()
        cum_inflow_amom = zeros_G3.copy()
        cum_outflow_amom = zeros_G3.copy()

        # Cumulative mass source: no mass transfer
        cum_mass_source = np.zeros((K,), dtype=float)

        fields = {
            "budget_cumulative_external_angular_impulse": cum_ang_impulse,
            "budget_cumulative_external_impulse": cum_impulse,
            "budget_cumulative_inflow_angular_momentum": cum_inflow_amom,
            "budget_cumulative_inflow_momentum": cum_inflow_mom,
            "budget_cumulative_outflow_angular_momentum": cum_outflow_amom,
            "budget_cumulative_outflow_momentum": cum_outflow_mom,
            "budget_domain_weights": budget_domain_weights,
            "budget_external_couple": ext_couple,
            "budget_external_force": ext_force,
            "budget_force_position": force_pos,
            "cumulative_mass_source": cum_mass_source,
        }

        attributes = {
            "inertia_tensor": inertia,
            "mass_quantity": mass_quantity,
        }

        scalars = {
            "reference_angular_momentum": self.ref_angmom,
            "reference_momentum": self.ref_momentum,
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
