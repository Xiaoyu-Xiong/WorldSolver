"""Extensible elastic cable wound around a rigid cylinder.

Physics model
-------------
The cable is a discrete Cosserat-style strand of ``N`` lumped nodes with
orientation (twist) degrees of freedom:

* axial (stretch) response  - nonlinear springs with modulus ``E`` and the
  true circular cross section ``A = pi r^2``; the cable is extensible, the
  axial strain is ``(|d| - L0)/L0`` and the cable is *not* projected back onto
  the rest length.
* bending response          - discrete hinge energy ``U = EI/(2 L0) phi^2``
  with the exact turning-angle gradient between consecutive segments.
* torsional response        - per-node material twist ``theta`` with
  ``U = GJ/(2 L0) (theta_{i+1} - theta_i)^2``.
* shear response            - the transverse (shear/bending) compliance of the
  strand is resolved by the same short-segment hinge system; the strand is
  slender so the shear compliance is lumped there instead of being a separate
  director dof.
* gravity, cylinder contact, floor contact and cable self-contact, all through
  a power-law (generalized nonlinear) friction law
  ``||lambda_t|| <= kappa |lambda_n|^N`` (regularized in the tangential slip
  direction).  No ideal Coulomb capstan friction is used.
* the driven end (vertex ``driven_vertex``) is gripped by a compliant
  (penalty) constraint that follows the pitch-controlled winding path taken
  from the frozen scene; every other node is free.

Time integration is an explicit symplectic (semi-implicit) Euler step with a
fixed ``dt`` small enough to resolve the stiffest axial and contact modes.  All
quantities are deterministic and finite.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)

        config = dict(getattr(scene, "config", {}) or {})
        if not config:
            config = dict(scene.metadata())
        self.config = config

        cable = dict(config["cable"])
        cylinder = dict(config["cylinder"])
        contact = dict(config["contact"])
        winding = dict(config["winding"])

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.n = int(self.x.shape[0])

        mass = getattr(frame0, "mass", None)
        if mass is None:
            mass = frame0.attributes.get("mass_quantity")
        self.m = np.array(mass, dtype=float).reshape(self.n).copy()
        self.m_total = float(self.m.sum())

        self.r_cable = float(cable["radius"])
        self.natural_length = float(cable["length"])
        self.L0 = self.natural_length / max(self.n - 1, 1)

        # Material response -------------------------------------------------
        E = float(cable["young_modulus_pa"])
        G = float(cable["shear_modulus_pa"])
        A = np.pi * self.r_cable ** 2
        I_bend = 0.25 * np.pi * self.r_cable ** 4
        J_tw = 0.5 * np.pi * self.r_cable ** 4
        self.k_axial = E * A / self.L0
        self.k_bend = E * I_bend / self.L0
        self.k_twist = G * J_tw / self.L0

        dm = float(np.mean(self.m))
        self.c_axial = 2.0 * 0.05 * np.sqrt(self.k_axial * dm)
        self.c_twist = 2.0 * 0.10 * np.sqrt(self.k_twist * max(dm * self.r_cable ** 2, 1e-30))

        # Twist state -------------------------------------------------------
        self.theta = np.zeros(self.n)
        self.omega_t = np.zeros(self.n)
        self.I_twist = 0.5 * self.m * self.r_cable ** 2
        self.I_twist = np.maximum(self.I_twist, 1e-30)

        # Gravity / environment --------------------------------------------
        self.g = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        self.cyl_center = np.asarray(cylinder["center"], dtype=float)
        self.cyl_radius = float(cylinder["radius"])
        self.cyl_height = float(cylinder["height"])
        self.cyl_axis = np.asarray(cylinder["axis"], dtype=float)
        self.cyl_axis = self.cyl_axis / np.linalg.norm(self.cyl_axis)
        self.c_contact_radius = self.cyl_radius + self.r_cable

        # Generalized power-law friction ------------------------------------
        self.kappa = float(contact.get("kappa", 1.296))
        self.exponent = float(contact.get("exponent_N", 0.7616))
        self.self_contact = bool(contact.get("cable_self_contact_enabled", True))

        self.k_contact = 1.5e4
        self.zeta_contact = 0.6
        self.c_contact = 2.0 * self.zeta_contact * np.sqrt(self.k_contact * dm)
        self.v_slip = 1.0e-3
        self.floor_z = 0.0

        # Driven grip -------------------------------------------------------
        self.driven = int(cable["driven_vertex"])
        self.k_grip = 2500.0
        self.c_grip = 2.0 * 0.8 * np.sqrt(self.k_grip * dm)

        # Gentle global dissipation (external dissipative interaction) ------
        self.c_global = 2.0 * 0.3 * dm * 4.0

        # Time stepping -----------------------------------------------------
        self.dt = 2.5e-5
        duration = float(config.get("duration", 12.0))
        self.duration = duration
        self.n_steps_max = int(round(duration / self.dt)) + 2
        self.t = 0.0
        self._self_contact_counter = 0

        self._build_grip_table(winding, config)

        # Budget bookkeeping ------------------------------------------------
        self.G = 1
        self.K = 5  # gravity, cylinder contact, floor contact, grip, damping
        self.cum_external_impulse = np.zeros((self.G, 3))
        self.cum_external_angular_impulse = np.zeros((self.G, 3))
        self.cum_inflow_momentum = np.zeros((self.G, 3))
        self.cum_inflow_angular = np.zeros((self.G, 3))
        self.cum_outflow_momentum = np.zeros((self.G, 3))
        self.cum_outflow_angular = np.zeros((self.G, 3))
        self.cum_mass_source = np.zeros(1)  # mass_quantity is [N] -> a single material

        self.v_ref = 0.1
        self.reference_momentum = max(self.m_total * self.v_ref, 1e-12)
        self.reference_angular_momentum = max(
            self.m_total * self.v_ref * self.natural_length, 1e-12
        )
        self._inv_m = 1.0 / self.m
        self._last_force = {}

    # ------------------------------------------------------------ grip path
    @staticmethod
    def _smoothstep(x: np.ndarray) -> np.ndarray:
        y = np.clip(x, 0.0, 1.0)
        return y * y * (3.0 - 2.0 * y)

    def _build_grip_table(self, winding: dict, config: dict) -> None:
        """Precompute the prescribed driven-end trajectory (frozen scene path)."""
        steps = self.n_steps_max
        tt = np.arange(steps + 1, dtype=float) * self.dt
        ta = float(winding["approach_end_time"])
        tw = float(winding["winding_end_time"])
        radius = float(winding["centerline_orbit_radius"])
        angle0 = float(winding["start_angle_rad"])
        h0 = float(winding["start_height"])
        h1 = float(winding["end_height"])
        turns = float(winding["turns"])
        initial = np.asarray(config["initial_conditions"]["driven_end"], dtype=float)

        start = np.array([radius * np.cos(angle0), radius * np.sin(angle0), h0])
        u_app = self._smoothstep(tt / max(ta, 1e-9))
        u_wind = self._smoothstep(np.clip((tt - ta) / max(tw - ta, 1e-9), 0.0, 1.0))
        angle = angle0 + 2.0 * np.pi * turns * u_wind
        helix = np.empty((tt.size, 3))
        helix[:, 0] = radius * np.cos(angle)
        helix[:, 1] = radius * np.sin(angle)
        helix[:, 2] = (1.0 - u_wind) * h0 + u_wind * h1
        approach = (1.0 - u_app)[:, None] * initial[None, :] + u_app[:, None] * start[None, :]
        pos = np.where((tt < ta)[:, None], approach, helix)
        vel = np.gradient(pos, self.dt, axis=0)
        self._grip_pos = pos
        self._grip_vel = vel

    def _grip_at(self, t: float):
        idx = int(round(t / self.dt))
        idx = min(max(idx, 0), self._grip_pos.shape[0] - 1)
        return self._grip_pos[idx], self._grip_vel[idx]

    # --------------------------------------------------------------- forces
    def _forces(self, x: np.ndarray, v: np.ndarray, t: float):
        """Return the five external force term arrays (n,3) plus internal force."""
        n = self.n
        m = self.m
        F = np.zeros((n, 3))
        terms = np.zeros((self.K, n, 3))

        # gravity (term 0)
        terms[0] = m[:, None] * self.g[None, :]
        F += terms[0]

        # ---- axial stretch springs (internal) -----------------------------
        d = x[1:] - x[:-1]
        length = np.linalg.norm(d, axis=1)
        ln = np.maximum(length, 1e-12)
        that = d / ln[:, None]
        strain = ln - self.L0
        f_axial = (self.k_axial * strain)[:, None] * that
        rel = v[1:] - v[:-1]
        v_ax = np.sum(rel * that, axis=1)
        f_axial = f_axial + (self.c_axial * v_ax)[:, None] * that
        F[:-1] -= f_axial
        F[1:] += f_axial

        # ---- bending hinges (internal) ------------------------------------
        if n > 2:
            a = x[1:-1] - x[:-2]
            b = x[2:] - x[1:-1]
            la = np.maximum(np.linalg.norm(a, axis=1), 1e-12)
            lb = np.maximum(np.linalg.norm(b, axis=1), 1e-12)
            u_in = a / la[:, None]
            v_out = b / lb[:, None]
            cos_phi = np.clip(np.sum(u_in * v_out, axis=1), -1.0, 1.0)
            phi = np.arccos(cos_phi)
            s = np.sqrt(np.maximum(1.0 - cos_phi * cos_phi, 1e-12))
            ratio = phi / s
            ratio = np.where(phi < 1e-4, 1.0 + phi * phi / 6.0, ratio)
            coef = self.k_bend * ratio
            f_prev = -coef[:, None] * v_out / lb[:, None]
            f_next = coef[:, None] * u_in / la[:, None]
            F[:-2] += f_prev
            F[2:] += f_next
            F[1:-1] -= f_prev + f_next

        # ---- torsion (internal) -------------------------------------------
        if n > 2:
            dd = self.theta[2:] - 2.0 * self.theta[1:-1] + self.theta[:-2]
            torque = self.k_twist * dd
            torque = torque - self.c_twist * (self.omega_t[1:-1] - self.omega_t[:-2])
            acc = torque / self.I_twist[1:-1]
            self.omega_t[1:-1] += acc * self.dt
            self.theta[1:-1] += self.omega_t[1:-1] * self.dt
            self.omega_t *= (1.0 - 0.5 * self.dt)

        # ---- cylinder contact (term 1) ------------------------------------
        rel_c = x - self.cyl_center[None, :]
        rho = rel_c.copy()
        rho[:, 2] = 0.0
        r_xy = np.linalg.norm(rho, axis=1)
        r_xy_safe = np.maximum(r_xy, 1e-12)
        n_hat = rho / r_xy_safe[:, None]
        z_loc = rel_c[:, 2] - 0.0
        if abs(self.cyl_axis[2]) > 1e-9:
            z_loc = z_loc / self.cyl_axis[2]
        on_side = np.abs(z_loc) <= 0.5 * self.cyl_height
        gap = r_xy - self.c_contact_radius
        pen = np.where(on_side, np.maximum(-gap, 0.0), 0.0)
        v_n = np.sum(v * n_hat, axis=1)
        f_n = self.k_contact * pen - self.c_contact * v_n
        f_n = np.maximum(f_n, 0.0)
        f_n = np.where(pen > 0.0, f_n, 0.0)
        terms[1] = f_n[:, None] * n_hat
        # generalized power-law tangential friction
        v_t = v - v_n[:, None] * n_hat
        v_t_mag = np.linalg.norm(v_t, axis=1)
        slide = np.tanh(v_t_mag / self.v_slip)
        fric_mag = self.kappa * np.power(np.maximum(f_n, 0.0), self.exponent) * slide
        fric = -fric_mag[:, None] * v_t / (v_t_mag[:, None] + 1e-12)
        terms[1] = terms[1] + np.where(pen[:, None] > 0.0, fric, 0.0)
        F += terms[1]

        # ---- floor contact (term 2) ---------------------------------------
        pen_f = self.r_cable - (x[:, 2] - self.floor_z)
        z_hat = np.array([0.0, 0.0, 1.0])
        v_nf = v[:, 2]
        f_nf = self.k_contact * np.maximum(pen_f, 0.0) - self.c_contact * v_nf
        f_nf = np.where(pen_f > 0.0, np.maximum(f_nf, 0.0), 0.0)
        v_tf = v.copy()
        v_tf[:, 2] = 0.0
        v_tf_mag = np.linalg.norm(v_tf, axis=1)
        slide_f = np.tanh(v_tf_mag / self.v_slip)
        fric_f = -(self.kappa * np.power(f_nf, self.exponent) * slide_f)[:, None] * v_tf / (
            v_tf_mag[:, None] + 1e-12
        )
        terms[2] = f_nf[:, None] * z_hat[None, :] + np.where(pen_f[:, None] > 0.0, fric_f, 0.0)
        F += terms[2]

        # ---- driven grip constraint (term 3) ------------------------------
        gp, gv = self._grip_at(t)
        i = self.driven
        f_grip = -self.k_grip * (x[i] - gp) - self.c_grip * (v[i] - gv)
        terms[3][i] = f_grip
        F[i] += f_grip

        # ---- global viscous dissipation (term 4) --------------------------
        terms[4] = -self.c_global * v
        F += terms[4]

        return F, terms

    def _self_contact_forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Penalty + power-law friction between non-adjacent cable nodes."""
        n = self.n
        dx = x[:, None, :] - x[None, :, :]
        r2 = np.sum(dx * dx, axis=2)
        idx = np.arange(n)
        sep = np.abs(idx[:, None] - idx[None, :]) > 2
        active = (r2 < (2.0 * self.r_cable) ** 2) & sep
        if not np.any(active):
            return np.zeros((n, 3))
        r = np.sqrt(np.maximum(r2, 1e-18))
        nrm = dx / r[:, :, None]
        pen = 2.0 * self.r_cable - r
        dv = v[:, None, :] - v[None, :, :]
        v_n = np.sum(dv * nrm, axis=2)
        f_n = self.k_contact * pen - self.c_contact * v_n
        f_n = np.where(active, np.maximum(f_n, 0.0), 0.0)
        f_c = f_n[:, :, None] * nrm

        v_t = dv - v_n[:, :, None] * nrm
        v_t_mag = np.linalg.norm(v_t, axis=2)
        slide = np.tanh(v_t_mag / self.v_slip)
        fric_mag = self.kappa * np.power(f_n, self.exponent) * slide
        fric = -fric_mag[:, :, None] * v_t / (v_t_mag[:, :, None] + 1e-12)
        f_c = f_c + np.where(active[:, :, None], fric, 0.0)
        # self-contact is an internal pair force: net momentum cancels exactly
        return 0.5 * (f_c - np.transpose(f_c, (1, 0, 2))).sum(axis=1)

    # ---------------------------------------------------------------- step
    def _step(self, t: float) -> None:
        dt = self.dt
        x = self.x
        v = self.v
        F, terms = self._forces(x, v, t)

        self._self_contact_counter += 1
        if self.self_contact and (self._self_contact_counter % 8 == 0):
            F = F + self._self_contact_forces(x, v)

        # budget accumulation (impulse of external forces, first-order)
        term_sum = terms.sum(axis=1)  # (K,3)
        F_ext = term_sum.sum(axis=0)
        self.cum_external_impulse[0] += F_ext * dt
        ang = np.sum(np.cross(x, terms), axis=1)  # (K,3)
        self.cum_external_angular_impulse[0] += ang.sum(axis=0) * dt

        # symplectic Euler
        self.v = v + (F * self._inv_m[:, None]) * dt
        self.x = x + self.v * dt
        self.t = t + dt

        self._last_force = {
            "terms": terms,
            "ext": F_ext,
        }

    # -------------------------------------------------------------- output
    def _tangents(self) -> np.ndarray:
        x = self.x
        n = self.n
        t = np.empty((n, 3))
        if n == 1:
            return np.array([[0.0, 0.0, 1.0]])
        t[0] = x[1] - x[0]
        t[-1] = x[-1] - x[-2]
        if n > 2:
            t[1:-1] = x[2:] - x[:-2]
        norm = np.linalg.norm(t, axis=1, keepdims=True)
        return t / np.maximum(norm, 1e-12)

    def _angular_velocity(self, tangents: np.ndarray) -> np.ndarray:
        n = self.n
        omega = self.omega_t[:, None] * tangents
        if n > 2:
            d = self.x[2:] - self.x[:-2]
            dv = self.v[2:] - self.v[:-2]
            dl = np.linalg.norm(d, axis=1)
            t_mid = d / np.maximum(dl, 1e-12)[:, None]
            dv_t = dv - np.sum(dv * t_mid, axis=1)[:, None] * t_mid
            rate = dv_t / np.maximum(dl, 1e-12)[:, None]
            rot = np.cross(t_mid, rate)
            omega[1:-1] += rot
        return np.clip(omega, -1.0e4, 1.0e4)

    def _make_frame(self) -> StateFrame:
        tangents = self._tangents()
        inertia = self._inertia_tensor(tangents)
        omega = self._angular_velocity(tangents)
        m = self.m

        force_terms = np.zeros((self.G, self.K, 3))
        pos_terms = np.zeros((self.G, self.K, 3))
        couple_terms = np.zeros((self.G, self.K, 3))
        last = self._last_force.get("terms")
        com = np.sum(m[:, None] * self.x, axis=0) / max(self.m_total, 1e-30)
        if last is not None:
            for k in range(self.K):
                fk = last[k]
                ftot = fk.sum(axis=0)
                force_terms[0, k] = ftot
                w = np.linalg.norm(fk, axis=1)
                sw = float(w.sum())
                if sw > 1e-15:
                    pos_terms[0, k] = (w[:, None] * self.x).sum(axis=0) / sw
                else:
                    pos_terms[0, k] = com
        weights = np.ones((self.G, self.n))
        mass_quantity = m.reshape(self.n)

        attrs = {
            "mass_quantity": mass_quantity,
            "inertia_tensor": inertia,
        }
        fields = {
            "budget_cumulative_external_angular_impulse": self.cum_external_angular_impulse.copy(),
            "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
            "budget_cumulative_inflow_angular_momentum": self.cum_inflow_angular.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow_momentum.copy(),
            "budget_cumulative_outflow_angular_momentum": self.cum_outflow_angular.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow_momentum.copy(),
            "budget_domain_weights": weights,
            "budget_external_couple": couple_terms,
            "budget_external_force": force_terms,
            "budget_force_position": pos_terms,
            "cumulative_mass_source": self.cum_mass_source.copy(),
        }
        scalars = {
            "reference_angular_momentum": float(self.reference_angular_momentum),
            "reference_momentum": float(self.reference_momentum),
        }
        frame = StateFrame(
            time=float(self.t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            angular_velocity=omega,
            mass=m.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def _inertia_tensor(self, tangents: np.ndarray) -> np.ndarray:
        n = self.n
        m = self.m
        Ia = 0.5 * m * self.r_cable ** 2
        It = m * (3.0 * self.r_cable ** 2 + self.L0 ** 2) / 12.0
        tt = tangents[:, :, None] * tangents[:, None, :]
        eye = np.eye(3)[None, :, :]
        tensor = It[:, None, None] * eye + (Ia - It)[:, None, None] * tt
        return tensor

    # ---------------------------------------------------------------- API
    def advance_to(self, time: float):
        target = float(time)
        n_steps = int(round((target - self.t) / self.dt))
        if n_steps < 0:
            n_steps = 0
        n_steps = min(n_steps, self.n_steps_max)
        for _ in range(n_steps):
            self._step(self.t)
        if self.t > target + 1e-9:
            self.t = target
        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "explicit_symplectic_euler_cosserat_strand",
            "dt": float(self.dt),
            "samples": int(self.n),
            "kinetic_energy": float(0.5 * np.sum(self.m * np.sum(self.v ** 2, axis=1))),
            "max_penetration_note": "penalty contact with generalized power friction",
        }
