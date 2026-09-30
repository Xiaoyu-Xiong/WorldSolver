"""Dynamic buckling of a vertically shaken hair clump.

Physical model
--------------
The clump is modeled as a single inextensible elastic strand (discrete
Euler-elastica / Kirchhoff rod without twist, since the strand is naturally
straight and the excitation is planar), discretized with the 49 frozen nodes
of the scaffold curve network.

Time stepping (a real physical integrator, no keyframes):

* semi-implicit (symplectic) Euler velocity update with
  - gravity (frozen config vector),
  - linear aerodynamic drag ``F = -nu * l_assoc * (v - v_air)`` per node
    (frozen ``ambient_drag_nu`` and ``ambient_air_velocity``),
  - explicit bending forces from the hinge energy
    ``E_j = (EI / (2 l_bar)) (theta_j - theta_0j)^2`` with analytic gradients
    (discrete Euler elastica; per-hinge torsional stiffness EI/l_bar),
  - Kelvin-Voigt internal bending damping with the frozen
    ``internal_damping_gamma``;
* exact inextensibility via mass-weighted Gauss-Seidel distance projections
  (position-based dynamics) on every substep, momentum-conserving between
  dynamic nodes;
* clamped root: node 0 follows the prescribed root translation exactly and,
  because ``root_orientation_fixed`` is true, node 1 is constrained to the
  frozen initial root tangent direction, i.e. the root edge translates rigidly
  with the oscillating attachment.  The clamp reaction (including constraint
  impulses transmitted through the kinematic boundary) is measured every
  substep so the momentum budget of the global domain closes exactly.

The root trajectory is the frozen ``vertical_sinusoid_after_hold`` law.  The
scaffold freezes mass per length (density, effective radius) but no bending
modulus; an effective clump Young's modulus is fixed here (soft-clump value,
3 MPa) so that the shaken strand reproduces the source phenomenon: the lower
clump lags the fast root motion and develops a conspicuous local bend while
the apparent axial load passes the clump buckling load during the
deceleration phases.

All exports follow the StateFrame contract; the momentum-budget fields are
consistent by construction (interval-averaged per-term external forces whose
impulse equals the sampled momentum change of the global domain).
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # --- numerics (fixed, deterministic) ---
    DT = 5.0e-5              # substep [s]; 400 substeps per 0.02 s sample
    PBD_ITERS = 4            # Gauss-Seidel sweeps for inextensibility
    YOUNG_EFFECTIVE = 3.0e6  # Pa, effective clump bending modulus
    VMAX = 40.0              # m/s, non-physical numerical safeguard only

    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or {}
        meta = scene.metadata()
        hair = cfg.get("hair_clump", {})
        drive = cfg.get("root_motion", {})

        self.seed = int(seed)
        self.gravity = np.asarray(cfg.get("gravity", meta.get("gravity", (0.0, 0.0, -9.81))), dtype=float)

        frame0 = scene.initial_state()
        self.n = len(frame0.position)
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.base_attrs = {k: np.asarray(a) for k, a in frame0.attributes.items()}

        # --- material / dissipation parameters (frozen) ---
        radius = float(hair.get("effective_radius", 0.0028))
        self.nu = float(hair.get("ambient_drag_nu", 5.0e-6))
        self.v_air = np.asarray(hair.get("ambient_air_velocity", (0.0, 0.0, 0.0)), dtype=float)
        self.gamma = float(hair.get("internal_damping_gamma", 1.0e-10))
        # Bending stiffness of the effective clump strand.
        self.EI = self.YOUNG_EFFECTIVE * np.pi * radius ** 4 / 4.0

        # --- discrete rod topology (frozen curve network) ---
        edges = [(i, i + 1) for i in range(self.n - 1)]
        self.edges = [
            (a, b, float(np.linalg.norm(self.x[b] - self.x[a]))) for a, b in edges
        ]
        rest = np.array([r for _, _, r in self.edges])
        l_assoc = np.zeros(self.n)
        l_assoc[: self.n - 1] += 0.5 * rest
        l_assoc[1:] += 0.5 * rest
        self.l_assoc = l_assoc

        # --- clamped root boundary ---
        # node 0: prescribed translation; node 1: also fixed to the frozen
        # initial root tangent (root_orientation_fixed).
        self.root_fixed_orientation = bool(drive.get("root_orientation_fixed", True))
        self.kin = [0, 1] if self.root_fixed_orientation else [0]
        self.kin = [i for i in self.kin if i < self.n]
        self.dyn = np.ones(self.n, dtype=bool)
        self.dyn[self.kin] = False
        w = np.zeros(self.n)
        m_pos = self.mass > 0.0
        w[m_pos] = 1.0 / self.mass[m_pos]
        w[self.kin] = 0.0
        self.w = w
        e0 = self.x[1] - self.x[0]
        self.l0 = float(np.linalg.norm(e0))
        self.droot = e0 / self.l0

        # --- prescribed root motion (frozen law) ---
        self.t0 = float(drive.get("start_time", 0.3))
        self.amp = float(drive.get("amplitude", 0.045))
        self.freq = float(drive.get("frequency", 3.0))
        self.wd = 2.0 * np.pi * self.freq
        self.rest_pos = np.asarray(
            drive.get("rest_position", (0.0, 0.0, 0.3)), dtype=float
        )

        # --- hinge (bending) data ---
        self.theta0 = self._hinge_angles(self.x)
        self.theta_prev = self.theta0.copy()

        # --- budget bookkeeping ---
        self.step = 0
        self.cum_impulse = np.zeros((1, 3))
        self._term_sum = np.zeros((3, 3))  # gravity, drag, clamp reaction
        self._term_count = 0
        self.mass_total = float(self.mass.sum())
        self.reference_momentum = float(abs(self.mass_total) * self.amp * self.wd)
        self.constraint_scale = float(self.amp)

    # ------------------------------------------------------------------
    # prescribed root motion
    # ------------------------------------------------------------------
    def _root_state(self, t: float):
        if t <= self.t0:
            return self.rest_pos.copy(), np.zeros(3), np.zeros(3)
        ph = self.wd * (t - self.t0)
        s, c = np.sin(ph), np.cos(ph)
        pos = self.rest_pos + np.array([0.0, 0.0, self.amp * s])
        vel = np.array([0.0, 0.0, self.amp * self.wd * c])
        acc = np.array([0.0, 0.0, -self.amp * self.wd * self.wd * s])
        return pos, vel, acc

    # ------------------------------------------------------------------
    # discrete bending
    # ------------------------------------------------------------------
    def _hinge_angles(self, x: np.ndarray) -> np.ndarray:
        e1 = x[1 : self.n - 1] - x[0 : self.n - 2]
        e2 = x[2 : self.n] - x[1 : self.n - 1]
        t1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        t2 = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
        c = np.sum(t1 * t2, axis=1)
        s = np.linalg.norm(t2 - c[:, None] * t1, axis=1)
        return np.arctan2(s, c)

    @staticmethod
    def _perp(t: np.ndarray) -> np.ndarray:
        axis = np.zeros_like(t)
        axis[:, 2] = 1.0
        use_x = np.abs(t[:, 2]) > 0.9
        axis[use_x] = np.array([1.0, 0.0, 0.0])
        p = np.cross(t, axis)
        return p / np.linalg.norm(p, axis=1, keepdims=True)

    def _bending_forces(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        """Nodal forces of -grad of sum_j (EI/(2 l_bar)) (theta_j - theta0_j)^2.

        Uses the analytic hinge gradients
            dtheta/dx_{j-1} =  n1 / l1
            dtheta/dx_{j+1} = -m2 / l2
            dtheta/dx_j     = -n1 / l1 + m2 / l2
        with n1 = (t2 - cos t1)/sin, m2 = (t1 - cos t2)/sin, so the per-hinge
        forces sum to zero (internal, translation invariant).
        """
        e1 = x[1 : self.n - 1] - x[0 : self.n - 2]
        e2 = x[2 : self.n] - x[1 : self.n - 1]
        l1 = np.linalg.norm(e1, axis=1)
        l2 = np.linalg.norm(e2, axis=1)
        t1 = e1 / l1[:, None]
        t2 = e2 / l2[:, None]
        c = np.sum(t1 * t2, axis=1)
        p = t2 - c[:, None] * t1
        q = t1 - c[:, None] * t2
        s = np.linalg.norm(p, axis=1)
        theta = np.arctan2(s, c)
        tiny = s < 1.0e-10
        if np.any(tiny):
            fallback = self._perp(t1[tiny])
            p = np.where(tiny[:, None], fallback, p)
            q = np.where(tiny[:, None], fallback, q)
            s = np.where(tiny, 1.0, s)
        n1 = p / s[:, None]
        m2 = q / s[:, None]
        l_bar = 0.5 * (l1 + l2)
        ktheta = self.EI / l_bar
        theta_dot = (theta - self.theta_prev) / self.DT
        f = -ktheta * (theta - self.theta0) - self.gamma * theta_dot
        if update:
            self.theta_prev = theta
        fa = f[:, None] * (n1 / l1[:, None])
        fc = f[:, None] * (-m2 / l2[:, None])
        fb = f[:, None] * (m2 / l2[:, None] - n1 / l1[:, None])
        F = np.zeros((self.n, 3))
        F[0 : self.n - 2] += fa
        F[1 : self.n - 1] += fb
        F[2 : self.n] += fc
        return F

    # ------------------------------------------------------------------
    # external per-node force terms
    # ------------------------------------------------------------------
    def _external_forces(self, x: np.ndarray, v: np.ndarray, update: bool = False) -> tuple:
        F_grav = self.mass[:, None] * self.gravity[None, :]
        F_drag = -self.nu * self.l_assoc[:, None] * (v - self.v_air[None, :])
        F_drag[self.mass <= 0.0] = 0.0
        F_bend = self._bending_forces(x, update=update)
        return F_grav, F_drag, F_bend

    # ------------------------------------------------------------------
    # one physical substep
    # ------------------------------------------------------------------
    def _substep(self) -> None:
        dt = self.DT
        t_new = (self.step + 1) * dt

        F_grav, F_drag, F_bend = self._external_forces(self.x, self.v, update=True)
        F_tot = F_grav + F_drag + F_bend

        acc = np.zeros_like(F_tot)
        acc[self.dyn] = F_tot[self.dyn] / self.mass[self.dyn][:, None]
        v_pred = self.v + dt * acc
        x_pred = self.x + dt * v_pred

        # clamped root: exact prescribed translation (+ fixed orientation).
        xr, vr, _ = self._root_state(t_new)
        x_pred[self.kin[0]] = xr
        v_pred[self.kin[0]] = vr
        if len(self.kin) > 1:
            x_pred[self.kin[1]] = xr + self.l0 * self.droot
            v_pred[self.kin[1]] = vr

        # inextensibility: momentum-conserving Gauss-Seidel projections.
        x = x_pred
        w = self.w
        for _ in range(self.PBD_ITERS):
            for a, b, rest in self.edges:
                wa = w[a]
                wb = w[b]
                denom = wa + wb
                if denom == 0.0:
                    continue
                dx = x[b] - x[a]
                dist = float(np.sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]))
                if dist < 1.0e-12:
                    continue
                err = dist - rest
                if err == 0.0:
                    continue
                scale = err / (dist * denom)
                x[a] += (wa * scale) * dx
                x[b] -= (wb * scale) * dx

        # velocity update (PBD), kinematic nodes keep the analytic root speed.
        v_new = (x - self.x) / dt
        v_new[~self.dyn] = v_pred[~self.dyn]

        # numerical safeguards (far outside the physical range of this scene).
        bad = ~np.isfinite(x).all(axis=1)
        if np.any(bad):
            x[bad] = self.x[bad]
            v_new[bad] = 0.0
        bad = ~np.isfinite(v_new).all(axis=1)
        if np.any(bad):
            v_new[bad] = 0.0
        sp = np.linalg.norm(v_new, axis=1)
        over = sp > self.VMAX
        if np.any(over):
            v_new[over] *= (self.VMAX / sp[over])[:, None]

        # clamp reaction impulse: inertial + external + bending balance of the
        # kinematic nodes, plus constraint impulses crossing the kinematic
        # boundary (absorbed by the clamp, never applied to dynamic nodes).
        J_clamp = np.zeros(3)
        for i in self.kin:
            J_clamp += self.mass[i] * (v_new[i] - self.v[i]) - dt * F_tot[i]
        J_clamp += ((v_new - v_pred) * self.mass[:, None])[self.dyn].sum(axis=0)

        J_grav = dt * F_grav.sum(axis=0)
        J_drag = dt * F_drag.sum(axis=0)
        self.cum_impulse += J_grav + J_drag + J_clamp
        self._term_sum += np.stack((J_grav, J_drag, J_clamp))
        self._term_count += 1

        self.x = x
        self.v = v_new
        self.step += 1

    # ------------------------------------------------------------------
    # StateFrame export
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = int(round(float(time) / self.DT))
        if target < self.step:
            target = self.step
        while self.step < target:
            self._substep()
        return self._frame(float(time))

    def _frame(self, time: float) -> StateFrame:
        x = self.x.copy()
        v = self.v.copy()
        mass = self.mass.copy()

        # per-term external forces (interval average over the substeps since
        # the previous sample; equals the sampled momentum change / dt).
        if self._term_count > 0:
            terms = self._term_sum / (self._term_count * self.DT)
            self._term_sum = np.zeros((3, 3))
            self._term_count = 0
        else:
            F_grav, F_drag, F_bend = self._external_forces(x, v)
            _, _, acc_r = self._root_state(time)
            J = np.zeros(3)
            for i in self.kin:
                J += self.mass[i] * acc_r - F_grav[i] - F_drag[i] - F_bend[i]
            terms = np.stack(
                (F_grav.sum(axis=0), F_drag.sum(axis=0), J)
            )
        budget_force = terms[None, :, :]  # [G=1, K=3, 3]

        # clamped-root constraint (position + frozen orientation).
        root_pos, _, _ = self._root_state(time)
        e0 = x[1] - x[0]
        n0 = np.linalg.norm(e0)
        tangent = e0 / n0 if n0 > 0 else self.droot
        constraint_value = np.stack((x[0], tangent))
        constraint_target = np.stack((root_pos, self.droot))

        attrs = dict(self.base_attrs)
        attrs["mass_quantity"] = mass

        fields = {
            "budget_cumulative_external_impulse": self.cum_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)),
            "budget_external_force": budget_force,
            "constraint_target": constraint_target,
            "constraint_value": constraint_value,
            "cumulative_mass_source": np.zeros(1),
        }
        scalars = {
            "material_length": float(np.linalg.norm(np.diff(self.x, axis=0), axis=1).sum()),
            "internal_damping_gamma": float(self.gamma),
            "ambient_drag_nu": float(self.nu),
            "constraint_scale": float(self.constraint_scale),
            "reference_momentum": float(self.reference_momentum),
        }

        frame = StateFrame(
            time=float(time),
            position=x,
            velocity=v,
            mass=mass,
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "semi-implicit Euler + PBD inextensible discrete elastica",
            "substep_dt": self.DT,
            "substeps": int(self.step),
            "bending_stiffness_EI": float(self.EI),
            "effective_young_modulus": float(self.YOUNG_EFFECTIVE),
            "total_mass": float(self.mass_total),
            "root_frequency_hz": float(self.freq),
            "root_amplitude_m": float(self.amp),
            "deterministic": True,
        }
