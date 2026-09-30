"""Physically-based solver for the elastic-rod ground stick-slip scene.

Model
-----
The volumetric rod (60 surface vertices, 10 rings x 6 sides) is simulated as a
3-D mass-spring lattice: ring hoop springs, ring diagonals (cross-section
rigidity), axial springs, and inter-ring shear diagonals.  This lattice is a
standard discrete discretization of a thin elastic solid and reproduces
axial, bending, and torsional resistance of the rod.

* Gravity acts on every vertex.
* The driven top ring is kinematically prescribed: a smooth press ramp
  (t in [0, 0.8] s, dz = -0.05 m) followed by a constant sideways drag
  velocity (t in [0.8, 8] s, vx = 0.22 m/s).
* The ground (top surface z = 0, modeled with a thin compliant layer of
  1 mm) exerts a penalty normal force plus regularized Coulomb friction
  (mu_s = mu_d = 0.35) on every rod vertex -- frozen candidate contact set
  C = N, activation reported per frame.  The friction law produces genuine
  stick (tangential velocity below tolerance) and slip (saturated force)
  so the lower end catches, the rod bends, and the lower end slips forward
  repeatedly.
* Time integration: symplectic Euler at dt = 2.5e-4 s with light spring and
  global damping for numerical stability.

All required raw trajectory fields are exported with the frozen shapes:
budgets use G = 1 global rod domain with K = 3 external-force terms
(gravity, ground contact traction, prescribed-drive reaction).
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # --- numerical parameters -------------------------------------------------
    DT = 2.5e-4          # integration step (s)
    KC = 1500.0          # ground normal penalty stiffness (N/m)
    CC = 0.30            # ground normal damping (N/(m/s))
    Z_CONTACT = 0.001    # compliant ground layer thickness (m)
    EPS_V = 5.0e-4       # friction regularization velocity (m/s)
    C_SPRING = 0.05      # axial spring damping
    C_GLOBAL = 0.01      # global velocity damping
    K_AX = 60.0          # axial (inter-ring) springs
    K_SH = 25.0          # shear diagonals
    K_RING = 80.0        # hoop springs
    K_DIAG = 80.0        # cross-section diagonals

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame0 = scene.initial_state()
        x0 = np.asarray(frame0.position, dtype=float)
        m0 = np.asarray(frame0.mass, dtype=float)
        attrs0 = {k: np.asarray(v, dtype=float) for k, v in frame0.attributes.items()}
        n = len(x0)

        # ---- canonical ordering: ring index k (by z), side j (by angle) ------
        zr = np.round(x0[:, 2], 9)
        ang = np.arctan2(x0[:, 1], x0[:, 0])
        order = np.lexsort((ang, zr))          # canon c -> original order[c]
        rings = len(np.unique(zr))
        sides = n // rings
        inv = np.empty(n, dtype=int)           # original o -> canon inv[o]
        inv[order] = np.arange(n)

        X = x0[order].copy()
        M = m0[order].copy()
        TOP = attrs0.get("driven_top", np.zeros(n))[order] > 0.5

        # Tiny deterministic bias so buckling prefers the drag (+x) direction.
        rng = np.random.default_rng(seed)
        k_ring = np.arange(n) // sides
        X[:, 0] += 0.002 * np.sin(np.pi * k_ring / (rings - 1))
        X += rng.normal(0.0, 1e-6, size=X.shape)

        # ---- spring lattice --------------------------------------------------
        si, sj, sk = [], [], []
        for k in range(rings):
            for j in range(sides):
                a = k * sides + j
                si.append(a); sj.append(k * sides + (j + 1) % sides); sk.append(self.K_RING)
                si.append(a); sj.append(k * sides + (j + 2) % sides); sk.append(self.K_DIAG)
                if k + 1 < rings:
                    si.append(a); sj.append((k + 1) * sides + j); sk.append(self.K_AX)
                    si.append(a); sj.append((k + 1) * sides + (j + 1) % sides); sk.append(self.K_SH)
                    si.append(a); sj.append((k + 1) * sides + (j - 1) % sides); sk.append(self.K_SH)
        self.SI = np.asarray(si, dtype=int)
        self.SJ = np.asarray(sj, dtype=int)
        self.SK = np.asarray(sk, dtype=float)
        d0 = X[self.SI] - X[self.SJ]
        self.REST = np.linalg.norm(d0, axis=1)

        # ---- deformation-gradient elements: 9 segments between ring pairs ----
        self.segments = []
        for k in range(rings - 1):
            idx_a = np.arange(k * sides, (k + 1) * sides)
            idx_b = np.arange((k + 1) * sides, (k + 2) * sides)
            pa = X[idx_a]
            mid = 0.5 * (pa.mean(axis=0) + X[idx_b].mean(axis=0))
            P = np.vstack([pa, mid])                       # (sides+1, 3)
            Pc = P - P.mean(axis=0)
            self.segments.append((idx_a, idx_b, np.linalg.pinv(Pc)))

        # ---- prescribed top motion --------------------------------------------
        bc = cfg["boundary_conditions"]["driven_top"]
        self.press_t0, self.press_t1 = map(float, bc["press_interval"])
        self.press_disp = np.asarray(bc["press_displacement"], dtype=float)
        self.drag_t0, self.drag_t1 = map(float, bc["drag_interval"])
        self.drag_vel = np.asarray(bc["drag_velocity"], dtype=float)
        self.X_TOP0 = X[TOP].copy()
        self.TOP_IDX = np.nonzero(TOP)[0]

        self.mu = float(cfg["contact"]["ground_friction_coefficient"])
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.duration = float(cfg["duration"])

        # ---- state -------------------------------------------------------------
        self.X = X
        self.V = np.zeros_like(X)
        self.M = M
        self.t = 0.0
        self.order = order
        self.inv = inv
        self.n = n
        self.rings = rings
        self.sides = sides
        self.attrs0 = {k: v[order] for k, v in attrs0.items()}

        self.impulse = np.zeros(3)                       # cumulative external impulse [G=1,3]
        self.mass_source = np.zeros(1)
        self.steps = 0

        # frozen reference scales
        total_mass = float(M.sum())
        self.ref_force = total_mass * float(np.linalg.norm(self.gravity))
        self.ref_velocity = float(np.linalg.norm(self.drag_vel))
        self.ref_momentum = total_mass * max(self.ref_velocity, 1e-6)
        self.stick_tol = 1.0e-3

        # evaluate contact/force state once for the t = 0 sample frame
        self._evaluate_report(np.zeros(3))

    # ------------------------------------------------------------------ motion
    def _top_motion(self, t: float):
        """Prescribed top displacement and velocity at time t."""
        if self.press_t1 > self.press_t0:
            tau = min(max((t - self.press_t0) / (self.press_t1 - self.press_t0), 0.0), 1.0)
        else:
            tau = 1.0 if t >= self.press_t1 else 0.0
        s = tau * tau * (3.0 - 2.0 * tau)                     # smoothstep
        ds = 6.0 * tau * (1.0 - tau) / max(self.press_t1 - self.press_t0, 1e-12)
        if tau <= 0.0 or tau >= 1.0:
            ds = 0.0
        disp = self.press_disp * s
        vel = self.press_disp * ds
        tg = min(max(t, self.drag_t0), self.drag_t1) - self.drag_t0
        disp = disp + self.drag_vel * tg
        if self.drag_t0 <= t < self.drag_t1:
            vel = vel + self.drag_vel
        return disp, vel

    # ------------------------------------------------------------------ forces
    def _spring_forces(self, X, V, F):
        d = X[self.SI] - X[self.SJ]
        L = np.linalg.norm(d, axis=1)
        L = np.maximum(L, 1e-12)
        dirn = d / L[:, None]
        rel = ((V[self.SI] - V[self.SJ]) * dirn).sum(axis=1)
        fmag = -(self.SK * (L - self.REST) + self.C_SPRING * rel)
        f = fmag[:, None] * dirn
        np.add.at(F, self.SI, f)
        np.add.at(F, self.SJ, -f)

    def _contact_forces(self, X, V):
        pen = self.Z_CONTACT - X[:, 2]
        act = pen > 0.0
        N = self.KC * pen + self.CC * np.maximum(0.0, -V[:, 2])
        N = np.where(act, np.maximum(N, 0.0), 0.0)
        vt = V[:, :2].copy()
        speed = np.sqrt((vt * vt).sum(axis=1))
        denom = np.sqrt(speed * speed + self.EPS_V * self.EPS_V)
        coef = np.where(act, self.mu * N / np.maximum(denom, 1e-15), 0.0)
        Ft = -coef[:, None] * vt
        return act, N, Ft

    def _evaluate_report(self, drive_total):
        """Contact / force report at the current state (for frame output)."""
        act, N, Ft = self._contact_forces(self.X, self.V)
        F = np.zeros_like(self.X)
        F += self.M[:, None] * self.gravity
        self._spring_forces(self.X, self.V, F)
        F[:, 2] += N
        F[:, :2] += Ft
        F -= self.C_GLOBAL * self.V
        grav_total = (self.M[:, None] * self.gravity).sum(axis=0)
        contact_total = np.array([Ft[:, 0].sum(), Ft[:, 1].sum(), N.sum()])
        self.report = {
            "active": act.astype(float),
            "N": N,
            "Ft": Ft,
            "budget": np.stack([grav_total, contact_total, drive_total])[None, :, :],
        }

    # ------------------------------------------------------------------ stepping
    def advance_to(self, time: float):
        time = float(time)
        n_sub = int(round((time - self.t) / self.DT)) if time > self.t else 0
        drive_total = self.report["budget"][0, 2].copy() if self.steps == 0 else self._last_drive
        for _ in range(max(n_sub, 0)):
            t_old = self.t
            t_new = t_old + self.DT
            X, V, M = self.X, self.V, self.M

            F = np.zeros_like(X)
            F += M[:, None] * self.gravity
            self._spring_forces(X, V, F)
            act, N, Ft = self._contact_forces(X, V)
            F[:, 2] += N
            F[:, :2] += Ft
            F -= self.C_GLOBAL * V

            V_old_top = V[self.TOP_IDX].copy()
            V += self.DT * F / M[:, None]
            X += self.DT * V

            disp, vel = self._top_motion(t_new)
            a_req = (vel - V_old_top) / self.DT
            f_drive = M[self.TOP_IDX, None] * a_req - F[self.TOP_IDX]
            drive_total = f_drive.sum(axis=0)

            X[self.TOP_IDX] = self.X_TOP0 + disp
            V[self.TOP_IDX] = vel

            grav_total = (M[:, None] * self.gravity).sum(axis=0)
            contact_total = np.array([Ft[:, 0].sum(), Ft[:, 1].sum(), N.sum()])
            ext_total = grav_total + contact_total + drive_total
            self.impulse += ext_total * self.DT

            self.report = {
                "active": act.astype(float),
                "N": N,
                "Ft": Ft,
                "budget": np.stack([grav_total, contact_total, drive_total])[None, :, :],
            }
            self._last_drive = drive_total
            self.t = t_new
            self.steps += 1

        return self._make_frame(time)

    # ------------------------------------------------------------------ output
    def _def_gradients(self):
        Fs = np.empty((len(self.segments), 3, 3))
        for e, (ia, ib, pinv_ref) in enumerate(self.segments):
            pa = self.X[ia]
            mid = 0.5 * (pa.mean(axis=0) + self.X[ib].mean(axis=0))
            P = np.vstack([pa, mid])
            Pc = P - P.mean(axis=0)
            Fs[e] = (pinv_ref @ Pc).T
        return Fs

    def _make_frame(self, t):
        inv = self.inv
        pos = self.X[inv]
        vel = self.V[inv]
        mass = self.M[inv]
        rep = self.report
        C = self.n
        fields = {
            "budget_cumulative_external_impulse": self.impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)),
            "budget_external_force": rep["budget"].copy(),
            "cumulative_mass_source": self.mass_source.copy(),
            "element_deformation_gradient": self._def_gradients(),
            "friction_active_mask": rep["active"][inv].copy(),
            "friction_dynamic_coefficient": np.full(C, self.mu),
            "friction_normal": np.tile(np.array([0.0, 0.0, 1.0]), (C, 1)),
            "friction_normal_force": rep["N"][inv].copy(),
            "friction_relative_velocity": vel.copy(),
            "friction_static_coefficient": np.full(C, self.mu),
            "friction_tangential_force": rep["Ft"][inv].copy(),
        }
        attrs = {k: v[inv].copy() for k, v in self.attrs0.items()}
        attrs["mass_quantity"] = mass.copy()
        scalars = {
            "friction_stick_speed_tolerance": self.stick_tol,
            "reference_force": self.ref_force,
            "reference_momentum": self.ref_momentum,
            "reference_velocity": self.ref_velocity,
        }
        frame = StateFrame(
            float(t),
            pos,
            vel,
            mass=mass,
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "steps": self.steps,
            "time": self.t,
            "cumulative_external_impulse": self.impulse.tolist(),
        }
