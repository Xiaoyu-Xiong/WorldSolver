"""Physically-based solver for the inclined-cloth analytic benchmark.

The cloth is a 25x25 mass sheet that starts parallel to a rigid inclined
plane, falls under gravity onto the plane and slides with Coulomb friction.

Numerical method: substepped position-based dynamics (XPBD-style).
Each substep performs
  1. semi-implicit velocity update with gravity,
  2. position prediction,
  3. Jacobi iterations of in-plane distance constraints (structural + shear)
     interleaved with non-penetration projection against the plane surface,
  4. Coulomb friction applied at contacting vertices (tangential position
     correction capped by mu times the normal correction),
  5. velocity update from positions.
This reproduces the analytical sliding acceleration
    a = g * (sin(theta) - mu * cos(theta))
after the impact transient, and the impulsive friction loss
    dv_t = mu * v_n
during the landing impact, both of which follow from Coulomb's law.
"""
from __future__ import annotations

import numpy as np


class BenchmarkSolver:
    """XPBD cloth solver with Coulomb friction contact on a rigid incline."""

    SUBSTEPS_PER_SAMPLE = 50   # substeps per sample_dt -> dt_sub = 1/3000 s
    SOLVER_ITERATIONS = 12     # constraint iterations per substep
    RELAXATION = 0.4           # Jacobi relaxation for distance constraints

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or {}
        meta = scene.metadata()
        init = scene.initial_state()

        self.seed = int(seed)
        self.x0 = np.array(init.position, dtype=float)
        self.v0 = np.array(init.velocity, dtype=float)
        self.mass = np.array(init.mass, dtype=float)
        self.uv = np.array(init.attributes["uv"], dtype=float)
        self.n_vertices = len(self.x0)
        self.inv_mass = 1.0 / self.mass

        grid = meta["topology"]["grid_shape"]
        self.nx, self.ny = int(grid[0]), int(grid[1])
        side = float(cfg.get("cloth", {}).get("side_length", 1.2))
        self.side_length = side

        # gravity
        self.gravity = np.asarray(meta.get("gravity", (0.0, 0.0, -9.81)), dtype=float)

        # incline geometry (25 deg ramp, rotation about y)
        incl = cfg.get("incline", {})
        angle = np.deg2rad(float(incl.get("angle_degrees", 25.0)))
        self.theta = angle
        # ramp axes: u uphill, v cross-slope, n outward normal
        self.ax_u = np.array([np.cos(angle), 0.0, np.sin(angle)])
        self.ax_v = np.array([0.0, 1.0, 0.0])
        self.ax_n = np.array([-np.sin(angle), 0.0, np.cos(angle)])
        self.downhill = -self.ax_u
        center = np.asarray(incl.get("plane_translation", (0.0, 0.0, 0.38)), dtype=float)
        half_t = float(incl.get("plane_half_thickness", 0.04))
        # point on the top surface of the ramp
        self.plane_point = center + half_t * self.ax_n

        contact_cfg = cfg.get("contact", {})
        self.contact_thickness = float(contact_cfg.get("thickness", 0.004))
        self.cloth_name = str(meta.get("cloth_name", "inclined_cloth"))

        # friction
        self.mu = float(meta.get("coulomb_friction", 0.2))

        # initial normal gap of the cloth above the ramp surface
        gaps0 = self._gap(self.x0)
        self.initial_gap = float(np.mean(gaps0))

        # diagnostics: analytical friction solution
        diag = meta.get("diagnostics", {}) or {}
        ana = diag.get("analytical_friction_solution", {}) or {}
        g = float(np.linalg.norm(self.gravity))
        self.g_mag = g
        self.a_slide = float(ana.get("acceleration_magnitude",
                                     g * max(0.0, np.sin(angle) - self.mu * np.cos(angle))))
        self.g_t = g * np.sin(angle)          # downslope gravity component
        self.g_n = g * np.cos(angle)          # into-plane gravity component
        # analytic landing transient
        self.t_fall = float(np.sqrt(2.0 * max(self.initial_gap, 1e-9) / self.g_n))
        self.v_impact_n = self.g_n * self.t_fall
        self.v_t_at_impact = self.g_t * self.t_fall - self.mu * self.v_impact_n
        self.disp_at_impact = 0.5 * self.g_t * self.t_fall ** 2

        # distance constraints (structural + shear) on the regular grid
        self._build_springs(side)

        # time stepping state
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 60.0))
        self.dt = self.sample_dt / self.SUBSTEPS_PER_SAMPLE
        self._step = 0
        self.x = self.x0.copy()
        self.v = self.v0.copy()

        self._initial_frame_scalars = dict(init.scalars)

    # ------------------------------------------------------------- geometry
    def _gap(self, p: np.ndarray) -> np.ndarray:
        """Signed distance from cloth points to the ramp top surface."""
        return (p - self.plane_point) @ self.ax_n

    def _build_springs(self, side: float) -> None:
        nx, ny = self.nx, self.ny
        idx = np.arange(nx * ny).reshape(ny, nx)
        pairs = []
        # structural
        a = idx[:, :-1].ravel(); b = idx[:, 1:].ravel()
        pairs.append(np.stack([a, b], axis=1))
        a = idx[:-1, :].ravel(); b = idx[1:, :].ravel()
        pairs.append(np.stack([a, b], axis=1))
        # shear
        a = idx[:-1, :-1].ravel(); b = idx[1:, 1:].ravel()
        pairs.append(np.stack([a, b], axis=1))
        a = idx[:-1, 1:].ravel(); b = idx[1:, :-1].ravel()
        pairs.append(np.stack([a, b], axis=1))
        self.spring = np.concatenate(pairs, axis=0)
        self.spring_ia = self.spring[:, 0]
        self.spring_ib = self.spring[:, 1]
        d = self.x0[self.spring_ib] - self.x0[self.spring_ia]
        self.spring_rest = np.linalg.norm(d, axis=1)

    # ----------------------------------------------------------- simulation
    def _substep(self) -> None:
        dt = self.dt
        x_prev = self.x
        # 1./2. semi-implicit velocity update and position prediction
        v = self.v + self.gravity * dt
        p = x_prev + v * dt
        p_pred = p.copy()

        ia, ib = self.spring_ia, self.spring_ib
        rest = self.spring_rest
        relax = self.RELAXATION
        n = self.ax_n
        mu = self.mu

        # 3. constraint iterations
        for _ in range(self.SOLVER_ITERATIONS):
            # Jacobi distance-constraint projection (equal masses)
            d = p[ib] - p[ia]
            length = np.linalg.norm(d, axis=1)
            length = np.where(length < 1e-12, 1e-12, length)
            corr = ((length - rest) / length) * 0.5
            delta = np.zeros_like(p)
            np.add.at(delta, ia, d * corr[:, None])
            np.add.at(delta, ib, -d * corr[:, None])
            p += relax * delta
            # non-penetration against the ramp surface
            s = (p - self.plane_point) @ n
            bad = s < 0.0
            if np.any(bad):
                p[bad] -= s[bad, None] * n[None, :]

        # 4. Coulomb friction at vertices that were pushed out of the plane
        s_pred = (p_pred - self.plane_point) @ n
        touching = s_pred < 0.0
        if np.any(touching):
            corr_n = np.einsum("ij,j->i", p - p_pred, n)  # normal correction (>0)
            corr_n = np.where(touching, np.maximum(corr_n, 0.0), 0.0)
            dp = p - x_prev
            dp_n = (dp @ n)[:, None] * n[None, :]
            d_t = dp - dp_n                       # tangential displacement
            lt = np.linalg.norm(d_t, axis=1)
            limit = mu * corr_n                    # Coulomb cap
            stick = touching & (lt <= limit)
            slide = touching & (lt > limit) & (lt > 1e-15)
            p[stick] -= d_t[stick]
            scale = np.zeros_like(lt)
            scale[slide] = limit[slide] / lt[slide]
            p -= d_t * scale[:, None]

        # 5. velocity update from positions
        self.v = (p - x_prev) / dt
        self.x = p
        self._step += 1

    # ------------------------------------------------------------ analytics
    def _analytic(self, t: float):
        """Analytical downslope displacement/velocity (drop + frictional slide)."""
        if t <= self.t_fall:
            return 0.5 * self.g_t * t * t, self.g_t * t
        tau = t - self.t_fall
        disp = self.disp_at_impact + self.v_t_at_impact * tau + 0.5 * self.a_slide * tau * tau
        vel = self.v_t_at_impact + self.a_slide * tau
        return disp, vel

    # ------------------------------------------------------------- sampling
    def _deformation_gradient(self, p: np.ndarray) -> np.ndarray:
        """Per-vertex 2x2 material deformation gradient dP/d(U,V) in ramp axes."""
        nx, ny = self.nx, self.ny
        g = p.reshape(ny, nx, 3)
        du = self.side_length / (nx - 1) if nx > 1 else 1.0
        dv = self.side_length / (ny - 1) if ny > 1 else 1.0
        dPdV, dPdU = np.gradient(g, dv, du, axis=(0, 1))
        eu, ev = self.ax_u, self.ax_v
        F = np.empty((ny, nx, 2, 2))
        F[..., 0, 0] = dPdU @ eu
        F[..., 0, 1] = dPdV @ eu
        F[..., 1, 0] = dPdU @ ev
        F[..., 1, 1] = dPdV @ ev
        return F.reshape(nx * ny, 2, 2)

    def _frame(self, time: float):
        x, v = self.x, self.v
        gaps = self._gap(x)
        disp = float(np.mean((x - self.x0) @ self.downhill))
        vel = float(np.mean(v @ self.downhill))
        a_disp, a_vel = self._analytic(time)
        disp_err = abs(disp - a_disp)
        vel_err = abs(vel - a_vel)

        attrs = {
            "uv": self.uv.copy(),
            "cloth_vertex": np.ones(self.n_vertices),
            "deformation_gradient": self._deformation_gradient(x),
            "mass_quantity": self.mass.copy(),
        }
        fields = {
            "constraint_target": np.zeros(self.n_vertices),
            "constraint_value": gaps.copy(),
            "contact_signed_distance": gaps.copy(),
            "cumulative_mass_source": np.array([0.0]),
        }
        scalars = {
            "friction_coefficient": float(self.mu),
            "constraint_residual": float(max(0.0, -float(gaps.min()))),
            "constraint_scale": float(self.initial_gap),
            "contact_length_scale": float(self.contact_thickness),
            "analytical_displacement_error": float(disp_err),
            "analytical_velocity_error": float(vel_err),
            "downslope_displacement": disp,
            "downslope_velocity": vel,
        }
        from worldsolver.contracts import StateFrame

        frame = StateFrame(
            float(time),
            x.copy(),
            v.copy(),
            mass=self.mass.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        if target < self._step:
            # re-run from the initial state for determinism
            self.x = self.x0.copy()
            self.v = self.v0.copy()
            self._step = 0
        while self._step < target:
            self._substep()
        return self._frame(float(time))

    def diagnostics(self) -> dict:
        gaps = self._gap(self.x)
        disp = float(np.mean((self.x - self.x0) @ self.downhill))
        a_disp, a_vel = self._analytic(self._step * self.dt)
        return {
            "substep_dt": self.dt,
            "steps": self._step,
            "min_gap": float(gaps.min()),
            "max_penetration": float(max(0.0, -float(gaps.min()))),
            "downslope_displacement": disp,
            "analytical_displacement": a_disp,
            "analytical_velocity": a_vel,
            "friction_coefficient": self.mu,
            "sliding_acceleration": self.a_slide,
        }
