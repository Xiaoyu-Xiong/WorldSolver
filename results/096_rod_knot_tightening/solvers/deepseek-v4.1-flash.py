"""Deterministic physics solver for scene 096: a frictionless elastic rod knot.

The runner calls::

    solver.initialize(scene, seed)
    frames = [solver.advance_to(float(t)) for t in scene.sample_times()]

Physical model
--------------
The strand is a discrete inextensible (Kirchhoff-type) elastic rod:

* ``N = 161`` equal point masses carrying ``total_mass`` (gravity is zero),
  supplying the per-sample mass binding used by the trajectory contract;
* ``N - 1`` edge distance constraints with uniform rest length
  ``rest_length / (N - 1)`` enforced by a momentum-conserving Jacobi position
  projection (zero-compliance XPBD relaxation, i.e. inextensible);
* discrete bending resistance: every interior vertex stores the elastic
  energy ``EI * theta**2 / (2 h)`` of the turning angle ``theta`` between its
  two edges and receives the analytic nodal gradient of that energy;
* frictionless finite-thickness self contact: every pair of non-adjacent
  segments is a capsule of radius ``contact_distance / 2`` and a
  non-penetration constraint is projected, with the contact impulse spread
  consistently over the four adjacent vertices (zero weight on the controlled
  vertices);
* the two controlled vertices (0 and 160) are kinematically prescribed by the
  frozen smoothstep waypoints, and the reaction impulse needed to hold them on
  that trajectory is booked as the external impulse/force budget.

Numerical method
----------------
A substepped position-based (XPBD) integrator: explicit Euler prediction with
the bending force, projection of positions back onto the stretch and contact
manifolds, then the constraint impulses are added to the velocities
``v <- v + dx_constraint / dt`` (implicit-Euler-like behaviour for stiff
constraints).  Everything is deterministic and free of random numbers.

The scaffold's authored preparation curve (10.06 m) is longer than the rod
contour length (6.95 m), so ``initialize`` first relaxes the strand
quasi-statically onto its inextensible, contact-free manifold at the frozen
t=0 endpoint targets; time stepping then starts from that admissible state.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _smoothstep(x: float) -> float:
    """Frozen endpoint interpolation used by the scene."""
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def _closest_points(p0, p1, q0, q1):
    """Closest parameter pair (s, t) between many segments in parallel."""
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0
    a = _dot(d1, d1)
    e = _dot(d2, d2)
    b = _dot(d1, d2)
    c = _dot(d1, r)
    f = _dot(d2, r)
    den = a * e - b * b
    s = np.zeros_like(a)
    t = np.zeros_like(a)
    ok = den > 1e-14
    if ok.any():
        s[ok] = np.clip((b[ok] * f[ok] - c[ok] * e[ok]) / den[ok], 0.0, 1.0)
        t[ok] = np.clip((a[ok] * f[ok] - b[ok] * c[ok]) / den[ok], 0.0, 1.0)
    if (~ok).any():
        e_safe = np.where(e[~ok] > 1e-14, e[~ok], 1.0)
        t[~ok] = np.clip(f[~ok] / e_safe, 0.0, 1.0)
    return s, t


class BenchmarkSolver:
    """Deterministic physics solver: frictionless elastic rod knot tightening."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = getattr(scene, "config", {}) or {}
        rod = cfg.get("rod", {})
        contact = cfg.get("self_contact", {})
        manip = cfg.get("endpoint_manipulation", {})

        self.n = int(rod.get("vertex_count", 161))
        self.radius = float(rod.get("radius", 0.035))
        self.contact_distance = float(contact.get("contact_distance", 2.0 * self.radius))
        self.friction = float(contact.get("friction_coefficient", 0.0))
        self.total_mass = float(rod.get("total_mass", 0.12))
        self.rest_length = float(rod.get("rest_length", 6.95))
        self.flexural_rigidity = 2.0e-6  # EI of the soft elastic strand [N*m^2]
        self.dt_sub = 0.0025
        self.max_speed = 8.0  # numerical safety clamp on |v| [m/s]

        self.mass = np.full(self.n, self.total_mass / self.n, dtype=float)
        self.inv_mass = 1.0 / self.mass

        initial = scene.initial_state()
        x0 = np.array(initial.position, dtype=float)
        self.attributes = {k: np.array(v, dtype=float) for k, v in initial.attributes.items()}

        # Inextensible edge rest lengths honour the frozen rod contour length.
        self.rest = np.full(self.n - 1, self.rest_length / (self.n - 1), dtype=float)

        # Controlled vertices: 0 = left tail, 160 = threading tail.
        ctrl = list(manip.get("controlled_vertices", [0, self.n - 1]))
        self.controlled = np.array(ctrl, dtype=int)
        self.wp_left = self._waypoints(manip["left_tail"]["waypoints"])
        self.wp_right = self._waypoints(manip["threading_tail"]["waypoints"])
        self.v_ref = self._characteristic_speed()

        # Frictionless self-contact candidate pairs (skip adjacent segments).
        ii, jj = np.meshgrid(np.arange(self.n - 1), np.arange(self.n - 1), indexing="ij")
        keep = jj >= ii + 2
        self.ci = ii[keep].astype(int)
        self.cj = jj[keep].astype(int)

        # Prepare a length-consistent, contact-free loose loop: the authored
        # preparation curve is longer than the rod, so the strand is relaxed
        # quasi-statically onto its inextensible manifold before time stepping.
        self.x = self._prepare(x0)
        self.v = np.zeros_like(self.x)

        self.t_cur = 0.0
        self.frame_count = 0
        self.substep_count = 0

        # Budget bookkeeping: one global strand domain (G = 1), two external
        # force terms (the two driven tails), one conserved material (K = 1).
        self.J_ext = np.zeros((1, 3))
        self.F_ext_acc = np.zeros((2, 3))
        self.F_ext_mean = np.zeros((1, 2, 3))
        self.frame_elapsed = 0.0
        self.mass_source = np.zeros(1)
        self.constraint_scale = float(self.rest_length)
        self.reference_momentum = float(self.total_mass * self.v_ref)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _waypoints(entries) -> list:
        return [(float(w["time"]), np.array(w["position"], dtype=float)) for w in entries]

    def _characteristic_speed(self) -> float:
        speed = 0.0
        for wps in (self.wp_left, self.wp_right):
            for (ta, pa), (tb, pb) in zip(wps[:-1], wps[1:]):
                dt = tb - ta
                if dt > 0.0:
                    speed = max(speed, 1.5 * float(np.linalg.norm(pb - pa)) / dt)
        return max(speed, 1e-6)

    @staticmethod
    def _eval_waypoints(wps, t: float):
        if t <= wps[0][0]:
            return wps[0][1].copy(), np.zeros(3)
        for (ta, pa), (tb, pb) in zip(wps[:-1], wps[1:]):
            if t <= tb:
                dt = tb - ta
                s = (t - ta) / dt
                u = _smoothstep(s)
                du = 6.0 * s * (1.0 - s) / dt
                return pa + u * (pb - pa), du * (pb - pa)
        return wps[-1][1].copy(), np.zeros(3)

    def _targets(self, t: float):
        pos = np.empty((2, 3))
        vel = np.empty((2, 3))
        pos[0], vel[0] = self._eval_waypoints(self.wp_left, t)
        pos[1], vel[1] = self._eval_waypoints(self.wp_right, t)
        return pos, vel

    # -------------------------------------------------------------- geometry
    def _edge_lengths(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(_dot(x[1:] - x[:-1], x[1:] - x[:-1]))

    def _project_stretch(self, x: np.ndarray, factor: float, weights: np.ndarray) -> np.ndarray:
        """Jacobi projection of every edge onto its inextensible rest length."""
        d = x[1:] - x[:-1]
        length = np.sqrt(_dot(d, d))
        safe = np.maximum(length, 1e-12)
        e = ((length - self.rest) / safe)[:, None] * d
        w0 = weights[:-1]
        w1 = weights[1:]
        wsum = w0 + w1
        good = wsum > 1e-12
        wsum_safe = np.where(good, wsum, 1.0)
        corr = np.zeros_like(x)
        np.add.at(corr, np.arange(self.n - 1), (factor * w0 / wsum_safe)[:, None] * e)
        np.add.at(corr, np.arange(1, self.n), -(factor * w1 / wsum_safe)[:, None] * e)
        return x + corr

    def _contact_pairs(self, x: np.ndarray):
        """Broad phase: candidate pairs whose midpoints can be within contact."""
        p0 = x[self.ci]
        p1 = x[self.ci + 1]
        q0 = x[self.cj]
        q1 = x[self.cj + 1]
        d1 = p1 - p0
        d2 = q1 - q0
        len_i = np.sqrt(_dot(d1, d1))
        len_j = np.sqrt(_dot(d2, d2))
        mid = 0.5 * (p0 + p1 - q0 - q1)
        cut = 0.5 * (len_i + len_j) + self.contact_distance
        near = _dot(mid, mid) < cut * cut
        return np.nonzero(near)[0]

    def _project_contact(self, x: np.ndarray, factor: float, weights: np.ndarray) -> np.ndarray:
        """Frictionless finite-thickness non-penetration projection."""
        near = self._contact_pairs(x)
        if near.size == 0:
            return x
        i = self.ci[near]
        j = self.cj[near]
        p0 = x[i]
        p1 = x[i + 1]
        q0 = x[j]
        q1 = x[j + 1]
        s, t = _closest_points(p0, p1, q0, q1)
        delta = (p0 + s[:, None] * (p1 - p0)) - (q0 + t[:, None] * (q1 - q0))
        dist = np.sqrt(_dot(delta, delta))
        dist_safe = np.maximum(dist, 1e-12)
        pen = self.contact_distance - dist
        act = pen > 0.0
        if not act.any():
            return x
        i = i[act]
        j = j[act]
        s = s[act]
        t = t[act]
        pen = np.minimum(pen[act], 0.5 * self.contact_distance)
        n = delta[act] / dist_safe[act][:, None]

        w = weights
        a0 = (1.0 - s) ** 2 * w[i]
        a1 = s ** 2 * w[i + 1]
        b0 = (1.0 - t) ** 2 * w[j]
        b1 = t ** 2 * w[j + 1]
        denom = np.maximum(a0 + a1 + b0 + b1, 1e-12)

        corr = np.zeros_like(x)
        scale = factor * pen / denom
        np.add.at(corr, i, (w[i] * (1.0 - s) * scale)[:, None] * n)
        np.add.at(corr, i + 1, (w[i + 1] * s * scale)[:, None] * n)
        np.add.at(corr, j, -(w[j] * (1.0 - t) * scale)[:, None] * n)
        np.add.at(corr, j + 1, -(w[j + 1] * t * scale)[:, None] * n)
        return x + corr

    def _bending_force(self, x: np.ndarray) -> np.ndarray:
        """Analytic nodal force of the discrete turning-angle bending energy."""
        if self.n < 3:
            return np.zeros_like(x)
        e1 = x[1:-1] - x[:-2]
        e2 = x[2:] - x[1:-1]
        l1 = np.maximum(np.sqrt(_dot(e1, e1)), 1e-12)
        l2 = np.maximum(np.sqrt(_dot(e2, e2)), 1e-12)
        t1 = e1 / l1[:, None]
        t2 = e2 / l2[:, None]
        c = np.clip(_dot(t1, t2), -1.0, 1.0)
        sin_t = np.sqrt(np.maximum(1.0 - c * c, 0.0))
        theta = np.arccos(c)
        h = 0.5 * (self.rest[:-1] + self.rest[1:])
        coef = np.where(sin_t > 1e-8, self.flexural_rigidity * theta / (h * np.maximum(sin_t, 1e-8)), 0.0)
        # gradients of cos(theta) with respect to the three vertices
        gm = -(t2 - c[:, None] * t1) / l1[:, None]
        gp = (t1 - c[:, None] * t2) / l2[:, None]
        fm = coef[:, None] * gm
        fp = coef[:, None] * gp
        f = np.zeros_like(x)
        np.add.at(f, np.arange(0, self.n - 2), fm)
        np.add.at(f, np.arange(1, self.n - 1), -(fm + fp))
        np.add.at(f, np.arange(2, self.n), fp)
        return f

    # ------------------------------------------------------------ preparation
    def _prepare(self, x0: np.ndarray) -> np.ndarray:
        """Quasi-static relaxation onto the inextensible, contact-free manifold."""
        x = x0.copy()
        weights = self.inv_mass.copy()
        weights[self.controlled] = 0.0
        tgt, _ = self._targets(0.0)
        x[self.controlled] = tgt
        for it in range(6000):
            x = self._project_stretch(x, 0.8, weights)
            if it % 2 == 0:
                x = self._project_contact(x, 0.6, weights)
            if it % 50 == 49:
                strain = np.max(np.abs(self._edge_lengths(x) - self.rest) / self.rest)
                near = self._contact_pairs(x)
                if strain < 5e-4 and near.size == 0:
                    break
        x[self.controlled] = tgt
        return x

    # --------------------------------------------------------------- stepping
    def _step(self, dt: float) -> None:
        x = self.x
        v = self.v
        t_next = self.t_cur + dt
        tgt_pos, tgt_vel = self._targets(t_next)

        weights = self.inv_mass.copy()
        weights[self.controlled] = 0.0

        # explicit bending force + explicit Euler predictor
        f_bend = self._bending_force(x)
        v_pred = v + (f_bend * self.inv_mass[:, None]) * dt
        x_pred = x + v_pred * dt
        x_pred[self.controlled] = tgt_pos

        # position projection onto stretch + frictionless contact manifolds
        x_new = x_pred
        for _ in range(3):
            x_new = self._project_stretch(x_new, 1.0, weights)
        for _ in range(2):
            x_new = self._project_contact(x_new, 1.0, weights)

        # velocity update with constraint impulses
        v_new = v_pred + (x_new - x_pred) / dt

        # external (drive) impulse = momentum change minus internal impulses
        constraint_impulse = self.mass[:, None] * (x_new - x_pred) / dt
        internal_impulse = f_bend * dt + constraint_impulse
        ext_impulse = self.mass[:, None] * (v_new - v) - internal_impulse
        np.clip(v_new, -self.max_speed, self.max_speed, out=v_new)
        v_new[self.controlled] = tgt_vel
        drive = ext_impulse[self.controlled]
        self.J_ext[0] += drive.sum(axis=0)
        self.F_ext_acc += drive
        self.frame_elapsed += dt
        self.substep_count += 1

        self.x = x_new
        self.v = v_new
        self.t_cur = t_next

    # ------------------------------------------------------------------ frames
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        self.F_ext_acc[:] = 0.0
        self.frame_elapsed = 0.0
        guard = 0
        while self.t_cur < target - 1e-12 and guard < 200000:
            dt = min(self.dt_sub, target - self.t_cur)
            if dt <= 1e-12:
                break
            self._step(dt)
            guard += 1
        if self.frame_elapsed > 0.0:
            self.F_ext_mean[0] = self.F_ext_acc / self.frame_elapsed
        else:
            self.F_ext_mean[0] = 0.0
        self.frame_count += 1
        return self._frame(target)

    def _frame(self, t: float) -> StateFrame:
        ctrl = self.controlled
        tgt, _ = self._targets(t)
        attrs = {k: np.array(v, dtype=float) for k, v in self.attributes.items()}
        attrs["mass_quantity"] = self.mass.copy()
        frame = StateFrame(
            float(t),
            self.x.copy(),
            self.v.copy(),
            orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n, 1)),
            angular_velocity=np.zeros((self.n, 3)),
            mass=self.mass.copy(),
            attributes=attrs,
            fields={
                "budget_domain_weights": np.ones((1, self.n)),
                "budget_cumulative_external_impulse": self.J_ext.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "budget_external_force": self.F_ext_mean.copy(),
                "cumulative_mass_source": self.mass_source.copy(),
                "constraint_value": self.x[ctrl].copy(),
                "constraint_target": tgt,
            },
            scalars={
                "constraint_scale": self.constraint_scale,
                "reference_momentum": self.reference_momentum,
                "contact_length_scale": self.contact_distance,
                "friction_coefficient": self.friction,
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        strain = 0.0
        if self.t_cur >= 0.0:
            strain = float(np.max(np.abs(self._edge_lengths(self.x) - self.rest) / self.rest))
        return {
            "substeps": int(self.substep_count),
            "frames": int(self.frame_count),
            "max_edge_strain": strain,
            "final_time": float(self.t_cur),
        }
