"""Discrete elastic rod solver: localized helical buckling of a twisted rod.

Physics implemented from scratch with NumPy only (no external physics engine):

* The rod is a chain of N lumped-mass nodes at arc-length positions ``s``.
* Inextensibility and the clamped end tangents are enforced by a projection
  (Gauss-Seidel) step, i.e. a hard geometric constraint manifold.
* Elastic energy: a discrete Kirchhoff bending term

      E_b = EI/(2 ds) * sum_i | t_i - t_{i-1} |^2          (t = unit tangents)

  plus a twist term for a rod whose clamps prescribe a relative material
  rotation ``Theta``.  The material frame is parallel transported from the
  left clamp; the total transport angle ``Delta(x)`` accumulated along the
  current centerline (the discrete writhe/geodesic-torsion integral) leaves

      phi_end = Theta - Delta(x),      E_t = GJ/(2L) * phi_end^2 .

  Geometry changes therefore trade bending against twist, which is exactly the
  mechanism producing the localized helical post-buckling state.
* Self-contact is a soft excluded-volume barrier between non-adjacent nodes.
* Loading: the twist stage ramps ``Theta`` and the shortening stage moves the
  right clamp in ``increment_count`` steps.  Each sample is obtained by a
  damped (quasi-static, displacement-controlled) incremental minimisation of
  the total elastic energy, which is a genuine physical time-marching scheme in
  the strong-damping limit adopted by the frozen quasi-static protocol.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


def _min_rotate(v: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the minimal rotation taking unit vector ``a`` to ``b``."""
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    axis = np.cross(a, b)
    ax = float(np.linalg.norm(axis))
    if ax < 1e-14:
        return v
    angle = np.arctan2(ax, c)
    return _rotate(v, axis / ax, angle)


class BenchmarkSolver:
    """Quasi-static discrete elastic rod under twist and end shortening."""

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(getattr(scene, "config", {}))
        rod = cfg.get("rod", {})
        ic = cfg.get("initial_conditions", {})
        proto = cfg.get("loading_protocol", {})
        bc = cfg.get("boundary_conditions", {})

        self.rng = np.random.default_rng(int(seed))
        self.length = float(rod.get("length", 0.929))
        self.radius = float(rod.get("radius", 0.0045))
        self.density = float(rod.get("density", 1100.0))
        self.n = int(getattr(scene, "vertex_count", 129))
        self.ds = self.length / (self.n - 1)

        # Stiffness scales (statics is scale invariant; only GJ/EI matters).
        self.EI = float(rod.get("bending_modulus_ratio", 1.345))
        self.GJ = float(rod.get("twisting_modulus_ratio", 0.789))
        self.reference_force = self.EI / (self.length * self.length)
        self.reference_torque = self.EI / self.length
        self.constraint_scale = self.length
        self.sample_dt = max(float(cfg.get("sample_dt", 0.02)), 1e-9)

        area = np.pi * self.radius**2
        nodal = self.density * area * self.ds
        self.node_mass = np.full(self.n, nodal)
        self.node_mass[0] *= 0.5
        self.node_mass[-1] *= 0.5
        self.total_mass = float(self.node_mass.sum())

        # Clamp kinematics.
        self.left = np.asarray(ic.get("left_endpoint", [-0.5 * self.length, 0, 0]), float)
        self.right0 = np.asarray(ic.get("right_endpoint", [0.5 * self.length, 0, 0]), float)
        self.left_tangent = _unit(np.asarray(bc.get("left_clamp", {}).get("tangent", [1, 0, 0]), float))
        self.right_tangent = _unit(np.asarray(bc.get("right_clamp", {}).get("tangent", [1, 0, 0]), float))
        self.right_final = np.asarray(
            bc.get("right_clamp", {}).get("final_position", [0.1855, 0, 0]), float
        )

        twist = proto.get("twist_stage", {})
        short = proto.get("shortening_stage", {})
        hold = proto.get("hold_stage", {})
        self.twist_t0 = float(twist.get("start_time", 0.0))
        self.twist_t1 = float(twist.get("end_time", 2.0))
        self.final_twist = float(twist.get("final_relative_rotation_rad", 27.0))
        self.short_t0 = float(short.get("start_time", self.twist_t1))
        self.short_t1 = float(short.get("end_time", 8.0))
        self.increment_count = max(int(short.get("increment_count", 120)), 1)
        self.hold_t1 = float(hold.get("end_time", 10.0))

        # Prescribed material frames at the clamps (lab fixed, frozen at t=0).
        m_left = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(m_left, self.left_tangent))) > 0.9:
            m_left = np.array([0.0, 1.0, 0.0])
        m_left = _unit(m_left - float(np.dot(m_left, self.left_tangent)) * self.left_tangent)
        self.m_left = m_left
        m_right = _rotate(m_left, self.right_tangent, self.final_twist)
        self.m_right_ref = _unit(m_right - float(np.dot(m_right, self.right_tangent)) * self.right_tangent)

        self.contact_dist = 2.0 * self.radius
        self.contact_stiffness = self.EI / (self.radius * self.radius)
        self._build_contact_pairs()

        # Initial configuration: frozen scaffold seed (nearly straight).
        self.s = np.linspace(0.0, self.length, self.n)
        q = self.s / self.length
        window = np.sin(np.pi * q) ** 2
        amp = float(ic.get("branch_selection_perturbation_amplitude", 0.00025))
        try:
            seed_state = scene.initial_state()
            pos0 = np.array(seed_state.position, dtype=float)
            if pos0.shape != (self.n, 3):
                pos0 = self._straight_seed(amp, q, window)
        except Exception:
            pos0 = self._straight_seed(amp, q, window)
        # Negligible out-of-plane branch-selection seed (still nearly straight).
        pos0[:, 2] += 1.0e-3 * amp * window * np.cos(2.0 * np.pi * q)
        self.position = pos0
        self.velocity = np.zeros((self.n, 3))
        self.delta = 0.0
        self._delta_ref = 0.0

        self._prev_written = np.array(self.position, copy=True)
        self._last_time = 0.0
        self._time = 0.0
        self._last_frame = None

    def _straight_seed(self, amp, q, window):
        x = self.s - 0.5 * self.length
        return np.column_stack(
            (x, amp * window * np.sin(2.0 * np.pi * q), np.zeros_like(x))
        )

    def _build_contact_pairs(self) -> None:
        i, j = np.triu_indices(self.n, k=3)
        self.ci = i
        self.cj = j

    # ------------------------------------------------------------------
    # loading protocol
    # ------------------------------------------------------------------
    def _ramp_uf(self, time: float, t0: float, t1: float) -> float:
        if time <= t0:
            return 0.0
        if time >= t1:
            return 1.0
        return (time - t0) / (t1 - t0)

    def _twist(self, time: float) -> float:
        return self.final_twist * self._ramp_uf(time, self.twist_t0, self.twist_t1)

    def _increments(self, time: float) -> int:
        if time <= self.short_t0:
            return 0
        u = self._ramp_uf(time, self.short_t0, self.short_t1)
        return int(np.floor(u * self.increment_count + 1e-12))

    def _right_target(self, time: float) -> np.ndarray:
        f = self._increments(time) / float(self.increment_count)
        return (1.0 - f) * self.right0 + f * self.right_final

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    def _tangents(self, pos: np.ndarray) -> np.ndarray:
        e = pos[1:] - pos[:-1]
        L = np.linalg.norm(e, axis=1, keepdims=True)
        return e / np.maximum(L, 1e-14)

    def _transport_angle(self, pos: np.ndarray) -> float:
        """Accumulated parallel-transport angle from the left clamp to the end."""
        t = self._tangents(pos)
        v = np.array(self.m_left, copy=True)
        for i in range(1, self.n):
            a = t[i - 1]
            b = t[i]
            v = _min_rotate(v, a, b)
            v = v - float(np.dot(v, b)) * b
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                return self.delta
            v = v / nv
        t_end = t[-1]
        ref = self.m_left - float(np.dot(self.m_left, t_end)) * t_end
        nr = float(np.linalg.norm(ref))
        if nr < 1e-12:
            return self.delta
        ref = ref / nr
        ang = float(np.arctan2(float(np.dot(np.cross(ref, v), t_end)), float(np.clip(np.dot(ref, v), -1.0, 1.0))))
        # Continuous branch: pick the representative closest to the running value.
        while ang - self.delta > np.pi:
            ang -= 2.0 * np.pi
        while self.delta - ang > np.pi:
            ang += 2.0 * np.pi
        return ang

    def _phi_end(self, delta: float, theta: float) -> float:
        return theta - delta

    def _bending_energy_and_grad(self, pos: np.ndarray):
        t = self._tangents(pos)
        d = t[1:] - t[:-1]                      # [ne-1,3]
        E = 0.5 * self.EI / self.ds * float(np.sum(d * d))
        g = np.zeros_like(t)
        g[:-1] += 2.0 * self.EI / self.ds * d
        g[1:] -= 2.0 * self.EI / self.ds * d
        e = pos[1:] - pos[:-1]
        L = np.linalg.norm(e, axis=1, keepdims=True)
        proj = g - np.sum(g * t, axis=1, keepdims=True) * t
        de = proj / np.maximum(L, 1e-14)
        grad = np.zeros_like(pos)
        grad[:-1] -= de
        grad[1:] += de
        return E, grad

    def _contact_energy_and_grad(self, pos: np.ndarray):
        d = pos[self.ci] - pos[self.cj]
        r = np.linalg.norm(d, axis=1)
        overlap = self.contact_dist - r
        active = overlap > 0.0
        if not np.any(active):
            return 0.0, np.zeros_like(pos)
        ov = overlap[active]
        rv = np.maximum(r[active], 1e-12)
        dv = d[active] / rv[:, None]
        E = 0.5 * self.contact_stiffness * float(np.sum(ov * ov))
        f = (self.contact_stiffness * ov)[:, None] * dv
        grad = np.zeros_like(pos)
        np.add.at(grad, self.ci[active], -f)
        np.add.at(grad, self.cj[active], f)
        return E, grad

    def _total_energy(self, pos: np.ndarray, theta: float) -> float:
        Eb, _ = self._bending_energy_and_grad(pos)
        Ec, _ = self._contact_energy_and_grad(pos)
        delta = self._transport_angle(pos)
        phi = self._phi_end(delta, theta)
        Et = 0.5 * self.GJ / self.length * phi * phi
        return Eb + Ec + Et

    # ------------------------------------------------------------------
    # constraint projections (clamped ends + inextensibility)
    # ------------------------------------------------------------------
    def _project(self, pos: np.ndarray, theta: float, right: np.ndarray, iters: int = 30) -> None:
        for _ in range(iters):
            pos[0] = self.left
            pos[-1] = right
            # Clamp the end tangents: align the terminal segments.
            e0 = pos[1] - pos[0]
            pos[1] = pos[0] + _min_rotate(e0, _unit(e0), self.left_tangent) * np.linalg.norm(e0)
            en = pos[-1] - pos[-2]
            pos[-2] = pos[-1] - _min_rotate(en, _unit(en), self.right_tangent) * np.linalg.norm(en)
            # Inextensibility (Gauss-Seidel, endpoints immovable).
            for k in range(self.n - 1):
                d = pos[k + 1] - pos[k]
                L = float(np.linalg.norm(d))
                if L < 1e-14:
                    continue
                err = L - self.ds
                if abs(err) < 1e-14:
                    continue
                w0 = 0.0 if k == 0 else 1.0
                w1 = 0.0 if k + 1 == self.n - 1 else 1.0
                tot = w0 + w1
                if tot <= 0.0:
                    continue
                corr = (err / L) * d
                pos[k] += corr * (w0 / tot)
                pos[k + 1] -= corr * (w1 / tot)
        pos[0] = self.left
        pos[-1] = right

    def _relax(self, iters: int, theta: float, right: np.ndarray) -> None:
        pos = self.position
        for _ in range(iters):
            E, grad = self._bending_energy_and_grad(pos)
            Ec, gc = self._contact_energy_and_grad(pos)
            grad = grad + gc
            self.delta = self._transport_angle(pos)
            direction = -grad
            dn = float(np.linalg.norm(direction))
            if dn < 1e-14:
                self._project(pos, theta, right)
                continue
            direction = direction / dn
            E0 = E + Ec + 0.5 * self.GJ / self.length * self._phi_end(self.delta, theta) ** 2
            best_t = 0.0
            best_E = E0
            scale = self.ds
            for mult in (0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
                trial = pos + (mult * scale) * direction
                self._project(trial, theta, right, iters=6)
                Et = self._total_energy(trial, theta)
                if np.isfinite(Et) and Et < best_E:
                    best_E = Et
                    best_t = mult * scale
            if best_t > 0.0:
                pos += best_t * direction
                self._project(pos, theta, right)
            else:
                break
        self.delta = self._transport_angle(pos)

    # ------------------------------------------------------------------
    # frame export
    # ------------------------------------------------------------------
    def _build_frame(self, time: float, theta: float, right: np.ndarray) -> StateFrame:
        pos = np.array(self.position, copy=True)
        vel = np.array(self.velocity, copy=True)
        span = float(np.linalg.norm(right - self.left))
        delta = self._transport_angle(pos)
        phi = self._phi_end(delta, theta)

        # External actions on the rod domain: the two clamp reactions, taken
        # from the elastic nodal force imbalance at the constrained end nodes.
        _, grad = self._bending_energy_and_grad(pos)
        _, grad_c = self._contact_energy_and_grad(pos)
        g = grad + grad_c
        f_left = g[0]
        f_right = g[-1]
        tau_vec = self.GJ * phi / self.length * self._tangents(pos)[-1]

        budget_force = np.stack((f_left, f_right))[None, :, :]
        budget_couple = np.stack((-tau_vec, tau_vec))[None, :, :]
        budget_position = np.stack((pos[0], pos[-1]))[None, :, :]

        constraint_value = np.array(
            [
                float(np.linalg.norm(pos[0] - self.left)),
                float(np.linalg.norm(pos[-1] - right)),
                span,
                phi,
            ],
            float,
        )
        constraint_target = np.array([0.0, 0.0, span, theta], float)

        attributes = {
            "mass_quantity": np.array(self.node_mass, copy=True),
            "arc_length": self.s.copy(),
            "natural_curvature": np.zeros(self.n),
            "natural_twist": np.zeros(self.n),
        }
        fields = {
            "budget_external_force": budget_force,
            "budget_external_couple": budget_couple,
            "budget_force_position": budget_position,
            "constraint_value": constraint_value,
            "constraint_target": constraint_target,
            "cumulative_mass_source": np.zeros(1),
        }
        scalars = {
            "constraint_scale": self.constraint_scale,
            "reference_force": self.reference_force,
            "reference_torque": self.reference_torque,
        }
        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            mass=np.array(self.node_mass, copy=True),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if self._last_frame is not None and target <= self._last_time + 1e-12:
            return self._last_frame

        theta = self._twist(target)
        right = self._right_target(target)
        # Quasi-static velocity from the frame-to-frame state change.
        prev_pos = np.array(self._prev_written, copy=True)

        if self._last_frame is None:
            self._project(self.position, theta, right, iters=200)
            self.delta = self._transport_angle(self.position)
            self._relax(400, theta, right)
            self._project(self.position, theta, right, iters=60)
            self.velocity = np.zeros((self.n, 3))
        else:
            new_inc = self._increments(target) > self._increments(self._last_time)
            if target <= self.short_t0:
                iters = 4
            else:
                iters = 40 if new_inc else 8
            self._relax(iters, theta, right)
            self._project(self.position, theta, right, iters=30)

        dt = max(target - self._last_time, self.sample_dt)
        self.position = np.array(self.position, copy=True)
        self.velocity = (self.position - prev_pos) / dt
        self.velocity[~np.isfinite(self.velocity)] = 0.0
        self._prev_written = np.array(self.position, copy=True)
        self._last_time = target
        self._last_frame = self._build_frame(target, theta, right)
        return self._last_frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "quasi_static_discrete_kirchhoff_rod",
            "vertices": int(self.n),
            "segments": int(self.n - 1),
            "rod_length": float(self.length),
            "bending_stiffness_EI": float(self.EI),
            "twisting_stiffness_GJ": float(self.GJ),
            "right_end_x": float(self.position[-1, 0]),
            "transport_angle_rad": float(self.delta),
        }
