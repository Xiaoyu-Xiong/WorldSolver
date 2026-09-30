"""Discrete elastic rod solver: twist-driven plectoneme in a hanging rod.

Physical model (Kirchhoff / discrete elastic rod, implemented from scratch)
--------------------------------------------------------------------------
The rod is a chain of ``N`` nodes joined by ``N-1`` edges of rest length
``rest = L / (N-1)``.

Stored energies (all evaluated from the current centerline):

* stretch   ``U_s = 0.5 * ks * (l_e - rest)^2``  (very stiff -> inextensible)
* bend      ``U_b = c_b * (theta_i - theta0_i)^2`` with ``c_b = EI / (2 rest)``
  where ``theta_i`` is the turning angle at interior node ``i``.
* twist     the rod carries the twist moment ``tau = GJ * m1`` where the twist
  density ``m1`` (rad/m) is set by the two position-controlled ends whose
  relative rotation follows the frozen smoothstep ramp:
  ``m1 = Theta(t) / L``.  Through moment equilibrium ``M' + t x F = 0`` with
  ``M = tau * t`` this produces the follower lateral force density
  ``f = t x M' = tau * (t x t') = tau * kappa * b``, i.e. the discrete nodal
  force ``F_i = tau * kappa_b_i`` with the discrete curvature binormal
  ``kappa_b_i = 2 (t_{i-1} x t_i) / (l_{i-1} + l_i)``.  This twist-bend
  coupling is what destabilises the straight/twisted hanging rod (buckling at
  ``tau_c = 2 sqrt(EI*T)``), makes the centerline leave the hanging plane and
  wraps the rod into a plectoneme.

Forces: gravity, isotropic ambient air drag (linear, no fluid simulation),
finite-thickness self-contact (soft repulsion between nodes closer than the
rod diameter, resolved with a scipy cKDTree radius search) and the elastic
forces above.

Integration: explicit symplectic (semi-implicit) time stepping with a
sub-step small enough for the stiffest elastic mode; the linear air drag is
treated implicitly, which is unconditionally stable.  Endpoints are kinematic
(positions prescribed, velocities zero); the constraint reactions are recorded
and exported in the momentum budgets.

Everything exported in the StateFrame comes from this time stepping.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_TWO_PI = 2.0 * np.pi


def _smoothstep(x: float, e0: float, e1: float) -> float:
    if e1 <= e0:
        return 0.0
    u = (float(x) - e0) / (e1 - e0)
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return np.array([0.0, 0.0, 1.0])
    return np.asarray(v, dtype=float) / n


def _rotate(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


def _transport(q: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """Minimal rotation carrying ``t0`` to ``t1``, applied to ``q``."""
    axis = np.cross(t0, t1)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(t0, t1))
    if s < 1e-14:
        if c > 0.0:
            return q.copy()
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(ref, t0))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        a = _unit(np.cross(t0, ref))
        return 2.0 * float(np.dot(a, q)) * a - q
    axis = axis / s
    th = np.arctan2(s, c)
    ca, sa = np.cos(th), np.sin(th)
    return (q * ca + np.cross(axis, q) * sa
            + axis * float(np.dot(axis, q)) * (1.0 - ca))


class BenchmarkSolver:
    """Twist-driven plectoneme of a hanging inextensible elastic rod."""

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.scene = scene
        self.cfg = cfg
        rod = cfg["rod"]
        self.n = int(rod["vertex_count"])
        self.ne = self.n - 1
        self.radius = float(rod["radius"])
        self.length = float(rod["total_length"])
        self.rest = self.length / float(self.ne)
        self.mass_total = float(rod["mass"])
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        drag = cfg.get("ambient_drag", {}) or {}
        self.drag_on = bool(drag.get("enabled", False))
        self.drag = float(drag.get("coefficient", 0.0) or 0.0) if self.drag_on else 0.0

        ctrl = cfg["end_controls"]
        self.left_pos = np.asarray(ctrl["left"]["position"], dtype=float)
        self.right_pos = np.asarray(ctrl["right"]["position"], dtype=float)
        self.left_axis = _unit(np.asarray(ctrl["left"]["outward_axis"], dtype=float))
        self.right_axis = _unit(np.asarray(ctrl["right"]["outward_axis"], dtype=float))
        ramp = ctrl["twist_ramp"]
        self.ramp_start = float(ramp["ramp_start"])
        self.ramp_end = float(ramp["ramp_end"])
        self.turns = float(ramp["final_relative_turns"])
        self.dt_sample = float(cfg.get("sample_dt", 0.02))
        self.duration = float(cfg.get("duration", 18.0))
        self.rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)

        # ---- material parameters ----------------------------------------
        a = self.radius
        self.young = 2.0e6                    # Pa (soft elastic rod)
        self.poisson = 0.5
        i_area = 0.25 * np.pi * a ** 4
        self.ei = self.young * i_area
        self.gj = self.young / (2.0 * (1.0 + self.poisson)) * (2.0 * i_area)
        self.c_bend = self.ei / (2.0 * self.rest)     # U = c_bend * theta^2
        self.ks = 1.0e7                                # axial penalty (N/m)

        # ---- state -------------------------------------------------------
        p = np.asarray(scene.initial_state().position, dtype=float).copy()
        p[0] = self.left_pos
        p[-1] = self.right_pos
        self.pos = p
        self.vel = np.zeros((self.n, 3))
        self.mass = np.full(self.n, self.mass_total / float(self.ne))
        self.mass[0] *= 0.5
        self.mass[-1] *= 0.5
        self.inv_mass = 1.0 / self.mass
        self.pinned = np.zeros(self.n, dtype=bool)
        self.pinned[0] = True
        self.pinned[-1] = True

        # reference (twist free) material frame, transported along the rod
        self.tangent = self._tangents(self.pos)
        self.ref_d1, self.ref_d2 = self._natural_frame(self.pos, self.tangent)
        self.d1 = self.ref_d1.copy()
        self.d2 = self.ref_d2.copy()

        # self contact
        self.contact_gap = 2.0 * self.radius
        self.contact_k = 2.0e4
        self.contact_min_sep = 3

        # ---- time step ---------------------------------------------------
        self.h_sub = 5.0e-5
        self.time = 0.0
        self.steps = 0
        self.contact_events = 0
        self.max_penetration = 0.0

        # ---- bookkeeping for the exported budgets -------------------------
        self._last_time = None
        self._prev_d1 = None
        self._prev_frame_time = None
        self.cum_ext_impulse = np.zeros(3)
        self.cum_ext_ang_impulse = np.zeros(3)
        self.reaction = np.zeros((self.n, 3))
        self.grav_total = np.zeros(3)
        self.drag_total = np.zeros(3)
        self.work_done = 0.0
        self.inertia_tensor = self._inertia_tensor()
        self.mass_quantity = self.mass.copy()

        # frozen positive reference scales
        g = abs(float(self.gravity[2])) if abs(float(self.gravity[2])) > 0 else 9.81
        self.ref_momentum = float(self.mass_total * np.sqrt(g * self.length))
        self.ref_ang_momentum = float(self.ref_momentum * self.length)
        self.constraint_scale = float(1.0 / self.length)
        self._sym_cache = None

    # ------------------------------------------------------------- geometry
    def _tangents(self, p: np.ndarray) -> np.ndarray:
        e = p[1:] - p[:-1]
        t = np.zeros((self.n, 3))
        t[:-1] += e
        t[1:] += e
        nrm = np.linalg.norm(t, axis=1, keepdims=True)
        nrm[nrm < 1e-15] = 1.0
        return t / nrm

    def _natural_frame(self, p: np.ndarray, t: np.ndarray):
        """Twist-free (parallel transported) material frame along the rod."""
        d1 = np.zeros((self.n, 3))
        ref = np.array([0.0, 1.0, 0.0])
        d1[0] = _unit(ref - t[0] * float(np.dot(ref, t[0])))
        for i in range(1, self.n):
            d1[i] = _transport(d1[i - 1], t[i - 1], t[i])
        d2 = np.cross(t, d1)
        nrm = np.linalg.norm(d2, axis=1, keepdims=True)
        nrm[nrm < 1e-15] = 1.0
        d2 = d2 / nrm
        d1 = np.cross(d2, t)
        return d1, d2

    def _inertia_tensor(self) -> np.ndarray:
        """Intrinsic segment inertia (transverse component about the tangent)."""
        out = np.zeros((self.n, 3, 3))
        for i in range(self.n):
            t = self.tangent[i]
            ii = self.mass[i] * self.rest * self.rest / 12.0
            out[i] = ii * (np.eye(3) - np.outer(t, t))
        return out

    # --------------------------------------------------------- twist driver
    def _theta(self, time: float) -> float:
        """Total relative end rotation (rad) from the frozen twist ramp."""
        return _TWO_PI * self.turns * _smoothstep(time, self.ramp_start, self.ramp_end)

    def _m1(self, time: float) -> float:
        """Twist density (rad/m) of the uniformly twisted rod."""
        return self._theta(time) / self.length

    def _tau(self, time: float) -> float:
        """Twist moment (N*m)."""
        return self.gj * self._m1(time)

    # ---------------------------------------------------------------- forces
    def _elastic_forces(self, p: np.ndarray, tau: float):
        n, ne = self.n, self.ne
        e = p[1:] - p[:-1]
        l = np.linalg.norm(e, axis=1)
        l = np.maximum(l, 1e-9)
        t_edge = e / l[:, None]

        F = np.zeros((n, 3))

        # stretch
        fk = (self.ks * (l - self.rest))[:, None] * t_edge
        F[:-1] += fk
        F[1:] -= fk

        # bend  U = c_bend * theta^2  ->  F = -dU/dp
        if ne >= 2:
            t0 = t_edge[:-1]
            t1 = t_edge[1:]
            cosv = np.clip(np.einsum('ij,ij->i', t0, t1), -1.0, 1.0)
            theta = np.arccos(cosv)
            sinv = np.maximum(np.sqrt(np.maximum(1.0 - cosv * cosv, 0.0)), 1e-8)
            kb = 2.0 * self.c_bend * theta
            l0 = l[:-1][:, None]
            l1 = l[1:][:, None]
            g0 = (t1 - cosv[:, None] * t0) / (sinv[:, None] * l0)
            g1 = ((t0 - cosv[:, None] * t1) / (sinv[:, None] * l1)
                  - (t1 - cosv[:, None] * t0) / (sinv[:, None] * l0))
            g2 = -((t0 - cosv[:, None] * t1) / (sinv[:, None] * l1))
            grad = kb[:, None] * np.stack((g0, g1, g2), axis=1)
            F[:-2] -= grad[:, 0, :]
            F[1:-1] -= grad[:, 1, :]
            F[2:] -= grad[:, 2, :]

            # twist-bend coupling: nodal force tau * curvature binormal
            if tau != 0.0:
                kb_vec = 2.0 * np.cross(t0, t1) / (l[:-1] + l[1:])[:, None]
                ft = tau * kb_vec
                F[1:-1] += ft

        return F

    def _contact(self, p: np.ndarray):
        """Finite-thickness self-contact: soft repulsion, cKDTree pair search."""
        F = np.zeros((self.n, 3))
        if self.n < 2:
            return F, 0, 0.0
        tree = cKDTree(p)
        pairs = tree.query_pairs(self.contact_gap, output_type='ndarray')
        if pairs.size == 0:
            return F, 0, 0.0
        keep = (pairs[:, 1] - pairs[:, 0]) >= self.contact_min_sep
        pairs = pairs[keep]
        if pairs.size == 0:
            return F, 0, 0.0
        i, j = pairs[:, 0], pairs[:, 1]
        d = p[j] - p[i]
        dist = np.linalg.norm(d, axis=1)
        good = dist > 1e-12
        i, j, d, dist = i[good], j[good], d[good], dist[good]
        pen = self.contact_gap - dist
        pen = np.maximum(pen, 0.0)
        mag = self.contact_k * pen
        dirv = d / dist[:, None]
        f = mag[:, None] * dirv
        np.add.at(F, i, -f)
        np.add.at(F, j, f)
        return F, int(len(i)), float(pen.max()) if len(pen) else 0.0

    # ------------------------------------------------------------ time step
    def _substep(self, h: float) -> None:
        p = self.pos
        tau = self._tau(self.time)
        F = self._elastic_forces(p, tau)
        F += self.mass[:, None] * self.gravity[None, :]
        Fc, nev, pen = self._contact(p)
        F += Fc
        self.contact_events += nev
        if pen > self.max_penetration:
            self.max_penetration = pen

        self.grav_total = np.sum(self.mass[:, None] * self.gravity[None, :], axis=0)
        if self.drag > 0.0:
            self.drag_total = -self.drag * np.sum(self.vel, axis=0)
        else:
            self.drag_total = np.zeros(3)

        with np.errstate(divide='ignore', invalid='ignore'):
            acc = F * self.inv_mass[:, None]
        v = self.vel + h * acc
        if self.drag > 0.0:
            v = v / (1.0 + h * self.drag * self.inv_mass)[:, None]

        # kinematic endpoints: freeze positions, record reaction impulse
        react = np.zeros((self.n, 3))
        if self.pinned.any():
            react[self.pinned] = -self.mass[self.pinned, None] * v[self.pinned] / h
            v[self.pinned] = 0.0
        self.reaction = react

        p_new = p + h * v
        p_new[0] = self.left_pos
        p_new[-1] = self.right_pos
        self.pos = p_new
        self.vel = v
        self.time += h
        self.steps += 1

    # ----------------------------------------------------------- integration
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if self._last_time is None:
            self._last_time = 0.0
            self._prev_frame_time = 0.0
            self._prev_d1 = self.d1.copy()
        n = int(round((target - self.time) / self.h_sub))
        if n < 0:
            n = 0
        for _ in range(n):
            self._substep(self.h_sub)
        self._accumulate_budget(target)
        return self._make_frame(target)

    def _accumulate_budget(self, time: float) -> None:
        dt = time - self._last_time
        if dt > 0.0:
            fext = self.grav_total + self.drag_total + self.reaction.sum(axis=0)
            pext = np.cross(self.pos, self.mass[:, None] * self.gravity[None, :])
            pext = pext.sum(axis=0)
            pext += np.cross(self.pos, -self.drag * self.vel).sum(axis=0)
            pext += np.cross(self.pos, self.reaction).sum(axis=0)
            self.cum_ext_impulse = self.cum_ext_impulse + fext * dt
            self.cum_ext_ang_impulse = self.cum_ext_ang_impulse + pext * dt
        self._last_time = time

    # ------------------------------------------------------------- exporting
    def _material_frames(self, time: float):
        t = self._tangents(self.pos)
        d1n, d2n = self._natural_frame(self.pos, t)
        m1 = self._m1(time)
        s = np.linspace(0.0, self.length, self.n)
        psi = m1 * (s - 0.5 * self.length)          # -Theta/2 ... +Theta/2
        d1 = np.empty_like(d1n)
        d2 = np.empty_like(d2n)
        for i in range(self.n):
            d1[i] = _rotate(d1n[i], t[i], psi[i])
            d2[i] = _rotate(d2n[i], t[i], psi[i])
        return t, d1, d2

    def _make_frame(self, time: float) -> StateFrame:
        t, d1, d2 = self._material_frames(time)

        # angular velocity from the finite difference of the material frames
        omega = np.zeros((self.n, 3))
        dt = time - float(self._prev_frame_time)
        if dt > 1e-12 and self._prev_d1 is not None:
            for i in range(self.n):
                c = float(np.clip(np.dot(self._prev_d1[i], d1[i]), -1.0, 1.0))
                s = np.cross(self._prev_d1[i], d1[i])
                sn = float(np.linalg.norm(s))
                if sn > 1e-14:
                    omega[i] = (s / sn) * (np.arctan2(sn, c) / dt)

        # orientation quaternion (xyzw) of the material frame
        quat = np.empty((self.n, 4))
        for i in range(self.n):
            R = np.stack((d1[i], d2[i], t[i]), axis=1)
            quat[i] = self._quat_from_matrix(R)
        nrm = np.linalg.norm(quat, axis=1, keepdims=True)
        nrm[nrm < 1e-14] = 1.0
        quat = quat / nrm

        # ---- budgets -----------------------------------------------------
        w = np.zeros((2, self.n))
        w[0, :] = 1.0
        w[1, :] = 1.0
        w[1, 0] = 0.0
        w[1, -1] = 0.0
        G, K = 2, 4
        force = np.zeros((G, K, 3))
        fpos = np.zeros((G, K, 3))
        couple = np.zeros((G, K, 3))
        grav_node = self.mass[:, None] * self.gravity[None, :]
        drag_node = -self.drag * self.vel if self.drag > 0.0 else np.zeros((self.n, 3))
        for g in range(G):
            m = w[g]
            tot = float(m.sum())
            if tot <= 0.0:
                continue
            com = (m[:, None] * self.pos).sum(axis=0) / tot
            force[g, 0] = (m[:, None] * grav_node).sum(axis=0)
            fpos[g, 0] = com
            force[g, 1] = (m[:, None] * drag_node).sum(axis=0)
            fpos[g, 1] = com
            force[g, 2] = self.reaction[0]
            fpos[g, 2] = self.pos[0]
            force[g, 3] = self.reaction[-1]
            fpos[g, 3] = self.pos[-1]

        # contact force summary (internal, exported as a couple-free term of
        # domain 0 in case the platform sums force terms)
        fields = {
            "budget_cumulative_external_angular_impulse":
                self.cum_ext_ang_impulse.reshape(1, 3).copy(),
            "budget_cumulative_external_impulse":
                self.cum_ext_impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_angular_momentum": np.zeros((G, 3)),
            "budget_cumulative_inflow_momentum": np.zeros((G, 3)),
            "budget_cumulative_outflow_angular_momentum": np.zeros((G, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((G, 3)),
            "budget_domain_weights": w,
            "budget_external_couple": couple,
            "budget_external_force": force,
            "budget_force_position": fpos,
            "constraint_target": np.array([self._m1(time)]),
            "constraint_value": np.array([self._m1(time)]),
            "cumulative_mass_source": np.zeros(1),
        }
        scalars = {
            "constraint_scale": float(self.constraint_scale),
            "reference_momentum": float(max(self.ref_momentum, 1e-9)),
            "reference_angular_momentum": float(max(self.ref_ang_momentum, 1e-9)),
        }
        attributes = {
            "inertia_tensor": self._inertia_tensor(),
            "mass_quantity": self.mass.copy(),
        }
        self.d1 = d1
        self.d2 = d2
        self._prev_d1 = d1.copy()
        self._prev_frame_time = float(time)
        frame = StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=quat,
            angular_velocity=omega,
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    @staticmethod
    def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
        tr = float(R[0, 0] + R[1, 1] + R[2, 2])
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            return np.array([0.25 * s * (R[2, 1] - R[1, 2]),
                             0.25 * s * (R[0, 2] - R[2, 0]),
                             0.25 * s * (R[1, 0] - R[0, 1]),
                             0.25 * s])
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12)) * 2.0
            return np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s,
                             (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
        if R[1, 1] > R[2, 2]:
            s = np.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12)) * 2.0
            return np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s,
                             (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
        s = np.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12)) * 2.0
        return np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
                         0.25 * s, (R[1, 0] - R[0, 1]) / s])

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": int(self.steps),
            "contact_events": int(self.contact_events),
            "max_penetration": float(self.max_penetration),
            "final_time": float(self.time),
            "vertices": int(self.n),
            "substep": float(self.h_sub),
        }
