"""Physically-based elastic-rod solver for the trefoil-knot twist snap.

The rod is modelled as a discrete inextensible elastic strand: vertices are
connected by stiff edge-length constraints, bending resistance comes from
angle springs relative to the initial trefoil shape, and twist is introduced
through torsional (dihedral) springs whose rest angles are ramped in time to
mimic the driven right clamp.  Self-contact is handled by a short-range repulsive
penalty between non-adjacent centerline samples, and gravity/damping provide
dissipation.  The implementation is intentionally conservative (small explicit
time step, multiple constraint projection iterations) so the 10-second dynamics
remain stable and deterministic.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < eps, 1.0, n)
    return v / n


def _quat_multiply(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions in xyzw order."""
    x0, y0, z0, w0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x1, y1, z1, w1 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    return np.stack(
        [
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        ],
        axis=-1,
    )


def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix (columns are image of x,y,z axes) to xyzw quaternion."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array([
            (m21 - m12) * s,
            (m02 - m20) * s,
            (m10 - m01) * s,
            0.25 / s,
        ])
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        return np.array([
            0.25 * s,
            (m01 + m10) / s,
            (m02 + m20) / s,
            (m21 - m12) / s,
        ])
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        return np.array([
            (m01 + m10) / s,
            0.25 * s,
            (m12 + m21) / s,
            (m02 - m20) / s,
        ])
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        return np.array([
            (m02 + m20) / s,
            (m12 + m21) / s,
            0.25 * s,
            (m10 - m01) / s,
        ])


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = scene.config

        # ----- physical parameters from scaffold -----
        rod_cfg = self.config["rod"]
        self.n = int(rod_cfg["vertex_count"])
        self.rod_mass = float(rod_cfg["mass"])
        self.rod_radius = float(rod_cfg["radius"])
        self.rod_length = float(rod_cfg["total_length"])
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        clamp_cfg = self.config["clamps"]
        self.left_pos = np.asarray(clamp_cfg["left"]["position"], dtype=float)
        self.right_pos = np.asarray(clamp_cfg["right"]["position"], dtype=float)
        self.twist_ramp = clamp_cfg["twist_ramp"]

        # ----- initial state -----
        init = scene.initial_state()
        self.x = init.position.copy().astype(float)
        self.v = init.velocity.copy().astype(float)
        self.mass = init.mass.copy().astype(float)
        self._base_attributes = {k: np.asarray(v, dtype=float).copy() for k, v in init.attributes.items()}

        # Keep controlled endpoints exactly on the clamps.
        self.x[0] = self.left_pos
        self.x[-1] = self.right_pos
        self.v[0] = 0.0
        self.v[-1] = 0.0

        # ----- discrete rod geometry -----
        self.e0 = np.diff(self.x, axis=0)
        self.l0 = np.linalg.norm(self.e0, axis=1)
        # Arc-length parameter for twist distribution.
        self.arc = np.zeros(self.n)
        for i in range(1, self.n):
            self.arc[i] = self.arc[i - 1] + self.l0[i - 1]
        self.L = self.arc[-1]
        if self.L < 1e-9:
            self.L = 1.0

        # Rest bending angles from the initial trefoil shape.
        self.rest_bend = self._compute_bend_angles(self.x)
        # Rest torsion (geometric dihedral) of the initial shape.
        self.rest_torsion = self._compute_torsion_angles(self.x)

        # ----- simulation constants -----
        self.dt = 1.0e-4  # seconds
        self.k_bend = 5.0e-2          # N*m
        self.k_torsion = 3.0e-2       # N*m
        self.k_contact = 8.0e3        # N/m  (penalty stiffness)
        self.c_contact = 0.5          # contact damping coefficient
        self.k_damp = 0.15            # velocity damping (kg/s)
        self.contact_dist = 2.0 * self.rod_radius

        self.time = 0.0
        self._prev_orientation: np.ndarray | None = None

        # Deterministic light asymmetry to seed the snap.
        np.random.seed(seed)
        idx = np.arange(self.n)
        # A tiny transverse perturbation breaks perfect reflection symmetry.
        perturb = np.zeros_like(self.v)
        perturb[:, 0] = 1.0e-5 * np.sin(7.0 * idx / self.n)
        perturb[:, 1] = 1.0e-5 * np.cos(5.0 * idx / self.n)
        self.v += perturb

    def diagnostics(self) -> dict[str, object]:
        return {
            "time_step": self.dt,
            "vertex_count": self.n,
            "contact_distance": self.contact_dist,
            "bending_stiffness": self.k_bend,
            "torsional_stiffness": self.k_torsion,
            "contact_stiffness": self.k_contact,
        }

    # ------------------------------------------------------------------ helpers
    def _ramp_angle(self, t: float) -> float:
        r = self.twist_ramp
        t0 = float(r["hold_twist_free_until"])
        t1 = float(r["ramp_start"])
        t2 = float(r["ramp_end"])
        turns = float(r["final_relative_turns"])
        if t <= t1:
            u = 0.0
        elif t >= t2:
            u = 1.0
        else:
            u = (t - t1) / (t2 - t1)
            u = u * u * (3.0 - 2.0 * u)
        return 2.0 * math.pi * turns * u

    def _compute_bend_angles(self, x: np.ndarray) -> np.ndarray:
        e = np.diff(x, axis=0)
        l = np.linalg.norm(e, axis=1, keepdims=True)
        l = np.where(l < 1e-12, 1.0, l)
        t = e / l
        out = np.zeros(self.n)
        for i in range(1, self.n - 1):
            c = float(np.dot(t[i - 1], t[i]))
            s = float(np.linalg.norm(np.cross(t[i - 1], t[i])))
            out[i] = math.atan2(s, c)
        return out

    def _compute_torsion_angles(self, x: np.ndarray) -> np.ndarray:
        nseg = self.n - 3
        out = np.zeros(nseg)
        for i in range(nseg):
            p1, p2, p3, p4 = x[i], x[i + 1], x[i + 2], x[i + 3]
            b1 = p2 - p1
            b2 = p3 - p2
            b3 = p4 - p3
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            a1 = np.linalg.norm(n1)
            a2 = np.linalg.norm(n2)
            if a1 < 1e-12 or a2 < 1e-12:
                out[i] = 0.0
                continue
            n1u = n1 / a1
            n2u = n2 / a2
            m = _normalize(b2)
            x_ = np.dot(n1u, n2u)
            y_ = np.dot(np.cross(n1u, n2u), m)
            out[i] = math.atan2(y_, x_)
        return out

    def _bending_forces(self, x: np.ndarray) -> np.ndarray:
        f = np.zeros_like(x)
        e = np.diff(x, axis=0)
        l = np.linalg.norm(e, axis=1, keepdims=True)
        l = np.where(l < 1e-12, 1.0, l)
        t = e / l
        for i in range(1, self.n - 1):
            t1 = t[i - 1]
            t2 = t[i]
            l1 = l[i - 1, 0]
            l2 = l[i, 0]
            c = np.dot(t1, t2)
            s = np.linalg.norm(np.cross(t1, t2))
            if s < 1e-10:
                continue
            theta = math.atan2(s, c)
            diff = theta - self.rest_bend[i]
            # gradients of cos(theta) wrt the three involved vertices
            g0 = (c * t1 - t2) / l1
            g2 = (t1 - c * t2) / l2
            g1 = -(g0 + g2)
            # d theta / d x = -(1/s) d cos(theta)/d x
            scale = self.k_bend * diff / s
            f[i - 1] += scale * g0
            f[i] += scale * g1
            f[i + 1] += scale * g2
        return f

    def _torsion_forces(self, x: np.ndarray, t: float) -> np.ndarray:
        f = np.zeros_like(x)
        nseg = self.n - 3
        if nseg <= 0:
            return f
        total_twist = self._ramp_angle(t)
        for i in range(nseg):
            p1, p2, p3, p4 = x[i], x[i + 1], x[i + 2], x[i + 3]
            b1 = p2 - p1
            b2 = p3 - p2
            b3 = p4 - p3
            n1 = np.cross(b1, b2)
            n2 = np.cross(b2, b3)
            a1sq = float(np.dot(n1, n1))
            a2sq = float(np.dot(n2, n2))
            if a1sq < 1e-16 or a2sq < 1e-16:
                continue
            n1u = n1 / np.sqrt(a1sq)
            n2u = n2 / np.sqrt(a2sq)
            m = _normalize(b2)
            x_ = np.dot(n1u, n2u)
            y_ = np.dot(np.cross(n1u, n2u), m)
            phi = math.atan2(y_, x_)
            # target dihedral increases linearly along the rod from rest.
            s_center = 0.25 * (self.arc[i] + self.arc[i + 1] + self.arc[i + 2] + self.arc[i + 3]) / self.L
            target = self.rest_torsion[i] + total_twist * s_center
            delta = phi - target
            delta = math.atan2(np.sin(delta), np.cos(delta))
            # signed scalar error
            # gradients of phi (known dihedral-gradient formula)
            b2_norm = np.linalg.norm(b2)
            g1 = -b2_norm * n1 / a1sq
            g4 = -b2_norm * n2 / a2sq
            b1sq = float(np.dot(b1, b1))
            b3sq = float(np.dot(b3, b3))
            if b1sq < 1e-16 or b3sq < 1e-16:
                continue
            A = float(np.dot(b1, b2)) / b1sq
            B = float(np.dot(b3, b2)) / b3sq
            g2 = (A - 1.0) * g1 - B * g4
            g3 = -A * g1 + (B - 1.0) * g4
            scale = -self.k_torsion * delta
            f[i] += scale * g1
            f[i + 1] += scale * g2
            f[i + 2] += scale * g3
            f[i + 3] += scale * g4
        return f

    def _contact_forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        f = np.zeros_like(x)
        if self.n < 6:
            return f
        tree = cKDTree(x)
        # query_pairs returns unordered pairs with 0 <= i < j < N
        pairs = tree.query_pairs(self.contact_dist + 0.05, output_type="ndarray")
        if pairs.size == 0:
            return f
        i = pairs[:, 0]
        j = pairs[:, 1]
        # exclude adjacent and near-adjacent segments
        sep = np.abs(j - i)
        mask = sep > 2
        if not np.any(mask):
            return f
        i = i[mask]
        j = j[mask]
        dx = x[j] - x[i]
        dists = np.linalg.norm(dx, axis=1)
        if np.any(dists == 0.0):
            return f
        nvec = dx / dists[:, None]
        rel_v = v[j] - v[i]
        overlap = self.contact_dist - dists
        # Hertz-like: force only when surfaces would intersect
        active = overlap > 0.0
        if not np.any(active):
            return f
        i = i[active]
        j = j[active]
        n_active = nvec[active]
        overlap = overlap[active]
        rel_v = rel_v[active]
        fn = self.k_contact * overlap[:, None] * n_active
        fd = self.c_contact * np.sum(rel_v * n_active, axis=1)[:, None] * n_active
        force = fn + fd
        np.add.at(f, i, -force)
        np.add.at(f, j, force)
        return f

    def _project_lengths(self, x: np.ndarray) -> None:
        # Gauss-Seidel-style edge-length projection (inextensibility).
        for _ in range(3):
            for i in range(self.n - 1):
                p1 = x[i]
                p2 = x[i + 1]
                e = p2 - p1
                l = np.linalg.norm(e)
                if l < 1e-12:
                    continue
                diff = (l - self.l0[i]) / l
                d = 0.5 * diff * e
                x[i] += d
                x[i + 1] -= d

    def _step(self, t: float) -> None:
        f = np.zeros_like(self.x)
        f += self.mass[:, None] * self.gravity
        f += self._bending_forces(self.x)
        f += self._torsion_forces(self.x, t)
        f += self._contact_forces(self.x, self.v)
        # velocity damping
        self.v = (1.0 - self.k_damp * self.dt) * self.v

        a = f / self.mass[:, None]
        self.v += a * self.dt
        self.x += self.v * self.dt

        # Inextensibility projection.
        self._project_lengths(self.x)

        # Kinematic clamps.
        self.x[0] = self.left_pos
        self.x[-1] = self.right_pos
        self.v[0] = 0.0
        self.v[-1] = 0.0

        self.time += self.dt

    # ---------------------------------------------------------------- trajectory
    def _orientation(self, x: np.ndarray, t: float) -> np.ndarray:
        # Local tangent per vertex.
        e = np.diff(x, axis=0)
        tangents = np.zeros_like(x)
        tangents[1:-1] = _normalize(e[:-1] + e[1:])
        tangents[0] = _normalize(e[0])
        tangents[-1] = _normalize(e[-1])

        total_twist = self._ramp_angle(t)
        q_out = np.zeros((self.n, 4))
        for i in range(self.n):
            t_i = tangents[i]
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(t_i, ref)) > 0.98:
                ref = np.array([1.0, 0.0, 0.0])
            u = ref - np.dot(ref, t_i) * t_i
            u = _normalize(u)[0]
            v = np.cross(t_i, u)
            R = np.column_stack((u, v, t_i))
            q_align = _quat_from_matrix(R)
            psi = total_twist * (self.arc[i] / self.L)
            q_twist = np.array([0.0, 0.0, math.sin(0.5 * psi), math.cos(0.5 * psi)])
            q_out[i] = _quat_multiply(q_align, q_twist)
        return q_out

    def _angular_velocity(self, q_prev: np.ndarray, q_cur: np.ndarray, dt: float) -> np.ndarray:
        if q_prev is None:
            return np.zeros((self.n, 3))
        dq = _quat_multiply(q_cur, np.stack([-q_prev[:, 0], -q_prev[:, 1], -q_prev[:, 2], q_prev[:, 3]], axis=1))
        # normalize to shortest path
        dq = np.where(dq[:, 3:4] < 0.0, -dq, dq)
        vec = dq[:, :3]
        w = dq[:, 3]
        sin_half = np.linalg.norm(vec, axis=1)
        angle = 2.0 * np.arctan2(sin_half, w)
        # avoid division by zero
        scale = np.zeros_like(sin_half)
        mask = sin_half > 1e-12
        scale[mask] = angle[mask] / sin_half[mask]
        axis = vec * scale[:, None]
        return axis / dt

    def advance_to(self, time: float) -> StateFrame:
        if time < self.time:
            time = self.time
        n_steps = int(round((time - self.time) / self.dt))
        for _ in range(n_steps):
            self._step(self.time)
        # If there is a tiny remainder due to rounding, take one more fractional step.
        if self.time < time - 1e-14:
            dt0 = self.dt
            frac = time - self.time
            self.dt = frac
            self._step(self.time)
            self.dt = dt0

        # Build per-vertex orientation and angular velocity.
        orientation = self._orientation(self.x, time)
        ang_vel = self._angular_velocity(self._prev_orientation, orientation, self.config["sample_dt"])
        self._prev_orientation = orientation.copy()

        # Required budget / constraint / scalar fields.
        G, K, C = 1, 1, self.n - 1
        edge_vecs = np.diff(self.x, axis=0)
        cur_len = np.linalg.norm(edge_vecs, axis=1)

        fields = {
            "budget_cumulative_external_angular_impulse": np.zeros((G, 3)),
            "budget_cumulative_external_impulse": np.zeros((G, 3)),
            "budget_cumulative_inflow_angular_momentum": np.zeros((G, 3)),
            "budget_cumulative_inflow_momentum": np.zeros((G, 3)),
            "budget_cumulative_outflow_angular_momentum": np.zeros((G, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((G, 3)),
            "budget_domain_weights": np.ones((G, self.n)),
            "budget_external_couple": np.zeros((G, K, 3)),
            "budget_external_force": np.zeros((G, K, 3)),
            "budget_force_position": np.zeros((G, K, 3)),
            "constraint_target": self.l0.copy(),
            "constraint_value": cur_len.copy(),
            "cumulative_mass_source": np.zeros((K,)),
        }
        total_gravity = np.sum(self.mass[:, None] * self.gravity, axis=0)
        com = np.sum(self.mass[:, None] * self.x, axis=0) / np.sum(self.mass)
        fields["budget_external_force"][0, 0] = total_gravity
        fields["budget_force_position"][0, 0] = com

        attributes = dict(self._base_attributes)
        attributes["inertia_tensor"] = np.zeros((self.n, 3, 3))
        attributes["mass_quantity"] = self.mass.copy()

        scalars = {
            "constraint_scale": float(np.mean(self.l0)),
            "reference_angular_momentum": 1.0,
            "reference_momentum": 1.0,
        }

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=orientation,
            angular_velocity=ang_vel,
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame
