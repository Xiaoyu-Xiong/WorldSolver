"""Discrete Cosserat-rod solver for the hanging-rod plectoneme scene.

The rod is modelled as a chain of rigid cylindrical segments with centreline
positions and orientations.  Stretch, bending and torsion are penalised,
gravity and ambient drag act on the centreline, and a finite-thickness
vertex-vertex repulsion handles self-contact.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# Quaternion helpers (scalar-last xyzw convention)
# ---------------------------------------------------------------------------
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=float,
    )


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=float)


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    half = 0.5 * angle
    s = math.sin(half)
    c = math.cos(half)
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    return np.array([n[0] * s, n[1] * s, n[2] * s, c], dtype=float)


def _rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) v by quaternion q.  Supports broadcasting."""
    qv = q[..., :3]
    qs = q[..., 3]
    cross1 = np.cross(qv, v)
    if qv.ndim == 1:
        return v + 2.0 * np.cross(qv, cross1 + qs * v)
    return v + 2.0 * np.cross(qv, cross1 + qs[:, None] * v)


def _matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a scalar-last quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    return q


def _rest_orientation(tangent: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Build a quaternion that maps local z to tangent and local y to a reference."""
    z = tangent / np.linalg.norm(tangent)
    if ref is None:
        ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z, ref)) > 0.99:
        ref = np.array([0.0, 0.0, 1.0])
    y = ref - np.dot(ref, z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    R = np.column_stack((x, y, z))  # columns are local axes in world
    return _matrix_to_quat(R)


def _quat_to_rotvec(delta: np.ndarray) -> np.ndarray:
    """Rotation vector of a quaternion (xyzw)."""
    vec = delta[:3]
    scalar = delta[3]
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros(3)
    angle = 2.0 * math.atan2(norm, scalar)
    return vec / norm * angle


class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        self._seed = seed
        self._scene = scene
        self._config = scene.config
        self._rng = np.random.default_rng(seed)

        # Initial geometry from the scene
        init_frame = scene.initial_state()
        self._x = np.array(init_frame.position, dtype=float, copy=True)
        self._v = np.zeros_like(self._x)
        self._n = self._x.shape[0]
        self._m = self._n - 1

        # Masses and edge masses
        self._m_vert = np.array(init_frame.mass, dtype=float, copy=True)
        self._m_edge = 0.5 * (self._m_vert[:-1] + self._m_vert[1:])

        # Rest edge lengths from initial shape
        self._l0 = np.linalg.norm(self._x[1:] - self._x[:-1], axis=1)

        # Rest orientations
        self._q_rest = np.zeros((self._m, 4))
        for i in range(self._m):
            self._q_rest[i] = _rest_orientation(self._x[i + 1] - self._x[i])

        # Current orientations = rest, zero angular velocity
        self._q = np.array(self._q_rest, copy=True)
        self._omega = np.zeros((self._m, 3))

        # Fixed endpoint positions
        controls = self._config["end_controls"]
        self._left_pos = np.array(controls["left"]["position"], dtype=float)
        self._right_pos = np.array(controls["right"]["position"], dtype=float)

        # Apply fixed positions explicitly
        self._x[0] = self._left_pos
        self._x[-1] = self._right_pos

        # Physical parameters
        self._gravity = np.array(self._config["gravity"], dtype=float)
        drag_cfg = self._config.get("ambient_drag", {})
        self._drag_coeff = float(drag_cfg.get("coefficient", 0.035)) if drag_cfg.get("enabled") else 0.0

        rod_cfg = self._config["rod"]
        self._radius = float(rod_cfg["radius"])
        self._contact_diameter = 2.0 * self._radius

        # Stiffnesses - chosen to produce twist buckling on the 7.35 m rod
        # while remaining stable with the explicit sub-step used below.
        self._k_stretch = 2000.0
        self._k_bend = 0.02
        self._k_twist = 0.015
        self._k_contact = 500.0
        self._d_contact = 2.0

        # Inertia of each cylindrical edge segment (body frame, diagonal)
        r = self._radius
        I_axial = 0.5 * self._m_edge * r * r
        I_perp = (1.0 / 12.0) * self._m_edge * (3.0 * r * r + self._l0 * self._l0)
        self._I_body = np.stack([I_perp, I_perp, I_axial], axis=1)  # (M,3)

        # Time-stepping
        self._sample_dt = float(self._config["sample_dt"])
        # Use a small fixed sub-step; sample_dt is a multiple.
        self._dt = self._sample_dt / 40.0  # 0.0005 s for sample_dt = 0.02

        # Cumulative budget quantities
        self._cum_external_impulse = np.zeros((3, 3))
        self._cum_external_ang_impulse = np.zeros((3, 3))
        self._cum_inflow_momentum = np.zeros((3, 3))
        self._cum_outflow_momentum = np.zeros((3, 3))
        self._cum_inflow_ang_momentum = np.zeros((3, 3))
        self._cum_outflow_ang_momentum = np.zeros((3, 3))

        # Domain masks (rows are independent budgets)
        n = self._n
        self._domain_weights = np.zeros((3, n), dtype=float)
        self._domain_weights[0, :] = 1.0  # whole rod
        mid = n // 2
        self._domain_weights[1, :mid] = 1.0
        self._domain_weights[2, mid:] = 1.0

        self._current_time = 0.0
        self._frame_cache: dict[float, StateFrame] = {}
        self._initialized = True

    def _end_targets(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return target end orientation quaternions at time t."""
        targets = self._scene.end_control_targets(t)
        left = targets["left"]
        right = targets["right"]
        q_left = _quat_mul(
            _quat_from_axis_angle(left["rotation_axis"], left["angle_radians"]),
            self._q_rest[0],
        )
        q_right = _quat_mul(
            _quat_from_axis_angle(right["rotation_axis"], right["angle_radians"]),
            self._q_rest[-1],
        )
        return q_left, q_right

    def _apply_kinematic_constraints(self, t: float) -> None:
        self._x[0] = self._left_pos
        self._x[-1] = self._right_pos
        q_left, q_right = self._end_targets(t)
        self._q[0] = q_left
        self._q[-1] = q_right
        self._v[0] = 0.0
        self._v[-1] = 0.0
        self._omega[0] = 0.0
        self._omega[-1] = 0.0

    def _compute_stretch(self, f_vert: np.ndarray, tau: np.ndarray) -> None:
        """Penalty stretch coupling positions to orientations."""
        local_z = _rotate_vector(self._q, np.array([0.0, 0.0, 1.0]))
        desired = local_z * self._l0[:, None]
        actual = self._x[1:] - self._x[:-1]
        diff = actual - desired
        f = self._k_stretch * diff

        f_vert[:-1] += f
        f_vert[1:] -= f

        # Torque on each edge from the stretch spring
        tau += self._k_stretch * self._l0[:, None] * np.cross(local_z, diff)

    def _compute_bend_twist(self, tau: np.ndarray) -> None:
        """Rotational springs between adjacent edge orientations."""
        # Pre-compute rest relative rotations in world frame
        if not hasattr(self, "_rest_rel"):
            qL = self._q_rest[:-1]
            qR = self._q_rest[1:]
            # delta_rest = qR * qL^{-1}
            delta_rest = np.zeros((self._m - 1, 4))
            for i in range(self._m - 1):
                delta_rest[i] = _quat_mul(qR[i], _quat_inv(qL[i]))
            self._rest_rel = delta_rest
            # Average edge length at each joint
            self._lbar = 0.5 * (self._l0[:-1] + self._l0[1:])

        qL = self._q[:-1]
        qR = self._q[1:]
        for i in range(self._m - 1):
            delta = _quat_mul(qR[i], _quat_inv(qL[i]))
            # strain rotation relative to rest
            strain_q = _quat_mul(_quat_inv(self._rest_rel[i]), delta)
            phi = _quat_to_rotvec(strain_q)

            # Project rotation into twist (along current tangent) and bending parts.
            # Use the left edge's local z as the common tangent approximation.
            local_z = _rotate_vector(qL[i], np.array([0.0, 0.0, 1.0]))
            t = local_z / (np.linalg.norm(local_z) + 1e-12)
            phi_twist = np.dot(phi, t) * t
            phi_bend = phi - phi_twist

            torque = -(self._k_bend * phi_bend + self._k_twist * phi_twist) / self._lbar[i]
            tau[i] += -torque
            tau[i + 1] += torque

    def _compute_contact(self, f_vert: np.ndarray) -> None:
        """Finite-thickness repulsion between non-adjacent vertices."""
        tree = cKDTree(self._x)
        pairs = tree.query_pairs(r=self._contact_diameter + 0.005, output_type="ndarray")
        if pairs.size == 0:
            return
        i = pairs[:, 0]
        j = pairs[:, 1]
        # ignore adjacent vertices along the chain
        adjacent = np.abs(i - j) <= 2
        pairs = pairs[~adjacent]
        if pairs.size == 0:
            return
        i = pairs[:, 0]
        j = pairs[:, 1]
        dvec = self._x[j] - self._x[i]
        d = np.linalg.norm(dvec, axis=1)
        if d.size == 0:
            return
        mask = d < self._contact_diameter
        if not np.any(mask):
            return
        i = i[mask]
        j = j[mask]
        d = d[mask]
        dvec = dvec[mask]
        n = dvec / d[:, None]
        penetration = self._contact_diameter - d
        # add velocity damping in the contact direction
        vdiff = self._v[j] - self._v[i]
        if vdiff.ndim == 1:
            vdiff = vdiff[None, :]
        vn = np.einsum("ij,ij->i", vdiff, n)
        fmag = self._k_contact * penetration - self._d_contact * vn
        fmag = np.maximum(fmag, 0.0)
        f = fmag[:, None] * n
        # apply to vertices
        np.add.at(f_vert, i, -f)
        np.add.at(f_vert, j, f)

    def _integrate_substep(self, t: float) -> None:
        n = self._n
        f_vert = np.zeros((n, 3), dtype=float)
        tau = np.zeros((self._m, 3), dtype=float)

        # Gravity and drag
        f_grav = self._m_vert[:, None] * self._gravity
        f_drag = -self._drag_coeff * self._v
        f_vert += f_grav + f_drag

        # Elastic and contact forces
        self._compute_stretch(f_vert, tau)
        self._compute_bend_twist(tau)
        self._compute_contact(f_vert)

        # Update vertices
        self._v += self._dt * f_vert / self._m_vert[:, None]
        self._x += self._dt * self._v

        # Update orientations
        # alpha in body frame to respect anisotropic inertia
        alpha_body = _rotate_vector(_quat_inv(self._q), tau) / self._I_body
        alpha_world = _rotate_vector(self._q, alpha_body)
        self._omega += self._dt * alpha_world

        # Quaternion integration
        omega_q = np.concatenate([self._omega, np.zeros((self._m, 1))], axis=1)
        q_dot = 0.5 * np.array(
            [_quat_mul(omega_q[i], self._q[i]) for i in range(self._m)]
        )
        self._q += self._dt * q_dot
        norms = np.linalg.norm(self._q, axis=1, keepdims=True)
        self._q /= norms

        # Budgets: whole rod only for impulse accounting
        # Domain rows: 0 whole, 1 left half, 2 right half
        for g in range(3):
            w = self._domain_weights[g, :]
            idx = np.where(w > 0)[0]
            if idx.size == 0:
                continue
            # gravity and drag external forces
            F_g = f_grav[idx].sum(axis=0)
            F_d = f_drag[idx].sum(axis=0)
            self._cum_external_impulse[g] += (F_g + F_d) * self._dt
            # angular impulse about inertial origin
            r = self._x[idx]
            self._cum_external_ang_impulse[g] += (
                np.cross(r, f_grav[idx]).sum(axis=0) + np.cross(r, f_drag[idx]).sum(axis=0)
            ) * self._dt

        # Enforce kinematic end conditions
        self._apply_kinematic_constraints(t + self._dt)

    def _build_frame(self, t: float) -> StateFrame:
        # Vertex orientation: map edge orientations onto vertices
        orient = np.zeros((self._n, 4), dtype=float)
        orient[:-1] = self._q
        orient[-1] = self._q[-1]
        # normalise just in case
        orient /= np.linalg.norm(orient, axis=1, keepdims=True)

        ang_vel = np.zeros((self._n, 3), dtype=float)
        ang_vel[:-1] = self._omega
        ang_vel[-1] = self._omega[-1]

        # Inertia tensor per sample in world coordinates
        inertia = np.zeros((self._n, 3, 3), dtype=float)
        for i in range(self._n - 1):
            R = np.column_stack(
                (
                    _rotate_vector(self._q[i], np.array([1.0, 0.0, 0.0])),
                    _rotate_vector(self._q[i], np.array([0.0, 1.0, 0.0])),
                    _rotate_vector(self._q[i], np.array([0.0, 0.0, 1.0])),
                )
            )
            Ib = np.diag(self._I_body[i])
            inertia[i] = R @ Ib @ R.T
        inertia[-1] = inertia[-2]

        # Domain budgets (current external force and positions)
        G, K = 3, 2
        ext_force = np.zeros((G, K, 3), dtype=float)
        force_pos = np.zeros((G, K, 3), dtype=float)
        f_grav = self._m_vert[:, None] * self._gravity
        f_drag = -self._drag_coeff * self._v
        for g in range(G):
            idx = np.where(self._domain_weights[g, :] > 0)[0]
            if idx.size == 0:
                continue
            ext_force[g, 0] = f_grav[idx].sum(axis=0)
            ext_force[g, 1] = f_drag[idx].sum(axis=0)
            force_pos[g, 0] = self._x[idx].mean(axis=0)
            force_pos[g, 1] = self._x[idx].mean(axis=0)

        # Constraints: left/right position and orientation targets
        q_left, q_right = self._end_targets(t)
        constraint_value = np.zeros((4, 4), dtype=float)
        constraint_target = np.zeros((4, 4), dtype=float)
        # positions padded to length 4
        constraint_value[0, :3] = self._x[0]
        constraint_target[0, :3] = self._left_pos
        constraint_value[1, :3] = self._x[-1]
        constraint_target[1, :3] = self._right_pos
        # orientations
        constraint_value[2] = orient[0]
        constraint_target[2] = q_left
        constraint_value[3] = orient[-1]
        constraint_target[3] = q_right

        attributes = {
            "inertia_tensor": inertia,
            "mass_quantity": np.array(self._m_vert),
            "rod_radius": np.full(self._n, self._radius),
            "contact_diameter": np.full(self._n, self._contact_diameter),
        }

        fields = {
            "budget_cumulative_external_angular_impulse": np.array(self._cum_external_ang_impulse),
            "budget_cumulative_external_impulse": np.array(self._cum_external_impulse),
            "budget_cumulative_inflow_angular_momentum": np.array(self._cum_inflow_ang_momentum),
            "budget_cumulative_inflow_momentum": np.array(self._cum_inflow_momentum),
            "budget_cumulative_outflow_angular_momentum": np.array(self._cum_outflow_ang_momentum),
            "budget_cumulative_outflow_momentum": np.array(self._cum_outflow_momentum),
            "budget_domain_weights": np.array(self._domain_weights),
            "budget_external_couple": np.zeros((G, K, 3), dtype=float),
            "budget_external_force": ext_force,
            "budget_force_position": force_pos,
            "constraint_target": constraint_target,
            "constraint_value": constraint_value,
            "cumulative_mass_source": np.zeros(1, dtype=float),
        }

        scalars = {
            "constraint_scale": 1.0,
            "reference_momentum": 1.0,
            "reference_angular_momentum": 7.2,
        }

        frame = StateFrame(
            time=t,
            position=np.array(self._x),
            velocity=np.array(self._v),
            orientation=orient,
            angular_velocity=ang_vel,
            mass=np.array(self._m_vert),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        if time in self._frame_cache:
            return self._frame_cache[time]

        target = float(time)
        if target < self._current_time:
            # Should not happen with sequential sample times; reset to start.
            self.__init__()  # type: ignore[misc]

        n_sub = int(round((target - self._current_time) / self._dt))
        # Safety: avoid infinite/negative steps
        if n_sub < 0:
            n_sub = 0

        for _ in range(n_sub):
            self._integrate_substep(self._current_time)
            self._current_time += self._dt

        # If target is not an exact multiple of dt, take a final fractional step
        remaining = target - self._current_time
        if remaining > 1e-12:
            old_dt = self._dt
            self._dt = remaining
            self._integrate_substep(self._current_time)
            self._current_time = target
            self._dt = old_dt

        frame = self._build_frame(target)
        self._frame_cache[target] = frame
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "seed": self._seed,
            "dt": self._dt,
            "vertices": self._n,
            "edges": self._m,
        }
