"""Discrete Cosserat rod: twist localization in a V-shaped anisotropic rod.

Model
-----
The rod is discretised into ``ns = N-1`` rigid edge segments, each carrying a
material frame stored as a unit quaternion (xyzw).  Neighbouring segments are
coupled by

* a stiff penalty ball joint with strain-rate damping, enforcing centerline
  continuity and (near) inextensibility, and
* an elastic bend/twist couple  ``tau = R_j (K e_j) / l``  where ``e_j`` is the
  rotation vector of the reference-relative frame mismatch and
  ``K = diag(E I1, E I2, G J)`` is the anisotropic Kirchhoff stiffness with
  ``I1 = w t^3/12`` (easy axis), ``I2 = t w^3/12`` (hard axis, 18.8x stiffer).

The natural configuration is the frozen untwisted V, so the apex kink is stress
free and only deviations from it store energy.  Because the apex carries a large
natural curvature while ``E I2 >> E I1``, rotating the material frame through
the bend converts a natural easy-axis bend into a hard-axis bend.  That
anisotropic barrier is the mechanism that first localises the imposed rotation
on the driven arm and only later lets twist pass the bend (partly by trading
twist for out-of-plane writhe, which the free arms can do).  Nothing is
scripted: the only prescribed data are the two clamps and the frozen rotation
schedule.

Dissipation is a viscous medium drag (``-c_d m v``, ``-c_w I w``), reported as
an external budget term, plus internal strain-rate damping that cancels
pairwise within the rod domain.

Integration is semi-implicit (symplectic) Euler on the segment rigid-body
dynamics at a fixed sub-step well below the stiffest joint period.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-12


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternion arrays."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(np.broadcast(a, b).shape, dtype=float)
    out[..., 0] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 1] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 2] = aw * bz + ax * by - ay * bx + az * bw
    out[..., 3] = aw * bw - ax * bx - ay * by - az * bz
    return out


def _qconj(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=float, copy=True)
    out[..., :3] *= -1.0
    return out


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v by quaternions q (xyzw)."""
    u = q[..., :3]
    w = q[..., 3:4]
    uv = np.cross(u, v)
    return v + 2.0 * (w * uv + np.cross(u, uv))


def _qnormalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, _EPS)


def _rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector of xyzw quaternions, taking the short arc."""
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    v = q[..., :3]
    w = np.clip(q[..., 3], -1.0, 1.0)
    s = np.linalg.norm(v, axis=-1)
    angle = 2.0 * np.arctan2(s, w)
    scale = np.where(s > 1e-9, angle / np.maximum(s, _EPS), 2.0)
    return v * scale[..., None]


def _mat_to_quat(m: np.ndarray) -> np.ndarray:
    """Convert [...,3,3] rotation matrices to xyzw quaternions (Shepperd)."""
    m = np.asarray(m, dtype=float)
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    diag = np.stack((m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]), axis=-1)
    use_trace = trace > np.max(diag, axis=-1)
    # Trace-dominant branch.
    s = np.sqrt(np.maximum(1.0 + trace, _EPS)) * 2.0
    qt = np.stack(
        (
            (m[..., 2, 1] - m[..., 1, 2]) / s,
            (m[..., 0, 2] - m[..., 2, 0]) / s,
            (m[..., 1, 0] - m[..., 0, 1]) / s,
            0.25 * s,
        ),
        axis=-1,
    )
    # Largest-diagonal branch, evaluated for each axis then selected.
    idx = np.argmax(diag, axis=-1)
    qd = np.zeros_like(qt)
    for axis in range(3):
        j = (axis + 1) % 3
        k = (axis + 2) % 3
        sa = np.sqrt(np.maximum(1.0 + diag[..., axis] - diag[..., j] - diag[..., k], _EPS)) * 2.0
        cand = np.empty_like(qt)
        cand[..., axis] = 0.25 * sa
        cand[..., j] = (m[..., j, axis] + m[..., axis, j]) / sa
        cand[..., k] = (m[..., k, axis] + m[..., axis, k]) / sa
        cand[..., 3] = (m[..., k, j] - m[..., j, k]) / sa
        qd = np.where((idx == axis)[..., None], cand, qd)
    q = np.where(use_trace[..., None], qt, qd)
    return _qnormalize(q)
class BenchmarkSolver:
    """Explicit Cosserat-rod solver for the frozen V-shaped anisotropic rod."""

    # Sub-step chosen ~7x below the stability limit of the stiffest joint mode.
    DT_TARGET = 1.25e-5

    def initialize(self, scene, seed: int) -> None:
        config = dict(scene.config)
        self.config = config
        rod = config["rod"]
        bc = config["boundary_conditions"]

        self.n_vertices = nv = int(rod["vertex_count"])
        self.apex = int(rod["apex_vertex"])
        self.width = float(rod["cross_section_width"])
        self.thickness = float(rod["cross_section_thickness"])
        self.density = float(rod["density"])
        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        # ---- frozen reference centerline (identical construction to scene.py)
        control = np.asarray(config["initial_conditions"]["centerline_points"], dtype=float)
        left = np.linspace(control[0], control[1], self.apex + 1)
        right = np.linspace(control[1], control[2], nv - self.apex)[1:]
        self.x_ref = np.vstack((left, right))

        edges = np.diff(self.x_ref, axis=0)
        self.l0 = np.linalg.norm(edges, axis=1)
        self.ns = ns = nv - 1
        self.arc_length = np.r_[0.0, np.cumsum(self.l0)]

        # ---- reference material frames: d1 = width axis (+y), d3 = tangent
        d3 = edges / self.l0[:, None]
        d1 = np.tile(np.array([0.0, 1.0, 0.0]), (ns, 1))
        d1 = d1 - d3 * np.sum(d1 * d3, axis=1, keepdims=True)
        d1 /= np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), _EPS)
        d2 = np.cross(d3, d1)
        rot_ref = np.stack((d1, d2, d3), axis=-1)  # columns are the directors
        self.q_ref = _mat_to_quat(rot_ref)

        # Reference relative rotations: identity everywhere but the apex kink,
        # so the untwisted V is the stress-free natural configuration.
        self.q_rel_ref = _qmul(_qconj(self.q_ref[:-1]), self.q_ref[1:])

        # ---- anisotropic Kirchhoff stiffness of the flat cross-section
        area = self.width * self.thickness
        young = 1.0e7
        poisson = 0.5
        shear = young / (2.0 * (1.0 + poisson))
        i1 = self.width * self.thickness**3 / 12.0  # easy axis (about d1)
        i2 = self.thickness * self.width**3 / 12.0  # hard axis (about d2)
        ratio = self.thickness / self.width
        torsion = (
            self.width
            * self.thickness**3
            * (1.0 / 3.0 - 0.21 * ratio * (1.0 - ratio**4 / 12.0))
        )
        self.stiffness = np.array([young * i1, young * i2, shear * torsion])
        self.young = young
        self.area = area

        # ---- segment mass and body-frame inertia (rectangular prism)
        self.seg_mass = self.density * area * self.l0
        w2 = self.width**2
        t2 = self.thickness**2
        l2 = self.l0**2
        self.inertia = np.stack(
            (
                self.seg_mass * (t2 + l2) / 12.0,
                self.seg_mass * (w2 + l2) / 12.0,
                self.seg_mass * (w2 + t2) / 12.0,
            ),
            axis=-1,
        )

        # ---- joint penalty and damping coefficients
        mean_l0 = float(np.mean(self.l0))
        self.mean_l0 = mean_l0
        self.l_joint = 0.5 * (self.l0[:-1] + self.l0[1:])
        self.k_joint = 1.5 * young * area / mean_l0
        self.c_joint = 3.0
        self.c_rot = 2.0e-4
        self.drag_linear = 4.0
        self.drag_angular = 40.0

        # ---- clamps
        self.driven_seg = 0
        self.fixed_seg = ns - 1
        schedule = np.asarray(bc["driven_end"]["rotation_schedule"], dtype=float)
        self.sched_t = schedule[:, 0]
        self.sched_a = schedule[:, 1]
        self.drive_axis = d3[self.driven_seg].copy()
        self.q_driven_ref = self.q_ref[self.driven_seg].copy()

        # ---- state: rigid segment centers, frames, and velocities
        self.x = 0.5 * (self.x_ref[:-1] + self.x_ref[1:])
        self.q = self.q_ref.copy()
        self.v = np.zeros((ns, 3))
        self.w = np.zeros((ns, 3))
        self.x_kin = self.x[[self.driven_seg, self.fixed_seg]].copy()
        self.dynamic = np.ones(ns, dtype=bool)
        self.dynamic[[self.driven_seg, self.fixed_seg]] = False

        self._apply_drive(0.0)
        self.time = 0.0
        self.steps = 0
        self.twist_raw = np.zeros(ns)
        self.twist_acc = np.zeros(ns)
        self.vertex_mass = self._vertex_mass()
        self.reference_force = float(self.stiffness[1] / mean_l0**2)
        self.reference_torque = float(self.stiffness[1] / mean_l0)
        self.max_speed = 0.0
        self.max_strain = 0.0
    # ------------------------------------------------------------------ setup
    def _vertex_mass(self) -> np.ndarray:
        """Lumped vertex masses, identical to the scene's initial state."""
        vertex_lengths = np.empty(self.n_vertices)
        vertex_lengths[0] = 0.5 * self.l0[0]
        vertex_lengths[-1] = 0.5 * self.l0[-1]
        vertex_lengths[1:-1] = 0.5 * (self.l0[:-1] + self.l0[1:])
        return self.density * self.area * vertex_lengths

    def _drive_angle(self, t: float) -> tuple[float, float]:
        """Prescribed end rotation and its rate from the frozen schedule."""
        angle = float(np.interp(t, self.sched_t, self.sched_a))
        rate = 0.0
        for i in range(len(self.sched_t) - 1):
            t0, t1 = self.sched_t[i], self.sched_t[i + 1]
            if t0 <= t <= t1 and t1 > t0:
                rate = float((self.sched_a[i + 1] - self.sched_a[i]) / (t1 - t0))
                break
        return angle, rate

    def _apply_drive(self, t: float) -> None:
        """Impose both clamps: driven material-frame rotation and fixed end."""
        angle, rate = self._drive_angle(t)
        self.drive_angle = angle
        half = 0.5 * angle
        q_axis = np.array(
            [
                self.drive_axis[0] * np.sin(half),
                self.drive_axis[1] * np.sin(half),
                self.drive_axis[2] * np.sin(half),
                np.cos(half),
            ]
        )
        d = self.driven_seg
        f = self.fixed_seg
        self.q[d] = _qnormalize(_qmul(q_axis, self.q_driven_ref))
        self.x[d] = self.x_kin[0]
        self.v[d] = 0.0
        self.w[d] = rate * self.drive_axis
        self.q[f] = self.q_ref[f]
        self.x[f] = self.x_kin[1]
        self.v[f] = 0.0
        self.w[f] = 0.0

    # ----------------------------------------------------------------- forces
    def _directors(self, q: np.ndarray) -> np.ndarray:
        e3 = np.zeros((self.ns, 3))
        e3[:, 2] = 1.0
        return _qrot(q, e3)

    def _forces(self):
        q, x, v, w = self.q, self.x, self.v, self.w
        force = np.zeros((self.ns, 3))
        torque = np.zeros((self.ns, 3))

        # Anisotropic bend/twist couple from the reference-relative mismatch.
        q_rel = _qmul(_qconj(q[:-1]), q[1:])
        strain = _rotvec(_qmul(_qconj(self.q_rel_ref), q_rel))
        moment = _qrot(q[:-1], strain * self.stiffness / self.l_joint[:, None])
        moment = moment + self.c_rot * (w[1:] - w[:-1])
        torque[:-1] += moment
        torque[1:] -= moment

        # Penalty ball joints keep the centerline continuous and inextensible.
        arm = 0.5 * self.l0[:, None] * self._directors(q)
        end = x + arm
        start = x - arm
        gap = start[1:] - end[:-1]
        vel_end = v[:-1] + np.cross(w[:-1], arm[:-1])
        vel_start = v[1:] + np.cross(w[1:], -arm[1:])
        joint = self.k_joint * gap + self.c_joint * (vel_start - vel_end)
        force[:-1] += joint
        force[1:] -= joint
        torque[:-1] += np.cross(arm[:-1], joint)
        torque[1:] += np.cross(-arm[1:], -joint)

        # External medium drag (plus gravity, zero in this scene).
        f_ext = -self.drag_linear * self.seg_mass[:, None] * v
        f_ext = f_ext + self.seg_mass[:, None] * self.gravity
        w_body = _qrot(_qconj(q), w)
        t_ext = -self.drag_angular * _qrot(q, self.inertia * w_body)
        force += f_ext
        torque += t_ext
        return force, torque, f_ext, t_ext

    def _step(self, dt: float) -> None:
        force, torque, _, _ = self._forces()
        dyn = self.dynamic
        q = self.q
        w_body = _qrot(_qconj(q), self.w)
        t_body = _qrot(_qconj(q), torque)
        acc_body = (t_body - np.cross(w_body, self.inertia * w_body)) / self.inertia
        acc_world = _qrot(q, acc_body)

        self.v[dyn] += dt * force[dyn] / self.seg_mass[dyn, None]
        self.w[dyn] += dt * acc_world[dyn]
        self.x[dyn] += dt * self.v[dyn]
        spin = np.zeros((self.ns, 4))
        spin[:, :3] = self.w
        self.q = _qnormalize(q + 0.5 * dt * _qmul(spin, q))
        self.time += dt
        self.steps += 1
        self._apply_drive(self.time)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.time + 1e-15:
            n_sub = int(np.ceil((target - self.time) / self.DT_TARGET))
            n_sub = max(n_sub, 1)
            dt = (target - self.time) / n_sub
            for _ in range(n_sub):
                self._step(dt)
            self.time = target
            self._apply_drive(target)
        return self._frame(target)
    # ------------------------------------------------------------------ export
    def _frame(self, t: float) -> StateFrame:
        ns = self.ns
        nv = self.n_vertices
        q, x, v, w = self.q, self.x, self.v, self.w
        force, torque, f_ext, t_ext = self._forces()

        d3 = self._directors(q)
        arm = 0.5 * self.l0[:, None] * d3
        p_end = x + arm
        p_start = x - arm
        v_end = v + np.cross(w, arm)
        v_start = v + np.cross(w, -arm)

        position = np.empty((nv, 3))
        position[0] = p_start[0]
        position[-1] = p_end[-1]
        position[1:-1] = 0.5 * (p_end[:-1] + p_start[1:])
        velocity = np.empty((nv, 3))
        velocity[0] = v_start[0]
        velocity[-1] = v_end[-1]
        velocity[1:-1] = 0.5 * (v_end[:-1] + v_start[1:])

        # Orientation relative to the frozen reference frame: identity at t = 0,
        # exactly as the scene's untwisted initial state declares.
        q_rel = _qmul(q, _qconj(self.q_ref))
        orientation = np.empty((nv, 4))
        orientation[0] = q_rel[0]
        orientation[-1] = q_rel[-1]
        a, b = q_rel[:-1], q_rel[1:]
        sign = np.where(np.sum(a * b, axis=1, keepdims=True) < 0.0, -1.0, 1.0)
        orientation[1:-1] = _qnormalize(a + sign * b)
        angular_velocity = np.empty((nv, 3))
        angular_velocity[0] = w[0]
        angular_velocity[-1] = w[-1]
        angular_velocity[1:-1] = 0.5 * (w[:-1] + w[1:])

        # Material twist about the current tangent, unwrapped in time so full
        # turns accumulate instead of folding back into (-pi, pi].
        psi = 2.0 * np.arctan2(np.sum(q_rel[:, :3] * d3, axis=1), q_rel[:, 3])
        delta = psi - self.twist_raw
        delta -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
        self.twist_acc = self.twist_acc + delta
        self.twist_raw = psi
        twist_v = np.empty(nv)
        twist_v[0] = self.twist_acc[0]
        twist_v[-1] = self.twist_acc[-1]
        twist_v[1:-1] = 0.5 * (self.twist_acc[:-1] + self.twist_acc[1:])

        # ---- momentum/angular-momentum budget for the whole-rod domain.
        # Clamp reactions are the constraint forces and couples that hold the
        # two kinematic end segments; drag is reported per segment.  Internal
        # elastic and joint pairs cancel inside this single combined domain.
        n_terms = 2 + ns
        b_force = np.zeros((1, n_terms, 3))
        b_pos = np.zeros((1, n_terms, 3))
        b_couple = np.zeros((1, n_terms, 3))
        d_i, f_i = self.driven_seg, self.fixed_seg
        b_force[0, 0] = -force[d_i]
        b_pos[0, 0] = x[d_i]
        b_couple[0, 0] = -torque[d_i]
        b_force[0, 1] = -force[f_i]
        b_pos[0, 1] = x[f_i]
        b_couple[0, 1] = -torque[f_i]
        b_force[0, 2:] = f_ext
        b_pos[0, 2:] = x
        b_couple[0, 2:] = t_ext

        # ---- inextensibility of the exported centerline
        seg_len = np.linalg.norm(np.diff(position, axis=0), axis=1)
        strain = float(np.max(np.abs(seg_len / self.l0 - 1.0)))
        self.max_strain = max(self.max_strain, strain)
        self.max_speed = max(self.max_speed, float(np.max(np.linalg.norm(velocity, axis=1))))

        attributes = {
            "arc_length": self.arc_length,
            "material_twist": twist_v,
            "cross_section_width": np.full(nv, self.width),
            "cross_section_thickness": np.full(nv, self.thickness),
            "is_apex": (np.arange(nv) == self.apex).astype(float),
            "is_driven_end": (np.arange(nv) == 0).astype(float),
            "is_fixed_end": (np.arange(nv) == nv - 1).astype(float),
            "mass_quantity": self.vertex_mass.copy(),
        }
        fields = {
            "budget_external_force": b_force,
            "budget_force_position": b_pos,
            "budget_external_couple": b_couple,
            "constraint_value": seg_len,
            "constraint_target": self.l0.copy(),
            "cumulative_mass_source": np.zeros(1),
        }
        scalars = {
            "driven_end_angle": float(self.drive_angle),
            "constraint_scale": float(self.mean_l0),
            "reference_force": self.reference_force,
            "reference_torque": self.reference_torque,
        }
        frame = StateFrame(
            time=float(t),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.vertex_mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "discrete Cosserat rod, semi-implicit Euler, penalty ball joints",
            "sub_steps": int(self.steps),
            "sub_step_dt": float(self.DT_TARGET),
            "segments": int(self.ns),
            "bend_stiffness_easy": float(self.stiffness[0]),
            "bend_stiffness_hard": float(self.stiffness[1]),
            "twist_stiffness": float(self.stiffness[2]),
            "anisotropy_ratio": float(self.stiffness[1] / self.stiffness[0]),
            "joint_stiffness": float(self.k_joint),
            "max_axial_strain": float(self.max_strain),
            "max_vertex_speed": float(self.max_speed),
            "driven_end_angle_final": float(self.drive_angle),
            "twist_driven_arm": float(self.twist_acc[2]),
            "twist_fixed_arm": float(self.twist_acc[self.ns - 3]),
            "total_mass": float(np.sum(self.vertex_mass)),
        }
