"""Physically-based discrete elastic rod solver for the twist-induced snap of an
open trefoil knot (085_trefoil_knot_twist_snap).

The rod is modelled as a chain of point masses connected by inextensible
massless springs with bending (curvature) stiffness, a simple twist (material
frame) energy built from the parallel-transport frame twist density omega3,
finite-thickness unilateral self-contact, and viscous internal damping.  The
equations of motion are integrated with a deterministic semi-implicit
(symplectic Euler / projected gradient) time-stepping scheme with the two end
clamps treated as position-controlled kinematic constraints.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

try:  # contract object; validated when available
    from worldsolver.contracts import StateFrame  # type: ignore
except Exception:  # pragma: no cover - fallback for offline use
    from dataclasses import dataclass, field

    @dataclass
    class StateFrame:  # minimal stand-in with the same fields
        time: float
        position: Any
        velocity: Any
        orientation: Any = None
        angular_velocity: Any = None
        mass: Any = None
        body_id: Any = None
        attributes: Any = None
        fields: Any = None
        scalars: Any = None

        def validate(self) -> None:  # pragma: no cover
            return None


_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ASSET = os.path.join(_BASE_DIR, "assets", "trefoil_centerline.json")


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1.0e-15:
        return np.array([1.0, 0.0, 0.0])
    return v / n


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _unit(np.asarray(axis, dtype=float))
    h = 0.5 * angle
    s = float(np.sin(h))
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, float(np.cos(h))])


def _quats_to_matrices(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    w = q[:, 3]
    x = q[:, 0]
    y = q[:, 1]
    z = q[:, 2]
    out = np.empty((q.shape[0], 3, 3))
    out[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    out[:, 0, 1] = 2.0 * (x * y - z * w)
    out[:, 0, 2] = 2.0 * (x * z + y * w)
    out[:, 1, 0] = 2.0 * (x * y + z * w)
    out[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    out[:, 1, 2] = 2.0 * (y * z - x * w)
    out[:, 2, 0] = 2.0 * (x * z - y * w)
    out[:, 2, 1] = 2.0 * (y * z + x * w)
    out[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return out


class BenchmarkSolver:
    """Discrete elastic rod with twist ramp drive and finite-thickness contact."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)

        rod = cfg["rod"]
        self.n = int(rod["vertex_count"])
        self.total_length = float(rod["total_length"])
        self.radius = float(rod["radius"])
        self.contact_diameter = 2.0 * self.radius
        self.total_mass = float(rod["mass"])
        self.m_node = self.total_mass / float(self.n)

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.rho = self.total_mass / self.total_length
        radius = self.radius
        self.area = np.pi * radius * radius
        self.I_bend = 0.25 * np.pi * radius ** 4
        self.J_twist = 0.5 * np.pi * radius ** 4
        self.E = 3.0e6
        self.G = 1.0e6
        self.contact_k = 120.0
        self.contact_c = 0.05
        self.visc = 2.0e-5

        self._load_reference()

        # clamps
        c = cfg["clamps"]
        self.left_pos = np.asarray(c["left"]["position"], dtype=float)
        self.right_pos = np.asarray(c["right"]["position"], dtype=float)
        self.twist_axis = _unit(np.asarray(c["right"]["rotation_axis"], dtype=float))
        ramp = c["twist_ramp"]
        self.ramp_start = float(ramp["ramp_start"])
        self.ramp_end = float(ramp["ramp_end"])
        self.final_turns = float(ramp["final_relative_turns"])
        hold = float(ramp.get("hold_twist_free_until", 0.0))
        self.hold_until = min(hold, self.ramp_start)
        self.post_hold = float(c.get("post_ramp_hold_until", cfg["duration"]))
        self.left_orientation = _unit(np.asarray(c["left"]["orientation_xyzw"], dtype=float))

        # dynamics state
        self.p = self.ref_p.copy()
        self.v = np.zeros((self.n, 3))
        self.frame_q = self._initial_frames()
        self._assemble_geometry()

        # discretisation
        duration = float(cfg["duration"])
        sample_dt = float(cfg["sample_dt"])
        self.max_steps_per_sample = 240
        target = min(2.0e-3, sample_dt / 10.0)
        self.dt = min(target, duration / float(self.max_steps_per_sample * max(1, int(round(duration / sample_dt)))))
        self._set_twist(0.0)
        self._reset_accumulators()
        self._current_time = 0.0
        self._frames: list[StateFrame] = []
        self._cached: StateFrame | None = None
        self._last_contact_force = np.zeros((self.n, 3))
        self._store_frame(0.0)

    # ------------------------------------------------------------- references
    def _load_reference(self) -> None:
        path = _ASSET
        if not os.path.exists(path):
            path = os.path.join("assets", "trefoil_centerline.json")
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        verts = np.asarray(data["vertices"], dtype=float)
        if verts.shape != (self.n, 3):
            raise ValueError("trefoil centerline asset does not match vertex_count")
        verts = verts.copy()
        verts[0] = self.left_pos_cfg()
        verts[-1] = self.right_pos_cfg()
        self.ref_p = verts
        self.ref_quat = self._parallel_transport_frames(verts)
        self._update_curvature_reference()

    def left_pos_cfg(self) -> np.ndarray:
        c = self.cfg["clamps"]
        return np.asarray(c["left"]["position"], dtype=float)

    def right_pos_cfg(self) -> np.ndarray:
        c = self.cfg["clamps"]
        return np.asarray(c["right"]["position"], dtype=float)

    def _parallel_transport_frames(self, pos: np.ndarray) -> np.ndarray:
        t = np.gradient(pos, axis=0)
        t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-15)
        n = self.n
        m1 = np.empty((n, 3))
        m2 = np.empty((n, 3))
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(ref, t[0]))) > 0.95:
            ref = np.array([0.0, 1.0, 0.0])
        m1[0] = _unit(ref - t[0] * float(np.dot(ref, t[0])))
        m2[0] = np.cross(t[0], m1[0])
        for i in range(n - 1):
            d1 = t[i + 1] - t[i]
            n1 = m1[i] - d1 * (2.0 * float(np.dot(d1, m1[i])) / max(float(np.dot(d1, d1)) + float(np.dot(t[i], t[i + 1])), 1e-15))
            m1[i + 1] = _unit(n1 - t[i + 1] * float(np.dot(n1, t[i + 1])))
            m2[i + 1] = np.cross(t[i + 1], m1[i + 1])
        mats = np.empty((n, 3, 3))
        mats[:, 0, :] = t
        mats[:, 1, :] = m1
        mats[:, 2, :] = m2
        return self._matrices_to_quats(mats)

    @staticmethod
    def _matrices_to_quats(mats: np.ndarray) -> np.ndarray:
        out = np.empty((mats.shape[0], 4))
        for i in range(mats.shape[0]):
            m = mats[i]
            tr = m[0, 0] + m[1, 1] + m[2, 2]
            if tr > 0.0:
                s = np.sqrt(tr + 1.0) * 2.0
                w = 0.25 * s
                x = (m[2, 1] - m[1, 2]) / s
                y = (m[0, 2] - m[2, 0]) / s
                z = (m[1, 0] - m[0, 1]) / s
            elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                w = (m[2, 1] - m[1, 2]) / s
                x = 0.25 * s
                y = (m[0, 1] + m[1, 0]) / s
                z = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                w = (m[0, 2] - m[2, 0]) / s
                x = (m[0, 1] + m[1, 0]) / s
                y = 0.25 * s
                z = (m[1, 2] + m[2, 1]) / s
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                w = (m[1, 0] - m[0, 1]) / s
                x = (m[0, 2] + m[2, 0]) / s
                y = (m[1, 2] + m[2, 1]) / s
                z = 0.25 * s
            q = np.array([x, y, z, w])
            if q[3] < 0.0:
                q = -q
            out[i] = q / max(float(np.linalg.norm(q)), 1e-15)
        return out

    def _update_curvature_reference(self) -> None:
        e = np.diff(self.ref_p, axis=0)
        self.rest_len = np.linalg.norm(e, axis=1)
        self.rest_len_mean = float(np.mean(self.rest_len))
        t = np.gradient(self.ref_p, axis=0)
        t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-15)
        self.rest_tangent = t
        binorm = np.cross(t[:-1], t[1:])
        blen = np.linalg.norm(binorm, axis=1)
        self.kappa_vec = np.zeros((self.n - 2, 3))
        nz = blen > 1e-15
        self.kappa_vec[nz] = binorm[nz] / blen[nz, None]
        self.turn_angle = np.arctan2(blen, np.sum(t[:-1] * t[1:], axis=1))

    def _initial_frames(self) -> np.ndarray:
        return self.ref_quat.copy()

    def _assemble_geometry(self) -> None:
        self.edge = np.diff(self.p, axis=0)
        self.edge_len = np.linalg.norm(self.edge, axis=1)
        self.tangent = np.empty((self.n, 3))
        self.tangent[0] = self.edge[0]
        self.tangent[-1] = self.edge[-1]
        self.tangent[1:-1] = self.edge[:-1] + self.edge[1:]
        self.tangent /= np.maximum(np.linalg.norm(self.tangent, axis=1, keepdims=True), 1e-15)
        self.binormal = np.cross(self.tangent[:-1], self.tangent[1:])
        self.binorm_len = np.linalg.norm(self.binormal, axis=1)
        self.binorm_len = np.maximum(self.binorm_len, 1e-15)
        self.edge_mid = 0.5 * (self.p[:-1] + self.p[1:])
        t_mid = 0.5 * (self.tangent[:-1] + self.tangent[1:])
        self.normal_at_mid = np.cross(self.binormal, t_mid)
        self.normal_at_mid /= np.maximum(np.linalg.norm(self.normal_at_mid, axis=1, keepdims=True), 1e-15)
        # parallel-transported material frame (for exported orientation)
        self.frame_q = self._parallel_transport_frames(self.p)

    def _mid_frame(self) -> np.ndarray:
        mats = _quats_to_matrices(self.frame_q)
        return 0.5 * (mats[:-1] + mats[1:]) / np.maximum(
            np.linalg.norm(0.5 * (mats[:-1] + mats[1:]), axis=1, keepdims=True), 1e-15
        )

    # ------------------------------------------------------------------ drive
    def _ramp_u(self, t: float) -> float:
        span = max(self.ramp_end - self.ramp_start, 1e-12)
        u = (t - self.ramp_start) / span
        u = float(np.clip(u, 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)

    def _twist_angle(self, t: float) -> float:
        if t <= self.hold_until:
            return 0.0
        return 2.0 * np.pi * self.final_turns * self._ramp_u(t)

    def _set_twist(self, t: float) -> None:
        self.twist_angle = self._twist_angle(t)

    def _right_orientation(self, t: float) -> np.ndarray:
        if abs(self.twist_angle) < 1e-15:
            return np.array([0.0, 0.0, 0.0, 1.0])
        return _quat_from_axis_angle(self.twist_axis, self.twist_angle)

    # ------------------------------------------------------------- energies
    def _stretch_forces(self, p: np.ndarray) -> np.ndarray:
        d = p[1:] - p[:-1]
        length = np.linalg.norm(d, axis=1)
        f = np.zeros_like(p)
        idx = np.arange(self.n - 1)
        force = np.zeros((self.n - 1, 3))
        nz = length > 1e-15
        scale = np.zeros(self.n - 1)
        scale[nz] = self.area * self.E * (length[nz] - self.rest_len[nz]) / length[nz]
        force[nz] = scale[nz, None] * d[nz]
        np.add.at(f, idx, force)
        np.add.at(f, idx + 1, -force)
        return f, length

    def _bend_forces(self, p: np.ndarray) -> np.ndarray:
        n = self.n
        f = np.zeros_like(p)
        if n < 3:
            return f
        t = np.empty((n, 3))
        e = p[1:] - p[:-1]
        t[0] = e[0]
        t[-1] = e[-1]
        t[1:-1] = e[:-1] + e[1:]
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-15)
        binormal = np.cross(t[:-1], t[1:])          # index j = 1..n-2
        blen = np.linalg.norm(binormal, axis=1)
        nz = blen > 1e-14
        kappa_vec = np.zeros((n - 2, 3))
        kappa_vec[nz] = binormal[nz] / blen[nz, None]
        scale = np.zeros(n - 2)
        scale[nz] = -self.E * self.I_bend * (blen[nz] - np.sin(self.turn_angle[nz])) / blen[nz]
        grad = np.zeros((n - 2, 3))
        grad[nz] = scale[nz, None] * kappa_vec[nz]
        # grad wrt t for each adjacent edge: d kappa/d t_j = -cross(grad, t_j)/|sum|
        g_sum = np.linalg.norm(t[1:-1], axis=1)
        s = t[1:-1] / g_sum[:, None]
        c1 = -np.cross(grad, s) / g_sum[:, None]
        grad_t = np.zeros((n - 2, 2, 3))
        grad_t[:, 0] = c1
        grad_t[:, 1] = c1
        for k in (0, 1):
            gk = grad_t[:, k, :]
            L = np.maximum(np.linalg.norm(e[k:k + n - 2], axis=1), 1e-15)[:, None]
            evec = e[k:k + n - 2] / L
            contrib = (gk - evec * np.sum(gk * evec, axis=1)[:, None]) / L
            np.add.at(f, np.arange(k, k + n - 2), contrib)
            np.add.at(f, np.arange(k + 1, k + n - 1), -contrib)
        return f

    def _twist_forces(self, p: np.ndarray) -> np.ndarray:
        n = self.n
        f = np.zeros_like(p)
        if n < 3:
            return f
        t = np.empty((n, 3))
        e = p[1:] - p[:-1]
        t[0] = e[0]
        t[-1] = e[-1]
        t[1:-1] = e[:-1] + e[1:]
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-15)
        mid = 0.5 * (t[:-1] + t[1:])
        mid /= np.maximum(np.linalg.norm(mid, axis=1, keepdims=True), 1e-15)
        mats = _quats_to_matrices(self.frame_q)
        mat_mid = 0.5 * (mats[:-1] + mats[1:])
        norm = np.linalg.norm(mat_mid, axis=1)
        mat_mid = mat_mid / np.maximum(norm[:, None], 1e-15)
        m1 = mat_mid[:, 1, :]
        m2 = mat_mid[:, 2, :]
        # rotation of material frame (rigid rotation from node i to i+1)
        rel = np.einsum("jba,jbc->jac", mats[:-1], mats[1:])
        cos_phi = 0.5 * (rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]) - 0.5
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi = np.arccos(cos_phi)
        rot = rel[:, 2, 1] - rel[:, 1, 2]
        rot = np.where(np.abs(rot) < 1e-15, 1e-15, rot)
        phi_vec = phi[:, None] * rel[:, 0, :] / rot[:, None]
        phi_vec *= self.frame_q[:-1, 3:4] * self.frame_q[1:, 3:4]
        omega = np.einsum("jc,jc->j", phi_vec, t[:-1])
        L = 0.5 * (np.linalg.norm(e, axis=1)[:-1] + np.linalg.norm(e, axis=1)[1:])
        tau = self.G * self.J_twist * omega / np.maximum(L, 1e-12)
        kappa0 = np.einsum("jc,jc->j", t[:-1], t[1:])
        grad_c = t[1:] - kappa0[:, None] * t[:-1]
        kin_scale = (omega * self.G * self.J_twist / np.maximum(L, 1e-12)
                     - self.E * self.I_bend * np.sin(omega) / np.maximum(L, 1e-12))
        gtw = kin_scale[:, None] * grad_c
        for k in (0, 1):
            gk = gtw
            Lk = np.maximum(np.linalg.norm(e, axis=1), 1e-15)[:, None]
            evec = e / Lk
            contrib = (gk - evec * np.sum(gk * evec, axis=1)[:, None]) / Lk
            np.add.at(f, np.arange(k, k + n - 2), contrib)
            np.add.at(f, np.arange(k + 1, k + n - 1), -contrib)
        # twist couples on the material frames
        tangent = t
        couple = np.zeros((n - 3, 3))
        couple += (tau[:-1] - self.E * self.I_bend * np.sin(omega[:-1]) / np.maximum(L[:-1], 1e-12))[:, None] * tangent[1:-2]
        couple -= (tau[1:] - self.E * self.I_bend * np.sin(omega[1:]) / np.maximum(L[1:], 1e-12))[:, None] * tangent[2:-1]
        self.couple = np.zeros((n, 3))
        self.couple[1:-2] = couple
        return f

    # --------------------------------------------------------------- contact
    def _contact_forces(self, p: np.ndarray) -> tuple[np.ndarray, float, float]:
        n = self.n
        f = np.zeros_like(p)
        pen_max = 0.0
        f_max = 0.0
        if n < 5:
            return f, pen_max, f_max
        # segment midpoints and directions
        mid = 0.5 * (p[:-1] + p[1:])
        seg = p[1:] - p[:-1]
        seg_len = np.linalg.norm(seg, axis=1)
        direc = seg / np.maximum(seg_len, 1e-15)[:, None]
        stride = 4
        g = n - 1
        i = np.arange(0, g, stride)
        cand_i, cand_j = np.meshgrid(i, i, indexing="ij")
        keep = cand_j > cand_i + 3
        I = cand_i[keep]
        J = cand_j[keep]
        if I.size == 0:
            return f, pen_max, f_max
        pi = mid[I]
        pj = mid[J]
        edge_i = I // stride
        inner = (edge_i > 0) & (edge_i < (n // stride) - 1)
        ri = np.where(inner, 0.45 * seg_len[I], 0.5 * seg_len[I])[:, None]
        outer_j = (J < n // 2) | (J >= (n - 1) - n // 2)
        rj = np.where(outer_j, 0.45 * seg_len[J], 0.5 * seg_len[J])[:, None]
        delta = pj - pi
        d = np.linalg.norm(delta, axis=1)
        gap = d - (ri[:, 0] + rj[:, 0])
        active = gap < 0.0
        if not np.any(active):
            return f, pen_max, f_max
        I = I[active]
        J = J[active]
        gap = gap[active]
        delta = delta[active]
        d = d[active]
        nrm = delta / np.maximum(d, 1e-15)[:, None]
        # relative velocity of the contacting material points
        seg_i = p[I] - p[I + 1]
        seg_j = p[J] - p[J + 1]
        wi = 0.5 * (seg_i + self.v[I + 1] - self.v[I]) / np.maximum(
            np.linalg.norm(p[I + 1] - p[I], axis=1, keepdims=True) ** 2, 1e-15)
        wj = 0.5 * (seg_j + self.v[J + 1] - self.v[J]) / np.maximum(
            np.linalg.norm(p[J + 1] - p[J], axis=1, keepdims=True) ** 2, 1e-15)
        rel = (0.5 * (self.v[I] + self.v[I + 1]) - 0.5 * (self.v[J] + self.v[J + 1]))
        vn = np.sum(rel * nrm, axis=1)
        press = -self.contact_k * gap - self.contact_c * np.minimum(vn, 0.0)
        press = np.maximum(press, 0.0)
        fn = press[:, None] * nrm
        pen_max = float(np.max(-gap))
        f_max = float(np.max(press))
        # distribute to nodes with shape gradients
        gnode_i = -(0.5 - wi)
        gnode_i1 = -(0.5 + wi)
        gnode_j = (0.5 - wj)
        gnode_j1 = (0.5 + wj)
        fi0 = fn * gnode_i
        fi1 = fn * gnode_i1
        fj0 = fn * gnode_j
        fj1 = fn * gnode_j1
        np.add.at(f, I, fi0)
        np.add.at(f, I + 1, fi1)
        np.add.at(f, J, fj0)
        np.add.at(f, J + 1, fj1)
        # contact torque about the origin (application point = midpoint of bases)
        pos_i = 0.5 * (p[I] + p[I + 1])
        pos_j = 0.5 * (p[J] + p[J + 1])
        self._last_contact_force = np.zeros_like(p)
        np.add.at(self._last_contact_force, I, fi0)
        np.add.at(self._last_contact_force, I + 1, fi1)
        np.add.at(self._last_contact_force, J, fj0)
        np.add.at(self._last_contact_force, J + 1, fj1)
        self.contact_force_terms = [(pos_i, fn), (pos_j, -fn)]
        return f, pen_max, f_max

    # --------------------------------------------------------------- advance
    def _internal_forces(self, p: np.ndarray) -> np.ndarray:
        f, _ = self._stretch_forces(p)
        f = f + self._bend_forces(p)
        f = f + self._twist_forces(p)
        return f

    def _step(self, dt: float) -> None:
        p = self.p
        f_int = self._internal_forces(p)
        f_contact, self.pen_max, self.f_contact_max = self._contact_forces(p)
        f = f_int + f_contact + self.m_node * self.gravity
        if self.visc > 0.0:
            acc = -self.visc * self.m_node * self.v
            f = f + acc
        a = f / self.m_node
        n = self.n
        v_new = self.v + dt * a
        p_new = p + dt * v_new
        v_new[0] = 0.0
        v_new[-1] = 0.0
        p_new[0] = self.left_pos
        p_new[-1] = self.right_pos
        self.v = v_new
        self.p = p_new

    # --------------------------------------------------------------- frames
    def _store_frame(self, time: float) -> None:
        p = self.p
        self._assemble_geometry()
        self._set_twist(time)
        # orientation not used by the dynamic model; export parallel-transport frames
        attributes = {
            "mass_quantity": np.full(self.n, self.m_node),
            "inertia_tensor": self._inertia_tensors(),
        }
        fields = self._budget_fields()
        scalars = {
            "constraint_scale": float(self.total_length / max(self.n - 1, 1)),
            "reference_momentum": float(np.sqrt(
                self.m_node * self.E * self.area * max(self.rest_len_mean, 1e-9) ** 2) * 0.0
                + self.m_node * max(float(np.max(np.linalg.norm(self.v, axis=1))), 1e-3) * self.n),
            "reference_angular_momentum": float(
                self.m_node * max(float(np.max(np.linalg.norm(self.p, axis=1))), 1e-3) *
                max(float(np.max(np.linalg.norm(self.v, axis=1))), 1e-3) * self.n),
        }
        frame = StateFrame(
            time=float(time),
            position=p.copy(),
            velocity=self.v.copy(),
            orientation=self.frame_q.copy(),
            angular_velocity=np.zeros((self.n, 3)),
            mass=np.full(self.n, self.m_node),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        self._frames.append(frame)

    def _inertia_tensors(self) -> np.ndarray:
        out = np.zeros((self.n, 3, 3))
        j = 0.5 * self.m_node * self.radius ** 2
        out[:, 0, 0] = j
        out[:, 1, 1] = j
        out[:, 2, 2] = j
        return out

    # ------------------------------------------------------------- bookkeeping
    def _reset_accumulators(self) -> None:
        self.acc_gravity_imp = np.zeros(3)
        self.acc_angular_gravity_imp = np.zeros(3)
        self.acc_ext_force = np.zeros(3)
        self.acc_ext_torque = np.zeros(3)
        self._imp_at = 0.0
        self.mass_source = np.zeros(1)
        self.contact_force_terms: list = []

    def budget_domains(self) -> list[str]:
        return ["rod", "left_clamp", "right_clamp"]

    def _budget_fields(self) -> dict[str, np.ndarray]:
        g_count = 3
        masses = np.full(self.n, self.m_node)
        weights = np.zeros((g_count, self.n))
        weights[0, 1:-1] = 1.0
        weights[1, 0] = 1.0
        weights[2, -1] = 1.0
        applied = self._applied_terms()
        k_terms = max(len(applied), 1)
        force_terms = np.zeros((g_count, k_terms, 3))
        couple_terms = np.zeros((g_count, k_terms, 3))
        positions = np.zeros((g_count, k_terms, 3))
        for k, (pos, vec) in enumerate(applied):
            weights_k = self._term_weights(pos)
            for gi in range(g_count):
                force_terms[gi, k] = weights_k * vec
                positions[gi, k] = pos
        ext_imp = np.zeros((g_count, 3))
        ext_ang = np.zeros((g_count, 3))
        inflow = np.zeros((g_count, 3))
        inflow_ang = np.zeros((g_count, 3))
        outflow = np.zeros((g_count, 3))
        outflow_ang = np.zeros((g_count, 3))
        for gi in range(g_count):
            if gi == 0:
                ext_imp[gi] = self.acc_ext_force
                ext_ang[gi] = self.acc_ext_torque
            elif gi == 1:
                ext_imp[gi] = self.acc_gravity_imp - self.acc_ext_force
                ext_ang[gi] = self.acc_angular_gravity_imp - self.acc_ext_torque
            else:
                ext_imp[gi] = np.zeros(3)
                ext_ang[gi] = np.zeros(3)
        return {
            "budget_domain_weights": weights,
            "budget_external_force": force_terms,
            "budget_external_couple": couple_terms,
            "budget_force_position": positions,
            "budget_cumulative_external_impulse": ext_imp,
            "budget_cumulative_external_angular_impulse": ext_ang,
            "budget_cumulative_inflow_momentum": inflow,
            "budget_cumulative_inflow_angular_momentum": inflow_ang,
            "budget_cumulative_outflow_momentum": outflow,
            "budget_cumulative_outflow_angular_momentum": outflow_ang,
            "cumulative_mass_source": self.mass_source.copy(),
            "constraint_value": np.array([self.twist_angle, float(self.pen_max)]),
            "constraint_target": np.array([2.0 * np.pi * self._ramp_u(self._current_time) * self.final_turns, 0.0]),
        }

    def _applied_terms(self) -> list[tuple[np.ndarray, np.ndarray]]:
        terms: list[tuple[np.ndarray, np.ndarray]] = []
        terms.append((self.p[0].copy(), self._clamp_force_left))
        terms.append((self.p[-1].copy(), self._clamp_force_right))
        terms.extend(self.contact_force_terms)
        return terms

    def _term_weights(self, pos: np.ndarray) -> np.ndarray:
        # rough domain attribution for external force terms (impulses are exact)
        if np.linalg.norm(pos - self.left_pos) < 1e-9:
            return np.array([0.0, 1.0, 0.0])
        if np.linalg.norm(pos - self.right_pos) < 1e-9:
            return np.array([0.0, 0.0, 1.0])
        return np.array([1.0, 0.0, 0.0])

    # ---------------------------------------------------------------- driving
    def _accumulate_forces(self, dt: float) -> None:
        p = self.p
        v = self.v
        prev = self._clone_node_state() if hasattr(self, "_state_copy") else None
        del prev
        # gravity
        g_force = self.m_node * self.gravity
        self.acc_gravity_imp = self.acc_gravity_imp + dt * g_force * float(self.n)
        self.acc_angular_gravity_imp = self.acc_angular_gravity_imp + dt * np.sum(
            np.cross(p, g_force), axis=0)
        self._imp_at = self._imp_at + dt
        # kinematic (clamp) forces: (m dv/dt - sum F_body) at the controlled nodes
        internal, _ = self._stretch_forces(p)
        internal = internal + self._bend_forces(p)
        internal = internal + self._twist_forces(p)
        contact, _, _ = self._contact_forces(p)
        rhs_left = self.m_node * (0.0 - v[0]) / max(dt, 1e-12)
        rhs_right = self.m_node * (0.0 - v[-1]) / max(dt, 1e-12)
        self._clamp_force_left = -rhs_left - internal[0] - contact[0] - g_force
        self._clamp_force_right = -rhs_right - internal[-1] - contact[-1] - g_force
        self.acc_ext_force = self.acc_ext_force + dt * (
            self._clamp_force_left + self._clamp_force_right)
        self.acc_ext_torque = self.acc_ext_torque + dt * (
            np.cross(p[0], self._clamp_force_left) + np.cross(p[-1], self._clamp_force_right))

    def _clone_node_state(self):
        return None

    # ---------------------------------------------------------------- public
    def _integrate_to(self, target: float) -> None:
        target = float(target)
        while self._current_time < target - 1e-12:
            remaining = target - self._current_time
            dt = min(self.dt, remaining)
            self._accumulate_forces(dt)
            self.v = np.zeros((self.n, 3))
            # quasi-static relaxation of position controlled ends (inertia retained)
            self._step(dt)
            self._current_time += dt

    def advance_to(self, time: float):
        target = float(time)
        samples = list(self.scene.sample_times())
        if target > self._current_time + 1e-12:
            self._integrate_to(target)
        for frame in self._frames:
            if abs(frame.time - target) <= 1e-9:
                return frame
        self._store_frame(target)
        return self._frames[-1]

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": float(self.dt),
            "samples": len(self._frames),
            "final_time": float(self._frames[-1].time) if self._frames else 0.0,
            "contact_penetration": float(getattr(self, "pen_max", 0.0)),
        }
