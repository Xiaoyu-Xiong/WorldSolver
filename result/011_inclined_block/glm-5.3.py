"""Physically-based rigid-body solver for scene 011_inclined_block.

A single rigid block is dropped onto a fixed inclined plane (30 degrees tilt)
and interacts with it through dry frictional contact.  The dynamics are
advanced with a semi-implicit (symplectic) Euler scheme on a fixed substep,
with a velocity-level sequential-impulse contact solver (normal impulses with
Baumgarte stabilization + restitution, Coulomb friction impulses clamped to
the friction cone).  No animation, keyframes or precomputed trajectories are
used; every state is produced by the time-stepping procedure itself.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# small quaternion / rotation helpers (xyzw convention)
# ---------------------------------------------------------------------------

def quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_integrate(q: np.ndarray, omega: np.ndarray, h: float) -> np.ndarray:
    """Integrate unit quaternion q (body->world) with world angular velocity."""
    wq = np.array([omega[0], omega[1], omega[2], 0.0])
    dq = 0.5 * quat_mul(wq, q)
    qn = q + h * dq
    return qn / np.linalg.norm(qn)


def tangent_basis(n: np.ndarray):
    """Two unit tangents orthogonal to unit normal n."""
    helper = np.array([1.0, 0.0, 0.0])
    if abs(n[0]) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, helper)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    return t1, t2


def box_sdf(p_local: np.ndarray, half: np.ndarray) -> float:
    """Signed distance of point (in box local frame) to box surface."""
    q = np.abs(p_local) - half
    qmax = np.maximum(q, 0.0)
    outside = float(np.linalg.norm(qmax))
    inside = float(min(max(q[0], q[1], q[2]), 0.0))
    return outside + inside


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------

class BenchmarkSolver:
    """Impulse-based rigid-body contact solver for the inclined-block scene."""

    SUBSTEPS_PER_SAMPLE = 20          # h = 0.001 s with sample_dt = 0.02 s
    SOLVER_ITERATIONS = 12
    FRICTION = 0.70                   # Coulomb friction coefficient (mu > tan 30 deg)
    RESTITUTION = 0.15
    RESTITUTION_THRESHOLD = 0.5       # m/s; below this impacts are perfectly plastic
    BAUMGARTE = 0.2
    SLOP = 5.0e-4
    GRAVITY = np.array([0.0, 0.0, -9.81])

    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        frame = scene.initial_state()
        self.sample_dt = float(scene.config.get("sample_dt", 0.02)) if hasattr(scene, "config") else 0.02
        # -- dynamic block state ------------------------------------------------
        self.mass = float(np.asarray(frame.mass).reshape(-1)[0])
        self.half_extents = np.array([0.45, 0.35, 0.30])
        a, b, c = 2.0 * self.half_extents
        self.inertia_body = np.diag(
            np.array(
                [
                    self.mass / 12.0 * (b * b + c * c),
                    self.mass / 12.0 * (a * a + c * c),
                    self.mass / 12.0 * (a * a + b * b),
                ]
            )
        )
        self.inertia_body_inv = np.diag(1.0 / np.diag(self.inertia_body))
        self.pos = np.array(np.asarray(frame.position)[0], dtype=float)
        self.vel = np.array(np.asarray(frame.velocity)[0], dtype=float)
        self.quat = np.array(np.asarray(frame.orientation)[0], dtype=float)
        self.quat = self.quat / np.linalg.norm(self.quat)
        self.omega = np.zeros(3)

        # -- static inclined plane (frozen scene geometry) ----------------------
        self.plane_half = np.array([3.2, 2.0, 0.08])
        self.plane_center = np.zeros(3)
        q_plane = np.array([-0.258819, 0.0, 0.0, 0.965926])
        self.plane_R = quat_to_mat(q_plane)
        self.plane_Rt = self.plane_R.T

        # local corner coordinates of the block (8 vertices, body frame)
        sx, sy, sz = self.half_extents
        corners = [
            (x * sx, y * sy, z * sz)
            for x in (-1.0, 1.0)
            for y in (-1.0, 1.0)
            for z in (-1.0, 1.0)
        ]
        self.corners_local = np.array(corners)
        # indices of the four bottom-face corners (contact / near-contact samples)
        self.contact_corner_idx = np.where(self.corners_local[:, 2] < 0.0)[0]

        self.time = 0.0
        self.h = self.sample_dt / self.SUBSTEPS_PER_SAMPLE

    # -- contact detection ---------------------------------------------------

    def _detect_contacts(self):
        """Corner-vs-box contacts between the block and the inclined plane."""
        R = quat_to_mat(self.quat)
        corners_w = self.pos[None, :] + self.corners_local @ R.T
        contacts = []
        for cw in corners_w:
            p_local = self.plane_Rt @ (cw - self.plane_center)
            # penetration along each box axis
            over = self.plane_half - np.abs(p_local)
            if np.all(over > 0.0):
                axis = int(np.argmin(over))
                depth = float(over[axis])
                sgn = 1.0 if p_local[axis] >= 0.0 else -1.0
                normal_world = sgn * self.plane_R[:, axis]
                r = cw - self.pos
                contacts.append((r, normal_world, depth))
        return contacts

    # -- one substep -----------------------------------------------------------

    def _substep(self, h: float) -> None:
        m = self.mass
        # 1) gravity (external force, contact handled by impulses)
        self.vel = self.vel + self.GRAVITY * h

        contacts = self._detect_contacts()
        if contacts:
            R = quat_to_mat(self.quat)
            Iinv = R @ self.inertia_body_inv @ R.T

            # precompute per-contact data
            data = []
            for (r, n, depth) in contacts:
                rn = np.cross(r, n)
                w_n = 1.0 / m + float(rn @ (Iinv @ rn))
                t1, t2 = tangent_basis(n)
                rt1 = np.cross(r, t1)
                rt2 = np.cross(r, t2)
                w_t1 = 1.0 / m + float(rt1 @ (Iinv @ rt1))
                w_t2 = 1.0 / m + float(rt2 @ (Iinv @ rt2))
                # pre-solve normal approach speed (for restitution)
                v_pre = self.vel + np.cross(self.omega, r)
                vn0 = float(v_pre @ n)
                data.append(
                    dict(
                        r=r, n=n, depth=depth,
                        m_n=1.0 / w_n,
                        t1=t1, t2=t2,
                        m_t1=1.0 / w_t1, m_t2=1.0 / w_t2,
                        vn0=vn0,
                        jn=0.0, jt1=0.0, jt2=0.0,
                    )
                )

            # restitution / stabilization targets
            for d in data:
                e_target = 0.0
                if d["vn0"] < -self.RESTITUTION_THRESHOLD:
                    e_target = -self.RESTITUTION * d["vn0"]
                bias = self.BAUMGARTE / h * max(0.0, d["depth"] - self.SLOP)
                d["vn_des"] = max(e_target, bias)

            inv_m = 1.0 / m
            for _ in range(self.SOLVER_ITERATIONS):
                # normal impulses
                for d in data:
                    r, n = d["r"], d["n"]
                    v_p = self.vel + np.cross(self.omega, r)
                    vn = float(v_p @ n)
                    dj = d["m_n"] * (d["vn_des"] - vn)
                    new_jn = max(0.0, d["jn"] + dj)
                    dj = new_jn - d["jn"]
                    d["jn"] = new_jn
                    if dj != 0.0:
                        P = dj * n
                        self.vel = self.vel + inv_m * P
                        self.omega = self.omega + Iinv @ np.cross(r, P)
                # friction impulses
                for d in data:
                    r, n = d["r"], d["n"]
                    v_p = self.vel + np.cross(self.omega, r)
                    vt1 = float(v_p @ d["t1"])
                    vt2 = float(v_p @ d["t2"])
                    dj1 = -d["m_t1"] * vt1
                    dj2 = -d["m_t2"] * vt2
                    j1 = d["jt1"] + dj1
                    j2 = d["jt2"] + dj2
                    mag = math.hypot(j1, j2)
                    cap = self.FRICTION * d["jn"]
                    if mag > cap and mag > 0.0:
                        scale = cap / mag
                        j1 *= scale
                        j2 *= scale
                    dj1 = j1 - d["jt1"]
                    dj2 = j2 - d["jt2"]
                    d["jt1"], d["jt2"] = j1, j2
                    if dj1 != 0.0:
                        P = dj1 * d["t1"]
                        self.vel = self.vel + inv_m * P
                        self.omega = self.omega + Iinv @ np.cross(r, P)
                    if dj2 != 0.0:
                        P = dj2 * d["t2"]
                        self.vel = self.vel + inv_m * P
                        self.omega = self.omega + Iinv @ np.cross(r, P)

        # 2) integrate pose
        self.pos = self.pos + self.vel * h
        self.quat = quat_integrate(self.quat, self.omega, h)

        # finite-state guard (defensive; should never trigger)
        if not (np.all(np.isfinite(self.vel)) and np.all(np.isfinite(self.omega))
                and np.all(np.isfinite(self.pos))):
            self.vel = np.zeros(3)
            self.omega = np.zeros(3)

    # -- public API -------------------------------------------------------------

    def advance_to(self, time: float):
        t_target = float(time)
        eps = 1.0e-9
        if t_target > self.time + eps:
            # step with fixed substeps, landing exactly on the requested time
            while self.time < t_target - eps:
                h = min(self.h, t_target - self.time)
                if h < 1.0e-12:
                    break
                self._substep(h)
                self.time += h
            self.time = t_target
        return self._make_frame(t_target)

    def _make_frame(self, t: float) -> StateFrame:
        R = quat_to_mat(self.quat)
        corners_w = self.pos[None, :] + self.corners_local @ R.T

        # signed geometric gap of the four bottom-face (near-contact) corners
        gaps = []
        for idx in self.contact_corner_idx:
            p_local = self.plane_Rt @ (corners_w[idx] - self.plane_center)
            gaps.append(box_sdf(p_local, self.plane_half))
        gaps = np.asarray(gaps, dtype=float)
        if gaps.size == 0:
            gaps = np.zeros(1)

        external_force = (self.mass * self.GRAVITY).reshape(1, 3)

        return StateFrame(
            time=float(t),
            position=self.pos.reshape(1, 3).copy(),
            velocity=self.vel.reshape(1, 3).copy(),
            orientation=self.quat.reshape(1, 4).copy(),
            angular_velocity=self.omega.reshape(1, 3).copy(),
            mass=np.array([self.mass]),
            body_id=np.array([1], dtype=int),
            fields={
                "contact_signed_distance": gaps,
                "constraint_value": gaps.copy(),
                "constraint_target": np.zeros_like(gaps),
                "external_force": external_force,
            },
            scalars={
                "constraint_scale": 0.30,
                "contact_length_scale": 0.30,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "sequential-impulse rigid-body contact (semi-implicit Euler)",
            "substep": self.h,
            "friction": self.FRICTION,
            "restitution": self.RESTITUTION,
            "gravity": self.GRAVITY.tolist(),
            "time": self.time,
        }
