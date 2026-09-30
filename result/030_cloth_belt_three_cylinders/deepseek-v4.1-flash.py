"""Physically-based solver for the closed cloth belt on three driven cylinders.

Model
-----
The scaffold freezes the belt as a 120 x 9 particle grid (a closed tube of
cloth) that initially wraps the three parallel cylinders.  The belt is
simulated with small-step eXtended Position Based Dynamics (XPBD, Macklin et
al. 2016): compliant distance constraints carry the membrane response
(longitudinal, transverse, shear and soft bending), and the cylinders act as
analytic kinematic colliders with one-sided normal contact plus Coulomb
friction (mu = 0.4).  Every frame is produced by actually time-stepping the
constrained particle system; nothing is keyframed or scripted.

Prescribed kinematics (frozen config)
-------------------------------------
* top cylinder (body 3) rises 0.92 -> 1.62 m over the lift window with a
  smooth velocity profile, then holds, tensioning the closed belt.
* the two outer cylinders (bodies 1, 2) spin through one forward revolution
  during the forward window, pause, then spin back through one revolution.
  ``outer_rotation_signs[0]`` defines the forward sense; the loop wraps the
  outer side of *both* outer cylinders, so a coherent frictional drive uses
  that single common world sense for the pair.

Exported state follows the frozen StateFrame contract: raw positions,
velocities, per-node mass, per-node deformation gradient, signed contact gaps
and the frozen contact length scale.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

RING_COUNT = 120
WIDTH_NODES = 9
N_CLOTH = RING_COUNT * WIDTH_NODES
SUBSTEPS = 60      # XPBD substeps per animation frame
ITERS = 2          # constraint iterations per substep
DAMPING = 4.0      # global velocity damping [1/s]
MAX_SPEED = 25.0   # safety clamp [m/s]

# frozen material parameters of the belt (N/m per constraint)
K_LONG = 3.0e3
K_WIDTH = 3.0e3
K_SHEAR = 1.0e3
K_BEND = 2.5e-1

BASE_QUAT = np.array([0.707106781, 0.0, 0.0, 0.707106781], dtype=float)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions ``a`` (first) and ``b``."""
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


def _scatter_add(dst: np.ndarray, index: np.ndarray, values: np.ndarray) -> None:
    """Accumulate ``values`` into rows ``index`` of ``dst`` (duplicates kept)."""
    size = dst.shape[0]
    for column in range(dst.shape[1]):
        dst[:, column] += np.bincount(
            index, weights=values[:, column], minlength=size
        )


class BenchmarkSolver:
    """XPBD cloth-belt solver driven by the frozen kinematic cylinders."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.config = scene.config
        config = self.config
        belt = config["belt"]
        cylinders = config["cylinders"]

        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.thickness = float(belt["thickness"])
        self.radius = float(cylinders["radius"])
        self.contact_radius = self.radius + 0.5 * self.thickness
        self.cyl_half_height = 0.5 * float(cylinders["length"])
        self.mu = float(config["friction"]["coefficient"])
        self.omega_speed = float(cylinders["outer_angular_speed"])
        self.revolutions = float(cylinders["outer_drive_revolutions"])
        signs = np.asarray(cylinders["outer_rotation_signs"], dtype=float)
        self.spin_sign = float(signs[0]) if signs[0] != 0.0 else 1.0
        self.timeline = config["timeline"]
        self.sample_dt = float(config.get("sample_dt", 1.0 / 60.0))
        centres_initial = np.asarray(cylinders["initial_centers_xz"], dtype=float)
        centres_final = np.asarray(cylinders["final_centers_xz"], dtype=float)
        self.cyl_x = centres_initial[:, 0].copy()
        self.cyl_z = centres_initial[:, 1].copy()
        self.z_start = float(centres_initial[2, 1])
        self.z_final = float(centres_final[2, 1])

        # ---- frozen initial frame (authoritative state bindings)
        frame0 = scene.initial_state()
        position = np.asarray(frame0.position, dtype=float)
        velocity = np.asarray(frame0.velocity, dtype=float)
        self.n = position.shape[0]
        self.n_cloth = self.n - 3
        if self.n_cloth != N_CLOTH:
            raise ValueError("unexpected cloth particle count")
        self.pos = position.copy()
        self.vel = velocity.copy()
        self.body_id = (
            np.asarray(frame0.body_id).astype(np.int64).copy()
            if frame0.body_id is not None
            else np.r_[np.zeros(self.n_cloth, dtype=np.int64), [1, 2, 3]]
        )
        if frame0.mass is not None:
            self.mass = np.asarray(frame0.mass, dtype=float).copy()
        else:
            self.mass = np.r_[
                np.full(self.n_cloth, float(belt["total_mass"]) / self.n_cloth),
                np.zeros(3),
            ]
        self.mass_quantity = self.mass.copy()
        self.inv_mass = np.where(
            self.mass > 0.0, 1.0 / np.maximum(self.mass, 1e-30), 0.0
        )
        uv = frame0.attributes.get("uv") if frame0.attributes else None
        self.uv = (
            np.asarray(uv, dtype=float).copy()
            if uv is not None
            else np.zeros((self.n, 2))
        )

        # ---- frozen reference (material) configuration
        reference, faces = self._load_material_mesh(scene)
        self.reference = reference
        self.faces = faces
        self.tri_areas = self._triangle_areas(reference, faces)
        self.ref_dm_inv = self._reference_basis_inverse(reference, faces)

        # ---- constraint network
        self.ca, self.cb, self.rest, self.compliance = self._build_constraints(
            reference
        )

        # ---- dynamic state containers
        self.time = 0.0
        self.orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n, 1))
        self.orientation[self.n_cloth :] = BASE_QUAT
        self.angular_velocity = np.zeros((self.n, 3))
        self.cyl_omega = np.zeros(3)   # world y angular velocity per cylinder
        self.cyl_vz = np.zeros(3)      # axial (z) translation velocity per cylinder
        self.friction_lambda = np.zeros(self.n_cloth)
        self.normal_lambda = np.zeros(self.n_cloth)
        self._inc = np.zeros((self.n, 3))
        self._update_kinematics(0.0)

    def _load_material_mesh(self, scene):
        cloth = getattr(scene, "_cloth", None)
        faces = getattr(scene, "_faces", None)
        if cloth is None or faces is None:
            base = Path(__file__).resolve().parent.parent
            data = json.loads(
                (base / "assets" / "closed_cloth_belt_topology.json").read_text(
                    encoding="utf-8"
                )
            )
            cloth = data["vertices"]
            faces = data["faces"]
        return (
            np.asarray(cloth, dtype=float)[: self.n_cloth],
            np.asarray(faces, dtype=np.int64),
        )

    @staticmethod
    def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        tri = vertices[faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        return 0.5 * np.linalg.norm(cross, axis=1)

    @staticmethod
    def _reference_basis_inverse(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Inverse of the per-triangle material basis Dm = [e1, e2, n*h]."""
        tri = vertices[faces]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        normal = np.cross(e1, e2)
        length = np.linalg.norm(normal, axis=1)
        scale = np.sqrt(np.maximum(length, 1e-16))
        unit = normal / np.maximum(length, 1e-16)[:, None]
        dm = np.stack([e1, e2, unit * scale[:, None]], axis=2)
        det = np.linalg.det(dm)
        degenerate = ~np.isfinite(det) | (np.abs(det) < 1e-18)
        if degenerate.any():
            dm[degenerate] = np.eye(3)
        return np.linalg.inv(dm)

    def _build_constraints(self, vertices):
        index = np.arange(RING_COUNT * WIDTH_NODES).reshape(RING_COUNT, WIDTH_NODES)
        ring_next = np.roll(index, -1, axis=0)
        ring_next2 = np.roll(index, -2, axis=0)

        pairs = [
            # longitudinal (around the closed loop)
            (index.ravel(), ring_next.ravel(), K_LONG),
            # transverse (across the belt width)
            (index[:, :-1].ravel(), index[:, 1:].ravel(), K_WIDTH),
            # shear diagonals
            (index[:, :-1].ravel(), ring_next[:, 1:].ravel(), K_SHEAR),
            (index[:, 1:].ravel(), ring_next[:, :-1].ravel(), K_SHEAR),
            # soft bending (material curvature / anti-folding)
            (index.ravel(), ring_next2.ravel(), K_BEND),
            (index[:, :-2].ravel(), index[:, 2:].ravel(), K_BEND),
        ]

        first = np.concatenate([p[0] for p in pairs])
        second = np.concatenate([p[1] for p in pairs])
        stiffness = np.concatenate(
            [np.full(len(p[0]), p[2], dtype=float) for p in pairs]
        )
        rest = np.linalg.norm(vertices[first] - vertices[second], axis=1)
        keep = rest > 1e-9
        first, second, stiffness, rest = (
            first[keep],
            second[keep],
            stiffness[keep],
            rest[keep],
        )
        return (
            first.astype(np.int64),
            second.astype(np.int64),
            rest,
            1.0 / stiffness,
        )

    # ------------------------------------------------------- prescribed motion
    def _top_height(self, t: float) -> float:
        start, end = self.timeline["lift"]
        if t <= start:
            return self.z_start
        if t >= end:
            return self.z_final
        s = (t - start) / (end - start)
        return self.z_start + (self.z_final - self.z_start) * (
            3.0 * s * s - 2.0 * s**3
        )

    def _top_velocity(self, t: float) -> float:
        start, end = self.timeline["lift"]
        if t <= start or t >= end:
            return 0.0
        s = (t - start) / (end - start)
        return (self.z_final - self.z_start) * 6.0 * s * (1.0 - s) / (end - start)

    def _drive_angle(self, t: float) -> float:
        """Signed outer-cylinder spin angle [rad] (forward, pause, reverse)."""
        forward_start, forward_end = self.timeline["forward"]
        reverse_start, reverse_end = self.timeline["reverse"]
        total = 2.0 * np.pi * self.revolutions
        if t <= forward_start:
            return 0.0
        if t < forward_end:
            return total * (t - forward_start) / (forward_end - forward_start)
        if t <= reverse_start:
            return total
        if t < reverse_end:
            return total * (1.0 - (t - reverse_start) / (reverse_end - reverse_start))
        return 0.0

    def _drive_omega(self, t: float) -> float:
        forward_start, forward_end = self.timeline["forward"]
        reverse_start, reverse_end = self.timeline["reverse"]
        total = 2.0 * np.pi * self.revolutions
        if forward_start < t < forward_end:
            return total / (forward_end - forward_start)
        if reverse_start < t < reverse_end:
            return -total / (reverse_end - reverse_start)
        return 0.0

    def _update_kinematics(self, t: float) -> None:
        nc = self.n_cloth
        z_top = self._top_height(t)
        vz_top = self._top_velocity(t)
        self.pos[nc + 2, 2] = z_top
        self.vel[nc + 2] = 0.0
        self.vel[nc + 2, 2] = vz_top
        self.cyl_vz[:] = 0.0
        self.cyl_vz[2] = vz_top

        omega = self.spin_sign * self._drive_omega(t)
        self.cyl_omega[:] = [omega, omega, 0.0]
        angle = self.spin_sign * self._drive_angle(t)
        spin = np.array(
            [0.0, np.sin(0.5 * angle), 0.0, np.cos(0.5 * angle)], dtype=float
        )
        for k in range(3):
            row = nc + k
            self.vel[row, :] = 0.0
            self.angular_velocity[row, :] = 0.0
            if k < 2:
                self.angular_velocity[row, 1] = omega
                self.orientation[row] = _quat_mul(spin, BASE_QUAT)
            else:
                self.vel[row, 2] = vz_top
        self.pos[nc : nc + 2, 0] = self.cyl_x[:2]
        self.pos[nc : nc + 3, 1] = 0.0
        self.pos[nc : nc + 2, 2] = self.cyl_z[:2]

    # --------------------------------------------------------------- stepping
    def _solve_distance(self, x: np.ndarray, lam: np.ndarray, h: float) -> None:
        first, second = self.ca, self.cb
        pa = x[first]
        pb = x[second]
        delta = pa - pb
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta)) + 1e-12
        normal = delta / dist[:, None]
        constraint = dist - self.rest
        wa = self.inv_mass[first]
        wb = self.inv_mass[second]
        wsum = wa + wb
        atilde = self.compliance / (h * h)
        dlambda = (-constraint - atilde * lam) / (wsum + atilde)
        self._inc[:] = 0.0
        _scatter_add(self._inc, first, (dlambda * wa)[:, None] * normal)
        _scatter_add(self._inc, second, -(dlambda * wb)[:, None] * normal)
        x += self._inc
        lam += dlambda

    def _solve_contacts(self, x: np.ndarray, x_prev: np.ndarray, h: float) -> None:
        nc = self.n_cloth
        centres_x = self.pos[nc : nc + 3, 0]
        centres_z = self.pos[nc : nc + 3, 2]
        dx = x[:nc, 0:1] - centres_x[None, :]
        dz = x[:nc, 2:3] - centres_z[None, :]
        radial = np.sqrt(dx * dx + dz * dz) + 1e-12
        gap = radial - self.contact_radius
        inside = np.abs(x[:nc, 1:2]) <= self.cyl_half_height
        gap = np.where(inside, gap, 1.0)
        choice = np.argmin(gap, axis=1)
        rows = np.arange(nc)
        active = gap[rows, choice] < 0.0
        if not active.any():
            return
        dxs = dx[rows, choice]
        dzs = dz[rows, choice]
        rs = radial[rows, choice]
        nx = dxs / rs
        nz = dzs / rs
        correction = np.where(active, -gap[rows, choice], 0.0)
        x[:nc, 0] += nx * correction
        x[:nc, 2] += nz * correction

        # normal impulse per node (XPBD impulse units kg*m), accumulated over
        # the iterations of this substep so the friction clamp stays consistent
        self.normal_lambda += correction * self.mass[:nc]
        lam_n = self.normal_lambda

        # Coulomb friction opposing tangential relative slip
        omega = self.cyl_omega[choice]
        surf_x = omega * dzs * h
        surf_z = -omega * dxs * h
        surf_y = self.cyl_vz[choice] * h
        ex = (x[:nc, 0] - x_prev[:nc, 0]) - surf_x
        ey = (x[:nc, 1] - x_prev[:nc, 1]) - surf_y
        ez = (x[:nc, 2] - x_prev[:nc, 2]) - surf_z
        en = ex * nx + ez * nz
        tx = ex - en * nx
        ty = ey
        tz = ez - en * nz
        tmag = np.sqrt(tx * tx + ty * ty + tz * tz)
        safe = np.where(tmag > 1e-12, tmag, 1.0)
        tx = tx / safe
        ty = ty / safe
        tz = tz / safe
        dlambda = -tmag * self.mass[:nc]
        limit = self.mu * lam_n
        trial = np.clip(self.friction_lambda + dlambda, -limit, limit)
        applied = trial - self.friction_lambda
        self.friction_lambda = trial
        applied = np.where(active, applied, 0.0)
        scale = applied * self.inv_mass[:nc]
        x[:nc, 0] += scale * tx
        x[:nc, 1] += scale * ty
        x[:nc, 2] += scale * tz

    def _substep(self, h: float) -> None:
        nc = self.n_cloth
        t_next = self.time + h
        self._update_kinematics(t_next)
        x = self.pos
        x_prev = x.copy()
        self.vel[:nc] += self.gravity * h
        x[:nc] += self.vel[:nc] * h

        lam = np.zeros(len(self.ca))
        self.friction_lambda[:] = 0.0
        self.normal_lambda[:] = 0.0
        for _ in range(ITERS):
            self._solve_distance(x, lam, h)
            self._solve_contacts(x, x_prev, h)

        self.vel[:nc] = (x[:nc] - x_prev[:nc]) / h
        self.vel[:nc] *= max(0.0, 1.0 - DAMPING * h)
        speed = np.linalg.norm(self.vel[:nc], axis=1)
        fast = speed > MAX_SPEED
        if fast.any():
            self.vel[:nc][fast] *= (MAX_SPEED / speed[fast])[:, None]
        self.pos[:nc] = x[:nc]
        self.time = t_next

    def advance_to(self, time: float):
        target = float(time)
        if target > self.time + 1e-12:
            total = target - self.time
            steps = max(1, int(round(total / (self.sample_dt / SUBSTEPS))))
            h = total / steps
            for _ in range(steps):
                self._substep(h)
            self.time = target
        return self._frame(self.time)

    # ---------------------------------------------------------------- exports
    def _deformation_gradient(self) -> np.ndarray:
        current = self.pos[: self.n_cloth]
        tri = current[self.faces]
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        normal = np.cross(e1, e2)
        length = np.linalg.norm(normal, axis=1)
        unit = normal / np.maximum(length, 1e-16)[:, None]
        scale = np.sqrt(np.maximum(length, 1e-16))
        ds = np.stack([e1, e2, unit * scale[:, None]], axis=2)
        f_tri = np.einsum("nij,njk->nik", ds, self.ref_dm_inv)
        f_tri = np.where(np.isfinite(f_tri), f_tri, 0.0)

        flat_faces = self.faces.ravel()
        weight = np.bincount(
            flat_faces, weights=np.tile(self.tri_areas, 3), minlength=self.n_cloth
        )
        safe = np.maximum(weight, 1e-30)
        gradient = np.zeros((self.n, 3, 3))
        for i in range(3):
            for j in range(3):
                acc = np.bincount(
                    flat_faces,
                    weights=np.tile(f_tri[:, i, j], 3),
                    minlength=self.n_cloth,
                )
                gradient[: self.n_cloth, i, j] = acc / safe
        gradient[self.n_cloth :] = np.eye(3)
        return gradient

    def _contact_gaps(self) -> np.ndarray:
        nc = self.n_cloth
        centres_x = self.pos[nc : nc + 3, 0]
        centres_z = self.pos[nc : nc + 3, 2]
        dx = self.pos[:nc, 0:1] - centres_x[None, :]
        dz = self.pos[:nc, 2:3] - centres_z[None, :]
        radial = np.sqrt(dx * dx + dz * dz)
        gap = radial - self.contact_radius
        inside = np.abs(self.pos[:nc, 1:2]) <= self.cyl_half_height
        gap = np.where(inside, gap, 1.0)
        return np.min(gap, axis=1)

    def _frame(self, t: float) -> StateFrame:
        frame = StateFrame(
            time=float(t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=self.orientation.copy(),
            angular_velocity=self.angular_velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "uv": self.uv.copy(),
                "mass_quantity": self.mass_quantity.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self._contact_gaps(),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "scene_id": self.config.get("scene_id", "unknown"),
            "seed": self.seed,
            "substeps_per_sample": SUBSTEPS,
            "constraint_count": int(len(self.ca)),
            "cloth_particles": int(self.n_cloth),
            "time": float(self.time),
            "min_contact_gap": float(np.min(self._contact_gaps())),
            "max_speed": float(np.max(np.linalg.norm(self.vel[: self.n_cloth], axis=1))),
        }
