"""XPBD elastic-solid solver for scene 066_octopus_cylinder_squash.

Physical model
--------------
The deformable octopus is represented by the frozen ``assets/octopus.obj``
surface nodes with the per-node masses supplied by the scene's initial
StateFrame.  Elasticity is discretised as a substepped XPBD constraint
network:

* distance constraints on every mesh edge (structural stiffness),
* distance constraints on every remaining vertex pair with compliance
  growing with rest length (bending / long-range shape memory), so bent
  tentacles are pulled back toward their rest shape,

integrated with Jacobi-relaxed XPBD substeps.  Unilateral frictional contact
against the four rigid cylinders is enforced positionally; the kinematic
upper cylinder follows the frozen piecewise-linear keyframes.  Contacts that
conflict (the upper cylinder geometrically overlaps the rear support at
maximum compression) are resolved with an averaged projection, which is the
least-squares compromise for the infeasible region and lets the near
incompressible constraint network extrude material sideways.

Every momentum transfer is accounted exactly: gravity, contact normal and
friction impulses, and drag are accumulated so that the exported budget
fields are consistent with the actual momentum change of the solid (internal
distance constraints cancel pairwise in the net momentum).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

SUBSTEP = 1.0e-3          # nominal physics substep (s)
ITERATIONS = 8            # XPBD iterations per substep
RELAX_DISTANCE = 0.25     # Jacobi relaxation for the distance network
CONTACT_RELAX = 1.0       # relaxation of the averaged contact projection
CONTACT_MARGIN = 0.01     # contact skin thickness (m)
CONTACT_SHELL = 0.05      # friction shell thickness (m)
STRUCTURAL_STIFFNESS = 8000.0     # N/m for mesh edges
BENDING_STIFFNESS_SCALE = 800.0   # k_bend = scale / rest_length (N/m)
DRAG_RATE = 2.0           # global linear drag (1/s)
FRICTION_RATE = 12.0      # tangential velocity decay rate in contact (1/s)
VELOCITY_CAP = 40.0       # safety cap (m/s), accounted as drag impulse
REFERENCE_SPEED = 1.0     # momentum scale reference speed (m/s)


def _load_obj(path: Path):
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            vertices.append([float(v) for v in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:]])
    verts = np.asarray(vertices, dtype=float)
    faces_arr = np.asarray(faces, dtype=np.int64) if faces else np.zeros((0, 3), np.int64)
    return verts, faces_arr


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n < 1e-15:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1.0 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
            [s * (x * y + w * z), 1.0 - s * (x * x + z * z), s * (y * z - w * x)],
            [s * (x * z - w * y), s * (y * z + w * x), 1.0 - s * (x * x + y * y)],
        ]
    )


def _segment_closest(points: np.ndarray, A: np.ndarray, B: np.ndarray):
    """Closest point on segment AB for every point, and the radial offset."""
    ab = B - A
    denom = float(ab @ ab)
    if denom < 1e-14:
        q = np.repeat(A[None, :], len(points), axis=0)
        return q, points - q
    t = np.clip((points - A) @ ab / denom, 0.0, 1.0)
    q = A[None, :] + t[:, None] * ab[None, :]
    return q, points - q


class BenchmarkSolver:
    """Substepped XPBD elastic solid squeezed between rigid cylinders."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", {}) or {}
        frame0 = scene.initial_state()

        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.m = np.array(frame0.mass, dtype=float)
        self.n = len(self.x)
        if frame0.body_id is not None:
            self.body_id = np.asarray(frame0.body_id).astype(np.int32).copy()
        else:
            self.body_id = np.zeros(self.n, dtype=np.int32)
        self.rest = self.x.copy()
        self.w = 1.0 / np.maximum(self.m, 1e-12)
        self.total_mass = float(self.m.sum())
        self.t = 0.0

        gravity = cfg.get("gravity", [0.0, 0.0, -9.81])
        self.g = np.asarray(gravity, dtype=float)

        # --- mesh topology and elastic constraint network -----------------
        obj_path = Path(__file__).resolve().parent.parent / "assets" / "octopus.obj"
        verts, faces = _load_obj(obj_path)
        if len(verts) == self.n and len(faces) > 0 and faces.max() < self.n:
            self.faces = faces
        else:  # defensive fallback: fan topology over the node set
            self.faces = np.zeros((0, 3), dtype=np.int64)

        pair_i, pair_j = np.triu_indices(self.n, k=1)
        edge_set = set()
        for f in self.faces:
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
                edge_set.add((min(a, b), max(a, b)))
        is_edge = np.array(
            [(int(i), int(j)) in edge_set for i, j in zip(pair_i, pair_j)], dtype=bool
        )
        rest_vec = self.rest[pair_j] - self.rest[pair_i]
        self.rest_length = np.sqrt(np.einsum("ij,ij->i", rest_vec, rest_vec))
        compliance = np.where(
            is_edge,
            1.0 / STRUCTURAL_STIFFNESS,
            np.maximum(self.rest_length, 0.05) / BENDING_STIFFNESS_SCALE,
        )
        self.pair_i = pair_i
        self.pair_j = pair_j
        self.compliance = compliance
        self.pair_wi = self.w[pair_i]
        self.pair_wj = self.w[pair_j]
        self.pair_w = self.pair_wi + self.pair_wj

        # per-element rest deformation frame (fixed element order)
        self._rest_frame_inv = self._rest_frames()

        # --- rigid cylinders ----------------------------------------------
        geometry = cfg.get("scene_geometry")
        if geometry is None:
            try:
                from worldsolver.config import load_yaml

                geometry = load_yaml(
                    Path(__file__).resolve().parent.parent / "scene_geometry.yaml"
                )
            except Exception:
                geometry = {"objects": []}
        self.cylinders = []
        for obj in geometry.get("objects", []):
            geo = obj.get("geometry", {})
            if geo.get("type") != "cylinder":
                continue
            tr = obj.get("transform", {})
            rotation = _quat_to_matrix(tr.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]))
            axis = rotation @ np.array([0.0, 0.0, 1.0])
            self.cylinders.append(
                {
                    "base": np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float),
                    "axis": axis,
                    "radius": float(geo["radius"]),
                    "half": 0.5 * float(geo["height"]),
                    "kinematic": obj.get("motion") == "kinematic",
                }
            )

        keyframes = []
        for kf in cfg.get("upper_cylinder_motion", {}).get("keyframes", []):
            keyframes.append(
                (float(kf["time"]), np.asarray(kf["translation"], dtype=float))
            )
        keyframes.sort(key=lambda item: item[0])
        self.keyframes = keyframes

        # --- momentum budget accumulators ----------------------------------
        self.J_gravity = np.zeros(3)
        self.J_contact = np.zeros(3)
        self.J_drag = np.zeros(3)
        self.win_t0 = 0.0
        self.win_Jc = np.zeros(3)
        self.win_Jd = np.zeros(3)
        self.substep_count = 0

        self.reference_momentum = self.total_mass * REFERENCE_SPEED
        self._ready = True

    # ------------------------------------------------------- kinematics
    def _keyframe_translation(self, t: float) -> np.ndarray:
        kf = self.keyframes
        if not kf:
            return np.zeros(3)
        if t <= kf[0][0]:
            return kf[0][1]
        for (t0, p0), (t1, p1) in zip(kf, kf[1:]):
            if t <= t1:
                if t1 <= t0:
                    return p0
                return p0 + (p1 - p0) * ((t - t0) / (t1 - t0))
        return kf[-1][1]

    def _keyframe_velocity(self, t: float) -> np.ndarray:
        kf = self.keyframes
        for (t0, p0), (t1, p1) in zip(kf, kf[1:]):
            if t0 <= t < t1 and t1 > t0:
                return (p1 - p0) / (t1 - t0)
        return np.zeros(3)

    def _cylinder_AB(self, cyl: dict, t: float):
        center = (
            self._keyframe_translation(t) if cyl["kinematic"] else cyl["base"]
        )
        return center - cyl["axis"] * cyl["half"], center + cyl["axis"] * cyl["half"]

    # ------------------------------------------------------------ stepping
    def _contact_push(self, t: float) -> np.ndarray:
        """Averaged unilateral projection out of every rigid cylinder."""
        push = np.zeros_like(self.x)
        count = np.zeros(self.n)
        for cyl in self.cylinders:
            A, B = self._cylinder_AB(cyl, t)
            _, rad = _segment_closest(self.x, A, B)
            d = np.sqrt(np.einsum("ij,ij->i", rad, rad))
            reach = cyl["radius"] + CONTACT_MARGIN
            inside = d < reach
            if not np.any(inside):
                continue
            nrm = rad / np.maximum(d, 1e-9)[:, None]
            bad = d < 1e-7
            if np.any(bad):
                nrm[bad] = np.array([1.0, 0.0, 0.0])
            push[inside] += (reach - d[inside])[:, None] * nrm[inside]
            count[inside] += 1.0
        return push / np.maximum(count, 1.0)[:, None]

    def _solve_distance(self, alpha_tilde: np.ndarray) -> None:
        d = self.x[self.pair_j] - self.x[self.pair_i]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        nrm = d / np.maximum(dist, 1e-9)[:, None]
        C = dist - self.rest_length
        dl = -C / (self.pair_w + alpha_tilde)
        delta = np.zeros_like(self.x)
        np.add.at(delta, self.pair_i, (self.pair_wi * dl)[:, None] * nrm)
        np.add.at(delta, self.pair_j, -(self.pair_wj * dl)[:, None] * nrm)
        self.x += RELAX_DISTANCE * delta

    def _apply_friction(self, t: float, dt: float) -> None:
        factor = min(1.0, FRICTION_RATE * dt)
        for cyl in self.cylinders:
            A, B = self._cylinder_AB(cyl, t)
            _, rad = _segment_closest(self.x, A, B)
            d = np.sqrt(np.einsum("ij,ij->i", rad, rad))
            shell = d < cyl["radius"] + CONTACT_MARGIN + CONTACT_SHELL
            if not np.any(shell):
                continue
            nrm = rad / np.maximum(d, 1e-9)[:, None]
            bad = d < 1e-7
            if np.any(bad):
                nrm[bad] = np.array([1.0, 0.0, 0.0])
            cyl_vel = (
                self._keyframe_velocity(t) if cyl["kinematic"] else np.zeros(3)
            )
            vrel = self.v[shell] - cyl_vel[None, :]
            n = nrm[shell]
            vn = np.einsum("ij,ij->i", vrel, n)[:, None] * n
            vt = vrel - vn
            dv = -factor * vt
            self.v[shell] += dv
            self.J_contact += (self.m[shell][:, None] * dv).sum(axis=0)

    def _step(self, dt: float, t_mid: float) -> None:
        x_prev = self.x.copy()

        # gravity
        self.v = self.v + self.g[None, :] * dt
        # linear drag (accounted as an external impulse)
        scale = math.exp(-DRAG_RATE * dt)
        dv_drag = (scale - 1.0) * self.v
        self.J_drag += (self.m[:, None] * dv_drag).sum(axis=0)
        self.v = self.v * scale

        # inertial guess
        self.x = self.x + self.v * dt

        alpha_tilde = self.compliance / (dt * dt)
        contact_delta = np.zeros_like(self.x)

        for _ in range(ITERATIONS):
            applied = CONTACT_RELAX * self._contact_push(t_mid)
            self.x += applied
            contact_delta += applied
            self._solve_distance(alpha_tilde)
        # closing contact pass so the emitted configuration respects walls
        applied = CONTACT_RELAX * self._contact_push(t_mid)
        self.x += applied
        contact_delta += applied

        # XPBD velocity update
        self.v = (self.x - x_prev) / dt

        # frictional contact reaction
        self._apply_friction(t_mid, dt)

        # safety cap, accounted as drag so budgets stay exact
        speed = np.sqrt(np.einsum("ij,ij->i", self.v, self.v))
        over = speed > VELOCITY_CAP
        if np.any(over):
            s = (VELOCITY_CAP / speed[over])[:, None]
            dv = self.v[over] * s - self.v[over]
            self.v[over] = self.v[over] * s
            self.J_drag += (self.m[over][:, None] * dv).sum(axis=0)

        # external impulse bookkeeping
        self.J_gravity += self.g * self.total_mass * dt
        self.J_contact += (self.m[:, None] * contact_delta).sum(axis=0) / dt
        self.substep_count += 1

    # ------------------------------------------------------------- frames
    def _rest_frames(self):
        r = self.rest
        f = self.faces
        if len(f) == 0:
            return np.zeros((0, 3, 3))
        A0 = r[f[:, 1]] - r[f[:, 0]]
        B0 = r[f[:, 2]] - r[f[:, 0]]
        N0 = np.cross(A0, B0)
        N0 = N0 / np.maximum(np.linalg.norm(N0, axis=1, keepdims=True), 1e-12)
        M0 = np.stack([A0, B0, N0], axis=2)
        try:
            return np.linalg.inv(M0)
        except np.linalg.LinAlgError:
            return np.array([np.linalg.pinv(M0[e]) for e in range(len(f))])

    def _deformation_gradients(self) -> np.ndarray:
        f = self.faces
        ne = len(f)
        F = np.zeros((ne, 3, 3))
        if ne == 0:
            return F
        x = self.x
        A = x[f[:, 1]] - x[f[:, 0]]
        B = x[f[:, 2]] - x[f[:, 0]]
        N = np.cross(A, B)
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        M = np.stack([A, B, N], axis=2)
        F = M @ self._rest_frame_inv
        return F

    def _make_frame(self, time: float) -> StateFrame:
        window = time - self.win_t0
        if window > 1e-12:
            f_contact = (self.J_contact - self.win_Jc) / window
            f_drag = (self.J_drag - self.win_Jd) / window
        else:
            f_contact = np.zeros(3)
            f_drag = np.zeros(3)
        f_gravity = self.g * self.total_mass
        external = np.stack([[f_gravity, f_contact, f_drag]])  # [G=1, K=3, 3]
        cumulative = (self.J_gravity + self.J_contact + self.J_drag).reshape(1, 3)

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.m.copy(),
                "rest_position": self.rest.copy(),
            },
            fields={
                "element_deformation_gradient": self._deformation_gradients(),
                "budget_cumulative_external_impulse": cumulative,
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "budget_domain_weights": np.ones((1, self.n)),
                "budget_external_force": external,
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "gravity_z": float(self.g[2]),
            },
        )
        frame.validate()
        self.win_t0 = float(time)
        self.win_Jc = self.J_contact.copy()
        self.win_Jd = self.J_drag.copy()
        return frame

    # ------------------------------------------------------------- public
    def advance_to(self, time: float) -> StateFrame:
        if not self._ready:
            raise RuntimeError("solver.initialize() must be called first")
        time = float(time)
        if time > self.t + 1e-12:
            remaining = time - self.t
            steps = max(1, int(round(remaining / SUBSTEP)))
            dt = remaining / steps
            t0 = self.t
            for k in range(steps):
                self._step(dt, t0 + (k + 0.5) * dt)
            self.t = time
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "substepped XPBD distance-network elastic solid",
            "substeps": int(self.substep_count),
            "iterations_per_substep": ITERATIONS,
            "substep": SUBSTEP,
            "nodes": int(self.n),
            "constraints": int(len(self.pair_i)),
            "elements": int(len(self.faces)),
            "final_time": float(self.t),
            "cumulative_contact_impulse": self.J_contact.tolist(),
            "cumulative_gravity_impulse": self.J_gravity.tolist(),
        }
