"""Cloth sleeve on articulated cylinders.

Self-contained physically-based solver (no external physics engine).

Model
-----
* The sleeve is the frozen 17x17 thin-sheet mesh of ``scene.py``, posed as a
  closed tube fitted around the two collinear capsule limbs, then relaxed under
  gravity with a real variational (XPBD) time-integration before t = 0.
* Kinematics: both limbs are capsules along the frozen scene transform.  The
  driven joint rotates ``limb_lower`` about the elbow hinge with a smooth
  schedule, which compresses the sleeve on the inner side of the bend.
* Dynamics: sub-stepped position based dynamics (implicit backward-Euler in
  position space).  Constraints: rigid edge (membrane) distances, compliant
  quadratic bending distances, unilateral frictional contact against the two
  capsules and the ground box.  Velocities are recovered as (x - x_prev)/dt.
* Exports raw physical state only: mass, velocity, per-vertex deformation
  gradient, external (gravity) force, signed contact gaps and mass source.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_WORLD_Z = np.array([0.0, 0.0, 1.0])


def _quat_to_matrix(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3)
    a = axis / norm
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) * c + s * K + (1.0 - c) * np.outer(a, a)


def _smoothstep(t: float) -> float:
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class BenchmarkSolver:
    # ---- numerical parameters -------------------------------------------------
    SUBSTEPS = 32                 # sub-steps per exported sample
    ITERATIONS = 3                # constraint sweeps per sub-step
    BEND_ALPHA = 0.02             # bending compliance (m/N), keeps folds smooth
    FRICTION_LIMB = 0.45
    FRICTION_GROUND = 0.45
    DAMPING = 0.9998              # per sub-step velocity damping
    SETTLE_DAMPING = 0.996
    SETTLE_TIME = 1.0             # relaxed pre-roll before t = 0
    RELAX_DAMP_TIME = 0.45
    BEND_START = 0.4              # joint drive schedule (seconds)
    BEND_RAMP = 4.4
    BEND_MAX = 0.50               # radians of elbow flexion
    SEAM_OVERLAP = 1.004

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = scene.config
        self.config = config
        cloth = config["cloth"]

        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.thickness = float(cloth.get("thickness", 0.01))
        self.total_mass = float(cloth.get("total_mass", 0.1))
        self.sample_dt = float(config.get("sample_dt", 1.0 / 30.0))
        self.duration = float(config.get("duration", 7.0))
        self.dt_sub = self.sample_dt / float(self.SUBSTEPS)
        self.seed = int(seed)

        frame0 = scene.initial_state()
        rest = np.asarray(frame0.position, dtype=float).copy()
        self.n = int(len(rest))
        self.rest_position = rest
        self.mass = np.full(self.n, self.total_mass / self.n, dtype=float)
        self.inv_mass = float(self.n / self.total_mass)
        self.mass_ratio = self.total_mass / self.n

        nx, ny = (int(v) for v in cloth.get("grid_shape", [17, 17]))
        self.nx, self.ny = nx, ny
        self.size = (float(cloth["size"][0]), float(cloth["size"][1]))
        plane = str(cloth.get("plane", "xz"))
        if plane == "xz":
            self.mat2 = np.column_stack((rest[:, 0], rest[:, 2]))
        elif plane == "yz":
            self.mat2 = np.column_stack((rest[:, 1], rest[:, 2]))
        else:
            self.mat2 = np.column_stack((rest[:, 0], rest[:, 1]))

        self.faces = self._load_faces(config)
        self._build_constraints()

        # ---- frozen limb geometry -------------------------------------------
        self.limbs = self._load_limbs(config)
        self._setup_joint()

        # ---- ground box (cached) --------------------------------------------
        self._has_ground = False
        self._ground_center = np.zeros(3)
        self._ground_half = np.zeros(3)
        for obj in (config.get("scene_geometry") or {}).get("objects", []) or []:
            if (obj.get("geometry") or {}).get("type") == "box":
                self._has_ground = True
                self._ground_half = np.asarray(obj["geometry"]["half_extents"], dtype=float)
                self._ground_center = np.asarray(
                    (obj.get("transform") or {}).get("translation", [0.0, 0.0, 0.0]), dtype=float
                )
                break

        # ---- initial fitted sleeve ------------------------------------------
        self.pos = self._initial_sleeve()
        self.vel = np.zeros_like(self.pos)
        self.step = 0

        # bending rest geometry from the relaxed fitted pose (posed tube)
        db = self.pos[self.bend_e1] - self.pos[self.bend_e0]
        self.bend_rest = np.linalg.norm(db, axis=1)
        self.bend_deg = np.zeros(self.n, dtype=float)
        np.add.at(self.bend_deg, self.bend_e0, 1.0)
        np.add.at(self.bend_deg, self.bend_e1, 1.0)
        self.bend_deg = np.maximum(self.bend_deg, 1.0)

        self._update_limbs(0.0)
        # relax the fitted sleeve with real time stepping before sampling
        for _ in range(int(round(self.SETTLE_TIME / self.dt_sub))):
            self._substep(self.dt_sub, settling=True)
        self.vel *= 0.0
        self._max_stretch = 1.0
        self._min_gap = 0.0

    # ------------------------------------------------------------ mesh helpers
    def _load_faces(self, config: dict) -> np.ndarray:
        geometry = config.get("scene_geometry") or {}
        asset = None
        for obj in geometry.get("objects", []) or []:
            if obj.get("id") == "cloth":
                asset = (obj.get("geometry") or {}).get("topology_asset")
                break
        if asset:
            for candidate in (Path(str(asset)), Path.cwd() / str(asset)):
                try:
                    if candidate.is_file():
                        data = json.loads(candidate.read_text(encoding="utf-8"))
                        faces = np.asarray(data["faces"], dtype=np.int64)
                        if faces.ndim == 2 and faces.shape[1] == 3:
                            return faces
                except Exception:
                    continue
        nx, ny = self.nx, self.ny
        idx = lambda i, j: j * nx + i
        tris = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
                tris.append((a, b, c))
                tris.append((a, c, d))
        return np.asarray(tris, dtype=np.int64)

    def _build_constraints(self) -> None:
        f = self.faces
        raw = np.vstack((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
        edges = np.unique(np.sort(raw, axis=1), axis=0)
        self.edge_e0 = edges[:, 0].copy()
        self.edge_e1 = edges[:, 1].copy()

        nx, ny = self.nx, self.ny
        idx = lambda i, j: j * nx + i
        bend = []
        for j in range(ny):
            for i in range(nx - 2):
                bend.append((idx(i, j), idx(i + 2, j)))
        for j in range(ny - 2):
            for i in range(nx):
                bend.append((idx(i, j), idx(i, j + 2)))
        bend = np.asarray(bend, dtype=np.int64)
        self.bend_e0 = bend[:, 0].copy()
        self.bend_e1 = bend[:, 1].copy()

        # rest lengths from the material metric (flat sheet) for membrane edges
        d = self.rest_position[self.edge_e1] - self.rest_position[self.edge_e0]
        self.edge_rest = np.linalg.norm(d, axis=1)
        self.edge_deg = np.zeros(self.n, dtype=float)
        np.add.at(self.edge_deg, self.edge_e0, 1.0)
        np.add.at(self.edge_deg, self.edge_e1, 1.0)
        self.edge_deg = np.maximum(self.edge_deg, 1.0)

        # triangle areas (for area weighted vertex deformation gradients)
        p = self.rest_position
        a1 = p[f[:, 1]] - p[f[:, 0]]
        a2 = p[f[:, 2]] - p[f[:, 0]]
        self.face_area = 0.5 * np.linalg.norm(np.cross(a1, a2), axis=1)
        self.vertex_area = np.zeros(self.n, dtype=float)
        for c in range(3):
            np.add.at(self.vertex_area, f[:, c], self.face_area / 3.0)
        self.vertex_area = np.maximum(self.vertex_area, 1e-12)

        # material (rest) triangle edge matrices, one per face
        m = self.mat2
        dm = np.stack((m[f[:, 1]] - m[f[:, 0]], m[f[:, 2]] - m[f[:, 0]]), axis=2)  # (F,2,2)
        self.face_dm = dm
        self.face_dm_inv = np.linalg.inv(dm)

    def _load_limbs(self, config: dict) -> list[dict]:
        geometry = config.get("scene_geometry") or {}
        limbs = []
        for obj in geometry.get("objects", []) or []:
            geo = obj.get("geometry") or {}
            if geo.get("type") != "capsule" or obj.get("motion") != "kinematic":
                continue
            transform = obj.get("transform") or {}
            rot = _quat_to_matrix(transform.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]))
            center = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            axis = rot @ np.array([0.0, 0.0, 1.0])
            limbs.append(
                {
                    "id": str(obj.get("id", f"limb{len(limbs)}")),
                    "radius": float(geo.get("radius", 0.3)),
                    "half": 0.5 * float(geo.get("height", 1.0)),
                    "center": center,
                    "axis": axis / max(float(np.linalg.norm(axis)), 1e-12),
                    "rot": rot,
                }
            )
        if not limbs:  # defensive fallback matching the frozen scene
            limbs = [
                {
                    "id": "limb_upper",
                    "radius": 0.34,
                    "half": 0.65,
                    "center": np.array([-0.55, 0.0, 1.0]),
                    "axis": np.array([1.0, 0.0, 0.0]),
                    "rot": np.eye(3),
                },
                {
                    "id": "limb_lower",
                    "radius": 0.31,
                    "half": 0.65,
                    "center": np.array([0.55, 0.0, 1.0]),
                    "axis": np.array([1.0, 0.0, 0.0]),
                    "rot": np.eye(3),
                },
            ]
        return limbs

    def _setup_joint(self) -> None:
        base = self.limbs[0]
        order = np.argsort([float(np.dot(lb["center"], base["axis"])) for lb in self.limbs])
        self.static_limb = self.limbs[int(order[0])]
        self.driven_limb = self.limbs[int(order[-1])]

        axis_s = self.static_limb["axis"]
        midpoint = 0.5 * (self.static_limb["center"] + self.driven_limb["center"])
        # project the limb midpoint onto the static limb axis: this is the elbow
        self.hinge = self.static_limb["center"] + np.dot(
            midpoint - self.static_limb["center"], axis_s
        ) * axis_s

        candidate = np.cross(axis_s, _WORLD_Z)
        if float(np.linalg.norm(candidate)) < 1e-6:
            candidate = np.cross(axis_s, np.array([0.0, 1.0, 0.0]))
        if float(np.linalg.norm(candidate)) < 1e-6:
            candidate = np.array([0.0, 1.0, 0.0])
        candidate = candidate / float(np.linalg.norm(candidate))

        # choose the sign that lifts the driven limb (compression on inner side)
        distal = self.driven_limb["center"] + self.driven_limb["axis"] * self.driven_limb["half"]
        probe = _axis_angle_matrix(candidate, 0.2) @ (distal - self.hinge) + self.hinge
        if probe[2] < distal[2]:
            candidate = -candidate
        self.hinge_axis = candidate

        axis = self.static_limb["axis"]
        e_u = np.cross(axis, _WORLD_Z)
        if float(np.linalg.norm(e_u)) < 1e-6:
            e_u = np.cross(axis, np.array([0.0, 1.0, 0.0]))
        e_u = e_u / max(float(np.linalg.norm(e_u)), 1e-12)
        self.limb_frame = (e_u, np.cross(axis, e_u))
        self.driven_index = self.limbs.index(self.driven_limb)

    def _initial_sleeve(self) -> np.ndarray:
        nx, ny = self.nx, self.ny
        sx, sy = self.size
        k = np.arange(self.n)
        i = k % nx
        j = k // nx
        x = -0.5 * sx + (sx / (nx - 1)) * i
        v = -0.5 * sy + (sy / (ny - 1)) * j
        radius = sy / (2.0 * np.pi)
        phi = 0.5 * np.pi + self.SEAM_OVERLAP * (2.0 * np.pi) * (v / sy)
        center = self.hinge
        offset = self.static_limb["axis"] * x[:, None]
        e_u, e_v = self.limb_frame
        radial = np.sin(phi)[:, None] * e_u[None, :] + np.cos(phi)[:, None] * e_v[None, :]
        return center[None, :] + offset + radius * radial

    # ----------------------------------------------------------- limb motion
    def _joint_angle(self, t: float) -> float:
        return self.BEND_MAX * _smoothstep((t - self.BEND_START) / self.BEND_RAMP)

    def _update_limbs(self, theta: float) -> None:
        rot = _axis_angle_matrix(self.hinge_axis, theta)
        a_list, b_list, r_list = [], [], []
        for limb in self.limbs:
            center = limb["center"]
            axis = limb["axis"]
            if limb is self.driven_limb:
                center = self.hinge + rot @ (center - self.hinge)
                axis = rot @ axis
            a_list.append(center + axis * limb["half"])
            b_list.append(center - axis * limb["half"])
            r_list.append(limb["radius"])
        self.capsule_a = np.asarray(a_list)
        self.capsule_b = np.asarray(b_list)
        self.capsule_r = np.asarray(r_list)
        self.theta = theta

    # ------------------------------------------------------------ integration
    def _substep(self, dt: float, settling: bool = False) -> None:
        t = self.step * dt
        theta = self._joint_angle(t)
        self._update_limbs(theta)

        x_prev = self.pos
        vel = self.vel + dt * self.gravity[None, :]
        p = x_prev + dt * vel

        if settling:
            # settle immediately after the initial impact, never during playback
            if t < self.RELAX_DAMP_TIME:
                vel = vel * 0.9
                p = x_prev + dt * vel

        for _ in range(self.ITERATIONS):
            p = self._stretch_pass(p)
            p = self._bend_pass(p, dt)
            p = self._contact_pass(p, x_prev, dt)

        vel = (p - x_prev) / dt
        if settling:
            damp = self.SETTLE_DAMPING
        else:
            damp = self.DAMPING
        vel = vel * damp

        self.pos = p
        self.vel = vel
        self.step += 1

    def _stretch_pass(self, p: np.ndarray) -> np.ndarray:
        d = p[self.edge_e1] - p[self.edge_e0]
        length = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
        corr = (length - self.edge_rest) / length
        step = (0.5 * corr)[:, None] * d
        acc = np.zeros_like(p)
        np.add.at(acc, self.edge_e0, -step)
        np.add.at(acc, self.edge_e1, step)
        acc /= self.edge_deg[:, None]
        return p + acc

    def _bend_pass(self, p: np.ndarray, dt: float) -> np.ndarray:
        d = p[self.bend_e1] - p[self.bend_e0]
        length = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
        corr = length - self.bend_rest
        w = self.inv_mass
        alpha = self.BEND_ALPHA / (dt * dt)
        scale = (corr * w) / (2.0 * w + alpha)
        step = scale[:, None] * (d / length[:, None])
        acc = np.zeros_like(p)
        np.add.at(acc, self.bend_e0, -step)
        np.add.at(acc, self.bend_e1, step)
        acc /= np.maximum(self.bend_deg[:, None], 1.0)
        return p + acc

    def _contact_pass(self, p: np.ndarray, x_prev: np.ndarray, dt: float) -> np.ndarray:
        offset = 0.5 * self.thickness
        omega = self._joint_angular_velocity(dt) * self.hinge_axis
        for idx in range(len(self.capsule_a)):
            a = self.capsule_a[idx]
            b = self.capsule_b[idx]
            ab = b - a
            ab2 = float(ab @ ab)
            if ab2 < 1e-12:
                continue
            rel = p - a[None, :]
            t = np.clip((rel @ ab) / ab2, 0.0, 1.0)
            q = a[None, :] + t[:, None] * ab[None, :]
            diff = p - q
            dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
            gap = dist - (self.capsule_r[idx] + offset)
            act = np.flatnonzero(gap < 0.0)
            if act.size == 0:
                continue
            safe = np.maximum(dist[act], 1e-9)
            nrm = diff[act] / safe[:, None]
            shallow = dist[act] < 1e-9
            if np.any(shallow):
                nrm[shallow] = _WORLD_Z[None, :]
            p[act] = p[act] - gap[act][:, None] * nrm
            surf = np.zeros_like(nrm)
            if idx == self.driven_index:
                lever = q[act] - self.hinge[None, :]
                surf = np.cross(np.broadcast_to(omega, lever.shape), lever) * dt
            p[act] = self._apply_friction(p[act], x_prev[act], nrm, surf, self.FRICTION_LIMB)
        p = self._ground_contact(p, x_prev)
        return p

    def _joint_angular_velocity(self, dt: float) -> float:
        t = self.step * dt
        return (self._joint_angle(t + dt) - self._joint_angle(t)) / max(dt, 1e-12)

    def _apply_friction(
        self,
        p: np.ndarray,
        x_prev: np.ndarray,
        nrm: np.ndarray,
        surf: np.ndarray,
        mu: float,
    ) -> np.ndarray:
        disp = (p - x_prev) - surf
        dn = np.einsum("ij,ij->i", disp, nrm)
        tangent = disp - nrm * dn[:, None]
        slip = np.sqrt(np.einsum("ij,ij->i", tangent, tangent))
        limit = mu * np.maximum(dn, 0.0)
        excess = slip - limit
        mask = (excess > 1e-12) & (slip > 1e-12)
        if np.any(mask):
            correction = (
                (excess[mask] / slip[mask])[:, None] * tangent[mask]
            )
            p[mask] = p[mask] - correction
        return p

    def _ground_contact(self, p: np.ndarray, x_prev: np.ndarray) -> np.ndarray:
        if not self._has_ground:
            return p
        half = self._ground_half
        center = self._ground_center
        rel = np.abs(p - center[None, :]) - half[None, :]
        sdf = np.linalg.norm(np.maximum(rel, 0.0), axis=1) + np.minimum(rel.max(axis=1), 0.0)
        gap = sdf - 0.5 * self.thickness
        act = np.flatnonzero(gap < 0.0)
        if act.size == 0:
            return p
        nrm = np.zeros((act.size, 3))
        vec = p[act] - center[None, :]
        for axis in range(3):
            near = np.abs(np.abs(vec[:, axis]) - half[axis]) < 1e-6
            nrm[near, axis] = np.sign(vec[near, axis])
        bad = np.linalg.norm(nrm, axis=1) < 1e-9
        nrm[bad] = _WORLD_Z[None, :]
        p[act] = p[act] - gap[act][:, None] * nrm
        p[act] = self._apply_friction(
            p[act], x_prev[act], nrm, np.zeros_like(nrm), self.FRICTION_GROUND
        )
        return p

    # -------------------------------------------------------------- reporting
    def _deformation_gradient(self, p: np.ndarray) -> np.ndarray:
        f = self.faces
        d0 = p[f[:, 1]] - p[f[:, 0]]
        d1 = p[f[:, 2]] - p[f[:, 0]]
        e1 = d0 / (np.linalg.norm(d0, axis=1)[:, None] + 1e-12)
        nvec = np.cross(d0, d1)
        nvec = nvec / (np.linalg.norm(nvec, axis=1)[:, None] + 1e-12)
        e2 = np.cross(nvec, e1)
        ds = np.stack(
            (
                np.stack(
                    (np.einsum("ij,ij->i", d0, e1), np.einsum("ij,ij->i", d0, e2)), axis=1
                ),
                np.stack(
                    (np.einsum("ij,ij->i", d1, e1), np.einsum("ij,ij->i", d1, e2)), axis=1
                ),
            ),
            axis=2,
        )
        fmat = ds @ self.face_dm_inv
        out = np.zeros((self.n, 2, 2))
        weight = np.zeros(self.n)
        for row in range(3):
            np.add.at(out, f[:, row], self.face_area[:, None, None] * fmat)
            np.add.at(weight, f[:, row], self.face_area)
        weight = np.maximum(weight, 1e-12)
        out /= weight[:, None, None]
        return out

    def _contact_gaps(self, p: np.ndarray) -> np.ndarray:
        offset = 0.5 * self.thickness
        gaps = np.full(self.n, np.inf)
        for idx in range(len(self.capsule_a)):
            a = self.capsule_a[idx]
            b = self.capsule_b[idx]
            ab = b - a
            ab2 = float(ab @ ab)
            rel = p - a[None, :]
            t = np.clip((rel @ ab) / max(ab2, 1e-12), 0.0, 1.0)
            q = a[None, :] + t[:, None] * ab[None, :]
            dist = np.linalg.norm(p - q, axis=1)
            gaps = np.minimum(gaps, dist - (self.capsule_r[idx] + offset))
        if self._has_ground:
            rel = np.abs(p - self._ground_center[None, :]) - self._ground_half[None, :]
            sdf = np.linalg.norm(np.maximum(rel, 0.0), axis=1) + np.minimum(
                rel.max(axis=1), 0.0
            )
            gaps = np.minimum(gaps, sdf - offset)
        return gaps

    def _frame(self, time: float) -> StateFrame:
        p = self.pos
        gaps = self._contact_gaps(p)
        band = max(4.0 * self.thickness, 0.05)
        selected = gaps < band
        if int(np.count_nonzero(selected)) < 12:
            idx = np.argsort(gaps, kind="stable")[:12]
            distances = gaps[idx]
        else:
            distances = gaps[selected]
        distances = np.sort(distances)
        external = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(time),
            p.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(p),
            },
            fields={
                "contact_signed_distance": distances,
                "cumulative_mass_source": np.zeros(1),
                "external_force": external,
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        self._min_gap = float(np.min(gaps))
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        target = int(round(float(time) / self.dt_sub))
        if target < self.step:
            target = self.step
        while self.step < target:
            self._substep(self.dt_sub)
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        d = self.pos[self.edge_e1] - self.pos[self.edge_e0]
        length = np.linalg.norm(d, axis=1)
        ratio = length / np.maximum(self.edge_rest, 1e-12)
        return {
            "method": "substepped XPBD with frictional capsule contact",
            "substeps": int(self.step),
            "dt_sub": float(self.dt_sub),
            "max_edge_stretch": float(np.max(ratio)),
            "min_edge_stretch": float(np.min(ratio)),
            "min_contact_gap": float(self._min_gap),
            "joint_angle": float(self.theta),
            "seed": self.seed,
        }
