"""Physically-based solver: soft volumetric objects squeezed through a narrow outlet.

Model
-----
Five separate volumetric soft objects (the frozen ``soft_*`` deformable scene
objects) are each meshed with the closed 12-vertex / 20-triangle icosahedral
surface of ``assets/soft_body_unit.obj``.  The solid response is a discrete
elastic model of that surface:

* a network of edge springs (Neo-Hookean-like distance constraints) with
  per-body rest lengths taken from the frozen initial transform;
* a per-body enclosed-volume constraint that preserves the reference volume, so
  bodies flatten/elongate while being forced through the outlet and re-inflate
  into distinguishable bodies afterwards.

Integration is substepped XPBD (implicit-Euler position based dynamics), i.e. a
real, stable, deterministic time-stepping scheme (Macklin et al., XPBD).  All
contacts are resolved against the frozen rigid geometry of ``scene_geometry.yaml``
(chamber walls, funnel, outlet lips, collector, moving pusher plate with the
prescribed translation) and between the soft bodies themselves, with inelastic
normal response, Coulomb-style tangential friction and gravity from the config.
No animation tables, keyframes or precomputed motion are used: every state comes
from stepping this physical model.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame

# --------------------------------------------------------------------------
# frozen unit soft-body surface (assets/soft_body_unit.obj, icosahedron)
# --------------------------------------------------------------------------
_MESH_VERTICES = np.array(
    [
        [-0.525731, 0.850651, 0.000000],
        [0.525731, 0.850651, 0.000000],
        [-0.525731, -0.850651, 0.000000],
        [0.525731, -0.850651, 0.000000],
        [0.000000, -0.525731, 0.850651],
        [0.000000, 0.525731, 0.850651],
        [0.000000, -0.525731, -0.850651],
        [0.000000, 0.525731, -0.850651],
        [0.850651, 0.000000, -0.525731],
        [0.850651, 0.000000, 0.525731],
        [-0.850651, 0.000000, -0.525731],
        [-0.850651, 0.000000, 0.525731],
    ],
    dtype=float,
)
_MESH_FACES = np.array(
    [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ],
    dtype=np.int64,
)

# Fallback frozen body transforms (used only if scene geometry is unavailable).
_DEFAULT_BODY_TRANSFORMS = (
    ((-0.25, -0.12, 4.13), (0.0, 0.0, 0.0, 1.0), (0.45, 0.35, 0.33)),
    ((0.22, 0.12, 3.34), (0.0, 0.259, 0.0, 0.966), (0.33, 0.30, 0.55)),
    ((-0.18, 0.10, 2.55), (0.0, 0.0, 0.0, 1.0), (0.45, 0.39, 0.46)),
    ((0.18, -0.10, 1.72), (0.0, -0.174, 0.0, 0.985), (0.34, 0.32, 0.56)),
    ((-0.20, -0.02, 0.98), (0.0, 0.0, 0.0, 1.0), (0.42, 0.33, 0.30)),
)

# Fallback frozen static/kinematic box geometry (scene_geometry.yaml).
_DEFAULT_BOXES = (
    ("chamber_back", (0.84, 0.06, 2.0), (0.0, 0.58, 2.55), (0.0, 0.0, 0.0, 1.0), False),
    ("chamber_left", (0.06, 0.52, 2.0), (-0.78, 0.0, 2.55), (0.0, 0.0, 0.0, 1.0), False),
    ("chamber_right", (0.06, 0.52, 2.0), (0.78, 0.0, 2.55), (0.0, 0.0, 0.0, 1.0), False),
    ("funnel_left", (0.055, 0.52, 0.75), (-0.55, 0.0, 0.72), (0.0, -0.153, 0.0, 0.988), False),
    ("funnel_right", (0.055, 0.52, 0.75), (0.55, 0.0, 0.72), (0.0, 0.153, 0.0, 0.988), False),
    ("funnel_back", (0.72, 0.055, 0.75), (0.0, 0.38, 0.72), (0.19, 0.0, 0.0, 0.982), False),
    ("outlet_front_lip", (0.34, 0.2, 0.07), (0.0, -0.33, 0.0), (0.0, 0.0, 0.0, 1.0), False),
    ("outlet_back_lip", (0.34, 0.2, 0.07), (0.0, 0.33, 0.0), (0.0, 0.0, 0.0, 1.0), False),
    ("collector_floor", (1.25, 0.9, 0.06), (0.0, 0.0, -2.05), (0.0, 0.0, 0.0, 1.0), False),
    ("collector_back", (1.25, 0.05, 1.0), (0.0, 0.95, -1.05), (0.0, 0.0, 0.0, 1.0), False),
    ("collector_left", (0.05, 0.9, 1.0), (-1.3, 0.0, -1.05), (0.0, 0.0, 0.0, 1.0), False),
    ("collector_right", (0.05, 0.9, 1.0), (1.3, 0.0, -1.05), (0.0, 0.0, 0.0, 1.0), False),
    ("pusher_plate", (0.7, 0.5, 0.08), (0.0, 0.0, 4.85), (0.0, 0.0, 0.0, 1.0), True),
)

_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)

_SUBSTEP = 1.0 / 1000.0        # s, XPBD substep
_ITERATIONS = 2                # constraint iterations per substep
_SPRING_STIFFNESS = 400.0      # N/m, edge stiffness (soft solid)
_VOLUME_COMPLIANCE = 1.0e-9    # XPBD compliance for enclosed volume preservation
_WALL_MARGIN = 0.02            # m, node collision skin against rigid boxes
_BODY_GAP = 0.075              # m, node-node separation between distinct bodies
_FRICTION = 0.45               # tangential velocity reduction at contacts
_DAMPING = 0.6                 # 1/s, weak viscous damping
_MAX_SPEED = 30.0              # m/s, numerical safety clamp
_MAX_CONTACT_PUSH = 0.05       # m, per-substep positional correction safety cap


def _quat_to_matrix(quaternion: Any) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion (normalised, deterministic)."""
    x, y, z, w = (float(v) for v in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = set()
    for face in faces:
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges.add((int(min(a, b)), int(max(a, b))))
    return np.array(sorted(edges), dtype=np.int64)


class BenchmarkSolver:
    """Substepped XPBD soft-solid solver for the squeeze-out benchmark."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        self.time = 0.0
        config = getattr(scene, "config", None) or {}
        self.config = config

        state0 = scene.initial_state()
        centres = np.asarray(state0.position, dtype=float).copy()
        body_masses = np.asarray(state0.mass, dtype=float).reshape(-1).copy()
        n_bodies = int(len(centres))
        self.n_bodies = n_bodies

        geometry = config.get("scene_geometry") or {}
        objects = [o for o in geometry.get("objects", []) if isinstance(o, dict)]

        # ---- soft body reference (rest) configurations -------------------
        specs: list[tuple[Any, Any, Any]] = []
        for index in range(n_bodies):
            specs.append(_DEFAULT_BODY_TRANSFORMS[index % len(_DEFAULT_BODY_TRANSFORMS)])
        for obj in objects:
            if obj.get("semantic") != "deformable":
                continue
            binding = obj.get("state_binding") or {}
            if binding.get("selector") != "body_id":
                continue
            index = int(binding.get("value", -1))
            if 0 <= index < n_bodies:
                transform = obj.get("transform") or {}
                specs[index] = (
                    transform.get("translation", centres[index].tolist()),
                    transform.get("rotation_xyzw", (0.0, 0.0, 0.0, 1.0)),
                    transform.get("scale", (1.0, 1.0, 1.0)),
                )

        n_vertices = len(_MESH_VERTICES)
        self.n_nodes = n_bodies * n_vertices
        self.rest = np.zeros((self.n_nodes, 3), dtype=float)
        self.node_body = np.repeat(np.arange(n_bodies, dtype=np.int64), n_vertices)
        node_mass = np.zeros(self.n_nodes, dtype=float)
        for index in range(n_bodies):
            translation, quaternion, scale = specs[index]
            rotation = _quat_to_matrix(quaternion)
            scaled = _MESH_VERTICES * np.asarray(scale, dtype=float)
            sl = slice(index * n_vertices, (index + 1) * n_vertices)
            self.rest[sl] = np.asarray(translation, dtype=float) + scaled @ rotation.T
            node_mass[sl] = float(body_masses[index]) / n_vertices
        self.x = self.rest.copy()
        self.v = np.zeros_like(self.x)
        self.mass = node_mass
        self.w = 1.0 / np.maximum(node_mass, 1e-12)
        self.body_mass = body_masses.copy()
        self.body_id = self.node_body.copy()

        # ---- spring network ----------------------------------------------
        local_edges = _mesh_edges(_MESH_FACES)
        edge_i = []
        edge_j = []
        for index in range(n_bodies):
            base = index * n_vertices
            edge_i.append(local_edges[:, 0] + base)
            edge_j.append(local_edges[:, 1] + base)
        self.edge_i = np.concatenate(edge_i)
        self.edge_j = np.concatenate(edge_j)
        self.edge_rest = np.linalg.norm(
            self.rest[self.edge_i] - self.rest[self.edge_j], axis=1
        )
        self.lam_e = np.zeros(len(self.edge_i))
        self.alpha_spring = 1.0 / _SPRING_STIFFNESS

        # ---- volume constraints ------------------------------------------
        face_a = []
        face_b = []
        face_c = []
        for index in range(n_bodies):
            base = index * n_vertices
            face_a.append(_MESH_FACES[:, 0] + base)
            face_b.append(_MESH_FACES[:, 1] + base)
            face_c.append(_MESH_FACES[:, 2] + base)
        self.fa = np.concatenate(face_a)
        self.fb = np.concatenate(face_b)
        self.fc = np.concatenate(face_c)
        self.el_body = np.repeat(np.arange(n_bodies, dtype=np.int64), len(_MESH_FACES))
        self.rest_volume = self._body_volume(self.rest)
        self.lam_v = np.zeros(n_bodies)
        self.alpha_volume = _VOLUME_COMPLIANCE

        # reference basis (inverse) for each surface element's deformation gradient
        self.element_pinv = self._element_basis(self.rest)

        # ---- rigid box geometry ------------------------------------------
        self.boxes: list[dict[str, Any]] = []
        seen: set[str] = set()
        for obj in objects:
            if obj.get("motion") not in {"static", "kinematic"}:
                continue
            geom = obj.get("geometry") or {}
            if geom.get("type") != "box" or geom.get("half_extents") is None:
                continue
            half = np.asarray(geom["half_extents"], dtype=float).reshape(3)
            transform = obj.get("transform") or {}
            centre = np.asarray(transform.get("translation", (0.0, 0.0, 0.0)), dtype=float)
            rotation = _quat_to_matrix(transform.get("rotation_xyzw", (0.0, 0.0, 0.0, 1.0)))
            self.boxes.append(
                {
                    "id": str(obj.get("id", "")),
                    "half": half,
                    "centre0": centre,
                    "rotation": rotation,
                    "kinematic": obj.get("motion") == "kinematic",
                }
            )
            seen.add(str(obj.get("id", "")))
        if not self.boxes:
            for name, half, centre, quaternion, kinematic in _DEFAULT_BOXES:
                self.boxes.append(
                    {
                        "id": name,
                        "half": np.asarray(half, dtype=float),
                        "centre0": np.asarray(centre, dtype=float),
                        "rotation": _quat_to_matrix(quaternion),
                        "kinematic": bool(kinematic),
                    }
                )
                seen.add(name)

        # ---- prescribed pusher-plate motion ------------------------------
        plate = config.get("boundary_conditions", {}).get("plate_motion", {}) or {}
        self.plate_motion = {
            "object_id": str(plate.get("object_id", "pusher_plate")),
            "start_time": float(plate.get("start_time", 0.0)),
            "end_time": float(plate.get("end_time", 0.0)),
            "start_translation": np.asarray(
                plate.get("start_translation", (0.0, 0.0, 0.0)), dtype=float
            ),
            "end_translation": np.asarray(
                plate.get("end_translation", (0.0, 0.0, 0.0)), dtype=float
            ),
        }
        for box in self.boxes:
            if box["kinematic"]:
                box["centre0"] = self.plate_motion["start_translation"].copy()

        # ---- contact scratch / budget accumulators -----------------------
        self.contact_n = np.zeros((self.n_nodes, 3))
        self.contact_target_v = np.zeros((self.n_nodes, 3))
        self.contact_active = np.zeros(self.n_nodes, dtype=bool)
        self.acc_wall = np.zeros((n_bodies, 3))
        self.acc_pair = np.zeros((n_bodies, n_bodies, 3))
        self.acc_ext = np.zeros((6, 3))
        self.acc_force = np.zeros((6, 11, 3))
        self.acc_pair_force = np.zeros((n_bodies, n_bodies, 3))

        total_mass = float(np.sum(self.body_mass))
        plate_speed = 0.0
        span = self.plate_motion["end_time"] - self.plate_motion["start_time"]
        if span > 1e-9:
            plate_speed = float(
                np.linalg.norm(self.plate_motion["end_translation"] - self.plate_motion["start_translation"])
                / span
            )
        self.reference_momentum = max(total_mass * max(plate_speed, 0.1), 1e-6)
        self.h = _SUBSTEP

        # domain weights: row 0 = global, rows 1..n_bodies = single bodies
        weights = np.zeros((6, self.n_nodes), dtype=float)
        weights[0] = 1.0
        for index in range(n_bodies):
            weights[1 + index] = (self.node_body == index).astype(float)
        self.domain_weights = weights
        self.n_domains = weights.shape[0]
        self.n_terms = 11

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    def _plate_translation(self, t: float) -> np.ndarray:
        motion = self.plate_motion
        start, end = motion["start_time"], motion["end_time"]
        if end - start > 1e-12:
            frac = (t - start) / (end - start)
        else:
            frac = 1.0 if t >= start else 0.0
        frac = min(max(frac, 0.0), 1.0)
        return motion["start_translation"] + frac * (
            motion["end_translation"] - motion["start_translation"]
        )

    def _plate_velocity(self, t: float) -> np.ndarray:
        motion = self.plate_motion
        start, end = motion["start_time"], motion["end_time"]
        if end - start > 1e-12 and start <= t <= end:
            return (motion["end_translation"] - motion["start_translation"]) / (end - start)
        return np.zeros(3)

    def _element_basis(self, positions: np.ndarray) -> np.ndarray:
        pa = positions[self.fa]
        pb = positions[self.fb]
        pc = positions[self.fc]
        e1 = pb - pa
        e2 = pc - pa
        normal = np.cross(e1, e2)
        length = np.linalg.norm(normal, axis=1)
        bad = length < 1e-12
        length = np.where(bad, 1.0, length)
        normal = normal / length[:, None]
        basis = np.stack((e1, e2, normal), axis=2)
        pinv = np.empty_like(basis)
        for index in range(len(basis)):
            try:
                pinv[index] = np.linalg.inv(basis[index])
            except np.linalg.LinAlgError:
                pinv[index] = np.eye(3)
        return pinv

    def _body_volume(self, positions: np.ndarray) -> np.ndarray:
        pa = positions[self.fa]
        pb = positions[self.fb]
        pc = positions[self.fc]
        face_volume = np.einsum("ij,ij->i", pa, np.cross(pb, pc)) / 6.0
        return np.bincount(self.el_body, weights=face_volume, minlength=self.n_bodies)

    # ------------------------------------------------------------------
    # integration
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target):
            target = self.time
        dt = target - self.time
        if dt > 1e-12:
            steps = int(round(dt / self.h))
            if steps < 1:
                steps = 1
            step_dt = dt / steps
            for index in range(steps):
                self._step(step_dt, self.time + (index + 1) * step_dt)
            self.time = target
        elif dt < 0.0:
            self.time = target
        self._finalise_budget(max(dt, 0.0))
        return self._frame()

    def _step(self, h: float, t_end: float) -> None:
        x = self.x
        v = self.v

        # external gravity (external force term 0)
        v += _GRAVITY * h
        v *= max(0.0, 1.0 - _DAMPING * h)
        speed = np.linalg.norm(v, axis=1)
        too_fast = speed > _MAX_SPEED
        if np.any(too_fast):
            v[too_fast] *= (_MAX_SPEED / speed[too_fast])[:, None]

        x_prev = x.copy()
        x += v * h

        self.lam_e[:] = 0.0
        self.lam_v[:] = 0.0

        for _ in range(_ITERATIONS):
            self.contact_active[:] = False
            self.contact_n[:] = 0.0
            self.contact_target_v[:] = 0.0
            self._solve_springs(h)
            self._solve_volume(h)
            self._solve_walls(h, t_end)
            self._solve_body_contacts(h)

        new_v = (x - x_prev) / h
        self._apply_contact_velocity(new_v, h)
        speed = np.linalg.norm(new_v, axis=1)
        too_fast = speed > _MAX_SPEED
        if np.any(too_fast):
            new_v[too_fast] *= (_MAX_SPEED / speed[too_fast])[:, None]
        self.v = np.nan_to_num(new_v, nan=0.0, posinf=0.0, neginf=0.0)
        self.x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _solve_springs(self, h: float) -> None:
        x = self.x
        i = self.edge_i
        j = self.edge_j
        delta = x[i] - x[j]
        length = np.linalg.norm(delta, axis=1)
        length = np.maximum(length, 1e-9)
        normal = delta / length[:, None]
        constraint = length - self.edge_rest
        wi = self.w[i]
        wj = self.w[j]
        alpha = self.alpha_spring / (h * h)
        dlam = (-constraint - alpha * self.lam_e) / (wi + wj + alpha)
        np.add.at(x, i, (wi * dlam)[:, None] * normal)
        np.add.at(x, j, -(wj * dlam)[:, None] * normal)
        self.lam_e += dlam

    def _solve_volume(self, h: float) -> None:
        x = self.x
        pa = x[self.fa]
        pb = x[self.fb]
        pc = x[self.fc]
        cross_bc = np.cross(pb, pc)
        volume = np.bincount(
            self.el_body,
            weights=np.einsum("ij,ij->i", pa, cross_bc) / 6.0,
            minlength=self.n_bodies,
        )
        grad = np.zeros_like(x)
        np.add.at(grad, self.fa, cross_bc / 6.0)
        np.add.at(grad, self.fb, np.cross(pc, pa) / 6.0)
        np.add.at(grad, self.fc, np.cross(pa, pb) / 6.0)
        grad_sq = np.bincount(
            self.node_body,
            weights=np.sum((self.w[:, None] * grad) * grad, axis=1),
            minlength=self.n_bodies,
        )
        constraint = volume - self.rest_volume
        alpha = self.alpha_volume / (h * h)
        dlam = (-constraint - alpha * self.lam_v) / (grad_sq + alpha)
        correction = (self.w * dlam[self.node_body])[:, None] * grad
        magnitude = np.linalg.norm(correction, axis=1)
        too_big = magnitude > _MAX_CONTACT_PUSH
        if np.any(too_big):
            correction[too_big] *= (_MAX_CONTACT_PUSH / magnitude[too_big])[:, None]
        x += correction
        self.lam_v += dlam

    def _collide_box(self, box: dict[str, Any], h: float, t_end: float) -> None:
        x = self.x
        rotation = box["rotation"]
        if box["kinematic"]:
            centre = self._plate_translation(t_end)
            target_velocity = self._plate_velocity(t_end)
        else:
            centre = box["centre0"]
            target_velocity = np.zeros(3)
        half = box["half"]
        local = (x - centre) @ rotation
        over = np.abs(local) - half
        outside = np.max(over, axis=1) > 0.0
        target = np.zeros_like(local)

        # particles outside but within the collision skin
        q = np.clip(local, -half, half)
        d = local - q
        dist = np.linalg.norm(d, axis=1)
        near = outside & (dist < _WALL_MARGIN) & (dist > 1e-12)
        if np.any(near):
            n = d[near] / dist[near, None]
            target[near] = local[near] - n * (_WALL_MARGIN - dist[near])[:, None]
            normals = n @ rotation.T
            self.contact_active[near] = True
            self.contact_n[near] = normals
            self.contact_target_v[near] = target_velocity

        # particles whose centre lies inside the (thin) wall volume
        inner = np.flatnonzero(~outside)
        if len(inner):
            penetration = -over[inner]
            axis = np.argmin(penetration, axis=1)
            rows = np.arange(len(inner))
            sign = np.where(local[inner, axis] < 0.0, -1.0, 1.0)
            moved = local[inner].copy()
            moved[rows, axis] = sign * (half[axis] + _WALL_MARGIN)
            target[inner] = moved
            local_normals = np.zeros((len(inner), 3))
            local_normals[rows, axis] = sign
            self.contact_active[inner] = True
            self.contact_n[inner] = local_normals @ rotation.T
            self.contact_target_v[inner] = target_velocity

        delta = (target - local) @ rotation.T
        magnitude = np.linalg.norm(delta, axis=1)
        too_big = magnitude > _MAX_CONTACT_PUSH
        if np.any(too_big):
            scale = (_MAX_CONTACT_PUSH / magnitude[too_big])[:, None]
            delta[too_big] *= scale
        # only "near" and "inner" particles were given a target
        active = near.copy()
        if len(inner):
            active[inner] = True
        if not np.any(active):
            return
        delta[~active] = 0.0
        x += delta
        np.add.at(self.acc_wall, self.node_body, self.mass[:, None] * delta)

    def _solve_walls(self, h: float, t_end: float) -> None:
        for box in self.boxes:
            self._collide_box(box, h, t_end)

    def _solve_body_contacts(self, h: float) -> None:
        x = self.x
        n_vertices = len(_MESH_VERTICES)
        gap = _BODY_GAP
        for a in range(self.n_bodies):
            nodes_a = np.arange(a * n_vertices, (a + 1) * n_vertices)
            for b in range(a + 1, self.n_bodies):
                nodes_b = np.arange(b * n_vertices, (b + 1) * n_vertices)
                diff = x[nodes_a][:, None, :] - x[nodes_b][None, :, :]
                dist = np.linalg.norm(diff, axis=2)
                mask = dist < gap
                if not np.any(mask):
                    continue
                ia, ib = np.nonzero(mask)
                separation = np.maximum(dist[ia, ib], 1e-9)
                normal = diff[ia, ib] / separation[:, None]
                wa = self.w[nodes_a[ia]]
                wb = self.w[nodes_b[ib]]
                dlam = (gap - separation) / (wa + wb)
                push_a = (wa * dlam)[:, None] * normal
                push_b = -(wb * dlam)[:, None] * normal
                magnitude = np.linalg.norm(push_a, axis=1)
                too_big = magnitude > _MAX_CONTACT_PUSH
                if np.any(too_big):
                    scale = (_MAX_CONTACT_PUSH / magnitude[too_big])[:, None]
                    push_a[too_big] *= scale
                    push_b[too_big] *= scale
                np.add.at(x, nodes_a[ia], push_a)
                np.add.at(x, nodes_b[ib], push_b)
                self.acc_pair[a, b] += np.sum(self.mass[nodes_a[ia], None] * push_a, axis=0)
                self.acc_pair[b, a] -= np.sum(self.mass[nodes_a[ia], None] * push_a, axis=0)

    def _apply_contact_velocity(self, velocity: np.ndarray, h: float) -> None:
        active = self.contact_active
        if not np.any(active):
            return
        normals = self.contact_n[active]
        targets = self.contact_target_v[active]
        vel = velocity[active]
        vn = np.sum(vel * normals, axis=1)
        tn = np.sum(targets * normals, axis=1)
        slow = vn < tn
        if np.any(slow):
            vel[slow] += (tn[slow] - vn[slow])[:, None] * normals[slow]
        vn = np.sum(vel * normals, axis=1)
        tangential = vel - vn[:, None] * normals
        target_tangential = targets - np.sum(targets * normals, axis=1)[:, None] * normals
        tangential += _FRICTION * (target_tangential - tangential)
        velocity[active] = tangential + vn[:, None] * normals

    # ------------------------------------------------------------------
    # trajectory export
    # ------------------------------------------------------------------
    def _element_deformation_gradient(self) -> np.ndarray:
        x = self.x
        pa = x[self.fa]
        pb = x[self.fb]
        pc = x[self.fc]
        e1 = pb - pa
        e2 = pc - pa
        normal = np.cross(e1, e2)
        length = np.linalg.norm(normal, axis=1)
        length = np.where(length < 1e-12, 1.0, length)
        normal = normal / length[:, None]
        current = np.stack((e1, e2, normal), axis=2)
        return current @ self.element_pinv

    def _finalise_budget(self, dt_frame: float) -> None:
        dt = dt_frame if dt_frame > 1e-12 else 1.0
        self.acc_force[:] = 0.0
        # term 0: gravity
        self.acc_force[0, 0] = np.sum(self.body_mass) * _GRAVITY
        for index in range(self.n_bodies):
            self.acc_force[1 + index, 0] = self.body_mass[index] * _GRAVITY
        # terms 1..n_bodies: rigid-wall/plate contact on each body
        for index in range(self.n_bodies):
            force = self.acc_wall[index] / dt
            self.acc_force[0, 1 + index] = force
            self.acc_force[1 + index, 1 + index] = force
        # terms 1+n_bodies..: soft body-body contact forces
        for other in range(self.n_bodies):
            term = 1 + self.n_bodies + other
            for index in range(self.n_bodies):
                if index == other:
                    continue
                pair_force = self.acc_pair[index, other] / dt
                self.acc_force[0, term] += pair_force
                self.acc_force[1 + index, term] = pair_force
        self.acc_ext += np.sum(self.acc_force, axis=1) * max(dt_frame, 0.0)
        self.acc_pair_force = self.acc_pair / dt
        self.acc_wall[:] = 0.0
        self.acc_pair[:] = 0.0

    def _frame(self):
        time = float(self.time)
        interaction = np.zeros((self.n_bodies * (self.n_bodies - 1) // 2, 3), dtype=float)
        cursor = 0
        for a in range(self.n_bodies):
            for b in range(a + 1, self.n_bodies):
                interaction[cursor] = self.acc_pair_force[a, b]
                cursor += 1
        frame = StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "reference_position": self.rest.copy(),
            },
            fields={
                "element_deformation_gradient": self._element_deformation_gradient(),
                "interaction_force": interaction,
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": self.acc_force.copy(),
                "budget_cumulative_external_impulse": self.acc_ext.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((self.n_domains, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((self.n_domains, 3)),
                "cumulative_mass_source": np.zeros(self.n_bodies),
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "gravity_z": float(_GRAVITY[2]),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "particles": int(self.n_nodes),
            "bodies": int(self.n_bodies),
            "elements": int(len(self.fa)),
            "substep": float(_SUBSTEP),
            "final_time": float(self.time),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))),
        }
