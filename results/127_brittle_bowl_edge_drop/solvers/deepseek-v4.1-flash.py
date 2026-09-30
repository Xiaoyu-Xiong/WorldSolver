"""Brittle bowl edge drop solver (scene 127_brittle_bowl_edge_drop).

Physical model
--------------
The thin brittle bowl is discretized as a Lagrangian shell: every mesh vertex of
the frozen topology asset is a material sample with mass ``density * spacing**3``
(the same convention the frozen scene uses) and every mesh edge is an axial
spring with a brittle failure elongation.  The discretization is stepped with a
symplectic (semi-implicit) Euler integrator on a fixed micro time step.

Forces
  * gravity ``m * g`` (external),
  * axial spring forces on the edge network with viscous damping
    (internal -> not exported as external force),
  * brittle fracture: an edge whose elongation exceeds its (deterministically
    randomized, impact-site weakened) failure elongation is deleted, so crack
    networks grow from the impact point and release fragments.  A compressive
    failure limit is applied as well.
  * penalty floor contact with Coulomb friction and viscous normal damping
    against the frozen rigid ground box (external),
  * penalty repulsion between samples joined by already-broken edges so released
    fragments keep colliding (internal).

Every requested sample time is produced by integrating the state forward with
that fixed micro step, so states are deterministic and finite at every sample.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_WORKSPACE = Path(__file__).resolve().parents[1]
_FALLBACK_TOPOLOGY = "assets/bowl_topology.json"

# frozen-ish numerical parameters of the discretization
_MICRO_DT = 7.0e-5          # s, micro time step (stability limited by k/m)
_BOND_STIFFNESS = 5.0e6     # N/m, axial edge stiffness (uniform, thin shell)
_BOND_ZETA = 0.08           # axial dashpot ratio
_CONTACT_STIFFNESS = 1.0e7  # N/m penalty against the rigid floor
_CONTACT_ZETA = 0.20
_FRICTION = 0.45
_TANGENT_STIFFNESS = 2.0e6  # N/m regularized Coulomb tangency
_FRAGMENT_STIFFNESS = 5.0e6
_FRAGMENT_ZETA = 0.15
_ENV_DAMPING = 0.5          # 1/s, mild bulk dissipation (settling)
_MAX_SPEED = 25.0
_MAX_FLOOR_FORCE = 1.0e6
_FAIL_ELONGATION = 1.0e-3   # m, brittle failure elongation scale
_COMPRESSION_FACTOR = 2.5
_IMPACT_REF_DIST = 0.22     # m, radius of the impact-damaged zone


def _workspace_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for root in (_WORKSPACE, Path.cwd()):
        resolved = root / candidate
        if resolved.is_file():
            return resolved
    return _WORKSPACE / candidate


class BenchmarkSolver:
    """Deterministic explicit brittle-fracture shell solver."""

    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", None) or {})
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed * 7919 + 12127)

        self.gravity = np.asarray(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        ).reshape(3)
        sampling = config.get("sampling") or {}
        self.spacing = float(sampling.get("spacing", 0.14))
        self.density = float(sampling.get("density", 1350.0))
        self.sample_dt = float(config.get("sample_dt", 1.0 / 30.0))

        frame0 = scene.initial_state()
        self.x0 = np.array(frame0.position, dtype=float)
        self.x = self.x0.copy()
        self.v = np.zeros_like(self.x)
        mass = frame0.mass
        if mass is None:
            mass = np.full(len(self.x), self.density * self.spacing ** 3)
        self.m = np.maximum(np.asarray(mass, dtype=float).reshape(-1), 1.0e-9)
        self.m_inv = 1.0 / self.m
        self.n = int(len(self.x))
        self.mean_mass = float(np.mean(self.m))
        self.body_id = np.zeros(self.n, dtype=np.int32)
        radii = frame0.attributes.get("particle_radius")
        if radii is None:
            radii = np.full(self.n, 0.45 * self.spacing)
        self.radius = np.asarray(radii, dtype=float).reshape(-1)

        # ---- topology: frozen mesh edges become brittle axial bonds ----
        self.faces = self._load_faces(config)
        tri = self.faces.astype(np.intp)
        pairs = np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]), axis=0)
        pairs = np.sort(pairs, axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        pairs = np.unique(pairs, axis=0)
        self.ei = np.ascontiguousarray(pairs[:, 0])
        self.ej = np.ascontiguousarray(pairs[:, 1])
        self.L0 = np.maximum(
            np.linalg.norm(self.x0[self.ej] - self.x0[self.ei], axis=1), 1.0e-6
        )
        self.n_bonds = int(len(self.ei))
        self.active = np.ones(self.n_bonds, dtype=bool)
        self.broken = np.zeros(0, dtype=np.intp)
        self.broken_total = 0

        # brittle strength: deterministic scatter, weakened near the first
        # contact point (stress concentration of the edge impact).
        self.contact_point = self.x0[int(np.argmin(self.x0[:, 2]))].copy()
        midpoint = 0.5 * (self.x0[self.ei] + self.x0[self.ej])
        dist = np.linalg.norm(midpoint - self.contact_point[None, :], axis=1)
        proximity = np.clip(dist / _IMPACT_REF_DIST, 0.0, 1.0)
        scatter = 0.75 + 0.45 * rng.random(self.n_bonds)
        self.fail_elong = _FAIL_ELONGATION * (0.55 + 0.85 * proximity) * scatter
        # crack-tip stress concentration: edges adjacent to a broken edge lose
        # strength, so a crack network propagates away from the impact.
        self.fail_floor = 0.5 * self.fail_elong
        self._build_incidence()

        self.k_bond = _BOND_STIFFNESS
        self.c_bond = 2.0 * _BOND_ZETA * float(np.sqrt(_BOND_STIFFNESS * self.mean_mass))
        self.k_contact = _CONTACT_STIFFNESS
        self.c_contact = 2.0 * _CONTACT_ZETA * float(
            np.sqrt(_CONTACT_STIFFNESS * self.mean_mass)
        )
        self.k_tangent = _TANGENT_STIFFNESS
        self.mu = _FRICTION
        self.k_frag = _FRAGMENT_STIFFNESS
        self.c_frag = 2.0 * _FRAGMENT_ZETA * float(
            np.sqrt(_FRAGMENT_STIFFNESS * self.mean_mass)
        )

        # ---- frozen rigid floor from the scene geometry ----
        half = np.array([3.2, 1.4, 0.05], dtype=float)
        translation = np.array([0.0, 0.0, -0.05], dtype=float)
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []) or []:
            if obj.get("id") != "ground":
                continue
            spec = obj.get("geometry") or {}
            if spec.get("half_extents"):
                half = np.asarray(spec["half_extents"], dtype=float).reshape(3)
            transform = obj.get("transform") or {}
            if transform.get("translation"):
                translation = np.asarray(transform["translation"], dtype=float).reshape(3)
        self.floor_z = float(translation[2] + half[2])
        self.floor_center = translation[:2].copy()
        self.floor_extent = np.abs(half[:2]) + self.spacing

        # ---- deformation-gradient operator on the frozen reference mesh ----
        self._build_reference_operators()

        # ---- fixed micro time step aligned with the sample cadence ----
        self.substeps = max(1, int(np.ceil(self.sample_dt / _MICRO_DT)))
        self.dt = self.sample_dt / self.substeps
        self.steps = 0
        self.target_steps = 0

        extent = np.ptp(self.x0, axis=0)
        self.char_length = float(max(np.max(extent), self.spacing))
        self.last_gap = None

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _load_faces(self, config) -> np.ndarray:
        candidates: list[str] = []
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []) or []:
            spec = obj.get("geometry") or {}
            if obj.get("state_binding") is None and not spec.get("topology_asset"):
                continue
            for key in ("topology_asset", "asset"):
                value = spec.get(key)
                if value:
                    candidates.append(str(value))
        candidates.append(_FALLBACK_TOPOLOGY)
        for value in candidates:
            path = _workspace_path(value)
            if path.is_file() and path.suffix.casefold() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and "faces" in payload:
                    faces = np.asarray(payload["faces"], dtype=np.intp)
                    if faces.ndim == 2 and faces.shape[1] == 3:
                        return faces
        raise RuntimeError("Could not locate the frozen bowl topology asset")

    def _build_reference_operators(self) -> None:
        ref = self.x0
        faces = self.faces.astype(np.intp)
        pos = ref[faces]
        e01 = pos[:, 1] - pos[:, 0]
        e02 = pos[:, 2] - pos[:, 0]
        cross = np.cross(e01, e02)
        area2 = np.linalg.norm(cross, axis=1)
        valid = area2 > 1.0e-14
        faces = faces[valid]
        e01 = e01[valid]
        e02 = e02[valid]
        cross = cross[valid]
        area2 = area2[valid]
        scale = np.sqrt(area2)
        normal = cross / area2[:, None] * scale[:, None]
        material = np.stack((e01, e02, normal), axis=2)
        self.faces_valid = faces
        self.rest_area = 0.5 * area2
        self.dm_inv = np.linalg.inv(material)
        self.face_normals = cross / area2[:, None]

    def _build_incidence(self) -> None:
        """CSR map: sample -> incident bond indices (for crack propagation)."""
        nodes = np.concatenate((self.ei, self.ej))
        bonds = np.concatenate((np.arange(self.n_bonds), np.arange(self.n_bonds)))
        order = np.argsort(nodes, kind="stable")
        self.node_bonds = bonds[order].astype(np.intp)
        sorted_nodes = nodes[order]
        self.node_start = np.searchsorted(sorted_nodes, np.arange(self.n), side="left")
        self.node_end = np.searchsorted(sorted_nodes, np.arange(self.n), side="right")

    def _weaken_crack_tip(self, broken_mask: np.ndarray) -> None:
        """Reduce the strength of edges bounding a fresh crack surface."""
        fresh = np.flatnonzero(broken_mask)
        if fresh.size == 0:
            return
        nodes = np.unique(
            np.concatenate((self.ei[fresh], self.ej[fresh]))
        )
        starts = self.node_start[nodes]
        counts = self.node_end[nodes] - starts
        pieces = [
            self.node_bonds[start : start + count]
            for start, count in zip(starts, counts)
            if count > 0
        ]
        if not pieces:
            return
        affected = np.unique(np.concatenate(pieces))
        self.fail_elong[affected] = np.maximum(
            0.85 * self.fail_elong[affected], self.fail_floor[affected]
        )

    # ------------------------------------------------------------------
    # forces
    # ------------------------------------------------------------------
    def _floor_contact(self, x: np.ndarray, v: np.ndarray):
        force = np.zeros_like(x)
        gap = x[:, 2] - self.floor_z
        inside = (
            (np.abs(x[:, 0] - self.floor_center[0]) <= self.floor_extent[0])
            & (np.abs(x[:, 1] - self.floor_center[1]) <= self.floor_extent[1])
        )
        touching = (gap < 0.0) & inside
        if np.any(touching):
            idx = np.flatnonzero(touching)
            penetration = -gap[idx]
            normal = self.k_contact * penetration - self.c_contact * np.minimum(
                v[idx, 2], 0.0
            )
            normal = np.clip(normal, 0.0, _MAX_FLOOR_FORCE)
            force[idx, 2] += normal
            tangent = v[idx].copy()
            tangent[:, 2] = 0.0
            speed = np.linalg.norm(tangent, axis=1)
            magnitude = np.minimum(self.mu * normal, self.k_tangent * speed)
            denom = np.maximum(speed, 1.0e-9)
            force[idx, :2] -= (magnitude / denom)[:, None] * tangent[:, :2]
        return force, gap

    def _bond_forces(self, x: np.ndarray, v: np.ndarray, fracture: bool) -> np.ndarray:
        ei, ej = self.ei, self.ej
        delta = x[ej] - x[ei]
        length = np.linalg.norm(delta, axis=1)
        safe = np.maximum(length, 1.0e-12)
        direction = delta / safe[:, None]
        stretch = length - self.L0
        approach = np.einsum("ij,ij->i", v[ej] - v[ei], direction)

        active = self.active
        if fracture:
            broken = active & (
                (stretch > self.fail_elong)
                | (stretch < -_COMPRESSION_FACTOR * self.fail_elong)
            )
            if np.any(broken):
                self.active = active & ~broken
                newly = np.flatnonzero(broken).astype(np.intp)
                self.broken = np.concatenate((self.broken, newly))
                self.broken_total += int(newly.size)
                self._weaken_crack_tip(broken)
                active = self.active
        else:
            broken = None

        magnitude = self.k_bond * stretch - self.c_bond * approach
        magnitude = np.where(active, magnitude, 0.0)
        limit = 4.0 * np.max(self.fail_elong) * self.k_bond
        magnitude = np.clip(magnitude, -limit, limit)
        if broken is not None and np.any(broken):
            magnitude = np.where(broken, 0.0, magnitude)

        pair_force = magnitude[:, None] * direction
        out = np.zeros_like(x)
        for axis in range(3):
            out[:, axis] = np.bincount(
                ei, weights=pair_force[:, axis], minlength=self.n
            )
            out[:, axis] -= np.bincount(
                ej, weights=pair_force[:, axis], minlength=self.n
            )
        return out

    def _fragment_forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        if self.broken.size == 0:
            return out
        ei = self.ei[self.broken]
        ej = self.ej[self.broken]
        delta = x[ej] - x[ei]
        length = np.linalg.norm(delta, axis=1)
        overlap = (self.radius[ei] + self.radius[ej]) - length
        hit = overlap > 0.0
        if not np.any(hit):
            return out
        idx = np.flatnonzero(hit)
        i, j = ei[idx], ej[idx]
        dist = np.maximum(length[idx], 1.0e-12)
        direction = delta[idx] / dist[:, None]
        approach = np.einsum("ij,ij->i", v[j] - v[i], direction)
        normal = np.clip(
            np.maximum(overlap[idx], 0.0) * self.k_frag - self.c_frag * np.minimum(approach, 0.0),
            0.0,
            _MAX_FLOOR_FORCE,
        )
        # repulsion pushes i away from j (along -direction) and j along +direction
        pair_force = normal[:, None] * direction
        for axis in range(3):
            out[:, axis] -= np.bincount(i, weights=pair_force[:, axis], minlength=self.n)
            out[:, axis] += np.bincount(j, weights=pair_force[:, axis], minlength=self.n)
        return out

    # ------------------------------------------------------------------
    # integration
    # ------------------------------------------------------------------
    def _step(self) -> None:
        x, v = self.x, self.v
        force = self.m[:, None] * self.gravity[None, :]
        contact, _ = self._floor_contact(x, v)
        force += contact
        force += self._bond_forces(x, v, fracture=True)
        force += self._fragment_forces(x, v)
        force -= (self.m * _ENV_DAMPING)[:, None] * v
        np.nan_to_num(force, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        v += (force * self.m_inv[:, None]) * self.dt
        np.clip(v, -_MAX_SPEED, _MAX_SPEED, out=v)
        x += v * self.dt
        np.clip(x[:, 2], -1.0, 100.0, out=x[:, 2])
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        self.steps += 1

    def advance_to(self, time: float):
        time = float(time)
        target = int(round(time / self.dt))
        if target > self.steps:
            for _ in range(target - self.steps):
                self._step()
        self.steps = max(self.steps, target)
        return self._frame(time)

    # ------------------------------------------------------------------
    # diagnostics / export
    # ------------------------------------------------------------------
    def _deformation_gradient(self, x: np.ndarray) -> np.ndarray:
        faces = self.faces_valid
        current = x[faces]
        e01 = current[:, 1] - current[:, 0]
        e02 = current[:, 2] - current[:, 0]
        cross = np.cross(e01, e02)
        area2 = np.linalg.norm(cross, axis=1)
        safe = np.maximum(area2, 1.0e-14)
        scale = np.sqrt(safe)
        normal = cross / safe[:, None] * scale[:, None]
        deformed = np.stack((e01, e02, normal), axis=2)
        gradient = deformed @ self.dm_inv

        weight = self.rest_area
        accum = np.zeros((self.n, 3, 3))
        total = np.zeros(self.n)
        for corner in range(3):
            index = faces[:, corner]
            np.add.at(accum, index, weight[:, None, None] * gradient)
            np.add.at(total, index, weight)
        good = total > 1.0e-12
        accum[good] /= total[good][:, None, None]
        accum[~good] = np.eye(3)
        return accum

    def _frame(self, time: float) -> StateFrame:
        contact, gap = self._floor_contact(self.x, self.v)
        external = self.m[:, None] * self.gravity[None, :] + contact
        frame = StateFrame(
            time,
            self.x.copy(),
            self.v.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.m.copy(),
                "deformation_gradient": self._deformation_gradient(self.x),
                "particle_radius": self.radius.copy(),
                "material_label": self.body_id.copy(),
            },
            fields={
                "contact_signed_distance": gap,
                "cumulative_mass_source": np.zeros(1),
                "external_force": external,
            },
            scalars={
                "contact_length_scale": self.spacing,
                "grid_spacing": self.spacing,
                "characteristic_length": self.char_length,
                "reference_velocity": 1.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "seed": int(self.seed),
            "samples": int(self.n),
            "bonds": int(self.n_bonds),
            "broken_bonds": int(self.broken_total),
            "micro_dt": float(self.dt),
            "substeps_per_sample": int(self.substeps),
            "steps": int(self.steps),
            "min_height": float(np.min(self.x[:, 2])),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))),
        }
