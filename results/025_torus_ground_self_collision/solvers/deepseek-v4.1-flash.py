"""Deterministic physically-based solver for the torus ground/self-collision scene.

Physical model
--------------
The deformable torus is discretised by its frozen triangle mesh and simulated as
a mass-spring shell:

* every mesh edge is a structural (membrane) spring acting in tension and
  compression,
* every interior mesh edge carries a hinge spring joining the two opposite
  vertices of its adjacent triangles (discrete bending resistance),
* a one-sided internal-pressure constraint keeps each tube cross-section from
  collapsing below a fraction of the frozen minor radius,
* the rigid ground box is a non-penetration boundary with Coulomb friction and
  restitution,
* self-collision is a vertex-vertex non-penetration constraint between surface
  samples that are far apart along the surface (not sharing a mesh face).

Time integration is semi-implicit (symplectic) Euler with explicit elastic
forces.  Contact, internal-pressure and self-collision constraints are resolved
by position projection (position-based/XPBD style), which is stable for stiff
unilateral contacts.  All operations are deterministic and vectorised; no
randomness, keyframes or precomputed motion tables are used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_EPS = 1e-12
_MAX_SPEED = 40.0
_INTERNAL_PRESSURE_FRACTION = 0.55
_BENDING_STIFFNESS_RATIO = 0.5
_SELF_CONTACT_REPORT_SCALE = 2.0
_SELF_COLLISION_RELAXATION = 0.7


def _norm(v: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("ij,ij->i", v, v))


def _accumulate(index: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    """Deterministic scatter-add of ``values[index]`` into a [count,3] array."""
    out = np.empty((count, 3), dtype=float)
    for axis in range(3):
        out[:, axis] = np.bincount(index, weights=values[:, axis], minlength=count)
    return out


class BenchmarkSolver:
    """Mass-spring / position-based solver honouring the frozen scene bindings."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        self.config = cfg

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.duration = float(cfg.get("duration", 1.0))
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 60.0))
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.stiffness = max(float(cfg.get("stiffness", 1.0)), _EPS)
        self.damping = max(float(cfg.get("damping", 0.0)), 0.0)
        self.restitution = float(np.clip(cfg.get("restitution", 0.0), 0.0, 1.0))
        self.friction = max(float(cfg.get("friction", 0.0)), 0.0)
        self.contact_length_scale = float(cfg.get("contact_length_scale", 0.01))
        if not np.isfinite(self.contact_length_scale) or self.contact_length_scale <= 0.0:
            self.contact_length_scale = 0.01
        self.major_radius = float(cfg.get("major_radius", 0.0))
        self.minor_radius = float(cfg.get("minor_radius", 0.0))

        # Frozen initial state (contract-guaranteed), used verbatim at t = 0.
        initial = scene.initial_state()
        rest = np.asarray(initial.position, dtype=float).copy()
        velocity0 = np.asarray(initial.velocity, dtype=float).copy()
        mass = np.asarray(initial.mass, dtype=float).copy()
        if mass.shape != (len(rest),) or not np.isfinite(mass).all() or np.any(mass <= 0.0):
            total = float(cfg.get("subject_mass", 1.0))
            mass = np.full(len(rest), total / len(rest), dtype=float)
        self.rest = rest
        self.n = int(len(rest))
        self.mass = mass
        self.inv_mass = 1.0 / mass

        faces = self._load_faces(cfg)
        self.faces = faces

        # Membrane springs: unique mesh edges.
        triples = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        self.edges = np.unique(np.sort(triples, axis=1), axis=0)
        self.edge_rest = _norm(rest[self.edges[:, 0]] - rest[self.edges[:, 1]])

        # Hinge (bending) springs: opposite vertices of the two faces of an edge.
        opposite: dict[tuple[int, int], list[int]] = {}
        for face in faces:
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            for key, opp in (((a, b), c), ((b, c), a), ((c, a), b)):
                if key[0] > key[1]:
                    key = (key[1], key[0])
                opposite.setdefault(key, []).append(opp)
        hinges = [pair for pair in opposite.values() if len(pair) == 2]
        self.hinges = (
            np.asarray(hinges, dtype=np.int64)
            if hinges
            else np.zeros((0, 2), dtype=np.int64)
        )
        self.hinge_rest = (
            _norm(rest[self.hinges[:, 0]] - rest[self.hinges[:, 1]])
            if len(self.hinges)
            else np.zeros(0)
        )
        self.k_edge = self.stiffness
        self.k_hinge = _BENDING_STIFFNESS_RATIO * self.stiffness

        # Topological exclusions for self-collision (samples sharing a face).
        shared = np.zeros((self.n, self.n), dtype=bool)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            shared[faces[:, a], faces[:, b]] = True
            shared[faces[:, b], faces[:, a]] = True
        self.shared_face = shared

        # Tube cross-section bookkeeping: bin samples by angle about the ring.
        self._build_sections(rest)

        # Per-sample deformation-gradient operator at rest.
        self._build_deformation_operator(rest)

        # Integration constants.
        target_dt = 5.0e-4
        self.substeps = max(8, int(round(self.sample_dt / target_dt)))
        self.dt_sub = self.sample_dt / float(self.substeps)
        self.vel_damp = 1.0 / (1.0 + self.damping * self.dt_sub)
        self.rest_threshold = 0.05
        self.self_contact_distance = max(self.contact_length_scale, 1e-6)
        self.report_distance = _SELF_CONTACT_REPORT_SCALE * self.contact_length_scale

        self.x = rest.copy()
        self.v = velocity0.reshape(self.n, 3).copy()
        self.time = 0.0
        self.steps = 0
        self.frames_emitted = 0

    def _load_faces(self, cfg: dict) -> np.ndarray:
        asset = cfg.get("subject_asset")
        candidates: list[Path] = []
        if asset:
            path = Path(str(asset))
            if path.is_absolute():
                candidates.append(path)
            else:
                candidates.append(Path.cwd() / path)
                candidates.append(Path(__file__).resolve().parents[1] / path)
        candidates.append(Path.cwd() / "assets" / "subject.npz")
        for candidate in candidates:
            if candidate.is_file():
                with np.load(candidate) as data:
                    if "faces" in data:
                        return np.asarray(data["faces"], dtype=np.int64).reshape(-1, 3)
        raise RuntimeError("subject topology asset with faces is unavailable")

    def _build_sections(self, rest: np.ndarray) -> None:
        """Group samples into tube cross-sections using the ring-plane angle."""
        centred = rest - rest.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        axis_u, axis_v = vt[0], vt[1]
        angle = np.arctan2(centred @ axis_v, centred @ axis_u)
        keys = np.round(angle, 5)
        unique_keys = np.unique(keys)
        nbins = int(len(unique_keys))
        if nbins < 4 or nbins > max(4, self.n // 4):
            nbins = 48
            key_index = np.floor((angle + np.pi) / (2.0 * np.pi) * nbins).astype(np.int64)
            key_index = np.clip(key_index, 0, nbins - 1)
        else:
            key_index = np.searchsorted(unique_keys, keys)
        counts = np.bincount(key_index, minlength=nbins).astype(float)
        self.bin_index = key_index
        self.bin_counts = counts
        self.bin_valid = counts >= 3.0
        self.n_bins = nbins
        if self.minor_radius > 0.0:
            radius = self.minor_radius
        else:
            centres = np.zeros((nbins, 3))
            for axis in range(3):
                centres[:, axis] = np.bincount(
                    key_index, weights=rest[:, axis], minlength=nbins
                ) / np.maximum(counts, 1.0)
            radius = float(np.median(_norm(rest - centres[key_index])))
        self.section_radius_min = _INTERNAL_PRESSURE_FRACTION * max(radius, 1e-6)

    def _build_deformation_operator(self, rest: np.ndarray) -> None:
        edges = self.edges
        delta = rest[edges[:, 0]] - rest[edges[:, 1]]
        outer = delta[:, :, None] * delta[:, None, :]
        n = self.n
        operator = np.zeros((n, 3, 3))
        for axis_i in range(3):
            for axis_j in range(3):
                operator[:, axis_i, axis_j] = np.bincount(
                    edges[:, 0], weights=outer[:, axis_i, axis_j], minlength=n
                ) + np.bincount(edges[:, 1], weights=outer[:, axis_i, axis_j], minlength=n)
        trace = np.trace(operator, axis1=1, axis2=2) / 3.0
        scale = np.maximum(np.abs(trace), _EPS)
        regularised = operator + (1e-6 * scale)[:, None, None] * np.eye(3)[None, :, :]
        try:
            self.deformation_operator = np.linalg.inv(regularised)
        except np.linalg.LinAlgError:
            self.deformation_operator = np.linalg.pinv(regularised)

    # ---------------------------------------------------------------- physics
    def _elastic_force(self, x: np.ndarray) -> np.ndarray:
        force = np.zeros((self.n, 3), dtype=float)
        if len(self.edges):
            force += self._spring_force(x, self.edges, self.edge_rest, self.k_edge)
        if len(self.hinges):
            force += self._spring_force(x, self.hinges, self.hinge_rest, self.k_hinge)
        return force

    def _spring_force(
        self, x: np.ndarray, index: np.ndarray, rest: np.ndarray, stiffness: float
    ) -> np.ndarray:
        delta = x[index[:, 0]] - x[index[:, 1]]
        raw = _norm(delta)
        direction = delta / np.maximum(raw, _EPS)[:, None]
        # bounded strain keeps the explicit spring force finite in any pose
        strain = np.clip((raw - rest) / np.maximum(rest, _EPS), -1.0, 1.0)
        value = (-stiffness * strain)[:, None] * direction
        return _accumulate(index[:, 0], value, self.n) - _accumulate(index[:, 1], value, self.n)

    def _project_sections(self, x: np.ndarray) -> None:
        counts = self.bin_counts
        centres = np.zeros((self.n_bins, 3), dtype=float)
        for axis in range(3):
            centres[:, axis] = np.bincount(
                self.bin_index, weights=x[:, axis], minlength=self.n_bins
            ) / np.maximum(counts, 1.0)
        local = centres[self.bin_index]
        delta = x - local
        distance = _norm(delta) + _EPS
        direction = delta / distance[:, None]
        active = (distance < self.section_radius_min) & self.bin_valid[self.bin_index]
        if np.any(active):
            correction = np.where(active, self.section_radius_min - distance, 0.0)
            x += correction[:, None] * direction

    def _project_self_collision(self, x: np.ndarray, pairs: np.ndarray | None = None) -> None:
        distance_limit = self.self_contact_distance
        if pairs is None:
            tree = cKDTree(x)
            try:
                pairs = tree.query_pairs(distance_limit, output_type="ndarray")
            except TypeError:  # pragma: no cover - older scipy
                pairs = np.asarray(sorted(tree.query_pairs(distance_limit)), dtype=np.int64)
        if len(pairs) == 0:
            return
        keep = ~self.shared_face[pairs[:, 0], pairs[:, 1]]
        pairs = pairs[keep]
        if len(pairs) == 0:
            return
        delta = x[pairs[:, 0]] - x[pairs[:, 1]]
        distance = _norm(delta)
        active = distance < distance_limit
        if not np.any(active):
            return
        pairs = pairs[active]
        distance = distance[active]
        direction = delta[active] / np.maximum(distance, _EPS)[:, None]
        # each member of a pair separates by half of the remaining violation
        push = (0.5 * (distance_limit - distance))[:, None] * direction
        correction = np.zeros((self.n, 3), dtype=float)
        for axis in range(3):
            correction[:, axis] = np.bincount(
                pairs[:, 0], weights=push[:, axis], minlength=self.n
            ) - np.bincount(pairs[:, 1], weights=push[:, axis], minlength=self.n)
        x += _SELF_COLLISION_RELAXATION * correction

    def _project_ground(self, x: np.ndarray) -> None:
        below = x[:, 2] < self.ground_z
        if np.any(below):
            x[below, 2] = self.ground_z

    def _substep(self) -> None:
        dt = self.dt_sub
        x, v = self.x, self.v
        backup_x, backup_v = x.copy(), v.copy()

        force = self.mass[:, None] * self.gravity[None, :]
        force += self._elastic_force(x)
        v = v + force * self.inv_mass[:, None] * dt
        v *= self.vel_damp
        speed = _norm(v)
        fast = speed > _MAX_SPEED
        if np.any(fast):
            v[fast] *= (_MAX_SPEED / speed[fast])[:, None]

        predicted = v.copy()
        x_prev = x.copy()
        x = x + v * dt

        tree = cKDTree(x)
        try:
            pairs = tree.query_pairs(self.self_contact_distance, output_type="ndarray")
        except TypeError:  # pragma: no cover - older scipy
            pairs = np.asarray(sorted(tree.query_pairs(self.self_contact_distance)), dtype=np.int64)
        for _ in range(2):
            self._project_sections(x)
            self._project_self_collision(x, pairs=pairs)
            self._project_ground(x)

        velocity = (x - x_prev) / dt
        contact = x[:, 2] <= self.ground_z + 1e-12
        if np.any(contact):
            normal_before = predicted[contact, 2]
            approaching = normal_before < 0.0
            normal_after = velocity[contact, 2]
            bounce = approaching & (normal_before < -self.rest_threshold)
            normal_new = np.where(
                bounce,
                -self.restitution * normal_before,
                np.where(approaching, 0.0, normal_after),
            )
            velocity[contact, 2] = normal_new
            tangential = velocity[contact, :2]
            tangential_speed = _norm(tangential) + _EPS
            normal_change = np.abs(normal_new - normal_before)
            scale = np.clip(1.0 - self.friction * normal_change / tangential_speed, 0.0, 1.0)
            velocity[contact, :2] = tangential * scale[:, None]

        if not (np.isfinite(x).all() and np.isfinite(velocity).all()):
            x, velocity = backup_x, backup_v

        self.x, self.v = x, velocity

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt_sub))
        if target > self.steps:
            for _ in range(target - self.steps):
                self._substep()
            self.steps = target
        self.time = float(time)
        self.frames_emitted += 1
        return self._frame(float(time))

    def _deformation_gradient(self) -> np.ndarray:
        edges = self.edges
        delta = self.x[edges[:, 0]] - self.x[edges[:, 1]]
        rest_delta = self.rest[edges[:, 0]] - self.rest[edges[:, 1]]
        outer = delta[:, :, None] * rest_delta[:, None, :]
        accumulator = np.zeros((self.n, 3, 3), dtype=float)
        for axis_i in range(3):
            for axis_j in range(3):
                accumulator[:, axis_i, axis_j] = np.bincount(
                    edges[:, 0], weights=outer[:, axis_i, axis_j], minlength=self.n
                ) + np.bincount(
                    edges[:, 1], weights=outer[:, axis_i, axis_j], minlength=self.n
                )
        gradient = accumulator @ self.deformation_operator
        if not np.isfinite(gradient).all():
            gradient = np.tile(np.eye(3), (self.n, 1, 1))
        return gradient

    def _contact_field(self) -> np.ndarray:
        ground_gap = self.x[:, 2] - self.ground_z
        near = ground_gap < self.report_distance
        samples = [ground_gap[near]]
        tree = cKDTree(self.x)
        try:
            pairs = tree.query_pairs(self.report_distance, output_type="ndarray")
        except TypeError:  # pragma: no cover - older scipy
            pairs = np.asarray(sorted(tree.query_pairs(self.report_distance)), dtype=np.int64)
        if len(pairs):
            pairs = pairs[~self.shared_face[pairs[:, 0], pairs[:, 1]]]
        if len(pairs):
            gap = _norm(self.x[pairs[:, 0]] - self.x[pairs[:, 1]])
            samples.append(gap)
        contact = np.concatenate(samples) if samples else np.zeros(0)
        if contact.size == 0:
            contact = np.asarray([float(np.min(ground_gap))])
        return contact

    def _frame(self, time: float) -> StateFrame:
        gradient = self._deformation_gradient()
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": gradient,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "contact_signed_distance": self._contact_field(),
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "samples": int(self.n),
            "substeps": int(self.substeps),
            "substep_dt": float(self.dt_sub),
            "frames": int(self.frames_emitted),
            "self_contact_distance": float(self.self_contact_distance),
            "section_radius_min": float(self.section_radius_min),
            "stiffness": float(self.stiffness),
        }
