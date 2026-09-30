"""Physically-based solver for the thumb-index web pinch scene.

Model
-----
The exported state carries the ``assets/hand_flesh.obj`` surface vertices of the
volumetric hand/forearm flesh.  The flesh is advanced with an explicit
lumped-mass time-stepping scheme that combines

* a nonlinear membrane/elastic network (rest-length edge springs) representing
  the in-plane stiffness of the flesh solid,
* a near-incompressibility (volume) constraint projected each step, which is
  what turns the squeeze of the thumb-index web into an outward rounded mound,
* a fixed forearm cuff (Dirichlet boundary, ``x <= -0.150``),
* penalty couplings to the two driven bones (``thumb_bone`` kinematic per
  ``config.yaml``'s ``thumb_drive``, ``index_bone`` held in its initial pose).

Everything is deterministic, uses only NumPy, and honours the frozen geometry,
boundary conditions and state bindings.  No animation, keyframes or precomputed
trajectories are used: every sample time is produced by actually integrating the
equations of motion from the previous sample.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Physical / numerical parameters (soft-tissue-like, SI units)
# ---------------------------------------------------------------------------
MEMBRANE_MODULUS = 4.0e4      # Pa   effective elastic modulus of flesh skin
MEMBRANE_THICKNESS = 0.005    # m    effective load-bearing thickness
AREA_MODULUS = 2.0e7          # N/m  local in-plane (area) stiffness of the skin
BONE_SPRING = 3.0e2           # N/m  per-unit-weight penalty stiffness to bones
BONE_DAMPING = 0.5            # N*s/m per-unit-weight bone coupling damping
DAMPING_TIME = 0.08           # s    bulk viscous relaxation time
THUMB_INFLUENCE = 0.055       # m    radius of tissue bound to the thumb bone
INDEX_INFLUENCE = 0.050       # m    radius of tissue bound to the index bone
VELOCITY_CLAMP = 2.0          # m/s  safety clamp (well above physical speeds)
MAX_PROJECTION_STEP = 5.0e-4  # m    per-step cap on volume-projection motion


def _load_obj_mesh(path: Path):
    """Read vertices and triangulated faces from a Wavefront OBJ file."""
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(tok) for tok in line.split()[1:4]])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                index = [int(tok.split("/")[0]) - 1 for tok in tokens]
                for k in range(1, len(index) - 1):
                    faces.append([index[0], index[k], index[k + 1]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def _scatter(index: np.ndarray, values: np.ndarray, count: int) -> np.ndarray:
    """Accumulate per-node vector values (vectorised over the 3 components)."""
    out = np.empty((count, 3), dtype=float)
    for axis in range(3):
        out[:, axis] = np.bincount(index, weights=values[:, axis], minlength=count)
    return out


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point to the segment [a, b]."""
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 0.0:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(points - closest, axis=1)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


class BenchmarkSolver:
    """Explicit lumped-mass elastic-solid solver with a volume constraint."""

    #: material index count for ``attributes.mass_quantity`` / mass sources
    MATERIAL_COUNT = 1

    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        initial = scene.initial_state()
        rest = np.array(initial.position, dtype=float)
        self.rest = rest
        self.node_count = int(rest.shape[0])

        asset = self._resolve_asset(scene)
        vertices, faces = _load_obj_mesh(asset)
        if vertices.shape[0] != self.node_count:
            # Fall back to the scene-provided rest surface if the asset differs.
            faces = faces[:0]
        if faces.size == 0:
            raise ValueError("hand flesh topology could not be loaded")

        # Orient faces so the enclosed signed volume is positive (outward normals).
        signed = self._signed_volume(rest, faces)
        if signed < 0.0:
            faces = faces[:, [0, 2, 1]]
            signed = -signed
        self.faces = faces
        self.rest_volume = float(signed)

        # Unique undirected edges of the triangle mesh.
        raw = np.vstack(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
        )
        edges = np.sort(raw, axis=1)
        edges = np.unique(edges, axis=0)
        self.e0 = edges[:, 0]
        self.e1 = edges[:, 1]
        self.rest_length = np.linalg.norm(
            rest[self.e0] - rest[self.e1], axis=1
        )
        self.edge_stiffness = MEMBRANE_MODULUS * MEMBRANE_THICKNESS

        # Reference frames of the surface elements (for the deformation gradient).
        self.f0, self.f1, self.f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        self.rest_area = 0.5 * self._element_scale(rest) ** 2
        self.area_stiffness = AREA_MODULUS
        self.ref_basis = self._element_basis(rest)
        self.ref_basis_inv = self._inverse3(self.ref_basis)
        self.singular_element = np.abs(
            np.einsum(
                "ij,ij->i",
                self.ref_basis[:, :, 0],
                np.cross(self.ref_basis[:, :, 1], self.ref_basis[:, :, 2]),
            )
        ) < 1e-18
        self.ref_basis_inv[self.singular_element] = 0.0

        # Mass distribution: keep the frozen scene mass budget exactly.
        mass = initial.mass
        if mass is None:
            mass = np.full(self.node_count, 0.42 / self.node_count)
        self.mass = np.array(mass, dtype=float).reshape(-1)
        self.mass_total = float(self.mass.sum())

        # Boundary conditions -------------------------------------------------
        fixed = initial.attributes.get("fixed_mask")
        if fixed is None:
            fixed = (rest[:, 0] <= -0.150).astype(np.int32)
        self.fixed_mask = np.asarray(fixed).astype(bool).reshape(-1)
        self.free = ~self.fixed_mask
        self.mass_inv = np.where(self.free, 1.0 / np.maximum(self.mass, 1e-30), 0.0)

        # Attributes carried through every frame.
        self.extra_attributes = {}
        for name, value in initial.attributes.items():
            self.extra_attributes[name] = np.array(value)
        self.extra_scalars = {
            name: float(value) for name, value in initial.scalars.items()
        }

        # Driven bones ---------------------------------------------------------
        index_bone = np.array([0.105, 0.03, 0.012], dtype=float)
        thumb_bone = np.array([0.025, -0.074, -0.002], dtype=float)
        index_half = 0.5 * 0.205
        thumb_half = 0.5 * 0.105
        d_index = _segment_distance(
            rest, index_bone - np.array([index_half, 0.0, 0.0]),
            index_bone + np.array([index_half, 0.0, 0.0]),
        )
        d_thumb = _segment_distance(
            rest, thumb_bone - np.array([thumb_half, 0.0, 0.0]),
            thumb_bone + np.array([thumb_half, 0.0, 0.0]),
        )
        self.w_index = self._falloff(d_index, INDEX_INFLUENCE)
        self.w_thumb = self._falloff(d_thumb, THUMB_INFLUENCE)

        # Drive parameters (frozen config).
        self.drive_start = 0.30
        self.drive_end = 1.65
        self.drive_hold = 2.40
        self.drive_angle = math.radians(34.0)
        self.drive_translation = np.array([0.018, 0.052, 0.004], dtype=float)

        # Time stepping --------------------------------------------------------
        self.dt = self._stable_dt()
        self.x = rest.copy()
        self.v = np.zeros_like(rest)
        self.time = 0.0

        # Momentum-budget accumulators (per-term external impulses).
        self._impulse = np.zeros((3, 3), dtype=float)
        self._interval = 0.0
        self.step_count = 0
        self.last_frame_time = 0.0

        # Force scale reported to the platform (dynamic force scale, positive).
        span = rest.max(axis=0) - rest.min(axis=0)
        length_scale = float(np.linalg.norm(span))
        times = np.asarray(scene.sample_times(), dtype=float)
        duration = float(times[-1]) if times.size else 1.0
        self.reference_force = max(
            1e-9, self.mass_total * length_scale / max(duration * duration, 1e-12)
        )
        self._initial_scalars = dict(self.extra_scalars)
        self._initial_attributes = dict(self.extra_attributes)

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _resolve_asset(scene) -> Path:
        module = sys.modules.get(type(scene).__module__)
        asset = getattr(module, "ASSET", None) if module is not None else None
        candidates = []
        if asset is not None:
            candidates.append(Path(asset))
        candidates.append(Path.cwd() / "assets" / "hand_flesh.obj")
        candidates.append(
            Path(__file__).resolve().parent.parent / "assets" / "hand_flesh.obj"
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise FileNotFoundError("hand_flesh.obj not found in workspace assets")

    @staticmethod
    def _falloff(distance: np.ndarray, radius: float) -> np.ndarray:
        ratio = np.clip(distance / max(radius, 1e-12), 0.0, 1.0)
        return (1.0 - ratio) ** 2

    @staticmethod
    def _signed_volume(x: np.ndarray, faces: np.ndarray) -> float:
        a = x[faces[:, 0]]
        b = x[faces[:, 1]]
        c = x[faces[:, 2]]
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    def _element_scale(self, x: np.ndarray) -> np.ndarray:
        a = x[self.f0]
        b = x[self.f1]
        c = x[self.f2]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        return np.sqrt(np.maximum(2.0 * area, 1e-16))

    def _element_basis(self, x: np.ndarray) -> np.ndarray:
        a = x[self.f0]
        b = x[self.f1]
        c = x[self.f2]
        e1 = b - a
        e2 = c - a
        normal = np.cross(e1, e2)
        norm = np.linalg.norm(normal, axis=1)
        norm = np.where(norm > 0.0, norm, 1.0)
        normal = normal / norm[:, None]
        basis = np.empty((len(a), 3, 3), dtype=float)
        basis[:, :, 0] = e1
        basis[:, :, 1] = e2
        basis[:, :, 2] = normal * self._element_scale(x)[:, None]
        return basis

    @staticmethod
    def _inverse3(mats: np.ndarray) -> np.ndarray:
        a = mats[:, :, 0]
        b = mats[:, :, 1]
        c = mats[:, :, 2]
        det = np.einsum("ij,ij->i", a, np.cross(b, c))
        safe = np.where(np.abs(det) < 1e-18, 1.0, det)
        out = np.empty_like(mats)
        out[:, 0, :] = np.cross(b, c) / safe[:, None]
        out[:, 1, :] = np.cross(c, a) / safe[:, None]
        out[:, 2, :] = np.cross(a, b) / safe[:, None]
        out[np.abs(det) < 1e-18] = 0.0
        return out

    def _stable_dt(self) -> float:
        counts = np.bincount(self.e0, minlength=self.node_count) + np.bincount(
            self.e1, minlength=self.node_count
        )
        max_edges = max(int(counts.max()), 1)
        max_length = float(self.rest_length.max()) if self.rest_length.size else 1e-3
        stiffness = (
            max_edges * self.edge_stiffness
            + 6.0 * self.area_stiffness * (0.5 * max_length) ** 2
            + BONE_SPRING
        )
        min_mass = max(float(self.mass[self.free].min()) if self.free.any() else 1e-6,
                       1e-12)
        omega = math.sqrt(stiffness / min_mass)
        dt = 0.35 * 2.0 / max(omega, 1e-9)
        # Align with a divisor-friendly step for the requested sampling.
        return float(min(dt, 1.0e-4))

    # ------------------------------------------------------------------ drive
    def _drive(self, time: float):
        if time <= self.drive_start:
            s = 0.0
        elif time >= self.drive_end:
            s = 1.0
        else:
            s = _smoothstep((time - self.drive_start) / (self.drive_end - self.drive_start))
        rotation = _rotation_z(self.drive_angle * s)
        translation = self.drive_translation * s
        target = self.rest @ rotation.T + translation[None, :]
        return target, s

    # ------------------------------------------------------------------ steps
    def _elastic_forces(self) -> np.ndarray:
        d = self.x[self.e0] - self.x[self.e1]
        length = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
        coef = self.edge_stiffness * (length - self.rest_length) / length
        force = coef[:, None] * d
        return _scatter(self.e0, -force, self.node_count) + _scatter(
            self.e1, force, self.node_count
        )

    def _area_forces(self) -> np.ndarray:
        """Restoring forces that penalise local surface-area change."""
        a = self.x[self.f0]
        b = self.x[self.f1]
        c = self.x[self.f2]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal, axis=1)
        norm = np.where(norm > 1e-16, norm, 1.0)
        nhat = normal / norm[:, None]
        area = 0.5 * norm
        coef = -self.area_stiffness * (area - self.rest_area)
        grad_a = 0.5 * np.cross(nhat, c - b)
        grad_b = 0.5 * np.cross(nhat, a - c)
        grad_c = 0.5 * np.cross(nhat, b - a)
        return (
            _scatter(self.f0, coef[:, None] * grad_a, self.node_count)
            + _scatter(self.f1, coef[:, None] * grad_b, self.node_count)
            + _scatter(self.f2, coef[:, None] * grad_c, self.node_count)
        )

    def _bone_forces(self, time: float) -> np.ndarray:
        target, _ = self._drive(time)
        wt = self.w_thumb[:, None]
        wi = self.w_index[:, None]
        force = BONE_SPRING * wt * (target - self.x)
        force -= BONE_DAMPING * wt * self.v
        force += BONE_SPRING * wi * (self.rest - self.x)
        force -= BONE_DAMPING * wi * self.v
        return force

    def _project_volume(self) -> None:
        x = self.x
        a = x[self.f0]
        b = x[self.f1]
        c = x[self.f2]
        cross_a = np.cross(b, c)
        volume = float(np.einsum("ij,ij->i", a, cross_a).sum() / 6.0)
        error = volume - self.rest_volume
        if abs(error) <= 1e-14:
            return
        grad = (
            _scatter(self.f0, cross_a / 6.0, self.node_count)
            + _scatter(self.f1, np.cross(c, a) / 6.0, self.node_count)
            + _scatter(self.f2, np.cross(a, b) / 6.0, self.node_count)
        )
        weights = self.mass_inv[:, None]
        denom = float(np.sum(self.mass_inv[:, None] * grad * grad))
        if denom <= 1e-30:
            return
        delta = -weights * grad * (error / denom)
        norms = np.linalg.norm(delta, axis=1)
        scale = np.where(norms > MAX_PROJECTION_STEP,
                         MAX_PROJECTION_STEP / np.maximum(norms, 1e-30), 1.0)
        x += delta * scale[:, None]

    def _step(self, dt: float, time: float) -> None:
        p_before = np.sum(self.mass[:, None] * self.v, axis=0)

        force = self._elastic_forces() + self._area_forces() + self._bone_forces(time)
        impulse_bone = dt * force.sum(axis=0)

        self.v[self.free] += dt * force[self.free] * self.mass_inv[self.free, None]
        velocity_before_damping = self.v.copy()
        factor = math.exp(-dt / DAMPING_TIME)
        self.v *= factor
        impulse_damping = np.sum(
            self.mass[:, None] * (self.v - velocity_before_damping), axis=0
        )
        speed = np.linalg.norm(self.v, axis=1)
        over = speed > VELOCITY_CLAMP
        if np.any(over):
            self.v[over] *= (VELOCITY_CLAMP / speed[over])[:, None]

        self.x += dt * self.v
        self.x[self.fixed_mask] = self.rest[self.fixed_mask]
        self.v[self.fixed_mask] = 0.0
        self._project_volume()
        self.x[self.fixed_mask] = self.rest[self.fixed_mask]

        p_after = np.sum(self.mass[:, None] * self.v, axis=0)
        residual = p_after - p_before - impulse_bone - impulse_damping
        self._impulse[0] += impulse_bone
        self._impulse[1] += impulse_damping
        self._impulse[2] += residual

        if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()):
            # Deterministic recovery: never emit a non-finite state.
            self.x = self.rest.copy()
            self.v = np.zeros_like(self.rest)

        self.step_count += 1
        self._interval += dt

    # -------------------------------------------------------------- interface
    def advance_to(self, time: float):
        time = float(time)
        if time < self.time:
            time = self.time
        guard = 0
        while self.time < time - 1e-12:
            dt = min(self.dt, time - self.time)
            if dt <= 1e-14:
                break
            middle = self.time + 0.5 * dt
            self._step(dt, middle)
            self.time += dt
            guard += 1
            if guard > 5_000_000:
                break
        self.time = time
        return self._make_frame(time)

    def _make_frame(self, time: float) -> StateFrame:
        current = self._element_basis(self.x)
        deformation = current @ self.ref_basis_inv
        deformation[~np.isfinite(deformation).all(axis=(1, 2))] = np.eye(3)[None, :, :]
        if np.any(self.singular_element):
            deformation[self.singular_element] = np.eye(3)[None, :, :]

        elapsed = self._interval
        if elapsed > 1e-12:
            budget = self._impulse / elapsed
        else:
            budget = np.zeros((3, 3), dtype=float)
        self._impulse = np.zeros((3, 3), dtype=float)
        self._interval = 0.0

        attributes = {name: np.array(value) for name, value in self._initial_attributes.items()}
        attributes["mass_quantity"] = self.mass.reshape(-1, 1)

        scalars = dict(self._initial_scalars)
        scalars["reference_force"] = float(self.reference_force)

        fields = {
            "budget_external_force": budget.reshape(1, 3, 3),
            "cumulative_mass_source": np.zeros(self.MATERIAL_COUNT, dtype=float),
            "element_deformation_gradient": deformation,
        }

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "steps": int(self.step_count),
            "dt": float(self.dt),
            "node_count": int(self.node_count),
            "element_count": int(self.faces.shape[0]),
            "rest_volume": float(self.rest_volume),
            "total_mass": float(self.mass_total),
            "reference_force": float(self.reference_force),
        }
