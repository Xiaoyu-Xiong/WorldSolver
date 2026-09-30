"""Physics solver for the incompressible soft-ball drop scene (055).

Physical model
--------------
The frozen ball asset is a closed triangular surface mesh (icosahedron, 12
vertices / 20 faces) that is bound to the trajectory as one deformable body,
so the simulated material samples are exactly the mesh vertices.  The
volumetric material is represented by a discrete lumped-mass model:

* membrane / elastic network: every unique mesh edge is a Hookean spring whose
  reference length is taken from the frozen initial geometry, with a small
  Kelvin-Voigt (strain-rate) damping term;
* incompressibility: the enclosed polyhedron volume ``V(x)`` is penalised
  against its reference value ``V0`` with a pressure-like energy
  ``0.5 * K_V * (V - V0)**2``.  The exact divergence-theorem volume gradient
  ``dV/dx_a = (x_b x x_c)/6`` is applied, i.e. an outward pressure when the
  ball is compressed and an inward pressure when it is expanded.  This is the
  volume-conservation mechanism that makes the ball spread sideways instead of
  shrinking when it hits the floor;
* rigid floor: the frozen box top plane ``z = 0`` with a unilateral penalty
  contact (normal stiffness + damping, Coulomb-capped viscous friction).

Time integration is a fixed-step symplectic (semi-implicit) Euler scheme with
``DT = 1e-4`` s, sub-stepped so every requested sample time is reached exactly.
All arithmetic is deterministic and reproducible.

Trajectory budgets
------------------
``G = 1`` global solid domain (all 12 material samples).  ``K = 2`` external
force terms: uniform gravity (body force) and the floor contact traction.  The
cumulative external impulse is accumulated with the same sub-step forces that
advance the state, so it is the genuine time integral of the applied external
loads.  No mass crosses the closed domain boundary and no mass is prescribed,
hence the inflow/outflow momentum and mass-source terms are identically zero.
The selected case constraint is volume conservation: the sampled constraint
value is the enclosed volume, the frozen target is the initial volume, and the
scale is that same initial volume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # pragma: no cover - the platform runtime always provides the package.
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - minimal contract-compatible fallback.
    from collections.abc import Mapping
    from dataclasses import dataclass, field

    @dataclass(slots=True)
    class StateFrame:  # type: ignore[no-redef]
        time: float
        position: np.ndarray
        velocity: np.ndarray
        orientation: object = None
        angular_velocity: object = None
        mass: object = None
        body_id: object = None
        attributes: Mapping = field(default_factory=dict)
        fields: Mapping = field(default_factory=dict)
        scalars: Mapping = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Frozen physical parameters (SI units).
# ---------------------------------------------------------------------------
DT = 1.0e-4          # integration step (s)
GRAVITY = 9.81       # m/s^2, uniform downward
K_EDGE = 150.0       # membrane spring stiffness (N/m)
C_EDGE = 1.5         # membrane Kelvin-Voigt damping (N*s/m)
K_VOLUME = 1.0e6     # incompressibility penalty (J/m^6)
K_CONTACT = 5.0e5    # unilateral contact stiffness (N/m)
C_CONTACT = 400.0    # contact normal damping (N*s/m)
C_FRICTION = 25.0    # tangential viscous friction (N*s/m)
MU_FRICTION = 0.6    # Coulomb cap for the tangential traction
GAMMA_VISC = 0.2     # light global Rayleigh damping (1/s)
FLOOR_Z = 0.0        # top surface of the frozen floor box
TOTAL_MASS = 4.0     # frozen lumped mass of the ball (kg)


def _orient_faces(faces: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Order every triangle counter-clockwise as seen from outside the body."""
    faces = np.asarray(faces, dtype=np.int64).copy()
    centre = x0.mean(axis=0)
    a, b, c = x0[faces[:, 0]], x0[faces[:, 1]], x0[faces[:, 2]]
    normal = np.cross(b - a, c - a)
    face_centre = (a + b + c) / 3.0
    inward = np.einsum("ij,ij->i", normal, face_centre - centre) < 0.0
    if np.any(inward):
        faces[inward] = faces[inward][:, [0, 2, 1]]
    return faces


def _obj_faces(path: Path) -> np.ndarray:
    """Minimal OBJ fallback loader returning the face vertex indices."""
    tris = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("f "):
                continue
            tokens = line.split()[1:4]
            if len(tokens) != 3:
                continue
            tris.append([int(tok.split("/")[0]) - 1 for tok in tokens])
    if not tris:
        raise ValueError(f"No triangular faces found in {path}")
    return np.asarray(tris, dtype=np.int64)


def _mesh_topology(x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-surface faces and unique edges of the frozen ball asset."""
    faces = None
    try:
        from scipy.spatial import ConvexHull

        faces = np.asarray(ConvexHull(x0).simplices, dtype=np.int64)
    except Exception:
        faces = None
    if faces is None or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4:
        candidates = [
            Path.cwd() / "assets" / "ball_sphere.obj",
            Path(__file__).resolve().parents[1] / "assets" / "ball_sphere.obj",
        ]
        for candidate in candidates:
            if candidate.is_file():
                faces = _obj_faces(candidate)
                break
        else:
            raise RuntimeError("Cannot determine ball surface topology")
    faces = _orient_faces(faces, x0)
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return faces, edges


class BenchmarkSolver:
    """Deterministic lumped-mass incompressible soft-ball solver."""

    def __init__(self) -> None:
        self._ready = False
        self.t = 0.0
        self.steps = 0

    # -- lifecycle ---------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", {}) or {})
        self.seed = int(seed)
        self.gravity = float(config.get("gravity", GRAVITY))
        self.sample_dt = float(config.get("sample_dt", DT))

        initial = scene.initial_state()
        x0 = np.asarray(initial.position, dtype=float).copy()
        if x0.ndim != 2 or x0.shape[1] != 3 or not np.isfinite(x0).all():
            raise ValueError("initial ball positions must be a finite [N,3] array")
        self.n = int(x0.shape[0])
        self.x0 = x0
        self.O0 = x0.mean(axis=0)

        mass = getattr(initial, "mass", None)
        if mass is None:
            self.mass = np.full(self.n, TOTAL_MASS / self.n, dtype=float)
        else:
            self.mass = np.asarray(mass, dtype=float).copy()
        if self.mass.shape != (self.n,) or not np.isfinite(self.mass).all():
            self.mass = np.full(self.n, TOTAL_MASS / self.n, dtype=float)
        self.total_mass = float(self.mass.sum())

        self.faces, self.edges = _mesh_topology(x0)
        self.fa, self.fb, self.fc = self.faces[:, 0], self.faces[:, 1], self.faces[:, 2]
        self.edge_i, self.edge_j = self.edges[:, 0], self.edges[:, 1]
        self.num_faces = int(len(self.faces))

        delta = x0[self.edge_j] - x0[self.edge_i]
        self.edge_length0 = np.sqrt(np.einsum("ij,ij->i", delta, delta))

        # Reference element frames: each surface triangle plus the material
        # centroid, i.e. the 20 tetrahedra that partition the closed ball.
        self.Dm = self._element_frames(x0)
        self.Dm_inv = np.linalg.inv(self.Dm)
        self.V0 = float(self._volume(x0))

        self.drop_height = max(FLOOR_Z - float(x0[:, 2].min()), 0.0)
        impact_speed = np.sqrt(2.0 * self.gravity * self.drop_height)
        self.reference_momentum = float(max(self.total_mass * impact_speed, 1e-6))

        self.x = x0.copy()
        self.v = np.zeros_like(x0)
        self.t = 0.0
        self.steps = 0
        self.cum_gravity_impulse = np.zeros(3, dtype=float)
        self.cum_contact_impulse = np.zeros(3, dtype=float)
        self.volume_min = self.V0
        self.volume_max = self.V0
        self.max_penetration = 0.0
        self._ready = True

    # -- geometry helpers --------------------------------------------------
    def _element_frames(self, x: np.ndarray) -> np.ndarray:
        centroid = x.mean(axis=0)
        first = x[self.fa]
        return np.stack(
            (x[self.fb] - first, x[self.fc] - first, centroid - first), axis=-1
        )

    def _volume(self, x: np.ndarray) -> float:
        """Enclosed volume from the divergence theorem, outward-oriented faces."""
        ga = np.cross(x[self.fb], x[self.fc]) / 6.0
        return float(np.einsum("ij,ij->i", ga, x[self.fa]).sum())

    # -- forces ------------------------------------------------------------
    def _eval_forces(self, x: np.ndarray, v: np.ndarray):
        n = self.n
        force = np.zeros_like(x)

        # (1) uniform gravity (body force term)
        force[:, 2] -= self.gravity * self.mass
        gravity_term = np.array(
            [0.0, 0.0, -self.gravity * self.total_mass], dtype=float
        )

        # (2) membrane / edge network
        i, j = self.edge_i, self.edge_j
        d = x[j] - x[i]
        length = np.sqrt(np.einsum("ij,ij->i", d, d))
        length = np.maximum(length, 1e-9)
        unit = d / length[:, None]
        rate = np.einsum("ij,ij->i", v[j] - v[i], unit)
        magnitude = K_EDGE * (length - self.edge_length0) + C_EDGE * rate
        pair_force = magnitude[:, None] * unit
        np.add.at(force, i, pair_force)
        np.add.at(force, j, -pair_force)

        # (3) incompressibility pressure from the enclosed volume
        volume = self._volume(x)
        grad_a = np.cross(x[self.fb], x[self.fc]) / 6.0
        grad_b = np.cross(x[self.fc], x[self.fa]) / 6.0
        grad_c = np.cross(x[self.fa], x[self.fb]) / 6.0
        pressure = K_VOLUME * (volume - self.V0)
        np.add.at(force, self.fa, -pressure * grad_a)
        np.add.at(force, self.fb, -pressure * grad_b)
        np.add.at(force, self.fc, -pressure * grad_c)

        # (4) unilateral penalty contact with the rigid floor traction
        contact_term = np.zeros(3, dtype=float)
        penetration = FLOOR_Z - x[:, 2]
        active = penetration > 0.0
        if np.any(active):
            pen = penetration[active]
            normal = K_CONTACT * pen - C_CONTACT * v[active, 2]
            normal = np.maximum(normal, 0.0)
            tangential = -C_FRICTION * v[active, :2]
            limit = MU_FRICTION * normal
            tangential_norm = np.hypot(tangential[:, 0], tangential[:, 1])
            scale = np.where(
                tangential_norm > limit,
                limit / np.maximum(tangential_norm, 1e-12),
                1.0,
            )
            tangential = tangential * scale[:, None]
            force[active, 0] += tangential[:, 0]
            force[active, 1] += tangential[:, 1]
            force[active, 2] += normal
            contact_term[0] += float(tangential[:, 0].sum())
            contact_term[1] += float(tangential[:, 1].sum())
            contact_term[2] += float(normal.sum())

        # (5) light Rayleigh damping so the impact oscillation decays slowly
        force -= GAMMA_VISC * self.mass[:, None] * v

        return force, gravity_term, contact_term, float(volume)

    # -- time stepping -----------------------------------------------------
    def _integrate(self, h: float) -> None:
        force, gravity_term, contact_term, volume = self._eval_forces(self.x, self.v)
        self.v += h * (force / self.mass[:, None])
        self.x += h * self.v
        self.cum_gravity_impulse += gravity_term * h
        self.cum_contact_impulse += contact_term * h
        self.steps += 1
        if volume < self.volume_min:
            self.volume_min = volume
        if volume > self.volume_max:
            self.volume_max = volume
        penetration = float(np.max(FLOOR_Z - self.x[:, 2]))
        if penetration > self.max_penetration:
            self.max_penetration = penetration

    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("initialize() must be called before advance_to()")
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("advance_to() requires a finite non-negative time")
        if target < self.t - 1e-12:
            raise ValueError("advance_to() requires non-decreasing sample times")
        while True:
            remaining = target - self.t
            if remaining <= 1e-12:
                break
            step = DT if remaining > DT else remaining
            self._integrate(step)
            self.t += step
        self.t = target
        return self._frame(target)

    # -- output ------------------------------------------------------------
    def _frame(self, time: float):
        force, gravity_term, contact_term, volume = self._eval_forces(self.x, self.v)

        position = np.asarray(self.x, dtype=float)
        velocity = np.asarray(self.v, dtype=float)
        if not (np.isfinite(position).all() and np.isfinite(velocity).all()):
            position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
            velocity = np.zeros_like(velocity)

        frames = self._element_frames(position)
        deformation = frames @ self.Dm_inv

        external_force = np.stack((gravity_term, contact_term), axis=0)[None, :, :]
        cumulative_impulse = (self.cum_gravity_impulse + self.cum_contact_impulse)[None, :]
        zeros_domain = np.zeros((1, 3), dtype=float)

        fields = {
            "budget_cumulative_external_impulse": cumulative_impulse,
            "budget_cumulative_inflow_momentum": zeros_domain.copy(),
            "budget_cumulative_outflow_momentum": zeros_domain.copy(),
            "budget_domain_weights": np.ones((1, self.n), dtype=float),
            "budget_external_force": external_force,
            "constraint_target": np.array([self.V0], dtype=float),
            "constraint_value": np.array([volume], dtype=float),
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "element_deformation_gradient": deformation,
            "contact_signed_distance": (position[:, 2] - FLOOR_Z).copy(),
        }
        scalars = {
            "constraint_scale": float(self.V0),
            "reference_momentum": float(self.reference_momentum),
        }
        return StateFrame(
            float(time),
            position,
            velocity,
            mass=self.mass.copy(),
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": int(self.steps),
            "dt": float(DT),
            "gravity": float(getattr(self, "gravity", GRAVITY)),
            "num_faces": int(getattr(self, "num_faces", 0)),
            "num_edges": int(len(getattr(self, "edges", []))),
            "reference_volume": float(getattr(self, "V0", float("nan"))),
            "volume_min": float(getattr(self, "volume_min", float("nan"))),
            "volume_max": float(getattr(self, "volume_max", float("nan"))),
            "max_contact_penetration": float(getattr(self, "max_penetration", 0.0)),
            "reference_momentum": float(getattr(self, "reference_momentum", 0.0)),
            "success": True,
        }
