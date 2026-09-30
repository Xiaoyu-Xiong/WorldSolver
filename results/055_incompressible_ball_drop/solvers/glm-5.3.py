"""Physics solver for scene 055_incompressible_ball_drop.

A homogeneous, nearly incompressible elastic ball is dropped under uniform
gravity onto a rigid horizontal floor, flattens on impact, and recovers with a
volume-conserving elastic oscillation.

Method
------
* The frozen 12-vertex icosahedral surface (``assets/ball_sphere.obj``, the
  same vertex order as ``scene.py``) is subdivided once and every resulting
  surface triangle is joined to the ball centroid, producing a closed
  volumetric mesh of 80 linear tetrahedra (43 nodes).
* Constitutive model: Neo-Hookean strain energy with a large volumetric
  penalty (near-incompressibility),

      psi(F) = mu/2 (tr(F^T F) - 3) + lambda/2 (det F - 1)^2 ,

  evaluated per element; nodal forces are the exact energy gradient
  ``f_i = -d Psi / d x_i`` computed from the First Piola stress.
* Time integration: explicit symplectic Euler with a fixed conservative
  substep, uniform gravity, light aerodynamic drag, and penalty contact
  against the frozen floor plane (normal spring + impact damping + Coulomb
  friction with a smooth stick ramp).
* Exported StateFrames carry the 12 frozen surface samples (positions,
  velocities, per-sample mass), the per-element deformation gradients of all
  tetrahedra in a fixed element order, the external-force / momentum-budget
  bookkeeping of the simulated ball, and the volume-conservation constraint
  value and target.  All exported quantities are derived from the stepped
  physical state; nothing is keyframed or precomputed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # platform contract
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - keeps the module importable standalone
    from dataclasses import dataclass, field as _dc_field

    @dataclass
    class StateFrame:  # type: ignore[no-redef]
        time: float
        position: object
        velocity: object
        mass: object = None
        fields: object = _dc_field(default_factory=dict)
        scalars: object = _dc_field(default_factory=dict)


# ---------------------------------------------------------------------------
# Frozen physical constants and numerical parameters
# ---------------------------------------------------------------------------
GRAVITY = np.array([0.0, 0.0, -9.81])
MU = 70.0                    # shear modulus [Pa]
LAMBDA = 1400.0              # volumetric penalty (near-incompressible) [Pa]
CONTACT_STIFFNESS = 2.0e5    # floor normal penalty [N/m]
CONTACT_DAMPING = 100.0      # floor normal impact damping [N s/m]
FRICTION_COEFF = 0.45        # Coulomb friction coefficient
FRICTION_SPEED = 0.08        # stick ramp speed [m/s]
DRAG = 0.15                  # aerodynamic drag coefficient [1/s]
DT = 1.0e-4                  # fixed physical substep [s]
FLOOR_Z_DEFAULT = 0.0

_PHI = (1.0 + np.sqrt(5.0)) / 2.0
# Icosahedron in the exact vertex order of scene.py / the OBJ asset.
_FALLBACK_VERTICES = (
    (-1.0, _PHI, 0.0), (1.0, _PHI, 0.0), (-1.0, -_PHI, 0.0), (1.0, -_PHI, 0.0),
    (0.0, -1.0, _PHI), (0.0, 1.0, _PHI), (0.0, -1.0, -_PHI), (0.0, 1.0, -_PHI),
    (_PHI, 0.0, -1.0), (_PHI, 0.0, 1.0), (-_PHI, 0.0, -1.0), (-_PHI, 0.0, 1.0),
)
_FALLBACK_FACES = (
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
)


def _load_surface() -> tuple[np.ndarray, np.ndarray]:
    """Load the frozen ball surface (unit sphere, OBJ vertex order)."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    asset = Path(__file__).resolve().parents[1] / "assets" / "ball_sphere.obj"
    try:
        for line in asset.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                faces.append([int(token.split("/")[0]) - 1 for token in parts[1:4]])
    except OSError:
        pass
    if len(vertices) != 12 or len(faces) != 20:
        vertices = [list(v) for v in _FALLBACK_VERTICES]
        faces = [list(f) for f in _FALLBACK_FACES]
    verts = np.asarray(vertices, dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    return verts, np.asarray(faces, dtype=np.int64)


def _volume_mesh(corners: np.ndarray, faces: np.ndarray):
    """Subdivide the surface once and cone it to the centroid -> tet mesh."""
    nodes = [np.asarray(corner, dtype=float) for corner in corners]
    edge_mid: dict[tuple[int, int], int] = {}

    def mid(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        index = edge_mid.get(key)
        if index is None:
            point = 0.5 * (nodes[i] + nodes[j])
            point /= np.linalg.norm(point)
            nodes.append(point)
            index = len(nodes) - 1
            edge_mid[key] = index
        return index

    sub_faces: list[tuple[int, int, int]] = []
    for a, b, c in faces:
        ab, bc, ca = mid(int(a), int(b)), mid(int(b), int(c)), mid(int(c), int(a))
        sub_faces.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))

    center = len(nodes)
    nodes.append(np.zeros(3))
    X = np.vstack([node.reshape(1, 3) for node in nodes])

    tets: list[tuple[int, int, int, int]] = []
    for a, b, c in sub_faces:
        Dm = np.stack((X[a] - X[center], X[b] - X[center], X[c] - X[center]), axis=1)
        if np.linalg.det(Dm) < 0.0:
            b, c = c, b
        tets.append((center, a, b, c))
    corner_ids = np.arange(len(corners), dtype=np.int64)
    return X, np.asarray(tets, dtype=np.int64), corner_ids


def _safe_deformation(F: np.ndarray) -> np.ndarray:
    """Guard against (never expected) degenerate elements."""
    J = np.linalg.det(F)
    bad = ~np.isfinite(J) | (J < 1.0e-4)
    if bad.any():
        F = np.where(bad[:, None, None], 0.5 * (F + np.eye(3)), F)
    return F


class BenchmarkSolver:
    """Tetrahedral FEM solver for the incompressible soft ball drop."""

    def __init__(self) -> None:
        self._stats = {
            "method": "neo-hookean tet FEM, explicit symplectic Euler",
            "substeps": 0,
            "first_contact_time": None,
            "min_volume_ratio": 1.0,
            "max_node_speed": 0.0,
        }

    # -- setup ---------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        config = getattr(scene, "config", None) or {}

        corners, faces = _load_surface()
        X_unit, tets, corner_ids = _volume_mesh(corners, faces)

        # Frozen initial state: sphere translated by (0, 0, 2.25).
        initial = scene.initial_state()
        init_pos = np.asarray(initial.position, dtype=float)
        offset = init_pos.mean(axis=0) - corners.mean(axis=0)
        self.X = X_unit + offset

        self.tets = tets
        self.corner_ids = corner_ids
        self.n_nodes = len(self.X)

        # Reference element geometry.
        o = self.X[self.tets[:, 0]]
        Dm = np.stack(
            (
                self.X[self.tets[:, 1]] - o,
                self.X[self.tets[:, 2]] - o,
                self.X[self.tets[:, 3]] - o,
            ),
            axis=2,
        )
        self.Dm_inv = np.linalg.inv(Dm)
        self.Dm_inv_T = self.Dm_inv.transpose(0, 2, 1)
        self.V0 = np.abs(np.linalg.det(Dm)) / 6.0
        self.volume0 = float(self.V0.sum())

        # Mass distribution (frozen total mass from the scene).
        sample_mass = np.asarray(initial.mass, dtype=float)
        self.sample_mass = sample_mass
        total_mass = float(sample_mass.sum())
        self.total_mass = total_mass
        rho = total_mass / self.volume0
        self.mass = np.zeros(self.n_nodes)
        np.add.at(self.mass, self.tets, np.tile((rho * self.V0 / 4.0)[:, None], (1, 4)))
        self.inv_mass = 1.0 / self.mass
        self.mg = self.mass[:, None] * GRAVITY[None, :]

        self.floor_z = self._floor_level(config)
        gap = float(self.X[:, 2].min() - self.floor_z)
        self.reference_momentum = total_mass * np.sqrt(2.0 * 9.81 * max(gap, 1.0e-6))

        self._reset()

    @staticmethod
    def _floor_level(config: dict) -> float:
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []):
            if obj.get("id") != "ground":
                continue
            spec = obj.get("geometry", {}) or {}
            if spec.get("type") == "box" and spec.get("half_extents"):
                translation = (obj.get("transform") or {}).get("translation", [0.0, 0.0, 0.0])
                return float(translation[2]) + float(spec["half_extents"][2])
        return FLOOR_Z_DEFAULT

    def _reset(self) -> None:
        self.t = 0.0
        self.x = self.X.copy()
        self.v = np.zeros_like(self.X)
        self.impulse = np.zeros(3)

    # -- physics -------------------------------------------------------------
    def _deformation_gradients(self, x: np.ndarray) -> np.ndarray:
        o = x[self.tets[:, 0]]
        Ds = (
            x[self.tets[:, 1:]] - o[:, None, :]
        ).transpose(0, 2, 1)
        return _safe_deformation(Ds @ self.Dm_inv)

    def _elastic_force(self, x: np.ndarray) -> np.ndarray:
        F = self._deformation_gradients(x)
        J = np.linalg.det(F)
        F_inv_T = np.linalg.inv(F).transpose(0, 2, 1)
        # First Piola stress of psi = mu/2 (I_C - 3) + lambda/2 (J - 1)^2.
        P = MU * F + (LAMBDA * (J - 1.0) * J)[:, None, None] * F_inv_T
        H = self.V0[:, None, None] * (P @ self.Dm_inv_T)
        ha, hb, hc = H[:, :, 0], H[:, :, 1], H[:, :, 2]
        values = np.stack((-ha, -hb, -hc, ha + hb + hc), axis=1)
        force = np.zeros_like(x)
        np.add.at(force, self.tets, values)
        return force

    def _contact_force(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        force = np.zeros_like(x)
        pen = self.floor_z - x[:, 2]
        active = pen > 0.0
        if not active.any():
            return force
        pen_a = pen[active]
        normal = CONTACT_STIFFNESS * pen_a + CONTACT_DAMPING * np.maximum(-v[active, 2], 0.0)
        tangential = v[active, :2]
        speed = np.linalg.norm(tangential, axis=1)
        magnitude = FRICTION_COEFF * normal * np.tanh(speed / FRICTION_SPEED)
        friction = -magnitude[:, None] * tangential / np.maximum(speed, 1.0e-12)[:, None]
        force[active, 0] = friction[:, 0]
        force[active, 1] = friction[:, 1]
        force[active, 2] = normal
        return force

    def _step(self, dt: float) -> None:
        drag = -DRAG * (self.mass[:, None] * self.v)
        contact = self._contact_force(self.x, self.v)
        force = self._elastic_force(self.x) + self.mg + drag + contact
        self.v += dt * force * self.inv_mass[:, None]
        np.clip(self.v, -1.0e3, 1.0e3, out=self.v)
        self.x += dt * self.v
        self.t += dt
        self.impulse += dt * (self.mg.sum(axis=0) + contact.sum(axis=0) + drag.sum(axis=0))
        self._stats["substeps"] += 1
        if self._stats["first_contact_time"] is None and np.any(contact[:, 2] > 0.0):
            self._stats["first_contact_time"] = float(self.t)
        if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()):
            raise RuntimeError("numerical divergence in elastic ball simulation")

    # -- export --------------------------------------------------------------
    def _frame(self, time: float):
        F = self._deformation_gradients(self.x)
        volume = float(np.sum(self.V0 * np.linalg.det(F)))

        contact_total = self._contact_force(self.x, self.v).sum(axis=0)
        gravity_total = self.total_mass * GRAVITY
        drag_total = -DRAG * (self.mass[:, None] * self.v).sum(axis=0)
        external = np.stack((gravity_total, contact_total, drag_total))

        ids = self.corner_ids
        fields = {
            "budget_cumulative_external_impulse": self.impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, len(ids))),
            "budget_external_force": external.reshape(1, 3, 3),
            "constraint_target": np.array([self.volume0]),
            "constraint_value": np.array([volume]),
            "cumulative_mass_source": np.zeros(3),
            "element_deformation_gradient": F,
        }
        scalars = {
            "constraint_scale": self.volume0,
            "reference_momentum": float(self.reference_momentum),
        }

        self._stats["min_volume_ratio"] = min(
            self._stats["min_volume_ratio"], volume / self.volume0
        )
        self._stats["max_node_speed"] = max(
            self._stats["max_node_speed"], float(np.max(np.abs(self.v)))
        )

        return StateFrame(
            float(time),
            self.x[ids].copy(),
            self.v[ids].copy(),
            mass=self.sample_mass.copy(),
            fields=fields,
            scalars=scalars,
        )

    # -- contract entry points ----------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target < self.t - 1.0e-9:
            self._reset()
        while self.t < target - 1.0e-12:
            self._step(min(DT, target - self.t))
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return dict(self._stats)
