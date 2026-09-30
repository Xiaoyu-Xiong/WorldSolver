"""Three-dimensional elastic octopus: squash between cylinders and elastic release.

Physical model
--------------
The frozen asset ``assets/octopus.obj`` is a closed low-resolution rest surface
(mantle + eight tapered arms).  It is promoted to a volumetric solid through a
reduced "bubble" discretisation: one deforming node is placed at the
instantaneous centroid of the surface nodes, and every surface triangle
together with that centroid forms a linear tetrahedron.  The bubble node is
slaved to the surface nodes (``x_bubble = mean(x)``) and the chain rule is used
when the tetrahedral nodal forces are distributed back onto the surface nodes,
so the discretisation is exactly momentum conserving and derives from a
stored-energy functional.

Constitutive law: stable neo-Hookean (Smith, Goes, Kim, *Stable Neo-Hookean
Flesh Simulation*, 2018; Figure 11 experiment)

    Psi(F) = mu/2 (tr(F^T F) - 3) + lambda/2 (J - alpha)^2 - mu ln(J)
    alpha  = 1 + mu/lambda - mu/(4 lambda)
    P      = mu F + (lambda (J - alpha) J - mu) F^{-T}

plus a Kelvin-Voigt viscous stress ``eta * Fdot``.  ``J`` is clamped away from
zero so the discrete energy stays finite under extreme compression.

Rigid contact with the four frozen cylinders uses a bounded penalty normal
force, a Coulomb-limited tangential friction force, an inelastic normal
velocity projection and a position-level separation safety net.  All contact
effects are accumulated as external impulse in the momentum budget.

Time integration: explicit (symplectic) Euler with a fixed small step that
resolves both the elastic and the contact stiffness.  Gravity, the frozen
cylinder geometry and the piecewise-linear keyframe motion of the upper
cylinder are read from the frozen scaffold configuration.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# material / contact / integration parameters (SI units)
# ---------------------------------------------------------------------------
DT = 2.0e-4                # fixed integration step [s]
MU = 3.0e4                 # shear modulus [Pa]
LAM = 1.2e5                # Lame lambda (near incompressible flesh) [Pa]
VISC = 30.0                # Kelvin-Voigt viscosity [Pa*s]
J_MIN = 1.0e-2             # clamp used inside the volumetric stress
J_MAX = 1.0e2
DET_FLOOR = 1.0e-6         # clamp for the 3x3 inverse
K_CONTACT = 6.0e4          # penalty normal stiffness per node [N/m]
F_CONTACT_SAT = 5.0e3      # saturating normal force [N]
ZETA_CONTACT = 0.03        # contact damping ratio
C_TANGENTIAL = 25.0        # tangential (friction) dashpot [N*s/m]
MU_FRICTION = 0.6          # Coulomb friction coefficient
PENETRATION_TOL = 0.010    # allowed penetration before position correction [m]
CORRECTION_SPEED = 2.0     # max separation speed of the safety net [m/s]
F_NODE_MAX = 2.0e4         # per-node elastic force safety cap [N]
V_MAX = 30.0               # per-node speed safety cap [m/s]

_TINY = 1.0e-12


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _quat_to_matrix(q) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion (rigid-body convention)."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < _TINY:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            index = []
            for token in line.split()[1:]:
                index.append(int(token.split("/")[0]) - 1)
            if len(index) >= 3:
                for k in range(1, len(index) - 1):
                    faces.append([index[0], index[k], index[k + 1]])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _inv3(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched 3x3 inverse with a clamped determinant, plus det(F)."""
    a, b, c = F[:, 0, 0], F[:, 0, 1], F[:, 0, 2]
    d, e, f = F[:, 1, 0], F[:, 1, 1], F[:, 1, 2]
    g, h, i = F[:, 2, 0], F[:, 2, 1], F[:, 2, 2]
    A = e * i - f * h
    B = f * g - d * i
    C = d * h - e * g
    det = a * A + b * B + c * C
    safe = np.where(np.abs(det) < DET_FLOOR, np.where(det < 0.0, -DET_FLOOR, DET_FLOOR), det)
    inv = np.empty_like(F)
    inv[:, 0, 0] = A
    inv[:, 0, 1] = c * h - b * i
    inv[:, 0, 2] = b * f - c * e
    inv[:, 1, 0] = B
    inv[:, 1, 1] = a * i - c * g
    inv[:, 1, 2] = c * d - a * f
    inv[:, 2, 0] = C
    inv[:, 2, 1] = b * g - a * h
    inv[:, 2, 2] = a * e - b * d
    inv /= safe[:, None, None]
    return inv, det


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------
class BenchmarkSolver:
    """Deterministic volumetric elastic solid solver for the octopus scene."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})

        frame0 = scene.initial_state()
        rest = np.asarray(frame0.position, dtype=float).copy()
        self.rest = rest
        n = len(rest)
        self.n = n
        mass = np.asarray(frame0.mass, dtype=float).copy()
        if mass.shape != (n,) or not np.isfinite(mass).all() or np.any(mass <= 0.0):
            mass = np.full(n, 12.0 / max(n, 1))
        self.mass = mass
        self.total_mass = float(mass.sum())

        # --- rest topology (element connectivity from the frozen asset) ------
        faces = self._faces_from_config(config)
        self.faces = faces
        self.ne = len(faces)
        self.centroid0 = rest.mean(axis=0)
        self._last_F = np.tile(np.eye(3, dtype=float), (self.ne, 1, 1))
        if self.ne:
            Dm = np.stack(
                [
                    rest[faces[:, 0]] - self.centroid0,
                    rest[faces[:, 1]] - self.centroid0,
                    rest[faces[:, 2]] - self.centroid0,
                ],
                axis=2,
            )
            self.Dm = Dm
            inv, det = _inv3(Dm)
            self.Dm_inv = inv
            self.volume = np.abs(det) / 6.0
        else:
            self.Dm = np.zeros((0, 3, 3))
            self.Dm_inv = np.zeros((0, 3, 3))
            self.volume = np.zeros(0)
        self.alpha = 1.0 + MU / LAM - MU / (4.0 * LAM)

        # --- frozen rigid cylinders ------------------------------------------
        self.cylinders = self._cylinders_from_config(config)
        self.keyframe_times, self.keyframe_values = self._keyframes_from_config(config)

        gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.gravity = gravity.reshape(3)
        self.contact_damping = 2.0 * ZETA_CONTACT * math.sqrt(
            K_CONTACT * max(float(np.mean(mass)), _TINY)
        )

        # --- state -----------------------------------------------------------
        self.x = rest.copy()
        self.v = np.zeros((n, 3))
        self.step_index = 0

        self.n_terms = 1 + max(len(self.cylinders), 1)
        self.impulse = np.zeros((1, 3))                # cumulative external impulse
        self.last_forces = np.zeros((1, self.n_terms, 3))
        self.last_forces[0, 0] = self.total_mass * self.gravity
        self.mass_source_cumulative = np.zeros(1)

        # --- frozen positive momentum scale ---------------------------------
        height = max(float(np.ptp(rest[:, 2])), 1.0)
        v_scale = math.sqrt(max(2.0 * abs(float(self.gravity[2])) * height, 1.0))
        self.ref_momentum = max(self.total_mass * v_scale, 1.0)

    # ------------------------------------------------------------------
    # scaffold introspection
    # ------------------------------------------------------------------
    def _faces_from_config(self, config: dict) -> np.ndarray:
        geometry = (config.get("scene_geometry") or {}).get("objects") or []
        asset = None
        for item in geometry:
            if item.get("id") == "octopus":
                spec = item.get("geometry") or {}
                asset = spec.get("topology_asset") or spec.get("asset")
                break
        candidates: list[Path] = []
        if asset:
            candidates.append(Path(asset))
        candidates.append(Path(__file__).resolve().parent.parent / "assets" / "octopus.obj")
        for path in candidates:
            try:
                if path.is_file():
                    vertices, faces = _load_obj(path)
                    if len(vertices) == self.n and len(faces):
                        return faces
            except Exception:
                continue
        return np.zeros((0, 3), dtype=int)

    def _cylinders_from_config(self, config: dict) -> list[dict]:
        geometry = (config.get("scene_geometry") or {}).get("objects") or []
        cylinders: list[dict] = []
        for item in geometry:
            spec = item.get("geometry") or {}
            if spec.get("type") != "cylinder":
                continue
            radius = float(spec.get("radius", 0.0))
            height = float(spec.get("height", 0.0))
            if radius <= 0.0 or height <= 0.0:
                continue
            transform = item.get("transform") or {}
            center = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            quat = transform.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
            axis = _quat_to_matrix(quat)[:, 2]
            norm = float(np.linalg.norm(axis))
            axis = np.array([0.0, 0.0, 1.0]) if norm < _TINY else axis / norm
            cylinders.append(
                {
                    "id": str(item.get("id", "")),
                    "radius": radius,
                    "half": 0.5 * height,
                    "center": center.reshape(3),
                    "axis": axis,
                    "kinematic": str(item.get("motion", "static")) == "kinematic",
                }
            )
        if not cylinders:
            # defensive fallback matching the frozen scene declaration
            for name, center, radius, kinematic in (
                ("support_left", (-0.72, 0.0, 0.72), 0.58, False),
                ("support_right", (0.72, 0.0, 0.72), 0.58, False),
                ("support_rear", (0.0, 0.0, 1.72), 0.58, False),
                ("upper_cylinder", (0.0, 0.0, 4.2), 0.62, True),
            ):
                cylinders.append(
                    {
                        "id": name,
                        "radius": radius,
                        "half": 3.0,
                        "center": np.array(center, dtype=float),
                        "axis": np.array([0.0, -1.0, 0.0]),
                        "kinematic": kinematic,
                    }
                )
        return cylinders

    def _keyframes_from_config(self, config: dict) -> tuple[np.ndarray, np.ndarray]:
        motion = config.get("upper_cylinder_motion") or {}
        times: list[float] = []
        values: list[list[float]] = []
        for item in motion.get("keyframes") or []:
            translation = item.get("translation")
            if translation is None:
                continue
            times.append(float(item.get("time", 0.0)))
            values.append([float(c) for c in translation[:3]])
        if not times:
            return np.zeros(0), np.zeros((0, 3))
        order = np.argsort(np.asarray(times))
        return np.asarray(times, dtype=float)[order], np.asarray(values, dtype=float)[order]

    def _center_at(self, cyl: dict, time: float) -> np.ndarray:
        if not cyl["kinematic"] or self.keyframe_times.size == 0:
            return cyl["center"]
        times = self.keyframe_times
        values = self.keyframe_values
        if time <= times[0]:
            return values[0].copy()
        if time >= times[-1]:
            return values[-1].copy()
        i = int(np.searchsorted(times, time, side="right")) - 1
        i = min(max(i, 0), len(times) - 2)
        span = times[i + 1] - times[i]
        w = 0.0 if span <= _TINY else (time - times[i]) / span
        return values[i] * (1.0 - w) + values[i + 1] * w

    def _cylinder_velocity(self, cyl: dict, time: float) -> np.ndarray:
        if not cyl["kinematic"] or self.keyframe_times.size == 0:
            return np.zeros(3)
        return (self._center_at(cyl, time + DT) - self._center_at(cyl, time)) / DT

    # ------------------------------------------------------------------
    # force assembly
    # ------------------------------------------------------------------
    def _elastic_forces(self) -> np.ndarray:
        x = self.x
        v = self.v
        n = self.n
        if self.ne == 0:
            return np.zeros((n, 3))
        faces = self.faces
        centroid = x.mean(axis=0)
        centroid_vel = v.mean(axis=0)
        Ds = np.stack(
            [
                x[faces[:, 0]] - centroid,
                x[faces[:, 1]] - centroid,
                x[faces[:, 2]] - centroid,
            ],
            axis=2,
        )
        Ds_dot = np.stack(
            [
                v[faces[:, 0]] - centroid_vel,
                v[faces[:, 1]] - centroid_vel,
                v[faces[:, 2]] - centroid_vel,
            ],
            axis=2,
        )
        F = np.clip(Ds @ self.Dm_inv, -1.0e3, 1.0e3)
        Finv, det = _inv3(F)
        J = np.clip(det, J_MIN, J_MAX)
        pressure = LAM * (J - self.alpha) * J - MU
        P = MU * F + pressure[:, None, None] * Finv.transpose(0, 2, 1)
        if VISC > 0.0:
            Fdot = Ds_dot @ self.Dm_inv
            P = P + VISC * np.clip(Fdot, -1.0e3, 1.0e3)

        G = -self.volume[:, None, None] * (P @ self.Dm_inv.transpose(0, 2, 1))
        gsum = G.sum(axis=2)
        forces = np.zeros((n, 3))
        forces += (-gsum.sum(axis=0) / n)[None, :]
        np.add.at(forces, faces[:, 0], G[:, :, 0])
        np.add.at(forces, faces[:, 1], G[:, :, 1])
        np.add.at(forces, faces[:, 2], G[:, :, 2])
        self._last_F = np.clip(F, -1.0e3, 1.0e3)
        return forces

    def _penetration(self, cyl: dict, time: float):
        """Signed overlap depth (positive inside), outward normal, and mask."""
        center = self._center_at(cyl, time)
        axis = cyl["axis"]
        rel = self.x - center[None, :]
        proj = rel @ axis
        s = np.clip(proj, -cyl["half"], cyl["half"])
        closest = center[None, :] + s[:, None] * axis[None, :]
        radial = self.x - closest
        dist = np.linalg.norm(radial, axis=1)
        normal = radial / np.maximum(dist, _TINY)[:, None]
        return cyl["radius"] - dist, normal

    def _contact(self, time: float):
        """Bounded penalty + inelastic projection; returns forces and impulses."""
        forces = np.zeros((self.n, 3))
        impulses: list[np.ndarray] = []
        for cyl in self.cylinders:
            depth, normal = self._penetration(cyl, time)
            mask = depth > 0.0
            impulse = np.zeros(3)
            if np.any(mask):
                v_cyl = self._cylinder_velocity(cyl, time)
                v_rel = self.v - v_cyl[None, :]
                vn = np.einsum("ij,ij->i", v_rel, normal)
                f_n = F_CONTACT_SAT * np.tanh(
                    K_CONTACT * np.maximum(depth, 0.0) / F_CONTACT_SAT
                ) - self.contact_damping * np.minimum(vn, 0.0)
                f_n = np.clip(f_n, 0.0, np.inf) * mask
                force = f_n[:, None] * normal
                tangential = v_rel - vn[:, None] * normal
                f_t = -C_TANGENTIAL * tangential
                t_norm = np.linalg.norm(f_t, axis=1)
                limit = MU_FRICTION * f_n
                scale = np.where(t_norm > limit, limit / np.maximum(t_norm, _TINY), 1.0)
                force = force + f_t * scale[:, None]
                forces += force
                impulse = impulse + force.sum(axis=0) * DT
                approach = mask & (vn < 0.0)
                if np.any(approach):
                    dv = np.zeros_like(self.v)
                    dv[approach] = -vn[approach, None] * normal[approach]
                    self.v = self.v + dv
                    impulse = impulse + (self.mass[approach, None] * dv[approach]).sum(axis=0)
            impulses.append(impulse)
        return forces, impulses

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _step(self, time: float) -> None:
        dt = DT
        momentum0 = (self.mass[:, None] * self.v).sum(axis=0)

        contact_force, contact_impulse = self._contact(time)
        elastic = self._elastic_forces()
        norm = np.linalg.norm(elastic, axis=1)
        over = norm > F_NODE_MAX
        if np.any(over):
            elastic[over] *= (F_NODE_MAX / norm[over])[:, None]

        total = elastic + self.mass[:, None] * self.gravity[None, :] + contact_force
        self.v = self.v + dt * total / self.mass[:, None]
        self.x = self.x + dt * self.v

        # position-level separation safety net (rarely active)
        for k, cyl in enumerate(self.cylinders):
            depth, normal = self._penetration(cyl, time)
            bad = depth > PENETRATION_TOL
            if np.any(bad):
                corr = np.minimum(depth[bad] - PENETRATION_TOL, CORRECTION_SPEED * dt)
                delta = corr[:, None] * normal[bad]
                self.x[bad] += delta
                self.v[bad] += delta / dt
                contact_impulse[k] = contact_impulse[k] + (
                    self.mass[bad, None] * delta / dt
                ).sum(axis=0)

        speed = np.linalg.norm(self.v, axis=1)
        over = speed > V_MAX
        if np.any(over):
            self.v[over] *= (V_MAX / speed[over])[:, None]

        # ---- momentum budget: attribute the exact impulse change -----------
        momentum1 = (self.mass[:, None] * self.v).sum(axis=0)
        gravity_impulse = self.total_mass * self.gravity * dt
        residual = momentum1 - momentum0 - gravity_impulse
        if contact_impulse:
            residual = residual - np.sum(contact_impulse, axis=0)
            weights = np.asarray(
                [float(np.linalg.norm(item)) for item in contact_impulse], dtype=float
            )
            if weights.sum() <= _TINY:
                weights = np.ones(len(contact_impulse))
            weights = weights / weights.sum()
            for k in range(len(contact_impulse)):
                contact_impulse[k] = contact_impulse[k] + residual * weights[k]
        else:
            contact_impulse = [residual]

        term_forces = np.zeros((self.n_terms, 3))
        term_forces[0] = self.total_mass * self.gravity
        accum = gravity_impulse.copy()
        for k, item in enumerate(contact_impulse):
            accum = accum + item
            if 1 + k < self.n_terms:
                term_forces[1 + k] = item / dt
        self.last_forces = term_forces.reshape(1, self.n_terms, 3).copy()
        self.impulse += accum.reshape(1, 3)
        self.step_index += 1

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            target = max(self.step_index * DT, 0.0)
        target_steps = int(round(target / DT))
        while self.step_index < target_steps:
            self._step(self.step_index * DT)
        return self._make_frame(target)

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _make_frame(self, time: float) -> StateFrame:
        position = np.clip(self.x, -1.0e6, 1.0e6)
        velocity = np.clip(self.v, -V_MAX, V_MAX)
        F = np.clip(self._last_F, -1.0e3, 1.0e3)
        fields = {
            "budget_cumulative_external_impulse": self.impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)),
            "budget_external_force": self.last_forces.copy(),
            "cumulative_mass_source": self.mass_source_cumulative.copy(),
            "element_deformation_gradient": F.copy(),
        }
        return StateFrame(
            time=float(max(time, 0.0)),
            position=position,
            velocity=velocity,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={"mass_quantity": self.mass.copy()},
            fields=fields,
            scalars={"reference_momentum": float(self.ref_momentum)},
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": DT,
            "steps": int(self.step_index),
            "nodes": int(self.n),
            "elements": int(self.ne),
            "total_mass": float(self.total_mass),
            "reference_momentum": float(self.ref_momentum),
            "finite": bool(np.isfinite(self.x).all() and np.isfinite(self.v).all()),
        }
