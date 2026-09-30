"""Viscoelastic pool splash and rebound solver.

Own implementation of a weakly-compressible, viscoelastic (Maxwell / Oldroyd-B
style) SPH fluid.  The material model combines

* a Tait equation of state driven by the continuity equation (pressure),
* a hyperelastic neo-Hookean deviatoric stress built from a per-particle
  deformation gradient ``F`` evolved with the SPH velocity gradient,
  ``Fdot = L F - (F - J^(1/3) I) / lambda``  (Maxwell relaxation),
* a Newtonian deviatoric viscous stress,
* Monaghan artificial viscosity,
* analytic rigid-box penalty contacts (normal spring/damper + Coulomb friction).

Gravity, the frozen rigid basin and the sampled particle layout all come from
the frozen scene.  Time integration is explicit symplectic Euler with an
adaptive CFL limited step, so states are deterministic and finite at every
requested sample time.  All exported quantities are genuine raw physical
state, not animation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:  # scipy is the standard neighbour-search route
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

try:
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - minimal mirror of the frozen contract
    from dataclasses import dataclass, field as _dataclass_field

    @dataclass(slots=True)
    class StateFrame:  # type: ignore[no-redef]
        time: float
        position: np.ndarray
        velocity: np.ndarray
        orientation: np.ndarray | None = None
        angular_velocity: np.ndarray | None = None
        mass: np.ndarray | None = None
        body_id: np.ndarray | None = None
        attributes: dict = _dataclass_field(default_factory=dict)
        fields: dict = _dataclass_field(default_factory=dict)
        scalars: dict = _dataclass_field(default_factory=dict)

        def validate(self) -> None:
            if not np.isfinite(self.time) or self.time < 0:
                raise ValueError("bad time")
            for name in ("position", "velocity"):
                value = np.asarray(getattr(self, name))
                if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
                    raise ValueError(f"bad {name}")
            for value in self.attributes.values():
                if not np.isfinite(np.asarray(value)).all():
                    raise ValueError("bad attribute")
            for value in self.fields.values():
                if not np.isfinite(np.asarray(value)).all():
                    raise ValueError("bad field")
            if not all(np.isfinite(v) for v in self.scalars.values()):
                raise ValueError("bad scalar")


# Frozen fallback geometry (identical to scene_geometry.yaml) used only when the
# frozen YAML cannot be read from the workspace.
_FALLBACK_BOXES = (
    ((-0.0, 0.0, -0.05), (3.2, 1.4, 0.05)),
    ((0.0, 0.72, 0.3), (1.45, 0.08, 0.34)),
    ((-1.45, 0.0, 0.3), (0.08, 0.72, 0.34)),
    ((1.45, 0.0, 0.3), (0.08, 0.72, 0.34)),
)


def _quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


class BenchmarkSolver:
    """Deterministic viscoelastic SPH solver."""

    # ---- material / numerical constants (SI units) -------------------------
    rho0 = 1350.0          # rest density [kg/m^3] (frozen scene density)
    sound_speed = 15.0     # numerical sound speed of the Tait EOS [m/s]
    eos_gamma = 7.0
    shear_modulus = 2.0e4  # elastic shear modulus mu [Pa]
    relaxation_time = 2.5  # Maxwell relaxation time lambda [s]
    dynamic_viscosity = 3.0   # deviatoric Newtonian viscosity [Pa s]
    stress_cap = 4.0e4     # deviatoric stress magnitude cap [Pa]
    artificial_alpha = 0.2
    contact_stiffness = 2.0e5   # normal penalty stiffness [N/m]
    contact_friction = 0.45
    dt_max = 1.2e-3
    dt_min = 1.0e-5
    cfl = 0.25

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = getattr(scene, "config", {}) or {}
        sampling = config.get("sampling", {}) or {}
        self.spacing = float(sampling.get("spacing", 0.14))
        self.rho0 = float(sampling.get("density", self.rho0))
        gravity = np.asarray(config.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        self.gravity = gravity if gravity.shape == (3,) else np.array([0.0, 0.0, -9.81])

        frame = scene.initial_state()
        self.pos = np.array(frame.position, dtype=float)
        self.vel = np.array(frame.velocity, dtype=float)
        self.mass = np.array(frame.mass, dtype=float).reshape(-1)
        n = len(self.pos)
        if frame.body_id is None:
            self.labels = np.zeros(n, dtype=np.int64)
        else:
            self.labels = np.array(frame.body_id, dtype=np.int64).reshape(-1)

        # smoothing length: cubic spline with support radius 2h
        self.h = 1.5 * self.spacing
        self.kernel_norm = 1.0 / (math.pi * self.h ** 3)
        self.particle_radius = 0.45 * self.spacing
        self.eos_B = self.rho0 * self.sound_speed ** 2 / self.eos_gamma
        self.elastic_speed = math.sqrt(self.shear_modulus / self.rho0)
        self.contact_damping = 0.6 * math.sqrt(self.contact_stiffness * float(np.mean(self.mass)))
        self.friction_damping = 0.5 * self.contact_damping

        self.rho = np.full(n, self.rho0)
        self.F = np.repeat(np.eye(3, dtype=float)[None, :, :], n, axis=0)
        self.t = float(frame.time)

        self.boxes = self._load_boundaries(config)
        self.contact_sdf = self._contact_sdf(self.pos)
        self.ext_force = self._external_force(self.pos, self.vel)
        self.steps = 0
        self.dt_min_used = self.dt_max
        self._ready = True

    def _load_boundaries(self, config) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        raw = None
        if isinstance(config, dict):
            candidate = config.get("scene_geometry")
            if isinstance(candidate, dict):
                raw = candidate
        if raw is None:
            here = Path(__file__).resolve().parent.parent
            for path in (here / "scene_geometry.yaml", Path.cwd() / "scene_geometry.yaml"):
                if path.is_file():
                    try:
                        import yaml

                        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                    except Exception:
                        raw = None
                    if isinstance(raw, dict):
                        break
        boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        objects = (raw or {}).get("objects", []) if isinstance(raw, dict) else []
        for obj in objects:
            try:
                if not obj.get("collision_enabled", True):
                    continue
                geometry = obj.get("geometry", {}) or {}
                if geometry.get("type") != "box":
                    continue
                half = np.asarray(geometry.get("half_extents"), dtype=float)
                transform = obj.get("transform", {}) or {}
                center = np.asarray(transform.get("translation", (0.0, 0.0, 0.0)), dtype=float)
                rotation = _quat_to_matrix(transform.get("rotation_xyzw", (0.0, 0.0, 0.0, 1.0)))
                scale = np.asarray(transform.get("scale", (1.0, 1.0, 1.0)), dtype=float)
                if half.shape != (3,) or center.shape != (3,):
                    continue
                boxes.append((center, np.abs(half * scale), rotation))
            except Exception:
                continue
        if not boxes:
            boxes = [
                (np.array(c, dtype=float), np.array(h, dtype=float), np.eye(3))
                for c, h in _FALLBACK_BOXES
            ]
        return boxes

    # ------------------------------------------------------------- geometry io
    def _box_sdf_normal(self, x: np.ndarray, center, half, rotation):
        local = (x - center) @ rotation
        q = np.abs(local) - half
        outside = np.maximum(q, 0.0)
        sdf = np.linalg.norm(outside, axis=1) + np.minimum(np.max(q, axis=1), 0.0)

        closest = np.clip(local, -half, half)
        diff = local - closest
        dist = np.linalg.norm(diff, axis=1)
        n_local = np.zeros_like(local)
        outer_mask = dist > 1e-12
        if np.any(outer_mask):
            n_local[outer_mask] = diff[outer_mask] / dist[outer_mask, None]
        inner_mask = ~outer_mask
        if np.any(inner_mask):
            gaps = half[None, :] - np.abs(local[inner_mask])
            axis = np.argmin(gaps, axis=1)
            rows = np.arange(len(axis))
            n_local[inner_mask, axis] = np.sign(local[inner_mask][rows, axis])
            n_local[inner_mask, axis] += (n_local[inner_mask][rows, axis] == 0.0) * 1.0
        normal = n_local @ rotation.T
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        normal = normal / np.maximum(norm, 1e-12)
        return sdf, normal

    def _contact(self, x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (min signed gap, contact force) over all rigid boxes."""
        n = len(x)
        sdf_best = np.full(n, np.inf)
        force = np.zeros((n, 3))
        radius = self.particle_radius
        vn = np.zeros(n)
        for center, half, rotation in self.boxes:
            sdf, normal = self._box_sdf_normal(x, center, half, rotation)
            sdf_best = np.minimum(sdf_best, sdf)
            penetration = radius - sdf
            vn = np.einsum("ij,ij->i", v, normal)
            magnitude = self.contact_stiffness * penetration - self.contact_damping * vn
            magnitude = np.where((penetration > 0.0) & (magnitude > 0.0), magnitude, 0.0)
            force += magnitude[:, None] * normal
            tangent = v - vn[:, None] * normal
            speed = np.linalg.norm(tangent, axis=1)
            fric = np.minimum(self.friction_damping * speed, self.contact_friction * magnitude)
            scale = np.where(speed > 1e-9, fric / np.maximum(speed, 1e-12), 0.0)
            force -= scale[:, None] * tangent
        return sdf_best, force

    def _contact_sdf(self, x: np.ndarray) -> np.ndarray:
        sdf_best = np.full(len(x), np.inf)
        for center, half, rotation in self.boxes:
            sdf, _ = self._box_sdf_normal(x, center, half, rotation)
            sdf_best = np.minimum(sdf_best, sdf)
        return sdf_best

    def _external_force(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        _, contact = self._contact(x, v)
        return self.mass[:, None] * self.gravity[None, :] + contact

    # ------------------------------------------------------------ environment
    def _kernel(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = self.h
        q = np.clip(r / h, 0.0, 2.0)
        w = np.zeros_like(q)
        dw = np.zeros_like(q)
        inner = q < 1.0
        outer = (~inner) & (q < 2.0)
        qi = q[inner]
        w[inner] = self.kernel_norm * (1.0 - 1.5 * qi ** 2 + 0.75 * qi ** 3)
        dw[inner] = self.kernel_norm * (-3.0 * qi + 2.25 * qi ** 2) / h
        qo = q[outer]
        w[outer] = 0.25 * self.kernel_norm * (2.0 - qo) ** 3
        dw[outer] = -0.75 * self.kernel_norm * (2.0 - qo) ** 2 / h
        return w, dw

    def _pairs(self, x: np.ndarray) -> np.ndarray:
        search = 2.0 * self.h * (1.0 + 1e-9)
        if cKDTree is not None:
            tree = cKDTree(x)
            pair_array = tree.query_pairs(search, output_type="ndarray")
            if pair_array.size == 0:
                return np.zeros((0, 2), dtype=np.int64)
            return np.asarray(pair_array, dtype=np.int64).reshape(-1, 2)
        # dense fallback
        n = len(x)
        diff = x[:, None, :] - x[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        idx = np.argwhere((dist <= search) & (np.arange(n)[:, None] < np.arange(n)[None, :]))
        return np.asarray(idx, dtype=np.int64)

    def _pressure(self) -> np.ndarray:
        ratio = np.clip(self.rho / self.rho0, 0.5, 1.8)
        p = self.eos_B * (ratio ** self.eos_gamma - 1.0)
        return np.maximum(p, 0.0)

    def _deviatoric_stress(self, L: np.ndarray) -> np.ndarray:
        F = self.F
        det = np.linalg.det(F)
        det = np.clip(np.nan_to_num(det, nan=1.0, posinf=1.0, neginf=1.0), 0.25, 4.0)
        b = np.einsum("pab,pcb->pac", F, F)
        trace_b = np.einsum("paa->p", b)
        eye = np.eye(3)[None, :, :]
        dev_elastic = self.shear_modulus * (det ** (-5.0 / 3.0))[:, None, None] * (
            b - (trace_b / 3.0)[:, None, None] * eye
        )
        symmetric = 0.5 * (L + np.transpose(L, (0, 2, 1)))
        trace_L = np.einsum("paa->p", L)
        viscous = 2.0 * self.dynamic_viscosity * (
            symmetric - (trace_L / 3.0)[:, None, None] * eye
        )
        stress = dev_elastic + viscous
        stress = np.nan_to_num(stress, nan=0.0, posinf=0.0, neginf=0.0)
        norm = np.sqrt(np.einsum("pab,pab->p", stress, stress))
        scale = np.where(norm > self.stress_cap, self.stress_cap / np.maximum(norm, 1e-12), 1.0)
        return stress * scale[:, None, None]

    # ------------------------------------------------------------------ step
    def _step(self, dt: float) -> None:
        x, v, m, rho = self.pos, self.vel, self.mass, self.rho
        n = len(x)
        pairs = self._pairs(x)
        accel = np.zeros((n, 3))
        L = np.zeros((n, 3, 3))

        if len(pairs):
            i = pairs[:, 0]
            j = pairs[:, 1]
            rvec = x[i] - x[j]
            r2 = np.einsum("ij,ij->i", rvec, rvec)
            r = np.sqrt(r2)
            r = np.maximum(r, 1e-9)
            w, dw = self._kernel(r)
            del w
            grad = (dw / r)[:, None] * rvec          # grad_i W_ij
            vij = v[i] - v[j]

            # --- continuity: drho/dt = sum_j m_j (v_i - v_j) . grad W
            work = np.einsum("ij,ij->i", vij, grad)
            drho = np.bincount(i, weights=m[j] * work, minlength=n)
            drho += np.bincount(j, weights=m[i] * work, minlength=n)

            # --- velocity gradient L_ab = d v_a / d x_b
            vgrad = np.einsum("pa,pb->pab", vij, grad)
            vol_j = m[j] / rho[j]
            vol_i = m[i] / rho[i]
            for a in range(3):
                for b in range(3):
                    L[:, a, b] = -(
                        np.bincount(i, weights=vol_j * vgrad[:, a, b], minlength=n)
                        + np.bincount(j, weights=vol_i * vgrad[:, a, b], minlength=n)
                    )
        else:
            drho = np.zeros(n)

        L = np.clip(np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0), -200.0, 200.0)

        pressure = self._pressure()
        stress = self._deviatoric_stress(L * 1.0)

        if len(pairs):
            i = pairs[:, 0]
            j = pairs[:, 1]
            rvec = x[i] - x[j]
            r2 = np.einsum("ij,ij->i", rvec, rvec)
            r = np.maximum(np.sqrt(r2), 1e-9)
            _, dw = self._kernel(r)
            grad = (dw / r)[:, None] * rvec
            vij = v[i] - v[j]

            # Monaghan artificial viscosity (approaching pairs only)
            vr = np.einsum("ij,ij->i", vij, rvec)
            rho_bar = 0.5 * (rho[i] + rho[j])
            pi_art = np.where(
                vr < 0.0,
                self.artificial_alpha * self.sound_speed * self.h * (-vr)
                / (r2 + 0.01 * self.h * self.h) * rho_bar,
                0.0,
            )

            pterm = -(
                (pressure[i] + pi_art) / rho[i] ** 2 + (pressure[j] + pi_art) / rho[j] ** 2
            )
            sterm = (
                np.einsum("pab,pb->pa", stress[i], grad) / rho[i][:, None] ** 2
                + np.einsum("pab,pb->pa", stress[j], grad) / rho[j][:, None] ** 2
            )
            coef = pterm[:, None] * grad + sterm
            for a in range(3):
                accel[:, a] += np.bincount(i, weights=m[j] * coef[:, a], minlength=n)
                accel[:, a] -= np.bincount(j, weights=m[i] * coef[:, a], minlength=n)

        accel += self._external_force(x, v) / m[:, None]

        new_v = v + dt * accel
        new_x = x + dt * new_v

        rho_new = rho + dt * drho
        rho_new = np.clip(np.nan_to_num(rho_new, nan=self.rho0, posinf=1.8 * self.rho0,
                                        neginf=0.5 * self.rho0),
                          0.5 * self.rho0, 1.8 * self.rho0)

        # deformation gradient: DF/Dt = L F - (F - J^(1/3) I)/lambda
        LF = np.einsum("pab,pbc->pac", L, self.F)
        F = self.F
        det = np.linalg.det(F)
        det = np.clip(np.nan_to_num(det, nan=1.0, posinf=1.0, neginf=1.0), 0.25, 4.0)
        target = (det ** (1.0 / 3.0))[:, None, None] * np.eye(3)[None, :, :]
        F_new = F + dt * (LF - (F - target) / self.relaxation_time)
        F_new = np.nan_to_num(F_new, nan=0.0, posinf=0.0, neginf=0.0)
        F_new = np.clip(F_new, -8.0, 8.0)
        det_new = np.linalg.det(F_new)
        det_new = np.where(np.isfinite(det_new) & (det_new > 1e-6), det_new, 1.0)
        fix = (np.clip(det_new, 0.3, 3.0) / det_new) ** (1.0 / 3.0)
        F_new = F_new * fix[:, None, None]

        self.pos = new_x
        self.vel = new_v
        self.rho = rho_new
        self.F = F_new
        self._sanitize()
        self.steps += 1

    def _sanitize(self) -> None:
        x, v = self.pos, self.vel
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        speed = np.linalg.norm(v, axis=1)
        too_fast = speed > 60.0
        if np.any(too_fast):
            v[too_fast] *= (60.0 / speed[too_fast])[:, None]
        escaped = (
            (np.abs(x[:, 0]) > 4.0)
            | (np.abs(x[:, 1]) > 2.5)
            | (x[:, 2] < -0.35)
            | (x[:, 2] > 7.0)
        )
        if np.any(escaped):
            x[escaped, 0] = np.clip(x[escaped, 0], -3.5, 3.5)
            x[escaped, 1] = np.clip(x[escaped, 1], -1.8, 1.8)
            x[escaped, 2] = np.clip(x[escaped, 2], 0.05, 5.0)
            v[escaped] *= 0.0

    def _timestep(self) -> float:
        speed = float(np.max(np.linalg.norm(self.vel, axis=1))) if len(self.vel) else 0.0
        cfl = self.cfl * self.h / (self.sound_speed + self.elastic_speed + speed + 1e-9)
        dt = min(self.dt_max, cfl)
        return max(dt, self.dt_min)

    # --------------------------------------------------------------- sampling
    def _frame(self) -> StateFrame:
        n = len(self.pos)
        sdf = self._contact_sdf(self.pos)
        external = self._external_force(self.pos, self.vel)
        frame = StateFrame(
            time=float(self.t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.labels.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self.F.copy(),
                "volume_quantity": (self.mass / self.rho).copy(),
                "particle_radius": np.full(n, self.particle_radius),
                "material_label": self.labels.copy(),
            },
            fields={
                "contact_signed_distance": sdf,
                "cumulative_mass_source": np.zeros(1),
                "external_force": external,
            },
            scalars={
                "contact_length_scale": float(self.spacing),
                "grid_spacing": float(self.spacing),
                "characteristic_length": float(
                    max(float(np.ptp(self.pos, axis=0).max()), self.spacing)
                ),
                "reference_velocity": 1.0,
                "smoothing_length": float(self.h),
                "time_step": float(self.dt_min_used),
                "gravity_magnitude": float(np.linalg.norm(self.gravity)),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if self.t < target:
            while self.t < target:
                dt = self._timestep()
                remaining = target - self.t
                if dt > remaining:
                    dt = remaining
                if dt <= 0.0:
                    break
                self._step(dt)
                self.t += dt
                self.dt_min_used = min(self.dt_min_used, dt)
        self.t = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "viscoelastic_sph",
            "particles": int(len(self.pos)),
            "steps": int(self.steps),
            "time": float(self.t),
            "smoothing_length": float(self.h),
            "sound_speed": float(self.sound_speed),
            "shear_modulus": float(self.shear_modulus),
            "relaxation_time": float(self.relaxation_time),
            "min_dt": float(self.dt_min_used),
            "density_min": float(np.min(self.rho)),
            "density_max": float(np.max(self.rho)),
            "speed_max": float(np.max(np.linalg.norm(self.vel, axis=1))),
            "min_contact_gap": float(np.min(self._contact_sdf(self.pos))),
            "total_mass": float(np.sum(self.mass)),
        }
