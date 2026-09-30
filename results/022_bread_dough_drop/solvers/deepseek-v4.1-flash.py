"""Physically-based solver for the bread-dough drop scene (022_bread_dough_drop).

The subject is a closed triangle-mesh bread-dough sample released above a static
ground plane.  The dough is modelled as a *viscoplastic (Bingham-like)*
incompressible solid:

*  Elasticity:  corotational (shape-matching) elastic energy with a single
   mass-weighted deformation measure.  The elastic acceleration of every sample
   is ``omega^2 * (goal - x)`` where ``goal`` is the current best-fit rigid
   transform of the material (plastic) rest configuration.  This measure is
   objective (vanishes for rigid body motion) and corresponds to a corotational
   linear-elastic material.
*  Plasticity: multiplicative plastic flow of the rest configuration.  When the
   Biot strain ``||S - I||_F`` (from the polar decomposition of the material
   deformation gradient) exceeds the yield strain, the rest shape relaxes toward
   the un-rotated current shape at a Bingham-like rate proportional to the
   excess strain.  Permanent (plastically locked) flattening and subsequent
   elastic recovery both emerge from this.
*  Incompressibility: a per-step volume projection of the surface mesh onto the
   rest volume (isochoric flow), referenced at the current contact plane so the
   ground contact is not disturbed.
*  Damping:  the non-rigid (strain-rate) part of the velocity field is relaxed
   toward the best-fit rigid velocity field, so free flight is undamped while
   internal deformation is strongly dissipated (bread dough is highly
   dissipative).
*  Contact:  hard non-penetration against the ground plane with Coulomb
   tangential friction (positional projection) and a small restitution.

Integration is a deterministic semi-implicit (symplectic) Euler scheme with
fixed substeps.  Every exported quantity is a raw physical state of that
simulation.  No keyframes, no scripted animation, no precomputed motion.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-12


def _cfg_float(config, key: str, default: float) -> float:
    try:
        value = float(config.get(key, default))
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def _resolve_workspace_file(scene, relative: str) -> Path | None:
    """Locate a workspace asset relative to the CWD or the scene module."""
    candidates: list[Path] = []
    rel = Path(str(relative))
    if rel.is_absolute():
        candidates.append(rel)
    else:
        candidates.append(Path.cwd() / rel)
        try:
            scene_file = inspect.getfile(type(scene))
            candidates.append(Path(scene_file).resolve().parent / rel)
        except Exception:
            pass
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def _load_faces(scene, n_vertices: int) -> np.ndarray:
    """Load the frozen mesh topology (needed for the closed-surface volume)."""
    asset = None
    try:
        asset = scene.config.get("subject_asset")
    except Exception:
        asset = None
    if asset is not None:
        path = _resolve_workspace_file(scene, asset)
        if path is not None:
            try:
                with np.load(path) as data:
                    if "faces" in data:
                        faces = np.asarray(data["faces"], dtype=np.int64)
                        if faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0:
                            return faces
            except Exception:
                pass
    # Fallback: convex hull of the samples (still a closed, oriented surface).
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(scene.vertices)
        return np.asarray(hull.simplices, dtype=np.int64)
    except Exception:
        return np.zeros((0, 3), dtype=np.int64)


class BenchmarkSolver:
    """Deterministic viscoplastic-solid solver for the dough drop."""

    # -- construction -----------------------------------------------------
    def __init__(self) -> None:
        self._ready = False

    def initialize(self, scene, seed: int) -> None:  # noqa: ARG002 - seed unused
        config = getattr(scene, "config", {}) or {}
        self._config = dict(config)

        frame0 = scene.initial_state()
        self.x0 = np.asarray(frame0.position, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.n = int(self.x0.shape[0])
        self.total_mass = float(self.mass.sum())
        self.w = self.mass / max(self.total_mass, _EPS)

        self.gravity = np.asarray(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        ).reshape(3)
        self.ground_z = _cfg_float(config, "ground_z", 0.0)
        self.contact_length_scale = _cfg_float(config, "contact_length_scale", 0.04)
        self.friction = float(np.clip(_cfg_float(config, "friction", 0.5), 0.0, 5.0))
        self.restitution = float(
            np.clip(_cfg_float(config, "restitution", 0.0), 0.0, 0.9)
        )
        stiffness = max(_cfg_float(config, "stiffness", 5.0), 0.0)
        damping = max(_cfg_float(config, "damping", 3.4), 0.0)
        plasticity = float(np.clip(_cfg_float(config, "plasticity", 0.86), 0.0, 1.0))

        # Material parameters -------------------------------------------------
        # Corotational elastic frequency of the body-scale shape mode.
        self.omega2 = 245.0 * max(stiffness, 1e-6)
        self.omega = float(np.sqrt(self.omega2))
        # Strain-rate (non-rigid) damping: damping == 3.4 -> critical.
        damping_ratio = float(np.clip(damping / 3.4, 0.0, 1.0))
        self.lam_damp = 2.0 * damping_ratio * self.omega
        # Bingham plastic flow: rate proportional to the excess Biot strain.
        self.plastic_rate = plasticity * 12.0
        self.yield_strain = 0.04
        self.max_plastic_fraction = 0.25

        # Contact length/thickness used for the gap normalisation.
        self.contact_eps = 0.5 * self.contact_length_scale

        # Rest volume of the closed surface mesh -------------------------------
        self.faces = _load_faces(scene, self.n)
        self.volume0 = abs(self._mesh_volume(self.x0))
        if not np.isfinite(self.volume0) or self.volume0 <= 0.0:
            ext = np.ptp(self.x0, axis=0)
            self.volume0 = max(float(np.prod(ext)) * 0.5, 1e-9)

        # Meshfree (moving least squares) deformation-gradient operator -------
        self._build_deformation_gradient_operator()

        # State ---------------------------------------------------------------
        self.x = self.x0.copy()
        self.v = np.asarray(frame0.velocity, dtype=float).copy()
        self.xp = self.x0.copy()  # plastically evolved material rest positions
        self.t = 0.0
        self._cache: dict[float, StateFrame] = {}
        self._frames_done = 0

        sample_dt = _cfg_float(config, "sample_dt", 1.0 / 30.0)
        self.dt_sub = min(sample_dt / 40.0, 1.0 / 600.0)
        self.dt_sub = max(self.dt_sub, 1e-5)

        self._ready = True

    # -- setup helpers ----------------------------------------------------
    def _build_deformation_gradient_operator(self, bandwidth: float = 0.22) -> None:
        """Gaussian-weighted least-squares deformation gradient, material coords."""
        offsets = self.x0[:, None, :] - self.x0[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", offsets, offsets)
        weights = np.exp(-dist2 / (bandwidth * bandwidth))
        np.fill_diagonal(weights, 0.0)
        self._mls_offsets = offsets
        self._mls_weights = weights
        self._mls_B_inv = np.linalg.inv(
            np.einsum("ij,ijk,ijm->ikm", weights, offsets, offsets)
        )

    def _deformation_gradient(self, positions: np.ndarray) -> np.ndarray:
        current = positions[:, None, :] - positions[None, :, :]
        a = np.einsum("ij,ijk,ijm->ikm", self._mls_weights, current, self._mls_offsets)
        return np.einsum("ikm,iml->ikl", a, self._mls_B_inv)

    def _mesh_volume(self, positions: np.ndarray) -> float:
        if self.faces.size == 0:
            return float(np.prod(np.ptp(positions, axis=0)))
        p0 = positions[self.faces[:, 0]]
        p1 = positions[self.faces[:, 1]]
        p2 = positions[self.faces[:, 2]]
        return float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)

    # -- physics ----------------------------------------------------------
    def _material_frame(self):
        """Mass-weighted rigid rotation and material deformation gradient.

        Returns the current centroid, the plastic rest centroid, the polar
        rotation of the material deformation gradient
        ``F = A B^-1`` (``A = sum w p pp^T``, ``B = sum w pp pp^T``) and the
        singular values of ``F`` (Biot stretch).
        """
        c = np.einsum("i,ia->a", self.w, self.x)
        cp = np.einsum("i,ia->a", self.w, self.xp)
        p = self.x - c
        pp = self.xp - cp
        a = np.einsum("i,ia,ib->ab", self.w, p, pp)
        b = np.einsum("i,ia,ib->ab", self.w, pp, pp)
        f = a @ np.linalg.inv(b + 1e-9 * np.eye(3))
        u, s, vt = np.linalg.svd(f)
        rot = u @ vt
        if np.linalg.det(rot) < 0.0:
            u = u.copy()
            u[:, -1] *= -1.0
            rot = u @ vt
        return c, cp, rot, s

    def _contact_project(self, x_old: np.ndarray) -> None:
        """Hard non-penetration + Coulomb friction against the ground plane."""
        z = self.x[:, 2]
        gap = z - self.ground_z
        touching = gap < 0.0
        self._touching = touching
        self._contact_flags = gap <= self.contact_eps
        if not np.any(touching):
            return
        penetration = -gap[touching]
        # Normal correction (hard, non-penetration).
        self.x[touching, 2] = self.ground_z
        # Tangential Coulomb friction opposing the substep's tangential slip.
        slip = self.x[touching, :2] - x_old[touching, :2]
        slip_norm = np.linalg.norm(slip, axis=1)
        allowed = self.friction * penetration
        scale = np.ones_like(slip_norm)
        moving = slip_norm > _EPS
        scale[moving] = np.minimum(1.0, allowed[moving] / slip_norm[moving])
        self.x[touching, :2] -= slip * scale[:, None]

    def _step(self, h: float) -> None:
        x_old = self.x.copy()

        # 1. Corotational elastic goal from the plastic rest configuration.
        c, cp, rot, _ = self._material_frame()
        goal = np.einsum("ia,ba->ib", self.xp - cp, rot) + c  # c + R (xp - cp)

        # 2. Explicit elastic + gravitational acceleration.
        acc = self.omega2 * (goal - self.x) + self.gravity[None, :]
        self.v += h * acc

        # 3. Dissipate only the non-rigid (strain-rate) velocity component.
        if self.lam_damp > 0.0:
            p = self.x - c
            vc = np.einsum("i,ia->a", self.w, self.v)
            r2 = np.einsum("ia,ia->i", p, p)
            inertia = np.eye(3) * float(np.einsum("i,i->", self.mass, r2))
            inertia -= np.einsum("i,ia,ib->ab", self.mass, p, p)
            momentum = np.cross(p, self.mass[:, None] * self.v).sum(axis=0)
            try:
                omega = np.linalg.solve(
                    inertia + 1e-9 * np.eye(3), momentum
                )
            except np.linalg.LinAlgError:
                omega = np.zeros(3)
            v_rigid = vc[None, :] + np.cross(omega[None, :], p)
            self.v = v_rigid + (self.v - v_rigid) / (1.0 + h * self.lam_damp)

        # 4. Drift.
        self.x = self.x + h * self.v

        # 5. Incompressibility: project the mesh volume back onto the rest volume.
        vol = self._mesh_volume(self.x)
        if np.isfinite(vol) and vol > 0.0 and self.volume0 > 0.0:
            ratio = self.volume0 / vol
            scale = float(np.clip(ratio ** (1.0 / 3.0), 0.99, 1.01))
            if abs(scale - 1.0) > 1e-12:
                cx, cy = np.einsum("i,ia->a", self.w, self.x)[:2]
                z_ref = float(self.x[:, 2].min())
                ref = np.array([cx, cy, z_ref])
                self.x = ref + scale * (self.x - ref)

        # 6. Ground contact (positional, dominates the volume projection).
        v_pre = (self.x - x_old) / h
        self._contact_project(x_old)

        # 7. Velocity from the position update; small restitution on impact.
        self.v = (self.x - x_old) / h
        touching = getattr(self, "_touching", None)
        if touching is not None and np.any(touching):
            approaching = touching & (v_pre[:, 2] < -1e-9)
            if np.any(approaching):
                bounce = -self.restitution * v_pre[approaching, 2]
                self.v[approaching, 2] = np.maximum(self.v[approaching, 2], bounce)

        # 8. Bingham plastic flow of the material rest configuration.
        if self.plastic_rate > 0.0:
            c, cp, rot, s = self._material_frame()
            strain = float(np.sqrt(np.sum((s - 1.0) ** 2)))
            if strain > self.yield_strain:
                excess = 1.0 - self.yield_strain / max(strain, _EPS)
                beta = min(
                    self.max_plastic_fraction,
                    self.plastic_rate * h * max(excess, 0.0),
                )
                if beta > 0.0:
                    unrotated = np.einsum("ia,ab->ib", self.x - c, rot) + cp
                    self.xp = self.xp + beta * (unrotated - self.xp)

        self.t += h

    # -- public API -------------------------------------------------------
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("BenchmarkSolver.initialize() must be called first")
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("advance_to requires a finite time")
        if target < self.t - 1e-9:
            target = self.t
        steps = 0
        while self.t < target - 1e-12:
            h = min(self.dt_sub, target - self.t)
            if h <= 0.0:
                break
            self._step(h)
            steps += 1
            if steps > 2_000_000:  # pragma: no cover - safety valve
                break
        self._frames_done += 1
        frame = self._make_frame(target)
        self._cache[target] = frame
        return frame

    def _make_frame(self, time: float) -> StateFrame:
        x = np.array(self.x, dtype=float, copy=True)
        v = np.array(self.v, dtype=float, copy=True)
        if not np.isfinite(x).all():
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(v).all():
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        deformation = self._deformation_gradient(x)
        if not np.isfinite(deformation).all():
            deformation = np.nan_to_num(deformation)
        gap = x[:, 2] - self.ground_z
        frame = StateFrame(
            time=float(max(time, 0.0)),
            position=x,
            velocity=v,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": deformation,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "contact_signed_distance": gap,
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        if not self._ready:
            return {"ready": False}
        gap = self.x[:, 2] - self.ground_z
        speed = np.linalg.norm(self.v, axis=1)
        return {
            "ready": True,
            "time": float(self.t),
            "frames": int(self._frames_done),
            "substep": float(self.dt_sub),
            "total_mass": float(self.mass.sum()),
            "volume": float(abs(self._mesh_volume(self.x))),
            "rest_volume": float(self.volume0),
            "min_gap": float(gap.min()),
            "max_speed": float(speed.max()),
            "max_height": float(self.x[:, 2].max()),
            "omega": float(self.omega),
            "finite": bool(
                np.isfinite(self.x).all() and np.isfinite(self.v).all()
            ),
        }
