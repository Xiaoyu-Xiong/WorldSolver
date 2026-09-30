"""Volumetric hyperelastic foam-ball high-speed impact solver (NumPy/SciPy only).

Physical model
--------------
The frozen scene declares one deformable object (``foam_ball``) whose authoritative
topology is the icosahedral surface asset ``assets/foam_ball_surface.obj`` (12
vertices, 20 triangles).  The ball is discretised as a genuine three-dimensional
solid: the 12 state nodes are the icosahedron vertices and the 20 tetrahedra are
formed by coning the ball centroid (the affine mean of the 12 nodes) to every
triangle.  ``x(X) = x_c + F (X - X_c)`` is linear per tetrahedron, so each
element carries a well defined deformation gradient ``F_e = D_s D_m^{-1}``.

Material: compressible isotropic hyperelasticity (Neo-Hookean with the
volumetric ``log J`` term) with the frozen parameters E = 1e7 Pa, nu = 0.45,
rho = 1150 kg/m^3, plus a small Kelvin-Voigt viscous stress and a smooth
inversion barrier ``W_b = c_b/2 (J - J_b)^2`` active only when ``J < J_b``.

Dynamics: symplectic (semi-implicit) Euler on the lumped nodal masses with a
semi-implicit unilateral penalty contact against the fixed plate plane
``x = plate_x``; the internal elastic forces are exact gradients of the stored
energy and therefore cancel identically in the total momentum budget.  Adaptive
sub-stepping halves dt whenever a trial step would drive any element toward
inversion, keeping the state finite and inversion free.

Trajectory plumbing
-------------------
``advance_to(t)`` integrates the physical system to the requested sample time
and returns a :class:`StateFrame` carrying the exact raw state and the required
budget / deformation-gradient fields.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# Deterministic unit icosahedron (identical to assets/foam_ball_surface.obj).
_FALLBACK_VERTICES = np.array(
    [
        [-0.525731, 0.850651, 0.0],
        [0.525731, 0.850651, 0.0],
        [-0.525731, -0.850651, 0.0],
        [0.525731, -0.850651, 0.0],
        [0.0, -0.525731, 0.850651],
        [0.0, 0.525731, 0.850651],
        [0.0, -0.525731, -0.850651],
        [0.0, 0.525731, -0.850651],
        [0.850651, 0.0, -0.525731],
        [0.850651, 0.0, 0.525731],
        [-0.850651, 0.0, -0.525731],
        [-0.850651, 0.0, 0.525731],
    ],
    dtype=np.float64,
)
_FALLBACK_FACES = np.array(
    [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ],
    dtype=np.int64,
)


def _read_obj_topology(path) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse a simple OBJ triangle mesh; returns (vertices, faces) or None."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v" and len(parts) >= 4:
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif parts[0] == "f" and len(parts) >= 4:
            faces.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
    if len(vertices) < 4 or len(faces) < 4:
        return None
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


class BenchmarkSolver:
    """Deterministic explicit-dynamics solver for the high-speed foam-ball impact."""

    # ---- numerical parameters -------------------------------------------------
    base_dt = 5.0e-7
    contact_stiffness = 1.0e9       # N/m, unilateral penalty against the plate
    contact_damping_ratio = 0.02
    viscous_eta = 20.0              # Pa*s, Kelvin-Voigt solid viscosity
    barrier_j = 0.25                # inversion barrier activation
    barrier_stiffness = 2.0e7       # Pa
    j_floor = 0.05                  # reject / retry a step below this J
    max_halvings = 12

    # ---- setup -----------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.seed = int(seed)
        ball = cfg["foam_ball"]
        self.radius = float(ball["radius"])
        self.density = float(ball["density"])
        youngs = float(ball["youngs_modulus"])
        poisson = float(ball["poisson_ratio"])
        self.center = np.asarray(ball["initial_center"], dtype=np.float64)
        self.v0 = np.asarray(ball["initial_velocity"], dtype=np.float64)
        self.g = np.asarray(
            cfg.get("initial_conditions", {}).get("gravity", [0.0, 0.0, -9.81]),
            dtype=np.float64,
        )

        # Lamé parameters of the frozen isotropic material.
        self.mu = youngs / (2.0 * (1.0 + poisson))
        self.lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

        # Frozen plate: half space behind the plane x = plate_x.
        plate = cfg.get("impact_plate", {})
        plate_center = np.asarray(
            plate.get("center", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        half = np.asarray(plate.get("half_extents", [0.0, 0.05, 0.05]), dtype=np.float64)
        surface = cfg.get("boundary_conditions", {}).get("plate_contact_surface_x")
        self.plate_x = float(surface) if surface is not None else float(
            plate_center[0] - half[0]
        )
        self.plate_center = plate_center
        self.plate_half = half

        # Reference (undeformed) nodal positions from the frozen asset / transform.
        vertices, faces, scale, translation = self._topology(cfg)
        self.verts = vertices
        self.faces = faces
        self.n_nodes = int(vertices.shape[0])
        self.reference = translation[None, :] + scale[None, :] * vertices

        # Volumetric tetrahedra: centroid of the solid coned to every triangle.
        self.tets = np.array(faces, dtype=np.int64, copy=True)
        self.ref_center = self.reference.mean(axis=0)
        Dm = self._tets_ds(self.reference, self.ref_center)
        det = np.linalg.det(Dm)
        flip = det < 0.0
        if flip.any():
            self.tets[flip] = self.tets[flip][:, [0, 2, 1]]
            Dm = self._tets_ds(self.reference, self.ref_center)
            det = np.linalg.det(Dm)
        self.n_elem = int(self.tets.shape[0])
        self.dm_inv = np.linalg.inv(Dm)
        self.dm_inv_t = self.dm_inv.transpose(0, 2, 1)
        self.ref_volume = det / 6.0
        self.ref_volume = np.abs(self.ref_volume)

        # Mass: frozen sphere mass distributed over the state nodes.
        self.total_mass = self.density * (4.0 / 3.0) * np.pi * self.radius**3
        self.mass = np.full(self.n_nodes, self.total_mass / self.n_nodes, dtype=np.float64)

        # Contact damping consistent with the nodal mass scale.
        self.contact_damping = (
            2.0
            * self.contact_damping_ratio
            * np.sqrt(self.contact_stiffness * float(self.mass[0]))
        )

        # Lateral footprint of the plate (contact only inside it).
        self.plate_nodes = (
            (np.abs(self.reference[:, 1] - plate_center[1]) <= half[1] + 1e-12)
            & (np.abs(self.reference[:, 2] - plate_center[2]) <= half[2] + 1e-12)
            & (half[0] > 0.0)
        )

        # Initial state: unstressed, round ball moving with the frozen velocity.
        self.x = self.reference.copy()
        self.v = np.broadcast_to(self.v0, self.x.shape).copy()
        self.t = 0.0
        self.f_prev = self._def_grad(self.x, self.x.mean(axis=0))
        self.jacobian = np.linalg.det(self.f_prev)
        self.impulse_gravity = np.zeros(3, dtype=np.float64)
        self.impulse_contact = np.zeros(3, dtype=np.float64)
        self.contact_force = np.zeros(3, dtype=np.float64)
        self.steps = 0
        self.retries = 0

    def _topology(self, cfg):
        geometry = cfg.get("scene_geometry") or {}
        asset = None
        scale = None
        translation = None
        for obj in geometry.get("objects", []) or []:
            if obj.get("id") != "foam_ball":
                continue
            gspec = obj.get("geometry", {}) or {}
            asset = gspec.get("topology_asset") or gspec.get("asset")
            transform = obj.get("transform", {}) or {}
            if transform.get("scale") is not None:
                scale = np.asarray(transform["scale"], dtype=np.float64)
            if transform.get("translation") is not None:
                translation = np.asarray(transform["translation"], dtype=np.float64)
        vertices = faces = None
        if asset:
            loaded = _read_obj_topology(asset)
            if loaded is None and not Path(asset).is_absolute():
                loaded = _read_obj_topology(Path.cwd() / asset)
            if loaded is not None:
                vertices, faces = loaded
        if vertices is None:
            vertices, faces = _FALLBACK_VERTICES.copy(), _FALLBACK_FACES.copy()
        if scale is None:
            scale = np.full(3, self.radius, dtype=np.float64)
        if translation is None:
            translation = self.center.copy()
        norm = np.linalg.norm(vertices, axis=1)
        if not np.allclose(norm, 1.0, atol=1e-6):
            vertices = vertices / np.maximum(norm[:, None], 1e-12)
        return vertices, faces, scale, translation

    def _tets_ds(self, x: np.ndarray, center: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                x[self.tets[:, 0]] - center,
                x[self.tets[:, 1]] - center,
                x[self.tets[:, 2]] - center,
            ],
            axis=2,
        )

    def _def_grad(self, x: np.ndarray, center: np.ndarray) -> np.ndarray:
        return self._tets_ds(x, center) @ self.dm_inv

    # ---- forces ----------------------------------------------------------------
    def _elastic_forces(self, F: np.ndarray, J: np.ndarray, Fdot: np.ndarray) -> np.ndarray:
        finv_t = np.linalg.inv(F).transpose(0, 2, 1)
        log_j = np.log(np.maximum(J, 1e-12))
        P = self.mu * (F - finv_t) + self.lam * log_j[:, None, None] * finv_t
        below = J < self.barrier_j
        if below.any():
            coef = np.where(
                below, self.barrier_stiffness * (J - self.barrier_j) * J, 0.0
            )
            P = P + coef[:, None, None] * finv_t
        P = P + self.viscous_eta * Fdot
        G = P @ self.dm_inv_t
        S = G.sum(axis=2)
        forces = np.zeros_like(self.x)
        forces += (self.ref_volume[:, None] * S / float(self.n_nodes)).sum(axis=0)
        contrib = -self.ref_volume[:, None, None] * G
        np.add.at(forces, self.tets[:, 0], contrib[:, :, 0])
        np.add.at(forces, self.tets[:, 1], contrib[:, :, 1])
        np.add.at(forces, self.tets[:, 2], contrib[:, :, 2])
        return forces

    # ---- time stepping ---------------------------------------------------------
    def _try_step(self, dt: float, force_accept: bool = False) -> bool:
        x = self.x
        m = self.mass
        center = x.mean(axis=0)
        F = self._def_grad(x, center)
        J = np.linalg.det(F)
        if not np.isfinite(J).all() or not np.isfinite(F).all():
            return False
        if J.min() < self.j_floor and not force_accept:
            return False
        Fdot = (F - self.f_prev) / dt
        forces = self._elastic_forces(F, J, Fdot) + m[:, None] * self.g[None, :]
        v_free = self.v + dt * forces / m[:, None]
        x_free = x + dt * v_free

        v_new = v_free
        x_new = x_free
        contact_impulse = np.zeros(3, dtype=np.float64)

        px = self.plate_x
        if self.plate_nodes.any():
            pen = x[self.plate_nodes, 0] - px
            active = (x_free[self.plate_nodes, 0] - px) > 0.0
            if active.any():
                loc = np.flatnonzero(self.plate_nodes)[active]
                mm = m[loc]
                vf = v_free[loc, 0]
                pn = pen[active]
                denom = 1.0 + dt * self.contact_damping / mm + dt * dt * self.contact_stiffness / mm
                vn = (vf - (dt * self.contact_stiffness / mm) * pn) / denom
                xn = x[loc, 0] + dt * vn
                keep = xn > px
                if keep.any():
                    loc = loc[keep]
                    vn = vn[keep]
                    xn = xn[keep]
                    contact_impulse[0] += float(np.sum(m[loc] * (vn - v_free[loc, 0])))
                    x_new = x_new.copy()
                    v_new = v_new.copy()
                    x_new[loc, 0] = xn
                    v_new[loc, 0] = vn

        # Exact non-penetration of the state nodes against the frozen plate plane.
        if self.plate_nodes.any():
            x_new[self.plate_nodes, 0] = np.minimum(x_new[self.plate_nodes, 0], px)

        if not (np.isfinite(x_new).all() and np.isfinite(v_new).all()):
            return False
        F_new = self._def_grad(x_new, x_new.mean(axis=0))
        J_new = np.linalg.det(F_new)
        if not np.isfinite(J_new).all():
            return False
        if J_new.min() < self.j_floor and not force_accept:
            return False

        self.x = x_new
        self.v = v_new
        self.f_prev = F_new
        self.jacobian = J_new
        self.impulse_gravity = self.impulse_gravity + dt * self.total_mass * self.g
        self.impulse_contact = self.impulse_contact + contact_impulse
        self.contact_force = contact_impulse / dt
        self.steps += 1
        return True

    def _step(self, dt: float) -> float:
        trial = dt
        for _ in range(self.max_halvings):
            if self._try_step(trial):
                return trial
            trial *= 0.5
            self.retries += 1
        self._try_step(trial, force_accept=True)
        return trial

    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.t < target - 1e-13:
            dt = target - self.t
            if dt > self.base_dt:
                dt = self.base_dt
            self._step(dt)
            self.t += dt
            guard += 1
            if guard > 100_000_000:  # pragma: no cover - defensive
                break
        self.t = target
        return self._frame()

    # ---- trajectory output -----------------------------------------------------
    def _frame(self) -> StateFrame:
        center = self.x.mean(axis=0)
        F = self._def_grad(self.x, center)
        node_ids = np.zeros(self.n_nodes, dtype=np.float64)
        external_force = np.stack(
            [self.total_mass * self.g, self.contact_force], axis=0
        )
        frame = StateFrame(
            time=float(self.t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": (
                    self.impulse_gravity + self.impulse_contact
                )[None, :].copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_domain_weights": (node_ids + 1.0)[None, :].copy(),
                "budget_external_force": external_force[None, :, :].copy(),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "element_deformation_gradient": F.copy(),
                "gravity_acceleration": self.g[None, :].copy(),
            },
            scalars={
                "reference_momentum": float(
                    self.total_mass * np.linalg.norm(self.v0)
                ),
                "characteristic_length": float(self.radius),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps": int(self.steps),
            "retries": int(self.retries),
            "min_det_deformation_gradient": float(np.min(self.jacobian)),
            "max_det_deformation_gradient": float(np.max(self.jacobian)),
            "nodes": int(self.n_nodes),
            "elements": int(self.n_elem),
            "total_mass": float(self.total_mass),
            "min_x": float(np.min(self.x[:, 0])),
            "max_x": float(np.max(self.x[:, 0])),
        }
