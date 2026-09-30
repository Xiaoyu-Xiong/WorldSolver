"""Physics solver for scene 058_pressurized_rubber_tube.

Finite-element model of the hollow rubber tube wall:

* The structured node grid of the scene (outer layer first, then inner layer,
  each layer ordered as z-ring major / theta minor) is split into hexahedral
  cells which are each decomposed into six tetrahedra by the conforming Kuhn
  decomposition.  This yields a genuinely three-dimensional volumetric
  discretization of the finite-thickness wall.
* Hyperelastic compressible neo-Hookean material (Simo-Pister form) evaluated
  per tetrahedron with exact first-Piola-Kirchhoff stress and consistent nodal
  forces.
* Internal cavity pressure is applied as a follower load on the triangulated
  cavity wall (pressure_surface nodes) using current-configuration areas and
  outward normals.
* Both end rims (fixed_rim nodes) are fully constrained.
* Explicit symplectic (semi-implicit) Euler time integration with a
  CFL-limited step size and mass-proportional damping; the pressure follows
  the frozen ramp/hold schedule from the scene.

Everything is computed from the frozen scene configuration; no animation,
keyframes or precomputed trajectories are used.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Tetrahedral FEM solver with explicit dynamics for the pressurized tube."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        cfg = dict(getattr(scene, "config", {}) or {})
        self._scene = scene
        self._seed = int(seed)

        x_rest = np.asarray(frame0.position, dtype=np.float64).copy()
        n_total = x_rest.shape[0]
        self._n = n_total
        self._x = x_rest.copy()
        self._v = np.zeros_like(x_rest)
        self._x_rest = x_rest

        geom = dict(cfg.get("geometry", {}) or {})
        self._n_theta = int(geom.get("circumferential_segments", 32))
        n_k = int(geom.get("axial_segments", 16)) + 1  # axial node rings
        self._n_k = n_k
        surface = self._n_theta * n_k
        if 2 * surface != n_total:
            raise ValueError("unexpected node layout for hollow tube grid")

        material = dict(cfg.get("material", {}) or {})
        density = float(material.get("density", 1070.0))
        self._young = float(material.get("youngs_modulus", 1.0e5))
        self._poisson = float(material.get("poisson_ratio", 0.42))
        self._mu = self._young / (2.0 * (1.0 + self._poisson))
        self._bulk = self._young / (3.0 * (1.0 - 2.0 * self._poisson))
        self._lam = self._bulk - 2.0 * self._mu / 3.0

        mass = np.asarray(frame0.mass, dtype=np.float64)
        if mass.shape != (n_total,) or not np.isfinite(mass).all() or np.any(mass <= 0):
            raise ValueError("node masses must be positive and finite")
        self._m = mass

        attrs = {k: np.asarray(v).copy() for k, v in frame0.attributes.items()}
        fixed = np.asarray(attrs.get("fixed_rim", np.zeros(n_total))).astype(np.int64)
        self._fixed = np.nonzero(fixed != 0)[0]
        self._attrs0 = attrs

        # --- tetrahedral mesh (Kuhn decomposition of each wall cell) --------
        def nid(radial: int, k: int, i: int) -> int:
            return radial * surface + k * self._n_theta + (i % self._n_theta)

        # monotone paths on the unit cube: coordinates are
        # (theta offset, z-ring offset, radial offset 0=outer 1=inner)
        paths = []
        for perm in permutations(range(3)):
            cur = [0, 0, 0]
            path = [tuple(cur)]
            for axis in perm:
                cur[axis] = 1
                path.append(tuple(cur))
            paths.append(path)

        tet_list = []
        for k in range(n_k - 1):
            for i in range(self._n_theta):
                for path in paths:
                    tet_list.append(
                        [nid(c, k + b, i + a) for (a, b, c) in path]
                    )
        self._tets = np.asarray(tet_list, dtype=np.int64)
        self._ne = self._tets.shape[0]

        # cavity wall triangles (inner layer), theta-wrapping quads split in two
        tri_list = []
        for k in range(n_k - 1):
            for i in range(self._n_theta):
                a0 = nid(1, k, i)
                a1 = nid(1, k, i + 1)
                a2 = nid(1, k + 1, i)
                a3 = nid(1, k + 1, i + 1)
                tri_list.append((a0, a1, a2))
                tri_list.append((a1, a3, a2))
        self._tris = np.asarray(tri_list, dtype=np.int64)

        # rest geometry of the tets
        xt = x_rest[self._tets]  # [Ne,4,3]
        dm = np.stack(
            [xt[:, 1] - xt[:, 0], xt[:, 2] - xt[:, 0], xt[:, 3] - xt[:, 0]], axis=2
        )  # columns are material edge vectors
        self._dm_inv = np.linalg.inv(dm)
        self._dm_inv_t = np.transpose(self._dm_inv, (0, 2, 1))
        self._v0 = np.abs(np.linalg.det(dm)) / 6.0

        # minimum tet altitude for the CFL estimate
        faces = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
        area_sum = np.zeros(self._ne)
        for (ia, ib, ic) in faces:
            area_sum += 0.5 * np.linalg.norm(
                np.cross(xt[:, ib] - xt[:, ia], xt[:, ic] - xt[:, ia]), axis=1
            )
        alt = 3.0 * self._v0 / np.maximum(area_sum, 1e-30)
        h_min = float(np.min(alt))

        wave_speed = np.sqrt((self._bulk + 4.0 * self._mu / 3.0) / density)
        self._dt = float(np.clip(0.3 * h_min / wave_speed, 1.0e-6, 4.0e-4))

        # --- loading schedule ------------------------------------------------
        pc = dict(cfg.get("boundary_conditions", {}).get("cavity_pressure", {}) or {})
        self._p_max = float(pc.get("pressure_pa", frame0.scalars.get("cavity_pressure_pa", 0.0)))
        self._ramp_start = float(pc.get("ramp_start", 0.0))
        self._ramp_end = float(pc.get("ramp_end", 0.3))
        self._hold_until = float(pc.get("hold_until", 1.0))
        self._pressure_fn = getattr(scene, "pressure_at", None)

        gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=np.float64)
        self._gravity = gravity

        # damping (mass-proportional Rayleigh), strong enough to settle the
        # slow quasi-static response, weak compared with elastic wave speeds
        self._cd = 50.0

        # reference force scale: full design pressure acting on the cavity wall
        inner_r = float(geom.get("inner_radius", 0.4))
        height = float(geom.get("height", 1.6))
        self._ref_force = float(
            max(abs(self._p_max) * 2.0 * np.pi * inner_r * height, 1.0)
        )

        self._t = 0.0
        self._steps = 0
        self._ready = True

    # ------------------------------------------------------------- pressure
    def _pressure(self, time: float) -> float:
        if self._pressure_fn is not None:
            return float(self._pressure_fn(time))
        fraction = np.clip(
            (time - self._ramp_start) / max(self._ramp_end - self._ramp_start, 1e-12),
            0.0,
            1.0,
        )
        if time > self._hold_until:
            fraction = 1.0
        return float(self._p_max) * float(fraction)

    # --------------------------------------------------------- elastic forces
    def _elastic_forces(self, x: np.ndarray):
        """Nodal elastic forces and per-element deformation gradients."""
        ne = self._ne
        x0 = x[self._tets[:, 0]]
        ds = np.empty((ne, 3, 3))
        for j in (1, 2, 3):
            ds[:, :, j - 1] = x[self._tets[:, j]] - x0
        f = ds @ self._dm_inv  # deformation gradients [Ne,3,3]
        det_f = np.linalg.det(f)
        try:
            f_inv_t = np.linalg.inv(f).transpose(0, 2, 1)
        except np.linalg.LinAlgError:
            f_inv_t = np.linalg.pinv(f).transpose(0, 2, 1)
        log_j = np.log(np.clip(det_f, 1.0e-12, None))
        # Simo-Pister compressible neo-Hookean first Piola-Kirchhoff stress
        p_stress = self._mu * (f - f_inv_t) + (self._lam * log_j)[:, None, None] * f_inv_t
        g = (self._v0[:, None, None] * p_stress) @ self._dm_inv_t  # [Ne,3,3]

        w = np.empty((ne, 4, 3))
        w[:, 0, :] = g[:, 0, :] + g[:, 1, :] + g[:, 2, :]
        w[:, 1, :] = -g[:, 0, :]
        w[:, 2, :] = -g[:, 1, :]
        w[:, 3, :] = -g[:, 2, :]

        idx = self._tets.ravel()
        forces = np.zeros((self._n, 3))
        for c in range(3):
            forces[:, c] = np.bincount(
                idx, weights=w[:, :, c].ravel(), minlength=self._n
            )
        return forces, f

    # -------------------------------------------------------- pressure forces
    def _pressure_forces(self, x: np.ndarray, pressure: float) -> np.ndarray:
        forces = np.zeros((self._n, 3))
        if pressure == 0.0:
            return forces
        a = x[self._tris[:, 0]]
        b = x[self._tris[:, 1]]
        c = x[self._tris[:, 2]]
        cross = np.cross(b - a, c - a)
        twice_area = np.linalg.norm(cross, axis=1)
        normal = cross / np.maximum(twice_area, 1e-30)[:, None]
        centroid = (a + b + c) / 3.0
        # follower load: normal points away from the cavity axis (into the wall)
        flip = (normal[:, 0] * centroid[:, 0] + normal[:, 1] * centroid[:, 1]) < 0.0
        normal[flip] *= -1.0
        tri_force = (pressure * twice_area * 0.5 / 3.0)[:, None] * normal
        idx = self._tris.ravel()
        for comp in range(3):
            forces[:, comp] = np.bincount(
                idx, weights=tri_force[:, comp], minlength=self._n
            )
        return forces

    # ---------------------------------------------------------------- stepping
    def _step(self, h: float) -> None:
        t_new = self._t + h
        pressure = self._pressure(0.5 * (self._t + t_new))
        f_int, _ = self._elastic_forces(self._x)
        f_pre = self._pressure_forces(self._x, pressure)
        f_tot = f_int + f_pre + self._gravity[None, :] * self._m[:, None]
        accel = f_tot / self._m[:, None] - self._cd * self._v
        v_new = self._v + h * accel
        x_new = self._x + h * v_new
        # constrained rims
        v_new[self._fixed] = 0.0
        x_new[self._fixed] = self._x_rest[self._fixed]
        if not (np.isfinite(x_new).all() and np.isfinite(v_new).all()):
            raise RuntimeError("non-finite state encountered during time stepping")
        self._x = x_new
        self._v = v_new
        self._t = t_new
        self._steps += 1

    # ------------------------------------------------------------------ frames
    def _make_frame(self, time: float) -> StateFrame:
        x = self._x
        f_int, def_grad = self._elastic_forces(x)
        pressure = self._pressure(time)
        f_pre = self._pressure_forces(x, pressure)
        f_tot = f_int + f_pre + self._gravity[None, :] * self._m[:, None]

        # reaction needed at the constrained rims
        rim_reaction = -f_tot[self._fixed].sum(axis=0)
        pressure_resultant = f_pre.sum(axis=0)
        gravity_total = self._gravity * float(self._m.sum())
        budget = np.array([[pressure_resultant, gravity_total, rim_reaction]], dtype=np.float64)

        attributes = {k: v.copy() for k, v in self._attrs0.items()}
        attributes["mass_quantity"] = self._m.copy()

        scalars = {
            "cavity_pressure_pa": float(pressure),
            "reference_force": float(self._ref_force),
            "youngs_modulus": float(self._young),
            "poisson_ratio": float(self._poisson),
            "time_step": float(self._dt),
        }
        frame = StateFrame(
            time=float(time),
            position=x.copy(),
            velocity=self._v.copy(),
            mass=self._m.copy(),
            attributes=attributes,
            fields={
                "budget_external_force": budget,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "element_deformation_gradient": def_grad.astype(np.float64, copy=True),
                "external_force": f_pre.copy(),
            },
            scalars=scalars,
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- interface
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("solver is not initialized")
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("time must be finite and non-negative")
        if target < self._t - 1.0e-9:
            raise ValueError("advance_to cannot move backwards in time")
        guard = 0
        while self._t < target - 1.0e-12:
            h = min(self._dt, target - self._t)
            self._step(h)
            guard += 1
            if guard > 4_000_000:
                raise RuntimeError("time stepping failed to reach sample time")
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        if not self._ready:
            return {"success": False}
        x = self._x
        radius = np.linalg.norm(x[:, :2], axis=1)
        free = np.ones(self._n, dtype=bool)
        free[self._fixed] = False
        return {
            "success": True,
            "time": float(self._t),
            "time_step": float(self._dt),
            "steps": int(self._steps),
            "nodes": int(self._n),
            "tetrahedra": int(self._ne),
            "fixed_nodes": int(len(self._fixed)),
            "youngs_modulus": float(self._young),
            "poisson_ratio": float(self._poisson),
            "max_radius_all": float(radius.max()),
            "max_radius_free": float(radius[free].max()),
            "initial_outer_radius": float(np.linalg.norm(self._x_rest[:, :2], axis=1).max()),
            "final_cavity_pressure_pa": float(self._pressure(self._t)),
        }
