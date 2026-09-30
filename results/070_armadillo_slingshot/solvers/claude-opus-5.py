"""Explicit finite-element elastodynamics solver for the Armadillo slingshot scene.

Physical model
--------------
* Volumetric linear tetrahedral finite elements built from the frozen topology
  asset (``assets/armadillo_solid.npz``).  Degenerate cells produced by the
  polar caps of the surface asset are detected and deactivated.
* Stable Neo-Hookean elasticity (Smith et al. style regularisation),

      Psi(F) = mu/2 (tr(F^T F) - 3) - mu (J - 1) + lam/2 (J - 1)^2
      P(F)   = mu F + (lam (J - 1) - mu) * dJ/dF ,

  with Lame parameters obtained from the frozen material metadata
  (E = 45 kPa, nu = 0.45, rho = 1000 kg/m^3).
* Kelvin-Voigt strain-rate viscosity (internal, momentum conserving) plus a
  light linear medium drag (external, reported in the momentum budget).
* Bonded interface springs weld the individually meshed body parts of the
  frozen asset into one connected volumetric body.  The weld radius is the
  smallest of a deterministic ladder that makes the assembly simply connected.
* Boundary conditions: the two palm grasp patches are held fixed; the temporary
  pulled patch is kinematically driven by a C2 quintic ramp of the frozen
  loading target displacement and is fully removed at ``release_s``.
* Time stepping: mass-lumped symplectic (semi-implicit) Euler with a stability
  limited step size derived from element altitudes and nodal weld stiffness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

_ROOT = Path(__file__).resolve().parent.parent
_SOLID = _ROOT / "assets" / "armadillo_solid.npz"


def _quintic(u: float) -> tuple[float, float, float]:
    """C2 smooth ramp s(u), s'(u), s''(u) on [0, 1] (zero end velocity/accel)."""
    if u <= 0.0:
        return 0.0, 0.0, 0.0
    if u >= 1.0:
        return 1.0, 0.0, 0.0
    s = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)
    ds = 30.0 * u * u * (1.0 - 2.0 * u + u * u)
    dds = 60.0 * u * (1.0 - 3.0 * u + 2.0 * u * u)
    return s, ds, dds


def _min_altitude(tets: np.ndarray, x: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Smallest vertex-to-opposite-face distance of each tetrahedron."""
    combos = ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2))
    out = np.full(len(tets), np.inf)
    for a, b, c, _d in combos:
        area = 0.5 * np.linalg.norm(
            np.cross(x[tets[:, b]] - x[tets[:, a]], x[tets[:, c]] - x[tets[:, a]]), axis=1
        )
        out = np.minimum(out, 3.0 * vol / np.maximum(area, 1e-14))
    return out


class BenchmarkSolver:
    """Solver-side implementation of the elastic slingshot experiment."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config
        frame0 = scene.initial_state()

        self.x0 = np.ascontiguousarray(np.asarray(frame0.position, dtype=np.float64))
        self.n = int(len(self.x0))
        self.x = self.x0.copy()
        self.v = np.ascontiguousarray(np.asarray(frame0.velocity, dtype=np.float64)).copy()

        gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)
        self.gravity = gravity.reshape(3)

        material = dict(config.get("material_reference", {}) or {})
        self.rho = float(material.get("density_kg_m3", 1000.0))
        youngs = float(material.get("youngs_modulus_pa", 45000.0))
        poisson = float(material.get("poisson_ratio", 0.45))
        self.mu = youngs / (2.0 * (1.0 + poisson))
        self.lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

        self._build_mesh()
        self._build_boundary(frame0, config)
        self._build_welds()
        self._choose_timestep(config)

        # Momentum bookkeeping (all external actions are continuous forces, so
        # the discrete-impulse and mass-transfer channels are exactly zero).
        self.zero_g3 = np.zeros((1, 3), dtype=np.float64)
        self.domain_weights = np.ones((1, self.n), dtype=np.float64)
        pull_delta = float(np.linalg.norm(self.pull_vector))
        ramp = max(self.pull_peak - self.pull_start, 1e-6)
        self.v_char = max(pull_delta / ramp, 1e-3)
        self.reference_momentum = float(self.total_mass * self.v_char)
        self.time = 0.0
        self._retries = 0
        self._substeps_used = 0

    # -------------------------------------------------------------- mesh data
    def _build_mesh(self) -> None:
        data = np.load(_SOLID)
        tets = np.asarray(data["tetrahedra"], dtype=np.int64)
        tets = tets[(tets >= 0).all(axis=1) & (tets < self.n).all(axis=1)]
        self.n_elements_total = int(len(tets))
        self.tets_all = tets

        x = self.x0
        edges = np.stack(
            [x[tets[:, 1]] - x[tets[:, 0]], x[tets[:, 2]] - x[tets[:, 0]], x[tets[:, 3]] - x[tets[:, 0]]],
            axis=2,
        )
        det = np.linalg.det(edges)
        vol = np.abs(det) / 6.0
        active = vol > 1e-8
        self.active = active

        tets = tets.copy()
        flip = active & (det < 0.0)
        tets[flip] = tets[flip][:, [0, 2, 1, 3]]
        self.tets_all = tets
        self.tets = np.ascontiguousarray(tets[active])

        edges = np.stack(
            [
                self.x0[self.tets[:, 1]] - self.x0[self.tets[:, 0]],
                self.x0[self.tets[:, 2]] - self.x0[self.tets[:, 0]],
                self.x0[self.tets[:, 3]] - self.x0[self.tets[:, 0]],
            ],
            axis=2,
        )
        self.vol0 = np.abs(np.linalg.det(edges)) / 6.0
        self.Bm = np.linalg.inv(edges)
        self.ne = int(len(self.tets))
        self.scatter_idx = self.tets.reshape(-1)

        mass = np.zeros(self.n, dtype=np.float64)
        node_mass = np.repeat(self.vol0 * self.rho / 4.0, 4)
        np.add.at(mass, self.scatter_idx, node_mass)
        floor = 0.1 * mass.sum() / max(self.n, 1)
        self.mass = np.maximum(mass, floor)
        self.total_mass = float(self.mass.sum())
        self.inv_mass = 1.0 / self.mass
        self.mass3 = self.mass[:, None]
        self.h_min = float(np.min(_min_altitude(self.tets, self.x0, self.vol0)))

        # per-part label (each connected tet component of the frozen asset)
        rows, cols = [], []
        for i in range(4):
            for j in range(i + 1, 4):
                rows.append(self.tets[:, i])
                cols.append(self.tets[:, j])
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        graph = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(self.n, self.n))
        self.tet_graph = graph
        _ncomp, labels = connected_components(graph, directed=False)
        self.part = labels

    # ---------------------------------------------------------- boundary data
    def _build_boundary(self, frame0, config: dict) -> None:
        bc = dict(config.get("boundary_conditions", {}) or {})
        attributes = {k: np.asarray(v, dtype=np.float64) for k, v in dict(frame0.attributes).items()}
        self.attr0 = attributes

        fixed_name = str(bc.get("fixed_attribute", "fixed_grasp"))
        fixed_value = float(bc.get("fixed_value", 1))
        fixed = attributes.get(fixed_name)
        if fixed is None:
            centers = np.asarray(
                bc.get("fixed_grasp_centers", [[-0.92, 0.0, 2.12], [0.92, 0.0, 2.12]]),
                dtype=np.float64,
            )
            mask = np.zeros(self.n, dtype=bool)
            for center in centers:
                mask |= np.linalg.norm(self.x0 - center, axis=1) < 0.22
            self.fixed_mask = mask
        else:
            self.fixed_mask = np.isclose(np.asarray(fixed, dtype=np.float64).reshape(-1), fixed_value)

        pull = dict(bc.get("temporary_pull", {}) or {})
        pull_name = str(pull.get("attribute", "pulled_patch"))
        pull_value = float(pull.get("value", 1))
        pulled = attributes.get(pull_name)
        if pulled is None:
            self.pull_mask = (self.x0[:, 2] < 0.42) & (np.abs(self.x0[:, 0]) < 0.58)
        else:
            self.pull_mask = np.isclose(np.asarray(pulled, dtype=np.float64).reshape(-1), pull_value)
        self.pull_mask &= ~self.fixed_mask

        target_start = np.asarray(pull.get("target_start", [0.0, 0.0, 1.25]), dtype=np.float64)
        target_peak = np.asarray(pull.get("target_peak", [0.0, 0.0, 3.65]), dtype=np.float64)
        self.pull_vector = (target_peak - target_start).reshape(3)

        timeline = dict(config.get("event_timeline", {}) or {})
        interval = pull.get("active_interval_s", [0.0, 1.0])
        self.pull_start = float(timeline.get("pull_start_s", interval[0]))
        self.pull_peak = float(timeline.get("pull_peak_s", 0.9))
        self.release = float(timeline.get("release_s", interval[1]))
        if self.pull_peak <= self.pull_start:
            self.pull_peak = self.pull_start + 1e-3
        if self.release < self.pull_peak:
            self.release = self.pull_peak

        self.fixed_idx = np.flatnonzero(self.fixed_mask)
        self.pull_idx = np.flatnonzero(self.pull_mask)
        self.free_mask = ~(self.fixed_mask | self.pull_mask)
        self.pull_x0 = self.x0[self.pull_idx].copy()

    def _pull_state(self, t: float) -> tuple[float, float, float, bool]:
        """Ramp value, rate and acceleration of the prescribed pull, and activity."""
        active = t <= self.release + 1e-12
        if not active:
            return 0.0, 0.0, 0.0, False
        span = self.pull_peak - self.pull_start
        u = (t - self.pull_start) / span
        s, ds, dds = _quintic(u)
        return s, ds / span, dds / (span * span), True

    # ----------------------------------------------------------- weld springs
    def _build_welds(self) -> None:
        tree = cKDTree(self.x0)
        chosen = None
        for radius in (0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.30):
            pairs = np.asarray(sorted(tree.query_pairs(radius)), dtype=np.int64)
            if len(pairs) == 0:
                continue
            keep = self.part[pairs[:, 0]] != self.part[pairs[:, 1]]
            pairs = pairs[keep]
            if len(pairs) == 0:
                continue
            r = np.concatenate([self.tet_graph.row, pairs[:, 0]])
            c = np.concatenate([self.tet_graph.col, pairs[:, 1]])
            graph = sp.coo_matrix((np.ones(len(r)), (r, c)), shape=(self.n, self.n))
            ncomp, _labels = connected_components(graph, directed=False)
            chosen = pairs
            if ncomp == 1:
                break
        if chosen is None or len(chosen) == 0:
            chosen = np.zeros((0, 2), dtype=np.int64)

        self.spring_a = np.ascontiguousarray(chosen[:, 0])
        self.spring_b = np.ascontiguousarray(chosen[:, 1])
        self.rest = np.linalg.norm(self.x0[self.spring_b] - self.x0[self.spring_a], axis=1)
        self.k_spring = 2.0e4
        deg = np.zeros(self.n, dtype=np.float64)
        if len(self.spring_a):
            np.add.at(deg, self.spring_a, self.k_spring)
            np.add.at(deg, self.spring_b, self.k_spring)
        self.node_k = deg
        eff = np.minimum(self.mass[self.spring_a], self.mass[self.spring_b]) if len(self.spring_a) else np.ones(1)
        self.c_spring = 0.3 * np.sqrt(self.k_spring * np.maximum(eff, 1e-6))

    # -------------------------------------------------------------- time step
    def _choose_timestep(self, config: dict) -> None:
        wave = np.sqrt((self.lam + 2.0 * self.mu) / max(self.rho, 1e-9))
        omega_tet = 2.0 * wave / max(self.h_min, 1e-6)
        omega_spring = float(np.sqrt(np.max(self.node_k * self.inv_mass))) if self.node_k.max() > 0 else 0.0
        omega = max(omega_tet, omega_spring, 1.0)
        self.omega_max = omega
        self.beta = min(0.004, 0.4 / omega)
        self.drag = 0.5
        self.sample_dt = float(config.get("sample_dt", 0.02))
        limit = 0.15 * (2.0 / omega)
        dt = min(self.sample_dt, limit)
        substeps = max(1, int(np.ceil(self.sample_dt / dt)))
        substeps = min(substeps, 1200)
        self.substeps = substeps
        self.dt = self.sample_dt / substeps
        self.v_clamp = 400.0
