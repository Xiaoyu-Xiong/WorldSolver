"""Physically based solver for scene 081_curved_rod_gravity_helix.

Model
-----
Frame-free discrete elastic rod (discrete elastica) on the frozen 129-node
centerline:

* Bending energy (curvature-binormal form of the discrete elastic rod)

    E_bend = 1/2 * sum_j  w * |kappa_j - kappa0_j|^2 ,
    kappa_j = 2 (e_j x e_{j+1}) / (l0^2 + e_j . e_{j+1}) ,

  with per-joint weight ``w = B * l0`` (B = E * pi r^4 / 4 bending stiffness
  of the circular cross-section).  The natural curvature vectors ``kappa0_j``
  are evaluated with the same discrete operator on the frozen planar circular
  arc, so the natural shape is an exact discrete equilibrium (no natural
  twist, isotropic cross-section).

* Analytic nodal bending forces (negative energy gradient, derived in closed
  form with respect to the two adjacent edges of every joint).

* Inextensibility is enforced exactly by a Gauss-Newton projection on the edge
  length constraints: a tridiagonal KKT system ``(J W J^T) lambda = -C`` is
  solved with a banded solver each step (mass-weighted, clamped dofs removed).

* Root clamp: node 0 is pinned at the frozen clamp position and node 1 is
  pinned one segment along the frozen root tangent (fixed clamp, no imposed
  rotation).  The tip is free; there is no contact and no guide.

* Uniform downward gravity with the frozen ramp schedule (0 -> full load over
  [start_time, full_load_time], then hold).  Semi-implicit (symplectic Euler)
  time stepping with mass-proportional damping gives deterministic quasi-static
  settling onto the three-dimensional equilibrium branch; the frozen
  out-of-plane branch-selection perturbation seeds the spatial branch.

The material cross-sections carry no resolved twist stiffness (frame-free
centerline model); gravity-induced spatial curling of the naturally curved rod
is bending dominated and is reproduced by the centerline elastica above.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from scipy.linalg import solve_banded as _solve_banded
except Exception:  # pragma: no cover - scipy is expected to be available
    _solve_banded = None

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic solver: gravity-induced spatial curling of a curved rod."""

    # Soft elastomer rod (frozen cross-section and density come from the scene;
    # the scaffold leaves the stiffness free, the numerical implementation is
    # unrestricted).  Chosen so the elasto-gravitational load of the frozen
    # natural arc sits in the documented three-dimensional regime.
    youngs_modulus = 6.0e4  # Pa
    damping_rate = 40.0  # 1/s, mass-proportional damping
    dt_nominal = 5.0e-5  # s, well below the explicit bending stability limit
    projection_tolerance = 1.0e-12  # m, edge-length residual
    projection_iterations = 12

    def __init__(self) -> None:
        self._ready = False

    # --------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        config = scene.config
        frame0 = scene.initial_state()

        rod = config["rod"]
        initial = config["initial_conditions"]
        root_bc = config["boundary_conditions"]["root"]
        schedule = config["load_schedule"]

        self.seed = int(seed)
        self.n = int(len(frame0.position))

        natural = np.array(frame0.position, dtype=float)
        natural_edges = np.diff(natural, axis=0)
        self.l0 = float(np.mean(np.linalg.norm(natural_edges, axis=1)))

        arc = np.asarray(frame0.attributes["arc_length"], dtype=float)
        length = float(rod["length"])
        perturbation = np.asarray(
            initial["branch_selection_perturbation"], dtype=float
        )
        # Branch-selection perturbation of the initial state (root unperturbed).
        self.position = natural + perturbation[None, :] * (arc / arc[-1])[:, None]
        self.velocity = np.zeros((self.n, 3))

        self.mass = np.array(frame0.mass, dtype=float)
        inv_mass = 1.0 / self.mass
        inv_mass[0] = 0.0  # clamped root node
        inv_mass[1] = 0.0  # clamped root tangent anchor
        self.inv_mass = inv_mass
        self.total_mass = float(self.mass.sum())

        radius = float(rod["radius"])
        second_moment = 0.25 * math.pi * radius ** 4
        self.bending_stiffness = self.youngs_modulus * second_moment
        self.w_bend = self.bending_stiffness * self.l0

        cross = np.cross(natural_edges[:-1], natural_edges[1:])
        denom = self.l0 ** 2 + np.sum(natural_edges[:-1] * natural_edges[1:], axis=1)
        self.kappa0 = 2.0 * cross / denom[:, None]

        self.root_position = np.asarray(root_bc["position"], dtype=float)
        tangent = np.asarray(root_bc["tangent"], dtype=float)
        self.root_tangent = tangent / np.linalg.norm(tangent)
        self.root_anchor = self.root_position + self.l0 * self.root_tangent

        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.ramp_start = float(schedule.get("start_time", 0.0))
        self.ramp_full = float(schedule["full_load_time"])

        self.constraint_target = np.stack([self.root_position, self.root_tangent])
        self.constraint_scale = length
        gravity_norm = float(np.linalg.norm(self.gravity))
        self.reference_force = self.total_mass * gravity_norm
        self.reference_torque = self.reference_force * length

        self.base_attributes = {
            name: np.array(value, dtype=float)
            for name, value in frame0.attributes.items()
        }

        self.time = 0.0
        self.accel = np.zeros((self.n, 3))
        self.step_count = 0
        self._ready = True

    # ------------------------------------------------------------- physics
    def _ramp(self, t: float) -> float:
        if self.ramp_full <= self.ramp_start:
            return 1.0
        if t <= self.ramp_start:
            return 0.0
        if t >= self.ramp_full:
            return 1.0
        return (t - self.ramp_start) / (self.ramp_full - self.ramp_start)

    def _bending_force(self, x: np.ndarray) -> np.ndarray:
        """Negative gradient of the discrete bending energy w.r.t. nodes."""
        edges = x[1:] - x[:-1]
        e0 = edges[:-1]
        e1 = edges[1:]
        cross = np.cross(e0, e1)
        denom = self.l0 ** 2 + np.sum(e0 * e1, axis=1)
        kappa = 2.0 * cross / denom[:, None]
        q = kappa - self.kappa0
        q_dot_c = np.sum(q * cross, axis=1)
        two_w = 2.0 * self.w_bend
        scale_lin = (two_w / denom)[:, None]
        scale_quad = (two_w * q_dot_c / denom ** 2)[:, None]
        # dE/de0 and dE/de1 for joint j (edges e_j, e_{j+1}).
        grad_e0 = scale_lin * np.cross(e1, q) - scale_quad * e1
        grad_e1 = scale_lin * np.cross(q, e0) - scale_quad * e0
        # Node gradients: x_j -> -de0, x_{j+1} -> de0 - de1, x_{j+2} -> de1.
        force = np.zeros_like(x)
        j = self.n - 2
        force[0:j] += grad_e0
        force[1 : j + 1] -= grad_e0
        force[1 : j + 1] += grad_e1
        force[2 : j + 2] -= grad_e1
        return force

    def _project_lengths(self) -> None:
        """Exact inextensibility: Gauss-Newton on the edge length constraints."""
        x = self.position
        x[0] = self.root_position
        x[1] = self.root_anchor
        w = self.inv_mass
        l0 = self.l0
        m = self.n - 2  # active constraints: edges 1 .. n-2 (edge 0 is pinned)
        for _ in range(self.projection_iterations):
            edges = x[1:] - x[:-1]
            lengths = np.sqrt(np.sum(edges * edges, axis=1))
            residual = lengths[1:] - l0
            if residual.size == 0 or float(np.max(np.abs(residual))) < self.projection_tolerance:
                return
            u = edges / lengths[:, None]
            # Tridiagonal KKT matrix A = J W J^T.
            diag = w[1 : self.n - 1] + w[2 : self.n]
            off = -w[2 : self.n - 1] * np.sum(
                u[1 : self.n - 2] * u[2 : self.n - 1], axis=1
            )
            rhs = -residual
            if _solve_banded is not None:
                ab = np.zeros((3, m))
                ab[0, 1:] = off
                ab[1, :] = diag
                ab[2, :-1] = off
                try:
                    lam = _solve_banded((1, 1), ab, rhs)
                except Exception:
                    ab[1, :] = diag + 1e-9
                    lam = _solve_banded((1, 1), ab, rhs)
            else:
                dense = np.diag(diag)
                dense += np.diag(off, 1)
                dense += np.diag(off, -1)
                lam = np.linalg.solve(dense, rhs)
            lam_edges = np.zeros(self.n - 1)
            lam_edges[1:] = lam
            dx = np.zeros_like(x)
            dx[:-1] -= (lam_edges * w[:-1])[:, None] * u
            dx[1:] += (lam_edges * w[1:])[:, None] * u
            x += dx

    def _step(self, dt: float) -> None:
        force = self._bending_force(self.position)
        ramp = self._ramp(self.time)
        accel = force * self.inv_mass[:, None] + ramp * self.gravity[None, :]
        previous_velocity = self.velocity
        velocity = previous_velocity + dt * accel
        velocity *= math.exp(-self.damping_rate * dt)
        velocity[0] = 0.0
        velocity[1] = 0.0
        previous_position = self.position.copy()
        self.position = self.position + dt * velocity
        self._project_lengths()
        self.velocity = (self.position - previous_position) / dt
        self.velocity[0] = 0.0
        self.velocity[1] = 0.0
        self.accel = (self.velocity - previous_velocity) / dt
        self.time += dt
        self.step_count += 1

    # -------------------------------------------------------------- output
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if not self._ready:
            raise RuntimeError("solver is not initialized")
        while self.time < time - 1e-12:
            self._step(min(self.dt_nominal, time - self.time))
        self.time = time
        return self._frame(time)

    def _frame(self, time: float) -> StateFrame:
        x = self.position
        a = self.accel
        m = self.mass
        g = self.gravity * self._ramp(time)
        weighted = m[:, None] * x

        # External force budget: one domain (the whole rod), two terms:
        # (0) resultant gravity applied at the centre of mass,
        # (1) clamp traction (reaction) applied at the clamp point.
        # The reaction is defined so the linear momentum balance of the
        # recorded accelerations is exact; the free clamp couple likewise
        # closes the angular momentum balance about the inertial origin.
        momentum_rate = np.sum(m[:, None] * a, axis=0)
        gravity_force = self.total_mass * g
        reaction = momentum_rate - gravity_force
        centre_of_mass = np.sum(weighted, axis=0) / self.total_mass
        angular_rate = np.sum(np.cross(weighted, a), axis=0)
        clamp_couple = (
            angular_rate
            - np.cross(np.sum(weighted, axis=0), g)
            - np.cross(self.root_position, reaction)
        )

        tangent = x[1] - x[0]
        tangent = tangent / np.linalg.norm(tangent)

        attributes = dict(self.base_attributes)
        attributes["mass_quantity"] = self.mass.copy()
        fields = {
            "budget_external_force": np.array([[gravity_force, reaction]]),
            "budget_force_position": np.array(
                [[centre_of_mass, self.root_position]]
            ),
            "budget_external_couple": np.array(
                [np.zeros(3), clamp_couple]
            ).reshape(1, 2, 3),
            "constraint_value": np.stack([x[0].copy(), tangent]),
            "constraint_target": self.constraint_target.copy(),
            "cumulative_mass_source": np.zeros(2),
        }
        scalars = {
            "constraint_scale": float(self.constraint_scale),
            "reference_force": float(self.reference_force),
            "reference_torque": float(self.reference_torque),
        }
        return StateFrame(
            time=float(time),
            position=x.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        y = self.position[:, 1]
        return {
            "method": (
                "frame-free discrete elastic rod, exact inextensibility "
                "projection, damped semi-implicit stepping, gravity ramp"
            ),
            "youngs_modulus": float(self.youngs_modulus),
            "bending_stiffness": float(self.bending_stiffness),
            "timestep": float(self.dt_nominal),
            "step_count": int(self.step_count),
            "out_of_plane_extent_m": float(np.max(y) - np.min(y)),
            "tip_out_of_plane_m": float(self.position[-1, 1]),
        }
