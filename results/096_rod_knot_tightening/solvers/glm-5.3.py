"""XPBD-based physical solver for tightening a frictionless elastic-rod knot.

Discretization
--------------
The strand is a chain of ``N`` nodes joined by ``N-1`` edges.

* Inextensibility: zero-compliance XPBD distance constraints on every edge,
  with rest length ``rest_length / (N-1)`` taken from the frozen config.
  Constraints are solved with a two-color (even/odd edge) Gauss-Seidel sweep
  so tension propagates along the strand each substep.
* Bending resistance: discrete Euler-Bernoucli bending energy
  ``E = k_b/2 * sum_i |x[i-1] - 2 x[i] + x[i+1]|^2`` whose nodal forces are
  the (negative) second difference of the discrete curvature, plus internal
  viscous damping of the same operator (an internal force pair, so it cancels
  in the combined-domain momentum budget).
* Frictionless finite-thickness self-contact: edge-edge non-penetration
  constraints at the frozen contact distance (2 * rod radius) projected every
  solver round; only normal corrections are applied (friction coefficient 0),
  distributed with inverse-mass weights so every correction is an
  equal-and-opposite impulse pair.
* Endpoint manipulation: both strand ends are driven along the frozen
  smoothstep waypoint targets.  The manipulator impulse required to realize
  the prescribed motion is accumulated and exported as the external-force
  budget of the (single, whole-strand) domain.

Gravity is zero and no mass crosses the domain boundary, so the exported
momentum budget closes by construction: the change of total strand momentum
between samples equals the accumulated external impulse.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic XPBD rod solver with frictionless self-contact."""

    def __init__(self) -> None:
        self.scene = None
        self.seed = 0
        self.t = 0.0
        self._frame_time = -1.0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = scene.config
        rod = config["rod"]
        contact = config["self_contact"]

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)
        self.n = len(self.x)
        self.v = np.zeros_like(self.x)
        self.mass = np.array(frame0.mass, dtype=float)
        self.w = 1.0 / self.mass

        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.rest_length = float(rod["rest_length"])
        self.ds = self.rest_length / (self.n - 1)
        self.radius = float(rod["radius"])
        self.contact_distance = float(contact["contact_distance"])
        self.friction = float(contact["friction_coefficient"])

        # Numerical parameters (frozen for determinism).
        self.dt_base = 2.5e-4          # substep length [s]
        self.solver_rounds = 6         # constraint rounds per substep
        self.bend_stiffness = 150.0    # nodal bending stiffness [N/m]
        self.bend_damping = 0.02       # internal bending viscosity [kg/s]
        self.max_speed = 500.0         # divergence safety clamp [m/s]

        self.ne = self.n - 1
        self.even_edges = np.arange(0, self.ne, 2)
        self.odd_edges = np.arange(1, self.ne, 2)

        index = np.arange(self.n)
        self.vertex_pair_mask = np.abs(index[:, None] - index[None, :]) >= 3
        self.upper = index[:, None] < index[None, :]
        # Broad-phase radius: contact distance plus enough slack to cover the
        # motion that can happen inside one substep.
        self.broad_radius = self.contact_distance + 2.0 * self.ds

        # Budget bookkeeping.
        self.cumulative_impulse = np.zeros(3)
        self._interval_impulse = np.zeros(3)
        self._interval_start = 0.0

        # Cached prescribed endpoint positions at the current time.
        self._target_now = self._target_array(0.0)

        # Base per-vertex attributes carried from the frozen initial frame.
        self._base_attributes = {
            name: np.array(value, dtype=float).copy()
            for name, value in frame0.attributes.items()
        }

        self._substep_count = 0
        self._contact_pair_peak = 0
        self._t = 0.0

    # ------------------------------------------------------------- utilities
    def _target_array(self, time: float) -> np.ndarray:
        targets = self.scene.endpoint_targets(float(time))
        return np.array(
            [targets["left_tail"], targets["threading_tail"]], dtype=float
        )

    # ------------------------------------------------------------- dynamics
    def _bending_forces(self) -> np.ndarray:
        """Nodal forces of the discrete bending energy (+ internal damping)."""
        x, v = self.x, self.v
        n = self.n
        kb, cb = self.bend_stiffness, self.bend_damping

        curvature = np.zeros_like(x)
        curvature[1:-1] = x[:-2] - 2.0 * x[1:-1] + x[2:]
        vel_lap = np.zeros_like(v)
        vel_lap[1:-1] = v[:-2] - 2.0 * v[1:-1] + v[2:]

        force = np.zeros_like(x)
        # Interior: negative second difference of the curvature vector.
        force[1:-1] = -kb * (
            curvature[:-2] - 2.0 * curvature[1:-1] + curvature[2:]
        ) - cb * (vel_lap[:-2] - 2.0 * vel_lap[1:-1] + vel_lap[2:])
        # Free ends: only the adjacent curvature term contributes.
        force[0] = -kb * curvature[1] - cb * vel_lap[1]
        force[-1] = -kb * curvature[-2] - cb * vel_lap[-2]
        return force

    def _stretch_pass(self, edges: np.ndarray) -> None:
        """One Gauss-Seidel color sweep of zero-compliance edge constraints."""
        if len(edges) == 0:
            return
        x, w, ds = self.x, self.w, self.ds
        a = x[edges]
        b = x[edges + 1]
        delta = b - a
        length = np.linalg.norm(delta, axis=1)
        length = np.maximum(length, 1e-12)
        direction = delta / length[:, None]
        cstr = length - ds
        wa = w[edges]
        wb = w[edges + 1]
        lam = cstr / (wa + wb)
        # Nodes inside one color sweep are distinct: direct scatter is safe.
        x[edges] = a + (wa * lam)[:, None] * direction
        x[edges + 1] = b - (wb * lam)[:, None] * direction

    def _contact_candidates(self) -> tuple[np.ndarray, np.ndarray]:
        """Broad phase: candidate edge pairs that may be within contact range."""
        x = self.x
        sq = np.einsum("ij,ij->i", x, x)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (x @ x.T)
        np.maximum(d2, 0.0, out=d2)
        mask = self.vertex_pair_mask & self.upper
        mask &= d2 < self.broad_radius * self.broad_radius
        ia, ib = np.nonzero(mask)
        if len(ia) == 0:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
        # Vertex a touches edges a-1 and a; likewise for b.
        ea = np.concatenate([ia - 1, ia - 1, ia, ia])
        eb = np.concatenate([ib - 1, ib, ib - 1, ib])
        valid = (ea >= 0) & (ea < self.ne) & (eb >= 0) & (eb < self.ne)
        codes = np.unique(ea[valid] * self.ne + eb[valid])
        edge_a = codes // self.ne
        edge_b = codes % self.ne
        self._contact_pair_peak = max(self._contact_pair_peak, int(len(codes)))
        return edge_a, edge_b

    def _contact_pass(self, edge_a: np.ndarray, edge_b: np.ndarray) -> None:
        """Project penetrating edge pairs apart along their common normal."""
        if len(edge_a) == 0:
            return
        x, w = self.x, self.w
        p1 = x[edge_a]
        q1 = x[edge_a + 1]
        p2 = x[edge_b]
        q2 = x[edge_b + 1]
        d1 = q1 - p1
        d2 = q2 - p2
        a2 = np.einsum("ij,ij->i", d1, d1)
        e2 = np.einsum("ij,ij->i", d2, d2)

        # Alternating clamped projection: closest points between segments.
        s = np.full(len(edge_a), 0.5)
        t = np.zeros(len(edge_a))
        for _ in range(4):
            c1 = p1 + s[:, None] * d1
            t = np.clip(
                np.einsum("ij,ij->i", c1 - p2, d2) / np.maximum(e2, 1e-24),
                0.0, 1.0,
            )
            c2 = p2 + t[:, None] * d2
            s = np.clip(
                np.einsum("ij,ij->i", c2 - p1, d1) / np.maximum(a2, 1e-24),
                0.0, 1.0,
            )
        c1 = p1 + s[:, None] * d1
        c2 = p2 + t[:, None] * d2
        diff = c2 - c1
        dist = np.linalg.norm(diff, axis=1)
        penetration = self.contact_distance - dist
        active = penetration > 0.0
        if not np.any(active):
            return

        ia = edge_a[active]
        ib = edge_b[active]
        sa = s[active]
        tb = t[active]
        pen = penetration[active]
        safe = np.maximum(dist[active], 1e-12)
        normal = diff[active] / safe[:, None]

        w_a = (1.0 - sa) ** 2 * w[ia] + sa ** 2 * w[ia + 1]
        w_b = (1.0 - tb) ** 2 * w[ib] + tb ** 2 * w[ib + 1]
        lam = pen / np.maximum(w_a + w_b, 1e-30)

        # Inverse-mass weighted distribution keeps every correction an
        # equal-and-opposite impulse pair (total momentum conserving).
        np.add.at(x, ia, -(1.0 - sa)[:, None] * (w[ia] * lam)[:, None] * normal)
        np.add.at(x, ia + 1, -sa[:, None] * (w[ia + 1] * lam)[:, None] * normal)
        np.add.at(x, ib, (1.0 - tb)[:, None] * (w[ib] * lam)[:, None] * normal)
        np.add.at(x, ib + 1, tb[:, None] * (w[ib + 1] * lam)[:, None] * normal)

    def _substep(self, dt: float, t_new: float) -> None:
        x_prev = self.x.copy()
        target_old = self._target_now
        target_new = self._target_array(t_new)

        # Explicit force step (bending + internal damping + gravity).
        force = self._bending_forces()
        self.v += dt * (force * self.w[:, None] + self.gravity[None, :])
        speed = np.linalg.norm(self.v, axis=1)
        if np.any(speed > self.max_speed):
            scale = self.max_speed / np.maximum(speed, 1e-30)
            self.v *= scale[:, None]
        self.x += dt * self.v

        # Constraint projection: stretch colors interleaved with contacts.
        edge_a, edge_b = self._contact_candidates()
        for _ in range(self.solver_rounds):
            self._stretch_pass(self.even_edges)
            self._stretch_pass(self.odd_edges)
            self._contact_pass(edge_a, edge_b)

        # Position-derived velocity (XPBD style).
        self.v = (self.x - x_prev) / dt

        # Kinematic endpoint override + manipulator impulse bookkeeping.
        prescribed = (target_new - target_old) / dt
        impulse = self.mass[0] * (prescribed[0] - self.v[0])
        impulse += self.mass[-1] * (prescribed[1] - self.v[-1])
        self.x[0] = target_new[0]
        self.x[-1] = target_new[1]
        self.v[0] = prescribed[0]
        self.v[-1] = prescribed[1]

        # Gravity impulse on the closed strand domain.
        gravity_impulse = self.mass.sum() * self.gravity * dt
        self.cumulative_impulse += impulse + gravity_impulse
        self._interval_impulse += impulse + gravity_impulse

        self._target_now = target_new
        self._t = t_new
        self._substep_count += 1

    # ---------------------------------------------------------------- frames
    def advance_to(self, time: float) -> StateFrame:
        t = float(time)
        if t > self._t + 1e-12:
            remaining = t - self._t
            steps = max(1, int(round(remaining / self.dt_base)))
            dt = remaining / steps
            for _ in range(steps):
                t_old = self._t
                t_new = t_old + dt
                self._substep(dt, t_new)
            self._t = t
        return self._build_frame(self._t)

    def _build_frame(self, t: float) -> StateFrame:
        n = self.n
        interval = t - self._interval_start
        if interval > 1e-12:
            external_force = self._interval_impulse / interval
        else:
            external_force = np.zeros(3)
        self._interval_impulse[:] = 0.0
        self._interval_start = t

        target = self._target_array(t)
        attributes = dict(self._base_attributes)
        attributes["mass_quantity"] = self.mass.copy()

        fields = {
            "budget_cumulative_external_impulse":
                self.cumulative_impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, n)),
            "budget_external_force": external_force[None, None, :].copy(),
            "constraint_target": target.copy(),
            "constraint_value": np.array([self.x[0], self.x[-1]]),
            "cumulative_mass_source": np.zeros(1),
        }
        scalars = {
            "constraint_scale": self.rest_length,
            "reference_momentum": float(self.mass.sum()),
            "friction_coefficient": self.friction,
        }

        frame = StateFrame(
            t,
            self.x.copy(),
            self.v.copy(),
            orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1)),
            angular_velocity=np.zeros((n, 3)),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        edge_vectors = np.diff(self.x, axis=0)
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        strain = np.abs(edge_lengths - self.ds) / self.ds
        return {
            "time": float(self._t),
            "substeps": int(self._substep_count),
            "max_edge_strain": float(np.max(strain)) if len(strain) else 0.0,
            "mean_edge_strain": float(np.mean(strain)) if len(strain) else 0.0,
            "peak_contact_pairs": int(self._contact_pair_peak),
            "total_length": float(edge_lengths.sum()),
            "rest_length": float(self.rest_length),
            "friction_coefficient": float(self.friction),
        }
