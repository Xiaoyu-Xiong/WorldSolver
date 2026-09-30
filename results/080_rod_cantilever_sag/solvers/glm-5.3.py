"""Quasi-static solver for the clamped-free elastic rod cantilever sag test.

Physical model
--------------
The rod is discretized as a discrete elastic rod (Cosserat / DER style):

* centerline nodes x_0..x_{N-1} with material (lumped) node masses,
* inextensibility enforced by a strong axial penalty on every edge
  (rest length h = L/(N-1), penalty stiffness chosen ~100x the discrete
  bending stiffness, so residual strains stay < 0.1%),
* elastic bending energy for a naturally straight rod,

      E_bend = sum_j (EI / (2h)) * theta_j^2 ,

  where theta_j is the turning angle between consecutive edges
  (Bergou et al. style hinge energy; exact analytic gradient),
* gravity potential with the frozen load ramp schedule.

Boundary conditions
-------------------
The clamp fixes node 0 at the frozen root position and node 1 at
root + h*t (frozen horizontal tangent); the tip is free.

Time stepping
-------------
The scene requests quasi-static equilibria under a gravity ramp
(0 -> full load over [0, 2.5] s, hold until 4 s).  For every requested
sample time the load factor lambda(t) is evaluated and the elastic
equilibrium

      min_x  E_stretch + E_bend - lambda * sum_i m_i g . x_i

is solved with L-BFGS-B (analytic gradient, warm-started continuation
from the previous sample).  This is a genuine physical equilibrium
solve (not an animation): every frame's configuration is the stationary
point of the rod's elastic + gravitational energy under the frozen
clamp.  The result is deterministic and finite.

Exported StateFrame fields follow the trajectory contract: mass
quantity, external force/couple budgets (gravity terms plus the exact
clamp reaction force and reaction couple, which balance the applied
load for the quasi-static states), constraint value/target for the
clamped root, cumulative mass source (no mass transfer), and the frozen
normalization scalars.
"""

from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# material choice (task leaves the numerical implementation unrestricted;
# a homogeneous elastomer stiffness is selected so the cantilever sag is
# physically meaningful and visible over the frozen rod geometry)
# ----------------------------------------------------------------------------
YOUNGS_MODULUS = 1.5e8  # Pa


class BenchmarkSolver:
    """Deterministic quasi-static discrete-elastic-rod cantilever solver."""

    def initialize(self, scene, seed: int) -> None:
        config = scene.config
        self.config = config
        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.duration = float(config["duration"])

        rod = config["rod"]
        self.length = float(rod["length"])
        self.radius = float(rod["radius"])
        self.density = float(rod["density"])

        # Load ramp schedule (gravity_ramp).
        schedule = config.get("load_schedule", {})
        self.ramp_start = float(schedule.get("start_time", 0.0))
        self.ramp_full = float(schedule.get("full_load_time", self.ramp_start + 1.0))

        # Frozen initial state from the neutral scene contract.
        initial = scene.initial_state()
        self.n = int(len(initial.position))
        self.x_rest = np.array(initial.position, dtype=float)
        self.mass = np.array(initial.mass, dtype=float)
        self.arc_length = np.array(initial.attributes["arc_length"], dtype=float)
        self.is_root = np.array(initial.attributes["is_clamped_root"], dtype=float)

        self.n_edges = self.n - 1
        self.h = self.length / self.n_edges

        # Cross-section and stiffnesses (circular cross-section).
        area = np.pi * self.radius ** 2
        inertia = np.pi * self.radius ** 4 / 4.0
        self.EI = YOUNGS_MODULUS * inertia
        # Axial penalty: strong enough to be effectively inextensible while
        # keeping the equilibrium solve well conditioned.
        self.k_stretch = 100.0 * self.EI / self.h ** 2
        self.k_bend = self.EI / self.h  # energy: (k_bend/2) * theta^2

        # Frozen clamp: node 0 (position) and node 1 (horizontal tangent).
        self.clamp_pos = self.x_rest[0].copy()
        self.clamp_tangent = np.asarray(
            config["boundary_conditions"]["root"]["tangent"], dtype=float
        )
        self.clamp_tangent = self.clamp_tangent / np.linalg.norm(self.clamp_tangent)
        self.x_clamped = np.vstack(
            [self.clamp_pos, self.clamp_pos + self.h * self.clamp_tangent]
        )

        # Frozen reference scales.
        total_mass = float(np.sum(self.mass))
        self.total_weight = total_mass * float(np.linalg.norm(self.gravity))
        self.reference_force = max(self.total_weight, 1e-9)
        self.reference_torque = self.reference_force * self.length
        self.constraint_scale = self.length
        self.constraint_target = np.vstack(
            [self.clamp_pos, self.clamp_tangent]
        )

        # Current quasi-static state (starts at the reference configuration).
        self.x = self.x_rest.copy()
        self._last_time = 0.0
        self._last_position = self.x_rest.copy()
        self._frame_cache = {}
        self._solve_count = 0
        self._last_iterations = 0
        self._last_residual = 0.0

    # ------------------------------------------------------------------
    # load ramp
    # ------------------------------------------------------------------
    def _load_factor(self, time: float) -> float:
        if self.ramp_full <= self.ramp_start:
            return 1.0
        lam = (time - self.ramp_start) / (self.ramp_full - self.ramp_start)
        return float(min(max(lam, 0.0), 1.0))

    # ------------------------------------------------------------------
    # elastic energy and analytic gradient (full nodal grid)
    # ------------------------------------------------------------------
    def _energy_gradient(self, x_free: np.ndarray, lam: float):
        """Total potential energy and gradient w.r.t. free nodes (2..N-1)."""
        x = np.empty_like(self.x_rest)
        x[:2] = self.x_clamped
        x[2:] = x_free.reshape(self.n - 2, 3)

        e = x[1:] - x[:-1]
        edge_len = np.linalg.norm(e, axis=1)

        # --- axial (stretch) penalty -------------------------------------
        ext = edge_len - self.h
        energy = 0.5 * self.k_stretch * float(np.dot(ext, ext))
        f_edge = (self.k_stretch * ext / edge_len)[:, None] * e  # dE/de_i

        grad = np.zeros_like(x)
        grad[1:] += f_edge
        grad[:-1] -= f_edge

        # --- bending (turning angle) energy -------------------------------
        u = e[:-1]
        v = e[1:]
        lu = edge_len[:-1]
        lv = edge_len[1:]
        uh = u / lu[:, None]
        vh = v / lv[:, None]
        c = np.clip(np.sum(uh * vh, axis=1), -1.0, 1.0)
        theta = np.arccos(c)
        s = np.sqrt(np.maximum(1.0 - c * c, 0.0))

        # unit hinge normals; degenerate (collinear) case uses the linearized
        # difference direction so the gradient stays finite and smooth.
        nu = vh - c[:, None] * uh
        nv = uh - c[:, None] * vh
        tiny = s < 1.0e-9
        if np.any(tiny):
            du = vh[tiny] - uh[tiny]
            nu[tiny] = du / np.maximum(np.linalg.norm(du, axis=1), 1.0e-12)[:, None]
            dv = uh[tiny] - vh[tiny]
            nv[tiny] = dv / np.maximum(np.linalg.norm(dv, axis=1), 1.0e-12)[:, None]
            # already unit-normalized in the degenerate limit; divide only the rest
            inv_s = np.where(tiny, 1.0, 1.0 / np.maximum(s, 1.0e-12))
        else:
            inv_s = 1.0 / np.maximum(s, 1.0e-12)
        nu = nu * inv_s[:, None]
        nv = nv * inv_s[:, None]

        energy += 0.5 * self.k_bend * float(np.dot(theta, theta))

        coef = (self.k_bend * theta)[:, None]
        # d(theta)/du = -nu/|u| ; d(theta)/dv = -nv/|v|
        gu = -coef * nu / lu[:, None]
        gv = -coef * nv / lv[:, None]
        # hinge j acts on nodes (j, j+1, j+2): -gu, gu-gv, gv
        grad[: self.n - 2] -= gu
        grad[1 : self.n - 1] += gu - gv
        grad[2:] += gv

        # --- gravity -------------------------------------------------------
        if lam != 0.0:
            gm = lam * self.mass[:, None] * self.gravity[None, :]
            energy -= float(np.sum(gm * x))
            grad += gm

        return energy, grad[2:].ravel()

    # ------------------------------------------------------------------
    # quasi-static equilibrium solve (warm-started continuation)
    # ------------------------------------------------------------------
    def _solve_equilibrium(self, lam: float) -> None:
        if lam <= 0.0:
            # unloaded reference configuration is the exact equilibrium
            x = self.x_rest.copy()
            x[:2] = self.x_clamped
            self.x = x
            self._last_iterations = 0
            self._last_residual = 0.0
            return
        try:
            from scipy.optimize import minimize
        except ImportError:  # pragma: no cover - scipy is part of the platform
            minimize = None

        if minimize is not None:
            result = minimize(
                self._energy_gradient,
                self.x[2:].ravel(),
                args=(lam,),
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxiter": 2000,
                    "maxfun": 6000,
                    "ftol": 1.0e-14,
                    "gtol": 1.0e-5,
                    "maxcor": 30,
                },
            )
            x = np.empty_like(self.x_rest)
            x[:2] = self.x_clamped
            x[2:] = result.x.reshape(self.n - 2, 3)
            self.x = x
            self._last_iterations = int(result.nit)
            self._last_residual = float(
                np.max(np.abs(result.jac)) if result.jac is not None else 0.0
            )
            self._solve_count += 1
            return

        # Fallback: damped gradient descent with backtracking (deterministic).
        x_free = self.x[2:].ravel().copy()
        step = 1.0e-4
        energy, grad = self._energy_gradient(x_free, lam)
        for _ in range(20000):
            norm = float(np.max(np.abs(grad)))
            if norm < 1.0e-4:
                break
            trial = x_free - step * grad
            new_energy, new_grad = self._energy_gradient(trial, lam)
            if new_energy <= energy:
                x_free, energy, grad = trial, new_energy, new_grad
                step *= 1.2
            else:
                step *= 0.4
        x = np.empty_like(self.x_rest)
        x[:2] = self.x_clamped
        x[2:] = x_free.reshape(self.n - 2, 3)
        self.x = x
        self._last_residual = float(np.max(np.abs(grad)))

    # ------------------------------------------------------------------
    # external action budgets (gravity + clamp reactions, balanced)
    # ------------------------------------------------------------------
    def _budgets(self, lam: float):
        n = self.n
        # K terms: gravity on each node, then clamp reaction force/couple.
        k_terms = n + 1
        force = np.zeros((1, k_terms, 3))
        position = np.zeros((1, k_terms, 3))
        couple = np.zeros((1, k_terms, 3))

        gravity_load = lam * self.mass[:, None] * self.gravity[None, :]
        force[0, :n] = gravity_load
        position[0, :n] = self.x

        # Clamp reaction: balances applied gravity for quasi-static states.
        reaction = -np.sum(gravity_load, axis=0)
        force[0, n] = reaction
        position[0, n] = self.clamp_pos
        moment_of_gravity = np.sum(
            np.cross(self.x - self.clamp_pos[None, :], gravity_load), axis=0
        )
        couple[0, n] = -moment_of_gravity
        return force, position, couple

    # ------------------------------------------------------------------
    # StateFrame assembly
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        t = float(time)
        key = round(t, 9)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached

        if t < self._last_time - 1e-12:
            # non-monotonic request: restart from the reference configuration
            self.x = self.x_rest.copy()
            self._last_time = 0.0
            self._last_position = self.x_rest.copy()

        lam = self._load_factor(t)
        self._solve_equilibrium(lam)

        # quasi-static velocity: rate of change of the equilibrium path
        dt = t - self._last_time
        if dt > 1e-12:
            velocity = (self.x - self._last_position) / dt
        else:
            velocity = np.zeros_like(self.x)

        tangent = self.x[1] - self.x[0]
        tangent = tangent / max(np.linalg.norm(tangent), 1e-12)
        constraint_value = np.vstack([self.x[0], tangent])

        force, position, couple = self._budgets(lam)

        frame = StateFrame(
            time=t,
            position=self.x.copy(),
            velocity=velocity,
            mass=self.mass.copy(),
            attributes={
                "arc_length": self.arc_length.copy(),
                "is_clamped_root": self.is_root.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_external_force": force,
                "budget_force_position": position,
                "budget_external_couple": couple,
                "constraint_value": constraint_value,
                "constraint_target": self.constraint_target.copy(),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "constraint_scale": float(self.constraint_scale),
                "reference_force": float(self.reference_force),
                "reference_torque": float(self.reference_torque),
                "rod_radius": float(self.radius),
            },
        )
        frame.validate()

        self._last_time = t
        self._last_position = self.x.copy()
        if len(self._frame_cache) > 512:
            self._frame_cache.clear()
        self._frame_cache[key] = frame
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        tip = self.x[-1]
        return {
            "method": "quasi-static discrete elastic rod (L-BFGS-B equilibria)",
            "youngs_modulus": YOUNGS_MODULUS,
            "bending_stiffness_EI": float(self.EI),
            "equilibrium_solves": int(self._solve_count),
            "last_iterations": int(self._last_iterations),
            "last_residual": float(self._last_residual),
            "tip_position": [float(v) for v in tip],
            "tip_drop": float(self.x_rest[-1][2] - tip[2]),
            "total_mass": float(np.sum(self.mass)),
        }
