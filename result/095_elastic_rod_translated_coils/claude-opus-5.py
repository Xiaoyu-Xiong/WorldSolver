"""Discrete elastic-rod solver for translated coiling on a moving belt.

Physical model (implemented from scratch with NumPy):

* One inextensible, naturally straight, circular-cross-section elastic rod is
  discretized as a Lagrangian chain of lumped-mass nodes with fixed material
  segment length ``dL0``.
* Bending elasticity uses the discrete Euler-Bernoulli energy
  ``E_b = sum_i (EI/(2 dL^3)) |x_{i+1} - 2 x_i + x_{i-1}|^2`` whose gradient is
  the fourth-difference operator; a Rayleigh (stiffness proportional) viscous
  term damps the shortest bending waves.
* Inextensibility is enforced by a red/black Gauss-Seidel position projection of
  the edge-length constraints (a projective-dynamics / PBD relaxation) whose
  Lagrange multipliers give the physical tension field.
* Gravity is uniform and downward; mild mass-proportional aerodynamic drag keeps
  the slowly-fed heel in the quasi-static regime of the source experiment.
* The injector feeds material downward at a constant speed: the two nodes at the
  nozzle are kinematically prescribed (clamped, vertical) and a new material node
  enters the domain each time the fed length reaches one segment length.
* The belt is a rigid horizontal substrate translating at constant velocity.
  Deposited material sticks perfectly (no sliding, no lift-off) and is advected
  rigidly with the belt, which is the source substrate condition.

Time stepping is explicit symplectic-Euler on the free nodes followed by the
constraint projection and a PBD velocity update, with a fixed step chosen well
below the bending CFL limit, so the trajectory is deterministic and finite.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

_EPS = 1e-12


def _fourth_difference(values: np.ndarray) -> np.ndarray:
    """Discrete gradient operator of ``sum |v_{i+1}-2v_i+v_{i-1}|^2 / 2``.

    ``values`` has shape ``[n,3]``; the result has the same shape and is the
    symmetric fourth-difference stencil with free (moment-free) ends.
    """
    n = values.shape[0]
    out = np.zeros_like(values)
    if n < 3:
        return out
    second = np.zeros_like(values)
    second[1:-1] = values[2:] - 2.0 * values[1:-1] + values[:-2]
    out[1:-1] = second[:-2] - 2.0 * second[1:-1] + second[2:]
    out[0] = second[1]
    out[-1] = second[-2]
    return out


class BenchmarkSolver:
    """Discrete elastic rod fed onto a sticking, translating rigid belt."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = dict(scene.config) if hasattr(scene, "config") else {}
        self.config = config
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self._rng = rng

        rod = config["rod"]
        inj = config["injector"]
        belt = config["belt"]

        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.duration = float(config["duration"])
        self.sample_dt = float(config["sample_dt"])

        self.radius = float(rod["radius"])
        self.density = float(rod["density"])
        self.area = np.pi * self.radius ** 2
        self.mu = self.density * self.area                 # line density [kg/m]
        self.inertia = 0.25 * np.pi * self.radius ** 4     # second moment [m^4]
        self.youngs_modulus = 1.0e6                        # soft elastomer [Pa]
        self.bending_stiffness = self.youngs_modulus * self.inertia

        self.outlet = np.asarray(inj["outlet"], dtype=float)
        axis = np.asarray(inj["axis"], dtype=float)
        self.axis = axis / max(np.linalg.norm(axis), _EPS)
        self.inject_speed = float(inj["injection_speed"])
        self.feed_start = float(inj["feed_start"])
        self.feed_end = float(inj["feed_end"])

        half = np.asarray(belt["half_extents"], dtype=float)
        center = np.asarray(belt["center"], dtype=float)
        self.belt_surface_z = float(belt["surface_z"])
        self.belt_min_xy = center[:2] - half[:2]
        self.belt_max_xy = center[:2] + half[:2]
        belt_axis = np.asarray(belt["translation_axis"], dtype=float)
        belt_axis = belt_axis / max(np.linalg.norm(belt_axis), _EPS)
        self.belt_velocity = float(belt["speed"]) * belt_axis
        self.contact_height = self.belt_surface_z + self.radius

        # ---- discretization from the frozen initial state ----------------
        frame0 = scene.initial_state()
        x0 = np.asarray(frame0.position, dtype=float).copy()
        n0 = x0.shape[0]
        self.dL0 = float(rod["initial_deployed_length"]) / (n0 - 1)
        self.length0 = float(rod["initial_deployed_length"])

        fed_total = self.inject_speed * max(self.feed_end - self.feed_start, 0.0)
        max_new = int(np.floor(fed_total / self.dL0)) + 3
        self.nmax = n0 + max_new + 2
        self.end = self.nmax - 1               # one-past-last active index
        self.start = self.end - n0             # first active index

        self.x = np.zeros((self.nmax, 3), dtype=float)
        self.v = np.zeros((self.nmax, 3), dtype=float)
        self.x[self.start:self.end] = x0
        self.v[self.start:self.end] = np.asarray(frame0.velocity, dtype=float)
        self.n_dep = 0                          # deposited suffix length
        self.inserted = 0
        self.frac = 0.0                         # extruded fraction of one segment

        # ---- numerical parameters --------------------------------------
        # Discrete bending operator spectral radius: omega_max = 4 sqrt(EI/mu)/dL^2
        omega_max = 4.0 * np.sqrt(self.bending_stiffness / self.mu) / self.dL0 ** 2
        dt_limit = 2.0 / max(omega_max, _EPS)
        target = 0.25 * dt_limit
        nsub = int(np.ceil(self.sample_dt / target))
        self.nsub = int(min(max(nsub, 1), 520))
        self.dt = self.sample_dt / self.nsub
        # Rayleigh (stiffness proportional) bending damping and air drag.
        self.beta = min(2.0 / max(omega_max, _EPS), 0.25 * self.dt / 1e-6 * 1e-6)
        self.beta = 2.0 / max(omega_max, _EPS)
        self.drag_rate = 3.0                    # 1/s, mass proportional
        self.time = 0.0
        self.max_edges = self.nmax - 1

        # frozen normalization scales
        self.total_mass_final = self.mu * (self.length0 + fed_total)
        self.gravity_norm = float(np.linalg.norm(self.gravity)) or 9.81
        self.gravito_bending_length = float(
            (self.bending_stiffness / (self.mu * self.gravity_norm)) ** (1.0 / 3.0)
        )
        self.reference_force = float(self.total_mass_final * self.gravity_norm)
        self.reference_torque = float(self.reference_force * self.gravito_bending_length)
        self.constraint_scale = float(self.dL0)

        self._solveh = None
        try:
            from scipy.linalg import solveh_banded
            self._solveh = solveh_banded
        except Exception:
            self._solveh = None

        self._reset_budget()
        self._steps = 0
        self._max_stretch = 0.0

    # -------------------------------------------------------------- budget
    def _reset_budget(self) -> None:
        self._imp_grav = np.zeros(3)
        self._imp_drag = np.zeros(3)
        self._imp_sub = np.zeros(3)
        self._imp_noz = np.zeros(3)
        self._pos_grav = np.zeros(3)
        self._pos_drag = np.zeros(3)
        self._pos_sub = np.zeros(3)
        self._pos_noz = np.zeros(3)
        self._wsum_grav = 0.0
        self._wsum_drag = 0.0
        self._wsum_sub = 0.0
        self._wsum_noz = 0.0
        self._torque_grav = np.zeros(3)
        self._torque_drag = np.zeros(3)
        self._torque_sub = np.zeros(3)
        self._torque_noz = np.zeros(3)
        self._dt_accum = 0.0

    def _accumulate(self, name: str, force: np.ndarray, position: np.ndarray, dt: float) -> None:
        imp = getattr(self, f"_imp_{name}")
        pos = getattr(self, f"_pos_{name}")
        wsum = getattr(self, f"_wsum_{name}")
        torque = getattr(self, f"_torque_{name}")
        imp += force.sum(axis=0) * dt
        weight = np.linalg.norm(force, axis=1) * dt
        wsum += float(weight.sum())
        pos += (position * weight[:, None]).sum(axis=0)
        torque += np.cross(position, force).sum(axis=0) * dt
        setattr(self, f"_imp_{name}", imp)
        setattr(self, f"_pos_{name}", pos)
        setattr(self, f"_wsum_{name}", wsum)
        setattr(self, f"_torque_{name}", torque)

    def _node_masses(self, n: int) -> np.ndarray:
        """Lumped mass for the ``n`` active nodes, front-extruding fraction aware."""
        mass = np.full(n, self.mu * self.dL0)
        if n >= 2:
            mass[0] = self.mu * self.dL0 * (self.frac + 0.5)
            mass[-1] = self.mu * self.dL0 * 0.5
        elif n == 1:
            mass[0] = self.mu * self.dL0 * self.frac
        return mass

    # -------------------------------------------------------------- physics
    def _bending_force(self, x: np.ndarray) -> np.ndarray:
        return -(self.bending_stiffness / self.dL0 ** 3) * _fourth_difference(x)

    def _bending_damp_force(self, v: np.ndarray) -> np.ndarray:
        return -(self.beta * self.bending_stiffness / self.dL0 ** 3) * _fourth_difference(v)

    def _project_inextensible(self, x: np.ndarray, mass: np.ndarray, free_lo: int,
                               free_hi: int) -> tuple[np.ndarray, np.ndarray]:
        """Fast tridiagonal (single Newton step) inextensibility projection.

        Returns corrected positions and the per-edge tension magnitudes
        (Lagrange multiplier / dt^2), one entry per edge in ``x`` (length n-1).
        """
        n = x.shape[0]
        tension = np.zeros(max(n - 1, 0))
        if free_hi <= free_lo:
            return x, tension
        # Edges with at least one free endpoint: [free_lo-1 .. free_hi-1] inclusive,
        # clipped to valid edge range [0, n-2].
        e_lo = max(free_lo - 1, 0)
        e_hi = min(free_hi, n - 1)          # exclusive edge upper bound (edge idx e connects e,e+1)
        if e_hi <= e_lo:
            return x, tension
        edges = np.arange(e_lo, e_hi)
        d = x[edges + 1] - x[edges]
        length = np.linalg.norm(d, axis=1)
        length = np.maximum(length, _EPS)
        u = d / length[:, None]
        c = length - self.dL0
        inv_mass = 1.0 / mass
        w_i = np.where((edges >= free_lo) & (edges < free_hi), inv_mass[np.clip(edges, 0, n - 1)], 0.0)
        w_j = np.where((edges + 1 >= free_lo) & (edges + 1 < free_hi),
                        inv_mass[np.clip(edges + 1, 0, n - 1)], 0.0)
        denom = np.maximum(w_i + w_j, _EPS)
        m = len(edges)
        lower = np.zeros(m)
        diag = (w_i + w_j)
        upper = np.zeros(m)
        for k in range(m - 1):
            if edges[k] + 1 == edges[k + 1]:
                shared = edges[k] + 1
                if free_lo <= shared < free_hi:
                    coupling = inv_mass[shared] * float(np.dot(u[k], u[k + 1]))
                    upper[k] = -coupling
                    lower[k + 1] = -coupling
        diag = np.maximum(diag, 1e-9)
        ab = np.zeros((2, m))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        try:
            if self._solveh is not None:
                lam = self._solveh(ab, c, lower=False)
            else:
                lam = c / diag
        except Exception:
            lam = c / diag
        dx = np.zeros_like(x)
        for k, e in enumerate(edges):
            i, j = e, e + 1
            if free_lo <= i < free_hi:
                dx[i] += inv_mass[i] * lam[k] * u[k]
            if free_lo <= j < free_hi:
                dx[j] -= inv_mass[j] * lam[k] * u[k]
        tension[edges] = lam / max(self.dt ** 2, _EPS)
        return x + dx, tension
