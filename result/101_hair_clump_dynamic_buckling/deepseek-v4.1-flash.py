"""Physically-based solver for dynamic buckling of a vertically shaken hair clump.

Model
-----
The hair clump is a discrete elastic rod (one chain of ``N`` nodes) whose
segments are effectively inextensible.  The root is *clamped*: nodes 0 and 1
are rigidly attached to the kinematically driven root and keep the frozen root
orientation.  Nodes 2..N-1 are free and carry the distributed mass of the rod.

Physics included
----------------
* gravity,
* bending elasticity with the discrete curvature energy
  ``E_b = (EI/2) int kappa^2 ds``  ->  ``E_b = (EI/(2 h^3)) sum |e_i|^2``
  with ``e_i = x_{i+1} - 2 x_i + x_{i-1}`` (exact constant stiffness matrix),
* axial inextensibility through very stiff (but finite) elastic springs,
* Kelvin-Voigt internal hair damping proportional to the bending operator,
* ambient air drag ``nu`` against the ambient air velocity.

Numerics
--------
Backward-Euler time stepping.  The bending operator is used implicitly with its
exact constant Hessian, the axial springs are linearised with their exact
Hessian (rank-one per segment, geometric stiffness retained), and the drag /
internal damping are implicit as well.  The resulting constant-size SPD system
is solved directly at every substep, so the scheme is unconditionally stable for
any stiff parameter and remains deterministic and finite.  Momentum is accounted
for exactly: the cumulative external impulse equals the domain momentum change.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Elastic-rod solver for the shaken hair-clump buckling scene."""

    def initialize(self, scene: Any, seed: int) -> None:
        config = dict(scene.config)
        self._seed = int(seed)
        hair = config["hair_clump"]
        drive = config["root_motion"]

        self._g = np.asarray(config["gravity"], dtype=float).reshape(3)
        self._L = float(hair["material_length"])
        self._radius = float(hair["effective_radius"])
        self._rho = float(hair["density"])
        self._gamma = float(hair["internal_damping_gamma"])
        self._nu = float(hair["ambient_drag_nu"])
        self._v_air = np.asarray(hair["ambient_air_velocity"], dtype=float).reshape(3)

        # --- kinematic root drive -------------------------------------------
        self._rest = np.asarray(drive["rest_position"], dtype=float).reshape(3)
        axis = np.asarray(drive["axis"], dtype=float).reshape(3)
        self._axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        self._amp = float(drive["amplitude"])
        self._freq = float(drive["frequency"])
        self._phase = float(drive.get("phase", 0.0))
        self._t_start = float(drive["start_time"])
        self._omega = 2.0 * math.pi * self._freq

        # --- initial state from the frozen scene ----------------------------
        initial = scene.initial_state()
        x = np.array(initial.position, dtype=float)
        self._N = int(x.shape[0])
        self._h = self._L / float(self._N - 1)
        self._mass = np.array(initial.mass, dtype=float)
        self._v = np.zeros_like(x)

        # Clamped root: node 1 keeps a fixed offset from node 0, along the
        # (fixed) root orientation.  The clump hangs along -z.
        self._clamp_dir = np.array([0.0, 0.0, -1.0])
        x[0] = self._rest
        x[1] = self._rest + self._clamp_dir * self._h
        self._x = x

        # --- material parameters (effective modulus of the hair clump) ------
        E = 8.0e8
        area = math.pi * self._radius ** 2
        self._EI = E * math.pi * self._radius ** 4 / 4.0
        self._k_ax = E * area / self._h
        self._k_b = self._EI / self._h ** 3
        # ambient drag per node (kg/s) and Kelvin-Voigt internal damping ratio
        self._c_drag = self._nu * self._h
        self._c_int = self._gamma / self._rho

        # --- discrete bending operator (exact second difference) ------------
        n = self._N
        D = np.zeros((n - 2, n), dtype=float)
        for row, i in enumerate(range(1, n - 1)):
            D[row, i - 1] = 1.0
            D[row, i] = -2.0
            D[row, i + 1] = 1.0
        self._Kb = self._k_b * (D.T @ D)

        self._free = np.arange(2, n, dtype=int)
        nf = self._free.size
        self._m_free = self._mass[self._free]
        self._Mrep = np.repeat(self._m_free, 3)
        Kb_ff = self._Kb[np.ix_(self._free, self._free)]
        # coupling of the free nodes to the prescribed nodes 0 and 1
        self._Kb_cross = self._Kb[np.ix_(self._free, [0, 1])].sum(axis=1)  # (nf,3)
        C_ff = self._c_drag * np.eye(3 * nf) + self._c_int * Kb_ff

        self._dt = 2.5e-4
        self._A_base = (
            (self._Mrep / self._dt ** 2)[:, None] * np.eye(3 * nf)
            + Kb_ff
            + C_ff / self._dt
        )
        self._C_ff = C_ff

        # --- bookkeeping -----------------------------------------------------
        self._t = 0.0
        self._steps = 0
        self._P0 = (self._mass[:, None] * self._v).sum(axis=0)
        self._P = self._P0.copy()
        self._F_ext_last = np.zeros(3)
        total_mass = float(self._mass.sum())
        g_scale = float(abs(self._g[2])) if abs(self._g[2]) > 1e-12 else 1.0
        self._ref_momentum = max(total_mass * math.sqrt(g_scale * self._L), 1e-12)
        self._total_mass = total_mass
        self._mass_quantity = self._mass.copy()

    # ------------------------------------------------------------------ drive
    def _root_state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Kinematic root position / velocity (vertical sinusoid after a hold)."""
        if t <= self._t_start:
            return self._rest.copy(), np.zeros(3)
        tau = t - self._t_start
        arg = self._omega * tau + self._phase
        pos = self._rest + self._axis * (self._amp * math.sin(arg))
        vel = self._axis * (self._amp * self._omega * math.cos(arg))
        return pos, vel

    # ----------------------------------------------------------------- forces
    def _axial_force(self, x: np.ndarray) -> np.ndarray:
        d = x[1:] - x[:-1]
        length = np.linalg.norm(d, axis=1)
        length = np.maximum(length, 1e-12)
        u = d / length[:, None]
        f = self._k_ax * (length - self._h)[:, None] * u
        F = np.zeros_like(x)
        F[:-1] += f
        F[1:] -= f
        return F

    def _add_axial_hessian(self, A: np.ndarray, x: np.ndarray) -> None:
        d = x[1:] - x[:-1]
        length = np.linalg.norm(d, axis=1)
        length = np.maximum(length, 1e-12)
        u = d / length[:, None]
        # geometric transverse factor, kept non-negative for SPD-ness
        s = np.maximum(0.0, 1.0 - self._h / length)
        eye = np.eye(3)
        for j in range(1, self._N - 1):
            uu = np.outer(u[j], u[j])
            H = self._k_ax * (uu + s[j] * (eye - uu))
            i0, i1 = j, j + 1
            if i0 >= 2:
                a = 3 * (i0 - 2)
                A[a:a + 3, a:a + 3] += H
                if i1 >= 2:
                    b = 3 * (i1 - 2)
                    A[a:a + 3, b:b + 3] -= H
                    A[b:b + 3, a:a + 3] -= H
                    A[b:b + 3, b:b + 3] += H
            elif i1 >= 2:
                b = 3 * (i1 - 2)
                A[b:b + 3, b:b + 3] += H

    # ------------------------------------------------------------------ step
    def _step(self, dt: float) -> None:
        x = self._x
        v = self._v
        free = self._free
        nf3 = 3 * free.size
        v_free = v[free].reshape(-1)
        P_prev = (self._mass[:, None] * v).sum(axis=0)

        x0_new, v0_new = self._root_state(self._t + dt)
        delta0 = x0_new - x[0]

        F_ax = self._axial_force(x)
        Kb_x = self._Kb @ x

        rhs = (
            (self._m_free / dt)[:, None] * v[free]
            + self._m_free[:, None] * self._g
            + F_ax[free]
            - Kb_x[free]
            - self._Kb_cross @ delta0
            - (self._C_ff @ v_free).reshape(-1, 3)
            + (self._c_drag / dt) * self._v_air[None, :]
        )

        A = self._A_base.copy()
        self._add_axial_hessian(A, x)
        delta = np.linalg.solve(A, rhs.reshape(-1)).reshape(-1, 3)

        x[free] += delta
        v[free] = delta / dt
        x[0] = x0_new
        v[0] = v0_new
        x[1] = x0_new + self._clamp_dir * self._h
        v[1] = v0_new

        if not (np.isfinite(x).all() and np.isfinite(v).all()):
            raise FloatingPointError("non-finite rod state during time stepping")

        P_new = (self._mass[:, None] * v).sum(axis=0)
        self._F_ext_last = (P_new - P_prev) / dt
        self._P = P_new
        self._t += dt
        self._steps += 1

    # --------------------------------------------------------------- advance
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        tol = 1e-12 * max(1.0, abs(target))
        guard = 0
        while self._t < target - tol:
            h = min(self._dt, target - self._t)
            if h <= 1e-15:
                break
            self._step(h)
            guard += 1
            if guard > 100000000:
                raise RuntimeError("step loop failed to reach sample time")
        if abs(self._t - target) <= 1e-9:
            self._t = target
        return self._frame()

    # ----------------------------------------------------------------- frame
    def _frame(self) -> StateFrame:
        n = self._N
        x = self._x
        v = self._v

        seg = np.linalg.norm(x[1:] - x[:-1], axis=1)
        f_grav = (self._mass[:, None] * self._g[None, :]).sum(axis=0)
        f_drag = -self._c_drag * (v - self._v_air[None, :]).sum(axis=0)
        f_root = self._F_ext_last - f_grav - f_drag

        impulse = (self._P - self._P0).reshape(1, 3)
        index = np.arange(n, dtype=int)
        frame = StateFrame(
            time=float(self._t),
            position=x.copy(),
            velocity=v.copy(),
            mass=self._mass.copy(),
            attributes={
                "mass_quantity": self._mass.copy(),
                "node_index": index,
                "is_root": (index == 0).astype(int),
                "is_free_tip": (index == n - 1).astype(int),
                "kinematic_root": (index <= 1).astype(int),
            },
            fields={
                "budget_cumulative_external_impulse": impulse,
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "budget_domain_weights": np.ones((1, n)),
                "budget_external_force": np.stack((f_grav, f_drag, f_root))[None, :, :],
                "constraint_target": np.full(n - 1, self._h),
                "constraint_value": seg,
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "constraint_scale": float(self._h),
                "reference_momentum": float(self._ref_momentum),
            },
        )
        frame.validate()
        return frame

    # ----------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        x = self._x
        e = x[2:] - 2.0 * x[1:-1] + x[:-2]
        elastic_energy = 0.5 * self._k_b * float((e ** 2).sum())
        return {
            "dt": self._dt,
            "steps": int(self._steps),
            "node_count": int(self._N),
            "segment_length": float(self._h),
            "total_mass": float(self._total_mass),
            "elastic_energy": elastic_energy,
            "max_speed": float(np.linalg.norm(self._v, axis=1).max()),
            "reference_momentum": float(self._ref_momentum),
        }
