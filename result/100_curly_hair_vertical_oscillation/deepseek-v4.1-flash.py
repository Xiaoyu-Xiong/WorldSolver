"""Solver for the vertically driven curly hair clump (scene 100).

Physical model
--------------
The clump is represented by one physical guide strand: a discrete,
inextensible, bending-elastic rod (a Cosserat centreline reduced to its
position degrees of freedom).  Each of the ``N`` nodes carries the mass of the
material segment it represents.  The discretisation contains

* bending elasticity with the frozen natural geometry (a circular helix) as its
  stress-free curvature, so the strand keeps its curl;
* an exact (directly solved) inextensibility projection so the *material*
  arclength stays equal to ``material_length`` at every step;
* gravity, ambient drag against still air and a small explicit internal
  (curvature-rate) damping;
* a kinematic root node clamped to the prescribed vertical sinusoid with fixed
  orientation; the tip is free.

Numerics
--------
The bending stiffness is the stiffest term, so it is integrated with a
linearly-implicit (A-stable) semi-implicit Euler update::

    (M + h^2 K_b + h C_drag) v' = M v + h (F_ext + f0 - K_b p)

whose left-hand matrix is constant and factorised once.  The inextensibility
constraint ``|p_{i+1}-p_i| = ds`` is then enforced by a direct Newton
projection in segment space (a small tridiagonal solve), followed by the
consistent velocity update ``v = (p_new - p_old)/h``.  The scheme is
deterministic, unconditionally stable in the stiff bending modes and resolves
the root-driven axial/curling response of the clump.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic elastic-rod solver for the curly hair clump."""

    #: Young's modulus of the keratin guide strand (Pa).  The frozen config does
    #: not fix a bending modulus, so a standard physical value is used.
    _YOUNGS_MODULUS = 4.0e9
    _SUBSTEP = 1.0e-3

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.config = cfg
        guide = cfg["physical_guide"]
        self.drive = cfg["root_motion"]
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.n = self.pos.shape[0]

        self.material_length = float(guide["material_length"])
        self.ds = self.material_length / (self.n - 1)
        radius = float(guide["radius"])
        density = float(guide["density"])
        self.line_density = density * np.pi * radius**2
        self.bending_modulus = self._YOUNGS_MODULUS * np.pi * radius**4 / 4.0
        self.internal_gamma = float(guide["internal_damping_gamma"])
        self.drag_nu = float(guide["ambient_drag_nu"])

        # Per-node ambient drag (half weight at the free tip).
        self.drag = np.full(self.n, self.drag_nu * self.ds)
        self.drag[0] = 0.0
        self.drag[-1] *= 0.5

        self.m_free = self.mass[1:]
        self.inv_m_free = np.where(self.m_free > 0.0, 1.0 / self.m_free, 0.0)

        # Frozen natural (stress-free) discrete curvature of the initial helix:
        # second differences are translation invariant, so the natural shape
        # follows the translating root without any extra bookkeeping.
        self.rest_curvature = self.pos[2:] - 2.0 * self.pos[1:-1] + self.pos[:-2]

        # Second-difference operator D (rows = interior nodes 1..n-2).
        rows = np.arange(self.n - 2)
        dmat = csr_matrix(
            (np.concatenate([np.full(self.n - 2, 1.0), np.full(self.n - 2, -2.0), np.full(self.n - 2, 1.0)]),
             (np.concatenate([rows, rows, rows]), np.concatenate([rows, rows + 1, rows + 2]))),
            shape=(self.n - 2, self.n),
        )
        self.d_free = dmat[:, 1:]
        self.d_free_t = self.d_free.T.tocsr()
        self.d_root = dmat[:, 0].toarray().ravel()  # unit vector, row 0 -> root

        self.c_bend = self.bending_modulus / self.ds**3
        self.k_bend = (self.c_bend * (self.d_free_t @ self.d_free)).toarray()
        self.c_int_damp = self.internal_gamma * self.bending_modulus / self.ds

        # Constant implicit operator for the free nodes.
        h = self._SUBSTEP
        mat = np.diag(self.m_free) + h * h * self.k_bend + h * np.diag(self.drag[1:])
        self.chol = cho_factor(mat, lower=True)

        # Prescribed root state and bookkeeping.
        self.rest_position = np.asarray(self.drive["rest_position"], dtype=float)
        self.axis = np.asarray(self.drive["axis"], dtype=float)
        self.axis = self.axis / np.linalg.norm(self.axis)
        self.amplitude = float(self.drive["amplitude"])
        self.omega = 2.0 * np.pi * float(self.drive["frequency"])
        self.start_time = float(self.drive["start_time"])
        self.phase = float(self.drive.get("phase", 0.0))

        self.pos[0] = self._root_state(0.0)[0]
        self.vel[0] = 0.0
        self.time = 0.0
        self.cum_impulse = np.zeros(3)
        self.cum_gravity_impulse = np.zeros(3)
        self.cum_drag_impulse = np.zeros(3)
        self.last_terms = np.zeros((3, 3))
        self.total_mass = float(self.mass.sum())
        drive_speed = self.amplitude * self.omega
        self.reference_momentum = max(self.total_mass * drive_speed, 1.0e-12)
        self.constraint_scale = max(self.material_length, self.ds)
        self.pending = []

    # ---------------------------------------------------------------- helpers
    def _root_state(self, time: float):
        if time <= self.start_time:
            return self.rest_position.copy(), np.zeros(3)
        phase = self.omega * (time - self.start_time) + self.phase
        offset = self.amplitude * np.sin(phase)
        return (
            self.rest_position + self.axis * offset,
            self.axis * (self.amplitude * self.omega * np.cos(phase)),
        )

    def _project(self, pos: np.ndarray) -> None:
        """Direct Newton projection onto the inextensibility manifold."""
        ds = self.ds
        wnode = np.zeros(self.n)
        wnode[1:] = self.inv_m_free
        for _ in range(4):
            diff = pos[1:] - pos[:-1]
            length = np.linalg.norm(diff, axis=1)
            length = np.maximum(length, 1.0e-12)
            tangent = diff / length[:, None]
            violation = length - ds
            if np.max(np.abs(violation)) < 1.0e-14:
                break
            diag = wnode[:-1] + wnode[1:]
            off = -wnode[1:-1] * np.sum(tangent[:-1] * tangent[1:], axis=1)
            amat = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
            lam = np.linalg.solve(amat, -violation)
            term = np.zeros_like(pos)
            term[:-1] -= lam[:, None] * tangent
            term[1:] += lam[:, None] * tangent
            pos += wnode[:, None] * term

    def _substep(self, h: float, time_new: float) -> None:
        root_pos, root_vel = self._root_state(time_new)
        free = slice(1, self.n)
        p_free = self.pos[free]
        v_free = self.vel[free]

        # External body forces (gravity) and the tiny internal viscous damping.
        gravity_force = self.m_free[:, None] * self.gravity[None, :]
        bend_vel = self.d_free @ v_free
        damp_force = -self.c_int_damp * (self.d_free_t @ bend_vel)

        # Root contribution to the bending force (prescribed node).
        root_stencil = np.zeros_like(self.rest_curvature)
        root_stencil[0] = root_pos
        f0 = -self.c_bend * (self.d_free_t @ (root_stencil - self.rest_curvature))

        rhs = self.m_free[:, None] * v_free + h * (
            gravity_force + damp_force + f0 - self.k_bend @ p_free
        )
        v_new = cho_solve(self.chol, rhs)

        pos_try = self.pos.copy()
        pos_try[free] = p_free + h * v_new
        pos_try[0] = root_pos
        self._project(pos_try)

        v_proj = (pos_try[free] - p_free) / h
        delta_p = np.sum(self.m_free[:, None] * (v_proj - v_free), axis=0)

        drag_force = -np.sum(self.drag[1:, None] * v_proj, axis=0)
        gravity_total = np.sum(gravity_force, axis=0)
        external_total = delta_p / h
        root_force = external_total - gravity_total - drag_force

        self.cum_impulse += delta_p
        self.cum_gravity_impulse += gravity_total * h
        self.cum_drag_impulse += drag_force * h
        self.last_terms = np.vstack([gravity_total, drag_force, root_force])

        self.pos[free] = pos_try[free]
        self.pos[0] = root_pos
        self.vel[free] = v_proj
        self.vel[0] = root_vel
        self.time = time_new

    # -------------------------------------------------------------- interface
    def advance_to(self, time: float):
        time = float(time)
        count = int(round((time - self.time) / self._SUBSTEP))
        if count > 0:
            h = (time - self.time) / count
            for step in range(1, count + 1):
                self._substep(h, self.time + h)
        return self._frame()

    def _frame(self) -> StateFrame:
        n = self.n
        node_index = np.arange(n, dtype=int)
        mass_quantity = self.mass.reshape(n, 1)
        frame = StateFrame(
            time=float(self.time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": mass_quantity,
                "node_index": node_index,
                "is_root": (node_index == 0).astype(np.float64),
                "kinematic_root": (node_index == 0).astype(np.float64),
                "is_free_tip": (node_index == n - 1).astype(np.float64),
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_impulse.reshape(1, 3),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "budget_domain_weights": np.ones((1, n)),
                "budget_external_force": self.last_terms.reshape(1, 3, 3),
                "constraint_value": self.pos[0].reshape(1, 3),
                "constraint_target": self._root_state(self.time)[0].reshape(1, 3),
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "material_length": self.material_length,
                "internal_damping_gamma": self.internal_gamma,
                "ambient_drag_nu": self.drag_nu,
                "constraint_scale": self.constraint_scale,
                "reference_momentum": self.reference_momentum,
                "characteristic_length": self.material_length,
                "total_mass": self.total_mass,
                "bending_modulus": self.bending_modulus,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "node_count": int(self.n),
            "material_length": self.material_length,
            "segment_length": self.ds,
            "bending_modulus": self.bending_modulus,
            "substep": self._SUBSTEP,
            "final_time": self.time,
            "cumulative_external_impulse": self.cum_impulse.tolist(),
        }
