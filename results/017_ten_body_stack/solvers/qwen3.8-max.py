"""Solver for 017_ten_body_stack.

Physics: ten axis-aligned rigid boxes (0.9 x 0.9 x 0.2 m) stacked above a
static ground plane settle under gravity.  Because every body stays
axis-aligned and no horizontal forces act, the motion reduces to vertical
translation of each body under gravity plus contact forces at the 11
contact interfaces (ground-body_01 and body_i-body_{i+1}, i=1..9).

Contacts use a stiff linear penalty (spring-damper, repulsive-only) model:

    f = max(0, -k*g - c*gdot)      (g = signed gap, gdot = gap rate)

with k chosen so static deflection under the full stack load is << the body
thickness (sub-millimetre penetration), and damping ratio ~0.3 so impact
oscillations decay quickly and the stack settles.  Integration is
semi-implicit Euler with a fixed substep (h = 2.5e-4 s) that keeps the
penalty dynamics well inside its stability region.

All required trajectory fields are exported per sample:
  fields.contact_signed_distance [C=11], fields.external_force [N,3],
  fields.interaction_force [K=22,3], scalars.contact_length_scale,
  state.position/velocity/mass/body_id (+ orientation, angular_velocity).
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81])

# Geometry (from scene_geometry.yaml)
BODY_HALF_Z = 0.1          # half thickness of each box
GROUND_TOP_Z = 0.0         # top surface of the ground box

# Penalty contact parameters
K_CONTACT = 4.0e5          # N/m contact stiffness
ZETA = 0.3                 # damping ratio for contacts
M_EFF = 0.5                # effective mass of a body-body contact pair
C_CONTACT = 2.0 * ZETA * np.sqrt(K_CONTACT * M_EFF)

H_SUB = 2.5e-4             # integration substep (s)


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        self._seed = int(seed)
        self._pos0 = np.asarray(frame.position, dtype=float).copy()
        self._vel0 = np.asarray(frame.velocity, dtype=float).copy()
        n = self._pos0.shape[0]
        self._n = n
        mass = frame.mass
        self._mass = (np.ones(n, dtype=float) if mass is None
                      else np.asarray(mass, dtype=float).copy())
        self._body_id = (np.arange(1, n + 1, dtype=int) if frame.body_id is None
                         else np.asarray(frame.body_id).copy())

        # state (only z evolves; x,y velocities remain zero -> x,y fixed)
        self._z = self._pos0[:, 2].copy()
        self._vz = self._vel0[:, 2].copy()
        self._t = 0.0

        # contact pairs: (a, b) with a = lower index (-1 means ground)
        pairs = [(-1, 0)] + [(i, i + 1) for i in range(n - 1)]
        self._pairs = pairs
        self._n_contacts = len(pairs)

        # cache constant external force (gravity only)
        self._ext_force = np.tile(GRAVITY, (n, 1)) * self._mass[:, None]

        self._max_pen = 0.0
        self._steps = 0

    # ------------------------------------------------------------------
    # contact kinematics / forces
    # ------------------------------------------------------------------
    def _gaps(self) -> tuple[np.ndarray, np.ndarray]:
        z, vz = self._z, self._vz
        gaps = np.empty(self._n_contacts)
        rates = np.empty(self._n_contacts)
        for c, (a, b) in enumerate(self._pairs):
            if a < 0:
                gaps[c] = z[b] - BODY_HALF_Z - GROUND_TOP_Z
                rates[c] = vz[b]
            else:
                gaps[c] = z[b] - z[a] - 2.0 * BODY_HALF_Z
                rates[c] = vz[b] - vz[a]
        return gaps, rates

    def _contact_forces(self, gaps: np.ndarray, rates: np.ndarray) -> np.ndarray:
        pen = np.minimum(gaps, 0.0)            # <= 0 penetration
        f = -K_CONTACT * pen - C_CONTACT * rates
        active = gaps < 0.0
        return np.where(active, np.maximum(f, 0.0), 0.0)

    def _step(self, h: float) -> None:
        gaps, rates = self._gaps()
        f = self._contact_forces(gaps, rates)
        self._max_pen = max(self._max_pen, float(-np.min(gaps)) if len(gaps) else 0.0)

        # net vertical force on each body: gravity + contact (+f on upper body,
        # -f on lower body of each pair; ground reaction only loads body 0)
        fz = self._mass * GRAVITY[2]
        for c, (a, b) in enumerate(self._pairs):
            fz[b] += f[c]
            if a >= 0:
                fz[a] -= f[c]

        # semi-implicit Euler
        self._vz += (fz / self._mass) * h
        self._z += self._vz * h
        self._t += h
        self._steps += 1

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self._t - 1e-9:
            raise ValueError("advance_to called with time before current time")
        remaining = target - self._t
        while remaining > 1e-12:
            h = min(H_SUB, remaining)
            self._step(h)
            remaining = target - self._t

        n = self._n
        position = self._pos0.copy()
        position[:, 2] = self._z
        velocity = np.zeros((n, 3), dtype=float)
        velocity[:, 2] = self._vz

        gaps, rates = self._gaps()
        f = self._contact_forces(gaps, rates)

        # interaction forces: two rows per contact pair (equal and opposite,
        # force on lower body then force on upper body), +z is the contact
        # normal direction for this axis-aligned stack.
        inter = np.zeros((2 * self._n_contacts, 3), dtype=float)
        inter[0::2, 2] = -f   # on lower body / ground (reaction)
        inter[1::2, 2] = f    # on upper body

        return StateFrame(
            time=target,
            position=position,
            velocity=velocity,
            orientation=np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
            angular_velocity=np.zeros((n, 3), dtype=float),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            fields={
                "contact_signed_distance": gaps.astype(float),
                "external_force": self._ext_force.copy(),
                "interaction_force": inter,
            },
            scalars={"contact_length_scale": 2.0 * BODY_HALF_Z},
        )

    def diagnostics(self) -> dict[str, object]:
        kinetic = 0.5 * float(np.sum(self._mass * self._vz ** 2))
        return {
            "time": self._t,
            "steps": self._steps,
            "max_penetration_m": self._max_pen,
            "kinetic_energy_J": kinetic,
            "n_contacts": self._n_contacts,
        }
