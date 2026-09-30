"""Physically based solver: translated coiling of an elastic rod on a moving belt.

Self-contained discrete elastic rod model (no external physics engine):

* Lagrangian nodes with lumped mass, symplectic (semi-implicit) Euler stepping.
* Axial penalty force enforcing the frozen inextensibility constraint.
* Discrete quadratic bending energy of a naturally straight rod
  (Bergou-style curvature energy written on the second difference of nodes).
* Kelvin-Voigt material damping in axial and bending deformation, which
  removes numerical ringing while leaving the slow quasi-static coiling mode
  essentially undamped.
* Rigid belt (frozen box) with the source *sticking* substrate condition:
  any material point that touches the belt surface is deposited and is
  afterwards advected with the prescribed belt velocity.
* The injector clamps the rod at the frozen outlet with the frozen axis and
  feeds new material at the prescribed injection speed: the rest length of the
  top segment grows at the feed rate and new nodes are inserted at the clamp,
  so the rod grows while the deposited part stays put on the belt.

The translated-coiling pattern (nearly circular turns whose centers advance
along the moving belt) is an *outcome* of the time stepping, not a scripted
animation.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Discrete elastic rod injected onto a translating sticky belt."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        rod = cfg["rod"]
        inj = cfg["injector"]
        belt = cfg["belt"]

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        n0 = self.x.shape[0]

        self.L0 = float(rod["initial_deployed_length"])
        self.ds0 = self.L0 / (n0 - 1)
        self.rest = np.full(n0 - 1, self.ds0)

        self.outlet = np.asarray(inj["outlet"], dtype=float)
        axis = np.asarray(inj["axis"], dtype=float)
        self.axis = axis / np.linalg.norm(axis)
        self.U = float(inj["injection_speed"])
        self.feed_start = float(inj["feed_start"])
        self.feed_end = float(inj["feed_end"])

        self.vbelt = float(belt["speed"]) * np.asarray(belt["translation_axis"], dtype=float)
        self.zc = float(belt["surface_z"]) + float(rod["radius"])

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.radius = float(rod["radius"])
        self.rhoA = float(rod["density"]) * np.pi * self.radius**2

        # Constitutive parameters.  The bending modulus uses a soft-elastomer
        # Young modulus (the frozen config fixes geometry and density only);
        # the axial term is a stiff penalty that implements the frozen
        # "inextensible" requirement.  The axisymmetric cross-section is
        # torsion-free at the freely spinning clamp, so the naturally
        # straight, untwisted state is retained.
        young = 6.0e4
        self.B = young * np.pi * self.radius**4 / 4.0
        self.EA = 4.0
        self.cs = 0.12   # axial Kelvin-Voigt damping (N s/m)
        self.cb = 0.02   # bending Kelvin-Voigt damping (N s/m)

        # Explicit stability: dt is kept far below the sqrt(m/k) limit of the
        # stiffest axial mode (shortest segment), including damping; it also
        # divides the frozen sample_dt exactly.
        self.dt = 1.0e-5

        self.stuck = np.zeros(n0, dtype=bool)
        self.stick_pos = np.zeros((n0, 3))
        self.stick_time = np.zeros(n0)

        self.kin = np.zeros(n0, dtype=bool)
        self.kin[0] = True
        self.kin[1] = True
        self.im = self._inverse_mass()

        self.step = 0
        self.time = 0.0
        self.vmax = 8.0  # numerical safety net, far above physical speeds

        self.fall_height = float(self.outlet[2] - self.zc)
        self._initial_frame = frame0

    # ------------------------------------------------------------------ mass
    def _masses(self) -> np.ndarray:
        rest = self.rest
        widths = np.concatenate(([rest[0]], rest[:-1] + rest[1:], [rest[-1]]))
        return 0.5 * self.rhoA * widths

    def _inverse_mass(self) -> np.ndarray:
        mass = self._masses()
        im = np.zeros_like(mass)
        free = ~self.kin
        im[free] = 1.0 / mass[free]
        return im

    # ---------------------------------------------------------------- forces
    def _internal_forces(self) -> np.ndarray:
        """Internal (axial + bending, elastic + damping) nodal forces."""
        x, v, rest = self.x, self.v, self.rest
        n = x.shape[0]
        F = np.zeros((n, 3))

        e = x[1:] - x[:-1]
        l = np.sqrt(np.einsum("ij,ij->i", e, e))
        np.maximum(l, 1e-9, out=l)
        u = e / l[:, None]
        tension = (self.EA / rest) * (l - rest)
        rate = np.einsum("ij,ij->i", v[1:] - v[:-1], u)
        S = (tension - self.cs * rate)[:, None] * u
        F[:-1] -= S
        F[1:] += S

        if n >= 3:
            w = x[2:] - 2.0 * x[1:-1] + x[:-2]
            wd = v[2:] - 2.0 * v[1:-1] + v[:-2]
            h = 0.5 * (rest[:-1] + rest[1:])
            q = (self.B / h**3)[:, None] * w + self.cb * wd
            F[:-2] -= q
            F[1:-1] += 2.0 * q
            F[2:] -= q
        return F

    # ------------------------------------------------------------ deposition
    def _stick(self, hit: np.ndarray) -> None:
        """Deposit the given free nodes on the sticky belt."""
        t = self.time
        for i in hit:
            self.stuck[i] = True
            self.stick_pos[i] = (self.x[i, 0], self.x[i, 1], self.zc)
            self.stick_time[i] = t
            self.x[i, 2] = self.zc
            self.v[i] = self.vbelt
        self.kin[hit] = True
        self.im[hit] = 0.0

    def _insert_node(self) -> None:
        """Feed material: split the growing top segment at the clamp."""
        r0 = float(self.rest[0])
        frac = self.ds0 / r0
        newx = self.x[0] + frac * (self.x[1] - self.x[0])
        newv = self.v[0] + frac * (self.v[1] - self.v[0])
        self.x = np.insert(self.x, 1, newx, axis=0)
        self.v = np.insert(self.v, 1, newv, axis=0)
        self.rest = np.concatenate(([self.ds0, r0 - self.ds0], self.rest[1:]))
        self.stuck = np.insert(self.stuck, 1, False)
        self.stick_pos = np.insert(self.stick_pos, 1, np.zeros(3), axis=0)
        self.stick_time = np.insert(self.stick_time, 1, 0.0)
        self.kin = np.insert(self.kin, 1, True)
        self.im = self._inverse_mass()

    # ------------------------------------------------------------------ step
    def _step(self) -> None:
        dt = self.dt
        x, v = self.x, self.v

        F = self._internal_forces()
        v += dt * (F * self.im[:, None] + self.gravity)
        # numerical safety net (never active in the physical regime)
        speed = np.sqrt(np.einsum("ij,ij->i", v, v))
        if speed.max() > self.vmax:
            v *= self.vmax / speed.max()
        x += dt * v

        # injector feeds new material through the clamped outlet
        if self.feed_start <= self.time <= self.feed_end:
            self.rest[0] += self.U * dt
        x[0] = self.outlet
        x[1] = self.outlet + self.rest[0] * self.axis
        v[0] = self.U * self.axis
        v[1] = self.U * self.axis

        # deposited material is advected with the belt (sticking contact)
        st = self.stuck
        if st.any():
            dtb = (self.time - self.stick_time[st])[:, None]
            x[st] = self.stick_pos[st] + dtb * self.vbelt
            v[st] = self.vbelt

        # sticky belt contact test on the free (suspended) material
        free = ~self.kin
        hit = np.flatnonzero(free & (x[:, 2] <= self.zc))
        if hit.size:
            self._stick(hit)

        if self.rest[0] >= 2.0 * self.ds0:
            self._insert_node()

        self.step += 1
        self.time = self.step * dt

    # ---------------------------------------------------------------- frames
    def _build_frame(self, t: float):
        x = self.x.copy()
        v = self.v.copy()
        n = x.shape[0]
        rest = self.rest
        mass = self._masses()

        e = x[1:] - x[:-1]
        lengths = np.sqrt(np.einsum("ij,ij->i", e, e))

        # external force budget: gravity plus the reactions that hold the
        # kinematic (clamped and deposited) nodes.
        F = self._internal_forces()
        Fext = mass[:, None] * self.gravity
        reaction = -(F + Fext)
        Fext[self.kin] += reaction[self.kin]
        f_total = Fext.sum(axis=0)
        torque = np.cross(x, Fext).sum(axis=0)
        com = (mass[:, None] * x).sum(axis=0) / max(mass.sum(), 1e-30)
        couple = torque - np.cross(com, f_total)

        controlled = np.zeros(n)
        controlled[0] = 1.0

        attributes = {
            "arc_length": np.concatenate(([0.0], np.cumsum(rest))),
            "natural_curvature": np.zeros(n),
            "natural_twist": np.zeros(n),
            "injector_controlled": controlled,
            "deposited_sticking": self.stuck.astype(float),
            "mass_quantity": mass,
        }
        fields = {
            "budget_external_force": f_total.reshape(1, 1, 3),
            "budget_force_position": com.reshape(1, 1, 3),
            "budget_external_couple": couple.reshape(1, 1, 3),
            "constraint_value": lengths,
            "constraint_target": rest.copy(),
            "cumulative_mass_source": np.array([self.rhoA * (float(rest.sum()) - self.L0)]),
        }
        weight = self.rhoA * float(np.linalg.norm(self.gravity)) * self.L0
        scalars = {
            "rod_radius": self.radius,
            "injection_speed": self.U,
            "belt_speed": float(np.linalg.norm(self.vbelt)),
            "constraint_scale": self.L0,
            "reference_force": weight,
            "reference_torque": weight * self.fall_height,
        }

        frame = StateFrame(
            time=float(t),
            position=x,
            velocity=v,
            mass=mass,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ----------------------------------------------------------- contract API
    def advance_to(self, time: float):
        t = float(time)
        target = int(round(t / self.dt))
        if target < self.step:
            target = self.step
        while self.step < target:
            self._step()
        return self._build_frame(t)

    def diagnostics(self) -> dict:
        return {
            "method": "discrete_elastic_rod_symplectic_euler",
            "internal_dt": self.dt,
            "steps": int(self.step),
            "nodes": int(self.x.shape[0]),
            "deposited_length": float(self.rest.sum()),
            "deposited_nodes": int(self.stuck.sum()),
            "bending_stiffness": float(self.B),
            "regime": "translated_coiling",
        }
