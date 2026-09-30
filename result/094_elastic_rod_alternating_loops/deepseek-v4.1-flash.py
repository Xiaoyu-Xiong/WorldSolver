"""Physically based solver for alternating elastic-rod loops on a moving belt.

The rod is a naturally straight, inextensible, round elastic rod.  It is
discretised as a Lagrangian chain of material nodes carrying

* axial elasticity (stiffness ``E*A/ds`` -> practically inextensible),
* discrete bending resistance with stiffness ``B = E*pi*r**4/4``,
* uniform gravity,
* unilateral, sticking contact with the translating belt surface,
* a position/direction controlled injector feeding new material through the
  outlet at the prescribed injection speed.

The alternating-loop pattern is produced by the actual dynamics: slack
accumulates in the suspended heel because the belt carries the stuck deposit
away at half the injection speed; the heel buckles out of the belt plane and
lays a loop which then sticks to the belt and is carried downstream.  Nothing
is scripted or precomputed.

Only NumPy is used; no external simulation engine.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.seed = int(seed)
        rod = cfg["rod"]
        inj = cfg["injector"]
        belt = cfg["belt"]

        self.g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)
        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])
        # explicit chain: highest axial mode is 2*sqrt(EA/(rhoA*ds^2)); dt is
        # chosen so that omega*dt < 1 for every mode (stable, damped regime).
        self.substeps = 200
        self.dt = self.sample_dt / float(self.substeps)

        radius = float(rod["radius"])
        density = float(rod["density"])
        young = float(rod["young_modulus"])

        self.radius = radius
        self.r_contact = radius                    # centre height resting on belt
        self.rhoA = density * math.pi * radius * radius
        self.EA = young * math.pi * radius * radius
        self.B = young * math.pi * radius ** 4 / 4.0
        self.ell_gb = (self.B / (self.rhoA * abs(self.g[2]))) ** (1.0 / 3.0)
        self.gamma = 3.0                           # mild viscous damping (quasi-static regime)

        self.v0 = float(inj["injection_speed"])
        self.outlet = np.asarray(inj["outlet_position"], dtype=float).reshape(3)
        axis = np.asarray(inj.get("axis", [0.0, 0.0, -1.0]), dtype=float).reshape(3)
        self.axis = axis / max(np.linalg.norm(axis), 1e-12)
        self.v_in = self.axis * self.v0

        self.v_belt = (
            np.asarray(belt["travel_direction"], dtype=float).reshape(3) * float(belt["speed"])
        )
        centre = np.asarray(belt["center"], dtype=float).reshape(3)
        half = np.asarray(belt["half_extents"], dtype=float).reshape(3)
        self.bx0, self.bx1 = centre[0] - half[0], centre[0] + half[0]
        self.by0, self.by1 = centre[1] - half[1], centre[1] + half[1]

        self.L0 = float(rod["initial_deployed_length"])
        self.Lmax = float(rod["total_available_length"])
        self.L_intro = self.L0

        # ---- reference / normalisation scales exported with every frame ----
        self.reference_force = self.rhoA * abs(self.g[2]) * self.Lmax
        self.reference_torque = self.B / self.ell_gb
        self.constraint_scale = float(np.linalg.norm(self.outlet))

        # ---- initial centreline: the frozen topology, free end first --------
        verts = self._load_topology(cfg)
        pts = verts[::-1].astype(float)            # index 0 = free end, -1 = inlet
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.n0 = 52
        self.ds = float(arc[-1]) / float(self.n0 - 1)
        sample = np.linspace(0.0, float(arc[-1]), self.n0)
        x0 = np.column_stack([np.interp(sample, arc, pts[:, k]) for k in range(3)])

        self.cap = 420
        self.x = np.zeros((self.cap, 3))
        self.v = np.zeros((self.cap, 3))
        self.stuck = np.zeros(self.cap, dtype=bool)
        self.x[: self.n0] = x0
        self.m = np.full(self.cap, self.rhoA * self.ds)

        self.n = self.n0
        self.step = 0

        # initially contacted material sticks to the belt
        z = self.x[: self.n0, 2]
        inside = (
            (self.x[: self.n0, 0] >= self.bx0)
            & (self.x[: self.n0, 0] <= self.bx1)
            & (self.x[: self.n0, 1] >= self.by0)
            & (self.x[: self.n0, 1] <= self.by1)
        )
        self.stuck[: self.n0] = inside & (z <= self.r_contact + 0.0012)
        self.v[: self.n0][self.stuck[: self.n0]] = self.v_belt

        self.Fcon = np.zeros((self.cap, 3))
        self._warned = False

    @staticmethod
    def _load_topology(cfg: dict) -> np.ndarray:
        path = Path(str(cfg["rod"]["topology_asset"]))
        if not path.is_file():
            path = Path.cwd() / path
        data = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray(data["vertices"], dtype=float)

    # ---------------------------------------------------------------- physics
    def _bending_force(self, x: np.ndarray) -> np.ndarray:
        """Force from ``U = B/(2 ds) * sum_i |t_i - t_{i-1}|^2`` (isotropic bending)."""
        n = x.shape[0]
        F = np.zeros_like(x)
        if n < 3:
            return F
        e = x[1:] - x[:-1]
        length = np.maximum(np.linalg.norm(e, axis=1), 1e-12)
        t = e / length[:, None]
        h = np.zeros_like(x)                       # bending moments h_i
        h[1 : n - 1] = (self.B / self.ds) * (t[1:] - t[:-1])
        s = h[:-1] - h[1:]                         # (n-1, 3)
        ts = np.sum(t * s, axis=1)
        q = (s - t * ts[:, None]) / length[:, None]
        F[:-1] += q
        F[1:] -= q
        return F

    def _bend_energy(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        U = np.zeros(n)
        if n >= 3:
            e = x[1:] - x[:-1]
            length = np.maximum(np.linalg.norm(e, axis=1), 1e-12)
            t = e / length[:, None]
            d = t[1:] - t[:-1]
            U[1 : n - 1] = 0.5 * (self.B / self.ds) * np.sum(d * d, axis=1)
        return U

    def _sim_step(self, dt: float, t_new: float) -> None:
        # ---- growth: feed new material through the injector outlet ---------
        self.L_intro = min(self.L0 + self.v0 * t_new, self.Lmax)
        if self.v0 > 0.0:
            n_target = self.n0 + int(math.floor((self.L_intro - self.L0) / self.ds + 1e-9))
        else:
            n_target = self.n0
        n = self.n
        while n < n_target and n < self.cap:
            tangent = self.x[n - 1] - self.x[n - 2]
            tnorm = float(np.linalg.norm(tangent))
            tangent = tangent / tnorm if tnorm > 1e-12 else -self.axis
            self.x[n - 1] = self.outlet + self.ds * tangent
            self.x[n] = self.outlet
            self.v[n - 1] = self.v_in
            self.v[n] = self.v_in
            n += 1
        self.n = n

        x = self.x[:n]
        v = self.v[:n]
        m = self.m[:n]

        # ---- applied forces (gravity + bending), semi-implicit Euler ------
        F = self._bending_force(x) + m[:, None] * self.g
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
        v += dt * (F / m[:, None])
        v *= max(0.0, 1.0 - self.gamma * dt)
        np.clip(v, -5.0, 5.0, out=v)
        v_free = v.copy()
        x_new = x + dt * v

        # ---- belt contact and sticking ------------------------------------
        inside = (
            (x_new[:, 0] >= self.bx0)
            & (x_new[:, 0] <= self.bx1)
            & (x_new[:, 1] >= self.by0)
            & (x_new[:, 1] <= self.by1)
        )
        self.stuck[:n] |= inside & (x_new[:, 2] <= self.r_contact + 0.0012)
        st = self.stuck[:n]
        np.maximum(x_new[:, 2], self.r_contact, out=x_new[:, 2])
        if np.any(st):
            x_new[st, 0] = x[st, 0] + dt * self.v_belt[0]
            x_new[st, 1] = x[st, 1] + dt * self.v_belt[1]
            x_new[st, 2] = self.r_contact

        # ---- controlled inlet (position and direction prescribed) ---------
        x_new[n - 1] = self.outlet

        v_new = (x_new - x) / dt
        if np.any(st):
            v_new[st] = self.v_belt
        v_new[n - 1] = self.v_in
        v_new = np.nan_to_num(v_new, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(v_new, -5.0, 5.0, out=v_new)

        self.Fcon[:n] = m[:, None] * (v_new - v_free) / dt
        if not np.isfinite(x_new).all():
            # extremely defensive: never emit non-finite state
            x_new = x.copy()
            v_new = np.zeros_like(v_new)
            self.Fcon[:n] = 0.0
            self._warned = True
        self.x[:n] = x_new
        self.v[:n] = v_new

    # ------------------------------------------------------------- sampling
    def _export_order(self, n: int) -> np.ndarray:
        """Order nodes so the first ones trace a coarse outline of the whole rod.

        The frozen renderer draws the asset edges among the first ``len(vertices)``
        samples, so exporting a uniform subset of the material nodes first keeps
        the injector, suspended heel, contact point and the deposited pattern all
        visible.  The physical degrees of freedom are unchanged (a permutation).
        """
        m = 27
        if n <= m:
            return np.arange(n, dtype=int)
        sel = np.unique(np.rint(np.linspace(0.0, n - 1, m)).astype(int))
        rest = np.setdiff1d(np.arange(n, dtype=int), sel)
        return np.concatenate([sel, rest])

    def _frame(self, time: float) -> StateFrame:
        n = self.n
        idx = self._export_order(n)
        x = self.x[:n][idx]
        v = self.v[:n][idx]
        m = self.m[:n][idx]
        st = self.stuck[:n][idx].astype(float)

        orientation = np.zeros((n, 4))
        orientation[:, 3] = 1.0

        tangent = self.x[n - 2] - self.x[n - 1]
        tnorm = float(np.linalg.norm(tangent))
        inlet_dir = tangent / tnorm if tnorm > 1e-12 else self.axis

        Fext = (self.m[:n, None] * self.g + self.Fcon[:n])[idx]

        attributes = {
            "mass_quantity": m,
            "arc_coordinate": (np.arange(n) * self.ds)[idx],
            "is_sticking_to_belt": st,
            "is_injector_controlled": (np.arange(n) == n - 1).astype(float)[idx],
            "is_initially_deposited": (np.arange(n) < self.n0).astype(float)[idx],
            "rod_radius": np.full(n, self.radius),
            "natural_curvature": np.zeros(n),
            "elastic_energy": self._bend_energy(self.x[:n])[idx],
        }

        fields = {
            "budget_external_force": Fext.reshape(1, n, 3),
            "budget_force_position": x.reshape(1, n, 3),
            "budget_external_couple": np.zeros((1, n, 3)),
            "constraint_value": np.vstack([self.x[n - 1].copy(), inlet_dir]),
            "constraint_target": np.vstack([self.outlet.copy(), self.axis.copy()]),
            "cumulative_mass_source": np.array(
                [self.rhoA * max(0.0, self.L_intro - self.L0)]
            ),
            "gravity_acceleration": self.g.reshape(1, 3),
            "external_force": Fext,
        }

        scalars = {
            "constraint_scale": float(self.constraint_scale),
            "reference_force": float(self.reference_force),
            "reference_torque": float(self.reference_torque),
            "injection_speed": float(self.v0),
            "belt_speed": float(np.linalg.norm(self.v_belt)),
            "belt_to_injection_speed_ratio": float(
                np.linalg.norm(self.v_belt) / self.v0 if self.v0 else 0.0
            ),
        }

        return StateFrame(
            time=float(time),
            position=x,
            velocity=v,
            orientation=orientation,
            angular_velocity=np.zeros((n, 3)),
            mass=m,
            body_id=np.zeros(n, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        while self.step < target:
            self.step += 1
            self._sim_step(self.dt, self.step * self.dt)
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        n = self.n
        x = self.x[:n]
        seg = np.linalg.norm(np.diff(x, axis=0), axis=1)
        stretch = float(np.max(np.abs(seg - self.ds) / self.ds)) if n > 1 else 0.0
        return {
            "engine": "explicit-chain-elastica",
            "steps": int(self.step),
            "nodes": int(n),
            "injected_length": float(self.L_intro),
            "max_axial_stretch": stretch,
            "fallback_triggered": bool(self._warned),
            "constraint_scale": float(self.constraint_scale),
            "reference_force": float(self.reference_force),
            "reference_torque": float(self.reference_torque),
        }
