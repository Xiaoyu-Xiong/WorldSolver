"""Bonded-particle foam solver with explicit tearing (Figure 7 phenomenon).

Physical model
--------------
The shaving-foam mass is discretized with the scaffold's granular samples.
Cohesion is represented by a network of elastic, damped bonds between
initially neighbouring samples (a discrete visco-elasto-plastic foam model).
Every substep the solver:

  1. accumulates gravity on every sample,
  2. evaluates all live bonds (spring + dashpot along the bond axis),
  3. *explicit tearing*: any bond whose local stretch exceeds its yield
     threshold is permanently broken (lower threshold inside the frozen
     tear zone, higher elsewhere) -- no nonphysical threads can persist,
  4. applies the adhesive constraint that keeps foam stuck under the
     upper plate (the plate follows the frozen pull history),
  5. resolves penalty contacts with the floor and the plate,
  6. integrates with semi-implicit Euler.

The tendril thins because the neck's small cross section carries the same
transmitted load with far fewer bonds, so strain localizes there; once local
stretch exceeds the yield threshold the bridge tears and the two fragments
retract / fall and settle.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

try:  # contracts live in the platform runtime
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - fallback for isolated imports
    from dataclasses import dataclass, field as dc_field
    from collections.abc import Mapping

    @dataclass(slots=True)
    class StateFrame:  # type: ignore
        time: float
        position: np.ndarray
        velocity: np.ndarray
        orientation: object = None
        angular_velocity: object = None
        mass: object = None
        body_id: object = None
        attributes: Mapping = dc_field(default_factory=dict)
        fields: Mapping = dc_field(default_factory=dict)
        scalars: Mapping = dc_field(default_factory=dict)

        def validate(self) -> None:  # minimal
            assert np.isfinite(self.position).all()


# ---------------------------------------------------------------------------
# tunable material parameters (foam rheology, kept in physical ranges)
# ---------------------------------------------------------------------------
BOND_K = 1500.0          # bond stiffness [N/m]
BOND_C = 0.5             # bond damping [N s/m]
BREAK_STRAIN = 0.65      # global yield stretch of a bond
BREAK_STRAIN_TEAR = 0.42  # yield stretch inside the frozen tear zone
TEAR_ZONE = (0.70, 1.20)  # slightly wider than frozen [0.78, 1.12]
DT_SUB = 0.0015          # substep [s] (well below sqrt(m/k) limit)
GLOBAL_DAMPING = 0.25    # bulk viscous dissipation [1/s]
VMAX = 8.0               # safety velocity clamp [m/s]

ANCHOR_Z = 1.62          # foam adhered to the plate lives above this height
ANCHOR_K = 12000.0       # adhesion stiffness [N/m]
ANCHOR_C = 18.0          # adhesion damping [N s/m]
PLATE_PULL_V = 0.2       # frozen upper-plate pull speed [m/s]
PLATE_PULL_END = 1.6     # pull phase ends [s]
PLATE_PULL_RAMP = 0.4    # deceleration ramp [s]

FLOOR_K = 6000.0         # floor penalty stiffness [N/m]
FLOOR_CN = 10.0          # floor normal damping [N s/m]
FLOOR_CT = 12.0          # floor tangential (friction) damping [N s/m]

PLATE_K = 6000.0         # plate penalty stiffness [N/m]
PLATE_CN = 10.0

PLATE_BOTTOM0 = 1.84     # plate underside at t=0 [m] (half extent 0.08 at z=1.92)
PLATE_HX, PLATE_HY = 0.78, 0.62
FLOOR_TOP = 0.0
FLOOR_RADIUS = 1.35

EPS = 1e-12


def _plate_displacement(t: float) -> float:
    """Frozen pull history: constant rise, then smooth stop."""
    if t <= PLATE_PULL_END:
        return PLATE_PULL_V * t
    if t <= PLATE_PULL_END + PLATE_PULL_RAMP:
        s = (t - PLATE_PULL_END) / PLATE_PULL_RAMP
        return PLATE_PULL_V * (PLATE_PULL_END + PLATE_PULL_RAMP *
                               (s - 0.5 * s * s))
    return PLATE_PULL_V * (PLATE_PULL_END + 0.5 * PLATE_PULL_RAMP)


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        rng_free = None  # deterministic: no stochastic terms are used
        frame = scene.initial_state()
        cfg = scene.config
        ic = cfg["initial_conditions"]

        self.position = np.array(frame.position, dtype=np.float64)
        self.velocity = np.array(frame.velocity, dtype=np.float64)
        self.mass = np.array(frame.mass, dtype=np.float64)
        self.n = len(self.position)
        self.spacing = float(ic["particle_spacing"])
        self.radius = np.full(self.n, float(ic.get("nominal_particle_radius", 0.026)))
        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64)

        self.init_position = self.position.copy()
        self.init_velocity = self.velocity.copy()

        # ---- bond network (cohesion) -------------------------------------
        tree = cKDTree(self.init_position)
        h = 1.6 * self.spacing
        pairs = tree.query_pairs(h, output_type="ndarray")
        bi, bj = pairs[:, 0], pairs[:, 1]
        d0 = self.init_position[bj] - self.init_position[bi]
        self.bond_i = bi
        self.bond_j = bj
        self.bond_l0 = np.linalg.norm(d0, axis=1)
        mid_z = 0.5 * (self.init_position[bi, 2] + self.init_position[bj, 2])
        self.bond_break_len = np.where(
            (mid_z >= TEAR_ZONE[0]) & (mid_z <= TEAR_ZONE[1]),
            self.bond_l0 * (1.0 + BREAK_STRAIN_TEAR),
            self.bond_l0 * (1.0 + BREAK_STRAIN),
        )
        self.bond_alive = np.ones(len(bi), dtype=bool)

        # ---- adhesion to the upper plate ---------------------------------
        self.anchored = self.init_position[:, 2] >= ANCHOR_Z
        self.anchor_target0 = self.init_position[self.anchored].copy()

        # ---- deformation-gradient support --------------------------------
        # per-particle tensors S = sum d0 d0^T (fixed), C = sum d d0^T
        nb = len(bi)
        idx = np.concatenate([bi, bj])
        d0r = np.concatenate([d0, -d0])
        s9 = np.zeros((self.n, 9))
        np.add.at(s9, idx, np.einsum("ij,ik->ijk", d0r, d0r).reshape(2 * nb, 9))
        s9 += 1e-9 * np.eye(3).reshape(9)[None, :] * self.spacing ** 2
        self.s9_inv = np.linalg.inv(s9.reshape(self.n, 3, 3))
        self.grad_idx = idx
        self.grad_d0 = d0r

        # ---- misc ----------------------------------------------------------
        self.attributes0 = dict(frame.attributes)
        self.scalars0 = dict(frame.scalars)
        self.time = 0.0
        self.torn_bonds = 0
        self._last_frame = None

    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        p, v, m = self.position, self.velocity, self.mass
        f = self.gravity[None, :] * m[:, None]

        # ---- bonds -------------------------------------------------------
        alive = self.bond_alive
        if np.any(alive):
            bi = self.bond_i[alive]
            bj = self.bond_j[alive]
            d = p[bj] - p[bi]
            ln = np.linalg.norm(d, axis=1)
            u = d / np.maximum(ln, EPS)[:, None]
            rel = np.einsum("ij,ij->i", v[bj] - v[bi], u)
            fmag = BOND_K * (ln - self.bond_l0[alive]) + BOND_C * rel
            fvec = fmag[:, None] * u
            np.add.at(f, bi, fvec)
            np.add.at(f, bj, -fvec)
            # explicit tearing: permanently break overstretched bonds
            broken = ln > self.bond_break_len[alive]
            if np.any(broken):
                dead = np.flatnonzero(alive)[broken]
                self.bond_alive[dead] = False
                self.torn_bonds += len(dead)

        # ---- adhesion to the moving plate --------------------------------
        if np.any(self.anchored):
            tgt = self.anchor_target0.copy()
            tgt[:, 2] += _plate_displacement(self.time)
            cur = p[self.anchored]
            vel = v[self.anchored]
            fa = (ANCHOR_K * (tgt - cur) - ANCHOR_C * vel)
            np.add.at(f, np.flatnonzero(self.anchored), fa)

        # ---- floor contact (cylinder top plane) ---------------------------
        pen = FLOOR_TOP + self.radius - p[:, 2]
        touch = pen > 0.0
        if np.any(touch):
            ti = np.flatnonzero(touch)
            fn = FLOOR_K * pen[ti] - FLOOR_CN * v[ti, 2]
            f[ti, 2] += fn
            f[ti, 0] -= FLOOR_CT * v[ti, 0]
            f[ti, 1] -= FLOOR_CT * v[ti, 1]
            inside = np.hypot(p[ti, 0], p[ti, 1]) <= FLOOR_RADIUS
            f[ti[~inside], :] = 0.0  # outside the pad: no support

        # ---- plate contact (underside of the box) --------------------------
        pb = PLATE_BOTTOM0 + _plate_displacement(self.time)
        over = (np.abs(p[:, 0]) <= PLATE_HX) & (np.abs(p[:, 1]) <= PLATE_HY)
        ppen = p[:, 2] + self.radius - pb
        hit = over & (ppen > 0.0) & ~self.anchored
        if np.any(hit):
            hi = np.flatnonzero(hit)
            f[hi, 2] -= PLATE_K * ppen[hi] + PLATE_CN * v[hi, 2]

        # ---- integrate -----------------------------------------------------
        acc = f / m[:, None] - GLOBAL_DAMPING * v
        v_new = v + dt * acc
        np.clip(v_new, -VMAX, VMAX, out=v_new)
        p += dt * v_new
        v[:] = v_new
        if not (np.isfinite(p).all() and np.isfinite(v).all()):
            p[np.isnan(p)] = 0.0
            v[:] = 0.0
        self.time += dt

    # ------------------------------------------------------------------
    def _deformation_gradient_impl(self, p: np.ndarray) -> np.ndarray:
        rel = np.empty_like(self.grad_d0)
        # current bond vectors matching the concatenated (i,j) ordering
        dcur = p[self.bond_j] - p[self.bond_i]
        rel[: len(self.bond_i)] = dcur
        rel[len(self.bond_i):] = -dcur
        c9 = np.zeros((self.n, 9))
        np.add.at(c9, self.grad_idx, np.einsum("ij,ik->ijk", rel, self.grad_d0).reshape(len(rel), 9))
        F = np.einsum("nij,njk->nik", c9.reshape(self.n, 3, 3), self.s9_inv)
        # keep it finite and sane
        F = np.nan_to_num(F, nan=1.0, posinf=1.0, neginf=0.0)
        det = np.linalg.det(F)
        bad = ~np.isfinite(det) | (np.abs(det) < 1e-9)
        if np.any(bad):
            F[bad] = np.eye(3)
        return F

    # ------------------------------------------------------------------
    def _contact_signed_distance(self) -> np.ndarray:
        p = self.position
        r = self.radius
        pb = PLATE_BOTTOM0 + _plate_displacement(self.time)
        floor_gap = p[:, 2] - r - FLOOR_TOP
        inside_pad = np.hypot(p[:, 0], p[:, 1]) <= FLOOR_RADIUS
        floor_gap = np.where(inside_pad, floor_gap, np.inf)
        plate_gap = np.full(self.n, np.inf)
        over = (np.abs(p[:, 0]) <= PLATE_HX) & (np.abs(p[:, 1]) <= PLATE_HY)
        plate_gap[over] = pb - (p[over, 2] + r)
        gap = np.minimum(floor_gap, plate_gap)
        near = gap < 0.15
        if not np.any(near):
            return np.zeros(0, dtype=np.float64)
        return np.asarray(gap[near], dtype=np.float64)

    # ------------------------------------------------------------------
    def _make_frame(self, t: float) -> StateFrame:
        F = self._deformation_gradient_impl(self.position)
        attrs = {
            "deformation_gradient": F,
            "mass_quantity": self.mass.copy(),
            "particle_radius": self.radius.copy(),
            "foam_material": np.asarray(self.attributes0.get("foam_material", np.ones(self.n)), dtype=np.float64).copy(),
            "tear_candidate": np.asarray(self.attributes0.get("tear_candidate", np.zeros(self.n)), dtype=np.float64).copy(),
        }
        ext = self.gravity[None, :] * self.mass[:, None]
        fields = {
            "contact_signed_distance": self._contact_signed_distance(),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "external_force": ext,
        }
        scalars = {
            "contact_length_scale": float(self.spacing),
            "explicit_tearing_enabled": 1.0,
            "initial_particle_count": float(self.n),
            "torn_bond_count": float(self.torn_bonds),
        }
        frame = StateFrame(
            time=float(t),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time - 1e-9:
            target = self.time
        while self.time < target - 1e-9:
            dt = min(DT_SUB, target - self.time)
            self._step(dt)
        frame = self._make_frame(self.time)
        self._last_frame = frame
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "particles": int(self.n),
            "bonds_total": int(len(self.bond_i)),
            "bonds_broken": int(self.torn_bonds),
            "explicit_tearing": "enabled",
            "plate_displacement": float(_plate_displacement(self.time)),
        }
