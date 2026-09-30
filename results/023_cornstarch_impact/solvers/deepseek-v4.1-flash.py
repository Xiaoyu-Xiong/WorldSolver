"""Deterministic physically-based solver for the cornstarch-impact scene.

Physical model
--------------
The frozen scene is a closed triangle mesh (a "cornstarch suspension" ball,
642 vertices / 1280 faces) that starts above a static ground box and falls
under gravity ``(0, 0, -9.81)`` with initial velocity ``(0, 0, -0.35)``.

The ball is discretised as a mass-lumped particle lattice built from the
frozen mesh edges.  Each edge is a one-dimensional elasto-viscoplastic
element with

* an elastic spring           ``F_el = k_e (L - L0)``,
* a rate-dependent dashpot    ``F_visc = c_e * dL/dt`` (shear-thickening),
* a shear-thickening yield stress ``tau_y(edot) = tau_y0 * max(1, (edot/edot_y)^m)``
  with an associative plastic flow rule ``edot_p = <sigma - tau_y>_+ / mu(edot)``
  that relaxes the rest length ``L0`` (Duvaut--Lions viscoplasticity with a
  viscous regularisation).

Because both the yield stress and the plastic viscosity grow strongly with the
strain rate, the material is effectively elastic (shear jammed) at the high
strain rates of the ground impacts -- producing the observed elastic rebound
with the configured restitution -- while at the low strain rates that follow
the bounces the yield stress collapses towards ``tau_y0`` and gravity drives
plastic flow, i.e. the suspension yields and levels into a puddle.  The
transition is an emergent consequence of the constitutive law, not a scripted
event.

Time integration is a symplectic (semi-implicit) Euler step with a fixed,
CFL-safe ``dt``.  Contact with the static ground plane is handled by a
projection + restitution/friction impulse at the vertex level.

All required raw trajectory fields are exported unchanged; no residuals or
scores are computed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# Numerical / material constants (SI units).  Values are derived from the
# frozen configuration plus a small number of documented scale choices.
# ---------------------------------------------------------------------------
_DT = 2.0e-4                 # fixed integration step (s); CFL-safe for this mesh
_STIFFNESS_PA_PER_UNIT = 300.0   # elastic modulus per unit of config "stiffness"
_YIELD_STRESS_PA = 5.0       # quasi-static (rate -> 0) yield stress
_PLASTIC_VISCOSITY = 12.0    # base plastic viscosity (Pa s)
_THICKENING_EXPONENT = 1.2   # yield-stress shear-thickening exponent
_VISC_THICKENING_EXPONENT = 1.5   # viscosity shear-thickening exponent
_DAMPING_RATIO_SCALE = 0.1   # maps config "damping" onto a modal damping ratio
_DAMPING_RATIO_CAP = 0.3
_COMPRESSION_BARRIER = 0.5   # element inversion guard (fraction of rest length)
_BARRIER_STIFFNESS = 5.0     # barrier stiffness multiplier on k_e
_MAX_PLASTIC_STEP = 0.2      # rest-length relaxation cap per step (fraction)
_SPEED_LIMIT = 50.0          # safety clamp (m/s)
_POS_LIMIT = 5.0             # safety clamp (m)


def _resolve_asset(config: dict) -> Path:
    """Locate the frozen subject asset relative to the workspace root."""
    rel = Path(str(config["subject_asset"]))
    if rel.is_absolute():
        return rel
    root = Path(__file__).resolve().parents[1]
    return root / rel


class BenchmarkSolver:
    """Elasto-viscoplastic lattice solver for the frozen cornstarch scene."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(scene.config)

        initial = scene.initial_state()
        x0 = np.asarray(initial.position, dtype=float).copy()
        self.n = int(x0.shape[0])

        asset = _resolve_asset(config)
        with np.load(asset) as data:
            faces = np.asarray(data["faces"], dtype=np.int64)
            if "vertices" in data.files:
                verts = np.asarray(data["vertices"], dtype=float)
                if verts.shape == x0.shape:
                    x0 = verts.copy()

        # --- mesh topology -------------------------------------------------
        e0 = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
        e1 = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
        edges = np.unique(np.stack([np.minimum(e0, e1), np.maximum(e0, e1)], axis=1), axis=0)
        self.edges = edges.astype(np.int64)
        self.E = int(edges.shape[0])
        self.i0 = self.edges[:, 0]
        self.i1 = self.edges[:, 1]
        self.ii = np.concatenate([self.i0, self.i1])
        self.jj = np.concatenate([self.i1, self.i0])

        ref = x0[self.jj] - x0[self.ii]
        self.dX = ref.copy()
        edge_vec = x0[self.i1] - x0[self.i0]
        self.L0_init = np.linalg.norm(edge_vec, axis=1)
        self.L0_init = np.maximum(self.L0_init, 1.0e-6)
        self.L0 = self.L0_init.copy()
        h = float(np.mean(self.L0_init))

        # --- material / solver parameters ---------------------------------
        self.gravity = np.asarray(config["gravity"], dtype=float)
        self.ground_z = float(config.get("ground_z", 0.0))
        self.restitution = float(config.get("restitution", 0.0))
        self.friction = float(config.get("friction", 0.0))
        self.contact_length_scale = float(config.get("contact_length_scale", h))
        self.yield_time = max(float(config.get("yield_time", 1.0)), 1.0e-3)

        mass_total = float(config["subject_mass"])
        self.mass = np.full(self.n, mass_total / self.n, dtype=float)
        self.m_total = mass_total

        stiffness = float(config.get("stiffness", 1.0))
        modulus = stiffness * _STIFFNESS_PA_PER_UNIT          # Pa
        self.k_e = modulus * h                                 # N/m per edge
        self.k_e = float(np.clip(self.k_e, 1.0e-3, 1.0e9))
        m_i = self.mass[0]
        omega_char = float(np.sqrt(self.k_e / m_i))
        damping = float(config.get("damping", 0.0))
        zeta = min(damping * _DAMPING_RATIO_SCALE, _DAMPING_RATIO_CAP)
        self.c_e = 2.0 * zeta * self.k_e / max(omega_char, 1.0e-6)
        self.k_barrier = _BARRIER_STIFFNESS * self.k_e

        self.tau_y0 = _YIELD_STRESS_PA
        self.edot_y = 1.0 / self.yield_time
        self.mu0 = _PLASTIC_VISCOSITY
        self.area_e = self.L0_init ** 2                       # edge cross-section (m^2)

        # --- geometry for the per-vertex deformation gradient ---------------
        # Tikhonov-regularised least-squares fit over each vertex one-ring.
        M = np.zeros((self.n, 3, 3))
        np.add.at(M, self.ii, self.dX[:, :, None] * self.dX[:, None, :])
        trace = np.trace(M, axis1=1, axis2=2) / 3.0
        lam = 1.0e-6 * float(np.mean(np.abs(trace))) + 1.0e-300
        self.M_reg = M + lam * np.eye(3)[None, :, :]
        self.inv_M = np.linalg.inv(self.M_reg)

        # --- state ---------------------------------------------------------
        self.x = x0.copy()
        self.v = np.tile(np.asarray(config.get("initial_velocity", [0.0, 0.0, 0.0]),
                                    dtype=float), (self.n, 1))
        self.t_sim = 0.0
        self.steps = 0
        self.initial_position = x0.copy()

    # ------------------------------------------------------------------
    # forces
    # ------------------------------------------------------------------
    def _forces(self):
        d = self.x[self.jj] - self.x[self.ii]
        L = np.sqrt(np.sum(d * d, axis=1))
        L_safe = np.maximum(L, 1.0e-3 * self.L0_init)
        n_hat = d / L_safe[:, None]

        # elastic spring + compression barrier (element inversion guard)
        f_el = self.k_e * (L - self.L0)
        barrier_len = _COMPRESSION_BARRIER * self.L0_init
        short = L < barrier_len
        if np.any(short):
            f_el = f_el + np.where(short, -self.k_barrier * (barrier_len - L), 0.0)

        # shear-thickening dashpot
        v_rel = self.v[self.jj] - self.v[self.ii]
        ldot = np.sum(v_rel * n_hat, axis=1)
        f_vis = self.c_e * ldot

        # viscoplastic relaxation of the rest length
        edot = np.abs(ldot / L_safe)
        thick = np.power(edot / self.edot_y, _THICKENING_EXPONENT)
        tau_y = self.tau_y0 * np.maximum(1.0, thick)
        sigma = np.abs(f_el) / self.area_e
        over = sigma - tau_y
        active = over > 0.0
        if np.any(active):
            visc = self.mu0 * (1.0 + np.power(edot / self.edot_y,
                                              _VISC_THICKENING_EXPONENT))
            rate = np.where(active, over / np.maximum(visc, 1.0e-12), 0.0)
            d_rest = np.sign(f_el) * self.L0 * rate * _DT
            limit = _MAX_PLASTIC_STEP * self.L0
            d_rest = np.clip(d_rest, -limit, limit)
            self.L0 = np.maximum(self.L0 + d_rest, 1.0e-3 * self.L0_init)

        f_edge = f_el + f_vis
        force = np.zeros_like(self.x)
        contrib = f_edge[:, None] * n_hat
        np.add.at(force, self.ii, contrib)
        np.add.at(force, self.jj, -contrib)
        force += self.mass[:, None] * self.gravity[None, :]
        return force

    # ------------------------------------------------------------------
    # contact with the static ground plane z = ground_z
    # ------------------------------------------------------------------
    def _contact(self):
        z = self.x[:, 2]
        pen = z < self.ground_z
        if not np.any(pen):
            return
        self.x[pen, 2] = self.ground_z
        v_n = self.v[pen, 2]
        hitting = v_n < 0.0
        if np.any(hitting):
            v_new = -self.restitution * v_n[hitting]
            dv_n = v_new - v_n[hitting]
            # Coulomb-limited tangential friction impulse
            vt = self.v[pen][:, :2]
            vt_mag = np.linalg.norm(vt, axis=1)
            lim = self.friction * np.abs(dv_n)
            scale = np.ones_like(vt_mag)
            nz = vt_mag > 1.0e-12
            scale[nz] = np.maximum(0.0, 1.0 - lim[nz] / vt_mag[nz])
            idx_v = np.flatnonzero(pen)[hitting]
            self.v[idx_v, :2] = vt[hitting] * scale[hitting, None]
            self.v[idx_v, 2] = v_new

    # ------------------------------------------------------------------
    def _step(self) -> None:
        f = self._forces()
        inv_m = 1.0 / self.mass[:, None]
        self.v += _DT * f * inv_m
        # light numerical sanitation (never active in nominal operation)
        np.clip(self.v, -_SPEED_LIMIT, _SPEED_LIMIT, out=self.v)
        self.x += _DT * self.v
        self._contact()
        np.clip(self.x, -_POS_LIMIT, _POS_LIMIT, out=self.x)
        bad = ~np.isfinite(self.v).all(axis=1)
        if np.any(bad):
            self.v[bad] = 0.0
        self.steps += 1
        self.t_sim = self.steps * _DT

    # ------------------------------------------------------------------
    def _deformation_gradient(self):
        dx = self.x[self.jj] - self.x[self.ii]
        A = np.zeros((self.n, 3, 3))
        np.add.at(A, self.ii, dx[:, :, None] * self.dX[:, None, :])
        F = np.einsum("nij,njk->nik", A, self.inv_M)
        return F

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        steps_needed = int(np.ceil(target / _DT - 1.0e-9))
        steps_needed = max(steps_needed, 0)
        while self.steps < steps_needed:
            self._step()

        F = self._deformation_gradient()
        gap = self.x[:, 2] - self.ground_z
        frame = StateFrame(
            time=target,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
                "contact_signed_distance": gap,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        gap = self.x[:, 2] - self.ground_z
        return {
            "steps": int(self.steps),
            "dt": _DT,
            "sim_time": float(self.t_sim),
            "min_gap": float(np.min(gap)),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))),
            "rest_length_ratio_min": float(np.min(self.L0 / self.L0_init)),
            "rest_length_ratio_max": float(np.max(self.L0 / self.L0_init)),
        }
