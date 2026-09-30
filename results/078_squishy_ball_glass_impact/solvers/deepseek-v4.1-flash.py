"""Physically-based solver for the tendril-covered squishy-ball glass impact.

Model
-----
The toy is a three-dimensional tetrahedral solid (6082 nodes, 16631 tets) taken
from the frozen topology asset.  It is advanced with an explicit velocity-Verlet
integration of a compressible Neo-Hookean hyperelastic material (mass lumping),
with:

* a momentum-preserving deviatoric (body-frame) viscous damping that dissipates
  internal deformation energy without damping free translational flight, and
* a one-sided penalty contact against the front face of the fixed glass wall
  (the wall occupies x in [0, 0.024] because of the frozen geometry), so the
  leading tendrils touch first and then the core flattens against the glass.

All quantities are deterministic and finite.  Every required raw trajectory
field is exported on every sampled ``StateFrame``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


def _cofactor(F: np.ndarray) -> np.ndarray:
    """Cofactor matrix of a stack of 3x3 row-major matrices.

    ``cofactor(F) / det(F) == inverse(F).T``.
    """
    a = F[..., 0, 0]; b = F[..., 0, 1]; c = F[..., 0, 2]
    d = F[..., 1, 0]; e = F[..., 1, 1]; f = F[..., 1, 2]
    g = F[..., 2, 0]; h = F[..., 2, 1]; i = F[..., 2, 2]
    C = np.empty_like(F)
    C[..., 0, 0] = e * i - f * h
    C[..., 0, 1] = -(d * i - f * g)
    C[..., 0, 2] = d * h - e * g
    C[..., 1, 0] = -(b * i - c * h)
    C[..., 1, 1] = a * i - c * g
    C[..., 1, 2] = -(a * h - b * g)
    C[..., 2, 0] = b * f - c * e
    C[..., 2, 1] = -(a * f - c * d)
    C[..., 2, 2] = a * e - b * d
    return C


class BenchmarkSolver:
    """Explicit nonlinear-FEM squishy ball impacting a fixed glass wall."""

    # Frozen glass front face: translation.x - half_extents.x = 0.012 - 0.012.
    WALL_X = 0.0

    # Material / numerical parameters.
    YOUNG = 900.0          # Pa, soft squishy rubber
    POISSON = 0.40
    CFL = 0.20             # explicit safety factor on the dilatational wave speed
    GAMMA = 30.0           # 1/s, deviatoric (body-frame) viscous damping rate
    BETA = 0.0015          # s, small stiffness-proportional damping
    J_MIN = 0.25           # volumetric floor guarding the log barrier
    P_MAX = 2.0e4          # Pa, constitutive stress clamp (numerical safety)
    F_MAX = 200.0          # N, nodal force clamp (numerical safety)
    V_MAX = 40.0           # m/s, velocity clamp (numerical safety)
    K_CONTACT = 3.0e3      # N/m, per-node one-sided penalty stiffness
    C_CONTACT = 0.5        # N*s/m, per-node contact damping

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)

        frame = scene.initial_state()
        self.x = np.array(frame.position, dtype=np.float64)
        self.v = np.array(frame.velocity, dtype=np.float64)
        self.m = np.array(frame.mass, dtype=np.float64)
        self.body_id = np.array(frame.body_id, dtype=np.int64)
        self.n = len(self.m)
        self.component = np.zeros(self.n, dtype=np.int64)
        self.inv_m = 1.0 / self.m[:, None]

        self.mass_total = float(self.m.sum())
        self.speed0 = float(np.linalg.norm(self.v[0]))
        # Positive momentum scale frozen by the platform contract.
        self.reference_momentum = float(self.mass_total * self.speed0)

        self._load_topology()
        self._configure_material()
        self._configure_time_step()

        self._time = 0.0
        self._impulse = np.zeros(3, dtype=np.float64)
        self._contact_now = np.zeros(3, dtype=np.float64)
        self.domain_weights = np.ones((1, self.n), dtype=np.float64)

        # Report the initial equilibrium deformation gradient immediately.
        self._F_out = self._element_deformation_gradient(self.x)

        self._ready = True

    def _asset_path(self) -> Path:
        asset = getattr(self.scene, "_asset", None)
        if asset is not None and Path(asset).is_file():
            return Path(asset)
        workspace = Path(__file__).resolve().parent.parent
        return workspace / "assets" / "squishy_ball_volume.npz"

    def _load_topology(self) -> None:
        with np.load(self._asset_path()) as data:
            tets = np.asarray(data["tets"], dtype=np.int64)
            rest = np.asarray(data["rest_position"], dtype=np.float64)
        self.Ne = int(tets.shape[0])
        self.tets_all = tets
        self.rest = rest

        p = rest[tets]
        det = np.einsum(
            "ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]
        )
        vol = np.abs(det) / 6.0
        # Drop degenerate slivers whose rest shape cannot be inverted safely.
        valid = vol > 1.0e-9
        if not np.any(valid):
            valid = np.ones_like(vol, dtype=bool)
        self.valid_mask = valid
        self.tets = tets[valid]
        self.vol = vol[valid]
        self.M = int(self.tets.shape[0])
        self.t0 = self.tets[:, 0]
        self.t1 = self.tets[:, 1]
        self.t2 = self.tets[:, 2]
        self.t3 = self.tets[:, 3]

        pv = rest[self.tets]
        Dm = np.empty((self.M, 3, 3), dtype=np.float64)
        Dm[:, :, 0] = pv[:, 1] - pv[:, 0]
        Dm[:, :, 1] = pv[:, 2] - pv[:, 0]
        Dm[:, :, 2] = pv[:, 3] - pv[:, 0]
        self.Dm_inv = np.linalg.inv(Dm)

        # Smallest valid element edge drives the explicit stability limit.
        edges = []
        for a in range(3):
            for b in range(a + 1, 4):
                edges.append(np.linalg.norm(pv[:, a] - pv[:, b], axis=1))
        self.h_min = float(min(e.min() for e in edges))

    def _configure_material(self) -> None:
        E, nu = self.YOUNG, self.POISSON
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.density = self.mass_total / float(self.vol.sum())
        self.c_p = float(np.sqrt((self.lam + 2.0 * self.mu) / self.density))
        self.muv = self.BETA * self.mu
        self.lamv = self.BETA * self.lam

    def _configure_time_step(self) -> None:
        dt_wave = self.CFL * self.h_min / self.c_p
        m_min = float(self.m.min())
        dt_contact = 2.0 * float(np.sqrt(m_min / self.K_CONTACT)) * 0.5
        self.dt = float(max(min(dt_wave, dt_contact), 1.0e-6))

    # ------------------------------------------------------------------ forces
    def _scatter(self, fe: np.ndarray) -> np.ndarray:
        f = np.zeros((self.n, 3), dtype=np.float64)
        for a in range(4):
            idx = self.tets[:, a]
            f[:, 0] += np.bincount(idx, weights=fe[:, a, 0], minlength=self.n)
            f[:, 1] += np.bincount(idx, weights=fe[:, a, 1], minlength=self.n)
            f[:, 2] += np.bincount(idx, weights=fe[:, a, 2], minlength=self.n)
        return f

    def _internal_forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        x0 = x[self.t0]; x1 = x[self.t1]; x2 = x[self.t2]; x3 = x[self.t3]
        Ds = np.empty((self.M, 3, 3), dtype=np.float64)
        Ds[:, :, 0] = x1 - x0
        Ds[:, :, 1] = x2 - x0
        Ds[:, :, 2] = x3 - x0
        F = Ds @ self.Dm_inv

        C = _cofactor(F)
        det = F[:, 0, 0] * C[:, 0, 0] + F[:, 0, 1] * C[:, 0, 1] + F[:, 0, 2] * C[:, 0, 2]
        det_c = np.maximum(det, self.J_MIN)
        FinvT = C / det_c[:, None, None]
        P = self.mu * (F - FinvT) + self.lam * np.log(det_c)[:, None, None] * FinvT
        np.clip(P, -self.P_MAX, self.P_MAX, out=P)
        G = -self.vol[:, None, None] * (P @ self.Dm_inv)

        fe = np.empty((self.M, 4, 3), dtype=np.float64)
        fe[:, 0] = -G.sum(axis=2)
        fe[:, 1] = G[:, :, 0]
        fe[:, 2] = G[:, :, 1]
        fe[:, 3] = G[:, :, 2]

        # Small stiffness-proportional (Kelvin-Voigt, small-strain) damping.
        v0 = v[self.t0]; v1 = v[self.t1]; v2 = v[self.t2]; v3 = v[self.t3]
        Dv = np.empty((self.M, 3, 3), dtype=np.float64)
        Dv[:, :, 0] = v1 - v0
        Dv[:, :, 1] = v2 - v0
        Dv[:, :, 2] = v3 - v0
        Fv = Dv @ self.Dm_inv
        eps = 0.5 * (Fv + Fv.swapaxes(1, 2))
        tr = eps[:, 0, 0] + eps[:, 1, 1] + eps[:, 2, 2]
        Pv = 2.0 * self.muv * eps + self.lamv * tr[:, None, None] * np.eye(3)
        Gv = -self.vol[:, None, None] * (Pv @ self.Dm_inv)
        fe[:, 0] += -Gv.sum(axis=2)
        fe[:, 1] += Gv[:, :, 0]
        fe[:, 2] += Gv[:, :, 1]
        fe[:, 3] += Gv[:, :, 2]

        f = self._scatter(fe)

        # Momentum-preserving deviatoric damping: no force for rigid translation.
        v_cm = (self.m[:, None] * v).sum(axis=0) / self.mass_total
        f -= self.GAMMA * self.m[:, None] * (v - v_cm)
        np.nan_to_num(f, copy=False, nan=0.0, posinf=self.F_MAX, neginf=-self.F_MAX)
        np.clip(f, -self.F_MAX, self.F_MAX, out=f)
        return f

    def _contact_force(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        pen = x[:, 0] - self.WALL_X
        fcon = np.zeros((self.n, 3), dtype=np.float64)
        active = pen > 0.0
        if np.any(active):
            vn = v[active, 0]
            fn = self.K_CONTACT * pen[active] + self.C_CONTACT * np.maximum(vn, 0.0)
            fn = np.maximum(fn, 0.0)
            fcon[active, 0] = -fn
        return fcon

    def _total_force(self, x: np.ndarray, v: np.ndarray):
        fcon = self._contact_force(x, v)
        f = self._internal_forces(x, v) + fcon
        return f, fcon.sum(axis=0)

    # ------------------------------------------------------------------- step
    def _step(self, h: float) -> None:
        x = self.x
        v = self.v
        f0, c0 = self._total_force(x, v)
        v_half = v + (0.5 * h) * (f0 * self.inv_m)
        x_new = x + h * v_half
        f1, c1 = self._total_force(x_new, v_half)
        v_new = v_half + (0.5 * h) * (f1 * self.inv_m)

        np.clip(v_new, -self.V_MAX, self.V_MAX, out=v_new)
        np.nan_to_num(x_new, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        np.nan_to_num(v_new, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        self._impulse += 0.5 * (c0 + c1) * h
        self._contact_now = c1
        self.x = x_new
        self.v = v_new
        self._time += h

    # ------------------------------------------------------------------ output
    def _element_deformation_gradient(self, x: np.ndarray) -> np.ndarray:
        out = np.tile(np.eye(3, dtype=np.float64), (self.Ne, 1, 1))
        x0 = x[self.t0]; x1 = x[self.t1]; x2 = x[self.t2]; x3 = x[self.t3]
        Ds = np.empty((self.M, 3, 3), dtype=np.float64)
        Ds[:, :, 0] = x1 - x0
        Ds[:, :, 1] = x2 - x0
        Ds[:, :, 2] = x3 - x0
        out[self.valid_mask] = Ds @ self.Dm_inv
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def _make_frame(self, time: float) -> StateFrame:
        return StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "component": self.component.copy(),
                "mass_quantity": self.m[:, None].copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self._impulse[None, :].copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
                "budget_domain_weights": self.domain_weights.copy(),
                "budget_external_force": self._contact_now[None, None, :].copy(),
                "cumulative_mass_source": np.zeros((1,), dtype=np.float64),
                "element_deformation_gradient": self._F_out.copy(),
            },
            scalars={"reference_momentum": self.reference_momentum},
        )

    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("BenchmarkSolver.initialize() must be called before advance_to()")
        target = float(time)
        if target <= self._time + 1.0e-12:
            return self._make_frame(self._time)
        while self._time < target - 1.0e-12:
            h = min(self.dt, target - self._time)
            if h <= 1.0e-15:
                break
            self._step(h)
        self._time = target
        self._F_out = self._element_deformation_gradient(self.x)
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "nodes": int(self.n),
            "elements": int(self.Ne),
            "active_elements": int(self.M),
            "time_step": float(self.dt),
            "wave_speed": float(self.c_p),
            "density": float(self.density),
            "min_edge": float(self.h_min),
            "mu": float(self.mu),
            "lambda": float(self.lam),
            "total_mass": float(self.mass_total),
            "reference_momentum": float(self.reference_momentum),
            "time": float(self._time),
            "net_impulse": self._impulse.tolist(),
        }
