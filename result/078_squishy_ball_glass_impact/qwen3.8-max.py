"""Physically-based solver: tendril-covered squishy ball impacting a glass wall.

Method
------
Explicit (symplectic-Euler) co-rotational-free nonlinear FEM on the tetrahedral
volume mesh:

* Material: compressible "stable" Neo-Hookean energy per tetrahedron,
      W = mu/2 (I1 - 3) - mu L(J) + lam/2 L(J)^2,
  where L is log(J) smoothly extended linearly below a floor J0 so the energy
  and stress remain finite and strongly restorative as J -> 0 (inversion
  barrier).  P = mu F + (lam L - mu) L' cof(F),  cof(F) = J F^-T.
* Kelvin-Voigt viscoelasticity: P += eta * dF/dt (rate damping).
* Contact with the fixed glass wall (plane x = 0, wall occupies x >= 0):
  penalty normal spring + normal approach damping + regularized Coulomb
  friction acting on penetrating nodes.  No other external forces (no
  gravity): the momentum budget of the body is changed only by wall traction.

The ball starts at the scene-prescribed state and flies at 1.75 m/s toward the
wall; front tendrils touch first (~0.19 s), bend and spread, the ball flattens
(maximum compression ~0.5 s), then elasticity pushes it round again and the
tendrils peel off the glass during rebound (visible well before 0.92 s).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_ASSET = Path(__file__).resolve().parent.parent / "assets" / "squishy_ball_volume.npz"


class BenchmarkSolver:
    # ---- material / numerics -------------------------------------------------
    YOUNG = 90.0          # Pa  (very soft rubber toy; total mass 1 kg, rho ~ 11.7)
    POISSON = 0.35
    DT = 5.0e-4           # s   (CFL safe: c ~ 3.5 m/s, h_min ~ 0.011 m)
    ETA = 8.0e-3          # Pa*s Kelvin-Voigt rate damping
    J_FLOOR = 0.25        # inversion barrier floor for log J
    K_CONTACT = 1500.0    # N/m penalty per node
    C_CONTACT = 0.30      # N*s/m normal approach damping
    FRICTION = 0.25
    V_REG = 1.0e-3        # m/s friction regularization

    def initialize(self, scene, seed: int) -> None:
        frame0 = scene.initial_state()
        x0 = np.asarray(frame0.position, dtype=np.float64).copy()
        v0 = np.asarray(frame0.velocity, dtype=np.float64).copy()
        m = np.asarray(frame0.mass, dtype=np.float64).copy()
        body_id = np.asarray(frame0.body_id).copy()

        with np.load(_ASSET) as data:
            tets = np.asarray(data["tets"], dtype=np.int64)

        self._seed = int(seed)
        self._x = x0
        self._v = v0
        self._m = m
        self._inv_m = 1.0 / m
        self._body_id = body_id
        self._component = np.asarray(frame0.attributes.get("component", np.zeros(len(x0), dtype=np.int64))).copy()
        self._tets = tets
        self._ne = len(tets)
        self._n = len(x0)
        self._t = 0.0

        # elasticity
        E, nu = self.YOUNG, self.POISSON
        self._mu = E / (2.0 * (1.0 + nu))
        self._lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # rest element geometry
        p = x0[tets]                                   # [Ne,4,3]
        dm = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=-1)  # columns = edges
        det = np.linalg.det(dm)
        self._vol = np.abs(det) / 6.0
        self._a = np.linalg.inv(dm)                    # Dm^-1
        self._at = np.transpose(self._a, (0, 2, 1))    # (Dm^-1)^T

        # flat scatter indices for np.add.at
        base = np.arange(3)[None, :]
        self._scatter = [(tets[:, k][:, None] * 3 + base).ravel() for k in range(4)]

        self._f_prev = np.tile(np.eye(3), (self._ne, 1, 1))   # rest deformation gradient
        self._force = np.zeros((self._n, 3))

        # momentum budget bookkeeping (single global domain G=1, one material K=1)
        total_mass = float(m.sum())
        v_init = np.asarray(scene.config["initial_conditions"]["squishy_ball"]["linear_velocity"], dtype=np.float64)
        self._reference_momentum = float(total_mass * np.linalg.norm(v_init))
        self._cum_impulse = np.zeros((1, 3))
        self._last_contact_force = np.zeros(3)
        self._steps = 0
        self._max_pen = 0.0

    # ---- core ----------------------------------------------------------------
    def _deformation_gradients(self):
        p = self._x[self._tets]
        ds = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=-1)
        return np.einsum("nij,njk->nik", ds, self._a)

    def _step(self, dt: float) -> None:
        x, v, m = self._x, self._v, self._m
        f = np.zeros_like(x)

        # --- elastic + viscous tet forces ---
        F = self._deformation_gradients()
        dFdt = (F - self._f_prev) / dt

        # cofactor matrix C = J F^-T ; C[i] = cross(F[i+1], F[i+2])
        cof = np.empty_like(F)
        cof[:, 0] = np.cross(F[:, 1], F[:, 2])
        cof[:, 1] = np.cross(F[:, 2], F[:, 0])
        cof[:, 2] = np.cross(F[:, 0], F[:, 1])
        J = np.einsum("nij,nij->n", F, cof) / 3.0

        j0 = self.J_FLOOR
        jc = np.maximum(J, j0)
        log_j = np.log(jc)
        L = np.where(J > j0, log_j, np.log(j0) + (J - j0) / j0)
        Lp = np.where(J > j0, 1.0 / jc, 1.0 / j0)

        mu, lam, eta = self._mu, self._lam, self.ETA
        coeff = (lam * L - mu) * Lp
        P = mu * (F + eta / mu * dFdt) + coeff[:, None, None] * cof

        G = self._vol[:, None, None] * np.einsum("nij,nkj->nik", P, self._at)  # dE/dDs
        flat = f.reshape(-1)
        g0, g1, g2 = G[:, :, 0], G[:, :, 1], G[:, :, 2]
        np.add.at(flat, self._scatter[1], (-g0).ravel())
        np.add.at(flat, self._scatter[2], (-g1).ravel())
        np.add.at(flat, self._scatter[3], (-g2).ravel())
        np.add.at(flat, self._scatter[0], (g0 + g1 + g2).ravel())

        # --- wall contact (plane x = 0; glass occupies x >= 0) ---
        pen = -x[:, 0]
        hit = pen > 0.0
        contact_total = np.zeros(3)
        if np.any(hit):
            idx = np.nonzero(hit)[0]
            fn = self.K_CONTACT * pen[idx]
            approaching = np.minimum(v[idx, 0], 0.0)
            fn = fn - self.C_CONTACT * approaching          # >= 0 always
            fc = np.zeros((len(idx), 3))
            fc[:, 0] = fn
            vt = v[idx, 1:]
            speed = np.linalg.norm(vt, axis=1)
            ft_mag = np.minimum(self.FRICTION * fn, self.FRICTION * fn * speed / (speed + self.V_REG) + 1e-300)
            scale = np.where(speed > 1e-12, ft_mag / np.maximum(speed, 1e-12), 0.0)
            fc[:, 1:] = -vt * scale[:, None]
            np.add.at(f, idx, fc)
            contact_total = fc.sum(axis=0)
            self._max_pen = max(self._max_pen, float(pen[idx].max()))
        self._last_contact_force = contact_total
        self._cum_impulse += dt * contact_total.reshape(1, 3)

        # --- integrate (symplectic Euler) ---
        v += dt * f * self._inv_m[:, None]
        x += dt * v

        self._f_prev = F
        self._t += dt
        self._steps += 1

    # ---- driver ---------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self._t < target - 1e-12:
            dt = min(self.DT, target - self._t)
            self._step(dt)

        F = self._deformation_gradients()
        n = self._n
        mass = self._m
        ones = np.ones((1, n))
        ext_force = np.zeros((1, 1, 3))
        ext_force[0, 0] = self._last_contact_force
        zero3 = np.zeros((1, 3))
        frame = StateFrame(
            time=self._t,
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=mass.copy(),
            body_id=self._body_id.copy(),
            attributes={
                "component": self._component.copy(),
                "mass_quantity": mass.copy(),
            },
            fields={
                "budget_domain_weights": ones,
                "budget_external_force": ext_force,
                "budget_cumulative_external_impulse": self._cum_impulse.copy(),
                "budget_cumulative_inflow_momentum": zero3.copy(),
                "budget_cumulative_outflow_momentum": zero3.copy(),
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": F.astype(np.float64),
            },
            scalars={"reference_momentum": self._reference_momentum},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        ke = float(0.5 * np.sum(self._m * np.sum(self._v ** 2, axis=1)))
        px = float(np.sum(self._m * self._v[:, 0]))
        return {
            "time": self._t,
            "steps": self._steps,
            "kinetic_energy": ke,
            "momentum_x": px,
            "cumulative_contact_impulse_x": float(self._cum_impulse[0, 0]),
            "max_penetration": self._max_pen,
            "min_x": float(self._x[:, 0].min()),
        }
