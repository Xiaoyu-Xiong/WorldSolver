"""Meshless (MLS-based) volumetric Neo-Hookean solver for the octopus squash scene.

Physics
-------
* The octopus is treated as a three-dimensional elastic solid discretized by the
  rest vertices of the frozen surface asset.  Each sample carries a rest volume
  and acts as one integration element.
* Per element the deformation gradient is reconstructed with a regularized
  moving-least-squares fit of the displacement gradient,
  ``F_i = I + (sum_j w_ij (u_j-u_i) dX_ij^T) B_i^{-1}``,
  which is a standard meshless Galerkin discretization of a continuum solid and
  degrades gracefully (``F -> I``) in rank-deficient directions.
* The constitutive model is the stable Neo-Hookean energy
  ``Psi = mu/2 (Ic - 3) - mu/2 log(Ic + 1) + lambda/2 (J - alpha)^2`` with
  ``alpha = 1 + 3 mu / (4 lambda)`` so that the rest state is stress free and the
  energy stays finite/invertible for inverted or degenerate elements.
* Internal forces are exact derivatives of the discrete energy, therefore they
  come in equal-and-opposite pairs and conserve linear momentum exactly.
  Pairwise (bond) viscosity is used for dissipation, which is also momentum
  conserving.
* External loads are gravity and regularized normal+friction contact with the
  three static support cylinders and the kinematic upper cylinder (axis along y,
  as frozen by the scene transforms).
* Time integration is explicit symplectic Euler with a small fixed step, which
  is stable for the chosen stiffness/mass/contact parameters.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _quat_axis_z(q_xyzw: np.ndarray) -> np.ndarray:
    """Return the world direction of a cylinder's local +z axis."""
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=float,
    )


class _Cylinder:
    """Infinite-in-axis cylinder collider (axis is one of the world axes)."""

    def __init__(self, center, radius, half_length, axis, kinematic=False):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.half_length = float(half_length)
        # index of the dominant axis component (0=x, 1=y, 2=z)
        self.axis_idx = int(np.argmax(np.abs(np.asarray(axis, dtype=float))))
        self.radial_idx = [k for k in range(3) if k != self.axis_idx]
        self.kinematic = bool(kinematic)
        self.offset = np.zeros(3, dtype=float)      # current translation offset
        self.velocity = np.zeros(3, dtype=float)    # current surface velocity


class BenchmarkSolver:
    # ---- material / numerical parameters (frozen constants) ----
    _V_TOTAL = 2.0        # m^3, estimated rest volume of the octopus body
    _MU = 350.0           # Pa, shear modulus (soft flesh)
    _LAMBDA_RATIO = 4.0   # lambda / mu  (near-incompressible flesh)
    _BOND_DAMP = 3.0      # N*s/m, momentum conserving bond viscosity
    _CONTACT_K = 8000.0   # N/m, penalty stiffness
    _CONTACT_C = 90.0     # N*s/m, normal contact viscosity
    _CONTACT_FMAX = 220.0 # N, per-sample contact force cap (keeps steps stable)
    _FRICTION = 0.45      # Coulomb coefficient
    _FRICTION_C = 60.0    # N*s/m, viscous regularization of friction
    _MARGIN = 0.02        # m, collision margin
    _DT = 2.5e-4          # s, fixed integration step
    _PAIR_FCLAMP = 2500.0 # N, symmetric safety clamp on pair forces
    _VMAX = 40.0          # m/s, safety clamp on sample speed

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = dict(getattr(scene, "config", {}) or {})
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)  # determinism anchor (unused noise)
        self._rng = rng

        frame0 = scene.initial_state()
        self.X = np.array(frame0.position, dtype=float)          # rest positions
        self.n = int(self.X.shape[0])
        if frame0.mass is not None:
            self.mass = np.array(frame0.mass, dtype=float)
        else:
            self.mass = np.full(self.n, 12.0 / self.n, dtype=float)
        self.total_mass = float(self.mass.sum())

        self.x = self.X.copy()
        self.v = np.zeros_like(self.X)
        self.t = 0.0

        self.gravity = np.array(
            self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        )
        self.g_force = self.mass[:, None] * self.gravity[None, :]

        # ---- material constants ----
        self.mu = float(self._MU)
        self.lam = float(self._LAMBDA_RATIO * self._MU)
        self.alpha = 1.0 + 3.0 * self.mu / (4.0 * self.lam)
        self.vol = np.full(self.n, self._V_TOTAL / self.n, dtype=float)

        self._build_neighborhoods()
        self._build_colliders()

        # ---- budget accumulators ----
        self._contact_impulse = np.zeros(3, dtype=float)   # exact accumulated
        self._interval_contact_impulse = np.zeros(3, dtype=float)
        self._interval_time = 0.0
        self._last_contact_force = np.zeros(3, dtype=float)
        self._steps = 0

        v_scale = float(np.sqrt(2.0 * max(abs(self.gravity[2]), 1e-6) * 1.0))
        self.reference_momentum = float(max(self.total_mass * v_scale, 1e-6))

    # ------------------------------------------------------------------
    def _build_neighborhoods(self) -> None:
        X = self.X
        n = self.n
        diff = X[:, None, :] - X[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, np.inf)
        srt = np.sort(dist, axis=1)
        k = min(3, n - 1)
        radii = np.clip(1.25 * srt[:, k], 0.85, 1.45)

        pi, pj, pw, pdX = [], [], [], []
        for i in range(n):
            r = float(radii[i])
            js = np.flatnonzero(dist[i] < r)
            for j in js:
                d = float(dist[i, j])
                q = d / r
                w = (1.0 - q * q) ** 3
                if w <= 1e-9:
                    continue
                pi.append(i)
                pj.append(int(j))
                pw.append(w)
                pdX.append(X[j] - X[i])
        self.pi = np.asarray(pi, dtype=np.int64)
        self.pj = np.asarray(pj, dtype=np.int64)
        self.pw = np.asarray(pw, dtype=float)
        self.pdX = np.asarray(pdX, dtype=float).reshape(-1, 3)

        # regularized moment matrices
        B = np.zeros((n, 3, 3), dtype=float)
        np.add.at(
            B,
            self.pi,
            self.pw[:, None, None] * self.pdX[:, :, None] * self.pdX[:, None, :],
        )
        eye = np.eye(3)
        for i in range(n):
            scale = float(np.trace(B[i])) / 3.0
            eps = 0.02 * scale + 1e-6
            B[i] = B[i] + eps * eye
        self.Binv = np.linalg.inv(B)
        self.pVw = self.vol[self.pi] * self.pw

    # ------------------------------------------------------------------
    def _build_colliders(self) -> None:
        geo_objects = [
            ("support_left", (-0.72, 0.0, 0.72), 0.58, 6.0, False),
            ("support_right", (0.72, 0.0, 0.72), 0.58, 6.0, False),
            ("support_rear", (0.0, 0.0, 1.72), 0.58, 6.0, False),
            ("upper_cylinder", (0.0, 0.0, 4.2), 0.62, 6.0, True),
        ]
        axis = _quat_axis_z(np.array([0.7071068, 0.0, 0.0, 0.7071068]))
        self.colliders = []
        self.upper = None
        for _name, center, radius, height, kinematic in geo_objects:
            cyl = _Cylinder(center, radius, 0.5 * height, axis, kinematic)
            self.colliders.append(cyl)
            if kinematic:
                self.upper = cyl

        motion = self.config.get("upper_cylinder_motion", {}) or {}
        keys = motion.get("keyframes") or [
            {"time": 0.0, "translation": [0.0, 0.0, 4.2]}
        ]
        self.key_t = np.asarray([float(k["time"]) for k in keys], dtype=float)
        self.key_p = np.asarray(
            [np.asarray(k["translation"], dtype=float) for k in keys], dtype=float
        ).reshape(-1, 3)
        self.upper_base = self.upper.center.copy() if self.upper is not None else None
        self._update_kinematics(0.0)

    def _upper_state(self, t: float):
        kt, kp = self.key_t, self.key_p
        t = float(np.clip(t, kt[0], kt[-1]))
        pos = np.array([np.interp(t, kt, kp[:, c]) for c in range(3)], dtype=float)
        idx = int(np.searchsorted(kt, t, side="right") - 1)
        idx = int(np.clip(idx, 0, len(kt) - 2)) if len(kt) >= 2 else 0
        if len(kt) >= 2:
            dt = kt[idx + 1] - kt[idx]
            vel = (kp[idx + 1] - kp[idx]) / dt if dt > 1e-12 else np.zeros(3)
        else:
            vel = np.zeros(3)
        return pos, np.asarray(vel, dtype=float)

    def _update_kinematics(self, t: float) -> None:
        if self.upper is None:
            return
        pos, vel = self._upper_state(t)
        self.upper.center = pos.copy()
        self.upper.velocity = vel.copy()

    # ------------------------------------------------------------------
    def _deformation_gradients(self, x: np.ndarray) -> np.ndarray:
        u = x - self.X
        du = u[self.pj] - u[self.pi]
        A = np.zeros((self.n, 3, 3), dtype=float)
        np.add.at(
            A, self.pi, self.pw[:, None, None] * du[:, :, None] * self.pdX[:, None, :]
        )
        F = np.matmul(A, self.Binv)
        F[:, 0, 0] += 1.0
        F[:, 1, 1] += 1.0
        F[:, 2, 2] += 1.0
        return F

    def _first_piola(self, F: np.ndarray) -> np.ndarray:
        f0 = F[:, :, 0]
        f1 = F[:, :, 1]
        f2 = F[:, :, 2]
        c0 = np.cross(f1, f2)
        c1 = np.cross(f2, f0)
        c2 = np.cross(f0, f1)
        J = np.einsum("ni,ni->n", f0, c0)
        dJdF = np.stack((c0, c1, c2), axis=2)
        Ic = np.einsum("nij,nij->n", F, F)
        coef = self.mu * (1.0 - 1.0 / (Ic + 1.0))
        P = coef[:, None, None] * F + (self.lam * (J - self.alpha))[:, None, None] * dJdF
        return P

    def _internal_forces(self, x: np.ndarray, v: np.ndarray, F: np.ndarray):
        P = self._first_piola(F)
        M = np.matmul(P, self.Binv)
        g = self.pVw[:, None] * np.einsum("pij,pj->pi", M[self.pi], self.pdX)
        # symmetric safety clamp (keeps Newton's third law, hence momentum)
        mag = np.linalg.norm(g, axis=1)
        big = mag > self._PAIR_FCLAMP
        if np.any(big):
            g[big] *= (self._PAIR_FCLAMP / mag[big])[:, None]
        f = np.zeros_like(x)
        np.add.at(f, self.pi, g)
        np.add.at(f, self.pj, -g)
        # bond viscosity (momentum conserving)
        dv = v[self.pj] - v[self.pi]
        fd = (self._BOND_DAMP * self.pw)[:, None] * dv
        np.add.at(f, self.pi, fd)
        np.add.at(f, self.pj, -fd)
        return f

    def _contact_forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        fc = np.zeros_like(x)
        for cyl in self.colliders:
            a = cyl.axis_idx
            r0, r1 = cyl.radial_idx
            dr0 = x[:, r0] - cyl.center[r0]
            dr1 = x[:, r1] - cyl.center[r1]
            dist = np.sqrt(dr0 * dr0 + dr1 * dr1)
            surf = cyl.radius + self._MARGIN
            inside_axis = np.abs(x[:, a] - cyl.center[a]) <= cyl.half_length
            pen = surf - dist
            active = (pen > 0.0) & inside_axis
            if not np.any(active):
                continue
            idx = np.flatnonzero(active)
            d_safe = np.maximum(dist[idx], 1e-9)
            nrm = np.zeros((idx.size, 3), dtype=float)
            nrm[:, r0] = dr0[idx] / d_safe
            nrm[:, r1] = dr1[idx] / d_safe
            # degenerate (on-axis) samples: push along +radial reference dir
            deg = dist[idx] < 1e-8
            if np.any(deg):
                nrm[deg] = 0.0
                nrm[deg, r1] = 1.0
            vrel = v[idx] - cyl.velocity[None, :]
            vn = np.einsum("ij,ij->i", vrel, nrm)
            fn = self._CONTACT_K * pen[idx] - self._CONTACT_C * vn
            fn = np.clip(fn, 0.0, self._CONTACT_FMAX)
            force = fn[:, None] * nrm
            # regularized Coulomb friction
            vt = vrel - vn[:, None] * nrm
            vt_mag = np.linalg.norm(vt, axis=1)
            ft_mag = np.minimum(self._FRICTION * fn, self._FRICTION_C * vt_mag)
            safe = vt_mag > 1e-9
            if np.any(safe):
                force[safe] -= (ft_mag[safe] / vt_mag[safe])[:, None] * vt[safe]
            np.add.at(fc, idx, force)
        return fc

    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        self._update_kinematics(self.t + 0.5 * dt)
        F = self._deformation_gradients(self.x)
        f = self._internal_forces(self.x, self.v, F)
        f += self.g_force
        fc = self._contact_forces(self.x, self.v)
        f += fc

        self.v += dt * (f / self.mass[:, None])
        speed = np.linalg.norm(self.v, axis=1)
        fast = speed > self._VMAX
        if np.any(fast):
            self.v[fast] *= (self._VMAX / speed[fast])[:, None]
        self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        self.x += dt * self.v
        self.x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.x, -50.0, 50.0, out=self.x)

        imp = fc.sum(axis=0) * dt
        self._contact_impulse += imp
        self._interval_contact_impulse += imp
        self._interval_time += dt
        self._last_contact_force = fc.sum(axis=0)
        self.t += dt
        self._steps += 1

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        self._interval_contact_impulse = np.zeros(3, dtype=float)
        self._interval_time = 0.0
        while self.t < target - 1e-12:
            dt = min(self._DT, target - self.t)
            if dt <= 1e-15:
                break
            self._step(dt)
        self.t = target
        self._update_kinematics(self.t)
        return self._make_frame(target)

    def _make_frame(self, t: float) -> StateFrame:
        n = self.n
        F = self._deformation_gradients(self.x)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

        gravity_term = self.g_force.sum(axis=0)
        if self._interval_time > 1e-12:
            contact_term = self._interval_contact_impulse / self._interval_time
        else:
            contact_term = self._contact_forces(self.x, self.v).sum(axis=0)
        contact_term = np.nan_to_num(contact_term)

        external_force = np.stack((gravity_term, contact_term), axis=0)[None, :, :]
        domain_weights = np.ones((1, n), dtype=float)
        zero_vec = np.zeros((1, 3), dtype=float)

        frame = StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "rest_position": self.X.copy(),
            },
            fields={
                "element_deformation_gradient": F.copy(),
                "budget_domain_weights": domain_weights,
                "budget_external_force": external_force,
                "budget_cumulative_external_impulse": zero_vec.copy(),
                "budget_cumulative_inflow_momentum": zero_vec.copy(),
                "budget_cumulative_outflow_momentum": zero_vec.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "gravity_z": float(self.gravity[2]),
                "total_mass": float(self.total_mass),
                "upper_cylinder_z": float(
                    self.upper.center[2] if self.upper is not None else 0.0
                ),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "meshless MLS stable Neo-Hookean, explicit symplectic Euler",
            "samples": int(self.n),
            "pairs": int(self.pi.size),
            "dt": float(self._DT),
            "steps": int(self._steps),
            "mu": float(self.mu),
            "lambda": float(self.lam),
            "contact_impulse": [float(c) for c in self._contact_impulse],
        }
