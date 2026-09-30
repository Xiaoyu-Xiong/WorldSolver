"""Physically-based Newton cradle breakaway solver.

Model
-----
Seven identical balls, each suspended by a rigid (inextensible, massless)
string from a fixed pivot on the cradle top bar.  Each ball is therefore a
planar pendulum with generalized coordinate theta_i:

    p_i = pivot_i + L * (sin(theta_i), 0, -cos(theta_i))
    v_i = L * omega_i * (cos(theta_i), 0, sin(theta_i))

The string constraint is enforced exactly by this parameterization, so the
support reaction never appears as an external force; it is exported in
``fields.interaction_force`` as the ball/string tension coupling.

Free-flight dynamics:  theta_ddot = -(g/L) sin(theta) - c_air * omega,
integrated with semi-implicit Euler at h = 2e-4 s.

Impacts: adjacent-pair sphere contacts solved with a sequential-impulse
(Gauss-Seidel) sweep in the direction of impulse propagation, with a
near-elastic normal restitution (steel spheres).  Sequential propagation
through the touching chain reproduces the simultaneous-impact breakaway
result: the incoming end ball stops, interior balls remain nearly at rest,
and the far end ball ejects with a matching swing velocity.

Exported fields per frame:
  state.position/velocity/orientation/angular_velocity/mass/body_id
  fields.contact_signed_distance  [6]  gaps of adjacent ball pairs
  fields.external_force           [7,3] gravity + air drag
  fields.interaction_force        [13,3] 6 contact pair forces + 7 tensions
  scalars.contact_length_scale
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

_H = 2.0e-4          # physics substep (s)
_GS_ITERS = 4        # Gauss-Seidel sweeps per substep
_E_BALL = 0.99       # sphere-sphere normal restitution (near-elastic steel)
_C_AIR = 0.03        # linear air damping on angular rate (1/s)
_SLOP = 1.0e-4       # contact allowance (m)
_FALLBACK_L = 1.10   # string length (m)
_FALLBACK_R = 0.23   # ball radius (m)
_FALLBACK_PIVOT_Z = 2.15
_THETA0 = -0.60      # initial displacement of ball_0 (rad), released from rest


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        bodies = cfg["bodies"]
        self.n = len(bodies)
        self.m = np.array([float(b["mass"]) for b in bodies])
        self.g = np.array(cfg["gravity"], dtype=float)
        rest = np.array([b["position"] for b in bodies], dtype=float)

        radius, pivot_z = _FALLBACK_R, _FALLBACK_PIVOT_Z
        geom = cfg.get("scene_geometry")
        if isinstance(geom, dict):
            for obj in geom.get("objects", []):
                gid = obj.get("id", "")
                g = obj.get("geometry", {})
                if gid.startswith("ball_") and g.get("type") == "sphere":
                    radius = float(g.get("radius", radius))
                elif gid.startswith("string_") and g.get("type") == "capsule":
                    pivot_z = max(pivot_z, float(obj["transform"]["translation"][2]))
        self.r = radius

        self.pivot = np.zeros((self.n, 3))
        self.pivot[:, 0] = rest[:, 0]
        self.pivot[:, 1] = rest[:, 1]
        self.pivot[:, 2] = pivot_z
        self.L = np.linalg.norm(self.pivot - rest, axis=1)
        if not np.all(np.isfinite(self.L)) or np.any(self.L <= 1e-6):
            self.L[:] = _FALLBACK_L

        # Initial generalized state: end ball displaced, row touching at rest.
        self.theta = np.zeros(self.n)
        self.omega = np.zeros(self.n)
        d = rest - self.pivot
        self.theta[0] = np.arctan2(d[0, 0], -d[0, 2]) + _THETA0

        self.t = 0.0
        self._last_lambda = np.zeros(self.n - 1)
        self._diag = {"contacts": 0, "steps": 0}

    # --------------------------------------------------------------- helpers
    def _pos(self):
        st, ct = np.sin(self.theta), np.cos(self.theta)
        p = self.pivot.copy()
        p[:, 0] += self.L * st
        p[:, 2] -= self.L * ct
        return p

    def _vel(self):
        st, ct = np.sin(self.theta), np.cos(self.theta)
        v = np.zeros((self.n, 3))
        v[:, 0] = self.L * self.omega * ct
        v[:, 2] = self.L * self.omega * st
        return v

    def _gaps(self, pos):
        d = np.diff(pos, axis=0)
        dist = np.linalg.norm(d, axis=1)
        return dist - 2.0 * self.r, d / dist[:, None]

    # ------------------------------------------------------------------ step
    def _substep(self, h, lam_acc):
        gvec = self.g
        # --- free flight: semi-implicit Euler on (theta, omega) ---
        st, ct = np.sin(self.theta), np.cos(self.theta)
        # tangential gravity component: g . (cos, 0, sin)
        g_tan = gvec[0] * ct + gvec[2] * st
        alpha = g_tan / self.L - _C_AIR * self.omega
        self.omega = self.omega + alpha * h
        self.theta = self.theta + self.omega * h

        # --- contacts between adjacent balls (sequential impulses) ---
        lam = np.zeros(self.n - 1)
        idx = np.arange(self.n - 1)
        for it in range(_GS_ITERS):
            forward = (it % 2) == 0
            order = idx if forward else idx[::-1]
            pos = self._pos()
            gaps, normals = self._gaps(pos)
            for k in order:
                g = gaps[k]
                if g > _SLOP:
                    continue
                nk = normals[k]
                i, j = k, k + 1
                sti, ctj = np.sin(self.theta[i]), np.cos(self.theta[j])
                stj, cti = np.sin(self.theta[j]), np.cos(self.theta[i])
                ti = np.array((cti, 0.0, sti))
                tj = np.array((ctj, 0.0, stj))
                ci = self.L[i] * float(ti @ nk)
                cj = self.L[j] * float(tj @ nk)
                vrel = cj * self.omega[j] - ci * self.omega[i]
                # generalized inertia of a pendulum coordinate is m L^2
                keff = ci * ci / (self.m[i] * self.L[i] ** 2) + cj * cj / (
                    self.m[j] * self.L[j] ** 2
                )
                if keff <= 1e-14:
                    continue
                e = _E_BALL if g < -1e-9 or vrel < -1e-6 else 0.0
                target = -e * min(vrel, 0.0)
                # Baumgarte-style separation of persistent overlap
                bias = 0.0
                if g < -_SLOP:
                    bias = 0.2 * (-g - _SLOP) / h
                dlam = (target - vrel + bias) / keff
                if lam[k] + dlam < 0.0:
                    dlam = -lam[k]
                if dlam == 0.0:
                    continue
                lam[k] += dlam
                self.omega[i] -= dlam * ci / (self.m[i] * self.L[i] ** 2)
                self.omega[j] += dlam * cj / (self.m[j] * self.L[j] ** 2)
        lam_acc += lam
        if np.any(lam > 0.0):
            self._diag["contacts"] += 1

    # ------------------------------------------------------------ advance_to
    def advance_to(self, time: float):
        time = float(time)
        remaining = time - self.t
        if remaining > 1e-12:
            nsteps = max(1, int(np.ceil(remaining / _H)))
            h = remaining / nsteps
            lam_acc = np.zeros(self.n - 1)
            for _ in range(nsteps):
                self._substep(h, lam_acc)
            self._diag["steps"] += nsteps
            self.t = time
            self._last_lambda = lam_acc / max(remaining, 1e-12)
        else:
            self._last_lambda = np.zeros(self.n - 1)

        pos = self._pos()
        vel = self._vel()
        gaps, normals = self._gaps(pos)

        # External force: gravity + air drag (constraint/contact excluded).
        ext = np.tile(self.g * self.m[:, None], (1, 1))
        ext -= _C_AIR * self.m[:, None] * vel

        # Interaction forces: pair contact forces then string tensions.
        f_contact = self._last_lambda[:, None] * normals  # force on left ball
        d = pos - self.pivot
        rhat = d / np.linalg.norm(d, axis=1)[:, None]
        tension = self.m * (self.L * self.omega**2 - (self.g @ rhat.T))
        f_tension = tension[:, None] * (-rhat)  # pulls ball toward pivot
        interaction = np.vstack([f_contact, f_tension])

        return StateFrame(
            float(time),
            pos,
            vel,
            orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (self.n, 1)),
            angular_velocity=np.zeros((self.n, 3)),
            mass=self.m.copy(),
            body_id=np.arange(self.n, dtype=np.int32),
            attributes={},
            fields={
                "contact_signed_distance": np.asarray(gaps, dtype=float),
                "external_force": ext,
                "interaction_force": interaction,
            },
            scalars={"contact_length_scale": 0.02},
        )

    # ----------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "pendulum_newton_cradle",
            "substeps": int(self._diag["steps"]),
            "substeps_with_contact": int(self._diag["contacts"]),
            "restitution_ball": _E_BALL,
        }
