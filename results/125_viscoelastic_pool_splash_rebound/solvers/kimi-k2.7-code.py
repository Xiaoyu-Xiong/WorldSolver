"""Particle-based viscoelastic fluid solver for the pool splash and rebound scene.

The solver integrates a weakly-compressible SPH-style particle system with an
elastic deformation-gradient stress.  Each particle carries a deformation
gradient ``F`` that is advected by the local velocity gradient and relaxed
toward identity to model viscoelastic flow.  Gravity drives the falling mass,
and box-signed-distance collisions enforce the rigid basin.

Required StateFrame fields are produced at every requested sample time.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

        frame0 = scene.initial_state()
        self.x = np.asarray(frame0.position, dtype=float).copy()
        self.v = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.F = np.asarray(frame0.attributes["deformation_gradient"], dtype=float).copy()
        self.mass_quantity = np.asarray(frame0.attributes["mass_quantity"], dtype=float).copy()
        self.particle_radius = np.asarray(frame0.attributes["particle_radius"], dtype=float).copy()
        self.material_label = np.asarray(frame0.attributes["material_label"], dtype=np.int32).copy()

        scalars = frame0.scalars
        self.spacing = float(scalars["grid_spacing"])
        self.contact_length_scale = self.spacing
        self.characteristic_length = float(scalars.get("characteristic_length", self.spacing))
        self.reference_velocity = float(scalars.get("reference_velocity", 1.0))

        self.density0 = float(scene.config["sampling"]["density"])
        self.gravity = np.asarray(scene.config["gravity"], dtype=float)

        # Rigid obstacles from the frozen scene geometry.
        geom = scene.config.get("scene_geometry", {})
        self.colliders = []
        for obj in geom.get("objects", []):
            if obj.get("collision_enabled") and obj["geometry"]["type"] == "box":
                c = np.asarray(obj["transform"]["translation"], dtype=float)
                h = np.asarray(obj["geometry"]["half_extents"], dtype=float)
                self.colliders.append((c, h))

        # Numerical and material parameters.
        self.dt = 0.0005
        self.h = self.spacing
        self.c_s = 50.0  # speed of sound for the equation of state
        self.gamma = 7.0
        self.B = self.c_s * self.c_s * self.density0 / self.gamma
        self.mu = 10000.0  # shear elastic modulus
        self.lam = 15000.0  # second elastic modulus
        self.alpha = 0.25  # artificial viscosity coefficient
        self.tau_visco = 0.25  # elastic relaxation time (viscoelastic flow)
        self.restitution = 0.0
        self.friction = 0.1
        self.v_max = 30.0

        self.time = 0.0
        self.step_count = 0
        self._eye3 = np.eye(3)[None, ...]

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "step_count": self.step_count,
            "dt": self.dt,
            "particle_count": len(self.x),
        }

    def _kernel(self, r: np.ndarray, h: float) -> np.ndarray:
        q = r / h
        sigma = 1.0 / (np.pi * h ** 3)
        w = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        w[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
        w[m2] = 0.25 * (2.0 - q[m2]) ** 3
        return sigma * w

    def _kernel_grad(self, rvec: np.ndarray, h: float) -> np.ndarray:
        r = np.linalg.norm(rvec, axis=1)
        q = r / h
        sigma = 1.0 / (np.pi * h ** 3)
        dw = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        dw[m1] = -3.0 * q[m1] + 2.25 * q[m1] ** 2
        dw[m2] = -0.75 * (2.0 - q[m2]) ** 2
        dWdr = sigma / h * dw
        out = np.zeros_like(rvec)
        nz = r > 1e-12
        out[nz] = (dWdr[nz] / r[nz])[:, None] * rvec[nz]
        return out

    def _neighbors(self) -> tuple:
        tree = cKDTree(self.x)
        pairs = tree.query_pairs(r=2.0 * self.h, output_type="ndarray")
        if len(pairs) == 0:
            return (
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=int),
                np.empty((0, 3), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0,), dtype=float),
                np.empty((0, 3), dtype=float),
            )
        i = pairs[:, 0]
        j = pairs[:, 1]
        rvec = self.x[i] - self.x[j]
        dist = np.linalg.norm(rvec, axis=1)
        W = self._kernel(dist, self.h)
        gradW = self._kernel_grad(rvec, self.h)
        return i, j, rvec, dist, W, gradW

    def _stress(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (P, R, F_inv_T, J) using a fixed-corotated elastic model."""
        U, s, Vt = np.linalg.svd(F)
        s = np.clip(s, 0.1, 10.0)
        # Reconstruct a stabilized F for the stress calculation.
        F_stable = np.einsum("nij,nj,njk->nik", U, s, Vt)
        R = np.einsum("nij,njk->nik", U, Vt)
        F_inv_T = np.einsum("nij,nj,njk->nik", U, 1.0 / s, Vt)
        J = np.prod(s, axis=1)
        P = 2.0 * self.mu * (F_stable - R) + self.lam * (J - 1.0) * J[:, None, None] * F_inv_T
        return P, R, F_inv_T, J

    def _resolve_collisions(self) -> None:
        if not self.colliders:
            return
        centers = np.stack([c for c, _ in self.colliders])
        halfs = np.stack([h for _, h in self.colliders])
        # Signed distance for each particle against each box.
        rel = self.x[None, :, :] - centers[:, None, :]
        dvec = np.abs(rel) - halfs[:, None, :]
        sd_all = np.max(dvec, axis=2)
        min_idx = np.argmin(sd_all, axis=0)
        sd = sd_all[min_idx, np.arange(len(self.x))]

        # Gradient of the minimum signed-distance field.
        mask = np.isclose(dvec, sd_all[:, :, None], atol=1e-12)
        signs = np.sign(rel)
        grad_raw = signs * mask
        grad_norm = np.linalg.norm(grad_raw, axis=2)
        grad_norm[grad_norm == 0.0] = 1.0
        grad_unit = grad_raw / grad_norm[:, :, None]
        g = grad_unit[min_idx, np.arange(len(self.x))]

        pen = sd < self.particle_radius
        if not np.any(pen):
            return
        delta = self.particle_radius[pen] - sd[pen]
        self.x[pen] += g[pen] * delta[:, None]

        v_dot_g = np.sum(self.v[pen] * g[pen], axis=1)
        v_n = np.minimum(v_dot_g, 0.0)
        self.v[pen] -= (1.0 + self.restitution) * v_n[:, None] * g[pen]

        v_t = self.v[pen] - np.sum(self.v[pen] * g[pen], axis=1)[:, None] * g[pen]
        v_t_mag = np.linalg.norm(v_t, axis=1)
        reduction = np.clip(self.friction * np.abs(v_n), 0.0, v_t_mag)
        self.v[pen] -= (reduction / (v_t_mag + 1e-12))[:, None] * v_t

    def _contact_signed_distance(self) -> np.ndarray:
        """Union signed distance over all box obstacles (negative = penetration)."""
        if not self.colliders:
            return np.full(len(self.x), 1e6, dtype=float)
        centers = np.stack([c for c, _ in self.colliders])
        halfs = np.stack([h for _, h in self.colliders])
        rel = self.x[None, :, :] - centers[:, None, :]
        dvec = np.abs(rel) - halfs[:, None, :]
        sd_all = np.max(dvec, axis=2)
        return np.min(sd_all, axis=0)

    def _step(self, dt: float | None = None) -> None:
        dt = self.dt if dt is None else float(dt)
        N = len(self.x)
        m = self.mass
        h = self.h

        i, j, rvec, dist, W, gradW = self._neighbors()

        # Density via SPH sum.
        self_contrib = m[0] * (1.0 / (np.pi * h ** 3))
        rho = np.full(N, self_contrib, dtype=float)
        if len(i):
            np.add.at(rho, i, m[j] * W)
            np.add.at(rho, j, m[i] * W)
        rho = np.clip(rho, 0.1 * self.density0, 10.0 * self.density0)

        # Pressure (Tait).
        p = self.B * ((rho / self.density0) ** self.gamma - 1.0)

        # Velocity gradient for advecting F.
        V = m / rho
        L = np.zeros((N, 3, 3), dtype=float)
        if len(i):
            dv = self.v[j] - self.v[i]
            # L_i += V_j * (v_j - v_i) outer gradW_ij
            outer = dv[:, :, None] * gradW[:, None, :]
            np.add.at(L, i, V[j][:, None, None] * outer)

        # Viscoelastic evolution of F and stress computation.
        dFdt = np.einsum("nij,njk->nik", L, self.F) - (self.F - self._eye3) / self.tau_visco
        F_new = self.F + dt * dFdt
        # Clamp singular values to prevent inversion/explosion.
        U, s, Vt = np.linalg.svd(F_new)
        s = np.clip(s, 0.1, 10.0)
        self.F = np.einsum("nij,nj,njk->nik", U, s, Vt)
        P, _, _, _ = self._stress(self.F)

        # Forces.
        f = np.zeros((N, 3), dtype=float)

        # Pressure + artificial viscosity (Monaghan).
        if len(i):
            p_term = p[i] / rho[i] ** 2 + p[j] / rho[j] ** 2
            v_ij = self.v[i] - self.v[j]
            vr = np.sum(v_ij * rvec, axis=1)
            mu_visc = np.zeros_like(dist)
            mask = vr < 0.0
            mu_visc[mask] = h * vr[mask] / (dist[mask] ** 2 + 0.01 * h ** 2)
            pi_ij = np.zeros_like(dist)
            pi_ij[mask] = -self.alpha * self.c_s * mu_visc[mask]
            bracket = p_term + pi_ij
            f_pair = -m[i] * m[j] * bracket[:, None] * gradW
            np.add.at(f, i, f_pair)
            np.add.at(f, j, -f_pair)

        # Elastic stress force from the deformation gradient.
        if len(i):
            Psum = 0.5 * (P[i] + P[j])
            pg = np.einsum("mij,mj->mi", Psum, gradW)
            val = V[i] * V[j] * pg
            np.add.at(f, i, -val)
            np.add.at(f, j, val)

        # External force: gravity only.
        f += m[:, None] * self.gravity[None, :]

        # Integrate velocity and position.
        self.v += dt * f / m[:, None]
        self.x += dt * self.v

        # Soft velocity cap as a safeguard against run-away explicit integration.
        v_mag = np.linalg.norm(self.v, axis=1)
        scale = np.minimum(1.0, self.v_max / (v_mag + 1e-12))
        self.v *= scale[:, None]

        self._resolve_collisions()

        self.time += dt
        self.step_count += 1

    def _build_frame(self, time: float) -> StateFrame:
        # Number of conserved materials = distinct body ids.
        n_materials = int(self.body_id.max() + 1) if len(self.body_id) else 1
        return StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": self.F.copy(),
                "mass_quantity": self.mass_quantity.copy(),
                "particle_radius": self.particle_radius.copy(),
                "material_label": self.material_label.copy(),
            },
            fields={
                "contact_signed_distance": self._contact_signed_distance().copy(),
                "cumulative_mass_source": np.zeros(n_materials, dtype=float),
                "external_force": (self.mass[:, None] * self.gravity[None, :]).copy(),
            },
            scalars={
                "contact_length_scale": float(self.contact_length_scale),
                "grid_spacing": float(self.spacing),
                "characteristic_length": float(self.characteristic_length),
                "reference_velocity": float(self.reference_velocity),
            },
        )

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        # If already there (e.g. the initial frame), return current state.
        if target <= self.time + 1e-12:
            return self._build_frame(target)

        remaining = target - self.time
        n_full = int(remaining // self.dt)
        for _ in range(n_full):
            self._step()

        leftover = target - self.time
        if leftover > 1e-12:
            self._step(dt=leftover)

        return self._build_frame(target)
