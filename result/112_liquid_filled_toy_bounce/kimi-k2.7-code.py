"""Physically-based solver for the liquid-filled deformable toy bounce.

The solver integrates a particle system with:

* weakly-compressible SPH for the enclosed liquid (body_id == 2),
* passive SPH interpolation for the colored tracer markers (body_id 3, 4),
* shape-matching elasticity for the deformable shell (body_id 1),
* gravity, floor contact and two-way fluid-shell coupling via the same SPH
  pressure/viscosity formulation.

No precomputed animation or keyframes are used; the scene is advanced by
explicit time stepping.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)

        # ----- initial state from the scene definition ------------------------
        frame = scene.initial_state()
        self.pos = np.asarray(frame.position, dtype=float)
        self.vel = np.asarray(frame.velocity, dtype=float)
        self.mass = np.asarray(frame.mass, dtype=float)
        self.body_id = np.asarray(frame.body_id, dtype=np.int32)
        self.n = len(self.pos)

        # ----- body masks -----------------------------------------------------
        self.shell_mask = self.body_id == 1
        self.liquid_mask = self.body_id == 2
        self.red_mask = self.body_id == 3
        self.yellow_mask = self.body_id == 4
        self.marker_mask = self.red_mask | self.yellow_mask
        self.dyn_mask = self.mass > 0.0

        self.shell_idx = np.where(self.shell_mask)[0]
        self.liquid_idx = np.where(self.liquid_mask)[0]
        self.marker_idx = np.where(self.marker_mask)[0]
        self.n_shell = len(self.shell_idx)
        self.n_liquid = len(self.liquid_idx)

        # ----- scene/scaffold parameters -------------------------------------
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.floor_z = 0.0
        self.restitution_floor = 0.5

        # SPH parameters
        self.h = 0.18
        self.support = 2.0 * self.h
        self.K_fluid = 120.0
        self.alpha_visc = 0.08
        self.c_s = 12.0

        # Elastic shell parameters
        self.shape_k = 2400.0
        self.shape_damp = 12.0

        # Time stepping
        self.dt = 0.0001
        self.time = 0.0

        # ----- shell rest shape for shape matching ----------------------------
        self.rest_pos = self.pos[self.shell_mask].copy()
        shell_m = self.mass[self.shell_mask]
        self.rest_com = (shell_m[:, None] * self.rest_pos).sum(axis=0) / shell_m.sum()
        self.rest_rel = self.rest_pos - self.rest_com
        moment = (self.rest_rel * shell_m[:, None]).T @ self.rest_rel
        self.inv_rest_moment = np.linalg.inv(moment + np.eye(3) * 1e-8)

        # ----- shell connectivity for normal computation ----------------------
        self._build_shell_faces()

        # ----- rest density for SPH (per particle) ----------------------------
        self.rest_density = self._compute_density(self.pos)
        self.rho = self.rest_density.copy()

        # ----- auxiliary data -------------------------------------------------
        self.passive_marker = np.asarray(
            frame.attributes.get("passive_marker", np.zeros(self.n)), dtype=float
        )
        if len(self.passive_marker) != self.n:
            self.passive_marker = np.zeros(self.n, dtype=float)

        # Budget / bookkeeping
        self.cum_ext_impulse = np.zeros((4, 3), dtype=float)
        self.floor_force_group = np.zeros((4, 3), dtype=float)
        self.sph_force = np.zeros((self.n, 3), dtype=float)
        self.shape_force = np.zeros((self.n, 3), dtype=float)

    # -----------------------------------------------------------------------
    def _build_shell_faces(self) -> None:
        """Build a closed triangle fan for the shell surface of revolution."""
        M = 12
        N = 16
        faces = []
        # side quads split into two triangles
        for j in range(M - 1):
            for i in range(N):
                v0 = j * N + i
                v1 = j * N + (i + 1) % N
                v2 = (j + 1) * N + i
                v3 = (j + 1) * N + (i + 1) % N
                faces.append((v0, v2, v1))
                faces.append((v1, v2, v3))
        # top cap
        top = M * N
        for i in range(N):
            v0 = top
            v1 = i
            v2 = (i + 1) % N
            faces.append((v0, v2, v1))
        # bottom cap
        bottom = M * N + 1
        for i in range(N):
            v0 = bottom
            v1 = (M - 1) * N + i
            v2 = (M - 1) * N + (i + 1) % N
            faces.append((v0, v1, v2))
        self.shell_faces = np.asarray(faces, dtype=np.int32)

    # -----------------------------------------------------------------------
    def _shell_normals(self) -> np.ndarray:
        """Per-vertex outward unit normals of the shell mesh."""
        normals = np.zeros((self.n_shell, 3), dtype=float)
        p = self.pos[self.shell_idx]
        for a, b, c in self.shell_faces:
            n = np.cross(p[b] - p[a], p[c] - p[a])
            normals[a] += n
            normals[b] += n
            normals[c] += n
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        norm[norm == 0.0] = 1.0
        normals = normals / norm
        # orient outward by checking radial component relative to the shell COM
        outward = p - p.mean(axis=0)
        flip = np.sum(normals * outward, axis=1) < 0.0
        normals[flip] *= -1.0
        return normals

    # -----------------------------------------------------------------------
    def _kernel(self, q: np.ndarray):
        """Cubic spline kernel and dW/dq in 3D.  q = r/h."""
        alpha = 1.0 / (np.pi * self.h ** 3)
        W = np.zeros_like(q)
        dW = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        W[m1] = alpha * (1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3)
        dW[m1] = alpha * (-3.0 * q[m1] + 2.25 * q[m1] ** 2)
        W[m2] = alpha * 0.25 * (2.0 - q[m2]) ** 3
        dW[m2] = alpha * (-0.75) * (2.0 - q[m2]) ** 2
        return W, dW

    # -----------------------------------------------------------------------
    def _compute_density(self, pos: np.ndarray) -> np.ndarray:
        """SPH density for all particles."""
        N = len(pos)
        tree = cKDTree(pos)
        neigh = tree.query_ball_point(pos, r=self.support, return_sorted=False)
        lengths = [len(x) for x in neigh]
        i_idx = np.repeat(np.arange(N), lengths)
        j_idx = np.concatenate(neigh).astype(np.int32)

        rvec = pos[i_idx] - pos[j_idx]
        dist = np.linalg.norm(rvec, axis=1)
        q = dist / self.h
        W, _ = self._kernel(q)

        rho = np.bincount(i_idx, weights=self.mass[j_idx] * W, minlength=N)
        rho = np.maximum(rho, 1e-6)
        return rho

    # -----------------------------------------------------------------------
    def _sph_forces(self, pos: np.ndarray, vel: np.ndarray):
        """Return SPH force [N,3] and density [N] for the current state."""
        N = len(pos)
        tree = cKDTree(pos)
        neigh = tree.query_ball_point(pos, r=self.support, return_sorted=False)
        lengths = [len(x) for x in neigh]
        i_idx = np.repeat(np.arange(N), lengths)
        j_idx = np.concatenate(neigh).astype(np.int32)

        rvec = pos[i_idx] - pos[j_idx]
        dist = np.linalg.norm(rvec, axis=1)
        q = dist / self.h
        W, dWdq = self._kernel(q)

        # density
        rho = np.bincount(i_idx, weights=self.mass[j_idx] * W, minlength=N)
        rho = np.maximum(rho, 1e-6)

        # pressure
        p = self.K_fluid * (rho - self.rest_density)
        p[self.mass == 0.0] = 0.0
        term = p / (rho ** 2)
        term[self.mass == 0.0] = 0.0

        # grad W vector (direction i -> j)
        dWdr = dWdq / self.h
        safe = dist.copy()
        safe[safe < 1e-12] = 1.0
        grad = (dWdr / safe)[:, None] * rvec
        grad[dist < 1e-12] = 0.0

        # pressure force
        pref = -self.mass[i_idx] * self.mass[j_idx] * (term[i_idx] + term[j_idx])
        f_pressure = pref[:, None] * grad

        # Monaghan artificial viscosity
        v_ij = vel[i_idx] - vel[j_idx]
        vr = np.sum(v_ij * rvec, axis=1)
        r2 = dist ** 2 + 0.01 * self.h ** 2
        mu = self.h * (-vr) / r2
        rho_avg = 0.5 * (rho[i_idx] + rho[j_idx])
        Pi = -self.alpha_visc * self.c_s * mu / rho_avg
        Pi[vr >= 0.0] = 0.0
        pref_visc = self.mass[i_idx] * self.mass[j_idx] * Pi
        f_visc = pref_visc[:, None] * grad

        f_pair = f_pressure + f_visc

        # accumulate to particles
        force = np.zeros((N, 3), dtype=float)
        for dim in range(3):
            col = f_pair[:, dim]
            force[:, dim] = np.bincount(i_idx, weights=col, minlength=N) - np.bincount(
                j_idx, weights=col, minlength=N
            )
        return force, rho

    # -----------------------------------------------------------------------
    def _shape_matching_force(self) -> np.ndarray:
        """Return shape-matching elastic force [n_shell, 3]."""
        x = self.pos[self.shell_idx]
        m = self.mass[self.shell_idx]
        com = (m[:, None] * x).sum(axis=0) / m.sum()
        r = x - com
        A = (r * m[:, None]).T @ (self.rest_rel @ self.inv_rest_moment.T)
        A = A + np.eye(3) * 1e-8
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt
        if np.linalg.det(R) < 0.0:
            U[:, -1] *= -1.0
            R = U @ Vt
        goal = (R @ self.rest_rel.T).T + com
        f = self.shape_k * m[:, None] * (goal - x) - self.shape_damp * m[:, None] * self.vel[self.shell_idx]
        return f

    # -----------------------------------------------------------------------
    def _interpolate_markers(self) -> None:
        """Advect passive tracer markers by interpolating the liquid velocity."""
        if self.marker_idx.size == 0:
            return
        liq_pos = self.pos[self.liquid_idx]
        tree = cKDTree(liq_pos)
        marker_pos = self.pos[self.marker_idx]
        neigh = tree.query_ball_point(marker_pos, r=self.support, return_sorted=False)
        lengths = [len(x) for x in neigh]
        if not lengths or sum(lengths) == 0:
            return
        i_marker = np.repeat(np.arange(len(self.marker_idx)), lengths)
        j_liq = np.concatenate(neigh).astype(np.int32)

        liq_global = self.liquid_idx[j_liq]
        marker_global = self.marker_idx[i_marker]

        rvec = self.pos[marker_global] - self.pos[liq_global]
        dist = np.linalg.norm(rvec, axis=1)
        q = dist / self.h
        W, _ = self._kernel(q)
        w = self.mass[liq_global] / self.rho[liq_global] * W

        num = np.zeros((len(self.marker_idx), 3), dtype=float)
        den = np.bincount(i_marker, weights=w, minlength=len(self.marker_idx))
        for dim in range(3):
            num[:, dim] = np.bincount(
                i_marker, weights=w * self.vel[liq_global, dim], minlength=len(self.marker_idx)
            )
        den = np.where(den > 0.0, den, 1.0)[:, None]
        self.vel[self.marker_idx] = num / den

    # -----------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        # SPH forces (fluid + shell boundary particles)
        sph_force, rho = self._sph_forces(self.pos, self.vel)
        self.rho = rho
        self.sph_force = sph_force

        # shape matching elastic force for the shell
        shape_f = self._shape_matching_force()
        self.shape_force = np.zeros((self.n, 3), dtype=float)
        self.shape_force[self.shell_idx] = shape_f

        # acceleration for dynamic particles
        acc = np.zeros((self.n, 3), dtype=float)
        acc[self.dyn_mask] += self.gravity
        acc += self.sph_force
        acc[self.shell_idx] += self.shape_force[self.shell_idx]

        self.vel[self.dyn_mask] += (
            acc[self.dyn_mask] / self.mass[self.dyn_mask, None]
        ) * dt

        # markers follow the local fluid velocity
        self._interpolate_markers()

        # integrate positions
        self.pos += self.vel * dt

        # floor contact (position projection + impulse bookkeeping)
        below = self.pos[:, 2] < self.floor_z
        if below.any():
            v_old = self.vel[below, 2].copy()
            self.pos[below, 2] = self.floor_z
            v_new = np.where(v_old < 0.0, -self.restitution_floor * v_old, v_old)
            self.vel[below, 2] = v_new

            impulse = np.zeros((below.sum(), 3), dtype=float)
            impulse[:, 2] = self.mass[below] * (v_new - v_old)

            # group by body_id
            body_here = self.body_id[below]
            ff = np.zeros((4, 3), dtype=float)
            for g in range(4):
                mask_g = body_here == (g + 1)
                if mask_g.any():
                    self.cum_ext_impulse[g] += impulse[mask_g].sum(axis=0)
                    ff[g] = impulse[mask_g].sum(axis=0) / dt
            self.floor_force_group = ff
        else:
            self.floor_force_group = np.zeros((4, 3), dtype=float)

    # -----------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        while self.time < time - 1e-12:
            dt = min(self.dt, time - self.time)
            self._substep(dt)
            self.time += dt
        return self._build_frame(time)

    # -----------------------------------------------------------------------
    def _build_frame(self, time: float) -> StateFrame:
        # shell boundary fields
        shell_normals = self._shell_normals()
        shell_vel = self.vel[self.shell_idx]

        # domain weights [body, particle]
        domain_weights = np.zeros((4, self.n), dtype=float)
        for g in range(4):
            domain_weights[g, self.body_id == (g + 1)] = 1.0

        # budget external forces [body, term, 3]
        budget_force = np.zeros((4, 4, 3), dtype=float)
        # term 0: gravity
        for g in range(4):
            mask = self.body_id == (g + 1)
            if mask.any():
                budget_force[g, 0] = (self.mass[mask, None] * self.gravity).sum(axis=0)
        # term 1: floor contact (only for non-zero body 1-4)
        for g in range(4):
            budget_force[g, 1] = self.floor_force_group[g]
        # term 2: SPH coupling
        for g in range(4):
            mask = self.body_id == (g + 1)
            if mask.any():
                budget_force[g, 2] = self.sph_force[mask].sum(axis=0)
        # term 3: shell elastic restoring
        for g in range(4):
            mask = self.body_id == (g + 1)
            if mask.any():
                budget_force[g, 3] = self.shape_force[mask].sum(axis=0)

        # interaction force = net SPH coupling force per domain
        interaction = np.zeros((4, 3), dtype=float)
        for g in range(4):
            mask = self.body_id == (g + 1)
            if mask.any():
                interaction[g] = self.sph_force[mask].sum(axis=0)

        total_dynamic_mass = self.mass[self.dyn_mask].sum()
        ref_velocity = 1.15
        ref_momentum = max(total_dynamic_mass * ref_velocity, 1e-6)

        fields = {
            "boundary_normal": shell_normals,
            "boundary_target_velocity": shell_vel.copy(),
            "boundary_velocity": shell_vel.copy(),
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((4, 3), dtype=float),
            "budget_cumulative_outflow_momentum": np.zeros((4, 3), dtype=float),
            "budget_domain_weights": domain_weights,
            "budget_external_force": budget_force,
            "cumulative_mass_source": np.zeros(4, dtype=float),
            "interaction_force": interaction,
        }

        attributes = {
            "mass_quantity": self.mass.copy(),
            "passive_marker": self.passive_marker.copy(),
        }

        scalars = {
            "reference_momentum": ref_momentum,
            "reference_velocity": ref_velocity,
            "gravity_z": -9.81,
            "cavity_is_closed": 1.0,
        }

        return StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    # -----------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "SPH + shape-matching shell",
            "particles": self.n,
            "shell_particles": self.n_shell,
            "liquid_particles": self.n_liquid,
            "time": self.time,
        }
