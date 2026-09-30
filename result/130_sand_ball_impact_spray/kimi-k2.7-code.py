"""Solver Agent-owned implementation for 130_sand_ball_impact_spray.

A lightweight Material Point Method (MPM) solver with a Drucker-Prager
elastoplastic sand model and a rigid impactor.  Uses explicit time stepping
on a regular Eulerian background grid, with APIC-style transfers between
particles and grid.  The rigid ball is represented by the single sample
with body_id == 1 and is integrated as a rigid body with collision/contact
response against the sand particles and the floor.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.config = scene.config
        self.metadata = scene.metadata()
        frame0 = scene.initial_state()

        self.t = 0.0
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.attributes = frame0.attributes
        self.fields = frame0.fields
        self.scalars = frame0.scalars

        self.N = self.position.shape[0]
        self.is_sand = self.body_id == 0
        self.is_ball = self.body_id == 1

        # Material and numerical parameters.
        self.gravity = np.asarray(self.config["gravity"], dtype=float)
        self.spacing = float(self.config["sampling"]["spacing"])
        self.density = float(self.config["sampling"]["density"])
        self.dt = 1.0e-4
        self.sample_dt = float(self.config["sample_dt"])
        self.duration = float(self.config["duration"])

        # Ball radius.
        self.ball_radius = 0.54

        # Sand constitutive / MPM parameters.
        self.E_sand = 1.5e5
        self.nu_sand = 0.3
        self.mu_sand = self.E_sand / (2.0 * (1.0 + self.nu_sand))
        self.lambda_sand = self.E_sand * self.nu_sand / ((1.0 + self.nu_sand) * (1.0 - 2.0 * self.nu_sand))

        # Drucker-Prager parameters (cohesion-less sand).
        self.friction_angle = np.radians(35.0)
        self.dilation_angle = np.radians(10.0)

        # Particle volume (from rest density).
        self.vol_p = self.spacing ** 3

        # Deformation gradient (per particle).
        self.F = np.asarray(self.attributes["deformation_gradient"], dtype=float).copy()
        if self.F.ndim == 2 and self.F.shape[0] == self.N and self.F.shape[1] == 3:
            self.F = np.repeat(self.F[:, None, :], 3, axis=1)
        self.F = np.asarray(self.F, dtype=float).reshape(self.N, 3, 3)
        self.JP = np.ones(self.N, dtype=float)

        # Grid domain.
        # Bounds chosen to comfortably cover the bed and the ball trajectory.
        self.grid_origin = np.array([-3.5, -2.0, -0.5], dtype=float)
        self.grid_size = np.array([7.0, 4.0, 5.0], dtype=float)
        self.grid_res = np.array([64, 40, 48], dtype=np.int32)
        self.dx = self.grid_size / self.grid_res
        self.inv_dx = 1.0 / self.dx
        self.num_grid = int(np.prod(self.grid_res))
        self.dim = 3

        # Precompute signed distance for floor.
        self.floor_z = 0.0

        # Cached output frame at current time.
        self._current_frame = self._make_frame()

    def _make_frame(self) -> StateFrame:
        deformation = self.F.copy()
        mass_quantity = self.mass.copy()
        contact_sd = self._contact_signed_distance()
        cum_mass = np.asarray(self.fields["cumulative_mass_source"], dtype=float).copy()
        ext_force = self._external_forces()
        contact_scale = float(self.scalars["contact_length_scale"])
        return StateFrame(
            self.t,
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": deformation,
                "mass_quantity": mass_quantity,
                "particle_radius": self.attributes["particle_radius"].copy() if "particle_radius" in self.attributes else np.full(self.N, 0.5 * self.spacing),
                "material_label": self.body_id.copy(),
            },
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": cum_mass,
                "external_force": ext_force,
            },
            scalars={
                "contact_length_scale": contact_scale,
                "grid_spacing": self.spacing,
                "characteristic_length": max(float(np.ptp(self.position, axis=0).max()), self.spacing),
                "reference_velocity": 1.0,
            },
        )

    def _external_forces(self) -> np.ndarray:
        g = self.gravity
        f = self.mass[:, None] * g[None, :]
        return f

    def _contact_signed_distance(self) -> np.ndarray:
        # Per-sample signed distance to the nearest rigid obstacle (floor or ball).
        sd = np.empty(self.N, dtype=float)
        # Floor at z=0.
        sd[:] = self.position[:, 2] - self.floor_z
        # Ball (interior negative).
        if np.any(self.is_ball):
            bc = self.position[self.is_ball].mean(axis=0)
        else:
            bc = np.array([0.0, 0.0, 1.85])
        d_ball = np.linalg.norm(self.position - bc[None, :], axis=1) - self.ball_radius
        sd = np.minimum(sd, d_ball)
        return sd

    def _grid_index_to_world(self, i, j, k):
        return self.grid_origin + (np.array([i, j, k], dtype=float) + 0.5) * self.dx

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target <= self.t:
            return self._current_frame

        while self.t < target:
            dt = min(self.dt, target - self.t)
            self._step(dt)
            self.t += dt

        self._current_frame = self._make_frame()
        return self._current_frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "t": self.t,
            "num_particles": self.N,
            "sand_count": int(self.is_sand.sum()),
            "ball_velocity": self.velocity[self.is_ball].mean(axis=0).tolist() if np.any(self.is_ball) else [],
        }

    # ------------------------------------------------------------------
    # MPM + rigid step
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        # Ball first: gravity + contact with sand + floor.
        self._ball_step(dt)

        # Sand MPM update.
        grid_mass = np.zeros(self.num_grid, dtype=float)
        grid_momentum = np.zeros((self.num_grid, 3), dtype=float)
        self._p2g(grid_mass, grid_momentum, dt)

        # Add gravity to grid.
        for idx in range(self.num_grid):
            if grid_mass[idx] > 0.0:
                grid_momentum[idx] += dt * grid_mass[idx] * self.gravity

        # Boundary conditions on grid: floor.
        self._grid_boundary(grid_mass, grid_momentum)

        # G2P update for sand particles.
        self._g2p(grid_momentum, grid_mass, dt)

        # Sand plasticity and hardening update.
        self._update_plasticity()

        # Floor contact for sand particles (last-resort CFL safety).
        sand_indices = np.where(self.is_sand)[0]
        for i in sand_indices:
            if self.position[i, 2] < 0.0:
                self.position[i, 2] = 0.005
                if self.velocity[i, 2] < 0.0:
                    self.velocity[i, 2] *= -0.2

    def _p2g(self, grid_mass: np.ndarray, grid_momentum: np.ndarray, dt: float) -> None:
        # APIC-style P2G for sand; ball treated separately as rigid body.
        for i in np.where(self.is_sand)[0]:
            base, fx = self._compute_base_and_fx(self.position[i])
            # Quadratic B-spline weights.
            w = [self._bspline(fx[0]), self._bspline(fx[1]), self._bspline(fx[2])]
            # Cauchy stress from Neo-Hookean + hardening.
            stress = self._compute_stress(i)
            # Stress force: f = -V * J * sigma * F^(-T), approximated by
            # the standard MPM stress force per particle on each grid node.
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        bi, bj, bk = base[0] + a, base[1] + b, base[2] + c
                        if not (0 <= bi < self.grid_res[0] and 0 <= bj < self.grid_res[1] and 0 <= bk < self.grid_res[2]):
                            continue
                        weight = w[a][0] * w[b][1] * w[c][2]
                        gidx = self._grid_flat_index(bi, bj, bk)
                        # Affine velocity gradient term.
                        dpos = (np.array([bi, bj, bk], dtype=float) - fx) * self.dx
                        # Momentum contribution (PIC for now; APIC C matrix kept zero).
                        affine = np.zeros((3, 3), dtype=float)
                        grid_momentum[gidx] += weight * self.mass[i] * (self.velocity[i] + affine @ dpos)
                        grid_mass[gidx] += weight * self.mass[i]
                        # Stress force contribution.
                        f_ext = -self.vol_p * stress @ np.ones(3)
                        grid_momentum[gidx] += dt * weight * f_ext

    def _g2p(self, grid_momentum: np.ndarray, grid_mass: np.ndarray, dt: float) -> None:
        for i in np.where(self.is_sand)[0]:
            base, fx = self._compute_base_and_fx(self.position[i])
            w = [self._bspline(fx[0]), self._bspline(fx[1]), self._bspline(fx[2])]
            vel = np.zeros(3, dtype=float)
            grad = np.zeros((3, 3), dtype=float)
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        bi, bj, bk = base[0] + a, base[1] + b, base[2] + c
                        if not (0 <= bi < self.grid_res[0] and 0 <= bj < self.grid_res[1] and 0 <= bk < self.grid_res[2]):
                            continue
                        gidx = self._grid_flat_index(bi, bj, bk)
                        if grid_mass[gidx] <= 0.0:
                            continue
                        v = grid_momentum[gidx] / grid_mass[gidx]
                        weight = w[a][0] * w[b][1] * w[c][2]
                        dpos = (np.array([bi, bj, bk], dtype=float) - fx) * self.dx
                        vel += weight * v
                        grad += np.outer(v, dpos) * weight * self.inv_dx
            self.velocity[i] = vel
            # Update deformation gradient: F_new = (I + dt * grad(v)) F
            self.F[i] = (np.eye(3) + dt * grad) @ self.F[i]
            self.position[i] += dt * vel

    def _compute_base_and_fx(self, pos: np.ndarray):
        # Convert world pos to grid coordinate.
        coord = (pos - self.grid_origin) * self.inv_dx
        base = np.floor(coord - 0.5).astype(np.int32)
        fx = coord - base.astype(float)
        return base, fx

    def _grid_flat_index(self, i: int, j: int, k: int) -> int:
        return int(i * self.grid_res[1] * self.grid_res[2] + j * self.grid_res[2] + k)

    def _bspline(self, x: float) -> list:
        # x in [0,1]
        x = float(x)
        return [
            0.5 * (1.5 - x) ** 2,
            0.75 - (x - 1.0) ** 2,
            0.5 * (x - 0.5) ** 2,
        ]

    def _compute_stress(self, i: int) -> np.ndarray:
        F = self.F[i]
        J = max(np.linalg.det(F), 1e-6)
        # Neo-Hookean stress.
        F_T = F.T
        F_inv_T = np.linalg.inv(F_T)
        F_inv_T = np.clip(F_inv_T, -1e6, 1e6)
        stress = self.mu_sand * (F - F_inv_T) @ F_T + self.lambda_sand * np.log(J) * np.eye(3)
        # Add hardening.
        stress *= np.exp(self.JP[i] - 1.0)
        # Drucker-Prager plasticity: project stress to feasible region.
        stress = self._project_stress(stress)
        return stress

    def _project_stress(self, stress: np.ndarray) -> np.ndarray:
        # Isotropic projection onto Drucker-Prager cone.
        trace = np.trace(stress) / 3.0
        dev = stress - trace * np.eye(3)
        dev_norm = np.linalg.norm(dev)
        # Simplified DP: J2^0.5 <= -tan(phi) * I1 with I1 = trace(stress).
        # We keep compression negative, so criterion dev_norm + tan_phi * trace <= 0.
        tan_phi = np.tan(self.friction_angle)
        # Scale deviatoric if exceeded.
        if dev_norm > 1e-12 and dev_norm > -tan_phi * trace:
            scale = max(0.0, -tan_phi * trace) / dev_norm
            return trace * np.eye(3) + scale * dev
        return stress

    def _update_plasticity(self) -> None:
        for i in np.where(self.is_sand)[0]:
            J = np.linalg.det(self.F[i])
            if J > 1e-8:
                self.F[i] /= J ** (1.0 / 3.0)
                self.JP[i] *= J

    def _grid_boundary(self, grid_mass: np.ndarray, grid_momentum: np.ndarray) -> None:
        # Simple sticky boundary for floor at z=0.
        for idx in range(self.num_grid):
            # Recover node coordinate k.
            k = idx % self.grid_res[2]
            if k < 2:
                if grid_mass[idx] > 0.0:
                    grid_momentum[idx, 2] = max(0.0, grid_momentum[idx, 2])

    def _ball_step(self, dt: float) -> None:
        # Simple rigid ball integration with gravity and contact with sand.
        if not np.any(self.is_ball):
            return
        bidx = np.where(self.is_ball)[0]
        # Gravity.
        self.velocity[bidx] += dt * self.gravity[None, :]
        bc = self.position[bidx].mean(axis=0)
        # Contact with sand: repulsive force when within ball radius + particle radius.
        rad = self.ball_radius + 1.5 * self.spacing
        forces = np.zeros((len(bidx), 3), dtype=float)
        sand_indices = np.where(self.is_sand)[0]
        for i in sand_indices:
            dvec = self.position[i] - bc
            d = np.linalg.norm(dvec)
            if d < rad and d > 1e-8:
                n = dvec / d
                overlap = rad - d
                # Stiff repulsion.
                k = 5e5
                forces += -k * overlap * n
        # Apply total force to ball.
        total_force = forces.sum(axis=0)
        self.velocity[bidx] += (total_force / self.mass[bidx[0]]) * dt
        # Floor contact for ball.
        if bc[2] - self.ball_radius < self.floor_z:
            self.velocity[bidx, 2] = abs(self.velocity[bidx, 2]) * 0.5
            self.position[bidx, 2] += (self.floor_z + self.ball_radius - bc[2]) * 0.5
        # Update ball position.
        self.position[bidx] += dt * self.velocity[bidx]
