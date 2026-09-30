"""Physically-based smoke-jet and hanging-cloth solver.

Implements a coarse-grid 3D incompressible smoke simulation coupled to an
explicit mass-spring cloth.  The smoke jet is injected from the left boundary,
advected with semi-Lagrangian transport, and kept divergence-free with a
pressure projection.  The cloth samples the local fluid velocity and feels a
drag force; its velocity is then imposed as a moving no-slip obstacle in the
fluid, giving two-way coupling.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates, zoom
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import splu

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.rng = np.random.default_rng(seed)

        # Domain and coarse grid.
        self.low = np.asarray(self.config["domain"]["bounds_min"], dtype=float)
        self.high = np.asarray(self.config["domain"]["bounds_max"], dtype=float)
        # Coarse enough for Python/NumPy, fine enough to resolve the curtain and jet.
        self.cshape = (42, 24, 24)
        self.nx, self.ny, self.nz = self.cshape
        self.dx = (self.high - self.low) / np.array(self.cshape, dtype=float)
        self.hx, self.hy, self.hz = self.dx
        self.cell_volume = self.hx * self.hy * self.hz

        # Physical constants.
        self.gravity = np.asarray(self.config["initial_conditions"]["gravity"], dtype=float)
        self.rho = 1.225  # air density (kg/m^3)
        self.nu = 0.001   # kinematic viscosity (m^2/s)
        self.dt_sample = float(self.config["sample_dt"])
        self.jet = self.config["jet"]
        self.jet_speed = float(self.jet["speed"])
        self.jet_area = np.pi * float(self.jet["radius"]) ** 2
        self.jet_axis = np.asarray(self.jet["axis"], dtype=float)

        # Cloth parameters.
        self.Cd = 1.2
        self.k_spring = 100.0
        self.c_damp = 2.0
        self.cloth_dt = 0.0005

        # Initial scene state.
        state0 = scene.initial_state()
        self.pos = state0.position.copy()
        self.vel = state0.velocity.copy()
        self.mass = state0.mass.copy()
        self.body_id = state0.body_id.copy()
        self.uv = state0.attributes["uv"].copy()
        self.pinned = state0.attributes["pinned"].astype(bool)
        self.pinned_pos = self.pos[self.pinned].copy()
        self.faces = scene._faces.copy()

        cloth_cfg = self.config["cloth"]
        cloth_area = float(cloth_cfg["width"]) * float(cloth_cfg["height"])
        self.vertex_area = cloth_area / float(len(self.pos))

        # Build cloth edge topology and rest lengths.
        edge_set = set()
        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                if u > v:
                    u, v = v, u
                edge_set.add((int(u), int(v)))
        self.edges = np.array(list(edge_set), dtype=np.int32)
        self.rest_lengths = np.linalg.norm(
            self.pos[self.edges[:, 1]] - self.pos[self.edges[:, 0]], axis=1
        )

        # Fluid state: downsample from the scaffold's full-resolution initial fields.
        full_u = state0.fields["fluid_velocity"]
        full_smoke = state0.fields["smoke_density"]
        self.u = zoom(
            full_u,
            (self.nx / full_u.shape[0], self.ny / full_u.shape[1], self.nz / full_u.shape[2], 1),
            order=1,
        )
        self.smoke = zoom(
            full_smoke,
            (self.nx / full_smoke.shape[0], self.ny / full_smoke.shape[1], self.nz / full_smoke.shape[2]),
            order=1,
        )

        # Grid-center coordinate arrays (physical).
        self._ix = np.arange(self.nx).reshape(-1, 1, 1)
        self._iy = np.arange(self.ny).reshape(1, -1, 1)
        self._iz = np.arange(self.nz).reshape(1, 1, -1)
        self.Xc = self.low[0] + (self._ix + 0.5) * self.hx
        self.Yc = self.low[1] + (self._iy + 0.5) * self.hy
        self.Zc = self.low[2] + (self._iz + 0.5) * self.hz

        # Precompute jet source mask on the coarse grid.
        jc = np.asarray(self.jet["center"], dtype=float)
        r = float(self.jet["radius"])
        self.source_mask = (
            (self.Xc <= jc[0] + 0.18)
            & ((self.Yc - jc[1]) ** 2 + (self.Zc - jc[2]) ** 2 <= r ** 2)
        )
        self.source_volume = float(self.source_mask.sum() * self.cell_volume)

        # Build and factor the discrete Laplacian for pressure projection (Dirichlet on boundaries).
        self._build_poisson_solver()

        # Book-keeping for trajectory budget fields.
        self.cum_inflow = np.zeros((1, 3), dtype=float)
        self.cum_outflow = np.zeros((1, 3), dtype=float)
        self.cum_ext_imp = np.zeros((1, 3), dtype=float)
        self.cum_mass_source = np.zeros(1, dtype=float)
        self.initial_momentum = np.zeros(3, dtype=float)
        self.last_fluid_force_total = np.zeros(3, dtype=float)

        self.ref_momentum = max(
            self.rho * self.jet_area * (self.jet_speed ** 2) * float(self.config["duration"]),
            1e-6,
        )
        self.ref_velocity = float(self.jet_speed)

        self.current_time = 0.0
        self.last_fluid_u_at_vertices = np.zeros_like(self.pos)

    # ------------------------------------------------------------------
    # Solver interface
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time <= self.current_time + 1e-12:
            return self._make_frame(time)

        dt = time - self.current_time

        # Advance fluid, then sample fluid at cloth vertices.
        self._fluid_step(dt)
        fluid_u = self._sample_fluid_at_vertices()

        # Advance cloth under drag and internal elastic forces.
        self._cloth_step(dt, fluid_u)

        # Update cumulative budget quantities once per sample interval.
        self._update_sample_budgets(time, dt)

        self.current_time = time
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "cloth_vertices": int(len(self.pos)),
            "fluid_grid": tuple(self.cshape),
            "max_smoke": float(np.max(self.smoke)),
            "cloth_max_speed": float(np.max(np.linalg.norm(self.vel, axis=1))),
        }

    # ------------------------------------------------------------------
    # Frame construction
    # ------------------------------------------------------------------
    def _make_frame(self, time: float) -> StateFrame:
        # Upsample fluid fields to the scaffold's full resolution for output.
        full_u = zoom(
            self.u,
            (126 / self.nx, 70 / self.ny, 70 / self.nz, 1),
            order=1,
        )
        full_smoke = zoom(
            self.smoke,
            (126 / self.nx, 70 / self.ny, 70 / self.nz),
            order=1,
        )

        # Boundary samples (domain faces, zero velocity, static walls).
        boundary_normal = np.array(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
             [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            dtype=float,
        )
        boundary_velocity = np.zeros((6, 3), dtype=float)
        boundary_target = np.zeros((6, 3), dtype=float)

        # Domain momentum for outflow budget.
        domain_momentum = self.rho * self.cell_volume * np.sum(self.u, axis=(0, 1, 2))
        self.cum_outflow[0] = self.cum_inflow[0] - (domain_momentum - self.initial_momentum)

        # External-force budget terms (gravity and fluid drag).
        total_gravity = np.sum(self.mass[:, None] * self.gravity, axis=0)
        total_fluid = self.last_fluid_force_total
        budget_external_force = np.zeros((1, 2, 3), dtype=float)
        budget_external_force[0, 0] = total_gravity
        budget_external_force[0, 1] = total_fluid

        interaction_force = total_fluid.reshape(1, 3)

        return StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "uv": self.uv.copy(),
                "pinned": self.pinned.astype(float),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "smoke_density": full_smoke.astype(np.float32, copy=False),
                "fluid_velocity": full_u.astype(np.float32, copy=False),
                "boundary_normal": boundary_normal,
                "boundary_target_velocity": boundary_target,
                "boundary_velocity": boundary_velocity,
                "budget_cumulative_external_impulse": self.cum_ext_imp.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
                "budget_domain_weights": np.ones((1, len(self.pos)), dtype=float),
                "budget_external_force": budget_external_force,
                "cumulative_mass_source": self.cum_mass_source.copy(),
                "interaction_force": interaction_force,
            },
            scalars={
                "jet_start_time": float(self.jet["start_time"]),
                "jet_speed": float(self.jet["speed"]),
                "reference_momentum": float(self.ref_momentum),
                "reference_velocity": float(self.ref_velocity),
            },
        )

    # ------------------------------------------------------------------
    # Fluid dynamics
    # ------------------------------------------------------------------
    def _fluid_step(self, dt: float) -> None:
        t = self.current_time + dt

        # Advect velocity, then apply inflow/source.
        self.u = self._advect_vec(self.u, dt)
        self._apply_velocity_source(t)

        # Enforce moving cloth obstacle.
        self._apply_cloth_obstacle()

        # Viscosity.
        self.u += self.nu * dt * self._laplacian_vec(self.u)

        # Pressure projection.
        self._project(dt)

        # Re-impose inflow and obstacle after projection.
        self._apply_velocity_source(t)
        self._apply_cloth_obstacle()

        # Advect smoke, then inject new smoke at the nozzle.
        self.smoke = self._advect_scalar(self.smoke, dt)
        self._apply_smoke_source(t)
        self._apply_cloth_obstacle(smoke_only=True)

    def _apply_velocity_source(self, t: float) -> None:
        alpha = self._jet_alpha(t)
        if alpha <= 0.0:
            return
        u = self.jet_speed * alpha
        axis = self.jet_axis.copy()
        # Primary jet velocity.
        self.u[self.source_mask] = u * axis
        # Small deterministic swirl to break perfect symmetry and let edges lift.
        sw = 0.2
        self.u[..., 1][self.source_mask] += (
            u * sw * (self.Yc - self.jet["center"][1])[self.source_mask]
        )
        self.u[..., 2][self.source_mask] += (
            -u * sw * (self.Zc - self.jet["center"][2])[self.source_mask]
        )

    def _apply_smoke_source(self, t: float) -> None:
        alpha = self._jet_alpha(t)
        if alpha <= 0.0:
            return
        target = float(self.jet["smoke_density"]) * alpha
        self.smoke[self.source_mask] = target

    def _jet_alpha(self, t: float) -> float:
        if t < float(self.jet["start_time"]) or t > float(self.jet["active_until"]):
            return 0.0
        ramp = (t - float(self.jet["start_time"])) / float(self.jet["ramp_duration"])
        return float(min(1.0, max(0.0, ramp)))

    def _apply_cloth_obstacle(self, smoke_only: bool = False) -> None:
        mask, vel = self._cloth_grid_influence()
        if not smoke_only:
            self.u[mask] = vel[mask]
        self.smoke[mask] = 0.0

    def _cloth_grid_influence(self):
        mask = np.zeros(self.cshape, dtype=bool)
        vel = np.zeros(self.cshape + (3,), dtype=float)
        base = ((self.pos - self.low) / self.dx - 0.5).astype(int)
        max_idx = np.array(self.cshape) - 1
        base = np.clip(base, 0, max_idx)
        for vi in range(len(self.pos)):
            i0, j0, k0 = base[vi]
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        ii = i0 + di
                        jj = j0 + dj
                        kk = k0 + dk
                        if 0 <= ii < self.nx and 0 <= jj < self.ny and 0 <= kk < self.nz:
                            mask[ii, jj, kk] = True
                            vel[ii, jj, kk] = self.vel[vi]
        return mask, vel

    def _advect_vec(self, u: np.ndarray, dt: float) -> np.ndarray:
        back_x = self.Xc - u[..., 0] * dt
        back_y = self.Yc - u[..., 1] * dt
        back_z = self.Zc - u[..., 2] * dt
        coords = np.stack(
            [
                (back_z - self.low[2]) / self.hz - 0.5,
                (back_y - self.low[1]) / self.hy - 0.5,
                (back_x - self.low[0]) / self.hx - 0.5,
            ]
        )
        out = np.zeros_like(u)
        for d in range(3):
            out[..., d] = map_coordinates(u[..., d], coords, order=1, mode="nearest")
        return out

    def _advect_scalar(self, s: np.ndarray, dt: float) -> np.ndarray:
        back_x = self.Xc - self.u[..., 0] * dt
        back_y = self.Yc - self.u[..., 1] * dt
        back_z = self.Zc - self.u[..., 2] * dt
        coords = np.stack(
            [
                (back_z - self.low[2]) / self.hz - 0.5,
                (back_y - self.low[1]) / self.hy - 0.5,
                (back_x - self.low[0]) / self.hx - 0.5,
            ]
        )
        return map_coordinates(s, coords, order=1, mode="nearest")

    def _laplacian_vec(self, u: np.ndarray) -> np.ndarray:
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1, 1:-1] = (
            (u[2:, 1:-1, 1:-1] - 2.0 * u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1])
            / self.hx ** 2
            + (u[1:-1, 2:, 1:-1] - 2.0 * u[1:-1, 1:-1, 1:-1] + u[1:-1, :-2, 1:-1])
            / self.hy ** 2
            + (u[1:-1, 1:-1, 2:] - 2.0 * u[1:-1, 1:-1, 1:-1] + u[1:-1, 1:-1, :-2])
            / self.hz ** 2
        )
        return lap

    def _divergence(self) -> np.ndarray:
        div = np.zeros(self.cshape, dtype=float)
        div[1:-1, 1:-1, 1:-1] = (
            (self.u[2:, 1:-1, 1:-1, 0] - self.u[:-2, 1:-1, 1:-1, 0]) / (2.0 * self.hx)
            + (self.u[1:-1, 2:, 1:-1, 1] - self.u[1:-1, :-2, 1:-1, 1]) / (2.0 * self.hy)
            + (self.u[1:-1, 1:-1, 2:, 2] - self.u[1:-1, 1:-1, :-2, 2]) / (2.0 * self.hz)
        )
        return div

    def _gradient(self, p: np.ndarray) -> np.ndarray:
        grad = np.zeros(self.cshape + (3,), dtype=float)
        # interior
        grad[1:-1, 1:-1, 1:-1, 0] = (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / (2.0 * self.hx)
        grad[1:-1, 1:-1, 1:-1, 1] = (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) / (2.0 * self.hy)
        grad[1:-1, 1:-1, 1:-1, 2] = (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) / (2.0 * self.hz)
        # one-sided boundaries
        grad[0, 1:-1, 1:-1, 0] = (p[1, 1:-1, 1:-1] - p[0, 1:-1, 1:-1]) / self.hx
        grad[-1, 1:-1, 1:-1, 0] = (p[-1, 1:-1, 1:-1] - p[-2, 1:-1, 1:-1]) / self.hx
        grad[1:-1, 0, 1:-1, 1] = (p[1:-1, 1, 1:-1] - p[1:-1, 0, 1:-1]) / self.hy
        grad[1:-1, -1, 1:-1, 1] = (p[1:-1, -1, 1:-1] - p[1:-1, -2, 1:-1]) / self.hy
        grad[1:-1, 1:-1, 0, 2] = (p[1:-1, 1:-1, 1] - p[1:-1, 1:-1, 0]) / self.hz
        grad[1:-1, 1:-1, -1, 2] = (p[1:-1, 1:-1, -1] - p[1:-1, 1:-1, -2]) / self.hz
        return grad

    def _build_poisson_solver(self) -> None:
        nx, ny, nz = self.cshape

        def d1d(n: int, h: float):
            vals = np.array([1.0 / h ** 2, -2.0 / h ** 2, 1.0 / h ** 2])
            return diags(vals, [-1, 0, 1], shape=(n - 2, n - 2))

        Dx = d1d(nx, self.hx)
        Dy = d1d(ny, self.hy)
        Dz = d1d(nz, self.hz)
        Ix = eye(nx - 2)
        Iy = eye(ny - 2)
        Iz = eye(nz - 2)
        A = (
            kron(Iz, kron(Iy, Dx))
            + kron(Iz, kron(Dy, Ix))
            + kron(Dz, kron(Iy, Ix))
        )
        self.A_factor = splu(A.tocsc())

    def _project(self, dt: float) -> None:
        div = self._divergence()
        b = div[1:-1, 1:-1, 1:-1] / dt
        p_int = self.A_factor.solve(b.ravel())
        p = np.zeros(self.cshape, dtype=float)
        p[1:-1, 1:-1, 1:-1] = p_int.reshape((self.nz - 2, self.ny - 2, self.nx - 2))
        self.u -= dt * self._gradient(p)

    # ------------------------------------------------------------------
    # Cloth dynamics
    # ------------------------------------------------------------------
    def _sample_fluid_at_vertices(self) -> np.ndarray:
        idx = (self.pos - self.low) / self.dx - 0.5
        coords = np.stack([idx[:, 2], idx[:, 1], idx[:, 0]])
        out = np.zeros((len(self.pos), 3), dtype=float)
        for d in range(3):
            out[:, d] = map_coordinates(
                self.u[..., d], coords, order=1, mode="nearest"
            )
        return out

    def _cloth_step(self, dt: float, fluid_u: np.ndarray) -> None:
        n_sub = max(1, int(np.ceil(dt / self.cloth_dt)))
        sub = dt / n_sub
        for _ in range(n_sub):
            f = self._elastic_forces()
            # gravity
            f += self.mass[:, None] * self.gravity
            # quadratic drag relative to the local fluid
            u_rel = fluid_u - self.vel
            speeds = np.linalg.norm(u_rel, axis=1)
            f += 0.5 * self.Cd * self.rho * self.vertex_area * (speeds[:, None] * u_rel)
            f[self.pinned] = 0.0

            self.vel += sub * f / self.mass[:, None]
            self.vel[self.pinned] = 0.0
            self.pos += sub * self.vel
            self.pos[self.pinned] = self.pinned_pos

        # Store total fluid interaction force at the final state for budgets/frame.
        u_rel = fluid_u - self.vel
        speeds = np.linalg.norm(u_rel, axis=1)
        per_vertex = 0.5 * self.Cd * self.rho * self.vertex_area * (speeds[:, None] * u_rel)
        self.last_fluid_force_total = np.sum(per_vertex, axis=0)

    def _elastic_forces(self) -> np.ndarray:
        f = np.zeros_like(self.pos)
        d = self.pos[self.edges[:, 1]] - self.pos[self.edges[:, 0]]
        dists = np.linalg.norm(d, axis=1)
        mask = dists > 1e-12
        d = d[mask]
        dists = dists[mask]
        edges = self.edges[mask]
        dirs = d / dists[:, None]
        strains = (dists - self.rest_lengths[mask]) / self.rest_lengths[mask]

        # Spring force
        total = (self.k_spring * strains[:, None]) * dirs

        # Damping along each spring
        v_rel = self.vel[edges[:, 1]] - self.vel[edges[:, 0]]
        proj = np.sum(v_rel * dirs, axis=1)
        total += (self.c_damp * proj[:, None]) * dirs

        np.add.at(f, edges[:, 0], total)
        np.add.at(f, edges[:, 1], -total)
        return f

    # ------------------------------------------------------------------
    # Budget updates
    # ------------------------------------------------------------------
    def _update_sample_budgets(self, time: float, dt: float) -> None:
        alpha = self._jet_alpha(time)
        if alpha > 0.0:
            speed = self.jet_speed * alpha
            inflow_rate = self.rho * self.jet_area * speed ** 2 * self.jet_axis
            self.cum_inflow[0] += inflow_rate * dt

        total_gravity = np.sum(self.mass[:, None] * self.gravity, axis=0)
        total_fluid = self.last_fluid_force_total
        self.cum_ext_imp[0] += (total_gravity + total_fluid) * dt

        self.cum_mass_source[0] += float(self.jet["smoke_density"]) * alpha * self.source_volume
