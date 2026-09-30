"""Particle-based porous-cloth buoyancy solver.

Uses a particle system for both water and cloth.  Water is evolved with
weakly-compressible SPH pressure and artificial viscosity.  Each cloth patch is
treated as a mass-spring shell with structural, shear and bending springs.
Cloth particles feel gravity, a net buoyancy force based on local submergence,
and drag from the surrounding water.  The equal-and-opposite drag acts on the
water, giving two-way momentum exchange.  Tank walls/floor apply repulsive
boundary forces on all particles.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.frame0 = scene.initial_state()
        self.time = 0.0

        # Load configuration for physical constants.
        import yaml

        cfg_path = Path("config.yaml")
        if cfg_path.is_file():
            with cfg_path.open("r") as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}

        # Copy mutable state from the initial frame.
        self.position = self.frame0.position.copy()
        self.velocity = self.frame0.velocity.copy()
        self.mass = self.frame0.mass.copy()
        self.body_id = self.frame0.body_id.copy()
        self.attributes = {
            k: np.asarray(v).copy() for k, v in self.frame0.attributes.items()
        }

        # Phase masks.
        self.water_mask = self.body_id == 0
        self.cloth_mask = self.body_id > 0
        self.n = len(self.mass)

        # Per-cloth body masks.
        self.body_ids = np.unique(self.body_id[self.body_id > 0])
        self.body_ids.sort()

        # Gravity and water properties.
        g = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.gravity = g
        self.water_density = float(self.config.get("water", {}).get("density", 1000.0))
        self.free_surface_z = float(
            self.config.get("water", {}).get("free_surface_z", 2.15)
        )

        # Tank interior bounds (m).
        tank = self.config.get("tank", {})
        bmin = np.asarray(tank.get("interior_bounds_min", [-2.35, -1.0, 0.0]), dtype=float)
        bmax = np.asarray(tank.get("interior_bounds_max", [2.35, 1.0, 2.45]), dtype=float)
        self.tank_min = bmin
        self.tank_max = bmax
        self.boundary_margin = 0.08
        self.k_boundary = 8.0e3

        # Cloth spring network.
        cloth_cfg = self.config.get("cloth", {})
        self.nx, self.ny = map(int, cloth_cfg.get("grid_shape", [9, 7]))
        self.structural_k = 8.0e2
        self.shear_k = 4.0e2
        self.bending_k = 1.5e2
        self._build_cloth_springs()

        # SPH parameters.
        self.smooth_h = 0.28
        self.h = self.smooth_h
        self.alpha_kernel = 1.0 / (math.pi * self.h ** 3)
        self.beta_kernel = 1.0 / (math.pi * self.h ** 4)
        self.cs = 25.0
        self.gamma = 7.0
        self.B_sph = self.cs ** 2 * self.water_density / self.gamma
        self.visc = 0.015
        self.drag_coeff = 6.0
        self.drag_cutoff = 2.0 * self.h

        # Compute reference density from the initial configuration so that SPH
        # pressure vanishes at the released rest state.
        self.rho0 = self._compute_initial_density()

        # Budget domain bookkeeping: 0=water, 1=light, 2=neutral, 3=heavy.
        self.G = 4
        self.domain_masks = [self.water_mask]
        for bid in (1, 2, 3):
            self.domain_masks.append(self.body_id == bid)
        self.domain_weights = np.stack([m.astype(float) for m in self.domain_masks], axis=0)

        # Cumulative budget fields.
        self.cum_external_impulse = np.zeros((self.G, 3), dtype=float)
        self.cum_inflow_momentum = np.zeros((self.G, 3), dtype=float)
        self.cum_outflow_momentum = np.zeros((self.G, 3), dtype=float)
        self.cum_mass_source = np.zeros(2, dtype=float)

        # Per-term external force: [G, K, 3]; K=3 -> gravity, buoyancy, boundary.
        self.external_force_terms = np.zeros((self.G, 3, 3), dtype=float)
        self.interaction_force = np.zeros((self.G, 3), dtype=float)

        # Reference momentum scale: total mass * characteristic velocity.
        char_speed = math.sqrt(9.81 * max(1.0, self.free_surface_z))
        self.reference_momentum = float(np.sum(self.mass) * char_speed)
        if self.reference_momentum <= 0.0:
            self.reference_momentum = 1.0

        # Sub-stepping.
        self.sample_dt = float(self.config.get("sample_dt", 0.02))
        self.substeps = max(1, int(round(self.sample_dt / 0.002)))
        self.dt = self.sample_dt / self.substeps

        # Compute initial budget / interaction values for the first frame.
        self._update_budget_forces()

    def _build_cloth_springs(self) -> None:
        """Precompute cloth spring edges and per-edge parameters."""
        edges_i: list[int] = []
        edges_j: list[int] = []
        k_list: list[float] = []
        nx, ny = self.nx, self.ny
        n_per = nx * ny
        for b, _bid in enumerate((1, 2, 3), start=0):
            offset = b * n_per
            for iy in range(ny):
                for ix in range(nx):
                    idx = offset + iy * nx + ix
                    # Structural neighbours.
                    if ix < nx - 1:
                        edges_i.append(idx)
                        edges_j.append(idx + 1)
                        k_list.append(self.structural_k)
                    if iy < ny - 1:
                        edges_i.append(idx)
                        edges_j.append(idx + nx)
                        k_list.append(self.structural_k)
                    # Shear.
                    if ix < nx - 1 and iy < ny - 1:
                        edges_i.append(idx)
                        edges_j.append(idx + nx + 1)
                        k_list.append(self.shear_k)
                        edges_i.append(idx + 1)
                        edges_j.append(idx + nx)
                        k_list.append(self.shear_k)
                    # Bending (skip-one).
                    if ix < nx - 2:
                        edges_i.append(idx)
                        edges_j.append(idx + 2)
                        k_list.append(self.bending_k)
                    if iy < ny - 2:
                        edges_i.append(idx)
                        edges_j.append(idx + 2 * nx)
                        k_list.append(self.bending_k)
        self.edges_i = np.asarray(edges_i, dtype=np.int64)
        self.edges_j = np.asarray(edges_j, dtype=np.int64)
        self.edges_k = np.asarray(k_list, dtype=float)

        pos = self.position
        r_vec = pos[self.edges_j] - pos[self.edges_i]
        self.edges_rest = np.linalg.norm(r_vec, axis=1)

        m_i = self.mass[self.edges_i]
        m_j = self.mass[self.edges_j]
        m_eff = m_i * m_j / (m_i + m_j)
        # 20% of critical damping for the edge.
        self.edges_damp = 0.20 * 2.0 * np.sqrt(self.edges_k * m_eff)
        self.edges_damp = np.clip(self.edges_damp, 0.5, 20.0)

    def _compute_initial_density(self) -> float:
        """Average SPH density at t=0, used as the reference density."""
        pos = self.position
        tree = cKDTree(pos)
        pairs = tree.sparse_distance_matrix(tree, 2.0 * self.h, output_type="ndarray")
        i = pairs["i"]
        j = pairs["j"]
        r = pairs["v"]
        q = r / self.h
        f = np.where(q <= 1.0, 1.0 - 1.5 * q ** 2 + 0.75 * q ** 3, 0.0)
        f = np.where((q > 1.0) & (q <= 2.0), 0.25 * (2.0 - q) ** 3, f)
        w = self.alpha_kernel * f
        # Self contribution.
        rho = self.mass * self.alpha_kernel
        rho += np.bincount(i.astype(int), weights=self.mass[j] * w, minlength=self.n)
        rho += np.bincount(j.astype(int), weights=self.mass[i] * w, minlength=self.n)
        return float(np.mean(rho))

    def _sph_kernel_pair(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return W(r) and dW/dr for an array of distances (cubic spline, 3D)."""
        q = r / self.h
        f = np.zeros_like(q)
        df = np.zeros_like(q)
        mask1 = q <= 1.0
        mask2 = (q > 1.0) & (q <= 2.0)
        f[mask1] = 1.0 - 1.5 * q[mask1] ** 2 + 0.75 * q[mask1] ** 3
        df[mask1] = -3.0 * q[mask1] + 2.25 * q[mask1] ** 2
        f[mask2] = 0.25 * (2.0 - q[mask2]) ** 3
        df[mask2] = -0.75 * (2.0 - q[mask2]) ** 2
        W = self.alpha_kernel * f
        dWdr = self.beta_kernel * df
        return W, dWdr

    def _spring_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Accumulate cloth spring/damping forces into a per-particle array."""
        f = np.zeros_like(pos)
        if self.edges_i.size == 0:
            return f
        r_vec = pos[self.edges_j] - pos[self.edges_i]
        dist = np.linalg.norm(r_vec, axis=1)
        # Guard against degenerate edges.
        dist = np.where(dist < 1e-9, 1.0, dist)
        dir_vec = r_vec / dist[:, None]
        delta = dist - self.edges_rest
        v_rel = vel[self.edges_j] - vel[self.edges_i]
        v_proj = np.sum(v_rel * dir_vec, axis=1)
        f_mag = self.edges_k * delta + self.edges_damp * v_proj
        force = f_mag[:, None] * dir_vec
        np.add.at(f, self.edges_i, force)
        np.add.at(f, self.edges_j, -force)
        return f

    def _sph_forces(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return SPH pressure+viscous acceleration and per-water-particle density."""
        acc = np.zeros_like(pos)
        if not np.any(self.water_mask):
            return acc, np.zeros(self.n)

        tree = cKDTree(pos)
        pairs = tree.sparse_distance_matrix(tree, 2.0 * self.h, output_type="ndarray")
        if pairs.size == 0:
            return acc, np.full(self.n, self.rho0)

        i = pairs["i"].astype(np.int64)
        j = pairs["j"].astype(np.int64)
        r = pairs["v"]

        W, dWdr = self._sph_kernel_pair(r)
        # Density.
        rho = self.mass * self.alpha_kernel
        rho += np.bincount(i, weights=self.mass[j] * W, minlength=self.n)
        rho += np.bincount(j, weights=self.mass[i] * W, minlength=self.n)
        rho = np.maximum(rho, 1e-3)

        # Pressure.
        P = self.B_sph * ((rho / self.rho0) ** self.gamma - 1.0)
        P = np.maximum(P, -self.B_sph)  # allow mild tension, avoid huge negative values

        # Direction from i to j.
        r_vec = pos[j] - pos[i]
        r_safe = np.where(r < 1e-9, 1.0, r)[:, None]
        r_vec_norm = r_vec / r_safe
        grad_W = dWdr[:, None] * r_vec_norm

        # Pairwise pressure acceleration on i.
        pi = P[i]
        pj = P[j]
        rho_i = rho[i]
        rho_j = rho[j]
        m_j = self.mass[j]
        term = -(m_j / self.mass[i]) * (pi / (rho_i ** 2) + pj / (rho_j ** 2))[:, None] * grad_W
        # Symmetric on j.
        np.add.at(acc, i, term)
        np.add.at(acc, j, -term)

        # Artificial viscosity (pairwise, momentum conserving).
        v_rel = vel[j] - vel[i]
        visc_acc = self.visc * v_rel * W[:, None]
        visc_acc_i = visc_acc
        visc_acc_j = -visc_acc
        np.add.at(acc, i, visc_acc_i)
        np.add.at(acc, j, visc_acc_j)

        # Zero acceleration for non-water particles from SPH.
        acc[~self.water_mask] = 0.0
        return acc, rho

    def _drag_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Return drag acceleration on each particle due to cloth-water coupling."""
        acc = np.zeros_like(vel)
        if not np.any(self.water_mask) or not np.any(self.cloth_mask):
            return acc

        water_idx = np.nonzero(self.water_mask)[0]
        cloth_idx = np.nonzero(self.cloth_mask)[0]
        if water_idx.size == 0 or cloth_idx.size == 0:
            return acc

        tree_water = cKDTree(pos[water_idx])
        tree_cloth = cKDTree(pos[cloth_idx])
        # Query each water point against cloth points within cutoff.
        neighbours = tree_water.query_ball_tree(tree_cloth, self.drag_cutoff)

        for wi_local, c_local_list in enumerate(neighbours):
            if not c_local_list:
                continue
            wi = water_idx[wi_local]
            for ci_local in c_local_list:
                ci = cloth_idx[ci_local]
                r_vec = pos[ci] - pos[wi]
                r = float(np.linalg.norm(r_vec))
                if r > self.drag_cutoff or r < 1e-12:
                    continue
                q = r / self.drag_cutoff
                # Smooth kernel (cubic-like).
                w = 1.0 - 3.0 * q ** 2 + 2.0 * q ** 3
                v_rel = vel[ci] - vel[wi]
                f = self.drag_coeff * w * v_rel
                acc[wi] += f / self.mass[wi]
                acc[ci] -= f / self.mass[ci]
        return acc

    def _boundary_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Repulsive spring force for particles near tank walls/floor."""
        f = np.zeros_like(pos)
        margin = self.boundary_margin
        # x walls.
        d_left = pos[:, 0] - (self.tank_min[0] + margin)
        mask = d_left < 0.0
        f[mask, 0] += self.k_boundary * (-d_left[mask])
        d_right = (self.tank_max[0] - margin) - pos[:, 0]
        mask = d_right < 0.0
        f[mask, 0] -= self.k_boundary * (-d_right[mask])
        # y walls.
        d_back = pos[:, 1] - (self.tank_min[1] + margin)
        mask = d_back < 0.0
        f[mask, 1] += self.k_boundary * (-d_back[mask])
        d_front = (self.tank_max[1] - margin) - pos[:, 1]
        mask = d_front < 0.0
        f[mask, 1] -= self.k_boundary * (-d_front[mask])
        # z floor only (open top).
        d_floor = pos[:, 2] - (self.tank_min[2] + margin)
        mask = d_floor < 0.0
        f[mask, 2] += self.k_boundary * (-d_floor[mask])
        return f

    def _buoyancy_forces(self, pos: np.ndarray) -> np.ndarray:
        """Net gravity-buoyancy force on cloth particles only."""
        f = np.zeros_like(pos)
        if not np.any(self.cloth_mask):
            return f
        z = pos[self.cloth_mask, 2]
        # Smooth submergence factor based on distance below free surface.
        depth = self.free_surface_z - z
        s = np.clip(depth / 0.12 + 0.5, 0.0, 1.0)
        m_c = self.mass[self.cloth_mask]
        rho_c = self.attributes["material_density"][self.cloth_mask]
        factor = 1.0 - (self.water_density / rho_c) * s
        f[self.cloth_mask] = (factor * m_c)[:, None] * self.gravity[None, :]
        return f

    def _gravity_forces(self, pos: np.ndarray) -> np.ndarray:
        """Plain gravity on cloth particles only (water is neutrally buoyant)."""
        f = np.zeros_like(pos)
        if np.any(self.cloth_mask):
            f[self.cloth_mask] = self.mass[self.cloth_mask][:, None] * self.gravity[None, :]
        return f

    def _update_budget_forces(self) -> None:
        """Recompute external-force terms and interaction force for current state."""
        pos = self.position
        vel = self.velocity

        # Plain gravity on cloth.
        fg = self._gravity_forces(pos)
        # Buoyancy (net) on cloth.
        fb = self._buoyancy_forces(pos)
        # Boundary on all particles.
        fbnd = self._boundary_forces(pos, vel)

        # Per-domain per-term external force [G, K, 3].
        self.external_force_terms = np.zeros((self.G, 3, 3), dtype=float)
        for g in range(self.G):
            mask = self.domain_masks[g]
            if not np.any(mask):
                continue
            self.external_force_terms[g, 0] = np.sum(fg[mask], axis=0)
            self.external_force_terms[g, 1] = np.sum(fb[mask], axis=0)
            self.external_force_terms[g, 2] = np.sum(fbnd[mask], axis=0)

        # Interaction force: drag coupling on each domain.
        drag_acc = self._drag_forces(pos, vel)
        drag_f = drag_acc * self.mass[:, None]
        self.interaction_force = np.zeros((self.G, 3), dtype=float)
        for g in range(self.G):
            mask = self.domain_masks[g]
            if np.any(mask):
                self.interaction_force[g] = np.sum(drag_f[mask], axis=0)

    def _step(self, dt: float) -> None:
        pos = self.position
        vel = self.velocity

        # Accelerations.
        # Cloth springs.
        f_spring = self._spring_forces(pos, vel)
        # SPH for water.
        a_sph, _rho = self._sph_forces(pos, vel)
        # Gravity/buoyancy on cloth.
        f_grav = self._gravity_forces(pos)
        f_buoy = self._buoyancy_forces(pos)
        # Drag coupling.
        a_drag = self._drag_forces(pos, vel)
        # Boundary.
        f_bnd = self._boundary_forces(pos, vel)

        # Total acceleration.
        acc = np.zeros_like(pos)
        acc += f_spring / self.mass[:, None]
        acc += a_sph
        acc += f_grav / self.mass[:, None]
        acc += f_buoy / self.mass[:, None]
        acc += a_drag
        acc += f_bnd / self.mass[:, None]

        # Integrate (symplectic Euler).
        vel_new = vel + dt * acc
        pos_new = pos + dt * vel_new

        # Clamp tiny numerical drifts.
        if not np.isfinite(acc).all():
            raise RuntimeError("Non-finite acceleration encountered")
        if not np.isfinite(vel_new).all() or not np.isfinite(pos_new).all():
            raise RuntimeError("Non-finite state encountered after integration")

        self.velocity = vel_new
        self.position = pos_new

        # Update budget accumulators.
        self._update_budget_forces()
        total_external = np.sum(self.external_force_terms, axis=1)  # [G,3]
        self.cum_external_impulse += total_external * dt

    def advance_to(self, time: float) -> StateFrame:
        if time <= self.time:
            return self._make_frame(self.time)

        # Integrate with fixed substeps until reaching the requested sample time.
        remaining = time - self.time
        n_steps = int(round(remaining / self.dt))
        # Guard against rounding drift: ensure we land exactly on sample times.
        if abs(n_steps * self.dt - remaining) > 1e-12:
            n_steps = int(np.ceil(remaining / self.dt))
        for _ in range(n_steps):
            self._step(self.dt)
        self.time = time
        return self._make_frame(self.time)

    def _make_frame(self, time: float) -> StateFrame:
        # Per-sample mass quantity split by material (water / cloth).
        mass_quantity = np.zeros((self.n, 2), dtype=float)
        mass_quantity[self.water_mask, 0] = self.mass[self.water_mask]
        mass_quantity[self.cloth_mask, 1] = self.mass[self.cloth_mask]

        # Copy current attributes and augment.
        attrs = {k: np.asarray(v).copy() for k, v in self.attributes.items()}
        attrs["mass_quantity"] = mass_quantity

        fields = {
            "budget_cumulative_external_impulse": self.cum_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow_momentum.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow_momentum.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": self.external_force_terms.copy(),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "interaction_force": self.interaction_force.copy(),
        }

        scalars = dict(self.frame0.scalars)
        scalars["reference_momentum"] = float(self.reference_momentum)

        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "dt": self.dt,
            "substeps_per_sample": self.substeps,
            "reference_density": float(self.rho0),
            "particle_count": self.n,
        }
