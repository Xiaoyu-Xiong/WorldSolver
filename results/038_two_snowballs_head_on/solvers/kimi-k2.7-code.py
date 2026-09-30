"""Vectorized explicit MPM solver for two snowballs colliding head-on.

The runner calls::

    solver.initialize(scene, seed)
    frames = [solver.advance_to(float(t)) for t in scene.sample_times()]

``advance_to`` returns a ``worldsolver.contracts.StateFrame`` containing the
physical state and all required raw trajectory fields.  The implementation is a
lightweight Material Point Method with quadratic B-spline kernels, explicit
velocity update, and a sticky floor boundary condition.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self._scene = scene
        self._config = scene.config
        self._dt = float(self._config["sample_dt"])
        self._gravity = np.asarray(self._config["gravity"], dtype=float)
        self._floor_top = float(self._config["floor"]["top_z"])

        init = scene.initial_state()
        self._position: NDArray[np.float64] = np.asarray(init.position, dtype=float).copy()
        self._velocity: NDArray[np.float64] = np.asarray(init.velocity, dtype=float).copy()
        self._mass: NDArray[np.float64] = np.asarray(init.mass, dtype=float).copy()
        self._body_id: NDArray[np.int64] = np.asarray(init.body_id, dtype=np.int64).copy()
        self._F: NDArray[np.float64] = np.asarray(
            init.attributes["deformation_gradient"], dtype=float
        ).copy()
        self._volume0: NDArray[np.float64] = np.asarray(
            init.attributes["reference_volume"], dtype=float
        ).copy()
        self._mass_quantity: NDArray[np.float64] = np.asarray(
            init.attributes["mass_quantity"], dtype=float
        ).copy()
        self._n = self._position.shape[0]

        # MPM grid settings
        self._dx = 0.08
        # Domain large enough to contain the collision plus margin.
        self._grid_min = np.array([-2.5, -1.5, -0.45])
        self._grid_max = np.array([2.5, 1.5, 2.5])
        self._grid_shape = tuple(
            np.maximum(np.ceil((self._grid_max - self._grid_min) / self._dx).astype(int), 1)
        )
        self._nx, self._ny, self._nz = self._grid_shape
        self._n_nodes = self._nx * self._ny * self._nz

        # Material parameters for snow-like behaviour.
        # Young's modulus and Poisson ratio give the Lamé parameters.
        E = 1.5e5
        nu = 0.25
        self._mu = E / (2.0 * (1.0 + nu))
        self._lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self._damping = 0.3
        self._theta_c = 2.5e-2
        self._theta_s = 7.5e-3
        self._min_F = 0.45
        self._max_F = 1.35

        # Time step chosen to satisfy a CFL-like condition for the grid.
        c = (E / 400.0) ** 0.5
        self._sub_dt = min(0.0005, 0.3 * self._dx / max(c, 1.0))

        # Precompute floor-node mask: any grid node whose centre is at or below the floor.
        node_z = self._grid_min[2] + (np.arange(self._nx)[:, None, None] + 0.5) * self._dx
        node_z = node_z.reshape(-1)
        self._floor_node_mask = node_z <= self._floor_top

        self._last_time = 0.0
        self._last_frame = init
        self._substeps = 0

    # ------------------------------------------------------------------ helpers
    def _valid_node_mask(self, node_idx: NDArray[np.int64]) -> NDArray[np.bool_]:
        return (
            (node_idx[:, 0] >= 0)
            & (node_idx[:, 0] < self._nx)
            & (node_idx[:, 1] >= 0)
            & (node_idx[:, 1] < self._ny)
            & (node_idx[:, 2] >= 0)
            & (node_idx[:, 2] < self._nz)
        )

    def _node_to_flat(self, node_idx: NDArray[np.int64]) -> NDArray[np.int64]:
        return node_idx[:, 0] * self._ny * self._nz + node_idx[:, 1] * self._nz + node_idx[:, 2]

    def _flat_to_node(self, flat: NDArray[np.int64]) -> NDArray[np.int64]:
        idx = np.empty((flat.shape[0], 3), dtype=np.int64)
        idx[:, 0] = flat // (self._ny * self._nz)
        rem = flat % (self._ny * self._nz)
        idx[:, 1] = rem // self._nz
        idx[:, 2] = rem % self._nz
        return idx

    def _weights_and_grads(self, pos: NDArray[np.float64]):
        """Return quadratic B-spline weights/grads for all particles.

        Uses the standard centered quadratic B-spline with a 3-node 1D stencil.
        Returns:
            base:   [N,3] integer lower-left node of the 3x3x3 stencil.
            w:      [N,27] weights for each of the 27 stencils (flattened order).
            grad:   [N,27,3] weight gradients for each stencil.
        """
        xg = (pos - self._grid_min) / self._dx
        # Nearest grid node index, then the 3x3x3 stencil is centred on base+1.
        nearest = np.floor(xg + 0.5)
        base = (nearest - 1).astype(np.int64)
        s = xg - (base + 1)  # in [-0.5, 0.5]

        # 1D weights and derivatives w.r.t. physical coordinates.
        # N0 = 0.5*(0.5 - s)^2, dN0/dx = -(0.5 - s)/dx
        # N1 = 0.75 - s^2,      dN1/dx = -2*s/dx
        # N2 = 0.5*(0.5 + s)^2, dN2/dx =  (0.5 + s)/dx
        s = np.clip(s, -0.5, 0.5)
        h = self._dx
        w_dim = np.stack(
            [
                0.5 * (0.5 - s) ** 2,
                0.75 - s ** 2,
                0.5 * (0.5 + s) ** 2,
            ],
            axis=-1,
        )  # [N,3,3]
        dw_dim = np.stack(
            [
                -(0.5 - s) / h,
                -2.0 * s / h,
                (0.5 + s) / h,
            ],
            axis=-1,
        )  # [N,3,3]

        # Build 27 stencils.
        w_list = []
        grad_list = []
        offsets = np.array(
            [[i, j, k] for i in range(3) for j in range(3) for k in range(3)],
            dtype=np.int64,
        )
        for off in offsets:
            wx, wy, wz = w_dim[:, 0, off[0]], w_dim[:, 1, off[1]], w_dim[:, 2, off[2]]
            gwx, gwy, gwz = dw_dim[:, 0, off[0]], dw_dim[:, 1, off[1]], dw_dim[:, 2, off[2]]
            wijk = wx * wy * wz
            # grad_j (W_x W_y W_z) = (dWx/dx * Wy * Wz, Wx * dWy/dy * Wz, Wx * Wy * dWz/dz)
            gradx = gwx * wy * wz
            grady = wx * gwy * wz
            gradz = wx * wy * gwz
            grad_ijk = np.stack([gradx, grady, gradz], axis=-1)
            w_list.append(wijk)
            grad_list.append(grad_ijk)
        w = np.stack(w_list, axis=1)  # [N,27]
        grad = np.stack(grad_list, axis=1)  # [N,27,3]
        return base, w, grad, offsets

    def _stress(self, F: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return Cauchy stress for a batch of deformation gradients."""
        # Compute SVD and polar decomposition.  Use the singular values for a
        # simple Hencky/logarithmic Neo-Hookean model with hardening.
        U, s, Vh = np.linalg.svd(F)
        s = np.clip(s, 1e-6, None)
        log_s = np.log(s)
        trace_log = log_s.sum(axis=-1, keepdims=True)
        J = np.prod(s, axis=-1, keepdims=True)
        # Principal Kirchhoff stress: lambda*tr(log S) + 2*mu*log_s
        kirchhoff = self._lam * trace_log + 2.0 * self._mu * log_s
        # Convert to Cauchy stress.
        # Build the diagonal stress in the principal frame and rotate back.
        # principal_cauchy = kirchhoff / J
        diag = kirchhoff / np.clip(J, 1e-8, None)
        # Rotate to world frame: R = U Vh (with det correction if needed).
        R = U @ Vh
        det_r = np.linalg.det(R)
        flip = det_r < 0
        if np.any(flip):
            U[flip, :, -1] *= -1
            R = U @ Vh
        stress = R @ np.apply_along_axis(np.diag, -1, diag) @ R.swapaxes(-2, -1)
        # Keep numerically sane.
        stress = np.clip(stress, -5e5, 5e5)
        return stress

    def _plasticity(self, F: NDArray[np.float64]) -> NDArray[np.float64]:
        U, s, Vh = np.linalg.svd(F)
        s = np.clip(s, self._min_F, self._max_F)
        return U @ np.apply_along_axis(np.diag, -1, s) @ Vh

    # ------------------------------------------------------------------ physics
    def _substep(self, sub_dt: float) -> None:
        # 1. Particle-to-grid transfer.
        base, w, grad, offsets = self._weights_and_grads(self._position)

        # node indices for all particles and all 27 stencils.
        # shape: [N,27,3]
        node_idx = base[:, None, :] + offsets[None, :, :]
        flat = self._node_to_flat(node_idx.reshape(-1, 3)).reshape(self._n, 27)
        valid = self._valid_node_mask(node_idx.reshape(-1, 3)).reshape(self._n, 27)

        # Current volume per particle.
        J = np.linalg.det(self._F)
        volume = self._volume0 * np.clip(J, 0.1, 10.0)
        stress = self._stress(self._F)

        grid_mass = np.zeros(self._n_nodes, dtype=float)
        grid_momentum = np.zeros((self._n_nodes, 3), dtype=float)
        grid_force = np.zeros((self._n_nodes, 3), dtype=float)

        for s_idx in range(27):
            w_col = w[:, s_idx]
            g_col = grad[:, s_idx, :]
            f_col = node_idx[:, s_idx, :]
            flat_col = flat[:, s_idx]
            ok = valid[:, s_idx]
            if not np.any(ok):
                continue

            w_ok = w_col[ok]
            flat_ok = flat_col[ok]
            mass_ok = self._mass[ok]
            vel_ok = self._velocity[ok]
            g_ok = g_col[ok]
            vol_ok = volume[ok]
            stress_ok = stress[ok]

            # Mass and momentum scatter.
            np.add.at(grid_mass, flat_ok, w_ok * mass_ok)
            np.add.at(grid_momentum, (flat_ok, slice(None)), w_ok[:, None] * mass_ok[:, None] * vel_ok)

            # Stress force: -V_p * sigma * grad_w.
            # [M,3] = [M,3,3] @ [M,3]
            f_stress = -vol_ok[:, None] * np.einsum("mij,mj->mi", stress_ok, g_ok)
            np.add.at(grid_force, (flat_ok, slice(None)), f_stress)

        # 2. Grid velocity update.
        active = grid_mass > 1e-12
        grid_vel = np.zeros((self._n_nodes, 3), dtype=float)
        m_inv = np.zeros(self._n_nodes, dtype=float)
        m_inv[active] = 1.0 / grid_mass[active]
        grid_vel[active] = grid_momentum[active] * m_inv[active][:, None]

        # Add gravity and damping.
        grid_vel[active] += sub_dt * (
            self._gravity
            - self._damping * grid_vel[active]
            + grid_force[active] * m_inv[active][:, None]
        )

        # Floor boundary condition: sticky floor (normal velocity clamped to non-negative).
        floor_nodes = self._floor_node_mask.reshape(-1)
        grid_vel[floor_nodes, 2] = np.maximum(grid_vel[floor_nodes, 2], 0.0)

        # 3. Grid-to-particle transfer.
        new_vel = np.zeros((self._n, 3), dtype=float)
        new_grad = np.zeros((self._n, 3, 3), dtype=float)
        for s_idx in range(27):
            w_col = w[:, s_idx]
            g_col = grad[:, s_idx, :]
            flat_col = flat[:, s_idx]
            ok = valid[:, s_idx]
            if not np.any(ok):
                continue
            v_node = grid_vel[flat_col[ok]]
            new_vel[ok] += w_col[ok][:, None] * v_node
            # grad_v += v_i * grad_j N
            new_grad[ok] += v_node[:, :, None] * g_col[ok][:, None, :]

        self._velocity = new_vel
        # Update position and deformation gradient.
        self._position += sub_dt * self._velocity
        # Deformation gradient update: F_new = (I + dt * grad_v) F_old.
        F_inc = np.eye(3, dtype=float)[None, :, :] + sub_dt * new_grad
        self._F = self._plasticity(np.einsum("pij,pjk->pik", F_inc, self._F))

        # Post-stabilisation: keep particles inside the domain and above the floor.
        self._position[:, 2] = np.maximum(self._position[:, 2], self._floor_top)
        self._position = np.clip(
            self._position,
            self._grid_min + self._dx,
            self._grid_max - self._dx,
        )

    # ------------------------------------------------------------------ public API
    def advance_to(self, time: float) -> StateFrame:
        if time < self._last_time:
            raise ValueError("advance_to called with non-increasing time")
        elapsed = time - self._last_time
        if elapsed > 0.0:
            n_sub = max(int(np.ceil(elapsed / self._sub_dt)), 1)
            sub_dt = elapsed / n_sub
            for _ in range(n_sub):
                self._substep(sub_dt)
            self._last_time = time
            self._substeps += n_sub
        return self._build_frame(time)

    def _build_frame(self, time: float) -> StateFrame:
        external_force = self._mass[:, None] * self._gravity[None, :]
        floor_sdf = self._position[:, 2] - self._floor_top
        frame = StateFrame(
            time=float(time),
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={
                "deformation_gradient": self._F.copy(),
                "mass_quantity": self._mass_quantity.copy(),
                "reference_volume": self._volume0.copy(),
                "initial_component_id": self._body_id.copy(),
            },
            fields={
                "external_force": external_force,
                "floor_signed_distance": floor_sdf,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "initial_total_mass": float(self._mass.sum()),
                "initial_mass_each": float(self._mass.sum() / 2.0),
                "initial_total_volume": float(self._volume0.sum()),
                "particle_spacing": float(self._config["snowballs"]["particle_spacing"]),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps": self._substeps,
            "particle_count": self._n,
            "grid_shape": list(self._grid_shape),
            "sub_dt": self._sub_dt,
        }
