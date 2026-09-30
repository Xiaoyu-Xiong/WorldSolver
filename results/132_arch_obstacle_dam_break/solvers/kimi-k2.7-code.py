"""3-D SPH dam-break solver for the arch-obstacle scene.

The solver uses a weakly-compressible Smoothed Particle Hydrodynamics (SPH)
scheme with artificial viscosity, an isothermal Tait equation of state, and
axis-aligned box boundary collisions.  It runs purely on NumPy/SciPy and
advances the particles with explicit time stepping.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# Kernel helpers (3-D cubic spline)
# ---------------------------------------------------------------------------
def _W_vec(r: NDArray, h: float) -> NDArray:
    q = r / h
    alpha = 1.0 / (np.pi * h ** 3)
    w = np.zeros_like(r)
    m1 = q <= 1.0
    m2 = (q > 1.0) & (q < 2.0)
    w[m1] = alpha * (1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3)
    w[m2] = alpha * 0.25 * (2.0 - q[m2]) ** 3
    return w


def _dW_vec(r: NDArray, h: float) -> NDArray:
    q = r / h
    alpha = 1.0 / (np.pi * h ** 3)
    dw = np.zeros_like(r)
    m1 = q <= 1.0
    m2 = (q > 1.0) & (q < 2.0)
    dw[m1] = alpha * (-3.0 * q[m1] + 2.25 * q[m1] ** 2) / h
    dw[m2] = alpha * -0.75 * (2.0 - q[m2]) ** 2 / h
    return dw


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)
        self.scene = scene
        self.seed = seed

        # Initial state from the scene scaffold.
        frame0 = scene.initial_state()
        self.position: NDArray = np.asarray(frame0.position, dtype=float).copy()
        self.velocity: NDArray = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass: NDArray = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id: NDArray = np.asarray(frame0.body_id, dtype=np.int32).copy()

        self.attributes = {
            "mass_quantity": np.asarray(frame0.attributes["mass_quantity"], dtype=float).copy(),
            "particle_radius": np.asarray(frame0.attributes["particle_radius"], dtype=float).copy(),
            "material_label": np.asarray(frame0.attributes["material_label"], dtype=np.int32).copy(),
            "deformation_gradient": np.asarray(
                frame0.attributes["deformation_gradient"], dtype=float
            ).copy(),
        }

        # Configuration.
        self.gravity = np.asarray(self.scene.config["gravity"], dtype=float)
        self.spacing = float(self.scene.config["sampling"]["spacing"])
        self.rho0 = float(self.scene.config["sampling"]["density"])

        # SPH parameters.
        self.h = 2.0 * self.spacing
        self.c_s = 40.0
        self.B = (self.c_s ** 2) * self.rho0 / 7.0
        self.alpha_visc = 1.0
        self.beta_visc = 0.0
        self.epsilon = 1e-6

        # Time bookkeeping.
        self.time = 0.0
        self.max_time = float(self.scene.config["duration"])
        self.dt_max = 1.0e-3

        # Boundary restitution.
        self.restitution = 0.0

        # Diagnostic grid.
        grid_shape = tuple(int(s) for s in self.scene.config["sampling"]["field_grid_shape"])
        self.grid_shape = grid_shape
        self.grid_dx = self.spacing
        self.grid_origin = np.array([-2.0, -1.05, 0.0], dtype=float)

        # Precompute cell centre positions as a (Ncells, 3) array.
        ix, iy, iz = np.meshgrid(
            np.arange(grid_shape[0]),
            np.arange(grid_shape[1]),
            np.arange(grid_shape[2]),
            indexing="ij",
        )
        self.grid_centres = (
            np.stack([ix.ravel(), iy.ravel(), iz.ravel()], axis=1) * self.grid_dx
            + self.grid_origin
        )

        # Boundary geometry: list of (centre, half_extents) for static walls.
        self.static_boxes = [
            (np.array([0.0, 0.0, -0.05]), np.array([3.2, 1.4, 0.05])),
            (np.array([-0.75, 0.0, 0.7]), np.array([0.22, 0.7, 0.7])),
            (np.array([0.75, 0.0, 0.7]), np.array([0.22, 0.7, 0.7])),
            (np.array([0.0, 0.0, 1.4]), np.array([0.95, 0.7, 0.18])),
        ]
        self.gate_box = (
            np.array([-1.55, 0.0, 0.55]),
            np.array([0.04, 0.65, 0.55]),
        )

        # Precompute boundary sample points and normals (static walls only).
        self._boundary_samples = self._build_boundary_samples()

        # Keep the same scalar normalisation constants as the scaffold.
        self.scalar_grid_spacing = self.spacing
        self.scalar_characteristic_length = float(frame0.scalars["characteristic_length"])
        self.scalar_reference_velocity = 1.0

    def _build_boundary_samples(self) -> tuple[NDArray, NDArray]:
        """Return (positions, outward unit normals) for the rigid walls."""
        samples: list[NDArray] = []
        normals: list[NDArray] = []
        for c, h in self.static_boxes:
            for axis in range(3):
                axes = [i for i in range(3) if i != axis]
                # Negative and positive faces.
                for side in (-1, 1):
                    face = self._face_samples(c, h, axis, side)
                    n = np.zeros((face.shape[0], 3), dtype=float)
                    n[:, axis] = float(side)
                    samples.append(face)
                    normals.append(n)
        pos = np.concatenate(samples) if samples else np.zeros((1, 3), dtype=float)
        nor = np.concatenate(normals) if normals else np.zeros((1, 3), dtype=float)
        return pos, nor

    def _face_samples(self, c: NDArray, h: NDArray, axis: int, side: int) -> NDArray:
        axes = [i for i in range(3) if i != axis]
        ranges = []
        for a in axes:
            lo = c[a] - h[a] + self.spacing * 0.5
            hi = c[a] + h[a]
            if hi <= lo:
                ranges.append(np.array([c[a]]))
            else:
                ranges.append(np.arange(lo, hi, self.spacing))
        uu, vv = np.meshgrid(ranges[0], ranges[1], indexing="ij")
        pts = np.zeros((uu.size, 3), dtype=float)
        pts[:, axes[0]] = uu.ravel()
        pts[:, axes[1]] = vv.ravel()
        pts[:, axis] = c[axis] + side * h[axis]
        return pts

    def _all_boxes(self) -> list[tuple[NDArray, NDArray]]:
        boxes = self.static_boxes.copy()
        # The gate is present only at the initial instant and during the
        # first integration substep.
        if self.time <= 0.0:
            boxes.append(self.gate_box)
        return boxes

    def _enforce_boundaries(self, pos: NDArray, vel: NDArray) -> None:
        """Resolve penetrations with all rigid boxes (vectorised)."""
        for c, h in self._all_boxes():
            d = pos - c
            ad = np.abs(d)
            vals = ad - h  # signed distance per axis (positive outside)
            inside = np.all(vals <= 0.0, axis=1)
            if not np.any(inside):
                continue

            pen = vals.max(axis=1)
            arg = np.argmax(vals, axis=1)
            n = np.zeros_like(pos)
            n[np.arange(len(pos)), arg] = np.sign(d[np.arange(len(pos)), arg])
            norm = np.linalg.norm(n, axis=1, keepdims=True)
            n[inside] /= norm[inside] + 1e-12

            # Push the particle back to the surface along the closest-axis normal.
            pos[inside] -= pen[inside][:, None] * n[inside]

            # Remove the inward normal component of velocity.
            vn = np.einsum("ij,ij->i", vel, n)
            collide = inside & (vn < 0.0)
            vel[collide] -= (1.0 + self.restitution) * vn[collide][:, None] * n[collide]

    def _sph_acceleration(self, pos: NDArray, vel: NDArray) -> NDArray:
        n = pos.shape[0]

        # Build neighbour pair list using a 2h cutoff.
        tree = cKDTree(pos)
        pairs = tree.sparse_distance_matrix(tree, 2.0 * self.h, output_type="ndarray")
        i = np.asarray(pairs["i"], dtype=np.int32)
        j = np.asarray(pairs["j"], dtype=np.int32)
        d = np.asarray(pairs["v"], dtype=float)

        # ---- Density --------------------------------------------------------
        W = _W_vec(d, self.h)
        rho = np.zeros(n, dtype=float)
        np.add.at(rho, i, self.mass[j] * W)

        # ---- Pressure + viscosity interactions (exclude self pairs) ---------
        mask = i != j
        i = i[mask]
        j = j[mask]
        d = d[mask]

        dx = pos[i] - pos[j]
        dW = _dW_vec(d, self.h)
        inv_r = 1.0 / (d + self.epsilon)
        grad = (dW * inv_r)[:, None] * dx  # grad W_ij = dW/dr * (x_i - x_j) / r

        # Pressure.
        ratio_i = rho[i] / self.rho0
        ratio_j = rho[j] / self.rho0
        ratio_i = np.clip(ratio_i, 1e-6, None)
        ratio_j = np.clip(ratio_j, 1e-6, None)
        p_i = np.maximum(self.B * (ratio_i ** 7 - 1.0), 0.0)
        p_j = np.maximum(self.B * (ratio_j ** 7 - 1.0), 0.0)

        acc = np.zeros((n, 3), dtype=float)
        term_p = -self.mass[j][:, None] * (
            p_i / (rho[i] ** 2) + p_j / (rho[j] ** 2)
        )[:, None] * grad
        np.add.at(acc, i, term_p)

        # Artificial viscosity (Monaghan).
        vji = vel[j] - vel[i]
        vr = np.einsum("ij,ij->i", vji, dx)
        mu = self.h * vr / (d ** 2 + self.epsilon)
        approaching = vr < 0.0
        pi = np.zeros_like(d)
        denom = 0.5 * (rho[i[approaching]] + rho[j[approaching]])
        pi[approaching] = (
            -self.alpha_visc * self.c_s * mu[approaching]
            + self.beta_visc * mu[approaching] ** 2
        ) / denom
        term_v = self.mass[j][:, None] * pi[:, None] * grad
        np.add.at(acc, i, term_v)

        # Gravity.
        acc += self.gravity
        return acc

    def _grid_diagnostics(self) -> tuple[NDArray, NDArray, NDArray]:
        """Compute active mask, diagnostic velocity field, and liquid level set."""
        tree = cKDTree(self.position)

        # Signed level set: distance to the nearest particle minus a particle
        # radius so that values are negative inside the fluid and positive outside.
        d_nearest, _ = tree.query(self.grid_centres, k=1)
        level_set = d_nearest.reshape(self.grid_shape) - self.attributes["particle_radius"][0]
        active = level_set < 0.0

        velocity_field = np.zeros(self.grid_shape + (3,), dtype=float)
        if np.any(active):
            # Neighbour lists for all grid centres.
            h_interp = self.h
            neigh = tree.query_ball_point(self.grid_centres, r=h_interp)
            flat_active = np.where(active.ravel())[0]
            vel_view = velocity_field.reshape((-1, 3))
            for flat_idx in flat_active:
                idx = neigh[flat_idx]
                if not idx:
                    continue
                diffs = self.position[idx] - self.grid_centres[flat_idx]
                r = np.linalg.norm(diffs, axis=1)
                w = _W_vec(r, h_interp)
                if w.sum() > 0.0:
                    vel_view[flat_idx] = np.average(
                        self.velocity[idx], axis=0, weights=w
                    )

        return active, velocity_field, level_set

    def _make_frame(self) -> StateFrame:
        active_mask, velocity_field, level_set = self._grid_diagnostics()
        boundary_pos, boundary_n = self._boundary_samples
        boundary_vel = np.zeros_like(boundary_pos)
        boundary_target = np.zeros_like(boundary_pos)

        fields = {
            "active_mask": active_mask,
            "boundary_normal": boundary_n,
            "boundary_target_velocity": boundary_target,
            "boundary_velocity": boundary_vel,
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "liquid_level_set": level_set,
            "velocity_field": velocity_field,
        }

        scalars = {
            "characteristic_length": self.scalar_characteristic_length,
            "grid_spacing": self.scalar_grid_spacing,
            "reference_velocity": self.scalar_reference_velocity,
        }

        return StateFrame(
            self.time,
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.attributes["mass_quantity"].copy(),
                "particle_radius": self.attributes["particle_radius"].copy(),
                "material_label": self.attributes["material_label"].copy(),
                "deformation_gradient": self.attributes["deformation_gradient"].copy(),
            },
            fields=fields,
            scalars=scalars,
        )

    def _substep(self, dt: float) -> None:
        acc = self._sph_acceleration(self.position, self.velocity)
        self.velocity += dt * acc
        self.position += dt * self.velocity
        self._enforce_boundaries(self.position, self.velocity)

    def advance_to(self, time: float) -> StateFrame:
        target = min(float(time), self.max_time)
        while self.time < target - 1e-12:
            remaining = target - self.time
            speed = np.max(np.linalg.norm(self.velocity, axis=1)) if self.position.shape[0] else 0.0
            dt_cfl = 0.25 * self.h / (self.c_s + speed + 1e-6)
            dt = min(remaining, self.dt_max, dt_cfl)
            if dt < 1e-9:
                break
            self._substep(dt)
            self.time += dt
        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "sph_arch_dam_break",
            "particles": int(self.position.shape[0]),
            "time": float(self.time),
            "seed": int(self.seed),
        }
