"""Simple explicit SPH liquid solver for the viscosity contrast splash scene.

The solver advances two side-by-side liquid columns (low and high viscosity)
under gravity, colliding with a ground plane and a central divider.  It exports
all required raw trajectory fields, including collocated diagnostic grids and
boundary samples.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit weakly-compressible SPH solver with a simple viscous smoothing model."""

    def __init__(self) -> None:
        self.scene = None
        self.seed = 0
        self.time = 0.0
        self.dt = 1e-3

        # particle / SPH parameters
        self.spacing = 0.14
        self.rho0 = 1000.0
        self.mass = self.rho0 * self.spacing ** 3
        self.h = 0.20  # smoothing length
        self.support = 2.0 * self.h
        self.kernel_alpha = 1.0 / np.pi
        self.cs = 25.0  # speed of sound for equation of state
        self.gamma = 7.0
        self.B = self.cs ** 2 * self.rho0 / self.gamma
        self.gravity = np.array([0.0, 0.0, -9.81])

        # viscosity contrast model: local velocity relaxation rate (1/s)
        # low-viscosity labels (0,2) get near-inviscid damping, high-viscosity
        # labels (1,3) get strong damping so the fluid remains cohesive.
        self.nu_low = 0.2
        self.nu_high = 10.0

        # boundary penalty
        self.k_boundary = 5e4
        self.c_boundary = 2.0

        # diagnostic grid
        self.nx, self.ny, self.nz = 24, 16, 28
        self.grid_spacing = 0.20
        self.grid_origin = np.array([-2.4, -1.6, 0.0])

        # state arrays (filled in initialize)
        self.pos = np.zeros((0, 3))
        self.vel = np.zeros((0, 3))
        self.label = np.zeros(0, dtype=np.int32)
        self.mass_arr = np.zeros(0)
        self.body_id = np.zeros(0, dtype=np.int32)
        self.attrs: dict = {}
        self.base_scalars: dict = {}
        self.nu = np.zeros(0)
        self.N = 0

        # boundary samples
        self.boundary_pos = np.zeros((0, 3))
        self.boundary_normal = np.zeros((0, 3))
        self.boundary_target_vel = np.zeros((0, 3))

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        frame0 = scene.initial_state()
        self.pos = np.asarray(frame0.position, dtype=float).copy()
        self.vel = np.asarray(frame0.velocity, dtype=float).copy()
        self.label = np.asarray(frame0.attributes["material_label"], dtype=np.int32)
        self.mass_arr = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.N = self.pos.shape[0]

        # preserve original attributes required by the trajectory contract
        self.attrs = {
            "mass_quantity": self.mass_arr.copy(),
            "particle_radius": np.asarray(frame0.attributes["particle_radius"], dtype=float).copy(),
            "material_label": self.label.copy(),
            "deformation_gradient": np.asarray(
                frame0.attributes["deformation_gradient"], dtype=float
            ).copy(),
        }

        # scalar values for trajectory output
        char_len = float(np.max(np.ptp(self.pos, axis=0)))
        if char_len <= 0.0 or not np.isfinite(char_len):
            char_len = self.spacing
        self.base_scalars = {
            "grid_spacing": self.grid_spacing,
            "characteristic_length": char_len,
            "reference_velocity": 1.0,
        }

        # viscosity per particle based on material (low vs high)
        self.nu = np.where(
            (self.label == 0) | (self.label == 2),
            self.nu_low,
            self.nu_high,
        )

        # static boundary samples: ground + divider faces
        self.boundary_pos, self.boundary_normal, self.boundary_target_vel = _make_boundary_samples()

    def diagnostics(self) -> dict[str, object]:
        return {
            "substeps": int(round(self.time / self.dt)),
            "dt": self.dt,
            "particles": self.N,
        }

    def advance_to(self, time: float) -> StateFrame:
        if time < self.time:
            raise ValueError("advance_to time must be non-decreasing")

        while self.time < time:
            step_dt = min(self.dt, time - self.time)
            self._substep(step_dt)

        return self._build_frame(time)

    # --------------------------------------------------------------------- SPH
    def _substep(self, dt: float) -> None:
        pos = self.pos
        vel = self.vel
        N = self.N

        tree = cKDTree(pos)
        pairs = tree.query_ball_point(pos, self.support, output_type="ndarray")

        i = pairs[:, 0]
        j = pairs[:, 1]
        r_vec = pos[i] - pos[j]
        r2 = np.einsum("ij,ij->i", r_vec, r_vec)
        r = np.sqrt(r2)
        q = r / self.h
        mask = q < 2.0
        i = i[mask]
        j = j[mask]
        r_vec = r_vec[mask]
        r = r[mask]
        q = q[mask]

        # cubic spline kernel F(q) and derivative F'(q)
        F = np.zeros_like(q)
        Fp = np.zeros_like(q)
        m1 = q < 1.0
        F[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
        Fp[m1] = -3.0 * q[m1] + 2.25 * q[m1] ** 2
        m2 = (~m1) & (q < 2.0)
        t = 2.0 - q[m2]
        F[m2] = 0.25 * t ** 3
        Fp[m2] = -0.75 * t ** 2

        inv_h3 = 1.0 / (self.h ** 3)
        inv_h4 = 1.0 / (self.h ** 4)
        Wval = self.kernel_alpha * inv_h3 * F

        # density
        rho = np.zeros(N)
        np.add.at(rho, i, self.mass * Wval)

        # pressure (Tait, clamped to zero tension)
        ratio = rho / self.rho0
        pressure = self.B * np.clip(ratio ** self.gamma - 1.0, 0.0, None)

        # acceleration accumulator
        acc = np.zeros((N, 3))

        # pressure force using symmetrized SPH (directed pairs)
        safe = r > 1e-12
        if np.any(safe):
            isafe = i[safe]
            jsafe = j[safe]
            r_vec_s = r_vec[safe]
            r_s = r[safe]
            Fp_s = Fp[safe]

            p_over = np.zeros(N)
            # avoid 0/0 for very low densities
            p_over[rho > 0] = pressure[rho > 0] / (rho[rho > 0] ** 2)

            grad = (
                self.kernel_alpha
                * inv_h4
                * (Fp_s / r_s)[:, None]
                * r_vec_s
            )
            term = self.mass * (p_over[isafe] + p_over[jsafe])[:, None] * grad
            np.add.at(acc, isafe, -term)

        # viscous smoothing: relax local velocity toward SPH-interpolated mean
        w = self.mass * Wval
        num = np.zeros((N, 3))
        denom = np.zeros(N)
        np.add.at(num, i, w[:, None] * vel[j])
        np.add.at(denom, i, w)
        # self contribution
        w_self = self.mass * self.kernel_alpha * inv_h3
        denom += w_self
        num += w_self * vel
        v_smooth = num / denom[:, None]
        acc += self.nu[:, None] * (v_smooth - vel)

        # body forces
        acc += self.gravity
        acc = self._boundary_forces(acc)

        # explicit Euler integration
        vel += dt * acc
        pos += dt * vel

        # hard boundary projection
        self._project_boundaries()

        self.time += dt

    # ---------------------------------------------------------------- boundary
    def _boundary_forces(self, acc: np.ndarray) -> np.ndarray:
        acc = _plane_penalty(
            self.pos, self.vel, acc, self.mass, self.support, self.k_boundary, self.c_boundary,
            np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
        )
        acc = _plane_penalty(
            self.pos, self.vel, acc, self.mass, self.support, self.k_boundary, self.c_boundary,
            np.array([0.0, 0.0, 3.8]), np.array([0.0, 0.0, -1.0]),
        )
        for x0, normal in [(-3.2, np.array([-1.0, 0.0, 0.0])),
                           (3.2, np.array([1.0, 0.0, 0.0]))]:
            acc = _plane_penalty(
                self.pos, self.vel, acc, self.mass, self.support, self.k_boundary, self.c_boundary,
                np.array([x0, 0.0, 0.0]), normal,
            )
        for y0, normal in [(-1.4, np.array([0.0, -1.0, 0.0])),
                           (1.4, np.array([0.0, 1.0, 0.0]))]:
            acc = _plane_penalty(
                self.pos, self.vel, acc, self.mass, self.support, self.k_boundary, self.c_boundary,
                np.array([0.0, y0, 0.0]), normal,
            )
        for x0, normal in [(-0.08, np.array([-1.0, 0.0, 0.0])),
                           (0.08, np.array([1.0, 0.0, 0.0]))]:
            acc = _plane_penalty(
                self.pos, self.vel, acc, self.mass, self.support, self.k_boundary, self.c_boundary,
                np.array([x0, 0.0, 0.0]), normal,
                ylim=(-0.9, 0.9), zlim=(0.0, 1.3),
            )
        return acc

    def _project_boundaries(self) -> None:
        pos = self.pos
        vel = self.vel

        # ground
        mask = pos[:, 2] < 0.0
        pos[mask, 2] = 0.0
        vel[mask, 2] = np.maximum(vel[mask, 2], 0.0)

        # top
        mask = pos[:, 2] > 3.8
        pos[mask, 2] = 3.8
        vel[mask, 2] = np.minimum(vel[mask, 2], 0.0)

        # side walls
        for x0, sign in [(-3.2, 1.0), (3.2, -1.0)]:
            mask = sign * (pos[:, 0] - x0) < 0.0
            pos[mask, 0] = x0
            v_n = vel[mask, 0] * (sign * -1.0)
            vel[mask, 0] = np.where(v_n < 0.0, vel[mask, 0] - v_n * (sign * -1.0), vel[mask, 0])

        for y0, sign in [(-1.4, 1.0), (1.4, -1.0)]:
            mask = sign * (pos[:, 1] - y0) < 0.0
            pos[mask, 1] = y0
            v_n = vel[mask, 1] * (sign * -1.0)
            vel[mask, 1] = np.where(v_n < 0.0, vel[mask, 1] - v_n * (sign * -1.0), vel[mask, 1])

        # divider
        inside = (
            (np.abs(pos[:, 0]) < 0.08)
            & (np.abs(pos[:, 1]) < 0.9)
            & (pos[:, 2] < 1.3)
        )
        if np.any(inside):
            x = pos[inside, 0]
            sign = np.sign(x)
            sign[sign == 0] = -1.0
            pos[inside, 0] = sign * 0.08
            n = np.column_stack([sign, np.zeros_like(sign), np.zeros_like(sign)])
            v_n = np.einsum("ij,ij->i", vel[inside], n)
            mask_in = v_n < 0.0
            vel[inside][mask_in] -= v_n[mask_in][:, None] * n[mask_in]

    # ----------------------------------------------------------------- frames
    def _build_frame(self, time: float) -> StateFrame:
        nx, ny, nz = self.nx, self.ny, self.nz
        ix = np.arange(nx) * self.grid_spacing + self.grid_origin[0]
        iy = np.arange(ny) * self.grid_spacing + self.grid_origin[1]
        iz = np.arange(nz) * self.grid_spacing + self.grid_origin[2]
        gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
        grid_points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

        v_field, active = self._interpolate_velocity(grid_points)
        v_field = v_field.reshape(nx, ny, nz, 3)
        active_mask = active.reshape(nx, ny, nz).astype(float)

        tree = cKDTree(self.pos)
        dist, _ = tree.query(grid_points, k=1)
        level = (dist - 0.5 * self.spacing).reshape(nx, ny, nz)

        b_vel, _ = self._interpolate_velocity(self.boundary_pos)

        K = int(np.max(self.label) + 1)
        cumulative_mass_source = np.zeros(max(K, 4), dtype=float)

        fields = {
            "active_mask": active_mask,
            "boundary_normal": self.boundary_normal.copy(),
            "boundary_target_velocity": self.boundary_target_vel.copy(),
            "boundary_velocity": b_vel,
            "cumulative_mass_source": cumulative_mass_source,
            "liquid_level_set": level,
            "velocity_field": v_field,
        }

        scalars = dict(self.base_scalars)

        return StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass_arr.copy(),
            body_id=self.body_id.copy(),
            attributes=self.attrs,
            fields=fields,
            scalars=scalars,
        )

    def _interpolate_velocity(self, query_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return velocity and support mask at arbitrary query positions."""
        if self.N == 0:
            return np.zeros_like(query_points), np.zeros(len(query_points), dtype=bool)

        tree = cKDTree(self.pos)
        pairs = tree.query_ball_point(query_points, self.support, output_type="ndarray")
        Q = query_points.shape[0]
        vel_out = np.zeros_like(query_points)
        if pairs.size == 0:
            return vel_out, np.zeros(Q, dtype=bool)

        g = pairs[:, 0]
        p = pairs[:, 1]
        r_vec = query_points[g] - self.pos[p]
        r = np.linalg.norm(r_vec, axis=1)
        q = r / self.h
        mask = q < 2.0
        g = g[mask]
        p = p[mask]
        r = r[mask]
        q = q[mask]

        F = np.zeros_like(q)
        m1 = q < 1.0
        F[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
        m2 = (~m1) & (q < 2.0)
        F[m2] = 0.25 * (2.0 - q[m2]) ** 3
        Wval = self.kernel_alpha / (self.h ** 3) * F

        num = np.zeros((Q, 3))
        denom = np.zeros(Q)
        np.add.at(num, g, Wval[:, None] * self.vel[p])
        np.add.at(denom, g, Wval)
        active = denom > 0.0
        denom_safe = np.where(active, denom, 1.0)
        vel_out = num / denom_safe[:, None]
        return vel_out, active


def _plane_penalty(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    mass: float,
    support: float,
    k: float,
    c: float,
    p0: np.ndarray,
    normal: np.ndarray,
    ylim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
) -> np.ndarray:
    d = np.dot(pos - p0, normal)
    mask = d < support
    if ylim is not None:
        mask &= (pos[:, 1] > ylim[0]) & (pos[:, 1] < ylim[1])
    if zlim is not None:
        mask &= (pos[:, 2] > zlim[0]) & (pos[:, 2] < zlim[1])
    if not np.any(mask):
        return acc
    penetration = support - d[mask]
    v_n = np.dot(vel[mask], normal)
    f_n = k * penetration - c * np.minimum(v_n, 0.0)
    acc[mask] += (f_n / mass)[:, None] * normal[None, :]
    return acc


def _make_boundary_samples() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = []
    normals = []

    # ground
    xs = np.linspace(-3.0, 3.0, 16)
    ys = np.linspace(-1.2, 1.2, 8)
    for x in xs:
        for y in ys:
            samples.append([x, y, 0.0])
            normals.append([0.0, 0.0, 1.0])

    # divider faces
    yd = np.linspace(-0.8, 0.8, 5)
    zd = np.linspace(0.2, 1.2, 5)
    for y in yd:
        for z in zd:
            samples.append([-0.08, y, z])
            normals.append([-1.0, 0.0, 0.0])
            samples.append([0.08, y, z])
            normals.append([1.0, 0.0, 0.0])

    pos = np.array(samples, dtype=float)
    nor = np.array(normals, dtype=float)
    target = np.zeros_like(pos)
    return pos, nor, target
