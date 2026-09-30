"""Simple physical solver for the cable-winding scene.

The cable is modelled as a discrete extensible elastic rod (stretch + bending)
with explicit time stepping.  Hard frictional contact with the cylinder and
floor is resolved with position projection and a generalized power-law friction
bound on the tangential impulse.  No precomputed animation tables are used.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic explicit elastic-rod solver."""

    def __init__(self) -> None:
        self.scene = None
        self.config = None
        self.x: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.m: np.ndarray | None = None
        self.t = 0.0
        self.dt = 5.0e-5
        self.n = 0

        # Material / geometric constants filled in initialize()
        self.l0 = 0.0
        self.cable_r = 0.0
        self.EA = 0.0
        self.k_bend = 0.0
        self.k_self = 1.0e4
        self.cyl_radius = 0.0
        self.cyl_height = 0.0
        self.cyl_half_height = 0.0
        self.g = np.array([0.0, 0.0, -9.81])
        self.kappa = 1.296
        self.friction_N = 0.7616
        self.visc = 0.1

        self.driven = 0
        self.duration = 0.0
        self.sample_dt = 0.02
        self.target_positions: np.ndarray | None = None
        self.target_velocities: np.ndarray | None = None

        self.cum_external_impulse = np.zeros(3)
        self.cum_external_angular_impulse = np.zeros(3)
        self.ref_p = 1.0
        self.ref_L = 1.0

    # ------------------------------------------------------------------ #
    # Protocol entry points
    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = scene.config

        init = scene.initial_state()
        self.x = np.asarray(init.position, dtype=float).copy()
        self.v = np.asarray(init.velocity, dtype=float).copy()
        self.m = np.asarray(init.mass, dtype=float).copy()
        self.n = self.x.shape[0]
        self.t = 0.0

        cable_cfg = self.config["cable"]
        length = float(cable_cfg["length"])
        nverts = int(cable_cfg["vertex_count"])
        self.l0 = length / (nverts - 1)
        self.cable_r = float(cable_cfg["radius"])
        area = np.pi * self.cable_r * self.cable_r
        E = float(cable_cfg["young_modulus_pa"])
        self.EA = E * area
        I_area = np.pi * self.cable_r**4 / 4.0
        EI = E * I_area
        self.k_bend = 4.0 * EI / (self.l0**3)

        cyl = self.config["cylinder"]
        self.cyl_radius = float(cyl["radius"])
        self.cyl_height = float(cyl["height"])
        self.cyl_half_height = self.cyl_height / 2.0

        self.kappa = float(self.config["contact"]["kappa"])
        self.friction_N = float(self.config["contact"]["exponent_N"])

        self.driven = int(cable_cfg["driven_vertex"])
        self.duration = float(self.config["duration"])
        self.sample_dt = float(self.config["sample_dt"])

        # Target grip trajectory sampled at output times.
        sample_times = self.scene.sample_times()
        n_samples = len(sample_times)
        self.target_positions = np.zeros((n_samples, 3))
        self.target_velocities = np.zeros((n_samples, 3))
        for i, ti in enumerate(sample_times):
            info = self.scene.grip_target(float(ti))
            self.target_positions[i] = info["position"]
        # Central-difference velocities.
        if n_samples >= 3:
            self.target_velocities[1:-1] = (
                self.target_positions[2:] - self.target_positions[:-2]
            ) / (2.0 * self.sample_dt)
        if n_samples >= 2:
            self.target_velocities[0] = (
                self.target_positions[1] - self.target_positions[0]
            ) / self.sample_dt
            self.target_velocities[-1] = (
                self.target_positions[-1] - self.target_positions[-2]
            ) / self.sample_dt

        total_mass = float(self.m.sum())
        self.ref_p = total_mass * np.sqrt(9.81 * length)
        self.ref_L = self.ref_p * length

        self.cum_external_impulse = np.zeros(3)
        self.cum_external_angular_impulse = np.zeros(3)

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        if time < self.t:
            time = self.t
        n_steps = int(round((time - self.t) / self.dt))
        for _ in range(max(0, n_steps)):
            self._step()
        return self._frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.t,
            "dt": self.dt,
            "n_vertices": self.n,
            "cable_mass": float(self.m.sum()) if self.m is not None else 0.0,
        }

    # ------------------------------------------------------------------ #
    # Time stepping
    # ------------------------------------------------------------------ #
    def _step(self) -> None:
        x = self.x
        v = self.v
        m = self.m
        dt = self.dt

        # Non-contact forces.
        F_int = self._internal_forces(x)
        F_self = self._self_contact_forces(x)
        F_grav = m[:, None] * self.g

        # Explicit velocity update with light viscous damping.
        v_new = v + dt * (F_int + F_self + F_grav) / m[:, None]
        v_new *= 1.0 - self.visc * dt

        # Predicted positions.
        x_new = x + dt * v_new

        # Hard contact resolution (cylinder + floor).
        x_new, v_new = self._resolve_contacts(x_new, v_new)

        # Kinematic driven end.
        pos_target, vel_target = self._driven_target(self.t + dt)
        x_new[self.driven] = pos_target
        v_new[self.driven] = vel_target

        # Budget updates: total external impulse = total momentum change.
        dp = m[:, None] * (v_new - v)
        self.cum_external_impulse += dp.sum(axis=0)
        self.cum_external_angular_impulse += np.cross(x_new, dp).sum(axis=0)

        self.x = x_new
        self.v = v_new
        self.t += dt

    # ------------------------------------------------------------------ #
    # Internal forces
    # ------------------------------------------------------------------ #
    def _internal_forces(self, x: np.ndarray) -> np.ndarray:
        F = np.zeros_like(x)
        # Stretch.
        edges = x[1:] - x[:-1]
        lengths = np.linalg.norm(edges, axis=1)
        e = edges / (lengths[:, None] + 1e-12)
        strain = (lengths - self.l0) / self.l0
        f_mag = self.EA * strain
        f = f_mag[:, None] * e
        F[1:] += f
        F[:-1] -= f

        # Bending via discrete curvature potential.
        if self.n > 2:
            d = np.zeros_like(x)
            d[1:-1] = x[1:-1] - (x[:-2] + x[2:]) / 2.0
            F_bend = -self.k_bend * d
            F_bend[1:] += 0.5 * self.k_bend * d[:-1]
            F_bend[:-1] += 0.5 * self.k_bend * d[1:]
            F += F_bend
        return F

    def _self_contact_forces(self, x: np.ndarray) -> np.ndarray:
        F = np.zeros_like(x)
        n = self.n
        d2 = 2.0 * self.cable_r
        # Pairwise differences and distances.
        dx = x[:, None, :] - x[None, :, :]
        d = np.linalg.norm(dx, axis=2)
        i_idx, j_idx = np.indices((n, n))
        mask = (d < d2) & (np.abs(i_idx - j_idx) > 2) & (i_idx < j_idx)
        if not np.any(mask):
            return F
        for i, j in zip(*np.where(mask)):
            nvec = dx[i, j] / (d[i, j] + 1e-12)
            pen = d2 - d[i, j]
            f = self.k_self * pen * nvec
            F[i] += f
            F[j] -= f
        return F

    # ------------------------------------------------------------------ #
    # Contact geometry
    # ------------------------------------------------------------------ #
    def _cylinder_penetration(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        pen = np.zeros(n)
        normal = np.zeros_like(x)

        xy = x[:, :2]
        d_xy = np.linalg.norm(xy, axis=1)
        z = x[:, 2]
        z_local = z - self.cyl_half_height
        R = self.cyl_radius
        r = self.cable_r
        H = self.cyl_height

        f1 = d_xy - R
        f2 = np.abs(z_local) - H / 2.0
        sd = np.maximum(f1, f2)
        active = sd < 1e3  # all points

        # Side-active.
        side = active & (f1 >= f2)
        if np.any(side):
            nvec = np.zeros((np.count_nonzero(side), 3))
            dside = d_xy[side]
            xy_side = xy[side]
            safe = dside > 1e-12
            nvec[safe, :2] = xy_side[safe] / dside[safe, None]
            if not safe.all():
                nvec[~safe, 0] = 1.0
            normal[side] = nvec

        # Cap-active.
        cap = active & (f2 > f1)
        if np.any(cap):
            nvec = np.zeros((np.count_nonzero(cap), 3))
            nvec[:, 2] = np.sign(z_local[cap])
            normal[cap] = nvec

        pen = np.maximum(0.0, -sd - r)
        return pen, normal

    def _floor_penetration(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pen = np.maximum(0.0, self.cable_r - x[:, 2])
        normal = np.zeros_like(x)
        normal[:, 2] = 1.0
        return pen, normal

    def _resolve_contacts(self, x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Cylinder.
        pen, n = self._cylinder_penetration(x)
        if pen.max() > 0.0:
            F_int = self._internal_forces(x)
            F_grav = self.m[:, None] * self.g
            F_pre = F_int + F_grav
            x, v = self._apply_contact_set(x, v, pen, n, F_pre)

        # Floor.
        pen, n = self._floor_penetration(x)
        if pen.max() > 0.0:
            F_int = self._internal_forces(x)
            F_grav = self.m[:, None] * self.g
            F_pre = F_int + F_grav
            x, v = self._apply_contact_set(x, v, pen, n, F_pre)

        return x, v

    def _apply_contact_set(
        self,
        x: np.ndarray,
        v: np.ndarray,
        pen: np.ndarray,
        n: np.ndarray,
        F_pre: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = pen > 0.0
        if not np.any(mask):
            return x, v

        m = self.m[mask]
        v_m = v[mask]
        n_m = n[mask]
        F_m = F_pre[mask]
        pen_m = pen[mask]

        # Normal component of velocity (positive = separating).
        v_n = np.einsum("ij,ij->i", v_m, n_m)

        # Estimate continuous normal reaction force from non-contact forces.
        fn = np.maximum(0.0, -np.einsum("ij,ij->i", F_m, n_m))

        # Normal impulse available for friction bound (reaction + velocity correction).
        lambda_n = fn * self.dt + m * np.maximum(0.0, -v_n)

        # Tangential velocity.
        v_t = v_m - v_n[:, None] * n_m
        v_t_mag = np.linalg.norm(v_t, axis=1)

        # Generalized power friction: |lambda_t| <= kappa * |lambda_n|^N.
        max_imp = self.kappa * np.power(lambda_n, self.friction_N)
        delta_v = max_imp / m
        scale = np.where(
            v_t_mag > 1e-12,
            np.minimum(1.0, delta_v / v_t_mag),
            0.0,
        )
        v_m = v_m - scale[:, None] * v_t

        # Normal projection: remove inward velocity and Baumgarte correct position.
        v_m = v_m - (v_n - pen_m / self.dt)[:, None] * n_m
        x = x.copy()
        x[mask] = x[mask] + pen_m[:, None] * n_m
        v[mask] = v_m
        return x, v

    # ------------------------------------------------------------------ #
    # Kinematic target
    # ------------------------------------------------------------------ #
    def _driven_target(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        if self.target_positions is None:
            return self.x[self.driven].copy(), np.zeros(3)
        if t >= self.duration:
            return self.target_positions[-1].copy(), self.target_velocities[-1].copy()
        idx = int(t // self.sample_dt)
        idx = max(0, min(idx, len(self.target_positions) - 2))
        alpha = (t - idx * self.sample_dt) / self.sample_dt
        pos = (1.0 - alpha) * self.target_positions[idx] + alpha * self.target_positions[idx + 1]
        vel = (1.0 - alpha) * self.target_velocities[idx] + alpha * self.target_velocities[idx + 1]
        return pos, vel

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    def _compute_angular_velocity(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        # Angular velocity of each edge, averaged to vertices.
        edges = x[1:] - x[:-1]
        v_diff = v[1:] - v[:-1]
        denom = np.sum(edges * edges, axis=1) + 1e-12
        omega_edges = np.cross(edges, v_diff) / denom[:, None]

        omega = np.zeros_like(v)
        omega[1:-1] = 0.5 * (omega_edges[:-1] + omega_edges[1:])
        omega[0] = omega_edges[0]
        omega[-1] = omega_edges[-1]
        return omega

    def _compute_inertia(self, x: np.ndarray) -> np.ndarray:
        n = self.n
        I_local = np.zeros((n, 3, 3))
        # Each vertex represents a cylindrical segment of rest length l0.
        m = self.m
        r = self.cable_r
        l0 = self.l0
        I_axial = 0.5 * m * r * r
        I_trans = (1.0 / 12.0) * m * (3.0 * r * r + l0 * l0)
        I_local[:, 0, 0] = I_trans
        I_local[:, 1, 1] = I_trans
        I_local[:, 2, 2] = I_axial

        # Tangents.
        tangents = np.zeros_like(x)
        edges = x[1:] - x[:-1]
        t_edge = edges / (np.linalg.norm(edges, axis=1, keepdims=True) + 1e-12)
        tangents[1:-1] = 0.5 * (t_edge[:-1] + t_edge[1:])
        tangents[0] = t_edge[0]
        tangents[-1] = t_edge[-1]
        tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-12

        # Build local frame with local z along tangent.
        world_z = np.array([0.0, 0.0, 1.0])
        I_world = np.zeros((n, 3, 3))
        for i in range(n):
            t = tangents[i]
            if np.linalg.norm(t - world_z) < 1e-6:
                u = np.array([1.0, 0.0, 0.0])
            else:
                u = np.cross(t, world_z)
                u /= np.linalg.norm(u) + 1e-12
            v_axis = np.cross(t, u)
            v_axis /= np.linalg.norm(v_axis) + 1e-12
            R = np.stack([u, v_axis, t], axis=1)  # columns are local axes
            I_world[i] = R @ I_local[i] @ R.T
        return I_world

    def _frame(self, time: float) -> StateFrame:
        x = self.x
        v = self.v
        m = self.m

        ang_vel = self._compute_angular_velocity(x, v)
        inertia = self._compute_inertia(x)

        # Budget fields: one global domain, one external-force term (gravity).
        com = (m[:, None] * x).sum(axis=0) / m.sum()
        gravity_force = (m[:, None] * self.g).sum(axis=0)
        external_force = gravity_force.reshape(1, 1, 3)
        external_couple = np.zeros((1, 1, 3))
        force_position = com.reshape(1, 1, 3)

        fields = {
            "budget_cumulative_external_angular_impulse": self.cum_external_angular_impulse.reshape(1, 3),
            "budget_cumulative_external_impulse": self.cum_external_impulse.reshape(1, 3),
            "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3)),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)),
            "budget_external_couple": external_couple,
            "budget_external_force": external_force,
            "budget_force_position": force_position,
            "cumulative_mass_source": np.zeros(1),
        }

        attributes = {
            "inertia_tensor": inertia,
            "mass_quantity": m,
        }

        scalars = {
            "reference_momentum": float(self.ref_p),
            "reference_angular_momentum": float(self.ref_L),
        }

        return StateFrame(
            time=time,
            position=x,
            velocity=v,
            mass=m,
            angular_velocity=ang_vel,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
