"""Finite-volume air / rigid-rotor solver for the wind-turbine scene.

The gas is represented on the frozen voxel grid.  A first-order upwind finite
volume step transports velocity and passive smoke, while an actuator disk
exchanges axial momentum and angular momentum with a one-degree-of-freedom
rigid rotor.  Rotor speed is obtained by integrating angular momentum.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic two-way coupled actuator-disk solver."""

    def initialize(self, scene, seed: int) -> None:
        del seed
        self.scene = scene
        cfg = scene.config
        self.cfg = cfg

        domain = cfg["domain"]
        turbine = cfg["turbine"]
        air = cfg["air"]
        self.shape = tuple(int(v) for v in domain["reference_grid"])
        self.lo = np.asarray(domain["bounds_min"], dtype=np.float64)
        self.hi = np.asarray(domain["bounds_max"], dtype=np.float64)
        self.spacing = (self.hi - self.lo) / (np.asarray(self.shape) - 1.0)
        self.x = np.linspace(self.lo[0], self.hi[0], self.shape[0])
        self.y = np.linspace(self.lo[1], self.hi[1], self.shape[1])
        self.z = np.linspace(self.lo[2], self.hi[2], self.shape[2])

        self.rho = float(air["density"])
        self.velocity = np.zeros((*self.shape, 3), dtype=np.float32)
        self.density = np.full(self.shape, self.rho, dtype=np.float32)
        self.smoke = np.zeros((3, *self.shape), dtype=np.float32)

        self.hub = np.asarray(turbine["hub_center"], dtype=np.float64)
        axis = np.asarray(turbine["axle_axis"], dtype=np.float64)
        self.axis = axis / np.linalg.norm(axis)
        self.radius = float(turbine["rotor_radius"])
        self.rotor_mass = float(turbine["mass"])
        # Polar inertia of three tapered radial blades and their hub.
        self.inertia_x = 0.32 * self.rotor_mass * self.radius**2
        self.inertia_tensor = np.diag(
            [self.inertia_x, 0.505 * self.inertia_x, 0.505 * self.inertia_x]
        ).reshape(1, 3, 3)

        self.angle = float(turbine["initial_angle"])
        self.omega = float(turbine["initial_angular_velocity"])
        friction = turbine["axle_friction"]
        self.friction_b = float(friction["viscous_coefficient"])
        self.friction_c = float(friction["coulomb_torque"])
        self.friction_eps = float(friction["regularization_speed"])

        self.tip_local = np.asarray(
            cfg["smoke"]["rotor_local_tip_positions"], dtype=np.float64
        )
        self.emission_start = float(cfg["smoke"]["emission_start"])
        self.emission_end = float(cfg["smoke"]["emission_end"])
        self.source_rate = float(cfg["smoke"]["source_rate_each"])

        self.rotor_ix = int(np.argmin(np.abs(self.x - self.hub[0])))
        yy, zz = np.meshgrid(
            self.y - self.hub[1], self.z - self.hub[2], indexing="ij"
        )
        self.disk_y = yy
        self.disk_z = zz
        self.disk_r2 = yy * yy + zz * zz
        self.disk_mask = self.disk_r2 <= self.radius**2
        self.disk_area_cells = int(np.count_nonzero(self.disk_mask))
        self.actuator_x_indices = np.arange(
            max(1, self.rotor_ix - 1), min(self.shape[0] - 1, self.rotor_ix + 2)
        )

        self.time = 0.0
        self.max_dt = min(
            float(cfg["sample_dt"]), 0.95 * self.spacing[0] / 6.0
        )
        self.cumulative_angular_impulse = 0.0
        self.last_aero_torque = 0.0
        self.last_friction_torque = 0.0
        self.last_thrust = 0.0
        self.step_count = 0

    def _inlet_velocity(self, time: float) -> np.ndarray:
        return np.asarray(self.scene.inlet_velocity(float(time)), dtype=np.float32)

    def _disk_inflow(self) -> float:
        ix = max(0, self.rotor_ix - 2)
        values = self.velocity[ix, :, :, 0][self.disk_mask]
        return float(np.clip(np.mean(values), 0.0, 8.0))

    def _aerodynamic_loads(self, inflow: float) -> tuple[float, float]:
        """Low-order blade-element power and thrust curves."""
        if inflow <= 1.0e-8:
            return 0.0, 0.0
        swept_area = math.pi * self.radius**2
        tip_speed_ratio = max(0.0, self.omega) * self.radius / inflow
        free_running_tsr = 4.5
        cq = 0.035 * max(0.0, 1.0 - tip_speed_ratio / free_running_tsr)
        torque = (
            0.5 * self.rho * swept_area * self.radius * cq * inflow**2
        )
        ct = 0.72 * max(0.25, 1.0 - 0.055 * tip_speed_ratio)
        thrust = 0.5 * self.rho * swept_area * ct * inflow**2
        return torque, thrust

    def _advect_velocity_x(self, dt: float) -> None:
        """Conservative donor-cell transport by the axial wind."""
        delta = self.velocity[1:] - self.velocity[:-1]
        courant = np.clip(
            self.velocity[1:, :, :, 0] * (dt / self.spacing[0]), 0.0, 0.98
        )
        self.velocity[1:] -= courant[..., None] * delta

    def _apply_actuator_reaction(
        self, dt: float, torque: float, thrust: float
    ) -> None:
        """Apply equal-and-opposite blade loads to the resolved gas."""
        cell_volume = float(np.prod(self.spacing))
        nplanes = len(self.actuator_x_indices)
        volume = self.disk_area_cells * nplanes * cell_volume
        axial_accel = thrust / (self.rho * max(volume, 1.0e-12))
        polar_volume = (
            nplanes * cell_volume * float(np.sum(self.disk_r2[self.disk_mask]))
        )
        swirl_factor = torque / (self.rho * max(polar_volume, 1.0e-12))
        for ix in self.actuator_x_indices:
            vx = self.velocity[ix, :, :, 0]
            vy = self.velocity[ix, :, :, 1]
            vz = self.velocity[ix, :, :, 2]
            vx[self.disk_mask] -= dt * axial_accel
            # This tangential force deposits exactly -torque in the gas.
            vy[self.disk_mask] += dt * swirl_factor * self.disk_z[self.disk_mask]
            vz[self.disk_mask] -= dt * swirl_factor * self.disk_y[self.disk_mask]

    def _advect_smoke(self, dt: float) -> None:
        """Dimensionally split first-order upwind passive-scalar step."""
        c = self.smoke
        cx = c[:, 1:] - c[:, :-1]
        ux = np.clip(self.velocity[1:, :, :, 0], 0.0, None)
        c[:, 1:] -= (dt / self.spacing[0]) * ux[None, ...] * cx

        vy = self.velocity[:, 1:-1, :, 1]
        back_y = c[:, :, 1:-1, :] - c[:, :, :-2, :]
        fwd_y = c[:, :, 2:, :] - c[:, :, 1:-1, :]
        grad_y = np.where(vy[None, ...] >= 0.0, back_y, fwd_y)
        c[:, :, 1:-1, :] -= (dt / self.spacing[1]) * vy[None, ...] * grad_y

        vz = self.velocity[:, :, 1:-1, 2]
        back_z = c[:, :, :, 1:-1] - c[:, :, :, :-2]
        fwd_z = c[:, :, :, 2:] - c[:, :, :, 1:-1]
        grad_z = np.where(vz[None, ...] >= 0.0, back_z, fwd_z)
        c[:, :, :, 1:-1] -= (dt / self.spacing[2]) * vz[None, ...] * grad_z

        c[:, 0, :, :] = 0.0
        c[:, :, 0, :] = 0.0
        c[:, :, -1, :] = 0.0
        c[:, :, :, 0] = c[:, :, :, 1]
        c[:, :, :, -1] = 0.0
        np.clip(c, 0.0, 4.0, out=c)

    def _rotated_tips(self) -> np.ndarray:
        co = math.cos(self.angle)
        si = math.sin(self.angle)
        tips = self.tip_local.copy()
        y0 = self.tip_local[:, 1]
        z0 = self.tip_local[:, 2]
        tips[:, 1] = co * y0 - si * z0
        tips[:, 2] = si * y0 + co * z0
        return tips + self.hub

    def _emit_smoke(self, dt: float, midpoint_time: float) -> None:
        if midpoint_time < self.emission_start or midpoint_time > self.emission_end:
            return
        amount = dt * self.source_rate / 0.08
        for channel, point in enumerate(self._rotated_tips()):
            grid = (point - self.lo) / self.spacing
            base = np.floor(grid).astype(int)
            frac = grid - base
            for ox in (0, 1):
                ix = base[0] + ox
                if ix < 0 or ix >= self.shape[0]:
                    continue
                wx = frac[0] if ox else 1.0 - frac[0]
                for oy in (0, 1):
                    iy = base[1] + oy
                    if iy < 0 or iy >= self.shape[1]:
                        continue
                    wy = frac[1] if oy else 1.0 - frac[1]
                    for oz in (0, 1):
                        iz = base[2] + oz
                        if iz < 0 or iz >= self.shape[2]:
                            continue
                        wz = frac[2] if oz else 1.0 - frac[2]
                        self.smoke[channel, ix, iy, iz] += (
                            amount * wx * wy * wz
                        )

    def _step(self, dt: float) -> None:
        midpoint = self.time + 0.5 * dt
        self._advect_velocity_x(dt)
        self.velocity[0, :, :, :] = self._inlet_velocity(midpoint)

        inflow = self._disk_inflow()
        aero_torque, thrust = self._aerodynamic_loads(inflow)
        friction_torque = (
            -self.friction_b * self.omega
            - self.friction_c * math.tanh(self.omega / self.friction_eps)
        )
        net_torque = aero_torque + friction_torque
        old_omega = self.omega
        new_omega = old_omega + dt * net_torque / self.inertia_x
        if new_omega < 0.0 and old_omega >= 0.0:
            new_omega = 0.0
            net_torque = (new_omega - old_omega) * self.inertia_x / dt
        self.omega = new_omega
        self.angle = (
            self.angle + 0.5 * dt * (old_omega + self.omega)
        ) % (2.0 * math.pi)
        self.cumulative_angular_impulse += dt * net_torque

        self._apply_actuator_reaction(dt, aero_torque, thrust)
        np.clip(
            self.velocity[:, :, :, 0], 0.0, 8.0,
            out=self.velocity[:, :, :, 0]
        )
        np.clip(
            self.velocity[:, :, :, 1:], -4.0, 4.0,
            out=self.velocity[:, :, :, 1:]
        )
        self.velocity[:, :, 0, :] = 0.0
        self.velocity[:, 0, :, :] = self.velocity[:, 1, :, :]
        self.velocity[:, -1, :, :] = self.velocity[:, -2, :, :]
        self.velocity[:, :, -1, :] = self.velocity[:, :, -2, :]

        self._advect_smoke(dt)
        self._emit_smoke(dt, midpoint)
        self.time += dt
        self.last_aero_torque = aero_torque
        self.last_friction_torque = friction_torque
        self.last_thrust = thrust
        self.step_count += 1

    def _blade_forces_and_positions(self) -> tuple[np.ndarray, np.ndarray]:
        positions = self._rotated_tips()
        rel = positions - self.hub
        radii = np.sqrt(rel[:, 1] ** 2 + rel[:, 2] ** 2)
        tangent = np.zeros((3, 3), dtype=np.float64)
        tangent[:, 1] = -rel[:, 2] / radii
        tangent[:, 2] = rel[:, 1] / radii
        tangential_each = self.last_aero_torque / float(np.sum(radii))
        forces = tangential_each * tangent
        forces[:, 0] = self.last_thrust / 3.0
        return forces, positions

    def _frame(self) -> StateFrame:
        half = 0.5 * self.angle
        orientation = np.array(
            [[math.sin(half), 0.0, 0.0, math.cos(half)]], dtype=np.float64
        )
        angular_velocity = (self.omega * self.axis).reshape(1, 3)
        blade_forces, blade_positions = self._blade_forces_and_positions()

        budget_forces = np.zeros((1, 4, 3), dtype=np.float64)
        budget_positions = np.zeros((1, 4, 3), dtype=np.float64)
        budget_couples = np.zeros((1, 4, 3), dtype=np.float64)
        budget_forces[0, :3] = blade_forces
        budget_positions[0, :3] = blade_positions
        budget_forces[0, 3, 0] = -self.last_thrust
        budget_positions[0, 3] = self.hub
        budget_couples[0, 0] = self.last_friction_torque * self.axis

        inlet = self._inlet_velocity(self.time).astype(np.float64)
        normals = np.array(
            [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
             [0, 0, 1], [0, 0, -1]],
            dtype=np.float64,
        )
        boundary_target = np.zeros((6, 3), dtype=np.float64)
        boundary_target[0] = inlet
        boundary_velocity = boundary_target.copy()
        inlet_speed = float(np.linalg.norm(self.cfg["air"]["inlet"]["velocity"]))

        frame = StateFrame(
            time=float(self.time),
            position=self.hub.reshape(1, 3).copy(),
            velocity=np.zeros((1, 3), dtype=np.float64),
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=np.array([self.rotor_mass], dtype=np.float64),
            body_id=np.array(
                [int(self.cfg["turbine"]["rotor_body_id"])], dtype=np.int64
            ),
            attributes={
                "rotor_angle": np.array([self.angle], dtype=np.float64),
                "inertia_tensor": self.inertia_tensor.copy(),
                "mass_quantity": np.array([self.rotor_mass], dtype=np.float64),
            },
            fields={
                "air_density": self.density,
                "velocity_field": self.velocity,
                "smoke_red": self.smoke[0],
                "smoke_green": self.smoke[1],
                "smoke_ochre": self.smoke[2],
                "grid_origin": self.lo.copy(),
                "rotor_local_tip_positions": self.tip_local.copy(),
                "boundary_normal": normals,
                "boundary_target_velocity": boundary_target,
                "boundary_velocity": boundary_velocity,
                "budget_cumulative_external_angular_impulse": np.array(
                    [[self.cumulative_angular_impulse, 0.0, 0.0]],
                    dtype=np.float64,
                ),
                "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3)),
                "budget_domain_weights": np.ones((1, 1), dtype=np.float64),
                "budget_external_couple": budget_couples,
                "budget_external_force": budget_forces,
                "budget_force_position": budget_positions,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "interaction_force": blade_forces,
            },
            scalars={
                "air_density": self.rho,
                "inlet_speed": inlet_speed,
                "axle_viscous_friction": self.friction_b,
                "axle_coulomb_friction": self.friction_c,
                "rotor_radius": self.radius,
                "reference_angular_momentum": self.inertia_x
                * inlet_speed / self.radius,
                "reference_velocity": inlet_speed,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not math.isfinite(target) or target < self.time - 1.0e-10:
            raise ValueError("advance_to requires a finite, nondecreasing time")
        while self.time < target - 1.0e-12:
            self._step(min(self.max_dt, target - self.time))
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "finite_volume_upwind_actuator_disk_two_way_coupling",
            "internal_steps": int(self.step_count),
            "final_rotor_angle": float(self.angle),
            "final_rotor_angular_velocity": float(self.omega),
            "rotor_inertia": float(self.inertia_x),
        }
