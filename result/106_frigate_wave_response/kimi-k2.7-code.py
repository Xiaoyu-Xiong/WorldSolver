"""Physics solver for the rigid frigate responding to incoming water waves.

The implementation uses a particle-based free-surface water field coupled to a
single rigid hull.  The fluid is driven by a prescribed incompressible Airy
wave; water particles are advected by the wave velocity field and repelled by
the impermeable hull.  The frigate is integrated as a rigid body with only the
scaffold-declared degrees of freedom (heave and pitch) using hydrostatic
pressure forces and a restoring offset calibrated to the initial waterline.

All trajectory data are exported through the standard StateFrame/trajectory
fields required by the platform.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        """Set up the coupled fluid/rigid state from the frozen scene."""
        self._rng = np.random.default_rng(seed)
        self._scene = scene
        self._config = scene.config

        # ----- physical constants from the scaffold --------------------------------
        self._g_vec = np.asarray(self._config["gravity"], dtype=float)
        self._g: float = float(np.linalg.norm(self._g_vec))
        self._rho0: float = float(self._config["fluid"]["density"])
        self._h: float = float(self._config["fluid"]["marker_spacing"])

        # wave parameters
        wave = self._config["incoming_waves"]
        self._wave_amp: float = float(wave["amplitude"])
        self._wave_k: float = 2.0 * np.pi / float(wave["wavelength"])
        self._wave_omega: float = 2.0 * np.pi / float(wave["period"])
        self._wave_phase: float = float(wave["phase_radians"])
        self._wave_dir = np.asarray(wave["propagation_direction"], dtype=float)
        self._wave_dir /= np.linalg.norm(self._wave_dir)
        self._H0: float = float(self._config["fluid"]["still_water_level"])

        # ship parameters
        ship = self._config["frigate"]
        self._ship_mass: float = float(ship["mass"])
        self._ship_z: float = float(ship["initial_center"][2])
        self._ship_zd: float = 0.0
        self._ship_theta: float = 0.0  # pitch about y
        self._ship_thetad: float = 0.0
        self._ship_x0: float = 0.0
        self._ship_y0: float = 0.0

        # initial state from the scene
        init: StateFrame = scene.initial_state()
        self._pos = np.asarray(init.position, dtype=float).copy()
        self._vel = np.asarray(init.velocity, dtype=float).copy()
        self._mass = np.asarray(init.mass, dtype=float).copy()
        self._orientation = np.asarray(init.orientation, dtype=float).copy()
        self._ang_vel = np.asarray(init.angular_velocity, dtype=float).copy()
        self._body_id = np.asarray(init.body_id, dtype=np.int64).copy()
        self._attributes = {
            name: np.asarray(arr, dtype=float).copy()
            for name, arr in init.attributes.items()
        }
        self._scalars = dict(init.scalars)

        self._is_ship = self._body_id == int(ship["body_id"])
        self._is_water = ~self._is_ship
        self._water_idx = np.nonzero(self._is_water)[0]
        self._ship_idx = np.nonzero(self._is_ship)[0][0]

        # ----- hull geometry (matches the analytic exclusion in scene.py) ----------
        self._hull_L: float = 4.35  # half-length
        self._hull_w0: float = 0.92
        self._hull_alpha: float = 0.35
        self._hull_top_local: float = 0.42
        self._hull_damping_z: float = 4_000.0
        self._hull_damping_theta: float = 30_000.0

        # Moment of inertia for pitch (box approximation)
        hull_Iy = self._ship_mass / 12.0 * ((2.0 * self._hull_L) ** 2 + (1.54) ** 2)
        self._ship_Iy: float = hull_Iy

        # Build a fixed set of hull surface samples for pressure integration.
        self._hull_samples = self._build_hull_samples()

        # Calibrate a constant offset so the ship starts in vertical equilibrium.
        self._buoyancy_offset = self._calibrate_buoyancy_offset()

        # ----- simulation bookkeeping ---------------------------------------------
        self._time: float = 0.0
        self._dt_sub: float = 0.005  # internal substep

        # Budget / bookkeeping arrays
        self._G = 2  # domains: ship, water
        self._K = 2  # phases/materials: ship, water
        self._cum_external_impulse = np.zeros((self._G, 3), dtype=float)
        self._cum_inflow_momentum = np.zeros((self._G, 3), dtype=float)
        self._cum_outflow_momentum = np.zeros((self._G, 3), dtype=float)
        self._cum_mass_source = np.zeros(self._K, dtype=float)

        # Boundary samples (static walls + wave boundary)
        self._boundary_pos, self._boundary_normal = self._build_boundary_samples()

        # Reference scales
        total_mass = float(np.sum(self._mass))
        self._ref_velocity = float(np.sqrt(self._g * self._H0))
        self._ref_momentum = total_mass * self._ref_velocity

        self._diagnostics: dict[str, object] = {
            "solver": "frigate_wave_response",
            "num_particles": len(self._pos),
            "num_water": int(self._is_water.sum()),
            "num_ship": int(self._is_ship.sum()),
            "dt_sub": self._dt_sub,
        }

    # --------------------------------------------------------------------------
    # Hull helpers
    # --------------------------------------------------------------------------
    def _hull_halfwidth(self, x: np.ndarray) -> np.ndarray:
        abs_x = np.abs(x)
        shape = np.clip(1.0 - abs_x / self._hull_L, 0.0, None) ** self._hull_alpha
        return self._hull_w0 * shape

    def _hull_bottom(self, x: np.ndarray) -> np.ndarray:
        return -1.12 + 0.16 * np.abs(x) / self._hull_L

    def _build_hull_samples(self) -> dict:
        """Create interior normal samples and area weights for pressure integration."""
        nx = 200
        nz = 20
        x = np.linspace(-self._hull_L + 1e-6, self._hull_L - 1e-6, nx)
        dx = x[1] - x[0]
        w = self._hull_halfwidth(x)
        b = self._hull_bottom(x)

        pos_list: list[np.ndarray] = []
        n_list: list[np.ndarray] = []
        dA_list: list[np.ndarray] = []

        # Bottom strip: normal points into the ship (upward)
        pos_list.append(np.column_stack((x, np.zeros_like(x), b)))
        n_list.append(np.tile(np.array([0.0, 0.0, 1.0]), (nx, 1)))
        dA_list.append(2.0 * w * dx)

        # Two side strips: normals point inward (-y and +y)
        z_side = np.linspace(0.0, 1.0, nz)
        for sign in (-1.0, 1.0):
            for zf in z_side:
                z_loc = b + zf * (self._hull_top_local - b)
                pos_list.append(np.column_stack((x, sign * w, z_loc)))
                n_list.append(np.tile(np.array([0.0, -sign, 0.0]), (nx, 1)))
                dA_list.append(np.full(nx, dx * (self._hull_top_local - b) / nz))

        pos = np.vstack(pos_list)
        normal = np.vstack(n_list)
        dA = np.hstack(dA_list)
        return {
            "pos_local": pos,
            "normal_local": normal,
            "dA": dA,
        }

    def _calibrate_buoyancy_offset(self) -> float:
        """Return the constant vertical force needed for initial static equilibrium."""
        F, _ = self._hull_pressure_force(0.0)
        return self._ship_mass * self._g - F[2]

    # --------------------------------------------------------------------------
    # Wave helpers
    # --------------------------------------------------------------------------
    def _surface_elevation(self, x: np.ndarray, t: float) -> np.ndarray:
        return self._H0 + self._wave_amp * np.sin(
            self._wave_k * x - self._wave_omega * t + self._wave_phase
        )

    def _wave_velocity(self, pos: np.ndarray, t: float) -> np.ndarray:
        """Airy-wave orbital velocity for the prescribed wave train."""
        x = pos[:, 0]
        z = pos[:, 2]
        theta = self._wave_k * x - self._wave_omega * t + self._wave_phase
        denom = np.sinh(self._wave_k * self._H0)
        # Clamp vertical coordinate above the seabed for numerical safety.
        z_eff = np.clip(z + self._H0, 0.0, None)
        coshk = np.cosh(self._wave_k * z_eff) / denom
        sinhk = np.sinh(self._wave_k * z_eff) / denom
        u = self._wave_amp * self._wave_omega * coshk * np.cos(theta)
        w = self._wave_amp * self._wave_omega * sinhk * np.sin(theta)
        out = np.zeros_like(pos)
        out[:, 0] = u
        out[:, 2] = w
        # Project onto the prescribed propagation direction (only x is non-zero).
        out[:, 0] *= self._wave_dir[0]
        return out

    # --------------------------------------------------------------------------
    # Boundary samples
    # --------------------------------------------------------------------------
    def _build_boundary_samples(self) -> tuple[np.ndarray, np.ndarray]:
        samples: list[tuple[np.ndarray, np.ndarray]] = []

        # Seabed at z = 0, normal upward from fluid
        xs = np.arange(-14.0, 15.0, 3.0)
        ys = np.arange(-5.0, 6.0, 3.0)
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        seabed = np.column_stack((xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())))
        samples.append((seabed, np.tile(np.array([0.0, 0.0, 1.0]), (len(seabed), 1))))

        # Port / starboard walls
        for y_wall, n_y in [(6.0, 1.0), (-6.0, -1.0)]:
            zs = np.arange(0.0, 2.1, 0.6)
            xx, zz = np.meshgrid(xs, zs, indexing="ij")
            wall = np.column_stack((xx.ravel(), np.full_like(xx.ravel(), y_wall), zz.ravel()))
            samples.append((wall, np.tile(np.array([0.0, n_y, 0.0]), (len(wall), 1))))

        # Inflow wave boundary at x = -15
        ys_in = np.arange(-5.0, 6.0, 3.0)
        zs_in = np.arange(0.0, 2.1, 0.6)
        yy, zz = np.meshgrid(ys_in, zs_in, indexing="ij")
        inflow = np.column_stack((np.full_like(yy.ravel(), -15.0), yy.ravel(), zz.ravel()))
        samples.append((inflow, np.tile(np.array([-1.0, 0.0, 0.0]), (len(inflow), 1))))

        pos = np.vstack([s[0] for s in samples])
        normal = np.vstack([s[1] for s in samples])
        return pos, normal

    # --------------------------------------------------------------------------
    # Ship dynamics
    # --------------------------------------------------------------------------
    def _rotation_matrix_y(self, theta: float) -> np.ndarray:
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])

    def _hull_pressure_force(self, t: float) -> tuple[np.ndarray, float]:
        """Return total pressure force and pitch torque on the hull at time t."""
        R = self._rotation_matrix_y(self._ship_theta)
        pos_local = self._hull_samples["pos_local"]
        normal_local = self._hull_samples["normal_local"]
        dA = self._hull_samples["dA"]

        pos_world = pos_local @ R.T + np.array([0.0, 0.0, self._ship_z])
        normal_world = normal_local @ R.T

        eta = self._surface_elevation(pos_world[:, 0], t)
        pressure = self._rho0 * self._g * np.maximum(0.0, eta - pos_world[:, 2])

        forces = pressure[:, None] * normal_world * dA[:, None]
        F = forces.sum(axis=0)

        r = pos_world - np.array([0.0, 0.0, self._ship_z])
        torque = np.cross(r, forces).sum(axis=0)
        return F, torque[1]

    def _integrate_ship(self, t: float, dt: float) -> None:
        F_pressure, M_pressure = self._hull_pressure_force(t)

        # Vertical dynamics (heave)
        F_gravity = -self._ship_mass * self._g
        F_damp = -self._hull_damping_z * self._ship_zd
        F_net_z = F_pressure[2] + F_gravity + self._buoyancy_offset + F_damp
        self._ship_zd += dt * (F_net_z / self._ship_mass)
        self._ship_z += dt * self._ship_zd

        # Pitch dynamics
        M_damp = -self._hull_damping_theta * self._ship_thetad
        M_net = M_pressure + M_damp
        self._ship_thetad += dt * (M_net / self._ship_Iy)
        self._ship_theta += dt * self._ship_thetad

    # --------------------------------------------------------------------------
    # Fluid / hull coupling
    # --------------------------------------------------------------------------
    def _apply_hull_repulsion(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> None:
        """Push water particles out of the impermeable hull volume."""
        # World -> local ship coordinates
        dx = pos[self._water_idx, 0] - self._ship_x0
        dy = pos[self._water_idx, 1] - self._ship_y0
        dz = pos[self._water_idx, 2] - self._ship_z

        ct = np.cos(self._ship_theta)
        st = np.sin(self._ship_theta)
        x_l = dx * ct + dz * st
        y_l = dy
        z_l = -dx * st + dz * ct

        abs_x = np.abs(x_l)
        in_len = abs_x < self._hull_L
        w = self._hull_halfwidth(x_l)
        b = self._hull_bottom(x_l)
        top = self._hull_top_local

        py = w - np.abs(y_l)
        pz = np.minimum(z_l - b, top - z_l)

        inside = in_len & (py > 0.0) & (pz > 0.0)
        if not np.any(inside):
            return

        py = np.where(inside, py, 0.0)
        pz = np.where(inside, pz, 0.0)

        # Build local outward normal pointing to the nearest face.
        n_local = np.zeros((len(x_l), 3), dtype=float)
        side_closer = py < pz
        # Side faces
        idx = inside & side_closer
        n_local[idx, 1] = -np.sign(y_l[idx])
        # Bottom / top faces
        bot_closer = inside & (~side_closer) & (z_l < (b + top) * 0.5)
        top_closer = inside & (~side_closer) & ~bot_closer
        n_local[bot_closer, 2] = -1.0
        n_local[top_closer, 2] = 1.0

        pen = np.minimum(py, pz)
        # Rotate local normal to world (only y-rotation)
        n_world = np.zeros_like(n_local)
        n_world[:, 0] = n_local[:, 2] * st + n_local[:, 0] * ct
        n_world[:, 1] = n_local[:, 1]
        n_world[:, 2] = n_local[:, 2] * ct - n_local[:, 0] * st

        speed = np.clip(pen / dt + 2.0, 0.0, 15.0)
        corrections = n_world * speed[:, None]
        vel[self._water_idx] += inside[:, None] * corrections

    def _apply_domain_boundaries(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> None:
        """Reflective clamps for the static tank walls."""
        # x boundaries
        left = pos[self._water_idx, 0] < -14.98
        right = pos[self._water_idx, 0] > 14.98
        vel[self._water_idx[left], 0] = np.maximum(vel[self._water_idx[left], 0], 0.0) + (
            (-14.98 - pos[self._water_idx[left], 0]) / dt
        )
        vel[self._water_idx[right], 0] = np.minimum(vel[self._water_idx[right], 0], 0.0) - (
            (pos[self._water_idx[right], 0] - 14.98) / dt
        )

        # y boundaries
        top = pos[self._water_idx, 1] > 5.98
        bot = pos[self._water_idx, 1] < -5.98
        vel[self._water_idx[top], 1] = np.minimum(vel[self._water_idx[top], 1], 0.0) - (
            (pos[self._water_idx[top], 1] - 5.98) / dt
        )
        vel[self._water_idx[bot], 1] = np.maximum(vel[self._water_idx[bot], 1], 0.0) + (
            (-5.98 - pos[self._water_idx[bot], 1]) / dt
        )

        # seabed (z = 0)
        below = pos[self._water_idx, 2] < 0.02
        vel[self._water_idx[below], 2] = np.maximum(vel[self._water_idx[below], 2], 0.0) + (
            (0.02 - pos[self._water_idx[below], 2]) / dt
        )

    # --------------------------------------------------------------------------
    # Frame construction
    # --------------------------------------------------------------------------
    def _build_mass_quantity(self) -> np.ndarray:
        n = len(self._mass)
        mq = np.zeros((n, 2), dtype=float)
        ship_mass = self._mass[self._ship_idx]
        mq[self._ship_idx, 0] = ship_mass
        mq[self._water_idx, 1] = self._mass[self._water_idx]
        return mq

    def _interaction_force(self) -> np.ndarray:
        """Return [ship, water] interaction force pair."""
        F, _ = self._hull_pressure_force(self._time)
        return np.stack([F, -F], axis=0)

    def _budget_external_force(self) -> np.ndarray:
        """Gravity as the external force for the two budget domains."""
        F = np.zeros((self._G, 1, 3), dtype=float)
        # Ship domain
        F[0, 0, :] = self._ship_mass * self._g_vec
        # Water domain
        water_mass = np.sum(self._mass[self._water_idx])
        F[1, 0, :] = water_mass * self._g_vec
        return F

    def _build_boundary_velocities(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Return boundary_velocity and boundary_target_velocity arrays."""
        target = np.zeros_like(self._boundary_pos)
        # Inflow boundary: prescribed wave velocity
        inflow_mask = self._boundary_pos[:, 0] <= -14.99
        if np.any(inflow_mask):
            target[inflow_mask] = self._wave_velocity(self._boundary_pos[inflow_mask], t)
        # Static walls have zero velocity
        velocity = np.zeros_like(self._boundary_pos)
        return velocity, target

    def _make_frame(self) -> StateFrame:
        # Ensure the ship particle reflects the rigid body state.
        self._pos[self._ship_idx] = np.array([self._ship_x0, self._ship_y0, self._ship_z])
        half = self._ship_theta / 2.0
        self._orientation[self._ship_idx] = np.array(
            [0.0, np.sin(half), 0.0, np.cos(half)]
        )
        self._vel[self._ship_idx] = np.array([0.0, 0.0, self._ship_zd])
        self._ang_vel[self._ship_idx] = np.array([0.0, self._ship_thetad, 0.0])

        # Budget and boundary fields
        boundary_vel, boundary_target = self._build_boundary_velocities(self._time)
        ext_force = self._budget_external_force()
        interaction = self._interaction_force()

        attributes = dict(self._attributes)
        attributes["mass_quantity"] = self._build_mass_quantity()

        fields = {
            "boundary_normal": self._boundary_normal.copy(),
            "boundary_target_velocity": boundary_target,
            "boundary_velocity": boundary_vel,
            "budget_cumulative_external_impulse": self._cum_external_impulse.copy(),
            "budget_cumulative_inflow_momentum": self._cum_inflow_momentum.copy(),
            "budget_cumulative_outflow_momentum": self._cum_outflow_momentum.copy(),
            "budget_domain_weights": self._budget_domain_weights(),
            "budget_external_force": ext_force,
            "cumulative_mass_source": self._cum_mass_source.copy(),
            "interaction_force": interaction,
        }

        scalars = dict(self._scalars)
        scalars["reference_momentum"] = self._ref_momentum
        scalars["reference_velocity"] = self._ref_velocity

        return StateFrame(
            time=self._time,
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            orientation=self._orientation.copy(),
            angular_velocity=self._ang_vel.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def _budget_domain_weights(self) -> np.ndarray:
        weights = np.zeros((self._G, len(self._mass)), dtype=float)
        weights[0, self._ship_idx] = 1.0
        weights[1, self._water_idx] = 1.0
        return weights

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        dt_frame = time - self._time
        if dt_frame < 0.0:
            raise ValueError("advance_to called with a time earlier than current state")

        n_sub = max(1, int(np.ceil(dt_frame / self._dt_sub)))
        dt = dt_frame / n_sub

        for _ in range(n_sub):
            # Fluid: advect with the prescribed wave field plus hull repulsion.
            v_wave = self._wave_velocity(self._pos[self._water_idx], self._time)
            self._vel[self._water_idx] = v_wave
            self._apply_hull_repulsion(self._pos, self._vel, dt)
            self._apply_domain_boundaries(self._pos, self._vel, dt)
            self._pos[self._water_idx] += self._vel[self._water_idx] * dt

            # Rigid ship: hydrostatic pressure integration.
            self._integrate_ship(self._time, dt)

            # Update budgets.
            ext = self._budget_external_force()
            self._cum_external_impulse += ext[:, 0, :] * dt
            self._time += dt

        # Ensure final ship particle is consistent before returning the frame.
        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return self._diagnostics
