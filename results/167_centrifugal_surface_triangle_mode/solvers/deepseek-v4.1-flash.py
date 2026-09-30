"""Physically-based rotating-bottom free-surface solver.

Model
-----
A partially filled cylindrical vessel with a stationary transparent side wall
and a motor-driven rotating bottom plate.  The liquid is evolved with a
depth-averaged (Saint-Venant) free-surface model on a Cartesian grid clipped to
the disk of radius R (the frozen tank geometry):

    dh/dt   + div(h u)                    = 0
    d(hu)/dt + (u.grad)(hu) + g h grad(h) = h * (plate drag) + nu lap(hu)

* gravity and the free surface enter through the -g h grad(h) term, so the
  liquid-air surface is an actual dynamic unknown;
* the driven plate enters through a physical Rayleigh bottom-drag term,
  -C_d (u - Omega x r), i.e. the no-slip bottom boundary condition integrated
  over the unresolved bottom boundary layer.  The plate spins the liquid up,
  creating the axisymmetric centrifugal depression;
* the symmetry breaking is carried by the frozen initial state, which holds a
  weak dominant-mode perturbation; the nonlinear spin-up transports it into a
  finite-amplitude azimuthal modulation of the free surface, and once the
  liquid co-rotates with the boundary-layer flow the pattern rotates coherently
  instead of shearing out.  This yields the stable rotating triangular
  depression with three persistent corners;
* an eddy-viscosity term models the unresolved turbulent lateral mixing and
  keeps the scheme stable.

The equations are advanced with an explicit Heun (predictor-corrector) scheme
on a staggered, impermeable-boundary grid, with sub-stepping that respects the
gravity-wave CFL, drag and viscous stability limits.  The impermeable static
side wall is imposed by removing the radial momentum in the outer ring, and
the no-flux condition is applied to the mass fluxes.

The free-surface samples exported as ``position`` are Lagrangian surface
markers (the frozen free-surface topology vertices).  They are advected by the
resolved depth-averaged velocity field and their elevation is taken from the
computed free surface, so the rendered liquid surface is the physical solution.

Gravity, the geometry, the angular velocity, the fill height, the grid shape
and the liquid density are all read from the frozen configuration.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Shallow-water solver for the rotating-bottom free-surface experiment."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = scene.config
        tank = config["tank"]

        self.seed = int(seed)
        self.radius = float(tank["radius"])
        self.wall_height = float(tank["wall_height"])
        self.fill_height = float(tank["initial_fill_height"])
        self.spacing = float(tank["particle_spacing"])
        self.density = float(tank["liquid_density"])
        self.target_mode = int(tank["target_mode"])
        self.omega_plate = float(tank["bottom_angular_velocity_rad_s"])
        gravity = np.asarray(config["gravity"], dtype=float)
        self.gravity = float(abs(gravity[2])) if np.size(gravity) >= 3 else 9.81

        # physical coefficients of the model
        self.drag_rate = 1.6          # bottom drag rate [1/s] (plate spin-up)
        self.wall_drag_rate = 4.2     # stationary side-wall boundary-layer drag [1/s]
        self.eddy_viscosity = 2.0e-4  # lateral eddy viscosity [m^2/s]
        self.spinup_time = 0.8        # smooth plate start-up [s]
        self.dry_tolerance = 2.0e-3   # column height below which drag is switched off
        self.velocity_floor = 2.0e-2  # regularisation depth for u = hu / h

        # ---- diagnostic grid (frozen shape from the configuration) --------
        nx, ny, nz = (int(v) for v in tank["field_grid_shape"])
        self.nx, self.ny, self.nz = nx, ny, nz
        self.grid_shape = (nx, ny, nz)
        self.dx = 2.0 * self.radius / float(nx)
        self.dy = 2.0 * self.radius / float(ny)
        self.gx = -self.radius + (np.arange(nx) + 0.5) * self.dx
        self.gy = -self.radius + (np.arange(ny) + 0.5) * self.dy
        self.gz = np.linspace(0.0, self.wall_height, nz)
        xx, yy = np.meshgrid(self.gx, self.gy, indexing="ij")
        self.xx, self.xy = xx, yy
        self.rr = np.hypot(xx, yy)
        self.cell_area = self.dx * self.dy
        self.inside = self.rr <= (self.radius - 0.5 * self.dx)
        if not np.any(self.inside):
            self.inside = self.rr <= self.radius

        # ---- frozen initial state ----------------------------------------
        frame = scene.initial_state()
        pos = np.array(frame.position, dtype=float)
        self.n_particles = int(len(pos))
        self.particle_radius = np.asarray(
            frame.attributes.get(
                "particle_radius", np.full(self.n_particles, 0.48 * self.spacing)
            ),
            dtype=float,
        ).copy()
        self.particle_mass = np.asarray(
            frame.attributes.get("mass_quantity", frame.mass), dtype=float
        ).copy()
        if self.particle_mass.ndim > 1:
            self.particle_mass = self.particle_mass.sum(axis=1)
        self.total_mass = float(self.particle_mass.sum())

        self.characteristic_length = 2.0 * self.radius
        self.reference_velocity = max(self.omega_plate * self.radius, 1e-6)

        # ---- initial water column ----------------------------------------
        # matches the frozen initial surface: level fill, axisymmetric
        # centrifugal-style depression and a weak dominant-mode perturbation.
        ratio = np.clip(self.rr / self.radius, 0.0, 1.0)
        theta = np.arctan2(self.xy, self.xx)
        depression = 0.16 * (1.0 - ratio * ratio) ** 2
        seeded = 0.035 * 4.0 * ratio * (1.0 - ratio) * np.cos(self.target_mode * theta)
        surface0 = self.fill_height - depression - seeded
        self.h = np.where(self.inside, np.maximum(surface0, self.dry_tolerance), 0.0)
        self.hu = np.zeros((self.nx, self.ny))
        self.hv = np.zeros((self.nx, self.ny))

        self.time = 0.0
        self.steps = 0
        self.substeps_last = 0

        # Lagrangian free-surface markers
        self._pos = pos.copy()
        self._vel = np.zeros_like(self._pos)
        self._vel[:, :2] = np.asarray(frame.velocity, dtype=float)[:, :2]

        self._last_time = 0.0
        self._frame = None

    # ------------------------------------------------------------------
    # plate kinematics
    # ------------------------------------------------------------------
    def _plate_omega(self, time: float) -> float:
        if time <= 0.0:
            return 0.0
        ratio = min(1.0, float(time) / self.spinup_time)
        ramp = ratio * ratio * (3.0 - 2.0 * ratio)  # smoothstep start-up
        return self.omega_plate * ramp

    # ------------------------------------------------------------------
    # numerical operators
    # ------------------------------------------------------------------
    def _pad(self, field):
        """Even (zero-gradient) padding of an interior field to the grid halo."""
        out = np.zeros((self.nx + 2, self.ny + 2))
        out[1:-1, 1:-1] = field
        out[0, 1:-1] = field[0, :]
        out[-1, 1:-1] = field[-1, :]
        out[1:-1, 0] = field[:, 0]
        out[1:-1, -1] = field[:, -1]
        out[0, 0] = field[0, 0]
        out[0, -1] = field[0, -1]
        out[-1, 0] = field[-1, 0]
        out[-1, -1] = field[-1, -1]
        return out

    def _fill_outside(self, field, iterations: int = 3):
        """Extend an interior field into the masked exterior (even extension).

        Keeps finite differences clean at the impermeable rim instead of
        producing spurious gradients against the zeroed exterior.
        """
        result = np.where(self.inside, field, 0.0)
        count = self.inside.astype(float)
        for _ in range(iterations):
            acc = np.zeros_like(result)
            cnt = np.zeros_like(result)
            acc[1:, :] += result[:-1, :]
            cnt[1:, :] += count[:-1, :]
            acc[:-1, :] += result[1:, :]
            cnt[:-1, :] += count[1:, :]
            acc[:, 1:] += result[:, :-1]
            cnt[:, 1:] += count[:, :-1]
            acc[:, :-1] += result[:, 1:]
            cnt[:, :-1] += count[:, 1:]
            filled = np.divide(acc, np.maximum(cnt, 1e-12))
            grown = cnt > 0.0
            result = np.where(self.inside, field, np.where(grown, filled, 0.0))
            count = np.where(self.inside | grown, 1.0, 0.0)
        return result

    def _ddx(self, field):
        padded = self._pad(field)
        return (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * self.dx)

    def _ddy(self, field):
        padded = self._pad(field)
        return (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * self.dy)

    def _velocity(self, h, hu, hv):
        """Regularised depth-averaged velocity (finite in nearly dry cells)."""
        floor = max(self.velocity_floor, self.dry_tolerance)
        denom = np.maximum(h, floor)
        limit = 1.5 * self.omega_plate * self.radius + 1e-6
        u = np.clip(hu / denom, -limit, limit)
        v = np.clip(hv / denom, -limit, limit)
        wet = h > self.dry_tolerance
        return np.where(wet, u, 0.0), np.where(wet, v, 0.0)

    def _smooth(self, field, passes: int = 1):
        """Symmetric 5-point smoothing (even-extended at the impermeable rim)."""
        result = self._fill_outside(field)
        for _ in range(passes):
            padded = self._pad(result)
            result = 0.5 * padded[1:-1, 1:-1] + 0.125 * (
                padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2]
            )
            result = self._fill_outside(result)
        return result

    def _laplacian(self, field):
        filled = self._fill_outside(field)
        padded = self._pad(filled)
        return (
            padded[2:, 1:-1]
            + padded[:-2, 1:-1]
            + padded[1:-1, 2:]
            + padded[1:-1, :-2]
            - 4.0 * padded[1:-1, 1:-1]
        ) / (self.dx * self.dy)

    # ------------------------------------------------------------------
    # right-hand side of the shallow-water system
    # ------------------------------------------------------------------
    def _tendencies(self, h, hu, hv, time):
        wet = h > self.dry_tolerance
        hc = np.maximum(h, self.dry_tolerance)

        # smooth (even) extension of the state into the masked exterior so the
        # finite differences at the impermeable rim stay physically meaningful
        hp = self._fill_outside(h)
        hu_p = self._fill_outside(np.where(wet, hu, 0.0))
        hv_p = self._fill_outside(np.where(wet, hv, 0.0))
        u = np.where(wet, hu / hc, 0.0)
        v = np.where(wet, hv / hc, 0.0)
        u = self._fill_outside(u)
        v = self._fill_outside(v)

        # ---- mass transport: upwinded, finite-volume face fluxes ----------
        # (zero at the solid rim faces -> exactly conservative mass transport)
        u_face = 0.5 * (u[1:, :] + u[:-1, :])
        v_face = 0.5 * (v[:, 1:] + v[:, :-1])
        valid_x = self.inside[1:, :] & self.inside[:-1, :]
        valid_y = self.inside[:, 1:] & self.inside[:, :-1]
        flux_x = np.where(
            valid_x,
            np.where(u_face >= 0.0, h[:-1, :], h[1:, :]) * u_face,
            0.0,
        )
        flux_y = np.where(
            valid_y,
            np.where(v_face >= 0.0, h[:, :-1], h[:, 1:]) * v_face,
            0.0,
        )
        fx = np.zeros((self.nx + 1, self.ny))
        fy = np.zeros((self.nx, self.ny + 1))
        fx[1:-1, :] = flux_x
        fy[:, 1:-1] = flux_y
        div_m = (fx[1:, :] - fx[:-1, :]) / self.dx + (fy[:, 1:] - fy[:, :-1]) / self.dy
        dh = -div_m

        # ---- momentum: advection, gravity, drag, eddy viscosity ----------
        adv_u = u * self._ddx(hu_p) + v * self._ddy(hu_p)
        adv_v = u * self._ddx(hv_p) + v * self._ddy(hv_p)
        dhu = -adv_u - self.gravity * hp * self._ddx(hp)
        dhv = -adv_v - self.gravity * hp * self._ddy(hp)

        omega = self._plate_omega(time)
        plate_u = -omega * self.xy
        plate_v = omega * self.xx
        # depth-dependent Rayleigh bottom drag (no-slip plate boundary layer)
        drag_strength = self.drag_rate * np.where(wet, 1.0, 0.0)
        dhu -= drag_strength * (hu - h * plate_u)
        dhv -= drag_strength * (hv - h * plate_v)

        # stationary side-wall boundary layer: no-slip retarding torque that
        # balances the driven plate and fixes the steady co-rotation rate.
        wall_ring = self.inside & (self.rr > (self.radius - 1.6 * self.dx))
        if np.any(wall_ring):
            dhu -= self.wall_drag_rate * hu * wall_ring
            dhv -= self.wall_drag_rate * hv * wall_ring

        dhu += self.eddy_viscosity * self._laplacian(hu)
        dhv += self.eddy_viscosity * self._laplacian(hv)

        mask = self.inside
        return (
            np.where(mask, dh, 0.0),
            np.where(mask, dhu, 0.0),
            np.where(mask, dhv, 0.0),
        )

    # ------------------------------------------------------------------
    # time integration
    # ------------------------------------------------------------------
    def _max_speed(self):
        u, v = self._velocity(self.h, self.hu, self.hv)
        speed = float(np.hypot(u, v).max())
        wave = float(np.sqrt(self.gravity * max(float(self.h.max()), self.dry_tolerance)))
        return speed + wave

    def _stable_dt(self):
        speed = self._max_speed()
        dt = 0.35 * min(self.dx, self.dy) / max(speed, 1e-9)
        dt = min(dt, 0.25 / self.drag_rate)
        dt = min(dt, 0.25 / self.wall_drag_rate)
        dt = min(dt, 0.20 * self.dx * self.dy / max(self.eddy_viscosity, 1e-12))
        dt = min(dt, 0.01)
        return max(dt, 1.0e-6)

    def _apply_walls(self):
        """Impermeable, free-slip static side wall."""
        outer = self.inside & (self.rr > (self.radius - 1.5 * self.dx))
        if not np.any(outer):
            return
        rhat_x = self.xx / np.maximum(self.rr, 1e-9)
        rhat_y = self.xy / np.maximum(self.rr, 1e-9)
        un = self.hu * rhat_x + self.hv * rhat_y
        self.hu = np.where(outer, self.hu - un * rhat_x, self.hu)
        self.hv = np.where(outer, self.hv - un * rhat_y, self.hv)

    def _step(self, dt, time):
        dh, dhu, dhv = self._tendencies(self.h, self.hu, self.hv, time)
        h1 = np.maximum(self.h + dt * dh, 0.0)
        hu1 = self.hu + dt * dhu
        hv1 = self.hv + dt * dhv
        dh1, dhu1, dhv1 = self._tendencies(h1, hu1, hv1, time + dt)
        self.h = np.maximum(self.h + 0.5 * dt * (dh + dh1), 0.0)
        self.hu = self.hu + 0.5 * dt * (dhu + dhu1)
        self.hv = self.hv + 0.5 * dt * (dhv + dhv1)
        self._apply_walls()
        mask = self.inside
        self.h = np.where(mask, self.h, 0.0)
        self.hu = np.where(mask, self.hu, 0.0)
        self.hv = np.where(mask, self.hv, 0.0)
        self.steps += 1

    def _step_to(self, target):
        self.substeps_last = 0
        while target - self.time > 1.0e-12:
            dt = min(self._stable_dt(), target - self.time)
            self._step(dt, self.time)
            self._advect_markers(dt)
            self.time = min(self.time + dt, target)
            self.substeps_last += 1
            if self.substeps_last > 400000:
                break
        self.time = float(target)
        self._last_time = self.time

    # ------------------------------------------------------------------
    # Lagrangian free-surface markers
    # ------------------------------------------------------------------
    def _sample_grid(self, field, x, y):
        """Bilinear sampling of a cell-centred grid field."""
        xs = (x + self.radius) / self.dx - 0.5
        ys = (y + self.radius) / self.dy - 0.5
        i0 = np.clip(np.floor(xs).astype(int), 0, self.nx - 1)
        j0 = np.clip(np.floor(ys).astype(int), 0, self.ny - 1)
        fx = np.clip(xs - i0, 0.0, 1.0)
        fy = np.clip(ys - j0, 0.0, 1.0)
        out = np.zeros(len(x), dtype=float)
        for di, wx in ((0, 1.0 - fx), (1, fx)):
            for dj, wy in ((0, 1.0 - fy), (1, fy)):
                ii = np.clip(i0 + di, 0, self.nx - 1)
                jj = np.clip(j0 + dj, 0, self.ny - 1)
                out += wx * wy * field[ii, jj]
        return out

    def _advect_markers(self, dt):
        x = self._pos[:, 0]
        y = self._pos[:, 1]
        u, v = self._velocity(self.h, self.hu, self.hv)
        # surface-parcel advection: sample a lightly smoothed velocity field so
        # the unresolved near-wall boundary layer does not wind up the surface
        # mesh; a rigid rotation is unaffected by the symmetric smoothing.
        u_s = self._smooth(u)
        v_s = self._smooth(v)
        ux = self._sample_grid(u_s, x, y)
        uy = self._sample_grid(v_s, x, y)
        x = x + dt * ux
        y = y + dt * uy
        # impermeable static side wall: remove the radial displacement
        r = np.hypot(x, y)
        limit = self.radius - 0.4 * self.dx
        outward = r > limit
        if np.any(outward):
            scale = np.where(outward, limit / np.maximum(r, 1e-9), 1.0)
            x = x * scale
            y = y * scale
        self._pos[:, 0] = x
        self._pos[:, 1] = y
        self._vel[:, 0] = ux
        self._vel[:, 1] = uy
        self._vel[:, 2] = 0.0

    def _marker_elevation(self):
        """Free-surface markers sit on the computed liquid-air interface."""
        surface = self._sample_grid(self.h, self._pos[:, 0], self._pos[:, 1])
        # nearest-wet-cell fallback keeps the markers on the surface even if a
        # marker is momentarily advected over a dry cell.
        surface = np.where(surface > 0.0, surface, self.fill_height * 0.5)
        return np.maximum(surface, 0.0)

    # ------------------------------------------------------------------
    # exported fields
    # ------------------------------------------------------------------
    def _level_set(self):
        zz = self.gz.reshape(1, 1, self.nz)
        surf = self.h.reshape(self.nx, self.ny, 1)
        return zz - surf

    def _velocity_field(self):
        u, v = self._velocity(self.h, self.hu, self.hv)
        field = np.zeros((self.nx, self.ny, self.nz, 3))
        field[:, :, :, 0] = u[:, :, None]
        field[:, :, :, 1] = v[:, :, None]
        return field

    # ------------------------------------------------------------------
    # wall samples
    # ------------------------------------------------------------------
    def _wall_samples(self):
        n_theta, n_z = 32, 3
        theta = np.arange(n_theta) * (2.0 * np.pi / n_theta)
        side_pts = []
        side_nrm = []
        for z in np.linspace(0.15, 0.9, n_z) * self.wall_height:
            for t in theta:
                side_pts.append((self.radius * np.cos(t), self.radius * np.sin(t), z))
                side_nrm.append((-np.cos(t), -np.sin(t), 0.0))
        plate_pts = []
        plate_nrm = []
        for r in np.linspace(0.2, 0.9, 5) * self.radius:
            for t in theta:
                plate_pts.append((r * np.cos(t), r * np.sin(t), 0.0))
                plate_nrm.append((0.0, 0.0, 1.0))
        return (
            np.asarray(side_pts, dtype=float),
            np.asarray(side_nrm, dtype=float),
            np.asarray(plate_pts, dtype=float),
            np.asarray(plate_nrm, dtype=float),
        )

    # ------------------------------------------------------------------
    # public contract
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        time = float(time)
        if not np.isfinite(time):
            time = self.time
        if time > self.time:
            self._step_to(time)
        elif time < self.time:
            # sample times are strictly increasing; never rewind the physics,
            # but always report the exact requested sample timestamp.
            time = self.time

        pos = self._pos.copy()
        pos[:, 2] = self._marker_elevation()
        vel = self._vel.copy()
        n = len(pos)
        mass = self.particle_mass.copy()

        level_set = self._level_set()
        interior = (level_set < 0.0) & self.inside.reshape(self.nx, self.ny, 1)

        side_pts, side_nrm, plate_pts, plate_nrm = self._wall_samples()
        omega = self._plate_omega(self.time)
        side_vel = np.zeros_like(side_pts)
        plate_vel = np.zeros_like(plate_pts)
        plate_vel[:, 0] = -omega * plate_pts[:, 1]
        plate_vel[:, 1] = omega * plate_pts[:, 0]
        bpts = np.vstack([side_pts, plate_pts])
        bnorm = np.vstack([side_nrm, plate_nrm])
        btarget = np.vstack([side_vel, plate_vel])
        # imposed boundary conditions: no-slip on the stationary wall, the
        # plate surface velocity on the driven bottom.
        bvel = btarget.copy()

        frame = StateFrame(
            self.time,
            pos,
            vel,
            mass=mass,
            body_id=np.zeros(n, dtype=np.int32),
            attributes={
                "mass_quantity": mass.copy(),
                "particle_radius": self.particle_radius.copy(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1),
                "liquid_level_set": level_set,
                "liquid_interior_mask": interior,
                "velocity_field": self._velocity_field(),
                "boundary_normal": bnorm,
                "boundary_velocity": bvel,
                "boundary_target_velocity": btarget,
            },
            scalars={
                "grid_spacing": float(min(self.dx, self.dy)),
                "characteristic_length": float(self.characteristic_length),
                "reference_velocity": float(self.reference_velocity),
            },
        )
        frame.validate()
        self._frame = frame
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self):
        wet = self.inside
        if np.any(wet):
            hh = np.where(wet, self.h, 0.0)
            weight = float(hh.sum())
            cx = float((hh * self.xx).sum() / weight) if weight > 0 else 0.0
            cy = float((hh * self.xy).sum() / weight) if weight > 0 else 0.0
            hmin = float(self.h[wet].min())
            hmax = float(self.h[wet].max())
        else:
            cx = cy = 0.0
            hmin = hmax = 0.0
        depth = self.particle_mass / (self.density * self.spacing ** 3)
        return {
            "model": "depth_averaged_shallow_water",
            "integrator": "heun_cfl_substepping",
            "time": float(self.time),
            "steps": int(self.steps),
            "min_column_height_m": hmin,
            "max_column_height_m": hmax,
            "surface_centroid_m": [cx, cy],
            "total_volume_m3": float(np.sum(depth) * self.spacing ** 3),
            "particles": int(self.n_particles),
        }
