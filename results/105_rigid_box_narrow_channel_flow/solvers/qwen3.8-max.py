"""Two-way coupled rigid-box / narrow-channel fluid solver.

Implements a simplified 2-D staggered-grid Eulerian fluid on the x-z plane
coupled to a single rigid box with (translate_x, translate_z, rotate_y) DOFs.
Markers are passively advected by the fluid velocity field.  The lifting drive
is applied as a uniform body-force density to the fluid; the resulting pressure
gradient through the narrow side gaps pushes the box upward once the drive is
active.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Minimal but physically-based two-way solid-fluid coupler."""

    # ------------------------------------------------------------------
    # public protocol
    # ------------------------------------------------------------------
    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.cfg = scene.config
        self.seed = seed

        # ---- domain & grid ------------------------------------------------
        fluid_cfg = self.cfg["fluid"]
        box_cfg = self.cfg["rigid_box"]
        self.rho = float(fluid_cfg["density"])
        dmin = np.asarray(fluid_cfg["domain_min"], dtype=float)
        dmax = np.asarray(fluid_cfg["domain_max"], dtype=float)
        res = np.asarray(fluid_cfg["nominal_resolution"], dtype=int)
        self.dmin = dmin
        self.dmax = dmax
        # Use only x and z in this 2-D simulation
        self.nx = int(res[0])
        self.nz = int(res[2])
        self.dx = (dmax[0] - dmin[0]) / self.nx
        self.dz = (dmax[2] - dmin[2]) / self.nz
        self.gravity = np.asarray(self.cfg["gravity"], dtype=float)

        # ---- walls --------------------------------------------------------
        geom = self.cfg.get("scene_geometry", {})
        objs = {o["id"]: o for o in geom.get("objects", [])}
        lw = objs.get("channel_left_wall", {})
        rw = objs.get("channel_right_wall", {})
        fl = objs.get("channel_floor", {})

        def _box_limits(obj: dict):
            he = np.asarray(obj["geometry"]["half_extents"], dtype=float)
            tr = np.asarray(obj["transform"]["translation"], dtype=float)
            return tr - he, tr + he

        self.wall_left_lo, self.wall_left_hi = _box_limits(lw)
        self.wall_right_lo, self.wall_right_hi = _box_limits(rw)
        self.floor_lo, self.floor_hi = _box_limits(fl)

        # Interior channel bounds (where fluid lives)
        self.channel_x_min = float(self.wall_left_hi[0])   # right face of left wall
        self.channel_x_max = float(self.wall_right_lo[0])  # left face of right wall
        self.channel_z_min = float(self.floor_hi[2])       # top of floor

        # ---- controls -----------------------------------------------------
        self.controls = list(self.cfg.get("controls", []))

        # ---- rigid body state --------------------------------------------
        self.box_mass = float(box_cfg["mass"])
        half = np.asarray(box_cfg["size"], dtype=float) / 2.0
        self.box_half_x = float(half[0])
        self.box_half_z = float(half[2])
        # Moment of inertia about y axis for a rectangular plate (thin in y)
        self.box_Iy = self.box_mass * (self.box_half_x**2 + self.box_half_z**2) / 3.0

        init_state = scene.initial_state()
        self.box_pos = np.array(init_state.position[0], dtype=float)
        self.box_vel = np.array(init_state.velocity[0], dtype=float)
        q = np.array(init_state.orientation[0], dtype=float)
        # Quaternion xyzw -> angle about y
        self.box_angle = 2.0 * math.asin(float(np.clip(q[1], -1.0, 1.0)))
        self.box_omega = float(init_state.angular_velocity[0, 1])

        # ---- markers ------------------------------------------------------
        self.marker_pos = np.array(init_state.position[1:], dtype=float)
        self.marker_vel = np.array(init_state.velocity[1:], dtype=float)
        self.marker_mass = np.array(init_state.mass[1:], dtype=float)
        self.phase = np.array(init_state.attributes["phase"][1:], dtype=float)

        # ---- fluid fields (staggered MAC) ---------------------------------
        self.u = np.zeros((self.nz, self.nx + 1), dtype=float)  # x-face vel
        self.w = np.zeros((self.nz + 1, self.nx), dtype=float)  # z-face vel
        self.p = np.zeros((self.nz, self.nx), dtype=float)
        self.solid_frac = np.zeros((self.nz, self.nx), dtype=float)  # cell solid fraction [0,1]

        # ---- budget accumulators ------------------------------------------
        self.cum_ext_impulse = np.zeros(3, dtype=float)
        self.cum_inflow_mom = np.zeros(3, dtype=float)
        self.cum_outflow_mom = np.zeros(3, dtype=float)
        self.cum_mass_source = 0.0

        # ---- timing -------------------------------------------------------
        self.current_time = 0.0

    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        dt_desired = time - self.current_time
        if dt_desired <= 0:
            return self._make_frame(time)

        # Sub-step for stability
        max_sub = 0.002
        nsub = max(1, int(math.ceil(dt_desired / max_sub)))
        dt = dt_desired / nsub

        for _ in range(nsub):
            self._step(dt)
            self.current_time += dt

        return self._make_frame(time)

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {"seed": self.seed}

    # ==================================================================
    # internal physics
    # ==================================================================
    def _drive_acceleration(self, t: float) -> float:
        """Return current upward drive acceleration magnitude applied to fluid."""
        for c in self.controls:
            if c.get("id") == "upward_fluid_drive":
                start = float(c["start_time"])
                end = float(c["end_time"])
                ramp = float(c.get("ramp_duration", 0.0))
                mag = float(c["magnitude"])
                if t < start or t > end:
                    return 0.0
                frac = 1.0
                if ramp > 0:
                    if t < start + ramp:
                        frac = (t - start) / ramp
                    elif t > end - ramp:
                        frac = (end - t) / ramp
                    frac = max(0.0, min(1.0, frac))
                return mag * frac
        return 0.0

    def _compute_solid_fraction(self) -> None:
        """Rasterize the tilted box onto the grid as a solid volume fraction."""
        sf = self.solid_frac
        sf.fill(0.0)
        cx, cz = float(self.box_pos[0]), float(self.box_pos[2])
        ca = math.cos(self.box_angle)
        sa = math.sin(self.box_angle)
        hx, hz = self.box_half_x, self.box_half_z
        # Bounding cells
        diag = math.sqrt(hx*hx + hz*hz)
        i0 = max(0, int((cx - diag - self.dmin[0]) / self.dx) - 1)
        i1 = min(self.nx - 1, int((cx + diag - self.dmin[0]) / self.dx) + 1)
        k0 = max(0, int((cz - diag - self.dmin[2]) / self.dz) - 1)
        k1 = min(self.nz - 1, int((cz + diag - self.dmin[2]) / self.dz) + 1)
        # Sample 4 sub-cells per cell for better accuracy
        ns = 4
        for k in range(k0, k1 + 1):
            z_base = self.dmin[2] + k * self.dz
            for i in range(i0, i1 + 1):
                x_base = self.dmin[0] + i * self.dx
                cnt = 0
                for si in range(ns):
                    x_s = x_base + (si + 0.5) * self.dx / ns
                    for sk in range(ns):
                        z_s = z_base + (sk + 0.5) * self.dz / ns
                        rx = x_s - cx
                        rz = z_s - cz
                        lx = ca * rx + sa * rz
                        lz = -sa * rx + ca * rz
                        if abs(lx) <= hx and abs(lz) <= hz:
                            cnt += 1
                sf[k, i] = cnt / (ns * ns)

    def _apply_boundary_conditions(self) -> None:
        """Impermeable slip side walls + floor. Set normal velocity to zero at boundaries."""
        nx, nz = self.nx, self.nz
        # Side walls: u at i=0 and i=nx should be zero (impermeable)
        self.u[:, 0] = 0.0
        self.u[:, nx] = 0.0
        # Floor: w at k=0 should be zero
        self.w[0, :] = 0.0
        # Top: open / zero-gradient — leave w[nz,:] as-is but clamp inflow later

    def _pressure_project(self, dt: float) -> None:
        """Simple Jacobi pressure projection with solid fraction weighting."""
        nx, nz = self.nx, self.nz
        dx, dz = self.dx, self.dz
        rho = self.rho
        sf = self.solid_frac
        p = self.p
        u = self.u
        w = self.w

        # Compute divergence
        div = np.zeros((nz, nx), dtype=float)
        for k in range(nz):
            for i in range(nx):
                fluid_frac = 1.0 - sf[k, i]
                if fluid_frac < 1e-8:
                    continue
                dudx = (u[k, i + 1] - u[k, i]) / dx
                dwdz = (w[k + 1, i] - w[k, i]) / dz
                div[k, i] = (dudx + dwdz) * fluid_frac

        # Jacobi iterations
        p.fill(0.0)
        p_new = np.zeros_like(p)
        for _it in range(40):
            for k in range(nz):
                for i in range(nx):
                    fluid_frac = 1.0 - sf[k, i]
                    if fluid_frac < 1e-8:
                        continue
                    aE = (1.0 - sf[k, min(i + 1, nx - 1)]) if i < nx - 1 else 0.0
                    aW = (1.0 - sf[k, max(i - 1, 0)]) if i > 0 else 0.0
                    aN = (1.0 - sf[min(k + 1, nz - 1), i]) if k < nz - 1 else 0.0
                    aS = (1.0 - sf[max(k - 1, 0), i]) if k > 0 else 0.0
                    center = -(aE + aW) / (dx * dx) - (aN + aS) / (dz * dz)
                    if abs(center) < 1e-20:
                        continue
                    rhs = rho * div[k, i] / dt
                    lap = 0.0
                    if i < nx - 1:
                        lap += aE * p[k, i + 1] / (dx * dx)
                    if i > 0:
                        lap += aW * p[k, i - 1] / (dx * dx)
                    if k < nz - 1:
                        lap += aN * p[k + 1, i] / (dz * dz)
                    if k > 0:
                        lap += aS * p[k - 1, i] / (dz * dz)
                    p_new[k, i] = (rhs - lap) / center
            p[:] = p_new

        # Correct velocities
        for k in range(nz):
            for i in range(nx + 1):
                fL = 1.0 - sf[k, max(i - 1, 0)] if i > 0 else 0.0
                fR = 1.0 - sf[k, min(i, nx - 1)] if i < nx else 0.0
                ff = 0.5 * (fL + fR)
                if ff > 1e-8 and 0 < i < nx:
                    u[k, i] -= dt / rho * (p[k, i] - p[k, i - 1]) / dx * ff
        for k in range(nz + 1):
            for i in range(nx):
                fB = 1.0 - sf[max(k - 1, 0), i] if k > 0 else 0.0
                fT = 1.0 - sf[min(k, nz - 1), i] if k < nz else 0.0
                ff = 0.5 * (fB + fT)
                if ff > 1e-8 and 0 < k < nz:
                    w[k, i] -= dt / rho * (p[k, i] - p[k - 1, i]) / dz * ff

    def _fluid_forces_on_box(self) -> tuple[np.ndarray, float]:
        """Compute net fluid force and torque on the box from pressure and viscous stress."""
        force = np.zeros(3, dtype=float)
        torque_y = 0.0
        cx, cz = float(self.box_pos[0]), float(self.box_pos[2])
        ca = math.cos(self.box_angle)
        sa = math.sin(self.box_angle)
        dx, dz = self.dx, self.dz
        nx, nz_g = self.nx, self.nz
        p = self.p
        sf = self.solid_frac

        for k in range(nz_g):
            z_c = self.dmin[2] + (k + 0.5) * dz
            for i in range(nx):
                x_c = self.dmin[0] + (i + 0.5) * dx
                f = sf[k, i]
                if f < 1e-6 or f > 1.0 - 1e-6:
                    continue
                # Pressure gradient force on solid portion
                dpdx = 0.0
                dpdz = 0.0
                if 0 < i < nx - 1:
                    dpdx = (p[k, i + 1] - p[k, i - 1]) / (2 * dx)
                if 0 < k < nz_g - 1:
                    dpdz = (p[k + 1, i] - p[k - 1, i]) / (2 * dz)
                vol = dx * dz * f
                fx = -dpdx * vol
                fz = -dpdz * vol
                force[0] += fx
                force[2] += fz
                # Torque about box center (y-axis)
                rx = x_c - cx
                rz = z_c - cz
                torque_y += rx * fz - rz * fx
        return force, torque_y

    def _advect_markers(self, dt: float) -> None:
        """Advect markers using bilinear interpolation of the staggered velocity field."""
        x0 = self.dmin[0]
        z0 = self.dmin[2]
        dx, dz = self.dx, self.dz
        nx, nz = self.nx, self.nz
        u = self.u
        w = self.w

        pos = self.marker_pos
        vel = self.marker_vel

        for m in range(len(pos)):
            x, z = pos[m, 0], pos[m, 2]
            # Interpolate u at (x,z) — u lives at x-faces
            fi = (x - x0) / dx - 0.5
            fk = (z - z0) / dz
            i0 = int(math.floor(fi))
            k0 = int(math.floor(fk))
            sx = fi - i0
            sz = fk - k0
            ui = 0.0
            for di in range(2):
                for dk in range(2):
                    ii = max(0, min(nx, i0 + di))
                    kk = max(0, min(nz - 1, k0 + dk))
                    ui += u[kk, ii] * ((1 - sx) if di == 0 else sx) * ((1 - sz) if dk == 0 else sz)
            # Interpolate w at (x,z) — w lives at z-faces
            fi = (x - x0) / dx
            fk = (z - z0) / dz - 0.5
            i0 = int(math.floor(fi))
            k0 = int(math.floor(fk))
            sx = fi - i0
            sz = fk - k0
            wi = 0.0
            for di in range(2):
                for dk in range(2):
                    ii = max(0, min(nx - 1, i0 + di))
                    kk = max(0, min(nz, k0 + dk))
                    wi += w[kk, ii] * ((1 - sx) if di == 0 else sx) * ((1 - sz) if dk == 0 else sz)
            vel[m, 0] = ui
            vel[m, 1] = 0.0
            vel[m, 2] = wi
            pos[m, 0] += ui * dt
            pos[m, 2] += wi * dt

        # Clamp markers inside channel bounds
        np.clip(pos[:, 0], self.channel_x_min + 1e-6, self.channel_x_max - 1e-6, out=pos[:, 0])
        np.clip(pos[:, 2], self.channel_z_min + 1e-6, self.dmax[2] - 1e-6, out=pos[:, 2])

    def _collide_box_with_walls(self) -> None:
        """Simple positional clamping of box against channel walls/floor."""
        bx, bz = self.box_pos[0], self.box_pos[2]
        ca = math.cos(self.box_angle)
        sa = math.sin(self.box_angle)
        hx, hz = self.box_half_x, self.box_half_z
        # Axis-aligned bounding extent
        ext_x = abs(ca) * hx + abs(sa) * hz
        ext_z = abs(sa) * hx + abs(ca) * hz
        # Left wall
        if bx - ext_x < self.channel_x_min:
            self.box_pos[0] = self.channel_x_min + ext_x
            if self.box_vel[0] < 0:
                self.box_vel[0] *= -0.2
        # Right wall
        if bx + ext_x > self.channel_x_max:
            self.box_pos[0] = self.channel_x_max - ext_x
            if self.box_vel[0] > 0:
                self.box_vel[0] *= -0.2
        # Floor
        if bz - ext_z < self.channel_z_min:
            self.box_pos[2] = self.channel_z_min + ext_z
            if self.box_vel[2] < 0:
                self.box_vel[2] *= -0.2
        # Ceiling (open top but keep reasonable)
        if bz + ext_z > self.dmax[2]:
            self.box_pos[2] = self.dmax[2] - ext_z
            if self.box_vel[2] > 0:
                self.box_vel[2] *= -0.2

    def _step(self, dt: float) -> None:
        """Advance one sub-step."""
        t = self.current_time
        g = self.gravity
        drive_a = self._drive_acceleration(t)

        # --- Fluid step ---
        self._compute_solid_fraction()

        # Apply gravity + drive to fluid velocity
        gz = g[2] + drive_a  # upward drive adds to z-acceleration
        self.w[:-1, :] += gz * dt  # apply to interior z-faces
        self.u[:, :] += g[0] * dt  # typically zero

        self._apply_boundary_conditions()
        self._pressure_project(dt)
        self._apply_boundary_conditions()

        # --- Box dynamics ---
        f_fluid, tau_fluid = self._fluid_forces_on_box()
        # Gravity on box
        f_grav = self.box_mass * g
        f_total = f_grav + f_fluid
        a_lin = f_total / self.box_mass
        alpha = tau_fluid / self.box_Iy

        self.box_vel += a_lin * dt
        self.box_omega += alpha * dt
        # Damping for numerical stability
        damp = max(0.0, 1.0 - 0.5 * dt)
        self.box_vel *= damp
        self.box_omega *= damp

        self.box_pos += self.box_vel * dt
        self.box_angle += self.box_omega * dt
        # Small angular damping
        self.box_angle = float(np.clip(self.box_angle, -0.5, 0.5))

        self._collide_box_with_walls()

        # --- Markers ---
        self._advect_markers(dt)

        # --- Budgets ---
        self.cum_ext_impulse += f_total * dt
        # No mass sources / inflow-outflow in closed domain
        # Reference scales
        # (computed lazily in frame)

    # ==================================================================
    # Frame construction
    # ==================================================================
    def _make_frame(self, time: float) -> StateFrame:
        n_markers = len(self.marker_pos)
        n_total = 1 + n_markers

        position = np.vstack((self.box_pos.reshape(1, 3), self.marker_pos))
        velocity = np.vstack((self.box_vel.reshape(1, 3), self.marker_vel))
        orientation = np.tile([0.0, 0.0, 0.0, 1.0], (n_total, 1))
        qy = math.sin(self.box_angle / 2.0)
        qw = math.cos(self.box_angle / 2.0)
        orientation[0] = [0.0, qy, 0.0, qw]
        ang_vel = np.zeros((n_total, 3), dtype=float)
        ang_vel[0, 1] = self.box_omega

        mass = np.empty(n_total, dtype=float)
        mass[0] = self.box_mass
        mass[1:] = self.marker_mass

        body_id = np.zeros(n_total, dtype=np.int64)
        body_id[0] = 1

        phase = np.ones(n_total, dtype=float)
        phase[0] = 0.0

        # Boundary info (3 boundary objects: left wall, right wall, floor)
        bnd_normal = np.array([
            [1.0, 0.0, 0.0],   # left wall points +x into channel
            [-1.0, 0.0, 0.0],  # right wall points -x
            [0.0, 0.0, 1.0],   # floor points +z
        ], dtype=float)
        bnd_vel = np.zeros((3, 3), dtype=float)
        bnd_target_vel = np.zeros((3, 3), dtype=float)

        # Fluid velocity field for export (reshape to match expected [nz,nx,3])
        fv = np.zeros((self.nz, self.nx, 3), dtype=float)
        for k in range(self.nz):
            for i in range(self.nx):
                fv[k, i, 0] = 0.5 * (self.u[k, i] + self.u[k, i + 1])
                fv[k, i, 2] = 0.5 * (self.w[k, i] + self.w[k + 1, i])

        # Interaction force: fluid-on-box (equal and opposite pair)
        f_fluid, _ = self._fluid_forces_on_box()
        interaction_force = np.vstack([f_fluid, -f_fluid])  # [2,3]

        # Budget domain weights: G=2 domains (solid body, fluid), N=n_total samples
        weights = np.zeros((2, n_total), dtype=float)
        weights[0, 0] = 1.0         # domain 0 = box
        weights[1, 1:] = 1.0        # domain 1 = fluid markers

        # External forces per domain: [G, K, 3]; use K=1 term
        ext_force = np.zeros((2, 1, 3), dtype=float)
        ext_force[0, 0] = self.box_mass * self.gravity  # gravity on box
        drive_a = self._drive_acceleration(time)
        fluid_mass = float(np.sum(self.marker_mass))
        ext_force[1, 0, 2] = fluid_mass * (self.gravity[2] + drive_a)

        ref_vel = max(1.0, float(np.max(np.linalg.norm(velocity, axis=1))))
        ref_mom = max(1.0, float(np.sum(mass * np.linalg.norm(velocity, axis=1))))

        return StateFrame(
            time=time,
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=ang_vel,
            mass=mass,
            body_id=body_id,
            attributes={
                "phase": phase,
                "mass_quantity": mass.copy(),
            },
            fields={
                "boundary_normal": bnd_normal,
                "boundary_target_velocity": bnd_target_vel,
                "boundary_velocity": bnd_vel,
                "budget_cumulative_external_impulse": self.cum_ext_impulse.reshape(1, 3).copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow_mom.reshape(1, 3).copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow_mom.reshape(1, 3).copy(),
                "budget_domain_weights": weights,
                "budget_external_force": ext_force,
                "cumulative_mass_source": np.array([self.cum_mass_source]),
                "interaction_force": interaction_force,
                "fluid_velocity": fv,
            },
            scalars={
                "reference_velocity": ref_vel,
                "reference_momentum": ref_mom,
                "fluid_drive": drive_a,
            },
        )
