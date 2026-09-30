"""Two-way coupled rigid-box / fluid solver for scene 105.

Physical model
--------------
2-D incompressible fluid on a staggered (MAC) grid covering the narrow
channel.  The heavy rigid box is represented by cut-cell solid volume
fractions; the pressure projection uses the variational cut-cell
discretization (open face fractions weighted fluxes), so the evolving box
changes the resolved flow and the integrated pressure field exerts force
and torque on the box (two-way coupling).

Boundary conditions: impermeable slip side walls (u = 0 at x = +/-0.5) and
a periodic through-flow direction along the tube axis, which lets the
prescribed uniform fluid drive produce a sustained upward flow through the
side gaps that lifts the box.

Time stepping: fixed sub-steps of semi-Lagrangian advection + body forces
(gravity and the ramped fluid drive) + pressure projection + explicit
rigid integration with an impulse/position contact solver against the
channel walls.  Passive mass-carrying markers are advected by the
resolved grid velocity (RK2) and are kept outside the rigid box.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

import scipy.sparse as _sp
import scipy.sparse.linalg as _spl


class BenchmarkSolver:
    """Solver-agent implementation for 105_rigid_box_narrow_channel_flow."""

    def __init__(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", None) or {}
        fluid = cfg.get("fluid", {}) or {}
        box_cfg = cfg.get("rigid_box", {}) or {}

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.rho = float(fluid.get("density", 1000.0))
        dmin = list(fluid.get("domain_min", [-0.5, -0.04, 0.05]))
        dmax = list(fluid.get("domain_max", [0.5, 0.04, 3.55]))
        res = list(fluid.get("nominal_resolution", [64, 1, 224]))
        self.x0, self.x1 = float(dmin[0]), float(dmax[0])
        self.z0, self.z1 = float(dmin[2]), float(dmax[2])
        self.depth = float(dmax[1] - dmin[1])
        self.nx = max(4, int(res[0]))
        self.nz = max(4, int(res[2]))
        self.dx = (self.x1 - self.x0) / self.nx
        self.dz = (self.z1 - self.z0) / self.nz
        self.Vc = self.dx * self.dz * self.depth
        self.Lz = self.z1 - self.z0

        # rigid box
        size = list(box_cfg.get("size", [0.86, 0.08, 0.48]))
        self.hx = 0.5 * float(size[0])
        self.hz = 0.5 * float(size[2])
        self.m_box = float(box_cfg.get("mass", 55.0))
        self.I_box = self.m_box * (float(size[0]) ** 2 + float(size[2]) ** 2) / 12.0

        # fluid drive control
        self.drive = None
        for control in cfg.get("controls", []) or []:
            if control.get("id") == "upward_fluid_drive":
                direction = np.asarray(control.get("direction", [0.0, 0.0, 1.0]), dtype=float)
                norm = float(np.linalg.norm(direction))
                if norm > 0.0:
                    direction = direction / norm
                self.drive = {
                    "direction": direction,
                    "magnitude": float(control.get("magnitude", 15.0)),
                    "start": float(control.get("start_time", 3.0)),
                    "end": float(control.get("end_time", 8.0)),
                    "ramp": float(control.get("ramp_duration", 0.35)),
                }
                break

        # grid coordinates
        self.x_f = self.x0 + self.dx * np.arange(self.nx + 1)
        self.x_c = self.x0 + self.dx * (np.arange(self.nx) + 0.5)
        self.z_c = self.z0 + self.dz * (np.arange(self.nz) + 0.5)
        self.z_f = self.z0 + self.dz * np.arange(self.nz)  # periodic w faces

        self._cx_grid, self._cz_grid = np.meshgrid(self.x_c, self.z_c, indexing="ij")
        self._ux_grid, self._uz_grid = np.meshgrid(
            self.x_f[1:self.nx], self.z_c, indexing="ij"
        )
        self._wx_grid, self._wz_grid = np.meshgrid(self.x_c, self.z_f, indexing="ij")

        # initial state from the frozen scene
        frame = scene.initial_state()
        pos = np.asarray(frame.position, dtype=float)
        vel = np.asarray(frame.velocity, dtype=float)
        quat = np.asarray(frame.orientation, dtype=float)
        angv = np.asarray(frame.angular_velocity, dtype=float)
        mass = np.asarray(frame.mass, dtype=float)
        body_id = np.asarray(frame.body_id)
        phase = np.asarray(frame.attributes["phase"], dtype=float)

        box_rows = np.where(body_id == 1)[0]
        marker_rows = np.where(phase == 1.0)[0]
        if box_rows.size == 0 or marker_rows.size == 0:
            raise ValueError("initial state is missing the box or the fluid markers")

        b = int(box_rows[0])
        self.cx = float(pos[b, 0])
        self.cz = float(pos[b, 2])
        self.vx = float(vel[b, 0])
        self.vz = float(vel[b, 2])
        q = quat[b]
        self.theta = 2.0 * float(np.arctan2(q[1], q[3]))
        self.omega = float(angv[b, 1])

        self.markers = pos[marker_rows][:, [0, 2]].copy()
        self.marker_mass = mass[marker_rows].copy()
        self.n_markers = self.markers.shape[0]

        # fluid grid state
        self.u = np.zeros((self.nx + 1, self.nz), dtype=float)
        self.w = np.zeros((self.nx, self.nz), dtype=float)
        self.phi = self._cell_phi(self.cx, self.cz, self.theta)

        # boundary samples (static channel walls)
        z_samples = np.linspace(self.z0 + 0.05, self.z1 - 0.05, 50)
        x_samples = np.linspace(self.x0 + 0.05, self.x1 - 0.05, 14)
        normals = np.zeros((50 + 50 + 14, 3), dtype=float)
        normals[:50, 0] = 1.0            # left wall, inner face x = x0
        normals[50:100, 0] = -1.0        # right wall, inner face x = x1
        normals[100:, 2] = 1.0           # floor top z = z0
        self.boundary_normal = normals
        self.boundary_velocity = np.zeros_like(normals)

        # budget / bookkeeping state
        m_fluid_markers = float(self.marker_mass.sum())
        self.reference_velocity = 1.0
        self.reference_momentum = float(self.m_box + m_fluid_markers) * 1.0
        self.cum_ext_impulse = np.zeros((2, 3), dtype=float)
        self.cum_inflow = np.zeros((2, 3), dtype=float)
        self.cum_outflow = np.zeros((2, 3), dtype=float)
        self.contact_impulse = np.zeros(3, dtype=float)
        self._contact_reported = np.zeros(3, dtype=float)
        self.last_fluid_force = np.zeros(3, dtype=float)
        self.last_traction = np.zeros(3, dtype=float)
        self._frame_time_prev = 0.0

        self.t = 0.0
        self.dt = float(cfg.get("sample_dt", 0.02)) / 8.0
        self._steps = 0
        self._initialized = True

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    def _inside(self, xq, zq, cx, cz, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        rx = np.asarray(xq, dtype=float) - cx
        rz = np.asarray(zq, dtype=float) - cz
        lx = c * rx - s * rz
        lz = s * rx + c * rz
        return (np.abs(lx) <= self.hx) & (np.abs(lz) <= self.hz)

    def _cell_phi(self, cx, cz, theta):
        acc = np.zeros((self.nx, self.nz), dtype=float)
        for ox in (-1.0, 0.0, 1.0):
            for oz in (-1.0, 0.0, 1.0):
                acc += self._inside(
                    self._cx_grid + ox * self.dx / 3.0,
                    self._cz_grid + oz * self.dz / 3.0,
                    cx, cz, theta,
                )
        return acc / 9.0

    def _face_open(self, cx, cz, theta):
        ou = np.zeros((self.nx - 1, self.nz), dtype=float)
        for oz in (-1.0, 0.0, 1.0):
            ou += ~self._inside(
                self._ux_grid, self._uz_grid + oz * self.dz / 3.0, cx, cz, theta
            )
        ow = np.zeros((self.nx, self.nz), dtype=float)
        for ox in (-1.0, 0.0, 1.0):
            ow += ~self._inside(
                self._wx_grid + ox * self.dx / 3.0, self._wz_grid, cx, cz, theta
            )
        return ou / 3.0, ow / 3.0

    # ------------------------------------------------------------------
    # staggered interpolation (periodic in z, walls in x)
    # ------------------------------------------------------------------
    def _interp_u(self, xq, zq):
        xq = np.asarray(xq, dtype=float)
        zq = np.asarray(zq, dtype=float)
        zz = self.z0 + np.mod(zq - self.z0, self.Lz)
        i = np.clip(np.floor((xq - self.x0) / self.dx).astype(np.int64), 0, self.nx - 1)
        fx = (xq - self.x_f[i]) / self.dx
        j = np.floor((zz - self.z0) / self.dz).astype(np.int64) % self.nz
        fz = (zz - self.z_c[j]) / self.dz
        jp = (j + 1) % self.nz
        u = self.u
        v00 = u[i, j]
        v01 = u[i, jp]
        v10 = u[i + 1, j]
        v11 = u[i + 1, jp]
        return (1.0 - fx) * ((1.0 - fz) * v00 + fz * v01) + fx * (
            (1.0 - fz) * v10 + fz * v11
        )

    def _interp_w(self, xq, zq):
        xq = np.asarray(xq, dtype=float)
        zq = np.asarray(zq, dtype=float)
        zz = self.z0 + np.mod(zq - self.z0, self.Lz)
        i = np.clip(np.floor((xq - self.x0) / self.dx).astype(np.int64), 0, self.nx - 1)
        fx = (xq - self.x_c[i]) / self.dx
        j = np.floor((zz - self.z0) / self.dz).astype(np.int64) % self.nz
        fz = (zz - self.z_f[j]) / self.dz
        jp = (j + 1) % self.nz
        w = self.w
        v00 = w[i, j]
        v01 = w[i, jp]
        v10 = w[i + 1, j]
        v11 = w[i + 1, jp]
        return (1.0 - fx) * ((1.0 - fz) * v00 + fz * v01) + fx * (
            (1.0 - fz) * v10 + fz * v11
        )

    # ------------------------------------------------------------------
    # physics pieces
    # ------------------------------------------------------------------
    def _drive_accel(self, t):
        if self.drive is None:
            return np.zeros(3)
        d = self.drive
        if t < d["start"] or t > d["end"]:
            return np.zeros(3)
        ramp = 1.0
        if d["ramp"] > 0.0:
            ramp = min(1.0, max(0.0, (t - d["start"]) / d["ramp"]))
        return d["direction"] * (d["magnitude"] * ramp)

    def _advect(self, dt):
        # u faces (interior)
        vx = self._interp_u(self._ux_grid, self._uz_grid)
        vz = self._interp_w(self._ux_grid, self._uz_grid)
        bx = np.clip(self._ux_grid - dt * vx, self.x0, self.x1)
        bz = self._uz_grid - dt * vz
        new_u = self.u.copy()
        new_u[1:self.nx, :] = self._interp_u(bx, bz)

        # w faces (all, periodic in z)
        vx = self._interp_u(self._wx_grid, self._wz_grid)
        vz = self._interp_w(self._wx_grid, self._wz_grid)
        bx = np.clip(self._wx_grid - dt * vx, self.x0, self.x1)
        bz = self._wz_grid - dt * vz
        new_w = self._interp_w(bx, bz)
        return new_u, new_w

    def _project(self, o_u, o_w, phi_old, phi_new, dt):
        nx, nz, rho = self.nx, self.nz, self.rho
        n = nx * nz

        # x-faces: face i (1..nx-1) between cells (i-1, j) [a] and (i, j) [b]
        I = np.arange(1, nx)
        J = np.arange(nz)
        aX = (I[:, None] - 1) * nz + J[None, :]
        bX = I[:, None] * nz + J[None, :]
        Ax = self.dz * self.depth
        Kx = (o_u ** 2) * Ax * dt / (rho * self.dx)
        fluxX = o_u * Ax * self.u[1:nx, :]

        # z-faces (periodic): face j between (i, (j-1)%nz) [a] and (i, j) [b]
        I2 = np.arange(nx)
        J2 = np.arange(nz)
        aZ = I2[:, None] * nz + ((J2[None, :] - 1) % nz)
        bZ = I2[:, None] * nz + J2[None, :]
        Az = self.dx * self.depth
        Kz = (o_w ** 2) * Az * dt / (rho * self.dz)
        fluxZ = o_w * Az * self.w

        aXr = aX.ravel()
        bXr = bX.ravel()
        aZr = aZ.ravel()
        bZr = bZ.ravel()

        div = np.bincount(
            np.concatenate([aXr, bXr, aZr, bZr]),
            weights=np.concatenate([fluxX.ravel(), -fluxX.ravel(),
                                    fluxZ.ravel(), -fluxZ.ravel()]),
            minlength=n,
        )
        rhs = ((phi_new - phi_old) * self.Vc / dt).reshape(-1) - div

        rows = np.concatenate([aXr, bXr, bXr, aXr, aZr, bZr, bZr, aZr])
        cols = np.concatenate([aXr, bXr, aXr, bXr, aZr, bZr, aZr, bZr])
        vals = np.concatenate([Kx.ravel(), Kx.ravel(), -Kx.ravel(), -Kx.ravel(),
                               Kz.ravel(), Kz.ravel(), -Kz.ravel(), -Kz.ravel()])
        A = _sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()

        # fully covered cells: pin their (otherwise empty) row
        diag = A.diagonal()
        deg = np.where(diag < 1e-13)[0]
        if deg.size:
            eye = _sp.coo_matrix(
                (np.ones(deg.size), (deg, deg)), shape=(n, n)
            ).tocsc()
            A = (A + eye).tocsc()
            rhs = rhs.copy()
            rhs[deg] = 0.0

        try:
            p = _spl.splu(A).solve(rhs)
        except Exception:
            pin = np.zeros(n)
            pin[0] = 1.0
            A2 = _sp.diags(pin).tocsc()
            Apin = A + A2
            rhs2 = rhs.copy()
            rhs2[0] = 0.0
            try:
                p = _spl.splu(Apin.tocsc()).solve(rhs2)
            except Exception:
                p = np.zeros(n)
        if not np.isfinite(p).all():
            p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

        p2 = p.reshape(nx, nz)
        self.u[1:nx, :] -= dt * (p2[1:nx, :] - p2[0:nx - 1, :]) / (rho * self.dx)
        self.w -= dt * (p[bZ] - p[aZ]) / (rho * self.dz)
        return p2

    def _box_force(self, p2, phi):
        gpx, gpz = np.gradient(phi, self.dx, self.dz)
        pf = p2 * self.Vc
        fx = pf * gpx
        fz = pf * gpz
        Fx = float(fx.sum())
        Fz = float(fz.sum())
        rx = self._cx_grid - self.cx
        rz = self._cz_grid - self.cz
        tau = float((rz * fx - rx * fz).sum())
        return Fx, Fz, tau

    def _enforce_blocked_faces(self, o_u, o_w):
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        rx = self._ux_grid - self.cx
        rz = self._uz_grid - self.cz
        vs_x = self.vx + self.omega * rz
        mask = o_u <= 0.0
        self.u[1:self.nx, :][mask] = vs_x[mask]
        rx = self._wx_grid - self.cx
        rz = self._wz_grid - self.cz
        vs_z = self.vz - self.omega * rx
        maskw = o_w <= 0.0
        self.w[maskw] = vs_z[maskw]

    def _contacts(self):
        planes = (
            ((1.0, 0.0), self.x0),      # left wall
            ((-1.0, 0.0), -self.x1),    # right wall
            ((0.0, 1.0), self.z0),      # floor
            ((0.0, -1.0), -self.z1),    # ceiling
        )
        corners = (
            (-self.hx, -self.hz), (self.hx, -self.hz),
            (self.hx, self.hz), (-self.hx, self.hz),
        )
        m = self.m_box
        inertia = self.I_box
        for _ in range(4):
            c = np.cos(self.theta)
            s = np.sin(self.theta)
            for lx_, lz_ in corners:
                wx = self.cx + c * lx_ + s * lz_
                wz = self.cz - s * lx_ + c * lz_
                rx = wx - self.cx
                rz = wz - self.cz
                for (nx_, nz_), off in planes:
                    pen = off - (nx_ * wx + nz_ * wz)
                    if pen <= 0.0:
                        continue
                    # normal impulse
                    vpx = self.vx + self.omega * rz
                    vpz = self.vz - self.omega * rx
                    vn = vpx * nx_ + vpz * nz_
                    rn = rz * nx_ - rx * nz_
                    denom = 1.0 / m + rn * rn / inertia
                    j = 0.0
                    if vn < 0.0:
                        j = -vn / denom
                        self.vx += j * nx_ / m
                        self.vz += j * nz_ / m
                        self.omega += j * rn / inertia
                        self.contact_impulse += (j * nx_, 0.0, j * nz_)
                    # friction
                    tx, tz = -nz_, nx_
                    vt = vpx * tx + vpz * tz
                    rt = rz * tx - rx * tz
                    denom_t = 1.0 / m + rt * rt / inertia
                    jt = -vt / denom_t
                    limit = 0.25 * abs(j)
                    jt = max(-limit, min(limit, jt))
                    self.vx += jt * tx / m
                    self.vz += jt * tz / m
                    self.omega += jt * rt / inertia
                    # positional correction (translation only)
                    self.cx += nx_ * pen * 0.4
                    self.cz += nz_ * pen * 0.4
                    wx = self.cx + c * lx_ + s * lz_
                    wz = self.cz - s * lx_ + c * lz_

    def _advect_markers(self, dt):
        px = self.markers[:, 0]
        pz = self.markers[:, 1]
        vx = self._interp_u(px, pz)
        vz = self._interp_w(px, pz)
        mx = px + 0.5 * dt * vx
        mz = pz + 0.5 * dt * vz
        vx2 = self._interp_u(mx, mz)
        vz2 = self._interp_w(mx, mz)
        nxp = px + dt * vx2
        nzp = pz + dt * vz2

        # periodic through-flow wrap: account momentum crossing boundaries
        down = nzp < self.z0
        up = nzp >= self.z1
        if np.any(down):
            mvx = float((self.marker_mass[down] * vx2[down]).sum())
            mvz = float((self.marker_mass[down] * vz2[down]).sum())
            self.cum_outflow[0] += np.array([mvx, 0.0, mvz])   # left at bottom
            self.cum_inflow[0] += np.array([mvx, 0.0, mvz])    # re-enters at top
            nzp[down] += self.Lz
        if np.any(up):
            mvx = float((self.marker_mass[up] * vx2[up]).sum())
            mvz = float((self.marker_mass[up] * vz2[up]).sum())
            self.cum_outflow[0] += np.array([mvx, 0.0, mvz])   # left at top
            self.cum_inflow[0] += np.array([mvx, 0.0, mvz])    # re-enters at bottom
            nzp[up] -= self.Lz
        nxp = np.clip(nxp, self.x0 + 1e-4, self.x1 - 1e-4)

        # keep markers outside the rigid box
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        rx = nxp - self.cx
        rz = nzp - self.cz
        lx = c * rx - s * rz
        lz = s * rx + c * rz
        inside = (np.abs(lx) <= self.hx) & (np.abs(lz) <= self.hz)
        if np.any(inside):
            ex = np.abs(lx) - self.hx
            ez = np.abs(lz) - self.hz
            use_x = ex > ez
            eps = 1e-3
            sx = np.where(lx >= 0.0, 1.0, -1.0)
            sz = np.where(lz >= 0.0, 1.0, -1.0)
            lx2 = np.where(use_x, sx * (self.hx + eps), lx)
            lz2 = np.where(use_x, lz, sz * (self.hz + eps))
            wx = self.cx + c * lx2 + s * lz2
            wz = self.cz - s * lx2 + c * lz2
            nxp = np.where(inside, np.clip(wx, self.x0 + 1e-4, self.x1 - 1e-4), nxp)
            nzp = np.where(inside, wz, nzp)

        self.markers[:, 0] = nxp
        self.markers[:, 1] = nzp

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _substep(self, dt):
        t_mid = self.t + 0.5 * dt
        a_drive = self._drive_accel(t_mid)

        # 1. advection + body forces on the fluid
        self.u, self.w = self._advect(dt)
        self.w += dt * (self.gravity[2] + a_drive[2])

        # 2. face open fractions at the current (pre-move) position
        o_u, o_w = self._face_open(self.cx, self.cz, self.theta)
        phi_old = self.phi

        # 3. rigid integration under gravity, then move
        self.vx += dt * self.gravity[0]
        self.vz += dt * self.gravity[2]
        self.cx += dt * self.vx
        self.cz += dt * self.vz
        self.theta += dt * self.omega
        if self.theta > np.pi:
            self.theta -= 2.0 * np.pi
        elif self.theta < -np.pi:
            self.theta += 2.0 * np.pi
        phi_new = self._cell_phi(self.cx, self.cz, self.theta)
        self.phi = phi_new

        # 4. pressure projection (cut-cell, two-way)
        p2 = self._project(o_u, o_w, phi_old, phi_new, dt)

        # 5. fluid velocities inside the solid follow the rigid motion
        self._enforce_blocked_faces(o_u, o_w)
        np.clip(self.u, -25.0, 25.0, out=self.u)
        np.clip(self.w, -25.0, 25.0, out=self.w)

        # 6. pressure force/torque on the box (two-way coupling)
        Fx, Fz, tau = self._box_force(p2, phi_new)
        self.vx += dt * Fx / self.m_box
        self.vz += dt * Fz / self.m_box
        self.omega += dt * tau / self.I_box
        self.last_fluid_force = np.array([Fx, 0.0, Fz])

        # wall pressure traction on the fluid (budget bookkeeping)
        traction_x = float(
            p2[0, :].sum() * self.dz * self.depth
            - p2[self.nx - 1, :].sum() * self.dz * self.depth
        )
        self.last_traction = np.array([traction_x, 0.0, 0.0])

        # 7. contacts with channel walls / floor / ceiling
        self._contacts()

        # 8. passive markers
        self._advect_markers(dt)

        # 9. budget accumulation (external impulses; interaction excluded)
        m_fluid = self.rho * float((1.0 - phi_new).sum()) * self.Vc
        g = self.gravity
        body_f = np.array([g[0], 0.0, g[2]]) + a_drive
        self.cum_ext_impulse[0] += dt * m_fluid * body_f
        self.cum_ext_impulse[0] += dt * self.last_traction
        self.cum_ext_impulse[1] += dt * self.m_box * np.array([g[0], 0.0, g[2]])

        self.t += dt
        self._steps += 1

    def advance_to(self, time):
        if not self._initialized:
            raise RuntimeError("solver.initialize() was not called")
        target = float(time)
        while self.t < target - 1e-12:
            dt = min(self.dt, target - self.t)
            if dt <= 1e-13:
                break
            self._substep(dt)
        return self._build_frame(self.t)

    # ------------------------------------------------------------------
    # StateFrame export
    # ------------------------------------------------------------------
    def _build_frame(self, t):
        n_m = self.n_markers
        N = n_m + 1
        mx = self.markers[:, 0]
        mz = self.markers[:, 1]
        mvx = self._interp_u(mx, mz)
        mvz = self._interp_w(mx, mz)

        position = np.empty((N, 3), dtype=float)
        position[0] = (self.cx, 0.0, self.cz)
        position[1:, 0] = mx
        position[1:, 1] = 0.0
        position[1:, 2] = mz

        velocity = np.empty((N, 3), dtype=float)
        velocity[0] = (self.vx, 0.0, self.vz)
        velocity[1:, 0] = mvx
        velocity[1:, 1] = 0.0
        velocity[1:, 2] = mvz

        orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (N, 1))
        orientation[0] = (0.0, np.sin(0.5 * self.theta), 0.0, np.cos(0.5 * self.theta))

        angular_velocity = np.zeros((N, 3), dtype=float)
        angular_velocity[0, 1] = self.omega

        mass = np.empty(N, dtype=float)
        mass[0] = self.m_box
        mass[1:] = self.marker_mass

        body_id = np.zeros(N, dtype=np.int64)
        body_id[0] = 1

        phase = np.ones(N, dtype=float)
        phase[0] = 0.0

        # diagnostic resolved velocity field at cell centers, (nz, nx, 3)
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        fluid_velocity = np.empty((self.nz, self.nx, 3), dtype=np.float32)
        fluid_velocity[:, :, 0] = uc.T.astype(np.float32)
        fluid_velocity[:, :, 1] = 0.0
        fluid_velocity[:, :, 2] = self.w.T.astype(np.float32)

        # budget reports
        weights = np.zeros((2, N), dtype=float)
        weights[0, 1:] = 1.0   # fluid marker domain
        weights[1, 0] = 1.0    # rigid box domain

        m_fluid = self.rho * float((1.0 - self.phi).sum()) * self.Vc
        a_drive = self._drive_accel(t)
        g = self.gravity
        f_box = self.last_fluid_force

        dt_frame = max(t - self._frame_time_prev, 1e-9)
        f_contact = (self.contact_impulse - self._contact_reported) / dt_frame
        self._contact_reported = self.contact_impulse.copy()
        self._frame_time_prev = t

        ext = np.zeros((2, 4, 3), dtype=float)
        ext[0, 0] = m_fluid * np.array([g[0], 0.0, g[2]])
        ext[0, 1] = m_fluid * a_drive
        ext[0, 2] = self.last_traction
        ext[0, 3] = -f_box
        ext[1, 0] = self.m_box * np.array([g[0], 0.0, g[2]])
        ext[1, 2] = f_contact
        ext[1, 3] = f_box

        interaction = np.zeros((2, 3), dtype=float)
        interaction[0] = f_box
        interaction[1] = -f_box

        frame = StateFrame(
            time=float(t),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "phase": phase,
                "mass_quantity": mass.copy(),
            },
            fields={
                "fluid_velocity": fluid_velocity,
                "boundary_normal": self.boundary_normal.copy(),
                "boundary_target_velocity": self.boundary_velocity.copy(),
                "boundary_velocity": self.boundary_velocity.copy(),
                "budget_domain_weights": weights,
                "budget_external_force": ext,
                "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
                "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
                "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "interaction_force": interaction,
            },
            scalars={
                "fluid_drive": float(a_drive[2]),
                "reference_velocity": float(self.reference_velocity),
                "reference_momentum": float(self.reference_momentum),
                "grid_spacing": float(self.dx),
                "characteristic_length": float(self.x1 - self.x0),
                "liquid_phase_id": 1.0,
            },
        )
        frame.position = np.nan_to_num(frame.position)
        frame.velocity = np.nan_to_num(frame.velocity)
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "mac_cutcell_two_way_fluid_rigid",
            "substeps": int(self._steps),
            "final_time": float(self.t),
            "box_center": [float(self.cx), 0.0, float(self.cz)],
            "box_angle_deg": float(np.degrees(self.theta)),
            "marker_count": int(self.n_markers),
        }
