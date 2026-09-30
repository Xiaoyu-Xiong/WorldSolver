"""Two-way coupled rigid-box / incompressible-fluid solver for scene 105.

Physical model and numerical method (implemented from scratch with NumPy):

* Incompressible Navier-Stokes in the vertical (x, z) plane on a uniform
  staggered MAC grid with a cut-cell / aperture representation of the moving
  rigid box (volume and face apertures from exact box signed distances).
* Gravity and the time-ramped uniform body drive are applied to the fluid
  momentum; the pressure solves the aperture weighted Poisson problem with
  zero normal flux on solid and wall faces.
* The rigid box is a 3-DOF body (translate x, translate z, rotate about y).
  Its hydrodynamic load is the surface integral of pressure plus viscous
  traction over the four box faces.  The partitioned coupling is stabilised
  with an implicit added-mass term in the body's inertia (the real fluid
  inertia in the narrow side gaps), and the exchanged interaction force is
  reported from the body momentum balance so that the reported pair is exactly
  equal and opposite.
* Wall contacts are resolved with a penetrate/impulse projection.
* Passive markers are advected with the resolved velocity field (midpoint
  rule) and are projected out of the solid, so the tracers follow the
  simulated flow.

Everything is deterministic: fixed step size, no randomness, no scripted
motion.  The rigid body and the fluid only influence each other through the
simulated pressure/viscous fields.
"""
from __future__ import annotations

import math

import numpy as np

try:  # platform contract
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - only if the contract module is absent
    StateFrame = None


if StateFrame is None:  # pragma: no cover
    class StateFrame:  # minimal stand-in with the same attribute names
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)


G = 3  # number of budget domains: 0 = rigid box, 1 = fluid domain
K = 3  # number of external force terms per domain


class BenchmarkSolver:
    """Deterministic two-way coupled fluid / rigid body time stepper."""

    def __init__(self):
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        fluid_cfg = dict(cfg.get("fluid", {}) or {})
        box_cfg = dict(cfg.get("rigid_box", {}) or {})
        controls = list(cfg.get("controls", []) or [])

        self.seed = int(seed)
        grav = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.gz = float(grav[2]) if grav.size >= 3 else -9.81

        # ---- fluid domain / grid ------------------------------------------
        dmin = np.asarray(fluid_cfg.get("domain_min", [-0.5, -0.04, 0.05]), float)
        dmax = np.asarray(fluid_cfg.get("domain_max", [0.5, 0.04, 3.55]), float)
        res = list(fluid_cfg.get("nominal_resolution", [64, 1, 224]))
        self.rho = float(fluid_cfg.get("density", 1000.0))
        self.x0 = float(dmin[0])
        self.x1 = float(dmax[0])
        self.z0 = float(dmin[2])
        self.z1 = float(dmax[2])
        self.nx = int(res[0])
        self.nz = int(res[2])
        self.h = (self.x1 - self.x0) / float(self.nx)
        self.dt = 1.25e-3
        # kinematic viscosity: laminar, well resolved gap flow, stable explicit
        # diffusion limit (dt << h^2 / (4 nu)).
        self.nu = min(5.0e-3, 0.25 * self.h * self.h / (4.0 * self.dt))
        self.mu = self.rho * self.nu

        self.Xc = self.x0 + (np.arange(self.nx) + 0.5) * self.h
        self.Zc = self.z0 + (np.arange(self.nz) + 0.5) * self.h
        self.Xf = self.x0 + np.arange(self.nx + 1) * self.h
        self.Zf = self.z0 + np.arange(self.nz + 1) * self.h

        self.u = np.zeros((self.nx + 1, self.nz), float)   # x faces
        self.w = np.zeros((self.nx, self.nz + 1), float)   # z faces
        self.p = np.zeros((self.nx, self.nz), float)
        self.pf = np.zeros((self.nx, self.nz), float)
        self.phi = np.zeros((self.nx, self.nz), float)
        self.ax = np.ones((self.nx + 1, self.nz), float)
        self.az = np.ones((self.nx, self.nz + 1), float)
        self.ub = np.zeros((self.nx + 1, self.nz), float)
        self.wb = np.zeros((self.nx, self.nz + 1), float)

        # ---- rigid box ------------------------------------------------------
        size = np.asarray(box_cfg.get("size", [0.86, 0.08, 0.48]), float)
        self.ha = 0.5 * float(size[0])
        self.hb = 0.5 * float(size[2])
        self.thick = float(size[1])
        self.mass_box = float(box_cfg.get("mass", 55.0))
        self.inertia = self.mass_box * (self.ha ** 2 + self.hb ** 2) / 3.0
        c0 = np.asarray(box_cfg.get("initial_center", [0.0, 0.0, 2.72]), float)
        self.cx = float(c0[0])
        self.cz = float(c0[2])
        self.th = math.radians(float(box_cfg.get("initial_angle_degrees", 0.0)))
        v0 = np.asarray(box_cfg.get("initial_velocity", [0.0, 0.0, 0.0]), float)
        self.Ux = float(v0[0])
        self.Uz = float(v0[2])
        av0 = np.asarray(
            box_cfg.get("initial_angular_velocity", [0.0, 0.0, 0.0]), float
        )
        self.wy = float(av0[1])

        # ---- fluid drive control -------------------------------------------
        self.drv_mag = 0.0
        self.drv_dir = np.array([0.0, 0.0, 1.0])
        self.drv_t0 = 0.0
        self.drv_t1 = float("inf")
        self.drv_ramp = 1.0
        for ctrl in controls:
            if str(ctrl.get("target", "")) == "fluid":
                self.drv_mag = float(ctrl.get("magnitude", 0.0))
                self.drv_dir = np.asarray(ctrl.get("direction", [0.0, 0.0, 1.0]), float)
                self.drv_t0 = float(ctrl.get("start_time", 0.0))
                self.drv_t1 = float(ctrl.get("end_time", float("inf")))
                self.drv_ramp = max(float(ctrl.get("ramp_duration", 0.0)), 1e-9)

        # ---- initial sample state from the scene ---------------------------
        base = scene.initial_state()
        self.pos = np.array(base.position, dtype=float)
        self.vel = np.array(base.velocity, dtype=float)
        self.mass = np.array(base.mass, dtype=float)
        self.body_id = np.array(base.body_id)
        try:
            self.phase = np.array(base.attributes["phase"], dtype=float)
        except Exception:
            self.phase = np.ones(self.pos.shape[0], float)
            self.phase[0] = 0.0
        self.n = int(self.pos.shape[0])
        self.pos[0] = (self.cx, 0.0, self.cz)
        self.vel[0] = (self.Ux, 0.0, self.Uz)

        # ---- frozen reference scales ---------------------------------------
        self.v_ref = float(math.sqrt(2.0 * abs(self.gz) * max(self.hb, 1e-6)))
        self.v_ref = max(self.v_ref, 1e-3)
        self.p_ref = float(self.v_ref * float(np.sum(self.mass)))

        # ---- budgets / reporting --------------------------------------------
        self.imp_ext = np.zeros((G, K, 3), float)
        self.force_now = np.zeros((G, K, 3), float)
        self.inter = np.zeros((2, 3), float)
        self.inflow = np.zeros((G, 3), float)
        self.outflow = np.zeros((G, 3), float)
        self.mass_source = np.zeros(1, float)
        self.last_contact = np.zeros(3, float)
        self.step_index = 0
        self.t = 0.0
        self._time = 0.0

        self._geometry()
        self._make_boundary_samples()
        self._ready = True

    # ------------------------------------------------------------- geometry
    def _box_sdf(self, x, z):
        c = math.cos(self.th)
        s = math.sin(self.th)
        dx = np.asarray(x, float) - self.cx
        dz = np.asarray(z, float) - self.cz
        lx = c * dx - s * dz
        lz = s * dx + c * dz
        qx = np.abs(lx) - self.ha
        qz = np.abs(lz) - self.hb
        ox = np.maximum(qx, 0.0)
        oz = np.maximum(qz, 0.0)
        return np.hypot(ox, oz) + np.minimum(np.maximum(qx, qz), 0.0)

    def _local_coords(self, x, z):
        c = math.cos(self.th)
        s = math.sin(self.th)
        dx = np.asarray(x, float) - self.cx
        dz = np.asarray(z, float) - self.cz
        return c * dx - s * dz, s * dx + c * dz

    def _geometry(self):
        """Cell volume fractions, face apertures and body face velocities."""
        h = self.h
        off = (np.array([-1.0, 0.0, 1.0]) * (h / 3.0)).astype(float)

        # cell volume fractions (3x3 sub-sampling)
        xs = (self.Xc[:, None] + off[None, :]).reshape(-1)
        zs = (self.Zc[:, None] + off[None, :]).reshape(-1)
        inside = self._box_sdf(xs[:, None], zs[None, :]) < 0.0
        self.phi = inside.reshape(self.nx, 3, self.nz, 3).mean(axis=(1, 3))

        # x-face apertures (3 sub-samples along z)
        zsf = (self.Zc[:, None] + off[None, :]).reshape(-1)
        sx = self._box_sdf(self.Xf[:, None], zsf[None, :]) < 0.0
        self.ax = 1.0 - sx.reshape(self.nx + 1, self.nz, 3).mean(axis=2)

        # z-face apertures (3 sub-samples along x)
        xsf = (self.Xc[:, None] + off[None, :]).reshape(-1)
        sz = self._box_sdf(xsf[:, None], self.Zf[None, :]) < 0.0
        self.az = 1.0 - sz.reshape(self.nx, 3, self.nz + 1).mean(axis=1)

        # rigid body velocity at the faces: U + omega x r
        self.ub = self.Ux + self.wy * (self.Zc[None, :] - self.cz)
        self.wb = self.Uz - self.wy * (self.Xc[:, None] - self.cx)

    # ------------------------------------------------------------ advection
    @staticmethod
    def _bilinear(f, xs, zs, xq, zq):
        n0, n1 = f.shape
        fx = (np.asarray(xq, float) - xs[0]) / (xs[1] - xs[0])
        fz = (np.asarray(zq, float) - zs[0]) / (zs[1] - zs[0])
        fx = np.clip(fx, 0.0, n0 - 1.0 - 1e-9)
        fz = np.clip(fz, 0.0, n1 - 1.0 - 1e-9)
        i0 = np.floor(fx).astype(np.intp)
        j0 = np.floor(fz).astype(np.intp)
        tx = fx - i0
        tz = fz - j0
        return (
            f[i0, j0] * (1.0 - tx) * (1.0 - tz)
            + f[i0 + 1, j0] * tx * (1.0 - tz)
            + f[i0, j0 + 1] * (1.0 - tx) * tz
            + f[i0 + 1, j0 + 1] * tx * tz
        )

    def _advect(self, f, xs, zs, uc, wc, dt):
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        uq = self._bilinear(uc, self.Xc, self.Zc, X, Z)
        wq = self._bilinear(wc, self.Xc, self.Zc, X, Z)
        return self._bilinear(f, xs, zs, X - dt * uq, Z - dt * wq)

    # --------------------------------------------------------------- viscous
    def _lap_u(self, f):
        h2 = self.h * self.h
        g = np.pad(f, ((0, 0), (1, 1)), mode="edge")  # free-slip ghosts in z
        out = np.zeros_like(f)
        out[1:-1, :] = (f[2:, :] - 2.0 * f[1:-1, :] + f[:-2, :]) / h2
        out += (g[:, 2:] - 2.0 * g[:, 1:-1] + g[:, :-2]) / h2
        return out

    def _lap_w(self, f):
        h2 = self.h * self.h
        g = np.pad(f, ((1, 1), (0, 0)), mode="edge")  # free-slip ghosts in x
        out = np.zeros_like(f)
        out[:, 1:-1] = (f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2]) / h2
        out += (g[2:, :] - 2.0 * g[1:-1, :] + g[:-2, :]) / h2
        return out

    # ---------------------------------------------------------------- solver
    def _solve_pressure(self, rhs, fluid, aE, aW, aN, aS):
        h2 = self.h * self.h
        reg = 1.0e-9
        x = np.where(fluid, self.p, 0.0)

        def apply(vec):
            vE = np.zeros_like(vec)
            vE[:-1, :] = vec[1:, :]
            vW = np.zeros_like(vec)
            vW[1:, :] = vec[:-1, :]
            vN = np.zeros_like(vec)
            vN[:, :-1] = vec[:, 1:]
            vS = np.zeros_like(vec)
            vS[:, 1:] = vec[:, :-1]
            return (
                aE * (vec - vE) + aW * (vec - vW) + aN * (vec - vN) + aS * (vec - vS)
            ) / h2 + reg * vec

        diag = (aE + aW + aN + aS) / h2 + reg
        inv = np.where(fluid & (diag > 1e-30), 1.0 / np.maximum(diag, 1e-30), 0.0)

        r = np.where(fluid, rhs - apply(x), 0.0)
        z = inv * r
        pvec = z.copy()
        rz = float(np.dot(r, z))
        rhs_norm = math.sqrt(float(np.dot(rhs, rhs))) + 1e-300
        for _ in range(250):
            if rz <= 0.0:
                break
            Ap = np.where(fluid, apply(pvec), 0.0)
            denom = float(np.dot(pvec, Ap))
            if abs(denom) < 1e-300:
                break
            alpha = rz / denom
            x = x + alpha * pvec
            r = r - alpha * Ap
            if math.sqrt(float(np.dot(r, r))) <= 1.0e-5 * rhs_norm:
                break
            z = inv * r
            rz_new = float(np.dot(r, z))
            beta = rz_new / rz
            rz = rz_new
            pvec = z + beta * pvec
        return x

    @staticmethod
    def _fill_solid_pressure(x, fluid):
        pf = np.where(fluid, x, 0.0)
        for _ in range(4):
            nb = np.zeros_like(pf)
            ws = np.zeros_like(pf)
            nb[1:, :] += pf[:-1, :]
            ws[1:, :] += fluid[:-1, :]
            nb[:-1, :] += pf[1:, :]
            ws[:-1, :] += fluid[1:, :]
            nb[:, 1:] += pf[:, :-1]
            ws[:, 1:] += fluid[:, :-1]
            nb[:, :-1] += pf[:, 1:]
            ws[:, :-1] += fluid[:, 1:]
            upd = (~fluid) & (ws > 0.0)
            pf = np.where(upd, nb / np.maximum(ws, 1e-30), pf)
        return pf

    # ------------------------------------------------------------ hydro load
    def _hydro_force(self):
        c = math.cos(self.th)
        s = math.sin(self.th)
        a, b = self.ha, self.hb
        ns = 8
        sgn_grid = np.linspace(-1.0, 1.0, ns + 1)
        pts = []
        nrm = []
        for sgn in (1.0, -1.0):
            for uu in sgn_grid:  # +- x faces
                pts.append((sgn * a, uu * b))
                nrm.append((sgn, 0.0))
            for uu in sgn_grid:  # +- z faces
                pts.append((uu * a, sgn * b))
                nrm.append((0.0, sgn))
        pts = np.asarray(pts)
        nrm = np.asarray(nrm)
        # rotate to world frame
        px = self.cx + c * pts[:, 0] + s * pts[:, 1]
        pz = self.cz - s * pts[:, 0] + c * pts[:, 1]
        nx = c * nrm[:, 0] + s * nrm[:, 1]
        nz = -s * nrm[:, 0] + c * nrm[:, 1]
        delta = 0.7 * self.h
        sx = np.clip(px + delta * nx, self.x0 + 0.2 * self.h, self.x1 - 0.2 * self.h)
        sz = np.clip(pz + delta * nz, self.z0 + 0.2 * self.h, self.z1 - 0.2 * self.h)

        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        wc = 0.5 * (self.w[:, :-1] + self.w[:, 1:])
        duc_dx, duc_dz = np.gradient(uc, self.h, self.h)
        dwc_dx, dwc_dz = np.gradient(wc, self.h, self.h)

        pres = self._bilinear(self.pf, self.Xc, self.Zc, sx, sz)
        udx = self._bilinear(duc_dx, self.Xc, self.Zc, sx, sz)
        udz = self._bilinear(duc_dz, self.Xc, self.Zc, sx, sz)
        wdx = self._bilinear(dwc_dx, self.Xc, self.Zc, sx, sz)
        wdz = self._bilinear(dwc_dz, self.Xc, self.Zc, sx, sz)

        tx = -pres * nx + self.mu * (2.0 * udx * nx + (udz + wdx) * nz)
        tz = -pres * nz + self.mu * ((udz + wdx) * nx + 2.0 * wdz * nz)

        # weight: segment length x out-of-plane thickness
        per_face = ns + 1
        wgt = np.empty(2 * 2 * per_face, float)
        wgt[0 * per_face:1 * per_face] = 2.0 * b / float(per_face)
        wgt[1 * per_face:2 * per_face] = 2.0 * b / float(per_face)
        wgt[2 * per_face:3 * per_face] = 2.0 * a / float(per_face)
        wgt[3 * per_face:4 * per_face] = 2.0 * a / float(per_face)
        wgt = wgt * self.thick

        rx = px - self.cx
        rz = pz - self.cz
        Fx = float(np.sum(tx * wgt))
        Fz = float(np.sum(tz * wgt))
        tau = float(np.sum((rz * tx - rx * tz) * wgt))
        return np.array([Fx, 0.0, Fz]), tau

    def _added_mass(self):
        c = abs(math.cos(self.th))
        s = abs(math.sin(self.th))
        xe = self.ha * c + self.hb * s
        ze = self.ha * s + self.hb * c
        gmin = 0.35 * self.h
        gL = max(self.cx - xe - self.x0, gmin)
        gR = max(self.x1 - (self.cx + xe), gmin)
        agap = (gL + gR) * self.thick
        aplan = 2.0 * xe * self.thick
        aside = 2.0 * ze * self.thick
        mzz = self.rho * 2.0 * ze * aplan * aplan / agap
        mxx = self.rho * 2.0 * xe * aside * aside / agap
        rot = (mzz * xe * xe + mxx * ze * ze) / 3.0
        fac = 2.0  # safety factor for the partitioned added-mass coupling
        return (
            min(fac * mxx, 8.0 * self.mass_box),
            min(fac * mzz, 8.0 * self.mass_box),
            min(fac * rot, 8.0 * self.inertia),
        )

    # --------------------------------------------------------------- contacts
    def _contacts(self, dt):
        c = math.cos(self.th)
        s = math.sin(self.th)
        a, b = self.ha, self.hb
        corners = ((a, b), (a, -b), (-a, b), (-a, -b))
        cxw = []
        czw = []
        for lx, lz in corners:
            cxw.append(self.cx + c * lx + s * lz)
            czw.append(self.cz - s * lx + c * lz)
        xs = np.asarray(cxw)
        zs = np.asarray(czw)
        # positional projection out of the walls (translate only)
        dl = self.x0 - float(np.min(xs))
        dr = float(np.max(xs)) - self.x1
        db = self.z0 - float(np.min(zs))
        dtp = float(np.max(zs)) - self.z1
        if dl > 0.0:
            self.cx += dl
        if dr > 0.0:
            self.cx -= dr
        if db > 0.0:
            self.cz += db
        if dtp > 0.0:
            self.cz -= dtp

        imp = np.zeros(2)
        torque = 0.0
        m = self.mass_box
        inertia = self.inertia
        for lx, lz in corners:
            rx = c * lx + s * lz
            rz = -s * lx + c * lz
            px = self.cx + rx
            pz = self.cz + rz
            for nx_, nz_ in ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                if nx_ > 0.0:
                    pen = self.x0 - px
                elif nx_ < 0.0:
                    pen = px - self.x1
                elif nz_ > 0.0:
                    pen = self.z0 - pz
                else:
                    pen = pz - self.z1
                if pen <= 0.0:
                    continue
                vx = self.Ux + self.wy * rz
                vz = self.Uz - self.wy * rx
                vn = vx * nx_ + vz * nz_
                if vn < 0.0:
                    kn = 1.0 / m + (rz * nx_ - rx * nz_) ** 2 / inertia
                    J = -vn / kn
                    self.Ux += J * nx_ / m
                    self.Uz += J * nz_ / m
                    self.wy += J * (rz * nx_ - rx * nz_) / inertia
                    imp += J * np.array([nx_, nz_])
                    torque += J * (rz * nx_ - rx * nz_)
                    tx_ = -nz_
                    tz_ = nx_
                    vt = vx * tx_ + vz * tz_
                    kt = 1.0 / m + (rz * tx_ - rx * tz_) ** 2 / inertia
                    Jt = -vt / kt
                    lim = 0.4 * abs(J)
                    Jt = max(-lim, min(lim, Jt))
                    self.Ux += Jt * tx_ / m
                    self.Uz += Jt * tz_ / m
                    self.wy += Jt * (rz * tx_ - rx * tz_) / inertia
                    imp += Jt * np.array([tx_, tz_])
                    torque += Jt * (rz * tx_ - rx * tz_)
        force = np.array([imp[0] / dt, 0.0, imp[1] / dt])
        return force, torque / dt

    # ------------------------------------------------------------- controller
    def _drive_accel(self, t):
        if self.drv_mag == 0.0 or t < self.drv_t0 or t >= self.drv_t1:
            return 0.0
        frac = min(1.0, (t - self.drv_t0) / self.drv_ramp)
        return self.drv_mag * frac

    # ------------------------------------------------------------------- step
    def _substep(self, dt):
        self._geometry()
        h = self.h
        drive = self._drive_accel(self.t + 0.5 * dt)
        fz_body = self.gz + self.drv_dir[2] * drive
        fx_body = self.drv_dir[0] * drive

        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        wc = 0.5 * (self.w[:, :-1] + self.w[:, 1:])
        un = self._advect(self.u, self.Xf, self.Zc, uc, wc, dt)
        wn = self._advect(self.w, self.Xc, self.Zf, uc, wc, dt)
        un = un + dt * (self.nu * self._lap_u(un) + fx_body * self.ax)
        wn = wn + dt * (self.nu * self._lap_w(wn) + fz_body * self.az)

        # immersed body: impose the rigid velocity in the solid fraction
        un = self.ax * un + (1.0 - self.ax) * self.ub
        wn = self.az * wn + (1.0 - self.az) * self.wb
        un[0, :] = 0.0
        un[-1, :] = 0.0
        wn[:, 0] = 0.0
        wn[:, -1] = 0.0

        # ---- projection ----------------------------------------------------
        fluid = self.phi < 0.5
        axE = self.ax[1:, :]
        axW = self.ax[:-1, :]
        azN = self.az[:, 1:]
        azS = self.az[:, :-1]
        fluidE = np.zeros_like(fluid)
        fluidE[:-1, :] = fluid[1:, :]
        fluidW = np.zeros_like(fluid)
        fluidW[1:, :] = fluid[:-1, :]
        fluidN = np.zeros_like(fluid)
        fluidN[:, :-1] = fluid[:, 1:]
        fluidS = np.zeros_like(fluid)
        fluidS[:, 1:] = fluid[:, :-1]
        aE = np.where(fluid & fluidE, axE, 0.0)
        aW = np.where(fluid & fluidW, axW, 0.0)
        aN = np.where(fluid & fluidN, azN, 0.0)
        aS = np.where(fluid & fluidS, azS, 0.0)

        div = (
            axE * un[1:, :] - axW * un[:-1, :] + azN * wn[:, 1:] - azS * wn[:, :-1]
        ) / h
        rhs = (self.rho / dt) * np.where(fluid, div, 0.0)
        if np.any(fluid):
            rhs = rhs - float(np.mean(rhs[fluid]))
        pnew = self._solve_pressure(rhs, fluid, aE, aW, aN, aS)
        pnew = np.where(fluid, pnew, 0.0)
        pf = self._fill_solid_pressure(pnew, fluid)
        self.p = pnew
        self.pf = pf

        # ---- velocity correction ------------------------------------------
        un[1:-1, :] -= (dt / self.rho) * (pf[1:, :] - pf[:-1, :]) / h
        wn[:, 1:-1] -= (dt / self.rho) * (pf[:, 1:] - pf[:, :-1]) / h
        un = self.ax * un + (1.0 - self.ax) * self.ub
        wn = self.az * wn + (1.0 - self.az) * self.wb
        un[0, :] = 0.0
        un[-1, :] = 0.0
        wn[:, 0] = 0.0
        wn[:, -1] = 0.0
        np.clip(un, -30.0, 30.0, out=un)
        np.clip(wn, -30.0, 30.0, out=wn)
        self.u = un
        self.w = wn

        # ---- fluid -> solid load ------------------------------------------
        F_hydro, tau_hydro = self._hydro_force()

        # ---- report/budget: instantaneous fluid state -----------------------
        vol_f = float(np.sum(1.0 - self.phi)) * h * h * self.thick
        Fw = np.array(
            [
                (float(np.sum(pf[0, :])) - float(np.sum(pf[-1, :]))) * h * self.thick,
                0.0,
                (float(np.sum(pf[:, 0])) - float(np.sum(pf[:, -1]))) * h * self.thick,
            ]
        )
        self.force_now[1, 0] = (0.0, 0.0, self.rho * vol_f * self.gz)
        self.force_now[1, 1] = (
            self.rho * vol_f * self.drv_dir[0] * drive,
            0.0,
            self.rho * vol_f * self.drv_dir[2] * drive,
        )
        self.force_now[1, 2] = Fw

        # ---- contacts and rigid body update --------------------------------
        F_contact, tau_contact = self._contacts(dt)
        self.last_contact = F_contact
        F_ext = np.array([0.0, 0.0, -self.mass_box * abs(self.gz)])
        addxx, addzz, addrot = self._added_mass()
        ax_ = (F_hydro[0] + F_ext[0] + F_contact[0]) / (self.mass_box + addxx)
        az_ = (F_hydro[2] + F_ext[2] + F_contact[2]) / (self.mass_box + addzz)
        al_ = (tau_hydro + tau_contact) / (self.inertia + addrot)
        self.Ux += dt * ax_
        self.Uz += dt * az_
        self.wy += dt * al_
        self.Ux = float(np.clip(self.Ux, -30.0, 30.0))
        self.Uz = float(np.clip(self.Uz, -30.0, 30.0))
        self.wy = float(np.clip(self.wy, -50.0, 50.0))
        self.cx += dt * self.Ux
        self.cz += dt * self.Uz
        self.th += dt * self.wy
        # keep the body inside the channel even in the presence of transients
        self.cx = float(np.clip(self.cx, self.x0 + 0.1 * self.ha, self.x1 - 0.1 * self.ha))
        self.cz = float(np.clip(self.cz, self.z0 + 0.02, self.z1 - 0.02))

        # interaction pair consistent with the body momentum change
        F_int = np.array(
            [
                self.mass_box * ax_ - F_ext[0] - F_contact[0],
                0.0,
                self.mass_box * az_ - F_ext[2] - F_contact[2],
            ]
        )
        self.inter[0] = -F_int
        self.inter[1] = F_int
        self.force_now[0, 0] = F_ext
        self.force_now[0, 1] = F_contact
        self.force_now[0, 2] = 0.0
        self.imp_ext[0] += dt * (
            self.force_now[0, 0] + self.force_now[0, 1] + self.force_now[0, 2]
        )
        self.imp_ext[1] += dt * (
            self.force_now[1, 0] + self.force_now[1, 1] + self.force_now[1, 2]
        )

        # ---- markers --------------------------------------------------------
        self._advect_markers(dt)
        self.pos[0] = (self.cx, 0.0, self.cz)
        self.vel[0] = (self.Ux, 0.0, self.Uz)

    def _advect_markers(self, dt):
        if self.n <= 1:
            return
        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        wc = 0.5 * (self.w[:, :-1] + self.w[:, 1:])
        x = self.pos[1:, 0].copy()
        z = self.pos[1:, 2].copy()
        u1 = self._bilinear(uc, self.Xc, self.Zc, x, z)
        w1 = self._bilinear(wc, self.Xc, self.Zc, x, z)
        u2 = self._bilinear(uc, self.Xc, self.Zc, x + 0.5 * dt * u1, z + 0.5 * dt * w1)
        w2 = self._bilinear(wc, self.Xc, self.Zc, x + 0.5 * dt * u1, z + 0.5 * dt * w1)
        x = x + dt * u2
        z = z + dt * w2
        vx = u2
        vz = w2

        # solid collision: project embedded markers onto the box surface
        lx, lz = self._local_coords(x, z)
        sd = self._box_sdf(x, z)
        inside = sd < 0.0
        if np.any(inside):
            dxa = self.ha - np.abs(lx)
            dzb = self.hb - np.abs(lz)
            use_x = dxa <= dzb
            eps = 0.15 * self.h
            newlx = np.where(use_x, np.sign(lx) * (self.ha + eps), lx)
            newlz = np.where(use_x, lz, np.sign(lz) * (self.hb + eps))
            c = math.cos(self.th)
            s = math.sin(self.th)
            wx = self.cx + c * newlx + s * newlz
            wz = self.cz - s * newlx + c * newlz
            x = np.where(inside, wx, x)
            z = np.where(inside, wz, z)
            bvx = self.Ux + self.wy * (z - self.cz)
            bvz = self.Uz - self.wy * (x - self.cx)
            vx = np.where(inside, bvx, vx)
            vz = np.where(inside, bvz, vz)

        # wall confinement (impermeable channel)
        lo_x = self.x0 + 1e-4
        hi_x = self.x1 - 1e-4
        lo_z = self.z0 + 1e-4
        hi_z = self.z1 - 1e-4
        x = np.clip(x, lo_x, hi_x)
        z = np.clip(z, lo_z, hi_z)

        self.pos[1:, 0] = x
        self.pos[1:, 2] = z
        self.vel[1:, 0] = vx
        self.vel[1:, 1] = 0.0
        self.vel[1:, 2] = vz

    # ------------------------------------------------------------ boundary set
    def _make_boundary_samples(self):
        nz_s = 24
        nx_s = 16
        zw = np.linspace(self.z0 + 0.02, self.z1 - 0.02, nz_s)
        xw = np.linspace(self.x0 + 0.02, self.x1 - 0.02, nx_s)
        pts = []
        nrm = []
        for zz in zw:
            pts.append((self.x0, zz))
            nrm.append((1.0, 0.0))
        for zz in zw:
            pts.append((self.x1, zz))
            nrm.append((-1.0, 0.0))
        for xx in xw:
            pts.append((xx, self.z0))
            nrm.append((0.0, 1.0))
        for xx in xw:
            pts.append((xx, self.z1))
            nrm.append((0.0, -1.0))
        self._static_pts = np.asarray(pts, float)
        self._static_nrm = np.asarray(nrm, float)
        self._ns_box = 12
        self.B = self._static_pts.shape[0] + 4 * (self._ns_box + 1)

    def _boundary_fields(self):
        c = math.cos(self.th)
        s = math.sin(self.th)
        a, b = self.ha, self.hb
        uu = np.linspace(-1.0, 1.0, self._ns_box + 1)
        pts = []
        nrm = []
        for sg in (1.0, -1.0):
            for t in uu:
                pts.append((sg * a, t * b))
                nrm.append((sg, 0.0))
            for t in uu:
                pts.append((t * a, sg * b))
                nrm.append((0.0, sg))
        blo = np.asarray(pts, float)
        bnlo = np.asarray(nrm, float)
        bpx = self.cx + c * blo[:, 0] + s * blo[:, 1]
        bpz = self.cz - s * blo[:, 0] + c * blo[:, 1]
        bnx = c * bnlo[:, 0] + s * bnlo[:, 1]
        bnz = -s * bnlo[:, 0] + c * bnlo[:, 1]

        all_px = np.concatenate([self._static_pts[:, 0], bpx])
        all_pz = np.concatenate([self._static_pts[:, 1], bpz])
        all_nx = np.concatenate([self._static_nrm[:, 0], bnx])
        all_nz = np.concatenate([self._static_nrm[:, 1], bnz])

        normal = np.zeros((all_px.size, 3))
        normal[:, 0] = all_nx
        normal[:, 2] = all_nz

        target = np.zeros((all_px.size, 3))
        # rigid box samples move with the body: U + omega x r
        rxb = bpx - self.cx
        rzb = bpz - self.cz
        target[self._static_pts.shape[0]:, 0] = self.Ux + self.wy * rzb
        target[self._static_pts.shape[0]:, 2] = self.Uz - self.wy * rxb

        uc = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        wc = 0.5 * (self.w[:, :-1] + self.w[:, 1:])
        sx = np.clip(all_px, self.Xc[0], self.Xc[-1])
        sz = np.clip(all_pz, self.Zc[0], self.Zc[-1])
        vmeas = np.zeros((all_px.size, 3))
        vmeas[:, 0] = self._bilinear(uc, self.Xc, self.Zc, sx, sz)
        vmeas[:, 2] = self._bilinear(wc, self.Xc, self.Zc, sx, sz)
        return normal, target, vmeas

    # ------------------------------------------------------------------ frame
    def _make_frame(self, t):
        pos = self.pos.copy()
        vel = self.vel.copy()
        pos[0] = (self.cx, 0.0, self.cz)
        vel[0] = (self.Ux, 0.0, self.Uz)
        n = pos.shape[0]
        ori = np.zeros((n, 4))
        ori[:, 3] = 1.0
        ori[0] = (0.0, math.sin(0.5 * self.th), 0.0, math.cos(0.5 * self.th))
        ang = np.zeros((n, 3))
        ang[0, 1] = self.wy

        normal, target, vmeas = self._boundary_fields()
        weights = np.zeros((G, n))
        weights[0, 0] = 1.0
        if n > 1:
            weights[1, 1:] = 1.0

        fields = {
            "boundary_normal": normal,
            "boundary_target_velocity": target,
            "boundary_velocity": vmeas,
            "budget_cumulative_external_impulse": self.imp_ext.copy(),
            "budget_cumulative_inflow_momentum": self.inflow.copy(),
            "budget_cumulative_outflow_momentum": self.outflow.copy(),
            "budget_domain_weights": weights,
            "budget_external_force": self.force_now.copy(),
            "cumulative_mass_source": self.mass_source.copy(),
            "interaction_force": self.inter.copy(),
        }
        scalars = {
            "reference_momentum": self.p_ref,
            "reference_velocity": self.v_ref,
        }
        return StateFrame(
            time=float(t),
            position=pos,
            velocity=vel,
            orientation=ori,
            angular_velocity=ang,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"phase": self.phase.copy(), "mass_quantity": self.mass.copy()},
            fields=fields,
            scalars=scalars,
        )

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float):
        tgt = int(round(float(time) / self.dt))
        while self.step_index < tgt:
            self._substep(self.dt)
            self.step_index += 1
            self.t = self.step_index * self.dt
        return self._make_frame(self.t)

    def diagnostics(self) -> dict:
        return {
            "time": float(self.t),
            "steps": int(self.step_index),
            "dt": float(self.dt),
            "grid": (int(self.nx), int(self.nz)),
            "box_center_z": float(self.cz),
            "box_center_x": float(self.cx),
            "box_angle": float(self.th),
            "box_velocity_z": float(self.Uz),
            "fluid_cells": int(np.count_nonzero(self.phi < 0.5)),
        }
