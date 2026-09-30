"""Michell buckling of a pre-twisted elastic ring.

The ring is simulated as a discrete Cosserat (Kirchhoff) rod.  Every edge of
the closed polyline carries a material director that can rotate about the
local tangent (the twist degree of freedom).  The elastic energy is

    E = 1/2 ks sum (l_i - l_0)^2                  (near inextensibility)
      + alpha / (2 l_0) sum |kappa_b,i|^2          (isotropic bending)
      + beta  / (2 l_0) sum tau_i^2                (material twist)

with the standard discrete curvature binormal

    kappa_b,i = 2 (t_{i-1} x t_i) / (1 + t_{i-1} . t_i)

and a discrete material twist per joint formed from the antisymmetric
combination of neighbouring directors

    tau_i = 1/2 ( m2_{i-1} . u_i  -  u_{i-1} . m2_i ),   m2 = t x u,

which vanishes identically for a parallel transported director field and
reduces to sin(delta) for a relative material rotation delta about the
tangent.  The director is parameterised by one angle theta measured against
the projection of +z onto the plane normal to t, so unit norm and
perpendicularity hold identically.

Time integration is velocity Verlet with momentum conserving internal viscous
damping.  Gravity is zero and there is no support or shape imposing geometry:
the closed, compatibly pre-twisted ring relaxes freely and the Michell
instability carries it out of its plane.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

try:  # optional acceleration; the model is unchanged without numba
    from numba import njit
except Exception:  # pragma: no cover

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def _wrap(func):
            return func

        return _wrap


@njit(cache=False)
def _energy(x, th, tb, ub, m2b, enode, l0, alpha, beta, ks):
    """Total elastic energy and per node energy contributions."""
    n = x.shape[0]
    es = 0.0
    for i in range(n):
        j = i + 1
        if j == n:
            j = 0
        dx = x[j, 0] - x[i, 0]
        dy = x[j, 1] - x[i, 1]
        dz = x[j, 2] - x[i, 2]
        li = math.sqrt(dx * dx + dy * dy + dz * dz)
        dd = li - l0
        es += dd * dd
        tx = dx / li
        ty = dy / li
        tz = dz / li
        tb[i, 0] = tx
        tb[i, 1] = ty
        tb[i, 2] = tz
        # reference director frame: projection of +z onto the normal plane
        at = tz
        b1x = -at * tx
        b1y = -at * ty
        b1z = 1.0 - at * tz
        bn = math.sqrt(b1x * b1x + b1y * b1y + b1z * b1z)
        if bn < 1e-9:
            b1x = 1.0
            b1y = 0.0
            b1z = 0.0
            bn = 1.0
        b1x /= bn
        b1y /= bn
        b1z /= bn
        b2x = ty * b1z - tz * b1y
        b2y = tz * b1x - tx * b1z
        b2z = tx * b1y - ty * b1x
        c = math.cos(th[i])
        s = math.sin(th[i])
        ux = c * b1x + s * b2x
        uy = c * b1y + s * b2y
        uz = c * b1z + s * b2z
        ub[i, 0] = ux
        ub[i, 1] = uy
        ub[i, 2] = uz
        m2b[i, 0] = ty * uz - tz * uy
        m2b[i, 1] = tz * ux - tx * uz
        m2b[i, 2] = tx * uy - ty * ux
        enode[i] = 0.5 * ks * dd * dd
    eb = 0.0
    et = 0.0
    cb = alpha / (2.0 * l0)
    ct = beta / (2.0 * l0)
    for i in range(n):
        ip = i - 1
        if ip < 0:
            ip = n - 1
        ax = tb[ip, 1] * tb[i, 2] - tb[ip, 2] * tb[i, 1]
        ay = tb[ip, 2] * tb[i, 0] - tb[ip, 0] * tb[i, 2]
        az = tb[ip, 0] * tb[i, 1] - tb[ip, 1] * tb[i, 0]
        den = (
            1.0
            + tb[ip, 0] * tb[i, 0]
            + tb[ip, 1] * tb[i, 1]
            + tb[ip, 2] * tb[i, 2]
        )
        inv = 2.0 / den
        kx = ax * inv
        ky = ay * inv
        kz = az * inv
        ebi = kx * kx + ky * ky + kz * kz
        tau = 0.5 * (
            m2b[ip, 0] * ub[i, 0]
            + m2b[ip, 1] * ub[i, 1]
            + m2b[ip, 2] * ub[i, 2]
            - ub[ip, 0] * m2b[i, 0]
            - ub[ip, 1] * m2b[i, 1]
            - ub[ip, 2] * m2b[i, 2]
        )
        eti = tau * tau
        eb += ebi
        et += eti
        enode[i] += cb * ebi + ct * eti
    return 0.5 * ks * es + cb * eb + ct * et


@njit(cache=False)
def _forces(x, th, fx, gth, tb, ub, m2b, enode, l0, alpha, beta, ks, h):
    """Central finite difference gradient of the elastic energy."""
    n = x.shape[0]
    _energy(x, th, tb, ub, m2b, enode, l0, alpha, beta, ks)
    for i in range(n):
        for k in range(3):
            old = x[i, k]
            x[i, k] = old + h
            e1 = _energy(x, th, tb, ub, m2b, enode, l0, alpha, beta, ks)
            x[i, k] = old - h
            e2 = _energy(x, th, tb, ub, m2b, enode, l0, alpha, beta, ks)
            x[i, k] = old
            fx[i, k] = -(e1 - e2) / (2.0 * h)
    for i in range(n):
        old = th[i]
        th[i] = old + h
        e1 = _energy(x, th, tb, ub, m2b, enode, l0, alpha, beta, ks)
        th[i] = old - h
        e2 = _energy(x, th, tb, ub, m2b, enode, l0, alpha, beta, ks)
        th[i] = old
        gth[i] = -(e1 - e2) / (2.0 * h)


@njit(cache=False)
def _accel(v, w, fx, gth, ax, ath, mass, itheta, cl, cv, ctl, ctv):
    """Nodal accelerations including momentum conserving internal damping."""
    n = v.shape[0]
    vm = np.zeros(3)
    wm = 0.0
    for i in range(n):
        vm[0] += v[i, 0]
        vm[1] += v[i, 1]
        vm[2] += v[i, 2]
        wm += w[i]
    inv_n = 1.0 / n
    vm[0] *= inv_n
    vm[1] *= inv_n
    vm[2] *= inv_n
    wm *= inv_n
    for i in range(n):
        im = i - 1
        if im < 0:
            im = n - 1
        ip = i + 1
        if ip >= n:
            ip = 0
        for k in range(3):
            d = cl * (v[im, k] + v[ip, k] - 2.0 * v[i, k]) + cv * (vm[k] - v[i, k])
            ax[i, k] = (fx[i, k] + d) / mass
        gd = ctl * (w[im] + w[ip] - 2.0 * w[i]) + ctv * (wm - w[i])
        ath[i] = (gth[i] + gd) / itheta


@njit(cache=False)
def _advance(
    x,
    v,
    th,
    w,
    nsteps,
    dt,
    l0,
    alpha,
    beta,
    ks,
    mass,
    itheta,
    cl,
    cv,
    ctl,
    ctv,
    h,
    fx,
    gth,
    ax,
    ath,
    tb,
    ub,
    m2b,
    enode,
):
    """Velocity Verlet (kick-drift-kick) time stepping."""
    n = x.shape[0]
    _forces(x, th, fx, gth, tb, ub, m2b, enode, l0, alpha, beta, ks, h)
    _accel(v, w, fx, gth, ax, ath, mass, itheta, cl, cv, ctl, ctv)
    half = 0.5 * dt
    for _ in range(nsteps):
        for i in range(n):
            for k in range(3):
                v[i, k] += half * ax[i, k]
            w[i] += half * ath[i]
        for i in range(n):
            for k in range(3):
                x[i, k] += dt * v[i, k]
            th[i] += dt * w[i]
        _forces(x, th, fx, gth, tb, ub, m2b, enode, l0, alpha, beta, ks, h)
        _accel(v, w, fx, gth, ax, ath, mass, itheta, cl, cv, ctl, ctv)
        for i in range(n):
            for k in range(3):
                v[i, k] += half * ax[i, k]
            w[i] += half * ath[i]


class BenchmarkSolver:
    """Freely relaxing closed pre-twisted elastic ring (Michell instability)."""

    DT = 2.5e-4          # stable explicit step for the bending / twist spectrum
    FD_STEP = 1.0e-6     # central difference step for the generalized forces

    def initialize(self, scene, seed: int) -> None:
        config = dict(scene.config)
        rod = config["rod"]
        initial = config["initial_conditions"]

        self.n = int(rod["vertex_count"])
        self.length = float(rod["length"])
        self.alpha = float(rod["bending_stiffness_alpha"])
        self.beta = float(rod["twisting_stiffness_beta"])
        self.radius = float(rod["radius"])
        self.rc = float(rod["cross_section_radius"])
        self.density = float(rod["density"])
        self.seed = int(seed)

        self.config_initial = initial
        n = self.n
        self.l0 = self.length / n
        area = math.pi * self.rc * self.rc
        self.line_mass = self.density * area
        self.mass = self.line_mass * self.l0
        self.itheta = self.mass * self.rc * self.rc * 0.5
        self.total_mass = self.mass * n
        # near inextensibility: keeps |l_i - l_0| / l_0 far below one percent
        self.ks = 2.0e5

        self.x = self._initial_positions(config, initial)
        self.th = self._initial_theta()

        self.v = np.zeros((n, 3), dtype=np.float64)
        self.w = np.zeros(n, dtype=np.float64)

        self.fx = np.zeros((n, 3), dtype=np.float64)
        self.gth = np.zeros(n, dtype=np.float64)
        self.ax = np.zeros((n, 3), dtype=np.float64)
        self.ath = np.zeros(n, dtype=np.float64)
        self.tb = np.zeros((n, 3), dtype=np.float64)
        self.ub = np.zeros((n, 3), dtype=np.float64)
        self.m2b = np.zeros((n, 3), dtype=np.float64)
        self.enode = np.zeros(n, dtype=np.float64)

        # internal (momentum conserving) viscous damping
        self.cl = 0.5 * self.mass        # relative nodal velocity laplacian
        self.cv = 0.25 * self.mass       # relaxation towards the mean velocity
        self.ctl = 12.5 * self.itheta    # twist laplacian
        self.ctv = 0.5 * self.itheta     # twist relaxation towards the mean rate

        self.time = 0.0
        self.step_count = 0

        self.v_ref = math.sqrt(self.alpha / self.line_mass) / self.radius
        self.p_ref = self.total_mass * self.v_ref
        self.l_ref = self.total_mass * self.radius * self.v_ref
        self.contact_scale = self.rc

        self._sync_derived()

    # ------------------------------------------------------------------
    # initial condition
    # ------------------------------------------------------------------
    def _initial_positions(self, config, initial):
        n = self.n
        phase = 2.0 * math.pi * np.arange(n) / n
        mode = int(initial["out_of_plane_mode"])
        amp = float(initial["out_of_plane_amplitude"])
        r = self.radius
        pos = np.column_stack(
            (
                r * np.cos(phase),
                r * np.sin(phase),
                amp * np.sin(mode * phase),
            )
        )
        return pos + np.asarray(initial["center"], dtype=np.float64)

    def _initial_theta(self):
        """Director angles storing the compatible integer winding twist."""
        n = self.n
        config = self.config_initial
        winding = int(config["material_frame_winding_number"])
        nxt = np.roll(np.arange(n), -1)
        edge = self.x[nxt] - self.x
        length = np.linalg.norm(edge, axis=1)
        tang = edge / length[:, None]
        # transported in plane normal of the unperturbed circle
        e_z = np.array([0.0, 0.0, 1.0])
        n_ref = np.cross(np.broadcast_to(e_z, tang.shape), tang)
        n_ref /= np.linalg.norm(n_ref, axis=1)[:, None]
        psi = 2.0 * math.pi * winding * np.arange(n) / n
        u = np.cos(psi)[:, None] * n_ref + np.sin(psi)[:, None] * np.cross(tang, n_ref)
        at = tang[:, 2]
        b1 = np.column_stack((-at * tang[:, 0], -at * tang[:, 1], 1.0 - at * tang[:, 2]))
        b1 /= np.linalg.norm(b1, axis=1)[:, None]
        b2 = np.cross(tang, b1)
        # unwrap so the stored winding (3 full turns) stays continuous
        return np.unwrap(np.arctan2(np.sum(u * b2, axis=1), np.sum(u * b1, axis=1)))

    # ------------------------------------------------------------------
    # derived geometry used for the exported state
    # ------------------------------------------------------------------
    def _sync_derived(self):
        n = self.n
        x = self.x
        th = self.th
        nxt = np.roll(np.arange(n), -1)
        edge = x[nxt] - x
        length = np.linalg.norm(edge, axis=1)
        tang = edge / length[:, None]
        at = tang[:, 2]
        b1 = np.column_stack((-at * tang[:, 0], -at * tang[:, 1], 1.0 - at * tang[:, 2]))
        bn = np.linalg.norm(b1, axis=1)
        bad = bn < 1e-9
        if np.any(bad):
            b1[bad] = np.array([1.0, 0.0, 0.0])
            bn[bad] = 1.0
        b1 /= bn[:, None]
        b2 = np.cross(tang, b1)
        u = np.cos(th)[:, None] * b1 + np.sin(th)[:, None] * b2
        m2 = np.cross(tang, u)
        tp = np.roll(tang, 1, axis=0)
        m2p = np.roll(m2, 1, axis=0)
        up = np.roll(u, 1, axis=0)
        cr = np.cross(tp, tang)
        den = 1.0 + np.sum(tp * tang, axis=1)
        kappa_b = 2.0 * cr / den[:, None]
        tau = 0.5 * (np.sum(m2p * u, axis=1) - np.sum(up * m2, axis=1))
        cb = self.alpha / (2.0 * self.l0)
        ct = self.beta / (2.0 * self.l0)
        bend = cb * np.sum(kappa_b * kappa_b, axis=1)
        twist = ct * tau * tau
        stretch = 0.5 * self.ks * (length - self.l0) ** 2
        self.edge_length = length
        self.tangent = tang
        self.director = u
        self.kappa_b = kappa_b
        self.tau = tau
        self.bend_node = bend
        self.twist_node = twist
        self.stretch_node = stretch
        self.elastic_node = bend + twist + stretch
        self.elastic_total = float(np.sum(self.elastic_node))
        self.kinetic = 0.5 * self.mass * float(np.sum(self.v * self.v)) + 0.5 * self.itheta * float(
            np.sum(self.w * self.w)
        )

    # ------------------------------------------------------------------
    # solver contract
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target <= self.time + 1e-12:
            return self._frame(target)
        nsteps = int(round((target - self.time) / self.DT))
        if nsteps < 1:
            nsteps = 1
        _advance(
            self.x,
            self.v,
            self.th,
            self.w,
            nsteps,
            self.DT,
            self.l0,
            self.alpha,
            self.beta,
            self.ks,
            self.mass,
            self.itheta,
            self.cl,
            self.cv,
            self.ctl,
            self.ctv,
            self.FD_STEP,
            self.fx,
            self.gth,
            self.ax,
            self.ath,
            self.tb,
            self.ub,
            self.m2b,
            self.enode,
        )
        self.step_count += nsteps
        self.time = self.time + nsteps * self.DT
        self._sync_derived()
        return self._frame(target)

    def _frame(self, time: float):
        n = self.n
        tang = self.tangent
        # intrinsic spin inertia of each rod cross section
        eye = np.eye(3)
        tt = tang[:, :, None] * tang[:, None, :]
        r2 = self.rc * self.rc
        inertia = self.mass * r2 * (0.25 * (eye[None, :, :] - tt) + 0.5 * tt)
        angular_velocity = self.w[:, None] * tang
        zeros_gk = np.zeros((1, 1, 3))
        centroid = np.mean(self.x, axis=0)
        position = np.broadcast_to(centroid, (1, 1, 3)).copy()
        fields = {
            "budget_cumulative_external_angular_impulse": np.zeros((1, 3)),
            "budget_cumulative_external_impulse": np.zeros((1, 3)),
            "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3)),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, n)),
            "budget_external_couple": zeros_gk.copy(),
            "budget_external_force": zeros_gk.copy(),
            "budget_force_position": position,
            "constraint_target": np.full(n, self.l0),
            "constraint_value": self.edge_length.copy(),
            "cumulative_mass_source": np.zeros(1),
        }
        attributes = {
            "arc_length": self.l0 * np.arange(n),
            "elastic_energy": self.elastic_node.copy(),
            "inertia_tensor": inertia,
            "mass_quantity": np.full(n, self.mass),
            "material_frame_angle": self.th.copy(),
            "natural_curvature": np.zeros(n),
            "natural_twist": np.zeros(n),
            "stored_twist_density": np.full(n, np.mean(self.tau) / self.l0),
        }
        scalars = {
            "constraint_scale": self.l0,
            "reference_angular_momentum": self.l_ref,
            "reference_momentum": self.p_ref,
            "contact_length_scale": self.contact_scale,
            "elastic_energy": self.elastic_total,
            "kinetic_energy": self.kinetic,
        }
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            angular_velocity=angular_velocity,
            mass=np.full(n, self.mass),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": int(self.step_count),
            "time": float(self.time),
            "vertex_count": int(self.n),
            "timestep": float(self.DT),
            "stretch_stiffness": float(self.ks),
            "elastic_energy": float(self.elastic_total),
            "kinetic_energy": float(self.kinetic),
            "out_of_plane_rms": float(np.sqrt(np.mean(self.x[:, 2] ** 2))),
            "out_of_plane_span": float(np.max(self.x[:, 2]) - np.min(self.x[:, 2])),
        }
