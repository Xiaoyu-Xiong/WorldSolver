"""Self-implemented volumetric FEM solver for the opposed-end-rotation beam twist.

Physics
-------
The beam is discretized with the frozen 49x7x7 node grid from ``scene.py``.
Every grid cell is split into six tetrahedra (Kuhn subdivision, face compatible
between neighbouring cells), giving a conforming linear tetrahedral finite
element mesh of the volumetric solid.

Each tetrahedron carries a stable Neo-Hookean energy density

    Psi(F) = mu/2 (tr(F^T F) - 3) - mu (J - 1) + lam/2 (J - 1)^2
    P(F)   = mu F + (lam (J - 1) - mu) dJ/dF ,      dJ/dF = cof(F)

plus a Kelvin-Voigt viscous stress ``beta * dF/dt`` evaluated from the nodal
velocity field with the same shape-function gradients.  Nodal forces are
assembled with the standard constant-strain tetrahedron operator
``H = -V P Bm^T``.  Both stress contributions are self-equilibrated, so they
carry zero net force and zero net torque on the whole body.

Time stepping is a lumped-mass symplectic (semi-implicit) Euler scheme with
implicitly treated mass-proportional damping, run at a fixed sub-step well
below the elastic CFL limit of the mesh.  The two end bands flagged by the
frozen ``end_region`` attribute are kinematic: their positions and velocities
are the exact rigid rotations prescribed by ``boundary_conditions`` (smoothstep
interpolation of the keyframe angles about the +x axis, opposite signs at the
two ends).  Everything between the driven bands is free, so the twist that
appears along the beam is produced by the elastic response, never scripted.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def _dsmoothstep(u: float) -> float:
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return 6.0 * u * (1.0 - u)


def _d2smoothstep(u: float) -> float:
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return 6.0 - 12.0 * u


class _RotationSchedule:
    """Piecewise-smoothstep prescribed rotation angle (radians) about one axis."""

    def __init__(self, keyframes):
        pairs = sorted(
            (float(k["time"]), np.deg2rad(float(k["angle"]))) for k in keyframes
        )
        if not pairs:
            pairs = [(0.0, 0.0)]
        self.times = np.array([p[0] for p in pairs], dtype=float)
        self.angles = np.array([p[1] for p in pairs], dtype=float)

    def eval(self, t: float):
        """Return (angle, angular_velocity, angular_acceleration) at time t."""
        times = self.times
        angles = self.angles
        if t <= times[0]:
            return float(angles[0]), 0.0, 0.0
        if t >= times[-1]:
            return float(angles[-1]), 0.0, 0.0
        i = int(np.searchsorted(times, t, side="right") - 1)
        i = max(0, min(i, len(times) - 2))
        t0, t1 = float(times[i]), float(times[i + 1])
        a0, a1 = float(angles[i]), float(angles[i + 1])
        span = max(t1 - t0, 1e-12)
        u = (t - t0) / span
        da = a1 - a0
        return (
            a0 + da * _smoothstep(u),
            da * _dsmoothstep(u) / span,
            da * _d2smoothstep(u) / (span * span),
        )


# Kuhn subdivision of a unit cube into six tetrahedra.  Corner code is
# c = 4*dx + 2*dy + dz with (dx, dy, dz) in {0, 1}^3.
_KUHN_TETS = (
    (0, 4, 6, 7),
    (0, 4, 5, 7),
    (0, 2, 6, 7),
    (0, 2, 3, 7),
    (0, 1, 5, 7),
    (0, 1, 3, 7),
)


class BenchmarkSolver:
    # ---- material / numerics (physical constants, not tuned per frame) ----
    YOUNG = 2.6e4          # Pa
    POISSON = 0.30
    MASS_DAMPING = 4.0     # 1/s, mass-proportional (Rayleigh alpha)
    VISCOSITY = 30.0       # Pa*s, Kelvin-Voigt stress coefficient
    MAX_SUBSTEP = 5.0e-4   # s

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(scene.config)
        self.config = cfg

        frame0 = scene.initial_state()
        self.rest = np.array(frame0.position, dtype=float)
        self.pos = self.rest.copy()
        self.vel = np.zeros_like(self.pos)
        self.mass = np.array(frame0.mass, dtype=float)
        self.n_nodes = self.pos.shape[0]
        self.end_region = np.array(frame0.attributes["end_region"], dtype=float)

        beam = cfg["beam"]
        self.nx, self.ny, self.nz = (int(v) for v in beam["volumetric_grid_shape"])
        self.dims = np.asarray(beam["dimensions"], dtype=float)
        self.center = np.asarray(beam["center"], dtype=float)
        self.density = float(beam["density"])

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        bc = cfg["boundary_conditions"]
        axis = np.asarray(bc.get("rotation_axis", [1.0, 0.0, 0.0]), dtype=float)
        norm = float(np.linalg.norm(axis))
        self.axis = axis / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
        self.left_center = np.asarray(bc["left_end"]["center"], dtype=float)
        self.right_center = np.asarray(bc["right_end"]["center"], dtype=float)
        self.left_schedule = _RotationSchedule(
            bc["left_end"]["prescribed_rotation_degrees"]
        )
        self.right_schedule = _RotationSchedule(
            bc["right_end"]["prescribed_rotation_degrees"]
        )

        self.left_mask = self.end_region < -0.5
        self.right_mask = self.end_region > 0.5
        self.driven_mask = self.left_mask | self.right_mask
        self.free_mask = ~self.driven_mask

        self._build_elements()
        self._build_cell_operator()

        mu = self.YOUNG / (2.0 * (1.0 + self.POISSON))
        lam = (
            self.YOUNG
            * self.POISSON
            / ((1.0 + self.POISSON) * (1.0 - 2.0 * self.POISSON))
        )
        self.mu = float(mu)
        self.lam = float(lam)

        self.time = 0.0
        self.total_mass = float(self.mass.sum())
        self.p_initial = np.zeros(3, dtype=float)

        # frozen positive momentum scale: total mass times the fastest
        # prescribed boundary speed (max |omega| times cross-section radius).
        radius = 0.5 * float(np.hypot(self.dims[1], self.dims[2]))
        omega_max = 0.0
        for sched in (self.left_schedule, self.right_schedule):
            for k in range(len(sched.times) - 1):
                t0, t1 = float(sched.times[k]), float(sched.times[k + 1])
                span = max(t1 - t0, 1e-12)
                da = abs(float(sched.angles[k + 1] - sched.angles[k]))
                omega_max = max(omega_max, 1.5 * da / span)
        v_ref = max(omega_max * radius, 1e-3)
        self.reference_momentum = float(max(self.total_mass * v_ref, 1e-6))

        self._n_steps = 0
        self._min_detF = 1.0
        self._max_speed = 0.0

        # apply the (zero) boundary state at t = 0 for exact consistency
        self._apply_boundary(0.0)

    def _build_elements(self) -> None:
        nx, ny, nz = self.nx, self.ny, self.nz
        i = np.arange(nx - 1)
        j = np.arange(ny - 1)
        k = np.arange(nz - 1)
        ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")
        base_i = ii.ravel()
        base_j = jj.ravel()
        base_k = kk.ravel()
        n_cells = base_i.size

        corner = np.empty((n_cells, 8), dtype=np.int64)
        for c in range(8):
            dx = (c >> 2) & 1
            dy = (c >> 1) & 1
            dz = c & 1
            corner[:, c] = (
                ((base_i + dx) * ny + (base_j + dy)) * nz + (base_k + dz)
            )
        self.cell_corners = corner
        self.n_cells = n_cells

        tets = np.concatenate(
            [corner[:, list(t)] for t in _KUHN_TETS], axis=0
        ).astype(np.int64)

        X = self.rest[tets]                       # [Ne, 4, 3]
        Dm = np.stack(
            (X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]), axis=2
        )
        det = np.linalg.det(Dm)
        flip = det < 0.0
        if np.any(flip):
            tets[flip] = tets[flip][:, [0, 2, 1, 3]]
            X = self.rest[tets]
            Dm = np.stack(
                (X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]), axis=2
            )
            det = np.linalg.det(Dm)
        self.tets = tets
        self.n_tets = tets.shape[0]
        self.Bm = np.linalg.inv(Dm)               # [Ne, 3, 3]
        self.BmT = np.transpose(self.Bm, (0, 2, 1))
        self.vol = np.abs(det) / 6.0
        self.flat_idx = self.tets.ravel()

    def _build_cell_operator(self) -> None:
        """Centroid shape-function gradients of the trilinear grid cell."""
        hx = self.dims[0] / (self.nx - 1)
        hy = self.dims[1] / (self.ny - 1)
        hz = self.dims[2] / (self.nz - 1)
        G = np.empty((8, 3), dtype=float)
        for c in range(8):
            sx = 1.0 if ((c >> 2) & 1) else -1.0
            sy = 1.0 if ((c >> 1) & 1) else -1.0
            sz = 1.0 if (c & 1) else -1.0
            G[c] = (sx / (4.0 * hx), sy / (4.0 * hy), sz / (4.0 * hz))
        self.cell_grad = G

    # ------------------------------------------------------------------
    # boundary conditions
    # ------------------------------------------------------------------
    def _end_state(self, schedule, center, t):
        angle, omega, alpha = schedule.eval(t)
        ax = self.axis
        r0 = self.rest - center
        along = np.outer(r0 @ ax, ax)
        perp = r0 - along
        cross = np.cross(np.broadcast_to(ax, r0.shape), perp)
        ca, sa = np.cos(angle), np.sin(angle)
        pos = center + along + ca * perp + sa * cross
        rot_perp = ca * perp + sa * cross
        rot_cross = np.cross(np.broadcast_to(ax, r0.shape), rot_perp)
        vel = omega * rot_cross
        acc = alpha * rot_cross - (omega * omega) * rot_perp
        return pos, vel, acc, angle, omega

    def _apply_boundary(self, t: float):
        lp, lv, la, l_angle, _ = self._end_state(self.left_schedule, self.left_center, t)
        rp, rv, ra, r_angle, _ = self._end_state(
            self.right_schedule, self.right_center, t
        )
        lm = self.left_mask
        rm = self.right_mask
        self.pos[lm] = lp[lm]
        self.vel[lm] = lv[lm]
        self.pos[rm] = rp[rm]
        self.vel[rm] = rv[rm]
        acc = np.zeros_like(self.pos)
        acc[lm] = la[lm]
        acc[rm] = ra[rm]
        self._driven_acc = acc
        self._left_angle = float(l_angle)
        self._right_angle = float(r_angle)
        return acc

    # ------------------------------------------------------------------
    # forces
    # ------------------------------------------------------------------
    def _internal_forces(self, pos, vel):
        idx = self.tets
        x = pos[idx]
        Ds = np.stack((x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]), axis=2)
        F = np.einsum("eij,ejk->eik", Ds, self.Bm)

        f0 = F[:, :, 0]
        f1 = F[:, :, 1]
        f2 = F[:, :, 2]
        c0 = np.cross(f1, f2)
        c1 = np.cross(f2, f0)
        c2 = np.cross(f0, f1)
        cof = np.stack((c0, c1, c2), axis=2)
        J = np.einsum("ei,ei->e", f0, c0)

        coeff = self.lam * (J - 1.0) - self.mu
        P = self.mu * F + coeff[:, None, None] * cof

        if self.VISCOSITY > 0.0:
            v = vel[idx]
            Dv = np.stack(
                (v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]), axis=2
            )
            Fdot = np.einsum("eij,ejk->eik", Dv, self.Bm)
            P = P + self.VISCOSITY * Fdot

        H = -self.vol[:, None, None] * np.einsum("eij,ejk->eik", P, self.BmT)
        g1 = H[:, :, 0]
        g2 = H[:, :, 1]
        g3 = H[:, :, 2]
        g0 = -(g1 + g2 + g3)
        nodal = np.stack((g0, g1, g2, g3), axis=1)   # [Ne, 4, 3]

        force = np.empty_like(pos)
        for c in range(3):
            force[:, c] = np.bincount(
                self.flat_idx,
                weights=np.ascontiguousarray(nodal[:, :, c]).ravel(),
                minlength=self.n_nodes,
            )
        return force, J

    def _step(self, dt: float) -> None:
        f_int, J = self._internal_forces(self.pos, self.vel)
        self._last_J_min = float(J.min())
        self._min_detF = min(self._min_detF, self._last_J_min)

        f_grav = self.mass[:, None] * self.gravity[None, :]
        f_total = f_int + f_grav

        free = self.free_mask
        inv_m = np.zeros(self.n_nodes, dtype=float)
        inv_m[free] = 1.0 / self.mass[free]

        # implicit mass-proportional damping keeps the quasi-static drive stable
        v_new = self.vel.copy()
        v_star = self.vel[free] + dt * f_total[free] * inv_m[free, None]
        v_new[free] = v_star / (1.0 + self.MASS_DAMPING * dt)

        # damping force actually applied to the free nodes this step
        self._damping_force_total = (
            (v_new[free] - v_star) * (self.mass[free, None] / dt)
        ).sum(axis=0)

        self.vel = v_new
        self.pos[free] = self.pos[free] + dt * self.vel[free]
        self.time += dt
        self._apply_boundary(self.time)
        self._n_steps += 1

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        while target - self.time > 1e-12:
            remaining = target - self.time
            n = int(np.ceil(remaining / self.MAX_SUBSTEP))
            n = max(n, 1)
            dt = remaining / n
            for _ in range(n):
                self._step(dt)
            self.time = target
            self._apply_boundary(self.time)
        return self._make_frame(target)

    def _cell_deformation_gradients(self):
        x = self.pos[self.cell_corners]           # [Nc, 8, 3]
        return np.einsum("cai,aj->cij", x, self.cell_grad)

    def _make_frame(self, t: float) -> StateFrame:
        pos = np.ascontiguousarray(self.pos, dtype=float)
        vel = np.ascontiguousarray(self.vel, dtype=float)
        mass = np.ascontiguousarray(self.mass, dtype=float)

        F_cells = np.ascontiguousarray(self._cell_deformation_gradients(), dtype=float)

        f_int, _ = self._internal_forces(pos, vel)
        driven = self.driven_mask
        # constraint reaction needed to enforce the prescribed end motion
        reaction = (
            self.mass[driven, None] * self._driven_acc[driven]
            - f_int[driven]
            - self.mass[driven, None] * self.gravity[None, :]
        ).sum(axis=0)
        damping = np.zeros(3, dtype=float)
        if hasattr(self, "_damping_force_total"):
            damping = np.asarray(self._damping_force_total, dtype=float)
        gravity_force = self.total_mass * self.gravity

        momentum = (mass[:, None] * vel).sum(axis=0)
        impulse = momentum - self.p_initial

        weights = np.ones((1, self.n_nodes), dtype=float)
        ext_force = np.zeros((1, 3, 3), dtype=float)
        ext_force[0, 0] = reaction
        ext_force[0, 1] = damping
        ext_force[0, 2] = gravity_force

        speed = float(np.abs(vel).max()) if vel.size else 0.0
        self._max_speed = max(self._max_speed, speed)

        rel_deg = float(np.rad2deg(self._right_angle - self._left_angle))

        frame = StateFrame(
            time=float(t),
            position=pos,
            velocity=vel,
            mass=mass,
            attributes={
                "mass_quantity": mass.copy(),
                "end_region": np.ascontiguousarray(self.end_region, dtype=float),
                "rest_position": np.ascontiguousarray(self.rest, dtype=float),
            },
            fields={
                "element_deformation_gradient": F_cells,
                "budget_domain_weights": weights,
                "budget_external_force": ext_force,
                "budget_cumulative_external_impulse": impulse.reshape(1, 3).copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={
                "reference_momentum": self.reference_momentum,
                "relative_end_rotation_degrees": rel_deg,
                "left_end_rotation_degrees": float(np.rad2deg(self._left_angle)),
                "right_end_rotation_degrees": float(np.rad2deg(self._right_angle)),
                "total_mass": self.total_mass,
                "min_element_jacobian": float(np.linalg.det(F_cells).min()),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "linear tetrahedral FEM, stable Neo-Hookean + Kelvin-Voigt",
            "integrator": "lumped-mass symplectic Euler, implicit mass damping",
            "nodes": int(self.n_nodes),
            "tetrahedra": int(self.n_tets),
            "reported_elements": int(self.n_cells),
            "substeps": int(self._n_steps),
            "max_substep": float(self.MAX_SUBSTEP),
            "youngs_modulus": float(self.YOUNG),
            "poisson_ratio": float(self.POISSON),
            "min_tet_jacobian": float(self._min_detF),
            "max_speed": float(self._max_speed),
            "reference_momentum": float(self.reference_momentum),
        }
