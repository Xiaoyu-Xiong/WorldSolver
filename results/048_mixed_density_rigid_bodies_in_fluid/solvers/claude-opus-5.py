"""Self-implemented coupled rigid-body / incompressible-fluid solver.

Physics
-------
* Fluid: closed, fully filled no-penetration tank discretized on a uniform
  staggered (MAC) grid.  Each substep performs semi-Lagrangian advection of the
  face velocities, adds the momentum-conserving reaction of the immersed rigid
  bodies, re-imposes no-penetration on the tank walls and finally removes the
  divergence with an exact pressure projection (sparse Poisson solve, Neumann
  boundaries, one pinned pressure degree of freedom).  The pressure is the
  dynamic pressure, i.e. the hydrostatic part is carried analytically, so the
  gravity acting on the fluid is exactly balanced by the hydrostatic gradient.
* Rigid bodies: full 6-DOF Newton-Euler integration.  Forces are gravity plus
  the hydrostatic (buoyancy) pressure resultant, quadratic form drag against the
  local fluid velocity, added-mass inertia, penalty normal + friction contacts
  with the tank walls and pairwise penalty contacts between bodies.
* Markers: passively advected tracers of the fluid velocity used to visualize
  the flow; they are projected out of the rigid bodies.

Everything is stepped explicitly in time with a fixed, stable substep; no
keyframes, no scripted motion, no precomputed trajectories.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from worldsolver.contracts import StateFrame


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from an (x, y, z, w) quaternion."""
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """First order quaternion update for a world-frame angular velocity."""
    x, y, z, w = q
    wx, wy, wz = omega
    dq = 0.5 * np.array(
        [
            w * wx + y * wz - z * wy,
            w * wy + z * wx - x * wz,
            w * wz + x * wy - y * wx,
            -(x * wx + y * wy + z * wz),
        ],
        dtype=float,
    )
    out = q + dt * dq
    n = float(np.linalg.norm(out))
    if not np.isfinite(n) or n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return out / n


def _trilerp(field: np.ndarray, pts: np.ndarray, origin: np.ndarray, dx: float) -> np.ndarray:
    """Clamped trilinear interpolation of a scalar grid sampled at origin+i*dx."""
    shape = field.shape
    g = (pts - origin[None, :]) / dx
    idx = np.floor(g).astype(np.int64)
    for d in range(3):
        np.clip(idx[:, d], 0, shape[d] - 2, out=idx[:, d])
    f = g - idx
    np.clip(f, 0.0, 1.0, out=f)
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    gx, gy, gz = 1.0 - fx, 1.0 - fy, 1.0 - fz
    c000 = field[i, j, k]
    c100 = field[i + 1, j, k]
    c010 = field[i, j + 1, k]
    c110 = field[i + 1, j + 1, k]
    c001 = field[i, j, k + 1]
    c101 = field[i + 1, j, k + 1]
    c011 = field[i, j + 1, k + 1]
    c111 = field[i + 1, j + 1, k + 1]
    return (
        gz * (gy * (gx * c000 + fx * c100) + fy * (gx * c010 + fx * c110))
        + fz * (gy * (gx * c001 + fx * c101) + fy * (gx * c011 + fx * c111))
    )


class _Body:
    """One fully coupled rigid body."""

    __slots__ = (
        "name",
        "shape",
        "radius",
        "height",
        "half",
        "volume",
        "density",
        "mass",
        "added_mass",
        "inertia_body",
        "r_eff",
        "area",
        "x",
        "v",
        "q",
        "omega",
        "wall_force",
        "wall_torque",
        "pair_force",
        "pair_torque",
        "coupling_force",
    )

    def __init__(self, name, shape, radius, height, half, density, x, q):
        self.name = name
        self.shape = shape
        self.radius = float(radius) if radius is not None else 0.0
        self.height = float(height) if height is not None else 0.0
        self.half = np.asarray(half, dtype=float) if half is not None else np.zeros(3)
        if shape == "sphere":
            self.volume = 4.0 / 3.0 * math.pi * self.radius ** 3
        elif shape == "capsule":
            self.volume = (
                math.pi * self.radius ** 2 * self.height
                + 4.0 / 3.0 * math.pi * self.radius ** 3
            )
        else:
            self.volume = 8.0 * float(np.prod(self.half))
        self.density = float(density)
        self.mass = self.density * self.volume
        self.added_mass = 0.5 * self.volume  # times fluid density, filled later
        self.r_eff = (3.0 * self.volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        self.area = math.pi * self.r_eff ** 2
        if shape == "box":
            e = 2.0 * self.half
            self.inertia_body = np.diag(
                self.mass
                / 12.0
                * np.array(
                    [
                        e[1] ** 2 + e[2] ** 2,
                        e[0] ** 2 + e[2] ** 2,
                        e[0] ** 2 + e[1] ** 2,
                    ]
                )
            )
        else:
            i0 = 0.4 * self.mass * self.r_eff ** 2
            self.inertia_body = np.diag([i0, i0, i0])
        self.x = np.asarray(x, dtype=float).copy()
        self.v = np.zeros(3)
        self.q = np.asarray(q, dtype=float).copy()
        self.omega = np.zeros(3)
        self.wall_force = np.zeros(3)
        self.wall_torque = np.zeros(3)
        self.pair_force = np.zeros(3)
        self.pair_torque = np.zeros(3)
        self.coupling_force = np.zeros(3)

    def support(self, rot: np.ndarray) -> np.ndarray:
        """Half-width of the shape along each world axis."""
        if self.shape == "sphere":
            return np.full(3, self.radius)
        if self.shape == "capsule":
            return self.radius + np.abs(rot[:, 2]) * (0.5 * self.height)
        return np.abs(rot) @ self.half


# --------------------------------------------------------------------------- #
# solver
# --------------------------------------------------------------------------- #
_BODY_SPECS = [
    # (name, shape, radius, height, half_extents, translation, rotation_xyzw)
    ("ultralight_black_sphere", "sphere", 0.25, None, None,
     (-0.92, 0.00, 2.14), (0.0, 0.0, 0.0, 1.0)),
    ("light_blue_body", "capsule", 0.22, 0.20, None,
     (-0.88, 0.06, 1.58), (0.116, 0.341, -0.042, 0.932)),
    ("buoyant_green_block", "box", None, None, (0.29, 0.20, 0.18),
     (0.08, 0.00, 1.72), (0.074, 0.194, 0.056, 0.977)),
    ("slightly_heavy_rock", "capsule", 0.24, 0.18, None,
     (0.05, 0.00, 0.78), (-0.07, 0.56, 0.103, 0.819)),
    ("heavy_red_cube", "box", None, None, (0.20, 0.20, 0.20),
     (0.32, -0.02, 0.39), (0.034, 0.07, 0.234, 0.969)),
    ("densest_white_sphere", "sphere", 0.23, None, None,
     (0.95, 0.00, 0.62), (0.0, 0.0, 0.0, 1.0)),
]


class BenchmarkSolver:
    """Rigid-body / fluid coupling solver (pure NumPy/SciPy implementation)."""

    # numerical parameters
    SUBSTEPS = 10          # substeps per sample interval
    DRAG_CD = 1.05         # bluff-body form drag coefficient
    DRAG_VEPS = 0.20       # low-speed regularization (m/s)
    ADDED_MASS_C = 0.5     # added mass coefficient
    FRICTION_MU = 0.35
    V_MAX = 8.0            # safety clamp (m/s)

    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)
        self.rho_f = float(cfg["fluid"]["density"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.sample_dt = float(cfg["sample_dt"])
        self.dt = self.sample_dt / float(self.SUBSTEPS)

        frame0 = scene.initial_state()
        pos0 = np.asarray(frame0.position, dtype=float)
        body_id0 = np.asarray(frame0.body_id, dtype=np.int64)
        self.body_id = body_id0.copy()
        self.n_markers = int(np.count_nonzero(body_id0 == 0))
        self.markers = pos0[: self.n_markers].copy()
        self.marker_vel = np.zeros_like(self.markers)
        self.marker_spacing = float(cfg["fluid"]["marker_spacing"])
        self.marker_mass = self.rho_f * self.marker_spacing ** 3
        self.marker_volume = self.marker_spacing ** 3

        # ---------------- fluid grid (uniform, staggered) ---------------- #
        lo = np.asarray(cfg["fluid"]["bounds_min"], dtype=float)
        hi = np.asarray(cfg["fluid"]["bounds_max"], dtype=float)
        self.dx = 0.1
        dx = self.dx
        self.nx = max(int(round((hi[0] - lo[0]) / dx)), 3)
        self.ny = max(int(round((hi[1] - lo[1]) / dx)), 3)
        self.nz = max(int(round((hi[2] - lo[2]) / dx)), 3)
        self.lo = lo.copy()
        self.hi = lo + dx * np.array([self.nx, self.ny, self.nz], dtype=float)
        nx, ny, nz = self.nx, self.ny, self.nz

        self.u = np.zeros((nx + 1, ny, nz))
        self.v = np.zeros((nx, ny + 1, nz))
        self.w = np.zeros((nx, ny, nz + 1))
        u0 = np.asarray(cfg["fluid"].get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float)
        self.u[1:nx, :, :] = u0[0]
        self.v[:, 1:ny, :] = u0[1]
        self.w[:, :, 1:nz] = u0[2]

        half = 0.5 * dx
        self.origin_u = np.array([lo[0], lo[1] + half, lo[2] + half])
        self.origin_v = np.array([lo[0] + half, lo[1], lo[2] + half])
        self.origin_w = np.array([lo[0] + half, lo[1] + half, lo[2]])
        self.origin_c = lo + half

        self.pos_u = self._face_positions(self.u.shape, self.origin_u)
        self.pos_v = self._face_positions(self.v.shape, self.origin_v)
        self.pos_w = self._face_positions(self.w.shape, self.origin_w)
        self.cell_centers = self._face_positions((nx, ny, nz), self.origin_c)

        self._build_poisson()

        # ---------------- rigid bodies ---------------- #
        ratios = [float(r) for r in cfg["rigid_bodies"]["density_ratios"]]
        specs = self._load_specs()
        self.bodies: list[_Body] = []
        for i, (spec, ratio) in enumerate(zip(specs, ratios)):
            name, shape, radius, height, hx, trans, rot = spec
            body = _Body(name, shape, radius, height, hx, ratio * self.rho_f, trans, rot)
            body.added_mass = self.ADDED_MASS_C * self.rho_f * body.volume
            self.bodies.append(body)
        self.n_bodies = len(self.bodies)
        self.pairs = [
            (i, j) for i in range(self.n_bodies) for j in range(i + 1, self.n_bodies)
        ]
        self.n_interaction = 2 * self.n_bodies + 2 * len(self.pairs)

        self.time = 0.0
        self._steps = 0
        self.characteristic_length = 0.5
        self._update_marker_velocities()

    # ------------------------------------------------------------------ #
    def _load_specs(self):
        """Read the frozen geometry when available, else use the frozen defaults."""
        try:
            import yaml  # noqa: WPS433

            path = Path(__file__).resolve().parent.parent / "scene_geometry.yaml"
            data = yaml.safe_load(path.read_text())
            specs = []
            for obj in data["objects"]:
                if obj.get("semantic") != "rigid":
                    continue
                geo = obj["geometry"]
                tr = obj.get("transform", {})
                specs.append(
                    (
                        obj["id"],
                        geo["type"],
                        geo.get("radius"),
                        geo.get("height"),
                        geo.get("half_extents"),
                        tr.get("translation", [0.0, 0.0, 0.0]),
                        tr.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]),
                    )
                )
            if len(specs) == len(_BODY_SPECS):
                return specs
        except Exception:  # pragma: no cover - fall back to frozen values
            pass
        return _BODY_SPECS

    def _face_positions(self, shape, origin):
        ax = [origin[d] + self.dx * np.arange(shape[d]) for d in range(3)]
        grid = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)
        return grid.reshape(-1, 3)

    def _build_poisson(self):
        nx, ny, nz = self.nx, self.ny, self.nz
        n = nx * ny * nz
        inv = 1.0 / (self.dx * self.dx)
        idx = np.arange(n).reshape(nx, ny, nz)
        rows = []
        cols = []
        vals = []
        diag = np.zeros((nx, ny, nz))
        slabs = [
            (idx[:-1, :, :], idx[1:, :, :], (slice(0, -1), slice(None), slice(None)),
             (slice(1, None), slice(None), slice(None))),
            (idx[:, :-1, :], idx[:, 1:, :], (slice(None), slice(0, -1), slice(None)),
             (slice(None), slice(1, None), slice(None))),
            (idx[:, :, :-1], idx[:, :, 1:], (slice(None), slice(None), slice(0, -1)),
             (slice(None), slice(None), slice(1, None))),
        ]
        for a, b, sa, sb in slabs:
            af = a.ravel()
            bf = b.ravel()
            rows.append(af)
            cols.append(bf)
            vals.append(np.full(af.size, inv))
            rows.append(bf)
            cols.append(af)
            vals.append(np.full(af.size, inv))
            diag[sa] -= inv
            diag[sb] -= inv
        rows.append(idx.ravel())
        cols.append(idx.ravel())
        vals.append(diag.ravel())
        mat = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        ).tolil()
        # pin one pressure degree of freedom (pure Neumann problem)
        mat[0, :] = 0.0
        mat[0, 0] = 1.0
        self.poisson = mat.tocsc()
        self.poisson_solve = spla.factorized(self.poisson)

    # ------------------------------------------------------------------ #
    # fluid helpers
    # ------------------------------------------------------------------ #
    def _sample_velocity(self, pts: np.ndarray) -> np.ndarray:
        return np.stack(
            (
                _trilerp(self.u, pts, self.origin_u, self.dx),
                _trilerp(self.v, pts, self.origin_v, self.dx),
                _trilerp(self.w, pts, self.origin_w, self.dx),
            ),
            axis=-1,
        )

    def _enforce_walls(self):
        self.u[0, :, :] = 0.0
        self.u[-1, :, :] = 0.0
        self.v[:, 0, :] = 0.0
        self.v[:, -1, :] = 0.0
        self.w[:, :, 0] = 0.0
        self.w[:, :, -1] = 0.0

    def _advect(self, dt: float):
        new = []
        for field, pos, origin in (
            (self.u, self.pos_u, self.origin_u),
            (self.v, self.pos_v, self.origin_v),
            (self.w, self.pos_w, self.origin_w),
        ):
            vel = self._sample_velocity(pos)
            back = pos - dt * vel
            np.clip(back, self.lo[None, :], self.hi[None, :], out=back)
            new.append(_trilerp(field, back, origin, self.dx).reshape(field.shape))
        self.u, self.v, self.w = new

    def _divergence(self):
        dx = self.dx
        return (
            (self.u[1:, :, :] - self.u[:-1, :, :])
            + (self.v[:, 1:, :] - self.v[:, :-1, :])
            + (self.w[:, :, 1:] - self.w[:, :, :-1])
        ) / dx

    def _project(self, dt: float):
        div = self._divergence()
        rhs = (self.rho_f / dt) * div.ravel()
        rhs[0] = 0.0
        p = self.poisson_solve(rhs)
        if not np.all(np.isfinite(p)):
            return
        p = p.reshape(self.nx, self.ny, self.nz)
        scale = dt / (self.rho_f * self.dx)
        self.u[1:-1, :, :] -= scale * (p[1:, :, :] - p[:-1, :, :])
        self.v[:, 1:-1, :] -= scale * (p[:, 1:, :] - p[:, :-1, :])
        self.w[:, :, 1:-1] -= scale * (p[:, :, 1:] - p[:, :, :-1])
        self._enforce_walls()

    def _spread_component(self, field, origin, center, radius, value):
        """Add a total momentum impulse `value*rho*dx^3` to one velocity component."""
        if abs(value) < 1e-14:
            return
        axes = []
        for d in range(3):
            n = field.shape[d]
            i0 = int(math.floor((center[d] - radius - origin[d]) / self.dx))
            i1 = int(math.ceil((center[d] + radius - origin[d]) / self.dx))
            i0 = max(i0, 0)
            i1 = min(i1, n - 1)
            if i1 < i0:
                return
            axes.append(np.arange(i0, i1 + 1))
        coords = [origin[d] + self.dx * axes[d] for d in range(3)]
        d2 = (
            (coords[0][:, None, None] - center[0]) ** 2
            + (coords[1][None, :, None] - center[1]) ** 2
            + (coords[2][None, None, :] - center[2]) ** 2
        )
        wgt = np.clip(1.0 - d2 / (radius * radius), 0.0, None) ** 3
        total = float(wgt.sum())
        if total <= 1e-12:
            return
        field[np.ix_(axes[0], axes[1], axes[2])] += value * wgt / total

    def _apply_body_reaction(self, dt: float):
        cell_mass = self.rho_f * self.dx ** 3
        for body in self.bodies:
            force = -body.coupling_force  # reaction of the body on the fluid
            radius = max(1.7 * body.r_eff, 1.6 * self.dx)
            scale = dt / cell_mass
            self._spread_component(self.u, self.origin_u, body.x, radius, scale * force[0])
            self._spread_component(self.v, self.origin_v, body.x, radius, scale * force[1])
            self._spread_component(self.w, self.origin_w, body.x, radius, scale * force[2])

    # ------------------------------------------------------------------ #
    # rigid bodies
    # ------------------------------------------------------------------ #
    def _wall_contacts(self, body, rot):
        """Penalty normal + friction contact with the closed tank."""
        force = np.zeros(3)
        torque = np.zeros(3)
        m_eff = body.mass + body.added_mass
        stiff = 2.0e4 + 6.0e2 * body.mass
        damp = 2.0 * 0.5 * math.sqrt(stiff * m_eff)
        sup = body.support(rot)
        for axis in range(3):
            for sign in (-1.0, 1.0):
                if sign < 0.0:
                    pen = self.lo[axis] - (body.x[axis] - sup[axis])
                else:
                    pen = (body.x[axis] + sup[axis]) - self.hi[axis]
                if pen <= 0.0:
                    continue
                normal = np.zeros(3)
                normal[axis] = -sign  # points away from the wall, into the domain
                r = np.zeros(3)
                r[axis] = sign * sup[axis]
                v_pt = body.v + np.cross(body.omega, r)
                vn = float(np.dot(v_pt, normal))
                fn = stiff * pen - damp * vn
                if fn <= 0.0:
                    continue
                fn = min(fn, 5.0e5)
                f = fn * normal
                v_t = v_pt - vn * normal
                speed = float(np.linalg.norm(v_t))
                if speed > 1e-9:
                    f = f - self.FRICTION_MU * fn * v_t / (speed + 0.05)
                force += f
                torque += np.cross(r, f)
        return force, torque

    def _pair_contacts(self):
        for body in self.bodies:
            body.pair_force = np.zeros(3)
            body.pair_torque = np.zeros(3)
        for i, j in self.pairs:
            a = self.bodies[i]
            b = self.bodies[j]
            d = b.x - a.x
            dist = float(np.linalg.norm(d))
            rsum = a.r_eff + b.r_eff
            if dist >= rsum or dist < 1e-9:
                continue
            normal = d / dist
            pen = rsum - dist
            m_red = 1.0 / (1.0 / (a.mass + a.added_mass) + 1.0 / (b.mass + b.added_mass))
            stiff = 2.0e4 + 6.0e2 * (a.mass + b.mass)
            damp = 2.0 * 0.5 * math.sqrt(stiff * m_red)
            v_rel = float(np.dot(b.v - a.v, normal))
            fn = stiff * pen - damp * v_rel
            if fn <= 0.0:
                continue
            fn = min(fn, 5.0e5)
            f = fn * normal  # on b, opposite on a
            ra = normal * a.r_eff
            rb = -normal * b.r_eff
            b.pair_force = b.pair_force + f
            a.pair_force = a.pair_force - f
            b.pair_torque = b.pair_torque + np.cross(rb, f)
            a.pair_torque = a.pair_torque + np.cross(ra, -f)

    def _step_bodies(self, dt: float):
        self._pair_contacts()
        centers = np.array([b.x for b in self.bodies], dtype=float)
        fluid_vel = self._sample_velocity(centers)
        for bi, body in enumerate(self.bodies):
            rot = _quat_to_matrix(body.q)
            u_rel = fluid_vel[bi] - body.v
            speed = float(np.linalg.norm(u_rel))
            f_drag = (
                0.5 * self.rho_f * self.DRAG_CD * body.area * (speed + self.DRAG_VEPS)
            ) * u_rel
            # gravity + hydrostatic pressure resultant (buoyancy)
            f_body = (body.mass - self.rho_f * body.volume) * self.gravity
            f_wall, t_wall = self._wall_contacts(body, rot)
            body.wall_force = f_wall
            body.wall_torque = t_wall
            f_known = f_body + f_drag + f_wall + body.pair_force
            m_eff = body.mass + body.added_mass
            accel = f_known / m_eff
            f_added = -body.added_mass * accel
            body.coupling_force = f_drag + f_added

            body.v = body.v + dt * accel
            np.clip(body.v, -self.V_MAX, self.V_MAX, out=body.v)
            body.x = body.x + dt * body.v

            # rotational dynamics with fluid angular damping
            inertia_w = rot @ body.inertia_body @ rot.T
            omega = body.omega
            om_mag = float(np.linalg.norm(omega))
            c_rot = 0.6 * self.rho_f * body.r_eff ** 5
            t_drag = -(c_rot * (om_mag + 0.6)) * omega
            torque = t_wall + body.pair_torque + t_drag
            gyro = np.cross(omega, inertia_w @ omega)
            inertia_eff = inertia_w * (1.0 + self.ADDED_MASS_C) + 1e-6 * np.eye(3)
            alpha = np.linalg.solve(inertia_eff, torque - gyro)
            body.omega = omega + dt * alpha
            np.clip(body.omega, -25.0, 25.0, out=body.omega)
            body.q = _quat_integrate(body.q, body.omega, dt)

            # hard safety confinement inside the closed tank
            sup = body.support(_quat_to_matrix(body.q))
            for axis in range(3):
                low = self.lo[axis] + 0.35 * sup[axis]
                high = self.hi[axis] - 0.35 * sup[axis]
                if body.x[axis] < low:
                    body.x[axis] = low
                    body.v[axis] = max(body.v[axis], 0.0)
                elif body.x[axis] > high:
                    body.x[axis] = high
                    body.v[axis] = min(body.v[axis], 0.0)

    # ------------------------------------------------------------------ #
    # markers
    # ------------------------------------------------------------------ #
    def _update_marker_velocities(self):
        self.marker_vel = self._sample_velocity(self.markers)

    def _step_markers(self, dt: float):
        vel = self._sample_velocity(self.markers)
        mid = self.markers + 0.5 * dt * vel
        np.clip(mid, self.lo[None, :] + 1e-4, self.hi[None, :] - 1e-4, out=mid)
        vel = self._sample_velocity(mid)
        self.markers = self.markers + dt * vel
        # keep tracers inside the closed tank
        eps = 0.02 * self.dx
        np.clip(
            self.markers,
            self.lo[None, :] + eps,
            self.hi[None, :] - eps,
            out=self.markers,
        )
        # project tracers out of the rigid bodies
        for body in self.bodies:
            rad = body.r_eff * 1.02
            d = self.markers - body.x[None, :]
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            inside = dist < rad
            if not np.any(inside):
                continue
            di = dist[inside]
            di = np.where(di < 1e-6, 1e-6, di)
            n = d[inside] / di[:, None]
            self.markers[inside] = body.x[None, :] + rad * n
            vel[inside] = vel[inside] + n * np.maximum(
                np.einsum("ij,j->i", n, body.v) - np.einsum("ij,ij->i", n, vel[inside]),
                0.0,
            )[:, None]
        np.clip(
            self.markers,
            self.lo[None, :] + eps,
            self.hi[None, :] - eps,
            out=self.markers,
        )
        self.marker_vel = vel

    # ------------------------------------------------------------------ #
    def _substep(self, dt: float):
        self._step_bodies(dt)
        self._advect(dt)
        self._apply_body_reaction(dt)
        self._enforce_walls()
        self._project(dt)
        self._step_markers(dt)
        self.time += dt
        self._steps += 1

    # ------------------------------------------------------------------ #
    # diagnostics / export
    # ------------------------------------------------------------------ #
    def _cell_velocity(self):
        uc = 0.5 * (self.u[1:, :, :] + self.u[:-1, :, :])
        vc = 0.5 * (self.v[:, 1:, :] + self.v[:, :-1, :])
        wc = 0.5 * (self.w[:, :, 1:] + self.w[:, :, :-1])
        return np.stack((uc, vc, wc), axis=-1)

    def _active_mask(self):
        mask = np.ones((self.nx, self.ny, self.nz), dtype=float)
        pts = self.cell_centers
        for body in self.bodies:
            d = pts - body.x[None, :]
            inside = np.einsum("ij,ij->i", d, d) < body.r_eff ** 2
            if np.any(inside):
                mask.reshape(-1)[inside] = 0.0
        return mask

    def _frame(self, time: float) -> StateFrame:
        nb = self.n_bodies
        nm = self.n_markers
        n = nm + nb

        pos = np.zeros((n, 3))
        vel = np.zeros((n, 3))
        orient = np.zeros((n, 4))
        omega = np.zeros((n, 3))
        mass = np.zeros(n)
        density = np.zeros(n)
        volume = np.zeros(n)
        ext = np.zeros((n, 3))

        pos[:nm] = self.markers
        vel[:nm] = self.marker_vel
        orient[:nm, 3] = 1.0
        mass[:nm] = self.marker_mass
        density[:nm] = self.rho_f
        volume[:nm] = self.marker_volume
        # gravity on a fluid tracer is exactly balanced by the hydrostatic
        # pressure gradient carried analytically -> zero net external load
        ext[:nm] = 0.0

        inter = np.zeros((self.n_interaction, 3))
        for bi, body in enumerate(self.bodies):
            k = nm + bi
            pos[k] = body.x
            vel[k] = body.v
            orient[k] = body.q
            omega[k] = body.omega
            mass[k] = body.mass
            density[k] = body.density
            volume[k] = body.volume
            ext[k] = (
                (body.mass - self.rho_f * body.volume) * self.gravity + body.wall_force
            )
            inter[2 * bi] = body.coupling_force
            inter[2 * bi + 1] = -body.coupling_force
        base = 2 * nb
        # paired body-body contact forces (equal and opposite by construction)
        for pi, (i, j) in enumerate(self.pairs):
            a = self.bodies[i]
            b = self.bodies[j]
            d = b.x - a.x
            dist = float(np.linalg.norm(d))
            rsum = a.r_eff + b.r_eff
            if dist >= rsum or dist < 1e-9:
                continue
            normal = d / dist
            pen = rsum - dist
            m_red = 1.0 / (1.0 / (a.mass + a.added_mass) + 1.0 / (b.mass + b.added_mass))
            stiff = 2.0e4 + 6.0e2 * (a.mass + b.mass)
            damp = 2.0 * 0.5 * math.sqrt(stiff * m_red)
            fn = stiff * pen - damp * float(np.dot(b.v - a.v, normal))
            if fn <= 0.0:
                continue
            fn = min(fn, 5.0e5)
            inter[base + 2 * pi] = fn * normal
            inter[base + 2 * pi + 1] = -fn * normal

        mass_quantity = np.zeros((n, 2))
        mass_quantity[:nm, 0] = self.marker_mass
        mass_quantity[nm:, 1] = mass[nm:]

        phase = np.zeros(n)
        phase[nm:] = 1.0

        frame = StateFrame(
            time=float(time),
            position=pos,
            velocity=vel,
            orientation=orient,
            angular_velocity=omega,
            mass=mass,
            body_id=self.body_id.astype(np.int32),
            attributes={
                "density": density,
                "density_ratio": density / self.rho_f,
                "coupled_phase": phase,
                "phase_id": phase,
                "marker_volume": volume,
                "mass_quantity": mass_quantity,
                "volume_quantity": volume,
            },
            fields={
                "velocity_field": self._cell_velocity(),
                "active_mask": self._active_mask(),
                "external_force": ext,
                "interaction_force": inter,
                "cumulative_mass_source": np.zeros(2),
            },
            scalars={
                "grid_spacing": float(self.dx),
                "characteristic_length": float(self.characteristic_length),
                "fluid_density": float(self.rho_f),
                "liquid_phase_id": 0.0,
                "reference_velocity": 1.0,
                "time_step": float(self.dt),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ #
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        max_steps = int(2.0 * self.SUBSTEPS * (target / self.sample_dt + 2.0)) + 64
        while self.time < target - 1e-9 and guard < max_steps:
            dt = min(self.dt, target - self.time)
            if dt <= 1e-12:
                break
            self._substep(dt)
            guard += 1
        self.time = max(self.time, 0.0)
        return self._frame(target)

    def diagnostics(self) -> dict:
        div = self._divergence()
        return {
            "substeps_taken": int(self._steps),
            "grid_shape": [int(self.nx), int(self.ny), int(self.nz)],
            "grid_spacing": float(self.dx),
            "max_abs_divergence": float(np.max(np.abs(div))),
            "n_markers": int(self.n_markers),
            "n_bodies": int(self.n_bodies),
            "body_heights": [float(b.x[2]) for b in self.bodies],
            "body_density_ratios": [float(b.density / self.rho_f) for b in self.bodies],
        }
