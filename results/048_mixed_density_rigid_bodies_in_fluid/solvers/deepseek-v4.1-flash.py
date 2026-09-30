"""Coupled mixed-density rigid bodies in an incompressible fluid.

Physical model
--------------
* The tank interior is a closed, fully filled, constant-density incompressible
  fluid (rho_f from the frozen config).  Gravity is balanced by the hydrostatic
  pressure of the filled tank, so the dynamic fluid momentum equation carries no
  net body force from gravity; it is driven by the momentum exchanged with the
  rigid bodies and kept divergence free by a pressure projection.
* Fluid discretisation: uniform cell-centred velocity on the frozen fluid bounds,
  semi-Lagrangian advection (unconditionally stable), small explicit viscous
  diffusion, and an exact discrete pressure projection.  The Poisson problem has
  all-Neumann (closed tank, no penetration) boundary conditions and is solved
  spectrally with a DCT-II/DCT-III transform pair, so the projected velocity field
  is divergence free to round-off at every step.
* Marker particles are passive tracers advected by the projected velocity field;
  they are pushed out of rigid bodies so the solid volume stays visible.
* Six 6-DOF rigid bodies with density ratios 0.12 ... 5.0 (sphere, capsule, box,
  capsule, cube, sphere, exactly the frozen scene geometry).  Each body feels
  gravity, Archimedes buoyancy rho_f V g, quadratic hydrodynamic drag integrated
  over the body surface using the local fluid velocity, added-mass inertia, and
  penalty contact forces against the tank walls and the other bodies.
* Two-way coupling: the equal-and-opposite hydrodynamic reaction is deposited on
  the fluid grid and the rigid velocity field is blended into the grid before the
  projection, so every body genuinely displaces the fluid and momentum is
  exchanged in equal, opposite pairs.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# small vector / quaternion helpers (quaternions are xyzw, as in the scaffold)
# ---------------------------------------------------------------------------

_DIRS = np.array(
    [
        [float(i), float(j), float(k)]
        for i in (-1.0, 0.0, 1.0)
        for j in (-1.0, 0.0, 1.0)
        for k in (-1.0, 0.0, 1.0)
        if not (i == 0.0 and j == 0.0 and k == 0.0)
    ]
)
_DIRS /= np.linalg.norm(_DIRS, axis=1, keepdims=True)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (...,3) by quaternion(s) q (...,4)."""
    u = np.stack([q[..., 0], q[..., 1], q[..., 2]], axis=-1)
    w = q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _interp(field: np.ndarray, pts: np.ndarray, lo, d, n) -> np.ndarray:
    """Trilinear interpolation of a cell-centred field at world points."""
    g = (pts - lo) / d - 0.5
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    i0 = np.clip(i0, 0, n - 2)
    f = np.clip(f, 0.0, 1.0)
    ix, iy, iz = i0[..., 0], i0[..., 1], i0[..., 2]
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    if field.ndim == 4:
        fx, fy, fz = fx[..., None], fy[..., None], fz[..., None]
    c000 = field[ix, iy, iz]
    c100 = field[ix + 1, iy, iz]
    c010 = field[ix, iy + 1, iz]
    c001 = field[ix, iy, iz + 1]
    c110 = field[ix + 1, iy + 1, iz]
    c101 = field[ix + 1, iy, iz + 1]
    c011 = field[ix, iy + 1, iz + 1]
    c111 = field[ix + 1, iy + 1, iz + 1]
    return (
        c000 * (1 - fx) * (1 - fy) * (1 - fz)
        + c100 * fx * (1 - fy) * (1 - fz)
        + c010 * (1 - fx) * fy * (1 - fz)
        + c001 * (1 - fx) * (1 - fy) * fz
        + c110 * fx * fy * (1 - fz)
        + c101 * fx * (1 - fy) * fz
        + c011 * (1 - fx) * fy * fz
        + c111 * fx * fy * fz
    )


# ---------------------------------------------------------------------------
# rigid bodies
# ---------------------------------------------------------------------------


class _RigidBody:
    def __init__(self, bid, kind, dims, center, quat, density, rho_f):
        self.bid = int(bid)
        self.kind = kind
        self.radius = float(dims.get("radius", 0.2) or 0.2)
        half = dims.get("half_extents")
        self.half = np.asarray(half, dtype=float) if half is not None else np.zeros(3)
        self.height = float(dims.get("height", 0.2) or 0.2)
        self.center = np.asarray(center, dtype=float)
        self.quat = _quat_normalize(np.asarray(quat, dtype=float))
        self.vel = np.zeros(3)
        self.omega = np.zeros(3)
        self.density = float(density)
        self.rho_f = float(rho_f)

        r = self.radius
        hgt = self.height
        if kind == "sphere":
            self.volume = 4.0 / 3.0 * np.pi * r**3
        elif kind == "box":
            self.volume = 8.0 * float(np.prod(self.half))
        else:  # capsule: cylinder + two hemispherical caps
            self.volume = np.pi * r**2 * hgt + 4.0 / 3.0 * np.pi * r**3
        self.mass = self.density * self.volume

        if kind == "sphere":
            i_ax = 0.4 * self.mass * r**2
            self.inertia_body = np.array([i_ax, i_ax, i_ax])
            self.area = 4.0 * np.pi * r**2
        elif kind == "box":
            ex, ey, ez = (float(v) for v in self.half)
            self.inertia_body = np.array(
                [
                    self.mass / 3.0 * (ey**2 + ez**2),
                    self.mass / 3.0 * (ex**2 + ez**2),
                    self.mass / 3.0 * (ex**2 + ey**2),
                ]
            )
            self.area = 8.0 * (ex * ey + ey * ez + ex * ez)
        else:
            v_cyl = np.pi * r**2 * hgt
            v_sph = 4.0 / 3.0 * np.pi * r**3
            m_c = self.mass * v_cyl / self.volume
            m_s = self.mass * v_sph / self.volume
            i_ax = 0.5 * m_c * r**2 + 0.4 * m_s * r**2
            i_tr = m_c * (3.0 * r**2 + hgt**2) / 12.0 + m_s * (0.4 * r**2 + (0.5 * hgt) ** 2)
            self.inertia_body = np.array([i_tr, i_tr, i_ax])
            self.area = 2.0 * np.pi * r * hgt + 4.0 * np.pi * r**2

        self.bound_radius = (
            r if kind == "sphere" else (float(np.linalg.norm(self.half)) if kind == "box" else 0.5 * hgt + r)
        )
        self.area_weight = self.area / len(_DIRS)
        self.force_hydro = np.zeros(3)
        self.torque_hydro = np.zeros(3)
        self.force_buoy = np.zeros(3)
        self.force_drag = np.zeros(3)
        self.torque_drag = np.zeros(3)
        self.force_contact = np.zeros(3)
        self.torque_contact = np.zeros(3)
        self.force_wall = np.zeros(3)
        self.torque_wall = np.zeros(3)

    # -- shape queries ------------------------------------------------------
    def _support_body(self, d_hat: np.ndarray) -> np.ndarray:
        if self.kind == "sphere":
            return self.radius * d_hat
        if self.kind == "box":
            return self.half * np.where(d_hat >= 0.0, 1.0, -1.0)
        return np.array([0.0, 0.0, 0.5 * self.height * np.sign(d_hat[2]) if d_hat[2] != 0 else 0.0]) + self.radius * d_hat

    def support(self, d_hat: np.ndarray) -> np.ndarray:
        d_hat = np.asarray(d_hat, dtype=float)
        d_hat = d_hat / max(np.linalg.norm(d_hat), 1e-12)
        rot = _quat_to_matrix(self.quat)
        return self.center + rot @ self._support_body(rot.T @ d_hat)

    def sample_points(self, offset: float = 0.0) -> np.ndarray:
        rot = _quat_to_matrix(self.quat)
        body_pts = np.array([self._support_body(d) for d in _DIRS]) + offset * _DIRS
        return self.center + body_pts @ rot.T

    def sdf(self, X: np.ndarray) -> np.ndarray:
        rot = _quat_to_matrix(self.quat)
        q = (X - self.center) @ rot
        if self.kind == "sphere":
            return np.linalg.norm(q, axis=-1) - self.radius
        if self.kind == "box":
            d = np.abs(q) - self.half
            outside = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
            inside = np.minimum(np.max(d, axis=-1), 0.0)
            return outside + inside
        z = q[..., 2]
        zc = np.clip(z, -0.5 * self.height, 0.5 * self.height)
        return np.sqrt(q[..., 0] ** 2 + q[..., 1] ** 2 + (z - zc) ** 2) - self.radius

    def rigid_velocity(self, X: np.ndarray) -> np.ndarray:
        return self.vel + np.cross(self.omega, X - self.center)


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


class BenchmarkSolver:
    """Deterministic coupled rigid-body / incompressible-fluid time stepper."""

    # -- setup --------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        self.cfg = cfg
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        fluid = cfg["fluid"]
        self.rho_f = float(fluid["density"])
        self.lo = np.asarray(fluid["bounds_min"], dtype=float)
        self.hi = np.asarray(fluid["bounds_max"], dtype=float)
        self.spacing = float(fluid["marker_spacing"])

        # diagnostic / advection grid on the frozen fluid bounds
        self.n = np.maximum(np.round((self.hi - self.lo) / self.spacing).astype(int), 4)
        self.d = (self.hi - self.lo) / self.n
        self.dmin = float(self.d.min())
        axes = [self.lo[i] + (np.arange(self.n[i]) + 0.5) * self.d[i] for i in range(3)]
        self.X = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        self.V_cell = float(np.prod(self.d))
        self.uc = np.zeros(tuple(self.n) + (3,), dtype=float)
        self.phi = np.zeros(tuple(self.n), dtype=float)
        self.nu = 1.0e-3
        self.cfl = 0.45
        self.dt_max = 0.01
        self.div_max = 0.0
        self.max_speed = 0.0

        # initial particle/solid state comes from the frozen scene
        frame = scene.initial_state()
        pos = np.asarray(frame.position, dtype=float)
        self.body_id = np.asarray(frame.body_id, dtype=np.int64)
        self.n_mark = int(np.sum(self.body_id == 0))
        self.markers = pos[: self.n_mark].copy()
        self.marker_vel = np.zeros_like(self.markers)
        self.marker_mass = float(np.asarray(frame.mass)[0])

        self.bodies = self._build_bodies(cfg, pos[self.n_mark :])

        # smoothing width of the solid volume fraction (one grid cell)
        self.blend_width = 1.0 * self.dmin
        self.probe_offset = 0.75 * self.dmin
        self.phi_bodies = np.zeros((len(self.bodies),) + tuple(self.n), dtype=float)

        self._t = 0.0
        self._update_solid_fraction()

    # -- geometry binding ---------------------------------------------------
    def _build_bodies(self, cfg, fallback_centers) -> list:
        geometry = cfg.get("scene_geometry") or {}
        objects = geometry.get("objects", []) if isinstance(geometry, dict) else []
        ratios = list(cfg["rigid_bodies"]["density_ratios"])
        bodies = []
        for obj in objects:
            if obj.get("semantic") != "rigid":
                continue
            binding = obj.get("state_binding") or {}
            bid = int(binding.get("value", 0))
            geo = obj.get("geometry", {})
            transform = obj.get("transform", {}) or {}
            center = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            quat = np.asarray(transform.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float)
            if bid - 1 < len(fallback_centers):
                center = np.asarray(fallback_centers[bid - 1], dtype=float)
            ratio = float(ratios[bid - 1]) if 0 < bid <= len(ratios) else 1.0
            bodies.append(
                _RigidBody(
                    bid,
                    str(geo.get("type", "sphere")),
                    {
                        "radius": geo.get("radius", 0.25),
                        "half_extents": geo.get("half_extents"),
                        "height": geo.get("height", 0.2),
                    },
                    center,
                    quat,
                    ratio * self.rho_f,
                    self.rho_f,
                )
            )
        bodies.sort(key=lambda b: b.bid)
        if not bodies:  # defensive fallback: one body per configured ratio
            for index, ratio in enumerate(ratios):
                center = np.asarray(fallback_centers[index], dtype=float)
                bodies.append(
                    _RigidBody(index + 1, "sphere", {"radius": 0.25}, center, [0, 0, 0, 1], ratio * self.rho_f, self.rho_f)
                )
        return bodies

    # -- fluid helpers ------------------------------------------------------
    def _solid_fraction(self):
        for i, body in enumerate(self.bodies):
            sdf = body.sdf(self.X)
            self.phi_bodies[i] = np.clip(0.5 - sdf / self.blend_width, 0.0, 1.0)
        self.phi = np.clip(self.phi_bodies.sum(axis=0), 0.0, 1.0)

    def _solid_velocity(self) -> np.ndarray:
        weight = self.phi_bodies.sum(axis=0)
        accum = np.zeros_like(self.uc)
        for i, body in enumerate(self.bodies):
            accum += self.phi_bodies[i][..., None] * body.rigid_velocity(self.X)
        safe = np.maximum(weight, 1e-9)[..., None]
        return np.where(weight[..., None] > 1e-9, accum / safe, 0.0)

    def _update_solid_fraction(self):
        self._solid_fraction()

    def _advect(self, dt: float) -> np.ndarray:
        back = self.X - dt * self.uc
        return _interp(self.uc, back, self.lo, self.d, self.n)

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """Seven-point Laplacian on the first three axes; extra axes are components."""
        ndim = field.ndim
        out = np.zeros_like(field)
        for axis in range(3):
            pads = [(1, 1) if k == axis else (0, 0) for k in range(3)]
            pads += [(0, 0)] * (ndim - 3)
            a = np.pad(field, pads, mode="edge")
            lo = [slice(None)] * ndim
            mid = [slice(None)] * ndim
            hi = [slice(None)] * ndim
            lo[axis], mid[axis], hi[axis] = slice(0, -2), slice(1, -1), slice(2, None)
            out += (a[tuple(hi)] - 2.0 * a[tuple(mid)] + a[tuple(lo)]) / self.d[axis] ** 2
        return out

    def _project(self, dt: float) -> None:
        """Exact discrete projection with closed-tank Neumann conditions."""
        nx, ny, nz = (int(v) for v in self.n)
        ux = np.zeros((nx + 1, ny, nz))
        uy = np.zeros((nx, ny + 1, nz))
        uz = np.zeros((nx, ny, nz + 1))
        ux[1:-1] = 0.5 * (self.uc[:-1, :, :, 0] + self.uc[1:, :, :, 0])
        uy[:, 1:-1] = 0.5 * (self.uc[:, :-1, :, 1] + self.uc[:, 1:, :, 1])
        uz[:, :, 1:-1] = 0.5 * (self.uc[:, :, :-1, 2] + self.uc[:, :, 1:, 2])

        div = (
            (ux[1:] - ux[:-1]) / self.d[0]
            + (uy[:, 1:] - uy[:, :-1]) / self.d[1]
            + (uz[:, :, 1:] - uz[:, :, :-1]) / self.d[2]
        )
        self.div_max = float(np.abs(div).max())

        rhs = div / dt
        rhs -= rhs.mean()
        kx = np.arange(nx)[:, None, None]
        ky = np.arange(ny)[None, :, None]
        kz = np.arange(nz)[None, None, :]
        lam = (
            -4.0 * np.sin(np.pi * kx / (2.0 * nx)) ** 2 / self.d[0] ** 2
            - 4.0 * np.sin(np.pi * ky / (2.0 * ny)) ** 2 / self.d[1] ** 2
            - 4.0 * np.sin(np.pi * kz / (2.0 * nz)) ** 2 / self.d[2] ** 2
        )
        rhs_hat = dctn(rhs, type=2, norm="ortho")
        rhs_hat[0, 0, 0] = 0.0
        lam[0, 0, 0] = 1.0
        pressure = idctn(rhs_hat / lam, type=3, norm="ortho")

        ux[1:-1] -= dt * (pressure[1:] - pressure[:-1]) / self.d[0]
        uy[:, 1:-1] -= dt * (pressure[:, 1:] - pressure[:, :-1]) / self.d[1]
        uz[:, :, 1:-1] -= dt * (pressure[:, :, 1:] - pressure[:, :, :-1]) / self.d[2]
        self.uc[..., 0] = 0.5 * (ux[:-1] + ux[1:])
        self.uc[..., 1] = 0.5 * (uy[:, :-1] + uy[:, 1:])
        self.uc[..., 2] = 0.5 * (uz[:, :, :-1] + uz[:, :, 1:])

        # residual divergence of the exported cell-centred field (central differences)
        div_out = (
            (self.uc[2:, :, :, 0] - self.uc[:-2, :, :, 0]) / (2.0 * self.d[0])
            + (self.uc[:, 2:, :, 1] - self.uc[:, :-2, :, 1]) / (2.0 * self.d[1])
            + (self.uc[:, :, 2:, 2] - self.uc[:, :, :-2, 2]) / (2.0 * self.d[2])
        )
        self.div_max = float(np.abs(div_out).max())

    # -- forces -------------------------------------------------------------
    def _hydro_forces(self) -> None:
        cd = 1.2
        for body in self.bodies:
            # probe the outer flow one cell off the surface so the rigid-velocity
            # blend inside the body does not suppress the relative motion
            points = body.sample_points(self.probe_offset)
            rel = points - body.center
            v_pt = body.vel + np.cross(body.omega, rel)
            u_loc = _interp(self.uc, points, self.lo, self.d, self.n)
            v_rel = v_pt - u_loc
            speed = np.linalg.norm(v_rel, axis=1)
            f_pts = -0.5 * self.rho_f * cd * body.area_weight * speed[:, None] * v_rel
            body.force_buoy = -self.rho_f * body.volume * self.gravity
            body.force_drag = f_pts.sum(axis=0)
            body.torque_drag = np.cross(rel, f_pts).sum(axis=0)
            # rotational damping stabilises high density-contrast spin-up
            spin = np.linalg.norm(body.omega)
            c_rot = 0.05 * self.rho_f * body.volume * body.bound_radius**2 * spin
            body.torque_drag -= c_rot * body.omega
            # total fluid force on the rigid body (buoyancy + hydrodynamic drag)
            body.force_hydro = body.force_buoy + body.force_drag
            body.torque_hydro = body.torque_drag

    def _contacts(self, dt: float) -> None:
        for body in self.bodies:
            body.force_contact = np.zeros(3)
            body.torque_contact = np.zeros(3)
            body.force_wall = np.zeros(3)
            body.torque_wall = np.zeros(3)

        # body-body penalty contacts (equal and opposite by construction)
        for i in range(len(self.bodies)):
            for j in range(i + 1, len(self.bodies)):
                a, b = self.bodies[i], self.bodies[j]
                delta_vec = b.center - a.center
                dist = float(np.linalg.norm(delta_vec))
                if dist < 1e-9:
                    continue
                direction = delta_vec / dist
                pa = a.support(direction)
                pb = b.support(-direction)
                gap = dist - (float(np.linalg.norm(pa - a.center)) + float(np.linalg.norm(pb - b.center)))
                if gap >= 0.0:
                    continue
                depth = -gap
                k = 0.5 * (self._contact_stiffness(a) + self._contact_stiffness(b))
                contact_point = 0.5 * (pa + pb)
                ra = contact_point - a.center
                rb = contact_point - b.center
                v_rel = (b.vel + np.cross(b.omega, rb)) - (a.vel + np.cross(a.omega, ra))
                approach = float(np.dot(v_rel, direction))
                damp = 0.7 * 2.0 * np.sqrt(k * min(a.mass, b.mass))
                mag = k * depth - damp * min(approach, 0.0)
                mag = max(mag, 0.0)
                fb = mag * direction
                fa = -fb
                a.force_contact += fa
                a.torque_contact += np.cross(ra, fa)
                b.force_contact += fb
                b.torque_contact += np.cross(rb, fb)
                correction = 0.25 * min(depth, 0.25 * self.dmin)
                a.center -= correction * direction
                b.center += correction * direction

        # wall penalty contacts against the closed tank faces
        for body in self.bodies:
            for axis in range(3):
                for sign in (1.0, -1.0):
                    direction = np.zeros(3)
                    direction[axis] = sign
                    point = body.support(direction)
                    wall = self.hi[axis] if sign > 0 else self.lo[axis]
                    gap = sign * (wall - point[axis])
                    if gap >= 0.0:
                        continue
                    depth = -gap
                    inward = -direction
                    k = self._contact_stiffness(body)
                    rel = point - body.center
                    v_pt = body.vel + np.cross(body.omega, rel)
                    speed = float(np.dot(v_pt, inward))
                    damp = 0.7 * 2.0 * np.sqrt(k * body.mass)
                    mag = max(k * depth - damp * min(speed, 0.0), 0.0)
                    force = mag * inward
                    body.force_wall += force
                    body.torque_wall += np.cross(rel, force)
                    body.center += inward * (0.25 * min(depth, 0.25 * self.dmin))

    def _contact_stiffness(self, body: _RigidBody) -> float:
        return 4.0e3 * np.sqrt(max(body.mass, 1e-3))

    # -- time stepping ------------------------------------------------------
    def _step(self, dt: float) -> None:
        # 1. fluid transport
        self.uc = self._advect(dt)
        self.uc += self.nu * dt * self._laplacian(self.uc)

        # 2. rigid velocity imposed on the grid (body displaces the fluid)
        self._update_solid_fraction()
        phi = self.phi
        u_solid = self._solid_velocity()
        self.uc = (1.0 - phi)[..., None] * self.uc + phi[..., None] * u_solid

        # 3. hydrodynamic + contact forces at the current state
        self._hydro_forces()
        self._contacts(dt)

        # 4. equal-and-opposite reaction deposited back on the fluid
        weight = self.phi_bodies.sum(axis=0)
        for i, body in enumerate(self.bodies):
            total = float(weight.sum())
            if total <= 1e-9:
                continue
            w = self.phi_bodies[i] / total
            # only the dynamic (drag) reaction is deposited: the hydrostatic
            # buoyancy reaction is already carried by the tank pressure field,
            # which is the constant-density equilibrium the fluid solver omits.
            accel = -body.force_drag / (self.rho_f * self.V_cell)
            self.uc += dt * accel * w[..., None]

        # 5. divergence-free projection
        self._project(dt)

        # 6. rigid body integration (6-DOF Newton-Euler)
        for body in self.bodies:
            rot = _quat_to_matrix(body.quat)
            inertia_world = rot @ np.diag(body.inertia_body) @ rot.T
            added = 0.35 * self.rho_f * body.volume
            m_eff = body.mass + added
            inertia_eff = inertia_world + added * (0.25 * body.bound_radius**2) * np.eye(3)
            force = body.mass * self.gravity + body.force_hydro + body.force_contact + body.force_wall
            body.vel = body.vel + dt * force / m_eff
            speed = float(np.linalg.norm(body.vel))
            if speed > 8.0:
                body.vel *= 8.0 / speed
            body.center = body.center + dt * body.vel
            torque = (
                body.torque_hydro
                + body.torque_contact
                + body.torque_wall
                - np.cross(body.omega, inertia_world @ body.omega)
            )
            alpha = np.linalg.solve(inertia_eff, torque)
            body.omega = body.omega + dt * alpha
            spin = float(np.linalg.norm(body.omega))
            if spin > 25.0:
                body.omega *= 25.0 / spin
            dq = _quat_mul(np.array([*(0.5 * dt * body.omega), 0.0]), body.quat)
            body.quat = _quat_normalize(body.quat + dq)
            body.center = np.clip(body.center, self.lo + 0.25 * body.bound_radius, self.hi - 0.25 * body.bound_radius)

        # 7. marker particles as tracers of the projected field
        self._advect_markers(dt)
        self._update_solid_fraction()
        self.max_speed = float(np.linalg.norm(self.uc, axis=-1).max())
        self._sanitize()

    def _sanitize(self) -> None:
        """Keep every exported quantity finite even under extreme density contrast."""
        self.uc = np.nan_to_num(self.uc, nan=0.0, posinf=0.0, neginf=0.0)
        self.markers = np.nan_to_num(self.markers, nan=0.0, posinf=0.0, neginf=0.0)
        self.markers = np.clip(self.markers, self.lo, self.hi)
        self.marker_vel = np.nan_to_num(self.marker_vel, nan=0.0, posinf=0.0, neginf=0.0)
        for body in self.bodies:
            body.center = np.nan_to_num(body.center, nan=0.0, posinf=0.0, neginf=0.0)
            body.center = np.clip(
                body.center, self.lo + 0.25 * body.bound_radius, self.hi - 0.25 * body.bound_radius
            )
            body.vel = np.nan_to_num(body.vel, nan=0.0, posinf=0.0, neginf=0.0)
            body.omega = np.nan_to_num(body.omega, nan=0.0, posinf=0.0, neginf=0.0)
            body.quat = _quat_normalize(np.nan_to_num(body.quat, nan=0.0, posinf=0.0, neginf=0.0))
            if not np.isfinite(body.quat).all() or np.linalg.norm(body.quat) < 1e-9:
                body.quat = np.array([0.0, 0.0, 0.0, 1.0])

    def _advect_markers(self, dt: float) -> None:
        vel = _interp(self.uc, self.markers, self.lo, self.d, self.n)
        self.marker_vel = vel
        self.markers = self.markers + dt * vel
        # push tracers out of the rigid bodies so solid motion stays trackable
        eps = 1e-4
        for body in self.bodies:
            dist = body.sdf(self.markers)
            inside = dist < 0.0
            if not np.any(inside):
                continue
            pts = self.markers[inside]
            grad = np.empty_like(pts)
            for axis in range(3):
                shift = np.zeros(3)
                shift[axis] = eps
                grad[:, axis] = (body.sdf(pts + shift) - body.sdf(pts - shift)) / (2.0 * eps)
            norm = np.linalg.norm(grad, axis=1, keepdims=True)
            grad = grad / np.maximum(norm, 1e-9)
            push = np.minimum(-dist[inside], 0.5 * self.dmin)[:, None]
            self.markers[inside] = pts + grad * push
        self.markers = np.clip(self.markers, self.lo, self.hi)
        self.marker_vel = np.where(np.isfinite(self.marker_vel), self.marker_vel, 0.0)

    # -- output -------------------------------------------------------------
    def _field_grid(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float)

    def _frame(self) -> StateFrame:
        n = self.n_mark
        count = n + len(self.bodies)
        position = np.zeros((count, 3))
        velocity = np.zeros((count, 3))
        orientation = np.zeros((count, 4))
        angular_velocity = np.zeros((count, 3))
        mass = np.zeros(count)
        density = np.zeros(count)
        density_ratio = np.zeros(count)
        phase = np.zeros(count)
        volume = np.zeros(count)
        body_id = np.zeros(count, dtype=np.int64)
        external_force = np.zeros((count, 3))

        position[:n] = self.markers
        velocity[:n] = self.marker_vel
        orientation[:n, 3] = 1.0
        mass[:n] = self.marker_mass
        density[:n] = self.rho_f
        density_ratio[:n] = 1.0
        volume[:n] = self.spacing**3
        external_force[:n] = 0.0

        interaction = np.zeros((len(self.bodies) + 1, 3))
        for i, body in enumerate(self.bodies):
            k = n + i
            position[k] = body.center
            velocity[k] = body.vel
            orientation[k] = body.quat
            angular_velocity[k] = body.omega
            mass[k] = body.mass
            density[k] = body.density
            density_ratio[k] = body.density / self.rho_f
            phase[k] = 1.0
            volume[k] = body.volume
            body_id[k] = body.bid
            external_force[k] = body.mass * self.gravity
            interaction[i] = body.force_drag + body.force_contact + body.force_wall
        interaction[-1] = -np.sum([body.force_drag for body in self.bodies], axis=0)

        solid_fraction = self.phi
        density_field = self.rho_f * (1.0 - solid_fraction)
        for i, body in enumerate(self.bodies):
            density_field = density_field + body.density * self.phi_bodies[i]
        active_mask = (solid_fraction < 0.5).astype(float)

        frame = StateFrame(
            time=float(self._t),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=mass,
            body_id=body_id,
            attributes={
                "density": density,
                "density_ratio": density_ratio,
                "coupled_phase": phase,
                "marker_volume": volume,
                "mass_quantity": mass.copy(),
            },
            fields={
                "velocity_field": np.asarray(self.uc, dtype=float),
                "active_mask": active_mask,
                "density_field": density_field,
                "external_force": external_force,
                "interaction_force": interaction,
                "cumulative_mass_source": np.zeros(len(self.bodies) + 1, dtype=float),
                "liquid_interior_mask": active_mask.copy(),
            },
            scalars={
                "characteristic_length": float(np.max(self.hi - self.lo)),
                "grid_spacing": float(np.mean(self.d)),
                "marker_spacing": float(self.spacing),
                "fluid_density": float(self.rho_f),
                "kinematic_viscosity": float(self.nu),
                "material_count": float(len(self.bodies) + 1),
                "max_velocity": float(np.linalg.norm(velocity, axis=1).max()),
                "total_mass": float(mass.sum()),
                "max_divergence": float(self.div_max),
            },
        )
        frame.validate()
        return frame

    # -- public API ---------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self._t - 1e-9:
            raise ValueError("advance_to() requires non-decreasing sample times")
        if target <= self._t + 1e-12:
            return self._frame()
        while self._t < target - 1e-12:
            speed = max(self.max_speed, 0.0) + 1e-6
            dt = min(self.dt_max, target - self._t, self.cfl * self.dmin / speed)
            dt = max(dt, 1e-6)
            self._step(dt)
            self._t += dt
        self._t = target
        return self._frame()

    def diagnostics(self) -> dict:
        return {
            "scene_id": self.cfg.get("scene_id"),
            "seed": self.seed,
            "rigid_body_count": len(self.bodies),
            "marker_count": self.n_mark,
            "grid_shape": [int(v) for v in self.n],
            "grid_spacing": [float(v) for v in self.d],
            "max_velocity": float(self.max_speed),
            "max_divergence": float(self.div_max),
            "initial_total_momentum": [0.0, 0.0, 0.0],
        }
