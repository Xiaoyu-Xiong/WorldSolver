"""Rigid-body stick-slip solver for the inclined plastic chair scene (014).

The chair is a single rigid body (mass/inertia taken from the frozen STL and the
scene's declared initial state).  It rests on the frozen inclined box.  Contact
is resolved at the four chair feet, each modelled as a sphere of the radius
measured on the frozen mesh:

  * normal direction : unilateral penalty spring-damper,
  * tangential       : Coulomb friction with an elastic anchor (contact / leg
                       compliance).  The anchor force grows while the foot is
                       stuck; when it reaches the static limit mu_s * f_n the
                       foot breaks loose, slides with the kinetic coefficient
                       and re-anchors once the slip velocity vanishes.

Because mu_s > tan(theta) > mu_k, the tangential drive (gravity) alternately
loads and releases the contact: the body speeds up elastically, slips, is
decelerated by kinetic friction and catches again - a genuine stick-slip
relaxation cycle.  The friction force acts below the centre of mass, so the
load is transferred between the leading and trailing feet, which adds the
visible rocking chatter.  Everything is stepped explicitly with a small,
fixed time step; no trajectory is precomputed or scripted.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# --------------------------------------------------------------------------
# physical parameters
# --------------------------------------------------------------------------
GRAVITY = 9.81
FALLBACK_MASS = 10.8

# contact model
K_NORMAL = 1.5e5       # N/m   unilateral normal stiffness per foot
ZETA_NORMAL = 0.35     #       normal damping ratio (per foot effective mass)
K_TANGENT = 450.0      # N/m   tangential anchor stiffness per foot (contact and
                       #       flexible plastic leg compliance in shear)
C_TANGENT = 4.0        # N*s/m tangential anchor damping per foot

MU_STATIC = 0.45       # friction coefficient at zero slip velocity
MU_KINETIC = 0.42      # friction coefficient at high slip velocity
V_STRIBECK = 0.04      # m/s   velocity-weakening scale

SLIP_EPS = 2.0e-4      # m/s   below this a sliding foot re-anchors (sticks)
CONTACT_SLOP = 1.0e-6  # m     nominal touch tolerance

REFERENCE_VELOCITY = 0.1   # m/s  frozen constraint normalization scale
NOMINAL_FOOT_RADIUS = 0.055
SIM_DT = 2.0e-4           # s    fixed physical time step

# fallback foot layout (measured on the frozen chair.stl, body frame, metres)
FALLBACK_FEET_LOCAL = np.array(
    [
        [0.5835, -0.42, -1.07185],
        [0.5835, 0.42, -1.07185],
        [-0.3465, -0.42, -1.07185],
        [-0.3465, 0.42, -1.07185],
    ],
    dtype=float,
)
# fallback inertia about the mesh origin (uniform density, mass 10.8 kg)
FALLBACK_INERTIA = np.array(
    [
        [6.2466, 0.0, 2.1816],
        [0.0, 7.2125, 0.0],
        [2.1816, 0.0, 3.2667],
    ],
    dtype=float,
)


# --------------------------------------------------------------------------
# small vector / quaternion helpers
# --------------------------------------------------------------------------
def _quat_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


def _quat_step(q: np.ndarray, omega: np.ndarray, h: float) -> np.ndarray:
    wx, wy, wz = float(omega[0]), float(omega[1]), float(omega[2])
    qx, qy, qz, qw = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # dq = 0.5 * (omega, 0) * q
    dqx = 0.5 * (wx * qw + wy * qz - wz * qy)
    dqy = 0.5 * (-wx * qz + wy * qw + wz * qx)
    dqz = 0.5 * (wx * qy - wy * qx + wz * qw)
    dqw = 0.5 * (-wx * qx - wy * qy - wz * qz)
    return _quat_normalize(np.array([qx + h * dqx, qy + h * dqy, qz + h * dqz, qw + h * dqw]))


# --------------------------------------------------------------------------
# frozen asset inspection
# --------------------------------------------------------------------------
def _read_binary_stl(path: Path):
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 84:
        return None
    count = struct.unpack("<I", data[80:84])[0]
    if count <= 0 or 84 + count * 50 > len(data):
        return None
    record = np.frombuffer(
        data[84:84 + count * 50],
        dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]),
        count=count,
    )
    tris = record["v"].reshape(count, 3, 3).astype(float)
    if not np.isfinite(tris).all():
        return None
    return tris


def _mesh_volume_inertia(tris: np.ndarray, mass: float):
    """Uniform-density volume and inertia tensor about the mesh origin."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    vols = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
    total = float(vols.sum())
    if not math.isfinite(total) or abs(total) < 1e-9:
        return None, None
    tet = np.stack([np.zeros_like(v0), v0, v1, v2], axis=1)
    s = tet.sum(axis=1)
    cov = np.einsum("nik,nil->nkl", tet, tet)
    second = np.sum(
        vols[:, None, None] * (cov + np.einsum("ni,nj->nij", s, s)) / 20.0, axis=0
    )
    rho = mass / total
    inertia = rho * (np.trace(second) * np.eye(3) - second)
    if not np.isfinite(inertia).all():
        return None, None
    return abs(total), inertia


def _feet_from_mesh(tris: np.ndarray):
    """Locate the four spherical foot pads as the bottom clusters of the mesh."""
    pts = tris.reshape(-1, 3)
    if len(pts) < 16:
        return None, None
    z_min = float(pts[:, 2].min())
    cap = pts[pts[:, 2] < z_min + 0.125]
    if len(cap) < 12:
        return None, None
    x_mid = 0.5 * (float(cap[:, 0].min()) + float(cap[:, 0].max()))
    centres = []
    radii = []
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            mask = ((cap[:, 0] > x_mid) if sx > 0 else (cap[:, 0] <= x_mid)) & (
                (cap[:, 1] * sy) > 0.0
            )
            sel = cap[mask]
            if len(sel) < 4:
                return None, None
            cx = float(sel[:, 0].mean())
            cy = float(sel[:, 1].mean())
            r = float(np.sqrt((sel[:, 0] - cx) ** 2 + (sel[:, 1] - cy) ** 2).max())
            if not (1e-3 < r < 0.25):
                return None, None
            centres.append([cx, cy, z_min + r])
            radii.append(r)
    return np.array(centres, dtype=float), float(np.mean(radii))


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
class BenchmarkSolver:
    """Explicit rigid-body contact solver with stick-slip friction."""

    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        self.config = config
        self.dt = SIM_DT

        initial = scene.initial_state()
        self.mass = float(np.asarray(initial.mass).reshape(-1)[0]) if initial.mass is not None else FALLBACK_MASS
        self.pos = np.asarray(initial.position, dtype=float).reshape(3).copy()
        quat = np.asarray(initial.orientation, dtype=float).reshape(4)
        self.quat = _quat_normalize(quat.copy())
        self.vel = np.asarray(initial.velocity, dtype=float).reshape(3).copy()
        if initial.angular_velocity is not None:
            self.ang = np.asarray(initial.angular_velocity, dtype=float).reshape(3).copy()
        else:
            self.ang = np.zeros(3)
        if not np.isfinite(self.pos).all() or not np.isfinite(self.vel).all():
            raise ValueError("initial state contains non-finite values")

        objects = self._objects(config)
        self.normal, self.plane_point = self._incline_plane(objects)
        feet, radius, inertia = self._chair_geometry(objects)
        self.feet_local = feet
        self.foot_radius = float(radius)
        self.inertia_body = inertia
        self.inertia_inv = np.linalg.inv(inertia)

        # normal damping from a target damping ratio on the per-foot effective mass
        m_eff = max(self.mass / 4.0, 1e-6)
        self.c_normal = 2.0 * ZETA_NORMAL * math.sqrt(K_NORMAL * m_eff)

        # contact bookkeeping
        R = _quat_matrix(self.quat)
        pts = self.pos + self.feet_local @ R.T
        self.anchor = self._project(pts)
        self.slipping = np.zeros(len(self.feet_local), dtype=bool)

        self.time = 0.0
        self.sample_dt = float(config.get("sample_dt", 0.01))
        self.contact_forces = np.zeros((len(self.feet_local), 3))
        self.slip_events = 0
        self._was_slipping = np.zeros(len(self.feet_local), dtype=bool)

    # -- geometry ---------------------------------------------------------
    @staticmethod
    def _objects(config: dict):
        geometry = config.get("scene_geometry") or {}
        items = geometry.get("objects") if isinstance(geometry, dict) else None
        return items if isinstance(items, list) else []

    @staticmethod
    def _find_object(objects, name):
        for item in objects:
            if isinstance(item, dict) and item.get("id") == name:
                return item
        return {}

    def _incline_plane(self, objects):
        incline = self._find_object(objects, "incline")
        transform = incline.get("transform") or {}
        translation = np.asarray(transform.get("translation", [0.0, 0.0, 0.35]), dtype=float).reshape(3)
        rotation = np.asarray(transform.get("rotation_xyzw", [0.0, 0.1564, 0.0, 0.9877]), dtype=float).reshape(4)
        geometry = incline.get("geometry") or {}
        half = np.asarray(geometry.get("half_extents", [3.5, 2.0, 0.08]), dtype=float).reshape(3)
        R = _quat_matrix(_quat_normalize(rotation))
        normal = R @ np.array([0.0, 0.0, 1.0])
        n = float(np.linalg.norm(normal))
        normal = normal / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])
        point = translation + R @ np.array([0.0, 0.0, float(half[2])])
        return normal, point

    def _chair_geometry(self, objects):
        chair = self._find_object(objects, "chair")
        geometry = chair.get("geometry") or {}
        asset = geometry.get("asset")
        tris = None
        for candidate in self._asset_candidates(asset):
            tris = _read_binary_stl(candidate)
            if tris is not None:
                break

        feet, radius, inertia = None, None, None
        if tris is not None:
            volume, inertia_mesh = _mesh_volume_inertia(tris, self.mass)
            if inertia_mesh is not None:
                inertia = inertia_mesh
            feet, radius = _feet_from_mesh(tris)

        if feet is None or radius is None:
            feet = FALLBACK_FEET_LOCAL.copy()
            radius = NOMINAL_FOOT_RADIUS
        if inertia is None:
            inertia = FALLBACK_INERTIA.copy()

        # symmetric positive definite inertia required by the integrator
        inertia = 0.5 * (inertia + inertia.T)
        eig = np.linalg.eigvalsh(inertia)
        if not np.isfinite(eig).all() or float(eig.min()) <= 0.0:
            inertia = FALLBACK_INERTIA.copy()
        return feet, float(radius), inertia

    @staticmethod
    def _asset_candidates(asset):
        names = [asset] if asset else []
        names.append("chair.stl")
        roots = [Path(__file__).resolve().parent.parent, Path.cwd()]
        seen = []
        for name in names:
            if not name:
                continue
            path = Path(str(name))
            if path.is_absolute():
                seen.append(path)
                continue
            for root in roots:
                seen.append(root / str(name))
        out = []
        for path in seen:
            if path not in out:
                out.append(path)
        return out

    # -- contact helpers --------------------------------------------------
    def _project(self, points: np.ndarray) -> np.ndarray:
        """Project world points onto the (fixed) incline plane."""
        signed = (points - self.plane_point) @ self.normal
        return points - signed[:, None] * self.normal[None, :]

    def _contact_state(self):
        """Kinematics of the four feet for the current rigid-body state."""
        R = _quat_matrix(self.quat)
        r_world = self.feet_local @ R.T
        points = self.pos + r_world
        gap = (points - self.plane_point) @ self.normal - self.foot_radius
        vel_points = self.vel + np.cross(self.ang, r_world)
        v_n = vel_points @ self.normal
        v_t = vel_points - v_n[:, None] * self.normal[None, :]
        contact_points = points - self.foot_radius * self.normal[None, :]
        return r_world, points, contact_points, gap, v_n, v_t

    # -- dynamics ---------------------------------------------------------
    def _step(self, h: float) -> None:
        m = self.mass
        n = self.normal
        R = _quat_matrix(self.quat)
        r_world, points, contact_points, gap, v_n, v_t = self._contact_state()
        on_plane = self._project(contact_points)

        penetration = -gap
        f_n = K_NORMAL * penetration - self.c_normal * v_n
        np.maximum(f_n, 0.0, out=f_n)
        speed = np.linalg.norm(v_t, axis=1)
        # kinetic (velocity weakened) coefficient used once a foot is sliding;
        # a foot whose anchor still holds sees the full static limit.
        mu_slip = MU_KINETIC + (MU_STATIC - MU_KINETIC) * np.exp(
            -((speed / V_STRIBECK) ** 2)
        )

        total_force = np.array([0.0, 0.0, -m * GRAVITY])
        total_torque = np.zeros(3)
        forces = np.zeros((len(self.feet_local), 3))

        for i in range(len(self.feet_local)):
            if f_n[i] <= 0.0 or gap[i] > CONTACT_SLOP:
                # separated: drop the anchor so the next touch starts clean
                f_n[i] = 0.0
                self.anchor[i] = on_plane[i]
                self.slipping[i] = False
                continue

            static_limit = MU_STATIC * f_n[i]
            if self.slipping[i]:
                limit = mu_slip[i] * f_n[i]
                if speed[i] < SLIP_EPS or limit <= 0.0:
                    self.anchor[i] = on_plane[i]
                    self.slipping[i] = False
                    f_t = -C_TANGENT * v_t[i]
                else:
                    u = v_t[i] / speed[i]
                    f_t = -limit * u
                    self.anchor[i] = on_plane[i] - (limit / K_TANGENT) * u
            else:
                limit = static_limit
                stretch = on_plane[i] - self.anchor[i]
                f_el = -K_TANGENT * stretch - C_TANGENT * v_t[i]
                mag = float(np.linalg.norm(f_el))
                if mag > limit:
                    if speed[i] > 1e-9:
                        u = v_t[i] / speed[i]
                    elif mag > 0.0:
                        u = f_el / mag
                    else:
                        u = np.zeros(3)
                    f_t = -limit * u
                    self.anchor[i] = on_plane[i] - (limit / K_TANGENT) * u
                    self.slipping[i] = True
                else:
                    f_t = f_el

            force = f_n[i] * n + f_t
            forces[i] = force
            total_force += force
            total_torque += np.cross(contact_points[i] - self.pos, force)
            if self.slipping[i] and not self._was_slipping[i]:
                self.slip_events += 1

        self._was_slipping = self.slipping.copy()
        self.contact_forces = forces

        # semi-implicit integration of the rigid body
        self.vel = self.vel + (total_force / m) * h
        inertia_world = R @ self.inertia_body @ R.T
        inertia_inv_world = R @ self.inertia_inv @ R.T
        gyro = np.cross(self.ang, inertia_world @ self.ang)
        alpha = inertia_inv_world @ (total_torque - gyro)
        self.ang = self.ang + alpha * h
        self.pos = self.pos + self.vel * h
        self.quat = _quat_step(self.quat, self.ang, h)

        for name in ("pos", "vel", "ang", "quat"):
            value = getattr(self, name)
            if not np.isfinite(value).all():
                raise FloatingPointError(f"non-finite rigid body state in {name}")

    # -- public contract --------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.time < target - 1e-9:
            h = min(self.dt, target - self.time)
            if h <= 0.0:
                break
            self._step(h)
            self.time += h
            guard += 1
            if guard > 5_000_000:
                raise RuntimeError("time stepping did not reach the requested sample time")
        if target > self.time:
            self.time = target
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        _, _, _, gap, _, v_t = self._contact_state()
        signed_distance = gap.astype(float)
        constraint_value = v_t.astype(float)
        constraint_target = np.zeros_like(constraint_value)
        external_force = np.array([[0.0, 0.0, -self.mass * GRAVITY]])
        normal_force = np.maximum(-K_NORMAL * signed_distance, 0.0)
        return StateFrame(
            float(time),
            self.pos.reshape(1, 3).copy(),
            self.vel.reshape(1, 3).copy(),
            orientation=self.quat.reshape(1, 4).copy(),
            angular_velocity=self.ang.reshape(1, 3).copy(),
            mass=np.array([self.mass]),
            body_id=np.array([0]),
            fields={
                "contact_signed_distance": signed_distance,
                "constraint_value": constraint_value,
                "constraint_target": constraint_target,
                "external_force": external_force,
                "contact_force": self.contact_forces.copy(),
                "friction_cone_limit": (MU_STATIC * normal_force).astype(float),
            },
            scalars={
                "constraint_scale": float(REFERENCE_VELOCITY),
                "contact_length_scale": float(self.foot_radius),
            },
        )

    def diagnostics(self) -> dict[str, object]:
        _, _, _, gap, _, v_t = self._contact_state()
        return {
            "solver": "explicit_rigid_body_stick_slip",
            "time": float(self.time),
            "mass": float(self.mass),
            "foot_radius": float(self.foot_radius),
            "sim_dt": float(self.dt),
            "slip_events": int(self.slip_events),
            "final_position": [float(v) for v in self.pos],
            "final_velocity": [float(v) for v in self.vel],
            "final_angular_velocity": [float(v) for v in self.ang],
            "min_contact_gap": float(np.min(gap)),
            "max_slip_speed": float(np.max(np.linalg.norm(v_t, axis=1))),
        }
