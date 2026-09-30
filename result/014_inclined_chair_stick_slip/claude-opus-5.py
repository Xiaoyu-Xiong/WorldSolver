"""Rigid-body stick-slip solver for the inclined plastic chair scene.

Physical model (implemented from scratch with NumPy only):

* The chair is a single rigid body; mass properties (centre of mass, inertia
  tensor) are integrated directly from ``chair.stl`` assuming uniform density
  scaled to the frozen scene mass.
* The incline is the frozen static box; its top face defines a half-space with
  outward normal ``R_incline @ z_hat``.
* The four chair feet (lowest vertex clusters of the mesh) are unilateral
  compliant contacts: a normal penalty spring-damper (Kelvin-Voigt) plus a
  regularised Coulomb friction element with tangential bristle compliance.
  Friction uses a Stribeck curve
  ``mu(v) = mu_k + (mu_s - mu_k) exp(-(|v|/v_s)) + alpha |v|`` modulated by a
  spatially periodic surface roughness field, so each foot alternates between
  sticking (tangential spring loads elastically inside the friction cone) and
  sliding (force saturates on the cone at the kinetic branch).  That
  alternation is the physical origin of the observed chatter.
* Time stepping is a fixed-step semi-implicit Euler scheme on the 6-DOF
  Newton-Euler equations with quaternion renormalisation each step.

No keyframes and no scripted motion: every exported sample is the result of
stepping the equations of motion from the frozen initial state.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81])


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion."""
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n <= 0.0:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ]
    )


def _quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Integrate an xyzw quaternion with world-frame angular velocity."""
    wx, wy, wz = (float(v) for v in omega)
    x, y, z, w = (float(v) for v in q)
    dq = 0.5 * np.array(
        [
            wx * w + wy * z - wz * y,
            wy * w + wz * x - wx * z,
            wz * w + wx * y - wy * x,
            -(wx * x + wy * y + wz * z),
        ]
    )
    out = np.asarray(q, dtype=float) + dt * dq
    norm = float(np.linalg.norm(out))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return out / norm


def _load_stl(path: Path) -> np.ndarray:
    """Return an [T,3,3] array of triangle vertices from a binary/ascii STL."""
    data = path.read_bytes()
    is_ascii = data[:5].lower() == b"solid" and b"facet" in data[:2048]
    if is_ascii:
        verts = []
        for line in data.decode("ascii", "ignore").splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[0] == "vertex":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        arr = np.asarray(verts, dtype=float)
        return arr.reshape(-1, 3, 3)
    count = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    record = np.dtype([("n", "<3f4"), ("v", "<3,3f4"), ("attr", "<u2")])
    tris = np.frombuffer(data[84 : 84 + count * record.itemsize], dtype=record)
    return tris["v"].reshape(-1, 3, 3).astype(float)


def _mass_properties(tris: np.ndarray, mass: float) -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass and inertia tensor about the COM for a uniform solid."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    volume = float(vol6.sum()) / 6.0
    if abs(volume) < 1e-12:
        centroid = tris.reshape(-1, 3).mean(axis=0)
        inertia = np.eye(3) * (mass * 0.1)
        return centroid, inertia
    vol = vol6 / 6.0
    tet_centroid = (a + b + c) / 4.0
    com = (vol[:, None] * tet_centroid).sum(axis=0) / volume
    density = mass / volume
    second = np.zeros((3, 3))
    zero = np.zeros((tris.shape[0], 1, 3))
    pts = np.concatenate([zero, tris], axis=1)  # [T,4,3] tetrahedra
    for i in range(pts.shape[0]):
        p = pts[i]
        s = p.sum(axis=0)
        second += vol[i] / 20.0 * (p.T @ p + np.outer(s, s))
    second *= density
    inertia_origin = np.trace(second) * np.eye(3) - second
    # Shift to the centre of mass (reverse parallel-axis theorem).
    d = com
    shift = mass * (float(d @ d) * np.eye(3) - np.outer(d, d))
    inertia = inertia_origin - shift
    inertia = 0.5 * (inertia + inertia.T)
    return com, inertia


def _foot_points(tris: np.ndarray) -> np.ndarray:
    """Cluster the lowest mesh vertices into foot contact points (body frame)."""
    verts = np.unique(tris.reshape(-1, 3).round(6), axis=0)
    z_min = float(verts[:, 2].min())
    low = verts[verts[:, 2] < z_min + 0.02]
    if len(low) == 0:
        low = verts[np.argsort(verts[:, 2])[:8]]
    # Greedy spatial clustering in the horizontal plane.
    clusters: list[list[np.ndarray]] = []
    radius = 0.15
    for p in low:
        for group in clusters:
            centre = np.mean(np.asarray(group), axis=0)
            if float(np.linalg.norm(p[:2] - centre[:2])) < radius:
                group.append(p)
                break
        else:
            clusters.append([p])
    points = np.asarray([np.mean(np.asarray(g), axis=0) for g in clusters], dtype=float)
    # Keep the lowest clusters (the feet) and order them deterministically.
    order = np.lexsort((points[:, 1], points[:, 0]))
    return points[order]


class BenchmarkSolver:
    """Semi-implicit rigid-body solver with regularised stick-slip friction."""

    # Contact / material parameters for a plastic chair on a plastic incline.
    # The incline is 18 deg, so tan(theta) = 0.3249: the peak (static) friction
    # coefficient sits just below that value, which keeps the chair creeping
    # downhill while the velocity-weakening branch of the Stribeck curve drives
    # a stick-slip limit cycle against the tangential contact compliance.
    K_N = 2.4e5          # normal penalty stiffness per foot [N/m]
    C_N = 9.0e2          # normal damping per foot [N s/m]
    K_T = 1.2e4          # tangential bristle stiffness per foot [N/m]
    C_T = 3.0e1          # tangential micro-slip damping [N s/m]
    MU_S = 0.380         # static friction coefficient (peak of Stribeck curve)
    MU_K = 0.200         # kinetic friction coefficient (plateau)
    V_STRIBECK = 0.020   # Stribeck decay velocity [m/s]
    MU_VISC = 0.80       # velocity-strengthening slope [s/m]
    ROUGH_AMP = 0.22     # relative roughness modulation of mu
    ROUGH_WAVE = 0.10    # roughness wavelength along the slope [m]
    DT = 2.0e-4          # physics substep [s]

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        geometry = config.get("scene_geometry") or {}
        objects = {obj["id"]: obj for obj in geometry.get("objects", [])}

        frame0 = scene.initial_state()
        self.body_id = np.asarray(
            frame0.body_id if frame0.body_id is not None else [0], dtype=np.int64
        ).reshape(-1)
        self.mass = float(np.asarray(frame0.mass, dtype=float).reshape(-1)[0])

        chair = objects.get("chair", {})
        chair_tf = chair.get("transform", {}) if isinstance(chair, dict) else {}
        q0 = np.asarray(
            frame0.orientation[0]
            if frame0.orientation is not None
            else chair_tf.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]),
            dtype=float,
        )
        self.q = q0 / max(float(np.linalg.norm(q0)), 1e-12)
        self.origin0 = np.asarray(frame0.position[0], dtype=float).copy()
        self.v = np.asarray(frame0.velocity[0], dtype=float).copy()
        self.w = np.asarray(
            frame0.angular_velocity[0]
            if frame0.angular_velocity is not None
            else np.zeros(3),
            dtype=float,
        ).copy()

        # --- mesh-derived rigid-body properties -------------------------------
        asset = "chair.stl"
        geom = chair.get("geometry", {}) if isinstance(chair, dict) else {}
        if isinstance(geom, dict):
            asset = str(geom.get("asset", asset))
        mesh_path = Path.cwd() / asset
        if not mesh_path.is_file():
            mesh_path = Path(__file__).resolve().parent.parent / asset
        tris = _load_stl(mesh_path)
        com, inertia = _mass_properties(tris, self.mass)
        feet = _foot_points(tris) - com
        # The frozen transform places the *mesh origin*; the dynamics integrate
        # the centre of mass, so keep the body-frame offset between them.
        self.com_body = com
        self.x = self.origin0 + _quat_to_matrix(self.q) @ com
        self.inertia_body = inertia
        self.inertia_body_inv = np.linalg.inv(inertia)
        self.r_body = feet                      # [C,3] contact offsets from COM
        self.n_contacts = int(self.r_body.shape[0])

        # --- frozen incline half-space ---------------------------------------
        incline = objects.get("incline", {})
        inc_tf = incline.get("transform", {}) if isinstance(incline, dict) else {}
        inc_geom = incline.get("geometry", {}) if isinstance(incline, dict) else {}
        half = np.asarray(
            inc_geom.get("half_extents", [3.5, 2.0, 0.08]), dtype=float
        ).reshape(3)
        inc_q = np.asarray(
            inc_tf.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]), dtype=float
        )
        inc_t = np.asarray(inc_tf.get("translation", [0.0, 0.0, 0.0]), dtype=float)
        R_inc = _quat_to_matrix(inc_q)
        self.plane_normal = R_inc[:, 2] / float(np.linalg.norm(R_inc[:, 2]))
        self.plane_point = inc_t + float(half[2]) * self.plane_normal
        # Downhill and cross-slope tangents of the inclined surface.
        self.slope_dir = R_inc[:, 0] / float(np.linalg.norm(R_inc[:, 0]))
        self.cross_dir = np.cross(self.plane_normal, self.slope_dir)
        cross_norm = float(np.linalg.norm(self.cross_dir))
        if cross_norm > 1e-12:
            self.cross_dir = self.cross_dir / cross_norm

        # --- integration bookkeeping -----------------------------------------
        self.time = 0.0
        self.dt = float(self.DT)
        # Tangential bristle deflection per contact (world frame, in-plane).
        self.slip_state = np.zeros((self.n_contacts, 3))
        self.contact_active = np.zeros(self.n_contacts, dtype=bool)

        # Frozen normalisation scales exported with every frame.
        self.contact_length_scale = float(
            max(np.mean(np.linalg.norm(self.r_body, axis=1)) * 0.05, 1e-3)
        )
        self.constraint_scale = float(self.MU_S * self.mass * abs(GRAVITY[2]))

        # Cached diagnostics of the last evaluated contact state.
        self._gap = np.zeros(self.n_contacts)
        self._cone_value = np.zeros(self.n_contacts)
        self._cone_target = np.zeros(self.n_contacts)
        self._external_force = self.mass * GRAVITY.copy()
        self._slip_events = 0
        self._sliding_prev = np.zeros(self.n_contacts, dtype=bool)
        self._max_penetration = 0.0
        self._steps = 0
        self._evaluate_contacts()

    # ------------------------------------------------------------------ physics
    def _friction_coefficient(self, speed: np.ndarray, contact_pos: np.ndarray) -> np.ndarray:
        """Stribeck friction curve modulated by a frozen surface roughness field."""
        s = np.abs(speed)
        mu = (
            self.MU_K
            + (self.MU_S - self.MU_K) * np.exp(-s / self.V_STRIBECK)
            + self.MU_VISC * s
        )
        # Deterministic roughness: periodic asperity field along the slope
        # direction, with a cross-slope component so the feet are not in phase.
        along = contact_pos @ self.slope_dir
        across = contact_pos @ self.cross_dir
        rough = np.sin(2.0 * np.pi * along / self.ROUGH_WAVE) * np.cos(
            2.0 * np.pi * across / (3.0 * self.ROUGH_WAVE)
        )
        return np.maximum(mu * (1.0 + self.ROUGH_AMP * rough), 0.02)

    def _contact_wrench(self, x, q, v, w, slip):
        """Contact forces/torques for the current state and bristle deflections.

        Returns ``(force, torque, new_slip, gap, cone_value, cone_limit)``.
        """
        R = _quat_to_matrix(q)
        r_world = self.r_body @ R.T                     # [C,3]
        p = x + r_world                                 # contact sample positions
        n = self.plane_normal
        gap = (p - self.plane_point) @ n                # signed distance
        v_pt = v + np.cross(w, r_world)                 # material point velocity
        v_n = v_pt @ n                                  # normal velocity component
        v_t = v_pt - v_n[:, None] * n                   # tangential velocity

        penetration = np.maximum(-gap, 0.0)
        active = penetration > 0.0
        # Kelvin-Voigt normal law; damping only while approaching or in contact.
        f_n = self.K_N * penetration - self.C_N * np.where(active, v_n, 0.0)
        f_n = np.where(active, np.maximum(f_n, 0.0), 0.0)

        # Tangential bristle: integrate the stick spring, then project onto the
        # friction cone (Coulomb saturation with a Stribeck coefficient).
        slip_new = slip + self.dt * v_t
        slip_new = slip_new - (slip_new @ n)[:, None] * n
        slip_new = np.where(active[:, None], slip_new, 0.0)

        f_t = -(self.K_T * slip_new + self.C_T * v_t)
        f_t = f_t - (f_t @ n)[:, None] * n
        f_t = np.where(active[:, None], f_t, 0.0)

        speed = np.linalg.norm(v_t, axis=1)
        mu = self._friction_coefficient(speed, p)
        limit = mu * f_n
        mag = np.linalg.norm(f_t, axis=1)
        over = mag > limit
        scale = np.where(over & (mag > 1e-12), limit / np.maximum(mag, 1e-12), 1.0)
        f_t = f_t * scale[:, None]
        # When sliding, relax the bristle deflection back onto the cone so the
        # stored elastic energy is released (this is the slip phase).
        if np.any(over):
            slip_new[over] = slip_new[over] * scale[over, None]

        f_total = f_n[:, None] * n + f_t
        force = f_total.sum(axis=0)
        torque = np.cross(r_world, f_total).sum(axis=0)
        return force, torque, slip_new, gap, np.linalg.norm(f_t, axis=1), limit

    def _step(self) -> None:
        dt = self.dt
        R = _quat_to_matrix(self.q)
        inertia_world = R @ self.inertia_body @ R.T
        inertia_world_inv = R @ self.inertia_body_inv @ R.T

        f_ext = self.mass * GRAVITY
        f_c, tau_c, slip_new, gap, cone_val, cone_lim = self._contact_wrench(
            self.x, self.q, self.v, self.w, self.slip_state
        )

        # Newton-Euler with the gyroscopic term, semi-implicit in velocity.
        acc = (f_ext + f_c) / self.mass
        ang_acc = inertia_world_inv @ (tau_c - np.cross(self.w, inertia_world @ self.w))

        self.v = self.v + dt * acc
        self.w = self.w + dt * ang_acc
        self.x = self.x + dt * self.v
        self.q = _quat_integrate(self.q, self.w, dt)
        self.slip_state = slip_new
        self.time += dt
        self._steps += 1

        sliding = cone_val >= cone_lim - 1e-9
        active = gap < 0.0
        self._slip_events += int(np.count_nonzero(sliding & active & ~self._sliding_prev))
        self._sliding_prev = sliding & active
        self._max_penetration = max(self._max_penetration, float(np.max(-gap, initial=0.0)))

    def _evaluate_contacts(self) -> None:
        """Refresh exported contact diagnostics without advancing the state."""
        _, _, _, gap, cone_val, cone_lim = self._contact_wrench(
            self.x, self.q, self.v, self.w, self.slip_state
        )
        self._gap = gap
        self._cone_value = cone_val
        self._cone_target = cone_lim
        self._external_force = self.mass * GRAVITY

    # ------------------------------------------------------------------ contract
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        guard = 0
        max_steps = int(abs(target - self.time) / self.dt) + 8
        while self.time < target - 1e-12 and guard < max_steps:
            remaining = target - self.time
            if remaining < self.dt:
                self.dt = remaining
                self._step()
                self.dt = float(self.DT)
            else:
                self._step()
            guard += 1
        self.time = target
        self._evaluate_contacts()
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        R = _quat_to_matrix(self.q)
        origin = self.x - R @ self.com_body          # mesh-origin (render) position
        velocity = self.v + np.cross(self.w, -(R @ self.com_body))
        gap = np.asarray(self._gap, dtype=float).reshape(-1)
        cone_value = np.asarray(self._cone_value, dtype=float).reshape(-1)
        cone_target = np.asarray(self._cone_target, dtype=float).reshape(-1)

        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(origin.reshape(1, 3)),
            velocity=np.ascontiguousarray(velocity.reshape(1, 3)),
            orientation=np.ascontiguousarray(self.q.reshape(1, 4)),
            angular_velocity=np.ascontiguousarray(self.w.reshape(1, 3)),
            mass=np.array([self.mass], dtype=float),
            body_id=np.asarray(self.body_id, dtype=np.int64).reshape(1),
            attributes={},
            fields={
                "constraint_value": cone_value,
                "constraint_target": cone_target,
                "contact_signed_distance": gap,
                "external_force": np.ascontiguousarray(
                    np.asarray(self._external_force, dtype=float).reshape(1, 3)
                ),
            },
            scalars={
                "constraint_scale": float(self.constraint_scale),
                "contact_length_scale": float(self.contact_length_scale),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "semi-implicit Euler rigid body, penalty contact, "
            "Stribeck bristle friction",
            "substep_dt": float(self.DT),
            "steps": int(self._steps),
            "contacts": int(self.n_contacts),
            "mass": float(self.mass),
            "inertia_diag": [float(v) for v in np.diag(self.inertia_body)],
            "slip_onset_events": int(self._slip_events),
            "max_penetration_m": float(self._max_penetration),
            "friction_static": float(self.MU_S),
            "friction_kinetic": float(self.MU_K),
            "incline_normal": [float(v) for v in self.plane_normal],
            "final_position": [float(v) for v in self.x],
            "final_velocity": [float(v) for v in self.v],
        }

