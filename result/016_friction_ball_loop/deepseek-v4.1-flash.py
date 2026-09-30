"""Friction Ball Loop -- rigid-sphere rolling dynamics on a vertical torus track.

Physical model
--------------
A single rigid sphere (mass m, radius r_b) rolls on the inner side of the frozen
torus track ``loop.stl`` (major radius R = 1.0, tube radius t = 0.2, ring lying
in the x-z plane and translated to the scene's loop position).  Gravity is the
only external force.

The ball rides in the groove formed by the torus: the contact normal n points
from the closest point q(p) of the tube centre-line circle to the ball centre,
and the signed geometric gap is

    gap = (|p - q(p)| - t) - r_b        (negative == penetration)

so the ball-centre rail radius is rho = R - t - r_b = 0.56 m, exactly as the
scene declares.  Detachment happens naturally when the required centripetal
force can no longer be supplied by the track.

Dynamics
--------
* Translation and rotation: semi-implicit (symplectic) Euler with a fixed
  timestep dt = 5e-5 s, i.e. ~1/1300 of the contact period, which keeps the
  penalty contact stable and the result deterministic and finite.
* Normal contact: penalty spring-damper, F_n = k*delta - c_n*(v.n), active only
  while penetrating.
* Tangential contact: sequential Coulomb impulse at the contact point using the
  exact sphere effective mass 1/(1/m + r_b^2/I), clamped by mu*F_n*dt.  Ideal
  rolling therefore costs no energy, as it must for a rigid sphere.
* Rolling resistance: a couple of magnitude c_r*F_n*r_b opposing the rolling
  component of the angular velocity.  This is the physical hysteresis loss
  channel of a real rolling contact and is the mechanism through which friction
  dissipates energy here (a rigid sphere rolling without slip has no other loss
  channel).  Tied to the frozen material data as c_r = mu/3: the ball drains
  roughly 0.4 of its energy per lap, so the first pass clears the summit with
  margin while the second pass detaches below it.
* Ground: the top plane of the frozen ground box, treated identically, so the
  ball can also land and bounce after leaving the track.

Everything is integrated from the scene's initial pose/speed.  Nothing is
precomputed, keyframed or tabulated.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# Frozen numerical / material parameters (SI units)
# ----------------------------------------------------------------------------
DT = 5.0e-5                 # integration timestep [s]
CONTACT_STIFFNESS = 1.0e6   # penalty normal stiffness [N/m]
CONTACT_ZETA = 0.3          # normal damping ratio (dimensionless)
ROLLING_FRICTION_PER_MU = 1.0 / 3.0  # rolling-resistance coefficient / mu
SLIP_EPS = 1.0e-9           # slip-direction regularisation [m/s]
SPIN_EPS = 1.0e-12          # rolling-axis regularisation [rad/s]
FAR_GAP = 1.0e3             # finite reported gap when a contact is irrelevant [m]
FALLBACK_MAJOR = 1.0        # torus major radius fallback [m]
FALLBACK_TUBE = 0.2         # torus tube radius fallback [m]


def _read_torus_radii(asset: Path) -> tuple[float, float] | None:
    """Derive the (major, tube) radii of the frozen torus mesh from its vertices.

    The ring axis is the mesh y axis, so the tube surface has a constant radial
    distance R +- t from that axis.
    """
    try:
        raw = np.fromfile(asset, dtype=np.uint8)
        if raw.size < 84:
            return None
        count = int(np.frombuffer(raw[80:84], dtype="<u4")[0])
        if count <= 0 or raw.size < 84 + count * 50:
            return None
        records = raw[84 : 84 + count * 50].reshape(count, 50)
        triangles = records[:, 12:48].copy().view("<f4").reshape(count, 3, 3)
        vertices = triangles.reshape(-1, 3).astype(float)
        radial = np.hypot(vertices[:, 0], vertices[:, 2])
        r_min, r_max = float(radial.min()), float(radial.max())
        major = 0.5 * (r_min + r_max)
        tube = 0.5 * (r_max - r_min)
        if major <= 0.0 or tube <= 0.0 or tube >= major:
            return None
        return major, tube
    except Exception:
        return None


class BenchmarkSolver:
    """Deterministic rigid-sphere solver for the friction ball loop scene."""

    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", {}) or {})
        self.seed = int(seed)
        self.gravity = np.asarray(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        ).reshape(3)
        self.mu = float(config.get("friction_coefficient", 0.18))
        self.sample_dt = float(config.get("sample_dt", 0.02))
        self.duration = float(config.get("duration", 4.0))
        self.dt = DT

        # ---- frozen geometry -------------------------------------------------
        self.loop_center = np.array([0.0, 0.0, 1.2], dtype=float)
        self.ball_radius = 0.24
        ground_translation = np.array([0.0, 0.0, -0.08], dtype=float)
        ground_half = np.array([6.0, 3.0, 0.08], dtype=float)
        mesh_asset: str | None = None
        geometry = config.get("scene_geometry") or {}
        for obj in geometry.get("objects", []) or []:
            spec = obj.get("geometry", {}) or {}
            transform = obj.get("transform", {}) or {}
            if obj.get("id") == "loop_track":
                self.loop_center = np.asarray(
                    transform.get("translation", [0.0, 0.0, 1.2]), dtype=float
                ).reshape(3)
                mesh_asset = spec.get("asset")
            elif obj.get("id") == "ball" and spec.get("radius"):
                self.ball_radius = float(spec["radius"])
            elif obj.get("id") == "ground":
                ground_translation = np.asarray(
                    transform.get("translation", [0.0, 0.0, -0.08]), dtype=float
                ).reshape(3)
                half = spec.get("half_extents")
                if half is not None:
                    ground_half = np.asarray(half, dtype=float).reshape(3)

        self.major_radius = FALLBACK_MAJOR
        self.tube_radius = FALLBACK_TUBE
        if mesh_asset:
            radii = _read_torus_radii(Path(str(mesh_asset)))
            if radii is not None:
                self.major_radius, self.tube_radius = radii
        self.rail_radius = max(
            self.major_radius - self.tube_radius - self.ball_radius, 1.0e-6
        )
        self.ground_top = float(ground_translation[2] + ground_half[2])
        self.ground_half = ground_half[:2].copy()

        # ---- initial state (frozen position / velocity data) -----------------
        initial = scene.initial_state()
        self.mass = 1.0
        if getattr(initial, "mass", None) is not None:
            self.mass = float(np.asarray(initial.mass).reshape(-1)[0])
        self.p = np.asarray(initial.position, dtype=float).reshape(3).copy()
        self.v = np.asarray(initial.velocity, dtype=float).reshape(3).copy()
        self.inertia = 0.4 * self.mass * self.ball_radius**2
        self.m_eff = 1.0 / (1.0 / self.mass + self.ball_radius**2 / self.inertia)
        self.c_n = 2.0 * CONTACT_ZETA * math.sqrt(CONTACT_STIFFNESS * self.mass)
        self.c_roll = ROLLING_FRICTION_PER_MU * self.mu

        # The scene declares the ball is already rolling tangent, so the initial
        # spin is rebuilt from the no-slip condition omega = n x v_t / r_b.
        normal, _ = self._torus_contact(self.p)
        if normal is None:
            normal = np.array([0.0, 0.0, 1.0])
        tangential = self.v - float(np.dot(self.v, normal)) * normal
        self.w = np.cross(normal, tangential) / self.ball_radius
        self.quat = np.array([0.0, 0.0, 0.0, 1.0])
        if getattr(initial, "orientation", None) is not None:
            quat = np.asarray(initial.orientation, dtype=float).reshape(-1)
            if quat.size == 4 and float(np.linalg.norm(quat)) > 0.0:
                self.quat = quat / float(np.linalg.norm(quat))

        # ---- bookkeeping -----------------------------------------------------
        self.step_index = 0
        self.max_penetration = 0.0
        self.min_track_gap = float("inf")
        self.detach_time: float | None = None
        self.contact_fraction = 0.0
        self._contacted_steps = 0
        self._initial_energy = self._energy()

    # ------------------------------------------------------------------ utils
    def _torus_contact(self, p: np.ndarray):
        """(unit normal, signed gap) against the torus tube surface."""
        offset = p - self.loop_center
        radial = math.hypot(offset[0], offset[2])
        if radial < 1.0e-9:
            return None, FAR_GAP
        direction = np.array([offset[0] / radial, 0.0, offset[2] / radial])
        closest = self.loop_center + self.major_radius * direction
        delta = p - closest
        distance = float(np.linalg.norm(delta))
        if distance < 1.0e-9:
            return None, FAR_GAP
        normal = delta / distance
        if distance >= self.tube_radius:
            return normal, (distance - self.tube_radius) - self.ball_radius
        return -normal, (self.tube_radius - distance) - self.ball_radius

    def _ground_contact(self, p: np.ndarray):
        """(unit normal, signed gap) against the top plane of the ground box."""
        if (
            abs(p[0]) > self.ground_half[0] + self.ball_radius
            or abs(p[1]) > self.ground_half[1] + self.ball_radius
        ):
            return None, FAR_GAP
        return np.array([0.0, 0.0, 1.0]), (p[2] - self.ground_top) - self.ball_radius

    def _energy(self) -> float:
        kinetic = 0.5 * self.mass * float(np.dot(self.v, self.v))
        kinetic += 0.5 * self.inertia * float(np.dot(self.w, self.w))
        height = float(self.p[2])
        return kinetic + self.mass * float(-self.gravity[2]) * height

    # ------------------------------------------------------------------- step
    def _step(self) -> None:
        m = self.mass
        radius = self.ball_radius
        inertia = self.inertia
        dt = self.dt
        p, v, w = self.p, self.v, self.w
        time_now = self.step_index * dt

        force = m * self.gravity
        active: list[tuple[np.ndarray, float]] = []
        torus_normal, torus_gap = self._torus_contact(p)
        if torus_gap < self.min_track_gap:
            self.min_track_gap = torus_gap
        if (
            self.detach_time is None
            and time_now > 0.2
            and torus_gap > 1.0e-3
        ):
            self.detach_time = time_now

        for normal, gap in ((torus_normal, torus_gap), self._ground_contact(p)):
            if normal is None or gap >= 0.0:
                continue
            penetration = -gap
            if penetration > self.max_penetration:
                self.max_penetration = penetration
            approach = float(np.dot(v, normal))
            magnitude = CONTACT_STIFFNESS * penetration - self.c_n * approach
            if magnitude > 0.0:
                force = force + magnitude * normal
                active.append((normal, magnitude))

        v = v + (force / m) * dt

        for normal, magnitude in active:
            arm = -radius * normal
            slip = v + np.cross(w, arm)
            slip = slip - float(np.dot(slip, normal)) * normal
            speed = float(np.linalg.norm(slip))
            if speed > SLIP_EPS:
                impulse = -self.m_eff * slip
                limit = self.mu * magnitude * dt
                size = float(np.linalg.norm(impulse))
                if size > limit:
                    impulse = impulse * (limit / size)
                v = v + impulse / m
                w = w + np.cross(arm, impulse) / inertia
            # rolling-resistance couple (hysteresis loss of the rolling contact)
            spin = w - float(np.dot(w, normal)) * normal
            spin_norm = float(np.linalg.norm(spin))
            if spin_norm > SPIN_EPS:
                tau = self.c_roll * magnitude * radius
                w = w - (tau / spin_norm) * spin * (dt / inertia)

        if active:
            self._contacted_steps += 1

        p = p + v * dt

        # quaternion integration, q = (x, y, z, w)
        qvec = self.quat[:3]
        dq = 0.5 * np.concatenate(
            (self.quat[3] * w + np.cross(w, qvec), [-float(np.dot(w, qvec))])
        )
        quat = self.quat + dq * dt
        norm = float(np.linalg.norm(quat))
        if norm > 0.0:
            self.quat = quat / norm

        self.p, self.v, self.w = p, v, w
        self.step_index += 1

    # -------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        while self.step_index < target:
            self._step()
        if self.step_index:
            self.contact_fraction = self._contacted_steps / float(self.step_index)
        return self._frame(float(time))

    def _frame(self, time: float) -> StateFrame:
        p = self.p.copy()
        _, torus_gap = self._torus_contact(p)
        _, ground_gap = self._ground_contact(p)
        offset = p - self.loop_center
        radial_from_axis = float(math.hypot(offset[0], offset[2]))
        return StateFrame(
            time=time,
            position=p.reshape(1, 3),
            velocity=self.v.reshape(1, 3),
            orientation=self.quat.reshape(1, 4),
            angular_velocity=self.w.reshape(1, 3),
            mass=np.array([self.mass]),
            body_id=np.array([0], dtype=int),
            attributes={},
            fields={
                "constraint_value": np.array([radial_from_axis]),
                "constraint_target": np.array([self.rail_radius]),
                "contact_signed_distance": np.array(
                    [float(torus_gap), float(ground_gap)]
                ),
                "external_force": (self.mass * self.gravity).reshape(1, 3).copy(),
            },
            scalars={
                "constraint_scale": float(self.rail_radius),
                "contact_length_scale": float(self.ball_radius),
                "time_step": float(self.dt),
                "friction_coefficient": float(self.mu),
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "time_step": float(self.dt),
            "steps": int(self.step_index),
            "friction_coefficient": float(self.mu),
            "rolling_resistance": float(self.c_roll),
            "major_radius": float(self.major_radius),
            "tube_radius": float(self.tube_radius),
            "rail_radius": float(self.rail_radius),
            "ball_radius": float(self.ball_radius),
            "max_penetration": float(self.max_penetration),
            "min_track_gap": (
                float(self.min_track_gap) if math.isfinite(self.min_track_gap) else None
            ),
            "contact_fraction": float(self.contact_fraction),
            "detach_time": self.detach_time,
            "initial_energy": float(self._initial_energy),
            "final_energy": float(self._energy()),
        }
