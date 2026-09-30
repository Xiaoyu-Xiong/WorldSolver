"""Frictional rigid-body solver for the 016_friction_ball_loop scene.

Physical model
--------------
A single rigid sphere (mass m, radius r, inertia I = 2/5 m r^2) moves under
gravity in contact with two frozen static boundaries:

  * ``loop_track`` -- a torus (major radius R, minor radius t, axis n) parsed
    from ``loop.stl``. The ball rides the *inner* rail: its centre travels on a
    circle of radius rho = R - t - r about the torus axis, so the sphere touches
    the tube's inner surface all the way around the vertical loop.
  * ``ground`` -- the top face of the frozen ground box.

Contacts use a compliant (penalty) normal law with linear contact damping,
regularised Coulomb friction at the frozen ``friction_coefficient``, and a
hysteretic rolling-resistance torque. The normal force is clamped to
compression only, so the ball genuinely *detaches* from the rail once the
required centripetal force can no longer be supplied -- that unforced loss of
contact is the phenomenon being reproduced, not a scripted event.

Time stepping is a fixed-step semi-implicit (symplectic) Euler scheme with the
quaternion advanced from the angular velocity and renormalised each step.
Sub-steps are integrated between requested sample times, so every sample is
reached by real physical stepping and the result is deterministic.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

_WORKSPACE = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict:
    """Best-effort YAML load; returns an empty mapping when unavailable."""
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _stl_vertices(path: Path) -> np.ndarray:
    """Read triangle vertices from a binary or ASCII STL file."""
    raw = path.read_bytes()
    if len(raw) >= 84:
        count = struct.unpack("<I", raw[80:84])[0]
        if count > 0 and len(raw) >= 84 + 50 * count:
            data = np.frombuffer(raw[84 : 84 + 50 * count], dtype=np.uint8)
            data = data.reshape(count, 50)[:, 12:48].copy()
            return data.view("<f4").reshape(count * 3, 3).astype(float)
    verts: list[list[float]] = []
    for line in raw.decode("utf-8", "ignore").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "vertex":
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"no vertices parsed from {path}")
    return np.asarray(verts, dtype=float)


def _fit_torus(verts: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Recover (axis, major_radius, minor_radius) of a torus point cloud.

    The axis is the direction of smallest extent. With half-extents h, the tube
    half-thickness is t = h[axis] and the major radius is R = max(h) - t.
    """
    half = 0.5 * (verts.max(axis=0) - verts.min(axis=0))
    axis_index = int(np.argmin(half))
    axis = np.zeros(3)
    axis[axis_index] = 1.0
    minor = float(half[axis_index])
    plane = [h for i, h in enumerate(half) if i != axis_index]
    minor = min(minor, 0.49 * float(max(plane)))
    major = float(max(plane)) - minor
    return axis, major, minor


def _quat_integrate(quat: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Advance an xyzw quaternion by world angular velocity omega over dt."""
    qx, qy, qz, qw = quat
    wx, wy, wz = omega
    dq = 0.5 * np.array(
        [
            wx * qw + wy * qz - wz * qy,
            wy * qw + wz * qx - wx * qz,
            wz * qw + wx * qy - wy * qx,
            -(wx * qx + wy * qy + wz * qz),
        ]
    )
    out = quat + dt * dq
    norm = float(np.linalg.norm(out))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return out / norm


class BenchmarkSolver:
    """Penalty-contact frictional rigid-body integrator for the loop scene."""

    # Numerical parameters (frozen, deterministic).
    SUBSTEP_DT = 2.0e-5          # s, fixed physics step
    CONTACT_STIFFNESS = 2.0e6    # N/m, normal penalty stiffness
    DAMPING_RATIO = 0.40         # dimensionless, contact damping ratio
    SLIP_REGULARISATION = 1.0e-3  # m/s, Coulomb friction smoothing velocity
    SPIN_REGULARISATION = 1.0e-2  # rad/s, rolling-resistance smoothing
    ROLLING_RESISTANCE = 0.05    # dimensionless, hysteretic rolling loss
    NEAR_CONTACT_MARGIN = 0.05   # m, gap below which a contact is reported

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        geometry = _load_yaml(_WORKSPACE / "scene_geometry.yaml")

        self.gravity = np.asarray(
            config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        ).reshape(3)
        self.mu = float(config.get("friction_coefficient", 0.18))
        self.sample_dt = float(config.get("sample_dt", 0.02))
        self.duration = float(config.get("duration", 4.0))

        # --- frozen geometry -------------------------------------------------
        objects = {}
        for obj in geometry.get("objects", []) or []:
            if isinstance(obj, dict) and "id" in obj:
                objects[str(obj["id"])] = obj

        ball = objects.get("ball", {})
        self.radius = float((ball.get("geometry") or {}).get("radius", 0.24))

        ground = objects.get("ground", {})
        g_geom = ground.get("geometry") or {}
        g_half = np.asarray(
            g_geom.get("half_extents", [6.0, 3.0, 0.08]), dtype=float
        ).reshape(3)
        g_trans = np.asarray(
            (ground.get("transform") or {}).get("translation", [0.0, 0.0, -0.08]),
            dtype=float,
        ).reshape(3)
        self.ground_top = float(g_trans[2] + g_half[2])
        self.ground_half_xy = g_half[:2].copy()
        self.ground_center_xy = g_trans[:2].copy()

        track = objects.get("loop_track", {})
        self.loop_center = np.asarray(
            (track.get("transform") or {}).get("translation", [0.0, 0.0, 1.2]),
            dtype=float,
        ).reshape(3)
        asset = (track.get("geometry") or {}).get("asset", "loop.stl")
        try:
            axis, major, minor = _fit_torus(_stl_vertices(_WORKSPACE / str(asset)))
        except Exception:
            axis, major, minor = np.array([0.0, 1.0, 0.0]), 1.0, 0.2
        self.loop_axis = axis / float(np.linalg.norm(axis))
        self.loop_major = float(major)
        self.loop_minor = float(minor)
        # Deterministic in-plane reference direction, used only in the
        # degenerate case where the ball centre sits exactly on the loop axis.
        helper = np.zeros(3)
        helper[int(np.argmin(np.abs(self.loop_axis)))] = 1.0
        ref = np.cross(self.loop_axis, helper)
        self._ref_perp = ref / float(np.linalg.norm(ref))
        self._last_contacts = []
        # Ball-centre rail radius on the inner surface of the tube.
        self.rail_radius = self.loop_major - self.loop_minor - self.radius

        # --- initial rigid-body state ---------------------------------------
        frame0 = scene.initial_state()
        self.mass = float(np.asarray(frame0.mass).reshape(-1)[0]) if frame0.mass is not None else 1.0
        self.body_id = (
            np.asarray(frame0.body_id, dtype=int).reshape(-1).copy()
            if frame0.body_id is not None
            else np.array([0], dtype=int)
        )
        self.inertia = 0.4 * self.mass * self.radius * self.radius

        self.pos = np.asarray(frame0.position, dtype=float).reshape(3).copy()
        self.vel = np.asarray(frame0.velocity, dtype=float).reshape(3).copy()
        if frame0.orientation is not None:
            self.quat = np.asarray(frame0.orientation, dtype=float).reshape(4).copy()
            norm = float(np.linalg.norm(self.quat))
            self.quat = self.quat / norm if norm > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])
        else:
            self.quat = np.array([0.0, 0.0, 0.0, 1.0])

        # Start in the rolling-without-slipping state implied by the scene:
        # omega = (n x v) / r with n the outward contact normal on the ball.
        contacts = self._contacts(self.pos, np.zeros(3), np.zeros(3))
        normal = None
        for gap, nrm, _ in contacts:
            if gap <= self.NEAR_CONTACT_MARGIN:
                normal = nrm
                break
        if normal is not None and float(np.linalg.norm(self.vel)) > 1e-9:
            self.omega = np.cross(normal, self.vel) / self.radius
        elif frame0.angular_velocity is not None:
            self.omega = np.asarray(frame0.angular_velocity, dtype=float).reshape(3).copy()
        else:
            self.omega = np.zeros(3)

        self.time = 0.0
        self._steps = 0
        self._max_penetration = 0.0
        self._min_gap_track = float("inf")
        self._detach_time = float("nan")
        self._was_on_track = True
        self._peak_speed = float(np.linalg.norm(self.vel))

        # Frozen normalisation scales.
        self.contact_length_scale = float(self.radius)
        speed0 = float(np.linalg.norm(self.vel))
        self.constraint_scale = float(speed0 if speed0 > 1e-9 else 1.0)

        stiffness = self.CONTACT_STIFFNESS
        self.contact_damping = (
            2.0 * self.DAMPING_RATIO * np.sqrt(stiffness * self.mass)
        )
        # Inverse effective mass seen by the contact point in the tangent
        # plane for a sphere touched at distance r from its centre:
        # 1/m + r^2/I. Used to bound the frictional impulse per step.
        self._tangent_inv_mass = 1.0 / self.mass + (self.radius ** 2) / self.inertia

    # ------------------------------------------------------------------ contacts
    def _contacts(self, pos, _vel=None, _omega=None):
        """Geometric contacts as (gap, outward unit normal on ball, point)."""
        out = []
        # loop_track: torus with axis self.loop_axis through self.loop_center.
        d = pos - self.loop_center
        a = float(np.dot(d, self.loop_axis))
        radial = d - a * self.loop_axis
        rho_r = float(np.linalg.norm(radial))
        u = radial / rho_r if rho_r > 1e-9 else self._ref_perp
        ring = self.loop_center + self.loop_major * u
        delta = pos - ring
        dist = float(np.linalg.norm(delta))
        normal = delta / dist if dist > 1e-9 else -u
        gap = dist - self.loop_minor - self.radius
        out.append((gap, normal, pos - self.radius * normal))
        # ground: top face of the frozen ground box.
        gnormal = np.array([0.0, 0.0, 1.0])
        gap_g = float(pos[2] - self.ground_top - self.radius)
        if not bool(
            np.all(np.abs(pos[:2] - self.ground_center_xy) <= self.ground_half_xy + self.radius)
        ):
            gap_g = max(gap_g, self.NEAR_CONTACT_MARGIN + 1.0)
        out.append((gap_g, gnormal, pos - self.radius * gnormal))
        return out

    # --------------------------------------------------------------- friction
    def _friction_force(self, fn: float, v_t: np.ndarray, dt: float) -> np.ndarray:
        """Coulomb friction opposing tangential slip.

        The magnitude is the Coulomb bound ``mu * fn``, but never more than the
        force that exactly arrests the slip within one step,
        ``|v_t| / (dt * (1/m + r^2/I))``. Below that threshold the contact
        sticks; above it the contact slides at the Coulomb limit. The clamp makes
        the explicit friction update unconditionally stable and strictly
        dissipative, which matters at hard impacts where ``fn`` is very large.
        """
        slip = float(np.linalg.norm(v_t))
        if slip <= 1e-14 or fn <= 0.0:
            return np.zeros(3)
        stick = slip / max(dt * self._tangent_inv_mass, 1e-12)
        magnitude = min(self.mu * fn, stick)
        return -(magnitude / slip) * v_t

    # --------------------------------------------------------------- integration
    def _step(self, dt: float) -> None:
        """One semi-implicit Euler step with compliant frictional contacts."""
        force = self.mass * self.gravity
        torque = np.zeros(3)
        report = []
        for gap, normal, point in self._contacts(self.pos):
            arm = point - self.pos
            v_point = self.vel + np.cross(self.omega, arm)
            v_n = float(np.dot(v_point, normal))
            v_t = v_point - v_n * normal
            contact_force = np.zeros(3)
            if gap < 0.0:
                fn = self.CONTACT_STIFFNESS * (-gap) - self.contact_damping * v_n
                if fn < 0.0:
                    fn = 0.0  # compression only: the ball may leave the rail
                f_t = self._friction_force(fn, v_t, dt)
                contact_force = fn * normal + f_t
                force = force + contact_force
                torque = torque + np.cross(arm, f_t)
                spin = self.omega - float(np.dot(self.omega, normal)) * normal
                spin_mag = float(np.linalg.norm(spin))
                if spin_mag > 0.0:
                    # Hysteretic rolling resistance, capped so a single step can
                    # never reverse the spin (unconditionally dissipative).
                    scale = self.ROLLING_RESISTANCE * fn * self.radius
                    mag = scale / max(spin_mag, self.SPIN_REGULARISATION)
                    mag = min(mag, self.inertia / dt)
                    torque = torque - mag * spin
                if -gap > self._max_penetration:
                    self._max_penetration = -gap
            report.append((gap, contact_force, v_t))

        self.vel = self.vel + dt * (force / self.mass)
        self.omega = self.omega + dt * (torque / self.inertia)
        self.pos = self.pos + dt * self.vel
        self.quat = _quat_integrate(self.quat, self.omega, dt)
        self.time += dt
        self._steps += 1
        self._last_contacts = report

        track_gap = report[0][0]
        if track_gap < self._min_gap_track:
            self._min_gap_track = track_gap
        speed = float(np.linalg.norm(self.vel))
        if speed > self._peak_speed:
            self._peak_speed = speed
        if self._was_on_track and self.time > 0.2 and track_gap > 0.02:
            self._detach_time = float(self.time)
            self._was_on_track = False

    def _contact_report(self):
        """Gaps, contact forces and tangential slip at the current state."""
        gaps = []
        forces = []
        slips = []
        for gap, normal, point in self._contacts(self.pos):
            arm = point - self.pos
            v_point = self.vel + np.cross(self.omega, arm)
            v_n = float(np.dot(v_point, normal))
            v_t = v_point - v_n * normal
            if gap < 0.0:
                fn = self.CONTACT_STIFFNESS * (-gap) - self.contact_damping * v_n
                fn = max(fn, 0.0)
                f_t = self._friction_force(fn, v_t, self.SUBSTEP_DT)
                forces.append(fn * normal + f_t)
                slips.append(v_t)
            else:
                forces.append(np.zeros(3))
                slips.append(np.zeros(3))
            gaps.append(gap)
        return (
            np.asarray(gaps, dtype=float),
            np.asarray(forces, dtype=float),
            np.asarray(slips, dtype=float),
        )

    # ------------------------------------------------------------------- frames
    def _frame(self):
        from worldsolver.contracts import StateFrame

        gaps, forces, slips = self._contact_report()
        position = self.pos.reshape(1, 3).copy()
        velocity = self.vel.reshape(1, 3).copy()
        orientation = self.quat.reshape(1, 4).copy()
        angular_velocity = self.omega.reshape(1, 3).copy()
        external_force = (self.mass * self.gravity).reshape(1, 3).copy()

        rolling = 0.0
        if gaps[0] < 0.0 and float(np.linalg.norm(slips[0])) < 1e-2:
            rolling = 1.0

        frame = StateFrame(
            time=float(self.time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=np.array([self.mass], dtype=float),
            body_id=self.body_id.copy(),
            attributes={"rolling": np.array([rolling], dtype=float)},
            fields={
                "external_force": external_force,
                "interaction_force": forces.copy(),
                "contact_signed_distance": gaps.copy(),
                "constraint_value": slips.copy(),
                "constraint_target": np.zeros_like(slips),
            },
            scalars={
                "gravity_z": float(self.gravity[2]),
                "contact_length_scale": float(self.contact_length_scale),
                "constraint_scale": float(self.constraint_scale),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        remaining = target - self.time
        if remaining > 0.0:
            steps = max(1, int(np.ceil(remaining / self.SUBSTEP_DT)))
            dt = remaining / steps
            for _ in range(steps):
                self._step(dt)
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        speed = float(np.linalg.norm(self.vel))
        return {
            "method": "penalty-contact frictional rigid body, semi-implicit Euler",
            "substep_dt": float(self.SUBSTEP_DT),
            "steps": int(self._steps),
            "friction_coefficient": float(self.mu),
            "rolling_resistance": float(self.ROLLING_RESISTANCE),
            "loop_major_radius": float(self.loop_major),
            "loop_minor_radius": float(self.loop_minor),
            "rail_radius": float(self.rail_radius),
            "ball_radius": float(self.radius),
            "max_penetration": float(self._max_penetration),
            "min_track_gap": float(self._min_gap_track),
            "track_detach_time": float(self._detach_time),
            "peak_speed": float(self._peak_speed),
            "final_speed": speed,
            "final_time": float(self.time),
            "initial_spin_source": (
                "derived from the rolling-without-slipping condition "
                "omega = (n x v)/r at the initial rail contact"
            ),
        }
