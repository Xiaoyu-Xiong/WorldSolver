"""Physics solver for scene 150_newton_cradle_breakaway.

Rigid-body Newton cradle:

* Each ball is a rigid sphere of mass ``m`` suspended from a fixed pivot by an
  inextensible massless string of length ``L`` (both taken from the frozen
  scene geometry: sphere radii, string capsule tops).
* Dynamics are advanced with a symplectic (semi-implicit) Euler scheme at a
  sub-step far below the contact-oscillator period.
* Unilateral contacts (ball-ball and ball-floor) use a compliant penalty law
  ``F = max(0, k*delta + c*delta_dot)`` with Coulomb friction; the damping
  coefficient is derived from a target coefficient of restitution so the
  impact response is a genuine physical model, not scripted motion.
* The string constraint is enforced by a SHAKE-style projection of the
  position back onto the sphere of radius ``L`` around the pivot and removal
  of the radial velocity component, i.e. the balls swing as pendulums.
* One end ball is displaced by a finite release angle and released from rest
  at t = 0, so the impact impulse propagates through the touching row and the
  opposite end ball breaks away - the phenomenon of Figure 3 of
  "Reflections on Simultaneous Impact".
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


def _damping_ratio(restitution: float) -> float:
    """Damping ratio giving coefficient of restitution ``e`` for a damped contact."""
    e = min(max(float(restitution), 1e-3), 0.9999)
    ln_e = math.log(e)
    return -ln_e / math.sqrt(math.pi * math.pi + ln_e * ln_e)


class BenchmarkSolver:
    """Deterministic rigid-pendulum contact solver for the Newton cradle scene."""

    # Numerical parameters (fixed, deterministic).
    SUBSTEPS_PER_SAMPLE = 400      # dt = 5e-5 s for sample_dt = 0.02 s
    CONTACT_STIFFNESS = 1.0e6      # N/m (compliant contact; peak penetration ~1 mm)
    BALL_RESTITUTION = 0.97        # steel-on-steel chain response
    FRICTION_REG_VEL = 5.0e-3      # m/s regularization for Coulomb friction
    NEAR_CONTACT_FACTOR = 5.0      # near-contact margin in contact length scales
    RELEASE_ANGLE_DEG = 30.0       # initial displacement of the end ball

    def __init__(self) -> None:
        self._ready = False
        self._seed = 0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config if hasattr(scene, "config") else {}
        self._seed = int(seed)

        bodies = cfg["bodies"]
        self.n = len(bodies)
        self.mass = np.asarray([float(b["mass"]) for b in bodies], dtype=float)
        self.body_id = np.arange(self.n, dtype=np.int32)
        self.gravity = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        self.sample_dt = float(cfg.get("sample_dt", 0.02))

        contact = cfg.get("contact", {}) or {}
        self.mu = float(contact.get("friction", 0.0))
        self.e_floor = float(contact.get("restitution", 0.0))
        self.length_scale = float(contact.get("length_scale", 0.02))

        self.dt = self.sample_dt / float(self.SUBSTEPS_PER_SAMPLE)

        # ---- frozen geometry -------------------------------------------------
        objects = []
        geo = cfg.get("scene_geometry")
        if isinstance(geo, dict):
            objects = geo.get("objects", []) or []

        radii_by_body: dict[int, float] = {}
        strings: list[tuple[float, float]] = []   # (x, pivot_z)
        floor_top = 0.0
        for obj in objects:
            geom = obj.get("geometry") or {}
            gtype = geom.get("type")
            transform = obj.get("transform") or {}
            tr = np.asarray(transform.get("translation", (0.0, 0.0, 0.0)), dtype=float)
            if gtype == "sphere" and obj.get("motion") == "dynamic":
                binding = obj.get("state_binding") or {}
                if binding.get("selector") == "body_id":
                    radii_by_body[int(binding["value"])] = float(geom.get("radius", 0.0))
            elif gtype == "capsule":
                height = float(geom.get("height", 0.0))
                strings.append((float(tr[0]), float(tr[2]) + 0.5 * height))
            elif gtype == "box" and obj.get("id") == "floor":
                half = np.asarray(geom.get("half_extents", (0.0, 0.0, 0.0)), dtype=float)
                floor_top = float(tr[2] + half[2])

        self.pos_rest = np.asarray([b["position"] for b in bodies], dtype=float)
        self.floor_top = floor_top

        # Ball radii (from binding when available, else from row spacing).
        spacing = float(np.min(np.diff(np.sort(self.pos_rest[:, 0])))) if self.n > 1 else 0.46
        fallback_radius = 0.5 * spacing
        self.radius = np.asarray(
            [radii_by_body.get(i, fallback_radius) for i in range(self.n)], dtype=float
        )

        # Pivot of each pendulum: top of the matching string capsule.
        self.pivot = np.zeros_like(self.pos_rest)
        self.length = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            best = None
            for sx, sz in strings:
                dist = abs(sx - float(self.pos_rest[i, 0]))
                if best is None or dist < best[0]:
                    best = (dist, sx, sz)
            if best is not None and best[0] < 0.25 * spacing + 1e-6:
                pivot = np.array([best[1], self.pos_rest[i, 1], best[2]])
            else:
                pivot = np.array([
                    self.pos_rest[i, 0], self.pos_rest[i, 1], self.pos_rest[i, 2] + 1.0
                ])
            self.pivot[i] = pivot
            self.length[i] = float(np.linalg.norm(self.pos_rest[i] - pivot))

        # ---- contact pairs (all ball-ball combinations) ----------------------
        pi, pj = np.triu_indices(self.n, k=1)
        self.pair_i, self.pair_j = pi, pj
        self.pair_rsum = self.radius[pi] + self.radius[pj]
        m_red = self.mass[pi] * self.mass[pj] / (self.mass[pi] + self.mass[pj])
        self.pair_k = np.full(len(pi), self.CONTACT_STIFFNESS)
        zeta = _damping_ratio(self.BALL_RESTITUTION)
        self.pair_c = 2.0 * zeta * np.sqrt(self.pair_k * m_red)
        # floor contact parameters (per body)
        self.floor_k = self.CONTACT_STIFFNESS
        self.floor_c = 2.0 * _damping_ratio(self.e_floor) * math.sqrt(
            self.floor_k * 0.5 * float(np.mean(self.mass))
        )

        self.margin = self.NEAR_CONTACT_FACTOR * self.length_scale

        # ---- initial state: row at rest, one end ball displaced --------------
        self._init_pos = self.pos_rest.copy()
        self._init_vel = np.zeros_like(self.pos_rest)
        if self.n > 1:
            axis = self.pos_rest[1] - self.pos_rest[0]
            norm = float(np.linalg.norm(axis))
            axis = axis / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
            theta = math.radians(self.RELEASE_ANGLE_DEG)
            r_hat = (self.pos_rest[0] - self.pivot[0]) / self.length[0]
            self._init_pos[0] = self.pivot[0] + self.length[0] * (
                math.cos(theta) * r_hat - math.sin(theta) * axis
            )

        self._reset()
        self._ready = True

    def _reset(self) -> None:
        self.pos = self._init_pos.copy()
        self.vel = self._init_vel.copy()
        self.step = 0
        self.export_step = 0
        self.pair_impulse = np.zeros((len(self.pair_i), 3))
        self.floor_impulse = np.zeros((self.n, 3))

    # ------------------------------------------------------------- integration
    def _step(self, dt: float) -> None:
        pos, vel = self.pos, self.vel
        force = np.zeros_like(pos)

        # ---- ball-ball compliant contact ------------------------------------
        delta = pos[self.pair_j] - pos[self.pair_i]
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        safe = np.maximum(dist, 1e-12)
        gap = dist - self.pair_rsum
        active = np.nonzero(gap < 0.0)[0]
        if active.size:
            n_vec = delta[active] / safe[active, None]
            pen = -gap[active]
            vrel = vel[self.pair_j[active]] - vel[self.pair_i[active]]
            vn = np.einsum("ij,ij->i", vrel, n_vec)
            fmag = np.maximum(
                self.pair_k[active] * pen - self.pair_c[active] * vn, 0.0
            )
            # regularized Coulomb friction on the tangential relative velocity
            vt = vrel - vn[:, None] * n_vec
            vtmag = np.sqrt(np.einsum("ij,ij->i", vt, vt))
            ftmag = self.mu * fmag * vtmag / (vtmag + self.FRICTION_REG_VEL)
            fvec = fmag[:, None] * n_vec - (ftmag / (vtmag + 1e-12))[:, None] * vt
            np.add.at(force, self.pair_j[active], fvec)
            np.add.at(force, self.pair_i[active], -fvec)
            self.pair_impulse[active] += fvec * dt

        # ---- ball-floor compliant contact ------------------------------------
        fgap = pos[:, 2] - (self.floor_top + self.radius)
        fact = np.nonzero(fgap < 0.0)[0]
        if fact.size:
            pen = -fgap[fact]
            vn = vel[fact, 2]
            fmag = np.maximum(self.floor_k * pen - self.floor_c * vn, 0.0)
            vt = vel[fact, :2].copy()
            vtmag = np.sqrt(np.einsum("ij,ij->i", vt, vt))
            ftmag = self.mu * fmag * vtmag / (vtmag + self.FRICTION_REG_VEL)
            fvec = np.zeros((fact.size, 3))
            fvec[:, 2] = fmag
            fvec[:, :2] = -(ftmag / (vtmag + 1e-12))[:, None] * vt
            force[fact] += fvec
            self.floor_impulse[fact] += fvec * dt

        # ---- symplectic Euler + pendulum constraint projection ---------------
        acc = self.gravity[None, :] + force / self.mass[:, None]
        vel += acc * dt
        pos += vel * dt

        rad = pos - self.pivot
        rad_norm = np.sqrt(np.einsum("ij,ij->i", rad, rad))
        rad_hat = rad / np.maximum(rad_norm, 1e-12)[:, None]
        pos[:] = self.pivot + self.length[:, None] * rad_hat
        vr = np.einsum("ij,ij->i", vel, rad_hat)
        vel -= vr[:, None] * rad_hat

    # ----------------------------------------------------------------- export
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("initialize() must be called before advance_to()")
        time = float(time)
        target = int(round(time / self.dt))
        if target < self.step:
            self._reset()
        while self.step < target:
            self._step(self.dt)
            self.step += 1

        steps = target - self.export_step
        elapsed = steps * self.dt if steps > 0 else 0.0

        # instantaneous geometric gaps for the near-contact set
        delta = self.pos[self.pair_j] - self.pos[self.pair_i]
        dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
        gap = dist - self.pair_rsum
        near = np.nonzero(gap < self.margin)[0]

        # average internal interaction forces over the last sample interval
        if elapsed > 0.0 and steps > 0:
            pair_force = self.pair_impulse[near] / elapsed
        else:
            pair_force = np.zeros((near.size, 3))
        floor_force = (
            self.floor_impulse / elapsed if elapsed > 0.0 else np.zeros((self.n, 3))
        )

        external = self.mass[:, None] * self.gravity[None, :] + floor_force

        contact_distance = gap[near] if near.size else np.zeros(0)

        self.pair_impulse[:] = 0.0
        self.floor_impulse[:] = 0.0
        self.export_step = target

        n = self.n
        frame = StateFrame(
            time=time,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1)),
            angular_velocity=np.zeros((n, 3)),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={"release_time": np.zeros(n)},
            fields={
                "external_force": external,
                "interaction_force": pair_force,
                "contact_signed_distance": contact_distance,
            },
            scalars={"contact_length_scale": self.length_scale},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        out: dict[str, object] = {"success": True}
        if self._ready:
            delta = self.pos[self.pair_j] - self.pos[self.pair_i]
            dist = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            gap = dist - self.pair_rsum
            out["min_ball_gap"] = float(np.min(gap)) if gap.size else 0.0
            out["max_speed"] = float(np.max(np.linalg.norm(self.vel, axis=1)))
            out["steps_taken"] = int(self.step)
            out["release_angle_deg"] = float(self.RELEASE_ANGLE_DEG)
        return out
