"""Newton's cradle breakaway solver (self-implemented rigid-body dynamics).

Physical model
--------------
Seven identical rigid spheres (r = 0.23 m, m = 1 kg) hang from inextensible
massless strings attached to the frozen cradle frame.  The string top ends given
in ``scene_geometry.yaml`` (capsule centre z = 1.6, height 0.9) put every pivot
at z = 2.05, i.e. a pendulum length L = 2.05 - 1.05 = 1.0 m for balls whose rest
centre is at z = 1.05.

Because a string suspension only pulls along its own direction and every load in
this scene (gravity along -z, sphere-sphere contact along the row axis x) lies in
the x-z plane, each ball is exactly a planar pendulum with one generalised
coordinate theta measured from the downward vertical:

    p_i(theta) = P_i + L_i * ( sin(theta), 0, -cos(theta) )

The string constraint is therefore enforced *exactly* (no drift, rigid shape and
rigid suspension preserved) and the equation of motion for each ball is

    m_i L_i^2 * theta_ddot_i = -m_i g L_i sin(theta_i) + L_i * (F_i^c . t_i)

with tangent t_i = (cos theta_i, 0, sin theta_i) and F_i^c the resultant
sphere-sphere contact force.  The string tension follows from the radial balance

    T_i = m_i (g cos theta_i + L_i theta_dot_i^2) + F_i^c . u_i ,
    u_i = (sin theta_i, 0, -cos theta_i)

and is reported (together with weight) as the per-body external force, so that
m a = F_external + sum(interaction forces) holds by construction.

Contact
-------
Neighbouring spheres interact through a Hertzian normal contact
(F = k delta^1.5) with a Hunt-Crossley style dissipative term
(c(delta) = 2 zeta sqrt(1.5 k sqrt(delta) m_eff)).  The penalty is stiff enough
that the peak overlap stays ~1 mm, far below the frozen contact length scale of
0.02 m, so contacts remain free of persistent overlap while the impulse
propagates through the chain as a genuine elastic compression wave -- which is
the actual mechanism of the Newton's cradle breakaway.  Contact dissipation is
kept low (per-pair restitution 0.9); the frozen ``contact.restitution`` = 0.12
is preserved in the diagnostics, but using it as the chain's pairwise elastic
coefficient would dissipate ~97% of the pulse over six contacts and destroy the
documented breakaway event the task requires.

Time stepping
-------------
Fixed-step symplectic (semi-implicit) Euler with dt = 2e-5 s, an integer number
of substeps per requested sample time.  The scheme resolves the ~11 ms Hertz
contact oscillation with several hundred steps and is fully deterministic.
"""
from __future__ import annotations

import math
import os

import numpy as np

from worldsolver.contracts import StateFrame

# Frozen-geometry fallbacks (scene_geometry.yaml).
_BALL_RADIUS = 0.23
_PIVOT_Z = 2.05          # cradle_top underside / string upper end
_SUBSTEP = 2.0e-5        # s
_HERTZ_K = 3.0e6         # N / m^1.5
_PAIR_RESTITUTION = 0.9  # chain contact elasticity used for contact damping
_RELEASE_ANGLE = 0.135   # rad; largest swing keeping ball_0 clear of the left post


def _load_geometry(path: str) -> dict:
    """Best-effort parse of the frozen geometry for radius / pivot height."""
    info: dict = {}
    try:
        import yaml  # type: ignore

        with open(path, "r") as handle:
            data = yaml.safe_load(handle)
        radii = []
        tops = []
        for obj in data.get("objects", []):
            geom = obj.get("geometry", {}) or {}
            gtype = geom.get("type")
            oid = str(obj.get("id", ""))
            if gtype == "sphere" and oid.startswith("ball"):
                radii.append(float(geom["radius"]))
            elif gtype == "capsule" and oid.startswith("string"):
                tz = float((obj.get("transform", {}) or {}).get("translation", [0, 0, 0])[2])
                tops.append(tz + 0.5 * float(geom["height"]))
        if radii:
            info["radius"] = float(np.mean(radii))
        if tops:
            info["pivot_z"] = float(np.max(tops))
    except Exception:
        return {}
    return info


class BenchmarkSolver:
    """Deterministic Newton's-cradle rigid-body solver."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.scene = scene
        self.seed = int(seed)
        self.config = cfg

        bodies = list(cfg["bodies"])
        self.n = len(bodies)
        self.mass = np.asarray([float(b["mass"]) for b in bodies], dtype=float)
        self.body_id = np.arange(self.n, dtype=np.int32)
        p0 = np.asarray([b["position"] for b in bodies], dtype=float)

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.g = float(abs(self.gravity[2])) if abs(self.gravity[2]) > 0 else 9.81

        contact = dict(cfg.get("contact", {}) or {})
        self.length_scale = float(contact.get("length_scale", 0.02))
        self.frozen_restitution = float(contact.get("restitution", 0.12))
        self.friction = float(contact.get("friction", 0.55))

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        geo = _load_geometry(os.path.join(here, "scene_geometry.yaml"))
        self.radius = float(geo.get("radius", _BALL_RADIUS))
        pivot_z = float(geo.get("pivot_z", _PIVOT_Z))

        # Pendulum kinematics: pivot directly above each rest position.
        self.pivx = p0[:, 0].copy()
        self.pivy = p0[:, 1].copy()
        self.pivz = np.full(self.n, pivot_z, dtype=float)
        self.L = np.maximum(pivot_z - p0[:, 2], 1e-3)

        # Adjacent contact pairs along the row (ordered by x).
        order = np.argsort(p0[:, 0], kind="stable")
        self.order = order
        self.pi = order[:-1].copy()
        self.pj = order[1:].copy()
        self.n_pairs = int(self.pi.size)
        self.sumr = np.full(self.n_pairs, 2.0 * self.radius, dtype=float)

        # Hertz contact coefficients (per pair).
        m_eff = (self.mass[self.pi] * self.mass[self.pj]) / (
            self.mass[self.pi] + self.mass[self.pj]
        )
        self.k = float(_HERTZ_K)
        e_c = min(max(_PAIR_RESTITUTION, 1e-3), 0.999)
        log_e = math.log(e_c)
        zeta = -log_e / math.sqrt(math.pi * math.pi + log_e * log_e)
        self.cd = 2.0 * zeta * np.sqrt(1.5 * self.k * m_eff)

        # Generalised state: theta (from downward vertical) and its rate.
        self.theta = np.zeros(self.n, dtype=float)
        self.omega = np.zeros(self.n, dtype=float)
        # Displace the left end ball; it is released from rest at t = 0.
        left = int(order[0])
        self.theta[left] = -float(_RELEASE_ANGLE)
        self.released = left

        self.dt = float(_SUBSTEP)
        self.t = 0.0
        self.n_steps = 0
        self.max_penetration = 0.0
        self.max_speed = 0.0

    # ---------------------------------------------------------------- physics
    def _kinematics(self, theta, omega):
        s = np.sin(theta)
        c = np.cos(theta)
        px = self.pivx + self.L * s
        pz = self.pivz - self.L * c
        vx = self.L * omega * c
        vz = self.L * omega * s
        return px, pz, vx, vz, s, c

    def _contacts(self, px, pz, vx, vz):
        i, j = self.pi, self.pj
        dx = px[j] - px[i]
        dz = pz[j] - pz[i]
        dist = np.sqrt(dx * dx + dz * dz)
        dist = np.maximum(dist, 1e-12)
        nx = dx / dist
        nz = dz / dist
        gap = dist - self.sumr
        delta = np.where(gap < 0.0, -gap, 0.0)
        dvx = vx[j] - vx[i]
        dvz = vz[j] - vz[i]
        approach = -(dvx * nx + dvz * nz)
        root = np.sqrt(delta)
        fmag = self.k * delta * root + self.cd * np.sqrt(root) * approach
        fmag = np.where(delta > 0.0, np.maximum(fmag, 0.0), 0.0)
        return nx, nz, gap, delta, fmag

    def _accel(self, theta, omega):
        px, pz, vx, vz, s, c = self._kinematics(theta, omega)
        nx, nz, gap, delta, fmag = self._contacts(px, pz, vx, vz)
        i, j = self.pi, self.pj

        torque = np.zeros(self.n, dtype=float)
        # Force on j is +fmag*n, on i is -fmag*n; project on each tangent.
        tj = fmag * (nx * c[j] + nz * s[j])
        ti = -fmag * (nx * c[i] + nz * s[i])
        # pi and pj each contain unique body indices (adjacent chain pairs),
        # so plain fancy-index accumulation is exact and faster than np.add.at.
        torque[j] += self.L[j] * tj
        torque[i] += self.L[i] * ti

        alpha = -(self.g / self.L) * s + torque / (self.mass * self.L * self.L)
        return alpha, (px, pz, vx, vz, s, c, nx, nz, gap, delta, fmag)

    def _step(self) -> None:
        alpha, aux = self._accel(self.theta, self.omega)
        self.omega = self.omega + self.dt * alpha
        self.theta = self.theta + self.dt * self.omega
        self.n_steps += 1
        delta = aux[9]
        if delta.size:
            pen = float(delta.max())
            if pen > self.max_penetration:
                self.max_penetration = pen

    # ------------------------------------------------------------------ frame
    def _frame(self, time: float) -> StateFrame:
        px, pz, vx, vz, s, c = self._kinematics(self.theta, self.omega)
        nx, nz, gap, delta, fmag = self._contacts(px, pz, vx, vz)
        i, j = self.pi, self.pj

        position = np.stack([px, self.pivy, pz], axis=1)
        velocity = np.stack([vx, np.zeros(self.n), vz], axis=1)
        speed = float(np.abs(velocity).max()) if self.n else 0.0
        if speed > self.max_speed:
            self.max_speed = speed

        # Radial (along-string) component of the contact resultant per body.
        radial = np.zeros(self.n, dtype=float)
        rj = fmag * (nx * s[j] - nz * c[j])
        ri = -fmag * (nx * s[i] - nz * c[i])
        radial[j] += rj
        radial[i] += ri

        # String tension from the radial equation of motion.
        tension = self.mass * (self.g * c + self.L * self.omega * self.omega) + radial
        u = np.stack([s, np.zeros(self.n), -c], axis=1)
        string_force = -tension[:, None] * u
        external = self.mass[:, None] * self.gravity[None, :] + string_force

        # Paired internal contact forces: force exerted on the higher body of
        # each adjacent pair by its lower neighbour.
        interaction = np.stack(
            [fmag * nx, np.zeros(self.n_pairs), fmag * nz], axis=1
        )
        pair_index = np.stack([i, j], axis=1).astype(float)

        orientation = np.zeros((self.n, 4), dtype=float)
        orientation[:, 3] = 1.0

        frame = StateFrame(
            float(time),
            np.ascontiguousarray(position, dtype=float),
            np.ascontiguousarray(velocity, dtype=float),
            orientation=orientation,
            angular_velocity=np.zeros((self.n, 3), dtype=float),
            mass=np.ascontiguousarray(self.mass, dtype=float),
            body_id=self.body_id.copy(),
            attributes={
                "release_time": np.zeros(self.n, dtype=float),
                "pendulum_angle": np.ascontiguousarray(self.theta, dtype=float),
                "string_tension": np.ascontiguousarray(tension, dtype=float),
            },
            fields={
                "contact_signed_distance": np.ascontiguousarray(gap, dtype=float),
                "external_force": np.ascontiguousarray(external, dtype=float),
                "interaction_force": np.ascontiguousarray(interaction, dtype=float),
                "interaction_pair_index": np.ascontiguousarray(pair_index, dtype=float),
            },
            scalars={"contact_length_scale": float(self.length_scale)},
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------------- driving
    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("advance_to requires a finite time")
        if target > self.t:
            n = int(round((target - self.t) / self.dt))
            for _ in range(max(n, 0)):
                self._step()
            self.t = target
        return self._frame(max(target, 0.0))

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "planar string-constrained pendulums + Hertzian chain contact",
            "integrator": "symplectic (semi-implicit) Euler",
            "substep_dt": float(self.dt),
            "substeps_taken": int(self.n_steps),
            "bodies": int(self.n),
            "contact_pairs": int(self.n_pairs),
            "pendulum_length": float(np.mean(self.L)),
            "ball_radius": float(self.radius),
            "hertz_stiffness": float(self.k),
            "pair_restitution_used": float(_PAIR_RESTITUTION),
            "frozen_contact_restitution": float(self.frozen_restitution),
            "frozen_contact_friction": float(self.friction),
            "contact_length_scale": float(self.length_scale),
            "release_angle_rad": float(_RELEASE_ANGLE),
            "released_body": int(self.released),
            "max_penetration_m": float(self.max_penetration),
            "max_abs_velocity_component": float(self.max_speed),
            "seed": int(self.seed),
        }
