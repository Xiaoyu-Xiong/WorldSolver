"""Physics solver for 012_billiard_impulse_chain.

Impulse-based rigid sphere dynamics implemented from scratch with NumPy:
 - semi-implicit Euler substeps,
 - sequential-impulse ball-ball collisions (near-elastic, Coulomb friction),
 - sphere-ground contact with normal impulse, Coulomb sliding friction and
   rolling resistance,
 - direct positional projection to keep the scene overlap-free.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

GROUND_HALF = np.array([4.5, 2.0, 0.05])
GROUND_Z = 0.0          # top surface of the ground box
NEAR_CONTACT = 0.05     # proximity threshold for reported contact samples [m]


class _Body:
    __slots__ = ("m", "I", "R", "x", "v", "w", "q")

    def __init__(self, m: float, R: float, x: np.ndarray, v: np.ndarray):
        self.m = m
        self.R = R
        self.I = 0.4 * m * R * R
        self.x = np.array(x, dtype=float)
        self.v = np.array(v, dtype=float)
        self.w = np.zeros(3)
        # orientation quaternion (w, x, y, z)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])


def _quat_integrate(q: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
    """Integrate a unit quaternion with angular velocity w over dt."""
    wn = float(np.linalg.norm(w))
    if wn * dt == 0.0:
        return q
    axis = w / wn
    half = 0.5 * wn * dt
    dq = np.concatenate(([np.cos(half)], np.sin(half) * axis))
    w0, x0, y0, z0 = q
    w1, x1, y1, z1 = dq
    qn = np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])
    return qn / float(np.linalg.norm(qn))


class BenchmarkSolver:
    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config if hasattr(scene, "config") else {}
        self.g = np.array(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.mu_roll = float(cfg.get("rolling_friction", 0.015))
        rng = np.random.default_rng(seed)
        self.rng = rng

        init = scene.initial_state()
        pos = np.asarray(init.position, dtype=float).reshape(-1, 3)
        vel = np.asarray(init.velocity, dtype=float).reshape(-1, 3)
        n = len(pos)

        masses = np.asarray(init.mass, dtype=float).reshape(-1) if init.mass is not None \
            else np.full(n, 0.17)
        radii = np.full(n, 0.2)
        # frozen geometry: all dynamic objects are 0.2 m spheres
        for i in range(n):
            radii[i] = 0.2

        self.bodies = [
            _Body(float(masses[i]), float(radii[i]), pos[i], vel[i]) for i in range(n)
        ]
        self.body_ids = (np.asarray(init.body_id, dtype=int).reshape(-1)
                         if init.body_id is not None else np.arange(n))
        self.n = n

        # material parameters (billiard-like)
        self.e_ball = 0.98          # ball-ball restitution
        self.mu_ball = 0.06         # ball-ball friction
        self.e_ground = 0.05        # ground restitution (felt)
        self.mu_ground = 0.45       # sliding friction on felt

        self.contact_length_scale = float(np.mean(radii))

        # substep
        self.dt_sub = 5e-5
        self.t = 0.0

        # impulse bookkeeping since last exported sample
        self._impulses: dict[tuple, np.ndarray] = {}
        self._prev_export_t = 0.0
        self._ready = True

    # ------------------------------------------------------------- stepping
    def _record_impulse(self, key: tuple, imp: np.ndarray) -> None:
        acc = self._impulses.get(key)
        if acc is None:
            self._impulses[key] = np.array(imp, dtype=float)
        else:
            self._impulses[key] = acc + imp

    def _ground_contact(self, b: _Body) -> bool:
        return b.x[2] - b.R <= GROUND_Z + 1e-9

    def _step(self, dt: float) -> None:
        g = self.g
        mu_g = self.mu_ground
        mu_r = self.mu_roll

        # --- external forces: gravity (+ rolling resistance as tangential force
        # applied through the ground contact for balls resting on the ground)
        for b in self.bodies:
            b.v += g * dt
            if self._ground_contact(b) and b.v[2] <= 1e-6:
                vt = b.v[:2]
                sp = float(np.linalg.norm(vt))
                if sp > 1e-9:
                    dec = mu_r * abs(g[2]) * dt
                    factor = max(0.0, 1.0 - dec / sp)
                    b.v[0] *= factor
                    b.v[1] *= factor

        # --- ground normal impulses (support / bounce)
        for i, b in enumerate(self.bodies):
            if b.x[2] - b.R < GROUND_Z:
                rcp = np.array([0.0, 0.0, -b.R])          # contact point offset
                # contact point velocity
                u = b.v + np.cross(b.w, rcp)
                if u[2] < 0.0:
                    jn = -(1.0 + self.e_ground) * b.m * u[2]
                    jv = np.array([0.0, 0.0, jn])
                    b.v += jv / b.m
                    b.w += np.cross(rcp, jv) / b.I
                    self._record_impulse((int(self.body_ids[i]), -1), jv)
                    # Coulomb friction on the tangential contact velocity
                    rcp = np.array([0.0, 0.0, -b.R])
                    u = b.v + np.cross(b.w, rcp)
                    ut = u[:2]
                    utn = float(np.linalg.norm(ut))
                    if utn > 1e-12:
                        m_eff = b.m / (1.0 + b.m * b.R * b.R / b.I)
                        jt = np.array([-ut[0], -ut[1], 0.0]) * m_eff
                        jmax = mu_g * jn
                        if float(np.linalg.norm(jt)) > jmax and jmax > 0.0:
                            jt = jt / float(np.linalg.norm(jt)) * jmax
                        b.v += jt / b.m
                        b.w += np.cross(rcp, jt) / b.I
                        self._record_impulse((int(self.body_ids[i]), -1), jt)

        # --- ball-ball impulses (iterated sequential impulses)
        n = self.n
        for _ in range(10):
            applied = False
            for a in range(n):
                for c in range(a + 1, n):
                    b1, b2 = self.bodies[a], self.bodies[c]
                    d = b2.x - b1.x
                    dist = float(np.linalg.norm(d))
                    rsum = b1.R + b2.R
                    if dist >= rsum or dist <= 1e-12:
                        continue
                    nrm = d / dist
                    # check actual separation gap (allow tiny slop)
                    pen = rsum - dist
                    vrel = b2.v - b1.v
                    vn = float(np.dot(vrel, nrm))
                    if vn >= 0.0 and pen <= 1e-9:
                        continue
                    if vn < 0.0:
                        k = 1.0 / b1.m + 1.0 / b2.m
                        jn = -(1.0 + self.e_ball) * vn / k
                        jv = jn * nrm
                        b1.v -= jv / b1.m
                        b2.v += jv / b2.m
                        # ball-ball friction (spin transfer), small
                        vt = vrel - vn * nrm
                        vtn = float(np.linalg.norm(vt))
                        if vtn > 1e-12 and self.mu_ball > 0.0:
                            jt = -vt / vtn * min(self.mu_ball * jn, vtn / k)
                            b1.v -= jt / b1.m
                            b2.v += jt / b2.m
                        key = (int(self.body_ids[a]), int(self.body_ids[c]))
                        self._record_impulse(key, jv)
                        applied = True
            if not applied:
                break

        # --- positional projection (keep the scene overlap-free)
        for b in self.bodies:
            if b.x[2] - b.R < GROUND_Z:
                b.x[2] = GROUND_Z + b.R
        for a in range(n):
            for c in range(a + 1, n):
                b1, b2 = self.bodies[a], self.bodies[c]
                d = b2.x - b1.x
                dist = float(np.linalg.norm(d))
                rsum = b1.R + b2.R
                if dist < rsum and dist > 1e-12:
                    nrm = d / dist
                    pen = rsum - dist
                    b1.x -= 0.5 * pen * nrm
                    b2.x += 0.5 * pen * nrm
                elif dist <= 1e-12:
                    b2.x[0] += 1e-6

        # --- integrate positions and orientations
        for b in self.bodies:
            b.x += b.v * dt
            b.q = _quat_integrate(b.q, b.w, dt)

        self.t += dt

    # ------------------------------------------------------------- sampling
    def advance_to(self, time: float) -> StateFrame:
        assert self._ready
        target = float(time)
        if target < self.t:
            target = self.t
        guard = 0
        while target - self.t > 1e-12:
            dt = min(self.dt_sub, target - self.t)
            self._step(dt)
            guard += 1
            if guard > 200000:
                break

        n = self.n
        pos = np.array([b.x for b in self.bodies], dtype=float)
        vel = np.array([b.v for b in self.bodies], dtype=float)
        omega = np.array([b.w for b in self.bodies], dtype=float)
        quat = np.array([b.q for b in self.bodies], dtype=float)
        mass = np.array([b.m for b in self.bodies], dtype=float)

        # external force per body: gravity only (contact forces are internal)
        ext = np.tile(self.g, (n, 1)) * mass[:, None]

        # interaction forces: impulses since last sample / sample interval
        span = max(target - self._prev_export_t, 1e-9)
        keys = sorted(self._impulses.keys())
        inter = np.zeros((len(keys), 3), dtype=float)
        for k, key in enumerate(keys):
            inter[k] = self._impulses[key] / span
        self._impulses.clear()
        self._prev_export_t = target

        # signed contact distances for active / near contacts
        gaps: list[float] = []
        for a in range(n):
            for c in range(a + 1, n):
                b1, b2 = self.bodies[a], self.bodies[c]
                d = b2.x - b1.x
                dist = float(np.linalg.norm(d))
                gap = dist - (b1.R + b2.R)
                if gap < NEAR_CONTACT:
                    gaps.append(gap)
        for b in self.bodies:
            gap = b.x[2] - b.R - GROUND_Z
            if gap < NEAR_CONTACT:
                gaps.append(gap)
        csd = np.array(gaps, dtype=float)

        frame = StateFrame(
            time=target,
            position=pos,
            velocity=vel,
            orientation=quat,
            angular_velocity=omega,
            mass=mass,
            body_id=self.body_ids.copy(),
            scalars={"contact_length_scale": float(self.contact_length_scale)},
            fields={
                "external_force": ext,
                "interaction_force": inter,
                "contact_signed_distance": csd,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "impulse-based rigid sphere dynamics (NumPy)",
            "substep": self.dt_sub,
            "bodies": self.n,
            "time": self.t,
        }
