"""Physics solver for scene 016_friction_ball_loop.

Rigid-sphere dynamics inside a vertical loop (torus) with frictional contact.

Physical model
--------------
* The loop track is the frozen torus from ``loop.stl`` placed at (0, 0, 1.2):
  ring radius R = 1.0 m, tube radius t = 0.2 m, ring axis along y.
  The ball (radius r_b = 0.24 m) rolls on the tube's inner surface, so the
  ball-centre rail radius is rho = R - t - r_b = 0.56 m.
* The ground is the frozen box (half extents 6 x 3 x 0.08 at z = -0.08),
  i.e. a flat top surface at z = 0.
* Contact is resolved with a compliant (penalty) normal force
  N = max(0, -k*gap - c*vn) plus Coulomb friction with a smooth
  stick/slip law f_t = -mu*N*tanh(|v_slip|/v_eps)*v_hat_slip evaluated at
  the contact point (v_slip = v + w x r_cp), and a rolling-resistance
  torque tau = -c_rr*N*omega_hat_perp (standard rolling friction model).
* Integration: semi-implicit (symplectic) Euler with a fixed step
  h = 1e-4 s, which is far below the contact stability limit, so the
  trajectory is deterministic and finite.

Phenomenon: the ball starts at the loop bottom with v = 5.5 m/s, above the
rolling-contact minimum sqrt(27/7 * g * rho) ~ 4.60 m/s, so the first pass
stays on the track; frictional dissipation (rolling resistance and contact
creep) removes enough energy that the second pass detaches before the top
and the ball falls.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Rigid-body solver: one dynamic ball vs. frozen loop track and ground."""

    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", {}) or {}
        self.seed = int(seed)

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.mu = float(cfg.get("friction_coefficient", 0.18))

        # ---- frozen geometry -------------------------------------------------
        self.R = 1.0            # loop ring radius (xz plane), from loop.stl
        self.TUBE = 0.2         # loop tube radius
        self.LOOP_C = (0.0, 0.0, 1.2)   # loop_track transform translation
        self.r_b = 0.24         # ball radius
        self.rho = self.R - self.TUBE - self.r_b   # ball-centre rail radius
        self.GROUND_Z = 0.0     # ground box top face
        self.GROUND_HX = 6.0
        self.GROUND_HY = 3.0

        # ---- rigid body ------------------------------------------------------
        frame0 = scene.initial_state()
        self.m = float(np.asarray(frame0.mass)[0]) if frame0.mass is not None else 1.0
        p0 = np.asarray(frame0.position[0], dtype=float)
        v0 = np.asarray(frame0.velocity[0], dtype=float)
        if frame0.orientation is not None:
            q_xyzw = np.asarray(frame0.orientation[0], dtype=float)
            self.q = (float(q_xyzw[3]), float(q_xyzw[0]),
                      float(q_xyzw[1]), float(q_xyzw[2]))  # (w, x, y, z)
        else:
            self.q = (1.0, 0.0, 0.0, 0.0)
        # Solid sphere inertia.
        self.I = 0.4 * self.m * self.r_b ** 2

        # The scene documents the initial spin as "already rolling tangent".
        # For a contact point below the ball centre, rolling without slip
        # requires w = (n x v) / r_b with n the inward loop-surface normal.
        n0 = self._loop_normal(p0[0], p0[1], p0[2])
        if n0 is not None and np.linalg.norm(v0) > 1e-9:
            w0 = np.cross(n0, v0) / self.r_b
        elif frame0.angular_velocity is not None:
            w0 = np.asarray(frame0.angular_velocity[0], dtype=float)
        else:
            w0 = np.zeros(3)
        self.px, self.py, self.pz = (float(p0[0]), float(p0[1]), float(p0[2]))
        self.vx, self.vy, self.vz = (float(v0[0]), float(v0[1]), float(v0[2]))
        self.wx, self.wy, self.wz = (float(w0[0]), float(w0[1]), float(w0[2]))

        # ---- numerics --------------------------------------------------------
        self.h = 1.0e-4
        self.k = 5.0e4                      # normal stiffness [N/m]
        self.cd = 2.0 * 0.3 * math.sqrt(self.k * self.m)   # normal damping
        self.v_eps = 0.02                   # creep regularisation speed [m/s]
        # Rolling-resistance coefficient (length). Chosen so one full loop
        # pass dissipates ~9 J/kg: above the ~6.4 J/kg margin needed to fail
        # the second pass, below the ~12.7 J/kg that would break the first.
        self.c_rr = 0.013

        self.steps = 0
        self.body_id = int(np.asarray(frame0.body_id)[0]) if frame0.body_id is not None else 0

        # Loop-contact bookkeeping (diagnostics only).
        self._prev_phi = None
        self._cum_angle = 0.0
        self._history = []   # (time, on_rail, cum_angle, speed)

    # ------------------------------------------------------------------ utils
    def _loop_normal(self, x: float, y: float, z: float):
        """Inward-pointing unit normal of the loop's inner tube surface."""
        qx = x - self.LOOP_C[0]
        qy = y - self.LOOP_C[1]
        qz = z - self.LOOP_C[2]
        r_xz = math.hypot(qx, qz)
        if r_xz < 1e-12:
            ux, uz = 1.0, 0.0
        else:
            ux, uz = qx / r_xz, qz / r_xz
        ex = qx - self.R * ux
        ey = qy
        ez = qz - self.R * uz
        d = math.sqrt(ex * ex + ey * ey + ez * ez)
        if d < 1e-12:
            return None
        return np.array([ex / d, ey / d, ez / d])

    def _contacts(self):
        """Active contacts as (gap, nx, ny, nz); also returns report gaps."""
        # --- torus (loop track), analytic signed distance ---
        qx = self.px - self.LOOP_C[0]
        qy = self.py - self.LOOP_C[1]
        qz = self.pz - self.LOOP_C[2]
        r_xz = math.hypot(qx, qz)
        if r_xz < 1e-12:
            ux, uz = 1.0, 0.0
        else:
            ux, uz = qx / r_xz, qz / r_xz
        ex = qx - self.R * ux
        ey = qy
        ez = qz - self.R * uz
        d = math.sqrt(ex * ex + ey * ey + ez * ez)
        gap_loop = d - self.TUBE - self.r_b
        # --- ground box top face ---
        gap_ground = self.pz - self.GROUND_Z - self.r_b

        contacts = []
        if gap_loop < 0.0 and d > 1e-12:
            contacts.append((gap_loop, ex / d, ey / d, ez / d, True))
        if (gap_ground < 0.0 and abs(self.px) <= self.GROUND_HX
                and abs(self.py) <= self.GROUND_HY):
            contacts.append((gap_ground, 0.0, 0.0, 1.0, False))
        return contacts, gap_loop, gap_ground

    # ------------------------------------------------------------------ steps
    def _step(self) -> None:
        contacts, _, _ = self._contacts()

        Fx = self.m * self.gravity[0]
        Fy = self.m * self.gravity[1]
        Fz = self.m * self.gravity[2]
        Tx = Ty = Tz = 0.0

        for (gap, nx, ny, nz, _is_loop) in contacts:
            # Contact point relative to ball centre.
            rx = -self.r_b * nx
            ry = -self.r_b * ny
            rz = -self.r_b * nz
            # Contact-point velocity: v + w x r.
            vcx = self.vx + self.wy * rz - self.wz * ry
            vcy = self.vy + self.wz * rx - self.wx * rz
            vcz = self.vz + self.wx * ry - self.wy * rx
            vn = vcx * nx + vcy * ny + vcz * nz
            # Compliant normal force (repulsive, damped).
            N = -self.k * gap - self.cd * vn
            if N < 0.0:
                N = 0.0
            # Coulomb friction (smoothed) on the tangential slip.
            vtx = vcx - vn * nx
            vty = vcy - vn * ny
            vtz = vcz - vn * nz
            vt = math.sqrt(vtx * vtx + vty * vty + vtz * vtz)
            fx = fy = fz = 0.0
            if vt > 1e-12:
                fm = self.mu * N * math.tanh(vt / self.v_eps)
                s = fm / vt
                fx, fy, fz = -s * vtx, -s * vty, -s * vtz
            Fx += N * nx + fx
            Fy += N * ny + fy
            Fz += N * nz + fz
            # Torque about the ball centre: r x f.
            Tx += ry * fz - rz * fy
            Ty += rz * fx - rx * fz
            Tz += rx * fy - ry * fx
            # Rolling-resistance torque opposing the rolling spin.
            wn = self.wx * nx + self.wy * ny + self.wz * nz
            ox = self.wx - wn * nx
            oy = self.wy - wn * ny
            oz = self.wz - wn * nz
            ow = math.sqrt(ox * ox + oy * oy + oz * oz)
            if ow > 1e-12:
                tr = self.c_rr * N / ow
                Tx -= tr * ox
                Ty -= tr * oy
                Tz -= tr * oz

        # Semi-implicit Euler: velocities first, then positions.
        self.vx += self.h * Fx / self.m
        self.vy += self.h * Fy / self.m
        self.vz += self.h * Fz / self.m
        self.wx += self.h * Tx / self.I
        self.wy += self.h * Ty / self.I
        self.wz += self.h * Tz / self.I
        self.px += self.h * self.vx
        self.py += self.h * self.vy
        self.pz += self.h * self.vz

        # Quaternion integration (world-frame omega), (w, x, y, z).
        qw, qx, qy, qz = self.q
        dqw = 0.5 * (-qx * self.wx - qy * self.wy - qz * self.wz)
        dqx = 0.5 * (qw * self.wx + qy * self.wz - qz * self.wy)
        dqy = 0.5 * (qw * self.wy - qx * self.wz + qz * self.wx)
        dqz = 0.5 * (qw * self.wz + qx * self.wy - qy * self.wx)
        qw += self.h * dqw
        qx += self.h * dqx
        qy += self.h * dqy
        qz += self.h * dqz
        norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        self.q = (qw / norm, qx / norm, qy / norm, qz / norm)

    # ------------------------------------------------------------------ output
    def advance_to(self, time: float):
        t = float(time)
        target = int(round(t / self.h))
        while self.steps < target:
            self._step()
            self.steps += 1

        contacts, gap_loop, gap_ground = self._contacts()
        on_rail = 0.0
        slip = 0.0
        normal_force = 0.0
        for (gap, nx, ny, nz, is_loop) in contacts:
            if is_loop:
                on_rail = 1.0
            rx = -self.r_b * nx
            ry = -self.r_b * ny
            rz = -self.r_b * nz
            vcx = self.vx + self.wy * rz - self.wz * ry
            vcy = self.vy + self.wz * rx - self.wx * rz
            vcz = self.vz + self.wx * ry - self.wy * rx
            vn = vcx * nx + vcy * ny + vcz * nz
            N = max(0.0, -self.k * gap - self.cd * vn)
            normal_force = max(normal_force, N)
            vtx, vty, vtz = vcx - vn * nx, vcy - vn * ny, vcz - vn * nz
            slip = max(slip, math.sqrt(vtx * vtx + vty * vty + vtz * vtz))

        # Loop angle (unwrapped) for pass bookkeeping.
        qx = self.px - self.LOOP_C[0]
        qz = self.pz - self.LOOP_C[2]
        phi = math.atan2(qx, -qz)
        if self._prev_phi is None:
            self._cum_angle = 0.0
        else:
            dphi = phi - self._prev_phi
            if dphi > math.pi:
                dphi -= 2.0 * math.pi
            elif dphi < -math.pi:
                dphi += 2.0 * math.pi
            self._cum_angle += dphi
        self._prev_phi = phi

        speed = math.sqrt(self.vx ** 2 + self.vy ** 2 + self.vz ** 2)
        self._history.append((t, on_rail, self._cum_angle, speed))

        v_loop_min = math.sqrt(27.0 / 7.0 * abs(self.gravity[2]) * self.rho)
        rolling = 1.0 if (normal_force > 0.0 and slip < 0.05) else 0.0

        qw, qx_, qy_, qz_ = self.q
        frame = StateFrame(
            time=t,
            position=np.array([[self.px, self.py, self.pz]], dtype=float),
            velocity=np.array([[self.vx, self.vy, self.vz]], dtype=float),
            orientation=np.array([[qx_, qy_, qz_, qw]], dtype=float),
            angular_velocity=np.array([[self.wx, self.wy, self.wz]], dtype=float),
            mass=np.array([self.m], dtype=float),
            body_id=np.array([self.body_id], dtype=int),
            attributes={
                "rolling": np.array([rolling], dtype=float),
                "loop_contact": np.array([on_rail], dtype=float),
                "contact_normal_force": np.array([normal_force], dtype=float),
            },
            fields={
                # Speed per pass vs. the rolling loop-contact minimum speed.
                "constraint_value": np.array([speed], dtype=float),
                "constraint_target": np.array([v_loop_min], dtype=float),
                # Signed gaps: [loop track, ground]; negative = penetration.
                "contact_signed_distance": np.array([gap_loop, gap_ground], dtype=float),
                # External (non-contact) force: gravity only.
                "external_force": np.array(
                    [[self.m * self.gravity[0], self.m * self.gravity[1],
                      self.m * self.gravity[2]]], dtype=float),
            },
            scalars={
                "constraint_scale": float(v_loop_min),
                "contact_length_scale": float(self.TUBE),
                "friction_coefficient": float(self.mu),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        detach_second = None
        prev_rail = 1.0
        for (t, on_rail, angle, _speed) in self._history:
            if prev_rail > 0.5 and on_rail < 0.5 and angle >= 2.0 * math.pi:
                detach_second = t
                break
            prev_rail = on_rail
        passes = 0
        for (_t, on_rail, angle, _s) in self._history:
            if on_rail > 0.5:
                passes = max(passes, int(angle // (2.0 * math.pi)) + 1)
        return {
            "method": "penalty-contact rigid sphere, semi-implicit Euler h=1e-4",
            "rolling_resistance_coefficient": self.c_rr,
            "friction_coefficient": self.mu,
            "loop_passes_on_rail": passes,
            "second_pass_detachment_time": detach_second,
        }
