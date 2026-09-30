"""Weighted seesaw solver: 2D rigid-body dynamics (x-z plane, rotation about y).

Physics model
-------------
- Board: hinged on top of the static pivot cylinder at H=(0,0,0.71); single
  generalized coordinate theta (rotation about +y).  Penalty contact between
  board bottom corners and the ground stops the tilt.
- Heavy / light blocks: free rigid boxes with penalty (spring-damper) contact
  against the board top surface (plane clipped to the board extent, plus
  rounded edge corners) and against the ground, with Coulomb-like friction.
- Reaction forces/torques from block contacts are applied back to the board,
  so load transfer through contact tips the seesaw: the heavy block drives the
  right side down, slides off the edge, and the light block then tips the
  board back the other way.
- Semi-implicit Euler at fixed h=1e-3 s.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

G = 9.81
MU = 0.30            # block/board and block/ground friction
HB = 0.71            # pivot-top hinge height
A = 1.8              # board half length
B = 0.07             # board half thickness
K_BLOCK = 1.0e5      # block contact stiffness
ZETA = 0.5           # contact damping ratio
K_CORNER = 2.0e5     # board-corner/ground stiffness
MARGIN = 5.0e-3      # contact detection margin
H_MAX = 1.0e-3       # integration step


def _rot(xb, zb, ct, st):
    """Board-frame -> world (x,z) rotation about y by theta."""
    return xb * ct + zb * st, -xb * st + zb * ct


class BenchmarkSolver:
    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        init = scene.initial_state()
        pos = np.asarray(init.position, dtype=float)
        self.mass = np.asarray(init.mass, dtype=float).copy()  # [board, heavy, light]
        self.body_id = np.asarray(init.body_id, dtype=int).copy()
        self.half = np.array([0.28, 0.22])  # block half extents (square section)
        self.iblk = [(1.0 / 6.0) * self.mass[1 + i] * (2.0 * self.half[i]) ** 2
                     for i in range(2)]
        self.iboard = (1.0 / 12.0) * self.mass[0] * ((2 * A) ** 2 + (2 * B) ** 2)
        self.theta = 0.0
        self.omega = 0.0
        self.p = [pos[1].copy(), pos[2].copy()]
        self.v = [np.zeros(3), np.zeros(3)]
        self.phi = [0.0, 0.0]
        self.phid = [0.0, 0.0]
        self.t = 0.0
        self.nsteps = 0
        self.max_pen = 0.0
        self.seed = int(seed)

    # ------------------------------------------------------------ dynamics
    def _board_center(self, ct, st):
        cx, cz = _rot(0.0, B, ct, st)
        return cx, cz + HB

    def _step(self, h: float) -> None:
        th = self.theta
        ct, st = np.cos(th), np.sin(th)

        # ---------------- board torque (gravity on offset center) ----------
        rcx, rcz = _rot(0.0, B, ct, st)          # board center rel. hinge
        tau = rcx * (-self.mass[0] * G)           # tau_y = r_x F_z - r_z F_x

        # ---------------- board corners vs ground --------------------------
        cc = 2.0 * 0.4 * np.sqrt(K_CORNER * self.mass[0])
        for sx in (-1.0, 1.0):
            rx, rz = _rot(sx * A, -B, ct, st)     # corner rel. hinge
            zc = HB + rz                          # corner world z (ground top=0)
            if zc < MARGIN:
                vz = -self.omega * rx             # corner velocity (omega y-hat x r)
                vx = self.omega * rz
                n_mag = K_CORNER * max(0.0, -zc) - cc * vz
                if n_mag > 0.0:
                    self.max_pen = max(self.max_pen, -zc)
                    fx = -MU * n_mag * np.tanh(vx / 0.05)
                    fz = n_mag
                    tau += rx * fz - rz * fx

        # ---------------- blocks --------------------------------------------
        for i in range(2):
            m = self.mass[1 + i]
            hb = self.half[i]
            p = self.p[i]
            v = self.v[i]
            fx, fz = 0.0, -m * G
            ang_torque = 0.0
            in_board_contact = False
            in_ground_contact = False

            # block vs ground (z >= 0 plane)
            dg = p[2] - hb
            if dg < MARGIN:
                cg = 2.0 * ZETA * np.sqrt(K_BLOCK * m)
                ng = K_BLOCK * max(0.0, -dg) - cg * v[2]
                if ng > 0.0:
                    in_ground_contact = True
                    self.max_pen = max(self.max_pen, -dg)
                    fz += ng
                    fx += -MU * ng * np.tanh(v[0] / 0.02)

            # block vs board top surface
            dx, dz = p[0] - 0.0, p[2] - HB
            xb = dx * ct - dz * st
            zb = dx * st + dz * ct
            if abs(xb) <= A:
                d = zb - B - hb
                nx, nz = _rot(0.0, 1.0, ct, st)
                rcx_c, rcz_c = _rot(xb, B, ct, st)
            elif xb > A:
                ex, ez = xb - A, zb - B
                dist = np.hypot(ex, ez) + 1e-12
                d = dist - hb
                bx, bz = ex / dist, ez / dist
                nx, nz = _rot(bx, bz, ct, st)
                rcx_c, rcz_c = _rot(A, B, ct, st)
            else:
                ex, ez = xb + A, zb - B
                dist = np.hypot(ex, ez) + 1e-12
                d = dist - hb
                bx, bz = ex / dist, ez / dist
                nx, nz = _rot(bx, bz, ct, st)
                rcx_c, rcz_c = _rot(-A, B, ct, st)

            if d < MARGIN:
                cb = 2.0 * ZETA * np.sqrt(K_BLOCK * m)
                vsx = self.omega * rcz_c          # board surface point velocity
                vsz = -self.omega * rcx_c
                rvx, rvz = v[0] - vsx, v[2] - vsz
                ddot = rvx * nx + rvz * nz
                n_mag = K_BLOCK * max(0.0, -d) - cb * ddot
                if n_mag > 0.0:
                    in_board_contact = True
                    self.max_pen = max(self.max_pen, -d)
                    vtx = rvx - ddot * nx
                    vtz = rvz - ddot * nz
                    vt_mag = np.hypot(vtx, vtz) + 1e-12
                    fr = MU * n_mag * np.tanh(vt_mag / 0.02)
                    ffx = n_mag * nx - fr * vtx / vt_mag
                    ffz = n_mag * nz - fr * vtz / vt_mag
                    fx += ffx
                    fz += ffz
                    # reaction on board at contact point rc (rel. hinge)
                    tau += rcx_c * (-ffz) - rcz_c * (-ffx)

            # block orientation: soft-align with contact surface
            ka = 200.0 * self.iblk[i]
            da = 20.0 * self.iblk[i]
            if in_board_contact:
                ang_torque = -ka * (self.phi[i] - th) - da * (self.phid[i] - self.omega)
            elif in_ground_contact:
                ang_torque = -ka * self.phi[i] - da * self.phid[i]

            # semi-implicit Euler for block
            v[0] += (fx / m) * h
            v[2] += (fz / m) * h
            p[0] += v[0] * h
            p[2] += v[2] * h
            self.phid[i] += (ang_torque / self.iblk[i]) * h
            self.phi[i] += self.phid[i] * h

            # hard floor safety (never interpenetrate ground)
            if p[2] < hb:
                p[2] = hb
                if v[2] < 0.0:
                    v[2] = 0.0

        # ---------------- board integration --------------------------------
        self.omega += (tau / self.iboard) * h
        self.theta += self.omega * h
        self.nsteps += 1

    # -------------------------------------------------------------- outputs
    def _signed_distances(self):
        ct, st = np.cos(self.theta), np.sin(self.theta)
        out = []
        for i in range(2):
            p = self.p[i]
            hb = self.half[i]
            dx, dz = p[0], p[2] - HB
            xb = dx * ct - dz * st
            zb = dx * st + dz * ct
            if abs(xb) <= A:
                d = zb - B - hb
            else:
                ex = xb - (A if xb > 0 else -A)
                ez = zb - B
                d = np.hypot(ex, ez) - hb
            out.append(d)
        for i in range(2):
            out.append(self.p[i][2] - self.half[i])  # block-ground gap
        return np.array(out, dtype=float)

    def _frame(self) -> StateFrame:
        th = self.theta
        ct, st = np.cos(th), np.sin(th)
        cx, cz = self._board_center(ct, st)
        rcx, rcz = _rot(0.0, B, ct, st)
        bvx = self.omega * rcz
        bvz = -self.omega * rcx
        position = np.array([
            [cx, 0.0, cz],
            [self.p[0][0], 0.0, self.p[0][2]],
            [self.p[1][0], 0.0, self.p[1][2]],
        ], dtype=float)
        velocity = np.array([
            [bvx, 0.0, bvz],
            [self.v[0][0], 0.0, self.v[0][2]],
            [self.v[1][0], 0.0, self.v[1][2]],
        ], dtype=float)
        ang = np.array([th, self.phi[0], self.phi[1]])
        orientation = np.zeros((3, 4), dtype=float)
        orientation[:, 3] = np.cos(ang / 2.0)
        orientation[:, 1] = np.sin(ang / 2.0)
        angular_velocity = np.array([
            [0.0, self.omega, 0.0],
            [0.0, self.phid[0], 0.0],
            [0.0, self.phid[1], 0.0],
        ], dtype=float)
        external_force = np.zeros((3, 3), dtype=float)
        external_force[:, 2] = -self.mass * G
        return StateFrame(
            time=float(self.t),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            fields={
                "constraint_value": np.array([th], dtype=float),
                "constraint_target": np.array([0.0], dtype=float),
                "contact_signed_distance": self._signed_distances(),
                "external_force": external_force,
            },
            scalars={
                "constraint_scale": 1.0,
                "contact_length_scale": 2.0 * B,
            },
        )

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.t < target - 1e-12:
            h = min(H_MAX, target - self.t)
            self._step(h)
            self.t += h
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "steps": self.nsteps,
            "max_penetration": float(self.max_pen),
            "theta": float(self.theta),
        }
