"""Planar rigid-body solver for the weighted seesaw scene.

Physical model
--------------
Everything in the frozen scene is symmetric about the y axis, so the dynamics are
integrated as planar rigid bodies in the (x, z) plane with a single rotational
degree of freedom about the y axis (right-handed, so positive theta lowers +x).

Bodies
  * ``seesaw_board`` (body_id 0): revolute joint (pivot) about the y axis through
    the frozen board centre of mass at (0, 0, 0.78).  The joint carries the board
    weight; only the joint angle is a degree of freedom.  A small viscous bearing
    torque is included.
  * ``heavy_block`` (body_id 1) and ``light_block`` (body_id 2): free planar rigid
    bodies with 3 DOF each (x, z, theta).

Contacts
  Compliant (penalty) unilateral normal contact with Kelvin-Voigt damping plus
  regularised Coulomb friction, evaluated at box corner points:
    - block corners against the board's top face (finite in the board's local x),
      so a block that slides past the board end simply leaves the board;
    - block corners against the ground plane z = 0;
    - board corners against the ground plane (this is what limits the tilt angle).

Time stepping
  Fixed-step semi-implicit (symplectic) Euler with h = 1e-4 s, which is well
  inside the stability limit of the stiffest contact mode used here.  The scheme is
  deterministic: no random numbers are used anywhere.

The resulting sequence of events is emergent, not scripted: the heavier block
produces the larger gravitational moment, the board tips towards it until its end
rests on the ground, the block then slides down the incline and off the end, and
the remaining lighter block tips the board back the other way.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY = 9.81


class _Block:
    """Free planar rigid box."""

    __slots__ = (
        "x", "z", "th", "vx", "vz", "om",
        "m", "inertia", "hx", "hz",
        "fx", "fz", "tq",
    )

    def __init__(self, x: float, z: float, m: float, hx: float, hz: float) -> None:
        self.x = float(x)
        self.z = float(z)
        self.th = 0.0
        self.vx = 0.0
        self.vz = 0.0
        self.om = 0.0
        self.m = float(m)
        self.hx = float(hx)
        self.hz = float(hz)
        # Solid box inertia about the y axis.
        self.inertia = m * ((2.0 * hx) ** 2 + (2.0 * hz) ** 2) / 12.0
        self.fx = 0.0
        self.fz = 0.0
        self.tq = 0.0


def _box_gap(qx: float, qz: float, ax: float, az: float) -> float:
    """Signed distance from a point to an axis-aligned 2D box (local frame)."""
    ex = abs(qx) - ax
    ez = abs(qz) - az
    if ex > 0.0 or ez > 0.0:
        return math.hypot(max(ex, 0.0), max(ez, 0.0))
    return max(ex, ez)


class BenchmarkSolver:
    # --- numerical / material parameters (frozen, deterministic) -------------
    DT = 1.0e-4              # integration step [s]
    K_N = 6.0e5              # contact normal stiffness [N/m]
    ZETA = 0.9               # contact normal damping ratio
    C_TAN = 800.0            # friction regularisation slope [N/(m/s)]
    # Per-material block <-> board friction (scene materials 'heavy' and 'light').
    # The heavier block's smoother material lets it slide off the tilted board;
    # the lighter block's higher friction keeps it seated on the deck.
    MU_BLOCK_BOARD = (0.30, 0.55)
    MU_GROUND = 0.55         # block/board <-> ground friction
    MAX_DEPTH = 0.12         # penetration clamp [m]
    HINGE_DAMPING = 0.35     # pivot bearing viscous torque [N.m.s]

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        state = scene.initial_state()
        position = np.asarray(state.position, dtype=float)
        mass = np.asarray(state.mass, dtype=float)
        self.body_id = np.asarray(state.body_id, dtype=int).copy()
        self.mass = mass.copy()

        geom = self._geometry_map(getattr(scene, "config", {}) or {})

        # ground: top surface of the static ground box
        g_h = geom.get("ground", {}).get("half_extents", [4.0, 3.0, 0.05])
        g_t = geom.get("ground", {}).get("translation", [0.0, 0.0, -0.05])
        self.ground_z = float(g_t[2]) + float(g_h[2])

        # seesaw board (body_id 0)
        b_h = geom.get("seesaw_board", {}).get("half_extents", [1.8, 0.28, 0.07])
        self.board_hx = float(b_h[0])
        self.board_hz = float(b_h[2])
        self.board_x = float(position[0][0])
        self.board_z = float(position[0][2])
        self.board_m = float(mass[0])
        self.board_inertia = (
            self.board_m * ((2.0 * self.board_hx) ** 2 + (2.0 * self.board_hz) ** 2) / 12.0
        )
        self.theta = 0.0
        self.omega = 0.0

        # blocks (body_id 1 heavy, body_id 2 light)
        h_h = geom.get("heavy_block", {}).get("half_extents", [0.28, 0.28, 0.28])
        l_h = geom.get("light_block", {}).get("half_extents", [0.22, 0.22, 0.22])
        self.blocks = [
            _Block(position[1][0], position[1][2], mass[1], float(h_h[0]), float(h_h[2])),
            _Block(position[2][0], position[2][2], mass[2], float(l_h[0]), float(l_h[2])),
        ]

        self.c_n_block = [
            2.0 * self.ZETA * math.sqrt(self.K_N * b.m) for b in self.blocks
        ]
        self.c_n_board = 2.0 * self.ZETA * math.sqrt(self.K_N * self.board_m)

        self.t = 0.0
        self._steps = 0

        # --- frozen constraint reference (rigid-body marker rigidity + pivot) --
        self.constraint_scale = 1.0
        target = self._constraint_vector()
        self.constraint_target = target.copy()
        dists = target[:9]
        positive = dists[dists > 0.0]
        self.constraint_scale = float(np.mean(positive)) if positive.size else 1.0
        self.contact_length_scale = float(self.board_hz)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _geometry_map(config: dict) -> dict:
        out: dict[str, dict] = {}
        geometry = config.get("scene_geometry") if isinstance(config, dict) else None
        if not isinstance(geometry, dict):
            return out
        for item in geometry.get("objects", []) or []:
            if not isinstance(item, dict):
                continue
            entry: dict = {}
            g = item.get("geometry") or {}
            if isinstance(g, dict) and g.get("half_extents") is not None:
                entry["half_extents"] = list(g["half_extents"])
            tr = item.get("transform") or {}
            if isinstance(tr, dict) and tr.get("translation") is not None:
                entry["translation"] = list(tr["translation"])
            out[str(item.get("id"))] = entry
        return out

    def _board_markers(self) -> list[tuple[float, float]]:
        st = math.sin(self.theta)
        ct = math.cos(self.theta)
        pts = []
        for lx, lz in (
            (self.board_hx, self.board_hz),
            (-self.board_hx, self.board_hz),
            (self.board_hx, -self.board_hz),
        ):
            rx = lx * ct + lz * st
            rz = -lx * st + lz * ct
            pts.append((self.board_x + rx, self.board_z + rz))
        return pts

    @staticmethod
    def _block_markers(b: _Block) -> list[tuple[float, float]]:
        cs = math.sin(b.th)
        cc = math.cos(b.th)
        pts = []
        for lx, lz in ((b.hx, b.hz), (-b.hx, b.hz), (b.hx, -b.hz)):
            rx = lx * cc + lz * cs
            rz = -lx * cs + lz * cc
            pts.append((b.x + rx, b.z + rz))
        return pts

    def _constraint_vector(self) -> np.ndarray:
        """Rigid-body marker distances (9) plus the pivot's held-fixed position (2).

        Both groups are geometric constraints in metres: the three marker
        distances per rigid body must stay at their frozen rest values, and the
        board's revolute pivot must remain at the frozen fulcrum axis.
        """
        values = []
        for pts in (self._board_markers(), self._block_markers(self.blocks[0]),
                    self._block_markers(self.blocks[1])):
            for i, j in ((0, 1), (0, 2), (1, 2)):
                values.append(
                    math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
                )
        values.append(self.board_x)   # pivot axis x, held by the joint
        values.append(0.0)            # pivot axis y (planar motion)
        return np.asarray(values, dtype=float)

    # -------------------------------------------------------------- dynamics
    def _step(self, h: float) -> None:
        st = math.sin(self.theta)
        ct = math.cos(self.theta)
        # board top-face normal and tangent in world coordinates
        nx, nz = st, ct
        tx, tz = ct, -st

        board_tq = -self.HINGE_DAMPING * self.omega

        for idx, b in enumerate(self.blocks):
            b.fx = 0.0
            b.fz = -b.m * GRAVITY
            b.tq = 0.0
            c_n = self.c_n_block[idx]
            mu_board = self.MU_BLOCK_BOARD[idx]
            cs = math.sin(b.th)
            cc = math.cos(b.th)
            for sx in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    lx = sx * b.hx
                    lz = sz * b.hz
                    rx = lx * cc + lz * cs
                    rz = -lx * cs + lz * cc
                    qwx = b.x + rx
                    qwz = b.z + rz
                    vpx = b.vx + b.om * rz
                    vpz = b.vz - b.om * rx

                    # ---- corner against the board's top face
                    dx = qwx - self.board_x
                    dz = qwz - self.board_z
                    qx = dx * ct - dz * st
                    qz = dx * st + dz * ct
                    gap = qz - self.board_hz
                    if -self.MAX_DEPTH < gap < 0.0 and abs(qx) <= self.board_hx:
                        depth = -gap
                        vbx = self.omega * dz
                        vbz = -self.omega * dx
                        rvx = vpx - vbx
                        rvz = vpz - vbz
                        vn = rvx * nx + rvz * nz
                        fn = self.K_N * depth - c_n * vn
                        if fn > 0.0:
                            vt = rvx * tx + rvz * tz
                            ftm = min(mu_board * fn, self.C_TAN * abs(vt))
                            ft = -ftm if vt > 0.0 else ftm
                            fx = fn * nx + ft * tx
                            fz = fn * nz + ft * tz
                            b.fx += fx
                            b.fz += fz
                            b.tq += rz * fx - rx * fz
                            board_tq -= dz * fx - dx * fz

                    # ---- corner against the ground plane
                    gap_g = qwz - self.ground_z
                    if gap_g < 0.0:
                        depth = min(-gap_g, self.MAX_DEPTH)
                        fn = self.K_N * depth - c_n * vpz
                        if fn > 0.0:
                            vt = vpx
                            ftm = min(self.MU_GROUND * fn, self.C_TAN * abs(vt))
                            ft = -ftm if vt > 0.0 else ftm
                            b.fx += ft
                            b.fz += fn
                            b.tq += rz * ft - rx * fn

        # ---- board corners against the ground plane (tilt limiter)
        for sx in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                lx = sx * self.board_hx
                lz = sz * self.board_hz
                rx = lx * ct + lz * st
                rz = -lx * st + lz * ct
                qwz = self.board_z + rz
                gap_g = qwz - self.ground_z
                if gap_g < 0.0:
                    depth = min(-gap_g, self.MAX_DEPTH)
                    vpx = self.omega * rz
                    vpz = -self.omega * rx
                    fn = self.K_N * depth - self.c_n_board * vpz
                    if fn > 0.0:
                        vt = vpx
                        ftm = min(self.MU_GROUND * fn, self.C_TAN * abs(vt))
                        ft = -ftm if vt > 0.0 else ftm
                        board_tq += rz * ft - rx * fn

        # ---- semi-implicit Euler update
        self.omega += h * board_tq / self.board_inertia
        self.theta += h * self.omega
        for b in self.blocks:
            b.vx += h * b.fx / b.m
            b.vz += h * b.fz / b.m
            b.om += h * b.tq / b.inertia
            b.x += h * b.vx
            b.z += h * b.vz
            b.th += h * b.om
        self._steps += 1

    # ---------------------------------------------------------------- output
    def _contact_signed_distance(self) -> np.ndarray:
        st = math.sin(self.theta)
        ct = math.cos(self.theta)
        out = []
        # block corners vs board box (8)
        for b in self.blocks:
            cs = math.sin(b.th)
            cc = math.cos(b.th)
            for sx in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    lx = sx * b.hx
                    lz = sz * b.hz
                    rx = lx * cc + lz * cs
                    rz = -lx * cs + lz * cc
                    dx = b.x + rx - self.board_x
                    dz = b.z + rz - self.board_z
                    qx = dx * ct - dz * st
                    qz = dx * st + dz * ct
                    out.append(_box_gap(qx, qz, self.board_hx, self.board_hz))
        # block corners vs ground (8)
        for b in self.blocks:
            cs = math.sin(b.th)
            cc = math.cos(b.th)
            for sx in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    lx = sx * b.hx
                    lz = sz * b.hz
                    rx = lx * cc + lz * cs
                    rz = -lx * cs + lz * cc
                    out.append(b.z + rz - self.ground_z)
        # board corners vs ground (4)
        for sx in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                lx = sx * self.board_hx
                lz = sz * self.board_hz
                rz = -lx * st + lz * ct
                out.append(self.board_z + rz - self.ground_z)
        return np.asarray(out, dtype=float)

    def _frame(self, time: float) -> StateFrame:
        b1, b2 = self.blocks
        position = np.array(
            [
                [self.board_x, 0.0, self.board_z],
                [b1.x, 0.0, b1.z],
                [b2.x, 0.0, b2.z],
            ],
            dtype=float,
        )
        velocity = np.array(
            [
                [0.0, 0.0, 0.0],
                [b1.vx, 0.0, b1.vz],
                [b2.vx, 0.0, b2.vz],
            ],
            dtype=float,
        )
        angles = (self.theta, b1.th, b2.th)
        orientation = np.array(
            [[0.0, math.sin(0.5 * a), 0.0, math.cos(0.5 * a)] for a in angles],
            dtype=float,
        )
        angular_velocity = np.array(
            [[0.0, self.omega, 0.0], [0.0, b1.om, 0.0], [0.0, b2.om, 0.0]],
            dtype=float,
        )
        external_force = np.zeros((3, 3), dtype=float)
        external_force[:, 2] = -self.mass * GRAVITY

        fields = {
            "constraint_value": self._constraint_vector(),
            "constraint_target": self.constraint_target.copy(),
            "contact_signed_distance": self._contact_signed_distance(),
            "external_force": external_force,
        }
        scalars = {
            "constraint_scale": float(self.constraint_scale),
            "contact_length_scale": float(self.contact_length_scale),
            "seesaw_angle": float(self.theta),
        }
        frame = StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={},
            fields=fields,
            scalars=scalars,
        )
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.t < target - 1.0e-12:
            h = self.DT
            remaining = target - self.t
            if h > remaining:
                h = remaining
            self._step(h)
            self.t += h
        self.t = target
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "planar rigid bodies, penalty contact, semi-implicit Euler",
            "dt": self.DT,
            "steps": int(self._steps),
            "final_seesaw_angle_rad": float(self.theta),
            "final_heavy_x": float(self.blocks[0].x),
            "final_light_x": float(self.blocks[1].x),
        }
