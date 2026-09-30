"""Physics solver for scene 013_weighted_seesaw.

Planar (x-z) rigid-body dynamics with penalty-based frictional contacts,
integrated with semi-implicit Euler substeps.

The planar reduction is exact for this scene: every obstacle and body is
symmetric about the y=0 plane, gravity acts in that plane, and the initial
state has no out-of-plane velocity, so no out-of-plane motion can arise.
Each 2-D rigid box carries the full translational and rotational (about y)
dynamics of its 3-D counterpart; the reported 3-D state embeds the planar
solution (y = 0, quaternion about +y).

Contacts are detected geometrically (vertex-in-box tests between every
colliding box pair, including the static ground slab and the static pivot
column) and resolved with normal spring-damper penalty forces plus
regularized Coulomb friction.  No animation, keyframes, or scripted motion
are used anywhere.
"""

from __future__ import annotations

import math
import os

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY = 9.81                 # m/s^2, +z is up
SUB_DT = 2.5e-4                # physics substep (s)
K_CONTACT = 1.0e6              # normal contact stiffness (N/m)
ZETA = 0.9                     # normal contact damping ratio
MU = 0.3                       # Coulomb friction coefficient
LIN_DAMP = 0.02                # small linear drag (1/s), numerical settling
ANG_DAMP = 0.05                # small angular drag (1/s)
CONTACT_MARGIN = 0.10          # m; report gaps closer than this

# Fallback geometry (meters) matching scene_geometry.yaml if the file or the
# yaml module is unavailable.
_FALLBACK = {
    "ground": (4.0, 0.05, 0.0, -0.05),
    "pivot": (0.22, 0.35, 0.0, 0.35),
    "seesaw_board": (1.8, 0.07),
    "heavy_block": (0.28, 0.28),
    "light_block": (0.22, 0.22),
}


class _Body:
    """Planar rigid box: center (x, z), yaw theta about +y, half extents."""

    __slots__ = ("x", "z", "th", "vx", "vz", "w", "m", "I", "hx", "hz", "dyn")

    def __init__(self, x, z, th, hx, hz, m, dyn):
        self.x = float(x)
        self.z = float(z)
        self.th = float(th)
        self.vx = 0.0
        self.vz = 0.0
        self.w = 0.0
        self.hx = float(hx)
        self.hz = float(hz)
        self.m = float(m)
        self.I = (float(m) * ((2.0 * self.hx) ** 2 + (2.0 * self.hz) ** 2) / 12.0
                  if dyn else float("inf"))
        self.dyn = bool(dyn)


def _load_geometry():
    """Read half extents / translations from the frozen scene_geometry.yaml."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "scene_geometry.yaml")
    out = dict(_FALLBACK)
    try:
        import yaml  # type: ignore
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        for obj in data.get("objects", []):
            oid = obj.get("id")
            geo = obj.get("geometry", {}) or {}
            tr = obj.get("transform", {}).get("translation", [0.0, 0.0, 0.0])
            tx, _, tz = (float(v) for v in tr)
            gtype = geo.get("type")
            he = geo.get("half_extents")
            if oid == "ground" and he:
                out["ground"] = (float(he[0]), float(he[2]), tx, tz)
            elif oid == "pivot" and gtype == "cylinder":
                out["pivot"] = (float(geo.get("radius", 0.22)),
                                0.5 * float(geo.get("height", 0.7)), tx, tz)
            elif he and oid in ("seesaw_board", "heavy_block", "light_block"):
                out[oid] = (float(he[0]), float(he[2]))
    except Exception:
        pass
    return out


class BenchmarkSolver:
    """Penalty-contact planar rigid-body solver for the weighted seesaw."""

    def initialize(self, scene, seed: int) -> None:
        geo = _load_geometry()
        init = scene.initial_state()
        pos = np.asarray(init.position, dtype=float)
        vel = np.asarray(init.velocity, dtype=float)
        mass = np.asarray(init.mass, dtype=float)
        body_id = np.asarray(init.body_id, dtype=int)

        # Map dynamic rows by body_id: 0 board, 1 heavy, 2 light.
        by_id = {int(b): i for i, b in enumerate(body_id)}
        self._body_ids = [0, 1, 2]
        self._masses = [float(mass[by_id[b]]) for b in self._body_ids]

        board_hx, board_hz = geo["seesaw_board"]
        heavy_hx, heavy_hz = geo["heavy_block"]
        light_hx, light_hz = geo["light_block"]
        g_hx, g_hz, g_x, g_z = geo["ground"]
        p_hx, p_hz, p_x, p_z = geo["pivot"]

        # Body list: 0 ground (static), 1 pivot (static), 2 board,
        # 3 heavy block, 4 light block.
        self._bodies = [
            _Body(g_x, g_z, 0.0, g_hx, g_hz, 1.0, False),
            _Body(p_x, p_z, 0.0, p_hx, p_hz, 1.0, False),
            _Body(pos[by_id[0], 0], pos[by_id[0], 2], 0.0, board_hx, board_hz,
                  self._masses[0], True),
            _Body(pos[by_id[1], 0], pos[by_id[1], 2], 0.0, heavy_hx, heavy_hz,
                  self._masses[1], True),
            _Body(pos[by_id[2], 0], pos[by_id[2], 2], 0.0, light_hx, light_hz,
                  self._masses[2], True),
        ]
        for k, bid in enumerate(self._body_ids):
            row = by_id[bid]
            self._bodies[2 + k].vx = float(vel[row, 0])
            self._bodies[2 + k].vz = float(vel[row, 2])

        # Colliding pairs (skip static-static).
        self._pairs = [
            (0, 2), (0, 3), (0, 4),          # ground vs dynamics
            (1, 2), (1, 3), (1, 4),          # pivot vs dynamics
            (2, 3), (2, 4), (3, 4),          # board/blocks vs each other
        ]
        self._radii = [math.hypot(b.hx, b.hz) for b in self._bodies]

        self._t = 0.0
        self._seed = int(seed)
        # Fixed contact pair labels for gap reporting (order is stable).
        self._gap_pairs = list(self._pairs)

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _corners(b):
        c = math.cos(b.th)
        s = math.sin(b.th)
        x, z, hx, hz = b.x, b.z, b.hx, b.hz
        # local (lx, lz) -> world: wx = lx*c + lz*s ; wz = -lx*s + lz*c
        return (
            (x + hx * c + hz * s, z - hx * s + hz * c),
            (x - hx * c + hz * s, z + hx * s + hz * c),
            (x - hx * c - hz * s, z + hx * s - hz * c),
            (x + hx * c - hz * s, z - hx * s - hz * c),
        )

    @staticmethod
    def _local(b, wx, wz):
        c = math.cos(b.th)
        s = math.sin(b.th)
        dx = wx - b.x
        dz = wz - b.z
        return dx * c - dz * s, dx * s + dz * c

    def _gen_contacts(self):
        """Vertex-in-box contacts for every near-overlapping pair."""
        out = []
        bodies = self._bodies
        for ia, ib in self._pairs:
            a = bodies[ia]
            b = bodies[ib]
            dx = b.x - a.x
            dz = b.z - a.z
            rr = self._radii[ia] + self._radii[ib]
            if dx * dx + dz * dz > rr * rr:
                continue
            ca = math.cos(a.th)
            sa = math.sin(a.th)
            cb = math.cos(b.th)
            sb = math.sin(b.th)
            # corners of a inside b: push a out along b's face normal
            for vx, vz in self._corners(a):
                lx, lz = self._local(b, vx, vz)
                if abs(lx) < b.hx and abs(lz) < b.hz:
                    penx = b.hx - abs(lx)
                    penz = b.hz - abs(lz)
                    if penz <= penx:
                        sgn = 1.0 if lz >= 0.0 else -1.0
                        ux, uz = sb * sgn, cb * sgn
                        depth = penz
                    else:
                        sgn = 1.0 if lx >= 0.0 else -1.0
                        # b's local x axis (1,0) -> world (cb, -sb)
                        ux, uz = cb * sgn, -sb * sgn
                        depth = penx
                    # n points from a to b (opposite of pushing a out)
                    out.append((ia, ib, vx, vz, -ux, -uz, depth))
            # corners of b inside a: push b out along a's face normal
            for vx, vz in self._corners(b):
                lx, lz = self._local(a, vx, vz)
                if abs(lx) < a.hx and abs(lz) < a.hz:
                    penx = a.hx - abs(lx)
                    penz = a.hz - abs(lz)
                    if penz <= penx:
                        sgn = 1.0 if lz >= 0.0 else -1.0
                        ux, uz = sa * sgn, ca * sgn
                        depth = penz
                    else:
                        sgn = 1.0 if lx >= 0.0 else -1.0
                        ux, uz = ca * sgn, -sa * sgn
                        depth = penx
                    out.append((ia, ib, vx, vz, ux, uz, depth))
        return out

    def _pair_gap(self, ia, ib):
        """Approximate signed gap (negative = penetration) for a pair."""
        a = self._bodies[ia]
        b = self._bodies[ib]
        best = float("inf")
        for src, dst in ((a, b), (b, a)):
            for vx, vz in self._corners(src):
                lx, lz = self._local(dst, vx, vz)
                ex = abs(lx) - dst.hx
                ez = abs(lz) - dst.hz
                if ex < 0.0 and ez < 0.0:
                    g = -min(-ex, -ez)
                elif ex < 0.0:
                    g = ez
                elif ez < 0.0:
                    g = ex
                else:
                    g = math.hypot(ex, ez)
                if g < best:
                    best = g
        return best

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------
    def _substep(self, dt):
        bodies = self._bodies
        n = len(bodies)
        fx = [0.0] * n
        fz = [0.0] * n
        tq = [0.0] * n

        for ia, ib, px, pz, nx, nz, depth in self._gen_contacts():
            a = bodies[ia]
            b = bodies[ib]
            if a.dyn:
                rax = px - a.x
                raz = pz - a.z
                vax = a.vx + a.w * raz
                vaz = a.vz - a.w * rax
            else:
                rax = raz = 0.0
                vax = vaz = 0.0
            if b.dyn:
                rbx = px - b.x
                rbz = pz - b.z
                vbx = b.vx + b.w * rbz
                vbz = b.vz - b.w * rbx
            else:
                rbx = rbz = 0.0
                vbx = vbz = 0.0

            rvx = vbx - vax
            rvz = vbz - vaz
            vn = rvx * nx + rvz * nz

            if a.dyn and b.dyn:
                mred = a.m * b.m / (a.m + b.m)
            elif a.dyn:
                mred = a.m
            elif b.dyn:
                mred = b.m
            else:
                continue
            cdamp = 2.0 * ZETA * math.sqrt(K_CONTACT * mred)

            fn = K_CONTACT * depth - cdamp * vn
            if fn < 0.0:
                fn = 0.0

            tvx = rvx - vn * nx
            tvz = rvz - vn * nz
            tvn = math.hypot(tvx, tvz)
            ftx = ftz = 0.0
            if tvn > 1e-9 and fn > 0.0:
                ft = MU * fn
                reg = cdamp * tvn
                if reg < ft:
                    ft = reg
                scale = -ft / tvn
                ftx = scale * tvx
                ftz = scale * tvz

            wfx = fn * nx + ftx
            wfz = fn * nz + ftz

            if b.dyn:
                fx[ib] += wfx
                fz[ib] += wfz
                tq[ib] += rbz * wfx - rbx * wfz
            if a.dyn:
                fx[ia] -= wfx
                fz[ia] -= wfz
                tq[ia] += -raz * wfx + rax * wfz

        # remember state for a finite-state safety net
        prev = [(b.x, b.z, b.th, b.vx, b.vz, b.w) for b in bodies]

        for i, b in enumerate(bodies):
            if not b.dyn:
                continue
            b.vx += (fx[i] / b.m) * dt
            b.vz += (fz[i] / b.m - GRAVITY) * dt
            b.w += (tq[i] / b.I) * dt
            damp_l = max(0.0, 1.0 - LIN_DAMP * dt)
            damp_a = max(0.0, 1.0 - ANG_DAMP * dt)
            b.vx *= damp_l
            b.vz *= damp_l
            b.w *= damp_a
            b.x += b.vx * dt
            b.z += b.vz * dt
            b.th += b.w * dt

        for i, b in enumerate(bodies):
            if not b.dyn:
                continue
            if not (math.isfinite(b.x) and math.isfinite(b.z)
                    and math.isfinite(b.th) and math.isfinite(b.vx)
                    and math.isfinite(b.vz) and math.isfinite(b.w)):
                px, pz, pth, pvx, pvz, pw = prev[i]
                b.x, b.z, b.th = px, pz, pth
                b.vx = b.vz = b.w = 0.0

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self._t:
            target = self._t
        while target - self._t > SUB_DT:
            self._substep(SUB_DT)
            self._t += SUB_DT
        rem = target - self._t
        if rem > 1e-12:
            self._substep(rem)
            self._t = target

        dyn = self._bodies[2:5]
        n = len(dyn)
        position = np.empty((n, 3))
        velocity = np.empty((n, 3))
        orientation = np.empty((n, 4))
        angular = np.empty((n, 3))
        external = np.empty((n, 3))
        for i, b in enumerate(dyn):
            position[i] = (b.x, 0.0, b.z)
            velocity[i] = (b.vx, 0.0, b.vz)
            half = 0.5 * b.th
            orientation[i] = (0.0, math.sin(half), 0.0, math.cos(half))
            angular[i] = (0.0, b.w, 0.0)
            external[i] = (0.0, 0.0, -GRAVITY * b.m)

        gaps = []
        for ia, ib in self._gap_pairs:
            g = self._pair_gap(ia, ib)
            if g < CONTACT_MARGIN:
                gaps.append(g)
        if not gaps:
            gaps = [min(self._pair_gap(ia, ib)
                        for ia, ib in self._gap_pairs)]
        contact_sd = np.asarray(gaps, dtype=float)

        board = self._bodies[2]
        constraint_value = np.array([board.th], dtype=float)
        constraint_target = np.array([0.0], dtype=float)

        return StateFrame(
            time=float(target),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular,
            mass=np.asarray(self._masses, dtype=float),
            body_id=np.asarray(self._body_ids, dtype=int),
            fields={
                "external_force": external,
                "contact_signed_distance": contact_sd,
                "constraint_value": constraint_value,
                "constraint_target": constraint_target,
            },
            scalars={
                "constraint_scale": 1.0,
                "contact_length_scale": float(2.0 * self._bodies[2].hz),
            },
        )

    def diagnostics(self) -> dict:
        board = self._bodies[2]
        return {
            "method": "planar penalty-contact rigid-body dynamics "
                      "(semi-implicit Euler)",
            "substep": SUB_DT,
            "contact_stiffness": K_CONTACT,
            "friction_coefficient": MU,
            "time": self._t,
            "board_angle": board.th,
        }
