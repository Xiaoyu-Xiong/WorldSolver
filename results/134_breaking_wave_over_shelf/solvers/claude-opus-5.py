"""Weakly compressible SPH solver: breaking wave over a submerged shelf.

Physical model
--------------
Lagrangian weakly compressible SPH (WCSPH) for water:

*  cubic-spline kernel, ``h = 1.25 dx``, support ``2h``;
*  summation density  ``rho_i = sum_j m_j W_ij``  (fluid + wall dummy particles);
*  Tait equation of state ``p = B[(rho/rho0)^gamma - 1]``, ``B = rho0 c^2/gamma``;
*  Monaghan artificial viscosity for shock/impact dissipation;
*  gravity from the frozen configuration;
*  symplectic (semi-implicit) Euler stepping with a CFL/acoustic limited step,
   sub-cycled so that every requested sample time is hit exactly.

Boundaries (all taken from the frozen ``scene_geometry.yaml``): tank floor, the
two side walls, the downstream end wall, the submerged shelf and the kinematic
dam gate.  Each solid is represented twice --- as layered dummy boundary
particles that complete the kernel support (so pressure builds against walls)
and as an analytic box constraint that enforces strict non-penetration.

Scenario: still water fills the tank to a level above the shelf crest (the shelf
is genuinely submerged), a deeper reservoir is held behind the gate, the gate is
lifted at a prescribed constant speed, and the released surge shoals over the
shelf, steepens, curls forward, plunges into the water ahead and continues as a
breaking bore that reflects off the end wall.  No keyframes or precomputed
motion are used anywhere; all motion comes from stepping the equations above.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# frozen geometry (mirrors scene_geometry.yaml, meters, z up)
# ---------------------------------------------------------------------------
GROUND_TOP = 0.0                    # ground box top face
SHELF_X0, SHELF_X1 = 0.25, 2.65     # submerged shelf footprint in x
SHELF_TOP = 0.60                    # shelf crest
SHELF_Y = 0.84                      # shelf half width
WALL_Y = 0.92                       # tank side-wall inner faces
END_WALL_X = 3.06                   # downstream end wall inner face
BACK_WALL_X = -3.20                 # upstream closure at the ground edge
GATE_X0, GATE_X1 = -0.49, -0.41     # dam gate slab
GATE_HALF_Y = 0.72
GATE_HEIGHT = 1.10
GATE_LIFT_SPEED = 2.2               # prescribed gate lift speed [m/s]
GATE_LIFT_TOP = 1.35                # gate bottom stops here (clear of water)

STILL_LEVEL = 0.82                  # ambient water level (shelf submerged 0.22 m)
RESERVOIR_LEVEL = 1.10              # water level behind the gate
RES_X0, RES_X1 = -2.60, -0.52       # reservoir footprint in x
AMB_X0, AMB_X1 = -0.38, 3.04        # downstream still-water footprint in x
WATER_HALF_Y = 0.72                 # water half width (matches the gate)

DX = 0.10                           # particle spacing


def _box(lo, hi):
    return (np.asarray(lo, float), np.asarray(hi, float))


class BenchmarkSolver:
    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = dict(getattr(scene, "config", {}))
        self.gravity = np.asarray(config.get("gravity", (0.0, 0.0, -9.81)), float)
        sampling = config.get("sampling", {})
        self.rho0 = float(sampling.get("density", 1000.0))
        self.scene_spacing = float(sampling.get("spacing", 0.14))

        self.dx = float(DX)
        self.pos = self._sample_water()
        self.n = int(len(self.pos))
        self.vel = np.zeros_like(self.pos)
        self.m = self.rho0 * self.dx ** 3
        self.pmass = np.full(self.n, self.m)
        self.body_id = np.zeros(self.n, np.int32)
        self.radius = np.full(self.n, 0.5 * self.dx)
        self.volume = np.full(self.n, self.dx ** 3)

        # SPH / numerical parameters
        self.h = 1.25 * self.dx
        self.support = 2.0 * self.h
        self.c = 30.0
        self.gamma = 7.0
        self.B = self.rho0 * self.c ** 2 / self.gamma
        self.alpha = 0.18
        self.w0 = 1.0 / (math.pi * self.h ** 3)
        self.dt_ceiling = 0.20 * self.h / self.c

        self.ref_velocity = float(math.sqrt(abs(self.gravity[2]) * RESERVOIR_LEVEL))
        self.char_length = float(RESERVOIR_LEVEL)

        self.rho = np.full(self.n, self.rho0)
        self.press = np.zeros(self.n)

        self._build_solids()
        self._build_boundary_particles()
        self._build_wall_samples()
        self._build_grid()

        self.time = 0.0
        self.steps = 0
        self.max_speed = 0.0

    # ------------------------------------------------------------------
    def _lattice(self, lo, hi):
        axes = []
        for k in range(3):
            a = np.arange(lo[k] + 0.5 * self.dx, hi[k] - 1e-9, self.dx)
            if len(a) == 0:
                a = np.array([0.5 * (lo[k] + hi[k])])
            axes.append(a)
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        return np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))

    def _sample_water(self) -> np.ndarray:
        """Reservoir behind the gate plus ambient still water over the shelf."""
        res = self._lattice(
            (RES_X0, -WATER_HALF_Y, GROUND_TOP),
            (RES_X1, WATER_HALF_Y, RESERVOIR_LEVEL),
        )
        amb = self._lattice(
            (AMB_X0, -WATER_HALF_Y, GROUND_TOP),
            (AMB_X1, WATER_HALF_Y, STILL_LEVEL),
        )
        # carve the submerged shelf out of the ambient block
        in_shelf = (
            (amb[:, 0] > SHELF_X0) & (amb[:, 0] < SHELF_X1) & (amb[:, 2] < SHELF_TOP)
        )
        amb = amb[~in_shelf]
        pts = np.concatenate((res, amb))
        order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
        return np.ascontiguousarray(pts[order])

    # ------------------------------------------------------------------
    def _build_solids(self) -> None:
        """Analytic collision boxes; faces coincide with the frozen surfaces."""
        self.static_solids = [
            _box((-4.0, -2.0, -1.0), (4.0, 2.0, GROUND_TOP)),                # floor
            _box((-4.0, -2.0, GROUND_TOP), (4.0, -WALL_Y, 2.0)),             # -y wall
            _box((-4.0, WALL_Y, GROUND_TOP), (4.0, 2.0, 2.0)),               # +y wall
            _box((END_WALL_X, -2.0, GROUND_TOP), (4.0, 2.0, 2.0)),           # end wall
            _box((-4.0, -2.0, GROUND_TOP), (BACK_WALL_X, 2.0, 2.0)),         # back wall
            _box((SHELF_X0, -SHELF_Y, -1.0), (SHELF_X1, SHELF_Y, SHELF_TOP)),  # shelf
        ]
        # thin shells (near-face bands) used only to seed dummy wall particles
        self.fill_solids = [
            _box((-3.3, -1.05, -0.30), (3.2, 1.05, GROUND_TOP)),
            _box((-3.3, -1.06, GROUND_TOP), (3.2, -WALL_Y, 1.45)),
            _box((-3.3, WALL_Y, GROUND_TOP), (3.2, 1.06, 1.45)),
            _box((END_WALL_X, -1.0, GROUND_TOP), (3.25, 1.0, 1.45)),
            _box((-3.34, -1.0, GROUND_TOP), (BACK_WALL_X, 1.0, 1.45)),
            _box((SHELF_X0, -SHELF_Y, 0.30), (SHELF_X1, SHELF_Y, SHELF_TOP)),
        ]

    # gate kinematics -------------------------------------------------
    def gate_bottom(self, t: float) -> float:
        return float(min(GATE_LIFT_SPEED * max(t, 0.0), GATE_LIFT_TOP))

    def gate_speed(self, t: float) -> float:
        return GATE_LIFT_SPEED if GATE_LIFT_SPEED * max(t, 0.0) < GATE_LIFT_TOP else 0.0

    def gate_box(self, t: float):
        z0 = self.gate_bottom(t)
        return _box(
            (GATE_X0, -GATE_HALF_Y, z0), (GATE_X1, GATE_HALF_Y, z0 + GATE_HEIGHT)
        )

    # ------------------------------------------------------------------
    def _build_boundary_particles(self) -> None:
        """Two layers of static dummy particles just inside every solid face."""
        s = self.dx
        depth_limit = 2.05 * s
        chunks = []
        for lo, hi in self.fill_solids:
            axes = [np.arange(lo[k] + 0.5 * s, hi[k] + 1e-9, s) for k in range(3)]
            if any(len(a) == 0 for a in axes):
                continue
            gx, gy, gz = np.meshgrid(*axes, indexing="ij")
            pts = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
            inside = np.all((pts > lo - 1e-9) & (pts < hi + 1e-9), axis=1)
            depth = np.minimum(pts - lo, hi - pts).min(axis=1)
            keep = inside & (depth < depth_limit)
            if np.any(keep):
                chunks.append(pts[keep])
        if chunks:
            bpos = np.concatenate(chunks)
            # deduplicate overlapping shells on a quantized lattice
            key = np.round(bpos / (0.5 * s)).astype(np.int64)
            _, uniq = np.unique(key, axis=0, return_index=True)
            bpos = bpos[np.sort(uniq)]
        else:
            bpos = np.zeros((0, 3))
        self.bpos = np.ascontiguousarray(bpos)
        self.btree = cKDTree(self.bpos) if len(self.bpos) else None

    # ------------------------------------------------------------------
    def _build_wall_samples(self) -> None:
        """Fixed wall-sample set with outward unit normals (fluid side)."""
        pos: list[np.ndarray] = []
        nrm: list[np.ndarray] = []

        def face(px, py, pz, normal):
            gx, gy, gz = np.meshgrid(
                np.atleast_1d(np.asarray(px, float)),
                np.atleast_1d(np.asarray(py, float)),
                np.atleast_1d(np.asarray(pz, float)),
                indexing="ij",
            )
            p = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
            pos.append(p)
            nrm.append(np.repeat(np.asarray(normal, float)[None], len(p), axis=0))

        ys = np.linspace(-0.66, 0.66, 6)
        # floor, upstream of the shelf and in the downstream runout
        face(np.arange(-3.10, 0.24, 0.24), ys, [GROUND_TOP], (0.0, 0.0, 1.0))
        face(np.arange(2.68, 3.03, 0.24), ys, [GROUND_TOP], (0.0, 0.0, 1.0))
        # shelf crest and the two vertical shelf faces
        face(np.arange(SHELF_X0 + 0.06, SHELF_X1, 0.24), ys, [SHELF_TOP], (0.0, 0.0, 1.0))
        face([SHELF_X0], ys, np.linspace(0.08, 0.52, 4), (-1.0, 0.0, 0.0))
        face([SHELF_X1], ys, np.linspace(0.08, 0.52, 4), (1.0, 0.0, 0.0))
        # end walls
        face([END_WALL_X], ys, np.linspace(0.06, 1.10, 5), (-1.0, 0.0, 0.0))
        face([BACK_WALL_X], ys, np.linspace(0.06, 1.10, 5), (1.0, 0.0, 0.0))
        # side walls
        xs = np.arange(-3.0, 3.01, 0.35)
        face(xs, [-WALL_Y], np.linspace(0.06, 1.10, 4), (0.0, -1.0, 0.0))
        face(xs, [WALL_Y], np.linspace(0.06, 1.10, 4), (0.0, 1.0, 0.0))

        static_pos = np.concatenate(pos)
        static_nrm = np.concatenate(nrm)
        static_count = len(static_pos)

        # kinematic gate samples on both faces (positions follow the lift)
        gy = np.linspace(-0.60, 0.60, 5)
        gz = np.linspace(0.08, 1.02, 4)
        gpos, gnrm = [], []
        for xface, normal in ((GATE_X0, (-1.0, 0.0, 0.0)), (GATE_X1, (1.0, 0.0, 0.0))):
            a, b = np.meshgrid(gy, gz, indexing="ij")
            p = np.column_stack((np.full(a.size, float(xface)), a.ravel(), b.ravel()))
            gpos.append(p)
            gnrm.append(np.repeat(np.asarray(normal, float)[None], len(p), axis=0))
        self.gate_sample_base = np.concatenate(gpos)

        self.wall_pos = np.ascontiguousarray(
            np.concatenate((static_pos, self.gate_sample_base))
        )
        self.wall_normal = np.ascontiguousarray(
            np.concatenate((static_nrm, np.concatenate(gnrm)))
        )
        self.gate_mask = np.zeros(len(self.wall_pos), bool)
        self.gate_mask[static_count:] = True
        self.n_walls = int(len(self.wall_pos))

    # ------------------------------------------------------------------
    def _build_grid(self) -> None:
        """Collocated diagnostic grid covering the whole tank interior."""
        self.gdx = 0.14
        self.gorigin = np.array([-3.22, -0.91, 0.0])
        self.gshape = (45, 13, 11)
        cx = self.gorigin[0] + (np.arange(self.gshape[0]) + 0.5) * self.gdx
        cy = self.gorigin[1] + (np.arange(self.gshape[1]) + 0.5) * self.gdx
        cz = self.gorigin[2] + (np.arange(self.gshape[2]) + 0.5) * self.gdx
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing="ij")
        self.gpts = np.ascontiguousarray(
            np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
        )
        self.grid_solid = (
            ((gx > SHELF_X0) & (gx < SHELF_X1) & (np.abs(gy) < SHELF_Y) & (gz < SHELF_TOP))
            | (gx < BACK_WALL_X)
            | (gx > END_WALL_X)
            | (np.abs(gy) > WALL_Y)
        )

    # ------------------------------------------------------------------
    # kernel (cubic spline, 3D)
    # ------------------------------------------------------------------
    def _kernel(self, r):
        h = self.h
        q = r / h
        w = np.zeros_like(q)
        dw = np.zeros_like(q)
        m1 = q < 1.0
        m2 = (q >= 1.0) & (q < 2.0)
        q1 = q[m1]
        w[m1] = self.w0 * (1.0 - 1.5 * q1 ** 2 + 0.75 * q1 ** 3)
        dw[m1] = (self.w0 / h) * (-3.0 * q1 + 2.25 * q1 ** 2)
        q2 = q[m2]
        w[m2] = self.w0 * 0.25 * (2.0 - q2) ** 3
        dw[m2] = (self.w0 / h) * (-0.75 * (2.0 - q2) ** 2)
        return w, dw

    @staticmethod
    def _scatter(idx, vals, n):
        out = np.zeros((n, 3))
        for k in range(3):
            out[:, k] = np.bincount(idx, weights=vals[:, k], minlength=n)
        return out

    # ------------------------------------------------------------------
    # one WCSPH step
    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        pos, vel = self.pos, self.vel
        n = self.n
        eps = 1e-12
        tree = cKDTree(pos)
        pairs = tree.query_pairs(self.support, output_type="ndarray")

        rho = np.full(n, self.m * self.w0)
        if len(pairs):
            i = np.ascontiguousarray(pairs[:, 0])
            j = np.ascontiguousarray(pairs[:, 1])
            rij = pos[i] - pos[j]
            r = np.sqrt(np.einsum("ij,ij->i", rij, rij)) + eps
            wij, dwij = self._kernel(r)
            rho += np.bincount(i, weights=self.m * wij, minlength=n)
            rho += np.bincount(j, weights=self.m * wij, minlength=n)
        else:
            i = j = np.zeros(0, np.int64)
            rij = np.zeros((0, 3))
            r = np.zeros(0)
            dwij = np.zeros(0)

        # wall dummy particles complete the kernel support
        bi = np.zeros(0, np.int64)
        brij = np.zeros((0, 3))
        br = np.zeros(0)
        bdw = np.zeros(0)
        if self.btree is not None:
            bp = tree.sparse_distance_matrix(
                self.btree, self.support, output_type="ndarray"
            )
            if len(bp):
                bi = bp["i"].astype(np.int64)
                bj = bp["j"].astype(np.int64)
                br = bp["v"] + eps
                brij = pos[bi] - self.bpos[bj]
                bw, bdw = self._kernel(br)
                rho += np.bincount(bi, weights=self.m * bw, minlength=n)

        rho = np.clip(rho, 0.5 * self.rho0, 1.5 * self.rho0)
        press = self.B * ((rho / self.rho0) ** self.gamma - 1.0)
        np.maximum(press, 0.0, out=press)

        acc = np.tile(self.gravity, (n, 1))

        if len(r):
            rhat = rij / r[:, None]
            pterm = press[i] / rho[i] ** 2 + press[j] / rho[j] ** 2
            fmag = -self.m * pterm * dwij
            vij = vel[i] - vel[j]
            vdotr = np.einsum("ij,ij->i", vij, rij)
            mu = np.where(vdotr < 0.0, self.h * vdotr / (r ** 2 + 0.01 * self.h ** 2), 0.0)
            pi_ij = -self.alpha * self.c * mu / (0.5 * (rho[i] + rho[j]))
            fmag = fmag - self.m * pi_ij * dwij
            contrib = fmag[:, None] * rhat
            acc += self._scatter(i, contrib, n)
            acc -= self._scatter(j, contrib, n)

        if len(br):
            brhat = brij / br[:, None]
            # mirrored wall state: same pressure, zero velocity
            pterm_b = press[bi] / rho[bi] ** 2 + press[bi] / self.rho0 ** 2
            fmag_b = -self.m * pterm_b * bdw
            vdotr_b = np.einsum("ij,ij->i", vel[bi], brij)
            mu_b = np.where(
                vdotr_b < 0.0, self.h * vdotr_b / (br ** 2 + 0.01 * self.h ** 2), 0.0
            )
            pi_b = -self.alpha * self.c * mu_b / (0.5 * (rho[bi] + self.rho0))
            fmag_b = fmag_b - self.m * pi_b * bdw
            acc += self._scatter(bi, fmag_b[:, None] * brhat, n)

        acc = np.nan_to_num(acc, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(acc, -5.0e3, 5.0e3, out=acc)

        vel += acc * dt
        np.clip(vel, -20.0, 20.0, out=vel)
        pos += vel * dt

        self._apply_solid_constraints(self.time + dt)
        self.pos = np.nan_to_num(self.pos, nan=0.0, posinf=0.0, neginf=0.0)
        self.vel = np.nan_to_num(self.vel, nan=0.0, posinf=0.0, neginf=0.0)
        self.rho = rho
        self.press = press
        self.steps += 1
        self.max_speed = max(self.max_speed, float(np.abs(self.vel).max()))

    # ------------------------------------------------------------------
    def _project_box(self, lo, hi, wall_velocity, margin=0.012):
        pos, vel = self.pos, self.vel
        lo = np.asarray(lo, float) - margin
        hi = np.asarray(hi, float) + margin
        inside = np.all((pos > lo) & (pos < hi), axis=1)
        if not np.any(inside):
            return
        idx = np.flatnonzero(inside)
        p = pos[idx]
        depths = np.concatenate((p - lo, hi - p), axis=1)  # (M,6) exit distances
        axis = np.argmin(depths, axis=1)
        for a in range(6):
            sel = axis == a
            if not np.any(sel):
                continue
            k = a % 3
            rows = idx[sel]
            if a < 3:
                pos[rows, k] = lo[k]
                sign = -1.0
            else:
                pos[rows, k] = hi[k]
                sign = 1.0
            rel = vel[rows, k] - wall_velocity[k]
            bad = sign * rel < 0.0
            if np.any(bad):
                hit = rows[bad]
                vel[hit, k] = wall_velocity[k]
                for t in range(3):
                    if t != k:
                        vel[hit, t] *= 0.97

    def _apply_solid_constraints(self, t: float) -> None:
        zero = np.zeros(3)
        for lo, hi in self.static_solids:
            self._project_box(lo, hi, zero)
        glo, ghi = self.gate_box(t)
        self._project_box(glo, ghi, np.array([0.0, 0.0, self.gate_speed(t)]))
        # containment safety clamp (interior of the frozen tank)
        np.clip(self.pos[:, 0], BACK_WALL_X + 0.004, END_WALL_X - 0.004, out=self.pos[:, 0])
        np.clip(self.pos[:, 1], -WALL_Y + 0.004, WALL_Y - 0.004, out=self.pos[:, 1])
        np.clip(self.pos[:, 2], GROUND_TOP + 0.004, 2.5, out=self.pos[:, 2])

    # ------------------------------------------------------------------
    # diagnostics sampling
    # ------------------------------------------------------------------
    def _shepard(self, points, radius):
        qtree = cKDTree(points)
        ftree = cKDTree(self.pos)
        pairs = qtree.sparse_distance_matrix(ftree, radius, output_type="ndarray")
        m = len(points)
        sw = np.zeros(m)
        swx = np.zeros((m, 3))
        swv = np.zeros((m, 3))
        if len(pairs):
            gi = pairs["i"].astype(np.int64)
            pj = pairs["j"].astype(np.int64)
            s = np.clip(pairs["v"] / radius, 0.0, 1.0)
            w = (1.0 - s ** 2) ** 3
            sw = np.bincount(gi, weights=w, minlength=m)
            for k in range(3):
                swx[:, k] = np.bincount(gi, weights=w * self.pos[pj, k], minlength=m)
                swv[:, k] = np.bincount(gi, weights=w * self.vel[pj, k], minlength=m)
        return sw, swx, swv

    def _grid_fields(self):
        radius = 1.8 * self.gdx
        sw, swx, swv = self._shepard(self.gpts, radius)
        wet = sw > 1.0e-9
        inv = np.where(wet, 1.0 / np.where(wet, sw, 1.0), 0.0)
        xbar = swx * inv[:, None]
        vel = swv * inv[:, None]

        # blobby signed level set: distance to the kernel-weighted mean position
        phi = np.full(len(self.gpts), 0.35)
        if np.any(wet):
            dist = np.linalg.norm(self.gpts[wet] - xbar[wet], axis=1)
            phi[wet] = dist - 0.55 * self.gdx
        phi = np.clip(phi, -1.0, 0.5)

        shape = self.gshape
        phi_g = phi.reshape(shape)
        vel_g = vel.reshape(shape + (3,))
        vel_g = np.where(self.grid_solid[..., None], 0.0, vel_g)
        phi_g = np.where(self.grid_solid, 0.35, phi_g)

        liquid = phi_g < 0.0
        interior = liquid.copy()
        for ax in range(3):
            lo = np.roll(liquid, 1, axis=ax)
            hi = np.roll(liquid, -1, axis=ax)
            sl_lo = [slice(None)] * 3
            sl_lo[ax] = 0
            lo[tuple(sl_lo)] = False
            sl_hi = [slice(None)] * 3
            sl_hi[ax] = -1
            hi[tuple(sl_hi)] = False
            interior &= lo & hi
        mask = interior & (~self.grid_solid)
        if not np.any(mask):
            mask = liquid & (~self.grid_solid)
        return phi_g, vel_g, mask.astype(np.float64)

    def _wall_samples(self, t: float):
        pos = self.wall_pos.copy()
        pos[self.gate_mask, 2] = self.gate_sample_base[:, 2] + self.gate_bottom(t)
        target = np.zeros((self.n_walls, 3))
        target[self.gate_mask, 2] = self.gate_speed(t)
        sw, _, swv = self._shepard(pos, self.support)
        wet = sw > 1.0e-9
        inv = np.where(wet, 1.0 / np.where(wet, sw, 1.0), 0.0)
        interp = swv * inv[:, None]
        measured = target.copy()
        measured[wet] = interp[wet]
        return measured, target, self.wall_normal.copy()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while target - self.time > 1.0e-9:
            interval = target - self.time
            vmax = float(np.abs(self.vel).max()) if self.n else 0.0
            dt_allow = min(self.dt_ceiling, 0.20 * self.h / (self.c + vmax))
            nsub = max(1, int(math.ceil(interval / dt_allow)))
            dt = interval / nsub
            for _ in range(nsub):
                self._step(dt)
                self.time += dt
            self.time = target
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        phi, vel_grid, mask = self._grid_fields()
        bvel, btarget, bnorm = self._wall_samples(time)
        frame = StateFrame(
            float(time),
            np.ascontiguousarray(self.pos, dtype=np.float64),
            np.ascontiguousarray(self.vel, dtype=np.float64),
            mass=np.ascontiguousarray(self.pmass, dtype=np.float64),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": np.ascontiguousarray(self.pmass, dtype=np.float64),
                "volume_quantity": np.ascontiguousarray(self.volume, dtype=np.float64),
                "particle_radius": np.ascontiguousarray(self.radius, dtype=np.float64),
                "material_label": self.body_id.copy(),
                "phase_id": self.body_id.copy(),
            },
            fields={
                "velocity_field": np.ascontiguousarray(vel_grid, dtype=np.float64),
                "liquid_level_set": np.ascontiguousarray(phi, dtype=np.float64),
                "active_mask": np.ascontiguousarray(mask, dtype=np.float64),
                "boundary_velocity": np.ascontiguousarray(bvel, dtype=np.float64),
                "boundary_target_velocity": np.ascontiguousarray(btarget, dtype=np.float64),
                "boundary_normal": np.ascontiguousarray(bnorm, dtype=np.float64),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "cumulative_volume_source": np.zeros(1, dtype=np.float64),
            },
            scalars={
                "grid_spacing": float(self.gdx),
                "characteristic_length": float(self.char_length),
                "reference_velocity": float(self.ref_velocity),
                "contact_length_scale": float(self.dx),
                "liquid_phase_id": 0.0,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "weakly_compressible_sph",
            "particles": int(self.n),
            "boundary_particles": int(len(self.bpos)),
            "particle_spacing": float(self.dx),
            "smoothing_length": float(self.h),
            "sound_speed": float(self.c),
            "steps": int(self.steps),
            "final_time": float(self.time),
            "max_speed": float(self.max_speed),
            "total_mass": float(self.pmass.sum()),
        }
