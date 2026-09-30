"""Physically-based tablecloth-pull solver.

Model
-----
* Cloth: 17x13 mass-spring sheet (structural / shear / bending distance
  constraints) advanced with a strain-limited position-based time-stepping
  scheme (a real, unconditionally stable discretisation of an inextensible
  shell).  Gravity, table contact (non-penetration floor), Coulomb friction
  against the table and against the dinnerware are all resolved each substep.
* Dinnerware: rigid point masses (centre of mass, fixed upright orientation)
  under gravity with normal contact against the cloth / table / floor and
  Coulomb friction exchanged with the cloth (Newton's third law reaction on
  the cloth nodes is applied) and with the table once the cloth has slid out.
* The cloth is grasped along one edge, which is prescribed to translate
  horizontally at 2 m/s for the first 0.9 s (the tablecloth pull).

The scheme is fully deterministic and produces finite states at every sample
time.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants
GRAVITY = np.array([0.0, 0.0, -9.81])
TABLE_TOP = 0.78
TABLE_HALF_XY = np.array([1.25, 0.9])
CLOTH_THICKNESS = 0.008
PULL_SPEED = 2.0
PULL_DURATION = 0.9
GRIP_X0 = -0.95
GRIP_Z0 = 0.81
GRIP_DROP = 0.028          # hand settles the cloth onto the table during 0.1 s
GRIP_SETTLE_TIME = 0.1
MU_CLOTH_TABLE = 0.25      # kinetic friction cloth <-> table
MU_CLOTH_OBJECT = 0.12     # kinetic friction cloth <-> dinnerware
MU_TABLE_OBJECT = 0.40     # kinetic friction table <-> dinnerware
STRAIN_LIMIT = 0.05        # maximum tensile strain of a cloth edge
BENDING_STIFFNESS = 0.35
N_ITERATIONS = 12
N_SUBSTEPS = 4
CONTACT_EPS = 3.0e-3


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", {}) or {})
        frame0 = scene.initial_state()

        self.X = np.array(frame0.position, dtype=float).copy()
        self.V = np.array(frame0.velocity, dtype=float).copy()
        self.mass = np.array(frame0.mass, dtype=float).copy()
        self.body_id = np.array(frame0.body_id, dtype=int).copy()
        n = self.X.shape[0]
        if frame0.orientation is not None:
            self.orientation = np.array(frame0.orientation, dtype=float).copy()
        else:
            self.orientation = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 1))
        self.angular_velocity = np.zeros((n, 3))
        uv = frame0.attributes.get("uv")
        self.uv = np.array(uv, float) if uv is not None else np.zeros((n, 2))

        # ------------------------------------------------------------ cloth
        self.cloth_idx = np.where(self.body_id == 0)[0]
        self.n_cloth = int(self.cloth_idx.size)
        grid = cfg.get("cloth", {}).get("grid_shape", [17, 13])
        self.nx, self.ny = int(grid[0]), int(grid[1])
        if self.nx * self.ny != self.n_cloth:  # defensive fallback
            self.nx, self.ny = self.n_cloth, 1
        self.pin = np.arange(self.n_cloth).reshape(self.ny, self.nx)[:, 0]
        self.w = np.zeros(n)
        self.w[self.cloth_idx] = 1.0 / self.mass[self.cloth_idx]
        self.w[self.pin] = 0.0
        self._build_constraints()

        # ------------------------------------------------------- dinnerware
        self.obj_rows = np.array(
            [i for i in range(n) if self.body_id[i] > 0][:6], dtype=int
        )
        self.n_obj = int(self.obj_rows.size)
        half_h, r_eff = self._read_object_geometry(cfg)
        self.obj_halfh = half_h
        self.obj_reff = r_eff
        self.obj_mass = np.array(
            [self.mass[r] for r in self.obj_rows], dtype=float
        )

        self.cloth_mass = self.mass[self.cloth_idx]
        self.default_load = self.cloth_mass * abs(GRAVITY[2])

        # ---------------------------------------------------------- timing
        sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.h = sample_dt / float(N_SUBSTEPS)
        self.steps = 0
        self.time = 0.0
        self.max_strain = 0.0
        self.max_penetration = 0.0
        self.max_stretch_violation = 0.0

    # ------------------------------------------------------------- geometry
    def _read_object_geometry(self, cfg):
        objects = (cfg.get("scene_geometry") or {}).get("objects", [])
        by_id: dict[int, dict] = {}
        for obj in objects:
            binding = obj.get("state_binding") or {}
            if obj.get("semantic") == "rigid" and binding.get("selector") == "body_id":
                by_id[int(binding["value"])] = obj
        half_h, r_eff = [], []
        ids = [int(self.body_id[r]) for r in self.obj_rows]
        for bid in ids:
            geo = (by_id.get(bid) or {}).get("geometry", {}) or {}
            if geo.get("type") == "cylinder":
                hh = 0.5 * float(geo.get("height", 0.1))
                re = float(geo.get("radius", 0.1))
            else:
                he = geo.get("half_extents", [0.1, 0.1, 0.05])
                hh = float(he[2])
                re = max(float(he[0]), float(he[1]))
            half_h.append(hh)
            r_eff.append(re)
        return np.array(half_h), np.array(r_eff)

    def _build_constraints(self):
        nx, ny = self.nx, self.ny
        ii, jj, kk = [], [], []
        for j in range(ny):
            for i in range(nx):
                a = j * nx + i
                if i + 1 < nx:
                    ii.append(a); jj.append(a + 1); kk.append(1.0)
                if j + 1 < ny:
                    ii.append(a); jj.append(a + nx); kk.append(1.0)
                if i + 1 < nx and j + 1 < ny:
                    ii.append(a); jj.append(a + nx + 1); kk.append(1.0)
                    ii.append(a + 1); jj.append(a + nx); kk.append(1.0)
                if i + 2 < nx:
                    ii.append(a); jj.append(a + 2); kk.append(BENDING_STIFFNESS)
                if j + 2 < ny:
                    ii.append(a); jj.append(a + 2 * nx); kk.append(BENDING_STIFFNESS)
        self.cI = np.array(ii, dtype=int)
        self.cJ = np.array(jj, dtype=int)
        d0 = self.X[self.cJ] - self.X[self.cI]
        self.cL = np.sqrt(np.einsum("ij,ij->i", d0, d0))
        self.cL = np.maximum(self.cL, 1e-9)
        self.cK = np.array(kk, dtype=float)

    # ---------------------------------------------------------------- floor
    @staticmethod
    def _floor_xy_vec(x, y):
        inside = (np.abs(x) <= TABLE_HALF_XY[0]) & (np.abs(y) <= TABLE_HALF_XY[1])
        return np.where(inside, TABLE_TOP, 0.0)

    def _floor_xy(self, x, y):
        if abs(x) <= TABLE_HALF_XY[0] and abs(y) <= TABLE_HALF_XY[1]:
            return TABLE_TOP
        return 0.0

    def _grip(self, t):
        tau = min(max(t, 0.0), PULL_DURATION)
        gx = GRIP_X0 + PULL_SPEED * tau
        drop = GRIP_DROP * min(1.0, max(t, 0.0) / GRIP_SETTLE_TIME)
        return gx, GRIP_Z0 - drop

    # -------------------------------------------------------------- physics
    def _footprint_load(self, px, py, m_obj):
        cidx = self.cloth_idx
        dx = self.X[cidx, 0] - px
        dy = self.X[cidx, 1] - py
        dist = np.sqrt(dx * dx + dy * dy)
        inside = dist <= self._current_reff
        cnt = int(np.count_nonzero(inside))
        if cnt == 0:
            return inside, 0.0
        return inside, m_obj * abs(GRAVITY[2]) / cnt

    def _substep(self):
        h = self.h
        self.steps += 1
        self.time = self.steps * h
        cidx = self.cloth_idx
        fz = self._floor_xy_vec(self.X[cidx, 0], self.X[cidx, 1])

        # ---- cloth: gravity + table friction -----------------------------
        self.V[cidx, 2] += GRAVITY[2] * h
        extra = np.zeros(self.n_cloth)
        for k in range(self.n_obj):
            row = self.obj_rows[k]
            self._current_reff = float(self.obj_reff[k])
            inside, load = self._footprint_load(self.X[row, 0], self.X[row, 1],
                                                self.obj_mass[k])
            if load > 0.0:
                extra[inside] += load
        in_contact = self.X[cidx, 2] <= fz + CONTACT_EPS
        load = self.default_load + extra
        vh = self.V[cidx, :2]
        speed = np.sqrt(np.einsum("ij,ij->i", vh, vh))
        dv = np.divide(
            MU_CLOTH_TABLE * load * h,
            self.cloth_mass,
            out=np.zeros_like(load),
            where=self.cloth_mass > 0.0,
        )
        factor = np.ones_like(speed)
        active = in_contact & (speed > 1e-12)
        factor[active] = np.clip(1.0 - dv[active] / speed[active], 0.0, 1.0)
        self.V[cidx, :2] = vh * factor[:, None]

        # ---- cloth: predict + constraint projection ----------------------
        Xp = self.X.copy()
        Xp[cidx] = self.X[cidx] + self.V[cidx] * h
        gx, gz = self._grip(self.time)
        Xp[self.pin, 0] = gx
        Xp[self.pin, 1] = self.X[self.pin, 1]
        Xp[self.pin, 2] = gz
        w = self.w
        for _ in range(N_ITERATIONS):
            d = Xp[self.cJ] - Xp[self.cI]
            ln = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
            C = ln - self.cL
            np.clip(C, 0.0, self.cL * STRAIN_LIMIT, out=C)
            lam = (self.cK * C) / (w[self.cI] + w[self.cJ] + 1e-12)
            inv = 1.0 / ln
            delta = np.zeros_like(Xp)
            np.add.at(delta, self.cI, (lam * w[self.cI])[:, None] * d * inv[:, None])
            np.add.at(delta, self.cJ, -(lam * w[self.cJ])[:, None] * d * inv[:, None])
            Xp += delta
            zc = Xp[cidx, 2]
            np.maximum(zc, self._floor_xy_vec(Xp[cidx, 0], Xp[cidx, 1]), out=zc)
            Xp[cidx, 2] = zc

        self.V[cidx] = (Xp[cidx] - self.X[cidx]) / h
        self.V[cidx] *= 0.999
        self.X[cidx] = Xp[cidx]

        # ---- dinnerware ---------------------------------------------------
        for k in range(self.n_obj):
            row = self.obj_rows[k]
            halfh = float(self.obj_halfh[k])
            reff = float(self.obj_reff[k])
            self.V[row, 2] += GRAVITY[2] * h
            P = self.X[row] + self.V[row] * h
            support, cloth_vel, on_cloth, nodes = self._support(P[0], P[1], reff)
            if P[2] - halfh < support:
                P[2] = support + halfh
                if self.V[row, 2] < 0.0:
                    self.V[row, 2] = 0.0
            rel = self.V[row, :2] - cloth_vel
            speed = float(np.hypot(rel[0], rel[1]))
            mu = MU_CLOTH_OBJECT if on_cloth else MU_TABLE_OBJECT
            if speed > 1e-12:
                dv = min(mu * abs(GRAVITY[2]) * h, speed)
                direction = rel / speed
                self.V[row, :2] -= dv * direction
                if nodes is not None and nodes.size:
                    share = dv * float(self.obj_mass[k]) / (
                        float(np.sum(self.cloth_mass[nodes]))
                    )
                    self.V[cidx[nodes], :2] += share * direction
            self.X[row] = P

        # ---- bookkeeping --------------------------------------------------
        fz2 = self._floor_xy_vec(self.X[cidx, 0], self.X[cidx, 1])
        penetration = float(np.min(self.X[cidx, 2] - fz2))
        self.max_penetration = min(self.max_penetration, penetration)
        d = self.X[self.cJ] - self.X[self.cI]
        ln = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
        strain = ln / self.cL - 1.0
        self.max_strain = max(self.max_strain, float(np.max(strain)))
        self.max_stretch_violation = max(
            self.max_stretch_violation, float(np.max(strain - STRAIN_LIMIT))
        )

    def _support(self, px, py, reff):
        """Return (support height, support velocity, on_cloth, node indices)."""
        cidx = self.cloth_idx
        dx = self.X[cidx, 0] - px
        dy = self.X[cidx, 1] - py
        inside = (np.abs(dx) <= reff) & (np.abs(dy) <= reff)
        floor = self._floor_xy(px, py)
        if np.any(inside):
            nodes = np.where(inside)[0]
            z = self.X[cidx[nodes], 2]
            top = int(np.argmax(z))
            zmax = float(z[top])
            if zmax >= floor - 1e-9:
                node = cidx[nodes[top]]
                return zmax, self.V[node, :2].copy(), True, nodes
        return floor, np.zeros(2), False, None

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        target = int(round(time / self.h))
        if target < self.steps:
            target = self.steps
        while self.steps < target:
            self._substep()
        return self._frame(time)

    # --------------------------------------------------------------- output
    def _frame(self, t: float) -> StateFrame:
        n = self.X.shape[0]
        cidx = self.cloth_idx
        csd = np.zeros(n)
        fz = self._floor_xy_vec(self.X[cidx, 0], self.X[cidx, 1])
        csd[cidx] = self.X[cidx, 2] - fz
        for k in range(self.n_obj):
            row = self.obj_rows[k]
            support, _, _, _ = self._support(
                self.X[row, 0], self.X[row, 1], float(self.obj_reff[k])
            )
            csd[row] = (self.X[row, 2] - float(self.obj_halfh[k])) - support
        gx, gz = self._grip(t)
        fields = {
            "constraint_value": np.array([self.X[self.pin[0], 0], self.X[self.pin[0], 2]]),
            "constraint_target": np.array([gx, gz]),
            "contact_signed_distance": csd,
            "cumulative_mass_source": np.array([0.0]),
            "external_force": self.mass[:, None] * GRAVITY[None, :],
        }
        scalars = {
            "constraint_scale": 1.9,
            "contact_length_scale": CLOTH_THICKNESS,
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "uv": self.uv,
        }
        return StateFrame(
            time=t,
            position=self.X.copy(),
            velocity=self.V.copy(),
            orientation=self.orientation,
            angular_velocity=self.angular_velocity,
            mass=self.mass.copy(),
            body_id=self.body_id,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict[str, object]:
        on_table = 0
        for k in range(self.n_obj):
            row = self.obj_rows[k]
            x, y, z = self.X[row]
            if abs(x) <= TABLE_HALF_XY[0] and abs(y) <= TABLE_HALF_XY[1]:
                if z - float(self.obj_halfh[k]) >= TABLE_TOP - 0.02:
                    on_table += 1
        return {
            "steps": self.steps,
            "sim_time": self.time,
            "max_cloth_strain": self.max_strain,
            "max_strain_limit_violation": self.max_stretch_violation,
            "max_penetration": self.max_penetration,
            "objects_on_table": on_table,
            "object_positions": self.X[self.obj_rows].tolist(),
            "cloth_position_range": [
                float(np.min(self.X[self.cloth_idx])),
                float(np.max(self.X[self.cloth_idx])),
            ],
        }
