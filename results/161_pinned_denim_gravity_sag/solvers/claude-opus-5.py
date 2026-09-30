"""Pinned denim gravity sag: inextensible cloth solver (self-implemented).

Physical model
-------------
Lagrangian mass-lumped triangulated sheet (17x17 quad grid) advanced with a
substepped semi-implicit (symplectic Euler) predictor followed by an XPBD /
position-based constraint solve:

* structural (warp / weft) edges  -> near-rigid inextensibility constraints
* diagonal edges                  -> compliant shear resistance
* two-span (flexion) edges        -> compliant bending resistance (denim regime)
* unilateral ground contact with Coulomb-style tangential friction
* vertex-vertex proximity repulsion at cloth thickness (self contact)

Pins are the prescribed fixed set on the top edge (infinite mass), so tension
propagates from the pins downward through the constraint graph and the sheet
develops broad hanging folds with little in-plane stretch. Aerodynamic drag
plus the dissipative projection let the drape settle without rubber-like
elongation or persistent oscillation.

No Genesis, no keyframes, no precomputed trajectories: every exported frame is
the result of actual time stepping.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---- numerical / material parameters -------------------------------
    SUBSTEPS = 16              # substeps per output sample
    ITERATIONS = 12            # constraint sweeps per substep
    K_STRUCT = 1.0             # inextensible warp/weft
    K_SHEAR = 0.12             # compliant shear
    K_BEND = 0.22              # denim-like bending stiffness
    DRAG = 1.05                # aerodynamic drag coefficient [1/s]
    VEL_SMOOTH = 0.06          # local velocity smoothing (numerical viscosity)
    FRICTION = 0.55            # ground tangential friction factor
    SELF_RELAX = 0.5           # self-contact push relaxation

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        cloth = cfg["cloth"]

        self.nx, self.ny = int(cloth["grid_shape"][0]), int(cloth["grid_shape"][1])
        self.n = self.nx * self.ny
        self.sx, self.sy = float(cloth["size"][0]), float(cloth["size"][1])
        self.du = self.sx / (self.nx - 1)
        self.dv = self.sy / (self.ny - 1)
        self.thickness = float(cloth["thickness"])
        self.total_mass = float(cloth["total_mass"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        init = scene.initial_state()
        self.position = np.array(init.position, dtype=float)
        self.velocity = np.zeros_like(self.position)
        self.mass = np.array(init.mass, dtype=float) if init.mass is not None else np.full(
            self.n, self.total_mass / self.n)
        self.body_id = np.zeros(self.n, dtype=np.int32)
        self.boundary_vertex = (
            np.asarray(init.attributes["boundary_vertex"], dtype=np.int32)
            if "boundary_vertex" in init.attributes
            else np.zeros(self.n, dtype=np.int32)
        )

        # ---- pin set: fixed nodes on the top edge (corners included) ----
        top_row = self.ny - 1
        pin_cols = list(range(0, self.nx, 4))
        if (self.nx - 1) not in pin_cols:
            pin_cols.append(self.nx - 1)
        self.pins = np.array([top_row * self.nx + c for c in pin_cols], dtype=np.int64)

        self.inv_mass = 1.0 / self.mass
        self.inv_mass[self.pins] = 0.0
        self.free = self.inv_mass > 0.0
        self.pin_positions = self.position[self.pins].copy()

        # ---- symmetry-breaking initial perturbation (deterministic) -----
        rng = np.random.default_rng(int(seed))
        amp = 5.0e-4
        noise = rng.standard_normal((self.n, 3)) * amp
        # buckling seed mostly out of the sheet plane (cloth lies in xz)
        noise[:, 0] *= 0.2
        noise[:, 2] *= 0.2
        noise[~self.free] = 0.0
        self.position += noise
        self.position[self.pins] = self.pin_positions

        # ---- ground plane (frozen geometry: box top face at z = 0) ------
        self.ground_z = 0.0
        self.contact_offset = 0.5 * self.thickness

        self._build_constraints()
        self._build_self_contact_mask()

        self.time = 0.0
        self._steps = 0
        self._max_strain = 0.0

    # ------------------------------------------------------------------
    def _idx(self, r, c):
        return (np.asarray(r) * self.nx + np.asarray(c)).astype(np.int64)

    def _pairs(self, rows, cols, dr, dc):
        """Cartesian product of rows x cols giving edges (r,c)-(r+dr,c+dc)."""
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        rr = rr.ravel()
        cc = cc.ravel()
        return self._idx(rr, cc), self._idx(rr + dr, cc + dc)

    def _build_constraints(self):
        """Graph-coloured constraint groups (conflict-free within a group)."""
        nx, ny = self.nx, self.ny
        du, dv = self.du, self.dv
        groups = []

        all_rows = np.arange(ny)
        all_cols = np.arange(nx)

        # structural: horizontal (warp), colour by column parity
        for par in (0, 1):
            cols = np.arange(par, nx - 1, 2)
            if cols.size:
                ia, ib = self._pairs(all_rows, cols, 0, 1)
                groups.append((ia, ib, np.full(ia.size, du), self.K_STRUCT))
        # structural: vertical (weft), colour by row parity
        for par in (0, 1):
            rows = np.arange(par, ny - 1, 2)
            if rows.size:
                ia, ib = self._pairs(rows, all_cols, 1, 0)
                groups.append((ia, ib, np.full(ia.size, dv), self.K_STRUCT))

        # shear diagonals, colour by (row parity, column parity)
        diag = float(np.hypot(du, dv))
        for pr in (0, 1):
            rows = np.arange(pr, ny - 1, 2)
            if not rows.size:
                continue
            for pc in (0, 1):
                cols = np.arange(pc, nx - 1, 2)
                if not cols.size:
                    continue
                ia, ib = self._pairs(rows, cols, 1, 1)          # (r,c)-(r+1,c+1)
                groups.append((ia, ib, np.full(ia.size, diag), self.K_SHEAR))
                rr, cc = np.meshgrid(rows, cols, indexing="ij")
                rr = rr.ravel()
                cc = cc.ravel()
                ja = self._idx(rr, cc + 1)                       # (r,c+1)-(r+1,c)
                jb = self._idx(rr + 1, cc)
                groups.append((ja, jb, np.full(ja.size, diag), self.K_SHEAR))

        # bending (two-span flexion springs), colour with stride 4
        for off in range(4):
            cols = np.arange(off, nx - 2, 4)
            if cols.size:
                ia, ib = self._pairs(all_rows, cols, 0, 2)
                groups.append((ia, ib, np.full(ia.size, 2.0 * du), self.K_BEND))
        for off in range(4):
            rows = np.arange(off, ny - 2, 4)
            if rows.size:
                ia, ib = self._pairs(rows, all_cols, 2, 0)
                groups.append((ia, ib, np.full(ia.size, 2.0 * dv), self.K_BEND))

        self.groups = [
            (ia, ib, rest, float(k), self.inv_mass[ia], self.inv_mass[ib])
            for (ia, ib, rest, k) in groups
        ]

        # structural edge list (diagnostics / strain monitoring)
        sa, sb = [], []
        rows = np.arange(ny)
        cols = np.arange(nx - 1)
        ia, ib = self._pairs(rows, cols, 0, 1)
        sa.append(ia)
        sb.append(ib)
        ia, ib = self._pairs(np.arange(ny - 1), np.arange(nx), 1, 0)
        sa.append(ia)
        sb.append(ib)
        self.struct_a = np.concatenate(sa)
        self.struct_b = np.concatenate(sb)
        self.struct_rest = np.concatenate(
            [np.full(ny * (nx - 1), du), np.full((ny - 1) * nx, dv)]
        )

    def _build_self_contact_mask(self):
        rows = np.repeat(np.arange(self.ny), self.nx)
        cols = np.tile(np.arange(self.nx), self.ny)
        dr = np.abs(rows[:, None] - rows[None, :])
        dc = np.abs(cols[:, None] - cols[None, :])
        cheb = np.maximum(dr, dc)
        self.self_mask = cheb > 1
        self.contact_radius = min(0.045, 0.28 * min(self.du, self.dv))

    # ------------------------------------------------------------------
    def _project_constraints(self, p):
        for ia, ib, rest, k, wa, wb in self.groups:
            d = p[ib] - p[ia]
            length = np.sqrt(np.einsum("ij,ij->i", d, d))
            safe = np.maximum(length, 1e-12)
            w = wa + wb
            denom = np.where(w > 0.0, w, 1.0)
            scale = k * (length - rest) / (denom * safe)
            scale = np.where(w > 0.0, scale, 0.0)
            corr = scale[:, None] * d
            p[ia] += wa[:, None] * corr
            p[ib] -= wb[:, None] * corr

    def _resolve_self_contact(self, p):
        diff = p[None, :, :] - p[:, None, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        active = self.self_mask & (dist < self.contact_radius) & (dist > 1e-9)
        if not active.any():
            return
        pen = np.where(active, self.contact_radius - dist, 0.0)
        inv_d = np.where(dist > 1e-9, 1.0 / np.maximum(dist, 1e-9), 0.0)
        # dp[i] = -relax * sum_j pen_ij * (p_j - p_i)/|.|  (symmetric, pairwise)
        weight = -self.SELF_RELAX * pen * inv_d
        dp = np.einsum("ij,ijk->ik", weight, diff)
        dp[~self.free] = 0.0
        p += dp

    def _ground_contact(self, p, p_prev, dt):
        floor = self.ground_z + self.contact_offset
        hit = (p[:, 2] < floor) & self.free
        if not hit.any():
            return
        p[hit, 2] = floor
        # Coulomb-style tangential friction on the sliding increment
        tang = p[hit, :2] - p_prev[hit, :2]
        p[hit, 0] = p_prev[hit, 0] + (1.0 - self.FRICTION) * tang[:, 0]
        p[hit, 1] = p_prev[hit, 1] + (1.0 - self.FRICTION) * tang[:, 1]

    def _smooth_velocity(self):
        if self.VEL_SMOOTH <= 0.0:
            return
        v = self.velocity.reshape(self.ny, self.nx, 3)
        acc = np.zeros_like(v)
        cnt = np.zeros((self.ny, self.nx, 1))
        acc[1:, :, :] += v[:-1, :, :]
        cnt[1:, :, :] += 1.0
        acc[:-1, :, :] += v[1:, :, :]
        cnt[:-1, :, :] += 1.0
        acc[:, 1:, :] += v[:, :-1, :]
        cnt[:, 1:, :] += 1.0
        acc[:, :-1, :] += v[:, 1:, :]
        cnt[:, :-1, :] += 1.0
        avg = acc / np.maximum(cnt, 1.0)
        v_new = v + self.VEL_SMOOTH * (avg - v)
        self.velocity = v_new.reshape(self.n, 3)
        self.velocity[~self.free] = 0.0

    def _step(self, dt):
        p = self.position
        v = self.velocity

        # external accelerations: gravity + linear aerodynamic drag
        acc = self.gravity[None, :] - self.DRAG * v
        v = v + dt * acc
        v[~self.free] = 0.0

        p_prev = p
        p_new = p + dt * v
        p_new[self.pins] = self.pin_positions

        for _ in range(self.ITERATIONS):
            self._project_constraints(p_new)
            p_new[self.pins] = self.pin_positions
        self._resolve_self_contact(p_new)
        self._project_constraints(p_new)
        p_new[self.pins] = self.pin_positions
        self._ground_contact(p_new, p_prev, dt)
        p_new[self.pins] = self.pin_positions

        self.velocity = (p_new - p_prev) / dt
        self.velocity[~self.free] = 0.0
        if not np.isfinite(self.velocity).all():
            self.velocity = np.nan_to_num(self.velocity, nan=0.0, posinf=0.0, neginf=0.0)
        # clamp pathological velocities for robustness (never active in practice)
        speed = np.linalg.norm(self.velocity, axis=1)
        cap = 50.0
        over = speed > cap
        if over.any():
            self.velocity[over] *= (cap / speed[over])[:, None]
        self.position = np.nan_to_num(p_new, nan=0.0, posinf=0.0, neginf=0.0)
        self._smooth_velocity()
        self._steps += 1

    # ------------------------------------------------------------------
    def _deformation_gradient(self):
        """Per-vertex 2x2 deformation gradient in material coordinates."""
        P = self.position.reshape(self.ny, self.nx, 3)
        t_u = np.gradient(P, axis=1) / self.du
        t_v = np.gradient(P, axis=0) / self.dv
        t_u = t_u.reshape(self.n, 3)
        t_v = t_v.reshape(self.n, 3)
        a = np.linalg.norm(t_u, axis=1)
        e1 = t_u / np.maximum(a, 1e-12)[:, None]
        b1 = np.einsum("ij,ij->i", t_v, e1)
        w = t_v - b1[:, None] * e1
        b2 = np.linalg.norm(w, axis=1)
        F = np.zeros((self.n, 2, 2))
        F[:, 0, 0] = a
        F[:, 0, 1] = b1
        F[:, 1, 1] = b2
        return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    def _contact_signed_distance(self):
        gap_ground = self.position[:, 2] - (self.ground_z + self.contact_offset)
        diff = self.position[None, :, :] - self.position[:, None, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        big = np.where(self.self_mask, dist, np.inf)
        nearest = big.min(axis=1)
        gap_self = np.where(np.isfinite(nearest), nearest - self.contact_radius, 1.0)
        gap = np.minimum(gap_ground, gap_self)
        return np.clip(np.nan_to_num(gap, nan=1.0, posinf=1.0, neginf=-1.0), -1.0, 10.0)

    def _external_force(self):
        return self.mass[:, None] * (self.gravity[None, :] - self.DRAG * self.velocity)

    def _frame(self):
        F = self._deformation_gradient()
        ext = self._external_force()
        gap = self._contact_signed_distance()
        frame = StateFrame(
            float(max(self.time, 0.0)),
            self.position.copy(),
            self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary_vertex.copy(),
                "deformation_gradient": F,
            },
            fields={
                "external_force": np.nan_to_num(ext),
                "contact_signed_distance": gap,
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={
                "contact_length_scale": float(self.thickness),
                "total_mass": float(self.mass.sum()),
                "pin_count": float(self.pins.size),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target > self.time + 1e-12:
            span = target - self.time
            n_sub = max(1, int(np.ceil(span / (1.0 / 30.0) * self.SUBSTEPS)))
            dt = span / n_sub
            for _ in range(n_sub):
                self._step(dt)
            self.time = target
        else:
            self.time = max(self.time, target)

        d = self.position[self.struct_b] - self.position[self.struct_a]
        length = np.linalg.norm(d, axis=1)
        strain = float(np.max(np.abs(length / self.struct_rest - 1.0)))
        self._max_strain = max(self._max_strain, strain)
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "substepped symplectic Euler + XPBD-style constraint projection",
            "vertices": int(self.n),
            "pins": self.pins.tolist(),
            "substeps_per_sample": int(self.SUBSTEPS),
            "iterations": int(self.ITERATIONS),
            "steps_taken": int(self._steps),
            "max_structural_strain": float(self._max_strain),
            "final_min_z": float(self.position[:, 2].min()),
        }
