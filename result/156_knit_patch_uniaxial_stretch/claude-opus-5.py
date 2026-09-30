"""Yarn-constrained knit shell solver: uniaxial stretch of a knitted patch.

Physical model
--------------
* Lagrangian mass-spring shell on the frozen 17x17 material grid (289 samples,
  0.32 kg).  Anisotropic course/wale structural springs with strain locking,
  soft shear diagonals and two-apart bending springs give the thin-sheet
  response (sag, wrinkles, folds).
* Per-cell one-sided yarn-length constraint ``lx + gamma * ly <= L0`` models the
  nearly inextensible yarn inside a knitted loop: once the initial loop slack is
  consumed a cell can only extend along the pulling direction by contracting
  transversely.  This reproduces loop rotation / tightening, the J-shaped
  stiffening and the strong transverse contraction (necking) of a knit patch
  under uniaxial extension.
* Boundary contact with the frozen ground box and the two grip boxes uses
  penalty normal forces, Coulomb friction and a positional non-penetration
  projection.  Clamped edge samples follow their grip kinematically.
* Time stepping: semi-implicit (symplectic) Euler with a fixed substep well
  inside the explicit stability limit of the stiffest constraint.  Fully
  deterministic (one seeded imperfection field, no runtime randomness).
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _smootherstep(u: float) -> float:
    """C2 ramp on [0, 1] with zero first and second derivative at both ends."""
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def _smootherstep_rate(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 30.0 * u * u * (u - 1.0) * (u - 1.0)


class BenchmarkSolver:
    # --- knit constitutive parameters (SI) -------------------------------
    K_WARP = 90.0        # course direction (x) structural stiffness [N/m]
    K_WEFT = 70.0        # wale direction (y) structural stiffness [N/m]
    K_SHEAR = 12.0       # diagonal shear springs [N/m]
    K_BEND = 3.0         # two-apart bending springs [N/m]
    LOCK_STRAIN = 0.22   # yarn straightens out -> stiff locking beyond this
    K_LOCK = 2600.0      # post-lock tangent stiffness [N/m]
    C_STRUCT = 0.35      # structural spring damping [N s/m]
    C_SHEAR = 0.08
    C_BEND = 0.03
    K_YARN = 5000.0      # per-cell yarn-length constraint stiffness [N/m]
    C_YARN = 1.2
    YARN_GAMMA = 1.35    # transverse yarn share per unit course length
    YARN_SLACK = 0.05    # relaxed loop slack (fraction of rest yarn length)
    AIR_DRAG = 0.02      # linear aerodynamic drag [N s/m per sample]
    K_CONTACT = 4000.0   # boundary penalty stiffness [N/m]
    C_CONTACT = 3.0
    MU_FRICTION = 0.45
    DT_MAX = 1.25e-4     # substep [s]
    V_MAX = 20.0         # velocity clamp (numerical safeguard) [m/s]

    # --- prescribed grip schedule: settle, pull, hold, unload, hold ------
    GRIP_SEGMENTS = (
        (0.00, 0.50, 0.00, 0.00),
        (0.50, 3.50, 0.00, 0.95),
        (3.50, 4.60, 0.95, 0.95),
        (4.60, 5.80, 0.95, 0.40),
        (5.80, 1.0e9, 0.40, 0.40),
    )

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        config = dict(scene.config)
        cloth = config["cloth"]
        self.nx, self.ny = (int(v) for v in cloth["grid_shape"])
        self.thickness = float(cloth["thickness"])
        self.gravity = np.asarray(config.get("gravity", (0.0, 0.0, -9.81)), dtype=float)

        frame0 = scene.initial_state()
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.n = len(self.position)
        self.velocity = np.zeros_like(self.position)
        total_mass = float(cloth["total_mass"])
        self.mass = np.full(self.n, total_mass / self.n)
        self.inv_mass = 1.0 / self.mass

        sx, sy = (float(v) for v in cloth["size"])
        self.hx = sx / (self.nx - 1)
        self.hy = sy / (self.ny - 1)

        self._build_topology()
        self._build_boundaries(config)

        # single deterministic out-of-plane imperfection field (seeds buckling)
        rng = np.random.default_rng(int(seed))
        bump = 1.0e-4 * (2.0 * rng.random(self.n) - 1.0)
        bump[self.pin_index] = 0.0
        self.position[:, 2] += bump

        self.time = 0.0
        self._last_internal = np.zeros_like(self.position)
        self._steps = 0

    # -------------------------------------------------------------- topology
    def _build_topology(self) -> None:
        nx, ny = self.nx, self.ny
        grid = np.arange(nx * ny).reshape(ny, nx)  # row-major, x fastest
        rest = self.position

        i_list, j_list, k_list, c_list, lock_list = [], [], [], [], []

        def add(a, b, stiffness, damping, lockable):
            a = np.asarray(a).ravel()
            b = np.asarray(b).ravel()
            i_list.append(a)
            j_list.append(b)
            k_list.append(np.full(a.size, stiffness))
            c_list.append(np.full(a.size, damping))
            lock_list.append(np.full(a.size, lockable, dtype=bool))

        add(grid[:, :-1], grid[:, 1:], self.K_WARP, self.C_STRUCT, True)   # x
        add(grid[:-1, :], grid[1:, :], self.K_WEFT, self.C_STRUCT, True)   # y
        add(grid[:-1, :-1], grid[1:, 1:], self.K_SHEAR, self.C_SHEAR, False)
        add(grid[:-1, 1:], grid[1:, :-1], self.K_SHEAR, self.C_SHEAR, False)
        if nx > 2:
            add(grid[:, :-2], grid[:, 2:], self.K_BEND, self.C_BEND, False)
        if ny > 2:
            add(grid[:-2, :], grid[2:, :], self.K_BEND, self.C_BEND, False)

        self.s_i = np.concatenate(i_list)
        self.s_j = np.concatenate(j_list)
        self.s_k = np.concatenate(k_list)
        self.s_c = np.concatenate(c_list)
        self.s_lock = np.concatenate(lock_list)
        self.s_rest = np.linalg.norm(rest[self.s_j] - rest[self.s_i], axis=1)
        self.s_rest = np.maximum(self.s_rest, 1.0e-9)

        # yarn cells: four edges per quad (two along x, two along y)
        c00 = grid[:-1, :-1].ravel()
        c10 = grid[:-1, 1:].ravel()
        c01 = grid[1:, :-1].ravel()
        c11 = grid[1:, 1:].ravel()
        self.yx_a = np.concatenate([c00, c01])   # bottom + top edges (along x)
        self.yx_b = np.concatenate([c10, c11])
        self.yy_a = np.concatenate([c00, c10])   # left + right edges (along y)
        self.yy_b = np.concatenate([c01, c11])
        self.n_cells = c00.size
        lx0 = 0.5 * (np.linalg.norm(rest[c10] - rest[c00], axis=1)
                     + np.linalg.norm(rest[c11] - rest[c01], axis=1))
        ly0 = 0.5 * (np.linalg.norm(rest[c01] - rest[c00], axis=1)
                     + np.linalg.norm(rest[c11] - rest[c10], axis=1))
        self.yarn_budget = (1.0 + self.YARN_SLACK) * (lx0 + self.YARN_GAMMA * ly0)

    # ------------------------------------------------------- grips / colliders
    def _build_boundaries(self, config: dict) -> None:
        geometry = config.get("scene_geometry") or {}
        boxes = []
        for item in geometry.get("objects", []) or []:
            geom = item.get("geometry", {}) or {}
            if geom.get("type") != "box" or not item.get("collision_enabled", False):
                continue
            half = np.asarray(geom.get("half_extents", (0.0, 0.0, 0.0)), dtype=float)
            transform = item.get("transform", {}) or {}
            center = np.asarray(transform.get("translation", (0.0, 0.0, 0.0)), dtype=float)
            boxes.append({
                "id": str(item.get("id", "box")),
                "half": half,
                "center": center,
                "kinematic": str(item.get("motion", "static")) == "kinematic",
            })
        if not boxes:  # frozen-scene fallback
            boxes = [
                {"id": "ground", "half": np.array([2.4, 2.0, 0.05]),
                 "center": np.array([0.0, 0.0, -0.05]), "kinematic": False},
                {"id": "grip_left", "half": np.array([0.12, 1.25, 0.12]),
                 "center": np.array([-1.5, 0.0, 2.15]), "kinematic": True},
                {"id": "grip_right", "half": np.array([0.12, 1.25, 0.12]),
                 "center": np.array([1.5, 0.0, 2.15]), "kinematic": True},
            ]
        self.boxes = boxes

        grips = [b for b in boxes if b["kinematic"]]
        if len(grips) >= 2:
            grips.sort(key=lambda b: float(b["center"][0]))
            fixed_grip, moving_grip = grips[0], grips[-1]
        else:
            fixed_grip = {"half": np.array([0.12, 1.25, 0.12]),
                          "center": np.array([-1.5, 0.0, 2.15])}
            moving_grip = {"half": np.array([0.12, 1.25, 0.12]),
                           "center": np.array([1.5, 0.0, 2.15])}
        self.moving_grip = moving_grip
        self.moving_center0 = np.asarray(moving_grip["center"], dtype=float).copy()
        self.moving_box_index = next(
            (idx for idx, b in enumerate(boxes) if b is moving_grip), None
        )

        pos = self.position
        tol = 1.0e-6
        pin_fixed = self._inside(pos, fixed_grip, tol)
        pin_move = self._inside(pos, moving_grip, tol)
        if not pin_fixed.any():
            pin_fixed = np.isclose(pos[:, 0], pos[:, 0].min())
        if not pin_move.any():
            pin_move = np.isclose(pos[:, 0], pos[:, 0].max())
        pin_move = pin_move & ~pin_fixed

        self.pin_fixed_mask = pin_fixed
        self.pin_move_mask = pin_move
        pin_mask = pin_fixed | pin_move
        self.pin_mask = pin_mask
        self.pin_index = np.flatnonzero(pin_mask)
        self.pin_move_index = np.flatnonzero(pin_move)
        self.free_index = np.flatnonzero(~pin_mask)
        self.pin_rest = pos[self.pin_index].copy()
        self.pin_move_rest = pos[self.pin_move_index].copy()
        self.pin_is_moving = pin_move[self.pin_index]

    @staticmethod
    def _inside(points: np.ndarray, box: dict, tol: float) -> np.ndarray:
        half = np.asarray(box["half"], dtype=float) + tol
        local = np.abs(points - np.asarray(box["center"], dtype=float))
        return np.all(local <= half, axis=1)

    # ------------------------------------------------------- grip kinematics
    def _grip_state(self, t: float) -> tuple[float, float]:
        """Return (displacement, velocity) of the moving grip along +x."""
        for t0, t1, d0, d1 in self.GRIP_SEGMENTS:
            if t < t1 or t1 >= 1.0e8:
                span = t1 - t0
                if span <= 0.0 or span >= 1.0e8:
                    return float(d1), 0.0
                u = (t - t0) / span
                disp = d0 + (d1 - d0) * _smootherstep(u)
                rate = (d1 - d0) * _smootherstep_rate(u) / span
                return float(disp), float(rate)
        return float(self.GRIP_SEGMENTS[-1][3]), 0.0

    def _collider_state(self, t: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
        disp, rate = self._grip_state(t)
        centers, velocities = [], []
        for idx, box in enumerate(self.boxes):
            center = np.asarray(box["center"], dtype=float).copy()
            vel = np.zeros(3)
            if idx == self.moving_box_index:
                center = self.moving_center0 + np.array([disp, 0.0, 0.0])
                vel = np.array([rate, 0.0, 0.0])
            centers.append(center)
            velocities.append(vel)
        return centers, velocities

    # ---------------------------------------------------------- internal force
    def _internal_force(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        force = np.zeros_like(pos)

        d = pos[self.s_j] - pos[self.s_i]
        length = np.linalg.norm(d, axis=1)
        length = np.maximum(length, 1.0e-9)
        direction = d / length[:, None]
        strain = length / self.s_rest - 1.0
        magnitude = self.s_k * strain
        extra = np.where(
            self.s_lock & (strain > self.LOCK_STRAIN),
            self.K_LOCK * (strain - self.LOCK_STRAIN),
            0.0,
        )
        magnitude = magnitude + extra
        rel = np.einsum("ij,ij->i", vel[self.s_j] - vel[self.s_i], direction)
        magnitude = magnitude + self.s_c * rel
        self._scatter(force, self.s_i, self.s_j, magnitude[:, None] * direction)

        # one-sided per-cell yarn-length constraint (loop inextensibility)
        dx = pos[self.yx_b] - pos[self.yx_a]
        dy = pos[self.yy_b] - pos[self.yy_a]
        lx = np.maximum(np.linalg.norm(dx, axis=1), 1.0e-9)
        ly = np.maximum(np.linalg.norm(dy, axis=1), 1.0e-9)
        ux = dx / lx[:, None]
        uy = dy / ly[:, None]
        half = self.n_cells
        lx_cell = 0.5 * (lx[:half] + lx[half:])
        ly_cell = 0.5 * (ly[:half] + ly[half:])
        gap = lx_cell + self.YARN_GAMMA * ly_cell - self.yarn_budget
        vx = np.einsum("ij,ij->i", vel[self.yx_b] - vel[self.yx_a], ux)
        vy = np.einsum("ij,ij->i", vel[self.yy_b] - vel[self.yy_a], uy)
        rate = 0.5 * (vx[:half] + vx[half:]) + self.YARN_GAMMA * 0.5 * (vy[:half] + vy[half:])
        tension = np.where(gap > 0.0, self.K_YARN * gap + self.C_YARN * rate, 0.0)
        tension = np.maximum(tension, 0.0)
        tx = np.concatenate([0.5 * tension, 0.5 * tension])
        ty = np.concatenate([0.5 * self.YARN_GAMMA * tension,
                             0.5 * self.YARN_GAMMA * tension])
        self._scatter(force, self.yx_a, self.yx_b, tx[:, None] * ux)
        self._scatter(force, self.yy_a, self.yy_b, ty[:, None] * uy)
        return force

    def _scatter(self, out: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray,
                 f_ab: np.ndarray) -> None:
        """Accumulate +f on a and -f on b (f_ab pulls a toward b)."""
        idx = np.concatenate([idx_a, idx_b])
        values = np.concatenate([f_ab, -f_ab])
        n = self.n
        for axis in range(3):
            out[:, axis] += np.bincount(idx, weights=values[:, axis], minlength=n)

    # ------------------------------------------------------- boundary contact
    def _box_gap_normal(self, points: np.ndarray, center: np.ndarray,
                        half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local = points - center
        q = np.abs(local) - half
        q_pos = np.maximum(q, 0.0)
        outside = np.linalg.norm(q_pos, axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        distance = outside + inside
        normal = np.where(q > 0.0, np.sign(local) * 1.0, 0.0)
        deep = outside <= 1.0e-12
        if np.any(deep):
            axis = np.argmax(q[deep], axis=1)
            n_deep = np.zeros((int(deep.sum()), 3))
            n_deep[np.arange(n_deep.shape[0]), axis] = np.sign(
                local[deep][np.arange(n_deep.shape[0]), axis] + 1.0e-15
            )
            normal[deep] = n_deep
        norm = np.linalg.norm(normal, axis=1)
        safe = norm > 1.0e-12
        normal = np.where(safe[:, None], normal / np.where(norm[:, None] > 0, norm[:, None], 1.0),
                          np.array([0.0, 0.0, 1.0]))
        return distance, normal

    def _contact(self, pos: np.ndarray, vel: np.ndarray, t: float,
                 project: bool) -> tuple[np.ndarray, np.ndarray]:
        """Penalty + friction boundary contact on the free samples."""
        free = self.free_index
        force = np.zeros_like(pos)
        radius = 0.5 * self.thickness
        gaps = np.full(free.size, np.inf)
        centers, velocities = self._collider_state(t)
        points = pos[free]
        for box, center, box_vel in zip(self.boxes, centers, velocities):
            distance, normal = self._box_gap_normal(points, center, np.asarray(box["half"]))
            gap = distance - radius
            gaps = np.minimum(gaps, gap)
            hit = gap < 0.0
            if not np.any(hit):
                continue
            idx = free[hit]
            n_hit = normal[hit]
            depth = -gap[hit]
            rel_v = vel[idx] - box_vel
            v_n = np.einsum("ij,ij->i", rel_v, n_hit)
            f_n = self.K_CONTACT * depth - self.C_CONTACT * v_n
            f_n = np.maximum(f_n, 0.0)
            v_t = rel_v - v_n[:, None] * n_hit
            speed_t = np.linalg.norm(v_t, axis=1)
            scale = np.where(speed_t > 1.0e-8,
                             -self.MU_FRICTION * f_n / np.maximum(speed_t, 1.0e-8), 0.0)
            scale = np.maximum(scale, -self.MU_FRICTION * f_n / 1.0e-8)
            f_total = f_n[:, None] * n_hit + np.clip(scale, -1.0e6, 0.0)[:, None] * v_t
            np.add.at(force, idx, f_total)
            if project:
                pos[idx] += (depth[:, None] + 1.0e-9) * n_hit
                v_in = np.minimum(np.einsum("ij,ij->i", vel[idx] - box_vel, n_hit), 0.0)
                vel[idx] -= v_in[:, None] * n_hit
        gaps = np.where(np.isfinite(gaps), gaps, 1.0)
        return force, gaps

    def _environment_force(self, pos: np.ndarray, vel: np.ndarray, t: float,
                           project: bool) -> tuple[np.ndarray, np.ndarray]:
        force = self.mass[:, None] * self.gravity[None, :]
        force = force - self.AIR_DRAG * vel
        contact, gaps = self._contact(pos, vel, t, project)
        return force + contact, gaps

    # ------------------------------------------------------------- integration
    def _step(self, dt: float) -> None:
        pos, vel = self.position, self.velocity
        f_int = self._internal_force(pos, vel)
        f_env, _ = self._environment_force(pos, vel, self.time, False)
        self._last_internal = f_int
        acc = (f_int + f_env) * self.inv_mass[:, None]
        vel += dt * acc
        speed = np.linalg.norm(vel, axis=1)
        hot = speed > self.V_MAX
        if np.any(hot):
            vel[hot] *= (self.V_MAX / speed[hot])[:, None]
        vel[self.pin_index] = 0.0
        pos += dt * vel

        t_next = self.time + dt
        disp, rate = self._grip_state(t_next)
        pos[self.pin_index] = self.pin_rest
        vel[self.pin_index] = 0.0
        moving = self.pin_move_index
        if moving.size:
            pos[moving] = self.pin_move_rest + np.array([disp, 0.0, 0.0])
            vel[moving] = np.array([rate, 0.0, 0.0])

        self._environment_force(pos, vel, t_next, True)
        self.time = t_next
        self._steps += 1

        if not (np.isfinite(pos).all() and np.isfinite(vel).all()):
            pos[:] = np.nan_to_num(pos, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
            vel[:] = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(pos, -50.0, 50.0, out=pos)
        np.clip(vel, -self.V_MAX, self.V_MAX, out=vel)

    # ------------------------------------------------------ output quantities
    def _deformation_gradient(self) -> np.ndarray:
        nx, ny = self.nx, self.ny
        p = self.position.reshape(ny, nx, 3)
        dpdx = np.gradient(p, self.hx, axis=1)
        dpdy = np.gradient(p, self.hy, axis=0)
        a = dpdx.reshape(-1, 3)
        b = dpdy.reshape(-1, 3)
        na = np.linalg.norm(a, axis=1)
        e1 = a / np.maximum(na, 1.0e-12)[:, None]
        b_par = np.einsum("ij,ij->i", b, e1)
        b_perp = b - b_par[:, None] * e1
        nb = np.linalg.norm(b_perp, axis=1)
        f = np.zeros((self.n, 2, 2))
        f[:, 0, 0] = na
        f[:, 0, 1] = b_par
        f[:, 1, 1] = np.maximum(nb, 1.0e-9)
        return f

    def _make_frame(self, time: float) -> StateFrame:
        pos = self.position.copy()
        vel = self.velocity.copy()
        f_int = self._internal_force(pos, vel)
        f_env, gaps = self._environment_force(pos, vel, self.time, False)

        # external force: environmental loads plus grip reaction on clamped
        # samples (m * a_prescribed - f_internal), excluding internal pairs.
        external = f_env.copy()
        pin = self.pin_index
        if pin.size:
            external[pin] = -f_int[pin]
        moving = self.pin_move_index
        if moving.size:
            eps = 1.0e-4
            _, r_plus = self._grip_state(self.time + eps)
            _, r_minus = self._grip_state(max(self.time - eps, 0.0))
            accel = (r_plus - r_minus) / (2.0 * eps)
            external[moving] += self.mass[moving, None] * np.array([accel, 0.0, 0.0])

        gaps = np.asarray(gaps, dtype=float)
        if gaps.size == 0:
            gaps = np.zeros(1)
        gaps = np.clip(np.nan_to_num(gaps, nan=0.0, posinf=1.0e3, neginf=-1.0e3), -10.0, 1.0e3)

        frame = StateFrame(
            float(time),
            pos,
            vel,
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
                "boundary_vertex": self.pin_mask.astype(np.int32),
                "grip_vertex": (self.pin_fixed_mask.astype(np.int32)
                                + 2 * self.pin_move_mask.astype(np.int32)),
            },
            fields={
                "contact_signed_distance": gaps,
                "cumulative_mass_source": np.zeros(1),
                "external_force": external,
            },
            scalars={
                "contact_length_scale": float(self.thickness),
                "grip_displacement": float(self._grip_state(self.time)[0]),
            },
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- interface
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.time:
            remaining = target - self.time
            n_sub = max(int(np.ceil(remaining / self.DT_MAX)), 1)
            dt = remaining / n_sub
            for _ in range(n_sub):
                self._step(dt)
            self.time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        disp, _ = self._grip_state(self.time)
        return {
            "method": "mass_spring_shell_with_one_sided_yarn_length_constraint",
            "integrator": "semi_implicit_euler",
            "substeps": int(self._steps),
            "substep_dt": float(self.DT_MAX),
            "samples": int(self.n),
            "yarn_cells": int(self.n_cells),
            "springs": int(self.s_i.size),
            "pinned_samples": int(self.pin_index.size),
            "final_grip_displacement": float(disp),
            "max_speed": float(np.max(np.linalg.norm(self.velocity, axis=1))),
        }
