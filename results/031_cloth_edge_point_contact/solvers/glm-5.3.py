"""Physically-based cloth solver for scene 031_cloth_edge_point_contact.

A square cloth sheet falls under gravity onto a straight sharp edge and an
isolated sharp point, drapes over both contacts, and comes to rest without
penetrating any obstacle.

Method
------
Position-based dynamics (PBD) with substepping:

* semi-implicit velocity update with gravity and light aerodynamic drag,
* distance constraints for structural (stretch), shear and bending response,
* hard non-penetration against the frozen obstacles via analytic signed
  distance fields (exact box SDFs for ground/edge, exact circular-cone SDF
  for the sharp point) evaluated with the frozen thickness offset,
* position-level Coulomb friction at contacting nodes (coefficient from the
  frozen configuration),
* per-node deformation gradient reconstructed from the material-space
  neighbourhood of every sample.

The state is advanced by an explicit substep loop; every requested sample
time is reached deterministically and all arrays are finite.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """PBD cloth solver honouring the frozen scene configuration."""

    SUBSTEPS_PER_SAMPLE = 8
    SOLVER_ITERATIONS = 6
    STRETCH_STIFFNESS = 1.0
    SHEAR_STIFFNESS = 0.35
    BEND_STIFFNESS = 0.05
    DRAG_PER_SUBSTEP = 4.0e-4
    MAX_SPEED = 25.0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        init = scene.initial_state()

        self.seed = int(seed)
        self.config = cfg
        self.X0 = np.array(init.position, dtype=float)
        self.x = self.X0.copy()
        self.v = np.array(init.velocity, dtype=float)
        self.uv = np.array(init.attributes["uv"], dtype=float)
        self.n_nodes = len(self.x)

        self.gravity = np.array(cfg["gravity"], dtype=float)
        self.mu = float(cfg["friction"]["coefficient"])

        cloth = cfg["cloth"]
        self.total_mass = float(cloth["total_mass"])
        self.node_mass = np.full(self.n_nodes, self.total_mass / self.n_nodes)
        self.thickness = float(cloth["thickness"])
        self.margin = float(cfg["contact"]["thickness_offset"])

        self.sample_dt = float(cfg["sample_dt"])
        self.h = self.sample_dt / self.SUBSTEPS_PER_SAMPLE
        self.t = 0.0
        self.substeps = 0

        self._build_springs(int(cloth["grid_shape"][0]), int(cloth["grid_shape"][1]))
        self._build_obstacles(cfg)
        self._build_gradient_operator()

    def _build_springs(self, nx: int, ny: int) -> None:
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
        flat = (iy * nx + ix)

        def rest(ia: np.ndarray, ib: np.ndarray):
            length = np.linalg.norm(self.X0[ia] - self.X0[ib], axis=1)
            return ia.astype(np.int64), ib.astype(np.int64), length

        # structural: direct grid neighbours
        m = (ix < nx - 1)
        self.st_ia, self.st_ib, self.st_rest = rest(flat[m], flat[m] + 1)
        m = (iy < ny - 1)
        ia, ib, r = rest(flat[m], flat[m] + nx)
        self.st_ia = np.concatenate([self.st_ia, ia])
        self.st_ib = np.concatenate([self.st_ib, ib])
        self.st_rest = np.concatenate([self.st_rest, r])

        # shear: cell diagonals
        cell = (iy < ny - 1) & (ix < nx - 1)
        a = flat[cell]
        d1_ia, d1_ib, d1_r = rest(a, a + nx + 1)
        d2_ia, d2_ib, d2_r = rest(a + 1, a + nx)
        self.sh_ia = np.concatenate([d1_ia, d2_ia])
        self.sh_ib = np.concatenate([d1_ib, d2_ib])
        self.sh_rest = np.concatenate([d1_r, d2_r])

        # bending: skip-one neighbours
        m = (ix < nx - 2)
        b1_ia, b1_ib, b1_r = rest(flat[m], flat[m] + 2)
        m = (iy < ny - 2)
        b2_ia, b2_ib, b2_r = rest(flat[m], flat[m] + 2 * nx)
        self.bd_ia = np.concatenate([b1_ia, b2_ia])
        self.bd_ib = np.concatenate([b1_ib, b2_ib])
        self.bd_rest = np.concatenate([b1_r, b2_r])

    def _build_obstacles(self, cfg: dict) -> None:
        self.boxes = []
        for key in ("ground", "sharp_edge"):
            spec = cfg[key]
            self.boxes.append(
                (
                    np.asarray(spec["center"], dtype=float),
                    np.asarray(spec["half_extents"], dtype=float),
                )
            )
        point = cfg["sharp_point"]
        self.cone_tip = np.asarray(point["tip"], dtype=float)
        self.cone_h = float(point["height"])
        self.cone_rb = float(point["base_radius"])
        self.cone_slant = float(np.hypot(self.cone_h, self.cone_rb))

    def _build_gradient_operator(self) -> None:
        """Least-squares operator mapping neighbour rest/spatial edges to F."""
        nx = int(self.config["cloth"]["grid_shape"][0])
        ny = int(self.config["cloth"]["grid_shape"][1])
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
        flat = (iy * nx + ix)
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        src: list[int] = []
        dst: list[int] = []
        for dx, dy in offsets:
            jx = ix + dx
            jy = iy + dy
            valid = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
            src.extend(flat[valid].tolist())
            dst.extend(flat[valid] + dy * nx + dx)
        self.fg_i = np.asarray(src, dtype=np.int64)
        self.fg_j = np.asarray(dst, dtype=np.int64)
        dX = self.X0[self.fg_j] - self.X0[self.fg_i]
        self.fg_du = dX[:, 0]
        self.fg_dv = dX[:, 1]
        n = self.n_nodes
        m00 = np.bincount(self.fg_i, weights=self.fg_du ** 2, minlength=n)
        m01 = np.bincount(self.fg_i, weights=self.fg_du * self.fg_dv, minlength=n)
        m11 = np.bincount(self.fg_i, weights=self.fg_dv ** 2, minlength=n)
        det = m00 * m11 - m01 ** 2
        self.fg_det = det + 1e-14
        self.fg_m00, self.fg_m01, self.fg_m11 = m00, m01, m11

    # ------------------------------------------------------------------- SDFs
    def _box_sdf(self, points: np.ndarray):
        """Exact signed distance and outward normal of an axis-aligned box."""
        center, he = self._current_box
        q = points - center
        clamped = np.clip(q, -he, he)
        diff = q - clamped
        dist = np.linalg.norm(diff, axis=1)
        outside = dist > 1e-12
        inner = he - np.abs(q)
        axis = np.argmin(inner, axis=1)
        rows = np.arange(len(q))
        inner_d = inner[rows, axis]
        d = np.where(outside, dist, -inner_d)
        n_out = diff / np.maximum(dist, 1e-12)[:, None]
        n_in = np.zeros_like(points)
        n_in[rows, axis] = np.sign(q[rows, axis])
        n_in[rows, axis] = np.where(n_in[rows, axis] == 0.0, 1.0, n_in[rows, axis])
        normal = np.where(outside[:, None], n_out, n_in)
        return d, normal

    def _cone_sdf(self, points: np.ndarray):
        """Signed distance and outward normal of the sharp cone (tip up)."""
        A, H, rb, L = self.cone_tip, self.cone_h, self.cone_rb, self.cone_slant
        rel = points - A
        s = -rel[:, 2]  # depth below the tip along the axis (down positive)
        xy = rel[:, :2]
        rho = np.hypot(xy[:, 0], xy[:, 1])
        safe_rho = np.maximum(rho, 1e-12)
        ux = np.where(rho > 1e-12, xy[:, 0] / safe_rho, 1.0)
        uy = np.where(rho > 1e-12, xy[:, 1] / safe_rho, 0.0)

        # exterior candidates: apex, base rim, base disk, clamped lateral surface
        d_apex = np.hypot(s, rho)
        d_rim = np.hypot(s - H, rho - rb)
        d_disk = np.where(rho <= rb, np.abs(s - H), np.inf)
        t = np.clip((s * H + rho * rb) / L, 0.0, L)
        s_c = t * H / L
        rho_c = t * rb / L
        d_lat = np.hypot(s - s_c, rho - rho_c)
        d_ext = np.minimum(np.minimum(d_apex, d_rim), np.minimum(d_disk, d_lat))

        # interior exit distances (lateral surface / base)
        s_clip = np.clip(s, 0.0, H)
        radius_at = rb * s_clip / H
        inside = (s >= 0.0) & (s <= H) & (rho <= radius_at)
        exit_lat = (radius_at - rho) * (H / L)
        exit_base = H - s
        d_int = np.minimum(exit_lat, exit_base)
        d = np.where(inside, -d_int, d_ext)

        # outward normals
        n_lat = np.empty_like(points)
        n_lat[:, 0] = ux * (H / L)
        n_lat[:, 1] = uy * (H / L)
        n_lat[:, 2] = rb / L

        n_apex = rel / np.maximum(d_apex, 1e-12)[:, None]
        n_rim = np.empty_like(points)
        n_rim[:, 0] = xy[:, 0] - rb * ux
        n_rim[:, 1] = xy[:, 1] - rb * uy
        n_rim[:, 2] = H - s
        n_rim /= np.maximum(d_rim, 1e-12)[:, None]
        n_disk = np.zeros_like(points)
        n_disk[:, 2] = -1.0

        stack = np.stack([d_apex, d_rim, d_disk, d_lat], axis=1)
        which = np.argmin(stack, axis=1)
        n_ext = np.where(
            (which == 0)[:, None], n_apex,
            np.where((which == 1)[:, None], n_rim,
                     np.where((which == 2)[:, None], n_disk, n_lat)),
        )
        n_int = np.where((exit_base <= exit_lat)[:, None], n_disk, n_lat)
        normal = np.where(inside[:, None], n_int, n_ext)
        return d, normal

    def _obstacle_distances(self, points: np.ndarray):
        """Min signed distance over all obstacles, plus per-obstacle data."""
        gap = np.full(len(points), np.inf)
        contacts = []
        for center, he in self.boxes:
            self._current_box = (center, he)
            d, n = self._box_sdf(points)
            contacts.append((d, n))
            gap = np.minimum(gap, d)
        d, n = self._cone_sdf(points)
        contacts.append((d, n))
        gap = np.minimum(gap, d)
        return gap, contacts

    # ------------------------------------------------------------------ steps
    def _solve_distance(self, ia, ib, rest, stiffness) -> None:
        delta = self.x[ia] - self.x[ib]
        length = np.linalg.norm(delta, axis=1)
        safe = np.maximum(length, 1e-12)
        scale = np.where(length > 1e-12, (length - rest) / safe, 0.0)
        scale *= 0.5 * stiffness
        corr = delta * scale[:, None]
        n = self.n_nodes
        for axis in range(3):
            update = (
                np.bincount(ia, weights=-corr[:, axis], minlength=n)
                + np.bincount(ib, weights=corr[:, axis], minlength=n)
            )
            self.x[:, axis] += update

    def _solve_collisions(self) -> None:
        _, contacts = self._obstacle_distances(self.x)
        for d, n in contacts:
            bad = d < self.margin
            if np.any(bad):
                self.x[bad] += n[bad] * (self.margin - d[bad])[:, None]

    def _apply_friction(self, x_prev: np.ndarray) -> None:
        _, contacts = self._obstacle_distances(self.x)
        for d, n in contacts:
            active = d < self.margin + 5.0e-4
            idx = np.nonzero(active)[0]
            if idx.size == 0:
                continue
            dx = self.x[idx] - x_prev[idx]
            n_k = n[idx]
            dn = np.sum(dx * n_k, axis=1)
            dxt = dx - dn[:, None] * n_k
            dxt_mag = np.linalg.norm(dxt, axis=1)
            limit = self.mu * np.abs(dn)
            scale = np.where(
                dxt_mag <= limit,
                1.0,
                np.where(dxt_mag > 1e-12, limit / np.maximum(dxt_mag, 1e-12), 0.0),
            )
            self.x[idx] -= dxt * scale[:, None]

    def _substep(self, h: float) -> None:
        self.v = self.v + self.gravity[None, :] * h
        self.v *= 1.0 - self.DRAG_PER_SUBSTEP
        speed = np.linalg.norm(self.v, axis=1)
        fast = speed > self.MAX_SPEED
        if np.any(fast):
            self.v[fast] *= (self.MAX_SPEED / speed[fast])[:, None]

        x_prev = self.x.copy()
        self.x = self.x + self.v * h

        for _ in range(self.SOLVER_ITERATIONS):
            self._solve_distance(self.st_ia, self.st_ib, self.st_rest, self.STRETCH_STIFFNESS)
            self._solve_distance(self.sh_ia, self.sh_ib, self.sh_rest, self.SHEAR_STIFFNESS)
            self._solve_distance(self.bd_ia, self.bd_ib, self.bd_rest, self.BEND_STIFFNESS)
            self._solve_collisions()

        self._apply_friction(x_prev)
        self.v = (self.x - x_prev) / h

        bad = ~np.isfinite(self.x).all(axis=1)
        if np.any(bad):
            self.x[bad] = x_prev[bad]
        badv = ~np.isfinite(self.v).all(axis=1)
        if np.any(badv):
            self.v[badv] = 0.0
        self.substeps += 1

    # ----------------------------------------------------------------- frames
    def _deformation_gradient(self) -> np.ndarray:
        dx = self.x[self.fg_j] - self.x[self.fg_i]
        i = self.fg_i
        n = self.n_nodes
        r = []
        for axis in range(3):  # R[:, axis, :] = sum dx_axis * [du, dv]
            r.append(
                np.stack(
                    [
                        np.bincount(i, weights=dx[:, axis] * self.fg_du, minlength=n),
                        np.bincount(i, weights=dx[:, axis] * self.fg_dv, minlength=n),
                    ],
                    axis=1,
                )
            )
        F = np.zeros((n, 3, 3))
        det = self.fg_det
        m00, m01, m11 = self.fg_m00, self.fg_m01, self.fg_m11
        for axis in range(3):
            F[:, axis, 0] = (r[axis][:, 0] * m11 - r[axis][:, 1] * m01) / det
            F[:, axis, 1] = (r[axis][:, 1] * m00 - r[axis][:, 0] * m01) / det
        F[:, 0, 2] = 0.0
        F[:, 1, 2] = 0.0
        F[:, 2, 2] = 1.0
        return F

    def _frame(self, time: float) -> StateFrame:
        gap, _ = self._obstacle_distances(self.x)
        signed_gap = gap - self.margin
        signed_gap = np.where(np.isfinite(signed_gap), signed_gap, 0.0)
        frame = StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            mass=self.node_mass.copy(),
            attributes={
                "uv": self.uv.copy(),
                "mass_quantity": self.node_mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": signed_gap,
                "cumulative_mass_source": np.zeros(1),
            },
            scalars={"contact_length_scale": self.thickness},
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------------- API
    def advance_to(self, time: float):
        time = float(time)
        if time > self.t:
            count = int(np.floor((time - self.t) / self.h + 1.0e-9))
            for _ in range(count):
                self._substep(self.h)
                self.t += self.h
            if self.t < time - 1.0e-9:  # residual sliver (should not happen)
                self._substep(time - self.t)
                self.t = time
        return self._frame(time)

    def diagnostics(self) -> dict[str, object]:
        gap, _ = self._obstacle_distances(self.x)
        return {
            "method": "position-based-dynamics cloth",
            "substeps": int(self.substeps),
            "substep_dt": float(self.h),
            "iterations_per_substep": int(self.SOLVER_ITERATIONS),
            "min_obstacle_gap": float(np.min(gap)) if len(gap) else 0.0,
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))),
            "time": float(self.t),
        }
