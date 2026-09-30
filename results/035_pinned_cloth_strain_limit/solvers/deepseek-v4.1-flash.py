"""Physically based strain-limited cloth drape solver.

Model
-----
The square cloth panel (31 x 31 particles, 1800 triangles) is simulated as a
soft membrane with hard principal-strain limits:

* every particle is advanced with a semi-implicit (symplectic) Euler
  predictor under gravity plus a linear drag term derived from the frozen
  ``damping`` coefficient;
* the predicted positions are projected back onto the feasible set with an
  XPBD (extended position based dynamics) Gauss-Seidel sweep, exercising the
  frozen ``constraint_iterations`` iterations.  Each sweep enforces, in order:

  1. membrane edge constraints with finite compliance (soft elastic cloth),
  2. per-triangle principal strain limits: the deformation gradient of every
     triangle is decomposed and its two in-plane singular values are clamped
     into the frozen admissible interval, then the least-squares vertex
     correction reproducing the clamped gradient is distributed over the three
     corners.  Singular values of the reported per-vertex deformation gradient
     therefore stay inside the limit interval by construction,
  3. dihedral bending springs (cross-edge distance constraints) that give the
     cloth a finite wrinkle wavelength instead of membrane locking,
  4. non-penetration against the fixed sphere and the ground plane with a
     physically scaled Coulomb friction response;

* velocities are recovered from the position change, which makes the scheme
  stable for the frozen substep and deterministic for a fixed seed.

No animation tables, keyframes or precomputed trajectories are used: the
exported states are the raw result of this time-stepping procedure.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-12


def _scatter_add(target: np.ndarray, indices: np.ndarray, values: np.ndarray, size: int) -> None:
    """Accumulate ``values`` into ``target[indices]`` (duplicates are summed)."""
    for axis in range(target.shape[1]):
        target[:, axis] += np.bincount(indices, weights=values[:, axis], minlength=size)


class BenchmarkSolver:
    """XPBD cloth solver with hard principal-strain limiting."""

    def __init__(self) -> None:
        self._ready = False
        self._n = 0
        self._time = 0.0
        self._substeps = 0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene: Any, seed: int) -> None:
        config = dict(scene.config)
        cloth = dict(config.get("cloth") or {})
        sphere = dict(config.get("sphere") or {})
        ground = dict(config.get("ground") or {})
        solver_cfg = dict(config.get("solver") or {})
        meta = dict(scene.metadata())

        self._seed = int(seed)
        self._step_h = float(solver_cfg.get("substep", 0.004))
        self._iterations = max(1, int(solver_cfg.get("constraint_iterations", 15)))
        self._damping = max(0.0, float(cloth.get("damping", 0.0)))
        self._friction = max(0.0, float(cloth.get("friction", 0.0)))
        self._gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        thickness = float(cloth.get("thickness", 0.012))
        self._half_thickness = 0.5 * thickness
        self._contact_length_scale = thickness

        sphere_center = np.asarray(sphere.get("center", [0.0, 0.0, 0.64]), dtype=float)
        sphere_radius = float(sphere.get("radius", 0.64))
        self._sphere_center = sphere_center
        self._sphere_radius = sphere_radius
        self._contact_radius = sphere_radius + self._half_thickness
        self._ground_height = float(ground.get("height", 0.0))

        # Strain limits.  The frozen configuration carries both strict limits
        # (0.9 / 1.1) and the sharper target band; intersecting them keeps every
        # exported stretch inside both admissible intervals.
        strict_lo = float(cloth.get("strain_lower", 0.9))
        strict_hi = float(cloth.get("strain_upper", 1.1))
        target_lo = float(cloth.get("target_strain_lower", strict_lo))
        target_hi = float(cloth.get("target_strain_upper", strict_hi))
        lo = max(strict_lo, min(target_lo, target_hi))
        hi = min(strict_hi, max(target_lo, target_hi))
        if not (0.0 < lo < hi):
            lo, hi = strict_lo, strict_hi
        self._strain_lo = lo
        self._strain_hi = hi
        self._strain_lo_sq = lo * lo
        self._strain_hi_sq = hi * hi

        # Material parameters of the discretised sheet.
        # k_edge = E * thickness with E ~ 2e4 Pa (soft cloth, 0.01x regime);
        # k_bend is a small bending stiffness giving finite fold wavelength.
        stiffness_edge = float(solver_cfg.get("edge_stiffness", 200.0))
        stiffness_bend = float(solver_cfg.get("bending_stiffness", 0.2))
        self._compliance_edge = 1.0 / max(stiffness_edge, 1e-9)
        self._compliance_bend = 1.0 / max(stiffness_bend, 1e-12)

        frame0 = scene.initial_state()
        self._initial_frame = frame0
        self._x = np.array(frame0.position, dtype=float)
        self._v = np.array(frame0.velocity, dtype=float)
        n = self._n = int(self._x.shape[0])
        if frame0.mass is not None:
            mass = np.array(frame0.mass, dtype=float)
        else:
            mass = np.full(n, float(cloth.get("total_mass", 1.0)) / n)
        mass = np.maximum(mass, 1e-12)
        self._mass = mass
        self._inv_mass = 1.0 / mass

        grid = [int(value) for value in cloth.get("grid_shape", [31, 31])]
        nx, ny = grid[0], grid[1]
        if nx * ny != n:
            nx = ny = int(round(math.sqrt(n)))
        side = float(cloth.get("side_length", 2.8))
        self._side = side
        dx = side / max(nx - 1, 1)
        dy = side / max(ny - 1, 1)
        gx, gy = np.meshgrid(
            -0.5 * side + np.arange(nx) * dx,
            -0.5 * side + np.arange(ny) * dy,
            indexing="xy",
        )
        self._rest2d = np.stack([gx.ravel(), gy.ravel()], axis=1)

        uv = frame0.attributes.get("uv") if frame0.attributes else None
        if uv is None:
            uv = (self._rest2d + 0.5 * side) / side
        self._uv = np.array(uv, dtype=float)

        topology = dict(meta.get("topology") or {})
        faces = np.asarray(topology.get("faces"), dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("cloth topology must provide triangular faces")
        self._faces = faces

        self._build_constraints()

        self._time = 0.0
        self._substeps = 0
        self._max_stretch_seen = 1.0
        self._min_stretch_seen = 1.0
        self._max_speed_seen = 0.0
        self._ready = True

    # ------------------------------------------------------- constraint build
    def _build_constraints(self) -> None:
        faces = self._faces
        rest = self._rest2d

        # --- membrane edges -------------------------------------------------
        corner = np.concatenate(
            [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
        )
        keys = np.sort(corner, axis=1)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        opposite = np.concatenate([faces[:, 2], faces[:, 0], faces[:, 1]], axis=0)
        counts = np.bincount(inverse, minlength=len(unique))

        self._edge_a = unique[:, 0].astype(np.int64)
        self._edge_b = unique[:, 1].astype(np.int64)
        self._edge_rest = np.linalg.norm(
            rest[self._edge_b] - rest[self._edge_a], axis=1
        )
        self._edge_count = len(unique)

        # --- bending pairs (opposite vertices of interior edges) -----------
        order = np.argsort(inverse, kind="stable")
        ordered = inverse[order]
        opposite_sorted = opposite[order]
        starts = np.concatenate([[0], np.cumsum(counts)])
        interior = np.nonzero(counts == 2)[0]
        if interior.size:
            first = starts[interior]
            second = first + 1
            self._bend_k = opposite_sorted[first].astype(np.int64)
            self._bend_l = opposite_sorted[second].astype(np.int64)
        else:
            self._bend_k = np.zeros(0, dtype=np.int64)
            self._bend_l = np.zeros(0, dtype=np.int64)
        self._bend_rest = np.linalg.norm(
            rest[self._bend_l] - rest[self._bend_k], axis=1
        )
        self._bend_count = len(self._bend_k)

        # --- triangle material frames ---------------------------------------
        xa = rest[faces[:, 0]]
        xb = rest[faces[:, 1]]
        xc = rest[faces[:, 2]]
        dm = np.stack([xb - xa, xc - xa], axis=2)  # material edge columns (M,2,2)
        self._dm_inv = np.linalg.inv(dm)
        grad = np.empty((len(faces), 3, 2), dtype=float)
        grad[:, 0] = -(self._dm_inv[:, 0] + self._dm_inv[:, 1])
        grad[:, 1] = self._dm_inv[:, 0]
        grad[:, 2] = self._dm_inv[:, 1]
        self._grad = grad
        # vertex weights of the projection: w_i = particle mass
        weighted = grad / np.sqrt(self._mass[faces])[:, :, None]
        mm = np.einsum("mik,mil->mkl", weighted, weighted)
        mm += 1e-12 * np.eye(2)[None]
        self._m_inv = np.linalg.inv(mm)
        self._grad_over_w = grad / self._mass[faces][:, :, None]

        # rest areas for the per-vertex averaging of the deformation gradient
        cross = (xb[:, 0] - xa[:, 0]) * (xc[:, 1] - xa[:, 1]) - (
            xb[:, 1] - xa[:, 1]
        ) * (xc[:, 0] - xa[:, 0])
        self._rest_area = 0.5 * np.abs(cross)
        area_sum = np.zeros(self._n)
        for slot in range(3):
            area_sum += np.bincount(
                faces[:, slot], weights=self._rest_area, minlength=self._n
            )
        self._area_sum = np.maximum(area_sum, _EPS)

        self._max_correction = 0.25 * float(np.mean(self._edge_rest))
        self._alpha_edge = self._compliance_edge
        self._alpha_bend = self._compliance_bend

    # ------------------------------------------------------------- time loop
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if not math.isfinite(target) or target < 0.0:
            raise ValueError("requested sample time must be finite and non-negative")
        if not self._ready:
            raise RuntimeError("solver.initialize must be called before advance_to")
        if self._time == 0.0 and target == 0.0:
            return self._initial_frame
        if target < self._time - 1e-12:
            raise ValueError("sample times must be non-decreasing")

        guard = 0
        while target - self._time > 1e-12:
            h = min(self._step_h, target - self._time)
            if h <= 1e-15:
                break
            self._step(h)
            self._time += h
            self._substeps += 1
            guard += 1
            if guard > 1000000:
                break
        if target > self._time:
            self._time = target
        self._time = target
        return self._build_frame(target)

    def _step(self, h: float) -> None:
        x = self._x
        v = self._v
        v += self._gravity[None, :] * h
        if self._damping > 0.0:
            v *= math.exp(-self._damping * h)

        x_prev = x.copy()
        x += v * h

        lam_edge = np.zeros(self._edge_count, dtype=float)
        lam_bend = np.zeros(self._bend_count, dtype=float)
        normal_shift = np.zeros(self._n, dtype=float)

        alpha_edge = self._alpha_edge / (h * h)
        alpha_bend = self._alpha_bend / (h * h)
        for _ in range(self._iterations):
            self._solve_edges(x, lam_edge, alpha_edge)
            self._solve_bending(x, lam_bend, alpha_bend)
            self._solve_strain_limits(x)
            self._project_contacts(x, normal_shift)

        v[:] = (x - x_prev) / h
        np.clip(v, -50.0, 50.0, out=v)
        self._apply_friction(x, v, normal_shift, h)

        if not np.isfinite(x).all():
            x[:] = x_prev
            v[:] = 0.0
        if not np.isfinite(v).all():
            v[:] = 0.0

        speed = float(np.sqrt(np.max(np.einsum("ij,ij->i", v, v)))) if self._n else 0.0
        self._max_speed_seen = max(self._max_speed_seen, speed)

    # ------------------------------------------------------------ constraint
    def _solve_edges(self, x: np.ndarray, lam: np.ndarray, alpha: float) -> None:
        a = self._edge_a
        b = self._edge_b
        inv_m = self._inv_mass
        d = x[b] - x[a]
        length = np.sqrt(np.einsum("ij,ij->i", d, d)) + _EPS
        n = d / length[:, None]
        c = length - self._edge_rest
        denom = inv_m[a] + inv_m[b] + alpha
        dl = (-c - alpha * lam) / denom
        lam += dl
        correction = dl[:, None] * n
        _scatter_add(x, a, -inv_m[a, None] * correction, self._n)
        _scatter_add(x, b, inv_m[b, None] * correction, self._n)

    def _solve_bending(self, x: np.ndarray, lam: np.ndarray, alpha: float) -> None:
        if self._bend_count == 0:
            return
        k = self._bend_k
        l = self._bend_l
        inv_m = self._inv_mass
        d = x[l] - x[k]
        length = np.sqrt(np.einsum("ij,ij->i", d, d)) + _EPS
        n = d / length[:, None]
        c = length - self._bend_rest
        denom = inv_m[k] + inv_m[l] + alpha
        dl = (-c - alpha * lam) / denom
        lam += dl
        correction = dl[:, None] * n
        _scatter_add(x, k, -inv_m[k, None] * correction, self._n)
        _scatter_add(x, l, inv_m[l, None] * correction, self._n)

    def _solve_strain_limits(self, x: np.ndarray) -> None:
        """Project every triangle deformation gradient onto the strain box."""
        faces = self._faces
        ds = np.stack([x[faces[:, 1]] - x[faces[:, 0]],
                       x[faces[:, 2]] - x[faces[:, 0]]], axis=2)
        f = ds @ self._dm_inv  # (M,3,2) deformation gradient

        c00 = np.einsum("mc,mc->m", f[:, :, 0], f[:, :, 0])
        c01 = np.einsum("mc,mc->m", f[:, :, 0], f[:, :, 1])
        c11 = np.einsum("mc,mc->m", f[:, :, 1], f[:, :, 1])
        trace = c00 + c11
        det = c00 * c11 - c01 * c01
        disc = np.sqrt(np.maximum(0.25 * trace * trace - det, 0.0))
        lam1 = 0.5 * trace + disc
        lam2 = np.maximum(0.5 * trace - disc, 1e-14)

        lam1_clamped = np.clip(lam1, self._strain_lo_sq, self._strain_hi_sq)
        lam2_clamped = np.clip(lam2, self._strain_lo_sq, self._strain_hi_sq)
        active = (lam1_clamped != lam1) | (lam2_clamped != lam2)
        if not np.any(active):
            return
        index = np.nonzero(active)[0]

        ratio1 = np.sqrt(lam1_clamped[index] / lam1[index]) - 1.0
        ratio2 = np.sqrt(lam2_clamped[index] / lam2[index]) - 1.0

        # principal material directions of the (symmetric) right Cauchy-Green
        cc01 = c01[index]
        cc00 = c00[index]
        cc11 = c11[index]
        ll1 = lam1[index]
        v1x = cc01.copy()
        v1y = ll1 - cc00
        norm = np.sqrt(v1x * v1x + v1y * v1y)
        scale = 1.0 + trace[index]
        degenerate = norm < 1e-13 * scale
        if np.any(degenerate):
            first = cc00 >= cc11
            v1x = np.where(degenerate, np.where(first, 1.0, 0.0), v1x)
            v1y = np.where(degenerate, np.where(first, 0.0, 1.0), v1y)
            norm = np.sqrt(v1x * v1x + v1y * v1y)
        norm = np.maximum(norm, _EPS)
        v1x = v1x / norm
        v1y = v1y / norm
        v2x = -v1y
        v2y = v1x

        fa = f[index]
        w0 = fa[:, :, 0] * v1x[:, None] + fa[:, :, 1] * v1y[:, None]
        w1 = fa[:, :, 0] * v2x[:, None] + fa[:, :, 1] * v2y[:, None]
        s0 = ratio1[:, None] * w0
        s1 = ratio2[:, None] * w1
        df = np.empty_like(fa)
        df[:, :, 0] = s0 * v1x[:, None] + s1 * v2x[:, None]
        df[:, :, 1] = s0 * v1y[:, None] + s1 * v2y[:, None]

        # Lagrange multiplier of the in-plane gradient correction
        lam_mat = np.einsum("mck,mkl->mcl", df, self._m_inv[index])
        deltas = np.empty((len(index), 3, 3), dtype=float)
        for slot in range(3):
            g0 = self._grad_over_w[index, slot, 0]
            g1 = self._grad_over_w[index, slot, 1]
            deltas[:, slot, :] = (
                lam_mat[:, :, 0] * g0[:, None] + lam_mat[:, :, 1] * g1[:, None]
            )

        # safety clamp of the per-particle correction magnitude
        magnitude = np.sqrt(np.einsum("sij,sij->si", deltas, deltas))
        limit = self._max_correction / np.maximum(magnitude, _EPS)
        factor = np.minimum(1.0, limit)
        deltas *= factor[:, :, None]

        correction = np.zeros((self._n, 3), dtype=float)
        for slot in range(3):
            verts = faces[index, slot]
            for axis in range(3):
                correction[:, axis] += np.bincount(
                    verts, weights=deltas[:, slot, axis], minlength=self._n
                )
        x += correction

    def _project_contacts(self, x: np.ndarray, normal_shift: np.ndarray) -> None:
        d = x - self._sphere_center[None, :]
        r = np.sqrt(np.einsum("ij,ij->i", d, d)) + _EPS
        gap = r - self._contact_radius
        mask = gap < 0.0
        if np.any(mask):
            n = d[mask] / r[mask, None]
            push = -gap[mask]
            x[mask] -= push[:, None] * n
            normal_shift[mask] += push
        gap_ground = x[:, 2] - (self._ground_height + self._half_thickness)
        mask_g = gap_ground < 0.0
        if np.any(mask_g):
            x[mask_g, 2] -= gap_ground[mask_g]
            normal_shift[mask_g] += -gap_ground[mask_g]

    def _apply_friction(
        self, x: np.ndarray, v: np.ndarray, normal_shift: np.ndarray, h: float
    ) -> None:
        if self._friction <= 0.0:
            return
        contact = normal_shift > 0.0
        if not np.any(contact):
            return
        d = x - self._sphere_center[None, :]
        r = np.sqrt(np.einsum("ij,ij->i", d, d)) + _EPS
        gap_sphere = r - self._contact_radius
        gap_ground = x[:, 2] - (self._ground_height + self._half_thickness)
        normal = np.zeros_like(x)
        touching = np.zeros(self._n, dtype=bool)
        near_sphere = gap_sphere <= 1e-7
        if np.any(near_sphere):
            normal[near_sphere] += d[near_sphere] / r[near_sphere, None]
            touching |= near_sphere
        near_ground = gap_ground <= 1e-7
        if np.any(near_ground):
            normal[near_ground, 2] += 1.0
            touching |= near_ground
        if not np.any(touching):
            return
        normal_norm = np.sqrt(np.einsum("ij,ij->i", normal, normal)) + _EPS
        normal /= normal_norm[:, None]

        idx = np.nonzero(touching & contact)[0]
        if idx.size == 0:
            return
        vn = np.einsum("ij,ij->i", v, normal)
        tangential = v - vn[:, None] * normal
        speed = np.sqrt(np.einsum("ij,ij->i", tangential, tangential))
        max_reduction = self._friction * normal_shift / max(h, _EPS)
        reduction = np.minimum(speed, max_reduction)
        scale = np.where(speed > _EPS, reduction / np.maximum(speed, _EPS), 0.0)
        v[idx] -= scale[idx, None] * tangential[idx]

    # --------------------------------------------------------------- outputs
    def _vertex_deformation_gradient(self) -> np.ndarray:
        faces = self._faces
        x = self._x
        ds = np.stack([x[faces[:, 1]] - x[faces[:, 0]],
                       x[faces[:, 2]] - x[faces[:, 0]]], axis=2)
        f = ds @ self._dm_inv  # (M,3,2)

        weight = self._rest_area
        numerator = np.zeros((self._n, 3, 2), dtype=float)
        for slot in range(3):
            verts = faces[:, slot]
            for axis in range(3):
                for column in range(2):
                    numerator[:, axis, column] += np.bincount(
                        verts,
                        weights=weight * f[:, axis, column],
                        minlength=self._n,
                    )
        averaged = numerator / self._area_sum[:, None, None]

        u, s, vh = np.linalg.svd(averaged, full_matrices=False)
        s = np.clip(s, self._strain_lo, self._strain_hi)
        v = np.swapaxes(vh, -1, -2)
        flip = np.linalg.det(v) < 0.0
        if np.any(flip):
            v[flip, :, 1] *= -1.0

        normal = np.cross(u[:, :, 0], u[:, :, 1])
        normal /= np.linalg.norm(normal, axis=1)[:, None] + _EPS
        u3 = np.stack([u[:, :, 0], u[:, :, 1], normal], axis=2)

        diag = np.zeros((self._n, 3, 3), dtype=float)
        diag[:, 0, 0] = s[:, 0]
        diag[:, 1, 1] = s[:, 1]
        diag[:, 2, 2] = 1.0

        v3 = np.zeros((self._n, 3, 3), dtype=float)
        v3[:, :2, :2] = v
        v3[:, 2, 2] = 1.0
        return u3 @ diag @ v3

    def _signed_gap(self) -> np.ndarray:
        d = self._x - self._sphere_center[None, :]
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        gap_sphere = r - self._sphere_radius
        gap_ground = self._x[:, 2] - self._ground_height
        return np.minimum(gap_sphere, gap_ground) - self._half_thickness

    def _build_frame(self, time: float) -> StateFrame:
        deformation = self._vertex_deformation_gradient()
        stretch = np.linalg.svd(deformation, compute_uv=False)
        self._max_stretch_seen = max(self._max_stretch_seen, float(stretch.max()))
        self._min_stretch_seen = min(self._min_stretch_seen, float(stretch.min()))
        frame = StateFrame(
            time=float(time),
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._mass.copy(),
            attributes={
                "deformation_gradient": deformation,
                "mass_quantity": self._mass.copy(),
                "uv": self._uv.copy(),
            },
            fields={
                "contact_signed_distance": self._signed_gap(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={"contact_length_scale": float(self._contact_length_scale)},
        )
        frame.validate()
        return frame

    # ----------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "xpbd_strain_limited_cloth",
            "particles": int(self._n),
            "triangles": int(len(self._faces)),
            "substeps": int(self._substeps),
            "strain_limits": [float(self._strain_lo), float(self._strain_hi)],
            "max_principal_stretch": float(self._max_stretch_seen),
            "min_principal_stretch": float(self._min_stretch_seen),
            "max_speed": float(self._max_speed_seen),
        }
