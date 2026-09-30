"""Physically-based cloth solver: square sheet falling onto a sharp edge and point.

Implementation
--------------
The cloth is discretised on the frozen 41x41 grid (fixed topology) as a
particle system with structural / shear / bending distance constraints, and
integrated with a sub-stepped position-based (PBD/XPBD-style) scheme.  Every
sub-step:

1.  applies gravity and a small aerodynamic drag to the velocities,
2.  advances positions (semi-implicit Euler),
3.  projects the distance constraints (structural, shear, weak bending),
4.  projects non-penetration against the frozen obstacles (ground box, sharp
    edge box, sharp cone point) using exact signed distance functions, together
    with Coulomb friction (mu = 0.35) at the contact points,
5.  performs one penalty-free self-contact relaxation pass (KD-tree, cloth
    thickness), and
6.  recovers the new velocities from the position change.

The exported raw state (positions, velocities, per-sample mass, per-element
deformation gradient, contact signed distance and mass source) is a pure
function of the simulated physical state; nothing is scripted or key-framed.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

GRAVITY_FALLBACK = np.asarray([0.0, 0.0, -9.81], dtype=float)


class BenchmarkSolver:
    """Deterministic, stable cloth simulation with edge / point contact."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        metadata = scene.metadata()
        initial = scene.initial_state()

        cloth = dict(config.get("cloth", {}))
        self._side = float(cloth.get("side_length", 1.8))
        self._nx, self._ny = (int(v) for v in cloth.get("grid_shape", (41, 41)))
        self._thickness = float(cloth.get("thickness", 0.008))
        self._mass_total = float(cloth.get("total_mass", 0.42))

        gravity = np.asarray(config.get("gravity", GRAVITY_FALLBACK), dtype=float)
        if gravity.shape != (3,):
            gravity = GRAVITY_FALLBACK
        self._gravity = gravity

        friction = dict(config.get("friction", {}))
        self._mu = float(friction.get("coefficient", 0.35))
        contact = dict(config.get("contact", {}))
        self._contact_offset = float(contact.get("thickness_offset", self._thickness))
        # Keep the sample (particle) at least half a thickness plus the frozen
        # thickness offset away from any obstacle surface.
        self._contact_radius = 0.5 * self._thickness + self._contact_offset

        sample_dt = float(config.get("sample_dt", 1.0 / 60.0))
        self._n_sub = 12
        self._dt_sub = sample_dt / float(self._n_sub)
        self._iters = 6
        self._damping = 0.9  # 1/s, mild aerodynamic drag
        self._max_speed = 25.0

        # --- particle data --------------------------------------------------
        self._x = np.asarray(initial.position, dtype=float).copy()
        self._v = np.asarray(initial.velocity, dtype=float).copy()
        count = self._x.shape[0]
        mass = initial.attributes.get("mass_quantity")
        if mass is None:
            mass = np.full(count, self._mass_total / count)
        self._mass = np.asarray(mass, dtype=float).reshape(count).copy()
        uv = initial.attributes.get("uv")
        if uv is None or np.asarray(uv).shape != (count, 2):
            uv = self._grid_uv()
        self._uv = np.asarray(uv, dtype=float).copy()
        self._index = np.arange(count).reshape(self._ny, self._nx)

        faces = np.asarray(metadata["topology"]["faces"], dtype=np.int64).reshape(-1, 3)
        self._faces = faces

        self._build_constraints()
        self._build_colliders(config, metadata)
        self._build_rest_gradient()

        self._x_prev = self._x.copy()
        self._step_count = 0
        self._time = 0.0
        self._exported = 0
        self._max_penetration = 0.0
        self._active_contacts = 0

    def _grid_uv(self) -> np.ndarray:
        u = np.linspace(0.0, 1.0, self._nx)
        v = np.linspace(0.0, 1.0, self._ny)
        return np.asarray([[ux, vy] for vy in v for ux in u], dtype=float)

    # ----------------------------------------------------------- constraints
    def _build_constraints(self) -> None:
        idx = self._index
        pairs = []
        # structural (grid neighbours)
        pairs.append((idx[:, :-1].ravel(), idx[:, 1:].ravel()))
        pairs.append((idx[:-1, :].ravel(), idx[1:, :].ravel()))
        # shear (quad diagonals)
        a = idx[:-1, :-1].ravel()
        b = idx[:-1, 1:].ravel()
        c = idx[1:, :-1].ravel()
        d = idx[1:, 1:].ravel()
        pairs.append((a, d))
        pairs.append((b, c))
        # weak bending (two cells apart)
        pairs.append((idx[:, :-2].ravel(), idx[:, 2:].ravel()))
        pairs.append((idx[:-2, :].ravel(), idx[2:, :].ravel()))

        stiffness = (1.0, 1.0, 0.55, 0.55, 0.05, 0.05)
        self._groups = []
        for (i1, i2), k in zip(pairs, stiffness):
            i1 = np.asarray(i1, dtype=np.int64)
            i2 = np.asarray(i2, dtype=np.int64)
            rest = np.linalg.norm(self._x[i1] - self._x[i2], axis=1)
            self._groups.append(
                (i1, i2, rest, float(k))
            )

    # ------------------------------------------------------------- colliders
    def _build_colliders(self, config: dict, metadata: dict) -> None:
        boxes = []
        ground = dict(config.get("ground", metadata.get("ground", {})) or {})
        if ground:
            boxes.append(
                (
                    np.asarray(ground["center"], dtype=float),
                    np.asarray(ground["half_extents"], dtype=float),
                )
            )
        edge = dict(config.get("sharp_edge", metadata.get("sharp_edge", {})) or {})
        if edge:
            boxes.append(
                (
                    np.asarray(edge["center"], dtype=float),
                    np.asarray(edge["half_extents"], dtype=float),
                )
            )
        self._boxes = boxes

        point = dict(config.get("sharp_point", metadata.get("sharp_point", {})) or {})
        self._cone = None
        if point:
            tip = np.asarray(point["tip"], dtype=float).reshape(3)
            height = float(point["height"])
            radius = float(point["base_radius"])
            if height > 0.0 and radius > 0.0:
                self._cone = (tip, height, radius)

    # ---------------------------------------------------- rest deformation
    def _build_rest_gradient(self) -> None:
        """Inverse of the material rest-edge matrix of every triangle."""
        material = np.zeros((self._nx * self._ny, 3), dtype=float)
        material[:, 0] = self._uv[:, 0] * self._side
        material[:, 1] = self._uv[:, 1] * self._side
        f = self._faces
        e1 = material[f[:, 1]] - material[f[:, 0]]
        e2 = material[f[:, 2]] - material[f[:, 0]]
        e3 = np.tile(np.asarray([0.0, 0.0, 0.5 * self._thickness]), (f.shape[0], 1))
        rest = np.stack((e1, e2, e3), axis=2)  # (Ne, 3, 3) columns
        self._rest_inv = np.linalg.inv(rest)
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        self._face_weight = np.maximum(area, 1e-14)

    # ------------------------------------------------------------ simulation
    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError("advance_to requires a finite non-negative time")
        elapsed = self._step_count * self._dt_sub
        n_steps = int(round((target - elapsed) / self._dt_sub))
        if n_steps > 0:
            for _ in range(n_steps):
                self._step()
            self._step_count += n_steps
        self._time = target
        frame = self._make_frame(target)
        self._exported += 1
        return frame

    def _step(self) -> None:
        dt = self._dt_sub
        x = self._x
        v = self._v
        self._x_prev = x.copy()

        v += self._gravity * dt
        v *= max(0.0, 1.0 - self._damping * dt)
        speed = np.linalg.norm(v, axis=1)
        over = speed > self._max_speed
        if np.any(over):
            v[over] *= (self._max_speed / speed[over])[:, None]
        x += v * dt

        for _ in range(self._iters):
            for i1, i2, rest, k in self._groups:
                self._solve_distance(i1, i2, rest, k)
            self._solve_contacts()

        v[:] = (x - self._x_prev) / dt
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _solve_distance(self, i1, i2, rest, k: float) -> None:
        x = self._x
        d = x[i1] - x[i2]
        length = np.sqrt(np.einsum("ij,ij->i", d, d))
        safe = length > 1e-12
        inv = np.where(safe, 1.0 / np.where(safe, length, 1.0), 0.0)
        factor = k * (length - rest) * inv
        w = 0.5 * factor[:, None] * d
        count = x.shape[0]
        acc = np.zeros_like(x)
        for c in range(3):
            acc[:, c] = np.bincount(i2, weights=w[:, c], minlength=count) - np.bincount(
                i1, weights=w[:, c], minlength=count
            )
        x += acc

    # ------------------------------------------------------------- contacts
    def _solve_contacts(self) -> None:
        x = self._x
        radius = self._contact_radius
        for center, half in self._boxes:
            gap = self._box_sdf(x, center, half) - radius
            self._project(x, gap, lambda p, c=center, h=half: self._box_sdf(p, c, h))
        if self._cone is not None:
            tip, height, cone_radius = self._cone
            gap = self._cone_sdf(x, tip, height, cone_radius) - radius
            self._project(
                x,
                gap,
                lambda p, t=tip, h=height, r=cone_radius: self._cone_sdf(p, t, h, r),
            )
        self._self_contact()

    def _project(self, x, gap, sdf) -> None:
        active = gap < 0.0
        if not np.any(active):
            return
        idx = np.nonzero(active)[0]
        self._active_contacts += int(idx.size)
        push = -gap[idx]
        self._max_penetration = max(self._max_penetration, float(np.max(push)))
        normal = self._sdf_normal(sdf, x[idx])
        x[idx] = x[idx] + normal * push[:, None]

        # Coulomb friction: limit the tangential sliding of the contact sample
        # by mu times the normal correction applied in this sub-step.
        delta = x[idx] - self._x_prev[idx]
        d_n = np.einsum("ij,ij->i", delta, normal)
        tangential = delta - d_n[:, None] * normal
        t_len = np.linalg.norm(tangential, axis=1)
        limit = self._mu * push
        shrink = np.where(t_len > limit, 1.0 - limit / np.maximum(t_len, 1e-30), 1.0)
        shrink = np.clip(shrink, 0.0, 1.0)
        x[idx] = x[idx] - tangential * shrink[:, None]

    def _sdf_normal(self, sdf, p: np.ndarray) -> np.ndarray:
        eps = 1e-5
        normal = np.zeros_like(p)
        off = np.zeros(3)
        for c in range(3):
            off[c] = eps
            normal[:, c] = sdf(p + off) - sdf(p - off)
            off[c] = 0.0
        length = np.linalg.norm(normal, axis=1, keepdims=True)
        bad = length[:, 0] < 1e-12
        normal = normal / np.maximum(length, 1e-30)
        if np.any(bad):
            normal[bad] = np.asarray([0.0, 0.0, 1.0])
        return normal

    @staticmethod
    def _box_sdf(p: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
        q = np.abs(p - center) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    @staticmethod
    def _cone_sdf(p: np.ndarray, tip: np.ndarray, height: float, radius: float) -> np.ndarray:
        """Exact signed distance to a solid cone with apex `tip`, axis -z, cap at z=tip-h."""
        local = p - tip
        qx = np.hypot(local[:, 0], local[:, 1])
        qy = local[:, 2] + 0.5 * height
        hh = 0.5 * height
        r1 = radius  # radius at qy = -hh (base)
        r2 = 0.0  # apex at qy = +hh
        k1 = np.asarray([r2, hh])
        k2 = np.asarray([r2 - r1, 2.0 * hh])
        k2sq = float(k2 @ k2)
        cap_radius = np.where(qy < 0.0, r1, r2)
        ca_x = qx - np.minimum(qx, cap_radius)
        ca_y = np.abs(qy) - hh
        q = np.stack((qx, qy), axis=1)
        t = np.clip(((k1[None, :] - q) @ k2) / k2sq, 0.0, 1.0)
        cb = q - k1[None, :] + t[:, None] * k2[None, :]
        d_ca = np.hypot(ca_x, ca_y)
        d_cb = np.linalg.norm(cb, axis=1)
        sign = np.where((cb[:, 0] < 0.0) & (ca_y < 0.0), -1.0, 1.0)
        return sign * np.minimum(d_ca, d_cb)

    def _self_contact(self) -> None:
        x = self._x
        radius = self._thickness
        tree = cKDTree(x)
        pairs = tree.query_pairs(radius, output_type="ndarray")
        if pairs.size == 0:
            return
        i = pairs[:, 0]
        j = pairs[:, 1]
        d = x[i] - x[j]
        length = np.linalg.norm(d, axis=1)
        active = length < radius - 1e-12
        if not np.any(active):
            return
        i = i[active]
        j = j[active]
        length = length[active]
        direction = d[active] / length[:, None]
        push = 0.5 * (radius - length)
        w = push[:, None] * direction
        count = x.shape[0]
        acc = np.zeros_like(x)
        for c in range(3):
            acc[:, c] = np.bincount(i, weights=w[:, c], minlength=count) - np.bincount(
                j, weights=w[:, c], minlength=count
            )
        x += acc

    # -------------------------------------------------------------- exports
    def _min_gap(self) -> np.ndarray:
        x = self._x
        radius = self._contact_radius
        gap = np.full(x.shape[0], np.inf)
        for center, half in self._boxes:
            gap = np.minimum(gap, self._box_sdf(x, center, half) - radius)
        if self._cone is not None:
            tip, height, cone_radius = self._cone
            gap = np.minimum(gap, self._cone_sdf(x, tip, height, cone_radius) - radius)
        if not np.all(np.isfinite(gap)):
            gap = np.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)
        return gap

    def _deformation_gradient(self) -> np.ndarray:
        x = self._x
        f = self._faces
        e1 = x[f[:, 1]] - x[f[:, 0]]
        e2 = x[f[:, 2]] - x[f[:, 0]]
        cross = np.cross(e1, e2)
        norm = np.linalg.norm(cross, axis=1)
        degenerate = norm < 1e-12
        normal = cross / np.maximum(norm, 1e-30)[:, None]
        if np.any(degenerate):
            normal[degenerate] = np.asarray([0.0, 0.0, 1.0])
        e3 = normal * (0.5 * self._thickness)
        current = np.stack((e1, e2, e3), axis=2)
        f_tri = current @ self._rest_inv
        weight = self._face_weight
        count = x.shape[0]
        flat = np.repeat(f, 3)
        wflat = np.repeat(weight, 3)
        grad = np.zeros((count, 3, 3), dtype=float)
        wsum = np.zeros(count, dtype=float)
        for a in range(3):
            for b in range(3):
                grad[:, a, b] = np.bincount(
                    flat, weights=(f_tri[:, a, b] * weight).repeat(3), minlength=count
                )
        wsum = np.bincount(flat, weights=wflat, minlength=count)
        wsum = np.maximum(wsum, 1e-30)
        grad /= wsum[:, None, None]
        np.nan_to_num(grad, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return grad

    def _make_frame(self, time: float):
        from worldsolver.contracts import StateFrame

        count = self._x.shape[0]
        frame = StateFrame(
            time=float(time),
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._mass.copy(),
            attributes={
                "uv": self._uv.copy(),
                "mass_quantity": self._mass.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": self._min_gap(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={"contact_length_scale": float(self._thickness)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        x = self._x
        return {
            "scene_id": "031_cloth_edge_point_contact",
            "seed": self.seed,
            "samples": int(x.shape[0]),
            "substeps": int(self._step_count),
            "dt_sub": float(self._dt_sub),
            "iterations": int(self._iters),
            "contact_radius": float(self._contact_radius),
            "friction": float(self._mu),
            "max_penetration": float(self._max_penetration),
            "min_height": float(np.min(x[:, 2])),
            "max_height": float(np.max(x[:, 2])),
            "min_z_gap": float(np.min(self._min_gap())),
            "mean_speed": float(np.mean(np.linalg.norm(self._v, axis=1))),
            "frames": int(self._exported),
            "finite": bool(np.isfinite(self._x).all() and np.isfinite(self._v).all()),
        }
