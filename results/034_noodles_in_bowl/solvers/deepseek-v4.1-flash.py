"""Deterministic physically based solver for scene 034 (noodles in a bowl).

Physical model
--------------
The frozen topology discretises every noodle as a two node rod: 1250 point
masses (0.00125 kg each), 625 inextensible segments of ~0.40 m with radius
0.006 m.  Each noodle is therefore a rigid dumbbell segment.  The dynamics is
integrated with a position based dynamics (PBD) / XPBD style time stepping
scheme:

* gravity is applied as the only external force (``fields.external_force``);
* a distance constraint per noodle keeps the segment inextensible
  (exact projection, so the rod stays rigid and cannot bend/stretch);
* the axisymmetric bowl interior is a signed distance obstacle with a finite
  wall thickness; nodes are kept at one noodle radius from the inner surface;
* capsule/capsule (segment/segment) contacts give the finite thickness
  self contact and mutual contact of the noodles;
* a velocity level tangential projection supplies Coulomb like friction so
  the pile dissipates energy and settles into a stable equilibrium.

The scheme steps real physical time with fixed size substeps and is fully
deterministic (no randomness, deterministic pair ordering).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

try:  # scipy is available in the runtime, but never required
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


# Frozen inner profile of the bowl in the (radius, height) half plane,
# taken from the scene geometry (fallback if the yaml cannot be read).
_DEFAULT_BOWL_PROFILE = np.asarray(
    (
        (0.12, 0.00),
        (0.38, 0.08),
        (0.66, 0.34),
        (0.72, 0.48),
    ),
    dtype=float,
)
_DEFAULT_WALL_THICKNESS = 0.025


def _closest_points_between_segments(p1, q1, p2, q2):
    """Vectorised closest points between two sets of 3D segments.

    ``p1/q1`` and ``p2/q2`` are ``[M, 3]`` arrays of segment end points.
    Returns ``(dist, s, t, diff)`` with ``diff = cA - cB`` the vector from the
    closest point on segment B to the closest point on segment A.
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d1, r)
    f = np.einsum("ij,ij->i", d2, r)
    a = np.maximum(a, 1e-18)
    e = np.maximum(e, 1e-18)
    denom = a * e - b * b
    parallel = denom <= 1e-12 * a * e
    s = np.zeros_like(a)
    np.divide(b * f - c * e, denom, out=s, where=~parallel)
    s = np.clip(s, 0.0, 1.0)
    t = (b * s + f) / e
    t_clamped = np.clip(t, 0.0, 1.0)
    s_low = np.clip(-c / a, 0.0, 1.0)
    s_high = np.clip((b - c) / a, 0.0, 1.0)
    s = np.where(t < 0.0, s_low, np.where(t > 1.0, s_high, s))
    c_a = p1 + d1 * s[:, None]
    c_b = p2 + d2 * t_clamped[:, None]
    diff = c_a - c_b
    dist = np.linalg.norm(diff, axis=1)
    return dist, s, t_clamped, diff


class BenchmarkSolver:
    """PBD/XPBD solver for 625 two node rod noodles falling into a bowl."""

    # numerical parameters (fixed => deterministic)
    H_TARGET = 1.0 / 240.0        # target physical substep
    ITERATIONS = 3                # constraint relaxation sweeps per substep
    RELAX_SCHEDULE = (0.6, 0.85, 1.0)   # per sweep positional contact relaxation
    FRICTION = 0.45               # tang. velocity removed per substep (rod)
    WALL_FRICTION = 0.45          # tang. velocity removed per substep (bowl)
    DAMPING = 0.9995              # mild numerical drag per substep
    PAIR_MARGIN = 0.035           # broad phase inflation (m)
    MAX_CORRECTION = 0.02         # cap on a single positional correction
    MAX_VELOCITY = 12.0           # safety clamp (m/s)
    NEAR_CONTACT = 0.02           # reporting band for contact field (m)

    def __init__(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        frame = scene.initial_state()
        meta = dict(scene.metadata() or {})
        self.seed = int(seed)

        gravity = cfg.get("gravity", (0.0, 0.0, -9.81))
        self.gravity = np.asarray(gravity, dtype=float).reshape(3)
        self.radius = float(cfg.get("radius", 0.006))
        self.noodle_length = float(cfg.get("noodle_length", 0.4))

        self.x = np.array(frame.position, dtype=float, copy=True)
        self.v = np.array(frame.velocity, dtype=float, copy=True)
        n_nodes = self.x.shape[0]

        # ---- per sample mass (state.mass / attributes.mass_quantity) --------
        mass_quantity = frame.attributes.get("mass_quantity")
        if frame.mass is not None:
            mass = np.array(frame.mass, dtype=float, copy=True)
        elif mass_quantity is not None:
            mq = np.asarray(mass_quantity, dtype=float)
            mass = mq if mq.ndim == 1 else mq.sum(axis=1)
        else:
            mass = np.full(n_nodes, 1.0 / max(n_nodes, 1), dtype=float)
        if mass.shape[0] != n_nodes:
            mass = np.full(n_nodes, float(np.mean(mass)), dtype=float)
        mass = np.where(np.isfinite(mass) & (mass > 0.0), mass, 1e-9)
        self.mass = mass
        if mass_quantity is not None:
            self.mass_quantity = np.array(mass_quantity, dtype=float, copy=True)
            if self.mass_quantity.shape[0] != n_nodes:
                self.mass_quantity = mass.copy()
        else:
            self.mass_quantity = self.mass.copy()
        self.inv_mass = 1.0 / self.mass

        # ---- fiber topology (frozen) ---------------------------------------
        edges = meta.get("edges")
        if edges is None:
            idx = np.arange(n_nodes)
            edges = np.stack((idx[0::2], idx[1::2]), axis=1)
        edges = np.asarray(edges, dtype=np.int64)
        self.rod_a = np.ascontiguousarray(edges[:, 0])
        self.rod_b = np.ascontiguousarray(edges[:, 1])
        # measured rest lengths of the frozen initial state (no initial jolt)
        self.rest = np.linalg.norm(self.x[self.rod_b] - self.x[self.rod_a], axis=1)
        self.rest = np.maximum(self.rest, 1e-6)
        self.n_rods = self.rod_a.shape[0]

        # ---- contact scales -------------------------------------------------
        scalars = dict(frame.scalars or {})
        self.contact_length_scale = float(scalars.get("contact_length_scale", self.radius))
        if not math.isfinite(self.contact_length_scale) or self.contact_length_scale <= 0.0:
            self.contact_length_scale = self.radius
        self.mass_source = np.zeros(1, dtype=float)

        # ---- bowl geometry --------------------------------------------------
        self.wall_thickness = float(meta.get("wall_thickness", _DEFAULT_WALL_THICKNESS))
        if not math.isfinite(self.wall_thickness) or self.wall_thickness <= 0.0:
            self.wall_thickness = _DEFAULT_WALL_THICKNESS
        self.profile = self._load_bowl_profile(cfg)

        # ---- broad phase ----------------------------------------------------
        self._pair_radius = float(np.max(self.rest)) + 2.0 * self.radius + self.PAIR_MARGIN

        # ---- wall contact state (reused for friction) ------------------------
        self._wall_gap = np.full(n_nodes, np.inf)
        self._wall_dir = np.zeros((n_nodes, 3))
        self._wall_dir[:, 2] = 1.0

        self.time = 0.0
        self.substeps = 0
        self._initialized = True

    # -------------------------------------------------------------- geometry
    def _load_bowl_profile(self, cfg) -> np.ndarray:
        profile = None
        for candidate in (
            Path.cwd() / "scene_geometry.yaml",
            Path(__file__).resolve().parent.parent / "scene_geometry.yaml",
        ):
            try:
                text = candidate.read_text(encoding="utf-8")
            except OSError:
                continue
            profile = self._parse_profile(text)
            if profile is not None:
                break
        if profile is None:
            profile = _DEFAULT_BOWL_PROFILE.copy()
            bowl = cfg.get("bowl") or {}
            rim = float(bowl.get("rim_radius", profile[-1, 0]))
            height = float(bowl.get("height", profile[-1, 1]))
            if profile[-1, 0] > 0.0 and profile[-1, 1] > 0.0:
                profile[:, 0] *= rim / profile[-1, 0]
                profile[:, 1] *= height / profile[-1, 1]
        if profile.shape[0] < 2:
            profile = _DEFAULT_BOWL_PROFILE.copy()
        return np.ascontiguousarray(profile, dtype=float)

    @staticmethod
    def _parse_profile(text: str):
        try:
            import yaml  # PyYAML ships with the frozen runtime
        except Exception:
            return None
        try:
            data = yaml.safe_load(text)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        for obj in data.get("objects", []) or []:
            if not isinstance(obj, dict) or obj.get("id") != "bowl":
                continue
            geometry = obj.get("geometry") or {}
            inner = geometry.get("inner_profile")
            if inner:
                arr = np.asarray(inner, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 2:
                    return arr
        return None

    # ------------------------------------------------------------ time stepping
    def advance_to(self, time: float):
        if not self._initialized:
            raise RuntimeError("BenchmarkSolver.initialize must be called first")
        target = float(time)
        if target > self.time + 1e-12:
            remaining = target - self.time
            n_sub = max(1, int(round(remaining / self.H_TARGET)))
            h = remaining / n_sub
            for _ in range(n_sub):
                self._step(h)
            self.substeps += n_sub
            self.time = target
        return self._frame()

    def _step(self, h: float) -> None:
        x = self.x
        v = self.v
        v = v + h * self.gravity[None, :]
        x_old = x
        x = x + h * v

        pairs = None
        dist = s = t = diff = None
        for iteration in range(self.ITERATIONS):
            relax = self.RELAX_SCHEDULE[min(iteration, len(self.RELAX_SCHEDULE) - 1)]
            self._solve_lengths(x)
            self._solve_bowl(x, relax)
            pairs = self._candidate_pairs(x)
            if pairs.shape[0]:
                p1 = x[self.rod_a[pairs[:, 0]]]
                q1 = x[self.rod_b[pairs[:, 0]]]
                p2 = x[self.rod_a[pairs[:, 1]]]
                q2 = x[self.rod_b[pairs[:, 1]]]
                dist, s, t, diff = _closest_points_between_segments(p1, q1, p2, q2)
                self._solve_rod_contacts(x, pairs, dist, s, t, diff, relax)

        v = (x - x_old) / h
        if pairs is not None and pairs.shape[0]:
            self._apply_rod_friction(v, pairs, dist, s, t, diff)
        self._apply_wall_friction(v)
        v *= self.DAMPING

        speed = np.linalg.norm(v, axis=1)
        fast = speed > self.MAX_VELOCITY
        if np.any(fast):
            v[fast] *= (self.MAX_VELOCITY / speed[fast])[:, None]

        if not (np.isfinite(x).all() and np.isfinite(v).all()):
            x = np.where(np.isfinite(x), x, self.x)
            v = np.where(np.isfinite(v), v, 0.0)
        self.x = x
        self.v = v

    # ----------------------------------------------------------- constraints
    def _solve_lengths(self, x: np.ndarray) -> None:
        pa = self.rod_a
        pb = self.rod_b
        d = x[pb] - x[pa]
        length = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        wa = self.inv_mass[pa]
        wb = self.inv_mass[pb]
        corr = (length - self.rest) / (wa + wb)
        normal = d / length[:, None]
        np.add.at(x, pa, (wa * corr)[:, None] * normal)
        np.add.at(x, pb, -(wb * corr)[:, None] * normal)

    def _solve_bowl(self, x: np.ndarray, relax: float) -> None:
        gap, direction = self._bowl_signed_distance(x)
        self._wall_gap = gap
        self._wall_dir = direction
        pen = self.radius - gap
        active = (pen > 0.0) & (gap > -self.wall_thickness)
        if np.any(active):
            amount = np.clip(pen[active], 0.0, self.MAX_CORRECTION)
            x[active] += (relax * amount)[:, None] * direction[active]

    def _solve_rod_contacts(self, x, pairs, dist, s, t, diff, relax: float) -> None:
        d_min = 2.0 * self.radius
        mask = dist < d_min
        if not np.any(mask):
            return
        idx = np.nonzero(mask)[0]
        d = np.maximum(dist[idx], 1e-9)
        normal = diff[idx] / d[:, None]
        pen = np.clip(d_min - d, 0.0, self.MAX_CORRECTION)
        push = 0.5 * relax * pen
        a0 = self.rod_a[pairs[idx, 0]]
        a1 = self.rod_b[pairs[idx, 0]]
        b0 = self.rod_a[pairs[idx, 1]]
        b1 = self.rod_b[pairs[idx, 1]]
        sa = s[idx][:, None]
        ta = t[idx][:, None]
        ca = push[:, None] * normal
        np.add.at(x, a0, ca * (1.0 - sa))
        np.add.at(x, a1, ca * sa)
        np.add.at(x, b0, -ca * (1.0 - ta))
        np.add.at(x, b1, -ca * ta)

    # -------------------------------------------------------------- friction
    def _apply_rod_friction(self, v, pairs, dist, s, t, diff) -> None:
        d_min = 2.0 * self.radius
        mask = dist < d_min + 0.004
        if not np.any(mask):
            return
        idx = np.nonzero(mask)[0]
        d = np.maximum(dist[idx], 1e-9)
        normal = diff[idx] / d[:, None]
        a0 = self.rod_a[pairs[idx, 0]]
        a1 = self.rod_b[pairs[idx, 0]]
        b0 = self.rod_a[pairs[idx, 1]]
        b1 = self.rod_b[pairs[idx, 1]]
        sa = s[idx][:, None]
        ta = t[idx][:, None]
        va = (1.0 - sa) * v[a0] + sa * v[a1]
        vb = (1.0 - ta) * v[b0] + ta * v[b1]
        rel = va - vb
        rel_n = np.einsum("ij,ij->i", rel, normal)[:, None] * normal
        rel_t = rel - rel_n
        dv = -self.FRICTION * rel_t
        np.add.at(v, a0, 0.5 * (1.0 - sa) * dv)
        np.add.at(v, a1, 0.5 * sa * dv)
        np.add.at(v, b0, -0.5 * (1.0 - ta) * dv)
        np.add.at(v, b1, -0.5 * ta * dv)

    def _apply_wall_friction(self, v: np.ndarray) -> None:
        gap = self._wall_gap
        if gap.shape[0] != v.shape[0]:
            return
        mask = (gap < self.radius + 0.004) & (gap > -self.wall_thickness)
        if not np.any(mask):
            return
        normal = self._wall_dir[mask]
        velocity = v[mask]
        v_n = np.einsum("ij,ij->i", velocity, normal)[:, None] * normal
        v_t = velocity - v_n
        v[mask] = v_n + (1.0 - self.WALL_FRICTION) * v_t

    # ------------------------------------------------------------ broad phase
    def _candidate_pairs(self, x: np.ndarray) -> np.ndarray:
        centers = 0.5 * (x[self.rod_a] + x[self.rod_b])
        radius = self._pair_radius
        if cKDTree is not None:
            tree = cKDTree(centers)
            pairs = tree.query_pairs(radius, output_type="ndarray")
            return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        diff = centers[:, None, :] - centers[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        iu = np.triu_indices(centers.shape[0], 1)
        keep = d2[iu] < radius * radius
        return np.stack((iu[0][keep], iu[1][keep]), axis=1)

    # ---------------------------------------------------------- bowl signed dist
    def _bowl_signed_distance(self, x: np.ndarray):
        """Signed clearance to the bowl interior surface and inward direction.

        ``gap`` is positive inside the cavity (surface to node distance) and
        negative for nodes inside the wall material.  ``direction`` is the unit
        direction that increases the clearance (i.e. into the cavity).
        """
        rho = np.hypot(x[:, 0], x[:, 1])
        z = x[:, 2]
        profile = self.profile
        n = x.shape[0]
        best_d = np.full(n, np.inf)
        best_wr = np.zeros(n)
        best_wz = np.zeros(n)
        best_tr = np.ones(n)
        best_tz = np.zeros(n)
        best_side = np.ones(n)
        for k in range(profile.shape[0] - 1):
            ar, az = profile[k]
            br, bz = profile[k + 1]
            abr = br - ar
            abz = bz - az
            denom = abr * abr + abz * abz
            if denom <= 0.0:
                continue
            wr = rho - ar
            wz = z - az
            u = np.clip((wr * abr + wz * abz) / denom, 0.0, 1.0)
            wrho = rho - (ar + u * abr)
            wzz = z - (az + u * abz)
            d = np.hypot(wrho, wzz)
            side = abr * wzz - abz * wrho
            upd = d < best_d
            best_d = np.where(upd, d, best_d)
            best_wr = np.where(upd, wrho, best_wr)
            best_wz = np.where(upd, wzz, best_wz)
            inv = 1.0 / math.sqrt(denom)
            best_tr = np.where(upd, abr * inv, best_tr)
            best_tz = np.where(upd, abz * inv, best_tz)
            best_side = np.where(upd, side, best_side)

        degenerate = best_d < 1e-10
        safe = np.where(degenerate, 1.0, best_d)
        # unit vector from the closest surface point to the node (2D, rho-z)
        what_r = np.where(degenerate, -best_tz, best_wr / safe)
        what_z = np.where(degenerate, best_tr, best_wz / safe)
        sign = np.where(best_side >= 0.0, 1.0, -1.0)
        sign = np.where(degenerate, 1.0, sign)
        gap = sign * best_d

        radial = np.zeros(n)
        tangential = np.zeros(n)
        nonzero = rho > 1e-12
        radial[nonzero] = x[nonzero, 0] / rho[nonzero]
        tangential[nonzero] = x[nonzero, 1] / rho[nonzero]
        direction = np.stack(
            (sign * what_r * radial, sign * what_r * tangential, sign * what_z), axis=1
        )
        return gap, direction

    # ------------------------------------------------------------ diagnostics
    def _contact_signed_distance(self, x: np.ndarray) -> np.ndarray:
        gap, _ = self._bowl_signed_distance(x)
        values = np.minimum(gap - self.radius, self.NEAR_CONTACT)
        pairs = self._candidate_pairs(x)
        if pairs.shape[0]:
            p1 = x[self.rod_a[pairs[:, 0]]]
            q1 = x[self.rod_b[pairs[:, 0]]]
            p2 = x[self.rod_a[pairs[:, 1]]]
            q2 = x[self.rod_b[pairs[:, 1]]]
            dist, _, _, _ = _closest_points_between_segments(p1, q1, p2, q2)
            d_min = 2.0 * self.radius
            near = dist < d_min + self.NEAR_CONTACT
            if np.any(near):
                idx = np.nonzero(near)[0]
                value = dist[idx] - d_min
                np.minimum.at(values, self.rod_a[pairs[idx, 0]], value)
                np.minimum.at(values, self.rod_b[pairs[idx, 0]], value)
                np.minimum.at(values, self.rod_a[pairs[idx, 1]], value)
                np.minimum.at(values, self.rod_b[pairs[idx, 1]], value)
        return values

    def _frame(self) -> StateFrame:
        x = self.x
        v = self.v
        contact = self._contact_signed_distance(x)
        external = self.mass[:, None] * self.gravity[None, :]
        return StateFrame(
            time=float(self.time),
            position=np.array(x, dtype=float, copy=True),
            velocity=np.array(v, dtype=float, copy=True),
            mass=np.array(self.mass, dtype=float, copy=True),
            attributes={"mass_quantity": np.array(self.mass_quantity, dtype=float, copy=True)},
            fields={
                "contact_signed_distance": contact,
                "cumulative_mass_source": np.array(self.mass_source, dtype=float, copy=True),
                "external_force": external,
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )

    def diagnostics(self) -> dict:
        try:
            x = self.x
            v = self.v
            contact = self._contact_signed_distance(x)
            speed = np.linalg.norm(v, axis=1)
            inside = np.hypot(x[:, 0], x[:, 1]) <= float(self.profile[-1, 0]) + self.radius
            height = float(np.max(x[inside, 2])) if np.any(inside) else float(np.max(x[:, 2]))
            return {
                "substeps": int(self.substeps),
                "final_time": float(self.time),
                "min_contact_signed_distance": float(np.min(contact)),
                "penetration_sample_count": int(np.count_nonzero(contact < -1e-6)),
                "max_speed": float(np.max(speed)) if speed.size else 0.0,
                "mean_speed": float(np.mean(speed)) if speed.size else 0.0,
                "fill_height": float(height - float(self.profile[0, 1])),
                "kinetic_energy": float(0.5 * np.sum(self.mass * speed * speed)),
                "noodle_count": int(self.n_rods),
                "contact_length_scale": float(self.contact_length_scale),
            }
        except Exception:  # diagnostics must never break the run
            return {}
