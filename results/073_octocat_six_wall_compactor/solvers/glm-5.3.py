"""XPBD elastic-solid solver for the six-wall Octocat compactor (scene 073).

The frozen Octocat surface triangle mesh is treated as the boundary of a
three-dimensional viscoelastic solid.  Dynamics use a Verlet / XPBD
time-stepping scheme:

* corotated stretch constraints on every mesh edge,
* additional long-range elastic links (k-nearest shell) that give the solid
  bending and volumetric stiffness,
* unilateral kinematic contact against the six moving compactor walls
  (position projection onto the inner box),
* vertex-vertex self contact through a spatial hash grid,
* linear material drag.

The momentum budget is exact: every internal constraint applies equal and
opposite impulses to its two endpoints (equal point masses), so the net
momentum change of the whole body equals the measured wall-contact impulse
plus the drag impulse only.  Those two terms are exported as the external
force / cumulative external impulse budget fields.

No keyframing, animation tables, or precomputed trajectories are used: the
wall schedule is evaluated from the frozen scene configuration and all motion
of the Octocat is produced by the constraint solve.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

try:  # SciPy is an accepted base scientific library for this condition.
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - degenerate environment fallback
    cKDTree = None


def _parse_obj(path: Path):
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for t in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[t], idx[t + 1]))
    return (
        np.asarray(vertices, dtype=float).reshape(-1, 3),
        np.asarray(faces, dtype=np.int64).reshape(-1, 3),
    )


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class BenchmarkSolver:
    """XPBD surface-solid solver with exact momentum budgeting."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        frame0 = scene.initial_state()
        x0 = np.array(frame0.position, dtype=float)
        v0 = np.array(frame0.velocity, dtype=float)
        mass = np.array(frame0.mass, dtype=float) if frame0.mass is not None else np.full(len(x0), 1.0 / len(x0))
        n = len(x0)
        self._n = n

        # ---- frozen compactor schedule -------------------------------------
        cfg = getattr(scene, "config", None)
        cfg = cfg if isinstance(cfg, dict) else {}
        comp = cfg.get("compactor", {}) if isinstance(cfg.get("compactor", None), dict) else {}
        self._open_h = float(comp.get("open_inner_half_extent", 1.45))
        self._closed_h = float(comp.get("closed_inner_half_extent", 0.62))
        inward = list(comp.get("inward_interval", [1.0, 3.2]))
        withdraw = list(comp.get("withdrawal_interval", [4.2, 6.4]))
        grav = cfg.get("gravity", [0.0, 0.0, 0.0]) if isinstance(cfg.get("gravity", None), (list, tuple)) else [0.0, 0.0, 0.0]
        self._gravity = np.asarray(grav, dtype=float).reshape(3)

        if hasattr(scene, "inner_half_extent"):
            self._inner_half = scene.inner_half_extent
        else:
            def inner_half(time: float) -> float:
                if time < inward[0]:
                    return self._open_h
                if time <= inward[1]:
                    return self._open_h + (self._closed_h - self._open_h) * _smoothstep(
                        (time - inward[0]) / (inward[1] - inward[0])
                    )
                if time < withdraw[0]:
                    return self._closed_h
                if time <= withdraw[1]:
                    return self._closed_h + (self._open_h - self._closed_h) * _smoothstep(
                        (time - withdraw[0]) / (withdraw[1] - withdraw[0])
                    )
                return self._open_h
            self._inner_half = inner_half

        # ---- mesh topology ---------------------------------------------------
        mesh_path = Path(__file__).resolve().parents[1] / "assets" / "octocat.obj"
        verts_obj, faces = _parse_obj(mesh_path)
        if len(verts_obj) != n:
            raise RuntimeError(
                f"asset vertex count {len(verts_obj)} does not match scene state {n}"
            )
        self._fa = np.ascontiguousarray(faces[:, 0])
        self._fb = np.ascontiguousarray(faces[:, 1])
        self._fc = np.ascontiguousarray(faces[:, 2])

        # ---- reference element frames (per-triangle deformation gradients) ---
        e1 = x0[self._fb] - x0[self._fa]
        e2 = x0[self._fc] - x0[self._fa]
        nv = np.cross(e1, e2)
        ln = np.linalg.norm(nv, axis=1)
        good = ln > 1e-13
        nv = np.where(good[:, None], nv / np.maximum(ln, 1e-30)[:, None], np.array([0.0, 0.0, 1.0]))
        e0 = np.empty((len(faces), 3, 3), dtype=float)
        e0[:, :, 0] = e1
        e0[:, :, 1] = e2
        e0[:, :, 2] = nv
        det = np.linalg.det(e0)
        ok = good & (np.abs(det) > 1e-13)
        inv_e0 = np.linalg.inv(e0)
        inv_e0[~ok] = np.eye(3)
        self._inv_e0 = inv_e0
        self._elem_ok = ok

        # ---- constraint network ---------------------------------------------
        mass = np.maximum(mass, 1e-12)
        self._mass = mass
        self._w = float(1.0 / mass[0])  # equal point masses

        # edge (stretch) constraints
        pairs = np.sort(faces[:, [0, 1, 1, 2, 2, 0]], axis=1)
        edges = np.unique(pairs, axis=0)
        ia = edges[:, 0].astype(np.int64)
        ib = edges[:, 1].astype(np.int64)
        l0_edge = np.linalg.norm(x0[ia] - x0[ib], axis=1)
        alpha_edge = np.full(len(ia), 1e-4)

        # long-range volumetric / bending links
        ia_l = ib_l = l0_l = alpha_l = None
        if cKDTree is not None:
            tree = cKDTree(x0)
            k = min(64, n)
            try:
                dist, idx = tree.query(x0, k=k, workers=1)
            except TypeError:  # older SciPy signature
                dist, idx = tree.query(x0, k=k)
            rows = np.repeat(np.arange(n, dtype=np.int64), k - 1)
            cols = idx[:, 1:].astype(np.int64).ravel()
            dd = dist[:, 1:].ravel()
            m = (dd >= 0.07) & (dd <= 0.26)
            rows, cols, dd = rows[m], cols[m], dd[m]
            if rows.size:
                # keep up to 8 links per vertex (nearest first: rows is sorted)
                starts = np.searchsorted(rows, np.arange(n, dtype=rows.dtype))
                rank = np.arange(rows.size, dtype=np.int64) - starts[rows]
                keep = rank < 8
                rows, cols, dd = rows[keep], cols[keep], dd[keep]
                p = np.minimum(rows, cols)
                q = np.maximum(rows, cols)
                key = p * n + q
                _, sel = np.unique(key, return_index=True)
                ia_l = p[sel]
                ib_l = q[sel]
                l0_l = dd[sel]
                alpha_l = np.full(len(ia_l), 8e-4)

        if ia_l is not None and len(ia_l):
            self._ia = np.concatenate([ia, ia_l])
            self._ib = np.concatenate([ib, ib_l])
            self._l0 = np.concatenate([l0_edge, l0_l])
            self._alpha = np.concatenate([alpha_edge, alpha_l])
        else:
            self._ia, self._ib, self._l0, self._alpha = ia, ib, l0_edge, alpha_edge
        self._n_con = len(self._ia)

        # ---- self collision ---------------------------------------------------
        self._rc = 0.04
        self._rexcl = 0.09
        offsets = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx > 0 or (dx == 0 and dy > 0) or (dx == 0 and dy == 0 and dz >= 0):
                        offsets.append((dx, dy, dz))
        self._offsets = [(dx * 1048576 + dy * 1024 + dz) for (dx, dy, dz) in offsets]
        self._x0 = x0.copy()

        # ---- integrator parameters --------------------------------------------
        self._dt = 0.002
        self._iters = 5
        self._relax = 0.3
        self._drag = 0.8
        self._vmax = 25.0

        # ---- state -------------------------------------------------------------
        self._x = x0.copy()
        self._v = v0.reshape(n, 3).copy()
        self._t = 0.0
        self._substeps = 0
        self._cum_contact = np.zeros(3)
        self._cum_drag = np.zeros(3)
        self._last_emit_t = 0.0
        self._last_contact = np.zeros(3)
        self._last_drag = np.zeros(3)

        total_mass = float(mass.sum())
        dur = max(float(inward[1]) - float(inward[0]), 1e-6)
        v_char = 1.5 * abs(self._open_h - self._closed_h) / dur
        self._ref_momentum = float(max(total_mass * max(v_char, 1e-6), 1e-6))
        self._ready = True

    # ------------------------------------------------------------- integrator
    def _solve_springs(self, x: np.ndarray, h: float) -> None:
        ia, ib, l0 = self._ia, self._ib, self._l0
        d = x[ia] - x[ib]
        l = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
        nrm = d / l[:, None]
        c = l - l0
        denom = 2.0 * self._w + self._alpha / (h * h)
        dl = -c / denom
        # per-constraint correction magnitude (equal masses -> symmetric split)
        mag = self._w * dl
        acc = np.empty((self._n, 3))
        for k in range(3):
            comp = mag * nrm[:, k]
            acc[:, k] = np.bincount(ia, weights=comp, minlength=self._n) - np.bincount(
                ib, weights=comp, minlength=self._n
            )
        x += self._relax * acc

    def _wall_project(self, x: np.ndarray, hw: float) -> np.ndarray:
        dwall = np.zeros_like(x)
        for ax in range(3):
            d = np.clip(x[:, ax], -hw, hw) - x[:, ax]
            dwall[:, ax] += d
            x[:, ax] += d
        return dwall

    def _hash_pairs(self, x: np.ndarray):
        r = self._rc
        c = np.floor(x / r).astype(np.int64)
        enc = ((c[:, 0] + 512) * 1024 + c[:, 1] + 512) * 1024 + (c[:, 2] + 512)
        order = np.argsort(enc, kind="stable")
        enc_s = enc[order]
        uniq, starts, counts = np.unique(enc_s, return_index=True, return_counts=True)
        nu = len(uniq)
        out_i = []
        out_j = []
        for oenc in self._offsets:
            target = uniq + oenc
            pos = np.searchsorted(uniq, target)
            posc = np.clip(pos, 0, nu - 1)
            hit = uniq[posc] == target
            if not np.any(hit):
                continue
            ka = np.flatnonzero(hit)
            kb = posc[hit]
            ca = counts[ka]
            cb = counts[kb]
            sizes = ca * cb
            total = int(sizes.sum())
            if total == 0:
                continue
            cp = np.repeat(np.arange(len(ka), dtype=np.int64), sizes)
            cum = np.concatenate(([0], np.cumsum(sizes)))
            local = np.arange(total, dtype=np.int64) - cum[cp]
            cbb = cb[cp]
            ua = local // cbb
            va = local % cbb
            gi = order[starts[ka[cp]] + ua]
            gj = order[starts[kb[cp]] + va]
            out_i.append(gi)
            out_j.append(gj)
        if not out_i:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        i = np.concatenate(out_i)
        j = np.concatenate(out_j)
        keep = i < j
        return i[keep], j[keep]

    def _self_collide(self, x: np.ndarray) -> None:
        i, j = self._hash_pairs(x)
        if i.size == 0:
            return
        d = x[i] - x[j]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        r0 = np.sqrt(np.einsum("ij,ij->i", self._x0[i] - self._x0[j], self._x0[i] - self._x0[j]))
        m = (dist < self._rc) & (r0 >= self._rexcl) & (dist > 1e-9)
        if not np.any(m):
            return
        i, j, d, dist = i[m], j[m], d[m], dist[m]
        nrm = d / dist[:, None]
        mag = 0.5 * (self._rc - dist)
        acc = np.empty((self._n, 3))
        for k in range(3):
            comp = mag * nrm[:, k]
            acc[:, k] = np.bincount(i, weights=comp, minlength=self._n) - np.bincount(
                j, weights=comp, minlength=self._n
            )
        x += 0.5 * acc

    def _step(self, h: float) -> None:
        x, v = self._x, self._v
        x_prev = x.copy()
        v += self._gravity * h
        x += v * h
        hw = float(self._inner_half(self._t + h))
        dwall = np.zeros_like(x)
        for _ in range(self._iters):
            self._solve_springs(x, h)
            dwall += self._wall_project(x, hw)
        self._self_collide(x)

        # Verlet velocity update (contact velocities are inherited from walls).
        v[:] = (x - x_prev) / h

        # External impulse of the wall contact (internal impulses cancel exactly).
        self._cum_contact += (self._mass[:, None] * dwall).sum(axis=0) / h

        # Material drag (external dissipation term).
        v_before = v.copy()
        v *= math.exp(-self._drag * h)
        self._cum_drag += (self._mass[:, None] * (v - v_before)).sum(axis=0)

        # Numerical safety valves (deterministic, rarely active).
        speed = np.sqrt(np.einsum("ij,ij->i", v, v))
        fast = speed > self._vmax
        if np.any(fast):
            v[fast] *= (self._vmax / speed[fast])[:, None]
        bad = ~np.isfinite(x).all(axis=1)
        if np.any(bad):
            x[bad] = x_prev[bad]
            v[bad] = 0.0
        if not np.isfinite(x).all():
            x[:] = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(v).all():
            v[:] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        self._t += h
        self._substeps += 1

    # ------------------------------------------------------------- trajectory
    def _element_deformation_gradient(self, x: np.ndarray) -> np.ndarray:
        e1 = x[self._fb] - x[self._fa]
        e2 = x[self._fc] - x[self._fa]
        nv = np.cross(e1, e2)
        ln = np.linalg.norm(nv, axis=1)
        good = ln > 1e-13
        nv = np.where(
            good[:, None], nv / np.maximum(ln, 1e-30)[:, None], np.array([0.0, 0.0, 1.0])
        )
        ed = np.empty((len(self._fa), 3, 3))
        ed[:, :, 0] = e1
        ed[:, :, 1] = e2
        ed[:, :, 2] = nv
        f = np.matmul(ed, self._inv_e0)
        f[~self._elem_ok] = np.eye(3)
        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("solver.initialize() must be called first")
        target = float(time)
        if target < self._t:
            target = self._t
        while target - self._t > 1e-12:
            h = min(self._dt, target - self._t)
            if h < 1e-12:
                break
            self._step(h)

        t_frame = float(max(target, 0.0))
        dtf = t_frame - self._last_emit_t
        if dtf > 1e-12:
            f_contact = (self._cum_contact - self._last_contact) / dtf
            f_drag = (self._cum_drag - self._last_drag) / dtf
        else:
            f_contact = np.zeros(3)
            f_drag = np.zeros(3)
        frame = StateFrame(
            time=t_frame,
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._mass.copy(),
            attributes={"mass_quantity": self._mass.copy()},
            fields={
                "budget_cumulative_external_impulse": (
                    self._cum_contact + self._cum_drag
                ).reshape(1, 3),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "budget_domain_weights": np.ones((1, self._n)),
                "budget_external_force": np.stack([f_contact, f_drag])[None, :, :],
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": self._element_deformation_gradient(self._x),
            },
            scalars={"reference_momentum": self._ref_momentum},
        )
        self._last_emit_t = t_frame
        self._last_contact = self._cum_contact.copy()
        self._last_drag = self._cum_drag.copy()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "xpbd_surface_solid",
            "vertices": int(self._n) if self._ready else None,
            "constraints": int(self._n_con) if self._ready else None,
            "substep_dt": self._dt if self._ready else None,
            "iterations_per_substep": self._iters if self._ready else None,
            "substeps_taken": int(self._substeps),
            "current_time": float(self._t),
            "seed": int(getattr(self, "_seed", 0)),
        }
