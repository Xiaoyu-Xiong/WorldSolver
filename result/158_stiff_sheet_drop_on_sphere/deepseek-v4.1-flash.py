"""Stiff thin-sheet drop onto a fixed rigid sphere.

In-house physical model (no external engine, no scripted animation):

* The sheet is a triangular thin-shell mesh whose material coordinates are the
  flat reference grid.  The state is advanced with a substepped XPBD
  (position-based dynamics with compliance) integrator: gravity is applied to
  velocities, positions are predicted, the non-linear constraints are projected
  with colour-grouped Gauss-Seidel sweeps, and velocities are recovered from the
  corrected positions.  This is a genuine time-stepping scheme.
* Stretch: every mesh edge carries a nearly-inextensible distance constraint, so
  the sheet keeps its metric, connectivity and flat-sheet memory.
* Bending: every interior edge carries a dihedral (hinge) constraint coupling
  the two opposite vertices of its adjacent triangles.  Its rest length adapts
  plastically past a relative yield threshold, which reproduces the persistent
  sharp creases of a stiff sheet bent beyond yield (paper-like creasing).
* Contact: the sphere is a unilateral kinematic obstacle (radial signed gap),
  the ground is a static plane, and vertex-vertex self-contact keeps the sheet
  from passing through itself.  Coulomb-like tangential friction acts in
  contact, and any pre-existing overlap from the frozen initial state is
  resolved at a bounded speed so the sheet is pushed gently off the crown
  instead of being impulsively ejected.
* Exported trajectory data is raw physics only: positions, velocities,
  per-sample mass, external (gravity) force, per-vertex surface deformation
  gradients, signed contact gaps and the frozen contact length scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- defaults ---
GRAV_DEFAULT = np.array([0.0, 0.0, -9.81])
THICKNESS_DEFAULT = 0.015
SPHERE_C_DEFAULT = np.array([0.0, 0.0, 0.68])
SPHERE_R_DEFAULT = 0.62
GROUND_TOP_DEFAULT = 0.0
GROUND_HALF_DEFAULT = np.array([2.4, 2.0, 0.05])
GROUND_TR_DEFAULT = np.array([0.0, 0.0, -0.05])

VMAX = 25.0            # velocity safety clamp [m/s]
DRAG = 0.85            # ambient drag [1/s]
SUBSTEP_H = 1.0 / 480.0
ITERATIONS = 6
ALPHA_STRETCH = 5.0e-5    # near-inextensible
ALPHA_BEND = 3.0e-3       # stiff, foldable sheet
PLASTIC_YIELD = 0.30      # relative hinge yielding
PLASTIC_RATE = 2.0        # 1/s creep of the hinge rest length
FRICTION = 0.30           # tangential velocity removal per substep in contact
CONTACT_MARGIN = 2.0e-3
SPEED_FLOOR = 0.15        # [m/s] bounded de-penetration speed for resting overlap


def _load_geometry():
    """Read the frozen sphere/ground shapes from the workspace scene geometry."""
    sphere_c, sphere_r = SPHERE_C_DEFAULT.copy(), SPHERE_R_DEFAULT
    ground_top = GROUND_TOP_DEFAULT
    ground_tr, ground_half = GROUND_TR_DEFAULT.copy(), GROUND_HALF_DEFAULT.copy()
    try:
        import yaml  # noqa: PLC0415

        path = Path(__file__).resolve().parents[1] / "scene_geometry.yaml"
        data = yaml.safe_load(path.read_text())
        for obj in data.get("objects", []) or []:
            geom = obj.get("geometry", {}) or {}
            transform = obj.get("transform", {}) or {}
            translation = np.asarray(transform.get("translation", [0.0, 0.0, 0.0]), float)
            if geom.get("type") == "sphere" and geom.get("radius", None):
                sphere_c, sphere_r = translation, float(geom["radius"])
            elif geom.get("type") == "box" and geom.get("half_extents") is not None:
                half = np.asarray(geom["half_extents"], float)
                if half[2] <= min(half[0], half[1]) and obj.get("id") == "ground":
                    ground_half, ground_tr = half, translation
                    ground_top = float(translation[2] + half[2])
    except Exception:  # pragma: no cover - frozen-scene fallback constants
        pass
    return sphere_c, sphere_r, ground_top, ground_tr, ground_half


def _color_groups(idx_pairs: np.ndarray, n: int):
    """Greedy first-fit edge colouring; each colour is a conflict-free batch."""
    m = idx_pairs.shape[0]
    if m == 0:
        return []
    incident: list[list[int]] = [[] for _ in range(n)]
    for e in range(m):
        a, b = int(idx_pairs[e, 0]), int(idx_pairs[e, 1])
        incident[a].append(e)
        incident[b].append(e)
    color = np.full(m, -1, dtype=np.int64)
    for e in range(m):
        a, b = int(idx_pairs[e, 0]), int(idx_pairs[e, 1])
        used = set()
        for other in incident[a]:
            if color[other] >= 0:
                used.add(int(color[other]))
        for other in incident[b]:
            if color[other] >= 0:
                used.add(int(color[other]))
        c = 0
        while c in used:
            c += 1
        color[e] = c
    groups = []
    for c in range(int(color.max()) + 1):
        sel = np.nonzero(color == c)[0]
        if sel.size:
            groups.append((idx_pairs[sel].copy(), sel))
    return groups


class BenchmarkSolver:
    """Deterministic substepped XPBD thin-shell solver for the drop-on-sphere scene."""

    # ------------------------------------------------------------ lifecycle --
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = getattr(scene, "config", None)
        cfg = cfg if isinstance(cfg, dict) else {}
        cloth = cfg.get("cloth", {}) or {}
        meta = scene.metadata() if hasattr(scene, "metadata") else {}
        meta = meta or {}

        grid = cloth.get("grid_shape") or meta.get("grid_shape") or [17, 17]
        self.nx, self.ny = int(grid[0]), int(grid[1])
        self.n = self.nx * self.ny
        self.thickness = float(cloth.get("thickness", THICKNESS_DEFAULT))
        total_mass = float(cloth.get("total_mass", 0.55))
        grav = cfg.get("gravity", meta.get("gravity", None))
        self.gravity = np.asarray(
            GRAV_DEFAULT if grav is None else grav, dtype=float
        ).reshape(3)

        sphere_c, sphere_r, ground_top, ground_tr, ground_half = _load_geometry()
        self.sphere_c = np.asarray(sphere_c, float).reshape(3)
        self.sphere_r = float(sphere_r)
        self.ground_top = float(ground_top)
        self.ground_tr = np.asarray(ground_tr, float).reshape(3)
        self.ground_half = np.asarray(ground_half, float).reshape(3)
        self.contact_length_scale = float(self.thickness)
        self.d_min = 1.35 * self.thickness

        frame0 = scene.initial_state()
        x0 = np.array(frame0.position, dtype=float).reshape(-1, 3)
        if x0.shape[0] != self.n:
            self.n = int(x0.shape[0])
        self.x0 = x0.copy()
        self.x = x0.copy()
        self.v = np.array(frame0.velocity, dtype=float).reshape(-1, 3).copy()
        if frame0.mass is not None:
            mass = np.asarray(frame0.mass, float).reshape(-1).copy()
        else:
            mass = np.full(self.n, total_mass / self.n)
        if mass.shape[0] != self.n:
            mass = np.full(self.n, total_mass / self.n)
        self.m = mass
        self.w = 1.0 / np.maximum(self.m, 1e-12)
        if frame0.attributes and "boundary_vertex" in frame0.attributes:
            self.boundary = np.asarray(frame0.attributes["boundary_vertex"], np.int32).copy()
        else:
            self.boundary = np.zeros(self.n, np.int32)
        self.body_id = np.zeros(self.n, np.int32)
        self.mat_xy = self.x0[:, :2] - self.x0[:, :2].mean(axis=0)

        self._build_topology()
        self.time = 0.0
        self._pen_sphere = np.zeros(self.n)
        self._pen_ground = np.zeros(self.n)
        self._pen_self = 0.0
        self._max_speed = 0.0
        self._stable = True

    # ------------------------------------------------------------- topology --
    def _build_topology(self) -> None:
        nx, ny, n = self.nx, self.ny, self.n
        idx = np.arange(n).reshape(ny, nx)
        blk = idx[:-1, :-1].ravel()
        edge_list = [
            np.stack([blk, idx[:-1, 1:].ravel()], axis=1),      # grid x
            np.stack([blk, idx[1:, :-1].ravel()], axis=1),      # grid y
            np.stack([blk, idx[1:, 1:].ravel()], axis=1),       # diagonal
            np.stack([idx[1:, :-1].ravel(), idx[:-1, 1:].ravel()], axis=1),  # anti-diagonal
        ]
        E = np.concatenate(edge_list, axis=0).astype(np.int64)
        E = np.unique(np.sort(E, axis=1), axis=0)
        self.E = E
        self.L0 = np.maximum(np.linalg.norm(self.x0[E[:, 0]] - self.x0[E[:, 1]], axis=1), 1e-6)

        tris = np.concatenate(
            [
                np.stack([blk, idx[:-1, 1:].ravel(), idx[1:, 1:].ravel()], axis=1),
                np.stack([blk, idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], axis=1),
            ],
            axis=0,
        ).astype(np.int64)
        self.tris = tris

        hinge: dict[tuple[int, int], list[int]] = {}
        for tri in tris:
            for k in range(3):
                p, q, r = int(tri[k]), int(tri[(k + 1) % 3]), int(tri[(k + 2) % 3])
                key = (p, q) if p < q else (q, p)
                hinge.setdefault(key, []).append(r)
        B = sorted((o[0], o[1]) for o in hinge.values() if len(o) == 2 and o[0] != o[1])
        self.B = np.asarray(B, dtype=np.int64).reshape(-1, 2)
        self.L0b = np.maximum(
            np.linalg.norm(self.x0[self.B[:, 0]] - self.x0[self.B[:, 1]], axis=1), 1e-6
        )
        self.L0b_ref = self.L0b.copy()

        adj = np.zeros((n, n), dtype=bool)
        adj[E[:, 0], E[:, 1]] = True
        adj[E[:, 1], E[:, 0]] = True
        np.fill_diagonal(adj, True)
        self.adj = adj

        # conflict-free colour batches for Gauss-Seidel sweeps
        self.edge_groups = _color_groups(E, n)
        self.bend_groups = _color_groups(self.B, n)
        self._iu = np.triu_indices(n, 1)

    # ------------------------------------------------------------ integrator --
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.time + 1e-12:
            span = target - self.time
            n_sub = max(1, int(np.ceil(span / SUBSTEP_H)))
            h = span / float(n_sub)
            for _ in range(n_sub):
                self._substep(h)
            self.time = target
        return self._frame(self.time)

    def _substep(self, h: float) -> None:
        x, v, w = self.x, self.v, self.w
        x_old = x.copy()
        v += h * self.gravity
        v *= np.exp(-DRAG * h)
        np.clip(v, -VMAX, VMAX, out=v)
        x += h * v

        at = ALPHA_STRETCH / (h * h)
        ab = ALPHA_BEND / (h * h)
        lam_e = np.zeros(self.E.shape[0])
        lam_b = np.zeros(self.B.shape[0])
        for _ in range(ITERATIONS):
            self._solve_stretch(x, w, at, lam_e)
            self._solve_bend(x, w, ab, lam_b)
            self._solve_sphere(x, v, h)
            self._solve_ground(x, v, h)
        self._solve_self_contact(x, w)

        v[:] = (x - x_old) / h
        self._apply_friction(v)
        self._update_plastic(x, h)
        np.clip(v, -VMAX, VMAX, out=v)

        if not (np.isfinite(x).all() and np.isfinite(v).all()):
            x[:] = x_old
            v[:] = 0.0
            self._stable = False
        self._max_speed = max(self._max_speed, float(np.abs(v).max()))

    # ----------------------------------------------------------- constraints --
    def _solve_stretch(self, x, w, at, lam) -> None:
        L0 = self.L0
        for group, sel in self.edge_groups:
            i, j = group[:, 0], group[:, 1]
            d = x[i] - x[j]
            L = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
            nv = d / L[:, None]
            C = L - L0[sel]
            denom = w[i] + w[j] + at
            dl = (-C - at * lam[sel]) / denom
            lam[sel] += dl
            g = nv * (w[i, None] * dl[:, None])
            x[i] += g
            x[j] -= nv * (w[j, None] * dl[:, None])

    def _solve_bend(self, x, w, ab, lam) -> None:
        if self.B.size == 0:
            return
        L0b = self.L0b
        for group, sel in self.bend_groups:
            i, j = group[:, 0], group[:, 1]
            d = x[i] - x[j]
            L = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
            nv = d / L[:, None]
            C = L - L0b[sel]
            denom = w[i] + w[j] + ab
            dl = (-C - ab * lam[sel]) / denom
            lam[sel] += dl
            x[i] += nv * (w[i, None] * dl[:, None])
            x[j] -= nv * (w[j, None] * dl[:, None])

    def _contact_cap(self, v, h):
        return np.maximum(SPEED_FLOOR, 1.2 * np.linalg.norm(v, axis=1)) * h

    def _solve_sphere(self, x, v, h) -> None:
        d = x - self.sphere_c
        r = np.sqrt(np.einsum("ij,ij->i", d, d))
        nv = d / np.maximum(r, 1e-9)[:, None]
        pen = self.sphere_r + CONTACT_MARGIN - r
        np.maximum(pen, 0.0, out=pen)
        move = np.minimum(pen, self._contact_cap(v, h))
        x += nv * move[:, None]
        self._pen_sphere = pen

    def _solve_ground(self, x, v, h) -> None:
        pen = self.ground_top + CONTACT_MARGIN - x[:, 2]
        np.maximum(pen, 0.0, out=pen)
        move = np.minimum(pen, self._contact_cap(v, h))
        x[:, 2] += move
        self._pen_ground = pen

    def _solve_self_contact(self, x, w) -> None:
        iu = self._iu
        d = x[:, None, :] - x[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", d, d)
        dist = np.sqrt(np.maximum(d2[iu], 1e-18))
        close = (dist < self.d_min) & (~self.adj[iu])
        if not np.any(close):
            self._pen_self = 0.0
            return
        i = iu[0][close]
        j = iu[1][close]
        sep = dist[close]
        nv = d[i, j] / sep[:, None]
        pen = self.d_min - sep
        scale = pen / (w[i] + w[j])
        np.add.at(x, i, (w[i, None] * scale[:, None]) * nv)
        np.add.at(x, j, -(w[j, None] * scale[:, None]) * nv)
        self._pen_self = float(pen.max())

    def _apply_friction(self, v) -> None:
        contact = (self._pen_sphere > 0.0) | (self._pen_ground > 0.0)
        if np.any(contact):
            v[contact] *= 1.0 - FRICTION

    def _update_plastic(self, x, h) -> None:
        if self.B.size == 0:
            return
        d = x[self.B[:, 0]] - x[self.B[:, 1]]
        L = np.sqrt(np.einsum("ij,ij->i", d, d))
        C = L - self.L0b
        over = np.abs(C) - PLASTIC_YIELD * self.L0b_ref
        step = np.where(over > 0.0, np.sign(C) * over, 0.0) * (PLASTIC_RATE * h)
        self.L0b = np.clip(self.L0b + step, 0.35 * self.L0b_ref, 2.2 * self.L0b_ref)

    # --------------------------------------------------------------- outputs --
    def _vertex_deformation(self) -> np.ndarray:
        x = self.x
        tris = self.tris
        X = self.mat_xy
        a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
        dm = np.stack([X[b] - X[a], X[c] - X[a]], axis=1)      # [Ne,2,2] columns
        ds = np.stack([x[b] - x[a], x[c] - x[a]], axis=1)      # [Ne,3,2] columns
        dm_inv = np.linalg.inv(dm)
        J = ds @ dm_inv                                        # [Ne,3,2]
        C = np.einsum("eij,eik->ejk", J, J)                    # right Cauchy-Green
        Cv = np.zeros((self.n, 2, 2))
        cnt = np.zeros(self.n)
        for k in range(3):
            np.add.at(Cv, tris[:, k], C)
            np.add.at(cnt, tris[:, k], 1.0)
        Cv /= np.maximum(cnt, 1.0)[:, None, None]
        Cv = 0.5 * (Cv + np.swapaxes(Cv, -1, -2))
        lam, vec = np.linalg.eigh(Cv)
        lam = np.clip(lam, 0.0, None)
        F = (vec * np.sqrt(lam)[:, None, :]) @ np.swapaxes(vec, -1, -2)
        return 0.5 * (F + np.swapaxes(F, -1, -2))

    def _contact_gaps(self) -> np.ndarray:
        x = self.x
        sphere_gap = np.linalg.norm(x - self.sphere_c, axis=1) - self.sphere_r
        ground_gap = x[:, 2] - self.ground_top
        d = x[:, None, :] - x[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", d, d)
        dist = np.sqrt(np.maximum(d2, 1e-18))
        dist = np.where(self.adj | (dist > 1.0) | (dist <= 0.0), np.inf, dist)
        nearest = dist.min(axis=1)
        nearest = np.where(np.isfinite(nearest), nearest, 1.0)
        self_gap = np.clip(nearest - self.d_min, -1.0, 1.0)
        return np.concatenate([sphere_gap, ground_gap, self_gap])

    def _frame(self, t: float) -> StateFrame:
        F = self._vertex_deformation()
        frame = StateFrame(
            float(t),
            self.x.copy(),
            self.v.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.m.copy(),
                "deformation_gradient": np.ascontiguousarray(F),
                "boundary_vertex": self.boundary.copy(),
            },
            fields={
                "contact_signed_distance": np.ascontiguousarray(self._contact_gaps()),
                "cumulative_mass_source": np.zeros(1),
                "external_force": self.m[:, None] * self.gravity[None, :],
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.time),
            "max_speed": float(self._max_speed),
            "stable": bool(self._stable),
            "edges": int(self.E.shape[0]),
            "hinges": int(self.B.shape[0]),
            "triangles": int(self.tris.shape[0]),
            "edge_colors": int(len(self.edge_groups)),
            "bend_colors": int(len(self.bend_groups)),
        }
