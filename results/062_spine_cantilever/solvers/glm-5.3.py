"""Corotated finite-element solver for scene 062_spine_cantilever.

The frozen surface asset is a chain of 21 overlapping box segments (a broad
anchored root followed by ten vertebrae joined by thin flexible necks).  This
solver builds a conforming tetrahedral solid mesh over the union of those
boxes: thin neck sections keep a coarse grid, body sections carry extra
mid-surface nodes so the mesh ties conformingly at every segment interface.

The solid is stepped with explicit (symplectic) integration of a corotated
elastic constitutive model using the frozen Young's modulus, Poisson ratio
and density.  The root block (x <= frozen x_max) is clamped and gravity acts
from t = 0, so the unsupported arm sags downward while successive sections
bend at the thin connections.
"""
from __future__ import annotations

import numpy as np

try:  # platform contract type
    from worldsolver.contracts import StateFrame
except Exception:  # pragma: no cover - minimal stand-in with the same fields
    from dataclasses import dataclass, field
    from typing import Mapping

    @dataclass
    class StateFrame:  # type: ignore[no-redef]
        time: float
        position: object
        velocity: object
        orientation: object = None
        angular_velocity: object = None
        mass: object = None
        body_id: object = None
        attributes: Mapping = field(default_factory=dict)
        fields: Mapping = field(default_factory=dict)
        scalars: Mapping = field(default_factory=dict)

KEY_DECIMALS = 7


def _round_key(p) -> tuple:
    return (
        round(float(p[0]), KEY_DECIMALS),
        round(float(p[1]), KEY_DECIMALS),
        round(float(p[2]), KEY_DECIMALS),
    )


def _cross3(a, b):
    out = np.empty_like(a)
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    out[..., 0] = a1 * b2 - a2 * b1
    out[..., 1] = a2 * b0 - a0 * b2
    out[..., 2] = a0 * b1 - a1 * b0
    return out


def _cof(F):
    """Cofactor matrix of a stack of 3x3 matrices (equals J * F^-T)."""
    c1, c2, c3 = F[..., 0], F[..., 1], F[..., 2]
    cof = np.empty_like(F)
    cof[..., 0] = _cross3(c2, c3)
    cof[..., 1] = _cross3(c3, c1)
    cof[..., 2] = _cross3(c1, c2)
    return cof


def _det(F):
    c1, c2, c3 = F[..., 0], F[..., 1], F[..., 2]
    return np.einsum("...i,...i->...", c1, _cross3(c2, c3))


def _polar(F, iterations=5):
    """Batched orthogonal polar factor via Newton iteration."""
    R = F.copy()
    for _ in range(iterations):
        d = _det(R)
        R = 0.5 * (R + _cof(R) / d[..., None, None])
    return R


class BenchmarkSolver:
    """Explicit corotated FEM for the segmented rubber spine cantilever."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        meta = scene.metadata()
        config = getattr(scene, "config", None) or {}
        self.gravity = np.asarray(meta["gravity"], dtype=float)
        material = meta["material"]
        self.E = float(material["youngs_modulus_pa"])
        self.nu = float(material["poisson_ratio"])
        self.rho = float(material["density"])
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = (
            self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
            if self.nu < 0.5
            else 10.0 * self.mu
        )
        self.x_max_fixed = float(meta["fixed_region"]["x_max"])
        self.xi_d = 1.0  # mass-proportional (viscoelastic) damping, 1/s

        initial = scene.initial_state()
        obj = np.array(initial.position, dtype=float)
        self.obj_local = obj.copy()
        self.offset = self._scene_offset(config)
        self.obj_world = obj + self.offset[None, :]

        boxes = self._boxes(obj)
        mesh = self._build_mesh(boxes)
        (
            self.x0,
            self.tets,
            self.mapping,
            self.node_fixed,
        ) = mesh

        # rest-shape element data
        xt = self.x0[self.tets]
        Dm = np.transpose(xt[:, 1:] - xt[:, :1], (0, 2, 1))
        V0 = _det(Dm) / 6.0
        flip = V0 < 0.0
        if np.any(flip):
            t = self.tets.copy()
            t[flip, 2], t[flip, 3] = t[flip, 3].copy(), t[flip, 2].copy()
            self.tets = t
            xt = self.x0[self.tets]
            Dm = np.transpose(xt[:, 1:] - xt[:, :1], (0, 2, 1))
            V0 = _det(Dm) / 6.0
        if np.any(V0 <= 1e-12):
            raise ValueError("degenerate tetrahedron in generated solid mesh")
        self.V0 = V0
        det_Dm = _det(Dm)
        self.Dm_inv = _cof(Dm) / det_Dm[..., None, None]
        self.Dm_invT = np.transpose(self.Dm_inv, (0, 2, 1))

        self.nnodes = len(self.x0)
        self.tets_flat = self.tets.reshape(-1)
        self.m = np.bincount(
            np.repeat(np.arange(len(self.tets)), 4),
            weights=self.rho * V0 / 4.0,
            minlength=self.nnodes,
        )
        self.node_fixed = (self.x0[:, 0] <= self.x_max_fixed + 1e-9).astype(bool)

        # clamped surface samples (same rule as the frozen boundary condition)
        self.sample_fixed = (self.obj_world[:, 0] <= self.x_max_fixed + 1e-9).astype(float)

        # explicit stability limit from the dilatational wave speed
        h_min = self._min_edge(self.x0, self.tets)
        c_wave = np.sqrt((self.lam + 2.0 * self.mu) / self.rho)
        self.dt = float(0.3 * h_min / c_wave)

        self.sample_mass = np.full(len(obj), self.m.sum() / len(obj), dtype=float)
        self.total_mass = float(self.m.sum())
        self.reference_force = float(self.total_mass * np.linalg.norm(self.gravity))

        self._reset()
        self._ready = True

    # ------------------------------------------------------------- utilities
    @staticmethod
    def _scene_offset(config) -> np.ndarray:
        geometry = config.get("scene_geometry") if isinstance(config, dict) else None
        objects = geometry.get("objects", []) if isinstance(geometry, dict) else []
        for item in objects:
            if isinstance(item, dict) and item.get("id") == "spine_arm":
                translation = (item.get("transform") or {}).get("translation")
                if translation:
                    return np.asarray(translation, dtype=float)
        return np.zeros(3)

    @staticmethod
    def _boxes(obj):
        count = obj.shape[0]
        if count % 8:
            raise ValueError("unexpected spine surface vertex count")
        boxes = []
        for k in range(count // 8):
            chunk = obj[8 * k : 8 * k + 8]
            xs = np.unique(np.round(chunk[:, 0], 6))
            ys = np.unique(np.round(chunk[:, 1], 6))
            zs = np.unique(np.round(chunk[:, 2], 6))
            if len(xs) != 2 or len(ys) != 2 or len(zs) != 2:
                raise ValueError("spine surface is not a chain of box segments")
            boxes.append(
                {
                    "x0": float(xs[0]),
                    "x1": float(xs[1]),
                    "hy": 0.5 * float(ys[1] - ys[0]),
                    "hz": 0.5 * float(zs[1] - zs[0]),
                }
            )
        return boxes

    def _build_mesh(self, boxes):
        count = len(boxes)
        thin = []
        for k, box in enumerate(boxes):
            is_thin = False
            for j in (k - 1, k + 1):
                if 0 <= j < count:
                    other = boxes[j]
                    if box["hy"] < other["hy"] - 1e-9 and box["hz"] < other["hz"] - 1e-9:
                        is_thin = True
            thin.append(is_thin)

        # segment interfaces sit on the left face of the following box, so the
        # neighbouring segments overlap exactly like the frozen geometry and
        # the mesh tiles the union without gaps or double counting.
        left = [box["x0"] for box in boxes]
        right = [boxes[k + 1]["x0"] for k in range(count - 1)] + [boxes[-1]["x1"]]
        for k in range(count):
            if right[k] <= left[k] + 0.005 or right[k] > boxes[k]["x1"] + 1e-9:
                raise ValueError("spine segments do not form an overlapping chain")

        def thin_values(k, side):
            j = k - 1 if side == "left" else k + 1
            if 0 <= j < count and thin[j]:
                return boxes[j]["hy"], boxes[j]["hz"]
            return None

        nodes = []
        node_index = {}

        def node_id(p):
            key = _round_key(p)
            idx = node_index.get(key)
            if idx is None:
                idx = len(nodes)
                nodes.append([float(p[0]), float(p[1]), float(p[2])])
                node_index[key] = idx
            return idx

        tets = []
        for k, box in enumerate(boxes):
            x_l, x_r = left[k], right[k]
            ncell = max(1, int(round((x_r - x_l) / 0.03)))
            xs = np.linspace(x_l, x_r, ncell + 1)
            if thin[k]:
                y_lines = [
                    [-box["hy"], 0.0, box["hy"]],
                ] * (ncell + 1)
                z_lines = [[-box["hz"], box["hz"]]] * (ncell + 1)
            else:
                tl = thin_values(k, "left")
                tr = thin_values(k, "right")
                hyt_l = tl[0] if tl else (tr[0] if tr else 0.5 * box["hy"])
                hyt_r = tr[0] if tr else hyt_l
                hzt_l = tl[1] if tl else (tr[1] if tr else 0.5 * box["hz"])
                hzt_r = tr[1] if tr else hzt_l
                hyt_l = min(hyt_l, 0.9 * box["hy"])
                hyt_r = min(hyt_r, 0.9 * box["hy"])
                hzt_l = min(hzt_l, 0.9 * box["hz"])
                hzt_r = min(hzt_r, 0.9 * box["hz"])
                y_lines = []
                z_lines = []
                for p in range(ncell + 1):
                    s = p / ncell
                    hyt = hyt_l + (hyt_r - hyt_l) * s
                    hzt = hzt_l + (hzt_r - hzt_l) * s
                    y_lines.append([-box["hy"], -hyt, 0.0, hyt, box["hy"]])
                    z_lines.append([-box["hz"], -hzt, hzt, box["hz"]])
            grids = []
            for p in range(ncell + 1):
                plane = []
                for y in y_lines[p]:
                    row = []
                    for z in z_lines[p]:
                        row.append(node_id((xs[p], y, z)))
                    plane.append(row)
                grids.append(plane)
            for p in range(ncell):
                bottom, top = grids[p], grids[p + 1]
                for iy in range(len(y_lines[p]) - 1):
                    for iz in range(len(z_lines[p]) - 1):
                        hexa = (
                            bottom[iy][iz],
                            bottom[iy + 1][iz],
                            bottom[iy + 1][iz + 1],
                            bottom[iy][iz + 1],
                            top[iy][iz],
                            top[iy + 1][iz],
                            top[iy + 1][iz + 1],
                            top[iy][iz + 1],
                        )
                        for tet in (
                            (0, 1, 2, 6),
                            (0, 1, 5, 6),
                            (0, 4, 5, 6),
                            (0, 4, 6, 7),
                            (0, 3, 7, 6),
                            (0, 2, 3, 6),
                        ):
                            tets.append([hexa[i] for i in tet])

        # map every surface vertex onto the matching solid node; a corner
        # behind a truncated overlap snaps to the interface plane of its box.
        mapping = np.empty(len(self.obj_local), dtype=np.int64)
        node_array = np.asarray(nodes)
        for k, box in enumerate(boxes):
            for i in range(8):
                p = self.obj_local[8 * k + i]
                x_t = min(max(p[0], left[k]), right[k])
                idx = node_index.get(_round_key((x_t, p[1], p[2])))
                if idx is None:
                    delta = node_array - np.array([x_t, p[1], p[2]])
                    idx = int(np.argmin(np.einsum("ij,ij->i", delta, delta)))
                mapping[8 * k + i] = idx
        if len(np.unique(mapping)) != len(mapping):
            raise ValueError("surface vertices do not map to distinct solid nodes")

        x0 = node_array + self.offset[None, :]
        return x0, np.asarray(tets, dtype=np.int64), mapping, None

    @staticmethod
    def _min_edge(x0, tets=None):
        pts = x0[tets] if tets is not None else x0
        if pts.ndim == 2:
            return float(np.min(np.linalg.norm(pts, axis=1)))
        best = np.inf
        for i in range(4):
            for j in range(i + 1, 4):
                d = pts[:, i, :] - pts[:, j, :]
                best = min(best, float(np.min(np.sqrt(np.einsum("...i,...i->...", d, d)))))
        return best

    # -------------------------------------------------------------- dynamics
    def _reset(self) -> None:
        self.x = self.x0.copy()
        self.v = np.zeros_like(self.x0)
        self.t = 0.0
        self.substeps = 0

    def _kinematics(self, x):
        xt = x[self.tets]
        Ds = np.transpose(xt[:, 1:] - xt[:, :1], (0, 2, 1))
        return Ds @ self.Dm_inv

    def _internal(self, x):
        F = self._kinematics(x)
        R = _polar(F)
        J = _det(F)
        cofF = _cof(F)
        P = (
            2.0 * self.mu * (F - R)
            + self.lam * (J - 1.0)[..., None, None] * cofF
        )
        G = -self.V0[:, None, None] * (P @ self.Dm_invT)
        W = np.empty((len(self.tets), 4, 3))
        W[:, 1] = G[:, :, 0]
        W[:, 2] = G[:, :, 1]
        W[:, 3] = G[:, :, 2]
        W[:, 0] = -G.sum(axis=2)
        forces = np.empty((self.nnodes, 3))
        flat = self.tets_flat
        for c in range(3):
            forces[:, c] = np.bincount(
                flat, weights=W[:, :, c].reshape(-1), minlength=self.nnodes
            )
        return forces

    def _step(self, h) -> None:
        forces = self._internal(self.x)
        accel = (
            (forces + self.m[:, None] * self.gravity[None, :]) / self.m[:, None]
            - self.xi_d * self.v
        )
        self.v += h * accel
        self.v[self.node_fixed] = 0.0
        self.x += h * self.v
        self.x[self.node_fixed] = self.x0[self.node_fixed]

    def advance_to(self, time):
        if not self._ready:
            raise RuntimeError("solver is not initialized")
        time = float(time)
        if time < self.t - 1e-9:
            self._reset()
        while self.t < time - 1e-12:
            remaining = time - self.t
            if remaining >= self.dt:
                self._step(self.dt)
                self.t += self.dt
                self.substeps += 1
            else:
                self._step(remaining)
                self.t = time
            if self.substeps and self.substeps % 4000 == 0:
                if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()):
                    raise RuntimeError("numerical instability in elastic solver")
        return self._frame(time)

    # ----------------------------------------------------------------- frame
    def _frame(self, time):
        if time <= 0.0:
            position = self.obj_world.copy()
            velocity = np.zeros_like(self.obj_world)
        else:
            position = self.x[self.mapping].copy()
            velocity = self.v[self.mapping].copy()

        forces = self._internal(self.x)
        # support reaction on the clamped root block of the solid
        reaction = -(
            forces[self.node_fixed]
            + self.m[self.node_fixed, None] * self.gravity[None, :]
        ).sum(axis=0)
        gravity_total = self.total_mass * self.gravity
        drag = -self.xi_d * (self.m[:, None] * self.v).sum(axis=0)
        net_external = gravity_total + reaction + drag

        F = self._kinematics(self.x)

        return StateFrame(
            time=float(time),
            position=position,
            velocity=velocity,
            mass=self.sample_mass.copy(),
            attributes={
                "mass_quantity": self.sample_mass.copy(),
                "fixed": self.sample_fixed.copy(),
            },
            fields={
                "budget_external_force": np.asarray(net_external, dtype=float).reshape(1, 1, 3),
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": np.asarray(F, dtype=float),
            },
            scalars={
                "reference_force": self.reference_force,
                "gravity_x": float(self.gravity[0]),
                "gravity_y": float(self.gravity[1]),
                "gravity_z": float(self.gravity[2]),
            },
        )

    def diagnostics(self) -> dict:
        return {
            "method": "explicit corotated tetrahedral FEM",
            "solid_nodes": int(self.nnodes) if self._ready else None,
            "solid_tets": int(len(self.tets)) if self._ready else None,
            "substep_dt": float(self.dt) if self._ready else None,
            "substeps": int(self.substeps) if self._ready else None,
            "total_mass_kg": float(self.total_mass) if self._ready else None,
        }
