"""Physically-based cloth solver for the pinned denim gravity sag scene.

Method: XPBD (position based dynamics with compliance) on the 17x17 denim
grid.  Structural / shear / bending distance constraints are solved with
vectorised, vertex-disjoint Gauss-Seidel colouring so the sheet behaves as a
nearly inextensible heavy fabric.  The two top corners are pinned, gravity
sags the sheet into broad folds, and the hem can come to rest on the ground
plane (z = 0) with friction.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# XPBD compliances folded into per-substep alpha (dt ~ 3.3e-3 s).
A_STRETCH = 0.1     # nearly inextensible
A_SHEAR = 1.0
A_BEND = 200.0      # finite bending stiffness -> folds, not rubber


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        cloth = cfg["cloth"]
        self.g = np.asarray(cfg["gravity"], dtype=float)
        self.sample_dt = float(cfg["sample_dt"])

        frame0 = scene.initial_state()
        self.p0 = np.array(frame0.position, dtype=float)
        self.N = len(self.p0)
        self.mass = np.array(frame0.mass, dtype=float)
        self.boundary = np.asarray(frame0.attributes["boundary_vertex"]).copy()

        nx, ny = map(int, cloth["grid_shape"])
        self.nx, self.ny = nx, ny
        self.thickness = float(cloth.get("thickness", 0.015))
        self.radius = 0.5 * self.thickness

        self.p = self.p0.copy()
        self.v = np.zeros_like(self.p)
        self.t = 0.0

        # Pins: the two top corners of the sheet.
        z, x = self.p0[:, 2], self.p0[:, 0]
        top = z > z.max() - 1e-6
        corner = (x < x.min() + 1e-6) | (x > x.max() - 1e-6)
        self.pinned = top & corner
        self.w = np.where(self.pinned, 0.0, 1.0 / np.maximum(self.mass, 1e-12))

        self._build_constraints()
        self._build_triangles()

        self.substeps = 10
        self.iterations = 15
        self.drag = 0.6                # mild air damping -> settles down
        self.ground_friction = 0.90    # tangential velocity retention on contact

    def _build_constraints(self) -> None:
        nx, ny = self.nx, self.ny
        p0 = self.p0
        groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

        def K(i, j):
            return i * nx + j

        def add(i1, i2, alpha):
            i1 = np.asarray(i1, dtype=np.int64)
            i2 = np.asarray(i2, dtype=np.int64)
            if i1.size == 0:
                return
            rest = np.linalg.norm(p0[i2] - p0[i1], axis=1)
            groups.append((i1, i2, rest, alpha))

        # structural horizontal, grouped by j parity (vertex disjoint)
        for par in (0, 1):
            add([K(i, j) for i in range(ny) for j in range(par, nx - 1, 2)],
                [K(i, j + 1) for i in range(ny) for j in range(par, nx - 1, 2)],
                A_STRETCH)
        # structural vertical, grouped by i parity
        for par in (0, 1):
            add([K(i, j) for i in range(par, ny - 1, 2) for j in range(nx)],
                [K(i + 1, j) for i in range(par, ny - 1, 2) for j in range(nx)],
                A_STRETCH)
        # shear diagonals, grouped by (i%2, j%2)
        for pi in (0, 1):
            for pj in (0, 1):
                add([K(i, j) for i in range(pi, ny - 1, 2) for j in range(pj, nx - 1, 2)],
                    [K(i + 1, j + 1) for i in range(pi, ny - 1, 2) for j in range(pj, nx - 1, 2)],
                    A_SHEAR)
                add([K(i, j + 1) for i in range(pi, ny - 1, 2) for j in range(pj, nx - 1, 2)],
                    [K(i + 1, j) for i in range(pi, ny - 1, 2) for j in range(pj, nx - 1, 2)],
                    A_SHEAR)
        # bending (2-apart) horizontal, grouped by j % 4
        for r in range(4):
            add([K(i, j) for i in range(ny) for j in range(r, nx - 2, 4)],
                [K(i, j + 2) for i in range(ny) for j in range(r, nx - 2, 4)],
                A_BEND)
        # bending vertical, grouped by i % 4
        for r in range(4):
            add([K(i, j) for i in range(r, ny - 2, 4) for j in range(nx)],
                [K(i + 2, j) for i in range(r, ny - 2, 4) for j in range(nx)],
                A_BEND)

        self.groups = groups
        self.lam = [np.zeros(len(g[0])) for g in groups]

    def _build_triangles(self) -> None:
        nx, ny = self.nx, self.ny
        ii, jj = np.meshgrid(np.arange(ny - 1), np.arange(nx - 1), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        k00 = (ii * nx + jj).astype(np.int64)
        k01 = k00 + 1
        k10 = k00 + nx
        k11 = k10 + 1
        self.tris = np.concatenate(
            [np.stack([k00, k01, k10], axis=1), np.stack([k01, k11, k10], axis=1)],
            axis=0)
        # rest material coordinates (u, v) = (x0, z0)
        a, b, c = (self.p0[self.tris[:, k]] for k in (0, 1, 2))
        dm = np.empty((len(self.tris), 2, 2))
        dm[:, 0, 0] = b[:, 0] - a[:, 0]; dm[:, 0, 1] = c[:, 0] - a[:, 0]
        dm[:, 1, 0] = b[:, 2] - a[:, 2]; dm[:, 1, 1] = c[:, 2] - a[:, 2]
        self.dm_inv = np.linalg.inv(dm)
        self.vert_tris = np.zeros(self.N, dtype=np.int64)
        np.add.at(self.vert_tris, self.tris.ravel(), 1)
        self.vert_tris = np.maximum(self.vert_tris, 1)

    # ------------------------------------------------------------- simulation
    def _step(self, dt: float) -> None:
        p, v, w = self.p, self.v, self.w
        v += self.g * dt
        p_prev = p.copy()
        p += v * dt

        for lam, (i1, i2, rest, alpha) in zip(self.lam, self.groups):
            lam.fill(0.0)  # XPBD multipliers reset each substep
            w1 = w[i1]; w2 = w[i2]
            denom = w1 + w2
            active = denom > 0.0
            for _ in range(self.iterations):
                d = p[i2] - p[i1]
                dist = np.sqrt(np.einsum("ij,ij->i", d, d))
                dist = np.maximum(dist, 1e-12)
                c = dist - rest
                dlam = np.where(active, (-c - alpha * lam) / (denom + alpha), 0.0)
                lam += dlam
                corr = (dlam / dist)[:, None] * d
                p[i1] -= (w1[:, None] * corr)
                p[i2] += (w2[:, None] * corr)

        # ground contact (plane z = 0, top face of the ground box)
        below = p[:, 2] < self.radius
        if below.any():
            p[below, 2] = self.radius

        # pinned corners are immobile
        p[self.pinned] = self.p0[self.pinned]

        v = (p - p_prev) / dt
        if below.any():
            v[below, 0] *= self.ground_friction
            v[below, 1] *= self.ground_friction
            np.maximum(v[below, 2], 0.0, out=v[below, 2])
        v[self.pinned] = 0.0
        v *= np.exp(-self.drag * dt)
        self.v = v
        self.p = p
        self.t += dt

    # ----------------------------------------------------------------- output
    def _deformation_gradients(self) -> np.ndarray:
        a, b, c = (self.p[self.tris[:, k]] for k in (0, 1, 2))
        ds = np.empty((len(self.tris), 3, 2))
        ds[:, :, 0] = b - a
        ds[:, :, 1] = c - a
        f = ds @ self.dm_inv                       # [T,3,2]
        ct = np.einsum("tik,tij->tkj", f, f)       # C = F^T F, [T,2,2]
        det = ct[:, 0, 0] * ct[:, 1, 1] - ct[:, 0, 1] * ct[:, 1, 0]
        s = np.sqrt(np.maximum(det, 1e-16))
        tr = ct[:, 0, 0] + ct[:, 1, 1]
        root = np.sqrt(np.maximum(tr + 2.0 * s, 1e-16))
        u = np.empty_like(ct)
        u[:, 0, 0] = (ct[:, 0, 0] + s) / root
        u[:, 0, 1] = ct[:, 0, 1] / root
        u[:, 1, 0] = ct[:, 1, 0] / root
        u[:, 1, 1] = (ct[:, 1, 1] + s) / root
        fg = np.zeros((self.N, 2, 2))
        for k in range(3):
            np.add.at(fg, self.tris[:, k], u)
        return fg / self.vert_tris[:, None, None]

    def _frame(self) -> StateFrame:
        ext = self.mass[:, None] * self.g[None, :]
        csd = self.p[:, 2] - self.radius  # signed gap to ground plane
        return StateFrame(
            float(self.t),
            self.p.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": self._deformation_gradients(),
            },
            fields={
                "cumulative_mass_source": np.zeros(self.N),
                "external_force": ext,
                "contact_signed_distance": csd,
            },
            scalars={"contact_length_scale": self.thickness},
        )

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        dt_sub = self.sample_dt / self.substeps
        eps = 1e-12
        while self.t < target - eps:
            self._step(min(dt_sub, target - self.t))
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_vertices": int(self.N),
            "num_constraints": int(sum(len(g[0]) for g in self.groups)),
            "pinned": int(self.pinned.sum()),
            "time": float(self.t),
        }
