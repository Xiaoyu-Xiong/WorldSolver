"""Pinned denim gravity sag.

Physical model
--------------
A dense 17x17 denim sheet is built in the frozen scene plane (xz), pinned at its
two upper corner vertices and released from rest under gravity (0, 0, -9.81)
m/s^2.  The sheet is discretised as a triangle mesh with one diagonal per quad
(the frozen topology triangulation) and simulated with a substepped
position-based (XPBD) time integrator:

* every substep applies gravity and a light fabric damping to the velocities,
* a forward-Euler predictor advances the positions,
* the cloth is made *inextensible* by one-sided strain-limiting distance
  constraints (|edge| <= (1+eps) * rest length, eps = 0.5%) solved with
  vectorised Gauss-Seidel sweeps over proper edge colourings,
* a soft (compliance based, XPBD) one-sided second-neighbour constraint gives
  the fabric a finite bending stiffness so it drapes into broad folds instead
  of grid-scale crumpling,
* the ground box (top face z = 0) is handled by a non-penetration projection
  plus Coulomb friction, and the pinned vertices are held exactly fixed.

The simulation is deterministic: the only symmetry breaking is a small,
seed-derived, smooth out-of-plane imperfection of the initially flat sheet
(amplitude ~5 mm), which simply seeds the physical buckling instability of the
drape.

Reported raw fields match the frozen StateFrame/trajectory contract.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)


class BenchmarkSolver:
    """Substepped XPBD cloth solver for the pinned denim gravity sag."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        cloth = dict(cfg["cloth"])
        self.scene_id = str(cfg.get("scene_id", ""))
        self.seed = int(seed)

        nx, ny = (int(v) for v in cloth["grid_shape"])
        sx, sy = (float(v) for v in cloth["size"])
        self.nx, self.ny = nx, ny
        self.center = np.asarray(cloth["center"], dtype=float)
        self.plane = str(cloth.get("plane", "xy"))
        self.thickness = float(cloth.get("thickness", 0.01))
        self.total_mass = float(cloth["total_mass"])
        self.gravity = np.array(cfg.get("gravity", _GRAVITY), dtype=float)

        self.dx = sx / float(nx - 1)
        self.dy = sy / float(ny - 1)
        n = nx * ny
        self.n = n

        # --- rest configuration (identical ordering to the frozen topology) --
        xs = np.linspace(-0.5 * sx, 0.5 * sx, nx)
        ys = np.linspace(-0.5 * sy, 0.5 * sy, ny)
        gx, gy = np.meshgrid(xs, ys, indexing="xy")
        self.u = gx.ravel().copy()          # material coordinate 0
        self.vmat = gy.ravel().copy()       # material coordinate 1
        if self.plane == "xz":
            local = np.column_stack((self.u, np.zeros(n), self.vmat))
        else:
            local = np.column_stack((self.u, self.vmat, np.zeros(n)))
        self.rest = local + self.center
        self.x = self.rest.copy()

        # --- mass -------------------------------------------------------------
        self.mass = np.full(n, self.total_mass / float(n))
        self.inv_mass = 1.0 / self.mass
        self.body_id = np.zeros(n, dtype=np.int32)
        self.external_force = self.mass[:, None] * self.gravity[None, :]

        # --- boundary / pin sets ---------------------------------------------
        i_idx = np.arange(n) % nx
        j_idx = np.arange(n) // nx
        self.boundary = (
            (i_idx == 0) | (i_idx == nx - 1) | (j_idx == 0) | (j_idx == ny - 1)
        ).astype(np.int32)
        top = j_idx == ny - 1
        pins = top & ((i_idx == 0) | (i_idx == nx - 1))
        self.pins = np.nonzero(pins)[0]
        self.pinned_mask = pins.astype(np.int32)
        self.w = self.inv_mass.copy()
        self.w[self.pins] = 0.0
        self.x0 = self.rest.copy()

        # --- seed-driven smooth out-of-plane imperfection ---------------------
        rng = np.random.default_rng(self.seed)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        xi = (self.u - self.u.min()) / max(sx, 1e-12)
        zeta = (self.vmat - self.vmat.min()) / max(sy, 1e-12)
        shape = (
            0.62 * np.sin(np.pi * xi + 0.0)
            + 0.38 * np.sin(2.0 * np.pi * xi + phase)
        ) * np.sin(np.pi * zeta)
        amp = 0.004
        off = amp * shape
        axis = 1 if self.plane == "xz" else 2  # out-of-plane world axis
        self.x[:, axis] += off

        self.v = np.zeros((n, 3))

        # --- structural edges: horizontals, verticals, quad diagonals ---------
        tri = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = j * nx + i
                b = a + 1
                c = a + nx
                d = c + 1
                tri.append((a, b, d))
                tri.append((a, d, c))
        self.triangles = np.array(tri, dtype=np.int64)

        e0, e1, rest_len, color, groups = [], [], [], [], []
        for i in range(nx - 1):
            for j in range(ny):
                a = j * nx + i
                e0.append(a); e1.append(a + 1); rest_len.append(self.dx)
                color.append(i % 2); groups.append(0)
        for i in range(nx):
            for j in range(ny - 1):
                a = j * nx + i
                e0.append(a); e1.append(a + nx); rest_len.append(self.dy)
                color.append(j % 2); groups.append(1)
        diag = float(np.hypot(self.dx, self.dy))
        for i in range(nx - 1):
            for j in range(ny - 1):
                a = j * nx + i
                e0.append(a); e1.append(a + nx + 1); rest_len.append(diag)
                color.append((i + j) % 2); groups.append(2)
        self.edge_i = np.array(e0, dtype=np.int64)
        self.edge_j = np.array(e1, dtype=np.int64)
        self.edge_rest = np.array(rest_len, dtype=float)
        self.edge_color = np.array(color, dtype=np.int64)
        self.edge_group = np.array(groups, dtype=np.int64)
        self.stretch_limit = 1.005
        self.edge_lmax = self.edge_rest * self.stretch_limit
        self._edge_pass = []
        for grp in (0, 1, 2):
            for col in (0, 1):
                sel = np.nonzero((self.edge_group == grp) & (self.edge_color == col))[0]
                self._edge_pass.append(sel)

        # --- soft one-sided bending (second neighbour) constraints ------------
        b0, b1, brest, bcol, bgrp = [], [], [], [], []
        for i in range(nx - 2):
            for j in range(ny):
                a = j * nx + i
                b0.append(a); b1.append(a + 2); brest.append(2.0 * self.dx)
                bcol.append((i // 2) % 2); bgrp.append(0)
        for i in range(nx):
            for j in range(ny - 2):
                a = j * nx + i
                b0.append(a); b1.append(a + 2 * nx); brest.append(2.0 * self.dy)
                bcol.append((j // 2) % 2); bgrp.append(1)
        self.bend_i = np.array(b0, dtype=np.int64)
        self.bend_j = np.array(b1, dtype=np.int64)
        self.bend_rest = np.array(brest, dtype=float)
        self.bend_color = np.array(bcol, dtype=np.int64)
        self.bend_group = np.array(bgrp, dtype=np.int64)
        self._bend_pass = []
        for grp in (0, 1):
            for col in (0, 1):
                sel = np.nonzero((self.bend_group == grp) & (self.bend_color == col))[0]
                self._bend_pass.append(sel)
        self.bend_lambda = np.zeros(len(self.bend_i))
        # compliance in m/N -> finite (denim-like) bending stiffness
        self.bend_compliance = 0.05
        self.bend_slack = 1.0 - 1e-3

        # --- ground contact definition ---------------------------------------
        self.ground_top = 0.0
        ground = self._ground_box(cfg)
        self.ground_half = ground[0]
        self.ground_center = ground[1]

        # --- time integration parameters -------------------------------------
        sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.n_sub = 15
        self.dt_sub = sample_dt / float(self.n_sub)
        self.strain_iters = 8
        self.bend_iters = 4
        self.damping = 2.5            # 1/s, heavy-fabric dissipation
        self.friction = 0.7           # Coulomb-ish tangential damping at contact

        self.t = 0.0
        self._initial_frame = self._make_frame(0.0)

    # ------------------------------------------------------------- utilities
    @staticmethod
    def _ground_box(cfg: dict):
        half = np.array([2.4, 2.0, 0.05])
        center = np.array([0.0, 0.0, -0.05])
        geom = cfg.get("scene_geometry")
        if isinstance(geom, dict):
            for item in geom.get("objects", []):
                if item.get("id") == "ground":
                    g = item.get("geometry", {})
                    if "half_extents" in g:
                        half = np.asarray(g["half_extents"], dtype=float)
                    tr = item.get("transform", {}) or {}
                    if "translation" in tr:
                        center = np.asarray(tr["translation"], dtype=float)
        return half, center

    def _project_strain(self) -> None:
        x = self.x
        w = self.w
        for sel in self._edge_pass:
            if sel.size == 0:
                continue
            i = self.edge_i[sel]
            j = self.edge_j[sel]
            d = x[j] - x[i]
            length = np.sqrt(np.einsum("ij,ij->i", d, d))
            lmax = self.edge_lmax[sel]
            over = length - lmax
            act = over > 0.0
            if not np.any(act):
                continue
            inv = 1.0 / np.maximum(length, 1e-12)
            nrm = d * inv[:, None]
            wi = w[i]
            wj = w[j]
            denom = wi + wj
            good = act & (denom > 0.0)
            if not np.any(good):
                continue
            scale = np.where(good, over / np.maximum(denom, 1e-12), 0.0)
            corr_i = (wi * scale)[:, None] * nrm
            corr_j = (wj * scale)[:, None] * nrm
            x[i] += corr_i
            x[j] -= corr_j

    def _project_bending(self, h: float) -> None:
        x = self.x
        w = self.w
        alpha = self.bend_compliance / (h * h)
        for sel in self._bend_pass:
            if sel.size == 0:
                continue
            i = self.bend_i[sel]
            j = self.bend_j[sel]
            d = x[j] - x[i]
            length = np.sqrt(np.einsum("ij,ij->i", d, d))
            target = self.bend_rest[sel] * self.bend_slack
            c = length - target
            act = c < 0.0
            if not np.any(act):
                continue
            denom = w[i] + w[j] + alpha
            good = act & (denom > 0.0)
            if not np.any(good):
                continue
            lam = self.bend_lambda[sel]
            dlam = np.where(good, (-c - alpha * lam) / np.maximum(denom, 1e-12), 0.0)
            lam_new = np.maximum(lam + dlam, 0.0)
            self.bend_lambda[sel] = lam_new
            step = lam_new - lam
            inv = 1.0 / np.maximum(length, 1e-12)
            nrm = d * inv[:, None]
            x[i] -= (w[i] * step)[:, None] * nrm
            x[j] += (w[j] * step)[:, None] * nrm

    def _resolve_contact(self) -> None:
        x = self.x
        half = self.ground_half
        ctr = self.ground_center
        top = ctr[2] + half[2]
        inside = (np.abs(x[:, 0] - ctr[0]) <= half[0]) & (
            np.abs(x[:, 1] - ctr[1]) <= half[1]
        )
        pen = inside & (x[:, 2] < top)
        if np.any(pen):
            x[pen, 2] = top

    def _step(self, h: float) -> None:
        x, v, w = self.x, self.v, self.w
        damp = max(0.0, 1.0 - self.damping * h)
        v += h * self.gravity[None, :]
        v *= damp
        v[self.pins] = 0.0
        x_prev = x.copy()
        x += h * v
        x[self.pins] = self.x0[self.pins]

        for _ in range(self.strain_iters):
            self._project_strain()

        self.bend_lambda[:] = 0.0
        for _ in range(self.bend_iters):
            self._project_bending(h)
        # keep the (much stiffer) inextensibility limit satisfied afterwards
        self._project_strain()

        self._resolve_contact()

        v[:] = (x - x_prev) / h
        # friction where the sheet rests on / penetrates the ground
        half = self.ground_half
        ctr = self.ground_center
        top = ctr[2] + half[2]
        inside = (np.abs(x[:, 0] - ctr[0]) <= half[0]) & (
            np.abs(x[:, 1] - ctr[1]) <= half[1]
        )
        touch = inside & (x[:, 2] <= top + 1e-6)
        if np.any(touch):
            f = 1.0 - self.friction
            v[touch, 0] *= f
            v[touch, 1] *= f
            v[touch, 2] = np.minimum(v[touch, 2], 0.0)
        x[self.pins] = self.x0[self.pins]
        v[self.pins] = 0.0

    # ---------------------------------------------------------- deformation
    def _deformation_gradient(self) -> np.ndarray:
        """Area-weighted per-vertex 2x2 in-plane deformation gradient."""
        x = self.x
        tri = self.triangles
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ru = np.stack((self.u[b] - self.u[a], self.u[c] - self.u[a]), axis=1)
        rv = np.stack((self.vmat[b] - self.vmat[a], self.vmat[c] - self.vmat[a]), axis=1)
        det = ru[:, 0] * rv[:, 1] - ru[:, 1] * rv[:, 0]
        det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        inv = np.stack(
            (np.stack((rv[:, 1], -ru[:, 1]), axis=1),
             np.stack((-rv[:, 0], ru[:, 0]), axis=1)),
            axis=1,
        ) / det[:, None, None]

        E = np.stack((x[b] - x[a], x[c] - x[a]), axis=2)      # [Ne,3,2]
        Fm = np.einsum("eij,ejk->eik", E, inv)                # [Ne,3,2]

        e1 = E[:, :, 0]
        e2 = E[:, :, 1]
        nrm = np.cross(e1, e2)
        ln = np.linalg.norm(nrm, axis=1)
        ln = np.where(ln < 1e-14, 1e-14, ln)
        nrm = nrm / ln[:, None]
        t1 = e1 / np.maximum(np.linalg.norm(e1, axis=1), 1e-14)[:, None]
        t2 = np.cross(nrm, t1)
        f1 = np.stack(
            (np.einsum("ei,ei->e", t1, Fm[:, :, 0]),
             np.einsum("ei,ei->e", t2, Fm[:, :, 0])), axis=1)
        f2 = np.stack(
            (np.einsum("ei,ei->e", t1, Fm[:, :, 1]),
             np.einsum("ei,ei->e", t2, Fm[:, :, 1])), axis=1)
        F2 = np.stack((f1, f2), axis=2)                        # [Ne,2,2]

        area = 0.5 * np.abs(det)
        acc = np.zeros((self.n, 2, 2))
        weight = np.zeros(self.n)
        ident = np.eye(2)
        for k in range(3):
            vtx = tri[:, k]
            np.add.at(acc, vtx, area[:, None, None] * F2)
            np.add.at(weight, vtx, area)
        weight = np.where(weight < 1e-16, 1.0, weight)
        F = acc / weight[:, None, None]
        lonely = weight < 1e-15
        if np.any(lonely):
            F[lonely] = ident[None, :, :]
        return F

    # ------------------------------------------------------------- interface
    def _make_frame(self, time: float) -> StateFrame:
        F = self._deformation_gradient()
        gap = self.x[:, 2] - (self.ground_center[2] + self.ground_half[2])
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F,
                "boundary_vertex": self.boundary.copy(),
                "pinned_vertex": self.pinned_mask.copy(),
            },
            fields={
                "contact_signed_distance": gap,
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.external_force.copy(),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if target < 0.0:
            raise ValueError("advance_to requires a non-negative time")
        if target != self.t:
            while self.t < target - 1e-12:
                h = min(self.dt_sub, target - self.t)
                if h <= 1e-12:
                    break
                self._step(h)
                self.t += h
            self.t = target
        return self._make_frame(self.t)

    def diagnostics(self) -> dict[str, object]:
        return {
            "scene_id": self.scene_id,
            "solver": "substepped_xpbd_strain_limited_cloth",
            "vertex_count": int(self.n),
            "element_count": int(len(self.triangles)),
            "edge_count": int(len(self.edge_i)),
            "pin_vertices": [int(p) for p in self.pins],
            "substeps": int(self.n_sub),
            "strain_limit": float(self.stretch_limit),
            "contact_length_scale": float(self.thickness),
            "simulated_time": float(self.t),
        }
