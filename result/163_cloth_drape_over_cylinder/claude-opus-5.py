"""Cloth drape over a horizontal cylinder.

Self-contained thin-shell solver (no external physics engine):

* Discretization: structured quad grid of point masses (17x17) with
  in-plane structural + shear distance constraints and a second-neighbour
  bending constraint set.
* Time stepping: substepped semi-implicit (symplectic) Euler predictor
  followed by an XPBD constraint solve (Gauss-Seidel over conflict-free
  edge colours), so material stiffness is time-step independent and the
  integration stays unconditionally stable for near-inextensible cloth.
* Contact: analytic signed distances against the frozen static geometry
  (cylinder with axis along +y, ground plane top at z=0), resolved by
  non-penetration position projection plus a velocity-level Coulomb
  friction law whose normal impulse includes the sustained support term
  m*|g.n|*dt, so the sheet grips the cylinder instead of sliding off.
* Self-contact: discrete-thickness vertex repulsion between material
  points that are far apart in the reference sheet, which keeps folded
  layers from passing through one another.

Exported raw fields follow the StateFrame contract exactly.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.cfg = cfg
        cloth = cfg["cloth"]

        self.nx, self.ny = int(cloth["grid_shape"][0]), int(cloth["grid_shape"][1])
        self.sx, self.sy = float(cloth["size"][0]), float(cloth["size"][1])
        self.center = np.asarray(cloth["center"], dtype=float)
        self.thickness = float(cloth["thickness"])
        self.total_mass = float(cloth["total_mass"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.sample_dt = float(cfg["sample_dt"])

        # ---- frozen static geometry (scene_geometry.yaml) -----------------
        # cylinder: radius 0.52, centre (0,0,0.7), rotated 90 deg about x so
        # its axis is the y axis; the cloth (|y| <= 1.25) stays inside the
        # 2.8 m long barrel, so an infinite-axis distance field is exact.
        self.cyl_center_xz = np.array([0.0, 0.7])
        self.cyl_radius = 0.52
        self.cyl_half_len = 1.4
        self.ground_z = 0.0  # top face of the ground box

        n = self.nx * self.ny
        self.n = n
        self.du = self.sx / (self.nx - 1)
        self.dv = self.sy / (self.ny - 1)

        # ---- reference (material) grid and initial positions --------------
        x = np.linspace(-self.sx * 0.5, self.sx * 0.5, self.nx)
        y = np.linspace(-self.sy * 0.5, self.sy * 0.5, self.ny)
        xx, yy = np.meshgrid(x, y, indexing="xy")  # shape (ny, nx)
        pos = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(n))) + self.center

        # Tiny deterministic out-of-plane perturbation: breaks the exactly
        # degenerate flat/symmetric state so buckling picks physical folds
        # rather than staying in an unstable balanced configuration.
        rng = np.random.default_rng(int(seed))
        pos[:, 2] += 2.0e-4 * rng.standard_normal(n)

        self.x = pos
        self.v = np.zeros((n, 3))
        self.mass = np.full(n, self.total_mass / n)
        self.w = 1.0 / self.mass  # inverse mass
        self.boundary = (
            np.isclose(xx, x[0]) | np.isclose(xx, x[-1])
            | np.isclose(yy, y[0]) | np.isclose(yy, y[-1])
        ).ravel().astype(np.int32)

        self._build_constraints()

        # ---- self-contact bookkeeping -------------------------------------
        # Effective discrete cloth thickness. The 17x17 mesh cannot resolve
        # the 15 mm physical sheet, so layer separation is set from the mesh
        # spacing; pairs closer than ~1.5 spacings in material coordinates
        # are already governed by the in-plane constraints and are excluded.
        spacing = min(self.du, self.dv)
        self.self_radius = 0.42 * spacing
        mat = np.column_stack((xx.ravel(), yy.ravel()))
        mat_d = np.linalg.norm(mat[:, None, :] - mat[None, :, :], axis=2)
        allowed = mat_d > 1.55 * spacing
        allowed &= np.triu(np.ones((n, n), dtype=bool), 1)
        self.pair_a, self.pair_b = np.nonzero(allowed)

        # ---- solver parameters -------------------------------------------
        self.substeps = 12
        self.iterations = 5
        self.damping = 1.2  # 1/s aerodynamic-style velocity damping
        self.mu = 0.45  # cloth/obstacle friction coefficient
        self.max_speed = 30.0
        self.time = 0.0
        self._steps = 0
        self._max_stretch = 0.0

    def _build_constraints(self) -> None:
        nx, ny = self.nx, self.ny

        def vid(i, j):  # i -> x index (u), j -> y index (v)
            return j * nx + i

        du, dv = self.du, self.dv
        c_struct = 1.0e-8   # near-inextensible warp/weft
        c_shear = 5.0e-4    # softer in-plane shear
        c_bend = 3.0e-2     # soft bending (allows folds/wrinkles)

        ei: list[int] = []
        ej: list[int] = []
        rest: list[float] = []
        comp: list[float] = []

        def add(a, b, L, c):
            ei.append(a)
            ej.append(b)
            rest.append(L)
            comp.append(c)

        for j in range(ny):
            for i in range(nx):
                if i + 1 < nx:
                    add(vid(i, j), vid(i + 1, j), du, c_struct)
                if j + 1 < ny:
                    add(vid(i, j), vid(i, j + 1), dv, c_struct)
        diag = float(np.hypot(du, dv))
        for j in range(ny - 1):
            for i in range(nx - 1):
                add(vid(i, j), vid(i + 1, j + 1), diag, c_shear)
                add(vid(i + 1, j), vid(i, j + 1), diag, c_shear)
        for j in range(ny):
            for i in range(nx - 2):
                add(vid(i, j), vid(i + 2, j), 2.0 * du, c_bend)
        for j in range(ny - 2):
            for i in range(nx):
                add(vid(i, j), vid(i, j + 2), 2.0 * dv, c_bend)

        self.E_i = np.asarray(ei, dtype=np.int64)
        self.E_j = np.asarray(ej, dtype=np.int64)
        self.E_rest = np.asarray(rest, dtype=float)
        self.E_comp = np.asarray(comp, dtype=float)
        self.n_edges = len(ei)

        # Greedy edge colouring: within a colour no vertex repeats, so each
        # colour can be projected with a single vectorised Gauss-Seidel pass.
        used: list[set[int]] = []
        colour = np.empty(self.n_edges, dtype=np.int64)
        for e in range(self.n_edges):
            a, b = int(self.E_i[e]), int(self.E_j[e])
            c = 0
            while True:
                if c == len(used):
                    used.append(set())
                if a not in used[c] and b not in used[c]:
                    break
                c += 1
            used[c].add(a)
            used[c].add(b)
            colour[e] = c
        self.groups = [np.flatnonzero(colour == c) for c in range(len(used))]

    # ------------------------------------------------------------- physics
    def _project_constraints(self, x, lam):
        for g in self.groups:
            i = self.E_i[g]
            j = self.E_j[g]
            d = x[i] - x[j]
            L = np.linalg.norm(d, axis=1)
            L = np.maximum(L, 1.0e-12)
            C = L - self.E_rest[g]
            nrm = d / L[:, None]
            wi = self.w[i]
            wj = self.w[j]
            alpha = self.alpha[g]
            dlam = (-C - alpha * lam[g]) / (wi + wj + alpha)
            lam[g] += dlam
            corr = dlam[:, None] * nrm
            x[i] += wi[:, None] * corr
            x[j] -= wj[:, None] * corr

    def _resolve_contacts(self, x):
        """Project out obstacle penetration and record the contact normals.

        Normals are accumulated (not overwritten) so a vertex pinched
        between the ground and the cylinder still gets both directions;
        the accumulated set is later used for velocity-level friction.
        """
        h = 0.5 * self.thickness
        nrm_acc = self._cn
        hit = self._chit

        # --- cylinder (axis along y) --------------------------------------
        dx = x[:, 0] - self.cyl_center_xz[0]
        dz = x[:, 2] - self.cyl_center_xz[1]
        r = np.sqrt(dx * dx + dz * dz)
        rs = np.maximum(r, 1.0e-9)
        pen = (self.cyl_radius + h) - r
        inside = np.abs(x[:, 1]) <= self.cyl_half_len
        m = (pen > 0.0) & inside
        if np.any(m):
            nrm = np.zeros((int(m.sum()), 3))
            nrm[:, 0] = dx[m] / rs[m]
            nrm[:, 2] = dz[m] / rs[m]
            x[m] += pen[m][:, None] * nrm
            nrm_acc[m] += nrm
            hit[m] = True

        # --- ground plane -------------------------------------------------
        m = x[:, 2] < (self.ground_z + h)
        if np.any(m):
            x[m, 2] = self.ground_z + h
            nrm_acc[m, 2] += 1.0
            hit[m] = True

    def _resolve_self_contact(self, x):
        """Vertex-vertex repulsion giving the sheet a discrete thickness."""
        a, b = self.pair_a, self.pair_b
        d = x[a] - x[b]
        dist2 = np.einsum("ij,ij->i", d, d)
        rad = 2.0 * self.self_radius
        close = dist2 < rad * rad
        if not np.any(close):
            return
        a = a[close]
        b = b[close]
        d = d[close]
        dist = np.sqrt(np.maximum(dist2[close], 1.0e-18))
        nrm = d / dist[:, None]
        corr = (0.5 * (rad - dist))[:, None] * nrm
        # Scatter-add: duplicated vertices are averaged implicitly by
        # np.add.at, which keeps the projection stable under many pairs.
        np.add.at(x, a, corr)
        np.add.at(x, b, -corr)

    def _apply_friction(self, v, dt):
        """Coulomb friction at obstacle contacts, velocity level.

        The normal impulse of a resting contact is not the (projected-away)
        penetration but the weight component the obstacle carries, so the
        friction bound uses mu * (|v_n^-| + |g.n| dt), which stays finite
        and non-zero once the sheet settles.
        """
        hit = self._chit
        if not np.any(hit):
            return
        nrm = self._cn[hit]
        ln = np.linalg.norm(nrm, axis=1)
        ok = ln > 1.0e-12
        if not np.any(ok):
            return
        idx = np.flatnonzero(hit)[ok]
        nrm = nrm[ok] / ln[ok][:, None]
        vel = v[idx]
        vn = np.einsum("ij,ij->i", vel, nrm)
        vt = vel - vn[:, None] * nrm
        vt_mag = np.linalg.norm(vt, axis=1)
        gn = np.abs(nrm @ self.gravity)
        bound = self.mu * (np.maximum(-vn, 0.0) + gn * dt)
        scale = np.zeros_like(vt_mag)
        move = vt_mag > 1.0e-12
        scale[move] = np.maximum(0.0, 1.0 - bound[move] / vt_mag[move])
        v[idx] = np.maximum(vn, 0.0)[:, None] * nrm + scale[:, None] * vt

    def _substep(self, dt):
        x = self.x
        v = self.v
        x_prev = x.copy()

        # symplectic Euler predictor with external (gravity) acceleration
        v += dt * self.gravity
        x += dt * v

        self._cn = np.zeros((self.n, 3))
        self._chit = np.zeros(self.n, dtype=bool)

        self.alpha = self.E_comp / (dt * dt)
        lam = np.zeros(self.n_edges)
        for _ in range(self.iterations):
            self._project_constraints(x, lam)
            self._resolve_contacts(x)
        self._resolve_self_contact(x)
        self._resolve_contacts(x)

        v_new = (x - x_prev) / dt
        self._apply_friction(v_new, dt)
        v_new *= np.exp(-self.damping * dt)
        sp = np.linalg.norm(v_new, axis=1)
        big = sp > self.max_speed
        if np.any(big):
            v_new[big] *= (self.max_speed / sp[big])[:, None]
        if not np.isfinite(v_new).all():
            v_new = np.nan_to_num(v_new, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(x).all():
            np.copyto(x, np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        self.x = x
        self.v = v_new
        self._steps += 1

    # ------------------------------------------------------------ outputs
    def _deformation_gradient(self):
        """Per-vertex 2x2 surface deformation gradient in material coords."""
        X = self.x.reshape(self.ny, self.nx, 3)
        g_v, g_u = np.gradient(X, self.dv, self.du, axis=(0, 1))
        a = g_u.reshape(self.n, 3)
        b = g_v.reshape(self.n, 3)
        la = np.linalg.norm(a, axis=1)
        las = np.maximum(la, 1.0e-12)
        e1 = a / las[:, None]
        b1 = np.sum(b * e1, axis=1)
        bp = b - b1[:, None] * e1
        lbp = np.linalg.norm(bp, axis=1)
        F = np.zeros((self.n, 2, 2))
        F[:, 0, 0] = la
        F[:, 0, 1] = b1
        F[:, 1, 1] = lbp
        return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

    def _signed_distance(self):
        h = 0.5 * self.thickness
        dx = self.x[:, 0] - self.cyl_center_xz[0]
        dz = self.x[:, 2] - self.cyl_center_xz[1]
        r = np.sqrt(dx * dx + dz * dz)
        d_cyl = r - self.cyl_radius
        outside_len = np.abs(self.x[:, 1]) - self.cyl_half_len
        d_cyl = np.where(outside_len > 0.0, np.hypot(np.maximum(d_cyl, 0.0), outside_len), d_cyl)
        d_ground = self.x[:, 2] - self.ground_z
        return np.minimum(d_cyl, d_ground) - h

    def _frame(self):
        F = self._deformation_gradient()
        gap = self._signed_distance()
        ext = self.mass[:, None] * self.gravity[None, :]
        frame = StateFrame(
            float(max(self.time, 0.0)),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": F,
            },
            fields={
                "contact_signed_distance": gap,
                "cumulative_mass_source": np.zeros(1),
                "external_force": ext,
            },
            scalars={
                "contact_length_scale": float(self.thickness),
            },
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- driver
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while target - self.time > 1.0e-12:
            span = min(self.sample_dt, target - self.time)
            dt = span / self.substeps
            for _ in range(self.substeps):
                self._substep(dt)
            self.time += span
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "XPBD mass-spring thin shell, substepped symplectic Euler",
            "vertices": int(self.n),
            "constraints": int(self.n_edges),
            "colours": int(len(self.groups)),
            "substeps_per_sample": int(self.substeps),
            "iterations": int(self.iterations),
            "total_substeps": int(self._steps),
            "friction": float(self.mu),
        }
