"""Strain-limited cloth sphere-drape solver (pure NumPy PBD/XPBD).

Recreates the C-IPC Figure 15(c) phenomenon: a soft square cloth falls under
gravity onto a fixed sphere and settles into a wrinkled drape while hard
principal-strain limits prevent stretch/compression beyond [0.9, 1.1].

Method
------
* Semi-implicit prediction: v += g*h, heavy damping, x_pred = x + v*h.
* Iterative constraint solve (frozen iteration count) with colored
  Gauss-Seidel vectorized edge passes:
    - structural edges (soft membrane stiffness, like 0.01x),
    - shear diagonals (soft),
    - skip-2 bending edges (very soft),
    - hard strain-limit projection whenever an edge leaves
      [strain_lower, strain_upper] * rest_length.
* Position-level collision projection against the fixed sphere and the
  ground box (cloth half-thickness offset), with tangential (friction)
  velocity damping at contacts.
* Per-vertex deformation gradient from a least-squares fit of the incident
  one-ring in material coordinates; third column is the unit normal so the
  in-plane principal stretches are exactly the leading singular values.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class _Group:
    """A set of distance constraints whose vertices are mutually disjoint."""

    __slots__ = ("edges", "rest", "k")

    def __init__(self, edges, rest, k):
        self.edges = np.ascontiguousarray(edges, dtype=np.int64)
        self.rest = np.ascontiguousarray(rest, dtype=float)
        self.k = float(k)


class BenchmarkSolver:
    def __init__(self) -> None:
        self.t = 0.0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = getattr(scene, "config", {}) or {}
        meta = scene.metadata()
        frame0 = scene.initial_state()

        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.inv_mass = 1.0 / np.maximum(self.mass, 1e-12)
        self.n = len(self.x)

        uv = frame0.attributes.get("uv")
        self.uv = np.array(uv, dtype=float) if uv is not None else None

        cloth_cfg = cfg.get("cloth", {}) or {}
        cloth_meta = meta.get("cloth", {}) or {}
        grid = cloth_cfg.get("grid_shape") or cloth_meta.get("grid_shape") or [31, 31]
        self.nx, self.ny = int(grid[0]), int(grid[1])
        self.side = float(cloth_cfg.get("side_length", cloth_meta.get("side_length", 2.8)))
        self.thickness = float(
            cloth_cfg.get("thickness", cloth_meta.get("thickness", 0.012))
        )
        self.strain_lo = float(cloth_cfg.get("strain_lower", cloth_meta.get("strain_lower", 0.9)))
        self.strain_hi = float(cloth_cfg.get("strain_upper", cloth_meta.get("strain_upper", 1.1)))
        self.damping = float(cloth_cfg.get("damping", 1.35))
        self.friction = float(cloth_cfg.get("friction", 0.22))

        grav = cfg.get("gravity", meta.get("gravity", [0.0, 0.0, -9.81]))
        self.gravity = np.asarray(grav, dtype=float)

        sphere = cfg.get("sphere", {}) or meta.get("sphere", {}) or {}
        self.sphere_center = np.asarray(sphere.get("center", [0.0, 0.0, 0.64]), dtype=float)
        self.sphere_radius = float(sphere.get("radius", 0.64))
        self.ground_z = float(cfg.get("ground", meta.get("ground_height", 0.0)))

        solver_cfg = cfg.get("solver", {}) or {}
        self.substep = float(solver_cfg.get("substep", 0.004))
        self.iters = int(solver_cfg.get("constraint_iterations", 15))

        # --- build constraint groups ---------------------------------------
        # true vertex spacing from the frozen panel: side / (count - 1)
        self.dx_x = self.side / max(self.nx - 1, 1)
        self.dx_y = self.side / max(self.ny - 1, 1)
        self.groups = self._build_groups()
        # --- one-ring structure for per-vertex deformation gradient --------
        self._build_rings()

        self.stats = {
            "max_principal_stretch": 1.0,
            "min_principal_stretch": 1.0,
            "max_contact_penetration": 0.0,
        }

    def _build_groups(self):
        nx, ny = self.nx, self.ny
        dx = self.dx_x  # square panel: dx_x == dx_y
        dlen = np.sqrt(self.dx_x**2 + self.dx_y**2)

        def collect(items, k):
            if not items:
                return _Group(np.zeros((0, 2), dtype=np.int64), np.zeros(0), k)
            edges = np.array([it[0] for it in items], dtype=np.int64)
            rest = np.array([it[1] for it in items], dtype=float)
            return _Group(edges, rest, k)

        k_mem = 0.02    # soft membrane (0.01x-like), strain limits do the rest
        k_shear = 0.01
        k_bend = 0.008

        groups = []
        # structural horizontal (a,b)-(a,b+1), colored by parity of b
        for parity in (0, 1):
            items = []
            for a in range(nx):
                for b in range(ny - 1):
                    if b % 2 == parity:
                        items.append(((a * ny + b, a * ny + b + 1), dx))
            groups.append(collect(items, k_mem))
        # structural vertical (a,b)-(a+1,b), colored by parity of a
        for parity in (0, 1):
            items = []
            for a in range(nx - 1):
                for b in range(ny):
                    if a % 2 == parity:
                        items.append(((a * ny + b, (a + 1) * ny + b), self.dx_y))
            groups.append(collect(items, k_mem))
        # shear diagonal (a,b)-(a+1,b+1), colored by parity of a
        for parity in (0, 1):
            items = []
            for a in range(nx - 1):
                for b in range(ny - 1):
                    if a % 2 == parity:
                        items.append(((a * ny + b, (a + 1) * ny + b + 1), dlen))
            groups.append(collect(items, k_shear))
        # shear anti-diagonal (a,b+1)-(a+1,b), colored by parity of a
        for parity in (0, 1):
            items = []
            for a in range(nx - 1):
                for b in range(ny - 1):
                    if a % 2 == parity:
                        items.append(((a * ny + b + 1, (a + 1) * ny + b), dlen))
            groups.append(collect(items, k_shear))
        # bending horizontal skip-2 (a,b)-(a,b+2), colored by b%4<2
        for part in (0, 1):
            items = []
            for a in range(nx):
                for b in range(ny - 2):
                    if (b % 4 < 2) == (part == 0):
                        items.append(((a * ny + b, a * ny + b + 2), 2.0 * dx))
            groups.append(collect(items, k_bend))
        # bending vertical skip-2 (a,b)-(a+2,b), colored by a%4<2
        for part in (0, 1):
            items = []
            for a in range(nx - 2):
                for b in range(ny):
                    if (a % 4 < 2) == (part == 0):
                        items.append(((a * ny + b, (a + 2) * ny + b), 2.0 * dx))
            groups.append(collect(items, k_bend))
        return groups

    def _build_rings(self) -> None:
        nx, ny = self.nx, self.ny
        nbrs = -np.ones((self.n, 8), dtype=np.int64)
        slot = 0
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                if da == 0 and db == 0:
                    continue
                idx = slot
                slot += 1
                for a in range(nx):
                    for b in range(ny):
                        aa, bb = a + da, b + db
                        if 0 <= aa < nx and 0 <= bb < ny:
                            nbrs[a * ny + b, idx] = aa * ny + bb
        self.nbrs = nbrs
        valid = nbrs >= 0
        # material (rest, in-plane) coordinates
        mat = self.x[:, :2].copy()
        dM = np.zeros((self.n, 8, 2), dtype=float)
        rows, cols = np.nonzero(valid)
        dM[rows, cols] = mat[nbrs[rows, cols]] - mat[rows]
        A = np.einsum("npj,npk->pjk", dM, dM)  # (N,2,2)
        # regularize (rest grid guarantees non-singular, keep it safe)
        reg = 1e-12 * np.maximum(np.trace(A, axis1=1, axis2=2), 1e-12)
        A[:, 0, 0] += reg
        A[:, 1, 1] += reg
        self.A_inv = np.linalg.inv(A)
        self.dM = dM
        self.valid = valid

    # ----------------------------------------------------------------- solve
    def _substep(self, h: float) -> None:
        x_prev = self.x.copy()
        v = self.v + self.gravity[None, :] * h
        v *= 1.0 / (1.0 + self.damping * h)
        x = x_prev + v * h

        inv_m = self.inv_mass
        lo_f, hi_f = self.strain_lo, self.strain_hi
        r_eff = self.sphere_radius + 0.5 * self.thickness
        c = self.sphere_center
        z_min = self.ground_z + 0.5 * self.thickness

        for _ in range(self.iters):
            for grp in self.groups:
                e = grp.edges
                if len(e) == 0:
                    continue
                d = x[e[:, 1]] - x[e[:, 0]]
                ln = np.sqrt(np.einsum("ij,ij->i", d, d))
                safe = ln > 1e-12
                ln = np.where(safe, ln, 1e-12)
                rest = grp.rest
                # soft membrane spring toward rest length
                corr = grp.k * (ln - rest)
                # hard strain limiting
                hi = hi_f * rest
                lo = lo_f * rest
                corr = corr + np.where(ln > hi, ln - hi, 0.0)
                corr = corr + np.where(ln < lo, ln - lo, 0.0)
                w0 = inv_m[e[:, 0]]
                w1 = inv_m[e[:, 1]]
                ws = w0 + w1
                ws = np.where(ws > 0.0, ws, 1.0)
                n_dir = d / ln[:, None]
                scale = corr / ws
                x[e[:, 0]] += (w0 * scale)[:, None] * n_dir
                x[e[:, 1]] -= (w1 * scale)[:, None] * n_dir

            # fixed sphere collision (position projection)
            r = x - c[None, :]
            rn = np.sqrt(np.einsum("ij,ij->i", r, r))
            pen = rn < r_eff
            if np.any(pen):
                x[pen] = c[None, :] + (r[pen] / rn[pen, None]) * r_eff
            # ground box collision
            low = x[:, 2] < z_min
            if np.any(low):
                x[low, 2] = z_min

        v_new = (x - x_prev) / h
        # contact friction: tangential drag at contacting samples
        r = x - c[None, :]
        rn = np.sqrt(np.einsum("ij,ij->i", r, r))
        near_s = rn < r_eff + self.thickness
        if np.any(near_s):
            nrm = r / np.maximum(rn, 1e-12)[:, None]
            vn = np.einsum("ij,ij->i", v_new[near_s], nrm[near_s])
            vt = v_new[near_s] - vn[:, None] * nrm[near_s]
            vt *= np.exp(-self.friction * 40.0 * h)
            v_new[near_s] = vt + vn[:, None] * nrm[near_s]
        near_g = x[:, 2] < z_min + self.thickness
        if np.any(near_g):
            vt = v_new[near_g, :2] * np.exp(-self.friction * 40.0 * h)
            v_new[near_g, 0] = vt[:, 0]
            v_new[near_g, 1] = vt[:, 1]

        # stability guard
        speed = np.sqrt(np.einsum("ij,ij->i", v_new, v_new))
        fast = speed > 25.0
        if np.any(fast):
            v_new[fast] *= (25.0 / speed[fast])[:, None]

        self.x = x
        self.v = v_new

    # ---------------------------------------------------------------- output
    def _deformation_gradient(self) -> np.ndarray:
        x = self.x
        nbrs = self.nbrs
        valid = self.valid
        rows, cols = np.nonzero(valid)
        dx = np.zeros((self.n, 8, 3), dtype=float)
        dx[rows, cols] = x[nbrs[rows, cols]] - x[rows]
        B = np.einsum("npi,npj->nij", dx, self.dM)  # (N,3,2)
        G = B @ self.A_inv  # (N,3,2) in-plane deformation map
        c1 = G[:, :, 0]
        c2 = G[:, :, 1]
        nrm = np.cross(c1, c2)
        ln = np.sqrt(np.einsum("ij,ij->i", nrm, nrm))
        bad = ln < 1e-9
        if np.any(bad):
            nrm[bad] = np.array([0.0, 0.0, 1.0])
            ln = np.sqrt(np.einsum("ij,ij->i", nrm, nrm))
        nrm = nrm / ln[:, None]
        F = np.empty((self.n, 3, 3), dtype=float)
        F[:, :, 0] = c1
        F[:, :, 1] = c2
        F[:, :, 2] = nrm
        return F

    def _contact_signed_distance(self) -> np.ndarray:
        r = self.x - self.sphere_center[None, :]
        gap_s = np.sqrt(np.einsum("ij,ij->i", r, r)) - self.sphere_radius
        gap_g = self.x[:, 2] - self.ground_z
        return np.minimum(gap_s, gap_g) - 0.5 * self.thickness

    def _frame(self, time: float) -> StateFrame:
        F = self._deformation_gradient()
        stretches = np.linalg.svd(F, compute_uv=False)
        self.stats["max_principal_stretch"] = float(stretches.max())
        self.stats["min_principal_stretch"] = float(stretches.min())
        csd = self._contact_signed_distance()
        self.stats["max_contact_penetration"] = float(max(0.0, -csd.min()))
        attributes = {
            "deformation_gradient": F,
            "mass_quantity": self.mass.copy(),
        }
        if self.uv is not None:
            attributes["uv"] = self.uv.copy()
        return StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields={
                "contact_signed_distance": csd,
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )

    # ------------------------------------------------------------------- api
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.t:
            target = self.t
        while target - self.t > 1e-9:
            h = min(self.substep, target - self.t)
            self._substep(h)
            self.t += h
        if abs(self.t - target) <= 1e-9:
            self.t = target
        return self._frame(self.t)

    def diagnostics(self) -> dict[str, object]:
        return dict(self.stats)
