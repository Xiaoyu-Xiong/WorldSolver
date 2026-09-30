"""Self-contained elastic-solid solver for the six-wall Octocat compactor.

Physics
-------
* The Octocat surface asset is embedded in a hexahedral background lattice that
  is carved from the same implicit shape the asset was generated from.  Every
  solid lattice cell is split into six linear tetrahedra, giving a volumetric
  finite-element discretisation of the deformable body.
* Elasticity uses the inversion-tolerant stable Neo-Hookean model
  ``Psi = mu/2 (tr(F^T F) - 3) - mu (J - 1) + lam/2 (J - 1)^2`` whose first
  Piola-Kirchhoff stress is ``P = mu F + (lam (J-1) - mu) cof(F)``.
* Time integration is explicit symplectic (semi-implicit) Euler with a CFL
  limited step, graph-Laplacian strain-rate damping and penalty self-contact.
* The six kinematic walls are axis aligned planes whose inner half extent is
  given by the frozen scene schedule; nodes are projected onto the moving
  planes and the applied momentum change is accumulated as external impulse.
* Surface vertices (the reported samples) follow the lattice through fixed
  trilinear embedding weights, so the reported motion is a pure consequence of
  the simulated volumetric dynamics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:  # pragma: no cover - available in the platform image
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

try:  # pragma: no cover
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover
    _ndimage = None

from worldsolver.contracts import StateFrame

_LO = np.array([-1.15, -1.15, -0.82])
_HI = np.array([1.15, 1.15, 1.28])


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(np.sum((points - a) * ab, axis=-1) / float(np.dot(ab, ab)), 0.0, 1.0)
    return np.linalg.norm(points - (a + t[..., None] * ab), axis=-1)


def _octocat_field(p: np.ndarray) -> np.ndarray:
    """Implicit description of the frozen Octocat asset (negative inside)."""
    field = np.full(p.shape[:-1], 10.0)
    q = (p - np.array([0.0, 0.0, 0.35])) / np.array([0.55, 0.50, 0.68])
    field = np.minimum(field, np.linalg.norm(q, axis=-1) - 1.0)
    for cx in (-0.34, 0.34):
        tip = np.array([cx, 0.0, 1.18])
        base = np.array([cx * 0.72, 0.0, 0.78])
        field = np.minimum(field, _segment_distance(p, base, tip) / 0.18 - 1.0)
    for i in range(8):
        a = i * np.pi / 4.0 + np.pi / 8.0
        radial = np.array([np.cos(a), np.sin(a), 0.0])
        tangent = np.array([-np.sin(a), np.cos(a), 0.0])
        pts = [
            np.array([0.32 * radial[0], 0.32 * radial[1], -0.05]),
            np.array([0.62 * radial[0], 0.62 * radial[1], -0.30]),
            np.array([0.86 * radial[0], 0.86 * radial[1], -0.42]) + 0.12 * tangent,
            np.array([1.02 * radial[0], 1.02 * radial[1], -0.28]) + 0.28 * tangent,
        ]
        for j in range(3):
            radius = 0.15 - 0.025 * j
            field = np.minimum(
                field, _segment_distance(p, pts[j], pts[j + 1]) / radius - 1.0
            )
    return field


_HEX_TETS = np.array(
    [
        [0, 1, 3, 7],
        [0, 3, 2, 7],
        [0, 2, 6, 7],
        [0, 6, 4, 7],
        [0, 4, 5, 7],
        [0, 5, 1, 7],
    ],
    dtype=np.int64,
)


class BenchmarkSolver:
    """Explicit volumetric FEM solver with kinematic-wall and self contact."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))
        self.duration = float(cfg["duration"])
        comp = cfg["compactor"]
        self.open_h = float(comp["open_inner_half_extent"])
        self.closed_h = float(comp["closed_inner_half_extent"])
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float)

        frame0 = scene.initial_state()
        self.surface_x0 = np.asarray(frame0.position, dtype=float).copy()
        self.n_surface = int(self.surface_x0.shape[0])
        if frame0.mass is not None:
            self.sample_mass = np.asarray(frame0.mass, dtype=float).copy()
        else:
            self.sample_mass = np.full(self.n_surface, 1.0 / self.n_surface)
        self.total_mass = float(self.sample_mass.sum())

        self._build_lattice(h=0.12)
        self._build_embedding()
        self._setup_material()

        # dynamic state
        self.X = self.node_x0.copy()
        self.U = np.zeros_like(self.X)
        self.time = 0.0
        self.cum_impulse = np.zeros(3)
        self._pairs = np.zeros((0, 2), dtype=np.int64)
        self._pair_rest = np.zeros(0)
        self._steps = 0
        self._prev_time = 0.0

        # frozen positive momentum scale: total mass times the fastest wall speed
        span = self.open_h - self.closed_h
        t_in = comp["inward_interval"]
        t_out = comp["withdrawal_interval"]
        rate_in = span * 1.5 / max(float(t_in[1]) - float(t_in[0]), 1e-6)
        rate_out = span * 1.5 / max(float(t_out[1]) - float(t_out[0]), 1e-6)
        max_wall_speed = max(rate_in, rate_out)
        self.reference_momentum = max(self.total_mass * max_wall_speed, 1e-6)

    # ------------------------------------------------------------------
    # discretisation
    # ------------------------------------------------------------------
    def _build_lattice(self, h: float) -> None:
        self.h = float(h)
        origin = _LO - 1.5 * h
        upper = _HI + 1.5 * h
        dims = np.maximum(np.ceil((upper - origin) / h).astype(int), 2)
        self.origin = origin
        self.dims = dims  # number of cells per axis
        gx, gy, gz = (dims + 1)
        ax = origin[0] + h * np.arange(gx)
        ay = origin[1] + h * np.arange(gy)
        az = origin[2] + h * np.arange(gz)
        px, py, pz = np.meshgrid(ax, ay, az, indexing="ij")
        grid_pts = np.stack((px, py, pz), axis=-1)
        field = _octocat_field(grid_pts.reshape(-1, 3)).reshape(gx, gy, gz)
        self.grid_shape = (gx, gy, gz)

        # cell is solid when any corner is inside, or a surface vertex sits in it
        inside = field < 0.0
        corner = np.zeros(tuple(dims), dtype=bool)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    corner |= inside[dx:dx + dims[0], dy:dy + dims[1], dz:dz + dims[2]]
        idx = np.floor((self.surface_x0 - origin) / h).astype(int)
        idx = np.clip(idx, 0, dims - 1)
        corner[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        self.cell_idx = np.argwhere(corner)  # [Nc,3]
        if len(self.cell_idx) == 0:
            raise RuntimeError("empty lattice")

        # gather the 8 corner node ids of each solid cell
        node_lin = np.full((gx, gy, gz), -1, dtype=np.int64)
        offs = np.array(
            [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)], dtype=np.int64
        )
        used = self.cell_idx[:, None, :] + offs[None, :, :]
        flat = (used[..., 0] * gy + used[..., 1]) * gz + used[..., 2]
        uniq = np.unique(flat)
        node_lin.reshape(-1)[uniq] = np.arange(len(uniq))
        self.cell_nodes = node_lin.reshape(-1)[flat]  # [Nc,8]
        ni = uniq // (gy * gz)
        nj = (uniq // gz) % gy
        nk = uniq % gz
        self.node_x0 = np.stack(
            (origin[0] + h * ni, origin[1] + h * nj, origin[2] + h * nk), axis=1
        ).astype(float)
        self.n_nodes = len(self.node_x0)
        self.node_ijk = np.stack((ni, nj, nk), axis=1)
        self.node_lin = node_lin

        # tetrahedra: 6 per hexahedral cell, contiguous per cell
        tets = self.cell_nodes[:, _HEX_TETS.reshape(-1)].reshape(-1, 4)
        self.tets = tets
        self.n_cells = len(self.cell_idx)

    def _build_embedding(self) -> None:
        """Momentum-consistent embedding of the surface samples in the lattice.

        Surface displacements are trilinearly gathered from a diffused lattice
        displacement field, ``W = W_tri T^p``.  Because every row of ``W_tri``
        and of the diffusion operator ``T`` sums to one, and node masses are set
        to ``m0 * (T^T)^p W_tri^T 1``, the reported sample momentum equals the
        simulated lattice momentum exactly, so internal forces cannot create
        reported momentum.
        """
        h, origin, dims = self.h, self.origin, self.dims
        gx, gy, gz = self.grid_shape
        rel = (self.surface_x0 - origin) / h
        base = np.clip(np.floor(rel).astype(int), 0, dims - 1)
        frac = np.clip(rel - base, 0.0, 1.0)
        offs = np.array(
            [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)], dtype=np.int64
        )
        nodes = base[:, None, :] + offs[None, :, :]
        lin = (nodes[..., 0] * gy + nodes[..., 1]) * gz + nodes[..., 2]
        idx = self.node_lin.reshape(-1)[lin]
        if np.any(idx < 0):
            raise RuntimeError("surface sample outside solid lattice")
        wx = np.stack((1.0 - frac[:, 0], frac[:, 0]), axis=1)
        wy = np.stack((1.0 - frac[:, 1], frac[:, 1]), axis=1)
        wz = np.stack((1.0 - frac[:, 2], frac[:, 2]), axis=1)
        w = wx[:, offs[:, 0]] * wy[:, offs[:, 1]] * wz[:, offs[:, 2]]
        w = w / np.sum(w, axis=1, keepdims=True)
        self.emb_idx = idx.astype(np.int64)
        self.emb_w = w

        self._build_diffusion()
        col = np.bincount(
            self.emb_idx.reshape(-1), weights=self.emb_w.reshape(-1), minlength=self.n_nodes
        )
        mean_col = col.sum() / float(self.n_nodes)
        passes = 0
        while passes < 24 and col.min() < 0.15 * mean_col:
            col = self._diffuse_transpose(col)
            passes += 1
        self.diffusion_passes = passes
        col = np.maximum(col, 1e-3 * mean_col)
        self.node_report_mass = col * (self.total_mass / self.n_surface)

    def _build_diffusion(self) -> None:
        """Axis-neighbour averaging operator on the lattice node graph."""
        gy, gz = self.grid_shape[1], self.grid_shape[2]
        ijk = self.node_ijk
        neigh = np.full((self.n_nodes, 6), -1, dtype=np.int64)
        flat = self.node_lin.reshape(-1)
        col = 0
        for axis in range(3):
            for step in (-1, 1):
                shifted = ijk.copy()
                shifted[:, axis] += step
                valid = (shifted[:, axis] >= 0) & (
                    shifted[:, axis] < self.grid_shape[axis]
                )
                lin = (
                    np.clip(shifted[:, 0], 0, self.grid_shape[0] - 1) * gy
                    + np.clip(shifted[:, 1], 0, gy - 1)
                ) * gz + np.clip(shifted[:, 2], 0, gz - 1)
                cand = np.where(valid, flat[lin], -1)
                neigh[:, col] = cand
                col += 1
        self.node_neigh = neigh
        self.node_neigh_count = np.maximum((neigh >= 0).sum(axis=1), 1).astype(float)
        self.diffusion_alpha = 0.5

    def _diffuse(self, values: np.ndarray) -> np.ndarray:
        """Apply T once: (1-a) I + a * neighbour average (rows sum to one)."""
        a = self.diffusion_alpha
        acc = np.zeros_like(values)
        for c in range(6):
            nb = self.node_neigh[:, c]
            mask = nb >= 0
            if values.ndim == 1:
                acc[mask] += values[nb[mask]]
            else:
                acc[mask] += values[nb[mask], :]
        if values.ndim == 1:
            acc /= self.node_neigh_count
        else:
            acc /= self.node_neigh_count[:, None]
        return (1.0 - a) * values + a * acc

    def _diffuse_transpose(self, values: np.ndarray) -> np.ndarray:
        """Apply T^T once (column-wise scatter of the same weights)."""
        a = self.diffusion_alpha
        out = (1.0 - a) * values
        share = a * values / self.node_neigh_count
        for c in range(6):
            nb = self.node_neigh[:, c]
            mask = nb >= 0
            np.add.at(out, nb[mask], share[mask])
        return out

    def _smooth_nodal(self, values: np.ndarray) -> np.ndarray:
        out = values
        for _ in range(self.diffusion_passes):
            out = self._diffuse(out)
        return out

    def _gather(self, nodal: np.ndarray) -> np.ndarray:
        smoothed = self._smooth_nodal(nodal)
        return np.einsum("ij,ijk->ik", self.emb_w, smoothed[self.emb_idx])

    def _setup_material(self) -> None:
        X0 = self.node_x0
        t = self.tets
        e1 = X0[t[:, 1]] - X0[t[:, 0]]
        e2 = X0[t[:, 2]] - X0[t[:, 0]]
        e3 = X0[t[:, 3]] - X0[t[:, 0]]
        Dm = np.stack((e1, e2, e3), axis=2)  # [Ne,3,3] columns are edges
        det = np.linalg.det(Dm)
        # orient every tetrahedron positively so rest volumes are positive
        flip = det < 0.0
        if np.any(flip):
            t = t.copy()
            t[flip] = t[flip][:, [0, 2, 1, 3]]
            self.tets = t
            e1 = X0[t[:, 1]] - X0[t[:, 0]]
            e2 = X0[t[:, 2]] - X0[t[:, 0]]
            e3 = X0[t[:, 3]] - X0[t[:, 0]]
            Dm = np.stack((e1, e2, e3), axis=2)
            det = np.linalg.det(Dm)
        keep = det > 1e-12
        self.tets = self.tets[keep]
        Dm = Dm[keep]
        det = det[keep]
        self.Bm = np.linalg.inv(Dm)
        self.rest_volume = det / 6.0
        self.n_tets = len(self.tets)
        self.total_volume = float(self.rest_volume.sum())

        # elements reported as deformation gradients: the hexahedral cells
        self.tet_keep = keep
        self.cells_per_tet = np.repeat(np.arange(self.n_cells), 6)[keep]
        self.n_elements = self.n_cells

        # node masses follow the reported mass measure exactly, so the reported
        # momentum equals the simulated momentum of the lattice
        m = self.node_report_mass.astype(float)
        m = np.maximum(m, 1e-9 * self.total_mass)
        m *= self.total_mass / m.sum()
        self.node_mass = m
        self.inv_mass = 1.0 / m

        self.density = self.total_mass / max(self.total_volume, 1e-12)
        wave_speed = 22.0
        youngs = self.density * wave_speed * wave_speed
        nu = 0.35
        self.mu = youngs / (2.0 * (1.0 + nu))
        self.lam = youngs * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # edge list of the tetrahedral mesh for momentum-conserving damping
        ei = np.concatenate(
            [self.tets[:, a] for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
        )
        ej = np.concatenate(
            [self.tets[:, b] for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))]
        )
        lo = np.minimum(ei, ej)
        hi = np.maximum(ei, ej)
        edges = np.unique(np.stack((lo, hi), axis=1), axis=0)
        self.edges = edges
        deg = np.bincount(edges.reshape(-1), minlength=self.n_nodes).astype(float)
        self.node_degree = np.maximum(deg, 1.0)

        # explicit time step from the elastic wave CFL condition
        c = float(np.sqrt((self.lam + 2.0 * self.mu) / max(self.density, 1e-12)))
        dt_cfl = 0.25 * self.h / max(c, 1e-9)
        sample_dt = float(self.cfg["sample_dt"])
        substeps = int(max(1, np.ceil(sample_dt / min(dt_cfl, 2.0e-3))))
        substeps = int(min(substeps, 60))
        self.substeps = substeps
        self.dt = sample_dt / substeps
        self.damping = 0.25 * float(np.min(self.node_mass / self.node_degree)) / self.dt
        self.contact_stiffness = 4.0 * float(np.mean(self.node_mass)) / (self.dt * self.dt)
        self.contact_radius = 0.62 * self.h

    # ------------------------------------------------------------------
    # forces
    # ------------------------------------------------------------------
    def _deformation_gradients(self, X: np.ndarray) -> np.ndarray:
        t = self.tets
        e1 = X[t[:, 1]] - X[t[:, 0]]
        e2 = X[t[:, 2]] - X[t[:, 0]]
        e3 = X[t[:, 3]] - X[t[:, 0]]
        Ds = np.stack((e1, e2, e3), axis=2)
        return Ds @ self.Bm

    def _elastic_force(self, X: np.ndarray) -> np.ndarray:
        F = self._deformation_gradients(X)
        f0 = F[:, :, 0]
        f1 = F[:, :, 1]
        f2 = F[:, :, 2]
        c0 = np.cross(f1, f2)
        c1 = np.cross(f2, f0)
        c2 = np.cross(f0, f1)
        cof = np.stack((c0, c1, c2), axis=2)
        J = np.einsum("ij,ij->i", f0, c0)
        coeff = self.lam * (J - 1.0) - self.mu
        P = self.mu * F + coeff[:, None, None] * cof
        H = -self.rest_volume[:, None, None] * (P @ np.transpose(self.Bm, (0, 2, 1)))
        force = np.zeros_like(X)
        f_1 = H[:, :, 0]
        f_2 = H[:, :, 1]
        f_3 = H[:, :, 2]
        t = self.tets
        np.add.at(force, t[:, 1], f_1)
        np.add.at(force, t[:, 2], f_2)
        np.add.at(force, t[:, 3], f_3)
        np.add.at(force, t[:, 0], -(f_1 + f_2 + f_3))
        return force

    def _damping_force(self, U: np.ndarray) -> np.ndarray:
        e = self.edges
        dv = U[e[:, 1]] - U[e[:, 0]]
        f = self.damping * dv
        force = np.zeros_like(U)
        np.add.at(force, e[:, 0], f)
        np.add.at(force, e[:, 1], -f)
        return force

    def _rebuild_contact_pairs(self, X: np.ndarray) -> None:
        if cKDTree is None:
            self._pairs = np.zeros((0, 2), dtype=np.int64)
            self._pair_rest = np.zeros(0)
            return
        tree = cKDTree(X)
        pairs = tree.query_pairs(self.contact_radius * 1.6, output_type="ndarray")
        if len(pairs) == 0:
            self._pairs = np.zeros((0, 2), dtype=np.int64)
            self._pair_rest = np.zeros(0)
            return
        rest = np.linalg.norm(
            self.node_x0[pairs[:, 0]] - self.node_x0[pairs[:, 1]], axis=1
        )
        keep = rest > 1.9 * self.h  # not lattice neighbours: genuine self contact
        self._pairs = pairs[keep].astype(np.int64)
        self._pair_rest = rest[keep]

    def _contact_force(self, X: np.ndarray) -> np.ndarray:
        force = np.zeros_like(X)
        if len(self._pairs) == 0:
            return force
        a = self._pairs[:, 0]
        b = self._pairs[:, 1]
        d = X[a] - X[b]
        dist = np.linalg.norm(d, axis=1)
        active = dist < self.contact_radius
        if not np.any(active):
            return force
        a = a[active]
        b = b[active]
        d = d[active]
        dist = np.maximum(dist[active], 1e-9)
        n = d / dist[:, None]
        pen = self.contact_radius - dist
        mag = self.contact_stiffness * pen
        f = mag[:, None] * n
        np.add.at(force, a, f)
        np.add.at(force, b, -f)
        return force

    # ------------------------------------------------------------------
    # wall contact
    # ------------------------------------------------------------------
    def _wall_half_extent(self, time: float) -> float:
        return float(self.scene.inner_half_extent(time))

    def _apply_walls(self, X: np.ndarray, U: np.ndarray, time: float) -> np.ndarray:
        """Inelastic projection onto the six moving planes; returns impulse."""
        eps = 1e-4
        hw = self._wall_half_extent(time) + 0.5 * self.h - eps
        dh = self._wall_half_extent(time + self.dt) - self._wall_half_extent(
            max(time - self.dt, 0.0)
        )
        wall_speed = dh / (2.0 * self.dt) if time > 0.0 else 0.0
        impulse = np.zeros(3)
        for axis in range(3):
            over_hi = X[:, axis] > hw
            over_lo = X[:, axis] < -hw
            if np.any(over_hi):
                dv = (-wall_speed) - U[over_hi, axis]
                impulse[axis] += float(np.sum(self.node_mass[over_hi] * dv))
                X[over_hi, axis] = hw
                U[over_hi, axis] = -wall_speed
            if np.any(over_lo):
                dv = wall_speed - U[over_lo, axis]
                impulse[axis] += float(np.sum(self.node_mass[over_lo] * dv))
                X[over_lo, axis] = -hw
                U[over_lo, axis] = wall_speed
        return impulse

    # ------------------------------------------------------------------
    # time stepping
    # ------------------------------------------------------------------
    def _step(self, dt: float, time: float) -> np.ndarray:
        X, U = self.X, self.U
        force = self._elastic_force(X)
        force += self._damping_force(U)
        force += self._contact_force(X)
        force += self.node_mass[:, None] * self.gravity[None, :]
        U = U + dt * force * self.inv_mass[:, None]
        X = X + dt * U
        impulse = self._apply_walls(X, U, time + dt)
        self.X, self.U = X, U
        return impulse

    def _advance(self, target_time: float) -> np.ndarray:
        impulse = np.zeros(3)
        n = self.substeps
        dt = self.dt
        remaining = target_time - self.time
        if remaining <= 1e-12:
            return impulse
        steps = max(1, int(round(remaining / dt)))
        dt_eff = remaining / steps
        for i in range(steps):
            if self._steps % 20 == 0:
                self._rebuild_contact_pairs(self.X)
            impulse += self._step(dt_eff, self.time)
            self.time += dt_eff
            self._steps += 1
        self.time = target_time
        return impulse

    def advance_to(self, time: float) -> StateFrame:
        dt_elapsed = max(float(time) - self._prev_time, self.dt)
        impulse = self._advance(float(time))
        self.cum_impulse += impulse
        wall_force_avg = impulse / dt_elapsed
        self._prev_time = float(time)

        position = self.surface_x0 + self._gather(self.X - self.node_x0)
        velocity = self._gather(self.U)

        F = self._deformation_gradients(self.X)
        elem_F = np.tile(np.eye(3), (self.n_elements, 1, 1))
        sums = np.zeros((self.n_elements, 3, 3))
        counts = np.zeros(self.n_elements)
        np.add.at(sums, self.cells_per_tet, F)
        np.add.at(counts, self.cells_per_tet, 1.0)
        valid = counts > 0
        elem_F[valid] = sums[valid] / counts[valid, None, None]

        n = self.n_surface
        domain_weights = np.ones((1, n), dtype=float)
        gravity_force = np.zeros(3)
        ext_force = np.stack((gravity_force, wall_force_avg), axis=0)[None, :, :]

        frame = StateFrame(
            time=float(time),
            position=position.astype(float),
            velocity=velocity.astype(float),
            mass=self.sample_mass.astype(float),
            attributes={"mass_quantity": self.sample_mass.astype(float)},
            fields={
                "budget_cumulative_external_impulse": self.cum_impulse[None, :].astype(float),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "budget_domain_weights": domain_weights,
                "budget_external_force": ext_force.astype(float),
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": elem_F.astype(np.float32),
            },
            scalars={"reference_momentum": float(self.reference_momentum)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_nodes": int(self.n_nodes),
            "n_tets": int(self.n_tets),
            "n_elements": int(self.n_elements),
            "substeps_per_sample": int(self.substeps),
            "dt": float(self.dt),
            "diffusion_passes": int(self.diffusion_passes),
            "youngs_modulus": float(2.0 * self.mu * (1.0 + self.lam / (2.0 * self.mu + self.lam)) if (2*self.mu+self.lam) else 0.0),
        }
