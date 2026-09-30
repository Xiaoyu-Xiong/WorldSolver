"""Deformable torus solver: corotational tetrahedral FEM with constraint ground
contact and penalty self-collision.

Physics summary
---------------
* The frozen surface mesh is a structured torus grid (n_major rings of n_minor
  cross-section samples).  It is turned into a conforming tetrahedral mesh of
  the solid tube by adding one centreline node per cross-section; every
  cross-section fan triangle is extruded to the next ring and split into three
  tetrahedra with matching quad-face diagonals.
* Elasticity: corotational linear model P = 2 mu (F - R) + lam tr(R^T F - I) R
  with a volumetric barrier that reacts when an element approaches inversion.
* Dissipation: Kelvin-Voigt viscous stress built from the corotated strain
  rate, capped by the explicit stability bound.
* Integration: lumped-mass symplectic (semi-implicit) Euler at a CFL-limited
  substep; fully deterministic.
* Ground: hard positional constraint at the frozen ground plane with
  restitution and Coulomb friction on node velocities (no penetration).
* Self-collision: penalty repulsion with normal damping between surface
  samples close in space but far apart on the mesh.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


def _infer_ring_size(vertices: np.ndarray, minor_r: float, major_r: float) -> int:
    """Recover the structured torus stride (samples per cross-section)."""
    n = len(vertices)
    center = vertices.mean(axis=0)
    best_s, best_err = 4, np.inf
    for s in range(4, max(5, n // 3) + 1):
        if n % s:
            continue
        blocks = vertices.reshape(n // s, s, 3)
        ring_c = blocks.mean(axis=1)
        rad = np.linalg.norm(blocks - ring_c[:, None, :], axis=2)
        dist = np.linalg.norm(ring_c - center, axis=1)
        err = (abs(float(rad.mean()) - minor_r) + float(rad.std())
               + abs(float(dist.mean()) - major_r) + float(dist.std()))
        if err < best_err:
            best_err, best_s = err, s
    return int(best_s)


def _scatter(out: np.ndarray, index: np.ndarray, values: np.ndarray) -> None:
    """Accumulate ``values`` rows into ``out`` at ``index``."""
    n = out.shape[0]
    for k in range(3):
        out[:, k] += np.bincount(index, weights=values[:, k], minlength=n)[:n]


class BenchmarkSolver:
    """Explicit corotational tetrahedral FEM solver for the falling torus."""

    CFL_SAFETY = 0.25
    MAX_SUBSTEPS = 400
    PAIR_REFRESH = 8

    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))
        surface = np.asarray(scene.vertices, dtype=float).copy()
        self.n_surf = int(len(surface))
        minor_r = float(cfg.get("minor_radius", 0.17))
        major_r = float(cfg.get("major_radius", 0.47))
        self.n_minor = _infer_ring_size(surface, minor_r, major_r)
        self.n_major = self.n_surf // self.n_minor
        rings = surface.reshape(self.n_major, self.n_minor, 3)
        centrelines = rings.mean(axis=1)
        self.x0 = np.vstack([surface, centrelines])
        self.n_node = int(len(self.x0))

        self._build_tets()
        self._build_material(minor_r)
        self._build_average_operator()

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.restitution = float(cfg.get("restitution", 0.0))
        self.friction = float(cfg.get("friction", 0.0))
        self.thickness = float(cfg["contact_length_scale"])
        self.sample_dt = float(cfg["sample_dt"])

        self.x = self.x0.copy()
        v0 = np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float)
        self.v = np.tile(v0, (self.n_node, 1))
        self.time = 0.0
        self._steps = 0
        self._pairs = np.zeros((0, 2), dtype=np.int64)
        self._pair_age = 1 << 30
        self._impulse = np.zeros((self.n_node, 3), dtype=float)
        self._impulse_window = 0.0
        self._min_det = 1.0
        self._min_gap = float("inf")
        self._max_pairs = 0
        self._resets = 0
        self._x_safe = self.x.copy()
        self._v_safe = self.v.copy()
        self._set_contact_parameters()

    # ---------------------------------------------------------------- mesh
    def _build_tets(self) -> None:
        """Conforming tetrahedralisation of the solid tube.

        Cross-section ``i`` holds samples ``i*n_minor + j`` plus centreline node
        ``n_surf + i``.  Each surface quad (i,j)-(i,j+1)-(i+1,j+1)-(i+1,j) is
        joined to the two centreline nodes, giving a prism split into three
        tetrahedra with quad-face diagonals chosen consistently so neighbouring
        prisms share identical triangles.
        """
        nm, nj, ns = self.n_major, self.n_minor, self.n_surf
        i = np.arange(nm)
        j = np.arange(nj)
        ii, jj = np.meshgrid(i, j, indexing="ij")
        ii = ii.ravel()
        jj = jj.ravel()
        ip = (ii + 1) % nm
        jp = (jj + 1) % nj
        a = ii * nj + jj          # surface (i, j)
        b = ii * nj + jp          # surface (i, j+1)
        c = ns + ii               # centreline i
        d = ip * nj + jj          # surface (i+1, j)
        e = ip * nj + jp          # surface (i+1, j+1)
        f = ns + ip               # centreline i+1
        tets = np.concatenate(
            [
                np.stack([a, b, c, f], axis=1),
                np.stack([a, b, f, e], axis=1),
                np.stack([a, e, f, d], axis=1),
            ],
            axis=0,
        ).astype(np.int64)

        X = self.x0[tets]
        dm = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=2)
        det = np.linalg.det(dm)
        flip = det < 0.0
        if np.any(flip):
            tets[flip] = tets[flip][:, [0, 2, 1, 3]]
            X = self.x0[tets]
            dm = np.stack(
                [X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=2
            )
            det = np.linalg.det(dm)
        keep = det > 1e-12
        self.tets = np.ascontiguousarray(tets[keep])
        dm = dm[keep]
        self.rest_volume = det[keep] / 6.0
        self.dm_inv = np.linalg.inv(dm)
        self.n_tet = int(len(self.tets))

    # ------------------------------------------------------------ material
    def _build_material(self, minor_r: float) -> None:
        """Lumped masses, Lame parameters and the CFL-limited substep."""
        total_mass = float(self.cfg["subject_mass"])
        volume = float(self.rest_volume.sum())
        self.density = total_mass / max(volume, 1e-12)

        node_mass = np.zeros(self.n_node, dtype=float)
        share = self.density * self.rest_volume / 4.0
        for k in range(4):
            node_mass += np.bincount(
                self.tets[:, k], weights=share, minlength=self.n_node
            )[: self.n_node]
        node_mass = np.maximum(node_mass, 1e-9)
        node_mass *= total_mass / float(node_mass.sum())
        self.node_mass = node_mass
        self.inv_mass = 1.0 / node_mass

        # Exported per-sample mass: surface lumped mass plus an even share of
        # the hidden centreline mass of the same cross-section, so the exported
        # subset carries the whole frozen subject mass.
        surf_mass = node_mass[: self.n_surf].copy()
        centre_mass = node_mass[self.n_surf :]
        surf_mass = surf_mass + np.repeat(centre_mass / self.n_minor, self.n_minor)
        surf_mass *= total_mass / float(surf_mass.sum())
        self.surface_mass = surf_mass

        # Soft-solid modulus: frozen `stiffness` scales the natural pressure
        # obtained from the subject weight acting on the tube cross-section.
        g_mag = float(np.linalg.norm(np.asarray(self.cfg["gravity"], dtype=float)))
        g_mag = g_mag if g_mag > 1e-9 else 9.81
        pressure_scale = total_mass * g_mag / (np.pi * minor_r * minor_r)
        self.young = float(self.cfg.get("stiffness", 10.0)) * pressure_scale
        self.poisson = 0.35
        self.mu = self.young / (2.0 * (1.0 + self.poisson))
        self.lam = (
            self.young
            * self.poisson
            / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        )
        self.k_barrier = 4.0 * self.lam
        self.j_min = 0.35

        wave = float(np.sqrt((self.lam + 2.0 * self.mu) / max(self.density, 1e-12)))
        self.wave_speed = wave
        h_min = self._min_altitude()
        self.dt_max = self.CFL_SAFETY * h_min / max(wave, 1e-9)

    def _min_altitude(self) -> float:
        """Smallest tetrahedron altitude, the length scale of explicit CFL."""
        X = self.x0[self.tets]
        p0, p1, p2, p3 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        areas = np.stack(
            [
                np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1),
                np.linalg.norm(np.cross(p1 - p0, p3 - p0), axis=1),
                np.linalg.norm(np.cross(p2 - p0, p3 - p0), axis=1),
                np.linalg.norm(np.cross(p2 - p1, p3 - p1), axis=1),
            ],
            axis=1,
        ) * 0.5
        max_area = np.maximum(areas.max(axis=1), 1e-12)
        altitude = 3.0 * self.rest_volume / max_area
        return float(max(np.min(altitude), 1e-6))

    def _build_average_operator(self) -> None:
        """Volume-weighted tet -> surface-sample averaging operator."""
        rows = np.repeat(self.tets.reshape(-1), 1)
        cols = np.repeat(np.arange(self.n_tet, dtype=np.int64), 4)
        weights = np.repeat(self.rest_volume, 4)
        mask = rows < self.n_surf
        mat = sp.coo_matrix(
            (weights[mask], (rows[mask], cols[mask])),
            shape=(self.n_surf, self.n_tet),
        ).tocsr()
        totals = np.asarray(mat.sum(axis=1)).ravel()
        totals[totals <= 0.0] = 1.0
        self.avg_op = sp.diags(1.0 / totals) @ mat

    def _set_contact_parameters(self) -> None:
        """Substep count plus stable penalty-contact stiffness and damping."""
        self.substeps = int(max(1, min(self.MAX_SUBSTEPS,
                                       int(np.ceil(self.sample_dt / self.dt_max)))))
        self.dt = self.sample_dt / self.substeps
        m_min = float(np.min(self.node_mass[: self.n_surf]))
        # Explicit stability: k dt^2 / m <= factor  (factor < 4 for symplectic Euler).
        k_stable = 0.25 * m_min / (self.dt * self.dt)
        # Enough stiffness to carry the subject weight over one thickness.
        total_mass = float(self.cfg["subject_mass"])
        g_mag = float(np.linalg.norm(self.gravity)) or 9.81
        k_load = 4.0 * total_mass * g_mag / max(self.thickness, 1e-6)
        self.k_contact = float(min(k_stable, k_load))
        self.c_contact = 0.5 * float(np.sqrt(self.k_contact * m_min))
        # Kelvin-Voigt viscosity, capped by the explicit damping bound.
        omega_max = 2.0 * self.wave_speed / max(self._min_altitude(), 1e-9)
        eta_cap = 1.6 / (omega_max * omega_max * self.dt)
        self.eta = float(min(float(self.cfg.get("damping", 1.0)) * 1e-3, eta_cap))

    # -------------------------------------------------------------- physics
    def _deformation_gradient(self, x: np.ndarray) -> np.ndarray:
        X = x[self.tets]
        ds = np.stack(
            [X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=2
        )
        return ds @ self.dm_inv

    def _internal_forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Corotational elastic + viscous + inversion-barrier nodal forces."""
        F = self._deformation_gradient(x)
        det = np.linalg.det(F)
        self._min_det = min(self._min_det, float(det.min()))

        # Polar rotation from a batched SVD, with reflection repair.
        U, _, Vt = np.linalg.svd(F)
        sign = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
        sign[sign == 0.0] = 1.0
        U = U.copy()
        U[:, :, 2] *= sign[:, None]
        R = U @ Vt

        # First Piola-Kirchhoff stress of the corotated linear model.
        S = np.einsum("nki,nkj->nij", R, F)          # R^T F
        trace = np.trace(S, axis1=1, axis2=2) - 3.0
        eye = np.eye(3)[None, :, :]
        P = 2.0 * self.mu * (F - R) + self.lam * trace[:, None, None] * R

        # Volumetric barrier: resists elements approaching inversion.
        j_safe = np.maximum(det, 1e-4)
        under = j_safe < self.j_min
        if np.any(under):
            cof = np.empty_like(F)
            f0, f1, f2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
            cof[:, :, 0] = np.cross(f1, f2)
            cof[:, :, 1] = np.cross(f2, f0)
            cof[:, :, 2] = np.cross(f0, f1)
            gain = np.where(under, -self.k_barrier * (j_safe - self.j_min), 0.0)
            P = P + gain[:, None, None] * cof

        # Kelvin-Voigt viscous stress from the corotated strain rate.
        if self.eta > 0.0:
            Vg = np.stack(
                [v[self.tets][:, 1] - v[self.tets][:, 0],
                 v[self.tets][:, 2] - v[self.tets][:, 0],
                 v[self.tets][:, 3] - v[self.tets][:, 0]],
                axis=2,
            ) @ self.dm_inv
            Rt_dF = np.einsum("nki,nkj->nij", R, Vg)
            sym = 0.5 * (Rt_dF + np.swapaxes(Rt_dF, 1, 2))
            tr_rate = np.trace(sym, axis1=1, axis2=2)
            Pv = R @ (2.0 * self.eta * self.mu * sym
                      + self.eta * self.lam * tr_rate[:, None, None] * eye)
            P = P + Pv

        H = -(self.rest_volume[:, None, None] * P) @ np.swapaxes(self.dm_inv, 1, 2)
        force = np.zeros((self.n_node, 3), dtype=float)
        _scatter(force, self.tets[:, 1], H[:, :, 0])
        _scatter(force, self.tets[:, 2], H[:, :, 1])
        _scatter(force, self.tets[:, 3], H[:, :, 2])
        _scatter(force, self.tets[:, 0], -(H[:, :, 0] + H[:, :, 1] + H[:, :, 2]))
        return force

    def _topology_far(self, pairs: np.ndarray) -> np.ndarray:
        """Keep only sample pairs that are far apart on the mesh grid."""
        if len(pairs) == 0:
            return np.zeros(0, dtype=bool)
        ia, ja = np.divmod(pairs[:, 0], self.n_minor)
        ib, jb = np.divmod(pairs[:, 1], self.n_minor)
        di = np.abs(ia - ib)
        di = np.minimum(di, self.n_major - di)
        dj = np.abs(ja - jb)
        dj = np.minimum(dj, self.n_minor - dj)
        return (di > 2) | (dj > 2)

    def _rebuild_pairs(self) -> None:
        """Broad-phase self-collision candidates from a KD-tree query."""
        surf = self.x[: self.n_surf]
        tree = cKDTree(surf)
        raw = tree.query_pairs(r=1.8 * self.thickness, output_type="ndarray")
        if len(raw):
            raw = raw[np.lexsort((raw[:, 1], raw[:, 0]))]
            raw = raw[self._topology_far(raw)]
        self._pairs = np.ascontiguousarray(raw.astype(np.int64).reshape(-1, 2))
        self._pair_age = 0
        self._max_pairs = max(self._max_pairs, int(len(self._pairs)))

    def _self_collision_forces(self, force: np.ndarray) -> None:
        """Penalty repulsion with normal damping on close surface samples."""
        if self._pair_age >= self.PAIR_REFRESH:
            self._rebuild_pairs()
        pairs = self._pairs
        if len(pairs) == 0:
            return
        pa, pb = pairs[:, 0], pairs[:, 1]
        delta = self.x[pa] - self.x[pb]
        dist = np.linalg.norm(delta, axis=1)
        active = (dist < self.thickness) & (dist > 1e-9)
        if not np.any(active):
            return
        pa, pb = pa[active], pb[active]
        delta = delta[active]
        dist = dist[active]
        normal = delta / dist[:, None]
        depth = self.thickness - dist
        rel = np.einsum("ij,ij->i", self.v[pa] - self.v[pb], normal)
        magnitude = self.k_contact * depth - self.c_contact * np.minimum(rel, 0.0)
        magnitude = np.maximum(magnitude, 0.0)
        impulse = magnitude[:, None] * normal
        _scatter(force, pa, impulse)
        _scatter(force, pb, -impulse)

    def _apply_ground(self) -> None:
        """Hard ground constraint: no penetration, restitution and friction."""
        z = self.x[:, 2]
        hit = np.flatnonzero(z < self.ground_z)
        if hit.size == 0:
            return
        self.x[hit, 2] = self.ground_z
        vn = self.v[hit, 2]
        vt = self.v[hit, :2]
        v_before = self.v[hit].copy()
        approaching = vn < 0.0
        vn_new = np.where(approaching, -self.restitution * vn, vn)
        dvn = np.abs(vn_new - vn)
        speed_t = np.linalg.norm(vt, axis=1)
        scale = np.ones_like(speed_t)
        moving = speed_t > 1e-12
        scale[moving] = np.maximum(
            0.0, 1.0 - self.friction * dvn[moving] / speed_t[moving]
        )
        self.v[hit, 2] = vn_new
        self.v[hit, :2] = vt * scale[:, None]
        self._impulse[hit] += self.node_mass[hit, None] * (self.v[hit] - v_before)

    # ------------------------------------------------------------ stepping
    def _step(self, dt: float) -> None:
        force = self.node_mass[:, None] * self.gravity[None, :]
        force += self._internal_forces(self.x, self.v)
        self._self_collision_forces(force)
        self.v += dt * self.inv_mass[:, None] * force
        np.clip(self.v, -60.0, 60.0, out=self.v)
        self.x += dt * self.v
        self._apply_ground()
        self._pair_age += 1
        self._steps += 1
        healthy = (
            bool(np.isfinite(self.x).all())
            and bool(np.isfinite(self.v).all())
            and float(np.abs(self.x).max()) < 1.0e3
        )
        if healthy:
            self._x_safe = self.x.copy()
            self._v_safe = self.v.copy()
        else:
            self.x = self._x_safe.copy()
            self.v = np.zeros_like(self.v)
            self._pair_age = 1 << 30
            self._resets += 1

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        t_prev = self.time
        self._impulse[:] = 0.0
        if target > t_prev + 1e-15:
            n_target = int(round(target / self.dt))
            while self._steps < n_target:
                self._step(self.dt)
            self.time = target
        return self._frame(target, max(target - t_prev, 0.0))

    # -------------------------------------------------------------- outputs
    def _gap_field(self) -> np.ndarray:
        """Signed gap per surface sample: ground plane and self-contact."""
        surf = self.x[: self.n_surf]
        gap = surf[:, 2] - self.ground_z
        tree = cKDTree(surf)
        raw = tree.query_pairs(r=1.5 * self.thickness, output_type="ndarray")
        if len(raw):
            raw = raw[self._topology_far(raw)]
        if len(raw):
            dist = np.linalg.norm(surf[raw[:, 0]] - surf[raw[:, 1]], axis=1)
            self_gap = dist - 0.5 * self.thickness
            np.minimum.at(gap, raw[:, 0], self_gap)
            np.minimum.at(gap, raw[:, 1], self_gap)
        gap = np.nan_to_num(gap, nan=0.0, posinf=1.0e3, neginf=-1.0e3)
        self._min_gap = min(self._min_gap, float(gap.min()))
        return np.ascontiguousarray(gap)

    def _frame(self, time: float, window: float) -> StateFrame:
        n = self.n_surf
        F = self._deformation_gradient(self.x)
        F_node = np.asarray(
            self.avg_op @ np.ascontiguousarray(F.reshape(self.n_tet, 9))
        ).reshape(n, 3, 3)
        det = np.linalg.det(F_node)
        bad = ~np.isfinite(det) | (det < 1.0e-6)
        if np.any(bad):
            F_node[bad] = np.eye(3)
        external = self.surface_mass[:, None] * self.gravity[None, :]
        if window > 0.0:
            external = external + self._impulse[:n] / window
        frame = StateFrame(
            time=float(time),
            position=np.ascontiguousarray(self.x[:n]).copy(),
            velocity=np.ascontiguousarray(self.v[:n]).copy(),
            mass=self.surface_mass.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "mass_quantity": self.surface_mass.copy(),
                "deformation_gradient": np.ascontiguousarray(F_node),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": np.ascontiguousarray(external),
                "contact_signed_distance": self._gap_field(),
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit corotational tet FEM, constraint ground + penalty self-collision",
            "nodes": int(self.n_node),
            "surface_samples": int(self.n_surf),
            "tetrahedra": int(self.n_tet),
            "grid": [int(self.n_major), int(self.n_minor)],
            "substeps_per_sample": int(self.substeps),
            "dt": float(self.dt),
            "steps": int(self._steps),
            "young_modulus_pa": float(self.young),
            "poisson_ratio": float(self.poisson),
            "density": float(self.density),
            "wave_speed": float(self.wave_speed),
            "viscosity_s": float(self.eta),
            "contact_stiffness": float(self.k_contact),
            "max_self_pairs": int(self._max_pairs),
            "min_det_F": float(self._min_det),
            "min_signed_gap": float(self._min_gap),
            "state_resets": int(self._resets),
        }
