"""Explicit finite-element solver for the pneumatic bubble chamber scene.

Physical model
--------------
The silicone chamber wall is a three-dimensional solid.  It is discretised
with linear tetrahedra: the frozen surface mesh (448 nodes) supplies the inner
and outer wall faces and additional node levels are inserted through the wall
thickness so the bending of the thick shell is resolved.  The constitutive law
is a compressible Neo-Hookean material (log-J volumetric term) using the
frozen Young's modulus, Poisson ratio and density, regularised with a
Kelvin-Voigt viscous stress for high-frequency damping.

The prescribed internal pressure is applied as a follower traction on the
cavity-facing surface triangles (recomputed from the current geometry every
step).  Base and rim attachment nodes are clamped, matching the frozen
``fixed`` attribute.

Time integration is explicit symplectic Euler (central differences) with
mass-lumped nodes.  Quasi-static mass scaling (uniform, reported in the
diagnostics) keeps the explicit step affordable for this slow pressure ramp
without changing the static equilibrium.  The step is guarded: the determinant
of every element deformation gradient and the nodal velocities are checked and
the step is automatically halved on any violation, so states stay finite and
deterministic at every requested sample time.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY_Z = -9.81         # m/s^2 (up axis is +z)
RADIAL_GAPS = 3           # tet layers through the wall thickness
DT_TARGET = 3.5e-5        # preferred explicit time step (s)
CFL_SAFETY = 0.30
MASS_SCALE_CAP = 1.0e5
DAMPING_XI = 2.0          # Kelvin-Voigt viscosity (Pa*s)
J_MIN_ACCEPT = 0.02       # minimum accepted det(F) per step
V_MAX_ACCEPT = 60.0       # m/s, blow-up guard

# Freudenthal/Kuhn decomposition of a structured hexahedron into 6 tets.
# Corner bit encoding: bit0 = +theta, bit1 = +ring, bit2 = +radial level.
KUHN_TETS = (
    (0, 1, 3, 7),
    (0, 1, 5, 7),
    (0, 2, 3, 7),
    (0, 2, 6, 7),
    (0, 4, 5, 7),
    (0, 4, 6, 7),
)


def _inv_det(m: np.ndarray):
    """Batched 3x3 inverse and determinant via cofactors."""
    a, b, c = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    d, e, f = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    g, h, i = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    c00 = e * i - f * h
    c01 = f * g - d * i
    c02 = d * h - e * g
    det = a * c00 + b * c01 + c * c02
    inv = np.empty_like(m)
    inv[:, 0, 0] = c00
    inv[:, 0, 1] = c * h - b * i
    inv[:, 0, 2] = b * f - c * e
    inv[:, 1, 0] = c01
    inv[:, 1, 1] = a * i - c * g
    inv[:, 1, 2] = c * d - a * f
    inv[:, 2, 0] = c02
    inv[:, 2, 1] = b * g - a * h
    inv[:, 2, 2] = a * e - b * d
    inv /= det[:, None, None]
    return inv, det


def _det3(m: np.ndarray) -> np.ndarray:
    a, b, c = m[:, 0, 0], m[:, 0, 1], m[:, 0, 2]
    d, e, f = m[:, 1, 0], m[:, 1, 1], m[:, 1, 2]
    g, h, i = m[:, 2, 0], m[:, 2, 1], m[:, 2, 2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


class BenchmarkSolver:
    """Tetrahedral finite-element dynamics of the pressurised chamber."""

    def __init__(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", {}) or {}
        frame0 = scene.initial_state()
        pos = np.array(frame0.position, dtype=np.float64)
        faces = np.array(frame0.fields["surface_faces"], dtype=np.int64)
        fixed_attribute = np.array(frame0.attributes["fixed"], dtype=np.float64)
        n = pos.shape[0]
        half = n // 2

        # --- parse the ring/theta structure of the frozen surface mesh -----
        zc = pos[:half, 2]
        bounds = [0]
        for idx in range(1, half):
            if abs(zc[idx] - zc[bounds[-1]]) > 1e-9:
                bounds.append(idx)
        bounds.append(half)
        sizes = np.diff(np.asarray(bounds))
        if len(sizes) == 0 or np.any(sizes != sizes[0]):
            raise ValueError("unsupported surface mesh layout")
        nt = int(sizes[0])
        nr = int(len(sizes))
        if 2 * nr * nt != n:
            raise ValueError("unsupported surface mesh layout")
        if np.linalg.norm(pos[:half, :2], axis=1).mean() < np.linalg.norm(
            pos[half:, :2], axis=1
        ).mean():
            raise ValueError("expected outer surface nodes first")

        # --- build the volumetric grid (extra levels through thickness) ----
        levels = RADIAL_GAPS + 1
        inner = pos[half:]
        outer = pos[:half]
        grid = np.empty((levels, half, 3), dtype=np.float64)
        for lev in range(levels):
            s = lev / RADIAL_GAPS
            grid[lev] = (1.0 - s) * inner + s * outer
        ids = np.empty((levels, half), dtype=np.int64)
        ids[0] = np.arange(half, n)          # inner surface (asset order)
        ids[levels - 1] = np.arange(0, half)  # outer surface (asset order)
        for lev in range(1, levels - 1):
            ids[lev] = n + (lev - 1) * half + np.arange(half)
        n_nodes = n + (levels - 2) * half
        X = np.zeros((n_nodes, 3), dtype=np.float64)
        X[ids.reshape(-1)] = grid.reshape(-1, 3)

        # --- tetrahedra ----------------------------------------------------
        tets: list[tuple[int, int, int, int]] = []
        for ring in range(nr - 1):
            for lev in range(RADIAL_GAPS):
                for i in range(nt):
                    k = (i + 1) % nt
                    corner_nodes = [
                        ids[lev + (bit >> 2), (ring + ((bit >> 1) & 1)) * nt + (i if not (bit & 1) else k)]
                        for bit in range(8)
                    ]
                    for tet in KUHN_TETS:
                        tets.append(tuple(corner_nodes[b] for b in tet))
        tn = np.asarray(tets, dtype=np.int64)

        def material_edges(points: np.ndarray) -> np.ndarray:
            return np.stack(
                (points[:, 1] - points[:, 0], points[:, 2] - points[:, 0], points[:, 3] - points[:, 0]),
                axis=2,
            )

        xm = X[tn]
        Dm = material_edges(xm)
        if np.any(_det3(Dm) < 0.0):  # enforce positive orientation
            flip = _det3(Dm) < 0.0
            tmp = tn[flip, 1].copy()
            tn[flip, 1] = tn[flip, 2]
            tn[flip, 2] = tmp
            xm = X[tn]
            Dm = material_edges(xm)
        det_m = _det3(Dm)
        if np.any(det_m <= 0.0):
            raise ValueError("degenerate reference tetrahedron")
        V0 = np.abs(det_m) / 6.0
        Dm_inv, _ = _inv_det(Dm)
        Dm_invT = np.transpose(Dm_inv, (0, 2, 1))

        # --- element size (for the explicit stability estimate) ------------
        e1 = xm[:, 1] - xm[:, 0]
        e2 = xm[:, 2] - xm[:, 0]
        e3 = xm[:, 3] - xm[:, 0]
        areas = np.stack(
            (
                0.5 * np.linalg.norm(np.cross(xm[:, 2] - xm[:, 1], xm[:, 3] - xm[:, 1]), axis=1),
                0.5 * np.linalg.norm(np.cross(e2, e3), axis=1),
                0.5 * np.linalg.norm(np.cross(e1, e3), axis=1),
                0.5 * np.linalg.norm(np.cross(e1, e2), axis=1),
            ),
            axis=1,
        )
        h_min = float((3.0 * V0[:, None] / areas).min())

        # --- material and loading ------------------------------------------
        mat = config.get("material_reference", {}) or {}
        rho = float(mat.get("density_kg_m3", 1070.0))
        young = float(mat.get("youngs_modulus_pa", 100000.0))
        poisson = float(mat.get("poisson_ratio", 0.499))
        loading = config.get("loading", {}) or {}
        self.p_max = float(loading.get("pressure_pa", 4000.0))
        self.ramp_start = float(loading.get("ramp_start_s", 0.25))
        self.ramp_end = float(loading.get("ramp_end_s", 2.5))
        self.mu = young / (2.0 * (1.0 + poisson))
        self.lam = (
            young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
            if poisson < 0.5
            else 10.0 * young
        )
        self.xi = DAMPING_XI

        # --- masses ---------------------------------------------------------
        weights = np.broadcast_to(V0[:, None] / 4.0, tn.shape).reshape(-1)
        m_node = np.bincount(tn.reshape(-1), weights=weights, minlength=n_nodes) * rho
        m_export = m_node[:n].copy()
        for lev in range(1, levels - 1):
            m_export[ids[0]] += 0.5 * m_node[ids[lev]]
            m_export[ids[levels - 1]] += 0.5 * m_node[ids[lev]]
        if not np.all(m_export > 0.0):
            raise ValueError("non-positive lumped mass")

        # --- clamped attachment regions (base and rim, full thickness) ------
        ring_of_row = np.repeat(np.arange(nr), nt)
        fixed_row = (ring_of_row == 0) | (ring_of_row == nr - 1)
        fixed_mask = np.zeros(n_nodes, dtype=bool)
        for lev in range(levels):
            fixed_mask[ids[lev][fixed_row]] = True
        if not np.array_equal(fixed_mask[:n], fixed_attribute > 0.5):
            # trust the frozen attribute on the exported surface
            fixed_mask[:n] = fixed_attribute > 0.5

        # --- cavity (pressure-loaded) triangles ------------------------------
        cav = faces[np.min(faces, axis=1) >= half]
        if len(cav) == 0:
            raise ValueError("no cavity surface triangles found")
        v0f = pos[faces[:, 0]]
        v1f = pos[faces[:, 1]]
        v2f = pos[faces[:, 2]]
        signed_volume = float(np.einsum("ij,ij->i", v0f, np.cross(v1f, v2f)).sum() / 6.0)
        if signed_volume < 0.0:
            cav = cav[:, ::-1]  # make normals point out of the solid

        # --- explicit time step with quasi-static mass scaling ---------------
        c_wave = np.sqrt((self.lam + 2.0 * self.mu) / rho)
        dt_cfl = CFL_SAFETY * h_min / c_wave
        if dt_cfl >= DT_TARGET:
            mass_scale = 1.0
            dt = DT_TARGET
        else:
            mass_scale = min((DT_TARGET / dt_cfl) ** 2, MASS_SCALE_CAP)
            dt = dt_cfl * float(np.sqrt(mass_scale))

        # --- bookkeeping -----------------------------------------------------
        self.n_surface = n
        self.n_nodes = n_nodes
        self.tet_nodes = tn
        self.Dm_inv = Dm_inv
        self.Dm_invT = Dm_invT
        self.V0 = V0
        self.faces = faces
        self.cav_tris = cav
        # reusable force-evaluation buffers and scatter index maps
        ne = tn.shape[0]
        self._buf_Ds = np.empty((ne, 3, 3), dtype=np.float64)
        self._buf_Dsd = np.empty((ne, 3, 3), dtype=np.float64)
        self._buf_F = np.empty((ne, 3, 3), dtype=np.float64)
        self._buf_P = np.empty((ne, 3, 3), dtype=np.float64)
        self._buf_H = np.empty((ne, 3, 3), dtype=np.float64)
        self._buf_w = np.empty((ne, 4, 3), dtype=np.float64)
        self._tet_idx = (
            tn.reshape(-1)[:, None] * 3 + np.arange(3, dtype=np.int64)
        ).reshape(-1)
        self._cav_idx = (
            cav.reshape(-1)[:, None] * 3 + np.arange(3, dtype=np.int64)
        ).reshape(-1)
        self.fixed_attribute = fixed_attribute
        self.fixed_mask = fixed_mask
        self.free_mask = ~fixed_mask
        self.m_node = m_node
        self.m_sim = m_node * mass_scale
        self.mass_export = m_export
        self.total_mass = float(m_node.sum())
        self.g_nodal = m_node[:, None] * np.array([0.0, 0.0, GRAVITY_Z])
        self.dt = float(dt)
        self.mass_scale = float(mass_scale)
        self.h_min = h_min
        r_inner_mean = float(np.linalg.norm(pos[half:, :2], axis=1).mean())
        self.reference_force = float(self.p_max * np.pi * r_inner_mean**2)
        self.x = X.copy()
        self.v = np.zeros((n_nodes, 3), dtype=np.float64)
        self.time = 0.0
        self.step_count = 0
        self._initialized = True

    # ------------------------------------------------------------- pressure
    def _pressure(self, t: float) -> float:
        if t <= self.ramp_start:
            return 0.0
        if t >= self.ramp_end:
            return self.p_max
        return self.p_max * (t - self.ramp_start) / (self.ramp_end - self.ramp_start)

    # --------------------------------------------------------------- forces
    def _forces(self, x: np.ndarray, v: np.ndarray, pressure: float):
        """Nodal forces: Neo-Hookean elastic + Kelvin-Voigt + pressure + gravity."""
        tn = self.tet_nodes
        Ds = self._buf_Ds
        x0 = x[tn[:, 0]]
        Ds[:, :, 0] = x[tn[:, 1]] - x0
        Ds[:, :, 1] = x[tn[:, 2]] - x0
        Ds[:, :, 2] = x[tn[:, 3]] - x0
        F = self._buf_F
        np.matmul(Ds, self.Dm_inv, out=F)
        F_inv, J = _inv_det(F)
        F_invT = np.transpose(F_inv, (0, 2, 1))
        lnJ = np.log(np.clip(J, 1e-6, None))
        P = self._buf_P
        np.subtract(F, F_invT, out=P)
        P *= self.mu
        P += F_invT * (self.lam * lnJ)[:, None, None]

        v0 = v[tn[:, 0]]
        Dsd = self._buf_Dsd
        Dsd[:, :, 0] = v[tn[:, 1]] - v0
        Dsd[:, :, 1] = v[tn[:, 2]] - v0
        Dsd[:, :, 2] = v[tn[:, 3]] - v0
        P += self.xi * np.matmul(Dsd, self.Dm_inv)  # internal viscous stress

        H = self._buf_H
        np.matmul(P, self.Dm_invT, out=H)
        H *= -self.V0[:, None, None]
        w = self._buf_w
        w[:, 1] = H[:, :, 0]
        w[:, 2] = H[:, :, 1]
        w[:, 3] = H[:, :, 2]
        w[:, 0] = -np.sum(H, axis=2)
        f_int = np.bincount(
            self._tet_idx, weights=w.reshape(-1), minlength=3 * self.n_nodes
        ).reshape(self.n_nodes, 3)

        # Follower pressure traction on the cavity surface.  Triangle normals
        # point out of the solid, so the gas pushes with -p * area * normal.
        ct = self.cav_tris
        ca = x[ct[:, 0]]
        cb = x[ct[:, 1]]
        cc = x[ct[:, 2]]
        nrm = 0.5 * np.cross(cb - ca, cc - ca)
        contrib = (-pressure / 3.0) * nrm
        f_p = np.bincount(
            self._cav_idx,
            weights=np.repeat(contrib, 3, axis=0).reshape(-1),
            minlength=3 * self.n_nodes,
        ).reshape(self.n_nodes, 3)

        f_int += f_p
        f_int += self.g_nodal
        return f_int, f_p, F

    def _min_jacobian(self, x: np.ndarray) -> float:
        tn = self.tet_nodes
        x0 = x[tn[:, 0]]
        Ds = np.stack((x[tn[:, 1]] - x0, x[tn[:, 2]] - x0, x[tn[:, 3]] - x0), axis=2)
        return float(_det3(Ds @ self.Dm_inv).min())

    # ------------------------------------------------------------- stepping
    def _step(self, dt: float) -> None:
        for _ in range(90):
            pressure = self._pressure(self.time)
            f_total, _, _ = self._forces(self.x, self.v, pressure)
            v_new = self.v + dt * (f_total / self.m_sim[:, None])
            v_new[self.fixed_mask] = 0.0
            x_new = self.x + dt * v_new
            x_new[self.fixed_mask] = self.x[self.fixed_mask]
            ok = bool(np.isfinite(x_new).all() and np.isfinite(v_new).all())
            if ok:
                if self.free_mask.any():
                    vmax = float(np.linalg.norm(v_new[self.free_mask], axis=1).max())
                else:
                    vmax = 0.0
                ok = (self._min_jacobian(x_new) > J_MIN_ACCEPT) and (vmax < V_MAX_ACCEPT)
            if ok:
                self.x = x_new
                self.v = v_new
                self.time += dt
                self.step_count += 1
                return
            dt *= 0.5
            self.dt = dt
            if dt < 1e-9:
                break
        raise RuntimeError("explicit integration became unstable")

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("solver is not initialized")
        target = float(time)
        if target < self.time - 1e-9:
            raise ValueError("sample times must be non-decreasing")
        while target - self.time > 1e-9:
            step = min(self.dt, target - self.time)
            self._step(float(step))
        self.time = target
        return self._frame(target)

    # ---------------------------------------------------------------- export
    def _frame(self, t: float) -> StateFrame:
        pressure = self._pressure(t)
        f_total, f_p, F = self._forces(self.x, self.v, pressure)
        # External loads on the chamber body: pressure resultant, total weight,
        # and the clamp reactions that hold the base/rim attachment.
        reactions = -f_total[self.fixed_mask].sum(axis=0)
        gravity_total = np.array([0.0, 0.0, GRAVITY_Z * self.total_mass])
        f_ext = f_p.sum(axis=0) + gravity_total + reactions
        budget = np.stack((f_ext, f_ext)).reshape(2, 1, 3)
        n = self.n_surface
        frame = StateFrame(
            time=t,
            position=self.x[:n].copy(),
            velocity=self.v[:n].copy(),
            mass=self.mass_export.copy(),
            body_id=np.zeros(n, dtype=np.int32),
            attributes={
                "fixed": self.fixed_attribute.copy(),
                "mass_quantity": self.mass_export.copy(),
            },
            fields={
                "surface_faces": self.faces,
                "element_deformation_gradient": np.array(F, dtype=np.float64),
                "budget_external_force": budget,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            },
            scalars={
                "internal_pressure_pa": float(pressure),
                "reference_force": float(self.reference_force),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit tetrahedral FEM (Neo-Hookean, follower pressure)",
            "time_step_s": float(self.dt) if self._initialized else None,
            "mass_scale": float(self.mass_scale) if self._initialized else None,
            "min_element_size_m": float(self.h_min) if self._initialized else None,
            "steps": int(self.step_count),
            "tetrahedra": int(self.tet_nodes.shape[0]) if self._initialized else None,
            "current_time_s": float(self.time),
            "max_surface_speed_m_s": (
                float(np.linalg.norm(self.v[: self.n_surface], axis=1).max())
                if self._initialized
                else None
            ),
        }
