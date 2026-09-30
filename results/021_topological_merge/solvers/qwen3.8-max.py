"""Viscoplastic merging solver for scene 021_topological_merge.

Physical model
--------------
The material is discretised as the particle cloud supplied by the scene
(24 independent rounded blocks of cohesive viscoplastic solid).  The solver
is a Lagrangian particle elastoplasticity scheme:

* Neighbour discovery is purely geometric (radius search), so particles that
  belong to different initial blocks start interacting as soon as they touch.
  Every close pair is turned into a *permanent* material bond, which is exactly
  what produces repeated topological merges and persistent material
  connections without any remeshing.
* Each bond is an elasto-viscoplastic axial element: an elastic spring with a
  small yield strain, rate dependent plastic flow of the natural length
  (large viscoplastic deformation / squashing), normal viscous damping and
  Coulomb-limited tangential friction.  Plastic flow of the natural length is
  the dissipative mechanism that lets the blocks spread, flatten and freeze
  into one coherent pile.
* A penalty ground (no-penetration support at z = 0) with dry friction.
* Semi-implicit Euler time integration with a CFL-safe fixed substep.

Reported raw state: positions, velocities, masses, per-particle deformation
gradient (least squares fit of current separations onto the stored material
reference separations), per-sample mass quantity, external (gravity) force,
net internal contact/interaction force, cumulative mass source (zero: mass is
conserved), plus the connected-component group id showing the merge count.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = getattr(scene, "config", {}) or {}
        frame0 = scene.initial_state()

        self.x = np.array(frame0.position, dtype=np.float64, copy=True)
        self.v = np.array(frame0.velocity, dtype=np.float64, copy=True)
        self.m = np.array(frame0.mass, dtype=np.float64, copy=True)
        self.body0 = np.asarray(frame0.body_id).astype(np.int64).copy()
        self.attrs0 = {
            key: np.asarray(val, dtype=np.float64).copy()
            for key, val in frame0.attributes.items()
        }
        self.scalars0 = dict(frame0.scalars)
        self.N = int(self.x.shape[0])

        gravity = config.get("gravity", [0.0, 0.0, -9.81])
        self.gravity = np.asarray(gravity, dtype=np.float64).reshape(3)
        self.duration = float(config.get("duration", 9.0))

        spacing = float(self.scalars0.get("particle_spacing", 0.205))
        self.spacing = spacing
        radius = float(np.mean(self.attrs0.get("particle_radius", np.full(1, 0.5 * spacing))))
        self.particle_radius = radius

        # ---- time stepping (CFL limited by the elastic wave speed)
        self.dt = 1.5e-3
        self.time = 0.0
        self.steps = 0

        # ---- material parameters (soft cohesive clay-like viscoplastic solid)
        density = float(self.scalars0.get("rest_density", 1050.0))
        young = 2.0e5                       # Pa
        self.k_n = young * spacing          # axial bond stiffness [N/m]
        self.m_red = 0.5 * float(np.mean(self.m))
        self.zeta = 0.28                    # normal damping ratio
        self.c_n = 2.0 * self.zeta * np.sqrt(max(self.k_n * self.m_red, 1e-9))
        self.c_t = 0.50 * self.c_n          # tangential (viscous) damping
        self.mu_bond = 0.55                 # internal friction coefficient
        self.mu_ground = 0.60
        self.yield_strain = 0.04            # elastic range
        self.plastic_rate = 6.0             # 1/s viscoplastic flow rate
        self.f_max = 0.35 * self.k_n * spacing   # cohesion / force cap
        self.r_bond = 1.45 * spacing        # bonding (contact) radius
        self.min_ratio = 0.72               # densification limit
        self.max_ratio = 2.2                # ductile stretch limit
        self.v_max = 25.0
        self.drag = 0.02                    # weak air drag, aids settling

        # ground: box top surface at z = 0
        self.k_g = 16.0 * self.k_n
        self.c_g = 2.0 * 0.35 * np.sqrt(max(self.k_g * float(np.mean(self.m)), 1e-9))

        # ---- permanent material bonds
        self.bond_i = np.zeros(0, dtype=np.int64)
        self.bond_j = np.zeros(0, dtype=np.int64)
        self.bond_ref = np.zeros((0, 3), dtype=np.float64)
        self.bond_ref_len = np.zeros(0, dtype=np.float64)
        self.bond_L0 = np.zeros(0, dtype=np.float64)
        self.bond_keys = np.zeros(0, dtype=np.int64)
        self.bond_cap = 30 * max(self.N, 1)

        self.f_int = np.zeros((self.N, 3), dtype=np.float64)
        self.f_ext = self.m[:, None] * self.gravity[None, :]
        self.f_ext_report = self.f_ext.copy()

        # initial contact network (bonds inside each separate block)
        self._create_bonds()

    # ------------------------------------------------------------- bonding
    def _create_bonds(self) -> None:
        if self.N < 2:
            return
        tree = cKDTree(self.x)
        pairs = tree.query_pairs(self.r_bond, output_type="ndarray")
        if pairs.size == 0:
            return
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        lo = np.minimum(pairs[:, 0], pairs[:, 1])
        hi = np.maximum(pairs[:, 0], pairs[:, 1])
        keys = lo * self.N + hi
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        lo = lo[order]
        hi = hi[order]
        # keep only first occurrence of duplicated keys
        first = np.ones(len(keys), dtype=bool)
        first[1:] = keys[1:] != keys[:-1]
        keys, lo, hi = keys[first], lo[first], hi[first]

        if self.bond_keys.size:
            idx = np.searchsorted(self.bond_keys, keys)
            idx_c = np.clip(idx, 0, self.bond_keys.size - 1)
            exists = self.bond_keys[idx_c] == keys
        else:
            exists = np.zeros(len(keys), dtype=bool)
        new = ~exists
        if not np.any(new):
            return
        room = self.bond_cap - self.bond_keys.size
        if room <= 0:
            return
        if new.sum() > room:
            keep = np.flatnonzero(new)[:room]
            new = np.zeros_like(new)
            new[keep] = True
        ni = lo[new]
        nj = hi[new]
        ref = self.x[nj] - self.x[ni]
        ref_len = np.linalg.norm(ref, axis=1)
        good = ref_len > 1e-6
        ni, nj, ref, ref_len = ni[good], nj[good], ref[good], ref_len[good]
        L0 = np.clip(ref_len, self.min_ratio * self.spacing, self.max_ratio * self.spacing)

        self.bond_i = np.concatenate([self.bond_i, ni])
        self.bond_j = np.concatenate([self.bond_j, nj])
        self.bond_ref = np.concatenate([self.bond_ref, ref])
        self.bond_ref_len = np.concatenate([self.bond_ref_len, ref_len])
        self.bond_L0 = np.concatenate([self.bond_L0, L0])
        new_keys = np.sort(keys[new][good])
        self.bond_keys = np.concatenate([self.bond_keys, new_keys])
        if not np.all(np.diff(self.bond_keys) > 0):
            keep_order = np.argsort(self.bond_keys, kind="stable")
            self.bond_keys = self.bond_keys[keep_order]
            self.bond_i = self.bond_i[keep_order]
            self.bond_j = self.bond_j[keep_order]
            self.bond_ref = self.bond_ref[keep_order]
            self.bond_ref_len = self.bond_ref_len[keep_order]
            self.bond_L0 = self.bond_L0[keep_order]

    # ------------------------------------------------------------- physics
    def _step(self, dt: float) -> None:
        x, v, m = self.x, self.v, self.m
        n_part = self.N

        self._create_bonds()

        f_int = np.zeros((n_part, 3), dtype=np.float64)
        bi, bj = self.bond_i, self.bond_j
        if bi.size:
            dx = x[bj] - x[bi]
            dist = np.sqrt(np.einsum("ij,ij->i", dx, dx))
            dist = np.maximum(dist, 1e-9)
            nrm = dx / dist[:, None]
            L0 = self.bond_L0

            # --- rate dependent plastic flow of the natural length
            strain = (dist - L0) / L0
            flow = 1.0 - np.exp(-self.plastic_rate * dt)
            hi = strain > self.yield_strain
            lo = strain < -self.yield_strain
            target = np.where(hi, L0 * (1.0 + self.yield_strain),
                              L0 * (1.0 - self.yield_strain))
            L0 = np.where(hi | lo, L0 + flow * (dist - target), L0)
            L0 = np.clip(L0, self.min_ratio * self.bond_ref_len,
                         self.max_ratio * self.bond_ref_len)
            self.bond_L0 = L0

            # --- elastic axial force + normal viscous damping
            strain = (dist - L0) / L0
            f_n = self.k_n * L0 * strain
            rel_v = v[bj] - v[bi]
            v_n = np.einsum("ij,ij->i", rel_v, nrm)
            f_n = f_n + self.c_n * v_n
            f_n = np.clip(f_n, -3.0 * self.f_max, self.f_max)

            # --- tangential friction, Coulomb limited by compression + cohesion
            v_t = rel_v - v_n[:, None] * nrm
            f_t = self.c_t * v_t
            f_t_mag = np.sqrt(np.einsum("ij,ij->i", f_t, f_t))
            compression = np.maximum(0.0, -f_n)
            limit = self.mu_bond * compression + 0.12 * self.f_max
            scale = np.where(f_t_mag > limit, limit / np.maximum(f_t_mag, 1e-12), 1.0)
            f_t = f_t * scale[:, None]

            f_pair = f_n[:, None] * nrm + f_t
            for axis in range(3):
                f_int[:, axis] += np.bincount(bi, weights=f_pair[:, axis], minlength=n_part)
                f_int[:, axis] -= np.bincount(bj, weights=f_pair[:, axis], minlength=n_part)

        # --- ground support (fixed no-penetration plane at z = 0)
        r_p = self.particle_radius
        pen = r_p - x[:, 2]
        touch = pen > 0.0
        if np.any(touch):
            f_gz = self.k_g * pen - self.c_g * np.minimum(v[:, 2], 0.0)
            f_gz = np.maximum(f_gz, 0.0)
            f_gz = np.minimum(f_gz, 40.0 * self.f_max)
            f_ground = np.zeros((n_part, 3), dtype=np.float64)
            f_ground[:, 2] = np.where(touch, f_gz, 0.0)
            # dry friction against the tangential sliding velocity
            v_tan = v[:, :2].copy()
            speed = np.sqrt(np.einsum("ij,ij->i", v_tan, v_tan))
            active = touch & (speed > 1e-9)
            if np.any(active):
                f_fric = np.minimum(self.c_g * 0.25 * speed,
                                    self.mu_ground * f_gz)
                safe_speed = np.maximum(speed, 1e-9)
                f_ground[:2] -= np.where(active[:, None],
                                         (f_fric / safe_speed)[:, None] * v_tan, 0.0)
            f_int = f_int + f_ground

        self.f_int = f_int
        # reported external force: gravity body force plus weak ambient drag
        f_ext = self.f_ext - (self.drag * m)[:, None] * v
        self.f_ext_report = f_ext

        accel = (f_ext + f_int) / m[:, None]
        v_new = v + accel * dt
        speed = np.sqrt(np.einsum("ij,ij->i", v_new, v_new))
        over = speed > self.v_max
        if np.any(over):
            v_new[over] *= (self.v_max / speed[over])[:, None]
        self.v = v_new
        x_new = x + v_new * dt
        # hard safety clamp: never tunnel through the support
        below = x_new[:, 2] < 0.30 * r_p
        if np.any(below):
            x_new[below, 2] = 0.30 * r_p
            self.v[below, 2] = np.maximum(self.v[below, 2], 0.0)
        self.x = x_new
        self.time += dt
        self.steps += 1

    # ------------------------------------------------------ derived outputs
    def _deformation_gradients(self) -> np.ndarray:
        n = self.N
        eye = np.eye(3)
        F = np.broadcast_to(eye, (n, 3, 3)).copy()
        bi, bj = self.bond_i, self.bond_j
        if bi.size == 0:
            return F
        dx = self.x[bj] - self.x[bi]
        dX = self.bond_ref
        idx = np.concatenate([bi, bj])
        sdx = np.concatenate([dx, dx], axis=0)
        sdX = np.concatenate([dX, dX], axis=0)
        A = np.zeros((n, 9), dtype=np.float64)
        B = np.zeros((n, 9), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                A[:, 3 * a + b] = np.bincount(idx, weights=sdx[:, a] * sdX[:, b], minlength=n)
                B[:, 3 * a + b] = np.bincount(idx, weights=sdX[:, a] * sdX[:, b], minlength=n)
        count = np.bincount(idx, minlength=n).astype(np.float64)
        A = A.reshape(n, 3, 3)
        B = B.reshape(n, 3, 3)
        has = count > 0
        if not np.any(has):
            return F
        eps = 1e-6 * (np.maximum(B[:, 0, 0] + B[:, 1, 1] + B[:, 2, 2], 1e-12) / 3.0)
        B[has] += eps[has][:, None, None] * eye[None, :, :]
        B[~has] = eye[None, :, :]
        Ft = np.linalg.solve(np.swapaxes(B, 1, 2), np.swapaxes(A, 1, 2))
        F_new = np.swapaxes(Ft, 1, 2)
        F_new[~has] = eye[None, :, :]
        bad = ~np.isfinite(F_new).all(axis=(1, 2))
        F_new[bad] = eye[None, :, :]
        return F_new

    def _group_ids(self) -> np.ndarray:
        n = self.N
        if self.bond_i.size == 0:
            return np.arange(n, dtype=np.float64)
        data = np.ones(self.bond_i.size, dtype=np.int8)
        graph = coo_matrix((data, (self.bond_i, self.bond_j)), shape=(n, n)).tocsr()
        _, labels = connected_components(graph, directed=False)
        return labels.astype(np.float64)

    def _frame(self, time: float) -> StateFrame:
        n = self.N
        F = self._deformation_gradients()
        groups = self._group_ids()
        mass = self.m.copy()
        attributes = {key: val.copy() for key, val in self.attrs0.items()}
        attributes["deformation_gradient"] = F
        attributes["mass_quantity"] = mass.copy()
        attributes["group_id"] = groups
        attributes["coordination"] = np.bincount(
            np.concatenate([self.bond_i, self.bond_j]), minlength=n
        ).astype(np.float64) if self.bond_i.size else np.zeros(n)
        fields = {
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "external_force": self.f_ext_report.copy(),
            "interaction_force": self.f_int.copy(),
            "gravity_acceleration": self.gravity[None, :].copy(),
        }
        scalars = dict(self.scalars0)
        scalars["characteristic_length"] = float(self.spacing)
        scalars["group_count"] = float(len(np.unique(groups)))
        scalars["bond_count"] = float(self.bond_i.size)
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=mass,
            body_id=self.body0.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # -------------------------------------------------------------- driving
    def advance_to(self, time: float):
        target = float(time)
        if target <= self.time + 1e-12:
            return self._frame(self.time if target < self.time else target)
        guard = 0
        while self.time < target - 1e-9:
            remaining = target - self.time
            dt = self.dt if remaining > self.dt else remaining
            if dt < 1e-12:
                break
            self._step(dt)
            guard += 1
            if guard > 2_000_000:
                break
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "lagrangian_particle_elastoviscoplasticity",
            "particles": int(self.N),
            "bonds": int(self.bond_i.size),
            "substeps": int(self.steps),
            "substep_dt": float(self.dt),
            "final_time": float(self.time),
            "final_group_count": int(len(np.unique(self._group_ids()))),
        }
