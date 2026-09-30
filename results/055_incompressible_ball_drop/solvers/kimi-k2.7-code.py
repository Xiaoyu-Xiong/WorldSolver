"""Explicit soft-body solver for the incompressible ball drop scene.

The ball is modelled as a closed surface of mass points connected by edge
springs, with a per-tetrahedral-element volume constraint that penalises
deviation from the reference volume.  A horizontal floor supplies a penalty
contact force.  Integration is symplectic Euler with a small fixed substep.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = seed

        # ----- physical constants / parameters -----
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.floor_z = 0.0

        # Structural springs.
        self.k_edge = 1.5e4          # [N/m]
        self.c_edge = 0.6            # edge damping coefficient

        # Volume-preserving pressure / damping.
        self.k_volume = 4.0e5        # [N / m^4] (works on volume gradients)
        self.c_volume = 2.0e2        # volume-rate damping

        # Floor contact penalty.
        self.k_floor = 8.0e4         # [N/m]
        self.c_floor = 4.0e2         # contact velocity damping

        # ----- initial state -----
        init: StateFrame = scene.initial_state()
        self.pos = np.asarray(init.position, dtype=float).copy()
        self.vel = np.asarray(init.velocity, dtype=float).copy()
        self.mass = np.asarray(init.mass, dtype=float).copy()
        self.time = float(init.time)

        # ----- reference geometry -----
        self.X_ref = self.pos.copy()
        self.n = self.pos.shape[0]
        self.c_ref = self.X_ref.mean(axis=0)

        # Faces of a closed icosahedron in the same 1-indexed order as the OBJ.
        self.faces = np.array([
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ], dtype=int)

        # Ensure each face has positive signed volume with the centroid.
        self.ref_vol = np.zeros(len(self.faces), dtype=float)
        for e, (i0, i1, i2) in enumerate(self.faces):
            p0, p1, p2 = self._tet_points(self.X_ref, i0, i1, i2)
            v = _signed_tet_volume(p0, p1, p2)
            if v < 0.0:
                self.faces[e, [1, 2]] = self.faces[e, [2, 1]]
                p1, p2 = p2, p1
                v = _signed_tet_volume(p0, p1, p2)
            self.ref_vol[e] = v

        # Unique edges and rest lengths.
        edges = set()
        for i0, i1, i2 in self.faces:
            edges.add((min(i0, i1), max(i0, i1)))
            edges.add((min(i1, i2), max(i1, i2)))
            edges.add((min(i2, i0), max(i2, i0)))
        self.edges = list(edges)
        self.edge_rest = np.array(
            [np.linalg.norm(self.X_ref[i] - self.X_ref[j]) for i, j in self.edges],
            dtype=float,
        )

        # Precompute reference local bases for per-face deformation gradients.
        self.ref_inv_Dm = []
        for e, (i0, i1, i2) in enumerate(self.faces):
            p0, p1, p2 = self._tet_points(self.X_ref, i0, i1, i2)
            Dm = _basis_matrix(p0, p1, p2)
            self.ref_inv_Dm.append(np.linalg.inv(Dm))

        # Reference total volume.
        self.V0_total = float(self.ref_vol.sum())

        # ----- budget accounting -----
        self.G = 1
        self.K = 2
        self.cum_ext_impulse = np.zeros((self.G, 3), dtype=float)
        self.cum_inflow = np.zeros((self.G, 3), dtype=float)
        self.cum_outflow = np.zeros((self.G, 3), dtype=float)
        self.domain_weights = np.ones((self.G, self.n), dtype=float)

        # ----- integration -----
        # sample_dt = 0.02; use 100 substeps for explicit stability.
        self.substeps_per_sample = 100
        self.sample_dt = 0.02
        self.dt = self.sample_dt / self.substeps_per_sample

        self._frame_cache: dict[float, StateFrame] = {}

    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if time in self._frame_cache:
            return self._frame_cache[time]

        if time < self.time - 1e-12:
            raise ValueError(f"Cannot advance backwards: {time} < {self.time}")

        steps = int(round((time - self.time) / self.dt))
        for _ in range(steps):
            self._step(self.dt)
        # Residual for rounding.
        residual = time - self.time
        if residual > 1e-14:
            self._step(residual)

        frame = self._build_frame(time)
        self._frame_cache[time] = frame
        return frame

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        return {}

    # ------------------------------------------------------------------
    def _tet_points(self, X, i0, i1, i2):
        c = X.mean(axis=0)
        return X[i0] - c, X[i1] - c, X[i2] - c

    # ------------------------------------------------------------------
    def _step(self, dt: float) -> None:
        f = np.zeros_like(self.pos)

        # Gravity.
        f += self.mass[:, None] * self.gravity

        # Edge springs / dampers.
        for (i, j), L0 in zip(self.edges, self.edge_rest):
            r = self.pos[i] - self.pos[j]
            dist = np.linalg.norm(r)
            if dist < 1e-12:
                continue
            e = r / dist
            stretch = dist - L0
            rel_v = self.vel[i] - self.vel[j]
            fmag = self.k_edge * stretch + self.c_edge * np.dot(rel_v, e)
            f[i] -= fmag * e
            f[j] += fmag * e

        # Volume-preserving pressure (centroid tetrahedralisation).
        center = self.pos.mean(axis=0)
        for e, (i0, i1, i2) in enumerate(self.faces):
            p0, p1, p2 = self.pos[i0] - center, self.pos[i1] - center, self.pos[i2] - center
            v = _signed_tet_volume(p0, p1, p2)
            delta = v - self.ref_vol[e]
            g0, g1, g2 = _tet_volume_gradients(p0, p1, p2)
            verts = [i0, i1, i2]
            f_press = -self.k_volume * delta * np.stack([g0, g1, g2])
            vels = self.vel[verts]
            rate = g0 @ vels[0] + g1 @ vels[1] + g2 @ vels[2]
            f_damp = -self.c_volume * rate * np.stack([g0, g1, g2])
            f[verts] += f_press + f_damp

        # Floor contact.
        contact_f = self._contact_force(self.pos, self.vel)
        f += contact_f

        # Symplectic Euler update.
        self.vel += (f / self.mass[:, None]) * dt
        self.pos += self.vel * dt
        self.time += dt

        # External-force budget bookkeeping.
        gravity_total = (self.mass[:, None] * self.gravity).sum(axis=0)
        contact_total = contact_f.sum(axis=0)
        self.cum_ext_impulse[0] += (gravity_total + contact_total) * dt

    # ------------------------------------------------------------------
    def _build_frame(self, time: float) -> StateFrame:
        # Per-face deformation gradient.
        center = self.pos.mean(axis=0)
        F_arr = np.zeros((len(self.faces), 3, 3), dtype=float)
        cur_vol = np.zeros(len(self.faces), dtype=float)
        for e, (i0, i1, i2) in enumerate(self.faces):
            p0, p1, p2 = self.pos[i0] - center, self.pos[i1] - center, self.pos[i2] - center
            Ds = _basis_matrix(p0, p1, p2)
            F_arr[e] = Ds @ self.ref_inv_Dm[e]
            cur_vol[e] = _signed_tet_volume(p0, p1, p2)

        # External force totals for the current state.
        gravity_total = (self.mass[:, None] * self.gravity).sum(axis=0)
        contact_total = self._contact_force(self.pos, self.vel).sum(axis=0)
        budget_external_force = np.zeros((self.G, self.K, 3), dtype=float)
        budget_external_force[0, 0] = gravity_total
        budget_external_force[0, 1] = contact_total

        # Constraint: per-element volume must equal the reference volume.
        constraint_value = cur_vol
        constraint_target = self.ref_vol.copy()

        fields = {
            "budget_cumulative_external_impulse": self.cum_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cum_inflow.copy(),
            "budget_cumulative_outflow_momentum": self.cum_outflow.copy(),
            "budget_domain_weights": self.domain_weights.copy(),
            "budget_external_force": budget_external_force,
            "constraint_target": constraint_target,
            "constraint_value": constraint_value,
            "cumulative_mass_source": np.zeros(self.K, dtype=float),
            "element_deformation_gradient": F_arr,
        }

        scalars = {
            "constraint_scale": float(self.V0_total),
            "reference_momentum": float(np.sum(self.mass) * np.sqrt(9.81 * 1.0)),
        }

        return StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            scalars=scalars,
            fields=fields,
        )

    # ------------------------------------------------------------------
    def _contact_force(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        contact_f = np.zeros_like(pos)
        for i in range(self.n):
            z = pos[i, 2]
            if z < self.floor_z:
                d = self.floor_z - z
                vn = vel[i, 2]
                fn = self.k_floor * d - self.c_floor * vn
                if fn > 0.0:
                    contact_f[i, 2] = fn
        return contact_f


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _signed_tet_volume(p0, p1, p2):
    return float(np.dot(p0, np.cross(p1, p2)) / 6.0)


def _tet_volume_gradients(p0, p1, p2):
    g0 = np.cross(p1, p2) / 6.0
    g1 = np.cross(p2, p0) / 6.0
    g2 = np.cross(p0, p1) / 6.0
    return g0, g1, g2


def _basis_matrix(a, b, c):
    """3x3 matrix whose columns are edge a and b plus their unnormalised normal."""
    n = np.cross(a, b)
    return np.stack([a, b, n], axis=1)
