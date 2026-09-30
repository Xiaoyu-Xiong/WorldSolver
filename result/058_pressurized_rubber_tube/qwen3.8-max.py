"""Physically-based solver for inflation of an end-constrained rubber tube.

A three-dimensional linear-elastic finite element model of the finite-thickness
hollow wall is built from tetrahedra (each structured hexahedral wall cell is
split into six tetrahedra).  The equations of motion

    M a = f_pressure(t) + f_internal(u) + f_damping(v) + f_reaction

are advanced with an explicit symplectic (central-difference style) scheme using
a lumped mass matrix and a constant small-strain stiffness operator K.  Both end
rims are Dirichlet-fixed (zero displacement / velocity).  The prescribed cavity
pressure is applied as an outward radial traction on the inner (cavity) surface
nodes, following ``scene.pressure_at(t)``.

All required raw trajectory fields are exported on every StateFrame.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        g = cfg["geometry"]
        self.n_theta = int(g["circumferential_segments"])
        self.n_z = int(g["axial_segments"])
        self.inner_radius = float(g["inner_radius"])
        self.wall_thickness = float(g["wall_thickness"])
        self.height = float(g["height"])
        self.density = float(cfg["material"]["density"])
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=np.float64)
        p_cfg = cfg["boundary_conditions"]["cavity_pressure"]
        self.pressure_pa = float(p_cfg["pressure_pa"])

        # ---- material model (soft silicone rubber, slightly compressible to
        #      avoid volumetric locking of linear tetrahedra)
        E = 5.0e5
        nu = 0.40
        self.E, self.nu = E, nu
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        D = np.zeros((6, 6))
        D[:3, :3] = lam
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
        D[3, 3] = D[4, 4] = D[5, 5] = mu
        self.D = D

        # ---- reference state from the scene contract
        frame0 = scene.initial_state()
        X = np.asarray(frame0.position, dtype=np.float64).copy()
        self.X = X
        self.N = X.shape[0]
        self.fixed = np.asarray(frame0.attributes["fixed_rim"]).astype(bool)
        self.inner_mask = np.asarray(frame0.attributes["pressure_surface"]).astype(bool)
        self.base_attributes = dict(frame0.attributes)

        # ---- build tetrahedral mesh from the structured (theta, z, layer) grid
        nt, nz = self.n_theta, self.n_z
        n_layer = (nz + 1) * nt
        self.n_layer = n_layer

        def out_node(iz, it):
            return iz * nt + (it % nt)

        def in_node(iz, it):
            return n_layer + iz * nt + (it % nt)

        tets = []
        for iz in range(nz):
            for it in range(nt):
                v0 = out_node(iz, it)
                v1 = out_node(iz, it + 1)
                v2 = out_node(iz + 1, it + 1)
                v3 = out_node(iz + 1, it)
                v4 = in_node(iz, it)
                v5 = in_node(iz, it + 1)
                v6 = in_node(iz + 1, it + 1)
                v7 = in_node(iz + 1, it)
                # six-tetrahedra split about body diagonal (0,6)
                cycle = [v1, v2, v3, v7, v4, v5]
                for a, b in zip(cycle, cycle[1:] + [cycle[0]]):
                    tets.append((v0, v6, a, b))
        self.tets = np.asarray(tets, dtype=np.int64)
        self.Ne = self.tets.shape[0]

        # ---- per-tet geometry: shape-function gradients and volumes
        X4 = X[self.tets]                                   # (Ne,4,3)
        ones = np.ones((self.Ne, 4, 1))
        P = np.concatenate((ones, X4), axis=2)              # (Ne,4,4)
        C = np.linalg.inv(P)                               # (Ne,4,4)
        self.dN = np.ascontiguousarray(C[:, :, 1:])        # (Ne,4,3)
        det = np.linalg.det(P)
        self.Vt = np.abs(det) / 6.0                        # (Ne,)

        # ---- strain-displacement operators and element stiffness
        B = np.zeros((self.Ne, 6, 12))
        for k in range(4):
            dNx = self.dN[:, k, 0]
            dNy = self.dN[:, k, 1]
            dNz = self.dN[:, k, 2]
            c = 3 * k
            B[:, 0, c + 0] = dNx
            B[:, 1, c + 1] = dNy
            B[:, 2, c + 2] = dNz
            B[:, 3, c + 1] = dNz
            B[:, 3, c + 2] = dNy
            B[:, 4, c + 0] = dNz
            B[:, 4, c + 2] = dNx
            B[:, 5, c + 0] = dNy
            B[:, 5, c + 1] = dNx
        self.B = B
        Ke = self.Vt[:, None, None] * np.einsum("nai,ab,nbj->nij", B, self.D, B)

        gdof = (self.tets[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(self.Ne, 12)
        rows = np.broadcast_to(gdof[:, :, None], (self.Ne, 12, 12)).ravel()
        cols = np.broadcast_to(gdof[:, None, :], (self.Ne, 12, 12)).ravel()
        vals = Ke.ravel()
        n_dof = 3 * self.N
        self.K = sp.coo_matrix((vals, (rows, cols)), shape=(n_dof, n_dof)).tocsr()

        # ---- lumped mass
        m = np.zeros(self.N)
        contrib = self.density * self.Vt / 4.0              # (Ne,)
        for k in range(4):
            np.add.at(m, self.tets[:, k], contrib)
        m[m <= 0.0] = 1.0e-9
        self.mass = m
        self.Minv = 1.0 / m

        # ---- inner (cavity) surface nodal areas and radial directions
        inner_idx = np.where(self.inner_mask)[0]
        self.inner_idx = inner_idx
        area = np.zeros(self.N)
        ri = self.inner_radius
        for iz in range(nz):
            for it in range(nt):
                a00 = in_node(iz, it)
                a10 = in_node(iz, it + 1)
                a11 = in_node(iz + 1, it + 1)
                a01 = in_node(iz + 1, it)
                p00, p10, p11, p01 = X[a00], X[a10], X[a11], X[a01]
                qa = 0.5 * np.linalg.norm(np.cross(p10 - p00, p01 - p00))
                qb = 0.5 * np.linalg.norm(np.cross(p11 - p10, p01 - p10))
                cell = qa + qb
                for nd in (a00, a10, a11, a01):
                    area[nd] += cell / 4.0
        self.inner_area = area[inner_idx]                   # (Ni,)

        # ---- time-stepping parameters
        c_wave = np.sqrt(E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu) * self.density))
        h_min = float(np.cbrt(np.min(self.Vt)))
        self.dt_stable = 0.4 * h_min / c_wave
        self.dt = min(6.0e-4, self.dt_stable)
        self.alpha = 6.0                                    # mass-proportional damping

        # ---- frozen positive reference force scale
        self.reference_force = float(self.pressure_pa * self.inner_radius * self.height)

        # ---- state
        self.t = 0.0
        self.u = np.zeros((self.N, 3))
        self.v = np.zeros((self.N, 3))
        self._u0 = np.zeros((self.N, 3))

    # ------------------------------------------------------------- dynamics
    def _pressure_force(self, positions):
        f = np.zeros((self.N, 3))
        if self.inner_idx.size == 0:
            return f
        p = self.scene.pressure_at(self.t_mid)
        xy = positions[self.inner_idx, :2]
        r = np.linalg.norm(xy, axis=1)
        r = np.where(r < 1e-12, 1e-12, r)
        dirs = np.zeros_like(xy)
        dirs[:, 0] = xy[:, 0] / r
        dirs[:, 1] = xy[:, 1] / r
        f[self.inner_idx] = p * self.inner_area[:, None] * dirs
        return f

    def _accel(self, positions):
        u_flat = self.u.reshape(-1)
        f_int = -(self.K @ u_flat).reshape(self.N, 3)
        f_ext = self._pressure_force(positions)
        f_grav = self.mass[:, None] * self.gravity[None, :]
        f_damp = -self.alpha * self.mass[:, None] * self.v
        a = (f_ext + f_int + f_grav + f_damp) * self.Minv[:, None]
        a[self.fixed] = 0.0
        return a, f_int, f_ext + f_grav

    def _step(self, dt):
        pos = self.X + self.u
        self.t_mid = self.t + 0.5 * dt
        a, _, _ = self._accel(pos)
        self.v += dt * a
        self.v[self.fixed] = 0.0
        self.u += dt * self.v
        self.u[self.fixed] = 0.0
        self.t += dt

    # ------------------------------------------------------------- sampling
    def advance_to(self, time: float):
        target = float(time)
        if target > self.t:
            span = target - self.t
            n_sub = max(1, int(np.ceil(span / self.dt)))
            dt = span / n_sub
            for _ in range(n_sub):
                self._step(dt)
        self.t = target
        return self._make_frame()

    def _make_frame(self):
        u = self.u
        pos = self.X + u
        vel = self.v.copy()

        # deformation gradient per tetrahedral element
        u_tet = u[self.tets]                                # (Ne,4,3)
        G = np.einsum("nki,nkj->nij", u_tet, self.dN)       # (Ne,3,3)
        F = np.zeros((self.Ne, 3, 3))
        F[:] = np.eye(3)
        F += G

        # external-force budget for the combined tube domain (G=1, K=3 terms)
        pos_c = pos
        u_flat = u.reshape(-1)
        f_int = -(self.K @ u_flat).reshape(self.N, 3)
        self.t_mid = self.t
        f_press = self._pressure_force(pos_c)
        f_grav = self.mass[:, None] * self.gravity[None, :]
        grav_term = f_grav.sum(axis=0)
        press_term = f_press.sum(axis=0)
        f_ext_all = f_press + f_grav
        reaction = -(f_int + f_ext_all)[self.fixed].sum(axis=0)
        budget = np.zeros((1, 3, 3))
        budget[0, 0] = grav_term
        budget[0, 1] = press_term
        budget[0, 2] = reaction

        attrs = dict(self.base_attributes)
        attrs["mass_quantity"] = self.mass.copy()

        fields = {
            "budget_external_force": budget,
            "cumulative_mass_source": np.zeros(1),
            "element_deformation_gradient": F,
        }
        scalars = {
            "reference_force": self.reference_force,
            "cavity_pressure_pa": float(self.scene.pressure_at(self.t)),
        }
        frame = StateFrame(
            time=self.t,
            position=pos,
            velocity=vel,
            mass=self.mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int64),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ---------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        if getattr(self, "u", None) is None:
            return {}
        pos = self.X + self.u
        free = ~self.fixed
        mid_r = float(np.max(np.linalg.norm(pos[free][:, :2], axis=1))) if np.any(free) else 0.0
        return {
            "time": self.t,
            "n_nodes": int(self.N),
            "n_elements": int(self.Ne),
            "max_radial_extent": mid_r,
            "max_displacement": float(np.max(np.linalg.norm(self.u, axis=1))),
        }
