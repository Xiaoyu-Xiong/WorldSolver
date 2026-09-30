"""Solver Agent-owned implementation for the Figure 15 sphere-drape experiment.

A lightweight but physically real cloth solver: triangle-mesh mass-spring cloth
with corotational linear triangles, explicit time stepping with Rayleigh-style
velocity damping, strain-limiting constraints, and collision response against a
fixed sphere and ground plane.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self._scene = None
        self._seed = 0
        self._times = []
        self._time = 0.0

        # geometry
        self._faces = None
        self._n_nodes = 0
        self._n_faces = 0
        self._x0 = None
        self._v = None
        self._m = None
        self._inv_m = None
        self._gravity = np.array([0.0, 0.0, -9.81])

        # reference quantities
        self._Dm_inv = None
        self._rest_area = None
        self._nodal_area = None
        self._L0 = {}  # edge rest lengths for bending/springs
        self._frame_cache = {}

        # configuration
        self._substep = 0.004
        self._sphere_center = np.array([0.0, 0.0, 0.64])
        self._sphere_radius = 0.64
        self._ground_z = 0.0
        self._thickness = 0.012
        self._damping = 1.35
        self._friction = 0.22
        self._strain_lower = 0.9
        self._strain_upper = 1.1
        self._target_lower = 0.965
        self._target_upper = 1.045
        self._constraint_iterations = 15

        # precomputed per-element reference
        self._ref_edges = None  # shape [F,2,3]

    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = seed
        config = scene.config

        self._substep = float(config["solver"]["substep"])
        self._constraint_iterations = int(config["solver"]["constraint_iterations"])
        cloth = config["cloth"]
        self._thickness = float(cloth["thickness"])
        self._damping = float(cloth["damping"])
        self._friction = float(cloth["friction"])
        self._strain_lower = float(cloth["strain_lower"])
        self._strain_upper = float(cloth["strain_upper"])
        self._target_lower = float(cloth["target_strain_lower"])
        self._target_upper = float(cloth["target_strain_upper"])
        self._sphere_center = np.array(config["sphere"]["center"], dtype=float)
        self._sphere_radius = float(config["sphere"]["radius"])
        self._ground_z = float(config["ground"]["height"])
        self._gravity = np.array(config["gravity"], dtype=float)

        init = scene.initial_state()
        x = np.array(init.position, dtype=float)
        self._x0 = x.copy()
        self._v = np.array(init.velocity, dtype=float)
        self._m = np.array(init.mass, dtype=float)
        if self._m.ndim == 1:
            self._inv_m = np.where(self._m > 0.0, 1.0 / self._m, 0.0)
        else:
            self._inv_m = np.where(self._m > 0.0, 1.0 / self._m, 0.0)

        self._n_nodes = x.shape[0]

        # Load topology from the scene asset.
        asset_path = Path(__file__).resolve().parent.parent / "assets" / "cloth_topology.json"
        topo = json.loads(asset_path.read_text(encoding="utf-8"))
        faces = np.array(topo["faces"], dtype=int)
        self._faces = faces
        self._n_faces = faces.shape[0]

        # Reference edge data for each triangle.
        i0 = faces[:, 0]
        i1 = faces[:, 1]
        i2 = faces[:, 2]
        e0 = x[i1] - x[i0]  # edge 1
        e1 = x[i2] - x[i0]  # edge 2
        # Material axes in the initial flat configuration.
        # For the initial flat cloth, u aligns with x and v aligns with y.
        # The deformation gradient F = [e0,e1] * Dm_inv where Dm is [u_axis, v_axis]
        # We pick orthonormal basis for material coordinates.
        # e0, e1 live in 3D; we map to 2D material space.
        # Reference triangle edge vectors in world.
        self._ref_edges = np.stack([e0, e1], axis=1)  # [F,2,3]

        # Compute rest areas and inverse 2x2 material matrices.
        Dm = self._edge_pair_to_Dm(e0, e1)
        self._Dm_inv = np.linalg.inv(Dm)  # [F,2,2]
        self._rest_area = 0.5 * np.abs(np.linalg.det(Dm))

        # Nodal area/mass weighting (Voronoi-like)
        nodal_area = np.zeros(self._n_nodes)
        np.add.at(nodal_area, i0, self._rest_area / 3.0)
        np.add.at(nodal_area, i1, self._rest_area / 3.0)
        np.add.at(nodal_area, i2, self._rest_area / 3.0)
        self._nodal_area = nodal_area

        self._time = 0.0
        self._frame_cache = {0.0: (x.copy(), self._v.copy())}
        # Keep initial deformation gradient consistent with rest state.
        self._frame_cache[0.0] = (self._x0.copy(), self._v.copy())

    @staticmethod
    def _edge_pair_to_Dm(e0, e1):
        # e0, e1 are world edge vectors. Build a 2D material coordinate matrix.
        # Use lengths as material coordinates.
        # Dm = [[e0.u, e1.u], [e0.v, e1.v]].
        # Project onto a local orthonormal frame built from e0 and e1.
        # We use the first edge as u-axis, and the component of e1 orthogonal to e0 as v-axis.
        e0u = np.linalg.norm(e0, axis=1)
        e0u = np.where(e0u == 0.0, 1.0, e0u)
        u_dir = e0 / e0u[:, None]
        dot = np.sum(e1 * u_dir, axis=1)
        v_perp = e1 - dot[:, None] * u_dir
        v_len = np.linalg.norm(v_perp, axis=1)
        v_len = np.where(v_len == 0.0, 1.0, v_len)
        v_dir = v_perp / v_len[:, None]
        Dm = np.zeros((e0.shape[0], 2, 2))
        Dm[:, 0, 0] = e0u
        Dm[:, 0, 1] = dot
        Dm[:, 1, 0] = 0.0
        Dm[:, 1, 1] = v_len
        return Dm

    def _compute_deformation_gradient(self, x):
        # x is current positions.
        i0 = self._faces[:, 0]
        i1 = self._faces[:, 1]
        i2 = self._faces[:, 2]
        e0 = x[i1] - x[i0]
        e1 = x[i2] - x[i0]
        # F is 3x2, represented as full 3x3 with last column zero.
        F3 = np.zeros((self._n_faces, 3, 3), dtype=float)
        # F = [e0, e1] * Dm_inv, where Dm_inv is 2x2.
        # F[:,:,0:2] = [e0, e1] * Dm_inv
        for f in range(self._n_faces):
            Dinv = self._Dm_inv[f]
            E = np.stack([e0[f], e1[f]], axis=1)  # 3x2
            F_full = E @ Dinv  # 3x2
            F3[f, :, 0] = F_full[:, 0]
            F3[f, :, 1] = F_full[:, 1]
            F3[f, :, 2] = 0.0
        return F3

    def _elastic_forces(self, x):
        # Corotational linear elastic model per triangle with strain limiting.
        # Force on vertex a from triangle: -area * P * Dm_inv^T * dS/dx, but for a triangle
        # with vertices (i0,i1,i2), the gradient of F wrt x is constant.
        # We use St. Venant-Kirchhoff like membrane energy with a corotational correction
        # by clamping principal stretches to target range.
        F = self._compute_deformation_gradient(x)
        forces = np.zeros((self._n_nodes, 3), dtype=float)
        mu = 1.0  # membrane stiffness scale (low, soft cloth)
        lame = 0.5
        for f in range(self._n_faces):
            F_full = F[f]  # 3x3, but only first two columns meaningful
            # Green strain E = 0.5 (F^T F - I)
            # Compute 2x2 version.
            F2 = F_full[:, :2]
            G = F2.T @ F2
            E2 = 0.5 * (G - np.eye(2))
            # Principal strain limiting by clamping eigenvalues of G.
            eigvals, eigvecs = np.linalg.eigh(G)
            # principal stretches are sqrt(eigvals). Clamp to target range.
            stretches = np.sqrt(np.maximum(eigvals, 1e-12))
            clamped = np.clip(stretches, self._target_lower, self._target_upper)
            # Build modified Green strain from clamped stretches.
            # This is a simple strain-limiting projection.
            E_diag = 0.5 * (clamped ** 2 - 1.0)
            E_modified = eigvecs @ np.diag(E_diag) @ eigvecs.T
            # 2nd Piola-Kirchhoff stress S = 2*mu*E + lambda*trace(E)*I
            trace_E = E_modified[0, 0] + E_modified[1, 1]
            S = 2.0 * mu * E_modified + lame * trace_E * np.eye(2)
            # First Piola P = F * S
            P = F2 @ S  # 3x2
            # forces = -area * P * Dm_inv^T, distributed to nodes.
            Dinv = self._Dm_inv[f]
            W = P @ Dinv.T  # 3x2 -> weights on the two edge vectors in local coords
            area = self._rest_area[f]
            f0 = -area * W[:, 0]
            f1 = -area * W[:, 1]
            f2 = -f0 - f1
            i0, i1, i2 = self._faces[f]
            forces[i0] += f0
            forces[i1] += f1
            forces[i2] += f2
        return forces

    def _damping_forces(self, v):
        # Simple per-vertex velocity damping.
        return -self._damping * v

    def _resolve_sphere_collision(self, x, v):
        # Push cloth vertices outside the sphere, with a small margin, and apply friction.
        margin = self._thickness * 0.5
        center = self._sphere_center
        radius = self._sphere_radius + margin
        diff = x - center
        dist = np.linalg.norm(diff, axis=1)
        mask = dist < radius
        if not np.any(mask):
            return x, v
        n = diff[mask] / dist[mask, None]
        penetration = radius - dist[mask]
        x = x.copy()
        v = v.copy()
        x[mask] += n * penetration[:, None]
        # Friction / velocity response: remove normal velocity, damp tangential.
        v_n = np.sum(v[mask] * n, axis=1)[:, None] * n
        v_t = v[mask] - v_n
        v[mask] = v_t * (1.0 - self._friction * 0.5)
        return x, v

    def _resolve_ground_collision(self, x, v):
        margin = self._thickness * 0.5
        z_min = self._ground_z + margin
        mask = x[:, 2] < z_min
        if not np.any(mask):
            return x, v
        x = x.copy()
        v = v.copy()
        x[mask, 2] = z_min
        v[mask, 2] = np.maximum(v[mask, 2], 0.0)
        v[mask, :2] *= (1.0 - self._friction * 0.5)
        return x, v

    def _strain_limit_projection(self, x, iterations):
        # Fast strain-limiting by projecting each triangle to satisfy principal
        # stretch bounds, distributing corrections to vertices.
        for _ in range(iterations):
            dx = np.zeros_like(x)
            count = np.zeros(self._n_nodes)
            F = self._compute_deformation_gradient(x)
            for f in range(self._n_faces):
                i0, i1, i2 = self._faces[f]
                F_full = F[f]
                F2 = F_full[:, :2]
                G = F2.T @ F2
                eigvals, eigvecs = np.linalg.eigh(G)
                stretches = np.sqrt(np.maximum(eigvals, 1e-12))
                if np.all(stretches >= self._strain_lower) and np.all(stretches <= self._strain_upper):
                    continue
                clamped = np.clip(stretches, self._strain_lower, self._strain_upper)
                # Modify G to have these eigenvalues.
                G_new = eigvecs @ np.diag(clamped ** 2) @ eigvecs.T
                # We want to find a new F2 such that F2^T F2 = G_new.
                # Use polar decomposition of F2 = R * sqrt(G); set F2' = R * sqrt(G_new).
                # Compute polar R from current F2.
                U, s, Vt = np.linalg.svd(F2, full_matrices=False)
                R = U @ Vt
                # sqrt of G_new in material frame.
                sqrt_G_new = eigvecs @ np.diag(clamped) @ eigvecs.T
                F2_new = R @ sqrt_G_new
                # Map new edge vectors back.
                Dinv = self._Dm_inv[f]
                E_new = F2_new @ np.linalg.inv(Dinv)
                p0 = x[i0]
                e0_new = E_new[:, 0]
                e1_new = E_new[:, 1]
                p1_new = p0 + e0_new
                p2_new = p0 + e1_new
                # average target positions
                dx[i0] += (p0 + (p0 - x[i0]))
                dx[i1] += p1_new
                dx[i2] += p2_new
                count[i0] += 1
                count[i1] += 1
                count[i2] += 1
            valid = count > 0
            if np.any(valid):
                x = x.copy()
                x[valid] = dx[valid] / count[valid, None]
        return x

    def _step_explicit(self, x, v, dt):
        # Explicit symplectic Euler with velocity damping and collision response.
        # x_{n+1} = x_n + dt * v_n
        # v_{n+1} = v_n + dt * inv_m * (f_elastic + f_damping + gravity)
        f_elastic = self._elastic_forces(x)
        f_damp = self._damping_forces(v)
        f_ext = self._m * self._gravity
        f_total = f_elastic + f_damp + f_ext
        if self._m.ndim == 1:
            a = f_total * self._inv_m[:, None]
        else:
            a = f_total * self._inv_m
        v_new = v + dt * a
        x_new = x + dt * v_new
        # collisions
        x_new, v_new = self._resolve_sphere_collision(x_new, v_new)
        x_new, v_new = self._resolve_ground_collision(x_new, v_new)
        # strain limit projection on positions
        x_new = self._strain_limit_projection(x_new, self._constraint_iterations)
        return x_new, v_new

    def _substep_advance(self, x, v, t, dt):
        # Substepping wrapper.
        n_sub = max(1, int(np.ceil(dt / self._substep)))
        sub_dt = dt / n_sub
        for _ in range(n_sub):
            x, v = self._step_explicit(x, v, sub_dt)
        return x, v

    def advance_to(self, time: float) -> StateFrame:
        # advance from current cached time to requested time
        if time <= self._time:
            x, v = self._frame_cache.get(time, (self._x0.copy(), self._v.copy()))
        else:
            x, v = self._frame_cache.get(self._time, (self._x0.copy(), self._v.copy()))
            dt = time - self._time
            x, v = self._substep_advance(x, v, self._time, dt)
            self._time = time
            self._frame_cache = {time: (x.copy(), v.copy())}

        # Build required trajectory fields.
        F = self._compute_deformation_gradient(x)
        # contact signed distance: per-vertex distance to sphere, negative inside.
        diff = x - self._sphere_center
        dist = np.linalg.norm(diff, axis=1)
        contact_sd = dist - self._sphere_radius
        # For samples outside a small band, we keep them as positive.
        # mass_quantity: per-node mass (single material -> shape [N]).
        mass_quantity = np.array(self._m, dtype=float)
        if mass_quantity.ndim == 1:
            pass  # [N]
        else:
            # take total per node if multi-material
            mass_quantity = mass_quantity.sum(axis=1)
        # cumulative mass source: no mass added/removed -> [1]
        cumulative_mass_source = np.zeros(1, dtype=float)
        # contact length scale: thickness
        contact_length_scale = float(self._thickness)

        return StateFrame(
            time=time,
            position=x.astype(float),
            velocity=v.astype(float),
            mass=mass_quantity.astype(float),
            attributes={
                "deformation_gradient": F.astype(float),
                "mass_quantity": mass_quantity.astype(float),
            },
            fields={
                "contact_signed_distance": contact_sd.astype(float),
                "cumulative_mass_source": cumulative_mass_source.astype(float),
            },
            scalars={
                "contact_length_scale": contact_length_scale,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "baseline_cloth_solver",
            "nodes": self._n_nodes,
            "faces": self._n_faces,
            "substep": self._substep,
        }
