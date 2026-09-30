"""Physically-based heterogeneous gummy-bear solver.

The bonded bear is modelled as a soft outer triangle mesh (mass-spring
continuum) that carries a near-rigid interior.  The interior vertices are
kinematically bound to the deforming outer shell so that the stiff core turns
with the surrounding bear, while the outer shell stores and releases elastic
energy under gravity and ground contact.  Time integration is a damped
semi-implicit (symplectic) Euler scheme with fixed sub-steps.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from worldsolver.contracts import StateFrame


NDArrayf = NDArray[np.float64]


class BenchmarkSolver:
    """Solver for the bonded heterogeneous gummy-bear scene."""

    def initialize(self, scene, seed: int) -> None:
        """Set up rest geometry, topology, masses and time-stepping state."""
        self._rng = np.random.default_rng(seed)
        self._scene = scene

        # Material parameters from the scaffold.
        init = scene.initial_state()
        init.validate()

        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.ground_height = 0.0
        self.ground_stiffness = 2.0e4  # N/m penalty
        self.ground_damping = 0.5      # contact damping scalar
        self.friction_mu = 0.3

        self.soft_young = 6.0e4
        self.soft_poisson = 0.45
        self.density = 1000.0

        # Effective shell thickness used to scale edge stiffness.
        self.thickness = 0.003

        # Global velocity damping (air / numerical).
        self.global_damping = 0.3

        # Time stepping.
        self.dt = 1.0e-4
        self.current_time = 0.0

        # Body identifiers and counts.
        self.body_id: NDArray[np.int64] = np.asarray(init.body_id, dtype=np.int64)
        self.outer_mask = self.body_id == 0
        self.core_mask = self.body_id == 1
        self.outer_idx = np.nonzero(self.outer_mask)[0]
        self.core_idx = np.nonzero(self.core_mask)[0]
        self.outer_local = np.arange(len(self.outer_idx), dtype=np.int64)
        self.core_local = np.arange(len(self.core_idx), dtype=np.int64)

        # Mass (kept per-sample from the scene).
        self.mass = np.asarray(init.mass, dtype=np.float64)
        if self.mass is None:
            n_total = len(init.position)
            self.mass = np.full(n_total, 1.0 / n_total, dtype=np.float64)

        # Parse the two meshes; these provide topology only.
        outer_verts, outer_faces = _load_obj(Path("assets/gummy_bear_outer.obj"))
        _core_verts, core_faces = _load_obj(Path("assets/gummy_bear_stiff.obj"))
        del outer_verts

        # Current state.
        self.X = np.asarray(init.position, dtype=np.float64).copy()
        self.V = np.asarray(init.velocity, dtype=np.float64).copy()

        # Store reference configuration for the outer shell.
        self.X0_outer = self.X[self.outer_idx].copy()
        self.V0_outer = self.V[self.outer_idx].copy()

        # Outer topology in local indexing.
        self.outer_faces = np.asarray(outer_faces, dtype=np.int64)
        self.core_faces = np.asarray(core_faces, dtype=np.int64)
        self.outer_edges, self.edge_k = self._build_outer_edges()
        self.edge_L0 = np.linalg.norm(
            self.X0_outer[self.outer_edges[:, 1]] - self.X0_outer[self.outer_edges[:, 0]], axis=1
        )

        # Effective per-edge stiffness (Pa * m / m = Pa, but used as N/m after
        # scaling by a representative cross-section).
        # Scale by thickness to obtain a physical spring constant.
        self.edge_k = self.edge_k * self.thickness

        # Per-edge damping coefficient (Rayleigh, zeta ~ 0.15).
        m_eff = 0.5 * self.mass[self.outer_edges].mean(axis=1)
        self.edge_c = 0.3 * np.sqrt(m_eff * self.edge_k)

        # Map each core vertex to the nearest point on the outer reference shell.
        self.core_tri_idx, self.core_bary = self._bind_core_to_shell()

        # Cumulative budget quantities.
        self.cumulative_external_impulse = np.zeros((2, 3), dtype=np.float64)

    def _build_outer_edges(self) -> Tuple[NDArrayf, NDArrayf]:
        """Return unique undirected edges of the outer shell and a base stiffness."""
        edges = set()
        for tri in self.outer_faces:
            a, b, c = sorted(tri)
            edges.add((a, b))
            edges.add((b, c))
            edges.add((a, c))
        edges = np.asarray(sorted(edges), dtype=np.int64)
        # A uniform base stiffness is later scaled by the edge rest length so
        # that shorter edges (higher strain resolution) are slightly softer.
        L0 = np.linalg.norm(
            self.X0_outer[edges[:, 1]] - self.X0_outer[edges[:, 0]], axis=1
        )
        k = self.soft_young / (1.0 + L0)
        return edges, k

    def _bind_core_to_shell(self) -> Tuple[NDArray[np.int64], NDArrayf]:
        """Barycentric binding of each core point to the reference outer shell."""
        core_points = self.X[self.core_idx]
        tri_idx = np.empty(len(core_points), dtype=np.int64)
        bary = np.empty((len(core_points), 3), dtype=np.float64)

        X0 = self.X0_outer
        faces = self.outer_faces
        for i, p in enumerate(core_points):
            # Search all outer triangles for the closest point (one-off cost).
            best = _nearest_triangle_point(p, X0[faces[:, 0]], X0[faces[:, 1]], X0[faces[:, 2]])
            tri_idx[i] = best[0]
            bary[i] = best[1]
        return tri_idx, bary

    def _update_core(self) -> None:
        """Kinematically embed the rigid core in the deforming outer shell."""
        Xo = self.X[self.outer_idx]
        Vo = self.V[self.outer_idx]
        faces = self.outer_faces[self.core_tri_idx]
        w = self.core_bary
        # positions
        self.X[self.core_idx] = (
            w[:, 0:1] * Xo[faces[:, 0]]
            + w[:, 1:2] * Xo[faces[:, 1]]
            + w[:, 2:3] * Xo[faces[:, 2]]
        )
        # velocities (same barycentric interpolation)
        self.V[self.core_idx] = (
            w[:, 0:1] * Vo[faces[:, 0]]
            + w[:, 1:2] * Vo[faces[:, 1]]
            + w[:, 2:3] * Vo[faces[:, 2]]
        )

    def _substep(self, dt: float) -> None:
        """Advance the outer shell by one sub-step and re-bind the core."""
        outer = self.outer_idx
        n_total = len(self.X)

        forces = np.zeros((n_total, 3), dtype=np.float64)

        # Gravity on all dynamic samples.
        forces[:, 2] += self.mass * self.gravity[2]

        # Edge spring / damping forces on the outer shell.
        i = self.outer_edges[:, 0]
        j = self.outer_edges[:, 1]
        xi = self.X[i]
        xj = self.X[j]
        r = xj - xi
        L = np.linalg.norm(r, axis=1, keepdims=True)
        L = np.where(L < 1e-12, 1.0, L)
        u = r / L

        # Spring part.
        f_spring = self.edge_k * (L.squeeze() - self.edge_L0)
        # Damping part along the edge.
        vi = self.V[i]
        vj = self.V[j]
        vrel = np.sum((vj - vi) * u, axis=1)
        f_damp = self.edge_c * vrel
        f_total = f_spring + f_damp
        f_vec = f_total[:, None] * u

        np.add.at(forces, i, f_vec)
        np.add.at(forces, j, -f_vec)

        # Ground contact for the outer shell.
        z = self.X[outer, 2]
        penetrate = -z
        in_contact = penetrate > 0.0
        if np.any(in_contact):
            v_outer = self.V[outer]
            v_z = v_outer[:, 2]
            # Normal force: penalty + damping on closing velocity.
            f_n_mag = self.ground_stiffness * penetrate - self.ground_damping * v_z
            f_n_mag = np.maximum(f_n_mag, 0.0)
            f_n = np.zeros((len(outer), 3), dtype=np.float64)
            f_n[in_contact, 2] = f_n_mag[in_contact]

            # Coulomb friction opposing tangential velocity.
            v_t = v_outer.copy()
            v_t[:, 2] = 0.0
            v_t_norm = np.linalg.norm(v_t, axis=1)
            friction_mask = in_contact & (v_t_norm > 1e-9)
            if np.any(friction_mask):
                dir_t = v_t[friction_mask] / v_t_norm[friction_mask, None]
                f_t = -self.friction_mu * f_n_mag[friction_mask, None] * dir_t
                f_n[friction_mask] += f_t
            forces[outer] += f_n

        # Light global velocity damping for stability.
        forces[outer] -= self.global_damping * self.V[outer]

        # Integrate the outer shell (semi-implicit Euler).
        self.V[outer] += dt * forces[outer] / self.mass[outer, None]
        self.X[outer] += dt * self.V[outer]

        # Re-derive the rigid core from the new outer configuration.
        self._update_core()

        # Budget accounting: external impulse on outer and core domains.
        # Gravity.
        grav_outer = self.mass[outer].sum() * self.gravity
        grav_core = self.mass[self.core_idx].sum() * self.gravity
        # Contact on outer only.
        contact_outer = np.zeros(3, dtype=np.float64)
        if np.any(in_contact):
            # Recompute contact force vector (same as above).
            z = self.X[outer, 2]
            penetrate = -z
            in_contact2 = penetrate > 0.0
            v_z = self.V[outer, 2]
            f_n_mag = self.ground_stiffness * penetrate - self.ground_damping * v_z
            f_n_mag = np.maximum(f_n_mag, 0.0)
            contact_forces = np.zeros((len(outer), 3), dtype=np.float64)
            contact_forces[in_contact2, 2] = f_n_mag[in_contact2]
            # friction again
            v_t = self.V[outer].copy()
            v_t[:, 2] = 0.0
            v_t_norm = np.linalg.norm(v_t, axis=1)
            friction_mask = in_contact2 & (v_t_norm > 1e-9)
            if np.any(friction_mask):
                dir_t = v_t[friction_mask] / v_t_norm[friction_mask, None]
                contact_forces[friction_mask] += (
                    -self.friction_mu * f_n_mag[friction_mask, None] * dir_t
                )
            contact_outer = contact_forces.sum(axis=0)

        domain_force = np.array([grav_outer + contact_outer, grav_core], dtype=np.float64)
        self.cumulative_external_impulse += domain_force * dt

    def advance_to(self, time: float) -> StateFrame:
        """Step physics to ``time`` and return a StateFrame."""
        while self.current_time < time - 1e-12:
            remaining = time - self.current_time
            step = min(self.dt, remaining)
            self._substep(step)
            self.current_time += step
        return self._build_frame(time)

    def diagnostics(self) -> dict:
        return {
            "dt": self.dt,
            "outer_vertices": int(self.outer_mask.sum()),
            "core_vertices": int(self.core_mask.sum()),
        }

    # ------------------------------------------------------------------
    # StateFrame construction
    # ------------------------------------------------------------------
    def _build_frame(self, time: float) -> StateFrame:
        n = len(self.X)
        # Element-wise deformation gradient: outer triangles first, then core.
        F_outer = self._element_deformation_gradients(self.X0_outer, self.X[self.outer_idx], self.outer_faces)
        F_core = np.tile(np.eye(3, dtype=np.float64)[None, :, :], (len(self.core_faces), 1, 1))
        F_all = np.concatenate([F_outer, F_core], axis=0)

        # Domain weights: row 0 = outer, row 1 = core.
        domain_weights = np.zeros((2, n), dtype=np.float64)
        domain_weights[0, self.outer_idx] = 1.0
        domain_weights[1, self.core_idx] = 1.0

        # External force per domain (gravity and contact).
        grav = np.zeros((2, 3), dtype=np.float64)
        grav[0] = self.mass[self.outer_idx].sum() * self.gravity
        grav[1] = self.mass[self.core_idx].sum() * self.gravity
        contact = np.zeros((2, 3), dtype=np.float64)
        # Contact on outer.
        z = self.X[self.outer_idx, 2]
        penetrate = -z
        in_contact = penetrate > 0.0
        if np.any(in_contact):
            v_z = self.V[self.outer_idx, 2]
            f_n_mag = self.ground_stiffness * penetrate - self.ground_damping * v_z
            f_n_mag = np.maximum(f_n_mag, 0.0)
            cf = np.zeros((len(self.outer_idx), 3), dtype=np.float64)
            cf[in_contact, 2] = f_n_mag[in_contact]
            v_t = self.V[self.outer_idx].copy()
            v_t[:, 2] = 0.0
            v_t_norm = np.linalg.norm(v_t, axis=1)
            mask = in_contact & (v_t_norm > 1e-9)
            if np.any(mask):
                dir_t = v_t[mask] / v_t_norm[mask, None]
                cf[mask] += -self.friction_mu * f_n_mag[mask, None] * dir_t
            contact[0] = cf.sum(axis=0)
        budget_external_force = np.stack([grav, contact], axis=1)  # (2,2,3)

        # Conserved mass split per material.
        mass_quantity = np.zeros((n, 2), dtype=np.float64)
        mass_quantity[self.outer_idx, 0] = self.mass[self.outer_idx]
        mass_quantity[self.core_idx, 1] = self.mass[self.core_idx]

        return StateFrame(
            time=float(time),
            position=self.X.copy(),
            velocity=self.V.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "material_region": self.body_id.astype(np.float64).copy(),
                "mass_quantity": mass_quantity,
            },
            scalars={"reference_momentum": 1.0},
            fields={
                "element_deformation_gradient": F_all,
                "budget_domain_weights": domain_weights,
                "budget_external_force": budget_external_force,
                "budget_cumulative_external_impulse": self.cumulative_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((2, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum": np.zeros((2, 3), dtype=np.float64),
                "cumulative_mass_source": np.zeros(2, dtype=np.float64),
                "interaction_force": np.zeros((2, 3), dtype=np.float64),
            },
        )

    @staticmethod
    def _element_deformation_gradients(
        X0: NDArrayf, X: NDArrayf, faces: NDArray[np.int64]
    ) -> NDArrayf:
        """Compute per-triangle 3x3 deformation gradients."""
        F = np.empty((len(faces), 3, 3), dtype=np.float64)
        for e, tri in enumerate(faces):
            x0 = X0[tri[0]]
            e1 = X0[tri[1]] - x0
            e2 = X0[tri[2]] - x0
            n0 = _unit(np.cross(e1, e2))

            y0 = X[tri[0]]
            f1 = X[tri[1]] - y0
            f2 = X[tri[2]] - y0
            n = _unit(np.cross(f1, f2))

            D0 = np.stack([e1, e2, n0], axis=1)  # 3x3
            D = np.stack([f1, f2, n], axis=1)     # 3x3
            try:
                F[e] = D @ np.linalg.inv(D0)
            except np.linalg.LinAlgError:
                F[e] = np.eye(3, dtype=np.float64)
        return F


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _unit(v: NDArrayf) -> NDArrayf:
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return np.zeros_like(v)
    return v / norm


def _load_obj(path: Path) -> Tuple[NDArrayf, List[List[int]]]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            parts = line.split()[1:]
            idx = [int(p.split("/")[0]) - 1 for p in parts]
            if len(idx) >= 3:
                faces.append(idx[:3])
                if len(idx) == 4:
                    faces.append([idx[0], idx[2], idx[3]])
    return np.asarray(vertices, dtype=np.float64), faces


def _nearest_triangle_point(
    p: NDArrayf,
    A: NDArrayf,
    B: NDArrayf,
    C: NDArrayf,
) -> Tuple[int, NDArrayf]:
    """Return (triangle_index, barycentric_coords) of the closest point on a set of triangles."""
    # A, B, C are (T,3)
    v0 = B - A
    v1 = C - A
    v2 = p - A
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-18, 1.0, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    # Clamp to triangle.
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    w = np.clip(w, 0.0, 1.0)
    # Re-normalize (approximate closest point on triangle).
    s = u + v + w
    s = np.where(s < 1e-12, 1.0, s)
    u /= s
    v /= s
    w /= s
    # Compute closest points.
    closest = u[:, None] * A + v[:, None] * B + w[:, None] * C
    dists = np.linalg.norm(closest - p, axis=1)
    best = int(np.argmin(dists))
    return best, np.array([u[best], v[best], w[best]], dtype=np.float64)
