"""Elastic torus pile solver.

Implements a physically-based explicit time-stepper for forty incompressible
elastic tori represented as closed-surface volumetric bodies.  Each torus is
discretized by the provided triangle mesh; incompressibility is enforced by a
per-body volume constraint on the enclosed volume.  Edge springs preserve local
shape, gravity drives the pile, and vertex-vertex/vertex-obstacle impulses handle
contact with the floor, posts and other tori.

Only NumPy and SciPy are used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Solver for 057_elastic_tori_pile."""

    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = seed

        # Physical parameters.
        self._g = np.array([0.0, 0.0, -9.81])
        self._dt = 1e-3
        self._k_edge = 4.0e3  # edge spring stiffness (N/m)
        self._c_edge = 0.8  # edge damping
        self._k_vol = 5.0e3  # volume stiffness (Pa)
        self._c_vol = 2.0e3  # volume damping (Pa*s/m^3)
        self._damping = 0.3  # global velocity damping (1/s)
        self._contact_r = 0.06  # vertex-vertex contact threshold (m)
        self._restitution = 0.0

        # Load torus reference mesh.
        self._ref_pos_local, self._triangles_local = self._load_torus()
        self._verts_per_body = len(self._ref_pos_local)
        self._faces_per_body = len(self._triangles_local)

        # Edge list and rest lengths.
        self._edges_local = self._build_edges(self._triangles_local)

        # Initial state from frozen scene.
        frame0 = scene.initial_state()
        self._time = float(frame0.time)
        self._pos = np.asarray(frame0.position, dtype=float).copy()
        self._vel = np.asarray(frame0.velocity, dtype=float).copy()
        self._mass = np.asarray(frame0.mass, dtype=float).copy()
        self._body_id = np.asarray(frame0.body_id, dtype=int).copy()
        self._orient = np.asarray(frame0.orientation, dtype=float).copy()
        self._rest_pos = np.asarray(frame0.attributes["rest_position"], dtype=float).copy()

        self._n = len(self._pos)
        self._bodies = int(self._body_id.max()) + 1

        # Global mesh topology.
        self._edges_i, self._edges_j, self._edge_rest = self._globalize_edges()
        self._triangles = self._globalize_triangles()
        self._tri_body = self._triangles[:, 0] // self._verts_per_body

        # Volume data.
        self._rest_volume = self._body_volumes(self._rest_pos)
        self._prev_volume = self._rest_volume.copy()

        # Posts from scene geometry.
        self._posts = self._load_posts()

        # Budget accumulators.
        self._cum_external_impulse = np.zeros((self._bodies, 3), dtype=float)

        # Domain weights (one row per torus).
        self._domain_weights = np.zeros((self._bodies, self._n), dtype=float)
        for b in range(self._bodies):
            start = b * self._verts_per_body
            self._domain_weights[b, start : start + self._verts_per_body] = 1.0

    # --------------------------------------------------------------------- #
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self._time:
            self._step_to(target)
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {"dt": self._dt, "substeps": int(self._time / self._dt)}

    # --------------------------------------------------------------------- #
    def _load_torus(self) -> tuple[np.ndarray, np.ndarray]:
        asset = Path("assets/torus.obj")
        if not asset.exists():
            raise FileNotFoundError("assets/torus.obj not found")
        verts, faces = [], []
        with asset.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":
                    verts.append([float(x) for x in parts[1:4]])
                elif parts[0] == "f":
                    faces.append([int(x.split("/")[0]) - 1 for x in parts[1:4]])
        return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)

    def _build_edges(self, faces: np.ndarray) -> np.ndarray:
        edge_set = set()
        for a, b, c in faces:
            edge_set.add((min(a, b), max(a, b)))
            edge_set.add((min(b, c), max(b, c)))
            edge_set.add((min(c, a), max(c, a)))
        return np.asarray(sorted(edge_set), dtype=int)

    def _globalize_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        local_i = self._edges_local[:, 0]
        local_j = self._edges_local[:, 1]
        i_list, j_list, rest_list = [], [], []
        for b in range(self._bodies):
            off = b * self._verts_per_body
            i_list.append(local_i + off)
            j_list.append(local_j + off)
            p_i = self._rest_pos[local_i + off]
            p_j = self._rest_pos[local_j + off]
            rest_list.append(np.linalg.norm(p_j - p_i, axis=1))
        return (
            np.concatenate(i_list),
            np.concatenate(j_list),
            np.concatenate(rest_list),
        )

    def _globalize_triangles(self) -> np.ndarray:
        triangles = []
        for b in range(self._bodies):
            off = b * self._verts_per_body
            triangles.append(self._triangles_local + off)
        return np.vstack(triangles)

    def _load_posts(self) -> list[tuple[float, float, float, float]]:
        return [
            (2.35, 0.0, 0.13, 2.8),
            (2.0352, 1.175, 0.13, 2.8),
            (1.175, 2.0352, 0.13, 2.8),
            (0.0, 2.35, 0.13, 2.8),
            (-1.175, 2.0352, 0.13, 2.8),
            (-2.0352, 1.175, 0.13, 2.8),
            (-2.35, 0.0, 0.13, 2.8),
            (-2.0352, -1.175, 0.13, 2.8),
            (-1.175, -2.0352, 0.13, 2.8),
            (-0.0, -2.35, 0.13, 2.8),
            (1.175, -2.0352, 0.13, 2.8),
            (2.0352, -1.175, 0.13, 2.8),
        ]

    # --------------------------------------------------------------------- #
    def _step_to(self, target: float) -> None:
        while self._time < target:
            dt = min(self._dt, target - self._time)
            self._substep(dt)

    def _substep(self, dt: float) -> None:
        # Elastic edge forces.
        f = np.zeros_like(self._pos)
        dx = self._pos[self._edges_j] - self._pos[self._edges_i]
        dist = np.linalg.norm(dx, axis=1)
        dist = np.maximum(dist, 1e-12)
        e = dx / dist[:, None]
        fmag = self._k_edge * (dist - self._edge_rest)
        dv = self._vel[self._edges_j] - self._vel[self._edges_i]
        vrel = np.einsum("ij,ij->i", dv, e)
        fmag = fmag + self._c_edge * vrel
        f_edge = fmag[:, None] * e
        np.add.at(f, self._edges_i, f_edge)
        np.add.at(f, self._edges_j, -f_edge)

        # Gravity.
        f[:, 2] += self._mass * self._g[2]

        # Per-body volume preservation (incompressibility).
        current_volume = self._body_volumes(self._pos)
        dVdt = (current_volume - self._prev_volume) / dt if dt > 0.0 else np.zeros_like(current_volume)
        pressure = self._k_vol * (current_volume - self._rest_volume) / self._rest_volume
        pressure += self._c_vol * dVdt / self._rest_volume
        self._add_volume_forces(f, pressure)
        self._prev_volume = current_volume

        # Velocity update.
        self._vel += (f / self._mass[:, None]) * dt
        self._vel *= 1.0 - self._damping * dt

        # Position update.
        self._pos += self._vel * dt

        # Contact impulse accumulator.
        contact_impulse = np.zeros((self._bodies, 3), dtype=float)

        # Inter-body vertex-vertex contact.
        tree = cKDTree(self._pos)
        pairs = tree.query_pairs(r=self._contact_r, output_type="ndarray")
        if pairs.size:
            i = pairs[:, 0]
            j = pairs[:, 1]
            diff_body = self._body_id[i] != self._body_id[j]
            i = i[diff_body]
            j = j[diff_body]
            if i.size:
                dx = self._pos[i] - self._pos[j]
                d = np.linalg.norm(dx, axis=1)
                valid = d > 1e-12
                i = i[valid]
                j = j[valid]
                dx = dx[valid]
                d = d[valid]
                n = dx / d[:, None]
                overlap = self._contact_r - d
                correction = 0.5 * overlap[:, None] * n
                self._pos[i] += correction
                self._pos[j] -= correction
                vrel = np.einsum("ij,ij->i", self._vel[i] - self._vel[j], n)
                mask = vrel < 0.0
                if np.any(mask):
                    ii = i[mask]
                    jj = j[mask]
                    n_m = n[mask]
                    m_i = self._mass[ii]
                    m_j = self._mass[jj]
                    v_m = vrel[mask]
                    eff_mass = 1.0 / (1.0 / m_i + 1.0 / m_j)
                    impulse_mag = -(1.0 + self._restitution) * eff_mass * v_m
                    self._vel[ii] += (impulse_mag / m_i)[:, None] * n_m
                    self._vel[jj] -= (impulse_mag / m_j)[:, None] * n_m
                    np.add.at(contact_impulse, self._body_id[ii], impulse_mag[:, None] * n_m)
                    np.add.at(contact_impulse, self._body_id[jj], -impulse_mag[:, None] * n_m)

        # Floor.
        pen = -self._pos[:, 2]
        floor_mask = (pen > 0.0) & (self._vel[:, 2] < 0.0)
        if np.any(floor_mask):
            m_mask = self._mass[floor_mask]
            v_z = self._vel[floor_mask, 2]
            impulse = -m_mask * v_z
            np.add.at(contact_impulse[:, 2], self._body_id[floor_mask], impulse)
            self._pos[floor_mask, 2] = 0.0
            self._vel[floor_mask, 2] = 0.0

        # Posts.
        for px, py, radius, height in self._posts:
            r_vec = self._pos[:, :2] - np.array([px, py])
            r = np.linalg.norm(r_vec, axis=1)
            mask = (self._pos[:, 2] > 0.0) & (self._pos[:, 2] < height) & (r < radius)
            if not np.any(mask):
                continue
            unit = r_vec[mask] / r[mask][:, None]
            self._pos[mask, :2] = np.array([px, py]) + unit * radius
            v_rad = np.einsum("ij,ij->i", self._vel[mask, :2], unit)
            mask_in = v_rad < 0.0
            if np.any(mask_in):
                bmask = self._body_id[mask]
                m_mask = self._mass[mask]
                impulse_mag = -m_mask[mask_in] * v_rad[mask_in]
                np.add.at(
                    contact_impulse[:, :2],
                    bmask[mask_in],
                    impulse_mag[:, None] * unit[mask_in],
                )
                self._vel[mask, :2][mask_in] -= v_rad[mask_in][:, None] * unit[mask_in]

        # Update cumulative external impulse (contact only; gravity is budget_external_force).
        self._cum_external_impulse += contact_impulse
        self._time += dt

    # --------------------------------------------------------------------- #
    def _add_volume_forces(self, f: np.ndarray, pressure: np.ndarray) -> None:
        """Add pressure forces from the per-body volume constraint.

        For a triangle with vertices a,b,c the contribution to the signed enclosed
        volume is V = (1/6) p_a . (p_b x p_c).  The gradients are
        grad_a V = (1/6) (p_b x p_c), grad_b V = (1/6) (p_c x p_a),
        grad_c V = (1/6) (p_a x p_b).
        The force on each vertex is -p_body * grad V, where p_body is the body's
        pressure.
        """
        tri = self._triangles
        p0 = self._pos[tri[:, 0]]
        p1 = self._pos[tri[:, 1]]
        p2 = self._pos[tri[:, 2]]
        g0 = np.cross(p1, p2) / 6.0
        g1 = np.cross(p2, p0) / 6.0
        g2 = np.cross(p0, p1) / 6.0
        coeff = -pressure[self._tri_body][:, None]
        np.add.at(f, tri[:, 0], coeff * g0)
        np.add.at(f, tri[:, 1], coeff * g1)
        np.add.at(f, tri[:, 2], coeff * g2)

    def _body_volumes(self, pos: np.ndarray) -> np.ndarray:
        tri = self._triangles
        p0 = pos[tri[:, 0]]
        p1 = pos[tri[:, 1]]
        p2 = pos[tri[:, 2]]
        vol = np.einsum("ij,ij->i", p0, np.cross(p1, p2)) / 6.0
        total = np.zeros(self._bodies, dtype=float)
        np.add.at(total, self._tri_body, vol)
        return total

    def _deformation_gradients(self, pos: np.ndarray) -> np.ndarray:
        tri = self._triangles
        p0 = pos[tri[:, 0]]
        p1 = pos[tri[:, 1]]
        p2 = pos[tri[:, 2]]
        q0 = self._rest_pos[tri[:, 0]]
        q1 = self._rest_pos[tri[:, 1]]
        q2 = self._rest_pos[tri[:, 2]]
        E = np.stack([p1 - p0, p2 - p0], axis=2)  # (T,3,2)
        D = np.stack([q1 - q0, q2 - q0], axis=2)  # (T,3,2)
        A = np.einsum("nki,nkj->nij", D, D)  # (T,2,2)
        invA = np.linalg.inv(A)
        pinvD = np.einsum("nij,nkj->nik", invA, D)  # (T,2,3)
        F = np.einsum("nik,nkj->nij", E, pinvD)  # (T,3,3)
        return F

    def _make_frame(self, time: float) -> StateFrame:
        current_volume = self._body_volumes(self._pos)
        F = self._deformation_gradients(self._pos)

        # Per-body gravity external force.
        gravity_force = np.zeros((self._bodies, 3), dtype=float)
        np.add.at(gravity_force, self._body_id, self._mass[:, None] * self._g)

        frame = StateFrame(
            time=time,
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            orientation=self._orient.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={
                "rest_position": self._rest_pos.copy(),
                "mass_quantity": self._mass.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self._cum_external_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((self._bodies, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((self._bodies, 3), dtype=float),
                "budget_domain_weights": self._domain_weights.copy(),
                "budget_external_force": gravity_force[:, None, :],
                "constraint_value": current_volume.copy(),
                "constraint_target": self._rest_volume.copy(),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "element_deformation_gradient": F,
            },
            scalars={
                "constraint_scale": 1.0,
                "reference_momentum": 100.0,
            },
        )
        frame.validate()
        return frame
