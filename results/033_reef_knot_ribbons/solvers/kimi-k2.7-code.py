"""Physical solver for the reef-knot ribbons scene.

The simulation uses an explicit time-stepped mass-spring cloth model with
strain limiting, pairwise collision repulsion, and Coulomb friction.  The
two ribbons are pulled outward by scripting the velocity of their end
vertices, which reproduces the tightening of the reef knot.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        np.random.seed(seed)

        # Configuration
        self.config = getattr(scene, "config", {})
        ribbon_cfg = self.config.get("ribbon", {})
        self.thickness = float(ribbon_cfg.get("thickness", 0.006))
        self.strain_upper = float(ribbon_cfg.get("strain_upper", 1.134))
        self.friction = float(ribbon_cfg.get("friction", 0.02))
        self.pull_speed = float(self.config.get("pull_speed", 0.32))
        self.gravity = np.asarray(
            self.config.get("gravity", [0.0, 0.0, 0.0]), dtype=float
        )

        # Initial state
        init = scene.initial_state()
        self.x = np.asarray(init.position, dtype=float).copy()
        self.v = np.asarray(init.velocity, dtype=float).copy()
        if init.body_id is None:
            raise ValueError("StateFrame.body_id is required for ribbon binding")
        self.body_id = np.asarray(init.body_id, dtype=int).copy()
        self.num_nodes = self.x.shape[0]

        if init.mass is not None:
            self.mass = np.asarray(init.mass, dtype=float).copy()
        else:
            self.mass = None

        # Topology from the frozen metadata
        meta = scene.metadata()
        self.faces = np.asarray(meta["topology"]["faces"], dtype=int)

        # Unique edges (i < j)
        edge_set: set[tuple[int, int]] = set()
        for f in self.faces:
            a, b, c = f
            for i, j in ((a, b), (b, c), (a, c)):
                if i > j:
                    i, j = j, i
                edge_set.add((i, j))
        self.edges = np.asarray(sorted(edge_set), dtype=int)

        # Reference geometry
        self.rest_len = np.linalg.norm(
            self.x[self.edges[:, 0]] - self.x[self.edges[:, 1]], axis=1
        )
        self._build_triangle_reference_data()

        # Mass distribution
        if self.mass is None:
            total_ref_area = np.sum(self.tri_area)
            # approximate ribbon mass from area, thickness, and a nominal density
            density = 500.0
            total_mass = total_ref_area * self.thickness * density
            self.mass = np.full(self.num_nodes, total_mass / self.num_nodes, dtype=float)

        # End vertices for scripted pulling (one left and one right column per ribbon)
        self.end_left: list[NDArray[np.int64]] = []
        self.end_right: list[NDArray[np.int64]] = []
        self.mobile = np.ones(self.num_nodes, dtype=bool)
        for bid in np.unique(self.body_id):
            mask = self.body_id == bid
            idx = np.nonzero(mask)[0]
            xs = self.x[idx, 0]
            min_x, max_x = xs.min(), xs.max()
            tol = 0.05
            left = idx[np.abs(self.x[idx, 0] - min_x) < tol]
            right = idx[np.abs(self.x[idx, 0] - max_x) < tol]
            self.end_left.append(left)
            self.end_right.append(right)
            self.mobile[left] = False
            self.mobile[right] = False

        # Pair exclusion for collision: edge-sharing vertices on the same ribbon
        self.excluded = np.zeros((self.num_nodes, self.num_nodes), dtype=bool)
        for i, j in self.edges:
            self.excluded[i, j] = True
            self.excluded[j, i] = True
        self.eye = np.eye(self.num_nodes, dtype=bool)

        # Contact / material parameters
        self.contact_radius = 2.0 * self.thickness
        self.contact_margin = self.contact_radius
        self.k_spring = 1e4
        self.k_contact = 1e5
        self.damping = 0.05
        self.max_dt = 5e-4

        self.time = float(init.time)

        # Trajectory bookkeeping
        self.cumulative_mass_source = np.array([0.0], dtype=float)

        # Cache the initial frame so that advance_to(0.0) is cheap and consistent
        self._current_frame = self._build_frame()

    def _build_triangle_reference_data(self) -> None:
        """Precompute per-triangle reference Dm inverse and reference areas."""
        x0 = self.x
        a = self.faces[:, 0]
        b = self.faces[:, 1]
        c = self.faces[:, 2]
        E1 = x0[b] - x0[a]
        E2 = x0[c] - x0[a]
        self.tri_area = 0.5 * np.linalg.norm(np.cross(E1, E2), axis=1)

        # Orthonormal basis (t1, t2, n) for each reference triangle
        t1 = E1 / np.linalg.norm(E1, axis=1)[:, None]
        n = np.cross(E1, E2)
        n /= np.linalg.norm(n, axis=1)[:, None]
        t2 = np.cross(n, t1)

        # Material edge matrix Dm (2x2), columns are E1,E2 in material basis
        Dm = np.zeros((self.faces.shape[0], 2, 2), dtype=float)
        Dm[:, 0, 0] = np.einsum("ij,ij->i", E1, t1, dtype=float)
        Dm[:, 0, 1] = np.einsum("ij,ij->i", E1, t2, dtype=float)
        Dm[:, 1, 0] = np.einsum("ij,ij->i", E2, t1, dtype=float)
        Dm[:, 1, 1] = np.einsum("ij,ij->i", E2, t2, dtype=float)
        self.tri_Dm_inv = np.linalg.inv(Dm)
        self.tri_t1 = t1
        self.tri_t2 = t2
        self.tri_n = n

    def diagnostics(self) -> dict[str, object]:
        return {
            "num_nodes": self.num_nodes,
            "num_triangles": self.faces.shape[0],
            "num_edges": self.edges.shape[0],
            "current_time": float(self.time),
        }

    def advance_to(self, time: float) -> StateFrame:
        if time <= self.time:
            return self._current_frame

        dt = float(time) - self.time
        if dt <= 0.0:
            return self._current_frame

        n_sub = max(1, int(np.ceil(dt / self.max_dt)))
        h = dt / n_sub
        for _ in range(n_sub):
            self._substep(h)

        self.time = float(time)
        self._current_frame = self._build_frame()
        return self._current_frame

    def _substep(self, h: float) -> None:
        x = self.x
        v = self.v

        # Elastic edge forces
        diff = x[self.edges[:, 0]] - x[self.edges[:, 1]]
        L = np.linalg.norm(diff, axis=1)
        safe = L > 1e-12
        spring_dir = np.zeros_like(diff)
        np.divide(diff, L[:, None], out=spring_dir, where=safe[:, None])
        spring_mag = self.k_spring * (L - self.rest_len)
        spring_vec = spring_mag[:, None] * spring_dir

        forces = np.zeros((self.num_nodes, 3), dtype=float)
        np.add.at(forces, self.edges[:, 0], spring_vec)
        np.add.at(forces, self.edges[:, 1], -spring_vec)

        # Contact + friction forces (vectorized pairwise)
        forces += self._contact_forces(x, v)

        # Integrate
        v += h * (forces / self.mass[:, None] + self.gravity[None, :])
        v *= 1.0 - self.damping

        # Scripted end pulling
        self._apply_end_constraints(v)

        # Update positions
        x += h * v

        # Keep end vertices exactly on the scripted path
        self._reset_end_positions()

        # Strain limit (enforces the strict upper bound on edge stretch)
        self._strain_limit()

    def _contact_forces(self, x: NDArray, v: NDArray) -> NDArray:
        dx = x[:, None, :] - x[None, :, :]  # NxNx3
        d = np.linalg.norm(dx, axis=2)

        # Avoid self and excluded pairs
        mask = (~self.eye) & (~self.excluded)
        # Only act on pairs close enough to be active/near contacts
        gap = d - self.contact_radius
        mask &= gap < self.contact_margin

        # Normal direction
        n = np.zeros_like(dx)
        np.divide(dx, d[:, :, None], out=n, where=d[:, :, None] > 0.0)

        # Normal force magnitude (positive = repulsive when gap < 0)
        fn_mag = -self.k_contact * gap
        fn_mag = np.where(mask & (gap < 0.0), fn_mag, 0.0)

        fn = n * fn_mag[:, :, None]

        # Tangential relative velocity
        v_rel = v[:, None, :] - v[None, :, :]
        v_n_mag = np.einsum("ijk,ijk->ij", v_rel, n)
        v_t = v_rel - v_n_mag[:, :, None] * n
        v_t_mag = np.linalg.norm(v_t, axis=2, keepdims=True)

        ft = np.zeros_like(v_t)
        active = (fn_mag > 0.0) & (v_t_mag[:, :, 0] > 1e-8)
        active3 = active[:, :, None]
        ft_dir = np.zeros_like(v_t)
        np.divide(v_t, v_t_mag, out=ft_dir, where=v_t_mag > 0.0)
        ft = np.where(active3, -self.friction * fn_mag[:, :, None] * ft_dir, 0.0)

        pair_forces = np.where(mask[:, :, None], fn + ft, 0.0)
        return pair_forces.sum(axis=1)

    def _apply_end_constraints(self, v: NDArray) -> None:
        left_dir = np.array([-self.pull_speed, 0.0, 0.0], dtype=float)
        right_dir = np.array([self.pull_speed, 0.0, 0.0], dtype=float)
        for left, right in zip(self.end_left, self.end_right):
            v[left] = left_dir
            v[right] = right_dir

    def _reset_end_positions(self) -> None:
        # After drift during the previous substep, re-impose the exact pulled
        # position for the end vertices.  The interior vertices remain dynamic.
        for left in self.end_left:
            self.x[left, 0] = self.x[left, 0]  # no-op placeholder
        # The end vertices are already advanced with the scripted velocity,
        # so no additional correction is required here.

    def _strain_limit(self) -> None:
        target = self.rest_len * self.strain_upper
        # Multiple passes to distribute corrections
        for _ in range(5):
            diff = self.x[self.edges[:, 0]] - self.x[self.edges[:, 1]]
            L = np.linalg.norm(diff, axis=1)
            over = L > target
            if not np.any(over):
                break
            # displacement to apply to each side if both were free
            correction = diff * ((L - target) / (2.0 * L))[:, None]
            # avoid division by zero; over already ensures L > 0
            for (i, j), corr in zip(self.edges, correction):
                mi = self.mobile[i]
                mj = self.mobile[j]
                if mi and mj:
                    self.x[i] -= corr
                    self.x[j] += corr
                elif mi:
                    self.x[i] -= 2.0 * corr
                elif mj:
                    self.x[j] += 2.0 * corr
                # else both fixed: cannot be corrected

    def _build_frame(self) -> StateFrame:
        F = self._compute_deformation_gradient()
        signed = self._contact_signed_distances()
        return StateFrame(
            time=float(self.time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "contact_signed_distance": signed,
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
            },
            scalars={
                "contact_length_scale": float(self.thickness),
            },
        )

    def _compute_deformation_gradient(self) -> NDArray:
        x = self.x
        a = self.faces[:, 0]
        b = self.faces[:, 1]
        c = self.faces[:, 2]
        e1 = x[b] - x[a]
        e2 = x[c] - x[a]
        # Ds columns in world (3x2)
        Ds = np.stack([e1, e2], axis=2)
        # F = Ds @ Dm_inv for each triangle -> (T,3,2)
        F2 = np.einsum("tij,tjk->tik", Ds, self.tri_Dm_inv)
        # Third column is the deformed normal
        n = np.cross(F2[:, :, 0], F2[:, :, 1])
        norm = np.linalg.norm(n, axis=1)
        norm = np.where(norm > 0.0, norm, 1.0)
        n /= norm[:, None]
        F = np.zeros((self.faces.shape[0], 3, 3), dtype=float)
        F[:, :, :2] = F2
        F[:, :, 2] = n

        # Average to nodes weighted by reference area
        per_node = np.zeros((self.num_nodes, 3, 3), dtype=float)
        weight = np.zeros(self.num_nodes, dtype=float)
        w = self.tri_area
        for idx in range(3):
            nodes = self.faces[:, idx]
            np.add.at(per_node, (nodes, slice(None), slice(None)), F * w[:, None, None])
            np.add.at(weight, nodes, w)
        weight = np.where(weight > 0.0, weight, 1.0)
        per_node /= weight[:, None, None]
        return per_node

    def _contact_signed_distances(self) -> NDArray:
        dx = self.x[:, None, :] - self.x[None, :, :]
        d = np.linalg.norm(dx, axis=2)
        gap = d - self.contact_radius
        mask = (~self.eye) & (~self.excluded) & (gap < self.contact_margin)
        signed = gap[mask]
        if signed.size == 0:
            return np.zeros(0, dtype=float)
        return signed
