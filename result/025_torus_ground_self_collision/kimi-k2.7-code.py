"""Physically-based elastic solver for the torus ground self-collision scene.

Implements a stable explicit time-stepping integrator for a deformable triangle
mesh subject to gravity, internal elastic edge forces, ground contact, and
penalty-based self-collision.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - runtime fallback only
    cKDTree = None  # type: ignore


class BenchmarkSolver:
    """Explicit spring-network deformable solver.

    The solver uses a fixed sub-step that is a small integer fraction of the
    requested sample interval.  It pre-computes all sample frames during
    ``initialize`` so ``advance_to`` is a fast lookup.
    """

    def initialize(self, scene, seed: int) -> None:
        np.random.seed(seed)

        self.config = scene.config

        # Physical parameters.
        self.stiffness = float(self.config.get("stiffness", 13.0))
        self.damping_coeff = float(self.config.get("damping", 1.2))
        self.restitution = float(self.config.get("restitution", 0.2))
        self.friction_coeff = float(self.config.get("friction", 0.58))
        self.contact_length_scale = float(self.config.get("contact_length_scale", 0.04))
        self.minor_radius = float(self.config.get("minor_radius", 0.17))
        self.ground_z = float(self.config.get("ground_z", 0.0))
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # Initial state from the frozen scene.
        frame0 = scene.initial_state()
        self.position = np.asarray(frame0.position, dtype=float).copy()
        self.velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.N = self.position.shape[0]
        self.body_id = np.zeros(self.N, dtype=np.int64)

        # Reference (material) configuration.
        self.rest_position = self.position.copy()

        # Load topology and build edge list.
        self.edges: np.ndarray | None = None
        self.rest_len: np.ndarray | None = None
        self._load_edges()

        # Neighbor lists for per-vertex deformation gradients and self-collision.
        self.neighbors: list[list[int]] = [[] for _ in range(self.N)]
        self.edge_pair_set: set[frozenset[int]] = set()
        if self.edges is not None:
            for i, j in self.edges:
                self.neighbors[i].append(j)
                self.neighbors[j].append(i)
                self.edge_pair_set.add(frozenset((int(i), int(j))))

        # Contact parameters.
        self.self_cut = 2.0 * self.contact_length_scale  # penalty activation distance
        self.k_self = 100.0 * self.stiffness
        self.c_self = 0.5 * self.damping_coeff
        self.k_ground = 500.0 * self.stiffness
        self.c_ground = 2.0 * self.damping_coeff

        # Time-stepping.
        self.sample_dt = float(self.config["sample_dt"])
        # Use a small fixed sub-step to keep the explicit spring integrator stable.
        self._fixed_dt = float(self.sample_dt / 240.0)
        self.dt = self._fixed_dt
        self.duration = float(self.config["duration"])

        # Pre-compute all requested frames.
        self.sample_times = scene.sample_times()
        self._frames: list[StateFrame] = []
        self._fixed_dt = float(self.dt)
        self._step_and_sample()

    def _load_edges(self) -> None:
        asset = Path(self.config["subject_asset"])
        if not asset.is_file():
            return
        try:
            data = np.load(asset)
        except Exception:
            return
        if "triangles" not in data:
            return
        triangles = np.asarray(data["triangles"], dtype=np.int64)
        if triangles.size == 0:
            return
        e0 = triangles[:, [0, 1]]
        e1 = triangles[:, [1, 2]]
        e2 = triangles[:, [2, 0]]
        edges = np.concatenate([e0, e1, e2], axis=0)
        edges = np.sort(edges, axis=1)
        self.edges = np.unique(edges, axis=0)
        x0 = self.rest_position[self.edges[:, 0]]
        x1 = self.rest_position[self.edges[:, 1]]
        self.rest_len = np.linalg.norm(x1 - x0, axis=1)

    def _elastic_forces(self) -> np.ndarray:
        forces = np.zeros((self.N, 3), dtype=float)
        if self.edges is None or self.rest_len is None:
            return forces
        i = self.edges[:, 0]
        j = self.edges[:, 1]
        dx = self.position[j] - self.position[i]
        dv = self.velocity[j] - self.velocity[i]
        lengths = np.linalg.norm(dx, axis=1)
        safe = lengths > 1e-12
        if not np.any(safe):
            return forces
        dir_vec = np.zeros_like(dx)
        dir_vec[safe] = dx[safe] / lengths[safe][:, None]
        strain = np.zeros_like(lengths)
        strain[safe] = lengths[safe] - self.rest_len[safe]
        v_rel = np.einsum("ij,ij->i", dv, dir_vec)
        fmag = self.stiffness * strain + self.damping_coeff * v_rel
        f_edge = fmag[:, None] * dir_vec
        np.add.at(forces, i, f_edge)
        np.add.at(forces, j, -f_edge)
        return forces

    def _self_collision_forces(self) -> np.ndarray:
        forces = np.zeros((self.N, 3), dtype=float)
        if self.N < 2:
            return forces

        pairs = self._nearby_pairs()
        if pairs is None or len(pairs) == 0:
            return forces

        i = pairs[:, 0]
        j = pairs[:, 1]
        dx = self.position[j] - self.position[i]
        dv = self.velocity[j] - self.velocity[i]
        dist = np.linalg.norm(dx, axis=1)
        valid = (dist > 1e-12) & (dist < self.self_cut)
        if not np.any(valid):
            return forces

        i = i[valid]
        j = j[valid]
        dx = dx[valid]
        dv = dv[valid]
        dist = dist[valid]
        dir_vec = dx / dist[:, None]
        overlap = self.self_cut - dist
        v_n = np.einsum("ij,ij->i", dv, dir_vec)
        fmag = self.k_self * overlap - self.c_self * v_n
        f_edge = fmag[:, None] * dir_vec
        np.add.at(forces, i, -f_edge)
        np.add.at(forces, j, f_edge)
        return forces

    def _nearby_pairs(self) -> np.ndarray | None:
        if cKDTree is not None:
            tree = cKDTree(self.position)
            pairs = tree.query_pairs(r=self.self_cut, output_type="ndarray")
            return pairs
        # Fallback brute force for very small scenes.
        pairs = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                pairs.append((i, j))
        return np.asarray(pairs, dtype=np.int64) if pairs else None

    def _ground_forces(self) -> np.ndarray:
        forces = np.zeros((self.N, 3), dtype=float)
        z = self.position[:, 2]
        v_z = self.velocity[:, 2]
        gap = z - self.ground_z
        in_contact = gap < 0.0
        if not np.any(in_contact):
            return forces

        depth = -gap[in_contact]
        fn = self.k_ground * depth - self.c_ground * v_z[in_contact]
        np.maximum(fn, 0.0, out=fn)
        # Normal force up.
        forces[in_contact, 2] += fn

        # Coulomb friction on the tangential plane.
        v_t = self.velocity[in_contact, :2]
        v_t_norm = np.linalg.norm(v_t, axis=1)
        sliding = v_t_norm > 1e-8
        if np.any(sliding):
            idx = np.where(in_contact)[0][sliding]
            dir_t = v_t[sliding] / v_t_norm[sliding][:, None]
            f_t = np.minimum(self.friction_coeff * fn[sliding], self.mass[idx] * v_t_norm[sliding] / self.dt)
            forces[idx, :2] -= f_t[:, None] * dir_t
        return forces

    def _external_force(self) -> np.ndarray:
        return self.mass[:, None] * self.gravity[None, :]

    def _step(self) -> None:
        # Symplectic (explicit) Euler integration.
        f_ext = self._external_force()
        f_elastic = self._elastic_forces()
        f_ground = self._ground_forces()
        f_self = self._self_collision_forces()

        total_force = f_ext + f_elastic + f_ground + f_self
        inv_mass = 1.0 / self.mass
        self.velocity += self.dt * total_force * inv_mass[:, None]

        # Ground restitution: reflect downward normal velocity at contact.
        in_contact = self.position[:, 2] <= self.ground_z
        approaching = self.velocity[:, 2] < 0.0
        reflect = in_contact & approaching
        self.velocity[reflect, 2] = -self.restitution * self.velocity[reflect, 2]

        self.position += self.dt * self.velocity

    def _deformation_gradients(self) -> np.ndarray:
        F = np.zeros((self.N, 3, 3), dtype=float)
        for i in range(self.N):
            nbrs = self.neighbors[i]
            if len(nbrs) < 3:
                F[i] = np.eye(3)
                continue
            rest_nbr = self.rest_position[nbrs]
            cur_nbr = self.position[nbrs]
            dX = rest_nbr - self.rest_position[i]
            dx = cur_nbr - self.position[i]
            w = 1.0 / (np.einsum("ij,ij->i", dX, dX) + 1e-12)
            # Least-squares: F * dX = dx.
            A = (dX.T * w).dot(dX)
            B = (dx.T * w).dot(dX)
            try:
                A_inv = np.linalg.pinv(A)
                Fi = B.dot(A_inv)
            except Exception:  # pragma: no cover
                Fi = np.eye(3)
            if not np.isfinite(Fi).all():
                Fi = np.eye(3)
            F[i] = Fi
        return F

    def _make_frame(self, time: float) -> StateFrame:
        F = self._deformation_gradients()
        gap = self.position[:, 2] - self.ground_z
        external = self.mass[:, None] * self.gravity[None, :]
        return StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": external,
                "contact_signed_distance": gap,
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )

    def _step_and_sample(self) -> None:
        t = 0.0
        self._frames.append(self._make_frame(t))
        sample_idx = 1
        while sample_idx < len(self.sample_times):
            target = float(self.sample_times[sample_idx])
            # Integrate until we reach the next output time.
            while t < target - 1e-12:
                # Don't overshoot.
                step = min(self._fixed_dt, target - t)
                self.dt = step
                self._step()
                t += step
            # Clamp to target to avoid floating-point drift.
            self._frames.append(self._make_frame(target))
            sample_idx += 1

    def advance_to(self, time: float) -> StateFrame:
        for frame in self._frames:
            if abs(frame.time - time) < 1e-9:
                return frame
        # Fallback: return the closest stored frame.
        best = min(self._frames, key=lambda f: abs(f.time - time))
        return best

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "elastic_spring_network",
            "substep_dt": self.dt,
            "frames_stored": len(self._frames),
        }
