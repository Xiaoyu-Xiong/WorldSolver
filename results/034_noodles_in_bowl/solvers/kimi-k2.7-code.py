"""Physically-based solver for 034_noodles_in_bowl.

The scene contains 625 straight noodles, each represented by two endpoint
nodes (1250 nodes total).  The solver integrates the endpoint positions with
a position-based dynamics style time stepping scheme:

    1. explicit gravity velocity update,
    2. semi-implicit position prediction,
    3. bowl collision correction,
    4. PBD iterations enforcing rod length and noodle/noodle repulsion,
    5. velocity reconstruction from position change.

Only NumPy/SciPy are used; no Genesis code is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        self.seed = seed
        self.config = scene.config

        # Gravity and geometry parameters.
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.radius = float(self.config.get("radius", 0.006))
        self.noodle_length = float(self.config.get("noodle_length", 0.4))
        self.sample_dt = float(self.config["sample_dt"])

        # Time stepping.
        self.substeps = 60
        self.dt = self.sample_dt / self.substeps
        self.constraint_iters = 3
        self.damping = 0.5  # exponential velocity damping, 1/s

        # Initial state from the frozen scaffold.
        frame0 = scene.initial_state()
        self.pos = np.asarray(frame0.position, dtype=float).copy()
        self.vel = np.asarray(frame0.velocity, dtype=float).copy()
        self.time = float(frame0.time)

        n_nodes = self.pos.shape[0]
        if frame0.mass is not None and np.asarray(frame0.mass).shape[0] == n_nodes:
            self.mass = np.asarray(frame0.mass, dtype=float).copy()
        else:
            density = 1000.0  # kg/m^3, typical for pasta/water mixture
            rod_mass = density * np.pi * self.radius**2 * self.noodle_length
            self.mass = np.full(n_nodes, 0.5 * rod_mass, dtype=float)

        # Topology: each edge is one noodle.
        workspace = Path(__file__).resolve().parent.parent
        meta = json.loads((workspace / "assets" / "metadata.json").read_text(encoding="utf-8"))
        self.edges = np.asarray(meta["edge"], dtype=int)

        # Map each node to its noodle (edge) index.
        self.node_noodle = np.empty(n_nodes, dtype=int)
        self.node_noodle[self.edges[:, 0]] = np.arange(self.edges.shape[0])
        self.node_noodle[self.edges[:, 1]] = np.arange(self.edges.shape[0])

        # Rest lengths of the noodles.
        p0 = self.pos[self.edges[:, 0]]
        p1 = self.pos[self.edges[:, 1]]
        self.rest_len = np.linalg.norm(p1 - p0, axis=1)

        # Bowl profile (meridian), including an artificial bottom disk at r=0.
        profile = [(0.0, 0.0)] + list(self.config.get("bowl", {}).get("inner_profile", [
            [0.12, 0.0],
            [0.38, 0.08],
            [0.66, 0.34],
            [0.72, 0.48],
        ]))
        profile = np.asarray(profile, dtype=float)
        self.rim_z = profile[-1, 1]
        self.bowl_r0 = profile[:-1, 0]
        self.bowl_z0 = profile[:-1, 1]
        self.bowl_dr = np.diff(profile[:, 0])
        self.bowl_dz = np.diff(profile[:, 1])
        self.bowl_len2 = self.bowl_dr**2 + self.bowl_dz**2
        self.bowl_len = np.sqrt(self.bowl_len2)

        # Self-contact samples along each noodle.
        self.samples_per_noodle = 4
        self.sample_t = np.linspace(0.0, 1.0, self.samples_per_noodle, dtype=float)
        self.sample_tau = np.tile(self.sample_t, self.edges.shape[0])
        self.sample_noodle = np.repeat(np.arange(self.edges.shape[0]), self.samples_per_noodle)

        # Caches.
        self._node_idx = np.arange(n_nodes)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def advance_to(self, time: float) -> StateFrame:
        if time <= self.time + 1e-12:
            return self._make_frame(time)

        n_steps = int(round((time - self.time) / self.dt))
        for _ in range(n_steps):
            self._substep()
            self.time += self.dt

        return self._make_frame(time)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "solver": "pbd_noodle_solver",
            "seed": self.seed,
            "substeps_per_sample": self.substeps,
            "dt": self.dt,
            "node_count": self.pos.shape[0],
            "noodle_count": self.edges.shape[0],
        }

    # --------------------------------------------------------------------- #
    # Time stepping
    # --------------------------------------------------------------------- #
    def _substep(self) -> None:
        dt = self.dt
        # Gravity.
        self.vel += self.gravity * dt
        # Predict position.
        p_old = self.pos.copy()
        self.pos += self.vel * dt
        # Constraint/collision projection.
        for _ in range(self.constraint_iters):
            self._resolve_bowl(self.pos)
            self._distance_constraints(self.pos)
            self._self_contact()
        # Recompute velocity from corrected position.
        self.vel = (self.pos - p_old) / dt
        # Global damping to help reach a stable equilibrium.
        self.vel *= np.exp(-self.damping * dt)

    # --------------------------------------------------------------------- #
    # Bowl collision
    # --------------------------------------------------------------------- #
    def _bowl_distance_and_normal(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Signed distance (positive inside) and inward unit normal for the bowl."""
        rho = np.hypot(pos[:, 0], pos[:, 1])
        z = pos[:, 2]

        rho = rho[:, None]  # (N,1)
        z = z[:, None]

        r0 = self.bowl_r0[None, :]  # (1,P)
        z0 = self.bowl_z0[None, :]
        dr = self.bowl_dr[None, :]
        dz = self.bowl_dz[None, :]
        len2 = self.bowl_len2[None, :]

        # Project onto each segment.
        t = ((rho - r0) * dr + (z - z0) * dz) / len2
        t = np.clip(t, 0.0, 1.0)
        cr = r0 + t * dr
        cz = z0 + t * dz
        diff_r = rho - cr
        diff_z = z - cz
        dist = np.sqrt(diff_r**2 + diff_z**2)

        # Cross product gives the side of the directed segment the point lies on.
        cross = dr * (z - z0) - dz * (rho - r0)
        signed = np.where(cross >= 0.0, dist, -dist)

        # Pick the segment that realises the closest point.
        seg_idx = np.argmin(np.abs(signed), axis=1)
        sd = signed[np.arange(pos.shape[0]), seg_idx]

        # Inward 2-D normal for the selected segment.
        nr = -self.bowl_dz[seg_idx] / self.bowl_len[seg_idx]
        nz = self.bowl_dr[seg_idx] / self.bowl_len[seg_idx]

        # Convert to 3-D, axisymmetric about the z-axis.
        rho_xy = np.maximum(np.hypot(pos[:, 0], pos[:, 1]), 1e-12)
        nx = nr * (pos[:, 0] / rho_xy)
        ny = nr * (pos[:, 1] / rho_xy)
        n = np.stack([nx, ny, nz], axis=1)
        norm = np.linalg.norm(n, axis=1)
        n[norm > 0] /= norm[norm > 0][:, None]
        return sd, n

    def _resolve_bowl(self, pos: np.ndarray) -> None:
        sd, n = self._bowl_distance_and_normal(pos)
        pen = self.radius - sd
        # Only apply to points inside/under the rim region.
        mask = (pen > 0.0) & (pos[:, 2] <= self.rim_z + 2.0 * self.radius)
        if not np.any(mask):
            return
        pos[mask] += n[mask] * pen[mask][:, None]

        # Remove wall-normal velocity to avoid bounces (tangential motion stays).
        vdot = np.einsum("ij,ij->i", self.vel[mask], n[mask])
        self.vel[mask] -= (vdot / np.maximum(np.linalg.norm(n[mask], axis=1), 1e-12))[:, None] * n[mask]

    # --------------------------------------------------------------------- #
    # Rod length constraints
    # --------------------------------------------------------------------- #
    def _distance_constraints(self, pos: np.ndarray) -> None:
        p0 = pos[self.edges[:, 0]]
        p1 = pos[self.edges[:, 1]]
        dvec = p1 - p0
        d = np.linalg.norm(dvec, axis=1)
        mask = d > 1e-12
        if not np.any(mask):
            return
        diff = 0.5 * (d[mask] - self.rest_len[mask]) / d[mask]
        u = dvec[mask] / d[mask][:, None]
        pos[self.edges[mask, 0]] += diff[:, None] * u
        pos[self.edges[mask, 1]] -= diff[:, None] * u

    # --------------------------------------------------------------------- #
    # Self contact
    # --------------------------------------------------------------------- #
    def _sample_positions(self, pos: np.ndarray) -> np.ndarray:
        p0 = pos[self.edges[:, 0]]
        p1 = pos[self.edges[:, 1]]
        samples = p0[:, None, :] + self.sample_t[None, :, None] * (p1 - p0)[:, None, :]
        return samples.reshape(-1, 3)

    def _self_contact(self) -> None:
        samples = self._sample_positions(self.pos)
        dmax = 2.0 * self.radius
        pairs = cKDTree(samples).query_pairs(dmax, output_type="ndarray")
        if pairs.size == 0:
            return

        # Filter out samples belonging to the same noodle.
        mask = self.sample_noodle[pairs[:, 0]] != self.sample_noodle[pairs[:, 1]]
        pairs = pairs[mask]
        if pairs.size == 0:
            return

        a = pairs[:, 0]
        b = pairs[:, 1]
        dvec = self.pos[self.sample_to_node(b)] - self.pos[self.sample_to_node(a)]
        d = np.linalg.norm(dvec, axis=1)
        valid = d > 1e-12
        if not np.any(valid):
            return

        a = a[valid]
        b = b[valid]
        d = d[valid]
        dvec = dvec[valid]
        u = dvec / d[:, None]
        overlap = 0.5 * (dmax - d)[:, None] * u

        tau_a = self.sample_tau[a]
        tau_b = self.sample_tau[b]
        noodle_a = self.sample_noodle[a]
        noodle_b = self.sample_noodle[b]
        a0 = self.edges[noodle_a, 0]
        a1 = self.edges[noodle_a, 1]
        b0 = self.edges[noodle_b, 0]
        b1 = self.edges[noodle_b, 1]

        delta = np.zeros_like(self.pos)
        np.add.at(delta, a0, -(1.0 - tau_a)[:, None] * overlap)
        np.add.at(delta, a1, -tau_a[:, None] * overlap)
        np.add.at(delta, b0, (1.0 - tau_b)[:, None] * overlap)
        np.add.at(delta, b1, tau_b[:, None] * overlap)
        self.pos += delta

    def sample_to_node(self, sample_idx: np.ndarray) -> np.ndarray:
        tau = self.sample_tau[sample_idx]
        noodle = self.sample_noodle[sample_idx]
        # Use the closest endpoint as a representative node for distance calculation.
        return np.where(tau < 0.5, self.edges[noodle, 0], self.edges[noodle, 1])

    # --------------------------------------------------------------------- #
    # Trajectory frame construction
    # --------------------------------------------------------------------- #
    def _make_frame(self, time: float) -> StateFrame:
        ext_force = (self.mass[:, None] * self.gravity).astype(float)
        contact_signed = self._compute_contact_signed_distances()

        return StateFrame(
            time=float(time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes={"mass_quantity": self.mass.copy()},
            fields={
                "external_force": ext_force,
                "cumulative_mass_source": np.array([0.0], dtype=float),
                "contact_signed_distance": contact_signed,
            },
            scalars={"contact_length_scale": self.radius},
        )

    def _compute_contact_signed_distances(self) -> np.ndarray:
        sd, _ = self._bowl_distance_and_normal(self.pos)
        bowl_gaps = sd - self.radius
        gaps = [bowl_gaps]

        # Node-node near contacts (different noodles only).
        dmax = 3.0 * self.radius
        pairs = cKDTree(self.pos).query_pairs(dmax, output_type="ndarray")
        if pairs.size:
            nid = self.node_noodle
            diff = self.pos[pairs[:, 1]] - self.pos[pairs[:, 0]]
            d = np.linalg.norm(diff, axis=1)
            valid = (nid[pairs[:, 0]] != nid[pairs[:, 1]]) & (d > 0.0)
            gaps.append(d[valid] - 2.0 * self.radius)

        return np.concatenate(gaps).astype(float)
