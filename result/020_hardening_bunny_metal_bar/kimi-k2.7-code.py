"""Viscoplastic point-cloud solver for the hardening bunny metal-bar scene.

The bunny is represented by the vertices in ``assets/subject.npz``.  A peridynamics-style
bond network is built from the rest configuration.  Each bond stores a plastic rest length and
a work-hardening variable, giving strongly deformed regions a stiffer elastic response.
The equations are time-stepped explicitly with gravity, internal bond forces, damping and
contact/penalty collisions against the ground and the horizontal metal bar.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit time-stepping solver for a viscoplastic bunny colliding with a metal bar."""

    # --------------------------------------------------------------------- #
    # Solver contract
    # --------------------------------------------------------------------- #
    def initialize(self, scene, seed: int) -> None:
        self._config = scene.config
        cfg = self._config

        # Physical parameters from the frozen configuration.
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.contact_scale = float(cfg.get("contact_length_scale", 0.04))
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.bar_z = float(cfg.get("bar_z", 0.66))
        self.bar_radius = float(cfg.get("bar_radius", 0.085))
        self.restitution = float(cfg.get("restitution", 0.28))
        self.friction = float(cfg.get("friction", 0.55))
        self.stiffness = float(cfg.get("stiffness", 18.0))
        self.hardening_rate = float(cfg.get("hardening_rate", 1.8))
        self.damping = float(cfg.get("damping", 1.4))

        # Time discretisation: fixed number of substeps per sample interval.
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.substeps = 40
        self.dt = self.sample_dt / self.substeps

        # State at t = 0.
        frame0 = scene.initial_state()
        self.x = np.asarray(frame0.position, dtype=float).copy()
        self.v = np.asarray(frame0.velocity, dtype=float).copy()
        self.m = np.asarray(frame0.mass, dtype=float)
        self.N = self.x.shape[0]
        self.X = self.x.copy()

        # If there are no particles, the solver can still produce a trivial frame.
        if self.N == 0:
            self._time = 0.0
            self._contact_acc = np.zeros(3, dtype=float)
            self._contact_steps = 0
            return

        # Try to extract a tetrahedral mesh; if one is present it gives a more accurate
        # volume estimate for the particle volumes used by the bond model.
        self._has_tets = False
        self._tet_volumes = None
        self._load_tet_volume_estimate()

        # Build the peridynamic bond network from the rest shape.
        self._build_bonds()

        # Plastic state per bond: plastic rest length and accumulated plastic work.
        self.Lp = self.rest_lengths.copy()
        self.alpha = np.zeros(self.n_bonds, dtype=float)

        # Bookkeeping for contact-force reporting.
        self._time = 0.0
        self._contact_acc = np.zeros(3, dtype=float)
        self._contact_steps = 0

    def advance_to(self, time: float) -> StateFrame:
        if time < self._time:
            time = self._time

        if self.N == 0:
            self._time = time
            return self._build_frame(time)

        # Integrate with a fixed substep so the sample times are hit exactly.
        n = max(1, int(round((time - self._time) / self.dt))) if time > self._time else 0
        self._contact_acc = np.zeros(3, dtype=float)
        self._contact_steps = 0
        for _ in range(n):
            self._step(self.dt)
            self._time += self.dt

        return self._build_frame(self._time)

    def diagnostics(self) -> dict[str, object]:
        return {}

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _load_tet_volume_estimate(self) -> None:
        """Load a tetrahedral topology if it exists and precompute element volumes."""
        try:
            from pathlib import Path

            asset = Path("assets/subject.npz")
            data = np.load(asset)
            tets = None
            for key in ("elements", "tets", "indices"):
                if key in data:
                    arr = np.asarray(data[key], dtype=int)
                    if arr.ndim == 2 and arr.shape[1] == 4:
                        tets = arr
                        break
            data.close()
            if tets is not None and len(tets) > 0:
                X = self.X
                i0 = tets[:, 0]
                i1 = tets[:, 1]
                i2 = tets[:, 2]
                i3 = tets[:, 3]
                D = np.stack([X[i1] - X[i0], X[i2] - X[i0], X[i3] - X[i0]], axis=-1)
                vol = np.abs(np.linalg.det(D)) / 6.0
                self._tet_volumes = vol
                self._has_tets = True
        except Exception:
            self._has_tets = False
            self._tet_volumes = None

    def _build_bonds(self) -> None:
        """Construct the peridynamics-style bond network from the rest positions."""
        X = self.X
        N = self.N

        # Characteristic length: average nearest-neighbour distance scaled by 2.5.
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(X)
            dist, _ = tree.query(X, k=2)
            avg_nn = float(np.mean(dist[:, 1]))
            self.h = 2.5 * max(avg_nn, 1e-6)
            pairs = tree.query_pairs(self.h, output_type="ndarray")
        except Exception:  # pragma: no cover - fallback if scipy is missing
            d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
            d2[np.arange(N), np.arange(N)] = np.inf
            pairs = np.argwhere(d2 < self.h**2)
            pairs = pairs[pairs[:, 0] < pairs[:, 1]]

        if pairs.size == 0 or pairs.shape[0] == 0:
            # Degenerate single-particle scene; create no bonds.
            self.bonds_i = np.zeros(0, dtype=int)
            self.bonds_j = np.zeros(0, dtype=int)
            self.rest_lengths = np.zeros(0, dtype=float)
            self.n_bonds = 0
            self.V = self._particle_volumes(np.zeros(0, dtype=int))
            self.adj = [np.zeros(0, dtype=int) for _ in range(N)]
            self.adj_dX = [np.zeros((0, 3), dtype=float) for _ in range(N)]
            return

        i = pairs[:, 0]
        j = pairs[:, 1]
        dX = X[j] - X[i]
        L0 = np.linalg.norm(dX, axis=1)
        # Remove zero-length bonds (coincident vertices).
        valid = L0 > 1e-12
        i = i[valid]
        j = j[valid]
        L0 = L0[valid]

        self.bonds_i = i
        self.bonds_j = j
        self.rest_lengths = L0
        self.n_bonds = len(L0)

        # Per-particle volume estimate used for the bond stiffness.
        self.V = self._particle_volumes(i, j)

        # Adjacency for per-particle deformation-gradient fitting.
        adj = [[] for _ in range(N)]
        for a, b in zip(i, j):
            adj[a].append(b)
            adj[b].append(a)
        self.adj = [np.asarray(a, dtype=int) for a in adj]
        self.adj_dX = [X[self.adj[k]] - X[k] for k in range(N)]

    def _particle_volumes(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Estimate a per-particle volume from local neighbour counts within the cutoff."""
        X = self.X
        N = self.N
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(X)
            counts = np.asarray([len(pts) for pts in tree.query_ball_point(X, self.h)], dtype=float)
        except Exception:  # pragma: no cover
            d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
            counts = np.count_nonzero(d2 < self.h**2, axis=1).astype(float) - 1.0
        counts = np.maximum(counts, 1.0)
        sphere_vol = (4.0 / 3.0) * np.pi * self.h**3
        vol = sphere_vol / counts

        # If a tetrahedral mesh was available, normalise by the actual total volume.
        if self._has_tets and self._tet_volumes is not None:
            total_tet = float(np.sum(self._tet_volumes))
            total_est = float(np.sum(vol))
            if total_est > 0.0:
                vol *= total_tet / total_est
        return vol

    def _bond_forces(self, dt: float) -> np.ndarray:
        """Compute internal bond forces and update plastic state."""
        f = np.zeros((self.N, 3), dtype=float)
        if self.n_bonds == 0:
            return f

        i = self.bonds_i
        j = self.bonds_j
        dx = self.x[j] - self.x[i]
        L = np.linalg.norm(dx, axis=1)
        # Protect against degenerate configurations.
        L = np.maximum(L, 1e-12)
        n = dx / L[:, None]

        L0 = self.rest_lengths
        Lp = self.Lp
        V = self.V

        # Effective stiffness: base Lamé-like modulus with work hardening.
        # k has units N/m; displacement times k gives Newtons.
        k = self.stiffness * (V[i] + V[j]) / (2.0 * L0**2)
        k = k * (1.0 + self.hardening_rate * self.alpha)

        # Elastic part of the bond deformation.
        e = L - Lp
        fmag = k * e
        fvec = fmag[:, None] * n

        np.add.at(f, i, fvec)
        np.add.at(f, j, -fvec)

        # Viscoplastic flow: absorb elastic strain into the plastic rest length.
        # The flow rate is modulated by the hardening variable so that strongly
        # deformed bonds become stiffer and jiggle more.
        yield_strain = 0.01
        plastic_rate = 5.0
        active = np.abs(e) > yield_strain * L0
        flow = plastic_rate * np.maximum(np.abs(e) - yield_strain * L0, 0.0) * dt
        self.Lp += np.sign(e) * flow
        self.alpha += np.abs(flow) / np.maximum(L0, 1e-12)

        return f

    def _resolve_ground(self, dt: float) -> None:
        mask = self.x[:, 2] < self.ground_z
        if not np.any(mask):
            return
        idx = np.nonzero(mask)[0]
        x = self.x[idx]
        v = self.v[idx]
        m = self.m[idx]

        # Penalty position correction.
        x[:, 2] = self.ground_z

        # Normal impulse.
        v_before = v.copy()
        v_n = v[:, 2].copy()
        v_n_new = np.where(v_n < 0.0, -self.restitution * v_n, v_n)
        v[:, 2] = v_n_new

        # Coulomb friction on tangential velocity.
        v_t = v.copy()
        v_t[:, 2] = 0.0
        v_t_mag = np.linalg.norm(v_t, axis=1)
        max_t = self.friction * np.maximum(v_n_new, 0.0)
        scale = np.where(v_t_mag > 1e-12, np.minimum(max_t / v_t_mag, 1.0), 1.0)
        v_t = (v_t.T * scale).T
        v[:, :2] = v_t[:, :2]

        # Record contact force = total impulse / dt.
        impulse = (m[:, None] * (v - v_before)).sum(axis=0)
        self._contact_acc += impulse / dt
        self._contact_steps += 1

        self.x[idx] = x
        self.v[idx] = v

    def _resolve_bar(self, dt: float) -> None:
        dx = self.x[:, 0]
        dz = self.x[:, 2] - self.bar_z
        r = np.sqrt(dx * dx + dz * dz)
        mask = r < self.bar_radius
        if not np.any(mask):
            return
        idx = np.nonzero(mask)[0]
        x = self.x[idx]
        v = self.v[idx]
        m = self.m[idx]
        r_i = r[idx]
        dx_i = dx[idx]
        dz_i = dz[idx]

        # Normal vector from cylinder axis to the particle.
        normal = np.zeros((len(idx), 3), dtype=float)
        denom = np.maximum(r_i, 1e-12)
        normal[:, 0] = dx_i / denom
        normal[:, 2] = dz_i / denom

        # Position correction.
        x += normal * (self.bar_radius - r_i)[:, None]

        v_before = v.copy()
        v_n = np.einsum("ij,ij->i", v, normal)
        v_n_new = np.where(v_n < 0.0, -self.restitution * v_n, v_n)
        v += (v_n_new - v_n)[:, None] * normal

        # Tangential friction.
        v_n_after_vec = (v_n_new)[:, None] * normal
        v_t = v - v_n_after_vec
        v_t_mag = np.linalg.norm(v_t, axis=1)
        max_t = self.friction * np.maximum(v_n_new, 0.0)
        scale = np.where(v_t_mag > 1e-12, np.minimum(max_t / v_t_mag, 1.0), 1.0)
        v_t = (v_t.T * scale).T
        v = v_n_after_vec + v_t

        impulse = (m[:, None] * (v - v_before)).sum(axis=0)
        self._contact_acc += impulse / dt
        self._contact_steps += 1

        self.x[idx] = x
        self.v[idx] = v

    def _step(self, dt: float) -> None:
        # Gravity.
        f = self.m[:, None] * self.gravity[None, :]
        # Internal bond forces (updates plastic state).
        f += self._bond_forces(dt)

        # Acceleration and explicit velocity update.
        a = f / self.m[:, None]
        self.v += dt * a

        # Damping applied to internal (relative) velocities so that the global
        # rigid-body motion (fall, bounce) is not artificially suppressed.
        if self.N > 1:
            v_mean = self.v.mean(axis=0)
            v_rel = self.v - v_mean
            v_rel *= 1.0 / (1.0 + self.damping * dt)
            self.v = v_mean + v_rel

        # Position update.
        self.x += dt * self.v

        # Contact resolution and reaction-force bookkeeping.
        self._resolve_ground(dt)
        self._resolve_bar(dt)

    def _deformation_gradient(self) -> np.ndarray:
        """Compute a per-particle deformation gradient from local affine fits."""
        F = np.tile(np.eye(3, dtype=float), (self.N, 1, 1))
        for i in range(self.N):
            neigh = self.adj[i]
            if len(neigh) < 3:
                continue
            dX = self.adj_dX[i]  # rest offsets of neighbours
            dx = self.x[neigh] - self.x[i]
            A = dX.T @ dX
            B = dX.T @ dx
            try:
                A_inv = np.linalg.inv(A + 1e-8 * np.eye(3))
                F[i] = A_inv @ B
            except np.linalg.LinAlgError:
                continue
        return F

    def _signed_distance(self) -> np.ndarray:
        """Per-particle signed distance to the closest obstacle (ground or bar)."""
        sd_ground = self.x[:, 2] - self.ground_z
        dx = self.x[:, 0]
        dz = self.x[:, 2] - self.bar_z
        sd_bar = np.sqrt(dx * dx + dz * dz) - self.bar_radius
        return np.minimum(sd_ground, sd_bar)

    def _build_frame(self, time: float) -> StateFrame:
        if self.N == 0:
            return StateFrame(
                time=time,
                position=np.zeros((0, 3), dtype=float),
                velocity=np.zeros((0, 3), dtype=float),
                mass=np.zeros(0, dtype=float),
                body_id=np.zeros(0, dtype=np.int64),
                attributes={
                    "mass_quantity": np.zeros(0, dtype=float),
                    "deformation_gradient": np.zeros((0, 3, 3), dtype=float),
                },
                fields={
                    "cumulative_mass_source": np.zeros(1, dtype=float),
                    "contact_signed_distance": np.zeros(0, dtype=float),
                    "interaction_force": np.zeros((1, 3), dtype=float),
                },
                scalars={"contact_length_scale": self.contact_scale},
            )

        F = self._deformation_gradient()
        sd = self._signed_distance()
        if self._contact_steps > 0:
            interaction = (self._contact_acc / self._contact_steps).reshape(1, 3)
        else:
            interaction = np.zeros((1, 3), dtype=float)

        frame = StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=np.zeros(self.N, dtype=np.int64),
            attributes={
                "mass_quantity": self.m.copy(),
                "deformation_gradient": F,
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "contact_signed_distance": sd,
                "interaction_force": interaction,
            },
            scalars={"contact_length_scale": self.contact_scale},
        )
        frame.validate()
        return frame
