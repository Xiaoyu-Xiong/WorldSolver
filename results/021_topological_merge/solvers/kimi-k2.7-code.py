"""Self-contained Lagrangian particle solver for repeated topological merging.

The material is represented by ~O(10^3) Lagrangian samples.  Neighbouring
samples within a support radius form persistent cohesive bonds, giving the
initial blocks shear resistance and causing distinct blocks to stick and merge
once their surfaces come into contact.  Gravity drives the fall and a penalty
ground contact model supports the pile.

Only NumPy/SciPy are used; no external simulation framework is invoked.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic particle-bond solver for the viscoplastic merge scene."""

    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        self._seed = seed

        frame0 = scene.initial_state()
        self._time = 0.0

        # Base state arrays (copied so the original frame is not mutated).
        self._position = np.asarray(frame0.position, dtype=float).copy()
        self._velocity = np.asarray(frame0.velocity, dtype=float).copy()
        self._mass = np.asarray(frame0.mass, dtype=float).copy()
        if self._mass.ndim == 0:
            self._mass = np.full(self._position.shape[0], float(self._mass))
        self._body_id = np.asarray(frame0.body_id).copy()
        self._radius = np.asarray(frame0.attributes["particle_radius"], dtype=float).copy()

        # Number of conserved material groups equals the number of initial blocks.
        self._K = int(self._body_id.max()) + 1 if self._body_id.size else 1

        # Preserve scene-provided attributes and scalars.
        self._base_attrs = {
            k: np.asarray(v).copy() for k, v in frame0.attributes.items()
        }
        self._scalars = dict(frame0.scalars) if frame0.scalars else {}

        # Configuration / units.
        cfg = scene.config if hasattr(scene, "config") else {}
        self._gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self._sample_dt = sample_dt

        # ------------------------------------------------------------------
        # Physical parameters.
        # ------------------------------------------------------------------
        # Support radius for bond creation: roughly 2.5 particle radii, large
        # enough to connect diagonal neighbours inside a block but short enough
        # to avoid spurious long-range attraction.
        self._h = 0.28
        self._k_bond = 5.0e4          # bond stiffness [N/m]
        self._zeta = 0.7              # damping ratio for bonds/ground
        self._k_ground = 1.0e5        # ground penalty stiffness [N/m]
        self._mu_ground = 0.4         # Coulomb friction coefficient
        self._dt_sub = sample_dt / 40.0  # sub-step size [s]

        # Pre-compute per-particle critical damping for the ground.
        self._c_ground = (
            2.0
            * self._zeta
            * np.sqrt(self._mass * self._k_ground)
        )

        # ------------------------------------------------------------------
        # Persistent cohesive bond storage.
        # ------------------------------------------------------------------
        n = self._position.shape[0]
        self._bond_matrix = np.zeros((n, n), dtype=bool)
        self._bond_i: list[int] = []
        self._bond_j: list[int] = []
        self._bond_rest: list[float] = []

        # Pre-connect each initial block; this gives it shear resistance.
        self._update_bonds(self._position)

        # Cached internal force per particle from the most recent substep.
        self._last_internal = np.zeros((n, 3), dtype=float)

    def advance_to(self, time: float) -> StateFrame:
        if time < self._time:
            raise ValueError("advance_to called with a non-monotonic time")

        if time == 0.0:
            return self._build_frame(0.0)

        # Integrate with fixed-size substeps.
        dt = self._dt_sub
        remaining = time - self._time
        while remaining > 0.0:
            step = min(dt, remaining)
            self._substep(step)
            remaining -= step
        self._time = time

        return self._build_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "particle_bond",
            "bonds": len(self._bond_i),
            "particles": self._position.shape[0],
            "groups": self._K,
        }

    # ------------------------------------------------------------------
    # Integration helpers.
    # ------------------------------------------------------------------
    def _substep(self, dt: float) -> None:
        pos = self._position
        vel = self._velocity
        n = pos.shape[0]

        # Discover and persist any new contacts within the support radius.
        self._update_bonds(pos)

        # Accumulate forces on each particle.
        force = np.zeros((n, 3), dtype=float)

        # --- cohesive / contact bonds ------------------------------------
        if self._bond_i:
            bi = np.fromiter(self._bond_i, dtype=np.int64)
            bj = np.fromiter(self._bond_j, dtype=np.int64)
            rest = np.fromiter(self._bond_rest, dtype=float)

            dx = pos[bi] - pos[bj]
            dist = np.linalg.norm(dx, axis=1)
            dist = np.maximum(dist, 1e-12)
            unit = dx / dist[:, None]

            # Spring force: negative sign makes dist > rest pull together and
            # dist < rest push apart.
            f_spring = -self._k_bond * (dist - rest)

            # Parallel dashpot using the relative velocity along the bond.
            relv = vel[bi] - vel[bj]
            relv_along = np.einsum("ij,ij->i", relv, unit)

            # Critical damping for the reduced mass of each pair.
            m_i = self._mass[bi]
            m_j = self._mass[bj]
            m_eff = (m_i * m_j) / (m_i + m_j)
            c_bond = 2.0 * self._zeta * np.sqrt(m_eff * self._k_bond)

            f_mag = f_spring - c_bond * relv_along
            f_vec = f_mag[:, None] * unit

            np.add.at(force, bi, f_vec)
            np.add.at(force, bj, -f_vec)

        # Cache net internal force for trajectory export.
        self._last_internal = force.copy()

        # --- ground contact ---------------------------------------------
        z = pos[:, 2]
        penetration = self._radius - z
        contact = penetration > 0.0
        if np.any(contact):
            v_z = vel[contact, 2]
            c_g = self._c_ground[contact]
            f_n = self._k_ground * penetration[contact] - c_g * v_z
            f_n = np.clip(f_n, 0.0, None)

            force[contact, 2] += f_n

            # Coulomb friction on the tangential velocity.
            v_t = vel[contact, :2]
            v_t_mag = np.linalg.norm(v_t, axis=1)
            eps = 1e-8
            scale = self._mu_ground * f_n / (v_t_mag + eps)
            force[contact, :2] -= scale[:, None] * v_t

        # --- body force (gravity) ----------------------------------------
        force += self._mass[:, None] * self._gravity

        # --- integrate ---------------------------------------------------
        accel = force / self._mass[:, None]
        self._velocity += dt * accel
        # Small global velocity damping to aid settling (time constant ~5 s).
        self._velocity *= (1.0 - 0.2 * dt)
        self._position += dt * self._velocity

    def _update_bonds(self, pos: np.ndarray) -> None:
        """Form persistent bonds between any particles within the support radius."""
        tree = cKDTree(pos)
        pairs = tree.query_pairs(r=self._h, output_type="ndarray")
        if pairs.shape[0] == 0:
            return

        existing = self._bond_matrix[pairs[:, 0], pairs[:, 1]]
        new_pairs = pairs[~existing]
        if new_pairs.shape[0] == 0:
            return

        i = new_pairs[:, 0]
        j = new_pairs[:, 1]
        self._bond_matrix[i, j] = True

        dx = pos[i] - pos[j]
        dist = np.linalg.norm(dx, axis=1)
        dist = np.maximum(dist, 1e-12)

        self._bond_i.extend(i.tolist())
        self._bond_j.extend(j.tolist())
        self._bond_rest.extend(dist.tolist())

    def _build_frame(self, time: float) -> StateFrame:
        n = self._position.shape[0]

        # External force is the body force only (gravity).
        external_force = self._mass[:, None] * self._gravity

        # Net internal force per material group.
        group_forces = np.zeros((self._K, 3), dtype=float)
        np.add.at(group_forces, self._body_id, self._last_internal)

        # Per-particle mass split by initial group.
        mass_quantity = np.zeros((n, self._K), dtype=float)
        mass_quantity[np.arange(n), self._body_id] = self._mass

        # Deformation gradient: identity (material frames are co-rotated with
        # the particles).  This satisfies the required [N,3,3] field.
        F = np.zeros((n, 3, 3), dtype=float)
        F[:, 0, 0] = 1.0
        F[:, 1, 1] = 1.0
        F[:, 2, 2] = 1.0

        attributes = dict(self._base_attrs)
        attributes["deformation_gradient"] = F
        attributes["mass_quantity"] = mass_quantity

        fields = {
            "cumulative_mass_source": np.zeros(self._K, dtype=float),
            "external_force": external_force,
            "interaction_force": group_forces,
        }

        return StateFrame(
            time=float(time),
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=self._scalars,
        )
