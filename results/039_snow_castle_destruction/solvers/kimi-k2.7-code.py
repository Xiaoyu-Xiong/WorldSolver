"""Snow castle destruction solver using an explicit bonded-particle DEM.

The implementation steps a real physical time integrator:
- Snow particles are bonded to their initial neighbours to give the castle
  cohesive strength.
- Bonds break when stretched past a brittle threshold, giving fracture.
- Broken and unbonded particles interact through stiff repulsive contact
  with Coulomb friction.
- The orange striker ball is a rigid spherical body that collides with the
  snow and the floor.
- Gravity, floor contact, and damping provide the expected collapse and
  deposition behaviour.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.scene = None
        self.seed = 0
        self.config: dict = {}

        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float)
        self.dt = 0.0005
        self.time = 0.0

        self.position: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        self.mass: np.ndarray | None = None
        self.body_id: np.ndarray | None = None
        self.radius: np.ndarray | None = None

        self.is_snow: np.ndarray | None = None
        self.is_ball: np.ndarray | None = None

        # Bonded neighbour pairs (snow only), built from the initial layout.
        self.bond_i: np.ndarray = np.empty(0, dtype=int)
        self.bond_j: np.ndarray = np.empty(0, dtype=int)
        self.bond_d0: np.ndarray = np.empty(0, dtype=float)
        self.bond_alive: np.ndarray = np.empty(0, dtype=bool)

        # Deformation gradient placeholder.
        self.F: np.ndarray | None = None
        self.contact_scale = 0.075

        # Physical parameters.
        self.k_contact = 2.0e5
        self.c_contact = 2.5e2
        self.k_bond = 8.0e3
        self.d_break = 2.5e-2
        self.mu_floor = 0.65
        self.mu_snow = 0.40
        self.floor_z = 0.0

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = seed
        self.config = getattr(scene, "config", {})

        initial = scene.initial_state()
        self.gravity = np.asarray(
            self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float
        )

        self.position = np.asarray(initial.position, dtype=float).copy()
        self.velocity = np.asarray(initial.velocity, dtype=float).copy()
        self.mass = np.asarray(initial.mass, dtype=float).copy()
        self.body_id = np.asarray(initial.body_id, dtype=np.int32).copy()
        self.time = 0.0

        self.is_snow = self.body_id == 1
        self.is_ball = self.body_id == 2

        # Particle/contact radius.  Use half the lattice spacing so particles
        # are just touching at rest and the initial bonded configuration is
        # close to equilibrium.
        ic = self.config.get("initial_conditions", {})
        spacing = float(ic.get("snow_particle_spacing", 0.075))
        self.contact_scale = spacing
        r_snow = spacing * 0.5
        self.radius = np.full(len(self.position), r_snow, dtype=float)
        if "striker_radius" in ic:
            self.radius[self.is_ball] = float(ic["striker_radius"])

        # Build initial neighbour bonds for the snow particles.
        snow_idx = np.where(self.is_snow)[0]
        if len(snow_idx) > 1:
            snow_pos = self.position[snow_idx]
            tree = cKDTree(snow_pos)
            # A bond is any pair of snow particles within 2.2 lattice spacings
            # in the initial configuration.
            pairs_local = tree.query_pairs(
                r=2.2 * spacing, output_type="ndarray"
            )
            self.bond_i = snow_idx[pairs_local[:, 0]]
            self.bond_j = snow_idx[pairs_local[:, 1]]
            self.bond_d0 = np.linalg.norm(
                self.position[self.bond_j] - self.position[self.bond_i], axis=1
            )
            self.bond_alive = np.ones(len(self.bond_i), dtype=bool)
        else:
            self.bond_i = np.empty(0, dtype=int)
            self.bond_j = np.empty(0, dtype=int)
            self.bond_d0 = np.empty(0, dtype=float)
            self.bond_alive = np.empty(0, dtype=bool)

        # Deformation gradient: identity for every particle.
        self.F = np.tile(np.eye(3, dtype=float)[None], (len(self.position), 1, 1))

    def _floor_contact(self, force: np.ndarray) -> np.ndarray:
        """Add floor contact forces with Coulomb friction."""
        z = self.position[:, 2]
        r = self.radius
        pen = r - z  # positive when particle centre is below its radius -> contact
        mask = pen > 0.0
        if not np.any(mask):
            return force

        idx = np.where(mask)[0]
        delta = pen[idx]
        v = self.velocity[idx]
        v_n = v[:, 2]

        f_n_mag = self.k_contact * delta - self.c_contact * np.clip(v_n, -1.0, 1.0)
        f_n_mag = np.maximum(f_n_mag, 0.0)

        # Tangential velocity in the floor plane.
        v_t = v.copy()
        v_t[:, 2] = 0.0
        v_t_mag = np.linalg.norm(v_t, axis=1)

        f_t = np.zeros_like(v_t)
        nonzero = v_t_mag > 1e-8
        if np.any(nonzero):
            f_t[nonzero] = (
                -self.mu_floor
                * f_n_mag[nonzero, None]
                * v_t[nonzero]
                / v_t_mag[nonzero, None]
            )

        force[idx, 2] += f_n_mag
        force[idx] += f_t
        return force

    def _ball_snow_contact(self, force: np.ndarray) -> np.ndarray:
        """Contact between the striker ball and all snow particles."""
        ball_idx_arr = np.where(self.is_ball)[0]
        snow_idx = np.where(self.is_snow)[0]
        if len(ball_idx_arr) == 0 or len(snow_idx) == 0:
            return force

        b = ball_idx_arr[0]
        ball_pos = self.position[b]
        ball_r = self.radius[b]
        ball_v = self.velocity[b]

        dvec = self.position[snow_idx] - ball_pos[None, :]
        d = np.linalg.norm(dvec, axis=1)
        rsum = ball_r + self.radius[snow_idx]
        mask = d < rsum
        if not np.any(mask):
            return force

        s = snow_idx[mask]
        dn = d[mask][:, None]
        n = dvec[mask] / dn

        delta = (rsum - d)[mask]
        v_rel = self.velocity[s] - ball_v[None, :]
        v_n = np.sum(v_rel * n, axis=1)

        f_n_mag = self.k_contact * delta - self.c_contact * v_n
        f_n_mag = np.maximum(f_n_mag, 0.0)
        f_n = f_n_mag[:, None] * n

        v_t = v_rel - v_n[:, None] * n
        v_t_mag = np.linalg.norm(v_t, axis=1)
        f_t = np.zeros_like(v_t)
        nonzero = v_t_mag > 1e-8
        if np.any(nonzero):
            f_t[nonzero] = (
                -self.mu_snow
                * f_n_mag[nonzero, None]
                * v_t[nonzero]
                / v_t_mag[nonzero, None]
            )

        f_b_s = f_n + f_t
        force[s] += f_b_s
        force[b] -= np.sum(f_b_s, axis=0)
        return force

    def _snow_snow_contact(self, force: np.ndarray) -> np.ndarray:
        """Repulsive/frictional contact between currently overlapping snow particles."""
        snow_idx = np.where(self.is_snow)[0]
        if len(snow_idx) < 2:
            return force

        snow_pos = self.position[snow_idx]
        tree = cKDTree(snow_pos)
        max_r = float(self.radius[self.is_snow].max())
        pairs_local = tree.query_pairs(r=2.0 * max_r, output_type="ndarray")
        if len(pairs_local) == 0:
            return force

        i = snow_idx[pairs_local[:, 0]]
        j = snow_idx[pairs_local[:, 1]]

        dvec = self.position[j] - self.position[i]
        d = np.linalg.norm(dvec, axis=1)
        rsum = self.radius[i] + self.radius[j]
        mask = d < rsum
        if not np.any(mask):
            return force

        i = i[mask]
        j = j[mask]
        dvec = dvec[mask]
        d = d[mask]
        rsum = rsum[mask]

        n = dvec / d[:, None]
        delta = rsum - d

        v_i = self.velocity[i]
        v_j = self.velocity[j]
        v_rel = v_j - v_i
        v_n = np.sum(v_rel * n, axis=1)

        f_n_mag = self.k_contact * delta - self.c_contact * v_n
        f_n_mag = np.maximum(f_n_mag, 0.0)
        f_n = f_n_mag[:, None] * n

        v_t = v_rel - v_n[:, None] * n
        v_t_mag = np.linalg.norm(v_t, axis=1)
        f_t = np.zeros_like(v_t)
        nonzero = v_t_mag > 1e-8
        if np.any(nonzero):
            f_t[nonzero] = (
                -self.mu_snow
                * f_n_mag[nonzero, None]
                * v_t[nonzero]
                / v_t_mag[nonzero, None]
            )

        f_pair = f_n + f_t
        np.add.at(force, i, f_pair)
        np.add.at(force, j, -f_pair)
        return force

    def _bond_forces(self, force: np.ndarray) -> np.ndarray:
        """Cohesive bonds between initial snow neighbours; break under tension."""
        if len(self.bond_i) == 0:
            return force

        i = self.bond_i
        j = self.bond_j

        dvec = self.position[j] - self.position[i]
        d = np.linalg.norm(dvec, axis=1)
        n = dvec / d[:, None]

        delta = d - self.bond_d0
        f_mag = self.k_bond * delta

        # Bonds only act in tension and break past the brittle limit.
        new_broken = (delta > self.d_break) & self.bond_alive
        self.bond_alive[new_broken] = False

        f_mag[delta <= 0.0] = 0.0
        f_mag[~self.bond_alive] = 0.0

        f_vec = f_mag[:, None] * n
        np.add.at(force, i, f_vec)
        np.add.at(force, j, -f_vec)
        return force

    def _compute_forces(self) -> np.ndarray:
        """Aggregate gravity, floor, ball, snow-snow and bond forces."""
        n = len(self.position)
        force = np.zeros((n, 3), dtype=float)

        # Gravity (also fills the external_force field).
        force += self.mass[:, None] * self.gravity[None, :]

        # Contact and cohesive forces.
        force = self._floor_contact(force)
        force = self._ball_snow_contact(force)
        force = self._snow_snow_contact(force)
        force = self._bond_forces(force)

        return force

    def _step(self, dt: float) -> None:
        """Symplectic Euler substep."""
        force = self._compute_forces()
        self.velocity += dt * force / self.mass[:, None]
        self.position += dt * self.velocity

    def _contact_signed_distances(self) -> np.ndarray:
        """Collect signed gaps for all active contacts; negative means penetration."""
        sdistances = []

        # Floor contacts.
        pen_floor = self.radius - self.position[:, 2]
        mask_floor = pen_floor > 0.0
        if np.any(mask_floor):
            sdistances.append(-pen_floor[mask_floor])

        # Ball-snow contacts.
        ball_idx_arr = np.where(self.is_ball)[0]
        snow_idx = np.where(self.is_snow)[0]
        if len(ball_idx_arr) > 0 and len(snow_idx) > 0:
            b = ball_idx_arr[0]
            dvec = self.position[snow_idx] - self.position[b]
            d = np.linalg.norm(dvec, axis=1)
            rsum = self.radius[b] + self.radius[snow_idx]
            sdistances.append(d - rsum)

        # Snow-snow contacts.
        if len(snow_idx) > 1:
            snow_pos = self.position[snow_idx]
            tree = cKDTree(snow_pos)
            max_r = float(self.radius[self.is_snow].max())
            pairs_local = tree.query_pairs(r=2.0 * max_r, output_type="ndarray")
            if len(pairs_local) > 0:
                i = snow_idx[pairs_local[:, 0]]
                j = snow_idx[pairs_local[:, 1]]
                dvec = self.position[j] - self.position[i]
                d = np.linalg.norm(dvec, axis=1)
                rsum = self.radius[i] + self.radius[j]
                sdistances.append(d - rsum)

        if sdistances:
            return np.concatenate(sdistances)
        return np.zeros(0, dtype=float)

    def _make_frame(self, time: float) -> StateFrame:
        external_force = self.mass[:, None] * self.gravity[None, :]
        contact_signed = self._contact_signed_distances()

        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": self.F.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "contact_signed_distance": contact_signed,
                "cumulative_mass_source": np.array([0.0], dtype=float),
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": float(self.contact_scale),
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        if self.time < time - 1e-12:
            # Clamp to a non-negative substep count.
            remaining = time - self.time
            n_steps = max(1, int(np.ceil(remaining / self.dt)))
            sub_dt = remaining / n_steps
            for _ in range(n_steps):
                self._step(sub_dt)
                self.time += sub_dt
        return self._make_frame(time)

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "num_particles": len(self.position) if self.position is not None else 0,
            "num_snow": int(np.sum(self.is_snow)) if self.is_snow is not None else 0,
            "num_bonds": int(np.sum(self.bond_alive)) if self.bond_alive is not None else 0,
        }
