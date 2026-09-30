"""DEM-style solver for the ground-impact oobleck benchmark.

The solver time-steps two spherical granular bodies with gravity, floor contact,
and velocity-dependent (shear-thickening) inter-particle contacts.  The right
body is treated as oobleck (stiffer/less-damped at high impact speeds, so it
rebounds; softer/more-damped at low speeds, so it flows).  The left body is a
viscoplastic comparison that damps and spreads on first contact.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import Scene, Solver, StateFrame


class BenchmarkSolver(Solver):
    def __init__(self) -> None:
        self._initialized = False

    def initialize(self, scene: Scene, seed: int) -> None:
        self._scene = scene
        self._seed = seed

        # Load gravity from scene metadata so the solver matches config.yaml.
        self._gravity = np.asarray(scene.metadata()["gravity"], dtype=np.float64)

        # Initial particle state.
        frame = scene.initial_state()
        self._position = np.asarray(frame.position, dtype=np.float64)
        self._velocity = np.asarray(frame.velocity, dtype=np.float64)
        self._mass = np.asarray(frame.mass, dtype=np.float64)
        self._body_id = np.asarray(frame.body_id, dtype=np.int32)
        self._particle_radius = np.asarray(frame.attributes["particle_radius"], dtype=np.float64)
        self._rest_position = np.asarray(frame.attributes["rest_position"], dtype=np.float64)
        self._material_label = np.asarray(frame.attributes["material_label"], dtype=np.int32)

        # Time-stepping.
        self._time = float(frame.time)
        self._frame_dt = 0.02
        self._substeps = 100
        self._dt = self._frame_dt / self._substeps

        # Contact parameters.
        self._k_visco = 3.0e4
        self._c_visco = 200.0
        self._mu_visco = 0.3

        # Oobleck shear-thickens: high speed -> high stiffness, low damping;
        # low speed -> lower stiffness, high damping -> flow.
        self._k_oobleck = 1.5e5
        self._c_oobleck = 200.0
        self._mu_oobleck = 0.3
        self._alpha = 3.0
        self._beta = 3.0

        self._initialized = True

    def _contact_params(self, speed: np.ndarray, is_oobleck: np.ndarray) -> tuple:
        """Return velocity-dependent stiffness and damping for contact."""
        k_base = np.where(is_oobleck, self._k_oobleck, self._k_visco)
        c_base = np.where(is_oobleck, self._c_oobleck, self._c_visco)
        k = k_base * (1.0 + self._alpha * speed)
        c = c_base / (1.0 + self._beta * speed)
        return k, c

    def _compute_forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Compute gravity, floor, and inter-particle forces."""
        n = len(pos)
        r = float(self._particle_radius[0])
        forces = (self._mass[:, None] * self._gravity).astype(np.float64)

        # Floor contact (plane z=0).  Signed gap = z - r; negative means touching.
        signed = pos[:, 2] - r
        active = signed < 0.0
        if np.any(active):
            delta = -signed[active]
            v_z = vel[active, 2]
            is_oobleck = self._body_id[active] == 2
            s = np.abs(v_z)
            k, c = self._contact_params(s, is_oobleck)
            f_n = k * delta - c * v_z
            f_n = np.maximum(f_n, 0.0)

            # Coulomb friction tangential to the floor.
            v_xy = vel[active, :2]
            v_xy_mag = np.linalg.norm(v_xy, axis=1)
            mu = np.where(is_oobleck, self._mu_oobleck, self._mu_visco)
            dir_xy = np.zeros_like(v_xy)
            mask = v_xy_mag > 1e-12
            dir_xy[mask] = v_xy[mask] / v_xy_mag[mask][:, None]
            f_t = -mu[:, None] * f_n[:, None] * dir_xy

            f_floor = np.zeros((np.sum(active), 3))
            f_floor[:, 2] = f_n
            f_floor[:, :2] = f_t
            forces[active] += f_floor

        # Inter-particle contacts, processed body-by-body so the two spheres
        # only interact internally.
        for body in (1, 2):
            mask = self._body_id == body
            idx = np.where(mask)[0]
            if len(idx) < 2:
                continue

            p = pos[mask]
            tree = cKDTree(p)
            pairs = tree.query_pairs(r=2.0 * r, output_type="ndarray")
            if len(pairs) == 0:
                continue

            i = pairs[:, 0]
            j = pairs[:, 1]
            dx = p[j] - p[i]
            dist = np.linalg.norm(dx, axis=1)
            valid = dist > 1e-12
            if not np.all(valid):
                pairs = pairs[valid]
                if len(pairs) == 0:
                    continue
                i = pairs[:, 0]
                j = pairs[:, 1]
                dx = p[j] - p[i]
                dist = np.linalg.norm(dx, axis=1)

            n_norm = dx / dist[:, None]
            v_i = vel[idx[i]]
            v_j = vel[idx[j]]
            v_rel = v_j - v_i
            v_n = np.sum(v_rel * n_norm, axis=1)
            delta = 2.0 * r - dist

            is_oobleck = np.full(len(i), body == 2, dtype=bool)
            s = np.abs(v_n)
            k, c = self._contact_params(s, is_oobleck)
            f_mag = k * delta - c * v_n
            f_mag = np.maximum(f_mag, 0.0)

            # Tangential Coulomb friction.
            mu = self._mu_oobleck if body == 2 else self._mu_visco
            v_t = v_rel - v_n[:, None] * n_norm
            v_t_mag = np.linalg.norm(v_t, axis=1)
            dir_t = np.zeros_like(v_t)
            mask_t = v_t_mag > 1e-12
            dir_t[mask_t] = v_t[mask_t] / v_t_mag[mask_t][:, None]
            f_pair = f_mag[:, None] * n_norm - (mu * f_mag)[:, None] * dir_t

            # Accumulate equal-and-opposite forces.
            np.add.at(forces, idx[i], f_pair)
            np.add.at(forces, idx[j], -f_pair)

        return forces

    def _step(self) -> None:
        """Advance one sub-step using symplectic Euler."""
        m = self._mass[:, None]
        forces = self._compute_forces(self._position, self._velocity)
        self._velocity += (forces / m) * self._dt
        self._position += self._velocity * self._dt
        self._time += self._dt

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("Solver has not been initialized")
        if time + 1e-12 < self._time:
            raise ValueError("Cannot advance backwards in time")

        while self._time < time - 1e-12:
            self._step()

        return self._make_frame(time)

    def _make_frame(self, time: float) -> StateFrame:
        n = len(self._position)

        # Deformation gradient: identity baseline (material coordinates).
        deformation_gradient = np.tile(np.eye(3, dtype=np.float64), (n, 1, 1))
        # Per-sample mass contribution.
        mass_quantity = self._mass.copy()
        # Signed distance to contact floor for every sample.
        signed = self._position[:, 2] - float(self._particle_radius[0])
        # External force is gravity (excludes contact forces).
        external_force = self._mass[:, None] * self._gravity

        return StateFrame(
            time=float(time),
            position=self._position.copy(),
            velocity=self._velocity.copy(),
            mass=self._mass.copy(),
            body_id=self._body_id.copy(),
            attributes={
                "particle_radius": self._particle_radius.copy(),
                "rest_position": self._rest_position.copy(),
                "material_label": self._material_label.copy(),
                "deformation_gradient": deformation_gradient,
                "mass_quantity": mass_quantity,
            },
            fields={
                "contact_signed_distance": signed,
                "cumulative_mass_source": np.zeros(2, dtype=np.float64),
                "external_force": external_force,
            },
            scalars={
                "contact_length_scale": float(self._particle_radius[0] * 2.0),
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self._time,
            "substeps_per_frame": self._substeps,
        }
