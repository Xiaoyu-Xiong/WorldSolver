"""2D-thin-slab DEM solver for the hourglass scene.

The hourglass is treated as a kinematic rigid container.  Sand grains are
spherical particles with normal contact springs, damping and regularised
Coulomb friction.  The container boundary is the inner surface of the
axisymmetric hourglass profile, plus a thin-slab constraint in the y-direction.
No genesis/gs code is used; integration is explicit with a fixed sub-step.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import Solver, StateFrame


class BenchmarkSolver(Solver):
    def __init__(self) -> None:
        self._initialized = False

    def initialize(self, scene, seed: int | None = None) -> None:
        import yaml

        # Load authoritative configuration and geometry from the workspace.
        config_path = Path("config.yaml")
        geom_path = Path("scene_geometry.yaml")
        self.config = yaml.safe_load(config_path.read_text())
        self.geom = yaml.safe_load(geom_path.read_text())

        # Initial state is provided deterministically by the scene.
        frame0 = scene.initial_state()
        self.pos = np.asarray(frame0.position, dtype=float).copy()
        self.vel = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.n = self.pos.shape[0]

        # Physical parameters from the public config.
        ic = self.config["initial_conditions"]
        self.radius = float(ic["particle_radius"])
        self.density = float(ic["particle_density"])
        self.slab_half_depth = float(self.config["boundary_conditions"]["slab_half_depth"])

        # Event schedule (release, flips).
        self.events = list(self.config.get("events", []))
        self.gravity = np.asarray(self.config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # Hourglass profile.
        self._parse_hourglass()

        # Contact parameters.
        # Effective mass for damping calibration.
        m_eff = float(np.median(self.mass)) if self.n > 0 else 1.0
        self.kp = 2.0e4          # particle-particle normal stiffness
        self.kw = 2.0e4          # wall normal stiffness
        self.damping_ratio = 0.15
        self.friction = 0.5
        self.kt = 0.5 * self.kp  # tangential regularisation
        self.cp = 2.0 * np.sqrt(m_eff * self.kp) * self.damping_ratio
        self.cw = self.cp

        self.sub_dt = 2.0e-4
        self.current_time = 0.0
        self.substep_count = 0
        self._initialized = True

    def _parse_hourglass(self) -> None:
        hg = next(o for o in self.geom["objects"] if o["id"] == "hourglass_shell")
        profile = np.asarray(hg["geometry"]["inner_profile"], dtype=float)
        r_prof = profile[:, 0]
        z_prof = profile[:, 1]
        translation = np.asarray(hg["transform"]["translation"], dtype=float)
        # Use the geometric centre of the profile as the rotation pivot, so the
        # hourglass flips in place about the y-axis.
        z_center = 0.5 * (z_prof.min() + z_prof.max())
        self.center = translation.copy()
        self.center[2] += z_center
        self.profile_z = z_prof - z_center
        self.profile_r = r_prof
        # Segment derivatives for normal / distance calculations.
        dz = np.diff(self.profile_z)
        dr = np.diff(self.profile_r)
        self.seg_rprime = dr / dz
        self.z_min = float(self.profile_z[0])
        self.z_max = float(self.profile_z[-1])

    def _angle_and_omega(self, t: float) -> tuple[float, float]:
        """Return the container rotation angle (rad) and angular velocity (rad/s)."""
        angle = 0.0
        omega = 0.0
        for ev in self.events:
            if ev.get("type") != "flip":
                continue
            t0 = float(ev["time"])
            dur = float(ev["duration"])
            ang_deg = float(ev["angle_degrees"])
            t1 = t0 + dur
            if t <= t0:
                continue
            if t >= t1:
                angle += np.radians(ang_deg)
            else:
                frac = (t - t0) / dur
                delta = np.radians(ang_deg)
                angle += delta * frac
                omega = delta / dur
        return float(angle), float(omega)

    def _local_xz(self, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the local x- and z-coordinates of all particles in the
        hourglass frame for the given rotation angle about the y-axis."""
        c = np.cos(theta)
        s = np.sin(theta)
        dx = self.pos[:, 0] - self.center[0]
        dz = self.pos[:, 2] - self.center[2]
        x_local = dx * c + dz * s
        z_local = -dx * s + dz * c
        return x_local, z_local

    def _hourglass_signed_center(self, x_local: np.ndarray, z_local: np.ndarray) -> tuple:
        """Signed distance of the particle *centres* to the hourglass wall.

        Returns
        -------
        signed_center : np.ndarray, positive outside, negative inside.
        r_vals, rprime : interpolated profile radius and derivative.
        """
        # Find segment for each z.
        z_arr = self.profile_z
        idx = np.searchsorted(z_arr, z_local, side="right") - 1
        idx = np.clip(idx, 0, len(z_arr) - 2)
        dz = z_arr[idx + 1] - z_arr[idx]
        dr = self.profile_r[idx + 1] - self.profile_r[idx]
        rprime = dr / dz
        r_vals = self.profile_r[idx] + rprime * (z_local - z_arr[idx])

        absx = np.abs(x_local)
        denom = np.sqrt(1.0 + rprime * rprime)
        d_side = (absx - r_vals) / denom
        d_bottom = self.z_min - z_local
        d_top = z_local - self.z_max
        signed_center = np.maximum(d_side, np.maximum(d_bottom, d_top))
        return signed_center, r_vals, rprime, d_side, d_bottom, d_top

    def _wall_and_slab_forces(self, theta: float, omega: float, acc: np.ndarray) -> None:
        pos = self.pos
        vel = self.vel
        n = self.n
        x_local, z_local = self._local_xz(theta)
        signed_center, r_vals, rprime, d_side, d_bottom, d_top = self._hourglass_signed_center(
            x_local, z_local
        )

        # Active boundary: the one with the largest signed distance.
        is_side = (d_side >= d_bottom) & (d_side >= d_top)
        is_top = (~is_side) & (d_top >= d_bottom)
        is_bottom = (~is_side) & (~is_top)

        c = np.cos(theta)
        s = np.sin(theta)

        # Local outward normals.
        n_local = np.zeros((n, 3), dtype=float)
        # side
        mask = is_side
        sign_x = np.sign(x_local[mask] + 1e-12)
        den = np.sqrt(1.0 + rprime[mask] ** 2)
        n_local[mask, 0] = sign_x / den
        n_local[mask, 2] = -rprime[mask] / den
        # top/bottom caps
        n_local[is_top] = np.array([0.0, 0.0, 1.0])
        n_local[is_bottom] = np.array([0.0, 0.0, -1.0])

        # Rotate normals to world coordinates.
        n_world = np.empty_like(n_local)
        n_world[:, 0] = n_local[:, 0] * c + n_local[:, 2] * s
        n_world[:, 2] = -n_local[:, 0] * s + n_local[:, 2] * c
        n_world[:, 1] = n_local[:, 1]

        # Wall velocity from container rotation about y.
        dx = pos[:, 0] - self.center[0]
        dz = pos[:, 2] - self.center[2]
        v_wall = np.zeros_like(pos)
        v_wall[:, 0] = omega * dz
        v_wall[:, 2] = -omega * dx

        # x-z hourglass wall contact.
        phi_xz = -signed_center  # centre distance, positive inside.
        gap_xz = phi_xz - self.radius
        mask_xz = gap_xz < 0.0
        if np.any(mask_xz):
            rel = vel[mask_xz] - v_wall[mask_xz]
            vn = np.einsum("ij,ij->i", rel, n_world[mask_xz])
            # Normal force magnitude (positive = compressive)
            fn_load = -self.kw * gap_xz[mask_xz]
            fn_mag = fn_load + self.cw * np.maximum(0.0, vn)
            force = (fn_mag[:, None] * n_world[mask_xz]) / self.mass[mask_xz][:, None]
            acc[mask_xz] += force

            # Tangential friction.
            v_n_vec = (vn[:, None]) * n_world[mask_xz]
            v_t = rel - v_n_vec
            v_t_mag = np.linalg.norm(v_t, axis=1)
            f_t_mag = np.minimum(self.friction * fn_load, self.kt * v_t_mag)
            v_t_dir = -v_t / (v_t_mag[:, None] + 1e-12)
            f_t = (f_t_mag[:, None] * v_t_dir) / self.mass[mask_xz][:, None]
            acc[mask_xz] += f_t

        # Thin-slab wall in y.
        absy = np.abs(pos[:, 1])
        phi_y = self.slab_half_depth - absy
        gap_y = phi_y - self.radius
        mask_y = gap_y < 0.0
        if np.any(mask_y):
            n_y = np.zeros((n, 3), dtype=float)
            n_y[mask_y, 1] = np.sign(pos[mask_y, 1] + 1e-12)
            rel = vel[mask_y]  # wall stationary in y
            vn = np.einsum("ij,ij->i", rel, n_y[mask_y])
            fn_load = -self.kw * gap_y[mask_y]
            fn_mag = fn_load + self.cw * np.maximum(0.0, vn)
            force = (fn_mag[:, None] * n_y[mask_y]) / self.mass[mask_y][:, None]
            acc[mask_y] += force

            v_n_vec = (vn[:, None]) * n_y[mask_y]
            v_t = rel - v_n_vec
            v_t_mag = np.linalg.norm(v_t, axis=1)
            f_t_mag = np.minimum(self.friction * fn_load, self.kt * v_t_mag)
            v_t_dir = -v_t / (v_t_mag[:, None] + 1e-12)
            f_t = (f_t_mag[:, None] * v_t_dir) / self.mass[mask_y][:, None]
            acc[mask_y] += f_t

    def _particle_particle_forces(self, acc: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        tree = cKDTree(self.pos)
        cutoff = 2.0 * self.radius
        pairs = tree.sparse_distance_matrix(tree, max_distance=cutoff, output_type="ndarray")
        if pairs.size == 0:
            return
        i = pairs["i"]
        j = pairs["j"]
        d = pairs["v"]
        # Keep only unique i<j pairs.
        mask = i < j
        i = i[mask]
        j = j[mask]
        d = d[mask]
        if i.size == 0:
            return

        pi = self.pos[i]
        pj = self.pos[j]
        vi = self.vel[i]
        vj = self.vel[j]
        n = (pi - pj) / d[:, None]
        v_rel = vi - vj
        vn = np.einsum("ij,ij->i", v_rel, n)
        gap = d - 2.0 * self.radius  # negative when overlapping
        active = gap < 0.0
        if not np.any(active):
            return

        fn_load = -self.kp * gap
        fn_load = np.where(active, fn_load, 0.0)
        fn_mag = fn_load + self.cp * np.maximum(0.0, vn) * (active.astype(float))
        f_n = fn_mag[:, None] * n

        # Tangential friction.
        v_n_vec = (vn[:, None]) * n
        v_t = v_rel - v_n_vec
        v_t_mag = np.linalg.norm(v_t, axis=1)
        f_t_mag = np.minimum(self.friction * fn_load, self.kt * v_t_mag)
        v_t_dir = -v_t / (v_t_mag[:, None] + 1e-12)
        f_t = f_t_mag[:, None] * v_t_dir

        f_i = f_n + f_t
        f_i = f_i / self.mass[i][:, None]
        f_j = -f_i  # action/reaction, same mass factor used for j below

        np.add.at(acc, i, f_i)
        np.add.at(acc, j, f_j / (self.mass[j][:, None] / self.mass[i][:, None]))
        # Simpler: precompute per-particle acceleration from pair force using equal action/reaction.
        # Redo above to avoid confusion:

    # The previous method is correct enough but the mass scaling for j is messy.
    # The solver below overrides the implementation with a cleaner version.

    def _particle_particle_forces(self, acc: np.ndarray) -> None:
        from scipy.spatial import cKDTree

        tree = cKDTree(self.pos)
        cutoff = 2.0 * self.radius
        pairs = tree.sparse_distance_matrix(tree, max_distance=cutoff, output_type="ndarray")
        if pairs.size == 0:
            return
        i = pairs["i"]
        j = pairs["j"]
        d = pairs["v"]
        mask = i < j
        i = i[mask]
        j = j[mask]
        d = d[mask]
        if i.size == 0:
            return

        pi = self.pos[i]
        pj = self.pos[j]
        vi = self.vel[i]
        vj = self.vel[j]
        n = (pi - pj) / d[:, None]
        v_rel = vi - vj
        vn = np.einsum("ij,ij->i", v_rel, n)
        gap = d - 2.0 * self.radius
        active = gap < 0.0
        if not np.any(active):
            return

        # Force on i; j gets the opposite.
        fn_load = np.where(active, -self.kp * gap, 0.0)
        fn_mag = fn_load + self.cp * np.maximum(0.0, vn) * (active.astype(float))
        f_n = fn_mag[:, None] * n

        v_n_vec = (vn[:, None]) * n
        v_t = v_rel - v_n_vec
        v_t_mag = np.linalg.norm(v_t, axis=1)
        f_t_mag = np.minimum(self.friction * fn_load, self.kt * v_t_mag)
        v_t_dir = -v_t / (v_t_mag[:, None] + 1e-12)
        f_t = f_t_mag[:, None] * v_t_dir

        f_i = f_n + f_t
        # Convert to accelerations.
        inv_mi = 1.0 / self.mass[i]
        inv_mj = 1.0 / self.mass[j]
        np.add.at(acc, i, f_i * inv_mi[:, None])
        np.add.at(acc, j, -f_i * inv_mj[:, None])

    def _step(self, dt: float) -> None:
        theta, omega = self._angle_and_omega(self.current_time)
        acc = np.empty_like(self.pos)
        acc[:] = self.gravity

        self._particle_particle_forces(acc)
        self._wall_and_slab_forces(theta, omega, acc)

        self.vel += acc * dt
        self.pos += self.vel * dt
        self.substep_count += 1

    def _contact_signed_distance(self, theta: float) -> np.ndarray:
        x_local, z_local = self._local_xz(theta)
        signed_center, *_ = self._hourglass_signed_center(x_local, z_local)
        phi_xz = -signed_center
        phi_y = self.slab_half_depth - np.abs(self.pos[:, 1])
        return np.minimum(phi_xz, phi_y) - self.radius

    def advance_to(self, time: float) -> StateFrame:
        if not self._initialized:
            raise RuntimeError("Solver has not been initialized")
        target = float(time)
        while self.current_time < target:
            dt = min(self.sub_dt, target - self.current_time)
            self._step(dt)
            self.current_time += dt

        theta, _ = self._angle_and_omega(self.current_time)
        contact_sd = self._contact_signed_distance(theta)
        external_force = self.mass[:, None] * self.gravity

        return StateFrame(
            time=float(self.current_time),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            attributes={"mass_quantity": self.mass.copy()},
            fields={
                "contact_signed_distance": contact_sd,
                "cumulative_mass_source": np.array([0.0], dtype=float),
                "external_force": external_force,
            },
            scalars={"contact_length_scale": float(self.radius)},
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "substep_count": self.substep_count,
            "sub_dt": self.sub_dt,
            "particle_count": self.n,
            "initialized": self._initialized,
        }
