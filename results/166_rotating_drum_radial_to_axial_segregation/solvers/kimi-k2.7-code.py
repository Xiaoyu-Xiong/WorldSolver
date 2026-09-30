"""DEM-style solver for rotating drum radial-to-axial segregation.

Implements a soft-sphere discrete element method with gravity, rotating cylindrical
wall, end caps, inter-particle contact, and phenomenological segregation forcing.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.config = scene.config
        cent = self.config["centrifuge"]

        self.R = float(cent["radius"])
        self.L = float(cent["length"])
        self.zc = float(cent["center_height"])
        self.omega = float(cent["angular_velocity_rad_s"])
        self.rho = float(cent["particle_density"])
        self.h = float(cent["spacing"])
        self.species_count = int(cent["species_count"])
        self.grid_shape = tuple(int(v) for v in cent["field_grid_shape"])
        self.g = np.array(self.config["gravity"], dtype=float)

        # DEM contact parameters
        self.k_n = 1.0e5
        self.mu = 0.35
        self.damping_frac = 0.3

        # Segregation / banding parameters
        self.a_perc = 1.0  # radial percolation acceleration magnitude
        self.a_band = 0.15  # axial concentration-gradient drift magnitude
        self.nbins = 30  # axial bins for concentration field

        # Time step (s)
        self.dt = 2.0e-4

        # Initial state from the benchmark scene
        frame0 = scene.initial_state()
        self.t = float(frame0.time)
        self.pos = np.asarray(frame0.position, dtype=float).copy()
        self.vel = np.asarray(frame0.velocity, dtype=float).copy()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        attrs = frame0.attributes
        self.species = np.asarray(attrs["species_id"], dtype=np.int32).copy()
        self.radius = np.asarray(attrs["particle_radius"], dtype=float).copy()
        self.mass_quantity = np.asarray(attrs["mass_quantity"], dtype=float).copy()

        # Preserve frozen field/scalar shells from the initial frame
        self._fields_template = {k: v for k, v in frame0.fields.items()}
        self._scalars_template = {k: v for k, v in frame0.scalars.items()}

        # Derived damping coefficient (scalar approximation)
        m_avg = float(np.mean(self.mass))
        self.c_n = self.damping_frac * 2.0 * np.sqrt(m_avg * self.k_n)

        # Cached contact diagnostics for the current state
        self._last_signed = np.zeros(0, dtype=float)
        self._last_external = np.zeros_like(self.pos)

    def _external_force(self) -> np.ndarray:
        """Return per-particle external force (gravity, percolation, axial drift)."""
        N = self.pos.shape[0]
        f_ext = self.mass[:, None] * self.g[None, :]

        # Radial percolation: small species (0) driven inward, large species (1) outward
        r_vec = self.pos[:, 1:3] - np.array([0.0, self.zc])
        r = np.linalg.norm(r_vec, axis=1)
        r_safe = np.maximum(r, 1e-8)
        e_r = r_vec / r_safe[:, None]
        sign = np.where(self.species == 0, -1.0, 1.0)[:, None]
        e_r3 = np.zeros((N, 3), dtype=float)
        e_r3[:, 1:3] = e_r
        f_ext += self.mass[:, None] * self.a_perc * sign * e_r3

        # Axial concentration-gradient drift: move up own concentration gradient
        a_x = self._axial_acceleration()
        f_ext[:, 0] += self.mass * a_x

        return f_ext

    def _axial_acceleration(self) -> np.ndarray:
        """Compute axial acceleration for each particle from concentration gradient."""
        x = self.pos[:, 0]
        b = np.clip(
            ((x + self.L / 2.0) / self.L * self.nbins).astype(int), 0, self.nbins - 1
        )
        counts = np.zeros((self.species_count, self.nbins), dtype=float)
        for s in range(self.species_count):
            mask = self.species == s
            np.add.at(counts[s], b[mask], 1)

        total = counts.sum(axis=0)
        phi = counts / np.maximum(total, 1.0)

        # Smooth to suppress small-wavelength noise and set band scale
        phi_smooth = gaussian_filter1d(phi, sigma=2.0, mode="nearest", axis=1)

        dx = self.L / self.nbins
        dphi = np.zeros_like(phi_smooth)
        dphi[:, 1:-1] = (phi_smooth[:, 2:] - phi_smooth[:, :-2]) / (2.0 * dx)
        dphi[:, 0] = (phi_smooth[:, 1] - phi_smooth[:, 0]) / dx
        dphi[:, -1] = (phi_smooth[:, -1] - phi_smooth[:, -2]) / dx

        # Remove net drift per species
        dphi -= dphi.mean(axis=1, keepdims=True)

        a = np.zeros(self.pos.shape[0], dtype=float)
        for s in range(self.species_count):
            mask = self.species == s
            a[mask] = self.a_band * dphi[s, b[mask]]
        return a

    def _signed_distances(self) -> np.ndarray:
        """Collect signed gaps for active/near particle-particle and wall contacts."""
        N = self.pos.shape[0]
        signed: list[np.ndarray] = []

        # Particle-particle
        max_cut = 2.0 * float(self.radius.max()) + 1e-6
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(max_cut, output_type="ndarray")
        if pairs.shape[0] > 0:
            i = pairs[:, 0]
            j = pairs[:, 1]
            dx = self.pos[j] - self.pos[i]
            dist = np.linalg.norm(dx, axis=1)
            rsum = self.radius[i] + self.radius[j]
            signed.append(dist - rsum)

        # Cylindrical wall (signed gap from the inner wall surface)
        r_vec = self.pos[:, 1:3] - np.array([0.0, self.zc])
        r = np.linalg.norm(r_vec, axis=1)
        signed.append((self.R - self.radius) - r)

        # End caps
        signed.append((self.L / 2.0 - self.radius) - self.pos[:, 0])
        signed.append(self.pos[:, 0] - (-self.L / 2.0 + self.radius))

        return np.concatenate(signed) if signed else np.zeros(0, dtype=float)

    def _substep(self) -> None:
        """Advance one DEM substep."""
        N = self.pos.shape[0]

        # External forces
        f = self._external_force()

        # Particle-particle contact
        max_cut = 2.0 * float(self.radius.max()) + 1e-6
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(max_cut, output_type="ndarray")
        if pairs.shape[0] > 0:
            i = pairs[:, 0]
            j = pairs[:, 1]
            dx = self.pos[j] - self.pos[i]
            dist = np.linalg.norm(dx, axis=1)
            dist = np.maximum(dist, 1e-12)
            n = dx / dist[:, None]
            rsum = self.radius[i] + self.radius[j]
            overlap = rsum - dist

            active = overlap > 0.0
            if np.any(active):
                v_rel = self.vel[j] - self.vel[i]
                v_n = np.sum(v_rel * n, axis=1)
                fn = self.k_n * overlap - self.c_n * v_n
                fn = np.maximum(fn, 0.0)

                f_n_vec = fn[:, None] * n

                # Tangential friction (dynamic Coulomb)
                v_t = v_rel - v_n[:, None] * n
                vt_mag = np.linalg.norm(v_t, axis=1)
                t_dir = np.zeros_like(v_t)
                mask = vt_mag > 1e-8
                t_dir[mask] = v_t[mask] / vt_mag[mask, None]
                ft_mag = self.mu * self.k_n * overlap
                f_t_vec = -ft_mag[:, None] * t_dir

                f_i = f_n_vec + f_t_vec
                f_i = f_i * active[:, None].astype(float)
                np.add.at(f, i, f_i)
                np.add.at(f, j, -f_i)

        # Cylindrical wall contact
        r_vec = self.pos[:, 1:3] - np.array([0.0, self.zc])
        r = np.linalg.norm(r_vec, axis=1)
        r = np.maximum(r, 1e-12)
        e_r = r_vec / r[:, None]
        n_w = -e_r  # inward normal

        v_wall = np.zeros((N, 3), dtype=float)
        v_wall[:, 1] = -self.omega * (self.pos[:, 2] - self.zc)
        v_wall[:, 2] = self.omega * self.pos[:, 1]

        delta_wall = self.radius + r - self.R
        active_w = delta_wall > 0.0
        if np.any(active_w):
            v_rel_w = (self.vel - v_wall)[:, 1:3]
            v_close_w = np.sum(v_rel_w * n_w, axis=1)
            fn_w = self.k_n * delta_wall - self.c_n * v_close_w
            fn_w = np.maximum(fn_w, 0.0)

            f_n_w = (fn_w * n_w.T).T

            v_t_w = v_rel_w - v_close_w[:, None] * n_w
            vt_mag_w = np.linalg.norm(v_t_w, axis=1)
            t_dir_w = np.zeros_like(v_t_w)
            mask_w = vt_mag_w > 1e-8
            t_dir_w[mask_w] = v_t_w[mask_w] / vt_mag_w[mask_w, None]
            ft_mag_w = self.mu * self.k_n * delta_wall
            f_t_w = -ft_mag_w[:, None] * t_dir_w

            f_wall = np.zeros((N, 3), dtype=float)
            f_wall[:, 1:3] = (f_n_w + f_t_w) * active_w[:, None].astype(float)
            f += f_wall

        # End-cap contacts
        # Right cap at x = +L/2
        delta_r = self.pos[:, 0] + self.radius - self.L / 2.0
        active_r = delta_r > 0.0
        if np.any(active_r):
            v_close_r = -self.vel[:, 0]  # (v_i - 0) . (-1,0,0)
            fn_r = self.k_n * delta_r - self.c_n * v_close_r
            fn_r = np.maximum(fn_r, 0.0)
            v_t_r = self.vel[:, 1:3]
            vt_mag_r = np.linalg.norm(v_t_r, axis=1)
            t_dir_r = np.zeros_like(v_t_r)
            mask_r = vt_mag_r > 1e-8
            t_dir_r[mask_r] = v_t_r[mask_r] / vt_mag_r[mask_r, None]
            ft_mag_r = self.mu * self.k_n * delta_r
            f_t_r = -ft_mag_r[:, None] * t_dir_r

            f[:, 0] += -fn_r * active_r
            f[:, 1:3] += f_t_r * active_r[:, None]

        # Left cap at x = -L/2
        delta_l = -self.L / 2.0 + self.radius - self.pos[:, 0]
        active_l = delta_l > 0.0
        if np.any(active_l):
            v_close_l = self.vel[:, 0]  # (v_i - 0) . (1,0,0)
            fn_l = self.k_n * delta_l - self.c_n * v_close_l
            fn_l = np.maximum(fn_l, 0.0)
            v_t_l = self.vel[:, 1:3]
            vt_mag_l = np.linalg.norm(v_t_l, axis=1)
            t_dir_l = np.zeros_like(v_t_l)
            mask_l = vt_mag_l > 1e-8
            t_dir_l[mask_l] = v_t_l[mask_l] / vt_mag_l[mask_l, None]
            ft_mag_l = self.mu * self.k_n * delta_l
            f_t_l = -ft_mag_l[:, None] * t_dir_l

            f[:, 0] += fn_l * active_l
            f[:, 1:3] += f_t_l * active_l[:, None]

        # Integrate
        self.vel += f / self.mass[:, None] * self.dt
        self.pos += self.vel * self.dt

        # Soft confinement / velocity clamp for stability
        self._confine()
        self._clamp_velocity()

        # Cache diagnostics for the updated state
        self._last_signed = self._signed_distances()
        self._last_external = self._external_force()

    def _confine(self) -> None:
        # Cylinder
        y_off = self.pos[:, 1] - 0.0
        z_off = self.pos[:, 2] - self.zc
        r = np.sqrt(y_off * y_off + z_off * z_off)
        max_r = self.R - self.radius - 5e-5
        over = r > max_r
        if np.any(over):
            scale = max_r[over] / r[over]
            self.pos[over, 1] = y_off[over] * scale
            self.pos[over, 2] = self.zc + z_off[over] * scale

        # End caps
        margin = self.radius + 5e-5
        x_min = -self.L / 2.0 + margin
        x_max = self.L / 2.0 - margin
        self.pos[:, 0] = np.clip(self.pos[:, 0], x_min, x_max)

    def _clamp_velocity(self) -> None:
        v_max = 6.0
        speed = np.linalg.norm(self.vel, axis=1)
        over = speed > v_max
        if np.any(over):
            self.vel[over] = (
                self.vel[over] / speed[over, None] * v_max
            )

    def _make_frame(self) -> StateFrame:
        attributes = {
            "species_id": self.species,
            "particle_radius": self.radius,
            "mass_quantity": self.mass_quantity,
        }
        fields = {k: v for k, v in self._fields_template.items()}
        fields["contact_signed_distance"] = self._last_signed.copy()
        fields["external_force"] = self._last_external.copy()
        fields["cumulative_mass_source"] = np.zeros(self.species_count, dtype=float)

        scalars = {k: v for k, v in self._scalars_template.items()}
        scalars["contact_length_scale"] = self.h

        return StateFrame(
            self.t,
            self.pos.copy(),
            self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.t < target - 1e-12:
            # Use the fixed substep size for deterministic integration
            self._substep()
            self.t += self.dt
        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        return {}
