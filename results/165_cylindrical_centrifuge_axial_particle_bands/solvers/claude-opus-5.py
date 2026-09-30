"""Horizontal cylindrical centrifuge with viscous carrier fluid.

Physical model
-------------
A rigid horizontal drum (axis = world x, inner radius R, length L) is filled
with a very viscous Newtonian carrier fluid and spun about its axis with a
prescribed angular velocity that ramps up viscously to the frozen target
``angular_velocity_rad_s``.  The carrier fluid is treated as being in
solid-body rotation with the drum (valid for a fully filled drum after the
viscous spin-up time), so the fluid velocity field is
``u_f(x) = Omega x_hat x r_perp`` and the fluid pressure field satisfies
``grad p = rho_f (g + Omega^2 r_perp)``.

Suspended particles are non-neutrally buoyant spheres of two species with
different radii (persistent ``species_id``).  Because the particles are large
and the carrier is highly viscous, the particle dynamics are inertia-dominated
only in the rotational (advective) part and strongly overdamped in the drift
part.  We therefore integrate the dynamics in the co-rotating frame, where the
carrier fluid is at rest and the quasi-steady force balance for each particle
is

    zeta_i * v_i = dm_i * (g_rot(t) + Omega(t)^2 r_perp)   (buoyancy-corrected
                                                            gravity + effective
                                                            centrifugal load)
                 + F_contact_i                             (pairwise soft
                                                            frictional contacts)
                 + F_hydro_i                               (axial hydrodynamic
                                                            coupling in the
                                                            concentrated layer)
                 + F_wall_i                                (drum reaction)

with Stokes drag coefficient ``zeta_i = 6 pi mu a_i``.  The advective part of
the motion is handled exactly by the frame rotation, which keeps the scheme
unconditionally stable for the rotation and free of spurious radial drift.

Axial banding
-------------
Hydrodynamic interactions inside the concentrated wall layer of a rotating
suspension make the axial species distribution unstable: a species-rich region
draws more of that species in.  This is modelled as a conservative
Cahn-Hilliard-type axial flux computed at run time from the *measured* axial
composition field ``psi(x) = (c_A - c_B)/(c_A + c_B)``:

    mu_ch = -psi + psi^3 - kappa d2psi/dx2
    v_axial(species) = -s * M d(mu_ch)/dx ,   s = +1 (A), -1 (B)

whose linear growth rate ``sigma(k) = M (k^2 - kappa k^4)`` selects a finite
band wavelength ``2 pi sqrt(2 kappa)`` and saturates nonlinearly.  The coupling
is gated by the locally measured packing (neighbour count), so it is inactive
in the initial dilute mixed state and only switches on after the particles have
been centrifuged into the dense layer.  Nothing about the band positions is
prescribed: they grow from the seeded compositional noise of the initial state.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---------------- setup ----------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        cen = dict(cfg["centrifuge"])

        self.seed = int(seed)
        self.radius = float(cen["radius"])
        self.length = float(cen["length"])
        self.center_height = float(cen["center_height"])
        self.omega0 = float(cen["angular_velocity_rad_s"])
        self.spacing = float(cen["spacing"])
        self.rho_p = float(cen["particle_density"])
        self.species_count = int(cen["species_count"])
        self.grid_shape = tuple(int(v) for v in cen["field_grid_shape"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.sample_dt = float(cfg["sample_dt"])
        self.duration = float(cfg["duration"])

        frame0 = scene.initial_state()
        pos0 = np.asarray(frame0.position, dtype=float)
        self.n = int(pos0.shape[0])
        self.species = np.asarray(frame0.attributes["species_id"], dtype=np.int32)
        self.a = np.asarray(frame0.attributes["particle_radius"], dtype=float)
        self.mass = np.asarray(frame0.mass, dtype=float)
        self.mass_quantity = np.asarray(frame0.attributes["mass_quantity"], dtype=float)
        self.body_id = np.zeros(self.n, dtype=np.int32)
        self.scalars = {
            "grid_spacing": float(frame0.scalars["grid_spacing"]),
            "characteristic_length": float(frame0.scalars["characteristic_length"]),
            "contact_length_scale": float(frame0.scalars.get("contact_length_scale", self.spacing)),
        }

        # co-rotating frame coordinates (axis x, origin on the drum axis)
        self.q = np.empty((self.n, 3), dtype=float)
        self.q[:, 0] = pos0[:, 0]
        self.q[:, 1] = pos0[:, 1]
        self.q[:, 2] = pos0[:, 2] - self.center_height

        # ---- material / fluid properties ----
        self.rho_f = 1000.0        # viscous carrier fluid density [kg/m^3]
        self.mu_f = 60.0           # carrier dynamic viscosity [Pa s]
        self.vol = (4.0 * np.pi / 3.0) * self.a ** 3
        self.mass_f = self.rho_f * self.vol            # displaced fluid mass
        self.dmass = self.mass - self.mass_f           # buoyancy-corrected mass
        self.zeta = 6.0 * np.pi * self.mu_f * self.a   # Stokes drag coefficient
        self.inv_zeta = 1.0 / self.zeta

        # ---- numerics ----
        self.dt = 2.0e-3
        self.spin_tau = 0.4                # viscous spin-up time constant [s]
        self.k_contact = 3000.0            # soft-sphere normal stiffness [N/m]
        self.k_wall = 6000.0               # drum wall stiffness [N/m]
        self.v_max = 3.0                   # drift speed cap [m/s]
        self.max_step_disp = 0.02          # displacement cap per step [m]
        self.cut = 2.0 * float(self.a.max())
        self.skin = 0.05

        # ---- axial hydrodynamic coupling (band instability) ----
        self.n_bins = int(self.grid_shape[0])
        self.dxb = self.length / self.n_bins
        self.bin_centers = -0.5 * self.length + (np.arange(self.n_bins) + 0.5) * self.dxb
        self.ch_kappa = 0.018      # gradient-energy coefficient  -> band wavelength
        self.ch_mob = 0.09         # axial mobility [m^2/s]
        self.v_axial_max = 0.5     # cap on hydrodynamic axial drift [m/s]
        self.gate_n0 = 7.0         # neighbour count where coupling switches on
        self.gate_width = 7.0

        # ---- diagnostic grid ----
        nx, ny, nz = self.grid_shape
        self.gx = np.linspace(-0.5 * self.length, 0.5 * self.length, nx + 1)
        self.gy = np.linspace(-self.radius, self.radius, ny + 1)
        self.gz = np.linspace(-self.radius, self.radius, nz + 1)
        cx = 0.5 * (self.gx[:-1] + self.gx[1:])
        cy = 0.5 * (self.gy[:-1] + self.gy[1:])
        cz = 0.5 * (self.gz[:-1] + self.gz[1:])
        CY, CZ = np.meshgrid(cy, cz, indexing="ij")
        inside = (CY ** 2 + CZ ** 2) <= self.radius ** 2
        self.active_mask = np.broadcast_to(inside[None, :, :], (nx, ny, nz)).copy()
        self.cell_x = cx
        self.cell_y = cy
        self.cell_z = cz
        CX3 = cx[:, None, None]
        CY3 = cy[None, :, None]
        CZ3 = cz[None, None, :]
        self._cx3 = np.broadcast_to(CX3, (nx, ny, nz))
        self._cy3 = np.broadcast_to(CY3, (nx, ny, nz))
        self._cz3 = np.broadcast_to(CZ3, (nx, ny, nz))

        # ---- running state ----
        self.time = 0.0
        self.theta = 0.0
        self.step_count = 0
        self.pairs = np.zeros((0, 2), dtype=np.int64)
        self._build_ref = self.q.copy()
        self._rebuild_pairs()

        # force diagnostics of the current state
        self.f_ext_rot = np.zeros((self.n, 3), dtype=float)
        self.f_int_rot = np.zeros((self.n, 3), dtype=float)
        self.v_rot = np.zeros((self.n, 3), dtype=float)
        self.max_overlap = 0.0
        self._evaluate_forces()

    # ---------------- helpers ----------------
    def _omega(self, t: float) -> float:
        """Viscously ramped prescribed drum rotation rate."""
        return self.omega0 * (1.0 - np.exp(-t / self.spin_tau))

    def _gravity_rot(self, theta: float) -> np.ndarray:
        """World gravity expressed in the co-rotating frame (rotation about x)."""
        c, s = np.cos(theta), np.sin(theta)
        gx, gy, gz = self.gravity
        return np.array([gx, c * gy + s * gz, -s * gy + c * gz], dtype=float)

    def _rot_to_lab(self, vec: np.ndarray, theta: float) -> np.ndarray:
        c, s = np.cos(theta), np.sin(theta)
        out = np.empty_like(vec)
        out[:, 0] = vec[:, 0]
        out[:, 1] = c * vec[:, 1] - s * vec[:, 2]
        out[:, 2] = s * vec[:, 1] + c * vec[:, 2]
        return out

    def _rebuild_pairs(self) -> None:
        tree = cKDTree(self.q)
        pairs = tree.query_pairs(r=self.cut + self.skin, output_type="ndarray")
        self.pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        self._build_ref = self.q.copy()

    def _maybe_rebuild(self) -> None:
        disp = np.max(np.abs(self.q - self._build_ref))
        if disp > 0.5 * self.skin:
            self._rebuild_pairs()

    def _contact_forces(self):
        """Soft-sphere pairwise repulsion + local packing measure."""
        f = np.zeros((self.n, 3), dtype=float)
        neigh = np.zeros(self.n, dtype=float)
        kdiag = np.zeros(self.n, dtype=float)
        if self.pairs.shape[0] == 0:
            return f, neigh, kdiag, 0.0
        i = self.pairs[:, 0]
        j = self.pairs[:, 1]
        d = self.q[i] - self.q[j]
        dist = np.sqrt(np.einsum("ij,ij->i", d, d))
        sig = self.a[i] + self.a[j]
        # local concentration measure: neighbours within 1.2 contact diameters
        close = dist < 1.2 * sig
        if np.any(close):
            neigh += np.bincount(i[close], minlength=self.n).astype(float)
            neigh += np.bincount(j[close], minlength=self.n).astype(float)
        act = (dist < sig) & (dist > 1.0e-12)
        if not np.any(act):
            return f, neigh, kdiag, 0.0
        ia, ja = i[act], j[act]
        overlap = sig[act] - dist[act]
        nvec = d[act] / dist[act][:, None]
        fmag = self.k_contact * overlap
        fv = nvec * fmag[:, None]
        for c in range(3):
            f[:, c] += np.bincount(ia, weights=fv[:, c], minlength=self.n)
            f[:, c] -= np.bincount(ja, weights=fv[:, c], minlength=self.n)
        # diagonal stiffness estimate for the semi-implicit drift update
        kdiag += self.k_contact * np.bincount(ia, minlength=self.n)
        kdiag += self.k_contact * np.bincount(ja, minlength=self.n)
        return f, neigh, kdiag, float(overlap.max())

    def _wall_forces(self):
        """Reaction of the rigid drum wall and end caps (co-rotating frame)."""
        f = np.zeros((self.n, 3), dtype=float)
        yz = self.q[:, 1:3]
        r = np.sqrt(yz[:, 0] ** 2 + yz[:, 1] ** 2)
        r_lim = self.radius - self.a
        pen = r - r_lim
        out = pen > 0.0
        if np.any(out):
            safe = np.maximum(r[out], 1.0e-12)
            nvec = yz[out] / safe[:, None]
            f[out, 1] -= self.k_wall * pen[out] * nvec[:, 0]
            f[out, 2] -= self.k_wall * pen[out] * nvec[:, 1]
        x_lim = 0.5 * self.length - self.a
        hi = self.q[:, 0] > x_lim
        lo = self.q[:, 0] < -x_lim
        f[hi, 0] -= self.k_wall * (self.q[hi, 0] - x_lim[hi])
        f[lo, 0] += self.k_wall * (-x_lim[lo] - self.q[lo, 0])
        kdiag = np.zeros(self.n, dtype=float)
        kdiag[out] += self.k_wall
        kdiag[hi] += self.k_wall
        kdiag[lo] += self.k_wall
        return f, kdiag

    @staticmethod
    def _smooth(f: np.ndarray) -> np.ndarray:
        fp = np.concatenate((f[:1], f, f[-1:]))
        return 0.25 * fp[:-2] + 0.5 * fp[1:-1] + 0.25 * fp[2:]

    def _d1(self, f: np.ndarray) -> np.ndarray:
        fp = np.concatenate((f[:1], f, f[-1:]))
        return (fp[2:] - fp[:-2]) / (2.0 * self.dxb)

    def _d2(self, f: np.ndarray) -> np.ndarray:
        fp = np.concatenate((f[:1], f, f[-1:]))
        return (fp[2:] - 2.0 * f + fp[:-2]) / (self.dxb ** 2)

    def _axial_hydro_velocity(self, neigh: np.ndarray) -> np.ndarray:
        """Axial drift from the measured composition field (Cahn-Hilliard flux)."""
        idx = np.clip(
            ((self.q[:, 0] + 0.5 * self.length) / self.dxb).astype(np.int64),
            0,
            self.n_bins - 1,
        )
        is_a = self.species == 0
        w = self.vol
        cA = np.bincount(idx[is_a], weights=w[is_a], minlength=self.n_bins)
        cB = np.bincount(idx[~is_a], weights=w[~is_a], minlength=self.n_bins)
        cA = self._smooth(self._smooth(cA))
        cB = self._smooth(self._smooth(cB))
        psi = (cA - cB) / (cA + cB + 1.0e-12)
        psi = np.clip(psi, -1.5, 1.5)
        mu_ch = -psi + psi ** 3 - self.ch_kappa * self._d2(psi)
        v_bins = -self.ch_mob * self._d1(mu_ch)
        v_bins = np.clip(v_bins, -self.v_axial_max, self.v_axial_max)
        v_part = np.interp(self.q[:, 0], self.bin_centers, v_bins)
        sign = np.where(is_a, 1.0, -1.0)
        gate = np.clip((neigh - self.gate_n0) / self.gate_width, 0.0, 1.0)
        return sign * gate * v_part

    def _evaluate_forces(self) -> None:
        """Assemble the quasi-steady force balance for the current state."""
        omega = self._omega(self.time)
        g_rot = self._gravity_rot(self.theta)

        f_contact, neigh, k_c, max_ov = self._contact_forces()
        f_wall, k_w = self._wall_forces()
        self.max_overlap = max_ov
        # Diagonal implicit-Euler mobility: the stiff elastic response is treated
        # implicitly, so the overdamped drift update is stable for any dt.
        mob = self.inv_zeta / (1.0 + (k_c + k_w) * self.dt * self.inv_zeta)

        # buoyancy-corrected gravity + effective centrifugal load
        acc = np.empty((self.n, 3), dtype=float)
        acc[:, 0] = g_rot[0]
        acc[:, 1] = g_rot[1] + omega * omega * self.q[:, 1]
        acc[:, 2] = g_rot[2] + omega * omega * self.q[:, 2]
        f_body = self.dmass[:, None] * acc

        # axial hydrodynamic coupling expressed as a force
        v_ax = self._axial_hydro_velocity(neigh)
        f_hydro = np.zeros((self.n, 3), dtype=float)
        f_hydro[:, 0] = self.zeta * v_ax

        f_eff = f_body + f_contact + f_hydro + f_wall
        v = f_eff * mob[:, None]
        speed = np.sqrt(np.einsum("ij,ij->i", v, v))
        big = speed > self.v_max
        if np.any(big):
            v[big] *= (self.v_max / speed[big])[:, None]
        self.v_rot = v

        # ---- exported diagnostics ----
        # real external forces: weight + fluid pressure-gradient (buoyancy incl.
        # the rotational part) + rigid-wall reaction
        press = np.empty((self.n, 3), dtype=float)
        press[:, 0] = g_rot[0]
        press[:, 1] = g_rot[1] + omega * omega * self.q[:, 1]
        press[:, 2] = g_rot[2] + omega * omega * self.q[:, 2]
        self.f_ext_rot = self.mass[:, None] * g_rot[None, :] - self.mass_f[:, None] * press + f_wall
        # internal / multiphysics pair forces: contacts, hydrodynamic coupling,
        # and viscous drag exchanged with the carrier fluid
        f_drag = -self.zeta[:, None] * v
        self.f_int_rot = f_contact + f_hydro + f_drag
        self.neigh = neigh

    def _step(self, dt: float) -> None:
        self._evaluate_forces()
        disp = self.v_rot * dt
        mag = np.sqrt(np.einsum("ij,ij->i", disp, disp))
        big = mag > self.max_step_disp
        if np.any(big):
            disp[big] *= (self.max_step_disp / mag[big])[:, None]
        self.q += disp

        # hard confinement inside the drum (projection, keeps material captured)
        r_lim = self.radius - self.a
        r = np.sqrt(self.q[:, 1] ** 2 + self.q[:, 2] ** 2)
        over = r > r_lim
        if np.any(over):
            scale = r_lim[over] / np.maximum(r[over], 1.0e-12)
            self.q[over, 1] *= scale
            self.q[over, 2] *= scale
        x_lim = 0.5 * self.length - self.a
        np.clip(self.q[:, 0], -x_lim, x_lim, out=self.q[:, 0])

        # exact advance of the prescribed rigid rotation
        self.theta += self._omega(self.time + 0.5 * dt) * dt
        self.time += dt
        self.step_count += 1
        self._maybe_rebuild()

    # ---------------- sampling ----------------
    def _lab_state(self):
        theta = self.theta
        omega = self._omega(self.time)
        c, s = np.cos(theta), np.sin(theta)
        pos = np.empty((self.n, 3), dtype=float)
        pos[:, 0] = self.q[:, 0]
        pos[:, 1] = c * self.q[:, 1] - s * self.q[:, 2]
        pos[:, 2] = s * self.q[:, 1] + c * self.q[:, 2] + self.center_height
        # lab velocity = solid-body carrier rotation + rotated drift
        dy = pos[:, 1]
        dz = pos[:, 2] - self.center_height
        vel = self._rot_to_lab(self.v_rot, theta)
        vel[:, 1] += -omega * dz
        vel[:, 2] += omega * dy
        return pos, vel

    def _velocity_field(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        nx, ny, nz = self.grid_shape
        omega = self._omega(self.time)
        field = np.zeros((nx, ny, nz, 3), dtype=float)
        # background carrier fluid (solid-body rotation about the drum axis)
        field[..., 1] = -omega * self._cz3
        field[..., 2] = omega * self._cy3
        ix = np.clip(np.searchsorted(self.gx, pos[:, 0], side="right") - 1, 0, nx - 1)
        iy = np.clip(np.searchsorted(self.gy, pos[:, 1], side="right") - 1, 0, ny - 1)
        iz = np.clip(
            np.searchsorted(self.gz, pos[:, 2] - self.center_height, side="right") - 1,
            0,
            nz - 1,
        )
        flat = (ix * ny + iy) * nz + iz
        ncell = nx * ny * nz
        wsum = np.bincount(flat, weights=self.mass, minlength=ncell)
        acc = np.empty((ncell, 3), dtype=float)
        for c in range(3):
            acc[:, c] = np.bincount(flat, weights=self.mass * vel[:, c], minlength=ncell)
        occupied = wsum > 0.0
        if np.any(occupied):
            mean = np.zeros((ncell, 3), dtype=float)
            mean[occupied] = acc[occupied] / wsum[occupied][:, None]
            flat_field = field.reshape(ncell, 3)
            flat_field[occupied] = mean[occupied]
            field = flat_field.reshape(nx, ny, nz, 3)
        field[~self.active_mask] = 0.0
        return field

    def _frame(self, time: float) -> StateFrame:
        pos, vel = self._lab_state()
        f_ext = self._rot_to_lab(self.f_ext_rot, self.theta)
        f_int_lab = self._rot_to_lab(self.f_int_rot, self.theta)
        inter = np.zeros((self.species_count, 3), dtype=float)
        for k in range(self.species_count):
            sel = self.species == k
            if np.any(sel):
                inter[k] = f_int_lab[sel].sum(axis=0)
        vfield = self._velocity_field(pos, vel)
        frame = StateFrame(
            float(time),
            np.ascontiguousarray(pos),
            np.ascontiguousarray(vel),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "species_id": self.species.copy(),
                "particle_radius": self.a.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "cumulative_mass_source": np.zeros(self.species_count, dtype=float),
                "velocity_field": vfield,
                "active_mask": self.active_mask.copy(),
                "external_force": np.ascontiguousarray(f_ext),
                "interaction_force": inter,
            },
            scalars=dict(self.scalars),
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1.0e-12:
            raise ValueError("advance_to() must be called with non-decreasing times")
        while self.time < target - 1.0e-12:
            dt = min(self.dt, target - self.time)
            self._step(dt)
        self.time = target
        self._evaluate_forces()
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "particles": int(self.n),
            "steps": int(self.step_count),
            "dt": float(self.dt),
            "time": float(self.time),
            "drum_angle_rad": float(self.theta),
            "omega_rad_s": float(self._omega(self.time)),
            "carrier_viscosity_pa_s": float(self.mu_f),
            "carrier_density": float(self.rho_f),
            "max_contact_overlap_m": float(self.max_overlap),
            "contact_pairs": int(self.pairs.shape[0]),
            "band_wavelength_m": float(2.0 * np.pi * np.sqrt(2.0 * self.ch_kappa)),
        }
