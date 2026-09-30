"""Cylindrical centrifuge axial particle-band segregation solver.

Rotating-frame Lagrangian suspension model:

* prescribed drum rotation -> solid-body co-rotating viscous carrier fluid,
* buoyancy-corrected centrifugal + gravity body forces, Coriolis terms,
* Stokes drag (semi-implicit, unconditionally stable) toward the local fluid
  velocity, giving differential centrifuging (larger species migrates
  radially outward faster -> nonuniform cross-section),
* soft-sphere particle/particle and particle/wall contacts (viscoelastic
  normal force, end caps and cylindrical wall of the frozen drum),
* two-way hydrodynamic coupling: the particle loading drives an axial
  back-flow velocity proportional to the (smoothed) axial gradient of each
  species' own concentration.  Because the drag acts on particles moving with
  that flow, small axial concentration perturbations are amplified into
  localized particle-rich bands (the PRL 89, 104301 mechanism), saturating at
  the packing limit.  The coupling is gated by the degree of radial
  organization so radial centrifuging precedes axial band formation.

No Genesis: discretization, time stepping and coupling are implemented here
with NumPy/SciPy only.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Rotating-frame particle solver for the centrifuge segregation scene."""

    def __init__(self) -> None:
        self.t = 0.0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        c = cfg["centrifuge"]
        self.R = float(c["radius"])
        self.L = float(c["length"])
        self.h = float(c["spacing"])
        self.omega = float(c["angular_velocity_rad_s"])
        self.center = np.array([0.0, 0.0, float(c["center_height"])], dtype=float)
        self.g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.rho_f = 1000.0                      # carrier fluid density [kg/m^3]
        self.rho_p = float(c["particle_density"])
        self.mu = 70.0                           # carrier dynamic viscosity [Pa s]
        self.dt = 0.002

        frame0 = scene.initial_state()
        self.mass = np.asarray(frame0.mass, dtype=float).copy()
        self.species = np.asarray(frame0.attributes["species_id"], dtype=np.int32).copy()
        self.radius = np.asarray(frame0.attributes["particle_radius"], dtype=float).copy()
        self.mass_quantity = np.asarray(frame0.attributes["mass_quantity"], dtype=float).copy()
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32).copy()
        self.N = int(self.mass.size)
        self.K = int(self.mass_quantity.shape[1]) if self.mass_quantity.ndim == 2 else 1

        # rotating-frame state (origin on the drum axis)
        self.pos = np.asarray(frame0.position, dtype=float).copy() - self.center
        self.vel = np.zeros((self.N, 3), dtype=float)
        self.tau = self.mass / (6.0 * np.pi * self.mu * self.radius)  # Stokes time
        self.buoy = 1.0 - self.rho_f / self.rho_p

        # ---- axial concentration bins along the drum axis ------------------
        self.n_bins = 48
        self.dx_bin = self.L / self.n_bins
        self.bin_vol = np.pi * self.R ** 2 * self.dx_bin
        self.vol = (4.0 / 3.0) * np.pi * self.radius ** 3
        self.vol_k = np.array([
            float(np.mean(self.vol[self.species == k])) for k in range(self.K)
        ])
        self.sigma_bins = 1.8                    # smoothing length ~0.18 m
        self.c_pack = 0.62                       # packing saturation fraction
        self.A0 = 1.2                            # coupling amplitude [m^2/s]
        self.s_k = 1.0 + 0.5 * np.arange(self.K)  # species coupling contrast
        self.u_cap = 0.35                        # axial back-flow speed cap

        # gating: axial instability driven by the centrifuged wall layer
        self.r_gate = 0.7 * self.R
        self.f_wall0 = float(np.mean(self._r_perp() > self.r_gate))

        # ---- contacts -------------------------------------------------------
        self.k_wall = 15000.0
        self.c_wall = 30.0
        self.k_pair = 9000.0
        self.c_pair = 20.0
        self.pair_radius = 2.0 * float(self.radius.max()) + 0.08
        self.pair_refresh = 10
        self._pi = np.zeros(0, dtype=np.int64)
        self._pj = np.zeros(0, dtype=np.int64)
        self._pair_age = self.pair_refresh

        # ---- diagnostic grid ------------------------------------------------
        self.grid_shape = tuple(int(v) for v in c["field_grid_shape"])
        self.grid_h = float(c["spacing"])
        nx, ny, nz = self.grid_shape
        self.gx = (np.arange(nx) + 0.5) * self.grid_h - nx * self.grid_h / 2.0
        self.gy = (np.arange(ny) + 0.5) * self.grid_h - ny * self.grid_h / 2.0
        self.gz = self.center[2] + (np.arange(nz) + 0.5) * self.grid_h - nz * self.grid_h / 2.0
        yy, zz = np.meshgrid(self.gy, self.gz - self.center[2], indexing="ij")
        inside = (yy ** 2 + zz ** 2) <= self.R ** 2
        self.active_mask = np.broadcast_to(inside, self.grid_shape).copy()

        self._last_c_tot = np.zeros(self.n_bins)
        self._last_u_mean = np.zeros(self.n_bins)

    # ------------------------------------------------------------- utilities
    def _r_perp(self) -> np.ndarray:
        return np.hypot(self.pos[:, 1], self.pos[:, 2])

    def _rot_x(self, theta: float) -> np.ndarray:
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, ct, -st],
                         [0.0, st, ct]])

    # ---------------------------------------------------------- body forces
    def _body_force(self) -> np.ndarray:
        """External field forces in the rotating frame (per particle)."""
        theta = self.omega * self.t
        ct, st = np.cos(theta), np.sin(theta)
        # gravity seen in the rotating frame: R_x(-theta) g
        gy = self.g[1] * ct + self.g[2] * st
        gz = -self.g[1] * st + self.g[2] * ct
        bc = self.buoy
        w2 = self.omega ** 2
        F = np.zeros((self.N, 3))
        # buoyancy-corrected centrifugal field (radial, static in this frame)
        F[:, 1] = self.mass * (bc * w2 * self.pos[:, 1] + bc * gy) + self.mass * 2.0 * self.omega * self.vel[:, 2]
        F[:, 2] = self.mass * (bc * w2 * self.pos[:, 2] + bc * gz) - self.mass * 2.0 * self.omega * self.vel[:, 1]
        return F

    # ------------------------------------------------- hydrodynamic coupling
    def _axial_flow(self):
        """Axial back-flow velocity driven by the particle concentration.

        Returns (per-particle axial fluid speed, per-bin mean speed, total
        smoothed concentration).  Drift is toward each species' own
        concentration gradient (amplifying perturbations), scaled by the
        centrifuged wall-layer fraction and quenched at the packing limit.
        """
        idx = np.clip(((self.pos[:, 0] + self.L / 2.0) / self.dx_bin).astype(np.int64),
                      0, self.n_bins - 1)
        conc = np.zeros((self.K, self.n_bins))
        for k in range(self.K):
            cnt = np.bincount(idx[self.species == k], minlength=self.n_bins)
            conc[k] = gaussian_filter1d(cnt * self.vol_k[k] / self.bin_vol,
                                        self.sigma_bins, mode="nearest")
        grad = np.stack([np.gradient(conc[k], self.dx_bin) for k in range(self.K)])
        c_tot = conc.sum(axis=0)

        f_wall = float(np.mean(self._r_perp() > self.r_gate))
        s_gate = min(max((f_wall - self.f_wall0) / 0.30, 0.0), 1.0)
        pack = 1.0 - c_tot / self.c_pack

        ub = np.empty((self.K, self.n_bins))
        for k in range(self.K):
            ub[k] = np.clip(self.s_k[k] * s_gate * self.A0 * pack * grad[k],
                            -self.u_cap, self.u_cap)
        u_particle = ub[self.species, idx]
        u_mean = (conc * ub).sum(axis=0) / np.maximum(c_tot, 1e-9)
        self._last_c_tot = c_tot
        self._last_u_mean = u_mean
        return u_particle, u_mean, c_tot

    # -------------------------------------------------------------- contacts
    def _contacts(self) -> np.ndarray:
        """Internal contact forces (wall, end caps, particle pairs)."""
        F = np.zeros((self.N, 3))

        # cylindrical wall (stationary in the rotating frame)
        r = self._r_perp()
        lim = self.R - self.radius
        pen = r - lim
        m = pen > 0.0
        if np.any(m):
            inv = 1.0 / np.maximum(r[m], 1e-9)
            ny = self.pos[m, 1] * inv
            nz = self.pos[m, 2] * inv
            vn = self.vel[m, 1] * ny + self.vel[m, 2] * nz
            fn = -self.k_wall * pen[m] - self.c_wall * np.maximum(vn, 0.0)
            F[m, 1] += fn * ny
            F[m, 2] += fn * nz

        # end caps at |x| = L/2
        limx = self.L / 2.0 - self.radius
        penx = np.abs(self.pos[:, 0]) - limx
        mx = penx > 0.0
        if np.any(mx):
            sgn = np.sign(self.pos[mx, 0])
            vxs = self.vel[mx, 0] * sgn
            fx = -self.k_wall * penx[mx] - self.c_wall * np.maximum(vxs, 0.0)
            F[mx, 0] += fx * sgn

        # particle-particle viscoelastic normal contacts (cached neighbor pairs)
        if self._pair_age >= self.pair_refresh:
            tree = cKDTree(self.pos)
            pairs = tree.query_pairs(r=self.pair_radius, output_type="ndarray")
            if pairs.size:
                self._pi = pairs[:, 0].astype(np.int64).copy()
                self._pj = pairs[:, 1].astype(np.int64).copy()
            else:
                self._pi = np.zeros(0, dtype=np.int64)
                self._pj = np.zeros(0, dtype=np.int64)
            self._pair_age = 0
        else:
            self._pair_age += 1

        if self._pi.size:
            ia, ja = self._pi, self._pj
            d = self.pos[ja] - self.pos[ia]
            dist = np.sqrt(np.einsum("ij,ij->i", d, d))
            ov = self.radius[ia] + self.radius[ja] - dist
            act = (ov > 0.0) & (dist > 1e-9)
            if np.any(act):
                i2, j2 = ia[act], ja[act]
                g = d[act] / dist[act][:, None]
                vn = np.einsum("ij,ij->i", self.vel[i2] - self.vel[j2], g)
                fmag = self.k_pair * ov[act] + self.c_pair * vn
                fx, fy, fz = -fmag * g[:, 0], -fmag * g[:, 1], -fmag * g[:, 2]
                F[:, 0] += np.bincount(i2, weights=fx, minlength=self.N) \
                           + np.bincount(j2, weights=-fx, minlength=self.N)
                F[:, 1] += np.bincount(i2, weights=fy, minlength=self.N) \
                           + np.bincount(j2, weights=-fy, minlength=self.N)
                F[:, 2] += np.bincount(i2, weights=fz, minlength=self.N) \
                           + np.bincount(j2, weights=-fz, minlength=self.N)
        return F

    # ------------------------------------------------------------ time step
    def _step(self, dt: float) -> None:
        u_particle, _, _ = self._axial_flow()
        F_body = self._body_force()
        F_con = self._contacts()

        # semi-implicit Stokes drag toward the local fluid velocity
        beta = dt / self.tau
        acc = (F_body + F_con) / self.mass[:, None]
        u_vec = np.zeros((self.N, 3))
        u_vec[:, 0] = u_particle
        self.vel = (self.vel + dt * acc + beta[:, None] * u_vec) / (1.0 + beta)[:, None]

        # velocity cap keeps the scheme robust
        sp = np.sqrt(np.einsum("ij,ij->i", self.vel, self.vel))
        fast = sp > 3.0
        if np.any(fast):
            self.vel[fast] *= (3.0 / sp[fast])[:, None]

        self.pos += dt * self.vel

        # hard confinement inside the frozen drum
        r = self._r_perp()
        rmax = self.R - 0.2 * self.radius
        over = r > rmax
        if np.any(over):
            fac = (rmax[over] / r[over])
            self.pos[over, 1] *= fac
            self.pos[over, 2] *= fac
        xmax = self.L / 2.0 - 0.2 * self.radius
        ox = np.abs(self.pos[:, 0]) > xmax
        if np.any(ox):
            self.pos[ox, 0] = np.sign(self.pos[ox, 0]) * xmax[ox]

        self.t += dt

    # ---------------------------------------------------------------- frames
    def advance_to(self, time: float):
        t_target = max(float(time), 0.0)
        while self.t < t_target - 1e-12:
            dt = min(self.dt, t_target - self.t)
            self._step(dt)
        return self._frame(t_target)

    def _frame(self, time: float) -> StateFrame:
        theta = self.omega * self.t
        Rx = self._rot_x(theta)

        pos_lab = self.pos @ Rx.T + self.center
        vrot = self.vel.copy()
        vrot[:, 1] += -self.omega * self.pos[:, 2]   # solid-body rotation term
        vrot[:, 2] += self.omega * self.pos[:, 1]
        v_lab = vrot @ Rx.T

        u_particle, u_mean, _ = self._axial_flow()
        F_body = self._body_force()
        F_con = self._contacts()
        u_vec = np.zeros((self.N, 3))
        u_vec[:, 0] = u_particle
        F_drag = -(self.mass / self.tau)[:, None] * (self.vel - u_vec)
        F_int = F_con + F_drag

        external_force = F_body @ Rx.T
        interaction_force = F_int @ Rx.T

        # diagnostic fluid velocity grid (lab frame): solid-body rotation plus
        # the axial back-flow (axial direction is invariant under the rotation)
        u_cell = u_mean[np.clip(
            ((self.gx + self.L / 2.0) / self.dx_bin).astype(np.int64),
            0, self.n_bins - 1)]
        V = np.zeros(self.grid_shape + (3,))
        V[..., 0] = u_cell[:, None, None]
        V[..., 1] = -self.omega * (self.gz[None, None, :] - self.center[2])
        V[..., 2] = self.omega * self.gy[None, :, None]

        frame = StateFrame(
            time=float(time),
            position=pos_lab,
            velocity=v_lab,
            mass=self.mass.copy(),
            body_id=self.body_id,
            attributes={
                "species_id": self.species.copy(),
                "particle_radius": self.radius.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "active_mask": self.active_mask.copy(),
                "velocity_field": V,
                "cumulative_mass_source": np.zeros(self.K),
                "external_force": external_force,
                "interaction_force": interaction_force,
            },
            scalars={
                "grid_spacing": float(self.grid_h),
                "characteristic_length": 2.0 * self.R,
                "contact_length_scale": float(self.h),
                "reference_velocity": float(self.omega * self.R),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        c = self._last_c_tot
        return {
            "time": float(self.t),
            "n_particles": int(self.N),
            "wall_layer_fraction": float(np.mean(self._r_perp() > self.r_gate)),
            "axial_concentration_min": float(c.min()) if c.size else 0.0,
            "axial_concentration_max": float(c.max()) if c.size else 0.0,
            "max_speed_rotating_frame": float(
                np.max(np.sqrt(np.einsum("ij,ij->i", self.vel, self.vel)))),
        }
