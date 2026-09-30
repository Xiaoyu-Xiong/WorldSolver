"""Self-implemented material point method solver for ground-impact oobleck.

Physical model
--------------
Explicit MLS-MPM (moving least squares MPM with APIC affine transfers) on a
uniform Cartesian grid, integrated with symplectic Euler under a sound-speed
CFL limit.  Each material point carries an elastic deformation gradient ``F``
and an accumulated equivalent plastic strain ``q``.

* Elasticity: compressible Neo-Hookean.  The Kirchhoff stress used for the
  grid force is ``tau = mu (F F^T - I) + lambda ln(J) I``.
* Deviatoric flow: von Mises radial return with a shear-rate dependent yield
  stress that models dilatant jamming of a cornstarch suspension plus
  thixotropic breakdown of the jammed micro-structure,

      sigma_y = sigma_res + sigma_0 exp(-q/q_s) + K gammadot^n exp(-q/q_j)

  so a fast first impact jams and stores elastic energy (rebound), while the
  broken-down material yields on the slower second impact and spreads.  The
  return map is isochoric, hence ``det(F)`` remains the true local volume
  ratio and mass is conserved exactly.
* A small Newtonian deviatoric viscosity regularises sub-grid noise.
* Contact: rigid floor at the frozen ``plane_z`` enforced with separating grid
  boundary conditions, Coulomb friction, and a particle-level non-penetration
  projection; slip conditions on the outer grid walls.

The comparison body (body_id 1) uses the same discretisation with a low,
rate-insensitive Bingham yield stress, so it flows on first contact while the
oobleck target (body_id 2) rebounds first and flows afterwards.
"""
from __future__ import annotations

import math

import numpy as np

from worldsolver.contracts import StateFrame

_IDENT = np.eye(3, dtype=np.float64)
_OFFSETS = tuple((i, j, k) for i in range(3) for j in range(3) for k in range(3))
_SQRT_1P5 = math.sqrt(1.5)
_JAM_CAP = 2.5e5


def _det3(f: np.ndarray) -> np.ndarray:
    """Determinants of a stack of 3x3 matrices."""
    return (
        f[:, 0, 0] * (f[:, 1, 1] * f[:, 2, 2] - f[:, 1, 2] * f[:, 2, 1])
        - f[:, 0, 1] * (f[:, 1, 0] * f[:, 2, 2] - f[:, 1, 2] * f[:, 2, 0])
        + f[:, 0, 2] * (f[:, 1, 0] * f[:, 2, 1] - f[:, 1, 1] * f[:, 2, 0])
    )


def _lame(youngs: float, poisson: float) -> tuple[float, float]:
    mu = youngs / (2.0 * (1.0 + poisson))
    lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam


# Per-body constitutive parameters keyed by the frozen state binding value.
_MATERIALS = {
    1: {  # warm-coral viscoplastic comparison body: yields immediately, flows early
        "youngs": 3.0e6,
        "poisson": 0.30,
        "yield_static": 2.5e3,
        "yield_residual": 2.0e3,
        "rate_coefficient": 2.0e2,
        "rate_exponent": 1.0,
        "q_soften": 1.0e9,
        "q_dejam": 1.0e9,
        "viscosity": 12.0,
    },
    2: {  # saturated yellow-green oobleck: rate jamming plus thixotropic breakdown
        "youngs": 3.0e6,
        "poisson": 0.30,
        "yield_static": 6.0e4,
        "yield_residual": 3.0e3,
        "rate_coefficient": 3.0e3,
        "rate_exponent": 1.0,
        "q_soften": 0.04,
        "q_dejam": 0.20,
        "viscosity": 12.0,
    },
}


class BenchmarkSolver:
    """Deterministic MLS-MPM solver for the frozen oobleck impact scene."""

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = dict(scene.config)
        self.config = config
        initial = config["initial_conditions"]

        frame0 = scene.initial_state()
        self.position = np.ascontiguousarray(np.asarray(frame0.position, dtype=np.float64))
        self.velocity = np.ascontiguousarray(np.asarray(frame0.velocity, dtype=np.float64))
        self.mass = np.ascontiguousarray(np.asarray(frame0.mass, dtype=np.float64))
        self.body_id = np.ascontiguousarray(np.asarray(frame0.body_id, dtype=np.int32))
        self.n = int(self.position.shape[0])

        self.gravity = np.asarray(config["gravity"], dtype=np.float64)
        self.floor_z = float(config["boundary_conditions"]["ground"]["plane_z"])
        self.spacing = float(initial["particle_spacing"])
        self.density = float(initial["density"])
        self.radius = float(initial["sphere_radius"])
        self.particle_radius = 0.5 * self.spacing
        self.contact_length_scale = float(self.spacing)
        self.friction = 0.35

        # Reference particle volume from the frozen density and sampled mass.
        self.volume0 = self.mass / self.density

        # Lagrangian state: elastic deformation gradient and plastic strain.
        self.F = np.broadcast_to(_IDENT, (self.n, 3, 3)).copy()
        self.C = np.zeros((self.n, 3, 3), dtype=np.float64)
        self.plastic_strain = np.zeros(self.n, dtype=np.float64)
        self.strain_rate = np.zeros(self.n, dtype=np.float64)

        self._assign_materials()
        self._build_grid()
        self._choose_timestep()

        self.material_ids = (1, 2)
        self.material_masks = [self.body_id == mid for mid in self.material_ids]
        self.mass_quantity = np.zeros((self.n, len(self.material_ids)), dtype=np.float64)
        for column, mask in enumerate(self.material_masks):
            self.mass_quantity[mask, column] = self.mass[mask]
        self.initial_material_mass = self.mass_quantity.sum(axis=0)
        # No prescribed inflow or outflow in this scene: sources stay zero.
        self.cumulative_mass_source = np.zeros(len(self.material_ids), dtype=np.float64)

        self.external_force = self.mass[:, None] * self.gravity[None, :]
        self.time = 0.0
        self.step_index = 0
        self.total_steps = 0
        self._weight_cache: list | None = None

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------
    def _assign_materials(self) -> None:
        n = self.n
        self.mu = np.empty(n, dtype=np.float64)
        self.lam = np.empty(n, dtype=np.float64)
        self.yield_static = np.empty(n, dtype=np.float64)
        self.yield_residual = np.empty(n, dtype=np.float64)
        self.rate_coefficient = np.empty(n, dtype=np.float64)
        self.rate_exponent = np.empty(n, dtype=np.float64)
        self.q_soften = np.empty(n, dtype=np.float64)
        self.q_dejam = np.empty(n, dtype=np.float64)
        self.viscosity = np.empty(n, dtype=np.float64)

        default = _MATERIALS[2]
        wave_speeds = []
        for body_value in np.unique(self.body_id):
            params = _MATERIALS.get(int(body_value), default)
            mask = self.body_id == body_value
            mu, lam = _lame(params["youngs"], params["poisson"])
            self.mu[mask] = mu
            self.lam[mask] = lam
            self.yield_static[mask] = params["yield_static"]
            self.yield_residual[mask] = params["yield_residual"]
            self.rate_coefficient[mask] = params["rate_coefficient"]
            self.rate_exponent[mask] = params["rate_exponent"]
            self.q_soften[mask] = params["q_soften"]
            self.q_dejam[mask] = params["q_dejam"]
            self.viscosity[mask] = params["viscosity"]
            wave_speeds.append(math.sqrt((lam + 2.0 * mu) / self.density))
        self.wave_speed = float(max(wave_speeds)) if wave_speeds else 1.0
        self.max_viscosity = float(self.viscosity.max()) if self.n else 0.0

    def _build_grid(self) -> None:
        self.dx = 2.2 * self.spacing
        self.inv_dx = 1.0 / self.dx

        lower = self.position.min(axis=0)
        upper = self.position.max(axis=0)
        margin_xy = 1.15
        origin = np.array(
            [
                lower[0] - margin_xy,
                lower[1] - margin_xy,
                self.floor_z - 3.0 * self.dx,
            ],
            dtype=np.float64,
        )
        far = np.array(
            [
                upper[0] + margin_xy,
                upper[1] + margin_xy,
                upper[2] + 4.0 * self.dx,
            ],
            dtype=np.float64,
        )
        self.origin = origin
        counts = np.ceil((far - origin) * self.inv_dx).astype(np.int64) + 1
        self.res = tuple(int(max(c, 8)) for c in counts)
        nx, ny, nz = self.res
        self.node_count = nx * ny * nz
        self.grid_strides = (ny * nz, nz, 1)

        # Node z coordinates (flattened) for floor / wall boundary conditions.
        kk = np.arange(nz, dtype=np.float64)
        node_z = np.broadcast_to(kk[None, None, :], self.res).reshape(-1)
        self.node_z_world = origin[2] + node_z * self.dx
        self.floor_nodes = np.flatnonzero(self.node_z_world <= self.floor_z + 1e-12)

        idx_i = np.arange(nx)
        idx_j = np.arange(ny)
        gi = np.broadcast_to(idx_i[:, None, None], self.res).reshape(-1)
        gj = np.broadcast_to(idx_j[None, :, None], self.res).reshape(-1)
        gk = np.broadcast_to(np.arange(nz)[None, None, :], self.res).reshape(-1)
        self.wall_x = np.flatnonzero((gi < 2) | (gi >= nx - 2))
        self.wall_y = np.flatnonzero((gj < 2) | (gj >= ny - 2))
        self.wall_top = np.flatnonzero(gk >= nz - 2)

        # Keep particles strictly inside the interpolation stencil support.
        pad = 1.5 * self.dx
        self.pos_min = origin + pad
        self.pos_max = origin + (np.asarray(self.res, dtype=np.float64) - 1.0) * self.dx - pad
        self.pos_min[2] = self.floor_z - 0.5 * self.dx

    def _choose_timestep(self) -> None:
        self.sample_dt = float(self.config["sample_dt"])
        cfl = 0.28 * self.dx / max(self.wave_speed, 1e-9)
        if self.max_viscosity > 0.0:
            cfl = min(cfl, 0.25 * self.density * self.dx**2 / self.max_viscosity)
        substeps = int(math.ceil(self.sample_dt / cfl))
        self.substeps = int(min(max(substeps, 1), 140))
        self.dt = self.sample_dt / self.substeps

    # ------------------------------------------------------------------
    # constitutive model
    # ------------------------------------------------------------------
    def _kirchhoff_stress(self) -> np.ndarray:
        """Neo-Hookean Kirchhoff stress plus a Newtonian deviatoric term."""
        f = self.F
        b = f @ np.transpose(f, (0, 2, 1))
        j = np.clip(_det3(f), 0.25, 4.0)
        tau = self.mu[:, None, None] * (b - _IDENT)
        tau += (self.lam * np.log(j))[:, None, None] * _IDENT

        # Rate-of-deformation from the APIC affine velocity gradient.
        d = 0.5 * (self.C + np.transpose(self.C, (0, 2, 1)))
        trace_d = d[:, 0, 0] + d[:, 1, 1] + d[:, 2, 2]
        dev_d = d - (trace_d / 3.0)[:, None, None] * _IDENT
        self.strain_rate = np.sqrt(
            np.maximum(2.0 * np.einsum("nij,nij->n", dev_d, dev_d), 0.0)
        )
        tau += (2.0 * self.viscosity)[:, None, None] * dev_d
        return np.clip(tau, -5.0e7, 5.0e7)

    def _yield_stress(self) -> np.ndarray:
        """Shear-rate hardening (jamming) with thixotropic breakdown."""
        soften = np.exp(-self.plastic_strain / self.q_soften)
        dejam = np.exp(-self.plastic_strain / self.q_dejam)
        rate = np.maximum(self.strain_rate, 0.0)
        jam = self.rate_coefficient * np.power(rate, self.rate_exponent)
        jam = np.minimum(jam, _JAM_CAP)
        return self.yield_residual + self.yield_static * soften + jam * dejam

    def _plastic_return(self) -> None:
        """Isochoric von Mises radial return applied to the elastic F."""
        f = self.F
        b = f @ np.transpose(f, (0, 2, 1))
        tau = self.mu[:, None, None] * (b - _IDENT)
        trace_tau = tau[:, 0, 0] + tau[:, 1, 1] + tau[:, 2, 2]
        dev = tau - (trace_tau / 3.0)[:, None, None] * _IDENT
        von_mises = _SQRT_1P5 * np.sqrt(
            np.maximum(np.einsum("nij,nij->n", dev, dev), 0.0)
        )
        sigma_y = self._yield_stress()

        beta = np.where(
            von_mises > sigma_y,
            1.0 - sigma_y / np.maximum(von_mises, 1e-12),
            0.0,
        )
        beta = np.clip(beta, 0.0, 0.85)
        active = beta > 1e-12
        if not np.any(active):
            return

        # Contract the deviatoric part of F toward the identity, then rescale
        # so the determinant (hence the local volume and mass) is unchanged.
        j_before = _det3(f)
        scale = (1.0 - beta)[:, None, None]
        f_new = np.where(active[:, None, None], _IDENT + scale * (f - _IDENT), f)
        j_new = _det3(f_new)
        ratio = np.where(
            np.abs(j_new) > 1e-9,
            np.clip(j_before / np.where(np.abs(j_new) > 1e-9, j_new, 1.0), 0.2, 5.0),
            1.0,
        )
        f_new *= np.cbrt(ratio)[:, None, None]
        self.F = np.where(np.isfinite(f_new), f_new, f)

        # Equivalent plastic strain increment from the stress relaxation.
        dq = beta * von_mises / (3.0 * np.maximum(self.mu, 1e-9))
        self.plastic_strain = np.minimum(self.plastic_strain + dq, 1.0e3)

    def _update_strain_rate(self) -> None:
        d = 0.5 * (self.C + np.transpose(self.C, (0, 2, 1)))
        trace_d = d[:, 0, 0] + d[:, 1, 1] + d[:, 2, 2]
        dev_d = d - (trace_d / 3.0)[:, None, None] * _IDENT
        self.strain_rate = np.sqrt(
            np.maximum(2.0 * np.einsum("nij,nij->n", dev_d, dev_d), 0.0)
        )

    # ------------------------------------------------------------------
    # MLS-MPM step
    # ------------------------------------------------------------------
    def _step(self) -> None:
        dt = self.dt
        dx = self.dx
        inv_dx = self.inv_dx
        nx, ny, nz = self.res
        s0, s1, _ = self.grid_strides

        tau = self._kirchhoff_stress()
        affine = (-dt * 4.0 * inv_dx * inv_dx * self.volume0)[:, None, None] * tau
        affine += self.mass[:, None, None] * self.C

        gp = (self.position - self.origin) * inv_dx
        base = np.floor(gp - 0.5).astype(np.int64)
        np.clip(base[:, 0], 0, nx - 3, out=base[:, 0])
        np.clip(base[:, 1], 0, ny - 3, out=base[:, 1])
        np.clip(base[:, 2], 0, nz - 3, out=base[:, 2])
        fx = gp - base
        w = (
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        )

        grid_m = np.zeros(self.node_count, dtype=np.float64)
        grid_v = np.zeros((self.node_count, 3), dtype=np.float64)
        indices: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        base_flat = base[:, 0] * s0 + base[:, 1] * s1 + base[:, 2]
        p_momentum = self.mass[:, None] * self.velocity

        for oi, oj, ok in _OFFSETS:
            weight = w[oi][:, 0] * w[oj][:, 1] * w[ok][:, 2]
            idx = base_flat + oi * s0 + oj * s1 + ok
            dpos = np.empty_like(fx)
            dpos[:, 0] = (oi - fx[:, 0]) * dx
            dpos[:, 1] = (oj - fx[:, 1]) * dx
            dpos[:, 2] = (ok - fx[:, 2]) * dx
            contribution = weight[:, None] * (
                p_momentum + np.einsum("nij,nj->ni", affine, dpos)
            )
            grid_m += np.bincount(idx, weights=weight * self.mass, minlength=self.node_count)
            for c in range(3):
                grid_v[:, c] += np.bincount(
                    idx, weights=contribution[:, c], minlength=self.node_count
                )
            indices.append(idx)
            weights.append(weight)

        active = grid_m > 1e-12
        inv_m = np.zeros_like(grid_m)
        inv_m[active] = 1.0 / grid_m[active]
        grid_v *= inv_m[:, None]
        grid_v[active] += dt * self.gravity[None, :]
        self._apply_grid_boundaries(grid_v, active)
        np.nan_to_num(grid_v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        new_v = np.zeros_like(self.velocity)
        new_c = np.zeros_like(self.C)
        for offset, idx, weight in zip(_OFFSETS, indices, weights):
            oi, oj, ok = offset
            gv = grid_v[idx]
            new_v += weight[:, None] * gv
            dpos = np.empty_like(fx)
            dpos[:, 0] = oi - fx[:, 0]
            dpos[:, 1] = oj - fx[:, 1]
            dpos[:, 2] = ok - fx[:, 2]
            new_c += (4.0 * inv_dx * weight)[:, None, None] * np.einsum(
                "ni,nj->nij", gv, dpos
            )

        self.velocity = new_v
        self.C = new_c
        self.F = (_IDENT + dt * new_c) @ self.F
        self._update_strain_rate()
        self._plastic_return()

        self.position = self.position + dt * self.velocity
        np.nan_to_num(self.position, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        # Particle-level non-penetration against the rigid floor.
        floor_limit = self.floor_z + self.particle_radius
        below = self.position[:, 2] < floor_limit
        if np.any(below):
            self.position[below, 2] = floor_limit
            self.velocity[below, 2] = np.maximum(self.velocity[below, 2], 0.0)
        np.clip(self.position, self.pos_min, self.pos_max, out=self.position)
        self._sanitise()
        self.step_index += 1
        self.total_steps += 1

    def _apply_grid_boundaries(self, grid_v: np.ndarray, active: np.ndarray) -> None:
        floor = self.floor_nodes
        if floor.size:
            vn = grid_v[floor, 2]
            approaching = vn < 0.0
            if np.any(approaching):
                sel = floor[approaching]
                normal_impulse = -grid_v[sel, 2]
                grid_v[sel, 2] = 0.0
                tangential = grid_v[sel, :2]
                speed = np.sqrt(np.einsum("ni,ni->n", tangential, tangential))
                scale = np.where(
                    speed > 1e-12,
                    np.maximum(0.0, 1.0 - self.friction * normal_impulse / np.maximum(speed, 1e-12)),
                    0.0,
                )
                grid_v[sel, :2] = tangential * scale[:, None]
        if self.wall_x.size:
            grid_v[self.wall_x, 0] = 0.0
        if self.wall_y.size:
            grid_v[self.wall_y, 1] = 0.0
        if self.wall_top.size:
            grid_v[self.wall_top, 2] = np.minimum(grid_v[self.wall_top, 2], 0.0)
        grid_v[~active] = 0.0

    def _sanitise(self) -> None:
        np.nan_to_num(self.velocity, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.velocity, -60.0, 60.0, out=self.velocity)
        bad = ~np.isfinite(self.F)
        if np.any(bad):
            self.F = np.where(bad, np.broadcast_to(_IDENT, self.F.shape), self.F)
        np.clip(self.F, -8.0, 8.0, out=self.F)
        np.nan_to_num(self.C, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(self.C, -5.0e3, 5.0e3, out=self.C)
        np.nan_to_num(self.plastic_strain, copy=False, nan=0.0, posinf=1.0e3, neginf=0.0)

    # ------------------------------------------------------------------
    # sampling interface
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        target = float(time)
        if target < 0.0:
            raise ValueError("advance_to requires a non-negative time")
        target_step = int(round(target / self.dt))
        while self.step_index < target_step:
            self._step()
        self.time = target
        return self._frame(target)

    def _elastic_energy(self) -> float:
        f = self.F
        b = f @ np.transpose(f, (0, 2, 1))
        trace_b = b[:, 0, 0] + b[:, 1, 1] + b[:, 2, 2]
        j = np.clip(_det3(f), 0.25, 4.0)
        log_j = np.log(j)
        psi = 0.5 * self.mu * (trace_b - 3.0) - self.mu * log_j + 0.5 * self.lam * log_j**2
        return float(np.sum(psi * self.volume0))

    def _frame(self, time: float) -> StateFrame:
        gap = self.position[:, 2] - self.particle_radius - self.floor_z
        gap = np.nan_to_num(gap, nan=0.0, posinf=1.0e3, neginf=-1.0e3)

        speed_sq = np.einsum("ni,ni->n", self.velocity, self.velocity)
        kinetic = float(0.5 * np.sum(self.mass * speed_sq))
        potential = float(
            -np.sum(self.mass * (self.position @ self.gravity))
        )
        elastic = self._elastic_energy()

        oobleck = self.material_masks[1]
        if np.any(oobleck):
            centre = self.position[oobleck].mean(axis=0)
            planar = self.position[oobleck, :2] - centre[None, :2]
            spread = float(np.sqrt(np.einsum("ni,ni->n", planar, planar)).max())
            lowest = float(self.position[oobleck, 2].min())
            highest = float(self.position[oobleck, 2].max())
            centre_z = float(centre[2])
        else:  # pragma: no cover - the frozen scene always has oobleck samples
            spread = 0.0
            lowest = highest = centre_z = 0.0

        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": self.F.copy(),
                "mass_quantity": self.mass_quantity.copy(),
                "particle_radius": np.full(self.n, self.particle_radius),
                "material_label": self.body_id.copy(),
                "plastic_strain": self.plastic_strain.copy(),
                "equivalent_strain_rate": self.strain_rate.copy(),
                "volume": self.volume0 * np.clip(_det3(self.F), 1e-3, 1e3),
            },
            fields={
                "contact_signed_distance": gap.copy(),
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "external_force": self.external_force.copy(),
            },
            scalars={
                "contact_length_scale": self.contact_length_scale,
                "total_mass": float(self.mass.sum()),
                "kinetic_energy": kinetic,
                "potential_energy": potential,
                "elastic_energy": elastic,
                "total_energy": kinetic + potential + elastic,
                "min_contact_gap": float(gap.min()),
                "oobleck_spread_radius": spread,
                "oobleck_min_height": lowest,
                "oobleck_max_height": highest,
                "oobleck_centre_height": centre_z,
                "timestep": float(self.dt),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        if not hasattr(self, "position"):
            return {"initialized": False}
        return {
            "initialized": True,
            "method": "explicit MLS-MPM (APIC) with rate-dependent von Mises viscoplasticity",
            "particle_count": int(self.n),
            "grid_resolution": list(self.res),
            "grid_spacing": float(self.dx),
            "timestep": float(self.dt),
            "substeps_per_sample": int(self.substeps),
            "total_steps": int(self.total_steps),
            "wave_speed": float(self.wave_speed),
            "simulated_time": float(self.time),
            "final_max_plastic_strain": float(self.plastic_strain.max()),
            "total_mass": float(self.mass.sum()),
        }
