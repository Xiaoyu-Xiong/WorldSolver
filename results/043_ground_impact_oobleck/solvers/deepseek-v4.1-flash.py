"""Ground-impact oobleck: a self-contained Material Point Method solver.

The frozen scaffold only supplies initial particle samples, geometry and the
trajectory contract; no solver backend is available in this environment, so
the continuum model of the source paper (an MPM for shear-dependent flows) is
discretized directly with NumPy:

* MLS-MPM on a quadratic B-spline background grid: particle-to-grid mass,
  momentum and pressure-gradient transfer, gravity update on the grid,
  grid-to-particle affine velocity and velocity-gradient reconstruction.
* Deformation gradient advection ``F <- (I + dt * grad v) F`` plus a
  deviatoric J2 radial-return plastic correction.
* Compressible pressure closure ``p = E * max(0, 1 - det F)`` whose gradient
  is transferred to the grid (the floor impact stores compression energy and
  releases it as the rebound).
* Shear-rate dependent yield stress with a thixotropic structure memory:
  the fast first impact locks the material (high yield stress, elastic
  rebound), the accumulated compressive work fluidizes the structure, and
  the slower second impact flows and spreads.
* Frictional, non-penetrating contact against the frozen floor box
  (half-extents 2.8 x 2.8 x 0.04, top face at z = 0) and mutual
  non-penetration of the two dynamic bodies.

Body 2 (right, saturated yellow-green) is the oobleck target; body 1 (left,
warm coral) is the viscoplastic comparison body of the source figure.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# Frozen contact normalization length (mean particle / element size).
CONTACT_LENGTH_SCALE = 0.045

_GRAVITY_DEFAULT = [0.0, 0.0, -9.81]

# Per-material rheology and contact constants (SI).
_OOBLECK = {
    "bulk_modulus": 1.6e5,   # compression stiffness [Pa]
    "yield_0": 9000.0,       # intact-structure yield stress [Pa]
    "flow_viscosity": 0.45,  # low-shear dynamic viscosity [Pa*s]
    "thicken": 3.0,          # shear-thickening coefficient
    "thicken_n": 1.6,        # shear-thickening exponent (>1)
    "damping": 8.0,          # compressive damping rate [1/s]
    "friction": 0.22,        # floor friction coefficient
    "restitution": 0.02,     # floor restitution (body elasticity drives rebound)
    "fluidization": 1.25,    # structure breakdown per unit compressive work
}
_CORAL = {
    "bulk_modulus": 4.0e4,
    "yield_0": 120.0,
    "flow_viscosity": 6.0,
    "thicken": 0.0,
    "thicken_n": 1.0,
    "damping": 260.0,
    "friction": 0.34,
    "restitution": 0.02,
    "fluidization": 4.0,
}

_DT_STRAIN_CAP = 0.35      # bounds the per-step deformation increment
_DET_BOUNDS = (0.35, 3.0)
_INTERNAL_DT = 0.00125     # physical sub-step of the time integrator
_MAX_SPEED = 40.0          # numerical bound on reconstructed velocity
_MAX_GRAD = 1.5e3          # numerical bound on the velocity gradient


class BenchmarkSolver:
    """Deterministic NumPy MPM solver for the ground-impact oobleck scene."""

    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        config = scene.config
        initial = config["initial_conditions"]

        self.sample_dt = float(config["sample_dt"])
        self.duration = float(config["duration"])
        self.gravity = np.asarray(
            config.get("gravity", _GRAVITY_DEFAULT), dtype=np.float64
        )
        self.floor_z = float(config["boundary_conditions"]["ground"]["plane_z"])

        frame = scene.initial_state()
        self.position = np.array(frame.position, dtype=np.float64)
        self.velocity = np.array(frame.velocity, dtype=np.float64)
        self.mass = np.array(frame.mass, dtype=np.float64)
        self.body_id = np.array(frame.body_id, dtype=np.int32)
        self.rest_position = np.array(
            frame.attributes.get("rest_position", frame.position), dtype=np.float64
        )
        self.density = float(initial["density"])
        self.volume = self.mass / self.density
        self.radius = 0.5 * float(initial["particle_spacing"])
        n = len(self.position)

        self.deformation = np.tile(np.eye(3, dtype=np.float64), (n, 1, 1))
        self.fluidization = np.zeros(n, dtype=np.float64)
        self.mass_quantity = self.mass.copy()
        self.volume_quantity = self.volume.copy()
        self.n_bodies = int(self.body_id.max()) if n else 0

        # Gravity is the only external force (paired internal contacts excluded).
        self.external_force = np.zeros((n, 3), dtype=np.float64)
        self.external_force[:, 2] = self.mass * self.gravity[2]
        self.cumulative_mass_source = np.zeros(2, dtype=np.float64)

        # Background grid: deterministic, covers the floor and both bodies.
        self.grid_dx = CONTACT_LENGTH_SCALE
        self.grid_low = np.array([-1.7, -1.1, self.floor_z], dtype=np.float64)
        high = np.array([1.7, 1.1, 2.7], dtype=np.float64)
        self.grid_shape = np.maximum(
            np.ceil((high - self.grid_low) / self.grid_dx).astype(np.int64) + 1, 3
        )
        self.ngrid = int(np.prod(self.grid_shape))

        # Physical sub-stepping of the integrator (stable explicit scheme).
        self.substeps = max(int(round(self.sample_dt / _INTERNAL_DT)), 1)
        self.dt = self.sample_dt / self.substeps
        self.step_index = 0

        self.time = 0.0
        self._initial_spread = self._spread_radius()

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        """Advance the physical time integrator to ``time`` and sample it."""
        target = float(time)
        target_step = int(round(target / self.dt))
        while self.step_index < target_step:
            self._step()
            self.step_index += 1
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        vel = self.velocity
        return {
            "solver": "mls_mpm_shear_thickening_compressible",
            "integration_substeps": int(self.substeps),
            "internal_dt": float(self.dt),
            "particle_count": int(len(self.position)),
            "time": float(self.time),
            "min_position_z": float(np.min(self.position[:, 2])),
            "spread_radius": float(max(self._spread_radius(), self._initial_spread)),
            "kinetic_energy": float(
                0.5 * np.sum(self.mass * np.einsum("ij,ij->i", vel, vel))
            ),
            "total_mass": float(self.mass_quantity.sum()),
            "max_deformation_norm": float(
                np.max(np.linalg.norm(self.deformation, axis=(1, 2)))
            ),
            "mean_fluidization": float(self.fluidization.mean()),
            "gravity_z": float(self.gravity[2]),
        }

    # ------------------------------------------------------------------
    # physical time step
    # ------------------------------------------------------------------
    def _step(self) -> None:
        dt = self.dt
        shape = self.grid_shape
        mass = self.mass
        cidx, cw, cgrad = self._transfer_weights()

        pw = cw * mass[:, None]
        gmass = np.zeros(self.ngrid, dtype=np.float64)
        gmom = np.zeros((self.ngrid, 3), dtype=np.float64)
        np.add.at(gmass, cidx, pw)
        np.add.at(gmom, cidx, pw[:, :, None] * self.velocity[:, None, :])

        # external (gravity) force lumped onto the grid
        gfrc = np.zeros((self.ngrid, 3), dtype=np.float64)
        np.add.at(gfrc, cidx, pw[:, :, None] * self.gravity[None, None, :])
        # internal pressure-gradient force: -V * p * grad(w)
        pressure = self._pressure()
        np.add.at(
            gfrc, cidx,
            -self.volume[:, None, None] * pressure[:, None, None] * cgrad,
        )

        safe = gmass > 1e-14
        gvel = np.zeros_like(gmom)
        gvel[safe] = gmom[safe] / gmass[safe, None]
        gvel[safe] += dt * gfrc[safe] / gmass[safe, None]

        self.velocity = np.einsum("nk,nkc->nc", cw, gvel[cidx])
        grad_v = np.einsum("nk,nki,nkj->nij", cw, cgrad, gvel[cidx])

        # Numerical safety: bound the reconstructed fields so the explicit
        # scheme can never emit non-finite samples.
        speed = np.linalg.norm(self.velocity, axis=1)
        overspeed = speed > _MAX_SPEED
        if np.any(overspeed):
            self.velocity[overspeed] *= (_MAX_SPEED / speed[overspeed])[:, None]
        bad_v = ~np.isfinite(self.velocity)
        if np.any(bad_v):
            self.velocity[bad_v] = 0.0
        bad_g = ~np.isfinite(grad_v)
        if np.any(bad_g):
            grad_v[bad_g] = 0.0
        grad_norm = np.linalg.norm(grad_v, axis=(1, 2))
        hot = grad_norm > _MAX_GRAD
        if np.any(hot):
            grad_v[hot] *= (_MAX_GRAD / grad_norm[hot])[:, None, None]

        # deformation gradient advection with a bounded strain increment
        strain = grad_v * dt
        norm = np.linalg.norm(strain, axis=(1, 2))
        scale = np.where(norm > _DT_STRAIN_CAP, _DT_STRAIN_CAP / np.maximum(norm, 1e-12), 1.0)
        eye = np.eye(3, dtype=np.float64)[None, :, :]
        self.deformation = self._matmul3(
            eye + strain * scale[:, None, None], self.deformation
        )

        self._structure_evolution(grad_v, dt)
        self._plastic_flow(grad_v, dt)
        self._compressible_damping()
        self._clamp_deformation()

        self._resolve_contacts()
        self.position = self.position + self.velocity * dt

    def _transfer_weights(self):
        """Quadratic B-spline weights and their spatial gradients (1/m) for
        the 27 grid nodes surrounding each particle."""
        shape = self.grid_shape
        base = np.floor((self.position - self.grid_low) / self.grid_dx).astype(np.int64)
        base = np.clip(base, 0, shape - 2)
        frac = (self.position - self.grid_low) / self.grid_dx - base
        wx, dwx = self._axis(frac[:, 0])
        wy, dwy = self._axis(frac[:, 1])
        wz, dwz = self._axis(frac[:, 2])
        w = (wx, wy, wz)
        dw = (dwx / self.grid_dx, dwy / self.grid_dx, dwz / self.grid_dx)

        n = len(self.position)
        cidx = np.empty((n, 27), dtype=np.int64)
        cw = np.empty((n, 27), dtype=np.float64)
        cgrad = np.empty((n, 27, 3), dtype=np.float64)
        k = 0
        for i in range(3):
            for j in range(3):
                for l in range(3):
                    cx = np.clip(base[:, 0] + i - 1, 0, shape[0] - 1)
                    cy = np.clip(base[:, 1] + j - 1, 0, shape[1] - 1)
                    cz = np.clip(base[:, 2] + l - 1, 0, shape[2] - 1)
                    cidx[:, k] = (cx * shape[1] + cy) * shape[2] + cz
                    cw[:, k] = w[0][:, i] * w[1][:, j] * w[2][:, l]
                    cgrad[:, k, 0] = dw[0][:, i] * w[1][:, j] * w[2][:, l]
                    cgrad[:, k, 1] = w[0][:, i] * dw[1][:, j] * w[2][:, l]
                    cgrad[:, k, 2] = w[0][:, i] * w[1][:, j] * dw[2][:, l]
                    k += 1
        return cidx, cw, cgrad

    @staticmethod
    def _axis(f: np.ndarray):
        w = np.empty((len(f), 3), dtype=np.float64)
        w[:, 0] = 0.5 * (1.0 - f) ** 2
        w[:, 1] = 0.75 - (f - 0.5) ** 2
        w[:, 2] = 0.5 * f**2
        dw = np.empty((len(f), 3), dtype=np.float64)
        dw[:, 0] = -(1.0 - f)
        dw[:, 1] = 1.0 - 2.0 * f
        dw[:, 2] = f
        return w, dw

    # ------------------------------------------------------------------
    # rheology
    # ------------------------------------------------------------------
    def _material(self, name: str) -> np.ndarray:
        oob = self.body_id == 2
        return np.where(oob, _OOBLECK[name], _CORAL[name]).astype(np.float64)

    def _pressure(self) -> np.ndarray:
        """Compressible pressure closure p = E * max(0, 1 - det F)."""
        det = np.linalg.det(self.deformation)
        compression = np.clip(1.0 - det, 0.0, 0.75)
        return self._material("bulk_modulus") * compression

    def _structure_evolution(self, grad_v: np.ndarray, dt: float) -> None:
        """Thixotropic memory: accumulated compressive work breaks down the
        jammed structure, lowering the yield stress for later impacts."""
        det = np.linalg.det(self.deformation)
        compression = np.clip(1.0 - det, 0.0, 0.75)
        shear = self._shear_rate(grad_v)
        rate = self._material("fluidization") * compression * (1.0 + shear)
        self.fluidization = np.clip(self.fluidization + rate * dt, 0.0, 1.0)

    def _shear_rate(self, grad_v: np.ndarray) -> np.ndarray:
        D = 0.5 * (grad_v + np.transpose(grad_v, (0, 2, 1)))
        return np.sqrt(2.0 * np.einsum("nij,nij->n", D, D))

    def _plastic_flow(self, grad_v: np.ndarray, dt: float) -> None:
        """Deviatoric J2 radial return: the viscous stress is
        ``tau = mu(|D|) |D|`` with ``mu = mu0 + K |D|^(n-1)`` and the yield
        threshold drops as the structure fluidizes."""
        D = 0.5 * (grad_v + np.transpose(grad_v, (0, 2, 1)))
        trace = np.trace(D, axis1=1, axis2=2) / 3.0
        dev = D - trace[:, None, None] * np.eye(3)[None]
        dev_norm = np.sqrt(np.einsum("nij,nij->n", dev, dev))
        shear = np.sqrt(2.0) * dev_norm

        mu = self._material("flow_viscosity") + self._material("thicken") * np.power(
            np.maximum(shear, 1e-6), self._material("thicken_n") - 1.0
        )
        tau = mu * shear
        yield_stress = self._material("yield_0") * (1.0 - 0.9 * self.fluidization)
        excess = np.clip(1.0 - yield_stress / np.maximum(tau, 1e-12), 0.0, 1.0)

        direction = np.where(
            dev_norm[:, None, None] > 1e-12,
            dev / np.maximum(dev_norm, 1e-12)[:, None, None],
            0.0,
        )
        flow = direction * (excess * dev_norm * dt)[:, None, None]
        norm = np.linalg.norm(flow, axis=(1, 2))
        flow = flow * np.where(norm > 0.5, 0.5 / np.maximum(norm, 1e-12), 1.0)[:, None, None]
        eye = np.eye(3, dtype=np.float64)[None, :, :]
        self.deformation = self._matmul3(eye - flow, self.deformation)

    def _compressible_damping(self) -> None:
        """Dissipation of compressive motion (material-dependent)."""
        det = np.linalg.det(self.deformation)
        compression = np.clip(1.0 - det, 0.0, 0.75)
        lam = self._material("damping") * compression
        factor = np.clip(lam * self.dt, 0.0, 0.5)
        self.velocity *= (1.0 - factor)[:, None]

    def _clamp_deformation(self) -> None:
        det = np.linalg.det(self.deformation)
        bad = (det < _DET_BOUNDS[0]) | (det > _DET_BOUNDS[1])
        if np.any(bad):
            target = np.where(det[bad] < _DET_BOUNDS[0], _DET_BOUNDS[0], _DET_BOUNDS[1])
            factor = (target / np.maximum(np.abs(det[bad]), 1e-12)) ** (1.0 / 3.0)
            self.deformation[bad] *= factor[:, None, None]
        norm = np.linalg.norm(self.deformation, axis=(1, 2))
        big = norm > 6.0
        if np.any(big):
            self.deformation[big] *= (6.0 / norm[big])[:, None, None]

    # ------------------------------------------------------------------
    # contact
    # ------------------------------------------------------------------
    def _resolve_contacts(self) -> None:
        vel = self.velocity
        pos = self.position
        r = self.radius
        pen = (self.floor_z + r) - pos[:, 2]
        touching = pen > 0.0
        if np.any(touching):
            idx = np.nonzero(touching)[0]
            vz = vel[idx, 2]
            rest = self._material("restitution")[idx]
            friction = self._material("friction")[idx]
            incoming = np.abs(np.minimum(vz, 0.0))
            vel[idx, 2] = np.where(vz < 0.0, -rest * vz, vz)
            vt = vel[idx, :2]
            vt_norm = np.linalg.norm(vt, axis=1)
            limit = friction * (1.0 + rest) * incoming
            shrink = np.where(
                vt_norm > 1e-12, np.minimum(1.0, limit / np.maximum(vt_norm, 1e-12)), 0.0
            )
            vel[idx, :2] = vt * (1.0 - shrink)[:, None]
            pos[idx, 2] = self.floor_z + r
        if self.n_bodies > 1:
            self._body_body_contact()

    def _body_body_contact(self) -> None:
        r2 = (1.15 * self.radius) ** 2
        for b in range(1, self.n_bodies + 1):
            for c in range(b + 1, self.n_bodies + 1):
                mb = self.body_id == b
                mc = self.body_id == c
                pb = self.position[mb]
                pc = self.position[mc]
                if len(pb) == 0 or len(pc) == 0:
                    continue
                diff = pb[:, None, :] - pc[None, :, :]
                dist2 = np.einsum("ijk,ijk->ij", diff, diff)
                hits = dist2 < r2
                if not np.any(hits):
                    continue
                pairs = np.argwhere(hits)
                dist = np.sqrt(np.maximum(dist2[hits], 1e-12))
                normal = diff[hits] / dist[:, None]
                overlap = 1.15 * self.radius - dist
                ib = np.nonzero(mb)[0][pairs[:, 0]]
                ic = np.nonzero(mc)[0][pairs[:, 1]]
                self.position[ib] -= 0.5 * overlap[:, None] * normal
                self.position[ic] += 0.5 * overlap[:, None] * normal
                vrel = np.einsum(
                    "ij,ij->i", self.velocity[ib] - self.velocity[ic], normal
                )
                approaching = vrel < 0.0
                if np.any(approaching):
                    j = -1.02 * vrel[approaching] / (
                        1.0 / self.mass[ib[approaching]]
                        + 1.0 / self.mass[ic[approaching]]
                    )
                    impulse = j[:, None] * normal[approaching]
                    self.velocity[ib[approaching]] += (
                        impulse / self.mass[ib[approaching]][:, None]
                    )
                    self.velocity[ic[approaching]] -= (
                        impulse / self.mass[ic[approaching]][:, None]
                    )

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------
    def _spread_radius(self) -> float:
        mask = self.body_id == 2
        if not np.any(mask):
            return 0.0
        pts = self.position[mask]
        center = pts.mean(axis=0)
        return float(
            np.sqrt(np.mean(np.sum((pts[:, :2] - center[:2]) ** 2, axis=1)))
        )

    @staticmethod
    def _matmul3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.einsum("nij,njk->nik", a, b)

    def _frame(self) -> StateFrame:
        pos = np.array(self.position, dtype=np.float64)
        vel = np.array(self.velocity, dtype=np.float64)
        gap = pos[:, 2] - (self.floor_z + self.radius)
        frame = StateFrame(
            time=float(self.time),
            position=pos,
            velocity=vel,
            mass=np.array(self.mass_quantity, dtype=np.float64),
            body_id=np.array(self.body_id, dtype=np.int32),
            attributes={
                "deformation_gradient": np.array(self.deformation, dtype=np.float64),
                "mass_quantity": np.array(self.mass_quantity, dtype=np.float64),
                "particle_radius": np.full(len(pos), self.radius),
                "rest_position": np.array(self.rest_position, dtype=np.float64),
            },
            fields={
                "contact_signed_distance": np.array(gap, dtype=np.float64),
                "cumulative_mass_source": np.array(
                    self.cumulative_mass_source, dtype=np.float64
                ),
                "external_force": np.array(self.external_force, dtype=np.float64),
            },
            scalars={
                "contact_length_scale": CONTACT_LENGTH_SCALE,
                "total_mass": float(self.mass_quantity.sum()),
                "kinetic_energy": float(
                    0.5 * np.sum(self.mass * np.einsum("ij,ij->i", vel, vel))
                ),
                "gravity_z": float(self.gravity[2]),
            },
        )
        frame.validate()
        return frame
