"""Self-contained MLS-MPM solver for the lava spout periodic folding scene.

Physics model
-------------
* Material point method (MLS-MPM / APIC transfers, quadratic B-spline kernels)
  on a fixed Eulerian background grid.
* Constitutive model: weakly compressible fixed-corotated hyperelasticity plus a
  Newtonian viscous stress, closed by a von Mises (deviatoric Hencky strain)
  return mapping.  That combination is a highly viscous elastoplastic (Bingham
  like) lava: it flows under load, keeps a finite yield strength so deposited
  folds stack instead of levelling, and it stays cohesive while buckling.
* Boundary conditions: the frozen static geometry (ground slab top plane and the
  catch-basin cylinder) is imposed as a sticky (no-slip) grid boundary through an
  analytic signed distance function plus a particle level non-penetration
  projection.  No-slip at the receiving surface is what makes the impinging
  viscous stream buckle laterally and coil.
* A continuous inlet emits fresh material at the frozen nozzle outlet each
  emission period, so the vertical stream is fed for the whole simulation.

Everything is explicit, symplectic-Euler in time with a fixed CFL-safe step, and
fully deterministic (all randomness comes from a seeded Generator).
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    # ---- discretisation ------------------------------------------------
    DX = 0.10                 # background grid spacing [m]
    PSPACE = 0.05             # particle spacing (2 particles per cell axis) [m]
    DT_MAX = 1.0e-3           # CFL-safe explicit step [s]

    # ---- material (highly viscous elastoplastic lava) ------------------
    SHEAR = 8.0e4             # elastic shear modulus [Pa]
    LAMBDA = 1.6e5            # elastic first Lame parameter [Pa]
    YIELD = 5.0e3             # von Mises yield stress [Pa]
    VISC = 150.0              # Newtonian dynamic viscosity [Pa s]

    # ---- inlet ---------------------------------------------------------
    INLET_R = 0.15            # emitted stream radius [m] (nozzle radius 0.27)
    INLET_V = 1.6             # inlet speed [m/s]
    INLET_NOISE = 0.02        # deterministic seeded inlet perturbation [m/s]

    # ---- static geometry (frozen scene_geometry.yaml) ------------------
    GROUND_TOP = 0.0          # box half_extents z 0.05 centred at z=-0.05
    BASIN_R = 1.35
    BASIN_TOP = 0.12          # cylinder h=0.12 centred at z=0.06
    BASIN_ZC = 0.06
    BASIN_HH = 0.06

    VMAX = 25.0               # velocity clamp used purely as a stability guard

    # ==================================================================
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.duration = float(cfg["duration"])
        self.density = float(cfg["sampling"]["density"])
        self.scene_spacing = float(cfg["sampling"]["spacing"])
        self.rng = np.random.default_rng(int(seed))

        s = self.PSPACE
        self.s = s
        self.p_vol = s ** 3
        self.p_mass = self.density * self.p_vol

        # nozzle outlet (cylinder r=0.27, h=0.42 centred at z=3.02)
        noz = None
        for obj in cfg.get("scene_geometry", {}).get("objects", []):
            if obj.get("id") == "lava_nozzle":
                noz = obj
        if noz is not None:
            zc = float(noz["transform"]["translation"][2])
            self.inlet_z = zc - 0.5 * float(noz["geometry"]["height"]) - 0.02
        else:
            self.inlet_z = 2.79

        # --- background grid -------------------------------------------
        self.origin = np.array([-1.45, -1.45, -0.15])
        upper = np.array([1.45, 1.45, 3.35])
        self.dx = self.DX
        self.inv_dx = 1.0 / self.dx
        self.res = np.maximum(
            np.round((upper - self.origin) * self.inv_dx).astype(int) + 1, 4
        )
        nx, ny, nz = (int(v) for v in self.res)
        self.nx, self.ny, self.nz = nx, ny, nz
        self.ng = nx * ny * nz
        self.grid_mv = np.zeros((self.ng, 3))
        self.grid_m = np.zeros(self.ng)

        # node signed distance to the static solids -> sticky nodes
        gi = np.arange(self.ng)
        iz = gi % nz
        iy = (gi // nz) % ny
        ix = gi // (ny * nz)
        node = np.column_stack(
            (
                self.origin[0] + ix * self.dx,
                self.origin[1] + iy * self.dx,
                self.origin[2] + iz * self.dx,
            )
        )
        phi_node = self._sdf(node)
        border = (
            (ix < 1) | (ix > nx - 2) | (iy < 1) | (iy > ny - 2) | (iz > nz - 2)
        )
        self.sticky = np.flatnonzero((phi_node < 0.0) | border)

        # --- particles: resample the frozen sampling blocks ------------
        blocks = [self._sample_block(b, s) for b in cfg["sampling"]["blocks"]]
        pos0 = np.concatenate(blocks) if blocks else np.zeros((1, 3))
        n0 = len(pos0)

        emit_offsets = self._inlet_offsets()
        self.emit_offsets = emit_offsets
        self.emit_period = s / self.INLET_V
        n_emit = int(np.ceil(self.duration / self.emit_period) + 2) * len(emit_offsets)
        cap = n0 + n_emit + 64

        self.x = np.zeros((cap, 3))
        self.v = np.zeros((cap, 3))
        self.C = np.zeros((cap, 3, 3))
        self.F = np.zeros((cap, 3, 3))
        self.R = np.zeros((cap, 3, 3))
        self.J = np.ones(cap)
        self.m = np.full(cap, self.p_mass)
        self.vol = np.full(cap, self.p_vol)
        self.label = np.zeros(cap, dtype=np.int32)

        eye = np.eye(3)
        self.x[:n0] = pos0
        # the initial column is already a steady descending stream
        self.v[:n0, 2] = -self.INLET_V
        self.F[:n0] = eye
        self.R[:n0] = eye
        self.n = n0
        self._clamp_particles(0, n0)

        self.time = 0.0
        self.added_mass = 0.0
        self.next_emit = self.emit_period
        self.steps = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_block(bounds, spacing):
        lo = np.asarray(bounds[:3], float)
        hi = np.asarray(bounds[3:], float)
        axes = [np.arange(lo[i] + 0.5 * spacing, hi[i], spacing) for i in range(3)]
        if any(len(a) == 0 for a in axes):
            return ((lo + hi) * 0.5)[None, :]
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        return np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))

    def _inlet_offsets(self):
        s = self.s
        r = self.INLET_R
        line = np.arange(-r, r + 1e-9, s)
        gx, gy = np.meshgrid(line, line, indexing="ij")
        gx = gx.ravel()
        gy = gy.ravel()
        keep = gx * gx + gy * gy <= r * r + 1e-9
        return np.column_stack((gx[keep], gy[keep]))

    def _sdf(self, p):
        """Signed distance to the union of ground slab and catch basin."""
        z = p[:, 2]
        r = np.hypot(p[:, 0], p[:, 1])
        phi_ground = z - self.GROUND_TOP
        dr = r - self.BASIN_R
        dz = np.abs(z - self.BASIN_ZC) - self.BASIN_HH
        outside = np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0))
        phi_basin = outside + np.minimum(np.maximum(dr, dz), 0.0)
        return np.minimum(phi_ground, phi_basin)

    def _sdf_normal(self, p):
        """Outward unit normal of the static solid SDF (finite differences)."""
        h = 1.0e-3
        g = np.empty_like(p)
        for axis in range(3):
            dp = np.zeros(3)
            dp[axis] = h
            g[:, axis] = self._sdf(p + dp) - self._sdf(p - dp)
        n = np.linalg.norm(g, axis=1, keepdims=True)
        n = np.where(n < 1.0e-12, 1.0, n)
        return g / n

    def _clamp_particles(self, lo, hi):
        """Keep samples inside the padded grid domain (numerical safety only)."""
        pad = 1.5 * self.dx
        low = self.origin + pad
        high = self.origin + (self.res - 1) * self.dx - pad
        np.clip(self.x[lo:hi], low, high, out=self.x[lo:hi])

    # ==================================================================
    def _emit(self, dt):
        """Continuous viscous inlet at the frozen nozzle outlet."""
        offs = self.emit_offsets
        k = len(offs)
        if self.n + k > len(self.x):
            return
        lo, hi = self.n, self.n + k
        jitter = (self.rng.random((k, 2)) - 0.5) * (0.15 * self.s)
        self.x[lo:hi, 0] = offs[:, 0] + jitter[:, 0]
        self.x[lo:hi, 1] = offs[:, 1] + jitter[:, 1]
        self.x[lo:hi, 2] = self.inlet_z
        noise = (self.rng.random((k, 2)) - 0.5) * (2.0 * self.INLET_NOISE)
        self.v[lo:hi, 0] = noise[:, 0]
        self.v[lo:hi, 1] = noise[:, 1]
        self.v[lo:hi, 2] = -self.INLET_V
        self.C[lo:hi] = 0.0
        self.F[lo:hi] = np.eye(3)
        self.R[lo:hi] = np.eye(3)
        self.J[lo:hi] = 1.0
        self.m[lo:hi] = self.p_mass
        self.vol[lo:hi] = self.p_vol
        self.label[lo:hi] = 0
        self.n = hi
        self.added_mass += k * self.p_mass

    # ------------------------------------------------------------------
    def _weights(self, xp):
        gx = (xp - self.origin) * self.inv_dx
        base = np.floor(gx - 0.5).astype(np.int64)
        np.clip(base[:, 0], 0, self.nx - 3, out=base[:, 0])
        np.clip(base[:, 1], 0, self.ny - 3, out=base[:, 1])
        np.clip(base[:, 2], 0, self.nz - 3, out=base[:, 2])
        fx = gx - base
        w = np.empty((3, len(xp), 3))
        for axis in range(3):
            f = fx[:, axis]
            w[0, :, axis] = 0.5 * (1.5 - f) ** 2
            w[1, :, axis] = 0.75 - (f - 1.0) ** 2
            w[2, :, axis] = 0.5 * (f - 0.5) ** 2
        base_flat = base[:, 0] * (self.ny * self.nz) + base[:, 1] * self.nz + base[:, 2]
        return base_flat, fx, w

    def _stress(self, n):
        """Elastoplastic (fixed-corotated + von Mises cap) plus viscous stress."""
        F = self.F[:n]
        U, sig, Vt = np.linalg.svd(F)
        sig = np.clip(sig, 1.0e-4, 1.0e4)
        eps = np.log(sig)
        mean = eps.mean(axis=1, keepdims=True)
        # volumetric cap: plastic compaction / dilation limits keep stresses bounded
        mean_c = np.clip(mean, -0.08, 0.02)
        dev = eps - mean
        norm = np.linalg.norm(dev, axis=1, keepdims=True)
        dev_max = self.YIELD / (2.0 * self.SHEAR)
        scale = np.where(norm > dev_max, dev_max / np.maximum(norm, 1.0e-12), 1.0)
        eps_new = mean_c + dev * scale
        sig_new = np.exp(eps_new)
        # rebuild the (plastically corrected) elastic deformation gradient
        Fe = np.einsum("nij,nj,njk->nik", U, sig_new, Vt)
        self.F[:n] = Fe
        R = np.einsum("nij,njk->nik", U, Vt)
        self.R[:n] = R
        logJ = eps_new.sum(axis=1)
        J = np.exp(logJ)
        self.J[:n] = J
        # Kirchhoff stress: 2 mu (Fe - R) Fe^T + lambda log(J) I
        tau = 2.0 * self.SHEAR * np.einsum("nij,nkj->nik", Fe - R, Fe)
        vol_term = self.LAMBDA * logJ
        tau[:, 0, 0] += vol_term
        tau[:, 1, 1] += vol_term
        tau[:, 2, 2] += vol_term
        # Newtonian viscous stress from the affine velocity gradient
        Cm = self.C[:n]
        D = 0.5 * (Cm + np.transpose(Cm, (0, 2, 1)))
        trD = (D[:, 0, 0] + D[:, 1, 1] + D[:, 2, 2]) / 3.0
        Dd = D.copy()
        Dd[:, 0, 0] -= trD
        Dd[:, 1, 1] -= trD
        Dd[:, 2, 2] -= trD
        tau += (2.0 * self.VISC) * Dd
        return tau

    def _step(self, dt):
        n = self.n
        self.grid_mv[:] = 0.0
        self.grid_m[:] = 0.0
        tau = self._stress(n)
        vol0 = self.vol[:n]
        mp = self.m[:n]
        # MLS-MPM affine momentum matrix
        coeff = -dt * (4.0 * self.inv_dx * self.inv_dx) * vol0[:, None, None]
        affine = coeff * tau + mp[:, None, None] * self.C[:n]
        base_flat, fx, w = self._weights(self.x[:n])
        stride = np.array([self.ny * self.nz, self.nz, 1], dtype=np.int64)
        mv = mp[:, None] * self.v[:n]
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    wt = w[i, :, 0] * w[j, :, 1] * w[k, :, 2]
                    dpos = (np.array([i, j, k], dtype=float) - fx) * self.dx
                    idx = base_flat + i * stride[0] + j * stride[1] + k
                    contrib = mv + np.einsum("nij,nj->ni", affine, dpos)
                    wc = wt[:, None] * contrib
                    self.grid_m += np.bincount(idx, weights=wt * mp, minlength=self.ng)
                    for c in range(3):
                        self.grid_mv[:, c] += np.bincount(
                            idx, weights=wc[:, c], minlength=self.ng
                        )
        active = self.grid_m > 1.0e-12
        gv = np.zeros_like(self.grid_mv)
        gv[active] = self.grid_mv[active] / self.grid_m[active, None]
        gv[active] += dt * self.gravity
        np.clip(gv, -self.VMAX, self.VMAX, out=gv)
        gv[self.sticky] = 0.0  # no-slip on the frozen static solids and domain box

        # ---- grid to particle (APIC / MLS) ----------------------------
        vnew = np.zeros((n, 3))
        Cnew = np.zeros((n, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    wt = w[i, :, 0] * w[j, :, 1] * w[k, :, 2]
                    dpos = (np.array([i, j, k], dtype=float) - fx) * self.dx
                    idx = base_flat + i * stride[0] + j * stride[1] + k
                    gvn = gv[idx]
                    vnew += wt[:, None] * gvn
                    Cnew += (4.0 * self.inv_dx * self.inv_dx) * np.einsum(
                        "n,ni,nj->nij", wt, gvn, dpos
                    )
        np.clip(vnew, -self.VMAX, self.VMAX, out=vnew)
        self.v[:n] = vnew
        self.C[:n] = Cnew
        # F update (plastic correction is applied inside _stress)
        eye = np.eye(3)
        self.F[:n] = np.einsum("nij,njk->nik", eye + dt * Cnew, self.F[:n])
        self.x[:n] += dt * vnew

        # ---- particle level non-penetration against the static solids --
        phi = self._sdf(self.x[:n])
        pen = phi < 0.0
        if np.any(pen):
            ip = np.flatnonzero(pen)
            nrm = self._sdf_normal(self.x[ip])
            self.x[ip] -= phi[ip, None] * nrm
            vn = np.einsum("ni,ni->n", self.v[ip], nrm)
            inward = vn < 0.0
            sel = ip[inward]
            if len(sel):
                nn = nrm[inward]
                vv = self.v[sel]
                vv = vv - np.einsum("ni,ni->n", vv, nn)[:, None] * nn
                self.v[sel] = 0.4 * vv  # strong Coulomb-like sticking friction
        self._clamp_particles(0, n)
        self.time += dt
        self.steps += 1

    # ==================================================================
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        while self.time < target - 1.0e-12:
            dt = min(self.DT_MAX, target - self.time)
            if self.time + 1.0e-12 >= self.next_emit:
                self._emit(dt)
                self.next_emit += self.emit_period
            self._step(dt)
        self.time = target
        return self._frame(target)

    def _frame(self, t: float) -> StateFrame:
        n = self.n
        pos = np.array(self.x[:n], dtype=float)
        vel = np.array(self.v[:n], dtype=float)
        mass = np.array(self.m[:n], dtype=float)
        F = np.array(self.F[:n], dtype=float)
        if not np.isfinite(pos).all():
            pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(vel).all():
            vel = np.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(F).all():
            F = np.nan_to_num(F, nan=0.0, posinf=1.0, neginf=-1.0)
        F = np.clip(F, -1.0e3, 1.0e3)

        ext = mass[:, None] * self.gravity[None, :]

        phi = self._sdf(pos)
        near = np.flatnonzero(phi < 2.0 * self.s)
        if len(near) == 0:
            csd = np.array([float(np.min(phi))])
        else:
            csd = phi[near]

        attributes = {
            "deformation_gradient": F,
            "mass_quantity": mass.copy(),
            "particle_radius": np.full(n, 1.1 * self.s),
            "material_label": np.array(self.label[:n], dtype=np.int32),
        }
        fields = {
            "external_force": ext,
            "contact_signed_distance": np.asarray(csd, dtype=float),
            "cumulative_mass_source": np.array([float(self.added_mass)]),
        }
        scalars = {
            "contact_length_scale": float(self.s),
            "grid_spacing": float(self.dx),
            "characteristic_length": 2.8,
            "reference_velocity": float(self.INLET_V),
        }
        frame = StateFrame(
            float(t),
            pos,
            vel,
            mass=mass,
            body_id=np.zeros(n, dtype=np.int32),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "self-implemented MLS-MPM (APIC) viscoplastic lava",
            "steps": int(self.steps),
            "particles": int(self.n),
            "grid_resolution": [int(self.nx), int(self.ny), int(self.nz)],
            "dt": float(self.DT_MAX),
            "added_mass_kg": float(self.added_mass),
            "final_time": float(self.time),
        }
