"""Physically-based foam tendril tearing solver (damage MPM).

A fixed-corotated elastic MPM discretization of the shaving-foam bridge with
Wolper-style scalar damage: once the accumulated tensile plastic strain in the
neck exceeds a failure threshold the particle loses all cohesion (stress -> 0),
producing a genuine tear into two disconnected fragments (no thread artifact).
Gravity, the prescribed initial pull velocities, floor contact, and adhesive
contact with the static upper plate drive the deformation.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

# ---------------- material / numerical parameters ----------------
RHO0 = 180.0            # bulk density (kg/m^3), from scene config
YOUNG = 2.0e5           # Pa
POISSON = 0.3
MU = YOUNG / (2.0 * (1.0 + POISSON))
LAMBDA = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
LAM_YIELD = 1.18        # tensile stretch at which damage starts flowing
LAM_COMP = 0.85         # compressive clamp
D_FAIL = 0.25           # plastic stretch integral to full loss of cohesion
DT_SUB = 8.0e-4         # substep (CFL ~ 0.34 for c_p ~ 39 m/s, h = 0.09)

H = 0.09                # grid spacing = 2 * particle spacing
ORIGIN = np.array([-0.855, -0.855, -0.045])
NX, NY, NZ = 20, 20, 24

FLOOR_Z = 0.0
PLATE_Z = 1.84          # underside of static upper plate
PLATE_HX, PLATE_HY = 0.78, 0.62
STICK_LO = 1.74         # adhesive band under plate (foam stuck to plate)


class BenchmarkSolver:
    # ------------------------- setup -------------------------
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=np.float64)
        self.v = np.array(frame0.velocity, dtype=np.float64)
        self.m = np.array(frame0.mass, dtype=np.float64)
        self.n = len(self.x)
        attrs = frame0.attributes
        self.particle_radius = np.array(attrs.get(
            "particle_radius", np.full(self.n, 0.026)), dtype=np.float64)
        self.foam_material = np.array(attrs.get(
            "foam_material", np.ones(self.n)), dtype=np.float64)
        self.tear_candidate = np.array(attrs.get(
            "tear_candidate", np.zeros(self.n)), dtype=np.float64)
        self.v0 = self.m / RHO0                      # rest volume
        self.F = np.tile(np.eye(3), (self.n, 1, 1))  # deformation gradient
        self.C = np.zeros((self.n, 3, 3))            # APIC affine matrix
        self.D = np.zeros(self.n)                    # damage in [0, 1]
        self.g = np.array([0.0, 0.0, -9.81])
        self.t = 0.0
        self.tear_time = -1.0

        # static node masks
        gx = ORIGIN[0] + H * np.arange(NX)
        gy = ORIGIN[1] + H * np.arange(NY)
        gz = ORIGIN[2] + H * np.arange(NZ)
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        node_x = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        self.node_coords = node_x
        in_foot = (np.abs(node_x[:, 0]) <= PLATE_HX) & (np.abs(node_x[:, 1]) <= PLATE_HY)
        self.node_floor = node_x[:, 2] <= FLOOR_Z + 1e-9
        self.node_solid = in_foot & (node_x[:, 2] >= PLATE_Z - 1e-9)
        self.node_stick = in_foot & (node_x[:, 2] >= STICK_LO) & (node_x[:, 2] < PLATE_Z - 1e-9)
        self.G = NX * NY * NZ
        offsets = np.array([[i, j, k] for i in (-1, 0, 1)
                            for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=np.int64)
        self.offsets = offsets  # (27,3)

    # ------------------------- MPM substep -------------------------
    def _weights(self):
        q = (self.x - ORIGIN) / H                       # (N,3)
        base = np.floor(q - 0.5).astype(np.int64)
        base = np.clip(base, 1, np.array([NX, NY, NZ]) - 3)
        f = q - base                                     # in [0.5, 1.5)
        w = np.stack((0.5 * (1.5 - f) ** 2,
                      0.75 - (f - 1.0) ** 2,
                      0.5 * (f - 0.5) ** 2), axis=2)     # (N,3,3)
        dw = np.stack((-(1.5 - f),
                       -2.0 * (f - 1.0),
                       (f - 0.5)), axis=2) / H           # d/dx of weights
        wx, wy, wz = w[:, 0], w[:, 1], w[:, 2]
        dx_, dy_, dz_ = dw[:, 0], dw[:, 1], dw[:, 2]
        W = np.einsum("na,nb,nc->nabc", wx, wy, wz).reshape(self.n, 27)
        GX = np.einsum("na,nb,nc->nabc", dx_, wy, wz).reshape(self.n, 27)
        GY = np.einsum("na,nb,nc->nabc", wx, dy_, wz).reshape(self.n, 27)
        GZ = np.einsum("na,nb,nc->nabc", wx, wy, dz_).reshape(self.n, 27)
        grad = np.stack((GX, GY, GZ), axis=2)            # (N,27,3)
        idx = base[:, None, :] + self.offsets[None, :, :]  # (N,27,3)
        flat = (idx[:, :, 0] * NY + idx[:, :, 1]) * NZ + idx[:, :, 2]
        return W, grad, flat

    def _stress(self):
        """Return-mapped fixed corotated stress with scalar damage."""
        Ft = self.F
        Ct = np.einsum("nji,njk->nik", Ft, Ft)
        eig, V = np.linalg.eigh(Ct)
        lam = np.sqrt(np.clip(eig, 1e-12, None))
        # tensile damage accumulation (only above yield stretch)
        dp = np.clip(lam - LAM_YIELD, 0.0, None).sum(axis=1)
        self.D = np.clip(self.D + dp / D_FAIL, 0.0, 1.0)
        lam_p = np.clip(lam, LAM_COMP, LAM_YIELD)
        Fe = np.einsum("nij,nj,nkj->nik", V, lam_p, V)
        # polar factor via Newton iterations
        R = Fe.copy()
        for _ in range(4):
            R = 0.5 * (R + np.linalg.inv(R).transpose(0, 2, 1))
        I3 = np.eye(3)
        FR = Fe - R
        sig = MU * (FR @ Fe.transpose(0, 2, 1))
        sig = sig + (LAMBDA * np.einsum("nii->n", FR))[:, None, None] * I3[None]
        scale = np.where(self.D >= 0.995, 0.0, 1.0 - self.D)
        self.F = Fe
        return sig * scale[:, None, None]

    def _contact_particles(self):
        # floor
        pen = self.x[:, 2] < FLOOR_Z
        if pen.any():
            self.x[pen, 2] = FLOOR_Z
            self.v[pen, 2] = np.maximum(self.v[pen, 2], 0.0)
            self.v[pen, 0] *= 0.85
            self.v[pen, 1] *= 0.85
        # plate underside (foam adheres; cannot penetrate upward)
        foot = ((np.abs(self.x[:, 0]) <= PLATE_HX) &
                (np.abs(self.x[:, 1]) <= PLATE_HY))
        hit = foot & (self.x[:, 2] > PLATE_Z)
        if hit.any():
            self.x[hit, 2] = PLATE_Z
            self.v[hit, 2] = np.minimum(self.v[hit, 2], 0.0)

    def _substep(self):
        W, grad, flat = self._weights()
        sig = self._stress()
        node_pos = self.node_coords[flat]                  # (N,27,3)
        dx = node_pos - self.x[:, None, :]
        # internal force contribution: -V0 * sigma @ gradW
        f_int = -self.v0[:, None, None] * np.einsum("nij,nkj->nik", sig, grad)
        # affine (APIC) momentum + external gravity + internal force
        mom_p = self.m[:, None, None] * (
            self.v[:, None, :] + np.einsum("nij,nkj->nik", self.C, dx))
        contrib = W[:, :, None] * (mom_p + DT_SUB * (
            f_int + self.m[:, None, None] * self.g[None, None, :]))
        mass_c = W * self.m[:, None]

        fi = flat.ravel()
        gm = np.bincount(fi, weights=mass_c.ravel(), minlength=self.G)
        gmv = np.zeros((self.G, 3))
        for d in range(3):
            gmv[:, d] = np.bincount(fi, weights=contrib[:, :, d].ravel(),
                                    minlength=self.G)
        active = gm > 1e-10
        gv = np.zeros((self.G, 3))
        gv[active] = gmv[active] / gm[active, None]
        gv *= 0.999  # slight numerical damping

        # grid-level boundary conditions
        gv[self.node_floor, 2] = np.maximum(gv[self.node_floor, 2], 0.0)
        gv[self.node_solid] = 0.0
        stick = self.node_stick
        gv[stick, 2] = np.maximum(gv[stick, 2], 0.0)  # adhesion: no detachment
        gv[stick, 0] *= 0.85
        gv[stick, 1] *= 0.85

        # G2P
        vg = gv[flat]                                   # (N,27,3)
        self.v = np.einsum("nk,nkd->nd", W, vg)
        self.C = (4.0 / H ** 2) * np.einsum("nk,nkd,nke->nde", W, vg, dx)
        self.x += DT_SUB * self.v
        self.F += DT_SUB * (self.C @ self.F)
        self._contact_particles()
        self.t += DT_SUB

    # ------------------------- interface -------------------------
    def advance_to(self, time: float):
        target = float(time)
        n = int(round((target - self.t) / DT_SUB))
        for _ in range(max(n, 0)):
            self._substep()
            if self.tear_time < 0.0:
                neck = self.tear_candidate > 0.5
                if neck.any() and np.all(self.D[neck] >= 0.99):
                    self.tear_time = self.t
        self.t = target
        return self._make_frame(target)

    def _make_frame(self, time: float) -> StateFrame:
        # signed distance to nearest contact surface for near-contact samples
        floor_sd = self.x[:, 2] - FLOOR_Z
        foot = ((np.abs(self.x[:, 0]) <= PLATE_HX) &
                (np.abs(self.x[:, 1]) <= PLATE_HY))
        plate_sd = np.where(foot, PLATE_Z - self.x[:, 2], np.inf)
        sd = np.minimum(floor_sd, plate_sd)
        near = sd < 0.20
        if not near.any():
            near = np.zeros(self.n, dtype=bool)
            near[int(np.argmin(sd))] = True
        contact_sd = sd[near]

        ext_force = self.m[:, None] * self.g[None, :]
        frame = StateFrame(
            time=time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=np.zeros(self.n, dtype=np.int32),
            attributes={
                "particle_radius": self.particle_radius.copy(),
                "foam_material": self.foam_material.copy(),
                "tear_candidate": self.tear_candidate.copy(),
                "damage": self.D.copy(),
                "deformation_gradient": self.F.copy(),
                "mass_quantity": self.m.copy(),
            },
            fields={
                "contact_signed_distance": contact_sd.astype(np.float64),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "external_force": ext_force.astype(np.float64),
            },
            scalars={
                "contact_length_scale": 0.045,
                "explicit_tearing_enabled": 1.0,
                "max_damage": float(self.D.max()),
                "torn_fraction": float(np.mean(self.D >= 0.99)),
                "tear_time": float(self.tear_time),
                "particle_count": float(self.n),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "n_particles": self.n,
            "time": self.t,
            "max_damage": float(self.D.max()) if hasattr(self, "D") else 0.0,
            "torn_fraction": float(np.mean(self.D >= 0.99)) if hasattr(self, "D") else 0.0,
            "tear_time": float(getattr(self, "tear_time", -1.0)),
            "method": "damage_MPM_fixed_corotated",
        }
