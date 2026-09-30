"""Cloth drape over a fixed horizontal cylinder.

Physically-based explicit spring-mass cloth solver:

* Rectangular cloth grid (17x17) initialized flat above the cylinder.
* Structural / shear / bending distance springs with along-edge damping
  (stretch, shear and bending elasticity of a thin sheet).
* Gravity plus light aerodynamic drag as external forces.
* Penalty contacts (with Coulomb friction, normal damping and a hard
  projection fallback) against the frozen static cylinder and ground box.
* Semi-implicit (symplectic) Euler time stepping with fixed deterministic
  sub-steps; sample frames are produced by interpolation-free stepping to
  the exact requested sample time.
* Per-vertex 2D deformation gradients exported in a local sheet frame.

No animation, keyframes or precomputed trajectories are used: every sample
is the result of numerically integrating the equations of motion.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Mass-spring cloth solver for the cloth-drape-over-cylinder scene."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.p = np.array(frame0.position, dtype=float)
        self.v = np.zeros_like(self.p)
        self.N = int(len(self.p))
        self.mass = np.array(frame0.mass, dtype=float)
        self.boundary = np.array(
            frame0.attributes.get("boundary_vertex",
                                  np.zeros(self.N, dtype=np.int32))
        ).astype(np.int32)

        cloth = cfg["cloth"]
        self.nx, self.ny = (int(cloth["grid_shape"][0]),
                            int(cloth["grid_shape"][1]))
        sx, sy = float(cloth["size"][0]), float(cloth["size"][1])
        self.thickness = float(cloth["thickness"])
        self.g = np.array(cfg["gravity"], dtype=float)
        if self.nx * self.ny != self.N:
            raise ValueError("grid shape does not match vertex count")

        self.dx = sx / (self.nx - 1)
        self.dy = sy / (self.ny - 1)

        # Grid indices (meshgrid "xy" ordering: index = iy*nx + ix).
        self.ix = np.tile(np.arange(self.nx), self.ny)
        self.iy = np.repeat(np.arange(self.ny), self.nx)

        # ------------------------------------------------------ spring network
        # (i, j, rest length, stiffness, damping)
        k_stretch, c_stretch = 4000.0, 0.5
        k_shear, c_shear = 800.0, 0.2
        k_bend, c_bend = 60.0, 0.05

        idx = np.arange(self.N).reshape(self.ny, self.nx)
        I, J, rest, ks, cs = [], [], [], [], []

        def add(a, b, r, k, c):
            I.append(np.asarray(a, dtype=np.int64).ravel())
            J.append(np.asarray(b, dtype=np.int64).ravel())
            n = I[-1].size
            rest.append(np.full(n, float(r)))
            ks.append(np.full(n, float(k)))
            cs.append(np.full(n, float(c)))

        diag = float(np.hypot(self.dx, self.dy))

        # structural (x and y grid lines)
        add(idx[:, :-1], idx[:, 1:], self.dx, k_stretch, c_stretch)
        add(idx[:-1, :], idx[1:, :], self.dy, k_stretch, c_stretch)
        # shear (diagonals)
        add(idx[:-1, :-1], idx[1:, 1:], diag, k_shear, c_shear)
        add(idx[:-1, 1:], idx[1:, :-1], diag, k_shear, c_shear)
        # bending (skip-one neighbors)
        add(idx[:, :-2], idx[:, 2:], 2.0 * self.dx, k_bend, c_bend)
        add(idx[:-2, :], idx[2:, :], 2.0 * self.dy, k_bend, c_bend)

        self.I = np.concatenate(I)
        self.J = np.concatenate(J)
        self.rest = np.concatenate(rest)
        self.k_spring = np.concatenate(ks)
        self.c_spring = np.concatenate(cs)
        self.n_springs = self.I.size

        # ------------------------------------------------------------ contacts
        # Frozen static geometry (scene_geometry.yaml).
        self.cyl_c = np.array([0.0, 0.0, 0.7])          # cylinder centre
        self.cyl_R = 0.52                               # radius
        self.cyl_hh = 1.4                               # half height (axis ~ y)
        self.gnd_c = np.array([0.0, 0.0, -0.05])        # ground box centre
        self.gnd_h = np.array([2.4, 2.0, 0.05])         # ground half extents

        self.k_contact = 2.0e4                          # penalty stiffness
        self.c_contact = 3.0                            # normal damping
        self.mu = 0.4                                   # friction coefficient
        self.contact_margin = 0.5 * self.thickness      # activation margin
        self.proj_tol = 3.0 * self.thickness            # hard projection limit

        # ------------------------------------------------------- integrator
        self.dt = 2.5e-4          # fixed sub-step
        self.drag = 0.2           # aerodynamic drag (1/s)
        self.t = 0.0
        self.n_steps = 0
        self._ready = True

    # --------------------------------------------------------------- contacts
    def _cylinder_sdf(self, p):
        """Signed distance and outward normal to the capped cylinder."""
        q = p - self.cyl_c
        rad = np.sqrt(q[:, 0] ** 2 + q[:, 2] ** 2)
        rad_safe = np.maximum(rad, 1e-12)
        dr = rad - self.cyl_R
        ay = np.abs(q[:, 1])
        dy = ay - self.cyl_hh

        gap = np.where((dr > 0.0) & (dy > 0.0),
                       np.hypot(dr, dy), np.maximum(dr, dy))

        # outward normal
        n = np.zeros_like(p)
        rim = (dr > 0.0) & (dy > 0.0)
        lat = ~rim & (dr >= dy)
        cap = ~rim & (dr < dy)
        # radial direction (x, 0, z)
        ux = q[:, 0] / rad_safe
        uz = q[:, 2] / rad_safe
        n[rim, 0] = (ux[rim] * dr[rim])
        n[rim, 1] = np.sign(q[rim, 1]) * dy[rim]
        n[rim, 2] = (uz[rim] * dr[rim])
        ln = np.maximum(np.sqrt((n[rim] ** 2).sum(1)), 1e-12)
        n[rim] /= ln[:, None]
        n[lat, 0] = ux[lat]
        n[lat, 2] = uz[lat]
        n[cap, 1] = np.sign(q[cap, 1])
        return gap, n

    def _ground_sdf(self, p):
        """Signed distance and outward normal to the ground box."""
        q = np.abs(p - self.gnd_c) - self.gnd_h
        outside = np.sqrt((np.maximum(q, 0.0) ** 2).sum(1))
        gap = outside + np.minimum(np.max(q, axis=1), 0.0)
        axis = np.argmax(q, axis=1)
        n = np.zeros_like(p)
        rows = np.arange(self.N)
        n[rows, axis] = np.sign(p[rows, axis] - self.gnd_c[axis])
        # sign of a zero coordinate can be 0; normalise defensively
        ln = np.maximum(np.sqrt((n ** 2).sum(1)), 1e-12)
        n /= ln[:, None]
        return gap, n

    def _contact_force(self, p, v, h):
        """Penalty + friction contact forces against both static obstacles."""
        F = np.zeros_like(p)
        for gap, n in (self._cylinder_sdf(p), self._ground_sdf(p)):
            near = gap < self.contact_margin
            if not np.any(near):
                continue
            pen = np.maximum(-gap, 0.0)
            vn = (v * n).sum(1)
            # normal penalty (only when penetrating) + damping when closing in
            fn_mag = self.k_contact * pen
            fn_mag = fn_mag + np.where(near & (vn < 0.0),
                                       -self.c_contact * vn, 0.0)
            F += fn_mag[:, None] * n
            # Coulomb friction (impulse limited so it never reverses slip)
            vt = v - vn[:, None] * n
            vt_mag = np.sqrt((vt * vt).sum(1))
            slip = near & (vt_mag > 1e-9)
            if np.any(slip):
                needed = self.mass * vt_mag / max(h, 1e-12)
                fr = np.minimum(self.mu * fn_mag, needed)
                F -= np.where(slip, fr, 0.0)[:, None] * (
                    vt / np.maximum(vt_mag, 1e-12)[:, None])
        return F

    def _project_contacts(self):
        """Hard non-penetration fallback for deep penetrations."""
        for gap, n in (self._cylinder_sdf(self.p), self._ground_sdf(self.p)):
            deep = gap < -self.proj_tol
            if not np.any(deep):
                continue
            push = (-gap[deep] - 0.5 * self.proj_tol)
            self.p[deep] += push[:, None] * n[deep]
            sub = np.where(deep)[0]
            vn = (self.v[sub] * n[sub]).sum(1)
            neg = vn < 0.0
            self.v[sub[neg]] -= vn[neg, None] * n[sub][neg]

    # ------------------------------------------------------------- stepping
    def _step(self, h):
        p, v, m = self.p, self.v, self.mass

        F = np.zeros_like(p)
        # gravity (external)
        F += m[:, None] * self.g[None, :]
        # aerodynamic drag (external)
        F -= (self.drag * m)[:, None] * v

        # spring elastic + damping forces
        d = p[self.J] - p[self.I]
        L = np.sqrt((d * d).sum(1))
        L = np.maximum(L, 1e-9)
        e = d / L[:, None]
        ext = L - self.rest
        vr = ((v[self.J] - v[self.I]) * e).sum(1)
        fmag = self.k_spring * ext + self.c_spring * vr
        fe = fmag[:, None] * e
        for a in range(3):
            F[:, a] += (np.bincount(self.I, fe[:, a], minlength=self.N)
                        - np.bincount(self.J, fe[:, a], minlength=self.N))

        # contacts
        F += self._contact_force(p, v, h)

        # semi-implicit Euler
        v += (F / m[:, None]) * h
        # defensive speed clamp (keeps state finite under adversarial loads)
        sp = np.sqrt((v * v).sum(1))
        scale = np.minimum(1.0, 50.0 / np.maximum(sp, 1e-12))
        v *= scale[:, None]
        p += v * h

        self._project_contacts()
        self.t += h
        self.n_steps += 1

    # ------------------------------------------------------- export helpers
    def _deformation_gradient(self, p):
        """Per-vertex 2D deformation gradient in a local sheet frame."""
        N, nx, ny = self.N, self.nx, self.ny
        rows = np.arange(N)
        A = np.zeros((N, 3))
        B = np.zeros((N, 3))

        # image of the material x-direction (central / one-sided differences)
        hasr = self.ix < nx - 1
        hasl = self.ix > 0
        both = hasr & hasl
        onlyr = hasr & ~hasl
        A[both] = (p[rows[both] + 1] - p[rows[both] - 1]) / (2.0 * self.dx)
        A[onlyr] = (p[rows[onlyr] + 1] - p[rows[onlyr]]) / self.dx
        A[~hasr] = (p[rows[~hasr]] - p[rows[~hasr] - 1]) / self.dx

        # image of the material y-direction
        hasr = self.iy < ny - 1
        hasl = self.iy > 0
        both = hasr & hasl
        onlyr = hasr & ~hasl
        B[both] = (p[rows[both] + nx] - p[rows[both] - nx]) / (2.0 * self.dy)
        B[onlyr] = (p[rows[onlyr] + nx] - p[rows[onlyr]]) / self.dy
        B[~hasr] = (p[rows[~hasr]] - p[rows[~hasr] - nx]) / self.dy

        nrm = np.cross(A, B)
        ln = np.sqrt((nrm * nrm).sum(1))
        good = ln > 1e-12
        nrm[good] /= ln[good][:, None]
        nrm[~good] = np.array([0.0, 0.0, 1.0])

        la = np.maximum(np.sqrt((A * A).sum(1)), 1e-12)
        t1 = A / la[:, None]
        t2 = np.cross(nrm, t1)

        F2 = np.empty((N, 2, 2))
        F2[:, 0, 0] = (A * t1).sum(1)
        F2[:, 0, 1] = (B * t1).sum(1)
        F2[:, 1, 0] = (A * t2).sum(1)
        F2[:, 1, 1] = (B * t2).sum(1)
        return F2

    def _signed_distance(self, p):
        gap_c, _ = self._cylinder_sdf(p)
        gap_g, _ = self._ground_sdf(p)
        return np.minimum(gap_c, gap_g)

    def _make_frame(self):
        ext_force = self.mass[:, None] * self.g[None, :]
        frame = StateFrame(
            float(self.t),
            self.p.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.N, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "boundary_vertex": self.boundary.copy(),
                "deformation_gradient": self._deformation_gradient(self.p),
            },
            fields={
                "contact_signed_distance": self._signed_distance(self.p),
                "cumulative_mass_source": np.zeros(1),
                "external_force": ext_force,
            },
            scalars={"contact_length_scale": float(self.thickness)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("solver not initialized")
        target = float(time)
        guard = 0
        while self.t < target - 1e-9:
            h = min(self.dt, target - self.t)
            self._step(h)
            guard += 1
            if guard > 4_000_000:
                raise RuntimeError("time stepping failed to converge")
        if target >= self.t:
            self.t = target
        return self._make_frame()

    def diagnostics(self) -> dict:
        return {
            "time": float(self.t),
            "substeps": int(self.n_steps),
            "n_vertices": int(self.N),
            "n_springs": int(self.n_springs),
            "max_speed": float(np.sqrt((self.v ** 2).sum(1)).max())
            if self._ready else 0.0,
            "min_signed_distance": float(self._signed_distance(self.p).min())
            if self._ready else 0.0,
        }
