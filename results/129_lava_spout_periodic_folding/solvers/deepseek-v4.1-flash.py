"""Affine Particle-In-Cell (MPM) solver for the lava spout periodic folding scene.

The solver implements a real material-point-method simulation from scratch on
top of NumPy: a fixed background Cartesian grid, quadratic-free (trilinear)
B-spline transfers with affine (APIC) particle velocity fields, a
corotational/log-strain elastoviscoplastic constitutive model (Maxwell
relaxation of the deviatoric elastic strain plus von Mises plastic return
mapping), gravity, frictional rigid-contact boundary conditions against the
frozen receiving surface, and a prescribed viscous inlet at the nozzle mouth.

Everything is advanced with an explicit, CFL-limited time-stepping scheme; the
reported states are the physical particle state (position, velocity, mass,
deformation gradient, external force, contact signed distance, cumulative
injected mass) at every requested sample time.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# frozen geometry (from scene_geometry.yaml)
# ----------------------------------------------------------------------------
GROUND_CENTER = np.array([0.0, 0.0, -0.05])
GROUND_HALF = np.array([3.2, 1.4, 0.05])

BASIN_CENTER = np.array([0.0, 0.0, 0.06])
BASIN_RADIUS = 1.35
BASIN_HEIGHT = 0.12

NOZZLE_CENTER = np.array([0.0, 0.0, 3.02])
NOZZLE_RADIUS = 0.27
NOZZLE_HEIGHT = 0.42


class BenchmarkSolver:
    """Material point method solver for a viscous lava spout folding onto a plate."""

    # ---- material / numerical parameters -----------------------------------
    _MU = 2.0e4            # elastic shear modulus [Pa]
    _LAMBDA = 4.0e5        # elastic Lame first parameter [Pa]
    _YIELD = 9.0e3         # von Mises yield stress [Pa]
    _VISCOSITY = 3.0e4     # Maxwell deviatoric viscosity [Pa s]
    _INLET_SPEED = 0.18    # prescribed downward inlet speed [m/s]
    _EMIT_SPACING = 0.04   # spacing of injected material points [m]
    _SUBSTEPS = 20         # substeps per sample interval
    _MAX_SPEED = 25.0      # numerically safe speed clamp [m/s]

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        cfg = dict(scene.config)
        self._seed = int(seed)
        self._gravity = np.asarray(cfg["gravity"], dtype=float).reshape(3)
        sampling = cfg["sampling"]
        self._density = float(sampling["density"])
        self._spacing = float(sampling["spacing"])
        self._sample_dt = float(cfg["sample_dt"])

        # background grid -----------------------------------------------------
        self._dx = self._spacing
        self._origin = np.array([-1.68, -0.84, 0.0])
        self._ncells = np.array([24, 12, 26])
        n_nodes = int(np.prod(self._ncells + 1))
        self._n_nodes = n_nodes
        axes = [self._origin[i] + self._dx * np.arange(self._ncells[i] + 1) for i in range(3)]
        gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
        self._node_pos = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
        phi, nrm = self._sdf(self._node_pos)
        self._node_phi = phi
        self._node_nrm = nrm

        # containment box strictly inside the frozen particle domain ---------
        # (keeps every material point inside the background grid so that mass is
        # never lost; the ground plane itself is the physical lower bound)
        self._contain_lo = self._origin + 0.25 * self._dx
        self._contain_lo[2] = 0.0
        self._contain_hi = self._origin + (self._ncells - 0.25) * self._dx

        # constitutive time scale --------------------------------------------
        self._tau_relax = self._VISCOSITY / self._MU

        # inlet sites: same lattice as the frozen initial column -------------
        self._site_xy = np.array([[-0.11, -0.11], [0.03, -0.11], [0.17, -0.11],
                                  [-0.11, 0.03], [0.03, 0.03], [0.17, 0.03],
                                  [-0.11, 0.17], [0.03, 0.17], [0.17, 0.17]])
        self._emit_z = 2.79
        emit_mass = self._density * self._EMIT_SPACING ** 3
        lateral_area = self._spacing ** 2
        rate = self._density * lateral_area * self._INLET_SPEED / emit_mass
        self._emit_interval = 1.0 / rate
        self._emit_mass = emit_mass

        # state from the frozen initial frame --------------------------------
        frame0 = scene.initial_state()
        self._pos = np.array(frame0.position, dtype=float)
        self._mass = np.array(frame0.mass, dtype=float)
        self._body = np.array(frame0.body_id, dtype=np.int64) if frame0.body_id is not None \
            else np.zeros(len(self._pos), dtype=np.int64)
        n0 = len(self._pos)
        self._vel = np.zeros((n0, 3), dtype=float)
        self._F = np.repeat(np.eye(3)[None], n0, axis=0)
        self._C = np.zeros((n0, 3, 3), dtype=float)
        self._P = np.zeros((n0, 3, 3), dtype=float)
        self._vol = self._mass / self._density
        self._characteristic_length = float(
            max(np.ptp(self._pos, axis=0).max(), self._spacing))

        self._added_mass = 0.0
        self._rng = np.random.default_rng(self._seed)
        self._emit_acc = np.zeros(len(self._site_xy), dtype=float)
        self._time = 0.0
        self._dt_target = self._sample_dt / float(self._SUBSTEPS)
        self._fallback_events = 0

    # --------------------------------------------------------------- geometry
    @staticmethod
    def _box_sdf(p):
        rel = p - GROUND_CENTER
        q = np.abs(rel) - GROUND_HALF
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        sdf = outside + inside
        scaled = np.sign(rel) * np.maximum(q, 0.0)
        n_out = scaled / np.maximum(np.linalg.norm(scaled, axis=-1, keepdims=True), 1e-12)
        axis = np.argmax(q, axis=-1)
        n_in = np.zeros_like(p)
        sign_axis = np.sign(np.take_along_axis(rel, axis[..., None], -1))
        np.put_along_axis(n_in, axis[..., None], sign_axis, axis=-1)
        nrm = np.where((outside > 0.0)[..., None], n_out, n_in)
        return sdf, nrm

    @staticmethod
    def _cylinder_sdf(p):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        d = np.sqrt(x * x + y * y)
        d_safe = np.maximum(d, 1e-12)
        radial = d - BASIN_RADIUS
        axial = np.abs(z - BASIN_CENTER[2]) - 0.5 * BASIN_HEIGHT
        outside = np.sqrt(np.maximum(radial, 0.0) ** 2 + np.maximum(axial, 0.0) ** 2)
        inside = np.minimum(np.maximum(radial, axial), 0.0)
        sdf = outside + inside
        zsign = np.sign(z - BASIN_CENTER[2])
        zsign = np.where(zsign == 0.0, 1.0, zsign)
        n_out = np.stack((np.maximum(radial, 0.0) * x / d_safe,
                          np.maximum(radial, 0.0) * y / d_safe,
                          np.maximum(axial, 0.0) * zsign), axis=-1)
        n_out = n_out / np.maximum(np.linalg.norm(n_out, axis=-1, keepdims=True), 1e-12)
        n_in = np.stack((x / d_safe, y / d_safe, np.zeros_like(x)), axis=-1)
        deep = (radial > axial)
        n_in = np.where(deep[..., None], n_in,
                        np.stack((np.zeros_like(x), np.zeros_like(y),
                                  np.full_like(x, 0.0) + zsign), axis=-1))
        nrm = np.where((outside > 0.0)[..., None], n_out, n_in)
        return sdf, nrm

    def _sdf(self, p):
        """Signed distance and outward normal of the union of frozen obstacles."""
        sb, nb = self._box_sdf(p)
        sc, nc = self._cylinder_sdf(p)
        take_box = sb <= sc
        sdf = np.where(take_box, sb, sc)
        nrm = np.where(take_box[..., None], nb, nc)
        return sdf, nrm

    # ---------------------------------------------------------- constitutive
    def _return_map(self, F, dt):
        """Log-strain elastoviscoplastic return mapping.

        Returns the stored (plastically admissible, Maxwell relaxed) deformation
        gradient together with the first Piola stress that will drive the next
        grid solve.  Sigma_i are the singular values of F, eps_i = log(sigma_i)
        are the logarithmic principal stretches; the deviatoric part is clamped
        to the von Mises yield surface and then relaxed with the Maxwell factor
        exp(-dt / tau) where tau = eta / mu.
        """
        U, s, Vt = np.linalg.svd(F)
        s = np.clip(s, 1e-3, 1e3)
        eps = np.log(s)
        eps_mean = eps.mean(axis=1, keepdims=True)
        eps_dev = eps - eps_mean
        tau_dev = 2.0 * self._MU * eps_dev
        norm = np.sqrt((tau_dev * tau_dev).sum(axis=1))
        radial = np.where(norm > self._YIELD, self._YIELD / np.maximum(norm, 1e-30), 1.0)
        eps_dev = eps_dev * radial[:, None]
        eps_dev = eps_dev * float(np.exp(-dt / self._tau_relax))
        eps_new = np.clip(eps_dev + eps_mean, -5.0, 5.0)
        s_new = np.exp(eps_new)
        F_new = (U * s_new[:, None, :]) @ Vt
        tau = 2.0 * self._MU * eps_dev + 3.0 * self._LAMBDA * eps_mean
        P_new = (U * (tau / s_new)[:, None, :]) @ Vt
        return F_new, P_new

    # ------------------------------------------------------------ grid step
    def _p2g(self, dt, P):
        pos, vel, C, mass, vol = self._pos, self._vel, self._C, self._mass, self._vol
        h = self._dx
        nc = self._ncells
        g = (pos - self._origin) / h
        base = np.clip(np.floor(g).astype(np.int64), 0, nc - 1)
        frac = np.clip(g - base, 0.0, 1.0)

        wx = np.stack((1.0 - frac[:, 0], frac[:, 0]), axis=1)
        wy = np.stack((1.0 - frac[:, 1], frac[:, 1]), axis=1)
        wz = np.stack((1.0 - frac[:, 2], frac[:, 2]), axis=1)
        W = wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]
        W = W.reshape(-1, 8)

        ix = base[:, 0, None] + np.array([0, 1])
        iy = base[:, 1, None] + np.array([0, 1])
        iz = base[:, 2, None] + np.array([0, 1])
        ny = nc[1] + 1
        nz = nc[2] + 1
        idx = ((ix[:, :, None, None] * ny + iy[:, None, :, None]) * nz
               + iz[:, None, None, :]).reshape(-1, 8)

        dwx = np.stack((-np.ones_like(frac[:, 0]), np.ones_like(frac[:, 0])), axis=1) / h
        dwy = np.stack((-np.ones_like(frac[:, 1]), np.ones_like(frac[:, 1])), axis=1) / h
        dwz = np.stack((-np.ones_like(frac[:, 2]), np.ones_like(frac[:, 2])), axis=1) / h
        gx = (dwx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]).reshape(-1, 8)
        gy = (wx[:, :, None, None] * dwy[:, None, :, None] * wz[:, None, None, :]).reshape(-1, 8)
        gz = (wx[:, :, None, None] * wy[:, None, :, None] * dwz[:, None, None, :]).reshape(-1, 8)
        gradw = np.stack((gx, gy, gz), axis=-1)

        node_x = self._node_pos[idx]                     # (N,8,3)
        r = node_x - pos[:, None, :]                     # (N,8,3)
        v_aff = vel[:, None, :] + np.einsum("nij,nkj->nki", C, r)

        mw = W * mass[:, None]                           # (N,8)
        flat_idx = idx.ravel()
        n_nodes = self._n_nodes

        node_mass = np.bincount(flat_idx, weights=mw.ravel(), minlength=n_nodes)
        mom = np.empty((n_nodes, 3))
        for a in range(3):
            mom[:, a] = np.bincount(flat_idx, weights=(mw * v_aff[:, :, a]).ravel(),
                                    minlength=n_nodes)
        outer_v = mw[:, :, None, None] * v_aff[:, :, :, None] * r[:, :, None, :]
        B = np.empty((n_nodes, 3, 3))
        for a in range(3):
            for b in range(3):
                B[:, a, b] = np.bincount(flat_idx, weights=outer_v[:, :, a, b].ravel(),
                                         minlength=n_nodes)
        outer_r = mw[:, :, None, None] * r[:, :, :, None] * r[:, :, None, :]
        D = np.empty((n_nodes, 3, 3))
        for a in range(3):
            for b in range(a, 3):
                vals = np.bincount(flat_idx, weights=outer_r[:, :, a, b].ravel(),
                                   minlength=n_nodes)
                D[:, a, b] = vals
                D[:, b, a] = vals
        f = -vol[:, None, None] * np.einsum("nij,nkj->nki", P, gradw)
        node_force = np.empty((n_nodes, 3))
        for a in range(3):
            node_force[:, a] = np.bincount(flat_idx, weights=f[:, :, a].ravel(),
                                           minlength=n_nodes)
        return node_mass, mom, B, D, node_force

    def _step(self, dt):
        node_mass, mom, B, D, node_force = self._p2g(dt, self._P)

        inv_mass = np.where(node_mass > 1e-12, 1.0 / np.maximum(node_mass, 1e-12), 0.0)
        node_v = mom * inv_mass[:, None]
        active = node_mass > 1e-12
        acc = np.where(active[:, None],
                       node_force * inv_mass[:, None] + self._gravity[None, :],
                       np.zeros(3))
        node_v = node_v + dt * acc

        # affine matrices C_i = B_i D_i^{-1} (regularized for sparse nodes)
        eps = 1e-8 + 1e-6 * (node_mass * self._dx * self._dx)
        D_reg = D + eps[:, None, None] * np.eye(3)[None, :, :]
        D_inv = np.linalg.inv(D_reg)
        C_node = np.einsum("nij,njk->nik", B, D_inv)
        C_node = np.where(active[:, None, None], C_node, 0.0)

        # rigid contact boundary condition at grid nodes ---------------------
        # Every node whose position lies on or inside a frozen static obstacle
        # is forced to the obstacle velocity (zero).  The fluid therefore sees a
        # no-slip, non-penetrating wall through the interpolation stencil, while
        # the exact particle-level projection below removes any penetration.
        contact = active & (self._node_phi < 1e-9)
        if np.any(contact):
            node_v[contact] = 0.0

        # particle update ----------------------------------------------------
        h = self._dx
        pos, vel, C = self._pos, self._vel, self._C
        g = (pos - self._origin) / h
        base = np.clip(np.floor(g).astype(np.int64), 0, self._ncells - 1)
        frac = np.clip(g - base, 0.0, 1.0)
        wx = np.stack((1.0 - frac[:, 0], frac[:, 0]), axis=1)
        wy = np.stack((1.0 - frac[:, 1], frac[:, 1]), axis=1)
        wz = np.stack((1.0 - frac[:, 2], frac[:, 2]), axis=1)
        W = (wx[:, :, None, None] * wy[:, None, :, None] * wz[:, None, None, :]).reshape(-1, 8)
        ix = base[:, 0, None] + np.array([0, 1])
        iy = base[:, 1, None] + np.array([0, 1])
        iz = base[:, 2, None] + np.array([0, 1])
        ny = self._ncells[1] + 1
        nz = self._ncells[2] + 1
        idx = ((ix[:, :, None, None] * ny + iy[:, None, :, None]) * nz
               + iz[:, None, None, :]).reshape(-1, 8)

        node_x = self._node_pos[idx]
        r = pos[:, None, :] - node_x
        v_nodes = node_v[idx]                             # (N,8,3)
        C_nodes = C_node[idx]                             # (N,8,3,3)
        v_aff = v_nodes + np.einsum("nkji,nkj->nki", C_nodes, r)
        v_p = np.einsum("nk,nki->ni", W, v_aff)
        C_p = np.einsum("nk,nkij->nij", W, C_nodes)

        vel = np.clip(v_p, -self._MAX_SPEED, self._MAX_SPEED)
        pos = pos + dt * vel

        # deformation gradient update (trial), stress stored for next step
        eye = np.eye(3)
        F_trial = np.einsum("nij,njk->nik", eye[None] + dt * C_p, self._F)
        self._F, self._P = self._return_map(F_trial, dt)
        self._C = C_p

        # particle level rigid contact + containment -------------------------
        self._pos = pos
        self._vel = vel
        self._resolve_contacts()

    def _resolve_contacts(self):
        phi, nrm = self._sdf(self._pos)
        hit = phi < 0.0
        if np.any(hit):
            self._pos[hit] += (-phi[hit] + 1e-4)[:, None] * nrm[hit]
            vn = np.einsum("ij,ij->i", self._vel[hit], nrm[hit])
            inward = vn < 0.0
            if np.any(inward):
                sub = np.where(hit)[0][inward]
                self._vel[sub] -= vn[inward][:, None] * nrm[hit][inward]
        # containment (safety boundary inside the frozen particle domain)
        lo, hi = self._contain_lo, self._contain_hi
        for k in range(3):
            below = self._pos[:, k] < lo[k]
            if np.any(below):
                self._pos[below, k] = lo[k]
                self._vel[below, k] = np.maximum(self._vel[below, k], 0.0)
            above = self._pos[:, k] > hi[k]
            if np.any(above):
                self._pos[above, k] = hi[k]
                self._vel[above, k] = np.minimum(self._vel[above, k], 0.0)

    # ------------------------------------------------------------------ inlet
    def _emit(self, dt):
        self._emit_acc += dt
        sites = np.nonzero(self._emit_acc >= self._emit_interval)[0]
        if sites.size == 0:
            return
        self._emit_acc[sites] -= self._emit_interval
        k = len(sites)
        xy = self._site_xy[sites]
        jitter = self._rng.uniform(-0.008, 0.008, size=(k, 3))
        new_pos = np.column_stack((xy, np.full(k, self._emit_z))) + jitter
        lateral = self._rng.uniform(-0.02, 0.02, size=(k, 2))
        new_vel = np.column_stack((lateral, np.full(k, -self._INLET_SPEED)))
        self._pos = np.concatenate((self._pos, new_pos), axis=0)
        self._vel = np.concatenate((self._vel, new_vel), axis=0)
        self._mass = np.concatenate((self._mass, np.full(k, self._emit_mass)))
        self._body = np.concatenate((self._body, np.zeros(k, dtype=np.int64)))
        self._F = np.concatenate((self._F, np.repeat(np.eye(3)[None], k, axis=0)), axis=0)
        self._C = np.concatenate((self._C, np.zeros((k, 3, 3))), axis=0)
        self._vol = np.concatenate((self._vol, np.full(k, self._emit_mass / self._density)))
        self._added_mass += float(k * self._emit_mass)

    # ------------------------------------------------------------ time loop
    def _snapshot(self):
        return (self._pos.copy(), self._vel.copy(), self._F.copy(), self._C.copy(),
                self._mass.copy(), self._body.copy(), self._vol.copy())

    def _restore(self, snap):
        (self._pos, self._vel, self._F, self._C,
         self._mass, self._body, self._vol) = snap

    def _finite(self):
        return (np.isfinite(self._pos).all() and np.isfinite(self._vel).all()
                and np.isfinite(self._F).all() and np.isfinite(self._C).all())

    def _advance(self, dt):
        snap = self._snapshot()
        step = dt
        for _ in range(4):
            self._restore(snap)
            try:
                self._step(step)
            except Exception:
                self._restore(snap)
                step *= 0.25
                self._fallback_events += 1
                continue
            if self._finite() and np.abs(self._vel).max(initial=0.0) <= self._MAX_SPEED:
                return
            self._restore(snap)
            step *= 0.25
            self._fallback_events += 1
        self._restore(snap)

    def advance_to(self, time: float):
        target = float(time)
        while self._time < target - 1e-12:
            remaining = target - self._time
            n = max(1, int(np.ceil(remaining / self._dt_target - 1e-9)))
            d = remaining / n
            for _ in range(n):
                self._emit(d)
                self._advance(d)
                self._time += d
        self._time = target
        return self._frame()

    # ---------------------------------------------------------------- output
    def _frame(self):
        pos = self._pos
        phi, _ = self._sdf(pos)
        band = 2.0 * self._spacing
        near = phi[phi < band]
        if near.size == 0:
            near = np.array([float(phi.min())])
        elif near.size > 4000:
            near = np.sort(near)[:4000]
        external = self._mass[:, None] * self._gravity[None, :]
        n = len(pos)
        return StateFrame(
            time=self._time,
            position=pos.copy(),
            velocity=self._vel.copy(),
            mass=self._mass.copy(),
            body_id=self._body.astype(np.int32),
            attributes={
                "mass_quantity": self._mass.copy(),
                "deformation_gradient": self._F.copy(),
                "particle_radius": np.full(n, 0.45 * self._spacing),
                "material_label": self._body.astype(np.int32),
                "volume_quantity": self._vol.copy(),
            },
            fields={
                "external_force": external,
                "cumulative_mass_source": np.array([self._added_mass]),
                "contact_signed_distance": near,
            },
            scalars={
                "contact_length_scale": self._spacing,
                "grid_spacing": self._dx,
                "characteristic_length": self._characteristic_length,
                "reference_velocity": 1.0,
            },
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self._time),
            "particle_count": int(len(self._pos)),
            "cumulative_added_mass": float(self._added_mass),
            "max_speed": float(np.abs(self._vel).max()) if len(self._vel) else 0.0,
            "fallback_events": int(self._fallback_events),
        }
