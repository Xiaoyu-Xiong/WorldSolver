"""Implicit frictional cloth solver for the closed belt around three cylinders.

Physical model
--------------
* The closed cloth belt is a mass-spring shell: structural links along the ring
  and width material directions, shear links on both cell diagonals and
  two-neighbour bending links.  Link stiffnesses follow from the continuum
  (Young modulus, shear modulus and plate bending rigidity) and the local rest
  metric, so the discretisation is a consistent bar/plate model rather than a
  tuned animation.
* Time integration is fully implicit Euler with one Newton linearisation per
  sub-step, solved with a sparse direct factorisation (SuperLU).
* The three cylinders are prescribed kinematic obstacles.  The top cylinder
  translates upward during the lift window; the two outer cylinders spin about
  the shared axis during the forward / reverse windows (smooth trapezoidal
  angular-velocity profiles whose integral is exactly one revolution).
* Cloth/cylinder contact uses a penalty normal force with normal damping and an
  elasto-plastic (bristle) Coulomb friction law with a hard |f_t| <= mu*N cap.
  Friction is what transports belt material forward and then reverses it, and
  the Coulomb cap reproduces the capstan slip over the idle top cylinder.
* A mild positional projection removes residual penetration after every step so
  the belt never sinks into the cylinders.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _ss(x: float) -> float:
    """Smoothstep 3x^2-2x^3 clamped to [0,1] (mean value 0.5)."""
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return x * x * (3.0 - 2.0 * x)


def _ss_int(x: float) -> float:
    """Integral of the clamped smoothstep, _ss_int(1) = 0.5."""
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return x * x * x * (1.0 - 0.5 * x)


def _window(t: float, t0: float, t1: float, tau: float):
    """Trapezoidal drive window.

    Returns (profile, integral, scale) where profile is in [0,1] with smooth
    ramps of length ``tau`` at both ends, ``integral`` is the time integral of
    the profile from t0 to t and ``scale = (t1-t0)-tau`` is the total integral,
    so that ``integral/scale`` is a normalised progress going 0 -> 1.
    """
    T = float(t1) - float(t0)
    if T <= 0.0:
        return 0.0, 0.0, 1.0
    tau = float(min(max(tau, 1e-9), 0.45 * T))
    scale = T - tau
    if t <= t0:
        return 0.0, 0.0, scale
    if t >= t1:
        return 0.0, scale, scale
    if t < t0 + tau:
        return _ss((t - t0) / tau), tau * _ss_int((t - t0) / tau), scale
    if t <= t1 - tau:
        return 1.0, 0.5 * tau + (t - t0 - tau), scale
    return (
        _ss((t1 - t) / tau),
        0.5 * tau + (T - 2.0 * tau) + (0.5 * tau - tau * _ss_int((t1 - t) / tau)),
        scale,
    )


class BenchmarkSolver:
    """Agent solver for scene ``030_cloth_belt_three_cylinders``."""

    SUBSTEPS = 10            # sub-steps per sampled frame interval
    E_STRETCH = 1.5e5        # Pa, in-plane Young modulus of the belt cloth
    POISSON = 0.3
    ZETA_EDGE = 0.12         # per-link viscous damping ratio
    DRAG_TAU = 0.03          # s, global velocity damping time constant
    KC = 2.0e5               # N/m, contact penalty stiffness
    KF = 4.0e4               # N/m, friction bristle stiffness
    CNORM_ZETA = 0.5         # contact normal damping ratio
    PEN_TOL = 1.0e-5         # m, residual penetration left to the penalty
    LIFT_RAMP = 0.15         # s
    SPIN_RAMP = 0.15         # s
    MAX_SPEED = 25.0         # m/s, numerical safety clamp

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        meta = scene.metadata()
        frame0 = scene.initial_state()
        cfg = getattr(scene, "config", None)
        cfg = cfg if isinstance(cfg, dict) else {}
        belt_cfg = cfg.get("belt", {}) if isinstance(cfg.get("belt", {}), dict) else {}

        pos0 = np.array(frame0.position, dtype=float)
        vel0 = np.array(frame0.velocity, dtype=float)
        self.n_total = int(pos0.shape[0])
        self.body_id = np.array(frame0.body_id).astype(int) if frame0.body_id is not None \
            else np.zeros(self.n_total, dtype=int)
        self.mass_all = np.array(frame0.mass, dtype=float) if frame0.mass is not None \
            else np.ones(self.n_total)
        self.orientation0 = np.array(frame0.orientation, dtype=float) \
            if frame0.orientation is not None else np.tile([0.0, 0.0, 0.0, 1.0], (self.n_total, 1))

        grid = (meta.get("topology", {}) or {}).get("grid_shape", [120, 9])
        NR, NW = int(grid[0]), int(grid[1])
        n = NR * NW
        self.NR, self.NW, self.n = NR, NW, n
        faces = np.array((meta.get("topology", {}) or {}).get("faces", []), dtype=int).reshape(-1, 3)
        self.faces = faces

        self.X = pos0[:n].copy()
        self.V = vel0[:n].copy()
        self.X0 = pos0[:n].copy()
        m = self.mass_all[:n].copy()
        m[~np.isfinite(m)] = 0.0
        if not np.all(m > 0.0):
            pos = m[m > 0.0]
            m[~(m > 0.0)] = float(pos.mean()) if pos.size else 1.0
        self.m = m

        attrs = frame0.attributes or {}
        self.uv = np.array(attrs.get("uv", np.zeros((self.n_total, 2))), dtype=float)
        if self.uv.shape[0] != self.n_total:
            self.uv = np.zeros((self.n_total, 2))
        self.mass_quantity = np.array(attrs.get("mass_quantity", self.mass_all.copy()), dtype=float)
        if self.mass_quantity.shape[0] != self.n_total:
            self.mass_quantity = self.mass_all.copy()
        self.cumulative_mass_source = np.zeros(1)

        # ------------------------------------------------------- scene config
        self.gravity = np.array(meta.get("gravity", cfg.get("gravity", [0.0, 0.0, -9.81])), dtype=float)
        tl = meta.get("timeline", cfg.get("timeline", {})) or {}
        self.w_lift = tuple(float(v) for v in tl.get("lift", (0.0, 1.0)))
        self.w_forward = tuple(float(v) for v in tl.get("forward", (1.4, 3.4)))
        self.w_reverse = tuple(float(v) for v in tl.get("reverse", (3.7, 5.7)))

        cyl = meta.get("cylinders", cfg.get("cylinders", {})) or {}
        self.R = float(cyl.get("radius", 0.28))
        self.cyl_len = float(cyl.get("length", 1.5))
        self.init_xz = np.array(cyl.get("initial_centers_xz",
                                        [[-0.9, 0.58], [0.9, 0.58], [0.0, 0.92]]), dtype=float)
        self.final_xz = np.array(cyl.get("final_centers_xz", self.init_xz), dtype=float)
        if self.final_xz.shape != self.init_xz.shape:
            self.final_xz = self.init_xz.copy()
        signs = np.array(cyl.get("outer_rotation_signs", [1.0, -1.0]), dtype=float)
        revs = float(cyl.get("outer_drive_revolutions", 1.0))
        self.spin_theta = 2.0 * np.pi * revs * signs
        self.n_cyl = int(self.init_xz.shape[0])

        thickness = float(belt_cfg.get("thickness", meta.get("belt_thickness", 0.006)))
        self.thickness = thickness
        self.clearance = float(belt_cfg.get("centerline_clearance", 0.008))
        self.Reff = self.R + self.clearance
        self.mu = float(belt_cfg.get("friction", meta.get("coulomb_friction",
                                                          cfg.get("friction", {}).get("coefficient", 0.4)
                                                          if isinstance(cfg.get("friction", {}), dict) else 0.4)))
        prestrain = float(belt_cfg.get("prestrain", 0.02))
        self.contact_length_scale = thickness if thickness > 0 else 0.006

        # -------------------------------------------------------- elastic net
        self._build_links(prestrain, thickness)
        self._build_jacobian_pattern()
        self._build_scatter()
        self._build_deformation_operators()

        # ------------------------------------------------------------- state
        self.z_fric = np.zeros((n, 3))
        self.time = 0.0
        self.theta = np.zeros(self.n_cyl)
        self._last_gap = np.full(n, 1.0)
        self._lift_p = 0.0

        sample_dt = None
        try:
            times = scene.sample_times()
            if len(times) > 1:
                sample_dt = float(times[1] - times[0])
        except Exception:
            sample_dt = None
        if sample_dt is None or sample_dt <= 0:
            sample_dt = float(cfg.get("sample_dt", 1.0 / 60.0))
        self.h_sub = sample_dt / float(self.SUBSTEPS)

        self.stats = {
            "steps": 0,
            "max_penetration": 0.0,
            "min_gap": float("inf"),
            "max_speed": 0.0,
            "max_stretch": 1.0,
            "transport_arc": 0.0,
            "solve_failures": 0,
        }
        self._ready = True

    # ------------------------------------------------------------- mesh links
    def _build_links(self, prestrain: float, thickness: float) -> None:
        NR, NW, n = self.NR, self.NW, self.n
        idx = np.arange(n).reshape(NR, NW)
        nxt = np.roll(idx, -1, axis=0)
        nxt2 = np.roll(idx, -2, axis=0)

        blocks = [
            (idx.ravel(), nxt.ravel(), 0),                      # ring structural
            (idx[:, :-1].ravel(), idx[:, 1:].ravel(), 1),       # width structural
            (idx[:, :-1].ravel(), nxt[:, 1:].ravel(), 2),       # shear diagonal 1
            (idx[:, 1:].ravel(), nxt[:, :-1].ravel(), 2),       # shear diagonal 2
            (idx.ravel(), nxt2.ravel(), 3),                     # bending (ring)
            (idx[:, :-2].ravel(), idx[:, 2:].ravel(), 4),       # bending (width)
        ]
        self.e0 = np.concatenate([b[0] for b in blocks]).astype(np.int64)
        self.e1 = np.concatenate([b[1] for b in blocks]).astype(np.int64)
        self.etype = np.concatenate([np.full(len(b[0]), b[2]) for b in blocks]).astype(np.int64)
        E = len(self.e0)
        self.n_ring0 = n
        self.n_wid0 = n + self.n_ring0
        self.n_diag0 = self.n_wid0 + NR * (NW - 1)
        self.n_br0 = self.n_diag0 + 2 * NR * (NW - 1)
        self.n_bw0 = self.n_br0 + n

        d = self.X0[self.e1] - self.X0[self.e0]
        Lin = np.linalg.norm(d, axis=1)
        Lin = np.maximum(Lin, 1e-9)
        relax = 1.0 / (1.0 + prestrain)

        L0 = Lin * relax
        ring_L0 = L0[: self.n_ring0].reshape(NR, NW)
        wid_L0 = L0[self.n_wid0:self.n_diag0].reshape(NR, NW - 1)
        # bending links are straight in the natural (flat) state
        L0[self.n_br0:self.n_bw0] = (ring_L0 + np.roll(ring_L0, -1, axis=0)).ravel()
        L0[self.n_bw0:] = (wid_L0[:, :-1] + wid_L0[:, 1:]).ravel()
        self.L0 = L0
        self.Lin = Lin

        E_mod = self.E_STRETCH
        G_mod = E_mod / (2.0 * (1.0 + self.POISSON))
        D_mod = E_mod * thickness ** 3 / (12.0 * (1.0 - self.POISSON ** 2))
        w_rest = float(np.mean(wid_L0))

        ring_a = ring_L0                                   # (NR,NW) ring spacing
        ring_a_prev = np.roll(ring_a, 1, axis=0)
        cell_a = 0.5 * (ring_a[:, :-1] + ring_a_prev[:, :-1])   # (NR,NW-1)
        cell_b = np.broadcast_to(wid_L0, (NR, NW - 1))
        cell_a_f = cell_a.ravel()
        cell_b_f = cell_b.ravel()

        k = np.zeros(E)
        # ring structural: bar of cross section (w_rest * thickness)
        k[: self.n_ring0] = E_mod * thickness * w_rest / ring_a.ravel()
        # width structural
        a_w = 0.5 * (ring_a[:, :-1] + ring_a[:, 1:]) if NW > 2 else ring_a[:, :-1]
        a_w = np.broadcast_to(ring_a[:, :-1], (NR, NW - 1)) if a_w.shape != (NR, NW - 1) else a_w
        k[self.n_wid0:self.n_diag0] = E_mod * thickness * a_w.ravel() / wid_L0.ravel()
        # shear diagonals (both per cell)
        kd = G_mod * thickness * (cell_a_f ** 2 + cell_b_f ** 2) / (2.0 * cell_a_f * cell_b_f)
        ncell = NR * (NW - 1)
        k[self.n_diag0:self.n_br0] = np.concatenate([kd, kd])
        # bending links: plate rigidity
        a_br = 0.5 * (ring_a + np.roll(ring_a, -1, axis=0))
        k[self.n_br0:self.n_bw0] = 3.0 * D_mod * w_rest / np.maximum(a_br.ravel(), 1e-6) ** 2
        a_bw = np.broadcast_to(ring_a[:, :-2], (NR, max(NW - 2, 1)))
        k[self.n_bw0:] = 3.0 * D_mod * a_bw.ravel() / max(w_rest, 1e-6) ** 2
        self.k = np.maximum(k, 1e-9)

        m_avg = 0.5 * (self.m[self.e0] + self.m[self.e1])
        self.c = self.ZETA_EDGE * np.sqrt(2.0 * self.k * m_avg)

        rows = np.concatenate([self.e0, self.e1])
        cols = np.concatenate([np.arange(E), np.arange(E)])
        vals = np.concatenate([np.ones(E), -np.ones(E)])
        self.Cmat = sp.csr_matrix((vals, (rows, cols)), shape=(n, E))

    def _build_jacobian_pattern(self) -> None:
        n, E = self.n, len(self.e0)
        comp = np.arange(3)
        r0 = (3 * self.e0)[:, None] + comp[None, :]
        r1 = (3 * self.e1)[:, None] + comp[None, :]
        rows_00 = np.broadcast_to(r0[:, :, None], (E, 3, 3))
        cols_00 = np.broadcast_to(r0[:, None, :], (E, 3, 3))
        rows_11 = np.broadcast_to(r1[:, :, None], (E, 3, 3))
        cols_11 = np.broadcast_to(r1[:, None, :], (E, 3, 3))
        rows_01 = np.broadcast_to(r0[:, :, None], (E, 3, 3))
        cols_01 = np.broadcast_to(r1[:, None, :], (E, 3, 3))
        rows_10 = np.broadcast_to(r1[:, :, None], (E, 3, 3))
        cols_10 = np.broadcast_to(r0[:, None, :], (E, 3, 3))
        rd = np.broadcast_to(((3 * np.arange(n))[:, None] + comp[None, :])[:, :, None], (n, 3, 3))
        cd = np.broadcast_to(((3 * np.arange(n))[:, None] + comp[None, :])[:, None, :], (n, 3, 3))
        self.jrows = np.concatenate([rows_00.ravel(), rows_11.ravel(), rows_01.ravel(),
                                     rows_10.ravel(), rd.ravel()]).astype(np.int32)
        self.jcols = np.concatenate([cols_00.ravel(), cols_11.ravel(), cols_01.ravel(),
                                     cols_10.ravel(), cd.ravel()]).astype(np.int32)
        self.jshape = (3 * n, 3 * n)
        self.eye3 = np.eye(3)

    def _build_scatter(self) -> None:
        n = self.n
        self.ring_next = (np.arange(n) + self.NW) % n
        self.ring_prev = (np.arange(n) - self.NW) % n

    def _build_deformation_operators(self) -> None:
        f = self.faces
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        E1 = self.X0[b] - self.X0[a]
        E2 = self.X0[c] - self.X0[a]
        N = np.cross(E1, E2)
        area = 0.5 * np.linalg.norm(N, axis=1)
        Nhat = N / np.maximum(2.0 * area, 1e-12)[:, None]
        self.nrm0 = Nhat
        Bm = np.stack([E1, E2, Nhat], axis=2)
        self.invBm = np.linalg.inv(Bm)
        T = len(f)
        rows = np.concatenate([a, b, c])
        cols = np.concatenate([np.arange(T)] * 3)
        vals = np.concatenate([area] * 3)
        W = sp.coo_matrix((vals, (rows, cols)), shape=(self.n, T)).tocsr()
        sums = np.array(W.sum(axis=1)).ravel()
        sums[sums <= 0] = 1.0
        self.Wmat = sp.diags(1.0 / sums) @ W

    # ------------------------------------------------------- kinematic drivers
    def _lift(self, t: float):
        f, A, S = _window(t, self.w_lift[0], self.w_lift[1], self.LIFT_RAMP)
        return A / S, f / S

    def _spin(self, t: float, ci: int):
        """Angular velocity and accumulated angle of outer cylinder ``ci``."""
        omega = 0.0
        angle = 0.0
        for window, sign in ((self.w_forward, 1.0), (self.w_reverse, -1.0)):
            theta = sign * self.spin_theta[ci]
            f, A, S = _window(t, window[0], window[1], self.SPIN_RAMP)
            omega += theta * f / S
            angle += theta * A / S
        return omega, angle

    def _cylinder_state(self, t: float):
        p, dp = self._lift(t)
        centers = self.init_xz + (self.final_xz - self.init_xz) * p
        vz = (self.final_xz[:, 1] - self.init_xz[:, 1]) * dp
        omg = np.zeros(self.n_cyl)
        ang = np.zeros(self.n_cyl)
        for ci in range(min(2, self.n_cyl)):
            omg[ci], ang[ci] = self._spin(t, ci)
        self._lift_p = p
        return centers, vz, omg, ang

    # ---------------------------------------------------------------- contact
    def _contact(self, X, centers):
        rx = X[:, 0][:, None] - centers[None, :, 0]
        rz = X[:, 2][:, None] - centers[None, :, 1]
        dist = np.sqrt(rx * rx + rz * rz)
        dist = np.maximum(dist, 1e-9)
        imin = np.argmin(dist, axis=1)
        rows = np.arange(X.shape[0])
        d = dist[rows, imin]
        pen = np.maximum(0.0, self.Reff - d)
        Nrm = np.stack([rx[rows, imin] / d, np.zeros_like(d), rz[rows, imin] / d], axis=1)
        return pen, Nrm, imin, d - self.Reff

    # ------------------------------------------------------------------ stepping
    def advance_to(self, time: float):
        if not self._ready:
            raise RuntimeError("BenchmarkSolver.initialize() was not called")
        target = float(time)
        if target < self.time:
            target = self.time
        guard = 0
        max_guard = int(4.0 * (target - self.time) / self.h_sub) + 64
        while self.time < target - 1e-10 and guard < max_guard:
            h = min(self.h_sub, target - self.time)
            if h <= 0.0:
                break
            self._step(h, self.time)
            self.time += h
            guard += 1
        return self._make_frame(target)

    def _step(self, h: float, t: float) -> None:
        n = self.n
        X, V, m = self.X, self.V, self.m
        centers, vz, omg, _ang = self._cylinder_state(t + h)

        # ------------------------------------------------ internal link forces
        d = X[self.e1] - X[self.e0]
        L = np.sqrt(np.einsum("ij,ij->i", d, d))
        L = np.maximum(L, 1e-9)
        u = d / L[:, None]
        rv = V[self.e1] - V[self.e0]
        dLdt = np.einsum("ij,ij->i", rv, u)
        fmag = self.k * (L - self.L0) + self.c * dLdt
        F = self.Cmat @ (fmag[:, None] * u)

        # --------------------------------------------------------- body force
        F += m[:, None] * self.gravity[None, :]
        c_drag = m / self.DRAG_TAU
        F -= c_drag[:, None] * V

        # ------------------------------------------------------------- contact
        pen, Nrm, imin, gap = self._contact(X, centers)
        active = pen > 0.0
        idx = np.arange(n)
        rs = X - np.stack([centers[imin, 0], np.zeros(n), centers[imin, 1]], axis=1)
        om = omg[imin]
        vs = np.stack([om * rs[:, 2], np.zeros(n), vz[imin] - om * rs[:, 0]], axis=1)
        vrel = V - vs
        vn = np.einsum("ij,ij->i", vrel, Nrm)
        Nn = np.maximum(0.0, self.KC * pen - self.CNORM_ZETA * np.sqrt(self.KC * m) * np.minimum(vn, 0.0))
        Nn = np.where(active, Nn, 0.0)
        F += Nn[:, None] * Nrm

        vt = vrel - vn[:, None] * Nrm
        zp = self.z_fric + h * vt
        zmag = np.linalg.norm(zp, axis=1)
        fmag = self.KF * zmag
        cap = self.mu * Nn
        stick = np.where(fmag > 1e-18, np.minimum(1.0, cap / np.maximum(fmag, 1e-18)), 0.0)
        zp *= stick[:, None]
        F -= self.KF * zp
        F[~active] += 0.0

        # ------------------------------------------------------- linear system
        s = (L - self.L0) / L
        ac = h * (self.k * s)
        bc = h * self.k * (1.0 - s) + self.c
        uu = u[:, :, None] * u[:, None, :]
        B = ac[:, None, None] * self.eye3[None, :, :] + bc[:, None, None] * uu
        Bf = B.reshape(-1)

        NN = Nrm[:, :, None] * Nrm[:, None, :]
        Dblk = (m / h + c_drag)[:, None, None] * self.eye3[None, :, :]
        Dblk += active[:, None, None] * (
            (h * self.KC + self.CNORM_ZETA * np.sqrt(self.KC * m))[:, None, None] * NN
            + (self.KF * h * stick)[:, None, None] * (self.eye3[None, :, :] - NN)
        )
        data = np.concatenate([Bf, Bf, -Bf, -Bf, Dblk.reshape(-1)])
        A = sp.coo_matrix((data, (self.jrows, self.jcols)), shape=self.jshape).tocsc()
        rhs = F.reshape(-1)

        try:
            lu = spla.splu(A)
            dv = lu.solve(rhs)
        except Exception:
            self.stats["solve_failures"] += 1
            dv = np.zeros_like(rhs)
        if not np.all(np.isfinite(dv)):
            self.stats["solve_failures"] += 1
            dv = np.nan_to_num(dv, nan=0.0, posinf=0.0, neginf=0.0)
        dv = dv.reshape(n, 3)
        spd = np.linalg.norm(dv, axis=1)
        bad = spd > self.MAX_SPEED
        if np.any(bad):
            dv[bad] *= (self.MAX_SPEED / spd[bad])[:, None]

        V += dv
        X += h * V

        # --------------------------------------------------- contact projection
        pen2, Nrm2, imin2, gap2 = self._contact(X, centers)
        over = pen2 - self.PEN_TOL
        fix = over > 0.0
        if np.any(fix):
            X[fix] += Nrm2[fix] * over[fix, None]
            vnn = np.einsum("ij,ij->i", V[fix], Nrm2[fix])
            neg = vnn < 0.0
            if np.any(neg):
                sel = np.where(fix)[0][neg]
                V[sel] -= Nrm2[sel] * vnn[neg, None]

        # --------------------------------------------------- friction memory
        vrel2 = V - vs
        vn2 = np.einsum("ij,ij->i", vrel2, Nrm)
        zp = self.z_fric + h * (vrel2 - vn2[:, None] * Nrm)
        zmag = np.linalg.norm(zp, axis=1)
        fmag = self.KF * zmag
        cap = self.mu * Nn
        stick2 = np.where(fmag > 1e-18, np.minimum(1.0, cap / np.maximum(fmag, 1e-18)), 0.0)
        self.z_fric = np.where(active[:, None], zp * stick2[:, None], 0.0)

        # ---------------------------------------------------------- statistics
        st = self.stats
        st["steps"] += 1
        if np.any(active):
            st["max_penetration"] = max(st["max_penetration"], float(pen2[active].max()))
        st["min_gap"] = min(st["min_gap"], float(gap2.min()))
        vmax = float(np.abs(V).max())
        if vmax > st["max_speed"]:
            st["max_speed"] = vmax
        stretch = float(np.max(L[: self.n_ring0] / self.L0[: self.n_ring0])) if self.n_ring0 else 1.0
        if stretch > st["max_stretch"]:
            st["max_stretch"] = stretch
        self._last_gap = gap2
        self._last_centers = centers

    # ------------------------------------------------------------------ frames
    def _make_frame(self, t: float):
        from worldsolver.contracts import StateFrame

        n, ntot = self.n, self.n_total
        centers, vz, omg, ang = self._cylinder_state(t)

        position = np.zeros((ntot, 3))
        velocity = np.zeros((ntot, 3))
        position[:n] = self.X
        velocity[:n] = self.V
        for i in range(self.n_cyl):
            row = n + i
            position[row] = (centers[i, 0], 0.0, centers[i, 1])
            velocity[row] = (0.0, 0.0, vz[i])
        if ntot > n + self.n_cyl:
            position[n + self.n_cyl:] = position[n + self.n_cyl - 1]

        position = np.nan_to_num(position, nan=0.0, posinf=0.0, neginf=0.0)
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=0.0, neginf=0.0)

        orientation = self.orientation0.copy()
        orientation[:n] = (0.0, 0.0, 0.0, 1.0)
        angular_velocity = np.zeros((ntot, 3))
        for i in range(self.n_cyl):
            q0 = self.orientation0[n + i]
            s, c = np.sin(0.5 * ang[i]), np.cos(0.5 * ang[i])
            x0, y0, z0, w0 = q0
            orientation[n + i] = (
                w0 * 0.0 + x0 * c + y0 * s,
                w0 * 0.0 + y0 * c - x0 * s,
                w0 * s + z0 * c,
                w0 * c - z0 * s,
            )
            nrm = np.linalg.norm(orientation[n + i])
            orientation[n + i] /= nrm if nrm > 0 else 1.0
            angular_velocity[n + i] = (0.0, omg[i], 0.0)

        pen, Nrm, imin, gap = self._contact(self.X, centers)
        self._last_gap = gap

        # belt transport diagnostic (arc length travelled along the ring)
        tangent = self.X[self.ring_next] - self.X
        tl = np.linalg.norm(tangent, axis=1)
        safe = tl > 1e-9
        tangent = np.where(safe[:, None], tangent / np.maximum(tl, 1e-9)[:, None], 0.0)
        speed_along = np.einsum("ij,ij->i", self.V, tangent)
        self.stats["transport_arc"] = float(
            self.stats.get("transport_arc", 0.0) + float(np.mean(speed_along[safe])) * self.h_sub
        )

        frame = StateFrame(
            float(t),
            position,
            velocity,
            orientation=orientation,
            angular_velocity=angular_velocity,
            mass=self.mass_all.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "uv": self.uv,
                "mass_quantity": self.mass_quantity,
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "contact_signed_distance": gap.copy(),
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame

    def _deformation_gradient(self):
        f = self.faces
        if f.size == 0:
            out = np.zeros((self.n_total, 3, 3))
            out[:, np.arange(3), np.arange(3)] = 1.0
            return out
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        e1 = self.X[b] - self.X[a]
        e2 = self.X[c] - self.X[a]
        nrm = np.cross(e1, e2)
        ln = np.linalg.norm(nrm, axis=1)
        ok = ln > 1e-12
        nrm = np.where(ok[:, None], nrm / np.maximum(ln, 1e-12)[:, None], self.nrm0)
        Bs = np.stack([e1, e2, nrm], axis=2)
        Ftri = (Bs @ self.invBm).reshape(len(f), 9)
        Ftri = np.nan_to_num(Ftri, nan=0.0, posinf=0.0, neginf=0.0)
        Fn = (self.Wmat @ Ftri).reshape(self.n, 3, 3)
        out = np.zeros((self.n_total, 3, 3))
        out[: self.n] = Fn
        eye = np.eye(3)
        for i in range(self.n, self.n_total):
            out[i] = eye
        return out

    # ------------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict:
        st = dict(getattr(self, "stats", {}))
        st["min_gap"] = float(st.get("min_gap", 0.0)) if np.isfinite(st.get("min_gap", 0.0)) else 0.0
        st["solver"] = "implicit mass-spring shell with bristle Coulomb friction"
        st["substeps_per_frame"] = int(self.SUBSTEPS) if self._ready else 0
        st["time"] = float(getattr(self, "time", 0.0))
        st["belt_samples"] = int(getattr(self, "n", 0))
        st["links"] = int(len(getattr(self, "e0", np.zeros(0))))
        return st
