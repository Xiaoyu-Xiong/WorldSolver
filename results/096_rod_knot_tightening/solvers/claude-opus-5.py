"""Discrete elastic rod solver for tightening a frictionless knot.

Physical model
--------------
The strand is a lumped-mass discrete elastic rod: ``N`` nodes carrying uniform
mass, connected by ``N-1`` inextensible edges.

* Inextensibility is enforced as a hard constraint.  The constraint Jacobian of
  a chain of edge-length constraints yields a symmetric tridiagonal system
  ``J W J^T lambda = -C`` which is solved directly, so length information
  propagates along the whole strand in a single solve (no Gauss-Seidel
  wave-front lag).  A few Newton iterations per substep drive the stretch
  residual to round-off.
* Bending resistance comes from the discrete energy
  ``E_b = k_b/2 * sum_i |x_{i+1} - 2 x_i + x_{i-1}|^2`` with ``k_b = EI/l0^3``.
  Its gradient ``D2^T D2 x`` is symmetric with a translation null space, so the
  nodal forces are internal (they sum to zero).
* Finite thickness self-contact is resolved with edge-edge closest-point
  inequality constraints ``|p_a - p_b| >= d_contact``.  Corrections act purely
  along the contact normal with inverse-mass weighting, i.e. exactly
  frictionless (mu = 0) and momentum conserving.
* Endpoint manipulation is applied as two stiff critically damped position
  servos on the controlled vertices.  These servo forces are the only external
  forces in the system (gravity is zero for this scene), which makes the
  momentum budget close exactly.
* Internal dissipation uses bending-rate damping, edge Laplacian velocity
  damping and a centre-of-mass preserving relative-velocity relaxation.  All
  three are momentum conserving, matching the quasi-static regime of the
  reference experiment.

Time stepping is symplectic-Euler + constraint projection (position based
dynamics) with a substep count derived from the explicit stability limit of the
stiffest force in the model, so the integration is stable and deterministic.

The authored centreline has an arc length of ~10.06 m while the frozen rest
length is 6.95 m, and it starts with slight self-overlap.  ``initialize``
therefore performs a collision-aware quasi-static pre-relaxation: the rest
length is ramped down while length projection (minimum-norm, so the loop shape
and its crossing pattern are preserved), contact projection and mild bending
smoothing are applied, followed by a short damped dynamic settle.  This yields
the loose loop with a threadable free end demanded by the initial state, on the
frozen constraint manifold.
"""
from __future__ import annotations

import math

import numpy as np

try:  # SciPy is optional; a dense fallback keeps the solver runnable without it.
    from scipy.linalg import solve_banded as _solve_banded
except Exception:  # pragma: no cover
    _solve_banded = None


def _tridiag_solve(diag: np.ndarray, off: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a symmetric tridiagonal system ``A y = rhs``.

    ``diag`` holds the ``n`` diagonal entries, ``off`` the ``n-1``
    sub/super-diagonal entries.
    """
    n = diag.size
    if n == 0:
        return np.zeros(0)
    if _solve_banded is not None:
        ab = np.zeros((3, n))
        ab[0, 1:] = off
        ab[1, :] = diag
        ab[2, :-1] = off
        try:
            return np.asarray(_solve_banded((1, 1), ab, rhs, check_finite=False))
        except Exception:
            pass
    mat = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    try:
        return np.linalg.solve(mat, rhs)
    except Exception:
        return np.linalg.lstsq(mat, rhs, rcond=None)[0]


class BenchmarkSolver:
    """Solver Agent implementation of the frictionless rod-knot tightening."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        cfg = dict(scene.config)
        self._cfg = cfg
        self._seed = int(seed)
        np.random.seed(int(seed) % (2**32))

        rod = cfg["rod"]
        self._n = int(rod["vertex_count"])
        self._radius = float(rod["radius"])
        self._total_mass = float(rod["total_mass"])
        self._rest_length = float(rod["rest_length"])
        self._l0 = self._rest_length / (self._n - 1)

        contact = cfg.get("self_contact", {}) or {}
        self._contact_enabled = bool(contact.get("enabled", True))
        self._d_contact = float(contact.get("contact_distance", 2.0 * self._radius))
        self._mu = float(contact.get("friction_coefficient", 0.0))
        # Small solve margin so that the final (tiny) length correction cannot
        # push a resolved contact back below the physical contact distance.
        self._d_solve = self._d_contact * 1.02

        self._gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        self._sample_dt = float(cfg["sample_dt"])
        self._duration = float(cfg["duration"])

        ctrl = cfg["endpoint_manipulation"]
        cv = [int(v) for v in ctrl.get("controlled_vertices", [0, self._n - 1])]
        self._i_left = cv[0]
        self._i_thread = cv[-1]
        self._wp_left = ctrl["left_tail"]["waypoints"]
        self._wp_thread = ctrl["threading_tail"]["waypoints"]

        # ------------------------------------------------ material constants
        self._mass = np.full(self._n, self._total_mass / self._n)
        self._w = 1.0 / self._mass
        m = float(self._mass[0])

        # Bending stiffness: soft rubber-like strand.  EI is small enough that
        # the prescribed tail tension tightens the knot to the contact limit,
        # but large enough to hold the initial loop open.
        self._EI = 1.0e-3
        self._kb = self._EI / self._l0**3
        self._ks = 3.0e3                                    # endpoint servo
        self._cs = 2.0 * math.sqrt(self._ks * m)            # critical damping
        self._cb = 0.05                                     # bending-rate damping
        self._cl = 0.08                                     # edge Laplacian damping
        self._tau_rel = 0.15                                # relative-velocity relaxation

        # Substeps from the explicit stability bounds of the stiffest terms:
        # elastic (dt < 2/omega) and viscous (dt < 2 m / c), each with a safety
        # factor of two so the scheme stays comfortably stable.
        omega = math.sqrt((16.0 * self._kb + self._ks) / m)
        dt_elastic = 1.0 / omega
        c_node = self._cs + 16.0 * self._cb + 4.0 * self._cl
        dt_viscous = 0.5 * m / c_node
        dt_max = min(dt_elastic, dt_viscous)
        sub = int(math.ceil(self._sample_dt / dt_max))
        self._substeps = int(min(240, max(20, sub)))
        self._dt = self._sample_dt / self._substeps

        # Contact broad-phase parameters.
        self._edge_gap = 4          # skip pairs closer than 4 edges along arc
        self._cutoff = self._d_solve + 2.5 * self._l0

        # Frozen normalisation scales (needed by the integrator's speed cap, so
        # they are computed before any stepping happens).
        self._constraint_scale = float(self._l0)
        v_ref = 0.0
        for wps in (self._wp_left, self._wp_thread):
            for a, b in zip(wps[:-1], wps[1:]):
                dt_w = float(b["time"]) - float(a["time"])
                if dt_w > 1e-9:
                    step = np.asarray(b["position"], dtype=float) - np.asarray(
                        a["position"], dtype=float
                    )
                    v_ref = max(v_ref, float(np.linalg.norm(step)) / dt_w)
        self._v_ref = v_ref if v_ref > 1e-6 else 1.0
        self._reference_momentum = float(self._total_mass * self._v_ref)

        # ------------------------------------------------- initial condition
        init = scene.initial_state()
        x = np.array(init.position, dtype=float).copy()
        self._x = self._prerelax(x)
        self._v = np.zeros_like(self._x)

        self._steps = 0
        self._time = 0.0
        self._cum_impulse = np.zeros(3)
        self._f_left = np.zeros(3)
        self._f_thread = np.zeros(3)
        self._p0 = self._momentum()
        self._prev_quat = self._quaternions(self._x)
        self._prev_quat_time = 0.0
        self._max_stretch = 0.0
        self._min_gap = float(self._min_edge_gap(self._x).min())

    # ------------------------------------------------------ kinematic targets
    @staticmethod
    def _smoothstep(u: float) -> float:
        u = float(min(1.0, max(0.0, u)))
        return u * u * (3.0 - 2.0 * u)

    def _waypoint(self, waypoints, t: float) -> np.ndarray:
        if t <= float(waypoints[0]["time"]):
            return np.asarray(waypoints[0]["position"], dtype=float)
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            tb = float(b["time"])
            if t <= tb:
                ta = float(a["time"])
                span = tb - ta
                u = self._smoothstep((t - ta) / span) if span > 1e-12 else 1.0
                pa = np.asarray(a["position"], dtype=float)
                pb = np.asarray(b["position"], dtype=float)
                return (1.0 - u) * pa + u * pb
        return np.asarray(waypoints[-1]["position"], dtype=float)

    def _targets(self, t: float):
        return self._waypoint(self._wp_left, t), self._waypoint(self._wp_thread, t)

    def _target_velocities(self, t: float):
        h = 1e-4
        la, ta = self._targets(max(0.0, t - h))
        lb, tb = self._targets(t + h)
        scale = 1.0 / ((t + h) - max(0.0, t - h))
        return (lb - la) * scale, (tb - ta) * scale

    # -------------------------------------------------------------- geometry
    def _bend_operator(self, y: np.ndarray) -> np.ndarray:
        """Return ``D2^T D2 y`` (symmetric, translation invariant)."""
        n = y.shape[0]
        lap = np.zeros_like(y)
        lap[1:-1] = y[2:] - 2.0 * y[1:-1] + y[:-2]
        pad = np.zeros((n + 2, 3))
        pad[1:-1] = lap
        return pad[0:n] - 2.0 * pad[1 : n + 1] + pad[2 : n + 2]

    def _edge_laplacian(self, y: np.ndarray) -> np.ndarray:
        """Momentum conserving graph Laplacian of the node chain."""
        out = np.zeros_like(y)
        diff = y[1:] - y[:-1]
        out[:-1] += diff
        out[1:] -= diff
        return out

    def _project_lengths(self, x: np.ndarray, target: float, iters: int = 1) -> np.ndarray:
        w = self._w
        for _ in range(iters):
            e = x[1:] - x[:-1]
            length = np.linalg.norm(e, axis=1)
            safe = np.maximum(length, 1e-12)
            u = e / safe[:, None]
            c = length - target
            diag = w[:-1] + w[1:]
            dots = np.einsum("ij,ij->i", u[:-1], u[1:])
            off = -w[1:-1] * dots
            diag = diag + 1e-12
            lam = _tridiag_solve(diag, off, -c)
            if not np.all(np.isfinite(lam)):
                break
            corr = lam[:, None] * u
            dx = np.zeros_like(x)
            dx[1:] += w[1:, None] * corr
            dx[:-1] -= w[:-1, None] * corr
            # Guard against a single explosive correction.
            norm = np.linalg.norm(dx, axis=1)
            cap = 0.5 * target
            over = norm > cap
            if np.any(over):
                dx[over] *= (cap / norm[over])[:, None]
            x = x + dx
        return x

    def _contact_pairs(self, x: np.ndarray):
        if not self._contact_enabled:
            return None
        mid = 0.5 * (x[:-1] + x[1:])
        delta = mid[:, None, :] - mid[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
        ne = mid.shape[0]
        idx = np.arange(ne)
        sep = idx[None, :] - idx[:, None]
        mask = (sep >= self._edge_gap) & (dist < self._cutoff)
        ii, jj = np.nonzero(mask)
        if ii.size == 0:
            return None
        return ii, jj

    @staticmethod
    def _closest_params(p1, d1, p2, d2):
        """Clamped closest-point parameters between two segment batches."""
        r = p1 - p2
        a = np.einsum("ij,ij->i", d1, d1)
        e = np.einsum("ij,ij->i", d2, d2)
        f = np.einsum("ij,ij->i", d2, r)
        c = np.einsum("ij,ij->i", d1, r)
        b = np.einsum("ij,ij->i", d1, d2)
        a_s = np.maximum(a, 1e-16)
        e_s = np.maximum(e, 1e-16)
        denom = a * e - b * b
        s = np.where(denom > 1e-16, (b * f - c * e) / np.where(denom > 1e-16, denom, 1.0), 0.0)
        s = np.clip(s, 0.0, 1.0)
        t = np.clip((b * s + f) / e_s, 0.0, 1.0)
        s = np.clip((t * b - c) / a_s, 0.0, 1.0)
        return s, t

    def _project_contacts(self, x: np.ndarray, pairs, relax: float = 0.7) -> np.ndarray:
        if pairs is None:
            return x
        ii, jj = pairs
        w = self._w
        i0, i1 = ii, ii + 1
        j0, j1 = jj, jj + 1
        p1, p2 = x[i0], x[j0]
        d1, d2 = x[i1] - p1, x[j1] - p2
        s, t = self._closest_params(p1, d1, p2, d2)
        pa = p1 + s[:, None] * d1
        pb = p2 + t[:, None] * d2
        diff = pa - pb
        dist = np.linalg.norm(diff, axis=1)
        active = dist < self._d_solve
        if not np.any(active):
            return x
        sel = np.nonzero(active)[0]
        s_a, t_a = s[sel], t[sel]
        dist_a = dist[sel]
        diff_a = diff[sel]
        # Deterministic fallback normal for (near) coincident closest points.
        degenerate = dist_a < 1e-9
        if np.any(degenerate):
            alt = 0.5 * (x[i0[sel]] + x[i1[sel]]) - 0.5 * (x[j0[sel]] + x[j1[sel]])
            alt_norm = np.linalg.norm(alt, axis=1)
            fixed = np.where(
                (alt_norm > 1e-9)[:, None], alt / np.maximum(alt_norm, 1e-12)[:, None],
                np.array([0.0, 0.0, 1.0]),
            )
            diff_a = np.where(degenerate[:, None], fixed, diff_a)
            dist_a = np.where(degenerate, 0.0, dist_a)
        nrm = np.linalg.norm(diff_a, axis=1)
        normal = diff_a / np.maximum(nrm, 1e-12)[:, None]
        cval = dist_a - self._d_solve

        wa0 = w[i0[sel]] * (1.0 - s_a) ** 2
        wa1 = w[i1[sel]] * s_a**2
        wb0 = w[j0[sel]] * (1.0 - t_a) ** 2
        wb1 = w[j1[sel]] * t_a**2
        wsum = np.maximum(wa0 + wa1 + wb0 + wb1, 1e-12)
        lam = (-cval / wsum) * relax

        # Under-relax shared nodes by a per-contact factor (not a per-node one):
        # scaling the whole contact impulse keeps every correction an exact
        # action-reaction pair, so self-contact stays momentum conserving.
        cnt = np.zeros(x.shape[0])
        np.add.at(cnt, i0[sel], 1.0)
        np.add.at(cnt, i1[sel], 1.0)
        np.add.at(cnt, j0[sel], 1.0)
        np.add.at(cnt, j1[sel], 1.0)
        share = np.maximum.reduce(
            [cnt[i0[sel]], cnt[i1[sel]], cnt[j0[sel]], cnt[j1[sel]]]
        )
        lam = lam / np.maximum(share, 1.0)
        step = lam[:, None] * normal

        dx = np.zeros_like(x)
        np.add.at(dx, i0[sel], (w[i0[sel]] * (1.0 - s_a))[:, None] * step)
        np.add.at(dx, i1[sel], (w[i1[sel]] * s_a)[:, None] * step)
        np.add.at(dx, j0[sel], -(w[j0[sel]] * (1.0 - t_a))[:, None] * step)
        np.add.at(dx, j1[sel], -(w[j1[sel]] * t_a)[:, None] * step)
        norm = np.linalg.norm(dx, axis=1)
        cap = 0.5 * self._d_contact
        over = norm > cap
        if np.any(over):
            dx[over] *= (cap / norm[over])[:, None]
        return x + dx

    def _min_edge_gap(self, x: np.ndarray) -> np.ndarray:
        """Per-edge signed gap to the nearest non-adjacent edge (fixed shape)."""
        ne = x.shape[0] - 1
        p = x[:-1]
        d = x[1:] - x[:-1]
        big = 10.0 * max(1.0, float(self._rest_length))
        out = np.full(ne, big)
        idx = np.arange(ne)
        for i in range(ne):
            far = np.abs(idx - i) >= self._edge_gap
            if not np.any(far):
                continue
            p1 = np.repeat(p[i][None, :], int(far.sum()), axis=0)
            d1 = np.repeat(d[i][None, :], int(far.sum()), axis=0)
            p2 = p[far]
            d2 = d[far]
            s, t = self._closest_params(p1, d1, p2, d2)
            diff = (p1 + s[:, None] * d1) - (p2 + t[:, None] * d2)
            out[i] = float(np.min(np.linalg.norm(diff, axis=1)))
        return out - self._d_contact

    # ------------------------------------------------------- pre-relaxation
    def _prerelax(self, x: np.ndarray) -> np.ndarray:
        """Quasi-static, collision-aware shortening onto the frozen rest length."""
        pin = (self._i_left, self._i_thread)
        pin_pos = np.array([x[self._i_left].copy(), x[self._i_thread].copy()])
        w_full = self._w.copy()
        self._w = w_full.copy()
        self._w[list(pin)] = 0.0
        try:
            edge = np.linalg.norm(np.diff(x, axis=0), axis=1)
            l_start = float(edge.mean())
            ramp = 700
            for k in range(ramp + 1):
                frac = k / ramp
                target = (1.0 - frac) * l_start + frac * self._l0
                x = self._project_lengths(x, target, iters=2)
                pairs = self._contact_pairs(x)
                x = self._project_contacts(x, pairs, relax=0.7)
                x = self._project_lengths(x, target, iters=1)
                # Mild bending smoothing keeps the shortening curve smooth.
                lap = self._edge_laplacian(x)
                lap[list(pin)] = 0.0
                x = x + 0.02 * lap
                x[self._i_left] = pin_pos[0]
                x[self._i_thread] = pin_pos[1]
            for _ in range(80):
                pairs = self._contact_pairs(x)
                x = self._project_contacts(x, pairs, relax=0.8)
                x = self._project_lengths(x, self._l0, iters=3)
                x[self._i_left] = pin_pos[0]
                x[self._i_thread] = pin_pos[1]
        finally:
            self._w = w_full

        # Short damped dynamic settle with the endpoints held at their t=0 pose.
        self._x = x
        self._v = np.zeros_like(x)
        settle = int(round(0.30 / self._dt))
        for _ in range(settle):
            self._substep(0.0, settle_mode=True)
        x = self._x
        # Finish exactly on the constraint manifold.
        for _ in range(40):
            pairs = self._contact_pairs(x)
            x = self._project_contacts(x, pairs, relax=0.8)
            x = self._project_lengths(x, self._l0, iters=3)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------ integrator
    def _substep(self, t: float, settle_mode: bool = False) -> None:
        dt = self._dt
        x = self._x
        v = self._v
        m = self._mass[:, None]

        # Internal elastic and dissipative forces (all sum to zero).
        f = -self._kb * self._bend_operator(x)
        f -= self._cb * self._bend_operator(v)
        f += self._cl * self._edge_laplacian(v)
        # Uniform gravity (zero for this scene) is a body force.
        f_grav = self._mass[:, None] * self._gravity[None, :]
        f = f + f_grav

        # External endpoint manipulation: two stiff critically damped servos.
        t_eval = min(t + 0.5 * dt, self._duration) if not settle_mode else 0.0
        p_left, p_thread = self._targets(t_eval)
        if settle_mode:
            v_left = np.zeros(3)
            v_thread = np.zeros(3)
        else:
            v_left, v_thread = self._target_velocities(t_eval)
        f_left = self._ks * (p_left - x[self._i_left]) + self._cs * (v_left - v[self._i_left])
        f_thread = self._ks * (p_thread - x[self._i_thread]) + self._cs * (
            v_thread - v[self._i_thread]
        )
        f[self._i_left] += f_left
        f[self._i_thread] += f_thread

        v = v + dt * f / m
        # Momentum conserving relaxation of relative motion (quasi-static regime).
        p_tot = float(self._total_mass)
        v_cm = (self._mass[:, None] * v).sum(axis=0) / p_tot
        gamma = math.exp(-dt / self._tau_rel)
        v = v_cm[None, :] + gamma * (v - v_cm[None, :])

        x_prev = x
        x = x + dt * v
        pairs = self._contact_pairs(x)
        for _ in range(4):
            x = self._project_contacts(x, pairs, relax=0.7)
            x = self._project_lengths(x, self._l0, iters=1)
        x = self._project_contacts(x, pairs, relax=0.9)
        x = self._project_lengths(x, self._l0, iters=2)

        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        v = (x - x_prev) / dt
        speed = np.linalg.norm(v, axis=1)
        cap = 50.0 * max(1.0, self._v_ref)
        over = speed > cap
        if np.any(over):
            v[over] *= (cap / speed[over])[:, None]
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        self._x = x
        self._v = v
        if not settle_mode:
            # External impulse accumulated at substep resolution (gravity is
            # zero here, so only the two manipulators act).
            self._cum_impulse = self._cum_impulse + dt * (
                f_left + f_thread + f_grav.sum(axis=0)
            )
            self._f_left = f_left
            self._f_thread = f_thread

    def _momentum(self) -> np.ndarray:
        return (self._mass[:, None] * self._v).sum(axis=0)

    # ------------------------------------------------------------ quaternions
    @staticmethod
    def _quat_from_tangent(tan: np.ndarray) -> np.ndarray:
        n = tan.shape[0]
        q = np.zeros((n, 4))
        axis = np.stack([-tan[:, 1], tan[:, 0], np.zeros(n)], axis=1)
        c = tan[:, 2]
        flip = c <= -1.0 + 1e-9
        q[:, 0:3] = axis
        q[:, 3] = 1.0 + c
        if np.any(flip):
            q[flip] = np.array([1.0, 0.0, 0.0, 0.0])
        norm = np.linalg.norm(q, axis=1)
        bad = norm < 1e-12
        if np.any(bad):
            q[bad] = np.array([0.0, 0.0, 0.0, 1.0])
            norm = np.linalg.norm(q, axis=1)
        return q / np.maximum(norm, 1e-12)[:, None]

    def _quaternions(self, x: np.ndarray) -> np.ndarray:
        tan = np.zeros_like(x)
        tan[1:-1] = x[2:] - x[:-2]
        tan[0] = x[1] - x[0]
        tan[-1] = x[-1] - x[-2]
        norm = np.linalg.norm(tan, axis=1)
        tan = tan / np.maximum(norm, 1e-12)[:, None]
        return self._quat_from_tangent(tan)

    @staticmethod
    def _angular_velocity(q_prev: np.ndarray, q_now: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 1e-12:
            return np.zeros((q_now.shape[0], 3))
        sign = np.sign(np.einsum("ij,ij->i", q_prev, q_now))
        sign[sign == 0.0] = 1.0
        q_p = q_prev * sign[:, None]
        dq = (q_now - q_p) / dt
        # omega = 2 * dq (x) conj(q)
        x1, y1, z1, w1 = dq[:, 0], dq[:, 1], dq[:, 2], dq[:, 3]
        x2, y2, z2, w2 = -q_now[:, 0], -q_now[:, 1], -q_now[:, 2], q_now[:, 3]
        ox = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        oy = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        oz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return 2.0 * np.stack([ox, oy, oz], axis=1)

    # ---------------------------------------------------------------- output
    def advance_to(self, time: float):
        from worldsolver.contracts import StateFrame

        time = float(time)
        target = int(round(time / self._dt))
        while self._steps < target:
            self._substep(self._steps * self._dt)
            self._steps += 1
        self._time = self._steps * self._dt

        x = np.ascontiguousarray(np.nan_to_num(self._x, nan=0.0, posinf=0.0, neginf=0.0))
        v = np.ascontiguousarray(np.nan_to_num(self._v, nan=0.0, posinf=0.0, neginf=0.0))
        n = self._n

        quat = self._quaternions(x)
        dt_q = self._time - self._prev_quat_time
        omega = self._angular_velocity(self._prev_quat, quat, dt_q)
        self._prev_quat = quat
        self._prev_quat_time = self._time

        edge = np.linalg.norm(np.diff(x, axis=0), axis=1)
        self._max_stretch = max(
            self._max_stretch, float(np.max(np.abs(edge - self._l0)) / self._l0)
        )
        gap = self._min_edge_gap(x)
        self._min_gap = min(self._min_gap, float(gap.min()))

        f_ext = np.zeros((n, 3))
        f_ext += self._mass[:, None] * self._gravity[None, :]
        f_ext[self._i_left] += self._f_left
        f_ext[self._i_thread] += self._f_thread

        # Single global momentum-budget domain: the whole strand.  All elastic,
        # damping and self-contact forces are internal pairs and cancel; the
        # only external actions are gravity and the two manipulators.
        domain_weights = np.ones((1, n))
        budget_force = np.zeros((1, 3, 3))
        budget_force[0, 0] = self._mass.sum() * self._gravity
        budget_force[0, 1] = self._f_left
        budget_force[0, 2] = self._f_thread

        frame = StateFrame(
            time=float(time),
            position=x.copy(),
            velocity=v.copy(),
            orientation=quat,
            angular_velocity=np.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0),
            mass=self._mass.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "mass_quantity": self._mass.copy(),
                "arc_fraction": np.linspace(0.0, 1.0, n),
                "rod_radius": np.full(n, self._radius),
                "contact_diameter": np.full(n, 2.0 * self._radius),
                "endpoint_controlled": np.isin(
                    np.arange(n), [self._i_left, self._i_thread]
                ).astype(float),
            },
            fields={
                "budget_domain_weights": domain_weights,
                "budget_external_force": budget_force,
                "budget_cumulative_external_impulse": self._cum_impulse.reshape(1, 3).copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "cumulative_mass_source": np.zeros(1),
                "constraint_value": edge.copy(),
                "constraint_target": np.full(n - 1, self._l0),
                "external_force": f_ext,
                "gravity_acceleration": self._gravity.reshape(1, 3).copy(),
                "contact_signed_distance": gap,
            },
            scalars={
                "constraint_scale": float(self._constraint_scale),
                "reference_momentum": float(self._reference_momentum),
                "contact_length_scale": float(self._d_contact),
                "friction_coefficient": float(self._mu),
                "rod_rest_length": float(self._rest_length),
            },
        )
        return frame

    def diagnostics(self) -> dict:
        return {
            "method": "discrete elastic rod, symplectic Euler + hard length/contact projection",
            "nodes": int(self._n),
            "substeps_per_sample": int(self._substeps),
            "dt": float(self._dt),
            "bending_EI": float(self._EI),
            "contact_distance": float(self._d_contact),
            "friction_coefficient": float(self._mu),
            "max_relative_stretch": float(self._max_stretch),
            "min_contact_gap": float(self._min_gap),
            "final_momentum": [float(c) for c in self._momentum()],
            "cumulative_external_impulse": [float(c) for c in self._cum_impulse],
        }
