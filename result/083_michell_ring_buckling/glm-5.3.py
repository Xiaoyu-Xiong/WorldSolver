"""Physics solver: Michell buckling of a pre-twisted closed elastic ring.

Physical model
--------------
Discrete elastic rod (Kirchhoff rod) on a closed loop with n vertices and n
edges:

* vertices x_i (i = 0..n-1, cyclic), velocities v_i, per-edge material normal
  m1_j (unit, kept orthogonal to the edge tangent) and per-edge spin rate
  omega_j (single rotational degree of freedom of each edge frame);
* bending energy  E_b = (alpha/2) sum_i |t_i - t_{i-1}|^2 / l0
  (isropic rod, naturally straight -> zero natural curvature);
* twisting energy E_t = (beta/2) sum_i w_i^2 / l0, where the discrete twist
  angle w_i at vertex i is the rotation of the material frame between the two
  adjacent edges about the local tangent (naturally untwisted rod);
* exact internal force structure: force on a vertex = negative energy gradient
  plus the moment-propagation term G_{i-1} - G_i with G_j = (t_j x M_j)/l_j,
  where M_j is the internal moment carried by edge j.  The material frames are
  Lagrangian (they co-rotate with the edges and spin about them), so the
  linking number Tw + Wr of the closed ribbon is conserved and the stored
  twist (3 windings, above the Michell/Zajac threshold 2*pi*sqrt(3)) is
  converted into writhe as the ring buckles out of plane.

Time integration: symplectic Euler with per-edge red-black Gauss-Seidel
projection enforcing inextensibility (edge lengths frozen at l0), velocity
recovered from the actual displacement, viscous material damping and penalty
self-contact between non-adjacent samples.  Gravity is zero (free relaxation
of the preloaded closed ring; no supports, no twist injection after start).
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic discrete-elastic-rod solver for the pre-twisted ring."""

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        rod = cfg["rod"]
        init = cfg["initial_conditions"]

        frame0 = scene.initial_state()
        self.n = n = int(rod["vertex_count"])
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        self.l0 = float(rod["length"]) / n
        self.alpha = float(rod["bending_stiffness_alpha"])
        self.beta = float(rod["bending_stiffness_beta"])
        self.rod_radius = float(rod["cross_section_radius"])
        self.rho = float(rod["density"])
        self.R = float(rod["radius"])
        self.m_tot = float(np.sum(self.mass))

        # intrinsic (spin) inertia of each rod sample (circular cross section)
        Ip = np.pi * self.rod_radius ** 4 / 2.0  # polar second moment of area
        Ib = np.pi * self.rod_radius ** 4 / 4.0  # transverse second moment
        self.I_spin = self.rho * Ip * self.l0
        self.I_bend = self.rho * Ib * self.l0

        # index helpers (cyclic)
        self.idx = np.arange(n)
        self.ip = (self.idx - 1) % n
        self.inx = (self.idx + 1) % n

        # initial material frames: material normal makes `winding` full turns
        # around the ring, compatible with the closed-frame initial condition.
        winding = int(init["material_frame_winding_number"])
        phase = 2.0 * np.pi * (self.idx + 0.5) / n
        radial = np.column_stack((np.cos(phase), np.sin(phase), np.zeros(n)))
        zhat = np.zeros((n, 3))
        zhat[:, 2] = 1.0
        m1 = np.cos(winding * phase)[:, None] * radial + np.sin(winding * phase)[:, None] * zhat
        t0, _ = self._tangents(self.x)
        m1 = m1 - np.sum(m1 * t0, axis=1)[:, None] * t0
        self.m1 = m1 / np.linalg.norm(m1, axis=1)[:, None]
        self.omega = np.zeros(n)

        # self-contact pairs: exclude adjacent samples (|i-j| <= 1 cyclic)
        sep = np.abs(self.idx[:, None] - self.idx[None, :])
        sep = np.minimum(sep, n - sep)
        self._cmask = sep > 1
        self.contact_dist = 2.0 * self.rod_radius
        self.k_c = 300.0
        self.c_c = 2.0

        # integrator
        self.dt = 1.0e-3
        self.sweeps = 8
        self.c_t = 0.25   # translational material damping (1/s)
        self.c_s = 0.25   # spin material damping (1/s)
        self.t = 0.0
        self._last_Eb = 0.0
        self._last_Et = 0.0

        self._initial_attributes = {
            key: np.array(value, dtype=float) for key, value in frame0.attributes.items()
        }

        self._reference_momentum = max(self.m_tot * 0.5, 1e-6)
        self._reference_angular_momentum = max(self.m_tot * 0.5 * max(self.R, 1e-6), 1e-6)

    # ------------------------------------------------------------------ helpers
    def _tangents(self, x):
        e = x[self.inx] - x[self.idx]
        l = np.linalg.norm(e, axis=1)
        l = np.maximum(l, 1e-12)
        return e / l[:, None], l

    # ------------------------------------------------------------------ physics
    def _step(self, dt):
        n = self.n
        ip, inx = self.ip, self.inx
        x0 = self.x.copy()
        m1_prev = self.m1.copy()
        om_prev = self.omega.copy()

        t, l = self._tangents(self.x)
        m1 = self.m1

        # ---- bending: E_b = (alpha/2) sum_i |t_i - t_{i-1}|^2 / l0
        d = t - t[ip]                                   # discrete curvature vectors at vertices
        dEdt = (self.alpha / self.l0) * (d - d[inx])    # d E_b / d t_j
        Eb = 0.5 * self.alpha * np.sum(d * d) / self.l0

        # ---- twist: E_t = (beta/2) sum_i w_i^2 / l0
        tsum = t[ip] + t
        s = np.linalg.norm(tsum, axis=1)
        safe = np.maximum(s, 1e-8)
        tb = tsum / safe[:, None]                       # vertex tangent (average)
        m1p = m1[ip]
        a = np.cross(m1p, m1)                           # m1_{i-1} x m1_i
        u = np.sum(a * tb, axis=1)
        vv = np.sum(m1p * m1, axis=1)
        w = np.arctan2(u, vv)                           # discrete twist angle at vertex i
        den = u * u + vv * vv
        tiny = den < 1e-14
        den = np.where(tiny, 1.0, den)
        c = self.beta * w / self.l0                     # d E_t / d w_i
        Et = 0.5 * self.beta * np.sum(w * w) / self.l0

        # gradient of E_t w.r.t. the vertex tangent tb_i
        gt = ((c * np.where(tiny, 0.0, vv)) / den)[:, None] * a
        # gradients of E_t w.r.t. the material normals
        dE_next = ((np.where(tiny, 0.0, c)) / den)[:, None] * (
            vv[:, None] * np.cross(tb, m1p) - u[:, None] * m1p
        )                                                # d w_i / d m1_i
        dE_prev = ((np.where(tiny, 0.0, c)) / den)[:, None] * (
            vv[:, None] * np.cross(m1, tb) - u[:, None] * m1
        )                                                # d w_i / d m1_{i-1}
        dEdm1 = dE_next + dE_prev[inx]                   # d E_t / d m1_j

        # internal moments and the moment-propagation force term
        M = np.cross(m1, dEdm1)                          # moment carried by edge j
        G = np.cross(t, M) / self.l0                     # (t_j x M_j)/l0
        tau_spin = np.sum(M * t, axis=1)                 # twist moment about tangent

        # contribution of tb-dependence to dE/dt_j (t_j enters tb_j and tb_{j+1})
        Pgt = gt - np.sum(gt * tb, axis=1)[:, None] * tb
        Pgt = Pgt / safe[:, None]
        g_t = dEdt + Pgt + Pgt[inx]                      # total dE/dt_j

        # ---- generalized forces on vertices
        Pg = g_t - np.sum(g_t * t, axis=1)[:, None] * t  # P_j g_j (P projects on tangent plane)
        Pg = Pg / l[:, None]
        F = (Pg - Pg[ip]) + (G[ip] - G)

        # ---- self-contact (internal, non-adjacent samples)
        F = F + self._contact_forces()

        # ---- material damping
        F = F - self.c_t * self.mass[:, None] * self.v

        if not (np.isfinite(F).all() and np.isfinite(tau_spin).all()):
            # numerical safety: freeze motion at last good state
            self.v[:] = 0.0
            self.omega[:] = 0.0
            self.t += dt
            return

        # ---- symplectic Euler
        self.v = self.v + dt * F / self.mass[:, None]
        self.x = self.x + dt * self.v
        self.omega = self.omega + dt * (
            -tau_spin - self.c_s * self.I_spin * self.omega
        ) / self.I_spin

        # ---- inextensibility: red-black Gauss-Seidel projection on edge lengths
        self._project_lengths()
        self.v = (self.x - x0) / dt

        if not np.isfinite(self.x).all():
            self.x = x0
            self.v[:] = 0.0
            self.omega = om_prev
            self.m1 = m1_prev
            self.t += dt
            return

        # ---- update material frames: co-rotate with the edge, then spin
        t2, _ = self._tangents(self.x)
        m = self.m1 - np.sum(self.m1 * t2, axis=1)[:, None] * t2
        nm = np.linalg.norm(m, axis=1)
        nm = np.maximum(nm, 1e-12)
        m = m / nm[:, None]
        ang = self.omega * dt
        self.m1 = np.cos(ang)[:, None] * m + np.sin(ang)[:, None] * np.cross(t2, m)
        self.m1 = self.m1 / np.maximum(
            np.linalg.norm(self.m1, axis=1), 1e-12
        )[:, None]

        self.t += dt
        self._last_Eb = float(Eb)
        self._last_Et = float(Et)

    def _project_lengths(self):
        n = self.n
        x = self.x
        l0 = self.l0
        for _ in range(self.sweeps):
            for color in (0, 1):
                j = np.arange(color, n, 2)
                i2 = (j + 1) % n
                e = x[i2] - x[j]
                ln = np.linalg.norm(e, axis=1)
                ln = np.maximum(ln, 1e-12)
                corr = ((ln - l0) * 0.5 / ln)[:, None] * e
                x[j] += corr
                x[i2] -= corr

    def _contact_forces(self):
        x, v = self.x, self.v
        diff = x[:, None, :] - x[None, :, :]              # i - j
        d2 = np.sum(diff * diff, axis=2)
        active = self._cmask & (d2 < self.contact_dist ** 2) & (d2 > 1e-12)
        F = np.zeros_like(x)
        if not np.any(active):
            return F
        d = np.sqrt(d2)
        nv = diff / np.maximum(d, 1e-12)[:, :, None]
        vrel = v[:, None, :] - v[None, :, :]
        vn = np.sum(vrel * nv, axis=2)
        pen = np.where(active, self.contact_dist - d, 0.0)
        fmag = np.where(active, self.k_c * pen - self.c_c * vn, 0.0)
        F = np.sum(fmag[:, :, None] * nv, axis=1)
        return F

    # ------------------------------------------------------------------ output
    def _make_frame(self, time):
        n = self.n
        t, _ = self._tangents(self.x)
        tt = t[:, :, None] * t[:, None, :]                # (n,3,3) t t^T
        eye = np.eye(3)[None, :, :]
        inertia = self.I_spin * tt + self.I_bend * (eye - tt)
        spin_vec = self.omega[:, None] * t

        z = self.x[:, 2]
        amp = float(np.sqrt(np.mean(z * z)))

        attributes = dict(self._initial_attributes)
        attributes["inertia_tensor"] = inertia
        attributes["mass_quantity"] = self.mass.copy()

        fields = {
            "budget_cumulative_external_angular_impulse": np.zeros((1, 3)),
            "budget_cumulative_external_impulse": np.zeros((1, 3)),
            "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3)),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, n)),
            "budget_external_couple": np.zeros((1, 1, 3)),
            "budget_external_force": np.zeros((1, 1, 3)),
            "budget_force_position": np.zeros((1, 1, 3)),
            "constraint_value": np.array([amp], dtype=float),
            "constraint_target": np.array([0.0], dtype=float),
            "cumulative_mass_source": np.zeros(1),
        }
        scalars = {
            "constraint_scale": 1.0,
            "reference_momentum": self._reference_momentum,
            "reference_angular_momentum": self._reference_angular_momentum,
        }
        return StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            angular_velocity=spin_vec,
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    # ------------------------------------------------------------------ api
    def advance_to(self, time):
        time = float(time)
        if time > self.t:
            nsub = int(round((time - self.t) / self.dt))
            for _ in range(nsub):
                self._step(self.dt)
            self.t = time
        return self._make_frame(max(time, self.t) if time > 0.0 else 0.0)

    def diagnostics(self) -> dict[str, object]:
        z = self.x[:, 2]
        return {
            "time": float(self.t),
            "bending_energy": float(self._last_Eb),
            "twist_energy": float(self._last_Et),
            "max_out_of_plane": float(np.max(np.abs(z))),
            "substep_dt": float(self.dt),
        }
