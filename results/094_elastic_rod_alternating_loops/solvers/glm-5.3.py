"""Physics solver: alternating elastic-rod loops deposited on a moving belt.

Model
-----
The rod is a Lagrangian filament of position nodes with uniform arc-length
spacing ``ds``.  Elasticity follows the frozen material parameters:

* stretch: Hookean edge springs with stiffness ``E A / ds`` (the rod is
  essentially inextensible because ``E A`` is large),
* bending: discrete bending energy ``B/(2 ds^3) |x_j - 2 x_{j+1} + x_{j+2}|^2``
  with ``B = E I`` and a naturally straight rest shape.

Integration is a linearized semi-implicit Euler scheme: the tangent stiffness
of the elastic forces enters the left-hand side, a banded matrix solved with
a direct banded factorization every step (unconditionally stable and
deterministic).  Uniform gravity acts downward, with mass-proportional
damping consistent with the quasi-static regime.

Boundary conditions follow the frozen configuration:

* the not-yet-deployed supply is kinematically fed along the injector axis at
  the injection speed; a material node is released just below the outlet, so
  both inlet position and inlet direction are controlled;
* the belt is a rigid horizontal substrate translating at the belt speed; a
  node that touches the belt sticks permanently (no slip) and is thereafter
  carried with the belt, matching the frozen deposited-rod condition.

Because the belt speed is one half of the injection speed, the deposited
trace is compressed, the touchdown point repeatedly turns back, and loops are
laid on alternating sides of the belt-aligned trace.  Nothing is keyframed;
the pattern emerges from the time-stepping dynamics.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Deterministic filament-dynamics solver for the alternating-loop scene."""

    DT = 0.01          # integration step (s); two steps per output sample
    DAMPING = 8.0      # mass-proportional damping (1/s), quasi-static regime
    U_BAND = 8         # scalar bandwidth of the implicit system

    def __init__(self) -> None:
        self.ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        rod = cfg["rod"]
        inj = cfg["injector"]
        belt = cfg["belt"]

        self.g = np.asarray(cfg["gravity"], dtype=float)
        self.r = float(rod["radius"])
        self.rho = float(rod["density"])
        self.E = float(rod["young_modulus"])
        self.area = np.pi * self.r ** 2
        self.rhoA = self.rho * self.area
        self.B = self.E * (np.pi * self.r ** 4 / 4.0)
        self.ks = self.E * self.area / 1.0          # set after ds is known
        self.kb = self.B / 1.0

        self.U = float(inj["injection_speed"])
        self.V = float(belt["speed"])
        bdir = np.asarray(belt["travel_direction"], dtype=float)
        self.bdir = bdir / np.linalg.norm(bdir)
        self.v_belt = self.bdir * self.V
        self.outlet = np.asarray(inj["outlet_position"], dtype=float)
        self.outlet_z = float(self.outlet[2])
        self.axis = np.asarray(inj["axis"], dtype=float)
        self.axis = self.axis / np.linalg.norm(self.axis)
        self.v_inlet = self.axis * self.U

        self.L_total = float(rod["total_available_length"])
        self.L0 = float(rod["initial_deployed_length"])
        self.Lg = float(rod["gravito_bending_length"])

        # ---- discretization: uniform spacing, initial deployed part + supply
        n_edges0 = 171
        self.ds = self.L0 / n_edges0
        self.j_outlet = n_edges0
        n_supply = int(np.floor((self.L_total - self.L0) / self.ds))
        self.N = n_edges0 + 1 + n_supply
        self.supply_len = n_supply * self.ds

        self.ks = self.E * self.area / self.ds       # edge spring stiffness
        self.kb = self.B / self.ds ** 3              # bending stiffness

        # ---- initial geometry: resample the scene's initial centerline
        frame0 = scene.initial_state()
        poly = np.asarray(frame0.position, dtype=float)   # outlet -> bottom end
        seg = np.diff(poly, axis=0)
        seglen = np.linalg.norm(seg, axis=1)
        s_out = np.concatenate([[0.0], np.cumsum(seglen)])
        s_total = float(s_out[-1])
        kk = np.arange(n_edges0 + 1)
        s_node = np.clip(s_total - kk * self.ds, 0.0, s_total)
        x_dep = np.stack(
            [np.interp(s_node, s_out, poly[:, i]) for i in range(3)], axis=1
        )
        z_sup = self.outlet_z + (np.arange(1, n_supply + 1)) * self.ds
        x_sup = np.zeros((n_supply, 3))
        x_sup[:, 2] = z_sup
        x_sup += self.outlet - np.array([0.0, 0.0, self.outlet_z])
        self.x = np.vstack([x_dep, x_sup])          # node 0 = bottom (free end)

        # node masses (lumped, half at the two material ends)
        self.m = np.full(self.N, self.rhoA * self.ds)
        self.m[0] *= 0.5
        self.m[-1] *= 0.5

        self.v = np.zeros((self.N, 3))

        # sticking state
        self.stuck = np.zeros(self.N, dtype=bool)
        self.anchor = np.zeros((self.N, 3))
        self.initial_stuck = np.zeros(self.N, dtype=bool)
        # the short newly-contacted seed: bottom end rests on the moving belt
        self.stuck[0] = True
        self.initial_stuck[0] = True
        self.anchor[0] = self.x[0] - self.v_belt * 0.0
        self.x[0, 2] = self.r
        self.anchor[0, 2] = self.r
        self.v[0] = self.v_belt
        self.last_stuck = 0

        self.n_steps = 0
        self.t = 0.0
        # per-edge material rest lengths: the natural (stress-free) stretch
        # state is the frozen initial centerline, so the initial stretch
        # force vanishes exactly; supply edges are exactly ds
        self.Lrest = np.linalg.norm(np.diff(self.x, axis=0), axis=1)
        self._pinned, self._zpres = self._pinned_at(0.0)
        self._prev_tangent = self._tangents(self.x)
        self._prev_frame_t = 0.0

        self._build_assembly()

        self.ref_force = float(self.rhoA * self.L_total * np.linalg.norm(self.g))
        self.ref_torque = float(self.ref_force * self.Lg)
        self.ready = True

    # ------------------------------------------------------- kinematics helpers
    def _pinned_at(self, t):
        """Pinned (position-controlled) supply nodes at time ``t``."""
        j = np.arange(self.N, dtype=float)
        zpres = self.outlet_z + (j - self.j_outlet) * self.ds - self.U * t
        # a node is held until it has passed two spacings below the outlet,
        # which keeps the inlet position and direction controlled
        pinned = zpres > (self.outlet_z - 2.0 * self.ds)
        pinned[-1] = True   # the top material end is always still being fed
        return pinned, zpres

    def _tangents(self, x):
        tan = np.empty_like(x)
        tan[0] = x[1] - x[0]
        tan[-1] = x[-1] - x[-2]
        tan[1:-1] = x[2:] - x[:-2]
        n = np.linalg.norm(tan, axis=1)
        bad = n < 1e-12
        if np.any(bad):
            tan[bad] = np.array([0.0, 0.0, 1.0])
            n[bad] = 1.0
        return tan / n[:, None]

    # ------------------------------------------------------------ force model
    def _forces(self, x):
        """Internal (stretch + bending) plus gravity nodal forces."""
        N = self.N
        f = np.zeros((N, 3))
        d = x[1:] - x[:-1]
        L = np.linalg.norm(d, axis=1)
        L = np.maximum(L, 1e-12)
        dhat = d / L[:, None]
        s_e = self.ks * (L - self.Lrest)
        fe = s_e[:, None] * dhat
        f[:-1] += fe
        f[1:] -= fe
        C = x[:-2] - 2.0 * x[1:-1] + x[2:]
        f[:-2] -= self.kb * C
        f[1:-1] += 2.0 * self.kb * C
        f[2:] -= self.kb * C
        f += self.m[:, None] * self.g
        return f

    def _edge_tangent_stiffness(self, x):
        d = x[1:] - x[:-1]
        L = np.linalg.norm(d, axis=1)
        L = np.maximum(L, 1e-12)
        dhat = d / L[:, None]
        s_e = self.ks * (L - self.Lrest)
        outer = dhat[:, :, None] * dhat[:, None, :]
        eye = np.eye(3)
        T = self.ks * outer + (s_e / L)[:, None, None] * (eye - outer)
        return T

    # ---------------------------------------------------- banded assembly setup
    def _build_assembly(self):
        N = self.N
        M = 3 * N
        u = self.U_BAND
        self.M = M
        dt = self.DT

        a3, b3 = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")

        def block_idx(rn, cn):
            r = 3 * rn[:, None, None] + a3[None, :, :]
            c = 3 * cn[:, None, None] + b3[None, :, :]
            return ((u + r - c) * M + c).reshape(-1)

        E = N - 1
        p = np.arange(E)
        q = p + 1
        idx_stretch = np.concatenate(
            [block_idx(p, p), block_idx(q, q), block_idx(p, q), block_idx(q, p)]
        )
        self._n_stretch_per_block = E * 9

        # mass matrix (diagonal): m * (1/dt + damping)
        nodes = np.arange(N)
        midx = (u * M + 3 * nodes[:, None] + np.arange(3)[None, :]).reshape(-1)
        mvals = np.repeat(self.m * (1.0 / dt + self.DAMPING), 3)

        # bending blocks (constant, dt fixed): joints (a, b, c)
        J = np.arange(N - 2)
        a, b, c = J, J + 1, J + 2
        w_dtk = self.kb * dt
        bend_idx = []
        bend_vals = []
        k3 = np.arange(3)
        for rn, cn, w in (
            (a, a, 1.0), (c, c, 1.0), (b, b, 4.0),
            (a, b, -2.0), (b, a, -2.0), (b, c, -2.0), (c, b, -2.0),
            (a, c, 1.0), (c, a, 1.0),
        ):
            # each bending block is w * dt * kb * I: store its diagonal only
            r = 3 * rn[:, None] + k3[None, :]        # (J, 3)
            col = 3 * cn[:, None] + k3[None, :]      # (J, 3)
            flat = (u + r - col) * M + col           # (J, 3)
            bend_idx.append(flat.reshape(-1))
            bend_vals.append(np.full(flat.size, w * w_dtk))

        bend_idx = np.concatenate(bend_idx)
        bend_vals = np.concatenate(bend_vals)

        self._const_idx = np.concatenate([midx, bend_idx])
        self._const_vals = np.concatenate([mvals, bend_vals])
        self._idx_all = np.concatenate([self._const_idx, idx_stretch]).astype(np.int64)
        self._n_const = self._const_idx.size

    # -------------------------------------------------------------- time step
    def _step(self) -> None:
        dt = self.DT
        N = self.N
        M = self.M
        u = self.U_BAND

        t0 = self.n_steps * dt
        t1 = (self.n_steps + 1) * dt

        pinned, zpres = self._pinned_at(t1)
        x = self.x
        f = self._forces(x)

        # ---- assemble banded implicit matrix  A = M(1/dt+c) - dt*K
        T = self._edge_tangent_stiffness(x)
        Tf = T.reshape(-1)
        vals = np.empty(self._idx_all.size)
        vals[: self._n_const] = self._const_vals
        n = self._n_stretch_per_block
        vals[self._n_const + 0 * n: self._n_const + 1 * n] = dt * Tf
        vals[self._n_const + 1 * n: self._n_const + 2 * n] = dt * Tf
        vals[self._n_const + 2 * n: self._n_const + 3 * n] = -dt * Tf
        vals[self._n_const + 3 * n: self._n_const + 4 * n] = -dt * Tf
        ab = np.bincount(self._idx_all, weights=vals, minlength=(2 * u + 1) * M)
        ab = ab.reshape(2 * u + 1, M)

        # ---- constrained dofs: pinned supply + stuck (belt) nodes
        v_pres = np.zeros((N, 3))
        v_pres[pinned] = self.v_inlet
        v_pres[self.stuck] = self.v_belt
        constr = pinned | self.stuck
        nodes_c = np.where(constr)[0]
        R = (3 * nodes_c[:, None] + np.arange(3)[None, :]).reshape(-1)

        # replace constrained rows by identity rows (their prescribed velocity
        # enters the remaining equations through the retained columns)
        for o in range(2 * u + 1):
            jj = R + (u - o)
            ok = (jj >= 0) & (jj < M)
            if np.any(ok):
                ab[o, jj[ok]] = 0.0
        ab[u, R] = 1.0

        rhs = (self.m / dt)[:, None] * self.v + f
        rhs[nodes_c] = v_pres[nodes_c]

        v_new = solve_banded((u, u), ab, rhs.ravel()).reshape(N, 3)
        x_new = x + dt * v_new

        # ---- enforce prescribed kinematics exactly
        v = v_new
        v[pinned] = self.v_inlet
        x_new[pinned, 0] = self.outlet[0]
        x_new[pinned, 1] = self.outlet[1]
        x_new[pinned, 2] = zpres[pinned]
        xs = self.stuck
        v[xs] = self.v_belt
        x_new[xs] = self.anchor[xs] + self.v_belt * t1

        # ---- sticking contact with the translating belt
        free = ~(pinned | xs)
        touch = free & (x_new[:, 2] <= self.r + 1e-12)
        if np.any(touch):
            idx = np.where(touch)[0]
            x_new[idx, 2] = self.r
            self.anchor[idx] = x_new[idx] - self.v_belt * t1
            self.stuck[idx] = True
            v[idx] = self.v_belt
            self.last_stuck = int(idx[-1])

        self.x = x_new
        self.v = v
        self.n_steps += 1
        self.t = t1
        self._pinned = pinned
        self._zpres = zpres

    # ----------------------------------------------------------------- output
    def advance_to(self, time: float):
        if not self.ready:
            raise RuntimeError("solver not initialized")
        t = float(time)
        n_target = int(round(t / self.DT))
        while self.n_steps < n_target:
            self._step()
        return self._frame(t)

    def _frame(self, t: float):
        N = self.N
        x, v = self.x, self.v
        pinned, zpres = self._pinned, self._zpres
        if abs(self.t - t) > 1e-9:
            pinned, zpres = self._pinned_at(t)
        stuck = self.stuck
        f = self._forces(x)

        # ---- budget terms (whole-rod domain, three external terms)
        Fg = (self.m[:, None] * self.g).sum(axis=0)
        pg = (self.m[:, None] * x).sum(axis=0) / self.m.sum()
        nodes_p = np.where(pinned)[0]
        Fin = -f[nodes_p].sum(axis=0)
        grip = int(nodes_p.min())
        p_grip = x[grip].copy()
        nodes_s = np.where(stuck)[0]
        if nodes_s.size:
            Fb = -f[nodes_s].sum(axis=0)
            w = np.linalg.norm(f[nodes_s], axis=1)
            sw = w.sum()
            pb = (w[:, None] * x[nodes_s]).sum(axis=0) / sw if sw > 1e-15 else x[nodes_s].mean(axis=0)
        else:
            Fb = np.zeros(3)
            pb = np.zeros(3)

        budget_force = np.array([[Fg, Fin, Fb]], dtype=float)
        budget_pos = np.array([[pg, p_grip, pb]], dtype=float)
        budget_couple = np.zeros((1, 3, 3))

        # ---- constraint pairs: inlet kinematics and sticking deposition
        z_grip = self.outlet_z + (grip - self.j_outlet) * self.ds - self.U * t
        cval = np.array([x[grip], v[self.last_stuck]], dtype=float)
        ctgt = np.array(
            [[self.outlet[0], self.outlet[1], z_grip], self.v_belt], dtype=float
        )

        # ---- mass bookkeeping
        added_len = min(self.U * t, self.supply_len)
        cum_source = np.array([self.rhoA * added_len])

        # ---- orientation / angular velocity from centerline tangents
        tan = self._tangents(x)
        c = tan[:, 2]
        wq = np.sqrt(np.clip((1.0 + c) / 2.0, 0.0, 1.0))
        sq = np.sqrt(np.clip((1.0 - c) / 2.0, 0.0, 1.0))
        axis = np.stack([-tan[:, 1], tan[:, 0], np.zeros(N)], axis=1)
        nrm = np.linalg.norm(axis, axis=1)
        bad = nrm < 1e-10
        if np.any(bad):
            axis[bad] = np.array([1.0, 0.0, 0.0])
            nrm[bad] = 1.0
        axis = axis / nrm[:, None]
        quat = np.empty((N, 4))
        quat[:, 0] = sq * axis[:, 0]
        quat[:, 1] = sq * axis[:, 1]
        quat[:, 2] = sq * axis[:, 2]
        quat[:, 3] = wq

        dtf = t - self._prev_frame_t
        if dtf > 1e-12:
            omega = np.cross(self._prev_tangent, tan) / dtf
        else:
            omega = np.zeros((N, 3))
        self._prev_tangent = tan
        self._prev_frame_t = t

        attributes = {
            "mass_quantity": self.m.copy(),
            "arc_coordinate": np.arange(N) * self.ds,
            "is_injector_controlled": pinned.astype(float),
            "is_initially_deposited": self.initial_stuck.astype(float),
            "is_sticking_to_belt": stuck.astype(float),
            "is_dynamic": (~(pinned | stuck)).astype(float),
            "rod_radius": np.full(N, self.r),
            "natural_curvature": np.zeros(N),
        }
        fields = {
            "budget_external_force": budget_force,
            "budget_force_position": budget_pos,
            "budget_external_couple": budget_couple,
            "constraint_value": cval,
            "constraint_target": ctgt,
            "cumulative_mass_source": cum_source,
            "external_force": self.m[:, None] * self.g,
        }
        scalars = {
            "constraint_scale": self.L0,
            "reference_force": self.ref_force,
            "reference_torque": self.ref_torque,
            "injection_speed": self.U,
            "belt_speed": self.V,
            "belt_to_injection_speed_ratio": self.V / self.U,
            "released_length": self.L0 + added_len,
        }
        return StateFrame(
            time=t,
            position=x.copy(),
            velocity=v.copy(),
            orientation=quat,
            angular_velocity=omega,
            mass=self.m.copy(),
            body_id=np.zeros(N, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    def diagnostics(self) -> dict:
        if not self.ready:
            return {"success": False}
        seg = np.linalg.norm(np.diff(self.x, axis=0), axis=1)
        return {
            "method": "implicit filament dynamics (banded solve), sticking belt contact",
            "steps": int(self.n_steps),
            "nodes": int(self.N),
            "stuck_nodes": int(self.stuck.sum()),
            "min_edge_length": float(seg.min()),
            "max_edge_length": float(seg.max()),
            "min_z": float(self.x[:, 2].min()),
            "max_speed": float(np.linalg.norm(self.v, axis=1).max()),
            "time": float(self.t),
        }
