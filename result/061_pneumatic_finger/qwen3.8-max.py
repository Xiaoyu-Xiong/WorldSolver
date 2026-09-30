"""Physically-based solver for the six-chamber pneumatic soft finger.

Model
-----
The finger is a chambered elastomer beam whose strain-limiting passive wall
(the continuous underside slab) is much stiffer in extension than the narrow
compliant hinge gaps between the six inflatable chamber lobes.  Bending is
therefore concentrated at the six hinge stations between the chambers.

The dynamics are a planar (x-z) six-link rigid-chain (Cosserat-style) model:

* generalized coordinates : absolute link angles ``Theta_j``, j = 1..6
* hinges                  : torsional springs (stiffness ``k``) + viscous
                            damping (``c``) located at the compliant gaps
* driving load            : pneumatic opening moment per hinge,
                            ``M_p = p(t) * A_plan * e`` where ``A_plan`` is the
                            chamber plan area, ``e`` the lever arm from the
                            hinge (slab mid-height) to the chamber centroid,
                            and ``p(t)`` the prescribed chamber pressure ramp.
                            Because the chambers sit above the passive wall,
                            the moment curls the finger away from the chamber
                            side (tip rotates downward, -z), the classic
                            PneuNet bending response.
* gravity                 : included through the exact generalized forces.

Equations of motion ``M(Theta) Theta_ddot = Q`` are integrated with RK4 at a
fixed 2e-4 s substep.  Vertex positions of the render mesh are obtained by an
Euler-Bernoulli backbone mapping: the backbone integrates the (piecewise
linear, smoothly blended) rotation field theta(x); each mesh vertex is carried
by the backbone pose at its material x-station with its cross-section offset
rotated rigidly.  This keeps the base clamped (x <= 0.03 m maps identically),
bends several chamber sections relative to their neighbours, and holds a
curved shape while pressure is held.

All required raw trajectory fields are exported per frame:
``attributes.mass_quantity``, ``fields.budget_external_force``,
``fields.cumulative_mass_source``, ``fields.element_deformation_gradient``,
``scalars.reference_force``.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

G_ACC = 9.81
RHO = 1100.0  # silicone density kg/m^3

# --- geometry of the generated finger (meters) -----------------------------
SLAB_Y0, SLAB_Y1, SLAB_Z0, SLAB_Z1 = -0.020, 0.020, 0.0, 0.014
SLAB_X1 = 0.210
CHAM_W = 0.048          # y extent of a chamber lobe
CHAM_H = 0.036          # z extent (0.014 -> 0.050)
CHAM_LEN = 0.027
TIP_VOL = 0.020 * 0.040 * 0.030
Z_NEUTRAL = 0.007       # mid-height of the passive slab (bending neutral axis)

# hinge stations = centres of the compliant gaps between chamber lobes
HINGES = np.array([0.0375, 0.0695, 0.1015, 0.1335, 0.1655, 0.1975])
X_END = 0.220
N_LINKS = 6

# link spans: link j (1-based) runs from HINGES[j-1] to HINGES[j] (tip: X_END)
LINK_LEN = np.array([HINGES[0]] * 0 + [])  # placeholder replaced below
_spans = np.diff(np.concatenate([HINGES, [X_END]]))          # lengths links 1..6
LINK_LEN = _spans

SLAB_MASS_PER_X = (SLAB_Y1 - SLAB_Y0) * (SLAB_Z1 - SLAB_Z0) * RHO
CHAM_MASS = CHAM_LEN * CHAM_W * CHAM_H * RHO
TIP_MASS = TIP_VOL * RHO

def _link_masses():
    m = np.zeros(N_LINKS)
    for j in range(N_LINKS):
        x0, x1 = HINGES[j] - LINK_LEN[j], HINGES[j]
        # slab contribution within the span (slab ends at 0.210)
        sx = max(0.0, min(x1, SLAB_X1) - x0)
        m[j] += sx * SLAB_MASS_PER_X
    # one chamber lobe inside each of links 1..5 (chamber 1 lies in the fixed base)
    m[0:5] += CHAM_MASS
    # terminal block belongs to the last link
    m[5] += TIP_MASS
    return m

LINK_MASS = _link_masses()
LINK_R = np.array([0.016, 0.016, 0.016, 0.016, 0.016, 0.0110])
LINK_INERTIA = np.array([
    LINK_MASS[j] * (LINK_LEN[j] ** 2 + (0.043 if j < 5 else 0.030) ** 2) / 12.0
    for j in range(N_LINKS)
])

TOTAL_MASS = float(
    SLAB_X1 * SLAB_MASS_PER_X + 6.0 * CHAM_MASS + TIP_MASS
)

# --- actuation / material parameters ---------------------------------------
K_HINGE = 1.8     # N*m/rad torsional stiffness of the compliant gap
C_HINGE = 0.05    # N*m*s/rad hinge damping (near-critical, quasi-static跟踪)
A_PLAN = CHAM_LEN * CHAM_W            # chamber plan area
E_ARM = 0.032 - Z_NEUTRAL             # hinge -> chamber centroid lever arm
# chambers downstream of hinges 1..5 drive them; the tip hinge has no chamber
MP_MASK = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])

DT_SUB = 2.0e-4


class BenchmarkSolver:
    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = scene.config
        pr = cfg["pressure"]
        self.p_target = float(pr["chamber_pressure_pa"])
        self.t_ramp0 = float(pr["ramp_start_s"])
        self.t_ramp1 = float(pr["ramp_end_s"])
        self.t_hold = float(pr["hold_until_s"])

        state0 = scene.initial_state()
        self.rest = np.asarray(state0.position, dtype=float).copy()
        self.n_vert = self.rest.shape[0]
        self.fixed_base = np.asarray(state0.attributes["fixed_base"], dtype=float).copy()

        # vertex mass: distribute each 8-vertex box mass over its vertices
        vm = np.zeros(self.n_vert)
        box_slab_m = SLAB_X1 * SLAB_MASS_PER_X / 8.0
        box_cham_m = CHAM_MASS / 8.0
        box_tip_m = TIP_MASS / 8.0
        nb = self.n_vert // 8
        for b in range(nb):
            if b == 0:
                m = box_slab_m
            elif 1 <= b <= 6:
                m = box_cham_m
            else:
                m = box_tip_m
            vm[8 * b:8 * b + 8] = m
        self.vertex_mass = vm

        # rest boxes for deformation-gradient export (8 boxes, fixed order)
        self.rest_boxes = [self.rest[8 * b:8 * b + 8] for b in range(nb)]
        self.box_pinv = []
        for X in self.rest_boxes:
            Xc = X - X.mean(axis=0)
            self.box_pinv.append(np.linalg.pinv(Xc))

        # backbone sampling grid for the vertex mapping
        self.grid = np.linspace(HINGES[0], X_END + 1e-3, 401)

        # dynamic state
        self.theta = np.zeros(N_LINKS)
        self.omega = np.zeros(N_LINKS)
        self.t = 0.0
        self.pos_prev = self.rest.copy()
        self.theta_v_prev = np.zeros(self.n_vert)
        self.t_prev = 0.0
        self.momentum_prev = np.zeros(3)
        self.impulse_int = np.zeros(3)   # cumulative external impulse (diag)

        self.p_now = 0.0
        self._diag = {}

    # ------------------------------------------------------------- pressure
    def pressure(self, t: float) -> float:
        if t <= self.t_ramp0:
            return 0.0
        if t >= self.t_ramp1:
            return self.p_target
        frac = (t - self.t_ramp0) / (self.t_ramp1 - self.t_ramp0)
        return self.p_target * frac

    # ------------------------------------------------------------- dynamics
    def _deriv(self, y: np.ndarray, t: float):
        th = y[:N_LINKS]
        om = y[N_LINKS:]
        ct, st = np.cos(th), np.sin(th)
        # d c_a / d Theta_j  (2 x N per link a)
        J = np.zeros((N_LINKS, N_LINKS, 2))
        for a in range(N_LINKS):
            for j in range(a + 1):
                if j < a:
                    J[a, j] = (LINK_LEN[j] * (-st[j]), LINK_LEN[j] * ct[j])
                else:
                    J[a, j] = (LINK_R[a] * (-st[a]), LINK_R[a] * ct[a])
        M = np.zeros((N_LINKS, N_LINKS))
        for a in range(N_LINKS):
            M += LINK_MASS[a] * (J[a].T @ J[a])
        M += np.diag(LINK_INERTIA)
        # gravity generalized forces: Q_j = sum_a m_a * g . dc_a/dTheta_j
        g_vec = np.array([0.0, -G_ACC])
        Q = np.sum(LINK_MASS[:, None, None] * J * g_vec[None, None, :], axis=(0, 2))
        # hinge torques
        phi = th.copy()
        phi[1:] -= th[:-1]
        dphi = om.copy()
        dphi[1:] -= om[:-1]
        mp = self.pressure(t) * A_PLAN * E_ARM * MP_MASK
        tau = -K_HINGE * phi - C_HINGE * dphi - mp
        Qh = tau.copy()
        Qh[:-1] -= tau[1:]
        Q = Q + Qh
        acc = np.linalg.solve(M, Q)
        return np.concatenate([om, acc])

    def _integrate_to(self, t_target: float) -> None:
        while self.t < t_target - 1e-12:
            dt = min(DT_SUB, t_target - self.t)
            y = np.concatenate([self.theta, self.omega])
            t0 = self.t
            k1 = self._deriv(y, t0)
            k2 = self._deriv(y + 0.5 * dt * k1, t0 + 0.5 * dt)
            k3 = self._deriv(y + 0.5 * dt * k2, t0 + 0.5 * dt)
            k4 = self._deriv(y + dt * k3, t0 + dt)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.theta = y[:N_LINKS]
            self.omega = y[N_LINKS:]
            self.t = t0 + dt

    # ------------------------------------------------------ vertex kinematics
    def _theta_of_x(self, x: np.ndarray) -> np.ndarray:
        """Piecewise-linear blended rotation field theta(x)."""
        cum = np.concatenate([[0.0], self.theta])  # Theta_0 = 0
        ctrl_x = np.concatenate([[HINGES[0] - 1e-9], HINGES])
        ctrl_th = cum  # theta(h_j) = Theta_j ; 0 before first hinge
        th = np.interp(x, ctrl_x, ctrl_th, left=0.0, right=ctrl_th[-1])
        return th

    def _positions(self):
        th_v = self._theta_of_x(self.rest[:, 0])
        # backbone: integrate direction from the fixed hinge station
        th_g = self._theta_of_x(self.grid)
        ux, uz = np.cos(th_g), np.sin(th_g)
        ds = np.diff(self.grid)
        bx = HINGES[0] + np.concatenate([[0.0], np.cumsum(0.5 * (ux[1:] + ux[:-1]) * ds)])
        bz = Z_NEUTRAL + np.concatenate([[0.0], np.cumsum(0.5 * (uz[1:] + uz[:-1]) * ds)])
        xr = np.clip(self.rest[:, 0], self.grid[0], self.grid[-1])
        base_x = np.where(self.rest[:, 0] <= HINGES[0], self.rest[:, 0], np.interp(xr, self.grid, bx))
        base_z = np.where(self.rest[:, 0] <= HINGES[0], Z_NEUTRAL, np.interp(xr, self.grid, bz))
        dz = self.rest[:, 2] - Z_NEUTRAL
        sth, cth = np.sin(th_v), np.cos(th_v)
        pos = np.empty_like(self.rest)
        pos[:, 0] = base_x + dz * sth
        pos[:, 1] = self.rest[:, 1]
        pos[:, 2] = base_z + dz * cth
        return pos, th_v

    # ---------------------------------------------------------------- frame
    def _make_frame(self, t: float) -> StateFrame:
        pos, th_v = self._positions()
        if t <= 0.0:
            vel = np.zeros_like(pos)
            ang_vel = np.zeros_like(pos)
            dt_frame = None
        else:
            dt_frame = t - self.t_prev
            vel = (pos - self.pos_prev) / dt_frame
            ang_vel = np.zeros_like(pos)
            ang_vel[:, 1] = -(th_v - self.theta_v_prev) / dt_frame

        # orientation quaternion (xyzw): rotation about -y by theta
        q = np.zeros((self.n_vert, 4))
        q[:, 1] = -np.sin(0.5 * th_v)
        q[:, 3] = np.cos(0.5 * th_v)

        # per-element deformation gradient (affine least-squares fit per box)
        Fs = np.zeros((len(self.rest_boxes), 3, 3))
        for b, (X, pinv) in enumerate(zip(self.rest_boxes, self.box_pinv)):
            xc = pos[8 * b:8 * b + 8]
            xc = xc - xc.mean(axis=0)
            # least squares: x_c ~ X_c F^T  =>  F^T = pinv(X_c) @ x_c
            Fs[b] = (pinv @ xc).T

        # external-force budget: G = [soft_finger, base_support], K = 1 material
        p_now = self.pressure(t)
        mom = (self.vertex_mass[:, None] * vel).sum(axis=0)
        grav = np.array([0.0, 0.0, -TOTAL_MASS * G_ACC])
        if dt_frame is None:
            dpdt = np.zeros(3)
        else:
            dpdt = (mom - self.momentum_prev) / dt_frame
        # sealed chambers: net pressure traction on the solid is ~0; the anchor
        # (base support) supplies the reaction that closes the momentum budget.
        f_anchor = dpdt - grav
        budget = np.zeros((2, 1, 3))
        budget[0, 0] = grav + f_anchor          # finger: gravity + pressure(0) + anchor
        budget[1, 0] = -f_anchor                # support: reaction pair (static, massless)

        frame = StateFrame(
            time=float(t),
            position=pos,
            velocity=vel,
            orientation=q,
            angular_velocity=ang_vel,
            mass=self.vertex_mass.copy(),
            body_id=np.zeros(self.n_vert, dtype=np.int64),
            attributes={
                "mass_quantity": self.vertex_mass.copy(),
                "fixed_base": self.fixed_base.copy(),
                "rest_position": self.rest.copy(),
            },
            fields={
                "budget_external_force": budget,
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": Fs,
            },
            scalars={
                "reference_force": TOTAL_MASS * G_ACC,
                "chamber_pressure_pa": p_now,
            },
        )
        frame.validate()

        self.pos_prev = pos
        self.theta_v_prev = th_v
        self.t_prev = t
        self.momentum_prev = mom
        self.p_now = p_now
        return frame

    # -------------------------------------------------------------- API
    def advance_to(self, time: float):
        time = float(time)
        if time > self.t:
            self._integrate_to(time)
        return self._make_frame(time)

    def diagnostics(self) -> dict:
        tip = self.pos_prev[np.argmax(self.rest[:, 0])]
        return {
            "model": "planar six-link hinge chain with pneumatic opening moments",
            "final_time_s": float(self.t_prev),
            "final_pressure_pa": float(self.p_now),
            "hinge_angles_rad": [float(a) for a in self.theta],
            "total_bend_rad": float(self.theta.sum()),
            "tip_position_m": [float(v) for v in tip],
            "total_mass_kg": float(TOTAL_MASS),
            "substep_s": DT_SUB,
        }
