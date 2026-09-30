"""Poro-elasto-capillary sponge: saturation under a faucet and driven wringing.

Self-implemented physics (NumPy/SciPy only, no external simulator):

* Solid: the sponge is a 9x4 corner lattice (identical ordering to
  ``assets/sponge_topology.obj`` and to ``scene.py``) carried by a damped
  structural / shear / bending spring network, integrated with semi-implicit
  Euler at dt_s = 5e-5 s (10 substeps per fluid step).
* Pore liquid: per-vertex stored liquid mass with a Darcy/Richards style
  transport on the lattice graph (capillary potential + gravity head, upwind
  relative permeability).  Saturation is S = pore_mass /
  (rho_w * phi * eta * V_local); the local bulk volume is measured from the
  *deformed* lattice, so twisting compresses pores, raises pore pressure and
  expels stored liquid.  Pore pressure pushes back on the solid matrix.
* Free water: weakly compressible SPH (cubic spline kernel, Tait EOS, Monaghan
  artificial viscosity), penalty sink walls, faucet inlet and an open drain.
* Two-way coupling: jet particles are absorbed into pores (mass and momentum
  transfer), supersaturated pores expel particles with the matching solid
  recoil, and particles that are not absorbed exchange penalty contact and drag
  impulse pairs with the deformed (twisting, narrowing) sponge faces.

Every impulse used by the integrator is accumulated exactly, so the exported
momentum budgets close to round-off for each declared domain.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

RHO_W = 1000.0
N_SEC = 9
N_COR = 4
N_NODE = N_SEC * N_COR

# --- solid network ---------------------------------------------------------
K_AXIAL = 2500.0
K_SECTION = 900.0
K_SHEAR = 900.0
K_BEND = 300.0
BETA_DAMP = 2.0e-4          # spring damping = BETA_DAMP * k  [N*s/m]
C_AIR_SOLID = 0.02          # ambient drag on sponge vertices [N*s/m]
V_MAX_SOLID = 4.0

# --- pore model ------------------------------------------------------------
ETA_RET = 0.20              # capillary-retained fraction of the pore space
P_CAP = 2000.0              # capillary potential scale [Pa]
K_PORE = 5.0e3              # pore stiffness for supersaturation [Pa]
G_LONG = 3.0e-5             # lattice conductance, along the sponge [kg/(Pa*s)]
G_CROSS = 4.0e-5            # lattice conductance, across a section
G_SURF = 1.2e-4             # surface leak conductance [kg/(Pa*s)]
ABS_GATE = 1.15             # absorption stops above this saturation
AREA_FAC = 0.25             # lateral area factor for the pore-pressure force
S_DRIP = 0.90

# --- SPH -------------------------------------------------------------------
H_SMOOTH = 0.011
R_SUPPORT = 2.0 * H_SMOOTH
C_SOUND = 8.0
GAMMA_TAIT = 7.0
ALPHA_VISC = 0.15
N_WATER_MAX = 1400
V_MAX_WATER = 6.0

# --- contacts --------------------------------------------------------------
K_WALL = 300.0
C_WALL = 0.6
K_FLOOR = 1500.0
C_FLOOR = 0.8
MU_FLOOR = 0.6
K_CONTACT = 40.0
C_CONTACT = 0.06
C_TANG = 0.02
MARGIN_CONTACT = 0.006
MARGIN_ABSORB = 0.002

DT_FLUID = 5.0e-4
N_SUBSTEP = 10

Z_FLOOR = 0.015
Y_BACK = 0.264
X_WALL = 0.384
Z_DRAIN = 0.0175
BOUND_MIN = np.array([-0.38, -0.27, -1.0])
BOUND_MAX = np.array([0.38, 0.27, 0.62])


def _smoothstep(t: float, t0: float, t1: float) -> tuple[float, float]:
    """C1 ramp value and its time derivative (zero slope at both ends)."""
    if t1 <= t0:
        return (1.0 if t >= t1 else 0.0), 0.0
    x = (t - t0) / (t1 - t0)
    if x <= 0.0:
        return 0.0, 0.0
    if x >= 1.0:
        return 1.0, 0.0
    return x * x * (3.0 - 2.0 * x), 6.0 * x * (1.0 - x) / (t1 - t0)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def _rot_x(vec: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    out = np.empty_like(vec)
    out[..., 0] = vec[..., 0]
    out[..., 1] = c * vec[..., 1] - s * vec[..., 2]
    out[..., 2] = s * vec[..., 1] + c * vec[..., 2]
    return out


def _kernel_w(q: np.ndarray, h: float) -> np.ndarray:
    """Cubic spline kernel, 3D, support 2h."""
    sigma = 1.0 / (np.pi * h ** 3)
    w = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    w[m1] = 1.0 - 1.5 * q[m1] ** 2 + 0.75 * q[m1] ** 3
    w[m2] = 0.25 * (2.0 - q[m2]) ** 3
    return sigma * w


def _kernel_dwdq(q: np.ndarray, h: float) -> np.ndarray:
    sigma = 1.0 / (np.pi * h ** 3)
    d = np.zeros_like(q)
    m1 = q < 1.0
    m2 = (q >= 1.0) & (q < 2.0)
    d[m1] = -3.0 * q[m1] + 2.25 * q[m1] ** 2
    d[m2] = -0.75 * (2.0 - q[m2]) ** 2
    return sigma * d / h

class BenchmarkSolver:
    """Self-implemented poro-elastic sponge + SPH water solver."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = dict(scene.config)
        self.rng = np.random.default_rng(int(seed))
        self.gravity = np.asarray(cfg["gravity"], float)
        ic = cfg["initial_conditions"]
        self.porosity = float(ic["sponge_porosity"])

        src = ic["water_source"]
        self.src_center = np.asarray(src["center"], float)
        direction = np.asarray(src["direction"], float)
        self.src_dir = direction / max(np.linalg.norm(direction), 1e-12)
        self.src_radius = float(src["radius"])
        self.src_speed = float(src["speed"])
        self.src_t0 = float(src["starts_at"])
        self.src_t1 = float(src["stops_at"])
        self.src_area = np.pi * self.src_radius ** 2
        self.mdot = RHO_W * self.src_area * self.src_speed

        controls = cfg["controls"]
        self.grips = []
        for key, lo, hi in (("left_end", 0, 8), ("right_end", 28, 36)):
            spec = controls[key]
            axis = np.asarray(spec["twist_axis"], float)
            self.grips.append({
                "body_id": int(spec["body_id"]),
                "c0": np.asarray(spec["initial_position"], float),
                "lift": np.asarray(spec["lift_displacement"], float),
                "angle": np.deg2rad(float(spec["twist_degrees"])) * np.sign(axis[0] or 1.0),
                "t_on": float(spec["active_from"]),
                "idx": np.arange(lo, hi, dtype=int),
                "offset": None,
            })
        ev = cfg["event_sequence"]
        self.lift_iv = [float(v) for v in ev["lift_interval"]]
        self.wring_iv = [float(v) for v in ev["wring_interval"]]
        self.q_base = np.array([0.0, 0.7071068, 0.0, 0.7071068])

        frame0 = scene.initial_state()
        pos = np.asarray(frame0.position, float)
        vel = np.asarray(frame0.velocity, float)
        mass = np.asarray(frame0.mass, float)
        bid = np.asarray(frame0.body_id)
        self.x_s = pos[bid == 1].copy()
        self.v_s = np.zeros_like(self.x_s)
        self.m_dry = mass[bid == 1].copy()
        self.x_w = pos[bid == 2].copy()
        self.v_w = vel[bid == 2].copy()
        self.m_p = float(mass[bid == 2][0])
        self.m_grip = mass[bid >= 10].copy()

        # ---- spring network on the 9x4 corner lattice -----------------
        edges: list[tuple[int, int]] = []
        stiff: list[float] = []
        pore_edges: list[tuple[int, int]] = []
        pore_cond: list[float] = []
        ring = ((0, 1), (1, 2), (2, 3), (3, 0))
        for s in range(N_SEC):
            b = s * N_COR
            for a, c in ring:
                edges.append((b + a, b + c))
                stiff.append(K_SECTION)
                pore_edges.append((b + a, b + c))
                pore_cond.append(G_CROSS)
            for a, c in ((0, 2), (1, 3)):
                edges.append((b + a, b + c))
                stiff.append(K_SECTION)
        for s in range(N_SEC - 1):
            b, n = s * N_COR, (s + 1) * N_COR
            for c in range(N_COR):
                edges.append((b + c, n + c))
                stiff.append(K_AXIAL)
                pore_edges.append((b + c, n + c))
                pore_cond.append(G_LONG)
            for a in range(N_COR):
                for c in range(N_COR):
                    if a != c:
                        edges.append((b + a, n + c))
                        stiff.append(K_SHEAR)
        for s in range(N_SEC - 2):
            b, n = s * N_COR, (s + 2) * N_COR
            for c in range(N_COR):
                edges.append((b + c, n + c))
                stiff.append(K_BEND)
        self.e_i = np.array([e[0] for e in edges], dtype=int)
        self.e_j = np.array([e[1] for e in edges], dtype=int)
        self.e_k = np.array(stiff, float)
        self.e_c = BETA_DAMP * self.e_k
        self.e_l0 = np.linalg.norm(self.x_s[self.e_i] - self.x_s[self.e_j], axis=1)
        self.p_i = np.array([e[0] for e in pore_edges], dtype=int)
        self.p_j = np.array([e[1] for e in pore_edges], dtype=int)
        self.p_g = np.array(pore_cond, float)

        # ---- pore state ------------------------------------------------
        self.pore = np.zeros(N_NODE)
        self.pend_m = np.zeros(N_NODE)
        self.pend_p = np.zeros((N_NODE, 3))
        self.sec_idx = np.arange(N_NODE).reshape(N_SEC, N_COR)

        vol0 = self._geom(self.x_s)[7]
        self.cap_ref = float(RHO_W * self.porosity * ETA_RET * vol0.sum())

        # ---- momentum budgets -----------------------------------------
        self.imp = np.zeros((3, 5, 3))     # cumulative external impulse terms
        self.f_now = np.zeros((3, 5, 3))   # instantaneous external forces
        self.in_mom = np.zeros((3, 3))
        self.out_mom = np.zeros((3, 3))
        self.src_mass = np.zeros(2)        # [solid, liquid] prescribed source
        self.inter = np.zeros((6, 3))      # paired interaction forces
        v_ref = float(np.sqrt(2.0 * 9.81 * 0.5))
        m_ref = float(self.m_dry.sum() + self.cap_ref + self.m_p * len(self.x_w))
        self.p_ref = max(m_ref * v_ref, 1e-6)

        self.t = 0.0
        # seed the reported instantaneous forces with the t=0 gravity loads so
        # the first exported frame carries physical (not zero) external terms
        self.f_now[0, 0] = self.m_dry.sum() * self.gravity
        self.f_now[1, 0] = self.m_p * len(self.x_w) * self.gravity
        self.emit_debt = 0.0
        self.n_abs = 0
        self.n_exp = 0
        self.n_drain = 0
        self.tree = None
        self.tree_age = 10 ** 9
        self.frame0 = self._make_frame()

    # ------------------------------------------------------------- geometry
    def _geom(self, x: np.ndarray):
        """Deformed slab frames, half extents and local bulk volumes."""
        p = x[self.sec_idx]                       # [S,4,3]
        cen = p.mean(axis=1)
        ey = (p[:, 1] + p[:, 2]) - (p[:, 0] + p[:, 3])
        ez = (p[:, 2] + p[:, 3]) - (p[:, 0] + p[:, 1])
        hy = np.maximum(0.25 * np.linalg.norm(ey, axis=1), 1e-4)
        hz = np.maximum(0.25 * np.linalg.norm(ez, axis=1), 1e-4)
        ey = ey / np.maximum(np.linalg.norm(ey, axis=1, keepdims=True), 1e-12)
        ez = ez - np.sum(ez * ey, axis=1, keepdims=True) * ey
        ez = ez / np.maximum(np.linalg.norm(ez, axis=1, keepdims=True), 1e-12)
        ex = np.cross(ey, ez)
        ex = ex / np.maximum(np.linalg.norm(ex, axis=1, keepdims=True), 1e-12)
        d = np.maximum(np.linalg.norm(np.diff(cen, axis=0), axis=1), 1e-4)
        dl = np.empty(N_SEC)
        dr = np.empty(N_SEC)
        dl[0] = d[0]
        dl[1:] = d
        dr[-1] = d[-1]
        dr[:-1] = d
        hx = 0.25 * (dl + dr)
        vol = np.maximum(8.0 * hx * hy * hz, 1e-9)
        return cen, ex, ey, ez, hx, hy, hz, vol

    def _node_capacity(self, vol: np.ndarray) -> np.ndarray:
        cap_sec = RHO_W * self.porosity * ETA_RET * vol
        return np.maximum(np.repeat(cap_sec / N_COR, N_COR), 1e-9)

    def _saturation(self, cap: np.ndarray) -> np.ndarray:
        return self.pore / cap

    @staticmethod
    def _pore_pressure(sat: np.ndarray) -> np.ndarray:
        return K_PORE * np.maximum(sat - 1.0, 0.0)

    def _node_mass(self) -> np.ndarray:
        return self.m_dry + self.pore

    # -------------------------------------------------------------- controls
    def _grip_command(self, grip: dict, t: float):
        """Prescribed grip centre pose/twist and the attached-node targets."""
        s_lift, ds_lift = _smoothstep(t, self.lift_iv[0], self.lift_iv[1])
        s_tw, ds_tw = _smoothstep(t, self.wring_iv[0], self.wring_iv[1])
        cen = grip["c0"] + s_lift * grip["lift"]
        vel = ds_lift * grip["lift"]
        ang = s_tw * grip["angle"]
        omega = ds_tw * grip["angle"]
        if grip["offset"] is None:
            return cen, vel, ang, omega, None, None
        off = _rot_x(grip["offset"], ang)
        x_t = cen + off
        v_t = vel + omega * np.stack(
            (np.zeros(len(off)), -off[:, 2], off[:, 1]), axis=1
        )
        return cen, vel, ang, omega, x_t, v_t

    # ------------------------------------------------------------------- SPH
    def _sph_forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        n = len(x)
        f = np.zeros((n, 3))
        if n < 2:
            return f
        if self.tree is None or self.tree_age >= 2 or self.tree.n != n:
            self.tree = cKDTree(x)
            self.tree_age = 0
            self.pairs = np.asarray(
                list(self.tree.query_pairs(R_SUPPORT)), dtype=int
            ).reshape(-1, 2)
        self.tree_age += 1
        pairs = self.pairs
        rho = np.full(n, self.m_p * _kernel_w(np.zeros(1), H_SMOOTH)[0])
        if len(pairs):
            d = x[pairs[:, 0]] - x[pairs[:, 1]]
            r = np.linalg.norm(d, axis=1)
            w = self.m_p * _kernel_w(r / H_SMOOTH, H_SMOOTH)
            np.add.at(rho, pairs[:, 0], w)
            np.add.at(rho, pairs[:, 1], w)
        self.rho_w = rho
        b_tait = RHO_W * C_SOUND ** 2 / GAMMA_TAIT
        pres = np.maximum(b_tait * ((rho / RHO_W) ** GAMMA_TAIT - 1.0), 0.0)
        if not len(pairs):
            return f
        i, j = pairs[:, 0], pairs[:, 1]
        r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        grad = (_kernel_dwdq(r / H_SMOOTH, H_SMOOTH) / r)[:, None] * d
        coeff = -(self.m_p ** 2) * (pres[i] / rho[i] ** 2 + pres[j] / rho[j] ** 2)
        fp = coeff[:, None] * grad
        dv = v[i] - v[j]
        mu = 0.5 * (rho[i] + rho[j])
        vis = (ALPHA_VISC * H_SMOOTH * C_SOUND * self.m_p ** 2 / mu)[:, None] * (
            dv / (r ** 2 + 0.01 * H_SMOOTH ** 2)[:, None]
        ) * np.sum(grad * d, axis=1)[:, None]
        pair_f = fp + vis
        np.add.at(f, i, pair_f)
        np.add.at(f, j, -pair_f)
        return f

    # ------------------------------------------------- water/sponge coupling
    def _classify(self, geom):
        """Nearest deformed slab, local coordinates and bilinear corner weights."""
        cen, ex, ey, ez, hx, hy, hz, _ = geom
        if not len(self.x_w):
            empty_i = np.zeros(0, dtype=int)
            return empty_i, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 4))
        rel = self.x_w[:, None, :] - cen[None, :, :]
        lx = np.einsum("psd,sd->ps", rel, ex)
        ly = np.einsum("psd,sd->ps", rel, ey)
        lz = np.einsum("psd,sd->ps", rel, ez)
        mx = np.abs(lx) - hx[None, :]
        my = np.abs(ly) - hy[None, :]
        mz = np.abs(lz) - hz[None, :]
        depth = np.maximum(np.maximum(mx, my), mz)
        sec = np.argmin(depth, axis=1)
        pid = np.arange(len(self.x_w))
        loc = np.stack((lx[pid, sec], ly[pid, sec], lz[pid, sec]), axis=1)
        mar = -np.stack((mx[pid, sec], my[pid, sec], mz[pid, sec]), axis=1)
        u = np.clip(0.5 * (loc[:, 1] / hy[sec] + 1.0), 0.0, 1.0)
        w = np.clip(0.5 * (loc[:, 2] / hz[sec] + 1.0), 0.0, 1.0)
        bw = np.stack((
            (1.0 - u) * (1.0 - w), u * (1.0 - w), u * w, (1.0 - u) * w,
        ), axis=1)
        return sec, loc, mar, bw

    def _coupling(self, geom, sec, mar, bw, absorb_mask):
        """Penalty contact + drag pairs between free water and the sponge."""
        f_w = np.zeros((len(self.x_w), 3))
        f_s = np.zeros((N_NODE, 3))
        if not len(self.x_w):
            return f_w, f_s
        _, ex, ey, ez, _, _, _, _ = geom
        hit = np.all(mar > -MARGIN_CONTACT, axis=1) & (~absorb_mask)
        if not np.any(hit):
            return f_w, f_s
        idx = np.nonzero(hit)[0]
        s = sec[idx]
        m = mar[idx]
        axis = np.argmin(m, axis=1)
        depth = np.maximum(m[np.arange(len(idx)), axis] + MARGIN_CONTACT, 0.0)
        basis = np.stack((ex[s], ey[s], ez[s]), axis=1)
        nrm = basis[np.arange(len(idx)), axis]
        loc_sgn = np.sign(np.sum((self.x_w[idx] - geom[0][s]) * nrm, axis=1))
        loc_sgn[loc_sgn == 0.0] = 1.0
        nrm = nrm * loc_sgn[:, None]
        node = self.sec_idx[s]                                  # [n,4]
        wgt = bw[idx]
        v_solid = np.einsum("nc,ncd->nd", wgt, self.v_s[node])
        dv = self.v_w[idx] - v_solid
        f_pair = (K_CONTACT * depth)[:, None] * nrm
        f_pair -= C_CONTACT * np.sum(dv * nrm, axis=1)[:, None] * nrm
        f_pair -= C_TANG * (dv - np.sum(dv * nrm, axis=1)[:, None] * nrm)
        np.add.at(f_w, idx, f_pair)
        np.add.at(f_s, node.reshape(-1), (-wgt[:, :, None] * f_pair[:, None, :]).reshape(-1, 3))
        return f_w, f_s

    def _wall_forces(self):
        """Penalty sink walls acting on the free water samples."""
        f = np.zeros((len(self.x_w), 3))
        if not len(self.x_w):
            return f
        x, v = self.x_w, self.v_w
        pen = Z_FLOOR - x[:, 2]
        act = pen > 0.0
        if np.any(act):
            fn = K_WALL * pen[act] - C_WALL * v[act, 2]
            fn = np.maximum(fn, 0.0)
            f[act, 2] += fn
            f[act, 0] -= C_WALL * v[act, 0]
            f[act, 1] -= C_WALL * v[act, 1]
        pen = x[:, 1] - Y_BACK
        act = pen > 0.0
        if np.any(act):
            f[act, 1] -= np.maximum(K_WALL * pen[act] + C_WALL * v[act, 1], 0.0)
        pen = x[:, 0] - X_WALL
        act = pen > 0.0
        if np.any(act):
            f[act, 0] -= np.maximum(K_WALL * pen[act] + C_WALL * v[act, 0], 0.0)
        pen = -X_WALL - x[:, 0]
        act = pen > 0.0
        if np.any(act):
            f[act, 0] += np.maximum(K_WALL * pen[act] - C_WALL * v[act, 0], 0.0)
        return f

    # ------------------------------------------------------ pore transport
    def _pore_transport(self, dt: float, cap: np.ndarray):
        """Darcy / Richards style redistribution of stored pore liquid."""
        sat = np.clip(self._saturation(cap), 0.0, 3.0)
        head = -P_CAP * (1.0 - np.minimum(sat, 1.0)) + self._pore_pressure(sat)
        head = head + RHO_W * self.gravity[2] * self.x_s[:, 2]
        dh = head[self.p_i] - head[self.p_j]
        don = np.where(dh > 0.0, self.p_i, self.p_j)
        kr = np.clip(sat[don], 0.0, 1.0) ** 3
        flow = self.p_g * kr * dh * dt
        avail = 0.25 * self.pore[don]
        flow = np.sign(flow) * np.minimum(np.abs(flow), avail)
        mom = self._node_mass()[:, None] * self.v_s
        np.add.at(self.pore, self.p_i, -flow)
        np.add.at(self.pore, self.p_j, flow)
        carry = flow[:, None] * self.v_s[don]
        np.add.at(mom, self.p_i, -carry)
        np.add.at(mom, self.p_j, carry)
        self.pore = np.maximum(self.pore, 0.0)
        self.v_s = mom / np.maximum(self._node_mass(), 1e-9)[:, None]

    def _absorb_mask(self, sec, mar, cap) -> np.ndarray:
        """Samples that are inside the sponge and may still be taken up."""
        if not len(self.x_w):
            return np.zeros(0, dtype=bool)
        sat_sec = self._saturation(cap).reshape(N_SEC, N_COR).mean(axis=1)
        inside = np.all(mar > MARGIN_ABSORB, axis=1)
        return inside & (sat_sec[sec] < ABS_GATE)

    def _absorb(self, sec, bw, mask) -> np.ndarray:
        """Capillary uptake of impinging jet water into the pore network."""
        if not len(self.x_w) or not np.any(mask):
            return mask
        idx = np.nonzero(mask)[0]
        node = self.sec_idx[sec[idx]].reshape(-1)
        wgt = bw[idx]
        dm = (self.m_p * wgt).reshape(-1)
        dp = (self.m_p * wgt[:, :, None] * self.v_w[idx][:, None, :]).reshape(-1, 3)
        mom = self._node_mass()[:, None] * self.v_s
        np.add.at(self.pore, node, dm)
        np.add.at(mom, node, dp)
        self.v_s = mom / np.maximum(self._node_mass(), 1e-9)[:, None]
        moved = dp.sum(axis=0)
        self.in_mom[0] += moved
        self.out_mom[1] += moved
        self.n_abs += len(idx)
        return mask

    def _expel(self, dt: float, cap: np.ndarray, geom):
        """Pore-pressure and gravity driven expulsion of stored pore liquid.

        Expulsion demand accumulates in a per-vertex buffer; once a buffer
        reaches one liquid sample, exactly that mass and its momentum leave the
        pore and enter the free-water phase, so mass and momentum move
        atomically between the two domains.
        """
        cen = geom[0]
        sat = self._saturation(cap)
        p_ex = self._pore_pressure(sat)
        drive = p_ex.copy()
        below = self.x_s[:, 2] < np.repeat(cen[:, 2], N_COR)
        drive[below] += np.maximum(sat[below] - S_DRIP, 0.0) * 400.0
        kr = np.clip(sat, 0.0, 1.5) ** 3
        demand = np.maximum(G_SURF * kr * drive * dt, 0.0)
        u_out = np.minimum(np.sqrt(2.0 * np.maximum(drive, 0.0) / RHO_W), 2.5)
        nrm = self.x_s - np.repeat(cen, N_COR, axis=0)
        nrm[:, 0] = 0.0
        nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        v_out = self.v_s + u_out[:, None] * nrm
        self.pend_m += demand
        self.pend_p += demand[:, None] * v_out
        cap_pend = 3.0 * self.m_p
        over = self.pend_m > cap_pend
        if np.any(over):
            scale = np.ones(N_NODE)
            scale[over] = cap_pend / self.pend_m[over]
            self.pend_m *= scale
            self.pend_p *= scale[:, None]

        ready = np.nonzero((self.pend_m >= self.m_p) & (self.pore >= self.m_p))[0]
        if not len(ready) or len(self.x_w) >= N_WATER_MAX:
            return
        v_spawn = self.pend_p[ready] / self.pend_m[ready][:, None]
        mom = self._node_mass()[ready][:, None] * self.v_s[ready]
        self.pore[ready] -= self.m_p
        self.pend_m[ready] -= self.m_p
        self.pend_p[ready] -= self.m_p * v_spawn
        carried = self.m_p * v_spawn
        self.v_s[ready] = (mom - carried) / np.maximum(
            self._node_mass()[ready], 1e-9
        )[:, None]
        self.out_mom[0] += carried.sum(axis=0)
        self.in_mom[1] += carried.sum(axis=0)
        x_spawn = self.x_s[ready] + 0.004 * nrm[ready]
        self.x_w = np.vstack((self.x_w, x_spawn))
        self.v_w = np.vstack((self.v_w, v_spawn))
        self.n_exp += len(ready)
        self.tree_age = 10 ** 9

    def _emit(self, dt: float, t: float):
        """Faucet inlet: prescribed mass flux entering the free-water domain."""
        if not (self.src_t0 <= t < self.src_t1):
            return
        self.emit_debt += self.mdot * dt
        n_new = int(self.emit_debt // self.m_p)
        room = N_WATER_MAX - len(self.x_w)
        n_new = int(min(n_new, max(room, 0)))
        if n_new <= 0:
            return
        self.emit_debt -= n_new * self.m_p
        rad = self.src_radius * np.sqrt(self.rng.random(n_new))
        ang = 2.0 * np.pi * self.rng.random(n_new)
        x_new = np.tile(self.src_center, (n_new, 1))
        x_new[:, 0] += rad * np.cos(ang)
        x_new[:, 1] += rad * np.sin(ang)
        x_new[:, 2] -= 0.055 * self.rng.random(n_new)
        v_new = np.tile(self.src_speed * self.src_dir, (n_new, 1))
        self.x_w = np.vstack((self.x_w, x_new))
        self.v_w = np.vstack((self.v_w, v_new))
        mom = (self.m_p * v_new).sum(axis=0)
        self.in_mom[1] += mom
        self.in_mom[2] += mom
        self.src_mass[1] += n_new * self.m_p
        self.tree_age = 10 ** 9

    def _drain(self):
        """Open drain / outflow removal of free-water samples."""
        if not len(self.x_w):
            return
        gone = self.x_w[:, 2] < Z_DRAIN
        gone |= np.any(self.x_w < BOUND_MIN, axis=1)
        gone |= np.any(self.x_w > BOUND_MAX, axis=1)
        if not np.any(gone):
            return
        mom = (self.m_p * self.v_w[gone]).sum(axis=0)
        self.out_mom[1] += mom
        self.out_mom[2] += mom
        self.src_mass[1] -= self.m_p * int(np.count_nonzero(gone))
        keep = ~gone
        self.x_w = self.x_w[keep]
        self.v_w = self.v_w[keep]
        self.n_drain += int(np.count_nonzero(gone))
        self.tree_age = 10 ** 9

    # -------------------------------------------------------- solid substep
    def _solid_step(self, dt: float, f_couple: np.ndarray, t_end: float, geom):
        """One semi-implicit Euler substep of the poro-elastic sponge lattice."""
        x, v = self.x_s, self.v_s
        mass = self._node_mass()

        # structural / shear / bending springs with Rayleigh-type damping
        d = x[self.e_j] - x[self.e_i]
        length = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        u = d / length[:, None]
        dv = np.sum((v[self.e_j] - v[self.e_i]) * u, axis=1)
        mag = self.e_k * (length - self.e_l0) + self.e_c * dv
        f_pair = mag[:, None] * u
        f_el = np.zeros((N_NODE, 3))
        np.add.at(f_el, self.e_i, f_pair)
        np.add.at(f_el, self.e_j, -f_pair)

        # pore pressure pushes the solid matrix outwards (internal pair)
        cen, ex, _, _, hx, hy, hz, vol = geom
        cap = self._node_capacity(vol)
        p_node = self._pore_pressure(self._saturation(cap))
        p_sec = p_node.reshape(N_SEC, N_COR).mean(axis=1)
        rad = x - np.repeat(cen, N_COR, axis=0)
        rad -= np.sum(rad * np.repeat(ex, N_COR, axis=0), axis=1)[:, None] * np.repeat(ex, N_COR, axis=0)
        rad = rad / np.maximum(np.linalg.norm(rad, axis=1, keepdims=True), 1e-9)
        rad = rad.reshape(N_SEC, N_COR, 3)
        rad = rad - rad.mean(axis=1, keepdims=True)
        area = (2.0 * hx * (hy + hz)).reshape(N_SEC, 1, 1)
        f_pore = (AREA_FAC * p_sec[:, None, None] * area * rad).reshape(N_NODE, 3)
        self.inter[2] = f_pore.sum(axis=0)
        self.inter[3] = -self.inter[2]
        self.inter[4] = f_el.sum(axis=0)
        self.inter[5] = -self.inter[4]

        f_grav = mass[:, None] * self.gravity[None, :]
        f_drag = -C_AIR_SOLID * v

        # sink floor: penalty normal contact with Coulomb-limited friction
        f_wall = np.zeros((N_NODE, 3))
        pen = Z_FLOOR - x[:, 2]
        act = pen > 0.0
        if np.any(act):
            fn = np.maximum(K_FLOOR * pen[act] - C_FLOOR * v[act, 2], 0.0)
            f_wall[act, 2] += fn
            v_t = v[act, :2]
            sp = np.maximum(np.linalg.norm(v_t, axis=1), 1e-9)
            f_t = np.minimum(MU_FLOOR * fn, C_FLOOR * sp)
            f_wall[act, 0] -= f_t * v_t[:, 0] / sp
            f_wall[act, 1] -= f_t * v_t[:, 1] / sp

        f_tot = f_el + f_pore + f_grav + f_drag + f_wall + f_couple
        v_new = v + dt * f_tot / np.maximum(mass, 1e-12)[:, None]

        # velocity limiter (numerical stabilization, exported honestly)
        sp = np.linalg.norm(v_new, axis=1)
        hot = sp > V_MAX_SOLID
        f_lim = np.zeros((N_NODE, 3))
        if np.any(hot):
            v_cap = v_new[hot] * (V_MAX_SOLID / sp[hot])[:, None]
            f_lim[hot] = mass[hot][:, None] * (v_cap - v_new[hot]) / dt
            v_new[hot] = v_cap

        # prescribed end controls (lift, then opposite twist)
        f_grip = np.zeros((N_NODE, 3))
        for grip in self.grips:
            if t_end < grip["t_on"]:
                continue
            cen_g, vel_g, ang, omega, x_t, v_t = self._grip_command(grip, t_end)
            if grip["offset"] is None:
                grip["offset"] = x[grip["idx"]] - cen_g
                cen_g, vel_g, ang, omega, x_t, v_t = self._grip_command(grip, t_end)
            idx = grip["idx"]
            f_grip[idx] += mass[idx][:, None] * (v_t - v_new[idx]) / dt
            v_new[idx] = v_t

        self.v_s = v_new
        self.x_s = x + dt * v_new
        for grip in self.grips:
            if grip["offset"] is None or t_end < grip["t_on"]:
                continue
            _, _, _, _, x_t, _ = self._grip_command(grip, t_end + dt)
            self.x_s[grip["idx"]] = x_t

        # hard no-penetration clamp on the sink floor
        low = self.x_s[:, 2] < Z_FLOOR
        if np.any(low):
            f_wall[low, 2] += mass[low] * np.maximum(-self.v_s[low, 2], 0.0) / dt
            self.x_s[low, 2] = Z_FLOOR
            self.v_s[low, 2] = np.maximum(self.v_s[low, 2], 0.0)

        # ---- impulse bookkeeping (solid domain 0, combined domain 2) ----
        terms = (
            (0, f_grav.sum(axis=0)),
            (1, f_wall.sum(axis=0)),
            (2, f_grip.sum(axis=0)),
            (3, f_couple.sum(axis=0)),
            (4, (f_drag + f_lim).sum(axis=0)),
        )
        for k, tot in terms:
            self.imp[0, k] += dt * tot
            self.f_now[0, k] = tot
            if k != 3:
                self.imp[2, k] += dt * tot

    # ------------------------------------------------------------ fluid step
    def _step(self, dt: float):
        t = self.t
        geom = self._geom(self.x_s)
        cap = self._node_capacity(geom[7])
        sec, loc, mar, bw = self._classify(geom)
        take = self._absorb_mask(sec, mar, cap)
        f_cw, f_cs = self._coupling(geom, sec, mar, bw, take)
        self.inter[0] = f_cs.sum(axis=0)
        self.inter[1] = -self.inter[0]

        n_w = len(self.x_w)
        if n_w:
            f_g = np.tile(self.m_p * self.gravity, (n_w, 1))
            f_wall = self._wall_forces()
            f_sph = self._sph_forces(self.x_w, self.v_w)
            v_new = self.v_w + dt * (f_g + f_wall + f_sph + f_cw) / self.m_p
            sp = np.linalg.norm(v_new, axis=1)
            hot = sp > V_MAX_WATER
            f_lim = np.zeros((n_w, 3))
            if np.any(hot):
                v_cap = v_new[hot] * (V_MAX_WATER / sp[hot])[:, None]
                f_lim[hot] = self.m_p * (v_cap - v_new[hot]) / dt
                v_new[hot] = v_cap
            self.v_w = v_new
            self.x_w = self.x_w + dt * self.v_w
            for k, tot in ((0, f_g.sum(axis=0)), (1, f_wall.sum(axis=0)),
                           (3, f_cw.sum(axis=0)), (4, f_lim.sum(axis=0))):
                self.imp[1, k] += dt * tot
                self.f_now[1, k] = tot
                if k != 3:
                    self.imp[2, k] += dt * tot
        else:
            self.f_now[1] = 0.0

        # pore physics: uptake, Darcy redistribution, expulsion
        if np.any(take):
            self._absorb(sec, bw, take)
            keep = ~take
            self.x_w = self.x_w[keep]
            self.v_w = self.v_w[keep]
            self.tree_age = 10 ** 9
        self._pore_transport(dt, cap)
        self._expel(dt, cap, geom)
        self._emit(dt, t)
        self._drain()

        # solid substeps with the fluid coupling load held fixed
        dt_s = dt / N_SUBSTEP
        for k in range(N_SUBSTEP):
            t_sub = t + k * dt_s
            self._solid_step(dt_s, f_cs, t_sub, self._geom(self.x_s))
        self.t = t + dt

    # ---------------------------------------------------------- frame export
    def _make_frame(self) -> StateFrame:
        geom = self._geom(self.x_s)
        cap = self._node_capacity(geom[7])
        sat_node = self._saturation(cap)
        n_w = len(self.x_w)
        n = N_NODE + n_w + 2

        x_g = np.zeros((2, 3))
        v_g = np.zeros((2, 3))
        q_g = np.zeros((2, 4))
        w_g = np.zeros((2, 3))
        for k, grip in enumerate(self.grips):
            cen_g, vel_g, ang, omega = self._grip_command(grip, self.t)[:4]
            x_g[k] = cen_g
            v_g[k] = vel_g
            q_tw = np.array([np.sin(0.5 * ang), 0.0, 0.0, np.cos(0.5 * ang)])
            q_g[k] = _quat_mul(q_tw, self.q_base)
            w_g[k, 0] = omega

        position = np.vstack((self.x_s, self.x_w, x_g))
        velocity = np.vstack((self.v_s, self.v_w, v_g))
        m_quant = np.zeros((n, 2))
        m_quant[:N_NODE, 0] = self.m_dry
        m_quant[:N_NODE, 1] = self.pore
        m_quant[N_NODE:N_NODE + n_w, 1] = self.m_p
        m_quant[N_NODE + n_w:, 0] = self.m_grip
        mass = m_quant.sum(axis=1)
        body_id = np.concatenate((
            np.ones(N_NODE, dtype=np.int32),
            np.full(n_w, 2, dtype=np.int32),
            np.array([g["body_id"] for g in self.grips], dtype=np.int32),
        ))
        orientation = np.zeros((n, 4))
        orientation[:, 3] = 1.0
        orientation[N_NODE + n_w:] = q_g
        angular = np.zeros((n, 3))
        angular[N_NODE + n_w:] = w_g

        saturation = np.concatenate((
            np.clip(sat_node, 0.0, 1.0), np.ones(n_w), np.zeros(2)
        ))
        porosity = np.concatenate((
            np.full(N_NODE, self.porosity), np.zeros(n_w + 2)
        ))
        phase = np.concatenate((np.zeros(N_NODE), np.ones(n_w), np.zeros(2)))

        weights = np.zeros((3, n))
        weights[0, :N_NODE] = 1.0
        weights[1, N_NODE:N_NODE + n_w] = 1.0
        weights[2, :N_NODE + n_w] = 1.0
        f_ext = self.f_now.copy()
        f_ext[2] = f_ext[0] + f_ext[1]
        f_ext[2, 3] = 0.0

        frame = StateFrame(
            time=float(self.t),
            position=position,
            velocity=velocity,
            orientation=orientation,
            angular_velocity=angular,
            mass=mass,
            body_id=body_id,
            attributes={
                "saturation": saturation,
                "porosity": porosity,
                "phase": phase,
                "mass_quantity": m_quant,
            },
            fields={
                "budget_domain_weights": weights,
                "budget_external_force": f_ext,
                "budget_cumulative_external_impulse": self.imp.sum(axis=1),
                "budget_cumulative_inflow_momentum": self.in_mom.copy(),
                "budget_cumulative_outflow_momentum": self.out_mom.copy(),
                "cumulative_mass_source": self.src_mass.copy(),
                "interaction_force": self.inter.copy(),
                "gravity_acceleration": self.gravity.reshape(1, 3),
            },
            scalars={
                "reference_momentum": float(self.p_ref),
                "pore_fluid_density": RHO_W,
                "gravity_z": float(self.gravity[2]),
                "pore_capacity_reference": float(self.cap_ref),
            },
        )
        frame.validate()
        return frame

    # --------------------------------------------------------------- stepping
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target <= 0.0:
            return self.frame0
        n_step = int(round((target - self.t) / DT_FLUID))
        for _ in range(max(n_step, 0)):
            self._step(DT_FLUID)
        self.t = target
        return self._make_frame()

    def diagnostics(self) -> dict[str, object]:
        geom = self._geom(self.x_s)
        cap = self._node_capacity(geom[7])
        sat = self._saturation(cap)
        return {
            "method": "spring-lattice poroelastic sponge + WCSPH water (self)",
            "dt_fluid": DT_FLUID,
            "dt_solid": DT_FLUID / N_SUBSTEP,
            "water_samples": int(len(self.x_w)),
            "pore_mass_kg": float(self.pore.sum()),
            "pore_capacity_kg": float(cap.sum()),
            "saturation_max": float(sat.max()),
            "saturation_mean": float(sat.mean()),
            "absorbed_samples": int(self.n_abs),
            "expelled_samples": int(self.n_exp),
            "drained_samples": int(self.n_drain),
            "inlet_mass_kg": float(self.src_mass[1]),
        }
