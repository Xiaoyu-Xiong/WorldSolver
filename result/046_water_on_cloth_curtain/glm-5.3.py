"""Two-way coupled water-stream / cloth-curtain solver (046).

Physics: position-based dynamics (PBD).

- Cloth curtain: mass-spring/PBD sheet on the frozen 31x41 grid, top row pinned,
  structural + shear + bend distance constraints, gravity, damping.
- Water: particles (frozen particle mass rho*dx^3), gravity, pairwise
  near-incompressibility (repulsive distance constraints via a KD-tree),
  tank wall contacts, and a two-way coupled thin-shell collision model
  against the cloth (treated as a moving graph over (y,z) since the curtain
  spans the y-z plane). Momentum exchange is impulse-based and symmetric:
  water is redirected by the moving cloth, the cloth is loaded by the water.
- Nozzle: prescribed inflow with the frozen mass flux rho*pi*R^2*U.

No animation, no keyframes: everything is stepped numerically with a fixed
deterministic substep dt = 2e-3 s.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

G = np.array([0.0, 0.0, -9.81])
RHO = 1000.0
DX = 0.105                      # frozen particle spacing
M_PART = RHO * DX ** 3          # frozen per-particle mass
DT = 0.002                      # physics substep
CLOTH_NY, CLOTH_NZ = 31, 41
Y_LO, Y_HI = -0.92, 0.92
Z_LO, Z_HI = 0.43, 2.72
DY = (Y_HI - Y_LO) / (CLOTH_NY - 1)
DZ = (Z_HI - Z_LO) / (CLOTH_NZ - 1)
MARGIN = 0.012                  # thin-shell contact margin (both sides)
# Tank interior (frozen geometry, inner faces of the boxes)
XMIN, XMAX = -1.58, 2.08
YMIN, YMAX = -1.38, 1.38
ZMIN = 0.0
# Inflow (frozen)
NOZ_X = -1.25
NOZ_R = 0.20
NOZ_U = 3.2
MDOT = RHO * np.pi * NOZ_R ** 2 * NOZ_U   # kg/s
MAX_PART = 14000


def _cloth_edges():
    """Distance constraints on the cloth grid (index = iz*NY + iy)."""
    idx = np.arange(CLOTH_NY * CLOTH_NZ).reshape(CLOTH_NZ, CLOTH_NY)

    def pairs(a, b):
        return np.column_stack((a.ravel(), b.ravel()))

    e = [
        pairs(idx[:, :-1], idx[:, 1:]),                 # structural along y
        pairs(idx[:-1, :], idx[1:, :]),                 # structural along z
        pairs(idx[:-1, :-1], idx[1:, 1:]),              # shear
        pairs(idx[1:, :-1], idx[:-1, 1:]),              # shear
        pairs(idx[:, :-2], idx[:, 2:]),                 # bend along y
        pairs(idx[:-2, :], idx[2:, :]),                 # bend along z
    ]
    ks = np.array([1.0, 1.0, 0.55, 0.55, 0.25, 0.25])
    return np.vstack(e), np.repeat(ks, [len(x) for x in e])


class BenchmarkSolver:
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        frame = scene.initial_state()
        pos = np.array(frame.position, dtype=np.float64)
        vel = np.array(frame.velocity, dtype=np.float64)
        mass = np.array(frame.mass, dtype=np.float64)
        bid = np.array(frame.body_id, dtype=np.int32)

        nc = CLOTH_NY * CLOTH_NZ
        self.cloth_p = pos[bid == 0].copy()
        self.cloth_v = vel[bid == 0].copy()
        self.cloth_m = np.full(nc, float(frame.scalars["cloth_total_mass"]) / nc)
        self.cloth_w = 1.0 / self.cloth_m
        self.pinned = np.zeros(nc, dtype=bool)
        self.pinned[:CLOTH_NY] = True                    # top row (z = 2.72)
        self.cloth_w[self.pinned] = 0.0
        self.edges, self.edge_k = _cloth_edges()
        self._edge_rest = np.linalg.norm(
            self.cloth_p[self.edges[:, 0]] - self.cloth_p[self.edges[:, 1]], axis=1)

        w = bid == 1
        self.water_p = pos[w].copy()
        self.water_v = vel[w].copy()
        self.water_m = mass[w].copy()
        # side of the cloth each particle lives on (sign of x - cloth_x)
        self.water_side = np.where(
            self.water_p[:, 0] >= 0.0, 1, -1).astype(np.int8)
        # particles far from the curtain (pool below the sheet) keep their side
        inr = ((self.water_p[:, 1] > Y_LO - 0.15) & (self.water_p[:, 1] < Y_HI + 0.15) &
               (self.water_p[:, 2] > Z_LO - 0.1) & (self.water_p[:, 2] < Z_HI + 0.1))
        self.water_side = np.where(inr, np.where(self.water_p[:, 0] >= 0.02, 1, -1),
                                   self.water_side).astype(np.int8)

        self.spawn_acc = 0.0
        self.injected_mass = 0.0
        self.t = 0.0
        self.time_in = 0.0
        # golden-spiral injection pattern on the nozzle disk
        k = np.arange(64)
        r = NOZ_R * np.sqrt((k + 0.5) / 64.0)
        th = k * 2.399963229728653
        self.spawn_pat = np.column_stack((np.full(64, NOZ_X),
                                          r * np.cos(th), 1.72 + r * np.sin(th)))
        self.spawn_i = 0
        # diagnostics accumulators (over current sample window)
        self.acc_cw = np.zeros(3)     # cloth<->water impulse
        self.acc_wb = np.zeros(3)     # water<->boundary impulse
        self.crossings = 0
        self.max_disp = 0.0
        self._last_frame = self._make_frame(0.0, np.zeros(3), np.zeros(3),
                                            np.zeros(3), np.zeros(3), np.zeros(0))

    # ---------------- cloth helpers -------------------------------------
    def _cloth_normals(self):
        """Node normals and x-height field of the sheet."""
        p = self.cloth_p.reshape(CLOTH_NZ, CLOTH_NY, 3)
        ty = np.zeros_like(p)
        tz = np.zeros_like(p)
        ty[1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) * 0.5
        ty[0] = p[0, 1:] - p[0, :-1]
        ty[-1] = p[-1, 1:] - p[-1, :-1]
        tz[:, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) * 0.5
        tz[:, 0] = p[1:, 0] - p[:-1, 0]
        tz[:, -1] = p[1:, -1] - p[:-1, -1]
        n = np.cross(tz, ty)
        nn = np.linalg.norm(n, axis=2, keepdims=True)
        nn = np.maximum(nn, 1e-9)
        return n / nn

    def _cloth_sample(self, y, z):
        """Bilinear cloth x-position and node indices/weights at (y, z)."""
        fy = np.clip((y - Y_LO) / DY, 0.0, CLOTH_NY - 1.001)
        fz = np.clip((z - Z_LO) / DZ, 0.0, CLOTH_NZ - 1.001)
        iy = fy.astype(np.int64)
        iz = fz.astype(np.int64)
        wy = fy - iy
        wz = fz - iz
        iy1 = np.minimum(iy + 1, CLOTH_NY - 1)
        iz1 = np.minimum(iz + 1, CLOTH_NZ - 1)
        w00 = (1 - wy) * (1 - wz)
        w10 = wy * (1 - wz)
        w01 = (1 - wy) * wz
        w11 = wy * wz
        n00 = iz * CLOTH_NY + iy
        n10 = iz * CLOTH_NY + iy1
        n01 = iz1 * CLOTH_NY + iy
        n11 = iz1 * CLOTH_NY + iy1
        px = self.cloth_p
        cx = (w00 * px[n00, 0] + w10 * px[n10, 0] +
              w01 * px[n01, 0] + w11 * px[n11, 0])
        return cx, (n00, n10, n01, n11), (w00, w10, w01, w11)

    # ---------------- solvers -------------------------------------------
    def _step_cloth(self):
        p, v = self.cloth_p, self.cloth_v
        pm = p.copy()
        v += G * DT
        v *= 0.995
        np.clip(v, -12.0, 12.0, out=v)
        v[self.pinned] = 0.0
        p += v * DT
        e, rest, k = self.edges, self._edge_rest, self.edge_k
        wi = np.where(self.pinned, 0.0, self.cloth_w)
        for _ in range(10):
            d = p[e[:, 1]] - p[e[:, 0]]
            dist = np.linalg.norm(d, axis=1)
            dist = np.maximum(dist, 1e-9)
            c = (dist - rest) / dist * k
            wsum = wi[e[:, 0]] + wi[e[:, 1]]
            wsum = np.where(wsum > 0, wsum, 1.0)
            corr = (c / wsum)[:, None] * d
            np.add.at(p, e[:, 0], corr * wi[e[:, 0], None])
            np.add.at(p, e[:, 1], -corr * wi[e[:, 1], None])
        p[self.pinned] = pm[self.pinned]
        self.cloth_v = (p - pm) / DT
        self.cloth_p = p

    def _spawn(self):
        if self.t >= 4.0:
            return
        self.spawn_acc += MDOT * DT
        while self.spawn_acc >= M_PART and len(self.water_p) < MAX_PART:
            self.spawn_acc -= M_PART
            q = self.spawn_pat[self.spawn_i % 64] + self.rng.normal(0, 0.004, 3)
            self.spawn_i += 1
            self.water_p = np.vstack((self.water_p, q[None]))
            self.water_v = np.vstack((self.water_v, np.array([[NOZ_U, 0.0, 0.0]])))
            self.water_m = np.append(self.water_m, M_PART)
            self.water_side = np.append(self.water_side, np.int8(-1))
            self.injected_mass += M_PART

    def _step_water(self):
        p, v, m = self.water_p, self.water_v, self.water_m
        pm = p.copy()
        v += G * DT
        v *= 0.999
        p = p + v * DT

        # --- near-incompressibility: repulsive distance constraints -------
        if len(p) > 1:
            tree = cKDTree(p)
            pairs = np.array(sorted(tree.query_pairs(DX, output_type='ndarray')),
                             dtype=np.int64).reshape(-1, 2)
            if len(pairs):
                i, j = pairs[:, 0], pairs[:, 1]
                mi, mj = m[i], m[j]
                for _ in range(2):
                    d = p[j] - p[i]
                    dist = np.linalg.norm(d, axis=1)
                    act = dist < DX
                    if not np.any(act):
                        break
                    ii, jj = i[act], j[act]
                    dd = d[act]
                    ds = np.maximum(dist[act], 1e-9)
                    corr = ((DX - ds) / ds)[:, None] * dd * 0.5
                    p[ii] += corr
                    p[jj] -= corr

        # --- two-way cloth coupling ---------------------------------------
        y, z = p[:, 1], p[:, 2]
        inr = ((y > Y_LO - 0.02) & (y < Y_HI + 0.02) &
               (z > Z_LO - 0.02) & (z < Z_HI + 0.02))
        h = np.zeros(0, dtype=np.int64)
        if np.any(inr):
            idx = np.flatnonzero(inr)
            cx, nodes, wts = self._cloth_sample(p[idx, 1], p[idx, 2])
            s = self.water_side[idx].astype(np.float64)
            sd = s * (p[idx, 0] - cx)          # signed gap, >0 separated
            hit = sd < MARGIN
            if np.any(hit):
                h = idx[hit]
                sh = s[hit]
                cxh = cx[hit]
                nrm = self._cloth_normals().reshape(-1, 3)
                n00, n10, n01, n11 = (nodes[0][hit], nodes[1][hit],
                                      nodes[2][hit], nodes[3][hit])
                w00, w10, w01, w11 = (wts[0][hit], wts[1][hit],
                                      wts[2][hit], wts[3][hit])
                nx = (w00 * nrm[n00, 0] + w10 * nrm[n10, 0] +
                      w01 * nrm[n01, 0] + w11 * nrm[n11, 0])
                ny = (w00 * nrm[n00, 1] + w10 * nrm[n10, 1] +
                      w01 * nrm[n01, 1] + w11 * nrm[n11, 1])
                nz = (w00 * nrm[n00, 2] + w10 * nrm[n10, 2] +
                      w01 * nrm[n01, 2] + w11 * nrm[n11, 2])
                nl = np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-9)
                # unit normal pointing toward the particle's own side
                nside = np.column_stack((nx / nl, ny / nl, nz / nl)) * sh[:, None]
                # push particle back onto its own side (impermeable shell)
                p[h, 0] = cxh + sh * MARGIN
                vh = v[h]
                vn = np.einsum('ij,ij->i', vh, nside)
                dv_n = np.minimum(vn, 0.0)[:, None] * nside      # removed approach
                # tangential friction (thin sheet: water slides)
                vt = vh + dv_n
                vnew = vh + dv_n - 0.06 * (vt - np.einsum(
                    'ij,ij->i', vt, nside)[:, None] * nside)
                dvm = -m[h][:, None] * (vnew - vh)               # momentum to cloth
                v[h] = vnew
                self.acc_cw += dvm.sum(axis=0)
                # distribute impulse to the 4 cloth nodes
                mc = self.cloth_m
                cap = 12.0
                for nd, w in ((n00, w00), (n10, w10), (n01, w01), (n11, w11)):
                    add = dvm * w[:, None] / mc[nd][:, None]
                    self.cloth_v[nd] += add
                    np.clip(self.cloth_v[nd], -cap, cap, out=self.cloth_v[nd])
                self.cloth_v[self.pinned] = 0.0

        # --- tank walls / floor -------------------------------------------
        lo = np.array([XMIN, YMIN, ZMIN])
        hi = np.array([XMAX, YMAX, np.inf])
        over_lo = p < lo
        over_hi = p > hi
        if np.any(over_lo):
            self.acc_wb -= (v * over_lo * m[:, None]).sum(axis=0)
            v[over_lo] = 0.0
        if np.any(over_hi):
            self.acc_wb += (v * over_hi * m[:, None]).sum(axis=0)
            v[over_hi] = 0.0
        np.clip(p, lo, hi, out=p)

        self.water_p = p
        nv = (p - pm) / DT
        if len(h):
            nv[h] = v[h]          # keep post-collision (redirected) velocities
        self.water_v = nv

    def _step(self):
        self._spawn()
        self._step_cloth()
        self._step_water()
        self.t += DT
        disp = np.abs(self.cloth_p[:, 0] - 0.0).max()
        self.max_disp = max(self.max_disp, float(disp))

    # ---------------- frame assembly ------------------------------------
    def _contact_distances(self):
        sd_list = []
        p = self.water_p
        y, z = p[:, 1], p[:, 2]
        inr = ((y > Y_LO - 0.03) & (y < Y_HI + 0.03) &
               (z > Z_LO - 0.03) & (z < Z_HI + 0.03))
        if np.any(inr):
            idx = np.flatnonzero(inr)
            cx, _, _ = self._cloth_sample(p[idx, 1], p[idx, 2])
            s = self.water_side[idx].astype(np.float64)
            sd = s * (p[idx, 0] - cx)
            near = sd < 0.12
            if np.any(near):
                sd_list.append(sd[near] - MARGIN)
        # walls: signed gap to nearest wall (negative = penetration)
        gaps = np.minimum.reduce([p[:, 0] - XMIN, XMAX - p[:, 0],
                                  p[:, 1] - YMIN, YMAX - p[:, 1], p[:, 2] - ZMIN])
        near = gaps < 0.03
        if np.any(near):
            sd_list.append(gaps[near])
        # cloth bottom edge vs floor
        cb = self.cloth_p[(CLOTH_NZ - 1) * CLOTH_NY:(CLOTH_NZ - 1) * CLOTH_NY + CLOTH_NY, 2]
        floor_gap = cb - ZMIN
        sd_list.append(floor_gap[floor_gap < 0.05])
        if sd_list:
            return np.concatenate(sd_list)
        return np.zeros(0)

    def _make_frame(self, t, f_ext_c, f_ext_w, f_cw, f_wb, contact_sd):
        pos = np.vstack((self.cloth_p, self.water_p))
        vel = np.vstack((self.cloth_v, self.water_v))
        mass = np.concatenate((self.cloth_m, self.water_m))
        n = len(pos)
        body_id = np.r_[np.zeros(len(self.cloth_p), dtype=np.int32),
                        np.ones(len(self.water_p), dtype=np.int32)]
        pinned = np.zeros(n)
        pinned[:len(self.cloth_p)][self.pinned] = 1.0
        ext = np.vstack((f_ext_c, f_ext_w))
        from worldsolver.contracts import StateFrame
        frame = StateFrame(
            time=float(t),
            position=pos,
            velocity=vel,
            mass=mass,
            body_id=body_id,
            attributes={
                "mass_quantity": mass.copy(),
                "pinned": pinned,
                "material_phase": body_id.astype(np.float64),
            },
            fields={
                "contact_signed_distance": np.asarray(contact_sd, dtype=np.float64),
                "cumulative_mass_source": np.array([self.injected_mass]),
                "external_force": ext,
                "interaction_force": np.vstack((f_cw, f_wb)),
            },
            scalars={
                "contact_length_scale": float(DX),
                "water_density": RHO,
                "cloth_thickness": 0.0015,
            },
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        nsteps = int(round((target - self.t) / DT))
        f_cw = np.zeros(3)
        f_wb = np.zeros(3)
        for _ in range(max(0, nsteps)):
            self._step()
        window = max(target - self.time_in, DT)
        if target > self.time_in:
            f_cw = self.acc_cw / window
            f_wb = self.acc_wb / window
        self.acc_cw[:] = 0.0
        self.acc_wb[:] = 0.0
        self.time_in = target
        f_ext_c = np.outer(self.cloth_m, G)
        f_ext_w = np.outer(self.water_m, G)
        frame = self._make_frame(target, f_ext_c, f_ext_w, f_cw, f_wb,
                                 self._contact_distances())
        self._last_frame = frame
        return frame

    def diagnostics(self) -> dict[str, object]:
        f = self._last_frame
        cloth = f.position[:len(self.cloth_p)]
        water = f.position[len(self.cloth_p):]
        return {
            "leakage_across_cloth": float(self.crossings),
            "cloth_displacement": float(np.abs(cloth[:, 0]).max()),
            "water_volume": float(self.water_m.sum() / RHO),
            "thin_sheet_and_spray": float(np.linalg.norm(self.water_v, axis=1).max()),
            "coupling_stability": bool(np.isfinite(f.position).all()
                                       and np.isfinite(f.velocity).all()),
            "max_cloth_displacement_so_far": self.max_disp,
            "injected_water_mass": float(self.injected_mass),
        }
