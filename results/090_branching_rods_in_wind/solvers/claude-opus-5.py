"""Self-implemented discrete elastic-rod network solver for wind-driven branching rods.

Physical model
--------------
The tree is a network of lumped-mass rod vertices connected by edges taken from the
frozen topology asset.  Elasticity is supplied by two rotation- and translation-invariant
internal energies, so every internal force set has exactly zero net force and zero net
torque (which keeps the exported momentum budgets consistent):

* axial (stretch) springs on every rod edge,
* discrete bending / joint springs on every pair of edges meeting at a vertex group,
  penalising the deviation of the turning angle from its reference value.  Pairs of
  edges belonging to different child branches at the same junction are included, which
  couples branch positions *and* orientations (a rigid couple in the sense of
  Section 9.4 of Discrete Elastic Rods) and makes the network twist-resistant.

External loading is gravity plus a prescribed wind body force with a
height-and-area-weighted spatial profile, keyframed mean speed and per-branch gust
phase offsets.  The wind force is evaluated on the relative velocity, which supplies
aerodynamic damping.  The root is clamped (root vertex plus the first trunk vertex are
held fixed); the corresponding constraint reactions are exported as external
boundary tractions.

Time stepping is an explicit symplectic (semi-implicit) Euler scheme with a fixed
sub-step chosen from the stiffest axial mode.  For that scheme the discrete momentum
and angular-momentum balances

    p_{n+1} - p_n = dt * sum_i F_i(x_n)
    L_{n+1} - L_n = dt * sum_i x_n_i x F_i(x_n)

hold exactly, so the cumulative impulse channels are accumulated with the same
forces and positions used by the integrator.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------------
# model constants (derived from the frozen config, see module docstring)
# ----------------------------------------------------------------------------------
# Effective bending moduli.  Because I = pi*r^4/4 falls off far faster than the wind
# load (~r) between levels, a single modulus would leave the trunk rigid and the twigs
# limp.  Each level is instead sized from the cantilever relation delta = w L^4/(8 E I)
# for a target peak tip deflection (trunk ~0.8 m, level-1 ~0.5 m, twigs ~0.35 m), which
# keeps every level visibly compliant while retaining distinct per-level amplitudes.
_E_BY_LEVEL = (6.4e8, 1.1e8, 5.7e7)          # Pa, effective bending modulus per level
_K_AXIAL_BY_LEVEL = (4.0e4, 8.0e3, 1.5e3)    # N/m, axial spring stiffness per level
# Rayleigh beta values are bounded by the explicit stability limit dt < 2 m / c_eff on
# the lightest twig nodes (m ~ 1.3e-4 kg), which caps the joint value near 1e-3 s.
_AXIAL_DAMP_TIME = 1.0e-4                    # s, Rayleigh beta for axial springs
_BEND_DAMP_TIME = 3.0e-4                     # s, Rayleigh beta for joint springs
_JUNCTION_STIFFENING = 3.0                   # rigid-couple factor at branch points
_WIND_AREA_FACTOR = (1.0, 1.3, 1.6)          # twig/foliage bulking of frontal area
_WIND_CD = 1.0                               # cylinder drag coefficient
_AIR_DENSITY = 1.2                           # kg/m^3
_WIND_PROFILE_EXP = 0.30                     # height weighting exponent
_GUST_DEPTH = 0.35                           # relative gust modulation amplitude
_GRAD_EPS = 1.0e-9                           # |t2 - cos*t1| floor for the angle gradient
_DEGENERATE_LEN = 1.0e-5                     # rest length below which an edge is a pin
_SUBSTEP_DT = 1.0e-4                         # s
_REF_SPEED = 20.0                            # m/s, frozen normalisation speed


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class BenchmarkSolver:
    """Explicit elastic-rod-network solver (NumPy only, no external physics engine)."""

    # -------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        self._seed = int(seed)
        cfg = self._scene_config(scene)
        self.cfg = cfg

        frame0 = scene.initial_state()
        self.x0 = np.array(frame0.position, dtype=np.float64)
        self.n = int(self.x0.shape[0])
        self.mass = np.array(frame0.mass, dtype=np.float64).reshape(self.n)
        self.base_attributes = {
            key: np.array(val, dtype=np.float64)
            for key, val in dict(frame0.attributes).items()
        }

        topo = self._load_topology(scene, cfg)
        self.edges = np.asarray(topo["edges"], dtype=np.int64).reshape(-1, 2)
        self.levels = np.asarray(topo["vertex_level"], dtype=np.int64).reshape(self.n)
        self.branch_id = np.asarray(topo["branch_id"], dtype=np.int64).reshape(self.n)
        self.lumped_len = np.asarray(topo["lumped_length"], dtype=np.float64).reshape(self.n)

        rod = cfg["rod_network"]
        radius_by_level = np.asarray(rod["radius_by_level"], dtype=np.float64)
        self.radius = radius_by_level[self.levels]

        self.gravity = np.asarray(cfg["gravity"], dtype=np.float64).reshape(3)

        # ---- wind description ------------------------------------------------
        wind = cfg["wind"]
        wdir = np.asarray(wind["direction"], dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(wdir))
        self.wind_dir = wdir / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
        keys = sorted(wind["keyframes"], key=lambda k: float(k["time"]))
        self.wind_times = np.asarray([float(k["time"]) for k in keys], dtype=np.float64)
        self.wind_speeds = np.asarray([float(k["speed"]) for k in keys], dtype=np.float64)
        self.gust_hz = float(wind.get("gust_frequency_hz", 0.5))
        offsets = np.asarray(wind.get("branch_phase_offsets", [0.0]), dtype=np.float64)
        if offsets.size == 0:
            offsets = np.zeros(1, dtype=np.float64)
        self.node_phase = 2.0 * np.pi * offsets[self.branch_id % offsets.size]

        area_factor = np.asarray(_WIND_AREA_FACTOR, dtype=np.float64)[
            np.clip(self.levels, 0, len(_WIND_AREA_FACTOR) - 1)
        ]
        self.wind_area = 2.0 * self.radius * self.lumped_len * area_factor
        self.wind_coeff = 0.5 * _AIR_DENSITY * _WIND_CD * self.wind_area
        self.z_ref = float(max(np.max(self.x0[:, 2]), 1.0))

        # ---- structural stiffness -------------------------------------------
        self._build_axial()
        self._build_joints()

        # ---- boundary conditions --------------------------------------------
        root = int(cfg["rigid_junctions"].get("fixed_root_vertex", 0))
        pinned = [root]
        # clamp the root orientation as well by holding the first trunk vertex.
        for a, b in self.edges:
            if int(a) == root and int(b) not in pinned:
                pinned.append(int(b))
                break
            if int(b) == root and int(a) not in pinned:
                pinned.append(int(a))
                break
        self.pinned = np.asarray(sorted(set(pinned)), dtype=np.int64)
        self.free_mask = np.ones(self.n, dtype=bool)
        self.free_mask[self.pinned] = False

        # ---- state ----------------------------------------------------------
        self.x = self.x0.copy()
        v0 = np.asarray(cfg["initial_conditions"].get("initial_velocity", [0.0, 0.0, 0.0]),
                        dtype=np.float64).reshape(3)
        self.v = np.tile(v0, (self.n, 1))
        self.v[self.pinned] = 0.0
        self.t = 0.0
        self.dt = float(_SUBSTEP_DT)
        self.inv_mass = np.zeros_like(self.mass)
        good = self.mass > 0.0
        self.inv_mass[good] = 1.0 / self.mass[good]

        # ---- budget accumulators --------------------------------------------
        self.cum_impulse = np.zeros(3, dtype=np.float64)
        self.cum_ang_impulse = np.zeros(3, dtype=np.float64)
        self.n_steps = 0
        self.max_speed = 0.0

        total_mass = float(np.sum(self.mass))
        self.ref_momentum = max(total_mass * _REF_SPEED, 1.0e-6)
        self.ref_ang_momentum = max(total_mass * _REF_SPEED * self.z_ref, 1.0e-6)

        # reference tangents (for the reported orientation quaternions)
        self.ref_tangent = self._vertex_tangent(self.x0)

        # cache force decomposition for the initial frame
        grav, wind_f, fint = self._forces(self.x, self.v, self.t)
        self._last_grav = grav
        self._last_wind = wind_f
        self._last_reaction = -(grav + wind_f + fint)[self.pinned]
        self._last_wind_speed = float(self._mean_wind_speed(self.t))

    # -------------------------------------------------------- config helpers
    @staticmethod
    def _scene_config(scene) -> dict:
        cfg = getattr(scene, "config", None)
        if isinstance(cfg, dict):
            return cfg
        meta = scene.metadata()
        if isinstance(meta, dict):
            return meta
        raise RuntimeError("scene does not expose its configuration")

    def _load_topology(self, scene, cfg: dict) -> dict:
        rel = str(cfg["rod_network"]["topology_asset"])
        candidates = []
        try:
            import inspect

            candidates.append(Path(inspect.getfile(type(scene))).resolve().parent)
        except Exception:
            pass
        candidates.append(Path(__file__).resolve().parent.parent)
        candidates.append(Path.cwd())
        for base in candidates:
            path = base / rel
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"topology asset not found: {rel}")

    # ------------------------------------------------------ structure build
    def _edge_level(self, a: int, b: int) -> int:
        return int(max(self.levels[a], self.levels[b]))

    def _build_axial(self) -> None:
        a = self.edges[:, 0]
        b = self.edges[:, 1]
        rest = np.linalg.norm(self.x0[b] - self.x0[a], axis=1)
        lvl = np.maximum(self.levels[a], self.levels[b])
        k_by_level = np.asarray(_K_AXIAL_BY_LEVEL, dtype=np.float64)
        k = k_by_level[np.clip(lvl, 0, len(_K_AXIAL_BY_LEVEL) - 1)]
        self.e_a = a
        self.e_b = b
        self.e_rest = rest
        self.e_k = k
        self.e_c = _AXIAL_DAMP_TIME * k
        self.e_level = lvl

    def _build_joints(self) -> None:
        """Angle springs between every pair of edges meeting at a (contracted) vertex."""
        uf = _UnionFind(self.n)
        for idx in range(len(self.e_a)):
            if self.e_rest[idx] < _DEGENERATE_LEN:
                uf.union(int(self.e_a[idx]), int(self.e_b[idx]))

        # directed non-degenerate edges leaving each contracted group
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for idx in range(len(self.e_a)):
            if self.e_rest[idx] < _DEGENERATE_LEN:
                continue
            a, b = int(self.e_a[idx]), int(self.e_b[idx])
            ga, gb = uf.find(a), uf.find(b)
            groups.setdefault(ga, []).append((a, b, idx))
            if gb != ga:
                groups.setdefault(gb, []).append((b, a, idx))

        E_by_level = np.asarray(_E_BY_LEVEL, dtype=np.float64)
        a1l: list[int] = []
        b1l: list[int] = []
        a2l: list[int] = []
        b2l: list[int] = []
        kbl: list[float] = []
        for _grp, out_edges in sorted(groups.items()):
            deg = len(out_edges)
            if deg < 2:
                continue
            stiffen = _JUNCTION_STIFFENING if deg >= 3 else 1.0
            for p in range(deg):
                for q in range(p + 1, deg):
                    ap, bp, ip = out_edges[p]
                    aq, bq, iq = out_edges[q]
                    lp = int(self.e_level[ip])
                    lq = int(self.e_level[iq])
                    rp = float(self.radius[bp])
                    rq = float(self.radius[bq])
                    ip_mom = 0.25 * np.pi * rp ** 4
                    iq_mom = 0.25 * np.pi * rq ** 4
                    ei = min(
                        E_by_level[min(lp, len(E_by_level) - 1)] * ip_mom,
                        E_by_level[min(lq, len(E_by_level) - 1)] * iq_mom,
                    )
                    lbar = 0.5 * (float(self.e_rest[ip]) + float(self.e_rest[iq]))
                    lbar = max(lbar, 1.0e-3)
                    a1l.append(ap)
                    b1l.append(bp)
                    a2l.append(aq)
                    b2l.append(bq)
                    kbl.append(stiffen * ei / lbar)

        self.j_a1 = np.asarray(a1l, dtype=np.int64)
        self.j_b1 = np.asarray(b1l, dtype=np.int64)
        self.j_a2 = np.asarray(a2l, dtype=np.int64)
        self.j_b2 = np.asarray(b2l, dtype=np.int64)
        self.j_kb = np.asarray(kbl, dtype=np.float64)
        self.j_cb = _BEND_DAMP_TIME * self.j_kb

        if self.j_a1.size:
            d1 = self.x0[self.j_b1] - self.x0[self.j_a1]
            d2 = self.x0[self.j_b2] - self.x0[self.j_a2]
            t1 = d1 / np.maximum(np.linalg.norm(d1, axis=1, keepdims=True), 1e-15)
            t2 = d2 / np.maximum(np.linalg.norm(d2, axis=1, keepdims=True), 1e-15)
            cos0 = np.clip(np.sum(t1 * t2, axis=1), -1.0, 1.0)
            self.j_theta0 = np.arccos(cos0)
        else:
            self.j_theta0 = np.zeros(0, dtype=np.float64)

        # scatter target indices for one fused accumulation
        self.int_index = np.concatenate(
            [self.e_a, self.e_b, self.j_a1, self.j_b1, self.j_a2, self.j_b2]
        )

    # ------------------------------------------------------------- forces
    def _mean_wind_speed(self, t: float) -> float:
        return float(np.interp(float(t), self.wind_times, self.wind_speeds))

    def _wind_force(self, x: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
        speed = self._mean_wind_speed(t)
        gust = 1.0 + _GUST_DEPTH * np.sin(2.0 * np.pi * self.gust_hz * float(t) + self.node_phase)
        height = np.clip(x[:, 2] / self.z_ref, 0.02, None) ** _WIND_PROFILE_EXP
        u = self.wind_dir[None, :] * (speed * gust * height)[:, None]
        rel = u - v
        mag = np.linalg.norm(rel, axis=1)
        return (self.wind_coeff * mag)[:, None] * rel

    def _internal_forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        # ---- axial springs (forces along the edge: torque free) ----
        d = x[self.e_b] - x[self.e_a]
        length = np.linalg.norm(d, axis=1)
        safe = np.maximum(length, 1e-12)
        tang = d / safe[:, None]
        dv = v[self.e_b] - v[self.e_a]
        stretch = length - self.e_rest
        rate = np.sum(dv * tang, axis=1)
        scal = -(self.e_k * stretch + self.e_c * rate)
        # for zero-rest-length pin edges the linear form -k*d is used directly
        pin = self.e_rest < _DEGENERATE_LEN
        f_edge = scal[:, None] * tang
        if np.any(pin):
            f_pin = -(self.e_k[pin, None] * d[pin] + self.e_c[pin, None] * dv[pin])
            f_edge[pin] = f_pin
        f_on_b = f_edge
        f_on_a = -f_edge

        # ---- joint (bending / rigid-couple) springs ----
        if self.j_a1.size:
            d1 = x[self.j_b1] - x[self.j_a1]
            d2 = x[self.j_b2] - x[self.j_a2]
            l1 = np.maximum(np.linalg.norm(d1, axis=1), 1e-12)
            l2 = np.maximum(np.linalg.norm(d2, axis=1), 1e-12)
            t1 = d1 / l1[:, None]
            t2 = d2 / l2[:, None]
            cos = np.clip(np.sum(t1 * t2, axis=1), -1.0, 1.0)
            theta = np.arccos(cos)
            # d(theta)/d(d1) = -(t2 - cos*t1) / (l1 * sin), and |t2 - cos*t1| = sin,
            # so normalising the numerator cancels the 1/sin analytically.  The
            # direction is undefined only for exactly (anti)parallel edges, where the
            # restoring magnitude is likewise degenerate, so a zero gradient is used.
            w1 = t2 - cos[:, None] * t1
            w2 = t1 - cos[:, None] * t2
            n1 = np.linalg.norm(w1, axis=1)
            n2 = np.linalg.norm(w2, axis=1)
            h1 = -np.where((n1 > _GRAD_EPS)[:, None], w1 / np.maximum(n1, _GRAD_EPS)[:, None],
                           0.0) / l1[:, None]
            h2 = -np.where((n2 > _GRAD_EPS)[:, None], w2 / np.maximum(n2, _GRAD_EPS)[:, None],
                           0.0) / l2[:, None]
            dv1 = v[self.j_b1] - v[self.j_a1]
            dv2 = v[self.j_b2] - v[self.j_a2]
            theta_dot = np.sum(h1 * dv1, axis=1) + np.sum(h2 * dv2, axis=1)
            fscal = self.j_kb * (theta - self.j_theta0) + self.j_cb * theta_dot
            g1 = fscal[:, None] * h1
            g2 = fscal[:, None] * h2
            j_fa1 = g1
            j_fb1 = -g1
            j_fa2 = g2
            j_fb2 = -g2
        else:
            zero = np.zeros((0, 3), dtype=np.float64)
            j_fa1 = j_fb1 = j_fa2 = j_fb2 = zero

        contrib = np.concatenate([f_on_a, f_on_b, j_fa1, j_fb1, j_fa2, j_fb2], axis=0)
        out = np.zeros((self.n, 3), dtype=np.float64)
        np.add.at(out, self.int_index, contrib)
        return out

    def _forces(self, x: np.ndarray, v: np.ndarray, t: float):
        grav = self.mass[:, None] * self.gravity[None, :]
        wind = self._wind_force(x, v, t)
        fint = self._internal_forces(x, v)
        return grav, wind, fint

    # -------------------------------------------------------------- stepping
    def _step(self, h: float) -> None:
        x, v = self.x, self.v
        grav, wind, fint = self._forces(x, v, self.t)
        total = grav + wind + fint

        pin = self.pinned
        reaction = -total[pin]
        total[pin] = 0.0

        # exact discrete budgets for symplectic Euler, using start-of-step positions
        f_sum = grav.sum(axis=0) + wind.sum(axis=0) + reaction.sum(axis=0)
        tau = np.cross(x, grav + wind).sum(axis=0) + np.cross(x[pin], reaction).sum(axis=0)
        self.cum_impulse += f_sum * h
        self.cum_ang_impulse += tau * h

        v += (total * self.inv_mass[:, None]) * h
        v[pin] = 0.0
        x += v * h
        x[pin] = self.x0[pin]

        self.t += h
        self.n_steps += 1
        self._last_grav = grav
        self._last_wind = wind
        self._last_reaction = reaction
        self._last_wind_speed = self._mean_wind_speed(self.t)
        sp = float(np.max(np.linalg.norm(v, axis=1))) if self.n else 0.0
        if sp > self.max_speed:
            self.max_speed = sp

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.t + 1e-15:
            span = target - self.t
            nsub = int(np.ceil(span / self.dt))
            nsub = max(nsub, 1)
            h = span / nsub
            for _ in range(nsub):
                self._step(h)
            self.t = target
        return self._make_frame(target)

    # ---------------------------------------------------------------- output
    def _vertex_tangent(self, x: np.ndarray) -> np.ndarray:
        acc = np.zeros((self.n, 3), dtype=np.float64)
        d = x[self.e_b] - x[self.e_a]
        nrm = np.linalg.norm(d, axis=1)
        keep = nrm > 1e-9
        if np.any(keep):
            unit = d[keep] / nrm[keep][:, None]
            np.add.at(acc, self.e_a[keep], unit)
            np.add.at(acc, self.e_b[keep], unit)
        mag = np.linalg.norm(acc, axis=1)
        fallback = mag < 1e-9
        acc[fallback] = np.array([0.0, 0.0, 1.0])
        mag = np.maximum(np.linalg.norm(acc, axis=1), 1e-12)
        return acc / mag[:, None]

    def _vertex_angular_velocity(self) -> np.ndarray:
        acc = np.zeros((self.n, 3), dtype=np.float64)
        cnt = np.zeros(self.n, dtype=np.float64)
        d = self.x[self.e_b] - self.x[self.e_a]
        dv = self.v[self.e_b] - self.v[self.e_a]
        sq = np.sum(d * d, axis=1)
        keep = sq > 1e-12
        if np.any(keep):
            omega = np.cross(d[keep], dv[keep]) / sq[keep][:, None]
            np.add.at(acc, self.e_a[keep], omega)
            np.add.at(acc, self.e_b[keep], omega)
            np.add.at(cnt, self.e_a[keep], 1.0)
            np.add.at(cnt, self.e_b[keep], 1.0)
        cnt = np.maximum(cnt, 1.0)
        return acc / cnt[:, None]

    def _orientation(self) -> np.ndarray:
        t_cur = self._vertex_tangent(self.x)
        t_ref = self.ref_tangent
        axis = np.cross(t_ref, t_cur)
        dot = np.clip(np.sum(t_ref * t_cur, axis=1), -1.0, 1.0)
        quat = np.zeros((self.n, 4), dtype=np.float64)
        s = np.linalg.norm(axis, axis=1)
        near = s < 1e-9
        # general case: shortest-arc quaternion
        angle = np.arctan2(s, dot)
        half = 0.5 * angle
        unit_axis = np.zeros_like(axis)
        ok = ~near
        if np.any(ok):
            unit_axis[ok] = axis[ok] / s[ok][:, None]
        quat[:, :3] = unit_axis * np.sin(half)[:, None]
        quat[:, 3] = np.cos(half)
        # anti-parallel / parallel degenerate cases
        if np.any(near):
            flip = near & (dot < 0.0)
            same = near & (dot >= 0.0)
            quat[same] = np.array([0.0, 0.0, 0.0, 1.0])
            if np.any(flip):
                helper = np.tile(np.array([1.0, 0.0, 0.0]), (int(np.sum(flip)), 1))
                tf = t_ref[flip]
                alt = np.cross(tf, helper)
                bad = np.linalg.norm(alt, axis=1) < 1e-6
                if np.any(bad):
                    alt[bad] = np.cross(tf[bad], np.array([0.0, 1.0, 0.0]))
                alt = alt / np.maximum(np.linalg.norm(alt, axis=1, keepdims=True), 1e-12)
                q = np.zeros((alt.shape[0], 4), dtype=np.float64)
                q[:, :3] = alt
                q[:, 3] = 0.0
                quat[flip] = q
        nrm = np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
        return quat / nrm

    def _make_frame(self, t: float) -> StateFrame:
        n = self.n
        x = np.ascontiguousarray(self.x, dtype=np.float64)
        v = np.ascontiguousarray(self.v, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        omega = np.nan_to_num(self._vertex_angular_velocity(), nan=0.0,
                              posinf=0.0, neginf=0.0)
        quat = np.nan_to_num(self._orientation(), nan=0.0, posinf=0.0, neginf=0.0)

        grav = np.nan_to_num(self._last_grav, nan=0.0, posinf=0.0, neginf=0.0)
        wind = np.nan_to_num(self._last_wind, nan=0.0, posinf=0.0, neginf=0.0)
        reaction = np.nan_to_num(self._last_reaction, nan=0.0, posinf=0.0, neginf=0.0)

        npin = int(self.pinned.size)
        k_terms = 2 * n + npin
        forces = np.zeros((1, k_terms, 3), dtype=np.float64)
        positions = np.zeros((1, k_terms, 3), dtype=np.float64)
        forces[0, :n] = grav
        positions[0, :n] = x
        forces[0, n:2 * n] = wind
        positions[0, n:2 * n] = x
        forces[0, 2 * n:] = reaction
        positions[0, 2 * n:] = x[self.pinned]

        attributes = {key: val.copy() for key, val in self.base_attributes.items()}
        attributes["inertia_tensor"] = np.zeros((n, 3, 3), dtype=np.float64)
        attributes["mass_quantity"] = self.mass.copy()
        attributes["rod_level"] = self.levels.astype(np.float64)
        attributes["branch_index"] = self.branch_id.astype(np.float64)

        fields = {
            "budget_domain_weights": np.ones((1, n), dtype=np.float64),
            "budget_external_force": forces,
            "budget_force_position": positions,
            "budget_external_couple": np.zeros((1, k_terms, 3), dtype=np.float64),
            "budget_cumulative_external_impulse": self.cum_impulse.reshape(1, 3).copy(),
            "budget_cumulative_external_angular_impulse":
                self.cum_ang_impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3), dtype=np.float64),
            "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3), dtype=np.float64),
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "external_force": np.ascontiguousarray(grav + wind),
        }

        scalars = {
            "wind_speed": float(self._last_wind_speed),
            "gravity_magnitude": float(np.linalg.norm(self.gravity)),
            "reference_momentum": float(self.ref_momentum),
            "reference_angular_momentum": float(self.ref_ang_momentum),
        }

        return StateFrame(
            time=float(max(t, 0.0)),
            position=x.copy(),
            velocity=v.copy(),
            orientation=quat,
            angular_velocity=omega,
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    # ---------------------------------------------------------- diagnostics
    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit symplectic Euler on a discrete elastic rod network",
            "substep_dt": float(self.dt),
            "substeps_taken": int(self.n_steps),
            "vertices": int(self.n),
            "edges": int(len(self.e_a)),
            "joint_springs": int(self.j_a1.size),
            "pinned_vertices": [int(i) for i in self.pinned],
            "total_mass": float(np.sum(self.mass)),
            "max_speed": float(self.max_speed),
            "seed": int(self._seed),
        }
