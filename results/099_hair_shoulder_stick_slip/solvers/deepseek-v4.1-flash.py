"""Elastic hair-strand solver with dry Coulomb stick-slip friction.

Physical model
--------------
Every hair strand is a discrete elastic rod: axial (stretch) springs along the
node chain plus a discrete bending resistance acting on consecutive node
triples, both measured relative to the frozen initial (wavy) reference
configuration.  Roots are kinematic and follow the prescribed head motion.

The state is advanced with a real fixed-step semi-implicit (symplectic) Euler
integration of Newton's equations under gravity.  Contacts against the frozen
body primitives (scaled sphere / capsules) and against neighbouring strands are
resolved with a mass-weighted non-penetration projection plus Coulomb friction
impulses using static == kinetic coefficient 0.3.  This gives genuine stick
(relative tangential velocity cancelled while the required impulse stays below
mu*N) followed by slip when the tangential demand exceeds the friction cone.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

try:  # optional acceleration structure, pure-numpy fallback below
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

R_HAIR = 0.0042
GRAVITY = np.array([0.0, 0.0, -9.81])
MU_S = 0.3
MU_K = 0.3
STICK_SPEED_TOLERANCE = 1e-3
REFERENCE_VELOCITY = 0.15


def _norm(a: np.ndarray) -> np.ndarray:
    return np.sqrt(np.einsum("...i,...i->...", a, a))


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


class BenchmarkSolver:
    """Runnable physics solver for the hair/shoulder stick-slip scene."""

    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        frame0 = scene.initial_state()
        self.N = int(len(frame0.position))
        self.x = np.array(frame0.position, dtype=float)
        self.v = np.array(frame0.velocity, dtype=float)
        self.mass = np.array(frame0.mass, dtype=float)
        attrs = frame0.attributes
        self.strand_id = np.array(attrs["strand_id"], dtype=int)
        self.node_index = np.array(attrs["node_index"], dtype=int)
        self.X0 = self.x.copy()

        nstrands = int(self.strand_id.max()) + 1
        npl = int(self.node_index.max()) + 1
        self.nstrands = nstrands
        self.nodes_per_strand = npl

        # chain edges: node id = strand_id * npl + node_index
        base_e = np.repeat(np.arange(nstrands, dtype=int) * npl, npl - 1)
        off_e = np.tile(np.arange(npl - 1, dtype=int), nstrands)
        self.ea = base_e + off_e
        self.eb = self.ea + 1
        # bending triples on consecutive nodes of each strand
        base_t = np.repeat(np.arange(nstrands, dtype=int) * npl, npl - 2)
        off_t = np.tile(np.arange(npl - 2, dtype=int), nstrands)
        self.ta = base_t + off_t
        self.tb = self.ta + 1
        self.tc = self.ta + 2

        d = self.x[self.eb] - self.x[self.ea]
        self.rest_len = np.maximum(_norm(d), 1e-6)
        self.rest_curv = self.x[self.ta] - 2.0 * self.x[self.tb] + self.x[self.tc]

        self.is_root = self.node_index == 0
        self.free = ~self.is_root
        self.inv_m = np.where(self.mass > 0.0, 1.0 / np.maximum(self.mass, 1e-12), 0.0)

        # integration + material parameters
        self.dt = 5.0e-4
        self.k_s = 250.0          # axial stiffness [N/m]
        self.c_s = 0.03           # axial viscous damping [N s/m]
        self.k_b = 0.22           # discrete bending stiffness [N/m]
        self.body_damping = 0.4   # light global velocity damping [1/s]

        # head motion (roots follow the frozen prescribed translation)
        motion = {}
        cfg = getattr(scene, "config", {}) or {}
        motion = cfg.get("head_motion", {}) or {}
        axis = np.asarray(motion.get("axis", [1.0, 0.0, 0.0]), dtype=float)
        na = float(np.linalg.norm(axis))
        self.axis = axis / na if na > 1e-12 else np.array([1.0, 0.0, 0.0])
        self.disp = float(motion.get("displacement", 0.18))
        self.t_start = float(motion.get("start_time", 0.45))
        self.t_stop = float(motion.get("stop_time", 2.35))

        self.colliders = self._build_colliders()

        # diagnostics / bookkeeping
        self.step_count = 0
        self.time = 0.0
        self.cum_external_impulse = np.zeros(3)
        self.external_force = np.zeros((4, 3))
        self._pair_node = np.empty(0, dtype=int)
        self._pair_seg = np.empty(0, dtype=int)
        self._pair_age = 999
        self.report = self._empty_report()
        self.contact_count_max = 0

    # ------------------------------------------------------------------ setup
    def _build_colliders(self):
        specs = [
            ("capsule", [0.0, 0.0, 0.62], [0.0, 0.0, 0.0, 1.0], [1.0, 0.72, 1.0], 0.34, 0.62),
            ("capsule", [0.0, 0.0, 1.02], [0.0, 0.7071068, 0.0, 0.7071068], [1.0, 1.0, 1.0], 0.145, 0.92),
            ("capsule", [0.0, 0.0, 1.28], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0], 0.115, 0.28),
            ("sphere", [0.0, 0.0, 1.62], [0.0, 0.0, 0.0, 1.0], [0.88, 0.95, 1.12], 0.265, 0.0),
        ]
        colliders = []
        for kind, t, q, s, r, h in specs:
            colliders.append(
                {
                    "kind": kind,
                    "T": np.asarray(t, dtype=float),
                    "R": _quat_matrix(q),
                    "S": np.asarray(s, dtype=float),
                    "r": float(r),
                    "h": float(h),
                }
            )
        return colliders

    @staticmethod
    def _empty_report():
        return {
            "normal": np.zeros((0, 3)),
            "normal_force": np.zeros((0,)),
            "tangential_force": np.zeros((0, 3)),
            "relative_velocity": np.zeros((0, 3)),
        }

    # ------------------------------------------------------------- kinematics
    def _motion_state(self, t: float):
        t0, t1 = self.t_start, self.t_stop
        if t <= t0:
            return 0.0, 0.0
        if t >= t1:
            return self.disp, 0.0
        span = max(t1 - t0, 1e-9)
        u = (t - t0) / span
        s = u * u * (3.0 - 2.0 * u)
        ds = 6.0 * u * (1.0 - u) / span
        return self.disp * s, self.disp * ds

    def _set_roots(self, t: float) -> None:
        off, vel = self._motion_state(t)
        self.x[self.is_root] = self.X0[self.is_root] + self.axis * off
        self.v[self.is_root] = self.axis * vel

    # -------------------------------------------------------------- mechanics
    def _elastic_forces(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        F = np.zeros_like(x)
        d = x[self.eb] - x[self.ea]
        L = _norm(d)
        Ls = np.maximum(L, 1e-9)
        u = d / Ls[:, None]
        ext = L - self.rest_len
        fd = self.c_s * np.einsum("ij,ij->i", v[self.eb] - v[self.ea], u)
        fe = (self.k_s * ext + fd)[:, None] * u
        np.add.at(F, self.ea, -fe)
        np.add.at(F, self.eb, fe)

        e = x[self.ta] - 2.0 * x[self.tb] + x[self.tc] - self.rest_curv
        fb = self.k_b * e
        np.add.at(F, self.ta, -fb)
        np.add.at(F, self.tb, 2.0 * fb)
        np.add.at(F, self.tc, -fb)
        return F

    def _sdf(self, coll: dict, P: np.ndarray):
        q = (P - coll["T"]) @ coll["R"] / coll["S"]
        if coll["kind"] == "sphere":
            dq = _norm(q)
            safe = np.maximum(dq, 1e-12)
            gl = q / safe[:, None]
            gl = np.where(dq[:, None] > 1e-12, gl, np.array([0.0, 0.0, 1.0]))
            dloc = dq - coll["r"]
        else:
            half = 0.5 * coll["h"]
            zc = np.clip(q[:, 2], -half, half)
            diff = q - np.stack([np.zeros_like(zc), np.zeros_like(zc), zc], axis=1)
            dq = _norm(diff)
            safe = np.maximum(dq, 1e-12)
            gl = diff / safe[:, None]
            gl = np.where(dq[:, None] > 1e-12, gl, np.array([0.0, 0.0, 1.0]))
            dloc = dq - coll["r"]
        gw = (gl / coll["S"]) @ coll["R"].T
        gn = np.maximum(_norm(gw), 1e-12)
        dist = dloc / gn
        normal = gw / gn[:, None]
        return dist, normal

    def _body_candidates(self):
        best_d = np.full(self.N, np.inf)
        best_n = np.zeros((self.N, 3))
        for coll in self.colliders:
            d, nrm = self._sdf(coll, self.x)
            better = d < best_d
            best_d[better] = d[better]
            best_n[better] = nrm[better]
        mask = (best_d < 0.0) & self.free
        idx = np.flatnonzero(mask)
        return idx, best_d[idx], best_n[idx]

    # ---------------------------------------------------------- contact solve
    def _body_pass(self, dt: float, first: bool):
        idx, gap, nrm = self._body_candidates()
        if first:
            self._body_report = self._empty_report()
            self._body_force_n = np.zeros(3)
            self._body_force_t = np.zeros(3)
        if idx.size == 0:
            return
        vn = np.einsum("ij,ij->i", self.v[idx], nrm)
        jn = self.mass[idx] * np.maximum(0.0, -vn)
        vt = self.v[idx] - vn[:, None] * nrm
        sp = _norm(vt)
        jt = np.minimum(self.mass[idx] * sp, MU_K * jn)
        jtv = -vt / np.maximum(sp, 1e-12)[:, None] * jt[:, None]
        imp = jn[:, None] * nrm + jtv
        np.add.at(self.v, idx, imp * self.inv_m[idx][:, None])
        sep = np.clip(-gap, 0.0, 0.02)
        np.add.at(self.x, idx, sep[:, None] * nrm)
        if first:
            # self.v[idx] already includes imp/m, so subtract it to recover the
            # pre-impulse relative velocity (the body is kinematic/static).
            self._body_report = {
                "normal": nrm.copy(),
                "normal_force": jn / dt,
                "tangential_force": jtv / dt,
                "relative_velocity": self.v[idx] - imp * self.inv_m[idx][:, None],
            }
            self._body_force_n = (jn[:, None] * nrm).sum(0) / dt
            self._body_force_t = jtv.sum(0) / dt

    def _refresh_hair_pairs(self):
        if cKDTree is None or self._pair_age < 3:
            self._pair_age += 1
            return
        self._pair_age = 0
        if self.ea.size == 0:
            return
        mid = 0.5 * (self.x[self.ea] + self.x[self.eb])
        tree = cKDTree(mid)
        free_idx = np.flatnonzero(self.free)
        if free_idx.size == 0:
            self._pair_node = np.empty(0, dtype=int)
            self._pair_seg = np.empty(0, dtype=int)
            return
        r_search = 0.5 * float(self.rest_len.max()) + 2.0 * R_HAIR + 0.012
        found = tree.query_ball_point(self.x[free_idx], r_search)
        counts = np.fromiter((len(v) for v in found), dtype=int, count=free_idx.size)
        total = int(counts.sum())
        if total == 0:
            self._pair_node = np.empty(0, dtype=int)
            self._pair_seg = np.empty(0, dtype=int)
            return
        flat = np.concatenate([np.asarray(v, dtype=int) for v in found if len(v)])
        node = np.repeat(free_idx, counts)
        keep = self.strand_id[node] != self.strand_id[self.ea[flat]]
        self._pair_node = node[keep]
        self._pair_seg = flat[keep]

    def _hair_pass(self, dt: float, first: bool):
        if first:
            self._hair_report = self._empty_report()
        pn = self._pair_node
        if pn.size == 0:
            return
        ps = self._pair_seg
        A = self.x[self.ea[ps]]
        B = self.x[self.eb[ps]]
        d = B - A
        dd = np.maximum(np.einsum("ij,ij->i", d, d), 1e-18)
        tp = np.clip(np.einsum("ij,ij->i", self.x[pn] - A, d) / dd, 0.0, 1.0)
        cp = A + tp[:, None] * d
        w = self.x[pn] - cp
        dist = _norm(w)
        good = (dist > 1e-9) & (dist < 2.0 * R_HAIR)
        if not np.any(good):
            return
        pn = pn[good]
        ps = ps[good]
        tp = tp[good]
        w = w[good]
        dist = dist[good]
        nrm = w / dist[:, None]
        gap = dist - 2.0 * R_HAIR
        wa = 1.0 - tp
        wb = tp
        ea = self.ea[ps]
        eb = self.eb[ps]
        inv = self.inv_m
        denom = inv[pn] + wa * wa * inv[ea] + wb * wb * inv[eb]
        m_eff = np.where(denom > 1e-12, 1.0 / np.maximum(denom, 1e-12), 0.0)
        vrel = self.v[pn] - (wa[:, None] * self.v[ea] + wb[:, None] * self.v[eb])
        vn = np.einsum("ij,ij->i", vrel, nrm)
        jn = m_eff * np.maximum(0.0, -vn)
        vt = vrel - vn[:, None] * nrm
        sp = _norm(vt)
        jt = np.minimum(m_eff * sp, MU_K * jn)
        jtv = -vt / np.maximum(sp, 1e-12)[:, None] * jt[:, None]
        imp = jn[:, None] * nrm + jtv
        np.add.at(self.v, pn, imp * inv[pn][:, None])
        np.add.at(self.v, ea, -(wa[:, None] * imp) * inv[ea][:, None])
        np.add.at(self.v, eb, -(wb[:, None] * imp) * inv[eb][:, None])
        sep = np.clip(-gap, 0.0, 0.02)
        Mi = self.mass[pn]
        Ms = wa * self.mass[ea] + wb * self.mass[eb]
        den = Mi + Ms
        alpha = np.where(den > 1e-12, Ms / np.maximum(den, 1e-12), 0.0)
        beta = 1.0 - alpha
        dseg = -beta[:, None] * sep[:, None] * nrm
        np.add.at(self.x, pn, alpha[:, None] * sep[:, None] * nrm)
        np.add.at(self.x, ea, wa[:, None] * dseg)
        np.add.at(self.x, eb, wb[:, None] * dseg)
        if first:
            self._hair_report = {
                "normal": nrm.copy(),
                "normal_force": jn / dt,
                "tangential_force": jtv / dt,
                "relative_velocity": vrel.copy(),
            }

    def _resolve_contacts(self, dt: float) -> None:
        self._pair_age += 1
        self._refresh_hair_pairs()
        for it in range(3):
            first = it == 0
            self._body_pass(dt, first)
            self._hair_pass(dt, first)
        body = self._body_report
        hair = self._hair_report
        self.report = {
            "normal": np.concatenate([body["normal"], hair["normal"]], axis=0),
            "normal_force": np.concatenate([body["normal_force"], hair["normal_force"]]),
            "tangential_force": np.concatenate(
                [body["tangential_force"], hair["tangential_force"]], axis=0
            ),
            "relative_velocity": np.concatenate(
                [body["relative_velocity"], hair["relative_velocity"]], axis=0
            ),
        }
        self.contact_count_max = max(self.contact_count_max, len(self.report["normal"]))

    # ------------------------------------------------------------------ steps
    def _step(self) -> None:
        dt = self.dt
        t_new = (self.step_count + 1) * dt
        x_prev = self.x.copy()
        v_prev = self.v.copy()
        F = self._elastic_forces(self.x, self.v)
        root_reaction = -F[self.is_root].sum(axis=0)
        acc = F * self.inv_m[:, None]
        acc[:, 2] += GRAVITY[2]
        self.v += dt * acc
        self.v *= max(0.0, 1.0 - self.body_damping * dt)
        self.x += dt * self.v
        self._set_roots(t_new)
        self._resolve_contacts(dt)
        self._set_roots(t_new)

        # safety net: never emit non-finite or unbounded states
        if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()):
            self.x = x_prev
            self.v = v_prev
        else:
            speed = _norm(self.v)
            too_fast = speed > 60.0
            if np.any(too_fast):
                self.v[too_fast] *= 60.0 / speed[too_fast][:, None]
            np.clip(self.x, -5.0, 5.0, out=self.x)

        grav_total = np.array([0.0, 0.0, GRAVITY[2] * float(self.mass.sum())])
        self.external_force = np.stack(
            [grav_total, self._body_force_n, self._body_force_t, root_reaction]
        )
        self.cum_external_impulse = self.cum_external_impulse + self.external_force.sum(0) * dt
        self.step_count += 1

    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        steps = 0
        while self.step_count < target and steps < 2_000_000:
            self._step()
            steps += 1
        self.time = float(time)
        return self._make_frame(float(time))

    # ----------------------------------------------------------------- output
    def _make_frame(self, time: float) -> StateFrame:
        rep = self.report
        C = int(len(rep["normal"]))
        fields = {
            "budget_cumulative_external_impulse": self.cum_external_impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.N)),
            "budget_external_force": self.external_force.reshape(1, 4, 3).copy(),
            "cumulative_mass_source": np.zeros(1),
            "friction_active_mask": np.ones(C),
            "friction_dynamic_coefficient": np.full(C, MU_K),
            "friction_static_coefficient": np.full(C, MU_S),
            "friction_normal": rep["normal"].copy(),
            "friction_normal_force": rep["normal_force"].copy(),
            "friction_relative_velocity": rep["relative_velocity"].copy(),
            "friction_tangential_force": rep["tangential_force"].copy(),
        }
        total_mass = float(self.mass.sum())
        scalars = {
            "reference_force": max(total_mass * abs(GRAVITY[2]), 1e-6),
            "reference_momentum": max(total_mass * REFERENCE_VELOCITY, 1e-6),
            "reference_velocity": REFERENCE_VELOCITY,
            "friction_stick_speed_tolerance": STICK_SPEED_TOLERANCE,
        }
        attributes = {
            "mass_quantity": self.mass.copy(),
            "strand_id": self.strand_id.copy(),
            "node_index": self.node_index.copy(),
            "is_root": self.is_root.astype(np.int64),
        }
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.strand_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "elastic_rod_coulomb_stick_slip",
            "internal_dt": self.dt,
            "steps": int(self.step_count),
            "nodes": int(self.N),
            "strands": int(self.nstrands),
            "max_contacts": int(self.contact_count_max),
            "total_hair_mass": float(self.mass.sum()),
            "friction_coefficient": MU_S,
        }
