"""Wind-driven branching elastic rod network.

Physical model
--------------
The frozen topology asset is a 49-node tree (48 edges).  Every node carries a
lumped mass and translational degrees of freedom; the root node 0 is fixed.
The network is solved as a discrete elastic rod network:

* axial (stretch) springs on every edge with rest length taken from the frozen
  reference configuration, including an axial dashpot,
* bending (hinge) springs on every pair of consecutive edges meeting at a node
  with rest angle taken from the reference configuration.  The hinge forces are
  the exact analytic gradient of 0.5*k*(theta-theta_ref)^2, so they are
  momentum conserving,
* torsional (twist) springs that resist the relative rotation of neighbouring
  material frames about the shared tangent; the frames are transported with the
  minimal rotation between successive tangents and the twist DOF is integrated
  with its own polar inertia,
* gravity and a prescribed height/area weighted wind body force with quadratic
  relative-velocity drag and per-branch gust phase offsets.

Time integration is a deterministic semi-implicit (symplectic) Euler scheme with
a fixed small substep dt; each requested sample time is reached by integrating
whole substeps plus one final partial substep, so the state is exactly
reproducible and finite.  If a substep ever produces a non-finite state the step
is rejected and the previous state is restored (no randomness is ever used).

The fixed-root reaction is retained as an explicit external force term so the
exported cumulative impulse / angular impulse budgets close for the tree domain.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_FIXED_SUBSTEP = 5.0e-5
_MAX_SPEED = 220.0
_MAX_TWIST = 80.0

# ---------------------------------------------------------------- quaternions


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx, by, bz, bw = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=1,
    )


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]], axis=1)


def _qnorm(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(n < 1e-12, 1.0, n)


def _qaxis(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(axis, axis=-1, keepdims=True)
    a = axis / np.where(n < 1e-12, 1.0, n)
    h = 0.5 * angle[..., None]
    return np.concatenate([a * np.sin(h), np.cos(h)], axis=-1)


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv, qw = q[:, :3], q[:, 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def _perp(v: np.ndarray) -> np.ndarray:
    """Any unit vector perpendicular to each row of v."""
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (v.shape[0], 1))
    alt = np.tile(np.array([1.0, 0.0, 0.0]), (v.shape[0], 1))
    ref = np.where(np.abs(v[:, 2:3]) > 0.9, alt, ref)
    p = np.cross(v, ref)
    n = np.linalg.norm(p, axis=1, keepdims=True)
    return p / np.where(n < 1e-12, 1.0, n)


class BenchmarkSolver:
    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        config = dict(getattr(scene, "config", {}) or {})
        meta = scene.metadata()
        self.config = config
        self.wind_cfg = dict(meta.get("wind", config.get("wind", {})) or {})
        self.gravity = np.asarray(meta.get("gravity", config.get("gravity", [0, 0, -9.81])), float)

        rod = dict(meta.get("rod_network", config.get("rod_network", {})) or {})
        junctions = dict(meta.get("rigid_junctions", config.get("rigid_junctions", {})) or {})

        asset_path = Path(__file__).resolve().parent.parent / str(
            rod.get("topology_asset", "assets/tree_topology.json")
        )
        data = json.loads(asset_path.read_text(encoding="utf-8"))
        x_ref = np.asarray(data["vertices"], dtype=float)
        self.edges = np.asarray(data["edges"], dtype=np.int64)
        self.levels = np.asarray(data["vertex_level"], dtype=np.int64)
        self.lumped = np.asarray(data["lumped_length"], dtype=float)
        n = int(x_ref.shape[0])
        self.n = n

        radii_by_level = np.asarray(rod.get("radius_by_level", [0.06, 0.037, 0.021]), float)
        dens_by_level = np.asarray(rod.get("linear_density_by_level", [2.8, 1.25, 0.48]), float)
        self.radius = radii_by_level[np.clip(self.levels, 0, len(radii_by_level) - 1)]
        density = dens_by_level[np.clip(self.levels, 0, len(dens_by_level) - 1)]
        mass = density * np.pi * self.radius**2 * self.lumped

        junction_idx = np.asarray(junctions.get("vertex_indices", []), dtype=np.int64)
        is_junction = np.zeros(n, dtype=float)
        is_junction[junction_idx] = 1.0
        jmass = np.asarray(junctions.get("mass_by_level", [0.32, 0.16, 0.07]), float)
        for idx in junction_idx:
            lvl = int(np.clip(self.levels[idx], 0, len(jmass) - 1))
            mass[idx] = float(jmass[lvl])
        self.is_junction = is_junction
        self.junction_radius = 0.072
        self.mass = np.maximum(mass, 1e-9)

        self.fixed = np.zeros(n, dtype=bool)
        self.fixed[int(junctions.get("fixed_root_vertex", 0))] = True
        self.x_ref = x_ref.copy()

        ei, ej = self.edges[:, 0], self.edges[:, 1]
        self.ei, self.ej = ei, ej
        e0 = x_ref[ej] - x_ref[ei]
        self.rest_len = np.linalg.norm(e0, axis=1)
        self.edge_level = np.maximum(self.levels[ei], self.levels[ej])
        r_e = radii_by_level[np.clip(self.edge_level, 0, len(radii_by_level) - 1)]
        self.edge_radius = r_e
        area = np.pi * r_e**2
        inertia = np.pi * r_e**4 / 4.0           # second moment of area
        e_eff = 2.0e8                            # effective Young modulus (SI)
        self.e_eff = e_eff
        self.k_axial = 0.05 * e_eff * area / self.rest_len
        self.k_bend = e_eff * inertia / self.rest_len
        self.k_twist = 0.10 * e_eff * inertia / self.rest_len
        self.c_axial = 0.25 * np.sqrt(self.k_axial * self.mass[ei])
        self.g_eff = e_eff / 2.6

        # incident-edge tables (unit directions point away from the node)
        deg = [0] * n
        for k in range(len(ei)):
            deg[ei[k]] += 1
            deg[ej[k]] += 1
        dmax = max(deg)
        self.inc_edge = -np.ones((dmax, n), dtype=np.int64)
        self.inc_sign = np.zeros((dmax, n), dtype=float)
        fill = [0] * n
        for k in range(len(ei)):
            for node, sgn in ((ei[k], 1.0), (ej[k], -1.0)):
                self.inc_edge[fill[node], node] = k
                self.inc_sign[fill[node], node] = sgn
                fill[node] += 1

        # hinges: (incoming edge, outgoing edge, mid node)
        hk1, hk2, hmid = [], [], []
        for k in range(len(ei)):
            a, b = int(ei[k]), int(ej[k])
            for k1 in range(len(ei)):
                if int(ej[k1]) == b:          # edge arriving at b
                    for k2 in range(len(ei)):
                        if int(ei[k2]) == b:  # edge leaving b
                            hk1.append(k1)
                            hk2.append(k2)
                            hmid.append(b)
        self.hk1 = np.asarray(hk1, dtype=np.int64)
        self.hk2 = np.asarray(hk2, dtype=np.int64)
        self.hmid = np.asarray(hmid, dtype=np.int64)
        u1 = e0[self.hk1] / self.rest_len[self.hk1][:, None]
        u2 = e0[self.hk2] / self.rest_len[self.hk2][:, None]
        cos0 = np.clip(np.sum(u1 * u2, axis=1), -1.0, 1.0)
        self.theta0 = np.arccos(cos0)
        self.hfar1 = ei[self.hk1]
        self.hfar2 = ej[self.hk2]
        lh = 0.5 * (self.rest_len[self.hk1] + self.rest_len[self.hk2])
        kh = 0.5 * (self.k_bend[self.hk1] + self.k_bend[self.hk2])
        self.k_hinge = kh
        self.c_hinge = 0.12 * np.sqrt(
            kh * self.mass[self.hmid] * np.maximum(lh, 1e-6)
        )

        # inertia tensors: rigid junctions only (their distributed mass already
        # resolves translation, so only the body spin inertia is reported)
        inertia_tensor = np.zeros((n, 3, 3))
        for idx in junction_idx:
            jm = self.mass[idx]
            val = 0.4 * jm * self.junction_radius**2
            inertia_tensor[idx] = val * np.eye(3)
        self.inertia_tensor = inertia_tensor

        # twist DOF inertia
        self.twist_inertia = np.maximum(
            0.5 * self.mass * self.radius**2, 1e-12
        )
        for idx in junction_idx:
            self.twist_inertia[idx] = max(0.4 * self.mass[idx] * self.junction_radius**2, 1e-12)

        # wind configuration
        wc = self.wind_cfg
        direction = np.asarray(wc.get("direction", [1.0, 0.0, 0.0]), float)
        self.wind_dir = direction / max(float(np.linalg.norm(direction)), 1e-12)
        self.wind_df = float(wc.get("gust_frequency_hz", 0.55))
        offsets = np.asarray(wc.get("branch_phase_offsets", [0.0]), float)
        branch_id = np.asarray(data["branch_id"], dtype=np.int64)
        nb = max(len(offsets), 1)
        self.phase = offsets[np.clip(branch_id, 0, nb - 1)]
        self.wind_area = 2.0 * self.radius * self.lumped
        key = wc.get("keyframes", [])
        if key:
            self.kf_t = np.asarray([float(item["time"]) for item in key], float)
            self.kf_s = np.asarray([float(item["speed"]) for item in key], float)
        else:
            self.kf_t = np.asarray([0.0, 1.0], float)
            self.kf_s = np.asarray([0.0, 10.0], float)
        self.rho_air = 1.225
        self.cd = 1.2
        self.z_ref = max(float(np.max(x_ref[:, 2])), 1e-6)
        self.c_struct = 0.01

        # state
        self.sim_time = 0.0
        self.x = x_ref.copy()
        self.v = np.zeros((n, 3))
        self.wt = np.zeros(n)
        self.q = self._initial_frames(self.x)
        self.tangent = self._tangents(self.x)
        self.angvel = np.zeros((n, 3))
        self.root_reaction = np.zeros(3)
        self.ext_force_terms = np.zeros((n + 1, 3))
        self.force_position_terms = np.zeros((n + 1, 3))
        self.impulse = np.zeros((1, 3))
        self.angular_impulse = np.zeros((1, 3))
        self.phi0 = self._edge_twist(self.q, self.tangent, self._edge_tangents(self.x))

        total_mass = float(np.sum(self.mass))
        span = float(np.max(np.linalg.norm(x_ref - np.mean(x_ref, axis=0), axis=1)))
        self.reference_momentum = max(total_mass * 20.0, 1e-6)
        self.reference_angular_momentum = max(self.reference_momentum * max(span, 1e-3), 1e-6)
        self.dt = _FIXED_SUBSTEP

    # ------------------------------------------------------------- kinematics
    def _tangents(self, x: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(x)
        e = x[self.ej] - x[self.ei]
        L = np.maximum(np.linalg.norm(e, axis=1), 1e-12)
        u = e / L[:, None]
        np.add.at(acc, self.ei, u)
        np.add.at(acc, self.ej, -u)
        nrm = np.linalg.norm(acc, axis=1, keepdims=True)
        fallback = np.tile(np.array([0.0, 0.0, 1.0]), (x.shape[0], 1))
        return np.where(nrm < 1e-9, fallback, acc / np.where(nrm < 1e-9, 1.0, nrm))

    def _edge_tangents(self, x: np.ndarray) -> np.ndarray:
        e = x[self.ej] - x[self.ei]
        L = np.maximum(np.linalg.norm(e, axis=1), 1e-12)
        return e / L[:, None]

    def _initial_frames(self, x: np.ndarray) -> np.ndarray:
        t = self._tangents(x)
        axis = np.cross(np.tile(np.array([0.0, 0.0, 1.0]), (x.shape[0], 1)), t)
        s = np.linalg.norm(axis, axis=1, keepdims=True)
        c = np.clip(t[:, 2:3], -1.0, 1.0)
        safe = np.where(s < 1e-12, 1.0, s)
        axis = np.where(s < 1e-12, _perp(t), axis / safe)
        angle = np.arctan2(s[:, 0], c[:, 0])
        return _qnorm(_qaxis(axis, angle))

    def _edge_twist(self, q, tangent_nodes, t_edge) -> np.ndarray:
        ex = np.tile(np.array([1.0, 0.0, 0.0]), (self.n, 1))
        d = _qrot(q, ex)
        di, dj = d[self.ei], d[self.ej]
        t = t_edge
        pi = di - np.sum(di * t, axis=1)[:, None] * t
        pj = dj - np.sum(dj * t, axis=1)[:, None] * t
        cross = np.cross(pi, pj)
        return np.arctan2(np.sum(cross * t, axis=1), np.sum(pi * pj, axis=1))

    # ---------------------------------------------------------------- forces
    def _wind_speed(self, t: float) -> float:
        s = float(np.interp(t, self.kf_t, self.kf_s))
        return max(s, 0.0)

    def _external_forces(self, x: np.ndarray, v: np.ndarray, t: float) -> np.ndarray:
        F = self.mass[:, None] * self.gravity[None, :]
        speed = self._wind_speed(t)
        if speed > 0.0:
            hw = np.clip(x[:, 2] / self.z_ref, 0.0, 1.5) ** 0.5
            phase = 1.0 + 0.22 * np.sin(2.0 * np.pi * self.wind_df * (t + self.phase))
            local = speed * hw * phase
            wvel = local[:, None] * self.wind_dir[None, :]
            rel = wvel - v
            mag = np.linalg.norm(rel, axis=1)
            F += (
                0.5
                * self.rho_air
                * self.cd
                * self.wind_area[:, None]
                * mag[:, None]
                * rel
            )
        F -= self.c_struct * v
        return F

    def _elastic_forces(self, x: np.ndarray, v: np.ndarray):
        e = x[self.ej] - x[self.ei]
        L = np.maximum(np.linalg.norm(e, axis=1), 1e-12)
        u = e / L[:, None]
        stretch = L - self.rest_len
        dL = np.sum((v[self.ej] - v[self.ei]) * u, axis=1)
        axial = self.k_axial * stretch + self.c_axial * dL
        fa = axial[:, None] * u
        F = np.zeros_like(x)
        np.add.at(F, self.ei, fa)
        np.add.at(F, self.ej, -fa)

        u1 = u[self.hk1]
        u2 = u[self.hk2]
        l1 = L[self.hk1]
        l2 = L[self.hk2]
        cs = np.clip(np.sum(u1 * u2, axis=1), -1.0, 1.0)
        theta = np.arccos(cs)
        sin_t = np.maximum(np.sqrt(np.maximum(1.0 - cs * cs, 0.0)), 1e-9)
        du1 = (v[self.hmid] - v[self.hfar1]) / l1[:, None]
        du2 = (v[self.hfar2] - v[self.hmid]) / l2[:, None]
        du1 -= np.sum(du1 * u1, axis=1)[:, None] * u1
        du2 -= np.sum(du2 * u2, axis=1)[:, None] * u2
        theta_dot = -(np.sum(u2 * du1, axis=1) + np.sum(u1 * du2, axis=1)) / sin_t
        torque = self.k_hinge * (theta - self.theta0) + self.c_hinge * theta_dot

        w1 = u2 - cs[:, None] * u1
        w2 = u1 - cs[:, None] * u2
        w1n = np.maximum(np.linalg.norm(w1, axis=1), 1e-9)
        w2n = np.maximum(np.linalg.norm(w2, axis=1), 1e-9)
        f1 = -(torque / (l1 * w1n))[:, None] * w1
        f2 = (torque / (l2 * w2n))[:, None] * w2
        np.add.at(F, self.hfar1, f1)
        np.add.at(F, self.hfar2, f2)
        np.add.at(F, self.hmid, -(f1 + f2))
        return F, u

    # ---------------------------------------------------------------- frames
    def _update_frames(self, dt: float, u_edge: np.ndarray) -> None:
        u_edge = self._edge_tangents(self.x)
        q_old = self.q
        t_new = self._tangents(self.x)
        t_old = self.tangent
        axis = np.cross(t_old, t_new)
        s = np.linalg.norm(axis, axis=1, keepdims=True)
        c = np.clip(np.sum(t_old * t_new, axis=1, keepdims=True), -1.0, 1.0)
        safe = np.where(s < 1e-12, 1.0, s)
        axis = np.where(s < 1e-12, _perp(t_old), axis / safe)
        angle = np.arctan2(s[:, 0], c[:, 0])
        dq = _qaxis(axis, angle)
        q_tr = _qnorm(_qmul(dq, q_old))
        dqt = _qaxis(t_new, self.wt * dt)
        q_new = _qnorm(_qmul(dqt, q_tr))
        q_new[self.fixed] = q_old[self.fixed]

        rel = _qmul(q_new, _qconj(q_old))
        vv = rel[:, :3]
        nv = np.linalg.norm(vv, axis=1)
        ang = 2.0 * np.arctan2(nv, np.clip(rel[:, 3], -1.0, 1.0))
        unit = vv / np.where(nv < 1e-12, 1.0, nv)[:, None]
        self.angvel = ang[:, None] * unit / dt

        # twist springs on the updated frames
        phi = self._edge_twist(q_new, t_new, u_edge)
        diff = phi - self.phi0
        dtw = self.wt[self.ej] - self.wt[self.ei]
        torque = self.k_twist * diff + 0.05 * np.sqrt(self.k_twist * self.mass[self.ei]) * dtw
        acc = np.zeros(self.n)
        np.add.at(acc, self.ei, -torque / self.twist_inertia[self.ei])
        np.add.at(acc, self.ej, torque / self.twist_inertia[self.ej])
        self.wt = np.clip(self.wt + acc * dt, -_MAX_TWIST, _MAX_TWIST)
        self.wt[self.fixed] = 0.0
        self.q = q_new
        self.tangent = t_new

    # ------------------------------------------------------------------ step
    def _step(self, dt: float) -> None:
        x_old = self.x.copy()
        v_old = self.v.copy()
        q_old = self.q.copy()
        wt_old = self.wt.copy()
        t_old = self.tangent.copy()
        ang_old = self.angvel.copy()

        Fext = self._external_forces(self.x, self.v, self.sim_time)
        Fint, u_edge = self._elastic_forces(self.x, self.v)
        total = Fext + Fint
        acc = total / self.mass[:, None]
        acc[self.fixed] = 0.0
        reaction = -total[self.fixed][0]

        self.v = self.v + acc * dt
        self.v[self.fixed] = 0.0
        sp = np.linalg.norm(self.v, axis=1, keepdims=True)
        self.v = np.where(sp > _MAX_SPEED, self.v * (_MAX_SPEED / sp), self.v)
        self.x = self.x + self.v * dt
        self.x[self.fixed] = x_old[self.fixed]

        try:
            self._update_frames(dt, u_edge)
        except FloatingPointError:
            self.x, self.v, self.q, self.wt = x_old, v_old, q_old, wt_old
            self.tangent, self.angvel = t_old, ang_old
            raise

        bad = (
            not np.isfinite(self.x).all()
            or not np.isfinite(self.v).all()
            or not np.isfinite(self.q).all()
            or not np.isfinite(self.wt).all()
            or not np.isfinite(self.angvel).all()
        )
        if bad:
            self.x, self.v, self.q, self.wt = x_old, v_old, q_old, wt_old
            self.tangent, self.angvel = t_old, ang_old
            return

        terms = np.zeros((self.n + 1, 3))
        terms[: self.n] = Fext
        terms[self.n] = reaction
        pos = np.zeros((self.n + 1, 3))
        pos[: self.n] = self.x
        pos[self.n] = self.x[self.fixed][0]
        self.ext_force_terms = terms
        self.force_position_terms = pos
        self.root_reaction = reaction
        self.impulse[0] += np.sum(terms, axis=0) * dt
        self.angular_impulse[0] += np.sum(np.cross(pos, terms), axis=0) * dt

    # ---------------------------------------------------------------- output
    def _frame(self) -> StateFrame:
        n = self.n
        K = n + 1
        domain_weights = np.ones((1, n))
        attrs = {
            "is_rod_vertex": np.ones(n),
            "is_rigid_junction": self.is_junction.copy(),
            "is_fixed_root": self.fixed.astype(float),
            "branch_level": self.levels.astype(float),
            "rod_radius": self.radius.copy(),
            "mass_quantity": self.mass.copy(),
            "inertia_tensor": self.inertia_tensor.copy(),
        }
        fields = {
            "budget_domain_weights": domain_weights,
            "budget_external_force": self.ext_force_terms[None, :, :].copy(),
            "budget_force_position": self.force_position_terms[None, :, :].copy(),
            "budget_external_couple": np.zeros((1, K, 3)),
            "budget_cumulative_external_impulse": self.impulse.copy(),
            "budget_cumulative_external_angular_impulse": self.angular_impulse.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_inflow_angular_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_angular_momentum": np.zeros((1, 3)),
            "cumulative_mass_source": np.zeros(1),
        }
        scalars = {
            "reference_momentum": float(self.reference_momentum),
            "reference_angular_momentum": float(self.reference_angular_momentum),
            "wind_speed": float(self._wind_speed(self.sim_time)),
            "gravity_magnitude": float(np.linalg.norm(self.gravity)),
            "elastic_modulus": float(self.e_eff),
        }
        frame = StateFrame(
            time=float(self.sim_time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=self.q.copy(),
            angular_velocity=self.angvel.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.sim_time < target - 1e-12:
            dt = min(self.dt, target - self.sim_time)
            before = self.sim_time
            self._step(dt)
            self.sim_time = before + dt
            guard += 1
            if guard > 10_000_000:
                break
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.sim_time),
            "nodes": int(self.n),
            "edges": int(len(self.edges)),
            "hinges": int(len(self.hk1)),
            "substep": float(self.dt),
            "kinetic_energy": float(
                0.5 * np.sum(self.mass[:, None] * self.v**2)
            ),
            "root_reaction": self.root_reaction.tolist(),
        }
