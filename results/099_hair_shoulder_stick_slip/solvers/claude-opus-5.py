"""Self-contained elastic-strand solver with nonsmooth Coulomb friction.

Physical model
--------------
* 84 wavy elastic strands (22 lumped nodes each) rooted on a kinematically
  driven head / neck / shoulder / torso assembly.
* Strand elasticity: axial stretch springs plus second-neighbour springs whose
  rest lengths store the wavy reference shape (discrete bending energy).
* Body forces: gravity and a light aerodynamic drag.
* Contacts (hair-body and hair-hair) are resolved at the velocity level with a
  Moreau-style time-stepping scheme: projected Jacobi normal impulses with a
  Baumgarte penetration bias, followed by a projected Coulomb friction step
  (mu = 0.3 for static and kinetic, at both contact families).  The friction
  projection can never reverse the tangential relative velocity, so genuine
  sticking and slipping are produced without any penalty-stiffness limit.

Everything is integrated explicitly in time with a fixed sub-step; no scripted
motion, keyframes or precomputed trajectories are used.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------- constants ---
DT_SUB = 2.0e-4          # explicit sub-step [s]
K_STRETCH = 400.0        # axial stiffness [N/m]
C_STRETCH = 0.02         # axial damping [N*s/m]
K_BEND = 3.0             # second-neighbour (bending) stiffness [N/m]
C_BEND = 0.006           # bending damping [N*s/m]
C_AIR = 3.0e-4           # per-node aerodynamic drag [N*s/m]
HAIR_HAIR_D0 = 0.014     # wisp-wisp contact distance [m]
NEIGHBOR_CUTOFF = 0.024  # broad-phase radius [m]
REBUILD_EVERY = 20       # sub-steps between broad-phase rebuilds
BODY_MARGIN = 0.002      # hair-body detection band [m]
NORMAL_ITERS = 3         # projected Jacobi sweeps for normal impulses
FRICTION_ITERS = 4       # projected Jacobi sweeps for Coulomb friction
BAUMGARTE = 0.25         # penetration recovery fraction per sub-step
MAX_PUSH_SPEED = 0.15    # cap on penetration recovery speed [m/s]
STICK_SPEED_TOL = 2.0e-3  # stick / slip classification tolerance [m/s]
HAIR_SLOTS = 3072        # reported hair-hair contact slots


def _quat_matrix(q) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion."""
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n <= 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class _Body:
    """Scaled capsule / ellipsoid collider with a translating transform."""

    def __init__(self, spec: dict):
        geom = spec.get("geometry", {})
        tr = spec.get("transform", {})
        self.kind = str(geom.get("type", "sphere"))
        self.radius = float(geom.get("radius", 0.1))
        self.half = 0.5 * float(geom.get("height", 0.0))
        self.center0 = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float)
        self.scale = np.asarray(tr.get("scale", [1.0, 1.0, 1.0]), dtype=float)
        self.rot = _quat_matrix(tr.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]))
        self.offset = np.zeros(3)

    def gap_normal(self, points: np.ndarray):
        """First-order signed distance and outward unit normal at ``points``."""
        delta = points - (self.center0 + self.offset)
        local = (delta @ self.rot) / self.scale
        if self.kind == "capsule" and self.half > 0.0:
            axial = np.clip(local[:, 2], -self.half, self.half)
            rel = local.copy()
            rel[:, 2] -= axial
        else:
            rel = local
        dist = np.linalg.norm(rel, axis=1)
        dist = np.maximum(dist, 1e-12)
        n_local = rel / dist[:, None]
        signed = dist - self.radius
        grad = (n_local / self.scale) @ self.rot.T
        gnorm = np.maximum(np.linalg.norm(grad, axis=1), 1e-12)
        return signed / gnorm, grad / gnorm[:, None]
class BenchmarkSolver:
    """Explicit strand dynamics with nonsmooth frictional contact."""

    # ------------------------------------------------------------ setup ---
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        cfg = dict(scene.config)
        self.config = cfg

        frame0 = scene.initial_state()
        self.pos = np.array(frame0.position, dtype=float)
        self.vel = np.zeros_like(self.pos)
        self.mass = np.array(frame0.mass, dtype=float)
        self.strand_id = np.asarray(frame0.attributes["strand_id"], dtype=int)
        self.node_index = np.asarray(frame0.attributes["node_index"], dtype=int)
        self.n = len(self.pos)
        self.n_strand = int(self.strand_id.max()) + 1
        self.per_strand = self.n // self.n_strand
        self.hair_radius = float(frame0.scalars.get("strand_radius", 0.0042))

        self.is_root = self.node_index == 0
        self.dyn = ~self.is_root
        self.root_idx = np.flatnonzero(self.is_root)
        self.dyn_idx = np.flatnonzero(self.dyn)
        # zero-mass roots are kinematic; give the dynamic nodes a safe inverse mass
        self.inv_mass = np.zeros(self.n)
        self.inv_mass[self.dyn] = 1.0 / self.mass[self.dyn]
        self.root_pos0 = self.pos[self.root_idx].copy()

        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        # ---- prescribed body / root motion -----------------------------
        motion = cfg["head_motion"]
        axis = np.asarray(motion["axis"], dtype=float)
        nrm = np.linalg.norm(axis)
        self.axis = axis / nrm if nrm > 0 else np.array([1.0, 0.0, 0.0])
        self.disp = float(motion["displacement"])
        self.t_start = float(motion["start_time"])
        self.t_stop = float(motion["stop_time"])

        # ---- colliders from the frozen geometry ------------------------
        geo = cfg.get("scene_geometry", {}) or {}
        self.bodies: list[_Body] = []
        for spec in geo.get("objects", []) or []:
            if spec.get("semantic") == "rigid" and spec.get("collision_enabled", False):
                self.bodies.append(_Body(spec))

        # ---- strand connectivity (reference lengths hold the waviness) --
        p3 = self.pos.reshape(self.n_strand, self.per_strand, 3)
        self.rest_stretch = np.linalg.norm(p3[:, 1:] - p3[:, :-1], axis=2)
        self.rest_bend = np.linalg.norm(p3[:, 2:] - p3[:, :-2], axis=2)

        # ---- hair-hair broad phase state -------------------------------
        self.hh_i = np.zeros(0, dtype=int)
        self.hh_j = np.zeros(0, dtype=int)
        self.step_count = 0

        # ---- reported contact slot layout ------------------------------
        self.n_body_slots = len(self.dyn_idx)
        self.n_slots = self.n_body_slots + HAIR_SLOTS
        self.mu_hair_body = float(cfg["friction"]["hair_body"]["static"])
        self.mu_hair_hair = float(cfg["friction"]["hair_hair"]["static"])
        self.mu_static = np.empty(self.n_slots)
        self.mu_static[: self.n_body_slots] = self.mu_hair_body
        self.mu_static[self.n_body_slots :] = self.mu_hair_hair
        self.mu_dynamic = self.mu_static.copy()
        self._reset_contact_report()

        # ---- accumulators for the momentum budget ----------------------
        self.n_terms = 4  # gravity, body contact, root transmission, drag
        self.force_accum = np.zeros((self.n_terms, 3))
        self.accum_steps = 0
        self.cum_impulse = np.zeros(3)
        self.time = 0.0

        length_scale = float(np.max(self.rest_stretch.sum(axis=1)))
        self.ref_velocity = float(np.sqrt(abs(self.gravity[2]) * max(length_scale, 1e-6)))
        total_mass = float(self.mass.sum())
        self.ref_force = max(total_mass * float(np.linalg.norm(self.gravity)), 1e-9)
        self.ref_momentum = max(total_mass * self.ref_velocity, 1e-9)

        self._probe_initial_terms()

    # -------------------------------------------------------- kinematics ---
    def _body_shift(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Prescribed translation and velocity of the rigid assembly."""
        t0, t1 = self.t_start, self.t_stop
        if t <= t0:
            s, ds = 0.0, 0.0
        elif t >= t1:
            s, ds = 1.0, 0.0
        else:
            u = (t - t0) / (t1 - t0)
            s = u - np.sin(2.0 * np.pi * u) / (2.0 * np.pi)
            ds = (1.0 - np.cos(2.0 * np.pi * u)) / (t1 - t0)
        return self.axis * (self.disp * s), self.axis * (self.disp * ds)

    def _reset_contact_report(self) -> None:
        self.rep_mask = np.zeros(self.n_slots)
        self.rep_normal = np.zeros((self.n_slots, 3))
        self.rep_normal[:, 2] = 1.0
        self.rep_fn = np.zeros(self.n_slots)
        self.rep_ft = np.zeros((self.n_slots, 3))
        self.rep_vrel = np.zeros((self.n_slots, 3))
    # ------------------------------------------------------ internal forces ---
    def _elastic_drag(self) -> np.ndarray:
        """Axial + bending spring forces and aerodynamic drag on every node."""
        ns, npn = self.n_strand, self.per_strand
        p = self.pos.reshape(ns, npn, 3)
        v = self.vel.reshape(ns, npn, 3)
        force = np.zeros_like(p)

        # axial stretch springs
        d = p[:, 1:] - p[:, :-1]
        length = np.maximum(np.linalg.norm(d, axis=2), 1e-12)
        unit = d / length[:, :, None]
        dv = v[:, 1:] - v[:, :-1]
        mag = K_STRETCH * (length - self.rest_stretch) + C_STRETCH * np.sum(dv * unit, axis=2)
        f = mag[:, :, None] * unit
        force[:, :-1] += f
        force[:, 1:] -= f

        # second-neighbour springs storing the wavy rest shape
        d2 = p[:, 2:] - p[:, :-2]
        len2 = np.maximum(np.linalg.norm(d2, axis=2), 1e-12)
        unit2 = d2 / len2[:, :, None]
        dv2 = v[:, 2:] - v[:, :-2]
        mag2 = K_BEND * (len2 - self.rest_bend) + C_BEND * np.sum(dv2 * unit2, axis=2)
        f2 = mag2[:, :, None] * unit2
        force[:, :-2] += f2
        force[:, 2:] -= f2

        out = force.reshape(self.n, 3)
        out -= C_AIR * self.vel
        out[self.is_root] = 0.0
        return out

    def _rebuild_pairs(self) -> None:
        """Broad phase for wisp-wisp contacts (same-strand neighbours excluded)."""
        tree = cKDTree(self.pos)
        pairs = tree.query_pairs(NEIGHBOR_CUTOFF, output_type="ndarray")
        if len(pairs) == 0:
            self.hh_i = np.zeros(0, dtype=int)
            self.hh_j = np.zeros(0, dtype=int)
            return
        i, j = pairs[:, 0], pairs[:, 1]
        keep = self.strand_id[i] != self.strand_id[j]
        keep &= self.dyn[i] & self.dyn[j]
        self.hh_i = i[keep]
        self.hh_j = j[keep]

    def _scatter(self, idx: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Sum ``values`` rows into an [N,3] array at node indices ``idx``."""
        out = np.zeros((self.n, 3))
        if len(idx) == 0:
            return out
        for k in range(3):
            out[:, k] = np.bincount(idx, weights=values[:, k], minlength=self.n)
        return out

    # ------------------------------------------------------------- one step ---
    def _step(self, t: float, dt: float, record: bool):
        """Advance one sub-step; return the four external-force budget terms."""
        shift, body_vel = self._body_shift(t + dt)
        for body in self.bodies:
            body.offset = shift

        f_elastic = self._elastic_drag()
        f_drag = -C_AIR * self.vel
        f_drag[self.is_root] = 0.0
        f_grav = self.mass[:, None] * self.gravity
        f_grav[self.is_root] = 0.0

        # explicit velocity prediction from smooth forces
        self.vel += dt * self.inv_mass[:, None] * (f_elastic + f_grav)
        self.vel[self.is_root] = body_vel

        # ---------------- hair-body narrow phase ------------------------
        best_gap = np.full(self.n, np.inf)
        best_nrm = np.zeros((self.n, 3))
        for body in self.bodies:
            gap, nrm = body.gap_normal(self.pos)
            gap = gap - self.hair_radius
            better = gap < best_gap
            best_gap = np.where(better, gap, best_gap)
            best_nrm = np.where(better[:, None], nrm, best_nrm)
        body_hit = self.dyn & (best_gap < BODY_MARGIN)
        b_idx = np.flatnonzero(body_hit)
        b_nrm = best_nrm[b_idx]
        b_gap = best_gap[b_idx]

        # ---------------- hair-hair narrow phase ------------------------
        if self.step_count % REBUILD_EVERY == 0:
            self._rebuild_pairs()
        hi, hj = self.hh_i, self.hh_j
        if len(hi):
            dvec = self.pos[hj] - self.pos[hi]
            dlen = np.maximum(np.linalg.norm(dvec, axis=1), 1e-12)
            h_gap = dlen - HAIR_HAIR_D0
            act = h_gap < 0.0
            hi, hj = hi[act], hj[act]
            h_nrm = dvec[act] / dlen[act][:, None]
            h_gap = h_gap[act]
        else:
            h_nrm = np.zeros((0, 3))
            h_gap = np.zeros(0)

        # per-node contact multiplicity for the Jacobi normalisation
        counts = np.zeros(self.n)
        if len(b_idx):
            counts += np.bincount(b_idx, minlength=self.n)
        if len(hi):
            counts += np.bincount(hi, minlength=self.n)
            counts += np.bincount(hj, minlength=self.n)
        weight = np.maximum(counts, 1.0)

        push = np.minimum(MAX_PUSH_SPEED, BAUMGARTE * np.maximum(-b_gap, 0.0) / dt) if len(b_idx) else np.zeros(0)
        push_h = np.minimum(MAX_PUSH_SPEED, BAUMGARTE * np.maximum(-h_gap, 0.0) / dt) if len(hi) else np.zeros(0)

        lam_b = np.zeros(len(b_idx))
        lam_h = np.zeros(len(hi))
        m_b = self.mass[b_idx] if len(b_idx) else np.zeros(0)
        if len(hi):
            m_red = 1.0 / (self.inv_mass[hi] + self.inv_mass[hj])
        else:
            m_red = np.zeros(0)

        for _ in range(NORMAL_ITERS):
            if len(b_idx):
                vn = np.sum((self.vel[b_idx] - body_vel) * b_nrm, axis=1)
                dl = m_b * (push - vn) / weight[b_idx]
                dl = np.maximum(lam_b + dl, 0.0) - lam_b
                lam_b += dl
                self.vel += self.inv_mass[:, None] * self._scatter(b_idx, dl[:, None] * b_nrm)
            if len(hi):
                vn = np.sum((self.vel[hj] - self.vel[hi]) * h_nrm, axis=1)
                dl = m_red * (push_h - vn) / np.maximum(weight[hi], weight[hj])
                dl = np.maximum(lam_h + dl, 0.0) - lam_h
                lam_h += dl
                imp = dl[:, None] * h_nrm
                self.vel += self.inv_mass[:, None] * (self._scatter(hj, imp) - self._scatter(hi, imp))
            self.vel[self.is_root] = body_vel
        # ---------------- projected Coulomb friction --------------------
        # Tangential impulses opposing the relative sliding velocity, capped by
        # mu * normal impulse.  Sticking contacts request exactly the impulse
        # that cancels the tangential velocity; when that exceeds the Coulomb
        # cone the contact slips at the kinetic limit.
        gam_b = np.zeros((len(b_idx), 3))
        gam_h = np.zeros((len(hi), 3))
        vrel_b = np.zeros((len(b_idx), 3))
        vrel_h = np.zeros((len(hi), 3))

        # Per-contact caps: the Coulomb cone, and the impulse that exactly
        # arrests the incoming tangential relative velocity.  Capping by the
        # latter means friction can never reverse sliding, so no energy is
        # injected and the pair impulses stay exactly antisymmetric.
        if len(b_idx):
            vrel_b = self.vel[b_idx] - body_vel
            vt0 = vrel_b - np.sum(vrel_b * b_nrm, axis=1)[:, None] * b_nrm
            cap_b = np.minimum(self.mu_hair_body * lam_b, m_b * np.linalg.norm(vt0, axis=1))
        else:
            cap_b = np.zeros(0)
        if len(hi):
            vrel_h = self.vel[hi] - self.vel[hj]
            vt0 = vrel_h - np.sum(vrel_h * h_nrm, axis=1)[:, None] * h_nrm
            cap_h = np.minimum(self.mu_hair_hair * lam_h, m_red * np.linalg.norm(vt0, axis=1))
        else:
            cap_h = np.zeros(0)

        for it in range(FRICTION_ITERS):
            if len(b_idx):
                vrel_b = self.vel[b_idx] - body_vel
                vt = vrel_b - np.sum(vrel_b * b_nrm, axis=1)[:, None] * b_nrm
                trial = gam_b - (m_b / weight[b_idx])[:, None] * vt
                mag = np.linalg.norm(trial, axis=1)
                over = mag > cap_b
                if np.any(over):
                    safe = np.maximum(mag[over], 1e-18)
                    trial[over] *= (cap_b[over] / safe)[:, None]
                delta = trial - gam_b
                gam_b = trial
                self.vel += self.inv_mass[:, None] * self._scatter(b_idx, delta)
            if len(hi):
                vrel_h = self.vel[hi] - self.vel[hj]
                vt = vrel_h - np.sum(vrel_h * h_nrm, axis=1)[:, None] * h_nrm
                scl = m_red / np.maximum(weight[hi], weight[hj])
                trial = gam_h - scl[:, None] * vt
                mag = np.linalg.norm(trial, axis=1)
                over = mag > cap_h
                if np.any(over):
                    safe = np.maximum(mag[over], 1e-18)
                    trial[over] *= (cap_h[over] / safe)[:, None]
                delta = trial - gam_h
                gam_h = trial
                self.vel += self.inv_mass[:, None] * (
                    self._scatter(hi, delta) - self._scatter(hj, delta)
                )
            self.vel[self.is_root] = body_vel

        ft_b = gam_b / dt
        ft_h = gam_h / dt

        # ---------------- position update -------------------------------
        self.pos += dt * self.vel
        self.pos[self.root_idx] = self.root_pos0 + shift

        # ---------------- budget terms ----------------------------------
        f_contact_body = np.zeros(3)
        if len(b_idx):
            f_contact_body = (lam_b[:, None] * b_nrm).sum(axis=0) / dt + ft_b.sum(axis=0)
        terms = np.zeros((self.n_terms, 3))
        terms[0] = f_grav.sum(axis=0)
        terms[1] = f_contact_body
        terms[2] = f_elastic[self.dyn].sum(axis=0) - f_drag[self.dyn].sum(axis=0)
        terms[3] = f_drag[self.dyn].sum(axis=0)

        if record:
            self._reset_contact_report()
            if len(b_idx):
                slot = np.searchsorted(self.dyn_idx, b_idx)
                fn = lam_b / dt
                self.rep_mask[slot] = (fn > 0.0).astype(float)
                self.rep_normal[slot] = b_nrm
                self.rep_fn[slot] = fn
                self.rep_ft[slot] = ft_b
                self.rep_vrel[slot] = vrel_b
            if len(hi):
                fn = lam_h / dt
                order = np.argsort(-fn)[:HAIR_SLOTS]
                slot = self.n_body_slots + np.arange(len(order))
                self.rep_mask[slot] = (fn[order] > 0.0).astype(float)
                self.rep_normal[slot] = -h_nrm[order]
                self.rep_fn[slot] = fn[order]
                self.rep_ft[slot] = ft_h[order]
                self.rep_vrel[slot] = vrel_h[order]

        self.step_count += 1
        return terms

    def _probe_initial_terms(self) -> None:
        """Force terms and contact state at t=0 from a throw-away sub-step."""
        pos, vel, count = self.pos.copy(), self.vel.copy(), self.step_count
        terms = self._step(0.0, DT_SUB, record=True)
        self.pos, self.vel, self.step_count = pos, vel, count
        self.init_terms = terms

    # ---------------------------------------------------------- interface ---
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target <= self.time + 1e-15:
            terms = self.init_terms
        else:
            accum = np.zeros((self.n_terms, 3))
            steps = 0
            t = self.time
            n_sub = max(1, int(np.ceil((target - t) / DT_SUB - 1e-9)))
            dt = (target - t) / n_sub
            for k in range(n_sub):
                terms_k = self._step(t, dt, record=(k == n_sub - 1))
                accum += terms_k
                self.cum_impulse += dt * terms_k.sum(axis=0)
                t += dt
                steps += 1
            self.time = target
            terms = accum / max(steps, 1)

        weights = np.zeros((1, self.n))
        weights[0, self.dyn] = 1.0

        frame = StateFrame(
            time=target,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.strand_id.copy(),
            attributes={
                "strand_id": self.strand_id.copy(),
                "node_index": self.node_index.copy(),
                "is_root": self.is_root.astype(int),
                "kinematic_root": self.is_root.astype(int),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "budget_domain_weights": weights,
                "budget_external_force": terms.reshape(1, self.n_terms, 3).copy(),
                "budget_cumulative_external_impulse": np.zeros((1, 3)),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
                "cumulative_mass_source": np.zeros(1),
                "friction_active_mask": self.rep_mask.copy(),
                "friction_static_coefficient": self.mu_static.copy(),
                "friction_dynamic_coefficient": self.mu_dynamic.copy(),
                "friction_normal": self.rep_normal.copy(),
                "friction_normal_force": self.rep_fn.copy(),
                "friction_tangential_force": self.rep_ft.copy(),
                "friction_relative_velocity": self.rep_vrel.copy(),
                "gravity_acceleration": self.gravity.reshape(1, 3).copy(),
            },
            scalars={
                "friction_stick_speed_tolerance": STICK_SPEED_TOL,
                "reference_force": self.ref_force,
                "reference_momentum": self.ref_momentum,
                "reference_velocity": self.ref_velocity,
                "strand_radius": self.hair_radius,
                "static_friction": self.mu_hair_body,
                "kinetic_friction": self.mu_hair_body,
                "head_stop_time": self.t_stop,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit strand dynamics with projected Coulomb friction",
            "sub_step": DT_SUB,
            "nodes": int(self.n),
            "strands": int(self.n_strand),
            "contact_slots": int(self.n_slots),
            "max_speed": float(np.max(np.linalg.norm(self.vel, axis=1))),
        }
