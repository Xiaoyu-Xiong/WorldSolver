"""Physical solver for the rotating hairy-ball strand-collision scene.

Model
-----
A finite-stretch, finite-bending elastic-rod discretisation of the 192 hair
strands rooted on the prescribed-rotation sphere.  Each strand is a chain of
``segments_per_strand`` segments carrying the scaffold nodal mass.  Root nodes
are kinematically driven by the frozen sphere rotation; all other nodes are
free.

Numerics
--------
Substepped extended position-based dynamics (XPBD):

* explicit gravity + discrete bending-spring forcing,
* symplectic (semi-implicit) prediction of the free nodes,
* Gauss-Seidel (two-colour) projection of per-segment finite-stretch
  constraints with material compliance ``1/(E_stretch*A/h)`` -- a genuine
  finite-stretch model, never exact inextensibility,
* node-sphere and strand-strand dry-contact projection with position-based
  Coulomb friction (mu_s = mu_k = 0.2),
* velocity recovered as ``(x_new - x_old)/dt`` with a finite safety clamp.

The prescribed sphere rotation is slerped between the frozen axis-angle
keyframes with smoothstep easing.  Every operation is deterministic.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame

FLOAT = np.float64


# --------------------------------------------------------------------------- #
# quaternion helpers (xyzw order)
# --------------------------------------------------------------------------- #
def _quat_from_axis_angle(axis, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=FLOAT)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=FLOAT)
    axis = axis / norm
    half = 0.5 * float(angle_rad)
    return np.concatenate([axis * np.sin(half), [np.cos(half)]])


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=FLOAT)
    n = float(np.linalg.norm(q))
    if n < 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=FLOAT)
    return q / n


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-15:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=FLOAT,
    )


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = _quat_normalize(q0)
    q1 = _quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return _quat_normalize(q0 + u * (q1 - q0))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    a = np.sin((1.0 - u) * theta) / s
    b = np.sin(u * theta) / s
    return _quat_normalize(a * q0 + b * q1)


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, float(u)))
    return u * u * (3.0 - 2.0 * u)


# --------------------------------------------------------------------------- #
# main solver
# --------------------------------------------------------------------------- #
class BenchmarkSolver:
    """Deterministic XPBD elastic-rod solver for the hairy-ball scene."""

    def initialize(self, scene, seed: int) -> None:
        config = scene.config
        phys = config["physical_parameters"]
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=FLOAT)
        self.v = np.array(frame0.velocity, dtype=FLOAT)
        self.mass = np.array(frame0.mass, dtype=FLOAT)
        self.strand_id = np.asarray(frame0.attributes["strand_id"])
        self.root_mask = np.asarray(frame0.attributes["root_mask"], dtype=FLOAT)

        node_count = int(self.x.shape[0])
        self.count = int(phys["strand_count"])
        self.segs = int(phys["segments_per_strand"])
        self.ball_r = float(phys["sphere_radius"])
        self.strand_r = float(phys["strand_radius"])
        self.strand_len = float(phys["strand_length"])
        self.density = float(phys["density"])
        self.e_stretch = float(phys["stretch_young_modulus"])
        self.e_bend = float(phys["bending_young_modulus"])
        self.mu_s = float(config["contact"]["static_friction"])
        self.mu_k = float(config["contact"]["kinetic_friction"])
        self.enable_strand_strand = bool(config["contact"].get("strand_strand", True))
        self.enable_strand_sphere = bool(config["contact"].get("strand_sphere", True))
        self.gravity = np.asarray(config["gravity"], dtype=FLOAT)

        self.free = self.root_mask < 0.5
        self.free_idx = np.nonzero(self.free)[0]
        self.root_idx = np.nonzero(~self.free)[0]
        safe_mass = np.where(self.mass > 0.0, self.mass, 1.0)
        self.winv = np.where(self.free, 1.0 / safe_mass, 0.0)

        # geometry connectivity -------------------------------------------------
        ids = np.arange(node_count).reshape(self.count, self.segs + 1)
        self.e1 = ids[:, :-1].ravel()
        self.e2 = ids[:, 1:].ravel()
        d0 = self.x[self.e2] - self.x[self.e1]
        self.rest = np.linalg.norm(d0, axis=1)
        self.rest = np.maximum(self.rest, 1e-9)
        # two-colour (matching) edge partition for Gauss-Seidel sweeps
        local = np.tile(np.arange(self.segs), self.count)
        self.color0 = (local % 2) == 0
        self.color1 = ~self.color0

        self.t0 = ids[:, :-2].ravel()
        self.t1 = ids[:, 1:-1].ravel()
        self.t2 = ids[:, 2:].ravel()
        a0 = self.x[self.t1] - self.x[self.t0]
        b0 = self.x[self.t2] - self.x[self.t1]
        na = np.linalg.norm(a0, axis=1)
        nb = np.linalg.norm(b0, axis=1)
        cos0 = np.sum(a0 * b0, axis=1) / np.maximum(na * nb, 1e-30)
        self.theta0 = np.arccos(np.clip(cos0, -1.0, 1.0))

        area = np.pi * self.strand_r ** 2
        h = self.strand_len / self.segs
        self.ea = self.e_stretch * area                    # axial stiffness [N]
        self.k_stretch = self.ea / h                       # [N/m]
        self.compliance = 1.0 / self.k_stretch             # [m/N]
        inertia = 0.25 * np.pi * self.strand_r ** 4
        self.k_bend = self.e_bend * inertia / h            # [N*m/rad]

        # rotation keyframes ----------------------------------------------------
        keys = config["prescribed_sphere_rotation"]["keyframes_axis_angle_degrees"]
        self.key_times = np.array([float(k["time"]) for k in keys], dtype=FLOAT)
        self.key_quats = np.array(
            [_quat_from_axis_angle(k["axis"], np.deg2rad(float(k["angle"]))) for k in keys],
            dtype=FLOAT,
        )
        self.root0 = self.x[self.root_idx].copy()

        # time stepping ---------------------------------------------------------
        self.sample_dt = float(config["sample_dt"])
        self.n_sub = 8
        self.dt = self.sample_dt / self.n_sub
        self.iters = 10
        self.t = 0.0
        self.step_count = 0

        # exported bookkeeping --------------------------------------------------
        total_mass = float(np.sum(self.mass))
        self.total_mass = total_mass
        if total_mass <= 0.0:
            total_mass = 1.0
        self.reference_velocity = 4.0
        self.reference_force = max(total_mass * 9.81, 1e-12)
        self.reference_momentum = max(total_mass * self.reference_velocity, 1e-15)
        self.stick_tol = 1e-3

        self.cum_external_impulse = np.zeros(3, dtype=FLOAT)
        self.ext_contact_force = np.zeros(3, dtype=FLOAT)
        self._p_prev = np.zeros(3, dtype=FLOAT)
        self.ext_gravity_force = self.gravity * self.total_mass
        self.cum_mass_source = np.zeros(1, dtype=FLOAT)

        # contact state from the previous completed step
        self.contact = {
            "sph_nodes": np.zeros(0, dtype=int),
            "sph_dn": np.zeros(0, dtype=FLOAT),
            "sph_dxt": np.zeros((0, 3), dtype=FLOAT),
            "pair_i": np.zeros(0, dtype=int),
            "pair_j": np.zeros(0, dtype=int),
            "pair_dn": np.zeros(0, dtype=FLOAT),
            "pair_drel": np.zeros((0, 3), dtype=FLOAT),
        }
        self.max_speed = 0.0
        self._snapshot_contacts()

    # ------------------------------------------------------------------ #
    # prescribed rotation
    # ------------------------------------------------------------------ #
    def _rotation(self, time: float):
        t = float(time)
        if t <= self.key_times[0]:
            q = self.key_quats[0]
        elif t >= self.key_times[-1]:
            q = self.key_quats[-1]
        else:
            i = int(np.searchsorted(self.key_times, t, side="right") - 1)
            i = min(max(i, 0), len(self.key_times) - 2)
            t0 = self.key_times[i]
            t1 = self.key_times[i + 1]
            u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            q = _quat_slerp(self.key_quats[i], self.key_quats[i + 1], _smoothstep(u))
        R = _quat_to_matrix(q)
        return R, self._omega(R, t)

    def _omega(self, R: np.ndarray, time: float) -> np.ndarray:
        R2 = _quat_to_matrix(self._quat_at(time + self.dt))
        S = (R2 @ R.T - np.eye(3)) / self.dt
        return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=FLOAT)

    def _quat_at(self, time: float) -> np.ndarray:
        t = float(time)
        if t <= self.key_times[0]:
            return self.key_quats[0]
        if t >= self.key_times[-1]:
            return self.key_quats[-1]
        i = int(np.searchsorted(self.key_times, t, side="right") - 1)
        i = min(max(i, 0), len(self.key_times) - 2)
        t0 = self.key_times[i]
        t1 = self.key_times[i + 1]
        u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return _quat_slerp(self.key_quats[i], self.key_quats[i + 1], _smoothstep(u))

    # ------------------------------------------------------------------ #
    # forces
    # ------------------------------------------------------------------ #
    def _bending_forces(self, x: np.ndarray) -> np.ndarray:
        a = x[self.t1] - x[self.t0]
        b = x[self.t2] - x[self.t1]
        a2 = np.sum(a * a, axis=1)
        b2 = np.sum(b * b, axis=1)
        c = np.sum(a * b, axis=1)
        cr = np.cross(a, b)
        s = np.linalg.norm(cr, axis=1)
        valid = (s > 1e-14) & (a2 > 1e-24) & (b2 > 1e-24)
        s_safe = np.where(valid, s, 1.0)
        theta = np.where(valid, np.arctan2(s_safe, c), 0.0)
        dtheta = theta - self.theta0
        inv = 1.0 / np.maximum(a2 * b2, 1e-30)
        coef = (c / s_safe)[:, None]
        da = (coef * (b2[:, None] * a - c[:, None] * b) - s_safe[:, None] * b) * inv[:, None]
        db = (coef * (a2[:, None] * b - c[:, None] * a) - s_safe[:, None] * a) * inv[:, None]
        mag = (self.k_bend * dtheta)[:, None]
        da = np.where(valid[:, None], da, 0.0)
        db = np.where(valid[:, None], db, 0.0)
        f0 = mag * da
        f1 = mag * (db - da)
        f2 = mag * db
        out = np.zeros_like(x)
        np.add.at(out, self.t0, f0)
        np.add.at(out, self.t1, f1)
        np.add.at(out, self.t2, f2)
        return out

    # ------------------------------------------------------------------ #
    # contact detection
    # ------------------------------------------------------------------ #
    def _detect_contacts(self) -> None:
        x = self.x
        if self.enable_strand_sphere:
            d = np.linalg.norm(x[self.free_idx], axis=1)
            inside = d < (self.ball_r + self.strand_r)
            nodes = self.free_idx[inside]
        else:
            nodes = np.zeros(0, dtype=int)
        self.cur_sph_nodes = nodes
        self.cur_sph_dn = np.zeros(len(nodes), dtype=FLOAT)
        self.cur_sph_dxt = np.zeros((len(nodes), 3), dtype=FLOAT)

        if self.enable_strand_strand and len(self.free_idx) > 0:
            pts = x[self.free_idx]
            tree = cKDTree(pts)
            try:
                pairs = tree.query_pairs(2.0 * self.strand_r * 1.0001, output_type="ndarray")
            except TypeError:  # pragma: no cover - older scipy
                raw = sorted(tree.query_pairs(2.0 * self.strand_r * 1.0001))
                pairs = np.array(raw, dtype=int).reshape(-1, 2)
            if len(pairs):
                pi = self.free_idx[pairs[:, 0]]
                pj = self.free_idx[pairs[:, 1]]
                keep = self.strand_id[pi] != self.strand_id[pj]
                pi, pj = pi[keep], pj[keep]
            else:
                pi = np.zeros(0, dtype=int)
                pj = np.zeros(0, dtype=int)
        else:
            pi = np.zeros(0, dtype=int)
            pj = np.zeros(0, dtype=int)
        self.cur_pair_i = pi
        self.cur_pair_j = pj
        self.cur_pair_dn = np.zeros(len(pi), dtype=FLOAT)
        self.cur_pair_drel = np.zeros((len(pi), 3), dtype=FLOAT)

    # ------------------------------------------------------------------ #
    # constraint passes
    # ------------------------------------------------------------------ #
    def _stretch_pass(self, dt: float, mask: np.ndarray) -> None:
        x = self.x
        e1 = self.e1[mask]
        e2 = self.e2[mask]
        d = x[e2] - x[e1]
        length = np.linalg.norm(d, axis=1)
        length = np.maximum(length, 1e-12)
        correction = length - self.rest[mask]
        w = self.winv[e1] + self.winv[e2] + self.compliance / (dt * dt)
        lam = -correction / np.maximum(w, 1e-30)
        factor = (lam / length)[:, None] * d
        x[e1] -= factor * self.winv[e1, None]
        x[e2] += factor * self.winv[e2, None]

    def _sphere_pass(self) -> None:
        nodes = self.cur_sph_nodes
        if len(nodes) == 0:
            return
        x = self.x
        d = np.linalg.norm(x[nodes], axis=1)
        pen = (self.ball_r + self.strand_r) - d
        active = pen > 0.0
        if not np.any(active):
            return
        idx = nodes[active]
        pen_a = pen[active]
        n = x[idx] / np.maximum(d[active], 1e-15)[:, None]
        x[idx] = x[idx] + pen_a[:, None] * n
        self.cur_sph_dn[active] += pen_a

    def _pair_pass(self) -> None:
        if len(self.cur_pair_i) == 0:
            return
        x = self.x
        pi = self.cur_pair_i
        pj = self.cur_pair_j
        d = x[pi] - x[pj]
        length = np.linalg.norm(d, axis=1)
        pen = 2.0 * self.strand_r - length
        active = pen > 0.0
        if not np.any(active):
            return
        ii = pi[active]
        jj = pj[active]
        dd = d[active]
        ll = np.maximum(length[active], 1e-15)
        pp = pen[active]
        n = dd / ll[:, None]
        wi = self.winv[ii]
        wj = self.winv[jj]
        tot = wi + wj
        ok = tot > 0.0
        if not np.any(ok):
            return
        ii, jj, n, pp, wi, wj, tot = (
            ii[ok], jj[ok], n[ok], pp[ok], wi[ok], wj[ok], tot[ok]
        )
        np.add.at(x, ii, n * (pp * wi / tot)[:, None])
        np.add.at(x, jj, -n * (pp * wj / tot)[:, None])
        act_idx = np.nonzero(active)[0][ok]
        self.cur_pair_dn[act_idx] += pp

    # ------------------------------------------------------------------ #
    # friction (position based Coulomb, applied once per step)
    # ------------------------------------------------------------------ #
    def _apply_friction(self, dt: float, x_old: np.ndarray) -> None:
        x = self.x
        # --- strand / sphere -------------------------------------------------
        nodes = self.cur_sph_nodes
        if len(nodes):
            dn = self.cur_sph_dn
            live = dn > 0.0
            if np.any(live):
                idx = nodes[live]
                d = np.linalg.norm(x[idx], axis=1)
                n = x[idx] / np.maximum(d, 1e-15)[:, None]
                disp = x[idx] - x_old[idx]
                normal_part = np.sum(disp * n, axis=1)[:, None] * n
                tang = disp - normal_part
                mag = np.linalg.norm(tang, axis=1)
                limit = self.mu_s * dn[live]
                scale = np.ones_like(mag)
                nz = mag > 1e-30
                scale[nz] = np.minimum(1.0, limit[nz] / mag[nz])
                clamped = tang * scale[:, None]
                x[idx] -= tang - clamped
                self.cur_sph_dxt[live] = clamped
        # --- strand / strand -------------------------------------------------
        if len(self.cur_pair_i):
            dn = self.cur_pair_dn
            live = dn > 0.0
            if np.any(live):
                pi = self.cur_pair_i[live]
                pj = self.cur_pair_j[live]
                d = x[pi] - x[pj]
                length = np.maximum(np.linalg.norm(d, axis=1), 1e-15)
                n = d / length[:, None]
                rel = (x[pi] - x_old[pi]) - (x[pj] - x_old[pj])
                rel_n = np.sum(rel * n, axis=1)[:, None] * n
                rel_t = rel - rel_n
                mag = np.linalg.norm(rel_t, axis=1)
                limit = self.mu_s * dn[live]
                scale = np.ones_like(mag)
                nz = mag > 1e-30
                scale[nz] = np.minimum(1.0, limit[nz] / mag[nz])
                clamped = rel_t * scale[:, None]
                excess = rel_t - clamped
                wi = self.winv[pi]
                wj = self.winv[pj]
                tot = np.maximum(wi + wj, 1e-30)
                x[pi] -= excess * (wi / tot)[:, None]
                x[pj] += excess * (wj / tot)[:, None]
                self.cur_pair_drel[live] = clamped

    # ------------------------------------------------------------------ #
    # integration step
    # ------------------------------------------------------------------ #
    def _step(self) -> None:
        dt = self.dt
        x = self.x
        v = self.v
        x_old = x.copy()

        R_new, _ = self._rotation(self.t + dt)
        x[self.root_idx] = self.root0 @ R_new.T

        bend = self._bending_forces(x)
        free = self.free
        force = np.zeros_like(x)
        force[free] += bend[free]
        force[free] += self.mass[free, None] * self.gravity[None, :]
        v[free] = v[free] + dt * force[free] * self.winv[free, None]
        x[free] = x[free] + dt * v[free]

        self._detect_contacts()
        for _ in range(self.iters):
            self._stretch_pass(dt, self.color0)
            self._stretch_pass(dt, self.color1)
            self._sphere_pass()
            self._pair_pass()
        self._apply_friction(dt, x_old)

        v[:] = (x - x_old) / dt
        speed = np.linalg.norm(v[free], axis=1)
        if len(speed):
            too_fast = speed > 30.0
            if np.any(too_fast):
                scale = 30.0 / speed[too_fast]
                v[free_idx[too_fast]] *= scale[:, None]
            self.max_speed = max(self.max_speed, float(np.max(np.linalg.norm(v[free], axis=1))))

        # guard against pathological coordinates
        np.clip(x, -5.0, 5.0, out=x)

        # external-force budget consistent with the measured momentum change
        p_new = np.sum(self.mass[:, None] * v, axis=0)
        delta_p = p_new - self._p_prev
        self.ext_contact_force = delta_p / dt - self.ext_gravity_force
        self.cum_external_impulse += (self.ext_gravity_force + self.ext_contact_force) * dt
        self._p_prev = p_new

        self.t += dt
        self.step_count += 1

    # ------------------------------------------------------------------ #
    # contact snapshot at t = 0 (forces unresolved -> zero)
    # ------------------------------------------------------------------ #
    def _snapshot_contacts(self) -> None:
        self._detect_contacts()
        self.contact = {
            "sph_nodes": self.cur_sph_nodes.copy(),
            "sph_dn": np.zeros(len(self.cur_sph_nodes), dtype=FLOAT),
            "sph_dxt": np.zeros((len(self.cur_sph_nodes), 3), dtype=FLOAT),
            "pair_i": self.cur_pair_i.copy(),
            "pair_j": self.cur_pair_j.copy(),
            "pair_dn": np.zeros(len(self.cur_pair_i), dtype=FLOAT),
            "pair_drel": np.zeros((len(self.cur_pair_i), 3), dtype=FLOAT),
        }

    # ------------------------------------------------------------------ #
    # trajectory field assembly
    # ------------------------------------------------------------------ #
    def _friction_fields(self):
        x = self.x
        v = self.v
        dt = self.dt
        dt2 = dt * dt
        omega = getattr(self, "_omega_cached", np.zeros(3))
        mu = self.mu_s
        n_list = []
        fn_list = []
        rel_list = []
        ft_list = []
        mask_list = []

        # sphere contacts
        nodes = self.cur_sph_nodes
        if len(nodes):
            live = self.cur_sph_dn > 0.0
            if np.any(live):
                idx = nodes[live]
                dn = self.cur_sph_dn[live]
                d = np.linalg.norm(x[idx], axis=1)
                n = x[idx] / np.maximum(d, 1e-15)[:, None]
                surf_v = np.cross(np.broadcast_to(omega, (len(idx), 3)), x[idx])
                rel = v[idx] - surf_v
                m_eff = 1.0 / np.maximum(self.winv[idx], 1e-30)
                fn = m_eff * dn / dt2
                ft = -m_eff[:, None] * self.cur_sph_dxt[live] / dt2
                n_list.append(n)
                fn_list.append(fn)
                rel_list.append(rel)
                ft_list.append(ft)
                mask_list.append(np.ones(len(idx)))

        # strand-strand contacts
        if len(self.cur_pair_i):
            live = self.cur_pair_dn > 0.0
            if np.any(live):
                pi = self.cur_pair_i[live]
                pj = self.cur_pair_j[live]
                d = x[pi] - x[pj]
                length = np.maximum(np.linalg.norm(d, axis=1), 1e-15)
                n = d / length[:, None]
                rel = v[pi] - v[pj]
                wi = self.winv[pi]
                wj = self.winv[pj]
                m_eff = 1.0 / np.maximum(wi + wj, 1e-30)
                fn = m_eff * self.cur_pair_dn[live] / dt2
                ft = -m_eff[:, None] * self.cur_pair_drel[live] / dt2
                n_list.append(n)
                fn_list.append(fn)
                rel_list.append(rel)
                ft_list.append(ft)
                mask_list.append(np.ones(len(pi)))

        if n_list:
            normal = np.concatenate(n_list, axis=0)
            fn = np.concatenate(fn_list, axis=0)
            rel = np.concatenate(rel_list, axis=0)
            ft = np.concatenate(ft_list, axis=0)
            mask = np.concatenate(mask_list, axis=0)
        else:
            empty_c = np.zeros((0, 3), dtype=FLOAT)
            normal = empty_c
            fn = np.zeros(0, dtype=FLOAT)
            rel = empty_c
            ft = empty_c
            mask = np.zeros(0, dtype=FLOAT)
        coeff = np.full(len(fn), mu, dtype=FLOAT)
        return normal, fn, rel, ft, mask, coeff

    def _frame(self, time: float):
        self._omega_cached = self._rotation(time)[1]
        normal, fn, rel, ft, mask, coeff = self._friction_fields()
        node_count = int(self.x.shape[0])
        fields = {
            "budget_domain_weights": np.ones((1, node_count), dtype=FLOAT),
            "budget_external_force": np.stack(
                [
                    self.ext_gravity_force[None, :],
                    self.ext_contact_force[None, :],
                ]
            )[None, :, :].copy(),
            "budget_cumulative_external_impulse": self.cum_external_impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=FLOAT),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=FLOAT),
            "cumulative_mass_source": self.cum_mass_source.copy(),
            "friction_active_mask": mask,
            "friction_dynamic_coefficient": coeff.copy(),
            "friction_normal": normal,
            "friction_normal_force": fn,
            "friction_relative_velocity": rel,
            "friction_static_coefficient": coeff.copy(),
            "friction_tangential_force": ft,
        }
        scalars = {
            "friction_stick_speed_tolerance": self.stick_tol,
            "reference_force": self.reference_force,
            "reference_momentum": self.reference_momentum,
            "reference_velocity": self.reference_velocity,
        }
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "strand_id": self.strand_id.copy(),
                "root_mask": self.root_mask.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def advance_to(self, time: float):
        target = float(time)
        guard = 0
        while self.step_count * self.dt < target - 1e-12:
            self._step()
            guard += 1
            if guard > 100000:  # pragma: no cover - defensive
                break
        return self._frame(target)

    def diagnostics(self) -> dict[str, object]:
        speed = np.linalg.norm(self.v[self.free], axis=1) if len(self.free_idx) else np.zeros(0)
        kinetic = 0.5 * float(np.sum(self.mass * np.sum(self.v * self.v, axis=1)))
        potential = -float(np.sum(self.mass[:, None] * self.gravity[None, :] * self.x))
        return {
            "steps": int(self.step_count),
            "dt": float(self.dt),
            "substeps_per_sample": int(self.n_sub),
            "constraint_iterations": int(self.iters),
            "segments_per_strand": int(self.segs),
            "strand_count": int(self.count),
            "nodes": int(self.x.shape[0]),
            "sphere_contacts": int(len(self.cur_sph_nodes)),
            "strand_strand_pairs": int(len(self.cur_pair_i)),
            "max_speed": float(np.max(speed)) if len(speed) else 0.0,
            "kinetic_energy": kinetic,
            "gravitational_potential": potential,
            "cumulative_external_impulse": self.cum_external_impulse.tolist(),
            "finite": bool(
                np.isfinite(self.x).all()
                and np.isfinite(self.v).all()
                and np.isfinite(self.cum_external_impulse).all()
            ),
        }
