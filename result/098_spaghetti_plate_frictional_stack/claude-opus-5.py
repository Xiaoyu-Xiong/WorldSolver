"""Frictional stacking of flexible spaghetti strands in a rigid plate.

Physical model
--------------
Each strand is a discrete elastic rod discretised as a mass chain:

* axial (stretch) springs with rest length from the frozen initial state,
* a linear bending energy ``0.5 * kb * |x_{i-1} - 2 x_i + x_{i+1}|^2`` with
  ``kb = EI / l0^3`` and a straight rest shape, giving real bending resistance
  while keeping every internal force an exact equal-and-opposite set,
* light internal Rayleigh damping on both modes (internal, momentum conserving).

Contacts use a compliant (penalty) normal law plus an implicit regularised
Coulomb friction law that is unconditionally stable and exactly consistent with
the reported contact fields:

    f_t = -mu * f_n * v_t / max(|v_t|, v_eps)

so sliding contacts (|v_t| > v_eps) carry exactly ``mu * f_n`` opposite to the
tangential relative velocity while sticking contacts satisfy ``|f_t| <= mu*f_n``.

Strand/strand contact uses capsule-capsule (segment-segment) proximity so no
strand tunnels between the widely spaced nodes; strand/plate contact uses the
analytic signed distance to the frozen axisymmetric container profile.

Integration is symplectic (semi-implicit) Euler with a fixed sub-step that
divides the sample interval exactly, so states are deterministic and finite.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)

# Frozen plate inner profile (r, z), extended by a flat floor inside r = 0.1 and
# by a steep containing wall outside the rim.
_PROFILE_R = np.array([0.0, 0.10, 0.40, 0.52, 0.57, 3.0], dtype=float)
_PROFILE_Z = np.array(
    [0.0, 0.0, 0.018, 0.075, 0.125, 0.125 + (3.0 - 0.57) * 4.0], dtype=float
)
_PROFILE_SLOPE = np.append(np.diff(_PROFILE_Z) / np.diff(_PROFILE_R), 0.0)


def _scatter3(count: int, index: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Accumulate ``values`` [M,3] into a [count,3] array at ``index``."""
    out = np.empty((count, 3), dtype=float)
    for k in range(3):
        out[:, k] = np.bincount(index, weights=values[:, k], minlength=count)
    return out


class BenchmarkSolver:
    """Self-contained discrete-elastic-rod solver with Coulomb friction."""

    # --- material / numerical parameters -------------------------------------
    STRETCH_STIFFNESS = 600.0      # N/m per axial spring
    STRETCH_DAMPING = 0.15         # N*s/m
    BENDING_EI = 0.012             # N*m^2 (flexible pasta strand)
    BENDING_DAMPING = 0.010        # N*s/m on the curvature rate
    CONTACT_STIFFNESS = 2500.0     # N/m penalty normal stiffness
    CONTACT_DAMPING = 0.55         # N*s/m normal dissipation
    FRICTION_VEPS = 2.0e-3         # m/s Coulomb regularisation / stick tolerance
    SUBSTEPS_PER_SAMPLE = 160
    BROADPHASE_INTERVAL = 8
    BROADPHASE_MARGIN = 0.02
    REPORT_MARGIN = 0.004          # m proximity margin for reported contacts

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.seed = int(seed)
        frame = scene.initial_state()
        cfg = getattr(scene, "config", {}) or {}

        self.position = np.array(frame.position, dtype=float)
        self.velocity = np.array(frame.velocity, dtype=float)
        self.mass = np.array(frame.mass, dtype=float)
        self.body_id = np.array(frame.body_id, dtype=int)
        self.n_nodes = self.position.shape[0]
        self.inv_mass = 1.0 / self.mass

        attrs = frame.attributes or {}
        self.node_index = np.array(
            attrs.get("node_index", np.zeros(self.n_nodes, dtype=int)), dtype=int
        )
        self.is_endpoint = np.array(
            attrs.get("is_endpoint", np.zeros(self.n_nodes, dtype=int)), dtype=int
        )

        self.n_strands = int(self.body_id.max()) + 1
        self.nodes_per_strand = self.n_nodes // self.n_strands
        self.radius = float((frame.scalars or {}).get("strand_radius", 0.006))

        contact_cfg = cfg.get("contact", {}) if isinstance(cfg, dict) else {}
        ss = contact_cfg.get("strand_strand", {})
        sp = contact_cfg.get("strand_plate", {})
        self.mu_ss_static = float(ss.get("static_friction", 0.3))
        self.mu_ss_dynamic = float(ss.get("kinetic_friction", 0.3))
        self.mu_sp_static = float(sp.get("static_friction", 0.3))
        self.mu_sp_dynamic = float(sp.get("kinetic_friction", 0.3))

        gvec = cfg.get("gravity") if isinstance(cfg, dict) else None
        self.gravity = np.array(gvec, dtype=float) if gvec is not None else GRAVITY.copy()

        sample_dt = float(cfg.get("sample_dt", 0.02)) if isinstance(cfg, dict) else 0.02
        self.sample_dt = sample_dt
        self.dt = sample_dt / float(self.SUBSTEPS_PER_SAMPLE)

        self._build_topology()
        self._build_domains()

        # accumulated (contact) impulses per node, split by interaction class
        self.imp_plate_n = np.zeros((self.n_nodes, 3), dtype=float)
        self.imp_plate_f = np.zeros((self.n_nodes, 3), dtype=float)
        self.imp_inter = np.zeros((self.n_nodes, 3), dtype=float)
        self.sim_time = 0.0
        self.step_count = 0
        self._pairs = None
        self._pair_age = 10 ** 9
        self._max_speed = 0.0
        self._min_gap = 0.0

        vfree = float(np.sqrt(max(2.0 * 9.81 * float(self.position[:, 2].max()), 1e-6)))
        self.reference_velocity = vfree
        total_mass = float(self.mass.sum())
        self.reference_force = total_mass * float(np.linalg.norm(self.gravity))
        self.reference_momentum = total_mass * vfree

    def _build_topology(self) -> None:
        n_s = self.n_strands
        n_p = self.nodes_per_strand
        base = (np.arange(n_s) * n_p)[:, None]
        seg_i = (base + np.arange(n_p - 1)[None, :]).ravel()
        self.seg_i = seg_i
        self.seg_j = seg_i + 1
        d = self.position[self.seg_j] - self.position[self.seg_i]
        self.rest_length = np.linalg.norm(d, axis=1)
        self.rest_length = np.maximum(self.rest_length, 1e-9)
        l0 = float(np.mean(self.rest_length))
        self.segment_length = l0
        self.bend_stiffness = self.BENDING_EI / (l0 ** 3)

        bend_a = (base + np.arange(n_p - 2)[None, :]).ravel()
        self.bend_a = bend_a
        self.bend_b = bend_a + 1
        self.bend_c = bend_a + 2

        # segment -> owning strand, for excluding self-contacts
        self.seg_strand = self.body_id[self.seg_i]
        self.n_segments = self.seg_i.size
        self.seg_bound = 0.5 * (l0 + 2.0 * self.radius)

    def _build_domains(self) -> None:
        g = self.n_strands + 1
        w = np.zeros((g, self.n_nodes), dtype=float)
        w[0, :] = 1.0
        for s in range(self.n_strands):
            w[s + 1, self.body_id == s] = 1.0
        self.domain_weights = w
        self.n_domains = g
        self.domain_mass = w @ self.mass
        self.gravity_force = self.domain_mass[:, None] * self.gravity[None, :]

    # ---------------------------------------------------------------- geometry
    def _plate_surface(self, r: np.ndarray):
        """Height and dz/dr of the container inner surface at radius ``r``."""
        idx = np.clip(np.searchsorted(_PROFILE_R, r, side="right") - 1, 0,
                      _PROFILE_R.size - 2)
        slope = _PROFILE_SLOPE[idx]
        z = _PROFILE_Z[idx] + slope * (r - _PROFILE_R[idx])
        return z, slope

    def _plate_contacts(self, position: np.ndarray):
        """Return (nodes, depth, normal) for nodes penetrating the plate."""
        r = np.sqrt(position[:, 0] ** 2 + position[:, 1] ** 2)
        zs, slope = self._plate_surface(r)
        inv = 1.0 / np.sqrt(1.0 + slope * slope)
        # outward (into the free interior) unit normal of the surface
        signed = (position[:, 2] - zs) * inv
        gap = signed - self.radius
        self._min_gap = float(gap.min()) if gap.size else 0.0
        hit = np.nonzero(gap < 0.0)[0]
        if hit.size == 0:
            return hit, np.zeros(0), np.zeros((0, 3))
        rs = np.maximum(r[hit], 1e-12)
        nx = -slope[hit] * position[hit, 0] / rs * inv[hit]
        ny = -slope[hit] * position[hit, 1] / rs * inv[hit]
        nz = inv[hit]
        normal = np.stack([nx, ny, nz], axis=1)
        return hit, -gap[hit], normal

    def _broadphase(self, position: np.ndarray) -> None:
        mid = 0.5 * (position[self.seg_i] + position[self.seg_j])
        thresh = 2.0 * self.seg_bound + self.BROADPHASE_MARGIN
        diff = mid[:, None, :] - mid[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        close = d2 < thresh * thresh
        np.fill_diagonal(close, False)
        close &= self.seg_strand[:, None] != self.seg_strand[None, :]
        close &= np.arange(self.n_segments)[:, None] < np.arange(self.n_segments)[None, :]
        pa, pb = np.nonzero(close)
        self._pairs = (pa, pb)
        self._pair_age = 0

    def _segment_contacts(self, position: np.ndarray):
        """Capsule-capsule contacts between segments of different strands."""
        if self._pairs is None:
            self._broadphase(position)
        pa, pb = self._pairs
        if pa.size == 0:
            z0 = np.zeros(0, dtype=int)
            return z0, z0, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 3))
        ia, ja = self.seg_i[pa], self.seg_j[pa]
        ib, jb = self.seg_i[pb], self.seg_j[pb]
        p1 = position[ia]
        d1 = position[ja] - p1
        p2 = position[ib]
        d2 = position[jb] - p2
        r12 = p1 - p2
        a = np.einsum("ij,ij->i", d1, d1)
        e = np.einsum("ij,ij->i", d2, d2)
        f = np.einsum("ij,ij->i", d2, r12)
        c = np.einsum("ij,ij->i", d1, r12)
        b = np.einsum("ij,ij->i", d1, d2)
        denom = np.maximum(a * e - b * b, 1e-18)
        s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
        t = (b * s + f) / np.maximum(e, 1e-18)
        t_c = np.clip(t, 0.0, 1.0)
        s = np.clip((b * t_c - c) / np.maximum(a, 1e-18), 0.0, 1.0)
        t = t_c
        ca = p1 + s[:, None] * d1
        cb = p2 + t[:, None] * d2
        delta = ca - cb
        dist = np.linalg.norm(delta, axis=1)
        gap = dist - 2.0 * self.radius
        keep = np.nonzero(gap < 0.0)[0]
        if keep.size == 0:
            z0 = np.zeros(0, dtype=int)
            return z0, z0, np.zeros(0), np.zeros(0), np.zeros(0), np.zeros((0, 3))
        k = keep
        dk = np.maximum(dist[k], 1e-12)
        normal = delta[k] / dk[:, None]
        degenerate = dist[k] < 1e-9
        if np.any(degenerate):
            normal[degenerate] = np.array([0.0, 0.0, 1.0])
        return pa[k], pb[k], s[k], t[k], -gap[k], normal

    # ------------------------------------------------------------------ solve
    def _internal_forces(self, position: np.ndarray, velocity: np.ndarray):
        force = np.zeros((self.n_nodes, 3), dtype=float)
        # axial springs
        d = position[self.seg_j] - position[self.seg_i]
        length = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        dirn = d / length[:, None]
        dv = velocity[self.seg_j] - velocity[self.seg_i]
        rate = np.einsum("ij,ij->i", dv, dirn)
        mag = self.STRETCH_STIFFNESS * (length - self.rest_length) \
            + self.STRETCH_DAMPING * rate
        f = mag[:, None] * dirn
        force += _scatter3(self.n_nodes, self.seg_i, f)
        force -= _scatter3(self.n_nodes, self.seg_j, f)
        # bending: E = 0.5 * kb * |lap|^2 with a straight rest shape
        lap = position[self.bend_a] - 2.0 * position[self.bend_b] + position[self.bend_c]
        lapv = velocity[self.bend_a] - 2.0 * velocity[self.bend_b] + velocity[self.bend_c]
        gen = -(self.bend_stiffness * lap + self.BENDING_DAMPING * lapv)
        force += _scatter3(self.n_nodes, self.bend_a, gen)
        force += _scatter3(self.n_nodes, self.bend_c, gen)
        force -= 2.0 * _scatter3(self.n_nodes, self.bend_b, gen)
        return force

    def _step(self) -> None:
        dt = self.dt
        x = self.position
        v = self.velocity

        if self._pair_age >= self.BROADPHASE_INTERVAL:
            self._broadphase(x)
        else:
            self._pair_age += 1

        pl_nodes, pl_depth, pl_normal = self._plate_contacts(x)
        sa, sb, ss_s, ss_t, ss_depth, ss_normal = self._segment_contacts(x)

        # contact multiplicity per node (used to keep friction impulses safe)
        counts = np.zeros(self.n_nodes, dtype=float)
        if pl_nodes.size:
            counts += np.bincount(pl_nodes, minlength=self.n_nodes).astype(float)
        if sa.size:
            for idx in (self.seg_i[sa], self.seg_j[sa], self.seg_i[sb], self.seg_j[sb]):
                counts += np.bincount(idx, minlength=self.n_nodes).astype(float)
        counts = np.maximum(counts, 1.0)

        force = self._internal_forces(x, v)
        force += self.mass[:, None] * self.gravity[None, :]

        # --- normal contact forces (penalty + dissipation) -------------------
        fn_plate = np.zeros(0)
        if pl_nodes.size:
            vrel_n = np.einsum("ij,ij->i", v[pl_nodes], pl_normal)
            fn = self.CONTACT_STIFFNESS * pl_depth - self.CONTACT_DAMPING * vrel_n
            fn = np.maximum(fn, 0.0)
            fn_plate = fn
            fvec = fn[:, None] * pl_normal
            add = _scatter3(self.n_nodes, pl_nodes, fvec)
            force += add
            self.imp_plate_n += add * dt

        if sa.size:
            wa0 = 1.0 - ss_s
            wa1 = ss_s
            wb0 = 1.0 - ss_t
            wb1 = ss_t
            va = v[self.seg_i[sa]] * wa0[:, None] + v[self.seg_j[sa]] * wa1[:, None]
            vb = v[self.seg_i[sb]] * wb0[:, None] + v[self.seg_j[sb]] * wb1[:, None]
            vrel = va - vb
            vrel_n = np.einsum("ij,ij->i", vrel, ss_normal)
            fn = self.CONTACT_STIFFNESS * ss_depth - self.CONTACT_DAMPING * vrel_n
            fn = np.maximum(fn, 0.0)
            fvec = fn[:, None] * ss_normal
            add = (
                _scatter3(self.n_nodes, self.seg_i[sa], fvec * wa0[:, None])
                + _scatter3(self.n_nodes, self.seg_j[sa], fvec * wa1[:, None])
                - _scatter3(self.n_nodes, self.seg_i[sb], fvec * wb0[:, None])
                - _scatter3(self.n_nodes, self.seg_j[sb], fvec * wb1[:, None])
            )
            force += add
            self.imp_inter += add * dt
            fn_seg = fn
        else:
            fn_seg = np.zeros(0)

        # --- explicit velocity predictor -------------------------------------
        v_half = v + dt * force * self.inv_mass[:, None]

        # --- implicit regularised Coulomb friction ---------------------------
        if pl_nodes.size:
            vt = v_half[pl_nodes] - np.einsum(
                "ij,ij->i", v_half[pl_nodes], pl_normal
            )[:, None] * pl_normal
            speed = np.linalg.norm(vt, axis=1)
            m_eff = self.mass[pl_nodes] / counts[pl_nodes]
            dv_max = self.mu_sp_dynamic * fn_plate * dt / np.maximum(m_eff, 1e-12)
            new_speed = np.where(
                speed - dv_max > self.FRICTION_VEPS,
                speed - dv_max,
                speed * self.FRICTION_VEPS
                / np.maximum(self.FRICTION_VEPS + dv_max, 1e-12),
            )
            imp = (m_eff * (speed - new_speed))[:, None] * (
                -vt / np.maximum(speed, 1e-12)[:, None]
            )
            add = _scatter3(self.n_nodes, pl_nodes, imp)
            v_half += add * self.inv_mass[:, None]
            self.imp_plate_f += add

        if sa.size:
            wa0 = 1.0 - ss_s
            wa1 = ss_s
            wb0 = 1.0 - ss_t
            wb1 = ss_t
            va = v_half[self.seg_i[sa]] * wa0[:, None] + v_half[self.seg_j[sa]] * wa1[:, None]
            vb = v_half[self.seg_i[sb]] * wb0[:, None] + v_half[self.seg_j[sb]] * wb1[:, None]
            vrel = va - vb
            vt = vrel - np.einsum("ij,ij->i", vrel, ss_normal)[:, None] * ss_normal
            speed = np.linalg.norm(vt, axis=1)
            mi = np.minimum(
                np.minimum(self.mass[self.seg_i[sa]] / counts[self.seg_i[sa]],
                           self.mass[self.seg_j[sa]] / counts[self.seg_j[sa]]),
                np.minimum(self.mass[self.seg_i[sb]] / counts[self.seg_i[sb]],
                           self.mass[self.seg_j[sb]] / counts[self.seg_j[sb]]),
            )
            m_eff = 0.5 * mi
            dv_max = self.mu_ss_dynamic * fn_seg * dt / np.maximum(m_eff, 1e-12)
            new_speed = np.where(
                speed - dv_max > self.FRICTION_VEPS,
                speed - dv_max,
                speed * self.FRICTION_VEPS
                / np.maximum(self.FRICTION_VEPS + dv_max, 1e-12),
            )
            imp = (m_eff * (speed - new_speed))[:, None] * (
                -vt / np.maximum(speed, 1e-12)[:, None]
            )
            add = (
                _scatter3(self.n_nodes, self.seg_i[sa], imp * wa0[:, None])
                + _scatter3(self.n_nodes, self.seg_j[sa], imp * wa1[:, None])
                - _scatter3(self.n_nodes, self.seg_i[sb], imp * wb0[:, None])
                - _scatter3(self.n_nodes, self.seg_j[sb], imp * wb1[:, None])
            )
            v_half += add * self.inv_mass[:, None]
            self.imp_inter += add

        self.velocity = v_half
        self.position = x + dt * v_half
        self.step_count += 1
        self.sim_time = self.step_count * self.dt
        self._max_speed = max(
            self._max_speed, float(np.abs(self.velocity).max()) if self.n_nodes else 0.0
        )

    # ----------------------------------------------------------- reporting
    def _report_contacts(self):
        """Recompute contacts and Coulomb forces from the current state."""
        x = self.position
        v = self.velocity
        rad = self.radius
        margin = self.REPORT_MARGIN

        # plate proximity
        r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        zs, slope = self._plate_surface(r)
        inv = 1.0 / np.sqrt(1.0 + slope * slope)
        gap_p = (x[:, 2] - zs) * inv - rad
        sel = np.nonzero(gap_p < margin)[0]
        rs = np.maximum(r[sel], 1e-12)
        n_p = np.stack(
            [-slope[sel] * x[sel, 0] / rs * inv[sel],
             -slope[sel] * x[sel, 1] / rs * inv[sel],
             inv[sel]], axis=1
        ) if sel.size else np.zeros((0, 3))
        depth_p = np.maximum(-gap_p[sel], 0.0) if sel.size else np.zeros(0)
        if sel.size:
            vrel_p = v[sel]
            vn = np.einsum("ij,ij->i", vrel_p, n_p)
            fn_p = np.maximum(self.CONTACT_STIFFNESS * depth_p
                              - self.CONTACT_DAMPING * vn, 0.0)
            fn_p = np.where(gap_p[sel] < 0.0, fn_p, 0.0)
        else:
            vrel_p = np.zeros((0, 3))
            fn_p = np.zeros(0)

        # strand-strand proximity (re-broadphase for a clean report)
        self._broadphase(x)
        pa, pb = self._pairs
        if pa.size:
            ia, ja = self.seg_i[pa], self.seg_j[pa]
            ib, jb = self.seg_i[pb], self.seg_j[pb]
            p1 = x[ia]
            d1 = x[ja] - p1
            p2 = x[ib]
            d2 = x[jb] - p2
            r12 = p1 - p2
            a = np.einsum("ij,ij->i", d1, d1)
            e = np.einsum("ij,ij->i", d2, d2)
            f = np.einsum("ij,ij->i", d2, r12)
            c = np.einsum("ij,ij->i", d1, r12)
            b = np.einsum("ij,ij->i", d1, d2)
            denom = np.maximum(a * e - b * b, 1e-18)
            s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            t = np.clip((b * s + f) / np.maximum(e, 1e-18), 0.0, 1.0)
            s = np.clip((b * t - c) / np.maximum(a, 1e-18), 0.0, 1.0)
            delta = (p1 + s[:, None] * d1) - (p2 + t[:, None] * d2)
            dist = np.linalg.norm(delta, axis=1)
            gap_s = dist - 2.0 * rad
            keep = np.nonzero(gap_s < margin)[0]
        else:
            keep = np.zeros(0, dtype=int)

        if keep.size:
            k = keep
            dk = np.maximum(dist[k], 1e-12)
            n_s = delta[k] / dk[:, None]
            bad = dist[k] < 1e-9
            if np.any(bad):
                n_s[bad] = np.array([0.0, 0.0, 1.0])
            depth_s = np.maximum(-gap_s[k], 0.0)
            sa_i, sa_j = self.seg_i[pa[k]], self.seg_j[pa[k]]
            sb_i, sb_j = self.seg_i[pb[k]], self.seg_j[pb[k]]
            wa0 = (1.0 - s[k])[:, None]
            wa1 = s[k][:, None]
            wb0 = (1.0 - t[k])[:, None]
            wb1 = t[k][:, None]
            va = v[sa_i] * wa0 + v[sa_j] * wa1
            vb = v[sb_i] * wb0 + v[sb_j] * wb1
            vrel_s = va - vb
            vn = np.einsum("ij,ij->i", vrel_s, n_s)
            fn_s = np.maximum(self.CONTACT_STIFFNESS * depth_s
                              - self.CONTACT_DAMPING * vn, 0.0)
            fn_s = np.where(gap_s[k] < 0.0, fn_s, 0.0)
        else:
            n_s = np.zeros((0, 3))
            vrel_s = np.zeros((0, 3))
            fn_s = np.zeros(0)
            sa_i = sa_j = sb_i = sb_j = np.zeros(0, dtype=int)
            wa0 = wa1 = wb0 = wb1 = np.zeros((0, 1))

        normal = np.concatenate([n_p, n_s], axis=0)
        vrel = np.concatenate([vrel_p, vrel_s], axis=0)
        fn = np.concatenate([fn_p, fn_s])
        mu_s = np.concatenate([
            np.full(fn_p.size, self.mu_sp_static),
            np.full(fn_s.size, self.mu_ss_static),
        ])
        mu_d = np.concatenate([
            np.full(fn_p.size, self.mu_sp_dynamic),
            np.full(fn_s.size, self.mu_ss_dynamic),
        ])

        if fn.size == 0:
            # Always emit at least one (inactive) contact row.
            node = int(np.argmin(gap_p))
            rr = max(float(r[node]), 1e-12)
            normal = np.array([[-slope[node] * x[node, 0] / rr * inv[node],
                                -slope[node] * x[node, 1] / rr * inv[node],
                                inv[node]]])
            vrel = v[node][None, :].copy()
            fn = np.zeros(1)
            mu_s = np.array([self.mu_sp_static])
            mu_d = np.array([self.mu_sp_dynamic])

        vt = vrel - np.einsum("ij,ij->i", vrel, normal)[:, None] * normal
        speed = np.linalg.norm(vt, axis=1)
        scale = mu_d * fn / np.maximum(speed, self.FRICTION_VEPS)
        ft = -scale[:, None] * vt
        mask = (fn > 0.0).astype(float)

        return {
            "normal": normal,
            "vrel": vrel,
            "fn": fn,
            "ft": ft,
            "mu_s": mu_s,
            "mu_d": mu_d,
            "mask": mask,
            "n_plate": int(fn_p.size),
            "plate_nodes": sel,
            "plate_fn": fn_p,
            "plate_ft": ft[: fn_p.size] if fn_p.size else np.zeros((0, 3)),
            "plate_normal": n_p,
            "seg_nodes": (sa_i, sa_j, sb_i, sb_j),
            "seg_w": (wa0, wa1, wb0, wb1),
            "seg_fn": fn_s,
            "seg_normal": n_s,
            "seg_ft": ft[fn_p.size:] if fn_s.size else np.zeros((0, 3)),
        }

    def _make_frame(self, time: float) -> StateFrame:
        rep = self._report_contacts()
        n = self.n_nodes
        w = self.domain_weights

        # per-node external force contributions at this sample state
        f_pn = np.zeros((n, 3))
        f_pf = np.zeros((n, 3))
        f_in = np.zeros((n, 3))
        if rep["plate_fn"].size:
            pn = rep["plate_nodes"]
            f_pn = _scatter3(n, pn, rep["plate_fn"][:, None] * rep["plate_normal"])
            f_pf = _scatter3(n, pn, rep["plate_ft"])
        if rep["seg_fn"].size:
            sa_i, sa_j, sb_i, sb_j = rep["seg_nodes"]
            wa0, wa1, wb0, wb1 = rep["seg_w"]
            fvec = rep["seg_fn"][:, None] * rep["seg_normal"] + rep["seg_ft"]
            f_in = (
                _scatter3(n, sa_i, fvec * wa0)
                + _scatter3(n, sa_j, fvec * wa1)
                - _scatter3(n, sb_i, fvec * wb0)
                - _scatter3(n, sb_j, fvec * wb1)
            )

        budget_force = np.stack(
            [self.gravity_force, w @ f_pn, w @ f_pf, w @ f_in], axis=1
        )

        impulse = (
            self.gravity_force * self.sim_time
            + w @ (self.imp_plate_n + self.imp_plate_f + self.imp_inter)
        )

        zeros_g3 = np.zeros((self.n_domains, 3))
        frame = StateFrame(
            time=float(time),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "strand_id": self.body_id.copy(),
                "node_index": self.node_index.copy(),
                "is_endpoint": self.is_endpoint.copy(),
            },
            fields={
                "budget_domain_weights": w.copy(),
                "budget_external_force": budget_force,
                "budget_cumulative_external_impulse": impulse,
                "budget_cumulative_inflow_momentum": zeros_g3.copy(),
                "budget_cumulative_outflow_momentum": zeros_g3.copy(),
                "cumulative_mass_source": np.zeros(1),
                "gravity_acceleration": self.gravity[None, :].copy(),
                "friction_active_mask": rep["mask"],
                "friction_normal": rep["normal"],
                "friction_normal_force": rep["fn"],
                "friction_tangential_force": rep["ft"],
                "friction_relative_velocity": rep["vrel"],
                "friction_static_coefficient": rep["mu_s"],
                "friction_dynamic_coefficient": rep["mu_d"],
                "external_force": self.mass[:, None] * self.gravity[None, :]
                + f_pn + f_pf,
            },
            scalars={
                "friction_stick_speed_tolerance": float(self.FRICTION_VEPS),
                "reference_force": float(self.reference_force),
                "reference_momentum": float(self.reference_momentum),
                "reference_velocity": float(self.reference_velocity),
                "contact_length_scale": float(self.radius),
                "strand_radius": float(self.radius),
                "static_friction": float(self.mu_ss_static),
                "kinetic_friction": float(self.mu_ss_dynamic),
            },
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------- API
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        # integrate with the fixed sub-step until the sample time is reached
        while self.sim_time < target - 0.5 * self.dt:
            self._step()
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "discrete elastic rods, semi-implicit Euler",
            "friction": "implicit regularised Coulomb (mu=0.3), penalty normal",
            "substep_dt": float(self.dt),
            "steps_taken": int(self.step_count),
            "strands": int(self.n_strands),
            "nodes": int(self.n_nodes),
            "bend_stiffness_N_per_m": float(self.bend_stiffness),
            "max_abs_velocity": float(self._max_speed),
            "min_plate_gap": float(self._min_gap),
            "final_max_height": float(self.position[:, 2].max()),
            "final_max_speed": float(np.abs(self.velocity).max()),
        }
