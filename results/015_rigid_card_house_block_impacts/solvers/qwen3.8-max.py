"""Physically-based rigid-body solver for the card-house block-impact scene.

Method: semi-implicit rigid body dynamics with quaternion orientation,
penalty (spring-damper) normal contact with Coulomb friction clipped for
stability, point-sampled box collision (corners + face centers, two-way),
AABB broadphase, and translational position projection to bound penetration.

Initial geometric overlaps present in the frozen scene (cards crossing at
apices, card feet dipping into the floor) are captured as reference
penetration offsets at t=0 so the house starts in static equilibrium and
only settles by the small designed gaps.

Impact blocks are held kinematic until their release time, then fall under
gravity and strike the card houses.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


def _quat_to_R(q):
    """[nb,4] xyzw quaternions -> [nb,3,3] local->world rotation matrices."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = q.shape[0]
    R = np.empty((n, 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _box_sample_points(h):
    """Local sample points for a box: 8 corners + 6 face centers."""
    hx, hy, hz = h
    pts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append([sx * hx, sy * hy, sz * hz])
    pts.extend([[hx, 0, 0], [-hx, 0, 0], [0, hy, 0],
                [0, -hy, 0], [0, 0, hz], [0, 0, -hz]])
    return np.array(pts, dtype=float)


class BenchmarkSolver:
    # --- numerical / material parameters ---
    SUBSTEPS = 20            # substeps per output sample (dt = 1e-3 s)
    K_NORMAL = 2.0e4         # penalty stiffness [N/m]
    ZETA = 0.5               # contact damping ratio (near-inelastic, e~0.1)
    MARGIN = 5.0e-3          # near-contact detection margin [m]
    BETA = 0.35              # position projection factor
    SLOP = 2.0e-4            # projection slop [m]
    F_MAX = 8.0e3            # per-contact force cap [N]

    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.gravity = np.array(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.mu = float(cfg["contact_parameters"]["friction_coefficient"])
        self.dt_sample = float(cfg["sample_dt"])
        self.dt_sub = self.dt_sample / self.SUBSTEPS

        frame0 = scene.initial_state()
        pos0 = np.array(frame0.position, dtype=float)
        quat0 = np.array(frame0.orientation, dtype=float)
        quat0 /= np.linalg.norm(quat0, axis=1, keepdims=True)
        self.mass = np.array(frame0.mass, dtype=float)
        self.body_ids = np.array(frame0.body_id, dtype=int)
        self.nb = len(self.mass)

        attrs = frame0.attributes
        self.release_time = np.array(attrs.get(
            "release_time", np.full(self.nb, -1.0)), dtype=float)
        self.is_block = np.array(attrs.get(
            "is_impact_block", np.zeros(self.nb)), dtype=float) > 0.5

        # --- geometry: half extents per dynamic body + static floor ---
        half = np.zeros((self.nb, 3))
        floor_h = np.array([3.2, 2.2, 0.05])
        floor_t = np.array([0.0, 0.0, -0.05])
        geom = cfg.get("scene_geometry")
        if geom is not None:
            binding_map = {}
            for obj in geom.get("objects", []):
                g = obj.get("geometry", {})
                if g.get("type") != "box":
                    continue
                tr = obj.get("transform", {})
                sb = obj.get("state_binding")
                if obj.get("motion") == "static" or sb is None:
                    if obj.get("motion") == "static":
                        floor_h = np.array(g["half_extents"], dtype=float)
                        floor_t = np.array(tr.get(
                            "translation", floor_t), dtype=float)
                    continue
                if sb.get("selector") == "body_id":
                    binding_map[int(sb["value"])] = np.array(
                        g["half_extents"], dtype=float)
            for k, idx in enumerate(self.body_ids):
                if int(idx) in binding_map:
                    half[k] = binding_map[int(idx)]
        else:
            half[:10] = [0.45, 0.18, 0.015]
            half[10] = half[11] = [0.16, 0.16, 0.16]
            half[12] = [0.19, 0.19, 0.19]

        # all bodies: 0..nb-1 dynamic, nb = static floor
        self.nall = self.nb + 1
        self.h = np.vstack([half, floor_h[None, :]])
        self.floor_idx = self.nb

        # state
        self.pos = np.zeros((self.nall, 3))
        self.pos[:self.nb] = pos0
        self.pos[self.floor_idx] = floor_t
        self.quat = np.zeros((self.nall, 4))
        self.quat[:self.nb] = quat0
        self.quat[self.floor_idx] = [0.0, 0.0, 0.0, 1.0]
        self.vel = np.zeros((self.nall, 3))
        self.om = np.zeros((self.nall, 3))

        self.inv_m = np.zeros(self.nall)
        self.inv_m[:self.nb] = 1.0 / self.mass
        # blocks held (kinematic) until release
        held = self.is_block.copy()
        self.inv_m[:self.nb][held] = 0.0
        self.inv_I = np.zeros((self.nall, 3))
        for k in range(self.nb):
            hx, hy, hz = self.h[k]
            Ix = self.mass[k] / 12.0 * (hy * hy + hz * hz)
            Iy = self.mass[k] / 12.0 * (hx * hx + hz * hz)
            Iz = self.mass[k] / 12.0 * (hx * hx + hy * hy)
            self.inv_I[k] = [1.0 / Ix, 1.0 / Iy, 1.0 / Iz]
        self.released = ~held  # dynamic cards released from t=0

        # local sample points per body
        self.pts = np.stack([_box_sample_points(self.h[k])
                             for k in range(self.nall)])
        self.npts = self.pts.shape[1]

        # pair list (all pairs; broadphase filters per substep)
        ii, jj = np.meshgrid(np.arange(self.nall), np.arange(self.nall),
                             indexing="ij")
        keep = ii < jj
        self.pair_i = ii[keep]
        self.pair_j = jj[keep]

        self.time = 0.0
        self.step_count = 0

        # reference penetration offsets from initial (frozen-scene) overlaps
        self.ref_pen = {}
        R = _quat_to_R(self.quat)
        self._refresh_kinematics(R)
        scan = self._scan_contacts(R)
        for (i, j, p), (_, pen) in scan.items():
            if pen > 0.0:
                self.ref_pen[(i, j, p)] = pen
        # initial frame field cache (no forces yet)
        self.last_forces = []
        self.last_sds = [v[0] for v in scan.values()]
        self.frame0 = frame0

    # ------------------------------------------------------------------
    def _refresh_kinematics(self, R):
        """World-space sample points and AABBs for all bodies."""
        # Pw[k,p] = pos[k] + R[k] @ pts[k,p]
        self.Pw = self.pos[:, None, :] + np.einsum(
            "kab,kpb->kpa", R, self.pts)
        ext = np.einsum("kab,kb->ka", np.abs(R), self.h) + self.MARGIN
        self.bb_min = self.pos - ext
        self.bb_max = self.pos + ext

    def _scan_contacts(self, R):
        """Detect point-in-box contacts within margin.

        Returns dict {(i,j,p): (sd, pen)} for point p of body i against box j.
        """
        out = {}
        lo_i = self.bb_min[self.pair_i]
        hi_i = self.bb_max[self.pair_i]
        lo_j = self.bb_min[self.pair_j]
        hi_j = self.bb_max[self.pair_j]
        overlap = (np.all(lo_i < hi_j, axis=1) & np.all(lo_j < hi_i, axis=1))
        for a, b in zip(self.pair_i[overlap], self.pair_j[overlap]):
            for (i, j) in ((a, b), (b, a)):
                D = (self.Pw[i] - self.pos[j]) @ R[j]      # [P,3] local
                s = np.abs(D) - self.h[j]                  # [P,3]
                sd = s.max(axis=1)
                hit = np.flatnonzero(sd < self.MARGIN)
                for p in hit:
                    out[(i, j, int(p))] = (float(sd[p]), float(-sd[p]))
        return out

    def _substep(self, h):
        t_new = self.time + h
        # release held blocks at their release times
        if np.any(self.is_block & ~self.released):
            newly = self.is_block & ~self.released & (t_new >= self.release_time)
            if np.any(newly):
                self.released |= newly
                self.inv_m[:self.nb][newly] = 1.0 / self.mass[newly]

        R = _quat_to_R(self.quat)
        self._refresh_kinematics(R)

        # gravity on released dynamic bodies
        gmask = np.zeros(self.nall, dtype=bool)
        gmask[:self.nb] = self.released
        self.vel[gmask] += self.gravity * h

        # world inverse inertia tensors
        Iw_inv = np.einsum("kab,kb,kcb->kac", R, self.inv_I, R)

        contacts = self._scan_contacts(R)

        k = self.K_NORMAL
        mu = self.mu
        force_log = []
        sd_log = []
        proj = []
        for (i, j, p), (sd, pen) in contacts.items():
            sd_log.append(sd)
            pen_eff = pen - self.ref_pen.get((i, j, p), 0.0)
            if pen_eff <= 0.0:
                continue
            pw = self.Pw[i, p]
            D = R[j].T @ (pw - self.pos[j])
            s = np.abs(D) - self.h[j]
            ax = int(np.argmax(s))
            nl = np.zeros(3)
            nl[ax] = 1.0 if D[ax] >= 0.0 else -1.0
            n = R[j] @ nl                       # outward from box j
            ri = pw - self.pos[i]
            rj = pw - self.pos[j]
            im_i = self.inv_m[i]
            im_j = self.inv_m[j]
            if im_i + im_j <= 0.0:
                continue
            vi = self.vel[i] + np.cross(self.om[i], ri)
            vj = self.vel[j] + np.cross(self.om[j], rj)
            rv = vi - vj
            vn = float(rv @ n)
            m_eff = 1.0 / (im_i + im_j)
            c_n = 2.0 * self.ZETA * np.sqrt(k * m_eff)
            fn = k * pen_eff + c_n * max(0.0, -vn)
            fn = min(max(fn, 0.0), self.F_MAX)
            F = fn * n
            vt = rv - vn * n
            vt_norm = float(np.linalg.norm(vt))
            if vt_norm > 1e-9:
                f_fric = min(mu * fn, m_eff * vt_norm / h)
                F = F - (f_fric / vt_norm) * vt
            # apply impulses sequentially
            self.vel[i] += F * (im_i * h)
            self.vel[j] -= F * (im_j * h)
            self.om[i] += Iw_inv[i] @ np.cross(ri, F) * h
            self.om[j] -= Iw_inv[j] @ np.cross(rj, F) * h
            force_log.append((i, j, F.copy()))
            proj.append((i, j, pen_eff, n))

        # integrate positions / orientations
        self.pos += self.vel * h
        om_v = self.om
        qv = self.quat[:, :3]
        w = self.quat[:, 3:4]
        dq_v = 0.5 * (w * om_v + np.cross(qv, om_v))
        dq_w = -0.5 * np.einsum("ka,ka->k", qv, om_v)[:, None]
        self.quat[:, :3] += dq_v * h
        self.quat[:, 3:] += dq_w * h
        self.quat /= np.linalg.norm(self.quat, axis=1, keepdims=True)
        self.pos[self.floor_idx] = self.pos[self.floor_idx]  # static stays
        self.vel[self.floor_idx] = 0.0
        self.om[self.floor_idx] = 0.0
        heldmask = np.zeros(self.nall, dtype=bool)
        heldmask[:self.nb] = self.is_block & ~self.released
        self.vel[heldmask] = 0.0
        self.om[heldmask] = 0.0

        # positional projection (translational split by inverse mass)
        for (i, j, pen_eff, n) in proj:
            if pen_eff <= self.SLOP:
                continue
            im_i = self.inv_m[i]
            im_j = self.inv_m[j]
            tot = im_i + im_j
            if tot <= 0.0:
                continue
            corr = self.BETA * (pen_eff - self.SLOP) / tot
            self.pos[i] += im_i * corr * n
            self.pos[j] -= im_j * corr * n

        self.time = t_new
        self.step_count += 1
        self.last_forces = force_log
        self.last_sds = sd_log

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        time = float(time)
        guard = 0
        while self.time < time - 1e-12 and guard < 10_000_000:
            h = min(self.dt_sub, time - self.time)
            self._substep(h)
            guard += 1
        return self._build_frame(self.time)

    def _build_frame(self, t):
        pos = self.pos[:self.nb].copy()
        vel = self.vel[:self.nb].copy()
        quat = self.quat[:self.nb].copy()
        om = self.om[:self.nb].copy()

        # external force: gravity on released bodies, zero while held
        ext = np.zeros((self.nb, 3))
        ext[self.released] = self.mass[self.released, None] * self.gravity

        # interaction forces aggregated per ordered contact direction (i,j)
        agg = {}
        for (i, j, F) in self.last_forces:
            key = (i, j)
            agg[key] = agg.get(key, np.zeros(3)) + F
        if agg:
            inter = np.stack(list(agg.values()))
        else:
            inter = np.zeros((0, 3))

        sds = np.asarray(self.last_sds, dtype=float)
        if sds.size == 0:
            sds = np.array([self.MARGIN])

        frame = StateFrame(
            time=float(t),
            position=pos,
            velocity=vel,
            orientation=quat,
            angular_velocity=om,
            mass=self.mass.copy(),
            body_id=self.body_ids.copy(),
            attributes={
                "release_time": np.array(self.release_time, dtype=float),
                "is_impact_block": np.array(self.is_block, dtype=float),
            },
            fields={
                "contact_signed_distance": sds,
                "external_force": ext,
                "interaction_force": inter,
            },
            scalars={
                "contact_length_scale": 0.03,
                "friction_coefficient": self.mu,
                "restitution": float(self.frame0.scalars.get("restitution", 0.1)),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict:
        return {
            "method": "penalty contact rigid body (semi-implicit Euler)",
            "substeps_per_sample": self.SUBSTEPS,
            "steps_executed": self.step_count,
            "contacts_last_scan": len(self.last_sds),
        }
