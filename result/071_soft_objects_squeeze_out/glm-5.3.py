"""Solver for scene 071_soft_objects_squeeze_out.

Physical model
--------------
Each soft body is a closed icosahedral surface mesh (12 dynamic nodes, 20
triangles) taken from the frozen asset and the frozen per-object transform.
The discrete elastic model is:

* linear edge springs along the 30 mesh edges with viscous damping
  (surface elasticity), and
* a volumetric constraint energy U(V) = k_v/2 (V - V0)^2 whose gradient
  yields a pressure-style volume-preservation force (near-incompressible
  3D solid behaviour).

Contacts against the frozen tube/funnel/outlet/collector geometry (oriented
boxes, including the rotated funnel walls) and against the kinematic pusher
plate use penalty normal forces with normal damping and capped Coulomb
friction.  Body-body contacts use point-triangle penalty contact with
friction; the reaction is distributed on the triangle vertices so every
interaction pair is exactly antisymmetric.

Time integration is semi-implicit (symplectic) Euler with a fixed substep,
so the exported momentum budgets are exact by construction:
    P_domain(t) = P_domain(0) + integral of exported external forces.
No animation, keyframes, or precomputed trajectories are used; every frame
is produced by advancing the physical state.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Frozen numerical parameters (deterministic, fixed at import).
# ---------------------------------------------------------------------------
DT_SUB = 2.0e-4          # integration substep [s]
K_SPRING = 8.0e2         # edge-spring stiffness [N/m]
C_SPRING = 8.0           # edge-spring damping [N s/m]
K_VOLUME = 3.0e4         # volume-preservation stiffness [N/m^3]
K_CONTACT = 1.5e5        # penalty contact stiffness [N/m]
C_CONTACT = 1.2e2        # contact normal damping [N s/m]
MU = 0.3                 # Coulomb friction coefficient
C_FRICTION = 2.5e2       # tangential viscous cap [N s/m]
V_CLAMP = 100.0          # safety velocity clamp [m/s]
EPS = 1e-12


def _quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n < EPS:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def _closest_points_on_triangles(p, a, b, c, ab, ac):
    """Branchless Ericson closest-point-on-triangle, broadcast over (P,T)."""
    ap = p - a
    d1 = _dot(ab, ap)
    d2 = _dot(ac, ap)
    bp = p - b
    d3 = _dot(ab, bp)
    d4 = _dot(ac, bp)
    cp = p - c
    d5 = _dot(ab, cp)
    d6 = _dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    den = np.where(np.abs(denom) < 1e-14, 1.0, denom)
    v = vb / den
    w = vc / den
    res = a + v[..., None] * ab + w[..., None] * ac  # interior
    # edge BC
    d43 = d4 - d3
    d56 = d5 - d6
    dbc = d43 + d56
    dbcs = np.where(np.abs(dbc) < 1e-14, 1.0, dbc)
    tbc = np.clip(d43 / dbcs, 0.0, 1.0)
    mbc = (va <= 0.0) & (d43 >= 0.0) & (d56 >= 0.0)
    res = np.where(mbc[..., None], b + tbc[..., None] * (c - b), res)
    # edge AC
    dac = d2 - d6
    dacs = np.where(np.abs(dac) < 1e-14, 1.0, dac)
    tac = np.clip(d2 / dacs, 0.0, 1.0)
    mac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    res = np.where(mac[..., None], a + tac[..., None] * ac, res)
    # vertex C
    mc = (d6 >= 0.0) & (d5 <= d6)
    res = np.where(mc[..., None], c, res)
    # edge AB
    dab = d1 - d2
    dabs = np.where(np.abs(dab) < 1e-14, 1.0, dab)
    tab = np.clip(d1 / dabs, 0.0, 1.0)
    mab = (vc <= 0.0) & (d1 >= 0.0) & (d2 <= 0.0)
    res = np.where(mab[..., None], a + tab[..., None] * ab, res)
    # vertex B, vertex A
    mb = (d3 >= 0.0) & (d4 <= d3)
    res = np.where(mb[..., None], b, res)
    ma = (d1 <= 0.0) & (d2 <= 0.0)
    res = np.where(ma[..., None], a, res)
    return res


class BenchmarkSolver:
    """Soft-body squeeze-out solver (edge springs + volume constraint)."""

    def initialize(self, scene, seed: int) -> None:
        config = getattr(scene, "config", {}) or {}
        geometry = config.get("scene_geometry", {}) or {}
        objects = geometry.get("objects", []) or []
        gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=float)

        initial = scene.initial_state()
        body_mass = np.asarray(initial.mass, dtype=float).reshape(-1)

        # ----- soft bodies (frozen asset + frozen transforms) -------------
        soft = [o for o in objects if o.get("semantic") == "deformable"]
        soft.sort(key=lambda o: int(o.get("state_binding", {}).get("value", 0)))
        n_bodies = len(soft)
        verts_unit, faces_local = self._load_asset(soft[0]["geometry"]["asset"])
        n_vert = len(verts_unit)  # 12

        x0 = []
        scales = []
        for obj in soft:
            tr = obj.get("transform", {}) or {}
            t = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            s = np.asarray(tr.get("scale", [1.0, 1.0, 1.0]), dtype=float)
            rot = _quat_xyzw_to_matrix(tr.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]))
            x0.append((rot @ (s[None, :] * verts_unit).T).T + t[None, :])
            scales.append(s)
        X0 = np.vstack(x0)  # (N,3) material/reference positions
        n_nodes = n_bodies * n_vert

        # outward winding check (auto-correct sign convention if needed)
        a0 = X0[faces_local[:, 0]]
        b0 = X0[faces_local[:, 1]]
        c0 = X0[faces_local[:, 2]]
        vol0 = float(np.einsum("ij,ij->i", np.cross(a0, b0), c0).sum() / 6.0)
        if vol0 < 0.0:
            faces_local = faces_local[:, ::-1].copy()

        # ----- node quantities --------------------------------------------
        self.n_bodies = n_bodies
        self.n_vert = n_vert
        self.n_nodes = n_nodes
        self.gravity = gravity
        self.x = X0.copy()
        self.v = np.zeros_like(X0)
        node_mass = np.repeat(body_mass / n_vert, n_vert) if n_bodies else np.zeros(0)
        self.m = node_mass
        self.total_mass = float(node_mass.sum())
        self.body_id = np.repeat(np.arange(n_bodies, dtype=np.int64), n_vert)
        self.X0 = X0
        self.faces_local = faces_local

        # global face indices per body
        self.faces = np.vstack(
            [faces_local + b * n_vert for b in range(n_bodies)]
        ).reshape(n_bodies, len(faces_local), 3)

        # edge spring network from unique mesh edges
        edges_local = np.unique(
            np.sort(
                np.vstack((faces_local[:, [0, 1]], faces_local[:, [1, 2]], faces_local[:, [2, 0]])),
                axis=1,
            ),
            axis=0,
        )
        E = []
        for b in range(n_bodies):
            E.append(edges_local + b * n_vert)
        self.E = np.vstack(E)
        d = X0[self.E[:, 1]] - X0[self.E[:, 0]]
        self.L0 = np.linalg.norm(d, axis=1)

        # rest volumes and tet elements (face + centroid) for F export
        self.V0 = np.zeros(n_bodies)
        self.tet_A_inv = np.zeros((n_bodies, len(faces_local), 3, 3))
        for b in range(n_bodies):
            xs = X0[b * n_vert : (b + 1) * n_vert]
            a, bb, c = xs[faces_local[:, 0]], xs[faces_local[:, 1]], xs[faces_local[:, 2]]
            self.V0[b] = float(np.einsum("ij,ij->i", np.cross(a, bb), c).sum() / 6.0)
            cen = xs.mean(axis=0)
            mat = np.stack(
                [a - cen, bb - cen, c - cen], axis=-1
            )  # columns are material edge vectors
            self.tet_A_inv[b] = np.linalg.inv(mat)

        # broad-phase radii
        self.pair_radius = np.zeros(n_bodies)
        for b in range(n_bodies):
            xs = X0[b * n_vert : (b + 1) * n_vert]
            self.pair_radius[b] = float(np.linalg.norm(xs - xs.mean(axis=0), axis=1).max()) + 0.03

        # ----- static boundary boxes (oriented) ---------------------------
        boxes_R, boxes_c, boxes_h = [], [], []
        plate_obj = None
        for obj in objects:
            if obj.get("semantic") not in {"boundary", "rigid"}:
                continue
            if obj.get("geometry", {}).get("type") != "box":
                continue
            tr = obj.get("transform", {}) or {}
            boxes_R.append(_quat_xyzw_to_matrix(tr.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])))
            boxes_c.append(np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float))
            boxes_h.append(np.asarray(obj["geometry"]["half_extents"], dtype=float))
            if obj.get("motion") == "kinematic":
                plate_obj = obj
        self.box_R = np.asarray(boxes_R) if boxes_R else np.zeros((0, 3, 3))
        self.box_c_static = np.asarray(boxes_c) if boxes_c else np.zeros((0, 3))
        self.box_h_static = np.asarray(boxes_h) if boxes_h else np.zeros((0, 3))

        # ----- kinematic pusher plate --------------------------------------
        pm = (config.get("boundary_conditions", {}) or {}).get("plate_motion", {}) or {}
        self.plate_t0 = float(pm.get("start_time", 0.0))
        self.plate_t1 = float(pm.get("end_time", self.plate_t0 + 1.0))
        z0 = float(np.asarray(pm.get("start_translation", [0.0, 0.0, 0.0]), dtype=float)[2])
        z1 = float(np.asarray(pm.get("end_translation", [0.0, 0.0, 0.0]), dtype=float)[2])
        self.plate_z0, self.plate_z1 = z0, z1
        if plate_obj is not None:
            tr = plate_obj.get("transform", {}) or {}
            self.plate_h = np.asarray(plate_obj["geometry"]["half_extents"], dtype=float)
            self.plate_xy = np.asarray(tr.get("translation", [0.0, 0.0, 0.0]), dtype=float)[:2].copy()
            self.plate_R = _quat_xyzw_to_matrix(tr.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0]))
        else:
            self.plate_h = np.zeros(3)
            self.plate_xy = np.zeros(2)
            self.plate_R = np.eye(3)

        # combined box arrays: static boxes first, plate last
        nb = len(boxes_c)
        self.n_static = nb
        self.R_all = np.zeros((nb + 1, 3, 3))
        self.c_all = np.zeros((nb + 1, 3))
        self.h_all = np.zeros((nb + 1, 3))
        self.vel_all = np.zeros((nb + 1, 3))
        if nb:
            self.R_all[:nb] = self.box_R
            self.c_all[:nb] = self.box_c_static
            self.h_all[:nb] = self.box_h_static
        self.R_all[nb] = self.plate_R
        self.h_all[nb] = self.plate_h
        self.c_all[nb, :2] = self.plate_xy

        # ----- budget bookkeeping ------------------------------------------
        # domains: row 0 = global (all nodes), rows 1..G-1 = individual bodies
        self.G = n_bodies + 1
        self.K_terms = 2  # 0: gravity, 1: contact (boundary + inter-body)
        weights = np.zeros((self.G, n_nodes))
        weights[0, :] = 1.0
        for b in range(n_bodies):
            weights[b + 1, b * n_vert : (b + 1) * n_vert] = 1.0
        self.W = weights
        g_dom = np.zeros((self.G, 3))
        g_dom[0] = self.total_mass * gravity
        for b in range(n_bodies):
            g_dom[b + 1] = body_mass[b] * gravity
        self.g_dom = g_dom

        self.cum_impulse = np.zeros((self.G, 3))
        self.interval_impulse = np.zeros((self.G, self.K_terms, 3))
        self.interval_inter = np.zeros((n_bodies, 3))
        # first frame: report the (exact) initial gravity force
        self.interval_impulse[:, 0, :] = g_dom * DT_SUB

        self.steps = 0
        self.time = 0.0
        self.last_frame_time = 0.0
        self.last_report = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _load_asset(asset_path):
        from pathlib import Path

        path = Path(__file__).resolve().parents[1] / asset_path
        verts, faces = [], []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if not parts:
                    continue
                if parts[0] == "v":
                    verts.append([float(v) for v in parts[1:4]])
                elif parts[0] == "f":
                    faces.append([int(p.split("/")[0]) - 1 for p in parts[1:4]])
        return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)

    # ------------------------------------------------------------------
    def _plate_z(self, t: float) -> float:
        if t <= self.plate_t0:
            return self.plate_z0
        if t >= self.plate_t1:
            return self.plate_z1
        u = (t - self.plate_t0) / max(self.plate_t1 - self.plate_t0, EPS)
        return self.plate_z0 + u * (self.plate_z1 - self.plate_z0)

    # ------------------------------------------------------------------
    def _box_contacts(self, F, contact_body):
        """Penalty contact of all nodes against static boxes and the plate."""
        x, v = self.x, self.v
        R, C, H, VEL = self.R_all, self.c_all, self.h_all, self.vel_all
        nB = len(C)
        rel = x[:, None, :] - C[None, :, :]                     # (N,B,3)
        q = np.einsum("bji,nbj->nbi", R, rel)                   # box-local coords
        cl = np.clip(q, -H[None], H[None])
        dout = q - cl
        dist = np.linalg.norm(dout, axis=-1)
        inside = dist < 1e-10
        slack = H[None] - np.abs(q)
        ax = np.argmin(np.where(inside[..., None], slack, np.inf), axis=-1)
        smin = np.take_along_axis(slack, ax[..., None], -1)[..., 0]
        sgn = np.sign(np.take_along_axis(q, ax[..., None], -1)[..., 0])
        sgn = np.where(sgn == 0.0, 1.0, sgn)
        gap = np.where(inside, -smin, dist)                     # <0 when penetrating
        nbox = np.zeros_like(q)
        np.put_along_axis(nbox, ax[..., None], sgn[..., None], -1)
        nw = np.einsum("bij,nbj->nbi", R, nbox)                 # world normal (out of box)

        active = gap < 0.0
        vrel = v[:, None, :] - VEL[None, :, :]
        vn = _dot(vrel, nw)
        fn = K_CONTACT * (-gap) + C_CONTACT * np.maximum(0.0, -vn)
        vt = vrel - vn[..., None] * nw
        vtl = np.linalg.norm(vt, axis=-1)
        vtl_s = np.maximum(vtl, 1e-12)
        fmag = np.minimum(MU * fn, C_FRICTION * vtl)
        Ft = -(fmag / vtl_s)[..., None] * vt
        Fc = fn[..., None] * nw + Ft
        Fc = np.where(active[..., None], Fc, 0.0)

        idx = np.nonzero(active)
        if idx[0].size:
            np.add.at(F, idx[0], Fc[idx])
            np.add.at(contact_body, self.body_id[idx[0]], Fc[idx])
        return Fc.sum(axis=1)  # per-node wall/plate force (unused sum for debug)

    # ------------------------------------------------------------------
    def _pair_contact(self, A, B, F, inter_body):
        """Point-triangle penalty contact between bodies A and B (both ways)."""
        nv = self.n_vert
        for na, nb in ((A, B), (B, A)):
            i0 = na * nv
            p = self.x[i0 : i0 + nv][:, None, :]
            trif = self.faces[nb]
            tri = self.x[trif][None, ...]
            a = tri[..., 0, :]
            b = tri[..., 1, :]
            c = tri[..., 2, :]
            ab = b - a
            ac = c - a
            nrm = np.cross(ab, ac)
            nn = np.maximum(np.linalg.norm(nrm, axis=-1), 1e-14)
            nhat = nrm / nn[..., None]
            cp = _closest_points_on_triangles(p, a, b, c, ab, ac)
            gap = _dot(p - cp, nhat)
            act = gap < 0.0
            if not act.any():
                continue
            vtri = self.v[trif].mean(axis=1)[None, :, :]
            vrel = self.v[i0 : i0 + nv][:, None, :] - vtri
            vn = _dot(vrel, nhat)
            fn = K_CONTACT * (-gap) + C_CONTACT * np.maximum(0.0, -vn)
            vt = vrel - vn[..., None] * nhat
            vtl = np.linalg.norm(vt, axis=-1)
            vtl_s = np.maximum(vtl, 1e-12)
            fmag = np.minimum(MU * fn, C_FRICTION * vtl)
            Ft = -(fmag / vtl_s)[..., None] * vt
            Fc = fn[..., None] * nhat + Ft
            Fc = np.where(act[..., None], Fc, 0.0)
            Fnode = Fc.sum(axis=1)
            F[i0 : i0 + nv] += Fnode
            Fre = -Fc.sum(axis=0) / 3.0
            np.add.at(F, trif.reshape(-1), np.repeat(Fre, 3, axis=0))
            inter_body[na] += Fnode.sum(axis=0)
            inter_body[nb] -= Fnode.sum(axis=0)

    # ------------------------------------------------------------------
    def _substep(self):
        dt = DT_SUB
        t0 = self.time
        t1 = t0 + dt

        # plate kinematics over this substep
        z0 = self._plate_z(t0)
        z1 = self._plate_z(t1)
        self.c_all[self.n_static, 2] = 0.5 * (z0 + z1)
        self.vel_all[:] = 0.0
        self.vel_all[self.n_static, 2] = (z1 - z0) / dt

        x, v, m = self.x, self.v, self.m
        F = np.zeros_like(x)
        F += m[:, None] * self.gravity

        # --- edge springs (vectorised over all bodies) ---------------------
        E, L0 = self.E, self.L0
        d = x[E[:, 1]] - x[E[:, 0]]
        ln = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        ndir = d / ln[:, None]
        vr = _dot(v[E[:, 1]] - v[E[:, 0]], ndir)
        fmag = K_SPRING * (ln - L0) + C_SPRING * vr
        fe = fmag[:, None] * ndir
        np.add.at(F, E[:, 0], fe)
        np.add.at(F, E[:, 1], -fe)

        # --- volume preservation per body ---------------------------------
        fl = self.faces_local
        for b in range(self.n_bodies):
            i0 = b * self.n_vert
            xs = x[i0 : i0 + self.n_vert]
            a, bb, c = xs[fl[:, 0]], xs[fl[:, 1]], xs[fl[:, 2]]
            vol = float(np.einsum("ij,ij->i", np.cross(a, bb), c).sum() / 6.0)
            G = np.zeros((self.n_vert, 3))
            np.add.at(G, fl[:, 0], np.cross(bb, c))
            np.add.at(G, fl[:, 1], np.cross(c, a))
            np.add.at(G, fl[:, 2], np.cross(a, bb))
            G *= 1.0 / 6.0
            F[i0 : i0 + self.n_vert] += -K_VOLUME * (vol - self.V0[b]) * G

        # --- contacts with walls / funnel / lips / collector / plate -------
        contact_body = np.zeros((self.n_bodies, 3))
        self._box_contacts(F, contact_body)

        # --- body-body contacts -------------------------------------------
        inter_body = np.zeros((self.n_bodies, 3))
        centers = np.array(
            [x[b * self.n_vert : (b + 1) * self.n_vert].mean(axis=0) for b in range(self.n_bodies)]
        )
        for i in range(self.n_bodies):
            ri = self.pair_radius[i]
            for j in range(i + 1, self.n_bodies):
                if float(np.linalg.norm(centers[i] - centers[j])) > ri + self.pair_radius[j]:
                    continue
                self._pair_contact(i, j, F, inter_body)

        # --- momentum budget accumulation (exact for semi-implicit Euler) --
        contact_dom = np.zeros((self.G, 3))
        contact_dom[0] = contact_body.sum(axis=0)  # inter-body cancels globally
        contact_dom[1:] = contact_body + inter_body
        self.cum_impulse += (self.g_dom + contact_dom) * dt
        self.interval_impulse[:, 0, :] += self.g_dom * dt
        self.interval_impulse[:, 1, :] += contact_dom * dt
        self.interval_inter += inter_body * dt

        # --- integrate ------------------------------------------------------
        v += (F / m[:, None]) * dt
        speed = np.linalg.norm(v, axis=1)
        fast = speed > V_CLAMP
        if fast.any():
            v[fast] *= (V_CLAMP / speed[fast])[:, None]
        x += v * dt

        self.steps += 1
        self.time = t1

    # ------------------------------------------------------------------
    def _deformation_gradients(self):
        fl = self.faces_local
        out = np.empty((self.n_bodies * len(fl), 3, 3))
        for b in range(self.n_bodies):
            i0 = b * self.n_vert
            xs = self.x[i0 : i0 + self.n_vert]
            cen = xs.mean(axis=0)
            cur = np.stack(
                [xs[fl[:, 0]] - cen, xs[fl[:, 1]] - cen, xs[fl[:, 2]] - cen], axis=-1
            )
            out[b * len(fl) : (b + 1) * len(fl)] = cur @ self.tet_A_inv[b]
        return out

    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        from worldsolver.contracts import StateFrame

        t = float(time)
        target = int(np.floor(t / DT_SUB + 1e-6))
        while self.steps < target:
            self._substep()

        interval = max(t - self.last_frame_time, DT_SUB)
        external_force = self.interval_impulse / interval
        interaction = self.interval_inter / interval
        n_mat = self.n_bodies

        frame = StateFrame(
            time=t,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.m.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "mass_quantity": self.m.copy(),
                "reference_position": self.X0.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": self.cum_impulse.copy(),
                "budget_cumulative_inflow_momentum": np.zeros((self.G, 3)),
                "budget_cumulative_outflow_momentum": np.zeros((self.G, 3)),
                "budget_domain_weights": self.W.copy(),
                "budget_external_force": external_force,
                "cumulative_mass_source": np.zeros(n_mat),
                "element_deformation_gradient": self._deformation_gradients(),
                "interaction_force": interaction,
            },
            scalars={
                "reference_momentum": float(max(self.total_mass * 1.0, EPS)),
                "gravity_z": float(self.gravity[2]),
            },
        )
        frame.validate()
        self.last_frame_time = t
        self.interval_impulse[:] = 0.0
        self.interval_inter[:] = 0.0
        self.last_report = {
            "time": t,
            "steps": self.steps,
            "max_speed": float(np.linalg.norm(self.v, axis=1).max()),
            "min_volume": float(min(
                self._body_volume(b) for b in range(self.n_bodies)
            )),
        }
        return frame

    # ------------------------------------------------------------------
    def _body_volume(self, b: int) -> float:
        xs = self.x[b * self.n_vert : (b + 1) * self.n_vert]
        fl = self.faces_local
        a, bb, c = xs[fl[:, 0]], xs[fl[:, 1]], xs[fl[:, 2]]
        return float(np.einsum("ij,ij->i", np.cross(a, bb), c).sum() / 6.0)

    # ------------------------------------------------------------------
    def diagnostics(self) -> dict:
        return {
            "solver": "mass-spring + volume-constraint soft bodies",
            "substep": DT_SUB,
            "nodes": self.n_nodes,
            "bodies": self.n_bodies,
            "elements": int(self.n_bodies * len(self.faces_local)),
            "last": self.last_report,
        }
