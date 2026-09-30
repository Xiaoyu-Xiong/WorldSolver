"""Physically-based solver for scene 032_tablecloth_pull_2_mps.

Model
-----
* Cloth: 17x13 mass-spring (XPBD distance constraints) sheet of 221 nodes,
  structural + bending edges, hard kinematic grip on the x=-0.95 edge that
  is pulled in +x at ``pull_speed`` for ``pull_duration`` seconds.
* Rigid dinnerware (bottle, glass, plate, butter dish, fork, knife): each is
  one rigid translation sample (its centre) coupled to the cloth through
  analytic cylinder/box signed-distance contacts with mass-weighted position
  projections and Coulomb friction impulses.
* Table: static box (SDF projection) with friction against cloth and bodies.
* Integrator: fixed sub-step XPBD (h = 1/480 s), 8 constraint iterations.

All required trajectory fields are exported every sample time.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


# ---------------------------------------------------------------- helpers
def _box_sdf(p, c, h):
    """Signed distance of points p [M,3] to box centred c with half extents h."""
    q = np.abs(p - c) - h
    qpos = np.maximum(q, 0.0)
    outer = np.sqrt((qpos ** 2).sum(axis=1))
    inner = np.minimum(q.max(axis=1), 0.0)
    d = outer + inner
    s = np.where(p - c >= 0.0, 1.0, -1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_out = s * qpos / np.where(outer > 0, outer, 1.0)[:, None]
    k = q.argmax(axis=1)
    n_in = np.zeros_like(p)
    n_in[np.arange(len(p)), k] = s[np.arange(len(p)), k]
    n = np.where(outer[:, None] > 0.0, n_out, n_in)
    return d, n


def _cyl_sdf(p, c, r, hh):
    """Signed distance of points p [M,3] to z-aligned cylinder (c, r, hh)."""
    dx = p[:, 0] - c[0]
    dy = p[:, 1] - c[1]
    dz = p[:, 2] - c[2]
    rxy = np.maximum(np.sqrt(dx * dx + dy * dy), 1e-9)
    qr = rxy - r
    qz = np.abs(dz) - hh
    qpos_r = np.maximum(qr, 0.0)
    qpos_z = np.maximum(qz, 0.0)
    outer = np.sqrt(qpos_r ** 2 + qpos_z ** 2)
    qmax = np.maximum(qr, qz)
    d = outer + np.minimum(qmax, 0.0)
    sz = np.where(dz >= 0.0, 1.0, -1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = np.where(outer > 0, 1.0 / np.where(outer > 0, outer, 1.0), 0.0)
    n = np.empty_like(p)
    n[:, 0] = dx / rxy * qpos_r * sc
    n[:, 1] = dy / rxy * qpos_r * sc
    n[:, 2] = qpos_z * sz * sc
    radial = qr >= qz
    inside = outer <= 0.0
    n_in = np.zeros_like(p)
    n_in[:, 0] = np.where(radial, dx / rxy, 0.0)
    n_in[:, 1] = np.where(radial, dy / rxy, 0.0)
    n_in[:, 2] = np.where(radial, 0.0, sz)
    n = np.where(inside[:, None], n_in, n)
    return d, n


class BenchmarkSolver:
    # frozen scene constants (scene_geometry.yaml)
    TABLE_C = np.array([0.0, 0.0, 0.72])
    TABLE_H = np.array([1.25, 0.9, 0.06])
    TABLE_TOP = 0.78
    BODY_SHAPES = {
        1: ("cyl", 0.085, 0.36),
        2: ("cyl", 0.07, 0.15),
        3: ("cyl", 0.24, 0.0175),
        4: ("box", np.array([0.19, 0.11, 0.035]), None),
        5: ("box", np.array([0.16, 0.025, 0.012]), None),
        6: ("box", np.array([0.19, 0.022, 0.012]), None),
    }

    DT = 1.0 / 480.0
    ITERS = 8
    K_STRUCT = 2.0e5
    K_BEND = 10.0
    MU_CLOTH_TABLE = 0.30
    MU_CLOTH_BODY = 0.12
    MU_BODY_TABLE = 0.40
    CONTACT_TOL = 0.005
    NODE_R = 0.004  # cloth half-thickness

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        f0 = scene.initial_state()
        self.pos0 = f0.position.copy()
        self.x = f0.position.copy()
        self.v = np.array(f0.velocity, dtype=float)
        self.m = np.array(f0.mass, dtype=float)
        self.body = np.array(f0.body_id, dtype=int)
        self.ori = np.array(f0.orientation, dtype=float)
        self.uv = np.array(f0.attributes["uv"], dtype=float)
        self.frozen_scalars = {k: float(v) for k, v in f0.scalars.items()}
        self.n = len(self.x)

        self.gravity = np.array(cfg.get("gravity", [0, 0, -9.81]), dtype=float)
        self.pull_speed = float(cfg.get("pull_speed", 2.0))
        self.pull_duration = float(cfg.get("pull_duration", 0.9))

        self.w = np.where(self.m > 0, 1.0 / np.maximum(self.m, 1e-12), 0.0)

        # cloth / rigid partition
        self.cloth = np.where(self.body == 0)[0]
        self.body_ids = sorted(set(int(b) for b in self.body if b != 0))
        self.body_idx = {b: int(np.where(self.body == b)[0][0]) for b in self.body_ids}

        # kinematic grip: cloth edge with minimum initial x, pulled along +x
        cx = self.pos0[self.cloth, 0]
        xmin = cx.min()
        self.grip = self.cloth[cx <= xmin + 1e-6]
        self.w[self.grip] = 0.0
        self.grip_pos0 = self.pos0[self.grip].copy()
        self.pull_total = self.pull_speed * self.pull_duration

        # grid topology (17 x 13 row-major)
        nx, ny = 17, 13
        idx = np.arange(nx * ny).reshape(ny, nx)
        se = np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()])
        se2 = np.concatenate([idx[:, 2:].ravel(), idx[2:, :].ravel()])
        se1 = np.concatenate([idx[:, :-2].ravel(), idx[:-2, :].ravel()])
        struct = np.stack([np.concatenate([idx[:, :-1].ravel(), idx[:-1, :].ravel()]),
                           np.concatenate([idx[:, 1:].ravel(), idx[1:, :].ravel()])], axis=1)
        bend = np.stack([se1, se2], axis=1)

        def prep(edges, k):
            e0 = self.cloth[edges[:, 0]]
            e1 = self.cloth[edges[:, 1]]
            rest = np.linalg.norm(self.pos0[e1] - self.pos0[e0], axis=1)
            at = (1.0 / k) / self.DT ** 2
            return dict(e0=e0, e1=e1, rest=rest, at=at,
                        w0=self.w[e0], w1=self.w[e1],
                        lam=np.zeros(len(e0)))

        self.edge_sets = (prep(struct, self.K_STRUCT), prep(bend, self.K_BEND))

        self.t = 0.0
        self.s_prev = 0.0
        self.steps = 0

    # ---------------------------------------------------------------- pull
    def _pull_disp(self, t):
        return self.pull_speed * min(t, self.pull_duration)

    def _apply_grip(self, s):
        self.x[self.grip, 0] = self.grip_pos0[:, 0] + s
        self.x[self.grip, 1] = self.grip_pos0[:, 1]
        self.x[self.grip, 2] = self.grip_pos0[:, 2]

    # ---------------------------------------------------------------- contacts
    def _body_surface(self, b, p):
        """SDF & outward normal (pointing from body towards p) for body b."""
        shape = self.BODY_SHAPES[b]
        c = self.x[self.body_idx[b]]
        if shape[0] == "cyl":
            return _cyl_sdf(p, c, shape[1] + self.NODE_R, shape[2] + self.NODE_R)
        return _box_sdf(p, c, shape[1] + self.NODE_R)

    def _collide(self):
        # cloth vs table box
        pc = self.x[self.cloth]
        d, nn = _box_sdf(pc, self.TABLE_C, self.TABLE_H)
        pen = (d < 0.0) & (self.w[self.cloth] > 0)
        if pen.any():
            self.x[self.cloth[pen]] += nn[pen] * (-d[pen][:, None])
        # cloth vs bodies
        for b in self.body_ids:
            bi = self.body_idx[b]
            d, nn = self._body_surface(b, pc)
            pen = d < 0.0
            if not pen.any():
                continue
            wn = self.w[self.cloth][pen]
            wb = self.w[bi]
            tot = wn + wb
            frac_n = np.where(tot > 0, wn / np.maximum(tot, 1e-30), 0.0)
            frac_b = np.where(tot > 0, wb / np.maximum(tot, 1e-30), 0.0)
            delta = (-d[pen])[:, None]
            self.x[self.cloth[pen]] += nn[pen] * delta * frac_n[:, None]
            corr_b = (nn[pen] * delta * frac_b[:, None]).sum(axis=0)
            self.x[bi] -= corr_b
            pc = self.x[self.cloth]
        # bodies vs table top plane
        for b in self.body_ids:
            bi = self.body_idx[b]
            shape = self.BODY_SHAPES[b]
            hh = shape[2] if shape[0] == "cyl" else float(shape[1][2])
            rr = shape[1] if shape[0] == "cyl" else float(np.max(shape[1][:2]))
            c = self.x[bi]
            center_on = (abs(c[0]) <= self.TABLE_H[0]) and (abs(c[1]) <= self.TABLE_H[1])
            if center_on and c[2] - hh < self.TABLE_TOP:
                self.x[bi, 2] = self.TABLE_TOP + hh

    # ---------------------------------------------------------------- friction
    def _friction(self, h):
        v = self.v
        g_n = -self.gravity[2]
        pc = self.x[self.cloth]
        mc = self.m[self.cloth]
        wc = self.w[self.cloth]

        # cloth vs table
        d, nn = _box_sdf(pc, self.TABLE_C, self.TABLE_H)
        act = (d < self.CONTACT_TOL) & (wc > 0)
        if act.any():
            na = nn[act]
            va = v[self.cloth[act]]
            vn = (va * na).sum(axis=1, keepdims=True)
            vt = va - vn * na
            tv = np.linalg.norm(vt, axis=1)
            jn = mc[act] * (g_n * h * np.maximum(na[:, 2], 0.0) + np.maximum(-vn[:, 0], 0.0))
            dv = np.minimum(self.MU_CLOTH_TABLE * jn / mc[act], tv)
            dv = np.where(tv > 1e-12, dv, 0.0)
            v[self.cloth[act]] -= (vt / np.maximum(tv, 1e-12)[:, None]) * dv[:, None]

        for b in self.body_ids:
            bi = self.body_idx[b]
            mb = self.m[bi]
            wb = self.w[bi]
            # cloth nodes touching body
            d, nn = self._body_surface(b, self.x[self.cloth])
            act = (d < self.CONTACT_TOL) & (wc > 0)
            if act.any():
                na = nn[act]
                vrel = v[self.cloth[act]] - v[bi]
                vn = (vrel * na).sum(axis=1)
                vt = vrel - vn[:, None] * na
                tv = np.linalg.norm(vt, axis=1)
                wn = wc[act]
                meff = 1.0 / (wn + wb)
                jn = meff * (np.maximum(-vn, 0.0) + g_n * h) + (mb * g_n * h) / act.sum()
                jt = np.minimum(self.MU_CLOTH_BODY * jn, meff * tv)
                jt = np.where(tv > 1e-12, jt, 0.0)
                that = vt / np.maximum(tv, 1e-12)[:, None]
                v[self.cloth[act]] -= that * (jt * wn)[:, None]
                v[bi] += (that * (jt * wb)[:, None]).sum(axis=0)
            # body vs table
            shape = self.BODY_SHAPES[b]
            hh = shape[2] if shape[0] == "cyl" else float(shape[1][2])
            c = self.x[bi]
            if (c[2] - hh < self.TABLE_TOP + self.CONTACT_TOL
                    and abs(c[0]) <= self.TABLE_H[0] and abs(c[1]) <= self.TABLE_H[1]):
                vb = v[bi].copy()
                vtn = vb.copy()
                vtn[2] = 0.0
                tv = np.linalg.norm(vtn)
                jn = mb * (g_n * h + max(0.0, -vb[2]))
                dv = min(self.MU_BODY_TABLE * jn / mb, tv)
                if tv > 1e-12:
                    v[bi, :2] -= vtn[:2] / tv * dv

    # ---------------------------------------------------------------- stepping
    def _solve_edges(self):
        x = self.x
        for es in self.edge_sets:
            e0, e1 = es["e0"], es["e1"]
            dvec = x[e1] - x[e0]
            L = np.linalg.norm(dvec, axis=1)
            Ls = np.maximum(L, 1e-12)
            C = L - es["rest"]
            dlam = (C - es["at"] * es["lam"]) / (es["w0"] + es["w1"] + es["at"])
            dirv = dvec / Ls[:, None]
            np.add.at(x, e0, (es["w0"] * dlam)[:, None] * dirv)
            np.add.at(x, e1, -(es["w1"] * dlam)[:, None] * dirv)
            es["lam"] += dlam

    def _substep(self, h, t1):
        x, v = self.x, self.v
        s1 = self._pull_disp(t1)
        for es in self.edge_sets:
            es["lam"][:] = 0.0
        x_prev = x.copy()
        v += self.gravity * h
        v *= 0.9995
        x += v * h
        self._apply_grip(s1)
        for _ in range(self.ITERS):
            self._solve_edges()
            self._collide()
            self._apply_grip(s1)
        self.v = (self.x - x_prev) / h
        self._friction(h)
        self.v[self.grip, 0] = (s1 - self.s_prev) / h
        self.v[self.grip, 1:] = 0.0
        self.s_prev = s1
        self.t = t1
        self.steps += 1

    def advance_to(self, time: float) -> StateFrame:
        time = float(time)
        guard = 0
        while self.t < time - 1e-9 and guard < 100000:
            h = min(self.DT, time - self.t)
            self._substep(h, self.t + h)
            guard += 1
        return self._frame(time)

    # ---------------------------------------------------------------- output
    def _contact_distances(self):
        pc = self.x[self.cloth]
        d, _ = _box_sdf(pc, self.TABLE_C, self.TABLE_H)
        body_d = np.full((len(self.body_ids), self.n), np.inf)
        for j, b in enumerate(self.body_ids):
            db, _ = self._body_surface(b, pc)
            body_d[j, self.cloth] = db
            shape = self.BODY_SHAPES[b]
            bi = self.body_idx[b]
            c = self.x[bi]
            hh = shape[2] if shape[0] == "cyl" else float(shape[1][2])
            # distance from body bottom to cloth support / table
            if shape[0] == "cyl":
                dx = pc[:, 0] - c[0]
                dy = pc[:, 1] - c[1]
                foot = (dx * dx + dy * dy) <= shape[1] ** 2
            else:
                hx = shape[1]
                foot = (np.abs(pc[:, 0] - c[0]) <= hx[0]) & (np.abs(pc[:, 1] - c[1]) <= hx[1])
            if foot.any():
                d_sup = (c[2] - hh) - pc[foot, 2].max()
            else:
                d_sup = np.inf
            d_tab = (c[2] - hh) - self.TABLE_TOP if (
                abs(c[0]) <= self.TABLE_H[0] and abs(c[1]) <= self.TABLE_H[1]) else np.inf
            body_d[j, bi] = min(d_sup, d_tab)
        allb = body_d.min(axis=0)
        dmin = np.full(self.n, np.inf)
        dmin[self.cloth] = np.minimum(d, allb[self.cloth])
        for b in self.body_ids:
            dmin[self.body_idx[b]] = allb[self.body_idx[b]]
        dmin[~np.isfinite(dmin)] = 1.0
        return np.clip(dmin, -1.0, 1.0)

    def _frame(self, t):
        s = self._pull_disp(t)
        gdisp = self.x[self.grip] - self.grip_pos0
        cval = np.array([gdisp[:, 0].mean(), gdisp[:, 1].mean()])
        ctar = np.array([s, 0.0])
        ext = self.m[:, None] * self.gravity[None, :]
        return StateFrame(
            time=float(t),
            position=self.x.copy(),
            velocity=self.v.copy(),
            orientation=self.ori.copy(),
            angular_velocity=np.zeros((self.n, 3)),
            mass=self.m.copy(),
            body_id=self.body.copy(),
            attributes={"mass_quantity": self.m.copy(), "uv": self.uv.copy()},
            fields={
                "cumulative_mass_source": np.zeros(1),
                "contact_signed_distance": self._contact_distances(),
                "constraint_value": cval,
                "constraint_target": ctar,
                "external_force": ext,
            },
            scalars=dict(self.frozen_scalars),
        )

    def diagnostics(self):
        return {"steps": self.steps, "time": self.t}
