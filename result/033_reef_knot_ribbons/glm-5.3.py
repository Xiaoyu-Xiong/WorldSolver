"""Physics solver for scene 033_reef_knot_ribbons.

Two flat ribbons tied in a reef-knot interlace are tightened by scripted end
pulling under Coulomb friction (mu = 0.02) and a strict strain limit
(strain_upper = 1.134, symmetric lower bound).  Each ribbon is a 29x5
node/face shell mesh.  The dynamics use a position-based dynamics scheme:

    predict -> project constraints (strain limit, elastic edges, contact,
    friction) -> velocity update,

with sub-stepping (h = 1/900 s).  Contact is point/triangle between and
within ribbons with a thickness gap d = 0.006 m; penetration is projected
out with a PBD point-triangle correction, a smooth anticipation barrier
prevents gap collapse, and friction is a position-level filter clamping the
tangential relative slip of a contact to mu times the accumulated normal
correction of the substep.

The end rows of each ribbon are kinematically pulled outward along the
ribbon axis at pull_speed.  Because the ribbons are nearly inextensible
(only ~4% slack plus the strain allowance), the scripted pull is capped by
a tautness controller: pulling pauses when the maximum axial edge strain
reaches a threshold below the frozen strain limit and resumes if the ribbon
goes slack again, so the knot tightens as far as the material allows
without violating the strain limit.

All required StateFrame trajectory fields are exported:
  attributes.deformation_gradient   [N,3,3] per-node (element-averaged)
  attributes.mass_quantity           [N]     per-node mass
  fields.contact_signed_distance     [C]     signed gaps at near contacts
  fields.cumulative_mass_source      [K]     zeros (no mass sources)
  scalars.contact_length_scale       scalar  ribbon thickness
plus fields.element_deformation_gradient [Ne,3,3].
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-12


def _norm(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt((v * v).sum(axis=axis))


def _point_triangle(p, t0, t1, t2):
    """Closest point of triangles to points (vectorised, Ericson barycentric).

    p, t0, t1, t2 : (M,3).  Returns (dist (M,), closest (M,3), bary (M,3)).
    """
    ab = t1 - t0
    ac = t2 - t0
    ap = p - t0
    d1 = (ab * ap).sum(1)
    d2 = (ac * ap).sum(1)
    bp = p - t1
    d3 = (ab * bp).sum(1)
    d4 = (ac * bp).sum(1)
    cp = p - t2
    d5 = (ab * cp).sum(1)
    d6 = (ac * cp).sum(1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    safe = np.where(np.abs(denom) < _EPS, _EPS, denom)
    v = vb / safe
    w = vc / safe
    bary = np.stack([1.0 - v - w, v, w], axis=1)

    def _ratio(num, den):
        dd = np.where(np.abs(den) < _EPS, _EPS, den)
        return np.clip(num / dd, 0.0, 1.0)

    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    if m.any():
        t = _ratio(d1[m], d1[m] - d3[m])
        bary[m] = np.stack([1.0 - t, t, np.zeros_like(t)], axis=1)
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    if m.any():
        t = _ratio(d2[m], d2[m] - d6[m])
        bary[m] = np.stack([1.0 - t, np.zeros_like(t), t], axis=1)
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    if m.any():
        t = _ratio(d4[m] - d3[m], (d4[m] - d3[m]) + (d5[m] - d6[m]))
        bary[m] = np.stack([np.zeros_like(t), 1.0 - t, t], axis=1)
    # vertex regions take priority
    m = (d1 <= 0) & (d2 <= 0)
    if m.any():
        bary[m] = np.array([1.0, 0.0, 0.0])
    m = (d3 >= 0) & (d4 <= d3)
    if m.any():
        bary[m] = np.array([0.0, 1.0, 0.0])
    m = (d6 >= 0) & (d5 <= d6)
    if m.any():
        bary[m] = np.array([0.0, 0.0, 1.0])
    bary = np.clip(bary, 0.0, 1.0)
    closest = bary[:, 0:1] * t0 + bary[:, 1:2] * t1 + bary[:, 2:3] * t2
    dist = _norm(p - closest)
    return dist, closest, bary


class _Body:
    """Per-ribbon mesh, constraint and scripted-pull data."""

    def __init__(self, body_id, global_idx, ni, nj, x0):
        self.body_id = body_id
        self.idx = np.asarray(global_idx, dtype=np.int64)  # local -> global
        n_loc = len(self.idx)
        assert n_loc == ni * nj

        def g(loc):
            return self.idx[np.asarray(loc, dtype=np.int64)]

        # --- constraint edges (local index = i + ni * j) ---
        i_idx = []
        j_idx = []
        for i in range(ni - 1):
            for j in range(nj):
                i_idx.append((i + ni * j, i + 1 + ni * j))
        for i in range(ni):
            for j in range(nj - 1):
                i_idx.append((i + ni * j, i + ni * (j + 1)))
        shear = []
        for i in range(ni - 1):
            for j in range(nj - 1):
                shear.append((i + ni * j, i + 1 + ni * (j + 1)))
                shear.append((i + 1 + ni * j, i + ni * (j + 1)))
        bend = []
        for i in range(ni - 2):
            for j in range(nj):
                bend.append((i + ni * j, i + 2 + ni * j))
        for i in range(ni):
            for j in range(nj - 2):
                bend.append((i + ni * j, i + ni * (j + 2)))
        # axial (length-direction) edges, used by the tautness controller
        axial = []
        for i in range(ni - 1):
            for j in range(nj):
                axial.append((i + ni * j, i + 1 + ni * j))

        def build(pairs):
            a = g([p[0] for p in pairs])
            b = g([p[1] for p in pairs])
            rest = _norm(x0[a] - x0[b])
            return a, b, rest

        self.struct = build(i_idx)
        self.shear = build(shear)
        self.bend = build(bend)
        self.axial = build(axial)

        # --- scripted pulling ends ---
        jm = nj // 2
        cl = x0[g([i + ni * jm for i in range(ni)])]
        mid = cl[ni // 2]
        d0 = cl[0] - mid
        d1 = cl[-1] - mid
        self.dir_a = d0 / max(_norm(d0), _EPS)
        self.dir_b = d1 / max(_norm(d1), _EPS)
        self.pin_a = g([ni * j for j in range(nj)])
        self.pin_b = g([(ni - 1) + ni * j for j in range(nj)])
        self.pull = 0.0
        self.pulling = True


class BenchmarkSolver:
    """PBD shell solver with strain limiting, contact and friction."""

    def __init__(self):
        self.t = 0.0
        self._frames_built = 0

    # ------------------------------------------------------------------ init
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        frame = scene.initial_state()

        self.x = np.array(frame.position, dtype=np.float64)
        self.v = np.array(frame.velocity, dtype=np.float64)
        n = len(self.x)
        self.n = n
        if frame.mass is not None:
            self.mass = np.array(frame.mass, dtype=np.float64)
        else:
            self.mass = np.ones(n)
        if frame.body_id is not None:
            self.body_id = np.array(frame.body_id, dtype=np.int64)
        else:
            self.body_id = np.zeros(n, dtype=np.int64)
        self.inv_m = 1.0 / np.maximum(self.mass, _EPS)
        self.x0 = self.x.copy()

        self.gravity = np.array(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=np.float64)
        ribbon = dict(cfg.get("ribbon", {}))
        self.ni, self.nj = (int(v) for v in ribbon.get("grid_shape", [29, 5]))
        self.thickness = float(ribbon.get("thickness", 0.006))
        self.strain_hi = float(ribbon.get("strain_upper", 1.134))
        self.strain_lo = 1.0 / self.strain_hi
        self.mu = float(ribbon.get("friction", 0.02))
        self.pull_speed = float(cfg.get("pull_speed", 0.32))

        # numerics (frozen, deterministic)
        self.h = 1.0 / 900.0
        self.iters = 8
        self.damping = 1.0e-4
        self.v_max = 30.0
        self.relax_spring = 0.35
        self.relax_strain = 0.85
        self.relax_contact = 0.9
        self.relax_friction = 0.9
        self.k_spring = {0: 0.5, 1: 0.25, 2: 0.08}  # struct, shear, bend
        self.d_contact = self.thickness
        self.g0 = 2.0 * self.thickness          # anticipation barrier width
        self.k_barrier = 0.25
        self.broad_r = 0.2
        self.report_gap = 0.04
        # tautness controller thresholds (below frozen strain limit)
        self.stop_strain = 1.10
        self.resume_strain = 1.06

        meta = dict(scene.metadata())
        faces_local = np.asarray(meta["topology"]["faces"], dtype=np.int64)

        # per-body data
        self.bodies = []
        for b in sorted(set(int(b) for b in self.body_id)):
            gidx = np.where(self.body_id == b)[0]
            self.bodies.append(_Body(b, gidx, self.ni, self.nj, self.x0))

        # contact sets: (vertex body, triangle body)
        pairs = [(0, 1), (1, 0), (0, 0), (1, 1)]
        self.contact_sets = []
        for vb, tb in pairs:
            verts = self.bodies[vb].idx
            tri = self.bodies[tb].idx[faces_local]
            excl = None
            if vb == tb:
                # exclude triangles incident on the query vertex
                excl = np.zeros((len(verts), len(tri)), dtype=bool)
                for k in range(3):
                    excl |= tri[None, :, k] == verts[:, None]
            self.contact_sets.append(
                {"verts": verts, "tri": tri, "excl": excl, "vb": vb, "tb": tb}
            )

        # per-element deformation gradient rest frames
        all_faces = []
        for b in self.bodies:
            all_faces.append(b.idx[faces_local])
        self.faces = np.concatenate(all_faces, axis=0)          # (Ne,3)
        self.n_elements = len(self.faces)
        a0 = self.x0[self.faces[:, 1]] - self.x0[self.faces[:, 0]]
        b0 = self.x0[self.faces[:, 2]] - self.x0[self.faces[:, 0]]
        n0 = np.cross(a0, b0)
        area2 = _norm(n0)
        self.rest_area = 0.5 * area2
        n0 = n0 / np.maximum(area2, _EPS)[:, None]
        m_rest = np.stack([a0, b0, n0], axis=-1)                # columns
        self.rest_frame_inv = np.linalg.inv(m_rest)
        self.faces_flat = self.faces.reshape(-1)

        self.t = 0.0
        self._last_diag = {}

    # --------------------------------------------------------------- helpers
    def _max_axial_strain(self, body) -> float:
        ea, eb, rest = body.axial
        ln = _norm(self.x[ea] - self.x[eb])
        return float(np.max(ln / rest))

    def _detect(self):
        """Broadphase contact candidates at the current positions."""
        x = self.x
        out = []
        r2 = self.broad_r * self.broad_r
        for cs in self.contact_sets:
            V = x[cs["verts"]]
            T = x[cs["tri"]]
            C = T.mean(axis=1)
            diff = V[:, None, :] - C[None, :, :]
            d2 = (diff * diff).sum(-1)
            mask = d2 < r2
            if cs["excl"] is not None:
                mask &= ~cs["excl"]
            vl, fi = np.nonzero(mask)
            out.append(
                {
                    "verts": cs["verts"],
                    "tri": cs["tri"],
                    "vi": cs["verts"][vl],
                    "fi": fi,
                    "lam": np.zeros(len(fi), dtype=np.float64),
                }
            )
        return out

    def _apply_pins(self):
        for b in self.bodies:
            s = b.pull
            self.x[b.pin_a] = self.x0[b.pin_a] + b.dir_a * s
            self.x[b.pin_b] = self.x0[b.pin_b] + b.dir_b * s

    def _pin_velocities(self):
        for b in self.bodies:
            spd = self.pull_speed if b.pulling else 0.0
            self.v[b.pin_a] = b.dir_a * spd
            self.v[b.pin_b] = b.dir_b * spd

    # ------------------------------------------------------------- dynamics
    def _spring_pass(self, edges, k):
        ea, eb, rest = edges
        x = self.x
        d = x[eb] - x[ea]
        ln = _norm(d)
        n = d / np.maximum(ln, _EPS)[:, None]
        err = ln - rest
        wsum = self.inv_m[ea] + self.inv_m[eb]
        corr = n * (err * k)[:, None]
        dx = np.zeros_like(x)
        np.add.at(dx, ea, corr * (self.inv_m[ea] / wsum)[:, None])
        np.add.at(dx, eb, -corr * (self.inv_m[eb] / wsum)[:, None])
        x += dx * self.relax_spring

    def _strain_pass(self, edges):
        ea, eb, rest = edges
        x = self.x
        d = x[eb] - x[ea]
        ln = _norm(d)
        n = d / np.maximum(ln, _EPS)[:, None]
        hi = rest * self.strain_hi
        lo = rest * self.strain_lo
        target = np.where(ln > hi, hi, np.where(ln < lo, lo, ln))
        err = ln - target
        wsum = self.inv_m[ea] + self.inv_m[eb]
        corr = n * err[:, None]
        dx = np.zeros_like(x)
        np.add.at(dx, ea, corr * (self.inv_m[ea] / wsum)[:, None])
        np.add.at(dx, eb, -corr * (self.inv_m[eb] / wsum)[:, None])
        x += dx * self.relax_strain

    def _contact_pass(self, cands, accumulate):
        x = self.x
        d_contact = self.d_contact
        for c in cands:
            vi = c["vi"]
            fi = c["fi"]
            if vi.size == 0:
                continue
            tri = c["tri"]
            P = x[vi]
            T0 = x[tri[fi, 0]]
            T1 = x[tri[fi, 1]]
            T2 = x[tri[fi, 2]]
            dist, cp, bary = _point_triangle(P, T0, T1, T2)
            diff = P - cp
            nd = _norm(diff)
            n = diff / np.maximum(nd, _EPS)[:, None]
            fn = np.cross(T1 - T0, T2 - T0)
            fn = fn / np.maximum(_norm(fn), _EPS)[:, None]
            bad = nd < _EPS
            if bad.any():
                sgn = np.sign((fn * (P - T0)).sum(1))
                sgn = np.where(sgn == 0.0, 1.0, sgn)
                n[bad] = (fn * sgn[:, None])[bad]

            gap = dist - d_contact
            pen = gap < 0.0
            wp = self.inv_m[vi]
            wk = self.inv_m[tri[fi]]
            denom = np.maximum(wp + (bary * bary * wk).sum(1), _EPS)
            tidx = tri[fi]

            # penetration pushout
            C = np.where(pen, -gap, 0.0)
            lam = C / denom
            if accumulate:
                c["lam"] += lam
            np.add.at(x, vi, n * (lam * wp * self.relax_contact)[:, None])
            dxt = -n[:, None, :] * (lam[:, None] * bary * wk)[:, :, None]
            np.add.at(x, tidx.reshape(-1), (dxt * self.relax_contact).reshape(-1, 3))

            # smooth anticipation barrier for near contacts
            m = (~pen) & (gap < self.g0)
            if m.any():
                gg = gap[m]
                push = self.k_barrier * (1.0 - gg / self.g0) ** 2 * self.g0
                Cb = np.zeros_like(C)
                Cb[m] = push
                lamb = Cb / denom
                if accumulate:
                    c["lam"] += 0.5 * lamb
                np.add.at(x, vi, n * (lamb * wp * self.relax_contact)[:, None])
                dxb = -n[:, None, :] * (lamb[:, None] * bary * wk)[:, :, None]
                np.add.at(
                    x, tidx.reshape(-1), (dxb * self.relax_contact).reshape(-1, 3)
                )

    def _friction_pass(self, cands, xs):
        x = self.x
        for c in cands:
            vi = c["vi"]
            fi = c["fi"]
            if vi.size == 0:
                continue
            tri = c["tri"]
            P = x[vi]
            P0 = xs[vi]
            T0 = x[tri[fi, 0]]
            T1 = x[tri[fi, 1]]
            T2 = x[tri[fi, 2]]
            dist, cp, bary = _point_triangle(P, T0, T1, T2)
            diff = P - cp
            nd = _norm(diff)
            n = diff / np.maximum(nd, _EPS)[:, None]
            fn = np.cross(T1 - T0, T2 - T0)
            fn = fn / np.maximum(_norm(fn), _EPS)[:, None]
            bad = nd < _EPS
            if bad.any():
                sgn = np.sign((fn * (P - T0)).sum(1))
                sgn = np.where(sgn == 0.0, 1.0, sgn)
                n[bad] = (fn * sgn[:, None])[bad]

            gap = dist - self.d_contact
            act = gap < self.g0
            tidx = tri[fi]
            d_tri = x[tidx] - xs[tidx]
            d = (P - P0) - (bary[:, :, None] * d_tri).sum(1)
            dn = (d * n).sum(1)
            dt = d - dn[:, None] * n
            l = _norm(dt)
            allow = self.mu * c["lam"]
            m = act & (l > allow) & (l > _EPS)
            if not m.any():
                continue
            D = np.zeros_like(d)
            D[m] = dt[m] * ((allow[m] / l[m] - 1.0)[:, None])
            wp = self.inv_m[vi]
            wk = self.inv_m[tidx]
            denom = np.maximum(wp + (bary * bary * wk).sum(1), _EPS)
            np.add.at(
                x, vi, D * (wp / denom)[:, None] * self.relax_friction
            )
            Dk = -D[:, None, :] * ((bary * wk) / denom[:, None])[:, :, None]
            np.add.at(x, tidx.reshape(-1), (Dk * self.relax_friction).reshape(-1, 3))

    def _substep(self, h):
        # tautness controller for the scripted end pulling
        for b in self.bodies:
            s_max = self._max_axial_strain(b)
            if b.pulling and s_max > self.stop_strain:
                b.pulling = False
            elif (not b.pulling) and s_max < self.resume_strain:
                b.pulling = True
            if b.pulling:
                b.pull += self.pull_speed * h

        x_prev = self.x.copy()
        if np.any(self.gravity != 0.0):
            self.v = self.v + self.gravity * h
        self.v *= 1.0 - self.damping
        self.x = self.x + self.v * h

        xs = self.x.copy()
        cands = self._detect()

        for _ in range(self.iters):
            for b in self.bodies:
                self._strain_pass(b.struct)
                self._spring_pass(b.shear, self.k_spring[1])
                self._spring_pass(b.bend, self.k_spring[2])
                self._spring_pass(b.struct, self.k_spring[0])
            self._contact_pass(cands, accumulate=True)
            self._apply_pins()

        self._friction_pass(cands, xs)
        self._apply_pins()

        self.v = (self.x - x_prev) / h
        self._pin_velocities()
        sp = _norm(self.v)
        fac = np.where(sp > self.v_max, self.v_max / np.maximum(sp, _EPS), 1.0)
        self.v = self.v * fac[:, None]
        self.t += h

    # ------------------------------------------------------------- reporting
    def _contact_gaps(self):
        x = self.x
        r2 = self.broad_r * self.broad_r
        gaps = []
        for cs in self.contact_sets:
            V = x[cs["verts"]]
            T = x[cs["tri"]]
            C = T.mean(axis=1)
            diff = V[:, None, :] - C[None, :, :]
            d2 = (diff * diff).sum(-1)
            mask = d2 < r2
            if cs["excl"] is not None:
                mask &= ~cs["excl"]
            vl, fi = np.nonzero(mask)
            if vl.size == 0:
                continue
            P = V[vl]
            T0 = T[fi, 0]
            T1 = T[fi, 1]
            T2 = T[fi, 2]
            dist, _, _ = _point_triangle(P, T0, T1, T2)
            gap = dist - self.d_contact
            sel = gap < self.report_gap
            if np.any(sel):
                gaps.append(gap[sel])
        if gaps:
            return np.concatenate(gaps)
        return np.zeros(0, dtype=np.float64)

    def _deformation_gradients(self):
        x = self.x
        a = x[self.faces[:, 1]] - x[self.faces[:, 0]]
        b = x[self.faces[:, 2]] - x[self.faces[:, 0]]
        nn = np.cross(a, b)
        nn = nn / np.maximum(_norm(nn), _EPS)[:, None]
        W = np.stack([a, b, nn], axis=-1)
        F_elem = W @ self.rest_frame_inv  # (Ne,3,3)
        # node-averaged (rest-area weighted)
        F_node = np.zeros((self.n, 3, 3))
        wsum = np.zeros(self.n)
        w = self.rest_area
        np.add.at(F_node, self.faces_flat, np.repeat(F_elem * w[:, None, None], 3, axis=0))
        np.add.at(wsum, self.faces_flat, np.repeat(w, 3))
        F_node = F_node / np.maximum(wsum, 1e-30)[:, None, None]
        return F_elem, F_node

    def _build_frame(self, time=None) -> StateFrame:
        frame_time = float(self.t if time is None else time)
        F_elem, F_node = self._deformation_gradients()
        gaps = self._contact_gaps()
        attrs = {
            "deformation_gradient": F_node,
            "mass_quantity": self.mass,
        }
        fields = {
            "contact_signed_distance": gaps,
            "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            "element_deformation_gradient": F_elem,
        }
        scalars = {
            "contact_length_scale": float(self.thickness),
        }
        self._last_diag = {
            "time": float(self.t),
            "min_contact_gap": float(gaps.min()) if gaps.size else None,
            "n_contact_samples": int(gaps.size),
            "pull": [float(b.pull) for b in self.bodies],
            "pulling": [bool(b.pulling) for b in self.bodies],
            "max_axial_strain": [self._max_axial_strain(b) for b in self.bodies],
        }
        return StateFrame(
            time=frame_time,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )

    # ------------------------------------------------------------------ api
    def advance_to(self, time: float):
        target = float(time)
        if target < self.t:
            target = self.t
        while target - self.t > 1e-9:
            h = min(self.h, target - self.t)
            self._substep(h)
        return self._build_frame(time=target)

    def diagnostics(self) -> dict:
        return dict(self._last_diag)
