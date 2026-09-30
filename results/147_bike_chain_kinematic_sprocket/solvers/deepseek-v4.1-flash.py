"""Physical solver for scene 147 ``bike_chain_kinematic_sprocket``.

Model
-----
Planar (x-z) rigid body dynamics of a closed chain of 20 pin jointed rigid
links looping around two toothed sprockets.  Both sprocket centres are pinned
to the world (their axles are static boundary objects); the driven sprocket
carries a prescribed constant torque, the other one is a free follower.  All
bodies feel gravity, the sprockets carry a small viscous bearing drag.

Numerics
--------
* semi implicit (symplectic) Euler for the free motion,
* velocity level constraint solve: joint and contact Jacobian rows are
  assembled each step and the small dense effective mass system
  ``J M^-1 J^T`` is solved directly (Cholesky) with a few projected
  iterations for Coulomb friction,
* Baumgarte stabilisation with clamped bias velocity plus restitution,
* position level (PBD style) projection of the chain pin joints every step,
  and a one-off projection of the initial state (joints + contacts) so the
  simulation starts from a consistent, non penetrating configuration.

Collision geometry (derived from the frozen assets and configuration)
--------------------------------------------------------------------
* a link is a capsule: axis segment half length 0.145 with end cap radius
  0.055 and pin joints at local ``x = +/-0.2`` (the mesh pin bores), pitch
  0.4 as in ``assets/chain_link.obj``,
* a sprocket is the toothed profile extracted from ``assets/sprocket.obj``,
  radially scaled so that the chain roll centres of the taut wrap coincide
  with the tooth crest radius: ``R_crest = (L - slack - 2*D)/(2*pi) - r_cap``.
"""

from __future__ import annotations

import math
import os

import numpy as np

from worldsolver.contracts import StateFrame

try:
    from scipy.linalg import cho_factor, cho_solve

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - NumPy fallback
    _HAVE_SCIPY = False

# ----------------------------------------------------------------------------
# frozen / physical constants
# ----------------------------------------------------------------------------
_DT = 5.0e-4            # integration time step [s]
_BETA = 0.2             # Baumgarte position feedback factor
_SLOP = 5.0e-4          # allowed penetration [m]
_BIAS_VMAX = 0.3        # maximum position-correction velocity [m/s]
_GEN_MARGIN = 4.0e-3    # contact generation margin [m]
_MAX_EDGES = 4          # contact edges kept per link/sprocket pair
_MAX_CONTACTS = 90      # hard cap on the number of contact points
_MAX_LIN_VEL = 30.0
_MAX_ANG_VEL = 80.0
_DRIVE_TORQUE = 6.0     # prescribed drive torque on the driven sprocket [N m]
_BEARING_DRAG = 1.0     # viscous bearing coefficient [N m s]
_PIN = 0.2              # pin anchors at local (+/-0.2, 0)
_CAP_R = 0.055          # link capsule (roller) radius [m]
_LINK_HALF = 0.145      # capsule axis half length [m]
_SLACK = 0.08           # chain slack of the taut wrap [m]
_INIT_PROJ_ITERS = 220  # initial state projection sweeps
_STEP_PROJ_ITERS = 6    # per step joint projection sweeps


# ----------------------------------------------------------------------------
# small vectorised 2-D helpers.  Planar coordinates are (x, z) of the world
# (x, y, z) frame; rotations act about world +y with the convention
#   R(t) (a, b) = (a cos t + b sin t, -a sin t + b cos t)
# so that the local +x axis maps to (cos t, -sin t).  A body spinning with
# angular velocity w moves a material point at world offset r with velocity
# w * perp(r), where perp(r) = (r_z, -r_x).
# ----------------------------------------------------------------------------
def _rot(theta, vec):
    """Rotate local (a, b) vectors by theta (broadcasting aware)."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.stack([vec[..., 0] * c + vec[..., 1] * s,
                     -vec[..., 0] * s + vec[..., 1] * c], axis=-1)


def _cross2(r, f):
    """y component of the 3-D cross product r x f for in-plane vectors."""
    return r[..., 0] * f[..., 1] - r[..., 1] * f[..., 0]


def _perpendicular(r):
    return np.stack([r[..., 1], -r[..., 0]], axis=-1)


def _closest_on_seg(p, q1, q2):
    d = q2 - q1
    l2 = np.sum(d * d, axis=-1, keepdims=True)
    l2 = np.where(l2 < 1.0e-18, 1.0, l2)
    t = np.sum((p - q1) * d, axis=-1, keepdims=True) / l2
    t = np.clip(t, 0.0, 1.0)
    return q1 + t * d


def _seg_seg(p1, p2, q1, q2):
    """Closest points between two batches of 2-D segments (exact in 2-D)."""
    ca1 = _closest_on_seg(p1, q1, q2)
    ca2 = _closest_on_seg(p2, q1, q2)
    cb1 = _closest_on_seg(q1, p1, p2)
    cb2 = _closest_on_seg(q2, p1, p2)
    d = np.stack([np.linalg.norm(p1 - ca1, axis=-1),
                  np.linalg.norm(p2 - ca2, axis=-1),
                  np.linalg.norm(q1 - cb1, axis=-1),
                  np.linalg.norm(q2 - cb2, axis=-1)], axis=-1)
    lp = np.stack([np.broadcast_to(p1, ca1.shape), np.broadcast_to(p2, ca2.shape),
                   cb1, cb2], axis=-2)
    ep = np.stack([ca1, ca2, np.broadcast_to(q1, cb1.shape),
                   np.broadcast_to(q2, cb2.shape)], axis=-2)
    k = np.argmin(d, axis=-1)
    dmin = np.take_along_axis(d, k[..., None], axis=-1)[..., 0]
    lpt = np.take_along_axis(lp, k[..., None, None], axis=-2)[..., 0, :]
    ept = np.take_along_axis(ep, k[..., None, None], axis=-2)[..., 0, :]
    return dmin, lpt, ept


def _points_in_poly(points, poly):
    """Vectorised ray-casting point in polygon test."""
    x = points[:, 0][:, None]
    y = points[:, 1][:, None]
    x1 = poly[None, :, 0]
    y1 = poly[None, :, 1]
    x2 = np.roll(poly, -1, axis=0)[None, :, 0]
    y2 = np.roll(poly, -1, axis=0)[None, :, 1]
    cond = (y1 > y) != (y2 > y)
    dy = np.where(np.abs(y2 - y1) < 1.0e-15, 1.0e-15, y2 - y1)
    xint = (x2 - x1) * (y - y1) / dy + x1
    hits = np.sum(cond & (x < xint), axis=1)
    return (hits % 2) == 1


def _load_profile(path_candidates):
    """Extract the ordered toothed profile (x, z) of the sprocket mesh."""
    for path in path_candidates:
        try:
            verts = []
            with open(path, "r") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) == 4 and parts[0] == "v":
                        verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            if not verts:
                continue
            v = np.asarray(verts, dtype=float)
            front = v[v[:, 1] > 0.0]
            if len(front) < 8:
                continue
            rad = np.hypot(front[:, 0], front[:, 2])
            out = []
            for a in range(len(front)):
                out.append((math.atan2(front[a, 2], front[a, 0]), rad[a], front[a, 0], front[a, 2]))
            out.sort()
            pts = []
            for ang, r, px, pz in out:
                if r < 1.0e-6:
                    continue
                if pts and abs(ang - pts[-1][0]) < 1.0e-9:
                    if r > pts[-1][1]:
                        pts[-1] = (ang, r, px, pz)
                    continue
                pts.append((ang, r, px, pz))
            if len(pts) >= 8:
                prof = np.asarray([[p[2], p[3]] for p in pts], dtype=float)
                handle_max = float(np.max(np.hypot(prof[:, 0], prof[:, 1])))
                return prof, handle_max
        except Exception:
            continue
    return None


def _analytic_profile(rmax, teeth=14, samples=336):
    """Fallback 14 tooth profile with crest radius rmax."""
    ang = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    phase = np.mod(ang, 2.0 * math.pi / teeth) / (2.0 * math.pi / teeth)
    r = np.where(phase < 0.5, rmax, 0.794 * rmax)
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=-1)


class BenchmarkSolver:
    """Deterministic time stepping solver (see module docstring)."""

    # -- construction --------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        self.cfg = cfg
        self.seed = int(seed)
        self.dt = _DT
        self.friction = float(cfg["contact"]["friction"])
        self.restitution = float(cfg["contact"]["restitution"])
        self.length_scale = float(cfg["contact"]["length_scale"])
        self.gravity = np.asarray(cfg["gravity"], dtype=float)

        frame0 = scene.initial_state()
        pos3 = np.asarray(frame0.position, dtype=float)
        mass = np.asarray(frame0.mass, dtype=float)
        self.n_body = int(pos3.shape[0])
        self.body_id = np.asarray(frame0.body_id, dtype=np.int32)
        self.mass = mass
        ids = [str(b["id"]) for b in cfg["bodies"]]
        self.ids = ids

        self.links = np.array([i for i, s in enumerate(ids) if s.startswith("chain_")], dtype=int)
        self.sprockets = np.array([i for i, s in enumerate(ids) if s.startswith("sprocket_")], dtype=int)
        n_link = int(self.links.size)
        if n_link < 3 or self.sprockets.size < 1:
            raise ValueError("scene must contain a chain of links and at least one sprocket")

        # driven / follower sprocket
        if "sprocket_right" in ids:
            self.driven = ids.index("sprocket_right")
        else:
            self.driven = int(self.sprockets[np.argmax(pos3[self.sprockets, 0])])
        self.follower = [int(k) for k in self.sprockets if int(k) != int(self.driven)]

        # planar state -------------------------------------------------------
        nb = self.n_body + 1                       # last slot: static ground
        self.nb = nb
        self.pos = np.zeros((nb, 2))
        self.pos[: self.n_body] = pos3[:, [0, 2]]
        self.theta = np.zeros(nb)
        self.vel = np.zeros((nb, 3))               # (vx, vz, omega)
        self.inv_m = np.zeros(nb)
        self.inv_i = np.zeros(nb)
        self.inv_m[: self.n_body] = np.where(mass > 0.0, 1.0 / np.maximum(mass, 1.0e-9), 0.0)
        for i in range(self.n_body):               # sprockets keep a fixed centre
            if i in set(int(k) for k in self.sprockets):
                self.inv_m[i] = 0.0
        # link inertia: rigid capsule of the frozen mesh (0.4 x 0.11 section)
        i_link = mass[self.links] * (0.4 ** 2 + 0.11 ** 2) / 12.0
        self.inv_i[self.links] = 1.0 / i_link
        self.link_i = i_link

        # sprocket geometry --------------------------------------------------
        here = os.path.dirname(os.path.abspath(__file__))
        cands = ["assets/sprocket.obj",
                 os.path.join(os.getcwd(), "assets", "sprocket.obj"),
                 os.path.join(here, "..", "assets", "sprocket.obj"),
                 os.path.join(here, "..", "..", "assets", "sprocket.obj")]
        loaded = _load_profile(cands)
        if loaded is None:
            raw_prof, rmax = _analytic_profile(0.68), 0.68
        else:
            raw_prof, rmax = loaded[0], loaded[1]

        # chain length of the frozen links and required taut wrap radius
        pitch = 2.0 * _PIN
        chain_len = pitch * n_link
        if self.sprockets.size >= 2:
            centres = self.pos[self.sprockets]
            span = 0.0
            for i in range(self.sprockets.size):
                for j in range(i + 1, self.sprockets.size):
                    span = max(span, float(np.linalg.norm(centres[i] - centres[j])))
        else:
            span = 0.0
        r_wrap = max(0.12, (chain_len - _SLACK - 2.0 * span) / (2.0 * math.pi))
        # A polygonal chain of solid capsule links cuts inside its tangent circle
        # at the mid chord, so the crest radius is reduced below the geometric
        # limit (r_wrap - r_cap): that keeps the wrapped links in contact with
        # the tooth crests instead of in permanent interference with them, the
        # surplus chain length simply hanging as slack on the unloaded run.
        r_crest = max(0.06, 0.90 * (r_wrap - _CAP_R))
        self.contact_scale = r_crest / rmax
        prof = raw_prof * self.contact_scale
        order = np.argsort(np.arctan2(prof[:, 1], prof[:, 0]))
        self.poly = prof[order]
        self.n_edge = int(self.poly.shape[0])
        self.r_crest = r_crest
        self.r_wrap = r_wrap

        # sprocket rotational inertia about its (fixed) centre
        i_sp = mass[self.sprockets] * (0.5 * r_crest ** 2 + 0.25 * (r_crest * 0.794) ** 2)
        self.inv_i[self.sprockets] = 1.0 / i_sp

        # chain topology -----------------------------------------------------
        self.n_link = n_link
        self.joint_a = self.links.copy()
        self.joint_b = np.roll(self.links, -1)
        self.anchor_a = np.full(n_link, _PIN)      # local +x pin
        self.anchor_b = np.full(n_link, -_PIN)     # local -x pin

        self._initial_orientations()
        self._project_state(_INIT_PROJ_ITERS, 0.6)
        self.vel[:] = 0.0
        self._capture_contacts()
        self.time = 0.0
        self.steps = 0
        self._jacobi = None
        self._lam = None
        self._contact_rows = None
        self._contact_geom = None
        self._joint_lam = np.zeros((n_link, 2))
        self._contact_force = np.zeros((nb, 2))
        self._axle_force = np.zeros((nb, 2))
        self._floor_force = np.zeros(2)

    def _initial_orientations(self):
        """Orient every link along the loop so the closed chain is consistent."""
        centres = self.pos[self.links]
        prev = np.roll(centres, 1, axis=0)
        nxt = np.roll(centres, -1, axis=0)
        d = nxt - prev
        self.theta[self.links] = np.arctan2(-d[:, 1], d[:, 0])

    # -- projection ----------------------------------------------------------
    def _joint_project(self, sweeps):
        la, lb = self.joint_a, self.joint_b
        for _ in range(sweeps):
            ra = _rot(self.theta[la], np.stack([self.anchor_a, np.zeros_like(self.anchor_a)], -1))
            rb = _rot(self.theta[lb], np.stack([self.anchor_b, np.zeros_like(self.anchor_b)], -1))
            err = (self.pos[la] + ra) - (self.pos[lb] + rb)
            wa = self.inv_m[la][:, None]
            wb = self.inv_m[lb][:, None]
            ia = self.inv_i[la][:, None]
            ib = self.inv_i[lb][:, None]
            pa = _perpendicular(ra)
            pb = _perpendicular(rb)
            ka = wa[:, :, None] * np.eye(2)[None] + ia[:, :, None] * (pa[:, :, None] * pa[:, None, :])
            kb = wb[:, :, None] * np.eye(2)[None] + ib[:, :, None] * (pb[:, :, None] * pb[:, None, :])
            k = ka + kb
            k = k + 1.0e-12 * np.eye(2)[None]
            lam = -np.linalg.solve(k, err)
            self.pos[la] += self.inv_m[la][:, None] * lam
            self.theta[la] += self.inv_i[la] * _cross2(ra, lam)
            self.pos[lb] -= self.inv_m[lb][:, None] * lam
            self.theta[lb] -= self.inv_i[lb] * _cross2(rb, lam)
        self.theta[: self.n_body] = np.clip(self.theta[: self.n_body], -1.0e3, 1.0e3)

    def _project_state(self, sweeps, relax):
        for _ in range(sweeps):
            self._joint_project(1)
            geom = self._capture_contacts()
            self._contact_project(geom, relax)

    def _contact_project(self, geom, relax):
        if geom is None or geom["n"] == 0:
            return
        ba = geom["ba"]
        bb = geom["bb"]
        ra = geom["ra"]
        rb = geom["rb"]
        nrm = geom["nrm"]
        depth = geom["depth"]
        ca = _cross2(ra, nrm)
        cb = _cross2(rb, nrm)
        k = self.inv_m[ba] + self.inv_m[bb] + self.inv_i[ba] * ca * ca + self.inv_i[bb] * cb * cb
        k = np.where(k < 1.0e-12, 1.0e-12, k)
        lam = np.maximum(0.0, depth) / k * relax
        self.pos[ba] += self.inv_m[ba][:, None] * (lam[:, None] * nrm)
        self.theta[ba] += self.inv_i[ba] * (lam * ca)
        self.pos[bb] -= self.inv_m[bb][:, None] * (lam[:, None] * nrm)
        self.theta[bb] -= self.inv_i[bb] * (lam * cb)

    # -- collision detection -------------------------------------------------
    def _sprocket_frames(self):
        frames = []
        for k in self.sprockets:
            w = self.pos[k] + _rot(self.theta[k], self.poly)
            frames.append((int(k), w, np.roll(w, -1, axis=0)))
        return frames

    def _contact_geometry(self):
        links = self.links
        n = self.n_link
        cap_a = self.pos[links] + _rot(self.theta[links], np.tile([-_PIN, 0.0], (n, 1)))
        cap_b = self.pos[links] + _rot(self.theta[links], np.tile([_PIN, 0.0], (n, 1)))

        ba, bb, ra, rb, nrm, dep = [], [], [], [], [], []
        csd = np.zeros(4 * n)
        ground = self.n_body

        # --- chain links vs each sprocket --------------------------------
        for s, (k, e1, e2) in enumerate(self._sprocket_frames()):
            d, lpt, ept = _seg_seg(cap_a[:, None, :], cap_b[:, None, :],
                                   e1[None, :, :], e2[None, :, :])
            inside = _points_in_poly(lpt[np.arange(n), np.argmin(d, axis=1)], e1)
            signed = np.where(inside, -d[np.arange(n), np.argmin(d, axis=1)],
                              d[np.arange(n), np.argmin(d, axis=1)]) - _CAP_R
            csd[s * n:(s + 1) * n] = signed
            hit = d < (_CAP_R + _GEN_MARGIN)
            for i in range(n):
                cols = np.nonzero(hit[i])[0]
                if cols.size == 0:
                    continue
                order = cols[np.argsort(d[i, cols])][:_MAX_EDGES]
                for j in order:
                    dd = float(d[i, j])
                    vec = lpt[i, j] - ept[i, j]
                    nv = math.hypot(float(vec[0]), float(vec[1]))
                    if nv < 1.0e-9:
                        vec = lpt[i, j] - self.pos[k]
                        nv = max(math.hypot(float(vec[0]), float(vec[1])), 1.0e-9)
                    nn = np.asarray(vec, dtype=float) / nv
                    ba.append(int(links[i]))
                    bb.append(int(k))
                    ra.append(lpt[i, j] - self.pos[links[i]])
                    rb.append(ept[i, j] - self.pos[k])
                    nrm.append(nn)
                    dep.append(_CAP_R - dd)

        # --- chain links vs static floor (z = 0 plane) --------------------
        low = np.where(cap_a[:, 1] < cap_b[:, 1], 0, 1)
        low_pt = np.where(low[:, None] == 0, cap_a, cap_b)
        gap = low_pt[:, 1] - _CAP_R
        csd[2 * n:3 * n] = gap
        for i in range(n):
            if gap[i] < _GEN_MARGIN:
                ba.append(int(links[i]))
                bb.append(int(ground))
                ra.append(low_pt[i] - self.pos[links[i]])
                rb.append(np.zeros(2))
                nrm.append(np.array([0.0, 1.0]))
                dep.append(-gap[i])

        # --- chain links vs each other (joint neighbours excluded) --------
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if (j - i) % n not in (1, n - 1)]
        if pairs:
            pa = np.asarray([p[0] for p in pairs])
            pb = np.asarray([p[1] for p in pairs])
            d2, lpt2, ept2 = _seg_seg(cap_a[pa], cap_b[pa], cap_a[pb], cap_b[pb])
            for idx, (i, j) in enumerate(pairs):
                if (j - i) % n == 2:
                    csd[3 * n + i] = d2[idx] - 2.0 * _CAP_R
                if d2[idx] < 2.0 * _CAP_R + _GEN_MARGIN:
                    vec = lpt2[idx] - ept2[idx]
                    nv = math.hypot(float(vec[0]), float(vec[1]))
                    if nv < 1.0e-9:
                        vec = self.pos[j] - self.pos[i]
                        nv = max(math.hypot(float(vec[0]), float(vec[1])), 1.0e-9)
                    ba.append(int(links[i]))
                    bb.append(int(links[j]))
                    ra.append(lpt2[idx] - self.pos[links[i]])
                    rb.append(ept2[idx] - self.pos[links[j]])
                    nrm.append(np.asarray(vec, dtype=float) / nv)
                    dep.append(2.0 * _CAP_R - float(d2[idx]))

        geom = {
            "n": len(ba),
            "ba": np.asarray(ba, dtype=int),
            "bb": np.asarray(bb, dtype=int),
            "ra": np.asarray(ra, dtype=float).reshape(-1, 2),
            "rb": np.asarray(rb, dtype=float).reshape(-1, 2),
            "nrm": np.asarray(nrm, dtype=float).reshape(-1, 2),
            "depth": np.asarray(dep, dtype=float),
            "csd": csd,
        }
        if geom["n"] > _MAX_CONTACTS:
            keep = np.argsort(-geom["depth"])[:_MAX_CONTACTS]
            for key in ("ba", "bb", "ra", "rb", "nrm", "depth"):
                geom[key] = geom[key][keep]
            geom["n"] = _MAX_CONTACTS
        self._geom = geom
        return geom

    def _capture_contacts(self):
        return self._contact_geometry()

    # -- velocity level constraint solve -------------------------------------
    def _solve_contacts(self, geom, dt):
        nb = self.nb
        n_link = self.n_link
        n_con = int(geom["n"])
        n_row = 2 * n_link + 2 * n_con
        if n_row == 0:
            return
        dirs = np.zeros((n_row, 2))
        b_a = np.zeros(n_row, dtype=int)
        b_b = np.zeros(n_row, dtype=int)
        r_a = np.zeros((n_row, 2))
        r_b = np.zeros((n_row, 2))
        target = np.zeros(n_row)
        kind = np.zeros(n_row, dtype=int)          # 1 normal, 2 friction

        # chain pin joints (closed loop)
        la, lb = self.joint_a, self.joint_b
        ra = _rot(self.theta[la], np.stack([self.anchor_a, np.zeros_like(self.anchor_a)], -1))
        rb = _rot(self.theta[lb], np.stack([self.anchor_b, np.zeros_like(self.anchor_b)], -1))
        err = (self.pos[la] + ra) - (self.pos[lb] + rb)
        bias = np.clip(-_BETA / dt * err, -_BIAS_VMAX, _BIAS_VMAX)
        for ax in range(2):
            sl = slice(ax * n_link, (ax + 1) * n_link)
            d = np.zeros((n_link, 2))
            d[:, ax] = 1.0
            dirs[sl] = d
            b_a[sl] = la
            b_b[sl] = lb
            r_a[sl] = ra
            r_b[sl] = rb
            target[sl] = bias[:, ax]

        if n_con > 0:
            nrm = geom["nrm"]
            vel = self.vel
            v_a = vel[geom["ba"], :2] + _perpendicular(geom["ra"]) * vel[geom["ba"], 2:3]
            v_b = vel[geom["bb"], :2] + _perpendicular(geom["rb"]) * vel[geom["bb"], 2:3]
            v_rel = np.sum((v_a - v_b) * nrm, axis=1)
            bias_n = np.clip(_BETA / dt * np.maximum(0.0, geom["depth"] - _SLOP), 0.0, _BIAS_VMAX)
            rest = np.where(v_rel < -0.25, -self.restitution * v_rel, 0.0)
            for row in range(n_con):
                sl = slice(2 * n_link + 2 * row, 2 * n_link + 2 * row + 2)
                tang = np.array([nrm[row, 1], -nrm[row, 0]])
                dirs[sl] = np.stack([nrm[row], tang])
                b_a[sl] = geom["ba"][row]
                b_b[sl] = geom["bb"][row]
                r_a[sl] = geom["ra"][row]
                r_b[sl] = geom["rb"][row]
                target[sl] = (bias_n[row] + rest[row], 0.0)
                kind[sl] = (1, 2)

        c_a = _cross2(r_a, dirs)
        c_b = _cross2(r_b, dirs)
        minv = np.concatenate([self.inv_m, self.inv_m, self.inv_i])
        jac = np.zeros((n_row, 3 * nb))
        rows = np.arange(n_row)
        jac[rows, 3 * b_a] = dirs[:, 0]
        jac[rows, 3 * b_a + 1] = dirs[:, 1]
        jac[rows, 3 * b_a + 2] = c_a
        jac[rows, 3 * b_b] = -dirs[:, 0]
        jac[rows, 3 * b_b + 1] = -dirs[:, 1]
        jac[rows, 3 * b_b + 2] = -c_b

        eff = jac @ (minv[:, None] * jac.T)
        eff = eff + 1.0e-9 * np.eye(n_row)
        fac = None
        if _HAVE_SCIPY:
            try:
                fac = cho_factor(eff, lower=True, check_finite=False)
            except Exception:
                fac = None

        def linsolve(rhs):
            if fac is not None:
                return cho_solve(fac, rhs, check_finite=False)
            return np.linalg.lstsq(eff, rhs, rcond=None)[0]

        vel = self.vel
        v0 = vel.reshape(-1).copy()
        lam = np.zeros(n_row)
        n_idx = np.nonzero(kind == 1)[0]
        t_idx = np.nonzero(kind == 2)[0]
        mu = self.friction
        for _ in range(3):
            rhs = target - jac @ v0
            lam += linsolve(rhs)
            if n_idx.size:
                lam[n_idx] = np.maximum(0.0, lam[n_idx])
            if t_idx.size:
                bound = mu * np.maximum(0.0, lam[n_idx])
                lam[t_idx] = np.clip(lam[t_idx], -bound, bound)
            v0 = vel.reshape(-1).copy()
            vel.reshape(-1)[:] = v0 + minv * (jac.T @ lam)
            v0 = vel.reshape(-1).copy()

        # bookkeeping for the exported interaction forces
        self._joint_lam = lam[:2 * n_link].reshape(2, n_link).T.copy()
        contact_force = np.zeros((nb, 2))
        if n_con > 0:
            for row in range(n_con):
                n_i = 2 * n_link + 2 * row
                f = (lam[n_i] * dirs[n_i] + lam[n_i + 1] * dirs[n_i + 1]) / dt
                contact_force[geom["ba"][row]] += f
                contact_force[geom["bb"][row]] -= f
        self._contact_force = contact_force
        self._floor_force = -contact_force[self.n_body].copy()

    # -- time stepping -------------------------------------------------------
    def _step(self, dt):
        moving = self.inv_m > 0.0
        self.vel[moving, 1] += self.gravity[2] * dt
        for k in self.sprockets:
            k = int(k)
            w = self.vel[k, 2]
            torque = _DRIVE_TORQUE if k == self.driven else 0.0
            self.vel[k, 2] += (torque - _BEARING_DRAG * w) * self.inv_i[k] * dt
        geom = self._contact_geometry()
        self._solve_contacts(geom, dt)
        speed = np.linalg.norm(self.vel[:, :2], axis=1)
        over = speed > _MAX_LIN_VEL
        if over.any():
            self.vel[over, :2] *= (_MAX_LIN_VEL / speed[over])[:, None]
        np.clip(self.vel[:, 2], -_MAX_ANG_VEL, _MAX_ANG_VEL, out=self.vel[:, 2])
        self.pos[: self.n_body] += self.vel[: self.n_body, :2] * dt
        self.theta[: self.n_body] += self.vel[: self.n_body, 2] * dt
        self._joint_project(_STEP_PROJ_ITERS)
        bad = ~np.isfinite(self.pos).all(axis=1) | ~np.isfinite(self.theta)
        if bad.any():
            self.pos[bad] = 0.0
            self.theta[bad] = 0.0
            self.vel[bad] = 0.0
        self.time += dt
        self.steps += 1

    def advance_to(self, time: float):
        t = float(time)
        n_target = int(round(t / self.dt))
        while self.steps < n_target:
            self._step(self.dt)
        return self._frame(t)

    # -- output --------------------------------------------------------------
    def _quaternions(self):
        n = self.n_body
        quat = np.zeros((n, 4))
        half = 0.5 * self.theta[:n]
        s = np.sin(half)
        c = np.cos(half)
        is_link = np.zeros(n, dtype=bool)
        is_link[self.links] = True
        h = 0.7071067811865476
        quat[~is_link] = np.stack([np.zeros(n), s, np.zeros(n), c], axis=-1)[~is_link]
        quat[is_link] = np.stack([-h * c, h * s, h * s, h * c], axis=-1)[is_link]
        norm = np.linalg.norm(quat, axis=1, keepdims=True)
        return quat / np.maximum(norm, 1.0e-12)

    def _frame(self, t):
        geom = self._contact_geometry()
        n = self.n_body
        zeros = np.zeros(n)
        position = np.stack([self.pos[:n, 0], zeros, self.pos[:n, 1]], axis=-1)
        velocity = np.stack([self.vel[:n, 0], zeros, self.vel[:n, 2]], axis=-1)
        ang_vel = np.stack([zeros, self.vel[:n, 2], zeros], axis=-1)
        external = np.stack([zeros, zeros, self.mass * self.gravity[2]], axis=-1)

        inter = np.zeros((50, 3))
        # force transmitted to link i+1 by link i through the shared pin
        joint = -self._joint_lam / self.dt
        inter[0:self.n_link, 0] = joint[:, 0]
        inter[0:self.n_link, 2] = joint[:, 1]
        inter[self.n_link:2 * self.n_link, 0] = -joint[:, 0]
        inter[self.n_link:2 * self.n_link, 2] = -joint[:, 1]
        axle = np.zeros((n, 2))
        for k in self.sprockets:
            k = int(k)
            axle[k] = -(self._contact_force[k] + self.mass[k] * np.array([0.0, self.gravity[2]]))
        sp = list(self.sprockets)
        for s, k in enumerate(sp[:2]):
            k = int(k)
            inter[40 + s, 0] = axle[k, 0]
            inter[40 + s, 2] = axle[k, 1]
            inter[42 + s, 0] = -axle[k, 0]
            inter[42 + s, 2] = -axle[k, 1]
            net = self._contact_force[k]
            inter[44 + s, 0] = net[0]
            inter[44 + s, 2] = net[1]
            inter[46 + s, 0] = -net[0]
            inter[46 + s, 2] = -net[1]
        inter[48, 0] = self._floor_force[0]
        inter[48, 2] = self._floor_force[1]
        inter[49, 0] = -self._floor_force[0]
        inter[49, 2] = -self._floor_force[1]

        frame = StateFrame(
            float(t),
            position,
            velocity,
            orientation=self._quaternions(),
            angular_velocity=ang_vel,
            mass=self.mass.copy(),
            body_id=self.body_id,
            attributes={},
            fields={
                "contact_signed_distance": np.asarray(geom["csd"], dtype=float),
                "external_force": external,
                "interaction_force": inter,
            },
            scalars={
                "contact_length_scale": self.length_scale,
                "drive_torque": _DRIVE_TORQUE,
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        geom = getattr(self, "_geom", None)
        depth = float(np.max(geom["depth"])) if geom is not None and geom["n"] else 0.0
        return {
            "time": float(getattr(self, "time", 0.0)),
            "steps": int(getattr(self, "steps", 0)),
            "contacts": int(geom["n"]) if geom is not None else 0,
            "max_penetration": depth,
            "sprocket_omega": {str(k): float(self.vel[k, 2]) for k in self.sprockets},
            "chain_speed": float(np.mean(np.linalg.norm(self.vel[self.links, :2], axis=1))),
            "contact_scale": float(self.contact_scale),
            "wrap_radius": float(self.r_wrap),
        }
