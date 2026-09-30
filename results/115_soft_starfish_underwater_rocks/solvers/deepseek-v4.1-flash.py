"""Agent-owned two-way coupled solver for the soft starfish / underwater rock bed scene.

The model is implemented from scratch on NumPy + SciPy (no external physics engine):

* Free-surface water: 3-D weakly-compressible SPH (cubic-spline kernel, Tait equation of
  state, Monaghan artificial viscosity, XSPH smoothing), bounded by the frozen clear tank
  walls and the fixed ellipsoidal rock bed through inelastic impulse contacts.
* Soft starfish: the frozen triangulated shell is discretised as a mass-spring body
  (mesh-edge membrane springs, second-neighbour bending springs, through-thickness core
  springs, internal damping) with lumped nodal masses taken from the frozen density.
* Two-way coupling: every wetted node samples the local SPH state and receives buoyancy,
  hydrodynamic drag and a soft no-penetration repulsion; the exactly opposite reaction is
  scattered back onto the neighbouring fluid particles, so the deforming solid changes the
  resolved flow while the flow changes the solid motion and deformation.
* Starfish-rock / starfish-wall contact is resolved as non-penetrating inelastic contacts
  with friction, so different arms land on different rocks at different times.
* Gravity, fluid coupling and contact impulses are accumulated per frozen budget domain and
  exported through the required trajectory fields.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

# --------------------------------------------------------------------- constants
GRAVITY = np.array([0.0, 0.0, -9.81])
G_UP = np.array([0.0, 0.0, 9.81])
WATER_RHO = 1000.0
STARFISH_RHO = 1080.0

FLUID_SPACING = 0.04           # frozen initial water sample spacing
FLUID_VOLUME = FLUID_SPACING ** 3
SMOOTH_H = 0.055               # SPH smoothing length
SOUND_SPEED = 10.0             # numerical speed of sound (weakly compressible)
ALPHA_VISC = 0.15              # Monaghan artificial viscosity
XSPH_EPS = 0.05                # XSPH velocity smoothing
DT_SUB = 1.0e-3                # fixed physical substep
VEL_MAX = 6.0                  # safety clamp only (never reached in normal operation)

CD_DRAG = 0.45                 # hydrodynamic drag coefficient of a node
K_REP = 230.0                  # node/fluid no-penetration stiffness (per fluid particle)
C_REP = 4.6                    # node/fluid no-penetration damping
D0_REP = 0.018                 # repulsion skin depth
COUPLE_K = 64                  # SPH neighbours sampled at each solid node
STAR_ACC_CAP = 200.0           # cap on coupling acceleration of a solid node

K_STRETCH = 25.0               # membrane spring stiffness
C_STRETCH = 0.5
K_BEND = 1.5                   # bending spring stiffness
C_BEND = 0.12
K_CORE = 7.5                   # through-thickness core spring stiffness
C_CORE = 0.2

FALLBACK_ROCKS = (
    ((-0.34, -0.25, 0.10), (1.25, 0.82, 0.78), 0.12),
    ((-0.10, -0.31, 0.11), (1.20, 0.78, 1.05), 0.105),
    ((0.18, -0.30, 0.09), (1.35, 0.78, 0.72), 0.115),
    ((0.39, -0.22, 0.12), (0.78, 1.12, 1.05), 0.11),
    ((-0.39, 0.00, 0.12), (0.90, 1.28, 1.00), 0.115),
    ((-0.16, -0.04, 0.13), (1.15, 0.88, 1.06), 0.125),
    ((0.09, -0.06, 0.10), (1.28, 0.85, 0.82), 0.10),
    ((0.34, 0.03, 0.14), (0.88, 1.12, 1.10), 0.13),
    ((-0.31, 0.27, 0.10), (1.30, 0.82, 0.78), 0.11),
    ((-0.05, 0.25, 0.15), (0.88, 1.20, 1.20), 0.12),
    ((0.20, 0.25, 0.09), (1.42, 0.78, 0.75), 0.105),
    ((0.42, 0.29, 0.11), (0.78, 1.12, 0.92), 0.115),
)
FALLBACK_LO = np.array([-0.5, -0.5, 0.0])
FALLBACK_HI = np.array([0.5, 0.5, 0.72])


# ------------------------------------------------------------------ SPH kernels
def _kernel_w(r, h):
    inv = 1.0 / (math.pi * h * h * h)
    q = r / h
    f = np.zeros_like(r)
    inner = q < 1.0
    outer = (q >= 1.0) & (q < 2.0)
    qi = q[inner]
    f[inner] = 1.0 - 1.5 * qi * qi + 0.75 * qi ** 3
    qo = q[outer]
    f[outer] = 0.25 * (2.0 - qo) ** 3
    return f * inv


def _kernel_dw(r, h):
    inv = 1.0 / (math.pi * h ** 4)
    q = r / h
    d = np.zeros_like(r)
    inner = q < 1.0
    outer = (q >= 1.0) & (q < 2.0)
    qi = q[inner]
    d[inner] = -3.0 * qi + 2.25 * qi * qi
    qo = q[outer]
    d[outer] = -0.75 * (2.0 - qo) ** 2
    return d * inv


W_ZERO = 1.0 / (math.pi * SMOOTH_H ** 3)


# ------------------------------------------------------------------- geometry
def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_faces():
    """Triangle connectivity of the frozen starfish shell (asset or generator)."""
    path = _workspace_root() / "assets" / "starfish.obj"
    try:
        verts, faces = [], []
        for line in path.read_text(encoding="ascii").splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                faces.append([int(tok.split("/")[0]) - 1 for tok in parts[1:4]])
        if len(verts) >= 4 and len(faces) >= 1:
            return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)
    except Exception:
        pass
    return _generate_starfish()


def _generate_starfish():
    """Frozen parametric starfish surface (same construction as the scaffold asset)."""
    verts = [[0.0, 0.0, 0.030], [0.0, 0.0, -0.030]]
    rings = []
    n_theta, n_radial = 100, 8
    for j in range(1, n_radial + 1):
        s = j / n_radial
        ring = []
        for i in range(n_theta):
            a = 2.0 * math.pi * i / n_theta
            outer = 0.205 + 0.145 * math.cos(5.0 * a)
            r = s * outer
            half_t = 0.010 + 0.031 * (1.0 - s) ** 0.65
            top = len(verts)
            verts.append([r * math.cos(a), r * math.sin(a), half_t])
            verts.append([r * math.cos(a), r * math.sin(a), -half_t])
            ring.append((top, top + 1))
        rings.append(ring)
    faces = []
    first = rings[0]
    for i in range(n_theta):
        k = (i + 1) % n_theta
        faces.extend([[0, first[i][0], first[k][0]], [1, first[k][1], first[i][1]]])
    for j in range(n_radial - 1):
        inner, outer = rings[j], rings[j + 1]
        for i in range(n_theta):
            k = (i + 1) % n_theta
            it, ib = inner[i]
            kt, kb = inner[k]
            ot, ob = outer[i]
            qt, qb = outer[k]
            faces.extend([[it, ot, qt], [it, qt, kt], [ib, kb, qb], [ib, qb, ob]])
    last = rings[-1]
    for i in range(n_theta):
        k = (i + 1) % n_theta
        t0, b0 = last[i]
        t1, b1 = last[k]
        faces.extend([[t0, b0, b1], [t0, b1, t1]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def _load_geometry():
    """Return (rocks, lo, hi, wall_samples) from the frozen scene geometry."""
    objs = None
    try:
        import yaml

        data = yaml.safe_load((_workspace_root() / "scene_geometry.yaml").read_text("utf-8"))
        objs = data.get("objects")
    except Exception:
        objs = None

    if not objs:
        rocks = tuple(
            (np.asarray(c, float), np.asarray(s, float), float(r)) for c, s, r in FALLBACK_ROCKS
        )
        return rocks, FALLBACK_LO.copy(), FALLBACK_HI.copy(), _wall_samples(
            FALLBACK_LO, FALLBACK_HI
        )

    rocks = []
    boxes = []
    for obj in objs:
        geom = obj.get("geometry", {}) or {}
        trans = obj.get("transform", {}) or {}
        if geom.get("type") == "sphere" and obj.get("semantic") == "rigid":
            c = np.asarray(trans.get("translation", [0.0, 0.0, 0.0]), dtype=float)
            s = np.asarray(trans.get("scale", [1.0, 1.0, 1.0]), dtype=float)
            rocks.append((c, s, float(geom.get("radius", 0.1))))
        elif geom.get("type") == "box":
            boxes.append(
                (
                    np.asarray(trans.get("translation", [0.0, 0.0, 0.0]), dtype=float),
                    np.asarray(geom.get("half_extents", [0.0, 0.0, 0.0]), dtype=float),
                )
            )
    if not rocks:
        rocks = [
            (np.asarray(c, float), np.asarray(s, float), float(r)) for c, s, r in FALLBACK_ROCKS
        ]

    lo = np.array([-1.0e30, -1.0e30, -1.0e30])
    hi = np.array([1.0e30, 1.0e30, 1.0e30])
    if boxes:
        for t, h in boxes:
            thin = int(np.argmin(h))
            if thin in (0, 1):
                if t[thin] < 0.0:
                    lo[thin] = t[thin] + h[thin]
                else:
                    hi[thin] = t[thin] - h[thin]
        lo[2] = max(float(t[2] - h[2]) for t, h in boxes)
        hi[2] = max(float(t[2] + h[2]) for t, h in boxes)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        lo, hi = FALLBACK_LO.copy(), FALLBACK_HI.copy()
    return tuple(rocks), lo, hi, _wall_samples(lo, hi)


def _wall_samples(lo, hi):
    """Static wall sample points, outward-from-fluid normals (tank + rock bed)."""
    pts, nrm = [], []
    for axis, sign in ((0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0), (2, -1.0)):
        plane = lo[axis] if sign < 0.0 else hi[axis]
        others = [a for a in range(3) if a != axis]
        for u in (-0.25, 0.0, 0.25):
            for v in (-0.25, 0.0, 0.25):
                p = np.zeros(3)
                p[axis] = plane
                p[others[0]] = u
                p[others[1]] = v
                n = np.zeros(3)
                n[axis] = sign
                pts.append(p)
                nrm.append(n)
    for c, s, radius in FALLBACK_ROCKS:
        cc = np.asarray(c, float)
        ss = np.asarray(s, float)
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)):
            dv = np.asarray(d, float)
            p = cc + dv * (radius * ss)
            n = dv / ss
            n = n / max(float(np.linalg.norm(n)), 1.0e-12)
            pts.append(p)
            nrm.append(n)
    return np.asarray(pts, float), np.asarray(nrm, float)


# --------------------------------------------------------------------- solver
class BenchmarkSolver:
    def __init__(self) -> None:
        self._ready = False

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        frame = scene.initial_state()
        pos = np.asarray(frame.position, dtype=float).copy()
        vel = np.asarray(frame.velocity, dtype=float).copy()
        mass = np.asarray(frame.mass, dtype=float).copy()
        body = np.asarray(frame.body_id).astype(np.int64).copy()
        rest = np.asarray(frame.attributes["rest_position"], dtype=float).copy()

        star_idx = np.nonzero(body == 1)[0]
        water_idx = np.nonzero(body == 2)[0]
        order = np.concatenate([star_idx, water_idx])
        pos = pos[order]
        vel = vel[order]
        mass = mass[order]
        body = body[order]
        rest = rest[order]

        self.n_star = int(star_idx.size)
        self.n_water = int(water_idx.size)
        self.n_total = self.n_star + self.n_water
        ns = self.n_star

        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.body = body
        self.mass_star = mass[:ns].copy()
        self.mass_water = mass[ns:].copy()
        self.mass_star_total = float(self.mass_star.sum())
        self.mass_water_total = float(self.mass_water.sum())
        self.rest_star = rest[:ns].copy()

        rocks, lo, hi, bnd = _load_geometry()
        self.rocks = []
        for c, s, radius in rocks:
            s = np.maximum(np.asarray(s, float), 1.0e-6)
            self.rocks.append((np.asarray(c, float), 1.0 / s, s, float(radius)))
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        self.bnd_points, self.bnd_normal = bnd

        self._build_shell()
        self._push_out_of_rocks(self.pos[ns:])
        self.rho_ref = self._reference_density()
        self._sound_b = self.rho_ref * SOUND_SPEED ** 2 / 7.0

        self.reference_velocity = float(
            math.sqrt(max(1.0e-6, 2.0 * abs(GRAVITY[2]) * 0.18))
        )
        self.reference_momentum = float(
            (self.mass_star_total + self.mass_water_total) * self.reference_velocity
        )

        self.domain_weights = np.zeros((3, self.n_total), dtype=float)
        self.domain_weights[0, :ns] = 1.0
        self.domain_weights[1, ns:] = 1.0
        self.domain_weights[2, :] = 1.0

        self._int_imp = np.zeros((3, 3, 3), dtype=float)
        self._cum_imp = np.zeros((3, 3), dtype=float)
        self._interval_steps = 0
        self._step_index = 0
        self._time = 0.0
        self._ready = True

    def _build_shell(self) -> None:
        ns = self.n_star
        verts, faces = _load_faces()
        ok = verts.shape[0] == ns and ns > 0 and np.allclose(
            verts - verts.mean(axis=0),
            self.rest_star - self.rest_star.mean(axis=0),
            atol=1.0e-6,
        )
        if not ok:
            # frozen generator topology, identical vertex ordering
            _, faces = _generate_starfish()
        f = np.asarray(faces, dtype=np.int64)
        # consistent outward orientation (top pole normal must point +z)
        p = self.rest_star
        cr = np.cross(p[f[:, 1]] - p[f[:, 0]], p[f[:, 2]] - p[f[:, 0]])
        area_vec = 0.5 * cr
        node_vec = np.zeros((ns, 3))
        for k in range(3):
            for c in range(3):
                node_vec[:, c] += np.bincount(
                    f[:, k], weights=area_vec[:, c], minlength=ns
                )
        node_vec /= 3.0
        top = int(np.argmax(p[:, 2]))
        if node_vec[top, 2] < 0.0:
            f = f[:, [0, 2, 1]]
            area_vec = -area_vec
            node_vec = -node_vec

        self.faces = f
        mag = np.linalg.norm(node_vec, axis=1)
        self.node_area = mag
        self.node_normal = node_vec / np.maximum(mag, 1.0e-12)[:, None]
        self.rest_normal = self.node_normal.copy()

        edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        adjacency = [set() for _ in range(ns)]
        for a, b in edges:
            adjacency[int(a)].add(int(b))
            adjacency[int(b)].add(int(a))
        bend = set()
        for a in range(ns):
            for b in adjacency[a]:
                for c in adjacency[b]:
                    if c != a and c not in adjacency[a]:
                        bend.add((a, c) if a < c else (c, a))
        bend_arr = np.asarray(sorted(bend), dtype=np.int64) if bend else np.zeros((0, 2), np.int64)

        # through-thickness core springs (consecutive top/bottom vertex pairs)
        core = self._core_pairs(edges)

        self.e0 = np.concatenate([edges[:, 0], bend_arr[:, 0], core[:, 0]]).astype(np.int64)
        self.e1 = np.concatenate([edges[:, 1], bend_arr[:, 1], core[:, 1]]).astype(np.int64)
        stiff = np.concatenate(
            [
                np.full(edges.shape[0], K_STRETCH),
                np.full(bend_arr.shape[0], K_BEND),
                np.full(core.shape[0], K_CORE),
            ]
        )
        damp = np.concatenate(
            [
                np.full(edges.shape[0], C_STRETCH),
                np.full(bend_arr.shape[0], C_BEND),
                np.full(core.shape[0], C_CORE),
            ]
        )
        d = p[self.e0] - p[self.e1]
        self.spring_l0 = np.linalg.norm(d, axis=1)
        self.spring_k = stiff
        self.spring_c = damp

    def _core_pairs(self, edges):
        """Top/bottom vertex pairs of the frozen shell (same ring index, consecutive id)."""
        ns = self.n_star
        p = self.rest_star
        pairs = []
        for a in range(0, ns - 1, 2):
            b = a + 1
            if b < ns:
                pairs.append((a, b))
        if not pairs:
            return np.zeros((0, 2), dtype=np.int64)
        arr = np.asarray(pairs, dtype=np.int64)
        d = np.linalg.norm(p[arr[:, 0]] - p[arr[:, 1]], axis=1)
        keep = d > 1.0e-6
        return arr[keep]

    def _push_out_of_rocks(self, wpos) -> None:
        for _ in range(3):
            for c, inv_scale, scale, radius in self.rocks:
                rel = wpos - c
                q = rel * inv_scale
                r2 = np.einsum("ij,ij->i", q, q)
                inside = r2 < 1.0
                if not inside.any():
                    continue
                k = np.nonzero(inside)[0]
                s = np.sqrt(np.maximum(r2[k], 1.0e-12))
                wpos[k] = c + rel[k] * (1.02 / s)[:, None]

    def _reference_density(self):
        nw = self.n_water
        wpos = self.pos[self.n_star :]
        wmass = self.mass_water
        rho = wmass * W_ZERO
        pairs = cKDTree(wpos).query_pairs(2.0 * SMOOTH_H, output_type="ndarray")
        if pairs.shape[0]:
            i, j = pairs[:, 0], pairs[:, 1]
            dr = wpos[i] - wpos[j]
            r = np.sqrt(np.einsum("ij,ij->i", dr, dr))
            w = _kernel_w(r, SMOOTH_H)
            rho = rho + np.bincount(i, weights=wmass[j] * w, minlength=nw)
            rho = rho + np.bincount(j, weights=wmass[i] * w, minlength=nw)
        return float(np.percentile(rho, 90.0))

    # ------------------------------------------------------------- stepping
    def _step(self, dt: float) -> None:
        ns = self.n_star
        nw = self.n_water
        pos = self.pos
        vel = self.vel
        wpos = pos[ns:]
        wvel = vel[ns:]
        wmass = self.mass_water
        spos = pos[:ns]
        svel = vel[:ns]
        smass = self.mass_star

        tree_w = cKDTree(wpos)
        pairs = tree_w.query_pairs(2.0 * SMOOTH_H, output_type="ndarray")

        # ---- SPH density and pressure
        rho = wmass * W_ZERO
        if pairs.shape[0]:
            i = pairs[:, 0].astype(np.int64)
            j = pairs[:, 1].astype(np.int64)
            dr = wpos[i] - wpos[j]
            r = np.sqrt(np.einsum("ij,ij->i", dr, dr))
            w = _kernel_w(r, SMOOTH_H)
            rho = rho + np.bincount(i, weights=wmass[j] * w, minlength=nw)
            rho = rho + np.bincount(j, weights=wmass[i] * w, minlength=nw)
        rho = np.maximum(rho, 0.2 * self.rho_ref)
        pres = self._sound_b * ((rho / self.rho_ref) ** 7 - 1.0)
        np.maximum(pres, 0.0, out=pres)

        # ---- SPH momentum: pressure + artificial viscosity + gravity
        acc = np.zeros((nw, 3))
        acc[:, 2] = GRAVITY[2]
        if pairs.shape[0]:
            i = pairs[:, 0].astype(np.int64)
            j = pairs[:, 1].astype(np.int64)
            dr = wpos[i] - wpos[j]
            r = np.maximum(np.sqrt(np.einsum("ij,ij->i", dr, dr)), 1.0e-9)
            dwdr = _kernel_dw(r, SMOOTH_H)
            pi, pj = pres[i], pres[j]
            ri, rj = rho[i], rho[j]
            vij = wvel[i] - wvel[j]
            vr = np.einsum("ij,ij->i", vij, dr)
            mu = SMOOTH_H * vr / (r * r + 0.01 * SMOOTH_H * SMOOTH_H)
            pi_visc = np.where(
                vr < 0.0, -ALPHA_VISC * SOUND_SPEED * mu / (0.5 * (ri + rj)), 0.0
            )
            s = (pi / (ri * ri) + pj / (rj * rj) + pi_visc) * dwdr / r
            for c in range(3):
                si = -wmass[j] * s * dr[:, c]
                sj = wmass[i] * s * dr[:, c]
                acc[:, c] += np.bincount(i, weights=si, minlength=nw)
                acc[:, c] += np.bincount(j, weights=sj, minlength=nw)

        # ---- two-way fluid / solid coupling
        fnode = np.zeros((ns, 3))
        frep = np.zeros((ns, COUPLE_K, 3))
        fluid_react = np.zeros((nw, 3))
        if ns > 0 and nw > 0:
            dist, nidx = tree_w.query(
                spos, k=min(COUPLE_K, nw), distance_upper_bound=2.0 * SMOOTH_H
            )
            valid = np.isfinite(dist)
            idx = np.where(valid, nidx, 0).astype(np.int64)
            wgt = np.zeros_like(dist)
            wgt[valid] = FLUID_VOLUME * _kernel_w(dist[valid], SMOOTH_H)
            sumw = wgt.sum(axis=1)
            chi = np.minimum(1.0, sumw)
            denom = np.maximum(sumw, 1.0e-12)
            uw = np.stack(
                [
                    (wgt * wvel[idx, c]).sum(axis=1) / denom
                    for c in range(3)
                ],
                axis=1,
            )
            vol = smass / STARFISH_RHO
            vrel = svel - uw
            spd = np.linalg.norm(vrel, axis=1)
            buoy = G_UP[None, :] * (chi * WATER_RHO * vol)[:, None]
            drag = (
                -0.5
                * CD_DRAG
                * WATER_RHO
                * (self.node_area * chi * spd)[:, None]
                * vrel
            )
            cap = smass * 40.0
            dmag = np.linalg.norm(drag, axis=1)
            scale = np.where(dmag > cap, cap / np.maximum(dmag, 1.0e-12), 1.0)
            fnode = buoy + drag * scale[:, None]

            rel = wpos[idx] - spos[:, None, :]
            d = np.linalg.norm(rel, axis=2)
            side = np.einsum("bkd,bd->bk", rel, self.node_normal)
            pen = D0_REP - side
            dirv = self.node_normal[:, None, :] * np.where(side >= 0.0, 1.0, -1.0)[:, :, None]
            act = valid & (pen > 0.0)
            vn = np.einsum("bkd,bkd->bk", wvel[idx] - svel[:, None, :], dirv)
            fmag = K_REP * np.maximum(pen, 0.0) - C_REP * np.minimum(vn, 0.0)
            fmag = np.where(act, np.maximum(fmag, 0.0), 0.0)
            frep = fmag[:, :, None] * dirv

            fnode_rep = -frep.sum(axis=1)
            ftot = fnode + fnode_rep
            fmag_tot = np.linalg.norm(ftot, axis=1)
            cap_tot = smass * STAR_ACC_CAP
            sc = np.where(fmag_tot > cap_tot, cap_tot / np.maximum(fmag_tot, 1.0e-12), 1.0)
            fnode = fnode * sc[:, None]
            frep = frep * sc[:, None, None]
            fnode_tot = fnode - frep.sum(axis=1)

            share = np.where(valid, wgt / denom[:, None], 0.0)
            mj = np.maximum(wmass[idx], 1.0e-12)
            for c in range(3):
                # equal-and-opposite reaction: coupling spread by kernel weight,
                # no-penetration repulsion applied to the contacted particles
                contrib = (
                    -(fnode[:, None, c] * share) + frep[:, :, c]
                ) / mj
                fluid_react[:, c] = np.bincount(
                    idx.ravel(), weights=contrib.ravel(), minlength=nw
                )
        else:
            fnode_tot = np.zeros((ns, 3))

        # ---- integrate fluid
        wvel += (acc + fluid_react) * dt
        np.clip(wvel, -VEL_MAX, VEL_MAX, out=wvel)
        wpos += wvel * dt

        # ---- integrate starfish (elastic springs + gravity + fluid coupling)
        sacc = np.zeros((ns, 3))
        if ns > 0:
            d = spos[self.e0] - spos[self.e1]
            L = np.maximum(np.linalg.norm(d, axis=1), 1.0e-9)
            lhat = d / L[:, None]
            vrel_n = np.einsum("ij,ij->i", svel[self.e0] - svel[self.e1], lhat)
            fsc = -self.spring_k * (L - self.spring_l0) - self.spring_c * vrel_n
            fv = fsc[:, None] * lhat
            for c in range(3):
                sacc[:, c] += (
                    np.bincount(self.e0, weights=fv[:, c], minlength=ns)
                    - np.bincount(self.e1, weights=fv[:, c], minlength=ns)
                ) / smass
            sacc += GRAVITY[None, :]
            sacc += fnode_tot / smass[:, None]
            np.clip(sacc, -5000.0, 5000.0, out=sacc)
            svel += sacc * dt
            spos += svel * dt

        # ---- XSPH velocity smoothing
        if pairs.shape[0]:
            i = pairs[:, 0].astype(np.int64)
            j = pairs[:, 1].astype(np.int64)
            dr = wpos[i] - wpos[j]
            r = np.sqrt(np.einsum("ij,ij->i", dr, dr))
            w = _kernel_w(r, SMOOTH_H)
            rbar = np.maximum(0.5 * (rho[i] + rho[j]), 1.0e-9)
            dv = wvel[j] - wvel[i]
            for c in range(3):
                wvel[:, c] += XSPH_EPS * np.bincount(
                    i, weights=(wmass[j] * w / rbar) * dv[:, c], minlength=nw
                )
                wvel[:, c] -= XSPH_EPS * np.bincount(
                    j, weights=(wmass[i] * w / rbar) * dv[:, c], minlength=nw
                )

        # ---- static contacts (tank walls, rock bed) for water and starfish
        jw = self._contact_points(wpos, wvel, wmass)
        js = self._contact_points(spos, svel, smass) if ns > 0 else np.zeros(3)

        # ---- budget impulses
        gt = GRAVITY * dt
        self._int_imp[0, 0] += self.mass_star_total * gt
        self._int_imp[1, 0] += self.mass_water_total * gt
        self._int_imp[2, 0] += (self.mass_star_total + self.mass_water_total) * gt
        self._int_imp[0, 1] += fnode_tot * dt
        self._int_imp[1, 1] -= fnode_tot * dt
        self._int_imp[0, 2] += js
        self._int_imp[1, 2] += jw
        self._int_imp[2, 2] += js + jw
        self._interval_steps += 1
        self._time += dt

    def _contact_points(self, pos, vel, m):
        """Inelastic impulse contacts of point masses with walls and rock ellipsoids."""
        imp = np.zeros(3)
        for ax in range(3):
            p = pos[:, ax]
            low = np.nonzero(p < self.lo[ax])[0]
            if low.size:
                pos[low, ax] = self.lo[ax]
                v = vel[low, ax]
                hit = low[v < 0.0]
                if hit.size:
                    vv = vel[hit, ax]
                    imp[ax] += float(np.sum(m[hit] * (-vv)))
                    vel[hit, ax] = 0.0
            high = np.nonzero(p > self.hi[ax])[0]
            if high.size:
                pos[high, ax] = self.hi[ax]
                v = vel[high, ax]
                hit = high[v > 0.0]
                if hit.size:
                    vv = vel[hit, ax]
                    imp[ax] -= float(np.sum(m[hit] * vv))
                    vel[hit, ax] = 0.0
        for c, inv_scale, scale, radius in self.rocks:
            rel = pos - c
            q = rel * inv_scale
            r2 = np.einsum("ij,ij->i", q, q)
            inside = np.nonzero(r2 < 1.0)[0]
            if not inside.size:
                continue
            s = np.sqrt(np.maximum(r2[inside], 1.0e-12))
            relk = rel[inside]
            pos[inside] = c + relk / s[:, None]
            n = relk * (inv_scale * inv_scale)
            n /= np.maximum(np.linalg.norm(n, axis=1), 1.0e-12)[:, None]
            v = vel[inside]
            vn = np.minimum(np.einsum("ij,ij->i", v, n), 0.0)
            dvn = vn[:, None] * n
            vel[inside] = v - dvn
            vt = vel[inside] - np.einsum("ij,ij->i", vel[inside], n)[:, None] * n
            fric = 0.2 * vt
            vel[inside] -= fric
            imp += np.sum(m[inside][:, None] * (dvn + fric), axis=0)
        return imp

    # ---------------------------------------------------------------- output
    def advance_to(self, time: float):
        target = int(round(float(time) / DT_SUB))
        while self._step_index < target:
            self._step(DT_SUB)
            self._step_index += 1
        return self._frame(float(time))

    def _frame(self, t: float):
        steps = self._interval_steps
        force = np.zeros((3, 3, 3))
        if steps > 0:
            force = self._int_imp / (steps * DT_SUB)
        else:
            g = GRAVITY
            force[0, 0, :] = self.mass_star_total * g
            force[1, 0, :] = self.mass_water_total * g
            force[2, 0, :] = (self.mass_star_total + self.mass_water_total) * g
        self._cum_imp += self._int_imp
        self._int_imp = np.zeros((3, 3, 3))
        self._interval_steps = 0

        interaction = np.zeros((2, 3))
        interaction[0, :] = force[0, 1, :]
        interaction[1, :] = force[0, 2, :]

        fields = {
            "boundary_normal": self.bnd_normal,
            "boundary_target_velocity": np.zeros((self.bnd_normal.shape[0], 3)),
            "boundary_velocity": np.zeros((self.bnd_normal.shape[0], 3)),
            "budget_domain_weights": self.domain_weights,
            "budget_external_force": force,
            "budget_cumulative_external_impulse": self._cum_imp.copy(),
            "budget_cumulative_inflow_momentum": np.zeros((3, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((3, 3)),
            "cumulative_mass_source": np.zeros(1),
            "interaction_force": interaction,
        }
        attributes = {"mass_quantity": self.mass.copy()}
        scalars = {
            "reference_momentum": self.reference_momentum,
            "reference_velocity": self.reference_velocity,
        }
        from worldsolver.contracts import StateFrame

        frame = StateFrame(
            time=t,
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        result: dict[str, object] = {
            "n_starfish_nodes": int(self.n_star),
            "n_water_particles": int(self.n_water),
            "reference_density": float(getattr(self, "rho_ref", 0.0)),
            "reference_velocity": float(getattr(self, "reference_velocity", 0.0)),
            "reference_momentum": float(getattr(self, "reference_momentum", 0.0)),
            "steps": int(self._step_index),
        }
        if self._ready:
            ns = self.n_star
            result["starfish_center"] = [float(v) for v in self.pos[:ns].mean(axis=0)]
            result["water_surface_z"] = float(
                np.percentile(self.pos[ns:, 2], 95.0)
            )
            result["max_speed"] = float(np.max(np.linalg.norm(self.vel, axis=1)))
        return result
