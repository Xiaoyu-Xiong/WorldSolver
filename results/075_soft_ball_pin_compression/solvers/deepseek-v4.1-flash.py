"""Physically based solver for the soft-ball pin-compression scene.

The frozen ball asset is a closed volumetric tetrahedral mesh (13 nodes, 20 tets,
20 surface triangles).  The solver integrates the actual equations of motion of that
solid with an explicit (symplectic-Euler) lumped-mass finite element scheme:

  * compressible neo-Hookean hyperelasticity per tetrahedron,
  * gravity,
  * mass-proportional Rayleigh damping plus velocity-level contact damping,
  * penalty (regularised non-penetration) contact between the ball surface and the
    two 81-segment pin arrays.  The lower array is static, the upper array follows
    the prescribed piecewise-linear translation.

The mesh, material density, gravity and the prescribed motion all come from the frozen
scaffold; no scripted animation or precomputed motion table is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# ----------------------------------------------------------------------------
# numerical / physical parameters (SI units)
# ----------------------------------------------------------------------------
_YOUNG = 2.0e5          # Pa, soft rubber
_POISSON = 0.40
_DT = 1.0e-3            # integration step [s] (sample_dt is a multiple of it)
_CONTACT_K = 1.0e5      # penalty stiffness per surface/segment sample [N/m]
_CONTACT_R = 0.009      # pin radius fallback [m]
_SAMPLE_DS = 0.02       # spacing of samples along each pin segment [m]
_GLOBAL_DAMP = 3.0      # mass-proportional damping [1/s]
_PRUNE = 0.55           # activity radius around the closest ball node [m]
_J_FLOOR = 0.02         # regularisation floor on J (prevents inversion blow-up)


def _read_json(paths):
    for path in paths:
        try:
            if path.is_file():
                return json.loads(path.read_text())
        except (OSError, ValueError):
            continue
    return None


def _dot(a, b):
    return np.sum(np.asarray(a) * np.asarray(b), axis=-1)


class BenchmarkSolver:
    """Lumped-mass explicit FEM solver with codimensional pin contact."""

    def __init__(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self._scene = scene
        cfg = getattr(scene, "config", None) or {}
        self._cfg = dict(cfg)
        self._seed = int(seed)
        self._g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float).reshape(3)

        frame = scene.initial_state()
        self._rest = np.array(frame.position, dtype=float, copy=True)
        self._x = self._rest.copy()
        self._v = np.array(frame.velocity, dtype=float, copy=True)
        mass = np.array(frame.mass, dtype=float, copy=True)
        if mass.shape != (self._x.shape[0],):
            mass = np.full(self._x.shape[0], float(np.sum(mass)) / self._x.shape[0])
        self._m = mass
        self._n = int(self._x.shape[0])
        self._mass_total = float(np.sum(self._m))

        root = self._workspace_root(scene)
        ball = _read_json(
            [
                root / "assets" / "soft_ball_volume.json",
                Path("assets") / "soft_ball_volume.json",
            ]
        )
        tets, faces = self._topology(ball)
        self._tets = tets
        self._faces = faces
        self._prepare_reference(tets, faces)

        young = float(cfg.get("material", {}).get("young_modulus", _YOUNG))
        poisson = float(cfg.get("material", {}).get("poisson_ratio", _POISSON))
        self._mu = young / (2.0 * (1.0 + poisson))
        self._lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
        self._density = float(
            cfg.get("initial_conditions", {}).get("ball_density", 980.0)
        )

        self._contact_k = _CONTACT_K
        self._contact_r = self._pin_radius(root)
        self._c_global = _GLOBAL_DAMP
        self._c_contact = 0.4 * np.sqrt(self._contact_k * max(self._mass_total, 1e-12))
        self._node_c_contact = 2.0 * np.sqrt(
            self._contact_k * max(self._mass_total / max(self._n, 1), 1e-12)
        )

        self._prepare_pins(root)
        self._prepare_motion(cfg)

        self._t = 0.0
        self._h = _DT
        self._impulse = np.zeros(3, dtype=float)
        self._last_contact_total = np.zeros(3, dtype=float)
        self._gravity_total = self._mass_total * self._g
        length_scale = 0.46
        self._ref_momentum = float(
            self._mass_total * np.sqrt(2.0 * abs(float(self._g[2])) * length_scale)
        )
        self._ref_momentum = max(self._ref_momentum, 1e-9)
        self._initialized = True

    def _workspace_root(self, scene) -> Path:
        here = Path(__file__).resolve().parent.parent
        if (here / "config.yaml").is_file():
            return here
        return Path.cwd()

    def _topology(self, ball):
        """Return (tets[Ne,4], faces[Nf,3]) from the frozen asset."""
        if isinstance(ball, dict) and "tetrahedra" in ball and "vertices" in ball:
            tets = np.asarray(ball["tetrahedra"], dtype=int).reshape(-1, 4)
            key = "faces" if "faces" in ball else "surface_triangles"
            faces = np.asarray(ball[key], dtype=int).reshape(-1, 3)
            return tets, faces
        # Fallback: build a Delaunay-style tetrahedralisation of the ball nodes.
        from scipy.spatial import ConvexHull

        hull = ConvexHull(self._rest)
        faces = np.asarray(hull.simplices, dtype=int)
        centre = np.argmin(np.linalg.norm(self._rest - self._rest.mean(axis=0), axis=1))
        tets = np.column_stack(
            (np.full(len(faces), int(centre), dtype=int), faces)
        )
        return tets, faces

    def _prepare_reference(self, tets, faces):
        x = self._rest
        # force positive tet orientation (det > 0)
        d0, d1, d2, d3 = x[tets[:, 0]], x[tets[:, 1]], x[tets[:, 2]], x[tets[:, 3]]
        dm = np.stack((d1 - d0, d2 - d0, d3 - d0), axis=2)
        det = np.linalg.det(dm)
        flip = det < 0.0
        if np.any(flip):
            tets = tets.copy()
            tets[flip, 1], tets[flip, 2] = tets[flip, 2], tets[flip, 1].copy()
            self._tets = tets
            d1, d2 = x[tets[:, 1]], x[tets[:, 2]]
            dm = np.stack((d1 - d0, d2 - d0, d3 - d0), axis=2)
            det = np.linalg.det(dm)
        self._V0 = np.abs(det) / 6.0
        self._Dm_inv = np.linalg.inv(dm)

        # orient surface faces outward
        c = x.mean(axis=0)
        e1 = x[faces[:, 1]] - x[faces[:, 0]]
        e2 = x[faces[:, 2]] - x[faces[:, 0]]
        nrm = np.cross(e1, e2)
        fc = (x[faces[:, 0]] + x[faces[:, 1]] + x[faces[:, 2]]) / 3.0
        inward = _dot(nrm, fc - c) < 0.0
        if np.any(inward):
            faces = faces.copy()
            faces[inward, 1], faces[inward, 2] = faces[inward, 2], faces[inward, 1].copy()
        self._faces = faces

    def _pin_radius(self, root: Path) -> float:
        for candidate in (root / "scene_geometry.yaml", Path("scene_geometry.yaml")):
            try:
                if not candidate.is_file():
                    continue
                import yaml

                data = yaml.safe_load(candidate.read_text())
                for item in data.get("objects", []):
                    geometry = item.get("geometry", {})
                    if geometry.get("type") == "curve_network" and geometry.get("radius"):
                        return float(geometry["radius"])
            except Exception:
                continue
        return _CONTACT_R

    def _prepare_pins(self, root: Path) -> None:
        lower = _read_json(
            [root / "assets" / "lower_pins.json", Path("assets") / "lower_pins.json"]
        )
        upper = _read_json(
            [root / "assets" / "upper_pins.json", Path("assets") / "upper_pins.json"]
        )
        p0 = []
        p1 = []
        flags = []
        for flag, data in ((0.0, lower), (1.0, upper)):
            if not isinstance(data, dict):
                continue
            verts = np.asarray(data["vertices"], dtype=float)
            edges = np.asarray(data["edges"], dtype=int)
            p0.append(verts[edges[:, 0]])
            p1.append(verts[edges[:, 1]])
            flags.append(np.full(len(edges), flag, dtype=float))
        if not p0:
            raise ValueError("pin assets could not be loaded")
        self._seg_p0 = np.concatenate(p0, axis=0)
        self._seg_p1 = np.concatenate(p1, axis=0)
        self._seg_flag = np.concatenate(flags, axis=0)

        # dense sample points along each pin segment
        bases = []
        sflags = []
        for i in range(len(self._seg_p0)):
            a, b = self._seg_p0[i], self._seg_p1[i]
            length = float(np.linalg.norm(b - a))
            count = max(2, int(round(length / _SAMPLE_DS)) + 1)
            frac = np.linspace(0.0, 1.0, count)[:, None]
            bases.append(a[None, :] + frac * (b - a)[None, :])
            sflags.append(np.full(count, self._seg_flag[i]))
        self._samp_base = np.concatenate(bases, axis=0)
        self._samp_flag = np.concatenate(sflags, axis=0)

    def _prepare_motion(self, cfg) -> None:
        bc = cfg.get("boundary_conditions", {}).get("upper_pin_array", {})
        axis = np.asarray(bc.get("axis", [0.0, 0.0, -1.0]), dtype=float).reshape(3)
        self._upper_axis = axis
        keyframes = bc.get("keyframes", [])
        times = np.asarray([float(k["time"]) for k in keyframes], dtype=float)
        disps = np.asarray([float(k["displacement"]) for k in keyframes], dtype=float)
        if len(times) == 0:
            times = np.array([0.0, 1.0])
            disps = np.array([0.0, 0.0])
        self._kf_t = times
        self._kf_d = disps

    def _upper_disp(self, t: float) -> float:
        return float(np.interp(t, self._kf_t, self._kf_d))

    def _upper_rate(self, t: float) -> float:
        times, disps = self._kf_t, self._kf_d
        if t <= times[0] or t >= times[-1]:
            return 0.0
        idx = int(np.searchsorted(times, t, side="right") - 1)
        idx = min(max(idx, 0), len(times) - 2)
        span = times[idx + 1] - times[idx]
        if span <= 0.0:
            return 0.0
        return float((disps[idx + 1] - disps[idx]) / span)

    # ------------------------------------------------------------- kinematics
    def _deformation(self, x):
        tets = self._tets
        d0 = x[tets[:, 0]]
        ds = np.stack(
            (x[tets[:, 1]] - d0, x[tets[:, 2]] - d0, x[tets[:, 3]] - d0), axis=2
        )
        return ds @ self._Dm_inv

    def _piola(self, F):
        j = np.linalg.det(F)
        jc = np.maximum(j, _J_FLOOR)
        finv = np.linalg.inv(F + 1.0e-10 * np.eye(3)[None, :, :])
        finvt = finv.transpose(0, 2, 1)
        lnj = np.log(jc)
        P = self._mu * (F - finvt) + self._lam * lnj[:, None, None] * finvt
        psi = (
            0.5 * self._mu * (np.trace(F.transpose(0, 2, 1) @ F, axis1=1, axis2=2) - 3.0)
            - self._mu * lnj
            + 0.5 * self._lam * lnj * lnj
        )
        return P, psi

    def _elastic_forces(self, x):
        P, _ = self._piola(self._deformation(x))
        W = P @ self._Dm_inv                      # (Ne,3,3)
        f = np.zeros((self._n, 3), dtype=float)
        tets = self._tets
        vol = self._V0[:, None]
        for j in range(3):
            np.add.at(f, tets[:, j + 1], -vol * W[:, :, j])
        np.add.at(f, tets[:, 0], vol * np.sum(W, axis=2))
        return f

    # ---------------------------------------------------------------- contact
    def _closest_on_triangles(self, p, a, b, c):
        """Closest point on each triangle (broadcast) - Ericson's algorithm."""
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = _dot(ab, ap)
        d2 = _dot(ac, ap)
        bp = p - b
        d3 = _dot(ab, bp)
        d4 = _dot(ac, bp)
        cp = p - c
        d5 = _dot(ab, cp)
        d6 = _dot(ac, cp)
        vc = d1 * d4 - d3 * d2
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4
        denom = va + vb + vc
        safe = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        q = a + ab * (vb / safe)[..., None] + ac * (vc / safe)[..., None]

        den = d1 - d3
        t_ab = np.where(np.abs(den) < 1e-30, 0.0, d1 / np.where(np.abs(den) < 1e-30, 1.0, den))
        q_ab = a + ab * t_ab[..., None]
        den = d2 - d6
        w_ac = np.where(np.abs(den) < 1e-30, 0.0, d2 / np.where(np.abs(den) < 1e-30, 1.0, den))
        q_ac = a + ac * w_ac[..., None]
        den = (d4 - d3) + (d5 - d6)
        t_bc = (d4 - d3) / np.where(np.abs(den) < 1e-30, 1.0, den)
        q_bc = b + (c - b) * t_bc[..., None]

        cond_a = (d1 <= 0.0) & (d2 <= 0.0)
        cond_b = (d3 >= 0.0) & (d4 <= d3)
        cond_c = (d6 >= 0.0) & (d5 <= d6)
        cond_ab = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        cond_ac = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
        cond_bc = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)

        q = np.where(cond_bc[..., None], q_bc, q)
        q = np.where(cond_ac[..., None], q_ac, q)
        q = np.where(cond_ab[..., None], q_ab, q)
        q = np.where(cond_c[..., None], c, q)
        q = np.where(cond_b[..., None], b, q)
        q = np.where(cond_a[..., None], a, q)
        return q

    def _contact(self, x, v, t):
        """Penalty contact of the pin samples against the deformed ball surface."""
        f = np.zeros((self._n, 3), dtype=float)
        damp = np.zeros(self._n, dtype=float)
        disp = self._upper_disp(t)
        rate = self._upper_rate(t)
        pts = self._samp_base + self._samp_flag[:, None] * (disp * self._upper_axis)
        vpts = (self._samp_flag * rate)[:, None] * self._upper_axis[None, :]

        # activity pruning: only pins near the deformed ball
        diff = pts[:, None, :] - x[None, :, :]
        near = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
        active = near < _PRUNE
        if not np.any(active):
            return f, damp

        p = pts[active][:, None, :]
        faces = self._faces
        a = x[faces[:, 0]][None, :, :]
        b = x[faces[:, 1]][None, :, :]
        c = x[faces[:, 2]][None, :, :]
        q = self._closest_on_triangles(p, a, b, c)          # (M,T,3)
        d2 = np.sum((p - q) ** 2, axis=2)                   # (M,T)
        idx = np.argmin(d2, axis=1)
        rows = np.arange(len(idx))
        tri = faces[idx]
        qs = q[rows, idx]
        pv = pts[active]
        vpv = vpts[active]

        A = x[tri[:, 0]]
        B = x[tri[:, 1]]
        C = x[tri[:, 2]]
        e0 = B - A
        e1 = C - A
        e2 = qs - A
        d00 = _dot(e0, e0)
        d01 = _dot(e0, e1)
        d11 = _dot(e1, e1)
        d20 = _dot(e2, e0)
        d21 = _dot(e2, e1)
        den = d00 * d11 - d01 * d01
        den = np.where(np.abs(den) < 1e-30, 1e-30, den)
        w1 = (d11 * d20 - d01 * d21) / den
        w2 = (d00 * d21 - d01 * d20) / den
        w0 = 1.0 - w1 - w2
        weights = np.stack((w0, w1, w2), axis=1)            # (M,3)

        nrm = np.cross(e0, e1)
        nlen = np.sqrt(np.sum(nrm * nrm, axis=1))
        nlen = np.where(nlen < 1e-30, 1e-30, nlen)
        nrm = nrm / nlen[:, None]
        dist = np.sqrt(np.sum((qs - pv) ** 2, axis=1))
        dist = np.maximum(dist, 1e-12)
        u = (qs - pv) / dist[:, None]
        inside = _dot(pv - qs, nrm) < 0.0
        phi = np.where(inside, -dist, dist)
        delta = self._contact_r - phi
        mask = delta > 0.0
        if not np.any(mask):
            return f, damp

        sigma = _dot(u, (weights[:, 0, None] * v[tri[:, 0]]
                         + weights[:, 1, None] * v[tri[:, 1]]
                         + weights[:, 2, None] * v[tri[:, 2]]) - vpv)
        fmag = self._contact_k * delta + self._c_contact * np.maximum(-sigma, 0.0)
        fmag = np.where(mask, fmag, 0.0)
        for k in range(3):
            np.add.at(f, tri[:, k], (fmag * weights[:, k])[:, None] * u)
        cnode = np.where(mask, self._node_c_contact, 0.0)
        for k in range(3):
            np.add.at(damp, tri[:, k], cnode * weights[:, k])
        return f, damp

    def _node_pin_contact(self, x, v, t):
        """Secondary penalty keeping ball nodes outside the pin cylinders."""
        f = np.zeros((self._n, 3), dtype=float)
        damp = np.zeros(self._n, dtype=float)
        disp = self._upper_disp(t)
        rate = self._upper_rate(t)
        p0 = self._seg_p0 + self._seg_flag[:, None] * (disp * self._upper_axis)
        p1 = self._seg_p1 + self._seg_flag[:, None] * (disp * self._upper_axis)
        e = p1 - p0
        ee = np.sum(e * e, axis=1)
        ee = np.where(ee < 1e-30, 1e-30, ee)
        w = x[:, None, :] - p0[None, :, :]                  # (N,S,3)
        tw = np.clip(np.sum(w * e[None, :, :], axis=2) / ee[None, :], 0.0, 1.0)
        cpt = p0[None, :, :] + tw[..., None] * e[None, :, :]
        dvec = x[:, None, :] - cpt
        dist = np.sqrt(np.sum(dvec * dvec, axis=2))
        dist = np.maximum(dist, 1e-12)
        delta = self._contact_r - dist
        vseg = (self._seg_flag * rate)[:, None] * self._upper_axis[None, :]
        sigma = np.sum((dvec / dist[:, :, None]) * (v[:, None, :] - vseg[None, :, :]), axis=2)
        fmag = self._contact_k * delta + self._c_contact * np.maximum(-sigma, 0.0)
        fmag = np.where(delta > 0.0, fmag, 0.0)
        f = np.sum(fmag[:, :, None] * dvec / dist[:, :, None], axis=1)
        damp = np.sum(np.where(delta > 0.0, self._node_c_contact, 0.0), axis=1)
        return f, damp

    def _step(self, h):
        x = self._x
        v = self._v
        f = self._elastic_forces(x)
        fc, damp = self._contact(x, v, self._t)
        f = f + fc + self._m[:, None] * self._g[None, :]
        fnp, dnp = self._node_pin_contact(x, v, self._t)
        f = f + fnp
        damp = damp + dnp
        denom = self._m + h * (self._c_global * self._m + damp)
        v = (self._m[:, None] * v + h * f) / denom[:, None]
        x = x + h * v
        self._x = x
        self._v = v
        contact_total = np.sum(fc + fnp, axis=0)
        self._impulse = self._impulse + h * (self._gravity_total + contact_total)
        self._last_contact_total = contact_total.copy()

    def _frame(self):
        x = self._x
        F = self._deformation(x)
        P, psi = self._piola(F)
        energy = np.zeros(self._n, dtype=float)
        for j in range(4):
            np.add.at(energy, self._tets[:, j], self._V0 * psi / 4.0)
        contact = self._last_contact_total
        fields = {
            "budget_cumulative_external_impulse": self._impulse[None, :].copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self._n)),
            "budget_external_force": np.stack(
                (self._gravity_total, contact), axis=0
            )[None, :, :].copy(),
            "cumulative_mass_source": np.zeros(1),
            "element_deformation_gradient": F.copy(),
            "elastic_internal_force": self._elastic_forces(x),
            "elastic_first_piola_stress": P.copy(),
            "gravity_acceleration": self._g[None, :].copy(),
        }
        attributes = {
            "mass_quantity": self._m.copy(),
            "rest_position": self._rest.copy(),
            "elastic_energy": energy,
        }
        scalars = {
            "reference_momentum": float(self._ref_momentum),
            "density": float(self._density),
            "contact_stiffness": float(self._contact_k),
        }
        return StateFrame(
            time=float(self._t),
            position=x.copy(),
            velocity=self._v.copy(),
            mass=self._m.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )

    # ----------------------------------------------------------------- public
    def advance_to(self, time: float):
        if not self._initialized:
            raise RuntimeError("initialize() must be called before advance_to()")
        target = float(time)
        if target < self._t - 1e-9:
            raise ValueError("advance_to() requires non-decreasing times")
        while self._t < target - 1e-12:
            h = min(self._h, target - self._t)
            self._step(h)
            self._t += h
        self._t = target
        return self._frame()

    def diagnostics(self) -> dict:
        return {
            "nodes": int(self._n),
            "elements": int(len(self._tets)),
            "time_step": float(self._h),
            "mass_total": float(self._mass_total),
            "reference_momentum": float(self._ref_momentum),
            "simulation_time": float(self._t),
        }
