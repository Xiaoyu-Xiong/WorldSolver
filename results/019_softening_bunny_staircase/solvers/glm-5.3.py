"""Viscoplastic softening bunny on a staircase.

Physical model
--------------
The deformable subject is discretized as a dense mass-spring network built
from the frozen subject mesh vertices: every vertex pair closer than a
material radius is connected by a spring (short surface edges are retained
with priority), which yields an isotropic, volume-coherent network.

Each substep performs genuine explicit (symplectic Euler) time integration:

  * elastic spring forces  F = k (L - L0) with per-spring rest length L0,
  * viscous damping along each spring (Rayleigh-type, subcritical),
  * gravity from the frozen config,
  * plasticity: when the spring strain exceeds a yield strain the rest
    length shifts permanently (plastic flow),
  * work softening: accumulated plastic work reduces the spring stiffness
    exponentially (softening_rate from the frozen config), so repeated
    stair impacts progressively soften the impacted regions which then
    droop over the steps,
  * contact with the frozen staircase/ground boxes via a signed distance
    field: penetration projection, restitution and Coulomb friction, with
    the contact impulse accumulated for reporting.

Per-vertex deformation gradients are recovered by a weighted least-squares
fit of the material neighborhood onto the current neighborhood (with a
ridge toward the identity for the surface-normal direction).

All parameters come from the frozen configuration; no keyframes, no
precomputed motion, no external data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

try:  # optional fast neighbor search; brute force fallback below
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None

# Fallback geometry (identical to the frozen scene_geometry.yaml), used only
# if the frozen file cannot be read.
_FALLBACK_BOXES = (
    ((0.0, 0.0, -0.04), (3.0, 1.6, 0.04)),
    ((-0.75, 0.0, 0.96), (0.36, 1.05, 0.14)),
    ((-0.15, 0.0, 0.68), (0.36, 1.05, 0.14)),
    ((0.45, 0.0, 0.40), (0.36, 1.05, 0.14)),
    ((1.05, 0.0, 0.12), (0.36, 1.05, 0.14)),
)


def _load_boxes() -> list[tuple[np.ndarray, np.ndarray]]:
    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        import yaml

        path = Path(__file__).resolve().parents[1] / "scene_geometry.yaml"
        data = yaml.safe_load(path.read_text())
        for obj in data.get("objects", []):
            geo = obj.get("geometry", {}) or {}
            if geo.get("type") != "box":
                continue
            if not bool(obj.get("collision_enabled", True)):
                continue
            he = np.asarray(geo["half_extents"], dtype=float)
            tr = obj.get("transform", {}) or {}
            tr = np.asarray(tr.get("translation", (0.0, 0.0, 0.0)), dtype=float)
            if he.shape == (3,) and tr.shape == (3,):
                boxes.append((tr, he))
    except Exception:
        boxes = []
    if not boxes:
        boxes = [
            (np.asarray(c, dtype=float), np.asarray(h, dtype=float))
            for c, h in _FALLBACK_BOXES
        ]
    return boxes


class BenchmarkSolver:
    """Explicit viscoplastic mass-spring solver for the bunny staircase."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        self._seed = int(seed)

        X0 = np.array(np.asarray(scene.vertices, dtype=float), dtype=float)
        n = len(X0)
        self._X0 = X0
        self._n = n

        g = np.asarray(cfg.get("gravity", (0.0, 0.0, -9.81)), dtype=float)
        if g.shape != (3,):
            g = np.asarray((0.0, 0.0, -9.81), dtype=float)
        self._g = g

        total_mass = float(cfg.get("subject_mass", 1.0))
        m_v = total_mass / n
        self._m = np.full(n, m_v, dtype=float)
        self._m_v = m_v

        v0 = np.asarray(cfg.get("initial_velocity", (0.0, 0.0, 0.0)), dtype=float)
        if v0.shape != (3,):
            v0 = np.zeros(3)
        self._v = np.tile(v0, (n, 1))
        self._x = X0.copy()

        stiffness = float(cfg.get("stiffness", 20.0))
        stiffness = max(stiffness, 1.0)
        self._rate = float(cfg.get("softening_rate", 0.3))
        damping = float(cfg.get("damping", 1.0))
        self._e = float(cfg.get("restitution", 0.1))
        self._mu = float(cfg.get("friction", 0.5))
        self._cls = float(cfg.get("contact_length_scale", 0.04))
        self._gz = float(cfg.get("ground_z", 0.0))
        self._boxes = _load_boxes()

        # ---- spring network -------------------------------------------------
        if cKDTree is not None:
            tree = cKDTree(X0)
            dnn, _ = tree.query(X0, k=2)
            h = float(np.mean(dnn[:, 1]))
            radius = 2.8 * h
            pairs = tree.query_pairs(radius, output_type="ndarray")
        else:  # brute force neighborhood
            diff = X0[:, None, :] - X0[None, :, :]
            dist = np.sqrt((diff * diff).sum(axis=2))
            dnn = np.sort(dist, axis=1)
            h = float(np.mean(dnn[:, 1]))
            radius = 2.8 * h
            ii, jj = np.nonzero(np.triu(dist <= radius, 1))
            pairs = np.stack((ii, jj), axis=1)
        if len(pairs) == 0:
            raise ValueError("empty spring network")

        lp = np.linalg.norm(X0[pairs[:, 0]] - X0[pairs[:, 1]], axis=1)
        order = np.argsort(lp, kind="stable")
        pairs = pairs[order]

        cap = 44  # per-vertex spring cap -> bounded explicit time step
        counts = [0] * n
        kept: list[tuple[int, int]] = []
        for i, j in pairs.tolist():
            if counts[i] < cap and counts[j] < cap:
                counts[i] += 1
                counts[j] += 1
                kept.append((i, j))
        spr = np.asarray(kept, dtype=np.int64).reshape(-1, 2)
        self._a = spr[:, 0].copy()
        self._b = spr[:, 1].copy()
        s = len(self._a)
        self._s = s

        Li = np.linalg.norm(X0[self._a] - X0[self._b], axis=1)
        Li = np.maximum(Li, 1e-9)
        self._Li = Li
        self._L0 = Li.copy()
        self._W = np.zeros(s, dtype=float)
        self._soft = np.ones(s, dtype=float)

        # Constitutive calibration from the frozen config. ``stiffness`` is
        # interpreted as the reciprocal design static strain: the elastic
        # modulus is chosen so the self-weight stress produces a static
        # strain of 1/stiffness, and the yield strain is 2/stiffness so the
        # body holds its shape while falling but yields under impacts.
        gmag = float(np.linalg.norm(g))
        gmag = max(gmag, 1e-6)
        extent = X0.max(axis=0) - X0.min(axis=0)
        width = float(extent[1]) if len(extent) >= 2 else 0.1
        area = max(width * width, 1e-4)
        sigma_char = total_mass * gmag / area
        E = stiffness * sigma_char
        self._k0 = max(E * h, 1e-6)
        self._eps_y = 2.0 / stiffness
        # reference plastic energy per spring (softening normalization)
        self._w_ref = 3.0 * self._k0 * (Li * Li) * (self._eps_y**2)
        self._k = np.full(s, self._k0, dtype=float)
        self._zeta = min(0.5, max(damping, 0.0) * 0.1)

        # ---- stable explicit time step --------------------------------------
        deg = np.bincount(self._a, minlength=n) + np.bincount(self._b, minlength=n)
        max_ksum = max(float(deg.max()) * self._k0, 1e-9)
        dt = 0.2 * 2.0 * np.sqrt(m_v / max_ksum)  # well inside the Euler limit
        dt = float(min(dt, 2.5e-4))
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1.0e-5
        self._dt = dt

        self._t = 0.0
        self._nsub = 0
        self._contact_impulse = np.zeros(3, dtype=float)
        self._last_frame_t = 0.0
        self._ready = True

    # ------------------------------------------------------------------
    # signed distance to the frozen obstacles
    # ------------------------------------------------------------------
    def _sdf(self, pts: np.ndarray):
        m = len(pts)
        best_d = np.full(m, 1e18, dtype=float)
        best_n = np.zeros((m, 3), dtype=float)
        ar = np.arange(m)
        for c, hx in self._boxes:
            q = pts - c[None, :]
            a = np.abs(q) - hx[None, :]
            amax = a.max(axis=1)
            outside = amax > 0.0
            d_out = np.sqrt((np.maximum(a, 0.0) ** 2).sum(axis=1))
            d = np.where(outside, d_out, amax)  # amax < 0 inside
            # outward normal
            no = q * (a > 0.0)
            nn = np.sqrt((no * no).sum(axis=1))
            ok = outside & (nn > 1e-12)
            no = no / np.maximum(nn, 1e-12)[:, None]
            k = np.argmax(a, axis=1)
            ni = np.zeros((m, 3), dtype=float)
            ni[ar, k] = np.sign(q[ar, k])
            nrm = np.where(ok[:, None], no, ni)
            upd = d < best_d
            best_d = np.where(upd, d, best_d)
            best_n = np.where(upd[:, None], nrm, best_n)
        # safety ground plane from the frozen config
        dp = pts[:, 2] - self._gz
        upd = dp < best_d
        if np.any(upd):
            best_d = np.where(upd, dp, best_d)
            zhat = np.zeros((m, 3), dtype=float)
            zhat[:, 2] = 1.0
            best_n = np.where(upd[:, None], zhat, best_n)
        return best_d, best_n

    # ------------------------------------------------------------------
    # one explicit physical substep
    # ------------------------------------------------------------------
    def _step(self, h: float) -> None:
        a = self._a
        b = self._b
        x = self._x
        v = self._v

        # spring forces
        u = x[b] - x[a]
        L = np.sqrt((u * u).sum(axis=1))
        Ls = np.maximum(L, 1e-12)
        dirn = u / Ls[:, None]
        k = self._k
        Fmag = k * (L - self._L0)
        dv = v[b] - v[a]
        mu_red = 0.5 * self._m_v
        c_s = 2.0 * self._zeta * np.sqrt(k * mu_red)
        damp = c_s * (dv * dirn).sum(axis=1)
        f = (Fmag + damp)[:, None] * dirn

        idx = np.concatenate((a, b))
        vals = np.concatenate((f, -f))
        force = np.empty((self._n, 3), dtype=float)
        for ax in range(3):
            force[:, ax] = np.bincount(idx, vals[:, ax], minlength=self._n)

        # semi-implicit Euler
        v += h * (self._g[None, :] + force * (1.0 / self._m_v))
        sp = np.sqrt((v * v).sum(axis=1))
        fast = sp > 50.0
        if np.any(fast):
            v[fast] *= (50.0 / sp[fast])[:, None]
        x += h * v

        # contact handling (projection + restitution + Coulomb friction)
        d, nrm = self._sdf(x)
        pen = d < 0.0
        if np.any(pen):
            ip = np.nonzero(pen)[0]
            x[ip] += nrm[ip] * (-d[ip])[:, None]
            vp = v[ip]
            npn = nrm[ip]
            vn = (vp * npn).sum(axis=1)
            Jn = np.where(vn < 0.0, -(1.0 + self._e) * vn, 0.0)
            vp = vp + Jn[:, None] * npn
            vt = vp - (vp * npn).sum(axis=1)[:, None] * npn
            vtm = np.sqrt((vt * vt).sum(axis=1))
            vts = np.maximum(vtm, 1e-12)
            Jt = np.minimum(self._mu * Jn, vtm)
            vp = vp - (Jt / vts)[:, None] * vt
            v[ip] = vp
            imp = self._m_v * (Jn[:, None] * npn - (Jt / vts)[:, None] * vt)
            self._contact_impulse += imp.sum(axis=0)

        # plasticity with work softening
        eps = (L - self._L0) / self._Li
        over = np.abs(eps) - self._eps_y
        ymask = over > 0.0
        if np.any(ymask):
            yi = np.nonzero(ymask)[0]
            dL = np.sign(eps[yi]) * over[yi] * self._Li[yi]
            dW = np.abs(Fmag[yi]) * np.abs(dL)
            self._L0[yi] += dL
            self._W[yi] += dW
            self._soft[yi] = np.exp(-self._rate * self._W[yi] / self._w_ref[yi])
            self._k[yi] = self._k0 * self._soft[yi]

        self._nsub += 1

    # ------------------------------------------------------------------
    # per-vertex deformation gradient (least squares, ridge to identity)
    # ------------------------------------------------------------------
    def _deformation_gradient(self) -> np.ndarray:
        a = self._a
        b = self._b
        u0 = self._X0[b] - self._X0[a]
        ut = self._x[b] - self._x[a]
        uu = u0[:, :, None] * u0[:, None, :]
        vu = ut[:, :, None] * u0[:, None, :]
        A = np.zeros((self._n, 3, 3), dtype=float)
        C = np.zeros((self._n, 3, 3), dtype=float)
        np.add.at(A, a, uu)
        np.add.at(A, b, uu)
        np.add.at(C, a, vu)
        np.add.at(C, b, vu)
        tr = A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]
        lam = 1e-3 * np.maximum(tr, 1e-12) / 3.0 + 1e-12
        eye = np.eye(3)[None, :, :]
        Areg = A + lam[:, None, None] * eye
        Creg = C + lam[:, None, None] * eye
        Ft = np.linalg.solve(Areg, np.swapaxes(Creg, 1, 2))
        F = np.swapaxes(Ft, 1, 2)
        return F

    # ------------------------------------------------------------------
    # frame export
    # ------------------------------------------------------------------
    def _frame(self, t: float) -> StateFrame:
        d, _ = self._sdf(self._x)
        near = d < 3.0 * self._cls
        csd = d[near]
        if csd.size == 0:
            csd = np.asarray([float(d.min())], dtype=float)
        dt_int = t - self._last_frame_t
        if dt_int > 1e-9:
            interaction = (self._contact_impulse / dt_int).reshape(1, 3)
        else:
            interaction = np.zeros((1, 3), dtype=float)
        self._contact_impulse[:] = 0.0
        self._last_frame_t = t

        F = self._deformation_gradient()
        frame = StateFrame(
            time=float(t),
            position=self._x.copy(),
            velocity=self._v.copy(),
            mass=self._m.copy(),
            body_id=np.zeros(self._n, dtype=np.int64),
            attributes={
                "mass_quantity": self._m.copy(),
                "deformation_gradient": F,
            },
            fields={
                "contact_signed_distance": np.ascontiguousarray(csd, dtype=float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "interaction_force": interaction,
                "external_force": self._m[:, None] * self._g[None, :],
            },
            scalars={"contact_length_scale": float(self._cls)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------
    # solver API
    # ------------------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        if not self._ready:
            raise RuntimeError("initialize() must be called before advance_to()")
        t = float(time)
        if t < 0.0:
            t = 0.0
        if t > self._t + 1e-12:
            guard = 0
            limit = 8_000_000
            while self._t < t - 1e-12:
                h = min(self._dt, t - self._t)
                self._step(h)
                self._t += h
                guard += 1
                if guard > limit:
                    break
        return self._frame(t)

    def diagnostics(self) -> dict[str, object]:
        if not self._ready:
            return {}
        return {
            "time": float(self._t),
            "substeps": int(self._nsub),
            "substep_dt": float(self._dt),
            "n_vertices": int(self._n),
            "n_springs": int(self._s),
            "plastic_work_joule": float(self._W.sum()),
            "mean_softening": float(self._soft.mean()),
            "min_softening": float(self._soft.min()),
        }
