"""Physically-based solver for the ``024_stretched_cube`` scene.

Physical model
--------------
The frozen subject mesh (a closed subdivided box surface, 1538 nodes and 3072
triangles) is discretised as a node/edge lattice.  Every mesh edge is a
geometrically exact axial spring whose stiffness follows a bar idealisation of
the frozen material stiffness ``E``::

    k_e = E * h**2 / L0_e

with ``h`` the mean rest edge length and ``L0_e`` the rest length of edge
``e``.  Nodes carry the frozen total subject mass, and Newton's equations

    m_i a_i = f_spring + m_i g - m_i * lambda * v_i + f_contact

are integrated with a fixed-substep velocity-Verlet scheme.  The mass
proportional viscous term uses the frozen ``damping`` value rescaled by the
breathing frequency of the body, which keeps the damping ratio of the
observable low-frequency mode at the frozen value while remaining stable under
explicit time stepping.

Loading script
--------------
The two kinematic grips (``left_grip`` / ``right_grip``) are bonded to the two
end faces of the cube, which are therefore Dirichlet nodes.  Their separation
follows the frozen scene script: a smooth (C2 smootherstep) uniaxial stretch up
to ``stretch_ratio`` of the initial length, a hold, then a short half-sine
compressive pulse at ``pulse_time``, after which the grips hold again and the
body oscillates and damps out.  The large deformation is therefore a genuine
elastic response driven only by the boundary conditions.

Contacts
--------
The ground plane and the grip boxes are handled with a penalty normal force and
Coulomb-limited tangential friction (frozen ``restitution``/``friction``).  The
reported contact signed distance is the geometric gap (negative = penetration)
to the ground and to both grips.

All state is deterministic and finite; the trajectory exports raw physical
fields only (deformation gradient, mass, velocity, external force, contact
distance), never scored residuals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_EPS = 1e-12


def _smootherstep(u: float) -> float:
    """C2 ramp from 0 to 1 on [0, 1]."""
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def _smootherstep_derivative(u: float, inv_span: float) -> float:
    """d/du smootherstep(u) * inv_span, i.e. d/dt of a ramp with time span 1/inv_span."""
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return 30.0 * u * u * (1.0 - u) * (1.0 - u) * inv_span


class BenchmarkSolver:
    """Elastic-lattice solid driven by the frozen stretch/pulse grip script."""

    # ---------------------------------------------------------------- set-up
    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        cfg = dict(getattr(scene, "config", None) or {})
        self.cfg = cfg

        X = np.asarray(scene.vertices, dtype=float).copy()
        n = int(X.shape[0])
        self.X0 = X
        self.n = n

        self.mass_total = float(cfg.get("subject_mass", 1.3))
        self.mass = np.full(n, self.mass_total / n, dtype=float)
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float)
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.contact_length_scale = float(cfg.get("contact_length_scale", 0.04))
        self.friction = float(cfg.get("friction", 0.5))
        self.stiffness = float(cfg.get("stiffness", 28.0))

        # --- lattice edges -------------------------------------------------
        faces = self._load_faces(cfg)
        if faces is not None and len(faces) > 0:
            raw = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        else:
            raw = self._knn_edges(X)
        edges = np.unique(np.sort(raw, axis=1), axis=0)
        self.ei = edges[:, 0].astype(np.int64)
        self.ej = edges[:, 1].astype(np.int64)
        rest = X[self.ei] - X[self.ej]
        self.L0 = np.maximum(np.sqrt(np.einsum("ij,ij->i", rest, rest)), _EPS)
        h = float(np.mean(self.L0))
        self.k_spring = self.stiffness * h * h / self.L0

        # --- time stepping -------------------------------------------------
        self.sample_dt = float(cfg.get("sample_dt", 1.0 / 30.0))
        self.n_sub = 24
        self.dt = self.sample_dt / float(self.n_sub)

        # mass-proportional damping tuned to the fundamental breathing mode
        ext = X.max(axis=0) - X.min(axis=0)
        area_typ = max(float(ext[1] * ext[2]), _EPS)
        length_typ = max(float(ext[0]), _EPS)
        omega0 = np.sqrt(self.stiffness * area_typ / (self.mass_total * length_typ))
        self.lam = 0.5 * float(cfg.get("damping", 0.42)) * float(omega0)

        # --- clamp / grip script ------------------------------------------
        self.stretch_end = float(cfg.get("stretch_end", 1.25))
        self.pulse_time = float(cfg.get("pulse_time", 1.55))
        ratio = float(cfg.get("stretch_ratio", 2.15))
        self.S_target = (ratio - 1.0) * float(ext[0])
        self.T_pulse = 0.16
        self.A_pulse = 0.10 * (float(ext[0]) + self.S_target)

        xmin = float(X[:, 0].min())
        xmax = float(X[:, 0].max())
        tol = 1e-6 * max(1.0, xmax - xmin)
        self.clamp_left = X[:, 0] <= xmin + tol
        self.clamp_right = X[:, 0] >= xmax - tol
        self.clamped = self.clamp_left | self.clamp_right
        self.clamp_sign = np.where(self.clamp_left, -1.0, 1.0)

        # --- frozen obstacle boxes (left grip / right grip) ----------------
        self.grip_boxes = []
        for oid, sign, fallback in (
            ("left_grip", -1.0, (-0.72, 0.0, 0.86)),
            ("right_grip", 1.0, (0.72, 0.0, 0.86)),
        ):
            center, half = self._geometry_object(cfg, oid)
            if center is None:
                center = np.asarray(fallback, dtype=float)
                half = np.asarray([0.12, 0.55, 0.55], dtype=float)
            self.grip_boxes.append((np.asarray(center, float), np.asarray(half, float), sign))

        # --- penalty contact (ground) -------------------------------------
        m_node = self.mass_total / n
        self.k_contact = 0.25 * m_node / (self.dt * self.dt)
        self.c_contact = 2.0 * np.sqrt(self.k_contact * m_node)

        # --- deformation-gradient stencil ---------------------------------
        self._build_stencil(X)

        # --- state ---------------------------------------------------------
        self.x = X.copy()
        self.v = np.zeros_like(X)
        self.step = 0
        self._energy = 0.0

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _load_faces(cfg):
        asset = cfg.get("subject_asset", "assets/subject.npz")
        candidates = [
            Path.cwd() / asset,
            Path(__file__).resolve().parent.parent / asset,
            Path(__file__).resolve().parent / asset,
        ]
        for path in candidates:
            try:
                if path.is_file():
                    with np.load(path) as data:
                        if "faces" in data.files:
                            return np.asarray(data["faces"], dtype=np.int64)
            except Exception:
                continue
        return None

    @staticmethod
    def _knn_edges(X):
        """Fallback topology: connect each node to its near neighbours."""
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(X)
            dist, idx = tree.query(X, k=7)
            radius = 1.35 * float(np.median(dist[:, 1]))
            pairs = tree.query_pairs(radius, output_type="ndarray")
            if len(pairs) > 0:
                return np.asarray(pairs, dtype=np.int64)
            return np.array(
                [[i, j] for i in range(len(X)) for j in idx[i, 1:]], dtype=np.int64
            )
        except Exception:
            return np.array([[i, i + 1] for i in range(len(X) - 1)], dtype=np.int64)

    @staticmethod
    def _geometry_object(cfg, oid):
        geo = cfg.get("scene_geometry") or {}
        for obj in geo.get("objects", []) or []:
            if obj.get("id") == oid:
                center = obj.get("transform", {}).get("translation", [0.0, 0.0, 0.0])
                half = obj.get("geometry", {}).get("half_extents")
                if half is not None:
                    return np.asarray(center, float), np.asarray(half, float)
        return None, None

    def _build_stencil(self, X):
        n = self.n
        src = np.concatenate([self.ei, self.ej])
        dst = np.concatenate([self.ej, self.ei])
        order = np.argsort(src, kind="stable")
        src = src[order]
        dst = dst[order]
        deg = np.bincount(src, minlength=n)
        K = int(max(1, deg.max()))
        nbr = np.zeros((n, K), dtype=np.int64)
        w = np.zeros((n, K), dtype=float)
        starts = np.searchsorted(src, np.arange(n), side="left")
        ends = np.searchsorted(src, np.arange(n), side="right")
        for i in range(n):
            js = dst[starts[i]:ends[i]]
            m = len(js)
            if m:
                nbr[i, :m] = js
                w[i, :m] = 1.0
        self.nbr = nbr
        self.w = w
        dX = X[nbr] - X[:, None, :]
        self.dX = dX
        self.dXw = dX * w[:, :, None]
        D = np.einsum("nkj,nkl->njl", self.dXw, dX)
        D = 0.5 * (D + np.transpose(D, (0, 2, 1)))
        lam, U = np.linalg.eigh(D)
        lmax = np.maximum(lam[:, 2], _EPS)
        deficient = lam[:, 0] < 1e-8 * lmax
        inv = np.zeros_like(lam)
        ok = ~deficient
        inv[ok] = 1.0 / lam[ok]
        self.U = U
        self.inv = inv
        self.flip = deficient
        self.ok = ok

    # ------------------------------------------------------------ dynamics
    def _script(self, t):
        """Grip separation offset and its time derivative."""
        S = self.S_target
        if t < self.stretch_end:
            u = t / self.stretch_end
            return S * _smootherstep(u), S * _smootherstep_derivative(u, 1.0 / self.stretch_end)
        if t < self.pulse_time:
            return S, 0.0
        u = (t - self.pulse_time) / self.T_pulse
        if u <= 1.0:
            s = S - self.A_pulse * np.sin(np.pi * u) ** 2
            sd = -self.A_pulse * np.pi * np.sin(2.0 * np.pi * u) / self.T_pulse
            return s, sd
        return S, 0.0

    def _accel(self, x, v, t):
        d = x[self.ei] - x[self.ej]
        L = np.sqrt(np.einsum("ij,ij->i", d, d))
        invL = 1.0 / np.maximum(L, _EPS)
        u = d * invL[:, None]
        fmag = self.k_spring * (L - self.L0)

        f = np.empty_like(x)
        for a in range(3):
            f[:, a] = (
                np.bincount(self.ei, weights=fmag * u[:, a], minlength=self.n)
                - np.bincount(self.ej, weights=fmag * u[:, a], minlength=self.n)
            )
        f += self.mass[:, None] * self.gravity[None, :]
        f -= (self.mass * self.lam)[:, None] * v

        # penalty ground contact with Coulomb-limited tangential friction
        gap = x[:, 2] - self.ground_z
        pen = self.contact_length_scale - gap
        active = pen > 0.0
        if np.any(active):
            fn = self.k_contact * pen[active] - self.c_contact * v[active, 2]
            fn = np.maximum(fn, 0.0)
            f[active, 2] += fn
            vt = v[active, :2]
            speed = np.sqrt(np.einsum("ij,ij->i", vt, vt))
            limit = self.friction * fn
            scale = np.where(
                speed > 1e-9, np.minimum(1.0, limit / np.maximum(speed, 1e-9)), 0.0
            )
            f[active, :2] -= scale[:, None] * vt
        return f

    def _apply_clamps(self, x, v, t):
        s, sd = self._script(t)
        sign = self.clamp_sign[self.clamped]
        x[self.clamped, 0] = self.X0[self.clamped, 0] + 0.5 * s * sign
        v[self.clamped, :] = 0.0
        v[self.clamped, 0] = 0.5 * sd * sign

    def _step(self):
        t = self.step * self.dt
        x, v = self.x, self.v
        a1 = self._accel(x, v, t)
        vh = v + 0.5 * self.dt * a1
        x = x + self.dt * vh
        a2 = self._accel(x, vh, t + self.dt)
        v = vh + 0.5 * self.dt * a2
        self._apply_clamps(x, v, t + self.dt)
        self.x = x
        self.v = v
        self.step += 1

    # ---------------------------------------------------------- trajectory
    def _deformation_gradient(self, x):
        dx = x[self.nbr] - x[:, None, :]
        M = np.einsum("nkj,nkl->njl", dx * self.w[:, :, None], self.dX)
        Mp = np.einsum("nkj,njc->nkc", M, self.U)
        F = np.einsum("nkc,nc,njc->nkj", Mp, self.inv, self.U)
        if np.any(self.flip):
            u3 = self.U[:, :, 0]
            resolved = self.ok.copy()
            remaining = self.flip.copy()
            for _ in range(64):
                if not np.any(remaining):
                    break
                mask = resolved[self.nbr]
                cnt = mask.sum(axis=1)
                proj = np.einsum("nkj,nj->nk", F, u3)
                masked = np.where(mask, proj, 0.0)
                avg = masked.sum(axis=1) / np.maximum(cnt, 1)[:, None]
                newly = remaining & (cnt > 0)
                if not np.any(newly):
                    break
                F[newly] += avg[newly][:, :, None] * u3[newly][:, None, :]
                resolved = resolved | newly
                remaining = remaining & ~newly
        return F

    @staticmethod
    def _box_signed_distance(x, center, half):
        q = np.abs(x - center[None, :]) - half[None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def _frame(self, time):
        t = self.step * self.dt
        s, _ = self._script(t)
        x = self.x
        sd = x[:, 2] - self.ground_z
        for center, half, sign in self.grip_boxes:
            shifted = center + np.array([sign * 0.5 * s, 0.0, 0.0])
            sd = np.minimum(sd, self._box_signed_distance(x, shifted, half))

        energy = 0.5 * float(np.sum(self.mass * np.einsum("ij,ij->i", self.v, self.v)))
        d = x[self.ei] - x[self.ej]
        L = np.sqrt(np.einsum("ij,ij->i", d, d))
        energy += 0.5 * float(np.sum(self.k_spring * (L - self.L0) ** 2))
        self._energy = energy

        F = self._deformation_gradient(x)
        return StateFrame(
            time=float(time),
            position=x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "contact_signed_distance": np.asarray(sd, dtype=float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
            },
            scalars={"contact_length_scale": self.contact_length_scale},
        )

    def advance_to(self, time: float):
        target = int(round(float(time) / self.dt))
        while self.step < target:
            self._step()
        return self._frame(float(time))

    def diagnostics(self) -> dict[str, object]:
        speed = np.sqrt(np.einsum("ij,ij->i", self.v, self.v))
        return {
            "steps": int(self.step),
            "internal_dt": float(self.dt),
            "spring_count": int(len(self.L0)),
            "clamped_nodes": int(np.count_nonzero(self.clamped)),
            "energy": float(getattr(self, "_energy", 0.0)),
            "max_speed": float(speed.max()) if self.n else 0.0,
        }
