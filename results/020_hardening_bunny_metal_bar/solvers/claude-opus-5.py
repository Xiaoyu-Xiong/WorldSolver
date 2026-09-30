"""Viscoplastic deformable-solid solver with work hardening (Genesis-free).

Physical model
--------------
The bunny surface mesh (closed, manifold) is discretized as a viscoplastic
membrane shell enclosing an incompressible interior:

* Structural springs on every mesh edge and bending springs across every
  interior edge carry the deviatoric (shape) response.  Each spring owns a
  *plastic* rest length ``lp`` that flows viscoplastically once its elastic
  strain exceeds a von-Mises-like yield strain.
* Work hardening: each spring accumulates equivalent plastic strain ``q``.
  Both its yield strain and its stiffness grow with ``q``, so strongly
  deformed regions become stiffer / more elastic and stop flowing -- this
  produces the post-impact jiggle and the localized hardening of the
  reference paper.
* A global volumetric penalty (bulk term) resists compression of the enclosed
  volume, giving the body clay-like near-incompressible bulk behaviour.
* Penalty contact with Coulomb friction against the frozen static geometry
  (ground box top plane and the metal-bar cylinder).

Time integration is explicit symplectic (semi-implicit) Euler with a fixed
sub-step ceiling; sub-steps are chosen so every requested sample time is hit
exactly, which makes the trajectory deterministic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_DT_MAX = 1.2e-4          # sub-step ceiling [s]
_VEL_CLAMP = 40.0         # velocity safety clamp [m/s]


def _unit(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / np.maximum(norm, eps)


class BenchmarkSolver:
    """Explicit viscoplastic solid solver for the hardening-bunny scene."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)
        self._rng = np.random.default_rng(int(seed))

        # -------- geometry -------------------------------------------------
        self.X0 = np.asarray(scene.vertices, dtype=float).copy()
        self.faces = self._load_faces(scene, cfg)
        self.n = int(self.X0.shape[0])

        self.x = self.X0.copy()
        v0 = np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float)
        self.v = np.tile(v0, (self.n, 1))

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.total_mass = float(cfg["subject_mass"])
        self.mass = np.full(self.n, self.total_mass / self.n, dtype=float)
        self.inv_mass = 1.0 / self.mass

        # -------- spring topology -----------------------------------------
        edges = self._unique_edges(self.faces)
        bend = self._bending_pairs(self.faces)
        springs = np.vstack([edges, bend]) if len(bend) else edges
        self.sa = springs[:, 0].astype(np.int64)
        self.sb = springs[:, 1].astype(np.int64)
        self.n_spring = int(len(self.sa))
        self.n_edge = int(len(edges))

        self.l0 = np.linalg.norm(self.X0[self.sb] - self.X0[self.sa], axis=1)
        self.l0 = np.maximum(self.l0, 1e-6)
        self.lp = self.l0.copy()          # plastic rest length
        self.q = np.zeros(self.n_spring)  # accumulated equivalent plastic strain

        # -------- material parameters (from frozen config) ----------------
        stiff = float(cfg.get("stiffness", 18.0))
        self.k_struct = 40.0 * stiff                     # [N/m] membrane spring
        self.k_bend = 0.25 * self.k_struct               # [N/m] bending spring
        self.k0 = np.concatenate([
            np.full(self.n_edge, self.k_struct),
            np.full(self.n_spring - self.n_edge, self.k_bend),
        ])
        self.k_vol = 1800.0 * stiff                      # [Pa] bulk penalty
        self.hardening_rate = float(cfg.get("hardening_rate", 1.8))
        self.q_cap = 0.8
        self.yield0 = 0.03                               # yield strain
        self.tau_plastic = 0.03                          # viscoplastic time [s]
        self.beta = 5.7e-4 * float(cfg.get("damping", 1.4))   # stiffness-prop. damping
        self.alpha_drag = 0.2                            # mass-prop. drag [1/s]

        # -------- contact parameters --------------------------------------
        self.k_contact = 2.5e4
        rest = float(np.clip(cfg.get("restitution", 0.28), 1e-3, 0.999))
        lnr = np.log(rest)
        zeta = -lnr / np.sqrt(np.pi ** 2 + lnr ** 2)
        self.c_contact = 2.0 * zeta * np.sqrt(self.k_contact * self.mass[0])
        self.mu = float(cfg.get("friction", 0.55))
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.bar_z = float(cfg.get("bar_z", 0.66))
        self.bar_r = float(cfg.get("bar_radius", 0.085))
        self.contact_length_scale = float(cfg.get("contact_length_scale", 0.04))

        # -------- rest volume & deformation-gradient precompute ------------
        self.V0 = abs(self._enclosed_volume(self.X0))
        self._setup_mls()

        self.t = 0.0
        self.contact_force = np.zeros((self.n, 3))
        self._steps = 0
        self._max_pen = 0.0

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _load_faces(scene, cfg) -> np.ndarray:
        faces = getattr(scene, "faces", None)
        if faces is not None:
            return np.asarray(faces, dtype=np.int64)
        base = Path(getattr(scene, "asset_root", "") or ".")
        candidates = []
        mod = getattr(type(scene), "__module__", None)
        if mod:
            import sys
            module = sys.modules.get(mod)
            mfile = getattr(module, "__file__", None)
            if mfile:
                candidates.append(Path(mfile).resolve().parent / cfg["subject_asset"])
        candidates.append(base / cfg["subject_asset"])
        candidates.append(Path.cwd() / cfg["subject_asset"])
        candidates.append(Path(__file__).resolve().parent.parent / cfg["subject_asset"])
        for path in candidates:
            try:
                if path.is_file():
                    with np.load(path) as data:
                        return np.asarray(data["faces"], dtype=np.int64)
            except Exception:
                continue
        raise FileNotFoundError("subject topology asset not found")

    @staticmethod
    def _unique_edges(faces: np.ndarray) -> np.ndarray:
        e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = np.sort(e, axis=1)
        return np.unique(e, axis=0)

    @staticmethod
    def _bending_pairs(faces: np.ndarray) -> np.ndarray:
        """Opposite-vertex pairs across each interior edge (bending springs)."""
        keys = []
        opp = []
        for i, j, k in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            e = np.sort(faces[:, [i, j]], axis=1)
            keys.append(e)
            opp.append(faces[:, k])
        keys = np.vstack(keys)
        opp = np.concatenate(opp)
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        keys = keys[order]
        opp = opp[order]
        same = np.all(keys[1:] == keys[:-1], axis=1)
        idx = np.nonzero(same)[0]
        pairs = np.stack([opp[idx], opp[idx + 1]], axis=1)
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        pairs = np.sort(pairs, axis=1)
        return np.unique(pairs, axis=0)

    def _enclosed_volume(self, x: np.ndarray) -> float:
        p0 = x[self.faces[:, 0]]
        p1 = x[self.faces[:, 1]]
        p2 = x[self.faces[:, 2]]
        return float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)

    def _vertex_normals(self, x: np.ndarray) -> np.ndarray:
        f = self.faces
        p0, p1, p2 = x[f[:, 0]], x[f[:, 1]], x[f[:, 2]]
        fn = np.cross(p1 - p0, p2 - p0)
        nrm = np.zeros((self.n, 3))
        for c in range(3):
            nrm[:, c] = np.bincount(f.ravel(),
                                    weights=np.repeat(fn[:, c], 3),
                                    minlength=self.n)
        return _unit(nrm)

    # ------------------------------------------- deformation-gradient stencil
    def _setup_mls(self) -> None:
        """Precompute a moving-least-squares stencil for per-vertex grad(u)."""
        ea = self.sa[: self.n_edge]
        eb = self.sb[: self.n_edge]
        idx = np.concatenate([ea, eb])
        nbr = np.concatenate([eb, ea])
        order = np.argsort(idx, kind="stable")
        idx = idx[order]
        nbr = nbr[order]
        counts = np.bincount(idx, minlength=self.n)
        kmax = int(counts.max())
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        slot = np.arange(len(idx)) - np.repeat(starts, counts)

        nb = np.repeat(np.arange(self.n)[:, None], kmax, axis=1)
        w = np.zeros((self.n, kmax))
        nb[idx, slot] = nbr
        dX = self.X0[nb] - self.X0[:, None, :]
        dist2 = np.sum(dX * dX, axis=2)
        wraw = np.zeros((self.n, kmax))
        valid = np.zeros((self.n, kmax), dtype=bool)
        valid[idx, slot] = True
        wraw[valid] = 1.0 / np.maximum(dist2[valid], 1e-12)
        w = wraw / np.maximum(wraw.sum(axis=1, keepdims=True), 1e-30)

        self.mls_nb = nb
        self.mls_w = w
        self.mls_dX = dX

        A = np.einsum("nk,nki,nkj->nij", w, dX, dX)
        n0 = self._vertex_normals(self.X0)
        self.rest_normal = n0
        trace = np.einsum("nii->n", A)
        lam = np.maximum(0.5 * trace / 3.0, 1e-12)
        A_reg = A + lam[:, None, None] * (n0[:, :, None] * n0[:, None, :])
        self.mls_Ainv = np.linalg.inv(A_reg)

        # orthonormal rest tangent basis for area-stretch evaluation
        helper = np.zeros_like(n0)
        amin = np.argmin(np.abs(n0), axis=1)
        helper[np.arange(self.n), amin] = 1.0
        t1 = _unit(np.cross(n0, helper))
        t2 = np.cross(n0, t1)
        self.rest_t1 = t1
        self.rest_t2 = t2

    def _deformation_gradient(self) -> np.ndarray:
        x = self.x
        dx = x[self.mls_nb] - x[:, None, :]
        B = np.einsum("nk,nki,nkj->nij", self.mls_w, dx, self.mls_dX)
        F_raw = np.einsum("nij,njk->nik", B, self.mls_Ainv)
        n0 = self.rest_normal
        # strip the (regularized, unresolved) normal direction -> in-plane part
        P = np.eye(3)[None, :, :] - n0[:, :, None] * n0[:, None, :]
        F_tan = np.einsum("nij,njk->nik", F_raw, P)

        a1 = np.einsum("nij,nj->ni", F_tan, self.rest_t1)
        a2 = np.einsum("nij,nj->ni", F_tan, self.rest_t2)
        J_tan = np.linalg.norm(np.cross(a1, a2), axis=1)
        J_tan = np.clip(J_tan, 1e-4, 1e4)

        V = abs(self._enclosed_volume(x))
        J_glob = float(np.clip(V / max(self.V0, 1e-12), 1e-3, 1e3))
        s = np.clip(J_glob / J_tan, 0.05, 20.0)

        nd = self._vertex_normals(x)
        F = F_tan + s[:, None, None] * (nd[:, :, None] * n0[:, None, :])
        return np.ascontiguousarray(F)

    # ------------------------------------------------------------ force terms
    def _scatter(self, idx: np.ndarray, vals: np.ndarray, out: np.ndarray) -> None:
        for c in range(3):
            out[:, c] += np.bincount(idx, weights=vals[:, c], minlength=self.n)

    def _internal_forces(self, dt: float) -> np.ndarray:
        x, v = self.x, self.v
        d = x[self.sb] - x[self.sa]
        L = np.linalg.norm(d, axis=1)
        L = np.maximum(L, 1e-9)
        u = d / L[:, None]

        k = self.k0 * (1.0 + self.hardening_rate * self.q)
        strain = L / self.lp - 1.0
        f_mag = k * (L - self.lp)
        vrel = np.einsum("ij,ij->i", v[self.sb] - v[self.sa], u)
        f_mag = f_mag + (self.beta * k) * vrel
        f = f_mag[:, None] * u

        forces = np.zeros((self.n, 3))
        idx = np.concatenate([self.sa, self.sb])
        vals = np.concatenate([f, -f], axis=0)
        self._scatter(idx, vals, forces)

        # --- viscoplastic flow with work hardening -------------------------
        yield_eff = self.yield0 * (1.0 + self.hardening_rate * self.q)
        excess = np.abs(strain) - yield_eff
        active = excess > 0.0
        if np.any(active):
            rate = np.where(active, excess / self.tau_plastic, 0.0)
            dq = np.minimum(rate * dt, np.maximum(excess, 0.0))
            self.lp = self.lp * (1.0 + np.sign(strain) * dq)
            np.clip(self.lp, 0.35 * self.l0, 2.8 * self.l0, out=self.lp)
            self.q = np.minimum(self.q + dq, self.q_cap)
        return forces

    def _volume_forces(self, forces: np.ndarray) -> None:
        x = self.x
        f = self.faces
        p0, p1, p2 = x[f[:, 0]], x[f[:, 1]], x[f[:, 2]]
        V = np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0
        lam = self.k_vol * (self.V0 - V) / max(self.V0, 1e-12)
        g0 = np.cross(p1, p2) / 6.0
        g1 = np.cross(p2, p0) / 6.0
        g2 = np.cross(p0, p1) / 6.0
        idx = np.concatenate([f[:, 0], f[:, 1], f[:, 2]])
        vals = lam * np.concatenate([g0, g1, g2], axis=0)
        self._scatter(idx, vals, forces)

    def _signed_distance(self, x: np.ndarray) -> np.ndarray:
        """Min signed gap to the frozen static geometry (ground top, metal bar)."""
        gap_ground = x[:, 2] - self.ground_z
        gap_bar = np.sqrt(x[:, 0] ** 2 + (x[:, 2] - self.bar_z) ** 2) - self.bar_r
        return np.minimum(gap_ground, gap_bar)

    def _contact(self, dt: float, forces: np.ndarray):
        """Penalty normal contact + impulse-limited Coulomb friction."""
        x, v = self.x, self.v
        cforce = np.zeros((self.n, 3))

        # --- obstacle 1: ground top plane (z = ground_z), normal +z ---------
        gap_g = x[:, 2] - self.ground_z
        nrm_g = np.zeros((self.n, 3))
        nrm_g[:, 2] = 1.0

        # --- obstacle 2: metal bar, infinite cylinder along y ---------------
        rx = x[:, 0]
        rz = x[:, 2] - self.bar_z
        rad = np.sqrt(rx * rx + rz * rz)
        rad_safe = np.maximum(rad, 1e-9)
        gap_b = rad - self.bar_r
        nrm_b = np.zeros((self.n, 3))
        nrm_b[:, 0] = rx / rad_safe
        nrm_b[:, 2] = rz / rad_safe

        for gap, nrm in ((gap_g, nrm_g), (gap_b, nrm_b)):
            act = gap < 0.0
            if not np.any(act):
                continue
            ia = np.nonzero(act)[0]
            pen = -gap[ia]
            nn = nrm[ia]
            vn = np.einsum("ij,ij->i", v[ia], nn)
            fn = self.k_contact * pen - self.c_contact * vn
            fn = np.maximum(fn, 0.0)
            fvec = fn[:, None] * nn

            vt = v[ia] - vn[:, None] * nn
            vt_mag = np.linalg.norm(vt, axis=1)
            safe = np.maximum(vt_mag, 1e-9)
            ft_cap = self.mass[ia] * vt_mag / dt
            ft = np.minimum(self.mu * fn, ft_cap)
            fvec = fvec - (ft / safe)[:, None] * vt

            np.add.at(cforce, ia, fvec)
            self._max_pen = max(self._max_pen, float(pen.max()))

        forces += cforce
        return cforce

    # ---------------------------------------------------------------- stepping
    def _step(self, dt: float) -> None:
        forces = self._internal_forces(dt)
        self._volume_forces(forces)
        forces += self.mass[:, None] * self.gravity[None, :]
        forces -= (self.alpha_drag * self.mass)[:, None] * self.v
        cforce = self._contact(dt, forces)

        self.v += dt * forces * self.inv_mass[:, None]
        np.clip(self.v, -_VEL_CLAMP, _VEL_CLAMP, out=self.v)
        self.x += dt * self.v

        # non-penetration safety projection (keeps the explicit scheme robust)
        self._project(dt)

        self.contact_force = cforce
        self.sdf = self._signed_distance(self.x)
        self._steps += 1

    def _project(self, dt: float) -> None:
        tol = 0.35 * self.contact_length_scale
        x, v = self.x, self.v
        below = x[:, 2] < self.ground_z - tol
        if np.any(below):
            x[below, 2] = self.ground_z - tol
            v[below, 2] = np.maximum(v[below, 2], 0.0)
        rx = x[:, 0]
        rz = x[:, 2] - self.bar_z
        rad = np.sqrt(rx * rx + rz * rz)
        inside = rad < self.bar_r - tol
        if np.any(inside):
            ii = np.nonzero(inside)[0]
            r = np.maximum(rad[ii], 1e-9)
            nx = rx[ii] / r
            nz = rz[ii] / r
            target = self.bar_r - tol
            x[ii, 0] = nx * target
            x[ii, 2] = self.bar_z + nz * target
            vn = v[ii, 0] * nx + v[ii, 2] * nz
            neg = vn < 0.0
            if np.any(neg):
                jj = ii[neg]
                vnn = vn[neg]
                v[jj, 0] -= vnn * nx[neg]
                v[jj, 2] -= vnn * nz[neg]

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.t + 1e-15:
            span = target - self.t
            nsub = max(1, int(np.ceil(span / _DT_MAX)))
            dt = span / nsub
            for _ in range(nsub):
                self._step(dt)
            self.t = target
        else:
            self.sdf = self._signed_distance(self.x)
        return self._frame(target)

    # ------------------------------------------------------------------ export
    def _frame(self, time: float) -> StateFrame:
        F = self._deformation_gradient()
        if not np.isfinite(F).all():
            eye = np.tile(np.eye(3), (self.n, 1, 1))
            bad = ~np.isfinite(F).all(axis=(1, 2))
            F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
            F[bad] = eye[bad]

        # per-vertex accumulated plastic strain (hardening localization)
        idx = np.concatenate([self.sa[: self.n_edge], self.sb[: self.n_edge]])
        wq = np.concatenate([self.q[: self.n_edge], self.q[: self.n_edge]])
        cnt = np.bincount(idx, minlength=self.n).astype(float)
        qv = np.bincount(idx, weights=wq, minlength=self.n) / np.maximum(cnt, 1.0)

        x = np.nan_to_num(self.x, nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
        sdf = np.nan_to_num(np.asarray(self.sdf, dtype=float), nan=0.0,
                            posinf=0.0, neginf=0.0)
        cf = np.nan_to_num(self.contact_force, nan=0.0, posinf=0.0, neginf=0.0)

        frame = StateFrame(
            time=float(max(time, 0.0)),
            position=np.ascontiguousarray(x, dtype=float),
            velocity=np.ascontiguousarray(v, dtype=float),
            mass=self.mass.copy(),
            body_id=np.zeros(self.n, dtype=np.int64),
            attributes={
                "mass_quantity": self.mass.copy(),
                "deformation_gradient": np.ascontiguousarray(F, dtype=float),
                "plastic_strain": np.ascontiguousarray(qv, dtype=float),
            },
            fields={
                "contact_signed_distance": np.ascontiguousarray(sdf, dtype=float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "interaction_force": np.ascontiguousarray(cf, dtype=float),
                "external_force": np.ascontiguousarray(
                    self.mass[:, None] * self.gravity[None, :], dtype=float),
            },
            scalars={"contact_length_scale": float(self.contact_length_scale)},
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        vol = abs(self._enclosed_volume(self.x))
        return {
            "method": "explicit viscoplastic membrane/bulk FEM-spring solid",
            "vertices": int(self.n),
            "springs": int(self.n_spring),
            "sub_steps": int(self._steps),
            "dt_max": float(_DT_MAX),
            "volume_ratio": float(vol / max(self.V0, 1e-12)),
            "max_plastic_strain": float(self.q.max()),
            "mean_plastic_strain": float(self.q.mean()),
            "max_penetration": float(self._max_pen),
            "total_mass": float(self.mass.sum()),
        }
