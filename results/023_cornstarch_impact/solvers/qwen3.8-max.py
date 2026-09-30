"""Physically based solver for scene 023_cornstarch_impact.

Model
-----
The subject is a cornstarch-water suspension blob represented by its surface
sample set (642 material particles).  The continuum is discretised directly on
those particles:

* elasticity  : shape matching (cluster polar decomposition) with a
  *shear thickening* stiffness  k(gdot) = k0 (1 + a sqrt(gdot)), which is the
  observable cornstarch behaviour - rigid/elastic under fast impact loading.
* incompressibility: a volume penalty pressure integrated over the closed
  triangle mesh, p = K (V0 - V)/V0, so the yielded blob keeps its volume and
  spreads into a puddle instead of collapsing.
* viscosity   : shear thickening XSPH-type Laplacian viscosity
  nu(gdot) = nu0 + nu1 sqrt(gdot) (capped), which transmits the impact
  impulse through the body and dissipates the rebound.
* viscoplastic yielding: once the loading stress relaxes (low local strain
  rate) and the configured yield time has elapsed, the material rest shape
  flows towards the current shape at rate ``pl`` (Bargteil-style plasticity).
  The rest shape therefore loses its memory and the blob flattens into a
  puddle; where the elastic strain drops below the yield strain the flow
  stops, which retains the puddle shape.
* ground      : unilateral contact with the plane z = ground_z handled by a
  penalty normal force plus an impulse resolution using the configured
  restitution and Coulomb friction.

Everything is integrated with a fixed-step semi-implicit (symplectic) Euler
scheme with sub-stepping small enough for the stiffest term.  No animation is
scripted: the rebound, the elastic->flowing transition and the final spread
all emerge from the equations above.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


def _smoothstep(u: np.ndarray | float) -> np.ndarray | float:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class BenchmarkSolver:
    """Elasto-viscoplastic (shear thickening) particle/shell solver."""

    # ---------------------------------------------------------------- setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.X0 = np.ascontiguousarray(frame0.position, dtype=float)
        self.N = int(len(self.X0))
        self.m = np.ascontiguousarray(frame0.mass, dtype=float)
        self.m_total = float(self.m.sum())
        self.g = np.asarray(cfg["gravity"], dtype=float).reshape(3)
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.L = float(cfg.get("contact_length_scale", 0.04))

        self.stiffness = float(cfg.get("stiffness", 16.0))
        self.yield_time = max(float(cfg.get("yield_time", 1.65)), 1e-3)
        self.damping = float(cfg.get("damping", 1.1))
        self.restitution = float(np.clip(cfg.get("restitution", 0.54), 0.0, 0.95))
        self.friction = float(cfg.get("friction", 0.68))

        self.faces = self._load_faces(cfg)

        # state
        self.Xr = self.X0.copy()          # rest (plastic) configuration
        self.x = self.X0.copy()           # current configuration
        self.v = np.array(frame0.velocity, dtype=float).copy()
        self.t = 0.0
        self.h = 1.0 / 600.0

        # ------------------------------------------------- neighbourhood
        d = np.sqrt(((self.X0[:, None, :] - self.X0[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        nn = float(d.min(axis=1).mean())
        self.rn = 2.25 * nn
        kmax = 22
        order = np.argsort(d, axis=1)[:, :kmax]
        dsel = np.take_along_axis(d, order, 1)
        self.nei = np.concatenate([np.arange(self.N)[:, None], order], axis=1)
        dd = np.concatenate([np.zeros((self.N, 1)), dsel], axis=1)
        q = np.clip(dd / self.rn, 0.0, 1.0)
        w = np.where(dd < self.rn, (1.0 - q ** 3) ** 3, 0.0)
        w[:, 0] = 1.0
        self.w = np.ascontiguousarray(w)
        self.Wr = self.w.sum(1)
        self.rn2 = self.rn * self.rn

        # rest frame of each cluster (material coordinates) + shell normal
        self.e1, self.e2, self.n0 = self._rest_frames(self.Xr)
        self._build_A_scale()

        # material constants
        self.ks0 = self.stiffness * 25.0
        self.alpha_st = 3.0
        self.gdot_cap = 200.0
        self.soft = 0.88                 # stiffness retained when fully yielded
        self.nu0 = 0.020 * self.damping / 1.1
        self.nu1 = 0.060
        self.nu_max = 0.35
        self.cdrag = 0.25 * self.damping
        self.Kbulk = 4000.0
        self.kc = 2.5e4
        self.v_thr = 0.08
        self.pl_rate = 2.5
        self.e_y0, self.e_y1 = 0.010, 0.060
        self.nu_rest = 0.010
        self.gdot_c = 2.0
        self.v_max = 60.0

        if self.faces is not None:
            self.V0 = self._volume(self.X0)
        else:
            self.V0 = 0.0

        self._diag = {
            "steps": 0,
            "min_volume_ratio": 1.0,
            "max_strain": 0.0,
            "contacts": 0,
        }

    # ------------------------------------------------------------- helpers
    def _load_faces(self, cfg: dict):
        rel = cfg.get("subject_asset", "assets/subject.npz")
        here = Path(__file__).resolve().parent
        cands = [Path(rel), here.parent / rel, here / rel, Path.cwd() / rel]
        for cand in cands:
            try:
                with np.load(cand) as data:
                    if "faces" in data.files:
                        faces = np.asarray(data["faces"], dtype=np.int64).reshape(-1, 3)
                        if self._volume_raw(self.X0, faces) < 0.0:
                            faces = faces[:, [0, 2, 1]]
                        return faces
            except Exception:
                continue
        return None

    @staticmethod
    def _volume_raw(x: np.ndarray, faces: np.ndarray) -> float:
        a = x[faces[:, 0]]
        b = x[faces[:, 1]]
        c = x[faces[:, 2]]
        cr = np.cross(b - a, c - a)
        return float((a * cr).sum() / 6.0)

    def _volume(self, x: np.ndarray) -> float:
        return self._volume_raw(x, self.faces)

    def _build_A_scale(self) -> None:
        xr = self.Xr[self.nei]
        Wr = self.Wr[:, None]
        Xc = np.einsum("nk,nkd->nd", self.w, xr) / Wr
        dX = xr - Xc[:, None, :]
        A = np.einsum("nk,nki,nkj->nij", self.w, dX, dX)
        scale = float(np.trace(A).mean() / 3.0)
        self.eta = max(scale, 1e-12) * 0.05
        self.A0_scale = max(scale, 1e-12)

    def _rest_frames(self, Xr: np.ndarray):
        xr = Xr[self.nei]
        Xc = np.einsum("nk,nkd->nd", self.w, xr) / self.Wr[:, None]
        dX = xr - Xc[:, None, :]
        C = np.einsum("nk,nki,nkj->nij", self.w, dX, dX)
        _, vecs = np.linalg.eigh(C)
        n0 = np.ascontiguousarray(vecs[:, :, 0])
        e1 = np.ascontiguousarray(vecs[:, :, 1])
        e2 = np.cross(n0, e1)
        nrm = np.linalg.norm(e2, axis=1, keepdims=True)
        e2 = np.where(nrm > 1e-12, e2 / np.maximum(nrm, 1e-12), vecs[:, :, 2])
        return e1, e2, n0

    def _vertex_normals(self, x: np.ndarray) -> np.ndarray:
        nrm = np.zeros((self.N, 3))
        if self.faces is None:
            return self.n0.copy()
        a = x[self.faces[:, 0]]
        b = x[self.faces[:, 1]]
        c = x[self.faces[:, 2]]
        cr = 0.5 * np.cross(b - a, c - a)
        for k in range(3):
            np.add.at(nrm, self.faces[:, k], cr)
        ln = np.linalg.norm(nrm, axis=1, keepdims=True)
        return np.where(ln > 1e-12, nrm / np.maximum(ln, 1e-12), self.n0)

    # ------------------------------------------------------------ physics
    def _flow_gate(self, gdot: np.ndarray) -> np.ndarray:
        ramp = _smoothstep(
            (self.t - 0.4 * self.yield_time) / (0.6 * self.yield_time)
        )
        gate = 1.0 / (1.0 + (gdot / self.gdot_c) ** 2)
        return float(ramp) * gate

    def _substep(self, h: float) -> None:
        x, v, Xr = self.x, self.v, self.Xr
        nei, w, Wr = self.nei, self.w, self.Wr
        N = self.N

        a = np.tile(self.g, (N, 1))

        xr = Xr[nei]
        xk = x[nei]
        Xc = np.einsum("nk,nkd->nd", w, xr) / Wr[:, None]
        xc = np.einsum("nk,nkd->nd", w, xk) / Wr[:, None]
        dX = xr - Xc[:, None, :]
        dx = xk - xc[:, None, :]
        A = np.einsum("nk,nki,nkj->nij", w, dx, dX)
        A += (self.eta * np.einsum("ni,nj->nij", self.n0, self.n0))

        U, _, Vt = np.linalg.svd(A)
        det = np.linalg.det(U @ Vt)
        Dm = np.zeros((N, 3, 3))
        Dm[:, 0, 0] = 1.0
        Dm[:, 1, 1] = 1.0
        Dm[:, 2, 2] = np.where(det >= 0.0, 1.0, -1.0)
        R = U @ Dm @ Vt

        goal = xc + np.einsum("nij,nj->ni", R, Xr - Xc)
        disp = goal - x
        strain = np.linalg.norm(disp, axis=1) / self.rn

        vk = v[nei]
        dv = vk - v[:, None, :]
        gdot = np.sqrt(np.einsum("nk,nkd->n", w, dv * dv) / Wr) / self.rn

        flow = self._flow_gate(gdot)
        g_st = np.sqrt(np.minimum(gdot, self.gdot_cap))
        ks = self.ks0 * (1.0 + self.alpha_st * g_st) * (1.0 - self.soft * flow)
        ks = np.minimum(ks, 40.0 * self.ks0)
        a += ks[:, None] * disp

        nu = np.minimum(self.nu0 + self.nu1 * g_st, self.nu_max)
        lap_v = np.einsum("nk,nkd->nd", w, dv) / Wr[:, None] / self.rn2
        a += nu[:, None] * lap_v

        vcom = (self.m[:, None] * v).sum(0) / self.m_total
        a -= self.cdrag * (v - vcom)

        if self.faces is not None:
            vol = self._volume(x)
            ratio = vol / self.V0 if self.V0 > 0 else 1.0
            self._diag["min_volume_ratio"] = min(self._diag["min_volume_ratio"], ratio)
            p = self.Kbulk * (1.0 - ratio)
            p = float(np.clip(p, -0.2 * self.Kbulk, 8.0 * self.Kbulk))
            fa = x[self.faces[:, 0]]
            fb = x[self.faces[:, 1]]
            fc = x[self.faces[:, 2]]
            cr = (p / 6.0) * np.cross(fb - fa, fc - fa)
            F = np.zeros((N, 3))
            for k in range(3):
                np.add.at(F, self.faces[:, k], cr)
            a += F / self.m[:, None]

        pen = np.maximum(self.ground_z - x[:, 2], 0.0)
        a[:, 2] += self.kc * pen

        v += a * h
        speed = np.linalg.norm(v, axis=1)
        bad = speed > self.v_max
        if bad.any():
            v[bad] *= (self.v_max / speed[bad])[:, None]
        x += v * h

        # unilateral ground contact resolution
        below = x[:, 2] < self.ground_z
        if below.any():
            vz = v[:, 2]
            hit = below & (vz < 0.0)
            strong = hit & (-vz > self.v_thr)
            if strong.any():
                v[strong, 2] = -self.restitution * vz[strong]
            weak = hit & ~strong
            if weak.any():
                v[weak, 2] = 0.0
            load = self.kc * pen + np.linalg.norm(self.g)
            dv_t = self.friction * load * h
            vt = v[:, :2]
            sp = np.linalg.norm(vt, axis=1)
            scale = np.clip(1.0 - dv_t / np.maximum(sp, 1e-9), 0.0, 1.0)
            idx = np.nonzero(below)[0]
            v[idx, 0] = vt[idx, 0] * scale[idx]
            v[idx, 1] = vt[idx, 1] * scale[idx]
            deep = below & (x[:, 2] < self.ground_z - 2.0e-3)
            x[deep, 2] = self.ground_z - 2.0e-3
            self._diag["contacts"] = int(below.sum())

        # viscoplastic flow of the rest configuration
        mask = _smoothstep((strain - self.e_y0) / max(self.e_y1 - self.e_y0, 1e-6))
        coef = np.clip(self.pl_rate * h * flow * mask, 0.0, 0.5)
        self._diag["max_strain"] = max(self._diag["max_strain"], float(strain.max()))
        if coef.max() > 0.0:
            Xr_new = Xr + coef[:, None] * (x - Xr)
            xr2 = Xr_new[nei]
            lap = np.einsum("nk,nkd->nd", w, xr2 - Xr_new[:, None, :]) / Wr[:, None]
            Xr_new += np.clip(self.nu_rest * h / self.rn2, 0.0, 0.2) * flow[:, None] * lap
            if np.isfinite(Xr_new).all():
                Xr[:] = Xr_new

        self._diag["steps"] += 1

    # ------------------------------------------------- deformation gradient
    def _deformation_gradient(self) -> np.ndarray:
        xr = self.Xr[self.nei]
        xk = self.x[self.nei]
        Xc = np.einsum("nk,nkd->nd", self.w, xr) / self.Wr[:, None]
        xc = np.einsum("nk,nkd->nd", self.w, xk) / self.Wr[:, None]
        dX = xr - Xc[:, None, :]
        dx = xk - xc[:, None, :]
        e1, e2, n0 = self._rest_frames(self.Xr)
        u = np.stack(
            (np.einsum("nkd,nd->nk", dX, e1), np.einsum("nkd,nd->nk", dX, e2)), axis=2
        )
        qq = np.stack(
            (
                np.einsum("nkd,nd->nk", dx, e1),
                np.einsum("nkd,nd->nk", dx, e2),
                np.einsum("nkd,nd->nk", dx, n0),
            ),
            axis=2,
        )
        G = np.einsum("nk,nkp,nkq->npq", self.w, qq, u)
        H = np.einsum("nk,nkp,kq->npq", self.w, u, u)
        H = H + 1e-9 * np.eye(2)[None, :, :] * max(self.A0_scale, 1.0)
        Fin = np.linalg.solve(np.swapaxes(H, 1, 2), np.swapaxes(G, 1, 2))
        Fin = np.swapaxes(Fin, 1, 2)                      # [N,3,2]
        area = np.linalg.norm(np.cross(Fin[:, :, 0], Fin[:, :, 1]), axis=1)
        sigma = np.clip(area, 0.05, 20.0)
        s = 1.0 / sigma
        ncur = self._vertex_normals(self.x)
        n_rest = np.stack(
            (
                np.einsum("nd,nd->n", ncur, e1),
                np.einsum("nd,nd->n", ncur, e2),
                np.einsum("nd,nd->n", ncur, n0),
            ),
            axis=1,
        )
        third = s[:, None] * n_rest
        F = np.concatenate([Fin, third[:, :, None]], axis=2)
        return np.where(np.isfinite(F), F, np.tile(np.eye(3), (self.N, 1, 1)))

    # --------------------------------------------------------------- frame
    def _frame(self) -> StateFrame:
        gap = self.x[:, 2] - self.ground_z
        band = 3.0 * self.L
        near = np.nonzero(gap < band)[0]
        if near.size < 8:
            near = np.argsort(gap)[: max(8, near.size)]
        contact = np.ascontiguousarray(gap[near])

        frame = StateFrame(
            time=float(self.t),
            position=np.array(self.x, dtype=float),
            velocity=np.array(self.v, dtype=float),
            mass=np.array(self.m, dtype=float),
            body_id=np.zeros(self.N, dtype=np.int64),
            attributes={
                "mass_quantity": np.array(self.m, dtype=float),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.m[:, None] * self.g[None, :],
                "contact_signed_distance": contact,
            },
            scalars={"contact_length_scale": float(self.L)},
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- driving
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.t:
            target = self.t
        span = target - self.t
        if span > 0.0:
            steps = max(1, int(np.ceil(span / self.h)))
            h = span / steps
            for _ in range(steps):
                self._substep(h)
                self.t += h
            self.t = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        info = dict(self._diag)
        info.update(
            {
                "time": float(self.t),
                "center_of_mass_z": float((self.m[:, None] * self.x).sum(0)[2] / self.m_total),
                "kinetic_energy": float(0.5 * (self.m * (self.v ** 2).sum(1)).sum()),
                "spread_radius": float(
                    np.sqrt(
                        ((self.x[:, :2] - (self.m[:, None] * self.x[:, :2]).sum(0) / self.m_total) ** 2)
                        .sum(1)
                        .mean()
                    )
                ),
            }
        )
        return info
