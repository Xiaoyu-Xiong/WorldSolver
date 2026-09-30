"""Viscoplastic shell solver for the cornstarch impact scene.

Physics model (self-contained, no genesis):

The subject is a closed triangular shell mesh.  Its vertices are Lagrangian
mass samples.  The discrete model combines

* edge springs along the mesh edges (elastic, viscous) whose rest lengths
  flow plastically with a Bingham-type law whose yield stress is rate
  stiffened (shear thickening: fast impacts stay elastic, slow loading
  flows),
* an internal normal-pressure law derived from the signed enclosed volume
  (keeps the blob coherent and drives puddle spreading once the shell has
  yielded),
* penalty ground contact with normal restitution damping derived from the
  frozen ``restitution`` and a Coulomb friction impulse clamp derived from
  the frozen ``friction``,
* gravity from the frozen config.

Time integration is semi-implicit (symplectic) Euler with a fixed substep
h = 1/2100 s (70 substeps per sample interval of 1/30 s), which makes the
whole run fully deterministic.

The yield stress schedule decays after the frozen ``yield_time`` (the
material leaves the shear-thickened/solid regime and starts to flow), which
reproduces the observed cornstarch behaviour: elastic bounce at high strain
rate, then viscoplastic collapse into a puddle.
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


def _bincount3(idx: np.ndarray, w: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, 3))
    for c in range(3):
        out[:, c] = np.bincount(idx, w[:, c], minlength=n)
    return out


class BenchmarkSolver:
    HZ = 2100.0  # substep rate; 70 substeps per sample (sample_dt = 1/30)

    # ---- initialization -------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.x = np.array(frame0.position, dtype=float).copy()
        self.v = np.array(frame0.velocity, dtype=float).copy()
        self.mass = np.array(frame0.mass, dtype=float).copy()
        self.n = int(self.x.shape[0])
        self.body_id = np.zeros(self.n, dtype=np.int64)
        self.step = 0

        self.gravity = np.asarray(cfg["gravity"], dtype=float)
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.scale = float(cfg["contact_length_scale"])
        self.restitution = float(cfg.get("restitution", 0.5))
        self.friction = float(cfg.get("friction", 0.5))
        self.damping = float(cfg.get("damping", 1.0))
        self.yield_time = float(cfg.get("yield_time", 1.6))
        stiff_kpa = float(cfg.get("stiffness", 16.0))
        subject_mass = float(cfg.get("subject_mass", float(self.mass.sum())))

        # --- mesh topology (frozen asset) ---
        asset = None
        try:
            asset = Path(inspect.getfile(type(scene))).parent / cfg["subject_asset"]
        except (TypeError, OSError):
            asset = None
        if asset is None or not asset.exists():
            asset = Path(cfg["subject_asset"])
        with np.load(asset) as data:
            faces = np.asarray(data["faces"], dtype=np.int64)

        def _volume(x: np.ndarray, f: np.ndarray) -> float:
            a = x[f[:, 0]]
            b = x[f[:, 1]]
            c = x[f[:, 2]]
            return float(np.sum((a * np.cross(b, c)).sum(axis=1)) / 6.0)

        if _volume(self.x, faces) < 0.0:
            faces = faces[:, ::-1].copy()
        self.faces = faces
        self.fa = faces[:, 0].copy()
        self.fb = faces[:, 1].copy()
        self.fc = faces[:, 2].copy()
        self.V0 = max(_volume(self.x, faces), 1e-9)

        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = np.sort(e, axis=1)
        self.edges = np.unique(e, axis=0)
        self.ea = self.edges[:, 0].copy()
        self.eb = self.edges[:, 1].copy()
        self.L0_init = np.linalg.norm(self.x[self.ea] - self.x[self.eb], axis=1)
        self.L0 = self.L0_init.copy()

        # per-vertex rest areas and rest normals
        nf0 = np.cross(self.x[self.fb] - self.x[self.fa], self.x[self.fc] - self.x[self.fa])
        area_f = 0.5 * np.linalg.norm(nf0, axis=1)
        self.area_v = np.zeros(self.n)
        for idx in (self.fa, self.fb, self.fc):
            self.area_v += np.bincount(idx, area_f / 3.0, minlength=self.n)
        n0 = np.zeros((self.n, 3))
        for idx in (self.fa, self.fb, self.fc):
            n0 += _bincount3(idx, nf0, self.n)
        self.n0 = n0 / np.maximum(np.linalg.norm(n0, axis=1), 1e-12)[:, None]
        self.w_n = float(np.mean(self.L0_init ** 2))

        # directed edges + per-vertex rest structure tensor for the
        # least-squares (best-fit) deformation gradient
        self.src = np.concatenate([self.ea, self.eb])
        self.dst = np.concatenate([self.eb, self.ea])
        self.rdir = self.x[self.dst] - self.x[self.src]
        A = np.zeros((self.n, 3, 3))
        for c in range(3):
            for d in range(3):
                A[:, c, d] = np.bincount(self.src, self.rdir[:, c] * self.rdir[:, d],
                                         minlength=self.n)
        A += self.w_n * np.einsum("mc,md->mcd", self.n0, self.n0)
        A += (1e-4 * self.w_n) * np.eye(3)[None, :, :]
        self.Ainv = np.linalg.inv(A)

        # --- material parameters (all derived from frozen config) ---
        # elastic modulus: stiffness [kPa] acting over the shell thickness
        # frozen in contact_length_scale -> edge spring constant [N/m]
        self.k_edge = stiff_kpa * 1000.0 * self.scale
        m_red = 0.5 * float(np.min(self.mass))
        self.c_edge = self.damping * 0.08 * math.sqrt(self.k_edge * m_red)

        # ground contact penalty + restitution damping
        self.k_contact = 500.0
        m_eff = max(subject_mass / 32.0, 1e-6)
        ln_e = -math.log(max(min(self.restitution, 0.999), 0.05))
        zeta = ln_e / math.sqrt(math.pi ** 2 + ln_e ** 2)
        self.c_contact = 2.0 * zeta * math.sqrt(self.k_contact * m_eff)

        # internal pressure (cohesion / volume cohesion of the suspension)
        self.k_pressure = 150.0  # Pa
        self.p_clamp = 400.0

        # viscoplastic yield: shear-thickening Bingham rest-length flow
        self.sig0 = 0.6           # N, yield force per spring in the solid regime
        self.sr_ref = 0.05        # m/s, rate stiffening reference
        self.eta_flow = 0.5       # N s/m, plastic viscosity
        self.flow_cap = 0.4       # m/s, max plastic flow speed
        self.resid = 0.12         # residual yield fraction after yielding
        self.tau_y = 0.35         # s, yield-stress decay time after yield_time

        # ambient / viscous drag (grows once the material flows)
        self.drag0 = 0.3
        self.drag1 = 3.0

    # ---- helpers ---------------------------------------------------------
    def _schedule(self, t: float) -> float:
        """Yield-stress schedule: solid before yield_time, then decays."""
        if t < self.yield_time:
            return 1.0
        return max(self.resid, math.exp(-(t - self.yield_time) / self.tau_y))

    def _vertex_normals(self) -> np.ndarray:
        x = self.x
        nf = np.cross(x[self.fb] - x[self.fa], x[self.fc] - x[self.fa])
        nrm = np.zeros((self.n, 3))
        for idx in (self.fa, self.fb, self.fc):
            nrm += _bincount3(idx, nf, self.n)
        return nrm / np.maximum(np.linalg.norm(nrm, axis=1), 1e-12)[:, None]

    def _volume(self) -> float:
        x = self.x
        a = x[self.fa]
        b = x[self.fb]
        c = x[self.fc]
        return float(np.sum((a * np.cross(b, c)).sum(axis=1)) / 6.0)

    # ---- time stepping ----------------------------------------------------
    def _step(self, h: float, t: float) -> None:
        x, v = self.x, self.v

        # elastic + viscous edge springs
        d = x[self.ea] - x[self.eb]
        L = np.maximum(np.sqrt(np.einsum("ij,ij->i", d, d)), 1e-9)
        u = d / L[:, None]
        dv = v[self.ea] - v[self.eb]
        vr = np.einsum("ij,ij->i", dv, u)
        fs = self.k_edge * (L - self.L0) + self.c_edge * vr
        fv = fs[:, None] * u
        f = np.zeros((self.n, 3))
        for c in range(3):
            f[:, c] = (np.bincount(self.ea, fv[:, c], minlength=self.n)
                       - np.bincount(self.eb, fv[:, c], minlength=self.n))

        # internal pressure from the signed enclosed volume
        vol = self._volume()
        p = self.k_pressure * (self.V0 - vol) / self.V0
        p = min(max(p, -self.p_clamp), self.p_clamp)
        nrm = self._vertex_normals()
        f += p * self.area_v[:, None] * nrm

        # ground contact (penalty normal force + restitution damping)
        gap = x[:, 2] - self.ground_z
        pen = gap < 0.0
        if np.any(pen):
            f[pen, 2] += -self.k_contact * gap[pen] - self.c_contact * v[pen, 2]

        sched = self._schedule(t)

        # semi-implicit Euler with viscous drag
        acc = f / self.mass[:, None] + self.gravity[None, :]
        drag = self.drag0 + self.drag1 * (1.0 - sched)
        v += acc * h
        v *= max(0.0, 1.0 - drag * h)
        x += v * h

        # Coulomb friction impulse clamp for contacting samples
        gap2 = x[:, 2] - self.ground_z
        pen2 = gap2 < 0.0
        if np.any(pen2):
            fn = -self.k_contact * gap2[pen2]
            mu_eff = self.friction * min(1.0, sched)
            dvmax = mu_eff * fn * h / self.mass[pen2]
            vt = np.hypot(v[pen2, 0], v[pen2, 1])
            sc = np.maximum(0.0, 1.0 - dvmax / np.maximum(vt, 1e-12))
            v[pen2, 0] *= sc
            v[pen2, 1] *= sc

        # safety: forbid deep tunneling through the ground
        zmin = self.ground_z - 0.12
        deep = x[:, 2] < zmin
        if np.any(deep):
            x[deep, 2] = zmin
            v[deep, 2] = np.maximum(v[deep, 2], 0.0)

        # shear-thickening Bingham plastic flow of the spring rest lengths
        s = self.k_edge * (L - self.L0)
        sig = self.sig0 * sched * (1.0 + np.abs(vr) / self.sr_ref)
        flow = np.abs(s) - sig
        np.maximum(flow, 0.0, out=flow)
        flow /= self.eta_flow
        np.minimum(flow, self.flow_cap, out=flow)
        self.L0 += flow * np.sign(s) * h
        np.clip(self.L0, 0.25 * self.L0_init, 2.5 * self.L0_init, out=self.L0)

    # ---- StateFrame export -------------------------------------------------
    def _deformation_gradient(self) -> np.ndarray:
        q = self.x[self.dst] - self.x[self.src]
        outer = np.einsum("mc,md->mcd", q, self.rdir)
        B = np.zeros((self.n, 3, 3))
        for c in range(3):
            for d in range(3):
                B[:, c, d] = np.bincount(self.src, outer[:, c, d], minlength=self.n)
        B += self.w_n * np.einsum("mc,md->mcd", self._vertex_normals(), self.n0)
        return np.matmul(B, self.Ainv)

    def _frame(self, time: float) -> StateFrame:
        F = self._deformation_gradient()
        gap = self.x[:, 2] - self.ground_z
        near = np.nonzero(gap < 3.0 * self.scale)[0]
        if near.size == 0:
            near = np.array([int(np.argmin(gap))], dtype=np.int64)
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "deformation_gradient": F,
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "contact_signed_distance": gap[near].astype(float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": self.mass[:, None] * self.gravity[None, :],
            },
            scalars={"contact_length_scale": self.scale},
        )
        frame.validate()
        return frame

    # ---- public API ---------------------------------------------------------
    def advance_to(self, time: float) -> StateFrame:
        h = 1.0 / self.HZ
        target = int(round(float(time) * self.HZ))
        while self.step < target:
            self._step(h, self.step * h)
            self.step += 1
        return self._frame(float(time))

    def diagnostics(self) -> dict:
        t = self.step / self.HZ
        return {
            "time": t,
            "substeps": int(self.step),
            "yield_schedule": self._schedule(t),
            "volume": self._volume(),
            "min_gap": float(np.min(self.x[:, 2] - self.ground_z)),
            "max_speed": float(np.max(np.linalg.norm(self.v, axis=1))),
        }
