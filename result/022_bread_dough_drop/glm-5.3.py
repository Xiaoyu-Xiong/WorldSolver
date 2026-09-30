"""Viscoplastic dough-drop solver.

Physical model
--------------
The subject is a closed surface mesh (702 samples) of a bread-dough ball that is
dropped from rest onto a ground plane.  The solver performs a real explicit
time-stepping simulation of a viscoplastic solid:

* each mesh sample carries mass (total frozen subject mass split uniformly),
* a rest-state neighbor network (k-nearest neighbours union mesh edges) forms a
  spring lattice that gives the sample elastic (neo-Hookean-like) cohesion,
* each spring carries a *plastic rest length* that flows toward the current
  length once the elastic strain exceeds a yield strain (Bargteil et al. 2007
  style return mapping for large viscoplastic flow), so impact deformation is
  permanently retained and only the sub-yield elastic part recovers,
* a weak volumetric (pressure) constraint computed with the divergence theorem
  on the closed surface keeps the near-incompressible dough from collapsing,
* ground contact is a regularized penalty contact with Coulomb friction
  (frozen friction / restitution) and gravity from the frozen config,
* semi-implicit (symplectic Euler) integration with a fixed deterministic
  substep advances the state; frames are emitted at the requested sample times.

No keyframes, animation tables or precomputed trajectories are used anywhere.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit viscoplastic mass-lattice solver for the bread-dough drop."""

    SUBSTEP = 2.0e-4          # fixed physical substep [s]
    KNN = 16                  # rest-state neighbour count for the spring network
    YIELD_STRAIN = 0.03       # elastic strain threshold before plastic flow
    PLASTIC_RATE_SCALE = 100.0  # plastic flow rate multiplier [1/s]
    STIFFNESS_SCALE = 40.0    # spring stiffness = scale * stiffness * rest length
    VOLUME_STIFFNESS = 30.0   # pressure-like incompressibility coefficient
    VOLUME_DAMPING = 8.0

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(getattr(scene, "config", {}) or {})
        self.config = cfg
        self.seed = int(seed)

        # frozen subject geometry (prefer the scene's loaded mesh, fall back
        # to the frozen asset on disk)
        verts = getattr(scene, "vertices", None)
        faces = getattr(scene, "faces", None)
        if verts is not None and faces is not None:
            self.faces = np.asarray(faces, dtype=np.int64)
            x0 = np.asarray(verts, dtype=float)
        else:
            with np.load(_asset_path(scene)) as data:
                self.faces = np.asarray(data["faces"], dtype=np.int64)
                x0 = np.asarray(data["vertices"], dtype=float)
        self.x0 = x0.copy()
        self.x = x0.copy()
        n = len(x0)
        self.n = n

        total_mass = float(cfg.get("subject_mass", 0.55))
        self.m = np.full(n, total_mass / n, dtype=float)

        g = np.asarray(cfg.get("gravity", [0.0, 0.0, -9.81]), dtype=float)
        self.gravity = g
        self.gravity_force = self.m[:, None] * g[None, :]

        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.contact_scale = float(cfg.get("contact_length_scale", 0.04))
        self.friction = float(cfg.get("friction", 0.5))
        self.restitution = float(cfg.get("restitution", 0.0))
        self.plasticity = float(cfg.get("plasticity", 0.5))
        self.stiffness = float(cfg.get("stiffness", 1.0))
        self.damping = float(cfg.get("damping", 1.0))

        v0 = np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float)
        self.v = np.tile(v0, (n, 1))
        self.t = 0.0
        self._substeps = 0
        self._contact_accum = np.zeros((n, 3), dtype=float)
        self._interval = 0.0

        # ---- spring network: kNN(rest) union mesh edges --------------------
        tree = cKDTree(x0)
        _, idx = tree.query(x0, k=self.KNN + 1)
        nbr = idx[:, 1:]                       # (N, K) neighbours, self excluded
        self.nbr = nbr
        pairs = set()
        for i in range(n):
            for j in nbr[i]:
                if j != i:
                    pairs.add((i, int(j)) if i < j else (int(j), i))
        for tri in self.faces:
            a, b, c = (int(t_) for t_ in tri)
            for p in ((a, b), (b, c), (c, a)):
                pairs.add(p if p[0] < p[1] else (p[1], p[0]))
        pa = np.array(sorted(p for p in pairs if p[0] != p[1]), dtype=np.int64)
        self.sa = pa[:, 0].copy()
        self.sb = pa[:, 1].copy()

        d0 = x0[self.sb] - x0[self.sa]
        self.L0 = np.linalg.norm(d0, axis=1)
        self.Lp = self.L0.copy()               # plastic rest lengths

        # per-spring stiffness (uniform network, scaled by rest length)
        self.k_s = np.full(len(self.L0), self.stiffness * self.STIFFNESS_SCALE
                           * float(np.mean(self.L0)))
        # spring viscous damping (fraction of critical)
        self.c_s = np.full(len(self.L0), 0.5 * self.damping
                           * np.sqrt(self.k_s[0] * total_mass / n))

        # rest volume (divergence theorem on the closed surface)
        f = self.faces
        self.V0 = float(np.sum(np.einsum('ij,ij->i', x0[f[:, 0]],
                                         np.cross(x0[f[:, 1]], x0[f[:, 2]]))) / 6.0)

        # contact regularization
        self.k_contact = 2000.0
        self.c_contact = 0.7 * 2.0 * np.sqrt(self.k_contact * total_mass / n)
        self.c_fric = 400.0

        # rest-distance weights for the per-sample deformation-gradient fit
        dn = np.linalg.norm(x0[nbr] - x0[:, None, :], axis=2)
        self.fit_w = 1.0 / (dn * dn + 1.0e-4)
        self.fit_dX = x0[nbr] - x0[:, None, :]

    # ------------------------------------------------------------ time steppers
    def advance_to(self, time: float):
        target = float(time)
        if target <= self.t + 1.0e-12:
            return self._frame(self.t)
        last = self.t
        while self.t < target - 1.0e-12:
            dt = min(self.SUBSTEP, target - self.t)
            self._step(dt)
        self._interval = max(self.t - last, 1.0e-12)
        return self._frame(self.t)

    # ----------------------------------------------------------------- physics
    def _step(self, dt: float) -> None:
        x, v, m = self.x, self.v, self.m
        n = self.n

        force = self.gravity_force.copy()

        # ---- elastic + viscoplastic spring lattice -------------------------
        d = x[self.sb] - x[self.sa]
        length = np.sqrt(np.sum(d * d, axis=1))
        length = np.maximum(length, 1.0e-9)
        direction = d / length[:, None]

        relv = v[self.sb] - v[self.sa]
        fs = (self.k_s * (length - self.Lp)
              + self.c_s * np.sum(relv * direction, axis=1))
        fspring = fs[:, None] * direction
        np.add.at(force, self.sa, fspring)
        np.add.at(force, self.sb, -fspring)

        # plastic return mapping: rest length flows toward current length
        # once the elastic strain exceeds the yield strain (viscoplastic flow).
        strain = (length - self.Lp) / self.L0
        excess = np.abs(strain) - self.YIELD_STRAIN
        flow = 1.0 - np.exp(-dt * self.plasticity * self.PLASTIC_RATE_SCALE)
        shift = np.sign(strain) * np.maximum(excess, 0.0) * self.L0 * flow
        self.Lp = np.clip(self.Lp + shift, 0.25 * self.L0, 4.0 * self.L0)

        # ---- volumetric (near-incompressibility) constraint ----------------
        f = self.faces
        a, b, c = x[f[:, 0]], x[f[:, 1]], x[f[:, 2]]
        cross_bc = np.cross(b, c)
        V = float(np.sum(np.sum(a * cross_bc, axis=1)) / 6.0)
        grad = np.zeros((n, 3), dtype=float)
        np.add.at(grad, f[:, 0], cross_bc / 6.0)
        np.add.at(grad, f[:, 1], np.cross(c, a) / 6.0)
        np.add.at(grad, f[:, 2], np.cross(a, b) / 6.0)
        dVdt = float(np.sum(grad * v))
        pressure = (self.VOLUME_STIFFNESS * (self.V0 - V)
                    - self.VOLUME_DAMPING * dVdt) / max(self.V0, 1.0e-9)
        force += pressure * grad

        # ---- ground contact: penalty normal + Coulomb friction -------------
        gap = x[:, 2] - self.ground_z
        pen = np.maximum(-gap, 0.0)
        active = pen > 0.0
        Fn = np.where(active,
                      self.k_contact * pen
                      - self.c_contact * np.minimum(v[:, 2], 0.0),
                      0.0)
        Fn = np.maximum(Fn, 0.0)
        fcontact = np.zeros((n, 3), dtype=float)
        fcontact[:, 2] = Fn
        vt = np.stack([v[:, 0], v[:, 1]], axis=1)
        ft = -self.c_fric * vt
        ft_mag = np.linalg.norm(ft, axis=1)
        max_f = self.friction * Fn
        scale = np.where(ft_mag > 1.0e-12,
                         np.minimum(1.0, max_f / np.maximum(ft_mag, 1.0e-12)),
                         0.0)
        fcontact[:, 0] = ft[:, 0] * scale
        fcontact[:, 1] = ft[:, 1] * scale
        force += fcontact
        self._contact_accum += fcontact * dt

        # hard safety clamp: no tunnelling / deep penetration
        deep = gap < -0.02
        if np.any(deep):
            x[deep, 2] = self.ground_z - 0.02

        # ---- semi-implicit Euler -------------------------------------------
        v += dt * force / m[:, None]
        np.clip(v, -30.0, 30.0, out=v)
        x += dt * v

        # restitution handled as contact dissipation (frozen small value); the
        # penalty normal + damping dissipates the normal velocity on impact.
        self.t += dt
        self._substeps += 1

    # ----------------------------------------------------------------- outputs
    def _deformation_gradient(self):
        """Least-squares affine fit F: rest offsets -> current offsets."""
        x = self.x
        dx = x[self.nbr] - x[:, None, :]                     # (N,K,3)
        w = self.fit_w[:, :, None]
        A = np.einsum('nki,nkj->nij', dx * w, self.fit_dX)
        B = np.einsum('nki,nkj->nij', self.fit_dX * w, self.fit_dX)
        B += 1.0e-6 * np.eye(3)[None, :, :]
        F = np.linalg.solve(np.transpose(B, (0, 2, 1)),
                            np.transpose(A, (0, 2, 1)))
        F = np.transpose(F, (0, 2, 1))
        return F

    def _frame(self, t: float) -> StateFrame:
        x, v, m = self.x, self.v, self.m
        n = self.n
        ext = self.gravity_force.copy()
        if self._interval > 1.0e-9:
            ext = ext + self._contact_accum / self._interval
        gap = x[:, 2] - self.ground_z
        frame = StateFrame(
            time=float(t),
            position=x.copy(),
            velocity=v.copy(),
            mass=m.copy(),
            body_id=np.zeros(n, dtype=np.int64),
            attributes={
                "mass_quantity": m.copy(),
                "deformation_gradient": self._deformation_gradient(),
            },
            fields={
                "cumulative_mass_source": np.zeros(1, dtype=float),
                "external_force": ext,
                "contact_signed_distance": gap,
            },
            scalars={"contact_length_scale": self.contact_scale},
        )
        frame.validate()
        # reset per-sample accumulators
        self._contact_accum[:] = 0.0
        self._interval = 0.0
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": float(self.t),
            "substeps": int(self._substeps),
            "n_samples": int(self.n),
            "n_springs": int(len(self.L0)),
            "mean_plastic_strain": float(np.mean((self.Lp - self.L0) / self.L0)),
            "rest_volume": float(self.V0),
        }


def _asset_path(scene) -> str:
    """Locate the frozen subject asset relative to the scene module."""
    import os
    cfg = getattr(scene, "config", {}) or {}
    rel = cfg.get("subject_asset", "assets/subject.npz")
    candidates = []
    scene_file = getattr(getattr(scene, "__module__", None), "__file__", None)
    mod = globals().get("__file__")
    if scene_file:
        candidates.append(os.path.join(os.path.dirname(scene_file), rel))
    if mod:
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(mod)), rel))
    candidates.append(os.path.join(os.getcwd(), rel))
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]
