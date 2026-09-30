"""Ductile hollow ball drop solver (scene 128).

Physical model
--------------
The hollow ductile shell is represented by the frozen mesh sampling: the
samples are the mesh vertices, connected by the mesh edges.  The shell is
integrated with a substepped XPBD (extended position-based dynamics) scheme,
which is an implicit, unconditionally stable time integrator:

  * every substep applies gravity, predicts positions, then projects the
    distance constraints of all alive shell edges plus the rigid floor,
    and finally recomputes velocities from the positional change;
  * each edge is an elastic-plastic bar: a radial return mapping with
    isotropic hardening moves its plastic rest length permanently once the
    axial strain exceeds the yield strain (permanent plastic set);
  * tensile plastic flow accumulates damage; damaged edges are softened
    (necking localisation) and fail once the accumulated tensile plastic
    strain passes a ductile fracture threshold -- the edge is removed and a
    tear opens.  Compressive plastic flow only buckles/crushes;
  * floor contact is inelastic (normal motion absorbed by projection) with
    Coulomb-like tangential friction, which crushes and flattens the shell.

No keyframes or scripted motion are used: all motion follows from the
time-stepping of the discrete Lagrangian shell under gravity.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

# Frozen scene geometry (scene_geometry.yaml): rigid ground box.
GROUND_CENTER = np.array([0.0, 0.0, -0.05])
GROUND_HALF = np.array([3.2, 1.4, 0.05])
FLOOR_Z = 0.0  # top surface of the ground box


class BenchmarkSolver:
    """XPBD elastic-plastic fracturing shell solver."""

    def __init__(self) -> None:
        self._ready = False

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        frame0 = scene.initial_state()

        self.x0 = np.array(frame0.position, dtype=float).copy()
        self.n = len(self.x0)
        self.mass = np.array(frame0.mass, dtype=float).copy()
        self.w = 1.0 / self.mass
        self.body_id = np.array(frame0.body_id, dtype=np.int32).copy()
        self.attr0 = dict(frame0.attributes)
        self.scalars0 = dict(frame0.scalars)

        self.g = np.asarray(cfg["gravity"], dtype=float)
        sampling = cfg["sampling"]
        self.spacing = float(sampling["spacing"])
        self.particle_radius = 0.45 * self.spacing

        # integrator
        self.dt_sub = 1.0 / 1920.0
        self.n_passes = 2
        self.relax = 0.5
        self.drag = 0.05          # mild aerodynamic drag (1/s)
        self.friction_rate = 6.0  # tangential velocity decay while in contact
        self.v_max = 60.0

        # ductile material parameters (edge bars)
        self.k0 = 2.0e6           # edge axial stiffness [N/m]
        self.eps_y0 = 0.02        # initial yield strain
        self.hardening = 0.15     # isotropic hardening modulus
        self.eps_fail = 0.55      # tensile plastic strain at ductile failure
        self.softening = 8.0      # necking stiffness reduction factor

        # shell connectivity from frozen topology asset
        topo = self._load_topology(scene)
        faces = np.asarray(topo["faces"], dtype=np.int64)
        e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = np.sort(e, axis=1)
        e = np.unique(e, axis=0)
        d0 = self.x0[e[:, 0]] - self.x0[e[:, 1]]
        l0 = np.linalg.norm(d0, axis=1)
        keep = l0 > 1e-6
        self.ei = e[keep, 0].copy()
        self.ej = e[keep, 1].copy()
        n_edges = len(self.ei)
        self.l0 = l0[keep].copy()             # original rest length
        self.lp = self.l0.copy()              # plastic rest length
        self.eps_acc = np.zeros(n_edges)      # accumulated tensile plastic strain
        self.n_edges0 = n_edges

        # deterministic per-edge fracture scatter (localisation seed)
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        self.eps_fail_e = self.eps_fail * rng.uniform(0.92, 1.08, n_edges)

        # state
        self.x = self.x0.copy()
        self.v = np.zeros((self.n, 3))
        self.t = 0.0
        self.first_fracture_time = None
        self.external_force = self.mass[:, None] * self.g[None, :]

        self._ready = True

    @staticmethod
    def _load_topology(scene) -> dict:
        candidates = []
        try:
            candidates.append(Path(inspect.getfile(type(scene))).resolve().parent)
        except (TypeError, OSError):
            pass
        candidates.append(Path.cwd())
        for root in candidates:
            p = root / "assets" / "hollow_ball_topology.json"
            if p.is_file():
                return json.loads(p.read_text(encoding="utf-8"))
        raise FileNotFoundError("assets/hollow_ball_topology.json not found")

    # -------------------------------------------------------------- dynamics
    def _solve_edges(self, h: float) -> None:
        if len(self.ei) == 0:
            return
        x, ei, ej, w = self.x, self.ei, self.ej, self.w
        d = x[ej] - x[ei]
        length = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
        n = d / length[:, None]

        # --- elastic-plastic return mapping with isotropic hardening -------
        eps = (length - self.lp) / self.lp
        ey = self.eps_y0 + self.hardening * self.eps_acc
        over = np.abs(eps) - ey
        flow = over > 0.0
        if np.any(flow):
            tensile = eps > 0.0
            dlp = np.sign(eps) * over * self.lp
            self.lp = self.lp + np.where(flow, dlp, 0.0)
            # damage accumulates only from tensile plastic flow (ductility)
            self.eps_acc = np.where(
                flow & tensile, self.eps_acc + over, self.eps_acc
            )

        # --- ductile fracture: edge death after substantial plastic strain --
        alive = self.eps_acc < self.eps_fail_e
        if not np.all(alive):
            if self.first_fracture_time is None:
                self.first_fracture_time = self.t
            self.ei = ei[alive]
            self.ej = ej[alive]
            self.lp = self.lp[alive]
            self.eps_acc = self.eps_acc[alive]
            self.eps_fail_e = self.eps_fail_e[alive]
            if len(self.ei) == 0:
                return
            ei, ej = self.ei, self.ej
            d = x[ej] - x[ei]
            length = np.sqrt(np.einsum("ij,ij->i", d, d)) + 1e-12
            n = d / length[:, None]

        # --- XPBD distance projection with damage softening (necking) -------
        dam = np.clip(self.eps_acc / self.eps_fail_e, 0.0, 1.0)
        k = self.k0 / (1.0 + self.softening * dam)
        alpha = 1.0 / k
        denom = w[ei] + w[ej] + alpha / (h * h)
        s = (length - self.lp) / denom
        corr = (self.relax * s)[:, None] * n
        np.add.at(x, ei, w[ei, None] * corr)
        np.add.at(x, ej, -w[ej, None] * corr)

    def _solve_floor(self) -> None:
        below = self.x[:, 2] < FLOOR_Z + self.particle_radius
        if np.any(below):
            self.x[below, 2] = FLOOR_Z + self.particle_radius

    def _substep(self, h: float) -> None:
        x_prev = self.x.copy()
        # explicit inertial prediction
        self.v += self.g[None, :] * h
        self.v *= max(0.0, 1.0 - self.drag * h)
        np.clip(self.v, -self.v_max, self.v_max, out=self.v)
        self.x += self.v * h
        # constraint projection (shell plasticity + rigid floor)
        for _ in range(self.n_passes):
            self._solve_edges(h)
        self._solve_floor()
        # velocity from positional change (implicit contact response)
        self.v = (self.x - x_prev) / h
        # inelastic friction on floor contacts
        contact = self.x[:, 2] < FLOOR_Z + self.particle_radius + 1e-4
        if np.any(contact):
            self.v[contact, :2] *= max(0.0, 1.0 - self.friction_rate * h)

    # ------------------------------------------------------------- sampling
    def _deformation_gradient(self) -> np.ndarray:
        """Least-squares per-sample deformation gradient from edge pairs."""
        F = np.zeros((self.n, 3, 3))
        if len(self.ei) == 0:
            F[:, 0, 0] = F[:, 1, 1] = F[:, 2, 2] = 1.0
            return F
        dX = self.x0[self.ej] - self.x0[self.ei]
        dx = self.x[self.ej] - self.x[self.ei]
        AX = np.einsum("ei,ej->eij", dX, dX)
        BX = np.einsum("ei,ej->eij", dx, dX)
        A = np.zeros((self.n, 3, 3))
        B = np.zeros((self.n, 3, 3))
        np.add.at(A, self.ei, AX)
        np.add.at(A, self.ej, AX)
        np.add.at(B, self.ei, BX)
        np.add.at(B, self.ej, BX)
        tr = np.einsum("nii->n", A)
        reg = (1e-4 * np.maximum(tr, 1e-9) + 1e-9)[:, None, None] * np.eye(3)
        Ainv = np.linalg.inv(A + reg)
        F = np.einsum("nij,njk->nik", B, Ainv)
        return F

    def _ground_sdf(self) -> np.ndarray:
        q = np.abs(self.x - GROUND_CENTER[None, :]) - GROUND_HALF[None, :]
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(q.max(axis=1), 0.0)
        return outside + inside

    def _frame(self, time: float) -> StateFrame:
        attrs = dict(self.attr0)
        attrs["deformation_gradient"] = self._deformation_gradient()
        attrs["mass_quantity"] = self.mass.copy()
        fields = {
            "cumulative_mass_source": np.zeros(self.n),
            "contact_signed_distance": self._ground_sdf(),
            "external_force": self.external_force.copy(),
        }
        scalars = dict(self.scalars0)
        scalars["contact_length_scale"] = self.spacing
        frame = StateFrame(
            float(time),
            self.x.copy(),
            self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attrs,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------- contract
    def advance_to(self, time: float) -> StateFrame:
        if not self._ready:
            raise RuntimeError("solver not initialized")
        t_target = max(float(time), 0.0)
        while self.t < t_target - 1e-9:
            h = min(self.dt_sub, t_target - self.t)
            self._substep(h)
            self.t += h
        if abs(self.t - t_target) < 1e-9:
            self.t = t_target
        return self._frame(t_target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_particles": int(self.n),
            "n_edges_initial": int(self.n_edges0),
            "n_edges_alive": int(len(self.ei)),
            "n_edges_broken": int(self.n_edges0 - len(self.ei)),
            "max_plastic_strain": float(self.eps_acc.max()) if len(self.eps_acc) else 0.0,
            "first_fracture_time": self.first_fracture_time,
            "time": float(self.t),
        }
