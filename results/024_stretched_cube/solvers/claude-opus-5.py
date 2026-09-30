"""Stretched-cube solver: inversion-safe Neo-Hookean tetrahedral FEM.

Physical model
--------------
The deformable cube is discretized as a regular hexahedral lattice split into
Kuhn tetrahedra.  Each tet carries a stable Neo-Hookean energy density

    Psi(F) = mu/2 (tr(F^T F) - 3) - mu (J - 1) + lam/2 (J - 1)^2

whose first Piola stress ``P = mu F + (lam (J-1) - mu) dJ/dF`` is polynomial in
F.  It therefore stays finite and well defined even for inverted elements
(det F <= 0), which the stretch/compression script deliberately provokes.

Time stepping is a symplectic-Euler (explicit) scheme with mass-proportional
damping plus a strain-rate (viscous) stress, substepped well below the elastic
CFL limit, so the trajectory is stable and deterministic.

Loading script
--------------
* ``0 <= t < stretch_end``  : the two grips clamp the end cross sections and
  separate smoothly until the cube is stretched by ``stretch_ratio``.
* hold phase               : the large-deformation state relaxes / settles.
* ``t >= pulse_time``      : a short compressive pulse (inward grip motion plus
  an inward body-force pulse) is applied, after which the grips return to the
  stretched separation and the cube oscillates and recovers.

Only the deformable subject is exported; its surface vertices are embedded in
the simulation lattice with barycentric weights.
"""
from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame

# Kuhn decomposition of the unit hexahedron into six tetrahedra.
_KUHN = (
    ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (1, 0, 0), (1, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)),
    ((0, 0, 0), (0, 1, 0), (0, 1, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1)),
    ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)),
)


def _smootherstep(u: float) -> tuple[float, float]:
    """C2 ramp on [0, 1] returning (value, d/du)."""
    u = min(max(u, 0.0), 1.0)
    return (
        u * u * u * (10.0 + u * (-15.0 + 6.0 * u)),
        30.0 * u * u * (1.0 + u * (-2.0 + u)),
    )


def _box_sdf(points: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    q = np.abs(points - center[None, :]) - half[None, :]
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def _box_contact(points, center, half):
    """Penetration depth (>0 inside) and outward normal for an AABB."""
    rel = points - center[None, :]
    q = np.abs(rel) - half[None, :]
    inside = np.all(q < 0.0, axis=1)
    axis = np.argmax(q, axis=1)
    rows = np.arange(len(points))
    depth = np.where(inside, -q[rows, axis], 0.0)
    normal = np.zeros_like(points)
    sign = np.sign(rel[rows, axis])
    sign = np.where(sign == 0.0, 1.0, sign)
    normal[rows, axis] = sign
    return inside, depth, normal


class BenchmarkSolver:
    """Explicit inversion-safe FEM solver for the stretched-cube scene."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)
        np.random.seed(self.seed % (2**32))

        self.vertices0 = np.asarray(scene.vertices, dtype=float).copy()
        self.n_out = len(self.vertices0)

        self.duration = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])
        self.gravity = np.asarray(cfg.get("gravity", [0.0, 0.0, 0.0]), dtype=float)
        self.total_mass = float(cfg.get("subject_mass", 1.0))
        self.contact_len = float(cfg.get("contact_length_scale", 0.04))
        self.ground_z = float(cfg.get("ground_z", 0.0))
        self.stretch_ratio = float(cfg.get("stretch_ratio", 2.0))
        self.stretch_end = float(cfg.get("stretch_end", 1.0))
        self.pulse_time = float(cfg.get("pulse_time", 1.5))
        self.damping = float(cfg.get("damping", 0.0))
        self.friction = float(cfg.get("friction", 0.0))
        self.restitution = float(cfg.get("restitution", 0.0))

        # --- reference (material) configuration -----------------------------
        # The asset geometry itself is the undeformed reference state, so the
        # exported deformation gradient starts at the identity and grows with
        # the prescribed stretch.
        self.lo = self.vertices0.min(axis=0)
        self.hi = self.vertices0.max(axis=0)
        self.center = 0.5 * (self.lo + self.hi)
        self.extent = self.hi - self.lo
        self.half = 0.5 * self.extent
        self.axis = 0

        self._build_mesh()
        self._build_material()
        self._build_embedding()

        # --- loading script -------------------------------------------------
        self.pulse_dur = 0.20
        self.pulse_dip = 0.15            # grip retraction in stretch units
        self.pulse_accel = 8.0           # inward body-force pulse [m/s^2]
        self.grip_half = np.asarray([0.12, 0.55, 0.55], dtype=float)
        self.grip_clearance = 0.01

        # --- integrator -----------------------------------------------------
        wave = float(np.sqrt((self.lam + 2.0 * self.mu) / max(self.rho, 1e-9)))
        dx = float(np.min(self.cell_rest))
        dt_cfl = 0.25 * dx / max(wave, 1e-9)
        self.dt = float(min(6.0e-4, max(dt_cfl, 5.0e-5)))
        self.substeps = max(1, int(np.ceil(self.sample_dt / self.dt)))
        self.dt = self.sample_dt / self.substeps

        self.time = 0.0
        self.pos = self.node_x0.copy()
        self.vel = np.zeros_like(self.pos)
        v_init = np.asarray(cfg.get("initial_velocity", [0.0, 0.0, 0.0]), dtype=float)
        self.vel[:] = v_init[None, :]
        self._apply_grips(0.0)
        self.vel[self.clamped] = self._grip_velocity(0.0)[self.clamped]

        self.out_mass = np.full(self.n_out, self.total_mass / self.n_out, dtype=float)
        self.cum_mass_source = np.zeros(1, dtype=float)
        self._max_speed = 0.0
        self._min_det = 1.0
        self._steps = 0

    # ------------------------------------------------------------------ mesh
    def _build_mesh(self) -> None:
        nx, ny, nz = 10, 6, 6
        self.res = (nx, ny, nz)
        self.axes = [
            np.linspace(self.lo[d], self.hi[d], (nx, ny, nz)[d] + 1)
            for d in range(3)
        ]
        gx, gy, gz = np.meshgrid(self.axes[0], self.axes[1], self.axes[2], indexing="ij")
        rest = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        self.node_rest = rest
        self.n_nodes = len(rest)
        self.node_x0 = rest.copy()
        self.cell_rest = self.extent / np.asarray([nx, ny, nz], dtype=float)

        def nid(i, j, k):
            return (i * (ny + 1) + j) * (nz + 1) + k

        tets = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    for tet in _KUHN:
                        tets.append(
                            [nid(i + o[0], j + o[1], k + o[2]) for o in tet]
                        )
        self.tets = np.asarray(tets, dtype=np.int64)
        self.tets_per_cell = len(_KUHN)

        # clamped end layers (grip jaws)
        ii = np.repeat(np.arange(nx + 1), (ny + 1) * (nz + 1))
        self.node_i = ii
        self.clamped = (ii == 0) | (ii == nx)
        self.free = ~self.clamped

    def _build_material(self) -> None:
        E = float(self.cfg.get("stiffness", 10.0))
        nu = 0.40
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        # Strain-rate (Kelvin-Voigt) viscosity: light, so the post-pulse
        # oscillation stays visible while high-frequency modes are suppressed.
        self.kd = 0.02 * self.mu

        X = self.node_rest[self.tets]
        d = np.stack([X[:, 1] - X[:, 0], X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]], axis=2)
        self.Dm_inv = np.linalg.inv(d)
        self.vol = np.abs(np.linalg.det(d)) / 6.0
        rest_volume = float(self.vol.sum())
        self.rho = self.total_mass / max(rest_volume, 1e-12)

        mass = np.zeros(self.n_nodes, dtype=float)
        contrib = self.rho * self.vol / 4.0
        for c in range(4):
            np.add.at(mass, self.tets[:, c], contrib)
        mass = np.maximum(mass, 1e-9)
        mass *= self.total_mass / mass.sum()
        self.node_mass = mass
        self.inv_mass = 1.0 / mass

    # ------------------------------------------------------------- embedding
    def _build_embedding(self) -> None:
        """Locate every exported surface vertex inside its host tetrahedron."""
        nx, ny, nz = self.res
        res = np.asarray([nx, ny, nz], dtype=np.int64)
        v = self.vertices0
        idx = np.zeros((self.n_out, 3), dtype=np.int64)
        for d in range(3):
            frac = (v[:, d] - self.lo[d]) / max(float(self.extent[d]), 1e-12) * res[d]
            idx[:, d] = np.clip(np.floor(frac).astype(np.int64), 0, res[d] - 1)
        cell = (idx[:, 0] * ny + idx[:, 1]) * nz + idx[:, 2]

        best_score = np.full(self.n_out, -np.inf)
        best_tet = np.zeros(self.n_out, dtype=np.int64)
        best_w = np.zeros((self.n_out, 4), dtype=float)
        for c in range(self.tets_per_cell):
            tid = cell * self.tets_per_cell + c
            corners = self.tets[tid]
            X0 = self.node_rest[corners[:, 0]]
            bary = np.einsum("nij,nj->ni", self.Dm_inv[tid], v - X0)
            w = np.concatenate([(1.0 - bary.sum(axis=1))[:, None], bary], axis=1)
            score = w.min(axis=1)
            take = score > best_score
            best_score = np.where(take, score, best_score)
            best_tet = np.where(take, tid, best_tet)
            best_w = np.where(take[:, None], w, best_w)

        self.emb_tet = best_tet
        self.emb_nodes = self.tets[best_tet]
        self.emb_w = best_w

    def _interpolate(self, values: np.ndarray) -> np.ndarray:
        return np.einsum("nc,ncj->nj", self.emb_w, values[self.emb_nodes])

    # --------------------------------------------------------- loading script
    def _stretch_state(self, t: float) -> tuple[float, float]:
        """Prescribed grip stretch ratio and its time derivative."""
        target = self.stretch_ratio
        if self.stretch_end > 1e-9:
            f, df = _smootherstep(t / self.stretch_end)
            s = 1.0 + (target - 1.0) * f
            ds = (target - 1.0) * df / self.stretch_end
        else:
            s, ds = target, 0.0
        if self.pulse_dur > 1e-9 and t >= self.pulse_time:
            u = (t - self.pulse_time) / self.pulse_dur
            if u < 1.0:
                sn = np.sin(np.pi * u)
                s -= self.pulse_dip * sn * sn
                ds -= (
                    self.pulse_dip
                    * 2.0
                    * np.pi
                    * sn
                    * np.cos(np.pi * u)
                    / self.pulse_dur
                )
        return float(s), float(ds)

    def _grip_state(self, t: float):
        s, ds = self._stretch_state(t)
        rel = self.node_rest[:, 0] - self.center[0]
        pos = self.node_rest.copy()
        pos[:, 0] = self.center[0] + rel * s
        vel = np.zeros_like(pos)
        vel[:, 0] = rel * ds
        return pos, vel, s

    def _apply_grips(self, t: float) -> None:
        pos, vel, _ = self._grip_state(t)
        m = self.clamped
        self.pos[m] = pos[m]
        self.vel[m] = vel[m]

    def _grip_velocity(self, t: float) -> np.ndarray:
        return self._grip_state(t)[1]

    def _collider_boxes(self, t: float):
        """Ground slab plus the two kinematic grip jaws at time ``t``."""
        s, _ = self._stretch_state(t)
        inner = self.half[0] * s + self.grip_clearance
        boxes = [
            (
                np.asarray([0.0, 0.0, self.ground_z - 0.04], dtype=float),
                np.asarray([2.3, 1.7, 0.04], dtype=float),
            )
        ]
        for sign in (-1.0, 1.0):
            cx = self.center[0] + sign * (inner + self.grip_half[0])
            boxes.append(
                (
                    np.asarray([cx, self.center[1], self.center[2]], dtype=float),
                    self.grip_half.copy(),
                )
            )
        return boxes

    def _pulse_acceleration(self, t: float, rest_x: np.ndarray) -> np.ndarray:
        """Inward compressive body-force pulse (momentum neutral)."""
        acc = np.zeros((len(rest_x), 3), dtype=float)
        if self.pulse_dur > 1e-9 and self.pulse_time <= t < self.pulse_time + self.pulse_dur:
            u = (t - self.pulse_time) / self.pulse_dur
            sn = np.sin(np.pi * u)
            mag = self.pulse_accel * sn * sn
            acc[:, 0] = -mag * np.sign(rest_x - self.center[0])
        return acc

    # ------------------------------------------------------------- mechanics
    def _deformation_gradient(self, pos: np.ndarray) -> np.ndarray:
        x = pos[self.tets]
        ds = np.stack([x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]], axis=2)
        return ds @ self.Dm_inv

    @staticmethod
    def _cofactor(F: np.ndarray) -> np.ndarray:
        """dJ/dF = [f1 x f2, f2 x f0, f0 x f1] (columns), inverse-free."""
        f0, f1, f2 = F[:, :, 0], F[:, :, 1], F[:, :, 2]
        return np.stack(
            [np.cross(f1, f2), np.cross(f2, f0), np.cross(f0, f1)], axis=2
        )

    def _piola(self, F: np.ndarray, Fdot: np.ndarray):
        """Stable Neo-Hookean stress; polynomial in F, safe under inversion."""
        J = np.linalg.det(F)
        dJ = self._cofactor(F)
        P = self.mu * F + (self.lam * (J - 1.0) - self.mu)[:, None, None] * dJ
        if self.kd > 0.0:
            P = P + self.kd * Fdot
        return P, J

    def _internal_forces(self, pos: np.ndarray, vel: np.ndarray):
        F = self._deformation_gradient(pos)
        Fdot = self._deformation_gradient_rate(vel)
        P, J = self._piola(F, Fdot)
        # f = -vol * P * Dm_inv^T distributed to nodes 1..3, node 0 gets -sum.
        H = -self.vol[:, None, None] * (P @ np.transpose(self.Dm_inv, (0, 2, 1)))
        force = np.zeros_like(pos)
        for c in range(3):
            np.add.at(force, self.tets[:, c + 1], H[:, :, c])
        np.add.at(force, self.tets[:, 0], -H.sum(axis=2))
        return force, F, J

    def _deformation_gradient_rate(self, vel: np.ndarray) -> np.ndarray:
        v = vel[self.tets]
        dv = np.stack([v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], v[:, 3] - v[:, 0]], axis=2)
        return dv @ self.Dm_inv

    def _resolve_contacts(self, t: float) -> None:
        free = self.free
        pts = self.pos[free]
        vels = self.vel[free]
        for center, half in self._collider_boxes(t):
            inside, depth, normal = _box_contact(pts, center, half)
            if not np.any(inside):
                continue
            sel = inside
            n = normal[sel]
            vn = np.einsum("ij,ij->i", vels[sel], n)
            # positional projection out of the collider
            pts[sel] += n * depth[sel, None]
            approaching = vn < 0.0
            dv = np.where(approaching, -(1.0 + self.restitution) * vn, 0.0)
            vels[sel] += n * dv[:, None]
            # Coulomb friction on the tangential component
            vn2 = np.einsum("ij,ij->i", vels[sel], n)
            vt = vels[sel] - n * vn2[:, None]
            speed = np.linalg.norm(vt, axis=1)
            keep = np.maximum(0.0, 1.0 - self.friction * np.abs(dv) / np.maximum(speed, 1e-9))
            vels[sel] = n * vn2[:, None] + vt * keep[:, None]
        self.pos[free] = pts
        self.vel[free] = vels

    def _step(self, dt: float) -> None:
        t_next = self.time + dt
        force, _, J = self._internal_forces(self.pos, self.vel)
        acc_ext = self.gravity[None, :] + self._pulse_acceleration(
            self.time, self.node_rest[:, 0]
        )
        force = force + self.node_mass[:, None] * acc_ext
        force = np.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0)

        acc = force * self.inv_mass[:, None]
        self.vel += dt * acc
        # mass-proportional (Rayleigh alpha) damping, unconditionally stable
        self.vel *= float(np.exp(-self.damping * dt))
        # safety clamp: keeps a hypothetical instability from producing infinities
        speed = np.linalg.norm(self.vel, axis=1)
        vmax = 60.0
        over = speed > vmax
        if np.any(over):
            self.vel[over] *= (vmax / speed[over])[:, None]

        self.pos += dt * self.vel
        self.time = t_next
        self._apply_grips(self.time)
        self._resolve_contacts(self.time)

        self._steps += 1
        self._max_speed = max(self._max_speed, float(np.max(np.abs(self.vel))))
        self._min_det = min(self._min_det, float(np.min(J)))

    # ------------------------------------------------------------------ frame
    def _frame(self) -> StateFrame:
        pos = self._interpolate(self.pos)
        vel = self._interpolate(self.vel)
        F_elem = self._deformation_gradient(self.pos)
        F_out = F_elem[self.emb_tet]

        acc_ext = self.gravity[None, :] + self._pulse_acceleration(
            self.time, self.vertices0[:, 0]
        )
        ext_force = self.out_mass[:, None] * acc_ext

        sdf = None
        for center, half in self._collider_boxes(self.time):
            d = _box_sdf(pos, center, half)
            sdf = d if sdf is None else np.minimum(sdf, d)

        frame = StateFrame(
            time=float(self.time),
            position=np.ascontiguousarray(pos, dtype=float),
            velocity=np.ascontiguousarray(vel, dtype=float),
            mass=np.ascontiguousarray(self.out_mass, dtype=float),
            body_id=np.zeros(self.n_out, dtype=np.int64),
            attributes={
                "mass_quantity": np.ascontiguousarray(self.out_mass, dtype=float),
                "deformation_gradient": np.ascontiguousarray(F_out, dtype=float),
            },
            fields={
                "cumulative_mass_source": np.ascontiguousarray(
                    self.cum_mass_source, dtype=float
                ),
                "external_force": np.ascontiguousarray(ext_force, dtype=float),
                "contact_signed_distance": np.ascontiguousarray(sdf, dtype=float),
            },
            scalars={"contact_length_scale": float(self.contact_len)},
        )
        frame.validate()
        return frame

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self.time - 1e-12:
            raise ValueError("advance_to must be monotonically non-decreasing")
        while self.time < target - 1e-12:
            self._step(min(self.dt, target - self.time))
        self.time = target
        return self._frame()

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit symplectic-Euler stable Neo-Hookean tet FEM",
            "nodes": int(self.n_nodes),
            "elements": int(len(self.tets)),
            "dt": float(self.dt),
            "substeps_per_sample": int(self.substeps),
            "steps_taken": int(self._steps),
            "youngs_modulus": float(self.cfg.get("stiffness", 0.0)),
            "lame_mu": float(self.mu),
            "lame_lambda": float(self.lam),
            "density": float(self.rho),
            "stretch_ratio": float(self.stretch_ratio),
            "min_det_F": float(self._min_det),
            "max_abs_velocity": float(self._max_speed),
        }
