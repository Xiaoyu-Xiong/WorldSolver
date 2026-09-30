"""Physics solver for scene 059_pneumatic_cushion.

Inflation of a sealed, finite-thickness rubber cushion by a prescribed
uniform internal chamber pressure.

Physical model
--------------
The cushion wall is a three-dimensional nearly-incompressible neo-Hookean
solid (Simo-Pister compressible neo-Hookean energy)

    psi(F) = mu/2 (tr(F^T F) - 3) - mu ln J + lambda/2 (ln J)^2 ,
    P(F)  = mu (F - F^{-T}) + lambda ln J F^{-T} ,

discretized with the frozen linear-tetrahedron mesh of the wall
(assets/cushion_solid.npz).  The internal pressure acts as a follower
load on the inner chamber surface.  It is applied through the exact
gradient of the chamber volume,

    V(x) = sum_T det[A_T, B_T, C_T] / 6   (outward-oriented triangles),

so the consistent nodal pressure force is f = p * dV/dx, which is
equivalent to (and numerically better behaved than) lumped surface
tractions and reproduces the traction distribution of a uniform pressure
acting on the inner chamber wall.

Time integration
----------------
Explicit central-difference dynamics with lumped nodal masses and
mass-proportional (Rayleigh) damping.  The pressure follows the frozen
ramp (0 -> 10 kPa over [0, 1.6] s, held to 2.0 s); the ramp is slow
compared with the structural response, so the dynamics track the
quasi-static inflation path while remaining a genuine physical
time-stepping scheme (no keyframes, no precomputed motion).  Rigid body
modes are projected out of the velocity field every step, which realizes
the "remove_rigid_body_modes" boundary condition (the cushion stays
centered without restraining face inflation).

All state is deterministic: fixed substep count, pure NumPy operations.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


def _find_asset(scene) -> Path:
    """Locate the frozen solid mesh asset for this scene."""
    candidates = []
    private = getattr(scene, "_asset", None)
    if private is not None:
        candidates.append(Path(private))
    try:
        import inspect

        candidates.append(
            Path(inspect.getfile(type(scene))).parent / "assets" / "cushion_solid.npz"
        )
    except Exception:  # pragma: no cover - defensive only
        pass
    candidates.append(Path("assets") / "cushion_solid.npz")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("cushion_solid.npz asset not found")


class BenchmarkSolver:
    """Explicit FEM solver for the pneumatic cushion inflation experiment."""

    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.seed = int(seed)

        geometry = cfg.get("geometry", {})
        outer = np.asarray(geometry.get("outer_dimensions_m", [1.2, 0.8, 0.55]), float)
        self.outer_dims = outer
        self.chamber_center = np.asarray(
            geometry.get("chamber_center_m", [0.0, 0.0, 0.0]), float
        )

        material = cfg.get("material_properties", {})
        self.young = float(material.get("young_modulus_pa", 90000.0))
        self.poisson = float(material.get("poisson_ratio", 0.499))
        self.density = float(material.get("density_kg_m3", 1070.0))
        self.mu = self.young / (2.0 * (1.0 + self.poisson))
        self.lam = (
            self.young * self.poisson / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        )
        self.bulk = self.lam + 2.0 * self.mu / 3.0

        loading = cfg.get("loading", {})
        self.p_target = float(loading.get("target_internal_pressure_pa", 10000.0))
        self.p_ramp_start = float(loading.get("ramp_start_s", 0.0))
        self.p_ramp_end = float(loading.get("ramp_end_s", 1.6))
        self.p_external = float(loading.get("external_pressure_pa", 0.0))
        self.gravity = np.asarray(cfg.get("gravity_m_s2", [0.0, 0.0, 0.0]), float)

        # Frozen force scale (constant across frames, strictly positive).
        self.reference_force = abs(self.p_target) * float(outer[0] * outer[1])

        asset = _find_asset(scene)
        with np.load(asset) as data:
            vertices = np.asarray(data["vertices"], dtype=np.float64)
            tets = np.asarray(data["tetrahedra"], dtype=np.int64)
            inner_faces = np.asarray(data["inner_chamber_faces"], dtype=np.int64)
            wall_region = np.asarray(data["wall_region"], dtype=np.int32)
            nodal_volume = np.asarray(data["nodal_volume"], dtype=np.float64)

        # Ensure a positive (counter-clockwise) tet orientation; permuting the
        # node order of a tet leaves its deformation gradient invariant.
        p0 = vertices[tets[:, 0]]
        dm_check = np.stack(
            [vertices[tets[:, 1]] - p0, vertices[tets[:, 2]] - p0, vertices[tets[:, 3]] - p0],
            axis=2,
        )
        neg = np.linalg.det(dm_check) < 0.0
        if np.any(neg):
            swap = tets[neg].copy()
            tmp = swap[:, 1].copy()
            swap[:, 1] = swap[:, 2]
            swap[:, 2] = tmp
            tets[neg] = swap

        self.vertices0 = vertices
        self.tets = tets
        self.inner_faces = self._orient_inner_faces(vertices, inner_faces)
        self.wall_region = wall_region

        # Material shape-function gradients: G[a] = M[a] @ inv(Dm), with
        # F = sum_a x_a (x) G_a.
        x0 = vertices[tets]  # (E,4,3)
        dm = np.stack([x0[:, 1] - x0[:, 0], x0[:, 2] - x0[:, 0], x0[:, 3] - x0[:, 0]], axis=2)
        self.vol_e = np.linalg.det(dm) / 6.0
        self.dm_inv = np.linalg.inv(dm)
        m_mat = np.array(
            [[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.grad = np.einsum("aj,ejk->eak", m_mat, self.dm_inv)  # (E,4,3)

        # Lumped nodal mass from the frozen nodal volumes.
        self.mass = self.density * nodal_volume
        if np.any(self.mass <= 0.0):
            self.mass = np.maximum(self.mass, 1e-9)

        # Stable explicit substep from the CFL condition of the
        # compressible neo-Hookean solid (longitudinal wave speed).
        h_min = self._min_edge(vertices, tets)
        c_wave = math.sqrt((self.bulk + 4.0 * self.mu / 3.0) / self.density)
        self.dt = 0.15 * h_min / c_wave
        self.damping = 20.0  # mass-proportional Rayleigh damping [1/s]

        # Dynamic state.
        self.x = vertices.copy()
        self.v = np.zeros_like(vertices)
        self.v_prev = np.zeros_like(vertices)
        self.time = 0.0
        self._steps = 0
        self._frames = 0

    # ------------------------------------------------------------------ setup

    @staticmethod
    def _min_edge(vertices: np.ndarray, tets: np.ndarray) -> float:
        h_min = np.inf
        for a, b in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            edge = np.linalg.norm(vertices[tets[:, a]] - vertices[tets[:, b]], axis=1)
            h_min = min(h_min, float(edge.min()))
        return h_min

    def _orient_inner_faces(self, vertices, faces):
        """Orient inner chamber triangles so normals point out of the chamber."""
        faces = faces.copy()
        a = vertices[faces[:, 0]]
        b = vertices[faces[:, 1]]
        c = vertices[faces[:, 2]]
        normal = np.cross(b - a, c - a)
        centroid = (a + b + c) / 3.0 - self.chamber_center
        flip = np.einsum("ij,ij->i", normal, centroid) < 0.0
        if np.any(flip):
            tmp = faces[flip, 1].copy()
            faces[flip, 1] = faces[flip, 2]
            faces[flip, 2] = tmp
        return faces

    # ----------------------------------------------------------------- forces

    def _pressure_at(self, t: float) -> float:
        if t <= self.p_ramp_start:
            return 0.0
        if t >= self.p_ramp_end:
            return self.p_target
        span = max(self.p_ramp_end - self.p_ramp_start, 1e-12)
        return self.p_target * (t - self.p_ramp_start) / span

    def _internal_force(self, x: np.ndarray) -> np.ndarray:
        """Elastic nodal forces of the neo-Hookean wall (negative gradient)."""
        xe = x[self.tets]  # (E,4,3)
        ds = np.stack(
            [xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0], xe[:, 3] - xe[:, 0]], axis=2
        )
        f_map = np.einsum("eij,ejk->eik", ds, self.dm_inv)  # (E,3,3)
        j_det = np.linalg.det(f_map)
        # Guard against (non-physical) inversion: keep the log finite.
        j_log = np.log(np.clip(j_det, 1e-6, None))
        f_inv = np.linalg.inv(f_map + 1e-12 * np.eye(3)[None, :, :])
        f_inv_t = np.transpose(f_inv, (0, 2, 1))
        p_stress = (
            self.mu * (f_map - f_inv_t)
            + (self.lam * j_log)[:, None, None] * f_inv_t
        )  # first Piola (E,3,3)
        f_elem = -self.vol_e[:, None, None] * np.einsum(
            "eij,eaj->eai", p_stress, self.grad
        )  # (E,4,3)
        f_int = np.zeros_like(x)
        np.add.at(f_int, self.tets, f_elem)
        return f_int

    def _pressure_force(self, x: np.ndarray, pressure: float) -> np.ndarray:
        """Consistent nodal follower load of the uniform chamber pressure."""
        f_ext = np.zeros_like(x)
        if pressure == 0.0:
            return f_ext
        a = x[self.inner_faces[:, 0]]
        b = x[self.inner_faces[:, 1]]
        c = x[self.inner_faces[:, 2]]
        # d/dA det[A,B,C]/6 = (B x C)/6, cyclic for the other vertices.
        g_a = np.cross(b, c) / 6.0
        g_b = np.cross(c, a) / 6.0
        g_c = np.cross(a, b) / 6.0
        np.add.at(f_ext, self.inner_faces[:, 0], pressure * g_a)
        np.add.at(f_ext, self.inner_faces[:, 1], pressure * g_b)
        np.add.at(f_ext, self.inner_faces[:, 2], pressure * g_c)
        return f_ext

    def _chamber_volume(self, x: np.ndarray) -> float:
        a = x[self.inner_faces[:, 0]]
        b = x[self.inner_faces[:, 1]]
        c = x[self.inner_faces[:, 2]]
        return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)

    def _remove_rigid_modes(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Project out rigid translation and rotation of the velocity field."""
        m = self.mass
        total = float(np.sum(m))
        com = np.sum(m[:, None] * x, axis=0) / total
        r = x - com
        v_lin = np.sum(m[:, None] * v, axis=0) / total
        ang_mom = np.sum(m[:, None] * np.cross(r, v), axis=0)
        inertia = np.einsum("i,ij,ik->jk", m, r, r)
        trace = float(np.trace(inertia))
        inertia = inertia * (-1.0)
        inertia[0, 0] += trace
        inertia[1, 1] += trace
        inertia[2, 2] += trace
        try:
            omega = np.linalg.solve(inertia, ang_mom)
        except np.linalg.LinAlgError:
            omega = np.zeros(3)
        return v - v_lin[None, :] - np.cross(omega[None, :], r)

    # ---------------------------------------------------------------- stepping

    def _step(self, h: float, pressure: float) -> None:
        """One central-difference substep of length h."""
        f_int = self._internal_force(self.x)
        f_p = self._pressure_force(self.x, pressure)
        f_g = self.mass[:, None] * self.gravity[None, :]
        accel = (f_int + f_p + f_g) / self.mass[:, None]
        # Mass-proportional Rayleigh damping integrated exactly.
        decay = 1.0 - 0.5 * self.damping * h
        grow = 1.0 + 0.5 * self.damping * h
        v_half = (self.v * decay + h * accel) / grow
        self.v_prev = self.v
        self.v = self._remove_rigid_modes(self.x, v_half)
        self.x = self.x + h * self.v
        self._steps += 1

    def advance_to(self, time: float):
        t_target = float(time)
        if t_target < self.time:
            raise ValueError("advance_to requires non-decreasing sample times")
        while t_target - self.time > 1e-12:
            remaining = t_target - self.time
            n_sub = int(math.ceil(remaining / self.dt - 1e-9)) or 1
            h = remaining / n_sub
            for _ in range(n_sub):
                self._step(h, self._pressure_at(self.time + h * (_ + 1)))
            self.time = t_target

        pressure = self._pressure_at(t_target)
        f_p = self._pressure_force(self.x, pressure)
        f_g = self.mass[:, None] * self.gravity[None, :]
        f_ext = f_p + f_g
        net_external = f_ext.sum(axis=0)

        # Per-element deformation gradient in material coordinates.
        xe = self.x[self.tets]
        ds = np.stack(
            [xe[:, 1] - xe[:, 0], xe[:, 2] - xe[:, 0], xe[:, 3] - xe[:, 0]], axis=2
        )
        f_map = np.einsum("eij,ejk->eik", ds, self.dm_inv)

        # Report the time-centred velocity at the sample instant.
        v_report = 0.5 * (self.v_prev + self.v)

        # Two budget domains (combined scene domain and the cushion_wall body
        # domain) x one conserved material.
        budget = np.zeros((2, 1, 3))
        budget[0, 0, :] = net_external  # combined domain: internal pairs cancel
        budget[1, 0, :] = net_external  # cushion_wall body domain

        frame = StateFrame(
            time=t_target,
            position=self.x.copy(),
            velocity=v_report,
            mass=self.mass.copy(),
            attributes={
                "mass_quantity": self.mass.copy(),
                "wall_region": self.wall_region.copy(),
            },
            fields={
                "element_deformation_gradient": f_map,
                "budget_external_force": budget,
                "cumulative_mass_source": np.zeros(1),
                "external_force": f_ext,
            },
            scalars={
                "internal_pressure_pa": float(pressure),
                "external_pressure_pa": float(self.p_external),
                "reference_force": float(self.reference_force),
                "chamber_volume_m3": float(self._chamber_volume(self.x)),
                "characteristic_length": float(np.linalg.norm(self.outer_dims)),
            },
        )
        frame.validate()
        self._frames += 1
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "solver": "explicit-neo-hookean-fem",
            "substep_dt_s": float(self.dt),
            "substeps_taken": int(self._steps),
            "frames_emitted": int(self._frames),
            "chamber_volume_m3": float(self._chamber_volume(self.x)),
            "internal_pressure_pa": float(self._pressure_at(self.time)),
            "max_displacement_m": float(
                np.max(np.linalg.norm(self.x - self.vertices0, axis=1))
            ),
            "all_finite": bool(
                np.isfinite(self.x).all() and np.isfinite(self.v).all()
            ),
        }
