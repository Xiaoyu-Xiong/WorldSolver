"""Physics solver for scene 063_rubber_cube_compression.

Lateral bulging of a compressed rubber cube.

Method
------
Volumetric (3D) finite elements on the frozen tetrahedral mesh of the cube
(729 nodes / 3072 linear tets).  Material: compressible neo-Hookean rubber

    psi(F) = mu/2 (tr(F^T F) - 3) - mu ln J + lambda/2 (ln J)^2 ,
    P(F)   = mu (F - F^-T) + lambda ln(J) F^-T .

Boundary conditions (from the frozen configuration):
  * z_min face : fully fixed support (all displacement components held).
  * z_max face : prescribed uniform downward surface traction, ramped
    linearly 0 -> 50 kPa between t = 0.25 s and t = 1.5 s, then held.

Time integration: symplectic (semi-implicit) Euler with a CFL-limited
substep, mass-lumped nodal masses, and unconditionally-stable implicit
mass-proportional (Rayleigh) damping so the slowly ramped response stays
essentially quasi-static and free of persistent ringing.  All state is
advanced by real force-based stepping; nothing is keyframed.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# frozen material / numerics constants (SI units)
# ---------------------------------------------------------------------------
GRAVITY = 9.80665          # m/s^2
YOUNG_MODULUS = 2.0e5      # Pa   (soft rubber)
POISSON_RATIO = 0.45       # near-incompressible rubber
DENSITY = 1100.0           # kg/m^3
MASS_DAMPING = 25.0        # 1/s   Rayleigh mass-proportional damping
CFL_SAFETY = 0.30          # substep safety factor


class BenchmarkSolver:
    """Tetrahedral FEM solver for the compressed rubber cube."""

    def initialize(self, scene, seed: int) -> None:
        self.seed = int(seed)
        self.config = getattr(scene, "config", {}) or {}

        # ---- frozen mesh -------------------------------------------------
        verts = getattr(scene, "vertices", None)
        tets = getattr(scene, "tetrahedra", None)
        faces = getattr(scene, "faces", None)
        if verts is None or tets is None or faces is None:
            asset = Path(__file__).resolve().parents[1] / "assets" / "rubber_cube_mesh.npz"
            with np.load(asset) as mesh:
                verts = np.asarray(mesh["vertices"], dtype=float)
                tets = np.asarray(mesh["tetrahedra"], dtype=np.int64)
                faces = np.asarray(mesh["faces"], dtype=np.int64)

        self.rest = np.array(verts, dtype=float)
        self.tets = np.array(tets, dtype=np.int64)
        self.faces = np.array(faces, dtype=np.int64)
        self.n_nodes = self.rest.shape[0]
        self.n_tets = self.tets.shape[0]

        # ---- boundary conditions from the frozen configuration -----------
        bc = self.config.get("boundary_conditions", {}) or {}
        top_bc = bc.get("top_compressive_traction", {}) or {}
        self.traction_dir = np.array(top_bc.get("direction", [0.0, 0.0, -1.0]), dtype=float)
        norm = float(np.linalg.norm(self.traction_dir))
        self.traction_dir = self.traction_dir / (norm if norm > 0 else 1.0)
        self.peak_pressure = float(top_bc.get("peak_magnitude_pa", 5.0e4))
        sched = top_bc.get("schedule") or [
            {"time": 0.0, "scale": 0.0},
            {"time": 0.25, "scale": 0.0},
            {"time": 1.5, "scale": 1.0},
            {"time": 2.0, "scale": 1.0},
        ]
        self._sched_t = np.array([float(p["time"]) for p in sched], dtype=float)
        self._sched_s = np.array([float(p["scale"]) for p in sched], dtype=float)
        order = np.argsort(self._sched_t)
        self._sched_t = self._sched_t[order]
        self._sched_s = self._sched_s[order]

        bottom_bc = bc.get("bottom_support", {}) or {}
        self._full_bottom_fix = str(bottom_bc.get("type", "fixed_support")) == "fixed_support"

        # ---- reference element data ---------------------------------------
        x0 = self.rest[self.tets[:, 0]]
        dm = np.transpose(self.rest[self.tets[:, 1:]] - x0[:, None, :], (0, 2, 1))
        det_dm = np.linalg.det(dm)
        # enforce positive orientation / positive rest volume
        flipped = det_dm < 0.0
        if np.any(flipped):
            self.tets[flipped] = self.tets[flipped][:, [0, 2, 1, 3]]
            x0 = self.rest[self.tets[:, 0]]
            dm = np.transpose(self.rest[self.tets[:, 1:]] - x0[:, None, :], (0, 2, 1))
            det_dm = np.linalg.det(dm)
        self.dm = dm
        self.minv = np.linalg.inv(dm)
        self.minv_t = np.transpose(self.minv, (0, 2, 1))
        self.rest_volume = det_dm / 6.0                     # (Ne,)
        self.total_volume = float(self.rest_volume.sum())

        # Lamé parameters
        self.mu = YOUNG_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
        self.lam = YOUNG_MODULUS * POISSON_RATIO / (
            (1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO)
        )

        # ---- lumped nodal mass --------------------------------------------
        tet_share = DENSITY * self.rest_volume[:, None] / 4.0   # (Ne,4)
        self.mass = np.bincount(
            self.tets.ravel(), weights=tet_share.ravel(), minlength=self.n_nodes
        )
        self.total_mass = float(self.mass.sum())

        # ---- face masks (from rest geometry) -------------------------------
        rest_attr = getattr(scene, "_rest_attributes", None)
        z = self.rest[:, 2]
        zmin, zmax = float(z.min()), float(z.max())
        span = max(zmax - zmin, 1e-12)
        tol = 1e-6 * span
        self.bottom_mask = np.isclose(z, zmin, atol=tol)
        self.top_mask = np.isclose(z, zmax, atol=tol)

        # ---- lumped traction area on the loaded (top) face -----------------
        tri = self.rest[self.faces]                          # (F,3,3)
        centroid = tri.mean(axis=1)
        area = 0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        )
        top_face = np.isclose(centroid[:, 2], zmax, atol=tol) & (area > 0.0)
        self.top_area = np.zeros(self.n_nodes, dtype=float)
        if np.any(top_face):
            face_ids = self.faces[top_face]
            share = np.full(face_ids.shape, area[top_face][:, None] / 3.0)
            self.top_area = np.bincount(
                face_ids.ravel(), weights=share.ravel(), minlength=self.n_nodes
            )
        self.top_total_area = float(self.top_area.sum())
        # fall back to a uniform split if surface faces are unavailable
        if self.top_total_area <= 0.0:
            n_top = int(self.top_mask.sum())
            self.top_area = np.where(self.top_mask, span * span / max(n_top, 1), 0.0)
            self.top_total_area = float(self.top_area.sum())

        self.reference_force = self.peak_pressure * self.top_total_area

        # ---- stable substep (CFL on dilatational wave speed) ---------------
        bulk_plus_shear = self.lam + 2.0 * self.mu       # K + 4G/3
        wave_speed = math.sqrt(max(bulk_plus_shear, 1.0) / DENSITY)
        # minimum tet altitude of the rest mesh
        altitudes = np.full(self.n_tets, np.inf)
        for k in range(4):
            opp = np.array([j for j in range(4) if j != k])
            edge_a = self.rest[self.tets[:, opp[1]]] - self.rest[self.tets[:, opp[0]]]
            edge_b = self.rest[self.tets[:, opp[2]]] - self.rest[self.tets[:, opp[0]]]
            face_area = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
            altitudes = np.minimum(altitudes, 3.0 * self.rest_volume / np.maximum(face_area, 1e-15))
        self.h_min = float(altitudes.min())
        dt_cfl = CFL_SAFETY * self.h_min / wave_speed
        sample_dt = float(self.config.get("sample_dt", 0.02) or 0.02)
        duration = float(self.config.get("duration", 2.0) or 2.0)
        n_sub = max(1, int(math.ceil(sample_dt / max(dt_cfl, 1e-9))))
        self.substeps_per_sample = n_sub
        self.dt = sample_dt / n_sub
        self.sample_dt = sample_dt
        self.duration = duration

        # ---- dynamic state --------------------------------------------------
        self.position = self.rest.copy()
        self.velocity = np.zeros((self.n_nodes, 3), dtype=float)
        self.time = 0.0
        self._steps = 0
        self._last_frame_time = -1.0

    # ------------------------------------------------------------------ utils
    def _pressure(self, t: float) -> float:
        return self.peak_pressure * float(np.interp(t, self._sched_t, self._sched_s))

    def _tet_frames(self, x: np.ndarray):
        """Return per-tet deformation gradients for current positions."""
        x0 = x[self.tets[:, 0]]
        ds = np.transpose(x[self.tets[:, 1:]] - x0[:, None, :], (0, 2, 1))
        return ds @ self.minv

    def _elastic_forces(self, x: np.ndarray):
        """Nodal internal elastic forces and per-tet F (neo-Hookean)."""
        fmat = self._tet_frames(x)                                    # (Ne,3,3)
        det_f = np.linalg.det(fmat)
        det_f = np.where(np.abs(det_f) < 1e-12, np.sign(det_f) * 1e-12 + 1e-12, det_f)
        inv_ft = np.transpose(np.linalg.inv(fmat), (0, 2, 1))
        log_j = np.log(np.abs(det_f))
        p = self.mu * (fmat - inv_ft) + self.lam * log_j[:, None, None] * inv_ft
        h = -self.rest_volume[:, None, None] * (p @ self.minv_t)      # (Ne,3,3)
        f1, f2, f3 = h[:, :, 0], h[:, :, 1], h[:, :, 2]
        f0 = -(f1 + f2 + f3)
        weights = np.stack([f0, f1, f2, f3], axis=1)                  # (Ne,4,3)
        ids = self.tets.ravel()
        f_node = np.empty((self.n_nodes, 3), dtype=float)
        flat = weights.reshape(-1, 3)
        for c in range(3):
            f_node[:, c] = np.bincount(ids, weights=flat[:, c], minlength=self.n_nodes)
        return f_node, fmat

    def _external_forces(self, v: np.ndarray, t: float):
        """Per-node external forces: gravity + prescribed top traction.

        (Rayleigh damping is applied implicitly in the integrator.)
        """
        f = np.zeros((self.n_nodes, 3), dtype=float)
        f[:, 2] -= self.mass * GRAVITY
        pressure = self._pressure(t)
        if pressure != 0.0:
            f += pressure * self.top_area[:, None] * self.traction_dir[None, :]
        return f

    # ------------------------------------------------------------- time step
    def _step(self, dt: float) -> None:
        x, v = self.position, self.velocity
        f_int, _ = self._elastic_forces(x)
        f_ext = self._external_forces(v, self.time)
        acc = (f_int + f_ext) / self.mass[:, None]
        # semi-implicit Euler with implicit mass-proportional damping
        v_new = (v + dt * acc) / (1.0 + dt * MASS_DAMPING)
        x_new = x + dt * v_new

        # fixed support: entire bottom face held (fixed_support)
        if self._full_bottom_fix:
            x_new[self.bottom_mask] = self.rest[self.bottom_mask]
            v_new[self.bottom_mask] = 0.0
        else:
            x_new[self.bottom_mask, 2] = self.rest[self.bottom_mask, 2]
            v_new[self.bottom_mask, 2] = 0.0

        # safety: no node may penetrate the rigid support plane z = 0
        below = x_new[:, 2] < 0.0
        if np.any(below):
            x_new[below, 2] = 0.0
            v_new[below, 2] = np.maximum(v_new[below, 2], 0.0)

        self.position = x_new
        self.velocity = v_new
        self.time += dt
        self._steps += 1

    # ------------------------------------------------------------- interface
    def advance_to(self, time: float):
        target = max(float(time), 0.0)
        guard = 0
        while self.time < target - 1e-12:
            dt = min(self.dt, target - self.time)
            self._step(dt)
            guard += 1
            if guard > 4_000_000:
                break
        return self._snapshot(self.time)

    def _snapshot(self, t: float):
        from worldsolver.contracts import StateFrame

        x, v = self.position, self.velocity
        f_int, fmat = self._elastic_forces(x)
        f_ext = self._external_forces(v, t)

        # external force terms on the single (cube) budget domain:
        # [0] gravity, [1] prescribed top traction, [2] Rayleigh damping
        f_grav = np.zeros((self.n_nodes, 3), dtype=float)
        f_grav[:, 2] = -self.mass * GRAVITY
        pressure = self._pressure(t)
        f_trac = pressure * self.top_area[:, None] * self.traction_dir[None, :]
        f_damp = -MASS_DAMPING * self.mass[:, None] * v
        budget = np.stack(
            [f_grav.sum(axis=0), f_trac.sum(axis=0), f_damp.sum(axis=0)]
        )[None, :, :]

        attributes = {
            "mass_quantity": self.mass.copy(),
            "rest_position": self.rest.copy(),
            "bottom_face": self.bottom_mask.astype(float),
            "top_face": self.top_mask.astype(float),
        }
        fields = {
            "element_deformation_gradient": np.asarray(fmat, dtype=float),
            "budget_external_force": budget,
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "external_force": f_grav + f_trac + f_damp,
            "elastic_internal_force": f_int,
        }
        scalars = {
            "reference_force": float(self.reference_force),
            "applied_traction_pa": float(pressure),
        }
        frame = StateFrame(
            time=float(t),
            position=x.copy(),
            velocity=v.copy(),
            mass=self.mass.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        height = float(
            self.position[self.top_mask, 2].max() - self.position[self.bottom_mask, 2].min()
        )
        return {
            "method": "tetrahedral_fem_neo_hookean_semi_implicit_euler",
            "nodes": self.n_nodes,
            "tets": self.n_tets,
            "young_modulus_pa": YOUNG_MODULUS,
            "poisson_ratio": POISSON_RATIO,
            "density_kg_m3": DENSITY,
            "mass_damping_sinv": MASS_DAMPING,
            "substep_dt": self.dt,
            "substeps_per_sample": self.substeps_per_sample,
            "steps_taken": self._steps,
            "total_mass_kg": self.total_mass,
            "reference_force_n": self.reference_force,
            "current_height_m": height,
        }
