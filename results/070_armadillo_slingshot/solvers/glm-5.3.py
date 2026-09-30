"""Physically-based solver for the Armadillo slingshot scene (070).

Discretization
--------------
The frozen topology asset (``assets/armadillo_solid.npz``) provides the
volumetric node set and a per-part tetrahedralisation of the Armadillo solid.
The solver discretises the three-dimensional elastic solid as a lumped-mass
node network connected by elastic distance elements (a truss/lattice
discretisation of the continuum solid): every node pair closer than a frozen
connectivity radius is an elastic element whose rest length is taken from the
reference configuration.  This both discretises each part volumetrically and
glues the overlapping body parts into one continuum body.

Time integration
----------------
Semi-implicit (symplectic) Euler with small fixed substeps, spring-aligned
viscous damping, and exact kinematic boundary conditions:

* ``fixed_grasp`` nodes are held at their reference positions (the hands).
* ``pulled_patch`` nodes are rigidly attached to the temporary pull target,
  whose position ramps smoothly from the initial patch centroid through the
  configured ``target_start`` to ``target_peak`` over ``[pull_start_s,
  pull_peak_s]`` and is held until ``release_s``, after which the constraint
  is removed and the body recoils elastically under gravity while the hands
  stay fixed.

All bookkeeping (external impulse, constraint reactions, per-element
deformation gradients) is exported through the StateFrame contract.  The
integrator is deterministic: fixed substep size, fixed element ordering and
pure NumPy arithmetic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:  # SciPy is available in the platform environment; fall back if not.
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def _smoothstep(u: float) -> float:
    u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else u)
    return u * u * (3.0 - 2.0 * u)


class BenchmarkSolver:
    """Lumped-node elastic solid solver with kinematic grasp/pull constraints."""

    # Effective lattice parameters.  The scaffold material block is explicitly
    # scene metadata only; these values define the actual discretised solid.
    CONNECT_RADIUS = 0.30      # m, frozen element connectivity radius
    H_CHAR = 0.12              # m, characteristic element spacing
    YOUNG_EFF = 2.4e5          # Pa, effective lattice Young's modulus
    DAMPING_ZETA = 0.04        # per-element viscous damping ratio
    DT_SUB = 2.0e-4            # s, fixed integration substep
    VEL_CLAMP = 100.0          # m/s, finiteness safety clamp only

    def initialize(self, scene, seed: int) -> None:
        config = scene.config
        self.seed = int(seed)
        self.gravity = np.asarray(config.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        timeline = config.get("event_timeline", {}) or {}
        self.t_pull_start = float(timeline.get("pull_start_s", 0.0))
        self.t_pull_peak = float(timeline.get("pull_peak_s", 0.9))
        self.t_release = float(timeline.get("release_s", 1.0))

        pull = (config.get("boundary_conditions", {}) or {}).get("temporary_pull", {}) or {}
        self.pull_target_start = np.asarray(pull.get("target_start", [0.0, 0.0, 1.25]), dtype=np.float64)
        self.pull_target_peak = np.asarray(pull.get("target_peak", [0.0, 0.0, 3.65]), dtype=np.float64)

        material = config.get("material_reference", {}) or {}
        self.density = float(material.get("density_kg_m3", 1000.0))

        # ---- initial state from the frozen scene -------------------------
        frame0 = scene.initial_state()
        self.x0 = np.array(frame0.position, dtype=np.float64)
        self.n = self.x0.shape[0]
        self.x = self.x0.copy()
        self.v = np.zeros_like(self.x0)

        fixed = np.asarray(frame0.attributes.get("fixed_grasp", np.zeros(self.n)), dtype=np.float64)
        pulled = np.asarray(frame0.attributes.get("pulled_patch", np.zeros(self.n)), dtype=np.float64)
        self.fixed = fixed > 0.5
        self.pulled = pulled > 0.5

        # ---- volumetric topology (frozen asset) --------------------------
        # The frozen scene module ships the solid topology next to itself; fall
        # back to the scaffold assets directory relative to this solver file.
        candidates = [Path(__file__).resolve().parent.parent / "assets" / "armadillo_solid.npz"]
        module = sys.modules.get(type(scene).__module__)
        if module is not None and hasattr(module, "SOLID"):
            candidates.insert(0, Path(module.SOLID))
        solid = next((p for p in candidates if Path(p).is_file()), candidates[-1])
        data = np.load(solid)
        self.tets = np.asarray(data["tetrahedra"], dtype=np.int64)
        ref = np.asarray(data["points"], dtype=np.float64)
        if ref.shape != self.x0.shape:
            ref = self.x0
        # Per-element deformation-gradient bookkeeping (fixed element order).
        t0, t1, t2, t3 = self.tets[:, 0], self.tets[:, 1], self.tets[:, 2], self.tets[:, 3]
        dm = np.stack(
            [ref[t1] - ref[t3], ref[t2] - ref[t3], ref[t0] - ref[t3]], axis=2
        )  # columns: edge vectors
        self.dm_inv = np.linalg.inv(dm)
        a = ref[t1] - ref[t3]
        b = ref[t2] - ref[t3]
        c = ref[t0] - ref[t3]
        tet_vol = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0)
        total_volume = float(tet_vol.sum())
        self.total_mass = self.density * total_volume
        self.mass = np.full(self.n, self.total_mass / self.n, dtype=np.float64)

        # ---- elastic distance-element network ----------------------------
        pts = self.x0
        if cKDTree is not None:
            pairs = cKDTree(pts).query_pairs(self.CONNECT_RADIUS, output_type="ndarray")
        else:  # pragma: no cover - deterministic brute force fallback
            diff = pts[:, None, :] - pts[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            ii, jj = np.where(np.triu(dist < self.CONNECT_RADIUS, 1))
            pairs = np.stack([ii, jj], axis=1)
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        keep = np.ones(len(pairs), dtype=bool)
        if len(pairs):
            l0 = np.linalg.norm(pts[pairs[:, 0]] - pts[pairs[:, 1]], axis=1)
            keep = l0 > 1e-9
            pairs = pairs[keep]
            l0 = l0[keep]
        self.spr_i = np.ascontiguousarray(pairs[:, 0])
        self.spr_j = np.ascontiguousarray(pairs[:, 1])
        self.l0 = np.ascontiguousarray(l0)
        l_eff = np.maximum(self.l0, 0.6 * self.H_CHAR)
        self.k_spr = self.YOUNG_EFF * self.H_CHAR * self.H_CHAR / l_eff
        self.c_spr = self.DAMPING_ZETA * np.sqrt(2.0 * self.k_spr * (self.total_mass / self.n))

        # ---- pull constraint description ---------------------------------
        self.pull_offsets = self.x0[self.pulled] - self.x0[self.pulled].mean(axis=0)
        self.pull_c0 = self.x0[self.pulled].mean(axis=0)
        self.pull_mass = float(self.mass[self.pulled].sum())
        self.fixed_mass = float(self.mass[self.fixed].sum())

        # ---- budget accumulators -----------------------------------------
        self.time = 0.0
        self.cum_external_impulse = np.zeros(3, dtype=np.float64)
        self.last_reaction = np.zeros(3, dtype=np.float64)
        self.substeps_taken = 0
        self.reference_momentum = float(self.total_mass * np.sqrt(9.81 * 2.0))

    # ------------------------------------------------------------------ pull
    def _pull_center(self, t: float):
        """Prescribed pull-target position at time ``t`` (None when inactive)."""
        if t < self.t_pull_start or t >= self.t_release:
            return None
        t_mid = 0.5 * (self.t_pull_start + self.t_pull_peak)
        if t < t_mid:
            s = _smoothstep((t - self.t_pull_start) / max(t_mid - self.t_pull_start, 1e-12))
            return self.pull_c0 + s * (self.pull_target_start - self.pull_c0)
        if t < self.t_pull_peak:
            s = _smoothstep((t - t_mid) / max(self.t_pull_peak - t_mid, 1e-12))
            return self.pull_target_start + s * (self.pull_target_peak - self.pull_target_start)
        return self.pull_target_peak.copy()

    # ------------------------------------------------------------- dynamics
    def _internal_forces(self) -> np.ndarray:
        """Elastic + viscous element forces accumulated on nodes."""
        xi = self.x[self.spr_i]
        xj = self.x[self.spr_j]
        dx = xj - xi
        length = np.sqrt(np.einsum("ij,ij->i", dx, dx))
        np.maximum(length, 1e-9, out=length)
        direction = dx / length[:, None]
        stretch = length - self.l0
        vi = self.v[self.spr_i]
        vj = self.v[self.spr_j]
        vrel = np.einsum("ij,ij->i", vj - vi, direction)
        magnitude = self.k_spr * stretch + self.c_spr * vrel
        fvec = magnitude[:, None] * direction
        n = self.n
        acc = np.zeros((n, 3), dtype=np.float64)
        for coord in range(3):
            w = fvec[:, coord]
            acc[:, coord] = (
                np.bincount(self.spr_i, weights=w, minlength=n)
                - np.bincount(self.spr_j, weights=w, minlength=n)
            )
        return acc

    def _step(self, dt: float) -> None:
        t_end = self.time + dt

        # Semi-implicit Euler: forces at the current state, then update.
        f_total = self._internal_forces()
        f_total += self.mass[:, None] * self.gravity[None, :]
        v_new = self.v + dt * f_total / self.mass[:, None]

        # Finiteness safety only (never active in the resolved regime).
        speed = np.sqrt(np.einsum("ij,ij->i", v_new, v_new))
        scale = np.minimum(1.0, self.VEL_CLAMP / np.maximum(speed, 1e-12))
        v_new *= scale[:, None]
        if not np.isfinite(v_new).all():
            v_new = np.nan_to_num(v_new, nan=0.0, posinf=0.0, neginf=0.0)
        x_new = self.x + dt * v_new

        # ---- kinematic constraints (fixed grasps, temporary pull) --------
        reaction = np.zeros(3, dtype=np.float64)
        if self.fixed.any():
            impulse = self.mass[self.fixed, None] * (0.0 - v_new[self.fixed])
            reaction += impulse.sum(axis=0)
            x_new[self.fixed] = self.x0[self.fixed]
            v_new[self.fixed] = 0.0

        c_end = self._pull_center(min(t_end, self.t_release))
        if c_end is not None and self.time < self.t_release and self.pulled.any():
            c_cur = self._pull_center(self.time)
            if c_cur is None:
                c_cur = c_end
            v_pull = (c_end - c_cur) / dt
            impulse = self.mass[self.pulled, None] * (v_pull[None, :] - v_new[self.pulled])
            reaction += impulse.sum(axis=0)
            x_new[self.pulled] = c_end[None, :] + self.pull_offsets
            v_new[self.pulled] = v_pull[None, :]

        if not np.isfinite(x_new).all():
            x_new = np.where(np.isfinite(x_new), x_new, self.x)

        # ---- budget bookkeeping ------------------------------------------
        # External agents on the solid: gravity plus the kinematic constraint
        # reactions.  The cumulative impulse integrates exactly these terms.
        reaction_force = reaction / dt if dt > 0 else np.zeros(3)
        self.cum_external_impulse += (
            self.total_mass * self.gravity * dt + reaction
        )
        self.last_reaction = reaction_force

        self.x = x_new
        self.v = v_new
        self.time = t_end
        self.substeps_taken += 1

    # ---------------------------------------------------------------- frame
    def _element_deformation_gradients(self) -> np.ndarray:
        x = self.x
        t0, t1, t2, t3 = self.tets[:, 0], self.tets[:, 1], self.tets[:, 2], self.tets[:, 3]
        ds = np.stack(
            [x[t1] - x[t3], x[t2] - x[t3], x[t0] - x[t3]], axis=2
        )
        f = np.einsum("nij,njk->nik", ds, self.dm_inv)
        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)

    def _make_frame(self, time: float):
        from worldsolver.contracts import StateFrame

        external_force = (self.total_mass * self.gravity + self.last_reaction).reshape(1, 1, 3)
        fields = {
            "budget_cumulative_external_impulse": self.cum_external_impulse.reshape(1, 3).copy(),
            "budget_cumulative_inflow_momentum": np.zeros((1, 3)),
            "budget_cumulative_outflow_momentum": np.zeros((1, 3)),
            "budget_domain_weights": np.ones((1, self.n)),
            "budget_external_force": external_force,
            "cumulative_mass_source": np.zeros(1),
            "element_deformation_gradient": self._element_deformation_gradients(),
        }
        scalars = {
            "reference_momentum": self.reference_momentum,
            "total_mass_kg": float(self.total_mass),
        }
        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "fixed_grasp": (self.fixed.astype(np.float64)),
                "pulled_patch": (self.pulled.astype(np.float64)),
                "mass_quantity": self.mass.copy(),
            },
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # ------------------------------------------------------------------ API
    def advance_to(self, time: float):
        target = float(time)
        if target < self.time:
            target = self.time
        while target - self.time > 1e-9:
            remaining = target - self.time
            n_full = int(np.floor(remaining / self.DT_SUB + 1e-9))
            for _ in range(n_full):
                self._step(self.DT_SUB)
            last = target - self.time
            if last > 1e-9:
                self._step(last)
            else:
                self.time = target
        self.time = target
        return self._make_frame(target)

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "lumped-node elastic distance-element network, semi-implicit Euler",
            "nodes": int(self.n),
            "elastic_elements": int(len(self.spr_i)),
            "tetrahedra": int(len(self.tets)),
            "substep_dt_s": self.DT_SUB,
            "substeps_taken": int(self.substeps_taken),
            "effective_youngs_modulus_pa": self.YOUNG_EFF,
            "total_mass_kg": float(self.total_mass),
            "reference_momentum_kg_m_s": float(self.reference_momentum),
        }
