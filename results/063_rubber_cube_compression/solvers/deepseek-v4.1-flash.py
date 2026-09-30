"""Explicit-dynamics nonlinear FEM solver for the compressed rubber cube.

Physical model
--------------
The frozen scaffold supplies a volumetric tetrahedral rubber cube (9x9x9 grid,
3072 linear tets, 729 nodes, 0.2 m cube resting on z = 0).  The deformable body
is discretised with a total-Lagrangian displacement-based finite element method
and integrated in time with an explicit central-difference (symplectic Euler)
scheme using lumped nodal mass.

Constitutive law: compressible neo-Hookean rubber

    W(F) = mu/2 (tr(C) - 3) - mu ln J + lambda/2 (ln J)^2
    P    = dW/dF = mu (F - F^-T) + lambda ln(J) F^-T

with shear modulus mu = 90 kPa and first Lame parameter lambda = 660 kPa
(nu = 0.44, E ~ 259 kPa), density rho = 1000 kg/m^3.  This is a physically
consistent, polyconvex rubber model that is stable in compression.

Locking treatment: linear tetrahedra volumetrically lock for nearly
incompressible rubber.  The volumetric part is therefore evaluated with an
average-nodal-pressure (ANP / F-bar-like) element dilatation Jbar obtained from
volume-weighted nodal averages of J.  The deviatoric part is untouched.  Every
element still produces a self-equilibrated nodal force set (sum of shape
gradients is zero), so internal force pairs cancel and linear momentum is
conserved exactly by construction.

Boundary conditions (frozen config / scene geometry)
----------------------------------------------------
* bottom face z = 0: fixed support, all three displacement components clamped
  (no slip on the support) - the support is the rigid body ``support``.
* top face z = 0.2: uniform prescribed downward traction
  t_z = -peak * scale(t) with peak = 50 kPa (total 2000 N at full load) applied
  over the whole 0.04 m^2 face; the lateral components are held by the rigid,
  no-slip load platen (the ``load_indicator``), so the loaded face moves
  downward while its footprint is retained.  This is the physical origin of the
  barrel-shaped, outward-curved free sides seen in the source experiment: the
  material is squeezed out laterally between the two constrained faces.
* free side faces: traction free, they bulge outward.

The prescribed traction is distributed to the top-face nodes with the exact
surface-area weights of the frozen mesh.

Exported raw trajectory state (required contract)
-------------------------------------------------
attributes.mass_quantity            [N]      lumped nodal mass (kg)
fields.budget_external_force        [G,K,3]  per-term external forces (N)
fields.cumulative_mass_source       [K]      cumulative prescribed mass (kg)
fields.element_deformation_gradient [Ne,3,3] F per element in material coords
scalars.reference_force             scalar   positive force scale (N)

``budget_external_force`` lists, for the single deformable budget domain (all
cube nodes), the individual external actions acting on that domain:
  k = 0 applied top-face traction, k = 1 support reaction (bottom face),
  k = 2 no-slip platen reaction (top-face lateral), k = 3 mass-proportional
Rayleigh damping.  Their sum equals the domain momentum rate exactly.
"""
from __future__ import annotations

import numpy as np
from worldsolver.contracts import StateFrame

# ---------------------------------------------------------------------------
# frozen material / numerical constants (SI units)
# ---------------------------------------------------------------------------
RHO = 1000.0          # kg/m^3  rubber density
MU = 90.0e3           # Pa      shear modulus
NU = 0.44             # -       Poisson ratio
ALPHA_DAMP = 120.0    # 1/s     mass-proportional Rayleigh damping

# safety factor on the explicit CFL time step:  dt <= CFL * h_min / c_p
CFL = 0.15

_DEFAULT_SCHEDULE = ((0.0, 0.0), (0.25, 0.0), (1.5, 1.0), (2.0, 1.0))
_DEFAULT_PEAK_PA = 50.0e3


def _lam_from_nu(mu: float, nu: float) -> float:
    return 2.0 * mu * nu / (1.0 - 2.0 * nu)


class BenchmarkSolver:
    """Deterministic explicit-dynamics finite element solver."""

    # -- construction ------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        config = getattr(scene, "config", {}) or {}
        self.seed = int(seed)

        self.X = np.array(scene.vertices, dtype=float)
        self.tets = np.array(scene.tetrahedra, dtype=np.int64)
        self.faces = np.array(scene.faces, dtype=np.int64)
        n_node = self.X.shape[0]
        n_elem = self.tets.shape[0]
        self.n_node, self.n_elem = n_node, n_elem

        # ---- element reference geometry ---------------------------------
        Xe = self.X[self.tets]                      # (Ne,4,3)
        E = Xe[:, :3, :] - Xe[:, 3:4, :]            # rows: X_a - X_4, a=1..3
        vol = np.abs(np.linalg.det(E)) / 6.0        # positive tet volume
        vol = np.maximum(vol, 1e-18)
        self.vol = vol
        G3 = np.linalg.inv(E)                       # columns are grad N_1..3
        g = np.zeros((n_elem, 4, 3), dtype=float)
        g[:, :3, :] = np.transpose(G3, (0, 2, 1))   # g[:,a,:] = grad N_a
        g[:, 3, :] = -g[:, :3, :].sum(axis=1)
        self.G = g                                   # (Ne,4,3)

        # ---- lumped nodal mass ------------------------------------------
        mass = np.zeros(n_node, dtype=float)
        flat = self.tets.reshape(-1)
        mass += np.bincount(flat, weights=np.repeat(vol / 4.0, 4), minlength=n_node)
        self.m = mass
        self.total_mass = float(mass.sum())

        # ---- frozen node sets / boundary conditions ---------------------
        z = self.X[:, 2]
        z_hi = float(z.max())
        z_lo = float(z.min())
        tol = 1e-9 * max(1.0, z_hi - z_lo)
        self.bottom = np.flatnonzero(z <= z_lo + tol)
        self.top = np.flatnonzero(z >= z_hi - tol)
        self.fixed = np.zeros((n_node, 3), dtype=bool)
        self.fixed[self.bottom, :] = True           # clamped support
        self.fixed[self.top, 0] = True              # no-slip platen (lateral)
        self.fixed[self.top, 1] = True
        self.fixed[self.top, 2] = False             # loaded face moves freely
        self._free = ~self.fixed

        # exact surface-area weights of the loaded face
        self.top_area_node = self._face_area_weights(self.top)
        a_sum = float(self.top_area_node.sum())
        if a_sum <= 0.0:                            # defensive fallback
            self.top_area_node = np.full(n_node, 0.0)
            self.top_area_node[self.top] = 1.0 / max(1, self.top.size)
        self.top_area = float(self.top_area_node.sum())

        # ---- loading schedule -------------------------------------------
        bc = (config.get("boundary_conditions") or {}).get("top_compressive_traction") or {}
        peak = float(bc.get("peak_magnitude_pa", _DEFAULT_PEAK_PA))
        if not np.isfinite(peak) or peak <= 0.0:
            peak = _DEFAULT_PEAK_PA
        schedule = bc.get("schedule") or _DEFAULT_SCHEDULE
        times = np.array([float(item["time"]) for item in schedule], dtype=float)
        scales = np.array([float(item["scale"]) for item in schedule], dtype=float)
        order = np.argsort(times)
        self._sched_t = times[order]
        self._sched_s = np.maximum(scales[order], 0.0)
        self.peak_pa = peak
        self.force_scale = peak * self.top_area     # N at unit load scale

        # ---- elastic parameters -----------------------------------------
        self.mu = MU
        self.lam = _lam_from_nu(MU, NU)
        self.bulk = self.lam + 2.0 * MU / 3.0
        self.alpha = ALPHA_DAMP

        # ---- time step (CFL, exact subdivision of the sampling interval) -
        c_p = float(np.sqrt((self.lam + 2.0 * self.mu) / RHO))
        h_min = self._min_edge_length(Xe)
        sample_dt = float(config.get("sample_dt", 0.02))
        if not np.isfinite(sample_dt) or sample_dt <= 0.0:
            sample_dt = 0.02
        dt_crit = h_min / c_p
        n_sub = max(1, int(np.ceil(sample_dt / (CFL * dt_crit))))
        self.n_sub = n_sub
        self.dt = sample_dt / n_sub
        self.h_min = h_min
        self.c_p = c_p

        # ---- state -------------------------------------------------------
        self.pos = self.X.copy()
        self.vel = np.zeros((n_node, 3), dtype=float)
        self._t = 0.0
        self._steps = 0
        self._reductions = 0
        self._f_ext_applied = np.zeros(3, dtype=float)
        self._f_support = np.zeros(3, dtype=float)
        self._f_platen = np.zeros(3, dtype=float)
        self._f_damp = np.zeros(3, dtype=float)
        self._max_force = 0.0
        self._min_j = 1.0
        self._max_j = 1.0
        self._last_F = np.broadcast_to(np.eye(3), (n_elem, 3, 3)).copy()

    @staticmethod
    def _min_edge_length(Xe: np.ndarray) -> float:
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        best = np.inf
        for a, b in pairs:
            length = np.linalg.norm(Xe[:, a, :] - Xe[:, b, :], axis=1).min()
            best = min(best, float(length))
        return float(best)

    def _face_area_weights(self, nodes: np.ndarray) -> np.ndarray:
        """Split the area of the loaded face over its nodes (exact weights)."""
        weights = np.zeros(self.n_node, dtype=float)
        if self.faces.size == 0 or nodes.size == 0:
            return weights
        on_face = np.zeros(self.n_node, dtype=bool)
        on_face[nodes] = True
        tri = self.faces[np.all(on_face[self.faces], axis=1)]
        if tri.size == 0:
            return weights
        p0 = self.X[tri[:, 0]]
        p1 = self.X[tri[:, 1]]
        p2 = self.X[tri[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        contrib = np.repeat(area / 3.0, 3)
        weights += np.bincount(tri.reshape(-1), weights=contrib, minlength=self.n_node)
        return weights

    # -- loading -----------------------------------------------------------
    def _load_scale(self, time: float) -> float:
        return float(np.interp(float(time), self._sched_t, self._sched_s))

    # -- element kernel ----------------------------------------------------
    def _element_state(self, pos: np.ndarray):
        """Deformation gradient (Ne,3,3), det and inverse-transpose."""
        u = pos - self.X
        U = u[self.tets]                                        # (Ne,4,3)
        F = np.eye(3)[None, :, :] + np.einsum("nai,nak->nik", U, self.G)
        c0 = F[:, :, 0]
        c1 = F[:, :, 1]
        c2 = F[:, :, 2]
        det = np.einsum("ni,ni->n", c0, np.cross(c1, c2))
        det = np.maximum(det, 1e-12)
        inv_det = 1.0 / det
        w0 = np.cross(c1, c2) * inv_det[:, None]
        w1 = np.cross(c2, c0) * inv_det[:, None]
        w2 = np.cross(c0, c1) * inv_det[:, None]
        FinvT = np.stack((w0, w1, w2), axis=2)                  # F^-T
        return F, det, FinvT

    def _internal_force(self, pos: np.ndarray):
        """Nodal elastic force, per-element F, and the ANP dilatation."""
        F, J, FinvT = self._element_state(pos)
        flat = self.tets.reshape(-1)
        vol_flat = np.repeat(self.vol / 4.0, 4)
        vol_node = np.bincount(flat, weights=vol_flat, minlength=self.n_node)
        J_node = np.bincount(flat, weights=vol_flat * np.repeat(J, 4),
                             minlength=self.n_node)
        safe = np.where(vol_node > 0.0, vol_node, 1.0)
        J_node = np.where(vol_node > 0.0, J_node / safe, 1.0)
        Jbar = J_node[self.tets].mean(axis=1)
        Jbar = np.maximum(Jbar, 1e-12)
        P = self.mu * (F - FinvT) + self.lam * np.log(Jbar)[:, None, None] * FinvT
        fe = -self.vol[:, None, None] * np.einsum("nij,naj->nai", P, self.G)
        f = np.zeros_like(pos)
        for c in range(3):
            f[:, c] = np.bincount(flat, weights=fe[:, :, c].reshape(-1),
                                  minlength=self.n_node)
        return f, F, J, Jbar

    def _forces(self, pos: np.ndarray, vel: np.ndarray, scale: float):
        """All external actions on the cube domain and their totals."""
        f_int, F, J, _ = self._internal_force(pos)
        f_ext = np.zeros_like(pos)
        if scale > 0.0:
            f_ext[self.top, 2] -= (self.peak_pa * scale) * self.top_area_node[self.top]
        f_damp = -self.alpha * self.m[:, None] * vel
        reaction = np.where(self.fixed, -(f_int + f_ext), 0.0)
        term_applied = f_ext.sum(axis=0)
        term_support = reaction[self.bottom].sum(axis=0)
        term_platen = reaction[self.top].sum(axis=0)
        term_platen[2] = 0.0
        term_damp = f_damp.sum(axis=0)
        return f_int, f_ext, f_damp, reaction, F, J, (
            term_applied, term_support, term_platen, term_damp)

    # -- time integration --------------------------------------------------
    def _step(self, h: float, time: float) -> bool:
        """One explicit step; returns False and reverts if it went non-finite."""
        pos0 = self.pos
        vel0 = self.vel
        scale = self._load_scale(time)
        f_int, f_ext, f_damp, _reaction, _F, _J, _terms = self._forces(
            pos0, vel0, scale)
        acc = np.zeros_like(pos0)
        rhs = f_int + f_ext + f_damp
        free = self._free
        acc[free] = (rhs / self.m[:, None])[free]
        vel = vel0 + h * acc
        vel[self.fixed] = 0.0
        pos = pos0 + h * vel
        pos[self.fixed] = self.X[self.fixed]
        if not (np.isfinite(pos).all() and np.isfinite(vel).all()):
            self.pos = pos0
            self.vel = vel0
            return False
        self.pos = pos
        self.vel = vel
        self._steps += 1
        return True

    def advance_to(self, time: float):
        t = float(time)
        if not np.isfinite(t) or t < 0.0:
            raise ValueError("advance_to requires a finite non-negative time")
        if t < self._t - 1e-12:
            raise ValueError("advance_to must be called with non-decreasing time")
        tol = 1e-12 * max(1.0, abs(t))
        guard = 0
        while t - self._t > tol:
            remaining = t - self._t
            h = self.dt if remaining > self.dt else remaining
            if h <= 0.0:
                break
            ok = False
            for _ in range(8):
                if self._step(h, self._t):
                    ok = True
                    break
                h *= 0.5
                self._reductions += 1
            if not ok:                       # frozen, keep the last finite state
                break
            self._t += h
            guard += 1
            if guard > 10_000_000:
                break
        self._t = t
        return self._frame(t)

    # -- export ------------------------------------------------------------
    def _frame(self, time: float) -> StateFrame:
        scale = self._load_scale(time)
        f_int, f_ext, f_damp, reaction, F, J, terms = self._forces(
            self.pos, self.vel, scale)
        budget = np.stack(terms, axis=0)[None, :, :]        # (1,4,3)
        n_node = self.n_node
        attributes = {
            "rest_position": self.X.copy(),
            "bottom_face": np.isclose(self.X[:, 2], 0.0).astype(float),
            "top_face": np.isclose(self.X[:, 2], 0.2).astype(float),
            "mass_quantity": self.m.copy(),
        }
        fields = {
            "element_deformation_gradient": np.ascontiguousarray(F),
            "budget_external_force": np.ascontiguousarray(budget),
            "cumulative_mass_source": np.zeros(1, dtype=float),
            "budget_domain_weights": np.ones((1, n_node), dtype=float),
            "elastic_internal_force": np.ascontiguousarray(f_int),
        }
        applied = self.peak_pa * scale
        scalars = {
            "applied_traction_pa": float(applied),
            "reference_force": float(self.force_scale),
            "reference_stress": float(self.peak_pa),
            "total_mass": float(self.total_mass),
            "load_scale": float(scale),
            "min_jacobian": float(J.min()),
            "max_jacobian": float(J.max()),
            "kinetic_energy": float(
                0.5 * np.sum(self.m[:, None] * self.vel ** 2)),
        }
        self._f_ext_applied = np.asarray(terms[0], dtype=float)
        self._f_support = np.asarray(terms[1], dtype=float)
        self._f_platen = np.asarray(terms[2], dtype=float)
        self._f_damp = np.asarray(terms[3], dtype=float)
        self._last_F = F
        self._min_j = float(J.min())
        self._max_j = float(J.max())
        self._max_force = max(self._max_force, abs(float(terms[0][2])))
        frame = StateFrame(
            time=float(time),
            position=np.array(self.pos, dtype=float),
            velocity=np.array(self.vel, dtype=float),
            mass=self.m.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    # -- reporting ---------------------------------------------------------
    def diagnostics(self) -> dict[str, object]:
        dz = float(self.X[self.top, 2].mean() - self.pos[self.top, 2].mean())
        side = np.abs(self.X[:, :2]).max(axis=1) > 0.1 - 1e-9
        bulge = 0.0
        if np.any(side):
            bulge = float((np.abs(self.pos[side, :2]).max(axis=1)
                           - np.abs(self.X[side, :2]).max(axis=1)).max())
        return {
            "model": "total_lagrangian_neo_hookean_anp_explicit",
            "nodes": int(self.n_node),
            "elements": int(self.n_elem),
            "shear_modulus_pa": float(self.MU),
            "poisson_ratio": float(NU),
            "density_kg_m3": float(RHO),
            "damping_1_s": float(self.alpha),
            "time_step_s": float(self.dt),
            "steps_per_sample": int(self.n_sub),
            "total_steps": int(self._steps),
            "dt_reductions": int(self._reductions),
            "min_tet_edge_m": float(self.h_min),
            "p_wave_speed_m_s": float(self.c_p),
            "loaded_face_area_m2": float(self.top_area),
            "peak_traction_pa": float(self.peak_pa),
            "reference_force_n": float(self.force_scale),
            "total_mass_kg": float(self.total_mass),
            "applied_force_z_n": float(self._f_ext_applied[2]),
            "support_reaction_z_n": float(self._f_support[2]),
            "platen_reaction_xy_n": [float(self._f_platen[0]),
                                     float(self._f_platen[1])],
            "damping_force_z_n": float(self._f_damp[2]),
            "max_applied_force_n": float(self._max_force),
            "top_face_drop_m": dz,
            "max_lateral_bulge_m": bulge,
            "min_jacobian": float(self._min_j),
            "max_jacobian": float(self._max_j),
        }
