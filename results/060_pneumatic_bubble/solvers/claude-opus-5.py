"""Pneumatic bubble chamber: explicit finite-element inflation of a volumetric
rubber wall.

Physical model
--------------
The frozen chamber-wall asset is a closed triangle mesh describing a thick
axisymmetric shell (an outer surface and an inner surface sharing the same
ring structure).  The enclosed material is meshed here as a genuine
three-dimensional solid: every (axial x circumferential) cell of the wall is a
hexahedral patch spanning the wall thickness, split into six tetrahedra
(Kuhn decomposition).  No extra nodes are introduced, so the node ordering is
exactly the asset's vertex ordering and the frozen surface topology stays
valid for rendering.

Constitutive law: compressible neo-Hookean split into isochoric and
volumetric parts,

    W = mu/2 * (J^{-2/3} tr(F^T F) - 3) + kappa/2 * (J - 1)^2

with mu = E / (2(1+nu)), kappa = E / (3(1-2nu)).  Because nu = 0.499 the
volumetric term is treated with a mean-dilatation (Q1/P0) formulation: the
hydrostatic pressure p = kappa (Jbar - 1) is constant over each hexahedral
patch, with Jbar the reference-volume-weighted average of J over the six
tetrahedra of the patch.  This removes the volumetric locking that linear
tetrahedra otherwise suffer at near-incompressibility and lets the wall
actually bulge.

A Kelvin-Voigt viscous stress eta * Fdot models the rubber's internal
damping; its nodal forces sum to zero element-wise, so it is an internal
force.  A weak mass-proportional ambient drag is applied and reported
explicitly as an external force term so that the momentum budget closes.

Loading: prescribed internal gauge pressure ramped as specified in the
frozen config, applied as a follower traction on the current (deformed)
inner-surface triangles.  Base and rim attachment rings are clamped
(fixed vertex attribute of the frozen asset).

Time stepping: symplectic (semi-implicit) Euler with a substep size derived
from the dilatational CFL condition of the actual element geometry.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import sparse

from worldsolver.contracts import StateFrame

_G_VEC = np.array([0.0, 0.0, -9.81], dtype=np.float64)


def _cofactor(F: np.ndarray) -> np.ndarray:
    """Cofactor matrix C = J * F^{-T} for a stack of 3x3 matrices."""
    C = np.empty_like(F)
    C[:, 0, :] = np.cross(F[:, 1, :], F[:, 2, :])
    C[:, 1, :] = np.cross(F[:, 2, :], F[:, 0, :])
    C[:, 2, :] = np.cross(F[:, 0, :], F[:, 1, :])
    return C


class BenchmarkSolver:
    """Explicit FEM solver for pressure-driven inflation of the chamber wall."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        cfg = dict(scene.config)
        self.cfg = cfg
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.x0 = np.array(frame0.position, dtype=np.float64)
        self.n_nodes = int(self.x0.shape[0])
        self.faces = np.asarray(frame0.fields["surface_faces"], dtype=np.int32)
        fixed_attr = np.asarray(frame0.attributes["fixed"], dtype=np.float64)
        self.fixed_flag = fixed_attr.astype(np.float64)
        self.fixed = fixed_attr > 0.5
        self.free = ~self.fixed

        # ---- material (frozen reference values) -------------------------
        mat = cfg["material_reference"]
        self.rho = float(mat["density_kg_m3"])
        E = float(mat["youngs_modulus_pa"])
        nu = float(mat["poisson_ratio"])
        self.mu = E / (2.0 * (1.0 + nu))
        self.kappa = E / (3.0 * (1.0 - 2.0 * nu))

        # ---- loading ----------------------------------------------------
        load = cfg["loading"]
        self.p_max = float(load["pressure_pa"])
        self.ramp_start = float(load["ramp_start_s"])
        self.ramp_end = float(load["ramp_end_s"])
        self.body_id_value = int(load.get("chamber_body_id", 0))

        self.duration = float(cfg["duration"])
        self.sample_dt = float(cfg["sample_dt"])

        # ---- recover the structured ring layout of the frozen asset -----
        self._build_topology()

        # ---- reference element quantities -------------------------------
        tets = self.tets
        X = self.x0
        Dm = np.stack(
            (
                X[tets[:, 1]] - X[tets[:, 0]],
                X[tets[:, 2]] - X[tets[:, 0]],
                X[tets[:, 3]] - X[tets[:, 0]],
            ),
            axis=2,
        )
        det = np.linalg.det(Dm)
        # enforce positive orientation by swapping the last two local nodes
        flip = det < 0.0
        if np.any(flip):
            tets[flip, 2], tets[flip, 3] = tets[flip, 3].copy(), tets[flip, 2].copy()
            self.tets = tets
            Dm = np.stack(
                (
                    X[tets[:, 1]] - X[tets[:, 0]],
                    X[tets[:, 2]] - X[tets[:, 0]],
                    X[tets[:, 3]] - X[tets[:, 0]],
                ),
                axis=2,
            )
            det = np.linalg.det(Dm)
        self.Bm = np.linalg.inv(Dm)
        self.V0 = det / 6.0
        self.n_elem = int(tets.shape[0])

        # lumped nodal mass from the actual solid density
        self.mass = np.zeros(self.n_nodes, dtype=np.float64)
        np.add.at(self.mass, tets.ravel(), np.repeat(self.rho * self.V0 / 4.0, 4))
        self.mass = np.maximum(self.mass, 1e-9)
        self.total_mass = float(self.mass.sum())

        # scatter operator: element-local nodal forces -> global nodes
        cols = np.arange(4 * self.n_elem, dtype=np.int64)
        rows = tets.ravel().astype(np.int64)
        self.S_elem = sparse.csr_matrix(
            (np.ones(4 * self.n_elem), (rows, cols)),
            shape=(self.n_nodes, 4 * self.n_elem),
        )

        # patch (hexahedral) grouping for mean dilatation
        self.n_patch = self.n_elem // 6
        w = self.V0.reshape(self.n_patch, 6)
        self.patch_w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-30)

        # ---- pressure surface -------------------------------------------
        self._build_pressure_faces()

        # ---- damping / stepping -----------------------------------------
        # Kelvin-Voigt viscosity: strongly damps structural (shear/bending)
        # modes while leaving the fast dilatational modes nearly untouched.
        self.eta = 60.0
        # weak ambient drag (reported as an external force term)
        self.c_ambient = 5.0

        c_wave = math.sqrt((self.kappa + 4.0 * self.mu / 3.0) / self.rho)
        self.c_wave = c_wave
        h_min = self._min_altitude()
        self.h_min = h_min
        # dilatational CFL of the actual element geometry
        dt_cfl = 0.30 * h_min / c_wave
        # explicit stability limit of the Kelvin-Voigt (diffusion-like) term
        dt_visc = 0.30 * self.rho * h_min * h_min / max(2.0 * self.eta, 1e-12)
        # ambient drag relaxation limit
        dt_drag = 0.30 * 2.0 / max(self.c_ambient, 1e-12)
        self.dt_max = float(min(dt_cfl, dt_visc, dt_drag, self.sample_dt))

        # ---- contact with the static mounting base ----------------------
        base_geom = None
        try:
            for obj in scene_geometry_objects(scene):
                if obj.get("id") == "mounting_base":
                    base_geom = obj
                    break
        except Exception:
            base_geom = None
        self.base_top_z = 0.0
        self.base_radius = 0.082
        if base_geom is not None:
            g = base_geom.get("geometry", {})
            tr = base_geom.get("transform", {}).get("translation", [0.0, 0.0, 0.0])
            self.base_radius = float(g.get("radius", 0.082))
            self.base_top_z = float(tr[2]) + 0.5 * float(g.get("height", 0.012))
        self.k_contact = 5.0e4
        self.c_contact = 20.0

        # ---- frozen reference force scale -------------------------------
        area_ref = float(self._pressure_area(self.x0))
        self.reference_force = float(max(self.p_max * area_ref, 1e-6))

        # ---- dynamic state ----------------------------------------------
        v0 = np.asarray(
            cfg.get("initial_conditions", {}).get("velocity_m_s", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        self.x = self.x0.copy()
        self.v = np.tile(v0.reshape(1, 3), (self.n_nodes, 1))
        self.v[self.fixed] = 0.0
        self.t = 0.0
        self.n_steps = 0
        self._max_speed = 0.0

    # ------------------------------------------------------- mesh utilities
    def _build_topology(self) -> None:
        """Recover ordered inner/outer rings from the frozen vertex cloud."""
        X = self.x0
        r = np.hypot(X[:, 0], X[:, 1])
        theta = np.arctan2(X[:, 1], X[:, 0])
        zq = np.round(X[:, 2], 9)
        z_levels = np.unique(zq)
        nr = int(z_levels.size)

        outer_rings = []
        inner_rings = []
        for zl in z_levels:
            idx = np.flatnonzero(zq == zl)
            rl = r[idx]
            split = 0.5 * (rl.min() + rl.max())
            out_idx = idx[rl > split]
            in_idx = idx[rl <= split]
            out_idx = out_idx[np.argsort(theta[out_idx], kind="stable")]
            in_idx = in_idx[np.argsort(theta[in_idx], kind="stable")]
            outer_rings.append(out_idx)
            inner_rings.append(in_idx)

        n_theta = int(min(min(o.size for o in outer_rings),
                          min(i.size for i in inner_rings)))
        self.nr = nr
        self.n_theta = n_theta
        self.outer = np.array([o[:n_theta] for o in outer_rings], dtype=np.int64)
        self.inner = np.array([i[:n_theta] for i in inner_rings], dtype=np.int64)

        j = np.arange(nr - 1)
        i = np.arange(n_theta)
        JJ, II = np.meshgrid(j, i, indexing="ij")
        JJ = JJ.ravel()
        II = II.ravel()
        I1 = (II + 1) % n_theta

        # hexahedral patch corners: bottom face 0-1-2-3, top face 4-5-6-7
        p = np.empty((JJ.size, 8), dtype=np.int64)
        p[:, 0] = self.inner[JJ, II]
        p[:, 1] = self.outer[JJ, II]
        p[:, 2] = self.outer[JJ, I1]
        p[:, 3] = self.inner[JJ, I1]
        p[:, 4] = self.inner[JJ + 1, II]
        p[:, 5] = self.outer[JJ + 1, II]
        p[:, 6] = self.outer[JJ + 1, I1]
        p[:, 7] = self.inner[JJ + 1, I1]

        kuhn = np.array(
            [
                [0, 1, 2, 6],
                [0, 2, 3, 6],
                [0, 3, 7, 6],
                [0, 7, 4, 6],
                [0, 4, 5, 6],
                [0, 5, 1, 6],
            ],
            dtype=np.int64,
        )
        tets = p[:, kuhn]                      # (n_patch, 6, 4)
        self.tets = tets.reshape(-1, 4).astype(np.int64)

    def _build_pressure_faces(self) -> None:
        """Triangles of the inner (cavity-facing) surface, oriented outward."""
        nr, nt = self.nr, self.n_theta
        tris = []
        for j in range(nr - 1):
            for i in range(nt):
                k = (i + 1) % nt
                a = self.inner[j, i]
                b = self.inner[j, k]
                c = self.inner[j + 1, k]
                d = self.inner[j + 1, i]
                tris.append((a, b, c))
                tris.append((a, c, d))
        tri = np.asarray(tris, dtype=np.int64)

        X = self.x0
        n = np.cross(X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]])
        centroid = (X[tri[:, 0]] + X[tri[:, 1]] + X[tri[:, 2]]) / 3.0
        radial = centroid.copy()
        radial[:, 2] = 0.0
        nrm = np.linalg.norm(radial, axis=1, keepdims=True)
        radial = radial / np.maximum(nrm, 1e-12)
        # the traction must push the solid away from the cavity axis
        bad = np.einsum("ij,ij->i", n, radial) < 0.0
        if np.any(bad):
            tri[bad, 1], tri[bad, 2] = tri[bad, 2].copy(), tri[bad, 1].copy()
        self.p_tris = tri

        ntri = tri.shape[0]
        cols = np.arange(3 * ntri, dtype=np.int64)
        rows = tri.ravel().astype(np.int64)
        self.S_tri = sparse.csr_matrix(
            (np.full(3 * ntri, 1.0 / 3.0), (rows, cols)),
            shape=(self.n_nodes, 3 * ntri),
        )

    def _min_altitude(self) -> float:
        X = self.x0
        t = self.tets
        a, b, c, d = X[t[:, 0]], X[t[:, 1]], X[t[:, 2]], X[t[:, 3]]
        areas = np.stack(
            (
                0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1),
                0.5 * np.linalg.norm(np.cross(b - a, d - a), axis=1),
                0.5 * np.linalg.norm(np.cross(c - a, d - a), axis=1),
                0.5 * np.linalg.norm(np.cross(c - b, d - b), axis=1),
            ),
            axis=1,
        )
        a_max = np.maximum(areas.max(axis=1), 1e-14)
        alt = 3.0 * np.abs(self.V0) / a_max
        return float(max(alt.min(), 1e-6))

    def _pressure_area(self, x: np.ndarray) -> float:
        tri = self.p_tris
        n = np.cross(x[tri[:, 1]] - x[tri[:, 0]], x[tri[:, 2]] - x[tri[:, 0]])
        return 0.5 * float(np.linalg.norm(n, axis=1).sum())

    # ------------------------------------------------------------- physics
    def pressure_at(self, t: float) -> float:
        if t <= self.ramp_start:
            return 0.0
        if t >= self.ramp_end:
            return self.p_max
        span = max(self.ramp_end - self.ramp_start, 1e-12)
        s = (t - self.ramp_start) / span
        # smooth (C1) ramp: avoids an impulsive jerk at ramp start/end
        return self.p_max * (s * s * (3.0 - 2.0 * s))

    def _deformation(self, x: np.ndarray, v: np.ndarray):
        t = self.tets
        Ds = np.stack(
            (x[t[:, 1]] - x[t[:, 0]], x[t[:, 2]] - x[t[:, 0]], x[t[:, 3]] - x[t[:, 0]]),
            axis=2,
        )
        Dv = np.stack(
            (v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]], v[t[:, 3]] - v[t[:, 0]]),
            axis=2,
        )
        F = np.einsum("eij,ejk->eik", Ds, self.Bm)
        Fdot = np.einsum("eij,ejk->eik", Dv, self.Bm)
        return F, Fdot

    def _internal_force(self, F: np.ndarray, Fdot: np.ndarray) -> np.ndarray:
        C = _cofactor(F)
        J = np.einsum("ei,ei->e", F[:, 0, :], C[:, 0, :])
        Js = np.clip(J, 0.05, 20.0)

        # mean dilatation (Q1/P0): one pressure per hexahedral patch
        Jp = (Js.reshape(self.n_patch, 6) * self.patch_w).sum(axis=1)
        p_hyd = np.repeat(self.kappa * (Jp - 1.0), 6)

        I1 = np.einsum("eij,eij->e", F, F)
        Jm23 = Js ** (-2.0 / 3.0)

        P = self.mu * Jm23[:, None, None] * F
        P -= ((self.mu / 3.0) * I1 * Jm23 / Js)[:, None, None] * C
        P += p_hyd[:, None, None] * C
        P += self.eta * Fdot

        Gm = -self.V0[:, None, None] * np.einsum("eij,ekj->eik", P, self.Bm)
        fe = np.empty((self.n_elem, 4, 3), dtype=np.float64)
        fe[:, 1, :] = Gm[:, :, 0]
        fe[:, 2, :] = Gm[:, :, 1]
        fe[:, 3, :] = Gm[:, :, 2]
        fe[:, 0, :] = -(fe[:, 1, :] + fe[:, 2, :] + fe[:, 3, :])
        return self.S_elem @ fe.reshape(-1, 3)

    def _pressure_force(self, x: np.ndarray, p: float) -> np.ndarray:
        if p == 0.0:
            return np.zeros((self.n_nodes, 3), dtype=np.float64)
        tri = self.p_tris
        x0 = x[tri[:, 0]]
        an = 0.5 * np.cross(x[tri[:, 1]] - x0, x[tri[:, 2]] - x0)
        f_tri = p * an                                    # (ntri, 3)
        return self.S_tri @ np.repeat(f_tri, 3, axis=0)

    def _contact_force(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        f = np.zeros((self.n_nodes, 3), dtype=np.float64)
        gap = self.base_top_z - x[:, 2]
        r = np.hypot(x[:, 0], x[:, 1])
        act = (gap > 0.0) & (r <= self.base_radius)
        if np.any(act):
            fz = self.k_contact * gap[act] - self.c_contact * v[act, 2]
            f[act, 2] = np.maximum(fz, 0.0)
        return f

    def _all_forces(self, x: np.ndarray, v: np.ndarray, t: float):
        F, Fdot = self._deformation(x, v)
        f_int = self._internal_force(F, Fdot)
        f_press = self._pressure_force(x, self.pressure_at(t))
        f_grav = self.mass[:, None] * _G_VEC[None, :]
        f_damp = -(self.c_ambient * self.mass)[:, None] * v
        f_cont = self._contact_force(x, v)
        return F, f_int, f_press, f_grav, f_damp, f_cont

    def _step(self, dt: float) -> None:
        _, f_int, f_press, f_grav, f_damp, f_cont = self._all_forces(self.x, self.v, self.t)
        f = f_int + f_press + f_grav + f_damp + f_cont
        a = f / self.mass[:, None]
        a[self.fixed] = 0.0
        if not np.all(np.isfinite(a)):
            a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        self.v += dt * a
        self.v[self.fixed] = 0.0
        # safety clamp: physical speeds here are far below this bound
        sp = np.linalg.norm(self.v, axis=1)
        self._max_speed = max(self._max_speed, float(sp.max()))
        over = sp > 40.0
        if np.any(over):
            self.v[over] *= (40.0 / sp[over])[:, None]
        self.x += dt * self.v
        self.x[self.fixed] = self.x0[self.fixed]
        self.t += dt
        self.n_steps += 1

    # -------------------------------------------------------------- output
    def advance_to(self, time: float):
        target = float(time)
        if target > self.t + 1e-12:
            remaining = target - self.t
            n = int(math.ceil(remaining / self.dt_max))
            n = max(n, 1)
            dt = remaining / n
            for _ in range(n):
                self._step(dt)
            self.t = target
        return self._frame(target)

    def _frame(self, time: float) -> StateFrame:
        x = self.x
        v = self.v
        F, f_int, f_press, f_grav, f_damp, f_cont = self._all_forces(x, v, self.t)

        # constraint reaction at the clamped base and rim rings: the force
        # required to hold those nodes at zero acceleration.
        total = f_int + f_press + f_grav + f_damp + f_cont
        f_react = np.zeros_like(total)
        f_react[self.fixed] = -total[self.fixed]

        terms = np.stack(
            (
                f_press.sum(axis=0),
                f_grav.sum(axis=0),
                f_damp.sum(axis=0),
                f_cont.sum(axis=0),
                f_react.sum(axis=0),
            ),
            axis=0,
        )
        budget = terms.reshape(1, 5, 3)

        Fout = np.nan_to_num(np.ascontiguousarray(F, dtype=np.float64),
                             nan=1.0, posinf=1.0, neginf=1.0)
        xo = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        vo = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        radial = np.hypot(xo[:, 0], xo[:, 1])
        frame = StateFrame(
            time=float(time),
            position=xo.copy(),
            velocity=vo.copy(),
            mass=self.mass.copy(),
            body_id=np.full(self.n_nodes, self.body_id_value, dtype=np.int32),
            attributes={
                "mass_quantity": self.mass.copy(),
                "fixed": self.fixed_flag.copy(),
            },
            fields={
                "budget_external_force": np.nan_to_num(budget),
                "budget_domain_weights": np.ones((1, self.n_nodes), dtype=np.float64),
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
                "element_deformation_gradient": Fout,
                "gravity_acceleration": _G_VEC.reshape(1, 3).copy(),
                "surface_faces": self.faces,
            },
            scalars={
                "reference_force": float(self.reference_force),
                "internal_pressure_pa": float(self.pressure_at(self.t)),
                "max_radius_m": float(radial.max()),
                "total_mass_kg": float(self.total_mass),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "n_nodes": int(self.n_nodes),
            "n_elements": int(self.n_elem),
            "n_patches": int(self.n_patch),
            "substeps": int(self.n_steps),
            "dt_max": float(self.dt_max),
            "wave_speed_m_s": float(self.c_wave),
            "shear_modulus_pa": float(self.mu),
            "bulk_modulus_pa": float(self.kappa),
            "total_mass_kg": float(self.total_mass),
            "max_speed_m_s": float(self._max_speed),
            "reference_force_n": float(self.reference_force),
        }


def scene_geometry_objects(scene):
    """Best-effort access to the frozen scene-geometry object list."""
    geom = getattr(scene, "geometry", None)
    if isinstance(geom, dict):
        return geom.get("objects", [])
    objects = getattr(geom, "objects", None)
    if objects is not None:
        out = []
        for obj in objects:
            out.append(
                {
                    "id": getattr(obj, "id", None),
                    "geometry": getattr(getattr(obj, "geometry", None), "__dict__", {}),
                    "transform": getattr(getattr(obj, "transform", None), "__dict__", {}),
                }
            )
        return out
    return []
