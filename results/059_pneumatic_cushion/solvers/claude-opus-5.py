"""Pneumatic cushion inflation solver (self-implemented, NumPy only).

Physical model
--------------
The finite-thickness rubber wall is treated as a three-dimensional nearly
incompressible solid discretised with the frozen linear tetrahedral mesh in
``assets/cushion_solid.npz``.  The stored energy is a compressible neo-Hookean
model

    W = mu/2 (I1 - 3) - mu ln J  +  kappa/2 (ln J_nodal)^2

with mu = E / (2 (1+nu)) and kappa = E / (3 (1-2 nu)).  The isochoric part is
evaluated per element; the volumetric part is evaluated on volume-averaged
*nodal* volume ratios (the averaged nodal pressure tetrahedron).  That
averaging is a variationally consistent modification of the strain energy, so
the assembled nodal forces remain conservative while the volumetric locking
that otherwise freezes linear tetrahedra at nu = 0.499 is removed.  Without it
the single-element-thick panels could not bulge.

The prescribed uniform chamber pressure is applied as a follower load on the
inner chamber surface using *current* (deformed) triangle area vectors, so the
traction stays normal to the wall while it balloons.  The inner surface is
closed and consistently oriented, hence the discrete pressure resultant
vanishes identically and no spurious net force enters the momentum budget.

Time integration is explicit symplectic Euler with semi-implicit
mass-proportional damping (dynamic relaxation towards the quasi-static
inflation path), sub-stepped below the elastic CFL limit of the mesh.  Rigid
translation and rotation modes are projected out of the velocity field every
step and the centre of mass is pinned, which realises the
``remove_rigid_body_modes`` boundary condition without restraining face
inflation.  All wall panels share nodes along edges and corners in the frozen
mesh, so the chamber stays sealed and connected by construction.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame

_TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
_J_FLOOR = 5.0e-2          # regularisation floor for log(J)
_V_MAX = 50.0              # runaway guard on nodal speed [m/s]
_DAMPING = 25.0            # mass-proportional damping coefficient [1/s]
_CFL_SAFETY = 0.12
_MAX_SUBSTEPS = 1500


class BenchmarkSolver:
    """Explicit dynamic-relaxation finite element solver for the cushion."""

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        self.config = dict(scene.config)
        self.seed = int(seed)
        np.random.seed(self.seed % (2**32))  # determinism hygiene (no RNG used)

        init = scene.initial_state()
        self.x = np.array(init.position, dtype=np.float64)
        self.x0 = self.x.copy()
        self.v = np.zeros_like(self.x)
        self.n_nodes = int(self.x.shape[0])
        mass = np.array(init.mass, dtype=np.float64).reshape(self.n_nodes)
        self.mass = np.maximum(mass, 1.0e-12)
        self.total_mass = float(self.mass.sum())
        region = init.attributes.get("wall_region") if init.attributes else None
        if region is None:
            region = np.zeros(self.n_nodes, dtype=np.int32)
        self.wall_region = np.asarray(region).astype(np.int32).reshape(self.n_nodes)

        topo = self._load_topology(scene)
        self.tets = np.array(topo["tetrahedra"], dtype=np.int64)
        self.inner_faces = np.array(topo["inner_chamber_faces"], dtype=np.int64)
        self.outer_faces = np.array(topo["outer_faces"], dtype=np.int64)
        self.n_elems = int(self.tets.shape[0])

        mat = self.config["material_properties"]
        young = float(mat["young_modulus_pa"])
        poisson = float(mat["poisson_ratio"])
        self.density = float(mat["density_kg_m3"])
        self.mu = young / (2.0 * (1.0 + poisson))
        self.kappa = young / (3.0 * (1.0 - 2.0 * poisson))

        load = self.config["loading"]
        self.p_start = float(load["initial_internal_pressure_pa"])
        self.p_target = float(load["target_internal_pressure_pa"])
        self.p_ext = float(load["external_pressure_pa"])
        self.ramp_start = float(load["ramp_start_s"])
        self.ramp_end = float(load["ramp_end_s"])
        gravity = np.asarray(self.config.get("gravity_m_s2", (0.0, 0.0, 0.0)), dtype=np.float64)
        self.gravity = gravity.reshape(3)

        self._build_reference_shapes()
        self._orient_surfaces()
        self._pick_time_step()

        self.t = 0.0
        self.n_steps = 0
        self._last_good = (self.x.copy(), self.v.copy())
        self._reverts = 0

    # ------------------------------------------------------- reference set-up
    def _load_topology(self, scene) -> dict:
        candidates = []
        asset = getattr(scene, "_asset", None)
        if asset is not None:
            candidates.append(Path(asset))
        try:
            import inspect

            candidates.append(
                Path(inspect.getfile(type(scene))).resolve().parent
                / "assets"
                / "cushion_solid.npz"
            )
        except Exception:  # pragma: no cover - defensive
            pass
        here = Path(__file__).resolve().parent
        candidates.append(here.parent / "assets" / "cushion_solid.npz")
        candidates.append(Path.cwd() / "assets" / "cushion_solid.npz")
        for path in candidates:
            try:
                if path is not None and Path(path).is_file():
                    with np.load(Path(path)) as data:
                        return {key: np.array(data[key]) for key in data.files}
            except Exception:  # pragma: no cover - defensive
                continue
        raise FileNotFoundError("cushion_solid.npz topology asset not found")

    def _build_reference_shapes(self) -> None:
        tets = self.tets
        ref = self.x0
        edges = np.stack(
            (
                ref[tets[:, 1]] - ref[tets[:, 0]],
                ref[tets[:, 2]] - ref[tets[:, 0]],
                ref[tets[:, 3]] - ref[tets[:, 0]],
            ),
            axis=2,
        )
        det = np.linalg.det(edges)
        flip = det < 0.0
        if np.any(flip):
            # Relabel local nodes so every element has positive rest volume.
            # F = Ds @ Dm^-1 is the same affine map for any local ordering.
            swapped = tets.copy()
            swapped[flip, 2] = tets[flip, 3]
            swapped[flip, 3] = tets[flip, 2]
            self.tets = tets = swapped
            edges = np.stack(
                (
                    ref[tets[:, 1]] - ref[tets[:, 0]],
                    ref[tets[:, 2]] - ref[tets[:, 0]],
                    ref[tets[:, 3]] - ref[tets[:, 0]],
                ),
                axis=2,
            )
            det = np.linalg.det(edges)
        self.dm_inv = np.linalg.inv(edges)
        self.vol0 = det / 6.0
        self.idx4 = tets.reshape(-1)
        self.nodal_vol0 = np.bincount(
            self.idx4,
            weights=np.repeat(self.vol0, 4) * 0.25,
            minlength=self.n_nodes,
        )
        self.nodal_vol0 = np.maximum(self.nodal_vol0, 1.0e-15)

    def _orient_surfaces(self) -> None:
        center = np.asarray(
            self.config.get("geometry", {}).get("chamber_center_m", (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ).reshape(3)
        self.chamber_center = center
        # Inner chamber faces: outward-from-solid normal must point into the
        # chamber (towards its centre).
        self.inner_faces = self._orient(self.inner_faces, center, inward=True)
        self.outer_faces = self._orient(self.outer_faces, center, inward=False)
        self.inner_idx3 = self.inner_faces.reshape(-1)
        self.outer_idx3 = self.outer_faces.reshape(-1)
        area = np.linalg.norm(self._area_vectors(self.x0, self.inner_faces), axis=1)
        self.inner_area0 = float(area.sum())
        self.ref_force = float(max(abs(self.p_target) * self.inner_area0, 1.0e-9))

    def _orient(self, faces: np.ndarray, center: np.ndarray, inward: bool) -> np.ndarray:
        ref = self.x0
        p0, p1, p2 = ref[faces[:, 0]], ref[faces[:, 1]], ref[faces[:, 2]]
        normal = np.cross(p1 - p0, p2 - p0)
        radial = (p0 + p1 + p2) / 3.0 - center
        dot = np.einsum("fi,fi->f", normal, radial)
        wrong = dot > 0.0 if inward else dot < 0.0
        out = faces.copy()
        if np.any(wrong):
            out[wrong, 1] = faces[wrong, 2]
            out[wrong, 2] = faces[wrong, 1]
        return out

    @staticmethod
    def _area_vectors(x: np.ndarray, faces: np.ndarray) -> np.ndarray:
        p0, p1, p2 = x[faces[:, 0]], x[faces[:, 1]], x[faces[:, 2]]
        return 0.5 * np.cross(p1 - p0, p2 - p0)

    def _pick_time_step(self) -> None:
        ref = self.x0
        tets = self.tets
        a_max = np.zeros(self.n_elems)
        for i, j, k in _TET_FACES:
            e1 = ref[tets[:, j]] - ref[tets[:, i]]
            e2 = ref[tets[:, k]] - ref[tets[:, i]]
            a_max = np.maximum(a_max, 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1))
        altitude = 3.0 * self.vol0 / np.maximum(a_max, 1.0e-14)
        h_min = float(np.min(altitude))
        wave = math.sqrt((self.kappa + 4.0 * self.mu / 3.0) / max(self.density, 1.0e-9))
        dt_cfl = _CFL_SAFETY * h_min / wave
        self.sample_dt = float(self.config["sample_dt"])
        n_sub = int(math.ceil(self.sample_dt / max(dt_cfl, 1.0e-12)))
        self.n_sub = int(min(max(n_sub, 1), _MAX_SUBSTEPS))
        self.dt = self.sample_dt / self.n_sub
        self.h_min = h_min
        self.wave_speed = wave
        self.damping = _DAMPING

    # ------------------------------------------------------------- load ramp
    def _pressure(self, t: float) -> float:
        if t <= self.ramp_start:
            return self.p_start
        span = self.ramp_end - self.ramp_start
        if span <= 0.0 or t >= self.ramp_end:
            return self.p_target
        frac = (t - self.ramp_start) / span
        return self.p_start + (self.p_target - self.p_start) * frac

    # -------------------------------------------------------------- kinematics
    def _deformation_gradient(self, x: np.ndarray) -> np.ndarray:
        tets = self.tets
        ds = np.stack(
            (
                x[tets[:, 1]] - x[tets[:, 0]],
                x[tets[:, 2]] - x[tets[:, 0]],
                x[tets[:, 3]] - x[tets[:, 0]],
            ),
            axis=2,
        )
        return ds @ self.dm_inv

    def _scatter(self, idx: np.ndarray, vals: np.ndarray, out: np.ndarray) -> None:
        n = self.n_nodes
        for c in range(3):
            out[:, c] += np.bincount(idx, weights=vals[:, c], minlength=n)

    # ------------------------------------------------------------------ forces
    def _elastic_force(self, x: np.ndarray, f_out: np.ndarray) -> np.ndarray:
        """Assemble neo-Hookean nodal forces with averaged nodal pressure."""
        F = self._deformation_gradient(x)
        # Cofactor matrix: cof(F) = det(F) * F^-T, built from column cross products.
        c0 = F[:, :, 0]
        c1 = F[:, :, 1]
        c2 = F[:, :, 2]
        cof = np.stack((np.cross(c1, c2), np.cross(c2, c0), np.cross(c0, c1)), axis=2)
        J = np.einsum("ei,ei->e", c0, cof[:, :, 0])
        Jc = np.maximum(J, _J_FLOOR)

        # Volume-averaged nodal volume ratio (anti-locking).
        w = self.vol0 * 0.25
        num = np.bincount(
            self.idx4, weights=np.repeat(Jc * w, 4), minlength=self.n_nodes
        )
        den = np.bincount(self.idx4, weights=np.repeat(w, 4), minlength=self.n_nodes)
        J_nodal = np.maximum(num / np.maximum(den, 1.0e-18), _J_FLOOR)
        # dW_vol/dJ_nodal, pushed back to each element through the averaging weights.
        p_nodal = self.kappa * np.log(J_nodal) / J_nodal
        p_elem = 0.25 * p_nodal[self.tets].sum(axis=1)

        # First Piola-Kirchhoff stress: isochoric neo-Hookean + averaged volumetric.
        coef = (p_elem - self.mu / Jc)[:, None, None]
        P = self.mu * F + coef * cof

        # H = -V0 * P * Dm^-T holds forces on local nodes 1..3.
        H = np.einsum("e,eij,ekj->eik", -self.vol0, P, self.dm_inv)
        f_out[:] = 0.0
        f1 = H[:, :, 0]
        f2 = H[:, :, 1]
        f3 = H[:, :, 2]
        self._scatter(self.tets[:, 1], f1, f_out)
        self._scatter(self.tets[:, 2], f2, f_out)
        self._scatter(self.tets[:, 3], f3, f_out)
        self._scatter(self.tets[:, 0], -(f1 + f2 + f3), f_out)
        return F

    def _pressure_force(self, x: np.ndarray, pressure: float, f_out: np.ndarray):
        """Follower pressure traction on the sealed chamber and outer surfaces."""
        res_in = np.zeros(3)
        res_out = np.zeros(3)
        if pressure != 0.0:
            av = self._area_vectors(x, self.inner_faces)
            # Chamber gas traction on the solid is -p * n_out, n_out points
            # into the chamber, so the load pushes the wall outwards.
            nodal = (-pressure / 3.0) * av
            self._scatter(self.inner_faces[:, 0], nodal, f_out)
            self._scatter(self.inner_faces[:, 1], nodal, f_out)
            self._scatter(self.inner_faces[:, 2], nodal, f_out)
            res_in = 3.0 * nodal.sum(axis=0)
        if self.p_ext != 0.0:
            av = self._area_vectors(x, self.outer_faces)
            nodal = (-self.p_ext / 3.0) * av
            self._scatter(self.outer_faces[:, 0], nodal, f_out)
            self._scatter(self.outer_faces[:, 1], nodal, f_out)
            self._scatter(self.outer_faces[:, 2], nodal, f_out)
            res_out = 3.0 * nodal.sum(axis=0)
        return res_in, res_out

    def _total_force(self, x: np.ndarray, pressure: float):
        f = np.zeros_like(x)
        F = self._elastic_force(x, f)
        f_el = f.copy()
        res_in, res_out = self._pressure_force(x, pressure, f)
        f_grav = self.mass[:, None] * self.gravity[None, :]
        f += f_grav
        return f, F, res_in, res_out, f_grav.sum(axis=0), f_el

    # ---------------------------------------------------------- rigid modes
    def _project_rigid_modes(self) -> None:
        m = self.mass[:, None]
        self.v -= (m * self.v).sum(axis=0) / self.total_mass
        com = (m * self.x).sum(axis=0) / self.total_mass
        r = self.x - com
        ang_mom = (self.mass[:, None] * np.cross(r, self.v)).sum(axis=0)
        r2 = np.einsum("ni,ni->n", r, r)
        inertia = np.eye(3) * float((self.mass * r2).sum()) - np.einsum(
            "n,ni,nj->ij", self.mass, r, r
        )
        try:
            omega = np.linalg.solve(inertia + 1.0e-9 * np.eye(3), ang_mom)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            omega = np.zeros(3)
        if np.all(np.isfinite(omega)):
            self.v -= np.cross(np.broadcast_to(omega, r.shape), r)
        # Pin the chamber centre so the cushion stays framed by the camera.
        self.x += self.chamber_center - com

    # ------------------------------------------------------------- integration
    def _step(self, dt: float) -> None:
        pressure = self._pressure(self.t + dt)
        f, _, _, _, _, _ = self._total_force(self.x, pressure)
        a = f / self.mass[:, None]
        # Semi-implicit mass-proportional damping: unconditionally stable.
        self.v = (self.v + dt * a) / (1.0 + dt * self.damping)
        speed = np.linalg.norm(self.v, axis=1)
        hot = speed > _V_MAX
        if np.any(hot):
            self.v[hot] *= (_V_MAX / speed[hot])[:, None]
        self.x = self.x + dt * self.v
        self._project_rigid_modes()
        self.t += dt
        self.n_steps += 1

    def _advance_by(self, target: float) -> None:
        while self.t < target - 1.0e-12:
            dt = min(self.dt, target - self.t)
            saved = (self.x.copy(), self.v.copy(), self.t)
            self._step(dt)
            if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()):
                # Numerical guard: roll back and retry with a smaller step.
                self.x, self.v, self.t = saved[0], saved[1], saved[2]
                self.dt = max(self.dt * 0.5, 1.0e-7)
                self.v[:] = 0.0
                self._reverts += 1
                if self._reverts > 40:
                    break
                continue
            self._last_good = (self.x.copy(), self.v.copy())
        if not (np.isfinite(self.x).all() and np.isfinite(self.v).all()):
            self.x, self.v = self._last_good[0].copy(), self._last_good[1].copy()

    # ------------------------------------------------------------------ output
    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.t:
            self._advance_by(target)
        self.t = target
        pressure = self._pressure(target)
        _, F, res_in, res_out, res_grav, _ = self._total_force(self.x, pressure)

        # Budget domain 0 = whole cushion wall; terms are the resultant
        # external loads acting on it (chamber pressure, ambient pressure,
        # gravity, damping reaction from rigid-mode projection).
        damp_res = -(self.damping * self.mass[:, None] * self.v).sum(axis=0)
        budget = np.zeros((1, 4, 3), dtype=np.float64)
        budget[0, 0] = res_in
        budget[0, 1] = res_out
        budget[0, 2] = res_grav
        budget[0, 3] = damp_res

        frame = StateFrame(
            time=target,
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "wall_region": self.wall_region.copy(),
                "mass_quantity": self.mass.copy(),
            },
            fields={
                "element_deformation_gradient": np.ascontiguousarray(F, dtype=np.float64),
                "budget_external_force": budget,
                "cumulative_mass_source": np.zeros(1, dtype=np.float64),
            },
            scalars={
                "internal_pressure_pa": float(pressure),
                "external_pressure_pa": float(self.p_ext),
                "reference_force": float(self.ref_force),
            },
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        disp = np.linalg.norm(self.x - self.x0, axis=1)
        return {
            "method": "explicit dynamic relaxation, neo-Hookean tetrahedra, "
            "averaged nodal pressure",
            "nodes": self.n_nodes,
            "elements": self.n_elems,
            "inner_faces": int(self.inner_faces.shape[0]),
            "dt": float(self.dt),
            "substeps_per_sample": int(self.n_sub),
            "steps_taken": int(self.n_steps),
            "wave_speed_m_s": float(self.wave_speed),
            "min_element_altitude_m": float(self.h_min),
            "shear_modulus_pa": float(self.mu),
            "bulk_modulus_pa": float(self.kappa),
            "damping_1_s": float(self.damping),
            "max_displacement_m": float(disp.max()),
            "mean_displacement_m": float(disp.mean()),
            "time_s": float(self.t),
            "step_reductions": int(self._reverts),
        }
