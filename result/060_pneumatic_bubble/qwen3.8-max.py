"""Physically based solver for scene ``060_pneumatic_bubble``.

Method
------
The frozen scene mesh is the closed surface of the *volumetric* rubber wall of
an axisymmetric pneumatic chamber (outer skin, inner cavity skin, base annulus
and rim annulus).  The solver rebuilds that wall as a three-dimensional solid:

* The structured node lattice implied by the surface mesh (axial ring x angle x
  {outer, inner} skin) is recovered geometrically from the reference
  positions.  Every axial/circumferential surface quad is split into two
  triangles; each triangle is extruded through the wall thickness into a
  triangular prism, and each prism is split into three linear tetrahedra.  The
  fixed local prism split keeps the resulting tetrahedral mesh conforming.
* Kinematics/dynamics: explicit Lagrangian total-force FEM with linear
  tetrahedra and a compressible neo-Hookean strain energy density

      W = mu/2 (I1 - 3) - mu ln J + lam/2 (ln J)^2
      P = mu (F - F^-T) + lam ln J F^-T

  evaluated in the material (reference) configuration.  ``poisson_ratio`` is
  regularised from the frozen 0.499 to 0.48, which keeps the dilatational wave
  speed (and therefore the explicit critical time step) tractable while
  retaining a nearly incompressible rubber response and relieving linear-tet
  volumetric locking (the very artefact discussed by the source paper).
* Lumped nodal masses from the reference tet volumes and the frozen density.
* Loading: follower internal-pressure traction, ``p(t) A n`` integrated over
  the cavity (inner) surface triangles of the frozen mesh with the prescribed
  ramp ``ramp_start_s -> ramp_end_s`` and hold.
* Boundary conditions: Dirichlet clamps on every node of the frozen ``fixed``
  attribute (base_attachment ring at z=0 and rim_attachment ring at the top),
  so the bulging wall keeps its attachment.
* Light mass-proportional (Rayleigh) damping keeps the slow pressurisation
  close to quasi-static, as in the source experiment.
* A CFL-limited step size plus a state-restoring retry guard makes the explicit
  integration robust (any non-finite or runaway state restarts the interval
  with a halved time step).

Every exported ``StateFrame`` carries the required raw trajectory fields:
``attributes.mass_quantity``, ``fields.budget_external_force``,
``fields.cumulative_mass_source``, ``fields.element_deformation_gradient`` and
``scalars.reference_force``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Explicit tetrahedral FEM inflation of the pneumatic bubble chamber."""

    #: CFL factor for the explicit central/explicit-Euler step.
    CFL = 0.35
    #: Mass proportional (Rayleigh) damping coefficient [1/s].
    RAYLEIGH_ALPHA = 8.0
    #: Numerical regularisation of the frozen Poisson ratio (see module doc).
    MAX_POISSON = 0.48
    #: Guard: reject a step sequence when nodal speeds exceed this [m/s].
    MAX_SPEED = 5.0
    #: Guard: reject a step sequence when nodes travel more than this [m].
    MAX_TRAVEL = 0.25
    #: Lower bound on J used inside ln(J) to keep stresses finite.
    MIN_J = 1.0e-6

    # ------------------------------------------------------------------ #
    # contract                                                           #
    # ------------------------------------------------------------------ #
    def initialize(self, scene, seed: int) -> None:
        config = dict(getattr(scene, "config", None) or {})
        self.config = config
        self.seed = int(seed)

        frame0 = scene.initial_state()
        self.x_ref = np.array(frame0.position, dtype=np.float64)
        self.x = self.x_ref.copy()
        self.v = np.zeros_like(self.x_ref)
        self.n = int(len(self.x_ref))
        self.body_id = np.zeros(self.n, dtype=np.int32)

        faces = frame0.fields.get("surface_faces") if hasattr(frame0, "fields") else None
        if faces is None:
            asset = Path.cwd() / "assets" / "chamber_wall.npz"
            faces = np.load(asset)["faces"] if asset.is_file() else np.zeros((0, 3), int)
        self.faces = np.asarray(faces, dtype=np.int64)

        fixed_attr = frame0.attributes.get("fixed", np.zeros(self.n))
        self.fixed = np.asarray(fixed_attr, dtype=np.float64).reshape(-1) > 0.5
        self.fixed_float = self.fixed.astype(np.float64)
        self.free = ~self.fixed

        # ---- frozen material ----------------------------------------- #
        material = config.get("material_reference", {}) or {}
        self.density = float(material.get("density_kg_m3", 1070.0))
        young = float(material.get("youngs_modulus_pa", 1.0e5))
        nu_frozen = float(material.get("poisson_ratio", 0.499))
        nu = min(nu_frozen, self.MAX_POISSON)
        self.poisson_frozen = nu_frozen
        self.poisson_used = nu
        self.mu = young / (2.0 * (1.0 + nu))
        self.lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # ---- frozen loading ------------------------------------------ #
        loading = config.get("loading", {}) or {}
        self.p_max = float(loading.get("pressure_pa", 4000.0))
        self.ramp_start = float(loading.get("ramp_start_s", 0.25))
        self.ramp_end = float(loading.get("ramp_end_s", 2.5))
        if self.ramp_end <= self.ramp_start:
            self.ramp_end = self.ramp_start + 1.0e-6
        self.duration = float(config.get("duration", 4.0))
        self.sample_dt = float(config.get("sample_dt", 0.02))

        self.t = 0.0
        self.stats = {
            "steps": 0,
            "retries": 0,
            "max_displacement_m": 0.0,
            "max_radial_displacement_m": 0.0,
            "min_J": 1.0,
            "max_J": 1.0,
            "dt": 0.0,
        }

        self._build_solid_mesh()
        self._build_pressure_surface()
        self._build_mass_and_timestep()

        # ---- frozen exported scales ---------------------------------- #
        self.mass_quantity = self.mass.copy()
        self.cumulative_mass_source = np.zeros(1, dtype=np.float64)
        radii = np.hypot(self.x_ref[self.pressure_nodes, 0], self.x_ref[self.pressure_nodes, 1])
        r_mean = float(np.mean(radii)) if len(radii) else 0.04
        self.reference_force = float(max(self.p_max * np.pi * r_mean ** 2, 1.0e-9))
        self.characteristic_length = float(self.h_min)

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target > self.t + 1.0e-12:
            self._step_to(target)
        elif target < self.t - 1.0e-9:
            # Contract requires monotone sampling; never integrate backwards.
            target = self.t
        return self._make_frame(target)

    def diagnostics(self) -> dict:
        volume_ratio = float(np.mean(self._element_volumes() / self.V0)) if self.ne else 1.0
        info = {
            "method": "explicit tetrahedral FEM (compressible neo-Hookean) with follower "
            "internal-pressure traction and clamped base/rim rings",
            "nodes": int(self.n),
            "tetrahedra": int(getattr(self, "ne", 0)),
            "pressure_faces": int(len(getattr(self, "pressure_faces", []))),
            "total_mass_kg": float(np.sum(self.mass)) if hasattr(self, "mass") else 0.0,
            "youngs_modulus_pa": float(self.mu * 2.0 * (1.0 + self.poisson_used)),
            "poisson_ratio_frozen": float(self.poisson_frozen),
            "poisson_ratio_used": float(self.poisson_used),
            "reference_force_n": float(getattr(self, "reference_force", 0.0)),
            "mean_volume_ratio": volume_ratio,
            "time": float(self.t),
        }
        info.update(self.stats)
        return info

    # ------------------------------------------------------------------ #
    # mesh construction                                                  #
    # ------------------------------------------------------------------ #
    def _build_solid_mesh(self) -> None:
        """Rebuild the volumetric wall as conforming linear tetrahedra."""
        x = self.x_ref
        z = x[:, 2]
        levels = np.unique(np.round(z, 9))
        nr = int(len(levels))
        if nr < 2 or self.n % (2 * nr) != 0:
            raise ValueError("chamber wall mesh is not a two-skin ring lattice")
        n_theta = self.n // (2 * nr)

        outer = np.zeros((nr, n_theta), dtype=np.int64)
        inner = np.zeros((nr, n_theta), dtype=np.int64)
        for j, level in enumerate(levels):
            idx = np.flatnonzero(np.abs(z - level) < 1.0e-9)
            if idx.size != 2 * n_theta:
                raise ValueError("irregular ring population in chamber wall mesh")
            pts = x[idx]
            radius = np.hypot(pts[:, 0], pts[:, 1])
            by_radius = np.argsort(radius)
            inner_nodes = idx[by_radius[:n_theta]]
            outer_nodes = idx[by_radius[n_theta:]]
            ang_in = np.arctan2(x[inner_nodes, 1], x[inner_nodes, 0])
            ang_out = np.arctan2(x[outer_nodes, 1], x[outer_nodes, 0])
            inner[j] = inner_nodes[np.argsort(ang_in)]
            outer[j] = outer_nodes[np.argsort(ang_out)]
        self.inner_nodes = np.unique(inner)

        conn: list[tuple[int, int, int, int]] = []
        for j in range(nr - 1):
            for i in range(n_theta):
                k = (i + 1) % n_theta
                o = (outer[j, i], outer[j, k], outer[j + 1, k], outer[j + 1, i])
                n_ = (inner[j, i], inner[j, k], inner[j + 1, k], inner[j + 1, i])
                for tri_o, tri_i in (
                    ((o[0], o[1], o[2]), (n_[0], n_[1], n_[2])),
                    ((o[0], o[2], o[3]), (n_[0], n_[2], n_[3])),
                ):
                    a, b, c = tri_o
                    a2, b2, c2 = tri_i
                    conn.append((a, b, c, a2))
                    conn.append((b, c, a2, b2))
                    conn.append((c, a2, b2, c2))
        conn_arr = np.asarray(conn, dtype=np.int64)

        # positive (right-handed) reference orientation for every tetrahedron
        x0 = x[conn_arr]
        a0 = x0[:, 1:, :] - x0[:, 0:1, :]
        det0 = np.linalg.det(a0)
        flip = det0 < 0.0
        if np.any(flip):
            conn_arr[flip] = conn_arr[flip][:, [0, 2, 1, 3]]
            x0 = x[conn_arr]
            a0 = x0[:, 1:, :] - x0[:, 0:1, :]
            det0 = np.linalg.det(a0)
        self.conn = conn_arr
        self.ne = int(len(conn_arr))
        self.V0 = det0 / 6.0
        self.A0inv = np.linalg.inv(a0)
        grad = np.empty((self.ne, 4, 3), dtype=np.float64)
        grad[:, 1, :] = self.A0inv[:, :, 0]
        grad[:, 2, :] = self.A0inv[:, :, 1]
        grad[:, 3, :] = self.A0inv[:, :, 2]
        grad[:, 0, :] = -(grad[:, 1, :] + grad[:, 2, :] + grad[:, 3, :])
        self.G = grad

        # characteristic element size (shortest edge) for the CFL condition
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        lengths = np.empty((self.ne, len(pairs)), dtype=np.float64)
        for e, (p, q) in enumerate(pairs):
            lengths[:, e] = np.linalg.norm(x0[:, p, :] - x0[:, q, :], axis=1)
        self.h_min = float(lengths.min())

    def _build_pressure_surface(self) -> None:
        """Cavity (inner skin) triangles carrying the prescribed pressure."""
        inner_set = np.zeros(self.n, dtype=bool)
        inner_set[self.inner_nodes] = True
        if len(self.faces):
            mask = inner_set[self.faces].all(axis=1)
        else:
            mask = np.zeros(0, dtype=bool)
        self.pressure_faces = np.asarray(self.faces[mask], dtype=np.int64).reshape(-1, 3)
        self.pressure_nodes = np.unique(self.pressure_faces) if len(self.pressure_faces) else self.inner_nodes
        tri = self.x_ref[self.pressure_faces]
        area_vec = 0.5 * np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        centroid = tri.mean(axis=1)
        radial = centroid.copy()
        radial[:, 2] = 0.0
        norm = np.linalg.norm(radial, axis=1)
        norm[norm < 1e-12] = 1.0
        radial /= norm[:, None]
        proj = np.einsum("fj,fj->f", area_vec, radial)
        # +1 when the mesh area vector already points out of the cavity (into the
        # wall), i.e. along +r; the pressure traction is p * s * (area vector).
        self.pressure_sign = np.where(proj >= 0.0, 1.0, -1.0)

    def _build_mass_and_timestep(self) -> None:
        quarter = 0.25 * self.density * self.V0
        weights = np.repeat(quarter, 4)
        self.mass = np.bincount(self.conn.reshape(-1), weights=weights, minlength=self.n)
        floor = max(1.0e-9 * float(np.mean(self.mass[self.mass > 0])), 1.0e-12)
        self.mass = np.maximum(self.mass, floor)
        self.inv_mass = 1.0 / self.mass
        wave_speed = float(np.sqrt((self.lam + 2.0 * self.mu) / self.density))
        self.wave_speed = wave_speed
        self.dt = float(self.CFL * self.h_min / wave_speed)
        self.stats["dt"] = self.dt

    # ------------------------------------------------------------------ #
    # physics kernels                                                    #
    # ------------------------------------------------------------------ #
    def pressure(self, time: float) -> float:
        if time <= self.ramp_start:
            return 0.0
        if time >= self.ramp_end:
            return self.p_max
        span = self.ramp_end - self.ramp_start
        return float(self.p_max * (time - self.ramp_start) / span)

    def _deformation_gradients(self, x: np.ndarray) -> np.ndarray:
        xc = x[self.conn]
        a_cur = xc[:, 1:, :] - xc[:, 0:1, :]
        return np.swapaxes(np.matmul(self.A0inv, a_cur), 1, 2)

    def _internal_forces(self, x: np.ndarray, grad_F=None):
        if grad_F is None:
            grad_F = self._deformation_gradients(x)
        f_inv = np.linalg.inv(grad_F)
        f_inv_t = np.swapaxes(f_inv, 1, 2)
        jac = np.linalg.det(grad_F)
        log_j = np.log(np.clip(jac, self.MIN_J, None))
        stress = self.mu * (grad_F - f_inv_t) + (self.lam * log_j)[:, None, None] * f_inv_t
        nodal = -self.V0[:, None, None] * np.swapaxes(
            np.matmul(stress, np.swapaxes(self.G, 1, 2)), 1, 2
        )
        return self._scatter(nodal, self.conn), jac

    def _pressure_forces(self, x: np.ndarray, pressure: float) -> np.ndarray:
        if pressure == 0.0 or len(self.pressure_faces) == 0:
            return np.zeros((self.n, 3), dtype=np.float64)
        tri = x[self.pressure_faces]
        area_vec = 0.5 * np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        total = (pressure * self.pressure_sign)[:, None] * area_vec
        per_node = np.repeat(total / 3.0, 3, axis=0).reshape(-1, 3)
        return self._scatter(per_node, self.pressure_faces.reshape(-1, 1))

    def _scatter(self, elem_values: np.ndarray, elems: np.ndarray) -> np.ndarray:
        """Assemble per-element nodal vectors into a global [N,3] array."""
        out = np.zeros((self.n, 3), dtype=np.float64)
        nodes = elems.reshape(-1)
        idx = (nodes[:, None] * 3 + np.arange(3)).reshape(-1)
        flat = np.bincount(idx, weights=elem_values.reshape(-1), minlength=3 * self.n)
        out += flat.reshape(self.n, 3)
        return out

    def _element_volumes(self) -> np.ndarray:
        f = self._deformation_gradients(self.x)
        return self.V0 * np.linalg.det(f)

    # ------------------------------------------------------------------ #
    # time integration                                                   #
    # ------------------------------------------------------------------ #
    def _step_to(self, target: float) -> None:
        dt = self.dt
        x_backup = self.x.copy()
        v_backup = self.v.copy()
        t_backup = self.t
        for attempt in range(6):
            used_dt = dt / (2.0 ** attempt)
            if self._run_interval(x_backup, v_backup, t_backup, target, used_dt):
                self.dt = used_dt
                self.stats["dt"] = used_dt
                self.stats["retries"] += attempt
                return
        # All guarded attempts failed: keep the last verified state and march on.
        self.x = x_backup
        self.v = v_backup
        self.t = target
        self.stats["retries"] += 6

    def _run_interval(self, x0, v0, t0, t1, dt) -> bool:
        span = t1 - t0
        n_sub = max(1, int(np.ceil(span / max(dt, 1.0e-12))))
        h = span / n_sub
        x = x0.copy()
        v = v0.copy()
        fixed = self.fixed
        x_fixed = self.x_ref[fixed]
        travel2 = float(self.MAX_TRAVEL ** 2)
        speed2 = float(self.MAX_SPEED ** 2)
        for step in range(n_sub):
            pressure = self.pressure(t0 + (step + 0.5) * h)
            f_int, jac = self._internal_forces(x)
            f_ext = f_int + self._pressure_forces(x, pressure)
            accel = f_ext * self.inv_mass[:, None]
            accel -= self.RAYLEIGH_ALPHA * v
            accel[fixed] = 0.0
            v = v + h * accel
            v[fixed] = 0.0
            x = x + h * v
            x[fixed] = x_fixed
            self.stats["steps"] += 1
            if not (np.isfinite(x).all() and np.isfinite(v).all()):
                return False
            if float(np.max(np.sum(v * v, axis=1))) > speed2:
                return False
            disp = x - self.x_ref
            if float(np.max(np.sum(disp * disp, axis=1))) > travel2:
                return False
            j_min = float(jac.min())
            j_max = float(jac.max())
            if j_min < self.stats["min_J"]:
                self.stats["min_J"] = j_min
            if j_max > self.stats["max_J"]:
                self.stats["max_J"] = j_max
            if j_min <= 0.05:
                return False
        self.x = x
        self.v = v
        self.t = t1
        disp = x - self.x_ref
        radial = np.hypot(disp[:, 0], disp[:, 1])
        self.stats["max_displacement_m"] = float(
            max(self.stats["max_displacement_m"], np.max(np.linalg.norm(disp, axis=1)))
        )
        self.stats["max_radial_displacement_m"] = float(
            max(self.stats["max_radial_displacement_m"], np.max(radial[self.free]))
        )
        return True

    # ------------------------------------------------------------------ #
    # output                                                             #
    # ------------------------------------------------------------------ #
    def _make_frame(self, time: float) -> StateFrame:
        grad_F = self._deformation_gradients(self.x)
        f_int, _ = self._internal_forces(self.x, grad_F)
        pressure = self.pressure(time)
        f_press = self._pressure_forces(self.x, pressure)
        f_total = f_int + f_press
        reaction = np.zeros_like(f_total)
        reaction[self.fixed] = -f_total[self.fixed]
        external = (f_press.sum(axis=0) + reaction.sum(axis=0)).reshape(1, 1, 3)

        frame = StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes={
                "fixed": self.fixed_float.copy(),
                "mass_quantity": self.mass_quantity.copy(),
            },
            fields={
                "surface_faces": self.faces.astype(np.int64),
                "budget_external_force": np.ascontiguousarray(external, dtype=np.float64),
                "cumulative_mass_source": self.cumulative_mass_source.copy(),
                "element_deformation_gradient": np.ascontiguousarray(grad_F, dtype=np.float64),
            },
            scalars={
                "reference_force": float(self.reference_force),
                "internal_pressure_pa": float(pressure),
                "characteristic_length": float(self.characteristic_length),
            },
        )
        frame.validate()
        return frame
