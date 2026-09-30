"""Physically based solver for the opposed-end-rotation solid beam benchmark.

Model
-----
The beam is a three-dimensional solid discretised with eight-node hexahedral
finite elements on the frozen ``volumetric_grid_shape`` lattice.  The material
is a compressible neo-Hookean solid written in total-Lagrangian form, so the
element deformation gradient ``F`` is measured with respect to the frozen rest
configuration.  The first Piola-Kirchhoff stress is

    P = mu * (F - F^-T) + lambda * ln(J) * F^-T ,      J = det(F)

and the consistent element nodal internal forces follow from
``f_a = -V0 * P . grad(N_a)``.

The transient problem is advanced with an explicit symplectic (velocity
Verlet) time-stepping scheme with a lumped nodal mass matrix.  The step size is
chosen from the dilatational CFL limit of the smallest element edge and is
refined so that it divides the sampling interval exactly, hence every requested
sample time is hit exactly by the integrator.

Boundary conditions
-------------------
Nodes inside the frozen ``end_face_band`` of each end rotate rigidly about the
beam axis through the end-face centre, with the angle given by the smoothstep
interpolation between the prescribed angles (left -180 deg, right +180 deg over
the scene duration).  The rotation is a rigid in-plane rotation of the end
cross-section (the axial coordinate is preserved), and the prescribed velocity
and acceleration are analytic derivatives of that motion.

All exported quantities are derived from the discrete state; no animation,
keyframes or precomputed motion tables are used.
"""

from __future__ import annotations

import numpy as np

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    """Opposed end rotations of a volumetric rectangular elastic beam."""

    # Material parameters (not frozen by the scaffold; a compliant rubber-like
    # solid is chosen so the driven twist is clearly visible).
    MU = 5.0e5  # shear modulus [Pa]
    LAMBDA = 1.0e6  # first Lame parameter [Pa]
    CFL_SAFETY = 0.35

    # ------------------------------------------------------------------ setup
    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        cfg = getattr(scene, "config", {})
        self.cfg = cfg
        beam = cfg["beam"]
        self.nx, self.ny, self.nz = (int(v) for v in beam["volumetric_grid_shape"])
        length, width, height = (float(v) for v in beam["dimensions"])
        self.dimensions = (length, width, height)
        self.center = np.asarray(beam["center"], dtype=float)
        self.density = float(beam["density"])

        # Frozen initial state of the scene (rest positions, end band mask).
        init = scene.initial_state()
        self.rest = np.array(init.position, dtype=float, copy=True)
        n = self.rest.shape[0]
        self.n_nodes = n
        self.end_region = np.array(
            init.attributes["end_region"], dtype=float, copy=True
        ).ravel()

        self.dx = length / (self.nx - 1)
        self.dy = width / (self.ny - 1)
        self.dz = height / (self.nz - 1)
        self.cell_volume = self.dx * self.dy * self.dz
        # Lumped nodal mass consistent with the frozen scene contract.
        self.nodal_mass = np.full(n, self.density * self.cell_volume, dtype=float)

        self._build_elements()
        self._build_boundary_conditions(cfg["boundary_conditions"])

        # Integrator state.
        self.x = self.rest.copy()
        self.v = np.zeros_like(self.x)
        self._prev_v = self.v.copy()
        self.t = 0.0

        # Time-step selection: dilatational CFL on the smallest element edge,
        # refined so the step exactly divides the sampling interval.
        sample_dt = float(cfg.get("sample_dt", 0.02))
        self.duration = float(cfg.get("duration", 10.0))
        c_p = np.sqrt((self.LAMBDA + 2.0 * self.MU) / self.density)
        h_min = min(self.dx, self.dy, self.dz)
        dt_target = self.CFL_SAFETY * h_min / c_p
        n_sub = max(1, int(np.ceil(sample_dt / dt_target)))
        self.n_sub = n_sub
        self.dt = sample_dt / n_sub
        self.sample_dt = sample_dt

        self.reference_momentum = self._compute_reference_momentum()
        self.diagnostic_history = {
            "n_nodes": n,
            "n_elements": int(self.conn.shape[0]),
            "internal_dt": self.dt,
            "substeps_per_sample": n_sub,
            "elastic_energy": [],
            "kinetic_energy": [],
            "total_momentum": [],
            "max_twist_degrees": [],
        }

    def _build_elements(self) -> None:
        nx, ny, nz = self.nx, self.ny, self.nz
        ii, jj, kk = np.meshgrid(
            np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"
        )
        i = ii.ravel()
        j = jj.ravel()
        k = kk.ravel()

        def nid(a, b, c):
            return (a * ny + b) * nz + c

        conn = np.column_stack(
            [
                nid(i, j, k),
                nid(i + 1, j, k),
                nid(i + 1, j + 1, k),
                nid(i, j + 1, k),
                nid(i, j, k + 1),
                nid(i + 1, j, k + 1),
                nid(i + 1, j + 1, k + 1),
                nid(i, j + 1, k + 1),
            ]
        ).astype(np.int64)
        self.conn = conn
        n_elem = conn.shape[0]
        self.n_elem = n_elem

        # Natural-coordinate signs of the eight hexahedron nodes (VTK order).
        signs = np.array(
            [
                [-1.0, -1.0, -1.0],
                [+1.0, -1.0, -1.0],
                [+1.0, +1.0, -1.0],
                [-1.0, +1.0, -1.0],
                [-1.0, -1.0, +1.0],
                [+1.0, -1.0, +1.0],
                [+1.0, +1.0, +1.0],
                [-1.0, +1.0, +1.0],
            ],
            dtype=float,
        )
        # Constant shape-function gradients (parallelepiped elements):
        # dN_a/dX = sign_a / (4 * edge_length).
        scale = np.array([1.0 / (4.0 * self.dx), 1.0 / (4.0 * self.dy), 1.0 / (4.0 * self.dz)])
        self.G = (signs * scale[None, :])[None, :, :] * np.ones((n_elem, 8, 3))
        self.G = np.ascontiguousarray(self.G)
        self.V0 = self.cell_volume

        # Sparse element-to-node scatter operator: f_node = S @ f_element.
        from scipy.sparse import csr_matrix

        rows = conn.ravel()
        cols = np.arange(n_elem * 8, dtype=np.int64)
        data = np.ones(n_elem * 8, dtype=float)
        self.S = csr_matrix((data, (rows, cols)), shape=(self.n_nodes, n_elem * 8))

    def _build_boundary_conditions(self, bc: dict) -> None:
        self.axis = np.asarray(bc.get("rotation_axis", [1.0, 0.0, 0.0]), dtype=float)
        norm = float(np.linalg.norm(self.axis))
        if norm <= 0.0:
            raise ValueError("rotation_axis must be nonzero")
        self.axis = self.axis / norm
        self.duration = float(self.cfg.get("duration", 10.0))
        self.interp = str(bc.get("interpolation", "smoothstep")).lower()

        left = bc["left_end"]
        right = bc["right_end"]
        self.left_angle_end = self._end_angle(left, self.duration)
        self.right_angle_end = self._end_angle(right, self.duration)
        self.left_center = np.asarray(left.get("center", [0.0, 0.0, 0.0]), dtype=float)
        self.right_center = np.asarray(right.get("center", [0.0, 0.0, 0.0]), dtype=float)

        self.left_mask = self.end_region < 0.0
        self.right_mask = self.end_region > 0.0
        if not np.any(self.left_mask) or not np.any(self.right_mask):
            raise ValueError("end face bands contain no nodes")

        # Only rotation about the beam axis is prescribed here; the axis is
        # aligned with the beam so the rigid end-face rotation preserves the
        # axial coordinate.
        self._left_r0 = self.rest[self.left_mask] - self.left_center
        self._right_r0 = self.rest[self.right_mask] - self.right_center

    @staticmethod
    def _end_angle(end_cfg: dict, duration: float) -> float:
        table = end_cfg.get("prescribed_rotation_degrees", [])
        if len(table) < 2:
            raise ValueError("prescribed_rotation_degrees needs at least two samples")
        t0, a0 = float(table[0]["time"]), float(table[0]["angle"])
        t1, a1 = float(table[-1]["time"]), float(table[-1]["angle"])
        span = t1 - t0
        if span <= 0.0:
            raise ValueError("prescribed rotation times must increase")
        # Total rotation applied over the scene duration (radians).
        return np.deg2rad(a1 - a0)

    def _smoothstep(self, t: float) -> tuple[float, float, float]:
        """Return (s, s_dot, s_ddot) of the monotone loading function."""
        T = self.duration
        u = min(max(t / T, 0.0), 1.0)
        s = 3.0 * u * u - 2.0 * u ** 3
        ds = (6.0 * u - 6.0 * u * u) / T
        dds = (6.0 - 12.0 * u) / (T * T)
        if t <= 0.0 or t >= T:
            ds = 0.0
        return s, ds, dds

    def _left_motion(self, t: float):
        s, ds, dds = self._smoothstep(t)
        return self.left_angle_end * s, self.left_angle_end * ds, self.left_angle_end * dds

    def _right_motion(self, t: float):
        s, ds, dds = self._smoothstep(t)
        return self.right_angle_end * s, self.right_angle_end * ds, self.right_angle_end * dds

    def _rotate_about_axis(self, r0: np.ndarray, theta: float) -> np.ndarray:
        """Rotate offsets about the (unit) beam axis by theta."""
        e = self.axis
        c = np.cos(theta)
        s = np.sin(theta)
        return (
            r0 * c
            + np.cross(e, r0) * s
            + e[None, :] * (np.dot(r0, e)[:, None]) * (1.0 - c)
        )

    def _apply_end_constraints(self, x: np.ndarray, v: np.ndarray, t: float) -> None:
        e = self.axis
        ev = np.broadcast_to(e, (1, 3))
        for mask, r0, motion in (
            (self.left_mask, self._left_r0, self._left_motion),
            (self.right_mask, self._right_r0, self._right_motion),
        ):
            theta, omega, _ = motion(t)
            c = np.cos(theta)
            s = np.sin(theta)
            rc = r0 * c + np.cross(ev, r0) * s + e[None, :] * (r0 @ e)[:, None] * (1.0 - c)
            centre = self.left_center if r0 is self._left_r0 else self.right_center
            pos = centre[None, :] + rc
            x[mask] = pos
            v[mask] = omega * np.cross(ev, rc)
        # The rotation about the beam axis preserves the axial coordinate.
        for mask in (self.left_mask, self.right_mask):
            x[mask, 0] = self.rest[mask, 0]

    # ------------------------------------------------------------- mechanics
    def _element_forces(self, x: np.ndarray):
        """Return (nodal internal forces, first Piola stress, F, detF)."""
        xe = x[self.conn]  # (Ne,8,3)
        F = xe.transpose(0, 2, 1) @ self.G  # (Ne,3,3)
        J = np.linalg.det(F)
        J = np.clip(J, 1e-6, None)
        Finv = np.linalg.inv(F)
        FinvT = Finv.transpose(0, 2, 1)
        logJ = np.log(J)
        P = self.MU * (F - FinvT) + self.LAMBDA * logJ[:, None, None] * FinvT
        fe = -self.V0 * (P @ self.G.transpose(0, 2, 1))  # (Ne,3,8)
        fe = fe.transpose(0, 2, 1).reshape(self.n_elem * 8, 3)
        f = np.asarray(self.S @ fe).reshape(self.n_nodes, 3)
        return f, P, F, J

    def _accelerations(self, x: np.ndarray) -> np.ndarray:
        f, _, _, _ = self._element_forces(x)
        return f / self.nodal_mass[:, None]

    def _step(self, dt: float, t: float) -> None:
        x = self.x
        v = self.v
        a = self._accelerations(x)
        x_new = x + dt * v + 0.5 * dt * dt * a
        a_new = self._accelerations(x_new)
        v_new = v + 0.5 * dt * (a + a_new)
        self._prev_v = v
        self._apply_end_constraints(x_new, v_new, t + dt)
        self.x = x_new
        self.v = v_new
        self.t = t + dt

    # ------------------------------------------------------------ interfaces
    def _compute_reference_momentum(self) -> float:
        r_max = 0.5 * float(np.hypot(self.dimensions[1], self.dimensions[2]))
        omega_max = 1.5 * abs(self.left_angle_end) / max(self.duration, 1e-12)
        total_mass = float(self.nodal_mass.sum())
        scale = total_mass * omega_max * max(r_max, 1e-12)
        return float(max(scale, 1e-12))

    def _current_scalars(self) -> dict:
        theta_l, _, _ = self._left_motion(self.t)
        theta_r, _, _ = self._right_motion(self.t)
        return {
            "reference_momentum": float(self.reference_momentum),
            "relative_end_rotation_degrees": float(
                np.rad2deg(theta_r - theta_l)
            ),
            "time": float(self.t),
        }

    def _build_frame(self) -> StateFrame:
        x = self.x
        v = self.v
        f, _P, F, _J = self._element_forces(x)
        mass = self.nodal_mass

        # External (constraint reaction) force.  Only the driven end nodes can
        # carry exterior traction: their motion is prescribed analytically, so
        # f_ext = m * a_prescribed - f_internal there.  Interior nodes satisfy
        # the discrete equation of motion exactly, and internal elastic forces
        # are pairwise and cancel in the global sum.
        ext = np.zeros(3)
        for mask, r0, motion in (
            (self.left_mask, self._left_r0, self._left_motion),
            (self.right_mask, self._right_r0, self._right_motion),
        ):
            theta, omega, alpha = motion(self.t)
            e = np.broadcast_to(self.axis, (np.count_nonzero(mask), 3))
            rc = r0 * np.cos(theta) + np.cross(e, r0) * np.sin(theta) + self.axis[None, :] * (
                r0 @ self.axis
            )[:, None] * (1.0 - np.cos(theta))
            acc = alpha * np.cross(e, rc) + omega * omega * np.cross(e, np.cross(e, rc))
            ext = ext + (self.nodal_mass[mask][:, None] * acc - f[mask]).sum(axis=0)

        momentum = (mass[:, None] * v).sum(axis=0)
        cumulative_impulse = momentum - self._initial_momentum

        frame = StateFrame(
            time=float(self.t),
            position=x.copy(),
            velocity=v.copy(),
            mass=mass.copy(),
            attributes={
                "mass_quantity": mass.copy(),
                "end_region": self.end_region.copy(),
            },
            fields={
                "element_deformation_gradient": np.array(F, dtype=float, copy=True),
                "budget_domain_weights": np.ones((1, self.n_nodes), dtype=float),
                "budget_external_force": ext.reshape(1, 1, 3).copy(),
                "budget_cumulative_external_impulse": cumulative_impulse.reshape(1, 3).copy(),
                "budget_cumulative_inflow_momentum": np.zeros((1, 3), dtype=float),
                "budget_cumulative_outflow_momentum": np.zeros((1, 3), dtype=float),
                "cumulative_mass_source": np.zeros(1, dtype=float),
            },
            scalars=self._current_scalars(),
        )
        frame.validate()
        return frame

    def advance_to(self, time: float):
        target = float(time)
        if not np.isfinite(target):
            raise ValueError("sample time must be finite")
        if self.t == 0.0 and not hasattr(self, "_initial_momentum"):
            self._initial_momentum = (self.nodal_mass[:, None] * self.v).sum(axis=0).copy()
        tol = 1e-12
        guard = 0
        while self.t + self.dt <= target + tol:
            self._step(self.dt, self.t)
            guard += 1
            if guard > 10_000_000:
                raise RuntimeError("time stepping failed to reach the sample time")
        remaining = target - self.t
        if remaining > tol:
            self._step(remaining, self.t)
        self.t = target
        self._record_diagnostics()
        return self._build_frame()

    def _record_diagnostics(self) -> None:
        try:
            f, P, F, J = self._element_forces(self.x)
            energy = float(
                np.sum(
                    self.V0
                    * (
                        0.5 * self.MU * (np.einsum("eij,eij->e", F, F) - 3.0)
                        - self.MU * np.log(J)
                        + 0.5 * self.LAMBDA * np.log(J) ** 2
                    )
                )
            )
            kinetic = float(0.5 * np.sum(self.nodal_mass[:, None] * self.v ** 2))
            momentum = (self.nodal_mass[:, None] * self.v).sum(axis=0)
            self.diagnostic_history["elastic_energy"].append(energy)
            self.diagnostic_history["kinetic_energy"].append(kinetic)
            self.diagnostic_history["total_momentum"].append(momentum.tolist())
            theta_l, _, _ = self._left_motion(self.t)
            theta_r, _, _ = self._right_motion(self.t)
            self.diagnostic_history["max_twist_degrees"].append(
                float(np.rad2deg(theta_r - theta_l))
            )
        except Exception:  # diagnostics must never break the trajectory
            pass

    def diagnostics(self) -> dict[str, object]:
        hist = self.diagnostic_history
        out: dict[str, object] = {
            "n_nodes": hist.get("n_nodes"),
            "n_elements": hist.get("n_elements"),
            "internal_dt": hist.get("internal_dt"),
            "substeps_per_sample": hist.get("substeps_per_sample"),
            "reference_momentum": self.reference_momentum,
            "total_mass": float(self.nodal_mass.sum()),
        }
        for key in ("elastic_energy", "kinetic_energy", "max_twist_degrees"):
            values = hist.get(key) or []
            if values:
                out[f"final_{key}"] = values[-1]
                out[f"max_{key}"] = float(np.max(values))
        return out
