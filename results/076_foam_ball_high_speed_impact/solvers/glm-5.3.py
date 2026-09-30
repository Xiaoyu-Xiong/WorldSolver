"""Explicit dynamic FEM solver for the high-speed foam-ball impact benchmark.

Physical model
--------------
The volumetric elastic practice ball is discretized with the frozen sample
set provided by ``BenchmarkScene.initial_state()`` (a regular lattice of
515 material points spanning the sphere of radius 0.02 m).  Complete grid
cells become 8-node hexahedral finite elements with a corotated linear
elastic material (E = 1e7 Pa, nu = 0.45).  Mean-dilatation (F-bar)
technology removes the volumetric locking of fully integrated Q1 elements
for the nearly incompressible foam.  The few lattice points that belong to
no complete cell (extreme surface points) are tied to their meshed
neighbours with distance springs of continuum-matched stiffness.

Time integration is explicit symplectic Euler (central-difference family)
with a CFL-safe step of 4e-6 s, one fifth of the dilatational wave transit
time of a cell.  The fixed steel plate is a rigid planar constraint at
x = 0: penetrating nodes are projected back onto the surface and their
normal velocity is removed (frictionless, persistent contact).  Contact
impulses are accumulated exactly and reported in the momentum-budget
fields.
"""
from __future__ import annotations

import numpy as np

SQRT3 = 1.7320508075688772


def _hex_shape_derivatives(h: float):
    """dN/dX at the 2x2x2 Gauss points and at the element centre.

    Node ordering matches the connectivity builder: index a runs over
    (di, dj, dk) in {0,1}^3 with xi_a = 2*d - 1, so the map is identical
    for every (axis-aligned, uniform) cell.
    """
    corners = np.array(
        [[di, dj, dk] for di in (0, 1) for dj in (0, 1) for dk in (0, 1)],
        dtype=np.float64,
    )
    xi = 2.0 * corners - 1.0  # (8,3) local corner coordinates
    # Gauss points, same ordering as the second einsum axis
    g = np.array(
        [[a, b, c] for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)],
        dtype=np.float64,
    ) / SQRT3
    points = np.concatenate([g, np.zeros((1, 3))], axis=0)  # last = centre

    dn = np.empty((len(points), 8, 3), dtype=np.float64)
    for p, (gx, gy, gz) in enumerate(points):
        # trilinear shape function gradients in natural coordinates
        dn[p, :, 0] = xi[:, 0] / 8.0 * (1.0 + xi[:, 1] * gy) * (1.0 + xi[:, 2] * gz)
        dn[p, :, 1] = xi[:, 1] / 8.0 * (1.0 + xi[:, 0] * gx) * (1.0 + xi[:, 2] * gz)
        dn[p, :, 2] = xi[:, 2] / 8.0 * (1.0 + xi[:, 0] * gx) * (1.0 + xi[:, 1] * gy)
    dn *= 2.0 / h  # chain rule: X = X_c + (h/2) * xi
    return dn[:-1], dn[-1], corners.astype(np.int64)


class BenchmarkSolver:
    """Corotated hex-FEM dynamics with rigid-plane contact."""

    def __init__(self) -> None:
        self._initialized = False

    # ------------------------------------------------------------------
    # initialization
    # ------------------------------------------------------------------
    def initialize(self, scene, seed: int) -> None:
        cfg = scene.config
        ball = cfg["foam_ball"]

        frame = scene.initial_state()
        self.x = np.asarray(frame.position, dtype=np.float64).copy()
        self.v = np.asarray(frame.velocity, dtype=np.float64).copy()
        self.mass = np.asarray(frame.mass, dtype=np.float64).copy()
        self.rest = self.x.copy()
        self.n = len(self.x)
        self.mass_safe = np.maximum(self.mass, 1e-12)
        self.mass_total = float(self.mass.sum())

        radius = float(ball["radius"])
        self.radius = radius
        self.h = radius / 5.0
        center = np.asarray(ball["initial_center"], dtype=np.float64)
        youngs = float(ball["youngs_modulus"])
        poisson = float(ball["poisson_ratio"])
        density = float(ball["density"])
        self.mu = youngs / (2.0 * (1.0 + poisson))
        self.lam = (
            youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
            if poisson < 0.5
            else 2.0 * self.mu
        )
        self.gravity = np.asarray(
            cfg.get("initial_conditions", {}).get("gravity", [0.0, 0.0, -9.81]),
            dtype=np.float64,
        )

        # --- rigid impact plate (fixed box, ball approaches its -x face) ---
        plate = cfg["impact_plate"]
        pc = np.asarray(plate["center"], dtype=np.float64)
        pe = np.asarray(plate["half_extents"], dtype=np.float64)
        self.plate_x = float(pc[0] - pe[0])
        self.plate_y = (float(pc[1] - pe[1]), float(pc[1] + pe[1]))
        self.plate_z = (float(pc[2] - pe[2]), float(pc[2] + pe[2]))

        # --- hexahedral mesh over the complete lattice cells -------------
        index = np.rint((self.rest - center) / self.h).astype(np.int64)
        lookup = {(int(i), int(j), int(k)): n for n, (i, j, k) in enumerate(index)}
        lo = index.min(axis=0)
        hi = index.max(axis=0)
        conn = []
        for i in range(lo[0], hi[0]):
            for j in range(lo[1], hi[1]):
                for k in range(lo[2], hi[2]):
                    nodes = []
                    for di in (0, 1):
                        for dj in (0, 1):
                            for dk in (0, 1):
                                node = lookup.get((i + di, j + dj, k + dk))
                                if node is None:
                                    nodes = None
                                    break
                                nodes.append(node)
                            if nodes is None:
                                break
                        if nodes is None:
                            break
                    if nodes is not None:
                        conn.append(nodes)
        self.conn = np.asarray(conn, dtype=np.int64)
        self.ne = len(self.conn)

        # --- distance springs for lattice points outside any element ----
        used = np.zeros(self.n, dtype=bool)
        used[self.conn.reshape(-1)] = True
        orphans = np.flatnonzero(~used)
        spring_a: list[int] = []
        spring_b: list[int] = []
        for a in orphans:
            delta = index[used] - index[a]
            cheb = np.max(np.abs(delta), axis=1)
            for radius_cells in (1, 2):
                near = np.flatnonzero(cheb <= radius_cells)
                if len(near) >= 3:
                    break
            candidates = np.flatnonzero(used)[near]
            for b in candidates:
                spring_a.append(int(a))
                spring_b.append(int(b))
        self.spring_a = np.asarray(spring_a, dtype=np.int64)
        self.spring_b = np.asarray(spring_b, dtype=np.int64)
        if self.spring_a.size:
            self.spring_rest = np.linalg.norm(
                self.rest[self.spring_b] - self.rest[self.spring_a], axis=1
            )
            # continuum-matched tie stiffness: E*h total per orphan point
            counts = np.bincount(
                np.searchsorted(orphans, self.spring_a), minlength=len(orphans)
            )
            per_point = youngs * self.h
            self.spring_k = np.repeat(per_point / np.maximum(counts, 1), counts)
        else:
            self.spring_rest = np.zeros(0)
            self.spring_k = np.zeros(0)

        # --- precomputed element data -----------------------------------
        self.dn_g, self.dn_c, _ = _hex_shape_derivatives(self.h)
        self.w0 = (self.h / 2.0) ** 3  # reference Jacobian determinant

        # --- integrator state and budgets --------------------------------
        wave_speed = np.sqrt(youngs / density)
        self.dt = min(4.0e-6, 0.2 * self.h / max(wave_speed, 1e-9) * 5.0)
        self.time = 0.0
        self.contact_impulse = np.zeros(3)
        self.gravity_impulse = np.zeros(3)
        self.contact_force = np.zeros(3)
        self.reference_momentum = self.mass_total * wave_speed
        self.contact_normal = np.array([1.0, 0.0, 0.0])
        self._initialized = True

    # ------------------------------------------------------------------
    # internal elastic forces
    # ------------------------------------------------------------------
    def _internal_forces(self) -> np.ndarray:
        xe = self.x[self.conn]  # (E,8,3)
        # deformation gradient at every Gauss point
        F = np.einsum("eai,gaj->egij", xe, self.dn_g)  # (E,8,3,3)
        # F-bar: element-mean dilatation to suppress volumetric locking
        J = np.linalg.det(F)
        Jbar = J.mean(axis=1, keepdims=True)
        scale = np.cbrt(
            np.clip(Jbar, 1e-12, None) / np.clip(J, 1e-12, None)
        )
        Fb = F * scale[..., None, None]
        # corotated elasticity: P = 2 mu (F - R) + lam tr(R^T F - 3) I
        u, _s, vt = np.linalg.svd(Fb)
        R = u @ vt
        tr = np.einsum("egij,egij->eg", R, Fb) - 3.0
        P = -2.0 * self.mu * (R - Fb)
        P[..., 0, 0] += self.lam * tr
        P[..., 1, 1] += self.lam * tr
        P[..., 2, 2] += self.lam * tr
        # nodal internal forces f_a = -sum_gp w detJ0 P_gp dN_a/dX
        fe = -self.w0 * np.einsum("egij,gaj->eai", P, self.dn_g)
        force = np.zeros((self.n, 3))
        np.add.at(force, self.conn.reshape(-1), fe.reshape(-1, 3))

        # tie springs for the orphan lattice points
        if self.spring_a.size:
            d = self.x[self.spring_b] - self.x[self.spring_a]
            length = np.linalg.norm(d, axis=1)
            stretch = self.spring_k * (length - self.spring_rest)
            dir = d / np.maximum(length, 1e-12)[:, None]
            fs = stretch[:, None] * dir
            np.add.at(force, self.spring_a, fs)
            np.add.at(force, self.spring_b, -fs)
        return force

    # ------------------------------------------------------------------
    # one explicit step with rigid-plane contact
    # ------------------------------------------------------------------
    def _step(self, h: float) -> None:
        force = self._internal_forces()
        accel = force / self.mass_safe[:, None] + self.gravity[None, :]
        self.v += accel * h
        self.x += self.v * h

        penetrating = (
            (self.x[:, 0] < self.plate_x)
            & (self.x[:, 1] > self.plate_y[0])
            & (self.x[:, 1] < self.plate_y[1])
            & (self.x[:, 2] > self.plate_z[0])
            & (self.x[:, 2] < self.plate_z[1])
        )
        if np.any(penetrating):
            vn = self.v[penetrating, 0]
            closing = vn < 0.0
            impulse = -float(np.sum(self.mass[penetrating][closing] * vn[closing]))
            if impulse > 0.0:
                self.contact_impulse += impulse * self.contact_normal
            self.v[penetrating, 0] = np.maximum(vn, 0.0)
            self.x[penetrating, 0] = self.plate_x
            self.contact_force = impulse * self.contact_normal / h
        else:
            self.contact_force = np.zeros(3)
        self.gravity_impulse += self.mass_total * self.gravity * h
        self.time += h

    # ------------------------------------------------------------------
    # StateFrame assembly
    # ------------------------------------------------------------------
    def _frame(self, time: float):
        from worldsolver.contracts import StateFrame

        xe = self.x[self.conn]
        F_center = np.einsum("eai,aj->eij", xe, self.dn_c)

        ball_external_force = self.mass_total * self.gravity + self.contact_force
        plate_external_force = -self.contact_force
        # shape (G=2, K=1, 3): one material, total external force per domain
        budget_force = np.stack(
            (
                ball_external_force[None, :],
                plate_external_force[None, :],
            ),
            axis=0,
        )
        cumulative_impulse = np.stack(
            (
                self.gravity_impulse + self.contact_impulse,
                -self.contact_impulse,
            )
        )
        zeros_g3 = np.zeros((2, 3))
        domain_weights = np.stack((np.ones(self.n), np.zeros(self.n)))

        return StateFrame(
            time=float(time),
            position=self.x.copy(),
            velocity=self.v.copy(),
            mass=self.mass.copy(),
            attributes={
                "rest_position": self.rest.copy(),
                "mass_quantity": self.mass.copy(),
                "particle_radius": np.full(self.n, 0.5 * self.h),
            },
            fields={
                "budget_cumulative_external_impulse": cumulative_impulse,
                "budget_cumulative_inflow_momentum": zeros_g3.copy(),
                "budget_cumulative_outflow_momentum": zeros_g3,
                "budget_domain_weights": domain_weights,
                "budget_external_force": budget_force,
                "cumulative_mass_source": np.zeros(1),
                "element_deformation_gradient": F_center,
            },
            scalars={
                "reference_momentum": float(self.reference_momentum),
                "total_mass": float(self.mass_total),
                "radius": float(self.radius),
            },
        )

    # ------------------------------------------------------------------
    # contract entry points
    # ------------------------------------------------------------------
    def advance_to(self, time: float):
        if not self._initialized:
            raise RuntimeError("solver.initialize() must run before advance_to()")
        target = float(time)
        guard = 0
        while self.time < target - 1e-12:
            h = min(self.dt, target - self.time)
            self._step(h)
            guard += 1
            if guard > 10_000_000:
                raise RuntimeError("advance_to failed to reach the target time")
        self.time = target
        frame = self._frame(target)
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "method": "explicit corotated hex-FEM with F-bar and rigid-plane contact",
            "samples": int(self.n) if self._initialized else None,
            "elements": int(self.ne) if self._initialized else None,
            "time_step": float(self.dt) if self._initialized else None,
            "current_time": float(self.time),
            "contact_impulse_x": float(self.contact_impulse[0])
            if self._initialized
            else 0.0,
        }
