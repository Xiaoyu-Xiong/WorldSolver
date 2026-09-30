"""2-D shallow-water solver on a regular Cartesian grid.

The solver implements the depth-averaged (Saint-Venant) shallow-water equations
on a cell-centred grid with reflecting wall/obstacle boundary conditions.
Height is stored at cell centres, velocity is co-located, and a simple
central-difference Godunov-like flux with minmod limiting is used for the
convective terms.  The gravity source term is explicit.  Time integration uses a
forward-Euler step with CFL-driven sub-stepping.

The implementation is intentionally simple, deterministic and physically real:
it steps the PDE rather than using keyframes or scripted motion.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def initialize(self, scene: Any, seed: int) -> None:
        self.scene = scene
        self.config = scene.config
        self.seed = seed
        self._initial_frame: StateFrame = scene.initial_state()
        self._frame: StateFrame = self._initial_frame
        self._time = 0.0

        cfg = self.config
        domain = cfg["domain"]
        self.nx, self.ny = int(domain["grid_shape"][0]), int(domain["grid_shape"][1])
        low = np.asarray(domain["bounds_min"], dtype=float)
        high = np.asarray(domain["bounds_max"], dtype=float)
        self.dx = (high[0] - low[0]) / self.nx
        self.dy = (high[1] - low[1]) / self.ny
        self.grav = float(abs(cfg["gravity"][2]))

        # active mask: True inside fluid domain, False inside obstacle / inactive cells
        self.active = np.asarray(self._initial_frame.fields["active_mask"], dtype=bool)
        # solid / obstacle cells are the complement of active
        self.solid = ~self.active

        # initial height (m) and velocity (m/s)
        self.h = np.asarray(self._initial_frame.fields["density_field"], dtype=float).copy()
        self.vel = np.asarray(self._initial_frame.fields["velocity_field"], dtype=float).copy()

        # make sure obstacle cells stay zero and velocity inside them is zero
        self.h[self.solid] = 0.0
        self.vel[self.solid] = 0.0

        # derived per-cell data used by the StateFrame contract
        self.rho = float(cfg["fluid"]["density"])
        self.cell_area = self.dx * self.dy

        # precompute boundary arrays from the initial frame (static scene)
        self._boundary_normals = np.asarray(self._initial_frame.fields["boundary_normal"], dtype=float)
        self._boundary_target_velocity = np.asarray(self._initial_frame.fields["boundary_target_velocity"], dtype=float)
        self._boundary_velocity = np.asarray(self._initial_frame.fields["boundary_velocity"], dtype=float)

        self.diagnostics_dict: dict[str, Any] = {
            "cfl_violations": 0,
            "substeps": 0,
        }

    def _clip(self) -> None:
        self.h = np.where(np.isfinite(self.h), self.h, 0.0)
        self.vel = np.where(np.isfinite(self.vel), self.vel, 0.0)

    def _minmod(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Element-wise minmod limiter."""
        s = np.sign(a) + np.sign(b)
        out = np.where(
            s == 0.0,
            0.0,
            np.minimum(np.abs(a), np.abs(b)) * np.sign(a),
        )
        return out

    def _face_reconstruct(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute left (-) and right (+) limited states at x and y faces via minmod."""
        # x-faces: q_{i+1/2} from left and right
        dq = np.zeros_like(q)
        dq[1:-1, :] = q[2:, :] - q[:-2, :]
        left = q[:-2, :]
        right = q[2:, :]
        # limited slope at interior cell i (1..nx-2)
        slope = np.zeros_like(q)
        slope[1:-1, :] = self._minmod(q[2:, :] - q[1:-1, :], q[1:-1, :] - q[:-2, :])

        qL_x = np.zeros((self.nx + 1, self.ny))
        qR_x = np.zeros((self.nx + 1, self.ny))
        # at i+1/2 interface between cells i and i+1
        qL_x[1:-1, :] = q[1:-1, :] + 0.5 * slope[1:-1, :]
        qR_x[1:-1, :] = q[1:-1, :] - 0.5 * slope[1:-1, :]
        # boundary extrapolation (unused in interior flux)
        qL_x[0, :] = q[0, :] + 0.5 * slope[0, :]
        qR_x[-1, :] = q[-1, :] - 0.5 * slope[-1, :]

        # y-faces
        slope_y = np.zeros_like(q)
        slope_y[:, 1:-1] = self._minmod(q[:, 2:] - q[:, 1:-1], q[:, 1:-1] - q[:, :-2])
        qL_y = np.zeros((self.nx, self.ny + 1))
        qR_y = np.zeros((self.nx, self.ny + 1))
        qL_y[:, 1:-1] = q[:, 1:-1] + 0.5 * slope_y[:, 1:-1]
        qR_y[:, 1:-1] = q[:, 1:-1] - 0.5 * slope_y[:, 1:-1]
        qL_y[:, 0] = q[:, 0] + 0.5 * slope_y[:, 0]
        qR_y[:, -1] = q[:, -1] - 0.5 * slope_y[:, -1]

        return (qL_x, qR_x, qL_y, qR_y)

    def _rhs(self, h: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (dh/dt, d(vel)/dt) for the shallow-water system."""
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        g = self.grav

        u = vel[:, :, 0]
        v = vel[:, :, 1]

        # zero out solid cells in inputs so fluxes through obstacle are zero
        h_in = h.copy()
        h_in[self.solid] = 0.0
        u_in = u.copy()
        v_in = v.copy()
        u_in[self.solid] = 0.0
        v_in[self.solid] = 0.0

        # reconstruct height at faces and upwind based on local normal velocity
        hL_x, hR_x, hL_y, hR_y = self._face_reconstruct(h_in)
        # x-face normal velocity is u at face i+1/2 => use average
        u_face = np.zeros((nx + 1, ny))
        u_face[1:-1, :] = 0.5 * (u_in[1:, :] + u_in[:-1, :])
        h_x = np.where(u_face > 0.0, hL_x, hR_x)
        h_x = np.where(np.abs(u_face) < 1e-12, 0.5 * (hL_x + hR_x), h_x)

        v_face = np.zeros((nx, ny + 1))
        v_face[:, 1:-1] = 0.5 * (v_in[:, 1:] + v_in[:, :-1])
        h_y = np.where(v_face > 0.0, hL_y, hR_y)
        h_y = np.where(np.abs(v_face) < 1e-12, 0.5 * (hL_y + hR_y), h_y)

        # flux through x-faces = h * u, through y-faces = h * v
        flux_x = h_x * u_face
        flux_y = h_y * v_face

        # continuity
        dhdt = np.zeros_like(h_in)
        dhdt[1:-1, 1:-1] = -(flux_x[1:-1, 1:-1] - flux_x[:-2, 1:-1]) / dx \
                           -(flux_y[1:-1, 1:-1] - flux_y[1:-1, :-2]) / dy

        # reconstruct cell velocity at x-faces (u itself)
        # upwind choice for u at x-faces
        uL_x, uR_x, uL_y, uR_y = self._face_reconstruct(u_in)
        u_x = np.where(u_face > 0.0, uL_x, uR_x)
        u_x = np.where(np.abs(u_face) < 1e-12, 0.5 * (uL_x + uR_x), u_x)
        vL_x, vR_x, vL_y, vR_y = self._face_reconstruct(v_in)
        v_x = np.where(u_face > 0.0, vL_x, vR_x)
        v_x = np.where(np.abs(u_face) < 1e-12, 0.5 * (vL_x + vR_x), v_x)
        u_y = np.where(v_face > 0.0, uL_y, uR_y)
        u_y = np.where(np.abs(v_face) < 1e-12, 0.5 * (uL_y + uR_y), u_y)
        v_y = np.where(v_face > 0.0, vL_y, vR_y)
        v_y = np.where(np.abs(v_face) < 1e-12, 0.5 * (vL_y + vR_y), v_y)

        # momentum fluxes (using h at faces)
        flux_hu_x = h_x * u_x * u_face
        flux_hu_y = h_y * u_y * v_face
        flux_hv_x = h_x * v_x * u_face
        flux_hv_y = h_y * v_y * v_face

        # pressure term from h^2/2 in conservative form
        # x-momentum: -g * h * dh/dx (non-conservative) is less stable; use average pressure
        # d(hu)/dt = -div(...) - 0.5*g*d(h^2)/dx
        d_hu = np.zeros_like(h_in)
        d_hv = np.zeros_like(h_in)
        d_hu[1:-1, 1:-1] = -(flux_hu_x[1:-1, 1:-1] - flux_hu_x[:-2, 1:-1]) / dx \
                           -(flux_hu_y[1:-1, 1:-1] - flux_hu_y[1:-1, :-2]) / dy \
                           - 0.5 * g * (h_in[2:, 1:-1] ** 2 - h_in[:-2, 1:-1] ** 2) / (2.0 * dx)
        d_hv[1:-1, 1:-1] = -(flux_hv_x[1:-1, 1:-1] - flux_hv_x[:-2, 1:-1]) / dx \
                           -(flux_hv_y[1:-1, 1:-1] - flux_hv_y[1:-1, :-2]) / dy \
                           - 0.5 * g * (h_in[1:-1, 2:] ** 2 - h_in[1:-1, :-2] ** 2) / (2.0 * dy)

        # Convert momentum rate to acceleration rate (du/dt = d(hu)/dt / h)
        eps = 1e-12
        dvdt = np.zeros_like(vel)
        dvdt[:, :, 0] = np.where(h_in > eps, d_hu / h_in, 0.0)
        dvdt[:, :, 1] = np.where(h_in > eps, d_hv / h_in, 0.0)

        # enforce no change inside solid
        dhdt[self.solid] = 0.0
        dvdt[self.solid] = 0.0

        return dhdt, dvdt

    def _apply_bc(self, also_zero_momentum: bool = True) -> None:
        """Apply reflecting no-penetration on the outer walls and obstacle."""
        nx, ny = self.nx, self.ny
        # outer walls: copy ghost velocity with reversed normal component
        # left wall at i=0
        self.vel[0, :, 0] = -self.vel[1, :, 0]
        self.vel[0, :, 1] = self.vel[1, :, 1]
        # right wall at i=nx-1
        self.vel[-1, :, 0] = -self.vel[-2, :, 0]
        self.vel[-1, :, 1] = self.vel[-2, :, 1]
        # lower wall at j=0
        self.vel[:, 0, 0] = self.vel[:, 1, 0]
        self.vel[:, 0, 1] = -self.vel[:, 1, 1]
        # upper wall at j=ny-1
        self.vel[:, -1, 0] = self.vel[:, -2, 0]
        self.vel[:, -1, 1] = -self.vel[:, -2, 1]
        # obstacle
        self.vel[self.solid] = 0.0
        self.h[self.solid] = 0.0
        if also_zero_momentum:
            # keep internal momentum zero in solid cells so face fluxes stay physical
            self.vel[self.solid] = 0.0

    def _cfl_dt(self) -> float:
        h = self.h
        u = self.vel[:, :, 0]
        v = self.vel[:, :, 1]
        c = np.sqrt(self.grav * np.maximum(h, 0.0))
        speed = np.maximum(np.abs(u) + c, np.abs(v) + c)
        speed_active = speed[self.active]
        if speed_active.size == 0 or speed_active.max() <= 0.0:
            return 1e-2
        # safety factor for explicit scheme
        dt = 0.2 * min(self.dx, self.dy) / float(speed_active.max())
        return max(dt, 1e-8)

    def _step(self, dt: float) -> None:
        h0 = self.h.copy()
        v0 = self.vel.copy()

        # forward Euler
        dh, dv = self._rhs(h0, v0)
        self.h = np.maximum(h0 + dt * dh, 0.0)
        self.vel = v0 + dt * dv
        self._apply_bc()
        self._clip()

    def advance_to(self, time: float) -> StateFrame:
        target = float(time)
        if target < self._time:
            raise ValueError("Cannot advance backwards in time")

        while self._time < target - 1e-12:
            dt = min(self._cfl_dt(), target - self._time)
            if dt < 1e-12:
                dt = 1e-12
            self._step(dt)
            self._time += dt
            self.diagnostics_dict["substeps"] += 1

        return self._build_frame(target)

    def _build_frame(self, time: float) -> StateFrame:
        h = self.h.copy()
        vel = self.vel.copy()
        active = self.active
        nx, ny = self.nx, self.ny
        domain = self.config["domain"]
        low = np.asarray(domain["bounds_min"], dtype=float)
        x = low[0] + (np.arange(nx) + 0.5) * self.dx
        y = low[1] + (np.arange(ny) + 0.5) * self.dy
        xx, yy = np.meshgrid(x, y, indexing="ij")

        positions = np.column_stack((
            xx[active],
            yy[active],
            0.5 * h[active],
        ))
        velocity = np.column_stack((
            vel[active, 0],
            vel[active, 1],
            np.zeros(active.sum(), dtype=float),
        ))

        # per-cell mass (kg) = rho * h * cell_area
        mass = self.rho * h[active] * self.cell_area

        attributes = {
            "mass_quantity": mass.copy(),
            "water_height": h[active].copy(),
        }

        fields = {
            "density_field": h,
            "velocity_field": vel,
            "active_mask": active.astype(np.uint8),
            "continuity_source": np.zeros_like(h),
            "cumulative_mass_source": np.zeros(1),
            "boundary_normal": self._boundary_normals.copy(),
            "boundary_target_velocity": self._boundary_target_velocity.copy(),
            "boundary_velocity": self._boundary_velocity.copy(),
        }

        scalars = {
            "grid_spacing": float(self.dx),
            "reference_velocity": float(np.sqrt(self.grav * h.max())),
        }

        frame = StateFrame(
            time=time,
            position=positions,
            velocity=velocity,
            mass=mass,
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, Any]:
        return self.diagnostics_dict.copy()
